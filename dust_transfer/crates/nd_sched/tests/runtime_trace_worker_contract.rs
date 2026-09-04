//! AST contract: numerical workers perform no direct I/O, blocking transport,
//! or runtime tracing. `cfg(test)` syntax is removed semantically, not by
//! guessing Rust item boundaries from braces.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{
    Arm, Attribute, Expr, Field, FieldPat, FieldValue, FnArg, ForeignItem, GenericParam, ImplItem,
    Item, Local, Macro, Meta, Pat, PatType, Path as SynPath, StmtMacro, Token, TraitItem, Type,
    UseTree, Variant,
};

/// One function inside a governed module that may raise findings the module
/// otherwise forbids.
///
/// The rules below are whole-module, so they cannot say "not in the worker
/// loop". Where a module legitimately contains bootstrap or teardown that the
/// steady-state path never reaches, the boundary is drawn here by name. Both
/// halves fail closed: a named function that no longer exists is an error, and
/// so is a function that raises anything other than exactly what it waives.
struct FnExemption {
    /// Definition-site name: `free_function`, or `Type::method` for an
    /// inherent or trait `impl` item. Matched against the definition, never a
    /// call site.
    name: &'static str,
    /// EXACTLY the findings this function may raise. Compared for equality, so
    /// a new construct inside the function fails, and a waiver that is no
    /// longer earned fails too rather than quietly widening the hole.
    waived: &'static [&'static str],
}

/// The rendezvous file is the only shared-filesystem object in the sharded
/// transport, and all three sites below touch it outside the steady-state
/// worker path: rank 0 publishes it once before any work is dispatched, a
/// worker reads it once at startup, and rank 0 unlinks it at teardown. Every
/// other function in the module — including `ShardWorkerSession::serve` and
/// `ShardedEventBatchEvaluator::run`, the two loops this contract is actually
/// about — stays covered.
const SHARDED_HYBRID_RENDEZVOUS: &[FnExemption] = &[
    FnExemption {
        // Teardown, after the transport is finished. A stale rendezvous file
        // names a dead port and a later worker reading it would hang, so the
        // unlink has to happen somewhere.
        name: "TcpShardTransport::drop",
        waived: &["std::fs"],
    },
    FnExemption {
        // Bootstrap: called once from `bind_and_accept`, before a single
        // worker has connected.
        //
        // The `print macro` waiver is the deliberate best-effort warning on
        // the directory `sync_all` that follows the rename. Past that rename
        // the endpoint is already live, so returning `Err` would abort rank 0
        // with the file still visible and burn every worker's full 900 s
        // connect timeout. None of the three costs the worker print rule
        // exists to prevent applies here: it is one line, on rank 0 only,
        // before any evaluation — not N ranks interleaving output, not a stdio
        // lock inside a hot loop, and not a substitute for the trace stream.
        name: "publish_rendezvous",
        waived: &["print macro", "std::fs"],
    },
    FnExemption {
        // Worker startup. `connect` polls the rendezvous path until rank 0
        // publishes it, then never touches the filesystem again; `serve`, the
        // loop it hands the socket to, is not exempt.
        name: "ShardWorkerSession::connect",
        waived: &["std::fs"],
    },
];

struct ModulePolicy {
    path: &'static str,
    hot_leaf: bool,
    io_exempt: &'static [FnExemption],
}

const MODULES: &[ModulePolicy] = &[
    ModulePolicy {
        path: "crates/lightyear_odeint_rs/src/rhs.rs",
        hot_leaf: true,
        io_exempt: &[],
    },
    ModulePolicy {
        path: "crates/lightyear_odeint_rs/src/integrator.rs",
        hot_leaf: true,
        io_exempt: &[],
    },
    ModulePolicy {
        path: "crates/two_phase_transfer_rs/src/evaluate.rs",
        hot_leaf: true,
        io_exempt: &[],
    },
    ModulePolicy {
        path: "crates/two_phase_transfer_rs/src/solve.rs",
        hot_leaf: true,
        io_exempt: &[],
    },
    ModulePolicy {
        path: "crates/two_phase_transfer_rs/src/batch_eci.rs",
        hot_leaf: true,
        io_exempt: &[],
    },
    ModulePolicy {
        path: "crates/dust_estimates_rs/src/mass_solver.rs",
        hot_leaf: true,
        io_exempt: &[],
    },
    ModulePolicy {
        path: "crates/nd_pipeline/src/native_mf.rs",
        hot_leaf: true,
        io_exempt: &[],
    },
    ModulePolicy {
        path: "crates/nd_pipeline/src/native_hybrid.rs",
        hot_leaf: true,
        io_exempt: &[],
    },
    ModulePolicy {
        path: "crates/nd_pipeline/src/native_hybrid/qualification.rs",
        hot_leaf: true,
        io_exempt: &[],
    },
    ModulePolicy {
        path: "crates/nd_pipeline/src/physics/orchestrate.rs",
        hot_leaf: true,
        io_exempt: &[],
    },
    ModulePolicy {
        path: "crates/nd_pipeline/src/population.rs",
        hot_leaf: false,
        io_exempt: &[],
    },
    ModulePolicy {
        path: "crates/nd_pipeline/src/sharded_hybrid.rs",
        hot_leaf: false,
        io_exempt: SHARDED_HYBRID_RENDEZVOUS,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Truth {
    False,
    Unknown,
    True,
}

impl Truth {
    const fn not(self) -> Self {
        match self {
            Self::False => Self::True,
            Self::Unknown => Self::Unknown,
            Self::True => Self::False,
        }
    }
}

fn parse_meta_list(list: &syn::MetaList) -> Option<Vec<Meta>> {
    Punctuated::<Meta, Token![,]>::parse_terminated
        .parse2(list.tokens.clone())
        .ok()
        .map(Punctuated::into_iter)
        .map(Iterator::collect)
}

/// Evaluate only facts fixed by a non-test build. Other cfg predicates stay
/// unknown, so feature-gated production is scanned conservatively.
fn production_truth(meta: &Meta) -> Truth {
    match meta {
        Meta::Path(path) if path.is_ident("test") => Truth::False,
        Meta::List(list) if list.path.is_ident("not") => {
            let Some(mut nested) = parse_meta_list(list) else {
                return Truth::Unknown;
            };
            if nested.len() != 1 {
                return Truth::Unknown;
            }
            production_truth(&nested.remove(0)).not()
        }
        Meta::List(list) if list.path.is_ident("all") => {
            let Some(nested) = parse_meta_list(list) else {
                return Truth::Unknown;
            };
            if nested
                .iter()
                .any(|predicate| production_truth(predicate) == Truth::False)
            {
                Truth::False
            } else if nested
                .iter()
                .all(|predicate| production_truth(predicate) == Truth::True)
            {
                Truth::True
            } else {
                Truth::Unknown
            }
        }
        Meta::List(list) if list.path.is_ident("any") => {
            let Some(nested) = parse_meta_list(list) else {
                return Truth::Unknown;
            };
            if nested
                .iter()
                .any(|predicate| production_truth(predicate) == Truth::True)
            {
                Truth::True
            } else if nested
                .iter()
                .all(|predicate| production_truth(predicate) == Truth::False)
            {
                Truth::False
            } else {
                Truth::Unknown
            }
        }
        Meta::Path(_) | Meta::NameValue(_) | Meta::List(_) => Truth::Unknown,
    }
}

fn meta_removes_from_production(meta: &Meta) -> bool {
    let Meta::List(list) = meta else {
        return false;
    };
    if list.path.is_ident("cfg") {
        let Some(mut predicates) = parse_meta_list(list) else {
            return false;
        };
        return predicates.len() == 1 && production_truth(&predicates.remove(0)) == Truth::False;
    }
    if !list.path.is_ident("cfg_attr") {
        return false;
    }

    let Some(mut arguments) = parse_meta_list(list) else {
        return false;
    };
    if arguments.len() < 2 || production_truth(&arguments.remove(0)) != Truth::True {
        return false;
    }
    arguments.iter().any(meta_removes_from_production)
}

fn is_test_only(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .any(|attribute| meta_removes_from_production(&attribute.meta))
}

fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(syn::ItemConst { attrs, .. })
        | Item::Enum(syn::ItemEnum { attrs, .. })
        | Item::ExternCrate(syn::ItemExternCrate { attrs, .. })
        | Item::Fn(syn::ItemFn { attrs, .. })
        | Item::ForeignMod(syn::ItemForeignMod { attrs, .. })
        | Item::Impl(syn::ItemImpl { attrs, .. })
        | Item::Macro(syn::ItemMacro { attrs, .. })
        | Item::Mod(syn::ItemMod { attrs, .. })
        | Item::Static(syn::ItemStatic { attrs, .. })
        | Item::Struct(syn::ItemStruct { attrs, .. })
        | Item::Trait(syn::ItemTrait { attrs, .. })
        | Item::TraitAlias(syn::ItemTraitAlias { attrs, .. })
        | Item::Type(syn::ItemType { attrs, .. })
        | Item::Union(syn::ItemUnion { attrs, .. })
        | Item::Use(syn::ItemUse { attrs, .. }) => attrs,
        _ => &[],
    }
}

fn impl_item_attrs(item: &ImplItem) -> &[Attribute] {
    match item {
        ImplItem::Const(syn::ImplItemConst { attrs, .. })
        | ImplItem::Fn(syn::ImplItemFn { attrs, .. })
        | ImplItem::Macro(syn::ImplItemMacro { attrs, .. })
        | ImplItem::Type(syn::ImplItemType { attrs, .. }) => attrs,
        _ => &[],
    }
}

fn trait_item_attrs(item: &TraitItem) -> &[Attribute] {
    match item {
        TraitItem::Const(syn::TraitItemConst { attrs, .. })
        | TraitItem::Fn(syn::TraitItemFn { attrs, .. })
        | TraitItem::Macro(syn::TraitItemMacro { attrs, .. })
        | TraitItem::Type(syn::TraitItemType { attrs, .. }) => attrs,
        _ => &[],
    }
}

fn foreign_item_attrs(item: &ForeignItem) -> &[Attribute] {
    match item {
        ForeignItem::Fn(syn::ForeignItemFn { attrs, .. })
        | ForeignItem::Macro(syn::ForeignItemMacro { attrs, .. })
        | ForeignItem::Static(syn::ForeignItemStatic { attrs, .. })
        | ForeignItem::Type(syn::ForeignItemType { attrs, .. }) => attrs,
        _ => &[],
    }
}

fn expr_attrs(expr: &Expr) -> &[Attribute] {
    match expr {
        Expr::Array(syn::ExprArray { attrs, .. })
        | Expr::Assign(syn::ExprAssign { attrs, .. })
        | Expr::Async(syn::ExprAsync { attrs, .. })
        | Expr::Await(syn::ExprAwait { attrs, .. })
        | Expr::Binary(syn::ExprBinary { attrs, .. })
        | Expr::Block(syn::ExprBlock { attrs, .. })
        | Expr::Break(syn::ExprBreak { attrs, .. })
        | Expr::Call(syn::ExprCall { attrs, .. })
        | Expr::Cast(syn::ExprCast { attrs, .. })
        | Expr::Closure(syn::ExprClosure { attrs, .. })
        | Expr::Const(syn::ExprConst { attrs, .. })
        | Expr::Continue(syn::ExprContinue { attrs, .. })
        | Expr::Field(syn::ExprField { attrs, .. })
        | Expr::ForLoop(syn::ExprForLoop { attrs, .. })
        | Expr::Group(syn::ExprGroup { attrs, .. })
        | Expr::If(syn::ExprIf { attrs, .. })
        | Expr::Index(syn::ExprIndex { attrs, .. })
        | Expr::Infer(syn::ExprInfer { attrs, .. })
        | Expr::Let(syn::ExprLet { attrs, .. })
        | Expr::Lit(syn::ExprLit { attrs, .. })
        | Expr::Loop(syn::ExprLoop { attrs, .. })
        | Expr::Macro(syn::ExprMacro { attrs, .. })
        | Expr::Match(syn::ExprMatch { attrs, .. })
        | Expr::MethodCall(syn::ExprMethodCall { attrs, .. })
        | Expr::Paren(syn::ExprParen { attrs, .. })
        | Expr::Path(syn::ExprPath { attrs, .. })
        | Expr::Range(syn::ExprRange { attrs, .. })
        | Expr::RawAddr(syn::ExprRawAddr { attrs, .. })
        | Expr::Reference(syn::ExprReference { attrs, .. })
        | Expr::Repeat(syn::ExprRepeat { attrs, .. })
        | Expr::Return(syn::ExprReturn { attrs, .. })
        | Expr::Struct(syn::ExprStruct { attrs, .. })
        | Expr::Try(syn::ExprTry { attrs, .. })
        | Expr::TryBlock(syn::ExprTryBlock { attrs, .. })
        | Expr::Tuple(syn::ExprTuple { attrs, .. })
        | Expr::Unary(syn::ExprUnary { attrs, .. })
        | Expr::Unsafe(syn::ExprUnsafe { attrs, .. })
        | Expr::While(syn::ExprWhile { attrs, .. })
        | Expr::Yield(syn::ExprYield { attrs, .. }) => attrs,
        _ => &[],
    }
}

fn generic_param_attrs(param: &GenericParam) -> &[Attribute] {
    match param {
        GenericParam::Lifetime(syn::LifetimeParam { attrs, .. })
        | GenericParam::Type(syn::TypeParam { attrs, .. })
        | GenericParam::Const(syn::ConstParam { attrs, .. }) => attrs,
    }
}

#[derive(Clone, Copy)]
struct Rules {
    worker_io: bool,
    hot_trace: bool,
}

struct ContractVisitor {
    rules: Rules,
    exemptions: &'static [FnExemption],
    findings: BTreeSet<&'static str>,
    /// Findings raised inside an exempt function, keyed by its index in
    /// `exemptions`.
    exempt_findings: BTreeMap<usize, BTreeSet<&'static str>>,
    /// Exempt entries whose definition the walk actually reached. An entry
    /// missing from this set names a function that no longer exists, or one
    /// that has been moved behind `cfg(test)`.
    seen_exemptions: BTreeSet<usize>,
    /// Enclosing `impl` self-type names, for `Type::method` matching.
    impl_types: Vec<String>,
    /// Enclosing exempt functions; findings route to the innermost.
    exempt_stack: Vec<usize>,
}

impl ContractVisitor {
    const fn new(rules: Rules, exemptions: &'static [FnExemption]) -> Self {
        Self {
            rules,
            exemptions,
            findings: BTreeSet::new(),
            exempt_findings: BTreeMap::new(),
            seen_exemptions: BTreeSet::new(),
            impl_types: Vec::new(),
            exempt_stack: Vec::new(),
        }
    }

    /// Route one finding to the innermost exempt function, or to the module's
    /// violation set when the walk is not inside one.
    fn record(&mut self, finding: &'static str) {
        if let Some(&index) = self.exempt_stack.last() {
            self.exempt_findings
                .entry(index)
                .or_default()
                .insert(finding);
        } else {
            self.findings.insert(finding);
        }
    }

    /// Enter `key` if it names an exempt function. The caller pops on the way
    /// out iff this returned `true`.
    fn enter_fn(&mut self, key: &str) -> bool {
        let Some(index) = self
            .exemptions
            .iter()
            .position(|exemption| exemption.name == key)
        else {
            return false;
        };
        self.seen_exemptions.insert(index);
        self.exempt_stack.push(index);
        true
    }

    fn record_segments(&mut self, segments: &[String]) {
        if self.rules.worker_io {
            if matches!(
                segments,
                [root, name]
                    if root == "std"
                        && matches!(
                            name.as_str(),
                            "print" | "println" | "eprint" | "eprintln"
                        )
            ) {
                self.record("print macro");
            }
            // Two segments, not three: the rule used to name `std::fs::File`
            // specifically, which meant `read_to_string`, `remove_file`,
            // `rename`, `create_dir_all` and `OpenOptions` were all invisible
            // to it. Every one of those blocks a worker exactly as hard as
            // opening a `File` does.
            if segments.windows(2).any(|parts| parts == ["std", "fs"]) {
                self.record("std::fs");
            }
            if segments.contains(&"BufWriter".to_owned()) {
                self.record("BufWriter");
            }
            if segments.contains(&"sync_channel".to_owned()) {
                self.record("sync_channel");
            }
            if segments
                .windows(2)
                .any(|parts| parts == ["mpsc", "channel"])
            {
                self.record("mpsc::channel");
            }
            if segments
                .iter()
                .any(|segment| segment == "log" || segment == "tracing")
            {
                self.record("log/tracing path");
            }
        }
        if self.rules.hot_trace {
            if segments.contains(&"nd_runtime_trace".to_owned()) {
                self.record("nd_runtime_trace");
            }
            if segments.contains(&"TraceEvent".to_owned()) {
                self.record("TraceEvent");
            }
        }
    }

    fn record_path(&mut self, path: &SynPath) {
        let segments: Vec<String> = path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        self.record_segments(&segments);
    }

    fn inspect_use_tree(&mut self, prefix: &mut Vec<String>, tree: &UseTree) {
        match tree {
            UseTree::Path(path) => {
                prefix.push(path.ident.to_string());
                self.inspect_use_tree(prefix, &path.tree);
                prefix.pop();
            }
            UseTree::Name(name) => {
                prefix.push(name.ident.to_string());
                self.record_segments(prefix);
                prefix.pop();
            }
            UseTree::Rename(rename) => {
                prefix.push(rename.ident.to_string());
                self.record_segments(prefix);
                prefix.pop();
            }
            UseTree::Glob(_) => self.record_segments(prefix),
            UseTree::Group(group) => {
                for tree in &group.items {
                    self.inspect_use_tree(prefix, tree);
                }
            }
        }
    }

    fn inspect_macro_tokens(&mut self, mac: &Macro) {
        if let Ok(file) = syn::parse2::<syn::File>(mac.tokens.clone()) {
            self.visit_file(&file);
            return;
        }
        if let Ok(expr) = syn::parse2::<Expr>(mac.tokens.clone()) {
            self.visit_expr(&expr);
            return;
        }
        let parser = Punctuated::<Expr, Token![,]>::parse_terminated;
        if let Ok(expressions) = parser.parse2(mac.tokens.clone()) {
            for expr in expressions {
                self.visit_expr(&expr);
            }
            return;
        }

        // Macro grammars may be neither Rust files nor expression lists. Walk
        // token trees recursively: delimiters are groups, so flattening text
        // misses patterns split across group edges. Literals are intentionally
        // absent; diagnostic strings are not executable code.
        let mut tokens = Vec::new();
        Self::collect_code_tokens(mac, &mut tokens);
        if self.rules.worker_io {
            if ["print", "println", "eprint", "eprintln"]
                .iter()
                .any(|name| {
                    tokens.windows(2).any(|window| window == [*name, "!"])
                        || Self::contains_path(&tokens, &["std", "::"], name)
                })
            {
                self.record("print macro");
            }
            if Self::contains_path(&tokens, &["std", "::"], "fs") {
                self.record("std::fs");
            }
            if tokens.iter().any(|token| token == "BufWriter") {
                self.record("BufWriter");
            }
            if tokens.iter().any(|token| token == "sync_channel") {
                self.record("sync_channel");
            }
            if Self::contains_path(&tokens, &["mpsc", "::"], "channel") {
                self.record("mpsc::channel");
            }
            if tokens
                .windows(2)
                .any(|window| matches!(window, [name, separator] if matches!(name.as_str(), "log" | "tracing") && separator == "::"))
            {
                self.record("log/tracing path");
            }
        }
        if self.rules.hot_trace {
            if tokens.iter().any(|token| token == "nd_runtime_trace") {
                self.record("nd_runtime_trace");
            }
            if tokens.iter().any(|token| token == "TraceEvent") {
                self.record("TraceEvent");
            }
            if tokens.windows(3).any(|window| window == [".", "emit", "("]) {
                self.record(".emit(");
            }
        }
    }

    fn contains_path(tokens: &[String], prefix: &[&str], leaf: &str) -> bool {
        let expected: Vec<&str> = prefix
            .iter()
            .copied()
            .chain(std::iter::once(leaf))
            .collect();
        tokens.iter().enumerate().any(|(start, _)| {
            tokens
                .get(start..)
                .is_some_and(|candidate| Self::matches_path(candidate, &expected))
        })
    }

    fn matches_path(tokens: &[String], expected: &[&str]) -> bool {
        let Some((wanted, remaining_expected)) = expected.split_first() else {
            return true;
        };
        let Some((actual, remaining_tokens)) = tokens.split_first() else {
            return false;
        };
        if actual == "(" {
            return Self::group_branches(tokens).is_some_and(|branches| {
                branches
                    .into_iter()
                    .any(|branch| Self::matches_path(branch, expected))
            });
        }
        actual == wanted && Self::matches_path(remaining_tokens, remaining_expected)
    }

    fn group_branches(tokens: &[String]) -> Option<Vec<&[String]>> {
        if tokens.first()? != "(" {
            return None;
        }
        let mut branches = Vec::new();
        let mut branch_start = 1_usize;
        let mut nested_depth = 0_usize;
        for (index, token) in tokens.iter().enumerate().skip(1) {
            match token.as_str() {
                "(" => nested_depth = nested_depth.checked_add(1)?,
                ")" if nested_depth == 0 => {
                    branches.push(tokens.get(branch_start..index)?);
                    return Some(branches);
                }
                ")" => nested_depth = nested_depth.checked_sub(1)?,
                "," if nested_depth == 0 => {
                    branches.push(tokens.get(branch_start..index)?);
                    branch_start = index.checked_add(1)?;
                }
                _ => {}
            }
        }
        None
    }

    fn collect_code_tokens(mac: &Macro, out: &mut Vec<String>) {
        let buffer = syn::buffer::TokenBuffer::new2(mac.tokens.clone());
        let mut raw = Vec::new();
        Self::collect_cursor_tokens(buffer.begin(), &mut raw);
        let mut tokens = raw.into_iter().peekable();
        while let Some(token) = tokens.next() {
            if token == ":" && tokens.peek().is_some_and(|next| next == ":") {
                tokens.next();
                out.push("::".to_owned());
            } else {
                out.push(token);
            }
        }
    }

    fn collect_cursor_tokens(mut cursor: syn::buffer::Cursor<'_>, out: &mut Vec<String>) {
        while !cursor.eof() {
            if let Some((inside, _, _, rest)) = cursor.any_group() {
                out.push("(".to_owned());
                Self::collect_cursor_tokens(inside, out);
                out.push(")".to_owned());
                cursor = rest;
            } else if let Some((ident, rest)) = cursor.ident() {
                out.push(ident.to_string());
                cursor = rest;
            } else if let Some((punct, rest)) = cursor.punct() {
                out.push(punct.as_char().to_string());
                cursor = rest;
            } else if let Some((_, rest)) = cursor.literal() {
                cursor = rest;
            } else {
                break;
            }
        }
    }
}

impl<'ast> Visit<'ast> for ContractVisitor {
    fn visit_file(&mut self, file: &'ast syn::File) {
        if !is_test_only(&file.attrs) {
            visit::visit_file(self, file);
        }
    }

    fn visit_item(&mut self, item: &'ast Item) {
        if matches!(item, Item::Verbatim(_)) {
            self.findings.insert("unparsed item syntax");
        } else if !is_test_only(item_attrs(item)) {
            visit::visit_item(self, item);
        }
    }

    fn visit_impl_item(&mut self, item: &'ast ImplItem) {
        if matches!(item, ImplItem::Verbatim(_)) {
            self.findings.insert("unparsed impl-item syntax");
        } else if !is_test_only(impl_item_attrs(item)) {
            visit::visit_impl_item(self, item);
        }
    }

    fn visit_trait_item(&mut self, item: &'ast TraitItem) {
        if matches!(item, TraitItem::Verbatim(_)) {
            self.findings.insert("unparsed trait-item syntax");
        } else if !is_test_only(trait_item_attrs(item)) {
            visit::visit_trait_item(self, item);
        }
    }

    fn visit_foreign_item(&mut self, item: &'ast ForeignItem) {
        if matches!(item, ForeignItem::Verbatim(_)) {
            self.findings.insert("unparsed foreign-item syntax");
        } else if !is_test_only(foreign_item_attrs(item)) {
            visit::visit_foreign_item(self, item);
        }
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        if matches!(expr, Expr::Verbatim(_)) {
            self.findings.insert("unparsed expression syntax");
        } else if !is_test_only(expr_attrs(expr)) {
            visit::visit_expr(self, expr);
        }
    }

    fn visit_pat(&mut self, pattern: &'ast Pat) {
        if matches!(pattern, Pat::Verbatim(_)) {
            self.findings.insert("unparsed pattern syntax");
        } else {
            visit::visit_pat(self, pattern);
        }
    }

    fn visit_type(&mut self, ty: &'ast Type) {
        if matches!(ty, Type::Verbatim(_)) {
            self.findings.insert("unparsed type syntax");
        } else {
            visit::visit_type(self, ty);
        }
    }

    fn visit_field(&mut self, field: &'ast Field) {
        if !is_test_only(&field.attrs) {
            visit::visit_field(self, field);
        }
    }

    fn visit_field_value(&mut self, field: &'ast FieldValue) {
        if !is_test_only(&field.attrs) {
            visit::visit_field_value(self, field);
        }
    }

    fn visit_field_pat(&mut self, field: &'ast FieldPat) {
        if !is_test_only(&field.attrs) {
            visit::visit_field_pat(self, field);
        }
    }

    fn visit_variant(&mut self, variant: &'ast Variant) {
        if !is_test_only(&variant.attrs) {
            visit::visit_variant(self, variant);
        }
    }

    fn visit_fn_arg(&mut self, arg: &'ast FnArg) {
        let attrs = match arg {
            FnArg::Receiver(receiver) => &receiver.attrs,
            FnArg::Typed(typed) => &typed.attrs,
        };
        if !is_test_only(attrs) {
            visit::visit_fn_arg(self, arg);
        }
    }

    fn visit_generic_param(&mut self, param: &'ast GenericParam) {
        if !is_test_only(generic_param_attrs(param)) {
            visit::visit_generic_param(self, param);
        }
    }

    fn visit_pat_type(&mut self, pattern: &'ast PatType) {
        if !is_test_only(&pattern.attrs) {
            visit::visit_pat_type(self, pattern);
        }
    }

    fn visit_arm(&mut self, arm: &'ast Arm) {
        if !is_test_only(&arm.attrs) {
            visit::visit_arm(self, arm);
        }
    }

    fn visit_local(&mut self, local: &'ast Local) {
        if !is_test_only(&local.attrs) {
            visit::visit_local(self, local);
        }
    }

    fn visit_stmt_macro(&mut self, statement: &'ast StmtMacro) {
        if !is_test_only(&statement.attrs) {
            visit::visit_stmt_macro(self, statement);
        }
    }

    fn visit_path(&mut self, path: &'ast SynPath) {
        self.record_path(path);
        visit::visit_path(self, path);
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        self.inspect_use_tree(&mut Vec::new(), &item.tree);
        visit::visit_item_use(self, item);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if self.rules.hot_trace && call.method == "emit" {
            self.record(".emit(");
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_macro(&mut self, mac: &'ast Macro) {
        self.record_path(&mac.path);
        if self.rules.worker_io {
            let name = mac
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string());
            if name.as_ref().is_some_and(|name| {
                matches!(name.as_str(), "print" | "println" | "eprint" | "eprintln")
            }) {
                self.record("print macro");
            }
        }
        self.inspect_macro_tokens(mac);
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        if let Some(name) = self_type_name(&item.self_ty) {
            self.impl_types.push(name);
            visit::visit_item_impl(self, item);
            self.impl_types.pop();
        } else {
            visit::visit_item_impl(self, item);
        }
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        let entered = self.enter_fn(&item.sig.ident.to_string());
        visit::visit_item_fn(self, item);
        if entered {
            self.exempt_stack.pop();
        }
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        let ident = item.sig.ident.to_string();
        let key = match self.impl_types.last() {
            Some(self_type) => format!("{self_type}::{ident}"),
            None => ident,
        };
        let entered = self.enter_fn(&key);
        visit::visit_impl_item_fn(self, item);
        if entered {
            self.exempt_stack.pop();
        }
    }
}

/// Last path segment of an `impl` self type, so `impl Drop for
/// TcpShardTransport` qualifies its `drop` as `TcpShardTransport::drop` rather
/// than matching every `drop` in the module.
fn self_type_name(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(path) => path
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string()),
        Type::Reference(reference) => self_type_name(&reference.elem),
        _ => None,
    }
}

struct ScanResult {
    /// Findings outside every exempt function: the module's violations.
    findings: BTreeSet<&'static str>,
    /// Findings inside each exempt function, keyed by index in the policy's
    /// exemption list.
    exempt_findings: BTreeMap<usize, BTreeSet<&'static str>>,
    /// Indices of exempt entries whose definition was reached.
    seen_exemptions: BTreeSet<usize>,
}

fn scan_source(
    source: &str,
    rules: Rules,
    exemptions: &'static [FnExemption],
) -> Result<ScanResult> {
    let file = syn::parse_file(source).context("Rust source parse failed")?;
    let mut visitor = ContractVisitor::new(rules, exemptions);
    visitor.visit_file(&file);
    Ok(ScanResult {
        findings: visitor.findings,
        exempt_findings: visitor.exempt_findings,
        seen_exemptions: visitor.seen_exemptions,
    })
}

/// Violations of a source with nothing exempted.
fn findings_of(source: &str, rules: Rules) -> Result<BTreeSet<&'static str>> {
    Ok(scan_source(source, rules, &[])?.findings)
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_workspace_text_files(root: &Path, paths: &[PathBuf]) -> Result<String> {
    let mut combined = String::new();
    for path in paths {
        let source = std::fs::read_to_string(root.join(path))
            .with_context(|| format!("failed to read {}", path.display()))?;
        combined.push_str(&source);
        combined.push('\n');
    }
    Ok(combined)
}

/// Identifiers whose presence on a line can make the visitor record `finding`.
///
/// These are the tokens the rules above actually match on, so replacing them
/// neutralizes a line for the scanner. Identifier for identifier, which is
/// why a neutralized source always still parses.
fn diagnostic_idents(finding: &str) -> &'static [&'static str] {
    match finding {
        "print macro" => &["print", "println", "eprint", "eprintln"],
        "std::fs" => &["fs"],
        "BufWriter" => &["BufWriter"],
        "sync_channel" => &["sync_channel"],
        "mpsc::channel" => &["mpsc", "channel"],
        "log/tracing path" => &["log", "tracing"],
        "nd_runtime_trace" => &["nd_runtime_trace"],
        "TraceEvent" => &["TraceEvent"],
        ".emit(" => &["emit"],
        _ => &[],
    }
}

/// Substitute identifier. Matches no rule, so a line rewritten to it is
/// invisible to the scanner while staying syntactically a line of Rust.
const INERT_IDENT: &str = "nd_contract_inert";

const CANDIDATE_LIMIT: usize = 12;

/// Confirmed lines behind a module's findings, and how many further confirmed
/// lines are not being shown.
#[derive(Default)]
struct Candidates {
    lines: Vec<String>,
    dropped: usize,
}

impl Candidates {
    /// Render for the failure message. Never silently truncates: a dropped
    /// count is stated, because a diagnostic that hides part of the answer
    /// costs the same debugging time as a wrong one.
    fn render(&self) -> String {
        if self.lines.is_empty() {
            return "(no line reproduces the finding on its own; inspect the parser finding \
                    directly — a path or import split across lines belongs to none of them)"
                .to_owned();
        }
        let listed = self.lines.join("\n  ");
        if self.dropped == 0 {
            listed
        } else {
            format!(
                "{listed}\n  ... and {} further confirmed line(s) not shown",
                self.dropped
            )
        }
    }
}

/// Split a line into identifier runs and everything else, in order.
fn ident_runs(line: &str) -> Vec<(bool, String)> {
    let mut runs: Vec<(bool, String)> = Vec::new();
    for character in line.chars() {
        let is_ident = character.is_alphanumeric() || character == '_';
        match runs.last_mut() {
            Some((kind, text)) if *kind == is_ident => text.push(character),
            _ => runs.push((is_ident, character.to_string())),
        }
    }
    runs
}

fn line_names_ident(line: &str, idents: &BTreeSet<&str>) -> bool {
    ident_runs(line)
        .iter()
        .any(|(is_ident, text)| *is_ident && idents.contains(text.as_str()))
}

fn neutralize_line(line: &str, idents: &BTreeSet<&str>) -> String {
    ident_runs(line)
        .into_iter()
        .map(|(is_ident, text)| {
            if is_ident && idents.contains(text.as_str()) {
                INERT_IDENT.to_owned()
            } else {
                text
            }
        })
        .collect()
}

fn scan_lines(
    lines: &[String],
    rules: Rules,
    exemptions: &'static [FnExemption],
) -> Option<BTreeSet<&'static str>> {
    scan_source(&lines.join("\n"), rules, exemptions)
        .ok()
        .map(|scan| scan.findings)
}

/// Locate the lines a module's findings actually came from.
///
/// A plain text search over the whole file cannot do this: it has no idea
/// which spans the visitor judged, so it happily points at `cfg(test)` code
/// and at exempt functions, which is how this diagnostic used to name an
/// innocent `std::fs::write` inside a unit test as the cause of a red gate.
///
/// `syn` here is built without `proc-macro2/span-locations`, so an AST node
/// carries no line number to report. What is available is the scanner itself:
/// neutralize every candidate line at once to get a baseline, then restore one
/// line at a time and keep only the lines that raise a finding the baseline
/// did not. That answers the question the visitor answers, one line at a time,
/// so `cfg(test)` pruning and function exemptions are honoured by construction.
fn candidate_lines(
    source: &str,
    findings: &BTreeSet<&'static str>,
    rules: Rules,
    exemptions: &'static [FnExemption],
) -> Candidates {
    let idents: BTreeSet<&str> = findings
        .iter()
        .flat_map(|finding| diagnostic_idents(finding).iter().copied())
        .collect();
    if idents.is_empty() {
        return Candidates::default();
    }

    let lines: Vec<String> = source.lines().map(ToOwned::to_owned).collect();
    let indices: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line_names_ident(line, &idents))
        .map(|(index, _)| index)
        .collect();

    let mut probe = lines.clone();
    for &index in &indices {
        if let Some(slot) = probe.get_mut(index) {
            *slot = neutralize_line(slot, &idents);
        }
    }
    // Unparseable input reaches this only from a caller that hand-built its
    // findings; fail open and show every textual candidate rather than assert
    // a filtered list we could not compute.
    let Some(baseline) = scan_lines(&probe, rules, exemptions) else {
        return truncate(
            indices
                .iter()
                .filter_map(|&index| render_line(&lines, index))
                .collect(),
        );
    };

    let mut confirmed = Vec::new();
    for &index in &indices {
        let (Some(original), Some(slot)) = (lines.get(index), probe.get_mut(index)) else {
            continue;
        };
        let neutralized = std::mem::replace(slot, original.clone());
        let probed = scan_lines(&probe, rules, exemptions);
        if let Some(slot) = probe.get_mut(index) {
            *slot = neutralized;
        }
        let raises_more = probed.is_none_or(|found| !found.is_subset(&baseline));
        if raises_more {
            if let Some(rendered) = render_line(&lines, index) {
                confirmed.push(rendered);
            }
        }
    }
    truncate(confirmed)
}

fn render_line(lines: &[String], index: usize) -> Option<String> {
    lines
        .get(index)
        .map(|line| format!("line {}: {}", index.saturating_add(1), line.trim()))
}

fn truncate(mut lines: Vec<String>) -> Candidates {
    let dropped = lines.len().saturating_sub(CANDIDATE_LIMIT);
    lines.truncate(CANDIDATE_LIMIT);
    Candidates { lines, dropped }
}

#[test]
fn production_workers_exclude_blocking_io_and_tracing() -> Result<()> {
    anyhow::ensure!(MODULES.len() == 12, "worker module inventory changed");
    anyhow::ensure!(
        MODULES.iter().filter(|module| module.hot_leaf).count() == 10,
        "hot-leaf module inventory changed"
    );

    anyhow::ensure!(
        MODULES
            .iter()
            .map(|module| module.io_exempt.len())
            .sum::<usize>()
            == 3,
        "worker IO exemption inventory changed"
    );

    let root = repo_root();
    let mut violations = Vec::new();
    for module in MODULES {
        let path = root.join(module.path);
        let rules = Rules {
            worker_io: true,
            hot_trace: module.hot_leaf,
        };
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let scan = scan_source(&source, rules, module.io_exempt)
            .with_context(|| format!("failed to scan {}", path.display()))?;

        // The exemptions fail closed in both directions. Without the existence
        // half, renaming an exempt function would silently move it back under
        // the rules while leaving a dead entry that looks like it still
        // governs something; without the equality half, an exemption granted
        // for one construct would cover every later one.
        for (index, exemption) in module.io_exempt.iter().enumerate() {
            anyhow::ensure!(
                scan.seen_exemptions.contains(&index),
                "{}: exempt function `{}` has no production definition in this module. \
                 Point the entry at the new name, or delete it.",
                module.path,
                exemption.name
            );
            let raised = scan
                .exempt_findings
                .get(&index)
                .cloned()
                .unwrap_or_default();
            let waived: BTreeSet<&'static str> = exemption.waived.iter().copied().collect();
            anyhow::ensure!(
                raised == waived,
                "{}: exempt function `{}` raises [{}] but is allowed exactly [{}]. \
                 Widening an exemption is a deliberate act; narrowing one is too.",
                module.path,
                exemption.name,
                raised.iter().copied().collect::<Vec<_>>().join(", "),
                waived.iter().copied().collect::<Vec<_>>().join(", ")
            );
        }

        if !scan.findings.is_empty() {
            let candidates = candidate_lines(&source, &scan.findings, rules, module.io_exempt);
            violations.push(format!(
                "{}: {}\n  confirmed lines:\n  {}",
                module.path,
                scan.findings.iter().copied().collect::<Vec<_>>().join(", "),
                candidates.render()
            ));
        }
    }
    anyhow::ensure!(
        violations.is_empty(),
        "forbidden production worker construct(s):\n{}",
        violations.join("\n")
    );
    Ok(())
}

#[test]
fn workspace_has_no_direct_log_dependency() -> Result<()> {
    let root = repo_root();
    let mut manifests = vec![PathBuf::from("Cargo.toml")];
    for entry in std::fs::read_dir(root.join("crates")).context("read crates directory")? {
        let path = entry?.path().join("Cargo.toml");
        if path.is_file() {
            manifests.push(path.strip_prefix(&root)?.to_owned());
        }
    }
    manifests.sort();
    let source = read_workspace_text_files(&root, &manifests)?;
    let direct_log = source.lines().any(|line| {
        let compact: String = line
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        compact.starts_with("log=") || compact.starts_with("log.workspace=")
    });
    anyhow::ensure!(!direct_log, "direct log facade dependency remains");
    Ok(())
}

#[test]
fn removed_nan_environment_controls_stay_unreachable() -> Result<()> {
    let root = repo_root();
    let mut paths = Vec::new();
    for relative in ["crates", "scripts"] {
        for entry in walkdir::WalkDir::new(root.join(relative)) {
            let entry = entry?;
            if entry.file_type().is_file()
                && entry
                    .path()
                    .extension()
                    .and_then(|extension| extension.to_str())
                    != Some("md")
            {
                paths.push(entry.path().strip_prefix(&root)?.to_owned());
            }
        }
    }
    paths.sort();
    for path in paths {
        let bytes = std::fs::read(root.join(&path))
            .with_context(|| format!("failed to read {}", path.display()))?;
        let Ok(source) = std::str::from_utf8(&bytes) else {
            continue;
        };
        for suffix in ["POISON", "GUARD_OFF", "PROBE"] {
            let name = format!("ND_NAN_{suffix}");
            anyhow::ensure!(
                !source.contains(&name),
                "{}: removed control remains: {name}",
                path.display()
            );
        }
    }
    Ok(())
}

#[test]
fn cfg_test_fields_do_not_hide_production_siblings_or_following_impls() -> Result<()> {
    let source = r#"
        struct Worker {
            #[cfg(test)]
            test_writer: BufWriter<Vec<u8>>,
            production_writer: BufWriter<Vec<u8>>,
        }
        impl Worker {
            fn production_after_field() { println!("production"); }
        }
    "#;
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: false,
        },
    )?;
    anyhow::ensure!(
        findings == BTreeSet::from(["BufWriter", "print macro"]),
        "unexpected findings: {findings:?}"
    );
    Ok(())
}

#[test]
fn cfg_test_nodes_are_excluded_but_later_production_is_scanned() -> Result<()> {
    let source = r#"
        #[cfg(test)]
        mod tests { fn probe() { println!("test module"); } }
        struct Worker {
            #[cfg(test)] test_writer: BufWriter<Vec<u8>>,
            live: usize,
        }
        fn worker(#[cfg(test)] trace: TraceEvent, live: usize) {
            call(#[cfg(test)] tracing::event!(), live);
            match live {
                #[cfg(test)] _ => println!("test arm"),
                _ => {}
            }
            #[cfg(test)]
            println!("test statement");
        }
        fn production_after_tests() { nd_runtime_trace::emit(); }
    "#;
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: true,
        },
    )?;
    anyhow::ensure!(
        findings == BTreeSet::from(["nd_runtime_trace"]),
        "unexpected findings: {findings:?}"
    );
    Ok(())
}

#[test]
fn cfg_evaluation_skips_only_definitely_test_only_nodes() -> Result<()> {
    let source = r#"
        #[cfg(all(test, feature = "probe"))]
        fn test_only() { println!("test"); }
        #[cfg(any(test, feature = "production_probe"))]
        fn maybe_production() { eprintln!("production feature"); }
        #[cfg_attr(not(test), cfg(test))]
        fn removed_in_production() { tracing::event!(); }
        #[cfg_attr(feature = "maybe_remove", cfg(test))]
        fn conservatively_production() { std::fs::File::open("x"); }
    "#;
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: false,
        },
    )?;
    anyhow::ensure!(
        findings == BTreeSet::from(["print macro", "std::fs"]),
        "unexpected findings: {findings:?}"
    );
    Ok(())
}

#[test]
fn malformed_rust_fails_closed() {
    let result = findings_of(
        "#[cfg(test)] mod unfinished {",
        Rules {
            worker_io: true,
            hot_trace: true,
        },
    );
    assert!(result.is_err(), "malformed source passed the scanner");
}

#[test]
fn comments_and_literals_are_not_code() -> Result<()> {
    let source = r##"
        fn clean() {
            // println!("comment");
            let _raw = r#"tracing::event!(); BufWriter"#;
            let _ordinary = "std::fs::File mpsc::channel";
        }
    "##;
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: true,
        },
    )?;
    anyhow::ensure!(findings.is_empty(), "unexpected findings: {findings:?}");
    Ok(())
}

const WORKER_IO_ONLY: Rules = Rules {
    worker_io: true,
    hot_trace: false,
};

/// `candidate_lines` with no exemptions, for the diagnostic self-tests.
fn diagnostic_for(source: &str, rules: Rules) -> Result<String> {
    let findings = findings_of(source, rules)?;
    Ok(candidate_lines(source, &findings, rules, &[]).render())
}

#[test]
fn violation_diagnostics_include_candidate_line_numbers() -> Result<()> {
    let source = "fn clean() {}\nfn worker() { println!(\"production\"); }\n";
    let diagnostic = diagnostic_for(source, WORKER_IO_ONLY)?;
    anyhow::ensure!(
        diagnostic.contains("line 2:") && diagnostic.contains("println!"),
        "missing candidate location: {diagnostic}"
    );
    Ok(())
}

#[test]
fn violation_diagnostics_keep_source_line_order() -> Result<()> {
    let source = "fn a() { let _: BufWriter<Vec<u8>>; }\nfn b() {}\nfn c() {}\nfn d() {}\n\
                  fn e() {}\nfn f() {}\nfn g() {}\nfn h() {}\nfn i() {}\n\
                  fn j() { let _: BufWriter<Vec<u8>>; }\n";
    let diagnostic = diagnostic_for(source, WORKER_IO_ONLY)?;
    let line_one = diagnostic.find("line 1:");
    let line_ten = diagnostic.find("line 10:");
    anyhow::ensure!(
        matches!((line_one, line_ten), (Some(one), Some(ten)) if one < ten),
        "candidate lines reordered: {diagnostic}"
    );
    Ok(())
}

/// The regression this diagnostic was rewritten for. The old text scan named
/// `sharded_hybrid.rs:2844` -- a `std::fs::write` inside a unit test -- as the
/// cause of a red gate, sending a reader to innocent code.
#[test]
fn violation_diagnostics_exclude_cfg_test_lines() -> Result<()> {
    let source = "#[cfg(test)]\nmod tests {\n    fn probe() { println!(\"test\"); }\n}\n\
                  fn worker() { println!(\"production\"); }\n";
    let diagnostic = diagnostic_for(source, WORKER_IO_ONLY)?;
    anyhow::ensure!(
        diagnostic.contains("line 5:") && !diagnostic.contains("line 3:"),
        "cfg(test) line reported as a production violation: {diagnostic}"
    );
    Ok(())
}

/// A line that mentions a watched token but cannot raise the finding on its
/// own is noise, not evidence.
#[test]
fn violation_diagnostics_exclude_lines_that_raise_nothing() -> Result<()> {
    let source = "fn worker() {\n    let file = String::new();\n    \
                  let _ = std::fs::read_to_string(&file);\n}\n";
    let diagnostic = diagnostic_for(source, WORKER_IO_ONLY)?;
    anyhow::ensure!(
        diagnostic.contains("line 3:") && !diagnostic.contains("line 2:"),
        "unrelated line reported: {diagnostic}"
    );
    Ok(())
}

/// Truncation must be visible. The previous implementation took twelve and
/// said nothing about the rest.
#[test]
fn violation_diagnostics_report_the_truncated_count() -> Result<()> {
    let body = "    println!(\"x\");\n".repeat(15);
    let source = format!("fn worker() {{\n{body}}}\n");
    let diagnostic = diagnostic_for(&source, WORKER_IO_ONLY)?;
    anyhow::ensure!(
        diagnostic.contains("and 3 further confirmed line(s) not shown"),
        "silent truncation: {diagnostic}"
    );
    Ok(())
}

/// An aliased import is the violating line; the call site through the alias is
/// not. Naming the import is what sends a reader somewhere they can fix it.
#[test]
fn violation_diagnostics_blame_the_alias_import_not_the_call() -> Result<()> {
    let source = "use std::println as worker_print;\nfn worker() { worker_print!(\"x\"); }\n";
    let diagnostic = diagnostic_for(source, WORKER_IO_ONLY)?;
    anyhow::ensure!(
        diagnostic.contains("line 1:") && !diagnostic.contains("line 2:"),
        "alias import not blamed: {diagnostic}"
    );
    Ok(())
}

/// A construct split across lines belongs to no single line, so the probe
/// confirms none of them. Saying that is right; listing every line that
/// mentions a token is what the old text scan did.
#[test]
fn violation_diagnostics_admit_when_no_line_reproduces_the_finding() -> Result<()> {
    let source = "fn worker() {\n    let _ = mpsc::\n        channel();\n}\n";
    let diagnostic = diagnostic_for(source, WORKER_IO_ONLY)?;
    anyhow::ensure!(
        diagnostic.contains("no line reproduces the finding on its own"),
        "a split construct was pinned to one of its lines: {diagnostic}"
    );
    Ok(())
}

#[test]
fn an_exempt_function_confines_its_waiver_to_itself() -> Result<()> {
    const EXEMPT: &[FnExemption] = &[FnExemption {
        name: "bootstrap",
        waived: &["std::fs"],
    }];
    let source = r#"
        fn bootstrap() { let _ = std::fs::read_to_string("x"); }
        fn worker_loop() { let _ = std::fs::read_to_string("x"); }
    "#;
    let scan = scan_source(source, WORKER_IO_ONLY, EXEMPT)?;
    anyhow::ensure!(
        scan.findings == BTreeSet::from(["std::fs"]),
        "exemption leaked past its function: {:?}",
        scan.findings
    );
    anyhow::ensure!(
        scan.exempt_findings.get(&0) == Some(&BTreeSet::from(["std::fs"])),
        "exempt finding not attributed: {:?}",
        scan.exempt_findings
    );
    anyhow::ensure!(scan.seen_exemptions.contains(&0), "definition not seen");
    Ok(())
}

#[test]
fn an_exempt_method_is_matched_by_its_self_type() -> Result<()> {
    const EXEMPT: &[FnExemption] = &[FnExemption {
        name: "Transport::drop",
        waived: &["std::fs"],
    }];
    let source = r#"
        impl Drop for Transport {
            fn drop(&mut self) { let _ = std::fs::remove_file("x"); }
        }
        impl Other {
            fn drop(&mut self) { let _ = std::fs::remove_file("x"); }
        }
    "#;
    let scan = scan_source(source, WORKER_IO_ONLY, EXEMPT)?;
    anyhow::ensure!(
        scan.findings == BTreeSet::from(["std::fs"]),
        "a bare method name exempted every same-named method: {:?}",
        scan.findings
    );
    Ok(())
}

/// The drift half of the fail-closed contract: renaming or `cfg(test)`-gating
/// an exempt function must leave its entry visibly unmatched, never silently
/// inert. The main test turns this into a hard failure.
#[test]
fn an_exempt_function_that_stops_existing_is_visible() -> Result<()> {
    const EXEMPT: &[FnExemption] = &[FnExemption {
        name: "bootstrap",
        waived: &["std::fs"],
    }];
    let renamed = r#"fn bootstrap_v2() { let _ = std::fs::read_to_string("x"); }"#;
    let scan = scan_source(renamed, WORKER_IO_ONLY, EXEMPT)?;
    anyhow::ensure!(
        scan.seen_exemptions.is_empty() && scan.findings == BTreeSet::from(["std::fs"]),
        "a renamed function kept its exemption: {:?} {:?}",
        scan.seen_exemptions,
        scan.findings
    );

    let hidden = r#"
        #[cfg(test)]
        fn bootstrap() { let _ = std::fs::read_to_string("x"); }
    "#;
    let scan = scan_source(hidden, WORKER_IO_ONLY, EXEMPT)?;
    anyhow::ensure!(
        scan.seen_exemptions.is_empty(),
        "a cfg(test) definition satisfied a production exemption"
    );
    Ok(())
}

/// The rule names `std::fs`, not `std::fs::File`. It used to name the type,
/// which left `read_to_string`, `remove_file`, `rename`, `create_dir_all` and
/// `OpenOptions` completely unguarded in every worker module.
#[test]
fn every_blocking_std_fs_entry_point_is_caught() -> Result<()> {
    for call in [
        "std::fs::read_to_string",
        "std::fs::write",
        "std::fs::remove_file",
        "std::fs::rename",
        "std::fs::create_dir_all",
        "std::fs::read_dir",
        "std::fs::OpenOptions::new",
        "std::fs::File::open",
    ] {
        let source = format!("fn worker() {{ let _ = {call}(\"x\"); }}");
        let findings = findings_of(&source, WORKER_IO_ONLY)?;
        anyhow::ensure!(
            findings == BTreeSet::from(["std::fs"]),
            "{call} escaped the blocking-IO rule: {findings:?}"
        );
    }
    let import = findings_of("use std::fs;\nfn worker() {}\n", WORKER_IO_ONLY)?;
    anyhow::ensure!(
        import == BTreeSet::from(["std::fs"]),
        "a bare `use std::fs` import escaped: {import:?}"
    );
    Ok(())
}

#[test]
fn unparseable_macro_dsl_recurses_through_groups() -> Result<()> {
    let source = r#"
        fn worker() {
            worker_dsl! {
                lane => [{
                    print!("x"); println!("x"); eprint!("x"); eprintln!("x");
                    log::event!(); tracing::event!();
                    std::fs::File; BufWriter; sync_channel; mpsc::channel;
                    nd_runtime_trace; TraceEvent; sink.emit(value);
                    "println! tracing:: std::fs::File sink.emit(value)"
                }]
            }
        }
    "#;
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: true,
        },
    )?;
    anyhow::ensure!(
        findings
            == BTreeSet::from([
                ".emit(",
                "BufWriter",
                "TraceEvent",
                "log/tracing path",
                "mpsc::channel",
                "nd_runtime_trace",
                "print macro",
                "std::fs",
                "sync_channel",
            ]),
        "nested macro DSL findings incomplete: {findings:?}"
    );
    Ok(())
}

#[test]
fn aliased_print_macro_import_is_still_forbidden() -> Result<()> {
    let source = r#"
        use std::println as worker_print;
        fn worker() { worker_print!("production"); }
    "#;
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: false,
        },
    )?;
    anyhow::ensure!(
        findings == BTreeSet::from(["print macro"]),
        "aliased print macro escaped detection: {findings:?}"
    );
    Ok(())
}

#[test]
fn print_macro_alias_import_is_order_independent() -> Result<()> {
    let source = r#"
        fn worker() { worker_print!("production"); }
        use std::println as worker_print;
    "#;
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: false,
        },
    )?;
    anyhow::ensure!(
        findings == BTreeSet::from(["print macro"]),
        "later alias import escaped detection: {findings:?}"
    );
    Ok(())
}

#[test]
fn print_macro_alias_inside_unparseable_dsl_is_forbidden() -> Result<()> {
    let source = r#"
        use std::eprintln as worker_print;
        fn worker() {
            worker_dsl! { lane => [{ worker_print!("production") }] }
        }
    "#;
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: false,
        },
    )?;
    anyhow::ensure!(
        findings == BTreeSet::from(["print macro"]),
        "DSL alias escaped detection: {findings:?}"
    );
    Ok(())
}

#[test]
fn print_macro_alias_defined_inside_unparseable_dsl_is_forbidden() -> Result<()> {
    let source = r#"
        fn worker() {
            worker_dsl! {
                lane => [{ use std::eprintln as worker_print; worker_print!("production") }]
            }
        }
    "#;
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: false,
        },
    )?;
    anyhow::ensure!(
        findings == BTreeSet::from(["print macro"]),
        "macro-local print alias escaped detection: {findings:?}"
    );
    Ok(())
}

#[test]
fn non_call_print_identifier_inside_unparseable_dsl_is_allowed() -> Result<()> {
    let source = r"
        fn worker() {
            worker_dsl! {
                lane => [{ let print = formatter; let println_count = print_count; }]
            }
        }
    ";
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: false,
        },
    )?;
    anyhow::ensure!(
        findings.is_empty(),
        "non-call print identifier produced false positive: {findings:?}"
    );
    Ok(())
}

#[test]
fn grouped_worker_imports_inside_unparseable_dsl_are_forbidden() -> Result<()> {
    let source = r"
        fn worker() {
            worker_dsl! {
                lane => [{
                    use std::fs::{File as WorkerFile};
                    use std::sync::mpsc::{channel as worker_channel};
                }]
            }
        }
    ";
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: false,
        },
    )?;
    anyhow::ensure!(
        findings == BTreeSet::from(["mpsc::channel", "std::fs"]),
        "grouped imports escaped fallback: {findings:?}"
    );
    Ok(())
}

#[test]
fn nested_prefix_groups_inside_unparseable_dsl_are_forbidden() -> Result<()> {
    let source = r"
        fn worker() {
            worker_dsl! {
                lane => [{
                    use std::{fs::{File as WorkerFile}};
                    use std::{sync::{mpsc::{channel as worker_channel}}};
                }]
            }
        }
    ";
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: false,
        },
    )?;
    anyhow::ensure!(
        findings == BTreeSet::from(["mpsc::channel", "std::fs"]),
        "nested grouped imports escaped fallback: {findings:?}"
    );
    Ok(())
}

#[test]
fn sibling_group_branches_inherit_their_path_prefix() -> Result<()> {
    let source = r"
        fn worker() {
            worker_dsl! {
                lane => [{ use std::{io, fs::{metadata, File as WorkerFile}}; }]
            }
        }
    ";
    let findings = findings_of(
        source,
        Rules {
            worker_io: true,
            hot_trace: false,
        },
    )?;
    anyhow::ensure!(
        findings == BTreeSet::from(["std::fs"]),
        "sibling grouped import escaped fallback: {findings:?}"
    );
    Ok(())
}
