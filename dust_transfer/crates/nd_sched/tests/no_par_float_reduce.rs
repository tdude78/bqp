//! Banned-pattern lint: no chained Rayon reductions in production crates.
//!
//! Parallel floating-point reductions are schedule-dependent because floating-
//! point arithmetic is not associative. Production code must collect indexed
//! results, then reduce them serially in stable order. This test parses every
//! `crates/*/src/**/*.rs` file and rejects reducing terminals reached from a
//! Rayon parallel-iterator entry point on the same method chain.
//!
//! The chain does not have to be textually contiguous: a parallel entry bound
//! to a local (`let it = v.par_iter().map(f);`) taints that local, so the
//! reduction two statements later is still seen. `rayon::join`/`scope`/`spawn`
//! are reported separately -- they carry no chain at all, so nothing here can
//! judge what their closures accumulate into, and each one needs a human
//! reason recorded in `ALLOWANCES`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprMethodCall, Local, Macro, Token};

/// Rayon entry points that hand work to more than one thread.
///
/// `par_chunks_exact`/`par_chunks_exact_mut` are the entries the SIMD batch
/// paths actually use (`satpy_core/src/lib.rs`, `lightyear_odeint_rs/src/batch.rs`),
/// so their absence here made the guard blind to the hottest parallel code in
/// the workspace.
const PARALLEL_ENTRIES: &[&str] = &[
    "par_iter",
    "par_iter_mut",
    "into_par_iter",
    "par_bridge",
    "par_chunks",
    "par_chunks_mut",
    "par_chunks_exact",
    "par_chunks_exact_mut",
    "par_rchunks",
    "par_rchunks_mut",
    "par_rchunks_exact",
    "par_rchunks_exact_mut",
    "par_windows",
    "par_split",
    "par_split_mut",
    "par_drain",
];

/// Terminals whose result depends on how the work was split across threads.
///
/// `sum`/`product`/`fold`/`reduce` combine in scheduling order. The `try_*`
/// family is the same combination with an early exit, which additionally makes
/// *which* error escapes schedule-dependent. `min_by`/`max_by` take a
/// comparator that is `PartialOrd`-shaped for floats, and even with a total
/// comparator Rayon does not specify which of several equal elements wins, so
/// the payload attached to a tie can move. `for_each_with`/`for_each_init`
/// hand each thread its own mutable state, which is the classic shape of an
/// accumulator that is later combined.
///
/// Plain `for_each` is deliberately absent: `par_iter().for_each(...)` writing
/// into disjoint indexed slots is the *prescribed* replacement for a parallel
/// reduction, and flagging it would make the correct pattern unusable.
const REDUCING_TERMINALS: &[&str] = &[
    "sum",
    "product",
    "fold",
    "fold_with",
    "fold_chunks",
    "fold_chunks_with",
    "reduce",
    "reduce_with",
    "try_fold",
    "try_fold_with",
    "try_reduce",
    "try_reduce_with",
    "min_by",
    "max_by",
    "min_by_key",
    "max_by_key",
    "for_each_with",
    "for_each_init",
    // The `try_*` early-exit family. The doc-comment above has always named it
    // -- "the same combination with an early exit, which additionally makes
    // *which* error escapes schedule-dependent" -- but the list held only
    // try_fold/try_reduce, so try_for_each and the *_any finders were never
    // scanned. A guard that omits what its own prose condemns is the exact
    // shape this file exists to prevent.
    "try_for_each",
    "try_for_each_with",
    "try_for_each_init",
    "find_any",
    "find_map_any",
    "position_any",
];

/// Free functions that spawn parallel work without an iterator chain.
///
/// These carry no chain for the visitor to walk, so what their closures write
/// into is invisible here. They are reported so a human states, once, what the
/// tasks combine.
///
/// `rayon::broadcast` is deliberately absent: it returns a `Vec` in thread-index
/// order, which is the indexed-collect pattern this guard exists to steer code
/// toward, not a reduction. Flagging it would contradict
/// `allows_par_iter_map_collect`. All five of its workspace call sites collect
/// worker identity or prime a per-thread cache.
const RAYON_SPAWNERS: &[&str] = &[
    "join",
    "join_context",
    "scope",
    "scope_fifo",
    "in_place_scope",
    "in_place_scope_fifo",
    "spawn",
    "spawn_fifo",
    // Same class as `spawn`: fire-and-forget across every thread, so what the
    // closure writes into is invisible to the chain walker. (`rayon::broadcast`
    // stays absent for the reason given above -- it returns an indexed Vec.)
    "spawn_broadcast",
];

/// Hits that are reviewed and accepted, with the count they are accepted at.
///
/// Keyed by (path relative to `crates/`, finding). The count is exact in both
/// directions on purpose: a new occurrence in an already-allowed file fires,
/// and an allowance left behind by deleted code fires too, so this list cannot
/// rot into a silent blanket exemption for a whole file.
///
/// An entry here is a claim that the construct is order-independent. Adding one
/// without being able to write that sentence is how the guard stops meaning
/// anything.
///
/// Empty since 2026-08-06. Both entries named `dust_estimates_rs`
/// (`src/compute.rs` and `src/parallel_branch_identity.rs`, one reviewed
/// `rayon::scope` each), and both files were deleted with the probabilistic
/// GMM dust-mass search — see `docs/REFACTOR_BLOCKLIST.md` entry B4. Because
/// the counts are exact in both directions, leaving the entries behind would
/// have fired this test, so an empty table is the correct state and not a
/// weakening: no file has an exemption.
const ALLOWANCES: &[(&str, &str, usize, &str)] = &[
    // Repopulated 2026-08-20, when the `try_*` family was added to
    // REDUCING_TERMINALS (it had always been named in the doc-comment above and
    // never actually scanned). Each entry below is the required claim: NO VALUE
    // IS COMBINED, so there is no reduction and no float ordering to protect.
    //
    // What these three do share is that `try_*` early-exits, so WHICH error
    // escapes when two rows fail is schedule-dependent. That is a diagnosis
    // concern, not a value concern -- no output bit moves -- and it is recorded
    // as a known open finding rather than hidden here. Restructuring to an
    // indexed collect would take the first Err in index order, at the cost of
    // the early exit; see the purge ledger's W3-D A1.
    (
        "lightyear_odeint_rs/src/batch.rs",
        "try_for_each_init",
        1,
        "par_chunks_exact zipped with par_chunks_exact_mut: each closure writes \
         only its own disjoint out_chunk, so nothing is combined across lanes \
         and the written bytes do not depend on scheduling.",
    ),
    (
        "lightyear_odeint_rs/src/batch.rs",
        "try_for_each",
        1,
        "The serial twin of the arm above, inside the same `if use_parallel` \
         branch; reached from the parallel entry only by the walker, and its \
         writes are the same disjoint per-chunk writes.",
    ),
];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Finding {
    /// Path relative to `crates/`, or the raw path for ad-hoc scans.
    file: String,
    /// `sum`, `try_reduce`, `rayon::scope`, ...
    what: String,
}

#[derive(Default)]
struct ParallelReductionVisitor {
    findings: Vec<String>,
    /// Locals currently bound to something that came off a parallel entry.
    tainted: Vec<String>,
}

impl ParallelReductionVisitor {
    fn inspect_macro_tokens(&mut self, mac: &Macro) {
        if let Ok(expr) = syn::parse2::<Expr>(mac.tokens.clone()) {
            self.visit_expr(&expr);
            return;
        }

        let parser = Punctuated::<Expr, Token![,]>::parse_terminated;
        if let Ok(expressions) = parser.parse2(mac.tokens.clone()) {
            for expr in expressions {
                self.visit_expr(&expr);
            }
        }
    }

    /// Whether `expr` reaches a Rayon entry, through method receivers or
    /// through a local that a previous statement tainted.
    fn chain_is_parallel(&self, expr: &Expr) -> bool {
        match expr {
            Expr::MethodCall(call) => {
                PARALLEL_ENTRIES.contains(&call.method.to_string().as_str())
                    || self.chain_is_parallel(&call.receiver)
            }
            Expr::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|ident| self.tainted.iter().any(|name| name == &ident.to_string())),
            Expr::Await(expr) => self.chain_is_parallel(&expr.base),
            Expr::Group(expr) => self.chain_is_parallel(&expr.expr),
            Expr::Paren(expr) => self.chain_is_parallel(&expr.expr),
            Expr::Reference(expr) => self.chain_is_parallel(&expr.expr),
            Expr::Try(expr) => self.chain_is_parallel(&expr.expr),
            _ => false,
        }
    }
}

/// The `rayon::` free function this call names, if it is one.
///
/// Matches both `rayon::scope(..)` and a bare `scope(..)` reached through a
/// `use rayon::scope;` import, since the guard cannot resolve imports.
fn rayon_spawner_name(call: &ExprCall) -> Option<String> {
    let Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    let segments: Vec<String> = path
        .path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    let last = segments.last()?;
    if !RAYON_SPAWNERS.contains(&last.as_str()) {
        return None;
    }
    // A bare `join()`/`scope()` is far more likely to be `JoinHandle::join`,
    // `str::join` or a local helper, so require the `rayon` qualifier.
    if !segments.iter().any(|segment| segment == "rayon") {
        return None;
    }
    Some(format!("rayon::{last}"))
}

impl<'ast> Visit<'ast> for ParallelReductionVisitor {
    fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
        let method = call.method.to_string();
        if REDUCING_TERMINALS.contains(&method.as_str()) && self.chain_is_parallel(&call.receiver) {
            self.findings.push(method);
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if let Some(name) = rayon_spawner_name(call) {
            self.findings.push(name);
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_local(&mut self, local: &'ast Local) {
        visit::visit_local(self, local);
        let Some(init) = local.init.as_ref() else {
            return;
        };
        if !self.chain_is_parallel(&init.expr) {
            return;
        }
        // Only simple `let name = ...` bindings are tracked. A destructuring
        // pattern splits the chain into pieces this guard cannot follow, and
        // guessing would produce hits nobody can act on.
        if let syn::Pat::Ident(pat) = &local.pat {
            self.tainted.push(pat.ident.to_string());
        }
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        let outer = std::mem::take(&mut self.tainted);
        visit::visit_item_fn(self, item);
        self.tainted = outer;
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        let outer = std::mem::take(&mut self.tainted);
        visit::visit_impl_item_fn(self, item);
        self.tainted = outer;
    }

    fn visit_macro(&mut self, mac: &'ast Macro) {
        self.inspect_macro_tokens(mac);
        visit::visit_macro(self, mac);
    }
}

fn detect_in_file(file: &syn::File) -> Vec<String> {
    let mut visitor = ParallelReductionVisitor::default();
    visitor.visit_file(file);
    visitor.findings
}

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn violation_in_statement(statement: &str) -> Option<String> {
    let wrapped = format!("fn lint_probe() {{ {statement}; }}");
    let file = syn::parse_file(&wrapped).ok()?;
    detect_in_file(&file)
        .into_iter()
        .next()
        .map(|what| format!("parallel iterator uses `.{what}`"))
}

fn findings_in_file(root: &Path, path: &Path) -> Vec<Finding> {
    let file = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    let src = match std::fs::read_to_string(path) {
        Ok(src) => src,
        Err(error) => {
            return vec![Finding {
                file,
                what: format!("read failed: {error}"),
            }];
        }
    };
    let parsed = match syn::parse_file(&src) {
        Ok(parsed) => parsed,
        Err(error) => {
            return vec![Finding {
                file,
                what: format!("parse failed: {error}"),
            }];
        }
    };

    detect_in_file(&parsed)
        .into_iter()
        .map(|what| Finding {
            file: file.clone(),
            what,
        })
        .collect()
}

fn scan_file(path: &Path) -> Vec<String> {
    findings_in_file(Path::new(""), path)
        .into_iter()
        .map(|finding| {
            format!(
                "{}: parallel iterator uses `.{}`",
                finding.file, finding.what
            )
        })
        .collect()
}

/// Count the `.rs` files a full `crates/*/src` walk would visit.
///
/// Exists so the workspace gate can prove its scan was non-empty; see the
/// assertion in `workspace_crates_have_no_parallel_float_reductions`.
fn crate_source_file_count(crates: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(crates) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path().join("src"))
        .filter(|src| src.is_dir())
        .map(|src| {
            walkdir::WalkDir::new(src)
                .into_iter()
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|x| x == "rs"))
                .count()
        })
        .sum()
}

fn collect_workspace_findings(crates: &Path) -> anyhow::Result<Vec<Finding>> {
    let entries = std::fs::read_dir(crates)
        .map_err(|error| anyhow::anyhow!("{}: read failed: {error}", crates.display()))?;
    let mut crate_dirs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    crate_dirs.sort();

    let mut findings = Vec::new();
    for crate_dir in crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        for entry in walkdir::WalkDir::new(&src_dir)
            .sort_by_file_name()
            .into_iter()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                findings.extend(findings_in_file(crates, path));
            }
        }
    }
    findings.sort();
    Ok(findings)
}

fn scan_workspace_crates_src(crates: &Path) -> Vec<String> {
    match collect_workspace_findings(crates) {
        Ok(findings) => findings
            .into_iter()
            .map(|finding| {
                format!(
                    "{}: parallel iterator uses `.{}`",
                    finding.file, finding.what
                )
            })
            .collect(),
        Err(error) => vec![error.to_string()],
    }
}

/// Subtract `ALLOWANCES` from `findings`, reporting both directions of drift.
fn unallowed(findings: &[Finding]) -> Vec<String> {
    unallowed_against(findings, ALLOWANCES)
}

/// The subtraction itself, over an explicit table.
///
/// Split out from [`unallowed`] so the both-directions self-check can exercise
/// it against a fixture. `ALLOWANCES` is legitimately empty, and a self-check
/// that borrows its first real entry stops running the moment that happens —
/// which is precisely when the empty table needs proving safe.
fn unallowed_against(
    findings: &[Finding],
    allowances: &[(&str, &str, usize, &str)],
) -> Vec<String> {
    let mut seen: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for finding in findings {
        let count = seen
            .entry((finding.file.as_str(), finding.what.as_str()))
            .or_insert(0_usize);
        *count = count.saturating_add(1);
    }

    let mut problems = Vec::new();
    for (file, what, expected, reason) in allowances {
        let found = seen.remove(&(file, what)).unwrap_or(0);
        if found != *expected {
            problems.push(format!(
                "{file}: allowance for `{what}` expects {expected} occurrence(s) but the scan \
                 found {found}. Re-review, then update the count or delete the allowance. \
                 Recorded reason: {reason}"
            ));
        }
    }
    for ((file, what), count) in seen {
        if what.starts_with("rayon::") {
            problems.push(format!(
                "{file}: {count} x `{what}` spawns parallel tasks with no chain to inspect. \
                 State in an ALLOWANCES entry what the tasks combine, or restructure so the \
                 result is an indexed collect."
            ));
        } else {
            problems.push(format!(
                "{file}: {count} x `.{what}` reached from a parallel entry (use indexed \
                 parallel collect + stable serial reduction, or add a reviewed ALLOWANCES entry)"
            ));
        }
    }
    problems.sort();
    problems
}

#[test]
fn workspace_crates_have_no_parallel_float_reductions() {
    // A scan that visits nothing reports no violations and passes. `read_dir`
    // failing does return a violation, so this fails closed on a missing tree
    // -- but a tree that exists and is empty, or a glob that stops matching,
    // is silent. Assert the walk found a plausible number of files first, so
    // "no violations" means "looked and found none".
    let scanned = crate_source_file_count(&crates_dir());
    assert!(
        scanned >= 200,
        "the workspace scan visited only {scanned} .rs files under crates/*/src; \
         it should see hundreds. A near-empty scan reports zero violations and \
         passes, which is indistinguishable from a clean workspace."
    );

    let findings = collect_workspace_findings(&crates_dir()).expect("workspace scan");
    let problems = unallowed(&findings);

    assert!(
        problems.is_empty(),
        "banned parallel reduction found:\n{}",
        problems.join("\n")
    );
}

mod self_check {
    use super::{
        collect_workspace_findings, crates_dir, scan_file, scan_workspace_crates_src, unallowed,
        unallowed_against, violation_in_statement, Finding,
    };

    #[test]
    fn detects_par_iter_sum() {
        let stmt = format!("let s = v.{}().map(f){}()", "par_iter", ".sum");
        assert!(violation_in_statement(&stmt).is_some());
    }

    #[test]
    fn detects_into_par_iter_reduce_and_par_bridge_fold() {
        let a = format!("x.{}().{}(id, op)", "into_par_iter", "reduce");
        let b = format!("x.{}().{}(init, op)", "par_bridge", "fold");
        assert!(violation_in_statement(&a).is_some());
        assert!(violation_in_statement(&b).is_some());
    }

    #[test]
    fn allows_par_iter_map_collect() {
        let stmt = format!("let v: Vec<_> = u.{}().map(f).collect()", "par_iter");
        assert!(violation_in_statement(&stmt).is_none());
    }

    #[test]
    fn allows_serial_sum() {
        assert!(violation_in_statement("let t: f64 = v.iter().copied().sum()").is_none());
    }

    /// Indexed parallel writes are the prescribed replacement for a reduction.
    /// If this ever starts failing the guard has made the correct pattern
    /// unusable and every real fix will look like a violation.
    #[test]
    fn allows_par_iter_for_each_indexed_write() {
        let stmt = format!(
            "out.{}().enumerate().for_each(|(i, slot)| {{ *slot = f(i); }})",
            "par_iter_mut"
        );
        assert!(violation_in_statement(&stmt).is_none());
    }

    #[test]
    fn detects_parallel_reduce_with() {
        let stmt = format!(
            "let best = if parallel {{ values.{}().filter_map(f).{}(pick_better) }} else {{ values.iter().filter_map(f).{}(pick_better) }}",
            "into_par_iter", "reduce_with", "reduce"
        );
        assert!(violation_in_statement(&stmt).is_some());
    }

    #[test]
    fn detects_turbofish_sum() {
        let stmt = format!(
            "let total = values.{}().copied().{}::<f64>()",
            "par_iter", "sum"
        );
        assert!(violation_in_statement(&stmt).is_some());
    }

    /// The entries the SIMD batch paths use. Before these were listed a `.sum()`
    /// on the hottest parallel chains in the workspace passed the guard.
    #[test]
    fn detects_par_chunks_exact_family() {
        for entry in [
            "par_chunks_exact",
            "par_chunks_exact_mut",
            "par_rchunks_exact",
            "par_drain",
        ] {
            let stmt = format!("let t: f64 = rows.{entry}(24).map(score).{}()", "sum");
            assert!(
                violation_in_statement(&stmt).is_some(),
                "`{entry}` is not recognised as a parallel entry"
            );
        }
    }

    #[test]
    fn detects_try_and_comparator_terminals() {
        for terminal in [
            "try_fold",
            "try_reduce",
            "try_reduce_with",
            "min_by",
            "max_by",
            "for_each_with",
            "for_each_init",
        ] {
            let stmt = format!("let r = rows.{}().{terminal}(a, b)", "par_iter");
            assert!(
                violation_in_statement(&stmt).is_some(),
                "`{terminal}` is not recognised as a reducing terminal"
            );
        }
    }

    /// A chain broken across statements. `chain_has_parallel_entry` used to walk
    /// only `.receiver`, so this whole shape was invisible.
    #[test]
    fn detects_reduction_through_a_local_binding() {
        let split = format!(
            "fn f(v: &[f64]) -> f64 {{ let it = v.{}().map(g); let s: f64 = it.{}(); s }}",
            "par_iter", "sum"
        );
        let file = syn::parse_file(&split).unwrap();
        assert_eq!(super::detect_in_file(&file).len(), 1, "{split}");

        let two_hops = format!(
            "fn f(v: &[f64]) -> f64 {{ let a = v.{}(); let b = a.map(g); b.{}(|x, y| x + y) }}",
            "into_par_iter", "reduce_with"
        );
        let file = syn::parse_file(&two_hops).unwrap();
        assert_eq!(super::detect_in_file(&file).len(), 1, "{two_hops}");
    }

    /// The taint must not leak between functions -- a serial `it` in one
    /// function is not the parallel `it` of another.
    #[test]
    fn taint_does_not_escape_its_function() {
        let src = format!(
            "fn parallel(v: &[f64]) -> f64 {{ let it = v.{}(); it.{}() }}\n\
             fn serial(v: &[f64]) -> f64 {{ let it = v.iter().copied(); it.{}() }}",
            "par_iter", "sum", "sum"
        );
        let file = syn::parse_file(&src).unwrap();
        assert_eq!(super::detect_in_file(&file).len(), 1, "{src}");
    }

    /// `rayon::broadcast` returns a thread-index-ordered `Vec`, so it is an
    /// indexed collect, not a reduction. This pins that exclusion as a decision
    /// rather than an oversight; a reduction *fed by* a broadcast is serial and
    /// therefore fine, but a parallel chain on top of one is not.
    #[test]
    fn broadcast_is_an_indexed_collect_not_a_reduction() {
        let stmt = format!("let names = {}::broadcast(|_| worker_name())", "rayon");
        assert!(violation_in_statement(&stmt).is_none());
        let reduced = format!(
            "let t: f64 = {}::broadcast(|_| load()).{}().{}()",
            "rayon", "par_iter", "sum"
        );
        assert!(violation_in_statement(&reduced).is_some());
    }

    #[test]
    fn detects_rayon_spawners_and_ignores_lookalikes() {
        for spawner in ["join", "scope", "spawn", "in_place_scope", "scope_fifo"] {
            let stmt = format!("{}::{spawner}(a, b)", "rayon");
            assert!(
                violation_in_statement(&stmt).is_some(),
                "`rayon::{spawner}` is invisible to the guard"
            );
        }
        // `JoinHandle::join`, `[&str]::join` and local helpers named `scope`
        // must not be swept up.
        assert!(violation_in_statement("let s = parts.join(\",\")").is_none());
        assert!(violation_in_statement("handle.join().unwrap()").is_none());
        assert!(violation_in_statement("let x = scope(a, b)").is_none());
    }

    #[test]
    fn handles_closure_semicolons_raw_urls_and_lifetimes() {
        let unique = format!(
            "nd_sched_no_par_float_reduce_syntax_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("syntax.rs");
        let src = r##"
fn closure_semicolon(rows: &[f64]) -> f64 {
    rows.par_iter().map(|x| { let y = *x; y }).sum::<f64>()
}
fn raw_url(rows: &[f64]) -> f64 {
    let _url = r#"https://example.invalid/path"#;
    rows.par_iter().copied().sum::<f64>()
}
fn lifetime<'a>(rows: &'a [f64]) -> f64 {
    rows.par_iter().map(|x: &'a f64| *x).sum::<f64>()
}
"##;
        std::fs::write(&path, src).unwrap();

        let violations = scan_file(&path);
        assert_eq!(violations.len(), 3, "{violations:#?}");

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn dynamic_workspace_traversal_covers_every_crate_src() {
        let unique = format!(
            "nd_sched_no_par_float_reduce_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let crates = root.join("crates");
        let kernel_src = crates.join("kernel_not_nd").join("src");
        let other_src = crates.join("another_kernel").join("src").join("nested");
        let ignored_tests = crates.join("kernel_not_nd").join("tests");
        std::fs::create_dir_all(&kernel_src).unwrap();
        std::fs::create_dir_all(&other_src).unwrap();
        std::fs::create_dir_all(&ignored_tests).unwrap();

        let old_grid_reduce = format!(
            "fn old_grid(rows: Vec<f64>) {{ let merged = rows.{}().map(process).{}(|| empty(), merge); }}",
            "into_par_iter", "reduce"
        );
        let old_compute_fold = format!(
            "fn old_compute(rows: Vec<f64>) {{ let total = rows.{}().map(score).{}(|| 0.0, |a, b| a + b); }}",
            "par_iter", "fold"
        );
        std::fs::write(kernel_src.join("lib.rs"), old_grid_reduce).unwrap();
        std::fs::write(other_src.join("module.rs"), old_compute_fold).unwrap();
        std::fs::write(
            ignored_tests.join("ignored.rs"),
            "let ignored = rows.par_iter().sum();",
        )
        .unwrap();

        let violations = scan_workspace_crates_src(&crates);
        assert_eq!(violations.len(), 2, "{violations:#?}");
        assert!(violations.iter().any(|v| v.contains("kernel_not_nd")));
        assert!(violations.iter().any(|v| v.contains("another_kernel")));
        assert!(violations.iter().all(|v| !v.contains("ignored.rs")));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ignores_comments_and_string_mentions() {
        let src = format!(
            "fn clean() {{ let _url = \"https://example.invalid\"; // mentions {}().{}()\n let x = 1; }}",
            "par_iter", ".sum"
        );
        let file = syn::parse_file(&src).unwrap();
        let mut path = std::env::temp_dir();
        path.push(format!("nd_sched_clean_{}.rs", std::process::id()));
        std::fs::write(&path, src).unwrap();
        assert!(super::detect_in_file(&file).is_empty());
        assert!(scan_file(&path).is_empty());
        std::fs::remove_file(path).unwrap();
    }

    /// Every `ALLOWANCES` entry must correspond to something the scan actually
    /// finds. Without this an allowance that outlives its code sits there
    /// looking like coverage; with it, the workspace gate reports the drift.
    ///
    /// `ALLOWANCES` IS EMPTY (see its declaration), so the loop below is a
    /// no-op today and the only live content of this test is that
    /// `collect_workspace_findings` walks the whole workspace without erroring.
    /// Said here, at the assertion site, because a green test whose loop never
    /// executes reads as coverage of the allowance contract and is not. The
    /// contract itself is proven against a FIXTURE table by the sibling below;
    /// this test goes live again the moment an entry is added.
    #[test]
    fn every_allowance_matches_a_real_hit() {
        let findings = collect_workspace_findings(&crates_dir()).expect("workspace scan");
        for (file, what, expected, _) in super::ALLOWANCES {
            let found = findings
                .iter()
                .filter(|f| f.file == *file && f.what == *what)
                .count();
            assert_eq!(
                found, *expected,
                "allowance {file} / {what} claims {expected} but the scan sees {found}"
            );
        }
    }

    /// `unallowed_against` must subtract only the exact accepted count, in both
    /// directions: exactly the accepted counts pass, and one more or one fewer
    /// each report.
    ///
    /// Runs against a FIXTURE table, not `ALLOWANCES`. The earlier version read
    /// the first real entry and `expect`ed one to exist, so it went from
    /// exercising the subtraction to failing outright the moment the real table
    /// emptied (2026-08-06, when the GMM cut removed both entries). A fixture
    /// keeps the property under test whatever the live table holds.
    #[test]
    fn allowances_are_exact_in_both_directions() {
        const FIXTURE: &[(&str, &str, usize, &str)] = &[
            ("fixture_crate/src/a.rs", "rayon::scope", 2, "fixture"),
            ("fixture_crate/src/b.rs", "sum", 1, "fixture"),
        ];
        let (file, what, ..) = *FIXTURE.first().expect("fixture table is non-empty");

        let baseline: Vec<Finding> = FIXTURE
            .iter()
            .flat_map(|(file, what, expected, _)| {
                (0..*expected).map(move |_| Finding {
                    file: (*file).to_string(),
                    what: (*what).to_string(),
                })
            })
            .collect();
        assert!(
            unallowed_against(&baseline, FIXTURE).is_empty(),
            "{:#?}",
            unallowed_against(&baseline, FIXTURE)
        );

        let mut one_more = baseline.clone();
        one_more.push(Finding {
            file: file.to_string(),
            what: what.to_string(),
        });
        assert_eq!(
            unallowed_against(&one_more, FIXTURE).len(),
            1,
            "one too many must report"
        );

        let mut one_fewer = baseline.clone();
        let position = one_fewer
            .iter()
            .position(|f| f.file == file && f.what == what)
            .expect("baseline contains the first allowance");
        one_fewer.remove(position);
        assert_eq!(
            unallowed_against(&one_fewer, FIXTURE).len(),
            1,
            "one too few must report"
        );

        let mut elsewhere = baseline;
        elsewhere.push(Finding {
            file: "some_other_crate/src/lib.rs".to_string(),
            what: what.to_string(),
        });
        assert_eq!(
            unallowed_against(&elsewhere, FIXTURE).len(),
            1,
            "an allowance must not cover the same construct in another file"
        );
    }

    /// The LIVE table must exempt only what it lists, not everything.
    ///
    /// This pins the half the fixture tests cannot: `unallowed` reads the real
    /// `ALLOWANCES`, so a finding in an unlisted file still reports.
    ///
    /// It asserts CONTAINMENT rather than an exact count. When the table was
    /// empty an exact `1` was the same statement, but a populated table also
    /// reports its own count-drift against this synthetic finding set -- each
    /// real allowance expects one occurrence and sees none here -- and that
    /// drift is the table working, not a failure. Pinning the count again
    /// would just re-encode "the table is empty".
    #[test]
    fn the_live_allowance_table_exempts_only_what_it_lists() {
        let planted = vec![Finding {
            file: "planted_crate/src/lib.rs".to_string(),
            what: "rayon::scope".to_string(),
        }];
        let problems = unallowed(&planted);
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("planted_crate/src/lib.rs")),
            "a finding in a file with no allowance must report; got {problems:?}"
        );
        for (file, what, ..) in super::ALLOWANCES {
            assert!(
                !(*file == "planted_crate/src/lib.rs" && *what == "rayon::scope"),
                "the fixture path must not collide with a real allowance"
            );
        }
    }
}
