//! Honesty note: every assertion here is a NEGATIVE source-text check
//! (`!source.contains("Name")`). If a listed item is reintroduced under a
//! different name, or a NEW module file is added that `include_str!` does not
//! read, these asserts stay green vacuously — they pin the absence of these
//! exact spellings in these exact files, not the absence of the capability.

fn declares_a_submodule(line: &str) -> bool {
    let mut rest = line.trim_start();
    if let Some(after_pub) = rest.strip_prefix("pub") {
        let after_pub = after_pub
            .strip_prefix('(')
            .and_then(|restriction| restriction.split_once(')'))
            .map_or(after_pub, |(_, tail)| tail);
        if after_pub.starts_with(char::is_whitespace) {
            rest = after_pub.trim_start();
        }
    }
    let Some(after_mod) = rest.strip_prefix("mod") else {
        return false;
    };
    if !after_mod.starts_with(char::is_whitespace) {
        return false;
    }
    let name: String = after_mod
        .trim_start()
        .chars()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    !name.is_empty() && name != "tests"
}

/// The visibility prefixes this must see. `pub mod` was invisible to the first
/// version of this helper, so a PUBLIC submodule -- the one most likely to
/// carry new surface -- evaded the count that exists to notice one.
#[test]
fn submodule_declarations_are_counted_at_every_visibility() {
    for declaration in [
        "mod observer;",
        "pub mod observer;",
        "pub(crate) mod observer;",
        "pub(super) mod observer;",
        "pub(self) mod observer;",
        "pub(in crate::physics) mod observer;",
        "pub\tmod observer;",
        "pub  mod observer;",
        "mod\tobserver;",
        "    pub mod observer;",
        "mod tests_helper;",
    ] {
        assert!(
            declares_a_submodule(declaration),
            "missed a submodule declaration: {declaration}"
        );
    }
    for other in [
        "mod tests {",
        "#[cfg(test)]",
        "use std::mod_like_name;",
        "// mod observer;",
        "let modulus = 3;",
        "module_path!();",
        "pubmod observer;",
    ] {
        assert!(
            !declares_a_submodule(other),
            "counted a non-declaration: {other}"
        );
    }
}

#[test]
fn scheduler_surface_excludes_dormant_compatibility_apis() {
    let cells = include_str!("../src/cells.rs");
    let flat = include_str!("../src/flat.rs");
    let pool = include_str!("../src/pool.rs");
    let seed = include_str!("../src/seed.rs");
    let root = include_str!("../src/lib.rs");

    // The five files above are hand-copied, and a sixth module would be
    // silently unscanned -- the negative checks below would then stay green
    // over a file nobody read. Count the crate root's own `mod` declarations.
    let declared = root
        .lines()
        .map(declares_a_submodule)
        .filter(|declared| *declared)
        .count();
    assert_eq!(
        declared + 1,
        5,
        "lean-surface scan reads 5 files but the crate declares {declared} modules"
    );

    for removed in [
        "WorkUnit",
        "VarianceClass",
        "should_outer_flatten",
        "batch_par_min_len",
    ] {
        assert!(!flat.contains(removed), "flat API still contains {removed}");
    }
    for removed in ["flat_eval_balanced", "balanced_lane_order"] {
        for (name, source) in [("flat", flat), ("lib", root)] {
            assert!(
                !source.contains(removed),
                "{name} API still contains {removed}"
            );
        }
    }
    // The `instrument` module (`BoundsTracker`/`BoundHit`) is itself gone — no
    // caller ever read it. These names were pinned absent from that module; now
    // they must stay absent from EVERY module, which is the stronger check.
    for removed in [
        "PhaseSample",
        "WorkerOccupancy",
        "time_phase",
        "tail_ratio",
        "worker_occupancy",
        "global_bounds",
        "BoundsTracker",
        "BoundHit",
    ] {
        for (name, source) in [
            ("cells", cells),
            ("flat", flat),
            ("pool", pool),
            ("seed", seed),
            ("lib", root),
        ] {
            assert!(
                !source.contains(removed),
                "{name} API still contains {removed}"
            );
        }
    }
    assert!(!pool.contains("pub fn install"));
    assert!(!seed.contains("leaf_rng"));

    for module in ["cells", "flat", "instrument", "pool", "seed"] {
        assert!(!root.contains(&format!("pub mod {module};")));
    }
}
