//! Shared declarative macros.
//!
//! Everything here is `#[macro_export]`, so the macros resolve any names in
//! their bodies at the CALL site: `wide_consts!` names `wide::f64x4` without
//! this crate depending on `wide`, and the test macros' `assert!`/`return`
//! land inside the calling test function. These are the canonical homes for
//! macros that previously lived as per-crate copies; call sites migrate to
//! them crate by crate.

/// Stamp `f64x4` literal vectors as `const` ITEMS, never inline
/// `f64x4::splat(..)` expressions.
///
/// The distinction is measured, not stylistic (see `jb_rs`'s `wide_const`,
/// jb2008.rs): on aarch64-macos an inline wide literal is materialised on the
/// stack through `bl _memset_pattern16` — a libc call per constant per
/// invocation — while a `const` item is rodata and loads with one `ldr q`.
/// The lane values are byte-identical either way; only the materialisation
/// differs, so swapping a splat of a compile-time value for the matching
/// const item cannot move a bit.
///
/// The body names `wide::f64x4`, resolved at the call site: the calling crate
/// must depend on `wide`; this crate does not.
#[macro_export]
macro_rules! wide_consts {
    ($($name:ident = $value:expr),+ $(,)?) => {
        $(const $name: wide::f64x4 = wide::f64x4::new([$value; 4]);)+
    };
}

/// Unwrap an `Ok` in a test without `unwrap`/`expect`: assert with the
/// error's Debug form, then early-return so the non-panicking lints stay
/// clean. Canonical semantics from `nd_config/tests/load.rs`.
#[macro_export]
macro_rules! require_ok {
    ($result:expr) => {{
        let result = $result;
        assert!(result.is_ok(), "expected success, got {result:?}");
        let Some(value) = result.ok() else {
            return;
        };
        value
    }};
}

/// Unwrap an `Err` in a test without `unwrap_err`/`expect_err`; counterpart
/// of [`require_ok!`]. Canonical semantics from `nd_config/tests/load.rs`.
#[macro_export]
macro_rules! require_err {
    ($result:expr) => {{
        let result = $result;
        assert!(result.is_err(), "expected error, got {result:?}");
        let Some(error) = result.err() else {
            return;
        };
        error
    }};
}

/// [`require_ok!`] with the test-SETUP failure message.
///
/// Use it when the `Result` being unwrapped is fixture/scaffolding rather
/// than the behaviour under test. Canonical semantics from the `satpy_core`
/// gravity tests.
#[macro_export]
macro_rules! test_ok {
    ($result:expr) => {{
        let result = $result;
        assert!(result.is_ok(), "test setup returned {result:?}");
        let Ok(value) = result else {
            return;
        };
        value
    }};
}
