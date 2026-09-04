//! Shared JD-closure tolerance floors.
//!
//! This module used to be the `PyO3` marshalling layer for the fraction-prepare
//! path: batch validators, `CTX_*`/`PREP_*`/`DIAG_*` status codes, packed-row
//! helpers and flag structs, all shaped around a Python calling convention.
//! Its own doc comments recorded that the values were "computed Python-side"
//! and that "the shim never uses them". With the Python layer gone, nothing in
//! the workspace reached any of it.
//!
//! What survives is the only part that had a real consumer: two numeric floors
//! shared with `two_phase_transfer_rs::postprocess::distribution`. That module
//! depends on this crate, so this stays their home; see
//! `docs/REFACTOR_BLOCKLIST.md`.
//!
//! The two consumers deliberately keep their own surrounding formulas — the
//! deleted `jd_closure_within_tolerance_core` here was a `black_box`-barriered
//! byte-for-byte `CPython` oracle port, while `distribution.rs`'s
//! `jd_closure_within_tolerance` is a plain internal consistency check. Only
//! the two numeric floors were ever shared, which is why removing the oracle
//! port does not touch `distribution.rs`.

/// Absolute floor, in seconds, below which a JD closure difference is treated
/// as physically indistinguishable from zero.
pub const JD_CLOSURE_PHYSICAL_FLOOR_S: f64 = 1.0e-6;

/// Multiplier applied to the local ULP of the compared Julian dates before it
/// is compared against the physical floor.
pub const JD_CLOSURE_ULP_MULTIPLIER: f64 = 8.0;
