//! Common utilities for orbital mechanics Rust crates.
//!
//! This crate provides shared numerical utilities and 6D types used across
//! `dust_estimates_rs` and `dust_splitting_rs`.
//!
//! # Modules
//!
//! - [`numerics`]: Numerical stability utilities (`safe_exp`)
//! - [`types6`]: 6D orbital state types and conversions
//! - [`macros`](module@self): exported declarative macros (`wide_consts!`,
//!   `require_ok!`, `require_err!`, `test_ok!`), resolved at the call site
//!
mod macros;
pub mod numerics;
pub mod types6;

// Re-export commonly used items at crate root
pub use numerics::{safe_exp, LOG_DBL_MAX, LOG_DBL_MIN};
pub use types6::{array_to_matrix6, slice_to_vector6, symmetrize_array, Matrix6x6, Vector6D, DIM};
