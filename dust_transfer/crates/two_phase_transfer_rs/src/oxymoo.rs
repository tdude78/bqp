//! `OxyMOO`: Rust-first optimization kernels for the dust runtime.
//!
//! The maintained multi-objective surface is [`Nsga2`]. Local scalar transfer
//! optimizers live in [`local`] and share `LocalOptimizerKind`,
//! `LocalOptimizerConfig`, `LocalScalarProblem3`, and `LocalOptimizeResult`.
//!
//! This was the `oxymoo` crate until it was absorbed here. It had exactly one
//! consumer in the workspace — this crate — so its ninety-nine public items
//! were API for nobody, and being `pub` in a library kept every one of them out
//! of rustc's dead-code analysis. `LocalOptimizerKind` and `TuneLevel` are the
//! only two names still re-exported from the crate root; everything else stops
//! here.

mod error;
pub mod local;
mod nsga2;
mod operators;
mod sort;
mod types;
mod validation;

#[cfg(test)]
mod local_nm_behavior_tests;
#[cfg(test)]
mod local_tests;
#[cfg(test)]
mod nsga2_tests;

pub use error::ArithmeticOverflow;
// The local-optimizer surface is reached through `crate::oxymoo::local::` — the
// flat re-export the crate root used to carry existed for external callers and
// has none now. `run_local_optimizer` dispatches on `LocalOptimizerKind`; the
// only callers that name the three concrete `run_*3` entry points are in
// `local` itself and in `local_tests`.
pub use local::DEFAULT_NM_MIN_ITERS;
pub use nsga2::Nsga2;
// Reached only by `nsga2_tests` and by `benches/oxymoo_nsga2_bench.rs`: the
// solver drives the sort through `Nsga2`, never directly.
#[cfg(any(test, feature = "bench-internal"))]
pub use sort::{crowding_distance, fast_nondominated_sort};
pub use types::{
    Nsga2Config, Nsga2Result, PopulationSnapshot, Problem, SortConfig, VariableKind, VariableSpec,
};
