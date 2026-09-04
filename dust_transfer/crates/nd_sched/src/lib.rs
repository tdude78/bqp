//! `nd_sched` — concurrency core for the NASA Dust optimizer.
//!
//! Pure scheduling infrastructure with no kernel dependencies. It provides the
//! single-global-pool contract the optimizer/pipeline wire into later:
//!
//! - [`init_global_pool`]: ONE process-wide rayon pool (16 MiB stacks). No scoped pools —
//!   the per-batch scoped-pool pattern is the anti-pattern this crate removes.
//! - [`seed_leaf`]: deterministic 7-coordinate seed-folding — a pure function of the
//!   work coordinates, NEVER of thread id or completion order.
//! - [`flat_eval`]: index-order-preserving parallel evaluation.
//! - [`run_cells`]: bounded cross-cell backlog drivers with no batch barrier,
//!   semaphore parking, or nested Rayon pool.
//!
//! The determinism guarantee: every result vector is keyed by input index, so
//! output order is independent of the nondeterministic execution order.

mod cells;
mod flat;
mod pool;
mod seed;

// Matches the former oracle pool: deep debug force batches overflow Rayon's
// smaller default stack. Both the global pool and cell drivers use this value.
const WORKER_STACK_BYTES: usize = 16 * 1024 * 1024;

pub use cells::run_cells;
pub use flat::flat_eval;
pub use pool::{
    configured_global_pool_threads, init_global_pool, init_global_pool_authoritative, num_threads,
    GlobalPoolAuthorityError,
};
pub use seed::seed_leaf;
