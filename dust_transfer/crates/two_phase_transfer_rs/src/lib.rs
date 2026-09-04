#![allow(non_snake_case)]
//! Two-phase orbital transfer optimization in Rust.
//!
//! This crate provides optimized trajectory planning for satellite constellation
//! deployment using a two-phase strategy.
//!
//! ## Internal benchmark surface
//!
//! The compact postprocess batch entrypoint is intentionally unavailable to
//! runtime consumers. Criterion harnesses enable `bench-internal` explicitly.
//!
//! ```compile_fail
//! use two_phase_transfer_rs::batch_postprocess_compact_candidates;
//! ```
//!
//! # Physical Defensibility References
//!
//! ## Lambert Problem (Trajectory Design)
//!
//! Reference: Izzo, D. (2014), "Revisiting Lambert's Problem,"
//! [`arXiv:1403.2705v2`](<https://arxiv.org/pdf/1403.2705v2>), Algorithm 1-2.
//!
//! The Lambert solver uses Lancaster-Blanchard formulation with Householder
//! iterations, achieving 2-iteration convergence for single-revolution cases.
//!
//! ## Momentum Transfer Physics
//!
//! Reference: Vallado (2013) Chapter 9: Orbit Determination, STM propagation.
//! Also: NASA DART Mission results (`β = 3.6` momentum enhancement factor).
//!
//! We use conservative momentum transfer without beta enhancement:
//! - `Δv = (m_dust × v_rel) / (m_target + m_dust)`
//! - For `m_dust << m_target`: `Δv ≈ (m_dust × v_rel) / m_target`
//!
//! ## NASA CARA Standards
//!
//! Reference: NASA CARA (Conjunction Assessment Risk Analysis)
//! [publications](<https://www.nasa.gov/cara/cara-publications/>).
//!
//! Physics parameters aligned with operational standards:
//! - `min_miss_distance`: 1.0 km (NASA CARA screening threshold)
//! - `hit_probability`: 0.90 (Phase C confidence target on captured mass)
//! - Covariance minimum eigenvalue: `1e-6 km²` (1 m² operational floor)

// mimalloc as this cdylib's global allocator: REMOVED 2026-06-05 (free-threading
// SIGSEGV). It was a Linux-only perf experiment; the only cited measurement
// (macOS `make perf-gate`) was +1.8% — within noise, no gain — and the Linux
// gain was never demonstrated. Under no-GIL CPython 3.14t the ThreadPoolExecutor
// churns worker threads and mimalloc's per-thread heap init races: SIGSEGV
// *inside the allocator* (mi_thread_init / mi_heap_main / _mi_subproc) on a
// routine Vec::with_capacity from a worker thread — e.g. reset_keplerian ->
// precompute_session_satellite_data (py_api.rs:853). rnsde (constant rebinds ->
// heavy per-worker allocation) crashed 16/16; nsga2 never rebinds, so it never
// hit it. glibc ptmalloc2 is thread-safe, so two_phase_transfer_rs.so now runs
// on the system allocator. The unused workspace dependency is removed.
//
// NOTE: lightyear_odeint_rs still has an opt-in `mimalloc-global` feature — if
// that .so ever shows the same crash under no-GIL, unregister it the same way.

pub mod batch_eci;
pub mod evaluate;
pub(crate) mod geometry;
pub mod hf_acceptance;
pub mod intercept;
// Absorbed from the `lambert_rs` crate, whose only production consumer was this
// one and whose only other dependent was a `satpy_core` bench. Private, so its
// 111 published items are inside rustc's dead-code analysis; nothing outside
// this crate ever named them.
mod lambert;
pub(crate) mod lambert_backend;
// Absorbed from the `oxymoo` crate, whose only workspace consumer was this
// one. Private, so its kernels are inside rustc's dead-code analysis; the
// two local-optimizer names below are the whole re-exported surface.
mod oxymoo;
mod postprocess;
pub mod py_config;
pub(crate) mod scratch;
pub mod solve;
mod solve_policy;
pub mod types;
pub mod verify;

// Re-export commonly used types.
pub use batch_eci::constellation_solve_batch_eci_precomputed;
pub use batch_eci::constellation_solve_population_batch_eci_precomputed;
// The local-optimizer knobs `TransferSearchConfig` carries. Downstream crates
// and bench harnesses name them through `two_phase_transfer_rs::*`; the rest of
// the absorbed optimizer stays private. The duplicate-compilation hazard this
// re-export used to defend against (a bench doing `use oxymoo::local::*` and
// getting a second `oxymoo` instance, cargo issue #6313) no longer exists:
// there is no second crate to resolve.
pub use crate::oxymoo::local::{LocalOptimizerKind, TuneLevel};
// The Lambert solver surface the three absorbed Criterion harnesses drive --
// `benches/lambert_{solver,batch_tof,cpu_baseline}_bench.rs`. Gated the same way
// the NSGA-II kernel below is: never in a runtime build. Production reaches all
// of these through `crate::lambert::`, not through the crate root.
#[cfg(feature = "bench-internal")]
pub use crate::lambert::{
    compute_lambert_geometry, izzo2015_batch_m_prograde, izzo2015_batch_tof,
    izzo2015_batch_tof_variable_r2, izzo2015_batch_tof_variable_r2_with_scratch,
    izzo2015_best_solution, izzo2015_impl, izzo2015_impl_with_geom_fast,
    solve_lambert_batch_tof_variable_r2_branch_best_with_scratch, LambertGeometry,
    VariableR2LambertScratch,
};
// The NSGA-II kernel surface `benches/oxymoo_nsga2_bench.rs` drives. Gated the
// same way the compact postprocess entry below is: never in a runtime build.
#[cfg(feature = "bench-internal")]
pub use crate::oxymoo::{
    crowding_distance, fast_nondominated_sort, Nsga2, Nsga2Config, Problem, SortConfig,
    VariableKind, VariableSpec,
};
// Postprocess native entries reachable by the nd_pipeline MF physics layer.
#[cfg(any(test, feature = "bench-internal"))]
pub use postprocess::{
    batch_postprocess_compact_candidates, CompactBatchPostprocessError,
    CompactBatchPostprocessInputs, CompactBatchPostprocessOutputs, CompactBatchTargetPhysics,
};
pub use postprocess::{
    canonical_strict_hf_gravity_identity, natural_state_position_residual_km,
    natural_state_velocity_residual_km_s, propagate_components_ukf_full_batch,
    AuthoritativeReleaseDistribution, NaturalConjunctionEnclosure, NaturalConjunctionFatalError,
    NaturalConjunctionInfeasible, NaturalConjunctionInputError, NaturalConjunctionOutcome,
    NaturalConjunctionScanAnchor, NaturalConjunctionWitnessResidual, NaturalObjectIdentity,
    NaturalObjectInput, PostprocessControl, PostprocessControlStatus,
    PostprocessDistributionStatus, PostprocessDustDistribution, PostprocessSessionError,
    StrictHfContextStatus, StrictHfForceAuthority, StrictHfGravityIdentity,
    TransferPostprocessSessionCore, UkfPropagationFailure, VerifiedNaturalConjunction,
    NATURAL_DENSE_ARC_AUTHORITY_CEILING_KM,
};
#[cfg(feature = "solver-qualification")]
pub use postprocess::{
    QualificationArmIdentity, QualificationDistributionRequest, QualificationLegFailureCode,
    QualificationLegInput, QualificationLegOutcome, QualificationLegPath, QualificationLegRecord,
    QualificationLegTrace, QualificationReleaseControlRequest, QualificationTraceError,
    QualificationTraceIdentity, MAX_QUALIFICATION_LEG_RECORDS,
};
pub use py_config::{PhysicsConfig, PhysicsConfigError, PostprocessConfig};
pub use solve::{solve_plan, J2ClosureSettings};
pub use types::{
    CompactTransferCandidate, ConstellationTransferCandidate, ConstellationTransferFront,
    EciBasicOrbit, ExecutionPolicy, OxyMooPolicy, PlanContext, PlanResult, SamplingMode,
    TransferFront, TransferLocalOptimizerChoice, TransferLocalOptimizerConfig, TransferRequest,
    INVALID_COST, MIN_TOF,
};
pub use verify::replay_transfer_controls;
