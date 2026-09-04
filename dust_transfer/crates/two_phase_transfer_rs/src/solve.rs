//! High-level two-phase transfer optimization.
//!
//! Batch ECI entries own the production candidate-front path: cheap orbital
//! heuristics rank deployer-target pairs, then each assessed pair uses
//! `OxyMOO` `NSGA-II` on the two transfer objectives:
//! minimize total delta-V and minimize time to intercept per intercept
//! relative velocity. Deterministic geometry seeds are still evaluated beside
//! the `OxyMOO` population so obvious good transfers are not missed by random
//! initialization.
//!
//! `solve_plan()` remains the scalar single-pair helper used by direct
//! single-pair callers and local-optimizer tests.

use crate::oxymoo::local::{
    run_local_optimizer, LocalOptimizeResult, LocalOptimizerConfig, LocalOptimizerKind,
    LocalScalarProblem3, TuneLevel,
};
use crate::oxymoo::{Nsga2, Nsga2Config, Nsga2Result, Problem, VariableKind, VariableSpec};
#[cfg(test)]
use crate::types::PairProxyModel;
use rustc_hash::{FxHashMap, FxHashSet};
#[cfg(test)]
use satpy_core::norm3;
#[cfg(feature = "bench-internal")]
use satpy_core::MU;
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::ops::Range;
#[cfg(any(test, feature = "bench-internal"))]
use std::time::Instant;

/// Wall-clock stage timer for the verified-superset instrumentation.
///
/// Stage timings are diagnostics: production consumers read only the count
/// fields of the stage metrics, so outside tests and `bench-internal` builds
/// the clock reads are compiled out and every `*_s` metric stays `0.0`. The
/// zero is exact -- no float work happens on the gated path -- so gating
/// cannot reorder or perturb any physics arithmetic.
#[derive(Clone, Copy, Debug)]
struct StageTimer {
    #[cfg(any(test, feature = "bench-internal"))]
    start: Instant,
}

impl StageTimer {
    #[cfg_attr(
        not(any(test, feature = "bench-internal")),
        allow(clippy::missing_const_for_fn)
    )]
    #[inline]
    fn start() -> Self {
        Self {
            #[cfg(any(test, feature = "bench-internal"))]
            start: Instant::now(),
        }
    }

    /// Elapsed wall seconds, or exactly `0.0` when timing is compiled out.
    #[cfg_attr(
        not(any(test, feature = "bench-internal")),
        allow(clippy::missing_const_for_fn, clippy::unused_self)
    )]
    #[inline]
    fn elapsed_s(self) -> f64 {
        #[cfg(any(test, feature = "bench-internal"))]
        {
            self.start.elapsed().as_secs_f64()
        }
        #[cfg(not(any(test, feature = "bench-internal")))]
        {
            0.0
        }
    }
}

use crate::evaluate::restore_evaluation_diagnostics;
use crate::evaluate::{
    compute_dep_period, eci_orbit_summary, enter_evaluation_diagnostic_region, evaluate_plan,
    evaluate_plan_branches_with_scratch, evaluate_plan_from_phase_with_lambert_scratch,
    evaluation_diagnostic_snapshot, leave_evaluation_diagnostic_region,
    propagate_candidate_state_at_epoch, record_phase_state_cache_lookup,
    EvaluationArithmeticOverflow, EvaluationDiagnosticCounters,
};
use crate::geometry::{combined_transfer_initial_guess, compute_time_to_nodes};
// heuristic_v2 removed during V3 demolition; simple grid search inlined below
#[cfg(test)]
use crate::types::BodyRole;
use crate::types::{
    BodyForceConfig, ConstellationTransferCandidate, ConstellationTransferFront,
    DeltaVAnchorPolicy, ExecutionPolicy, InvalidTargetPropagationAuthorityCode, OxyMooPolicy,
    PairPlanContextInputs, PlanContext, PlanContextTemplate, PlanResult, PolishScopePolicy,
    PsoPreset, SamplingMode, SearchDepthPolicy, TargetPropagationAuthority, TransferComplexity,
    TransferFront, TransferLocalOptimizerChoice, TransferLocalOptimizerConfig,
    VerifiedSupersetStageMetrics, WarmStartData, INVALID_COST,
};
#[cfg(test)]
use crate::verify::verify_transfer_result;

use rayon::prelude::*;

// NOTE: The global EXCELLENT_FOUND flag was removed (R-17). It was never written to
// and always returned false. The NmEarlyExitObserver in optimizer.rs is a no-op.

/// Lexicographic comparator chain, one key per line.
///
/// `lex_cmp!(left, right; asc (key), desc (key), int (key), int_desc (key))`
/// expands to exactly the hand-written
/// `left.key.partial_cmp(&right.key).unwrap_or(Ordering::Equal).then_with(..)`
/// chain (left-leaning, keys evaluated lazily in order), so sort results are
/// token-for-token unchanged. `asc`/`desc` are the NaN-tolerant float keys --
/// the `unwrap_or(Equal)` policy lives here and nowhere else -- and
/// `int`/`int_desc` use total `Ord::cmp`. Each key is a parenthesized field
/// path (method calls allowed, e.g. `(rel_v_proxy.abs())`) applied to both
/// bindings.
///
/// Declared before the `mod` items below so its textual scope reaches the
/// submodule comparators (`front.rs`, `pair_proxy.rs`) without a re-export.
macro_rules! lex_cmp {
    ($left:ident, $right:ident; $dir:ident $key:tt $(, $rest_dir:ident $rest_key:tt)* $(,)?) => {
        lex_cmp!(
            @fold $left, $right,
            (lex_cmp!(@key $dir $left, $right, $key));
            $($rest_dir $rest_key),*
        )
    };
    (@fold $left:ident, $right:ident, ($acc:expr);) => { $acc };
    (@fold $left:ident, $right:ident, ($acc:expr);
        $dir:ident $key:tt $(, $rest_dir:ident $rest_key:tt)*) => {
        lex_cmp!(
            @fold $left, $right,
            ($acc.then_with(|| lex_cmp!(@key $dir $left, $right, $key)));
            $($rest_dir $rest_key),*
        )
    };
    (@key asc $left:ident, $right:ident, ($($key:tt)+)) => {
        $left.$($key)+
            .partial_cmp(&$right.$($key)+)
            .unwrap_or(::core::cmp::Ordering::Equal)
    };
    (@key desc $left:ident, $right:ident, ($($key:tt)+)) => {
        $right.$($key)+
            .partial_cmp(&$left.$($key)+)
            .unwrap_or(::core::cmp::Ordering::Equal)
    };
    (@key int $left:ident, $right:ident, ($($key:tt)+)) => {
        $left.$($key)+.cmp(&$right.$($key)+)
    };
    (@key int_desc $left:ident, $right:ident, ($($key:tt)+)) => {
        $right.$($key)+.cmp(&$left.$($key)+)
    };
}

/// Isolated diagnostic/work-counter region for one parallel worker closure:
/// zero the evaluation-diagnostic region, snapshot the work counters, run
/// `body`, capture this unit's exact counter deltas, and restore both
/// thread-local baselines on EVERY exit path. Rayon may run the closure on the
/// caller thread; restoring unconditionally is what keeps a caller-thread
/// worker from being counted twice once the serial reduction folds the
/// captured deltas back in -- historically, restoring only on error left each
/// worker's totals growing for the life of the pool. The zeroed region also
/// means the f64 residual/timer sums come back as the unit's own exact sums
/// instead of a difference against a growing baseline. Pure control flow and
/// thread-local bookkeeping: no physics arithmetic.
fn with_isolated_diag_region<T>(
    body: impl FnOnce() -> Result<T, InvalidTargetPropagationAuthorityCode>,
) -> Result<
    (T, EvaluationDiagnosticCounters, WorkCountCounters),
    InvalidTargetPropagationAuthorityCode,
> {
    let diag_outer = enter_evaluation_diagnostic_region();
    let work_before = work_count_snapshot();
    let outcome = (|| {
        let value = body()?;
        let diag_delta = evaluation_diagnostic_snapshot();
        let work_delta = work_count_snapshot().delta_since(work_before)?;
        Ok((value, diag_delta, work_delta))
    })();
    restore_work_count_snapshot(work_before);
    restore_evaluation_diagnostics(&diag_outer);
    outcome
}

// ============================================================================
// Optimization Toggles for A/B Benchmarking
// ============================================================================

const COARSE_EARLY_STOP_MIN_EVALS: usize = 16;
const COARSE_EARLY_STOP_BEST_COST_KMS: f64 = 0.10;
const COARSE_EARLY_STOP_WORSE_MARGIN_KMS: f64 = 0.08;
const COARSE_EARLY_STOP_MIN_FINE_COUNT: usize = 6;

#[inline]
fn branch_expansion_capacity(
    source_count: usize,
    max_revs: i32,
) -> Result<usize, InvalidTargetPropagationAuthorityCode> {
    let rev_count = if max_revs <= 0 {
        0
    } else {
        usize::try_from(max_revs)
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?
    };
    let paths_per_source = rev_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(2))
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    source_count
        .checked_mul(paths_per_source)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
}

#[cfg(test)]
fn pair_cache_reset_enabled_from_value(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on" | "pair")
    )
}

const fn oxymoo_source_materialization_enabled(front_output_mode: FrontOutputMode) -> bool {
    matches!(front_output_mode, FrontOutputMode::VerifiedSuperset)
}

mod anchor;
#[cfg(any(test, feature = "bench-internal"))]
use anchor::select_delta_v_anchor_starts;
use anchor::{local_optimizer_failure_code, push_delta_v_anchor_candidates};
#[cfg(feature = "bench-internal")]
use anchor::{map_bench_local_optimizer_result, AnchorRunSettings};
#[cfg(test)]
use anchor::{run_delta_v_anchor_candidates, should_use_anchor_parallel};
mod branch_expansion;
use branch_expansion::expand_lambert_branch_candidates_for_superset;
#[cfg(test)]
use branch_expansion::{
    branch_expansion_sources_unique_by_repaired_decision,
    branch_expansion_sources_unique_by_repaired_decision_indexed,
    expand_lambert_branch_candidates_parallel, expand_lambert_branch_candidates_serial,
    should_use_branch_expansion_parallel, BRANCH_EXPANSION_PARALLEL_MIN_SOURCES,
};
mod context;
mod front;

/// Shadow-measurement entry for the front-reduction lane: run the full
/// pairwise-dominance finalizer over a candidate set. Production output is
/// unaffected — the verified-superset finalizer remains the wired path; this
/// exists so harnesses can price the cross-pair dominance delta on real
/// fronts without duplicating the dominance rule.
mod moo;
pub(crate) mod pair_proxy;
mod polish;
#[cfg(test)]
use polish::{
    final_candidate_polish_bounds, polish_candidates_parallel, should_use_polish_parallel,
    PolishAction, POLISH_PARALLEL_MIN_CANDIDATES,
};
use polish::{
    polish_transfer_candidate_delta_v, polish_transfer_candidates_delta_v,
    polish_transfer_candidates_delta_v_with_pre_polish_snapshot, PolishScopeStats,
};
mod seed_policy;
#[cfg(test)]
mod tests;
use crate::solve_policy::should_parallelize_selected_pairs;
#[cfg(all(feature = "bench-internal", not(test)))]
use context::build_cached_plan_context;
#[cfg(test)]
use context::build_cached_plan_context;
#[cfg(test)]
use context::target_plane_from_equinoctial;
// `EciBasicOrbit` lost its last production use in this module when the
// `objective_hint` deletion removed `EventPlan::targets_orbit_cached`. Both the
// test module and the `bench-internal` policy harness still build fixtures from
// it through `use super::*`, so the re-export follows those two cfgs exactly —
// narrowing it to `cfg(test)` alone leaves the feature build red while the
// default build stays green.
#[cfg(any(test, feature = "bench-internal"))]
use crate::types::EciBasicOrbit;
use front::{
    append_constellation_transfer_candidates, finalize_constellation_transfer_superset,
    finalize_verified_front, finalize_verified_superset, finalize_verified_superset_with_metrics,
    push_constellation_transfer_candidates, transfer_candidate_is_objective_finite,
    verified_front_from_plan, ConstellationFrontArchive,
};
#[cfg(test)]
use front::{
    constellation_candidate_dominates, filter_nondominated_transfer_candidates,
    finalize_constellation_transfer_front, finalize_verified_candidate,
    transfer_candidate_dominates, verification_tolerance_for_solve,
};
#[cfg(feature = "bench-internal")]
pub use moo::TransferMooBenchPolicy;
use moo::{
    anchor_parallel_count_snapshot, merge_work_counts, normalized_excess,
    oxymoo_batch_class_snapshot, record_anchor_parallel_runs, record_work_count,
    repair_transfer_decision, repaired_transfer_decision, reset_anchor_parallel_count,
    reset_oxymoo_batch_class, restore_work_count_snapshot, transfer_decision_key,
    transfer_moo_config_with_policy, transfer_moo_constraint_violation, transfer_moo_dv_reference,
    transfer_moo_population_generations, work_count_snapshot, TransferDecisionKey,
    TransferMooPlanCache, TransferMooPolicy, TransferMooProblem, WorkCountCounters,
    TRANSFER_MOO_PLAN_CACHE_MAX_ENTRIES,
};
#[cfg(test)]
use moo::{
    transfer_moo_config_with_initial_decisions, TransferMooEvalCache, TransferMooEvalCacheEntry,
    OXYMOO_BATCH_PARALLEL_MIN_ROWS,
};
#[cfg(test)]
use pair_proxy::pair_verification_limit;
#[cfg(test)]
use pair_proxy::select_pair_proxy_candidates;
use pair_proxy::{
    compute_sat_orbit_props, screen_pair_proxy_candidates, select_pair_proxy_candidates_reuse,
    target_orbit_invariants, PairProxyCandidate, PairProxyScratch, PairProxyTargetInput,
    SatOrbitProps,
};
#[cfg(test)]
use pair_proxy::{
    kepler_from_eci, make_pair_proxy_candidate, node_wait_min_from_eci, node_wait_proxy,
    node_wait_proxy_from_min_times, pair_proxy_time_per_relative_velocity, pair_time_proxy_and_cv,
    pair_x_hint, pair_x_hint_from_kepler, retain_pair_proxy_candidate,
};
#[cfg(test)]
use seed_policy::sort_grid_seed_candidates_by_hint;
use seed_policy::{
    build_single_pair_seeds, seed_is_duplicate, should_stop_coarse_stage, warm_start_matches_pair,
    SolveLocalWorkCache, SolverSeed,
};

const SINGLE_PAIR_LOWER_BOUNDS: [f64; 3] = [0.0, 0.5, 0.0];
const SINGLE_PAIR_UPPER_BOUNDS: [f64; 3] = [0.95, 1.5, 0.95];
const SINGLE_PAIR_TIME_PTS: &[f64] = &[0.08, 0.22, 0.40, 0.60];
const SINGLE_PAIR_PHASE_PTS: &[f64] = &[0.94, 1.00, 1.03, 1.06];
const SINGLE_PAIR_WAIT_PTS: &[f64] = &[0.05, 0.20, 0.40, 0.60];

/// Surviving TIME x PHASE x WAIT grid points after the physically-excluded
/// corner cut: 4x4x4 = 64 minus the 3 (time, wait) pairs x 4 phases with
/// `time + wait > 0.98`.
const DETERMINISTIC_GRID_POINT_COUNT: usize = 52;

/// Compile-time TIME x PHASE x WAIT deterministic decision grid: the nested
/// enumeration order of the three point slices above, skipping the physically
/// excluded corner (`time + wait > 0.98`).
///
/// Built FROM the production slices, so the table cannot shadow-drift from
/// them, and const f64 `+`/`>` evaluate with the same IEEE semantics as the
/// retired per-call runtime loops, so the surviving set and its order are
/// exactly theirs. The `filled` assert turns any source-slice edit that
/// changes the surviving count into a BUILD error.
#[expect(
    clippy::indexing_slicing,
    reason = "const context: the loop counters are bounded by the source slice \
              lengths, the write index is asserted to land exactly on \
              DETERMINISTIC_GRID_POINT_COUNT, and any violation is a \
              compile-time panic, i.e. a BUILD error"
)]
const DETERMINISTIC_GRID_POINTS: [[f64; 3]; DETERMINISTIC_GRID_POINT_COUNT] = {
    let mut points = [[0.0_f64; 3]; DETERMINISTIC_GRID_POINT_COUNT];
    let mut filled = 0;
    let mut time_index = 0;
    while time_index < SINGLE_PAIR_TIME_PTS.len() {
        let time2phase_ratio = SINGLE_PAIR_TIME_PTS[time_index];
        let mut phase_index = 0;
        while phase_index < SINGLE_PAIR_PHASE_PTS.len() {
            let phase_sma_ratio = SINGLE_PAIR_PHASE_PTS[phase_index];
            let mut wait_index = 0;
            while wait_index < SINGLE_PAIR_WAIT_PTS.len() {
                let waittime_ratio = SINGLE_PAIR_WAIT_PTS[wait_index];
                if time2phase_ratio + waittime_ratio > 0.98 {
                    // physically-excluded corner
                } else {
                    points[filled] = [time2phase_ratio, phase_sma_ratio, waittime_ratio];
                    filled += 1;
                }
                wait_index += 1;
            }
            phase_index += 1;
        }
        time_index += 1;
    }
    assert!(
        filled == DETERMINISTIC_GRID_POINT_COUNT,
        "deterministic grid point count drifted from its source slices"
    );
    points
};
const TRANSFER_MOO_OBJECTIVES: usize = 2;
const TRANSFER_MOO_VARIABLES: [VariableSpec; 3] = {
    let [time_lower, phase_lower, wait_lower] = SINGLE_PAIR_LOWER_BOUNDS;
    let [time_upper, phase_upper, wait_upper] = SINGLE_PAIR_UPPER_BOUNDS;
    [
        VariableSpec {
            lower: time_lower,
            upper: time_upper,
            kind: VariableKind::Continuous,
        },
        VariableSpec {
            lower: phase_lower,
            upper: phase_upper,
            kind: VariableKind::Continuous,
        },
        VariableSpec {
            lower: wait_lower,
            upper: wait_upper,
            kind: VariableKind::Continuous,
        },
    ]
};
const FINAL_CANDIDATE_POLISH_RADIUS: [f64; 3] = [0.015, 0.02, 0.015];
/// Exact Nelder-Mead iteration count for the final candidate polish.
///
/// This was `..._MAX_ITERS = 12`, and 12 was never the number of iterations
/// that ran. `nelder_mead_impl` scales the request by `TuneLevel::Default`'s
/// `iters_factor` of 0.3 and then floors it at `DEFAULT_NM_MIN_ITERS` = 10, so
/// 12 -> max(3.6, 10) -> 10, and so did every request from 0 to 33. That is
/// why sweeping the old constant down to 0 left all three `physics_3event`
/// events bit-identical, and why 12 -> 40 -> 120 (which reach 12 and 36) also
/// did: the knob was inert over its whole plausible range.
///
/// The polish call now pins `min_iters` as well, so the value here is the
/// iteration count. 10 preserves today's behaviour exactly.
///
/// 10 is the floor, measured. Swept downward with the count now real, selected
/// `total_dv` on the three `physics_3event` events, and the three-event stage-1
/// solve's wall time beside it (min over 3 blocks x 20 repeats, macOS release,
/// same host and session):
///
/// | iters | event 0 | event 1 | event 2 | wall | vs 10 |
/// |---:|---|---|---|---|---|
/// | 10 | 8.7605e-3 | 3.3337e-2 | 1.2587e-2 | 0.1162 s | 1.00x |
/// | 9  | 9.4073e-3 | 3.3650e-2 | 1.2843e-2 | 0.1131 s | 0.97x |
/// | 8  | 1.3921e-2 | 3.2534e-2 | 1.3554e-2 | 0.1100 s | 0.95x |
/// | 6  | 1.3921e-2 | 3.2534e-2 | 1.3554e-2 | 0.1013 s | 0.87x |
/// | 4  | 1.3259e-2 | 3.2534e-2 | 1.3554e-2 | 0.0937 s | 0.81x |
/// | 2  | 2.7685e-2 | 3.4174e-2 | 1.4188e-2 | 0.0863 s | 0.74x |
/// | 1  | 2.7685e-2 | 3.4174e-2 | 1.5169e-2 | 0.0817 s | 0.70x |
///
/// The very first step down, 10 -> 9, moves all three events and makes all
/// three worse (+7.4% / +0.9% / +2.0%) to buy 2.7% of stage-1 wall time. So
/// there is no free reduction here: the polish is insensitive ABOVE 10 and
/// sharply sensitive BELOW it, and 10 sits on the knee. Removing the polish
/// entirely costs 0.0486 s of the 0.1162 s (42%) and takes the three events to
/// 1.7506e-2 / 3.4174e-2 / 1.2908e-2 — a 2.0x regression on event 0.
///
/// Most polish calls run the cap out rather than converging: sampling every
/// call on the three events, the modal budget is 21-25 objective evaluations
/// with `converged = false`, i.e. the simplex sd tolerance is rarely reached.
const FINAL_CANDIDATE_POLISH_ITERS: usize = 10;
/// Acceptance slack on the polish and release-epoch-scan improvement tests.
///
/// NOT an early exit, and not a throughput knob: every use of it is checked
/// AFTER the candidate has already been evaluated, so widening it cannot skip
/// work. Measured, same harness as [`FINAL_CANDIDATE_POLISH_ITERS`]: raising it
/// to 1e-7 and 1e-5 leaves all three events' fronts bit-identical (front hash
/// included) at 0.1153-0.1159 s against 0.1162 s, i.e. flat. At 1e-3 the
/// selected `total_dv` is still bit-identical on all three events but the
/// fronts lose rows (221/268/279 -> 218/263/272) because the release-epoch scan
/// discards sub-1-m/s improvements, and the wall time is still flat. Pure loss.
/// (Measured 2026-07-27, pre-`nd-epsilon-membership`: the absolute front row
/// counts predate the 2026-08-13 reseal; the knee/flatness conclusion is what
/// this constant rests on, not those counts.)
const FINAL_CANDIDATE_POLISH_DV_EPS: f64 = 1e-9;
// PolishScopePolicy::NdEpsilon margins: a candidate skips the final NM polish
// only when another candidate beats it by more than these bounds on BOTH
// objectives at no-worse constraint violation. This is a measured
// quality/throughput TRADE, not a safety proof: polish rescues far larger
// than the 0.05 km/s margin exist (measured up to ~1.75 km/s on the gate
// canary via verified_superset_polish_dv_improvement_max_km_s), so skipped
// candidates CAN lose front-relevant refinement. The policy was promoted on
// multi-seed HV evidence (nsga2 x8: 1.50x at HV +2.7%), and the
// degenerate-front safety net below re-polishes when a front comes back
// empty/single-row with skips present.
//
// Re-tune machinery (design item d): the dv margin is overridable via the
// `nd_epsilon_dv_mps<N>` config token (PolishScopePolicy::NdEpsilonTuned);
// see that variant's doc in types.rs for the telemetry-driven campaign plan
// (polish_dv_improvement_max_km_s distribution -> p10-p25 -> HV A/B ->
// advisor sign-off). The constants below stay the `nd_epsilon` defaults.
pub(super) const POLISH_SCOPE_ND_EPS_DV_KM_S: f64 = 0.05;
pub(super) const POLISH_SCOPE_ND_EPS_TIME_FRAC: f64 = 0.05;
const POLISH_SCOPE_CV_TOL: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct J2ClosureSettings {
    pub max_iterations: usize,
    pub endpoint_target_km: f64,
    pub correction_step_gain: f64,
}

pub(crate) const DEFAULT_J2_MAX_ITERATIONS: usize = 8;
pub(crate) const DEFAULT_J2_ENDPOINT_TARGET_KM: f64 = 0.01;
pub(crate) const DEFAULT_J2_CORRECTION_STEP_GAIN: f64 = 0.7;

impl Default for J2ClosureSettings {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_J2_MAX_ITERATIONS,
            endpoint_target_km: DEFAULT_J2_ENDPOINT_TARGET_KM,
            correction_step_gain: DEFAULT_J2_CORRECTION_STEP_GAIN,
        }
    }
}

/// Adds the shared count fields (same names on both structs) with
/// overflow-checked addition. One list instead of forty six-line blocks; a
/// field named here but absent on either struct is a compile error.
macro_rules! project_diag_counts {
    ($merged:ident, $counters:ident; $($field:ident),+ $(,)?) => {
        $( checked_stage_metric_count_add(&mut $merged.$field, $counters.$field)?; )+
    };
}

#[inline]
fn add_evaluation_diagnostics_to_stage_metrics(
    metrics: &mut VerifiedSupersetStageMetrics,
    counters: &EvaluationDiagnosticCounters,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let mut merged = *metrics;
    project_diag_counts!(merged, counters;
        lambert_batch_call_count,
        lambert_batch_row_count,
        lambert_scalar_tof_count,
        lambert_branch_attempt_count,
        lambert_branch_valid_count,
        lambert_branch_rev0_count,
        lambert_branch_rev_gt0_count,
        lambert_branch_low_path_count,
        lambert_branch_high_path_count,
        lambert_branch_prograde_count,
        lambert_branch_retrograde_count,
        lambert_max_revs_gt0_call_count,
        near_pi_plane_eval_count,
        lambert_branch_selection_call_count,
        target_j2_batch_state_count,
        target_j2_simd4_chunk_count,
        target_j2_scalar_state_count,
        j2_propagate_state_count,
        phase_state_cache_hit_count,
        phase_state_cache_miss_count,
        j2_correction_gate_eval_count,
        j2_correction_rejected_count,
    );
    merged.j2_correction_residual_m_sum += counters.j2_correction_residual_m_sum;
    project_diag_counts!(merged, counters; j2_correction_residual_finite_count);
    merged.j2_correction_rejected_residual_m_sum += counters.j2_correction_rejected_residual_m_sum;
    project_diag_counts!(merged, counters;
        j2_correction_call_count,
        j2_correction_iteration_count,
        j2_correction_lambert_retry_count,
    );
    // .max, not addition -- preserved exactly from the hand-written projection.
    merged.branch_source_count = merged.branch_source_count.max(counters.branch_source_count);
    project_diag_counts!(merged, counters;
        branch_shared_prepare_count,
        branch_eval_call_count,
        branch_emitted_count,
        branch_rejected_count,
        branch_target_propagation_call_count,
        branch_lambert_sampling_call_count,
        branch_brent_call_count,
        branch_brent_eval_request_count,
        branch_brent_cache_hit_count,
        branch_brent_cache_miss_count,
        branch_j2_correction_call_count,
    );
    merged.branch_shared_prepare_s += counters.branch_shared_prepare_s;
    merged.branch_phase_release_s += counters.branch_phase_release_s;
    merged.branch_target_propagation_s += counters.branch_target_propagation_s;
    merged.branch_lambert_sampling_s += counters.branch_lambert_sampling_s;
    merged.branch_brent_s += counters.branch_brent_s;
    merged.branch_j2_correction_s += counters.branch_j2_correction_s;
    *metrics = merged;
    Ok(())
}

#[inline]
fn selected_pair_front_solve_percentiles(samples: &mut [f64]) -> (f64, f64, f64) {
    samples.sort_by(|left, right| {
        right
            .is_finite()
            .cmp(&left.is_finite())
            .then_with(|| left.total_cmp(right))
    });
    let finite_count = samples.partition_point(|value| value.is_finite());
    let Some(last_index) = finite_count.checked_sub(1) else {
        return (0.0, 0.0, 0.0);
    };
    let p50_index = (last_index / 2).saturating_add(last_index % 2);
    let p95_index = last_index.saturating_sub(last_index / 20);
    let (Some(&p50), Some(&p95), Some(&max)) = (
        samples.get(p50_index),
        samples.get(p95_index),
        samples.get(last_index),
    ) else {
        return (0.0, 0.0, 0.0);
    };
    (p50, p95, max)
}

#[inline]
fn branch_rows_per_source_percentiles(samples: &mut [usize]) -> (usize, usize, usize) {
    let Some(last_index) = samples.len().checked_sub(1) else {
        return (0, 0, 0);
    };
    samples.sort_unstable();
    let p50_index = (last_index / 2).saturating_add(last_index % 2);
    let p95_index = last_index.saturating_sub(last_index / 20);
    let (Some(&p50), Some(&p95), Some(&max)) = (
        samples.get(p50_index),
        samples.get(p95_index),
        samples.get(last_index),
    ) else {
        return (0, 0, 0);
    };
    (p50, p95, max)
}

/// Solve constellation candidates as one transfer Pareto front.
///
/// Front objective order:
/// 1. minimize total delta-V,
/// 2. minimize time from t0 to intercept per relative velocity at intercept.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontOutputMode {
    TransferPareto,
    VerifiedSuperset,
}

/// Result of solving one selected (satellite, target) pair for one event.
///
/// Mirrors the indexed tuple produced by the historical selected-pair
/// `map_init` closure: `order` is the pre-flatten index used for deterministic
/// reduction, the remaining fields feed the per-event reducers unchanged.
pub(crate) struct PairFrontResult {
    pub(crate) order: usize,
    pub(crate) sat_idx: usize,
    pub(crate) tgt_idx: usize,
    pub(crate) dv_proxy: f64,
    pub(crate) x_hint: [f64; 3],
    pub(crate) front: TransferFront,
}

/// Solve exactly one selected (satellite, target) pair for one event.
///
/// This is a byte-for-byte logic move of the selected-pair `map_init` closure
/// body that previously lived inline in
/// `constellation_solve_native_with_front_output_mode`. It is the SINGLE
/// implementation shared by (a) the in-function serial loop, (b) the
/// in-function L2 selected-pair `par_iter` closure, and (c) the flat
/// event×pair driver in `batch_eci.rs` — so those paths cannot diverge.
///
/// Determinism / no-Python contract: the per-worker `PlanContext` and
/// `TransferMooWorkspace` are supplied by the caller (via `map_init` on a
/// worker, or as plain locals on the serial path); the transitive callees
/// contain no Python, logging, or direct output calls, preserving the
/// historical GIL-deadlock fix without worker-side I/O.
///
/// # Errors
///
/// Returns [`InvalidTargetPropagationAuthorityCode::ArithmeticOverflow`] when
/// the selected pair cannot represent a required arithmetic result.
#[inline]
pub(crate) fn solve_one_selected_pair(
    plan: &EventPlan<'_>,
    candidate: &PairProxyCandidate,
    order: usize,
    local_ctx: &mut PlanContext,
    moo_workspace: &mut TransferMooWorkspace,
) -> Result<Option<PairFrontResult>, InvalidTargetPropagationAuthorityCode> {
    let sat_idx = candidate.sat_idx;
    let tgt_idx = candidate.tgt_idx;
    if sat_idx >= plan.n_sats {
        return Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow);
    }
    let (
        Some(&satellite),
        Some(&satellite_equ),
        Some(&props),
        Some(&target),
        Some(&target_body_force),
        Some(&target_equ),
        Some(&target_period_cached),
        Some(&target_orbit_valid),
        Some(&target_sma),
        Some(&target_period),
    ) = (
        plan.satellites.get(sat_idx),
        plan.satellites_equ.as_ref().get(sat_idx),
        plan.sat_props.get(sat_idx),
        plan.targets.get(tgt_idx),
        plan.target_body_forces.get(tgt_idx),
        plan.targets_equ.get(tgt_idx),
        plan.targets_period_cached.get(tgt_idx),
        plan.targets_orbit_valid.get(tgt_idx),
        plan.targets_sma.get(tgt_idx),
        plan.targets_period.get(tgt_idx),
    )
    else {
        return Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow);
    };
    let sat_idx_i32 = i32::try_from(sat_idx)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let tgt_idx_i32 = i32::try_from(tgt_idx)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let context_apply_start = StageTimer::start();
    local_ctx.apply_template_pair(
        &plan.pair_template,
        &PairPlanContextInputs {
            dep_eci: satellite,
            dep_equ: satellite_equ,
            epoch_jd: plan.epoch_jd,
            tgt_eci: target,
            tgt_equ: target_equ,
            dep_sma: props.sma_orbit,
            dep_period: props.period_orbit,
            dep_orbit_cached: props.orbit_cached,
            dep_orbit_valid: props.orbit_valid,
            tgt_period_cached: target_period_cached,
            tgt_orbit_valid: target_orbit_valid,
            tgt_sma: target_sma,
            tgt_period: target_period,
        },
    )?;
    local_ctx.target_body_force = target_body_force;
    let context_apply_s = context_apply_start.elapsed_s();
    let pair_warm_start = plan
        .warm_start
        .as_ref()
        .filter(|seed| warm_start_matches_pair(seed, sat_idx_i32, tgt_idx_i32));
    let front_solve_start = StageTimer::start();
    let front_output_mode = plan.front_output_mode();
    let mut front = match front_output_mode {
        FrontOutputMode::TransferPareto => solve_plan_oxymoo_front_with_pair_workspace(
            local_ctx,
            pair_warm_start,
            Some(candidate.x_hint),
            moo_workspace,
        )?,
        FrontOutputMode::VerifiedSuperset => {
            solve_plan_oxymoo_verified_superset_with_pair_workspace(
                local_ctx,
                pair_warm_start,
                Some(candidate.x_hint),
                moo_workspace,
            )?
        }
    };
    if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        front
            .verified_superset_metrics
            .selected_pair_context_apply_s = context_apply_s;
        front.verified_superset_metrics.selected_pair_front_solve_s = front_solve_start.elapsed_s();
    }
    Ok(Some(PairFrontResult {
        order,
        sat_idx,
        tgt_idx,
        dv_proxy: candidate.dv_proxy,
        x_hint: candidate.x_hint,
        front,
    }))
}

/// Immutable controls shared by every candidate pair in one constellation solve.
///
/// This is intentionally the sole internal configuration carrier for batch
/// events. Keeping it typed prevents positional forwarding from silently
/// changing force, numerical, or search authority between phases.
#[derive(Clone)]
pub(crate) struct ConstellationSolveConfiguration {
    pub max_time_s: f64,
    pub max_phase_dv: f64,
    pub max_transfer_dv: f64,
    pub max_revs: i32,
    pub min_perigee: f64,
    pub max_apogee: f64,
    pub pairs_to_verify: usize,
    pub sampling_mode: SamplingMode,
    pub search_depth: SearchDepthPolicy,
    pub epoch_jd: f64,
    pub distance_tol: f64,
    pub deployer_min_distance: f64,
    pub tof_penalty_weight: f64,
    pub revolution_cap: f64,
    pub target_propagation_authority: TargetPropagationAuthority,
    pub force_config: Option<std::sync::Arc<lightyear_odeint_rs::types::ForceConfig>>,
    pub require_high_fidelity: bool,
    pub j2_closure_settings: J2ClosureSettings,
    pub packed_coeffs: Option<std::sync::Arc<satpy_core::PackedGravityCoeffs>>,
    pub local_optimizer: TransferLocalOptimizerConfig,
    pub warm_start: Option<WarmStartData>,
}

/// One event prepared for either serial or flat batch pair reduction.
pub(crate) struct EventPlanRequest<'a, 'scratch> {
    pub satellites: &'a [[f64; 6]],
    pub satellites_equ_cached: Option<&'a [[f64; 6]]>,
    pub target1: &'a [f64; 6],
    pub target2: &'a [f64; 6],
    pub target_body_forces: [BodyForceConfig; 2],
    pub configuration: ConstellationSolveConfiguration,
    pub scratch: Option<&'scratch mut crate::scratch::SolveScratch>,
    pub front_output_mode: FrontOutputMode,
}

pub(crate) struct EventPlan<'a> {
    front_output_mode: FrontOutputMode,
    n_sats: usize,
    pair_template: PlanContextTemplate,
    satellites: &'a [[f64; 6]],
    satellites_equ: Cow<'a, [[f64; 6]]>,
    targets: [[f64; 6]; 2],
    target_body_forces: [BodyForceConfig; 2],
    targets_equ: [[f64; 6]; 2],
    targets_period_cached: [f64; 2],
    targets_orbit_valid: [bool; 2],
    targets_sma: [f64; 2],
    targets_period: [f64; 2],
    sat_props: Vec<SatOrbitProps>,
    epoch_jd: f64,
    warm_start: Option<WarmStartData>,
    selected_pairs: Vec<PairProxyCandidate>,
    /// Screening-phase metrics captured during `prepare_event`, folded back in
    /// by `reduce_event` so the verified-superset diagnostics match the inline
    /// per-event path.
    screen_metrics: VerifiedSupersetStageMetrics,
    /// Wall start of the selected-pair stage (for the aggregate solve-time
    /// metric reconstructed in `reduce_event`).
    selected_pair_solve_start: StageTimer,
}

impl EventPlan<'_> {
    /// Number of selected pairs in this event (the flatten width).
    #[inline]
    pub(crate) const fn selected_pair_count(&self) -> usize {
        self.selected_pairs.len()
    }

    /// Borrow selected-pair candidate at `slot`, if it exists.
    #[inline]
    pub(crate) fn selected_pair(&self, slot: usize) -> Option<&PairProxyCandidate> {
        self.selected_pairs.get(slot)
    }

    #[inline]
    pub(crate) const fn front_output_mode(&self) -> FrontOutputMode {
        self.front_output_mode
    }

    #[inline]
    pub(crate) const fn j2_closure_settings(&self) -> J2ClosureSettings {
        self.pair_template.j2_closure_settings
    }

    #[cfg(test)]
    pub(crate) const fn uses_borrowed_satellite_equ_state(&self) -> bool {
        matches!(self.satellites_equ, Cow::Borrowed(_))
    }
}

/// Phase A: per-event screening + setup, factored out of
/// `constellation_solve_native_with_front_output_mode` so it can be reused by
/// the flat event×pair driver (`batch_eci.rs`).
///
/// Performs target/satellite invariant precompute, pair-proxy screening,
/// `select_pair_proxy_candidates_reuse`, and `PlanContextTemplate` construction
/// — exactly the work the inline path did at the head of the solve. Returns an
/// [`EventPlan`] that borrows the caller-owned satellite arena, owns derived
/// state and selected pairs, and outlives any transient `scratch`. It is
/// `Send`/`Sync` for the flat driver while that immutable arena stays live.
/// Returns `Ok(None)` for the `n_sats == 0` early-out (caller substitutes an
/// empty front), matching the inline guard. Candidate-search authority and
/// arithmetic capacity failures are typed errors, never an indistinguishable
/// empty event.
///
/// This runs NO pair solve and NO rayon `par_iter`; the selected-pair stage is
/// owned by `solve_one_selected_pair`.
pub(crate) fn prepare_event<'a>(
    request: EventPlanRequest<'a, '_>,
) -> Result<Option<EventPlan<'a>>, InvalidTargetPropagationAuthorityCode> {
    let EventPlanRequest {
        satellites,
        satellites_equ_cached,
        target1,
        target2,
        target_body_forces,
        configuration,
        scratch,
        front_output_mode,
    } = request;
    let ConstellationSolveConfiguration {
        max_time_s,
        max_phase_dv,
        max_transfer_dv,
        max_revs,
        min_perigee,
        max_apogee,
        pairs_to_verify,
        sampling_mode,
        search_depth,
        epoch_jd,
        distance_tol,
        deployer_min_distance,
        tof_penalty_weight,
        revolution_cap,
        target_propagation_authority,
        force_config,
        require_high_fidelity,
        j2_closure_settings,
        packed_coeffs,
        local_optimizer,
        warm_start,
    } = configuration;
    crate::types::validate_candidate_search_authority(
        target_propagation_authority,
        force_config.as_deref(),
        require_high_fidelity,
    )
    .map_err(InvalidTargetPropagationAuthorityCode::CandidateSearch)?;
    for target_body_force in target_body_forces {
        crate::types::validate_target_body_force(target_propagation_authority, target_body_force)?;
    }
    let n_sats = satellites.len();
    if n_sats == 0 {
        return Ok(None);
    }
    let pair_proxy_capacity = pair_proxy_capacity_for_satellite_count(n_sats)?;
    let pair_screen_start = StageTimer::start();
    let mut screen_metrics = VerifiedSupersetStageMetrics::default();

    let mut target_one_equ = [0.0; 6];
    let mut target_two_equ = [0.0; 6];
    satpy_core::eci2equinoc_impl(target1, 6, 0.0, 0.0, &mut target_one_equ);
    satpy_core::eci2equinoc_impl(target2, 6, 0.0, 0.0, &mut target_two_equ);
    let target_one_orbit = target_orbit_invariants(target1);
    let target_two_orbit = target_orbit_invariants(target2);

    let satellites_equ = if let Some(cached) = satellites_equ_cached {
        Cow::Borrowed(cached)
    } else {
        let mut computed = Vec::new();
        try_reserve_transfer_capacity(&mut computed, n_sats)?;
        for satellite in satellites {
            let mut equ = [0.0; 6];
            satpy_core::eci2equinoc_impl(satellite, 6, 0.0, 0.0, &mut equ);
            computed.push(equ);
        }
        Cow::Owned(computed)
    };

    let mut sat_props = Vec::new();
    try_reserve_transfer_capacity(&mut sat_props, n_sats)?;
    for satellite in satellites {
        sat_props.push(compute_sat_orbit_props(satellite));
    }

    let mut local_pair_proxy_scratch;
    let pair_proxy_scratch: &mut PairProxyScratch = if let Some(scratch) = scratch {
        scratch.pair_proxy.prepare(pair_proxy_capacity)?;
        &mut scratch.pair_proxy
    } else {
        local_pair_proxy_scratch = PairProxyScratch::new(pair_proxy_capacity);
        local_pair_proxy_scratch.prepare(pair_proxy_capacity)?;
        &mut local_pair_proxy_scratch
    };
    let target_states = screen_pair_proxy_candidates(
        [
            PairProxyTargetInput {
                eci: target1,
                equ: target_one_equ,
                orbit: target_one_orbit,
            },
            PairProxyTargetInput {
                eci: target2,
                equ: target_two_equ,
                orbit: target_two_orbit,
            },
        ],
        satellites,
        &sat_props,
        max_time_s,
        pairs_to_verify,
        search_depth.pair_proxy_model,
        pair_proxy_scratch,
    )?;

    let pair_selection_meta =
        select_pair_proxy_candidates_reuse(pair_proxy_scratch, pairs_to_verify)?;
    if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        let [target0_count, target1_count] = pair_selection_meta.selected_by_target;
        screen_metrics.pair_screen_s = pair_screen_start.elapsed_s();
        screen_metrics.pair_proxy_candidate_count = pair_selection_meta.total_pairs;
        screen_metrics.selected_pair_count = pair_selection_meta.selected_pairs;
        screen_metrics.pair_proxy_exact_mode = pair_selection_meta.exact_mode;
        screen_metrics.selected_pair_target0_count = target0_count;
        screen_metrics.selected_pair_target1_count = target1_count;
    }

    let execution_policy = ExecutionPolicy {
        use_high_fidelity: force_config.is_some(),
        require_high_fidelity,
        allow_parallel: false,
        allow_oxymoo_batch_parallel: false,
        allow_branch_expansion_parallel: false,
        allow_polish_parallel: false,
        allow_anchor_parallel: false,
        allow_deterministic_grid_parallel: false,
    };
    let pair_template = PlanContextTemplate {
        max_time_s,
        tof_penalty_weight,
        revolution_cap,
        max_phase_dv,
        max_transfer_dv,
        min_perigee,
        max_apogee,
        max_revs,
        sampling_mode,
        execution_policy,
        j2_closure_settings,
        search_depth,
        distance_tol,
        deployer_min_distance,
        target_propagation_authority,
        force_config,
        packed_coeffs,
        local_optimizer,
    };

    let mut selected_pairs = Vec::new();
    try_reserve_transfer_capacity(&mut selected_pairs, pair_proxy_scratch.selected().len())?;
    selected_pairs.extend_from_slice(pair_proxy_scratch.selected());
    let [target_one, target_two] = target_states;
    Ok(Some(EventPlan {
        front_output_mode,
        n_sats,
        pair_template,
        satellites,
        satellites_equ,
        targets: [*target_one.eci, *target_two.eci],
        target_body_forces,
        targets_equ: [target_one.equ, target_two.equ],
        targets_period_cached: [target_one.period_cached, target_two.period_cached],
        targets_orbit_valid: [target_one.orbit_valid, target_two.orbit_valid],
        targets_sma: [target_one.sma, target_two.sma],
        targets_period: [target_one.period, target_two.period],
        sat_props,
        epoch_jd,
        warm_start,
        selected_pairs,
        screen_metrics,
        selected_pair_solve_start: StageTimer::start(),
    }))
}

#[inline]
fn pair_proxy_capacity_for_satellite_count(
    satellite_count: usize,
) -> Result<usize, InvalidTargetPropagationAuthorityCode> {
    satellite_count
        .checked_mul(2)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
}

/// Per-event metric flags threaded into [`reduce_event`].
///
/// These are pure diagnostic counters (folded via `add_assign`); they have no
/// effect on the returned candidates. They exist so both the in-function path
/// (which may run the L2 selected-pair `par_iter`) and the flat event×pair
/// driver (always serial-per-event under one outer rayon boundary) reconstruct
/// the historical `VerifiedSupersetStageMetrics` event counters.
#[derive(Clone, Copy, Default)]
pub(crate) struct EventReduceFlags {
    /// This event's selected pairs were solved via the L2 `par_iter` closure.
    pub(crate) selected_pair_parallel: bool,
    /// The L2 selected-pair parallel policy was enabled for this event.
    pub(crate) selected_pair_parallel_policy_enabled: bool,
    /// This event ran under the outer batch-parallel boundary (flat driver).
    pub(crate) outer_batch_parallel: bool,
    /// Effective Rayon width of this execution path. Inline execution reports
    /// one without consulting the process-global Rayon pool.
    pub(crate) rayon_current_num_threads: usize,
}

/// Phase C: reduce the solved selected-pair fronts of ONE event into its
/// `ConstellationTransferFront`, in the fixed `pair_slot` order.
///
/// `results` must be the per-pair outputs in selected-pair order (`order` /
/// `pair_slot` ascending. A missing selected-pair slot or invalid pair index is
/// an authority arithmetic error before this reducer runs; the retained
/// `Option` shape serves the flat batch interface without converting such
/// corruption into a synthetic skipped result.
///
/// # Errors
///
/// Returns an authority arithmetic-overflow error if independently accumulated
/// verified-superset metric counts cannot be represented.
pub(crate) fn reduce_event(
    plan: &EventPlan<'_>,
    results: Vec<Option<PairFrontResult>>,
    flags: EventReduceFlags,
) -> Result<ConstellationTransferFront, crate::types::InvalidTargetPropagationAuthorityCode> {
    let front_output_mode = plan.front_output_mode;
    let mut superset_metrics = plan.screen_metrics;
    let mut archive = ConstellationFrontArchive::new();
    let mut superset_candidates: Vec<ConstellationTransferCandidate> = Vec::new();
    let mut selected_pair_front_solve_samples = Vec::new();
    try_reserve_transfer_capacity(&mut selected_pair_front_solve_samples, results.len())?;

    if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        {
            superset_metrics.rayon_current_num_threads = flags.rayon_current_num_threads;
        }
        if flags.selected_pair_parallel_policy_enabled {
            superset_metrics.selected_pair_parallel_policy_enabled_count = 1;
        }
        if flags.selected_pair_parallel {
            superset_metrics.selected_pair_parallel_event_count = 1;
        } else {
            superset_metrics.selected_pair_serial_event_count = 1;
        }
        if flags.outer_batch_parallel {
            superset_metrics.outer_batch_parallel_event_count = superset_metrics
                .outer_batch_parallel_event_count
                .checked_add(1)
                .ok_or(crate::types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        }
        for result in results.iter().flatten() {
            selected_pair_front_solve_samples.push(
                result
                    .front
                    .verified_superset_metrics
                    .selected_pair_front_solve_s,
            );
        }
    }

    for result in results.into_iter().flatten() {
        let PairFrontResult {
            order: _order,
            sat_idx,
            tgt_idx,
            dv_proxy,
            x_hint,
            front,
        } = result;
        match front_output_mode {
            FrontOutputMode::TransferPareto => {
                push_constellation_transfer_candidates(
                    &mut archive,
                    sat_idx,
                    tgt_idx,
                    dv_proxy,
                    x_hint,
                    front,
                )?;
            }
            FrontOutputMode::VerifiedSuperset => {
                let append_start = StageTimer::start();
                superset_metrics.add_assign(front.verified_superset_metrics)?;
                append_constellation_transfer_candidates(
                    &mut superset_candidates,
                    sat_idx,
                    tgt_idx,
                    dv_proxy,
                    x_hint,
                    front,
                )?;
                superset_metrics.selected_pair_result_append_s += append_start.elapsed_s();
            }
        }
    }

    if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        superset_metrics.selected_pair_solve_s = plan.selected_pair_solve_start.elapsed_s();
        let (p50, p95, pair_max_s) =
            selected_pair_front_solve_percentiles(&mut selected_pair_front_solve_samples);
        superset_metrics.selected_pair_front_solve_p50_s = p50;
        superset_metrics.selected_pair_front_solve_p95_s = p95;
        superset_metrics.selected_pair_front_solve_pair_max_s = pair_max_s;
        let explained = superset_metrics.selected_pair_context_apply_s
            + superset_metrics.selected_pair_front_solve_s
            + superset_metrics.selected_pair_result_append_s;
        superset_metrics.selected_pair_residual_s =
            (superset_metrics.selected_pair_solve_s - explained).max(0.0);
    }

    Ok(match front_output_mode {
        FrontOutputMode::TransferPareto => archive.into_front(),
        FrontOutputMode::VerifiedSuperset => {
            let constellation_finalize_start = StageTimer::start();
            let candidates =
                finalize_constellation_transfer_superset(superset_candidates).candidates;
            superset_metrics.constellation_finalize_s = constellation_finalize_start.elapsed_s();
            ConstellationTransferFront::with_verified_superset_metrics(candidates, superset_metrics)
        }
    })
}

/// Solve every selected pair of an [`EventPlan`] serially (one worker), in
/// selected-pair order, via the shared [`solve_one_selected_pair`].
///
/// This is the serial path an event solve uses when the L2 policy is off
/// (e.g. pool size 1 / nested), and is byte-for-byte the historical serial
/// loop (it calls the SAME leaf as the parallel path). Out-of-range pairs map
/// to `None`, preserving the drain shape consumed by [`reduce_event`].
fn solve_selected_pairs_serial(
    plan: &EventPlan<'_>,
) -> Result<Vec<Option<PairFrontResult>>, crate::types::InvalidTargetPropagationAuthorityCode> {
    let mut moo_workspace = TransferMooWorkspace::new();
    let mut ctx = PlanContext::with_j2_closure_settings(plan.j2_closure_settings());
    let mut results = Vec::new();
    try_reserve_transfer_capacity(&mut results, plan.selected_pair_count())?;
    for slot in 0..plan.selected_pair_count() {
        let candidate = plan
            .selected_pair(slot)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        let result = solve_one_selected_pair(plan, candidate, slot, &mut ctx, &mut moo_workspace)?;
        results.push(result);
    }
    Ok(results)
}

/// Implementation of `constellation_solve_native_with_front_output_mode`,
/// composed from the reusable Phase A / leaf / Phase C primitives.
///
/// Phase A is [`prepare_event`]; each selected pair is solved by the shared
/// [`solve_one_selected_pair`] (either via the L2 `par_iter` when the policy
/// admits it, or the serial path); Phase C is [`reduce_event`]. This keeps
/// the serial and L2-parallel paths from diverging (same leaf, same reducer)
/// and is the SAME machinery the flat event×pair driver in `batch_eci.rs`
/// reuses. `outer_batch_parallel` is always `false` here — the caller
/// (`batch_eci.rs::finish_event_timing`) still owns the outer-batch counter on
/// this in-function path.
fn constellation_solve_native_with_front_output_mode_impl(
    request: EventPlanRequest<'_, '_>,
) -> Result<ConstellationTransferFront, crate::types::InvalidTargetPropagationAuthorityCode> {
    let Some(plan) = prepare_event(request)? else {
        return Ok(ConstellationTransferFront::empty());
    };

    let selected_pair_parallel_enabled =
        should_parallelize_selected_pairs(plan.selected_pair_count());

    if selected_pair_parallel_enabled {
        let flags = EventReduceFlags {
            selected_pair_parallel: true,
            selected_pair_parallel_policy_enabled: true,
            outer_batch_parallel: false,
            rayon_current_num_threads: rayon::current_num_threads(),
        };
        let j2_closure_settings = plan.j2_closure_settings();
        let selected_pair_count = plan.selected_pair_count();
        let mut outcomes = Vec::new();
        try_reserve_transfer_capacity(&mut outcomes, selected_pair_count)?;
        (0..selected_pair_count)
            .into_par_iter()
            .map_init(
                || {
                    (
                        PlanContext::with_j2_closure_settings(j2_closure_settings),
                        TransferMooWorkspace::new(),
                    )
                },
                |(local_ctx, moo_workspace), slot| {
                    // Isolated region reduced in slot order below -- the shape
                    // every sibling parallel stage uses (`moo.rs`,
                    // `branch_expansion.rs`, `polish.rs`, `anchor.rs`, the
                    // deterministic grid). See `with_isolated_diag_region`.
                    with_isolated_diag_region(|| {
                        plan.selected_pair(slot)
                            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
                            .and_then(|candidate| {
                                solve_one_selected_pair(
                                    &plan,
                                    candidate,
                                    slot,
                                    local_ctx,
                                    moo_workspace,
                                )
                            })
                    })
                    .map(|(front, diag_delta, work_delta)| {
                        SelectedPairResult {
                            front,
                            diag_delta,
                            work_delta,
                        }
                    })
                },
            )
            .collect_into_vec(&mut outcomes);
        let mut results = Vec::new();
        try_reserve_transfer_capacity(&mut results, selected_pair_count)?;
        // Serial, slot order: fold each worker's counter contribution back into
        // the reducing thread before replaying the fronts.
        for outcome in outcomes {
            let outcome = outcome?;
            map_evaluation_arithmetic_overflow(crate::evaluate::merge_evaluation_diagnostics(
                &outcome.diag_delta,
            ))?;
            merge_work_counts(outcome.work_delta)?;
            results.push(outcome.front);
        }
        return reduce_event(&plan, results, flags);
    }

    let flags = EventReduceFlags {
        selected_pair_parallel: false,
        selected_pair_parallel_policy_enabled: selected_pair_parallel_enabled,
        outer_batch_parallel: false,
        rayon_current_num_threads: rayon::current_num_threads(),
    };

    let results = solve_selected_pairs_serial(&plan)?;
    reduce_event(&plan, results, flags)
}

pub(crate) fn constellation_solve_native_with_front_output_mode(
    request: EventPlanRequest<'_, '_>,
) -> Result<ConstellationTransferFront, crate::types::InvalidTargetPropagationAuthorityCode> {
    crate::types::validate_candidate_search_authority(
        request.configuration.target_propagation_authority,
        request.configuration.force_config.as_deref(),
        request.configuration.require_high_fidelity,
    )
    .map_err(crate::types::InvalidTargetPropagationAuthorityCode::CandidateSearch)?;
    constellation_solve_native_with_front_output_mode_impl(request)
}

/// Solve for the optimal two-phase transfer plan.
///
/// Uses deterministic heuristic-seeded search over a fixed candidate set and
/// returns the non-dominated transfer front used by the Rust-first pipeline.
///
/// # Arguments
/// * `ctx` - Plan context with orbital states and constraints
/// * `warm_start` - Optional seed from a prior nearby solve
///
/// # Returns
/// Pareto-ranked `TransferFront` with best transfer candidates found
///
/// # Errors
///
/// Returns an authority error when target-force, target-propagation, or
/// candidate-search controls conflict before search begins.
pub fn solve_plan(
    ctx: &mut PlanContext,
    warm_start: Option<&crate::types::WarmStartData>,
) -> Result<TransferFront, crate::types::InvalidTargetPropagationAuthorityCode> {
    crate::types::validate_candidate_search_authority(
        ctx.target_propagation_authority,
        ctx.force_config.as_deref(),
        ctx.execution_policy.use_high_fidelity || ctx.execution_policy.require_high_fidelity,
    )
    .map_err(crate::types::InvalidTargetPropagationAuthorityCode::CandidateSearch)?;
    crate::types::validate_target_propagation_authority(
        ctx.target_propagation_authority,
        ctx.target_body_force,
        ctx.force_config.as_deref(),
    )?;
    solve_plan_front_internal(ctx, warm_start)
}

#[inline]
fn map_evaluation_arithmetic_overflow<T>(
    result: Result<T, EvaluationArithmeticOverflow>,
) -> Result<T, InvalidTargetPropagationAuthorityCode> {
    result.map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
}

fn try_insert_solve_local_cache_entry<K, V>(
    entries: &mut FxHashMap<K, V>,
    key: K,
    value: V,
) -> Result<(), InvalidTargetPropagationAuthorityCode>
where
    K: Eq + std::hash::Hash,
{
    if !entries.contains_key(&key) {
        entries
            .try_reserve(1)
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    }
    entries.insert(key, value);
    Ok(())
}

fn evaluate_plan_local(
    x: &[f64; 3],
    ctx: &PlanContext,
    coarse_mode: bool,
    cache: &RefCell<SolveLocalWorkCache>,
) -> Result<PlanResult, crate::types::InvalidTargetPropagationAuthorityCode> {
    let [time2phase_ratio, _, waittime_ratio] = *x;
    // Every call computes one fresh `PlanResult`. `SolveLocalWorkCache` owns
    // only exact phase/orbit intermediates plus Lambert scratch.
    record_work_count(|counters| {
        counters.plan_full_evaluations = counters
            .plan_full_evaluations
            .checked_add(1)
            .ok_or(crate::types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        Ok(())
    })?;

    let phase_key = time2phase_ratio.to_bits();
    let dep_at_phase = {
        let cached_phase_state = {
            let borrowed = cache.borrow();
            borrowed.phase_state_cache.get(&phase_key).copied()
        };
        // Both sides of the one lookup, so `phase_state_cache_hit_count +
        // phase_state_cache_miss_count` is the lookup count by construction.
        // The miss side goes on to reach `j2_propagate_state_count` below,
        // which is what gives that bare miss tally a denominator.
        record_phase_state_cache_lookup(cached_phase_state.is_some())
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        if let Some(state) = cached_phase_state {
            state
        } else {
            if !time2phase_ratio.is_finite() {
                return evaluate_plan(x, ctx, coarse_mode);
            }
            let time2phase = time2phase_ratio * ctx.max_time_s;
            let Some(state) = propagate_candidate_state_at_epoch(
                &ctx.dep_equ,
                time2phase,
                ctx.epoch_jd,
                ctx.transfer_body_force(),
                ctx,
            )
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?
            else {
                return evaluate_plan(x, ctx, coarse_mode);
            };
            {
                let mut borrowed_cache = cache.borrow_mut();
                try_insert_solve_local_cache_entry(
                    &mut borrowed_cache.phase_state_cache,
                    phase_key,
                    state,
                )?;
            }
            state
        }
    };

    let dep_phase_orbit = {
        let cached_phase_orbit = {
            let borrowed = cache.borrow();
            borrowed.phase_orbit_cache.get(&phase_key).copied()
        };
        if let Some(orbit) = cached_phase_orbit {
            orbit
        } else {
            let Some(orbit) = eci_orbit_summary(&dep_at_phase) else {
                return evaluate_plan(x, ctx, coarse_mode);
            };
            {
                let mut borrowed_cache = cache.borrow_mut();
                try_insert_solve_local_cache_entry(
                    &mut borrowed_cache.phase_orbit_cache,
                    phase_key,
                    orbit,
                )?;
            }
            orbit
        }
    };

    let dep_period = compute_dep_period(ctx);
    let time2phase = time2phase_ratio * ctx.max_time_s;
    let waittime = waittime_ratio * ctx.max_time_s;

    // Pass 38a — release the cache borrow across the inner Lambert compute.
    // `mem::take` the Vec-backed scratch (preserving capacity via swap
    // ownership), run the compute without holding the borrow, then reacquire
    // to restore + insert.
    //
    // This used to be justified by "so a shared Mutex (under `parallel-eval`)
    // doesn't serialize all rayon workers on this scratch". There is no such
    // Mutex and no such feature: `parallel-eval` does not exist anywhere in
    // the tree, and this cache is a `RefCell` owned per rayon worker via
    // `map_init`, so no other thread can ever contend for it. Keeping the
    // narrow borrow is still right — holding a `RefCell` borrow across a long
    // call is how a future re-entrant path earns a panic — but do not read
    // this as evidence that the solve has lock contention to recover. It has
    // no locks at all on this path.
    let mut scratch = std::mem::take(&mut cache.borrow_mut().variable_r2_lambert_scratch);
    let result = map_evaluation_arithmetic_overflow(evaluate_plan_from_phase_with_lambert_scratch(
        x,
        ctx,
        coarse_mode,
        time2phase,
        waittime,
        dep_period,
        &dep_at_phase,
        Some(dep_phase_orbit),
        Some(&mut scratch),
    ));
    {
        let mut borrowed_cache = cache.borrow_mut();
        borrowed_cache.variable_r2_lambert_scratch = scratch;
    }
    result
}

fn prepare_single_pair_context(ctx: &mut PlanContext) {
    // Ensure caches are populated up front.
    if ctx.dep_sma <= 0.0 {
        ctx.cache_deployer_orbit();
    }
    if !ctx.tgt_orbit_valid {
        ctx.cache_target_orbit();
    }
    if !ctx.plane_angle_valid {
        ctx.cache_plane_angle();
    }
    ctx.execution_policy.allow_parallel = false;
    // Opt leaf stages into adaptive global-pool fan-out. Their runtime gates
    // fan out only for a top-level caller; a solve already running on a rayon
    // worker stays leaf-serial so outer/cross-cell work owns the pool. Inner TOF
    // iteration stays serial in either case (`allow_parallel = false`).
    ctx.execution_policy.allow_oxymoo_batch_parallel = true;
    ctx.execution_policy.allow_branch_expansion_parallel = true;
    ctx.execution_policy.allow_polish_parallel = true;
    ctx.execution_policy.allow_anchor_parallel = true;
    ctx.execution_policy.allow_deterministic_grid_parallel = true;
}

fn nm_max_iters_for_complexity(complexity: TransferComplexity) -> usize {
    PsoPreset::for_complexity(complexity, false)
        .max_iters
        .max(32)
}

const fn optimizer_seed_limit(kind: LocalOptimizerKind) -> usize {
    match kind {
        LocalOptimizerKind::NelderMead | LocalOptimizerKind::Lbfgs => 3,
        LocalOptimizerKind::Pso => 5,
    }
}

fn select_optimizer_start_seeds(
    ranked_seeds: &[(SolverSeed, PlanResult)],
    kind: LocalOptimizerKind,
) -> Result<Vec<SolverSeed>, InvalidTargetPropagationAuthorityCode> {
    let mut selected = Vec::new();
    let limit = optimizer_seed_limit(kind);
    try_reserve_transfer_capacity(&mut selected, ranked_seeds.len().min(limit))?;

    for (seed, _) in ranked_seeds.iter().take(limit) {
        if selected
            .iter()
            .any(|existing: &SolverSeed| seed_is_duplicate(&existing.x, &seed.x))
        {
            continue;
        }
        selected.push(*seed);
    }

    Ok(selected)
}

#[cfg(test)]
fn retain_objective_aware_seed_candidates(
    selected: &mut Vec<(SolverSeed, PlanResult)>,
    eligible: &[(SolverSeed, PlanResult)],
) {
    push_best_seed_by(selected, eligible, PlanResult::total_dv, false);
    push_best_seed_by(
        selected,
        eligible,
        PlanResult::time_per_relative_velocity_s_per_km_s,
        false,
    );

    for (seed, plan) in eligible {
        if !transfer_candidate_is_objective_finite(plan) {
            continue;
        }
        let dominated = eligible.iter().any(|(other_seed, other_plan)| {
            !seed_is_duplicate(&other_seed.x, &seed.x)
                && transfer_candidate_is_objective_finite(other_plan)
                && transfer_candidate_dominates(other_plan, plan)
        });
        if !dominated {
            push_unique_seed_plan(selected, *seed, plan.clone());
        }
    }
}

#[cfg(test)]
fn push_best_seed_by(
    selected: &mut Vec<(SolverSeed, PlanResult)>,
    eligible: &[(SolverSeed, PlanResult)],
    value: impl Fn(&PlanResult) -> f64,
    maximize: bool,
) {
    let best = eligible
        .iter()
        .filter(|(_, plan)| transfer_candidate_is_objective_finite(plan))
        .filter_map(|(seed, plan)| {
            let score = value(plan);
            score.is_finite().then_some((*seed, plan, score))
        })
        .min_by(|(_, _, left), (_, _, right)| {
            if maximize {
                right.partial_cmp(left).unwrap_or(Ordering::Equal)
            } else {
                left.partial_cmp(right).unwrap_or(Ordering::Equal)
            }
        });
    if let Some((seed, plan, _)) = best {
        push_unique_seed_plan(selected, seed, plan.clone());
    }
}

#[cfg(test)]
fn push_unique_seed_plan(
    selected: &mut Vec<(SolverSeed, PlanResult)>,
    seed: SolverSeed,
    plan: PlanResult,
) {
    if selected
        .iter()
        .any(|(existing, _)| seed_is_duplicate(&existing.x, &seed.x))
    {
        return;
    }
    selected.push((seed, plan));
}

fn auto_local_optimizer_kind(complexity: TransferComplexity, best_cost: f64) -> LocalOptimizerKind {
    if best_cost < 0.15 {
        return LocalOptimizerKind::NelderMead;
    }
    match complexity {
        TransferComplexity::Trivial | TransferComplexity::Easy => LocalOptimizerKind::NelderMead,
        TransferComplexity::Moderate if best_cost < 0.20 => LocalOptimizerKind::NelderMead,
        _ => LocalOptimizerKind::Pso,
    }
}

fn resolve_local_optimizer_kind(
    config: TransferLocalOptimizerConfig,
    complexity: TransferComplexity,
    best_cost: f64,
) -> LocalOptimizerKind {
    match config.choice {
        TransferLocalOptimizerChoice::Auto => auto_local_optimizer_kind(complexity, best_cost),
        TransferLocalOptimizerChoice::Fixed(kind) => kind,
    }
}

struct TransferLocalProblem<'a> {
    ctx: &'a PlanContext,
    cache: &'a RefCell<SolveLocalWorkCache>,
    coarse_mode: bool,
    gradient_enabled: bool,
}

impl LocalScalarProblem3 for TransferLocalProblem<'_> {
    fn value(&self, x: &[f64; 3]) -> anyhow::Result<f64> {
        Ok(evaluate_plan_local(x, self.ctx, self.coarse_mode, self.cache)?.cost)
    }

    fn value_gradient(&self, x: &[f64; 3]) -> Option<(f64, [f64; 3])> {
        if !self.gradient_enabled {
            return None;
        }
        // autodiff feature removed; gradient path not available
        let _ = x;
        None
    }
}

struct TransferDeltaVProblem<'a> {
    ctx: &'a PlanContext,
    cache: &'a RefCell<SolveLocalWorkCache>,
    coarse_mode: bool,
}

impl LocalScalarProblem3 for TransferDeltaVProblem<'_> {
    fn value(&self, x: &[f64; 3]) -> anyhow::Result<f64> {
        let plan = evaluate_plan_local(x, self.ctx, self.coarse_mode, self.cache)?;
        Ok(local_delta_v_score(x, &plan, self.ctx))
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "local optimizer ranking keeps its established non-fused IEEE rounding"
)]
fn local_delta_v_score(x: &[f64; 3], plan: &PlanResult, ctx: &PlanContext) -> f64 {
    let reference = transfer_moo_dv_reference(ctx);
    let cv = transfer_moo_constraint_violation(x, plan, ctx);
    let total_dv = plan.total_dv();
    if plan.valid && total_dv.is_finite() {
        total_dv + cv * reference * 10.0
    } else {
        reference * (1.0 + cv.max(1.0) * 10.0)
    }
}

fn local_config(
    kind: LocalOptimizerKind,
    max_iters: usize,
    tune: TuneLevel,
    seed: u64,
) -> LocalOptimizerConfig {
    LocalOptimizerConfig {
        kind,
        max_iters: max_iters.max(1),
        tolerance: 1e-6,
        seed,
        tune,
        min_iters: crate::oxymoo::DEFAULT_NM_MIN_ITERS,
    }
}

/// Pure policy shared by leaf fan-out gates. Leaf work may use the single global
/// rayon pool only from a top-level caller; a caller already running on a rayon
/// worker leaves that pool available for outer/cross-cell fan-out.
#[inline]
const fn should_use_leaf_parallel(
    allowed: bool,
    work_count: usize,
    minimum_work_count: usize,
    thread_count: usize,
    caller_is_top_level: bool,
) -> bool {
    allowed && work_count >= minimum_work_count && thread_count > 1 && caller_is_top_level
}

fn push_local_result_candidate(
    candidates: &mut Vec<PlanResult>,
    result: &LocalOptimizeResult,
    ctx: &PlanContext,
    cache: &RefCell<SolveLocalWorkCache>,
    warm_start_consumed: bool,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let mut plan = evaluate_plan_local(&result.x, ctx, false, cache)?;
    plan.func_evals = result.evaluations;
    plan.optimizer_func_evals = result.evaluations;
    plan.optimizer_converged = result.converged;
    plan.warm_start_used = warm_start_consumed;
    try_reserve_transfer_capacity(candidates, 1)?;
    candidates.push(plan);
    Ok(())
}

const fn transfer_plan_decision(plan: &PlanResult) -> [f64; 3] {
    [
        plan.time2phase_ratio,
        plan.phase_sma_ratio,
        plan.waittime_ratio,
    ]
}

fn transfer_decision_exists(candidates: &[PlanResult], x: &[f64; 3]) -> bool {
    candidates
        .iter()
        .any(|existing| seed_is_duplicate(&transfer_plan_decision(existing), x))
}

fn push_unique_transfer_candidate(
    candidates: &mut Vec<PlanResult>,
    plan: PlanResult,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let x = transfer_plan_decision(&plan);
    if transfer_decision_exists(candidates, &x) {
        return Ok(());
    }
    try_reserve_transfer_capacity(candidates, 1)?;
    candidates.push(plan);
    Ok(())
}

/// Candidates covered by the release-epoch line refinement, best total-dv
/// first. See [`refine_release_epoch_line`] for why so few suffice.
const RELEASE_EPOCH_REFINE_TOP_K: usize = 4;
/// Half-width of the line scan in decision units. `x0` and `x2` move together,
/// so the release epoch `x0 + x2` is covered over twice this — and the whole
/// scan stays inside `FINAL_CANDIDATE_POLISH_RADIUS`, i.e. inside the box the
/// final polish already claims to search.
const RELEASE_EPOCH_REFINE_HALF: f64 = 0.015;
/// Samples across the line. Odd, so the incumbent decision is one of them.
const RELEASE_EPOCH_REFINE_SAMPLES: usize = 61;

// The line scan must stay inside the box the final polish already claims, or
// the refinement is searching somewhere nothing else verified; and it must
// sample its own centre.
const _: () = {
    let [time_radius, _, wait_radius] = FINAL_CANDIDATE_POLISH_RADIUS;
    assert!(RELEASE_EPOCH_REFINE_HALF <= time_radius);
    assert!(RELEASE_EPOCH_REFINE_HALF <= wait_radius);
    assert!(RELEASE_EPOCH_REFINE_SAMPLES >= 3);
    assert!(RELEASE_EPOCH_REFINE_SAMPLES % 2 == 1);
};

/// Scan the release-epoch line through each of the best candidates.
///
/// `x0` (time to the phase burn) and `x2` (coast before the transfer) enter
/// the transfer almost only through their sum: `x0 + x2` fixes the release
/// epoch, and with `x1 ~ 1` the phasing orbit barely differs from the
/// deployer's, so moving mass between the two at constant sum barely moves the
/// departure state. Measured on the `physics_3event` fixture, sweeping
/// `(x0, x2)` at 2.5e-4: the median relative spread of dv along a constant-sum
/// line is 0.14, while the minimum over that line swings by more than 10x
/// between sums 7.5e-4 apart. The decision space is therefore one stiff
/// coordinate and one nearly flat one.
///
/// Nelder-Mead handles that badly. It contracts along the flat direction and
/// stalls: taking [`FINAL_CANDIDATE_POLISH_ITERS`] from its 10 to 12 and then
/// 36 leaves all three fixture events' selected dv bit-identical. What it needs
/// is not more iterations but coverage of the stiff coordinate, which a uniform
/// scan gives at bounded resolution and bounded cost.
///
/// Appends every strict improvement as a new candidate rather than replacing
/// the incumbent, so the front keeps the point the rest of the pipeline
/// already ranked and the refinement can only add.
fn refine_release_epoch_line(
    candidates: &mut Vec<PlanResult>,
    ctx: &PlanContext,
    cache: &RefCell<SolveLocalWorkCache>,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    if candidates.is_empty() {
        return Ok(());
    }
    let mut order = Vec::new();
    try_reserve_transfer_capacity(&mut order, candidates.len())?;
    for (index, candidate) in candidates.iter().enumerate() {
        if transfer_candidate_is_objective_finite(candidate) {
            order.push((index, candidate));
        }
    }
    order.sort_by(|(left_index, left), (right_index, right)| {
        left.total_dv()
            .total_cmp(&right.total_dv())
            .then_with(|| left_index.cmp(right_index))
    });
    order.truncate(RELEASE_EPOCH_REFINE_TOP_K);

    let mut refined = Vec::new();
    try_reserve_transfer_capacity(&mut refined, order.len())?;
    let last_sample = RELEASE_EPOCH_REFINE_SAMPLES
        .checked_sub(1)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let last_sample = u32::try_from(last_sample)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let last = f64::from(last_sample);
    let [time_lower, _, wait_lower] = SINGLE_PAIR_LOWER_BOUNDS;
    let [time_upper, _, wait_upper] = SINGLE_PAIR_UPPER_BOUNDS;
    for (_, candidate) in order {
        let start = repaired_transfer_decision(&transfer_plan_decision(candidate));
        if !start.iter().all(|value| value.is_finite()) {
            continue;
        }
        let [start_time, start_phase, start_wait] = start;
        let incumbent_dv = candidate.total_dv();
        let mut best: Option<([f64; 3], PlanResult)> = None;
        for sample in 0..RELEASE_EPOCH_REFINE_SAMPLES {
            let sample = u32::try_from(sample)
                .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            let shift = RELEASE_EPOCH_REFINE_HALF * (2.0 * f64::from(sample) / last - 1.0);
            let mut x = [start_time + shift, start_phase, start_wait + shift];
            let [time, _, wait] = x;
            if time < time_lower || time > time_upper || wait < wait_lower || wait > wait_upper {
                continue;
            }
            repair_transfer_decision(&mut x);
            let plan = evaluate_plan_local(&x, ctx, false, cache)?;
            if !transfer_candidate_is_objective_finite(&plan) {
                continue;
            }
            let dv = plan.total_dv();
            if dv >= incumbent_dv - FINAL_CANDIDATE_POLISH_DV_EPS {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(_, current)| dv < current.total_dv())
            {
                best = Some((x, plan));
            }
        }
        let Some((_, plan)) = best else {
            continue;
        };
        // The scan lands in the right basin; the existing final polish is the
        // right tool for the last few metres inside it.
        let polished =
            polish_transfer_candidate_delta_v(&plan, ctx, cache)?.map_or(plan, |polished| polished);
        refined.push(polished);
    }
    for plan in refined {
        push_unique_transfer_candidate(candidates, plan)?;
    }
    Ok(())
}

fn push_delta_v_anchor_probe_candidates(
    candidates: &mut Vec<PlanResult>,
    ctx: &PlanContext,
    cache: &RefCell<SolveLocalWorkCache>,
    center: [f64; 3],
    warm_start_consumed: bool,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    const PROBE_OFFSETS: [[f64; 3]; 4] = [
        [-0.02, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.0, 0.0, -0.02],
        [0.0, 0.0, 0.02],
    ];

    let [center_time, center_phase, center_wait] = center;
    for [time_offset, phase_offset, wait_offset] in PROBE_OFFSETS {
        let mut x = [
            center_time + time_offset,
            center_phase + phase_offset,
            center_wait + wait_offset,
        ];
        repair_transfer_decision(&mut x);
        if transfer_decision_exists(candidates, &x) {
            continue;
        }
        // 7.3 work-count audit: an anchor-stage probe plan evaluation.
        record_work_count(|counters| {
            counters.anchor_probe_evaluations = counters
                .anchor_probe_evaluations
                .checked_add(1)
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            Ok(())
        })?;
        let mut plan = evaluate_plan_local(&x, ctx, false, cache)?;
        if !transfer_candidate_is_objective_finite(&plan) {
            continue;
        }
        plan.func_evals = 0;
        plan.optimizer_func_evals = 0;
        plan.optimizer_converged = false;
        plan.warm_start_used = warm_start_consumed;
        push_unique_transfer_candidate(candidates, plan)?;
    }
    Ok(())
}

const DELTA_V_ANCHOR_SEED_LIMIT_MAX: usize = 3;

#[cfg(feature = "bench-internal")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeltaVAnchorBenchPolicy {
    Full,
    NoProbes,
    CostOnlyNoProbes,
    DvOnlyNoProbes,
    SeedLimit2,
    SeedLimit3,
}

#[cfg(feature = "bench-internal")]
impl From<DeltaVAnchorBenchPolicy> for DeltaVAnchorPolicy {
    fn from(policy: DeltaVAnchorBenchPolicy) -> Self {
        match policy {
            DeltaVAnchorBenchPolicy::Full => Self::Full,
            DeltaVAnchorBenchPolicy::NoProbes => Self::NoProbes,
            DeltaVAnchorBenchPolicy::CostOnlyNoProbes => Self::CostOnlyNoProbes,
            DeltaVAnchorBenchPolicy::DvOnlyNoProbes => Self::DvOnlyNoProbes,
            DeltaVAnchorBenchPolicy::SeedLimit2 => Self::SeedLimit2,
            DeltaVAnchorBenchPolicy::SeedLimit3 => Self::SeedLimit3,
        }
    }
}

impl DeltaVAnchorPolicy {
    #[inline]
    const fn use_cost_anchor(self) -> bool {
        matches!(
            self,
            Self::Full
                | Self::NoProbes
                | Self::CostOnlyNoProbes
                | Self::SeedLimit2
                | Self::SeedLimit3
        )
    }

    #[inline]
    const fn use_delta_v_anchor(self) -> bool {
        matches!(
            self,
            Self::Full
                | Self::NoProbes
                | Self::DvOnlyNoProbes
                | Self::SeedLimit2
                | Self::SeedLimit3
        )
    }

    #[inline]
    const fn use_probes(self) -> bool {
        matches!(self, Self::Full | Self::SeedLimit2 | Self::SeedLimit3)
    }

    #[inline]
    const fn seed_limit(self) -> usize {
        match self {
            Self::SeedLimit2 => 2,
            Self::SeedLimit3 => 3,
            _ => 1,
        }
    }
}

fn push_best_coarse_delta_v_seed_evaluations(
    eligible: &mut Vec<(SolverSeed, PlanResult)>,
    coarse_ranked: &[(SolverSeed, PlanResult)],
    ctx: &PlanContext,
    warm_start_consumed: bool,
    local_cache: &RefCell<SolveLocalWorkCache>,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let mut best: Option<(SolverSeed, f64)> = None;

    for (seed, plan) in coarse_ranked {
        if !transfer_candidate_is_objective_finite(plan) {
            continue;
        }
        if eligible
            .iter()
            .any(|(existing, _)| seed_is_duplicate(&existing.x, &seed.x))
            || best
                .as_ref()
                .is_some_and(|(existing, _)| seed_is_duplicate(&existing.x, &seed.x))
        {
            continue;
        }
        let score = plan.total_dv();
        if !score.is_finite() {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(_, best_score)| score < *best_score)
        {
            best = Some((*seed, score));
        }
    }

    if let Some((seed, _)) = best {
        let mut plan = evaluate_plan_local(&seed.x, ctx, false, local_cache)?;
        plan.warm_start_used = warm_start_consumed;
        eligible.push((seed, plan));
    }
    Ok(())
}

/// Wall-clock components measured in seconds.
#[derive(Clone, Copy, Debug, Default)]
struct SeedRankTiming {
    build: f64,
    coarse_eval: f64,
    fine_eval: f64,
    sort_select: f64,
}

fn provisional_fine_count_for_coarse_early_stop(
    coarse_ranked: &[(SolverSeed, PlanResult)],
    policy: &SearchDepthPolicy,
) -> Result<usize, InvalidTargetPropagationAuthorityCode> {
    // Keep sorted payload as source indices. `PlanResult::cost` can be NaN on
    // malformed physics input; the legacy partial comparator is intentionally
    // retained, and changing the sorted item type could change its comparison
    // schedule in that non-total case.
    let mut order = Vec::new();
    try_reserve_transfer_capacity(&mut order, coarse_ranked.len())?;
    order.extend(0..coarse_ranked.len());
    order.sort_by(|left, right| {
        let left_cost = coarse_ranked
            .get(*left)
            .map_or(f64::INFINITY, |(_, plan)| plan.cost);
        let right_cost = coarse_ranked
            .get(*right)
            .map_or(f64::INFINITY, |(_, plan)| plan.cost);
        left_cost
            .partial_cmp(&right_cost)
            .unwrap_or(Ordering::Equal)
    });
    let cutoff = order.len().min(policy.fine_total_limit);
    let cost_threshold = cutoff
        .checked_sub(1)
        .filter(|_| cutoff < order.len())
        .and_then(|index| order.get(index))
        .and_then(|index| coarse_ranked.get(*index))
        .map_or(f64::INFINITY, |(_, plan)| {
            plan.cost + policy.seed_fine_margin_km_s
        });

    Ok(order
        .iter()
        .take_while(|index| {
            coarse_ranked
                .get(**index)
                .is_some_and(|(_, plan)| plan.cost <= cost_threshold)
        })
        .count())
}

/// Cost cutoff for the sorted fine stage. A missing cutoff means every seed
/// remains eligible, matching the historical no-limit/all-seeds behavior.
fn fine_cost_threshold(
    coarse_ranked: &[(SolverSeed, PlanResult)],
    policy: &SearchDepthPolicy,
) -> f64 {
    let cutoff = coarse_ranked.len().min(policy.fine_total_limit);
    cutoff
        .checked_sub(1)
        .filter(|_| cutoff < coarse_ranked.len())
        .and_then(|index| coarse_ranked.get(index))
        .map_or(f64::INFINITY, |(_, plan)| {
            plan.cost + policy.seed_fine_margin_km_s
        })
}

#[inline]
fn compare_ranked_seeds(
    (left_seed, left_plan): &(SolverSeed, PlanResult),
    (right_seed, right_plan): &(SolverSeed, PlanResult),
) -> Ordering {
    let [left_time, left_phase, left_wait] = left_seed.x;
    let [right_time, right_phase, right_wait] = right_seed.x;
    // Key tuple: (cost, warm_start_used, time, phase, wait) -- cost ascending,
    // warm-started seeds first, then the seed point ascending per component.
    let left = (
        left_plan.cost,
        left_seed.warm_start_used,
        left_time,
        left_phase,
        left_wait,
    );
    let right = (
        right_plan.cost,
        right_seed.warm_start_used,
        right_time,
        right_phase,
        right_wait,
    );
    lex_cmp!(left, right; asc (0), int_desc (1), asc (2), asc (3), asc (4))
}

fn push_recent_coarse_cost(recent_costs: &mut VecDeque<f64>, plan_cost: f64) {
    if recent_costs.len() == 4 {
        let _ = recent_costs.pop_front();
    }
    recent_costs.push_back(plan_cost);
}

fn rank_seed_candidates_for_front(
    ctx: &PlanContext,
    warm_start: Option<&WarmStartData>,
    local_cache: &RefCell<SolveLocalWorkCache>,
) -> Result<
    (Vec<(SolverSeed, PlanResult)>, bool, SeedRankTiming),
    InvalidTargetPropagationAuthorityCode,
> {
    let mut timing = SeedRankTiming::default();
    let seed_build_start = StageTimer::start();
    let seeds = build_single_pair_seeds(ctx, warm_start)?;
    timing.build = seed_build_start.elapsed_s();
    if seeds.is_empty() {
        return Ok((Vec::new(), false, timing));
    }

    let warm_start_consumed = seeds.iter().any(|seed| seed.warm_start_used);
    let mut coarse_ranked = Vec::new();
    try_reserve_transfer_capacity(&mut coarse_ranked, seeds.len())?;
    let mut best_coarse_cost = f64::INFINITY;
    let mut recent_costs = VecDeque::new();
    recent_costs
        .try_reserve(4)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let coarse_eval_start = StageTimer::start();
    for seed in &seeds {
        let plan = evaluate_plan_local(&seed.x, ctx, true, local_cache)?;
        let plan_cost = if plan.valid && plan.cost < INVALID_COST {
            plan.cost
        } else {
            f64::INFINITY
        };
        best_coarse_cost = best_coarse_cost.min(plan_cost);
        push_recent_coarse_cost(&mut recent_costs, plan_cost);
        coarse_ranked.push((
            SolverSeed {
                x: seed.x,
                warm_start_used: seed.warm_start_used,
            },
            plan,
        ));

        if ctx.search_depth.coarse_early_stop
            && ctx.sampling_mode == SamplingMode::Fast
            && coarse_ranked.len() >= 8
            && coarse_ranked.len() % 2 == 0
        {
            let provisional_fine_count =
                provisional_fine_count_for_coarse_early_stop(&coarse_ranked, &ctx.search_depth)?;
            if should_stop_coarse_stage(
                &ctx.search_depth,
                coarse_ranked.len(),
                best_coarse_cost,
                &recent_costs,
                provisional_fine_count,
            ) {
                break;
            }
        }
    }
    timing.coarse_eval = coarse_eval_start.elapsed_s();

    let seed_sort_select_start = StageTimer::start();
    coarse_ranked.sort_by(|(_, left_plan), (_, right_plan)| {
        left_plan
            .cost
            .partial_cmp(&right_plan.cost)
            .unwrap_or(Ordering::Equal)
    });

    let cost_threshold = fine_cost_threshold(&coarse_ranked, &ctx.search_depth);
    timing.sort_select += seed_sort_select_start.elapsed_s();

    let fine_eval_start = StageTimer::start();
    let mut eligible_ranked = Vec::new();
    try_reserve_transfer_capacity(&mut eligible_ranked, coarse_ranked.len())?;
    for (seed, _) in coarse_ranked
        .iter()
        .take_while(|(_, plan)| plan.cost <= cost_threshold)
    {
        let mut plan = evaluate_plan_local(&seed.x, ctx, false, local_cache)?;
        plan.warm_start_used = warm_start_consumed;
        eligible_ranked.push((*seed, plan));
    }
    push_best_coarse_delta_v_seed_evaluations(
        &mut eligible_ranked,
        &coarse_ranked,
        ctx,
        warm_start_consumed,
        local_cache,
    )?;
    timing.fine_eval = fine_eval_start.elapsed_s();

    let seed_sort_select_start = StageTimer::start();
    eligible_ranked.sort_by(|(_, left_plan), (_, right_plan)| {
        left_plan
            .cost
            .partial_cmp(&right_plan.cost)
            .unwrap_or(Ordering::Equal)
    });
    // Every eligible seed remains in rank order. The former index-selection
    // pass started with every index, so its duplicate-aware re-additions could
    // not remove anything; moving this vector avoids that no-op allocation and
    // clone pass.
    let mut ranked = eligible_ranked;

    ranked.sort_by(compare_ranked_seeds);
    timing.sort_select += seed_sort_select_start.elapsed_s();

    Ok((ranked, warm_start_consumed, timing))
}

pub(crate) struct TransferMooWorkspace {
    candidates: Vec<PlanResult>,
    initial_decisions: Vec<f64>,
    seen_decisions: Vec<TransferDecisionKey>,
    plan_cache: FxHashMap<TransferDecisionKey, PlanResult>,
    seed_cache: RefCell<SolveLocalWorkCache>,
    fallback_cache: RefCell<SolveLocalWorkCache>,
}

impl TransferMooWorkspace {
    pub(crate) fn new() -> Self {
        Self {
            candidates: Vec::new(),
            initial_decisions: Vec::new(),
            seen_decisions: Vec::new(),
            plan_cache: FxHashMap::default(),
            seed_cache: RefCell::new(SolveLocalWorkCache::new()),
            fallback_cache: RefCell::new(SolveLocalWorkCache::new()),
        }
    }
}

enum TransferMooCandidateRows<'a> {
    Front(std::slice::Iter<'a, usize>),
    Range(Range<usize>),
}

impl Iterator for TransferMooCandidateRows<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Front(iter) => iter.next().copied(),
            Self::Range(range) => range.next(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OxyMooCandidateMaterializationError {
    ArithmeticOverflow,
    Authority(InvalidTargetPropagationAuthorityCode),
    OptimizerFailure,
    MalformedPopulation,
}

impl std::fmt::Display for OxyMooCandidateMaterializationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArithmeticOverflow => formatter.write_str("OxyMOO candidate arithmetic overflow"),
            Self::Authority(error) => error.fmt(formatter),
            Self::OptimizerFailure => formatter.write_str("OxyMOO optimizer failure"),
            Self::MalformedPopulation => formatter.write_str("OxyMOO population is malformed"),
        }
    }
}

impl std::error::Error for OxyMooCandidateMaterializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Authority(error) => Some(error),
            Self::ArithmeticOverflow | Self::OptimizerFailure | Self::MalformedPopulation => None,
        }
    }
}

fn classify_oxymoo_optimizer_error(error: &anyhow::Error) -> OxyMooCandidateMaterializationError {
    if let Some(error) = error.chain().find_map(|cause| {
        cause
            .downcast_ref::<OxyMooCandidateMaterializationError>()
            .copied()
    }) {
        return error;
    }
    if error
        .chain()
        .any(<dyn std::error::Error + 'static>::is::<crate::oxymoo::ArithmeticOverflow>)
    {
        return OxyMooCandidateMaterializationError::ArithmeticOverflow;
    }
    if let Some(authority) = error.chain().find_map(|cause| {
        cause
            .downcast_ref::<InvalidTargetPropagationAuthorityCode>()
            .copied()
    }) {
        return OxyMooCandidateMaterializationError::Authority(authority);
    }
    OxyMooCandidateMaterializationError::OptimizerFailure
}

fn checked_usize_add(
    current: usize,
    added: usize,
) -> Result<usize, OxyMooCandidateMaterializationError> {
    current
        .checked_add(added)
        .ok_or(OxyMooCandidateMaterializationError::ArithmeticOverflow)
}

fn checked_usize_mul(
    left: usize,
    right: usize,
) -> Result<usize, OxyMooCandidateMaterializationError> {
    left.checked_mul(right)
        .ok_or(OxyMooCandidateMaterializationError::ArithmeticOverflow)
}

fn reserve_transfer_moo_initial_decision_buffers(
    decisions: &mut Vec<f64>,
    seen: &mut Vec<TransferDecisionKey>,
    seed_count: usize,
) -> Result<(), OxyMooCandidateMaterializationError> {
    let decision_capacity = checked_usize_mul(seed_count, 3)?;
    let decision_additional = if decisions.len() < decision_capacity {
        decision_capacity
            .checked_sub(decisions.len())
            .ok_or(OxyMooCandidateMaterializationError::ArithmeticOverflow)?
    } else {
        0
    };
    let key_additional = if seen.len() < seed_count {
        seed_count
            .checked_sub(seen.len())
            .ok_or(OxyMooCandidateMaterializationError::ArithmeticOverflow)?
    } else {
        0
    };
    decisions
        .try_reserve(decision_additional)
        .map_err(|_| OxyMooCandidateMaterializationError::ArithmeticOverflow)?;
    seen.try_reserve(key_additional)
        .map_err(|_| OxyMooCandidateMaterializationError::ArithmeticOverflow)?;
    decisions.clear();
    seen.clear();
    Ok(())
}

fn fill_transfer_moo_initial_decisions(
    pair_hint: Option<[f64; 3]>,
    ranked_seeds: &[(SolverSeed, PlanResult)],
    decisions: &mut Vec<f64>,
    seen: &mut Vec<TransferDecisionKey>,
) -> Result<(), OxyMooCandidateMaterializationError> {
    let hint_count = usize::from(pair_hint.is_some());
    let seed_count = checked_usize_add(ranked_seeds.len(), hint_count)?;
    reserve_transfer_moo_initial_decision_buffers(decisions, seen, seed_count)?;
    if let Some(hint) = pair_hint {
        push_unique_transfer_moo_initial_decision(decisions, seen, hint);
    }
    for (seed, plan) in ranked_seeds {
        if transfer_candidate_is_objective_finite(plan) {
            push_unique_transfer_moo_initial_decision(decisions, seen, seed.x);
        }
    }
    Ok(())
}

#[cfg(test)]
fn transfer_moo_initial_decisions(
    pair_hint: Option<[f64; 3]>,
    ranked_seeds: &[(SolverSeed, PlanResult)],
) -> anyhow::Result<Vec<f64>> {
    let mut decisions = Vec::new();
    let mut seen = Vec::new();
    fill_transfer_moo_initial_decisions(pair_hint, ranked_seeds, &mut decisions, &mut seen)?;
    Ok(decisions)
}

fn push_unique_transfer_moo_initial_decision(
    decisions: &mut Vec<f64>,
    seen: &mut Vec<TransferDecisionKey>,
    decision: [f64; 3],
) {
    let repaired = repaired_transfer_decision(&decision);
    let key = transfer_decision_key(&repaired);
    if seen.contains(&key) {
        return;
    }
    seen.push(key);
    decisions.extend_from_slice(&repaired);
}

fn truncate_transfer_moo_initial_decisions(
    decisions: &mut Vec<f64>,
    decision_limit: Option<usize>,
) -> Result<(), OxyMooCandidateMaterializationError> {
    if let Some(limit) = decision_limit {
        let decision_count = checked_usize_mul(limit, TRANSFER_MOO_VARIABLES.len())?;
        decisions.truncate(decision_count);
    }
    Ok(())
}

fn count_warm_start_initial_decisions(
    ranked_seeds: &[(SolverSeed, PlanResult)],
    decisions: &[f64],
) -> usize {
    ranked_seeds
        .iter()
        .filter(|(seed, plan)| seed.warm_start_used && transfer_candidate_is_objective_finite(plan))
        .filter(|(seed, _)| {
            let repaired = repaired_transfer_decision(&seed.x);
            decisions
                .chunks_exact(TRANSFER_MOO_VARIABLES.len())
                .any(|chunk| {
                    let Ok(&[time, phase, wait]) = <&[f64; 3]>::try_from(chunk) else {
                        return false;
                    };
                    seed_is_duplicate(&repaired, &[time, phase, wait])
                })
        })
        .count()
}

fn preload_retained_transfer_moo_plans(
    problem: &TransferMooProblem,
    ranked_seeds: &[(SolverSeed, PlanResult)],
    decisions: &[f64],
) -> Result<usize, OxyMooCandidateMaterializationError> {
    let mut retained_keys = FxHashSet::default();
    let retained_key_capacity = decisions
        .len()
        .checked_div(TRANSFER_MOO_VARIABLES.len())
        .ok_or(OxyMooCandidateMaterializationError::ArithmeticOverflow)?;
    retained_keys
        .try_reserve(retained_key_capacity)
        .map_err(|_| OxyMooCandidateMaterializationError::ArithmeticOverflow)?;
    for chunk in decisions.chunks_exact(TRANSFER_MOO_VARIABLES.len()) {
        retained_keys.insert(transfer_decision_key(&repaired_transfer_decision(chunk)));
    }
    let preloaded = ranked_seeds
        .iter()
        .try_fold(0usize, |count, (seed, plan)| {
            let x = repaired_transfer_decision(&seed.x);
            let key = transfer_decision_key(&x);
            if transfer_candidate_is_objective_finite(plan) && retained_keys.contains(&key) {
                checked_usize_add(count, 1)
            } else {
                Ok(count)
            }
        })?;
    for (seed, plan) in ranked_seeds {
        let x = repaired_transfer_decision(&seed.x);
        let key = transfer_decision_key(&x);
        if transfer_candidate_is_objective_finite(plan) && retained_keys.contains(&key) {
            problem
                .preload_plan(key, &x, plan.clone())
                .map_err(OxyMooCandidateMaterializationError::Authority)?;
        }
    }
    Ok(preloaded)
}

fn run_transfer_moo_optimizer(
    optimizer: Nsga2<TransferMooProblem>,
    policy: TransferMooPolicy,
) -> anyhow::Result<(TransferMooProblem, Nsga2Result)> {
    #[cfg(feature = "bench-internal")]
    {
        if policy.use_stable_objective_stop() {
            return run_transfer_moo_optimizer_with_stable_objective_stop(optimizer);
        }
    }
    #[cfg(not(feature = "bench-internal"))]
    let _ = policy;
    optimizer.run_owned_with_problem()
}

#[cfg(feature = "bench-internal")]
fn run_transfer_moo_optimizer_with_stable_objective_stop(
    mut optimizer: Nsga2<TransferMooProblem>,
) -> anyhow::Result<(TransferMooProblem, Nsga2Result)> {
    let mut previous = first_front_objective_signature(&optimizer)?;
    let mut stable_count = 0usize;
    for _ in 0..5 {
        optimizer.step()?;
        let current = first_front_objective_signature(&optimizer)?;
        if !current.is_empty() && current == previous {
            stable_count = checked_usize_add(stable_count, 1)?;
            if stable_count >= 2 {
                return Ok(optimizer.into_problem_and_result());
            }
        } else {
            stable_count = 0;
        }
        previous = current;
    }
    Ok(optimizer.into_problem_and_result())
}

#[cfg(feature = "bench-internal")]
fn first_front_objective_signature(
    optimizer: &Nsga2<TransferMooProblem>,
) -> anyhow::Result<Vec<([u64; 2], u64)>> {
    let population = optimizer.population();
    let Some(first_front) = optimizer.fronts().first() else {
        return Ok(Vec::new());
    };
    let mut signature = Vec::with_capacity(first_front.len());
    for &row in first_front {
        if row >= population.len() {
            return Err(anyhow::anyhow!(
                "OxyMOO front row {row} is outside the population"
            ));
        }
        let objectives = population.objectives(row)?;
        let (Some(&objective_0), Some(&objective_1)) = (objectives.first(), objectives.get(1))
        else {
            continue;
        };
        let constraint_violation = *population.constraint_violations.get(row).ok_or_else(|| {
            anyhow::anyhow!("population constraint-violation row {row} is missing")
        })?;
        signature.push((
            [objective_0.to_bits(), objective_1.to_bits()],
            constraint_violation.to_bits(),
        ));
    }
    signature.sort_unstable();
    Ok(signature)
}

fn repaired_population_transfer_decision(
    population: &crate::oxymoo::PopulationSnapshot,
    row: usize,
) -> Result<[f64; 3], OxyMooCandidateMaterializationError> {
    let decision = population
        .decision(row)
        .map_err(|_| OxyMooCandidateMaterializationError::MalformedPopulation)?;
    let decision: &[f64; 3] = decision
        .try_into()
        .map_err(|_| OxyMooCandidateMaterializationError::MalformedPopulation)?;
    Ok(repaired_transfer_decision(decision))
}

fn push_oxymoo_transfer_candidates(
    out: &mut Vec<PlanResult>,
    ctx: &PlanContext,
    pair_hint: Option<[f64; 3]>,
    ranked_seeds: &[(SolverSeed, PlanResult)],
    warm_start_consumed: bool,
    front_output_mode: FrontOutputMode,
    workspace: &mut TransferMooWorkspace,
    policy: TransferMooPolicy,
    metrics: Option<&mut VerifiedSupersetStageMetrics>,
) -> Result<(), OxyMooCandidateMaterializationError> {
    let mut metrics = metrics;
    let (population_size, generations) = policy.population_generations();
    fill_transfer_moo_initial_decisions(
        pair_hint,
        ranked_seeds,
        &mut workspace.initial_decisions,
        &mut workspace.seen_decisions,
    )?;
    truncate_transfer_moo_initial_decisions(
        &mut workspace.initial_decisions,
        policy.initial_decision_limit(),
    )?;
    let mut map = std::mem::take(&mut workspace.plan_cache);
    map.clear();
    let batch_count = checked_usize_add(generations, 1)?;
    let cache_reservation =
        checked_usize_mul(population_size, batch_count)?.min(TRANSFER_MOO_PLAN_CACHE_MAX_ENTRIES);
    map.try_reserve(cache_reservation)
        .map_err(|_| OxyMooCandidateMaterializationError::ArithmeticOverflow)?;
    let plan_cache: TransferMooPlanCache = RefCell::new(map);
    let problem = TransferMooProblem::new(ctx.clone(), plan_cache, policy)
        .map_err(OxyMooCandidateMaterializationError::Authority)?;
    let _preloaded =
        preload_retained_transfer_moo_plans(&problem, ranked_seeds, &workspace.initial_decisions)?;
    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.warm_start_oxymoo_initial_count = checked_usize_add(
            metrics.warm_start_oxymoo_initial_count,
            count_warm_start_initial_decisions(ranked_seeds, &workspace.initial_decisions),
        )?;
    }
    let initial_decisions = std::mem::take(&mut workspace.initial_decisions);
    let transfer_moo_config = transfer_moo_config_with_policy(ctx, initial_decisions, policy)
        .map_err(OxyMooCandidateMaterializationError::Authority)?;
    // 7.4 work-count audit: the `evaluate_batch` override classifies each batch
    // as parallel or serial on this (front-solve) thread. Reset the tally before
    // `Nsga2::new` — which evaluates the initial population batch — so the
    // initial batch plus every generation batch is attributed to the metrics.
    reset_oxymoo_batch_class();
    let optimizer = match Nsga2::new(problem, transfer_moo_config) {
        Ok(optimizer) => optimizer,
        Err(error) => return Err(classify_oxymoo_optimizer_error(&error)),
    };
    let nsga_start = StageTimer::start();
    let (problem, result) = match run_transfer_moo_optimizer(optimizer, policy) {
        Ok(problem_result) => problem_result,
        Err(error) => return Err(classify_oxymoo_optimizer_error(&error)),
    };
    let oxymoo_batch_class = oxymoo_batch_class_snapshot();
    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.nsga_run_s += nsga_start.elapsed_s();
    }

    workspace.fallback_cache.borrow_mut().clear();
    let candidate_rows = match front_output_mode {
        FrontOutputMode::TransferPareto => {
            let Some(first_front) = result.fronts.first() else {
                workspace.plan_cache = problem.into_plan_cache().into_inner();
                workspace.plan_cache.clear();
                return Err(OxyMooCandidateMaterializationError::OptimizerFailure);
            };
            TransferMooCandidateRows::Front(first_front.iter())
        }
        FrontOutputMode::VerifiedSuperset => {
            TransferMooCandidateRows::Range(0..result.population.len())
        }
    };

    let mut verified_seen_decisions = FxHashSet::default();
    verified_seen_decisions
        .try_reserve(result.population.len())
        .map_err(|_| OxyMooCandidateMaterializationError::ArithmeticOverflow)?;
    out.try_reserve(result.population.len())
        .map_err(|_| OxyMooCandidateMaterializationError::ArithmeticOverflow)?;
    let source_materialization_enabled = oxymoo_source_materialization_enabled(front_output_mode);
    let mut materialize_plan_cache_hit_count = 0usize;
    let mut materialize_plan_cache_miss_count = 0usize;
    let mut materialize_recompute_count = 0usize;
    let materialize_start = StageTimer::start();
    for row in candidate_rows {
        if row >= result.population.len() {
            return Err(OxyMooCandidateMaterializationError::MalformedPopulation);
        }
        let constraint_violation = *result
            .population
            .constraint_violations
            .get(row)
            .ok_or(OxyMooCandidateMaterializationError::MalformedPopulation)?;
        if constraint_violation > 0.0 {
            continue;
        }
        let x = repaired_population_transfer_decision(&result.population, row)?;
        let key = transfer_decision_key(&x);
        if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset)
            && !verified_seen_decisions.insert(key)
        {
            continue;
        }

        let source_plan = if source_materialization_enabled {
            let cached = problem.take_cached_source_plan(key);
            match cached {
                Some(plan) if transfer_candidate_is_objective_finite(&plan) => {
                    if metrics.is_some() {
                        materialize_plan_cache_hit_count =
                            checked_usize_add(materialize_plan_cache_hit_count, 1)?;
                    }
                    Some(plan)
                }
                _ => {
                    if metrics.is_some() {
                        materialize_plan_cache_miss_count =
                            checked_usize_add(materialize_plan_cache_miss_count, 1)?;
                    }
                    None
                }
            }
        } else {
            None
        };

        // Verified-superset materialization can now carry the exact finite
        // source plans evaluated by OxyMOO. Keep the old recompute path only as
        // a miss fallback and for TransferPareto, where historical trust-canary
        // evidence made cache-content sensitivity too risky.
        let plan = if let Some(plan) = source_plan {
            plan
        } else {
            if metrics.is_some() {
                materialize_recompute_count = checked_usize_add(materialize_recompute_count, 1)?;
            }
            evaluate_plan_local(&x, ctx, false, &workspace.fallback_cache)
                .map_err(OxyMooCandidateMaterializationError::Authority)?
        };
        if !transfer_candidate_is_objective_finite(&plan) {
            continue;
        }
        let mut plan = plan;
        let evaluation_count = u64::try_from(result.evaluations)
            .map_err(|_| OxyMooCandidateMaterializationError::ArithmeticOverflow)?;
        plan.func_evals = evaluation_count;
        plan.optimizer_func_evals = evaluation_count;
        plan.optimizer_converged = result.generations > 0;
        plan.warm_start_used = warm_start_consumed;
        out.push(plan);
    }
    if let Some(metrics) = metrics {
        metrics.nsga_materialize_s += materialize_start.elapsed_s();
        metrics.nsga_materialize_plan_cache_hit_count = checked_usize_add(
            metrics.nsga_materialize_plan_cache_hit_count,
            materialize_plan_cache_hit_count,
        )?;
        metrics.nsga_materialize_plan_cache_miss_count = checked_usize_add(
            metrics.nsga_materialize_plan_cache_miss_count,
            materialize_plan_cache_miss_count,
        )?;
        metrics.nsga_materialize_recompute_count = checked_usize_add(
            metrics.nsga_materialize_recompute_count,
            materialize_recompute_count,
        )?;
        // 7.3/7.4 work-count audit: OxyMOO NSGA-II eval accounting. Every
        // `Problem::evaluate`-equivalent row is either an eval-cache hit or a
        // miss; a miss forces one full objective evaluation, so misses == the
        // full-eval count and hits + misses == the total generation evals
        // (population x (1 init + generations)). The parallel batch path (7.4)
        // preserves that split via an intra-batch duplicate pre-scan, and
        // resolves the cache serially, so these counts are identical whether the
        // batches ran serially or fanned out across the rayon pool. Each of the
        // (1 init + generations) batches is classified parallel or serial by the
        // `evaluate_batch` override; the two counts sum to `generations + 1`.
        let oxymoo_full_evals = problem.eval_cache_misses();
        metrics.oxymoo_eval_cache_hit_count = checked_usize_add(
            metrics.oxymoo_eval_cache_hit_count,
            problem.eval_cache_hits(),
        )?;
        metrics.oxymoo_eval_cache_miss_count =
            checked_usize_add(metrics.oxymoo_eval_cache_miss_count, oxymoo_full_evals)?;
        metrics.oxymoo_full_eval_count =
            checked_usize_add(metrics.oxymoo_full_eval_count, oxymoo_full_evals)?;
        metrics.oxymoo_parallel_batch_count = checked_usize_add(
            metrics.oxymoo_parallel_batch_count,
            oxymoo_batch_class.parallel,
        )?;
        metrics.oxymoo_serial_batch_count =
            checked_usize_add(metrics.oxymoo_serial_batch_count, oxymoo_batch_class.serial)?;
        if source_materialization_enabled
            && materialize_plan_cache_hit_count > 0
            && materialize_plan_cache_miss_count == 0
            && materialize_recompute_count == 0
        {
            metrics.nsga_materialize_all_exact_count =
                checked_usize_add(metrics.nsga_materialize_all_exact_count, 1)?;
        }
    }
    workspace.plan_cache = problem.into_plan_cache().into_inner();
    workspace.plan_cache.clear();
    Ok(())
}

#[cfg(test)]
fn run_oxymoo_transfer_candidates(
    ctx: &PlanContext,
    warm_start_consumed: bool,
) -> anyhow::Result<Vec<PlanResult>> {
    let (population_size, _) = transfer_moo_population_generations();
    let mut out = Vec::with_capacity(population_size);
    let mut workspace = TransferMooWorkspace::new();
    push_oxymoo_transfer_candidates(
        &mut out,
        ctx,
        None,
        &[],
        warm_start_consumed,
        FrontOutputMode::TransferPareto,
        &mut workspace,
        TransferMooPolicy::Full,
        None,
    )?;
    Ok(out)
}

#[inline]
const fn checked_stage_metric_count_add(
    current: &mut usize,
    incoming: usize,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let Some(sum) = current.checked_add(incoming) else {
        return Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow);
    };
    *current = sum;
    Ok(())
}

#[inline]
fn checked_stage_metric_count_delta(
    after: usize,
    before: usize,
) -> Result<usize, InvalidTargetPropagationAuthorityCode> {
    after
        .checked_sub(before)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
}

#[inline]
fn try_reserve_transfer_capacity<T>(
    values: &mut Vec<T>,
    additional: usize,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    values
        .try_reserve(additional)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
}

#[inline]
fn oxymoo_candidate_capacity(
    ranked_seed_count: usize,
    population_size: usize,
    delta_v_anchor_policy: DeltaVAnchorPolicy,
) -> Result<usize, InvalidTargetPropagationAuthorityCode> {
    let anchor_capacity = delta_v_anchor_policy
        .seed_limit()
        .checked_add(1)
        .and_then(|count| count.checked_mul(5))
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    ranked_seed_count
        .checked_add(population_size)
        .and_then(|count| count.checked_add(anchor_capacity))
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
}

const fn handle_oxymoo_candidate_materialization_result(
    result: Result<(), OxyMooCandidateMaterializationError>,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    match result {
        Ok(()) => Ok(()),
        Err(OxyMooCandidateMaterializationError::ArithmeticOverflow) => {
            Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
        }
        Err(OxyMooCandidateMaterializationError::Authority(error)) => Err(error),
        Err(
            OxyMooCandidateMaterializationError::OptimizerFailure
            | OxyMooCandidateMaterializationError::MalformedPopulation,
        ) => Err(InvalidTargetPropagationAuthorityCode::OptimizerFailure),
    }
}

fn verified_superset_deterministic_grid_fallback(
    ctx: &mut PlanContext,
    warm_start_consumed: bool,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    prepare_single_pair_context(ctx);
    // The grid points (TIME x PHASE x WAIT in nested order, physically-excluded
    // corner already skipped) are the compile-time DETERMINISTIC_GRID_POINTS
    // table. Both the serial and parallel paths consume this exact ordered
    // list, so their per-point branch pushes land in the identical grid-index
    // sequence.
    let grid_points: &[[f64; 3]] = &DETERMINISTIC_GRID_POINTS;
    let capacity = branch_expansion_capacity(grid_points.len(), ctx.max_revs)?;

    let mut candidates = Vec::new();
    try_reserve_transfer_capacity(&mut candidates, capacity)?;
    {
        if should_use_deterministic_grid_parallel(ctx, grid_points.len()) {
            deterministic_grid_fallback_candidates_parallel(
                ctx,
                grid_points,
                warm_start_consumed,
                &mut candidates,
            )?;
        } else {
            deterministic_grid_fallback_candidates_serial(
                ctx,
                grid_points,
                warm_start_consumed,
                &mut candidates,
            )?;
        }
    }
    finalize_verified_superset(ctx, &mut candidates)
}

/// Serial reference for the deterministic grid fallback: one shared Lambert
/// scratch, per-grid-point branch evaluation, in grid-index-order pushes with
/// the grid-fallback field overrides. This is the byte-identical reference the
/// 7.4 parallel path reproduces.
fn deterministic_grid_fallback_candidates_serial(
    ctx: &PlanContext,
    grid_points: &[[f64; 3]],
    warm_start_consumed: bool,
    candidates: &mut Vec<PlanResult>,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    // rust-alloc#2: one Lambert scratch for the whole grid-fallback loop.
    let mut lambert_scratch = crate::lambert::VariableR2LambertScratch::default();
    for x in grid_points {
        let branch_plans = map_evaluation_arithmetic_overflow(
            evaluate_plan_branches_with_scratch(x, ctx, false, &mut lambert_scratch),
        )?;
        for mut plan in branch_plans {
            plan.func_evals = 0;
            plan.optimizer_func_evals = 0;
            plan.optimizer_converged = false;
            plan.warm_start_used = warm_start_consumed;
            candidates.push(plan);
        }
    }
    Ok(())
}

/// 7.4: minimum grid-point count below which fanning the grid fallback out is
/// not worth the rayon dispatch. The single-pair fallback grid is 4x4x4 minus a
/// few excluded corners, so it always clears this when it fires.
const DETERMINISTIC_GRID_PARALLEL_MIN_POINTS: usize = 2;

#[cfg(test)]
thread_local! {
    static DETERMINISTIC_GRID_PARALLEL_PATH_HITS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
fn reset_deterministic_grid_parallel_path_hits() {
    DETERMINISTIC_GRID_PARALLEL_PATH_HITS.with(|hits| hits.set(0));
}

#[cfg(test)]
fn deterministic_grid_parallel_path_hits() -> usize {
    DETERMINISTIC_GRID_PARALLEL_PATH_HITS.with(Cell::get)
}

#[cfg(test)]
fn record_deterministic_grid_parallel_path_hit() -> Result<(), InvalidTargetPropagationAuthorityCode>
{
    DETERMINISTIC_GRID_PARALLEL_PATH_HITS.with(|hits| {
        let next = hits
            .get()
            .checked_add(1)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        hits.set(next);
        Ok(())
    })
}

/// Runtime gate for deterministic-grid fallback. Top-level multi-thread calls
/// use global rayon fan-out; calls already on a rayon worker stay leaf-serial.
fn should_use_deterministic_grid_parallel(ctx: &PlanContext, point_count: usize) -> bool {
    should_use_leaf_parallel(
        ctx.execution_policy.allow_deterministic_grid_parallel,
        point_count,
        DETERMINISTIC_GRID_PARALLEL_MIN_POINTS,
        rayon::current_num_threads(),
        rayon::current_thread_index().is_none(),
    )
}

/// One selected pair's front plus the counter contribution its worker made,
/// carried out of the pool so the reduction owns it in slot order.
struct SelectedPairResult {
    front: Option<PairFrontResult>,
    diag_delta: EvaluationDiagnosticCounters,
    work_delta: WorkCountCounters,
}

struct GridPointResult {
    branch_plans: Vec<PlanResult>,
    diag_delta: EvaluationDiagnosticCounters,
    work_delta: WorkCountCounters,
}

/// 7.4 parallel deterministic grid fallback: evaluate each grid point's Lambert
/// branch plans across the rayon pool with a per-worker Lambert scratch (bounded
/// memory: scratch per worker, not per point), then replay the field overrides
/// and pushes in serial grid-index order so the candidate `Vec` and every
/// INTEGER work/diagnostic counter are bit-identical to the serial reference.
///
/// Per-point evaluation is pure given `ctx` (the `_with_scratch` Lambert batch
/// entries clear every scratch field at entry, exactly as the serial single
/// shared scratch relies on), so the ordered flatten reproduces the serial push
/// order. Each point captures its own diagnostic/work counter contribution; the
/// front-solve thread folds them in grid-index order after the join, keeping
/// integer counts exact and the caller's around-fallback snapshot diff
/// (`deterministic_fallback_full_eval_count`) accurate.
///
/// The f64 diagnostic fields (`j2_correction_residual_m_sum` in metres, the
/// `*_s` sub-timers) are reduction-order deterministic — schedule-independent,
/// fixed by grid index — but NOT bit-identical to the serial reference. Serial
/// accumulates every term into one running sum; this folds per-point sums.
/// Reduction ORDER is the same either way; the GROUPING is not, and `+` on f64
/// is not associative. Do not restructure the reduction to close that gap: the
/// property worth having is determinism, which this already has.
fn deterministic_grid_fallback_candidates_parallel(
    ctx: &PlanContext,
    grid_points: &[[f64; 3]],
    warm_start_consumed: bool,
    candidates: &mut Vec<PlanResult>,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    #[cfg(test)]
    record_deterministic_grid_parallel_path_hit()?;

    // PASS 1 (parallel): per-point branch physics off-thread with a per-worker
    // Lambert scratch and no shared mutable state. Each closure restores its
    // thread-local baselines on every exit, then the reduction owns its captured
    // delta. That includes a Rayon task run by this caller thread.
    let mut computed = Vec::new();
    try_reserve_transfer_capacity(&mut computed, grid_points.len())?;
    grid_points
        .par_iter()
        .map_init(
            crate::lambert::VariableR2LambertScratch::default,
            |scratch, x| {
                with_isolated_diag_region(|| {
                    map_evaluation_arithmetic_overflow(evaluate_plan_branches_with_scratch(
                        x, ctx, false, scratch,
                    ))
                })
                .map(|(branch_plans, diag_delta, work_delta)| GridPointResult {
                    branch_plans,
                    diag_delta,
                    work_delta,
                })
            },
        )
        .collect_into_vec(&mut computed);

    // PASS 2 (serial, grid-index order): reduce the per-point counter deltas and
    // replay the serial field overrides / pushes exactly.
    let mut diag_total = EvaluationDiagnosticCounters::default();
    let mut work_total = WorkCountCounters::default();
    for result in computed {
        let result = result?;
        map_evaluation_arithmetic_overflow(diag_total.add_delta(&result.diag_delta))?;
        work_total.add_delta(result.work_delta)?;
        for mut plan in result.branch_plans {
            plan.func_evals = 0;
            plan.optimizer_func_evals = 0;
            plan.optimizer_converged = false;
            plan.warm_start_used = warm_start_consumed;
            candidates.push(plan);
        }
    }
    map_evaluation_arithmetic_overflow(crate::evaluate::merge_evaluation_diagnostics(&diag_total))?;
    merge_work_counts(work_total)?;
    Ok(())
}

#[cfg(test)]
fn solve_plan_oxymoo_front(
    ctx: &mut PlanContext,
    warm_start: Option<&WarmStartData>,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    let transfer_moo_policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let delta_v_anchor_policy = ctx.search_depth.delta_v_anchor_policy;
    solve_plan_oxymoo_front_internal(
        ctx,
        warm_start,
        None,
        FrontOutputMode::TransferPareto,
        None,
        delta_v_anchor_policy,
        transfer_moo_policy,
    )
}

#[inline]
fn solve_plan_oxymoo_front_with_pair_workspace(
    ctx: &mut PlanContext,
    warm_start: Option<&WarmStartData>,
    pair_hint: Option<[f64; 3]>,
    workspace: &mut TransferMooWorkspace,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    let transfer_moo_policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let delta_v_anchor_policy = ctx.search_depth.delta_v_anchor_policy;
    solve_plan_oxymoo_front_internal(
        ctx,
        warm_start,
        pair_hint,
        FrontOutputMode::TransferPareto,
        Some(workspace),
        delta_v_anchor_policy,
        transfer_moo_policy,
    )
}

#[inline]
fn solve_plan_oxymoo_verified_superset_with_pair_workspace(
    ctx: &mut PlanContext,
    warm_start: Option<&WarmStartData>,
    pair_hint: Option<[f64; 3]>,
    workspace: &mut TransferMooWorkspace,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    let transfer_moo_policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let delta_v_anchor_policy = ctx.search_depth.delta_v_anchor_policy;
    solve_plan_oxymoo_front_internal(
        ctx,
        warm_start,
        pair_hint,
        FrontOutputMode::VerifiedSuperset,
        Some(workspace),
        delta_v_anchor_policy,
        transfer_moo_policy,
    )
}

struct OxyMooFrontState {
    candidates: Vec<PlanResult>,
    ranked_seeds: Vec<(SolverSeed, PlanResult)>,
    warm_start_consumed: bool,
    metrics: VerifiedSupersetStageMetrics,
    /// Totals displaced by the front's open diagnostic region, `Some` exactly
    /// when the region is open. The region's own work is read straight off the
    /// (zeroed) thread-local at the exits below, so
    /// `VerifiedSupersetStageMetrics::j2_correction_residual_m_sum` -- metres,
    /// and the only reported physics number this module derives from telemetry
    /// -- is this front's exact sum rather than a difference against the
    /// thread's whole-campaign history.
    eval_diag_outer: Option<EvaluationDiagnosticCounters>,
}

struct OxyMooPolishPass {
    stats: PolishScopeStats,
    scope_snapshot: Option<Vec<PlanResult>>,
}

fn prepare_oxymoo_front_state(
    ctx: &mut PlanContext,
    warm_start: Option<&WarmStartData>,
    workspace: &mut TransferMooWorkspace,
    front_output_mode: FrontOutputMode,
    delta_v_anchor_policy: DeltaVAnchorPolicy,
) -> Result<OxyMooFrontState, InvalidTargetPropagationAuthorityCode> {
    let prepare_start = StageTimer::start();
    prepare_single_pair_context(ctx);
    let prepare_single_pair_context_s = prepare_start.elapsed_s();
    workspace.seed_cache.borrow_mut().clear();
    let seed_rank_start = StageTimer::start();
    let (ranked_seeds, warm_start_consumed, seed_timing) =
        rank_seed_candidates_for_front(ctx, warm_start, &workspace.seed_cache)?;
    let seed_rank_s = seed_rank_start.elapsed_s();

    let (population_size, _) = transfer_moo_population_generations();
    let candidate_capacity =
        oxymoo_candidate_capacity(ranked_seeds.len(), population_size, delta_v_anchor_policy)?;
    let mut candidates = std::mem::take(&mut workspace.candidates);
    candidates.clear();
    try_reserve_transfer_capacity(&mut candidates, candidate_capacity)?;
    for (_, candidate) in &ranked_seeds {
        let mut candidate = candidate.clone();
        candidate.func_evals = 0;
        candidate.optimizer_func_evals = 0;
        candidate.optimizer_converged = false;
        candidate.warm_start_used = warm_start_consumed;
        candidates.push(candidate);
    }

    let mut metrics = VerifiedSupersetStageMetrics::default();
    let eval_diag_outer = if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        Some(enter_evaluation_diagnostic_region())
    } else {
        None
    };
    if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        metrics.prepare_single_pair_context_s = prepare_single_pair_context_s;
        metrics.seed_rank_s = seed_rank_s;
        metrics.seed_build_s = seed_timing.build;
        metrics.seed_coarse_eval_s = seed_timing.coarse_eval;
        metrics.seed_fine_eval_s = seed_timing.fine_eval;
        metrics.seed_sort_select_s = seed_timing.sort_select;
        metrics.warm_start_received_count =
            usize::from(warm_start.is_some_and(|seed| seed.valid && seed.cost.is_finite()));
        metrics.warm_start_pair_match_count = usize::from(warm_start_consumed);
        metrics.warm_start_seed_consumed_count = usize::from(warm_start_consumed);
        metrics.warm_start_fine_seed_selected_count = ranked_seeds
            .iter()
            .filter(|(seed, _)| seed.warm_start_used)
            .count();
    }
    Ok(OxyMooFrontState {
        candidates,
        ranked_seeds,
        warm_start_consumed,
        metrics,
        eval_diag_outer,
    })
}

fn materialize_oxymoo_front_candidates(
    state: &mut OxyMooFrontState,
    ctx: &PlanContext,
    pair_hint: Option<[f64; 3]>,
    workspace: &mut TransferMooWorkspace,
    front_output_mode: FrontOutputMode,
    delta_v_anchor_policy: DeltaVAnchorPolicy,
    transfer_moo_policy: TransferMooPolicy,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let delta_v_anchor_start = StageTimer::start();
    // Difference the per-thread work tally across anchor materialization.
    // Worker deltas have already been folded by the parallel anchor path.
    let anchor_work_before = work_count_snapshot();
    reset_anchor_parallel_count();
    push_delta_v_anchor_candidates(
        &mut state.candidates,
        ctx,
        &state.ranked_seeds,
        state.warm_start_consumed,
        &workspace.seed_cache,
        delta_v_anchor_policy,
    )?;
    if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        let anchor_work = work_count_snapshot();
        state.metrics.delta_v_anchor_s = delta_v_anchor_start.elapsed_s();
        state.metrics.anchor_full_eval_count = checked_stage_metric_count_delta(
            anchor_work.plan_full_evaluations,
            anchor_work_before.plan_full_evaluations,
        )?;
        state.metrics.anchor_nm_run_count = checked_stage_metric_count_delta(
            anchor_work.anchor_nm_runs,
            anchor_work_before.anchor_nm_runs,
        )?;
        state.metrics.anchor_nm_iteration_count = checked_stage_metric_count_delta(
            anchor_work.anchor_nm_iterations,
            anchor_work_before.anchor_nm_iterations,
        )?;
        state.metrics.anchor_probe_eval_count = checked_stage_metric_count_delta(
            anchor_work.anchor_probe_evaluations,
            anchor_work_before.anchor_probe_evaluations,
        )?;
        state.metrics.anchor_parallel_count = anchor_parallel_count_snapshot();
        state.metrics.pre_oxymoo_candidate_count = state.candidates.len();
    }

    let oxymoo_start = StageTimer::start();
    let materialization = push_oxymoo_transfer_candidates(
        &mut state.candidates,
        ctx,
        pair_hint,
        &state.ranked_seeds,
        state.warm_start_consumed,
        front_output_mode,
        workspace,
        transfer_moo_policy,
        if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
            Some(&mut state.metrics)
        } else {
            None
        },
    );
    handle_oxymoo_candidate_materialization_result(materialization)?;
    if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        state.metrics.oxymoo_s = oxymoo_start.elapsed_s();
        state.metrics.post_oxymoo_candidate_count = state.candidates.len();
    }
    Ok(())
}

fn polish_oxymoo_front_candidates(
    state: &mut OxyMooFrontState,
    ctx: &PlanContext,
    workspace: &TransferMooWorkspace,
    front_output_mode: FrontOutputMode,
) -> Result<OxyMooPolishPass, InvalidTargetPropagationAuthorityCode> {
    let polish_start = StageTimer::start();
    let polish_candidate_count = state.candidates.len();
    let polish_work_before = work_count_snapshot();
    let want_pre_polish_snapshot =
        !matches!(
            ctx.search_depth.polish_scope_policy,
            PolishScopePolicy::Full
        ) && matches!(front_output_mode, FrontOutputMode::VerifiedSuperset);
    let (polish_stats, polish_scope_snapshot) =
        polish_transfer_candidates_delta_v_with_pre_polish_snapshot(
            &mut state.candidates,
            ctx,
            &workspace.seed_cache,
            ctx.search_depth.polish_scope_policy,
            want_pre_polish_snapshot,
        )?;
    if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        state.metrics.polish_s = polish_start.elapsed_s();
        state.metrics.polish_candidate_count = polish_candidate_count;
        state.metrics.polish_scope_skipped_count = polish_stats.scope_skipped_count;
        state.metrics.polish_dv_improvement_max_km_s = polish_stats.dv_improvement_max_km_s;
        state.metrics.polish_parallel_count = polish_stats.polish_parallel_count;
        state.metrics.polish_full_eval_count = checked_stage_metric_count_delta(
            work_count_snapshot().plan_full_evaluations,
            polish_work_before.plan_full_evaluations,
        )?;
    }

    refine_release_epoch_line(&mut state.candidates, ctx, &workspace.seed_cache)?;
    Ok(OxyMooPolishPass {
        stats: polish_stats,
        scope_snapshot: polish_scope_snapshot,
    })
}

fn finalize_oxymoo_front_primary(
    state: &mut OxyMooFrontState,
    ctx: &PlanContext,
    front_output_mode: FrontOutputMode,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        let branch_start = StageTimer::start();
        let branch_work_before = work_count_snapshot();
        let (expanded, branch_eval_s) = expand_lambert_branch_candidates_for_superset(
            ctx,
            std::mem::take(&mut state.candidates),
            &mut state.metrics,
        )?;
        state.candidates = expanded;
        state.metrics.branch_expand_s = branch_start.elapsed_s();
        state.metrics.branch_eval_s = branch_eval_s;
        state.metrics.branch_full_eval_count = checked_stage_metric_count_delta(
            work_count_snapshot().plan_full_evaluations,
            branch_work_before.plan_full_evaluations,
        )?;
        state.metrics.post_branch_candidate_count = state.candidates.len();
    }

    let finalize_start = StageTimer::start();
    let verified_front = match front_output_mode {
        FrontOutputMode::TransferPareto => finalize_verified_front(ctx, &mut state.candidates)?,
        FrontOutputMode::VerifiedSuperset => finalize_verified_superset_with_metrics(
            ctx,
            &mut state.candidates,
            Some(&mut state.metrics),
        )?,
    };
    if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        state.metrics.finalize_s = finalize_start.elapsed_s();
        state.metrics.post_finalize_candidate_count = verified_front.len();
    }
    Ok(verified_front)
}

fn apply_oxymoo_scoped_polish_fallback(
    state: &mut OxyMooFrontState,
    ctx: &PlanContext,
    workspace: &TransferMooWorkspace,
    polish_pass: &mut OxyMooPolishPass,
    verified_front: &mut TransferFront,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    if verified_front.len() < 2 && polish_pass.stats.scope_skipped_count > 0 {
        if let Some(mut fallback_pool) = polish_pass.scope_snapshot.take() {
            let fallback_start = StageTimer::start();
            let fallback_work_before = work_count_snapshot();
            // Re-polish the whole pre-polish pool at Full scope. The scoped pass
            // skipped candidates the front now turns out to need, and polish is a
            // pure function of the pre-polish candidate, so the candidates the
            // scoped pass did polish come back bit-identical.
            let fallback_stats = polish_transfer_candidates_delta_v(
                &mut fallback_pool,
                ctx,
                &workspace.seed_cache,
                PolishScopePolicy::Full,
            )?;
            state.candidates = fallback_pool;
            let mut fallback_metrics = VerifiedSupersetStageMetrics::default();
            let (expanded, _) = expand_lambert_branch_candidates_for_superset(
                ctx,
                std::mem::take(&mut state.candidates),
                &mut fallback_metrics,
            )?;
            state.candidates = expanded;
            *verified_front = finalize_verified_superset_with_metrics(
                ctx,
                &mut state.candidates,
                Some(&mut fallback_metrics),
            )?;
            state.metrics.polish_scope_fallback_count = 1;
            state.metrics.polish_scope_fallback_s = fallback_start.elapsed_s();
            state.metrics.polish_scope_fallback_full_eval_count = checked_stage_metric_count_delta(
                work_count_snapshot().plan_full_evaluations,
                fallback_work_before.plan_full_evaluations,
            )?;
            state.metrics.polish_dv_improvement_max_km_s = state
                .metrics
                .polish_dv_improvement_max_km_s
                .max(fallback_stats.dv_improvement_max_km_s);
            state.metrics.post_finalize_candidate_count = verified_front.len();
        }
    }
    Ok(())
}

fn solve_plan_oxymoo_front_internal(
    ctx: &mut PlanContext,
    warm_start: Option<&WarmStartData>,
    pair_hint: Option<[f64; 3]>,
    front_output_mode: FrontOutputMode,
    workspace: Option<&mut TransferMooWorkspace>,
    delta_v_anchor_policy: DeltaVAnchorPolicy,
    transfer_moo_policy: TransferMooPolicy,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    let mut local_workspace = None;
    let workspace = workspace.map_or_else(
        || local_workspace.get_or_insert_with(TransferMooWorkspace::new),
        |workspace| workspace,
    );
    let mut state = prepare_oxymoo_front_state(
        ctx,
        warm_start,
        workspace,
        front_output_mode,
        delta_v_anchor_policy,
    )?;
    materialize_oxymoo_front_candidates(
        &mut state,
        ctx,
        pair_hint,
        workspace,
        front_output_mode,
        delta_v_anchor_policy,
        transfer_moo_policy,
    )?;
    let mut polish_pass =
        polish_oxymoo_front_candidates(&mut state, ctx, workspace, front_output_mode)?;
    let mut verified_front = finalize_oxymoo_front_primary(&mut state, ctx, front_output_mode)?;
    apply_oxymoo_scoped_polish_fallback(
        &mut state,
        ctx,
        workspace,
        &mut polish_pass,
        &mut verified_front,
    )?;

    if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
        verified_front.verified_superset_metrics = state.metrics;
    }
    state.candidates.clear();
    workspace.candidates = std::mem::take(&mut state.candidates);
    if !verified_front.is_empty() {
        if matches!(front_output_mode, FrontOutputMode::VerifiedSuperset) {
            if let Some(outer) = state.eval_diag_outer {
                // Region close: the zeroed thread-local IS this front's own
                // contribution, so no subtraction and no baseline error.
                let delta = evaluation_diagnostic_snapshot();
                map_evaluation_arithmetic_overflow(leave_evaluation_diagnostic_region(
                    &outer, &delta,
                ))?;
                add_evaluation_diagnostics_to_stage_metrics(&mut state.metrics, &delta)?;
                verified_front.verified_superset_metrics = state.metrics;
            }
        }
        return Ok(verified_front);
    }

    match front_output_mode {
        FrontOutputMode::TransferPareto => {
            let mut fallback = solve_plan_deterministic_grid(ctx)?;
            fallback.warm_start_used = state.warm_start_consumed;
            verified_front_from_plan(ctx, fallback)
        }
        FrontOutputMode::VerifiedSuperset => {
            let fallback_start = StageTimer::start();
            let deterministic_work_before = work_count_snapshot();
            let mut fallback =
                verified_superset_deterministic_grid_fallback(ctx, state.warm_start_consumed)?;
            state.metrics.deterministic_fallback_s = fallback_start.elapsed_s();
            state.metrics.deterministic_fallback_count = 1;
            state.metrics.deterministic_fallback_full_eval_count =
                checked_stage_metric_count_delta(
                    work_count_snapshot().plan_full_evaluations,
                    deterministic_work_before.plan_full_evaluations,
                )?;
            if let Some(outer) = state.eval_diag_outer {
                // Region close; see the sibling exit above.
                let delta = evaluation_diagnostic_snapshot();
                map_evaluation_arithmetic_overflow(leave_evaluation_diagnostic_region(
                    &outer, &delta,
                ))?;
                add_evaluation_diagnostics_to_stage_metrics(&mut state.metrics, &delta)?;
            }
            fallback.verified_superset_metrics = state.metrics;
            Ok(fallback)
        }
    }
}

#[cfg(feature = "bench-internal")]
mod bench_policy;
#[cfg(feature = "bench-internal")]
pub use bench_policy::{
    bench_delta_v_anchor_policy_report, bench_transfer_moo_policy_report,
    bench_verified_superset_leo_with_delta_v_anchor_policy,
    bench_verified_superset_leo_with_transfer_moo_policy, DeltaVAnchorPolicyBenchReport,
    TransferMooPolicyBenchReport,
};

/// Rank the scalar solver's seeds without adding multi-objective anchor rows.
///
/// The two sorts deliberately retain their historical `partial_cmp` schedule:
/// malformed physics inputs can carry NaN costs, for which changing the
/// comparator or its placement would change candidate order.
fn rank_legacy_scalar_seed_candidates(
    ctx: &PlanContext,
    seeds: &[SolverSeed],
    warm_start_consumed: bool,
    local_cache: &RefCell<SolveLocalWorkCache>,
) -> Result<Vec<(SolverSeed, PlanResult)>, InvalidTargetPropagationAuthorityCode> {
    // Stage 1: coarse evaluation of all seeds (Brent-only relaxation)
    let mut coarse_ranked = Vec::new();
    try_reserve_transfer_capacity(&mut coarse_ranked, seeds.len())?;
    let mut best_coarse_cost = f64::INFINITY;
    let mut recent_costs = VecDeque::new();
    recent_costs
        .try_reserve(4)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    for seed in seeds {
        let plan = evaluate_plan_local(&seed.x, ctx, true, local_cache)?;
        let plan_cost = if plan.valid && plan.cost < INVALID_COST {
            plan.cost
        } else {
            f64::INFINITY
        };
        best_coarse_cost = best_coarse_cost.min(plan_cost);
        push_recent_coarse_cost(&mut recent_costs, plan_cost);
        coarse_ranked.push((
            SolverSeed {
                x: seed.x,
                warm_start_used: seed.warm_start_used,
            },
            plan,
        ));

        if ctx.search_depth.coarse_early_stop
            && ctx.sampling_mode == SamplingMode::Fast
            && coarse_ranked.len() >= 8
            && coarse_ranked.len() % 2 == 0
        {
            let provisional_fine_count =
                provisional_fine_count_for_coarse_early_stop(&coarse_ranked, &ctx.search_depth)?;
            if should_stop_coarse_stage(
                &ctx.search_depth,
                coarse_ranked.len(),
                best_coarse_cost,
                &recent_costs,
                provisional_fine_count,
            ) {
                break;
            }
        }
    }
    coarse_ranked.sort_by(|(_, left_plan), (_, right_plan)| {
        left_plan
            .cost
            .partial_cmp(&right_plan.cost)
            .unwrap_or(Ordering::Equal)
    });

    // Stage 2: fine evaluation of top seeds with cost-gap margin.
    let cost_threshold = fine_cost_threshold(&coarse_ranked, &ctx.search_depth);
    let mut ranked_seeds = Vec::new();
    try_reserve_transfer_capacity(&mut ranked_seeds, coarse_ranked.len())?;
    for (seed, _) in coarse_ranked
        .iter()
        .take_while(|(_, plan)| plan.cost <= cost_threshold)
    {
        let mut plan = evaluate_plan_local(&seed.x, ctx, false, local_cache)?;
        plan.warm_start_used = warm_start_consumed;
        ranked_seeds.push((*seed, plan));
    }

    ranked_seeds.sort_by(compare_ranked_seeds);
    Ok(ranked_seeds)
}

#[derive(Clone, Copy)]
struct ScalarPsoSettings {
    nm_max_iters: usize,
    tune: TuneLevel,
    seed: u64,
    warm_start_consumed: bool,
}

fn push_scalar_pso_global_and_polish_candidates(
    candidates: &mut Vec<PlanResult>,
    ctx: &PlanContext,
    local_cache: &RefCell<SolveLocalWorkCache>,
    optimizer_start_seeds: &[SolverSeed],
    settings: ScalarPsoSettings,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let initial = optimizer_start_seeds
        .first()
        .map_or([0.5, 1.0, 0.25], |seed| seed.x);
    let pso_problem = TransferLocalProblem {
        ctx,
        cache: local_cache,
        coarse_mode: false,
        gradient_enabled: false,
    };
    let pso_iters = settings
        .nm_max_iters
        .checked_mul(2)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let global = run_local_optimizer(
        &pso_problem,
        SINGLE_PAIR_LOWER_BOUNDS,
        SINGLE_PAIR_UPPER_BOUNDS,
        initial,
        local_config(
            LocalOptimizerKind::Pso,
            pso_iters,
            settings.tune,
            settings.seed,
        ),
    )
    .map_err(|error| local_optimizer_failure_code(&error))?;
    let global_x = global.x;
    push_local_result_candidate(
        candidates,
        &global,
        ctx,
        local_cache,
        settings.warm_start_consumed,
    )?;

    let polish_problem = TransferLocalProblem {
        ctx,
        cache: local_cache,
        coarse_mode: false,
        gradient_enabled: false,
    };
    let polish = run_local_optimizer(
        &polish_problem,
        SINGLE_PAIR_LOWER_BOUNDS,
        SINGLE_PAIR_UPPER_BOUNDS,
        global_x,
        local_config(
            LocalOptimizerKind::NelderMead,
            settings.nm_max_iters,
            settings.tune,
            settings.seed,
        ),
    )
    .map_err(|error| local_optimizer_failure_code(&error))?;
    push_local_result_candidate(
        candidates,
        &polish,
        ctx,
        local_cache,
        settings.warm_start_consumed,
    )
}

fn solve_plan_front_internal(
    ctx: &mut PlanContext,
    warm_start: Option<&WarmStartData>,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    prepare_single_pair_context(ctx);
    let local_cache = RefCell::new(SolveLocalWorkCache::new());

    let seeds = build_single_pair_seeds(ctx, warm_start)?;
    if seeds.is_empty() {
        let fallback = solve_plan_deterministic_grid(ctx)?;
        return verified_front_from_plan(ctx, fallback);
    }
    let warm_start_consumed = seeds.iter().any(|seed| seed.warm_start_used);
    let ranked_seeds =
        rank_legacy_scalar_seed_candidates(ctx, &seeds, warm_start_consumed, &local_cache)?;

    let best_seed_cost = ranked_seeds
        .iter()
        .find_map(|(_, plan)| {
            if plan.valid && plan.cost < INVALID_COST {
                Some(plan.cost)
            } else {
                None
            }
        })
        .unwrap_or(INVALID_COST);

    let complexity = TransferComplexity::classify_from_ctx(ctx);
    let local_optimizer = ctx.local_optimizer;
    let optimizer_kind = resolve_local_optimizer_kind(local_optimizer, complexity, best_seed_cost);
    let tune = match local_optimizer.choice {
        TransferLocalOptimizerChoice::Auto => TuneLevel::Default,
        TransferLocalOptimizerChoice::Fixed(_) => local_optimizer.tune,
    };
    let nm_max_iters = nm_max_iters_for_complexity(complexity);
    let optimizer_start_seeds = select_optimizer_start_seeds(&ranked_seeds, optimizer_kind)?;

    let explicit_optimizer = matches!(
        local_optimizer.choice,
        TransferLocalOptimizerChoice::Fixed(_)
    );
    if explicit_optimizer && optimizer_start_seeds.is_empty() {
        return Ok(TransferFront::empty());
    }

    let initial_candidate_count = if explicit_optimizer {
        0
    } else {
        ranked_seeds.len().min(optimizer_start_seeds.len().max(1))
    };
    let mut candidates = Vec::new();
    try_reserve_transfer_capacity(&mut candidates, initial_candidate_count)?;
    if !explicit_optimizer {
        for (_seed, plan) in ranked_seeds.iter().take(optimizer_start_seeds.len().max(1)) {
            let mut raw = plan.clone();
            raw.func_evals = 0;
            raw.optimizer_func_evals = 0;
            raw.optimizer_converged = false;
            raw.warm_start_used = warm_start_consumed;
            candidates.push(raw);
        }
    }

    if optimizer_kind == LocalOptimizerKind::Pso {
        push_scalar_pso_global_and_polish_candidates(
            &mut candidates,
            ctx,
            &local_cache,
            &optimizer_start_seeds,
            ScalarPsoSettings {
                nm_max_iters,
                tune,
                seed: local_optimizer.seed,
                warm_start_consumed,
            },
        )?;
    }

    for seed in &optimizer_start_seeds {
        match optimizer_kind {
            LocalOptimizerKind::NelderMead | LocalOptimizerKind::Pso => {
                let coarse_iters = nm_max_iters
                    .checked_mul(7)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?
                    / 10;
                let coarse_problem = TransferLocalProblem {
                    ctx,
                    cache: &local_cache,
                    coarse_mode: true,
                    gradient_enabled: false,
                };
                let coarse = run_local_optimizer(
                    &coarse_problem,
                    SINGLE_PAIR_LOWER_BOUNDS,
                    SINGLE_PAIR_UPPER_BOUNDS,
                    seed.x,
                    local_config(
                        LocalOptimizerKind::NelderMead,
                        coarse_iters,
                        tune,
                        local_optimizer.seed,
                    ),
                )
                .map_err(|error| local_optimizer_failure_code(&error))?;

                let fine_iters = nm_max_iters
                    .checked_sub(coarse_iters)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                let fine_problem = TransferLocalProblem {
                    ctx,
                    cache: &local_cache,
                    coarse_mode: false,
                    gradient_enabled: false,
                };
                let fine = run_local_optimizer(
                    &fine_problem,
                    SINGLE_PAIR_LOWER_BOUNDS,
                    SINGLE_PAIR_UPPER_BOUNDS,
                    coarse.x,
                    local_config(
                        LocalOptimizerKind::NelderMead,
                        fine_iters,
                        tune,
                        local_optimizer.seed,
                    ),
                )
                .map_err(|error| local_optimizer_failure_code(&error))?;
                let mut optimized_plan = evaluate_plan_local(&fine.x, ctx, false, &local_cache)?;
                optimized_plan.func_evals = coarse
                    .evaluations
                    .checked_add(fine.evaluations)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                optimized_plan.optimizer_func_evals = optimized_plan.func_evals;
                optimized_plan.optimizer_converged = fine.converged;
                optimized_plan.warm_start_used = warm_start_consumed;
                try_reserve_transfer_capacity(&mut candidates, 1)?;
                candidates.push(optimized_plan);
            }
            LocalOptimizerKind::Lbfgs => {
                let problem = TransferLocalProblem {
                    ctx,
                    cache: &local_cache,
                    coarse_mode: false,
                    gradient_enabled: true,
                };
                let result = run_local_optimizer(
                    &problem,
                    SINGLE_PAIR_LOWER_BOUNDS,
                    SINGLE_PAIR_UPPER_BOUNDS,
                    seed.x,
                    local_config(
                        LocalOptimizerKind::Lbfgs,
                        nm_max_iters,
                        tune,
                        local_optimizer.seed,
                    ),
                )
                .map_err(|error| local_optimizer_failure_code(&error))?;
                push_local_result_candidate(
                    &mut candidates,
                    &result,
                    ctx,
                    &local_cache,
                    warm_start_consumed,
                )?;
            }
        }
    }

    let verified_front = finalize_verified_front(ctx, &mut candidates)?;
    if !verified_front.is_empty() {
        return Ok(verified_front);
    }

    if explicit_optimizer {
        return Ok(TransferFront::empty());
    }

    let mut fallback = solve_plan_deterministic_grid(ctx)?;
    fallback.warm_start_used = warm_start_consumed;
    verified_front_from_plan(ctx, fallback)
}

/// Deterministic grid-search solver.
///
/// Evaluates the fixed [`DETERMINISTIC_GRID_POINTS`] candidate grid over
/// [time2phase, `phase_sma`, waittime] ratios and returns the best valid
/// `PlanResult`.
///
/// This is retained as a fallback path for difficult or under-constrained cases.
fn solve_plan_deterministic_grid(
    ctx: &mut PlanContext,
) -> Result<PlanResult, InvalidTargetPropagationAuthorityCode> {
    prepare_single_pair_context(ctx);
    let local_cache = RefCell::new(SolveLocalWorkCache::new());

    let mut best: Option<([f64; 3], PlanResult)> = None;
    // DETERMINISTIC_GRID_POINTS is the same nested enumeration this loop used
    // to spell out, with the `t + w > 0.98` feasibility guard already applied.
    for &x in &DETERMINISTIC_GRID_POINTS {
        let res = evaluate_plan_local(&x, ctx, false, &local_cache)?;
        if !res.valid {
            continue;
        }
        match &best {
            None => best = Some((x, res)),
            Some((best_x, best_res)) => {
                // Reproduce native f64 equality without changing the
                // tie rule: signed zeroes tie and NaN never does.
                let res_cost_bits = res.cost.to_bits();
                let best_cost_bits = best_res.cost.to_bits();
                let sign_bit = (-0.0_f64).to_bits();
                let costs_tie = !res.cost.is_nan()
                    && (res_cost_bits == best_cost_bits
                        || (res_cost_bits & !sign_bit == 0 && best_cost_bits & !sign_bit == 0));
                if res.cost < best_res.cost
                    || (costs_tie && (x[0], x[1], x[2]) < (best_x[0], best_x[1], best_x[2]))
                {
                    best = Some((x, res));
                }
            }
        }
    }

    Ok(best.map_or_else(PlanResult::invalid, |(_, res)| res))
}
