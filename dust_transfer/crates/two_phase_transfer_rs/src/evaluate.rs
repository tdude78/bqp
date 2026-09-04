//! Core physics evaluation for two-phase transfer planning.
//!
//! This module implements `evaluate_plan` - the heart of the transfer solver.
//! Given a 3-parameter vector (`time2phase_ratio`, `phase_sma_ratio`, `waittime_ratio`),
//! it computes the optimal transfer trajectory and total delta-V cost.

use crate::lambert_backend::{
    visit_lambert_branch_solutions_pruned_with_r1,
    visit_lambert_exact_branch_solutions_pruned_with_r1, LambertProblem,
};
use num_traits::ToPrimitive;
use ordered_float::OrderedFloat;
use pdqsort;
use satpy_core::{
    cross3, eci2equinoc_impl, eci2kep_impl, equinoc_prop_from_impl, equinoc_prop_j2_from_impl,
    equinoc_prop_j2_step_impl, norm3, optim::minimize_scalar_bounded, MU, RE, SEC_PER_DAY,
};
use smallvec::SmallVec;
#[cfg(test)]
use std::cell::RefCell;
use std::time::Instant;

use crate::types::{
    BodyForceConfig, BodyRole, BranchRejectionToken, EciBasicOrbit, LambertBranchSelection,
    LambertSolutionEx, PlanContext, PlanResult, PropagationFidelity, ReplayProvenance,
    TimingFailureToken, INVALID_COST, MIN_TOF,
};
#[inline]
fn candidate_search_is_supported(ctx: &PlanContext) -> bool {
    crate::types::validate_candidate_search_authority(
        ctx.target_propagation_authority,
        ctx.force_config.as_deref(),
        ctx.execution_policy.use_high_fidelity || ctx.execution_policy.require_high_fidelity,
    )
    .is_ok()
}

#[inline]
fn validate_public_evaluate_plan_authority(
    ctx: &PlanContext,
) -> Result<(), crate::types::InvalidTargetPropagationAuthorityCode> {
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
    )
}

#[inline]
fn unsupported_candidate_search_result() -> PlanResult {
    let mut result = PlanResult::invalid();
    result.branch_rejection = BranchRejectionToken::UnsupportedHighFidelityCandidateSearch;
    result
}

const BRENT_FINE_MAX_ITERATIONS: usize = 50;
// One pre-seeded quantized cost, the bounded pre-scan, then Brent's initial
// objective evaluation plus its bounded fine iterations. Keep both cache forms
// inline across that complete bounded search.
const BRENT_CACHE_INLINE_CAPACITY: usize =
    1 + BRENT_PRESCAN_MAX_SAMPLES + 1 + BRENT_FINE_MAX_ITERATIONS;

type BrentLocalCache = SmallVec<[(i64, f64); BRENT_CACHE_INLINE_CAPACITY]>;
// Cost reuse remains intentionally quantized; physical solution reuse requires
// exact TOF bits so neighboring values in one 0.1-second bin never alias.
type BrentExactSolutionCache = SmallVec<[(u64, LambertSolutionEx); BRENT_CACHE_INLINE_CAPACITY]>;

pub(crate) mod diagnostics;

#[cfg(feature = "bench-internal")]
pub use diagnostics::EvaluationArithmeticOverflow;
#[cfg(not(feature = "bench-internal"))]
pub(crate) use diagnostics::EvaluationArithmeticOverflow;
use diagnostics::{
    checked_diagnostic_counter_add, j2_iteration_count_as_u32, record_branch_brent,
    record_branch_brent_cache_counts, record_branch_emitted, record_branch_eval_call,
    record_branch_j2_correction, record_branch_lambert_sampling, record_branch_rejected,
    record_branch_shared_prepare, record_branch_source, record_branch_target_propagation,
    record_evaluation_diagnostic, record_j2_correction, record_j2_propagate_state,
    record_j2_residual_gate, record_lambert_batch_call, record_lambert_batch_work,
    record_lambert_scalar_tof_calls, record_target_j2_batch_state_count,
    record_target_j2_scalar_state, record_target_j2_simd4_chunk,
};
pub(crate) use diagnostics::{
    enter_evaluation_diagnostic_region, evaluation_diagnostic_snapshot,
    leave_evaluation_diagnostic_region, merge_evaluation_diagnostics,
    record_lambert_branch_solution, record_phase_state_cache_lookup,
    restore_evaluation_diagnostics, EvaluationDiagnosticCounters,
};
#[cfg(test)]
use diagnostics::{
    hf_propagation_telemetry_snapshot, record_hf_propagation_stage, HfPropagationStage,
    HfPropagationTelemetry,
};

thread_local! {
    #[cfg(test)]
    static HF_MULTI_TOF_TEST_CALLS: RefCell<(usize, usize, bool)> = const { RefCell::new((0, 0, false)) };
    #[cfg(test)]
    static HF_MULTI_TOF_TEST_REBASE_OBSERVED: RefCell<bool> = const { RefCell::new(false) };
}

#[inline]
fn brent_cache_lookup(cache: &BrentLocalCache, tof_key: i64) -> Option<f64> {
    cache
        .iter()
        .find_map(|(key, cost)| (*key == tof_key).then_some(*cost))
}

#[inline]
fn brent_tof_cache_key(tof: f64) -> Result<i64, EvaluationArithmeticOverflow> {
    (tof * 10.0)
        .round()
        .to_i64()
        .ok_or(EvaluationArithmeticOverflow)
}

#[inline]
fn brent_cache_insert_first(cache: &mut BrentLocalCache, tof_key: i64, cost: f64) {
    if brent_cache_lookup(cache, tof_key).is_none() {
        cache.push((tof_key, cost));
    }
}

#[inline]
fn brent_exact_solution_lookup(
    cache: &[(u64, LambertSolutionEx)],
    tof: f64,
) -> Option<LambertSolutionEx> {
    let tof_bits = tof.to_bits();
    cache
        .iter()
        .find_map(|(bits, solution)| (*bits == tof_bits).then_some(*solution))
}

#[inline]
fn brent_refinement_required(bracket_lo: f64, bracket_hi: f64) -> bool {
    bracket_hi - bracket_lo >= 60.0
}

/// Target TOF resolution of the pre-scan, in seconds. Also the width of the
/// widest cost basin the scan is allowed to step over.
///
/// Measured on `physics_3event` against an exhaustive (513-sample) TOF search,
/// over 1324 decision points around event 0's selected plan, as the fraction
/// of points whose transfer dv comes out more than 10% above exhaustive, with
/// the whole three-event stage-1 solve's wall time beside it (min of 3 x 50
/// repeats, macOS release):
///
/// | samples | bad points | stage-1 wall | vs today |
/// |---|---|---|---|
/// | today (no scan) | 91.6% | 0.054 s | 1.00x |
/// | 9  | 84.9% | 0.070 s | 1.30x |
/// | 13 | 71.4% | 0.072 s | 1.33x |
/// | 17 | 62.1% | 0.077 s | 1.43x |
/// | 25 | 61.4% | 0.096 s | 1.78x |
/// | 33 | 49.8% | 0.116 s | 2.15x |
/// | 65 | 36.7% | 0.159 s | 2.94x |
///
/// That table measured the retired Hohmann-narrowed brackets. F3 now brackets
/// the full physical interval, so its conclusion does not transfer: the
/// committed three-event science test still regressed event 2 at 200 s, while
/// 100 s finds a strictly lower-delta-v valid result on all three events. The
/// 129-sample ceiling covers the revolution-capped fixture interval at that
/// resolution without turning the pre-scan into an unbounded search.
const BRENT_PRESCAN_RESOLUTION_S: f64 = 100.0;
/// Coarse optimizer screening still spans the complete hard interval, but at a
/// lower density; retained candidates are re-evaluated by the 100 s fine lane.
/// The committed three-event science test is the non-regression authority for
/// whether this screening density preserves every selected basin.
const BRENT_PRESCAN_COARSE_RESOLUTION_S: f64 = 1_200.0;
/// Sample-count clamp. The floor keeps a bracket bracketed at all; the ceiling
/// caps what one Brent call can cost on the widest geometries.
const BRENT_PRESCAN_COUNT_RANGE: (usize, usize) = (5, 129);
/// Inline capacity for the batched pre-scan's per-sample buffers: the clamp
/// ceiling above, so no scan ever spills to the heap.
const BRENT_PRESCAN_MAX_SAMPLES: usize = BRENT_PRESCAN_COUNT_RANGE.1;

/// Number of uniform samples the TOF pre-scan takes across Brent's bracket
/// before Brent runs. See [`brent_prescan_bracket`].
#[inline]
fn brent_prescan_count(
    bracket_span: f64,
    coarse_mode: bool,
    is_large_plane_change: bool,
) -> Result<usize, EvaluationArithmeticOverflow> {
    let resolution_s = if coarse_mode && !is_large_plane_change {
        BRENT_PRESCAN_COARSE_RESOLUTION_S
    } else {
        BRENT_PRESCAN_RESOLUTION_S
    };
    if !bracket_span.is_finite() || bracket_span <= 0.0 {
        return Ok(BRENT_PRESCAN_COUNT_RANGE.0);
    }
    let (min_count, max_count) = BRENT_PRESCAN_COUNT_RANGE;
    let interval_count = (bracket_span / resolution_s)
        .ceil()
        .to_usize()
        .ok_or(EvaluationArithmeticOverflow)?;
    let count = interval_count
        .checked_add(1)
        .ok_or(EvaluationArithmeticOverflow)?;
    Ok(count.clamp(min_count, max_count))
}

#[inline]
fn tof_grid_sample_count(
    span: f64,
    is_simple_transfer: bool,
) -> Result<Option<usize>, EvaluationArithmeticOverflow> {
    let (sample_spacing_s, count_offset, minimum, maximum) = if is_simple_transfer {
        (8000.0, 3, 4, 7)
    } else {
        (6000.0, 4, 5, 9)
    };
    if !span.is_finite() || span < 0.0 {
        return Ok(None);
    }
    let interval_count = (span / sample_spacing_s)
        .ceil()
        .to_usize()
        .ok_or(EvaluationArithmeticOverflow)?;
    let count = interval_count
        .checked_add(count_offset)
        .ok_or(EvaluationArithmeticOverflow)?;
    Ok(Some(count.clamp(minimum, maximum)))
}

#[inline]
fn half_tof_budget_as_i32(tof_budget: usize) -> Result<i32, EvaluationArithmeticOverflow> {
    let half_budget = tof_budget
        .checked_div(2)
        .ok_or(EvaluationArithmeticOverflow)?;
    i32::try_from(half_budget).map_err(|_| EvaluationArithmeticOverflow)
}

/// Bracket Brent around the best point of a uniform scan of `[lo, hi]`.
///
/// Brent's method is a LOCAL minimiser. The transfer-cost-vs-TOF curve is not
/// unimodal — Lambert branch changes and revolution boundaries put several
/// separated minima inside one bracket — so running Brent on the whole bracket
/// converges to whichever basin the golden-section start point happens to land
/// in. That start point is a fixed affine function of the bracket ends, so a
/// 1-ULP move in either end can select a different basin: the mechanism behind
/// the macOS/Linux `stage1_transfer_solve_matches_oracle_event_dv` split.
///
/// Sampling the bracket uniformly first and handing Brent only the interval
/// `[t[k-1], t[k+1]]` around the best sample `t[k]` makes basin selection a
/// property of the scan (deterministic, resolution-bounded) instead of a
/// property of the arithmetic. Brent then only polishes inside one basin,
/// which is exactly the problem it is good at.
///
/// Returns `(lo, hi, scan_tof, scan_cost)` for the selected scanned basin. The
/// caller owns and retains the incoming incumbent independently.
fn brent_prescan_bracket(
    bracket_lo: f64,
    bracket_hi: f64,
    samples: usize,
    best_tof: f64,
    best_cost: f64,
    mut eval: impl FnMut(f64) -> Result<f64, EvaluationArithmeticOverflow>,
) -> Result<(f64, f64, f64, f64), EvaluationArithmeticOverflow> {
    let Some(ladder) = PrescanLadder::new(bracket_lo, bracket_hi, samples)? else {
        return Ok((bracket_lo, bracket_hi, best_tof, best_cost));
    };
    let mut best_idx = 0usize;
    let mut best_scan = f64::INFINITY;
    for i in 0..samples {
        let tof = ladder.sample_tof(i)?;
        let cost = eval(tof)?;
        if cost < best_scan {
            best_scan = cost;
            best_idx = i;
        }
    }
    // No sample resolved to a real Lambert solution: there is no basin to
    // bracket, so hand Brent the untouched interval it would have had.
    if !best_scan.is_finite() || best_scan >= INVALID_COST {
        return Ok((bracket_lo, bracket_hi, best_tof, best_cost));
    }
    let step = ladder.step();
    let last_sample_index = samples.checked_sub(1).ok_or(EvaluationArithmeticOverflow)?;
    if best_idx > last_sample_index {
        return Err(EvaluationArithmeticOverflow);
    }
    let previous_index = if best_idx == 0 {
        0
    } else {
        best_idx
            .checked_sub(1)
            .ok_or(EvaluationArithmeticOverflow)?
    };
    let next_index = if best_idx == last_sample_index {
        last_sample_index
    } else {
        best_idx
            .checked_add(1)
            .ok_or(EvaluationArithmeticOverflow)?
    };
    let previous_index = previous_index
        .to_f64()
        .ok_or(EvaluationArithmeticOverflow)?;
    let next_index = next_index.to_f64().ok_or(EvaluationArithmeticOverflow)?;
    let best_index = best_idx.to_f64().ok_or(EvaluationArithmeticOverflow)?;
    let lo = bracket_lo + step * previous_index;
    let hi = bracket_lo + step * next_index;
    // The caller retains the incoming incumbent independently. This tuple
    // describes the scanned basin only, so its candidate must lie inside its
    // bracket even when the incumbent is better than every scan sample.
    Ok((lo, hi, bracket_lo + step * best_index, best_scan))
}

/// The pre-scan's uniform sample ladder over `[bracket_lo, bracket_hi]`.
///
/// This type exists so the ladder has ONE definition. [`brent_prescan_bracket`]
/// and [`prescan_eval_samples_batched`] have to agree on three things — whether
/// a ladder exists at all, how many samples it holds, and the exact `tof` of
/// each one — because the batched path evaluates the ladder while the bracket
/// path replays those costs positionally. While each carried its own copy of
/// the guard and of the `bracket_lo + span * i / last` expression, editing
/// either copy desynchronised the replay: the bracket asks for a cost the batch
/// never produced and the plan fails with `EvaluationArithmeticOverflow`.
#[derive(Clone, Copy)]
struct PrescanLadder {
    bracket_lo: f64,
    span: f64,
    last: f64,
}

impl PrescanLadder {
    /// `None` when the bracket has no ladder: fewer than three samples, or ends
    /// that are inverted or NaN.
    fn new(
        bracket_lo: f64,
        bracket_hi: f64,
        samples: usize,
    ) -> Result<Option<Self>, EvaluationArithmeticOverflow> {
        if samples < 3 || bracket_hi <= bracket_lo || bracket_hi.is_nan() || bracket_lo.is_nan() {
            return Ok(None);
        }
        let last = samples
            .checked_sub(1)
            .and_then(|count| count.to_f64())
            .ok_or(EvaluationArithmeticOverflow)?;
        Ok(Some(Self {
            bracket_lo,
            span: bracket_hi - bracket_lo,
            last,
        }))
    }

    /// The `tof` of sample `index`.
    fn sample_tof(self, index: usize) -> Result<f64, EvaluationArithmeticOverflow> {
        let sample_index = index.to_f64().ok_or(EvaluationArithmeticOverflow)?;
        Ok(self.bracket_lo + self.span * sample_index / self.last)
    }

    /// Spacing between adjacent samples.
    fn step(self) -> f64 {
        self.span / self.last
    }
}

/// Inputs shared by every sample of one batched pre-scan (see
/// [`prescan_eval_samples_batched`]). All fields are the exact values the
/// sequential `eval_tof` closure would have captured.
struct PrescanBatchInputs<'a> {
    ctx: &'a PlanContext,
    dep_at_release: &'a [f64; 6],
    r1_cache: &'a crate::lambert::LambertR1Cache,
    fixed_offset: f64,
    max_transfer_headroom_s: f64,
    bracket_lo: f64,
    bracket_hi: f64,
    samples: usize,
}

/// Per-sample outcome of the batched pre-scan's lookup pass.
enum PrescanSampleSlot {
    /// Cost served by the cache as it stood when this sample was reached.
    Ready(f64),
    /// Cost comes from pending miss `index`. This also covers a later sample
    /// whose quantized key collides with an earlier miss in the SAME scan: the
    /// sequential path would have served it from the entry that miss had just
    /// inserted, so it aliases the miss instead of solving again.
    Pending(usize),
}

/// One recorded cache miss of the batched pre-scan's lookup pass.
struct PrescanPendingSample {
    tof: f64,
    key: i64,
}

/// Batched evaluation of one Brent bracket's pre-scan sample ladder.
///
/// Sequential-equivalence contract: for each sample, in ladder order, this
/// performs the same cache lookup, the same request/hit/miss bookkeeping, the
/// same propagation, prune, and cap arithmetic, and folds the same enumerated
/// candidates in the same order as running the `eval_tof` closure per sample
/// would have. The only restructuring is that the misses' Lambert variants are
/// enumerated in ONE cross-TOF streaming pack
/// (`visit_lambert_branch_solutions_pruned_with_r1_multi_tof`, or its
/// selected-branch counterpart when a branch selection is active), whose
/// per-lane bits equal the single-problem pack's bits by kernel lane
/// independence (`crate::lambert::find_xy_simd4_m_variant_per_lane_t`). Cache
/// insertions and exact-solution pushes happen in miss order — which is sample
/// order — so both caches end bit-identical to the sequential path's, and the
/// returned ladder of costs replayed through [`brent_prescan_bracket`] selects
/// the same basin.
///
/// Returns `Ok(None)` — evaluate nothing, caller falls back to the sequential
/// closure — when [`brent_prescan_bracket`]'s own guard would evaluate nothing
/// anyway.
fn prescan_eval_samples_batched(
    inputs: &PrescanBatchInputs<'_>,
    brent_cache: &mut BrentLocalCache,
    exact_cache: &mut BrentExactSolutionCache,
    request_count: &mut usize,
    hit_count: &mut usize,
    miss_count: &mut usize,
) -> Result<Option<SmallVec<[f64; BRENT_PRESCAN_MAX_SAMPLES]>>, EvaluationArithmeticOverflow> {
    let Some(ladder) = PrescanLadder::new(inputs.bracket_lo, inputs.bracket_hi, inputs.samples)?
    else {
        return Ok(None);
    };

    // Lookup pass, in ladder order against the cache as it stands.
    let mut slots: SmallVec<[PrescanSampleSlot; BRENT_PRESCAN_MAX_SAMPLES]> = SmallVec::new();
    let mut pending: SmallVec<[PrescanPendingSample; BRENT_PRESCAN_MAX_SAMPLES]> = SmallVec::new();
    for i in 0..inputs.samples {
        let tof = ladder.sample_tof(i)?;
        *request_count = request_count
            .checked_add(1)
            .ok_or(EvaluationArithmeticOverflow)?;
        let key = brent_tof_cache_key(tof)?;
        if let Some(cost) = brent_cache_lookup(brent_cache, key) {
            *hit_count = hit_count
                .checked_add(1)
                .ok_or(EvaluationArithmeticOverflow)?;
            slots.push(PrescanSampleSlot::Ready(cost));
            continue;
        }
        if let Some(position) = pending.iter().position(|entry| entry.key == key) {
            *hit_count = hit_count
                .checked_add(1)
                .ok_or(EvaluationArithmeticOverflow)?;
            slots.push(PrescanSampleSlot::Pending(position));
            continue;
        }
        *miss_count = miss_count
            .checked_add(1)
            .ok_or(EvaluationArithmeticOverflow)?;
        slots.push(PrescanSampleSlot::Pending(pending.len()));
        pending.push(PrescanPendingSample { tof, key });
    }

    // Pre-Lambert pass per miss, in miss (= sample) order: the same finite /
    // MIN_TOF / headroom guards, propagation, norm guards, and prune
    // arithmetic as `lambert_solve_raw` + `select_lambert_branch_solution_with_r1`.
    //
    // Two problem lists, and exactly one of them is ever non-empty: the
    // unselected route enumerates every branch of each miss, the selected route
    // one exact `(rev, low_path)` branch. `problem_states` carries the arrival
    // state for whichever list is in play, so the fold below is shared.
    let mut problems: SmallVec<[crate::lambert::MultiTofBranchProblem; BRENT_PRESCAN_MAX_SAMPLES]> =
        SmallVec::new();
    let mut exact_problems: SmallVec<
        [crate::lambert::MultiTofExactBranchProblem; BRENT_PRESCAN_MAX_SAMPLES],
    > = SmallVec::new();
    let mut problem_states: SmallVec<[[f64; 6]; BRENT_PRESCAN_MAX_SAMPLES]> = SmallVec::new();
    let mut pending_problem_slots: SmallVec<[Option<usize>; BRENT_PRESCAN_MAX_SAMPLES]> =
        SmallVec::new();
    // The departure state is fixed for this whole miss list, and both branch
    // bounds below open by re-deriving quantities from it. Hoisted, verbatim
    // and bit-identical, exactly as the r1-side Lambert cache above it is.
    let departure_bounds = crate::lambert_backend::DepartureBoundCache::new(inputs.dep_at_release);
    for entry in &pending {
        let tof = entry.tof;
        if !tof.is_finite() || tof < MIN_TOF || tof > inputs.max_transfer_headroom_s {
            pending_problem_slots.push(None);
            continue;
        }
        let propagated = propagate_candidate_target_at_authoritative_offset(
            inputs.ctx,
            inputs.fixed_offset + tof,
        )?;
        let Some(tgt_state) = propagated else {
            pending_problem_slots.push(None);
            continue;
        };
        let r1 = [
            inputs.dep_at_release[0],
            inputs.dep_at_release[1],
            inputs.dep_at_release[2],
        ];
        let r2 = [tgt_state[0], tgt_state[1], tgt_state[2]];
        if norm3(&r1) <= 0.0 || norm3(&r2) <= 0.0 {
            pending_problem_slots.push(None);
            continue;
        }
        let Some(branch_max_revs) = select_lambert_branch_ceiling(inputs.ctx) else {
            pending_problem_slots.push(None);
            continue;
        };
        let dv_cap = inputs.ctx.max_transfer_dv;
        let branch_max_revs = crate::lambert_backend::max_revolutions_below_dv_cap_cached(
            &departure_bounds,
            &tgt_state,
            tof,
            dv_cap,
            branch_max_revs,
        );
        let include_retrograde = crate::lambert_backend::retrograde_departure_dv_lower_bound_cached(
            &departure_bounds,
            &tgt_state,
        ) < dv_cap;
        // The selected-branch route's own early reject, which the sequential
        // `select_lambert_branch_solution_with_r1` applies before it enumerates.
        if inputs
            .ctx
            .lambert_branch_selection
            .is_some_and(|selection| selection.rev > branch_max_revs)
        {
            pending_problem_slots.push(None);
            continue;
        }
        pending_problem_slots.push(Some(problem_states.len()));
        problem_states.push(tgt_state);
        if let Some(selection) = inputs.ctx.lambert_branch_selection {
            exact_problems.push(crate::lambert::MultiTofExactBranchProblem {
                state2: tgt_state,
                tof,
                rev: selection.rev,
                low_path: selection.low_path,
                include_retrograde,
            });
        } else {
            problems.push(crate::lambert::MultiTofBranchProblem {
                state2: tgt_state,
                tof,
                m_max: branch_max_revs,
                include_retrograde,
            });
        }
    }

    // One streaming enumeration across every surviving miss; candidates fold
    // per problem through the same filter/argmin as the sequential selector.
    let mut bests: SmallVec<[Option<LambertSolutionEx>; BRENT_PRESCAN_MAX_SAMPLES]> =
        SmallVec::new();
    bests.resize(problem_states.len(), None);
    {
        let mut fold = |problem_index: usize,
                        m: i32,
                        low_path: bool,
                        prograde: bool,
                        dv_vec: [f64; 3],
                        arrival_dv_vec: [f64; 3]| {
            let (Some(tgt_state), Some(best)) = (
                problem_states.get(problem_index),
                bests.get_mut(problem_index),
            ) else {
                return;
            };
            fold_lambert_branch_candidate(
                inputs.ctx,
                tgt_state,
                best,
                m,
                low_path,
                prograde,
                dv_vec,
                arrival_dv_vec,
            );
        };
        if inputs.ctx.lambert_branch_selection.is_some() {
            crate::lambert_backend::visit_lambert_exact_branch_solutions_pruned_with_r1_multi_tof(
                inputs.r1_cache,
                inputs.dep_at_release,
                &exact_problems,
                &mut fold,
            )?;
        } else {
            crate::lambert_backend::visit_lambert_branch_solutions_pruned_with_r1_multi_tof(
                inputs.r1_cache,
                inputs.dep_at_release,
                &problems,
                true,
                &mut fold,
            )?;
        }
    }

    // Results back in miss order: exact-solution pushes and cache inserts in
    // exactly the order the sequential path would have made them.
    let mut pending_costs: SmallVec<[f64; BRENT_PRESCAN_MAX_SAMPLES]> = SmallVec::new();
    for (entry, problem_slot) in pending.iter().zip(&pending_problem_slots) {
        let solution = problem_slot.and_then(|slot| bests.get(slot).copied().flatten());
        let cost = solution.map_or(INVALID_COST, |sol| {
            exact_cache.push((entry.tof.to_bits(), sol));
            sol.cost
        });
        brent_cache_insert_first(brent_cache, entry.key, cost);
        pending_costs.push(cost);
    }

    let mut costs: SmallVec<[f64; BRENT_PRESCAN_MAX_SAMPLES]> = SmallVec::new();
    for slot in &slots {
        match slot {
            PrescanSampleSlot::Ready(cost) => costs.push(*cost),
            PrescanSampleSlot::Pending(index) => costs.push(
                pending_costs
                    .get(*index)
                    .copied()
                    .ok_or(EvaluationArithmeticOverflow)?,
            ),
        }
    }
    Ok(Some(costs))
}

/// Replay a batched pre-scan's costs through [`brent_prescan_bracket`].
///
/// The bracket asks for sample `i`'s cost and gets the batch's `i`-th entry:
/// both walk the same [`PrescanLadder`], so position is identity. One helper
/// rather than a copy per call site, so the two routes cannot drift into
/// replaying differently.
fn brent_prescan_bracket_from_costs(
    bracket_lo: f64,
    bracket_hi: f64,
    samples: usize,
    best_tof: f64,
    best_cost: f64,
    costs: SmallVec<[f64; BRENT_PRESCAN_MAX_SAMPLES]>,
) -> Result<(f64, f64, f64, f64), EvaluationArithmeticOverflow> {
    let mut replay = costs.into_iter();
    brent_prescan_bracket(
        bracket_lo,
        bracket_hi,
        samples,
        best_tof,
        best_cost,
        |_tof| replay.next().ok_or(EvaluationArithmeticOverflow),
    )
}

/// Inline capacity of the batched TOF scan's per-sample buffers.
///
/// `MAX_TOF_SAMPLES` is the sampling BUDGET's ceiling (256 under
/// `dissertation_production.yaml`), not the working size: the sampling
/// heuristics land at a few dozen surviving TOFs, measured over 1,063,458
/// scans of the 8-design/24-event census at a mean of 16.1 with every scan at
/// 4 or more. Sizing these buffers to 256 would add tens of KiB to a frame
/// that already carries the propagation buffers at that width; above this
/// capacity they spill to the heap, which is transparent to results.
const SCAN_BATCH_INLINE_SAMPLES: usize = 32;

/// Inputs of one batched TOF sampling scan (see
/// [`scan_lambert_samples_batched`]). `tgt_states` is the full propagated
/// ladder and `tof_to_idx` maps each surviving TOF back into it, exactly as the
/// sequential loop indexes them.
struct ScanBatchInputs<'a> {
    ctx: &'a PlanContext,
    dep_at_release: &'a [f64; 6],
    r1_cache: &'a crate::lambert::LambertR1Cache,
    tgt_states: &'a [[f64; 6]],
    tof_to_idx: &'a [usize],
    valid_tofs: &'a [f64],
}

/// Batched evaluation of the TOF sampling scan's Lambert solves.
///
/// Sequential-equivalence contract: for each surviving TOF, in sample order,
/// this applies the same revolution ceiling, the same acceptance cap, the same
/// multi-rev energy prune and the same retrograde prune as
/// `select_lambert_branch_solution_with_r1`, and folds the enumerated
/// candidates through the same [`fold_lambert_branch_candidate`] filter and
/// argmin. The only restructuring is that every surviving TOF is enumerated in
/// ONE cross-TOF streaming pack
/// (`visit_lambert_branch_solutions_pruned_with_r1_multi_tof`), whose per-lane
/// bits equal the single-problem pack's by kernel lane independence
/// (`crate::lambert::find_xy_simd4_m_variant_per_lane_t`).
///
/// The caller's running `best_cost` is deliberately not an input. It is not one
/// in the sequential path either: it gates only the outer argmin across
/// samples, never the enumerator's own prunes, which read `max_transfer_dv`
/// and are therefore invariant across the scan. Batching loses
/// no pruning, and a caller folding these results in sample order reproduces
/// the sequential selection — including tie-breaks, since both keep the first
/// sample to achieve a cost under a strict `<`.
///
/// Only the unselected-branch route comes here; the selected-branch route
/// enumerates one exact branch through a different pack and stays sequential.
/// Returns one entry per `valid_tofs` element, in the same order.
fn scan_lambert_samples_batched(
    inputs: &ScanBatchInputs<'_>,
) -> Result<
    SmallVec<[Option<LambertSolutionEx>; SCAN_BATCH_INLINE_SAMPLES]>,
    EvaluationArithmeticOverflow,
> {
    let ctx = inputs.ctx;
    let dv_cap = ctx.max_transfer_dv;
    // Same reasoning as `dv_cap` above, for the two branch bounds: their
    // `state1` half is fixed across the scan, so the sequential path rebuilt
    // it once per TOF sample.
    let departure_bounds = crate::lambert_backend::DepartureBoundCache::new(inputs.dep_at_release);

    let mut problems: SmallVec<[crate::lambert::MultiTofBranchProblem; SCAN_BATCH_INLINE_SAMPLES]> =
        SmallVec::new();
    let mut problem_slots: SmallVec<[Option<usize>; SCAN_BATCH_INLINE_SAMPLES]> = SmallVec::new();
    for (sample_index, &tof) in inputs.valid_tofs.iter().enumerate() {
        let Some(tgt_state) = inputs
            .tof_to_idx
            .get(sample_index)
            .and_then(|state_index| inputs.tgt_states.get(*state_index))
        else {
            problem_slots.push(None);
            continue;
        };
        let Some(branch_max_revs) = select_lambert_branch_ceiling(ctx) else {
            problem_slots.push(None);
            continue;
        };
        let branch_max_revs = crate::lambert_backend::max_revolutions_below_dv_cap_cached(
            &departure_bounds,
            tgt_state,
            tof,
            dv_cap,
            branch_max_revs,
        );
        let include_retrograde = crate::lambert_backend::retrograde_departure_dv_lower_bound_cached(
            &departure_bounds,
            tgt_state,
        ) < dv_cap;
        problem_slots.push(Some(problems.len()));
        problems.push(crate::lambert::MultiTofBranchProblem {
            state2: *tgt_state,
            tof,
            m_max: branch_max_revs,
            include_retrograde,
        });
    }

    let mut bests: SmallVec<[Option<LambertSolutionEx>; SCAN_BATCH_INLINE_SAMPLES]> =
        SmallVec::new();
    bests.resize(problems.len(), None);
    crate::lambert_backend::visit_lambert_branch_solutions_pruned_with_r1_multi_tof(
        inputs.r1_cache,
        inputs.dep_at_release,
        &problems,
        true,
        |problem_index, m, low_path, prograde, dv_vec, arrival_dv_vec| {
            let (Some(problem), Some(best)) =
                (problems.get(problem_index), bests.get_mut(problem_index))
            else {
                return;
            };
            fold_lambert_branch_candidate(
                ctx,
                &problem.state2,
                best,
                m,
                low_path,
                prograde,
                dv_vec,
                arrival_dv_vec,
            );
        },
    )?;

    let mut solutions: SmallVec<[Option<LambertSolutionEx>; SCAN_BATCH_INLINE_SAMPLES]> =
        SmallVec::new();
    for slot in &problem_slots {
        solutions.push(slot.and_then(|index| bests.get(index).copied().flatten()));
    }
    Ok(solutions)
}

// ============================================================================
// Target State Propagation with Caching
// ============================================================================

/// Propagate target states with the canonical MF/J2 model.
#[inline]
fn propagate_target_cached(
    ctx: &PlanContext,
    t_vals: &[f64],
    out_states: &mut [f64],
) -> Result<(), EvaluationArithmeticOverflow> {
    record_target_j2_batch_state_count(t_vals.len())?;
    equinoc_prop_j2_step_impl(&ctx.tgt_equ, t_vals, 0.0, out_states);
    Ok(())
}

/// SIMD variant for 4 states at once.
#[inline]
fn propagate_target_cached_simd4(
    ctx: &PlanContext,
    t_chunk: &[f64; 4],
    out_chunk: &mut [f64],
) -> Result<(), EvaluationArithmeticOverflow> {
    record_target_j2_simd4_chunk()?;
    equinoc_prop_j2_step_impl(&ctx.tgt_equ, t_chunk, 0.0, out_chunk);
    Ok(())
}

#[inline]
const fn target_propagation_uses_high_fidelity(ctx: &PlanContext) -> bool {
    ctx.execution_policy.use_high_fidelity
        && matches!(
            ctx.target_propagation_authority,
            crate::types::TargetPropagationAuthority::HighFidelity
        )
}

#[inline]
const fn target_propagation_uses_j2(ctx: &PlanContext) -> bool {
    matches!(
        ctx.target_propagation_authority,
        crate::types::TargetPropagationAuthority::MfJ2
    )
}

#[inline]
fn target_propagation_authority_is_consistent(ctx: &PlanContext) -> bool {
    let execution_policy_consistent = !matches!(
        ctx.target_propagation_authority,
        crate::types::TargetPropagationAuthority::HighFidelity
    ) || ctx.execution_policy.use_high_fidelity;
    let authority_consistent = crate::types::validate_target_propagation_authority(
        ctx.target_propagation_authority,
        ctx.target_body_force,
        ctx.force_config.as_deref(),
    )
    .is_ok();
    execution_policy_consistent && authority_consistent
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "each validated TOF maps to its exact six-scalar output lane"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "callers allocate exactly six output scalars per requested TOF"
)]
fn propagate_target_analytical(ctx: &PlanContext, t_vals: &[f64], out_states: &mut [f64]) {
    for (idx, dt) in t_vals.iter().copied().enumerate() {
        equinoc_prop_from_impl(&ctx.tgt_equ, dt, &mut out_states[idx * 6..idx * 6 + 6]);
    }
}

#[inline]
pub(crate) fn propagate_candidate_target_at_authoritative_offset(
    ctx: &PlanContext,
    dt: f64,
) -> Result<Option<[f64; 6]>, EvaluationArithmeticOverflow> {
    if !target_propagation_authority_is_consistent(ctx)
        || target_propagation_uses_high_fidelity(ctx)
        || ctx.execution_policy.require_high_fidelity
    {
        return Ok(None);
    }
    if target_propagation_uses_j2(ctx) {
        record_target_j2_scalar_state()?;
        let mut output = [0.0; 6];
        satpy_core::equinoc_prop_j2_from_impl(&ctx.tgt_equ, dt, &mut output);
        return Ok(all_finite(&output).then_some(output));
    }
    let mut output = [0.0; 6];
    equinoc_prop_from_impl(&ctx.tgt_equ, dt, &mut output);
    Ok(all_finite(&output).then_some(output))
}

pub(crate) fn propagate_high_fidelity_target_at_authoritative_offset_checked(
    ctx: &PlanContext,
    dt: f64,
) -> Result<[f64; 6], TransferPropagationFailure> {
    if !target_propagation_authority_is_consistent(ctx)
        || !target_propagation_uses_high_fidelity(ctx)
        || !ctx.execution_policy.use_high_fidelity
    {
        return Err(TransferPropagationFailure::Authority);
    }
    propagate_high_fidelity_state_at_epoch_checked(
        &ctx.tgt_equ,
        dt,
        ctx.epoch_jd,
        ctx.target_body_force,
        ctx,
    )
}

#[derive(Clone, Debug)]
pub struct EventResult {
    pub event_id: i32,
    pub mass: f64,
    pub total_dv: f64,
    pub success: bool,
}

impl Default for EventResult {
    fn default() -> Self {
        Self {
            event_id: -1,
            mass: f64::NAN,
            total_dv: f64::NAN,
            success: false,
        }
    }
}

/// Maximum TOF sample capacity in grid search. The per-solve sample count is
/// bounded by `SearchDepthPolicy::tof_sample_budget` (default 64), which keeps
/// unset policies bit-identical to pre-policy builds.
use crate::types::{all_finite, MAX_TOF_SAMPLES};

/// Minimum separation between TOF samples (seconds).
///
/// This is the only definition in the workspace. A dead `pub` namesake carrying
/// `0.005` sat in `types.rs` beside `MAX_TOF_SAMPLES` until it was removed; see
/// the note there before reintroducing a shared constant under this name.
const TOF_SAMPLE_SEPARATION: f64 = 120.0;

// J2 closure LIBRARY DEFAULTS (solve.rs DEFAULT_J2_* constants):
//   max_iterations = 8, endpoint_target_km = 0.01 (10 m), correction_step_gain = 0.7
//
// THESE ARE NOT WHAT PRODUCTION RUNS, and reading them as authoritative has
// already misled one audit. Part A compiles its own values
// (`nd_config/src/part_a_science.rs`, the `j2_max_iterations` /
// `j2_endpoint_target_km` / `j2_correction_step_gain` literals, routed
// through the closure built in `nd_pipeline/src/physics/transfer.rs`,
// which is the only path):
//   max_iterations = 5, endpoint_target_km = 0.01, correction_step_gain = 1.0
//
// The gain difference is load-bearing, not cosmetic. Measured over 118,648 loop
// entries: the miss-to-target-shift map is affine with Jacobian ~= I, so gain
// 1.0 IS the exact Newton step and converges in a median of 2 steps. Cost is
// perfectly symmetric in |1 - gain| -- +/-0.05 costs one extra step, +/-0.15
// costs three. At the library's 0.7 the median is 8 steps, so a build that
// actually ran gain 0.7 with max_iterations 5 would exhaust its budget and fail
// the acceptance gate on nearly every candidate.
//
// Changing either set requires updating both files.
// ============================================================================
// Helper functions
// ============================================================================

/// Check if all elements are finite
/// Copy 3 elements
#[inline]
const fn copy3(dst: &mut [f64; 3], src: &[f64; 3]) {
    *dst = *src;
}

#[inline]
fn rendezvous_arrival_dv(payload_state: &[f64; 6], target_state: &[f64; 6]) -> [f64; 3] {
    let [_, _, _, payload_velocity_x, payload_velocity_y, payload_velocity_z] = *payload_state;
    let [_, _, _, target_velocity_x, target_velocity_y, target_velocity_z] = *target_state;
    [
        target_velocity_x - payload_velocity_x,
        target_velocity_y - payload_velocity_y,
        target_velocity_z - payload_velocity_z,
    ]
}

#[inline]
fn transfer_dv_limit_penalty(final_transfer_dv_norm: f64, max_transfer_dv: f64) -> f64 {
    if final_transfer_dv_norm > max_transfer_dv {
        1000.0 + (final_transfer_dv_norm - max_transfer_dv) * 500.0
    } else {
        0.0
    }
}

/// Add velocity to state
#[inline]
fn add_velocity(state: &mut [f64; 6], dv: &[f64; 3]) {
    let [_, _, _, velocity_x, velocity_y, velocity_z] = state;
    let [delta_v_x, delta_v_y, delta_v_z] = *dv;
    *velocity_x += delta_v_x;
    *velocity_y += delta_v_y;
    *velocity_z += delta_v_z;
}

/// Euclidean distance between 3D points
#[inline]
#[expect(
    clippy::indexing_slicing,
    reason = "private helper receives exact length-three position slices at every call site"
)]
fn vec_distance(a: &[f64], b: &[f64]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute radius tolerance based on perigee/apogee bounds
#[inline]
fn radius_tolerance(min_perigee: f64, max_apogee: f64) -> f64 {
    const BASE_TOL: f64 = 0.1; // km
    let range = max_apogee - min_perigee;
    if range > 0.0 && range.is_finite() {
        BASE_TOL + range * 1e-5
    } else {
        BASE_TOL
    }
}

/// Return source epoch for a segment beginning after `elapsed_s` from a
/// stamped base.  Kept separate to make segment epoch construction auditable
/// in callers and tests.
#[inline]
pub(crate) fn propagation_epoch_for_segment(base_jd: f64, elapsed_s: f64) -> f64 {
    base_jd + elapsed_s / SEC_PER_DAY
}

/// Typed failure from one authoritative transfer propagation.
///
/// Keep the physical source intact until a caller deliberately classifies it
/// as a normal infeasible trial. Binary-eclipse and ephemeris failures are
/// numerical/authority failures, never empty plans.
#[derive(Clone, Debug, PartialEq)]
pub enum TransferPropagationFailure {
    ArithmeticOverflow,
    Census(lightyear_odeint_rs::probe::PropagationCensusError),
    InvalidInput,
    Authority,
    MissingHighFidelityAssets,
    Ephemeris(lightyear_odeint_rs::precomputed_ephem::EphemerisCoverageError),
    Final(lightyear_odeint_rs::integrator::FinalPropagationFailure),
    NonFiniteOutput,
}

impl TransferPropagationFailure {
    const fn disposition(&self) -> TransferPropagationDisposition {
        use lightyear_odeint_rs::integrator::FinalPropagationFailure as Final;
        use lightyear_odeint_rs::EclipseError;

        match self {
            Self::NonFiniteOutput
            | Self::Final(
                Final::Ground
                | Final::LeftEarth
                | Final::Eccentricity
                | Final::NanState
                | Final::EventInvalid
                | Final::IntegrationFailure
                | Final::Eclipse(
                    EclipseError::Geometry
                    | EclipseError::UninitializedSide
                    | EclipseError::NonProgress
                    | EclipseError::Chatter
                    | EclipseError::Bracket
                    | EclipseError::EventOverlap
                    | EclipseError::SplitLimit
                    | EclipseError::Envelope,
                ),
            ) => TransferPropagationDisposition::CandidateInfeasible,
            Self::ArithmeticOverflow
            | Self::Census(_)
            | Self::InvalidInput
            | Self::Authority
            | Self::MissingHighFidelityAssets
            | Self::Ephemeris(_)
            | Self::Final(
                Final::Gravity(_)
                | Final::Census(_)
                // Same reasoning as the eclipse-authority note below: the route
                // cannot run the requested stepper, which is true of every
                // candidate, not of this one. Calling it infeasible would drop
                // rows one at a time over a configuration fault.
                | Final::MethodUnsupported
                // FATAL, not infeasible, and the distinction is the whole point.
                // `EclipseError::Authority` means the strict-HF enclosure REFUSED
                // the configuration handed to it. Classifying that as
                // `CandidateInfeasible` says "this candidate did not work" about a
                // run whose inputs are wrong, so the campaign continues and quietly
                // drops every row. That is exactly what happened while
                // `eclipse_coordinator` flattened the refusal into
                // `EclipseError::Geometry`, which lives in the infeasible arm above.
                | Final::Eclipse(EclipseError::Gravity(_) | EclipseError::Authority(_)),
            ) => TransferPropagationDisposition::FatalAuthority,
        }
    }

    /// True only when the sealed force/ephemeris authority is valid and the
    /// individual candidate failed numerically or geometrically.
    #[must_use]
    pub const fn is_candidate_infeasible_under_valid_authority(&self) -> bool {
        matches!(
            self.disposition(),
            TransferPropagationDisposition::CandidateInfeasible
        )
    }

    /// True when continuing would hide missing, corrupt, or exhausted runtime
    /// authority rather than reject one candidate.
    #[must_use]
    pub const fn is_authority_failure(&self) -> bool {
        matches!(
            self.disposition(),
            TransferPropagationDisposition::FatalAuthority
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransferPropagationDisposition {
    CandidateInfeasible,
    FatalAuthority,
}

impl std::fmt::Display for TransferPropagationFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArithmeticOverflow => {
                formatter.write_str("transfer propagation arithmetic overflow")
            }
            Self::Census(error) => write!(formatter, "transfer propagation census: {error}"),
            Self::Ephemeris(error) => write!(formatter, "transfer ephemeris: {error}"),
            Self::Final(error) => write!(formatter, "transfer propagation: {error}"),
            Self::InvalidInput => formatter.write_str("transfer propagation invalid input"),
            Self::Authority => formatter.write_str("transfer propagation authority mismatch"),
            Self::MissingHighFidelityAssets => {
                formatter.write_str("transfer propagation missing high-fidelity assets")
            }
            Self::NonFiniteOutput => formatter.write_str("transfer propagation non-finite output"),
        }
    }
}

impl std::error::Error for TransferPropagationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Census(error) => Some(error),
            Self::Ephemeris(error) => Some(error),
            Self::Final(error) => Some(error),
            Self::ArithmeticOverflow
            | Self::InvalidInput
            | Self::Authority
            | Self::MissingHighFidelityAssets
            | Self::NonFiniteOutput => None,
        }
    }
}

impl From<lightyear_odeint_rs::integrator::FinalPropagationFailure> for TransferPropagationFailure {
    fn from(failure: lightyear_odeint_rs::integrator::FinalPropagationFailure) -> Self {
        match failure {
            lightyear_odeint_rs::integrator::FinalPropagationFailure::Census(error) => {
                Self::Census(error)
            }
            failure => Self::Final(failure),
        }
    }
}

#[cfg(test)]
mod transfer_propagation_failure_tests {
    use super::TransferPropagationFailure;
    use std::error::Error as _;

    #[test]
    fn arithmetic_overflow_failure_has_stable_display() {
        assert_eq!(
            TransferPropagationFailure::ArithmeticOverflow.to_string(),
            "transfer propagation arithmetic overflow"
        );
    }

    #[test]
    fn census_failure_preserves_its_exact_cause() {
        let failure = TransferPropagationFailure::Census(
            lightyear_odeint_rs::probe::PropagationCensusError::MutexPoisoned,
        );

        assert_eq!(
            failure.to_string(),
            "transfer propagation census: propagation census mutex poisoned"
        );
        assert_eq!(
            failure.source().map(ToString::to_string).as_deref(),
            Some("propagation census mutex poisoned")
        );
    }

    #[test]
    fn final_census_failure_unwraps_at_the_transfer_boundary() {
        let failure = TransferPropagationFailure::from(
            lightyear_odeint_rs::integrator::FinalPropagationFailure::Census(
                lightyear_odeint_rs::probe::PropagationCensusError::Allocation,
            ),
        );

        assert_eq!(
            failure,
            TransferPropagationFailure::Census(
                lightyear_odeint_rs::probe::PropagationCensusError::Allocation,
            )
        );
    }

    #[test]
    fn failure_classification_is_exhaustive_and_disjoint() {
        use super::TransferPropagationDisposition;
        use lightyear_odeint_rs::integrator::FinalPropagationFailure as Final;
        use lightyear_odeint_rs::probe::PropagationCensusError;
        use lightyear_odeint_rs::EclipseError;
        use satpy_core::GravityError;

        /// The expected classification, restated independently of
        /// `disposition` as an exhaustive match with no wildcard at any
        /// depth: a variant added anywhere in the failure tree fails to
        /// compile HERE until this test classifies it.
        fn expected_disposition(
            failure: &TransferPropagationFailure,
        ) -> TransferPropagationDisposition {
            match failure {
                TransferPropagationFailure::ArithmeticOverflow
                | TransferPropagationFailure::Census(_)
                | TransferPropagationFailure::InvalidInput
                | TransferPropagationFailure::Authority
                | TransferPropagationFailure::MissingHighFidelityAssets
                | TransferPropagationFailure::Ephemeris(_) => {
                    TransferPropagationDisposition::FatalAuthority
                }
                TransferPropagationFailure::NonFiniteOutput => {
                    TransferPropagationDisposition::CandidateInfeasible
                }
                TransferPropagationFailure::Final(failure) => match failure {
                    Final::Ground
                    | Final::LeftEarth
                    | Final::Eccentricity
                    | Final::NanState
                    | Final::EventInvalid
                    | Final::IntegrationFailure => {
                        TransferPropagationDisposition::CandidateInfeasible
                    }
                    Final::Gravity(_) | Final::Census(_) | Final::MethodUnsupported => {
                        TransferPropagationDisposition::FatalAuthority
                    }
                    Final::Eclipse(eclipse) => match eclipse {
                        EclipseError::Gravity(_) | EclipseError::Authority(_) => {
                            TransferPropagationDisposition::FatalAuthority
                        }
                        EclipseError::Geometry
                        | EclipseError::UninitializedSide
                        | EclipseError::NonProgress
                        | EclipseError::Chatter
                        | EclipseError::Bracket
                        | EclipseError::EventOverlap
                        | EclipseError::SplitLimit
                        | EclipseError::Envelope => {
                            TransferPropagationDisposition::CandidateInfeasible
                        }
                    },
                },
            }
        }

        // One witness value per leaf the match above distinguishes.
        // `EclipseError::Authority` has no witness: its payload type is
        // deliberately private to lightyear_odeint_rs, so it is covered by
        // the compile-time arm above but cannot be exercised from here.
        let witnesses = [
            TransferPropagationFailure::ArithmeticOverflow,
            TransferPropagationFailure::Census(PropagationCensusError::MutexPoisoned),
            TransferPropagationFailure::InvalidInput,
            TransferPropagationFailure::Authority,
            TransferPropagationFailure::MissingHighFidelityAssets,
            TransferPropagationFailure::Ephemeris(
                lightyear_odeint_rs::precomputed_ephem::EphemerisCoverageError::NonFiniteArc {
                    jd_a: f64::NAN,
                    jd_b: f64::NAN,
                },
            ),
            TransferPropagationFailure::NonFiniteOutput,
            TransferPropagationFailure::Final(Final::Ground),
            TransferPropagationFailure::Final(Final::LeftEarth),
            TransferPropagationFailure::Final(Final::Eccentricity),
            TransferPropagationFailure::Final(Final::NanState),
            TransferPropagationFailure::Final(Final::EventInvalid),
            TransferPropagationFailure::Final(Final::IntegrationFailure),
            TransferPropagationFailure::Final(Final::Gravity(GravityError::InvalidRadius)),
            TransferPropagationFailure::Final(Final::Census(PropagationCensusError::Allocation)),
            TransferPropagationFailure::Final(Final::Eclipse(EclipseError::Gravity(
                GravityError::InvalidState,
            ))),
            TransferPropagationFailure::Final(Final::Eclipse(EclipseError::Geometry)),
            TransferPropagationFailure::Final(Final::Eclipse(EclipseError::UninitializedSide)),
            TransferPropagationFailure::Final(Final::Eclipse(EclipseError::NonProgress)),
            TransferPropagationFailure::Final(Final::Eclipse(EclipseError::Chatter)),
            TransferPropagationFailure::Final(Final::Eclipse(EclipseError::Bracket)),
            TransferPropagationFailure::Final(Final::Eclipse(EclipseError::EventOverlap)),
            TransferPropagationFailure::Final(Final::Eclipse(EclipseError::SplitLimit)),
            TransferPropagationFailure::Final(Final::Eclipse(EclipseError::Envelope)),
        ];

        for failure in witnesses {
            let candidate_infeasible = failure.is_candidate_infeasible_under_valid_authority();
            let authority_failure = failure.is_authority_failure();
            assert_ne!(candidate_infeasible, authority_failure, "{failure:?}");
            let expected = expected_disposition(&failure);
            assert_eq!(
                candidate_infeasible,
                expected == TransferPropagationDisposition::CandidateInfeasible,
                "{failure:?}"
            );
            assert_eq!(
                authority_failure,
                expected == TransferPropagationDisposition::FatalAuthority,
                "{failure:?}"
            );
        }
    }
}

/// Resolve one body-specific force model across a stamped absolute arc.
///
/// This is shared by scalar and multi-TOF HF propagation so force authority
/// cannot silently differ between lanes. Hybrid transfer and target arcs use
/// sealed body-specific 5x5-gravity + drag + SRP + Sun + Moon tuples.
/// MF propagation follows the separate analytical J2 path.
fn stamped_body_force_config(
    ctx: &PlanContext,
    source_jd: f64,
    dt: f64,
    body_force: BodyForceConfig,
) -> Result<lightyear_odeint_rs::types::ForceConfig, TransferPropagationFailure> {
    let mut config = *ctx
        .force_config
        .as_ref()
        .ok_or(TransferPropagationFailure::MissingHighFidelityAssets)?
        .as_ref();
    config.am_ratio = body_force.am_ratio;
    config.cd = body_force.cd;
    config.cr = body_force.cr;
    config
        .with_ephemeris_for_arc(source_jd, source_jd + dt / SEC_PER_DAY)
        .map_err(TransferPropagationFailure::Ephemeris)
}

/// Propagate one target state to multiple absolute offsets in one HF arc.
///
/// Input order is arbitrary. Integration uses a sorted, deduplicated grid,
/// then maps complete results back to caller order. Any terminal/integration
/// failure leaves caller output untouched; callers must never silently retry
/// under a different execution path.
/// Propagate one target onto an arbitrarily long, strictly increasing offset
/// grid in a single integration.
///
/// [`propagate_high_fidelity_target_multi_tof_checked`] already integrates once
/// for many offsets, but caps at [`MAX_TOF_SAMPLES`] because it sorts and
/// dedupes into fixed-size buffers on the stack. A dense fourteen-day grid does
/// not need that machinery: the caller supplies the grid already ordered, so
/// there is nothing to sort, nothing to dedupe, and no reason for a cap.
/// Chunking a dense grid through the capped entry point instead would
/// re-integrate from the epoch once per chunk, which is the cost this exists to
/// avoid.
///
/// The grid is reconstructed from the solver's own accepted steps
/// ([`SampledOutputMode::Interpolated`]). Forcing solver steps onto a coarse
/// dense grid instead breaks eclipse bracketing: the event scanner requires a
/// certified relative-motion bound between consecutive scan endpoints, and a
/// LEO object crosses several hundred kilometres in one 60 s step, so the
/// bracket check fails outright. Letting the solver keep its own step control
/// also keeps the arc at its natural cost.
///
/// `offsets_s` must be finite, strictly increasing and strictly positive.
pub(crate) fn propagate_high_fidelity_target_dense_grid_checked(
    ctx: &PlanContext,
    offsets_s: &[f64],
    out_states: &mut [[f64; 6]],
) -> Result<(), TransferPropagationFailure> {
    if offsets_s.is_empty()
        || offsets_s.len() != out_states.len()
        || !target_propagation_authority_is_consistent(ctx)
        || !target_propagation_uses_high_fidelity(ctx)
        || !ctx.execution_policy.use_high_fidelity
        || offsets_s
            .first()
            .is_none_or(|offset| !offset.is_finite() || *offset <= 0.0)
        || offsets_s.windows(2).any(|window| {
            let [lower, upper] = window else {
                return true;
            };
            !upper.is_finite() || *upper <= *lower
        })
    {
        return Err(TransferPropagationFailure::InvalidInput);
    }
    let t_final_s = *offsets_s
        .last()
        .ok_or(TransferPropagationFailure::InvalidInput)?;

    let packed = ctx
        .packed_coeffs
        .as_ref()
        .ok_or(TransferPropagationFailure::MissingHighFidelityAssets)?;
    let config = std::sync::Arc::new(stamped_body_force_config(
        ctx,
        ctx.epoch_jd,
        t_final_s,
        ctx.target_body_force,
    )?);
    let gravity = lightyear_odeint_rs::ScalarGravityAssets::new(packed.clone());
    let propagation_context =
        lightyear_odeint_rs::ScalarPropagationContext::new(ctx.epoch_jd, config, gravity);
    let request = lightyear_odeint_rs::ScalarPropagationRequest::new(
        &propagation_context,
        ctx.tgt_equ,
        offsets_s,
        0.0,
        t_final_s,
    )
    .with_events(true)
    .with_output_mode(lightyear_odeint_rs::SampledOutputMode::Interpolated);
    let result = lightyear_odeint_rs::integrate_adaptive(request)
        .map_err(TransferPropagationFailure::Census)?;
    if let Some(failure) = lightyear_odeint_rs::integrator::final_propagation_failure(&result) {
        return Err(TransferPropagationFailure::from(failure));
    }
    if result.max_steps_exceeded
        || result.times.len() != offsets_s.len()
        || result.states.len() != offsets_s.len()
    {
        return Err(TransferPropagationFailure::Final(
            lightyear_odeint_rs::integrator::FinalPropagationFailure::IntegrationFailure,
        ));
    }

    for (((actual_time, delta), expected_time), out_state) in result
        .times
        .iter()
        .copied()
        .zip(&result.states)
        .zip(offsets_s.iter().copied())
        .zip(out_states.iter_mut())
    {
        let time_tol = 8.0 * f64::EPSILON * expected_time.abs().max(actual_time.abs()).max(1.0);
        if !actual_time.is_finite() || (actual_time - expected_time).abs() > time_tol {
            return Err(TransferPropagationFailure::Final(
                lightyear_odeint_rs::integrator::FinalPropagationFailure::IntegrationFailure,
            ));
        }
        equinoc_prop_from_impl(&ctx.tgt_equ, actual_time, out_state);
        for (state_component, delta_component) in out_state.iter_mut().zip(delta) {
            *state_component += *delta_component;
        }
        if !all_finite(out_state) {
            return Err(TransferPropagationFailure::NonFiniteOutput);
        }
    }
    Ok(())
}

pub(crate) fn propagate_high_fidelity_target_multi_tof_checked(
    ctx: &PlanContext,
    offsets_s: &[f64],
    out_states: &mut [[f64; 6]],
) -> Result<(), TransferPropagationFailure> {
    #[cfg(test)]
    {
        let forced_failure = HF_MULTI_TOF_TEST_CALLS.with(|counts| {
            let mut counts = counts.borrow_mut();
            counts.0 = counts
                .0
                .checked_add(1)
                .ok_or(TransferPropagationFailure::InvalidInput)?;
            Ok::<_, TransferPropagationFailure>(counts.2)
        })?;
        if forced_failure {
            return Err(TransferPropagationFailure::Final(
                lightyear_odeint_rs::integrator::FinalPropagationFailure::IntegrationFailure,
            ));
        }
    }
    if offsets_s.is_empty()
        || offsets_s.len() != out_states.len()
        || offsets_s.len() > MAX_TOF_SAMPLES
        || !target_propagation_authority_is_consistent(ctx)
        || !target_propagation_uses_high_fidelity(ctx)
        || offsets_s
            .iter()
            .any(|offset| !offset.is_finite() || *offset <= 0.0)
    {
        return Err(TransferPropagationFailure::InvalidInput);
    }

    let mut order: SmallVec<[(OrderedFloat<f64>, usize); MAX_TOF_SAMPLES]> = offsets_s
        .iter()
        .copied()
        .enumerate()
        .map(|(index, offset)| (OrderedFloat(offset), index))
        .collect();
    pdqsort::sort_by_key(&mut order, |(offset, _)| *offset);

    let mut unique_offsets = [0.0_f64; MAX_TOF_SAMPLES];
    let mut unique_for_original = [0_usize; MAX_TOF_SAMPLES];
    let mut unique_count = 0_usize;
    for (offset, original_index) in order.iter().copied() {
        let offset = offset.into_inner();
        let is_new = unique_count
            .checked_sub(1)
            .and_then(|index| unique_offsets.get(index))
            .is_none_or(|previous_offset| previous_offset.to_bits() != offset.to_bits());
        if is_new {
            let Some(unique_offset) = unique_offsets.get_mut(unique_count) else {
                return Err(TransferPropagationFailure::InvalidInput);
            };
            *unique_offset = offset;
            unique_count = unique_count
                .checked_add(1)
                .ok_or(TransferPropagationFailure::InvalidInput)?;
        }
        let Some(original_slot) = unique_for_original.get_mut(original_index) else {
            return Err(TransferPropagationFailure::InvalidInput);
        };
        *original_slot = unique_count
            .checked_sub(1)
            .ok_or(TransferPropagationFailure::InvalidInput)?;
    }
    let tf = *unique_offsets
        .get(
            unique_count
                .checked_sub(1)
                .ok_or(TransferPropagationFailure::InvalidInput)?,
        )
        .ok_or(TransferPropagationFailure::InvalidInput)?;
    let unique_offsets_used = unique_offsets
        .get(..unique_count)
        .ok_or(TransferPropagationFailure::InvalidInput)?;

    let packed = ctx
        .packed_coeffs
        .as_ref()
        .ok_or(TransferPropagationFailure::MissingHighFidelityAssets)?;
    let config = stamped_body_force_config(ctx, ctx.epoch_jd, tf, ctx.target_body_force)?;
    #[cfg(test)]
    let hf_grid_start = crate::types::verified_superset_deep_telemetry_enabled().then(Instant::now);
    let mut unique_states = [[f64::NAN; 6]; MAX_TOF_SAMPLES];
    let config = std::sync::Arc::new(config);
    let gravity = lightyear_odeint_rs::ScalarGravityAssets::new(packed.clone());
    let propagation_context =
        lightyear_odeint_rs::ScalarPropagationContext::new(ctx.epoch_jd, config, gravity);
    let request = lightyear_odeint_rs::ScalarPropagationRequest::new(
        &propagation_context,
        ctx.tgt_equ,
        unique_offsets_used,
        0.0,
        tf,
    )
    .with_events(true)
    .with_output_mode(lightyear_odeint_rs::SampledOutputMode::ForceEvaluationTimes);
    let result = lightyear_odeint_rs::integrate_adaptive(request)
        .map_err(TransferPropagationFailure::Census)?;
    if let Some(failure) = lightyear_odeint_rs::integrator::final_propagation_failure(&result) {
        return Err(TransferPropagationFailure::from(failure));
    }
    if result.max_steps_exceeded
        || result.times.len() != unique_count
        || result.states.len() != unique_count
    {
        return Err(TransferPropagationFailure::Final(
            lightyear_odeint_rs::integrator::FinalPropagationFailure::IntegrationFailure,
        ));
    }
    #[cfg(test)]
    HF_MULTI_TOF_TEST_REBASE_OBSERVED
        .with(|observed| *observed.borrow_mut() = result.perturb_deviation_fired);

    for (index, ((actual_time, delta), expected_time)) in result
        .times
        .iter()
        .copied()
        .zip(&result.states)
        .zip(unique_offsets_used.iter().copied())
        .enumerate()
    {
        let time_tol = 8.0 * f64::EPSILON * expected_time.abs().max(actual_time.abs()).max(1.0);
        if !actual_time.is_finite() || (actual_time - expected_time).abs() > time_tol {
            return Err(TransferPropagationFailure::Final(
                lightyear_odeint_rs::integrator::FinalPropagationFailure::IntegrationFailure,
            ));
        }
        let Some(unique_state) = unique_states.get_mut(index) else {
            return Err(TransferPropagationFailure::InvalidInput);
        };
        equinoc_prop_from_impl(&ctx.tgt_equ, actual_time, unique_state);
        for (state_component, delta_component) in unique_state.iter_mut().zip(delta) {
            *state_component += *delta_component;
        }
        if !all_finite(unique_state) {
            return Err(TransferPropagationFailure::NonFiniteOutput);
        }
    }

    for (out_state, &unique_index) in out_states
        .iter_mut()
        .zip(unique_for_original.iter().take(offsets_s.len()))
    {
        *out_state = *unique_states
            .get(unique_index)
            .ok_or(TransferPropagationFailure::InvalidInput)?;
    }
    #[cfg(test)]
    if let Some(start) = hf_grid_start {
        record_hf_propagation_stage(
            HfPropagationStage::TargetGrid {
                requested_states: offsets_s.len(),
                unique_attempted_states: unique_count,
            },
            start.elapsed().as_secs_f64(),
        )
        .map_err(|_| TransferPropagationFailure::ArithmeticOverflow)?;
    }
    Ok(())
}

/// Propagate one explicitly stamped ECI segment.
///
/// `source_jd` is the epoch of `equ`/`eci`, not the base-event epoch.  The
/// Lightyear force model consumes this exact source epoch, so a second segment
/// cannot silently restart ephemeris time at event launch.  `body_force`
/// carries the body role and non-gravitational coefficients for this arc.
/// Shared guard-and-asset prologue of
/// [`propagate_high_fidelity_state_at_epoch_checked`] and its `_observed`
/// twin: the authority checks, the gravity-asset lookup, and the stamped
/// force config, in the exact order both bodies used to spell inline.
///
/// Control flow and asset plumbing only — no numerical work — so sharing it
/// moves no bits. It exists because the `_observed` body hand-copied these
/// predicates verbatim, and a hand-copied guard misses the next case added
/// to only one copy.
fn checked_hf_state_prologue(
    dt: f64,
    source_jd: f64,
    body_force: BodyForceConfig,
    ctx: &PlanContext,
) -> Result<
    (
        std::sync::Arc<satpy_core::PackedGravityCoeffs>,
        lightyear_odeint_rs::types::ForceConfig,
    ),
    TransferPropagationFailure,
> {
    if !source_jd.is_finite() || !dt.is_finite() {
        return Err(TransferPropagationFailure::InvalidInput);
    }
    if !ctx.execution_policy.use_high_fidelity
        || body_force.fidelity != PropagationFidelity::HighFidelity
    {
        return Err(TransferPropagationFailure::Authority);
    }
    if matches!(body_force.role, BodyRole::DiagnosticTarget)
        && (!body_force.matches_exact(ctx.target_body_force)
            || crate::types::validate_target_body_force(
                ctx.target_propagation_authority,
                body_force,
            )
            .is_err())
    {
        return Err(TransferPropagationFailure::Authority);
    }
    if ctx.execution_policy.use_high_fidelity
        && matches!(body_force.role, BodyRole::TransferVehicle)
        && (body_force.fidelity != PropagationFidelity::HighFidelity
            || !body_force.am_ratio.is_finite()
            || body_force.am_ratio <= 0.0
            || !body_force.cd.is_finite()
            || body_force.cd <= 0.0
            || !body_force.cr.is_finite()
            || body_force.cr < 0.0)
    {
        return Err(TransferPropagationFailure::Authority);
    }
    let packed = ctx
        .packed_coeffs
        .as_ref()
        .ok_or(TransferPropagationFailure::MissingHighFidelityAssets)?;
    let config = stamped_body_force_config(ctx, source_jd, dt, body_force)?;
    Ok((packed.clone(), config))
}

/// Shared `dt == 0` boundary for the three checked HF propagation twins:
/// the analytic state at zero offset, finiteness-gated.
fn hf_zero_dt_state(equ: &[f64; 6]) -> Result<[f64; 6], TransferPropagationFailure> {
    let mut output = [0.0; 6];
    equinoc_prop_from_impl(equ, 0.0, &mut output);
    all_finite(&output)
        .then_some(output)
        .ok_or(TransferPropagationFailure::NonFiniteOutput)
}

/// Shared baseline-add epilogue for the three checked HF propagation twins:
/// rebuild the analytic baseline at `dt` and add the integrated delta
/// component-wise. The component addition order is exactly the one every
/// caller spelled out, so the extraction is bit-neutral.
fn hf_baseline_add_epilogue(
    equ: &[f64; 6],
    dt: f64,
    final_delta: &[f64; 6],
) -> Result<[f64; 6], TransferPropagationFailure> {
    let mut output = [0.0; 6];
    let mut baseline = [0.0; 6];
    equinoc_prop_from_impl(equ, dt, &mut baseline);
    for ((output_component, baseline_component), delta_component) in
        output.iter_mut().zip(&baseline).zip(final_delta)
    {
        *output_component = *baseline_component + *delta_component;
    }
    all_finite(&output)
        .then_some(output)
        .ok_or(TransferPropagationFailure::NonFiniteOutput)
}

pub(crate) fn propagate_high_fidelity_state_at_epoch_checked(
    equ: &[f64; 6],
    dt: f64,
    source_jd: f64,
    body_force: BodyForceConfig,
    ctx: &PlanContext,
) -> Result<[f64; 6], TransferPropagationFailure> {
    let (packed, config) = checked_hf_state_prologue(dt, source_jd, body_force, ctx)?;
    if dt == 0.0 {
        return hf_zero_dt_state(equ);
    }
    let config = std::sync::Arc::new(config);
    let gravity = lightyear_odeint_rs::ScalarGravityAssets::new(packed);
    let propagation_context =
        lightyear_odeint_rs::ScalarPropagationContext::new(source_jd, config, gravity);
    let t_eval = [dt];
    let request = lightyear_odeint_rs::ScalarPropagationRequest::new(
        &propagation_context,
        *equ,
        &t_eval,
        0.0,
        dt,
    )
    .with_events(true);
    let final_delta = lightyear_odeint_rs::integrate_final_checked(request)
        .map_err(TransferPropagationFailure::from)?;
    hf_baseline_add_epilogue(equ, dt, &final_delta)
}

/// Propagate one stamped HF segment through the independent fixed-Ic witness.
///
/// # Errors
///
/// Returns [`TransferPropagationFailure::InvalidInput`] for a non-finite `dt`
/// or `source_jd`, [`TransferPropagationFailure::Authority`] when the context
/// or the stamped body force is not the sealed strict-HF authority,
/// [`TransferPropagationFailure::Final`] when the witness integration itself
/// fails, and [`TransferPropagationFailure::NonFiniteOutput`] when the
/// resulting state is not finite.
pub fn propagate_high_fidelity_state_independent_witness(
    equ: &[f64; 6],
    dt: f64,
    source_jd: f64,
    body_force: BodyForceConfig,
    ctx: &PlanContext,
    dt_max_s: f64,
    tolerance: f64,
) -> Result<[f64; 6], TransferPropagationFailure> {
    let (packed, config) = checked_hf_state_prologue(dt, source_jd, body_force, ctx)?;
    if dt == 0.0 {
        return hf_zero_dt_state(equ);
    }
    let gravity = lightyear_odeint_rs::ScalarGravityAssets::new(packed);
    let propagation_context = lightyear_odeint_rs::ScalarPropagationContext::new(
        source_jd,
        std::sync::Arc::new(config),
        gravity,
    );
    let final_delta = lightyear_odeint_rs::independent_witness::integrate_fixed_ic_witness(
        &propagation_context,
        *equ,
        0.0,
        dt,
        dt_max_s,
        tolerance,
    )
    .map_err(TransferPropagationFailure::from)?;
    hf_baseline_add_epilogue(equ, dt, &final_delta)
}

/// Feature-only result of one actual checked scalar propagation.
///
/// `scalar_observation` is absent only when no scalar solve starts (for
/// example a zero-duration analytical boundary). The qualification caller
/// records it directly and fails closed on a missing expected observation.
#[cfg(feature = "solver-qualification")]
pub(crate) struct ObservedHighFidelityState {
    pub(crate) outcome: Result<[f64; 6], TransferPropagationFailure>,
    pub(crate) scalar_observation: Option<lightyear_odeint_rs::ObservedFinalLeg>,
}

/// Execute the same checked-final numerical core with feature-only local
/// metrics. It accepts no alternate force, solver, or asset controls.
#[cfg(feature = "solver-qualification")]
pub(crate) fn propagate_high_fidelity_state_at_epoch_checked_observed(
    equ: &[f64; 6],
    dt: f64,
    source_jd: f64,
    body_force: BodyForceConfig,
    ctx: &PlanContext,
) -> Result<ObservedHighFidelityState, TransferPropagationFailure> {
    let (packed, config) = checked_hf_state_prologue(dt, source_jd, body_force, ctx)?;
    if dt == 0.0 {
        return Ok(ObservedHighFidelityState {
            outcome: hf_zero_dt_state(equ),
            scalar_observation: None,
        });
    }
    let config = std::sync::Arc::new(config);
    let gravity = lightyear_odeint_rs::ScalarGravityAssets::new(packed);
    let propagation_context =
        lightyear_odeint_rs::ScalarPropagationContext::new(source_jd, config, gravity);
    let t_eval = [dt];
    let observed = lightyear_odeint_rs::integrate_final_checked_observed(
        lightyear_odeint_rs::ScalarPropagationRequest::new(
            &propagation_context,
            *equ,
            &t_eval,
            0.0,
            dt,
        )
        .with_events(true),
    );
    let outcome = observed.outcome.map_or_else(
        |failure| Err(TransferPropagationFailure::from(failure)),
        |final_delta| hf_baseline_add_epilogue(equ, dt, &final_delta),
    );
    Ok(ObservedHighFidelityState {
        outcome,
        scalar_observation: Some(observed),
    })
}

/// Infallible candidate policy for MF/J2 search only.
///
/// Strict-HF contexts are rejected before any propagation. A typed strict
/// failure must use [`propagate_high_fidelity_state_at_epoch_checked`].
pub(crate) fn propagate_candidate_state_at_epoch(
    equ: &[f64; 6],
    dt: f64,
    source_jd: f64,
    body_force: BodyForceConfig,
    ctx: &PlanContext,
) -> Result<Option<[f64; 6]>, EvaluationArithmeticOverflow> {
    if !source_jd.is_finite()
        || !dt.is_finite()
        || ctx.execution_policy.use_high_fidelity
        || ctx.execution_policy.require_high_fidelity
        || body_force.fidelity == PropagationFidelity::HighFidelity
    {
        return Ok(None);
    }
    if matches!(body_force.role, BodyRole::DiagnosticTarget)
        && (!body_force.matches_exact(ctx.target_body_force)
            || crate::types::validate_target_body_force(
                ctx.target_propagation_authority,
                body_force,
            )
            .is_err())
    {
        return Ok(None);
    }
    record_j2_propagate_state()?;
    let mut output = [0.0; 6];
    equinoc_prop_j2_from_impl(equ, dt, &mut output);
    Ok(all_finite(&output).then_some(output))
}

/// MF/J2 propagation used inside candidate search.
///
/// Strict-HF propagation is not a candidate-search mode; it starts only after
/// the MF front at the checked replay/lowering boundary.
fn propagate_candidate_search_state_at_epoch(
    equ: &[f64; 6],
    dt: f64,
    source_jd: f64,
    body_force: BodyForceConfig,
    ctx: &PlanContext,
) -> Result<Option<[f64; 6]>, EvaluationArithmeticOverflow> {
    propagate_candidate_state_at_epoch(equ, dt, source_jd, body_force, ctx)
}

#[inline]
fn pre_hf_j2_residual_blocks_acceptance(
    use_high_fidelity: bool,
    residual_m: f64,
    tolerance_m: f64,
) -> bool {
    !(use_high_fidelity || residual_m.is_finite() && residual_m <= tolerance_m)
}

#[inline]
pub(crate) fn post_hf_residual_accepts(residual_m: f64, tolerance_m: f64) -> bool {
    residual_m.is_finite() && residual_m <= tolerance_m
}

#[inline]
pub(crate) fn compute_dep_period(ctx: &PlanContext) -> f64 {
    let mut dep_period = ctx.dep_period;
    if dep_period <= 0.0 {
        if let Some(orbit) = EciBasicOrbit::from_eci(&ctx.dep_eci) {
            if orbit.sma > 0.0 {
                let sma_cubed = orbit.sma * orbit.sma * orbit.sma;
                dep_period = std::f64::consts::TAU * (sma_cubed / MU).sqrt();
            }
        }
        if dep_period <= 0.0 {
            // Fallback to circular period at current radius if all else fails
            let r = norm3(&[ctx.dep_eci[0], ctx.dep_eci[1], ctx.dep_eci[2]]);
            if r > 0.0 {
                let sma_cubed = r * r * r;
                dep_period = std::f64::consts::TAU * (sma_cubed / MU).sqrt();
            }
        }
    }
    dep_period
}

pub(crate) fn evaluate_plan_from_phase_with_lambert_scratch(
    x: &[f64; 3],
    ctx: &PlanContext,
    coarse_mode: bool,
    time2phase: f64,
    waittime: f64,
    dep_period: f64,
    dep_at_phase: &[f64; 6],
    dep_phase_orbit_override: Option<EciOrbitSummary>,
    variable_r2_scratch: Option<&mut crate::lambert::VariableR2LambertScratch>,
) -> Result<PlanResult, EvaluationArithmeticOverflow> {
    if !candidate_search_is_supported(ctx) {
        return Ok(unsupported_candidate_search_result());
    }
    let mut result = evaluate_plan_from_phase_with_lambert_scratch_impl(
        x,
        ctx,
        coarse_mode,
        time2phase,
        waittime,
        dep_period,
        dep_at_phase,
        dep_phase_orbit_override,
        variable_r2_scratch,
    )?;
    result.replay_provenance = replay_provenance_from_context(ctx);
    Ok(result)
}

const fn replay_provenance_from_context(ctx: &PlanContext) -> ReplayProvenance {
    ReplayProvenance {
        launch_pre_impulse_state: ctx.dep_eci,
        base_epoch_jd: ctx.epoch_jd,
        max_time_s: ctx.max_time_s,
        max_phase_dv: ctx.max_phase_dv,
        max_transfer_dv: ctx.max_transfer_dv,
        revolution_cap: ctx.revolution_cap,
        min_perigee: ctx.min_perigee,
        max_apogee: ctx.max_apogee,
        distance_tol: ctx.distance_tol,
        deployer_min_distance: ctx.deployer_min_distance,
        max_revs: ctx.max_revs,
        target_propagation_mode: ctx.target_propagation_authority.as_force_config_code(),
        target_am_ratio: ctx.target_body_force.am_ratio,
        target_cd: ctx.target_body_force.cd,
        target_cr: ctx.target_body_force.cr,
    }
}

/// Strategic TOF sample ladder, stamped verbatim into BOTH plan evaluators
/// (the scalar `evaluate_plan_from_phase_with_lambert_scratch_impl` and the
/// branch-fanout `prepare_branch_shared_work`): Hohmann and variants, synodic
/// samples, multi-rev, minimum-energy, revolution-entry, large-plane-change
/// extras, then uniform span sampling and dedup.
///
/// Every named geometry point is a sampling priority inside the caller's hard
/// physical interval, never an admissibility bound. Both callers pass the
/// interval from [`admissible_tof_interval`], so the uniform tail covers its
/// lower and upper endpoints even when every strategic point lies elsewhere.
///
/// The two call sites historically carried near-verbatim copies with micro
/// drift (`x*x*x` vs `.powi(3)` period spellings, `dep_period` vs
/// `phase_period` naming, different bail values). Divergent pieces stay at the
/// call sites as VERBATIM token arguments (`*_setup` statement lists and
/// `*_sample` expressions), so each site's exact FP expression trees are
/// reproduced token-for-token and expansion is bit-identical to the retired
/// copies. Every caller local the shared body touches is passed as an ident
/// argument (macro hygiene makes that mandatory, which is what keeps the
/// contract explicit). Loop binders (`multi_rev_var`, `rev_entry_var`) come
/// from the call site so the site's sample tokens can reference them.
macro_rules! tof_sample_ladder {
    (
        ctx = $ctx:ident,
        phase_sma = $phase_sma:ident,
        dep_at_release = $dep_at_release:ident,
        plane_angle_cached = $plane_angle_cached:ident,
        tof_lower = $tof_lower:ident,
        tof_upper = $tof_upper:ident,
        span = $span:ident,
        sample_count = $sample_count:ident,
        is_simple_transfer = $is_simple_transfer:ident,
        tof_budget = $tof_budget:ident,
        tof_samples = $tof_samples:ident,
        tof_sample_n = $tof_sample_n:ident,
        hohmann_tof = $hohmann_tof:ident,
        tgt_period = $tgt_period:ident,
        period = $period:ident,
        period_setup = [$($period_setup:tt)*],
        transfer_sma_hohmann = $transfer_sma_hohmann:ident,
        transfer_period = $transfer_period:ident,
        transfer_period_setup = [$($transfer_period_setup:tt)*],
        multi_rev_var = $multi_rev_var:ident,
        multi_rev_setup = [$($multi_rev_setup:tt)*],
        multi_rev_sample = [$($multi_rev_sample:tt)*],
        rev_entry_var = $rev_entry_var:ident,
        rev_entry_setup = [$($rev_entry_setup:tt)*],
        rev_entry_sample = [$($rev_entry_sample:tt)*],
        bail = $bail:expr,
    ) => {
        // Physical endpoints have first claim on the fixed sample budget.
        // The zero-span case deduplicates these into the one admissible point.
        add_tof_sample(
            &mut $tof_samples,
            &mut $tof_sample_n,
            $tof_budget,
            $tof_lower,
            $tof_lower,
            $tof_upper,
        );
        add_tof_sample(
            &mut $tof_samples,
            &mut $tof_sample_n,
            $tof_budget,
            $tof_upper,
            $tof_lower,
            $tof_upper,
        );

        if $phase_sma > 0.0 && $ctx.tgt_orbit_valid && $ctx.tgt_sma > 0.0 {
            $($period_setup)*
            // Hohmann transfer time and variants (C++ lines 1606-1613)
            let $transfer_sma_hohmann = 0.5 * ($phase_sma + $ctx.tgt_sma);
            if $transfer_sma_hohmann > 0.0 {
                $($transfer_period_setup)*
                $hohmann_tof = 0.5 * $transfer_period;
                add_tof_sample(
                    &mut $tof_samples,
                    &mut $tof_sample_n,
                    $tof_budget,
                    $hohmann_tof,
                    $tof_lower,
                    $tof_upper,
                );
                let hohmann_factors: &[f64] = &[0.8, 0.9, 1.1, 1.2];
                for &factor in hohmann_factors {
                    add_tof_sample(
                        &mut $tof_samples,
                        &mut $tof_sample_n,
                        $tof_budget,
                        $hohmann_tof * factor,
                        $tof_lower,
                        $tof_upper,
                    );
                }

                // Spend at most half of the sealed budget on a dense
                // Hohmann-centred priority grid. The remaining samples are
                // spread across the complete hard interval below, so this
                // improves local resolution without changing admissibility.
                let priority_lower = ($hohmann_tof * 0.4).max($tof_lower);
                let priority_upper = ($hohmann_tof * 2.2).min($tof_upper);
                let priority_span = priority_upper - priority_lower;
                if tof_grid_sample_count(priority_span, $is_simple_transfer)?.is_some() {
                    let priority_count = $tof_budget
                        .checked_div(2)
                        .ok_or(EvaluationArithmeticOverflow)?;
                    let Some(priority_count_f64) = priority_count.to_f64() else {
                        return $bail;
                    };
                    let priority_step = if priority_count > 1 {
                        priority_span / (priority_count_f64 - 1.0)
                    } else {
                        priority_span
                    };
                    for i in 0..priority_count {
                        let Some(i_f64) = i.to_f64() else {
                            return $bail;
                        };
                        add_tof_sample(
                            &mut $tof_samples,
                            &mut $tof_sample_n,
                            $tof_budget,
                            priority_lower + i_f64 * priority_step,
                            $tof_lower,
                            $tof_upper,
                        );
                    }
                }

            }

            // Synodic period samples (C++ lines 1617-1628)
            //
            // NOTE: 10s period-difference threshold (down from 60s) so Walker-like
            // constellations with near-identical periods keep synodic samples.
            if $period.is_finite()
                && $tgt_period.is_finite()
                && $period > 0.0
                && $tgt_period > 0.0
            {
                let period_diff = ($period - $tgt_period).abs();
                if period_diff > 10.0 {
                    let synodic = ($period * $tgt_period / period_diff).abs();
                    if synodic.is_finite() && synodic > 0.0 {
                        let synodic_max = 3;
                        for k in 1..=synodic_max {
                            add_tof_sample(
                                &mut $tof_samples,
                                &mut $tof_sample_n,
                                $tof_budget,
                                f64::from(k) * synodic,
                                $tof_lower,
                                $tof_upper,
                            );
                        }
                    }
                }
            }

            // Multi-revolution variants (C++ lines 1631-1637)
            if $period.is_finite() && $period > 0.0 && $hohmann_tof > 0.0 {
                let historical_multi_rev_max = 2;
                let tof_budget_half = half_tof_budget_as_i32($tof_budget)?;
                let multi_rev_max = $ctx
                    .max_revs
                    .max(historical_multi_rev_max)
                    .min(tof_budget_half);
                for $multi_rev_var in 1..=multi_rev_max {
                    $($multi_rev_setup)*
                    add_tof_sample(
                        &mut $tof_samples,
                        &mut $tof_sample_n,
                        $tof_budget,
                        $($multi_rev_sample)*,
                        $tof_lower,
                        $tof_upper,
                    );
                }
            }

            // Minimum-energy TOF approximation (C++ lines 1639-1653)
            let r1_norm = norm3(&[$dep_at_release[0], $dep_at_release[1], $dep_at_release[2]]);
            let r2_norm = norm3(&[$ctx.tgt_eci[0], $ctx.tgt_eci[1], $ctx.tgt_eci[2]]);
            let dx = $dep_at_release[0] - $ctx.tgt_eci[0];
            let dy = $dep_at_release[1] - $ctx.tgt_eci[1];
            let dz = $dep_at_release[2] - $ctx.tgt_eci[2];
            let chord = (dx * dx + dy * dy + dz * dz).sqrt();
            if r1_norm > 0.0 && r2_norm > 0.0 && chord > 0.0 {
                let s_param = (r1_norm + r2_norm + chord) / 2.0;
                let t_min_energy = std::f64::consts::PI * (s_param.powi(3) / (8.0 * MU)).sqrt();
                if t_min_energy.is_finite() && t_min_energy > 0.0 {
                    add_tof_sample(
                        &mut $tof_samples,
                        &mut $tof_sample_n,
                        $tof_budget,
                        t_min_energy,
                        $tof_lower,
                        $tof_upper,
                    );
                    add_tof_sample(
                        &mut $tof_samples,
                        &mut $tof_sample_n,
                        $tof_budget,
                        t_min_energy * 0.9,
                        $tof_lower,
                        $tof_upper,
                    );
                    add_tof_sample(
                        &mut $tof_samples,
                        &mut $tof_sample_n,
                        $tof_budget,
                        t_min_energy * 1.1,
                        $tof_lower,
                        $tof_upper,
                    );
                }
            }

            // Revolution entry points (C++ lines 1657-1663)
            if $period.is_finite() && $period > 0.0 && $hohmann_tof > 0.0 {
                let historical_rev_entry_max = 2;
                let tof_budget_half = half_tof_budget_as_i32($tof_budget)?;
                let rev_entry_max = $ctx
                    .max_revs
                    .max(historical_rev_entry_max)
                    .min(tof_budget_half);
                for $rev_entry_var in 1..=rev_entry_max {
                    $($rev_entry_setup)*
                    add_tof_sample(
                        &mut $tof_samples,
                        &mut $tof_sample_n,
                        $tof_budget,
                        $($rev_entry_sample)*,
                        $tof_lower,
                        $tof_upper,
                    );
                }
            }

            // Additional samples for large plane changes (retrograde transfers):
            // the optimal TOF sits near the half-period or specific period
            // fractions, plus the ~1 hour band common for LEO.
            if $plane_angle_cached > 1.57 && $period > 0.0 {
                let period_fractions: &[f64] = &[0.6, 0.65, 0.7, 0.75, 0.8];
                for &frac in period_fractions {
                    add_tof_sample(
                        &mut $tof_samples,
                        &mut $tof_sample_n,
                        $tof_budget,
                        $period * frac,
                        $tof_lower,
                        $tof_upper,
                    );
                }
                let hour_samples: &[f64] = &[3400.0, 3500.0, 3600.0, 3700.0, 3800.0];
                for &t in hour_samples {
                    add_tof_sample(
                        &mut $tof_samples,
                        &mut $tof_sample_n,
                        $tof_budget,
                        t,
                        $tof_lower,
                        $tof_upper,
                    );
                }
            }
        }

        // Uniform sampling over span. Guard: when sample_count == 1,
        // (sample_count - 1) == 0 causes division by zero; use span as step
        // (only tof_lower is sampled anyway).
        let Some(remaining_budget) = $tof_budget
            .min(MAX_TOF_SAMPLES)
            .checked_sub($tof_sample_n)
        else {
            return $bail;
        };
        let uniform_count = $sample_count.min(remaining_budget);
        let Some(uniform_count_f64) = uniform_count.to_f64() else {
            return $bail;
        };
        let step = if uniform_count > 1 {
            $span / (uniform_count_f64 - 1.0)
        } else {
            $span
        };
        for i in 0..uniform_count {
            let Some(i_f64) = i.to_f64() else {
                return $bail;
            };
            add_tof_sample(
                &mut $tof_samples,
                &mut $tof_sample_n,
                $tof_budget,
                $tof_lower + i_f64 * step,
                $tof_lower,
                $tof_upper,
            );
        }

        deduplicate_tof_samples(&mut $tof_samples, &mut $tof_sample_n);
    };
}

/// Brent `eval_tof` cache-bookkeeping closure, stamped verbatim into both
/// evaluators: request count, 0.1s-bin cache lookup, hit/miss tallies, raw
/// Lambert solve on miss, exact-solution capture, keep-first insert. The
/// serial and branch copies historically had to be edited in lockstep; both
/// now expand this exact body, with only the context/prepared accessors
/// passed per site.
macro_rules! brent_eval_tof_closure {
    (
        ctx = $ctx:expr,
        dep_at_release = $dep_at_release:expr,
        fixed_offset = $fixed_offset:expr,
        max_transfer_headroom_s = $max_transfer_headroom_s:expr,
        r1_cache = $r1_cache:ident,
        departure_bounds = $departure_bounds:ident,
        cache = $cache:ident,
        exact_solution_cache = $exact_solution_cache:ident,
        request_count = $request_count:ident,
        hit_count = $hit_count:ident,
        miss_count = $miss_count:ident,
    ) => {
        |tof: f64| -> Result<f64, EvaluationArithmeticOverflow> {
            $request_count = $request_count
                .checked_add(1)
                .ok_or(EvaluationArithmeticOverflow)?;
            // Quantize TOF to 0.1s bins for caching.
            let tof_key = brent_tof_cache_key(tof)?;
            if let Some(cached_cost) = brent_cache_lookup(&$cache, tof_key) {
                $hit_count = $hit_count
                    .checked_add(1)
                    .ok_or(EvaluationArithmeticOverflow)?;
                return Ok(cached_cost);
            }
            $miss_count = $miss_count
                .checked_add(1)
                .ok_or(EvaluationArithmeticOverflow)?;
            // Cache miss - compute the Lambert solution.
            let solution = lambert_solve_raw(
                tof,
                $ctx,
                $dep_at_release,
                &$r1_cache,
                &$departure_bounds,
                $fixed_offset,
                $max_transfer_headroom_s,
            )?;
            let cost = solution.map_or(INVALID_COST, |sol| {
                $exact_solution_cache.push((tof.to_bits(), sol));
                sol.cost
            });
            brent_cache_insert_first(&mut $cache, tof_key, cost);
            Ok(cost)
        }
    };
}

#[expect(
    clippy::too_many_lines,
    reason = "single linear solver path preserves scratch lifetime and discrete decision order"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "validated fixed-size state and TOF buffers retain their established exact lane mapping"
)]
fn evaluate_plan_from_phase_with_lambert_scratch_impl(
    x: &[f64; 3],
    ctx: &PlanContext,
    coarse_mode: bool,
    time2phase: f64,
    waittime: f64,
    dep_period: f64,
    dep_at_phase: &[f64; 6],
    dep_phase_orbit_override: Option<EciOrbitSummary>,
    variable_r2_scratch: Option<&mut crate::lambert::VariableR2LambertScratch>,
) -> Result<PlanResult, EvaluationArithmeticOverflow> {
    let mut res = PlanResult::invalid();
    if !target_propagation_authority_is_consistent(ctx) {
        return Ok(res);
    }

    let time2phase_ratio = x[0];
    let phase_sma_ratio = x[1];
    let waittime_ratio = x[2];

    res.time2phase_ratio = time2phase_ratio;
    res.phase_sma_ratio = phase_sma_ratio;
    res.waittime_ratio = waittime_ratio;
    res.dep_period = dep_period;

    // Input validation
    if !time2phase_ratio.is_finite() || !phase_sma_ratio.is_finite() || !waittime_ratio.is_finite()
    {
        return Ok(res);
    }
    if time2phase_ratio + waittime_ratio >= 1.0 {
        return Ok(res);
    }

    res.time2phase = time2phase;
    res.waittime = waittime;

    let max_transfer_headroom_s = match transfer_timing_window(ctx, time2phase, waittime) {
        Ok(headroom) => headroom,
        Err(reason) => {
            res.timing_failure_reason = reason;
            return Ok(res);
        }
    };

    let radius_tol = radius_tolerance(ctx.min_perigee, ctx.max_apogee);
    // Get orbital elements at phase point
    let dep_phase_orbit = dep_phase_orbit_override.or_else(|| eci_orbit_summary(dep_at_phase));
    let Some(dep_phase_orbit) = dep_phase_orbit else {
        res.time2phase = time2phase;
        res.waittime = waittime;
        return Ok(res);
    };

    let phase_base_sma = dep_phase_orbit.sma;
    let phase_sma = phase_base_sma * phase_sma_ratio;
    if !phase_sma.is_finite()
        || phase_sma < ctx.min_perigee - radius_tol
        || phase_sma > ctx.max_apogee + radius_tol
    {
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        return Ok(res);
    }

    let radius = dep_phase_orbit.r_mag;
    let vel_mag = dep_phase_orbit.v_mag;
    if radius <= 0.0 || vel_mag <= 0.0 || radius.is_nan() || vel_mag.is_nan() {
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        return Ok(res);
    }

    // Compute phase burn (tangential maneuver)
    let target_speed_sq = MU * (2.0 / radius - 1.0 / phase_sma);
    if !target_speed_sq.is_finite() || target_speed_sq <= 0.0 {
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        return Ok(res);
    }

    let target_speed = target_speed_sq.sqrt();
    let inv_vel_mag = 1.0 / vel_mag;
    let velocity_hat = [
        dep_at_phase[3] * inv_vel_mag,
        dep_at_phase[4] * inv_vel_mag,
        dep_at_phase[5] * inv_vel_mag,
    ];
    let speed_delta = target_speed - vel_mag;
    let phase_dv = [
        velocity_hat[0] * speed_delta,
        velocity_hat[1] * speed_delta,
        velocity_hat[2] * speed_delta,
    ];

    let phase_dv_norm = speed_delta.abs();

    // Soft penalty for phase dV violation
    let phase_dv_penalty = if phase_dv_norm > ctx.max_phase_dv {
        1000.0 + (phase_dv_norm - ctx.max_phase_dv) * 500.0
    } else {
        0.0
    };
    if phase_dv_penalty > 0.0 {
        res.valid = false;
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        res.phase_dv = phase_dv;
        res.phase_dv_norm = phase_dv_norm;
        res.timing_failure_reason = TimingFailureToken::PhaseDvBoundExceeded;
        return Ok(res);
    }

    // Apply phase burn
    let mut dep_after_phase = *dep_at_phase;
    add_velocity(&mut dep_after_phase, &phase_dv);
    if !all_finite(&dep_after_phase) {
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        return Ok(res);
    }

    // Check angular momentum (orbit validity)
    let r_vec = [dep_after_phase[0], dep_after_phase[1], dep_after_phase[2]];
    let v_vec = [dep_after_phase[3], dep_after_phase[4], dep_after_phase[5]];
    let h_vec = cross3(&r_vec, &v_vec);
    let h_sq = h_vec[0] * h_vec[0] + h_vec[1] * h_vec[1] + h_vec[2] * h_vec[2];
    if !h_sq.is_finite() || h_sq <= 0.0 {
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        return Ok(res);
    }

    let inv_mu_a = h_sq / (MU * phase_sma);
    if !inv_mu_a.is_finite() || inv_mu_a <= 0.0 {
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        return Ok(res);
    }

    let mut ecc_sq = 1.0 - inv_mu_a;
    if ecc_sq < 0.0 {
        if ecc_sq > -1e-10 {
            ecc_sq = 0.0; // Numerical artifact for near-circular
        } else {
            res.time2phase = time2phase;
            res.waittime = waittime;
            res.phase_sma = phase_sma;
            return Ok(res);
        }
    }

    let ecc = ecc_sq.sqrt();
    let dep_perigee = phase_sma * (1.0 - ecc);
    let dep_apogee = phase_sma * (1.0 + ecc);

    if !dep_perigee.is_finite()
        || !dep_apogee.is_finite()
        || dep_perigee < ctx.min_perigee - radius_tol
        || dep_apogee > ctx.max_apogee + radius_tol
    {
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        return Ok(res);
    }

    // Propagate to release point
    let mut dep_after_phase_equ = [0.0; 6];
    if !eci_to_equinoctial(&dep_after_phase, &mut dep_after_phase_equ) {
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        return Ok(res);
    }

    let Some(dep_at_release) = propagate_candidate_search_state_at_epoch(
        &dep_after_phase_equ,
        waittime,
        propagation_epoch_for_segment(ctx.epoch_jd, time2phase),
        ctx.transfer_body_force(),
        ctx,
    )?
    else {
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        return Ok(res);
    };

    let Ok(tof_interval) = admissible_tof_interval(ctx, dep_period, max_transfer_headroom_s) else {
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        res.phase_dv_norm = phase_dv_norm;
        res.timing_failure_reason = TimingFailureToken::TransferRevolutionCapExceeded;
        return Ok(res);
    };
    let tof_lower = tof_interval.lower;
    let tof_upper = tof_interval.upper;

    let fixed_offset = waittime + time2phase;
    let span = tof_interval.span;

    // Adaptive TOF sampling
    let altitude_diff = (phase_sma - dep_phase_orbit.sma).abs();
    let is_simple_transfer = altitude_diff < 500.0 && ecc < 0.05;

    // `SamplingMode` has one variant (Fast), so this is the coarse-for-speed
    // grid unconditionally.
    tof_grid_sample_count(span, is_simple_transfer)?.ok_or(EvaluationArithmeticOverflow)?;
    let sample_count = ctx.search_depth.clamped_tof_budget();

    let mut best_tof = 0.0;
    let mut best_sol: Option<LambertSolutionEx> = None;
    let mut best_cost = INVALID_COST;

    // Compute plane angle once (reused for TOF sampling and search window sizing)
    let plane_angle_cached = if ctx.plane_angle_valid {
        ctx.plane_angle
    } else {
        let h_dep = cross3(
            &[ctx.dep_eci[0], ctx.dep_eci[1], ctx.dep_eci[2]],
            &[ctx.dep_eci[3], ctx.dep_eci[4], ctx.dep_eci[5]],
        );
        let h_tgt = cross3(
            &[ctx.tgt_eci[0], ctx.tgt_eci[1], ctx.tgt_eci[2]],
            &[ctx.tgt_eci[3], ctx.tgt_eci[4], ctx.tgt_eci[5]],
        );
        let h_dep_norm = norm3(&h_dep);
        let h_tgt_norm = norm3(&h_tgt);
        if h_dep_norm > 1e-10 && h_tgt_norm > 1e-10 {
            let cos_angle = (h_dep[0] * h_tgt[0] + h_dep[1] * h_tgt[1] + h_dep[2] * h_tgt[2])
                / (h_dep_norm * h_tgt_norm);
            cos_angle.clamp(-1.0, 1.0).acos()
        } else {
            0.0
        }
    };

    if plane_angle_cached > 2.97 {
        record_evaluation_diagnostic(|counters| {
            checked_diagnostic_counter_add(&mut counters.near_pi_plane_eval_count, 1)?;
            Ok(())
        })?;
    }

    // Generate TOF samples (insertion capped by the runtime search-depth budget)
    let tof_budget = ctx.search_depth.clamped_tof_budget();
    let mut tof_samples = [0.0; MAX_TOF_SAMPLES];
    let mut tof_sample_n = 0;

    // Strategic TOF sampling - aligned with C++ two_phase_transfer_native.hpp (Jan 2026)
    let mut hohmann_tof = 0.0;
    let tgt_period = ctx.tgt_period;

    tof_sample_ladder!(
        ctx = ctx,
        phase_sma = phase_sma,
        dep_at_release = dep_at_release,
        plane_angle_cached = plane_angle_cached,
        tof_lower = tof_lower,
        tof_upper = tof_upper,
        span = span,
        sample_count = sample_count,
        is_simple_transfer = is_simple_transfer,
        tof_budget = tof_budget,
        tof_samples = tof_samples,
        tof_sample_n = tof_sample_n,
        hohmann_tof = hohmann_tof,
        tgt_period = tgt_period,
        period = dep_period,
        period_setup = [
            let phase_sma_cubed = phase_sma * phase_sma * phase_sma;
            let dep_period = 2.0 * std::f64::consts::PI * (phase_sma_cubed / MU).sqrt();
        ],
        transfer_sma_hohmann = transfer_sma_hohmann,
        transfer_period = transfer_period,
        transfer_period_setup = [
            let transfer_sma_hohmann_cubed =
                transfer_sma_hohmann * transfer_sma_hohmann * transfer_sma_hohmann;
            let transfer_period =
                2.0 * std::f64::consts::PI * (transfer_sma_hohmann_cubed / MU).sqrt();
        ],
        multi_rev_var = m,
        multi_rev_setup = [
            let multi_rev_tof = hohmann_tof + f64::from(m) * dep_period;
        ],
        multi_rev_sample = [multi_rev_tof],
        rev_entry_var = n,
        rev_entry_setup = [
            let t_rev_entry = hohmann_tof + (f64::from(n) - 0.5) * dep_period;
        ],
        rev_entry_sample = [t_rev_entry],
        bail = Ok(res),
    );

    // Batch Lambert processing propagates the target for each TOF before solving.
    // FIX (Jan 2026): Now propagates target for EACH TOF using izzo2015_batch_tof_variable_r2
    if tof_sample_n > 0 {
        let branch_target_propagation_start =
            ctx.lambert_branch_selection.is_some().then(Instant::now);
        // Unpack results into separate vectors (SmallVec keeps typical sizes on stack).
        // rust-alloc#3: inline capacity = MAX_TOF_SAMPLES so the reserve()
        // below never heap-spills. dissertation_production.yaml pins
        // tof_sample_budget=256 (> the old 64 inline), which forced ALL of
        // these to heap on every call; SmallVec inline/heap is transparent to
        // results (bit-identical) and the frame already carries
        // MAX_TOF_SAMPLES-sized scalar arrays (t_vals/out_states below).
        let mut target_positions: SmallVec<[[f64; 3]; MAX_TOF_SAMPLES]> = SmallVec::new();
        let mut v2_refs: SmallVec<[[f64; 3]; MAX_TOF_SAMPLES]> = SmallVec::new();
        let mut valid_tofs: SmallVec<[f64; MAX_TOF_SAMPLES]> = SmallVec::new();
        let mut tof_to_idx: SmallVec<[usize; MAX_TOF_SAMPLES]> = SmallVec::new();
        target_positions.reserve(tof_sample_n);
        v2_refs.reserve(tof_sample_n);
        valid_tofs.reserve(tof_sample_n);
        tof_to_idx.reserve(tof_sample_n);

        // rust-alloc#4: these three buffers keep MAX_TOF_SAMPLES of inline
        // capacity (no heap spill, same stack frame as the fixed arrays they
        // replace) but are only *initialized* out to `tof_sample_n`. As fixed
        // arrays they cost a 12 KiB `memset` per plan evaluation while
        // `tof_sample_n` is bounded by the sampling heuristics above at a few
        // dozen — `tof_sample_budget` caps insertion, it does not raise the
        // sample count. Callgrind on `batch_1_verified_superset` charged
        // 9.25 % of the whole MF profile to that one dead memset. Only the
        // `..tof_sample_n` prefix is ever read, so this is dead-store removal:
        // every value the solver observes is written by the same code as
        // before, in the same order.
        let mut t_vals: SmallVec<[f64; MAX_TOF_SAMPLES]> = SmallVec::new();
        t_vals.resize(tof_sample_n, 0.0_f64);
        for i in 0..tof_sample_n {
            t_vals[i] = fixed_offset + tof_samples[i];
        }
        let out_state_len = tof_sample_n
            .checked_mul(6)
            .ok_or(EvaluationArithmeticOverflow)?;
        let mut out_states: SmallVec<[f64; MAX_TOF_SAMPLES * 6]> = SmallVec::new();
        out_states.resize(out_state_len, 0.0_f64);
        debug_assert!(
            tof_sample_n <= MAX_TOF_SAMPLES,
            "tof_sample_n {tof_sample_n} exceeds MAX_TOF_SAMPLES {MAX_TOF_SAMPLES}"
        );
        // Exact, not `>=`. The buffer is now sized to the used prefix, so a
        // `>=` form would be vacuous and would stop catching the thing it was
        // written to catch: a length that disagrees with `tof_sample_n`.
        debug_assert_eq!(
            out_states.len(),
            out_state_len,
            "out_states length disagrees with {tof_sample_n} states"
        );

        let mut tgt_states_full: SmallVec<[[f64; 6]; MAX_TOF_SAMPLES]> = SmallVec::new();
        tgt_states_full.resize(tof_sample_n, [0.0_f64; 6]);

        {
            // Batch-propagate using equinoctial elements (no Lightyear)
            // Use cache if available for significant speedup
            if target_propagation_uses_j2(ctx) {
                let mut t_chunks = t_vals.chunks_exact(4);
                let mut out_state_chunks = out_states.chunks_exact_mut(24);
                for (t_values, out_state_values) in t_chunks.by_ref().zip(out_state_chunks.by_ref())
                {
                    let t_chunk = [t_values[0], t_values[1], t_values[2], t_values[3]];
                    propagate_target_cached_simd4(ctx, &t_chunk, out_state_values)?;
                }
                let remaining_t_vals = t_chunks.remainder();
                let remaining_out_states = out_state_chunks.into_remainder();
                if !remaining_t_vals.is_empty() {
                    propagate_target_cached(ctx, remaining_t_vals, remaining_out_states)?;
                }
            } else {
                propagate_target_analytical(ctx, &t_vals, &mut out_states);
            }

            for (i, out_state) in out_states.chunks_exact(6).enumerate() {
                let tgt_state = [
                    out_state[0],
                    out_state[1],
                    out_state[2],
                    out_state[3],
                    out_state[4],
                    out_state[5],
                ];
                if all_finite(&tgt_state) {
                    tgt_states_full[i] = tgt_state;
                    target_positions.push([tgt_state[0], tgt_state[1], tgt_state[2]]);
                    v2_refs.push([tgt_state[3], tgt_state[4], tgt_state[5]]);
                    valid_tofs.push(tof_samples[i]);
                    tof_to_idx.push(i);
                }
            }
        }

        if let Some(start) = branch_target_propagation_start {
            record_branch_target_propagation(start.elapsed().as_secs_f64())?;
        }

        if !valid_tofs.is_empty() {
            let branch_lambert_sampling_start =
                ctx.lambert_branch_selection.is_some().then(Instant::now);
            let r1_transfer = [dep_at_release[0], dep_at_release[1], dep_at_release[2]];
            // Determine max revolutions using first valid target position as reference
            let r1_norm = norm3(&r1_transfer);
            let r2_norm = norm3(&target_positions[0]);

            if r1_norm > 0.0 && r2_norm > 0.0 {
                // perf-hunt-r2 #4 (2026-07-08): dead `_requires_higher_m`
                // geometry computation removed (never read in the crate).

                // Multi-rev Lambert is controlled by the caller's `max_revs` bound (Lambert M).
                // Do not silently disable multi-rev in fast sampling: if the caller requested it,
                // it must be honored for consistency and monotonic behavior.
                let max_revs = ctx.max_revs.max(0);

                if max_revs == 0 && ctx.lambert_branch_selection.is_none() {
                    let v1_ref = [dep_at_release[3], dep_at_release[4], dep_at_release[5]];
                    record_lambert_batch_call(valid_tofs.len())?;
                    let mut batch_results_owned = None;
                    let batch_results: &[crate::lambert::BatchTofResult] = variable_r2_scratch
                        .map_or_else(
                            || {
                                batch_results_owned
                                    .insert(crate::lambert::izzo2015_batch_tof_variable_r2(
                                        MU,
                                        &r1_transfer,
                                        &target_positions,
                                        &v1_ref,
                                        &v2_refs,
                                        &valid_tofs,
                                        0,
                                    ))
                                    .as_slice()
                            },
                            |scratch| {
                                crate::lambert::izzo2015_batch_tof_variable_r2_with_scratch(
                                    MU,
                                    &r1_transfer,
                                    &target_positions,
                                    &v1_ref,
                                    &v2_refs,
                                    &valid_tofs,
                                    0,
                                    scratch,
                                )
                            },
                        );
                    for (batch_idx, result) in batch_results.iter().enumerate() {
                        if !result.valid {
                            continue;
                        }
                        let dv_vec = [
                            result.v1[0] - v1_ref[0],
                            result.v1[1] - v1_ref[1],
                            result.v1[2] - v1_ref[2],
                        ];
                        let v2_ref = v2_refs[batch_idx];
                        let arrival_dv_vec = [
                            v2_ref[0] - result.v2[0],
                            v2_ref[1] - result.v2[1],
                            v2_ref[2] - result.v2[2],
                        ];
                        let dv_depart = norm3(&dv_vec);
                        if dv_depart.is_finite()
                            && dv_depart < ctx.max_transfer_dv
                            && dv_depart < best_cost
                        {
                            let tgt_state = tgt_states_full[tof_to_idx[batch_idx]];
                            best_cost = dv_depart;
                            best_tof = valid_tofs[batch_idx];
                            best_sol = Some(LambertSolutionEx {
                                cost: dv_depart,
                                dv: dv_vec,
                                arrival_dv: arrival_dv_vec,
                                best_M: result.m,
                                low_path: true,
                                prograde: result.prograde,
                                tgt_state,
                                valid: true,
                            });
                        }
                    }
                } else if ctx.lambert_branch_selection.is_none() {
                    // T1.7: the departure state is fixed across this TOF scan,
                    // so the r1-side Lambert cache is loop-invariant. Hoisting
                    // it is bit-identical (see crate::lambert::LambertR1Cache).
                    //
                    // R21 S1: the samples are independent problems against that
                    // one departure, so they enumerate in a single cross-TOF
                    // pack instead of one enumeration per sample. Sequential
                    // equivalence — including which sample wins — is
                    // `scan_lambert_samples_batched`'s contract. This scan was
                    // ~30% of Stage 1 and ran at the single-problem pack's mean
                    // fill; the census measured a mean of 16.1 samples per scan
                    // with none below 4, so the pack now runs near-full.
                    let scan_r1_cache = crate::lambert::LambertR1Cache::new(&r1_transfer);
                    // The name predates the batch: this counts the scan's TOF
                    // entries, which is what the census reads, and the count is
                    // the same one the sequential loop below records per
                    // sample. `has_branch_selection` is false by this arm's
                    // own condition.
                    record_lambert_scalar_tof_calls(max_revs, false, valid_tofs.len())?;
                    let scan_solutions = scan_lambert_samples_batched(&ScanBatchInputs {
                        ctx,
                        dep_at_release: &dep_at_release,
                        r1_cache: &scan_r1_cache,
                        tgt_states: &tgt_states_full,
                        tof_to_idx: &tof_to_idx,
                        valid_tofs: &valid_tofs,
                    })?;
                    for (solution, &tof) in scan_solutions.iter().zip(valid_tofs.iter()) {
                        let Some(sol) = *solution else {
                            continue;
                        };
                        if sol.cost < best_cost {
                            best_cost = sol.cost;
                            best_tof = tof;
                            best_sol = Some(sol);
                        }
                    }
                } else {
                    // Selected-branch route: one exact `(m, low_path)` branch
                    // per sample through a different pack, still sequential.
                    let scan_r1_cache = crate::lambert::LambertR1Cache::new(&r1_transfer);
                    let scan_departure_bounds =
                        crate::lambert_backend::DepartureBoundCache::new(&dep_at_release);
                    for (batch_idx, &tof) in valid_tofs.iter().enumerate() {
                        let tgt_state = tgt_states_full[tof_to_idx[batch_idx]];
                        record_lambert_scalar_tof_calls(
                            max_revs,
                            ctx.lambert_branch_selection.is_some(),
                            1,
                        )?;
                        if let Some(sol) = select_lambert_branch_solution_with_r1(
                            ctx,
                            &dep_at_release,
                            &scan_r1_cache,
                            &scan_departure_bounds,
                            &tgt_state,
                            tof,
                        )? {
                            if sol.cost < best_cost {
                                best_cost = sol.cost;
                                best_tof = tof;
                                best_sol = Some(sol);
                            }
                        }
                    }
                }
            }
            if let Some(start) = branch_lambert_sampling_start {
                record_branch_lambert_sampling(start.elapsed().as_secs_f64())?;
            }
        }
    }

    // Extract best_sol value immediately after check to avoid distant unwrap()
    let Some(best_sol_value) = best_sol else {
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        return Ok(res);
    };

    // Refine best TOF with bounded minimization
    // Jan 2026: Aligned with C++ two_phase_transfer_native.hpp (line 1754)
    // C++ uses max(600, 0.25*span) - larger window to allow refinement to find global minimum
    //
    // For large plane changes (retrograde transfers), use wider search window
    // to avoid getting stuck in local minima. The TOF landscape has multiple
    // basins for 180-degree plane reversals.
    let is_large_plane_change = plane_angle_cached > 1.57; // > 90 degrees (includes retrograde)

    let bracket_lo = tof_lower;
    let bracket_hi = tof_upper;

    let mut final_tof = best_tof;
    let mut final_sol = best_sol_value;

    if brent_refinement_required(bracket_lo, bracket_hi) {
        let branch_brent_start = ctx.lambert_branch_selection.is_some().then(Instant::now);
        // =====================================================================
        // OPTIMIZATION 2: Coarse Brent for Screening
        // Use coarser tolerance (10x) and fewer iterations (15 vs 70) during
        // candidate screening. This reduces Lambert calls by ~35% of Brent iterations.
        // Fine tolerance only needed for final validation, not exploration.
        //
        // Exception: For large plane changes, always use fine tolerance to avoid
        // missing narrow optimal TOF regions.
        // =====================================================================
        // Performance: Reduced fine-mode iterations from 70 to 50
        let maxiter = if coarse_mode && !is_large_plane_change {
            12 // Coarse: fewer iterations for screening (reduced from 15)
        } else {
            BRENT_FINE_MAX_ITERATIONS // Fine: full iterations for validation (reduced from 70)
        };

        // Coarse tolerance: 10x larger for screening, fine for validation
        // Exception: Large plane changes always use fine tolerance
        let x_tolerance = if coarse_mode && !is_large_plane_change {
            // Coarse: 10% of bracket span, clamped to [10, 300] seconds
            ((bracket_hi - bracket_lo) * 0.10).clamp(10.0, 300.0)
        } else {
            // Fine: 1% of bracket span, clamped to [1, 60] seconds
            ((bracket_hi - bracket_lo) * 0.01).clamp(1.0, 60.0)
        };

        // Local cache for Brent minimization (quantized to 0.1s bins)
        // Reduces redundant Lambert calls when Brent explores nearby TOF values
        let mut brent_cache = BrentLocalCache::new();
        let mut brent_exact_solution_cache = BrentExactSolutionCache::new();
        // perf-hunt-r2 #3: departure state is fixed across this whole Brent
        // search — hoist the r1-side Lambert cache (bit-identical).
        let brent_r1_cache = crate::lambert::LambertR1Cache::new(&[
            dep_at_release[0],
            dep_at_release[1],
            dep_at_release[2],
        ]);
        // Both branch bounds re-derive the same departure half on every TOF
        // they see, so it is hoisted beside the r1-side cache and for the same
        // reason. Keep the two together: hoisting one and forgetting the other
        // is how this cost survived the first pass.
        let brent_departure_bounds =
            crate::lambert_backend::DepartureBoundCache::new(&dep_at_release);
        let mut brent_eval_request_count = 0usize;
        let mut brent_cache_hit_count = 0usize;
        let mut brent_cache_miss_count = 0usize;

        // Seed the cache with the sampling stage's own best so the pre-scan
        // cannot pay for a TOF that was already solved.
        let bracket_span = bracket_hi - bracket_lo;
        if bracket_span > 1.0 {
            let best_tof_key = brent_tof_cache_key(best_tof)?;
            brent_cache_insert_first(&mut brent_cache, best_tof_key, best_cost);
        }

        // Batched pre-scan: evaluate the whole sample ladder through the
        // cross-TOF streaming pack before the sequential closure exists, then
        // replay the costs through `brent_prescan_bracket` unchanged.
        // Sequential-equivalent by `prescan_eval_samples_batched`'s contract.
        let prescan_samples =
            brent_prescan_count(bracket_span, coarse_mode, is_large_plane_change)?;
        let prescan_costs = prescan_eval_samples_batched(
            &PrescanBatchInputs {
                ctx,
                dep_at_release: &dep_at_release,
                r1_cache: &brent_r1_cache,
                fixed_offset,
                max_transfer_headroom_s,
                bracket_lo,
                bracket_hi,
                samples: prescan_samples,
            },
            &mut brent_cache,
            &mut brent_exact_solution_cache,
            &mut brent_eval_request_count,
            &mut brent_cache_hit_count,
            &mut brent_cache_miss_count,
        )?;

        let mut eval_tof = brent_eval_tof_closure!(
            ctx = ctx,
            dep_at_release = &dep_at_release,
            fixed_offset = fixed_offset,
            max_transfer_headroom_s = max_transfer_headroom_s,
            r1_cache = brent_r1_cache,
            departure_bounds = brent_departure_bounds,
            cache = brent_cache,
            exact_solution_cache = brent_exact_solution_cache,
            request_count = brent_eval_request_count,
            hit_count = brent_cache_hit_count,
            miss_count = brent_cache_miss_count,
        );

        // Multimodal-safe bracketing: scan first, then let Brent polish one basin.
        let (scan_lo, scan_hi, scan_tof, scan_cost) = if let Some(costs) = prescan_costs {
            brent_prescan_bracket_from_costs(
                bracket_lo,
                bracket_hi,
                prescan_samples,
                best_tof,
                best_cost,
                costs,
            )?
        } else {
            brent_prescan_bracket(
                bracket_lo,
                bracket_hi,
                prescan_samples,
                best_tof,
                best_cost,
                &mut eval_tof,
            )?
        };
        let x_tolerance = ((scan_hi - scan_lo) * 0.01)
            .clamp(1.0, 60.0)
            .min(x_tolerance);

        let mres = minimize_scalar_bounded(&mut eval_tof, scan_lo, scan_hi, x_tolerance, maxiter)?;

        if scan_cost < best_cost {
            if let Some(sol) = brent_exact_solution_lookup(&brent_exact_solution_cache, scan_tof) {
                final_tof = scan_tof;
                final_sol = sol;
                best_cost = sol.cost;
            }
        }

        if mres.converged && mres.x.is_finite() {
            let refined_opt = if let Some(solution) =
                brent_exact_solution_lookup(&brent_exact_solution_cache, mres.x)
            {
                Some(solution)
            } else {
                lambert_solve_raw(
                    mres.x,
                    ctx,
                    &dep_at_release,
                    &brent_r1_cache,
                    &brent_departure_bounds,
                    fixed_offset,
                    max_transfer_headroom_s,
                )?
            };
            if let Some(refined) = refined_opt {
                if refined.cost < best_cost {
                    final_tof = mres.x;
                    final_sol = refined;
                    best_cost = refined.cost;
                }
            }
        }
        if let Some(start) = branch_brent_start {
            record_branch_brent_cache_counts(
                brent_eval_request_count,
                brent_cache_hit_count,
                brent_cache_miss_count,
            )?;
            record_branch_brent(start.elapsed().as_secs_f64())?;
        }
    }

    if !final_sol.valid || !best_cost.is_finite() {
        res.cost = INVALID_COST - 2000.0 + phase_dv_norm;
        res.valid = false;
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        res.tof = final_tof;
        res.phase_dv_norm = phase_dv_norm;
        return Ok(res);
    }

    if transfer_tof_exceeds_revolution_cap(ctx, final_tof, dep_period) {
        res.cost = INVALID_COST - 1900.0 + phase_dv_norm;
        res.valid = false;
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        res.tof = final_tof;
        res.phase_dv_norm = phase_dv_norm;
        res.timing_failure_reason = TimingFailureToken::TransferRevolutionCapExceeded;
        return Ok(res);
    }

    // Lightweight iterative J2 endpoint closure over Lambert endpoint.
    let j2_settings = ctx.j2_closure_settings;
    let physical_target_state = final_sol.tgt_state;
    let branch_j2_correction_start = ctx.lambert_branch_selection.is_some().then(Instant::now);
    let (refined_sol, iter_count, residual_m) = apply_iterative_j2_lambert_correction(
        ctx,
        &dep_at_release,
        final_tof,
        propagation_epoch_for_segment(ctx.epoch_jd, time2phase + waittime),
        final_sol,
        j2_settings,
    )?;
    if let Some(start) = branch_j2_correction_start {
        record_branch_j2_correction(start.elapsed().as_secs_f64())?;
    }
    final_sol = refined_sol;
    let j2_iteration_count = j2_iteration_count_as_u32(iter_count)?;
    let j2_endpoint_residual_m = residual_m;

    let j2_residual_blocks = pre_hf_j2_residual_blocks_acceptance(
        ctx.execution_policy.use_high_fidelity,
        j2_endpoint_residual_m,
        j2_settings.endpoint_target_km * 1000.0,
    );
    record_j2_residual_gate(j2_residual_blocks, j2_endpoint_residual_m)?;
    if j2_residual_blocks {
        res.cost = INVALID_COST - 1750.0 + phase_dv_norm;
        res.valid = false;
        res.time2phase = time2phase;
        res.waittime = waittime;
        res.phase_sma = phase_sma;
        res.tof = final_tof;
        res.phase_dv_norm = phase_dv_norm;
        res.j2_iteration_count = j2_iteration_count;
        res.j2_endpoint_residual_m = j2_endpoint_residual_m;
        return Ok(res);
    }

    let final_transfer_dv = final_sol.dv;
    let cached_hf_endpoint = None;

    let transfer_dv_norm = norm3(&final_transfer_dv);
    let transfer_dv_penalty = transfer_dv_limit_penalty(transfer_dv_norm, ctx.max_transfer_dv);
    let total_dv = phase_dv_norm + transfer_dv_norm;

    let running_total_cost = total_dv;

    // Build final result
    let mut transfer_start = dep_at_release;
    add_velocity(&mut transfer_start, &final_transfer_dv);

    // Validate transfer orbit
    if let Some(transfer_orbit) = eci_orbit_summary(&transfer_start) {
        let radius_tol = radius_tolerance(ctx.min_perigee, ctx.max_apogee);

        let perigee_violation = (ctx.min_perigee - radius_tol - transfer_orbit.perigee).max(0.0);
        let apogee_violation = (transfer_orbit.apogee - (ctx.max_apogee + radius_tol)).max(0.0);
        let orbit_penalty = (perigee_violation + apogee_violation) * 100.0;

        let re_entry_penalty = if transfer_orbit.perigee < RE {
            (RE - transfer_orbit.perigee) * 10000.0
        } else {
            0.0
        };

        if transfer_orbit.perigee >= RE {
            let mut transfer_equ = [0.0; 6];
            let dep_at_intercept = match cached_hf_endpoint {
                Some(endpoint) => Some(endpoint),
                None if !eci_to_equinoctial(&transfer_start, &mut transfer_equ) => None,
                None => propagate_candidate_search_state_at_epoch(
                    &transfer_equ,
                    final_tof,
                    propagation_epoch_for_segment(ctx.epoch_jd, time2phase + waittime),
                    ctx.transfer_body_force(),
                    ctx,
                )?,
            };
            if let Some(dep_at_intercept) = dep_at_intercept {
                let post_hf_endpoint_residual_m = recompute_post_hf_endpoint_residual_m(
                    &dep_at_intercept,
                    &physical_target_state,
                );
                let distance = post_hf_endpoint_residual_m / 1000.0;

                let distance_penalty = if distance > ctx.distance_tol {
                    (distance - ctx.distance_tol) * 1000.0
                } else {
                    0.0
                };

                // Success! Build final result
                let mut deployer_release_equ = [0.0; 6];
                if eci_to_equinoctial(&dep_at_release, &mut deployer_release_equ) {
                    let Some(deployer_at_intercept) = propagate_candidate_search_state_at_epoch(
                        &deployer_release_equ,
                        final_tof,
                        propagation_epoch_for_segment(ctx.epoch_jd, time2phase + waittime),
                        ctx.transfer_body_force(),
                        ctx,
                    )?
                    else {
                        return Ok(res);
                    };
                    {
                        let deployer_distance =
                            vec_distance(&deployer_at_intercept[..3], &physical_target_state[..3]);

                        let deployer_penalty = if deployer_distance < ctx.deployer_min_distance {
                            (ctx.deployer_min_distance - deployer_distance) * 1000.0
                        } else {
                            0.0
                        };

                        let final_distance = if distance < ctx.distance_tol {
                            0.0
                        } else {
                            distance
                        };
                        res.payload_intercept_state = dep_at_intercept;
                        res.target_intercept_state = physical_target_state;
                        res.deployer_intercept_state = deployer_at_intercept;
                        res.release_state = dep_at_release;
                        res.cost = final_distance
                            + running_total_cost
                            + deployer_penalty
                            + phase_dv_penalty
                            + transfer_dv_penalty
                            + distance_penalty
                            + orbit_penalty
                            + re_entry_penalty;
                        res.post_hf_endpoint_residual_m = post_hf_endpoint_residual_m;
                        res.valid = post_hf_residual_accepts(
                            post_hf_endpoint_residual_m,
                            ctx.distance_tol * 1000.0,
                        ) && re_entry_penalty == 0.0
                            && orbit_penalty == 0.0
                            && phase_dv_penalty == 0.0
                            && transfer_dv_penalty == 0.0
                            && distance_penalty == 0.0
                            && deployer_penalty == 0.0;
                        res.time2phase = time2phase;

                        res.waittime = waittime;
                        res.tof = final_tof;
                        res.distance = final_distance;
                        res.deployer_distance = deployer_distance;
                        res.phase_sma = phase_sma;
                        copy3(&mut res.phase_dv, &phase_dv);
                        copy3(&mut res.transfer_dv, &final_transfer_dv);
                        let arrival_dv =
                            rendezvous_arrival_dv(&dep_at_intercept, &physical_target_state);
                        copy3(&mut res.arrival_dv, &arrival_dv);
                        res.phase_dv_norm = phase_dv_norm;
                        res.transfer_dv_norm = transfer_dv_norm;
                        res.arrival_dv_norm = norm3(&arrival_dv);
                        res.best_M = final_sol.best_M;
                        res.set_accepted_branch(
                            final_sol.best_M,
                            final_sol.low_path,
                            final_tof,
                            transfer_dv_norm,
                            res.arrival_dv_norm,
                        );
                        res.prograde = final_sol.prograde;
                        res.j2_iteration_count = j2_iteration_count;
                        res.j2_endpoint_residual_m = j2_endpoint_residual_m;

                        // Julian dates
                        res.waittime_jd_start = ctx.epoch_jd + time2phase / 86400.0;
                        res.tof_jd_start = res.waittime_jd_start + waittime / 86400.0;
                        res.intercept_jd = res.tof_jd_start + final_tof / 86400.0;

                        return Ok(res);
                    }
                }
            }
        }
    }

    res.time2phase = time2phase;
    res.waittime = waittime;
    res.phase_sma = phase_sma;
    res.tof = final_tof;
    res.phase_dv_norm = phase_dv_norm;
    res.transfer_dv_norm = transfer_dv_norm;
    res.cost = INVALID_COST - 500.0 + running_total_cost;
    res.j2_iteration_count = j2_iteration_count;
    res.j2_endpoint_residual_m = j2_endpoint_residual_m;
    res.valid = false;
    Ok(res)
}

/// Convert ECI to equinoctial elements
pub fn eci_to_equinoctial(eci: &[f64; 6], equ: &mut [f64; 6]) -> bool {
    eci2equinoc_impl(eci, 6, 0.0, 0.0, equ);
    all_finite(equ)
}

/// Basic orbital elements from ECI state
#[derive(Clone, Copy, Default)]
pub struct EciOrbitSummary {
    pub sma: f64,
    pub ecc: f64,
    pub inc: f64,
    pub raan: f64,
    pub r_mag: f64,
    pub v_mag: f64,
    pub perigee: f64,
    pub apogee: f64,
}

/// Compute basic orbital elements from ECI state
pub(crate) fn eci_orbit_summary(state: &[f64; 6]) -> Option<EciOrbitSummary> {
    let mut kep = [0.0; 6];
    eci2kep_impl(state, false, true, &mut kep); // radians, true anomaly

    let sma = kep[0];
    let ecc = kep[1];
    let inc = kep[2];
    let raan = kep[3];

    if !sma.is_finite() || !ecc.is_finite() || sma <= 0.0 || !(0.0..1.0).contains(&ecc) {
        return None;
    }

    let r_mag = norm3(&[state[0], state[1], state[2]]);
    let v_mag = norm3(&[state[3], state[4], state[5]]);

    Some(EciOrbitSummary {
        sma,
        ecc,
        inc,
        raan,
        r_mag,
        v_mag,
        perigee: sma * (1.0 - ecc),
        apogee: sma * (1.0 + ecc),
    })
}

// ============================================================================
// TOF Sampling
// ============================================================================

/// Add a TOF sample if within bounds and not too close to existing samples
fn add_tof_sample(
    samples: &mut [f64; MAX_TOF_SAMPLES],
    count: &mut usize,
    budget: usize,
    tof: f64,
    lower: f64,
    upper: f64,
) {
    if *count >= budget.min(MAX_TOF_SAMPLES) {
        return;
    }
    if !tof.is_finite() || tof < lower || tof > upper {
        return;
    }
    // Check for duplicates (within separation threshold).
    if samples
        .iter()
        .take(*count)
        .any(|sample| (*sample - tof).abs() < TOF_SAMPLE_SEPARATION)
    {
        return;
    }
    let Some(slot) = samples.get_mut(*count) else {
        return;
    };
    *slot = tof;
    let Some(next_count) = (*count).checked_add(1) else {
        return;
    };
    *count = next_count;
}

/// Deduplicate and sort TOF samples
fn deduplicate_tof_samples(samples: &mut [f64; MAX_TOF_SAMPLES], count: &mut usize) {
    let bounded_count = (*count).min(samples.len());
    *count = bounded_count;
    if bounded_count <= 1 {
        return;
    }

    // Sort using pdqsort for hot path performance
    let Some(slice) = samples.get_mut(..bounded_count) else {
        return;
    };
    pdqsort::sort_by_key(slice, |x| OrderedFloat(*x));

    // Deduplicate
    let mut write: usize = 1;
    let mut read: usize = 1;
    while read < slice.len() {
        let Some(value) = slice.get(read).copied() else {
            break;
        };
        let Some(previous_index) = write.checked_sub(1) else {
            break;
        };
        let Some(previous) = slice.get(previous_index).copied() else {
            break;
        };
        if (value - previous).abs() >= TOF_SAMPLE_SEPARATION {
            let Some(destination) = slice.get_mut(write) else {
                break;
            };
            *destination = value;
            let Some(next_write) = write.checked_add(1) else {
                break;
            };
            write = next_write;
        }
        let Some(next_read) = read.checked_add(1) else {
            break;
        };
        read = next_read;
    }
    *count = write;
}

// ============================================================================
// Lambert Evaluation
// ============================================================================

#[inline]
const fn lambert_branch_allowed(ctx: &PlanContext, rev: i32, low_path: bool) -> bool {
    match ctx.lambert_branch_selection {
        Some(selection) => selection.rev == rev && selection.low_path == low_path,
        None => true,
    }
}

// perf-hunt-r2 #3: the uncached `select_lambert_branch_solution` wrapper was
// removed — every caller now hoists a `LambertR1Cache` and uses the
// `_with_r1` entry below (bit-identical per `crate::lambert` documentation).

/// `select_lambert_branch_solution` fast entry taking a precomputed
/// departure-side cache (`LambertR1Cache::new` of `dep_at_release`'s
/// position). Callers scanning many TOFs against one fixed departure state
/// hoist the cache once; results are bit-identical to the uncached entry by
/// `crate::lambert` documentation (`LambertR1Cache`).
fn select_lambert_branch_solution_with_r1(
    ctx: &PlanContext,
    dep_at_release: &[f64; 6],
    r1_cache: &crate::lambert::LambertR1Cache,
    departure: &crate::lambert_backend::DepartureBoundCache,
    tgt_state: &[f64; 6],
    tof: f64,
) -> Result<Option<LambertSolutionEx>, EvaluationArithmeticOverflow> {
    let Some(branch_max_revs) = select_lambert_branch_ceiling(ctx) else {
        return Ok(None);
    };
    // Acceptance cap shared by the multi-rev and retrograde prunes: every
    // solution the visitor keeps satisfies `dv < max_transfer_dv`, so nothing
    // whose departure dv is bounded below by `dv_cap` can survive.
    let dv_cap = ctx.max_transfer_dv;
    // Multi-rev energy prune. A branch at `m` revolutions has to fit `m`
    // orbital periods inside the TOF, which caps its speed at r1 and so floors
    // its departure dv; above `max_revolutions_below_dv_cap` that floor already
    // meets the acceptance cap. Skipping those branches removes only solutions
    // the filters below reject, so the surviving argmin — and every float
    // derived from it — is unchanged.
    let branch_max_revs = crate::lambert_backend::max_revolutions_below_dv_cap_cached(
        departure,
        tgt_state,
        tof,
        dv_cap,
        branch_max_revs,
    );
    if ctx
        .lambert_branch_selection
        .is_some_and(|selection| selection.rev > branch_max_revs)
    {
        return Ok(None);
    }
    select_lambert_branch_solution_bounded(
        ctx,
        dep_at_release,
        r1_cache,
        departure,
        tgt_state,
        tof,
        branch_max_revs,
    )
}

/// The revolution ceiling this context asks for, or `None` when the requested
/// branch is outside it.
#[inline]
fn select_lambert_branch_ceiling(ctx: &PlanContext) -> Option<i32> {
    match ctx.lambert_branch_selection {
        Some(selection) if selection.rev < 0 || selection.rev > ctx.max_revs.max(0) => None,
        Some(selection) => Some(selection.rev),
        None => Some(ctx.max_revs.max(0)),
    }
}

/// Unpruned survivor selection: the oracle the multi-rev prune tests compare
/// against.
///
/// Takes the same path as `select_lambert_branch_solution_with_r1` with the
/// revolution cap disabled, so a test can assert the two agree wherever the
/// acceptance cap is above the energy bound. It has no non-test caller.
#[cfg(test)]
fn select_lambert_branch_solution_uncapped(
    ctx: &PlanContext,
    dep_at_release: &[f64; 6],
    r1_cache: &crate::lambert::LambertR1Cache,
    tgt_state: &[f64; 6],
    tof: f64,
) -> Result<Option<LambertSolutionEx>, EvaluationArithmeticOverflow> {
    let Some(branch_max_revs) = select_lambert_branch_ceiling(ctx) else {
        return Ok(None);
    };
    select_lambert_branch_solution_bounded(
        ctx,
        dep_at_release,
        r1_cache,
        &crate::lambert_backend::DepartureBoundCache::new(dep_at_release),
        tgt_state,
        tof,
        branch_max_revs,
    )
}

/// Fold one enumerated Lambert branch candidate into the running best.
///
/// This is the acceptance filter and argmin of
/// `select_lambert_branch_solution_bounded`'s visitor, extracted so the
/// batched pre-scan path folds candidates through EXACTLY the same code: the
/// same branch-allowed check, the same finite/cap filters, and the
/// same strict `<` (first-in-enumeration-order wins ties).
#[inline]
fn fold_lambert_branch_candidate(
    ctx: &PlanContext,
    tgt_state: &[f64; 6],
    best: &mut Option<LambertSolutionEx>,
    m: i32,
    low_path: bool,
    prograde: bool,
    dv_vec: [f64; 3],
    arrival_dv_vec: [f64; 3],
) {
    if !lambert_branch_allowed(ctx, m, low_path) {
        return;
    }
    let dv_norm = norm3(&dv_vec);
    if !(dv_norm.is_finite() && dv_norm < ctx.max_transfer_dv) {
        return;
    }
    if best.as_ref().is_none_or(|current| dv_norm < current.cost) {
        *best = Some(LambertSolutionEx {
            cost: dv_norm,
            dv: dv_vec,
            arrival_dv: arrival_dv_vec,
            best_M: m,
            low_path,
            prograde,
            tgt_state: *tgt_state,
            valid: true,
        });
    }
}

fn select_lambert_branch_solution_bounded(
    ctx: &PlanContext,
    dep_at_release: &[f64; 6],
    r1_cache: &crate::lambert::LambertR1Cache,
    departure: &crate::lambert_backend::DepartureBoundCache,
    tgt_state: &[f64; 6],
    tof: f64,
    branch_max_revs: i32,
) -> Result<Option<LambertSolutionEx>, EvaluationArithmeticOverflow> {
    let dv_cap = ctx.max_transfer_dv;
    let mut best: Option<LambertSolutionEx> = None;
    let mut visit_solution =
        |m: i32, low_path: bool, prograde: bool, dv_vec: [f64; 3], arrival_dv_vec: [f64; 3]| {
            fold_lambert_branch_candidate(
                ctx,
                tgt_state,
                &mut best,
                m,
                low_path,
                prograde,
                dv_vec,
                arrival_dv_vec,
            );
        };
    // Retrograde solutions depart against the deployer's transfer-plane
    // tangential velocity, so their dv is bounded below by that tangential
    // speed. When the bound already exceeds every acceptance threshold the
    // retrograde solves are skipped outright: they could only produce
    // solutions the filters above reject, and prograde floats are untouched.
    let include_retrograde =
        crate::lambert_backend::retrograde_departure_dv_lower_bound_cached(departure, tgt_state)
            < dv_cap;
    let problem = LambertProblem::new(r1_cache, dep_at_release, tgt_state, tof);
    if let Some(selection) = ctx.lambert_branch_selection {
        visit_lambert_exact_branch_solutions_pruned_with_r1(
            problem,
            selection.rev,
            selection.low_path,
            include_retrograde,
            &mut visit_solution,
        )?;
    } else {
        visit_lambert_branch_solutions_pruned_with_r1(
            problem,
            branch_max_revs,
            true,
            include_retrograde,
            &mut visit_solution,
        )?;
    }
    Ok(best)
}

fn branch_batch_result_to_solution(
    ctx: &PlanContext,
    dep_at_release: &[f64; 6],
    tgt_state: &[f64; 6],
    result: &crate::lambert::BranchBatchTofResult,
) -> Option<LambertSolutionEx> {
    if !result.valid || !lambert_branch_allowed(ctx, result.m, result.low_path) {
        return None;
    }
    let dv_norm = result.dv_depart;
    if !(dv_norm.is_finite() && dv_norm < ctx.max_transfer_dv) {
        return None;
    }
    let dv = [
        result.v1[0] - dep_at_release[3],
        result.v1[1] - dep_at_release[4],
        result.v1[2] - dep_at_release[5],
    ];
    let arrival_dv = [
        tgt_state[3] - result.v2[0],
        tgt_state[4] - result.v2[1],
        tgt_state[5] - result.v2[2],
    ];
    if !all_finite(&dv) || !all_finite(&arrival_dv) || !all_finite(tgt_state) {
        return None;
    }
    Some(LambertSolutionEx {
        cost: dv_norm,
        dv,
        arrival_dv,
        best_M: result.m,
        low_path: result.low_path,
        prograde: result.prograde,
        tgt_state: *tgt_state,
        valid: true,
    })
}

/// Solve Lambert problem for a given TOF
// perf-hunt-r2 #3 (2026-07-08): takes the departure-side Lambert cache so
// Brent bracket searches (<=50 calls per candidate against one fixed
// departure state) stop rebuilding it per call. Bit-identical per the
// `crate::lambert` `_with_r1` documentation (same rationale as T1.7).
fn lambert_solve_raw(
    tof: f64,
    ctx: &PlanContext,
    dep_at_release: &[f64; 6],
    r1_cache: &crate::lambert::LambertR1Cache,
    departure: &crate::lambert_backend::DepartureBoundCache,
    fixed_offset: f64,
    max_transfer_headroom_s: f64,
) -> Result<Option<LambertSolutionEx>, EvaluationArithmeticOverflow> {
    if !tof.is_finite() || tof < MIN_TOF || tof > max_transfer_headroom_s {
        return Ok(None);
    }

    // Propagate target to intercept time
    let Some(tgt_state) =
        propagate_candidate_target_at_authoritative_offset(ctx, fixed_offset + tof)?
    else {
        return Ok(None);
    };

    let r1 = [dep_at_release[0], dep_at_release[1], dep_at_release[2]];
    let r2 = [tgt_state[0], tgt_state[1], tgt_state[2]];

    let r1_norm = norm3(&r1);
    let r2_norm = norm3(&r2);
    if r1_norm <= 0.0 || r2_norm <= 0.0 {
        return Ok(None);
    }

    // perf-hunt-r2 #4 (2026-07-08): removed the dead `_requires_higher_m`
    // computation (sma_ratio / unit vectors / dot / clamp) — the binding was
    // never read anywhere in the crate, and this ran on every Brent
    // iteration.
    select_lambert_branch_solution_with_r1(
        ctx,
        dep_at_release,
        r1_cache,
        departure,
        &tgt_state,
        tof,
    )
}

// perf-hunt-r2 #3: cache-threaded like lambert_solve_raw (J2 retry loop
// calls this <=8 times against one fixed departure state).
fn lambert_solve_with_target_state(
    tof: f64,
    ctx: &PlanContext,
    dep_at_release: &[f64; 6],
    r1_cache: &crate::lambert::LambertR1Cache,
    departure: &crate::lambert_backend::DepartureBoundCache,
    tgt_state: &[f64; 6],
) -> Result<Option<LambertSolutionEx>, EvaluationArithmeticOverflow> {
    if !tof.is_finite() || tof < MIN_TOF || tof > ctx.max_time_s {
        return Ok(None);
    }
    if !all_finite(tgt_state) {
        return Ok(None);
    }

    let r1 = [dep_at_release[0], dep_at_release[1], dep_at_release[2]];
    let r2 = [tgt_state[0], tgt_state[1], tgt_state[2]];
    let r1_norm = norm3(&r1);
    let r2_norm = norm3(&r2);
    if r1_norm <= 0.0 || r2_norm <= 0.0 {
        return Ok(None);
    }

    select_lambert_branch_solution_with_r1(ctx, dep_at_release, r1_cache, departure, tgt_state, tof)
}

fn compute_transfer_endpoint_residual_km(
    ctx: &PlanContext,
    dep_at_release: &[f64; 6],
    sol: &LambertSolutionEx,
    nominal_target_state: &[f64; 6],
    tof: f64,
    transfer_source_jd: f64,
) -> Result<Option<(f64, [f64; 3])>, EvaluationArithmeticOverflow> {
    let mut transfer_start = *dep_at_release;
    add_velocity(&mut transfer_start, &sol.dv);
    let mut transfer_equ = [0.0; 6];
    if !eci_to_equinoctial(&transfer_start, &mut transfer_equ) {
        return Ok(None);
    }
    let Some(dep_at_intercept) = propagate_candidate_search_state_at_epoch(
        &transfer_equ,
        tof,
        transfer_source_jd,
        ctx.transfer_body_force(),
        ctx,
    )?
    else {
        return Ok(None);
    };
    let miss = [
        dep_at_intercept[0] - nominal_target_state[0],
        dep_at_intercept[1] - nominal_target_state[1],
        dep_at_intercept[2] - nominal_target_state[2],
    ];
    Ok(Some((norm3(&miss), miss)))
}

/// Recompute final endpoint residual from the post-HF propagated states.
///
/// ``PlanResult.j2_endpoint_residual_m`` remains the pre-HF J2 closure
/// residual for compatibility. Acceptance telemetry must use this value after
/// the final transfer burn and HF propagation have been applied.
#[inline]
pub(crate) fn recompute_post_hf_endpoint_residual_m(
    payload_at_intercept: &[f64; 6],
    target_at_intercept: &[f64; 6],
) -> f64 {
    let miss = [
        payload_at_intercept[0] - target_at_intercept[0],
        payload_at_intercept[1] - target_at_intercept[1],
        payload_at_intercept[2] - target_at_intercept[2],
    ];
    norm3(&miss) * 1000.0
}

fn apply_iterative_j2_lambert_correction(
    ctx: &PlanContext,
    dep_at_release: &[f64; 6],
    tof: f64,
    transfer_source_jd: f64,
    mut sol: LambertSolutionEx,
    j2_settings: crate::solve::J2ClosureSettings,
) -> Result<(LambertSolutionEx, usize, f64), EvaluationArithmeticOverflow> {
    let nominal_target_state = sol.tgt_state;
    let mut synthetic_target_state = sol.tgt_state;
    let mut final_miss_distance_km = f64::NAN;
    let mut correction_steps = 0usize;
    let mut lambert_retry_count = 0usize;
    // perf-hunt-r2 #3: departure state fixed across the whole retry loop.
    let j2_r1_cache = crate::lambert::LambertR1Cache::new(&[
        dep_at_release[0],
        dep_at_release[1],
        dep_at_release[2],
    ]);
    // Same fixed departure state, same reasoning, same loop.
    let j2_departure_bounds = crate::lambert_backend::DepartureBoundCache::new(dep_at_release);

    // Evaluate residual at the initial solve and after every correction step.
    // With max_iter=8 this yields up to 8 correction steps plus a final residual check.
    for step in 0..=j2_settings.max_iterations {
        let Some((miss_km, miss_vec)) = compute_transfer_endpoint_residual_km(
            ctx,
            dep_at_release,
            &sol,
            &nominal_target_state,
            tof,
            transfer_source_jd,
        )?
        else {
            break;
        };
        final_miss_distance_km = miss_km;
        if miss_km <= j2_settings.endpoint_target_km {
            break;
        }
        if step == j2_settings.max_iterations {
            break;
        }

        synthetic_target_state[0] -= miss_vec[0] * j2_settings.correction_step_gain;
        synthetic_target_state[1] -= miss_vec[1] * j2_settings.correction_step_gain;
        synthetic_target_state[2] -= miss_vec[2] * j2_settings.correction_step_gain;

        lambert_retry_count = lambert_retry_count
            .checked_add(1)
            .ok_or(EvaluationArithmeticOverflow)?;
        if let Some(updated) = lambert_solve_with_target_state(
            tof,
            ctx,
            dep_at_release,
            &j2_r1_cache,
            &j2_departure_bounds,
            &synthetic_target_state,
        )? {
            sol = updated;
            correction_steps = correction_steps
                .checked_add(1)
                .ok_or(EvaluationArithmeticOverflow)?;
        } else {
            break;
        }
    }

    // Preserve physical target semantics for downstream consumers.
    sol.tgt_state = nominal_target_state;

    let closure_error_meters = final_miss_distance_km * 1000.0;
    record_j2_correction(
        correction_steps,
        lambert_retry_count,
        final_miss_distance_km * 1000.0,
    )?;

    Ok((sol, correction_steps, closure_error_meters))
}

// ============================================================================
// Main Evaluation Function
// ============================================================================

#[inline]
fn transfer_timing_window(
    ctx: &PlanContext,
    time2phase: f64,
    waittime: f64,
) -> Result<f64, TimingFailureToken> {
    let pre_sum = time2phase + waittime;
    let intercept_time_budget = ctx.intercept_time_budget_s();
    if !pre_sum.is_finite() || pre_sum > intercept_time_budget {
        return Err(TimingFailureToken::InterceptTransferTimeExceeded);
    }
    let max_transfer_headroom_s = intercept_time_budget - pre_sum;
    if !max_transfer_headroom_s.is_finite() || max_transfer_headroom_s < MIN_TOF {
        return Err(TimingFailureToken::InterceptInsufficientLead);
    }
    Ok(max_transfer_headroom_s)
}

#[inline]
fn transfer_revolution_cap_s(ctx: &PlanContext, dep_period: f64) -> Option<f64> {
    if ctx.revolution_cap.is_finite()
        && ctx.revolution_cap > 0.0
        && dep_period.is_finite()
        && dep_period > 0.0
    {
        Some(ctx.revolution_cap * dep_period)
    } else {
        None
    }
}

/// The one physical TOF search interval used by both production evaluators.
#[derive(Clone, Copy, Debug)]
struct AdmissibleTofInterval {
    lower: f64,
    upper: f64,
    span: f64,
}

#[inline]
fn admissible_tof_interval(
    ctx: &PlanContext,
    dep_period: f64,
    max_transfer_headroom_s: f64,
) -> Result<AdmissibleTofInterval, TimingFailureToken> {
    debug_assert!(max_transfer_headroom_s.is_finite());
    debug_assert!(max_transfer_headroom_s >= MIN_TOF);
    let lower = MIN_TOF;
    let upper = transfer_revolution_cap_s(ctx, dep_period)
        .map_or(max_transfer_headroom_s, |cap_s| {
            max_transfer_headroom_s.min(cap_s)
        });
    if upper < lower {
        return Err(TimingFailureToken::TransferRevolutionCapExceeded);
    }
    Ok(AdmissibleTofInterval {
        lower,
        upper,
        span: upper - lower,
    })
}

#[inline]
fn transfer_tof_exceeds_revolution_cap(ctx: &PlanContext, tof: f64, dep_period: f64) -> bool {
    transfer_revolution_cap_s(ctx, dep_period)
        .is_some_and(|limit_s| tof.is_finite() && tof > limit_s + 1e-3)
}

/// Evaluate a transfer plan with given parameters.
///
/// # Arguments
/// * `x` - Parameter vector `[time2phase_ratio, phase_sma_ratio, waittime_ratio]`
/// * `ctx` - Planning context with orbital states and constraints
/// * `coarse_mode` - If true, use coarse Brent refinement (faster, less precise)
///
/// # Returns
/// * `PlanResult` with computed trajectory and cost, or invalid result
///
/// Note: This function only reads from ctx (no mutations).
/// The `&PlanContext` signature allows parallel fitness evaluation without mutex.
///
/// # Errors
///
/// Returns a typed authority error before candidate-search work when the
/// candidate-search mode, target body force, or force configuration is invalid.
pub fn evaluate_plan(
    x: &[f64; 3],
    ctx: &PlanContext,
    coarse_mode: bool,
) -> Result<PlanResult, crate::types::InvalidTargetPropagationAuthorityCode> {
    validate_public_evaluate_plan_authority(ctx)?;
    evaluate_plan_internal(x, ctx, coarse_mode)
        .map_err(|_| crate::types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
}

fn evaluate_plan_internal(
    x: &[f64; 3],
    ctx: &PlanContext,
    coarse_mode: bool,
) -> Result<PlanResult, EvaluationArithmeticOverflow> {
    let mut res = PlanResult::invalid();
    if !candidate_search_is_supported(ctx) {
        return Ok(unsupported_candidate_search_result());
    }

    let time2phase_ratio = x[0];
    let phase_sma_ratio = x[1];
    let waittime_ratio = x[2];

    let _radius_tol = radius_tolerance(ctx.min_perigee, ctx.max_apogee);

    res.time2phase_ratio = time2phase_ratio;
    res.phase_sma_ratio = phase_sma_ratio;
    res.waittime_ratio = waittime_ratio;
    // Ensure dep_period is initialized if missing
    let dep_period = compute_dep_period(ctx);
    res.dep_period = dep_period;

    // Input validation
    if !time2phase_ratio.is_finite() || !phase_sma_ratio.is_finite() || !waittime_ratio.is_finite()
    {
        return Ok(res);
    }
    if time2phase_ratio + waittime_ratio >= 1.0 {
        return Ok(res);
    }

    let time2phase = time2phase_ratio * ctx.max_time_s;
    let waittime = waittime_ratio * ctx.max_time_s;
    if let Err(reason) = transfer_timing_window(ctx, time2phase, waittime) {
        res.timing_failure_reason = reason;
        res.time2phase = time2phase;
        res.waittime = waittime;
        return Ok(res);
    }

    // Propagate deployer to phase point
    let Some(dep_at_phase) = propagate_candidate_search_state_at_epoch(
        &ctx.dep_equ,
        time2phase,
        ctx.epoch_jd,
        ctx.transfer_body_force(),
        ctx,
    )?
    else {
        res.time2phase = time2phase;
        res.waittime = waittime;
        return Ok(res);
    };

    evaluate_plan_from_phase_with_lambert_scratch(
        x,
        ctx,
        coarse_mode,
        time2phase,
        waittime,
        dep_period,
        &dep_at_phase,
        None,
        None,
    )
}

struct PreparedTransferWork {
    x: [f64; 3],
    ctx: PlanContext,
    coarse_mode: bool,
    time2phase: f64,
    waittime: f64,
    dep_period: f64,
    // Test-only replay inputs stay out of production prepared work. They are
    // carried rather than recomputed so the scalar parity replay cannot drift
    // from the real prepared path.
    #[cfg(test)]
    dep_at_phase: [f64; 6],
    #[cfg(test)]
    dep_phase_orbit: EciOrbitSummary,
    phase_sma: f64,
    phase_dv: [f64; 3],
    phase_dv_norm: f64,
    dep_at_release: [f64; 6],
    max_transfer_headroom_s: f64,
    tof_lower: f64,
    tof_upper: f64,
    fixed_offset: f64,
    plane_angle_cached: f64,
    tgt_states_full: [[f64; 6]; MAX_TOF_SAMPLES],
    // rust-alloc#3: sized to MAX_TOF_SAMPLES (matching tgt_states_full) so a
    // tof_sample_budget above 64 (dissertation_production pins 256) does not
    // heap-spill all four on every branch prepare. One local instance per
    // evaluate call, passed by reference — never stored in collections.
    r2_vec: SmallVec<[[f64; 3]; MAX_TOF_SAMPLES]>,
    v2_refs: SmallVec<[[f64; 3]; MAX_TOF_SAMPLES]>,
    valid_tofs: SmallVec<[f64; MAX_TOF_SAMPLES]>,
    tof_to_idx: SmallVec<[usize; MAX_TOF_SAMPLES]>,
}

impl PreparedTransferWork {
    fn invalid_result(&self) -> PlanResult {
        let mut res = PlanResult::invalid();
        res.time2phase_ratio = self.x[0];
        res.phase_sma_ratio = self.x[1];
        res.waittime_ratio = self.x[2];
        res.dep_period = self.dep_period;
        res.time2phase = self.time2phase;
        res.waittime = self.waittime;
        res.phase_sma = self.phase_sma;
        res.phase_dv = self.phase_dv;
        res.phase_dv_norm = self.phase_dv_norm;
        res.release_state = self.dep_at_release;
        res
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "one ordered setup path keeps shared branch buffers and rejection order auditable"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "validated fixed-size state and TOF buffers retain their established exact lane mapping"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "validated fixed-size TOF lane offsets retain their established mapping and operation order"
)]
#[expect(
    clippy::large_stack_frames,
    reason = "fixed branch buffers prevent per-evaluation heap allocation and keep lane order stable"
)]
fn prepare_branch_shared_work(
    x: &[f64; 3],
    ctx: &PlanContext,
    coarse_mode: bool,
) -> Result<Option<PreparedTransferWork>, EvaluationArithmeticOverflow> {
    if !candidate_search_is_supported(ctx) {
        return Ok(None);
    }
    if !target_propagation_authority_is_consistent(ctx) {
        return Ok(None);
    }
    let total_start = Instant::now();
    let time2phase_ratio = x[0];
    let phase_sma_ratio = x[1];
    let waittime_ratio = x[2];
    if !time2phase_ratio.is_finite() || !phase_sma_ratio.is_finite() || !waittime_ratio.is_finite()
    {
        return Ok(None);
    }
    if time2phase_ratio + waittime_ratio >= 1.0 {
        return Ok(None);
    }

    let mut branch_base_ctx = ctx.clone();
    branch_base_ctx.lambert_branch_selection = None;

    let time2phase = time2phase_ratio * branch_base_ctx.max_time_s;
    let waittime = waittime_ratio * branch_base_ctx.max_time_s;
    let Ok(max_transfer_headroom_s) =
        transfer_timing_window(&branch_base_ctx, time2phase, waittime)
    else {
        return Ok(None);
    };

    let dep_period = compute_dep_period(&branch_base_ctx);
    let phase_release_start = Instant::now();
    let Some(dep_at_phase) = propagate_candidate_search_state_at_epoch(
        &branch_base_ctx.dep_equ,
        time2phase,
        branch_base_ctx.epoch_jd,
        branch_base_ctx.transfer_body_force(),
        &branch_base_ctx,
    )?
    else {
        return Ok(None);
    };
    let Some(dep_phase_orbit) = eci_orbit_summary(&dep_at_phase) else {
        return Ok(None);
    };
    let radius_tol = radius_tolerance(branch_base_ctx.min_perigee, branch_base_ctx.max_apogee);
    let phase_base_sma = dep_phase_orbit.sma;
    let phase_sma = phase_base_sma * phase_sma_ratio;
    if !phase_sma.is_finite()
        || phase_sma < branch_base_ctx.min_perigee - radius_tol
        || phase_sma > branch_base_ctx.max_apogee + radius_tol
    {
        return Ok(None);
    }

    let radius = dep_phase_orbit.r_mag;
    let vel_mag = dep_phase_orbit.v_mag;
    if radius <= 0.0 || vel_mag <= 0.0 || radius.is_nan() || vel_mag.is_nan() {
        return Ok(None);
    }

    let target_speed_sq = MU * (2.0 / radius - 1.0 / phase_sma);
    if !target_speed_sq.is_finite() || target_speed_sq <= 0.0 {
        return Ok(None);
    }

    let target_speed = target_speed_sq.sqrt();
    let inv_vel_mag = 1.0 / vel_mag;
    let velocity_hat = [
        dep_at_phase[3] * inv_vel_mag,
        dep_at_phase[4] * inv_vel_mag,
        dep_at_phase[5] * inv_vel_mag,
    ];
    let speed_delta = target_speed - vel_mag;
    let phase_dv = [
        velocity_hat[0] * speed_delta,
        velocity_hat[1] * speed_delta,
        velocity_hat[2] * speed_delta,
    ];
    let phase_dv_norm = speed_delta.abs();
    if phase_dv_norm > branch_base_ctx.max_phase_dv {
        return Ok(None);
    }

    let mut dep_after_phase = dep_at_phase;
    add_velocity(&mut dep_after_phase, &phase_dv);
    if !all_finite(&dep_after_phase) {
        return Ok(None);
    }

    let r_vec = [dep_after_phase[0], dep_after_phase[1], dep_after_phase[2]];
    let v_vec = [dep_after_phase[3], dep_after_phase[4], dep_after_phase[5]];
    let h_vec = cross3(&r_vec, &v_vec);
    let h_sq = h_vec[0] * h_vec[0] + h_vec[1] * h_vec[1] + h_vec[2] * h_vec[2];
    if !h_sq.is_finite() || h_sq <= 0.0 {
        return Ok(None);
    }

    let inv_mu_a = h_sq / (MU * phase_sma);
    if !inv_mu_a.is_finite() || inv_mu_a <= 0.0 {
        return Ok(None);
    }
    let mut ecc_sq = 1.0_f64 - inv_mu_a;
    if ecc_sq < 0.0 {
        if ecc_sq > -1e-10 {
            ecc_sq = 0.0;
        } else {
            return Ok(None);
        }
    }
    let ecc = ecc_sq.sqrt();
    let dep_perigee = phase_sma * (1.0 - ecc);
    let dep_apogee = phase_sma * (1.0 + ecc);
    if !dep_perigee.is_finite()
        || !dep_apogee.is_finite()
        || dep_perigee < branch_base_ctx.min_perigee - radius_tol
        || dep_apogee > branch_base_ctx.max_apogee + radius_tol
    {
        return Ok(None);
    }

    let mut dep_after_phase_equ = [0.0; 6];
    if !eci_to_equinoctial(&dep_after_phase, &mut dep_after_phase_equ) {
        return Ok(None);
    }

    let Some(dep_at_release) = propagate_candidate_search_state_at_epoch(
        &dep_after_phase_equ,
        waittime,
        propagation_epoch_for_segment(branch_base_ctx.epoch_jd, time2phase),
        branch_base_ctx.transfer_body_force(),
        &branch_base_ctx,
    )?
    else {
        return Ok(None);
    };
    let phase_release_s = phase_release_start.elapsed().as_secs_f64();

    let Ok(tof_interval) =
        admissible_tof_interval(&branch_base_ctx, dep_period, max_transfer_headroom_s)
    else {
        return Ok(None);
    };
    let tof_lower = tof_interval.lower;
    let tof_upper = tof_interval.upper;

    let fixed_offset = waittime + time2phase;
    let span = tof_interval.span;
    let altitude_diff = (phase_sma - dep_phase_orbit.sma).abs();
    let is_simple_transfer = altitude_diff < 500.0 && ecc < 0.05;
    tof_grid_sample_count(span, is_simple_transfer)?.ok_or(EvaluationArithmeticOverflow)?;
    let sample_count = branch_base_ctx.search_depth.clamped_tof_budget();

    let plane_angle_cached = if branch_base_ctx.plane_angle_valid {
        branch_base_ctx.plane_angle
    } else {
        let h_dep = cross3(
            &[
                branch_base_ctx.dep_eci[0],
                branch_base_ctx.dep_eci[1],
                branch_base_ctx.dep_eci[2],
            ],
            &[
                branch_base_ctx.dep_eci[3],
                branch_base_ctx.dep_eci[4],
                branch_base_ctx.dep_eci[5],
            ],
        );
        let h_tgt = cross3(
            &[
                branch_base_ctx.tgt_eci[0],
                branch_base_ctx.tgt_eci[1],
                branch_base_ctx.tgt_eci[2],
            ],
            &[
                branch_base_ctx.tgt_eci[3],
                branch_base_ctx.tgt_eci[4],
                branch_base_ctx.tgt_eci[5],
            ],
        );
        let h_dep_norm = norm3(&h_dep);
        let h_tgt_norm = norm3(&h_tgt);
        if h_dep_norm > 1e-10 && h_tgt_norm > 1e-10 {
            let cos_angle = (h_dep[0] * h_tgt[0] + h_dep[1] * h_tgt[1] + h_dep[2] * h_tgt[2])
                / (h_dep_norm * h_tgt_norm);
            cos_angle.clamp(-1.0, 1.0).acos()
        } else {
            0.0
        }
    };

    let tof_budget = branch_base_ctx.search_depth.clamped_tof_budget();
    let mut tof_samples = [0.0; MAX_TOF_SAMPLES];
    let mut tof_sample_n = 0;
    let mut hohmann_tof = 0.0;
    let tgt_period = branch_base_ctx.tgt_period;
    tof_sample_ladder!(
        ctx = branch_base_ctx,
        phase_sma = phase_sma,
        dep_at_release = dep_at_release,
        plane_angle_cached = plane_angle_cached,
        tof_lower = tof_lower,
        tof_upper = tof_upper,
        span = span,
        sample_count = sample_count,
        is_simple_transfer = is_simple_transfer,
        tof_budget = tof_budget,
        tof_samples = tof_samples,
        tof_sample_n = tof_sample_n,
        hohmann_tof = hohmann_tof,
        tgt_period = tgt_period,
        period = phase_period,
        period_setup = [
            let phase_period = 2.0 * std::f64::consts::PI * ((phase_sma.powi(3)) / MU).sqrt();
        ],
        transfer_sma_hohmann = transfer_sma_hohmann,
        transfer_period = transfer_period,
        transfer_period_setup = [
            let transfer_period =
                2.0 * std::f64::consts::PI * ((transfer_sma_hohmann.powi(3)) / MU).sqrt();
        ],
        multi_rev_var = m,
        multi_rev_setup = [],
        multi_rev_sample = [hohmann_tof + f64::from(m) * phase_period],
        rev_entry_var = n,
        rev_entry_setup = [],
        rev_entry_sample = [hohmann_tof + (f64::from(n) - 0.5) * phase_period],
        bail = Ok(None),
    );
    if tof_sample_n == 0 {
        return Ok(None);
    }

    let target_start = Instant::now();
    // rust-alloc#3: MAX_TOF_SAMPLES inline (see PreparedTransferWork fields).
    let mut target_positions: SmallVec<[[f64; 3]; MAX_TOF_SAMPLES]> = SmallVec::new();
    let mut v2_refs: SmallVec<[[f64; 3]; MAX_TOF_SAMPLES]> = SmallVec::new();
    let mut valid_tofs: SmallVec<[f64; MAX_TOF_SAMPLES]> = SmallVec::new();
    let mut tof_to_idx: SmallVec<[usize; MAX_TOF_SAMPLES]> = SmallVec::new();
    target_positions.reserve(tof_sample_n);
    v2_refs.reserve(tof_sample_n);
    valid_tofs.reserve(tof_sample_n);
    tof_to_idx.reserve(tof_sample_n);
    let mut t_vals = [0.0_f64; MAX_TOF_SAMPLES];
    for i in 0..tof_sample_n {
        t_vals[i] = fixed_offset + tof_samples[i];
    }
    let mut out_states = [0.0_f64; MAX_TOF_SAMPLES * 6];
    let mut tgt_states_full = [[0.0_f64; 6]; MAX_TOF_SAMPLES];

    {
        let mut simd_idx = 0usize;
        if target_propagation_uses_j2(&branch_base_ctx) && tof_sample_n >= 4 {
            while simd_idx + 4 <= tof_sample_n {
                let t_chunk = [
                    t_vals[simd_idx],
                    t_vals[simd_idx + 1],
                    t_vals[simd_idx + 2],
                    t_vals[simd_idx + 3],
                ];
                let base = simd_idx * 6;
                propagate_target_cached_simd4(
                    &branch_base_ctx,
                    &t_chunk,
                    &mut out_states[base..base + 24],
                )?;
                simd_idx += 4;
            }
        }
        if target_propagation_uses_j2(&branch_base_ctx) && simd_idx < tof_sample_n {
            propagate_target_cached(
                &branch_base_ctx,
                &t_vals[simd_idx..tof_sample_n],
                &mut out_states[simd_idx * 6..simd_idx * 6 + (tof_sample_n - simd_idx) * 6],
            )?;
        } else if !target_propagation_uses_j2(&branch_base_ctx) {
            propagate_target_analytical(
                &branch_base_ctx,
                &t_vals[..tof_sample_n],
                &mut out_states[..tof_sample_n * 6],
            );
        }
        for i in 0..tof_sample_n {
            let base = i * 6;
            let tgt_state = [
                out_states[base],
                out_states[base + 1],
                out_states[base + 2],
                out_states[base + 3],
                out_states[base + 4],
                out_states[base + 5],
            ];
            if all_finite(&tgt_state) {
                tgt_states_full[i] = tgt_state;
                target_positions.push([tgt_state[0], tgt_state[1], tgt_state[2]]);
                v2_refs.push([tgt_state[3], tgt_state[4], tgt_state[5]]);
                valid_tofs.push(tof_samples[i]);
                tof_to_idx.push(i);
            }
        }
    }
    let target_propagation_s = target_start.elapsed().as_secs_f64();
    if valid_tofs.is_empty() {
        return Ok(None);
    }

    record_branch_shared_prepare(
        total_start.elapsed().as_secs_f64(),
        phase_release_s,
        target_propagation_s,
    )?;

    Ok(Some(PreparedTransferWork {
        x: *x,
        ctx: branch_base_ctx,
        coarse_mode,
        time2phase,
        waittime,
        dep_period,
        #[cfg(test)]
        dep_at_phase,
        #[cfg(test)]
        dep_phase_orbit,
        phase_sma,
        phase_dv,
        phase_dv_norm,
        dep_at_release,
        max_transfer_headroom_s,
        tof_lower,
        tof_upper,
        fixed_offset,
        plane_angle_cached,
        tgt_states_full,
        r2_vec: target_positions,
        v2_refs,
        valid_tofs,
        tof_to_idx,
    }))
}

/// Branch-invariant Lambert lane data for one prepared source.
///
/// Every `(rev, low_path)` branch of a source solves the same
/// `(r1, r2_vec, tofs)` batch, so the per-lane `m_max` bound and geometry are
/// built once per source and shared, instead of once per branch.
fn branch_lane_prep(prepared: &PreparedTransferWork) -> crate::lambert::BranchLanePrep {
    let mut prep = crate::lambert::BranchLanePrep::default();
    prep.rebuild(
        MU,
        &[
            prepared.dep_at_release[0],
            prepared.dep_at_release[1],
            prepared.dep_at_release[2],
        ],
        prepared.r2_vec.as_slice(),
        prepared.valid_tofs.as_slice(),
    );
    prep
}

#[expect(
    clippy::too_many_lines,
    reason = "one ordered branch path preserves shared scratch and exact selection semantics"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "prepared branch buffers carry validated fixed-size lane mappings"
)]
fn evaluate_prepared_plan_branch(
    prepared: &PreparedTransferWork,
    branch_ctx: &mut PlanContext,
    rev: i32,
    low_path: bool,
    lane_prep: &crate::lambert::BranchLanePrep,
    lambert_scratch: &mut crate::lambert::VariableR2LambertScratch,
) -> Result<PlanResult, EvaluationArithmeticOverflow> {
    record_branch_eval_call()?;
    branch_ctx.lambert_branch_selection = Some(LambertBranchSelection { rev, low_path });
    let mut res = prepared.invalid_result();

    let lambert_start = Instant::now();
    let mut best_tof = 0.0;
    let mut best_sol: Option<LambertSolutionEx> = None;
    let mut best_cost = INVALID_COST;

    let r1_transfer = [
        prepared.dep_at_release[0],
        prepared.dep_at_release[1],
        prepared.dep_at_release[2],
    ];
    let v1_ref = [
        prepared.dep_at_release[3],
        prepared.dep_at_release[4],
        prepared.dep_at_release[5],
    ];
    let r1_norm = norm3(&r1_transfer);
    let r2_norm = norm3(&prepared.r2_vec[0]);
    if r1_norm > 0.0 && r2_norm > 0.0 {
        let max_revs = branch_ctx.max_revs.max(0);
        record_lambert_batch_call(prepared.valid_tofs.len())?;
        // Whole-batch retrograde prune, mirroring the single-lane gate in
        // select_lambert_branch_solution: every accepted solution must pass
        // dv < max_transfer_dv, so dv_cap below is the acceptance cap. The
        // retrograde departure dv is rigorously bounded below per lane, and
        // the prograde basis rotates with each arrival state, so retrograde is
        // dropped only when EVERY lane's bound already meets/exceeds the cap
        // (min-over-lanes bound >= cap); one lane below the cap keeps
        // retrograde for the batch.
        let dv_cap = branch_ctx.max_transfer_dv;
        let include_retrograde = crate::lambert_backend::batch_retrograde_included(
            &prepared.dep_at_release,
            prepared
                .tof_to_idx
                .iter()
                .map(|&state_idx| &prepared.tgt_states_full[state_idx]),
            dv_cap,
        );
        let branch_results =
            crate::lambert::solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_lanes(
                MU,
                &r1_transfer,
                prepared.r2_vec.as_slice(),
                &v1_ref,
                prepared.v2_refs.as_slice(),
                prepared.valid_tofs.as_slice(),
                max_revs,
                low_path,
                include_retrograde,
                Some((rev, low_path)),
                Some(lane_prep),
                lambert_scratch,
            );
        // rust-alloc#3: up to valid_tofs.len() (<= MAX_TOF_SAMPLES) entries;
        // 64 inline heap-spilled under tof_sample_budget=256 configs.
        let mut batch_ranked: SmallVec<[(f64, usize); MAX_TOF_SAMPLES]> = SmallVec::new();
        for (batch_idx, result) in branch_results.iter().enumerate() {
            if !result.valid {
                continue;
            }
            let dv_norm = result.dv_depart;
            if !(dv_norm.is_finite() && dv_norm < branch_ctx.max_transfer_dv) {
                continue;
            }
            batch_ranked.push((dv_norm, batch_idx));
        }
        pdqsort::sort_by(&mut batch_ranked, |left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let batch_best_cost = batch_ranked
            .first()
            .map_or(INVALID_COST, |(cost, _idx)| *cost);
        let mut ranked_seen = [false; MAX_TOF_SAMPLES];
        for (_cost, idx) in &batch_ranked {
            if *idx < ranked_seen.len() {
                ranked_seen[*idx] = true;
            }
        }
        for (idx, seen) in ranked_seen
            .iter()
            .take(prepared.valid_tofs.len())
            .enumerate()
        {
            if !seen {
                batch_ranked.push((batch_best_cost, idx));
            }
        }
        for (_batch_cost, batch_idx) in &batch_ranked {
            let tof = prepared.valid_tofs[*batch_idx];
            let tgt_state = prepared.tgt_states_full[prepared.tof_to_idx[*batch_idx]];
            if let Some(sol) = branch_batch_result_to_solution(
                branch_ctx,
                &prepared.dep_at_release,
                &tgt_state,
                &branch_results[*batch_idx],
            ) {
                if sol.cost < best_cost {
                    best_tof = tof;
                    best_cost = sol.cost;
                    best_sol = Some(sol);
                }
            }
        }
        record_lambert_batch_work(lambert_scratch.branch_telemetry())?;
    }
    record_branch_lambert_sampling(lambert_start.elapsed().as_secs_f64())?;

    let Some(best_sol_value) = best_sol else {
        return Ok(res);
    };

    let is_large_plane_change = prepared.plane_angle_cached > 1.57;
    let bracket_lo = prepared.tof_lower;
    let bracket_hi = prepared.tof_upper;
    let mut final_tof = best_tof;
    let mut final_sol = best_sol_value;

    if brent_refinement_required(bracket_lo, bracket_hi) {
        let brent_start = Instant::now();
        let maxiter = if prepared.coarse_mode && !is_large_plane_change {
            12
        } else {
            BRENT_FINE_MAX_ITERATIONS
        };
        let x_tolerance = if prepared.coarse_mode && !is_large_plane_change {
            ((bracket_hi - bracket_lo) * 0.10).clamp(10.0, 300.0)
        } else {
            ((bracket_hi - bracket_lo) * 0.01).clamp(1.0, 60.0)
        };
        let mut brent_cache = BrentLocalCache::new();
        let mut brent_exact_solution_cache = BrentExactSolutionCache::new();
        // perf-hunt-r2 #3: hoisted departure-side Lambert cache (fixed
        // departure state across this Brent search; bit-identical).
        let brent_departure_bounds =
            crate::lambert_backend::DepartureBoundCache::new(&prepared.dep_at_release);
        let brent_r1_cache = crate::lambert::LambertR1Cache::new(&[
            prepared.dep_at_release[0],
            prepared.dep_at_release[1],
            prepared.dep_at_release[2],
        ]);
        let mut brent_eval_request_count = 0usize;
        let mut brent_cache_hit_count = 0usize;
        let mut brent_cache_miss_count = 0usize;
        let bracket_span = bracket_hi - bracket_lo;
        if bracket_span > 1.0 {
            let best_tof_key = brent_tof_cache_key(best_tof)?;
            brent_cache_insert_first(&mut brent_cache, best_tof_key, best_cost);
        }

        // Same multimodal-safe bracketing as the scalar evaluator. The two
        // paths must agree: the front reports the BRANCH row's dv, and the
        // scalar plan dv is what the outer search steers on.
        //
        // R21 S4: this route's pre-scan is the selected-branch counterpart of
        // the scalar evaluator's, and it batches the same way — one cross-TOF
        // pack over the ladder's misses instead of one two-lane pack per
        // sample. Sequential equivalence is `prescan_eval_samples_batched`'s
        // contract; it runs before `eval_tof` exists so the caches it fills are
        // the ones the sequential closure would have filled.
        let census_samples =
            brent_prescan_count(bracket_span, prepared.coarse_mode, is_large_plane_change)?;
        let prescan_costs = prescan_eval_samples_batched(
            &PrescanBatchInputs {
                ctx: branch_ctx,
                dep_at_release: &prepared.dep_at_release,
                r1_cache: &brent_r1_cache,
                fixed_offset: prepared.fixed_offset,
                max_transfer_headroom_s: prepared.max_transfer_headroom_s,
                bracket_lo,
                bracket_hi,
                samples: census_samples,
            },
            &mut brent_cache,
            &mut brent_exact_solution_cache,
            &mut brent_eval_request_count,
            &mut brent_cache_hit_count,
            &mut brent_cache_miss_count,
        )?;

        let mut eval_tof = brent_eval_tof_closure!(
            ctx = branch_ctx,
            dep_at_release = &prepared.dep_at_release,
            fixed_offset = prepared.fixed_offset,
            max_transfer_headroom_s = prepared.max_transfer_headroom_s,
            r1_cache = brent_r1_cache,
            departure_bounds = brent_departure_bounds,
            cache = brent_cache,
            exact_solution_cache = brent_exact_solution_cache,
            request_count = brent_eval_request_count,
            hit_count = brent_cache_hit_count,
            miss_count = brent_cache_miss_count,
        );

        let (scan_lo, scan_hi, scan_tof, scan_cost) = if let Some(costs) = prescan_costs {
            brent_prescan_bracket_from_costs(
                bracket_lo,
                bracket_hi,
                census_samples,
                best_tof,
                best_cost,
                costs,
            )?
        } else {
            brent_prescan_bracket(
                bracket_lo,
                bracket_hi,
                census_samples,
                best_tof,
                best_cost,
                &mut eval_tof,
            )?
        };
        let x_tolerance = ((scan_hi - scan_lo) * 0.01)
            .clamp(1.0, 60.0)
            .min(x_tolerance);

        let mres = minimize_scalar_bounded(&mut eval_tof, scan_lo, scan_hi, x_tolerance, maxiter)?;

        if scan_cost < best_cost {
            if let Some(sol) = brent_exact_solution_lookup(&brent_exact_solution_cache, scan_tof) {
                final_tof = scan_tof;
                final_sol = sol;
                best_cost = sol.cost;
            }
        }

        if mres.converged && mres.x.is_finite() {
            let refined_opt = if let Some(solution) =
                brent_exact_solution_lookup(&brent_exact_solution_cache, mres.x)
            {
                Some(solution)
            } else {
                lambert_solve_raw(
                    mres.x,
                    branch_ctx,
                    &prepared.dep_at_release,
                    &brent_r1_cache,
                    &brent_departure_bounds,
                    prepared.fixed_offset,
                    prepared.max_transfer_headroom_s,
                )?
            };
            if let Some(refined) = refined_opt {
                if refined.cost < best_cost {
                    final_tof = mres.x;
                    final_sol = refined;
                    best_cost = refined.cost;
                }
            }
        }
        record_branch_brent_cache_counts(
            brent_eval_request_count,
            brent_cache_hit_count,
            brent_cache_miss_count,
        )?;
        record_branch_brent(brent_start.elapsed().as_secs_f64())?;
    }

    if !final_sol.valid || !best_cost.is_finite() {
        res.cost = INVALID_COST - 2000.0 + prepared.phase_dv_norm;
        res.valid = false;
        res.tof = final_tof;
        return Ok(res);
    }

    if transfer_tof_exceeds_revolution_cap(branch_ctx, final_tof, prepared.dep_period) {
        res.cost = INVALID_COST - 1900.0 + prepared.phase_dv_norm;
        res.valid = false;
        res.tof = final_tof;
        res.timing_failure_reason = TimingFailureToken::TransferRevolutionCapExceeded;
        return Ok(res);
    }

    let j2_start = Instant::now();
    let j2_settings = branch_ctx.j2_closure_settings;
    let physical_target_state = final_sol.tgt_state;
    let (refined_sol, iter_count, residual_m) = apply_iterative_j2_lambert_correction(
        branch_ctx,
        &prepared.dep_at_release,
        final_tof,
        propagation_epoch_for_segment(branch_ctx.epoch_jd, prepared.time2phase + prepared.waittime),
        final_sol,
        j2_settings,
    )?;
    final_sol = refined_sol;
    let j2_iteration_count = j2_iteration_count_as_u32(iter_count)?;
    let j2_endpoint_residual_m = residual_m;
    record_branch_j2_correction(j2_start.elapsed().as_secs_f64())?;

    let j2_residual_blocks = pre_hf_j2_residual_blocks_acceptance(
        branch_ctx.execution_policy.use_high_fidelity,
        j2_endpoint_residual_m,
        j2_settings.endpoint_target_km * 1000.0,
    );
    record_j2_residual_gate(j2_residual_blocks, j2_endpoint_residual_m)?;
    if j2_residual_blocks {
        res.cost = INVALID_COST - 1750.0 + prepared.phase_dv_norm;
        res.valid = false;
        res.tof = final_tof;
        res.j2_iteration_count = j2_iteration_count;
        res.j2_endpoint_residual_m = j2_endpoint_residual_m;
        return Ok(res);
    }

    let final_transfer_dv = final_sol.dv;
    let cached_hf_endpoint = None;

    let transfer_dv_norm = norm3(&final_transfer_dv);
    let transfer_dv_penalty =
        transfer_dv_limit_penalty(transfer_dv_norm, branch_ctx.max_transfer_dv);
    let total_dv = prepared.phase_dv_norm + transfer_dv_norm;
    let running_total_cost = total_dv;

    let mut transfer_start = prepared.dep_at_release;
    add_velocity(&mut transfer_start, &final_transfer_dv);
    if let Some(transfer_orbit) = eci_orbit_summary(&transfer_start) {
        let radius_tol = radius_tolerance(branch_ctx.min_perigee, branch_ctx.max_apogee);
        let perigee_violation =
            (branch_ctx.min_perigee - radius_tol - transfer_orbit.perigee).max(0.0);
        let apogee_violation =
            (transfer_orbit.apogee - (branch_ctx.max_apogee + radius_tol)).max(0.0);
        let orbit_penalty = (perigee_violation + apogee_violation) * 100.0;
        let re_entry_penalty = if transfer_orbit.perigee < RE {
            (RE - transfer_orbit.perigee) * 10000.0
        } else {
            0.0
        };
        if transfer_orbit.perigee >= RE {
            let mut transfer_equ = [0.0; 6];
            let dep_at_intercept = match cached_hf_endpoint {
                Some(endpoint) => Some(endpoint),
                None if !eci_to_equinoctial(&transfer_start, &mut transfer_equ) => None,
                None => propagate_candidate_search_state_at_epoch(
                    &transfer_equ,
                    final_tof,
                    propagation_epoch_for_segment(
                        branch_ctx.epoch_jd,
                        prepared.time2phase + prepared.waittime,
                    ),
                    branch_ctx.transfer_body_force(),
                    branch_ctx,
                )?,
            };
            if let Some(dep_at_intercept) = dep_at_intercept {
                let post_hf_endpoint_residual_m = recompute_post_hf_endpoint_residual_m(
                    &dep_at_intercept,
                    &physical_target_state,
                );
                let distance = post_hf_endpoint_residual_m / 1000.0;
                let distance_penalty = if distance > branch_ctx.distance_tol {
                    (distance - branch_ctx.distance_tol) * 1000.0
                } else {
                    0.0
                };
                let mut deployer_release_equ = [0.0; 6];
                if eci_to_equinoctial(&prepared.dep_at_release, &mut deployer_release_equ) {
                    let Some(deployer_at_intercept) = propagate_candidate_search_state_at_epoch(
                        &deployer_release_equ,
                        final_tof,
                        propagation_epoch_for_segment(
                            branch_ctx.epoch_jd,
                            prepared.time2phase + prepared.waittime,
                        ),
                        branch_ctx.transfer_body_force(),
                        branch_ctx,
                    )?
                    else {
                        return Ok(res);
                    };
                    {
                        let deployer_distance =
                            vec_distance(&deployer_at_intercept[..3], &physical_target_state[..3]);
                        let deployer_penalty =
                            if deployer_distance < branch_ctx.deployer_min_distance {
                                (branch_ctx.deployer_min_distance - deployer_distance) * 1000.0
                            } else {
                                0.0
                            };
                        let final_distance = if distance < branch_ctx.distance_tol {
                            0.0
                        } else {
                            distance
                        };
                        res.payload_intercept_state = dep_at_intercept;
                        res.target_intercept_state = physical_target_state;
                        res.deployer_intercept_state = deployer_at_intercept;
                        res.release_state = prepared.dep_at_release;
                        res.cost = final_distance
                            + running_total_cost
                            + deployer_penalty
                            + transfer_dv_penalty
                            + distance_penalty
                            + orbit_penalty
                            + re_entry_penalty;
                        res.post_hf_endpoint_residual_m = post_hf_endpoint_residual_m;
                        res.valid = post_hf_residual_accepts(
                            post_hf_endpoint_residual_m,
                            branch_ctx.distance_tol * 1000.0,
                        ) && re_entry_penalty == 0.0
                            && orbit_penalty == 0.0
                            && transfer_dv_penalty == 0.0
                            && distance_penalty == 0.0
                            && deployer_penalty == 0.0;
                        res.tof = final_tof;
                        res.distance = final_distance;
                        res.deployer_distance = deployer_distance;
                        copy3(&mut res.transfer_dv, &final_transfer_dv);
                        let arrival_dv =
                            rendezvous_arrival_dv(&dep_at_intercept, &physical_target_state);
                        copy3(&mut res.arrival_dv, &arrival_dv);
                        res.transfer_dv_norm = transfer_dv_norm;
                        res.arrival_dv_norm = norm3(&arrival_dv);
                        res.best_M = final_sol.best_M;
                        res.set_accepted_branch(
                            final_sol.best_M,
                            final_sol.low_path,
                            final_tof,
                            transfer_dv_norm,
                            res.arrival_dv_norm,
                        );
                        res.prograde = final_sol.prograde;
                        res.j2_iteration_count = j2_iteration_count;
                        res.j2_endpoint_residual_m = j2_endpoint_residual_m;
                        res.waittime_jd_start = branch_ctx.epoch_jd + prepared.time2phase / 86400.0;
                        res.tof_jd_start = res.waittime_jd_start + prepared.waittime / 86400.0;
                        res.intercept_jd = res.tof_jd_start + final_tof / 86400.0;
                        if res.valid && res.cost < INVALID_COST {
                            record_branch_emitted()?;
                        }
                        return Ok(res);
                    }
                }
            }
        }
    }

    res.tof = final_tof;
    res.transfer_dv_norm = transfer_dv_norm;
    res.cost = INVALID_COST - 500.0 + running_total_cost;
    res.j2_iteration_count = j2_iteration_count;
    res.j2_endpoint_residual_m = j2_endpoint_residual_m;
    res.valid = false;
    Ok(res)
}

#[cfg(test)]
fn evaluate_plan_branches_reference(
    x: &[f64; 3],
    ctx: &PlanContext,
    coarse_mode: bool,
) -> Result<Vec<PlanResult>, EvaluationArithmeticOverflow> {
    if !candidate_search_is_supported(ctx) {
        return Ok(Vec::new());
    }
    let time2phase_ratio = x[0];
    let waittime_ratio = x[2];
    if !time2phase_ratio.is_finite() || !x[1].is_finite() || !waittime_ratio.is_finite() {
        return Ok(Vec::new());
    }
    if time2phase_ratio + waittime_ratio >= 1.0 {
        return Ok(Vec::new());
    }

    let time2phase = time2phase_ratio * ctx.max_time_s;
    let waittime = waittime_ratio * ctx.max_time_s;
    if transfer_timing_window(ctx, time2phase, waittime).is_err() {
        return Ok(Vec::new());
    }

    let dep_period = compute_dep_period(ctx);
    let dep_at_phase = propagate_candidate_state_at_epoch(
        &ctx.dep_equ,
        time2phase,
        ctx.epoch_jd,
        ctx.transfer_body_force(),
        ctx,
    )?;
    let Some(dep_at_phase) = dep_at_phase else {
        return Ok(Vec::new());
    };
    let Some(dep_phase_orbit) = eci_orbit_summary(&dep_at_phase) else {
        return Ok(Vec::new());
    };

    let max_revs = ctx.max_revs.max(0);
    let mut out = branch_plan_output(max_revs)?;
    let mut branch_ctx = ctx.clone();
    for rev in 0..=max_revs {
        let low_paths: &[bool] = if rev == 0 { &[true] } else { &[true, false] };
        for &low_path in low_paths {
            branch_ctx.lambert_branch_selection = Some(LambertBranchSelection { rev, low_path });
            let plan = evaluate_plan_from_phase_with_lambert_scratch(
                x,
                &branch_ctx,
                coarse_mode,
                time2phase,
                waittime,
                dep_period,
                &dep_at_phase,
                Some(dep_phase_orbit),
                None,
            )?;
            if plan.valid && plan.cost < INVALID_COST {
                out.push(plan);
            }
        }
    }
    sort_and_dedup_branch_plans(&mut out);
    Ok(out)
}

fn sort_and_dedup_branch_plans(out: &mut Vec<PlanResult>) {
    out.sort_by(|left, right| {
        left.total_dv()
            .partial_cmp(&right.total_dv())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.total_time()
                    .partial_cmp(&right.total_time())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.branch_rev.cmp(&right.branch_rev))
            .then_with(|| left.branch_low_path.cmp(&right.branch_low_path))
    });
    out.dedup_by(|right, left| {
        left.branch_rev == right.branch_rev
            && left.branch_low_path == right.branch_low_path
            && (left.tof - right.tof).abs() <= 1e-6
            && (left.total_dv() - right.total_dv()).abs() <= 1e-12
    });
}

#[inline]
fn branch_plan_output(max_revs: i32) -> Result<Vec<PlanResult>, EvaluationArithmeticOverflow> {
    let revolutions = usize::try_from(max_revs.max(0)).map_err(|_| EvaluationArithmeticOverflow)?;
    let capacity = revolutions
        .checked_add(1)
        .and_then(|source_count| source_count.checked_mul(2))
        .ok_or(EvaluationArithmeticOverflow)?;
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|_| EvaluationArithmeticOverflow)?;
    Ok(out)
}

macro_rules! define_evaluate_plan_branches_with_scratch {
    ($visibility:vis) => {
        /// Branch evaluation with a caller-provided Lambert scratch (rust-alloc#2).
        ///
        /// Loop callers construct ONE `VariableR2LambertScratch` above the loop so
        /// the four scratch buffers reach steady-state capacity once instead of
        /// reallocating per candidate/grid point. Reuse across calls is bit-identical
        /// because the `_with_scratch` Lambert batch entries clear every scratch
        /// field at entry.
        ///
        /// # Errors
        ///
        /// Returns [`EvaluationArithmeticOverflow`] when evaluator-owned diagnostic
        /// accounting cannot represent the required work.
        $visibility fn evaluate_plan_branches_with_scratch(
            x: &[f64; 3],
            ctx: &PlanContext,
            coarse_mode: bool,
            lambert_scratch: &mut crate::lambert::VariableR2LambertScratch,
        ) -> Result<Vec<PlanResult>, EvaluationArithmeticOverflow> {
            if !candidate_search_is_supported(ctx) {
                return Ok(Vec::new());
            }
            let Some(prepared) = prepare_branch_shared_work(x, ctx, coarse_mode)? else {
                return Ok(Vec::new());
            };
            record_branch_source()?;
            let max_revs = ctx.max_revs.max(0);
            let mut out = branch_plan_output(max_revs)?;
            let mut branch_ctx = prepared.ctx.clone();
            // Every branch below solves the SAME Lambert batch — same departure state,
            // same arrival positions, same TOFs — and differs only in which
            // `(rev, low_path)` row it selects. The `m_max` bound and the Lambert
            // geometry are functions of that batch alone, so they are built once here
            // instead of `1 + 2 * max_revs` times (nine at production `max_revs = 4`).
            let lane_prep = branch_lane_prep(&prepared);
            for rev in 0..=max_revs {
                let low_paths: &[bool] = if rev == 0 { &[true] } else { &[true, false] };
                for &low_path in low_paths {
                    let plan = evaluate_prepared_plan_branch(
                        &prepared,
                        &mut branch_ctx,
                        rev,
                        low_path,
                        &lane_prep,
                        lambert_scratch,
                    )?;
                    if plan.valid && plan.cost < INVALID_COST {
                        out.push(plan);
                    } else {
                        record_branch_rejected()?;
                    }
                }
            }
            sort_and_dedup_branch_plans(&mut out);
            Ok(out)
        }
    };
}

#[cfg(feature = "bench-internal")]
define_evaluate_plan_branches_with_scratch!(pub);

#[cfg(not(feature = "bench-internal"))]
define_evaluate_plan_branches_with_scratch!(pub(crate));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        BodyForceConfig, BodyRole, ExecutionPolicy, PlanContext, PropagationFidelity, SamplingMode,
        StampedEciState, TargetPropagationAuthority,
    };

    /// Records `calls` J2 corrections of `residual_m` metres each through the
    /// production record site, so the thread-local grows exactly the way a
    /// running campaign grows it.
    fn record_j2_residuals(calls: usize, residual_m: f64) {
        for _ in 0..calls {
            record_j2_correction(1, 0, residual_m).expect("residual record must not overflow");
        }
    }

    /// The left-associated sum the record loop above produces from zero.
    fn left_associated_sum(calls: usize, residual_m: f64) -> f64 {
        let mut total = 0.0_f64;
        for _ in 0..calls {
            total += residual_m;
        }
        total
    }

    /// A work unit's reported J2 residual must not depend on how long the
    /// thread executing it has been alive.
    ///
    /// `j2_correction_residual_m_sum` is METRES and reaches
    /// `VerifiedSupersetStageMetrics`, so this is a reported physics number.
    /// Production used to report it as `snapshot() - baseline`, i.e.
    /// `fl(B + a_1 + ... + a_k) - B` against a thread-local that nothing zeroes
    /// outside tests -- so `B` was the thread's whole-campaign history and the
    /// cancellation error grew with campaign length. The region API accumulates
    /// the unit into a fresh zero instead, which is exact.
    ///
    /// Both shapes are computed here on the same thread from the same history,
    /// so the test records what the defect cost as well as proving it is gone.
    #[test]
    fn diagnostic_region_reports_a_unit_independently_of_thread_history() {
        std::thread::spawn(|| {
            // Each arm: (history calls, history residual m, expected history m,
            // work-unit residual m, minimum relative error the subtraction
            // shape must show).
            //
            // Histories are accumulated through the real record site, and
            // 1e5 x 1e7 m / 1e6 x 1e7 m are both exact in binary64 -- so the
            // baseline is a round number and every bit lost below is lost by
            // the subtraction, not by building the baseline.
            //
            // Two work-unit scales, because the defect's size is set by the
            // ratio of the baseline's ulp to the unit: 1 mm is a converged
            // endpoint closure, and 6.32 m is the mean accepted residual this
            // crate's own `audit_work_count_metrics` LEO fixture produces
            // (5.21e3 m over 824 accepted corrections). The mm scale is where
            // the reported number is destroyed; the metre scale is what a
            // realistic campaign sees, and it is small. Both are unbounded in
            // the baseline, which is the actual finding.
            const UNIT_CALLS: usize = 5;
            const ARMS: [(usize, f64, f64, f64, f64); 4] = [
                (100_000, 1.0e7, 1.0e12, 1.0e-3, 0.02),
                (1_000_000, 1.0e7, 1.0e13, 1.0e-3, 0.5),
                (100_000, 1.0e7, 1.0e12, 6.32, 1.0e-6),
                (1_000_000, 1.0e7, 1.0e13, 6.32, 1.0e-5),
            ];

            for (history_calls, history_residual_m, expected_history_m, unit_residual_m, min_rel) in
                ARMS
            {
                let truth = left_associated_sum(UNIT_CALLS, unit_residual_m);
                assert!(
                    truth > 0.0,
                    "the unit under test must record a nonzero residual"
                );

                // Reference arm: the same unit on a thread with no history.
                restore_evaluation_diagnostics(&EvaluationDiagnosticCounters::default());
                let fresh_outer = enter_evaluation_diagnostic_region();
                record_j2_residuals(UNIT_CALLS, unit_residual_m);
                let fresh = evaluation_diagnostic_snapshot();
                restore_evaluation_diagnostics(&fresh_outer);
                assert_eq!(
                    fresh.j2_correction_residual_m_sum.to_bits(),
                    truth.to_bits(),
                    "a zero-based region must report the unit's own sum exactly"
                );

                // Each arm starts from a genuinely fresh thread state, so the
                // history it builds is the only history in play.
                restore_evaluation_diagnostics(&EvaluationDiagnosticCounters::default());
                record_j2_residuals(history_calls, history_residual_m);
                let history = evaluation_diagnostic_snapshot().j2_correction_residual_m_sum;
                // The history is the poison. Confirm it APPLIED, and that it
                // landed on the exact value the arithmetic below assumes --
                // a baseline that silently failed to build proves nothing.
                assert_eq!(
                    history.to_bits(),
                    expected_history_m.to_bits(),
                    "thread history must be {expected_history_m:e} m, got {history:e}"
                );

                // FIXED shape: zero-based region.
                let outer = enter_evaluation_diagnostic_region();
                assert_eq!(
                    evaluation_diagnostic_snapshot(),
                    EvaluationDiagnosticCounters::default(),
                    "entering a region must zero the thread-local, otherwise \
                     'accumulate from a fresh zero' is a claim and not a fact"
                );
                record_j2_residuals(UNIT_CALLS, unit_residual_m);
                let fixed = evaluation_diagnostic_snapshot();
                leave_evaluation_diagnostic_region(&outer, &fixed)
                    .expect("region close must not overflow");
                assert_eq!(
                    evaluation_diagnostic_snapshot()
                        .j2_correction_residual_m_sum
                        .to_bits(),
                    (expected_history_m + truth).to_bits(),
                    "closing a region must fold the unit back into the history"
                );

                // DEFECT shape, spelled out exactly as production used to:
                // subtract the pre-unit snapshot from the post-unit snapshot.
                let before = evaluation_diagnostic_snapshot().j2_correction_residual_m_sum;
                record_j2_residuals(UNIT_CALLS, unit_residual_m);
                let after = evaluation_diagnostic_snapshot().j2_correction_residual_m_sum;
                let subtracted = after - before;

                assert_eq!(
                    fixed.j2_correction_residual_m_sum.to_bits(),
                    fresh.j2_correction_residual_m_sum.to_bits(),
                    "at history {expected_history_m:e} m the region-reported \
                     residual moved: {} vs the fresh-thread {}",
                    fixed.j2_correction_residual_m_sum,
                    fresh.j2_correction_residual_m_sum
                );
                assert_ne!(
                    subtracted.to_bits(),
                    truth.to_bits(),
                    "at history {expected_history_m:e} m the subtraction shape \
                     must be shown LOSING precision, or this test is not \
                     demonstrating anything: got {subtracted:e} for {truth:e}"
                );
                let relative_error = (subtracted - truth).abs() / truth;
                assert!(
                    relative_error > min_rel,
                    "at history {expected_history_m:e} m with a {unit_residual_m:e} m \
                     unit the subtraction shape lost only {relative_error:e} \
                     relative, under the {min_rel:e} this arm exists to show"
                );
                println!(
                    "history {expected_history_m:e} m, unit {unit_residual_m:e} m: \
                     truth {truth:e} m, region {:e} m (exact), subtraction \
                     {subtracted:e} m ({:+.4}% error)",
                    fixed.j2_correction_residual_m_sum,
                    100.0 * (subtracted - truth) / truth
                );
            }
        })
        .join()
        .expect("diagnostic history thread must not panic");
    }

    // 7.3 work-count audit: prove the per-worker reduction primitive keeps
    // the serial around-solve accounting exact. Runs on a freshly spawned
    // thread so the thread-local counters start at zero and cannot be
    // perturbed by concurrently running tests.
    #[test]
    fn merge_evaluation_diagnostics_is_serial_identity() {
        std::thread::spawn(|| {
            // 1. Landing the primitive is a no-op for serial code: merging a
            //    zero delta leaves the thread-local snapshot byte-identical.
            let before = evaluation_diagnostic_snapshot();
            merge_evaluation_diagnostics(&EvaluationDiagnosticCounters::default())
                .expect("zero diagnostic merge must not overflow");
            let after = evaluation_diagnostic_snapshot();
            assert_eq!(
                before, after,
                "merging a zero delta must not perturb serial counts"
            );

            // 2. Reduction correctness: a worker-computed delta folded in via
            //    the primitive matches doing the same work inline. Emit some
            //    real diagnostic increments, capture their delta, reset by
            //    spawning a second thread, and replay through the merge.
            let base = evaluation_diagnostic_snapshot();
            record_lambert_branch_solution(0, true, true)
                .expect("first branch diagnostic record must not overflow");
            record_lambert_branch_solution(2, false, false)
                .expect("second branch diagnostic record must not overflow");
            let inline = evaluation_diagnostic_snapshot();
            let worker_delta = inline
                .delta_since(base)
                .expect("monotonic diagnostic snapshot must have a delta");
            assert_eq!(worker_delta.lambert_branch_valid_count, 2);

            // The parent thread's counters already hold `inline`; merging the
            // same worker delta again must add exactly the delta (idempotent
            // field-wise accumulation), never drop or double-scale a field.
            merge_evaluation_diagnostics(&worker_delta)
                .expect("diagnostic merge must not overflow");
            let merged = evaluation_diagnostic_snapshot();
            let expected_valid = inline
                .lambert_branch_valid_count
                .checked_add(worker_delta.lambert_branch_valid_count)
                .expect("replayed valid-count delta must remain representable");
            let expected_rev0 = inline
                .lambert_branch_rev0_count
                .checked_add(worker_delta.lambert_branch_rev0_count)
                .expect("replayed rev0-count delta must remain representable");
            let expected_rev_gt0 = inline
                .lambert_branch_rev_gt0_count
                .checked_add(worker_delta.lambert_branch_rev_gt0_count)
                .expect("replayed rev-gt0-count delta must remain representable");
            assert_eq!(merged.lambert_branch_valid_count, expected_valid,);
            assert_eq!(merged.lambert_branch_rev0_count, expected_rev0,);
            assert_eq!(merged.lambert_branch_rev_gt0_count, expected_rev_gt0,);
        })
        .join()
        .expect("reduction identity thread panicked");
    }

    #[test]
    fn diagnostic_source_overflow_preserves_thread_local_snapshot() {
        std::thread::spawn(|| {
            let before = EvaluationDiagnosticCounters {
                lambert_branch_attempt_count: usize::MAX,
                lambert_branch_valid_count: 17,
                ..EvaluationDiagnosticCounters::default()
            };
            restore_evaluation_diagnostics(&before);

            assert_eq!(
                record_lambert_branch_solution(0, true, true),
                Err(EvaluationArithmeticOverflow)
            );
            assert_eq!(evaluation_diagnostic_snapshot(), before);
        })
        .join()
        .expect("diagnostic source overflow thread panicked");
    }

    #[test]
    fn diagnostic_delta_overflow_is_atomic_at_outer_and_hf_levels() {
        let before = EvaluationDiagnosticCounters {
            lambert_batch_call_count: 17,
            lambert_batch_row_count: usize::MAX,
            hf_propagation: HfPropagationTelemetry {
                target_grid_call_count: 17,
                target_grid_requested_state_count: usize::MAX,
                ..HfPropagationTelemetry::default()
            },
            ..EvaluationDiagnosticCounters::default()
        };
        let delta = EvaluationDiagnosticCounters {
            lambert_batch_call_count: 1,
            lambert_batch_row_count: 1,
            hf_propagation: HfPropagationTelemetry {
                target_grid_call_count: 1,
                target_grid_requested_state_count: 1,
                ..HfPropagationTelemetry::default()
            },
            ..EvaluationDiagnosticCounters::default()
        };
        let mut counters = before;

        assert_eq!(
            counters.add_delta(&delta),
            Err(EvaluationArithmeticOverflow)
        );
        assert_eq!(counters, before);

        let mut hf = before.hf_propagation;
        assert_eq!(
            hf.add_delta(&delta.hf_propagation),
            Err(EvaluationArithmeticOverflow)
        );
        assert_eq!(hf, before.hf_propagation);
    }

    #[test]
    fn diagnostic_delta_underflow_is_typed_and_exact_boundary_succeeds() {
        let current = EvaluationDiagnosticCounters::default();
        let before = EvaluationDiagnosticCounters {
            branch_source_count: 1,
            ..EvaluationDiagnosticCounters::default()
        };
        assert_eq!(
            current.delta_since(before),
            Err(EvaluationArithmeticOverflow)
        );

        let mut counters = EvaluationDiagnosticCounters {
            branch_source_count: usize::MAX
                .checked_sub(1)
                .expect("max-minus-one test counter must remain representable"),
            ..EvaluationDiagnosticCounters::default()
        };
        let one = EvaluationDiagnosticCounters {
            branch_source_count: 1,
            ..EvaluationDiagnosticCounters::default()
        };
        counters
            .add_delta(&one)
            .expect("max-minus-one plus one must remain representable");
        assert_eq!(counters.branch_source_count, usize::MAX);
    }

    #[test]
    fn j2_iteration_count_narrowing_is_checked() {
        assert_eq!(j2_iteration_count_as_u32(7), Ok(7));
        assert_eq!(
            j2_iteration_count_as_u32(usize::MAX),
            Err(EvaluationArithmeticOverflow)
        );
    }

    #[test]
    fn strict_hf_transfer_preserves_mf_catalogue_target_authority() {
        let ctx = PlanContext {
            execution_policy: ExecutionPolicy {
                use_high_fidelity: true,
                require_high_fidelity: true,
                allow_parallel: false,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            target_propagation_authority: TargetPropagationAuthority::MfJ2,
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };

        assert!(!target_propagation_uses_high_fidelity(&ctx));
        assert!(target_propagation_uses_j2(&ctx));
        assert_eq!(
            replay_provenance_from_context(&ctx).target_propagation_mode,
            1
        );
    }

    #[test]
    fn target_propagation_dispatch_maps_kepler_j2_and_hf_exactly() {
        let mut ctx = PlanContext {
            execution_policy: ExecutionPolicy {
                use_high_fidelity: false,
                require_high_fidelity: false,
                allow_parallel: false,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            target_propagation_authority: TargetPropagationAuthority::AnalyticalKepler,
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };
        assert!(!target_propagation_uses_high_fidelity(&ctx));
        assert!(!target_propagation_uses_j2(&ctx));
        assert_eq!(
            replay_provenance_from_context(&ctx).target_propagation_mode,
            2
        );

        ctx.target_propagation_authority = TargetPropagationAuthority::MfJ2;
        assert!(!target_propagation_uses_high_fidelity(&ctx));
        assert!(target_propagation_uses_j2(&ctx));
        assert_eq!(
            replay_provenance_from_context(&ctx).target_propagation_mode,
            1
        );

        ctx.target_propagation_authority = TargetPropagationAuthority::HighFidelity;
        assert!(!target_propagation_authority_is_consistent(&ctx));
        ctx.execution_policy.use_high_fidelity = true;
        assert!(!target_propagation_authority_is_consistent(&ctx));
        ctx.force_config = Some(std::sync::Arc::new(
            lightyear_odeint_rs::types::ForceConfig {
                sph_order: 5,
                force_flags: lightyear_odeint_rs::types::ForceFlags::DRAG
                    | lightyear_odeint_rs::types::ForceFlags::SRP
                    | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
                    | lightyear_odeint_rs::types::ForceFlags::MOON_GRAVITY,
                atm_model: crate::types::HIGH_FIDELITY_ATM_MODEL,
                target_propagation_mode: TargetPropagationAuthority::HighFidelity
                    .as_force_config_code(),
                ..Default::default()
            },
        ));
        ctx.target_body_force =
            BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 50.0, 2.2, 1.3);
        assert!(target_propagation_authority_is_consistent(&ctx));
        assert!(target_propagation_uses_high_fidelity(&ctx));
        assert!(!target_propagation_uses_j2(&ctx));

        ctx.force_config = Some(std::sync::Arc::new(
            lightyear_odeint_rs::types::ForceConfig {
                target_propagation_mode: TargetPropagationAuthority::MfJ2.as_force_config_code(),
                ..Default::default()
            },
        ));
        assert!(!target_propagation_authority_is_consistent(&ctx));
    }

    fn independent_hf_segment_reference(
        eci: &[f64; 6],
        equ: &[f64; 6],
        dt: f64,
        source_jd: f64,
        body_force: BodyForceConfig,
        ctx: &PlanContext,
    ) -> [f64; 6] {
        let mut config = *ctx
            .force_config
            .as_ref()
            .expect("force-enabled reference requires config")
            .as_ref();
        config.am_ratio = body_force.am_ratio;
        config.cd = body_force.cd;
        config.cr = body_force.cr;
        if matches!(body_force.role, BodyRole::TransferVehicle) {
            config.force_flags &= !(lightyear_odeint_rs::types::ForceFlags::DRAG
                | lightyear_odeint_rs::types::ForceFlags::SRP);
        }
        let config = config
            .with_ephemeris_for_arc(source_jd, source_jd + dt / SEC_PER_DAY)
            .expect("reference must retain all enabled forces across full arc");
        let gravity = lightyear_odeint_rs::ScalarGravityAssets::new(
            ctx.packed_coeffs.clone().expect("packed coefficients"),
        );
        let propagation_context = lightyear_odeint_rs::ScalarPropagationContext::new(
            source_jd,
            std::sync::Arc::new(config),
            gravity,
        );
        let t_eval = [dt];
        let request = lightyear_odeint_rs::ScalarPropagationRequest::new(
            &propagation_context,
            *equ,
            &t_eval,
            0.0,
            dt,
        )
        .with_events(true);
        let delta = lightyear_odeint_rs::integrate_final_checked(request)
            .expect("independent Lightyear reference must propagate");
        let mut baseline = [0.0; 6];
        equinoc_prop_from_impl(equ, dt, &mut baseline);
        for (baseline_component, delta_component) in baseline.iter_mut().zip(delta) {
            *baseline_component += delta_component;
        }
        assert!(all_finite(&baseline));
        let _ = eci; // documents that `equ` is derived from this exact source state.
        baseline
    }

    fn in_repo_hybrid_5x5_coefficients() -> (Vec<f64>, Vec<f64>, usize) {
        const ORDER: usize = 5;
        const SOURCE: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt"
        ));
        let stride = ORDER
            .checked_add(2)
            .expect("fixture stride must remain representable");
        let coefficient_count = stride
            .checked_mul(stride)
            .expect("fixture coefficient dimensions must remain representable");
        let mut c = vec![0.0; coefficient_count];
        let mut s = vec![0.0; coefficient_count];
        *c.get_mut(0)
            .expect("fixture coefficient storage must include C00") = 1.0;
        for line in SOURCE.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.split_whitespace();
            let degree = fields
                .next()
                .expect("DIR-R6 degree")
                .parse::<usize>()
                .expect("numeric DIR-R6 degree");
            let order = fields
                .next()
                .expect("DIR-R6 order")
                .parse::<usize>()
                .expect("numeric DIR-R6 order");
            let c_normalized = fields
                .next()
                .expect("DIR-R6 C coefficient")
                .parse::<f64>()
                .expect("numeric DIR-R6 C coefficient");
            let s_normalized = fields
                .next()
                .expect("DIR-R6 S coefficient")
                .parse::<f64>()
                .expect("numeric DIR-R6 S coefficient");
            if degree > ORDER || order > degree {
                continue;
            }
            let lower_factor = degree
                .checked_sub(order)
                .and_then(|value| value.checked_add(1))
                .expect("validated DIR-R6 degree/order must define a lower factor");
            let upper_factor = degree
                .checked_add(order)
                .expect("validated DIR-R6 degree/order must define an upper factor");
            let factorial_ratio = (lower_factor..=upper_factor)
                .map(|value| {
                    f64::from(
                        u32::try_from(value)
                            .expect("fixture factorial factors must fit in f64's exact u32 range"),
                    )
                })
                .product::<f64>();
            let symmetry = if order == 0 { 1.0 } else { 2.0 };
            let normalization_degree = degree
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .expect("validated DIR-R6 degree must define a normalization factor");
            let normalization_factor = f64::from(
                u32::try_from(normalization_degree)
                    .expect("fixture normalization factor must fit in f64's exact u32 range"),
            );
            let normalization = (factorial_ratio / (symmetry * normalization_factor)).sqrt();
            let index = degree
                .checked_mul(stride)
                .and_then(|value| value.checked_add(order))
                .expect("validated DIR-R6 coefficient index must remain representable");
            *c.get_mut(index)
                .expect("validated DIR-R6 C coefficient index must fit fixture storage") =
                c_normalized / normalization;
            *s.get_mut(index)
                .expect("validated DIR-R6 S coefficient index must fit fixture storage") =
                s_normalized / normalization;
        }
        for degree in 2..=ORDER {
            assert!(
                (0..=degree).any(|order| {
                    let index = degree
                        .checked_mul(stride)
                        .and_then(|value| value.checked_add(order))
                        .expect("validated fixture degree/order index must remain representable");
                    c.get(index)
                        .expect("validated fixture C index must fit coefficient storage")
                        .to_bits()
                        != 0.0_f64.to_bits()
                        || s.get(index)
                            .expect("validated fixture S index must fit coefficient storage")
                            .to_bits()
                            != 0.0_f64.to_bits()
                }),
                "in-repo gravity fixture must contain degree-{degree} authority"
            );
        }
        (c, s, stride)
    }

    fn assert_exact_hybrid_force_fixture(ctx: &PlanContext) {
        let config = ctx
            .force_config
            .as_ref()
            .expect("hybrid fixture requires force config");
        assert_eq!(config.sph_order, 5);
        assert_eq!(
            config.force_flags,
            lightyear_odeint_rs::types::ForceFlags::DRAG
                | lightyear_odeint_rs::types::ForceFlags::SRP
                | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
                | lightyear_odeint_rs::types::ForceFlags::MOON_GRAVITY,
            "hybrid fixture must enable exactly drag+SRP+Sun+Moon"
        );
        assert_eq!(config.atm_model, crate::types::HIGH_FIDELITY_ATM_MODEL);
    }

    fn dedicated_target_timeline_endpoint(
        state: [f64; 6],
        dt: f64,
        body_force: BodyForceConfig,
        ctx: &PlanContext,
    ) -> [f64; 6] {
        let mut config = **ctx
            .force_config
            .as_ref()
            .expect("target timeline reference requires force config");
        config.am_ratio = body_force.am_ratio;
        config.cd = body_force.cd;
        config.cr = body_force.cr;
        let gravity = lightyear_odeint_rs::ScalarGravityAssets::new(
            ctx.packed_coeffs
                .clone()
                .expect("target packed coefficients"),
        );
        let propagation_context = lightyear_odeint_rs::ScalarPropagationContext::new(
            ctx.epoch_jd,
            std::sync::Arc::new(config),
            gravity,
        );
        let session = lightyear_odeint_rs::LightyearSession::from_context(propagation_context);
        let mut endpoint = [f64::NAN; 6];
        let epoch_jd = [ctx.epoch_jd];
        let final_time_s = [dt];
        let request = lightyear_odeint_rs::VariableFinalBatchRequest {
            initial_eci_states: &state,
            epoch_jd: &epoch_jd,
            final_time_s: &final_time_s,
            t0_s: 0.0,
            ballistics: lightyear_odeint_rs::VariableFinalBallistics::default(),
        };
        session
            .integrate_variable_final_into(request, &mut endpoint)
            .expect("dedicated target timeline propagation");
        endpoint
    }

    #[test]
    fn hybrid_target_replay_matches_dedicated_ballistic_timeline_not_gravity_only() {
        let state = [6778.137, 0.0, 0.0, 0.0, 7.668_558, 0.0];
        let mut target_equ = [0.0; 6];
        assert!(eci_to_equinoctial(&state, &mut target_equ));
        let (c, s, stride) = in_repo_hybrid_5x5_coefficients();
        let packed = satpy_core::pack_gravity_coeffs(&c, &s, stride, 5)
            .expect("test gravity coefficients are valid");
        let target_force =
            BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 50.0, 2.2, 1.3);
        let ctx = PlanContext {
            epoch_jd: 2_460_000.5,
            tgt_eci: state,
            tgt_equ: target_equ,
            execution_policy: ExecutionPolicy {
                use_high_fidelity: true,
                require_high_fidelity: true,
                allow_parallel: false,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            target_propagation_authority: TargetPropagationAuthority::HighFidelity,
            target_body_force: target_force,
            force_config: Some(std::sync::Arc::new(
                lightyear_odeint_rs::types::ForceConfig {
                    sph_order: 5,
                    force_flags: lightyear_odeint_rs::types::ForceFlags::DRAG
                        | lightyear_odeint_rs::types::ForceFlags::SRP
                        | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
                        | lightyear_odeint_rs::types::ForceFlags::MOON_GRAVITY,
                    atm_model: crate::types::HIGH_FIDELITY_ATM_MODEL,
                    subtract_first_order: true,
                    dt_max: 30.0,
                    eps: 1.0e-9,
                    integrator_method: lightyear_odeint_rs::types::StepperMethod::Dopri5Compat,
                    target_propagation_mode: TargetPropagationAuthority::HighFidelity
                        .as_force_config_code(),
                    sun_pos: Some([149_600_000.0, 0.0, 0.0]),
                    moon_pos: Some([384_400.0, 0.0, 0.0]),
                    ..Default::default()
                },
            )),
            packed_coeffs: Some(std::sync::Arc::new(packed)),
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };
        assert_exact_hybrid_force_fixture(&ctx);
        let dt = 900.0;
        let expected = dedicated_target_timeline_endpoint(state, dt, target_force, &ctx);
        let replayed = propagate_high_fidelity_target_at_authoritative_offset_checked(&ctx, dt)
            .expect("authoritative high-fidelity target propagation");
        let gravity_only = independent_hf_segment_reference(
            &state,
            &target_equ,
            dt,
            ctx.epoch_jd,
            BodyForceConfig::gravity_only(BodyRole::DiagnosticTarget),
            &ctx,
        );

        let replay_delta = replayed
            .iter()
            .zip(expected)
            .map(|(actual, reference)| (actual - reference).abs())
            .fold(0.0_f64, f64::max);
        let gravity_delta = gravity_only
            .iter()
            .zip(expected)
            .map(|(actual, reference)| (actual - reference).abs())
            .fold(0.0_f64, f64::max);
        assert!(replay_delta < 1.0e-8, "replay delta={replay_delta:e}");
        assert!(
            gravity_delta > 1.0e-6,
            "hostile ballistic tuple must diverge from gravity-only; delta={gravity_delta:e}"
        );
    }

    fn create_hybrid_branch_grid_context() -> PlanContext {
        let mut ctx = create_leo_transfer_context();
        ctx.tgt_eci = ctx.dep_eci;
        ctx.tgt_equ = ctx.dep_equ;
        ctx.tgt_sma = ctx.dep_sma;
        ctx.tgt_period = ctx.dep_period;
        ctx.cache_target_orbit();
        ctx.cache_plane_angle();
        let (c, s, stride) = in_repo_hybrid_5x5_coefficients();
        let packed = satpy_core::pack_gravity_coeffs(&c, &s, stride, 5)
            .expect("test gravity coefficients are valid");
        ctx.epoch_jd = 2_460_000.5;
        ctx.max_time_s = 350.0;
        ctx.max_phase_dv = 10.0;
        ctx.max_transfer_dv = 100.0;
        ctx.max_revs = 0;
        ctx.distance_tol = 1.0e6;
        ctx.deployer_min_distance = 0.0;
        ctx.execution_policy = ExecutionPolicy {
            use_high_fidelity: true,
            require_high_fidelity: true,
            allow_parallel: false,
            allow_oxymoo_batch_parallel: false,
            allow_branch_expansion_parallel: false,
            allow_polish_parallel: false,
            allow_anchor_parallel: false,
            allow_deterministic_grid_parallel: false,
        };
        ctx.target_propagation_authority = TargetPropagationAuthority::HighFidelity;
        ctx.target_body_force =
            BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 50.0, 2.2, 1.3);
        ctx.force_config = Some(std::sync::Arc::new(
            lightyear_odeint_rs::types::ForceConfig {
                sph_order: 5,
                force_flags: lightyear_odeint_rs::types::ForceFlags::DRAG
                    | lightyear_odeint_rs::types::ForceFlags::SRP
                    | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
                    | lightyear_odeint_rs::types::ForceFlags::MOON_GRAVITY,
                atm_model: crate::types::HIGH_FIDELITY_ATM_MODEL,
                subtract_first_order: true,
                dt_max: 30.0,
                eps: 1.0e-9,
                integrator_method: lightyear_odeint_rs::types::StepperMethod::Dopri5Compat,
                target_propagation_mode: TargetPropagationAuthority::HighFidelity
                    .as_force_config_code(),
                am_ratio: 0.01,
                cd: 2.2,
                cr: 1.3,
                sun_pos: Some([149_600_000.0, 0.0, 0.0]),
                moon_pos: Some([384_400.0, 0.0, 0.0]),
                ..Default::default()
            },
        ));
        ctx.packed_coeffs = Some(std::sync::Arc::new(packed));
        ctx.j2_closure_settings.max_iterations = 0;
        ctx.search_depth.tof_sample_budget = 4;
        assert_exact_hybrid_force_fixture(&ctx);
        ctx
    }

    fn max_state_delta(left: &[f64; 6], right: &[f64; 6]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn hybrid_target_segment_rejects_non_authoritative_gravity_only_tuple() {
        let ctx = create_hybrid_branch_grid_context();
        let _rejection = propagate_high_fidelity_state_at_epoch_checked(
            &ctx.tgt_equ,
            300.0,
            ctx.epoch_jd,
            BodyForceConfig::gravity_only(BodyRole::DiagnosticTarget),
            &ctx,
        )
        .expect_err("HF target propagation must reject any tuple other than ctx.target_body_force");
    }

    #[test]
    fn hybrid_target_segment_rejects_signed_zero_tuple_bit_mismatch() {
        let mut ctx = create_hybrid_branch_grid_context();
        ctx.target_body_force.cr = 0.0;
        let hostile = BodyForceConfig {
            cr: -0.0,
            ..ctx.target_body_force
        };
        assert_eq!(
            hostile.cr.partial_cmp(&ctx.target_body_force.cr),
            Some(std::cmp::Ordering::Equal),
            "signed-zero tuple values must compare numerically equal before bit validation"
        );
        assert_ne!(hostile.cr.to_bits(), ctx.target_body_force.cr.to_bits());
        let _rejection = propagate_high_fidelity_state_at_epoch_checked(
            &ctx.tgt_equ,
            300.0,
            ctx.epoch_jd,
            hostile,
            &ctx,
        )
        .expect_err("sealed target tuple identity must distinguish +0.0 from -0.0");
    }

    #[test]
    fn segmented_hf_uses_advanced_source_epoch_not_base_epoch_reset() {
        let source = StampedEciState::new([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], 2_460_000.25);

        assert_ne!(
            propagation_epoch_for_segment(source.jd, 3600.0).to_bits(),
            propagation_epoch_for_segment(source.jd, 0.0).to_bits(),
            "second HF segment must not restart ephemeris time at base epoch"
        );
    }

    #[test]
    fn transfer_body_force_config_uses_explicit_tuple_not_dust() {
        let dust = BodyForceConfig::high_fidelity(BodyRole::Dust, 1.948, 2.2, 1.3);
        let ctx = PlanContext {
            execution_policy: ExecutionPolicy {
                use_high_fidelity: true,
                require_high_fidelity: true,
                allow_parallel: false,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            force_config: Some(std::sync::Arc::new(
                lightyear_odeint_rs::types::ForceConfig {
                    am_ratio: 0.01,
                    cd: 2.6,
                    cr: 1.1,
                    ..Default::default()
                },
            )),
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };
        let transfer = ctx.transfer_body_force();

        assert_eq!(dust.role, BodyRole::Dust);
        assert_eq!(dust.fidelity, PropagationFidelity::HighFidelity);
        assert_eq!(transfer.role, BodyRole::TransferVehicle);
        assert_eq!(transfer.fidelity, PropagationFidelity::HighFidelity);
        assert_eq!(transfer.am_ratio.to_bits(), 0.01_f64.to_bits());
        assert_eq!(transfer.cd.to_bits(), 2.6_f64.to_bits());
        assert_eq!(transfer.cr.to_bits(), 1.1_f64.to_bits());
        assert_ne!(transfer.am_ratio.to_bits(), dust.am_ratio.to_bits());
    }

    #[test]
    fn transfer_body_force_preserves_exact_hybrid_bundle() {
        let ctx = PlanContext {
            execution_policy: ExecutionPolicy {
                use_high_fidelity: true,
                require_high_fidelity: true,
                allow_parallel: false,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            force_config: Some(std::sync::Arc::new(
                lightyear_odeint_rs::types::ForceConfig {
                    force_flags: lightyear_odeint_rs::types::ForceFlags::DRAG
                        | lightyear_odeint_rs::types::ForceFlags::SRP
                        | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
                        | lightyear_odeint_rs::types::ForceFlags::MOON_GRAVITY,
                    sph_order: 5,
                    am_ratio: 0.01,
                    cd: 2.6,
                    cr: 1.1,
                    sun_pos: Some([149_600_000.0, 0.0, 0.0]),
                    moon_pos: Some([384_400.0, 0.0, 0.0]),
                    ..Default::default()
                },
            )),
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };

        let config = stamped_body_force_config(&ctx, 2_460_000.5, 60.0, ctx.transfer_body_force())
            .expect("explicit positions require no catalogue coverage");

        assert_eq!(config.sph_order, 5);
        assert_eq!(
            config.force_flags,
            lightyear_odeint_rs::types::ForceFlags::DRAG
                | lightyear_odeint_rs::types::ForceFlags::SRP
                | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
                | lightyear_odeint_rs::types::ForceFlags::MOON_GRAVITY
        );
        assert_eq!(config.am_ratio.to_bits(), 0.01_f64.to_bits());
        assert_eq!(config.cd.to_bits(), 2.6_f64.to_bits());
        assert_eq!(config.cr.to_bits(), 1.1_f64.to_bits());
    }

    #[test]
    fn transfer_hf_final_failure_is_not_retried_through_sampled_path() {
        let period_s = 4_920.0;
        let apogee_km = 7_000.0;
        let semi_major_km = (MU * (period_s / (2.0 * std::f64::consts::PI)).powi(2)).cbrt();
        let apogee_speed_km_s = (MU * (2.0 / apogee_km - 1.0 / semi_major_km)).sqrt();
        let state = [apogee_km, 0.0, 0.0, 0.0, apogee_speed_km_s, 0.0];
        let mut equ = [0.0; 6];
        assert!(eci_to_equinoctial(&state, &mut equ));

        let stride: usize = 7;
        let coefficient_count = stride
            .checked_mul(stride)
            .expect("test gravity coefficient dimensions must remain representable");
        let mut c_values = vec![0.0; coefficient_count];
        *c_values
            .get_mut(0)
            .expect("test gravity coefficient storage must include C00") = 1.0;
        let j2_index = 2_usize
            .checked_mul(stride)
            .expect("test gravity J2 index must remain representable");
        *c_values
            .get_mut(j2_index)
            .expect("test gravity coefficient storage must include C20") = -1.082_63e-3;
        let s_values = vec![0.0; coefficient_count];
        let packed = satpy_core::pack_gravity_coeffs(&c_values, &s_values, stride, 5)
            .expect("test gravity coefficients are valid");
        let ctx = PlanContext {
            epoch_jd: 2_460_310.5,
            execution_policy: ExecutionPolicy {
                use_high_fidelity: true,
                require_high_fidelity: true,
                allow_parallel: false,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            force_config: Some(std::sync::Arc::new(
                lightyear_odeint_rs::types::ForceConfig {
                    sph_order: 5,
                    force_flags: 0,
                    subtract_first_order: true,
                    dt_max: 60.0,
                    eps: 1.0e-8,
                    integrator_method: lightyear_odeint_rs::types::StepperMethod::Vern9,
                    ..Default::default()
                },
            )),
            packed_coeffs: Some(std::sync::Arc::new(packed)),
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };
        let _failure = propagate_high_fidelity_state_at_epoch_checked(
            &equ,
            period_s,
            ctx.epoch_jd,
            BodyForceConfig::gravity_only(BodyRole::TransferVehicle),
            &ctx,
        )
        .expect_err("terminal Ground event must fail closed without a sampled retry");
    }

    #[test]
    fn transfer_hf_eclipse_failure_preserves_exact_typed_cause() {
        let ctx = create_hybrid_branch_grid_context();
        let radius_km = 60_000.0;
        let speed_km_s = (MU / radius_km).sqrt();
        let state = [radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0];
        let mut equ = [0.0; 6];
        assert!(eci_to_equinoctial(&state, &mut equ));

        let result = propagate_high_fidelity_state_at_epoch_checked(
            &equ,
            60.0,
            ctx.epoch_jd,
            ctx.target_body_force,
            &ctx,
        );
        assert!(matches!(
            result,
            Err(TransferPropagationFailure::Final(
                lightyear_odeint_rs::integrator::FinalPropagationFailure::Eclipse(
                    lightyear_odeint_rs::EclipseError::Envelope,
                ),
            ))
        ));
    }

    #[test]
    fn stamped_body_force_rejects_full_arc_outside_ephemeris_before_rhs() {
        let flags = lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY;
        let ephem = lightyear_odeint_rs::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("sun ephemeris must load");
        let (_, end) = ephem
            .get(lightyear_odeint_rs::precomputed_ephem::Body::Sun)
            .expect("sun catalogue")
            .jd_range();
        let ctx = PlanContext {
            force_config: Some(std::sync::Arc::new(
                lightyear_odeint_rs::types::ForceConfig {
                    force_flags: flags,
                    ..Default::default()
                },
            )),
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };

        let outside_range_bits = end
            .to_bits()
            .checked_add(1)
            .expect("ephemeris endpoint ULP must remain representable");
        let error = stamped_body_force_config(
            &ctx,
            f64::from_bits(outside_range_bits),
            0.0,
            BodyForceConfig::gravity_only(BodyRole::DiagnosticTarget),
        )
        .expect_err("out-of-range absolute endpoint must fail before RHS");
        assert!(matches!(
            error,
            TransferPropagationFailure::Ephemeris(
                lightyear_odeint_rs::precomputed_ephem::EphemerisCoverageError::OutsideRange { .. }
            )
        ));
    }

    #[test]
    fn hybrid_multi_tof_unsorted_grid_matches_scalar_dynamic_force_and_original_order() {
        HF_MULTI_TOF_TEST_REBASE_OBSERVED.with(|observed| *observed.borrow_mut() = false);
        let mut ctx = create_hybrid_branch_grid_context();
        ctx.target_body_force.am_ratio = 0.01;
        let config = std::sync::Arc::make_mut(
            ctx.force_config
                .as_mut()
                .expect("hybrid fixture force config"),
        );
        config.sun_pos = None;
        config.moon_pos = None;
        assert_ne!(
            config.required_dynamic_ephemeris_flags()
                & lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY,
            0,
            "SRP fixture must resolve Sun position dynamically at each RHS epoch"
        );
        assert_exact_hybrid_force_fixture(&ctx);
        stamped_body_force_config(&ctx, ctx.epoch_jd, 900.0, ctx.target_body_force)
            .expect("dynamic-SRP fixture force config");

        let offsets = [86_400.0, 3_600.0, 21_600.0];
        let mut batched = [[f64::NAN; 6]; 3];
        let result = propagate_high_fidelity_target_multi_tof_checked(&ctx, &offsets, &mut batched);
        result.expect("multi-TOF propagation must succeed");

        let mut max_position_delta_km = 0.0_f64;
        let mut max_velocity_delta_km_s = 0.0_f64;
        for (offset, batched_state) in offsets.iter().copied().zip(&batched) {
            let scalar =
                propagate_high_fidelity_target_at_authoritative_offset_checked(&ctx, offset)
                    .expect("authoritative high-fidelity target propagation");
            let (batched_position, batched_velocity) = batched_state.split_at(3);
            let (scalar_position, scalar_velocity) = scalar.split_at(3);
            let position_delta = batched_position
                .iter()
                .zip(scalar_position)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            let velocity_delta = batched_velocity
                .iter()
                .zip(scalar_velocity)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            max_position_delta_km = max_position_delta_km.max(position_delta);
            max_velocity_delta_km_s = max_velocity_delta_km_s.max(velocity_delta);
        }
        assert!(
            max_position_delta_km < 2.0e-4 && max_velocity_delta_km_s < 2.0e-7,
            "day-arc batched/scalar mismatch: max_position_delta_km={max_position_delta_km:e}, max_velocity_delta_km_s={max_velocity_delta_km_s:e}"
        );
        let [first, second, _] = &batched;
        assert!(
            max_state_delta(first, second) > 1.0,
            "hostile unsorted outputs must map back to original TOF order"
        );
        assert!(
            HF_MULTI_TOF_TEST_REBASE_OBSERVED.with(|observed| *observed.borrow()),
            "day arc must exercise coordinator PerturbDeviation rebase"
        );
    }

    #[test]
    fn hybrid_multi_tof_duplicate_grid_reuses_exact_state_in_original_order() {
        let ctx = create_hybrid_branch_grid_context();
        let offsets = [900.0, 300.0, 900.0, 600.0];
        let mut outputs = [[f64::NAN; 6]; 4];

        propagate_high_fidelity_target_multi_tof_checked(&ctx, &offsets, &mut outputs)
            .expect("duplicate multi-TOF propagation must succeed");
        let [first, second, duplicate, third] = &outputs;
        for (first_component, duplicate_component) in first.iter().zip(duplicate) {
            assert_eq!(
                first_component.to_bits(),
                duplicate_component.to_bits(),
                "duplicate TOF must map one exact propagated state"
            );
        }
        assert!(max_state_delta(first, second) > 1.0);
        assert!(max_state_delta(duplicate, third) > 1.0);
    }

    #[test]
    fn hf_grid_telemetry_counts_requested_and_unique_propagation() {
        let ctx = create_hybrid_branch_grid_context();
        let offsets = [900.0, 300.0, 900.0, 600.0];
        let mut outputs = [[f64::NAN; 6]; 4];
        let before = hf_propagation_telemetry_snapshot();

        crate::types::with_verified_superset_deep_telemetry_for_test(true, || {
            propagate_high_fidelity_target_multi_tof_checked(&ctx, &offsets, &mut outputs)
                .expect("telemetry-enabled multi-TOF propagation must succeed");
        });

        let delta = hf_propagation_telemetry_snapshot()
            .delta_since(before)
            .expect("monotonic HF telemetry must have a delta");
        assert_eq!(delta.target_grid_call_count, 1);
        assert_eq!(delta.target_grid_requested_state_count, 4);
        assert_eq!(delta.target_grid_unique_attempted_state_count, 3);
        assert!(delta.target_grid_s.is_finite() && delta.target_grid_s >= 0.0);
    }

    #[test]
    fn deep_telemetry_disabled_hf_grid_execution_is_bit_identical_and_silent() {
        let ctx = create_hybrid_branch_grid_context();
        let offsets = [900.0, 300.0, 900.0, 600.0];
        let mut telemetry_off_outputs = [[f64::NAN; 6]; 4];
        let before_off = hf_propagation_telemetry_snapshot();

        crate::types::with_verified_superset_deep_telemetry_for_test(false, || {
            propagate_high_fidelity_target_multi_tof_checked(
                &ctx,
                &offsets,
                &mut telemetry_off_outputs,
            )
            .expect("telemetry-disabled multi-TOF propagation must succeed");
        });

        assert_eq!(
            hf_propagation_telemetry_snapshot()
                .delta_since(before_off)
                .expect("disabled HF telemetry snapshot must not underflow"),
            HfPropagationTelemetry::default(),
            "disabled telemetry must add no HF grid diagnostics"
        );

        let mut telemetry_on_outputs = [[f64::NAN; 6]; 4];
        crate::types::with_verified_superset_deep_telemetry_for_test(true, || {
            propagate_high_fidelity_target_multi_tof_checked(
                &ctx,
                &offsets,
                &mut telemetry_on_outputs,
            )
            .expect("telemetry-enabled multi-TOF propagation must succeed");
        });

        for (disabled, enabled) in telemetry_off_outputs.iter().zip(telemetry_on_outputs) {
            for (disabled_component, enabled_component) in disabled.iter().zip(enabled) {
                assert_eq!(disabled_component.to_bits(), enabled_component.to_bits());
            }
        }
    }

    #[test]
    fn hybrid_multi_tof_failure_is_atomic_without_scalar_fallback() {
        let ctx = create_hybrid_branch_grid_context();
        let offsets = [300.0, f64::NAN, 600.0];
        let sentinel = [[-12345.0; 6]; 3];
        let mut outputs = sentinel;

        let _failure =
            propagate_high_fidelity_target_multi_tof_checked(&ctx, &offsets, &mut outputs)
                .expect_err("nonfinite multi-TOF input must fail atomically");
        for (actual_row, sentinel_row) in outputs.iter().zip(sentinel) {
            for (actual, sentinel_value) in actual_row.iter().zip(sentinel_row) {
                assert_eq!(
                    actual.to_bits(),
                    sentinel_value.to_bits(),
                    "failed batch must not expose partial rows"
                );
            }
        }
    }

    #[test]
    fn hybrid_multi_tof_rejects_noncanonical_signed_zero_grid_atomically() {
        let ctx = create_hybrid_branch_grid_context();
        let sentinel = [[-12345.0; 6]; 2];
        for offsets in [[0.0, 300.0], [-0.0, 300.0]] {
            let mut outputs = sentinel;
            let _failure =
                propagate_high_fidelity_target_multi_tof_checked(&ctx, &offsets, &mut outputs)
                    .expect_err("noncanonical signed-zero multi-TOF grid must fail atomically");
            for (actual_row, sentinel_row) in outputs.iter().zip(sentinel) {
                for (actual, sentinel_value) in actual_row.iter().zip(sentinel_row) {
                    assert_eq!(actual.to_bits(), sentinel_value.to_bits());
                }
            }
        }
    }

    #[test]
    fn candidate_search_rejects_high_fidelity_before_any_rhs_work() {
        HF_MULTI_TOF_TEST_CALLS.with(|counts| *counts.borrow_mut() = (0, 0, true));
        let mut ctx = create_hybrid_branch_grid_context();
        ctx.max_time_s = 10_000.0;
        ctx.search_depth.tof_sample_budget = 16;

        let error = evaluate_plan(&[0.0, 1.0, 0.0], &ctx, true)
            .expect_err("unsupported high-fidelity candidate search must be typed");
        let (multi_calls, scalar_grid_calls, _) =
            HF_MULTI_TOF_TEST_CALLS.with(|counts| *counts.borrow());
        HF_MULTI_TOF_TEST_CALLS.with(|counts| counts.borrow_mut().2 = false);
        assert_eq!(
            error,
            crate::types::InvalidTargetPropagationAuthorityCode::CandidateSearch(
                crate::types::CandidateSearchAuthorityError::UnsupportedHighFidelityTransferSearch,
            ),
            "unsupported HF candidate search must not become an invalid plan"
        );
        assert_eq!(
            multi_calls, 0,
            "candidate search must reject before entering an HF target grid"
        );
        assert_eq!(
            scalar_grid_calls, 0,
            "candidate search must reject before any scalar fallback"
        );
    }

    #[test]
    fn evaluate_plan_rejects_invalid_target_body_force() {
        // Asserts the typed error only; unlike
        // `candidate_search_rejects_high_fidelity_before_any_rhs_work`, no
        // call counter proves the rejection precedes the search.
        let mut ctx = create_leo_transfer_context();
        ctx.target_body_force = BodyForceConfig::gravity_only(BodyRole::DiagnosticTarget);

        let error = evaluate_plan(&[0.0, 1.0, 0.0], &ctx, true)
            .expect_err("invalid target body force must be typed");

        assert_eq!(
            error,
            crate::types::InvalidTargetPropagationAuthorityCode::InvalidTargetBodyForce {
                authority: TargetPropagationAuthority::MfJ2,
            }
        );
    }

    #[test]
    fn evaluate_plan_maps_finite_unrepresentable_tof_grid_span_to_authority_overflow() {
        let mut ctx = create_leo_transfer_context();
        let (dep_eci, dep_equ, tgt_eci, tgt_equ, epoch_jd) = (
            ctx.dep_eci,
            ctx.dep_equ,
            ctx.tgt_eci,
            ctx.tgt_equ,
            ctx.epoch_jd,
        );
        // Keep the real LEO state and candidate path while invalidating its
        // cached orbit bounds through the production reset contract. The two
        // search settings below let the finite span reach grid conversion.
        ctx.reset(dep_eci, dep_equ, tgt_eci, tgt_equ, epoch_jd);
        ctx.max_time_s = f64::MAX;
        ctx.revolution_cap = 0.0;

        let error = evaluate_plan(&[0.0, 1.0, 0.0], &ctx, false)
            .expect_err("finite unrepresentable grid span must be typed");

        assert_eq!(
            error,
            crate::types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow
        );
    }

    #[test]
    fn candidate_search_branch_wrapper_rejects_high_fidelity_before_any_rhs_work() {
        HF_MULTI_TOF_TEST_CALLS.with(|counts| *counts.borrow_mut() = (0, 0, false));
        let mut ctx = create_hybrid_branch_grid_context();
        ctx.max_time_s = 10_000.0;
        ctx.search_depth.tof_sample_budget = 16;
        let mut scratch = crate::lambert::VariableR2LambertScratch::default();

        let front = evaluate_plan_branches_with_scratch(&[0.0, 1.0, 0.0], &ctx, true, &mut scratch)
            .expect("unsupported-search branch fixture must not overflow diagnostics");
        let (multi_calls, scalar_grid_calls, _) =
            HF_MULTI_TOF_TEST_CALLS.with(|counts| *counts.borrow());
        // Zeroed counters alone are also what a fixture that never reached the
        // HF path for an UNRELATED reason would produce; the empty front is
        // what says the wrapper rejected this call itself (the reference
        // sibling below asserts the same shape).
        assert!(
            front.is_empty(),
            "rejected HF candidate search must return an empty front"
        );
        assert_eq!(
            multi_calls, 0,
            "branch wrapper must reject before entering an HF target grid"
        );
        assert_eq!(
            scalar_grid_calls, 0,
            "branch wrapper must reject before any scalar fallback"
        );
    }

    #[test]
    fn candidate_search_preparation_rejects_unsupported_high_fidelity_before_rhs_work() {
        HF_MULTI_TOF_TEST_CALLS.with(|counts| *counts.borrow_mut() = (0, 0, true));
        let mut ctx = create_hybrid_branch_grid_context();
        ctx.max_time_s = 10_000.0;
        ctx.search_depth.tof_sample_budget = 16;

        let prepared = prepare_branch_shared_work(&[0.0, 1.0, 0.0], &ctx, true)
            .expect("unsupported-search preparation must not overflow diagnostics");
        let (multi_calls, scalar_grid_calls, _) =
            HF_MULTI_TOF_TEST_CALLS.with(|counts| *counts.borrow());
        HF_MULTI_TOF_TEST_CALLS.with(|counts| counts.borrow_mut().2 = false);
        assert!(
            prepared.is_none(),
            "unsupported HF candidate search must prepare no work"
        );
        assert_eq!(
            multi_calls, 0,
            "unsupported branch search must enter no HF target grid"
        );
        assert_eq!(
            scalar_grid_calls, 0,
            "unsupported branch search must enter no scalar fallback"
        );
    }

    #[test]
    fn candidate_search_reference_wrapper_rejects_high_fidelity() {
        let ctx = create_hybrid_branch_grid_context();

        assert!(
            evaluate_plan_branches_reference(&[0.0, 1.0, 0.0], &ctx, true)
                .expect("reference unsupported-search fixture must not overflow diagnostics")
                .is_empty(),
            "test reference must not run high-fidelity candidate propagation"
        );
    }

    #[test]
    fn hybrid_multi_tof_terminal_event_is_atomic() {
        let period_s = 4_920.0;
        let apogee_km = 7_000.0;
        let semi_major_km = (MU * (period_s / (2.0 * std::f64::consts::PI)).powi(2)).cbrt();
        let apogee_speed_km_s = (MU * (2.0 / apogee_km - 1.0 / semi_major_km)).sqrt();
        let state = [apogee_km, 0.0, 0.0, 0.0, apogee_speed_km_s, 0.0];
        let mut ctx = create_hybrid_branch_grid_context();
        ctx.tgt_eci = state;
        assert!(eci_to_equinoctial(&state, &mut ctx.tgt_equ));
        let offsets = [600.0, period_s];
        let sentinel = [[-12345.0; 6]; 2];
        let mut outputs = sentinel;

        let _terminal_event =
            propagate_high_fidelity_target_multi_tof_checked(&ctx, &offsets, &mut outputs)
                .expect_err("terminal event must reject the target batch");
        assert_eq!(
            outputs, sentinel,
            "terminal batch must expose no partial rows"
        );
    }

    fn segmented_hf_source_epoch_fixture() -> (PlanContext, [f64; 6], [f64; 6], BodyForceConfig) {
        let state0 = [6878.137, 0.0, 0.0, 0.0, 7.612_608, 0.0];
        let mut equ0 = [0.0; 6];
        assert!(eci_to_equinoctial(&state0, &mut equ0));

        // Minimal gravity table plus a time-dependent solar-gravity force.
        // `with_ephemeris` resolves the solar position at each segment's
        // midpoint; the test therefore exercises the same HF epoch input as
        // production rather than only testing timestamp arithmetic.
        let c = std::sync::Arc::new(vec![1.0, 0.0, 0.0, 0.0]);
        let s = std::sync::Arc::new(vec![0.0; 4]);
        let packed = std::sync::Arc::new(
            satpy_core::pack_gravity_coeffs(&c, &s, 2, 0)
                .expect("test gravity coefficients are valid"),
        );
        let ctx = PlanContext {
            epoch_jd: 2_460_000.25,
            execution_policy: ExecutionPolicy {
                use_high_fidelity: true,
                require_high_fidelity: true,
                allow_parallel: false,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            force_config: Some(std::sync::Arc::new(
                lightyear_odeint_rs::types::ForceConfig {
                    force_flags: lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
                        | lightyear_odeint_rs::types::ForceFlags::SRP,
                    sph_order: 0,
                    am_ratio: 0.01,
                    cd: 2.6,
                    cr: 1.1,
                    dt_max: 5.0,
                    eps: 1.0e-8,
                    ..Default::default()
                },
            )),
            packed_coeffs: Some(packed),
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };
        let dust = BodyForceConfig::high_fidelity(BodyRole::Dust, 1.948, 2.2, 1.3);

        (ctx, state0, equ0, dust)
    }

    #[test]
    fn segmented_hf_force_enabled_propagation_uses_each_segment_source_epoch() {
        let (ctx, state0, equ0, dust) = segmented_hf_source_epoch_fixture();
        let first_dt = SEC_PER_DAY;
        let second_dt = SEC_PER_DAY;

        let after_first = propagate_high_fidelity_state_at_epoch_checked(
            &equ0,
            first_dt,
            ctx.epoch_jd,
            dust,
            &ctx,
        )
        .expect("first dust segment");
        let mut equ_after_first = [0.0; 6];
        assert!(eci_to_equinoctial(&after_first, &mut equ_after_first));

        let stamped_second = propagate_high_fidelity_state_at_epoch_checked(
            &equ_after_first,
            second_dt,
            propagation_epoch_for_segment(ctx.epoch_jd, first_dt),
            dust,
            &ctx,
        )
        .expect("source-stamped second dust segment");
        let reset_second = propagate_high_fidelity_state_at_epoch_checked(
            &equ_after_first,
            second_dt,
            ctx.epoch_jd,
            dust,
            &ctx,
        )
        .expect("reset-source second dust segment");

        let reference_first =
            independent_hf_segment_reference(&state0, &equ0, first_dt, ctx.epoch_jd, dust, &ctx);
        let mut reference_first_equ = [0.0; 6];
        assert!(eci_to_equinoctial(
            &reference_first,
            &mut reference_first_equ
        ));
        let reference_second = independent_hf_segment_reference(
            &reference_first,
            &reference_first_equ,
            second_dt,
            propagation_epoch_for_segment(ctx.epoch_jd, first_dt),
            dust,
            &ctx,
        );

        let stamped_error = vec_distance(&stamped_second, &reference_second);
        let reset_delta = vec_distance(&reset_second, &stamped_second);
        assert!(
            stamped_error < 1.0e-9,
            "stamped propagation must agree with independent source-stamped reference: {stamped_error} km"
        );
        assert!(
            reset_delta > 1.0e-4,
            "old source reset must materially diverge from stamped replay: {reset_delta} km"
        );

        let transfer = propagate_high_fidelity_state_at_epoch_checked(
            &equ0,
            first_dt + second_dt,
            ctx.epoch_jd,
            ctx.transfer_body_force(),
            &ctx,
        )
        .expect("one-segment transfer propagation");
        let dust_one_segment = independent_hf_segment_reference(
            &state0,
            &equ0,
            first_dt + second_dt,
            ctx.epoch_jd,
            dust,
            &ctx,
        );
        assert!(
            vec_distance(&transfer, &dust_one_segment) > 1.0e-5,
            "sealed transfer tuple must not inherit dust force coefficients"
        );
    }

    /// Helper to create a typical GEO transfer test case
    fn create_geo_transfer_context() -> PlanContext {
        // LEO deployer at 400km altitude
        let leo_alt = 400.0;
        let leo_r = RE + leo_alt;
        let leo_v = (MU / leo_r).sqrt();
        let dep_eci = [leo_r, 0.0, 0.0, 0.0, leo_v, 0.0];

        // GEO target at 35,786km altitude
        let geo_alt = 35786.0;
        let geo_r = RE + geo_alt;
        let geo_v = (MU / geo_r).sqrt();
        let tgt_eci = [geo_r, 0.0, 0.0, 0.0, geo_v, 0.0];

        let mut dep_equ = [0.0; 6];
        eci_to_equinoctial(&dep_eci, &mut dep_equ);

        let mut tgt_equ = [0.0; 6];
        eci_to_equinoctial(&tgt_eci, &mut tgt_equ);

        PlanContext {
            dep_eci,
            dep_equ,
            tgt_eci,
            tgt_equ,
            dep_sma: leo_r,
            tgt_sma: geo_r,
            dep_period: std::f64::consts::TAU * (leo_r.powi(3) / MU).sqrt(),
            tgt_period: std::f64::consts::TAU * (geo_r.powi(3) / MU).sqrt(),
            max_time_s: 86400.0 * 5.0, // 5 days
            max_phase_dv: 0.5,
            max_transfer_dv: 10.0,
            min_perigee: RE + 200.0,
            max_apogee: geo_r + 1000.0,
            distance_tol: 0.010,
            deployer_min_distance: 0.12,
            max_revs: 2,
            revolution_cap: 100.0,
            tof_penalty_weight: 0.01,
            epoch_jd: 2_451_545.0,
            sampling_mode: SamplingMode::Fast,
            execution_policy: ExecutionPolicy {
                use_high_fidelity: false,
                require_high_fidelity: false,
                allow_parallel: true,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            tgt_orbit_valid: true,
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        }
    }

    fn create_leo_transfer_context() -> PlanContext {
        let dep_r = RE + 400.0;
        let dep_v = (MU / dep_r).sqrt();
        let dep_eci = [dep_r, 0.0, 0.0, 0.0, dep_v, 0.0];
        let tgt_r = RE + 500.0;
        let tgt_v = (MU / tgt_r).sqrt();
        let tgt_eci = [tgt_r, 0.0, 0.0, 0.0, tgt_v, 0.0];
        let mut dep_equ = [0.0; 6];
        eci_to_equinoctial(&dep_eci, &mut dep_equ);
        let mut tgt_equ = [0.0; 6];
        eci_to_equinoctial(&tgt_eci, &mut tgt_equ);
        let mut ctx = PlanContext {
            dep_eci,
            dep_equ,
            tgt_eci,
            tgt_equ,
            dep_sma: dep_r,
            tgt_sma: tgt_r,
            dep_period: std::f64::consts::TAU * (dep_r.powi(3) / MU).sqrt(),
            tgt_period: std::f64::consts::TAU * (tgt_r.powi(3) / MU).sqrt(),
            max_time_s: 86400.0,
            max_phase_dv: 1.0,
            max_transfer_dv: 2.0,
            min_perigee: RE + 200.0,
            max_apogee: 50000.0,
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            max_revs: 2,
            revolution_cap: 100.0,
            tof_penalty_weight: 0.1,
            epoch_jd: 2_451_545.0,
            sampling_mode: SamplingMode::Fast,
            execution_policy: ExecutionPolicy {
                use_high_fidelity: false,
                require_high_fidelity: false,
                allow_parallel: true,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };
        ctx.cache_target_orbit();
        ctx.cache_deployer_orbit();
        ctx.cache_plane_angle();
        ctx
    }

    #[test]
    fn direct_and_prepared_evaluators_stamp_immutable_replay_provenance() {
        let ctx = create_leo_transfer_context();
        let x = [0.0, 1.0, 0.0];
        let direct = evaluate_plan(&x, &ctx, false)
            .expect("direct provenance fixture must not overflow diagnostics");
        let prepared = evaluate_plan_from_phase_with_lambert_scratch(
            &x,
            &ctx,
            false,
            0.0,
            0.0,
            compute_dep_period(&ctx),
            &ctx.dep_eci,
            None,
            None,
        )
        .expect("prepared provenance fixture must not overflow diagnostics");
        for plan in [&direct, &prepared] {
            assert_eq!(
                plan.replay_provenance
                    .launch_pre_impulse_state
                    .map(f64::to_bits),
                ctx.dep_eci.map(f64::to_bits)
            );
            assert_eq!(
                plan.replay_provenance.base_epoch_jd.to_bits(),
                ctx.epoch_jd.to_bits()
            );
            assert_eq!(
                plan.replay_provenance.max_phase_dv.to_bits(),
                ctx.max_phase_dv.to_bits()
            );
            assert_eq!(
                plan.replay_provenance.max_transfer_dv.to_bits(),
                ctx.max_transfer_dv.to_bits()
            );
            assert_eq!(plan.replay_provenance.max_revs, ctx.max_revs);
        }
    }

    fn create_quarter_arc_transfer_context() -> PlanContext {
        let dep_r = 7000.0;
        let dep_v = (MU / dep_r).sqrt();
        let dep_eci = [dep_r, 0.0, 0.0, 0.0, dep_v, 0.0];
        let tgt_r = 7500.0;
        let tgt_v = (MU / tgt_r).sqrt();
        let tgt_eci = [0.0, tgt_r, 0.0, -tgt_v, 0.0, 0.0];
        let mut dep_equ = [0.0; 6];
        eci_to_equinoctial(&dep_eci, &mut dep_equ);
        let mut tgt_equ = [0.0; 6];
        eci_to_equinoctial(&tgt_eci, &mut tgt_equ);
        let mut ctx = PlanContext {
            dep_eci,
            dep_equ,
            tgt_eci,
            tgt_equ,
            dep_sma: dep_r,
            tgt_sma: tgt_r,
            dep_period: std::f64::consts::TAU * (dep_r.powi(3) / MU).sqrt(),
            tgt_period: std::f64::consts::TAU * (tgt_r.powi(3) / MU).sqrt(),
            max_time_s: 86400.0 * 2.0,
            max_phase_dv: 10.0,
            max_transfer_dv: 20.0,
            min_perigee: RE + 200.0,
            max_apogee: 50000.0,
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            max_revs: 2,
            revolution_cap: 100.0,
            tof_penalty_weight: 0.1,
            epoch_jd: 2_451_545.0,
            sampling_mode: SamplingMode::Fast,
            execution_policy: ExecutionPolicy {
                use_high_fidelity: false,
                require_high_fidelity: false,
                allow_parallel: true,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };
        ctx.cache_target_orbit();
        ctx.cache_deployer_orbit();
        ctx.cache_plane_angle();
        ctx
    }

    #[test]
    fn transfer_timing_window_rejects_phase_wait_at_budget() {
        let ctx = create_geo_transfer_context();
        let budget = ctx.intercept_time_budget_s();

        let err = transfer_timing_window(&ctx, budget, 0.0).unwrap_err();

        assert_eq!(err, TimingFailureToken::InterceptInsufficientLead);
        assert_eq!(err.as_str(), "solver_intercept_insufficient_lead");
    }

    #[test]
    fn transfer_timing_window_rejects_phase_wait_beyond_budget() {
        let ctx = create_geo_transfer_context();
        let budget = ctx.intercept_time_budget_s();

        let err = transfer_timing_window(&ctx, budget + 1.0, 0.0).unwrap_err();

        assert_eq!(err, TimingFailureToken::InterceptTransferTimeExceeded);
        assert_eq!(err.as_str(), "solver_intercept_transfer_time_exceeded");
    }

    #[test]
    fn transfer_timing_window_allows_exact_minimum_tof_overlap() {
        let ctx = create_geo_transfer_context();
        let budget = ctx.intercept_time_budget_s();

        let headroom = transfer_timing_window(&ctx, budget - MIN_TOF, 0.0).unwrap();

        assert_eq!(headroom.to_bits(), MIN_TOF.to_bits());
    }

    fn expand_tof_sample_ladder_for_test(
        ctx: &PlanContext,
        phase_sma: f64,
        dep_at_release: [f64; 6],
        interval: AdmissibleTofInterval,
        branch_route: bool,
    ) -> Result<([f64; MAX_TOF_SAMPLES], usize), EvaluationArithmeticOverflow> {
        let tof_lower = interval.lower;
        let tof_upper = interval.upper;
        let span = interval.span;
        let is_simple_transfer = false;
        tof_grid_sample_count(span, is_simple_transfer)?.expect("finite hard interval");
        let sample_count = ctx.search_depth.clamped_tof_budget();
        let tof_budget = ctx.search_depth.clamped_tof_budget();
        let mut tof_samples = [0.0; MAX_TOF_SAMPLES];
        let mut tof_sample_n = 0;
        let mut hohmann_tof = 0.0;
        let tgt_period = ctx.tgt_period;
        let plane_angle_cached = 0.0;

        if branch_route {
            tof_sample_ladder!(
                ctx = ctx,
                phase_sma = phase_sma,
                dep_at_release = dep_at_release,
                plane_angle_cached = plane_angle_cached,
                tof_lower = tof_lower,
                tof_upper = tof_upper,
                span = span,
                sample_count = sample_count,
                is_simple_transfer = is_simple_transfer,
                tof_budget = tof_budget,
                tof_samples = tof_samples,
                tof_sample_n = tof_sample_n,
                hohmann_tof = hohmann_tof,
                tgt_period = tgt_period,
                period = phase_period,
                period_setup = [
                    let phase_period =
                        2.0 * std::f64::consts::PI * ((phase_sma.powi(3)) / MU).sqrt();
                ],
                transfer_sma_hohmann = transfer_sma_hohmann,
                transfer_period = transfer_period,
                transfer_period_setup = [
                    let transfer_period = 2.0
                        * std::f64::consts::PI
                        * ((transfer_sma_hohmann.powi(3)) / MU).sqrt();
                ],
                multi_rev_var = m,
                multi_rev_setup = [],
                multi_rev_sample = [hohmann_tof + f64::from(m) * phase_period],
                rev_entry_var = n,
                rev_entry_setup = [],
                rev_entry_sample = [hohmann_tof + (f64::from(n) - 0.5) * phase_period],
                bail = Err(EvaluationArithmeticOverflow),
            );
        } else {
            tof_sample_ladder!(
                ctx = ctx,
                phase_sma = phase_sma,
                dep_at_release = dep_at_release,
                plane_angle_cached = plane_angle_cached,
                tof_lower = tof_lower,
                tof_upper = tof_upper,
                span = span,
                sample_count = sample_count,
                is_simple_transfer = is_simple_transfer,
                tof_budget = tof_budget,
                tof_samples = tof_samples,
                tof_sample_n = tof_sample_n,
                hohmann_tof = hohmann_tof,
                tgt_period = tgt_period,
                period = dep_period,
                period_setup = [
                    let phase_sma_cubed = phase_sma * phase_sma * phase_sma;
                    let dep_period =
                        2.0 * std::f64::consts::PI * (phase_sma_cubed / MU).sqrt();
                ],
                transfer_sma_hohmann = transfer_sma_hohmann,
                transfer_period = transfer_period,
                transfer_period_setup = [
                    let transfer_sma_hohmann_cubed = transfer_sma_hohmann
                        * transfer_sma_hohmann
                        * transfer_sma_hohmann;
                    let transfer_period = 2.0
                        * std::f64::consts::PI
                        * (transfer_sma_hohmann_cubed / MU).sqrt();
                ],
                multi_rev_var = m,
                multi_rev_setup = [
                    let multi_rev_tof = hohmann_tof + f64::from(m) * dep_period;
                ],
                multi_rev_sample = [multi_rev_tof],
                rev_entry_var = n,
                rev_entry_setup = [
                    let t_rev_entry = hohmann_tof + (f64::from(n) - 0.5) * dep_period;
                ],
                rev_entry_sample = [t_rev_entry],
                bail = Err(EvaluationArithmeticOverflow),
            );
        }

        Ok((tof_samples, tof_sample_n))
    }

    #[test]
    fn tof_sample_ladder_expansions_cover_hard_interval_below_former_hohmann_floor() {
        let mut ctx = create_geo_transfer_context();
        ctx.revolution_cap = 0.0;
        let transfer_sma = 0.5 * (ctx.dep_sma + ctx.tgt_sma);
        let former_hohmann_floor =
            0.4 * 0.5 * std::f64::consts::TAU * (transfer_sma.powi(3) / MU).sqrt();
        assert!(former_hohmann_floor > MIN_TOF);
        let hard_upper = former_hohmann_floor * 1.5;
        let interval = admissible_tof_interval(&ctx, ctx.dep_period, hard_upper)
            .expect("hard physical interval must be admissible");

        let scalar =
            expand_tof_sample_ladder_for_test(&ctx, ctx.dep_sma, ctx.dep_eci, interval, false)
                .expect("scalar ladder expansion");
        let branch =
            expand_tof_sample_ladder_for_test(&ctx, ctx.dep_sma, ctx.dep_eci, interval, true)
                .expect("branch ladder expansion");

        assert_eq!(scalar.1, branch.1, "scalar/branch sample count drift");
        let scalar_samples = scalar
            .0
            .get(..scalar.1)
            .expect("scalar sample count must fit storage");
        let branch_samples = branch
            .0
            .get(..branch.1)
            .expect("branch sample count must fit storage");
        assert_eq!(
            scalar_samples
                .iter()
                .copied()
                .map(f64::to_bits)
                .collect::<Vec<_>>(),
            branch_samples
                .iter()
                .copied()
                .map(f64::to_bits)
                .collect::<Vec<_>>(),
            "scalar/branch sample bits drift"
        );
        assert_eq!(
            scalar_samples
                .first()
                .expect("hard interval must emit its lower endpoint")
                .to_bits(),
            MIN_TOF.to_bits()
        );
        assert_eq!(
            scalar_samples
                .last()
                .expect("hard interval must emit its upper endpoint")
                .to_bits(),
            hard_upper.to_bits(),
            "production ladder must cover the hard upper bound"
        );
        assert!(
            scalar_samples.iter().any(|&tof| tof < former_hohmann_floor),
            "hard interval must expose admissible TOFs below the former Hohmann floor"
        );
    }

    #[test]
    fn exact_minimum_tof_interval_runs_one_deduplicated_sample() {
        let mut ctx = create_leo_transfer_context();
        ctx.revolution_cap = 0.0;
        let interval = admissible_tof_interval(&ctx, ctx.dep_period, MIN_TOF)
            .expect("the exact MIN_TOF point is physically admissible");
        let (samples, count) =
            expand_tof_sample_ladder_for_test(&ctx, ctx.dep_sma, ctx.dep_eci, interval, false)
                .expect("one-point ladder expansion");

        assert_eq!(count, 1, "zero-span hard interval must run one sample");
        assert_eq!(
            samples
                .first()
                .expect("one-point interval must emit one sample")
                .to_bits(),
            MIN_TOF.to_bits()
        );
    }

    #[test]
    fn transfer_dv_limit_penalty_uses_final_post_hf_impulse_norm() {
        let final_hf_norm = 2.2;
        let stale_lambert_norm = 1.8;
        let max_transfer_dv = 2.0;

        assert_eq!(
            transfer_dv_limit_penalty(stale_lambert_norm, max_transfer_dv).to_bits(),
            0.0_f64.to_bits(),
            "test setup should keep the stale Lambert impulse under the cap"
        );
        assert!(
            transfer_dv_limit_penalty(final_hf_norm, max_transfer_dv) > 0.0,
            "post-HF release impulse must be the value checked against max_transfer_dv"
        );
    }

    #[test]
    fn evaluate_plan_branches_returns_multiple_lambert_paths_for_one_decision() {
        let ctx = create_leo_transfer_context();
        let mut branch_ids = std::collections::BTreeSet::new();
        for point in [
            [0.00, 1.00, 0.00],
            [0.05, 1.00, 0.05],
            [0.10, 1.00, 0.10],
            [0.12, 1.00, 0.18],
            [0.15, 1.02, 0.20],
            [0.20, 1.10, 0.10],
        ] {
            let mut lambert_scratch = crate::lambert::VariableR2LambertScratch::default();
            branch_ids =
                evaluate_plan_branches_with_scratch(&point, &ctx, false, &mut lambert_scratch)
                    .expect("LEO branch fixture must not overflow diagnostics")
                    .iter()
                    .filter(|plan| plan.valid)
                    .map(|plan| (plan.branch_rev, plan.branch_low_path))
                    .collect();
            if branch_ids.len() > 1 {
                break;
            }
        }

        assert!(
            branch_ids.len() > 1,
            "expected multiple valid Lambert branches for one decision, got {branch_ids:?}"
        );
    }

    fn assert_branch_plan_sequences_match(left: &[PlanResult], right: &[PlanResult]) {
        assert_eq!(left.len(), right.len(), "branch result count drifted");
        for (idx, (left, right)) in left.iter().zip(right.iter()).enumerate() {
            assert_eq!(left.valid, right.valid, "valid drift at row {idx}");
            assert_eq!(
                left.branch_rev, right.branch_rev,
                "branch rev drift at row {idx}"
            );
            assert_eq!(
                left.branch_low_path, right.branch_low_path,
                "branch low_path drift at row {idx}"
            );
            for (label, l, r) in [
                ("cost", left.cost, right.cost),
                ("time2phase", left.time2phase, right.time2phase),
                ("waittime", left.waittime, right.waittime),
                ("tof", left.tof, right.tof),
                ("phase_sma", left.phase_sma, right.phase_sma),
                ("phase_dv_norm", left.phase_dv_norm, right.phase_dv_norm),
                (
                    "transfer_dv_norm",
                    left.transfer_dv_norm,
                    right.transfer_dv_norm,
                ),
                (
                    "arrival_dv_norm",
                    left.arrival_dv_norm,
                    right.arrival_dv_norm,
                ),
                ("distance", left.distance, right.distance),
                (
                    "deployer_distance",
                    left.deployer_distance,
                    right.deployer_distance,
                ),
                (
                    "j2_endpoint_residual_m",
                    left.j2_endpoint_residual_m,
                    right.j2_endpoint_residual_m,
                ),
            ] {
                assert_eq!(
                    l.to_bits(),
                    r.to_bits(),
                    "{label} drift at row {idx}: {l:?} != {r:?}"
                );
            }
            for (label, l, r) in [
                ("phase_dv", left.phase_dv, right.phase_dv),
                ("transfer_dv", left.transfer_dv, right.transfer_dv),
                ("arrival_dv", left.arrival_dv, right.arrival_dv),
            ] {
                for (component, (l, r)) in l.iter().zip(r.iter()).enumerate() {
                    assert_eq!(
                        l.to_bits(),
                        r.to_bits(),
                        "{label}[{component}] drift at row {idx}: {l:?} != {r:?}"
                    );
                }
            }
            for (label, l, r) in [
                (
                    "payload_intercept_state",
                    left.payload_intercept_state,
                    right.payload_intercept_state,
                ),
                (
                    "target_intercept_state",
                    left.target_intercept_state,
                    right.target_intercept_state,
                ),
                (
                    "deployer_intercept_state",
                    left.deployer_intercept_state,
                    right.deployer_intercept_state,
                ),
                ("release_state", left.release_state, right.release_state),
            ] {
                for (component, (l, r)) in l.iter().zip(r.iter()).enumerate() {
                    assert_eq!(
                        l.to_bits(),
                        r.to_bits(),
                        "{label}[{component}] drift at row {idx}: {l:?} != {r:?}"
                    );
                }
            }
            assert_eq!(
                left.timing_failure_reason, right.timing_failure_reason,
                "failure reason drift at row {idx}"
            );
            assert_eq!(
                left.j2_iteration_count, right.j2_iteration_count,
                "J2 iteration count drift at row {idx}"
            );
        }
    }

    #[test]
    fn evaluate_plan_branches_matches_reference_for_leo_cases() {
        // T2.1 prep: cover the production rev range (max_revs=4), not just
        // the historical 0..=2, so the batch-rewire parity gate is armed.
        for max_revs in 0..=4 {
            let mut ctx = create_leo_transfer_context();
            ctx.max_revs = max_revs;
            for point in [
                [0.00, 1.00, 0.00],
                [0.05, 1.00, 0.05],
                [0.10, 1.00, 0.10],
                [0.12, 1.00, 0.18],
                [0.15, 1.02, 0.20],
            ] {
                let reference = evaluate_plan_branches_reference(&point, &ctx, false)
                    .expect("reference branch fixture must not overflow diagnostics");
                let mut lambert_scratch = crate::lambert::VariableR2LambertScratch::default();
                let prepared =
                    evaluate_plan_branches_with_scratch(&point, &ctx, false, &mut lambert_scratch)
                        .expect("prepared branch fixture must not overflow diagnostics");
                assert_branch_plan_sequences_match(&prepared, &reference);
            }
        }
    }

    #[derive(Debug)]
    struct LambertJ2BranchParityCase {
        name: &'static str,
        ctx: PlanContext,
        point: [f64; 3],
        expect_empty: bool,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct BranchPlanParityRow {
        valid: bool,
        branch_rev: i32,
        branch_low_path: bool,
        cost_bits: u64,
        total_dv_bits: u64,
        total_time_bits: u64,
        tof_bits: u64,
        phase_sma_bits: u64,
        phase_dv_norm_bits: u64,
        transfer_dv_norm_bits: u64,
        arrival_dv_norm_bits: u64,
        branch_departure_dv_bits: u64,
        branch_arrival_dv_bits: u64,
        j2_iteration_count: u32,
        j2_endpoint_residual_m_bits: u64,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct BranchFrontParitySnapshot {
        rows: Vec<BranchPlanParityRow>,
        front_hash: u64,
    }

    fn mix_hash_u64(hash: &mut u64, value: u64) {
        *hash ^= value;
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }

    fn branch_front_parity_snapshot(plans: &[PlanResult]) -> BranchFrontParitySnapshot {
        let rows = plans
            .iter()
            .map(|plan| BranchPlanParityRow {
                valid: plan.valid,
                branch_rev: plan.branch_rev,
                branch_low_path: plan.branch_low_path,
                cost_bits: plan.cost.to_bits(),
                total_dv_bits: plan.total_dv().to_bits(),
                total_time_bits: plan.total_time().to_bits(),
                tof_bits: plan.tof.to_bits(),
                phase_sma_bits: plan.phase_sma.to_bits(),
                phase_dv_norm_bits: plan.phase_dv_norm.to_bits(),
                transfer_dv_norm_bits: plan.transfer_dv_norm.to_bits(),
                arrival_dv_norm_bits: plan.arrival_dv_norm.to_bits(),
                branch_departure_dv_bits: plan.branch_departure_dv.to_bits(),
                branch_arrival_dv_bits: plan.branch_arrival_dv.to_bits(),
                j2_iteration_count: plan.j2_iteration_count,
                j2_endpoint_residual_m_bits: plan.j2_endpoint_residual_m.to_bits(),
            })
            .collect::<Vec<_>>();
        let mut front_hash = 0xcbf2_9ce4_8422_2325_u64;
        for row in &rows {
            mix_hash_u64(&mut front_hash, u64::from(u8::from(row.valid)));
            mix_hash_u64(
                &mut front_hash,
                u64::from_ne_bytes(i64::from(row.branch_rev).to_ne_bytes()),
            );
            mix_hash_u64(&mut front_hash, u64::from(u8::from(row.branch_low_path)));
            mix_hash_u64(&mut front_hash, row.cost_bits);
            mix_hash_u64(&mut front_hash, row.total_dv_bits);
            mix_hash_u64(&mut front_hash, row.total_time_bits);
            mix_hash_u64(&mut front_hash, row.tof_bits);
            mix_hash_u64(&mut front_hash, row.phase_sma_bits);
            mix_hash_u64(&mut front_hash, row.phase_dv_norm_bits);
            mix_hash_u64(&mut front_hash, row.transfer_dv_norm_bits);
            mix_hash_u64(&mut front_hash, row.arrival_dv_norm_bits);
            mix_hash_u64(&mut front_hash, row.branch_departure_dv_bits);
            mix_hash_u64(&mut front_hash, row.branch_arrival_dv_bits);
            mix_hash_u64(&mut front_hash, u64::from(row.j2_iteration_count));
            mix_hash_u64(&mut front_hash, row.j2_endpoint_residual_m_bits);
        }
        BranchFrontParitySnapshot { rows, front_hash }
    }

    fn branch_parity_point_for(ctx: &PlanContext) -> [f64; 3] {
        const POINTS: [[f64; 3]; 6] = [
            [0.00, 1.00, 0.00],
            [0.05, 1.00, 0.05],
            [0.10, 1.00, 0.10],
            [0.12, 1.00, 0.18],
            [0.15, 1.02, 0.20],
            [0.20, 1.10, 0.10],
        ];
        let [_, fallback, ..] = POINTS;
        if ctx.max_revs <= 0 {
            return fallback;
        }
        POINTS
            .into_iter()
            .find(|point| {
                let plans = evaluate_plan_branches_reference(point, ctx, false)
                    .expect("branch parity point fixture must not overflow diagnostics");
                plans.iter().any(|plan| plan.branch_low_path)
                    && plans.iter().any(|plan| !plan.branch_low_path)
            })
            .unwrap_or(fallback)
    }

    fn lambert_j2_branch_parity_cases() -> Vec<LambertJ2BranchParityCase> {
        let mut cases = Vec::new();
        for max_revs in 0..=2 {
            let mut ctx = create_leo_transfer_context();
            ctx.max_revs = max_revs;
            let point = branch_parity_point_for(&ctx);
            cases.push(LambertJ2BranchParityCase {
                name: match max_revs {
                    0 => "max_revs_0",
                    1 => "max_revs_1",
                    _ => "max_revs_2",
                },
                ctx,
                point,
                expect_empty: false,
            });
        }

        let high_path_ctx = create_quarter_arc_transfer_context();
        let high_path_point = branch_parity_point_for(&high_path_ctx);
        cases.push(LambertJ2BranchParityCase {
            name: "quarter_arc_high_path",
            ctx: high_path_ctx,
            point: high_path_point,
            expect_empty: false,
        });

        let mut invalid_geometry = create_leo_transfer_context();
        invalid_geometry.dep_eci = [0.0; 6];
        invalid_geometry.dep_equ = [0.0; 6];
        invalid_geometry.dep_orbit_cached = EciBasicOrbit::default();
        invalid_geometry.dep_orbit_valid = false;
        cases.push(LambertJ2BranchParityCase {
            name: "invalid_deployer_geometry",
            ctx: invalid_geometry,
            point: [0.05, 1.00, 0.05],
            expect_empty: true,
        });
        cases
    }

    fn lambert_backend_covers_high_path_branch() -> Result<bool, EvaluationArithmeticOverflow> {
        let state1 = [7000.0, 0.0, 0.0, 0.0, (MU / 7000.0_f64).sqrt(), 0.0];
        let state2 = [0.0, 7000.0, 0.0, -(MU / 7000.0_f64).sqrt(), 0.0, 0.0];
        let tof = 43_200.0;
        let mut saw_high_path = false;
        crate::lambert_backend::visit_lambert_branch_solutions(
            &state1,
            &state2,
            tof,
            2,
            true,
            |_rev, low_path, _prograde, _departure, _arrival| {
                saw_high_path |= !low_path;
            },
        )?;
        Ok(saw_high_path)
    }

    fn assert_lambert_j2_branch_parity_case(
        case: &LambertJ2BranchParityCase,
    ) -> (Option<i32>, bool, bool, bool) {
        let reference = evaluate_plan_branches_reference(&case.point, &case.ctx, false)
            .expect("reference branch parity fixture must not overflow diagnostics");
        let mut prepared_scratch = crate::lambert::VariableR2LambertScratch::default();
        let prepared = evaluate_plan_branches_with_scratch(
            &case.point,
            &case.ctx,
            false,
            &mut prepared_scratch,
        )
        .expect("prepared branch parity fixture must not overflow diagnostics");
        let mut production_scratch = crate::lambert::VariableR2LambertScratch::default();
        let production = evaluate_plan_branches_with_scratch(
            &case.point,
            &case.ctx,
            false,
            &mut production_scratch,
        )
        .expect("production branch parity fixture must not overflow diagnostics");
        assert_branch_plan_sequences_match(&prepared, &reference);
        assert_branch_plan_sequences_match(&production, &reference);

        let reference_snapshot = branch_front_parity_snapshot(&reference);
        let prepared_snapshot = branch_front_parity_snapshot(&prepared);
        let production_snapshot = branch_front_parity_snapshot(&production);
        assert_eq!(
            prepared_snapshot, reference_snapshot,
            "{} prepared branch parity snapshot drifted",
            case.name
        );
        assert_eq!(
            production_snapshot, reference_snapshot,
            "{} production branch parity snapshot drifted",
            case.name
        );

        if case.expect_empty {
            assert!(
                reference_snapshot.rows.is_empty(),
                "{} should reject invalid geometry",
                case.name
            );
            return (None, false, false, true);
        }

        assert!(
            !reference_snapshot.rows.is_empty(),
            "{} should emit at least one valid branch row",
            case.name
        );
        let saw_low_path = reference_snapshot
            .rows
            .iter()
            .any(|row| row.branch_low_path);
        let saw_high_path = reference_snapshot
            .rows
            .iter()
            .any(|row| !row.branch_low_path);
        assert!(
            reference_snapshot
                .rows
                .iter()
                .all(|row| row.branch_rev <= case.ctx.max_revs.max(0)),
            "{} emitted a branch above max_revs={}",
            case.name,
            case.ctx.max_revs
        );
        if case.ctx.max_revs == 0 {
            assert!(
                reference_snapshot
                    .rows
                    .iter()
                    .all(|row| row.branch_rev == 0 && row.branch_low_path),
                "{} must keep M0 on the canonical low-path branch",
                case.name
            );
        }
        (Some(case.ctx.max_revs), saw_low_path, saw_high_path, false)
    }

    fn assert_lambert_j2_branch_parity_harness_covers_required_cases() {
        let mut covered_max_revs = std::collections::BTreeSet::new();
        let mut saw_low_path = false;
        let mut saw_high_path = lambert_backend_covers_high_path_branch()
            .expect("Lambert branch parity fixture must not overflow diagnostics");
        let mut saw_invalid_geometry_reject = false;
        for case in lambert_j2_branch_parity_cases() {
            let (covered_max_rev, case_saw_low_path, case_saw_high_path, rejected_geometry) =
                assert_lambert_j2_branch_parity_case(&case);
            if let Some(max_revs) = covered_max_rev {
                covered_max_revs.insert(max_revs);
            }
            saw_low_path |= case_saw_low_path;
            saw_high_path |= case_saw_high_path;
            saw_invalid_geometry_reject |= rejected_geometry;
        }
        assert_eq!(
            covered_max_revs,
            [0, 1, 2]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            "harness must cover max_revs=0/1/2"
        );
        assert!(saw_low_path, "harness must cover a low-path branch row");
        assert!(
            saw_high_path,
            "harness must cover Lambert high-path branch enumeration"
        );
        assert!(
            saw_invalid_geometry_reject,
            "harness must cover invalid geometry rejection"
        );
    }

    #[test]
    fn lambert_j2_branch_parity_harness_covers_required_cases() {
        assert_lambert_j2_branch_parity_harness_covers_required_cases();
    }

    #[test]
    fn evaluate_prepared_plan_branch_matches_selected_scalar_evaluator() {
        let ctx = create_leo_transfer_context();
        let point = [0.05, 1.00, 0.05];
        let prepared = prepare_branch_shared_work(&point, &ctx, false)
            .expect("test decision diagnostics must not overflow")
            .expect("test decision should prepare");
        let mut prepared_branch_ctx = prepared.ctx.clone();

        for (rev, low_path) in [(0, true), (1, true), (1, false), (2, true), (2, false)] {
            let mut selected_ctx = ctx.clone();
            selected_ctx.lambert_branch_selection = Some(LambertBranchSelection { rev, low_path });
            let expected = evaluate_plan_from_phase_with_lambert_scratch(
                &point,
                &selected_ctx,
                false,
                prepared.time2phase,
                prepared.waittime,
                prepared.dep_period,
                &prepared.dep_at_phase,
                Some(prepared.dep_phase_orbit),
                None,
            )
            .expect("selected scalar fixture must not overflow diagnostics");
            let mut lambert_scratch = crate::lambert::VariableR2LambertScratch::default();
            let actual = evaluate_prepared_plan_branch(
                &prepared,
                &mut prepared_branch_ctx,
                rev,
                low_path,
                &branch_lane_prep(&prepared),
                &mut lambert_scratch,
            )
            .expect("prepared branch fixture must not overflow diagnostics");

            assert_eq!(
                actual.valid, expected.valid,
                "validity drift for rev={rev}, low_path={low_path}"
            );
            if expected.valid {
                assert_branch_plan_sequences_match(&[actual], &[expected]);
            }
        }
    }

    #[test]
    fn prepared_selected_branch_records_exact_batch_work() {
        let ctx = create_leo_transfer_context();
        let point = [0.05, 1.00, 0.05];
        let prepared = prepare_branch_shared_work(&point, &ctx, false)
            .expect("test decision diagnostics must not overflow")
            .expect("test decision should prepare");
        assert!(
            !prepared.valid_tofs.is_empty(),
            "test setup must sample TOFs"
        );

        let before = evaluation_diagnostic_snapshot();
        let mut branch_ctx = prepared.ctx.clone();
        let mut lambert_scratch = crate::lambert::VariableR2LambertScratch::default();
        let _actual = evaluate_prepared_plan_branch(
            &prepared,
            &mut branch_ctx,
            0,
            true,
            &branch_lane_prep(&prepared),
            &mut lambert_scratch,
        )
        .expect("prepared branch fixture must not overflow diagnostics");
        let delta = evaluation_diagnostic_snapshot()
            .delta_since(before)
            .expect("prepared branch diagnostics must not underflow");
        let work = lambert_scratch.branch_telemetry();

        assert_eq!(
            delta.lambert_batch_simd_lane_solve_count,
            work.simd_lane_solves
        );
        assert_eq!(
            delta.lambert_batch_scalar_variant_solve_count,
            work.scalar_variant_solves
        );
        // R18: the selected-branch rows run through the pack kernel, so the
        // per-variant work is SIMD lanes now; the bound is unchanged — at most
        // one prograde and one retrograde variant per TOF, nothing unselected.
        assert_eq!(work.scalar_variant_solves, 0);
        assert!(work.simd_lane_solves > 0);
        let max_variant_solves = prepared
            .valid_tofs
            .len()
            .checked_mul(2)
            .expect("bounded test TOF count must fit usize");
        assert!(
            work.simd_lane_solves <= max_variant_solves,
            "selected M0 may solve only prograde and retrograde variants per TOF"
        );
    }

    /// Survivor identity under the multi-rev energy prune.
    ///
    /// The prune drops revolution branches whose departure dv is already
    /// bounded below by the acceptance cap, so the argmin the selector returns
    /// — and every float hanging off it — must be bit-identical to the
    /// unpruned enumeration. `assert_eq` on the raw bits, not a tolerance:
    /// anything short of bit equality moves a sealed digest.
    #[test]
    fn multi_rev_prune_selection_is_bit_identical_to_the_unpruned_enumeration() {
        let mut ctx = create_leo_transfer_context();
        ctx.max_revs = 4;
        let dep_r = RE + 400.0;
        let dep_v = (MU / dep_r).sqrt();
        let mut compared = 0_usize;
        let mut survivors = 0_usize;
        let mut multi_rev_survivors = 0_usize;
        let mut prune_fired = 0_usize;
        for max_transfer_dv in [0.05, 0.2, 0.75, 2.0] {
            ctx.max_transfer_dv = max_transfer_dv;
            for tof_step in 0..48_i32 {
                let tof = 900.0 + f64::from(tof_step) * 2_100.0;
                for target_step in 0..6_i32 {
                    let target_r = RE + 450.0 + f64::from(target_step) * 90.0;
                    let target_v = (MU / target_r).sqrt();
                    let angle = 0.3 + f64::from(target_step) * 0.9;
                    let dep_at_release = [dep_r, 0.0, 0.0, 0.0, dep_v, 0.02];
                    let tgt_state = [
                        target_r * angle.cos(),
                        target_r * angle.sin(),
                        0.0,
                        -target_v * angle.sin(),
                        target_v * angle.cos(),
                        0.0,
                    ];
                    let r1_cache = crate::lambert::LambertR1Cache::new(&[
                        dep_at_release[0],
                        dep_at_release[1],
                        dep_at_release[2],
                    ]);
                    for phase_dv_norm in [0.0, 0.1] {
                        let capped = select_lambert_branch_solution_with_r1(
                            &ctx,
                            &dep_at_release,
                            &r1_cache,
                            &crate::lambert_backend::DepartureBoundCache::new(&dep_at_release),
                            &tgt_state,
                            tof,
                        )
                        .expect("capped selection must not overflow diagnostics");
                        let uncapped = select_lambert_branch_solution_uncapped(
                            &ctx,
                            &dep_at_release,
                            &r1_cache,
                            &tgt_state,
                            tof,
                        )
                        .expect("uncapped selection must not overflow diagnostics");
                        compared += 1;
                        let dv_cap = ctx
                            .max_transfer_dv
                            .min((INVALID_COST - phase_dv_norm).max(0.0));
                        if crate::lambert_backend::max_revolutions_below_dv_cap(
                            &dep_at_release,
                            &tgt_state,
                            tof,
                            dv_cap,
                            ctx.max_revs,
                        ) < ctx.max_revs
                        {
                            prune_fired += 1;
                        }
                        match (capped, uncapped) {
                            (None, None) => {}
                            (Some(capped), Some(uncapped)) => {
                                survivors += 1;
                                if uncapped.best_M >= 1 {
                                    multi_rev_survivors += 1;
                                }
                                assert_eq!(
                                    capped.cost.to_bits(),
                                    uncapped.cost.to_bits(),
                                    "pruned survivor cost drifted at tof={tof}, \
                                     max_transfer_dv={max_transfer_dv}"
                                );
                                assert_eq!(capped.best_M, uncapped.best_M);
                                assert_eq!(capped.low_path, uncapped.low_path);
                                assert_eq!(capped.prograde, uncapped.prograde);
                                for (left, right) in capped.dv.iter().zip(uncapped.dv.iter()) {
                                    assert_eq!(left.to_bits(), right.to_bits());
                                }
                                for (left, right) in
                                    capped.arrival_dv.iter().zip(uncapped.arrival_dv.iter())
                                {
                                    assert_eq!(left.to_bits(), right.to_bits());
                                }
                            }
                            (capped, uncapped) => panic!(
                                "prune changed survivor existence at tof={tof}, \
                                 max_transfer_dv={max_transfer_dv}: {} vs {}",
                                capped.is_some(),
                                uncapped.is_some()
                            ),
                        }
                    }
                }
            }
        }
        assert!(
            compared > 2_000,
            "identity sweep must cover the branch grid"
        );
        assert!(
            prune_fired > 0,
            "identity sweep must contain cases where the prune actually cuts a branch"
        );
        assert!(
            multi_rev_survivors > 0,
            "identity sweep must contain multi-rev survivors, or it never tests \
             the branches the prune reasons about"
        );
        assert!(survivors > 0, "identity sweep must find survivors");
    }

    /// Multi-rev sole-survivor mechanism: when the free enumeration's winner is
    /// a multi-rev branch, the `m = 0` branch must have been ATTEMPTED — the
    /// energy prune's revolution ceiling never excludes zero revolutions, so
    /// the enumerator always solves it — and rejected for a stated cause:
    /// either its departure dv fails the acceptance filter
    /// (`dv < max_transfer_dv`, the exact predicate in
    /// `fold_lambert_branch_candidate`) or it lost the argmin on cost. A
    /// multi-rev "win" must never be the artifact of `m = 0` being silently
    /// skipped. The sweep must also contain at least one SOLE-SURVIVOR case
    /// (every enumerated `m = 0` branch over the cap), the 99.5% production
    /// mechanism, or the test proves nothing about it.
    #[test]
    fn multi_rev_winner_has_m0_attempted_and_rejected_for_cause() {
        let mut ctx = create_leo_transfer_context();
        ctx.max_revs = 4;
        let dep_r = RE + 400.0;
        let dep_v = (MU / dep_r).sqrt();
        let mut multi_rev_wins = 0_usize;
        let mut sole_survivor_wins = 0_usize;
        for max_transfer_dv in [0.05, 0.2, 0.75] {
            ctx.max_transfer_dv = max_transfer_dv;
            for tof_step in 0..24_i32 {
                let tof = 900.0 + f64::from(tof_step) * 4_200.0;
                for target_step in 0..6_i32 {
                    let target_r = RE + 450.0 + f64::from(target_step) * 90.0;
                    let target_v = (MU / target_r).sqrt();
                    let angle = 0.3 + f64::from(target_step) * 0.9;
                    let dep_at_release = [dep_r, 0.0, 0.0, 0.0, dep_v, 0.02];
                    let tgt_state = [
                        target_r * angle.cos(),
                        target_r * angle.sin(),
                        0.0,
                        -target_v * angle.sin(),
                        target_v * angle.cos(),
                        0.0,
                    ];
                    let r1_cache = crate::lambert::LambertR1Cache::new(&[
                        dep_at_release[0],
                        dep_at_release[1],
                        dep_at_release[2],
                    ]);
                    let selected = select_lambert_branch_solution_with_r1(
                        &ctx,
                        &dep_at_release,
                        &r1_cache,
                        &crate::lambert_backend::DepartureBoundCache::new(&dep_at_release),
                        &tgt_state,
                        tof,
                    )
                    .expect("free selection must not overflow diagnostics");
                    let Some(winner) = selected else {
                        continue;
                    };
                    if winner.best_M < 1 {
                        continue;
                    }
                    multi_rev_wins += 1;

                    // Re-enumerate every branch (m = 0 included, retrograde
                    // included — a superset of the pruned production visit) and
                    // audit the m = 0 candidates the winner had to beat.
                    let mut m0_dv_norms: Vec<f64> = Vec::new();
                    crate::lambert_backend::visit_lambert_branch_solutions(
                        &dep_at_release,
                        &tgt_state,
                        tof,
                        ctx.max_revs,
                        true,
                        |m, _low_path, _prograde, dv_vec, _arrival| {
                            if m == 0 {
                                m0_dv_norms.push(norm3(&dv_vec));
                            }
                        },
                    )
                    .expect("oracle enumeration must not overflow diagnostics");
                    assert!(
                        !m0_dv_norms.is_empty(),
                        "multi-rev winner at tof={tof}, cap={max_transfer_dv}: the m=0 \
                         branch was never enumerated — a silent skip, not a rejection"
                    );
                    let mut all_cap_rejected = true;
                    for dv_norm in &m0_dv_norms {
                        let cap_rejected = !(dv_norm.is_finite() && *dv_norm < ctx.max_transfer_dv);
                        let cost_beaten = dv_norm.is_finite() && *dv_norm >= winner.cost;
                        assert!(
                            cap_rejected || cost_beaten,
                            "multi-rev winner (m={}, cost={}) at tof={tof}, \
                             cap={max_transfer_dv}: an enumerated m=0 branch with \
                             dv={dv_norm} passed the acceptance filter and undercut the \
                             winner — the selector's argmin is broken",
                            winner.best_M,
                            winner.cost
                        );
                        if !cap_rejected {
                            all_cap_rejected = false;
                        }
                    }
                    if all_cap_rejected {
                        sole_survivor_wins += 1;
                    }
                }
            }
        }
        // The sweep is 3 dv caps x 24 tof steps x 6 target radii = 432 cases,
        // of which 26 currently produce a multi-rev winner and all 26 are
        // sole-survivor wins. These floors were `> 0`, which is a real floor
        // but a very weak one: a regression that took the multi-rev population
        // from 26 down to 1 would leave both green while the m=0 audit ran on
        // a single case. 15 keeps ample headroom for legitimate numerical
        // drift in which cases qualify, and still fails on a collapse.
        assert!(
            multi_rev_wins >= 15,
            "sweep produced only {multi_rev_wins} multi-rev winners out of 432 cases \
             (26 expected); the m=0 audit is running on too thin a population to mean anything"
        );
        assert!(
            sole_survivor_wins >= 15,
            "sweep produced only {sole_survivor_wins} sole-survivor multi-rev wins out of 432 \
             cases (26 expected) -- every m=0 branch over the dv cap is the mechanism this \
             test exists to pin"
        );
    }

    /// Whole-batch retrograde prune parity: with a transfer-dv cap below
    /// every lane's retrograde departure bound the batch caller drops the
    /// retrograde passes, and its results must remain bitwise identical to
    /// the scalar selected-branch evaluator (which prunes per lane).
    #[test]
    fn evaluate_prepared_plan_branch_batch_retrograde_prune_is_bit_identical() {
        let mut ctx = create_leo_transfer_context();
        // Well above the coplanar-ish prograde transfer dv, well below the
        // deployer tangential speed that lower-bounds every retrograde lane.
        ctx.max_transfer_dv = 0.5;
        let point = [0.05, 1.00, 0.05];
        let prepared = prepare_branch_shared_work(&point, &ctx, false)
            .expect("test decision diagnostics must not overflow")
            .expect("test decision should prepare");

        // The whole-batch prune must actually fire for this geometry: every
        // lane bound meets/exceeds the acceptance cap, which is
        // `max_transfer_dv` alone.
        assert!(
            !crate::lambert_backend::batch_retrograde_included(
                &prepared.dep_at_release,
                prepared.tof_to_idx.iter().map(|&state_idx| {
                    prepared
                        .tgt_states_full
                        .get(state_idx)
                        .expect("TOF index derives from the prepared target-state grid")
                }),
                ctx.max_transfer_dv,
            ),
            "test geometry must allow the whole-batch retrograde prune"
        );

        let mut prepared_branch_ctx = prepared.ctx.clone();
        let mut saw_valid_branch = false;
        for (rev, low_path) in [(0, true), (1, true), (1, false), (2, true), (2, false)] {
            let mut selected_ctx = ctx.clone();
            selected_ctx.lambert_branch_selection = Some(LambertBranchSelection { rev, low_path });
            let expected = evaluate_plan_from_phase_with_lambert_scratch(
                &point,
                &selected_ctx,
                false,
                prepared.time2phase,
                prepared.waittime,
                prepared.dep_period,
                &prepared.dep_at_phase,
                Some(prepared.dep_phase_orbit),
                None,
            )
            .expect("selected scalar fixture must not overflow diagnostics");
            let mut lambert_scratch = crate::lambert::VariableR2LambertScratch::default();
            let actual = evaluate_prepared_plan_branch(
                &prepared,
                &mut prepared_branch_ctx,
                rev,
                low_path,
                &branch_lane_prep(&prepared),
                &mut lambert_scratch,
            )
            .expect("prepared branch fixture must not overflow diagnostics");

            assert_eq!(
                actual.valid, expected.valid,
                "validity drift under batch retrograde prune for rev={rev}, low_path={low_path}"
            );
            if expected.valid {
                saw_valid_branch = true;
                assert!(
                    actual.prograde,
                    "pruned-batch valid branch must be prograde for rev={rev}, low_path={low_path}"
                );
                assert_branch_plan_sequences_match(&[actual], &[expected]);
            }
        }
        assert!(
            saw_valid_branch,
            "prune parity test must exercise at least one valid branch"
        );
    }

    #[test]
    fn prepared_branch_path_records_shared_work_and_reduces_target_grid_propagation() {
        let mut ctx = create_leo_transfer_context();
        ctx.target_propagation_authority = TargetPropagationAuthority::MfJ2;
        let point = [0.05, 1.00, 0.05];

        let before_reference = evaluation_diagnostic_snapshot();
        let reference = evaluate_plan_branches_reference(&point, &ctx, false)
            .expect("reference branch fixture must not overflow diagnostics");
        let after_reference = evaluation_diagnostic_snapshot();
        let reference_delta = after_reference
            .delta_since(before_reference)
            .expect("reference diagnostics must not underflow");
        assert!(!reference.is_empty(), "test setup must emit branch rows");

        let before_prepared = evaluation_diagnostic_snapshot();
        let mut lambert_scratch = crate::lambert::VariableR2LambertScratch::default();
        let prepared =
            evaluate_plan_branches_with_scratch(&point, &ctx, false, &mut lambert_scratch)
                .expect("prepared branch fixture must not overflow diagnostics");
        let after_prepared = evaluation_diagnostic_snapshot();
        let prepared_delta = after_prepared
            .delta_since(before_prepared)
            .expect("prepared diagnostics must not underflow");

        assert_branch_plan_sequences_match(&prepared, &reference);
        assert_eq!(prepared_delta.branch_shared_prepare_count, 1);
        let nonnegative_revs = usize::try_from(ctx.max_revs.max(0))
            .expect("nonnegative i32 revision count fits usize");
        let expected_branch_calls = nonnegative_revs
            .checked_add(1)
            .and_then(|count| count.checked_mul(2))
            .and_then(|count| count.checked_sub(1))
            .expect("bounded branch test count must fit usize");
        assert_eq!(prepared_delta.branch_eval_call_count, expected_branch_calls);
        assert_eq!(prepared_delta.branch_emitted_count, prepared.len());
        assert!(
            prepared_delta.target_j2_batch_state_count
                < reference_delta.target_j2_batch_state_count,
            "prepared path should batch-propagate the target grid once instead of per branch"
        );
    }

    #[test]
    fn evaluate_plan_branches_records_prepared_split_diagnostics() {
        let mut ctx = create_leo_transfer_context();
        ctx.target_propagation_authority = TargetPropagationAuthority::MfJ2;
        let point = [0.05, 1.00, 0.05];

        let before = evaluation_diagnostic_snapshot();
        let mut lambert_scratch = crate::lambert::VariableR2LambertScratch::default();
        let plans = evaluate_plan_branches_with_scratch(&point, &ctx, false, &mut lambert_scratch)
            .expect("prepared branch fixture must not overflow diagnostics");
        let after = evaluation_diagnostic_snapshot();
        let delta = after
            .delta_since(before)
            .expect("branch diagnostics must not underflow");

        let nonnegative_revs = usize::try_from(ctx.max_revs.max(0))
            .expect("nonnegative i32 revision count fits usize");
        let expected_branch_calls = nonnegative_revs
            .checked_add(1)
            .and_then(|count| count.checked_mul(2))
            .and_then(|count| count.checked_sub(1))
            .expect("bounded branch test count must fit usize");
        assert!(!plans.is_empty(), "test setup must emit branch plans");
        assert_eq!(delta.branch_shared_prepare_count, 1);
        assert_eq!(delta.branch_source_count, 1);
        assert_eq!(delta.branch_eval_call_count, expected_branch_calls);
        assert_eq!(delta.branch_emitted_count, plans.len());
        let emitted_or_rejected = delta
            .branch_emitted_count
            .checked_add(delta.branch_rejected_count)
            .expect("branch outcome count must fit usize");
        assert_eq!(emitted_or_rejected, delta.branch_eval_call_count);
        assert_eq!(delta.branch_target_propagation_call_count, 0);
        assert!(delta.target_j2_batch_state_count > 0);
        assert_eq!(
            delta.branch_lambert_sampling_call_count,
            expected_branch_calls
        );
        assert_eq!(
            delta.lambert_scalar_tof_count, 0,
            "prepared branch ranking must reuse batch Lambert rows, not scalar re-solve ranked TOFs"
        );
        assert!(
            delta.branch_target_propagation_s > 0.0,
            "target propagation timing should be recorded for prepared branch path"
        );
        assert!(
            delta.branch_lambert_sampling_s > 0.0,
            "Lambert sampling timing should be recorded for scalar branch path"
        );
        assert!(
            delta.branch_brent_call_count > 0,
            "at least one branch should enter Brent refinement"
        );
        assert!(
            delta.branch_brent_eval_request_count > 0,
            "Brent diagnostics should count objective requests"
        );
        let brent_cache_requests = delta
            .branch_brent_cache_hit_count
            .checked_add(delta.branch_brent_cache_miss_count)
            .expect("Brent cache request count must fit usize");
        assert_eq!(delta.branch_brent_eval_request_count, brent_cache_requests);
        assert!(
            delta.branch_brent_cache_miss_count > 0,
            "branch Brent cache diagnostics should count computed objective misses"
        );
        assert_eq!(
            delta.target_j2_scalar_state_count, delta.branch_brent_cache_miss_count,
            "prepared Brent winners must reuse exact solved Lambert states instead of adding one hidden target propagation per converged branch"
        );
        assert!(
            delta.branch_j2_correction_call_count > 0,
            "at least one branch should enter J2 correction"
        );
    }

    #[test]
    fn normal_brent_winner_reuses_exact_solved_lambert_state() {
        let mut ctx = create_leo_transfer_context();
        ctx.max_revs = 0;
        ctx.target_propagation_authority = TargetPropagationAuthority::MfJ2;
        ctx.lambert_branch_selection = Some(LambertBranchSelection {
            rev: 0,
            low_path: true,
        });
        ctx.j2_closure_settings.max_iterations = 0;
        let point = [0.05, 1.00, 0.05];

        let before = evaluation_diagnostic_snapshot();
        let _plan = evaluate_plan(&point, &ctx, false)
            .expect("Brent reuse fixture must not overflow diagnostics");
        let delta = evaluation_diagnostic_snapshot()
            .delta_since(before)
            .expect("Brent diagnostics must not underflow");

        assert_eq!(delta.branch_brent_call_count, 1);
        assert!(delta.branch_brent_cache_miss_count > 0);
        assert_eq!(
            delta.target_j2_scalar_state_count, delta.branch_brent_cache_miss_count,
            "normal Brent winner must reuse its exact solved Lambert state instead of one hidden final re-solve"
        );
    }

    #[test]
    fn hf_propagation_telemetry_records_grid_brent_and_refinement() {
        let baseline = hf_propagation_telemetry_snapshot();
        let ctx = create_leo_transfer_context();
        let point = [0.05, 1.00, 0.05];
        let before = evaluate_plan(&point, &ctx, false)
            .expect("telemetry fixture must not overflow diagnostics");

        record_hf_propagation_stage(
            HfPropagationStage::TargetGrid {
                requested_states: 8,
                unique_attempted_states: 6,
            },
            0.25,
        )
        .expect("test HF telemetry record must not overflow");
        record_hf_propagation_stage(HfPropagationStage::Brent, 0.50)
            .expect("test HF telemetry record must not overflow");
        record_hf_propagation_stage(HfPropagationStage::InterceptRefinement(7), 0.75)
            .expect("test HF telemetry record must not overflow");
        let enabled = hf_propagation_telemetry_snapshot()
            .delta_since(baseline)
            .expect("test HF telemetry delta must not underflow");
        assert_eq!(enabled.target_grid_call_count, 1);
        assert_eq!(enabled.target_grid_requested_state_count, 8);
        assert_eq!(enabled.target_grid_unique_attempted_state_count, 6);
        assert_eq!(enabled.target_grid_s.to_bits(), 0.25_f64.to_bits());
        assert_eq!(enabled.brent_call_count, 1);
        assert_eq!(enabled.brent_s.to_bits(), 0.50_f64.to_bits());
        assert_eq!(enabled.intercept_refinement_call_count, 1);
        assert_eq!(enabled.intercept_refinement_iteration_count, 7);
        assert_eq!(enabled.intercept_refinement_s.to_bits(), 0.75_f64.to_bits());

        let after = evaluate_plan(&point, &ctx, false)
            .expect("telemetry fixture must not overflow diagnostics");
        assert_eq!(
            branch_front_parity_snapshot(&[before]),
            branch_front_parity_snapshot(&[after]),
            "diagnostic-only telemetry must not mutate science output"
        );
    }

    #[test]
    fn hf_propagation_telemetry_reduces_with_worker_diagnostics() {
        std::thread::spawn(|| {
            let before = hf_propagation_telemetry_snapshot();
            let worker = EvaluationDiagnosticCounters {
                hf_propagation: HfPropagationTelemetry {
                    target_grid_call_count: 1,
                    target_grid_requested_state_count: 8,
                    target_grid_unique_attempted_state_count: 6,
                    target_grid_s: 0.25,
                    brent_call_count: 1,
                    brent_s: 0.50,
                    intercept_refinement_call_count: 1,
                    intercept_refinement_iteration_count: 7,
                    intercept_refinement_s: 0.75,
                },
                ..Default::default()
            };
            merge_evaluation_diagnostics(&worker)
                .expect("worker HF telemetry merge must not overflow");
            assert_eq!(
                hf_propagation_telemetry_snapshot()
                    .delta_since(before)
                    .expect("worker HF telemetry delta must not underflow"),
                worker.hf_propagation
            );
        })
        .join()
        .expect("HF propagation telemetry reduction thread");
    }

    #[test]
    fn brent_refinement_guard_flips_at_sixty_second_bracket() {
        // Pins the guard's own 60 s literal. It does NOT tie that literal to
        // the minimizer's fine-xatol clamp (the `.clamp(1.0, 60.0)` sites);
        // moving the clamp alone leaves this green.
        assert!(!brent_refinement_required(0.0, 59.999_999));
        assert!(brent_refinement_required(0.0, 60.0));
    }

    #[test]
    fn brent_cache_key_rejects_nonrepresentable_tof() {
        assert_eq!(brent_tof_cache_key(12_345.01), Ok(123_450));
        assert_eq!(
            brent_tof_cache_key(f64::NAN),
            Err(EvaluationArithmeticOverflow)
        );
        assert_eq!(
            brent_tof_cache_key(f64::INFINITY),
            Err(EvaluationArithmeticOverflow)
        );
        assert_eq!(
            brent_tof_cache_key(f64::MAX),
            Err(EvaluationArithmeticOverflow)
        );
    }

    #[test]
    fn brent_solution_reuse_rejects_different_bits_in_same_cost_cache_bin() {
        let tof = 12_345.01_f64;
        let same_bin_different_bits = 12_345.02_f64;
        let tof_key = (tof * 10.0)
            .round()
            .to_i64()
            .expect("finite test TOF must quantize to i64");
        let same_bin_key = (same_bin_different_bits * 10.0)
            .round()
            .to_i64()
            .expect("finite test TOF must quantize to i64");
        assert_eq!(tof_key, same_bin_key);
        assert_ne!(tof.to_bits(), same_bin_different_bits.to_bits());
        let solution = LambertSolutionEx {
            cost: 1.25,
            valid: true,
            ..Default::default()
        };
        let cache = [(tof.to_bits(), solution)];
        let recompute_count = std::cell::Cell::new(0usize);

        assert_eq!(
            brent_exact_solution_lookup(&cache, tof)
                .or_else(|| {
                    let next_count = recompute_count
                        .get()
                        .checked_add(1)
                        .expect("bounded test recompute count must fit usize");
                    recompute_count.set(next_count);
                    None
                })
                .expect("exact TOF bits must reuse solved Lambert state")
                .cost
                .to_bits(),
            solution.cost.to_bits()
        );
        assert_eq!(recompute_count.get(), 0, "exact reuse must avoid re-solve");

        let recomputed = LambertSolutionEx {
            cost: 2.5,
            valid: true,
            ..Default::default()
        };
        assert_eq!(
            brent_exact_solution_lookup(&cache, same_bin_different_bits)
                .or_else(|| {
                    let next_count = recompute_count
                        .get()
                        .checked_add(1)
                        .expect("bounded test recompute count must fit usize");
                    recompute_count.set(next_count);
                    Some(recomputed)
                })
                .expect("different exact TOF bits must recompute")
                .cost
                .to_bits(),
            recomputed.cost.to_bits()
        );
        assert_eq!(recompute_count.get(), 1, "same-bin miss must re-solve once");
    }

    #[test]
    fn transfer_revolution_cap_limit_uses_deployer_period() {
        let mut ctx = create_geo_transfer_context();
        ctx.revolution_cap = 0.5;

        let limit = transfer_revolution_cap_s(&ctx, ctx.dep_period).unwrap();

        let half_period = ctx.dep_period * 0.5;
        assert_eq!(limit.to_bits(), half_period.to_bits());
    }

    #[test]
    fn transfer_revolution_cap_detects_excess_tof() {
        let mut ctx = create_geo_transfer_context();
        ctx.revolution_cap = 0.5;

        let half_period = ctx.dep_period * 0.5;
        let excessive_tof = half_period + 1.0;
        assert!(transfer_tof_exceeds_revolution_cap(
            &ctx,
            excessive_tof,
            ctx.dep_period
        ));
    }

    #[test]
    fn iterative_j2_correction_does_not_mutate_the_seeded_target_state() {
        // The fixture deliberately biases the Lambert endpoint before seeding,
        // so the state compared below is the PERTURBED seed, not the nominal
        // target; the assertion is a pass-through check on the seeded state.
        let mut ctx = create_geo_transfer_context();
        // Avoid zero transfer-angle degeneracy by phase-shifting the target by 90 degrees.
        let tgt_r = ctx.tgt_sma;
        let tgt_v = (MU / tgt_r).sqrt();
        ctx.tgt_eci = [0.0, tgt_r, 0.0, -tgt_v, 0.0, 0.0];
        eci_to_equinoctial(&ctx.tgt_eci, &mut ctx.tgt_equ);
        ctx.cache_target_orbit();
        ctx.cache_plane_angle();
        let dep_at_release = ctx.dep_eci;
        let tof_candidates = [3.0 * 3600.0, 6.0 * 3600.0, 12.0 * 3600.0, 24.0 * 3600.0];

        let mut seeded_case: Option<(f64, LambertSolutionEx)> = None;
        for tof in tof_candidates {
            let solved = lambert_solve_with_target_state(
                tof,
                &ctx,
                &dep_at_release,
                &crate::lambert::LambertR1Cache::new(&[
                    dep_at_release[0],
                    dep_at_release[1],
                    dep_at_release[2],
                ]),
                &crate::lambert_backend::DepartureBoundCache::new(&dep_at_release),
                &ctx.tgt_eci,
            )
            .expect("Lambert seed fixture must not overflow diagnostics");
            if let Some(mut sol) = solved {
                // Inject a small synthetic endpoint bias to ensure at least one correction step.
                sol.tgt_state[0] += 0.05;
                sol.tgt_state[1] -= 0.04;
                sol.tgt_state[2] += 0.03;
                seeded_case = Some((tof, sol));
                break;
            }
        }

        let (tof, seeded_sol) =
            seeded_case.expect("expected at least one solvable Lambert seed case");
        let expected_nominal_target = seeded_sol.tgt_state;
        let j2_settings = crate::solve::J2ClosureSettings::default();

        let (refined_sol, correction_steps, residual_m) = apply_iterative_j2_lambert_correction(
            &ctx,
            &dep_at_release,
            tof,
            ctx.epoch_jd,
            seeded_sol,
            j2_settings,
        )
        .expect("J2 correction fixture must not overflow diagnostics");

        assert!(
            correction_steps >= 1,
            "test setup expected at least one correction step, got {correction_steps}"
        );
        assert!(
            correction_steps <= j2_settings.max_iterations,
            "correction steps {correction_steps} must be <= max {}",
            j2_settings.max_iterations
        );
        for (component, (&refined, &expected)) in refined_sol
            .tgt_state
            .iter()
            .zip(expected_nominal_target.iter())
            .enumerate()
        {
            assert!(
                (refined - expected).abs() <= 1e-12,
                "target component {component} drifted: got {refined:.15e}, expected {expected:.15e}"
            );
        }
        assert!(
            residual_m.is_finite(),
            "endpoint residual should remain finite, got {residual_m}"
        );
    }

    #[test]
    fn post_hf_endpoint_residual_is_recomputed_from_final_states() {
        let payload_at_intercept = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let target_at_intercept = [7000.001, 0.002, 0.0, 0.0, 7.5, 0.0];

        let residual_m =
            recompute_post_hf_endpoint_residual_m(&payload_at_intercept, &target_at_intercept);

        let expected_distance_squared_km2 = 0.001_f64.mul_add(0.001, 0.002_f64.powi(2));
        let expected_distance_km = expected_distance_squared_km2.sqrt();
        let expected_residual_m = expected_distance_km * 1000.0;
        assert!((residual_m - expected_residual_m).abs() < 1e-9);
    }

    #[test]
    fn hf_pre_residual_failure_does_not_block_post_residual_pass() {
        assert!(!pre_hf_j2_residual_blocks_acceptance(true, 10_000.0, 25.0));
        assert!(post_hf_residual_accepts(10.0, 25.0));
        assert!(!post_hf_residual_accepts(f64::NAN, 25.0));
        assert!(!post_hf_residual_accepts(26.0, 25.0));
    }

    #[test]
    fn strict_hf_missing_runtime_never_falls_back_to_j2() {
        let mut ctx = create_leo_transfer_context();
        ctx.execution_policy = ExecutionPolicy {
            use_high_fidelity: true,
            require_high_fidelity: true,
            allow_parallel: false,
            allow_oxymoo_batch_parallel: false,
            allow_branch_expansion_parallel: false,
            allow_polish_parallel: false,
            allow_anchor_parallel: false,
            allow_deterministic_grid_parallel: false,
        };
        let _missing_runtime = propagate_high_fidelity_state_at_epoch_checked(
            &ctx.dep_equ,
            60.0,
            ctx.epoch_jd,
            ctx.transfer_body_force(),
            &ctx,
        )
        .expect_err("strict HF without runtime assets must reject");

        ctx.execution_policy.require_high_fidelity = false;
        let _optional_runtime = propagate_high_fidelity_state_at_epoch_checked(
            &ctx.dep_equ,
            60.0,
            ctx.epoch_jd,
            ctx.transfer_body_force(),
            &ctx,
        )
        .expect_err("missing optional HF runtime must not fall back to J2");

        ctx.force_config = Some(std::sync::Arc::new(
            lightyear_odeint_rs::types::ForceConfig {
                force_flags: lightyear_odeint_rs::types::ForceFlags::DRAG
                    | lightyear_odeint_rs::types::ForceFlags::SRP
                    | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
                    | lightyear_odeint_rs::types::ForceFlags::MOON_GRAVITY,
                sph_order: 5,
                am_ratio: 0.01,
                cd: 2.2,
                cr: 1.3,
                sun_pos: Some([149_600_000.0, 0.0, 0.0]),
                moon_pos: Some([384_400.0, 0.0, 0.0]),
                ..Default::default()
            },
        ));
        let _incomplete_runtime = propagate_high_fidelity_state_at_epoch_checked(
            &ctx.dep_equ,
            60.0,
            ctx.epoch_jd,
            ctx.transfer_body_force(),
            &ctx,
        )
        .expect_err("incomplete HF runtime assets must not fall back to J2");

        ctx.execution_policy.use_high_fidelity = false;
        assert!(propagate_candidate_state_at_epoch(
            &ctx.dep_equ,
            60.0,
            ctx.epoch_jd,
            ctx.transfer_body_force(),
            &ctx,
        )
        .expect("MF fallback fixture must not overflow diagnostics")
        .is_some());
    }

    #[test]
    fn test_iterative_j2_correction_respects_runtime_iteration_cap() {
        let mut ctx = create_geo_transfer_context();
        let tgt_r = ctx.tgt_sma;
        let tgt_v = (MU / tgt_r).sqrt();
        ctx.tgt_eci = [0.0, tgt_r, 0.0, -tgt_v, 0.0, 0.0];
        eci_to_equinoctial(&ctx.tgt_eci, &mut ctx.tgt_equ);
        ctx.cache_target_orbit();
        ctx.cache_plane_angle();

        let dep_at_release = ctx.dep_eci;
        let tof = 6.0 * 3600.0;
        let mut seeded_sol = lambert_solve_with_target_state(
            tof,
            &ctx,
            &dep_at_release,
            &crate::lambert::LambertR1Cache::new(&[
                dep_at_release[0],
                dep_at_release[1],
                dep_at_release[2],
            ]),
            &crate::lambert_backend::DepartureBoundCache::new(&dep_at_release),
            &ctx.tgt_eci,
        )
        .expect("Lambert fixture must not overflow diagnostics")
        .expect("expected solvable Lambert case");
        seeded_sol.tgt_state[0] += 0.08;
        seeded_sol.tgt_state[1] -= 0.06;
        seeded_sol.tgt_state[2] += 0.04;

        let tight_cap = crate::solve::J2ClosureSettings {
            max_iterations: 1,
            endpoint_target_km: 1.0e-12,
            correction_step_gain: 0.2,
        };
        let wider_cap = crate::solve::J2ClosureSettings {
            max_iterations: 3,
            ..tight_cap
        };
        let (tight_solution, tight_steps, tight_residual_m) =
            apply_iterative_j2_lambert_correction(
                &ctx,
                &dep_at_release,
                tof,
                ctx.epoch_jd,
                seeded_sol,
                tight_cap,
            )
            .expect("tight J2 correction fixture must not overflow diagnostics");
        let (wider_solution, wider_steps, wider_residual_m) =
            apply_iterative_j2_lambert_correction(
                &ctx,
                &dep_at_release,
                tof,
                ctx.epoch_jd,
                seeded_sol,
                wider_cap,
            )
            .expect("wide J2 correction fixture must not overflow diagnostics");

        assert_eq!(tight_steps, 1, "hostile fixture must consume tight cap");
        assert!(wider_steps > tight_steps);
        assert!(wider_steps <= wider_cap.max_iterations);
        assert_ne!(tight_residual_m.to_bits(), wider_residual_m.to_bits());
        assert_ne!(tight_solution.cost.to_bits(), wider_solution.cost.to_bits());
    }

    #[test]
    fn test_brent_local_cache_keeps_first_quantized_value() {
        let mut cache = BrentLocalCache::new();

        assert_eq!(brent_cache_lookup(&cache, 42), None);
        brent_cache_insert_first(&mut cache, 42, 1.25);
        brent_cache_insert_first(&mut cache, 42, 9.50);

        assert_eq!(brent_cache_lookup(&cache, 42), Some(1.25));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn brent_refinement_caches_hold_the_fine_search_bound_inline() {
        // Read the production capacity rather than restating a literal. A
        // local copy would pin the number, not the derivation, so raising the
        // pre-scan ceiling could spill both caches to the heap on every fine
        // search with this test still green.
        const MAX_DISTINCT_CACHE_ENTRIES: usize = BRENT_CACHE_INLINE_CAPACITY;
        assert_eq!(
            BRENT_CACHE_INLINE_CAPACITY,
            1 + BRENT_PRESCAN_MAX_SAMPLES + 1 + BRENT_FINE_MAX_ITERATIONS,
            "the inline capacity must stay the bounded fine search's own entry count"
        );
        let mut cost_cache = BrentLocalCache::new();
        let mut exact_cache = BrentExactSolutionCache::new();

        for index in 0..MAX_DISTINCT_CACHE_ENTRIES {
            let cost_key = i64::try_from(index).expect("fixed cache index must fit i64");
            let exact_key = u64::try_from(index).expect("fixed cache index must fit u64");
            brent_cache_insert_first(&mut cost_cache, cost_key, 0.0);
            exact_cache.push((exact_key, LambertSolutionEx::default()));
        }

        assert_eq!(cost_cache.len(), MAX_DISTINCT_CACHE_ENTRIES);
        assert_eq!(exact_cache.len(), MAX_DISTINCT_CACHE_ENTRIES);
        assert!(
            !cost_cache.spilled(),
            "the bounded fine search must not grow the quantized Brent cache on the heap"
        );
        assert!(
            !exact_cache.spilled(),
            "the bounded fine search must not grow the exact-solution cache on the heap"
        );
    }

    #[test]
    fn scan_sample_counts_reject_extreme_finite_spans() {
        assert_eq!(
            brent_prescan_count(f64::MAX, false, false),
            Err(EvaluationArithmeticOverflow)
        );
        assert_eq!(
            tof_grid_sample_count(f64::MAX, false),
            Err(EvaluationArithmeticOverflow)
        );

        // Fine validation resolves the full interval at 100 s. Coarse ranking
        // covers that same interval at its separately bounded density.
        assert_eq!(brent_prescan_count(6_400.0, false, false), Ok(65));
        assert_eq!(brent_prescan_count(6_400.0, true, false), Ok(7));
        assert_eq!(tof_grid_sample_count(8_000.0, true), Ok(Some(4)));
        assert_eq!(tof_grid_sample_count(6_000.0, false), Ok(Some(5)));
    }

    #[test]
    fn brent_prescan_bracket_keeps_scan_edges_inside_the_original_interval() {
        let first_objective = |tof: f64| Ok::<_, EvaluationArithmeticOverflow>(tof);
        let (first_lo, first_hi, first_tof, first_best_cost) =
            brent_prescan_bracket(0.0, 1_000.0, 5, f64::NAN, f64::INFINITY, first_objective)
                .expect("first-edge scan must not overflow diagnostics");
        assert_eq!(
            [first_lo, first_hi, first_tof, first_best_cost].map(f64::to_bits),
            [0.0, 250.0, 0.0, 0.0].map(f64::to_bits)
        );

        let last_objective = |tof: f64| Ok::<_, EvaluationArithmeticOverflow>(-tof);
        let (last_lo, last_hi, last_tof, last_best_cost) =
            brent_prescan_bracket(0.0, 1_000.0, 5, f64::NAN, f64::INFINITY, last_objective)
                .expect("last-edge scan must not overflow diagnostics");
        assert_eq!(
            [last_lo, last_hi, last_tof, last_best_cost].map(f64::to_bits),
            [750.0, 1_000.0, 1_000.0, -1_000.0].map(f64::to_bits)
        );
    }

    /// Bare Brent picks a basin by where its golden-section start lands; the
    /// pre-scan must pick one by cost. On a two-well cost with the DEEP well
    /// away from the golden start, Brent alone converges to the shallow well
    /// and the pre-scan bracket must exclude it.
    #[test]
    fn brent_prescan_bracket_selects_the_deep_well_brent_alone_misses() {
        // Shallow well at 300 (depth 1.0), deep well at 800 (depth 0.1).
        let cost = |t: f64| {
            let shallow_slope = 0.02 * (t - 300.0).abs();
            let shallow = 1.0 + shallow_slope;
            let deep_slope = 0.02 * (t - 800.0).abs();
            let deep = 0.1 + deep_slope;
            Ok::<_, EvaluationArithmeticOverflow>(shallow.min(deep).min(5.0))
        };

        let bare = minimize_scalar_bounded(cost, 0.0, 1000.0, 1.0, 50)
            .expect("test objective must not overflow diagnostics");
        assert!(bare.converged);
        assert!(
            (bare.x - 300.0).abs() < 50.0,
            "precondition: bare Brent lands in the shallow well, got {}",
            bare.x
        );

        let (lo, hi, best_tof, best_cost) =
            brent_prescan_bracket(0.0, 1000.0, 33, f64::NAN, f64::INFINITY, cost)
                .expect("test prescan must not overflow diagnostics");
        assert!(
            lo <= 800.0 && hi >= 800.0,
            "bracket [{lo}, {hi}] must contain the deep well"
        );
        assert!(
            lo > 400.0,
            "bracket [{lo}, {hi}] must exclude the shallow well"
        );
        assert!((best_tof - 800.0).abs() <= (hi - lo));
        assert!(best_cost < 1.0);

        let refined = minimize_scalar_bounded(cost, lo, hi, 0.5, 50)
            .expect("test objective must not overflow diagnostics");
        assert!(refined.converged);
        assert!(
            (refined.x - 800.0).abs() < 5.0,
            "Brent inside the scan bracket must find the deep well, got {}",
            refined.x
        );
    }

    /// Degenerate inputs must leave the caller's bracket and incumbent alone
    /// rather than narrowing onto nothing.
    #[test]
    fn brent_prescan_bracket_passes_through_degenerate_input() {
        let flat = |_: f64| Ok::<_, EvaluationArithmeticOverflow>(1.0_f64);
        let (lo, hi, tof, cost) = brent_prescan_bracket(10.0, 90.0, 2, 42.0, 7.0, flat)
            .expect("degenerate prescan must not overflow diagnostics");
        assert_eq!((lo, hi, tof, cost), (10.0, 90.0, 42.0, 7.0));

        let (lo, hi, tof, cost) = brent_prescan_bracket(90.0, 90.0, 33, 42.0, 7.0, flat)
            .expect("degenerate prescan must not overflow diagnostics");
        assert_eq!((lo, hi, tof, cost), (90.0, 90.0, 42.0, 7.0));

        // Nothing solvable anywhere in the bracket: no basin to bracket, so
        // Brent keeps the interval it would have had.
        let invalid = |_: f64| Ok::<_, EvaluationArithmeticOverflow>(INVALID_COST);
        let (lo, hi, tof, cost) =
            brent_prescan_bracket(10.0, 90.0, 33, 42.0, f64::INFINITY, invalid)
                .expect("invalid prescan must not overflow diagnostics");
        assert_eq!((lo, hi, tof), (10.0, 90.0, 42.0));
        assert!(cost.is_infinite());
    }

    /// A valid scan returns its own in-bracket candidate; callers preserve a
    /// better incoming incumbent separately.
    #[test]
    fn brent_prescan_bracket_keeps_scan_candidate_inside_its_bracket() {
        let cost = |t: f64| Ok::<_, EvaluationArithmeticOverflow>(1.0 + (t - 500.0).abs() / 1000.0);
        let (lo, hi, tof, best) = brent_prescan_bracket(0.0, 1000.0, 33, 123.0, 0.25, cost)
            .expect("incumbent prescan must not overflow diagnostics");
        assert!(
            lo <= tof && tof <= hi,
            "scan candidate {tof} escaped its returned bracket [{lo}, {hi}]"
        );
        assert_eq!(tof.to_bits(), 500.0_f64.to_bits());
        assert_eq!(best.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn tof_sample_dedup_caps_malformed_count_without_panic() {
        let mut samples = [0.0; MAX_TOF_SAMPLES];
        let mut count = MAX_TOF_SAMPLES
            .checked_add(1)
            .expect("fixed sample capacity must have a successor");

        deduplicate_tof_samples(&mut samples, &mut count);

        assert_eq!(count, 1);
    }

    #[test]
    fn brent_preseed_on_a_degenerate_fixture_stays_degenerate() {
        // Verify that the Brent pre-seed cache doesn't degrade solution quality.
        // Uses a LEO-to-LEO transfer where the bracket span exceeds 60s
        // (the threshold for Brent refinement) and >1s (the pre-seed threshold).
        let mut ctx = create_geo_transfer_context();
        // Widen phase-dv budget so the test geometry is solvable
        ctx.max_phase_dv = 1.5;
        ctx.cache_target_orbit();
        ctx.cache_plane_angle();

        // Evaluate several candidate points that exercise Brent refinement
        let test_points = [[0.12, 1.00, 0.18], [0.15, 1.02, 0.20], [0.10, 0.98, 0.15]];

        for point in &test_points {
            let result_fine = evaluate_plan(point, &ctx, false)
                .expect("fine Brent fixture must not overflow diagnostics");
            let result_coarse = evaluate_plan(point, &ctx, true)
                .expect("coarse Brent fixture must not overflow diagnostics");

            // THIS FIXTURE IS DEGENERATE, AND THAT IS WHY THIS BLOCK LOOKS
            // ODD. Measured 2026-08-06: every point yields
            // `valid == false` in both modes, and one yields cost exactly
            // `INVALID_COST` (1e9) -- the sentinel. The quality assertions
            // that used to sit here were each wrapped in `if result.valid`,
            // so not one of them ever executed: a test named
            // `..._produces_valid_result` passed while producing no valid
            // result at all, and it is the ONLY Brent pre-seed coverage in
            // the workspace. Opening the guards turns it red immediately with
            // "Fine-mode result should have finite cost, got 1000000000".
            //
            // Asserting the degenerate state is deliberate. It cannot claim
            // quality coverage that does not exist, and it converts silent
            // vacuity into a loud prompt: the day this fixture (or the
            // solver) starts producing a valid result, this trips and forces
            // whoever did it to restore the real assertions below.
            for (mode, result) in [("fine", &result_fine), ("coarse", &result_coarse)] {
                assert!(
                    !result.valid,
                    "{mode}-mode result is now VALID on this fixture. That is good news, \
                     and it means this test must stop asserting degeneracy and start \
                     asserting quality: require `cost.is_finite() && cost < INVALID_COST`, \
                     rename away from `degenerate_fixture`, and give the Brent pre-seed \
                     cache the coverage it currently has nowhere else."
                );
                assert!(
                    result.cost.is_finite(),
                    "{mode}-mode cost must stay finite even when invalid, got {}",
                    result.cost
                );
            }
        }
    }
}
