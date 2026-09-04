//! Delta-V polish of front candidates.
//!
//! Each surviving candidate gets a bounded local re-optimization around its own
//! decision, scoped by an ND-epsilon mask so only candidates that can still move
//! are re-run. The serial and parallel drivers emit the same rows in the same
//! order; the pre-polish snapshot and the pass-1 reuse path exist so the
//! degenerate-front fallback can re-polish the full scope without recomputing
//! what pass 1 already produced.

use rayon::prelude::*;

use super::{
    evaluate_plan_local, local_config, local_delta_v_score, map_evaluation_arithmetic_overflow,
    merge_work_counts, repaired_transfer_decision, run_local_optimizer, should_use_leaf_parallel,
    transfer_candidate_is_objective_finite, transfer_decision_key,
    transfer_moo_constraint_violation, transfer_plan_decision, try_reserve_transfer_capacity,
    EvaluationDiagnosticCounters, FxHashSet, InvalidTargetPropagationAuthorityCode,
    LocalOptimizerConfig, LocalOptimizerKind, PlanContext, PlanResult, PolishScopePolicy, RefCell,
    SolveLocalWorkCache, TransferDeltaVProblem, TuneLevel, WorkCountCounters,
    FINAL_CANDIDATE_POLISH_DV_EPS, FINAL_CANDIDATE_POLISH_ITERS, FINAL_CANDIDATE_POLISH_RADIUS,
    POLISH_SCOPE_CV_TOL, POLISH_SCOPE_ND_EPS_DV_KM_S, POLISH_SCOPE_ND_EPS_TIME_FRAC,
    SINGLE_PAIR_LOWER_BOUNDS, SINGLE_PAIR_UPPER_BOUNDS,
};

/// Runtime gate for delta-V polish. Top-level multi-thread calls use global
/// rayon fan-out; calls already on a rayon worker stay leaf-serial.
pub(super) fn should_use_polish_parallel(ctx: &PlanContext, n_polish: usize) -> bool {
    should_use_leaf_parallel(
        ctx.execution_policy.allow_polish_parallel,
        n_polish,
        POLISH_PARALLEL_MIN_CANDIDATES,
        rayon::current_num_threads(),
        rayon::current_thread_index().is_none(),
    )
}

/// 7.4 parallel Phase B: apply the DuplicateSkip/ScopeSkip flag writes serially
/// (in index order, driving `scope_skipped_count` exactly as the serial loop
/// would), then fan the independent per-candidate polishes out across the rayon
/// pool, then merge results and per-worker counter deltas back serially in
/// ascending index order.
///
/// Independence: given the pre-computed `actions`, each `Polish` candidate's
/// result is a pure function of its pre-polish `PlanResult` and `ctx`. Nothing a
/// polish reads is written by another candidate's polish — there is no shared
/// best-so-far, early-exit, shared budget, or ordering-dependent acceptance
/// (acceptance compares each polished candidate only against its own pre-polish
/// value). Each worker uses its own phase/orbit/Lambert scratch.
pub(super) fn polish_candidates_parallel(
    candidates: &mut [PlanResult],
    actions: &[PolishAction],
    ctx: &PlanContext,
    stats: &mut PolishScopeStats,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    // Serial flag pass (index order): DuplicateSkip/ScopeSkip only. Polish
    // candidates are left untouched so the parallel section can read their
    // pre-polish state through a shared borrow.
    for (candidate, action) in candidates.iter_mut().zip(actions.iter()) {
        match action {
            PolishAction::DuplicateSkip => candidate.polish_skipped = true,
            PolishAction::ScopeSkip => {
                candidate.polish_skipped = true;
                stats.scope_skipped_count = stats
                    .scope_skipped_count
                    .checked_add(1)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            }
            PolishAction::Polish => {}
        }
    }

    // PASS (parallel): each worker polishes one candidate against its OWN fresh
    // work cache — no shared-cache access — and captures its work-count /
    // diagnostic-counter delta so the front-solve thread's stage tallies stay
    // exact after a serial reduction.
    let mut candidates_to_polish = Vec::new();
    try_reserve_transfer_capacity(&mut candidates_to_polish, candidates.len())?;
    for (candidate, action) in candidates.iter().zip(actions) {
        if matches!(action, PolishAction::Polish) {
            candidates_to_polish.push(candidate);
        }
    }
    let mut computed = Vec::new();
    try_reserve_transfer_capacity(&mut computed, candidates_to_polish.len())?;
    candidates_to_polish
        .par_iter()
        .map(|candidate| {
            let local_cache = RefCell::new(SolveLocalWorkCache::new());
            super::with_isolated_diag_region(|| {
                polish_transfer_candidate_delta_v(candidate, ctx, &local_cache)
            })
            .map(|(polished, diag_delta, work_delta)| PolishOutput {
                polished,
                work_delta,
                diag_delta,
            })
        })
        .collect_into_vec(&mut computed);
    let polish_count = computed.len();

    // Serial reduction + writeback (ascending index order): merge counter
    // deltas and replace each polished candidate in place.
    let polished_candidates =
        candidates
            .iter_mut()
            .zip(actions)
            .filter_map(|(candidate, action)| {
                matches!(action, PolishAction::Polish).then_some(candidate)
            });
    for (candidate, out) in polished_candidates.zip(computed) {
        let out = out?;
        merge_work_counts(out.work_delta)?;
        map_evaluation_arithmetic_overflow(crate::evaluate::merge_evaluation_diagnostics(
            &out.diag_delta,
        ))?;
        if let Some(polished) = out.polished {
            let pre_polish_dv = candidate.total_dv();
            *candidate = polished;
            let improvement = pre_polish_dv - candidate.total_dv();
            if improvement.is_finite() && improvement > stats.dv_improvement_max_km_s {
                stats.dv_improvement_max_km_s = improvement;
            }
        }
    }

    stats.polish_parallel_count = stats
        .polish_parallel_count
        .checked_add(polish_count)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    Ok(())
}

pub(super) fn polish_transfer_candidates_delta_v(
    candidates: &mut [PlanResult],
    ctx: &PlanContext,
    cache: &RefCell<SolveLocalWorkCache>,
    scope_policy: PolishScopePolicy,
) -> Result<PolishScopeStats, InvalidTargetPropagationAuthorityCode> {
    Ok(polish_transfer_candidates_delta_v_with_pre_polish_snapshot(
        candidates,
        ctx,
        cache,
        scope_policy,
        false,
    )?
    .0)
}

/// Final delta-V polish with an optional lazily-taken pre-polish snapshot.
///
/// Perf #12: the degenerate-front safety net in the `VerifiedSuperset` front
/// solve needs the PRE-polish pool (polish mutates candidates in place, so
/// it is unrecoverable afterwards), but it can only fire when scoped polish
/// actually skipped at least one candidate (`scope_skipped_count > 0`) — a
/// condition documented as never occurring alongside a degenerate front on
/// healthy shapes. Historically the whole pool was eagerly deep-cloned on
/// every `NdEpsilon` superset solve.
///
/// This variant first classifies every candidate (Phase A). Classification
/// is a pure function of the pre-polish pool: the duplicate-decision set is
/// determined by the ordered pre-polish decision keys (in the historical
/// single-pass loop each key was inserted before its candidate could be
/// mutated), and the epsilon mask is computed once up front from pre-polish
/// values. Phase A therefore proves the exact final `scope_skipped_count`
/// BEFORE any mutation, so the snapshot clone can be skipped entirely when
/// that count is zero, while a snapshot that is taken is still captured
/// pre-mutation and is byte-identical to the historical eager clone.
/// Phase B then applies the actions in the original index order, so every
/// flag write, in-place polish replacement, and polish/cache call happens in
/// exactly the historical sequence.
pub(super) fn polish_transfer_candidates_delta_v_with_pre_polish_snapshot(
    candidates: &mut [PlanResult],
    ctx: &PlanContext,
    cache: &RefCell<SolveLocalWorkCache>,
    scope_policy: PolishScopePolicy,
    want_pre_polish_snapshot: bool,
) -> Result<(PolishScopeStats, Option<Vec<PlanResult>>), InvalidTargetPropagationAuthorityCode> {
    let scope_skip =
        match scope_policy {
            PolishScopePolicy::Full => None,
            PolishScopePolicy::NdEpsilon => Some(polish_scope_nd_epsilon_mask(
                candidates,
                ctx,
                POLISH_SCOPE_ND_EPS_DV_KM_S,
            )?),
            PolishScopePolicy::NdEpsilonTuned { dv_eps_m_per_s } => Some(
                polish_scope_nd_epsilon_mask(candidates, ctx, f64::from(dv_eps_m_per_s) / 1000.0)?,
            ),
        };

    // Phase A: pure classification; no candidate is mutated yet.
    let mut seen_decisions = FxHashSet::default();
    seen_decisions
        .try_reserve(candidates.len())
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let mut actions = Vec::new();
    try_reserve_transfer_capacity(&mut actions, candidates.len())?;
    let mut planned_scope_skips = 0_usize;
    for (index, candidate) in candidates.iter().enumerate() {
        let mut duplicate = false;
        if transfer_candidate_is_objective_finite(candidate) {
            let start = repaired_transfer_decision(&transfer_plan_decision(candidate));
            if start.iter().all(|value| value.is_finite()) {
                duplicate = !seen_decisions.insert(transfer_decision_key(&start));
            }
        }
        let action = if duplicate {
            PolishAction::DuplicateSkip
        } else if scope_skip
            .as_ref()
            .and_then(|mask| mask.get(index))
            .is_some_and(|skip| *skip)
        {
            planned_scope_skips = planned_scope_skips
                .checked_add(1)
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            PolishAction::ScopeSkip
        } else {
            PolishAction::Polish
        };
        actions.push(action);
    }

    // Lazy pre-polish snapshot (see doc comment): the loop below produces
    // scope_skipped_count == planned_scope_skips, so whenever the caller's
    // degenerate-front fallback is reachable a snapshot exists.
    let snapshot = if want_pre_polish_snapshot && planned_scope_skips > 0 {
        let mut snapshot = Vec::new();
        try_reserve_transfer_capacity(&mut snapshot, candidates.len())?;
        snapshot.extend_from_slice(candidates);
        Some(snapshot)
    } else {
        None
    };

    // Phase B: apply the actions in the original index order.
    let mut stats = PolishScopeStats {
        scope_skipped_count: 0,
        dv_improvement_max_km_s: 0.0,
        polish_parallel_count: 0,
    };

    // Fan independent per-candidate polishes across the global rayon pool only
    // for a top-level multi-thread caller. Nested outer-worker calls use the
    // byte-identical serial loop below.
    {
        let n_polish = actions
            .iter()
            .filter(|action| matches!(action, PolishAction::Polish))
            .count();
        if should_use_polish_parallel(ctx, n_polish) {
            polish_candidates_parallel(candidates, &actions, ctx, &mut stats)?;
            debug_assert_eq!(stats.scope_skipped_count, planned_scope_skips);
            return Ok((stats, snapshot));
        }
    }

    for (candidate, action) in candidates.iter_mut().zip(actions) {
        match action {
            PolishAction::DuplicateSkip => {
                candidate.polish_skipped = true;
            }
            PolishAction::ScopeSkip => {
                candidate.polish_skipped = true;
                stats.scope_skipped_count = stats
                    .scope_skipped_count
                    .checked_add(1)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            }
            PolishAction::Polish => {
                let pre_polish_dv = candidate.total_dv();
                if let Some(polished) = polish_transfer_candidate_delta_v(candidate, ctx, cache)? {
                    *candidate = polished;
                    let improvement = pre_polish_dv - candidate.total_dv();
                    if improvement.is_finite() && improvement > stats.dv_improvement_max_km_s {
                        stats.dv_improvement_max_km_s = improvement;
                    }
                }
            }
        }
    }
    debug_assert_eq!(stats.scope_skipped_count, planned_scope_skips);
    Ok((stats, snapshot))
}

pub(super) fn final_candidate_polish_bounds(center: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let [time, phase, wait] = *center;
    let [time_radius, phase_radius, wait_radius] = FINAL_CANDIDATE_POLISH_RADIUS;
    let [time_lower, phase_lower, wait_lower] = SINGLE_PAIR_LOWER_BOUNDS;
    let [time_upper, phase_upper, wait_upper] = SINGLE_PAIR_UPPER_BOUNDS;
    (
        [
            (time - time_radius).max(time_lower),
            (phase - phase_radius).max(phase_lower),
            (wait - wait_radius).max(wait_lower),
        ],
        [
            (time + time_radius).min(time_upper),
            (phase + phase_radius).min(phase_upper),
            (wait + wait_radius).min(wait_upper),
        ],
    )
}

pub(super) fn polish_transfer_candidate_delta_v(
    candidate: &PlanResult,
    ctx: &PlanContext,
    cache: &RefCell<SolveLocalWorkCache>,
) -> Result<Option<PlanResult>, InvalidTargetPropagationAuthorityCode> {
    if !transfer_candidate_is_objective_finite(candidate) {
        return Ok(None);
    }
    let source_decision = transfer_plan_decision(candidate);
    let start = repaired_transfer_decision(&source_decision);
    if !start.iter().all(|value| value.is_finite()) {
        return Ok(None);
    }

    let (lower, upper) = final_candidate_polish_bounds(&start);
    let problem = TransferDeltaVProblem {
        ctx,
        cache,
        coarse_mode: false,
    };
    let result = match run_local_optimizer(
        &problem,
        lower,
        upper,
        start,
        LocalOptimizerConfig {
            // Both bounds, so the count is exact rather than whatever the tune
            // factor and the shared floor happen to produce. See the constant.
            min_iters: FINAL_CANDIDATE_POLISH_ITERS,
            ..local_config(
                LocalOptimizerKind::NelderMead,
                FINAL_CANDIDATE_POLISH_ITERS,
                TuneLevel::Default,
                ctx.local_optimizer.seed,
            )
        },
    ) {
        Ok(result) => result,
        Err(error) => {
            if let Some(authority) = error.downcast_ref::<InvalidTargetPropagationAuthorityCode>() {
                return Err(*authority);
            }
            if error
                .downcast_ref::<crate::oxymoo::ArithmeticOverflow>()
                .is_some()
            {
                return Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow);
            }
            return Err(InvalidTargetPropagationAuthorityCode::OptimizerFailure);
        }
    };

    let mut polished = evaluate_plan_local(&result.x, ctx, false, cache)?;
    if !transfer_candidate_is_objective_finite(&polished) {
        return Ok(None);
    }
    let original_cv = transfer_moo_constraint_violation(&start, candidate, ctx);
    let polished_cv = transfer_moo_constraint_violation(&result.x, &polished, ctx);
    if polished_cv > original_cv + 1e-12 {
        return Ok(None);
    }
    let candidate_dv = candidate.total_dv();
    let result_dv = polished.total_dv();
    if result_dv > candidate_dv + FINAL_CANDIDATE_POLISH_DV_EPS {
        return Ok(None);
    }
    let candidate_score = local_delta_v_score(&start, candidate, ctx);
    let result_score = local_delta_v_score(&result.x, &polished, ctx);
    if result_score > candidate_score + FINAL_CANDIDATE_POLISH_DV_EPS {
        return Ok(None);
    }

    polished.func_evals = candidate
        .func_evals
        .checked_add(result.evaluations)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    polished.optimizer_func_evals = candidate
        .optimizer_func_evals
        .checked_add(result.evaluations)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    polished.optimizer_converged = candidate.optimizer_converged || result.converged;
    polished.warm_start_used = candidate.warm_start_used;
    polished.polish_steps = candidate
        .polish_steps
        .checked_add(1)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    polished.polish_evals = candidate
        .polish_evals
        .checked_add(result.evaluations)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    polished.polish_time_us = candidate.polish_time_us;
    polished.polish_skipped = false;
    polished.escape_triggered = candidate.escape_triggered;
    Ok(Some(polished))
}

#[derive(Clone, Copy)]
struct PolishScopePoint {
    delta_v: f64,
    total_time: f64,
    constraint_violation: f64,
}

/// Epsilon-dominance polish-scope mask: `true` at index i means candidate i
/// is skipped by `PolishScopePolicy::NdEpsilon` (or its tuned variant).
/// Candidate i is skipped iff some candidate j is at least `dv_eps_km_s`
/// better on total dv AND at least `POLISH_SCOPE_ND_EPS_TIME_FRAC` better on
/// total time at no-worse constraint violation. `dv_eps_km_s` is
/// `POLISH_SCOPE_ND_EPS_DV_KM_S` for `NdEpsilon` and the token-supplied
/// margin for `NdEpsilonTuned` (see the campaign plan on that variant).
/// Heuristic, not a guarantee: measured polish rescues exceed these margins
/// by orders of magnitude, so a skipped candidate can forgo front-relevant
/// refinement — the policy is justified empirically (multi-seed HV A/B),
/// see the margin constants' note.
fn polish_scope_nd_epsilon_mask(
    candidates: &[PlanResult],
    ctx: &PlanContext,
    dv_eps_km_s: f64,
) -> Result<Vec<bool>, InvalidTargetPropagationAuthorityCode> {
    let mut points = Vec::new();
    try_reserve_transfer_capacity(&mut points, candidates.len())?;
    for candidate in candidates {
        let point = if transfer_candidate_is_objective_finite(candidate) {
            let decision = transfer_plan_decision(candidate);
            let point = PolishScopePoint {
                delta_v: candidate.total_dv(),
                total_time: candidate.total_time(),
                constraint_violation: transfer_moo_constraint_violation(&decision, candidate, ctx),
            };
            (point.delta_v.is_finite()
                && point.total_time.is_finite()
                && point.constraint_violation.is_finite())
            .then_some(point)
        } else {
            None
        };
        points.push(point);
    }

    let mut skip_mask = Vec::new();
    try_reserve_transfer_capacity(&mut skip_mask, points.len())?;
    for (candidate_index, current) in points.iter().enumerate() {
        let skip = current.is_some_and(|current| {
            points
                .iter()
                .enumerate()
                .any(|(other_index, other)| match other {
                    Some(other) if candidate_index != other_index => {
                        other.constraint_violation
                            <= current.constraint_violation + POLISH_SCOPE_CV_TOL
                            && other.delta_v <= current.delta_v - dv_eps_km_s
                            && other.total_time
                                <= current.total_time * (1.0 - POLISH_SCOPE_ND_EPS_TIME_FRAC)
                    }
                    _ => false,
                })
        });
        skip_mask.push(skip);
    }
    Ok(skip_mask)
}

pub(super) struct PolishScopeStats {
    pub(super) scope_skipped_count: usize,
    pub(super) dv_improvement_max_km_s: f64,
    /// 7.4 work-count audit: number of candidates whose delta-V polish ran on
    /// the rayon fan-out path (0 on the serial reference path).
    pub(super) polish_parallel_count: usize,
}

/// Per-candidate action for the final delta-V polish pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PolishAction {
    /// Exact duplicate repaired decision: flag `polish_skipped`, no polish.
    DuplicateSkip,
    /// `NdEpsilon` scope-mask skip: flag `polish_skipped`, count, no polish.
    ScopeSkip,
    /// Run the local polish.
    Polish,
}

struct PolishOutput {
    polished: Option<PlanResult>,
    work_delta: WorkCountCounters,
    diag_delta: EvaluationDiagnosticCounters,
}

/// 7.4: below this many `PolishAction::Polish` candidates the fan-out overhead
/// is not worth it, so the polish stage stays on the serial reference path. The
/// front-solve typically presents 16-28 candidates, so real solves clear this.
pub(super) const POLISH_PARALLEL_MIN_CANDIDATES: usize = 2;
