//! Delta-V and cost anchor stage.
//!
//! Before the population search runs, a small set of Nelder-Mead runs is
//! launched from ranked seeds to anchor each objective: the cost anchor drives
//! `TransferLocalProblem`, the delta-V anchors drive `TransferDeltaVProblem`.
//! Both objectives are pure functions of `(x, ctx)`. Independent per-worker
//! phase/orbit scratch lets the parallel driver reproduce the serial rows.

use rayon::prelude::*;

use super::{
    evaluate_plan_local, local_config, map_evaluation_arithmetic_overflow, merge_work_counts,
    nm_max_iters_for_complexity, push_delta_v_anchor_probe_candidates,
    push_unique_transfer_candidate, record_anchor_parallel_runs, record_work_count,
    run_local_optimizer, seed_is_duplicate, should_use_leaf_parallel,
    transfer_candidate_is_objective_finite, try_reserve_transfer_capacity, DeltaVAnchorPolicy,
    EvaluationDiagnosticCounters, InvalidTargetPropagationAuthorityCode, LocalOptimizerKind,
    PlanContext, PlanResult, RefCell, SolveLocalWorkCache, SolverSeed, TransferComplexity,
    TransferDeltaVProblem, TransferLocalOptimizerChoice, TransferLocalProblem, TuneLevel,
    WorkCountCounters, DELTA_V_ANCHOR_SEED_LIMIT_MAX, SINGLE_PAIR_LOWER_BOUNDS,
    SINGLE_PAIR_UPPER_BOUNDS,
};

/// Which local objective a delta-v anchor stage NM run optimizes. The cost
/// anchor drives [`TransferLocalProblem`] (cost objective); the delta-v anchors
/// drive [`TransferDeltaVProblem`] (total-dv objective). Both are pure
/// functions of `(x, ctx)`, so their NM trajectories are independent.
#[derive(Clone, Copy)]
enum AnchorKind {
    Cost,
    DeltaV,
}

/// One anchor optimization to run: its objective kind and precomputed start.
/// The start comes from the ranked seeds (never from a prior anchor's result),
/// so anchors are fully independent and can run in any order / in parallel.
#[derive(Clone, Copy)]
struct AnchorJob {
    kind: AnchorKind,
    start: [f64; 3],
}

/// Shared local-optimizer controls for one anchor pass.
///
/// Keeping these controls together prevents serial, Rayon, and benchmark
/// anchor paths from accidentally forwarding different iteration/seed policy.
#[derive(Clone, Copy)]
pub(super) struct AnchorRunSettings {
    pub(super) max_iters: usize,
    pub(super) tune: TuneLevel,
    pub(super) seed: u64,
    pub(super) warm_start_consumed: bool,
}

/// One ranked delta-v anchor start. A fixed-size ordered array keeps this
/// bounded selection allocation-free for all policy arms.
#[derive(Clone, Copy)]
pub(super) struct DeltaVAnchorStart {
    pub(super) point: [f64; 3],
    pub(super) score: f64,
}

pub(super) fn select_delta_v_anchor_starts(
    ranked_seeds: &[(SolverSeed, PlanResult)],
    cost_anchor: Option<[f64; 3]>,
    policy: DeltaVAnchorPolicy,
) -> [Option<DeltaVAnchorStart>; DELTA_V_ANCHOR_SEED_LIMIT_MAX] {
    let mut starts = [None; DELTA_V_ANCHOR_SEED_LIMIT_MAX];
    if !policy.use_delta_v_anchor() {
        return starts;
    }

    for (candidate_seed, plan) in ranked_seeds {
        if !transfer_candidate_is_objective_finite(plan) {
            continue;
        }
        if cost_anchor
            .as_ref()
            .is_some_and(|existing| seed_is_duplicate(existing, &candidate_seed.x))
        {
            continue;
        }
        if starts
            .iter()
            .flatten()
            .any(|existing| seed_is_duplicate(&existing.point, &candidate_seed.x))
        {
            continue;
        }
        let score = plan.total_dv();
        if !score.is_finite() {
            continue;
        }

        let mut carry = Some(DeltaVAnchorStart {
            point: candidate_seed.x,
            score,
        });
        for slot in starts.iter_mut().take(policy.seed_limit()) {
            let Some(candidate) = carry else {
                break;
            };
            match *slot {
                None => {
                    *slot = Some(candidate);
                    carry = None;
                }
                Some(existing) if candidate.score < existing.score => {
                    *slot = Some(candidate);
                    carry = Some(existing);
                }
                Some(_) => {}
            }
        }
    }

    starts
}

/// The finished NM product of one anchor: the resolved candidate plan (with
/// optimizer bookkeeping fields set) and the fine-stage optimum `fine_x` used to
/// center the probe candidates. The subsequent output push + probe evaluation
/// are performed by the caller (serially, in anchor order) so the cross-anchor
/// output dedup is preserved exactly.
struct AnchorNmOutcome {
    plan: PlanResult,
    fine_x: [f64; 3],
}

pub(super) fn local_optimizer_failure_code(
    error: &anyhow::Error,
) -> InvalidTargetPropagationAuthorityCode {
    if let Some(authority) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<InvalidTargetPropagationAuthorityCode>())
    {
        return *authority;
    }
    if error
        .chain()
        .any(<dyn std::error::Error + 'static>::is::<crate::oxymoo::ArithmeticOverflow>)
    {
        return InvalidTargetPropagationAuthorityCode::ArithmeticOverflow;
    }
    InvalidTargetPropagationAuthorityCode::OptimizerFailure
}

#[cfg(feature = "bench-internal")]
pub(super) fn map_bench_local_optimizer_result<T>(
    result: anyhow::Result<T>,
) -> Result<T, InvalidTargetPropagationAuthorityCode> {
    result.map_err(|error| local_optimizer_failure_code(&error))
}

/// Run one anchor's coarse + fine Nelder-Mead passes against the local cache and
/// resolve the candidate plan. This is the shared kernel of the serial anchor
/// pushers and the parallel fan-out: the operations, order, and work-count
/// accounting are identical on both paths, so switching a run onto a rayon
/// worker only changes which cache/thread executes it, never the values.
fn run_anchor_nm(
    kind: AnchorKind,
    ctx: &PlanContext,
    local_cache: &RefCell<SolveLocalWorkCache>,
    start: [f64; 3],
    settings: AnchorRunSettings,
) -> Result<AnchorNmOutcome, InvalidTargetPropagationAuthorityCode> {
    let doubled_iters = settings
        .max_iters
        .checked_mul(2)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let coarse_iters = (doubled_iters / 3).max(1);
    // A zero requested budget historically normalized both passes to one
    // iteration. Preserve that explicit policy while rejecting all true
    // arithmetic overflow/underflow shapes.
    let remaining_iters = if settings.max_iters == 0 {
        0
    } else {
        settings
            .max_iters
            .checked_sub(coarse_iters)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?
    };
    let fine_iters = remaining_iters.max(1);
    let coarse_config = local_config(
        LocalOptimizerKind::NelderMead,
        coarse_iters,
        settings.tune,
        settings.seed,
    );
    let fine_config = local_config(
        LocalOptimizerKind::NelderMead,
        fine_iters,
        settings.tune,
        settings.seed,
    );

    let coarse = match kind {
        AnchorKind::Cost => run_local_optimizer(
            &TransferLocalProblem {
                ctx,
                cache: local_cache,
                coarse_mode: true,
                gradient_enabled: false,
            },
            SINGLE_PAIR_LOWER_BOUNDS,
            SINGLE_PAIR_UPPER_BOUNDS,
            start,
            coarse_config,
        ),
        AnchorKind::DeltaV => run_local_optimizer(
            &TransferDeltaVProblem {
                ctx,
                cache: local_cache,
                coarse_mode: true,
            },
            SINGLE_PAIR_LOWER_BOUNDS,
            SINGLE_PAIR_UPPER_BOUNDS,
            start,
            coarse_config,
        ),
    }
    .map_err(|error| local_optimizer_failure_code(&error))?;
    // 7.3 work-count audit: one NM run; iterations tracked by optimizer evals.
    let coarse_evaluations = usize::try_from(coarse.evaluations)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    record_work_count(|counters| {
        counters.anchor_nm_runs = counters
            .anchor_nm_runs
            .checked_add(1)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        counters.anchor_nm_iterations = counters
            .anchor_nm_iterations
            .checked_add(coarse_evaluations)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        Ok(())
    })?;

    let fine = match kind {
        AnchorKind::Cost => run_local_optimizer(
            &TransferLocalProblem {
                ctx,
                cache: local_cache,
                coarse_mode: false,
                gradient_enabled: false,
            },
            SINGLE_PAIR_LOWER_BOUNDS,
            SINGLE_PAIR_UPPER_BOUNDS,
            coarse.x,
            fine_config,
        ),
        AnchorKind::DeltaV => run_local_optimizer(
            &TransferDeltaVProblem {
                ctx,
                cache: local_cache,
                coarse_mode: false,
            },
            SINGLE_PAIR_LOWER_BOUNDS,
            SINGLE_PAIR_UPPER_BOUNDS,
            coarse.x,
            fine_config,
        ),
    }
    .map_err(|error| local_optimizer_failure_code(&error))?;
    let fine_evaluations = usize::try_from(fine.evaluations)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    record_work_count(|counters| {
        counters.anchor_nm_runs = counters
            .anchor_nm_runs
            .checked_add(1)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        counters.anchor_nm_iterations = counters
            .anchor_nm_iterations
            .checked_add(fine_evaluations)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        Ok(())
    })?;

    let mut plan = evaluate_plan_local(&fine.x, ctx, false, local_cache)?;
    plan.func_evals = coarse
        .evaluations
        .checked_add(fine.evaluations)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    plan.optimizer_func_evals = plan.func_evals;
    plan.optimizer_converged = fine.converged;
    plan.warm_start_used = settings.warm_start_consumed;
    Ok(AnchorNmOutcome {
        plan,
        fine_x: fine.x,
    })
}
/// Runtime gate for delta-v anchors. Top-level multi-thread calls use global
/// rayon fan-out; calls already on a rayon worker stay leaf-serial.
pub(super) fn should_use_anchor_parallel(ctx: &PlanContext, job_count: usize) -> bool {
    should_use_leaf_parallel(
        ctx.execution_policy.allow_anchor_parallel,
        job_count,
        1,
        rayon::current_num_threads(),
        rayon::current_thread_index().is_none(),
    )
}

/// One worker's finished anchor: its NM outcome, plus the work-count and
/// diagnostic contributions that worker accumulated in isolation.
///
/// The deltas travel with the outcome because the reduction that folds them
/// back is the ordered serial pass in
/// [`push_delta_v_anchor_candidates_parallel`], not the parallel one that
/// produced them.
struct AnchorWorkerResult {
    outcome: AnchorNmOutcome,
    work_delta: WorkCountCounters,
    diag_delta: EvaluationDiagnosticCounters,
}

/// Parallel delta-v anchor stage: run each anchor's coarse+fine NM across the
/// rayon pool on isolated per-worker caches, then replay the candidate pushes
/// and probe evaluations serially in anchor-index order.
///
/// Identity: anchors are independent optimizations from precomputed starts, so
/// each NM trajectory — and therefore `fine_x` and the resolved plan — is
/// bit-identical to the serial reference regardless of its worker scratch.
/// The output push + probe evaluation (whose dedup spans all anchors' `out`
/// entries, and whose probe-evaluation work count depends only on that dedup) run
/// serially in the exact anchor order, so the candidate Vec and probe accounting
/// match serial byte-for-byte. Per-worker work-count / diagnostic contributions
/// are reduced back deterministically in anchor order — which makes the f64
/// diagnostic fields (`j2_correction_residual_m_sum` in metres, the `*_s`
/// sub-timers) schedule-independent, NOT bit-identical to the serial reference:
/// serial accumulates every term into one running sum and this folds per-anchor
/// sums, same order but different grouping, and `+` on f64 is not associative.
/// Integer counters are exact either way.
fn push_delta_v_anchor_candidates_parallel(
    out: &mut Vec<PlanResult>,
    ctx: &PlanContext,
    local_cache: &RefCell<SolveLocalWorkCache>,
    jobs: &[AnchorJob],
    settings: AnchorRunSettings,
    emit_probes: bool,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    // PASS 1 (parallel): each anchor's NM runs in isolation on a fresh
    // per-worker cache with no shared-cache access. Every exit restores its
    // thread-local baselines before the ordered post-join reduction.
    let mut results: Vec<Result<AnchorWorkerResult, InvalidTargetPropagationAuthorityCode>> =
        Vec::new();
    try_reserve_transfer_capacity(&mut results, jobs.len())?;
    jobs.par_iter()
        .map(|job| {
            let worker_cache = RefCell::new(SolveLocalWorkCache::new());
            super::with_isolated_diag_region(|| {
                run_anchor_nm(job.kind, ctx, &worker_cache, job.start, settings)
            })
            .map(|(outcome, diag_delta, work_delta)| AnchorWorkerResult {
                outcome,
                work_delta,
                diag_delta,
            })
        })
        .collect_into_vec(&mut results);

    // PASS 2 (serial, anchor-index order): fold each worker's counter deltas back
    // into the front thread and replay the output pushes + probe evaluations in
    // exact order, preserving the serial candidate order and the cross-anchor
    // output dedup / probe-eval accounting.
    for result in results {
        let result = result?;
        merge_work_counts(result.work_delta)?;
        map_evaluation_arithmetic_overflow(crate::evaluate::merge_evaluation_diagnostics(
            &result.diag_delta,
        ))?;
        record_anchor_parallel_runs(result.work_delta.anchor_nm_runs)?;
        push_unique_transfer_candidate(out, result.outcome.plan)?;
        if emit_probes {
            push_delta_v_anchor_probe_candidates(
                out,
                ctx,
                local_cache,
                result.outcome.fine_x,
                settings.warm_start_consumed,
            )?;
        }
    }
    Ok(())
}

fn push_nm_delta_v_anchor(
    out: &mut Vec<PlanResult>,
    ctx: &PlanContext,
    local_cache: &RefCell<SolveLocalWorkCache>,
    start: [f64; 3],
    settings: AnchorRunSettings,
    emit_probes: bool,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let outcome = run_anchor_nm(AnchorKind::DeltaV, ctx, local_cache, start, settings)?;
    push_unique_transfer_candidate(out, outcome.plan)?;
    if emit_probes {
        push_delta_v_anchor_probe_candidates(
            out,
            ctx,
            local_cache,
            outcome.fine_x,
            settings.warm_start_consumed,
        )?;
    }
    Ok(())
}

fn push_nm_cost_anchor(
    out: &mut Vec<PlanResult>,
    ctx: &PlanContext,
    local_cache: &RefCell<SolveLocalWorkCache>,
    start: [f64; 3],
    settings: AnchorRunSettings,
    emit_probes: bool,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let outcome = run_anchor_nm(AnchorKind::Cost, ctx, local_cache, start, settings)?;
    push_unique_transfer_candidate(out, outcome.plan)?;
    if emit_probes {
        push_delta_v_anchor_probe_candidates(
            out,
            ctx,
            local_cache,
            outcome.fine_x,
            settings.warm_start_consumed,
        )?;
    }
    Ok(())
}

pub(super) fn push_delta_v_anchor_candidates(
    out: &mut Vec<PlanResult>,
    ctx: &PlanContext,
    ranked_seeds: &[(SolverSeed, PlanResult)],
    warm_start_consumed: bool,
    local_cache: &RefCell<SolveLocalWorkCache>,
    policy: DeltaVAnchorPolicy,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    if ranked_seeds.is_empty() {
        return Ok(());
    }

    let complexity = TransferComplexity::classify_from_ctx(ctx);
    let tune = match ctx.local_optimizer.choice {
        TransferLocalOptimizerChoice::Auto => TuneLevel::Default,
        TransferLocalOptimizerChoice::Fixed(_) => ctx.local_optimizer.tune,
    };
    let max_iters = nm_max_iters_for_complexity(complexity);
    let seed = ctx.local_optimizer.seed;
    let settings = AnchorRunSettings {
        max_iters,
        tune,
        seed,
        warm_start_consumed,
    };
    let cost_anchor = ranked_seeds
        .iter()
        .find(|(_, plan)| transfer_candidate_is_objective_finite(plan))
        .map(|(seed, _)| seed.x);
    let starts = select_delta_v_anchor_starts(ranked_seeds, cost_anchor, policy);

    // Build the ordered anchor job list. Order MUST match the serial reference:
    // cost anchor first, then delta-v starts in rank order.
    let emit_probes = policy.use_probes();
    let job_capacity = policy
        .seed_limit()
        .checked_add(1)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let mut jobs = Vec::new();
    try_reserve_transfer_capacity(&mut jobs, job_capacity)?;
    if policy.use_cost_anchor() {
        if let Some(start) = cost_anchor {
            jobs.push(AnchorJob {
                kind: AnchorKind::Cost,
                start,
            });
        }
    }
    for start in starts.into_iter().flatten() {
        jobs.push(AnchorJob {
            kind: AnchorKind::DeltaV,
            start: start.point,
        });
    }

    if should_use_anchor_parallel(ctx, jobs.len()) {
        push_delta_v_anchor_candidates_parallel(
            out,
            ctx,
            local_cache,
            &jobs,
            settings,
            emit_probes,
        )?;
        return Ok(());
    }

    // Serial reference path: run each anchor in order against the shared cache.
    for job in &jobs {
        match job.kind {
            AnchorKind::Cost => {
                push_nm_cost_anchor(out, ctx, local_cache, job.start, settings, emit_probes)?;
            }
            AnchorKind::DeltaV => {
                push_nm_delta_v_anchor(out, ctx, local_cache, job.start, settings, emit_probes)?;
            }
        }
    }
    Ok(())
}
#[cfg(test)]
pub(super) fn run_delta_v_anchor_candidates(
    ctx: &PlanContext,
    ranked_seeds: &[(SolverSeed, PlanResult)],
    warm_start_consumed: bool,
    local_cache: &RefCell<SolveLocalWorkCache>,
) -> Result<Vec<PlanResult>, InvalidTargetPropagationAuthorityCode> {
    let capacity = DeltaVAnchorPolicy::Full
        .seed_limit()
        .checked_add(1)
        .and_then(|count| count.checked_mul(5))
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let mut out = Vec::new();
    try_reserve_transfer_capacity(&mut out, capacity)?;
    push_delta_v_anchor_candidates(
        &mut out,
        ctx,
        ranked_seeds,
        warm_start_consumed,
        local_cache,
        DeltaVAnchorPolicy::Full,
    )?;
    Ok(out)
}
