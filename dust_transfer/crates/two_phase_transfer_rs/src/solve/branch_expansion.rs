//! Lambert branch expansion over the verified-superset candidate pool.
//!
//! Each surviving source decision is re-evaluated with the multi-revolution
//! Lambert branches enabled, emitting one row per admissible branch. The serial
//! and parallel drivers here produce the same rows in the same order; the
//! parallel one exists only to move the per-source physics off the calling
//! thread.

use rayon::prelude::*;

use super::{
    branch_expansion_capacity, branch_rows_per_source_percentiles, checked_stage_metric_count_add,
    evaluate_plan_branches_with_scratch, map_evaluation_arithmetic_overflow, merge_work_counts,
    repaired_transfer_decision, should_use_leaf_parallel, transfer_decision_key,
    try_reserve_transfer_capacity, EvaluationDiagnosticCounters, FxHashSet,
    InvalidTargetPropagationAuthorityCode, PlanContext, PlanResult, StageTimer,
    VerifiedSupersetStageMetrics, WorkCountCounters, INVALID_COST,
};

pub(super) fn expand_lambert_branch_candidates_for_superset(
    ctx: &PlanContext,
    candidates: Vec<PlanResult>,
    metrics: &mut VerifiedSupersetStageMetrics,
) -> Result<(Vec<PlanResult>, f64), InvalidTargetPropagationAuthorityCode> {
    let expansion_sources =
        branch_expansion_sources_unique_by_repaired_decision_indexed(candidates)?;
    metrics.branch_source_count = expansion_sources.len();
    if ctx.max_revs.max(0) == 0
        && expansion_sources.iter().all(|(_, source)| {
            source.valid
                && source.cost < INVALID_COST
                && source.best_M == 0
                && source.branch_rev == 0
                && source.branch_low_path
        })
    {
        // Early-return path: each source emits exactly itself; no per-source
        // Lambert evaluation happens.
        let source_count = expansion_sources.len();
        metrics.branch_rows_per_source_p50 = usize::from(source_count > 0);
        metrics.branch_rows_per_source_p95 = usize::from(source_count > 0);
        metrics.branch_rows_per_source_max = usize::from(source_count > 0);
        checked_stage_metric_count_add(&mut metrics.branch_emitted_count, source_count)?;
        let mut expanded = Vec::new();
        try_reserve_transfer_capacity(&mut expanded, source_count)?;
        for (_, source) in expansion_sources {
            expanded.push(source);
        }
        return Ok((expanded, 0.0));
    }
    {
        if should_use_branch_expansion_parallel(ctx, expansion_sources.len()) {
            return expand_lambert_branch_candidates_parallel(ctx, &expansion_sources, metrics);
        }
    }
    expand_lambert_branch_candidates_serial(ctx, expansion_sources, metrics)
}

/// Keep-first dedup key for expanded branch plans: the decision triple plus
/// the branch identity and total delta-v, all f64 fields by exact bits.
type BranchPlanDedupKey = (u64, u64, u64, i32, bool, u64, u64);

/// Shared per-plan bookkeeping for both branch-expansion drivers: stamp the
/// source's optimizer provenance onto the branch plan, build the keep-first
/// dedup key, and push first-seen plans. The serial reference and the 7.4
/// parallel replay stay byte-identical by construction because both arms run
/// this exact body.
#[expect(
    clippy::inline_always,
    reason = "hot inner-loop body extracted purely for dedup; this workspace has \
              measured a function boundary on an extracted stage costing ~10%, so \
              the extraction must not introduce one"
)]
#[inline(always)]
fn stamp_and_push_branch_plan(
    mut plan: PlanResult,
    source: &PlanResult,
    seen: &mut FxHashSet<BranchPlanDedupKey>,
    expanded: &mut Vec<PlanResult>,
) {
    plan.func_evals = source.func_evals;
    plan.optimizer_func_evals = source.optimizer_func_evals;
    plan.optimizer_converged = source.optimizer_converged;
    plan.warm_start_used = source.warm_start_used;
    let key = (
        plan.time2phase_ratio.to_bits(),
        plan.phase_sma_ratio.to_bits(),
        plan.waittime_ratio.to_bits(),
        plan.branch_rev,
        plan.branch_low_path,
        plan.branch_tof_s.to_bits(),
        plan.total_dv().to_bits(),
    );
    if seen.insert(key) {
        expanded.push(plan);
    }
}

/// Serial reference path for the verified-superset Lambert branch expansion:
/// one shared Lambert scratch, in-source-order pushes with keep-first dedup.
/// This is the byte-identical reference the 7.4 parallel path reproduces.
pub(super) fn expand_lambert_branch_candidates_serial(
    ctx: &PlanContext,
    expansion_sources: Vec<(usize, PlanResult)>,
    metrics: &mut VerifiedSupersetStageMetrics,
) -> Result<(Vec<PlanResult>, f64), InvalidTargetPropagationAuthorityCode> {
    let expanded_capacity = branch_expansion_capacity(expansion_sources.len(), ctx.max_revs)?;
    let mut expanded = Vec::new();
    try_reserve_transfer_capacity(&mut expanded, expanded_capacity)?;
    // rust-alloc#4 (safe half): order-independent membership dedup feeding a
    // separately-sorted Vec, so FxHashSet drops the per-key BTree node
    // allocations with no ordering effect.
    let mut seen = FxHashSet::default();
    seen.try_reserve(expanded_capacity)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let mut branch_eval_s = 0.0_f64;
    let mut branch_rows_per_source = Vec::new();
    try_reserve_transfer_capacity(&mut branch_rows_per_source, expansion_sources.len())?;
    // rust-alloc#2: one Lambert scratch for the whole expansion loop.
    let mut lambert_scratch = crate::lambert::VariableR2LambertScratch::default();
    for (_, source) in expansion_sources {
        let x = [
            source.time2phase_ratio,
            source.phase_sma_ratio,
            source.waittime_ratio,
        ];
        if !x.iter().all(|value| value.is_finite()) {
            branch_rows_per_source.push(0);
            continue;
        }
        let branch_eval_start = StageTimer::start();
        let branch_plans = map_evaluation_arithmetic_overflow(
            evaluate_plan_branches_with_scratch(&x, ctx, false, &mut lambert_scratch),
        )?;
        branch_eval_s += branch_eval_start.elapsed_s();
        branch_rows_per_source.push(branch_plans.len());
        for plan in branch_plans {
            stamp_and_push_branch_plan(plan, &source, &mut seen, &mut expanded);
        }
    }
    let (p50, p95, max) = branch_rows_per_source_percentiles(&mut branch_rows_per_source);
    metrics.branch_rows_per_source_p50 = p50;
    metrics.branch_rows_per_source_p95 = p95;
    metrics.branch_rows_per_source_max = max;
    Ok((expanded, branch_eval_s))
}

/// 7.4: minimum source count below which fanning branch expansion out is not
/// worth the rayon dispatch. Multi-rev verified-superset fronts clear this.
pub(super) const BRANCH_EXPANSION_PARALLEL_MIN_SOURCES: usize = 2;

/// Runtime gate for branch expansion. Top-level multi-thread calls use global
/// rayon fan-out; calls already on a rayon worker stay leaf-serial.
pub(super) fn should_use_branch_expansion_parallel(ctx: &PlanContext, source_count: usize) -> bool {
    should_use_leaf_parallel(
        ctx.execution_policy.allow_branch_expansion_parallel,
        source_count,
        BRANCH_EXPANSION_PARALLEL_MIN_SOURCES,
        rayon::current_num_threads(),
        rayon::current_thread_index().is_none(),
    )
}

struct BranchSourceResult {
    branch_plans: Vec<PlanResult>,
    finite: bool,
    eval_s: f64,
    diag_delta: EvaluationDiagnosticCounters,
    work_delta: WorkCountCounters,
}

/// 7.4 parallel Lambert branch expansion: compute each source's branch plans
/// across the rayon pool with a per-worker Lambert scratch (bounded memory:
/// scratch per worker, not per source), then perform the pushes / keep-first
/// dedup / metric accumulation in a serial source-index pass so the expanded
/// `Vec` and every INTEGER stage counter are bit-identical to the serial
/// reference.
///
/// Per-source evaluation is pure given `ctx` (the `_with_scratch` Lambert batch
/// entries clear every scratch field at entry, exactly as the serial single
/// shared scratch already relies on), so the ordered flatten reproduces the
/// serial push order. Each source captures its own diagnostic/work counter
/// contribution; the front-solve thread folds them in source-index order after
/// the join, keeping integer counts exact and f64 sub-timer sums
/// reduction-order deterministic (schedule-independent) but NOT bit-identical
/// to the serial reference — same order, different grouping, and `+` on f64 is
/// not associative.
pub(super) fn expand_lambert_branch_candidates_parallel(
    ctx: &PlanContext,
    expansion_sources: &[(usize, PlanResult)],
    metrics: &mut VerifiedSupersetStageMetrics,
) -> Result<(Vec<PlanResult>, f64), InvalidTargetPropagationAuthorityCode> {
    // PASS 1 (parallel): per-source branch physics off-thread with a per-worker
    // Lambert scratch and no shared mutable state. Each worker captures its own
    // diagnostic/work counter delta so the reduction can be replayed serially.
    let mut computed = Vec::new();
    try_reserve_transfer_capacity(&mut computed, expansion_sources.len())?;
    expansion_sources
        .par_iter()
        .map_init(
            crate::lambert::VariableR2LambertScratch::default,
            |scratch, (_, source)| {
                super::with_isolated_diag_region(|| {
                    let x = [
                        source.time2phase_ratio,
                        source.phase_sma_ratio,
                        source.waittime_ratio,
                    ];
                    if !x.iter().all(|value| value.is_finite()) {
                        return Ok(None);
                    }
                    let eval_start = StageTimer::start();
                    let branch_plans = map_evaluation_arithmetic_overflow(
                        evaluate_plan_branches_with_scratch(&x, ctx, false, scratch),
                    )?;
                    Ok(Some((branch_plans, eval_start.elapsed_s())))
                })
                .map(|(payload, diag_delta, work_delta)| match payload {
                    Some((branch_plans, eval_s)) => BranchSourceResult {
                        branch_plans,
                        finite: true,
                        eval_s,
                        diag_delta,
                        work_delta,
                    },
                    // Non-finite source: nothing ran inside the zeroed region,
                    // so the captured deltas are exactly the defaults the
                    // hand-written early return used to stamp.
                    None => BranchSourceResult {
                        branch_plans: Vec::new(),
                        finite: false,
                        eval_s: 0.0,
                        diag_delta,
                        work_delta,
                    },
                })
            },
        )
        .collect_into_vec(&mut computed);

    // PASS 2 (serial, source-index order): reduce the per-source counter deltas
    // and replay the serial pushes / dedup / metric accumulation exactly.
    let expanded_capacity = branch_expansion_capacity(expansion_sources.len(), ctx.max_revs)?;
    let mut expanded = Vec::new();
    try_reserve_transfer_capacity(&mut expanded, expanded_capacity)?;
    let mut seen = FxHashSet::default();
    seen.try_reserve(expanded_capacity)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let mut branch_eval_s = 0.0_f64;
    let mut branch_rows_per_source = Vec::new();
    try_reserve_transfer_capacity(&mut branch_rows_per_source, expansion_sources.len())?;
    let mut diag_total = EvaluationDiagnosticCounters::default();
    let mut work_total = WorkCountCounters::default();
    for ((_, source), result) in expansion_sources.iter().zip(computed) {
        let result = result?;
        // Fold in source-index order so f64 sub-timer sums are reduction-order
        // deterministic and integer counts match the serial running totals.
        map_evaluation_arithmetic_overflow(diag_total.add_delta(&result.diag_delta))?;
        work_total.add_delta(result.work_delta)?;
        branch_eval_s += result.eval_s;
        checked_stage_metric_count_add(&mut metrics.branch_parallel_count, 1)?;
        if !result.finite {
            branch_rows_per_source.push(0);
            continue;
        }
        branch_rows_per_source.push(result.branch_plans.len());
        for plan in result.branch_plans {
            stamp_and_push_branch_plan(plan, source, &mut seen, &mut expanded);
        }
    }
    // Every worker restores its thread-local baselines before this ordered
    // reduction, including the Rayon driving thread and every error exit.
    map_evaluation_arithmetic_overflow(crate::evaluate::merge_evaluation_diagnostics(&diag_total))?;
    merge_work_counts(work_total)?;
    let (p50, p95, max) = branch_rows_per_source_percentiles(&mut branch_rows_per_source);
    metrics.branch_rows_per_source_p50 = p50;
    metrics.branch_rows_per_source_p95 = p95;
    metrics.branch_rows_per_source_max = max;
    Ok((expanded, branch_eval_s))
}

#[cfg(test)]
pub(super) fn branch_expansion_sources_unique_by_repaired_decision(
    candidates: Vec<PlanResult>,
) -> Result<Vec<PlanResult>, InvalidTargetPropagationAuthorityCode> {
    let sources = branch_expansion_sources_unique_by_repaired_decision_indexed(candidates)?;
    let mut output = Vec::new();
    try_reserve_transfer_capacity(&mut output, sources.len())?;
    for (_, source) in sources {
        output.push(source);
    }
    Ok(output)
}

/// Source dedup for the Lambert branch expansion, retaining each survivor's
/// ORIGINAL pool index (its position in the input candidate pool) alongside the
/// candidate. Ignoring the index reproduces the historical `Vec<PlanResult>`
/// dedup exactly (same survivors, same order).
pub(super) fn branch_expansion_sources_unique_by_repaired_decision_indexed(
    candidates: Vec<PlanResult>,
) -> Result<Vec<(usize, PlanResult)>, InvalidTargetPropagationAuthorityCode> {
    let mut sources = Vec::new();
    try_reserve_transfer_capacity(&mut sources, candidates.len())?;
    let mut seen = FxHashSet::default();
    seen.try_reserve(candidates.len())
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    for (pool_index, mut source) in candidates.into_iter().enumerate() {
        let raw = [
            source.time2phase_ratio,
            source.phase_sma_ratio,
            source.waittime_ratio,
        ];
        if !raw.iter().all(|value| value.is_finite()) {
            continue;
        }
        let repaired = repaired_transfer_decision(&raw);
        let key = transfer_decision_key(&repaired);
        if !seen.insert(key) {
            continue;
        }
        let repair_changed = raw
            .iter()
            .zip(repaired.iter())
            .any(|(before, after)| before.to_bits() != after.to_bits());
        if repair_changed {
            source.valid = false;
        }
        source.time2phase_ratio = repaired[0];
        source.phase_sma_ratio = repaired[1];
        source.waittime_ratio = repaired[2];
        sources.push((pool_index, source));
    }
    Ok(sources)
}
