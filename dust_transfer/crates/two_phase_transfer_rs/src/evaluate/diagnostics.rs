//! Evaluator-owned diagnostic counters and HF propagation telemetry.
//!
//! Every counter update and cross-worker reduction is checked: an overflow is
//! typed as [`EvaluationArithmeticOverflow`] and never becomes a clamped
//! diagnostic that could seal partial work. Nothing in this module touches
//! physics -- it only accounts for work the evaluator already did.

use std::cell::RefCell;

use crate::types::counter_roster;

counter_roster! {
    error = EvaluationArithmeticOverflow;
    overflow = EvaluationArithmeticOverflow;
    sub = test;
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct EvaluationDiagnosticCounters {
        count lambert_batch_call_count: usize,
        count lambert_batch_row_count: usize,
        count lambert_batch_simd_lane_solve_count: usize,
        count lambert_batch_scalar_variant_solve_count: usize,
        count lambert_scalar_tof_count: usize,
        count lambert_branch_attempt_count: usize,
        count lambert_branch_valid_count: usize,
        count lambert_branch_rev0_count: usize,
        count lambert_branch_rev_gt0_count: usize,
        count lambert_branch_low_path_count: usize,
        count lambert_branch_high_path_count: usize,
        count lambert_branch_prograde_count: usize,
        count lambert_branch_retrograde_count: usize,
        count lambert_max_revs_gt0_call_count: usize,
        count lambert_branch_selection_call_count: usize,
        count target_j2_batch_state_count: usize,
        count target_j2_simd4_chunk_count: usize,
        count target_j2_scalar_state_count: usize,
        count j2_propagate_state_count: usize,
        /// Lookups of the per-thread phase-state sub-cache that were SERVED from
        /// the cache, and lookups that were not.
        ///
        /// These two exist as a pair on purpose. `j2_propagate_state_count` alone
        /// is a bare miss-side tally: a phase-state cache miss reaches
        /// `propagate_candidate_state_at_epoch` and increments it unless one of
        /// that function's guards returns first, and six other call sites that
        /// never consult this cache increment it too. So nothing about it is
        /// conserved, and its run-to-run wobble under a leaf fan-out has no
        /// denominator to be judged against. The pair does have one — `hit + miss`
        /// is the number of lookups the search REQUESTED,
        /// which is a property of the problem and not of the work-stealing
        /// partition. `tests/width_identity.rs` asserts exactly that, the same
        /// shape as `oxymoo_eval_cache_hit_count` / `oxymoo_eval_cache_miss_count`.
        ///
        /// Both are incremented at the single lookup site
        /// (`solve::evaluate_plan_local`, keyed on
        /// `time2phase_ratio.to_bits()`), so the sum cannot drift away from the
        /// lookup count without that site being edited.
        count phase_state_cache_hit_count: usize,
        count phase_state_cache_miss_count: usize,
        /// Times the post-closure residual gate was evaluated. This, not
        /// `j2_correction_call_count`, is the denominator for the rejected
        /// fraction: some invocations return before reaching the gate.
        count j2_correction_gate_eval_count: usize,
        /// Times that gate blocked acceptance, i.e. the closure finished with a
        /// residual worse than `endpoint_target_km` and the plan was invalidated.
        count j2_correction_rejected_count: usize,
        /// Sum of finite end-of-closure residuals (metres), with its own count, so
        /// the mean survives cross-worker and cross-event merges. A mean far above
        /// the tolerance says the fixed point is not converging; a mean just above
        /// it says the tolerance is the thing that is mis-set.
        ///
        /// This is the one REPORTED PHYSICS number this module carries, which is
        /// why regions accumulate from zero rather than subtracting a baseline --
        /// see [`enter_evaluation_diagnostic_region`].
        f64_sum j2_correction_residual_m_sum: f64,
        count j2_correction_residual_finite_count: usize,
        /// Residual sum (metres) over REJECTED invocations only. With the totals
        /// above this separates the two populations: subtracting it gives the
        /// accepted mean, and it alone gives the rejected mean. A rejected mean
        /// near the tolerance means the budget is one or two steps short; a
        /// rejected mean orders of magnitude above it means more steps would not
        /// help and those geometries are simply not closing.
        f64_sum j2_correction_rejected_residual_m_sum: f64,
        /// Invocations of `apply_iterative_j2_lambert_correction`. Pairs with
        /// `j2_correction_lambert_retry_count` to give retries-per-invocation,
        /// which is the only way to see whether the endpoint closure converges
        /// early or runs to `J2ClosureSettings::max_iterations` every time.
        count j2_correction_call_count: usize,
        count j2_correction_iteration_count: usize,
        count j2_correction_lambert_retry_count: usize,
        count branch_source_count: usize,
        count branch_shared_prepare_count: usize,
        count branch_eval_call_count: usize,
        count branch_emitted_count: usize,
        count branch_rejected_count: usize,
        count branch_target_propagation_call_count: usize,
        count branch_lambert_sampling_call_count: usize,
        count branch_brent_call_count: usize,
        count branch_brent_eval_request_count: usize,
        count branch_brent_cache_hit_count: usize,
        count branch_brent_cache_miss_count: usize,
        count branch_j2_correction_call_count: usize,
        f64_sum branch_shared_prepare_s: f64,
        f64_sum branch_phase_release_s: f64,
        f64_sum branch_target_propagation_s: f64,
        f64_sum branch_lambert_sampling_s: f64,
        f64_sum branch_brent_s: f64,
        f64_sum branch_j2_correction_s: f64,
        /// Plan evaluations whose cached plane angle exceeds ~170 degrees
        /// (near-pi Lambert geometry; tracked to size a future robust fix).
        count near_pi_plane_eval_count: usize,
        /// Deep HF profiling, excluded from stage-metric and science output maps.
        nested hf_propagation: HfPropagationTelemetry,
    }
}

/// Arithmetic overflow while collecting evaluator-owned diagnostic work.
///
/// This stays separate from target-propagation authority: callers map it at
/// their explicit public boundary, rather than reclassifying a diagnostic
/// accounting failure as a force-model decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluationArithmeticOverflow;

impl std::fmt::Display for EvaluationArithmeticOverflow {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("evaluation diagnostic arithmetic overflow")
    }
}

impl std::error::Error for EvaluationArithmeticOverflow {}

#[inline]
pub(super) fn checked_diagnostic_counter_add(
    counter: &mut usize,
    incoming: usize,
) -> Result<(), EvaluationArithmeticOverflow> {
    *counter = counter
        .checked_add(incoming)
        .ok_or(EvaluationArithmeticOverflow)?;
    Ok(())
}

#[inline]
pub(super) fn j2_iteration_count_as_u32(
    iteration_count: usize,
) -> Result<u32, EvaluationArithmeticOverflow> {
    u32::try_from(iteration_count).map_err(|_| EvaluationArithmeticOverflow)
}

// Every counter update and cross-worker reduction is checked. An overflow is
// typed and never becomes a clamped diagnostic that could seal partial work.

impl EvaluationDiagnosticCounters {
    /// Field-wise `self - before`. **`cfg(test)`, deliberately.**
    ///
    /// Exact only for the integer fields. `j2_correction_residual_m_sum` and
    /// `j2_correction_rejected_residual_m_sum` are METRES and the six `*_s`
    /// fields are seconds, so on those this evaluates
    /// `fl(B + a_1 + ... + a_k) - B` and returns a value whose error grows
    /// with the baseline `B`, not with the work being measured.
    ///
    /// Production used to measure work units this way against the thread-local
    /// totals, and nothing zeroes those outside tests -- so `B` was the
    /// executing thread's whole-campaign history and the error in every
    /// reported residual grew with campaign length and thread count. The
    /// production shape is now [`enter_evaluation_diagnostic_region`] /
    /// [`leave_evaluation_diagnostic_region`]: accumulate the unit into a fresh
    /// zero and merge, so the reported contribution is the exact
    /// left-associated sum of the unit's own terms.
    ///
    /// It survives as a test tool because a test controls its own baseline and
    /// knows it is small. Do not lift the `cfg(test)` to bring it back: the
    /// compiler refusing to resolve it is what keeps the trap from returning.
    #[cfg(test)]
    #[inline]
    pub fn delta_since(self, before: Self) -> Result<Self, EvaluationArithmeticOverflow> {
        Self::roster_delta_since(&self, &before)
    }

    /// Field-wise merge in roster declaration order, transactional: an
    /// overflow error leaves `self` unchanged.
    #[inline]
    pub fn add_delta(&mut self, delta: &Self) -> Result<(), EvaluationArithmeticOverflow> {
        let mut merged = *self;
        Self::roster_add(&mut merged, delta)?;
        *self = merged;
        Ok(())
    }
}
counter_roster! {
    error = EvaluationArithmeticOverflow;
    overflow = EvaluationArithmeticOverflow;
    sub = test;
    /// Deep HF diagnostics, outside science rows and stage metrics.
    ///
    /// Carried by existing evaluation-diagnostic worker reductions. Records emit
    /// only after target-grid success: `requested` includes caller slots and
    /// duplicates; `unique_attempted` is deduplicated work in multi-TOF and equals
    /// requested work in scalar paths.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct HfPropagationTelemetry {
        count target_grid_call_count: usize,
        count target_grid_requested_state_count: usize,
        count target_grid_unique_attempted_state_count: usize,
        f64_sum target_grid_s: f64,
        count brent_call_count: usize,
        f64_sum brent_s: f64,
        count intercept_refinement_call_count: usize,
        count intercept_refinement_iteration_count: usize,
        f64_sum intercept_refinement_s: f64,
    }
}

impl HfPropagationTelemetry {
    /// Field-wise `self - before`. `cfg(test)` for the same reason as
    /// [`EvaluationDiagnosticCounters::delta_since`] -- see that doc comment.
    #[cfg(test)]
    #[inline]
    pub fn delta_since(self, before: Self) -> Result<Self, EvaluationArithmeticOverflow> {
        Self::roster_delta_since(&self, &before)
    }

    /// Field-wise merge in roster declaration order, transactional: an
    /// overflow error leaves `self` unchanged.
    #[inline]
    pub(super) fn add_delta(&mut self, delta: &Self) -> Result<(), EvaluationArithmeticOverflow> {
        let mut merged = *self;
        Self::roster_add(&mut merged, delta)?;
        *self = merged;
        Ok(())
    }
}
#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) enum HfPropagationStage {
    TargetGrid {
        requested_states: usize,
        unique_attempted_states: usize,
    },
    Brent,
    InterceptRefinement(usize),
}

thread_local! {
    static EVALUATION_DIAGNOSTIC_COUNTERS: RefCell<EvaluationDiagnosticCounters> =
        RefCell::new(EvaluationDiagnosticCounters::default());
}

#[inline]
pub(super) fn record_evaluation_diagnostic(
    update: impl FnOnce(&mut EvaluationDiagnosticCounters) -> Result<(), EvaluationArithmeticOverflow>,
) -> Result<(), EvaluationArithmeticOverflow> {
    EVALUATION_DIAGNOSTIC_COUNTERS.with(|counters| {
        let mut next = *counters.borrow();
        update(&mut next)?;
        *counters.borrow_mut() = next;
        Ok(())
    })
}

#[inline]
pub fn evaluation_diagnostic_snapshot() -> EvaluationDiagnosticCounters {
    EVALUATION_DIAGNOSTIC_COUNTERS.with(|counters| *counters.borrow())
}

#[cfg(test)]
#[inline]
pub(super) fn hf_propagation_telemetry_snapshot() -> HfPropagationTelemetry {
    evaluation_diagnostic_snapshot().hf_propagation
}

#[cfg(test)]
#[inline]
pub(super) fn record_hf_propagation_stage(
    stage: HfPropagationStage,
    duration_s: f64,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        let counters = &mut counters.hf_propagation;
        match stage {
            HfPropagationStage::TargetGrid {
                requested_states,
                unique_attempted_states,
            } => {
                checked_diagnostic_counter_add(&mut counters.target_grid_call_count, 1)?;
                checked_diagnostic_counter_add(
                    &mut counters.target_grid_requested_state_count,
                    requested_states,
                )?;
                checked_diagnostic_counter_add(
                    &mut counters.target_grid_unique_attempted_state_count,
                    unique_attempted_states,
                )?;
                counters.target_grid_s += duration_s;
            }
            HfPropagationStage::Brent => {
                checked_diagnostic_counter_add(&mut counters.brent_call_count, 1)?;
                counters.brent_s += duration_s;
            }
            HfPropagationStage::InterceptRefinement(iterations) => {
                checked_diagnostic_counter_add(&mut counters.intercept_refinement_call_count, 1)?;
                checked_diagnostic_counter_add(
                    &mut counters.intercept_refinement_iteration_count,
                    iterations,
                )?;
                counters.intercept_refinement_s += duration_s;
            }
        }
        Ok(())
    })
}

#[inline]
pub fn merge_evaluation_diagnostics(
    delta: &EvaluationDiagnosticCounters,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| counters.add_delta(delta))
}

#[inline]
pub fn restore_evaluation_diagnostics(value: &EvaluationDiagnosticCounters) {
    EVALUATION_DIAGNOSTIC_COUNTERS.with(|counters| *counters.borrow_mut() = *value);
}

/// Opens a measurement region: zeroes this thread's diagnostic counters and
/// returns the totals it displaced.
///
/// Inside the region [`evaluation_diagnostic_snapshot`] IS the region's own
/// contribution -- no subtraction, so the f64 fields (metres and seconds) come
/// back as the exact left-associated sum of the terms this region recorded,
/// independent of how much the thread had already accumulated. Close with
/// [`leave_evaluation_diagnostic_region`] to keep the work, or
/// [`restore_evaluation_diagnostics`] with the returned totals to discard it.
///
/// Regions nest: an inner region's `outer` is the enclosing region's
/// in-progress sum, which the inner close folds back in.
#[inline]
#[must_use]
pub fn enter_evaluation_diagnostic_region() -> EvaluationDiagnosticCounters {
    EVALUATION_DIAGNOSTIC_COUNTERS.with(|counters| {
        let outer = *counters.borrow();
        *counters.borrow_mut() = EvaluationDiagnosticCounters::default();
        outer
    })
}

/// Closes a region opened by [`enter_evaluation_diagnostic_region`], restoring
/// `outer` with `recorded` folded in so the work stays visible to the enclosing
/// region.
///
/// An error exit that never calls this leaves the thread holding only the
/// region's partial work. That is sound because the absolute thread-local
/// total is never reported: every consumer reads a region's contribution, and
/// the parallel paths reduce region contributions explicitly. If a caller ever
/// needs the running total itself, this contract has to change first.
#[inline]
pub fn leave_evaluation_diagnostic_region(
    outer: &EvaluationDiagnosticCounters,
    recorded: &EvaluationDiagnosticCounters,
) -> Result<(), EvaluationArithmeticOverflow> {
    let mut merged = *outer;
    merged.add_delta(recorded)?;
    restore_evaluation_diagnostics(&merged);
    Ok(())
}

#[inline]
pub fn record_lambert_branch_solution(
    rev: i32,
    low_path: bool,
    prograde: bool,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.lambert_branch_attempt_count, 1)?;
        checked_diagnostic_counter_add(&mut counters.lambert_branch_valid_count, 1)?;
        if rev == 0 {
            checked_diagnostic_counter_add(&mut counters.lambert_branch_rev0_count, 1)?;
        } else if rev > 0 {
            checked_diagnostic_counter_add(&mut counters.lambert_branch_rev_gt0_count, 1)?;
        }
        if low_path {
            checked_diagnostic_counter_add(&mut counters.lambert_branch_low_path_count, 1)?;
        } else {
            checked_diagnostic_counter_add(&mut counters.lambert_branch_high_path_count, 1)?;
        }
        if prograde {
            checked_diagnostic_counter_add(&mut counters.lambert_branch_prograde_count, 1)?;
        } else {
            checked_diagnostic_counter_add(&mut counters.lambert_branch_retrograde_count, 1)?;
        }
        Ok(())
    })
}

#[inline]
pub(super) fn record_lambert_batch_call(
    row_count: usize,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.lambert_batch_call_count, 1)?;
        checked_diagnostic_counter_add(&mut counters.lambert_batch_row_count, row_count)?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_lambert_batch_work(
    telemetry: crate::lambert::VariableR2BranchTelemetry,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(
            &mut counters.lambert_batch_simd_lane_solve_count,
            telemetry.simd_lane_solves,
        )?;
        checked_diagnostic_counter_add(
            &mut counters.lambert_batch_scalar_variant_solve_count,
            telemetry.scalar_variant_solves,
        )?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_lambert_scalar_tof_calls(
    max_revs: i32,
    has_branch_selection: bool,
    count: usize,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.lambert_scalar_tof_count, count)?;
        if max_revs > 0 {
            checked_diagnostic_counter_add(&mut counters.lambert_max_revs_gt0_call_count, count)?;
        }
        if has_branch_selection {
            checked_diagnostic_counter_add(
                &mut counters.lambert_branch_selection_call_count,
                count,
            )?;
        }
        Ok(())
    })
}

#[inline]
pub(super) fn record_target_j2_batch_state_count(
    count: usize,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.target_j2_batch_state_count, count)?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_target_j2_simd4_chunk() -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.target_j2_simd4_chunk_count, 1)?;
        checked_diagnostic_counter_add(&mut counters.target_j2_batch_state_count, 4)?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_target_j2_scalar_state() -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.target_j2_scalar_state_count, 1)?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_j2_residual_gate(
    blocked: bool,
    residual_m: f64,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.j2_correction_gate_eval_count, 1)?;
        if blocked {
            checked_diagnostic_counter_add(&mut counters.j2_correction_rejected_count, 1)?;
            if residual_m.is_finite() {
                counters.j2_correction_rejected_residual_m_sum += residual_m;
            }
        }
        Ok(())
    })
}

#[inline]
pub(super) fn record_j2_propagate_state() -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.j2_propagate_state_count, 1)?;
        Ok(())
    })
}

/// Record one phase-state sub-cache lookup, on the side it landed.
///
/// Visible past this module because the lookup lives in `solve`, not in
/// `evaluate`: the miss side goes on to reach [`record_j2_propagate_state`] one
/// level down, but only the lookup site can see the hit. Keeping both
/// increments at that one site is what makes `hit + miss` the lookup count by
/// construction.
#[inline]
pub fn record_phase_state_cache_lookup(hit: bool) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        let counter = if hit {
            &mut counters.phase_state_cache_hit_count
        } else {
            &mut counters.phase_state_cache_miss_count
        };
        checked_diagnostic_counter_add(counter, 1)?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_j2_correction(
    iterations: usize,
    lambert_retries: usize,
    residual_m: f64,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.j2_correction_call_count, 1)?;
        if residual_m.is_finite() {
            counters.j2_correction_residual_m_sum += residual_m;
            checked_diagnostic_counter_add(&mut counters.j2_correction_residual_finite_count, 1)?;
        }
        checked_diagnostic_counter_add(&mut counters.j2_correction_iteration_count, iterations)?;
        checked_diagnostic_counter_add(
            &mut counters.j2_correction_lambert_retry_count,
            lambert_retries,
        )?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_branch_source() -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.branch_source_count, 1)?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_branch_shared_prepare(
    total_s: f64,
    phase_release_s: f64,
    target_propagation_s: f64,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.branch_shared_prepare_count, 1)?;
        counters.branch_shared_prepare_s += total_s;
        counters.branch_phase_release_s += phase_release_s;
        counters.branch_target_propagation_s += target_propagation_s;
        Ok(())
    })
}

#[inline]
pub(super) fn record_branch_eval_call() -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.branch_eval_call_count, 1)?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_branch_emitted() -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.branch_emitted_count, 1)?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_branch_rejected() -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.branch_rejected_count, 1)?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_branch_target_propagation(
    duration_s: f64,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.branch_target_propagation_call_count, 1)?;
        counters.branch_target_propagation_s += duration_s;
        Ok(())
    })
}

#[inline]
pub(super) fn record_branch_lambert_sampling(
    duration_s: f64,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.branch_lambert_sampling_call_count, 1)?;
        counters.branch_lambert_sampling_s += duration_s;
        Ok(())
    })
}

#[inline]
pub(super) fn record_branch_brent(duration_s: f64) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.branch_brent_call_count, 1)?;
        counters.branch_brent_s += duration_s;
        Ok(())
    })
}

#[inline]
pub(super) fn record_branch_brent_cache_counts(
    request_count: usize,
    hit_count: usize,
    miss_count: usize,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(
            &mut counters.branch_brent_eval_request_count,
            request_count,
        )?;
        checked_diagnostic_counter_add(&mut counters.branch_brent_cache_hit_count, hit_count)?;
        checked_diagnostic_counter_add(&mut counters.branch_brent_cache_miss_count, miss_count)?;
        Ok(())
    })
}

#[inline]
pub(super) fn record_branch_j2_correction(
    duration_s: f64,
) -> Result<(), EvaluationArithmeticOverflow> {
    record_evaluation_diagnostic(|counters| {
        checked_diagnostic_counter_add(&mut counters.branch_j2_correction_call_count, 1)?;
        counters.branch_j2_correction_s += duration_s;
        Ok(())
    })
}
