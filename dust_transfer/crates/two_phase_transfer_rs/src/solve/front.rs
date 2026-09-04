use std::cmp::Ordering;
use std::time::Instant;

use crate::types::{
    ConstellationTransferCandidate, ConstellationTransferFront,
    InvalidTargetPropagationAuthorityCode, PlanContext, PlanResult, ReplayProvenance,
    TransferFront, TransferObjectives, VerifiedSupersetStageMetrics, INVALID_COST,
};
use crate::verify::verify_transfer_result;

const TRANSFER_FRONT_TIME_RELATIVE_VELOCITY_EPS: f64 = 1e-6;

pub(super) fn verification_tolerance_for_solve(ctx: &PlanContext) -> f64 {
    if ctx.distance_tol.is_finite() && ctx.distance_tol > 0.0 {
        ctx.distance_tol
    } else {
        0.010
    }
}

const fn stamp_replay_provenance(plan: &mut PlanResult, ctx: &PlanContext) {
    plan.replay_provenance = ReplayProvenance {
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
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_tolerance_uses_configured_distance_tolerance() {
        let ctx = PlanContext {
            distance_tol: 0.025,
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };

        assert_eq!(
            verification_tolerance_for_solve(&ctx).to_bits(),
            0.025_f64.to_bits()
        );
    }

    /// The 10 m fallback fires only when the configured tolerance is unusable
    /// (non-finite or non-positive) and is strictly TIGHTER than the sealed
    /// campaign tolerance (`DISTANCE_TOL` = 25 m, compiled into Part A science
    /// as `distance_tol_km`), so a misconfigured context can only reject more
    /// candidates than the campaign gate — never verify one the sealed
    /// tolerance would refuse.
    #[test]
    fn verification_tolerance_fallback_is_tighter_than_the_sealed_tolerance() {
        // Fallback direction pin: tighter than the sealed campaign tolerance.
        const _: () = assert!(
            0.010 < crate::types::DISTANCE_TOL,
            "fallback tolerance must stay below the sealed campaign tolerance"
        );
        for unusable in [f64::NAN, 0.0, -0.025, f64::NEG_INFINITY, f64::INFINITY] {
            let ctx = PlanContext {
                distance_tol: unusable,
                ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
            };
            assert_eq!(
                verification_tolerance_for_solve(&ctx).to_bits(),
                0.010_f64.to_bits(),
                "fallback for distance_tol={unusable}"
            );
        }
    }

    #[test]
    fn verified_front_boundary_stamps_complete_replay_authority() {
        let ctx = PlanContext {
            dep_eci: [7000.0, 1.0, 2.0, 0.0, 7.5, 0.1],
            epoch_jd: 2_460_000.5,
            max_time_s: 7200.0,
            max_phase_dv: 0.5,
            max_transfer_dv: 2.0,
            max_revs: 1,
            revolution_cap: 2.0,
            min_perigee: 6500.0,
            max_apogee: 8000.0,
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };
        let mut plan = PlanResult::invalid();

        stamp_replay_provenance(&mut plan, &ctx);

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
            plan.replay_provenance.max_time_s.to_bits(),
            ctx.max_time_s.to_bits()
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
        assert_eq!(
            plan.replay_provenance.distance_tol.to_bits(),
            ctx.distance_tol.to_bits()
        );
        assert_eq!(
            plan.replay_provenance.deployer_min_distance.to_bits(),
            ctx.deployer_min_distance.to_bits()
        );
    }

    /// Verbatim pre-change implementation of
    /// `filter_nondominated_transfer_candidates` (parallel
    /// `Vec<Option<PlanResult>>` + `.take()`), kept as the parity baseline
    /// for the in-place retain/`mem::take` rewrite.
    fn reference_filter_nondominated(candidates: Vec<PlanResult>) -> TransferFront {
        let mut plans: Vec<Option<PlanResult>> = Vec::with_capacity(candidates.len());
        let mut objectives: Vec<TransferObjectives> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if !candidate.valid || candidate.cost >= INVALID_COST {
                continue;
            }
            let candidate_objectives = candidate.transfer_objectives();
            if !candidate_objectives.is_finite() {
                continue;
            }
            plans.push(Some(candidate));
            objectives.push(candidate_objectives);
        }
        if plans.is_empty() {
            return TransferFront::empty();
        }

        let mut order: Vec<usize> = (0..plans.len()).collect();
        order
            .sort_by(|left, right| compare_indexed_transfer_objectives(&objectives, *left, *right));
        let mut front_indices: Vec<usize> = Vec::with_capacity(order.len());
        let mut best_ratio = f64::INFINITY;
        for idx in order {
            let Some(candidate_objectives) = objectives.get(idx).copied() else {
                continue;
            };
            while let Some(&previous_idx) = front_indices.last() {
                let Some(previous_objectives) = objectives.get(previous_idx) else {
                    front_indices.pop();
                    continue;
                };
                if transfer_objectives_dominate(&candidate_objectives, previous_objectives) {
                    front_indices.pop();
                } else {
                    break;
                }
            }
            if candidate_objectives.time_per_relative_velocity_s_per_km_s
                + TRANSFER_FRONT_TIME_RELATIVE_VELOCITY_EPS
                < best_ratio
            {
                best_ratio = candidate_objectives.time_per_relative_velocity_s_per_km_s;
                front_indices.push(idx);
            }
        }

        front_indices.sort_by(|left, right| {
            let Some((Some(left_plan), left_objectives)) =
                plans.get(*left).zip(objectives.get(*left))
            else {
                return Ordering::Equal;
            };
            let Some((Some(right_plan), right_objectives)) =
                plans.get(*right).zip(objectives.get(*right))
            else {
                return Ordering::Equal;
            };
            let left_key =
                TransferFrontSortKey::from_plan_and_objectives(left_plan, left_objectives);
            let right_key =
                TransferFrontSortKey::from_plan_and_objectives(right_plan, right_objectives);
            compare_transfer_front_sort_keys(&left_key, &right_key)
        });
        front_indices.dedup_by(|right, left| {
            objectives
                .get(*left)
                .zip(objectives.get(*right))
                .is_some_and(|(left_objectives, right_objectives)| {
                    transfer_objectives_equal(left_objectives, right_objectives)
                })
        });

        let mut front = Vec::with_capacity(front_indices.len());
        for idx in front_indices {
            if let Some(Some(plan)) = plans.get_mut(idx) {
                let plan = std::mem::take(plan);
                front.push(plan);
            }
        }
        TransferFront::new(front)
    }

    /// Synthetic candidate mirroring solve.rs's test builder, with `best_M`
    /// used as a row-identity marker so ordering drift is visible.
    fn marked_candidate(
        total_dv: f64,
        total_time: f64,
        relative_velocity: f64,
        ratios: [f64; 3],
        marker: i32,
    ) -> PlanResult {
        let mut plan = PlanResult::invalid();
        plan.valid = true;
        plan.cost = total_dv;
        plan.phase_dv_norm = total_dv;
        plan.transfer_dv_norm = 0.0;
        plan.time2phase = total_time;
        plan.waittime = 0.0;
        plan.tof = 0.0;
        plan.payload_intercept_state[3] = relative_velocity;
        plan.target_intercept_state[3] = 0.0;
        plan.time2phase_ratio = ratios[0];
        plan.phase_sma_ratio = ratios[1];
        plan.waittime_ratio = ratios[2];
        plan.best_M = marker;
        plan
    }

    fn assert_fronts_bitwise_equal(actual: &TransferFront, expected: &TransferFront) {
        assert_eq!(
            actual.candidates.len(),
            expected.candidates.len(),
            "surviving front set size drift"
        );
        // The size pin above is satisfied by 0 == 0, and `zip` on two empty
        // fronts yields no rows -- so two solves that both produced nothing
        // would have their bit-identity certified without a row being
        // compared. A determinism gate must have something to be deterministic
        // about.
        assert!(
            !actual.candidates.is_empty(),
            "front comparison needs at least one surviving candidate to be meaningful"
        );
        for (row, (a, e)) in actual
            .candidates
            .iter()
            .zip(expected.candidates.iter())
            .enumerate()
        {
            assert_eq!(a.best_M, e.best_M, "row {row}: identity/ordering drift");
            assert_eq!(a.valid, e.valid, "row {row}: valid drift");
            assert_eq!(a.cost.to_bits(), e.cost.to_bits(), "row {row}: cost drift");
            assert_eq!(
                a.time2phase.to_bits(),
                e.time2phase.to_bits(),
                "row {row}: time2phase drift"
            );
            assert_eq!(
                a.payload_intercept_state[3].to_bits(),
                e.payload_intercept_state[3].to_bits(),
                "row {row}: relative-velocity component drift"
            );
            assert_eq!(
                a.time2phase_ratio.to_bits(),
                e.time2phase_ratio.to_bits(),
                "row {row}: time2phase_ratio drift"
            );
            assert_eq!(
                a.phase_sma_ratio.to_bits(),
                e.phase_sma_ratio.to_bits(),
                "row {row}: phase_sma_ratio drift"
            );
            assert_eq!(
                a.waittime_ratio.to_bits(),
                e.waittime_ratio.to_bits(),
                "row {row}: waittime_ratio drift"
            );
        }
    }

    #[test]
    fn filter_nondominated_matches_option_layer_baseline_set_and_ordering() -> anyhow::Result<()> {
        // Mixed batch: nondominated rows, a dominated row, an invalid row, an
        // INVALID_COST row, a non-finite-objective row, a near-duplicate
        // inside the dedup epsilons, an exact-objective tie broken by the
        // ratio sort keys, and a low-dv row placed LAST so the final front
        // ordering is not ascending in filtered index (exercising the
        // index-based move-out).
        let mut invalid_row = marked_candidate(0.30, 3000.0, 5.0, [0.2, 1.0, 0.2], 2);
        invalid_row.valid = false;
        let mut invalid_cost_row = marked_candidate(0.31, 3100.0, 5.0, [0.2, 1.0, 0.2], 4);
        invalid_cost_row.cost = INVALID_COST;
        let candidates = vec![
            marked_candidate(0.25, 4000.0, 4.0, [0.3, 1.0, 0.2], 1), // dominated
            invalid_row,
            marked_candidate(0.10, 7200.0, 2.0, [0.1, 1.0, 0.1], 3),
            invalid_cost_row,
            marked_candidate(0.20, 3600.0, 9.0, [0.2, 1.0, 0.0], 5),
            // Within dedup epsilons of marker 5 (dv diff 5e-11, ratio diff
            // ~5.6e-10 s per km/s).
            marked_candidate(0.20 + 5e-11, 3_600.000_005, 9.0, [0.4, 1.0, 0.2], 6),
            marked_candidate(0.15, 5000.0, 0.0, [0.3, 1.0, 0.3], 7), // non-finite
            // Exact objective tie with marker 3, later ratio sort keys.
            marked_candidate(0.10, 7200.0, 2.0, [0.5, 1.1, 0.3], 8),
            marked_candidate(0.05, 20000.0, 1.0, [0.6, 1.2, 0.4], 9), // low dv, last
        ];

        let actual = filter_nondominated_transfer_candidates(candidates.clone())?;
        let expected = reference_filter_nondominated(candidates);

        anyhow::ensure!(
            expected.candidates.len() > 1,
            "parity fixture must keep a multi-row front"
        );
        anyhow::ensure!(
            expected.candidates.iter().any(|plan| plan.best_M == 9),
            "parity fixture must keep the trailing low-dv row"
        );
        assert_fronts_bitwise_equal(&actual, &expected);
        Ok(())
    }

    #[test]
    fn filter_nondominated_matches_baseline_on_empty_survivor_set() -> anyhow::Result<()> {
        let mut invalid_row = marked_candidate(0.30, 3000.0, 5.0, [0.2, 1.0, 0.2], 1);
        invalid_row.valid = false;
        let nonfinite_row = marked_candidate(0.15, 5000.0, 0.0, [0.3, 1.0, 0.3], 2);
        let candidates = vec![invalid_row, nonfinite_row];

        let actual = filter_nondominated_transfer_candidates(candidates.clone())?;
        let expected = reference_filter_nondominated(candidates);

        anyhow::ensure!(
            expected.candidates.is_empty(),
            "baseline front must be empty"
        );
        anyhow::ensure!(actual.candidates.is_empty(), "fallible front must be empty");
        Ok(())
    }

    #[test]
    fn append_constellation_candidates_rejects_unrepresentable_indices() {
        let Some(too_large) = usize::try_from(i32::MAX)
            .ok()
            .and_then(|maximum| maximum.checked_add(1))
        else {
            return;
        };
        let mut out = Vec::new();
        let front =
            TransferFront::new(vec![marked_candidate(0.1, 7200.0, 2.0, [0.1, 1.0, 0.1], 1)]);

        let result =
            append_constellation_transfer_candidates(&mut out, too_large, 0, 0.1, [0.0; 3], front);

        assert_eq!(
            result,
            Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
        );
        assert!(out.is_empty());
    }
}

#[inline]
pub(super) fn verified_front_from_plan(
    ctx: &PlanContext,
    mut plan: PlanResult,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    stamp_replay_provenance(&mut plan, ctx);
    if plan.valid
        && plan.cost < INVALID_COST
        && verify_transfer_result(&plan, ctx, verification_tolerance_for_solve(ctx)).verified
    {
        let mut candidates = Vec::new();
        candidates
            .try_reserve(1)
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        candidates.push(plan);
        Ok(TransferFront::new(candidates))
    } else {
        Ok(TransferFront::empty())
    }
}

pub(super) fn push_constellation_transfer_candidates(
    archive: &mut ConstellationFrontArchive,
    sat_idx: usize,
    tgt_idx: usize,
    estimate: f64,
    estimated_x: [f64; 3],
    front: TransferFront,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let (sat_index, target_index) = constellation_candidate_indices(sat_idx, tgt_idx)?;
    for plan in front.candidates {
        if let Some(candidate) = ConstellationTransferCandidate::from_plan(
            sat_index,
            target_index,
            estimate,
            estimated_x,
            plan,
        ) {
            archive.insert(candidate)?;
        }
    }
    Ok(())
}

pub(super) fn append_constellation_transfer_candidates(
    out: &mut Vec<ConstellationTransferCandidate>,
    sat_idx: usize,
    tgt_idx: usize,
    estimate: f64,
    estimated_x: [f64; 3],
    front: TransferFront,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let (sat_index, target_index) = constellation_candidate_indices(sat_idx, tgt_idx)?;
    out.try_reserve(front.candidates.len())
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    for plan in front.candidates {
        if let Some(candidate) = ConstellationTransferCandidate::from_plan(
            sat_index,
            target_index,
            estimate,
            estimated_x,
            plan,
        ) {
            out.push(candidate);
        }
    }
    Ok(())
}

#[inline]
fn constellation_candidate_indices(
    sat_idx: usize,
    tgt_idx: usize,
) -> Result<(i32, i32), InvalidTargetPropagationAuthorityCode> {
    let sat_index = i32::try_from(sat_idx)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let target_index = i32::try_from(tgt_idx)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    Ok((sat_index, target_index))
}

pub(super) fn constellation_candidate_dominates(
    left: &ConstellationTransferCandidate,
    right: &ConstellationTransferCandidate,
) -> bool {
    transfer_candidate_dominates(&left.optimum, &right.optimum)
}

#[derive(Default)]
pub(super) struct ConstellationFrontArchive {
    candidates: Vec<ConstellationTransferCandidate>,
}

impl ConstellationFrontArchive {
    pub(super) const fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    pub(super) fn insert(
        &mut self,
        candidate: ConstellationTransferCandidate,
    ) -> Result<bool, InvalidTargetPropagationAuthorityCode> {
        if !constellation_candidate_is_front_valid(&candidate) {
            return Ok(false);
        }
        if self
            .candidates
            .iter()
            .any(|existing| transfer_objectives_equal(&existing.objectives, &candidate.objectives))
        {
            return Ok(false);
        }
        if self
            .candidates
            .iter()
            .any(|existing| constellation_candidate_dominates(existing, &candidate))
        {
            return Ok(false);
        }
        self.candidates
            .retain(|existing| !constellation_candidate_dominates(&candidate, existing));
        self.candidates
            .try_reserve(1)
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        self.candidates.push(candidate);
        Ok(true)
    }

    pub(super) fn into_front(mut self) -> ConstellationTransferFront {
        sort_constellation_front_candidates(&mut self.candidates);
        ConstellationTransferFront::new(self.candidates)
    }
}

#[inline]
pub(super) fn constellation_candidate_is_front_valid(
    candidate: &ConstellationTransferCandidate,
) -> bool {
    candidate.valid
        && candidate.optimum.valid
        && candidate.optimum.cost < INVALID_COST
        && candidate.objectives.is_finite()
}

fn sort_constellation_front_candidates(candidates: &mut [ConstellationTransferCandidate]) {
    candidates.sort_by(|left, right| {
        lex_cmp!(left, right;
            asc (objectives.total_dv),
            asc (objectives.time_per_relative_velocity_s_per_km_s),
            asc (objectives.total_time),
            desc (objectives.relative_velocity),
            int (sat_index),
            int (target_index),
            asc (optimum.time2phase_ratio),
            asc (optimum.phase_sma_ratio),
            asc (optimum.waittime_ratio),
        )
    });
}

pub(super) fn finalize_constellation_transfer_superset(
    mut candidates: Vec<ConstellationTransferCandidate>,
) -> ConstellationTransferFront {
    candidates.retain(constellation_candidate_is_front_valid);
    drop_nd_epsilon_dominated_constellation_candidates(&mut candidates);
    sort_constellation_front_candidates(&mut candidates);
    ConstellationTransferFront::new(candidates)
}

/// nd-epsilon MEMBERSHIP (sealed token `nd-epsilon-membership`): a candidate
/// that another candidate beats by `POLISH_SCOPE_ND_EPS_DV_KM_S` on total dv
/// AND by `POLISH_SCOPE_ND_EPS_TIME_FRAC` on total time leaves the merged
/// cross-pair superset before it can become strict-HF mass rows and
/// postprocess descriptors. This is exactly the divergence-gate-measured
/// predicate (docs/evidence/front-lane-20260813/): output rows bit-identical
/// at P4 and P24 sealed-stop cells, every objective delta zero, RHS
/// evaluations 6.4x down. Applied cross-pair at the ONE site that assembles
/// the constellation superset, so no second filter exists anywhere.
fn drop_nd_epsilon_dominated_constellation_candidates(
    candidates: &mut Vec<ConstellationTransferCandidate>,
) {
    let snapshot: Vec<(f64, f64)> = candidates
        .iter()
        .map(|candidate| (candidate.optimum.total_dv(), candidate.optimum.total_time()))
        .collect();
    let mut index = 0usize;
    candidates.retain(|current| {
        let current_dv = snapshot.get(index).map_or(f64::NAN, |entry| entry.0);
        let current_time = snapshot.get(index).map_or(f64::NAN, |entry| entry.1);
        let dominated = current_dv.is_finite()
            && current_time.is_finite()
            && snapshot.iter().enumerate().any(|(other_index, other)| {
                other_index != index
                    && other.0 <= current_dv - crate::solve::POLISH_SCOPE_ND_EPS_DV_KM_S
                    && other.1 <= current_time * (1.0 - crate::solve::POLISH_SCOPE_ND_EPS_TIME_FRAC)
            });
        index = index.wrapping_add(1);
        let _ = current;
        !dominated
    });
}

/// O(n^2) pairwise-dominance reference for the constellation front. Nothing in
/// production calls it; it is the independent implementation the epsilon-superset
/// finalizer's tests compare against, so it stays test-only rather than becoming
/// a second production path.
#[cfg(test)]
pub(super) fn finalize_constellation_transfer_front(
    candidates: Vec<ConstellationTransferCandidate>,
) -> ConstellationTransferFront {
    let valid: Vec<ConstellationTransferCandidate> = candidates
        .into_iter()
        .filter(constellation_candidate_is_front_valid)
        .collect();
    if valid.is_empty() {
        return ConstellationTransferFront::empty();
    }

    let mut dominated = vec![false; valid.len()];
    for (left_idx, left_candidate) in valid.iter().enumerate() {
        for (right_idx, right_candidate) in valid.iter().enumerate() {
            if right_idx <= left_idx {
                continue;
            }
            if constellation_candidate_dominates(left_candidate, right_candidate) {
                if let Some(is_dominated) = dominated.get_mut(right_idx) {
                    *is_dominated = true;
                }
            } else if constellation_candidate_dominates(right_candidate, left_candidate) {
                if let Some(is_dominated) = dominated.get_mut(left_idx) {
                    *is_dominated = true;
                }
            }
        }
    }

    let mut front: Vec<ConstellationTransferCandidate> = Vec::with_capacity(valid.len());
    for (candidate, is_dominated) in valid.into_iter().zip(dominated) {
        if !is_dominated {
            front.push(candidate);
        }
    }

    sort_constellation_front_candidates(&mut front);
    front.dedup_by(|right, left| transfer_objectives_equal(&left.objectives, &right.objectives));
    ConstellationTransferFront::new(front)
}

fn sort_plan_results(candidates: &mut [PlanResult]) {
    candidates.sort_by(|left, right| {
        lex_cmp!(left, right;
            asc (cost),
            asc (time2phase_ratio),
            asc (phase_sma_ratio),
            asc (waittime_ratio),
        )
    });
}

fn sort_transfer_front_results(candidates: &mut [PlanResult]) {
    candidates.sort_by(|left, right| {
        compare_transfer_front_sort_keys(
            &TransferFrontSortKey::from_plan(left),
            &TransferFrontSortKey::from_plan(right),
        )
    });
}

#[derive(Clone, Copy)]
struct TransferFrontSortKey {
    total_dv: f64,
    time_per_relative_velocity_s_per_km_s: f64,
    total_time: f64,
    relative_velocity: f64,
    time2phase_ratio: f64,
    phase_sma_ratio: f64,
    waittime_ratio: f64,
}

impl TransferFrontSortKey {
    fn from_plan(plan: &PlanResult) -> Self {
        let objectives = plan.transfer_objectives();
        Self::from_plan_and_objectives(plan, &objectives)
    }

    fn from_plan_and_objectives(plan: &PlanResult, objectives: &TransferObjectives) -> Self {
        Self {
            total_dv: objectives.total_dv,
            time_per_relative_velocity_s_per_km_s: objectives.time_per_relative_velocity_s_per_km_s,
            total_time: plan.total_time(),
            relative_velocity: plan.relative_velocity(),
            time2phase_ratio: plan.time2phase_ratio,
            phase_sma_ratio: plan.phase_sma_ratio,
            waittime_ratio: plan.waittime_ratio,
        }
    }
}

fn compare_transfer_front_sort_keys(
    left: &TransferFrontSortKey,
    right: &TransferFrontSortKey,
) -> Ordering {
    lex_cmp!(left, right;
        asc (total_dv),
        asc (time_per_relative_velocity_s_per_km_s),
        asc (total_time),
        desc (relative_velocity),
        asc (time2phase_ratio),
        asc (phase_sma_ratio),
        asc (waittime_ratio),
    )
}

pub(super) fn transfer_candidate_is_objective_finite(plan: &PlanResult) -> bool {
    plan.valid && plan.cost < INVALID_COST && plan.transfer_objectives().is_finite()
}

fn transfer_objectives_dominate(
    left_obj: &TransferObjectives,
    right_obj: &TransferObjectives,
) -> bool {
    const DV_EPS: f64 = 1e-12;
    const TIME_REL_V_EPS: f64 = 1e-6;

    let no_worse = left_obj.total_dv <= right_obj.total_dv + DV_EPS
        && left_obj.time_per_relative_velocity_s_per_km_s
            <= right_obj.time_per_relative_velocity_s_per_km_s + TIME_REL_V_EPS;
    let strictly_better = left_obj.total_dv + DV_EPS < right_obj.total_dv
        || left_obj.time_per_relative_velocity_s_per_km_s + TIME_REL_V_EPS
            < right_obj.time_per_relative_velocity_s_per_km_s;

    no_worse && strictly_better
}

pub(super) fn transfer_candidate_dominates(left: &PlanResult, right: &PlanResult) -> bool {
    transfer_objectives_dominate(&left.transfer_objectives(), &right.transfer_objectives())
}

pub(super) fn transfer_objectives_equal(
    left: &TransferObjectives,
    right: &TransferObjectives,
) -> bool {
    (left.total_dv - right.total_dv).abs() <= 1e-10
        && (left.time_per_relative_velocity_s_per_km_s
            - right.time_per_relative_velocity_s_per_km_s)
            .abs()
            <= 1e-6
}

fn compare_transfer_objectives(left: &TransferObjectives, right: &TransferObjectives) -> Ordering {
    left.total_dv
        .partial_cmp(&right.total_dv)
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            left.time_per_relative_velocity_s_per_km_s
                .partial_cmp(&right.time_per_relative_velocity_s_per_km_s)
                .unwrap_or(Ordering::Equal)
        })
}

#[cfg(test)]
fn compare_indexed_transfer_objectives(
    objectives: &[TransferObjectives],
    left: usize,
    right: usize,
) -> Ordering {
    let left = objectives
        .get(left)
        .expect("test baseline indexes its own objective vector");
    let right = objectives
        .get(right)
        .expect("test baseline indexes its own objective vector");
    compare_transfer_objectives(left, right)
}

pub(super) fn filter_nondominated_transfer_candidates(
    mut candidates: Vec<PlanResult>,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    // Compact the owned input Vec in place (order-preserving) instead of
    // moving survivors into a parallel Vec<Option<PlanResult>>: the input
    // allocation already stores the plans, so the Option layer and its
    // N-sized allocation are dropped and the surviving front rows are moved
    // out by index at the end.
    let mut objectives = Vec::new();
    objectives
        .try_reserve(candidates.len())
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    candidates.retain(|candidate| {
        if !candidate.valid || candidate.cost >= INVALID_COST {
            return false;
        }
        let candidate_objectives = candidate.transfer_objectives();
        if !candidate_objectives.is_finite() {
            return false;
        }
        objectives.push(candidate_objectives);
        true
    });
    if candidates.is_empty() {
        return Ok(TransferFront::empty());
    }

    let mut order = Vec::new();
    order
        .try_reserve(candidates.len())
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    for (index, (candidate, candidate_objectives)) in
        candidates.iter().zip(objectives.iter()).enumerate()
    {
        order.push((
            index,
            *candidate_objectives,
            TransferFrontSortKey::from_plan_and_objectives(candidate, candidate_objectives),
        ));
    }
    order.sort_by(|left, right| compare_transfer_objectives(&left.1, &right.1));
    let mut front_indices: Vec<(usize, TransferObjectives, TransferFrontSortKey)> = Vec::new();
    front_indices
        .try_reserve(order.len())
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let mut best_ratio = f64::INFINITY;
    for current in order {
        while let Some(previous) = front_indices.last() {
            if transfer_objectives_dominate(&current.1, &previous.1) {
                front_indices.pop();
            } else {
                break;
            }
        }
        if current.1.time_per_relative_velocity_s_per_km_s
            + TRANSFER_FRONT_TIME_RELATIVE_VELOCITY_EPS
            < best_ratio
        {
            best_ratio = current.1.time_per_relative_velocity_s_per_km_s;
            front_indices.push(current);
        }
    }

    front_indices.sort_by(|left, right| compare_transfer_front_sort_keys(&left.2, &right.2));
    front_indices.dedup_by(|right, left| transfer_objectives_equal(&left.1, &right.1));

    // front_indices are distinct, so each surviving row is moved out of the
    // owned candidates Vec exactly once; mem::take leaves an invalid
    // placeholder behind, which is dropped with the Vec.
    let mut front = Vec::new();
    front
        .try_reserve(front_indices.len())
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    for (idx, _, _) in front_indices {
        let candidate = candidates
            .get_mut(idx)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        front.push(std::mem::take(candidate));
    }
    Ok(TransferFront::new(front))
}

pub(super) fn finalize_verified_front(
    ctx: &PlanContext,
    candidates: &mut Vec<PlanResult>,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    let mut candidates_to_verify = std::mem::take(candidates);
    sort_plan_results(&mut candidates_to_verify);
    let tolerance_km = verification_tolerance_for_solve(ctx);
    let mut verified = Vec::new();
    verified
        .try_reserve(candidates_to_verify.len())
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    for mut candidate in candidates_to_verify {
        if !candidate.valid || candidate.cost >= INVALID_COST {
            continue;
        }
        stamp_replay_provenance(&mut candidate, ctx);
        let verification = verify_transfer_result(&candidate, ctx, tolerance_km);
        if verification.verified {
            verified.push(candidate);
        }
    }
    filter_nondominated_transfer_candidates(verified)
}

pub(super) fn finalize_verified_superset(
    ctx: &PlanContext,
    candidates: &mut Vec<PlanResult>,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    finalize_verified_superset_with_metrics(ctx, candidates, None)
}

pub(super) fn finalize_verified_superset_with_metrics(
    ctx: &PlanContext,
    candidates: &mut Vec<PlanResult>,
    metrics: Option<&mut VerifiedSupersetStageMetrics>,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    let mut candidates_to_verify = std::mem::take(candidates);
    sort_plan_results(&mut candidates_to_verify);
    let tolerance_km = verification_tolerance_for_solve(ctx);
    let mut verified = Vec::new();
    verified
        .try_reserve(candidates_to_verify.len())
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let verification_start = Instant::now();
    for mut candidate in candidates_to_verify {
        if !candidate.valid || candidate.cost >= INVALID_COST {
            continue;
        }
        stamp_replay_provenance(&mut candidate, ctx);
        if verify_transfer_result(&candidate, ctx, tolerance_km).verified {
            verified.push(candidate);
        }
    }
    if let Some(metrics) = metrics {
        metrics.verification_s += verification_start.elapsed().as_secs_f64();
    }
    sort_transfer_front_results(&mut verified);
    Ok(TransferFront::new(verified))
}

#[cfg(test)]
pub(super) fn finalize_verified_candidate(
    ctx: &PlanContext,
    candidates: &mut Vec<PlanResult>,
) -> Result<Option<PlanResult>, InvalidTargetPropagationAuthorityCode> {
    Ok(finalize_verified_front(ctx, candidates)?
        .candidates
        .into_iter()
        .next())
}
