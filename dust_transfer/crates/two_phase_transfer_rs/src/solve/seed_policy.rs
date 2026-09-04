use super::{
    combined_transfer_initial_guess, compute_time_to_nodes, COARSE_EARLY_STOP_BEST_COST_KMS,
    COARSE_EARLY_STOP_MIN_EVALS, COARSE_EARLY_STOP_MIN_FINE_COUNT,
    COARSE_EARLY_STOP_WORSE_MARGIN_KMS, SINGLE_PAIR_LOWER_BOUNDS, SINGLE_PAIR_PHASE_PTS,
    SINGLE_PAIR_TIME_PTS, SINGLE_PAIR_UPPER_BOUNDS, SINGLE_PAIR_WAIT_PTS,
};
use crate::evaluate::EciOrbitSummary;
use crate::types::{
    InvalidTargetPropagationAuthorityCode, PlanContext, SearchDepthPolicy, WarmStartData,
};
use rustc_hash::FxHashMap;
use satpy_core::eci2kep_impl;
use std::cmp::Ordering;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
pub(super) struct SolverSeed {
    pub(super) x: [f64; 3],
    pub(super) warm_start_used: bool,
}

#[derive(Default)]
pub(super) struct SolveLocalWorkCache {
    pub(super) phase_state_cache: FxHashMap<u64, [f64; 6]>,
    pub(super) phase_orbit_cache: FxHashMap<u64, EciOrbitSummary>,
    pub(super) variable_r2_lambert_scratch: crate::lambert::VariableR2LambertScratch,
}

impl SolveLocalWorkCache {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn clear(&mut self) {
        self.phase_state_cache.clear();
        self.phase_orbit_cache.clear();
    }
}

#[inline]
fn normalized_seed_distance(lhs: &[f64; 3], rhs: &[f64; 3]) -> f64 {
    let mut sum_sq = 0.0;
    for (((&lo, &hi), &lhs_value), &rhs_value) in SINGLE_PAIR_LOWER_BOUNDS
        .iter()
        .zip(SINGLE_PAIR_UPPER_BOUNDS.iter())
        .zip(lhs.iter())
        .zip(rhs.iter())
    {
        let span = (hi - lo).max(1e-12);
        let delta = (lhs_value - rhs_value) / span;
        sum_sq += delta * delta;
    }
    sum_sq.sqrt()
}

pub(super) fn sort_grid_seed_candidates_by_hint(seeds: &mut [SolverSeed], hint_x: [f64; 3]) {
    seeds.sort_by(|left, right| {
        let left_dist = normalized_seed_distance(&left.x, &hint_x);
        let right_dist = normalized_seed_distance(&right.x, &hint_x);
        left_dist
            .partial_cmp(&right_dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left.x[0]
                    .partial_cmp(&right.x[0])
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                left.x[1]
                    .partial_cmp(&right.x[1])
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                left.x[2]
                    .partial_cmp(&right.x[2])
                    .unwrap_or(Ordering::Equal)
            })
    });
}

#[inline]
const fn coarse_reject_margin_kms(policy: &SearchDepthPolicy) -> f64 {
    policy.coarse_reject_margin_km_s
}

#[inline]
pub(super) fn should_stop_coarse_stage(
    policy: &SearchDepthPolicy,
    evaluated_count: usize,
    best_coarse_cost: f64,
    recent_costs: &VecDeque<f64>,
    provisional_fine_count: usize,
) -> bool {
    if evaluated_count < COARSE_EARLY_STOP_MIN_EVALS {
        return false;
    }
    if !best_coarse_cost.is_finite() || best_coarse_cost > COARSE_EARLY_STOP_BEST_COST_KMS {
        return false;
    }
    if recent_costs.len() < 4 {
        return false;
    }
    if provisional_fine_count < COARSE_EARLY_STOP_MIN_FINE_COUNT {
        return false;
    }
    let margin = COARSE_EARLY_STOP_WORSE_MARGIN_KMS.max(coarse_reject_margin_kms(policy));
    recent_costs
        .iter()
        .all(|cost| !cost.is_finite() || *cost > best_coarse_cost + margin)
}

#[inline]
pub(super) fn seed_is_duplicate(lhs: &[f64; 3], rhs: &[f64; 3]) -> bool {
    lhs.iter()
        .zip(rhs.iter())
        .all(|(left, right)| (left - right).abs() <= 1e-9)
}

#[inline]
fn validate_seed_bounds(x: [f64; 3], clamp: bool) -> Option<[f64; 3]> {
    if !x.iter().all(|value| value.is_finite()) {
        return None;
    }

    let mut out = x;
    if clamp {
        out[0] = out[0].clamp(SINGLE_PAIR_LOWER_BOUNDS[0], SINGLE_PAIR_UPPER_BOUNDS[0]);
        out[1] = out[1].clamp(SINGLE_PAIR_LOWER_BOUNDS[1], SINGLE_PAIR_UPPER_BOUNDS[1]);
        out[2] = out[2].clamp(SINGLE_PAIR_LOWER_BOUNDS[2], SINGLE_PAIR_UPPER_BOUNDS[2]);
    } else if out[0] < SINGLE_PAIR_LOWER_BOUNDS[0]
        || out[0] > SINGLE_PAIR_UPPER_BOUNDS[0]
        || out[1] < SINGLE_PAIR_LOWER_BOUNDS[1]
        || out[1] > SINGLE_PAIR_UPPER_BOUNDS[1]
        || out[2] < SINGLE_PAIR_LOWER_BOUNDS[2]
        || out[2] > SINGLE_PAIR_UPPER_BOUNDS[2]
    {
        return None;
    }

    if out[0] + out[2] >= 0.98 {
        if !clamp {
            return None;
        }
        let headroom = 0.98 - out[0];
        if headroom <= 0.0 {
            return None;
        }
        out[2] = out[2].min(headroom.max(0.0));
    }

    Some(out)
}

fn push_solver_seed(seeds: &mut Vec<SolverSeed>, x: [f64; 3], warm_start_used: bool, clamp: bool) {
    let Some(normalized) = validate_seed_bounds(x, clamp) else {
        return;
    };
    if seeds
        .iter()
        .any(|existing| seed_is_duplicate(&existing.x, &normalized))
    {
        return;
    }
    seeds.push(SolverSeed {
        x: normalized,
        warm_start_used,
    });
}

#[inline]
pub(super) const fn warm_start_matches_pair(
    warm_start: &WarmStartData,
    sat_index: i32,
    target_index: i32,
) -> bool {
    warm_start.sat_index == sat_index && warm_start.target_index == target_index
}

#[inline]
fn single_pair_phase_angle(dep_equ: &[f64; 6], tgt_equ: &[f64; 6]) -> f64 {
    let diff = (tgt_equ[5] - dep_equ[5])
        .abs()
        .rem_euclid(std::f64::consts::TAU);
    if diff > std::f64::consts::PI {
        std::f64::consts::TAU - diff
    } else {
        diff
    }
}

pub(super) fn build_single_pair_seeds(
    ctx: &PlanContext,
    warm_start: Option<&WarmStartData>,
) -> Result<Vec<SolverSeed>, InvalidTargetPropagationAuthorityCode> {
    let mut seeds = Vec::new();
    super::try_reserve_transfer_capacity(&mut seeds, 80)?;
    let grid_capacity = SINGLE_PAIR_TIME_PTS
        .len()
        .checked_mul(SINGLE_PAIR_PHASE_PTS.len())
        .and_then(|count| count.checked_mul(SINGLE_PAIR_WAIT_PTS.len()))
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let mut grid_seeds = Vec::new();
    super::try_reserve_transfer_capacity(&mut grid_seeds, grid_capacity)?;
    let mut grid_sort_hint = [0.0, 1.0, 0.0];

    if let Some(warm_start) = warm_start.filter(|seed| seed.valid && seed.cost.is_finite()) {
        push_solver_seed(&mut seeds, warm_start.x, true, false);
    }

    let mut dep_kep = [0.0; 6];
    let mut tgt_kep = [0.0; 6];
    eci2kep_impl(&ctx.dep_eci, false, true, &mut dep_kep);
    eci2kep_impl(&ctx.tgt_eci, false, true, &mut tgt_kep);

    if ctx.dep_sma > 0.0 && ctx.tgt_sma > 0.0 && ctx.max_time_s > 0.0 {
        let heuristic = combined_transfer_initial_guess(
            ctx.dep_sma,
            dep_kep[2],
            ctx.tgt_sma,
            tgt_kep[2],
            single_pair_phase_angle(&ctx.dep_equ, &ctx.tgt_equ),
            ctx.max_time_s,
        );
        grid_sort_hint = heuristic;
        push_solver_seed(&mut seeds, heuristic, false, true);
        push_solver_seed(&mut seeds, [heuristic[0], 1.0, 0.0], false, true);
        push_solver_seed(
            &mut seeds,
            [heuristic[0] * 0.5, heuristic[1], heuristic[2]],
            false,
            true,
        );
        push_solver_seed(&mut seeds, [0.0, heuristic[1], heuristic[2]], false, true);

        if let Some((dt_an, dt_dn)) = compute_time_to_nodes(&dep_kep) {
            for dt in [dt_an, dt_dn] {
                push_solver_seed(
                    &mut seeds,
                    [dt / ctx.max_time_s, heuristic[1], heuristic[2]],
                    false,
                    true,
                );
            }
        }

        if let Some((dt_an, dt_dn)) = compute_time_to_nodes(&tgt_kep) {
            for dt in [dt_an, dt_dn] {
                push_solver_seed(
                    &mut seeds,
                    [dt / ctx.max_time_s, 1.0, heuristic[2]],
                    false,
                    true,
                );
            }
        }
    }

    push_solver_seed(&mut seeds, [0.0, 1.0, 0.0], false, false);
    push_solver_seed(&mut seeds, [0.05, 1.0, 0.0], false, false);

    // The retired triple loop enumerated all 64 raw triples; the 12 excluded
    // from DETERMINISTIC_GRID_POINTS (`time + wait > 0.98`) were all rejected
    // by `validate_seed_bounds` (`out[0] + out[2] >= 0.98`, unclamped), so
    // iterating the compile-time table yields the identical seed list in the
    // identical order.
    for &x in &super::DETERMINISTIC_GRID_POINTS {
        push_solver_seed(&mut grid_seeds, x, false, false);
    }

    sort_grid_seed_candidates_by_hint(&mut grid_seeds, grid_sort_hint);
    seeds.extend(grid_seeds);

    Ok(seeds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_start_pair_filter_accepts_only_exact_pair() {
        let warm_start = WarmStartData {
            x: [0.1, 1.0, 0.2],
            cost: 1.0,
            valid: true,
            sat_index: 3,
            target_index: 1,
        };

        assert!(warm_start_matches_pair(&warm_start, 3, 1));
        assert!(!warm_start_matches_pair(&warm_start, 2, 1));
        assert!(!warm_start_matches_pair(&warm_start, 3, 0));
        assert!(!warm_start_matches_pair(&WarmStartData::default(), 3, 1));
    }
}
