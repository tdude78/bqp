use std::cmp::Ordering;

use satpy_core::{norm3, MU};

use crate::geometry::{
    combined_transfer_initial_guess, compute_time_to_nodes, hohmann_delta_v, plane_change_delta_v,
};
use crate::types::{
    EciBasicOrbit, InvalidTargetPropagationAuthorityCode, PairProxyModel, INVALID_COST,
};

use super::{normalized_excess, try_reserve_transfer_capacity};

#[inline]
pub(super) fn pair_verification_limit(pairs_to_verify: usize, available_pairs: usize) -> usize {
    if pairs_to_verify == 0 {
        available_pairs
    } else {
        pairs_to_verify.min(available_pairs)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PairProxyCandidate {
    pub(super) sat_idx: usize,
    pub(super) tgt_idx: usize,
    pub(super) x_hint: [f64; 3],
    pub(super) dv_proxy: f64,
    pub(super) time_proxy_s: f64,
    pub(super) rel_v_proxy: f64,
    pub(super) time_per_rel_v_proxy: f64,
    pub(super) cv_proxy: f64,
}

impl PairProxyCandidate {
    #[inline]
    const fn is_finite(&self) -> bool {
        self.dv_proxy.is_finite()
            && self.time_proxy_s.is_finite()
            && self.rel_v_proxy.is_finite()
            && self.time_per_rel_v_proxy.is_finite()
            && self.cv_proxy.is_finite()
    }
}

#[inline]
pub(super) fn retain_pair_proxy_candidate(
    candidate: &PairProxyCandidate,
    pairs_to_verify: usize,
) -> bool {
    pairs_to_verify == 0 || candidate.dv_proxy < INVALID_COST
}

#[derive(Debug, Default)]
pub struct PairProxyScratch {
    pub(super) candidates: Vec<PairProxyCandidate>,
    finite: Vec<PairProxyCandidate>,
    selected: Vec<PairProxyCandidate>,
    domination_counts: Vec<usize>,
    dominates: Vec<Vec<usize>>,
    current: Vec<usize>,
    next: Vec<usize>,
    split_rows: Vec<usize>,
    distances: Vec<f64>,
    sort_rows: Vec<(usize, PairProxyCandidate)>,
    order: Vec<(usize, f64)>,
    ranked: Vec<(usize, f64, PairProxyCandidate)>,
}

impl PairProxyScratch {
    pub(crate) fn new(_pair_capacity: usize) -> Self {
        // Capacity is fallibly established by `prepare` immediately before a
        // selection. Keeping construction allocation-free lets reusable batch
        // scratch exist without turning a failed reservation into a panic.
        Self::default()
    }

    pub(crate) fn prepare(
        &mut self,
        pair_capacity: usize,
    ) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        self.candidates.clear();
        self.finite.clear();
        self.selected.clear();
        self.domination_counts.clear();
        self.current.clear();
        self.next.clear();
        self.split_rows.clear();
        self.distances.clear();
        self.sort_rows.clear();
        self.order.clear();
        self.ranked.clear();
        reserve_pair_proxy_capacity(&mut self.candidates, pair_capacity)?;
        reserve_pair_proxy_capacity(&mut self.finite, pair_capacity)?;
        reserve_pair_proxy_capacity(&mut self.selected, pair_capacity)?;
        reserve_pair_proxy_capacity(&mut self.domination_counts, pair_capacity)?;
        reserve_pair_proxy_capacity(&mut self.dominates, pair_capacity)?;
        reserve_pair_proxy_capacity(&mut self.current, pair_capacity)?;
        reserve_pair_proxy_capacity(&mut self.next, pair_capacity)?;
        reserve_pair_proxy_capacity(&mut self.split_rows, pair_capacity)?;
        reserve_pair_proxy_capacity(&mut self.distances, pair_capacity)?;
        reserve_pair_proxy_capacity(&mut self.sort_rows, pair_capacity)?;
        reserve_pair_proxy_capacity(&mut self.order, pair_capacity)?;
        reserve_pair_proxy_capacity(&mut self.ranked, pair_capacity)?;
        for dominated in &mut self.dominates {
            dominated.clear();
        }
        Ok(())
    }

    pub(super) fn selected(&self) -> &[PairProxyCandidate] {
        &self.selected
    }
}

fn reserve_pair_proxy_capacity<T>(
    values: &mut Vec<T>,
    required_capacity: usize,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    if values.capacity() >= required_capacity {
        return Ok(());
    }
    // `try_reserve` takes capacity relative to the current length, while this
    // helper's contract is an absolute required capacity.
    let additional = required_capacity
        .checked_sub(values.len())
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    super::try_reserve_transfer_capacity(values, additional)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_pair_proxy_capacity_grows_to_required_total_after_clear(
    ) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        let mut values = Vec::with_capacity(8);
        values.push(0usize);
        let retained_capacity = values.capacity();
        values.clear();
        let required_capacity = retained_capacity
            .checked_add(1)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;

        reserve_pair_proxy_capacity(&mut values, required_capacity)?;

        if values.capacity() < required_capacity {
            return Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow);
        }
        Ok(())
    }

    #[test]
    fn reserve_pair_proxy_capacity_keeps_existing_capacity_fast_path(
    ) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        let mut values = Vec::with_capacity(8);
        values.push(0usize);
        let existing_capacity = values.capacity();
        values.clear();

        reserve_pair_proxy_capacity(&mut values, existing_capacity)?;

        if values.capacity() != existing_capacity {
            return Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow);
        }
        Ok(())
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(super) struct PairProxySelection {
    pub(super) selected: Vec<PairProxyCandidate>,
    pub(super) total_pairs: usize,
    pub(super) selected_pairs: usize,
    pub(super) selected_layers: usize,
    pub(super) omitted_layers: usize,
    pub(super) selected_by_target: [usize; 2],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PairProxySelectionMeta {
    pub(super) total_pairs: usize,
    pub(super) selected_pairs: usize,
    pub(super) selected_layers: usize,
    pub(super) omitted_layers: usize,
    pub(super) exact_mode: bool,
    pub(super) selected_by_target: [usize; 2],
}

#[cfg(test)]
pub(super) fn select_pair_proxy_candidates(
    candidates: Vec<PairProxyCandidate>,
    pairs_to_verify: usize,
) -> Result<PairProxySelection, InvalidTargetPropagationAuthorityCode> {
    let pair_capacity = candidates.len();
    let mut scratch = PairProxyScratch::new(pair_capacity);
    scratch.prepare(pair_capacity)?;
    scratch.candidates = candidates;
    let meta = select_pair_proxy_candidates_reuse(&mut scratch, pairs_to_verify)?;
    Ok(PairProxySelection {
        selected: scratch.selected,
        total_pairs: meta.total_pairs,
        selected_pairs: meta.selected_pairs,
        selected_layers: meta.selected_layers,
        omitted_layers: meta.omitted_layers,
        selected_by_target: meta.selected_by_target,
    })
}

pub(super) fn select_pair_proxy_candidates_reuse(
    scratch: &mut PairProxyScratch,
    pairs_to_verify: usize,
) -> Result<PairProxySelectionMeta, InvalidTargetPropagationAuthorityCode> {
    let total_pairs = scratch.candidates.len();
    scratch.finite.clear();
    scratch.selected.clear();
    if pairs_to_verify == 0 {
        // Exact mode attempts every satellite-target pair. Proxy non-finiteness
        // is not a physical infeasibility certificate and may not prune a pair.
        scratch.selected.extend_from_slice(&scratch.candidates);
        let selected_by_target = pair_proxy_target_counts(&scratch.selected)?;
        return Ok(PairProxySelectionMeta {
            selected_pairs: scratch.selected.len(),
            total_pairs,
            exact_mode: true,
            selected_by_target,
            ..PairProxySelectionMeta::default()
        });
    }
    scratch.finite.extend(
        scratch
            .candidates
            .iter()
            .copied()
            .filter(PairProxyCandidate::is_finite),
    );
    if scratch.finite.is_empty() {
        return Ok(PairProxySelectionMeta {
            total_pairs,
            ..PairProxySelectionMeta::default()
        });
    }

    let limit = pair_verification_limit(pairs_to_verify, scratch.finite.len());
    if limit == 0 {
        return Ok(PairProxySelectionMeta {
            total_pairs,
            ..PairProxySelectionMeta::default()
        });
    }

    let bounded = select_bounded_pair_proxy_fronts_reuse(scratch, limit)?;
    let selected_by_target = pair_proxy_target_counts(&scratch.selected)?;
    Ok(PairProxySelectionMeta {
        selected_pairs: scratch.selected.len(),
        total_pairs,
        selected_layers: bounded.selected_layers,
        omitted_layers: bounded.omitted_layers,
        exact_mode: false,
        selected_by_target,
    })
}

#[derive(Debug, Default)]
struct BoundedPairProxyFrontSelection {
    selected_layers: usize,
    omitted_layers: usize,
}

fn select_bounded_pair_proxy_fronts_reuse(
    scratch: &mut PairProxyScratch,
    limit: usize,
) -> Result<BoundedPairProxyFrontSelection, InvalidTargetPropagationAuthorityCode> {
    let candidates = scratch.finite.as_slice();
    let n = candidates.len();
    if n == 0 || limit == 0 {
        return Ok(BoundedPairProxyFrontSelection::default());
    }
    scratch.domination_counts.clear();
    scratch.domination_counts.resize(n, 0);
    while scratch.dominates.len() < n {
        scratch.dominates.push(Vec::new());
    }
    for dominated in scratch.dominates.iter_mut().take(n) {
        dominated.clear();
    }

    for (left, left_candidate) in candidates.iter().enumerate() {
        let right_start = left
            .checked_add(1)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        for (right, right_candidate) in candidates.iter().enumerate().skip(right_start) {
            if pair_proxy_dominates(left_candidate, right_candidate) {
                {
                    let dominated_rows = scratch
                        .dominates
                        .get_mut(left)
                        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                    dominated_rows
                        .try_reserve(1)
                        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                    dominated_rows.push(right);
                }
                let domination_count = scratch
                    .domination_counts
                    .get_mut(right)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                *domination_count = domination_count
                    .checked_add(1)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            } else if pair_proxy_dominates(right_candidate, left_candidate) {
                {
                    let dominated_rows = scratch
                        .dominates
                        .get_mut(right)
                        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                    dominated_rows
                        .try_reserve(1)
                        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                    dominated_rows.push(left);
                }
                let domination_count = scratch
                    .domination_counts
                    .get_mut(left)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                *domination_count = domination_count
                    .checked_add(1)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            }
        }
    }

    scratch.current.clear();
    scratch.current.extend(
        scratch
            .domination_counts
            .iter()
            .enumerate()
            .filter_map(|(idx, &count)| (count == 0).then_some(idx)),
    );
    scratch.selected.clear();
    let mut selected_layers = 0usize;
    let mut omitted_layers = 0usize;

    while !scratch.current.is_empty() {
        if scratch
            .current
            .iter()
            .any(|&row| candidates.get(row).is_none())
        {
            return Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow);
        }
        sort_pair_proxy_rows(candidates, &mut scratch.current, &mut scratch.sort_rows)?;
        let remaining = limit
            .checked_sub(scratch.selected.len())
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        if remaining == 0 {
            omitted_layers = usize::from(scratch.selected.len() < n);
            break;
        }

        if scratch.current.len() > remaining {
            pair_proxy_split_order_reuse(
                candidates,
                &scratch.current,
                &mut scratch.split_rows,
                &mut scratch.distances,
                &mut scratch.sort_rows,
                &mut scratch.order,
                &mut scratch.ranked,
            )?;
            for &row in scratch.split_rows.iter().take(remaining) {
                scratch.selected.push(
                    *candidates
                        .get(row)
                        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?,
                );
            }
            selected_layers = selected_layers
                .checked_add(1)
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            omitted_layers = usize::from(scratch.selected.len() < n);
            break;
        }

        scratch.next.clear();
        for &row in &scratch.current {
            scratch.selected.push(
                *candidates
                    .get(row)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?,
            );
            for &dominated in scratch
                .dominates
                .get(row)
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?
            {
                let domination_count = scratch
                    .domination_counts
                    .get_mut(dominated)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                *domination_count = domination_count
                    .checked_sub(1)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                if *domination_count == 0 {
                    scratch.next.push(dominated);
                }
            }
        }
        selected_layers = selected_layers
            .checked_add(1)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        if scratch.selected.len() >= limit {
            omitted_layers = usize::from(scratch.selected.len() < n);
            break;
        }
        std::mem::swap(&mut scratch.current, &mut scratch.next);
    }

    Ok(BoundedPairProxyFrontSelection {
        selected_layers,
        omitted_layers,
    })
}

#[inline]
fn pair_proxy_dominates(left: &PairProxyCandidate, right: &PairProxyCandidate) -> bool {
    const EPS: f64 = 1.0e-12;
    let no_worse = left.cv_proxy <= right.cv_proxy + EPS
        && left.dv_proxy <= right.dv_proxy + EPS
        && left.time_per_rel_v_proxy <= right.time_per_rel_v_proxy + EPS;
    let strictly_better = left.cv_proxy + EPS < right.cv_proxy
        || left.dv_proxy + EPS < right.dv_proxy
        || left.time_per_rel_v_proxy + EPS < right.time_per_rel_v_proxy;
    no_worse && strictly_better
}

fn pair_proxy_split_order_reuse(
    candidates: &[PairProxyCandidate],
    front: &[usize],
    rows: &mut Vec<usize>,
    distances: &mut Vec<f64>,
    sort_rows: &mut Vec<(usize, PairProxyCandidate)>,
    order: &mut Vec<(usize, f64)>,
    ranked: &mut Vec<(usize, f64, PairProxyCandidate)>,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    rows.clear();
    if front.iter().any(|&row| candidates.get(row).is_none()) {
        return Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow);
    }
    if front.len() <= 2 {
        rows.extend_from_slice(front);
        return sort_pair_proxy_rows(candidates, rows, sort_rows);
    }

    distances.clear();
    distances.resize(front.len(), 0.0);
    for objective in 0..3 {
        order.clear();
        for (position, &row) in front.iter().enumerate() {
            let candidate = candidates
                .get(row)
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            order.push((position, pair_proxy_objective_value(candidate, objective)));
        }
        order.sort_by(|(_, left_value), (_, right_value)| {
            left_value
                .partial_cmp(right_value)
                .unwrap_or(Ordering::Equal)
        });
        let (first, min_value) = *order
            .first()
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        let (last, max_value) = *order
            .last()
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        *distances
            .get_mut(first)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)? = f64::INFINITY;
        *distances
            .get_mut(last)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)? = f64::INFINITY;
        let span = max_value - min_value;
        if span <= 0.0 || !span.is_finite() {
            continue;
        }
        for window in order.windows(3) {
            let (_, previous) = *window
                .first()
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            let (current_position, _) = *window
                .get(1)
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            let (_, next) = *window
                .get(2)
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            let distance = distances
                .get_mut(current_position)
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            if distance.is_finite() {
                *distance += (next - previous) / span;
            }
        }
    }

    ranked.clear();
    for (&row, &distance) in front.iter().zip(distances.iter()) {
        let candidate = *candidates
            .get(row)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        ranked.push((row, distance, candidate));
    }
    ranked.sort_by(
        |(_, left_distance, left_candidate), (_, right_distance, right_candidate)| {
            right_distance
                .partial_cmp(left_distance)
                .unwrap_or(Ordering::Equal)
                .then_with(|| pair_proxy_cmp(left_candidate, right_candidate))
        },
    );
    rows.extend(ranked.iter().map(|(row, _, _)| *row));
    Ok(())
}

fn sort_pair_proxy_rows(
    candidates: &[PairProxyCandidate],
    rows: &mut Vec<usize>,
    sort_rows: &mut Vec<(usize, PairProxyCandidate)>,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    sort_rows.clear();
    reserve_pair_proxy_capacity(sort_rows, rows.len())?;
    for row in rows.iter().copied() {
        let candidate = *candidates
            .get(row)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        sort_rows.push((row, candidate));
    }
    sort_rows.sort_by(|(_, left), (_, right)| pair_proxy_cmp(left, right));
    rows.clear();
    reserve_pair_proxy_capacity(rows, sort_rows.len())?;
    rows.extend(sort_rows.iter().map(|(row, _)| *row));
    Ok(())
}

#[inline]
const fn pair_proxy_objective_value(candidate: &PairProxyCandidate, objective: usize) -> f64 {
    match objective {
        0 => candidate.cv_proxy,
        1 => candidate.dv_proxy,
        _ => candidate.time_per_rel_v_proxy,
    }
}

#[inline]
pub(super) fn pair_proxy_time_per_relative_velocity(time_proxy_s: f64, rel_v_proxy: f64) -> f64 {
    let rel_v = rel_v_proxy.abs();
    if time_proxy_s.is_finite() && rel_v.is_finite() && rel_v > 0.0 {
        time_proxy_s / rel_v
    } else {
        INVALID_COST
    }
}

fn pair_proxy_cmp(left: &PairProxyCandidate, right: &PairProxyCandidate) -> Ordering {
    lex_cmp!(left, right;
        asc (cv_proxy),
        asc (dv_proxy),
        asc (time_per_rel_v_proxy),
        asc (time_proxy_s),
        desc (rel_v_proxy.abs()),
        int (tgt_idx),
        int (sat_idx),
    )
}

fn pair_proxy_target_counts(
    candidates: &[PairProxyCandidate],
) -> Result<[usize; 2], InvalidTargetPropagationAuthorityCode> {
    let mut counts = [0usize; 2];
    for candidate in candidates {
        let count = counts
            .get_mut(candidate.tgt_idx)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        *count = count
            .checked_add(1)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    }
    Ok(counts)
}
/// Pre-computed orbit properties for a single satellite.
/// Extracted to deduplicate the four branches in event preparation
/// that compute identical values into scratch vs local storage.
#[derive(Clone, Copy)]
pub struct SatOrbitProps {
    pub(super) sma_est: f64,
    pub(super) sma_orbit: f64,
    pub(super) period_orbit: f64,
    pub(super) orbit_cached: EciBasicOrbit,
    pub(super) orbit_valid: bool,
}

#[derive(Clone, Copy)]
pub(super) struct TargetOrbitInvariants {
    pub(super) period_cached: f64,
    pub(super) orbit_valid: bool,
    pub(super) sma: f64,
    pub(super) period: f64,
    pub(super) sma_norm: f64,
}

#[derive(Clone, Copy)]
pub(super) struct TargetScreenState<'a> {
    pub(super) eci: &'a [f64; 6],
    pub(super) equ: [f64; 6],
    pub(super) period_cached: f64,
    pub(super) orbit_valid: bool,
    pub(super) sma: f64,
    pub(super) period: f64,
    sma_norm: f64,
    kepler: [f64; 6],
    node_wait: f64,
}

pub(super) struct PairProxyTargetInput<'a> {
    pub(super) eci: &'a [f64; 6],
    pub(super) equ: [f64; 6],
    pub(super) orbit: TargetOrbitInvariants,
}

/// Compute orbit properties for a single satellite from its ECI state.
#[inline]
pub(super) fn compute_sat_orbit_props(sat_eci: &[f64; 6]) -> SatOrbitProps {
    let (sma_orbit, period_orbit, orbit_cached, orbit_valid) = EciBasicOrbit::from_eci(sat_eci)
        .map_or_else(
            || (0.0, 0.0, EciBasicOrbit::default(), false),
            |orbit| {
                let sma = orbit.sma;
                let period = 2.0 * std::f64::consts::PI * (sma * sma * sma / satpy_core::MU).sqrt();
                (sma, period, orbit, true)
            },
        );
    SatOrbitProps {
        sma_est: sma_orbit,
        sma_orbit,
        period_orbit,
        orbit_cached,
        orbit_valid,
    }
}

#[inline]
fn target_position_norm(target: &[f64; 6]) -> f64 {
    let [x, y, z, _, _, _] = *target;
    norm3(&[x, y, z])
}

#[inline]
pub(super) fn target_orbit_invariants(target: &[f64; 6]) -> TargetOrbitInvariants {
    let Some(orbit) = EciBasicOrbit::from_eci(target) else {
        return TargetOrbitInvariants {
            period_cached: 0.0,
            orbit_valid: false,
            sma: 0.0,
            period: 0.0,
            sma_norm: target_position_norm(target),
        };
    };

    let sma = orbit.sma;
    let period = if sma > 0.0 {
        std::f64::consts::TAU * ((sma * sma * sma) / MU).sqrt()
    } else {
        0.0
    };
    TargetOrbitInvariants {
        period_cached: period,
        orbit_valid: true,
        sma,
        period,
        sma_norm: if sma > 0.0 {
            sma
        } else {
            target_position_norm(target)
        },
    }
}

pub(super) fn make_pair_proxy_candidate(
    sat_idx: usize,
    tgt_idx: usize,
    satellite: &[f64; 6],
    target: &[f64; 6],
    sat_props: &SatOrbitProps,
    target_sma_proxy: f64,
    target_period: f64,
    target_orbit_valid: bool,
    node_wait_s: f64,
    x_hint: [f64; 3],
    max_time_s: f64,
    pair_proxy_model: PairProxyModel,
) -> PairProxyCandidate {
    let hohmann = hohmann_delta_v(sat_props.sma_est, target_sma_proxy);
    let plane = plane_change_delta_v(satellite, target);
    let (sum_proxy, _) = crate::geometry::node_aware_estimate(hohmann, plane);
    let dv_proxy = match pair_proxy_model {
        PairProxyModel::Sum => sum_proxy,
        PairProxyModel::Combined => {
            // The node-crossing shortcut can still beat the cosine-law
            // combine, so rank by whichever estimate is tighter.
            let combined = crate::geometry::combined_burn_delta_v(
                sat_props.sma_est,
                target_sma_proxy,
                crate::geometry::plane_cos_between(satellite, target),
            );
            combined.min(sum_proxy)
        }
    };
    let rel_v_proxy = relative_velocity_proxy(satellite, target);
    let (time_proxy_s, time_cv) = pair_time_proxy_and_cv(
        satellite,
        target,
        sat_props.sma_orbit,
        target_sma_proxy,
        sat_props.period_orbit,
        target_period,
        node_wait_s,
        max_time_s,
    );
    let mut cv_proxy = time_cv;
    if !sat_props.orbit_valid || !target_orbit_valid {
        cv_proxy += 1.0;
    }
    if !dv_proxy.is_finite() || !rel_v_proxy.is_finite() || rel_v_proxy.abs() <= 0.0 {
        cv_proxy += 1.0;
    }
    let time_per_rel_v_proxy = pair_proxy_time_per_relative_velocity(time_proxy_s, rel_v_proxy);

    PairProxyCandidate {
        sat_idx,
        tgt_idx,
        x_hint,
        dv_proxy,
        time_proxy_s,
        rel_v_proxy,
        time_per_rel_v_proxy,
        cv_proxy,
    }
}

#[inline]
fn relative_velocity_proxy(satellite: &[f64; 6], target: &[f64; 6]) -> f64 {
    let [_, _, _, satellite_velocity_x, satellite_velocity_y, satellite_velocity_z] = *satellite;
    let [_, _, _, target_velocity_x, target_velocity_y, target_velocity_z] = *target;
    norm3(&[
        satellite_velocity_x - target_velocity_x,
        satellite_velocity_y - target_velocity_y,
        satellite_velocity_z - target_velocity_z,
    ])
}

pub(super) fn pair_time_proxy_and_cv(
    satellite: &[f64; 6],
    target: &[f64; 6],
    sat_sma: f64,
    target_sma: f64,
    sat_period: f64,
    target_period: f64,
    node_wait_s: f64,
    max_time_s: f64,
) -> (f64, f64) {
    let transfer_sma = 0.5 * (sat_sma + target_sma);
    let hohmann_time = if transfer_sma > 0.0 && transfer_sma.is_finite() {
        std::f64::consts::PI * ((transfer_sma * transfer_sma * transfer_sma) / MU).sqrt()
    } else {
        f64::INFINITY
    };
    let phase_wait = phase_wait_proxy(satellite, target, sat_sma);
    let node_wait = node_wait_s;
    let wait_proxy = phase_wait.min(node_wait).min(sat_period).min(target_period);
    let raw_time = hohmann_time + wait_proxy.max(0.0);
    let bounded_time = if raw_time.is_finite() {
        raw_time.clamp(0.0, max_time_s.max(0.0))
    } else {
        max_time_s.max(0.0)
    };
    let cv = if raw_time.is_finite() {
        normalized_excess(raw_time, max_time_s)
    } else {
        1.0
    };
    (bounded_time, cv)
}

#[cfg(test)]
pub(super) fn pair_x_hint(satellite: &[f64; 6], target: &[f64; 6], max_time_s: f64) -> [f64; 3] {
    let mut dep_kep = [0.0_f64; 6];
    let mut tgt_kep = [0.0_f64; 6];
    satpy_core::eci2kep_impl(satellite, false, true, &mut dep_kep);
    satpy_core::eci2kep_impl(target, false, true, &mut tgt_kep);
    pair_x_hint_from_kepler(&dep_kep, &tgt_kep, max_time_s)
}

#[inline]
pub(super) fn pair_x_hint_from_kepler(
    dep_kep: &[f64; 6],
    tgt_kep: &[f64; 6],
    max_time_s: f64,
) -> [f64; 3] {
    let [dep_sma, _, dep_inclination, _, _, dep_phase] = *dep_kep;
    let [tgt_sma, _, tgt_inclination, _, _, tgt_phase] = *tgt_kep;
    if dep_kep.iter().all(|value| value.is_finite())
        && tgt_kep.iter().all(|value| value.is_finite())
        && dep_sma > 0.0
        && tgt_sma > 0.0
    {
        let phase_angle = (tgt_phase - dep_phase).rem_euclid(std::f64::consts::TAU);
        combined_transfer_initial_guess(
            dep_sma,
            dep_inclination,
            tgt_sma,
            tgt_inclination,
            phase_angle,
            max_time_s,
        )
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "proxy ranking keeps its established non-fused IEEE rounding"
)]
fn phase_wait_proxy(satellite: &[f64; 6], target: &[f64; 6], sat_sma: f64) -> f64 {
    if !(sat_sma > 0.0 && sat_sma.is_finite()) {
        return f64::INFINITY;
    }
    let [sat_x, sat_y, sat_z, _, _, _] = *satellite;
    let [tgt_x, tgt_y, tgt_z, _, _, _] = *target;
    let r_sat = [sat_x, sat_y, sat_z];
    let r_tgt = [tgt_x, tgt_y, tgt_z];
    let r_sat_norm = norm3(&r_sat);
    let r_tgt_norm = norm3(&r_tgt);
    if !(r_sat_norm > 0.0 && r_tgt_norm > 0.0) {
        return f64::INFINITY;
    }
    let dot = (sat_x * tgt_x + sat_y * tgt_y + sat_z * tgt_z) / (r_sat_norm * r_tgt_norm);
    let phase_angle = dot.clamp(-1.0, 1.0).acos();
    let mean_motion = (MU / (sat_sma * sat_sma * sat_sma)).sqrt();
    if mean_motion > 0.0 && mean_motion.is_finite() {
        phase_angle / mean_motion
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
pub(super) fn node_wait_proxy(satellite: &[f64; 6], target: &[f64; 6]) -> f64 {
    node_wait_proxy_from_min_times(
        node_wait_min_from_eci(satellite),
        node_wait_min_from_eci(target),
    )
}

#[inline]
#[cfg(test)]
pub(super) fn node_wait_min_from_eci(state: &[f64; 6]) -> f64 {
    node_wait_min_from_kepler(&kepler_from_eci(state))
}

#[inline]
fn node_wait_min_from_kepler(kep: &[f64; 6]) -> f64 {
    compute_time_to_nodes(kep).map_or(f64::INFINITY, |(ascending, descending)| {
        ascending.min(descending)
    })
}

#[inline]
pub(super) fn kepler_from_eci(state: &[f64; 6]) -> [f64; 6] {
    let mut kep = [0.0_f64; 6];
    satpy_core::eci2kep_impl(state, false, true, &mut kep);
    kep
}

#[inline]
pub(super) const fn node_wait_proxy_from_min_times(sat_node: f64, target_node: f64) -> f64 {
    sat_node.max(target_node)
}

/// Build the two target screen states, then append retained pair-proxy rows in
/// the historical satellite-major, target-minor order.
///
/// Keeping this setup separate from `prepare_event` shortens the public event
/// constructor without changing the state construction or proxy comparison
/// schedule.
pub(super) fn screen_pair_proxy_candidates<'a>(
    target_inputs: [PairProxyTargetInput<'a>; 2],
    satellites: &[[f64; 6]],
    sat_props: &[SatOrbitProps],
    max_time_s: f64,
    pairs_to_verify: usize,
    pair_proxy_model: PairProxyModel,
    pair_proxy_scratch: &mut PairProxyScratch,
) -> Result<[TargetScreenState<'a>; 2], InvalidTargetPropagationAuthorityCode> {
    let [target_one, target_two] = target_inputs;
    let target_one_kepler = kepler_from_eci(target_one.eci);
    let target_two_kepler = kepler_from_eci(target_two.eci);
    let target_one_node_wait = node_wait_min_from_kepler(&target_one_kepler);
    let target_two_node_wait = node_wait_min_from_kepler(&target_two_kepler);
    let target_states = [
        TargetScreenState {
            eci: target_one.eci,
            equ: target_one.equ,
            period_cached: target_one.orbit.period_cached,
            orbit_valid: target_one.orbit.orbit_valid,
            sma: target_one.orbit.sma,
            period: target_one.orbit.period,
            sma_norm: target_one.orbit.sma_norm,
            kepler: target_one_kepler,
            node_wait: target_one_node_wait,
        },
        TargetScreenState {
            eci: target_two.eci,
            equ: target_two.equ,
            period_cached: target_two.orbit.period_cached,
            orbit_valid: target_two.orbit.orbit_valid,
            sma: target_two.orbit.sma,
            period: target_two.orbit.period,
            sma_norm: target_two.orbit.sma_norm,
            kepler: target_two_kepler,
            node_wait: target_two_node_wait,
        },
    ];
    let mut satellite_keplers = Vec::new();
    try_reserve_transfer_capacity(&mut satellite_keplers, satellites.len())?;
    for satellite in satellites {
        satellite_keplers.push(kepler_from_eci(satellite));
    }
    let mut satellite_node_waits = Vec::new();
    try_reserve_transfer_capacity(&mut satellite_node_waits, satellite_keplers.len())?;
    for satellite_kepler in &satellite_keplers {
        satellite_node_waits.push(node_wait_min_from_kepler(satellite_kepler));
    }
    for (sat_idx, (((satellite, sat_kepler), &sat_node_wait), sat_props)) in satellites
        .iter()
        .zip(&satellite_keplers)
        .zip(&satellite_node_waits)
        .zip(sat_props)
        .enumerate()
    {
        for (tgt_idx, target) in target_states.iter().enumerate() {
            let node_wait_s = node_wait_proxy_from_min_times(sat_node_wait, target.node_wait);
            let x_hint = pair_x_hint_from_kepler(sat_kepler, &target.kepler, max_time_s);
            let proxy = make_pair_proxy_candidate(
                sat_idx,
                tgt_idx,
                satellite,
                target.eci,
                sat_props,
                target.sma_norm,
                target.period,
                target.orbit_valid,
                node_wait_s,
                x_hint,
                max_time_s,
                pair_proxy_model,
            );
            if retain_pair_proxy_candidate(&proxy, pairs_to_verify) {
                pair_proxy_scratch.candidates.push(proxy);
            }
        }
    }
    Ok(target_states)
}
