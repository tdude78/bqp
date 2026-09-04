//! Lambert solver backend.
//!
//! Uses the native Rust `izzo2015` implementation in `crate::lambert`.
//! C++ backend was deprecated in Phase 4 (2026-02-01) as Rust performance
//! reached parity.

use crate::evaluate::EvaluationArithmeticOverflow;
use satpy_core::{cross3, norm3, MU};

/// Immutable Lambert geometry shared by branch visitors for one time of flight.
#[derive(Clone, Copy)]
pub struct LambertProblem<'a> {
    r1_cache: &'a crate::lambert::LambertR1Cache,
    state1: &'a [f64; 6],
    state2: &'a [f64; 6],
    tof: f64,
}

impl<'a> LambertProblem<'a> {
    #[must_use]
    pub(crate) const fn new(
        r1_cache: &'a crate::lambert::LambertR1Cache,
        state1: &'a [f64; 6],
        state2: &'a [f64; 6],
        tof: f64,
    ) -> Self {
        Self {
            r1_cache,
            state1,
            state2,
            tof,
        }
    }
}

#[inline]
fn record_then_visit_lambert_solution(
    visit: &mut impl FnMut(i32, bool, bool, [f64; 3], [f64; 3]),
    rev: i32,
    low_path: bool,
    prograde: bool,
    dv_depart: [f64; 3],
    dv_arrive: [f64; 3],
) -> Result<(), EvaluationArithmeticOverflow> {
    crate::evaluate::record_lambert_branch_solution(rev, low_path, prograde)?;
    visit(rev, low_path, prograde, dv_depart, dv_arrive);
    Ok(())
}

/// Single-shot Lambert solver.
///
/// Solves Lambert problem and returns delta-V vectors.
///
/// # Arguments
/// * `r1` - Initial position [km]
/// * `r2` - Final position [km]
/// * `v1` - Initial velocity (for delta-V calculation) [km/s]
/// * `v2` - Final velocity (for delta-V calculation) [km/s]
/// * `tof` - Time of flight [s]
/// * `m` - Number of complete revolutions
/// * `prograde` - True for prograde transfer
/// * `lowpath` - True for low-path solution
///
/// # Returns
/// * `Some((dv_depart, dv_arrive))` on success
/// * `None` on solver failure
pub fn lambert_single_shot(
    r1: &[f64; 3],
    r2: &[f64; 3],
    v1: &[f64; 3],
    v2: &[f64; 3],
    tof: f64,
    m: i32,
    prograde: bool,
    lowpath: bool,
) -> Option<([f64; 3], [f64; 3])> {
    let res = crate::lambert::izzo2015_impl(
        MU,
        r1,
        r2,
        tof,
        m,
        prograde,
        lowpath,
        8,
        crate::lambert::CONVERGENCE_TOL,
        crate::lambert::CONVERGENCE_TOL,
    );
    if res.success {
        let [sol_v1_x, sol_v1_y, sol_v1_z] = res.v1;
        let [sol_v2_x, sol_v2_y, sol_v2_z] = res.v2;
        let &[v1_x, v1_y, v1_z] = v1;
        let &[v2_x, v2_y, v2_z] = v2;
        let dv1 = [sol_v1_x - v1_x, sol_v1_y - v1_y, sol_v1_z - v1_z];
        let dv2 = [v2_x - sol_v2_x, v2_y - sol_v2_y, v2_z - sol_v2_z];
        Some((dv1, dv2))
    } else {
        None
    }
}

/// Unpruned branch enumeration: the oracle the pruning tests compare against.
///
/// Production evaluates branches through
/// `visit_lambert_branch_solutions_pruned_with_r1`, which skips retrograde
/// lanes whose departure dv cannot beat the incumbent. This entry point takes
/// the same path with the prune disabled, so a test can assert the two agree
/// wherever the cap is below the bound. It has no non-test caller.
///
/// # Errors
///
/// Returns [`EvaluationArithmeticOverflow`] if
/// recording a valid branch would overflow its diagnostic counts. The failed
/// branch and every later branch are not passed to `visit`.
#[cfg(test)]
pub fn visit_lambert_branch_solutions(
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    m_max: i32,
    requested_low_path: bool,
    visit: impl FnMut(i32, bool, bool, [f64; 3], [f64; 3]),
) -> Result<(), EvaluationArithmeticOverflow> {
    visit_lambert_branch_solutions_pruned(
        state1,
        state2,
        tof,
        m_max,
        requested_low_path,
        true,
        visit,
    )
}

#[cfg(test)]
/// Test-only uncached branch visitor with a caller-selected retrograde lane.
///
/// # Errors
///
/// Returns [`EvaluationArithmeticOverflow`] if
/// branch diagnostic recording cannot represent a valid branch count.
pub fn visit_lambert_branch_solutions_pruned(
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    m_max: i32,
    requested_low_path: bool,
    include_retrograde: bool,
    visit: impl FnMut(i32, bool, bool, [f64; 3], [f64; 3]),
) -> Result<(), EvaluationArithmeticOverflow> {
    let r1 = [state1[0], state1[1], state1[2]];
    let r1_cache = crate::lambert::LambertR1Cache::new(&r1);
    visit_lambert_branch_solutions_pruned_with_r1(
        LambertProblem::new(&r1_cache, state1, state2, tof),
        m_max,
        requested_low_path,
        include_retrograde,
        visit,
    )
}

/// `visit_lambert_branch_solutions_pruned` fast entry taking a precomputed
/// departure-side cache.
///
/// `r1_cache` must be `LambertR1Cache::new` of `state1`'s position so callers
/// with a fixed departure state can hoist the r1 normalization out of a
/// per-TOF loop. Bit-identical to the uncached entry by `crate::lambert`
/// documentation (`LambertR1Cache`,
/// `for_each_lambert_m_prograde_lowpaths_pruned_with_r1`).
///
/// # Errors
///
/// Returns [`EvaluationArithmeticOverflow`] if
/// recording a valid branch would overflow its diagnostic counts. The failed
/// branch and every later branch are not passed to `visit`.
pub fn visit_lambert_branch_solutions_pruned_with_r1(
    problem: LambertProblem<'_>,
    m_max: i32,
    requested_low_path: bool,
    include_retrograde: bool,
    mut visit: impl FnMut(i32, bool, bool, [f64; 3], [f64; 3]),
) -> Result<(), EvaluationArithmeticOverflow> {
    let mut visit_error = None;
    crate::lambert::for_each_lambert_m_prograde_lowpaths_pruned_with_r1(
        MU,
        problem.r1_cache,
        problem.state1,
        problem.state2,
        problem.tof,
        m_max,
        requested_low_path,
        include_retrograde,
        |m, low_path, prograde, dv_depart, dv_arrive, valid| {
            if visit_error.is_some() {
                return;
            }
            if valid {
                if let Err(error) = record_then_visit_lambert_solution(
                    &mut visit, m, low_path, prograde, dv_depart, dv_arrive,
                ) {
                    visit_error = Some(error);
                }
            }
        },
    );
    visit_error.map_or(Ok(()), Err)
}

/// Cross-TOF streaming counterpart of
/// `visit_lambert_branch_solutions_pruned_with_r1`: one enumeration over many
/// `(state2, tof)` problems sharing a departure state, with the same per-branch
/// diagnostic recording per emitted solution.
///
/// Emission is problem-major in input order with per-problem bits identical to
/// the single-problem entry (see
/// `crate::lambert::for_each_lambert_m_prograde_lowpaths_pruned_with_r1_multi_tof`),
/// so folding `visit` per problem index reproduces the sequential path exactly.
///
/// # Errors
///
/// Returns [`EvaluationArithmeticOverflow`] if recording a valid branch would
/// overflow its diagnostic counts. The failed branch and every later branch
/// are not passed to `visit`.
pub fn visit_lambert_branch_solutions_pruned_with_r1_multi_tof(
    r1_cache: &crate::lambert::LambertR1Cache,
    state1: &[f64; 6],
    problems: &[crate::lambert::MultiTofBranchProblem],
    requested_low_path: bool,
    mut visit: impl FnMut(usize, i32, bool, bool, [f64; 3], [f64; 3]),
) -> Result<(), EvaluationArithmeticOverflow> {
    let mut visit_error = None;
    crate::lambert::for_each_lambert_m_prograde_lowpaths_pruned_with_r1_multi_tof(
        MU,
        r1_cache,
        state1,
        problems,
        requested_low_path,
        |problem_index, m, low_path, prograde, dv_depart, dv_arrive, valid| {
            if visit_error.is_some() {
                return;
            }
            if valid {
                if let Err(error) =
                    crate::evaluate::record_lambert_branch_solution(m, low_path, prograde)
                {
                    visit_error = Some(error);
                    return;
                }
                visit(problem_index, m, low_path, prograde, dv_depart, dv_arrive);
            }
        },
        || {},
    );
    visit_error.map_or(Ok(()), Err)
}

/// The departure-state half of the two branch-pruning bounds, computed once.
///
/// Both bounds below run on EVERY Lambert TOF sample, and both open by
/// re-deriving quantities that depend only on `state1` — a departure state
/// that is fixed for a whole plan evaluation, a whole pre-scan miss list, and
/// a whole `batch_retrograde_included` sweep. `max_revolutions_below_dv_cap`
/// alone was measured at 3.86% self CPU of an MF cell
/// (`docs/MF_COST_MAP.md`), for a body that is about thirty flops.
///
/// Same shape, and the same bit-identity argument, as the `LambertR1Cache`
/// hoist at `perf-hunt-r2 #3`: nothing is reassociated or approximated, the
/// hoisted expressions are moved verbatim, so every bound returns the bits it
/// returned before. `departure_bound_cache_matches_the_uncached_bounds` pins
/// that against the uncached entry points.
///
/// Deliberately NOT unified: `max_revolutions_below_dv_cap` rejects a
/// non-finite `state1` and `retrograde_departure_dv_lower_bound` does not.
/// This carries the finiteness flag so the first can keep its check and the
/// second can keep not having one.
#[derive(Clone, Copy, Debug)]
pub struct DepartureBoundCache {
    r1: [f64; 3],
    velocity: [f64; 3],
    /// `norm3(r1)`. Zero or negative means degenerate; both bounds bail on it.
    r1_norm: f64,
    /// `r1 / r1_norm`, only meaningful when `r1_norm > 0.0`.
    ir1: [f64; 3],
    /// `|v|^2`, in the association `departure_plane_split` uses.
    speed_squared: f64,
    /// Whether `r1` and the velocity are all finite. `r2` is checked per call.
    finite: bool,
}

impl DepartureBoundCache {
    /// Builds the cache for one departure state.
    #[must_use]
    pub fn new(state1: &[f64; 6]) -> Self {
        let &[r1_x, r1_y, r1_z, v1_x, v1_y, v1_z] = state1;
        let r1 = [r1_x, r1_y, r1_z];
        let velocity = [v1_x, v1_y, v1_z];
        let r1_norm = norm3(&r1);
        let ir1 = if r1_norm > 0.0 {
            [r1_x / r1_norm, r1_y / r1_norm, r1_z / r1_norm]
        } else {
            [0.0; 3]
        };
        Self {
            r1,
            velocity,
            r1_norm,
            ir1,
            speed_squared: v1_x.mul_add(v1_x, v1_y.mul_add(v1_y, v1_z * v1_z)),
            finite: r1
                .iter()
                .chain(velocity.iter())
                .all(|value| value.is_finite()),
        }
    }
}

/// Whole-batch retrograde inclusion predicate for the variable-r2 branch batch
/// solver.
///
/// The transfer-plane prograde basis rotates with each arrival
/// state, so a bound established against one lane does not cover the rest
/// (see `solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_scratch`'s
/// soundness contract). Retrograde must therefore stay included when ANY
/// lane's rigorous lower bound is below the acceptance cap — equivalently,
/// `min_over_lanes(bound) < dv_cap` — and may be dropped only when EVERY
/// lane's bound already meets or exceeds the cap.
pub fn batch_retrograde_included<'a>(
    dep_at_release: &[f64; 6],
    tgt_states: impl IntoIterator<Item = &'a [f64; 6]>,
    dv_cap: f64,
) -> bool {
    // One departure state across the whole sweep, so its half of the bound is
    // built once here rather than per lane. Bit-identical: the cached entry
    // moves the same expressions, it does not rewrite them.
    let departure = DepartureBoundCache::new(dep_at_release);
    tgt_states
        .into_iter()
        .any(|tgt_state| retrograde_departure_dv_lower_bound_cached(&departure, tgt_state) < dv_cap)
}

/// Rigorous lower bound on the departure dv of ANY retrograde Lambert branch
/// for this (state1, state2) pair: the deployer's tangential speed in the
/// transfer plane's prograde basis. Izzo retrograde solutions reverse the
/// tangential direction while their tangential speed is non-negative, so
/// projecting the dv onto the prograde tangential axis gives
/// `|dv| >= it1 . v_ref`. A 1e-9 km/s slack absorbs the float error of the
/// basis reconstruction; degenerate geometry returns 0.0 (never prunes).
/// **Test-only since the departure half was hoisted**, for the same reason as
/// [`max_revolutions_below_dv_cap`]: every production caller now goes through
/// [`retrograde_departure_dv_lower_bound_cached`], and this `cfg(test)` is what
/// keeps that true rather than a comment asking people to remember.
#[cfg(test)]
pub fn retrograde_departure_dv_lower_bound(state1: &[f64; 6], state2: &[f64; 6]) -> f64 {
    retrograde_departure_dv_lower_bound_cached(&DepartureBoundCache::new(state1), state2)
}

/// [`retrograde_departure_dv_lower_bound`] against a pre-built departure half.
///
/// `r1_norm` and `ir1` are the only `state1`-derived quantities the bound uses,
/// and both are lifted verbatim out of the per-sample path.
#[must_use]
pub fn retrograde_departure_dv_lower_bound_cached(
    departure: &DepartureBoundCache,
    state2: &[f64; 6],
) -> f64 {
    let [v1_x, v1_y, v1_z] = departure.velocity;
    let &[r2_x, r2_y, r2_z, ..] = state2;
    let r2 = [r2_x, r2_y, r2_z];
    let r1_norm = departure.r1_norm;
    let r2_norm = norm3(&r2);
    if r1_norm <= 0.0 || r2_norm <= 0.0 {
        return 0.0;
    }
    let ir1 = departure.ir1;
    let ir2 = [r2_x / r2_norm, r2_y / r2_norm, r2_z / r2_norm];
    let [ih_x, ih_y, ih_z] = cross3(&ir1, &ir2);
    let ih_norm = norm3(&[ih_x, ih_y, ih_z]);
    if ih_norm <= 0.0 {
        return 0.0;
    }
    let normalized_ih = [ih_x / ih_norm, ih_y / ih_norm, ih_z / ih_norm];
    let it1 = if ih_z < 0.0 {
        cross3(&ir1, &normalized_ih)
    } else {
        cross3(&normalized_ih, &ir1)
    };
    let [it1_x, it1_y, it1_z] = it1;
    let tangential_speed = it1_x * v1_x + it1_y * v1_y + it1_z * v1_z;
    (tangential_speed - 1e-9).max(0.0)
}

/// Relative time-of-flight slack absorbing the Izzo iteration's exit tolerance.
///
/// The bounds below are stated for the exact Lambert arc, but `izzo2015` stops
/// at `CONVERGENCE_TOL` on the non-dimensional `x`, so the reconstructed arc's
/// true time of flight differs from the requested one. Treating the arc as if
/// it had up to `1e-4` MORE time than requested only ever admits more
/// revolutions, so every bound below stays a lower bound on the dv the solver
/// actually reports. `CONVERGENCE_TOL` is `1e-6` on `x` and `dT/dx` is O(1)
/// away from the multi-rev minimum, so this is ~2 orders above the residual it
/// covers.
const TOF_RESIDUAL_SLACK_REL: f64 = 1e-4;

/// Absolute dv slack, mirroring [`retrograde_departure_dv_lower_bound`]: float
/// error in the basis reconstruction must never turn a feasible branch into a
/// pruned one.
const DV_BOUND_SLACK: f64 = 1e-9;

/// Rigorous lower bound on the departure dv of ANY Lambert branch at
/// `revolutions >= 1` for this `(state1, state2, tof)` triple.
///
/// Two facts bound the transfer velocity at r1, and they are independent:
///
/// 1. **Energy.** An arc that completes `m` full revolutions before arriving
///    spends more than `m` orbital periods in flight, so `m * T < tof` and
///    therefore `a < a_m = cbrt(mu * (tof / (2*pi*m))^2)`. Vis-viva at r1 then
///    gives `|v| = sqrt(mu * (2/|r1| - 1/a)) < sqrt(mu * (2/|r1| - 1/a_m))`,
///    which shrinks as `m` grows. Multi-rev arcs are elliptic by construction,
///    so this is the whole speed story; the `m = 0` branch admits hyperbolic
///    solutions and is deliberately NOT covered (the caller never prunes it).
/// 2. **Plane.** Izzo reconstructs `v1` from `ir1` and `it1`, both of which lie
///    in `span(r1, r2)`, for the prograde and the retrograde lane alike. The
///    deployer's velocity component normal to that plane therefore has to be
///    paid in full by the departure burn.
///
/// The two combine as a right triangle: `|dv|^2 >= v_perp^2 + max(0, |v_par| -
/// v_hi(m))^2`. Degenerate geometry (`|r1| = 0`, collinear `r1`/`r2`,
/// non-finite inputs) returns 0.0 and so never prunes.
///
/// Production never evaluates this per branch: it evaluates
/// [`max_revolutions_below_dv_cap`], which inverts this expression in closed
/// form. This entry point states the bound the inversion is derived from, so a
/// test can scan it directly and can check it against solved branches. It has
/// no non-test caller.
#[cfg(test)]
#[must_use]
pub fn multi_rev_departure_dv_lower_bound(
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    revolutions: i32,
) -> f64 {
    if revolutions < 1 {
        return 0.0;
    }
    let departure = DepartureBoundCache::new(state1);
    let Some(plane) = departure_plane_split(&departure, state2) else {
        return 0.0;
    };
    let r1_norm = departure.r1_norm;
    if r1_norm <= 0.0 || !tof.is_finite() || tof <= 0.0 {
        return 0.0;
    }
    let revolutions = f64::from(revolutions);
    let tof_slack = tof * (1.0 + TOF_RESIDUAL_SLACK_REL);
    let a_max = (MU * (tof_slack / (2.0 * std::f64::consts::PI * revolutions)).powi(2)).cbrt();
    let speed_max_squared = MU * (2.0 / r1_norm - 1.0 / a_max);
    let in_plane_gap = if speed_max_squared <= 0.0 {
        // No ellipse with a period this short reaches radius |r1| at all, so
        // the branch is empty; report the full in-plane speed as the gap
        // rather than infinity so the result stays a finite dv bound.
        plane.parallel_speed
    } else {
        (plane.parallel_speed - speed_max_squared.sqrt()).max(0.0)
    };
    let bound = plane
        .normal_speed
        .mul_add(plane.normal_speed, in_plane_gap * in_plane_gap)
        .sqrt();
    (bound - DV_BOUND_SLACK).max(0.0)
}

/// Largest revolution count in `0..=ceiling` whose
/// `multi_rev_departure_dv_lower_bound` is still below `dv_cap`.
///
/// Every branch above the returned count is bounded below by `dv_cap`, so it
/// can only produce solutions the caller's acceptance filters already reject
/// (`dv < dv_cap`). The bound is monotone non-decreasing in `m` — `a_m` shrinks
/// with `m`, so `v_hi(m)` shrinks and the gap grows — which makes "largest
/// admissible m" a single closed-form solve rather than a search:
///
/// ```text
/// bound(m) < cap
///   <=> (|v_par| - v_hi(m))^2 < cap^2 - v_perp^2
///   <=> v_hi(m) > v_need,  v_need = |v_par| - sqrt(cap^2 - v_perp^2)
///   <=> 1/a_m < 2/|r1| - v_need^2/mu = k
///   <=> m < tof * sqrt(mu * k^3) / (2*pi)
/// ```
///
/// The floor of that threshold is returned, which keeps the boundary case
/// `bound(m) == cap` (sound: acceptance is a strict `<`). Returns `ceiling`
/// unchanged whenever the geometry is degenerate or any input is non-finite,
/// so a bad input never prunes.
///
/// Production reaches this on every Lambert TOF sample, so it is two square
/// roots and no transcendental: the `cbrt` the bound is written with is
/// eliminated by comparing in `1/a` rather than in `a`.
/// **Test-only since the departure half was hoisted.** Production reaches
/// [`max_revolutions_below_dv_cap_cached`] from every one of its call sites,
/// and this `cfg(test)` is what keeps that true: a new production caller of
/// the uncached entry does not compile, so the departure state cannot start
/// being re-derived per sample again without someone noticing.
///
/// It stays as the oracle `departure_bound_cache_matches_the_uncached_bounds`
/// compares against, which is the whole proof that the hoist is bit-identical.
#[cfg(test)]
#[must_use]
pub fn max_revolutions_below_dv_cap(
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    dv_cap: f64,
    ceiling: i32,
) -> i32 {
    max_revolutions_below_dv_cap_cached(
        &DepartureBoundCache::new(state1),
        state2,
        tof,
        dv_cap,
        ceiling,
    )
}

/// [`max_revolutions_below_dv_cap`] against a pre-built departure half.
///
/// This is the entry production takes. The uncached one above states the
/// signature the bound is documented and tested against and forwards here, so
/// there is exactly one body.
#[must_use]
pub fn max_revolutions_below_dv_cap_cached(
    departure: &DepartureBoundCache,
    state2: &[f64; 6],
    tof: f64,
    dv_cap: f64,
    ceiling: i32,
) -> i32 {
    if ceiling <= 0 {
        return ceiling;
    }
    if !tof.is_finite() || tof <= 0.0 || dv_cap.is_nan() {
        return ceiling;
    }
    if dv_cap.is_infinite() && dv_cap > 0.0 {
        return ceiling;
    }
    let Some(plane) = departure_plane_split(departure, state2) else {
        return ceiling;
    };
    let r1_norm = departure.r1_norm;
    if r1_norm <= 0.0 {
        return ceiling;
    }
    let cap = dv_cap + DV_BOUND_SLACK;
    // Out-of-plane speed alone already meets the cap: no multi-rev branch can
    // land below it, and neither can any prograde/retrograde lane.
    let in_plane_budget_squared = cap.mul_add(cap, -(plane.normal_speed * plane.normal_speed));
    if in_plane_budget_squared <= 0.0 {
        return 0;
    }
    let speed_needed = plane.parallel_speed - in_plane_budget_squared.sqrt();
    if speed_needed <= 0.0 {
        // Even a parabolic-limit arc cannot be ruled out by speed alone.
        return ceiling;
    }
    let inverse_a_max = 2.0f64.mul_add(1.0 / r1_norm, -(speed_needed * speed_needed / MU));
    if inverse_a_max <= 0.0 {
        return 0;
    }
    let tof_slack = tof * (1.0 + TOF_RESIDUAL_SLACK_REL);
    let threshold = tof_slack * (MU * inverse_a_max * inverse_a_max * inverse_a_max).sqrt()
        / (2.0 * std::f64::consts::PI);
    if !threshold.is_finite() {
        return ceiling;
    }
    if threshold >= f64::from(ceiling) {
        return ceiling;
    }
    // Walk the (production: 0..=4) revolution range rather than casting the
    // float: `ceiling` is the caller's `max_revs`, so this is a handful of
    // compares and the floor stays exact.
    let mut capped = 0;
    for candidate in 1..=ceiling {
        if f64::from(candidate) > threshold {
            break;
        }
        capped = candidate;
    }
    capped
}

/// The deployer's velocity split about the `span(r1, r2)` transfer plane.
struct DeparturePlaneSplit {
    /// Speed normal to the transfer plane (unavoidable dv).
    normal_speed: f64,
    /// Speed inside the transfer plane.
    parallel_speed: f64,
}

/// Splits `state1`'s velocity about the transfer plane, or `None` when the
/// geometry is degenerate (collinear positions, zero norms, non-finite inputs).
fn departure_plane_split(
    departure: &DepartureBoundCache,
    state2: &[f64; 6],
) -> Option<DeparturePlaneSplit> {
    let r1 = departure.r1;
    let r2 = [state2[0], state2[1], state2[2]];
    let velocity = departure.velocity;
    if !departure.finite || !r2.iter().all(|value| value.is_finite()) {
        return None;
    }
    let normal = cross3(&r1, &r2);
    let normal_norm = norm3(&normal);
    if normal_norm <= 0.0 {
        return None;
    }
    let normal_speed =
        (velocity[0] * normal[0] + velocity[1] * normal[1] + velocity[2] * normal[2]) / normal_norm;
    let speed_squared = departure.speed_squared;
    Some(DeparturePlaneSplit {
        normal_speed: normal_speed.abs(),
        parallel_speed: (speed_squared - normal_speed * normal_speed)
            .max(0.0)
            .sqrt(),
    })
}

#[cfg(test)]
/// Test-only uncached exact branch visitor.
///
/// # Errors
///
/// Returns [`EvaluationArithmeticOverflow`] if
/// branch diagnostic recording cannot represent a valid branch count.
pub fn visit_lambert_exact_branch_solutions_pruned(
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    rev: i32,
    low_path: bool,
    include_retrograde: bool,
    visit: impl FnMut(i32, bool, bool, [f64; 3], [f64; 3]),
) -> Result<(), EvaluationArithmeticOverflow> {
    if rev < 0 {
        return Ok(());
    }
    let r1 = [state1[0], state1[1], state1[2]];
    let r1_cache = crate::lambert::LambertR1Cache::new(&r1);
    visit_lambert_exact_branch_solutions_pruned_with_r1(
        LambertProblem::new(&r1_cache, state1, state2, tof),
        rev,
        low_path,
        include_retrograde,
        visit,
    )
}

/// `visit_lambert_exact_branch_solutions_pruned` fast entry taking a
/// precomputed departure-side cache.
///
/// `r1_cache` must be `LambertR1Cache::new` of `state1`'s position; the
/// geometry build routes through `compute_lambert_geometry_with_r1`, which is
/// documented bit-identical to the uncached `compute_lambert_geometry`.
///
/// # Errors
///
/// Returns [`EvaluationArithmeticOverflow`] if
/// recording a valid branch would overflow its diagnostic counts. The failed
/// branch and every later branch are not passed to `visit`.
pub fn visit_lambert_exact_branch_solutions_pruned_with_r1(
    problem: LambertProblem<'_>,
    rev: i32,
    low_path: bool,
    include_retrograde: bool,
    mut visit: impl FnMut(i32, bool, bool, [f64; 3], [f64; 3]),
) -> Result<(), EvaluationArithmeticOverflow> {
    // R18: delegated to `crate::lambert::for_each_lambert_exact_branch_with_r1`,
    // which routes the selected branch through the singular production SIMD
    // pack. This makes the Brent /
    // pre-scan evaluations of a selected branch bit-identical to the
    // variable-r2 batch rows and to the per-candidate enumerator's pack lanes
    // for the same variant.
    let mut visit_error = None;
    crate::lambert::for_each_lambert_exact_branch_with_r1(
        MU,
        problem.r1_cache,
        problem.state1,
        problem.state2,
        problem.tof,
        rev,
        low_path,
        include_retrograde,
        |m, lane_low_path, prograde, dv_depart, dv_arrive, valid| {
            if visit_error.is_some() {
                return;
            }
            if valid {
                if let Err(error) = record_then_visit_lambert_solution(
                    &mut visit,
                    m,
                    lane_low_path,
                    prograde,
                    dv_depart,
                    dv_arrive,
                ) {
                    visit_error = Some(error);
                }
            }
        },
    );
    visit_error.map_or(Ok(()), Err)
}

/// Cross-TOF streaming counterpart of
/// [`visit_lambert_exact_branch_solutions_pruned_with_r1`]: one enumeration
/// over many `(state2, tof)` rows sharing a departure state and a selected
/// branch, with the same per-branch diagnostic recording per emitted solution.
///
/// Emission is row-major in input order with per-row bits identical to the
/// single-row entry (see
/// `crate::lambert::for_each_lambert_exact_branch_with_r1_multi_tof`), so folding
/// `visit` per row index reproduces the sequential path exactly.
///
/// # Errors
///
/// Returns [`EvaluationArithmeticOverflow`] if recording a valid branch would
/// overflow its diagnostic counts. The failed branch and every later branch
/// are not passed to `visit`.
pub fn visit_lambert_exact_branch_solutions_pruned_with_r1_multi_tof(
    r1_cache: &crate::lambert::LambertR1Cache,
    state1: &[f64; 6],
    problems: &[crate::lambert::MultiTofExactBranchProblem],
    mut visit: impl FnMut(usize, i32, bool, bool, [f64; 3], [f64; 3]),
) -> Result<(), EvaluationArithmeticOverflow> {
    let mut visit_error = None;
    crate::lambert::for_each_lambert_exact_branch_with_r1_multi_tof(
        MU,
        r1_cache,
        state1,
        problems,
        |problem_index, m, low_path, prograde, dv_depart, dv_arrive, valid| {
            if visit_error.is_some() {
                return;
            }
            if valid {
                if let Err(error) =
                    crate::evaluate::record_lambert_branch_solution(m, low_path, prograde)
                {
                    visit_error = Some(error);
                    return;
                }
                visit(problem_index, m, low_path, prograde, dv_depart, dv_arrive);
            }
        },
    );
    visit_error.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LEO-ish `(state1, state2, tof)` corpus spanning the
    /// altitudes, plane splits and flight times the MF lane actually solves.
    fn multi_rev_corpus() -> Vec<([f64; 6], [f64; 6], f64)> {
        let mut rows = Vec::new();
        let mut seed = 0x2026_0807_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = u32::try_from(seed >> 32).unwrap_or(0);
            f64::from(bits) / f64::from(u32::MAX)
        };
        for _ in 0..600 {
            let r1_norm = 6_700.0 + next() * 1_800.0;
            let angle1 = next() * std::f64::consts::TAU;
            let inclination = next() * 1.4;
            let r1 = [
                r1_norm * angle1.cos(),
                r1_norm * angle1.sin() * inclination.cos(),
                r1_norm * angle1.sin() * inclination.sin(),
            ];
            let circular = (MU / r1_norm).sqrt();
            // Velocity mostly along-track, with a deliberate out-of-plane tilt
            // so the plane term of the bound is exercised.
            let tilt = (next() - 0.5) * 0.4;
            let v1 = [
                -circular * angle1.sin() + tilt * circular * 0.3,
                circular * angle1.cos() * inclination.cos(),
                circular * angle1.cos() * inclination.sin() + tilt * circular * 0.2,
            ];
            let r2_norm = 6_700.0 + next() * 2_400.0;
            let angle2 = angle1 + 0.15 + next() * 5.9;
            let inclination2 = inclination + (next() - 0.5) * 0.25;
            let r2 = [
                r2_norm * angle2.cos(),
                r2_norm * angle2.sin() * inclination2.cos(),
                r2_norm * angle2.sin() * inclination2.sin(),
            ];
            let circular2 = (MU / r2_norm).sqrt();
            let v2 = [
                -circular2 * angle2.sin(),
                circular2 * angle2.cos() * inclination2.cos(),
                circular2 * angle2.cos() * inclination2.sin(),
            ];
            let tof = 600.0 + next() * 90_000.0;
            rows.push((
                [r1[0], r1[1], r1[2], v1[0], v1[1], v1[2]],
                [r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]],
                tof,
            ));
        }
        rows
    }

    /// The closed-form cap must agree exactly with a scan of the bound it is
    /// derived from: `max_revolutions_below_dv_cap` inverts
    /// `multi_rev_departure_dv_lower_bound` analytically, and an algebra slip
    /// there would silently prune a live branch.
    #[test]
    fn max_revolutions_cap_matches_a_scan_of_the_bound_it_inverts() {
        let mut saw_partial_cap = false;
        for (state1, state2, tof) in multi_rev_corpus() {
            for cap in [0.05, 0.25, 0.5, 1.0, 2.5, 6.0] {
                let ceiling = 6;
                let actual = max_revolutions_below_dv_cap(&state1, &state2, tof, cap, ceiling);
                let mut expected = ceiling;
                for revolutions in 1..=ceiling {
                    if multi_rev_departure_dv_lower_bound(&state1, &state2, tof, revolutions) >= cap
                    {
                        expected = revolutions - 1;
                        break;
                    }
                }
                if expected > 0 && expected < ceiling {
                    saw_partial_cap = true;
                }
                assert_eq!(
                    actual, expected,
                    "closed-form cap disagrees with the bound scan at cap={cap}, tof={tof}"
                );
            }
        }
        assert!(
            saw_partial_cap,
            "corpus must contain caps that land strictly inside 0..ceiling"
        );
    }

    /// Soundness, measured against the solver rather than argued: every
    /// multi-rev branch `izzo2015` actually returns must have a departure dv at
    /// or above the bound. The poison arm inflates the bound by 1.5x and has to
    /// break, or the corpus never approached the bound and the check is
    /// vacuous.
    #[test]
    fn multi_rev_bound_never_exceeds_a_solved_departure_dv() {
        let mut solved = 0_usize;
        let mut poisoned_violations = 0_usize;
        for (state1, state2, tof) in multi_rev_corpus() {
            let r1 = [state1[0], state1[1], state1[2]];
            let r2 = [state2[0], state2[1], state2[2]];
            let v1 = [state1[3], state1[4], state1[5]];
            let v2 = [state2[3], state2[4], state2[5]];
            for revolutions in 1..=4 {
                let bound = multi_rev_departure_dv_lower_bound(&state1, &state2, tof, revolutions);
                for low_path in [true, false] {
                    for prograde in [true, false] {
                        let Some((dv_depart, _)) = lambert_single_shot(
                            &r1,
                            &r2,
                            &v1,
                            &v2,
                            tof,
                            revolutions,
                            prograde,
                            low_path,
                        ) else {
                            continue;
                        };
                        let dv_norm = norm3(&dv_depart);
                        if !dv_norm.is_finite() {
                            continue;
                        }
                        solved += 1;
                        assert!(
                            bound <= dv_norm,
                            "multi-rev bound {bound} exceeds solved departure dv {dv_norm} \
                             at m={revolutions}, low_path={low_path}, prograde={prograde}, \
                             tof={tof}"
                        );
                        if bound * 1.5 > dv_norm {
                            poisoned_violations += 1;
                        }
                    }
                }
            }
        }
        assert!(
            solved > 1_000,
            "soundness corpus must solve a meaningful number of multi-rev branches, got {solved}"
        );
        assert!(
            poisoned_violations > 0,
            "a 1.5x-inflated bound must violate somewhere, or the corpus never \
             approaches the bound and the assertion above is vacuous"
        );
    }

    #[test]
    fn lambert_counter_overflow_stops_before_later_callback_mutates_output() {
        std::thread::spawn(|| {
            let before = crate::evaluate::evaluation_diagnostic_snapshot();
            let seeded = crate::evaluate::EvaluationDiagnosticCounters {
                lambert_branch_rev0_count: usize::MAX - 1,
                ..crate::evaluate::EvaluationDiagnosticCounters::default()
            };
            crate::evaluate::restore_evaluation_diagnostics(&seeded);

            let mut observed = Vec::new();
            let after_first;
            let overflow = {
                let mut visit = |rev, low_path, prograde, _, _| {
                    observed.push((rev, low_path, prograde));
                };
                record_then_visit_lambert_solution(
                    &mut visit,
                    0,
                    true,
                    true,
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                )
                .expect("first counter update must fit");
                after_first = crate::evaluate::evaluation_diagnostic_snapshot();
                record_then_visit_lambert_solution(
                    &mut visit,
                    0,
                    true,
                    true,
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                )
            };

            assert_eq!(overflow, Err(EvaluationArithmeticOverflow));
            assert_eq!(observed, vec![(0, true, true)]);
            assert_eq!(
                crate::evaluate::evaluation_diagnostic_snapshot(),
                after_first
            );
            crate::evaluate::restore_evaluation_diagnostics(&before);
        })
        .join()
        .expect("overflow isolation thread must not panic");
    }

    #[test]
    fn test_lambert_single_shot() {
        // Simple LEO transfer
        let r1 = [7000.0, 0.0, 0.0];
        let r2 = [0.0, 7500.0, 0.0];
        let v1 = [0.0, 7.5, 0.0];
        let v2 = [-7.3, 0.0, 0.0];
        let tof = 3600.0;

        let result = lambert_single_shot(&r1, &r2, &v1, &v2, tof, 0, true, true);
        assert!(result.is_some(), "Lambert should succeed");

        let (dv1, _dv2) = result.unwrap();
        let dv1_mag = (dv1[0] * dv1[0] + dv1[1] * dv1[1] + dv1[2] * dv1[2]).sqrt();
        assert!(dv1_mag < 10.0, "Delta-V should be reasonable");
    }

    #[test]
    fn test_lambert_batch() {
        let state1 = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let state2 = [0.0, 7500.0, 0.0, -7.3, 0.0, 0.0];
        let tof = 3600.0;

        let results = crate::lambert::izzo2015_batch_dv(MU, &state1, &state2, tof, 1, true);
        assert!(!results.is_empty(), "Should have results");

        // At least one should be valid
        let valid_count = results.iter().filter(|r| r.4).count();
        assert!(valid_count > 0, "At least one solution should be valid");
    }

    #[test]
    fn test_lambert_branch_enumeration_includes_high_path_multirev() {
        let state1 = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let state2 = [0.0, 7500.0, 0.0, -7.3, 0.0, 0.0];
        let tof = 7200.0;
        let mut saw_high_path_multirev = false;

        visit_lambert_branch_solutions(&state1, &state2, tof, 1, true, |m, low_path, _, _, _| {
            if m > 0 && !low_path {
                saw_high_path_multirev = true;
            }
        })
        .expect("test Lambert branch counters must fit usize");

        assert!(saw_high_path_multirev);
    }

    #[test]
    fn test_lambert_branch_enumeration_honors_large_m_max() {
        let r = 7000.0;
        let v = (MU / r).sqrt();
        let state1 = [r, 0.0, 0.0, 0.0, v, 0.0];
        let state2 = [0.0, r, 0.0, -v, 0.0, 0.0];
        let tof = 43200.0;
        let mut saw_m8_low = false;
        let mut saw_m8_high = false;

        visit_lambert_branch_solutions(&state1, &state2, tof, 8, true, |m, low_path, _, _, _| {
            if m == 8 && low_path {
                saw_m8_low = true;
            }
            if m == 8 && !low_path {
                saw_m8_high = true;
            }
        })
        .expect("test Lambert branch counters must fit usize");

        assert!(saw_m8_low, "expected low-path M=8 branch");
        assert!(saw_m8_high, "expected high-path M=8 branch");
    }

    #[test]
    fn selected_lambert_branch_visits_only_requested_rev_and_path() {
        let r = 7000.0;
        let v = (MU / r).sqrt();
        let state1 = [r, 0.0, 0.0, 0.0, v, 0.0];
        let state2 = [0.0, r, 0.0, -v, 0.0, 0.0];
        let tof = 43200.0;
        let mut visited = Vec::new();

        visit_lambert_exact_branch_solutions_pruned(
            &state1,
            &state2,
            tof,
            2,
            false,
            true,
            |m, low_path, _, _, _| visited.push((m, low_path)),
        )
        .expect("test Lambert branch counters must fit usize");

        assert!(!visited.is_empty(), "expected exact branch solutions");
        assert!(visited.iter().all(|entry| *entry == (2, false)));
    }

    #[test]
    fn test_lambert_batch_arrival_dv_matches_single_shot_convention() {
        let state1 = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let state2 = [0.0, 7500.0, 0.0, -7.3, 0.0, 0.0];
        let tof = 3600.0;

        let r1 = [state1[0], state1[1], state1[2]];
        let r2 = [state2[0], state2[1], state2[2]];
        let v1 = [state1[3], state1[4], state1[5]];
        let v2 = [state2[3], state2[4], state2[5]];
        let (_, single_arrival) =
            lambert_single_shot(&r1, &r2, &v1, &v2, tof, 0, true, true).unwrap();

        let batch = crate::lambert::izzo2015_batch_dv(MU, &state1, &state2, tof, 0, true)
            .into_iter()
            .find(|(m, prograde, _, _, valid)| *valid && *m == 0 && *prograde)
            .expect("expected M0 prograde solution");

        assert_eq!(batch.3.map(f64::to_bits), single_arrival.map(f64::to_bits));
    }
}

#[cfg(test)]
mod retro_prune_tests {
    use super::*;

    fn leo_pair() -> ([f64; 6], [f64; 6]) {
        let v1 = (MU / 6778.0_f64).sqrt();
        let v2 = (MU / 6878.0_f64).sqrt();
        let state1 = [6778.0, 0.0, 0.0, 0.0, v1 * 0.9, v1 * 0.43];
        let state2 = [0.0, 6878.0, 100.0, -v2, 0.0, 0.02];
        (state1, state2)
    }

    #[test]
    fn retro_bound_is_a_true_lower_bound_on_retrograde_dv() {
        let (state1, state2) = leo_pair();
        let bound = retrograde_departure_dv_lower_bound(&state1, &state2);
        assert!(
            bound > 0.0,
            "prograde-ish deployer must give positive bound"
        );
        for tof in [2400.0, 3600.0, 5400.0, 9000.0] {
            visit_lambert_branch_solutions(
                &state1,
                &state2,
                tof,
                4,
                true,
                |_m, _lp, prograde, dv_depart, _dv_arrive| {
                    if !prograde {
                        let dv = norm3(&dv_depart);
                        assert!(
                            dv >= bound,
                            "retrograde dv {dv} below bound {bound} at tof {tof}"
                        );
                    }
                },
            )
            .expect("test Lambert branch counters must fit usize");
        }
    }

    #[test]
    fn pruned_enumeration_matches_full_when_cap_below_bound() {
        let (state1, state2) = leo_pair();
        let bound = retrograde_departure_dv_lower_bound(&state1, &state2);
        // The cap sat at `bound * 0.5` until 2026-08-08, and at that value NO
        // branch solution survived the `dv >= cap` filter at any tested tof:
        // both arms held `None`, the bitwise compare passed as None == None,
        // and the parity claim had never once been exercised (established by
        // the non-vacuity assert below going red the moment it was added).
        // `cap = bound` keeps the property under test -- every retrograde
        // solution has dv >= bound and is filtered either way, so pruning them
        // cannot move the survivor set -- while letting sub-bound prograde
        // solutions actually populate the comparison.
        let cap = bound;
        let mut populated_tofs = 0_usize;
        for tof in [2400.0, 3600.0, 5400.0, 9000.0] {
            let collect = |include_retrograde: bool| {
                let mut best: Option<(f64, i32, bool, bool)> = None;
                visit_lambert_branch_solutions_pruned(
                    &state1,
                    &state2,
                    tof,
                    4,
                    true,
                    include_retrograde,
                    |m, lp, prograde, dv_depart, _| {
                        let dv = norm3(&dv_depart);
                        if dv >= cap {
                            return;
                        }
                        if best.is_none_or(|(b, _, _, _)| dv < b) {
                            best = Some((dv, m, lp, prograde));
                        }
                    },
                )
                .expect("test Lambert branch counters must fit usize");
                best
            };
            let full = collect(true);
            let pruned = collect(false);
            if full.is_some() {
                populated_tofs = populated_tofs.saturating_add(1);
            }
            assert_eq!(
                full.map(|(dv, m, lp, p)| (dv.to_bits(), m, lp, p)),
                pruned.map(|(dv, m, lp, p)| (dv.to_bits(), m, lp, p)),
                "tof {tof}: pruned best must be bitwise-equal to full best"
            );
        }
        // Non-vacuity: with no solution under the cap at ANY tof, every
        // compare above is None == None -- which is also what a broken
        // enumerator that visits nothing would produce, so the parity claim
        // would be about nothing. Measured 2026-08-08 at cap = bound: tofs
        // 2400, 3600 and 9000 populate; 5400 legitimately has no sub-bound
        // prograde solution. The floor is therefore 3, not 4, and a red here
        // means the survivor population collapsed, not that 5400 misbehaved.
        assert!(
            populated_tofs >= 3,
            "only {populated_tofs} of 4 tofs produced a survivor under cap \
             {cap}; the parity loop above mostly compared None == None"
        );
    }

    /// Batch lanes for the whole-batch prune tests: one departure state and
    /// several arrival states on distinct transfer planes (rotated per lane),
    /// sized past the SIMD chunk width so both the m=0 SIMD pass and the
    /// scalar multi-rev tail execute.
    fn batch_lanes() -> ([f64; 6], Vec<[f64; 6]>, Vec<f64>) {
        let (state1, base) = leo_pair();
        let mut lanes = Vec::new();
        let mut tofs = Vec::new();
        for (i, tof) in [2400.0, 3000.0, 3600.0, 5400.0, 7200.0, 9000.0]
            .into_iter()
            .enumerate()
        {
            let lane_index = f64::from(u32::try_from(i).expect("fixture lane index fits u32"));
            let angle = 0.05 * lane_index;
            let (sin_a, cos_a) = angle.sin_cos();
            lanes.push([
                base[0] * cos_a - base[1] * sin_a,
                base[0] * sin_a + base[1] * cos_a,
                base[2] + 25.0 * lane_index,
                base[3] * cos_a - base[4] * sin_a,
                base[3] * sin_a + base[4] * cos_a,
                base[5],
            ]);
            tofs.push(tof);
        }
        (state1, lanes, tofs)
    }

    fn solve_branch_batch(
        state1: &[f64; 6],
        lanes: &[[f64; 6]],
        tofs: &[f64],
        include_retrograde: bool,
        branch_selection: Option<(i32, bool)>,
    ) -> Vec<crate::lambert::BranchBatchTofResult> {
        solve_branch_batch_with_telemetry(state1, lanes, tofs, include_retrograde, branch_selection)
            .0
    }

    fn solve_branch_batch_with_telemetry(
        state1: &[f64; 6],
        lanes: &[[f64; 6]],
        tofs: &[f64],
        include_retrograde: bool,
        branch_selection: Option<(i32, bool)>,
    ) -> (
        Vec<crate::lambert::BranchBatchTofResult>,
        crate::lambert::VariableR2BranchTelemetry,
    ) {
        let r1 = [state1[0], state1[1], state1[2]];
        let v1_ref = [state1[3], state1[4], state1[5]];
        let r2_vec: Vec<[f64; 3]> = lanes.iter().map(|s| [s[0], s[1], s[2]]).collect();
        let v2_refs: Vec<[f64; 3]> = lanes.iter().map(|s| [s[3], s[4], s[5]]).collect();
        let mut scratch = crate::lambert::VariableR2LambertScratch::default();
        let rows =
            crate::lambert::solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_scratch(
                MU,
                &r1,
                &r2_vec,
                &v1_ref,
                &v2_refs,
                tofs,
                4,
                true,
                include_retrograde,
                branch_selection,
                &mut scratch,
            )
            .to_vec();
        (rows, scratch.branch_telemetry())
    }

    fn assert_branch_rows_bitwise_equal(
        full: &crate::lambert::BranchBatchTofResult,
        pruned: &crate::lambert::BranchBatchTofResult,
        lane_idx: usize,
    ) {
        assert_eq!(full.valid, pruned.valid, "lane {lane_idx}: valid drift");
        assert_eq!(full.m, pruned.m, "lane {lane_idx}: m drift");
        assert_eq!(
            full.low_path, pruned.low_path,
            "lane {lane_idx}: low_path drift"
        );
        assert_eq!(
            full.prograde, pruned.prograde,
            "lane {lane_idx}: prograde drift"
        );
        assert_eq!(
            full.tof.to_bits(),
            pruned.tof.to_bits(),
            "lane {lane_idx}: tof drift"
        );
        assert_eq!(
            full.dv_depart.to_bits(),
            pruned.dv_depart.to_bits(),
            "lane {lane_idx}: dv_depart drift"
        );
        assert_eq!(
            full.dv_arrive.to_bits(),
            pruned.dv_arrive.to_bits(),
            "lane {lane_idx}: dv_arrive drift"
        );
        for (axis, ((full_v1, pruned_v1), (full_v2, pruned_v2))) in full
            .v1
            .iter()
            .zip(&pruned.v1)
            .zip(full.v2.iter().zip(&pruned.v2))
            .enumerate()
        {
            assert_eq!(
                full_v1.to_bits(),
                pruned_v1.to_bits(),
                "lane {lane_idx}: v1[{axis}] drift"
            );
            assert_eq!(
                full_v2.to_bits(),
                pruned_v2.to_bits(),
                "lane {lane_idx}: v2[{axis}] drift"
            );
        }
    }

    /// Batch counterpart of `pruned_enumeration_matches_full_when_cap_below_bound`:
    /// when the acceptance cap sits below EVERY lane's retrograde bound, the
    /// pruned batch must be indistinguishable from the full batch after the
    /// downstream cap filter is applied.
    #[test]
    fn batch_pruned_solve_matches_full_when_cap_below_every_lane_bound() {
        let (state1, lanes, tofs) = batch_lanes();
        let bounds: Vec<f64> = lanes
            .iter()
            .map(|lane| retrograde_departure_dv_lower_bound(&state1, lane))
            .collect();
        let min_bound = bounds.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(min_bound > 0.0, "test lanes must all have a positive bound");
        let cap = min_bound * 0.5;
        assert!(
            !batch_retrograde_included(&state1, lanes.iter(), cap),
            "cap below every lane bound must allow the whole-batch prune"
        );

        let full = solve_branch_batch(&state1, &lanes, &tofs, true, None);
        let pruned = solve_branch_batch(&state1, &lanes, &tofs, false, None);
        assert_eq!(full.len(), pruned.len());

        let mut accepted_lanes = 0usize;
        for (lane_idx, (full_row, pruned_row)) in full.iter().zip(pruned.iter()).enumerate() {
            let full_accepted = full_row.valid && full_row.dv_depart < cap;
            let pruned_accepted = pruned_row.valid && pruned_row.dv_depart < cap;
            assert_eq!(
                full_accepted, pruned_accepted,
                "lane {lane_idx}: acceptance set drift under the cap filter"
            );
            if full_accepted {
                accepted_lanes += 1;
                assert!(
                    full_row.prograde,
                    "lane {lane_idx}: accepted-under-cap best must be prograde"
                );
                assert_branch_rows_bitwise_equal(full_row, pruned_row, lane_idx);
            }
        }
        assert!(
            accepted_lanes > 0,
            "test geometry must accept at least one prograde lane under the cap"
        );
    }

    /// CRITICAL direction check: the whole-batch predicate must be
    /// min-over-lanes (retrograde stays when ANY lane's bound is below the
    /// cap), not max-over-lanes.
    #[test]
    fn batch_retrograde_predicate_uses_min_over_lanes() {
        let (state1, lanes, _tofs) = batch_lanes();
        let bounds: Vec<f64> = lanes
            .iter()
            .map(|lane| retrograde_departure_dv_lower_bound(&state1, lane))
            .collect();
        let min_bound = bounds.iter().copied().fold(f64::INFINITY, f64::min);
        let max_bound = bounds.iter().copied().fold(0.0_f64, f64::max);
        assert!(min_bound > 0.0);

        // Homogeneous batch, cap below every bound: prune allowed.
        let cap = min_bound * 0.5;
        assert!(!batch_retrograde_included(&state1, lanes.iter(), cap));

        // Add one degenerate lane (r2 parallel to r1 -> bound 0.0): with the
        // same cap the MIN lane bound is now below the cap, so retrograde
        // must stay included even though every other lane's bound (and the
        // max) still exceeds the cap. A max-over-lanes wiring would wrongly
        // keep pruning here.
        let mut mixed = lanes.clone();
        mixed.push([
            state1[0] * 1.01,
            state1[1] * 1.01,
            state1[2] * 1.01,
            0.0,
            7.5,
            0.0,
        ]);
        assert_eq!(
            retrograde_departure_dv_lower_bound(&state1, mixed.last().unwrap()).to_bits(),
            0.0_f64.to_bits(),
            "degenerate lane must carry a zero bound"
        );
        assert!(
            batch_retrograde_included(&state1, mixed.iter(), cap),
            "one lane below the cap must force retrograde inclusion (min-over-lanes)"
        );

        // Cap above every bound: retrograde must stay included.
        assert!(batch_retrograde_included(
            &state1,
            lanes.iter(),
            max_bound + 1.0
        ));

        // Zero cap rejects everything downstream; bounds (>= 0) can never be
        // strictly below it, so the prune is always allowed.
        assert!(!batch_retrograde_included(&state1, mixed.iter(), 0.0));
    }

    /// `branch_selection = Some(..)` interaction: when the selected branch's
    /// per-lane best is prograde anyway, the `include_retrograde` flag is a
    /// no-op — full and pruned batches are bitwise identical on every lane.
    #[test]
    fn batch_pruned_flag_is_noop_when_selected_branch_best_is_prograde() {
        let (state1, lanes, tofs) = batch_lanes();
        for selection in [Some((0, true)), Some((1, true)), Some((1, false))] {
            let full = solve_branch_batch(&state1, &lanes, &tofs, true, selection);
            let pruned = solve_branch_batch(&state1, &lanes, &tofs, false, selection);
            assert_eq!(full.len(), pruned.len());
            let mut prograde_best_lanes = 0usize;
            for (lane_idx, (full_row, pruned_row)) in full.iter().zip(pruned.iter()).enumerate() {
                if full_row.valid {
                    assert!(
                        full_row.prograde,
                        "lane {lane_idx}: geometry must keep the selected-branch best prograde \
                         for selection {selection:?}"
                    );
                    prograde_best_lanes += 1;
                }
                assert_branch_rows_bitwise_equal(full_row, pruned_row, lane_idx);
            }
            if selection == Some((0, true)) {
                assert!(
                    prograde_best_lanes > 0,
                    "selected M0 low-path branch must produce at least one valid lane"
                );
            }
        }
    }

    #[test]
    fn production_max_revs4_branch_best_reports_m0_simd_prefill() {
        let (state1, lanes, tofs) = batch_lanes();
        let (_rows, telemetry) =
            solve_branch_batch_with_telemetry(&state1, &lanes, &tofs, true, None);
        assert_eq!(telemetry.m0_simd_prefill_lanes, 4);
        assert_eq!(
            telemetry.m0_simd_valid_lanes + telemetry.m0_scalar_fallback_lanes,
            telemetry.m0_simd_prefill_lanes
        );

        let (_selected_rows, selected_telemetry) =
            solve_branch_batch_with_telemetry(&state1, &lanes, &tofs, true, Some((0, true)));
        assert_eq!(selected_telemetry.m0_simd_prefill_lanes, 0);
    }

    #[test]
    fn pruned_enumeration_skips_only_retrograde() {
        let (state1, state2) = leo_pair();
        let mut full = Vec::new();
        let mut pruned = Vec::new();
        visit_lambert_branch_solutions(&state1, &state2, 3600.0, 4, true, |m, lp, p, _, _| {
            full.push((m, lp, p));
        })
        .expect("test Lambert branch counters must fit usize");
        visit_lambert_branch_solutions_pruned(
            &state1,
            &state2,
            3600.0,
            4,
            true,
            false,
            |m, lp, p, _, _| pruned.push((m, lp, p)),
        )
        .expect("test Lambert branch counters must fit usize");
        // Both assertions below hold when the visitors emit nothing: `all` is
        // vacuously true on an empty `pruned`, and the count equality holds as
        // 0 == 0 when `full` is empty too. A visitor that stopped enumerating,
        // or a Lambert solve that stopped converging, would leave this test
        // green while checking no branch at all.
        //
        // At tof = 3600 s on a LEO pair only the zero-revolution branches have
        // solutions, so the real corpus is 2 visited branches -- one prograde,
        // one retrograde -- of which the pruned walk must keep the 1 prograde.
        // The floors are that full enumeration, so they fail the moment either
        // walk collapses rather than tracking whatever survives.
        assert!(
            full.len() >= 2,
            "unpruned branch enumeration must visit both m=0 branches, visited {}",
            full.len()
        );
        // `!is_empty()` rather than `>= 1` to match the sibling above: a floor of
        // one IS non-emptiness, and spelling it as a number only looked uniform.
        assert!(
            !pruned.is_empty(),
            "pruned branch enumeration must keep the prograde m=0 branch, visited {}",
            pruned.len()
        );
        assert!(pruned.iter().all(|(_, _, p)| *p));
        assert_eq!(full.iter().filter(|(_, _, p)| *p).count(), pruned.len());
    }

    /// The cached branch bounds must return the SAME BITS as the uncached
    /// entry points, on every row, or the hoist is a numerical change wearing
    /// a performance change's clothes.
    ///
    /// Rows are swept rather than hand-picked because the two bounds have
    /// several early returns each -- degenerate norms, non-finite inputs,
    /// caps that no branch can meet -- and a hoist that is right on the main
    /// path can still be wrong on one of those. The sweep walks arrival
    /// geometry across the sphere, and includes the collinear and non-finite
    /// rows on purpose.
    #[test]
    fn departure_bound_cache_matches_the_uncached_bounds() {
        let departure_states = [
            [7000.0, 0.0, 0.0, 0.0, 7.546, 0.0],
            [6800.0, 500.0, -300.0, -0.6, 7.3, 1.9],
            [0.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            [7000.0, 0.0, 0.0, f64::NAN, 7.5, 0.0],
        ];
        let mut rows = 0_u32;
        for state1 in departure_states {
            let cache = DepartureBoundCache::new(&state1);
            for step in 0..64_i32 {
                let angle = f64::from(step) * 0.0982;
                let radius = 7000.0 + f64::from(step) * 25.0;
                let state2 = [
                    radius * angle.cos(),
                    radius * angle.sin(),
                    f64::from(step - 32) * 40.0,
                    -7.4 * angle.sin(),
                    7.4 * angle.cos(),
                    0.2,
                ];
                for tof in [600.0, 5400.0, 21_600.0, f64::INFINITY] {
                    for dv_cap in [0.05, 0.4, 3.0, f64::INFINITY] {
                        for ceiling in [0_i32, 1, 4] {
                            let want = max_revolutions_below_dv_cap(
                                &state1, &state2, tof, dv_cap, ceiling,
                            );
                            let got = max_revolutions_below_dv_cap_cached(
                                &cache, &state2, tof, dv_cap, ceiling,
                            );
                            assert_eq!(want, got, "revs at state1={state1:?} tof={tof} cap={dv_cap} ceiling={ceiling}");
                            rows = rows.saturating_add(1);
                        }
                    }
                }
                let want = retrograde_departure_dv_lower_bound(&state1, &state2);
                let got = retrograde_departure_dv_lower_bound_cached(&cache, &state2);
                assert_eq!(
                    want.to_bits(),
                    got.to_bits(),
                    "retrograde bound at state1={state1:?} state2={state2:?}: {want:e} != {got:e}"
                );
            }
            // Collinear arrival: the plane is undefined and both bounds must
            // decline to prune, through the cached entry as through the other.
            let collinear = [
                state1[0] * 2.0,
                state1[1] * 2.0,
                state1[2] * 2.0,
                0.0,
                0.0,
                0.0,
            ];
            assert_eq!(
                max_revolutions_below_dv_cap(&state1, &collinear, 5400.0, 0.4, 4),
                max_revolutions_below_dv_cap_cached(&cache, &collinear, 5400.0, 0.4, 4),
            );
            assert_eq!(
                retrograde_departure_dv_lower_bound(&state1, &collinear).to_bits(),
                retrograde_departure_dv_lower_bound_cached(&cache, &collinear).to_bits(),
            );
        }
        assert!(rows > 8_000, "sweep degenerated to {rows} rows");
    }
}
