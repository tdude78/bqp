//! Event detection and refinement for the Lightyear ODE integrator
//!
//! Implements event functions for terminal conditions (`ground`, `left_earth`, `nan_state`,
//! `eccentricity`) and non-terminal restart conditions (`perturb_deviation`).

// Event detection is actively used by integrator.rs when enable_events=true.

use crate::types::{EventDetection, EventType, InterpMethod, GROUND_ALTITUDE, MU, RE};
// Byte-identical local copy removed 2026-08-13; `satpy_core::norm3` is the
// same FMA-ordered body, so this import moves no bits.
use satpy_core::norm3;

#[inline]
#[must_use]
fn position_with_delta(delta: &[f64; 6], baseline: &[f64; 6]) -> [f64; 3] {
    let mut position = [0.0; 3];
    for ((component, baseline_component), correction) in
        position.iter_mut().zip(baseline).zip(delta)
    {
        *component = baseline_component + correction;
    }
    position
}

#[inline]
#[must_use]
fn state_with_delta(delta: &[f64; 6], baseline: &[f64; 6]) -> [f64; 6] {
    let mut state = [0.0; 6];
    for ((component, baseline_component), correction) in state.iter_mut().zip(baseline).zip(delta) {
        *component = baseline_component + correction;
    }
    state
}

/// WGS84 flattening, paired with [`RE`] as the semi-major axis.
///
/// Promoted to `satpy_core` 2026-07-25, which is the follow-up this comment used
/// to describe: `rhs.rs` asked for a shared home "when a second consumer
/// exists", this ground guard became that consumer, and both call sites moved
/// together.
use satpy_core::WGS84_FLATTENING;

/// WGS84 semi-minor (polar) radius in km: 6356.752314245 km, i.e. 21.385 km
/// below [`RE`]. That difference is exactly the error the ground guard used to
/// carry at the poles.
const RE_POLAR: f64 = RE * (1.0 - WGS84_FLATTENING);

/// Compute eccentricity squared from an ECI state.
#[inline]
fn compute_eccentricity_squared(state: &[f64; 6]) -> f64 {
    let &[rx, ry, rz, vx, vy, vz] = state;

    let r_mag_sq = rx * rx + ry * ry + rz * rz;
    if r_mag_sq < 1e-20 {
        return f64::NAN;
    }
    let r_mag = r_mag_sq.sqrt();

    let v_mag_sq = vx * vx + vy * vy + vz * vz;
    let r_dot_v = rx * vx + ry * vy + rz * vz;

    let inv_mu = 1.0 / MU;
    let coef1 = (v_mag_sq - MU / r_mag) * inv_mu;
    let coef2 = r_dot_v * inv_mu;

    let e_x = coef1 * rx - coef2 * vx;
    let e_y = coef1 * ry - coef2 * vy;
    let e_z = coef1 * rz - coef2 * vz;

    e_x * e_x + e_y * e_y + e_z * e_z
}

/// Check `perturb_deviation` event.
/// Returns squared distance from threshold if below, 0.0 if triggered
fn check_perturb_deviation(delta: &[f64; 6], _r_base: &[f64; 6]) -> f64 {
    let &[delta_x, delta_y, delta_z, ..] = delta;
    let pos_delta = (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z).sqrt();
    let mod_dist = pos_delta - crate::types::PERTURB_DEVIATION_THRESHOLD_KM;

    if mod_dist > 0.0 {
        0.0 // Triggered
    } else {
        mod_dist * mod_dist // Squared distance from threshold
    }
}

/// Check ground event (altitude < [`GROUND_ALTITUDE`] above the WGS84 ellipsoid)
///
/// Returns distance above the ground threshold in km (negative = below it).
/// The threshold surface is `R(phi) + GROUND_ALTITUDE`, where `R(phi)` is the
/// local ellipsoid radius at the state's geocentric latitude, NOT the constant
/// sphere `RE + GROUND_ALTITUDE` this used to compare against. The spherical
/// form was correct only on the equator and sat 21.385 km too high at the
/// poles, so it could declare impact for a vehicle still 21 km up.
///
/// The offset semantics are otherwise unchanged: the guard is still
/// "surface radius plus `GROUND_ALTITUDE`", with only the surface term moving.
///
/// # Formulation
///
/// `R(phi) = a*b / sqrt((a sin phi)^2 + (b cos phi)^2)` with `a` = [`RE`],
/// `b` = [`RE_POLAR`]. Substituting `sin phi = z/r`, `cos phi = rho/r`
/// (`rho^2 = x^2 + y^2`) clears the trig entirely:
///
/// `R = a*b*r / sqrt(a^2 z^2 + b^2 rho^2)`
///
/// # Cost
///
/// One extra `sqrt` over the old spherical form, and nothing else: no trig, no
/// `atan2`, no iteration, no branch on the hot path. A squared-radius
/// comparison would NOT save that `sqrt` — the guard must return a signed
/// distance for Brent to bracket in `check_event_crossing`, and squaring
/// `(R + GROUND_ALTITUDE)` leaves `R` appearing linearly anyway.
///
/// This deliberately does not use the Bowring geodetic reduction in `rhs.rs`,
/// which costs two `atan2` plus a `sin_cos`. Measured against it over
/// 0-90 deg latitude and 0-200 km altitude, the radial reduction used here
/// differs from true geodetic altitude by at most 1.09 m (worst case 45 deg at
/// 200 km) — four orders of magnitude below the 21.385 km defect being fixed.
///
/// # Frame caveat
///
/// `pos` is GCRS, and the WGS84 ellipsoid is Earth-fixed, so this flattens
/// about the GCRS z axis rather than the ITRS figure axis (the same known
/// inconsistency documented at length on `rhs::geodetic_altitude_km`). The
/// ~1.25e-3 rad pole tilt moves `R(phi)` by at most `(a-b) * tilt` ~ 27 m near
/// 45 deg, and by ~3 cm at the poles where `R` is stationary. Still 800x
/// better than the sphere it replaces; fixing it needs an ITRS rotation that
/// the event signature does not carry.
fn check_ground(delta: &[f64; 6], r_base: &[f64; 6]) -> f64 {
    let pos = position_with_delta(delta, r_base);
    let dist = norm3(&pos);

    let &[pos_x, pos_y, pos_z] = &pos;
    let rho_sq = pos_x.mul_add(pos_x, pos_y * pos_y);
    let z_sq = pos_z * pos_z;
    let denom_sq = (RE * RE).mul_add(z_sq, (RE_POLAR * RE_POLAR) * rho_sq);
    if denom_sq <= 0.0 {
        // Exactly at the geocentre: no latitude is defined. The nearest point
        // of the threshold surface is over a pole, so report that depth.
        return -RE_POLAR - GROUND_ALTITUDE;
    }

    let r_surface = RE * RE_POLAR * dist / denom_sq.sqrt();
    dist - r_surface - GROUND_ALTITUDE
}

/// Check `left_earth` event (infinite distance).
/// Returns inverse distance (approaches 0 at infinity)
fn check_left_earth(delta: &[f64; 6], r_base: &[f64; 6]) -> f64 {
    let pos = position_with_delta(delta, r_base);

    let &[pos_x, pos_y, pos_z] = &pos;
    let sum = pos_x + pos_y + pos_z;
    if !sum.is_finite() {
        return 0.0; // Triggered (escaped)
    }

    let dist = norm3(&pos);
    if dist < 1e-10 {
        return 1.0; // At origin - not escaped
    }
    1.0 / dist // Approaches 0 at infinity
}

/// Check `nan_state` event.
/// Returns 0.0 if NaN detected, 1.0 otherwise
fn check_nan_state(delta: &[f64; 6], r_base: &[f64; 6]) -> f64 {
    // Check baseline for NaN
    for &value in r_base.iter().take(6) {
        if value.is_nan() {
            return 0.0;
        }
    }

    // Full state
    let state = state_with_delta(delta, r_base);

    for &value in state.iter().take(6) {
        if value.is_nan() {
            return 0.0;
        }
    }
    1.0 // No NaN
}

/// Check eccentricity event (`e >= 1.0` = hyperbolic).
/// Returns 0.0 if hyperbolic, 1.0 otherwise
fn check_eccentricity(delta: &[f64; 6], r_base: &[f64; 6]) -> f64 {
    // Check baseline for NaN first
    for &value in r_base.iter().take(6) {
        if value.is_nan() {
            return 0.0;
        }
    }

    // Full ECI state
    let eci = state_with_delta(delta, r_base);

    let e_sq = compute_eccentricity_squared(&eci);
    if e_sq.is_nan() {
        return 0.0;
    }

    // Return 0 if hyperbolic (e >= 1 means e_sq >= 1), 1 otherwise
    if e_sq >= 1.0 {
        0.0
    } else {
        1.0
    }
}

/// Evaluate all event functions at current state.
pub fn evaluate_all_events(delta: &[f64; 6], r_base: &[f64; 6]) -> [f64; EventType::NUM_EVENTS] {
    [
        check_perturb_deviation(delta, r_base),
        check_ground(delta, r_base),
        check_left_earth(delta, r_base),
        check_nan_state(delta, r_base),
        check_eccentricity(delta, r_base),
    ]
}

/// Event state tracking during integration
pub struct EventState {
    pub(crate) prev_values: [f64; EventType::NUM_EVENTS],
    pub(crate) prev_time: f64,
    pub(crate) initialized: bool,
}

impl Default for EventState {
    fn default() -> Self {
        Self {
            prev_values: [
                1.0,    // perturb_deviation (squared dist from threshold)
                1000.0, // ground (positive = above ground)
                1.0,    // left_earth (inverse dist)
                1.0,    // nan_state (1 = no NaN)
                1.0,    // eccentricity (1 = bound orbit)
            ],
            prev_time: 0.0,
            initialized: false,
        }
    }
}

/// Check for event sign changes and refine if detected.
///
/// `prev_state` and `curr_state` are delta-states. Event functions are evaluated
/// against the baseline state from `get_baseline_fn`.
pub fn check_event_crossing(
    prev_state: &[f64; 6],
    curr_state: &[f64; 6],
    prev_dy: &[f64; 6],
    curr_dy: &[f64; 6],
    prev_time: f64,
    curr_time: f64,
    prev_values: &[f64; EventType::NUM_EVENTS],
    curr_values: &[f64; EventType::NUM_EVENTS],
    _t0_s: f64,
    get_baseline_fn: &mut dyn FnMut(f64) -> [f64; 6],
) -> EventDetection {
    let mut detection = EventDetection::default();
    let h = curr_time - prev_time;

    // Terminal events: check if value crossed to/through the triggering threshold.
    let terminal_events = [
        EventType::Ground,
        EventType::LeftEarth,
        EventType::NanState,
        EventType::Eccentricity,
    ];
    let &[prev_perturbation, prev_ground, prev_left_earth, prev_nan_state, prev_eccentricity] =
        prev_values;
    let &[curr_perturbation, curr_ground, curr_left_earth, curr_nan_state, curr_eccentricity] =
        curr_values;

    for &event in &terminal_events {
        let (prev_val, curr_val) = match event {
            EventType::Ground => (prev_ground, curr_ground),
            EventType::LeftEarth => (prev_left_earth, curr_left_earth),
            EventType::NanState => (prev_nan_state, curr_nan_state),
            EventType::Eccentricity => (prev_eccentricity, curr_eccentricity),
            EventType::PerturbDeviation => continue,
        };

        let triggered = match event {
            EventType::Ground => curr_val < 0.0 && prev_val >= 0.0,
            // These events are "approaches 0" style indicators.
            _ => curr_val < 1e-10 && prev_val > 1e-10,
        };

        if !triggered {
            continue;
        }

        detection.detected = true;
        detection.event_type = event;

        let refined_time = if event == EventType::Ground {
            // Refine ground crossing using Brent on the true ground function.
            let mut f = |t: f64| {
                let tau = (t - prev_time) / h;
                let delta =
                    crate::types::hermite_interp(prev_state, curr_state, prev_dy, curr_dy, h, tau);
                let r_base = get_baseline_fn(t);
                check_ground(&delta, &r_base)
            };

            brent_root(&mut f, prev_time, curr_time, 1e-9, 30)
        } else {
            // Discontinuous/threshold events: no reliable root; just midpoint.
            0.5 * (prev_time + curr_time)
        };

        detection.refined_time = refined_time;
        let tau = (refined_time - prev_time) / h;
        detection.state_at_event =
            crate::types::hermite_interp(prev_state, curr_state, prev_dy, curr_dy, h, tau);
        detection.interp_method = if event == EventType::Ground {
            InterpMethod::Hermite
        } else {
            InterpMethod::Linear
        };
        detection.interp_error = 0.0;

        return detection;
    }

    // perturb_deviation (non-terminal, triggers restart)
    {
        // Triggered when value becomes zero (position delta > threshold)
        if curr_perturbation < 1e-14 && prev_perturbation > 1e-14 {
            detection.detected = true;
            detection.event_type = EventType::PerturbDeviation;

            // Match C++: linear interpolation on the event function values.
            let mut alpha = prev_perturbation / (prev_perturbation - curr_perturbation + 1e-20);
            alpha = alpha.clamp(0.0, 1.0);

            detection.refined_time = prev_time + alpha * (curr_time - prev_time);
            for ((state_at_event, previous_state), current_state) in detection
                .state_at_event
                .iter_mut()
                .zip(prev_state)
                .zip(curr_state)
            {
                *state_at_event = previous_state + alpha * (current_state - previous_state);
            }
            detection.interp_method = InterpMethod::Linear;
            detection.interp_error = 0.0;

            return detection;
        }
    }

    detection
}

/// Brent root finder for mutable closures.
#[expect(
    clippy::float_cmp,
    reason = "Brent interpolation selection requires exact function-value equality for reproducible brackets"
)]
fn brent_root<F>(
    function: &mut F,
    mut bracket_left: f64,
    mut bracket_right: f64,
    tolerance: f64,
    max_iterations: usize,
) -> f64
where
    F: FnMut(f64) -> f64,
{
    let mut left_value = function(bracket_left);
    let mut right_value = function(bracket_right);

    if left_value * right_value > 0.0 {
        return f64::midpoint(bracket_left, bracket_right);
    }

    if left_value.abs() < right_value.abs() {
        std::mem::swap(&mut bracket_left, &mut bracket_right);
        std::mem::swap(&mut left_value, &mut right_value);
    }

    let mut reference_time = bracket_left;
    let mut reference_value = left_value;
    let mut candidate_time;
    let mut previous_time = 0.0;
    let mut used_bisection = true;

    for _ in 0..max_iterations {
        if right_value.abs() < tolerance {
            return bracket_right;
        }
        if (bracket_right - bracket_left).abs() < tolerance {
            return bracket_right;
        }

        if left_value != reference_value && right_value != reference_value {
            // Inverse quadratic interpolation
            candidate_time = bracket_left * right_value * reference_value
                / ((left_value - right_value) * (left_value - reference_value))
                + bracket_right * left_value * reference_value
                    / ((right_value - left_value) * (right_value - reference_value))
                + reference_time * left_value * right_value
                    / ((reference_value - left_value) * (reference_value - right_value));
        } else {
            // Secant
            candidate_time = bracket_right
                - right_value * (bracket_right - bracket_left) / (right_value - left_value);
        }

        let interpolation_bound = (3.0 * bracket_left + bracket_right) / 4.0;
        let outside_bracket = if bracket_left < bracket_right {
            candidate_time < interpolation_bound || candidate_time > bracket_right
        } else {
            candidate_time > interpolation_bound || candidate_time < bracket_right
        };
        let bisection_not_progressing = used_bisection
            && (candidate_time - bracket_right).abs()
                >= (bracket_right - reference_time).abs() / 2.0;
        let interpolation_not_progressing = !used_bisection
            && (candidate_time - bracket_right).abs()
                >= (reference_time - previous_time).abs() / 2.0;
        let bisection_bracket_small =
            used_bisection && (bracket_right - reference_time).abs() < tolerance;
        let interpolation_bracket_small =
            !used_bisection && (reference_time - previous_time).abs() < tolerance;

        if outside_bracket
            || bisection_not_progressing
            || interpolation_not_progressing
            || bisection_bracket_small
            || interpolation_bracket_small
        {
            candidate_time = f64::midpoint(bracket_left, bracket_right);
            used_bisection = true;
        } else {
            used_bisection = false;
        }

        let candidate_value = function(candidate_time);
        previous_time = reference_time;
        reference_time = bracket_right;
        reference_value = right_value;

        if left_value * candidate_value < 0.0 {
            bracket_right = candidate_time;
            right_value = candidate_value;
        } else {
            bracket_left = candidate_time;
            left_value = candidate_value;
        }

        if left_value.abs() < right_value.abs() {
            std::mem::swap(&mut bracket_left, &mut bracket_right);
            std::mem::swap(&mut left_value, &mut right_value);
        }
    }

    bracket_right
}

#[cfg(test)]
mod ground_ellipsoid_tests {
    use super::{check_ground, norm3, RE_POLAR, WGS84_FLATTENING};
    use crate::types::{GROUND_ALTITUDE, RE};

    /// Evaluate the guard at a bare ECI position (zero delta-state).
    fn ground_at(pos: [f64; 3]) -> f64 {
        let r_base = [pos[0], pos[1], pos[2], 0.0, 0.0, 0.0];
        check_ground(&[0.0; 6], &r_base)
    }

    /// What the guard returned before this fix, for the before/after margins.
    fn spherical_ground_at(pos: [f64; 3]) -> f64 {
        norm3(&pos) - RE - GROUND_ALTITUDE
    }

    /// Independent restatement of the ellipsoid radius, in the trigonometric
    /// form, so the algebraic simplification in `check_ground` is checked
    /// against something that is not itself.
    #[expect(
        clippy::imprecise_flops,
        reason = "independent expanded reference validates the production closed form without mirroring its algebra"
    )]
    fn ellipsoid_radius_at_geocentric_lat(phi_rad: f64) -> f64 {
        let (s, c) = phi_rad.sin_cos();
        let a_s = RE * s;
        let b_c = RE_POLAR * c;
        RE * RE_POLAR / (a_s * a_s + b_c * b_c).sqrt()
    }

    /// The semi-minor axis must be the published WGS84 polar radius. Without
    /// this the flattening is only checked against itself.
    #[test]
    fn polar_radius_is_the_published_wgs84_semi_minor_axis() {
        let published_km = 6_356_752.314_245_179 / 1000.0;
        assert!(
            (RE_POLAR - published_km).abs() < 1e-9,
            "RE_POLAR={RE_POLAR:.9} km, WGS84 semi-minor axis={published_km:.9} km \
             (f = 1/{:.9})",
            1.0 / WGS84_FLATTENING
        );
        assert!(
            (RE - RE_POLAR - 21.384_685_754_821).abs() < 1e-9,
            "equatorial-minus-polar must be 21.3847 km, got {:.9}",
            RE - RE_POLAR
        );
    }

    /// THE DEFECT. A vehicle over the pole at a radius between the polar and
    /// equatorial radii is above the real surface, and the spherical guard
    /// called it an impact.
    #[test]
    fn polar_state_above_the_ellipsoid_is_no_longer_a_false_impact() {
        // 6470 km geocentric: 113.25 km over the polar surface, i.e. above the
        // 100 km threshold, but 8.14 km "below" the spherical threshold.
        let pos = [0.0, 0.0, 6470.0];

        let before = spherical_ground_at(pos);
        let after = ground_at(pos);

        assert!(
            before < 0.0,
            "precondition: the spherical guard must have called this an impact, got {before:.6} km"
        );
        assert!(
            after > 0.0,
            "polar state at r=6470 km must be ABOVE ground: ellipsoidal guard says {after:+.6} km, \
             spherical guard said {before:+.6} km (false impact)"
        );
        // Margins, stated numerically: +13.248 km clear vs -8.137 km impacting,
        // a 21.385 km correction equal to a - b.
        assert!(
            (after - 13.247_685_754_821).abs() < 1e-6,
            "expected +13.2477 km above ground, got {after:+.9} km"
        );
        assert!(
            (before + 8.137_000_000_0).abs() < 1e-6,
            "expected -8.1370 km under the old guard, got {before:+.9} km"
        );
        assert!(
            (after - before - (RE - RE_POLAR)).abs() < 1e-9,
            "the correction at the pole must be exactly a - b = {:.6} km, got {:.6} km",
            RE - RE_POLAR,
            after - before
        );
    }

    /// SENSITIVITY. The fix must not have simply lowered the guard out of the
    /// way: 10 km below the polar threshold must still trigger, and the
    /// crossing must sit exactly at `RE_POLAR + GROUND_ALTITUDE`.
    #[test]
    fn polar_guard_still_fires_below_the_ellipsoidal_threshold() {
        let threshold = RE_POLAR + GROUND_ALTITUDE; // 6456.752314 km
        let below = ground_at([0.0, 0.0, threshold - 10.0]);
        let above = ground_at([0.0, 0.0, threshold + 10.0]);
        let at = ground_at([0.0, 0.0, threshold]);

        assert!(
            below < 0.0 && (below + 10.0).abs() < 1e-9,
            "10 km below the polar threshold must read -10 km, got {below:+.9} km"
        );
        assert!(
            above > 0.0 && (above - 10.0).abs() < 1e-9,
            "10 km above the polar threshold must read +10 km, got {above:+.9} km"
        );
        assert!(
            at.abs() < 1e-9,
            "the guard's root over the pole must be r = b + GROUND_ALTITUDE = {threshold:.6} km, \
             got residual {at:.3e} km"
        );
        // A metre of movement must be visible: the guard is a signed distance,
        // not a step.
        let one_m_up = ground_at([0.0, 0.0, threshold + 0.001]);
        assert!(
            (one_m_up - 0.001).abs() < 1e-9,
            "1 m above the threshold must read +0.001 km, got {one_m_up:+.9} km"
        );
    }

    /// The equator is where the old guard was already right, so the fix must
    /// change nothing measurable there.
    #[test]
    fn equatorial_guard_is_unchanged_by_the_fix() {
        for r in [6470.0_f64, 6478.137, 6500.0, 42164.0] {
            let pos = [r, 0.0, 0.0];
            let after = ground_at(pos);
            let before = spherical_ground_at(pos);
            assert!(
                (after - before).abs() < 1e-9,
                "equatorial guard moved at r={r:.3} km: {before:+.9} -> {after:+.9} km"
            );
            // And along +y, to show it is latitude and not x that matters.
            let after_y = ground_at([0.0, r, 0.0]);
            assert!(
                (after_y - before).abs() < 1e-9,
                "guard is not longitude-invariant at r={r:.3} km: {after_y:+.9} vs {before:+.9} km"
            );
        }
    }

    /// The closed form in `check_ground` must agree with the trigonometric
    /// statement of the same ellipsoid at every latitude, and must stay
    /// bracketed by the polar and equatorial spheres.
    #[test]
    fn guard_matches_the_ellipsoid_radius_at_every_latitude() {
        let mut worst = 0.0_f64;
        for deg in 0..=90 {
            let phi = f64::from(deg).to_radians();
            let r_surface = ellipsoid_radius_at_geocentric_lat(phi);
            let r = r_surface + GROUND_ALTITUDE + 25.0; // 25 km clear of the guard
            let (s, c) = phi.sin_cos();
            let value = ground_at([r * c, 0.0, r * s]);
            worst = worst.max((value - 25.0).abs());

            assert!(
                (RE_POLAR - 1e-12..=RE + 1e-12).contains(&r_surface),
                "ellipsoid radius at {deg} deg fell outside [b, a]: {r_surface:.9} km"
            );
        }
        assert!(
            worst < 1e-9,
            "closed form disagrees with the trigonometric ellipsoid radius by {worst:.3e} km"
        );
    }

    /// Degenerate input must stay negative (triggering) rather than go NaN,
    /// which would silently disable the terminal event.
    #[test]
    fn geocentre_reports_an_impact_rather_than_nan() {
        let value = ground_at([0.0, 0.0, 0.0]);
        assert!(
            value.is_finite() && value < 0.0,
            "the geocentre must read as an impact, got {value}"
        );
    }
}
