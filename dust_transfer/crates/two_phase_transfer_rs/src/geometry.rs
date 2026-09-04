//! Orbital geometry helpers for two-phase transfer.
//!
//! RTN basis, Hohmann estimates, plane change calculations, and B-plane projection.
//!
//! # Physical Defensibility References
//!
//! ## Hohmann Transfer
//! - Vallado, D.A. (2013) "Fundamentals of Astrodynamics and Applications", 4th Ed.
//!   Chapter 6: Orbital Maneuvers, Eqs. 6-6 through 6-8.
//! - Formula: `dV = |v_transfer - v_circular|` at each impulse point
//!
//! ## B-Plane Projection
//! - Vallado (2013), Chapter 8: B-plane Targeting
//! - ESA Space Debris Conference Paper SDC4-49: "Collision Avoidance for ESA Satellites"
//! - The B-plane is perpendicular to relative velocity at closest approach
//! - Covariance projection: `Cov_2D = P x Cov_3D x P^T`
//!
//! ## Plane Change Maneuvers
//! - Standard orbital mechanics: dV = 2 x v x sin(θ/2)
//! - Uses angular momentum vectors to determine inclination angle

use satpy_core::{cross3, norm3, MU};

const NODE_CROSSING_MARGIN: f64 = 1.2;

/// Compute Hohmann transfer delta-V between two circular orbits.
///
/// # Physical Defensibility
///
/// Reference: Vallado (2013) "Fundamentals of Astrodynamics", Chapter 6, Eqs. 6-6 to 6-8.
///
/// The Hohmann transfer is the minimum-energy two-impulse transfer between
/// circular coplanar orbits. The transfer orbit is an ellipse tangent to both
/// circular orbits.
///
/// Formulas (vis-viva equation):
/// - `v_circular = sqrt(μ/r)`
/// - `v_transfer = sqrt(μ(2/r - 1/a_t))`, where `a_t = (r1 + r2)/2`
/// - `dv1 = |v_transfer(r1) - v_circular(r1)|`
/// - `dv2 = |v_circular(r2) - v_transfer(r2)|`
/// - `total_dv = dv1 + dv2`
///
/// # Arguments
/// * `r1` - Radius of first circular orbit (km)
/// * `r2` - Radius of second circular orbit (km)
///
/// # Returns
/// Total delta-V (km/s) for the Hohmann transfer, or infinity if invalid
#[inline]
#[must_use]
pub fn hohmann_delta_v(r1: f64, r2: f64) -> f64 {
    if !r1.is_finite() || !r2.is_finite() || r1 <= 0.0 || r2 <= 0.0 {
        return f64::INFINITY;
    }

    // Transfer orbit semi-major axis
    let a_transfer = 0.5 * (r1 + r2);
    if a_transfer <= 0.0 || a_transfer.is_nan() {
        return f64::INFINITY;
    }

    // Vis-viva at r1 for circular and transfer orbits
    let v1_circ = (MU / r1).sqrt();
    let v1_transfer = (MU * (2.0 / r1 - 1.0 / a_transfer)).sqrt();

    // Vis-viva at r2 for circular and transfer orbits
    let v2_circ = (MU / r2).sqrt();
    let v2_transfer = (MU * (2.0 / r2 - 1.0 / a_transfer)).sqrt();

    let dv1 = (v1_transfer - v1_circ).abs();
    let dv2 = (v2_circ - v2_transfer).abs();

    dv1 + dv2
}

/// Compute plane change delta-V between two orbital planes.
///
/// # Physical Defensibility
///
/// Reference: Vallado (2013) "Fundamentals of Astrodynamics", Chapter 6.
///
/// A simple plane change at constant velocity uses:
/// - dv = 2 x v x sin(Δi/2)
///
/// where Δi is the angle between the two orbital planes, computed from
/// the angular momentum vectors: h = r x v
///
/// This is an approximation - combined maneuvers can be more efficient.
///
/// Uses the angular momentum vectors to determine plane angle,
/// and average velocity for the maneuver estimate.
///
/// # Arguments
/// * `dep_state` - Departure ECI state [x, y, z, vx, vy, vz]
/// * `tgt_state` - Target ECI state [x, y, z, vx, vy, vz]
///
/// # Returns
/// Plane change delta-V (km/s), or infinity if degenerate
#[inline]
#[must_use]
pub fn plane_change_delta_v(dep_state: &[f64; 6], tgt_state: &[f64; 6]) -> f64 {
    let &[dep_pos_x, dep_pos_y, dep_pos_z, dep_vel_x, dep_vel_y, dep_vel_z] = dep_state;
    let &[tgt_pos_x, tgt_pos_y, tgt_pos_z, tgt_vel_x, tgt_vel_y, tgt_vel_z] = tgt_state;
    let r_dep = [dep_pos_x, dep_pos_y, dep_pos_z];
    let v_dep = [dep_vel_x, dep_vel_y, dep_vel_z];
    let r_tgt = [tgt_pos_x, tgt_pos_y, tgt_pos_z];
    let v_tgt = [tgt_vel_x, tgt_vel_y, tgt_vel_z];

    let h_dep = cross3(&r_dep, &v_dep);
    let h_tgt = cross3(&r_tgt, &v_tgt);

    let h_dep_norm = norm3(&h_dep);
    let h_tgt_norm = norm3(&h_tgt);

    if h_dep_norm <= 0.0 || h_tgt_norm <= 0.0 || h_dep_norm.is_nan() || h_tgt_norm.is_nan() {
        return f64::INFINITY;
    }

    let [h_dep_x, h_dep_y, h_dep_z] = h_dep;
    let [h_tgt_x, h_tgt_y, h_tgt_z] = h_tgt;
    let dot_val =
        (h_dep_x * h_tgt_x + h_dep_y * h_tgt_y + h_dep_z * h_tgt_z) / (h_dep_norm * h_tgt_norm);
    let clamped_dot = dot_val.clamp(-1.0, 1.0);

    // Use average velocity for plane change estimate
    let v_dep_mag = norm3(&v_dep);
    let v_tgt_mag = norm3(&v_tgt);
    let v_avg = 0.5 * (v_dep_mag + v_tgt_mag);

    // Plane change dV = 2 * v * sin(angle/2), with sin(angle/2)
    // from cos(angle) to avoid acos+sin in the hot heuristic path.
    let sin_half = ((1.0 - clamped_dot) * 0.5).max(0.0).sqrt();
    2.0 * v_avg * sin_half
}

/// Cosine of the angle between two orbital planes (from angular momenta).
/// Returns NAN when either state is degenerate.
pub fn plane_cos_between(dep_state: &[f64; 6], tgt_state: &[f64; 6]) -> f64 {
    let &[dep_pos_x, dep_pos_y, dep_pos_z, dep_vel_x, dep_vel_y, dep_vel_z] = dep_state;
    let &[tgt_pos_x, tgt_pos_y, tgt_pos_z, tgt_vel_x, tgt_vel_y, tgt_vel_z] = tgt_state;
    let h_dep = cross3(
        &[dep_pos_x, dep_pos_y, dep_pos_z],
        &[dep_vel_x, dep_vel_y, dep_vel_z],
    );
    let h_tgt = cross3(
        &[tgt_pos_x, tgt_pos_y, tgt_pos_z],
        &[tgt_vel_x, tgt_vel_y, tgt_vel_z],
    );
    let h_dep_norm = norm3(&h_dep);
    let h_tgt_norm = norm3(&h_tgt);
    if h_dep_norm <= 0.0 || h_tgt_norm <= 0.0 || h_dep_norm.is_nan() || h_tgt_norm.is_nan() {
        return f64::NAN;
    }
    let [h_dep_x, h_dep_y, h_dep_z] = h_dep;
    let [h_tgt_x, h_tgt_y, h_tgt_z] = h_tgt;
    let dot_val =
        (h_dep_x * h_tgt_x + h_dep_y * h_tgt_y + h_dep_z * h_tgt_z) / (h_dep_norm * h_tgt_norm);
    dot_val.clamp(-1.0, 1.0)
}

/// Two-impulse Hohmann estimate with the plane change FOLDED INTO one burn
/// via the velocity-triangle cosine law, evaluating the rotation at the
/// departure burn vs the arrival burn and keeping the cheaper placement
/// (Vallado 2013, Ch. 6 combined maneuvers). Tighter than the separate
/// `hohmann + plane_change` sum, and reduces exactly to `hohmann_delta_v`
/// when the planes already align.
pub fn combined_burn_delta_v(r1: f64, r2: f64, cos_plane_angle: f64) -> f64 {
    if !r1.is_finite() || !r2.is_finite() || r1 <= 0.0 || r2 <= 0.0 {
        return f64::INFINITY;
    }
    if !cos_plane_angle.is_finite() {
        return f64::INFINITY;
    }
    let cos_gamma = cos_plane_angle.clamp(-1.0, 1.0);
    let a_transfer = 0.5 * (r1 + r2);
    if a_transfer <= 0.0 || a_transfer.is_nan() {
        return f64::INFINITY;
    }
    let v1_circ = (MU / r1).sqrt();
    let v2_circ = (MU / r2).sqrt();
    let v1_transfer = (MU * (2.0 / r1 - 1.0 / a_transfer)).max(0.0).sqrt();
    let v2_transfer = (MU * (2.0 / r2 - 1.0 / a_transfer)).max(0.0).sqrt();
    let rotate = |u: f64, w: f64| ((u * u + w * w - 2.0 * u * w * cos_gamma).max(0.0)).sqrt();
    let plane_at_arrival = (v1_transfer - v1_circ).abs() + rotate(v2_transfer, v2_circ);
    let plane_at_departure = rotate(v1_transfer, v1_circ) + (v2_transfer - v2_circ).abs();
    plane_at_arrival.min(plane_at_departure)
}

/// Node-aware transfer delta-V estimate.
///
/// Computes the minimum of:
/// 1. Direct estimate: `hohmann + plane_change` (traditional)
/// 2. Node-crossing estimate: `hohmann * margin` (exploits orbital geometry)
///
/// Node-crossing transfers avoid plane change cost by maneuvering at the
/// ascending/descending node where both orbital planes intersect.
///
/// # Arguments
/// * `hohmann` - Hohmann transfer delta-V (km/s)
/// * `plane` - Simple plane change delta-V (km/s)
///
/// # Returns
/// Tuple of (`estimate`, `used_node_crossing`) where:
/// - estimate: minimum of direct and node-crossing estimates
/// - `used_node_crossing`: true if node-crossing estimate was lower
///
/// The margin is the SEALED CONSTANT `NODE_CROSSING_MARGIN` (1.2, defined
/// above), not a knob. This section used to be titled "# Environment
/// Variables" and named `NASA_DUST_NODE_CROSSING_MARGIN`, which nothing in the
/// workspace reads -- setting it changed nothing and said nothing, which in a
/// bit-sealed tree is the worst way to be wrong. Moving the value moves every
/// digest that depends on this estimate, so it changes by reseal, not by env.
#[inline]
pub fn node_aware_estimate(hohmann: f64, plane: f64) -> (f64, bool) {
    let margin = NODE_CROSSING_MARGIN;

    // Handle infinity cases explicitly
    if !hohmann.is_finite() && !plane.is_finite() {
        return (f64::INFINITY, false);
    }

    if !hohmann.is_finite() {
        // Hohmann failed but plane is valid - use plane as estimate
        return (
            if plane.is_finite() {
                plane
            } else {
                f64::INFINITY
            },
            false,
        );
    }

    if !plane.is_finite() {
        // Plane failed but hohmann is valid - use node_crossing estimate
        return (hohmann * margin, true);
    }

    // Normal case: both finite
    let direct_estimate = hohmann + plane;
    let node_crossing_estimate = hohmann * margin;

    if node_crossing_estimate < direct_estimate {
        (node_crossing_estimate, true)
    } else {
        (direct_estimate, false)
    }
}

/// Compute time to reach ascending/descending nodes from current position.
///
/// Returns (`t_to_AN`, `t_to_DN`) in seconds, where:
/// - `t_to_AN`: time to reach ascending node
/// - `t_to_DN`: time to reach descending node
///
/// # Arguments
/// * `kep` - Keplerian elements [a, e, i, RAAN, omega, nu] (km, rad)
///
/// # Returns
/// `Some((t_to_AN, t_to_DN))` for valid elliptical orbits, `None` otherwise.
///
/// # Physical Defensibility
///
/// The ascending node is where the orbit crosses the equatorial plane going northward.
/// The true anomaly at AN is: `nu_AN = -omega (mod 2π)`
/// The true anomaly at DN is: `nu_DN = PI - omega (mod 2π)`
///
/// `Time to node = (M_node - M_current).rem_euclid(2π) / n`
/// where n = sqrt(μ/a³) is mean motion
#[inline]
pub fn compute_time_to_nodes(kep: &[f64; 6]) -> Option<(f64, f64)> {
    use std::f64::consts::{PI, TAU};

    let &[a, e, i, _, omega, nu] = kep;

    // Validate inputs
    if a <= 0.0 || !a.is_finite() || !(0.0..1.0).contains(&e) || !e.is_finite() {
        return None;
    }

    // Near-equatorial orbits have ill-defined nodes
    if i.abs() < 0.001 || (PI - i).abs() < 0.001 {
        return None;
    }

    // Mean motion n = sqrt(mu/a^3)
    let n = (MU / (a * a * a)).sqrt();
    if n <= 0.0 || !n.is_finite() {
        return None;
    }

    // True anomaly at ascending node: orbit crosses equator going up
    // At AN, the argument of latitude (omega + nu) = 0 (mod 2π)
    // So nu_AN = -omega (mod 2π)
    let ascending_node_true_anomaly = (-omega).rem_euclid(TAU);

    // True anomaly at descending node: orbit crosses equator going down
    // At DN, the argument of latitude = π
    // So nu_DN = π - omega (mod 2π)
    let descending_node_true_anomaly = (PI - omega).rem_euclid(TAU);

    // Convert current and target true anomalies to mean anomalies
    let mean_anomaly_current = satpy_core::true_to_mean_anomaly_impl(nu, e);
    let mean_anomaly_ascending =
        satpy_core::true_to_mean_anomaly_impl(ascending_node_true_anomaly, e);
    let mean_anomaly_descending =
        satpy_core::true_to_mean_anomaly_impl(descending_node_true_anomaly, e);

    // Handle potential NaN from conversion
    if !mean_anomaly_current.is_finite()
        || !mean_anomaly_ascending.is_finite()
        || !mean_anomaly_descending.is_finite()
    {
        return None;
    }

    // Forward time to each node (always positive, forward in time)
    let time_to_ascending = (mean_anomaly_ascending - mean_anomaly_current).rem_euclid(TAU) / n;
    let time_to_descending = (mean_anomaly_descending - mean_anomaly_current).rem_euclid(TAU) / n;

    Some((time_to_ascending, time_to_descending))
}

/// Compute analytical initial guess for combined Hohmann + plane change transfer.
///
/// Returns `[time2phase_ratio, phase_sma_ratio, wait_ratio]` as optimizer seed.
///
/// # Arguments
/// * `dep_sma` - Departure orbit semi-major axis (km)
/// * `dep_inc` - Departure orbit inclination (radians)
/// * `tgt_sma` - Target orbit semi-major axis (km)
/// * `tgt_inc` - Target orbit inclination (radians)
/// * `phase_angle` - Required phase angle adjustment (radians)
/// * `max_time_s` - Maximum solver horizon (seconds)
///
/// # Physical Defensibility
///
/// Uses Hohmann transfer time estimate as baseline for `time2phase_ratio`,
/// adjusts `phase_sma_ratio` based on plane change magnitude (raise apogee
/// for more efficient plane change at lower velocity), and estimates
/// `wait_ratio` from phase drift requirements.
#[inline]
pub fn combined_transfer_initial_guess(
    dep_sma: f64,
    dep_inc: f64,
    tgt_sma: f64,
    tgt_inc: f64,
    phase_angle: f64,
    max_time_s: f64,
) -> [f64; 3] {
    use std::f64::consts::PI;

    // Hohmann transfer SMA
    let transfer_sma = f64::midpoint(dep_sma, tgt_sma);

    // Transfer time (half Hohmann period)
    let transfer_time = PI * ((transfer_sma * transfer_sma * transfer_sma) / MU).sqrt();
    let time2phase_ratio = (transfer_time / max_time_s).clamp(0.01, 0.3);

    // Optimal phase SMA for plane change at apogee
    let plane_change = (tgt_inc - dep_inc).abs();
    let phase_sma_ratio = if plane_change > 0.01 {
        // Raise apogee for efficient plane change
        1.0 + 0.1 * plane_change.min(0.5)
    } else {
        1.0
    };

    // Phasing time based on phase angle
    let n_dep = (MU / (dep_sma * dep_sma * dep_sma)).sqrt();
    let phase_drift_time = if phase_angle.abs() > 0.01 {
        phase_angle.abs() / n_dep
    } else {
        0.0
    };
    let wait_ratio = (phase_drift_time / max_time_s).clamp(0.01, 0.9);

    [time2phase_ratio, phase_sma_ratio, wait_ratio]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_aware_estimate_coplanar() {
        // Coplanar: plane change is 0, direct estimate should win
        let (est, used_node) = node_aware_estimate(0.5, 0.0);
        assert!(!used_node, "Should use direct estimate for coplanar");
        assert!((est - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_node_aware_estimate_high_inclination() {
        // 60° inclination: plane change ~4 km/s, node-crossing should win
        let hohmann = 0.5;
        let plane = 4.0;
        let (est, used_node) = node_aware_estimate(hohmann, plane);
        assert!(
            used_node,
            "Should use node-crossing estimate for high inclination"
        );
        assert!(est < 1.0, "Estimate should be < 1 km/s, got {est}");
    }

    #[test]
    fn test_node_aware_estimate_infinite_hohmann_returns_plane_bits() {
        // Exact-bits form of the infinite-hohmann branch; the tolerance form
        // and the `used_node` flag are covered by
        // `test_node_aware_estimate_hohmann_infinity`. The infinite-plane case
        // lives in `test_node_aware_estimate_plane_infinity`.
        let (est, _) = node_aware_estimate(f64::INFINITY, 1.0);
        assert_eq!(est.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn test_node_aware_estimate_hohmann_infinity() {
        let (est, used_node) = node_aware_estimate(f64::INFINITY, 4.0);
        assert!(
            !used_node,
            "Should not use node-crossing when hohmann is infinity"
        );
        assert!((est - 4.0).abs() < 0.01, "Should return plane estimate");
    }

    #[test]
    fn test_node_aware_estimate_plane_infinity() {
        let (est, used_node) = node_aware_estimate(0.5, f64::INFINITY);
        assert!(used_node, "Should use node-crossing when plane is infinity");
        assert!((est - 0.6).abs() < 0.01, "Should return hohmann * 1.2");
    }

    #[test]
    fn test_node_aware_estimate_large_plane_vs_small_hohmann() {
        // Real scenario: small altitude change + large plane change
        let (est, used_node) = node_aware_estimate(0.093, 4.19);
        assert!(used_node, "Should use node-crossing for large plane change");
        assert!(est < 0.15, "Estimate should be ~0.11 km/s, got {est}");
    }

    #[test]
    fn test_compute_time_to_nodes_basic() {
        use std::f64::consts::PI;
        // ISS-like orbit: a=6778km, e=0.0005, i=51.6°, RAAN=0, omega=0, nu=0 (at AN)
        let kep = [6778.0, 0.0005, 51.6_f64.to_radians(), 0.0, 0.0, 0.0];

        let result = compute_time_to_nodes(&kep);
        assert!(result.is_some(), "Should return valid times");

        let (ascending_node_time_s, descending_node_time_s) = result.unwrap();
        // At nu=0 and omega=0, we're at the ascending node, so t_an should be ~0 or ~1 period
        // t_dn should be ~half period
        let period = 2.0 * PI * ((6778.0_f64.powi(3) / MU).sqrt());
        let half_period = period / 2.0;

        // t_an should be very small (we're at AN) or ~full period
        assert!(
            ascending_node_time_s < 60.0 || (period - ascending_node_time_s).abs() < 60.0,
            "At AN, t_an should be ~0 or ~period, got {ascending_node_time_s:.1}s (period={period:.1}s)"
        );

        // t_dn should be ~half period
        assert!(
            (descending_node_time_s - half_period).abs() < 120.0,
            "t_dn should be ~half period, got {descending_node_time_s:.1}s (expected ~{half_period:.1}s)"
        );
    }

    #[test]
    fn test_compute_time_to_nodes_equatorial() {
        // Equatorial orbit: i=0, nodes are undefined
        let kep = [7000.0, 0.001, 0.0, 0.0, 0.0, 0.0];
        let result = compute_time_to_nodes(&kep);
        assert!(result.is_none(), "Should return None for equatorial orbit");
    }

    #[test]
    fn test_compute_time_to_nodes_invalid() {
        // Invalid orbit: negative SMA
        let kep_neg_sma = [-1000.0, 0.1, 0.5, 0.0, 0.0, 0.0];
        assert!(
            compute_time_to_nodes(&kep_neg_sma).is_none(),
            "Should return None for negative SMA"
        );

        // Hyperbolic orbit: e >= 1
        let kep_hyp = [10000.0, 1.5, 0.5, 0.0, 0.0, 0.0];
        assert!(
            compute_time_to_nodes(&kep_hyp).is_none(),
            "Should return None for hyperbolic orbit"
        );
    }

    #[test]
    fn test_compute_time_to_nodes_high_inclination() {
        use std::f64::consts::PI;
        // High inclination orbit: i=80°
        let kep = [
            7000.0,
            0.01,
            80.0_f64.to_radians(),
            0.0,
            45.0_f64.to_radians(),
            30.0_f64.to_radians(),
        ];

        let result = compute_time_to_nodes(&kep);
        assert!(result.is_some(), "Should work for high inclination");

        let (ascending_node_time_s, descending_node_time_s) = result.unwrap();
        let period = 2.0 * PI * ((7000.0_f64.powi(3) / MU).sqrt());

        // Both times should be positive and less than one period
        assert!(
            ascending_node_time_s > 0.0 && ascending_node_time_s < period,
            "t_an should be in (0, period)"
        );
        assert!(
            descending_node_time_s > 0.0 && descending_node_time_s < period,
            "t_dn should be in (0, period)"
        );

        // The two nodes should be roughly half an orbit apart
        let node_diff = (ascending_node_time_s - descending_node_time_s).abs();
        let half_period = period / 2.0;
        let tolerance = period * 0.15; // 15% tolerance for eccentric orbits
        assert!(
            (node_diff - half_period).abs() < tolerance
                || (period - node_diff - half_period).abs() < tolerance,
            "Nodes should be ~half orbit apart: diff={node_diff:.1}s, expected~{half_period:.1}s"
        );
    }

    #[test]
    fn test_combined_burn_reduces_to_hohmann_when_coplanar() {
        for (r1, r2) in [(6778.0, 6878.0), (6678.0, 8378.0), (7378.0, 6878.0)] {
            let combined = combined_burn_delta_v(r1, r2, 1.0);
            let hohmann = hohmann_delta_v(r1, r2);
            assert!(
                (combined - hohmann).abs() < 1e-12,
                "coplanar combined {combined} != hohmann {hohmann}"
            );
        }
    }

    #[test]
    fn test_combined_burn_never_exceeds_separate_sum() {
        for deg in [0.0_f64, 5.0, 15.0, 30.0, 60.0, 90.0] {
            let cos_g = deg.to_radians().cos();
            for (r1, r2) in [(6778.0, 6878.0), (6678.0, 8378.0)] {
                let combined = combined_burn_delta_v(r1, r2, cos_g);
                // Separate-sum reference at the slower (outer) circular speed,
                // matching the classic split-maneuver bound.
                let v_slow = (MU / r1.max(r2)).sqrt();
                let sin_half = ((1.0 - cos_g) * 0.5_f64).max(0.0).sqrt();
                let separate = hohmann_delta_v(r1, r2) + 2.0 * v_slow * sin_half;
                assert!(
                    combined <= separate + 1e-12,
                    "deg={deg}: combined {combined} > separate {separate}"
                );
            }
        }
    }

    #[test]
    fn test_combined_burn_monotonic_in_plane_angle() {
        let mut last = 0.0;
        for deg in [0.0_f64, 5.0, 15.0, 30.0, 60.0, 90.0, 150.0] {
            let dv = combined_burn_delta_v(6778.0, 6878.0, deg.to_radians().cos());
            assert!(
                dv >= last - 1e-12,
                "dv must not decrease with plane angle: {deg} deg gave {dv} < {last}"
            );
            last = dv;
        }
    }

    #[test]
    fn test_combined_burn_degenerate_inputs() {
        assert!(combined_burn_delta_v(0.0, 6878.0, 1.0).is_infinite());
        assert!(combined_burn_delta_v(6778.0, f64::NAN, 1.0).is_infinite());
        assert!(combined_burn_delta_v(6778.0, 6878.0, f64::NAN).is_infinite());
    }

    #[test]
    fn test_plane_cos_between_orthogonal_and_aligned() {
        let v = (MU / 6778.0_f64).sqrt();
        let equatorial = [6778.0, 0.0, 0.0, 0.0, v, 0.0];
        let polar = [6778.0, 0.0, 0.0, 0.0, 0.0, v];
        let cos_same = plane_cos_between(&equatorial, &equatorial);
        let cos_orth = plane_cos_between(&equatorial, &polar);
        assert!((cos_same - 1.0).abs() < 1e-12);
        assert!(cos_orth.abs() < 1e-12);
        let degenerate = [0.0; 6];
        assert!(plane_cos_between(&degenerate, &equatorial).is_nan());
    }
}
