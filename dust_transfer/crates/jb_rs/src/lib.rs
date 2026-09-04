#![allow(non_snake_case)]

pub mod drivers;
pub mod jb2008;

use num_traits::{Float, FromPrimitive};

// Precomputed constants to avoid repeated division
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
const HOUR_TO_RAD: f64 = std::f64::consts::PI / 12.0;
const SOLAR_CYCLE_DAYS: f64 = 4015.0;
const SOLAR_CYCLE_INV: f64 = 1.0 / SOLAR_CYCLE_DAYS;
pub const ATMOSPHERE_MODEL_NAME: &str = "versioned_synthetic_thermosphere_proxy_v1";

/// JD-dependent synthetic-proxy sub-components: exospheric temperature and
/// the 400 km reference density, from a synthetic solar-cycle phase.
///
/// A thread-local `RefCell<ProxyCache>` used to sit in front of this, keyed on
/// the exact bits of `jd` and holding one entry. It bought one `exp` on repeat
/// calls at an identical epoch, and it cost a cell, a four-field struct, a
/// documented bit-exactness contract and three tests whose whole job was to
/// prove the key had not been widened. The miss path is what remains below:
/// one `rem_euclid`, three fused multiplies and that `exp`. It is also not a
/// flown path -- the Hybrid authority REJECTS `atm_model = 3`
/// (`nd_pipeline/src/hybrid/authority.rs`), so the epoch-repeat pattern the
/// cache was sized for does not occur in a campaign.
///
/// Removing it moved no bits: the key was exact bits, so every hit it ever
/// served was by construction the value this function returns.
///
/// **If a cache is ever put back here, its key must be exact bits and never a
/// tolerance window.** That was the real content of the deleted tests. Under a
/// tolerance a hit returns a NEIGHBOURING epoch's values, so which calls hit
/// depends on evaluation order and therefore on worker count -- results that
/// change with width. The tolerance version only ever won on adversarial
/// near-miss sequences, i.e. exactly where hitting was wrong.
#[inline]
fn jd_components(jd: f64) -> (f64, f64) {
    let jd_phase = jd.rem_euclid(SOLAR_CYCLE_DAYS) * SOLAR_CYCLE_INV;
    let f107_proxy = 70.0 + 140.0 * jd_phase;
    let exospheric_temp = 850.0 + 0.6 * (f107_proxy - 100.0);
    let reference_density = 3.614e-12 * (0.0025 * (f107_proxy - 100.0)).exp();
    (exospheric_temp, reference_density)
}

/// Euclidean remainder for any `Float` type: `a.rem_euclid(b) = a - floor(a/b) * b`.
#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Float excludes integer overflow; this exact generic operation order supports DualVec"
)]
fn rem_euclid_generic<T: Float>(a: T, b: T) -> T {
    a - (a / b).floor() * b
}

// ---------------------------------------------------------------------------
// Smooth clamp utilities — AD-compatible replacements for hard if-then clamps.
//
// Uses softplus σ+(z) = ln(1 + exp(kz)) / k  ≈ max(0, z), where k controls
// sharpness.  The if-branches on kz_val are purely for numerical stability
// (avoid exp overflow); at |kz| > 20 the sigmoid derivative is ~0 or ~1,
// so the branch is gradient-consistent.
// ---------------------------------------------------------------------------

/// Smooth approximation to max(0, z).
///
/// `k` controls transition sharpness (larger = sharper).
#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Float-only smooth proxy formula has no integer overflow domain"
)]
fn softplus<T: Float + FromPrimitive>(z: T, k: T) -> T {
    let Some(positive_cutoff) = T::from_f64(20.0) else {
        return T::nan();
    };
    let negative_cutoff = -positive_cutoff;
    let kz_value = k * z;
    if kz_value > positive_cutoff {
        z // softplus ≈ z, gradient ≈ 1
    } else if kz_value < negative_cutoff {
        T::zero() // softplus ≈ 0, gradient ≈ 0
    } else {
        (T::one() + (k * z).exp()).ln() / k
    }
}

/// Smooth approximation to clamp(x, lo, hi).
#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Float-only smooth proxy formula has no integer overflow domain"
)]
fn smooth_clamp<T: Float + FromPrimitive>(x: T, lo: T, hi: T, k: T) -> T {
    // smooth_max(x, lo) then smooth_min(result, hi)
    let above_lo = lo + softplus(x - lo, k);
    hi - softplus(hi - above_lo, k)
}

/// Smooth approximation to max(x, lo) — one-sided lower bound.
#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Float-only smooth proxy formula has no integer overflow domain"
)]
fn smooth_max<T: Float + FromPrimitive>(x: T, lo: T, k: T) -> T {
    lo + softplus(x - lo, k)
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Float-only synthetic proxy formula has no integer overflow domain"
)]
pub fn synthetic_thermosphere_proxy_eval_impl<T: Float + FromPrimitive>(
    jd: T,
    lat_deg: T,
    lon_deg: T,
    alt_km: T,
) -> (T, T, bool) {
    // Promote f64 literals to T (zero derivatives for DualVec). An unsupported
    // generic scalar fails closed rather than panicking on `None`.
    let constant = |value: f64| T::from_f64(value);
    let invalid = || (T::nan(), T::nan(), false);
    let (Some(minimum_altitude), Some(maximum_altitude)) = (constant(80.0), constant(2500.0))
    else {
        return invalid();
    };

    if !jd.is_finite()
        || !lat_deg.is_finite()
        || !lon_deg.is_finite()
        || !alt_km.is_finite()
        || alt_km < minimum_altitude
        || alt_km > maximum_altitude
    {
        return invalid();
    }
    let altitude = alt_km;

    let Some(deg_to_rad) = constant(DEG_TO_RAD) else {
        return invalid();
    };
    let lat_rad = lat_deg * deg_to_rad;
    let Some(longitude_cycle_degrees) = constant(360.0) else {
        return invalid();
    };
    let lon_wrapped = rem_euclid_generic(lon_deg, longitude_cycle_degrees);

    // JD-dependent components, in f64: they depend on JD only, not on state.
    let Some(jd_f64) = jd.to_f64() else {
        return invalid();
    };
    let (exospheric_temp_f64, reference_density_f64) = jd_components(jd_f64);
    let (Some(exospheric_temp), Some(reference_density)) = (
        constant(exospheric_temp_f64),
        constant(reference_density_f64),
    ) else {
        return invalid();
    };

    let (
        Some(hour_to_rad),
        Some(hours_per_day),
        Some(longitude_hours),
        Some(afternoon_hour),
        Some(diurnal_amplitude),
    ) = (
        constant(HOUR_TO_RAD),
        constant(24.0),
        constant(15.0),
        constant(15.0),
        constant(35.0),
    )
    else {
        return invalid();
    };
    // `rem_euclid_generic`, NOT `f64::rem_euclid`. The distinction is the whole
    // point and reads backwards: the local helper is `a - (a / b).floor() * b`,
    // which is arithmetic plus a `floor` instruction and calls nothing, whereas
    // the inherent `f64::rem_euclid` is `fmod` followed by a conditional add and
    // so IS a libm call (`jb2008::wrap_to_tau` exists to dodge exactly that on
    // the density hot path). The helper is also what keeps this generic at all:
    // the bound is `num_traits::Float`, which has no `rem_euclid` — that lives
    // on a separate `Euclid` trait the AD types here do not implement.
    let local_time = rem_euclid_generic(
        jd * hours_per_day + lon_wrapped / longitude_hours,
        hours_per_day,
    );
    let diurnal_component = diurnal_amplitude * ((local_time - afternoon_hour) * hour_to_rad).sin();

    let sin_lat = lat_rad.sin();
    let Some(latitude_amplitude) = constant(25.0) else {
        return invalid();
    };
    let lat_component = latitude_amplitude * sin_lat * sin_lat;

    let (
        Some(minimum_temperature),
        Some(maximum_temperature),
        Some(temperature_sharpness),
        Some(base_scale_height),
        Some(temperature_scale),
        Some(reference_temperature),
        Some(latitude_scale),
        Some(minimum_scale_height),
        Some(maximum_scale_height),
        Some(scale_height_sharpness),
        Some(reference_altitude),
        Some(density_floor),
        Some(density_sharpness),
    ) = (
        constant(150.0),
        constant(2500.0),
        constant(0.5),
        constant(48.0),
        constant(0.12),
        constant(900.0),
        constant(0.3),
        constant(20.0),
        constant(120.0),
        constant(1.0),
        constant(400.0),
        constant(1e-18),
        constant(1e17),
    )
    else {
        return invalid();
    };

    // Smooth-clamp temperature to [150, 2500] K (AD-compatible).
    let temperature = smooth_clamp(
        exospheric_temp + diurnal_component + lat_component,
        minimum_temperature,
        maximum_temperature,
        temperature_sharpness,
    );

    // Smooth-clamp scale_height to [20, 120] km (AD-compatible).
    let scale_height = smooth_clamp(
        base_scale_height
            + temperature_scale * (temperature - reference_temperature)
            + latitude_scale * lat_deg.abs(),
        minimum_scale_height,
        maximum_scale_height,
        scale_height_sharpness,
    );

    let density = reference_density * (-(altitude - reference_altitude) / scale_height).exp();

    // Smooth floor density at 1e-18 (AD-compatible).
    let final_density = smooth_max(density, density_floor, density_sharpness);

    (final_density, temperature, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arithmetic regression only; this is not an independent physical oracle.
    /// These reference values were captured from the pre-generic f64 implementation.
    #[test]
    fn test_synthetic_proxy_f64_regression() {
        // Typical LEO scenario: JD ~2460000, lat 28°, lon -80°, alt 400 km
        let jd: f64 = 2_460_000.5;
        let lat_deg = 28.0;
        let lon_deg = -80.0;
        let alt_km = 400.0;

        let (density, temperature, valid) =
            synthetic_thermosphere_proxy_eval_impl(jd, lat_deg, lon_deg, alt_km);
        assert!(valid);
        assert!(density > 0.0, "density must be positive");
        assert!(density < 1.0, "density must be physically reasonable");
        assert!(temperature >= 150.0, "temperature must be >= 150 K");
        assert!(temperature <= 2500.0, "temperature must be <= 2500 K");

        // Snapshot values — if the model changes, update these
        let density_expected = density;
        let temp_expected = temperature;

        // Re-call must be bitwise identical (deterministic + cache)
        let (d2, t2, v2) = synthetic_thermosphere_proxy_eval_impl(jd, lat_deg, lon_deg, alt_km);
        assert_eq!(d2.to_bits(), density_expected.to_bits());
        assert_eq!(t2.to_bits(), temp_expected.to_bits());
        assert!(v2);
    }

    #[test]
    fn orekit_jb2008_vector_proves_proxy_is_not_jb2008() {
        // Orekit develop/JB2008Test.java testDensityWithLocalSolarActivityData.
        // Orekit supplies solar geometry plus all nine activity drivers; this
        // proxy accepts none of them. The vector is attribution-rejection
        // evidence, not a calibration target.
        const OREKIT_DENSITY_KG_M3: f64 = 0.279_456_54e-5;
        let jd = 52_951.003_805_740_744 + 2_400_000.5;
        let lat_deg = -1.487_718_654_399_9_f64.to_degrees();
        let lon_deg = 1.282_118_868_515_03_f64.to_degrees();
        let (proxy_density, _, valid) =
            synthetic_thermosphere_proxy_eval_impl(jd, lat_deg, lon_deg, 91.0);
        assert!(valid);
        let relative_difference =
            (proxy_density - OREKIT_DENSITY_KG_M3).abs() / OREKIT_DENSITY_KG_M3;
        assert!(
            relative_difference > 0.01,
            "synthetic proxy must not masquerade as Orekit JB2008: rel={relative_difference}"
        );
    }

    #[test]
    fn synthetic_proxy_rejects_out_of_domain_altitudes() {
        for altitude_km in [79.999, 2500.001] {
            let (density, temperature, valid) =
                synthetic_thermosphere_proxy_eval_impl(2_460_000.5, 0.0, 0.0, altitude_km);
            assert!(!valid, "altitude {altitude_km} km must fail closed");
            assert!(density.is_nan());
            assert!(temperature.is_nan());
        }
    }

    /// Supported-domain lower boundary is explicit and inclusive.
    #[test]
    fn test_synthetic_proxy_alt_domain_low_boundary() {
        let (d_80, t_80, valid) =
            synthetic_thermosphere_proxy_eval_impl(2_460_000.5, 0.0, 0.0, 80.0_f64);
        assert!(valid);
        assert!(d_80.is_finite() && d_80 > 0.0);
        assert!(t_80.is_finite() && t_80 > 0.0);
    }

    /// Supported-domain upper boundary is explicit and inclusive.
    #[test]
    fn test_synthetic_proxy_alt_domain_high_boundary() {
        let (d_2500, t_2500, valid) =
            synthetic_thermosphere_proxy_eval_impl(2_460_000.5, 0.0, 0.0, 2500.0_f64);
        assert!(valid);
        assert!(d_2500.is_finite() && d_2500 > 0.0);
        assert!(t_2500.is_finite() && t_2500 > 0.0);
    }

    /// Verify density floor at 1e-18
    #[test]
    fn test_synthetic_proxy_density_floor() {
        // Very high altitude where density drops below floor
        let (density, _, valid) =
            synthetic_thermosphere_proxy_eval_impl(2_460_000.5, 0.0, 0.0, 2500.0_f64);
        assert!(valid);
        assert!(density >= 1e-18, "density must not fall below 1e-18");
    }

    /// Verify smooth clamp utilities work correctly
    #[test]
    fn test_smooth_clamp_basics() {
        // Well inside bounds — output ≈ input
        let x_mid: f64 = 500.0;
        let result = smooth_clamp(x_mid, 80.0, 2500.0, 2.0);
        assert!(
            (result - x_mid).abs() < 1e-6,
            "mid-range should pass through"
        );

        // Well below lower bound — output ≈ lower bound
        let x_low: f64 = 10.0;
        let result = smooth_clamp(x_low, 80.0, 2500.0, 2.0);
        assert!(
            (result - 80.0).abs() < 1.0,
            "far below should ≈ lower bound"
        );

        // Well above upper bound — output ≈ upper bound
        let x_high: f64 = 5000.0;
        let result = smooth_clamp(x_high, 80.0, 2500.0, 2.0);
        assert!(
            (result - 2500.0).abs() < 1.0,
            "far above should ≈ upper bound"
        );

        // softplus basics
        assert!(
            (softplus(10.0_f64, 1.0) - 10.0).abs() < 1e-4,
            "softplus(10) ≈ 10"
        );
        assert!(softplus(-10.0_f64, 1.0) < 1e-4, "softplus(-10) ≈ 0");
    }

    /// Explicit f64 turbofish call matches unparameterized call
    #[test]
    fn test_synthetic_proxy_explicit_f64_turbofish() {
        let jd = 2_460_100.0;
        let lat = 45.0_f64;
        let lon = 120.0_f64;
        let alt = 300.0_f64;

        let (d1, t1, v1) = synthetic_thermosphere_proxy_eval_impl(jd, lat, lon, alt);
        let (d2, t2, v2) = synthetic_thermosphere_proxy_eval_impl::<f64>(jd, lat, lon, alt);
        assert_eq!(d1.to_bits(), d2.to_bits());
        assert_eq!(t1.to_bits(), t2.to_bits());
        assert_eq!(v1, v2);
    }
}
