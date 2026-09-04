//! Right-hand side (RHS) for the Lightyear delta-state ODE
//!
//! Computes `d(delta_state)/dt = [delta_v; accel_pert + accel_kep_correction]`.

use crate::eclipse::{EclipseError, EclipseSide};
use crate::precomputed_ephem::{AllPrecomputedEphemeris, Body as EphemerisBody};
use crate::strict_hf_enclosure::{
    issue_for_rhs, validate_arc_coverage, StrictHfEnclosureAuthority,
};
use crate::types::{BodyInvariants, ForceConfig, ForceFlags, MU};
use anyhow::Context;
use satpy_core::frame_time::authority::{frame_authority, FrameRotation, FrameSegment};
use satpy_core::SEC_PER_DAY;
use satpy_core::{
    equinoc2eci_impl, EquinoctialBaseline, GravityCache, GravityError, PackedGravityCoeffs,
};
use std::cell::{Cell, UnsafeCell};

// NOTE: Coefficient vectors are shared via Arc to avoid large clones.
use jb_rs::{
    drivers::{Jb2008Drivers, UtcJulianDay},
    jb2008::{
        jb2008_density, jb2008_density_fitted_v7, jb2008_density_logquad_x4_approx_v1,
        jb2008_density_logquad_x4_approx_v2, Jb2008Input,
    },
    synthetic_thermosphere_proxy_eval_impl,
};

use wide::f64x4;

pub(crate) fn effective_scalar_srp(config: &ForceConfig) -> bool {
    (config.force_flags & ForceFlags::SRP) != 0
        && config.cr > 0.0
        && config.am_ratio > 0.0
        && config.p_sun > 0.0
}

// These are shared with `rhs_dual.rs`, which differentiates the same dynamics.
// They live in one module so the scalar and dual paths cannot drift apart --
// see `physical_constants.rs` for why that drift is silent when it happens.
#[cfg(test)]
use crate::physical_constants::LORENTZ_DIPOLE_THETA_RAD;
use crate::physical_constants::{
    AU_KM, BOLTZMANN_K, EARTH_DIPOLE_STRENGTH, ELEMENTARY_CHARGE, INV_LIGHT_SPEED_SQ, KM_TO_M,
    LORENTZ_THETA_COS, LORENTZ_THETA_SIN, MEAN_ION_MASS_KG, MIN_COULOMB_LOG, MIN_NUMBER_DENSITY,
    M_TO_KM, VACUUM_PERMITTIVITY,
};

/// WGS84 flattening. Paired with `ForceConfig::earth_radius` as the semi-major
/// axis, NOT with `GRAVITY_REFERENCE_RADIUS_KM` — the latter is the DIR-R6
/// gravity reference and is 54 cm SMALLER on purpose (6378.13646 km against
/// 6378.137 km, a 0.00054 km difference; see `types.rs`). This said "larger"
/// until 2026-08-04, which inverted the sign of the only quantity it states.
///
/// Promoted to `satpy_core` 2026-07-25 on exactly the condition this comment set
/// — the ellipsoidal ground guard in `events.rs` became the second consumer.
use satpy_core::WGS84_FLATTENING;

/// Height above the WGS84 reference ellipsoid, in km, from an Earth-centred
/// position in km.
///
/// Bowring's closed form (Bowring 1976, "Transformation from spatial to
/// geographical coordinates", Survey Review 23:323-327). Single pass, no
/// iteration: measured worst error 1.513e-9 m against the sealed Orekit
/// geodetic column over the 15 fixture cases spanning 200-1500 km. The height
/// form used here, `h = p*cos(lat) + z*sin(lat) - N*(1 - e^2 sin^2 lat)`, is an
/// exact identity and better conditioned than the textbook `p/cos(lat) - N`,
/// which divides by `cos(lat)` and degrades toward the pole.
///
/// # THE CALLER MUST PASS AN EARTH-FIXED (ITRS) POSITION
///
/// An earlier revision of this comment claimed the reduction was "frame-agnostic
/// by construction" because it consumes only `sqrt(x^2+y^2)` and `z`. **The
/// premise is true and the conclusion is false.** Those two quantities are
/// invariant under a rotation about z, but GCRS->ITRS is
/// `RPOM * R3(ERA) * RC2I` and only the `R3(ERA)` factor is such a rotation.
/// `RC2I` tilts the pole by ~2e-3 rad by 2022, so `z` is NOT invariant, and the
/// WGS84 ellipsoid is Earth-fixed by definition — reducing a GCRS position
/// flattens about the wrong axis.
///
/// MEASURED, instrumenting production at 60 deg geocentric latitude:
/// `h_gcrs 416.057038` vs `h_itrs 416.097133` km at 400 km, and
/// `216.057804` vs `216.097895` at 200 km — **40.09 m, and
/// ALTITUDE-INDEPENDENT**, because it is a latitude shift rather than a radial
/// one (`z_gcrs 5870.039` vs `z_itrs 5877.389`, a 7.35 km z-difference from
/// ~1.25e-3 rad of frame tilt). About 0.07% of density through a 60 km scale
/// height at 400 km — that is `40.09 m / 60 km`, recomputed here because this
/// line read "0.09%" until 2026-08-04, a figure carried over from the ~50 m
/// spherical-altitude case at the `jb2008_density_at_state` call site where it
/// IS right — and the same 40 m buys more at lower altitude as the scale
/// height shrinks.
///
/// Latitude and altitude both require ITRS in principle. Longitude is the one
/// that ADDITIONALLY requires ERA, and JB2008 needs no longitude — it takes a
/// pair of right ascensions and uses only their difference, which is the
/// satellite's hour angle relative to the Sun. That difference is unchanged by
/// the whole-turn offset between the equinox and the prime meridian.
///
/// # ALL FIVE GEOMETRIC ARGUMENTS ARE ITRS-REDUCED, AND THAT IS ONE DECISION
///
/// `jb2008_density_at_state` reduces the altitude, `sat_geocentric_lat_rad`,
/// `sat_ra_rad`, `sun_declination_rad` and `sun_ra_rad` from ONE Earth-fixed
/// rotation. The four angular arguments **must not be moved one at a time.**
/// Measured through the production RHS at the sealed 2022-08-12 epoch
/// (`tests/jb2008_angular_frame_consistency.rs` re-measures all of it):
///
/// | change | 200 km | 400 km | 800 km |
/// | --- | --- | --- | --- |
/// | latitude alone -> ITRS, equatorial | +0.0076% | +0.0245% | +0.0150% |
/// | latitude alone -> ITRS, 80 deg | +0.0466% | +0.1388% | +0.1246% |
/// | **all four -> ITRS, equatorial** | **+0.0027%** | **+0.0087%** | **+0.0048%** |
/// | **all four -> ITRS, 80 deg** | **+0.0113%** | **+0.0399%** | **+0.0410%** |
///
/// The RA shift partially CANCELS the latitude shift, so moving latitude alone
/// lands 2.8-4.1x past the consistent all-ITRS answer — further from it than
/// leaving latitude in GCRS. All four together or none.
///
/// The sealed Orekit fixture's `jb_primitive_inputs` names all four columns
/// after body-frame reductions (`sun_longitude_rad_as_sunRA`,
/// `sun_geodetic_latitude_rad_as_sunDecli`,
/// `satellite_geodetic_longitude_rad_as_satLon`,
/// `satellite_geodetic_latitude_rad_as_satLat`), which is why all-ITRS is the
/// target. Note what that fixture canNOT show: its declared
/// `rotation_convention` is a pure z-rotation, under which both conventions give
/// identical numbers. It is evidence about the intended FRAME and none at all
/// about the magnitude.
///
/// Whether the latitude is geocentric or geodetic is a SEPARATE question about
/// the reference surface, settled the other way — Bowman, not Orekit. See
/// `sat_geocentric_lat_rad` at the call site.
/// `1 - f`, the polar-to-equatorial axis ratio.
const ONE_MINUS_FLATTENING: f64 = 1.0 - WGS84_FLATTENING;
/// First eccentricity squared, `e^2 = f (2 - f)`. Depends only on the flattening.
const E_SQ: f64 = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING);
/// Second eccentricity squared, `e'^2 = (a^2 - b^2) / b^2`.
///
/// Independent of `a`, which is not obvious from that form: substituting
/// `b = a (1 - f)` cancels it, leaving `e'^2 = f (2 - f) / (1 - f)^2 = e^2 / (1 - f)^2`.
/// So this is a constant, not a per-call quantity, whatever ellipsoid radius the
/// caller passes.
const EP_SQ: f64 = E_SQ / (ONE_MINUS_FLATTENING * ONE_MINUS_FLATTENING);

/// `hypot(a, b)` for operands whose squares provably cannot overflow or
/// underflow.
///
/// `f64::hypot` is a libm call that pays for exponent scaling so that
/// `a*a + b*b` is safe for any finite input. Every operand this function sees
/// here is a geometric quantity in kilometres, or a product of two of them:
/// bounded below by the ellipsoid's semi-minor axis and above by a few times
/// Earth radius times the semi-major axis, i.e. everything lives inside
/// `[1e-3, 1e9]`. Squares of those land in `[1e-6, 1e18]`, against a binary64
/// range that does not overflow until `~1.8e308` and does not lose precision to
/// subnormals until `~2.2e-308`. The scaling is dead code at these magnitudes.
///
/// `geodetic_altitude_km_operands_stay_in_the_squarable_range` pins that
/// premise against the production altitude band, so a caller that starts
/// passing metres, or an ellipsoid in different units, fails there rather than
/// silently losing the guarantee.
#[inline]
fn hypot_unscaled(a: f64, b: f64) -> f64 {
    a.mul_add(a, b * b).sqrt()
}

fn geodetic_altitude_km(x_km: f64, y_km: f64, z_km: f64, semi_major_km: f64) -> f64 {
    let equatorial_radius_km = hypot_unscaled(x_km, y_km);
    if equatorial_radius_km == 0.0 {
        // On the spin axis the ellipsoid surface is the semi-minor axis.
        let polar_altitude_km = z_km.abs() - semi_major_km * ONE_MINUS_FLATTENING;
        return polar_altitude_km;
    }
    let semi_minor_km = semi_major_km * ONE_MINUS_FLATTENING;
    // Bowring's parametric-latitude seed, then one exact reduction.
    //
    // NEITHER ANGLE IS EVER USED -- only its sine and cosine -- so the `atan2`
    // and the `sin_cos` that immediately undoes it are both removable. For any
    // `(n, d)`, `(n, d) / |(n, d)|` IS the unit vector at angle `atan2(n, d)`,
    // so `sin = n / hypot(n, d)` and `cos = d / hypot(n, d)`, exactly, in all
    // four quadrants. That turns four libm transcendental calls per RHS
    // evaluation into one `hypot` and two divides each.
    //
    // `hypot(numerator, denominator) > 0` is guaranteed: the denominator is
    // `equatorial_radius_km * semi_minor_km`, and the early return above means
    // the equatorial radius is positive, so the two arguments cannot both be zero.
    let (theta_numerator, theta_denominator) =
        (z_km * semi_major_km, equatorial_radius_km * semi_minor_km);
    let theta_radius = hypot_unscaled(theta_numerator, theta_denominator);
    let (sin_theta, cos_theta) = (
        theta_numerator / theta_radius,
        theta_denominator / theta_radius,
    );
    let latitude_numerator = z_km + EP_SQ * semi_minor_km * sin_theta * sin_theta * sin_theta;
    let latitude_denominator =
        equatorial_radius_km - E_SQ * semi_major_km * cos_theta * cos_theta * cos_theta;
    let latitude_radius = hypot_unscaled(latitude_numerator, latitude_denominator);
    let (sin_lat, cos_lat) = (
        latitude_numerator / latitude_radius,
        latitude_denominator / latitude_radius,
    );
    // `1 - e^2 sin^2(lat)` appears twice; it was computed twice.
    let one_minus_e_sin_sq = 1.0_f64 - E_SQ * sin_lat * sin_lat;
    let prime_vertical_radius_km = semi_major_km / one_minus_e_sin_sq.sqrt();
    equatorial_radius_km * cos_lat + z_km * sin_lat - prime_vertical_radius_km * one_minus_e_sin_sq
}

#[cfg(test)]
mod geodetic_hypot_range_tests {
    use super::{geodetic_altitude_km, hypot_unscaled, ONE_MINUS_FLATTENING};

    /// The premise that licenses [`hypot_unscaled`]: every operand it is handed
    /// from [`geodetic_altitude_km`] has a square that is nowhere near the
    /// binary64 limits, so `hypot`'s exponent scaling was never doing anything.
    ///
    /// This walks the same expression tree the function does, over the whole
    /// production altitude band and a full sweep of latitude and longitude, and
    /// asserts on the SQUARES rather than trusting the prose above. If a caller
    /// ever starts passing metres, or an ellipsoid in other units, the squares
    /// move by six orders per unit change and this fails long before any
    /// overflow could.
    #[test]
    fn geodetic_altitude_km_operands_stay_in_the_squarable_range() {
        // THE SAFETY FLOOR, and it is the one that actually licenses the swap:
        // `a*a + b*b` differs from `hypot` only when the sum overflows or sinks
        // into the subnormals. Subnormals begin at ~2.2e-308; anything above
        // this bound retains full precision, so the scaling has nothing to do.
        //
        // This is deliberately NOT a "sums are around 1e18" assertion. The first
        // version of this test asserted a 1e-12 floor and FAILED, correctly, at
        // the poles: `cos(±π/2)` is ~6.1e-17 rather than 0, so the equatorial
        // radius there is ~4e-13 and its square ~1.6e-25. That is a real
        // production geometry, it is 283 orders clear of the subnormals, and the
        // squared form is exact for it. Note `geodetic_altitude_km`'s
        // `equatorial_radius_km == 0.0` early return does NOT fire at 4e-13 --
        // the polar branch is reached by exact zero only, which is pre-existing
        // behaviour that this change does not touch.
        const SQUARE_FLOOR: f64 = 1e-280;
        // The ceiling is a unit tripwire rather than a safety one: binary64 does
        // not overflow until ~1.8e308, but re-expressing these lengths in metres
        // would multiply every square by 1e6 and push the `z * semi_major`
        // product term from ~2e15 to ~2e27, tripping this.
        const SQUARE_MAX: f64 = 1e24;
        let semi_major_km = 6378.137;
        let semi_minor_km = semi_major_km * ONE_MINUS_FLATTENING;

        let mut checked = 0_u32;
        for altitude_step in 0..=40 {
            // 150 km (below any flown perigee) to 2150 km (above the ceiling).
            let radius_km = semi_major_km + 150.0 + f64::from(altitude_step) * 50.0;
            for lat_step in 0..=36 {
                let latitude = f64::from(lat_step - 18) * (std::f64::consts::PI / 36.0);
                for lon_step in 0..36 {
                    let longitude = f64::from(lon_step) * (std::f64::consts::TAU / 36.0);
                    let x_km = radius_km * latitude.cos() * longitude.cos();
                    let y_km = radius_km * latitude.cos() * longitude.sin();
                    let z_km = radius_km * latitude.sin();

                    let equatorial_radius_km = hypot_unscaled(x_km, y_km);
                    // The three operand pairs, in the order the function forms
                    // them.
                    let pairs = [
                        (x_km, y_km),
                        (z_km * semi_major_km, equatorial_radius_km * semi_minor_km),
                        (z_km, equatorial_radius_km),
                    ];
                    for (a, b) in pairs {
                        let sum_of_squares = a.mul_add(a, b * b);
                        assert!(
                            sum_of_squares.is_finite(),
                            "a={a:e} b={b:e} squared to a non-finite sum"
                        );
                        assert!(
                            sum_of_squares == 0.0 || sum_of_squares >= SQUARE_FLOOR,
                            "a={a:e} b={b:e} gives a^2+b^2={sum_of_squares:e}, below the \
                             {SQUARE_FLOOR:e} floor -- it is close enough to the subnormals \
                             that hypot's exponent scaling is no longer dead code, so \
                             `hypot_unscaled` is no longer licensed here"
                        );
                        assert!(
                            sum_of_squares <= SQUARE_MAX,
                            "a={a:e} b={b:e} gives a^2+b^2={sum_of_squares:e}, above \
                             {SQUARE_MAX:e} -- these operands are no longer kilometre-scale, \
                             so re-derive the range premise before trusting `hypot_unscaled`"
                        );
                    }
                    // And the function itself stays sane over the same sweep.
                    let altitude = geodetic_altitude_km(x_km, y_km, z_km, semi_major_km);
                    assert!(
                        altitude.is_finite() && (100.0..2300.0).contains(&altitude),
                        "altitude {altitude} km is outside the swept band"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 50_000, "sweep collapsed to {checked} samples");
    }
}

#[cfg(test)]
mod dipole_constant_tests {
    use super::{LORENTZ_DIPOLE_THETA_RAD, LORENTZ_THETA_COS, LORENTZ_THETA_SIN};

    /// The precomputed pair must still be the sine and cosine of the angle the
    /// source comment claims. Without this the angle is a comment, and comments
    /// do not fail.
    #[test]
    fn lorentz_dipole_sin_cos_match_their_stated_angle() {
        let (sin_theta, cos_theta) = LORENTZ_DIPOLE_THETA_RAD.sin_cos();
        assert!(
            (LORENTZ_THETA_SIN - sin_theta).abs() <= 1e-15,
            "LORENTZ_THETA_SIN={LORENTZ_THETA_SIN:.17e} but sin({LORENTZ_DIPOLE_THETA_RAD:.17e})={sin_theta:.17e}"
        );
        assert!(
            (LORENTZ_THETA_COS - cos_theta).abs() <= 1e-15,
            "LORENTZ_THETA_COS={LORENTZ_THETA_COS:.17e} but cos({LORENTZ_DIPOLE_THETA_RAD:.17e})={cos_theta:.17e}"
        );
    }
}

// `LIGHTYEAR_PACKED_GRAVITY` used to live here: a `Lazy<bool>` hardcoded `true`,
// then briefly a `const bool`. Neither removed the code it was supposed to gate.
// The unpacked arms are gone now because they were deleted, not because a flag
// folded — see the note on `GravityEvalMode` for why the fold could never work.

/// How the derivative evaluates spherical-harmonic gravity.
///
/// There used to be three more variants, `Unpacked*`, selected by a
/// `LIGHTYEAR_PACKED_GRAVITY` flag that was hardcoded `true`. Making that flag a
/// `const` was NOT enough to remove them, and the reason is worth recording
/// because it will recur: the flag is folded in the CONSTRUCTOR, which writes
/// the result into this field, while the derivative matches on
/// `self.gravity_mode` — a field LOAD in a different function. Killing the arms
/// would need interprocedural value-range propagation through a struct field,
/// which neither fat LTO nor `codegen-units=1` performs. Measured: the const
/// recovered 3,840 bytes of `__text` (the `once_cell` guard) and left 5,628
/// bytes of dense-gravity code sitting in the derivative.
///
/// Deleting the variants is what actually removes them. Raw-versus-packed
/// differential validation remains inside `satpy_core`; scalar RHS owns only
/// validated packed authority and never reaches the raw evaluator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GravityEvalMode {
    Packed,
    AnalyticCentral,
    ExplicitLowOrder,
}

#[derive(Clone)]
struct ThirdBodySimdPack {
    body_norm_x: f64x4,
    body_norm_y: f64x4,
    body_norm_z: f64x4,
    inv_body_dist: f64x4,
    mu_coef: f64x4,
    mask: f64x4,
    active: bool,
    all_active: bool,
}

impl ThirdBodySimdPack {
    #[inline]
    const fn inactive() -> Self {
        Self {
            body_norm_x: f64x4::ZERO,
            body_norm_y: f64x4::ZERO,
            body_norm_z: f64x4::ZERO,
            inv_body_dist: f64x4::ZERO,
            mu_coef: f64x4::ZERO,
            mask: f64x4::ZERO,
            active: false,
            all_active: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compute_drag, compute_lorentz_frame, compute_relativity, compute_srp_with_precomputed,
        validate_atmosphere_model_code, EclipseSide, ForceConfig, ForceFlags, FrameRotation,
        GravityEvalMode, LightyearRHS, RHSCache, AU_KM,
    };
    use crate::config::packed_constants_from_bytes;
    use satpy_core::{pack_gravity_coeffs, GravityError, PackedGravityCoeffs};
    use serde_json::Value;
    use std::sync::Arc;

    #[test]
    fn rejected_a2_and_rhs_context_api_are_absent() {
        // Absence gate over the WHOLE surface, not a hand-picked file list:
        // an earlier version of this test include_str!'d 10 named files,
        // leaving 13 lightyear src modules unscanned — reintroducing a
        // retired token in any of them passed green. The scan now walks
        // every .rs file under this crate's src/examples/tests directories
        // at runtime and prints/pins its set size so a collapsed scan is
        // loud (a clean scan needs its set size checked).
        //
        // It used to walk `crates/odesolve_lightyear` as a second root. That
        // crate is now `src/odesolve`, so the one root covers both and there
        // is no second tree to forget.
        fn collect_rs_sources(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
            let entries = std::fs::read_dir(dir)
                .unwrap_or_else(|error| panic!("absence scan cannot read {dir:?}: {error}"));
            for entry in entries {
                let path = entry.expect("absence scan dir entry").path();
                if path.is_dir() {
                    collect_rs_sources(&path, out);
                } else if path.extension().is_some_and(|extension| extension == "rs") {
                    let source = std::fs::read_to_string(&path)
                        .unwrap_or_else(|error| panic!("absence scan read {path:?}: {error}"));
                    out.push((path.display().to_string(), source));
                }
            }
        }

        let lightyear_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut sources: Vec<(String, String)> = Vec::new();
        for sub in ["src", "examples", "tests"] {
            let dir = lightyear_root.join(sub);
            if dir.is_dir() {
                collect_rs_sources(&dir, &mut sources);
            }
        }
        assert!(
            sources.len() >= 35,
            "absence scan set collapsed to {} files; expected the full \
             lightyear + odesolve source tree",
            sources.len()
        );
        for retired in [
            concat!("RhsEval", "Context"),
            concat!("RhsStage", "Use"),
            concat!("Stage8", "DensityAnchor"),
            concat!("stage8_", "density"),
            concat!("stage9_", "density"),
            concat!("Stage9", "Density"),
            concat!("STAGE9_", "DENSITY"),
            concat!("exact_stage9_", "density"),
            concat!("EXACT_STAGE9_", "DENSITY"),
            concat!("PROP_RHS_", "STAGE9"),
            concat!("rhs-stage-", "census"),
        ] {
            for (path, source) in &sources {
                assert!(
                    !source.contains(retired),
                    "rejected A2/context surface remains in {path}: {retired}"
                );
            }
        }
        // The absorbed solver has no feature arms, so every build compiles the
        // same one. This used to be checked as the absence of a `[features]`
        // table in the manifest of the standalone crate the solver came from.
        // That manifest was deleted when the crate was absorbed, and the
        // solver now lives inside a crate that DOES have features, so the
        // property is asserted where it actually holds — on the source. No
        // path is named above on purpose: the address would resolve nowhere.
        let mut odesolve_files = 0usize;
        for (path, source) in &sources {
            if !path.contains("/odesolve") {
                continue;
            }
            odesolve_files = odesolve_files.saturating_add(1);
            assert!(
                !source.contains(concat!("feature", " = \"")),
                "the absorbed ODE solver must stay feature-free: {path}"
            );
        }
        assert!(
            odesolve_files >= 15,
            "odesolve subtree scan collapsed to {odesolve_files} files"
        );
    }

    const FIXTURE: &str = include_str!("../tests/data/orekit_dir_r6_5x5_v1.json");
    const COEFFICIENTS: &[u8] =
        include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

    #[test]
    fn owned_atmosphere_validation_uses_anyhow_result() {
        fn assert_anyhow_result(_: anyhow::Result<()>) {}

        assert_anyhow_result(validate_atmosphere_model_code(i32::MIN));
    }

    // These tests deliberately parse the sealed coefficients LOCALLY
    // (`packed_constants_from_bytes`) and never publish to `GLOBAL_COEFFS`:
    // this lib-test binary also runs the `session.rs` and `batch.rs` test
    // installers, which publish DIFFERENT (synthetic) packs concurrently. A
    // global install-then-read here can observe another test's pack — the
    // fixture comparison below would then be red against synthetic gravity.
    // The publish path itself is covered by the child-process-isolated tests
    // in `config.rs`. Do not reintroduce `load_constants_from_bytes` here.

    fn hex_f64(value: &Value, field: &str) -> f64 {
        let encoded = value
            .as_str()
            .unwrap_or_else(|| panic!("{field} must be a binary64 hex string"));
        let bits = u64::from_str_radix(
            encoded
                .strip_prefix("0x")
                .unwrap_or_else(|| panic!("{field} must start with 0x")),
            16,
        )
        .unwrap_or_else(|error| panic!("{field} must contain 16 hex digits: {error}"));
        f64::from_bits(bits)
    }

    fn hex_vec3(case: &Value, field: &str) -> [f64; 3] {
        let values = case
            .get(field)
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("{field} must be an array"));
        let [first, second, third] = values.as_slice() else {
            panic!("{field} must contain three components");
        };
        [
            hex_f64(first, field),
            hex_f64(second, field),
            hex_f64(third, field),
        ]
    }

    fn assert_close(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64, label: &str) {
        let bound = abs_tol + rel_tol * expected.abs();
        let delta = (actual - expected).abs();
        assert!(
            delta <= bound,
            "{label}: actual={actual:.17e} expected={expected:.17e} delta={delta:.17e} bound={bound:.17e}"
        );
    }

    /// Identity GCRS->ITRS rotation, for pinning the production gravity path
    /// against a fixture whose positions are already Earth-fixed.
    ///
    /// This test used to call a private `accumulate_spherical_gravity_mode`
    /// sibling with `(sin_gmst, cos_gmst) = (0.0, 1.0)`. That sibling had no
    /// caller outside this module, so the DIR-R6 oracle was gating a path
    /// production had abandoned. The substitution is faithful because
    /// `(0.0, 1.0)` IS the zero-angle z-rotation: the deleted sibling applied
    /// it to the position on the way in and its inverse to the acceleration on
    /// the way out, which is exactly what `accumulate_spherical_gravity_frame`
    /// does with `r = I`. `delta_at_s` and
    /// `itrs_angular_velocity_gcrs` are not read on the gravity path.
    ///
    /// What changes is which summation kernel runs underneath:
    /// `spherical_gravity_impl_sincos_packed` before, `..._frame_packed` now.
    /// Those two are pinned to each other to 1e-13 relative by
    /// `satpy_core::gravity::frame_siblings_reproduce_the_frozen_sincos_path`,
    /// far inside this fixture's own tolerances.
    fn identity_rotation() -> FrameRotation {
        FrameRotation {
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            delta_at_s: 0.0,
            itrs_angular_velocity_gcrs: [0.0; 3],
        }
    }

    fn packed_test_gravity(
        order: usize,
        has_degree_one: bool,
    ) -> Result<Arc<PackedGravityCoeffs>, GravityError> {
        let stride = order.checked_add(1).ok_or(GravityError::UnsupportedOrder)?;
        let len = stride
            .checked_mul(stride)
            .ok_or(GravityError::InvalidCoefficientStorage)?;
        let mut c_coeffs = vec![0.0; len];
        let s_coeffs = vec![0.0; len];
        *c_coeffs
            .first_mut()
            .ok_or(GravityError::InvalidCoefficientStorage)? = 1.0;
        if has_degree_one && order >= 1 {
            *c_coeffs
                .get_mut(stride)
                .ok_or(GravityError::InvalidCoefficientStorage)? = 1.0e-6;
        }
        pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order).map(Arc::new)
    }

    fn packed_rhs(
        config: &ForceConfig,
        packed: Arc<PackedGravityCoeffs>,
    ) -> anyhow::Result<LightyearRHS> {
        LightyearRHS::try_new(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            2_451_545.0,
            Arc::new(*config),
            packed,
        )
    }

    #[test]
    fn validated_sun_position_preserves_bits_and_rejects_invalid_geometry() {
        let packed = packed_test_gravity(0, false).expect("test gravity fixture must pack");
        let make_rhs = |sun_pos| {
            LightyearRHS::new(
                [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
                0.0,
                2_460_000.5,
                Arc::new(ForceConfig {
                    sph_order: 0,
                    sun_pos: Some(sun_pos),
                    ..ForceConfig::default()
                }),
                Arc::clone(&packed),
            )
        };

        let valid = [149_597_870.7, -1234.5, 67.25];
        assert_eq!(
            make_rhs(valid)
                .validated_sun_position_at(2_460_000.5)
                .expect("finite nonzero Sun position")
                .map(f64::to_bits),
            valid.map(f64::to_bits)
        );
        for invalid in [
            [0.0; 3],
            [f64::NAN, 1.0, 0.0],
            [f64::INFINITY, 1.0, 0.0],
            [f64::MAX, f64::MAX, 0.0],
        ] {
            assert!(
                make_rhs(invalid)
                    .validated_sun_position_at(2_460_000.5)
                    .is_none(),
                "invalid Sun geometry must fail closed: {invalid:?}"
            );
        }
    }

    #[test]
    fn constructor_caps_packed_authority_once_and_builds_degree_one_pack() {
        let rhs = packed_rhs(
            &ForceConfig {
                sph_order: 2,
                subtract_first_order: true,
                ..ForceConfig::default()
            },
            packed_test_gravity(3, true).expect("packed scalar gravity fixture must pack"),
        )
        .expect("configured packed authority must construct");

        assert_eq!(rhs.packed.max_order(), 2);
        assert_eq!(rhs.gravity_mode, GravityEvalMode::ExplicitLowOrder);
        assert_eq!(
            rhs.packed_degree1
                .as_ref()
                .expect("nonzero degree one needs an explicit packed subtraction")
                .max_order(),
            1
        );
    }

    #[test]
    fn constructor_rejects_requested_order_wider_than_packed_authority() {
        let result = packed_rhs(
            &ForceConfig {
                sph_order: 2,
                ..ForceConfig::default()
            },
            packed_test_gravity(1, false).expect("packed scalar gravity fixture must pack"),
        );
        assert!(
            result.is_err(),
            "RHS must not widen packed gravity authority"
        );
        let Err(error) = result else {
            return;
        };

        assert!(
            error.to_string().contains("exceeds packed authority order"),
            "unexpected packed-authority rejection: {error}"
        );
    }

    #[test]
    fn scalar_packed_gravity_evaluator_error_returns_exact_error_and_latches_first_failure() {
        let rhs = packed_rhs(
            &ForceConfig {
                sph_order: 1,
                subtract_first_order: false,
                ..ForceConfig::default()
            },
            packed_test_gravity(1, false).expect("packed scalar gravity fixture must pack"),
        )
        .expect("valid packed authority must construct");

        assert_eq!(
            rhs.compute_internal(&[f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
            Err(GravityError::InvalidRadius),
            "invalid packed-gravity input must remain typed at the direct scalar boundary"
        );
        assert!(
            rhs.compute_internal(&[0.0; 6], 0.0)
                .expect("a valid direct scalar evaluation after failure must still run")
                .into_iter()
                .all(f64::is_finite),
            "a later valid evaluation must not replace the first gravity failure"
        );
        assert_eq!(
            rhs.take_gravity_error(),
            Some(GravityError::InvalidRadius),
            "RHS must retain the first typed gravity failure until its boundary consumes it"
        );
        assert_eq!(
            rhs.take_gravity_error(),
            None,
            "consuming the typed gravity failure must be one-shot"
        );
    }

    #[test]
    fn invalid_jd_returns_typed_gravity_error_and_reset_clears_latch() {
        let mut rhs = LightyearRHS::try_new(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            f64::NAN,
            Arc::new(ForceConfig {
                sph_order: 1,
                ..ForceConfig::default()
            }),
            packed_test_gravity(1, false).expect("packed scalar gravity fixture must pack"),
        )
        .expect("invalid epoch remains a runtime frame failure, not a constructor fallback");

        assert_eq!(
            rhs.compute_internal(&[0.0; 6], 0.0),
            Err(GravityError::InvalidTime),
            "invalid JD must fail as an invalid gravity evaluation time"
        );
        assert_eq!(rhs.take_gravity_error(), Some(GravityError::InvalidTime));

        rhs.compute_internal(&[0.0; 6], 0.0)
            .expect_err("invalid JD must not become valid through cache reuse");
        rhs.reset_cache();
        assert_eq!(
            rhs.take_gravity_error(),
            None,
            "a fresh propagation boundary must not inherit a consumed gravity failure"
        );
    }

    #[test]
    fn nonfinite_gravity_stage_time_returns_invalid_time_before_frame_rotation() {
        let rhs = packed_rhs(
            &ForceConfig {
                sph_order: 1,
                ..ForceConfig::default()
            },
            packed_test_gravity(1, false).expect("packed scalar gravity fixture must pack"),
        )
        .expect("valid packed authority must construct");

        assert_eq!(
            rhs.compute_internal(&[0.0; 6], f64::NAN),
            Err(GravityError::InvalidTime),
            "a non-finite RK-stage time must not become an invalid-radius fallback"
        );
        assert_eq!(rhs.take_gravity_error(), Some(GravityError::InvalidTime));
    }

    #[test]
    fn finite_stage_outside_frame_authority_returns_invalid_time() {
        let rhs = packed_rhs(
            &ForceConfig {
                sph_order: 1,
                ..ForceConfig::default()
            },
            packed_test_gravity(1, false).expect("packed scalar gravity fixture must pack"),
        )
        .expect("valid packed authority must construct");
        let outside_sealed_frame_s: f64 = 1_000_000_000.0;
        assert!(outside_sealed_frame_s.is_finite());

        assert_eq!(
            rhs.compute_internal(&[0.0; 6], outside_sealed_frame_s),
            Err(GravityError::InvalidTime),
            "a finite stage outside sealed frame authority must not become invalid radius"
        );
        assert_eq!(rhs.take_gravity_error(), Some(GravityError::InvalidTime));
    }

    #[test]
    fn finite_epoch_outside_frame_authority_returns_invalid_time() {
        let rhs = LightyearRHS::try_new(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            2_461_400.5,
            Arc::new(ForceConfig {
                sph_order: 1,
                ..ForceConfig::default()
            }),
            packed_test_gravity(1, false).expect("packed scalar gravity fixture must pack"),
        )
        .expect("finite epoch remains a runtime frame-authority failure");
        assert!(
            rhs.tai0_s.is_some(),
            "test requires a finite converted epoch beyond the sealed frame span"
        );

        assert_eq!(
            rhs.compute_internal(&[0.0; 6], 0.0),
            Err(GravityError::InvalidTime),
            "a finite epoch outside sealed frame authority must not become invalid radius"
        );
        assert_eq!(rhs.take_gravity_error(), Some(GravityError::InvalidTime));
    }

    #[test]
    fn non_gravity_frame_consumers_return_invalid_time_without_nan_rotation() {
        let configurations = [
            (
                "drag",
                ForceConfig {
                    sph_order: 0,
                    force_flags: ForceFlags::DRAG,
                    atm_model: 3,
                    am_ratio: 0.01,
                    cd: 2.2,
                    ..ForceConfig::default()
                },
            ),
            (
                "lorentz",
                ForceConfig {
                    sph_order: 0,
                    force_flags: ForceFlags::LORENTZ,
                    qm_ratio: 1.0e-7,
                    ..ForceConfig::default()
                },
            ),
            (
                "coulomb",
                ForceConfig {
                    sph_order: 0,
                    force_flags: ForceFlags::COULOMB_DRAG,
                    qm_ratio: 1.0e-7,
                    r_obj_m: 1.0e-3,
                    ..ForceConfig::default()
                },
            ),
        ];

        for (name, config) in configurations {
            let rhs = LightyearRHS::try_new(
                [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
                0.0,
                f64::NAN,
                Arc::new(config),
                packed_test_gravity(0, false).expect("packed scalar gravity fixture must pack"),
            )
            .expect("invalid epoch remains a runtime frame-authority failure");

            assert_eq!(
                rhs.compute_internal(&[0.0; 6], 0.0),
                Err(GravityError::InvalidTime),
                "{name} must return exact frame failure instead of a NaN rotation"
            );
            assert_eq!(
                rhs.take_gravity_error(),
                Some(GravityError::InvalidTime),
                "{name} must latch the exact first frame failure"
            );
        }
    }

    #[test]
    fn state_backed_force_helpers_return_nan_for_incomplete_state() {
        let incomplete_state = [0.0; 5];
        let incomplete_position = [0.0; 2];
        let rotation = identity_rotation();
        let sun = [AU_KM, 0.0, 0.0];

        for acceleration in [
            compute_drag(&incomplete_state, 1.0, 0.01, 2.2, &rotation),
            compute_relativity(&incomplete_state),
            compute_lorentz_frame(&incomplete_state, &rotation, 1.0e-7),
            compute_srp_with_precomputed(
                &incomplete_position,
                &sun,
                4.56e-6,
                1.0,
                0.01,
                EclipseSide::Lit,
            ),
        ] {
            assert!(acceleration.into_iter().all(f64::is_nan));
        }
    }

    fn compare_private_path_to_fixture() {
        let fixture: Value = serde_json::from_str(FIXTURE).expect("DIR-R6 fixture must be JSON");
        assert_eq!(
            fixture.get("schema").and_then(Value::as_str),
            Some("part_a_orekit_dir_r6_gravity_v1"),
            "accepted DIR-R6 fixture required"
        );
        let cases = fixture
            .get("cases")
            .and_then(Value::as_array)
            .expect("accepted DIR-R6 fixture must contain cases");
        assert_eq!(cases.len(), 5, "DIR-R6 fixture must contain five cases");
        let evaluation = fixture
            .get("evaluation")
            .unwrap_or_else(|| panic!("accepted DIR-R6 fixture must contain evaluation"));
        let abs_tol = hex_f64(
            evaluation
                .get("absolute_tolerance_m_s2")
                .unwrap_or_else(|| panic!("evaluation must contain absolute_tolerance_m_s2")),
            "absolute_tolerance_m_s2",
        );
        let rel_tol = hex_f64(
            evaluation
                .get("relative_tolerance")
                .unwrap_or_else(|| panic!("evaluation must contain relative_tolerance")),
            "relative_tolerance",
        );

        let packed = packed_constants_from_bytes(COEFFICIENTS, 5)
            .expect("sealed DIR-R6 coefficients must load");
        let config = Arc::new(ForceConfig {
            sph_order: 5,
            subtract_first_order: true,
            ..ForceConfig::default()
        });
        let rhs = LightyearRHS::new(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            2_451_545.0,
            config,
            packed,
        );
        assert_eq!(rhs.gravity_mode, GravityEvalMode::AnalyticCentral);

        for case in cases {
            let name = case
                .get("name")
                .and_then(Value::as_str)
                .expect("case name must be a string");
            let position_m = hex_vec3(case, "position_m");
            let expected = hex_vec3(case, "orekit_noncentral_m_s2");
            let [expected_x, expected_y, expected_z] = expected;
            let [x_km, y_km, z_km] = position_m.map(|position| position / 1000.0);
            let state = [x_km, y_km, z_km, 0.0, 0.0, 0.0];
            let r_km_sq = x_km.mul_add(x_km, y_km.mul_add(y_km, z_km * z_km));
            let mut actual_km_s2 = [0.0; 3];
            rhs.accumulate_spherical_gravity_frame(
                &state,
                &identity_rotation(),
                r_km_sq,
                r_km_sq.sqrt(),
                &mut RHSCache::default(),
                &mut actual_km_s2,
            )
            .expect("DIR-R6 packed gravity fixture must evaluate");
            let actual = actual_km_s2.map(|component| component * 1000.0);
            let [actual_x, actual_y, actual_z] = actual;
            for (axis, (actual_axis, expected_axis)) in [
                (actual_x, expected_x),
                (actual_y, expected_y),
                (actual_z, expected_z),
            ]
            .into_iter()
            .enumerate()
            {
                assert_close(
                    actual_axis,
                    expected_axis,
                    abs_tol,
                    rel_tol,
                    &format!("{name} axis {axis}"),
                );
            }
            let actual_norm = actual_x
                .mul_add(actual_x, actual_y.mul_add(actual_y, actual_z * actual_z))
                .sqrt();
            let expected_norm = expected_x
                .mul_add(
                    expected_x,
                    expected_y.mul_add(expected_y, expected_z * expected_z),
                )
                .sqrt();
            assert_close(
                actual_norm,
                expected_norm,
                abs_tol,
                rel_tol,
                &format!("{name} norm"),
            );
        }
    }

    #[test]
    fn dir_r6_packed_analytic_subtract_matches_orekit() {
        compare_private_path_to_fixture();
    }

    /// Scalar RHS retains only packed gravity authority. The raw differential
    /// oracle belongs to `satpy_core`, where raw source bytes remain local to
    /// its validation boundary.
    #[test]
    fn packed_gravity_authority_truncations_evaluate() {
        let packed = packed_constants_from_bytes(COEFFICIENTS, 8)
            .expect("sealed DIR-R6 coefficients must load");
        let position_itrs = [6_778.137, -125.0, 300.0];

        for order in [0, 1, 2, 5, 8] {
            let capped = packed
                .truncated_to(order)
                .expect("requested packed prefix must be authority-backed");
            let acceleration = satpy_core::spherical_gravity_impl_frame_packed(
                &position_itrs,
                &mut satpy_core::GravityCache::default(),
                &capped,
            )
            .expect("valid packed gravity evaluation must succeed");
            assert!(
                acceleration.into_iter().all(f64::is_finite),
                "packed gravity order {order} must remain finite"
            );
        }
    }
}

#[inline]
fn position3(values: &[f64]) -> Option<&[f64; 3]> {
    values.get(..3)?.try_into().ok()
}

#[inline]
fn state6(values: &[f64]) -> Option<&[f64; 6]> {
    values.try_into().ok()
}

#[inline]
fn accumulate_axes3(total: &mut [f64; 3], add: &[f64]) {
    let Some(&[add_x, add_y, add_z]) = position3(add) else {
        return;
    };
    let [total_x, total_y, total_z] = total;
    *total_x += add_x;
    *total_y += add_y;
    *total_z += add_z;
}

#[inline]
fn mul_add_axes3(total: &mut [f64; 3], vec: &[f64], scale: f64) {
    let Some(&[vec_x, vec_y, vec_z]) = position3(vec) else {
        return;
    };
    let [total_x, total_y, total_z] = total;
    *total_x = vec_x.mul_add(scale, *total_x);
    *total_y = vec_y.mul_add(scale, *total_y);
    *total_z = vec_z.mul_add(scale, *total_z);
}

/// Battin's f(q) formulation for numerically stable Encke gravity correction.
///
/// Replaces the naive subtraction `μ·(r_base/|r_base|³ - r_pert/|r_pert|³)`, which
/// suffers catastrophic cancellation when `δr` is small relative to `r_base`.
///
/// Instead computes `μ/|r_pert|³ · (f(q)·r_base - δr)`, where
/// `q = δr·(δr + 2·r_base) / |r_base|²` and
/// `f(q) = q·(3 + 3q + q²) / (1 + (1+q)^{3/2})`.
///
/// Reference: Battin (1999) "An Introduction to the Mathematics and Methods
/// of Astrodynamics", §9.3, Eq. 9.69–9.73
#[cfg_attr(not(feature = "profile-symbols"), inline)]
#[cfg_attr(feature = "profile-symbols", inline(never))]
fn battin_encke_gravity_correction(
    r_base: &[f64; 3],
    delta_r: &[f64; 3],
    r_pert_sq: f64,
    r_pert: f64,
    total_acc: &mut [f64; 3],
) {
    let &[r_base_x, r_base_y, r_base_z] = r_base;
    let &[delta_x, delta_y, delta_z] = delta_r;
    let [total_x, total_y, total_z] = total_acc;
    let r_base_sq = r_base_x.mul_add(r_base_x, r_base_y.mul_add(r_base_y, r_base_z * r_base_z));
    // Use positive tests: NaN > 0.0 is false, so this correctly
    // guards against degenerate orbits where equinoc2eci returns NaN.
    if !(r_base_sq > 0.0 && r_pert_sq > 0.0) {
        return;
    }

    // q = δr·(δr + 2·r_base) / |r_base|²
    // Expanded: q = (|δr|² + 2·δr·r_base) / |r_base|²
    let delta_dot_rbase = delta_x.mul_add(r_base_x, delta_y.mul_add(r_base_y, delta_z * r_base_z));
    let delta_sq = delta_x.mul_add(delta_x, delta_y.mul_add(delta_y, delta_z * delta_z));
    let q = (delta_sq + 2.0 * delta_dot_rbase) / r_base_sq;

    // f(q) = q * (3 + 3q + q²) / (1 + (1+q)^{3/2})
    let q1 = 1.0 + q;
    let fq = q * (3.0 + q * (3.0 + q)) / (1.0 + (q1 * q1.sqrt()));

    // μ·(r_base/|r_base|³ - r_pert/|r_pert|³) = μ/|r_pert|³ · (f(q)·r_base - δr)
    let inv_rpn3 = 1.0 / (r_pert_sq * r_pert);
    let mu_inv_rpn3 = MU * inv_rpn3;
    *total_x += mu_inv_rpn3 * fq.mul_add(r_base_x, -delta_x);
    *total_y += mu_inv_rpn3 * fq.mul_add(r_base_y, -delta_y);
    *total_z += mu_inv_rpn3 * fq.mul_add(r_base_z, -delta_z);
}

/// Compute atmospheric drag acceleration, in km/s².
///
/// Three adjacent `f64` parameters carry three different units, none checked by
/// the type system: `rho` in kg/m³, `am_ratio` in m²/kg, and dimensionless
/// `cd`. `state` is km and km/s; the body converts to SI for the drag equation
/// and back on the way out. Atmosphere velocity comes from the same resolved
/// full frame used by density geometry.
///
/// `am_ratio` before `cd` is the crate convention — it is the declaration order
/// of `ForceConfig` (`types.rs:279`). Transposing them here is silent: both are
/// order-1 positive numbers.
#[cfg_attr(not(feature = "profile-symbols"), inline)]
#[cfg_attr(feature = "profile-symbols", inline(never))]
fn compute_drag(
    state: &[f64],
    rho: f64,
    am_ratio: f64,
    cd: f64,
    rotation: &FrameRotation,
) -> [f64; 3] {
    if cd == 0.0 || am_ratio == 0.0 || rho <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let Some(&[r_x, r_y, r_z, v_x, v_y, v_z]) = state6(state) else {
        return [f64::NAN; 3];
    };

    // Atmosphere co-rotates with the full resolved Earth frame, not an
    // independent scalar rotation about GCRS +z.
    let [omega_x, omega_y, omega_z] = rotation.itrs_angular_velocity_gcrs;
    let v_rel_x = v_x - (omega_y * r_z - omega_z * r_y);
    let v_rel_y = v_y - (omega_z * r_x - omega_x * r_z);
    let v_rel_z = v_z - (omega_x * r_y - omega_y * r_x);

    // Convert to m/s
    let v_rel_m_x = v_rel_x * KM_TO_M;
    let v_rel_m_y = v_rel_y * KM_TO_M;
    let v_rel_m_z = v_rel_z * KM_TO_M;

    let v_rel_sq = v_rel_m_x.mul_add(
        v_rel_m_x,
        v_rel_m_y.mul_add(v_rel_m_y, v_rel_m_z * v_rel_m_z),
    );

    if v_rel_sq == 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let v_rel = v_rel_sq.sqrt();
    let a_coef = -0.5 * cd * am_ratio * rho * v_rel;

    [
        a_coef * v_rel_m_x * M_TO_KM,
        a_coef * v_rel_m_y * M_TO_KM,
        a_coef * v_rel_m_z * M_TO_KM,
    ]
}

#[cfg_attr(not(feature = "profile-symbols"), inline)]
#[cfg_attr(feature = "profile-symbols", inline(never))]
fn compute_relativity(state: &[f64]) -> [f64; 3] {
    let Some(&[r_x, r_y, r_z, v_x, v_y, v_z]) = state6(state) else {
        return [f64::NAN; 3];
    };
    let r = (r_x * r_x + r_y * r_y + r_z * r_z).sqrt();
    let v = (v_x * v_x + v_y * v_y + v_z * v_z).sqrt();
    if r == 0.0 || v == 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let inv_r = 1.0 / r;
    let inv_v = 1.0 / v;
    let r_hat_x = r_x * inv_r;
    let r_hat_y = r_y * inv_r;
    let r_hat_z = r_z * inv_r;
    let v_hat_x = v_x * inv_v;
    let v_hat_y = v_y * inv_v;
    let v_hat_z = v_z * inv_v;
    let mur = MU * inv_r;
    let v2 = v * v;
    let rv_dot = r_hat_x.mul_add(v_hat_x, r_hat_y.mul_add(v_hat_y, r_hat_z * v_hat_z));
    let mur_div_r = mur * inv_r;
    let scale = mur_div_r * INV_LIGHT_SPEED_SQ;

    let pt1_scale = 4.0 * mur - v2;
    let pt2_scale = 4.0 * v2 * rv_dot;
    [
        scale * (pt1_scale * r_hat_x + pt2_scale * v_hat_x),
        scale * (pt1_scale * r_hat_y + pt2_scale * v_hat_y),
        scale * (pt1_scale * r_hat_z + pt2_scale * v_hat_z),
    ]
}

/// Lorentz acceleration resolved through the full frame authority.
///
/// Scope decision 1 — the dipole comes OFF GMST. The geomagnetic dipole is an
/// Earth-fixed direction; the legacy form built it as `R3(-GMST) * [sin_theta,
/// 0, cos_theta]`, which silently makes the magnetic field depend on a rotation
/// that omits precession, nutation and polar motion. Here the same Earth-fixed
/// vector is carried to GCRS by the IAU 2006/2000A rotation instead.
///
/// Scope decision 4 — `v_rel` uses the full ITRS angular velocity rather than a
/// scalar z rate. The atmosphere and the field co-rotate about the CIP, which
/// is not the GCRS z axis; a scalar-z form assumes it is. The vector is derived
/// from the exact analytic derivative of the centred per-stage frame
/// interpolant.
#[cfg_attr(not(feature = "profile-symbols"), inline)]
#[cfg_attr(feature = "profile-symbols", inline(never))]
fn compute_lorentz_frame(state: &[f64], rotation: &FrameRotation, qm_ratio: f64) -> [f64; 3] {
    if qm_ratio == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let Some(&[r_x, r_y, r_z, v_x, v_y, v_z]) = state6(state) else {
        return [f64::NAN; 3];
    };
    let r_norm = (r_x * r_x + r_y * r_y + r_z * r_z).sqrt();
    if r_norm == 0.0 {
        return [0.0, 0.0, 0.0];
    }

    // Earth-fixed dipole direction, carried to GCRS by the authority rotation.
    let [dipole_x, dipole_y, dipole_z] =
        rotation.to_gcrs(&[LORENTZ_THETA_SIN, 0.0, LORENTZ_THETA_COS]);

    let inv_r = 1.0 / r_norm;
    let r_hat_x = r_x * inv_r;
    let r_hat_y = r_y * inv_r;
    let r_hat_z = r_z * inv_r;
    let dipole_strength_t_km3 = EARTH_DIPOLE_STRENGTH / (KM_TO_M * KM_TO_M * KM_TO_M);
    let dot = dipole_x.mul_add(r_hat_x, dipole_y.mul_add(r_hat_y, dipole_z * r_hat_z));
    let b_scale = dipole_strength_t_km3 * inv_r * inv_r * inv_r;
    let b_x = b_scale * (3.0 * dot * r_hat_x - dipole_x);
    let b_y = b_scale * (3.0 * dot * r_hat_y - dipole_y);
    let b_z = b_scale * (3.0 * dot * r_hat_z - dipole_z);

    // v_rel = v - omega_gcrs x r.
    let [omega_x, omega_y, omega_z] = rotation.itrs_angular_velocity_gcrs;
    let v_rel_x = v_x - (omega_y * r_z - omega_z * r_y);
    let v_rel_y = v_y - (omega_z * r_x - omega_x * r_z);
    let v_rel_z = v_z - (omega_x * r_y - omega_y * r_x);
    let v_rel_mps_x = v_rel_x * KM_TO_M;
    let v_rel_mps_y = v_rel_y * KM_TO_M;
    let v_rel_mps_z = v_rel_z * KM_TO_M;

    let acc_si_x = qm_ratio * (v_rel_mps_y * b_z - v_rel_mps_z * b_y);
    let acc_si_y = qm_ratio * (v_rel_mps_z * b_x - v_rel_mps_x * b_z);
    let acc_si_z = qm_ratio * (v_rel_mps_x * b_y - v_rel_mps_y * b_x);
    [acc_si_x * M_TO_KM, acc_si_y * M_TO_KM, acc_si_z * M_TO_KM]
}

/// Solar radiation pressure acceleration, in km/s².
///
/// NOTE THE PARAMETER ORDER: this takes `cr` BEFORE `am_ratio`, which is the
/// reverse of the crate convention set by `ForceConfig`'s declaration order
/// (`types.rs:279-281`, `am_ratio, cd, cr`) and the reverse of `compute_drag`
/// just above. Both are order-1 positive f64 with no unit in the type, so a
/// transposed call compiles, runs, and is wrong by the ratio of the two.
/// Recorded rather than fixed: normalising the order touches all eight call
/// sites, and the failure mode of getting one of them wrong is exactly the
/// silent one this note exists to warn about.
#[cfg_attr(not(feature = "profile-symbols"), inline)]
#[cfg_attr(feature = "profile-symbols", inline(never))]
fn compute_srp_with_precomputed(
    state: &[f64],
    sun_pos: &[f64; 3],
    p_sun: f64,
    cr: f64,
    am_ratio: f64,
    eclipse_side: EclipseSide,
) -> [f64; 3] {
    if cr == 0.0 || am_ratio == 0.0 || p_sun == 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let Some(&[sat_x, sat_y, sat_z]) = position3(state) else {
        return [f64::NAN; 3];
    };
    if eclipse_side == EclipseSide::Shadow {
        return [0.0, 0.0, 0.0];
    }

    let &[sun_x, sun_y, sun_z] = sun_pos;
    let rx = sun_x - sat_x;
    let ry = sun_y - sat_y;
    let rz = sun_z - sat_z;
    let dist_sq = rx.mul_add(rx, ry.mul_add(ry, rz * rz));

    if !(dist_sq.is_finite() && dist_sq > 0.0) {
        return [f64::NAN; 3];
    }

    let inv_dist = 1.0 / dist_sq.sqrt();
    let inverse_square_scale = (AU_KM * AU_KM) / dist_sq;
    let a_mag = p_sun * cr * am_ratio * inverse_square_scale * M_TO_KM;

    [
        -a_mag * rx * inv_dist,
        -a_mag * ry * inv_dist,
        -a_mag * rz * inv_dist,
    ]
}

/// Compute third-body gravity acceleration using precomputed invariants.
/// This avoids repeated sqrt/div operations for the same body.
///
/// TEST ADAPTER over `accumulate_thirdbody_grav_precomputed`, which is what
/// production calls. Returning a fresh array instead of accumulating into a
/// caller's buffer is convenient for the scalar-vs-SIMD comparisons and
/// pointless in the RHS, so this is `cfg(test)`.
#[cfg(test)]
#[inline]
fn compute_thirdbody_grav_precomputed(sat_pos: &[f64], inv: &BodyInvariants) -> [f64; 3] {
    let mut acc = [0.0; 3];
    accumulate_thirdbody_grav_precomputed(sat_pos, inv, &mut acc);
    acc
}

#[cfg_attr(not(feature = "profile-symbols"), inline)]
#[cfg_attr(feature = "profile-symbols", inline(never))]
fn accumulate_thirdbody_grav_precomputed(
    sat_pos: &[f64],
    inv: &BodyInvariants,
    acc: &mut [f64; 3],
) {
    let Some(&[sat_x, sat_y, sat_z]) = position3(sat_pos) else {
        return;
    };
    let [body_x, body_y, body_z] = inv.body_norm;
    let rel_x = body_x - sat_x * inv.inv_body_dist;
    let rel_y = body_y - sat_y * inv.inv_body_dist;
    let rel_z = body_z - sat_z * inv.inv_body_dist;

    let rel_dist_sq = rel_x.mul_add(rel_x, rel_y.mul_add(rel_y, rel_z * rel_z));

    if rel_dist_sq == 0.0 {
        return;
    }

    let rel_dist = rel_dist_sq.sqrt();
    let inv_rel_dist_cubed = 1.0 / (rel_dist_sq * rel_dist);
    let [acc_x, acc_y, acc_z] = acc;
    let updated_x = *acc_x + inv.mu_coef * (rel_x * inv_rel_dist_cubed - body_x);
    let updated_y = *acc_y + inv.mu_coef * (rel_y * inv_rel_dist_cubed - body_y);
    let updated_z = *acc_z + inv.mu_coef * (rel_z * inv_rel_dist_cubed - body_z);
    *acc_x = updated_x;
    *acc_y = updated_y;
    *acc_z = updated_z;
}

/// Compute third-body gravity acceleration (original version for SIMD tests)
#[cfg(test)]
#[inline]
fn compute_thirdbody_grav(sat_pos: &[f64], body_pos: &[f64; 3], mu_body: f64) -> [f64; 3] {
    BodyInvariants::precompute(body_pos, mu_body).map_or([0.0, 0.0, 0.0], |inv| {
        compute_thirdbody_grav_precomputed(sat_pos, &inv)
    })
}

/// SIMD third-body gravity: compute perturbations from 4 bodies simultaneously.
///
/// # Arguments
/// * `sat_pos` - Satellite position [x, y, z] in km
/// * `body_norm_x/y/z` - f64x4 containing normalized body position components
/// * `inv_body_dist` - f64x4 containing inverse body distances
/// * `mu_coef` - f64x4 containing `mu * inv_body_dist^2`
/// * `active_mask` - f64x4 mask (all bits set for active bodies, zero for inactive)
///
/// # Returns
/// [ax, ay, az] acceleration components (summed across all 4 bodies)
///
/// TEST ADAPTER. Production splats the satellite position once per RHS call and
/// goes straight to `compute_thirdbody_grav_simd4_lanes_splatted`, keeping the
/// result in lanes; this entry point splats per call and reduces to scalars,
/// which is only what the equivalence tests want.
#[cfg(test)]
#[inline]
fn compute_thirdbody_grav_simd4(
    sat_pos: &[f64; 3],
    body_norm_x: f64x4,
    body_norm_y: f64x4,
    body_norm_z: f64x4,
    inv_body_dist: f64x4,
    mu_coef: f64x4,
    active_mask: f64x4,
) -> [f64; 3] {
    let &[x, y, z] = sat_pos;
    let sat_x = f64x4::splat(x);
    let sat_y = f64x4::splat(y);
    let sat_z = f64x4::splat(z);
    compute_thirdbody_grav_simd4_splatted(
        sat_x,
        sat_y,
        sat_z,
        body_norm_x,
        body_norm_y,
        body_norm_z,
        inv_body_dist,
        mu_coef,
        active_mask,
    )
}

/// TEST ADAPTER: the scalar-reducing form of
/// `compute_thirdbody_grav_simd4_lanes_splatted`. Production keeps the lanes.
#[cfg(test)]
#[inline]
fn compute_thirdbody_grav_simd4_splatted(
    sat_x: f64x4,
    sat_y: f64x4,
    sat_z: f64x4,
    body_norm_x: f64x4,
    body_norm_y: f64x4,
    body_norm_z: f64x4,
    inv_body_dist: f64x4,
    mu_coef: f64x4,
    active_mask: f64x4,
) -> [f64; 3] {
    let (ax, ay, az) = compute_thirdbody_grav_simd4_lanes_splatted(
        sat_x,
        sat_y,
        sat_z,
        body_norm_x,
        body_norm_y,
        body_norm_z,
        inv_body_dist,
        mu_coef,
        active_mask,
    );
    [ax.reduce_add(), ay.reduce_add(), az.reduce_add()]
}

/// Distance-squared floor for the third-body denominator, as a `const` item:
/// rodata plus one `ldr q` per use. Written inline, `f64x4::splat(1e-30)` is
/// materialised through `bl _memset_pattern16` on aarch64-macos — a libc call
/// in the RHS loop body (measured in `jb_rs::jb2008`, see its `wide_const`
/// module). Lane values are byte-identical to the splat.
const REL_DIST_SQ_FLOOR: f64x4 = f64x4::new([1e-30; 4]);

#[inline]
fn compute_thirdbody_grav_simd4_lanes_splatted(
    sat_x: f64x4,
    sat_y: f64x4,
    sat_z: f64x4,
    body_norm_x: f64x4,
    body_norm_y: f64x4,
    body_norm_z: f64x4,
    inv_body_dist: f64x4,
    mu_coef: f64x4,
    active_mask: f64x4,
) -> (f64x4, f64x4, f64x4) {
    let zero = f64x4::ZERO;
    let one = f64x4::ONE;

    let rel_norm_x = body_norm_x - sat_x * inv_body_dist;
    let rel_norm_y = body_norm_y - sat_y * inv_body_dist;
    let rel_norm_z = body_norm_z - sat_z * inv_body_dist;

    // Relative distance squared (in normalized space)
    let rel_xy_sq = rel_norm_x * rel_norm_x + rel_norm_y * rel_norm_y;
    let rel_dist_sq = rel_xy_sq + rel_norm_z * rel_norm_z;

    // Safe relative distance
    let rel_dist_sq_safe = active_mask.select(rel_dist_sq.max(REL_DIST_SQ_FLOOR), one);
    let rel_dist = rel_dist_sq_safe.sqrt();
    let inv_rel_dist_cubed = one / (rel_dist_sq_safe * rel_dist);

    // Acceleration components (before masking)
    let first_raw = mu_coef * (rel_norm_x * inv_rel_dist_cubed - body_norm_x);
    let second_raw = mu_coef * (rel_norm_y * inv_rel_dist_cubed - body_norm_y);
    let third_raw = mu_coef * (rel_norm_z * inv_rel_dist_cubed - body_norm_z);

    // Apply mask (zero out inactive bodies)
    let first = active_mask.select(first_raw, zero);
    let second = active_mask.select(second_raw, zero);
    let third = active_mask.select(third_raw, zero);
    (first, second, third)
}

#[inline]
fn compute_thirdbody_grav_simd4_lanes_all_active(
    sat_x: f64x4,
    sat_y: f64x4,
    sat_z: f64x4,
    body_norm_x: f64x4,
    body_norm_y: f64x4,
    body_norm_z: f64x4,
    inv_body_dist: f64x4,
    mu_coef: f64x4,
) -> (f64x4, f64x4, f64x4) {
    let one = f64x4::ONE;
    let rel_norm_x = body_norm_x - sat_x * inv_body_dist;
    let rel_norm_y = body_norm_y - sat_y * inv_body_dist;
    let rel_norm_z = body_norm_z - sat_z * inv_body_dist;
    let rel_xy_sq = rel_norm_x * rel_norm_x + rel_norm_y * rel_norm_y;
    let rel_dist_sq = rel_xy_sq + rel_norm_z * rel_norm_z;
    let rel_dist_sq_safe = rel_dist_sq.max(REL_DIST_SQ_FLOOR);
    let rel_dist = rel_dist_sq_safe.sqrt();
    let inv_rel_dist_cubed = one / (rel_dist_sq_safe * rel_dist);
    let first = mu_coef * (rel_norm_x * inv_rel_dist_cubed - body_norm_x);
    let second = mu_coef * (rel_norm_y * inv_rel_dist_cubed - body_norm_y);
    let third = mu_coef * (rel_norm_z * inv_rel_dist_cubed - body_norm_z);
    (first, second, third)
}

/// Pack third-body invariants into SIMD-friendly layout.
/// Bodies are packed as: [Sun, Moon, Jupiter, Venus] in first call,
///                       [Mars, Saturn, 0, 0] in second call (if needed).
#[inline]
fn pack_thirdbody_invariants(
    b0: Option<BodyInvariants>,
    b1: Option<BodyInvariants>,
    b2: Option<BodyInvariants>,
    b3: Option<BodyInvariants>,
) -> (f64x4, f64x4, f64x4, f64x4, f64x4, f64x4) {
    // SIMD mask convention: all bits set = true (active), all zeros = false (inactive)
    // This matches the wide crate's select() semantics where mask bits select between values
    const SIMD_MASK_TRUE_BITS: u64 = !0u64;

    // Helper to extract fields from Option<BodyInvariants>, avoiding intermediate
    // stack arrays and the store-then-load pattern.
    #[inline]
    fn field_or_zero(body: Option<&BodyInvariants>, field: impl Fn(&BodyInvariants) -> f64) -> f64 {
        body.map_or(0.0, field)
    }
    #[inline]
    const fn mask_val(body: Option<&BodyInvariants>) -> u64 {
        if body.is_some() {
            SIMD_MASK_TRUE_BITS
        } else {
            0
        }
    }

    (
        f64x4::new([
            field_or_zero(b0.as_ref(), |invariants| invariants.body_norm[0]),
            field_or_zero(b1.as_ref(), |invariants| invariants.body_norm[0]),
            field_or_zero(b2.as_ref(), |invariants| invariants.body_norm[0]),
            field_or_zero(b3.as_ref(), |invariants| invariants.body_norm[0]),
        ]),
        f64x4::new([
            field_or_zero(b0.as_ref(), |invariants| invariants.body_norm[1]),
            field_or_zero(b1.as_ref(), |invariants| invariants.body_norm[1]),
            field_or_zero(b2.as_ref(), |invariants| invariants.body_norm[1]),
            field_or_zero(b3.as_ref(), |invariants| invariants.body_norm[1]),
        ]),
        f64x4::new([
            field_or_zero(b0.as_ref(), |invariants| invariants.body_norm[2]),
            field_or_zero(b1.as_ref(), |invariants| invariants.body_norm[2]),
            field_or_zero(b2.as_ref(), |invariants| invariants.body_norm[2]),
            field_or_zero(b3.as_ref(), |invariants| invariants.body_norm[2]),
        ]),
        f64x4::new([
            field_or_zero(b0.as_ref(), |invariants| invariants.inv_body_dist),
            field_or_zero(b1.as_ref(), |invariants| invariants.inv_body_dist),
            field_or_zero(b2.as_ref(), |invariants| invariants.inv_body_dist),
            field_or_zero(b3.as_ref(), |invariants| invariants.inv_body_dist),
        ]),
        f64x4::new([
            field_or_zero(b0.as_ref(), |invariants| invariants.mu_coef),
            field_or_zero(b1.as_ref(), |invariants| invariants.mu_coef),
            field_or_zero(b2.as_ref(), |invariants| invariants.mu_coef),
            field_or_zero(b3.as_ref(), |invariants| invariants.mu_coef),
        ]),
        f64x4::new([
            f64::from_bits(mask_val(b0.as_ref())),
            f64::from_bits(mask_val(b1.as_ref())),
            f64::from_bits(mask_val(b2.as_ref())),
            f64::from_bits(mask_val(b3.as_ref())),
        ]),
    )
}

#[inline]
fn make_thirdbody_pack(
    b0: Option<BodyInvariants>,
    b1: Option<BodyInvariants>,
    b2: Option<BodyInvariants>,
    b3: Option<BodyInvariants>,
) -> ThirdBodySimdPack {
    let (body_norm_x, body_norm_y, body_norm_z, inv_body_dist, mu_coef, mask) =
        pack_thirdbody_invariants(b0, b1, b2, b3);
    let active_count = usize::from(b0.is_some())
        .saturating_add(usize::from(b1.is_some()))
        .saturating_add(usize::from(b2.is_some()))
        .saturating_add(usize::from(b3.is_some()));
    let active = active_count > 0;
    let all_active = active_count == 4;
    ThirdBodySimdPack {
        body_norm_x,
        body_norm_y,
        body_norm_z,
        inv_body_dist,
        mu_coef,
        mask,
        active,
        all_active,
    }
}

/// Resolved atmospheric model variant — lifted from `atm_model: i32` at RHS construction time
/// to eliminate the per-call if-chain in the hot path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
enum AtmModel {
    None = 0,
    Exponential = 1,
    SyntheticThermosphereProxyV1 = 3,
    Jb2008 = 4,
    /// Candidate-only x4 approximation; model4 remains exact.
    Jb2008LogQuadratureX4ApproxV1 = 5,
    /// Coarse-abscissa x4 approximation (R16 arm C); model4 remains exact and
    /// model5 remains available for comparison.
    Jb2008LogQuadratureX4ApproxV2 = 6,
    /// Model 6's quadrature with the two fixed plans replaced by a degree-14
    /// fit in the exospheric temperature (R28's ladder, R31). Models 4, 5 and 6
    /// are all untouched and remain available for comparison.
    Jb2008FittedV7 = 7,
    /// Part A v3 keeps model 7's fitted density kernel but selects the sealed
    /// persistence-scenario driver authority.
    Jb2008FittedV7PartAV3Persistence = 8,
}

/// Immutable compiled JB2008 driver authority selected by an atmosphere model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Jb2008DriverAuthority {
    CompiledSetV2,
    PartAV3PersistenceV1,
}

impl Jb2008DriverAuthority {
    /// Load selected linked-image authority.
    ///
    /// # Errors
    ///
    /// Returns an error when compiled authority validation fails.
    pub fn load(self) -> anyhow::Result<std::sync::Arc<Jb2008Drivers>> {
        match self {
            Self::CompiledSetV2 => jb_rs::drivers::compiled_drivers(),
            Self::PartAV3PersistenceV1 => jb_rs::drivers::compiled_part_a_v3_drivers(),
        }
    }
}

/// Which JB2008 quadrature profile a scalar density call runs.
///
/// This was a `bool` while there were two profiles. It is an enum now so that
/// adding a third could not silently re-route the second: an `if` has no arm
/// for a value it does not name, and a `match` will not compile without one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Jb2008Profile {
    Exact,
    ApproxV1,
    ApproxV2,
    FittedV7,
}

impl AtmModel {
    #[inline]
    const fn uses_jb2008_drivers(self) -> bool {
        self.jb2008_driver_authority().is_some()
    }

    const fn jb2008_driver_authority(self) -> Option<Jb2008DriverAuthority> {
        match self {
            Self::Jb2008
            | Self::Jb2008LogQuadratureX4ApproxV1
            | Self::Jb2008LogQuadratureX4ApproxV2
            | Self::Jb2008FittedV7 => Some(Jb2008DriverAuthority::CompiledSetV2),
            Self::Jb2008FittedV7PartAV3Persistence => {
                Some(Jb2008DriverAuthority::PartAV3PersistenceV1)
            }
            Self::None | Self::Exponential | Self::SyntheticThermosphereProxyV1 => None,
        }
    }

    fn from_i32(v: i32) -> anyhow::Result<Self> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::Exponential),
            3 => Ok(Self::SyntheticThermosphereProxyV1),
            4 => Ok(Self::Jb2008),
            5 => Ok(Self::Jb2008LogQuadratureX4ApproxV1),
            6 => Ok(Self::Jb2008LogQuadratureX4ApproxV2),
            7 => Ok(Self::Jb2008FittedV7),
            8 => Ok(Self::Jb2008FittedV7PartAV3Persistence),
            other => Err(anyhow::anyhow!(
                "Unknown atm_model: {other}. Valid values: 0 (none), 1 (exponential), 3 (versioned_synthetic_thermosphere_proxy_v1), 4 (exact JB2008), 5 (candidate JB2008 log-quadrature x4 approximation v1), 6 (coarse-abscissa JB2008 log-quadrature x4 approximation v2), 7 (fitted-kernel JB2008, degree-14 fixed plans), 8 (Part A v3 fitted-kernel JB2008 persistence authority)"
            )),
        }
    }
}

pub(crate) fn validate_atmosphere_model_code(atm_model: i32) -> anyhow::Result<()> {
    AtmModel::from_i32(atm_model).map(|_| ())
}

/// Whether a raw `atm_model` code selects a JB2008 variant.
///
/// This is the only place that answer is spelled. Guards that need it live in
/// four modules across two crates, and before this existed each carried its own
/// `matches!(atm_model, 4 | 5)`. Landing model 6 (a697d6c) updated [`AtmModel`]
/// and left every hand-written copy behind, so six guards silently stopped
/// applying to the model production actually flies. Route new guards through
/// here: an unknown code is not JB2008, and a new JB2008 variant is one arm in
/// [`AtmModel::uses_jb2008_drivers`].
#[must_use]
pub fn atm_model_uses_jb2008_drivers(atm_model: i32) -> bool {
    AtmModel::from_i32(atm_model).is_ok_and(AtmModel::uses_jb2008_drivers)
}

/// Resolve one immutable JB2008 authority for a recognized atmosphere model.
#[must_use]
pub fn jb2008_driver_authority(atm_model: i32) -> Option<Jb2008DriverAuthority> {
    AtmModel::from_i32(atm_model)
        .ok()
        .and_then(AtmModel::jb2008_driver_authority)
}

/// Get atmospheric density from state.
#[inline]
fn density_from_state(
    state: &[f64],
    jd: f64,
    rotation: &FrameRotation,
    earth_radius: f64,
    atm_model: AtmModel,
    alt_km_precomputed: Option<f64>,
) -> f64 {
    if atm_model == AtmModel::None {
        return 0.0;
    }
    let Some(&[state_x, state_y, state_z]) = position3(state) else {
        return f64::NAN;
    };

    let alt_km = alt_km_precomputed.unwrap_or_else(|| {
        let r_km = state_x
            .mul_add(state_x, state_y.mul_add(state_y, state_z * state_z))
            .sqrt();
        r_km - earth_radius
    });

    // This function takes no cache, and that is the point: an altitude-only
    // cache is invalid for an epoch/latitude/longitude proxy and would create a
    // tolerance-independent error floor for every model. The rule used to be a
    // comment guarding an unused `_cache` parameter; taking no cache at all
    // states it in the signature instead.

    // Legacy models stop at 1000 km. JB2008 is exempted from that ceiling
    // here, and what bounds it lives in `jb2008_density_at_state` below:
    //
    // - LOWER bound: the kernel itself rejects `sat_altitude_m <
    //   JB_ALTITUDE_MIN_M` (90 km) with `Jb2008Error::AltitudeOutOfRange`.
    // - UPPER handling: there is NO validity-range rejection. Above 2500 km --
    //   the Jacchia-family ceiling per AIAA G-003C-2010 -- the helper caps the
    //   returned density at the exospheric ceiling (see
    //   `JB2008_EXOSPHERIC_*`), because the kernel's own extrapolation is
    //   non-monotone out there and overestimates the exosphere by 1-2 orders.
    //
    // An earlier version of this comment claimed JB2008 "owns its 90--3000 km
    // validity range in the dedicated RHS helper below"; no 3000 km bound was
    // ever implemented, and that lie survived one merge resolution after being
    // corrected once. If you touch this text, grep for the helper's cap tests
    // and keep the three statements above consistent with them.
    if !atm_model.uses_jb2008_drivers() && alt_km >= 1000.0 {
        return 0.0;
    }

    match atm_model {
        AtmModel::None => 0.0,
        AtmModel::Exponential => {
            const RHO0: f64 = 1.225;
            const H_KM: f64 = 8.5;
            if alt_km <= 0.0 {
                RHO0
            } else if alt_km >= 1000.0 {
                0.0
            } else {
                RHO0 * (-alt_km / H_KM).exp()
            }
        }
        AtmModel::SyntheticThermosphereProxyV1 => {
            // Scope decision 3: the proxy's LOCAL TIME is a function of the
            // Earth-fixed longitude and the civil epoch. Both now come from the
            // frame authority — the longitude via the full IAU 2006/2000A
            // rotation rather than `R3(GMST1982)`, and `jd` as the UTC the
            // driver path resolves, so local time is self-consistent with the
            // rotation that produced the longitude.
            let pos_itrs = rotation.to_itrs(&[state_x, state_y, state_z]);
            // UNITS. `geocentric_spherical_from_itrs` returns RADIANS; its
            // near-namesake `eci_to_geocentric_spherical` returns DEGREES. Both
            // live in `satpy_core/src/lib.rs`; grep the names rather than
            // trusting a line number, because the two this comment carried
            // (:493 and :525) had drifted to :810 and :831 by 2026-08-04. The proxy
            // takes `lat_deg`/`lon_deg`. Passing the radians straight through
            // collapsed a +/-77.6 deg latitude span onto +/-1.354 "deg", killing
            // the model's 25*sin^2(lat) and 0.3*|lat| terms and its local-time
            // dependence, and moved density by up to 41.5x at the fixture's
            // release states.
            let (lat_rad, lon_rad, alt) =
                satpy_core::geocentric_spherical_from_itrs(&pos_itrs, earth_radius);
            let (rho, _, _) = synthetic_thermosphere_proxy_eval_impl(
                jd,
                lat_rad.to_degrees(),
                lon_rad.to_degrees(),
                alt,
            );
            rho
        }
        // Scalar JB2008 must go through LightyearRHS::density_at_state so it
        // can use stage-resolved drivers and Sun geometry. Never proxy it.
        AtmModel::Jb2008
        | AtmModel::Jb2008LogQuadratureX4ApproxV1
        | AtmModel::Jb2008LogQuadratureX4ApproxV2
        | AtmModel::Jb2008FittedV7
        | AtmModel::Jb2008FittedV7PartAV3Persistence => f64::NAN,
    }
}

#[inline]
fn density_temperature_from_state(
    state: &[f64],
    jd: f64,
    rotation: &FrameRotation,
    earth_radius: f64,
    atm_model: AtmModel,
    alt_km_precomputed: Option<f64>,
) -> (f64, f64) {
    if atm_model.uses_jb2008_drivers() {
        return (f64::NAN, f64::NAN);
    }
    let rho = density_from_state(
        state,
        jd,
        rotation,
        earth_radius,
        atm_model,
        alt_km_precomputed,
    );
    if rho <= 0.0 {
        return (rho, 0.0);
    }
    let Some(&[state_x, state_y, state_z]) = position3(state) else {
        return (f64::NAN, f64::NAN);
    };

    // Approximate thermodynamic state using the synthetic proxy at the current
    // geocentric location. This keeps Coulomb drag physically bounded without
    // adding another atmosphere model dependency into the RHS hot path.
    let alt_km = alt_km_precomputed.unwrap_or_else(|| {
        (state_x.mul_add(state_x, state_y.mul_add(state_y, state_z * state_z))).sqrt()
            - earth_radius
    });
    let pos_itrs = rotation.to_itrs(&[state_x, state_y, state_z]);
    // RADIANS out, DEGREES in - see the units note on the density path above.
    let (lat_rad, lon_rad, alt) =
        satpy_core::geocentric_spherical_from_itrs(&pos_itrs, earth_radius);
    let (_, proxy_temp_k, _) = synthetic_thermosphere_proxy_eval_impl(
        jd,
        lat_rad.to_degrees(),
        lon_rad.to_degrees(),
        alt.max(alt_km),
    );
    let temperature_k = if proxy_temp_k.is_finite() && proxy_temp_k > 0.0 {
        proxy_temp_k
    } else {
        900.0
    };
    (rho, temperature_k)
}

// Test-only rotation perturbation, in radians about the CIP.
//
// `#[cfg(test)]` so it adds NO production surface — which is one of the things
// having these tests in-crate buys us. Thread-local so parallel tests cannot
// disturb each other. Used by the end-to-end golden's sensitivity proof: a
// pinned state that does not move under a small rotation perturbation is
// value-blind, and a value-blind golden detects nothing.
#[cfg(test)]
thread_local! {
    static TEST_ROTATION_PERTURB_RAD: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// Identity outside tests: the perturbation hook exists only under `cfg(test)`,
/// and this keeps the single call site uniform without a lint exemption.
#[cfg(not(test))]
#[inline]
const fn apply_test_rotation_perturbation(r: FrameRotation) -> FrameRotation {
    r
}

#[cfg(test)]
fn apply_test_rotation_perturbation(mut r: FrameRotation) -> FrameRotation {
    let eps = TEST_ROTATION_PERTURB_RAD.with(Cell::get);
    if eps == 0.0 {
        return r;
    }
    let (sn, cs) = eps.sin_cos();
    let [first_row, second_row, third_row] = r.r;
    let [first_x, first_y, first_z] = first_row;
    let [second_x, second_y, second_z] = second_row;
    let transformed_first = [
        cs * first_x + sn * second_x,
        cs * first_y + sn * second_y,
        cs * first_z + sn * second_z,
    ];
    let transformed_second = [
        -sn * first_x + cs * second_x,
        -sn * first_y + cs * second_y,
        -sn * first_z + cs * second_z,
    ];
    r.r = [transformed_first, transformed_second, third_row];
    r
}

/// Authority rotation at a UTC Julian Day, for tests that call the frame-aware
/// free functions directly. Uses the same sealed path production does.
#[cfg(test)]
fn test_frame_rotation(utc_jd: f64) -> FrameRotation {
    let big = (utc_jd - 0.5).floor() + 0.5;
    let tai_s = satpy_core::frame_time::authority::tai_seconds_from_utc_jd(big, utc_jd - big)
        .expect("test epoch inside the sealed span");
    frame_authority()
        .rotation_at(tai_s)
        .expect("test epoch resolves")
}

#[cfg_attr(feature = "profile-symbols", inline(never))]
fn compute_coulomb_drag(
    state: &[f64],
    jd: f64,
    rotation: &FrameRotation,
    qm_ratio: f64,
    r_obj_m: f64,
    omega_earth: f64,
    atm_model: AtmModel,
    earth_radius: f64,
    alt_km_precomputed: Option<f64>,
) -> [f64; 3] {
    if qm_ratio == 0.0 || r_obj_m <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let Some(&[r_x, r_y, _, v_x, v_y, v_z]) = state6(state) else {
        return [f64::NAN; 3];
    };

    let (rho, temperature_k) = density_temperature_from_state(
        state,
        jd,
        rotation,
        earth_radius,
        atm_model,
        alt_km_precomputed,
    );
    if rho <= 0.0 || temperature_k <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let first_position_m = r_x * KM_TO_M;
    let second_position_m = r_y * KM_TO_M;
    let eastward_velocity_m = v_x * KM_TO_M;
    let northward_velocity_m = v_y * KM_TO_M;
    let vertical_velocity_m = v_z * KM_TO_M;
    let velocity_norm = eastward_velocity_m
        .mul_add(
            eastward_velocity_m,
            northward_velocity_m.mul_add(
                northward_velocity_m,
                vertical_velocity_m * vertical_velocity_m,
            ),
        )
        .sqrt();
    if velocity_norm == 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let flow_x = eastward_velocity_m + omega_earth * second_position_m;
    let flow_y = northward_velocity_m - omega_earth * first_position_m;
    let flow_z = vertical_velocity_m;
    let flow_norm = (flow_x.mul_add(flow_x, flow_y.mul_add(flow_y, flow_z * flow_z))).sqrt();
    if flow_norm == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let inv_flow_norm = 1.0 / flow_norm;
    let dir_x = flow_x * inv_flow_norm;
    let dir_y = flow_y * inv_flow_norm;
    let dir_z = flow_z * inv_flow_norm;

    let avg_mass_i = MEAN_ION_MASS_KG;
    let n_i = (rho / avg_mass_i).max(MIN_NUMBER_DENSITY);
    let thermal_speed = (2.0 * BOLTZMANN_K * temperature_k / avg_mass_i.max(1e-30))
        .max(1e-12)
        .sqrt();

    // SI-form Debye length: λ_D = sqrt(ε₀ k_B T / (n_i e²)). See
    // VACUUM_PERMITTIVITY constant doc for context on the audit fix.
    let debye_length = (VACUUM_PERMITTIVITY * BOLTZMANN_K * temperature_k
        / (n_i * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE))
        .max(1e-24)
        .sqrt();
    let log_argument = debye_length / r_obj_m.max(1e-6);
    let coulomb_log = log_argument.max(MIN_COULOMB_LOG).ln();

    let ratio = velocity_norm / thermal_speed;
    let bracket = ratio.atan() - ratio / (1.0 + ratio * ratio);
    let term = qm_ratio * ELEMENTARY_CHARGE * n_i.sqrt() / velocity_norm.max(1.0);
    let prefactor = term * term;
    let accel_mag = 8.0 * prefactor * bracket * coulomb_log;

    [
        accel_mag * dir_x * M_TO_KM,
        accel_mag * dir_y * M_TO_KM,
        accel_mag * dir_z * M_TO_KM,
    ]
}

/// `LightyearRHS`: computes the right-hand side of the delta-state ODE.
/// Mutable cache for RHS computation (wrapped in `UnsafeCell` for interior mutability).
///
/// NOTE: `GravityCache` is embedded per RHS to eliminate thread-local access overhead.
/// Profiling showed `thread_local!{}.with()` was consuming 85% of execution time.
/// Each Rayon worker owns an independently constructed RHS, so memory footprint is identical to thread-local,
/// but direct pointer access via `UnsafeCell` is ~10x faster than `LocalKey::with()`.
///
/// THAT SIZE IS DERIVED, NOT REMEMBERED, because it is what the embedding
/// decision above is budgeted against. `GravityCacheGeneric<f64>` in
/// `satpy_core::gravity` OWNS two `[[f64; MAX_RECURSIVE_ORDER]]` tables — the
/// `v` and `w` recurrence workspaces — whose ROW COUNT is a slice length while
/// the row LENGTH is fixed at `MAX_RECURSIVE_ORDER` by the array type. The
/// per-instance heap cost is therefore `2 * rows * MAX_RECURSIVE_ORDER * 8 B`.
///
/// `GravityCache::new` takes `rows = MAX_RECURSIVE_ORDER = MAX_ORDER + 3 = 131`,
/// i.e. `2 * 131 * 131 * 8 = 274_576` B. This struct does NOT construct that
/// way. A fill for `order` touches only `order + 2` rows, and this RHS knows its
/// order at construction, so it uses `GravityCache::with_rows` — 7 rows at the
/// campaign's `sph_order = 5`, i.e. `2 * 7 * 131 * 8 = 14_672` B, 5.3% of the
/// full width. Recompute from the count of OWNED tables, the row count actually
/// requested, and `MAX_RECURSIVE_ORDER` whenever any of the three moves.
///
/// The cache also reaches `LegendreCoeffsSimd`'s `pt1` and `pt21_factor`, two
/// more tables of the same shape, but it BORROWS them: they are a
/// `&'static` into one process-wide `OnceLock`, so they cost `274_576` B ONCE for
/// the process rather than per RHS. They were owned per instance until
/// 2026-08-04; sharing them halved this struct's footprint and removed the
/// ~`16_770` divisions `LegendreCoeffsSimd::fixed` ran on every construction.
/// If that field ever goes back to being owned, this figure doubles to
/// `549_152` B and the count above becomes four.
#[derive(Clone)]
#[repr(align(64))]
struct RHSCache {
    cached_tof: f64,
    cached_r_state: [f64; 6],
    cache_valid: bool,
    /// Heap-resident gravity cache (OWNED `v`/`w`, sized in the struct-level
    /// note above) so constructing `LightyearRHS` inside Rayon workers does not
    /// consume worker stack.
    ///
    /// Built by [`RHSCache::with_gravity_rows`] from the RHS's own harmonic
    /// order wherever the order is known; `Default` falls back to the full
    /// width for the test and probe sites that have no order to hand.
    gravity_cache: Box<GravityCache>,

    /// Frame authority segment for the interval containing the current stage,
    /// keyed by its `(j, k)` index. Rebuilt only when the index changes, i.e.
    /// once per `SEGMENT_WIDTH_S` of simulated time rather than per stage.
    ///
    /// DELIBERATELY EXEMPT from `reset_cache`, unlike every other cache in this
    /// struct. A segment is a pure function of its `(j, k)` key and the sealed
    /// tables — no epoch base, no mutable EOP state — so a key match IS a
    /// correct segment and the cache is self-invalidating. Clearing it on reset
    /// would only force a rebuild that reproduces the same bytes. Verified, not
    /// assumed: freshly constructed and reset RHS instances produce
    /// BIT-IDENTICAL propagations.
    /// If `build_segment` ever gains a dependency outside `(j, k)`, this
    /// exemption becomes a stale-cache bug and must be removed.
    cached_segment: Option<FrameSegment>,
    cached_segment_key: (usize, usize),

    /// Resolved rotation for `cached_rotation_tai_s`, memoized because a single
    /// RHS evaluation resolves the rotation at one `t` more than once (gravity
    /// and drag both need it). Exempt from `reset_cache` for the same reason
    /// `cached_segment` is: it is keyed by absolute TAI and is a pure function
    /// of that key, so a key match IS a correct rotation.
    cached_rotation: Option<FrameRotation>,
    cached_rotation_tai_s: f64,

    /// Resolved UTC Julian Day for `cached_driver_utc_jd_tai_s`. Exempt from
    /// `reset_cache` on the same grounds as the two caches above: keyed by
    /// absolute TAI and a pure function of that key.
    cached_driver_utc_jd: f64,
    cached_driver_utc_jd_tai_s: f64,
}

impl RHSCache {
    /// Build with a recurrence workspace sized to `order`, not to `MAX_ORDER`.
    ///
    /// A fill for `order` writes only inside the `(order + 2)^2` square, so
    /// `order + 2` rows is the exact requirement; `GravityCache::with_rows`
    /// clamps the result into the safe range. Everything else matches
    /// [`Default`], which stays at full width for callers with no order.
    ///
    /// Deliberately NOT `Self { gravity_cache: .., ..Self::default() }`: struct
    /// update syntax would construct the full-width cache and immediately drop
    /// it, paying the 274 KB allocation this exists to avoid, on a path that
    /// runs inside Rayon workers.
    fn with_gravity_rows(order: usize) -> Self {
        Self::with_gravity_cache(Box::new(GravityCache::with_rows(order.saturating_add(2))))
    }

    /// Size the workspace from the harmonic authorities an RHS actually holds.
    ///
    /// `packed` is the main authority and `packed_degree1` a truncation of the
    /// same table, so the latter is normally narrower — but it is truncated to
    /// degree 1 from the UNCAPPED authority while `packed` is capped at the
    /// configured `sph_order`, so an `sph_order` below 1 would leave it wider.
    /// Take the max rather than assume the ordering.
    fn sized_for(packed: &PackedGravityCoeffs, degree1: Option<&PackedGravityCoeffs>) -> Self {
        let order = packed
            .max_order()
            .max(degree1.map_or(0, PackedGravityCoeffs::max_order));
        Self::with_gravity_rows(order)
    }

    /// Shared field initialiser; the workspace width is the only thing that
    /// varies between [`Self::with_gravity_rows`] and [`Default`].
    const fn with_gravity_cache(gravity_cache: Box<GravityCache>) -> Self {
        Self {
            cached_tof: -1e308,
            cached_r_state: [0.0; 6],
            cache_valid: false,
            gravity_cache,
            cached_segment: None,
            cached_segment_key: (usize::MAX, usize::MAX),
            cached_rotation: None,
            cached_rotation_tai_s: f64::NAN,
            cached_driver_utc_jd: f64::NAN,
            cached_driver_utc_jd_tai_s: f64::NAN,
        }
    }
}

impl Default for RHSCache {
    fn default() -> Self {
        Self::with_gravity_cache(Box::default())
    }
}

/// The stage-time baselines of one explicit RK step.
///
/// Sized for the widest tableau in use (Vern9's 16 stages). A step with fewer
/// stages fills a prefix and leaves `len` short; a step with MORE would fill
/// this and leave the rest to the unchanged per-call path, so the bound is a
/// capacity limit and never a correctness one.
#[derive(Clone)]
struct StageBaselineTable {
    /// `tof.to_bits()` per filled slot. Bit keys, not tolerances: this table
    /// answers only the exact times it was filled with.
    keys: [u64; MAX_PREFILLED_STAGES],
    states: [[f64; 6]; MAX_PREFILLED_STAGES],
    len: usize,
}

/// Vern9, the widest tableau the Part A propagator runs, has 16 stages.
const MAX_PREFILLED_STAGES: usize = 16;

impl StageBaselineTable {
    const fn empty() -> Self {
        Self {
            keys: [0; MAX_PREFILLED_STAGES],
            states: [[0.0; 6]; MAX_PREFILLED_STAGES],
            len: 0,
        }
    }

    /// The state filled for exactly `key`, if any.
    ///
    /// A linear scan, not a map: `len` is at most 16, the keys are `u64`, and
    /// the whole table is two cache lines' worth of keys sitting next to the
    /// code that just wrote them. A hash would cost more than the scan saves.
    #[inline]
    fn get(&self, key: u64) -> Option<[f64; 6]> {
        self.keys
            .iter()
            .take(self.len)
            .position(|&stored| stored == key)
            .and_then(|slot| self.states.get(slot).copied())
    }
}

/// Right-hand side for the Lightyear delta-state ODE.
///
/// Uses `UnsafeCell` for interior mutability to work with the `ode_solvers::System` trait,
/// which requires `&self` (not `&mut self`).
///
/// # Safety
/// Each RHS instance is only accessed from a single thread during integration.
/// The `ode_solvers::Dopri5` stepper calls `system(&self, ...)` sequentially,
/// never concurrently from multiple threads for the same RHS instance.
/// This makes `UnsafeCell` safe to use here, avoiding `RefCell` runtime borrow checking overhead.
pub struct LightyearRHS {
    // Baseline equinoctial state (constant during one integration segment)
    pub init_equinoc_state: [f64; 6],

    /// The time-invariant half of `equinoc2eci_impl`, derived from
    /// `init_equinoc_state` and rebuilt wherever that is written.
    ///
    /// The baseline-cache miss path converts the SAME six elements at a new `t`
    /// on most evaluations -- 0.85 calls per evaluation on the pinned strict-HF
    /// arc, roughly 5.6% of it -- and three `sqrt`, a reciprocal and a nine-
    /// product rotation in that conversion depend only on the elements. Holding
    /// them here computes them once per propagation instead of once per
    /// evaluation. `None` only where the elements are degenerate, which is the
    /// same condition under which the conversion writes NaN, so the miss path
    /// falls back to the all-in-one call and reproduces that NaN exactly.
    equinoc_baseline: Option<EquinoctialBaseline<f64>>,

    // Time parameters
    pub t0_s: f64, // Initial time offset (seconds)
    pub jd0: f64,  // Initial Julian date

    // Pre-computed inverse for JD calculation (avoids division in hot path)
    inv_sec_per_day: f64,

    /// Continuous-TAI seconds of `jd0`, resolved ONCE at construction.
    ///
    /// `jd0` arrives as a UTC Julian Day in a single binary64 — a frozen part of
    /// `try_new`'s signature. Interpreting it as UTC exactly once here, and then
    /// advancing by elapsed seconds, is what makes integration time continuous
    /// TAI: `t` keeps its meaning as elapsed seconds and never traverses the UTC
    /// discontinuity a leap second introduces.
    ///
    /// `None` when `jd0` cannot resolve under sealed time authority. `try_new`
    /// cannot start returning `Err` for epochs it accepts today without breaking
    /// callers, so scalar gravity carries this to an exact `InvalidTime` failure
    /// instead of fabricating a non-finite epoch.
    tai0_s: Option<f64>,

    // Force configuration
    pub config: std::sync::Arc<ForceConfig>,

    // Opaque capability present only for canonical Part A strict-HF force and
    // independently revalidated loaded gravity/ephemeris/atmosphere/frame
    // identities. Private, with no caller constructor or supplied hashes.
    strict_hf_enclosure_authority: Option<StrictHfEnclosureAuthority>,

    // Spherical-harmonic authority capped to this RHS's configured order.
    pub packed: std::sync::Arc<PackedGravityCoeffs>,
    // Present only when first-order subtraction needs authority-backed terms.
    packed_degree1: Option<std::sync::Arc<PackedGravityCoeffs>>,

    // Force-selection bits whose scalar parameters are valid, hoisted out of
    // the hot RHS loop. Static third-body presence stays in `config.force_flags`.
    active_force_flags: i32,
    baseline_cache_tol: f64,
    gravity_mode: GravityEvalMode,
    dynamic_ephemeris_flags: i32,
    dynamic_ephemeris: Option<std::sync::Arc<AllPrecomputedEphemeris>>,
    thirdbody_simd_packs: [ThirdBodySimdPack; 2],
    thirdbody_simd_pack_count: usize,

    // Atmospheric model resolved once at construction to avoid per-call if-chain in hot path.
    resolved_atm_model: AtmModel,
    // Compiled, hash-validated JB2008 authority resolved once per scalar RHS.
    jb2008_drivers: Option<std::sync::Arc<Jb2008Drivers>>,

    // Mutable cache wrapped in UnsafeCell for interior mutability (single-threaded access only)
    cache: UnsafeCell<RHSCache>,
    eclipse_side: Cell<Option<EclipseSide>>,
    eclipse_error: Cell<Option<EclipseError>>,
    /// First packed-gravity evaluator failure from this propagation. The ODE
    /// trait cannot return it directly, so scalar boundaries consume this exact
    /// latch after the solver stops on its non-finite derivative.
    gravity_error: Cell<Option<GravityError>>,
    /// Last Sun position resolved from the precomputed ephemeris, with the EXACT
    /// `f64` bits of the UTC Julian Day it was resolved at.
    ///
    /// See `dynamic_body_position` for why the key is complete and why it is a
    /// Julian Day rather than an integrator time.
    sun_position_memo: Cell<Option<(u64, [f64; 3])>>,
    /// Most-recent-first UTC Julian Days for the `&self`-only callers of
    /// `driver_utc_jd_at`, keyed on the EXACT bits of continuous TAI seconds.
    ///
    /// See `uncached_driver_utc_jd` for why this exists as a `Cell` beside the
    /// `RHSCache` rather than inside it, and why the key is complete.
    utc_jd_memo: Cell<[Option<(u64, f64)>; UTC_JD_MEMO_WAYS]>,
    /// Baseline ECI state at the EXACT bits of one time-of-flight.
    ///
    /// While an eclipse envelope is active, every RHS evaluation converts the
    /// SAME elements at the SAME `tof` twice: `validate_eclipse_envelope_at_delta`
    /// runs first (integrator.rs, before `compute_internal`) and paid a full
    /// `equinoc2eci_impl`, then the Encke baseline-cache miss arm — 85% of
    /// evaluations at the production tolerance policy — ran `state_at` on the
    /// identical arguments. Sampled on the pinned production arc (model 5),
    /// the two conversions together were 12.26% of arc wall: 6.75 points under
    /// the validator, 4.40 under the miss arm.
    ///
    /// The key is complete: the value is `equinoc_baseline.state_at(tof, 0.0)`,
    /// a pure function of `tof` and fields that only `reset_for_propagation`
    /// writes, and that function clears this memo. A hit therefore returns the
    /// exact bits the conversion would produce, and nothing downstream can
    /// distinguish the memo from recomputation. This memo does NOT replace the
    /// tolerance-keyed baseline cache in `RHSCache`: that one deliberately
    /// serves *stale* baselines within `baseline_cache_tol`, which is a
    /// different (and coarser) contract than exact-bit reuse.
    ///
    /// A `Cell` (unsynchronised) is sound here because this field makes
    /// `LightyearRHS` `!Sync`, so the compiler forbids sharing the RHS across
    /// threads; parallel work constructs one owned RHS per worker. Same for
    /// `eclipse_admit_span` below.
    baseline_exact_memo: Cell<Option<(u64, [f64; 6])>>,
    /// The one slot every [`BaselineCalculator`] this RHS hands out consults,
    /// instead of each keeping a private one that starts cold.
    ///
    /// # Why this is a separate slot from `baseline_exact_memo`
    ///
    /// The two answer the same question and do NOT agree to the bit.
    /// `baseline_state_at_exact` resolves through `state_at_seeded`, whose root
    /// depends on the previous solve's; `BaselineCalculator::get_baseline_state`
    /// calls `equinoc2eci_impl` unseeded. They differ in the last ULP, which is
    /// inside every tolerance either serves and outside the bit contract
    /// `strict_hf_pin` holds this crate to. Merging them would be a bit-mover
    /// carrying a full re-pin bill; keeping them apart makes SHARING free.
    ///
    /// # Why sharing is bit-identical
    ///
    /// `equinoc2eci_impl` is a pure function of `(init_equinoc_state, tof)`,
    /// the key is the exact bits of `tof`, and `init_equinoc_state` is fixed for
    /// as long as this slot lives -- `reset_for_propagation` replaces the
    /// elements and clears the slot in the same breath. So a hit returns exactly
    /// what the miss it replaces would have computed, and no caller can tell the
    /// difference. That is what makes this an empty-bill change: no digest
    /// moves, no accuracy ledger, no re-pin.
    ///
    /// # Why it was worth moving
    ///
    /// `baseline_calculator()` is called at two production sites and mints a
    /// FRESH calculator each time; R11 counted 136 instances serving 1,820
    /// consults on one arc, so each instance started cold and paid a guaranteed
    /// miss. The live hit rate was 27.58% against 61.10% for the same key stream
    /// replayed through a single slot that is not discarded -- 33.52 points of
    /// pure reconstruction loss, with ZERO near-misses at ULP scale (an exact
    /// key and a 1e-9 key gave the same 61.10%, so widening the key was never
    /// the lever; not throwing it away was).
    ///
    /// R11 built this as a thread-local 4-way LRU and dropped it at merge: that
    /// shape paid a TLS access plus a seven-word key compare on every one of the
    /// 1,820 consults, and measured 0.23% against a 0.35% layout floor. Living
    /// on the RHS costs neither -- the calculator already borrows the RHS for
    /// its lifetime, so a hit is the same single `u64` compare it always was.
    baseline_calc_memo: Cell<Option<(u64, [f64; 6])>>,
    /// Baselines for the stage times of the step currently being taken, filled
    /// by [`Self::prefill_stage_baselines`] before the stage loop asks for any
    /// of them.
    ///
    /// The baseline is a pure function of `tof`, and an explicit RK step's
    /// stage times are all known from `(t, h, c)` before its first evaluation,
    /// so the whole set can be resolved where the solves are INDEPENDENT rather
    /// than one at a time inside the serial stage chain. That is the entire
    /// point — see [`EquinoctialBaseline::state_at_seeded_x4`] for why
    /// independence, not width, is what pays.
    ///
    /// Keyed on exact `tof` bits, like [`Self::baseline_exact_memo`], so a hit
    /// is the value that time would have produced in this prefill and nothing
    /// interpolates. Queries that are not stage times — dense output, the
    /// eclipse scan's `state_at`, event legs — miss every key and take the
    /// unchanged path below, which is why this table can only add hits.
    ///
    /// `UnsafeCell` on the same argument as [`Self::cache`]: `LightyearRHS` is
    /// `!Sync` (the `Cell` fields above see to that), so the compiler forbids
    /// sharing one across threads and parallel work constructs one owned RHS per
    /// worker. Borrows taken here are short-lived and never overlap: the
    /// prefill's exclusive borrow ends before the stage loop's shared ones
    /// begin.
    stage_baselines: UnsafeCell<StageBaselineTable>,
    /// The previous converged `F - L` offset of the equinoctial longitude
    /// solve, as a seed for the next one.
    ///
    /// The solve is Halley's method from `F = L`, whose starting error is the
    /// eccentricity: at the pinned arc's `e = 0.025` that is three passes, two
    /// of them paying a `sin_cos`. Seeding from the previous root's offset
    /// starts within `e * dL` instead — measured mean 4.31e-4 against 1.80e-2
    /// — and the arc's 9,026 solves fall from 25,937 passes to 17,084.
    ///
    /// **Offset, not root, and the difference is not cosmetic.** `L` is
    /// `mod2pi`'d; a carried ROOT is a full revolution wrong on every call that
    /// crosses the seam, and those calls then cost *four* passes. The offset is
    /// bounded by `e` and never wraps. See `state_at_seeded`.
    ///
    /// **This is what moves the digests.** The loop exits on the step, so a
    /// different seed converges to a different last-ULP root. Every strict-HF
    /// and rect-loop digest, and the eclipse roots that ride on them, are
    /// re-pinned in the commit that introduced this field.
    ///
    /// Order-dependent where [`Self::baseline_exact_memo`] is not, so it is
    /// reset wherever that memo is. A fresh worker starts cold and pays one
    /// extra pass on its first solve rather than inheriting another worker's
    /// history.
    baseline_warm_offset: Cell<Option<f64>>,
    /// Integrator-time span whose every point is known to resolve to a UTC JD
    /// the Sun ephemeris table admits.
    ///
    /// `eclipse_sun_direction_path_bound` resolves UTC Julian Days for its two
    /// endpoints ONLY to range-check them — the bound it returns is
    /// `rate * |dt|` and never reads the values — and the eclipse scan calls it
    /// at every subdivision, so those two memoized calendar conversions were
    /// the entire cost of the function (3.1% of the model-5 arc wall after the
    /// exact-tof memo landed). Once both endpoints of a query lie inside this
    /// span the verdict is known and the conversions are skipped; the bound
    /// arithmetic is shared by both arms, so the returned bits are identical.
    ///
    /// Soundness of answering for INTERIOR points, including why this does not
    /// assume `taiutc` is monotone, is argued on
    /// [`Self::ADMIT_SPAN_MARGIN_DAYS`]: every recorded endpoint cleared both
    /// table edges by that margin, and the span cap keeps the possible
    /// leap-second excursion far inside it. The span only ever grows within one
    /// propagation and is cleared by `reset_for_propagation`.
    eclipse_admit_span: Cell<Option<(f64, f64)>>,
}

#[cfg(test)]
thread_local! {
    static TEST_RHS_CONSTRUCTIONS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_test_rhs_constructions() {
    TEST_RHS_CONSTRUCTIONS.set(0);
}

#[cfg(test)]
pub(crate) fn test_rhs_constructions() -> usize {
    TEST_RHS_CONSTRUCTIONS.get()
}

/// Entries in [`LightyearRHS::utc_jd_memo`].
///
/// Measured on the pinned 12-hour strict-HF production arc (`atm_model` 5),
/// replaying the real key stream against an LRU of each width. The uncached
/// path runs 28,346 times on that arc — 3.02 per RHS evaluation — and hits:
///
/// | ways | 1 | 2 | 4 | 8 | 16 | 32 |
/// |---|---|---|---|---|---|---|
/// | hit rate | 21.91% | 51.94% | **74.62%** | 76.83% | 77.33% | 77.87% |
///
/// Four is the knee: it takes 96% of everything 32 ways can reach, and every
/// width past it buys under a point. The working set is the handful of
/// endpoints one eclipse scan interval touches, so widening the array only
/// retains times the scan has already walked past.
///
/// # DISPUTED 2026-08-09 — this table did not reproduce, cause unknown
///
/// R25 re-measured the same two quantities and got **29.95% hit rate at four
/// ways and 0.96 uncached calls per RHS evaluation**, against the 74.62% and
/// 3.02 above. Both figures are recorded; neither is retracted, because the
/// disagreement has not been explained and the original was not obviously
/// taken wrong.
///
/// What is known about the gap:
///
/// - The table above is explicitly an `atm_model` **5** measurement and the
///   tree now ships model **6**. That cannot be the whole story: model 6 moved
///   the arc's RHS-evaluation count by a few percent, not by 3x.
/// - The rate is a ratio of eclipse-scan calendar traffic to RHS evaluations,
///   so anything that changed either side moves it. Several levers in that
///   window did (the memo below, the eclipse direction and h-carry work), and
///   no re-measurement was taken across them.
/// - The two runs may not be the same arc shape. The block above names "the
///   pinned 12-hour strict-HF production arc"; R25's number was taken on the
///   census arc.
///
/// Consequence: **do not size anything off either number.** The width choice
/// itself is not at risk — four ways is the knee under both hit rates, and the
/// array costs four `Option<(u64, f64)>` slots — but any claim of the form
/// "this memo is worth N% of the arc" needs a fresh measurement first. Settling
/// it needs a counter on the uncached branch reported against `RHS_EVALS` on
/// one named arc, which the census does not currently carry.
const UTC_JD_MEMO_WAYS: usize = 4;

impl LightyearRHS {
    /// Construct from a single-binary64 UTC Julian Day.
    ///
    /// # Collapsing a calendar instant into `jd0` costs accuracy
    ///
    /// This is a property of the CALLER's input, not a defect in `jd0`. A single
    /// binary64 near 2.46e6 has an ULP of `2^-31 d = 4.023314e-5 s`, so a caller
    /// that already knows an instant more precisely — from a calendar
    /// specification, or as two parts — loses that precision by collapsing it
    /// here. MEASURED cost of the collapse:
    ///
    /// | epoch | floor | at 7000 km |
    /// |---|---|---|
    /// | 2022-08-12T04:25:00 | 1.788139e-5 s | 9.127523 mm |
    /// | 2024-01-01T12:34:56.789 | -1.096725e-5 s | 5.598214 mm |
    /// | 2022-08-12T04:25:30.5 | -6.675720e-6 s | 3.407608 mm |
    /// | any exact UTC midnight | 0 s | 0 mm |
    ///
    /// The midnight rows are the dangerous ones: those JDs are exactly
    /// representable half-integers, so a caller or gate that only ever samples
    /// midnight measures no cost and concludes there is none. That is the same
    /// shape as a leap-second probe landing on an interpolation node — the one
    /// sample where the effect vanishes identically.
    ///
    /// # Part A production is NOT affected
    ///
    /// The sealed event bank is the epoch authority: `conjunction_jd` in the
    /// hash-pinned catalogue is itself a binary64, and no more-precise upstream
    /// representation survives. That value IS the definition of the epoch, so
    /// the Sterbenz split below recovers 100% of its information and production
    /// carries no floor at all.
    ///
    /// **Callers holding calendar or two-part time must use
    /// [`Self::try_new_two_part`]** rather than collapsing first. The cost is
    /// pinned by `try_new_single_f64_jd0_carries_a_measured_frame_floor`.
    ///
    /// # Errors
    ///
    /// Returns an error when atmosphere, ephemeris, frame, or force authority
    /// cannot construct a valid RHS.
    pub fn try_new(
        init_equinoc_state: [f64; 6],
        t0_s: f64,
        jd0: f64,
        config: std::sync::Arc<ForceConfig>,
        packed: std::sync::Arc<PackedGravityCoeffs>,
    ) -> anyhow::Result<Self> {
        // Sterbenz-exact split: recovers every bit `jd0` HAS. It cannot recover
        // what `jd0` never had, which is the floor documented above.
        let (utc_jd1, utc_jd2) = Self::split_jd(jd0);
        Self::try_new_two_part(init_equinoc_state, t0_s, utc_jd1, utc_jd2, config, packed)
    }

    /// Revalidate every loaded strict-HF asset over one requested elapsed arc.
    pub(crate) fn validate_strict_hf_arc(
        &self,
        elapsed_start_s: f64,
        elapsed_end_s: f64,
    ) -> anyhow::Result<()> {
        if self.strict_hf_enclosure_authority.is_none() {
            anyhow::bail!("strict-HF runtime authority is absent");
        }
        let (utc_jd1, utc_jd2) = Self::split_jd(self.jd0);
        validate_arc_coverage(
            &self.config,
            self.dynamic_ephemeris.as_deref(),
            self.jb2008_drivers.as_deref(),
            utc_jd1,
            utc_jd2,
            elapsed_start_s,
            elapsed_end_s,
        )
        .map_err(anyhow::Error::new)
    }

    /// Construct with the anchor epoch as a TWO-PART UTC Julian Day.
    ///
    /// This is the precision-carrying constructor and the one production must
    /// use. A single binary64 near 2.46e6 cannot represent an arbitrary instant
    /// more finely than `2^-31 d = 4.023314e-5 s`, which the Earth-rotation rate
    /// turns into up to 9.13 mm of frame error at 7000 km — 157x the segment
    /// cache's own residual and 13x the 1e-10 element bound the Task 5B routing
    /// REDs assert. Splitting the epoch here keeps that floor out of the chain.
    ///
    /// [`Self::try_new`] delegates here for callers that hold a collapsed `jd0`.
    ///
    /// # Errors
    ///
    /// Returns an error when atmosphere, ephemeris, frame, or force authority
    /// cannot construct a valid RHS.
    pub(crate) fn try_new_two_part(
        init_equinoc_state: [f64; 6],
        t0_s: f64,
        utc_jd1: f64,
        utc_jd2: f64,
        config: std::sync::Arc<ForceConfig>,
        packed: std::sync::Arc<PackedGravityCoeffs>,
    ) -> anyhow::Result<Self> {
        let requested_sph_order = config.sph_order;
        let packed_authority_order = packed.max_order();
        if requested_sph_order > packed_authority_order {
            return Err(anyhow::anyhow!(
                "requested spherical gravity order {requested_sph_order} exceeds packed authority order {packed_authority_order}"
            ));
        }
        let packed_degree1 = if config.subtract_first_order && packed.has_nonzero_degree1_terms() {
            Some(std::sync::Arc::new(
                packed
                    .truncated_to(1)
                    .context("deriving packed degree-one gravity authority")?,
            ))
        } else {
            None
        };
        let packed = if requested_sph_order == packed_authority_order {
            packed
        } else {
            std::sync::Arc::new(packed.truncated_to(requested_sph_order).with_context(|| {
                format!("capping packed gravity authority at order {requested_sph_order}")
            })?)
        };
        // Retained for the public field and for callers that read it; the
        // rotation never derives from this collapsed value.
        let jd0 = utc_jd1 + utc_jd2;
        let force_flags = config.force_flags;
        let dynamic_ephemeris_flags = config.dynamic_ephemeris_flags;
        let dynamic_ephemeris = if dynamic_ephemeris_flags != 0 {
            crate::precomputed_ephem::get_precomputed_ephemeris()
        } else {
            None
        };
        let has_drag_force = (force_flags & ForceFlags::DRAG) != 0;
        let has_thirdbody_force = (force_flags & ForceFlags::THIRDBODY_ALL) != 0;
        let has_lorentz_force = (force_flags & ForceFlags::LORENTZ) != 0;
        let has_coulomb_drag_force = (force_flags & ForceFlags::COULOMB_DRAG) != 0;
        let has_relativity_force = (force_flags & ForceFlags::RELATIVITY) != 0;
        let resolved_atm_model = AtmModel::from_i32(config.atm_model)?;
        if resolved_atm_model.uses_jb2008_drivers() && has_coulomb_drag_force {
            return Err(anyhow::anyhow!(
                "JB2008 exact/approximation modes cannot be combined with Coulomb drag"
            ));
        }
        let jb2008_drivers = resolved_atm_model
            .jb2008_driver_authority()
            .map(Jb2008DriverAuthority::load)
            .transpose()
            .context("loading compiled JB2008 drivers")?;
        let strict_hf_enclosure_authority = issue_for_rhs(
            &config,
            packed.as_ref(),
            dynamic_ephemeris.as_deref(),
            jb2008_drivers.as_deref(),
            utc_jd1,
            utc_jd2,
        )
        .map_err(anyhow::Error::new)
        .context("issuing strict-HF enclosure authority")?;
        let active_force_flags = (force_flags & ForceFlags::THIRDBODY_ALL)
            | (if has_drag_force && config.cd > 0.0 && config.am_ratio > 0.0 {
                ForceFlags::DRAG
            } else {
                0
            })
            | (if effective_scalar_srp(&config) {
                ForceFlags::SRP
            } else {
                0
            })
            | (if has_lorentz_force && config.qm_ratio != 0.0 {
                ForceFlags::LORENTZ
            } else {
                0
            })
            | (if has_coulomb_drag_force && config.qm_ratio != 0.0 && config.r_obj_m > 0.0 {
                ForceFlags::COULOMB_DRAG
            } else {
                0
            })
            | (if has_relativity_force {
                ForceFlags::RELATIVITY
            } else {
                0
            });
        let static_invariant = |flag: i32, inv: Option<BodyInvariants>| {
            if (dynamic_ephemeris_flags & flag) != 0 {
                None
            } else {
                inv
            }
        };
        let sun_inv = static_invariant(ForceFlags::SUN_GRAVITY, config.sun_invariants);
        let moon_inv = static_invariant(ForceFlags::MOON_GRAVITY, config.moon_invariants);
        let jupiter_inv = static_invariant(ForceFlags::JUPITER_GRAVITY, config.jupiter_invariants);
        let venus_inv = static_invariant(ForceFlags::VENUS_GRAVITY, config.venus_invariants);
        let mars_inv = static_invariant(ForceFlags::MARS_GRAVITY, config.mars_invariants);
        let saturn_inv = static_invariant(ForceFlags::SATURN_GRAVITY, config.saturn_invariants);
        let (thirdbody_simd_packs, thirdbody_simd_pack_count) = if has_thirdbody_force {
            let primary = make_thirdbody_pack(
                if (force_flags & ForceFlags::SUN_GRAVITY) != 0 {
                    sun_inv
                } else {
                    None
                },
                if (force_flags & ForceFlags::MOON_GRAVITY) != 0 {
                    moon_inv
                } else {
                    None
                },
                if (force_flags & ForceFlags::JUPITER_GRAVITY) != 0 {
                    jupiter_inv
                } else {
                    None
                },
                if (force_flags & ForceFlags::VENUS_GRAVITY) != 0 {
                    venus_inv
                } else {
                    None
                },
            );
            let secondary = make_thirdbody_pack(
                if (force_flags & ForceFlags::MARS_GRAVITY) != 0 {
                    mars_inv
                } else {
                    None
                },
                if (force_flags & ForceFlags::SATURN_GRAVITY) != 0 {
                    saturn_inv
                } else {
                    None
                },
                None,
                None,
            );
            match (primary.active, secondary.active) {
                (true, true) => ([primary, secondary], 2),
                (true, false) => ([primary, ThirdBodySimdPack::inactive()], 1),
                (false, true) => ([secondary, ThirdBodySimdPack::inactive()], 1),
                (false, false) => (
                    [ThirdBodySimdPack::inactive(), ThirdBodySimdPack::inactive()],
                    0,
                ),
            }
        } else {
            (
                [ThirdBodySimdPack::inactive(), ThirdBodySimdPack::inactive()],
                0usize,
            )
        };
        let baseline_cache_tol = (config.dt_max * 0.01).clamp(1e-3, 0.1);
        let use_analytic_first_order_subtraction =
            config.subtract_first_order && packed_degree1.is_none();
        let gravity_mode = match (
            config.subtract_first_order,
            use_analytic_first_order_subtraction,
        ) {
            (false, _) => GravityEvalMode::Packed,
            (true, true) => GravityEvalMode::AnalyticCentral,
            (true, false) => GravityEvalMode::ExplicitLowOrder,
        };
        // This constructor runs inside Rayon worker paths for HF batch
        // propagation (production enters through `try_new` and
        // `try_new_two_part`; `new` is test-only), so it must not perform I/O.
        let rhs = Self {
            init_equinoc_state,
            equinoc_baseline: EquinoctialBaseline::new(&init_equinoc_state, 6),
            t0_s,
            jd0,
            inv_sec_per_day: 1.0 / SEC_PER_DAY, // Pre-compute to avoid division in hot path
            tai0_s: satpy_core::frame_time::authority::tai_seconds_from_utc_jd(utc_jd1, utc_jd2)
                .ok(),
            config,
            strict_hf_enclosure_authority,
            // Workspace sized to this RHS's own harmonic order, not to
            // `MAX_ORDER`. Listed HERE, out of declaration order, because
            // struct-literal fields evaluate in written order and `packed,`
            // below moves the authority this needs to read.
            cache: UnsafeCell::new(RHSCache::sized_for(&packed, packed_degree1.as_deref())),
            packed,
            packed_degree1,
            active_force_flags,
            baseline_cache_tol,
            gravity_mode,
            dynamic_ephemeris_flags,
            dynamic_ephemeris,
            thirdbody_simd_packs,
            thirdbody_simd_pack_count,
            resolved_atm_model,
            jb2008_drivers,
            eclipse_side: Cell::new(None),
            eclipse_error: Cell::new(None),
            gravity_error: Cell::new(None),
            sun_position_memo: Cell::new(None),
            utc_jd_memo: Cell::new([None; UTC_JD_MEMO_WAYS]),
            baseline_exact_memo: Cell::new(None),
            baseline_calc_memo: Cell::new(None),
            stage_baselines: UnsafeCell::new(StageBaselineTable::empty()),
            baseline_warm_offset: Cell::new(None),
            eclipse_admit_span: Cell::new(None),
        };
        #[cfg(test)]
        TEST_RHS_CONSTRUCTIONS.set(TEST_RHS_CONSTRUCTIONS.get().saturating_add(1));
        Ok(rhs)
    }

    #[cfg(test)]
    pub(crate) fn new(
        init_equinoc_state: [f64; 6],
        t0_s: f64,
        jd0: f64,
        config: std::sync::Arc<ForceConfig>,
        packed: std::sync::Arc<PackedGravityCoeffs>,
    ) -> Self {
        Self::try_new(init_equinoc_state, t0_s, jd0, config, packed)
            .expect("atmosphere model must be validated before RHS construction")
    }

    #[inline]
    /// Spherical-harmonic gravity through the full IAU 2006/2000A rotation.
    ///
    /// Rotates the position into ITRS ONCE, evaluates every mode's harmonics in
    /// the Earth-fixed frame via the `_frame` siblings, and rotates the resulting
    /// acceleration back ONCE. The analytic-subtract term is a GCRS central-body
    /// correction and is therefore applied after the return rotation, not before.
    ///
    /// The legacy z-rotation `_sincos` dispatcher this replaced is retained in
    /// `satpy_core::gravity`, not here; the 4BG oracle and the pinned bench
    /// reach it there.
    fn accumulate_spherical_gravity_frame(
        &self,
        st_pert: &[f64; 6],
        rotation: &FrameRotation,
        r_km_sq: f64,
        r_km: f64,
        cache: &mut RHSCache,
        total_acc: &mut [f64; 3],
    ) -> Result<(), GravityError> {
        let &[st_x, st_y, st_z, ..] = st_pert;
        let pos_itrs = rotation.to_itrs(&[st_x, st_y, st_z]);
        let mut evaluate_packed = |packed: &PackedGravityCoeffs| {
            satpy_core::spherical_gravity_impl_frame_packed(
                &pos_itrs,
                &mut cache.gravity_cache,
                packed,
            )
        };
        let mut analytic_subtract = false;
        let acc_itrs = match self.gravity_mode {
            GravityEvalMode::Packed => evaluate_packed(&self.packed)?,
            GravityEvalMode::AnalyticCentral => {
                analytic_subtract = true;
                evaluate_packed(&self.packed)?
            }
            GravityEvalMode::ExplicitLowOrder => {
                let [full_x, full_y, full_z] = evaluate_packed(&self.packed)?;
                let packed_degree1 = self
                    .packed_degree1
                    .as_deref()
                    .ok_or(GravityError::InvariantViolation)?;
                let [low_x, low_y, low_z] = evaluate_packed(packed_degree1)?;
                [full_x - low_x, full_y - low_y, full_z - low_z]
            }
        };

        let acc_gcrs = rotation.to_gcrs(&acc_itrs);
        accumulate_axes3(total_acc, &acc_gcrs);
        if analytic_subtract {
            // Central-body term, GCRS by construction: applied AFTER the return
            // rotation because it is not an Earth-fixed quantity.
            let mu_inv_r3 = MU / (r_km_sq * r_km);
            mul_add_axes3(total_acc, &[st_x, st_y, st_z], mu_inv_r3);
        }
        Ok(())
    }

    #[cfg_attr(not(feature = "profile-symbols"), inline)]
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn accumulate_thirdbody_simd_prepacked(&self, sat_pos: &[f64], tb_acc: &mut [f64; 3]) {
        let Some(&[position_first, position_second, position_third]) = position3(sat_pos) else {
            return;
        };
        let first_lanes = f64x4::splat(position_first);
        let second_lanes = f64x4::splat(position_second);
        let third_lanes = f64x4::splat(position_third);
        let mut first_sum = f64x4::ZERO;
        let mut second_sum = f64x4::ZERO;
        let mut third_sum = f64x4::ZERO;

        for pack in self
            .thirdbody_simd_packs
            .iter()
            .take(self.thirdbody_simd_pack_count)
        {
            let (first, second, third) = if pack.all_active {
                compute_thirdbody_grav_simd4_lanes_all_active(
                    first_lanes,
                    second_lanes,
                    third_lanes,
                    pack.body_norm_x,
                    pack.body_norm_y,
                    pack.body_norm_z,
                    pack.inv_body_dist,
                    pack.mu_coef,
                )
            } else {
                compute_thirdbody_grav_simd4_lanes_splatted(
                    first_lanes,
                    second_lanes,
                    third_lanes,
                    pack.body_norm_x,
                    pack.body_norm_y,
                    pack.body_norm_z,
                    pack.inv_body_dist,
                    pack.mu_coef,
                    pack.mask,
                )
            };
            first_sum += first;
            second_sum += second;
            third_sum += third;
        }

        let [acc_x, acc_y, acc_z] = tb_acc;
        *acc_x += first_sum.reduce_add();
        *acc_y += second_sum.reduce_add();
        *acc_z += third_sum.reduce_add();
    }

    #[cfg_attr(not(feature = "profile-symbols"), inline)]
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn compute_thirdbody_acc_prepacked(&self, sat_pos: &[f64]) -> [f64; 3] {
        let mut tb_acc = [0.0f64; 3];

        {
            self.accumulate_thirdbody_simd_prepacked(sat_pos, &mut tb_acc);
        }

        tb_acc
    }

    /// Position of `body` from the precomputed ephemeris at UTC Julian Day `jd`,
    /// with the Sun memoized on the exact bits of `jd`.
    ///
    /// # Why the Sun and only the Sun
    ///
    /// Every RHS evaluation asks for the Sun THREE times at one bit-identical
    /// Julian Day, because three different forces need it and each resolves its
    /// own JD seam: drag through `jd_driver`, SRP and third-body gravity through
    /// `jd_ephem`. Those two seams stay separate on purpose (see
    /// `ephemeris_lookup_jd_at`), but today `ephemeris_lookup_jd_at` is a
    /// one-line forward to `driver_utc_jd_at` — the SAME function — so the three
    /// lookups are the same argument to the same pure function, and two of the
    /// three Chebyshev interpolations are pure waste. The Moon is asked for
    /// exactly once per evaluation and gains nothing, so it is not memoized; a
    /// second Moon consumer is the trigger to revisit that, not a rule.
    ///
    /// # Why the key is complete
    ///
    /// `body_position_uncached` reads exactly three things: `body`, `jd`, and
    /// `self.dynamic_ephemeris`. `body` is pinned to `Sun` by the branch. `jd` is
    /// the key, compared on `to_bits` so that no two distinct arguments can
    /// collide and so that a repeated NaN argument reuses the NaN result it
    /// already produced. `dynamic_ephemeris` is an `Arc` assigned once in
    /// `try_new` and never reassigned or mutated afterwards — the tables are
    /// immutable for the lifetime of the RHS. So a hit returns the bits the interpolation would have
    /// returned, and no invalidation hook is required anywhere.
    ///
    /// # Why a Julian Day and NOT an integrator time
    ///
    /// This is the load-bearing part. `reset_for_propagation` moves `t0_s` and
    /// `init_equinoc_state`, which changes the map from integrator time `t` to
    /// `jd` — so a memo keyed on `t`, or on `tai_s`, or on anything reset can
    /// move, would survive a reset and answer with the wrong epoch's Sun. `jd`
    /// is downstream of every such input: the reset changes WHICH `jd` is asked
    /// for, never what the ephemeris returns for a given one. That is why
    /// `reset_for_propagation` deliberately does not clear this cell, and why it
    /// must never be re-keyed onto a time.
    #[cfg_attr(not(feature = "profile-symbols"), inline)]
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn dynamic_body_position(&self, body: EphemerisBody, jd: f64) -> [f64; 3] {
        if !matches!(body, EphemerisBody::Sun) {
            return self.body_position_uncached(body, jd);
        }
        let key = jd.to_bits();
        if let Some((cached_key, cached_position)) = self.sun_position_memo.get() {
            if cached_key == key {
                return cached_position;
            }
        }
        let resolved = self.body_position_uncached(body, jd);
        self.sun_position_memo.set(Some((key, resolved)));
        resolved
    }

    /// The ephemeris interpolation `dynamic_body_position` memoizes.
    ///
    /// Kept separate so the memo has exactly one thing to be a pure function of;
    /// nothing outside `dynamic_body_position` should call this.
    #[cfg_attr(not(feature = "profile-symbols"), inline)]
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn body_position_uncached(&self, body: EphemerisBody, jd: f64) -> [f64; 3] {
        let Some(body_ephem) = self
            .dynamic_ephemeris
            .as_ref()
            .and_then(|ephem| ephem.get(body))
        else {
            // Every production constructor preflights required bodies. Preserve
            // failure as non-finite integration output if an internal caller
            // violates that invariant; never unwind from an ODE/Rayon worker.
            return [f64::NAN; 3];
        };
        let Ok(utc) = UtcJulianDay::new(jd) else {
            return [f64::NAN; 3];
        };
        if let Ok(position) = body_ephem.position_at_part_a_utc_jd(utc) {
            return position;
        }

        // Full absolute arc coverage was checked before RHS construction. Allow
        // only JD recomposition roundoff at an inclusive endpoint; a material
        // escape from the validated arc remains a non-finite solver failure.
        let (start, end) = body_ephem.jd_range();
        let clamped = jd.clamp(start, end);
        let roundoff_days = 2.0 * f64::EPSILON * jd.abs().max(start.abs()).max(end.abs());
        if (jd - clamped).abs() <= roundoff_days {
            let Ok(clamped_utc) = UtcJulianDay::new(clamped) else {
                return [f64::NAN; 3];
            };
            body_ephem
                .position_at_part_a_utc_jd_clamped(clamped_utc)
                .unwrap_or([f64::NAN; 3])
        } else {
            [f64::NAN; 3]
        }
    }

    #[cfg_attr(not(feature = "profile-symbols"), inline)]
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn accumulate_dynamic_thirdbody(&self, sat_pos: &[f64], jd: f64, tb_acc: &mut [f64; 3]) {
        let mut add_body = |body: EphemerisBody, flag: i32, mu: f64| {
            if (self.config.force_flags & flag) == 0 || (self.dynamic_ephemeris_flags & flag) == 0 {
                return;
            }
            let position = self.dynamic_body_position(body, jd);
            if let Some(invariants) = BodyInvariants::precompute(&position, mu) {
                accumulate_thirdbody_grav_precomputed(sat_pos, &invariants, tb_acc);
            }
        };
        add_body(
            EphemerisBody::Sun,
            ForceFlags::SUN_GRAVITY,
            self.config.mu_sun,
        );
        add_body(
            EphemerisBody::Moon,
            ForceFlags::MOON_GRAVITY,
            self.config.mu_moon,
        );
        add_body(
            EphemerisBody::Jupiter,
            ForceFlags::JUPITER_GRAVITY,
            self.config.mu_jupiter,
        );
        add_body(
            EphemerisBody::Venus,
            ForceFlags::VENUS_GRAVITY,
            self.config.mu_venus,
        );
        add_body(
            EphemerisBody::Mars,
            ForceFlags::MARS_GRAVITY,
            self.config.mu_mars,
        );
        add_body(
            EphemerisBody::Saturn,
            ForceFlags::SATURN_GRAVITY,
            self.config.mu_saturn,
        );
    }

    #[inline]
    fn validated_sun_position_at(&self, jd: f64) -> Option<[f64; 3]> {
        let sun_pos = if (self.dynamic_ephemeris_flags & ForceFlags::SUN_GRAVITY) != 0 {
            self.dynamic_body_position(EphemerisBody::Sun, jd)
        } else {
            self.config.sun_pos?
        };
        let dist_sq = sun_pos[0].mul_add(
            sun_pos[0],
            sun_pos[1].mul_add(sun_pos[1], sun_pos[2] * sun_pos[2]),
        );
        if !(dist_sq.is_finite() && dist_sq > 0.0) {
            return None;
        }
        Some(sun_pos)
    }

    /// Split a single-binary64 JD into an exact big part and a remainder.
    ///
    /// `big` is a half-integer below `2^22` and so is exact; the subtraction is
    /// exact by Sterbenz, so `frac` carries every remaining bit of `jd`. This
    /// recovers all the information `jd0` HAS — it cannot recover what `jd0`
    /// never had, namely the `2^-31 d = 4.023313522338867e-05 s` its own
    /// binary64 representation at 2.46e6 already discarded.
    #[inline]
    fn split_jd(jd: f64) -> (f64, f64) {
        let big = (jd - 0.5).floor() + 0.5;
        (big, jd - big)
    }

    /// Continuous-TAI seconds from the sealed span start at integrator time `t`.
    ///
    /// This is the authoritative instant for the Earth-fixed rotation. The other
    /// two seams derive their own scales from it, so the single undifferentiated
    /// scalar the RHS used to share between GMST, the JB drivers and the
    /// ephemeris no longer exists.
    #[cfg_attr(not(feature = "profile-symbols"), inline)]
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn tai_seconds_at(&self, t: f64) -> Option<f64> {
        // Elapsed seconds added to a TAI origin resolved once at construction.
        // NOT `jd0 + t/86400` re-interpreted: that divides elapsed time by a UTC
        // day, which is 86400 s on ordinary days and 86401 s on a leap day, so
        // the epoch gains a whole second every time it crosses one.
        let tai_s = self.tai0_s? + t;
        tai_s.is_finite().then_some(tai_s)
    }

    /// Resolve the GCRS->ITRS rotation at integrator time `t`.
    ///
    /// The segment is rebuilt only when the authority's `(j, k)` index changes —
    /// once per `SEGMENT_WIDTH_S` of simulated time, not per RK stage. Within a
    /// segment the centred rotation and its exact analytic derivative reuse one
    /// `sin_cos`; the sealed exact chain remains outside the hot path.
    ///
    /// `(j, k)` is a pure function of absolute TAI, so the rotation cannot depend
    /// on how work was cut into arcs or how many workers ran: the W1/W8 identity
    /// survives this cache.
    #[cfg_attr(not(feature = "profile-symbols"), inline)]
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn frame_rotation_at_checked(
        &self,
        t: f64,
        cache: &mut RHSCache,
    ) -> Result<FrameRotation, GravityError> {
        let tai_s = self.tai_seconds_at(t).ok_or(GravityError::InvalidTime)?;
        // One RHS evaluation asks for the rotation at the SAME `t` more than
        // once -- the spherical-harmonic block needs it to reach ITRS, and the
        // drag block needs it for the co-rotating atmosphere. Each ask re-ran
        // `rotation_at`: one `sin_cos` plus ~39 FMA, recomputed to produce bytes
        // that were already sitting in this cache.
        //
        // Keyed on `tai_s`, not on `t`. `t` is measured from `t0_s`, which
        // `reset_for_propagation` moves, so a `t` key would go stale across a
        // segment restart. `tai_s` is absolute, and the rotation is a pure
        // function of it -- the same argument that lets `cached_segment` skip
        // `reset_cache`, and the reason this needs no invalidation either.
        //
        // Invalid time returns above before this cache can reuse a prior frame.
        if let Some(rotation) = cache.cached_rotation {
            #[expect(
                clippy::float_cmp,
                reason = "absolute TAI is the exact cache key; approximate equality would reuse the wrong frame"
            )]
            if cache.cached_rotation_tai_s == tai_s {
                return Ok(apply_test_rotation_perturbation(rotation));
            }
        }
        let authority = frame_authority();
        let key = authority
            .segment_index(tai_s)
            .map_err(|_| GravityError::InvalidTime)?;
        if cache.cached_segment.is_none() || cache.cached_segment_key != key {
            let segment = authority
                .segment_cached(key.0, key.1)
                .map_err(|_| GravityError::InvalidTime)?;
            cache.cached_segment = Some(segment);
            cache.cached_segment_key = key;
        }
        let segment = cache
            .cached_segment
            .as_ref()
            .ok_or(GravityError::InvalidTime)?;
        let rotation = segment.rotation_at(tai_s);
        cache.cached_rotation = Some(rotation);
        cache.cached_rotation_tai_s = tai_s;
        // The test perturbation is applied to the memoized value on every
        // return rather than stored, so a test that changes it between calls
        // still sees its own change.
        Ok(apply_test_rotation_perturbation(rotation))
    }

    /// Julian Day handed to the precomputed-ephemeris lookup at integrator
    /// time `t`, **in the scale that table declares**.
    ///
    /// # The scale is a property of the TABLE, not of the physics
    ///
    /// `data/ephemeris/manifest.json` declares `"epoch_scale": "utc"`,
    /// `"interpolation": "linear_on_utc_jd_grid"`, and
    /// `"source_api_time_construction": "astropy.time.Time(jd, format='jd',
    /// scale='utc')"`. Each sample therefore holds a body position at a
    /// UTC-LABELLED instant, so recovering the position at physical instant X
    /// requires indexing with X's UTC JD.
    ///
    /// Task 5B-2 originally fed this a TT JD, on the reasoning that ephemerides
    /// are TDB-argument objects. That reasoning is correct for a DE Chebyshev
    /// kernel and WRONG for this table, and it cost ~69.184 s of index error —
    /// about 70 km of Moon position and 2062 km of Sun position.
    ///
    /// MEASURED against `nd_target_oracle_gen`'s sealed JPL Horizons fixture
    /// (`time_type: UT`, DE441), at the four fixture epochs that land on exact
    /// nodes of both tables:
    ///
    /// | epoch JD | \|sun.bin - Horizons\| | \|moon.bin - Horizons\| |
    /// |---|---|---|
    /// | 2458849.5 | 2.262 km | 6.725 km |
    /// | 2460679.5 | 1.525 km | 4.037 km |
    /// | 2460860.5 | 4.973 km | 6.481 km |
    /// | 2462867.5 | 3.161 km | 4.395 km |
    ///
    /// All eight inside the fixture's preregistered 10 km bound, against a
    /// 2062 km / 70.6 km signature had the tables been TT-labelled. Two bodies,
    /// two independent discriminators, 4018 days of span.
    ///
    /// # Those residuals are NODE-ONLY. Do not read them as an accuracy budget.
    ///
    /// All four fixture epochs land on EXACT grid nodes — index 0, 14640, 16088
    /// and 32144 (= `n_samples - 1`) for the Moon, and 0, 1830, 2011, 4018 for
    /// the Sun, verified from the sealed headers. Two-point linear interpolation
    /// is exact at a node, so the fixture measures interpolation error as
    /// identically ZERO. The 1.5-6.7 km residuals are model difference between
    /// the analytic packs and DE441 AT NODES, nothing more.
    ///
    /// Mid-interval error is `(h^2/8)|r_ddot|`, recomputed here from the sealed
    /// samples themselves by centred second difference rather than from an
    /// orbital model:
    ///
    /// | body | `h` | max `\|r_ddot\|` | mid-interval error |
    /// |---|---|---|---|
    /// | Moon | 10800 s | 3.139593e-6 km/s^2 | **45.8 km** |
    /// | Sun | 86400 s | 6.170440e-6 km/s^2 | **5757.8 km** |
    ///
    /// So this fix is CORRECT AND CHEAP, not the dominant error term. Ranked
    /// honestly: for the Moon the 70.6 km scale error is ~1.5x the 45.8 km
    /// interpolation error; for the Sun the 2062 km scale error is SMALLER than
    /// the 5757.8 km interpolation error. An earlier framing of this fix as
    /// "the leading error term, an order of magnitude above everything else"
    /// was wrong and is retracted. The fixture remains a valid instrument for
    /// the SCALE question precisely because a 69.184 s offset moves you OFF a
    /// node — which is why it detects the TT signature at all.
    ///
    /// # Why this was invisible
    ///
    /// `precomputed_ephem::get_position` takes a bare `f64` and names no time
    /// scale anywhere in that module; the scale lives only in the manifest. The
    /// one Sun path that stayed CORRECT through this whole episode is
    /// `jb2008_density_at_state`, and it stayed correct because it goes through
    /// `UtcJulianDay::new` — the scale is in its TYPE, so a TT value could not
    /// be passed silently. The two paths carrying a bare `f64` both went wrong.
    /// That is the argument for making the scale structural rather than
    /// documentary.
    ///
    /// **If the tables are ever regenerated on a TDB grid — which Task 6's
    /// DE440s kernel is, its manifest declaring `"kernel_argument": "TDB
    /// seconds past J2000"` — this must change with them.**
    /// `ephemeris_lookup_scale_matches_the_table_manifest` reads the manifest
    /// and fails if this function and the declared scale disagree, so the
    /// requirement flips automatically instead of silently going stale.
    ///
    /// `cache` is the memo described on `driver_utc_jd_at`. `None` means "no
    /// `RHSCache` available here" and returns a bit-identical value; the
    /// eclipse helpers use it because they hold only `&self`.
    ///
    /// It does NOT mean "unmemoized", and it is NOT the cold path this comment
    /// used to call it: that branch runs 3.02 times per RHS derivative and is
    /// 60% of every entry into `taiutc` (both figures DISPUTED — R25 measured
    /// 0.96 and 42%; see [`UTC_JD_MEMO_WAYS`]. The branch is hot at either).
    /// It has its own memo — see `uncached_driver_utc_jd`, which is what
    /// `None` reaches.
    #[inline]
    fn ephemeris_lookup_jd_at(&self, t: f64, cache: Option<&mut RHSCache>) -> f64 {
        // UTC, because the table is UTC-indexed. See above.
        self.driver_utc_jd_at(t, cache)
    }

    /// UTC Julian Day handed to the JB2008 driver lookup at integrator time `t`.
    ///
    /// Scope decision S10: integration time is TAI, so the lookup must convert
    /// back to UTC or it runs `TAI - UTC` late.
    ///
    /// This goes through the sealed `taiutc` rather than subtracting
    /// `delta_at / 86400`, and the difference is not cosmetic. `taiutc` makes a
    /// leap day 86401 seconds long, so the leap second 2016-12-31T23:59:60
    /// resolves to a day fraction of `86400/86401` and floors to UTC MJD 57753.
    /// Naive subtraction puts that same instant at exactly 57754.0 — the start of
    /// the NEXT day — and would select the wrong driver record for the one second
    /// in ~18 months when UTC labelling is non-monotonic.
    ///
    /// # Memoized, because every RHS evaluation asks for this twice
    ///
    /// `ephemeris_lookup_jd_at` forwards here, so the two seams at the top of
    /// `compute_internal_*` resolve the SAME conversion at the SAME `t` on every
    /// call. That conversion is not cheap: `taiutc` runs the sealed calendar
    /// path (`jd2cal` plus `dat`), which a hot-loop instruction profile puts at
    /// roughly 30% of the whole derivative -- more than the spherical-harmonic
    /// gravity it exists to support.
    ///
    /// The seams stay separate on purpose (see `ephemeris_lookup_jd_at`): they
    /// are the hook for giving the ephemeris and the drag driver different time
    /// scales. Collapsing the two call sites into one would delete that hook, so
    /// the cache goes HERE instead -- both seams keep their own names and call
    /// sites, and only the shared conversion underneath is reused.
    ///
    /// Keyed on continuous TAI rather than on `t`, because `t` is measured from
    /// `t0_s` and `reset_for_propagation` moves it. The result is a pure
    /// function of that key, so no invalidation is required and the value is
    /// bit-identical to recomputing.
    ///
    /// # The memo is passed in, not reached through `UnsafeCell`
    ///
    /// This used to take `&self` and open its own `unsafe { &mut *cache.get() }`
    /// borrow, justified only by a prose claim that "both call sites resolve
    /// their seams BEFORE taking their own cache borrow". There were never two
    /// non-test call sites; there are six, and the claim was the ONLY thing
    /// standing between this borrow and the long-lived `&mut RHSCache` held
    /// across `compute_internal_generic` -- i.e. between the code and UB by
    /// aliasing. Taking the borrow as an argument, the shape
    /// `frame_rotation_at_checked` already uses, hands that obligation to the
    /// borrow checker: an overlapping borrow is now a compile error rather than
    /// a comment someone has to keep true.
    ///
    /// `None` is for callers that hold only `&self` and cannot produce the
    /// borrow — the eclipse geometry helpers. Because the conversion is a pure
    /// function of `tai_s`, taking a different memo to it returns exactly the
    /// same bits.
    ///
    /// **This comment used to call those helpers cold and price the branch at
    /// "one extra `taiutc`". Both were wrong, and that error hid a ~4.4% arc
    /// cost.** Measured on the pinned production arc, the `None` branch runs
    /// 28,346 times against 9,383 RHS evaluations — 3.02 per derivative — and
    /// is 60% of every entry into the calendar path in the propagation, more
    /// than this cached seam. It now has its own memo,
    /// [`Self::uncached_driver_utc_jd`], which carries the numbers. Those two
    /// figures are DISPUTED (R25: 0.96 and 42%) — see [`UTC_JD_MEMO_WAYS`].
    /// The retraction this paragraph records is unaffected: the branch is hot
    /// under every measurement taken, and "cold, one extra `taiutc`" was wrong
    /// by a wide margin either way.
    #[inline]
    fn driver_utc_jd_at(&self, t: f64, cache: Option<&mut RHSCache>) -> f64 {
        let Some(tai_s) = self.tai_seconds_at(t) else {
            return f64::NAN;
        };
        let Some(cache) = cache else {
            return self.uncached_driver_utc_jd(tai_s);
        };
        #[expect(
            clippy::float_cmp,
            reason = "absolute TAI is the exact cache key; approximate equality would reuse the wrong UTC epoch"
        )]
        if cache.cached_driver_utc_jd_tai_s == tai_s {
            return cache.cached_driver_utc_jd;
        }
        let resolved = Self::driver_utc_jd_from_tai_seconds(tai_s);
        cache.cached_driver_utc_jd_tai_s = tai_s;
        cache.cached_driver_utc_jd = resolved;
        resolved
    }

    /// `driver_utc_jd_at` for the callers that hold only `&self`.
    ///
    /// # Why this is not the cold path its callers were assumed to be
    ///
    /// The block above hands `None` to "the cold eclipse geometry helpers" and
    /// prices that at "one extra `taiutc`". A sampling profile of the pinned
    /// 12-hour production arc says the helpers are the single largest consumer
    /// of the calendar path in the program: `eclipse_sun_direction_path_bound`
    /// alone reaches 4.29% of arc wall through `taiutc`, `eclipse::endpoint`
    /// another 1.22%, against 2.63% for the RHS's own cached seam.
    ///
    /// Counted on the same arc: this branch runs 28,346 times against 9,383 RHS
    /// evaluations — 3.02 per derivative, and 60% of every entry into `taiutc`
    /// in the propagation. It is the hot one.
    ///
    /// **DISPUTED 2026-08-09.** R25 re-measured the share and got **42%**, not
    /// 60%, alongside 0.96 calls per derivative rather than 3.02. Both are
    /// recorded and neither is retracted; see the disputed-figures note on
    /// [`UTC_JD_MEMO_WAYS`] for what is and is not known about the gap. The
    /// qualitative claim this paragraph exists to make — that this branch is
    /// hot, not cold, and is the largest single consumer of the calendar path —
    /// survives at either number, and that is the only thing the memo below is
    /// justified by. The percentages themselves should not be quoted onward
    /// until one arc is measured end to end.
    ///
    /// The reason is structural, not incidental. Eclipse detection scans an
    /// interval by asking for geometry at a walk of times, and consecutive
    /// intervals share endpoints — `path_bound(t_a, t_b)` converts both ends,
    /// then the next interval converts one of them again. So the redundancy is
    /// a repeated argument, which is what a memo is for.
    ///
    /// # Why the key is complete
    ///
    /// [`Self::driver_utc_jd_from_tai_seconds`] is an associated function: it
    /// takes `tai_s` and nothing else, and reads no field of `self` and no
    /// global. Its argument therefore IS its key, in full, and no invalidation
    /// hook can be required because there is nothing else for the result to
    /// depend on. That is a stronger argument than `sun_position_memo` needs
    /// (which additionally rests on `dynamic_ephemeris` being immutable), and
    /// it holds across `reset_for_propagation`: the key is absolute TAI, not
    /// integrator time `t`, so moving `t0_s` cannot make a stored entry wrong.
    ///
    /// Compared on exact bits, so a hit returns the bits the call would have
    /// returned. Nothing here is a tolerance, and nearby instants are never
    /// conflated — which matters because `taiutc` is deliberately non-monotonic
    /// across a leap second, so two times a microsecond apart may belong to
    /// different UTC days. Non-finite `tai_s` never arrives: `tai_seconds_at`
    /// filters it, and the caller returns `NaN` before reaching here.
    ///
    /// # Why a `Cell` here rather than the `RHSCache`
    ///
    /// These callers hold `&self` and cannot produce the `&mut RHSCache` the
    /// cached seam takes. Reaching into the `UnsafeCell` from `&self` is
    /// exactly the aliasing hazard the block above describes removing, and
    /// re-introducing it here would alias the long-lived `&mut RHSCache` held
    /// across `compute_internal_generic`. A `Cell` needs no `unsafe` at all,
    /// and `LightyearRHS` already carries four of them for this same reason.
    fn uncached_driver_utc_jd(&self, tai_s: f64) -> f64 {
        let key = tai_s.to_bits();
        let mut slots = self.utc_jd_memo.get();
        if let Some(index) = slots
            .iter()
            .position(|entry| matches!(entry, Some((stored, _)) if *stored == key))
        {
            if let Some((_, value)) = slots.get(index).copied().flatten() {
                // Promote the hit to the front. The range is bounded by the
                // `position` that produced it, and `get_mut` is the accessor
                // that says so without panicking.
                if let Some(window) = slots.get_mut(..=index) {
                    window.rotate_right(1);
                }
                self.utc_jd_memo.set(slots);
                return value;
            }
        }
        let resolved = Self::driver_utc_jd_from_tai_seconds(tai_s);
        // Insert at the front and drop the least recently used entry off the
        // end, which is what `rotate_right` moves into position 0.
        slots.rotate_right(1);
        if let Some(first) = slots.first_mut() {
            *first = Some((key, resolved));
        }
        self.utc_jd_memo.set(slots);
        resolved
    }

    /// The conversion the memo above memoizes: continuous TAI seconds to a UTC
    /// Julian Day through the sealed `taiutc` calendar path.
    #[inline]
    fn driver_utc_jd_from_tai_seconds(tai_s: f64) -> f64 {
        let (tai1, tai2) = satpy_core::frame_time::authority::tai_jd_from_seconds(tai_s);
        let (status, utc1, utc2) = satpy_core::frame_time::timescale::taiutc(tai1, tai2);
        if status < 0 {
            f64::NAN
        } else {
            utc1 + utc2
        }
    }

    /// Evaluate configured scalar drag density at one exact RK stage.
    ///
    /// JB2008 has its own UTC drivers and stage-resolved Sun geometry; all
    /// legacy models retain their existing scalar path.
    #[cfg_attr(not(feature = "profile-symbols"), inline)]
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn density_at_state(
        &self,
        state: &[f64],
        jd: f64,
        rotation: &FrameRotation,
        alt_km_precomputed: Option<f64>,
    ) -> f64 {
        match self.resolved_atm_model {
            AtmModel::Jb2008 => {
                self.jb2008_density_at_state(state, jd, rotation, Jb2008Profile::Exact)
            }
            AtmModel::Jb2008LogQuadratureX4ApproxV1 => {
                self.jb2008_density_at_state(state, jd, rotation, Jb2008Profile::ApproxV1)
            }
            AtmModel::Jb2008LogQuadratureX4ApproxV2 => {
                self.jb2008_density_at_state(state, jd, rotation, Jb2008Profile::ApproxV2)
            }
            // Model 8 keeps model 7's fitted density kernel and differs only in
            // driver authority (see `AtmModel::jb2008_driver_authority`), which
            // is resolved elsewhere, not here. Same body is intentional.
            AtmModel::Jb2008FittedV7 | AtmModel::Jb2008FittedV7PartAV3Persistence => {
                self.jb2008_density_at_state(state, jd, rotation, Jb2008Profile::FittedV7)
            }
            model => density_from_state(
                state,
                jd,
                rotation,
                self.config.earth_radius,
                model,
                alt_km_precomputed,
            ),
        }
    }

    /// Evaluation altitude above which JB2008's own extrapolation is not used as
    /// a density, in metres.
    ///
    /// AIAA G-003C-2010, *Guide to Reference and Standard Atmosphere Models*,
    /// gives 2500 km as the upper bound of the Jacchia family — J71, CIRA-72,
    /// JB2006 and JB2008 alike — and states that above it the density is assumed
    /// constant. JB2008's own validation corpus (Bowman et al., AIAA/AAS
    /// 2008-6438) is tighter still, 175–1000 km, which is where the drag data it
    /// was fitted to exist.
    const JB2008_EXTRAPOLATION_CEILING_M: f64 = 2_500.0 * KM_TO_M;

    /// Density returned in place of the model above `JB2008_EXTRAPOLATION_CEILING_M`,
    /// in kg/m³, when the model would return more.
    ///
    /// # Why a ceiling is needed at all
    ///
    /// Left alone, the kernel extrapolates without bound and its diffusion
    /// profile flattens into a plateau: at the sealed 2003 driver set it returns
    /// 7.720e-17 at 2500 km, 4.070e-17 at 3000 km, 4.009e-18 at 35,000 km and
    /// 4.030e-18 at the 41,378 km ceiling of Part A's transfer apogees. It is not
    /// even monotone — 5.453e-18 at 100,000 km is HIGHER than at 35,000 km, which
    /// no atmosphere does. Exospheric hydrogen at 3–6 Earth radii is roughly
    /// 1e-20 to 1e-19 kg/m³, so the plateau overestimates by one to two orders of
    /// magnitude, and every one of those orders lands in drag.
    ///
    /// # Why 1e-19 and not the two obvious alternatives
    ///
    /// - **Unbounded extrapolation** — what this replaces. Uses the model far
    ///   outside anything it was fitted to, and is non-monotone there.
    /// - **The AIAA constant-above-2500 convention**, i.e. evaluate at 2500 km
    ///   and hold that value. Read literally it is WORSE than doing nothing here:
    ///   it would pin 7.720e-17 all the way to apogee, 19x the plateau the code
    ///   already produces and ~800x the exospheric truth. The convention is
    ///   written for models that decay to a floor by 2500 km, not for one asked
    ///   about 41,378 km.
    /// - **1e-19, the upper end of the exospheric hydrogen literature.** Chosen
    ///   deliberately at the top of that range rather than the middle: it is
    ///   conservative in the drag-OVERestimate direction relative to truth, so
    ///   the change can only reduce a drag term that was already too large, and
    ///   it still sits far under the plateau it replaces.
    ///
    /// Applied as a `min`, not an assignment, so it is a ceiling and not a floor:
    /// if drivers ever push the model below it, the model wins.
    ///
    /// # The ceiling is C0-discontinuous, and that is priced
    ///
    /// Density steps by 772x at the crossing (7.720e-17 down to 1e-19), so the
    /// drag term the integrator sees is not continuous in altitude there. Price
    /// it: at the compiled `am_ratio` 1.948 m²/kg and `cd` 2.2, and at the
    /// fastest a bound orbit can cross 2500 km (escape speed there is 9.48
    /// km/s), the whole step is 1.5e-8 m/s². Local gravity at that radius is
    /// 5.06 m/s², so the discontinuity is 2.9e-9 of it — three times under the
    /// 1e-8 step tolerance. The controller cannot resolve it, so no rejection
    /// cascade forms at the crossing. Smoothing it would buy nothing and would
    /// move the sealed bits.
    const JB2008_EXOSPHERIC_DENSITY_CEILING_KG_M3: f64 = 1.0e-19;

    /// JB2008 density using UTC drivers plus current Earth-centred ICRS Sun.
    #[cfg_attr(not(feature = "profile-symbols"), inline)]
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn jb2008_density_at_state(
        &self,
        state: &[f64],
        jd: f64,
        rotation: &FrameRotation,
        profile: Jb2008Profile,
    ) -> f64 {
        crate::probe::scalar_add(&crate::probe::JB_ADAPTER_CALLS);

        let Some(drivers) = self.jb2008_drivers.as_ref() else {
            return f64::NAN;
        };
        let Ok(utc_jd) = UtcJulianDay::new(jd) else {
            return f64::NAN;
        };
        let Ok(modified_julian_day) = utc_jd.to_utc_mjd() else {
            return f64::NAN;
        };
        let Ok(driver) = drivers.lookup_utc_mjd(modified_julian_day) else {
            return f64::NAN;
        };

        // Preflight forces this catalogue-backed path even when caller gave a
        // static Sun override. Coordinates are Earth-centred ICRS kilometres.
        let sun_gcrs = self.dynamic_body_position(EphemerisBody::Sun, jd);

        // ONE Earth-fixed reduction feeds EVERY geometric argument below.
        //
        // Both the satellite and the Sun are rotated by the same `rotation`, so
        // the five geometric inputs — altitude, satellite latitude and right
        // ascension, solar declination and right ascension — are all measured
        // about the SAME pole. That consistency is the whole point; see the
        // frame note on `geodetic_altitude_km` for why a partial move is worse
        // than none.
        let Some(&[position_x, position_y, position_z]) = position3(state) else {
            return f64::NAN;
        };
        let pos_itrs = rotation.to_itrs(&[position_x, position_y, position_z]);
        let sun_itrs = rotation.to_itrs(&sun_gcrs);
        let sun_r = sun_itrs[0]
            .mul_add(
                sun_itrs[0],
                sun_itrs[1].mul_add(sun_itrs[1], sun_itrs[2] * sun_itrs[2]),
            )
            .sqrt();
        let sat_r = pos_itrs[0]
            .mul_add(
                pos_itrs[0],
                pos_itrs[1].mul_add(pos_itrs[1], pos_itrs[2] * pos_itrs[2]),
            )
            .sqrt();
        // Above the extrapolation ceiling the model does not get the last word
        // anyway, so do not run it.
        //
        // The `min` at the bottom of this function replaces the model's answer
        // with `JB2008_EXOSPHERIC_DENSITY_CEILING_KG_M3` whenever the model
        // returns more, and `examples/exospheric_ceiling_sweep` measures that it
        // ALWAYS returns more up there: 9,666,900 samples across every one of
        // the 10,741 driver days the compiled table covers (MJD 50,454-61,194,
        // two solar cycles) crossed with altitude and geometry, floor
        // 2.351e-18 kg/m^3 against the 1e-19 ceiling — 23.5x, on the exact and
        // the fitted profile alike. `jb2008_exospheric_ceiling_floor_survives_
        // the_whole_driver_table` keeps that premise from going stale.
        //
        // # Why this sits HERE and not at the top of the function
        //
        // Returning the ceiling early is only bit-identical where the flown
        // path would have returned a DENSITY. Where it would have returned NaN
        // the skip must too, because NaN is how this adapter fails closed: it
        // propagates into drag and stops the solver, and turning that into a
        // finite 1e-19 would let a run that should have failed finish quietly.
        //
        // That is not hypothetical. **303 of the 10,741 driver days (2.8%)
        // carry a non-positive solar index**, and the kernel refuses every one
        // of them with `NonPositiveSolarIndex`. An earlier revision of this
        // skip sat above the driver lookup and would have answered 1e-19 on all
        // of them. The sweep that "proved" it safe had swallowed those rows:
        // it took `unwrap_or(NAN)`, and NaN loses every `<`, so refusals never
        // reached the minimum it reported.
        //
        // So the skip sits after every gate the flown path applies — drivers
        // present, UTC resolvable, MJD resolvable, driver row found, position
        // readable — and replicates the only kernel precondition that can still
        // fire up here. The refusal census is exact: 272,700 refused samples is
        // 303 days x 900 grid points, so on this table `NonPositiveSolarIndex`
        // is the ONLY error above the ceiling. `AltitudeOutOfRange` cannot fire
        // (its bound is 90 km), `AngleOutOfRange` cannot (both angles arrive
        // from a clamped `asin`), and `NonFiniteInput` is covered by the
        // finiteness check below, since every remaining kernel input is a
        // finite-preserving function of these values.
        {
            // SPHERICAL altitude, and that direction is the safe one: geodetic
            // altitude subtracts a local radius that never exceeds the
            // equatorial one, so geodetic >= spherical and a spherical altitude
            // past the ceiling guarantees the geodetic altitude computed just
            // below is past it too. Points between the two take the full path.
            let above_ceiling =
                (sat_r - self.config.earth_radius) * KM_TO_M > Self::JB2008_EXTRAPOLATION_CEILING_M;
            let indices = [
                driver.f10,
                driver.f10b,
                driver.s10,
                driver.s10b,
                driver.m10,
                driver.m10b,
                driver.y10,
                driver.y10b,
            ];
            let kernel_would_accept = indices.iter().all(|index| *index > 0.0)
                && f64::from(driver.dtcval).is_finite()
                && sun_r > 0.0
                && sun_itrs.iter().all(|value| value.is_finite())
                && pos_itrs.iter().all(|value| value.is_finite());
            if above_ceiling && kernel_would_accept {
                return Self::JB2008_EXOSPHERIC_DENSITY_CEILING_KG_M3;
            }
        }

        // ELLIPSOIDAL altitude above the WGS84 reference surface. Bound here
        // rather than inline in the literal below because it is also what
        // decides whether the model is inside its validity range at all; the
        // frame and surface reasoning is on the field it feeds.
        let sat_altitude_m = geodetic_altitude_km(
            pos_itrs[0],
            pos_itrs[1],
            pos_itrs[2],
            self.config.earth_radius,
        ) * KM_TO_M;
        let input = Jb2008Input {
            // Preserve UTC JD -> MJD conversion exactly through driver types.
            mjd_utc: modified_julian_day.as_f64(),
            // ITRS, like every other angle here. "Declination" and "geocentric
            // latitude" name the same angle about the same axis — the Earth's
            // pole, i.e. the CIP — and the GCRS equator is the MEAN EQUATOR OF
            // J2000, which by 2022 is tilted from it by ~1.25e-3 rad.
            sun_declination_rad: (sun_itrs[2] / sun_r).clamp(-1.0, 1.0).asin(),
            // The kernel consumes only the satellite's hour angle relative to
            // the Sun (`jb_rs::jb2008`), so that is what this hands it. It used
            // to be handed the two right ascensions separately and subtract
            // them, which cost two `atan2` here and two whole-turn
            // normalizations there.
            //
            // `atan2` of the cross and dot of the two equatorial projections IS
            // that difference, in one call: for `u = (pos_x, pos_y)` and
            // `v = (sun_x, sun_y)`, `atan2(u x v, u . v)` is the signed angle
            // from `v` to `u` in `[-π, π]`. Being a difference, it is exactly
            // invariant under a common z-rotation — ITRS right ascension is
            // measured from the prime meridian rather than the equinox, and the
            // whole-turn offset cancels — but only to FIRST order under the full
            // IAU 2006/2000A chain. The residual is 6.65e-4 rad, i.e. 9.1
            // seconds of local solar time, at the sealed 2022-08-12 epoch. That
            // residual is a property of the frame chain and is unchanged by
            // computing the difference in one step instead of two.
            hour_angle_rad: pos_itrs[1]
                .mul_add(sun_itrs[0], -(pos_itrs[0] * sun_itrs[1]))
                .atan2(pos_itrs[0].mul_add(sun_itrs[0], pos_itrs[1] * sun_itrs[1])),
            // GEOCENTRIC, and deliberately so. Bowman's JB2008 defines SAT(2)
            // as "Geocentric Latitude of Position" in the original Fortran
            // header (Bowman et al., AIAA/AAS 2008-6438). The sealed Orekit
            // fixture's `satellite_geodetic_latitude_rad_as_satLat` column
            // records Orekit's own deviation from the model, not the model's
            // contract, so do NOT "fix" this to geodetic — that would move
            // production off the model spec and onto a third party's departure
            // from it. Worth 0.181 deg / 1.001x in density either way.
            //
            // GEOCENTRIC and ITRS are independent choices and both are
            // deliberate: geocentric-vs-geodetic is about the SURFACE the angle
            // is referred to, ITRS-vs-GCRS about the AXIS. Orekit's column names
            // (`..._as_satLat`, `..._as_sunDecli`, `..._as_sunRA`) show it
            // reduces all four in the body frame, so the frame is common ground;
            // only the surface is disputed.
            sat_geocentric_lat_rad: (pos_itrs[2] / sat_r).clamp(-1.0, 1.0).asin(),
            // ELLIPSOIDAL altitude above the WGS84 reference surface, which is
            // what JB2008 means by "Height of Position" and what the sealed
            // Orekit oracle declares as
            // `satellite_ellipsoidal_altitude_m_as_satAlt`.
            //
            // This was `sat_r - config.earth_radius` — a SPHERICAL altitude,
            // low by `a*f*sin^2(lat)` up to 21.385 km, so density came out high
            // on every step: 1.4216x at 400 km / 60 deg and 1.7630x at Part A's
            // 200 km minimum perigee. Gated by
            // `tests/jb2008_adapter_altitude_gate.rs`.
            //
            // `a` is `config.earth_radius`, whose documented purpose is
            // "geometry that reduces a position to an altitude" — NOT
            // `GRAVITY_REFERENCE_RADIUS_KM`, which is the DIR-R6 gravity
            // reference and differs by 54 cm on purpose.
            //
            // Reduced from the ITRS position, NOT from GCRS. `|r|` and the
            // equatorial-plane angle are invariant under a pure z-rotation, but
            // `to_itrs` is the FULL IAU 2006/2000A chain — precession-nutation
            // and polar motion tilt the axis, so `z` is not invariant. The
            // WGS84 ellipsoid is definitionally Earth-fixed, so reducing a GCRS
            // position flattens about the wrong pole. Worth ~50 m of altitude,
            // i.e. ~0.09% of density, on top of the 42% the spherical form cost.
            sat_altitude_m,
            f10: driver.f10,
            f10b: driver.f10b,
            s10: driver.s10,
            s10b: driver.s10b,
            m10: driver.m10,
            m10b: driver.m10b,
            y10: driver.y10,
            y10b: driver.y10b,
            dst_temperature_correction_k: f64::from(driver.dtcval),
        };
        crate::probe::scalar_add(&crate::probe::JB_KERNEL_CALLS);
        let density = match profile {
            Jb2008Profile::Exact => jb2008_density(input),
            Jb2008Profile::ApproxV1 => jb2008_density_logquad_x4_approx_v1(input),
            Jb2008Profile::ApproxV2 => jb2008_density_logquad_x4_approx_v2(input),
            Jb2008Profile::FittedV7 => jb2008_density_fitted_v7(input),
        };
        let rho = match density {
            Ok(rho) if rho.is_finite() => rho,
            Ok(_) | Err(_) => return f64::NAN,
        };
        // Outside the model's validity range the model does not get the last
        // word. See the two constants for the range, the ceiling, and the
        // alternatives rejected.
        if sat_altitude_m > Self::JB2008_EXTRAPOLATION_CEILING_M {
            rho.min(Self::JB2008_EXOSPHERIC_DENSITY_CEILING_KG_M3)
        } else {
            rho
        }
    }

    /// The whole drag block: the density evaluation and the drag acceleration it
    /// feeds, returned rather than accumulated.
    ///
    /// # This is a codegen boundary, and it is worth 3.7% of the arc
    ///
    /// Nothing here moved. The same statements run in the same order on the same
    /// values; the only change is that they live behind a function signature
    /// instead of inside `compute_internal_generic_unlatched`'s body, and that
    /// the acceleration comes back as a value the caller accumulates rather than
    /// being accumulated in place. It is BIT-IDENTICAL, which is exactly what a
    /// change that only rebrackets existing statements should be.
    ///
    /// That alone is **-3.7% of the production arc**, measured on the arc, 38/40
    /// and 40/40 rounds negative in two independent rotations against a 0.11%
    /// two-build null. See the commit that introduced it.
    ///
    /// # What it is NOT, and the two experiments that say so
    ///
    /// **Not an instruction-count saving.** `__text` moves by 20 bytes across the
    /// whole binary — five instructions on a 58 M-instruction propagation. There
    /// is nothing here for an instruction count to explain.
    ///
    /// **Not outlining.** `#[inline(always)]` reproduces the win in full
    /// (-3.54%), so the boundary is not paying for itself by keeping the kernel
    /// out of the caller. `#[inline(never)]` keeps only -1.49%, so a real call is
    /// WORSE than this and better than the original — do not "simplify" this to
    /// either attribute expecting the same number.
    ///
    /// What is left is scheduling and register allocation in
    /// `compute_internal_generic_unlatched`, which inlines the JB2008 kernel, the
    /// spherical-harmonic block, the third-body block and the eclipse SRP
    /// geometry into one body. That is the remaining explanation and it is NOT
    /// separately measured here; the two experiments above are what rules the
    /// other two out.
    ///
    /// Keep the `Option`. Returning the acceleration is what lets the density
    /// branch and the drag branch resolve inside this body rather than in the
    /// caller's.
    #[inline]
    fn drag_acceleration(
        &self,
        st_pert: &[f64; 6],
        t: f64,
        jd_driver: f64,
        alt_km: f64,
        cache: &mut RHSCache,
    ) -> Result<Option<[f64; 3]>, GravityError> {
        if (self.active_force_flags & ForceFlags::DRAG) == 0 {
            return Ok(None);
        }
        if self.resolved_atm_model == AtmModel::None {
            return Ok(None);
        }
        let rotation = self.frame_rotation_at_checked(t, cache)?;
        let density = self.density_at_state(st_pert, jd_driver, &rotation, Some(alt_km));

        if density.is_nan() || density > 0.0 {
            Ok(Some(compute_drag(
                st_pert,
                density,
                self.config.am_ratio,
                self.config.cd,
                &rotation,
            )))
        } else {
            Ok(None)
        }
    }

    #[cfg_attr(not(feature = "profile-symbols"), inline)]
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn compute_thirdbody_acc(&self, sat_pos: &[f64], jd: f64) -> [f64; 3] {
        let mut tb_acc = self.compute_thirdbody_acc_prepacked(sat_pos);
        self.accumulate_dynamic_thirdbody(sat_pos, jd, &mut tb_acc);
        tb_acc
    }

    #[cfg_attr(not(feature = "profile-symbols"), inline)]
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn accumulate_thirdbody(&self, sat_pos: &[f64], jd: f64, total_acc: &mut [f64; 3]) {
        if (self.active_force_flags & ForceFlags::THIRDBODY_ALL) == 0 {
            return;
        }
        accumulate_axes3(total_acc, &self.compute_thirdbody_acc(sat_pos, jd));
    }

    /// Generic internal compute function using interior mutability (works with &self).
    /// This preserves the full configurable-force path and serves as the parity baseline.
    ///
    /// # Errors
    ///
    /// Returns the exact gravity or frame-authority error produced by an active
    /// force evaluator, and latches the first such error for the ODE adapter.
    #[inline]
    pub fn compute_internal_generic(
        &self,
        delta: &[f64; 6],
        t: f64,
    ) -> Result<[f64; 6], GravityError> {
        let result = self.compute_internal_generic_unlatched(delta, t);
        if let Err(error) = result {
            self.record_gravity_error(error);
        }
        result
    }

    #[inline]
    fn compute_internal_generic_unlatched(
        &self,
        delta: &[f64; 6],
        t: f64,
    ) -> Result<[f64; 6], GravityError> {
        // Task 5B-2: each consumer takes its OWN time scale from the seam that
        // names it, instead of sharing one undifferentiated scalar.
        //
        // The seams do NOT agree bit-for-bit with the naive
        // `jd0 + t * inv_sec_per_day` that used to be computed here and handed
        // to everything. Note the operation: `inv_sec_per_day` is the
        // PRE-ROUNDED reciprocal `1.0 / SEC_PER_DAY` (see the constructor), so
        // this is a multiply and not a divide by 86400, and the two are not the
        // same in binary64. The multiply is what was measured. Measured over
        // one campaign-faithful hybrid batch by comparing `jd.to_bits()` against
        // `jd_driver.to_bits()`: 232,545 of the first 112,000,000 evaluations
        // differ, 0.208%, one call in 482.
        //
        // Only ONE divergence had its magnitude recorded, the first: 4.023e-5 s.
        // That equals `2^-31 d`, which is one ULP of an f64 near 2.46e6, so that
        // one observation is a one-ULP disagreement. The other 232,544
        // magnitudes were not sampled, so treat it as a single observation and
        // not as a bound or a typical value.
        //
        // The MECHANISM is not established here, only the disagreement. Both
        // quantities are ordinary single f64: `driver_utc_jd_at` ends in
        // `utc1 + utc2` (see its body), so this is not a two-part value meeting
        // a one-part one. They are two different arithmetic routes to the same
        // instant — direct scaling of `t` against `taiutc`'s calendar path — and
        // they DISAGREE on 0.208% of calls. How large those disagreements are is
        // unmeasured apart from the one sample above, and why they fall on those
        // calls and not the other 99.8% was not investigated. Do not restate the
        // single one-ULP sample as a property of the set.
        //
        // A comment here asserted the two were bit-identical. They are not. The
        // naive form is now built only where it is consumed.
        // SAFETY: this is the ONE `&mut RHSCache` alive in this function, and it
        // stays alive to the end of it -- the two seams below and every
        // `frame_rotation_at_checked` call reborrow THIS binding rather than
        // opening a second one, so the borrow checker, not a comment, is what
        // rules out aliasing `&mut`.
        //
        // Soundness across threads rests on `LightyearRHS` being `!Sync`: it
        // holds an `UnsafeCell<RHSCache>` and there is deliberately NO
        // `unsafe impl Sync for LightyearRHS` anywhere in this crate. `&self`
        // therefore cannot cross a thread boundary at all, so "single-threaded
        // during integration" is a fact the compiler enforces and not a
        // convention the callers observe. DO NOT add `unsafe impl Sync` to make
        // a `rayon` borrow compile: it would look locally harmless and would
        // silently invalidate every other `unsafe` block in this file at once,
        // with no other code change. Parallelism here is per-RHS-instance,
        // never shared-`&`.
        let cache = unsafe { &mut *self.cache.get() };

        // Explicit reborrows: `Option<&mut _>` construction does not reborrow
        // implicitly, and `cache` must stay usable below.
        let jd_ephem = self.ephemeris_lookup_jd_at(t, Some(&mut *cache));
        let jd_driver = self.driver_utc_jd_at(t, Some(&mut *cache));
        let tof = t - self.t0_s;

        // Get baseline state via equinoc2eci (use cache if possible). The
        // tolerance is the current eps-derived policy set by
        // `adapt_cache_policy_for_eps`.
        let baseline_stale = baseline_cache_is_stale(
            cache.cache_valid,
            tof,
            cache.cached_tof,
            self.baseline_cache_tol,
        );
        let r_state = if baseline_stale {
            crate::probe::tag_add(&crate::probe::BASELINE_MISS, crate::probe::current_tag());
            // `state_at` is the second half of `equinoc2eci_impl` and nothing
            // else -- see `EquinoctialBaseline` -- so this is the same
            // arithmetic in the same order, with the element-only half hoisted
            // out of the per-evaluation path. Degenerate elements leave the
            // baseline `None` and take the all-in-one call, which is the code
            // that decides to write NaN.
            debug_assert_eq!(
                self.equinoc_baseline.is_some(),
                EquinoctialBaseline::new(&self.init_equinoc_state, 6).is_some(),
                "equinoc_baseline must track init_equinoc_state"
            );
            // While an eclipse envelope is active the validator has already
            // resolved this exact `tof` a moment ago (integrator.rs runs it
            // before `compute_internal`), so this is usually a memo HIT. The
            // tolerance-keyed cache bookkeeping below is unchanged: this arm
            // still decides staleness by `baseline_cache_tol`, and the memo
            // only removes the duplicated conversion, never the policy.
            let state = self.baseline_state_at_exact(tof);
            cache.cached_tof = tof;
            cache.cached_r_state = state;
            cache.cache_valid = true;
            state
        } else {
            crate::probe::tag_add(&crate::probe::BASELINE_HIT, crate::probe::current_tag());
            cache.cached_r_state
        };

        // Perturbed state = baseline + delta
        let st_pert = [
            r_state[0] + delta[0],
            r_state[1] + delta[1],
            r_state[2] + delta[2],
            r_state[3] + delta[3],
            r_state[4] + delta[4],
            r_state[5] + delta[5],
        ];

        // Precompute altitude once for drag pre-check and density (P0+P6)
        let r_km_sq = st_pert[0].mul_add(
            st_pert[0],
            st_pert[1].mul_add(st_pert[1], st_pert[2] * st_pert[2]),
        );
        let r_km = r_km_sq.sqrt();
        let alt_km = r_km - self.config.earth_radius;

        // Compute perturbation accelerations
        let mut total_acc = [0.0; 3];

        // 1. Spherical harmonics (using embedded cache - eliminates thread-local overhead)
        // The constructor capped this immutable pack once to the requested order.
        if self.packed.max_order() > 0 {
            // Task 5B-2: the Earth-fixed rotation is the full IAU 2006/2000A
            // chain from the sealed authority, not `R3(GMST1982)`. The old
            // rotation omitted bias-precession-nutation, the equation of the
            // origins and polar motion — ~4.5e-3 in matrix elements, ~31.6 km
            // at 7000 km.
            // Frame authority failure remains an exact `InvalidTime` rather than
            // reaching the packed evaluator as a non-finite rotated position.
            let rotation = self.frame_rotation_at_checked(t, cache)?;
            self.accumulate_spherical_gravity_frame(
                &st_pert,
                &rotation,
                r_km_sq,
                r_km,
                cache,
                &mut total_acc,
            )?;
        }

        // 2. Dust forces. Every force below is evaluated on EVERY call; the
        // density/SRP/third-body sub-cycling that used to skip calls here is
        // gone (see `adapt_cache_policy_for_eps`).
        // Drag (flag = 1).
        if let Some(drag_acc) = self.drag_acceleration(&st_pert, t, jd_driver, alt_km, cache)? {
            accumulate_axes3(&mut total_acc, &drag_acc);
        }

        // SRP (flag = 2). Evaluated every call.
        if (self.active_force_flags & ForceFlags::SRP) != 0 {
            // Task 5B-2: SRP must read the Sun at the SAME epoch third-body
            // gravity does. Passing raw `jd` here while the ephemeris seam
            // supplied `jd_ephem` had two forces sampling one body 69.184 s
            // apart — an inconsistency introduced by the routing itself.
            if let Some(sun_pos) = self.validated_sun_position_at(jd_ephem) {
                if self.config.cr > 0.0 && self.config.am_ratio > 0.0 {
                    let srp_acc = self.eclipse_side.get().map_or_else(
                        || {
                            self.record_eclipse_error(EclipseError::UninitializedSide);
                            [f64::NAN; 3]
                        },
                        |side| {
                            compute_srp_with_precomputed(
                                &st_pert,
                                &sun_pos,
                                self.config.p_sun,
                                self.config.cr,
                                self.config.am_ratio,
                                side,
                            )
                        },
                    );
                    if !srp_acc.iter().all(|value| value.is_finite()) {
                        self.record_eclipse_error(EclipseError::Geometry);
                    }
                    accumulate_axes3(&mut total_acc, &srp_acc);
                }
            } else {
                self.record_eclipse_error(EclipseError::Geometry);
                accumulate_axes3(&mut total_acc, &[f64::NAN; 3]);
            }
        }

        self.accumulate_thirdbody(&st_pert[0..3], jd_ephem, &mut total_acc);

        if (self.active_force_flags & ForceFlags::RELATIVITY) != 0 {
            let rel_acc = compute_relativity(&st_pert);
            accumulate_axes3(&mut total_acc, &rel_acc);
        }

        if (self.active_force_flags & ForceFlags::LORENTZ) != 0 {
            let rotation = self.frame_rotation_at_checked(t, cache)?;
            let lorentz_acc = compute_lorentz_frame(&st_pert, &rotation, self.config.qm_ratio);
            accumulate_axes3(&mut total_acc, &lorentz_acc);
        }

        if (self.active_force_flags & ForceFlags::COULOMB_DRAG) != 0 {
            // The ONLY consumer of the naive Julian Day; see the seam note at
            // the top of this function. Built here rather than there so the
            // function does not open with a time scale that reads like the
            // master clock and is used once, in a branch the campaign force set
            // never enables.
            let jd = self.jd0 + t * self.inv_sec_per_day;
            let coulomb_rotation = self.frame_rotation_at_checked(t, cache)?;
            let coulomb_acc = compute_coulomb_drag(
                &st_pert,
                jd,
                &coulomb_rotation,
                self.config.qm_ratio,
                self.config.r_obj_m,
                self.config.omega_earth,
                self.resolved_atm_model,
                self.config.earth_radius,
                Some(alt_km),
            );
            accumulate_axes3(&mut total_acc, &coulomb_acc);
        }

        // Keplerian correction via Battin's f(q) — avoids catastrophic cancellation
        // in the naive μ·(r_base/|r_base|³ - r_pert/|r_pert|³) subtraction.
        let r_base = [r_state[0], r_state[1], r_state[2]];
        let delta_r = [delta[0], delta[1], delta[2]];
        battin_encke_gravity_correction(&r_base, &delta_r, r_km_sq, r_km, &mut total_acc);

        // Assemble RHS: dxdt = [delta_velocity; total_acceleration]
        Ok([
            delta[3],     // dx/dt = dvx
            delta[4],     // dy/dt = dvy
            delta[5],     // dz/dt = dvz
            total_acc[0], // dvx/dt = ax
            total_acc[1], // dvy/dt = ay
            total_acc[2], // dvz/dt = az
        ])
    }

    /// Compute the derivative through the single generic force path.
    ///
    /// # Errors
    ///
    /// Returns the exact gravity or frame-authority error produced by the
    /// generic force path, while latching its first failure for the ODE adapter.
    #[inline]
    pub fn compute_internal(&self, delta: &[f64; 6], t: f64) -> Result<[f64; 6], GravityError> {
        self.compute_internal_generic(delta, t)
    }

    /// Invalidate cached baseline/ephemeris state after a segment restart.
    ///
    /// Takes `&mut self` deliberately. This used to take `&self` and reach the
    /// cache through `UnsafeCell::get()`, for the single reason that
    /// `reset_for_propagation` wanted to call it -- and `reset_for_propagation`
    /// already holds `&mut self`. Both real call sites (that method and
    /// `integrator.rs`'s reusable-segment setup) own the RHS outright, so the
    /// exclusive borrow costs nothing and `UnsafeCell::get_mut` is safe:
    /// exclusivity is proved by the borrow checker rather than asserted in
    /// prose.
    pub fn reset_cache(&mut self) {
        let cache = self.cache.get_mut();
        cache.cache_valid = false;
        cache.cached_tof = -1e308;
        cache.cached_r_state = [0.0; 6];
        // A segment restart invalidates the gravity recurrence workspace.
        cache.gravity_cache.reset();
        self.clear_gravity_error();
    }

    /// Adapt the RHS caching strategy to the integration tolerance.
    ///
    /// This sets the one policy knob the RHS still has: how stale the cached
    /// Encke baseline may be. Force sub-cycling used to be the other knob, and
    /// is gone — see `reset_cache` for what remains cached.
    pub fn adapt_cache_policy_for_eps(&mut self, eps: f64) {
        // Baseline cache: scale with eps to ensure baseline staleness doesn't
        // dominate the error budget at any tolerance.
        // At v ≈ 7.5 km/s, cache_tol seconds → v·cache_tol km position drift.
        // Acceleration error ≈ 3μ/r⁴ · drift ≈ 3.8e-6 · drift [km/s²].
        // Keep this < eps/100 for safety margin:
        //   cache_tol = eps / (100 · 3.8e-6) ≈ eps · 2.6e3 [seconds], clamped.
        //
        // ASSIGNED, not `.min()`-ed. A `.min()` here is a monotone ratchet: the
        // tolerance would depend on the sequence of eps values this object has
        // been adapted to, so the same RHS adapted to 1e-11 and then to 1e-8
        // would not match one adapted to 1e-8 directly. `baseline_cache_tol` is
        // a pure function of `eps` and nothing else. This is not a behaviour
        // change: the ratchet was instrumented across the whole in-tree suite
        // (lightyear_odeint_rs, dust_estimates_rs, nd_pipeline) and clamped
        // zero times, because every call site adapts a given RHS to one eps.
        self.baseline_cache_tol = (eps * 2.6e3).clamp(1e-9, 0.1);
    }

    /// Prepare this RHS for a fresh propagation segment.
    ///
    /// This updates the baseline initial state/time and always invalidates all
    /// internal caches so no state leaks across independent propagations.
    pub fn reset_for_propagation(&mut self, init_equinoc_state: [f64; 6], t0_s: f64) {
        self.init_equinoc_state = init_equinoc_state;
        self.equinoc_baseline = EquinoctialBaseline::new(&init_equinoc_state, 6);
        self.t0_s = t0_s;
        self.eclipse_side.set(None);
        self.eclipse_error.set(None);
        // The memo's value depends on the elements assigned above, so a stale
        // entry would be WRONG here, not merely cold.
        self.baseline_exact_memo.set(None);
        // Same reason, and it is what makes the shared slot bit-identical: the
        // key is a `tof` and the elements that turn a `tof` into a state were
        // just replaced, so an entry that survived here would be WRONG.
        self.baseline_calc_memo.set(None);
        // Same reason, and more urgently: these entries are keyed on `tof`, and
        // the elements that turn a `tof` into a state were just replaced.
        //
        // SAFETY: `LightyearRHS` is `!Sync`, so this `&mut self` is the only
        // live reference to the RHS and nothing else can hold a borrow of the
        // table. See the field's own note.
        unsafe { *self.stage_baselines.get() = StageBaselineTable::empty() };
        // Not wrong if kept -- a seed is only ever a starting point, and the
        // loop converges from any of them -- but it must be cleared anyway, and
        // for a stronger reason than the memo's. This seed is a function of the
        // sequence of previous calls, so leaving it set would make one
        // propagation's arithmetic depend on which propagation ran before it in
        // this object. Cleared here, the seed chain begins at each
        // propagation's first solve and the arc is a pure function of its own
        // inputs, as `strict_hf_pin` requires.
        self.baseline_warm_offset.set(None);
        // Entries here would still be VALID -- `tai0_s` and the ephemeris do
        // not move -- but this function promises no state leaks across
        // propagations, and that promise is worth more than one warm span.
        self.eclipse_admit_span.set(None);
        self.reset_cache();
        // `sun_position_memo` is deliberately NOT cleared here: it is keyed on a
        // Julian Day, and a reset changes which Julian Day gets asked for, not
        // what the immutable ephemeris returns for one. See the "Why a Julian
        // Day and NOT an integrator time" note on `dynamic_body_position` before
        // adding a clear, and before ever re-keying that memo onto a time.
    }

    #[inline]
    pub(crate) fn set_eclipse_side(&self, side: EclipseSide) {
        self.eclipse_side.set(Some(side));
        self.eclipse_error.set(None);
    }

    #[inline]
    pub(crate) fn record_eclipse_error(&self, error: EclipseError) {
        if self.eclipse_error.get().is_none() {
            self.eclipse_error.set(Some(error));
        }
    }

    #[inline]
    pub(crate) fn take_eclipse_error(&self) -> Option<EclipseError> {
        self.eclipse_error.take()
    }

    #[inline]
    fn record_gravity_error(&self, error: GravityError) {
        if self.gravity_error.get().is_none() {
            self.gravity_error.set(Some(error));
        }
    }

    /// Clear a prior gravity evaluator failure before beginning a new solver run.
    #[inline]
    pub(crate) fn clear_gravity_error(&self) {
        self.gravity_error.set(None);
    }

    /// Consume the first exact packed-gravity error observed in this solver run.
    #[inline]
    pub(crate) fn take_gravity_error(&self) -> Option<GravityError> {
        self.gravity_error.take()
    }

    /// The baseline ECI state at exactly `tof`, through
    /// [`Self::baseline_exact_memo`] and seeded from
    /// [`Self::baseline_warm_offset`].
    ///
    /// # NOT a pure function of `(tof, elements)`, and that is deliberate
    ///
    /// This USED to be bit-identical to `equinoc2eci_impl(&self
    /// .init_equinoc_state, 6, tof, 0.0, ..)`. It is not any more. The
    /// longitude solve is seeded from the previous solve's root, its loop exits
    /// on the step rather than on a residual, and so the root it returns
    /// depends on which `tof` was asked for before this one. Two call orders
    /// over the same set of times return roots that agree to well inside the
    /// 1e-12 step tolerance and disagree in the last ULP.
    ///
    /// What survives, and is what every caller actually needs:
    ///
    /// * **One propagation is reproducible.** `reset_for_propagation` clears
    ///   the seed as well as the memo, and an integration issues its `tof`
    ///   values in a fixed order, so an arc is a pure function of its own
    ///   inputs. `strict_hf_pin` is the standing check.
    /// * **A memo hit is indistinguishable from the miss that filled it.**
    ///   Weaker than the old "indistinguishable from recomputation": a fresh
    ///   solve at that same `tof` later in the sequence would be seeded
    ///   differently and could differ in the last ULP.
    /// * **The degenerate fallback is unchanged** -- no baseline means no seed
    ///   to carry, so that arm is still exactly `equinoc2eci_impl`.
    /// * **Non-convergence still propagates.** `state_at_seeded` writes NaN
    ///   into `out[0]`, the memo stores and replays exactly that, and the seed
    ///   goes back to `None` so the next call starts where an unseeded loop
    ///   would.
    fn baseline_state_at_exact(&self, tof: f64) -> [f64; 6] {
        let key = tof.to_bits();
        if let Some((stored, state)) = self.baseline_exact_memo.get() {
            if stored == key {
                return state;
            }
        }
        // SAFETY: `LightyearRHS` is `!Sync` so no other thread can hold a
        // reference, and this borrow ends with the statement -- the prefill's
        // exclusive borrow is taken between steps, never while this runs.
        if let Some(state) = unsafe { &*self.stage_baselines.get() }.get(key) {
            crate::probe::tag_add(
                &crate::probe::BASELINE_STAGE_HIT,
                crate::probe::current_tag(),
            );
            return state;
        }
        let mut state = [0.0; 6];
        match self.equinoc_baseline {
            Some(baseline) => {
                // The returned offset is stored whatever it is, `None`
                // included: a non-convergent solve has no root to seed from,
                // and the next call must start where the unseeded loop starts
                // rather than from the last root before the failure.
                let offset =
                    baseline.state_at_seeded(tof, 0.0, self.baseline_warm_offset.get(), &mut state);
                self.baseline_warm_offset.set(offset);
            }
            None => equinoc2eci_impl(&self.init_equinoc_state, 6, tof, 0.0, &mut state),
        }
        self.baseline_exact_memo.set(Some((key, state)));
        state
    }

    /// Resolve the baselines for one explicit RK step's stage times, before the
    /// stage loop asks for any of them.
    ///
    /// `nodes` are the tableau's abscissas `c`, so stage `i` is evaluated at
    /// `t + c[i] * h` and its baseline argument is `t + c[i] * h - t0_s`. The
    /// integrator knows all of them before its first evaluation of the step,
    /// which is the fact this whole path rests on.
    ///
    /// # Why prefilling is a speed change and not just a reordering
    ///
    /// Resolved one at a time inside the stage loop, the solves form a single
    /// dependency chain: each seeds from the last, and each is three dependent
    /// divisions plus a `sin_cos`. Resolved here they are four independent
    /// chains at a time and the machine can overlap them —
    /// [`EquinoctialBaseline::state_at_seeded_x4`] carries the measurement.
    ///
    /// # Determinism
    ///
    /// Packs run in ascending stage index and each seeds from the previous
    /// pack's last lane, so the seed chain is a fixed function of `(t, h,
    /// nodes)` and the incoming seed. Nothing here reads a clock, an atomic, or
    /// another thread: the RHS is `!Sync` and a parallel run constructs one per
    /// worker, so an arc's prefill order is the order its own steps happen in.
    ///
    /// # Rejected steps
    ///
    /// A rejected step's entries are simply never asked for, and the next
    /// prefill overwrites them. That is the whole cost of a rejection here, and
    /// the arc's reject fraction is a few percent.
    ///
    /// A degenerate element set (no `equinoc_baseline`) fills nothing and every
    /// query falls through to the unchanged all-in-one path, which is the code
    /// that decides to write NaN.
    pub(crate) fn prefill_stage_baselines(&self, t: f64, h: f64, nodes: &[f64]) {
        let Some(baseline) = self.equinoc_baseline else {
            // SAFETY: see `baseline_state_at_exact`; `!Sync`, borrow ends here.
            unsafe { &mut *self.stage_baselines.get() }.len = 0;
            return;
        };
        let count = nodes.len().min(MAX_PREFILLED_STAGES);
        // SAFETY: `LightyearRHS` is `!Sync`, so nothing else can hold a
        // reference to this table, and the integrator calls this between steps
        // -- outside any stage-loop read. The borrow ends with this function.
        let table = unsafe { &mut *self.stage_baselines.get() };
        table.len = 0;
        let mut seed = self.baseline_warm_offset.get();
        let mut filled = 0_usize;
        let mut pack_times = [0.0_f64; 4];
        let mut pack_states = [[0.0_f64; 6]; 4];
        for chunk in nodes.get(..count).unwrap_or(&[]).chunks(4) {
            let width = chunk.len();
            for (slot, &node) in chunk.iter().enumerate() {
                let Some(time) = pack_times.get_mut(slot) else {
                    continue;
                };
                // `t + node * h`, NOT `node.mul_add(h, t)`. The key is the
                // exact bits of `tof`, and the stage loop reaches this time as
                // `t + node * h` -- two roundings -- then subtracts `t0_s`.
                // Fusing here is a different value in the last ULP, every key
                // misses, and the table silently becomes dead weight that still
                // pays for its solves. `stage_prefill_keys_match_the_stage_loop`
                // is the standing check.
                *time = (t + node * h) - self.t0_s;
            }
            // A short final chunk repeats its last time into the spare lanes.
            // Those lanes solve the same root as a real one, so they converge
            // in step with it and cannot extend the loop; their results are
            // dropped below.
            let last = pack_times
                .get(width.saturating_sub(1))
                .copied()
                .unwrap_or(0.0);
            for slot in width..4 {
                if let Some(time) = pack_times.get_mut(slot) {
                    *time = last;
                }
            }
            seed = baseline.state_at_seeded_x4(pack_times, 0.0, seed, &mut pack_states);
            for slot in 0..width {
                let (Some(&time), Some(&state)) = (pack_times.get(slot), pack_states.get(slot))
                else {
                    continue;
                };
                let (Some(key), Some(entry)) =
                    (table.keys.get_mut(filled), table.states.get_mut(filled))
                else {
                    continue;
                };
                *key = time.to_bits();
                *entry = state;
                filled = filled.saturating_add(1);
            }
            crate::probe::tag_add(&crate::probe::BASELINE_PREFILL, crate::probe::current_tag());
        }
        table.len = filled;
        // The prefill IS the seed chain for these times; leaving the old seed
        // would make a later non-stage query start from a root that is now
        // several stage times behind.
        self.baseline_warm_offset.set(seed);
    }

    pub(crate) fn eclipse_geometry_at_delta(
        &self,
        delta: &[f64; 6],
        t: f64,
    ) -> Result<([f64; 3], [f64; 3]), EclipseError> {
        let baseline = self.baseline_state_at_exact(t - self.t0_s);
        let position = [
            baseline[0] + delta[0],
            baseline[1] + delta[1],
            baseline[2] + delta[2],
        ];
        // `None`: this holds only `&self`, so it cannot produce the exclusive
        // cache borrow. The conversion is a pure function of TAI, so the value
        // is bit-identical; `None` routes to `uncached_driver_utc_jd`, which
        // memoizes it separately. This site is on the eclipse scan and is not
        // cold — see that function.
        let jd = self.ephemeris_lookup_jd_at(t, None);
        let Some(sun) = self.validated_sun_position_at(jd) else {
            return Err(EclipseError::Geometry);
        };
        crate::eclipse::classify_binary_cylinder(position, sun, self.config.earth_radius)?;
        Ok((position, sun))
    }

    pub(crate) fn validate_eclipse_envelope_at_delta(
        &self,
        delta: &[f64; 6],
        t: f64,
    ) -> Result<(), EclipseError> {
        // This runs on EVERY RHS evaluation while an envelope is active
        // (integrator.rs, before `compute_internal`), so the conversion it
        // resolves here is the one the Encke miss arm then reuses through
        // `baseline_exact_memo` instead of recomputing.
        let baseline = self.baseline_state_at_exact(t - self.t0_s);
        crate::eclipse::validate_part_a_eclipse_envelope(
            [
                baseline[0] + delta[0],
                baseline[1] + delta[1],
                baseline[2] + delta[2],
            ],
            [
                baseline[3] + delta[3],
                baseline[4] + delta[4],
                baseline[5] + delta[5],
            ],
        )
    }

    #[inline]
    pub(crate) const fn eclipse_envelope_is_active(&self) -> bool {
        self.eclipse_side.get().is_some()
    }

    pub(crate) fn eclipse_sun_at(&self, t: f64) -> Result<[f64; 3], EclipseError> {
        // `None` for the same reason as `eclipse_geometry_at_delta`.
        let jd = self.ephemeris_lookup_jd_at(t, None);
        self.validated_sun_position_at(jd)
            .ok_or(EclipseError::Geometry)
    }

    /// Longest a UTC day can be in TAI seconds, i.e. the reciprocal of the
    /// largest `|d(jd)/dt|` the sealed `taiutc` can produce. A leap day is 86401
    /// seconds long and every other day is exactly this, so dividing elapsed TAI
    /// seconds by this never understates elapsed Julian Days. See the bound
    /// below for why the slope, and not the endpoint difference, is what the
    /// soundness argument rests on.
    const SECONDS_PER_UTC_DAY: f64 = 86_400.0;

    /// Bound the angular path of the dynamic Sun direction across an interval.
    ///
    /// The retained table is Cartesian piecewise-linear, so the normalized Sun
    /// direction sweeps the unit sphere at a rate that is bounded, in closed
    /// form and over the WHOLE grid, by
    /// [`PrecomputedEphemeris::max_direction_rate_per_day`]. Angular path length
    /// is the integral of that rate, so the sweep across any interval is at most
    /// the supremum times the elapsed days — for every subinterval, including
    /// ones straddling a grid node. One multiply, and no node-crossing case.
    ///
    /// # Elapsed days come from `|dt| / 86400`, not from `jd_b - jd_a`
    ///
    /// The differencing form would need `jd(t)` to be MONOTONE in `t`, or the
    /// total variation this must bound would exceed the endpoint difference.
    /// `jd(t)` runs through the sealed `taiutc`, whose whole point at a leap
    /// second is that UTC labelling is not monotone there (see
    /// `driver_utc_jd_at`). Rather than rest a bound on re-deriving that it
    /// nonetheless comes out monotone, bound the SLOPE: `taiutc` makes a leap
    /// day 86401 seconds long, so `|d(jd)/dt|` is `1/86401` inside one and
    /// exactly `1/86400` everywhere else, hence at most `1/86400` always. Total
    /// variation of `jd` over the interval is then at most `|dt| / 86400`, which
    /// is what the supremum multiplies. This costs about 1e-5 relative looseness
    /// on a leap day and nothing at all on any other, and it removes the
    /// monotonicity question from the soundness argument entirely.
    ///
    /// The Julian Days are still resolved, and are still range-checked, because
    /// the erroring (never clamping) edge contract of the lookup this replaces
    /// is deliberate — see [`Self::eclipse_sun_unit_uncached`]. Only the
    /// interpolation, the two normalizations and the `atan2` are gone; the
    /// admission test is unchanged and `admits_part_a_utc_jd` is pinned equal to
    /// the interpolating entry point's own verdict.
    ///
    /// # This replaced a per-call two-lookup path sum, and it is looser
    ///
    /// The bound it supersedes summed exact great-circle steps between crossed
    /// grid nodes. Against that, this is 0.23% high on the pinned arc and at
    /// most 7.0% high anywhere in the catalogue, that ceiling being the
    /// perihelion-to-aphelion spread of the Sun's apparent rate across the
    /// grid's 4018 days. What consumes the number scales it by so little that
    /// the looseness does not reach the decisions: the term is 0.038% of
    /// `motion_bound_between` and 0.10% of `replay_root_uncertainty_km`, so even
    /// the 7.0% ceiling moves those by 0.0027% and 0.0072%. Both remain valid
    /// upper bounds, which is the only property either consumer's soundness
    /// rests on — a larger bound prunes less and subdivides more, never the
    /// reverse. It does move evaluation points, which is why this change carries
    /// a re-pin.
    ///
    /// # REFUTED — memoizing the direction does not pay. Do not re-litigate.
    ///
    /// Before this became a multiply, the obvious idea was to cache the
    /// normalized direction, since consecutive eclipse intervals share
    /// endpoints. **The repetition is real and large, and memoizing it still
    /// measured null.** It was built three ways and every one was priced. On the
    /// pinned 12-hour strict-HF arc `eclipse_sun_unit_uncached` ran **21,652
    /// times against 7,829 RHS evaluations** — 2.77 per derivative, and exactly
    /// 2.000 per call of this function, because the grid cadence is one day and
    /// the interior node loop therefore never fired on an arc this short. Only
    /// 3,825 of those arguments were distinct. Replaying the captured keys:
    ///
    /// | ways | 1 | 2 | 4 | 8 | 16 | 32 |
    /// |---|---|---|---|---|---|---|
    /// | LRU | 43.28% | 66.22% | 75.07% | 77.17% | 77.65% | 80.23% |
    /// | FIFO | 43.28% | 58.83% | 69.59% | 75.44% | 77.02% | 80.39% |
    ///
    /// with an infinite cache reaching 82.33%. A 4-way LRU measured 75.02% live.
    /// So the redundancy was there and a cache removed it. It did not turn into
    /// time, because THE MEMOIZED WORK WAS TOO CHEAP: a three-arm interleaved
    /// wall A/B over 30 pairs, against a control arm carrying the memo's exact
    /// code with its keys flipped to a verified 0.00% hit rate, split the cost
    /// from the saving at 0.65% and 0.79% of arc, i.e. a net **0.15% ± 0.16%**,
    /// under one sigma and under the 0.35% run-to-run layout floor. Scaling the
    /// caching arm to a perfect cache capped the whole idea near 1.1% of arc
    /// against a floor of roughly 0.4% for even a one-way probe. The lever was
    /// never the repetition; it was that the work did not have to be done at all.
    ///
    /// **The floor clause above is stale and the verdict does not need it**
    /// (checked 2026-08-11). "Under the 0.35% run-to-run layout floor" and the
    /// "roughly 0.4%" probe floor are both 25-propagation-block figures, and
    /// that block shape is what manufactured the 1.13% arc null floor this
    /// project parked a class of levers against; at one propagation per block
    /// two independently built binaries of identical source separate by 0.118%,
    /// so 0.15% is inside the reopened band and the 1.1% perfect-cache ceiling
    /// is well clear of it. What keeps this REFUTED is the OTHER half, which is
    /// the stronger half and is not a floor comparison at all: a zero-hit
    /// control arm carrying the memo's exact code split cost from saving at
    /// 0.65% against 0.79%, so the cache was measured to nearly pay for itself
    /// and the residue is what the sign rests on. Anyone re-opening this must
    /// beat that split, not the floor. Contrast the shared baseline slot at
    /// `Self::baseline_calc_memo`, which was dropped on a floor comparison
    /// alone and turned out to be worth -2.0% once the cost was moved off the
    /// consult path.
    ///
    /// # Both timing arms reach this, including the one labelled `noevents`
    ///
    /// `prop_timing`'s `enable_events = false` arm is a separate integrator
    /// path, but it still runs eclipse geometry and still calls this — verified
    /// by building it with a `panic!` at the top of this function, which fires
    /// on BOTH arms. It is therefore NOT a control for a change to this
    /// function, and an A/B that treats it as one reads the same effect twice
    /// and concludes, wrongly, that the effect must be code layout. The control
    /// that does work is a third build carrying everything EXCEPT this
    /// function's new body; measured that way, layout is null at
    /// -0.08% ± 0.27% and the algorithm is worth 5.39% ± 0.24%, 18 of 18.
    ///
    /// That figure was measured against 9516707, BEFORE the JB2008 libm
    /// retirement landed. On the base this actually ships on it is worth MORE:
    /// **7.14% ± 0.06% (n=20), replicated at 7.24% ± 0.04% (n=18)**, 4458015
    /// against its own supremum-free parent 70ce67e, min-of-block per arm with
    /// the arm order rotated. That is the number to quote, and it must be taken
    /// against 70ce67e — NOT against 0ecabc1, which folds in the whole atan line
    /// and would credit this change with 4.67% of somebody else's work. The
    /// three-arm decomposition, one rotating run, n=18, all 18/18:
    ///
    /// | | vs | effect |
    /// |---|---|---|
    /// | supremum alone | 4458015 vs 70ce67e | **+7.24% ± 0.04%** |
    /// | atan line alone | 70ce67e vs 0ecabc1 | +4.67% ± 0.13% |
    /// | composite | 4458015 vs 0ecabc1 | +11.57% ± 0.12% |
    ///
    /// The parts compose MULTIPLICATIVELY and exactly: 0.9276 × 0.9533 = 0.8843
    /// against a measured composite of 0.8843, agreeing to 0.00 percentage
    /// points. Percentages are not addends — 7.24 + 4.67 = 11.91 is wrong and
    /// 11.57 is right.
    ///
    /// # Bit-identical is not cost-identical, and that is measured here
    ///
    /// The atan line moves ZERO digits of ANY pin in this crate — rect-loop and
    /// V3 carry the same constants at 0ecabc1 and at 2673050 — and it is still
    /// worth 4.67% of arc, replicated at 4.72% ± 0.05% and split further to
    /// 6f785d2 alone at **+4.74% ± 0.06%** (18/18) with 2673050 measuring
    /// **-0.03% ± 0.04%** (9/18), i.e. null: the whole interval is one commit.
    ///
    /// Those atan figures are THIS harness's, on THIS arc, and they disagree
    /// 2.2x with the -2.14% recorded on `jb_rs`'s `atan_x4_dispatched` for the
    /// same code interval. Neither is quotable as "the atan number" without
    /// naming its workload — the two arcs run different atmosphere models,
    /// among other things. The reconciliation lives on that function; read it
    /// before citing either. It does not touch the supremum figures here, which
    /// are same-harness differences and cancel the workload out. A digest verdict says what the arithmetic PRODUCED,
    /// never what it COST. Do not reason from one to the other in either
    /// direction, and in particular do not treat a bit-neutral commit as a free
    /// baseline to measure someone else's lever against: doing that here would
    /// have inflated this lever by about sixty percent of its true size.
    ///
    /// # The 4.74%-vs-2.14% disagreement is RESOLVED: both were right
    ///
    /// That split left an open conflict — this decomposition put `6f785d2` at
    /// +4.74% while the lane that wrote it reported -2.14% for the same commit.
    /// Arbitrated by re-running the commit-level A/B at BOTH atmosphere models,
    /// two builds, rotating arm order, min-of-block per arm:
    ///
    /// | corpus | effect | absolute | arms |
    /// |---|---|---|---|
    /// | `atm_model: 4` (this harness) | **+4.40%** | 81.5 ns/eval | fully separated |
    /// | `atm_model: 5` (what production flies) | **+1.59%** | 20.6 ns/eval | 10 of 10 paired |
    ///
    /// Neither instrument was broken and neither number needs correcting. The
    /// ratio is the whole explanation: `ExactOrekitQuadrature` runs a 63-step
    /// middle plan where `LogQuadratureX4ApproxV1` runs 16, so model 4 issues
    /// roughly four times the `atan_x4` traffic per kernel call. The measured
    /// absolute savings, 81.5 against 20.6 ns/eval, reproduce that ratio; the
    /// arcs are visibly different workloads (7,976 evaluations against 7,560).
    /// A ceiling check settles it independently: model 5 runs ~17 `atan_x4` per
    /// evaluation at ~8.8 ns, so the entire block is ~150 ns of a ~1330 ns
    /// evaluation and even a hypothetical 100%-hit-rate specialisation could
    /// not exceed ~2.4% there. +4.74% is not merely high on model 5, it is
    /// unreachable — which is what identified the corpus as the variable.
    ///
    /// **So every arc-wall percentage taken through `prop_timing`, including
    /// the 7.24% above, is a model-4 number.** The decomposition's internal
    /// arithmetic is sound — the multiplicative composition closes to 0.00 pp
    /// twice — but those percentages are shares of an arc the campaign does not
    /// fly, and they do not transfer to model 5 unchanged.
    ///
    /// **Reproducing them now takes a named arm.** `prop_timing` no longer has a
    /// single default: as of the instrument fix it carries `_m4` and `_m5` tests
    /// for each of its three code paths, built from `exact_profile_dust_config`
    /// and `v3_production_config` respectively, and every `PROP_TIMING_BLOCK`
    /// line names its model in the `arm=` field. Everything above was measured
    /// on what is now the `_m4` arm.
    ///
    /// # THIS supremum on model 5: +2.73%, and the reason is not Amdahl
    ///
    /// Measured rather than inferred, `4458015 vs 70ce67e`, same harness and
    /// method as the arbitration, 40 arcs per arm:
    ///
    /// | corpus | whole-arc | RHS evaluations | per-evaluation |
    /// |---|---|---|---|
    /// | `atm_model: 4` | **+6.17%** | 7,976 → 7,653 (−4.05%) | 1760 → 1721 ns (−2.22%) |
    /// | `atm_model: 5` | **+2.73%** | 7,560 → 7,827 (**+3.53%**) | 1279 → 1202 ns (**−6.05%**) |
    ///
    /// **This change alters the evaluation COUNT, and in opposite directions on
    /// the two models.** On model 4 it removes 4% of the evaluations; on model 5
    /// it adds 3.5%. That is the dominant term and it is why the whole-arc
    /// number falls on the model production flies.
    ///
    /// Per evaluation the supremum is worth nearly three times MORE on model 5
    /// (−6.05% against −2.22%), exactly as an Amdahl argument predicts — model
    /// 5's arc is ~29% cheaper with the eclipse work largely unchanged, so
    /// eclipse is a larger share of it. Anyone reasoning from that alone will
    /// conclude the supremum is worth more on model 5. It is not, because the
    /// extra evaluations more than cancel it. **A per-evaluation share and a
    /// whole-arc share are different quantities here and move in opposite
    /// directions; that prediction was made explicitly before this run and was
    /// wrong.**
    ///
    /// Two things follow. Quote **whole-arc**, since it is the only metric that
    /// captures both terms. And the +3.53% evaluation count on model 5 needed
    /// an audit of its own, which is the section immediately below.
    ///
    /// # AUDITED: the +3.53% is a re-timed schedule, not extra eclipse work
    ///
    /// The obvious reading of the section above — the bound is looser, the
    /// soundness note says a looser bound "prunes less, subdivides more", so
    /// the extra evaluations are the extra subdivision — is WRONG, and wrong
    /// twice over. Counted on the V3 model-5 arc, `70ce67e` against `4458015`,
    /// with the propagation split by leg class:
    ///
    /// | leg class | `dt_max` | pre | post | delta |
    /// |---|---|---|---|---|
    /// | Encke segment legs | 300 s | 5,070 | 5,225 | **+155** |
    /// | bracket-replay refinement legs | 10 s | 1,790 | 1,902 | **+112** |
    /// | root-transaction proof/window legs | 10 s | 700 | 700 | **+0** |
    /// | whole arc | | 7,560 | 7,827 | **+267** |
    ///
    /// **The eclipse scan cannot spend an RHS evaluation at all.**
    /// `scan_crossings` hands `first_crossing_in_step` a `state_at` closure that
    /// is `AcceptedStepReconstruction::position_velocity`, i.e. a cubic Hermite
    /// interpolant of a step the solver has already accepted, and
    /// `motion_bound_between` is two more Hermite samples plus this function.
    /// Neither reaches `LightyearSystem::rhs`. Subdivision does move — 5,027 →
    /// 5,152 splits — and it buys exactly zero of the +267.
    ///
    /// **The subdivision does not even move because of the bound.** The sun
    /// term is 0.038% of `motion_bound_between` and the replacement is 0.23%
    /// looser on this arc, so `motion_bound` moves by under one part in a
    /// million. An extra split needs the pre-split bound to sit inside that
    /// margin of `MAX_BOUNDARY_SEPARATION_KM`, which over 5,027 intervals is an
    /// expectation of ~0.004 extra splits — ~0.13 even at the 7.0% whole-grid
    /// ceiling. The measured +125 is a thousand times that, because the scan is
    /// being handed DIFFERENT accepted steps. Its sign is not controlled by the
    /// bound either: −35 on model 4, and −8 / +45 on the two models at
    /// `eps / 1e4`.
    ///
    /// **What actually moves first is the OTHER consumer.** The first
    /// bit-level divergence on the arc is
    /// `deepest_directed_time_within_root_bound`, which bisects on
    /// `replay_root_uncertainty_km` to a binary64 fixed point. A relatively
    /// tiny change in the ball radius still lands the bisection on a different
    /// double: the first root-transaction leg ends at 1000.265058714 s instead
    /// of 1000.265058715 s, and the committed root time moves from
    /// `0x408f421ed869e7e8` to `0x408f421ed869bbb8` — 1.29 NANOSECONDS. The
    /// next Encke segment restarts from that root, takes the identical 8 steps
    /// for the identical 129 evaluations, and hits its rectification threshold
    /// at 2248.835 s instead of 2250.298 s. **A 1.29 ns root shift becomes a
    /// 1.46 s segment-boundary shift: an amplification of ~1.1e9.** Every
    /// segment boundary after it is a different schedule.
    ///
    /// So the +267 has no location. Bucketed into eight 5,400 s bins the arc
    /// moves +807 in one and −810 in the next — a single boundary sliding
    /// across a bin edge — and the bins' absolute movements sum to 2,021
    /// against a net of 267. **The number that gets quoted is 13% of the motion
    /// it is the residue of.** Two independent controls agree it is noise
    /// rather than work: the root-transaction legs are 28 runs / 42 steps / 700
    /// evaluations in ALL FOUR arms (both models, both tolerances, both
    /// commits) because their `dt_max` is clamped to
    /// `MAX_ROOT_REFINEMENT_STEP_S` and saturated; and re-running the same
    /// difference at `eps / 1e4` roughly halves it while KEEPING the opposite
    /// signs — model 5 25,223 → 25,587 (+1.44%), model 4 25,413 → 24,921
    /// (−1.94%). No real work term shrinks with tolerance and points opposite
    /// ways on two atmospheres of one arc.
    ///
    /// **Do not try to tighten the bound to recover the evaluations.** A
    /// per-interval or coarser-piecewise supremum would keep soundness and cost
    /// nothing to write, but the arithmetic above prices its effect on
    /// subdivision at a fraction of one split and its effect on RHS evaluations
    /// at zero. It would still move every digest, because it would still move
    /// that bisection's fixed point, and the sign of what came out would be as
    /// uncontrolled as it is here. Priced and declined.
    ///
    /// The model-4 arm replicates the 7.24% above at 6.17% — about 15% low, in
    /// the same direction and for the same reason as the atan replication above
    /// (this box sat at load 3.8–6.7 rather than quiet). Absolute levels here
    /// are conservative; the model-4-against-model-5 RATIO is the robust part.
    ///
    /// It went UP rather than down, and the reason is the plain Amdahl one: the
    /// libm and atan work made the REST of the arc cheaper, so the fixed cost
    /// this removes is a larger share of what remains. Compose the two levers
    /// MULTIPLICATIVELY, each measured on its own base — do not add their
    /// percentages, and in particular do not add the 5.39% above to anything,
    /// because that one is against a base that no longer exists.
    ///
    /// The levers do still overlap NUMERICALLY, which is a separate fact from
    /// their cost: on this base the SPARSE and MIXED rect-loop digests land on
    /// exactly the values they held before the libm change existed, so this
    /// change erases that change's effect on both eclipse-dominated cases. Cost
    /// overlap and digest overlap are not the same question and this pair
    /// answers them differently.
    pub(crate) fn eclipse_sun_direction_path_bound(
        &self,
        t_a: f64,
        t_b: f64,
    ) -> Result<f64, EclipseError> {
        if !t_a.is_finite() || !t_b.is_finite() {
            return Err(EclipseError::Geometry);
        }
        if (self.dynamic_ephemeris_flags & ForceFlags::SUN_GRAVITY) == 0 {
            let Some(sun) = self.config.sun_pos else {
                return Err(EclipseError::Geometry);
            };
            let norm_sq = sun.iter().map(|value| value * value).sum::<f64>();
            return if sun.iter().all(|value| value.is_finite())
                && norm_sq.is_finite()
                && norm_sq > 0.0
            {
                Ok(0.0)
            } else {
                Err(EclipseError::Geometry)
            };
        }

        let ephemeris = self
            .dynamic_ephemeris
            .as_deref()
            .and_then(|all| all.get(EphemerisBody::Sun))
            .ok_or(EclipseError::Geometry)?;
        // The admission verdict, via the validated span when it already covers
        // both endpoints. The two UTC resolutions below exist ONLY to feed the
        // range check — the bound itself is `rate * |dt|` and never reads a
        // Julian Day — and the eclipse scan asks this question at every
        // subdivision, so the conversions were the whole cost of this function.
        // See `eclipse_admit_span` for why interval coverage is sound without a
        // taiutc monotonicity assumption. The slow arm below is byte-for-byte
        // the check this fast path skips, and a span miss changes nothing.
        let span_admits = self
            .eclipse_admit_span
            .get()
            .is_some_and(|(lo, hi)| lo <= t_a && t_a <= hi && lo <= t_b && t_b <= hi);
        if !span_admits {
            // `None` for the same reason as `eclipse_geometry_at_delta`.
            let jd_a = self.ephemeris_lookup_jd_at(t_a, None);
            let jd_b = self.ephemeris_lookup_jd_at(t_b, None);
            if !ephemeris.admits_part_a_utc_jd(jd_a) || !ephemeris.admits_part_a_utc_jd(jd_b) {
                return Err(EclipseError::Geometry);
            }
            self.extend_admit_span(ephemeris, t_a, jd_a);
            self.extend_admit_span(ephemeris, t_b, jd_b);
        }
        let rate = ephemeris.max_direction_rate_per_day();
        // A non-finite supremum means the grid's direction is undefined
        // somewhere, which is the whole-table form of the `norm_sq > 0` test
        // this replaced. Fail closed.
        if !(rate.is_finite() && rate >= 0.0) {
            return Err(EclipseError::Geometry);
        }
        let elapsed_days = (t_b - t_a).abs() / Self::SECONDS_PER_UTC_DAY;
        let bound = rate * elapsed_days;
        if !bound.is_finite() {
            return Err(EclipseError::Geometry);
        }
        // Round one representable value outward, so round-to-nearest in the
        // multiply cannot leave a claimed upper bound below the true sweep.
        // Same discipline the summing form applied after each addition.
        if bound == 0.0 {
            return Ok(0.0);
        }
        let bits = bound
            .to_bits()
            .checked_add(1)
            .ok_or(EclipseError::Geometry)?;
        let rounded = f64::from_bits(bits);
        if rounded.is_finite() {
            Ok(rounded)
        } else {
            Err(EclipseError::Geometry)
        }
    }

    /// Margin, in Julian Days, by which a time's resolved UTC JD must clear
    /// BOTH table edges before that time may seed or extend
    /// [`Self::eclipse_admit_span`]. Five seconds.
    ///
    /// This is what lets span coverage answer the admission question for
    /// INTERIOR times without assuming `jd(t)` is monotone. `jd(t)` is affine
    /// with slope `1/86400` between leap seconds and `1/86401` inside one (see
    /// `SECONDS_PER_UTC_DAY`); whatever discontinuity a leap event contributes
    /// is bounded by the one second the event inserts, i.e. `1/86400` of a day
    /// per event. Leap seconds are at least six months apart, so a span capped
    /// at [`Self::ADMIT_SPAN_MAX_S`] (64 days) contains at most ONE event and
    /// the total excursion of `jd` beyond the interval spanned by its endpoint
    /// values is at most `1/86400` day. For `t` in a validated span,
    /// `jd(t) >= jd(lo) - 1/86400 >= table_start + margin - 1/86400` and
    /// symmetrically at the top, and `margin = 5/86400` covers that excursion
    /// five times over. A time within the margin of a table edge simply never
    /// enters the span and keeps taking the exact per-call check, which is the
    /// conservative direction.
    const ADMIT_SPAN_MARGIN_DAYS: f64 = 5.0 / 86_400.0;

    /// Longest validated span, in seconds: 64 days, so at most one leap-second
    /// event can sit inside it. The production propagation is half a day; the
    /// cap exists so the soundness argument above stays a one-event argument no
    /// matter how a caller schedules queries. Beyond it the span stops growing
    /// and out-of-span queries keep the exact check — never an error, only the
    /// old cost.
    const ADMIT_SPAN_MAX_S: f64 = 64.0 * 86_400.0;

    /// Record `t` (whose resolved UTC JD is `jd`) into the validated span, if
    /// `jd` clears both table edges by [`Self::ADMIT_SPAN_MARGIN_DAYS`].
    ///
    /// Every endpoint the span has ever recorded passed this margin test at its
    /// own resolved JD, which is the invariant the interval argument on the
    /// margin constant rests on. Non-finite `jd` fails `admits_part_a_utc_jd`
    /// and is never recorded; `t` is finite because the caller checked it.
    fn extend_admit_span(
        &self,
        ephemeris: &crate::precomputed_ephem::PrecomputedEphemeris,
        t: f64,
        jd: f64,
    ) {
        if !(ephemeris.admits_part_a_utc_jd(jd - Self::ADMIT_SPAN_MARGIN_DAYS)
            && ephemeris.admits_part_a_utc_jd(jd + Self::ADMIT_SPAN_MARGIN_DAYS))
        {
            return;
        }
        let (lo, hi) = self.eclipse_admit_span.get().unwrap_or((t, t));
        let lo = lo.min(t);
        let hi = hi.max(t);
        if hi - lo <= Self::ADMIT_SPAN_MAX_S {
            self.eclipse_admit_span.set(Some((lo, hi)));
        }
    }

    #[cfg(test)]
    pub(crate) const fn admit_span_for_test(&self) -> Option<(f64, f64)> {
        self.eclipse_admit_span.get()
    }

    #[cfg(test)]
    pub(crate) const fn cache_policy_for_test(&self) -> f64 {
        self.baseline_cache_tol
    }

    /// Create a lightweight baseline calculator for event detection.
    ///
    /// This avoids cloning the `GravityCache` when only baseline state is
    /// needed: two OWNED `[[f64; MAX_RECURSIVE_ORDER]]` tables at
    /// `2 * rows * 131 * 8` B — `274_576` B at the full width, `14_672` B at the
    /// `rows = 7` an RHS built for the campaign's `sph_order = 5` allocates. The
    /// two `LegendreCoeffsSimd` tables are borrowed `&'static` and are not
    /// cloned at any size. See `RHSCache` for the derivation.
    ///
    /// The returned calculator borrows this RHS's shared baseline slot rather
    /// than carrying a private one, so the 136 instances one arc mints all
    /// consult the same entry instead of each starting cold. See
    /// [`Self::baseline_calc_memo`] for why that is bit-identical and for what
    /// the reconstruction loss was costing.
    #[must_use]
    pub const fn baseline_calculator(&self) -> BaselineCalculator<'_> {
        BaselineCalculator::new(self.init_equinoc_state, self.t0_s, &self.baseline_calc_memo)
    }
}

#[inline]
fn baseline_cache_is_stale(
    cache_valid: bool,
    requested_tof: f64,
    cached_tof: f64,
    tolerance: f64,
) -> bool {
    let within_tolerance = matches!(
        (requested_tof - cached_tof).abs().partial_cmp(&tolerance),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    );
    !cache_valid || !within_tolerance
}

/// Lightweight struct for computing baseline ECI states.
/// Used by event detection to avoid cloning the heavy `GravityCache`.
///
/// The last exact time key is retained because event selection commonly asks
/// for the same endpoint more than once. Nearby times are never conflated:
/// that would move eclipse geometry by metres while the root contract is
/// measured in centimetres.
///
/// # The memo is the RHS's, not this struct's
///
/// This type used to own its slot. It is minted fresh at every call site — 136
/// instances serving 1,820 consults on one arc — so an owned slot was thrown
/// away 136 times and every instance paid a guaranteed opening miss. Borrowing
/// [`LightyearRHS::baseline_calc_memo`] instead keeps the entry across
/// instances at no per-consult cost: a hit is the same single `u64` compare it
/// always was. The `'rhs` borrow is also what keeps a hit CORRECT, because it
/// forbids the calculator from outliving the elements its key is interpreted
/// against.
///
/// Thread-safety of the `Cell` memo is compiler-enforced, not conventional:
/// the borrowed `Cell` makes this type `!Sync`, so a `&BaselineCalculator`
/// cannot be shared across threads at all. Same argument, at length, on
/// `LightyearRHS` — and the same prohibition: no `unsafe impl Sync`.
pub struct BaselineCalculator<'rhs> {
    pub init_equinoc_state: [f64; 6],
    pub t0_s: f64,
    memo: &'rhs Cell<Option<(u64, [f64; 6])>>,
}

impl<'rhs> BaselineCalculator<'rhs> {
    const fn new(
        init_equinoc_state: [f64; 6],
        t0_s: f64,
        memo: &'rhs Cell<Option<(u64, [f64; 6])>>,
    ) -> Self {
        Self {
            init_equinoc_state,
            t0_s,
            memo,
        }
    }

    /// Get the exact baseline ECI state at `t`.
    #[inline]
    #[must_use]
    pub(crate) fn get_baseline_state(&self, t: f64) -> [f64; 6] {
        let tof = t - self.t0_s;
        let key = tof.to_bits();
        if let Some((stored, state)) = self.memo.get() {
            if stored == key {
                crate::probe::tag_add(
                    &crate::probe::BASELINE_CALC_HIT,
                    crate::probe::current_tag(),
                );
                return state;
            }
        }
        crate::probe::tag_add(
            &crate::probe::BASELINE_CALC_MISS,
            crate::probe::current_tag(),
        );

        let mut state = [0.0; 6];
        equinoc2eci_impl(&self.init_equinoc_state, 6, tof, 0.0, &mut state);
        self.memo.set(Some((key, state)));
        state
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;

    const MU_SUN: f64 = 1.327_124_400_18e11;
    const MU_MOON: f64 = 4.9028e3;
    const MU_JUPITER: f64 = 1.266_865_34e8;
    const MU_VENUS: f64 = 3.24859e5;

    #[inline]
    fn inv(pos: &[f64; 3], mu: f64) -> Option<BodyInvariants> {
        BodyInvariants::precompute(pos, mu)
    }

    #[inline]
    fn simd_total_from_pack(sat_pos: &[f64; 3], pack: &ThirdBodySimdPack) -> [f64; 3] {
        compute_thirdbody_grav_simd4(
            sat_pos,
            pack.body_norm_x,
            pack.body_norm_y,
            pack.body_norm_z,
            pack.inv_body_dist,
            pack.mu_coef,
            pack.mask,
        )
    }

    fn assert_componentwise_relative_error(
        scalar: [f64; 3],
        simd: [f64; 3],
        max_relative_error: f64,
    ) {
        for (axis, (scalar_axis, simd_axis)) in scalar.into_iter().zip(simd).enumerate() {
            let relative_error = ((simd_axis - scalar_axis) / scalar_axis.abs().max(1e-20)).abs();
            assert!(
                relative_error < max_relative_error,
                "Component {axis} mismatch: scalar={scalar_axis}, simd={simd_axis}, rel_err={relative_error}"
            );
        }
    }

    #[test]
    fn test_thirdbody_simd_matches_scalar() {
        let sat_pos = [7000.0, 0.0, 0.0]; // LEO position
        let sun_pos = [1.496e8, 0.0, 0.0];
        let moon_pos = [384_400.0, 0.0, 0.0];
        let jupiter_pos = [7.785e8, 0.0, 0.0];
        let venus_pos = [1.082e8, 0.0, 0.0];

        let scalar_sun = compute_thirdbody_grav(&sat_pos, &sun_pos, MU_SUN);
        let scalar_moon = compute_thirdbody_grav(&sat_pos, &moon_pos, MU_MOON);
        let scalar_jupiter = compute_thirdbody_grav(&sat_pos, &jupiter_pos, MU_JUPITER);
        let scalar_venus = compute_thirdbody_grav(&sat_pos, &venus_pos, MU_VENUS);

        let scalar_total = [
            scalar_sun[0] + scalar_moon[0] + scalar_jupiter[0] + scalar_venus[0],
            scalar_sun[1] + scalar_moon[1] + scalar_jupiter[1] + scalar_venus[1],
            scalar_sun[2] + scalar_moon[2] + scalar_jupiter[2] + scalar_venus[2],
        ];

        // SIMD computation using precomputed invariants
        let pack = make_thirdbody_pack(
            inv(&sun_pos, MU_SUN),
            inv(&moon_pos, MU_MOON),
            inv(&jupiter_pos, MU_JUPITER),
            inv(&venus_pos, MU_VENUS),
        );
        let simd_total = simd_total_from_pack(&sat_pos, &pack);

        assert_componentwise_relative_error(scalar_total, simd_total, 1e-10);
    }

    #[test]
    fn test_thirdbody_simd_partial_active() {
        // Test with only 2 bodies active (Sun and Jupiter)
        let sat_pos = [7000.0, 1000.0, 500.0];
        let sun_pos = [1.496e8, 0.0, 0.0];
        let jupiter_pos = [7.785e8, 0.0, 0.0];

        // Scalar computation
        let scalar_sun = compute_thirdbody_grav(&sat_pos, &sun_pos, MU_SUN);
        let scalar_jupiter = compute_thirdbody_grav(&sat_pos, &jupiter_pos, MU_JUPITER);
        let scalar_total = [
            scalar_sun[0] + scalar_jupiter[0],
            scalar_sun[1] + scalar_jupiter[1],
            scalar_sun[2] + scalar_jupiter[2],
        ];

        // SIMD computation (Moon and Venus inactive)
        let pack = make_thirdbody_pack(
            inv(&sun_pos, MU_SUN),
            None,
            inv(&jupiter_pos, MU_JUPITER),
            None,
        );
        let simd_total = simd_total_from_pack(&sat_pos, &pack);

        assert_componentwise_relative_error(scalar_total, simd_total, 1e-10);
    }

    #[test]
    fn test_thirdbody_simd_all_inactive() {
        // Test with all bodies inactive
        let sat_pos = [7000.0, 0.0, 0.0];

        // SIMD computation with all None
        let pack = make_thirdbody_pack(None, None, None, None);
        let simd_total = simd_total_from_pack(&sat_pos, &pack);

        // Should return zero acceleration
        for &value in &simd_total {
            assert!(
                value.abs() < 1e-20,
                "Expected zero acceleration, got {value}"
            );
        }
    }

    #[test]
    fn test_thirdbody_simd_3d_positions() {
        // Satellite and bodies at arbitrary 3D positions (not axis-aligned)
        let sat_pos = [7000.0, -1500.0, 3200.0];
        let sun_pos = [1.0e8, 5.0e7, -2.0e7];
        let moon_pos = [200_000.0, 300_000.0, 50_000.0];
        let jupiter_pos = [5.0e8, -3.0e8, 1.0e8];
        let venus_pos = [8.0e7, 4.0e7, -1.0e7];

        let scalar_sun = compute_thirdbody_grav(&sat_pos, &sun_pos, MU_SUN);
        let scalar_moon = compute_thirdbody_grav(&sat_pos, &moon_pos, MU_MOON);
        let scalar_jupiter = compute_thirdbody_grav(&sat_pos, &jupiter_pos, MU_JUPITER);
        let scalar_venus = compute_thirdbody_grav(&sat_pos, &venus_pos, MU_VENUS);

        let scalar_total = [
            scalar_sun[0] + scalar_moon[0] + scalar_jupiter[0] + scalar_venus[0],
            scalar_sun[1] + scalar_moon[1] + scalar_jupiter[1] + scalar_venus[1],
            scalar_sun[2] + scalar_moon[2] + scalar_jupiter[2] + scalar_venus[2],
        ];

        let pack = make_thirdbody_pack(
            inv(&sun_pos, MU_SUN),
            inv(&moon_pos, MU_MOON),
            inv(&jupiter_pos, MU_JUPITER),
            inv(&venus_pos, MU_VENUS),
        );
        let simd_total = simd_total_from_pack(&sat_pos, &pack);

        assert_componentwise_relative_error(scalar_total, simd_total, 1e-10);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod arm64_force_tests {
    use super::*;
    use satpy_core::pack_gravity_coeffs;
    use std::mem::size_of;
    use std::sync::Arc;

    fn create_test_coefficients() -> Result<Arc<PackedGravityCoeffs>, GravityError> {
        const ORDER: usize = 0;
        const STRIDE: usize = 2;
        let c_coeffs = [1.0, 0.0, 0.0, 0.0];
        let s_coeffs = [0.0; 4];
        pack_gravity_coeffs(&c_coeffs, &s_coeffs, STRIDE, ORDER).map(Arc::new)
    }

    fn create_rhs_with_flags(
        force_flags: i32,
        qm_ratio: f64,
        r_obj_m: f64,
    ) -> Result<LightyearRHS, GravityError> {
        let order = 0;
        let packed = create_test_coefficients()?;
        let config = Arc::new(ForceConfig {
            sph_order: order,
            force_flags,
            atm_model: 3,
            qm_ratio,
            r_obj_m,
            dt_max: 60.0,
            eps: 1e-8,
            integrator_method: crate::types::StepperMethod::Dopri5Compat,
            ..ForceConfig::default()
        });
        let init_equ = [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0];
        Ok(LightyearRHS::new(
            init_equ,
            0.0,
            2_460_000.5,
            config,
            packed,
        ))
    }

    #[test]
    fn test_relativity_flag_contributes_nonzero_acceleration() {
        let rhs_none =
            create_rhs_with_flags(0, 0.0, 0.0).expect("arm64 baseline gravity fixture must pack");
        let rhs_rel = create_rhs_with_flags(ForceFlags::RELATIVITY, 0.0, 0.0)
            .expect("arm64 relativity gravity fixture must pack");
        let delta = [0.0; 6];
        let t = 0.0;
        let out_none = rhs_none
            .compute_internal_generic(&delta, t)
            .expect("baseline scalar force evaluation must remain valid");
        let out_rel = rhs_rel
            .compute_internal_generic(&delta, t)
            .expect("relativity scalar force evaluation must remain valid");
        let [_, _, _, rel_x, rel_y, rel_z] = out_rel;
        let [_, _, _, none_x, none_y, none_z] = out_none;
        let rel_norm = (rel_x * rel_x + rel_y * rel_y + rel_z * rel_z).sqrt();
        let none_norm = (none_x * none_x + none_y * none_y + none_z * none_z).sqrt();
        assert!(
            none_norm <= 1e-18,
            "expected zero baseline perturbation accel, got {none_norm}"
        );
        assert!(
            rel_norm > 0.0,
            "expected relativity acceleration contribution"
        );
    }

    #[test]
    fn test_lorentz_and_coulomb_flag_gating() {
        let state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let jd = 2_460_000.5;

        // Gate the function production actually calls. The GMST-argument
        // sibling this used to test had no caller outside this assertion, so
        // the flag gating was proven on an abandoned path while the live one
        // went uncovered.
        let rotation = test_frame_rotation(jd);
        let lorentz_zero = compute_lorentz_frame(&state, &rotation, 0.0);
        assert_eq!(lorentz_zero.map(f64::to_bits), [0_u64; 3]);
        let lorentz_nonzero = compute_lorentz_frame(&state, &rotation, 1e-7);
        assert!(
            lorentz_nonzero.iter().all(|v| v.is_finite()),
            "lorentz acceleration must remain finite"
        );
        // Without this the zero-q/m assertion above would also pass if the
        // function returned zero unconditionally.
        assert!(
            lorentz_nonzero.iter().any(|value| *value != 0.0),
            "a nonzero q/m must produce a nonzero Lorentz acceleration"
        );

        let coulomb_zero_qm = compute_coulomb_drag(
            &state,
            jd,
            &rotation,
            0.0,
            1e-3,
            7.292_115_0e-5,
            AtmModel::SyntheticThermosphereProxyV1,
            6378.137,
            None,
        );
        assert_eq!(coulomb_zero_qm.map(f64::to_bits), [0_u64; 3]);

        let coulomb_zero_radius = compute_coulomb_drag(
            &state,
            jd,
            &rotation,
            1e-7,
            0.0,
            7.292_115_0e-5,
            AtmModel::SyntheticThermosphereProxyV1,
            6378.137,
            None,
        );
        assert_eq!(coulomb_zero_radius.map(f64::to_bits), [0_u64; 3]);
    }

    #[test]
    fn synthetic_proxy_density_never_reuses_altitude_only_cache() {
        // The altitude-only cache fields this test once pre-seeded (`cached_alt`,
        // `cached_rho`) were deleted with the Task 5B-2 routing: they had become
        // write-only with no reader, so the reuse hazard is now structurally
        // impossible rather than merely guarded. The GUARD'S INTENT is kept —
        // the proxy density must depend on the epoch and the resulting Earth-fixed
        // longitude, not be a constant an altitude-keyed cache could serve.
        let state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let earth_radius = 6378.137;
        let altitude = 7000.0 - earth_radius;

        let density_at = |jd: f64| {
            density_from_state(
                &state,
                jd,
                &test_frame_rotation(jd),
                earth_radius,
                AtmModel::SyntheticThermosphereProxyV1,
                Some(altitude),
            )
        };

        // Six hours apart: same altitude, very different local time.
        let early = density_at(2_460_000.5);
        let later = density_at(2_460_000.75);
        assert!(early.is_finite() && later.is_finite());
        assert_ne!(
            early.to_bits(),
            later.to_bits(),
            "synthetic proxy density must depend on JD/latitude/longitude, not \
             altitude alone"
        );
    }

    #[test]
    fn removing_drag_removes_synthetic_proxy_acceleration() {
        let state = [6778.137, 0.0, 0.0, 0.0, 7.67, 0.0];
        let rho = density_from_state(
            &state,
            2_460_000.5,
            &test_frame_rotation(2_460_000.5),
            6378.137,
            AtmModel::SyntheticThermosphereProxyV1,
            Some(400.0),
        );
        let rotation = test_frame_rotation(2_460_000.5);
        let with_drag = compute_drag(&state, rho, 0.01, 2.2, &rotation);
        let without_drag = compute_drag(&state, 0.0, 0.01, 2.2, &rotation);
        assert!(with_drag.iter().any(|value| *value != 0.0));
        assert_eq!(without_drag.map(f64::to_bits), [0_u64; 3]);
    }

    #[test]
    fn atmosphere_corotating_state_has_zero_drag_for_tilted_frame_axis() {
        let omega = [1.0e-5, -2.0e-5, 7.0e-5];
        let [r_x, r_y, r_z] = [7000.0, 100.0, 500.0];
        let [omega_x, omega_y, omega_z] = omega;
        let state = [
            r_x,
            r_y,
            r_z,
            omega_y * r_z - omega_z * r_y,
            omega_z * r_x - omega_x * r_z,
            omega_x * r_y - omega_y * r_x,
        ];
        let rotation = FrameRotation {
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            delta_at_s: 0.0,
            itrs_angular_velocity_gcrs: omega,
        };
        let drag = compute_drag(&state, 1.0e-12, 0.01, 2.2, &rotation);
        assert_eq!(drag.map(f64::to_bits), [0_u64; 3]);
    }

    #[test]
    fn part_a_epoch_drag_matches_full_frame_vector_formula() {
        let rotation = test_frame_rotation(2_459_794.5);
        let state = [6_978.0, -311.0, 522.0, 0.41, 7.36, -0.82];
        let rho = 8.0e-13;
        let am_ratio = 1.948;
        let cd = 2.2;
        let actual = compute_drag(&state, rho, am_ratio, cd, &rotation);

        let [r_x, r_y, r_z, v_x, v_y, v_z] = state;
        let [omega_x, omega_y, omega_z] = rotation.itrs_angular_velocity_gcrs;
        let relative = [
            v_x - (omega_y * r_z - omega_z * r_y),
            v_y - (omega_z * r_x - omega_x * r_z),
            v_z - (omega_x * r_y - omega_y * r_x),
        ];
        let relative_mps = relative.map(|component| component * KM_TO_M);
        let speed_mps = relative_mps
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let scale = -0.5 * cd * am_ratio * rho * speed_mps * M_TO_KM;
        let expected = relative_mps.map(|component| scale * component);

        for (axis, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            let delta = (actual - expected).abs();
            let bound = 2.0e-15 * expected.abs().max(f64::MIN_POSITIVE);
            assert!(
                delta <= bound,
                "drag axis {axis}: actual={actual:.17e} expected={expected:.17e} delta={delta:.3e}"
            );
        }
        assert!(
            actual
                .into_iter()
                .zip(relative)
                .map(|(a, v)| a * v)
                .sum::<f64>()
                < 0.0,
            "drag must oppose full-frame atmosphere-relative velocity"
        );
    }

    #[test]
    fn dynamic_ephemeris_resolves_sun_at_each_rhs_epoch() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let flags = ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
        let ephem = crate::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("test ephemeris catalogues must load");
        let (start_jd, end_jd) = ephem
            .common_jd_range()
            .expect("test ephemeris range must exist");
        let jd0 = 0.5 * (start_jd + end_jd);
        let config = ForceConfig {
            sph_order: 0,
            force_flags: flags,
            am_ratio: 0.02,
            cr: 1.3,
            ..ForceConfig::default()
        }
        .with_ephemeris_for_arc(jd0, jd0 + 0.05)
        .expect("test arc must have dynamic ephemeris coverage");
        let packed = create_test_coefficients().expect("test gravity fixture must pack");
        let rhs = LightyearRHS::new(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            jd0,
            Arc::new(config),
            packed,
        );
        let later_jd = jd0 + 0.05;
        let sun0 = rhs
            .validated_sun_position_at(jd0)
            .expect("Sun at initial JD");
        let sun1 = rhs
            .validated_sun_position_at(later_jd)
            .expect("Sun at later JD");

        assert_eq!(
            sun0.map(f64::to_bits),
            ephem
                .get(crate::precomputed_ephem::Body::Sun)
                .unwrap()
                .position_at_part_a_utc_jd(UtcJulianDay::new(jd0).unwrap())
                .unwrap()
                .map(f64::to_bits)
        );
        assert_eq!(
            sun1.map(f64::to_bits),
            ephem
                .get(crate::precomputed_ephem::Body::Sun)
                .unwrap()
                .position_at_part_a_utc_jd(UtcJulianDay::new(later_jd).unwrap())
                .unwrap()
                .map(f64::to_bits)
        );
        assert_ne!(sun0.map(f64::to_bits), sun1.map(f64::to_bits));
    }

    #[test]
    fn effective_srp_requires_force_and_nonzero_coefficients_not_valid_geometry() {
        let dynamic = ForceConfig {
            force_flags: ForceFlags::SRP,
            dynamic_ephemeris_flags: ForceFlags::SUN_GRAVITY,
            sun_pos: None,
            am_ratio: 0.02,
            cr: 1.3,
            p_sun: 4.56e-6,
            ..ForceConfig::default()
        };
        assert!(effective_scalar_srp(&dynamic));
        assert!(!effective_scalar_srp(&ForceConfig {
            force_flags: 0,
            ..dynamic
        }));
        assert!(effective_scalar_srp(&ForceConfig {
            dynamic_ephemeris_flags: 0,
            ..dynamic
        }));
        assert!(!effective_scalar_srp(&ForceConfig { cr: 0.0, ..dynamic }));
    }

    #[test]
    fn srp_fixed_side_is_binary_and_boundary_policy_is_external() {
        let sun = [149_597_870.7, 0.0, 0.0];
        let earth_radius = 6378.137;
        let state = [-7000.0, earth_radius, 0.0, 0.0, 7.5, 0.0];
        assert_eq!(
            compute_srp_with_precomputed(
                &state,
                &sun,
                4.56e-6,
                1.3,
                0.01,
                crate::eclipse::EclipseSide::Shadow,
            )
            .map(f64::to_bits),
            [0_u64; 3]
        );
        assert_ne!(
            compute_srp_with_precomputed(
                &state,
                &sun,
                4.56e-6,
                1.3,
                0.01,
                crate::eclipse::EclipseSide::Lit,
            )
            .map(f64::to_bits),
            [0_u64; 3]
        );
    }

    #[test]
    fn srp_pressure_scales_with_inverse_square_sun_distance() {
        let state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let near_sun = [0.983 * AU_KM, 0.0, 0.0];
        let far_sun = [1.017 * AU_KM, 0.0, 0.0];
        let near = compute_srp_with_precomputed(
            &state,
            &near_sun,
            4.56e-6,
            1.3,
            0.01,
            crate::eclipse::EclipseSide::Lit,
        );
        let far = compute_srp_with_precomputed(
            &state,
            &far_sun,
            4.56e-6,
            1.3,
            0.01,
            crate::eclipse::EclipseSide::Lit,
        );
        let observed = near[0].abs() / far[0].abs();
        let near_distance = near_sun[0] - state[0];
        let far_distance = far_sun[0] - state[0];
        let expected = (far_distance / near_distance).powi(2);
        assert!((observed - expected).abs() < 1e-12);
    }

    #[test]
    fn lit_srp_invalid_satellite_sun_geometry_fails_nonfinite() {
        let state = [AU_KM, 0.0, 0.0, 0.0, 0.0, 0.0];
        let acceleration = compute_srp_with_precomputed(
            &state,
            &[AU_KM, 0.0, 0.0],
            4.56e-6,
            1.3,
            0.02,
            crate::eclipse::EclipseSide::Lit,
        );
        assert!(acceleration.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn test_rhs_cache_fits_comfortably_within_worker_stack_budget() {
        assert!(
            size_of::<RHSCache>() < 64 * 1024,
            "RHSCache too large for reliable Rayon worker stacks: {} bytes",
            size_of::<RHSCache>()
        );
    }

    #[test]
    fn independently_constructed_rhs_shares_immutable_inputs_not_mutable_cache() {
        let rhs = create_rhs_with_flags(0, 0.0, 0.0)
            .expect("first independent arm64 gravity fixture must pack");
        let independent = LightyearRHS::try_new(
            rhs.init_equinoc_state,
            rhs.t0_s,
            rhs.jd0,
            std::sync::Arc::clone(&rhs.config),
            std::sync::Arc::clone(&rhs.packed),
        )
        .expect("second independent arm64 gravity fixture must construct");

        assert!(std::sync::Arc::ptr_eq(&rhs.config, &independent.config));
        assert!(std::sync::Arc::ptr_eq(&rhs.packed, &independent.packed));
        assert_ne!(rhs.cache.get(), independent.cache.get());
    }
}

#[cfg(test)]
mod jb2008_rhs_tests {
    use super::*;
    use satpy_core::pack_gravity_coeffs;
    use std::sync::Arc;

    const JD0: f64 = 2_459_600.5;

    fn coefficients(order: usize) -> anyhow::Result<Arc<PackedGravityCoeffs>> {
        let stride = order
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("JB2008 test gravity stride overflow"))?;
        let len = stride
            .checked_mul(stride)
            .ok_or_else(|| anyhow::anyhow!("JB2008 test gravity storage overflow"))?;
        let mut c = vec![0.0; len];
        let s = vec![0.0; len];
        *c.first_mut()
            .ok_or_else(|| anyhow::anyhow!("JB2008 test gravity storage must be nonempty"))? = 1.0;
        Ok(Arc::new(pack_gravity_coeffs(&c, &s, stride, order)?))
    }

    fn construct(config: &ForceConfig) -> anyhow::Result<LightyearRHS> {
        let packed = coefficients(config.sph_order)?;
        LightyearRHS::try_new(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            JD0,
            Arc::new(*config),
            packed,
        )
    }

    /// Shared fixture for every JB2008 test RHS.
    ///
    /// Takes the ephemeris guard because `with_ephemeris_for_arc` reads the
    /// process-global catalogue, and the tests that publish a deliberately
    /// conflicting temp catalogue hold that same guard. Without it this
    /// fixture could observe their transient state and fail with `cached sun
    /// ephemeris SHA-256 ... conflicts with compiled SHA-256 ...` -- the
    /// order-dependent flake whose victim was whichever test happened to be
    /// scheduled beside the installer. The guard is reentrant, so the tests
    /// that already hold it across a fixture build do not deadlock; it is
    /// dropped when this function returns, so nothing is held across the
    /// caller's own work.
    fn jb2008_rhs(atm_model: i32, static_sun: Option<[f64; 3]>) -> LightyearRHS {
        let _ephemeris_guard = crate::precomputed_ephem::ephemeris_test_guard();
        let config = ForceConfig {
            sph_order: 0,
            force_flags: ForceFlags::DRAG,
            atm_model,
            am_ratio: 0.01,
            cd: 2.2,
            sun_pos: static_sun,
            ..ForceConfig::default()
        }
        .with_ephemeris_for_arc(JD0, JD0 + 2.0)
        .expect("JB2008 test arc must resolve");
        construct(&config).expect("scalar JB2008 RHS must construct")
    }

    fn model4_rhs(static_sun: Option<[f64; 3]>) -> LightyearRHS {
        jb2008_rhs(4, static_sun)
    }

    fn model5_rhs() -> LightyearRHS {
        jb2008_rhs(5, None)
    }

    /// Shared JB2008 fixture state. Named because one test asserts on the
    /// altitude it implies; a quiet edit here would leave that test green and
    /// its name false.
    const MODEL4_STATE: [f64; 6] = [7578.137, 350.0, 700.0, 0.0, 7.4, 0.1];

    fn model4_density(rhs: &LightyearRHS, jd: f64) -> f64 {
        rhs.density_at_state(&MODEL4_STATE, jd, &test_frame_rotation(jd), None)
    }

    #[test]
    fn jb2008_model_four_and_five_dispatch_match_their_real_driver_kernels() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let exact = model4_rhs(None);
        let approximation = model5_rhs();
        assert!(Arc::ptr_eq(
            exact.jb2008_drivers.as_ref().expect("model 4 drivers"),
            approximation
                .jb2008_drivers
                .as_ref()
                .expect("model 5 drivers")
        ));

        let corpus = [
            ([6578.137, 0.0, 0.0, 0.0, 7.8, 0.0], JD0),
            ([6100.0, 3200.0, 500.0, -1.0, 7.2, 0.2], JD0 + 0.75),
            ([5000.0, -4100.0, 2500.0, 2.0, 6.8, -0.4], JD0 + 1.5),
        ];
        for (state, jd) in corpus {
            let rotation = test_frame_rotation(jd);
            let exact_dispatch = exact.density_at_state(&state, jd, &rotation, None);
            let exact_kernel =
                exact.jb2008_density_at_state(&state, jd, &rotation, Jb2008Profile::Exact);
            let approximation_dispatch =
                approximation.density_at_state(&state, jd, &rotation, None);
            let approximation_kernel = approximation.jb2008_density_at_state(
                &state,
                jd,
                &rotation,
                Jb2008Profile::ApproxV1,
            );
            assert_eq!(exact_dispatch.to_bits(), exact_kernel.to_bits(), "jd={jd}");
            assert_eq!(
                approximation_dispatch.to_bits(),
                approximation_kernel.to_bits(),
                "jd={jd}"
            );
            assert_ne!(
                exact_dispatch.to_bits(),
                approximation_dispatch.to_bits(),
                "corpus failed to distinguish model dispatch at jd={jd}"
            );
        }
    }

    /// The Sun memo must return the bits the interpolation would have returned,
    /// at every JD and in every order, including after a `reset_for_propagation`.
    ///
    /// This is the whole justification for the memo: `dynamic_body_position` is a
    /// pure function of `(body, jd)` over an immutable table, so reuse is free.
    /// If this ever fails, the memo key has stopped being complete — fix the key,
    /// do not widen a tolerance, and do not re-pin a digest around it.
    #[test]
    fn sun_memo_is_bit_identical_to_the_uncached_interpolation() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let mut rhs = model4_rhs(None);

        // Deliberately NOT monotone, and with repeats: a memo that only ever
        // sees increasing arguments hides a stale-entry bug.
        let corpus = [
            JD0,
            JD0,
            JD0 + 1.0,
            JD0,
            JD0 + 0.5,
            JD0 + 1.999,
            JD0 + 0.5,
            JD0 + 1.0,
        ];
        for jd in corpus {
            let expected = rhs.body_position_uncached(EphemerisBody::Sun, jd);
            let actual = rhs.dynamic_body_position(EphemerisBody::Sun, jd);
            assert_eq!(
                actual.map(f64::to_bits),
                expected.map(f64::to_bits),
                "Sun memo diverged from the interpolation at jd={jd}"
            );
        }

        // A reset moves `t0_s` and the elements, i.e. which JD gets asked for.
        // It must not change what a given JD answers, and the surviving memo
        // entry must still be correct afterwards.
        rhs.reset_for_propagation([7000.0, 0.001, 0.0, 0.0, 0.0, 0.0], 12_345.0);
        for jd in [JD0 + 1.0, JD0, JD0 + 1.0] {
            let expected = rhs.body_position_uncached(EphemerisBody::Sun, jd);
            let actual = rhs.dynamic_body_position(EphemerisBody::Sun, jd);
            assert_eq!(
                actual.map(f64::to_bits),
                expected.map(f64::to_bits),
                "Sun memo diverged after reset_for_propagation at jd={jd}"
            );
        }
    }

    /// Non-vacuity for the test above: prove the memo READ path is live, and
    /// prove it is Sun-only.
    ///
    /// A bit-identity test alone passes just as happily against a memo that is
    /// never consulted, so it cannot on its own show the hoist does anything.
    /// This poisons the cell with a position the ephemeris cannot produce — the
    /// Sun is never 1 km from Earth's centre — and requires the poison to come
    /// back out. Both directions are asserted: hit at the poisoned key, real
    /// interpolation at every other key and for every other body.
    #[test]
    fn sun_memo_is_actually_consulted_and_is_sun_only() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let rhs = model4_rhs(None);

        let truth = rhs.body_position_uncached(EphemerisBody::Sun, JD0);
        let poison = [1.0, 2.0, 3.0];
        assert_ne!(
            truth.map(f64::to_bits),
            poison.map(f64::to_bits),
            "poison must be a value the ephemeris cannot return"
        );
        rhs.sun_position_memo.set(Some((JD0.to_bits(), poison)));

        // Direction 1: the memo is read, so the poison is what comes back.
        assert_eq!(
            rhs.dynamic_body_position(EphemerisBody::Sun, JD0)
                .map(f64::to_bits),
            poison.map(f64::to_bits),
            "Sun memo read path is dead: the hoist saves nothing"
        );

        // Direction 2: a different JD misses and interpolates, and doing so
        // evicts the poison rather than being answered by it.
        rhs.sun_position_memo.set(Some((JD0.to_bits(), poison)));
        let other = rhs.dynamic_body_position(EphemerisBody::Sun, JD0 + 1.0);
        assert_eq!(
            other.map(f64::to_bits),
            rhs.body_position_uncached(EphemerisBody::Sun, JD0 + 1.0)
                .map(f64::to_bits),
            "a different JD must not be answered from the memo"
        );

        // Direction 3: the Moon is not memoized and cannot be answered by the
        // Sun's entry, whatever key that entry carries.
        rhs.sun_position_memo.set(Some((JD0.to_bits(), poison)));
        assert_eq!(
            rhs.dynamic_body_position(EphemerisBody::Moon, JD0)
                .map(f64::to_bits),
            rhs.body_position_uncached(EphemerisBody::Moon, JD0)
                .map(f64::to_bits),
            "Moon must bypass the Sun memo entirely"
        );
        assert_eq!(
            rhs.sun_position_memo.get().map(|(key, _)| key),
            Some(JD0.to_bits()),
            "a Moon lookup must not write the Sun memo"
        );
    }

    /// `baseline_state_at_exact` with the memo taken out of the path.
    ///
    /// There is no production `_uncached` sibling to borrow the way the Sun
    /// memo has `body_position_uncached`, so this restates the two arms. It is
    /// a deliberate near-duplicate: if it ever stops matching the arms in
    /// `baseline_state_at_exact`, the tests below stop measuring the memo and
    /// start measuring this function, so change both together.
    ///
    /// `seed_offset` is passed in rather than read off `rhs` because every
    /// caller needs the value as it stood BEFORE the production call it is
    /// checking -- that call overwrites it.
    fn baseline_state_at_exact_unmemoized(
        rhs: &LightyearRHS,
        tof: f64,
        seed_offset: Option<f64>,
    ) -> ([f64; 6], Option<f64>) {
        let mut state = [0.0; 6];
        let offset = if let Some(baseline) = rhs.equinoc_baseline {
            baseline.state_at_seeded(tof, 0.0, seed_offset, &mut state)
        } else {
            equinoc2eci_impl(&rhs.init_equinoc_state, 6, tof, 0.0, &mut state);
            None
        };
        (state, offset)
    }

    /// The one-deep memo and the warm seed, restated. See
    /// `baseline_state_at_exact_unmemoized` on why this duplication is
    /// deliberate and what breaks if the two drift.
    ///
    /// It carries BOTH pieces of state, because the seed makes each call's
    /// answer depend on the calls before it. A reference holding only the memo
    /// would drift from production on the second call, and could only be
    /// repaired by deleting the seed.
    #[derive(Default)]
    struct BaselineReplay {
        memo: Option<(u64, [f64; 6])>,
        seed: Option<f64>,
    }

    impl BaselineReplay {
        fn state_at(&mut self, rhs: &LightyearRHS, tof: f64) -> [f64; 6] {
            let key = tof.to_bits();
            if let Some((stored, state)) = self.memo {
                if stored == key {
                    return state;
                }
            }
            let (state, offset) = baseline_state_at_exact_unmemoized(rhs, tof, self.seed);
            self.seed = offset;
            self.memo = Some((key, state));
            state
        }
    }

    fn baseline_memo_rhs(elements: [f64; 6], t0_s: f64) -> LightyearRHS {
        let mut rhs = model4_rhs(None);
        rhs.reset_for_propagation(elements, t0_s);
        assert!(
            rhs.equinoc_baseline.is_some(),
            "these elements must reach the hoisted arm, not the degenerate fallback"
        );
        rhs
    }

    const BASELINE_ELEMENTS_A: [f64; 6] = [7178.137, 0.0227, 0.0114, 0.0, 0.0, 1.2];
    const BASELINE_ELEMENTS_B: [f64; 6] = [7500.0, -0.031, 0.0072, 0.0, 0.0, 2.4];

    /// Bitwise equality of two states.
    ///
    /// `assert_eq!` on `[f64; 6]` is denied here (`float_cmp` on arrays), and
    /// the poison check wants EXACT bits anyway -- an epsilon compare would let
    /// a table that returns a nearly-right state pass as if it returned the
    /// poison.
    fn same_bits(left: &[f64; 6], right: &[f64; 6]) -> bool {
        left.iter()
            .zip(right.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    /// One abscissa by index, without indexing.
    fn node(index: usize) -> f64 {
        VERN9_NODES
            .get(index)
            .copied()
            .expect("VERN9_NODES has 16 entries")
    }

    /// Vern9's abscissas, restated so this module can build the stage set the
    /// integrator would hand `prefill_stage_baselines`.
    const VERN9_NODES: [f64; 16] = [
        0.0,
        0.034_62,
        0.097_024_350_638_780_45,
        0.145_536_525_958_170_68,
        0.561,
        0.229_007_911_590_485_03,
        0.544_992_088_409_515,
        0.645,
        0.483_75,
        0.067_57,
        0.25,
        0.659_065_061_873_099_9,
        0.820_6,
        0.901_2,
        1.0,
        1.0,
    ];

    /// The prefill must key on the SAME bits the stage loop will ask for.
    ///
    /// This is the check the whole path stands on. `solver.rs` reaches stage
    /// `i` at `t + c[i] * h` and the RHS then forms `t_stage - t0_s`. Spell
    /// either operation differently in the prefill and the `tof` is correct to
    /// the last ULP and WRONG as a key: every lookup misses, the table sits
    /// unread, and the only symptom is that the prefill's solves are pure loss.
    /// No pin moves, no accuracy gate moves, nothing else in the suite goes red
    /// — the arc just quietly gets slower than it was.
    ///
    /// # The shapes are chosen, not arbitrary, and here is what they catch
    ///
    /// The two ways to get this wrong are not equally easy to trip, which is
    /// why a single `(t, h, t0_s)` is not enough:
    ///
    /// * **Re-associating** to `(t - t0_s) + c[i] * h` is the likely mistake —
    ///   it reads as the same quantity and is the order the RHS's own `tof`
    ///   line suggests. It is also the loud one: at the first shape below it
    ///   moves 12 of the 16 keys, because `t - t0_s` discards low bits that
    ///   `t + c*h` still holds.
    /// * **Contracting** to `c[i].mul_add(h, t)` is nearly benign at these
    ///   magnitudes and that is exactly why it needs a shape picked for it.
    ///   `t` is 1e3..1e5 while `c*h` is at most ~60, so the sum's rounding
    ///   usually swamps the product's and the fused and unfused forms agree.
    ///   Measured over these nodes it moves 0 of 16 keys at most shapes and 1
    ///   of 16 at `(1000.0, 60.0)`, which is the shape that is here for it.
    ///
    /// A shape that discriminates neither is worth nothing here, so do not
    /// "simplify" this list to one entry.
    #[test]
    fn stage_prefill_keys_match_the_stage_loop() {
        // (t, h, t0_s)
        const SHAPES: [(f64, f64, f64); 4] = [
            (1_234.567_890_123_456_7, 41.253_456_789_012_3, 1_000.0),
            (1_234.5, 41.25, 1_000.0),
            (86_400.123_456_789, 37.911_337_991_133_79, 42_000.5),
            (1_000.0, 60.0, 13.7),
        ];
        for (t, h, t0_s) in SHAPES {
            let rhs = baseline_memo_rhs(BASELINE_ELEMENTS_A, t0_s);
            rhs.prefill_stage_baselines(t, h, &VERN9_NODES);
            let table = unsafe { &*rhs.stage_baselines.get() };
            for &node in &VERN9_NODES {
                // Exactly how the integrator forms it, then how the RHS does.
                let tof = (t + node * h) - rhs.t0_s;
                assert!(
                    table.get(tof.to_bits()).is_some(),
                    "at (t {t}, h {h}, t0_s {t0_s}) the stage time t + {node} * h gives tof \
                     {tof:e}, which the prefill did not key; the prefill and the stage loop \
                     compute the stage time differently"
                );
            }
        }
    }

    /// The shapes above must actually separate a right prefill from a wrong
    /// one, or `stage_prefill_keys_match_the_stage_loop` is decoration.
    ///
    /// Both wrong spellings are computed here directly and at least one key per
    /// shape must move. Without this, shrinking or "tidying" the shape list
    /// silently turns that test vacuous — which it WAS when first written, at a
    /// single benign shape where both wrong spellings reproduced every key.
    #[test]
    fn stage_prefill_key_shapes_discriminate_the_wrong_spellings() {
        const SHAPES: [(f64, f64, f64); 4] = [
            (1_234.567_890_123_456_7, 41.253_456_789_012_3, 1_000.0),
            (1_234.5, 41.25, 1_000.0),
            (86_400.123_456_789, 37.911_337_991_133_79, 42_000.5),
            (1_000.0, 60.0, 13.7),
        ];
        let mut assoc_moved = 0_usize;
        let mut fma_moved = 0_usize;
        for (t, h, t0_s) in SHAPES {
            for &node in &VERN9_NODES {
                let right = (t + node * h) - t0_s;
                if ((t - t0_s) + node * h).to_bits() != right.to_bits() {
                    assoc_moved += 1;
                }
                if (node.mul_add(h, t) - t0_s).to_bits() != right.to_bits() {
                    fma_moved += 1;
                }
            }
        }
        assert!(
            assoc_moved > 0,
            "no shape separates the re-associated spelling; the key-match test is vacuous"
        );
        assert!(
            fma_moved > 0,
            "no shape separates the FMA-contracted spelling; the key-match test is vacuous \
             against contraction, which is the quieter of the two mistakes"
        );
    }

    /// The prefilled table must be what `baseline_state_at_exact` ANSWERS FROM,
    /// not merely something that exists alongside it.
    ///
    /// Poison one slot and the query for that exact `tof` must return the
    /// poison. A table that is filled and then ignored -- the failure mode a
    /// key mismatch produces, and the one no other test can see -- returns the
    /// real state here and fails.
    #[test]
    fn stage_prefill_table_is_actually_consulted() {
        let rhs = baseline_memo_rhs(BASELINE_ELEMENTS_A, 1_000.0);
        let t = 1_234.5_f64;
        let h = 41.25_f64;
        let node = node(7);
        let tof = (t + node * h) - rhs.t0_s;

        rhs.prefill_stage_baselines(t, h, &VERN9_NODES);
        let honest = rhs.baseline_state_at_exact(tof);

        // The one-deep exact memo would shadow the table on a repeat query, so
        // it is cleared before the poisoned read.
        let poison = [-9.87e5; 6];
        {
            let table = unsafe { &mut *rhs.stage_baselines.get() };
            let slot = table
                .keys
                .iter()
                .take(table.len)
                .position(|&key| key == tof.to_bits())
                .expect("the stage time must be in the table");
            *table
                .states
                .get_mut(slot)
                .expect("the slot came from the parallel key array") = poison;
        }
        rhs.baseline_exact_memo.set(None);
        assert!(
            same_bits(&rhs.baseline_state_at_exact(tof), &poison),
            "the prefilled table is not consulted: the query bypassed it"
        );
        assert!(
            !same_bits(&honest, &poison),
            "the poison must differ from the real state or this proves nothing"
        );
    }

    /// A `tof` that is not a stage time must miss the table and take the
    /// unchanged path.
    ///
    /// Dense output, the eclipse scan's `state_at` and event legs all query
    /// arbitrary times. The table can only ADD hits; if it ever answered a time
    /// it was not filled with, it would be interpolating, which it must never
    /// do.
    #[test]
    fn stage_prefill_does_not_answer_non_stage_times() {
        let rhs = baseline_memo_rhs(BASELINE_ELEMENTS_A, 1_000.0);
        let t = 1_234.5_f64;
        let h = 41.25_f64;
        rhs.prefill_stage_baselines(t, h, &VERN9_NODES);
        let table = unsafe { &*rhs.stage_baselines.get() };
        // Between two stage times, and one ULP off a stage time.
        let between = (t + 0.3 * h) - rhs.t0_s;
        let on_stage = (t + node(4) * h) - rhs.t0_s;
        let nudged = f64::from_bits(on_stage.to_bits().wrapping_add(1));
        assert!(table.get(between.to_bits()).is_none());
        assert!(table.get(nudged.to_bits()).is_none());
        assert!(table.get(on_stage.to_bits()).is_some());
    }

    /// The exact-TOF baseline memo must return the bits an independent replay
    /// of the same call SEQUENCE produces, at every TOF, in every order, and
    /// across a `reset_for_propagation`.
    ///
    /// Modelled on `sun_memo_is_bit_identical_to_the_uncached_interpolation`,
    /// and it needs the analogue more than the Sun memo does. The Sun memo's
    /// key is complete on its own: `dynamic_body_position` is a pure function
    /// of `(body, jd)` over an immutable table. This memo's key is
    /// `tof.to_bits()` ALONE, and the value depends on the baseline elements
    /// too, so key completeness rests entirely on `reset_for_propagation`
    /// clearing the cell. That invariant held at the tip and nothing pinned
    /// it.
    ///
    /// **A sequence, not a per-call identity, and that is the warm seed's
    /// doing.** The reference below therefore carries BOTH pieces of state the
    /// production function carries -- the one-deep memo and the seed -- because
    /// the seed makes each call's answer depend on the calls before it. A
    /// reference holding only the memo would drift from production on the
    /// second call and could only be repaired by deleting the seed.
    #[test]
    fn baseline_exact_memo_matches_an_independent_replay_of_the_sequence() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let mut rhs = baseline_memo_rhs(BASELINE_ELEMENTS_A, 0.0);

        let mut replay = BaselineReplay::default();
        // Deliberately NOT monotone, and with repeats: a memo that only ever
        // sees increasing arguments hides a stale-entry bug.
        let corpus = [0.0, 0.0, 600.0, 0.0, 300.0, 5400.0, 300.0, 600.0];
        for tof in corpus {
            let expected = replay.state_at(&rhs, tof);
            let actual = rhs.baseline_state_at_exact(tof);
            assert_eq!(
                actual.map(f64::to_bits),
                expected.map(f64::to_bits),
                "baseline memo diverged from the replayed sequence at tof={tof}"
            );
        }

        // The second-writer case, which is what the key does not cover: the
        // same TOF under different elements. `reset_for_propagation` calls a
        // surviving entry "WRONG here, not merely cold"; this is the test that
        // says so. The replay is reset alongside, which is also what pins the
        // seed's own clear: leave `self.baseline_warm_offset` set across the
        // reset in production and the two sides part company here.
        let stale = rhs.baseline_state_at_exact(600.0);
        rhs.reset_for_propagation(BASELINE_ELEMENTS_B, 12_345.0);
        replay = BaselineReplay::default();
        let fresh = rhs.baseline_state_at_exact(600.0);
        assert_eq!(
            fresh.map(f64::to_bits),
            replay.state_at(&rhs, 600.0).map(f64::to_bits),
            "the same TOF under new elements returned a stale baseline state"
        );
        assert_ne!(
            fresh.map(f64::to_bits),
            stale.map(f64::to_bits),
            "the two element sets agree at this TOF, so the case above proves nothing"
        );
    }

    /// A propagation's baseline states must not depend on what this RHS did
    /// before `reset_for_propagation`.
    ///
    /// The seed is the only piece of `LightyearRHS` whose value is a function
    /// of the call ORDER rather than of a key, so it is the only one that can
    /// make one arc's arithmetic depend on the arc that ran before it in the
    /// same object. Batch callers reuse an RHS across segments and rely on
    /// `strict_hf_pin`-grade reproducibility; this is the unit-scale statement
    /// of that, and it goes red if `reset_for_propagation` stops clearing the
    /// seed while every bit-identity check above stays green.
    #[test]
    fn reset_for_propagation_clears_the_warm_seed() {
        const SEQUENCE: [f64; 3] = [900.0, 1800.0, 2700.0];
        const PROLOGUE: [f64; 3] = [120.0, 60.0, 4800.0];

        let run_sequence = |rhs: &mut LightyearRHS| {
            SEQUENCE.map(|tof| rhs.baseline_state_at_exact(tof).map(f64::to_bits))
        };

        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let mut rhs = baseline_memo_rhs(BASELINE_ELEMENTS_A, 0.0);

        // Two prologues that differ, so a surviving seed carries a different
        // value into the run that follows.
        rhs.reset_for_propagation(BASELINE_ELEMENTS_A, 0.0);
        let after_no_prologue = run_sequence(&mut rhs);

        rhs.reset_for_propagation(BASELINE_ELEMENTS_A, 0.0);
        for tof in PROLOGUE {
            rhs.baseline_state_at_exact(tof);
        }
        rhs.reset_for_propagation(BASELINE_ELEMENTS_A, 0.0);
        let after_reset_prologue = run_sequence(&mut rhs);
        assert_eq!(
            after_no_prologue, after_reset_prologue,
            "the same sequence after a reset returned different bits, so a \
             seed survived the reset and one propagation is reading another's \
             history"
        );

        // Non-vacuity: without the intervening reset the prologue DOES move
        // the answer, so the assertion above is testing the reset and not the
        // fact that the solve converges to the same place from anywhere.
        rhs.reset_for_propagation(BASELINE_ELEMENTS_A, 0.0);
        for tof in PROLOGUE {
            rhs.baseline_state_at_exact(tof);
        }
        let unreset = run_sequence(&mut rhs);
        assert_ne!(
            after_no_prologue, unreset,
            "a prologue left no trace even without a reset, so this test would \
             pass with the seed removed entirely and pins nothing"
        );
    }

    /// Non-vacuity for the test above: prove the memo READ path is live, that
    /// the key discriminates, and that the reset really clears.
    ///
    /// A bit-identity test alone passes just as happily against a memo that is
    /// never consulted. This poisons the cell with a state the conversion
    /// cannot produce and requires the poison to come back out at its own key
    /// and nowhere else.
    #[test]
    fn baseline_exact_memo_is_actually_consulted_and_keyed_on_tof() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let mut rhs = baseline_memo_rhs(BASELINE_ELEMENTS_A, 0.0);

        let tof = 600.0_f64;
        let poison = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_ne!(
            rhs.baseline_state_at_exact(tof).map(f64::to_bits),
            poison.map(f64::to_bits),
            "poison must be a state the conversion cannot return"
        );

        // Direction 1: the memo is read, so the poison is what comes back.
        rhs.baseline_exact_memo.set(Some((tof.to_bits(), poison)));
        assert_eq!(
            rhs.baseline_state_at_exact(tof).map(f64::to_bits),
            poison.map(f64::to_bits),
            "baseline memo read path is dead: the hoist saves nothing"
        );

        // Direction 2: a different TOF misses, recomputes, and takes the cell.
        // The reference is given the seed as it stands BEFORE the production
        // call, because that call consumes it and writes its own.
        rhs.baseline_exact_memo.set(Some((tof.to_bits(), poison)));
        let other_tof = tof + 1.0;
        let seed_before = rhs.baseline_warm_offset.get();
        assert_eq!(
            rhs.baseline_state_at_exact(other_tof).map(f64::to_bits),
            baseline_state_at_exact_unmemoized(&rhs, other_tof, seed_before)
                .0
                .map(f64::to_bits),
            "a different TOF must not be answered from the memo"
        );
        assert_eq!(
            rhs.baseline_exact_memo.get().map(|(key, _)| key),
            Some(other_tof.to_bits()),
            "a miss must write its own key"
        );

        // Direction 3: the reset clears. This is the whole of key
        // completeness, since the key carries no element information.
        rhs.baseline_exact_memo.set(Some((tof.to_bits(), poison)));
        rhs.reset_for_propagation(BASELINE_ELEMENTS_B, 0.0);
        assert!(
            rhs.baseline_exact_memo.get().is_none(),
            "reset_for_propagation must clear the baseline memo: the key does \
             not carry the elements, so a surviving entry is wrong, not cold"
        );
    }

    /// The uncapped kernel, at a fixed sealed driver set, across altitude.
    ///
    /// This is the "before" the exospheric ceiling replaces, pinned so the
    /// justification on `JB2008_EXOSPHERIC_DENSITY_CEILING_KG_M3` cannot go
    /// stale silently. It asserts the two properties that make unbounded
    /// extrapolation indefensible rather than merely uncertain: the profile
    /// FLATTENS into a plateau one to two orders above exospheric hydrogen, and
    /// above ~35,000 km it stops being monotone in altitude, which no atmosphere
    /// does.
    #[test]
    fn jb2008_unbounded_extrapolation_is_why_the_ceiling_exists() {
        fn kernel_at(altitude_km: f64) -> f64 {
            jb2008_density(Jb2008Input {
                mjd_utc: 52_951.003_805_740_744,
                sun_declination_rad: -0.285_987_757_544_287,
                // The sealed Orekit pair, differenced: sat_ra
                // 1.282_118_868_515_03 minus sun_ra 3.046_653_643_566_772.
                hour_angle_rad: 1.282_118_868_515_03 - 3.046_653_643_566_772,
                sat_geocentric_lat_rad: -1.487_718_654_399_9,
                sat_altitude_m: altitude_km * KM_TO_M,
                f10: 91.00,
                f10b: 137.10,
                s10: 108.80,
                s10b: 123.80,
                m10: 116.70,
                m10b: 128.50,
                y10: 168.00,
                y10b: 138.60,
                dst_temperature_correction_k: 43.0,
            })
            .expect("the kernel extrapolates rather than refusing")
        }

        // Inside the validation corpus the model decays steeply, as it should.
        let inside = [(175.0, 6.518e-10), (400.0, 2.682e-12), (1000.0, 2.747e-15)];
        for (altitude_km, expected) in inside {
            let rho = kernel_at(altitude_km);
            assert!(
                (rho / expected - 1.0).abs() < 1e-3,
                "kernel moved inside its validation corpus at {altitude_km} km: {rho:e}"
            );
        }

        // Outside it, the decay stalls. 41,378 km is the ceiling of Part A's
        // transfer apogees.
        let plateau = [
            (2500.0, 7.720e-17),
            (3000.0, 4.070e-17),
            (35_000.0, 4.009e-18),
            (41_378.0, 4.030e-18),
            (100_000.0, 5.453e-18),
        ];
        for (altitude_km, expected) in plateau {
            let rho = kernel_at(altitude_km);
            assert!(
                (rho / expected - 1.0).abs() < 1e-3,
                "extrapolation plateau moved at {altitude_km} km: {rho:e}"
            );
            assert!(
                rho > LightyearRHS::JB2008_EXOSPHERIC_DENSITY_CEILING_KG_M3,
                "ceiling would be inert at {altitude_km} km: {rho:e}"
            );
        }

        // NOT monotone: higher is denser out here.
        assert!(
            kernel_at(100_000.0) > kernel_at(35_000.0),
            "the non-monotonicity that motivates the ceiling has gone away"
        );

        // The AIAA constant-above-2500 convention, read literally, is worse than
        // doing nothing: it would hold the 2500 km value all the way out.
        assert!(
            kernel_at(2500.0) > kernel_at(41_378.0),
            "constant-above-2500 is only worse than the plateau while this holds"
        );
    }

    /// The premise that makes the above-ceiling skip BIT-IDENTICAL, as a
    /// standing tripwire.
    ///
    /// `jb2008_density_at_state` answers `JB2008_EXOSPHERIC_DENSITY_CEILING_KG_M3`
    /// directly above `JB2008_EXTRAPOLATION_CEILING_M` instead of running the
    /// model and then `min`-ing its answer down to that same constant. That is
    /// the same `f64` ONLY while the model exceeds the ceiling everywhere up
    /// there. If a driver-table update, a kernel change or a new profile ever
    /// pushed it below, the skip would silently stop being an optimisation and
    /// start being a physics change — with no digest movement to announce it,
    /// because the skip's whole point is that it moves nothing.
    ///
    /// So the certificate is a test, not a comment. This is the E3 pattern.
    ///
    /// # The exhaustive measurement, and how this is shrunk from it
    ///
    /// `examples/exospheric_ceiling_sweep` is the full version: every one of the
    /// 10,741 driver days the compiled table covers (MJD 50,454-61,194, two
    /// solar cycles) crossed with 10 altitudes and 90 geometries — 9,666,900
    /// samples, 19 s in a release build, far too slow to run in every sweep.
    /// Measured floor **2.351e-18 kg/m^3, 23.5x the ceiling**, on the exact and
    /// the fitted profile alike, with the minimum at 20,000 km under HIGH solar
    /// activity (F10.7 = 188.6).
    ///
    /// This wired version is that sweep on a coarse driver stride plus the
    /// full-resolution neighbourhood of the known minimum — about 49,000
    /// samples, two orders down. Two things make the shrink safe rather than
    /// merely cheap:
    ///
    /// * **The threshold has an order of margin over the grid.** It fires at
    ///   2x the ceiling against a measured floor of 23.5x, so a coarse point
    ///   would have to be 11.75x below its neighbours to hide a violation.
    ///   Exospheric density does not move that far across a 30-day driver
    ///   stride or a 60-degree hour angle; it is a smooth function of both.
    /// * **The minimum's neighbourhood is sampled densely**, so the one region
    ///   the full sweep says is closest to the ceiling is checked at full
    ///   resolution rather than stepped over.
    ///
    /// The second assertion is what keeps the shrink honest: the coarse floor
    /// must still land near the recorded exhaustive floor. A grid that drifted
    /// off the minimum would keep passing the threshold while no longer
    /// measuring the thing the threshold is about.
    #[test]
    fn jb2008_exospheric_ceiling_floor_survives_the_whole_driver_table() {
        /// Recorded by the exhaustive sweep. Not a tolerance — a landmark.
        const EXHAUSTIVE_FLOOR_KG_M3: f64 = 2.351e-18;
        /// Where the exhaustive sweep put the minimum.
        const MINIMUM_MJD: f64 = 52_282.0;
        /// Six hour angles: local noon, dusk, midnight, dawn and between.
        const HOUR_ANGLES_FULL: [f64; 6] = [
            0.0,
            std::f64::consts::FRAC_PI_3,
            2.0 * std::f64::consts::FRAC_PI_3,
            std::f64::consts::PI,
            4.0 * std::f64::consts::FRAC_PI_3,
            5.0 * std::f64::consts::FRAC_PI_3,
        ];
        /// Noon and midnight, the extremes of the diurnal bulge.
        const HOUR_ANGLES_COARSE: [f64; 2] = [0.0, std::f64::consts::PI];

        let ceiling = LightyearRHS::JB2008_EXOSPHERIC_DENSITY_CEILING_KG_M3;
        let ceiling_km = LightyearRHS::JB2008_EXTRAPOLATION_CEILING_M / KM_TO_M;
        let drivers = jb_rs::drivers::compiled_drivers().expect("compiled drivers must load");

        // Altitudes span the ceiling itself to well past Part A's 41,378 km
        // authorized apogee, because the plateau is NOT monotone out there.
        let altitudes_km = [
            ceiling_km + 0.001,
            2_600.0,
            3_000.0,
            5_000.0,
            10_000.0,
            20_000.0,
            35_000.0,
            41_378.0,
            50_000.0,
            100_000.0,
        ];

        let mut floor = f64::INFINITY;
        let mut floor_mjd = f64::NAN;
        let mut floor_altitude_km = f64::NAN;
        let mut samples = 0u64;
        let mut days = 0u64;
        let mut refused = 0u64;
        // Rows that actually reached the `density > 2.0 * ceiling` assertion.
        //
        // `samples` is incremented BEFORE the `kernel_accepts` skip and
        // `refused` counts the rows that were SKIPPED, so neither bounds the
        // number of rows this test compared -- if every sampled row were
        // refused, `samples > 40_000` and `refused > 0` would both still pass
        // with not one density assertion executed. Only the `floor` landmark
        // below catches that today, and it does so incidentally, because
        // `f64::INFINITY < 1.5 * EXHAUSTIVE_FLOOR_KG_M3` is false. Relax that
        // landmark to a one-sided or `is_finite`-guarded form and the test
        // goes vacuous with three counters still green. This counter states
        // the requirement directly instead of relying on that side effect.
        let mut accepted = 0u64;

        for day_offset in 0..40_000_i32 {
            let julian_day = 2_430_000.5_f64 + f64::from(day_offset);
            let Ok(utc_jd) = UtcJulianDay::new(julian_day) else {
                continue;
            };
            let Ok(modified_julian_day) = utc_jd.to_utc_mjd() else {
                continue;
            };
            let Ok(driver) = drivers.lookup_utc_mjd(modified_julian_day) else {
                continue;
            };
            let mjd = modified_julian_day.as_f64();
            days = days.saturating_add(1);

            // Coarse stride everywhere; full geometry near the known minimum.
            let near_minimum = (mjd - MINIMUM_MJD).abs() <= 15.0;
            if !near_minimum && day_offset % 30 != 0 {
                continue;
            }
            let (latitudes, declinations, hour_angles): (&[f64], &[f64], &[f64]) = if near_minimum {
                (
                    &[-1.2, -0.6, 0.0, 0.6, 1.2],
                    &[-0.41, 0.0, 0.41],
                    &HOUR_ANGLES_FULL,
                )
            } else {
                (&[-1.2, 0.0, 1.2], &[-0.41], &HOUR_ANGLES_COARSE)
            };

            for &altitude_km in &altitudes_km {
                for &latitude in latitudes {
                    for &declination in declinations {
                        for &hour_angle in hour_angles {
                            let input = Jb2008Input {
                                mjd_utc: mjd,
                                sun_declination_rad: declination,
                                hour_angle_rad: hour_angle,
                                sat_geocentric_lat_rad: latitude,
                                sat_altitude_m: altitude_km * KM_TO_M,
                                f10: driver.f10,
                                f10b: driver.f10b,
                                s10: driver.s10,
                                s10b: driver.s10b,
                                m10: driver.m10,
                                m10b: driver.m10b,
                                y10: driver.y10,
                                y10b: driver.y10b,
                                dst_temperature_correction_k: f64::from(driver.dtcval),
                            };
                            samples = samples.saturating_add(1);
                            // The skip only claims the rows the kernel ACCEPTS.
                            // Where it refuses, the flown path returns NaN and
                            // the skip must too, which is why it replicates this
                            // exact predicate. Asserting the refusal here is what
                            // keeps the two in step: a kernel that started
                            // refusing on some OTHER condition would fail this
                            // arm rather than quietly leave the skip answering
                            // 1e-19 where production answers NaN.
                            let kernel_accepts = [
                                driver.f10,
                                driver.f10b,
                                driver.s10,
                                driver.s10b,
                                driver.m10,
                                driver.m10b,
                                driver.y10,
                                driver.y10b,
                            ]
                            .iter()
                            .all(|index| *index > 0.0);
                            if !kernel_accepts {
                                refused = refused.saturating_add(1);
                                assert_eq!(
                                    jb2008_density(input),
                                    Err(jb_rs::jb2008::Jb2008Error::NonPositiveSolarIndex),
                                    "a non-positive solar index at MJD {mjd} no longer produces                                      NonPositiveSolarIndex; the above-ceiling skip replicates that                                      exact precondition and is now out of step with the kernel"
                                );
                                continue;
                            }
                            accepted = accepted.saturating_add(1);
                            // BOTH profiles: the ceiling is applied to whichever
                            // one ran, and production flies the fitted one.
                            for density in [
                                jb2008_density(input).expect("an accepted row must evaluate"),
                                jb2008_density_fitted_v7(input).expect("an accepted row must fit"),
                            ] {
                                assert!(
                                    density > 2.0 * ceiling,
                                    "JB2008 came within 2x of the exospheric ceiling above the \
                                     extrapolation ceiling: {density:e} at MJD {mjd}, \
                                     {altitude_km} km, lat {latitude}, dec {declination}, \
                                     hour angle {hour_angle}. The above-ceiling skip in \
                                     `jb2008_density_at_state` is NO LONGER BIT-IDENTICAL and \
                                     must be removed or re-argued before this test is relaxed."
                                );
                                if density < floor {
                                    floor = density;
                                    floor_mjd = mjd;
                                    floor_altitude_km = altitude_km;
                                }
                            }
                        }
                    }
                }
            }
        }

        assert!(
            days > 10_000,
            "only {days} driver days resolved, so this swept a table that is not the \
             one production flies"
        );
        assert!(
            samples > 40_000,
            "only {samples} samples taken; the grid collapsed and this test is vacuous"
        );
        // The real non-vacuity floor: rows that cleared the refusal skip and
        // had their density compared against the ceiling. See `accepted`.
        assert!(
            accepted > 40_000,
            "only {accepted} of {samples} sampled rows reached the ceiling comparison \
             ({refused} refused); the density assertion this test exists for was \
             effectively skipped"
        );
        // The refusal arm must be exercised, or the predicate the skip
        // replicates is untested and this test would keep passing if the two
        // drifted apart. The exhaustive sweep counts 303 refused driver days.
        assert!(
            refused > 0,
            "no refused driver row was sampled, so the NonPositiveSolarIndex arm              this test exists to pin was never reached"
        );
        // The shrink check. A coarse grid that wandered off the minimum would
        // keep clearing the threshold while measuring somewhere else.
        assert!(
            floor < 1.5 * EXHAUSTIVE_FLOOR_KG_M3,
            "coarse floor {floor:e} at MJD {floor_mjd} / {floor_altitude_km} km sits well above \
             the exhaustive sweep's {EXHAUSTIVE_FLOOR_KG_M3:e}: this grid no longer reaches the \
             minimum, so its verdict is about the wrong region. Re-run \
             `examples/exospheric_ceiling_sweep` and re-cut the stride."
        );
        assert!(
            floor > EXHAUSTIVE_FLOOR_KG_M3 / 1.5,
            "coarse floor {floor:e} is far BELOW the exhaustive sweep's \
             {EXHAUSTIVE_FLOOR_KG_M3:e}, which the full grid should have found: the kernel or the \
             driver table has moved and the exhaustive sweep needs re-running."
        );
    }

    /// The ceiling binds above the validity range and is inert below it.
    #[test]
    fn jb2008_exospheric_ceiling_binds_only_above_the_validity_range() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let ceiling = LightyearRHS::JB2008_EXOSPHERIC_DENSITY_CEILING_KG_M3;

        for rhs in [model4_rhs(None), model5_rhs()] {
            // Direction fixed, radius swept, so only altitude changes.
            let density_at_radius = |r_km: f64| {
                let direction: [f64; 3] = [7578.137, 350.0, 700.0];
                let norm = direction[0]
                    .mul_add(
                        direction[0],
                        direction[1].mul_add(direction[1], direction[2] * direction[2]),
                    )
                    .sqrt();
                let scale = r_km / norm;
                let state = [
                    direction[0] * scale,
                    direction[1] * scale,
                    direction[2] * scale,
                    0.0,
                    7.4,
                    0.1,
                ];
                rhs.density_at_state(&state, JD0, &test_frame_rotation(JD0), None)
            };

            // Below the ceiling altitude: the model is returned untouched, and
            // the values are far above the ceiling, so this is not vacuous.
            for r_km in [6578.0, 6778.0, 7378.0, 8378.0] {
                let rho = density_at_radius(r_km);
                assert!(
                    rho.is_finite() && rho > ceiling,
                    "ceiling must not bind at r={r_km} km: rho={rho:e}"
                );
            }

            // Above it: exactly the ceiling, because the model is above it
            // everywhere out here. 47,756 km is Part A's 41,378 km apogee.
            for r_km in [9378.0, 12_000.0, 20_000.0, 47_756.0] {
                let rho = density_at_radius(r_km);
                assert_eq!(
                    rho.to_bits(),
                    ceiling.to_bits(),
                    "ceiling must bind at r={r_km} km: rho={rho:e}"
                );
            }
        }
    }

    #[test]
    fn jb2008_model_code_is_valid() {
        assert!(validate_atmosphere_model_code(4).is_ok());
        assert!(validate_atmosphere_model_code(8).is_ok());
        assert_eq!(
            AtmModel::from_i32(3).expect("model3 remains valid"),
            AtmModel::SyntheticThermosphereProxyV1
        );
    }

    #[test]
    fn part_a_v3_model_uses_persistence_while_model_7_keeps_historical_v2() {
        let v3 = jb2008_driver_authority(8).expect("model 8 is JB2008");
        assert_eq!(v3, Jb2008DriverAuthority::PartAV3PersistenceV1);
        let v3_drivers = v3.load().expect("compiled Part A v3 drivers");
        let expected_v3 =
            jb_rs::drivers::compiled_part_a_v3_drivers().expect("compiled Part A v3 drivers");
        assert!(std::sync::Arc::ptr_eq(&v3_drivers, &expected_v3));

        let v2 = jb2008_driver_authority(7).expect("model 7 is JB2008");
        assert_eq!(v2, Jb2008DriverAuthority::CompiledSetV2);
        let v2_drivers = v2.load().expect("compiled v2 SET drivers");
        let expected_v2 = jb_rs::drivers::compiled_drivers().expect("compiled v2 SET drivers");
        assert!(std::sync::Arc::ptr_eq(&v2_drivers, &expected_v2));
        assert!(!std::sync::Arc::ptr_eq(&v2_drivers, &v3_drivers));

        let historical =
            jb_rs::drivers::UtcJulianDay::new(2_460_310.5).expect("historical UTC epoch");
        v2_drivers
            .validate_utc_arc(historical, historical)
            .expect("model 7 historical v2 epoch remains covered");
    }

    /// Every JB2008 code answers the one predicate, and nothing else does.
    ///
    /// Written after six guards across two crates were found still spelling
    /// `matches!(atm_model, 4 | 5)` — landing model 6 updated `AtmModel` and
    /// left the literal copies behind, so the shipped model stopped tripping a
    /// Coulomb-drag rejection, a UTC driver-arc validation, a dynamic-Sun
    /// requirement, a dual/STM rejection and a driver preflight. This asserts
    /// the codes explicitly rather than deriving them from `AtmModel`, so
    /// adding a variant without deciding whether it is JB2008 goes red here.
    #[test]
    fn every_jb2008_model_code_answers_the_shared_predicate() {
        // 7 moved from the unknown list to here when the fitted kernel landed,
        // and this test is the only thing that said so. It was written on a
        // tree where 7 did not exist, so listing it as unknown was correct
        // then; the fitted kernel landed on a different branch and added the
        // `AtmModel` arm. Neither side touched the other's lines, so the merge
        // reported zero conflicts and produced a tree whose enum called 7
        // JB2008 while this list called it unknown. A textual check that both
        // sides survived cannot see that; only the assertion can.
        for code in [4, 6, 7, 8] {
            assert!(
                atm_model_uses_jb2008_drivers(code),
                "atm_model {code} flies JB2008 and must trip every JB2008 guard"
            );
        }
        assert!(
            atm_model_uses_jb2008_drivers(5),
            "atm_model 5 is the comparison anchor and still flies JB2008 drivers"
        );
        for code in [0, 1, 2, 3] {
            assert!(
                !atm_model_uses_jb2008_drivers(code),
                "atm_model {code} is not JB2008"
            );
        }
        for code in [-1, 9, i32::MAX] {
            assert!(
                !atm_model_uses_jb2008_drivers(code),
                "an unknown atm_model {code} must not be treated as JB2008"
            );
        }
    }

    #[test]
    fn jb2008_density_is_finite_above_legacy_1000km_ceiling() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let rhs = model4_rhs(None);
        let altitude_km = MODEL4_STATE[0]
            .hypot(MODEL4_STATE[1])
            .hypot(MODEL4_STATE[2])
            - satpy_core::RE;
        assert!(
            altitude_km > 1000.0,
            "the fixture must sit above the legacy ceiling this test is named for, got {altitude_km} km"
        );
        let rho = model4_density(&rhs, JD0);

        assert!(rho.is_finite() && rho > 0.0, "rho={rho:?}");
    }

    #[test]
    fn jb2008_density_changes_on_consecutive_julian_days() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let rhs = model4_rhs(None);
        let rho0 = model4_density(&rhs, JD0);
        let rho1 = model4_density(&rhs, JD0 + 1.0);

        assert!(rho0.is_finite() && rho1.is_finite());
        assert_ne!(
            rho0.to_bits(),
            rho1.to_bits(),
            "JB2008 must resolve current drivers and Sun"
        );
    }

    #[test]
    fn jb2008_static_sun_override_cannot_freeze_density() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let explicit_sun = [149_597_870.7, 0.0, 0.0];
        let rhs = model4_rhs(Some(explicit_sun));
        let rho0 = model4_density(&rhs, JD0);
        let rho1 = model4_density(&rhs, JD0 + 1.0);
        let sun0 = rhs.dynamic_body_position(EphemerisBody::Sun, JD0);
        let sun1 = rhs.dynamic_body_position(EphemerisBody::Sun, JD0 + 1.0);

        assert_ne!(rhs.dynamic_ephemeris_flags & ForceFlags::SUN_GRAVITY, 0);
        assert_ne!(sun0.map(f64::to_bits), explicit_sun.map(f64::to_bits));
        assert_ne!(sun0.map(f64::to_bits), sun1.map(f64::to_bits));
        assert_ne!(rho0.to_bits(), rho1.to_bits());
    }

    /// The cache policy must be a pure function of `eps`, not of the sequence
    /// of tolerances this RHS has been adapted to.
    ///
    /// Replaces `jb2008_density_subcycle_is_always_one`, which pinned that
    /// JB2008 was exempt from force sub-cycling. Sub-cycling is gone -- it was
    /// measured never to skip a single evaluation on any integrated path,
    /// because every entry point adapts at eps < 1e-7 and every eps in the tree
    /// is tighter than that -- so an exemption from it no longer says anything.
    ///
    /// What is worth pinning is the defect that outlived it. `baseline_cache_tol`
    /// was updated with `.min()`: a monotone ratchet that made the tolerance a
    /// function of call history, so an RHS adapted to 1e-11 and then to 5e-7
    /// kept the 1e-11 tolerance and integrated differently from a freshly
    /// adapted one. No in-tree caller passed two different tolerances to one
    /// object, which is exactly why it survived unnoticed. This test fails
    /// against that ratchet.
    #[test]
    fn cache_policy_depends_only_on_eps_not_on_call_order() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();

        for target_eps in [5e-7, 5e-8, 1e-9, 1e-11] {
            let mut direct = model4_rhs(None);
            direct.adapt_cache_policy_for_eps(target_eps);

            let mut after_tighter = model4_rhs(None);
            after_tighter.adapt_cache_policy_for_eps(1e-11);
            after_tighter.adapt_cache_policy_for_eps(target_eps);

            let mut after_looser = model4_rhs(None);
            after_looser.adapt_cache_policy_for_eps(1e-3);
            after_looser.adapt_cache_policy_for_eps(target_eps);

            assert_eq!(
                direct.cache_policy_for_test().to_bits(),
                after_tighter.cache_policy_for_test().to_bits(),
                "eps={target_eps:e}: policy retained a previous TIGHTER adapt"
            );
            assert_eq!(
                direct.cache_policy_for_test().to_bits(),
                after_looser.cache_policy_for_test().to_bits(),
                "eps={target_eps:e}: policy retained a previous LOOSER adapt"
            );
        }
    }

    #[test]
    fn baseline_cache_staleness_preserves_finite_boundary_semantics() {
        let tolerance = 1.0_f64;
        let next_beyond = f64::from_bits(tolerance.to_bits() + 1);

        assert!(baseline_cache_is_stale(false, 0.0, 0.0, tolerance));
        assert!(!baseline_cache_is_stale(true, tolerance, 0.0, tolerance));
        assert!(baseline_cache_is_stale(true, next_beyond, 0.0, tolerance));
    }

    #[test]
    fn baseline_cache_staleness_fails_closed_for_unordered_policy_inputs() {
        assert!(baseline_cache_is_stale(true, f64::NAN, 0.0, 1.0));
        assert!(baseline_cache_is_stale(true, 0.0, f64::NAN, 1.0));
        assert!(baseline_cache_is_stale(true, 0.0, 0.0, f64::NAN));
        assert!(baseline_cache_is_stale(true, 0.0, 0.0, -1.0));
    }

    #[test]
    fn event_baseline_calculator_reuses_only_the_exact_time_key() {
        let config = ForceConfig::default();
        let mut rhs = construct(&config).expect("test RHS must construct");
        rhs.adapt_cache_policy_for_eps(1e-8);

        let cache_tol = rhs.cache_policy_for_test();
        assert!(
            cache_tol < 1e-3,
            "hostile case must distinguish adapted policy from the old fixed threshold"
        );

        let calculator = rhs.baseline_calculator();
        let populate_t = calculator.t0_s + 64.0;
        let inside_t = populate_t + cache_tol * 0.25;
        let inside_gap = inside_t - populate_t;

        assert!(inside_gap.abs() <= cache_tol);

        let populated = calculator.get_baseline_state(populate_t);
        let repeated = calculator.get_baseline_state(populate_t);
        assert_eq!(
            repeated.map(f64::to_bits),
            populated.map(f64::to_bits),
            "an identical time key must reuse the exact cached state"
        );

        let inside = calculator.get_baseline_state(inside_t);
        let mut expected_inside = [0.0; 6];
        equinoc2eci_impl(
            &calculator.init_equinoc_state,
            6,
            inside_t - calculator.t0_s,
            0.0,
            &mut expected_inside,
        );
        assert_ne!(
            expected_inside.map(f64::to_bits),
            populated.map(f64::to_bits),
            "hostile nearby query must produce a distinguishable baseline"
        );
        assert_eq!(
            inside.map(f64::to_bits),
            expected_inside.map(f64::to_bits),
            "a distinct time key must recompute its exact baseline"
        );
    }

    #[test]
    fn scalar_jb2008_rejects_coulomb_drag() {
        let config = ForceConfig {
            force_flags: ForceFlags::DRAG | ForceFlags::COULOMB_DRAG,
            atm_model: 4,
            ..ForceConfig::default()
        };
        let error = construct(&config)
            .err()
            .expect("JB2008 plus Coulomb drag must fail closed");

        let message = error.to_string();
        assert!(message.contains("JB2008") && message.contains("Coulomb"));
    }

    #[test]
    fn scalar_jb2008_allows_vern9_and_rkv98() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        for method in [
            crate::types::StepperMethod::Vern9,
            crate::types::StepperMethod::Rkv98,
        ] {
            let config = ForceConfig {
                force_flags: ForceFlags::DRAG,
                atm_model: 4,
                am_ratio: 0.01,
                cd: 2.2,
                integrator_method: method,
                ..ForceConfig::default()
            }
            .with_ephemeris_for_arc(JD0, JD0 + 1.0)
            .expect("JB2008 explicit scalar arc must resolve");
            assert!(construct(&config).is_ok(), "{method:?}");
        }
    }
}

/// Task 5B-2 REDs, IN-CRATE because they observe private state.
///
/// These replace `tests/frame_authority_routing.rs`, now deleted. Do not
/// recreate that file: an integration test links the ordinary rlib and can see
/// neither `#[cfg(test)]` items nor `pub(crate)`, so from outside the crate the
/// only ways to observe the RHS's resolved epoch or rotation are a genuinely
/// `pub` accessor or a cargo feature — both widening production surface purely
/// to look at private state.
///
/// These were originally integration tests in
/// `tests/frame_authority_routing.rs`. Three of the four could never have gone
/// green by routing: each hard-coded the defect as a literal in its own body and
/// then asserted the defect was absent, without calling production at all. An
/// integration test links the ordinary rlib and can see neither `#[cfg(test)]`
/// items nor `pub(crate)`, so the only escapes were a `pub` accessor or a cargo
/// feature — both widening production surface merely to observe internal state.
/// In-crate, the observation point is free, and "which epoch did the RHS
/// resolve" is a private-state question that should not be asked from outside.
#[cfg(test)]
mod frame_authority_routing_reds {
    use super::*;
    use satpy_core::frame_time::authority::tai_seconds_from_utc_jd;
    use satpy_core::frame_time::timescale::{dat, dtf2d_utc, DAYSEC};
    use satpy_core::pack_gravity_coeffs;
    use std::sync::Arc;

    /// UTC Julian Day of a calendar instant, via the sealed time-scale code.
    fn utc_jd(y: i32, m: i32, d: i32, hh: i32, mm: i32, ss: f64) -> f64 {
        let (status, d1, d2) = dtf2d_utc(y, m, d, hh, mm, ss);
        assert_eq!(status, 0, "must be a valid UTC instant");
        d1 + d2
    }

    /// `TAI - UTC` at a calendar day, recomputed from the sealed leap table.
    /// Never a literal 37: a memory-shaped constant is exactly what the standing
    /// rule forbids, and the superseded integration test hard-coded one.
    fn delta_at(y: i32, m: i32, d: i32) -> f64 {
        let (status, value) = dat(y, m, d, 0.0);
        assert!(status >= 0, "TAI-UTC must resolve");
        value
    }

    fn rhs_two_part(utc_jd1: f64, utc_jd2: f64) -> LightyearRHS {
        let stride = 2;
        let c = Arc::new(vec![1.0, 0.0, 0.0, 0.0]);
        let s = Arc::new(vec![0.0; 4]);
        let packed = Arc::new(
            pack_gravity_coeffs(&c, &s, stride, 0)
                .expect("two-part frame-routing test gravity coefficients must pack"),
        );
        let config = Arc::new(ForceConfig {
            sph_order: 0,
            ..ForceConfig::default()
        });
        LightyearRHS::try_new_two_part(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            utc_jd1,
            utc_jd2,
            config,
            packed,
        )
        .expect("RHS constructs")
    }

    fn rhs_at(jd0: f64) -> LightyearRHS {
        let stride = 2;
        let c = Arc::new(vec![1.0, 0.0, 0.0, 0.0]);
        let s = Arc::new(vec![0.0; 4]);
        let packed = Arc::new(
            pack_gravity_coeffs(&c, &s, stride, 0)
                .expect("frame-routing test gravity coefficients must pack"),
        );
        let config = Arc::new(ForceConfig {
            sph_order: 0,
            ..ForceConfig::default()
        });
        LightyearRHS::try_new([6778.0, 0.0, 0.0, 0.0, 7.67, 0.0], 0.0, jd0, config, packed)
            .expect("RHS constructs")
    }

    /// Fast-path bound on rotation-matrix elements: at least five times the
    /// element-equivalent of the segment cache's in-segment residual at 7000 km.
    ///
    /// At the canonical `SEGMENT_WIDTH_S = 1800` that residual is a MEASURED
    /// 0.067382 mm, i.e. 9.6260e-12 in element terms, so this bound sits at
    /// 10.39x. An earlier justification — "five times the 1.9e-11 element
    /// equivalent of the 0.13 mm in-segment residual" — was withdrawn: the
    /// 0.13 mm was never measured, and at the then-current `W = 3600` the true
    /// residual was 0.231948 mm (3.3135e-11), leaving this bound at 3.02x. The
    /// stated reason was false even though the value happened to be adequate;
    /// `W` was halved to restore the reason rather than restating it to fit.
    const FAST_PATH_ELEMENT_BOUND: f64 = 1e-10;

    /// RED 1 — the production rotation must BE the IAU 2006/2000A realization.
    ///
    /// In-crate, and constructed through [`LightyearRHS::try_new_two_part`], for
    /// two reasons. First, after routing, production resolves its rotation
    /// internally via the frame authority and no longer calls `eci2ecef_impl` on
    /// that path at all — so an integration test observing that free function
    /// would be watching something whose result no longer determines what
    /// production does, which is the defect that made three of these REDs
    /// tautological in the first place. Second, the rotation is private state,
    /// and an integration test links the ordinary rlib and can see neither
    /// `#[cfg(test)]` items nor `pub(crate)`.
    ///
    /// The two-part constructor is required, not incidental: collapsing this
    /// epoch into one binary64 costs 1.788139e-5 s = 9.127523 mm at 7000 km,
    /// which is 13x this bound. See `try_new`'s documentation.
    ///
    /// WHAT THIS WOULD FAIL TO CATCH: one epoch and one segment phase. It pins
    /// that the rotation IS the sealed chain there, not that the segment cache
    /// tracks it across a whole segment — `segment_cache_matches_the_exact_chain
    /// _within_the_declared_bound` in `satpy_core` covers that. It also compares
    /// the rotation the RHS resolves, not that every force term consumes it.
    #[test]
    fn production_rotation_is_iau_2006_2000a() {
        use satpy_core::frame_time::chain::{self, EopPolicy, Epoch};

        let finals = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/reference/frame_time/finals2000A.all"),
        )
        .expect("sealed finals2000A.all");

        let epoch = Epoch {
            y: 2022,
            m: 8,
            d: 12,
            hh: 4,
            mm: 25,
            ss: 0.0,
            name: "2022-08-12T04:25:00",
        };
        let (status, d1, d2) = dtf2d_utc(epoch.y, epoch.m, epoch.d, epoch.hh, epoch.mm, epoch.ss);
        assert_eq!(status, 0, "anchor must be a valid UTC instant");

        // Two-part in: no collapse, so the 9.13 mm floor never enters.
        let rhs = rhs_two_part(d1, d2);
        let mut cache = RHSCache::default();
        let produced = rhs
            .frame_rotation_at_checked(0.0, &mut cache)
            .expect("two-part sealed test epoch must resolve frame rotation")
            .r;

        let expected_dd = chain::frame_matrix(&epoch, EopPolicy::Real, 0.0, &finals)
            .expect("sealed frame input resolves");
        let mut worst = 0.0f64;
        for (produced_row, expected_row) in produced.into_iter().zip(expected_dd) {
            for (produced_element, expected_element) in produced_row.into_iter().zip(expected_row) {
                worst = worst.max((produced_element - expected_element.to_f64()).abs());
            }
        }
        let distance_at_7000_km_m = worst * 7.0e6;
        assert!(
            worst <= FAST_PATH_ELEMENT_BOUND,
            "production rotation must match the sealed IAU 2006/2000A chain within \
             {FAST_PATH_ELEMENT_BOUND:e}; worst element difference was {worst:e} \
             (~{distance_at_7000_km_m:.1} m at 7000 km)"
        );
    }

    /// Fixed inputs for the end-to-end golden, shared by the pin and its
    /// sensitivity proof so the two cannot silently drift apart.
    fn golden_inputs() -> (
        [f64; 6],
        f64,
        f64,
        Arc<ForceConfig>,
        Arc<PackedGravityCoeffs>,
    ) {
        let stride = 6;
        let c = Arc::new(vec![
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // degree 0
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // degree 1
            -1.082_63e-3,
            0.0,
            1.574_46e-6,
            0.0,
            0.0,
            0.0, // degree 2
            2.532_44e-6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // degree 3
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // degree 4
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // degree 5
        ]);
        let sc = Arc::new(vec![
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // degree 0
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // degree 1
            0.0,
            0.0,
            -9.038_04e-7,
            0.0,
            0.0,
            0.0, // degree 2
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // degree 3
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // degree 4
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // degree 5
        ]);
        let packed = Arc::new(
            pack_gravity_coeffs(&c, &sc, stride, 4)
                .expect("frame-routing golden gravity coefficients must pack"),
        );
        let config = Arc::new(ForceConfig {
            sph_order: 4,
            eps: 1.0e-10,
            ..ForceConfig::default()
        });
        // 2460310.5 is an exact half-integer JD, so `try_new`'s collapse floor
        // is identically zero here and the golden isolates the ROTATION.
        (
            [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
            2_460_310.5,
            1800.0,
            config,
            packed,
        )
    }

    fn golden_propagate() -> [f64; 6] {
        let (init, jd0, tf_s, config, packed) = golden_inputs();
        let gravity = crate::integrator::ScalarGravityAssets::new(packed);
        let context = crate::integrator::ScalarPropagationContext::new(jd0, config, gravity);
        crate::integrator::integrate_final_checked(
            crate::integrator::ScalarPropagationRequest::new(&context, init, &[tf_s], 0.0, tf_s)
                .with_events(false),
        )
        .expect("golden propagation must succeed")
    }

    /// END-TO-END REGRESSION DETECTOR for the routed frame authority.
    ///
    /// # This is NOT an accuracy claim
    ///
    /// These bits are what THIS code produces, not what is physically correct.
    /// No independent oracle exists for a full routed propagation until Task 6
    /// lands one. Do NOT cite this test as validation of the dynamics: it
    /// detects CHANGE, and change is only a defect if nothing was meant to move.
    /// If a deliberate physics change moves it, re-baseline with the before and
    /// after recorded — do not "fix" the code to preserve the pin.
    ///
    /// # Why it exists
    ///
    /// Dozens of tests execute the routed rotation, and when that rotation moved by
    /// ~31.6 km at 7000 km, exactly ONE changed value — and only incidentally,
    /// because it compares two integrators' accuracy ratios. Every other one is
    /// an invariance test (serial vs parallel, reusable vs fresh, sampled vs
    /// final, round-trip closure) and is structurally blind to a rotation
    /// applied consistently on BOTH sides of its own comparison. That is a gate
    /// that looks strong and verifies nothing. This test is the only one that
    /// can see a silent frame regression.
    ///
    /// Its value-sensitivity is PROVEN by
    /// `end_to_end_golden_is_sensitive_to_a_1e9_radian_rotation_change`. Without
    /// that proof this pin would itself be potentially value-blind.
    ///
    /// # SCOPE — narrower than the name suggests
    ///
    /// `ForceConfig::default()` has `force_flags: 0`, so this exercises
    /// SPHERICAL-HARMONIC GRAVITY AND THE ROTATION ONLY. Third-body, SRP and
    /// drag are all OFF, and the ephemeris is never read.
    ///
    /// That is not a hypothetical gap: flipping the ephemeris lookup from TT to
    /// UTC — a 69.184 s change worth ~70 km of Moon position — left these bits
    /// BIT-IDENTICAL. So this pin cannot detect an ephemeris, SRP or drag
    /// regression, and must not be cited as if it could. It detects exactly what
    /// its sensitivity proof demonstrates: a change in the Earth-fixed rotation.
    ///
    /// A full-force companion is worth having and is deliberately NOT added
    /// here, because the routed tree is currently blocked on an unresolved Vern9
    /// convergence regression and any full-force pin taken now would need
    /// re-baselining once that is understood.
    /// # THE PIN IS PER-BUILD-PROFILE, AND THAT IS NOT COSMETIC
    ///
    /// Optimized builds contract multiply-adds and vectorize the gravity sum
    /// differently, so `golden_propagate` lands on different bits under
    /// `--release` than under the default test profile. Measured spread between
    /// the two profiles, component-wise: 25, 100, 74, 90, 114, 13 ULP.
    ///
    /// That number has to be read against the detection signal. The sensitivity
    /// proof's 1e-9 rad rotation moves the state by 1.09e-10 km (release), and
    /// one ULP at this ~9500 km magnitude is ~1.8e-12 km — so the signal this
    /// pin exists to catch is worth about 60 ULP, and the cross-profile spread
    /// is up to 114 ULP. **The build-profile difference is LARGER than the
    /// regression the pin is designed to detect.**
    ///
    /// So a single tolerance-based pin is not available: any epsilon wide enough
    /// to span both profiles is wider than the defect, and would report green on
    /// exactly the frame regression this test is the only one able to see.
    ///
    /// Bit-exactness is therefore kept, and the profile is selected instead.
    /// Within one profile the computation is deterministic, so the 60 ULP signal
    /// is fully detectable; it is only the comparison ACROSS profiles that is
    /// meaningless. Both baselines are recorded below.
    ///
    /// Consequence for anyone citing this test: it detects a frame regression
    /// **within a fixed build profile on a fixed machine**. Like the original
    /// pin, it assumes the host reproduces the same bits; a different CPU may
    /// contract differently and require re-baselining. Do not cite it as a
    /// machine-independent invariant.
    #[test]
    #[cfg_attr(
        all(not(debug_assertions), not(feature = "bitpin")),
        ignore = "bit pin with debug and release(bitpin) baselines only; this profile's codegen \
                  matches neither capture (fast-test and plain release without the bitpin lane)"
    )]
    fn end_to_end_routed_propagation_matches_its_pinned_state() {
        // Captured from this implementation on 2026-07-24, immediately after the
        // Task 5B-2 routing landed, on macOS/arm64 against Apple's libm.
        // Bit-exact: a rotation regression below f64 rounding is still a
        // regression.
        //
        // The host was not recorded at capture time; the pass/fail pattern
        // establishes it, since a pin over transcendental output can only hold on
        // the libm that produced it. This one holds on macOS and fails on x86_64
        // Linux, where the state differs in the low ~2 hex digits of each word.
        //
        // The doc comment above warns that a different CPU may contract
        // differently. That is real but it is not the whole mechanism, and on the
        // evidence it is not the dominant one: the C library matters more than
        // the ISA. IEEE-754 requires `sqrt` to be correctly rounded and requires
        // nothing of sin/cos/tan/asin/acos/atan/exp/log/pow, so Apple libm and
        // glibc disagree by 1-2 ULP on all of them (measured: 141-1618 of 4000
        // sampled arguments per function) while `sqrt` agrees on 4000 of 4000.
        // No compiler flag reconciles that. Re-baselining is therefore required
        // per LIBM, not merely per CPU -- a Linux/aarch64 host would likely match
        // this Linux baseline, not this macOS one.
        // RE-BASELINED 2026-08-09 for the equinoctial warm seed
        // (`baseline_warm_offset`). Both profiles were re-captured on the same
        // host and in the same session, from the `got` lines of the failing
        // assertion below, before -> after:
        //
        //   DEBUG   0x40c28e113278ec3d -> 0x40c28e113278ec2c
        //           0xc0bf48b65c933c19 -> 0xc0bf48b65c933c04
        //           0xc009ecce37dabaf9 -> 0xc009ecce37dabb33
        //           0x4029d0242f7cf2da -> 0x4029d0242f7cf294
        //           0x40259866ceb81db8 -> 0x40259866ceb81dd3
        //           0xbf9613f302cc261b -> 0xbf9613f302cc260a
        //   RELEASE 0x40c28e113278ec56 -> 0x40c28e113278ec45
        //           0xc0bf48b65c933c7d -> 0xc0bf48b65c933ca0
        //           0xc009ecce37dabaaf -> 0xc009ecce37daba31
        //           0x4029d0242f7cf334 -> 0x4029d0242f7cf33e
        //           0x40259866ceb81d46 -> 0x40259866ceb81d48
        //           0xbf9613f302cc260e -> 0xbf9613f302cc2612
        //
        // CAUSE: the equinoctial longitude solve is now seeded from the
        // previous call's root, and its loop exits on the step rather than on a
        // residual, so it converges to a different last-ULP root. The move is
        // in the low hex digits of each word, which is the size this pin exists
        // to see; the accuracy question is `strict_hf_pin`'s and both of its
        // 1 m gates stayed green across the same change.
        //
        // RE-BASELINED AGAIN 2026-08-09 for the stage-baseline prefill
        // (`LightyearRHS::prefill_stage_baselines`). Both profiles re-captured
        // on the same host and in the same session, from the `got` lines,
        // before -> after:
        //
        //   DEBUG   0x40c28e113278ec2c -> 0x40c28e113278ec36
        //           0xc0bf48b65c933c04 -> 0xc0bf48b65c933bf6
        //           0xc009ecce37dabb33 -> 0xc009ecce37dabb39
        //           0x4029d0242f7cf294 -> 0x4029d0242f7cf2a2
        //           0x40259866ceb81dd3 -> 0x40259866ceb81dd1
        //           0xbf9613f302cc260a -> 0xbf9613f302cc2609
        //   RELEASE 0x40c28e113278ec45 -> 0x40c28e113278ec43
        //           0xc0bf48b65c933ca0 -> 0xc0bf48b65c933c0f
        //           0xc009ecce37daba31 -> 0xc009ecce37dabb19
        //           0x4029d0242f7cf33e -> 0x4029d0242f7cf2bc
        //           0x40259866ceb81d48 -> 0x40259866ceb81db9
        //           0xbf9613f302cc2612 -> 0xbf9613f302cc2600
        //
        // CAUSE: the same seed, shared four ways instead of chained. A pack's
        // four lanes now all start from ONE incoming offset, so three lanes in
        // four begin from a different point than the serial order gave them and
        // the step-exit loop lands on a different last-ULP root. Same size of
        // move, same reason it is not an accuracy verdict: `strict_hf_pin`'s
        // two 1 m gates stayed green across this change too.
        //
        // RE-BASELINED AGAIN 2026-08-11 for the stage-prefill node filter
        // (`integrator.rs::prefill_stage_times`). Both profiles re-captured on
        // the same host and in the same session, before -> after:
        //
        //   DEBUG   0x40c28e113278ec36 -> 0x40c28e113278ec30
        //           0xc0bf48b65c933bf6 -> 0xc0bf48b65c933baa
        //           0xc009ecce37dabb39 -> 0xc009ecce37dabb8b
        //           0x4029d0242f7cf2a2 -> 0x4029d0242f7cf24e
        //           0x40259866ceb81dd1 -> 0x40259866ceb81e23
        //           0xbf9613f302cc2609 -> 0xbf9613f302cc260a
        //   RELEASE 0x40c28e113278ec43 -> 0x40c28e113278ec42
        //           0xc0bf48b65c933c0f -> 0xc0bf48b65c933c78
        //           0xc009ecce37dabb19 -> 0xc009ecce37daba95
        //           0x4029d0242f7cf2bc -> 0x4029d0242f7cf31a
        //           0x40259866ceb81db9 -> 0x40259866ceb81d7b
        //           0xbf9613f302cc2600 -> 0xbf9613f302cc261a
        //
        // CAUSE: the prefill is no longer handed `c[0]` or the duplicate
        // `c[9]`, so eight nodes pack into two x4 solves instead of three and
        // node 8 moves into the first pack -- a different warm-start seed, a
        // different last-ULP root. Same size of move and the same reason it is
        // not an accuracy verdict: `strict_hf_pin`'s two 1 m gates stayed green
        // and its V3 endpoint moved 4 nm, 2.5 million x under its tripwire.
        const PINNED_DEBUG: [u64; 6] = [
            0x40c2_8e11_3278_ec30,
            0xc0bf_48b6_5c93_3baa,
            0xc009_ecce_37da_bb8b,
            0x4029_d024_2f7c_f24e,
            0x4025_9866_ceb8_1e23,
            0xbf96_13f3_02cc_260a,
        ];
        // Captured 2026-07-25 from `cargo test --release` on the same tree and
        // host as PINNED_DEBUG, with no source change between the two captures —
        // the whole delta is optimization-level float contraction. Re-captured
        // together with PINNED_DEBUG on 2026-08-09; see the ledger above.
        const PINNED_RELEASE: [u64; 6] = [
            0x40c2_8e11_3278_ec42,
            0xc0bf_48b6_5c93_3c78,
            0xc009_ecce_37da_ba95,
            0x4029_d024_2f7c_f31a,
            0x4025_9866_ceb8_1d7b,
            0xbf96_13f3_02cc_261a,
        ];
        // HOST AXIS ADDED 2026-08-28: the ledger above already established
        // that these macOS-minted baselines FAIL on x86_64 Linux (per-libm
        // law). The r55t2 TC deep-qualification made TinkerCliffs a gate host
        // for this lane, so Linux is now a first-class baseline instead of a
        // documented failure. Captured on TC (znver2, glibc) through the
        // sealed release wrapper at 9029ba0e, each deterministic across two
        // back-to-back runs, from the failing assertion's own `got` lines:
        //
        //   LINUX DEBUG           0x40c28e113278ec44
        //   (deep-qual lane)      0xc0bf48b65c933c73
        //                         0xc009ecce37dabaaf
        //                         0x4029d0242f7cf310
        //                         0x40259866ceb81d69
        //                         0xbf9613f302cc2614
        //   LINUX RELEASE(bitpin) measured IDENTICAL to LINUX DEBUG, all six
        //                         words — on this host the two profiles
        //                         coincide for this arc. That is a MEASURED
        //                         coincidence, not an assumption: the two
        //                         consts below are kept separate so a future
        //                         divergence re-pins one arm, not a shared
        //                         constant.
        //
        // The macOS baselines above are UNCHANGED — scoped, not re-baselined.
        const PINNED_DEBUG_LINUX: [u64; 6] = [
            0x40c2_8e11_3278_ec44,
            0xc0bf_48b6_5c93_3c73,
            0xc009_ecce_37da_baaf,
            0x4029_d024_2f7c_f310,
            0x4025_9866_ceb8_1d69,
            0xbf96_13f3_02cc_2614,
        ];
        const PINNED_RELEASE_LINUX: [u64; 6] = [
            0x40c2_8e11_3278_ec44,
            0xc0bf_48b6_5c93_3c73,
            0xc009_ecce_37da_baaf,
            0x4029_d024_2f7c_f310,
            0x4025_9866_ceb8_1d69,
            0xbf96_13f3_02cc_2614,
        ];
        // `debug_assertions` still picks WHICH baseline (a debug build is
        // debug codegen whatever features are on), but it no longer decides
        // whether the release arm RUNS: the cfg_attr above ignores every
        // non-debug profile outside the explicit `bitpin` lane, so fast-test
        // (debug-assertions=false, non-release codegen) can no longer reach
        // the release baselines under flags they were never captured on.
        let (pinned, profile) = match (cfg!(target_os = "macos"), cfg!(debug_assertions)) {
            (true, true) => (PINNED_DEBUG, "debug"),
            (true, false) => (PINNED_RELEASE, "release"),
            (false, true) => (PINNED_DEBUG_LINUX, "debug"),
            (false, false) => (PINNED_RELEASE_LINUX, "release"),
        };
        let got = golden_propagate();
        let got_bits = got.map(f64::to_bits);
        assert_eq!(
            got_bits,
            pinned,
            "routed end-to-end propagation moved ({profile} profile).\n  \
             got    {:?}\n  pinned {:?}\n\
             If a deliberate physics change caused this, re-baseline BOTH profile \
             pins with the before and after RECORDED; do not adjust the code to \
             preserve the pin, and do not re-baseline only the profile you ran.",
            got_bits.map(|b| format!("{b:#018x}")),
            pinned.map(|b| format!("{b:#018x}")),
        );
    }

    /// PROVES the golden above is value-SENSITIVE, not merely green.
    ///
    /// Perturbs the resolved rotation by 1e-9 rad — 120x above the segment
    /// cache's own 8.3013e-12 rad residual, and just BELOW the 1.3039e-9 rad
    /// `jd0` collapse floor at 0.77x of it, so it is a realistic silent
    /// regression rather than a caricature — and asserts the propagated state
    /// moves. A pin that does not move under this cannot detect a frame
    /// regression, which is precisely the failure mode of the other 50 tests.
    ///
    /// Until 2026-08-04 this claimed the perturbation was "~1300x SMALLER" than
    /// the collapse floor. It is 0.77x of it. The margin between the probe and
    /// the real error source is a factor of 1.3, not 1300, so the probe sits
    /// just inside realism rather than three orders of magnitude inside it: it
    /// is still the right size, but there is no headroom to spend.
    #[test]
    fn end_to_end_golden_is_sensitive_to_a_1e9_radian_rotation_change() {
        let baseline = golden_propagate();
        TEST_ROTATION_PERTURB_RAD.with(|c| c.set(1.0e-9));
        let perturbed = golden_propagate();
        TEST_ROTATION_PERTURB_RAD.with(|c| c.set(0.0));

        let delta = perturbed
            .into_iter()
            .zip(baseline)
            .map(|(perturbed_element, baseline_element)| {
                (perturbed_element - baseline_element).abs()
            })
            .fold(0.0f64, f64::max);
        assert!(
            delta > 0.0,
            "the end-to-end golden is VALUE-BLIND: a 1e-9 rad rotation change \
             left the propagated state bit-identical, so it cannot detect a \
             silent frame regression"
        );
        println!("golden sensitivity to 1e-9 rad: max|delta| = {delta:.6e}");
    }

    /// PINS the frame-error floor that the frozen single-`f64` [`LightyearRHS::
    /// try_new`] carries, so an invisible limitation becomes a checked property.
    ///
    /// Compares the two constructors at the same instant: `try_new_two_part`
    /// receives the epoch uncollapsed, `try_new` receives `d1 + d2`. The
    /// difference in the resolved TAI instant IS the floor.
    ///
    /// THE MIDNIGHT ROWS ARE THE POINT. Exact UTC midnights are half-integer
    /// Julian Days and therefore exactly representable, so they floor to zero.
    /// A gate sampling only midnight would measure no floor and conclude there
    /// is none — the same shape as a leap-second probe landing exactly on an
    /// interpolation node, where the defect it targets vanishes identically.
    /// Keep BOTH the zero and the non-zero rows; deleting the non-zero ones
    /// silently disarms this.
    ///
    /// WHAT THIS WOULD FAIL TO CATCH: it pins the floor at five epochs, not its
    /// supremum. The true bound is half an ULP of a binary64 at ~2.46e6, i.e.
    /// `2^-32 d = 2.011657e-5 s`; these samples reach 1.788e-5 s, close to but
    /// not at it. It also says nothing about whether production USES the
    /// two-part constructor — only that the two differ as documented.
    #[test]
    fn try_new_single_f64_jd0_carries_a_measured_frame_floor() {
        // Half an ULP at the Julian Day magnitude, the tightest possible bound.
        let half_ulp_s = 2.0f64.powi(-32) * DAYSEC;

        for (y, m, d, hh, mi, ss, expect_zero) in [
            (2022, 8, 12, 4, 25, 0.0, false),
            (2022, 8, 12, 4, 25, 30.5, false),
            (2024, 1, 1, 12, 34, 56.789, false),
            // Exactly-representable half-integer JDs: floor is identically zero.
            (2016, 12, 31, 0, 0, 0.0, true),
            (2017, 1, 1, 0, 0, 0.0, true),
        ] {
            let (status, d1, d2) = dtf2d_utc(y, m, d, hh, mi, ss);
            assert_eq!(status, 0);

            let exact = rhs_two_part(d1, d2)
                .tai_seconds_at(0.0)
                .expect("two-part test epoch must resolve to finite TAI");
            let collapsed = rhs_at(d1 + d2)
                .tai_seconds_at(0.0)
                .expect("collapsed test epoch must resolve to finite TAI");
            let floor_s = collapsed - exact;

            assert!(
                floor_s.abs() <= half_ulp_s,
                "{y}-{m:02}-{d:02}T{hh:02}:{mi:02} floor {floor_s:e} s exceeds half an \
                 ULP {half_ulp_s:e} s at the JD magnitude"
            );
            if expect_zero {
                assert_eq!(
                    floor_s.to_bits(),
                    0_u64,
                    "{y}-{m:02}-{d:02} midnight is an exactly representable half-integer \
                     JD and MUST floor to zero; a gate sampling only midnights would \
                     therefore see no floor at all"
                );
            } else {
                assert!(
                    floor_s != 0.0,
                    "{y}-{m:02}-{d:02}T{hh:02}:{mi:02}:{ss} must NOT be exactly \
                     representable, otherwise this row pins nothing"
                );
            }
        }

        // The worst sampled epoch, pinned in metres so the cost is legible.
        let (_s, d1, d2) = dtf2d_utc(2022, 8, 12, 4, 25, 0.0);
        let floor_s = rhs_at(d1 + d2)
            .tai_seconds_at(0.0)
            .expect("collapsed test epoch must resolve to finite TAI")
            - rhs_two_part(d1, d2)
                .tai_seconds_at(0.0)
                .expect("two-part test epoch must resolve to finite TAI");
        // Earth-rotation rate RECOMPUTED from the sealed `era`, not from a
        // literal ratio: differencing ERA across exactly one second of UT1. A
        // literal was wrong in the last ULP against this value elsewhere in the
        // tree, which is precisely why it is derived here too.
        let omega = {
            use satpy_core::frame_time::dd::from;
            use satpy_core::frame_time::era::era;
            use satpy_core::frame_time::timescale::DJM0;
            let a = era(from(DJM0), from(60_000.0));
            let b = era(from(DJM0), from(60_000.0 + 1.0 / DAYSEC));
            b.to_f64() - a.to_f64()
        };
        let mm_at_7000km = floor_s.abs() * omega * 7.0e9;
        assert!(
            (mm_at_7000km - 9.127_523).abs() < 1.0e-5,
            "2022-08-12T04:25 floor must be 9.127523 mm at 7000 km; got {mm_at_7000km}"
        );
    }

    /// RED 2 — elapsed integration time must be continuous TAI.
    ///
    /// A UTC day is 86400 s but the 2016-12-31 leap day is 86401 TAI seconds, so
    /// deriving the epoch as `jd0 + t/86400` gains a whole second across it.
    ///
    /// WHAT THIS WOULD FAIL TO CATCH: it only exercises an interval that SPANS a
    /// leap. An epoch derivation that is wrong by a constant offset, or wrong in
    /// a way that cancels over this particular interval, passes here. It also
    /// says nothing about the rotation built FROM the epoch — a correct instant
    /// fed to `R3(GMST)` still passes. RED 1 covers that half; the two are only
    /// jointly sufficient. It cannot detect an error below `1.0e-6` s, and in
    /// particular it cannot see `jd0`'s own `2^-31 d` floor, which is why the
    /// residual shortfall here is 0.9999945 s rather than exactly 1 s.
    #[test]
    fn integration_time_is_continuous_tai_across_the_2016_leap() {
        let jd0 = utc_jd(2016, 12, 31, 0, 0, 0.0);
        let rhs = rhs_at(jd0);

        let (b1, f1) = LightyearRHS::split_jd(utc_jd(2017, 1, 1, 0, 0, 0.0));
        let expected = tai_seconds_from_utc_jd(b1, f1).expect("end epoch in sealed span");

        // The leap day really is 86401 TAI seconds long; assert that first so a
        // failure below cannot be blamed on the premise.
        let (b0, f0) = LightyearRHS::split_jd(jd0);
        let start = tai_seconds_from_utc_jd(b0, f0).expect("start epoch in sealed span");
        let elapsed = expected - start;
        assert_eq!(
            elapsed.to_bits(),
            86_401.0_f64.to_bits(),
            "the 2016 leap day must be 86401 TAI seconds"
        );

        let resolved = rhs
            .tai_seconds_at(86401.0)
            .expect("leap-crossing test epoch must resolve to finite TAI");
        assert!(
            (resolved - expected).abs() < 1.0e-6,
            "integrating 86401 s from {jd0} must resolve to 2017-01-01T00:00:00 TAI; \
             resolved {resolved}, expected {expected}, off by {} s",
            resolved - expected
        );
    }

    /// RED 3 — the ephemeris lookup argument must be in the scale the loaded
    /// table DECLARES.
    ///
    /// Asserts the PROPERTY, not the answer. The manifest is read at test time
    /// and the required scale derived from it, so regenerating the tables on a
    /// TDB grid flips this test's requirement automatically rather than leaving
    /// a hard-coded expectation that silently stops matching the data.
    ///
    /// This test previously asserted TT unconditionally. That was wrong: these
    /// tables declare `epoch_scale: utc`, so a TT argument indexes ~69.184 s off
    /// — about 70 km of Moon and 2062 km of Sun position. The error was in the
    /// SPECIFICATION, not the implementation that satisfied it, which is why it
    /// survived review; encoding the dependency is what stops it recurring.
    ///
    /// WHAT THIS WOULD FAIL TO CATCH: it checks the epoch the RHS RESOLVES for
    /// the lookup, not that `get_position` receives it. It also trusts the
    /// manifest to describe the bytes; that link is verified separately by
    /// comparison against the sealed JPL Horizons fixture, quoted in
    /// `ephemeris_lookup_jd_at`'s documentation. And it recognises only the
    /// scales enumerated below — a table declaring anything else fails loudly
    /// rather than silently passing.
    #[test]
    fn ephemeris_lookup_scale_matches_the_table_manifest() {
        let manifest_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/ephemeris/manifest.json");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("manifest"))
                .expect("manifest parses");
        let declared = manifest
            .get("epoch_scale")
            .and_then(serde_json::Value::as_str)
            .expect("manifest must declare epoch_scale")
            .to_ascii_lowercase();

        let jd0 = utc_jd(2022, 8, 12, 4, 25, 0.0);
        let rhs = rhs_at(jd0);
        let resolved = rhs.ephemeris_lookup_jd_at(0.0, None);

        let expected = match declared.as_str() {
            "utc" | "ut" => jd0,
            "tt" | "tdb" => jd0 + (delta_at(2022, 8, 12) + 32.184) / DAYSEC,
            other => panic!(
                "unrecognised epoch_scale {other:?} in {}; add it here deliberately \
                 rather than letting the lookup default to one",
                manifest_path.display()
            ),
        };
        assert!(
            (resolved - expected).abs() < 1.0e-9,
            "manifest declares epoch_scale {declared:?}, so the lookup argument must \
             be {expected}; resolved {resolved}, off by {} s",
            (resolved - expected) * DAYSEC
        );
    }

    /// RED 4 — the JB2008 driver lookup must receive UTC derived from the RHS's
    /// TAI instant.
    ///
    /// Epoch chosen so the two disagree on which DRIVER DAY is selected. At
    /// `t = 86400` from 2016-12-31T00:00:00 the true instant is one second
    /// before 2017-01-01T00:00:00 TAI, i.e. the leap second 2016-12-31T23:59:60,
    /// whose UTC MJD is 57753. Today's `jd0 + 1.0` lands on 2017-01-01T00:00:00,
    /// UTC MJD 57754 — a different driver record.
    ///
    /// EPOCH-SENSITIVE BY CONSTRUCTION — DO NOT "SIMPLIFY" THE 86400.
    /// `t = 86401` would NOT work: there both the current and the true instant
    /// fall on MJD 57754, the lookup returns the SAME driver record, and the test
    /// passes today while proving nothing. Rounding 86400 up to the tidier 86401,
    /// or moving this to a non-leap epoch, silently disarms it. That is the
    /// node-aligned leap probe in a new costume: the declared post-leap gate
    /// epoch sat exactly ON an interpolation node, the one instant in three days
    /// where the defect it targeted vanished identically.
    ///
    /// WHAT THIS WOULD FAIL TO CATCH: it resolves the driver DAY only, so an
    /// error smaller than one day in the UTC conversion passes. Pre-routing it
    /// is red only because elapsed time is UTC-shaped; once routing lands it
    /// becomes a regression guard on the TAI to UTC conversion rather than a
    /// test of the conversion's precision. It observes the resolved JD, not the
    /// `UtcJulianDay` the lookup is actually constructed with.
    #[test]
    fn jb_driver_lookup_uses_utc_of_the_tai_instant() {
        let jd0 = utc_jd(2016, 12, 31, 0, 0, 0.0);
        let rhs = rhs_at(jd0);

        let resolved_mjd = (rhs.driver_utc_jd_at(86400.0, None) - 2_400_000.5).floor();
        assert_eq!(
            resolved_mjd.to_bits(),
            57_753.0_f64.to_bits(),
            "at t=86400 s the true instant is the 2016-12-31T23:59:60 leap second, \
             UTC MJD 57753; the driver lookup resolved MJD {resolved_mjd}"
        );
    }
}

/// The supremum bound must dominate the exact path sum it replaced.
///
/// The oracle here is the SUPERSEDED implementation, verbatim: sum the exact
/// great-circle steps between every crossed grid node. That algorithm was
/// correct and tight, and it is retained as the reference precisely because the
/// new bound's whole soundness claim is `supremum >= exact` on every interval —
/// a claim a test can only make against something that computes the exact
/// value. Nothing in production calls the oracle.
#[cfg(test)]
mod supremum_dominates_exact_path {
    use super::*;
    use crate::types::{ForceConfig, ForceFlags};
    use num_traits::ToPrimitive;
    use std::sync::Arc;

    const JD0: f64 = 2_459_600.5;

    /// Guarded for the same reason as `jb2008_rhs`: this fixture reads the
    /// process-global catalogue and must not observe a conflicting temp one.
    fn sun_rhs() -> LightyearRHS {
        let _ephemeris_guard = crate::precomputed_ephem::ephemeris_test_guard();
        let config = ForceConfig {
            sph_order: 0,
            force_flags: ForceFlags::DRAG,
            atm_model: 4,
            am_ratio: 0.01,
            cd: 2.2,
            sun_pos: None,
            ..ForceConfig::default()
        }
        .with_ephemeris_for_arc(JD0, JD0 + 2.0)
        .expect("JB2008 test arc must resolve");
        let c = vec![1.0; 1];
        let s = vec![0.0; 1];
        let packed = Arc::new(satpy_core::pack_gravity_coeffs(&c, &s, 1, 0).expect("pack"));
        LightyearRHS::try_new(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            JD0,
            Arc::new(config),
            packed,
        )
        .expect("scalar JB2008 RHS must construct")
    }

    fn unit(
        ephemeris: &crate::precomputed_ephem::PrecomputedEphemeris,
        jd: f64,
    ) -> Result<[f64; 3], EclipseError> {
        let utc = UtcJulianDay::new(jd).map_err(|_| EclipseError::Geometry)?;
        let position = ephemeris
            .position_at_part_a_utc_jd(utc)
            .map_err(|_| EclipseError::Geometry)?;
        let norm_sq = position
            .iter()
            .map(|component| component * component)
            .sum::<f64>();
        if !(norm_sq.is_finite() && norm_sq > 0.0) {
            return Err(EclipseError::Geometry);
        }
        let inverse_norm = 1.0 / norm_sq.sqrt();
        Ok([
            position[0] * inverse_norm,
            position[1] * inverse_norm,
            position[2] * inverse_norm,
        ])
    }

    fn separation(left: [f64; 3], right: [f64; 3]) -> Result<f64, EclipseError> {
        let cross = [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ];
        let cross_norm = cross
            .iter()
            .map(|component| component * component)
            .sum::<f64>()
            .sqrt();
        let dot = left[0].mul_add(right[0], left[1].mul_add(right[1], left[2] * right[2]));
        let value = cross_norm.atan2(dot);
        value
            .is_finite()
            .then_some(value)
            .ok_or(EclipseError::Geometry)
    }

    /// The superseded bound, verbatim.
    fn exact_path_sum(rhs: &LightyearRHS, t_a: f64, t_b: f64) -> Result<f64, EclipseError> {
        let ephemeris = rhs
            .dynamic_ephemeris
            .as_deref()
            .and_then(|all| all.get(EphemerisBody::Sun))
            .ok_or(EclipseError::Geometry)?;
        let jd_a = rhs.ephemeris_lookup_jd_at(t_a, None);
        let jd_b = rhs.ephemeris_lookup_jd_at(t_b, None);
        let cadence = ephemeris.dt_days();
        let forward = jd_b >= jd_a;
        let (jd_min, jd_max) = if forward { (jd_a, jd_b) } else { (jd_b, jd_a) };
        let (grid_start, _) = ephemeris.jd_range();
        let first_index = ((jd_min - grid_start) / cadence)
            .floor()
            .to_isize()
            .and_then(|index| index.checked_add(1))
            .ok_or(EclipseError::Geometry)?;
        let last_index = ((jd_max - grid_start) / cadence)
            .ceil()
            .to_isize()
            .and_then(|index| index.checked_sub(1))
            .ok_or(EclipseError::Geometry)?;
        let mut previous = unit(ephemeris, jd_a)?;
        let mut total = 0.0;
        let accumulate =
            |jd: f64, previous: &mut [f64; 3], total: &mut f64| -> Result<(), EclipseError> {
                let next = unit(ephemeris, jd)?;
                *total += separation(*previous, next)?;
                *previous = next;
                Ok(())
            };
        if forward {
            for index in first_index..=last_index {
                let index = index.to_f64().ok_or(EclipseError::Geometry)?;
                let jd = grid_start + index * cadence;
                if jd > jd_a && jd < jd_b {
                    accumulate(jd, &mut previous, &mut total)?;
                }
            }
        } else {
            for index in (first_index..=last_index).rev() {
                let index = index.to_f64().ok_or(EclipseError::Geometry)?;
                let jd = grid_start + index * cadence;
                if jd < jd_a && jd > jd_b {
                    accumulate(jd, &mut previous, &mut total)?;
                }
            }
        }
        accumulate(jd_b, &mut previous, &mut total)?;
        Ok(total)
    }

    /// Shortest interval on which the ORACLE is trustworthy, in seconds.
    ///
    /// The oracle's `atan2(|a x b|, a . b)` is evaluated between two unit
    /// vectors whose separation shrinks with the interval, so its cross product
    /// is a cancellation. Measured against the linear-in-`dt` reference the
    /// oracle itself establishes at `dt = 600 s`, it holds six digits at
    /// `dt = 1 s`, runs 0.18% HIGH at `dt = 1e-2 s`, 19.5% LOW at `dt = 1e-4 s`,
    /// and returns exactly 0.0 at `dt = 1e-7 s`. Below this cutoff the oracle is
    /// not a reference for anything, which is what
    /// `superseded_path_sum_collapses_where_the_supremum_does_not` records.
    const ORACLE_TRUSTWORTHY_ABOVE_S: f64 = 1.0;

    /// Corpus over the range where the oracle can referee: from one second up to
    /// intervals long enough to straddle grid nodes, in both time directions.
    /// The node-straddling cases are the ones the superseded implementation
    /// needed its interior loop for and the supremum handles with no special
    /// case at all.
    ///
    /// Shorter intervals are deliberately absent. They are not a gap in the
    /// argument: the supremum is exactly linear in `dt` and the true path is the
    /// integral of a rate this dominates, so establishing the rate is correct at
    /// ANY scale establishes the bound at every scale. Only the rate can be
    /// wrong, and the rate is what a long interval measures best.
    fn corpus() -> Vec<(f64, f64)> {
        let mut cases = Vec::new();
        // One grid interval is a day; the arc starts at JD0 = a grid node.
        for span in [
            1.0_f64, 100.0, 3_600.0, 43_200.0, 86_400.0, 90_000.0, 172_800.0,
        ] {
            let mut bases = vec![0.0_f64, 1.0, 12_345.0, 43_200.0, 86_399.0, 86_400.5];
            // Walk a full year either side of the arc. The supremum is a single
            // number for the whole grid, so the case that can break it is the
            // one nearest perihelion, where the Sun's apparent rate peaks — and
            // the case that shows how loose it can get is aphelion. Sampling
            // only around the arc's own epoch would test neither.
            for month in -12_i32..=12 {
                bases.push(f64::from(month) * 30.0 * 86_400.0);
            }
            for base in bases {
                cases.push((base, base + span));
                cases.push((base + span, base));
            }
        }
        for (t_a, t_b) in &cases {
            assert!((t_b - t_a).abs() >= ORACLE_TRUSTWORTHY_ABOVE_S);
        }
        cases
    }

    #[test]
    fn supremum_bound_is_never_below_the_exact_great_circle_path() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let rhs = sun_rhs();
        let mut compared = 0_usize;
        let mut worst_ratio = 0.0_f64;
        let mut node_straddling = 0_usize;
        for (t_a, t_b) in corpus() {
            let Ok(exact) = exact_path_sum(&rhs, t_a, t_b) else {
                continue;
            };
            let bound = rhs
                .eclipse_sun_direction_path_bound(t_a, t_b)
                .expect("supremum bound must resolve wherever the exact sum does");
            assert!(
                bound >= exact,
                "supremum bound {bound} is BELOW the exact path {exact} on [{t_a}, {t_b}] — \
                 the bound is unsound and every eclipse prune built on it is void"
            );
            if (t_b - t_a).abs() > 86_400.0 {
                node_straddling += 1;
            }
            if exact > 0.0 {
                worst_ratio = worst_ratio.max(bound / exact);
            }
            compared += 1;
        }
        assert!(
            compared >= 400,
            "corpus degenerated to {compared} comparisons; the test is not exercising the bound"
        );
        assert!(
            node_straddling >= 10,
            "only {node_straddling} intervals straddle a grid node; the case the old \
             implementation needed an interior loop for is not covered"
        );
        assert!(
            worst_ratio < 1.10,
            "supremum is {worst_ratio}x the exact path — looser than the catalogue's \
             perihelion-to-aphelion rate spread can explain, so the supremum is wrong"
        );
        println!("compared={compared} node_straddling={node_straddling} worst_ratio={worst_ratio}");
    }

    /// Non-vacuity: the assertion above must be able to FAIL. Shrinking the
    /// supremum below the grid's true rate has to break domination on the very
    /// same corpus, or the comparison is proving nothing about the bound.
    #[test]
    fn a_deflated_supremum_is_caught_by_the_same_comparison() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let rhs = sun_rhs();
        let ephemeris = rhs
            .dynamic_ephemeris
            .as_deref()
            .and_then(|all| all.get(EphemerisBody::Sun))
            .expect("sun table");
        let rate = ephemeris.max_direction_rate_per_day();
        assert!(
            rate.is_finite() && rate > 0.0,
            "the Sun grid must have a finite positive direction rate, got {rate}"
        );
        let deflated = rate * 0.5;
        let mut violations = 0_usize;
        for (t_a, t_b) in corpus() {
            let Ok(exact) = exact_path_sum(&rhs, t_a, t_b) else {
                continue;
            };
            let bound = deflated * (t_b - t_a).abs() / 86_400.0;
            if bound < exact {
                violations += 1;
            }
        }
        assert!(
            violations > 0,
            "halving the supremum violated domination on ZERO corpus intervals, so the \
             domination assertion cannot distinguish a correct supremum from a wrong one"
        );
    }

    /// Where the oracle collapses, the supremum does not — and that direction is
    /// the safe one.
    ///
    /// The eclipse scan subdivides until its whole motion bound is under
    /// `MAX_BOUNDARY_SEPARATION_KM`, one millimetre, which at orbital speed
    /// means intervals near 1e-7 s. That is four orders BELOW where the
    /// superseded path sum still resolved anything: its two unit vectors are
    /// then identical to within roundoff, the cross product cancels to nothing,
    /// and the bound it returned for the Sun axis was exactly 0.0. So at the
    /// depth where brackets are actually accepted, the term the old form
    /// contributed had silently vanished.
    ///
    /// The supremum cannot do that — it is a rate times a time, with no
    /// cancellation anywhere — and it reports a small positive angle there
    /// instead. A larger bound subdivides more and prunes less, so this replaces
    /// a term that had degraded to zero with one that is correct, in the
    /// conservative direction. It is recorded as a test because it is the one
    /// place the two forms differ by more than their looseness ratio.
    #[test]
    fn superseded_path_sum_collapses_where_the_supremum_does_not() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let rhs = sun_rhs();
        let deep = 1.0e-7_f64;
        let superseded = exact_path_sum(&rhs, 0.0, deep).expect("oracle resolves");
        assert_eq!(
            superseded.to_bits(),
            0.0_f64.to_bits(),
            "the superseded path sum no longer collapses at dt={deep} s, so this record \
             is stale and the comparison above should be extended down to it"
        );
        let bound = rhs
            .eclipse_sun_direction_path_bound(0.0, deep)
            .expect("supremum resolves");
        assert!(
            bound > 0.0,
            "the supremum collapsed to {bound} at dt={deep} s as well; it must not, or it \
             carries the same silent hole at the depth brackets are accepted"
        );
        // Linear in dt, so the deep value is the rate scaled down and nothing else.
        let rate_per_second = rhs
            .dynamic_ephemeris
            .as_deref()
            .and_then(|all| all.get(EphemerisBody::Sun))
            .expect("sun table")
            .max_direction_rate_per_day()
            / 86_400.0;
        let expected = rate_per_second * deep;
        assert!(
            bound >= expected && bound <= expected * 1.000_001,
            "deep-interval bound {bound} is not the rate scaled to {deep} s ({expected})"
        );
    }

    /// The cheap range verdict must equal the interpolating entry point's.
    #[test]
    fn admits_jd_matches_position_at_part_a_utc_jd() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let rhs = sun_rhs();
        let ephemeris = rhs
            .dynamic_ephemeris
            .as_deref()
            .and_then(|all| all.get(EphemerisBody::Sun))
            .expect("sun table");
        let (start, end) = ephemeris.jd_range();
        let mut probes = vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            start,
            end,
            start - 1.0e-9,
            end + 1.0e-9,
            start - 1.0,
            end + 1.0,
            f64::from_bits(start.to_bits() - 1),
            f64::from_bits(end.to_bits() + 1),
        ];
        for step in 0..64 {
            probes.push(start + (end - start) * f64::from(step) / 63.0);
        }
        for jd in probes {
            let interpolating = UtcJulianDay::new(jd)
                .ok()
                .and_then(|utc| ephemeris.position_at_part_a_utc_jd(utc).ok())
                .is_some();
            assert_eq!(
                ephemeris.admits_part_a_utc_jd(jd),
                interpolating,
                "range verdict diverged from the interpolating entry point at jd={jd}"
            );
        }
    }

    /// The validated span serves the admission verdict without changing one
    /// bit of the bound, and it never answers for a time the exact check would
    /// reject.
    #[test]
    fn admit_span_reuse_is_bit_identical_and_still_rejects_out_of_range() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let rhs = sun_rhs();
        assert!(
            rhs.admit_span_for_test().is_none(),
            "a fresh RHS must start with no validated span"
        );
        let cold = rhs
            .eclipse_sun_direction_path_bound(100.0, 200.0)
            .expect("in-range interval resolves");
        let (lo, hi) = rhs.admit_span_for_test().expect("the query seeds the span");
        assert!(
            lo <= 100.0 && hi >= 200.0,
            "span ({lo}, {hi}) does not cover the endpoints it just validated"
        );
        let warm = rhs
            .eclipse_sun_direction_path_bound(100.0, 200.0)
            .expect("span-covered interval resolves");
        assert_eq!(
            cold.to_bits(),
            warm.to_bits(),
            "the span fast path changed the bound's bits"
        );
        // An interior interval through the warm span against a fresh RHS that
        // must take the exact per-call check: same bits.
        let fresh = sun_rhs();
        let slow = fresh
            .eclipse_sun_direction_path_bound(120.0, 180.0)
            .expect("exact check resolves");
        let fast = rhs
            .eclipse_sun_direction_path_bound(120.0, 180.0)
            .expect("span-covered interior resolves");
        assert_eq!(
            slow.to_bits(),
            fast.to_bits(),
            "interior fast-path bound differs from the exact-check bound"
        );
        // A time beyond the table still errors, and the failed query must not
        // have grown the span toward it.
        let ephemeris = rhs
            .dynamic_ephemeris
            .as_deref()
            .and_then(|all| all.get(EphemerisBody::Sun))
            .expect("sun table");
        let (_, end) = ephemeris.jd_range();
        let t_out = (end - JD0 + 10.0) * 86_400.0;
        assert!(
            rhs.eclipse_sun_direction_path_bound(t_out, t_out + 1.0)
                .is_err(),
            "a time ten days past the table's end must keep failing closed"
        );
        let (lo_after, hi_after) = rhs.admit_span_for_test().expect("span survives");
        assert!(
            hi_after < t_out,
            "the rejected query leaked into the span: ({lo_after}, {hi_after}) reaches {t_out}"
        );
    }

    /// A time whose resolved JD clears the table but not the five-second
    /// margin must resolve through the exact check WITHOUT entering the span:
    /// the margin is what the interior-coverage argument rests on, so a
    /// margin-failing time in the span would be the unsound state.
    #[test]
    fn near_edge_time_resolves_but_never_enters_the_span() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let rhs = sun_rhs();
        let ephemeris = rhs
            .dynamic_ephemeris
            .as_deref()
            .and_then(|all| all.get(EphemerisBody::Sun))
            .expect("sun table");
        let (_, end) = ephemeris.jd_range();
        // Two seconds inside the table's end: admitted, but 5 s of margin
        // cannot clear the edge.
        let t_edge = (end - JD0) * 86_400.0 - 2.0;
        rhs.eclipse_sun_direction_path_bound(t_edge - 1.0, t_edge)
            .expect("a time two seconds inside the table must still resolve");
        if let Some((lo, hi)) = rhs.admit_span_for_test() {
            assert!(
                !(lo <= t_edge && t_edge <= hi),
                "a margin-failing time entered the span ({lo}, {hi})"
            );
        }
    }

    /// The span refuses to grow past its 64-day cap — the cap is what keeps
    /// the one-leap-second excursion argument a one-event argument — and
    /// queries beyond a capped span still resolve through the exact check.
    #[test]
    fn admit_span_growth_stops_at_the_cap() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let rhs = sun_rhs();
        rhs.eclipse_sun_direction_path_bound(0.0, 1.0)
            .expect("seed resolves");
        let far = 100.0 * 86_400.0;
        rhs.eclipse_sun_direction_path_bound(far, far + 1.0)
            .expect("a 100-day-out interval still resolves via the exact check");
        let (lo, hi) = rhs.admit_span_for_test().expect("span exists");
        assert!(
            hi - lo <= LightyearRHS::ADMIT_SPAN_MAX_S,
            "span ({lo}, {hi}) grew past the cap"
        );
    }
}
