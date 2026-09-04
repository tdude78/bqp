//! Production JB2008 adapter conformance: the altitude handed to the kernel
//! must be the WGS84 ELLIPSOIDAL altitude, not `|r| - a`.
//!
//! # What this closes
//!
//! `jb2008_eci_oracle.rs` declares its own scope as
//! `"...Rust primitive-kernel conformance only; production Rust adapter
//! comparison deferred"`, and its conformance test feeds the fixture's OWN
//! pre-mapped `jb_primitive_inputs` straight into `jb_rs::jb2008_density`.
//! The ECI -> (RA, latitude, altitude) adapter in `LightyearRHS` therefore has
//! no coverage at all. This file covers it. When the production fix lands, the
//! `claim_scope` string in that fixture should be narrowed to match reality.
//!
//! # THE LATITUDE STAYS GEOCENTRIC. Do NOT assert the fixture's column here.
//!
//! This is about the reference SURFACE, and is a different question from the
//! reference FRAME settled two sections below. Both are deliberate and they are
//! settled by different authorities: the surface by Bowman, the frame by
//! agreement between Bowman and Orekit.
//!
//! Bowman's original JB2008 defines `SAT(2)` as **geocentric** latitude:
//!
//! ```text
//! SAT(1) : Right Ascension of Position (radians)
//! SAT(2) : Geocentric Latitude of Position (radians)
//! SAT(3) : Height of Position (km)
//! ```
//!
//! (Bowman, Tobiska, Marcos, Huang, Lin, Burke, "A New Empirical Thermospheric
//! Density Model JB2008 Using New Solar and Geomagnetic Indices", AIAA/AAS
//! 2008-6438; header preserved verbatim in the published `JB2008.f` ports.)
//!
//! Production already passes geocentric latitude and is CORRECT there. The
//! sealed Orekit fixture's `satellite_geodetic_latitude_rad_as_satLat` records
//! that OREKIT deviates from Bowman on this argument -- it is not a statement
//! about what the model wants. Asserting the fixture's latitude column here
//! would drag production off Bowman and onto Orekit's deviation, introducing a
//! second defect opposite in sign to the one this file exists to catch. It is
//! worth only ~0.1% in density either way; the altitude is worth 20-60%.
//!
//! # THE ANGULAR ARGUMENTS ARE HELD FIXED, NOT IGNORED
//!
//! An earlier revision of this comment claimed `sat_ra_rad` "must stay an ECI
//! right ascension: the kernel consumes only `h = sat_ra - sun_ra`, so a common
//! z-rotation cancels exactly and there is nothing to fix". **The premise is
//! true and the conclusion is false** — the same trap the altitude comment in
//! `rhs.rs` already records for `z`. `to_itrs` is the full IAU 2006/2000A chain,
//! `RPOM * R3(ERA) * RC2I`, and only `R3(ERA)` is a rotation about z. The RA
//! difference is invariant to FIRST order, not exactly; the residual is 6.6e-4
//! rad of hour angle at the epoch used here.
//!
//! Production now reduces all four angular arguments from ITRS, and
//! `jb2008_angular_frame_consistency.rs` is the gate for that. This file
//! MIRRORS that choice so that the altitude is the only quantity under test.
//! If the two files disagree about the angular frame, this gate stops isolating
//! the altitude and starts misattributing an angular change as an altitude
//! defect — which is exactly what happened when the angular fix landed:
//! it reported "production fed JB2008 a spherical altitude" at ratio 1.000388,
//! when the altitude was untouched and correct.
//!
//! `angular_frame_matches_production_so_the_altitude_is_isolated` pins the
//! mirror so that divergence fails loudly instead of being misread.
//!
//! # The Earth radius is `ForceConfig::earth_radius`, NOT the gravity constant
//!
//! Three radii are in play and only one is right for a geodetic reduction:
//!
//! | constant | value (km) | role |
//! |---|---|---|
//! | `GRAVITY_REFERENCE_RADIUS_KM` | 6378.13646 | DIR-R6 potential reference. NOT geometry. |
//! | `satpy_core::RE` | 6378.137 | rounded WGS84 |
//! | `ForceConfig::earth_radius` | 6378.137 | "geometry that reduces a position to an altitude" |
//!
//! `ForceConfig::earth_radius` is the one, per its own doc comment. Reaching
//! for the gravity constant because it looks more precise would inject a 54 cm
//! systematic into an altitude -- a miniature of the defect being fixed. The
//! first test below pins this against the sealed fixture's own ellipsoid so the
//! choice cannot drift silently.

use std::sync::Arc;

use anyhow::Context;
use jb_rs::drivers::{compiled_drivers, UtcJulianDay};
use jb_rs::jb2008::{jb2008_density, Jb2008Input};
use lightyear_odeint_rs::precomputed_ephem::{get_precomputed_ephemeris, Body};
use lightyear_odeint_rs::rhs::LightyearRHS;
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags};
use satpy_core::frame_time::authority::{
    frame_authority, tai_jd_from_seconds, tai_seconds_from_utc_jd,
};
use satpy_core::frame_time::timescale::taiutc;
// The SHIPPED constant, deliberately not a local copy: this file carries the
// only gate binding a flattening to the sealed Orekit fixture, and a local
// copy would leave the production value unpinned.
use satpy_core::WGS84_FLATTENING;
use serde::Deserialize;

const FIXTURE: &str = include_str!("data/orekit_jb2008_eci_adapter_v1.json");

/// 2022-08-12. Inside the Part A window, the sealed EOP span, the ephemeris
/// coverage and the JB2008 driver coverage.
const TEST_EPOCH_JD: f64 = 2_459_794.5;

/// The compiled Part A science authority's dust drag coefficient, READ rather
/// than restated. This and [`DUST_AM_RATIO`] were local copies of the sealed
/// `nd_config/src/part_a_science.rs` values (2.2 / 1.948) citing the authority by line
/// number. NOT a pin in either form, and poison-proved so on 2026-08-08: the
/// gate drives the production RHS with `cd * am_ratio` and then inverts the
/// same product to recover density, so the factor cancels out of every
/// assertion (a 1.5x poison stayed green). What the value DOES scale is the
/// reported `a_drag` error and its "x the gravity y-coupling bug" comparison,
/// so it is bound to the authority to keep that printed comparison the
/// campaign's rather than a stale copy's.
const DUST_CD: f64 = nd_config::CompiledPartAScienceV1::part_a_v1()
    .hybrid()
    .dust_cd;
/// Sealed authority `dust_am_ratio`, bound like [`DUST_CD`]. The dust leg
/// carries ~195x the transfer leg's area-to-mass, which is why the drag defect
/// lands there and not on transfer.
const DUST_AM_RATIO: f64 = nd_config::CompiledPartAScienceV1::part_a_v1()
    .hybrid()
    .dust_am_ratio;

/// The Cunningham y-partial defect fixed earlier in this session, as a yardstick.
/// Not an assertion threshold -- reported so the two are compared on one scale.
const GRAVITY_Y_COUPLING_BUG_KM_S2: f64 = 2.920_579e-8;

#[derive(Deserialize)]
struct SealedFixture {
    earth: FixtureEarth,
    cases: Vec<FixtureCase>,
}

#[derive(Deserialize)]
struct FixtureEarth {
    shape: String,
    equatorial_radius_m: String,
    flattening: String,
}

#[derive(Deserialize)]
struct FixtureCase {
    id: String,
    inputs: FixtureInputs,
    expected: FixtureExpected,
    design: FixtureDesign,
}

#[derive(Deserialize)]
struct FixtureInputs {
    satellite_eci_m: [String; 3],
}

#[derive(Deserialize)]
struct FixtureExpected {
    satellite_body_m: [String; 3],
    satellite_geodetic: FixtureGeodetic,
    jb_primitive_inputs: FixturePrimitiveInputs,
}

#[derive(Deserialize)]
struct FixtureGeodetic {
    latitude_rad: String,
    altitude_m: String,
}

#[derive(Deserialize)]
struct FixturePrimitiveInputs {
    #[serde(rename = "satellite_ellipsoidal_altitude_m_as_satAlt")]
    satellite_ellipsoidal_altitude_m_as_sat_alt: String,
}

#[derive(Deserialize)]
struct FixtureDesign {
    geodetic_latitude_rad: String,
}

fn fixture() -> anyhow::Result<SealedFixture> {
    serde_json::from_str(FIXTURE)
        .map_err(|error| anyhow::anyhow!("sealed JB2008 adapter fixture: {error}"))
}

fn hex_f64(text: &str) -> anyhow::Result<f64> {
    let digits = text
        .strip_prefix("0x")
        .ok_or_else(|| anyhow::anyhow!("hex f64 must carry 0x prefix"))?;
    let bits = u64::from_str_radix(digits, 16)
        .map_err(|error| anyhow::anyhow!("hex f64 must parse: {error}"))?;
    Ok(f64::from_bits(bits))
}

fn vector3(values: &[String; 3]) -> anyhow::Result<[f64; 3]> {
    let [x, y, z] = values;
    Ok([hex_f64(x)?, hex_f64(y)?, hex_f64(z)?])
}

const fn same_f64(lhs: f64, rhs: f64) -> bool {
    match (lhs.classify(), rhs.classify()) {
        (std::num::FpCategory::Nan, _) | (_, std::num::FpCategory::Nan) => false,
        (std::num::FpCategory::Zero, std::num::FpCategory::Zero) => true,
        _ => lhs.to_bits() == rhs.to_bits(),
    }
}

fn strictly_greater(lhs: f64, rhs: f64) -> bool {
    matches!(lhs.partial_cmp(&rhs), Some(std::cmp::Ordering::Greater))
}

fn strictly_less(lhs: f64, rhs: f64) -> bool {
    matches!(lhs.partial_cmp(&rhs), Some(std::cmp::Ordering::Less))
}

#[expect(
    clippy::suboptimal_flops,
    reason = "the oracle preserves its established binary64 norm expression order"
)]
fn squared_norm3(values: &[f64; 3]) -> f64 {
    let &[x, y, z] = values;
    x * x + y * y + z * z
}

fn norm3(values: &[f64; 3]) -> f64 {
    squared_norm3(values).sqrt()
}

#[expect(
    clippy::imprecise_flops,
    reason = "the Bowring oracle preserves its established binary64 norm expression order"
)]
fn norm2(values: [f64; 2]) -> f64 {
    let [x, y] = values;
    (x * x + y * y).sqrt()
}

#[expect(
    clippy::suboptimal_flops,
    reason = "the oracle preserves the established binary64 altitude conversion order"
)]
fn altitude_error_m(altitude_km: f64, expected_altitude_m: f64) -> f64 {
    (altitude_km * 1000.0 - expected_altitude_m).abs()
}

#[expect(
    clippy::suboptimal_flops,
    reason = "the oracle preserves the established binary64 spherical-altitude expression"
)]
fn spherical_altitude_m(radius_m: f64, equatorial_radius_km: f64) -> f64 {
    radius_m - equatorial_radius_km * 1000.0
}

/// WGS84 geodetic latitude and ellipsoidal altitude from a body-fixed position.
///
/// Bowring fixed point. Lives in the test because the workspace has no geodetic
/// reduction -- `satpy_core::geocentric_spherical_from_itrs` is spherical and is
/// all there is. The production reduction belongs to whoever fixes the adapter;
/// this one exists only to compute what production OUGHT to have produced, and
/// `bowring_reduction_reproduces_the_sealed_orekit_geodetic` proves it agrees
/// with Orekit before any other test relies on it.
#[expect(
    clippy::suboptimal_flops,
    reason = "the Bowring oracle deliberately preserves its established binary64 operation order"
)]
fn geodetic_from_body_fixed_km(pos_km: &[f64; 3], a_km: f64, flattening: f64) -> (f64, f64) {
    let &[position_x, position_y, position_z] = pos_km;
    let eccentricity_squared = 2.0 * flattening - flattening * flattening;
    let transverse_radius = norm2([position_x, position_y]);
    if transverse_radius == 0.0 {
        // On the spin axis the fixed point degenerates; the polar radius is exact.
        let b_km = a_km * (1.0 - flattening);
        return (
            std::f64::consts::FRAC_PI_2.copysign(position_z),
            position_z.abs() - b_km,
        );
    }
    let mut lat = position_z.atan2(transverse_radius);
    for _ in 0..64 {
        let sin_lat = lat.sin();
        let prime_vertical_radius = a_km / (1.0 - eccentricity_squared * sin_lat * sin_lat).sqrt();
        let alt = transverse_radius / lat.cos() - prime_vertical_radius;
        let next = (position_z
            / (transverse_radius
                * (1.0
                    - eccentricity_squared * prime_vertical_radius
                        / (prime_vertical_radius + alt))))
            .atan();
        if (next - lat).abs() < 1e-15 {
            lat = next;
            break;
        }
        lat = next;
    }
    let sin_lat = lat.sin();
    let prime_vertical_radius = a_km / (1.0 - eccentricity_squared * sin_lat * sin_lat).sqrt();
    (lat, transverse_radius / lat.cos() - prime_vertical_radius)
}

/// The geometry radius production uses must BE the ellipsoid the oracle used.
///
/// Guards the trap: if someone repoints the drag altitude at
/// `GRAVITY_REFERENCE_RADIUS_KM`, or the WGS84 flattening above drifts, the
/// expected values in the gate below would silently stop meaning what they say.
#[test]
fn sealed_fixture_ellipsoid_matches_the_configured_geometry_radius() -> anyhow::Result<()> {
    let fixture = fixture()?;
    let sealed_a_m = hex_f64(&fixture.earth.equatorial_radius_m)?;
    let sealed_f = hex_f64(&fixture.earth.flattening)?;

    if fixture.earth.shape != "WGS84 OneAxisEllipsoid" {
        return Err(anyhow::anyhow!(
            "the oracle's declared shape is not the assumed WGS84 ellipsoid"
        ));
    }
    if !same_f64(ForceConfig::default().earth_radius, sealed_a_m / 1000.0) {
        return Err(anyhow::anyhow!(
            "ForceConfig::earth_radius must equal the oracle's equatorial radius; \
             it is the geometry radius, NOT GRAVITY_REFERENCE_RADIUS_KM (6378.13646)"
        ));
    }
    if !same_f64(WGS84_FLATTENING, sealed_f) {
        return Err(anyhow::anyhow!(
            "this file's flattening must equal the oracle's sealed flattening"
        ));
    }
    Ok(())
}

/// The in-test Bowring reduction must reproduce Orekit's `GeodeticPoint`.
///
/// Runs BEFORE the gate relies on it. If this is red, the gate below is
/// measuring the reduction rather than the adapter.
#[test]
fn bowring_reduction_reproduces_the_sealed_orekit_geodetic() -> anyhow::Result<()> {
    let a_km = ForceConfig::default().earth_radius;
    let mut worst_lat_rad = 0.0f64;
    let mut worst_alt_m = 0.0f64;

    for case in fixture()?.cases {
        let body_position_m = vector3(&case.expected.satellite_body_m)?;
        let position_km = body_position_m.map(|coordinate_m| coordinate_m / 1000.0);
        let (lat, alt_km) = geodetic_from_body_fixed_km(&position_km, a_km, WGS84_FLATTENING);

        let want_lat = hex_f64(&case.expected.satellite_geodetic.latitude_rad)?;
        let want_alt_m = hex_f64(&case.expected.satellite_geodetic.altitude_m)?;

        worst_lat_rad = worst_lat_rad.max((lat - want_lat).abs());
        worst_alt_m = worst_alt_m.max(altitude_error_m(alt_km, want_alt_m));
    }

    if !strictly_less(worst_lat_rad, 1e-12) {
        return Err(anyhow::anyhow!(
            "Bowring latitude disagrees with Orekit by {worst_lat_rad:e} rad"
        ));
    }
    if !strictly_less(worst_alt_m, 1e-6) {
        return Err(anyhow::anyhow!(
            "Bowring altitude disagrees with Orekit by {worst_alt_m:e} m"
        ));
    }
    Ok(())
}

/// Every sealed case's `satAlt` IS the ellipsoidal altitude, and `|r| - a` is not.
///
/// A record of the contract, independent of production. The equatorial cases are
/// the control: there the two definitions coincide exactly, so a nonzero delta
/// on those rows would mean the comparison itself is broken.
#[test]
fn sealed_satalt_is_ellipsoidal_and_differs_from_spherical_off_the_equator() -> anyhow::Result<()> {
    let a_km = ForceConfig::default().earth_radius;
    let mut rows = Vec::new();
    let mut max_delta_m = 0.0f64;

    for case in fixture()?.cases {
        let id = case.id;
        let eci_m = vector3(&case.inputs.satellite_eci_m)?;
        // Altitude is rotation-invariant in |r|, so the spherical form needs no
        // frame chain: |r_eci| == |r_itrs|.
        let spherical_alt_m = spherical_altitude_m(norm3(&eci_m), a_km);
        let sealed_alt_m = hex_f64(
            &case
                .expected
                .jb_primitive_inputs
                .satellite_ellipsoidal_altitude_m_as_sat_alt,
        )?;
        let design_lat_deg = hex_f64(&case.design.geodetic_latitude_rad)?.to_degrees();
        let delta_m = sealed_alt_m - spherical_alt_m;

        if design_lat_deg.abs() < 1e-9 {
            if !strictly_less(delta_m.abs(), 1e-6) {
                return Err(anyhow::anyhow!(
                    "{id}: equatorial control must have zero deficit, got {delta_m} m"
                ));
            }
        } else if !strictly_greater(delta_m, 1.0) {
            return Err(anyhow::anyhow!(
                "{id}: off-equator case must show a real deficit, got {delta_m} m"
            ));
        }
        max_delta_m = max_delta_m.max(delta_m);
        rows.push(format!(
            "{id:>8} lat {design_lat_deg:>6.1}  sealed {sealed_alt_m:>12.4} m  \
             spherical {spherical_alt_m:>12.4} m  deficit {delta_m:>10.4} m"
        ));
    }

    // a*f is the pole-limit deficit; nothing may exceed it.
    let a_f_m = a_km * 1000.0 * WGS84_FLATTENING;
    if !strictly_less(max_delta_m, a_f_m) {
        return Err(anyhow::anyhow!(
            "max deficit {max_delta_m} m exceeds a*f = {a_f_m} m:\n{}",
            rows.join("\n")
        ));
    }
    println!("{}", rows.join("\n"));
    println!("max deficit {max_delta_m:.4} m against a*f = {a_f_m:.4} m");
    Ok(())
}

// =============================================================================
// THE GATE
//
// RED until `sat_altitude_m` in `LightyearRHS::jb2008_density_at_state` is
// changed from `(|r| - earth_radius)` to the ellipsoidal altitude. It flips
// GREEN on that change alone, with no edit here -- that is the point of
// recovering the density from `compute_internal` rather than mirroring the
// formula.
//
// Construction:
// - DRAG only, `sph_order = 0`: gravity is skipped, no SRP, no third body, no
//   relativity/Lorentz/Coulomb. `required_dynamic_ephemeris_flags` still
//   attaches the Sun catalogue because `atm_model = 4` needs it.
// - `delta = 0` at `t = t0_s`, so the Battin Encke correction contributes
//   EXACTLY zero (`q = 0`, `f(0) = 0`) and `compute_internal[3..6]` is pure drag.
// - The expected density is not a pinned magic number. It is the same
//   `jb_rs::jb2008_density` kernel, fed the same epoch, the same catalogue Sun
//   and the same driver record production resolves, changing exactly ONE
//   argument: the altitude.
// =============================================================================

/// The four angular arguments this gate holds fixed, recorded so the mirror can
/// be pinned against production rather than merely asserted in a comment.
struct AngularFrame {
    sun_ra: f64,
    sun_declination: f64,
    sat_ra: f64,
    sat_geocentric_latitude: f64,
    /// The GEODETIC latitude of the same ITRS position. Not passed to the
    /// kernel — carried so a test can prove the reference took the geocentric
    /// one, since the two differ by ~0.181 deg at 60 deg.
    sat_geodetic_latitude: f64,
}

/// The gate's tolerance on the density ratio.
///
/// Sized to absorb epoch-plumbing roundoff and the difference between this
/// file's iterated Bowring and production's one-pass form (bounded at 1.5e-9 m
/// by `bowring_reduction_reproduces_the_sealed_orekit_geodetic`), while sitting
/// five orders below the ~1.4x defect the gate exists to catch. NOT to be
/// widened: `gate_predicate_still_rejects_a_spherical_altitude` measures the
/// margin, so a widening that swallowed the defect would fail there.
const GATE_RATIO_TOL: f64 = 1e-6;

/// One measured point: what production applied vs what the ellipsoidal altitude
/// gives, at a fixed physical state.
struct AdapterMeasurement {
    geocentric_lat_deg: f64,
    spherical_alt_km: f64,
    geodetic_alt_km: f64,
    rho_production: f64,
    rho_expected: f64,
    /// The same reference with ONLY the altitude reverted to `|r| - a`. The
    /// gate's falsifiability probe, not an expectation.
    rho_spherical: f64,
    /// The angular arguments the reference above was built from.
    angular_frame: AngularFrame,
    /// Drag magnitude production actually produced, straight from the RHS.
    a_drag_applied_km_s2: f64,
    /// Same state and same `v_rel`, scaled by the corrected density. Drag is
    /// linear in rho, so this is exact rather than a re-derivation.
    a_drag_correct_km_s2: f64,
}

/// The gate's decision, factored out so the sensitivity probe uses the SAME
/// predicate rather than a restatement of it that could drift.
fn ratio_passes(ratio: f64) -> bool {
    (ratio - 1.0).abs() < GATE_RATIO_TOL
}

impl AdapterMeasurement {
    fn ratio(&self) -> f64 {
        self.rho_production / self.rho_expected
    }

    /// What the gate would see if production reverted to a spherical altitude.
    fn spherical_ratio(&self) -> f64 {
        self.rho_spherical / self.rho_expected
    }

    fn error_km_s2(&self) -> f64 {
        self.a_drag_applied_km_s2 - self.a_drag_correct_km_s2
    }

    fn report(&self, label: &str) -> String {
        format!(
            "{label}\n  \
             geocentric lat {:.3} deg | spherical alt {:.4} km | ellipsoidal alt {:.4} km \
             | deficit {:.4} km\n  \
             rho_production {:.6e}  rho_expected {:.6e}  ratio {:.9}\n  \
             spherical-altitude revert would give ratio {:.6} (the defect this \
             gate detects)\n  \
             a_drag applied {:.6e}  correct {:.6e}  ERROR {:.6e} km/s^2 ({:.2}x the \
             {GRAVITY_Y_COUPLING_BUG_KM_S2:.6e} gravity y-coupling bug)",
            self.geocentric_lat_deg,
            self.spherical_alt_km,
            self.geodetic_alt_km,
            self.geodetic_alt_km - self.spherical_alt_km,
            self.rho_production,
            self.rho_expected,
            self.ratio(),
            self.spherical_ratio(),
            self.a_drag_applied_km_s2,
            self.a_drag_correct_km_s2,
            self.error_km_s2(),
            self.error_km_s2() / GRAVITY_Y_COUPLING_BUG_KM_S2,
        )
    }
}

/// Drive the real production RHS at one state and recover the density it applied.
///
/// The manifest-bound Part A UTC-JD lookup is the same Sun-resolution seam that
/// production uses, so the comparison stays like for like.
fn measure_adapter_at(spherical_alt_km: f64, lat_deg: f64) -> anyhow::Result<AdapterMeasurement> {
    let jd0 = TEST_EPOCH_JD;
    let cd = DUST_CD;
    let am_ratio = DUST_AM_RATIO;

    // Circular polar state: the sin^2(lat) deficit is large off the equator and
    // the geometry stays benign.
    let lat_c = lat_deg.to_radians();
    let r_km = ForceConfig::default().earth_radius + spherical_alt_km;
    let v_circ = (satpy_core::MU / r_km).sqrt();
    let eci = [
        r_km * lat_c.cos(),
        0.0,
        r_km * lat_c.sin(),
        -v_circ * lat_c.sin(),
        0.0,
        v_circ * lat_c.cos(),
    ];

    let mut init_equinoc = [0.0f64; 6];
    satpy_core::eci2equinoc_impl_f64(&eci, 6, 0.0, 0.0, &mut init_equinoc);
    if !init_equinoc.iter().all(|value| value.is_finite()) {
        return Err(anyhow::anyhow!(
            "chosen state must reduce to finite equinoctial elements"
        ));
    }

    let config = ForceConfig {
        sph_order: 0,
        force_flags: ForceFlags::DRAG,
        atm_model: 4,
        cd,
        am_ratio,
        ..ForceConfig::default()
    }
    .with_ephemeris_for_arc(jd0, jd0 + 0.01)
    .map_err(|error| anyhow::anyhow!("test epoch must have Sun ephemeris coverage: {error:?}"))?;
    let earth_radius = config.earth_radius;

    // sph_order = 0 skips gravity entirely; a minimal valid coefficient set.
    let stride = 2usize;
    let mut c_coeffs = vec![0.0; stride * stride];
    *c_coeffs
        .first_mut()
        .ok_or_else(|| anyhow::anyhow!("gravity C00 storage must not be empty"))? = 1.0;
    let s_coeffs = vec![0.0; stride * stride];
    let packed = satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, 0)
        .map_err(|error| anyhow::anyhow!("test gravity coefficients must pack: {error}"))?;

    // `try_new` is the public constructor; it applies the same Sterbenz split of
    // `jd0` that production uses, which the epoch reconstruction below mirrors.
    let rhs = LightyearRHS::try_new(init_equinoc, 0.0, jd0, Arc::new(config), Arc::new(packed))
        .context("constructing production RHS for a DRAG-only JB2008 config")?;

    let dxdt = rhs
        .compute_internal(&[0.0; 6], 0.0)
        .context("evaluating a valid DRAG-only JB2008 state")?;
    if !dxdt.iter().all(|value| value.is_finite()) {
        return Err(anyhow::anyhow!(
            "production RHS returned non-finite state: {dxdt:?} \
             (JB2008 drivers or Sun ephemeris may not cover jd0 = {jd0})"
        ));
    }

    let [position_x, position_y, position_z, velocity_x, velocity_y, velocity_z] = eci;
    let [_, _, _, acceleration_x, acceleration_y, acceleration_z] = dxdt;
    let a_mag_km = squared_norm3(&[acceleration_x, acceleration_y, acceleration_z]).sqrt();
    if !strictly_greater(a_mag_km, 0.0) {
        return Err(anyhow::anyhow!(
            "drag acceleration is zero; the JB2008 path did not run"
        ));
    }
    // Rebuild the epoch, Sun and drivers exactly as the RHS seams do.
    let big = (jd0 - 0.5).floor() + 0.5;
    let tai0_s = tai_seconds_from_utc_jd(big, jd0 - big)
        .map_err(|error| anyhow::anyhow!("epoch must fall inside sealed EOP span: {error:?}"))?;
    let (tai1, tai2) = tai_jd_from_seconds(tai0_s);
    let (status, utc1, utc2) = taiutc(tai1, tai2);
    if status != 0 {
        return Err(anyhow::anyhow!(
            "TAI->UTC must resolve at the test epoch: status {status}"
        ));
    }
    let jd_utc = utc1 + utc2;

    let ephem = get_precomputed_ephemeris()
        .ok_or_else(|| anyhow::anyhow!("compiled ephemeris catalogue must be available"))?;
    let utc_epoch = UtcJulianDay::new(jd_utc)
        .map_err(|error| anyhow::anyhow!("UTC Julian date must be valid: {error:?}"))?;
    let sun = ephem
        .get(Body::Sun)
        .ok_or_else(|| anyhow::anyhow!("compiled Sun catalogue must be available"))?
        .position_at_part_a_utc_jd(utc_epoch)
        .map_err(|error| {
            anyhow::anyhow!("Sun position must resolve at the test epoch: {error:?}")
        })?;
    let sun_gcrs = [sun[0], sun[1], sun[2]];

    let utc_mjd = utc_epoch
        .to_utc_mjd()
        .map_err(|error| anyhow::anyhow!("UTC modified Julian date must be valid: {error:?}"))?;
    let driver = compiled_drivers()
        .map_err(|error| anyhow::anyhow!("compiled JB2008 drivers must be available: {error:?}"))?
        .lookup_utc_mjd(utc_mjd)
        .map_err(|error| {
            anyhow::anyhow!("driver record must exist at the test epoch: {error:?}")
        })?;

    // GCRS norm, used only for the SPHERICAL altitude `|r| - a` — the defect
    // this gate detects. `|r|` is rotation-invariant, so no frame chain is owed
    // here; the ITRS norms below are separate and feed the angles.
    let pos_gcrs = [position_x, position_y, position_z];
    let sat_r = norm3(&pos_gcrs);

    // THE ELLIPSOID IS EARTH-FIXED, SO THE REDUCTION IS TOO.
    //
    // WGS84's flattening is about the Earth's FIGURE axis, not the GCRS z axis,
    // so the geodetic reduction must happen AFTER the GCRS->ITRS rotation.
    // `|r|` is rotation-invariant but `z` is NOT: GCRS->ITRS is
    // `RPOM * R3(ERA) * RC2I` and only the `R3(ERA)` factor is a rotation about
    // z. `RC2I` tilts the pole by ~2e-3 rad by 2022, which is ~40 m of altitude
    // at 60 deg via `d(alt)/d(lat) = a*f*sin(2*lat) = 18.5 km/rad`, i.e. ~0.09%
    // of density through a 60 km scale height.
    //
    // An earlier revision of this file reduced the GCRS state directly and
    // called it frame-agnostic. That was WRONG and is the reason this comment is
    // long: the shortcut is invisible while production is also GCRS-shaped and
    // only bites once production does the correct thing.
    //
    // This resolves the SAME rotation production does -- `frame_rotation_at(t)`
    // takes `tai_seconds_at(t) = tai0_s + t`, and `t = 0` here.
    let rotation = frame_authority().rotation_at(tai0_s).map_err(|error| {
        anyhow::anyhow!("frame rotation must resolve at the test epoch: {error:?}")
    })?;
    // Recover density by inverting the full-frame drag law production uses:
    // |a| km/s^2 = 0.5 * cd * (A/m) * rho * |v_rel_m|^2 * 1e-3.
    let [omega_x, omega_y, omega_z] = rotation.itrs_angular_velocity_gcrs;
    let v_rel = [
        velocity_x - (omega_y * position_z - omega_z * position_y),
        velocity_y - (omega_z * position_x - omega_x * position_z),
        velocity_z - (omega_x * position_y - omega_y * position_x),
    ];
    let v_rel_m_sq = squared_norm3(&v_rel) * 1.0e6;
    let rho_production = a_mag_km * 1.0e3 / (0.5 * cd * am_ratio * v_rel_m_sq);
    let pos_itrs = rotation.to_itrs(&pos_gcrs);
    let sun_itrs = rotation.to_itrs(&sun_gcrs);
    let (geodetic_lat, geodetic_alt_km) =
        geodetic_from_body_fixed_km(&pos_itrs, earth_radius, WGS84_FLATTENING);

    // Norms taken in ITRS so every angle below comes from ONE vector. The
    // rotation is orthogonal so these equal `sat_r`/`sun_r` to rounding; they
    // are recomputed rather than reused so the reduction has a single source.
    let sat_r_itrs = norm3(&pos_itrs);
    let sun_r_itrs = norm3(&sun_itrs);

    // Everything production passes, with ONE argument under test: the altitude.
    //
    // The four angular arguments MIRROR production's ITRS reduction. That is
    // what makes this an altitude gate rather than a whole-adapter gate: hold
    // the angles equal on both sides and the ratio isolates the altitude alone.
    // Latitude stays GEOCENTRIC (Bowman's SAT(2)), which is a separate question
    // from the frame and is settled the other way from Orekit.
    let [pos_itrs_x, pos_itrs_y, pos_itrs_z] = pos_itrs;
    let [sun_itrs_x, sun_itrs_y, sun_itrs_z] = sun_itrs;
    // Diagnostics only -- the kernel is handed the hour angle below. These two
    // are kept so the `AngularFrame` receipt still reports each right ascension
    // separately, which is what makes a frame mix-up readable in the output.
    let sun_ra_rad = sun_itrs_y.atan2(sun_itrs_x);
    let sat_ra_rad = pos_itrs_y.atan2(pos_itrs_x);
    let corrected = Jb2008Input {
        mjd_utc: utc_mjd.as_f64(),
        sun_declination_rad: (sun_itrs_z / sun_r_itrs).clamp(-1.0, 1.0).asin(),
        // Mirrors the production adapter: one `atan2` of the cross/dot pair.
        hour_angle_rad: pos_itrs_y
            .mul_add(sun_itrs_x, -(pos_itrs_x * sun_itrs_y))
            .atan2(pos_itrs_x.mul_add(sun_itrs_x, pos_itrs_y * sun_itrs_y)),
        sat_geocentric_lat_rad: (pos_itrs_z / sat_r_itrs).clamp(-1.0, 1.0).asin(),
        sat_altitude_m: geodetic_alt_km * 1000.0,
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
    let rho_expected = jb2008_density(corrected)
        .map_err(|error| anyhow::anyhow!("corrected JB2008 input must evaluate: {error:?}"))?;

    // THE PERTURBATION THAT PROVES THIS GATE IS STILL FALSIFIABLE.
    //
    // `corrected` with the ONE argument under test moved back to the defect:
    // `|r| - a` instead of the ellipsoidal altitude. Everything else, angles
    // included, is byte-identical. If the gate's predicate does not reject this,
    // the gate has stopped protecting the altitude fix and would sit green
    // through a revert. Asserted by
    // `gate_predicate_still_rejects_a_spherical_altitude`.
    let spherical = Jb2008Input {
        sat_altitude_m: (sat_r - earth_radius) * 1000.0,
        ..corrected
    };
    let rho_spherical = jb2008_density(spherical)
        .map_err(|error| anyhow::anyhow!("spherical JB2008 input must evaluate: {error:?}"))?;

    Ok(AdapterMeasurement {
        geocentric_lat_deg: (pos_itrs_z / sat_r_itrs).asin().to_degrees(),
        spherical_alt_km: sat_r - earth_radius,
        geodetic_alt_km,
        rho_production,
        rho_expected,
        rho_spherical,
        angular_frame: AngularFrame {
            sun_ra: sun_ra_rad,
            sun_declination: corrected.sun_declination_rad,
            sat_ra: sat_ra_rad,
            sat_geocentric_latitude: corrected.sat_geocentric_lat_rad,
            sat_geodetic_latitude: geodetic_lat,
        },
        a_drag_applied_km_s2: a_mag_km,
        // Drag is linear in rho at fixed v_rel, so scaling the measured
        // magnitude is exact.
        a_drag_correct_km_s2: a_mag_km * rho_expected / rho_production,
    })
}

/// Assert one measured point matches the ellipsoidal-altitude expectation.
///
/// # What a failure here does and does not tell you
///
/// The reference is production's own inputs with EXACTLY ONE quantity replaced:
/// the altitude. So a red here means production's `sat_altitude_m` disagrees
/// with the WGS84 ellipsoidal altitude — **provided the angular arguments still
/// match**, which `angular_frame_matches_production_so_the_altitude_is_isolated`
/// checks separately. If that companion is ALSO red, believe it first and treat
/// this one as a symptom: an angular-frame change shows up here misattributed as
/// an altitude defect, and the ratio it produces (1.0001-1.0004) is four orders
/// smaller than a genuine spherical-altitude revert (~1.4). The message below
/// names the discriminator rather than asserting a cause.
fn assert_adapter_matches_ellipsoidal(m: &AdapterMeasurement, label: &str) -> anyhow::Result<()> {
    println!("{}", m.report(label));
    if ratio_passes(m.ratio()) {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "{label}: production's JB2008 altitude argument disagrees with the WGS84 \
             ellipsoidal altitude. It applied rho = {:.6e} where the ellipsoidal \
             altitude gives {:.6e} (ratio {:.9}), worth {:.6e} km/s^2 of drag. \
             Ellipsoidal altitude {:.4} km vs spherical {:.4} km, a deficit of \
             {:.4} km.\n\
             WHICH DEFECT: a spherical-altitude revert lands at ratio {:.6} here — \
             if the observed {:.9} is orders smaller than that, the altitude is NOT \
             the cause and something the reference holds fixed has moved. The four \
             angular arguments are the usual candidate: this file mirrors \
             production's ITRS reduction of them, and \
             `angular_frame_matches_production_so_the_altitude_is_isolated` fails \
             first if that mirror has gone stale. Check it before touching \
             `sat_altitude_m` in `jb2008_density_at_state`.",
            m.rho_production,
            m.rho_expected,
            m.ratio(),
            m.error_km_s2(),
            m.geodetic_alt_km,
            m.spherical_alt_km,
            m.geodetic_alt_km - m.spherical_alt_km,
            m.spherical_ratio(),
            m.ratio(),
        ))
    }
}

/// THE SENSITIVITY PROOF. The gate must still reject a spherical altitude.
///
/// Runs the gate's OWN predicate, `ratio_passes`, against a locally perturbed
/// reference — production's exact inputs with the altitude reverted to
/// `|r| - a` and nothing else touched. Without this the gate could be made
/// unfalsifiable by a widened tolerance or a reference that quietly drifted onto
/// the same defect it checks, and it is the only instrument protecting a 42-76%
/// density fix.
///
/// Deliberately asserts a MARGIN, not merely a rejection: the defect must clear
/// the tolerance by at least four orders, which is what makes the tolerance
/// un-widenable in practice.
#[test]
fn gate_predicate_still_rejects_a_spherical_altitude() -> anyhow::Result<()> {
    for (alt_km, lat_deg) in [(400.0, 60.0), (200.0, 60.0)] {
        let m = measure_adapter_at(alt_km, lat_deg)?;
        let label = format!("{alt_km} km / {lat_deg} deg");
        let excess = (m.spherical_ratio() - 1.0).abs() / GATE_RATIO_TOL;
        println!(
            "{label}: spherical-altitude revert would give ratio {:.6} \
             ({:.0}x the {GATE_RATIO_TOL:e} tolerance)",
            m.spherical_ratio(),
            excess
        );
        if ratio_passes(m.spherical_ratio()) {
            return Err(anyhow::anyhow!(
                "{label}: the gate's predicate ACCEPTS a spherical altitude \
                 (ratio {:.9}). The gate is unfalsifiable and would sit green \
                 through a revert of the ellipsoidal-altitude fix.",
                m.spherical_ratio()
            ));
        }
        if !strictly_greater(excess, 1.0e4) {
            return Err(anyhow::anyhow!(
                "{label}: a spherical altitude clears the tolerance by only \
                 {excess:.1}x. The gate still rejects it, but the margin has \
                 collapsed — the tolerance, the reference or the state has drifted."
            ));
        }
    }
    Ok(())
}

/// The mirror must not go stale: this file's angular frame IS production's.
///
/// The gate above isolates the altitude only while both sides use the same four
/// angular arguments. Nothing structural enforces that — production is in
/// `rhs.rs` and the reference is in this file — so when production moved those
/// four from GCRS to ITRS, this gate went red and blamed the altitude.
///
/// This test makes that failure mode self-identifying. It recovers production's
/// density at a state where the two frames differ measurably, and checks it
/// against the reference built with THIS file's angles. Equality means the
/// mirror holds. It is deliberately ordered before the altitude gate in the
/// failure message above, because when both are red this one names the cause.
///
/// It cannot fail without the gate also failing, and that is the point: it
/// converts one misleading red into two, the second of which is correct.
#[test]
fn angular_frame_matches_production_so_the_altitude_is_isolated() -> anyhow::Result<()> {
    let m = measure_adapter_at(400.0, 60.0)?;
    let f = &m.angular_frame;
    println!(
        "reference angular frame (ITRS): sunRA {:.9} sunDec {:.9} satRA {:.9} satLat {:.9} rad",
        f.sun_ra, f.sun_declination, f.sat_ra, f.sat_geocentric_latitude
    );

    // The reference must be feeding the GEOCENTRIC latitude (Bowman's SAT(2)),
    // not the geodetic one. At this state the two separate by 0.155789 deg
    // (measured 2026-08-09; the ~0.181 deg this comment and the message below
    // used to quote is the separation's peak near 45 deg, not its value here),
    // so this is a decisive check rather than a tolerance question. Without it,
    // a switch to the geodetic convention on either side would surface as an
    // altitude ratio and be misread exactly the way the frame change was.
    //
    // The floor is 0.12 deg. It is not a ratchet against drift in the measured
    // 0.155789 — the failure it exists to catch drives the separation to
    // EXACTLY 0.0, because the two latitudes become the same number — so it
    // needs to sit far from zero, not close to the measurement. 0.12 leaves 23%
    // headroom for state and frame wander while staying an order clear of the
    // value a convention flip produces. The previous 0.15 left 3.9%, which is
    // ratchet-tight for a bound whose real discriminand is zero.
    let geoc_minus_geod_deg = (f.sat_geocentric_latitude - f.sat_geodetic_latitude).to_degrees();
    println!(
        "latitude convention: geocentric {:.6} deg, geodetic {:.6} deg, \
         separation {:.6} deg",
        f.sat_geocentric_latitude.to_degrees(),
        f.sat_geodetic_latitude.to_degrees(),
        geoc_minus_geod_deg
    );
    if !strictly_greater(geoc_minus_geod_deg.abs(), 0.12) {
        return Err(anyhow::anyhow!(
            "the reference latitude is within {:.6} deg of the GEODETIC value at \
             this state, where the two conventions separate by 0.155789 deg. \
             `sat_geocentric_lat_rad` must stay GEOCENTRIC per Bowman's SAT(2).",
            geoc_minus_geod_deg.abs()
        ));
    }

    if !ratio_passes(m.ratio()) {
        return Err(anyhow::anyhow!(
            "production's JB2008 angular arguments no longer match this file's \
             mirror. Production density {:.9e} vs reference {:.9e} (ratio {:.9}). \
             This gate isolates the ALTITUDE only while both sides reduce \
             sun_ra_rad, sun_declination_rad, sat_ra_rad and sat_geocentric_lat_rad \
             from ITRS. If production deliberately changed that reduction, update \
             the mirror in `measure_adapter_at` — do NOT widen GATE_RATIO_TOL, and \
             do not read this as an altitude defect. See \
             `jb2008_angular_frame_consistency.rs`, which owns the angular contract.",
            m.rho_production,
            m.rho_expected,
            m.ratio(),
        ));
    }
    Ok(())
}

#[test]
fn production_jb2008_altitude_input_is_ellipsoidal_not_spherical() -> anyhow::Result<()> {
    let m = measure_adapter_at(400.0, 60.0)?;
    assert_adapter_matches_ellipsoidal(&m, "400 km / 60 deg")
}

/// The low-perigee end of the Part A envelope, measured rather than extrapolated.
///
/// `nd_config/src/part_a_science.rs` sets `min_perigee_km = 6578.137`, i.e. 200 km
/// altitude, so this point is INSIDE the operating envelope. Density rises
/// steeply with decreasing altitude and the drag error is strictly proportional
/// to density, so this is where a spherical-altitude revert would be worst in
/// absolute terms even though the ratio is similar. The assertion is the
/// CORRECT (ellipsoidal) behavior at that corner.
#[test]
fn production_jb2008_altitude_is_ellipsoidal_at_part_a_low_perigee() -> anyhow::Result<()> {
    let m = measure_adapter_at(200.0, 60.0)?;
    assert_adapter_matches_ellipsoidal(&m, "200 km / 60 deg (Part A min_perigee_km = 6578.137)")
}
