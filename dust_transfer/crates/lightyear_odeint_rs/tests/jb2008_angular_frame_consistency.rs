//! Production JB2008 adapter conformance: ALL FOUR angular arguments must be
//! reduced from the SAME Earth-fixed (ITRS) position, not from GCRS.
//!
//! # What this closes
//!
//! `jb2008_adapter_altitude_gate.rs` covers `sat_altitude_m` and explicitly
//! scopes itself out of the angular arguments, asserting there that "`sat_ra_rad`
//! must stay an ECI right ascension: the kernel consumes only
//! `h = sat_ra - sun_ra`, so a common z-rotation cancels exactly and there is
//! nothing to fix". The premise is true and the conclusion is false, for the
//! same reason the altitude comment already records: GCRS->ITRS is
//! `RPOM * R3(ERA) * RC2I` and only `R3(ERA)` is a rotation about z. `RC2I`
//! tilts the pole, so the RA DIFFERENCE is not invariant either -- it is
//! invariant only to first order, leaving ~1e-3 rad of hour angle.
//!
//! # The four arguments, and why they move as a set
//!
//! `jb_rs::jb2008` consumes the angular inputs only through
//!
//! ```text
//! eta   = 0.5 * |satLat - sunDec|
//! theta = 0.5 * |satLat + sunDec|
//! h     = satRA - sunRA          (then tau, then local solar time)
//! ```
//!
//! plus `satLat` again inside `jb_dtc` and `jb_dlrsl`. `eta`/`theta` and `h`
//! respond to the frame tilt with OPPOSITE sign through `jb_tsub_l`, so moving
//! the latitude alone overshoots the consistent answer -- measured 3.5x at
//! 400 km. Either all four move or none do. This file pins "all four".
//!
//! # Which frame is right
//!
//! Bowman et al. (AIAA/AAS 2008-6438) name `SAT(2)` "Geocentric Latitude of
//! Position" and `SUN(2)` "Declination of Sun". A geocentric LATITUDE is
//! measured from the Earth's equator -- the CIP -- not from the GCRS equator,
//! which is the mean equator of J2000 and by 2026 differs from it by ~1.3e-3
//! rad. The same axis defines declination of date. The right ascensions enter
//! only as a difference, which is exactly the satellite's hour angle relative
//! to the Sun, and that difference is preserved by ITRS (TIRS differs from
//! true-of-date by a pure z-rotation about the same pole).
//!
//! The sealed Orekit oracle agrees on the FRAME. Its `jb_primitive_inputs`
//! column names are literal:
//!
//! ```text
//! sun_longitude_rad_as_sunRA
//! sun_geodetic_latitude_rad_as_sunDecli
//! satellite_geodetic_longitude_rad_as_satLon
//! satellite_geodetic_latitude_rad_as_satLat
//! ```
//!
//! -- every one of them a BODY-FRAME reduction. Orekit's deviation from Bowman
//! is GEODETIC vs geocentric latitude, which is a separate question and is NOT
//! adopted here; the frame choice is common ground.
//!
//! # What the sealed fixture canNOT settle
//!
//! That fixture's `time_and_frame_law` declares
//! `rotation_convention = "parent ECI to child body VECTOR_OPERATOR rotation
//! about +Z"` -- a PURE z-rotation. Under it, latitude and declination are
//! unchanged and the RA difference cancels exactly, so the fixture's numbers
//! are identical for both frame choices. It is evidence about Orekit's
//! INTENT (via the column names) and no evidence at all about the magnitude.
//! The magnitudes below are therefore measured through the production RHS
//! against the real IAU 2006/2000A chain, not read off the fixture.

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
// The SHIPPED constant, deliberately not a local copy -- see the note in
// jb2008_adapter_altitude_gate.rs.
use satpy_core::WGS84_FLATTENING;

/// 2022-08-12, the same epoch `jb2008_adapter_altitude_gate.rs` uses: inside the
/// Part A window, the sealed EOP span, the ephemeris and the driver coverage.
const TEST_EPOCH_JD: f64 = 2_459_794.5;

// Bound to the compiled science authority, not restated -- same reasoning as
// the WGS84_FLATTENING import above and the note on these two constants in
// `jb2008_adapter_altitude_gate.rs` (they were local copies of the sealed
// 2.2 / 1.948 until 2026-08-08). As there, `cd * am_ratio` cancels out of the
// density-ratio assertions -- it drives the RHS and then divides the recovered
// density -- so this binding keeps the printed diagnostics campaign-true
// rather than hardening a gate.
const DUST_CD: f64 = nd_config::CompiledPartAScienceV1::part_a_v1()
    .hybrid()
    .dust_cd;
const DUST_AM_RATIO: f64 = nd_config::CompiledPartAScienceV1::part_a_v1()
    .hybrid()
    .dust_am_ratio;

fn strictly_greater(left: f64, right: f64) -> bool {
    matches!(left.partial_cmp(&right), Some(std::cmp::Ordering::Greater))
}

fn strictly_less(left: f64, right: f64) -> bool {
    matches!(left.partial_cmp(&right), Some(std::cmp::Ordering::Less))
}

/// Ellipsoidal altitude in km from a body-fixed position, Bowring fixed point.
///
/// Mirrors the private `rhs::geodetic_altitude_km`. The altitude gate proves
/// this reduction against Orekit's `GeodeticPoint` to 1e-6 m before anything
/// relies on it, and the two forms are cross-checked against each other here by
/// the fact that the production comparison below closes to 1e-9 in density.
#[expect(
    clippy::suboptimal_flops,
    reason = "This oracle deliberately preserves the production binary64 operation order."
)]
fn ellipsoidal_alt_km(pos_km: &[f64; 3], a_km: f64) -> f64 {
    let f = WGS84_FLATTENING;
    let e2 = 2.0 * f - f * f;
    let p = pos_km[0].hypot(pos_km[1]);
    let z = pos_km[2];
    if p == 0.0 {
        return z.abs() - a_km * (1.0 - f);
    }
    let mut lat = z.atan2(p);
    for _ in 0..64 {
        let sin_lat = lat.sin();
        let n = a_km / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        let alt = p / lat.cos() - n;
        let next = (z / (p * (1.0 - e2 * n / (n + alt)))).atan();
        if (next - lat).abs() < 1e-15 {
            lat = next;
            break;
        }
        lat = next;
    }
    let sin_lat = lat.sin();
    let n = a_km / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    p * lat.cos() + z * sin_lat - n * (1.0 - e2 * sin_lat * sin_lat)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "This oracle deliberately preserves the production binary64 operation order."
)]
fn squared_norm3(v: &[f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

fn geocentric_lat_rad(v: &[f64; 3]) -> f64 {
    let r = squared_norm3(v).sqrt();
    (v[2] / r).clamp(-1.0, 1.0).asin()
}

/// One measured point: production density vs both candidate frame conventions.
struct FrameMeasurement {
    label: String,
    /// Density the production RHS actually applied, recovered from the drag.
    rho_production: f64,
    /// All four angular arguments from GCRS (the pre-fix convention).
    rho_all_gcrs: f64,
    /// All four angular arguments from ITRS (the target convention).
    rho_all_itrs: f64,
    /// Latitude alone moved to ITRS, RA and Sun left in GCRS. The inconsistent
    /// partial fix, measured to show it overshoots.
    rho_lat_only: f64,
    /// Hour angle `h = satRA - sunRA` under each convention, radians.
    h_gcrs: f64,
    h_itrs: f64,
    sat_lat_gcrs_deg: f64,
    sat_lat_itrs_deg: f64,
}

impl FrameMeasurement {
    /// `h_itrs - h_gcrs`, wrapped into `(-pi, pi]`.
    ///
    /// The raw difference carries a full turn: ITRS right ascension is measured
    /// from the prime meridian and GCRS from the equinox, so the two `h` values
    /// differ by an integer number of turns plus the physical shift. The kernel
    /// takes `rem_euclid(TAU)` of each RA before differencing, so only the
    /// wrapped part is observable -- reporting the raw difference would hide a
    /// 6.6e-4 rad signal inside a 6.28 rad bookkeeping offset.
    fn h_shift_rad(&self) -> f64 {
        let raw = self.h_itrs - self.h_gcrs;
        let wrapped = raw.rem_euclid(std::f64::consts::TAU);
        if wrapped > std::f64::consts::PI {
            wrapped - std::f64::consts::TAU
        } else {
            wrapped
        }
    }

    fn report(&self) -> String {
        format!(
            "{}\n  \
             satLat  GCRS {:>9.5} deg   ITRS {:>9.5} deg   delta {:>+9.5} deg\n  \
             h       GCRS {:>9.6} rad   ITRS {:>9.6} rad   wrapped delta {:>+9.3e} rad \
             ({:+.3} s of local solar time)\n  \
             rho all-GCRS {:.9e}   all-ITRS {:.9e}   lat-only {:.9e}\n  \
             delta(all-ITRS  vs all-GCRS) {:>+8.4} %\n  \
             delta(lat-only  vs all-GCRS) {:>+8.4} %   ({:.2}x the consistent move)\n  \
             production/all-ITRS ratio {:.12}",
            self.label,
            self.sat_lat_gcrs_deg,
            self.sat_lat_itrs_deg,
            self.sat_lat_itrs_deg - self.sat_lat_gcrs_deg,
            self.h_gcrs,
            self.h_itrs,
            self.h_shift_rad(),
            self.h_shift_rad() * 86400.0 / std::f64::consts::TAU,
            self.rho_all_gcrs,
            self.rho_all_itrs,
            self.rho_lat_only,
            100.0 * (self.rho_all_itrs / self.rho_all_gcrs - 1.0),
            100.0 * (self.rho_lat_only / self.rho_all_gcrs - 1.0),
            (self.rho_lat_only / self.rho_all_gcrs - 1.0)
                / (self.rho_all_itrs / self.rho_all_gcrs - 1.0),
            self.rho_production / self.rho_all_itrs,
        )
    }
}

/// Drive the real production RHS at one state and recover the density it applied.
///
/// `position_at_part_a_utc_jd` admits legacy epoch-tagged catalogues only after
/// their compiled Part A manifest authority has been validated. This test uses
/// the same Sun-resolution seam as production, so the comparison stays like for
/// like without a deprecated bare-`f64` access.
fn measure_at(
    spherical_alt_km: f64,
    lat_deg: f64,
    label: &str,
) -> anyhow::Result<FrameMeasurement> {
    let jd0 = TEST_EPOCH_JD;
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
    if !init_equinoc.iter().all(|v| v.is_finite()) {
        return Err(anyhow::anyhow!(
            "{label}: chosen state must reduce to finite equinoctial elements"
        ));
    }

    let config = ForceConfig {
        sph_order: 0,
        force_flags: ForceFlags::DRAG,
        atm_model: 4,
        cd: DUST_CD,
        am_ratio: DUST_AM_RATIO,
        ..ForceConfig::default()
    }
    .with_ephemeris_for_arc(jd0, jd0 + 0.01)
    .map_err(|error| anyhow::anyhow!("test epoch must have Sun ephemeris coverage: {error:?}"))?;
    let earth_radius = config.earth_radius;

    let stride = 2usize;
    let mut c_coeffs = vec![0.0; stride * stride];
    let c00 = c_coeffs
        .first_mut()
        .ok_or_else(|| anyhow::anyhow!("gravity C00 storage must not be empty"))?;
    *c00 = 1.0;
    let s_coeffs = vec![0.0; stride * stride];
    let packed = satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, 0)
        .map_err(|error| anyhow::anyhow!("test gravity coefficients must pack: {error}"))?;

    let rhs = LightyearRHS::try_new(init_equinoc, 0.0, jd0, Arc::new(config), Arc::new(packed))
        .context("constructing production RHS for a DRAG-only JB2008 config")?;

    let dxdt = rhs
        .compute_internal(&[0.0; 6], 0.0)
        .with_context(|| format!("{label}: evaluating a valid DRAG-only JB2008 state"))?;
    if !dxdt.iter().all(|v| v.is_finite()) {
        return Err(anyhow::anyhow!(
            "{label}: production RHS returned non-finite state: {dxdt:?}"
        ));
    }

    let a_mag_km = squared_norm3(&[dxdt[3], dxdt[4], dxdt[5]]).sqrt();
    if !strictly_greater(a_mag_km, 0.0) {
        return Err(anyhow::anyhow!(
            "{label}: drag acceleration is zero; the JB2008 path did not run"
        ));
    }
    // Rebuild the epoch, Sun, drivers and rotation exactly as the RHS seams do.
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
    let sun_gcrs = ephem
        .get(Body::Sun)
        .ok_or_else(|| anyhow::anyhow!("compiled Sun catalogue must be available"))?
        .position_at_part_a_utc_jd(utc_epoch)
        .map_err(|error| {
            anyhow::anyhow!("Sun position must resolve at the test epoch: {error:?}")
        })?;
    let sun_gcrs = [sun_gcrs[0], sun_gcrs[1], sun_gcrs[2]];

    let utc_mjd = utc_epoch
        .to_utc_mjd()
        .map_err(|error| anyhow::anyhow!("UTC modified Julian date must be valid: {error:?}"))?;
    let driver = compiled_drivers()
        .map_err(|error| anyhow::anyhow!("compiled JB2008 drivers must be available: {error:?}"))?
        .lookup_utc_mjd(utc_mjd)
        .map_err(|error| {
            anyhow::anyhow!("driver record must exist at the test epoch: {error:?}")
        })?;

    // The SAME rotation production resolves: `frame_rotation_at(t)` takes
    // `tai_seconds_at(t) = tai0_s + t`, and `t = 0` here.
    let rotation = frame_authority().rotation_at(tai0_s).map_err(|error| {
        anyhow::anyhow!("frame rotation must resolve at the test epoch: {error:?}")
    })?;
    let pos_gcrs = [eci[0], eci[1], eci[2]];
    let [omega_x, omega_y, omega_z] = rotation.itrs_angular_velocity_gcrs;
    let v_rel = [
        eci[3] - (omega_y * eci[2] - omega_z * eci[1]),
        eci[4] - (omega_z * eci[0] - omega_x * eci[2]),
        eci[5] - (omega_x * eci[1] - omega_y * eci[0]),
    ];
    let v_rel_m_sq = squared_norm3(&v_rel) * 1.0e6;
    let rho_production = a_mag_km * 1.0e3 / (0.5 * DUST_CD * DUST_AM_RATIO * v_rel_m_sq);
    let pos_itrs = rotation.to_itrs(&pos_gcrs);
    let sun_itrs = rotation.to_itrs(&sun_gcrs);

    // The altitude is ELLIPSOIDAL and ITRS-reduced under every variant below --
    // that argument is already fixed and is held constant so the numbers isolate
    // the angular change.
    let alt_m = ellipsoidal_alt_km(&pos_itrs, earth_radius) * 1000.0;

    let base = Jb2008Input {
        mjd_utc: utc_mjd.as_f64(),
        sun_declination_rad: 0.0,
        hour_angle_rad: 0.0,
        sat_geocentric_lat_rad: 0.0,
        sat_altitude_m: alt_m,
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

    let (ra_sat_gcrs, ra_sun_gcrs) = (
        pos_gcrs[1].atan2(pos_gcrs[0]),
        sun_gcrs[1].atan2(sun_gcrs[0]),
    );
    let (ra_sat_itrs, ra_sun_itrs) = (
        pos_itrs[1].atan2(pos_itrs[0]),
        sun_itrs[1].atan2(sun_itrs[0]),
    );
    let lat_gcrs = geocentric_lat_rad(&pos_gcrs);
    let lat_itrs = geocentric_lat_rad(&pos_itrs);
    let dec_gcrs = geocentric_lat_rad(&sun_gcrs);
    let dec_itrs = geocentric_lat_rad(&sun_itrs);

    // The kernel takes the hour angle, so each frame's variant differences its
    // own pair. That is the quantity this test was always about: the whole-turn
    // offset between the two frames cancels in the difference, and what survives
    // is the first-order residual the assertions below bound.
    let all_gcrs = Jb2008Input {
        sun_declination_rad: dec_gcrs,
        hour_angle_rad: ra_sat_gcrs - ra_sun_gcrs,
        sat_geocentric_lat_rad: lat_gcrs,
        ..base
    };
    let all_itrs = Jb2008Input {
        sun_declination_rad: dec_itrs,
        hour_angle_rad: ra_sat_itrs - ra_sun_itrs,
        sat_geocentric_lat_rad: lat_itrs,
        ..base
    };
    // The partial fix that was tried and reverted: latitude alone.
    let lat_only = Jb2008Input {
        sat_geocentric_lat_rad: lat_itrs,
        ..all_gcrs
    };

    Ok(FrameMeasurement {
        label: label.to_owned(),
        rho_production,
        rho_all_gcrs: jb2008_density(all_gcrs)
            .map_err(|error| anyhow::anyhow!("all-GCRS input must evaluate: {error:?}"))?,
        rho_all_itrs: jb2008_density(all_itrs)
            .map_err(|error| anyhow::anyhow!("all-ITRS input must evaluate: {error:?}"))?,
        rho_lat_only: jb2008_density(lat_only)
            .map_err(|error| anyhow::anyhow!("lat-only input must evaluate: {error:?}"))?,
        h_gcrs: ra_sat_gcrs - ra_sun_gcrs,
        h_itrs: ra_sat_itrs - ra_sun_itrs,
        sat_lat_gcrs_deg: lat_gcrs.to_degrees(),
        sat_lat_itrs_deg: lat_itrs.to_degrees(),
    })
}

/// The six-point grid: 200/400/800 km at an equatorial and a polar latitude.
///
/// 80 deg rather than 90: at exactly 90 the ECI position lies on the z axis,
/// `atan2(0, 0)` is 0 by convention rather than by geometry, and the right
/// ascension would carry no information. 80 deg is inside the polar regime --
/// `sin^2(lat)` is already 97% of its pole value -- while keeping the angle
/// well conditioned.
fn grid() -> anyhow::Result<Vec<FrameMeasurement>> {
    let mut out = Vec::new();
    for (alt, alt_label) in [(200.0, "200 km"), (400.0, "400 km"), (800.0, "800 km")] {
        for (lat, lat_label) in [(0.0, "equatorial"), (80.0, "polar (80 deg)")] {
            let label = format!("{alt_label} / {lat_label}");
            out.push(measure_at(alt, lat, &label)?);
        }
    }
    Ok(out)
}

/// THE GATE. Production must feed all four angular arguments from ITRS.
///
/// Tolerance 1e-9 in ratio: the two reductions here are the same arithmetic in
/// the same order, so only the Bowring altitude form differs (one-pass in
/// production, iterated here), which the altitude gate bounds at 1.5e-9 m.
/// The signal this gate exists to catch is ~4e-4 in ratio, five orders larger.
#[test]
fn production_jb2008_angular_arguments_are_all_itrs_reduced() -> anyhow::Result<()> {
    let mut worst = 0.0f64;
    let mut worst_label = String::new();
    for m in grid()? {
        println!("{}", m.report());
        let err = (m.rho_production / m.rho_all_itrs - 1.0).abs();
        if err > worst {
            worst = err;
            worst_label = m.label.clone();
        }
        if !strictly_greater((m.rho_all_itrs / m.rho_all_gcrs - 1.0).abs(), 1e-7) {
            return Err(anyhow::anyhow!(
                "{}: the all-GCRS and all-ITRS conventions are indistinguishable here, so \
                 this point proves nothing about production. Pick a state where they differ.",
                m.label
            ));
        }
        if !strictly_less(err, 1e-9) {
            return Err(anyhow::anyhow!(
                "{}: production density {:.9e} does not match the all-ITRS convention \
                 {:.9e} (ratio {:.12}). It matches all-GCRS at {:.12}. All four of \
                 sun_ra_rad, sun_declination_rad, sat_ra_rad and sat_geocentric_lat_rad \
                 in `jb2008_density_at_state` must be reduced from `pos_itrs` and the \
                 ITRS Sun -- moving fewer than four is worse than moving none.",
                m.label,
                m.rho_production,
                m.rho_all_itrs,
                m.rho_production / m.rho_all_itrs,
                m.rho_production / m.rho_all_gcrs,
            ));
        }
    }
    println!("worst production-vs-all-ITRS ratio error {worst:.3e} at {worst_label}");
    Ok(())
}

/// The partial fix overshoots. This is the reason the four move as a set.
///
/// Not a production assertion -- it is a property of the kernel, and it is what
/// makes "latitude alone" a REGRESSION rather than a half-improvement. If this
/// ever goes green-by-coincidence (the two moves stop opposing), the constraint
/// documented in `rhs.rs` has stopped being true and should be re-derived.
#[test]
fn moving_latitude_alone_overshoots_the_consistent_all_itrs_answer() -> anyhow::Result<()> {
    for m in grid()? {
        let consistent = m.rho_all_itrs / m.rho_all_gcrs - 1.0;
        let partial = m.rho_lat_only / m.rho_all_gcrs - 1.0;
        if strictly_less(consistent.abs(), 1e-12) {
            continue;
        }
        let overshoot = partial / consistent;
        println!(
            "{}: consistent {:+.6}% | lat-only {:+.6}% | overshoot {:.3}x",
            m.label,
            100.0 * consistent,
            100.0 * partial,
            overshoot
        );
        if !strictly_greater(overshoot, 1.5) {
            return Err(anyhow::anyhow!(
                "{}: latitude-alone no longer overshoots ({overshoot:.3}x). The \
                 'all four or none' constraint in `rhs.rs` rests on the RA shift \
                 opposing the latitude shift; re-derive it before trusting either.",
                m.label
            ));
        }
    }
    Ok(())
}

/// The frame tilt is real and is what makes this a defect rather than roundoff.
///
/// Guards against the case where `to_itrs` degenerates to a pure z-rotation --
/// under which every number in this file collapses to zero difference and every
/// assertion above passes vacuously.
#[test]
fn the_gcrs_to_itrs_pole_tilt_is_nonzero_at_the_test_epoch() -> anyhow::Result<()> {
    let m = measure_at(400.0, 80.0, "400 km / polar (80 deg)")?;
    let lat_shift_deg = (m.sat_lat_itrs_deg - m.sat_lat_gcrs_deg).abs();
    let h_shift_rad = m.h_shift_rad().abs();
    println!(
        "pole tilt at 2022-08-12: latitude {lat_shift_deg:.6} deg, \
         hour angle {h_shift_rad:.6e} rad"
    );
    if !strictly_greater(lat_shift_deg, 1e-3) {
        return Err(anyhow::anyhow!(
            "GCRS->ITRS moved the geocentric latitude by only {lat_shift_deg:e} deg; \
             if the rotation has become a pure z-rotation this whole file is vacuous"
        ));
    }
    if !strictly_greater(h_shift_rad, 1e-6) {
        return Err(anyhow::anyhow!(
            "GCRS->ITRS moved the hour angle by only {h_shift_rad:e} rad; the RA \
             difference is supposed to be invariant only to FIRST order"
        ));
    }
    Ok(())
}
