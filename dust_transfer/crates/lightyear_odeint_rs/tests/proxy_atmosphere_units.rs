//! The synthetic thermosphere proxy takes DEGREES. Its geometry source returns RADIANS.
//!
//! # The trap
//!
//! `satpy_core` has two near-namesake reductions with different units:
//!
//! - `satpy_core::geocentric_spherical_from_itrs` -> `(lat_RAD, lon_RAD, alt_km)`
//! - `satpy_core::eci_to_geocentric_spherical`    -> `(lat_DEG, lon_DEG, alt_km)`
//!
//! `jb_rs::synthetic_thermosphere_proxy_eval_impl(jd, lat_deg, lon_deg, alt_km)`
//! takes degrees. Production called the RADIANS one and passed the result
//! straight through, at `rhs.rs` on both the density and the temperature paths.
//!
//! Nothing crashed. A latitude spanning +/-77.6 deg arrived as +/-1.354 "deg",
//! which silently collapses the model's `25*sin^2(lat)` and `0.3*|lat|` terms and
//! flattens its local-time dependence, degenerating the proxy to a function of
//! altitude and epoch alone.
//!
//! This test pins the unit contract at the boundary rather than the call site, so
//! it stays valid if the call is refactored. It is the guard the original defect
//! lacked: both functions returned a plausible `(f64, f64, f64)`, so the compiler
//! could not object and no runtime check existed.

use jb_rs::synthetic_thermosphere_proxy_eval_impl;
/// The same Earth figure the two reductions under test are handed. Imported
/// rather than copied: a local `6378.137` here would let a re-figured
/// `satpy_core::RE` move the functions this file calls while the file kept
/// building its input against the old one, and both assertions below are
/// latitude round-trips that would still pass.
use satpy_core::RE as EARTH_RADIUS_KM;

/// An ITRS position at a high latitude, where the latitude-dependent terms bite.
fn high_latitude_itrs() -> [f64; 3] {
    // ~70 deg geocentric latitude at ~700 km altitude.
    let r = EARTH_RADIUS_KM + 700.0;
    let lat = 70.0_f64.to_radians();
    let lon = 40.0_f64.to_radians();
    [
        r * lat.cos() * lon.cos(),
        r * lat.cos() * lon.sin(),
        r * lat.sin(),
    ]
}

#[test]
fn the_two_geocentric_reductions_really_do_differ_in_units() {
    // If this ever fails, the units were unified upstream and the whole hazard
    // this file guards has been removed at the source - retire the test rather
    // than adjust it.
    let itrs = high_latitude_itrs();
    let (lat_rad, _, _) = satpy_core::geocentric_spherical_from_itrs(&itrs, EARTH_RADIUS_KM);
    assert!(
        (lat_rad - 70.0_f64.to_radians()).abs() < 1.0e-9,
        "geocentric_spherical_from_itrs no longer returns radians (got {lat_rad})"
    );
    assert!(
        lat_rad.abs() < std::f64::consts::FRAC_PI_2 + 1.0e-12,
        "a radian latitude cannot exceed pi/2; got {lat_rad}"
    );

    // The namesake reduction on the SAME vector (identity GMST, so the frame
    // rotation is a no-op and latitude is preserved) must come back in degrees.
    let (lat_deg, _, _) = satpy_core::eci_to_geocentric_spherical(&itrs, 0.0, 1.0, EARTH_RADIUS_KM);
    assert!(
        (lat_deg - 70.0).abs() < 1.0e-9,
        "eci_to_geocentric_spherical no longer returns degrees (got {lat_deg})"
    );
}

/// The proxy must actually RESPOND to latitude. Feeding it radians made it nearly
/// flat, which is what let the defect survive: the output stayed finite and
/// plausible.
#[test]
fn the_proxy_is_latitude_sensitive_in_degrees_and_nearly_flat_in_radians() {
    let jd = 2_459_798.6;
    let alt_km = 700.0;
    let lon_deg = 40.0_f64;

    let (rho_equator, _, ok_a) = synthetic_thermosphere_proxy_eval_impl(jd, 0.0, lon_deg, alt_km);
    let (rho_polar, _, ok_b) = synthetic_thermosphere_proxy_eval_impl(jd, 70.0, lon_deg, alt_km);
    assert!(ok_a && ok_b, "proxy must evaluate at these states");

    let degrees_span = (rho_polar / rho_equator - 1.0).abs();

    // The same physical latitudes and longitude, wrongly handed over as
    // radians. The equator arm converts too: at lat 0 the conversion only
    // moves the longitude, but leaving it in degrees would make this arm a
    // bit-identical re-call of the baseline rather than a radian replay.
    let (rho_equator_rad, _, _) =
        synthetic_thermosphere_proxy_eval_impl(jd, 0.0, lon_deg.to_radians(), alt_km);
    let (rho_polar_rad, _, _) = synthetic_thermosphere_proxy_eval_impl(
        jd,
        70.0_f64.to_radians(),
        lon_deg.to_radians(),
        alt_km,
    );
    let radians_span = (rho_polar_rad / rho_equator_rad - 1.0).abs();

    println!(
        "proxy latitude response: {:.4}% in degrees, {:.4}% when radians are mistaken for degrees",
        100.0 * degrees_span,
        100.0 * radians_span
    );

    assert!(
        degrees_span > 10.0 * radians_span,
        "the proxy's latitude response in degrees ({:.4}%) is not clearly larger than \
         the response it gives when fed radians ({:.4}%); this test can no longer \
         distinguish the unit defect",
        100.0 * degrees_span,
        100.0 * radians_span
    );
}
