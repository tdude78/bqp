//! The dense-arc segment window is physics, not setup overhead.
//!
//! A dense strict-HF arc is built as ~2,016 sequential 600 s segments, and every
//! segment re-enters `with_ephemeris_for_arc` with its own window. That call
//! looks like per-segment setup worth hoisting to once per object -- it reloads
//! driver authorities and revalidates catalogue coverage every time. It is not.
//! Third-body positions are resolved at the MIDPOINT of the window
//! (`jd_anchor = 0.5 * jd_a + 0.5 * jd_b`), so the window reaches the resolved
//! `ForceConfig` and therefore the RHS.
//!
//! This pins that. Hoisting the call out of the segment loop would replace each
//! segment's Sun position with the whole-arc midpoint's, which for a 14-day arc
//! is a different point in the solar system by millions of kilometres.

use lightyear_odeint_rs::types::{ForceConfig, ForceFlags};

const T0_JD_UTC: f64 = 2_461_270.225_335_648_3;
const SEC_PER_DAY: f64 = 86_400.0;
const SEGMENT_S: f64 = 600.0;
const ARC_DAYS: f64 = 14.0;

/// Sealed Part A strict-HF body forces. Mirrors
/// `two_phase_transfer_rs::StrictHfForceAuthority::PART_A`; that crate depends
/// on this one, so the constant cannot be imported here.
fn part_a_config() -> ForceConfig {
    ForceConfig {
        sph_order: 5,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        atm_model: 8,
        am_ratio: 0.01,
        cd: 2.2,
        cr: 1.3,
        ..ForceConfig::default()
    }
}

fn separation_km(first: [f64; 3], second: [f64; 3]) -> f64 {
    let dx = first[0] - second[0];
    let dy = first[1] - second[1];
    let dz = first[2] - second[2];
    dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
}

#[test]
fn segment_window_resolves_third_bodies_and_cannot_be_hoisted() {
    let segment_days = SEGMENT_S / SEC_PER_DAY;
    let base = part_a_config();

    let first = base
        .with_ephemeris_for_arc(T0_JD_UTC, T0_JD_UTC + segment_days)
        .expect("first segment window resolves");
    let second = base
        .with_ephemeris_for_arc(T0_JD_UTC + segment_days, T0_JD_UTC + 2.0 * segment_days)
        .expect("second segment window resolves");
    let whole = base
        .with_ephemeris_for_arc(T0_JD_UTC, T0_JD_UTC + ARC_DAYS)
        .expect("whole-arc window resolves");

    // Both bodies must actually be catalogue-resolved, or the rest is vacuous:
    // a config that resolved nothing would compare equal for the wrong reason.
    assert_ne!(
        first.dynamic_ephemeris_flags & ForceFlags::SUN_GRAVITY,
        0,
        "Sun must be dynamically resolved under JB2008 drag"
    );
    assert_ne!(
        first.dynamic_ephemeris_flags & ForceFlags::MOON_GRAVITY,
        0,
        "Moon must be dynamically resolved"
    );

    let first_sun = first.sun_pos.expect("first window resolves a Sun position");
    let second_sun = second
        .sun_pos
        .expect("second window resolves a Sun position");
    let whole_sun = whole.sun_pos.expect("whole arc resolves a Sun position");
    let first_moon = first
        .moon_pos
        .expect("first window resolves a Moon position");
    let second_moon = second
        .moon_pos
        .expect("second window resolves a Moon position");

    // Adjacent segments disagree: the window is not a formality.
    assert!(
        separation_km(first_sun, second_sun) > 1.0,
        "adjacent 600 s segments must resolve different Sun positions, got {} km",
        separation_km(first_sun, second_sun)
    );
    assert!(
        separation_km(first_moon, second_moon) > 1.0,
        "adjacent 600 s segments must resolve different Moon positions, got {} km",
        separation_km(first_moon, second_moon)
    );

    // And the hoist a caller would be tempted by is far worse than a rounding
    // difference: one whole-arc call is a different Sun by millions of km.
    assert!(
        separation_km(first_sun, whole_sun) > 1.0e6,
        "a single whole-arc resolution must not be mistakable for the first \
         segment's, got {} km",
        separation_km(first_sun, whole_sun)
    );
}
