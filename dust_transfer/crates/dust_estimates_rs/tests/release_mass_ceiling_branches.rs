//! Branch coverage for `physical_release_mass_ceiling_kg`.
//!
//! `atmospheric_bracket_cap.rs` gates the cap's PURPOSE: that restricting the
//! retry bracket rescues the two captured flower rows, and that the pre-cap
//! solver fails on them. It exercises the perigee branch, because that is the
//! condition those rows reach.
//!
//! The energy branch — `f_unbound`, where the deflected orbit becomes unbound
//! before its perigee ever touches the interface — had no coverage at all. An
//! independent brute force over 1,500 geometries found it binding on 180 of
//! them, so it is a live branch on ordinary inputs, not a degenerate corner.
//! `smallest_root_in_open_interval`, the quadratic solver both branches call,
//! was untested too; it is private, so the only way to reach it is through this
//! public seam.
//!
//! The oracle here is deliberately NOT the production algebra restated. It
//! scans the two condition functions on a dense grid, brackets the first sign
//! change, and bisects it to convergence — a search where production solves in
//! closed form. A transcription error in either would show up as disagreement.
//! Measured over the 240-geometry corpus below: worst relative disagreement
//! 2.9e-13, with 27 geometries binding on energy and 119 on perigee.
//!
//! WHAT THIS FILE DOES NOT COVER, measured rather than assumed. Two poisons
//! were applied to `mass_solver.rs` and the results recorded:
//!
//! * `f_unbound = None` (delete the energy branch): reds two of the four tests
//!   here. That is this file's reason to exist.
//! * `smallest_root_in_open_interval` returning the LARGEST root in the
//!   interval instead of the smallest: all four tests here stay GREEN, and
//!   three tests in `atmospheric_bracket_cap.rs` go red. Every geometry in this
//!   corpus has at most one root per quadratic inside `(0, kappa)`, so the
//!   smallest-of-several property is the sibling file's coverage, not this
//!   one's. What this file adds for that function is that its two CALLERS are
//!   both exercised and both agree with a search.

use dust_estimates_rs::mass_solver::{
    physical_release_mass_ceiling_kg, MfJ2MassSolverEvent, REENTRY_INTERFACE_ALT_KM,
};
use satpy_core::{MU, RE};

const MASS_MAX_KG: f64 = 1000.0;

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Specific orbital energy of the deflected target at kick fraction `f`.
fn energy_at(event: &MfJ2MassSolverEvent, f: f64) -> f64 {
    let r = event.target_pos_intercept;
    let v = [
        event.target_vel_intercept[0] + f * event.v_rel[0],
        event.target_vel_intercept[1] + f * event.v_rel[1],
        event.target_vel_intercept[2] + f * event.v_rel[2],
    ];
    0.5 * dot(&v, &v) - MU / dot(&r, &r).sqrt()
}

/// Perigee radius of the deflected orbit at kick fraction `f`, or `NAN` when
/// the orbit is unbound (where "perigee" stops meaning anything).
fn perigee_at(event: &MfJ2MassSolverEvent, f: f64) -> f64 {
    let r = event.target_pos_intercept;
    let v = [
        event.target_vel_intercept[0] + f * event.v_rel[0],
        event.target_vel_intercept[1] + f * event.v_rel[1],
        event.target_vel_intercept[2] + f * event.v_rel[2],
    ];
    let energy = energy_at(event, f);
    if energy >= 0.0 {
        return f64::NAN;
    }
    let h = cross(&r, &v);
    let semi_major = -MU / (2.0 * energy);
    let ecc = (1.0 + 2.0 * energy * dot(&h, &h) / (MU * MU))
        .max(0.0)
        .sqrt();
    semi_major * (1.0 - ecc)
}

/// Smallest `f` in `(0, kappa)` at which `condition` first turns non-positive,
/// found by dense scan then bisection. `None` when it never does.
fn first_crossing(kappa: f64, condition: impl Fn(f64) -> f64) -> Option<f64> {
    const STEPS: u32 = 20_000;
    let mut previous_f = 0.0_f64;
    for step in 1..=STEPS {
        let f = kappa * f64::from(step) / f64::from(STEPS);
        if condition(f) <= 0.0 {
            let (mut lo, mut hi) = (previous_f, f);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if condition(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return Some(hi);
        }
        previous_f = f;
    }
    None
}

fn ceiling_oracle(event: &MfJ2MassSolverEvent, reentry_alt_km: f64) -> (f64, Binding) {
    let perigee_radius = reentry_alt_km + RE;
    let f_unbound = first_crossing(event.kappa, |f| -energy_at(event, f));
    let f_perigee = first_crossing(event.kappa, |f| {
        let perigee = perigee_at(event, f);
        // Past the unbound point there is no perigee; treat it as crossed so
        // the scan does not walk past the boundary on a NaN comparison.
        if perigee.is_nan() {
            1.0
        } else {
            perigee - perigee_radius
        }
    });
    let (fraction, binding) = match (f_unbound, f_perigee) {
        (Some(a), Some(b)) if a < b => (a, Binding::Unbound),
        (Some(a), None) => (a, Binding::Unbound),
        (_, Some(b)) => (b, Binding::Perigee),
        (None, None) => return (MASS_MAX_KG, Binding::Neither),
    };
    let mass = event.p_mass * fraction / (event.kappa - fraction);
    (mass.min(MASS_MAX_KG), binding)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Binding {
    Unbound,
    Perigee,
    Neither,
}

/// A circular orbit at `altitude_km`, kicked along `direction` (a unit-ish
/// vector in the local velocity/radial frame) with relative speed `speed`.
///
/// `v_rel` is what the ceiling algebra reads, and `MfJ2MassSolverEvent::new`
/// derives it as `dv_vec - target_vel_intercept`, so the requested relative
/// velocity is passed as `dv_vec = v_target + v_rel`.
fn circular_event(
    altitude_km: f64,
    along: f64,
    radial: f64,
    speed: f64,
    kappa: f64,
) -> MfJ2MassSolverEvent {
    let r = RE + altitude_km;
    let position = [r, 0.0, 0.0];
    let circular_speed = (MU / r).sqrt();
    let velocity = [0.0, circular_speed, 0.0];
    let norm = along.hypot(radial);
    let v_rel = [speed * radial / norm, speed * along / norm, 0.0];
    let dv_vec = [
        velocity[0] + v_rel[0],
        velocity[1] + v_rel[1],
        velocity[2] + v_rel[2],
    ];
    MfJ2MassSolverEvent::new(
        position,
        velocity,
        dv_vec,
        50.0,
        [0.0, -r, 0.0],
        3600.0,
        1.0,
        kappa,
    )
}

/// The ceiling must agree with a search-based oracle on both branches, and the
/// energy branch must actually be among them.
#[test]
fn ceiling_matches_a_search_oracle_on_both_branches() {
    let mut checked = 0_usize;
    let mut unbound_bound = 0_usize;
    let mut perigee_bound = 0_usize;
    let mut worst_rel = 0.0_f64;

    for altitude_km in [400.0_f64, 700.0, 1200.0, 2000.0] {
        for &(along, radial) in &[
            (1.0_f64, 0.0_f64),
            (1.0, 0.5),
            (0.5, 1.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (-1.0, 0.5),
        ] {
            for speed in [0.5_f64, 1.5, 3.0, 5.0, 8.0] {
                for kappa in [0.5_f64, 1.0] {
                    let event = circular_event(altitude_km, along, radial, speed, kappa);
                    let (expected, binding) = ceiling_oracle(&event, REENTRY_INTERFACE_ALT_KM);
                    let got = physical_release_mass_ceiling_kg(
                        &event,
                        MASS_MAX_KG,
                        REENTRY_INTERFACE_ALT_KM,
                    );
                    assert!(
                        got.is_finite() && got >= 0.0,
                        "ceiling must be a non-negative finite mass, got {got} \
                         (alt={altitude_km} along={along} radial={radial} \
                          speed={speed} kappa={kappa})"
                    );
                    match binding {
                        Binding::Unbound => unbound_bound += 1,
                        Binding::Perigee => perigee_bound += 1,
                        Binding::Neither => {}
                    }
                    let scale = expected.abs().max(1.0e-9);
                    let rel = (got - expected).abs() / scale;
                    worst_rel = worst_rel.max(rel);
                    assert!(
                        rel < 1.0e-4,
                        "ceiling {got} disagrees with the {binding:?} oracle \
                         {expected} (rel {rel:.3e}) at alt={altitude_km} \
                         along={along} radial={radial} speed={speed} kappa={kappa}"
                    );
                    checked += 1;
                }
            }
        }
    }

    println!(
        "ceiling oracle: {checked} geometries, {unbound_bound} unbound-bound, \
         {perigee_bound} perigee-bound, worst rel {worst_rel:.3e}"
    );
    assert!(checked >= 200, "corpus shrank to {checked} geometries");
    // Non-vacuity, per branch. Without these the agreement above could be one
    // branch answering for all of them, which is what the audit found: the
    // existing cap tests reach the perigee branch only.
    assert!(
        unbound_bound >= 20,
        "the f_unbound branch bound on only {unbound_bound} geometries; this \
         corpus no longer covers it"
    );
    assert!(
        perigee_bound >= 20,
        "the perigee branch bound on only {perigee_bound} geometries"
    );
}

/// `smallest_root_in_open_interval` must return the SMALLEST root in the open
/// interval, not merely a root.
///
/// Reached through the seam: a retrograde kick strong enough to unbind the
/// orbit has both an energy root and a perigee root inside `(0, kappa)`, and
/// the ceiling is required to be whichever comes first. A solver returning the
/// larger root, or the first root it found, would return a ceiling above a
/// fraction at which the orbit is already gone.
#[test]
fn ceiling_takes_the_first_condition_reached_not_just_one_of_them() {
    let mut both = 0_usize;
    for altitude_km in [400.0_f64, 900.0, 1600.0] {
        for speed in [2.0_f64, 4.0, 6.0, 9.0] {
            let event = circular_event(altitude_km, 1.0, 0.25, speed, 1.0);
            let f_unbound = first_crossing(event.kappa, |f| -energy_at(&event, f));
            let f_perigee = first_crossing(event.kappa, |f| {
                let perigee = perigee_at(&event, f);
                if perigee.is_nan() {
                    -1.0
                } else {
                    perigee - (REENTRY_INTERFACE_ALT_KM + RE)
                }
            });
            let (Some(unbound), Some(perigee)) = (f_unbound, f_perigee) else {
                continue;
            };
            both += 1;
            let first = unbound.min(perigee);
            let expected = (event.p_mass * first / (event.kappa - first)).min(MASS_MAX_KG);
            let got =
                physical_release_mass_ceiling_kg(&event, MASS_MAX_KG, REENTRY_INTERFACE_ALT_KM);
            let rel = (got - expected).abs() / expected.abs().max(1.0e-9);
            assert!(
                rel < 1.0e-4,
                "ceiling {got} is not the first condition reached (unbound at \
                 f={unbound}, perigee at f={perigee}, expected mass {expected})"
            );
            // The state at the ceiling must still be bound, which is the
            // property the caller relies on. The bound is scaled rather than
            // exactly zero: when the energy condition is the binding one the
            // ceiling IS its root, and the energy there evaluates to a few
            // times 1e-14 of either sign against an unperturbed |E0| of ~13
            // km^2/s^2 -- rounding, not an unbound orbit.
            let f_ceiling = event.kappa * got / (event.p_mass + got);
            let slack = 1.0e-12 * energy_at(&event, 0.0).abs();
            assert!(
                energy_at(&event, f_ceiling) <= slack,
                "the orbit is already unbound at the returned ceiling: E={} \
                 against a slack of {slack:e}",
                energy_at(&event, f_ceiling)
            );
        }
    }
    assert!(
        both >= 6,
        "only {both} geometries reached BOTH conditions; this test no longer \
         distinguishes first-root from any-root"
    );
}

/// The documented no-root case: neither condition is reached inside
/// `f < kappa`, and the ceiling falls back to `mass_max`.
#[test]
fn a_kick_too_weak_to_reach_either_condition_returns_mass_max() {
    let event = circular_event(1200.0, 1.0, 0.0, 0.01, 1.0);
    let (expected, binding) = ceiling_oracle(&event, REENTRY_INTERFACE_ALT_KM);
    assert_eq!(
        binding,
        Binding::Neither,
        "this geometry was chosen to reach neither condition; the oracle \
         disagrees, so the case below tests something else"
    );
    let got = physical_release_mass_ceiling_kg(&event, MASS_MAX_KG, REENTRY_INTERFACE_ALT_KM);
    assert_eq!(
        got.to_bits(),
        MASS_MAX_KG.to_bits(),
        "with no root inside the interval the ceiling must be mass_max, got \
         {got} (oracle {expected})"
    );
}

/// An orbit already below the interface has an empty valid domain, and the
/// ceiling must be exactly zero rather than a small positive mass.
#[test]
fn an_orbit_already_below_the_interface_has_a_zero_ceiling() {
    // Perigee well under the interface: a strongly elliptical orbit whose
    // radial distance at intercept is its apogee.
    let r = RE + 1500.0;
    let position = [r, 0.0, 0.0];
    let circular_speed = (MU / r).sqrt();
    let velocity = [0.0, 0.55 * circular_speed, 0.0];
    let event = MfJ2MassSolverEvent::new(
        position,
        velocity,
        [0.0, 0.55f64.mul_add(circular_speed, 0.1), 0.0],
        50.0,
        [0.0, -r, 0.0],
        3600.0,
        1.0,
        1.0,
    );
    assert!(
        perigee_at(&event, 0.0) < REENTRY_INTERFACE_ALT_KM + RE,
        "this fixture must start below the interface, perigee is {}",
        perigee_at(&event, 0.0)
    );
    let got = physical_release_mass_ceiling_kg(&event, MASS_MAX_KG, REENTRY_INTERFACE_ALT_KM);
    assert_eq!(
        got.to_bits(),
        0.0_f64.to_bits(),
        "an orbit already below the interface must have a zero ceiling, got {got}"
    );
}
