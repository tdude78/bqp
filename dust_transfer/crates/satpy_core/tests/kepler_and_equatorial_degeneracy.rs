//! Regression gate for two verified defects in `satpy_core`'s conversion API.
//!
//! Adopted from the round-8 physics/numerics audit probe, where all of these
//! FAILED. They are gates now: each one names the exact region the shipped
//! in-crate tests stepped around, so the dodge cannot silently return.
//!
//! 1. `solve_kepler_e_simd` seeded `+pi` in every lane above the high-e
//!    threshold where the scalar seeds `sign(M) * pi`. For a wrapped `M < 0`
//!    the Halley iterate ran away and the step-only convergence gate froze it
//!    far from any root, silently. 747 of 145,440 grid samples came back with
//!    a Kepler residual up to 3.0616 rad.
//! 2. Exactly equatorial states destroyed the argument of periapsis in both
//!    `equinoc2kep_impl` (`atan2(+-0.0, +-0.0)`) and `eci2kep_impl` (argp left
//!    at zero whenever the node vector vanished). Round-trip position error
//!    reached 958 km through the equinoctial path and 12,426 km through ECI.

use satpy_core::{
    eci2kep_impl, equinoc2kep_impl, kep2eci_impl, kep2equinoc_impl, mean_to_true_anomaly_impl,
    mean_to_true_anomaly_simd, solve_kepler_e, solve_kepler_e_simd,
};
use wide::f64x4;

fn wrap_pi(x: f64) -> f64 {
    let t = x.rem_euclid(std::f64::consts::TAU);
    if t > std::f64::consts::PI {
        t - std::f64::consts::TAU
    } else {
        t
    }
}

/// Independent oracle: |E - e sin E - M|, everything reduced to (-pi, pi].
fn kepler_residual(anomaly: f64, m: f64, e: f64) -> f64 {
    wrap_pi(anomaly - e * anomaly.sin() - wrap_pi(m)).abs()
}

fn simd_eccentric_anomaly(m: f64, e: f64) -> f64 {
    solve_kepler_e_simd(f64x4::splat(m), f64x4::splat(e))
        .2
        .to_array()[0]
}

/// The single case the audit reported: e just past the high-e threshold, with M
/// in `(pi, 2pi)` so the wrapped mean anomaly is negative. This is the sample
/// that returned a 3.010 rad residual.
#[test]
fn simd_kepler_solves_the_case_the_wrong_seed_ran_away_from() {
    let (e, m) = (0.802_f64, 3.267_256_359_733_385_f64);
    let simd = simd_eccentric_anomaly(m, e);
    let scalar = solve_kepler_e(m, e);
    println!("e={e} M={m}");
    println!(
        "  scalar E = {scalar:.12}  residual = {:.3e}",
        kepler_residual(scalar, m, e)
    );
    println!(
        "  SIMD   E = {simd:.12}  residual = {:.3e}",
        kepler_residual(simd, m, e)
    );
    assert!(
        kepler_residual(simd, m, e) < 1e-12,
        "solve_kepler_e_simd returned E={simd} with Kepler residual {} rad",
        kepler_residual(simd, m, e)
    );
}

/// The whole `(e, M)` grid, judged against Kepler's equation itself rather than
/// against the scalar solver, so a shared defect could not hide. 145,440
/// samples: e up to 0.99, M over a full revolution.
///
/// This is the anti-dodge gate. The failing region was `M in (pi, 2pi)` at
/// `e >= 0.8019` — precisely where the in-crate lane tests had no sample.
#[test]
fn simd_kepler_solves_the_whole_grid() {
    let mut bad = 0usize;
    let mut total = 0usize;
    let mut worst = (0.0f64, 0.0f64, 0.0f64);
    let mut worst_scalar = 0.0f64;
    let mut lowest_bad_e = f64::INFINITY;
    // The uniform grid stops at e = 0.99 while `E_CLAMP_MAX_X4` admits
    // e = 1 - 1e-12, so the decade band 0.99 < e < 1 was never sampled -- and
    // that band is exactly where a step-only convergence test stops implying a
    // small residual, because near e = 1 the Newton step shrinks while the
    // residual does not. A uniform grid cannot reach it: the interesting
    // structure is logarithmic in (1 - e). Sample both.
    let uniform = (0..=100).map(|ei| f64::from(ei) * 0.0099);
    let decades = (2..=12).map(|k| 1.0 - 10.0_f64.powi(-k));
    for e in uniform.chain(decades) {
        for mi in 0..1440 {
            let m = f64::from(mi) * (std::f64::consts::TAU / 1440.0);
            total += 1;
            let scalar_residual = kepler_residual(solve_kepler_e(m, e), m, e);
            worst_scalar = worst_scalar.max(scalar_residual);
            let r = kepler_residual(simd_eccentric_anomaly(m, e), m, e);
            if r > worst.2 {
                worst = (e, m, r);
            }
            if r > 1e-12 {
                bad += 1;
                lowest_bad_e = lowest_bad_e.min(e);
            }
        }
    }
    println!("samples {total}; over 1e-12: {bad}; lowest such e = {lowest_bad_e}");
    println!(
        "worst SIMD residual {:.4e} rad at e={} M={}",
        worst.2, worst.0, worst.1
    );
    println!("worst scalar residual {worst_scalar:.4e} rad");
    assert_eq!(
        bad, 0,
        "solve_kepler_e_simd does not solve Kepler's equation"
    );
}

/// The high-e seed reached the public anomaly conversion, so gate that too.
#[test]
fn simd_mean_to_true_anomaly_tracks_the_scalar_past_pi() {
    // Four lanes all inside the previously-failing region, and deliberately
    // MIXED: two lanes take the high-e pi seed and two do not, so a fix that
    // only works when the whole vector is high-e would still fail here.
    let m_arr = [3.267_256_359_733_385, 4.5, 5.9, 3.5];
    let e_arr = [0.802, 0.95, 0.99, 0.3];
    let nu_simd = mean_to_true_anomaly_simd(f64x4::new(m_arr), f64x4::new(e_arr)).to_array();
    for (i, ((m, e), nu)) in m_arr.into_iter().zip(e_arr).zip(nu_simd).enumerate() {
        let nu_scalar = mean_to_true_anomaly_impl(m, e);
        println!("lane {i}: e={e} M={m} scalar={nu_scalar:.12} simd={nu:.12}");
        assert!(
            (nu_scalar - nu).abs() < 1e-9,
            "lane {i} (e={e}, M={m}) off by {} rad",
            (nu_scalar - nu).abs()
        );
    }
}

/// `e == KEPLER_GUESS_E` exactly: the scalar takes the pi seed there
/// (`!(e < 0.8)`), and the SIMD used `simd_gt`, so it took the other branch.
/// The lane test that covered e = 0.8 therefore exercised neither path's twin.
#[test]
fn simd_kepler_agrees_with_the_scalar_at_the_seed_threshold_exactly() {
    for m in [0.5, 2.0, 3.2, 4.0, 5.5, 6.0] {
        let simd = simd_eccentric_anomaly(m, 0.8);
        let r = kepler_residual(simd, m, 0.8);
        println!("e=0.8 M={m}: SIMD E={simd:.12} residual={r:.3e}");
        assert!(r < 1e-12, "e=0.8 M={m}: residual {r} rad");
    }
}

fn kep_to_eci(kep_deg: &[f64; 6]) -> [f64; 6] {
    let mut eci = [0.0f64; 6];
    kep2eci_impl(kep_deg, true, 0.0, 0.0, true, &mut eci);
    eci
}

fn position_error_km(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// FINDING 2a. Exactly equatorial equinoctial elements have `p = q = 0`, and
/// the argp recovery `atan2(h q - k p, k q + h p)` was then `atan2(+-0.0, +-0.0)`
/// — decided by sign bits. argp = 45 deg came back as 0 deg (958 km of position
/// error); argp = 200 deg came back as 180 deg (476 km).
///
/// The fix reports `raan = 0` and folds the whole longitude of periapsis into
/// argp, so the recovered elements are relabelled but describe the same orbit.
/// Judge that by POSITION, not by the labels.
#[test]
fn equinoctial_round_trip_preserves_equatorial_orbits() {
    for argp_deg in [0.0_f64, 45.0, 90.0, 180.0, 200.0, 359.5] {
        for e in [0.0, 1e-11, 0.001, 0.1, 0.7] {
            let kep = [7000.0, e, 0.0_f64, 0.0, argp_deg, 10.0];
            let mut equ = [0.0f64; 6];
            kep2equinoc_impl(&kep, true, true, &mut equ);
            let mut back = [0.0f64; 6];
            equinoc2kep_impl(&equ, true, true, &mut back);
            let back: [f64; 6] = back;
            let err = position_error_km(&kep_to_eci(&kep), &kep_to_eci(&back));
            println!(
                "e={e} argp_in={argp_deg:>6} -> p={} q={} raan_out={} argp_out={} err={err:.3e} km",
                equ[3], equ[4], back[3], back[4]
            );
            assert!(
                err < 1e-9,
                "equatorial round trip moved the orbit {err} km (e={e}, argp={argp_deg})"
            );
            // The relabelling convention itself: raan is reported as zero and
            // argp carries raan + argp.
            assert!(
                back[3] == 0.0,
                "equatorial raan should be pinned to 0, got {}",
                back[3]
            );
        }
    }
}

/// Continuity across `EQUATORIAL_PQ_TOL`: the guard must not introduce a jump
/// in the ORBIT. `i = 1e-9` deg round-tripped to 9e-8 km before the fix and
/// sits above the threshold, so it must still take the general branch and must
/// not regress.
#[test]
fn equinoctial_round_trip_is_continuous_across_the_equatorial_guard() {
    let mut previous: Option<(f64, [f64; 6])> = None;
    for inc_deg in [
        0.0, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-6, 1e-3, 0.5, 5.0,
    ] {
        let kep = [7000.0, 0.1, inc_deg, 30.0, 45.0, 10.0];
        let mut equ = [0.0f64; 6];
        kep2equinoc_impl(&kep, true, true, &mut equ);
        let mut back = [0.0f64; 6];
        equinoc2kep_impl(&equ, true, true, &mut back);
        let back: [f64; 6] = back;
        let eci = kep_to_eci(&back);
        let err = position_error_km(&kep_to_eci(&kep), &eci);
        println!(
            "i={inc_deg:e} deg: round-trip err {err:.3e} km, raan_out={}",
            back[3]
        );
        assert!(err < 1e-6, "i={inc_deg} deg round trip moved {err} km");
        if let Some((previous_inc, previous_eci)) = previous {
            let jump = position_error_km(&previous_eci, &eci);
            // Neighbouring inclinations differ by at most ~5 deg here; the
            // position may differ, but crossing the guard at 1e-12 must not add
            // a jump of its own. Bound the step by what the inclination change
            // alone can produce: a * di, generously.
            let bound = 7000.0 * (inc_deg - previous_inc).abs().to_radians() + 1e-6;
            assert!(
                jump <= bound,
                "crossing i={previous_inc} -> {inc_deg} deg jumped {jump} km, bound {bound} km"
            );
        }
        previous = Some((inc_deg, eci));
    }
}

/// FINDING 2b. `eci2kep_impl` left argp at zero whenever the node vector
/// vanished, discarding the orientation of an equatorial ellipse: 4,828 km of
/// round-trip position error at argp = 45 deg, 12,426 km at argp = 200 deg.
/// `i = 180` deg reaches the same branch because sin(pi) does not round to
/// zero — `node_norm` lands near 6e-12, under `NODE_VECTOR_FLOOR_KM2_PER_S`.
///
/// Judged against INCLINED controls at the same (e, argp, nu), because
/// `eci2kep_impl` has a conditioning floor here that has nothing to do with
/// inclination. It recovers the true anomaly as `acos(e_hat . r_hat)`, and at
/// nu = 0 that argument sits at 1, so whether it rounds to exactly 1.0 or to
/// 1 - 1ulp is a coin flip. Landing one ulp low gives nu = 1.5e-8 rad, and the
/// `rv < 0` quadrant test then reports it as 2pi - 1.5e-8, i.e. 3.1e-5 km of
/// position error at e = 0.7. Measured at i = 0, 1, 2, 10, 45, 90, 120, 180 deg
/// and clean at i = 0.5, 5, 28.5, 51.6, 165 — a coin flip, not a degeneracy.
///
/// So the bound is the WORST of several controls spanning the Part A
/// inclination range rather than a fixed number. The defect being gated was
/// 4,828 km, eight orders above that floor.
#[test]
fn eci_round_trip_preserves_equatorial_orbits() {
    fn round_trip_error(kep: &[f64; 6]) -> (f64, [f64; 6]) {
        let eci = kep_to_eci(kep);
        let mut back = [0.0f64; 6];
        eci2kep_impl(&eci, true, true, &mut back);
        let back: [f64; 6] = back;
        (position_error_km(&eci, &kep_to_eci(&back)), back)
    }

    let mut worst = 0.0f64;
    for inclination_deg in [0.0_f64, 180.0] {
        for argp_deg in [0.0_f64, 45.0, 90.0, 180.0, 200.0, 300.0] {
            for e in [0.0, 0.001, 0.1, 0.7] {
                for nu_deg in [0.0_f64, 10.0, 130.0, 250.0] {
                    let (err, back) =
                        round_trip_error(&[7000.0, e, inclination_deg, 0.0, argp_deg, nu_deg]);
                    // Controls span the Part A inclination bounds [0.5, 165],
                    // all far outside the degenerate branch, so their worst
                    // case is the conditioning floor of the shared nu recovery.
                    let control = [0.5_f64, 45.0, 90.0, 165.0]
                        .into_iter()
                        .map(|i| round_trip_error(&[7000.0, e, i, 0.0, argp_deg, nu_deg]).0)
                        .fold(0.0_f64, f64::max);
                    worst = worst.max(err);
                    assert!(
                        err <= control + 1e-9,
                        "i={inclination_deg} e={e} argp={argp_deg} nu={nu_deg}: \
                         ECI round trip moved {err} km against a {control} km inclined \
                         control (recovered {back:?})"
                    );
                }
            }
        }
    }
    println!("worst equatorial ECI round-trip error {worst:.3e} km");
    // Report the two headline cases explicitly.
    for (inclination_deg, argp_deg) in [(0.0_f64, 45.0_f64), (0.0, 200.0), (180.0, 45.0)] {
        let kep = [7000.0, 0.1, inclination_deg, 0.0, argp_deg, 10.0];
        let eci = kep_to_eci(&kep);
        let mut back = [0.0f64; 6];
        eci2kep_impl(&eci, true, true, &mut back);
        let back: [f64; 6] = back;
        println!(
            "i={inclination_deg} argp_in={argp_deg} -> argp_out={:.9} err={:.3e} km",
            back[4],
            position_error_km(&eci, &kep_to_eci(&back))
        );
    }
}

/// The inclined cases the equatorial guards must leave untouched. If either fix
/// widened its branch past the degenerate point this goes red.
#[test]
fn inclined_orbits_are_untouched_by_the_equatorial_guards() {
    for inclination_deg in [0.5_f64, 28.5, 51.6, 90.0, 98.7, 165.0] {
        for raan_deg in [0.0_f64, 75.0, 300.0] {
            for argp_deg in [0.0_f64, 45.0, 200.0] {
                let kep = [7000.0, 0.1, inclination_deg, raan_deg, argp_deg, 10.0];
                let eci = kep_to_eci(&kep);
                let mut back = [0.0f64; 6];
                eci2kep_impl(&eci, true, true, &mut back);
                let back: [f64; 6] = back;
                for (i, (got, want)) in back.into_iter().zip(kep).enumerate() {
                    assert!(
                        (got - want).abs() < 1e-9,
                        "eci2kep element {i}: got {got}, want {want} (i={inclination_deg})"
                    );
                }

                let mut equ = [0.0f64; 6];
                kep2equinoc_impl(&kep, true, true, &mut equ);
                let mut equ_back = [0.0f64; 6];
                equinoc2kep_impl(&equ, true, true, &mut equ_back);
                for (i, (got, want)) in equ_back.into_iter().zip(kep).enumerate() {
                    assert!(
                        (got - want).abs() < 1e-9,
                        "equinoc2kep element {i}: got {got}, want {want} (i={inclination_deg})"
                    );
                }
            }
        }
    }
}

/// E = pi is apoapsis, and the old SIMD formula was singular there.
///
/// It computed `nu = 2*atan(sqrt((1+e)/(1-e)) * sin E / (1 + cos E))`, whose
/// denominator `1 + cos E` vanishes at E = pi for EVERY eccentricity. The test
/// grid never sampled it: the four lanes were M = 0.5, 2.0, 5.5, 3.267, none of
/// which lands there. And M = pi puts E at pi exactly, for any e, because
/// M = E - e*sin(E) and sin(pi) = 0 -- so this is not an exotic input, it is
/// apoapsis of every orbit in the catalogue.
///
/// MEASURED, so nobody re-derives the scary version: the OLD form passes this
/// test too. `sin(pi)` in f64 is 1.2246e-16 rather than 0, so the quotient is
/// x/0 = +inf rather than 0/0 = NaN, and `atan(+inf)` is exactly pi/2, which
/// doubles to pi -- the right answer, reached by an accident of IEEE division.
/// This test therefore does NOT record a live failure; it pins the property so
/// the new form keeps it, and it is the domain-policy test below that
/// discriminates the two implementations.
#[test]
fn simd_mean_to_true_anomaly_is_finite_and_scalar_matching_at_apoapsis() {
    let pi = std::f64::consts::PI;
    let m_arr = [pi, pi, pi, pi];
    let e_arr = [0.0, 0.3, 0.802, 0.99];
    let nu_simd = mean_to_true_anomaly_simd(f64x4::new(m_arr), f64x4::new(e_arr)).to_array();

    for (i, ((m, e), nu)) in m_arr.into_iter().zip(e_arr).zip(nu_simd).enumerate() {
        assert!(
            nu.is_finite(),
            "lane {i} (e={e}) returned {nu} at E=pi; the half-tangent form is 0/0 there"
        );
        let nu_scalar = mean_to_true_anomaly_impl(m, e);
        assert!(
            (nu_scalar - nu).abs() < 1e-9,
            "lane {i} (e={e}) SIMD {nu} vs scalar {nu_scalar} at apoapsis"
        );
        // Apoapsis: the true anomaly is pi regardless of eccentricity.
        assert!(
            (nu - pi).abs() < 1e-9,
            "lane {i} (e={e}) put apoapsis at {nu}, not pi"
        );
    }
}

/// |e| >= 1 has no true anomaly on this branch, and both twins must say so.
///
/// The scalar has always returned NaN. The SIMD clamped to 1 - 1e-12 instead
/// and returned a fabricated elliptical answer for a hyperbolic orbit -- a
/// disagreement between two functions documented as the same computation, and
/// the more dangerous direction, since the vectorized path is the one that runs
/// in bulk.
#[test]
fn simd_mean_to_true_anomaly_refuses_non_elliptical_eccentricity() {
    let m_arr = [1.0, 1.0, 1.0, 1.0];
    let e_arr = [1.0, 1.5, -1.0, 0.5];
    let nu_simd = mean_to_true_anomaly_simd(f64x4::new(m_arr), f64x4::new(e_arr)).to_array();

    for (i, ((m, e), nu)) in m_arr.into_iter().zip(e_arr).zip(nu_simd).enumerate() {
        let nu_scalar = mean_to_true_anomaly_impl(m, e);
        assert_eq!(
            nu.is_nan(),
            nu_scalar.is_nan(),
            "lane {i} (e={e}): SIMD gave {nu}, scalar gave {nu_scalar} -- the \
             twins must agree on where they refuse"
        );
        if e.abs() < 1.0 {
            assert!(nu.is_finite(), "lane {i} (e={e}) must still answer");
        }
    }
}
