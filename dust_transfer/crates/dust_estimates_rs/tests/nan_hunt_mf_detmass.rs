//! PROBE (nan-hunt branch): one-call reproduction of the `det_mass ... got NaN`
//! abort that killed a sealed Exact36 nsga2/flower cell in the blind H64 replay.
//!
//! Inputs are the exact `DetMassRowInputs` captured from row 276 of event
//! `evt_20898_20563_212514662598` under seed 41127203, pop 4, design
//! x = [1904.303099899751, 88.75388882658235, 0.16693367405687046,
//!      196.4690859990079, 3.0, 3.0, 1.0].
//!
//! `cargo test --release -p dust_estimates_rs --test nan_hunt_mf_detmass -- --nocapture`

use dust_estimates_rs::mass_solver::{
    solve_single_event_mf_j2_unconstrained_bracket, solve_single_event_mf_j2_with_status,
    MfJ2MassSolveStatusCode, MfJ2MassSolverEvent, SolverConfig,
};
use satpy_core::{eci2equinoc_impl, equinoc_prop_j2_from_impl, MU};

#[expect(
    clippy::inconsistent_digit_grouping,
    reason = "captured production values; digits are bit-exact evidence and must not be edited"
)]
const TARGET_POS: [f64; 3] = [
    666.354_001_014_283_1,
    -2237.979_584_736_737_7,
    -6663.785_651_912_341,
];
const TARGET_VEL: [f64; 3] = [
    1.935_632_268_969_379_2,
    -6.800_088_428_550_588,
    2.477_168_251_036_653_5,
];
const DV_VEC: [f64; 3] = [
    -2.543_021_665_074_745_5,
    7.148_671_179_188_607,
    -2.141_705_688_340_100_4,
];
#[expect(
    clippy::inconsistent_digit_grouping,
    reason = "captured production values; digits are bit-exact evidence and must not be edited"
)]
const OTHER_CONJ: [f64; 3] = [
    359.780_610_720_872_35,
    -1161.215_882_391_510_1,
    -6955.256_925_339_665,
];
const P_MASS_KG: f64 = 31.0;
#[expect(
    clippy::inconsistent_digit_grouping,
    reason = "captured production values; digits are bit-exact evidence and must not be edited"
)]
const TOF_S: f64 = 46741.434_985_399_246;
const MIN_MISS_KM: f64 = 1.0;

// `nd_config::CompiledPartAScienceV1::part_a_v1()` production values.
const KAPPA: f64 = 1.0;
const XTOL_KG: f64 = 1.0e-6;
const RTOL: f64 = 1.0e-5;
const MAXITER: usize = 50;
const MASS_MAX_KG: f64 = 1000.0;

fn event() -> MfJ2MassSolverEvent {
    MfJ2MassSolverEvent::new(
        TARGET_POS,
        TARGET_VEL,
        DV_VEC,
        P_MASS_KG,
        OTHER_CONJ,
        TOF_S,
        MIN_MISS_KM,
        KAPPA,
    )
}

const fn config() -> SolverConfig {
    SolverConfig {
        xtol: XTOL_KG,
        rtol: RTOL,
        maxiter: MAXITER,
        mass_max: MASS_MAX_KG,
    }
}

/// The released cloud's post-transfer inertial velocity, verbatim from
/// `compute_new_velocity_mf_j2`: a linear blend toward `dv_vec` in the
/// momentum-transfer fraction `kappa * m / (p_mass + m)`.
fn new_velocity(mass: f64) -> [f64; 3] {
    let fraction = KAPPA * mass / (P_MASS_KG + mass);
    [
        TARGET_VEL[0] + fraction * (DV_VEC[0] - TARGET_VEL[0]),
        TARGET_VEL[1] + fraction * (DV_VEC[1] - TARGET_VEL[1]),
        TARGET_VEL[2] + fraction * (DV_VEC[2] - TARGET_VEL[2]),
    ]
}

/// Semi-major axis, eccentricity and perigee radius of the released orbit.
fn orbit(mass: f64) -> (f64, f64, f64) {
    let velocity = new_velocity(mass);
    let r = TARGET_POS
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let v_sq = velocity.iter().map(|value| value * value).sum::<f64>();
    let a = 1.0 / (2.0 / r - v_sq / MU);
    let h = [
        TARGET_POS[1] * velocity[2] - TARGET_POS[2] * velocity[1],
        TARGET_POS[2] * velocity[0] - TARGET_POS[0] * velocity[2],
        TARGET_POS[0] * velocity[1] - TARGET_POS[1] * velocity[0],
    ];
    let h_sq = h.iter().map(|value| value * value).sum::<f64>();
    let e = (1.0 - h_sq / (MU * a)).max(0.0).sqrt();
    (a, e, a * (1.0 - e))
}

/// Re-derivation of `compute_miss_distance_mf_j2` on the MF-J2 authority, so the
/// probe can see WHICH sub-step goes non-finite for a given trial mass.
fn miss_distance(mass: f64) -> (f64, &'static str) {
    let velocity = new_velocity(mass);
    let state = [
        TARGET_POS[0],
        TARGET_POS[1],
        TARGET_POS[2],
        velocity[0],
        velocity[1],
        velocity[2],
    ];
    let mut equ = [0.0_f64; 6];
    eci2equinoc_impl(&state, 6, 0.0, 0.0, &mut equ);
    if !equ[0].is_finite() || equ[0] <= 0.0 {
        return (f64::NAN, "eci2equinoc: invalid orbit");
    }
    let mut propagated = [0.0_f64; 6];
    equinoc_prop_j2_from_impl(&equ, TOF_S, &mut propagated);
    if !propagated.iter().all(|value| value.is_finite()) {
        return (f64::NAN, "equinoc_prop_j2: non-finite");
    }
    let dx = propagated[0] - OTHER_CONJ[0];
    let dy = propagated[1] - OTHER_CONJ[1];
    let dz = propagated[2] - OTHER_CONJ[2];
    (dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt(), "ok")
}

/// The captured row reproduces the campaign abort exactly: a NaN root mass with
/// status `MidNonFinite`, in one call and well under a second.
///
/// LEDGER. This used to assert the same of the SHIPPED solver, and that
/// assertion pinned the defect rather than the physics. The atmospheric bracket
/// cap now solves this row (see `tests/atmospheric_bracket_cap.rs`), so the
/// reproduction moves to the pre-cap entry point, which is retained precisely so
/// the original abort stays demonstrable. The row's reported deterministic mass
/// moved from NaN — substituted by the Stage 3 guard as the 1000 kg hard limit
/// and reported infeasible — to 1.4611052102062132e-5 kg, reported feasible.
#[test]
fn captured_flower_row_reproduces_nan_det_mass_before_the_cap() {
    let result = solve_single_event_mf_j2_unconstrained_bracket(&event(), &config());
    println!(
        "pre-cap: root_mass_kg={} status={:?} iterations={} miss0={} miss_upper={}",
        result.root_mass_kg,
        result.status,
        result.iterations,
        result.miss_at_zero_km,
        result.miss_at_upper_km
    );
    assert!(result.root_mass_kg.is_nan(), "expected a NaN root mass");
    assert_eq!(result.status, MfJ2MassSolveStatusCode::MidNonFinite);
}

/// ...and the shipped solver now recovers it, to the bit value the cap's own
/// gate pins. Stated here too so this file cannot drift back into asserting the
/// bug: whichever of the two tests is read first, the pair fixes the direction.
#[test]
fn the_shipped_solver_now_solves_the_captured_row() {
    let result = solve_single_event_mf_j2_with_status(&event(), &config());
    assert_eq!(result.status, MfJ2MassSolveStatusCode::Converged);
    assert_eq!(
        result.root_mass_kg.to_bits(),
        0x3eee_a43f_b5c4_ad04_u64,
        "recovered root moved: got {:e}",
        result.root_mass_kg
    );
}

/// The geometry, not the arithmetic: the release velocity is very nearly
/// anti-parallel to the target velocity, so the blended speed passes through a
/// near-zero minimum. Around that minimum the released cloud is on a
/// near-rectilinear orbit whose perigee is inside the Earth, and the J2 secular
/// propagator returns non-finite there.
#[test]
fn nan_window_is_a_subsurface_perigee_band() {
    let mut first_bad = None;
    let mut last_bad = None;
    let mut bad_count = 0_usize;
    let steps = 20_001_usize;
    for step in 0..steps {
        #[expect(
            clippy::cast_precision_loss,
            clippy::as_conversions,
            reason = "sweep index to sample coordinate; both fit f64 exactly"
        )]
        let mass_kg = MASS_MAX_KG * (step as f64) / ((steps - 1) as f64);
        let (miss, why) = miss_distance(mass_kg);
        if miss.is_finite() {
            continue;
        }
        bad_count += 1;
        if first_bad.is_none() {
            first_bad = Some((mass_kg, why));
        }
        last_bad = Some((mass_kg, why));
    }
    println!("non-finite miss samples: {bad_count} of {steps}");
    println!("first non-finite: {first_bad:?}");
    println!("last  non-finite: {last_bad:?}");
    for mass in [0.0, 5.0, 10.0, 20.0, 29.0, 31.0, 40.0, 60.0, 200.0, 1000.0] {
        let (a, e, perigee) = orbit(mass);
        let (miss, why) = miss_distance(mass);
        println!(
            "m={mass:>8.2} kg  a={a:>12.2} km  e={e:>10.7}  rp={perigee:>10.2} km  miss={miss:>14.4}  [{why}]"
        );
    }
    assert!(bad_count > 0, "expected a non-finite band on this row");
}

/// The bisector walks into that band, but not on its first step.
///
/// Renamed 2026-08-09 (R21) from
/// `bracket_endpoints_are_finite_but_the_midpoint_is_not`, which was false and
/// unasserted in the same breath: the first midpoint at 500 kg is finite
/// (miss = 12308 km), and the test only checked the two endpoints, so the
/// claim in its name was never evaluated. The typed status is `MidNonFinite`
/// because the FIFTH midpoint lands on the band -- bisecting [0, 1000] gives
/// 500, 250, 125, 62.5, 31.25, and 31.25 is the sliver
/// `nan_band_width_around_the_bisection_probe` measures. That is consistent
/// with the captured row reporting `iterations = 5`.
#[test]
fn bracket_endpoints_are_finite_and_the_nan_is_the_fifth_midpoint() {
    let (miss_lo, _) = miss_distance(0.0);
    let (miss_hi, _) = miss_distance(MASS_MAX_KG);
    let (miss_mid, why_mid) = miss_distance(0.5 * MASS_MAX_KG);
    println!("miss(0)={miss_lo} miss(1000)={miss_hi} miss(500)={miss_mid} [{why_mid}]");
    assert!(miss_lo.is_finite(), "lower bracket endpoint must be finite");
    assert!(miss_hi.is_finite(), "upper bracket endpoint must be finite");
    assert!(
        miss_mid.is_finite(),
        "the FIRST midpoint is finite; if this ever goes non-finite the status \
         story in this file changes and the name above is wrong again"
    );

    // Replay the descent the captured row takes: on this row every midpoint's
    // miss lands above the target, so the bisector keeps the lower half and the
    // upper bound halves each step (1000, 500, 250, 125, 62.5, 31.25). That is
    // the claim in the name; nothing else in this file evaluates it.
    let mut hi = MASS_MAX_KG;
    let mut first_bad_step = None;
    for step in 1..=8_u32 {
        let mid = 0.5 * hi;
        let (miss, why) = miss_distance(mid);
        println!("step {step}: mid={mid} miss={miss} [{why}]");
        if !miss.is_finite() {
            first_bad_step = Some((step, mid));
            break;
        }
        hi = mid;
    }
    assert_eq!(
        first_bad_step,
        Some((5, 31.25)),
        "the bisection's first non-finite midpoint moved"
    );
}

/// How wide is the non-finite band? A uniform 0.05 kg sweep of [0, 1000] finds
/// exactly one bad sample (31.25 kg), so the band is narrower than the grid.
/// This walks outward from 31.25 kg to measure it, and separates the two
/// sub-steps `propagate_target_for_mass_authority` can fail at.
#[test]
fn nan_band_width_around_the_bisection_probe() {
    let centre = 31.25_f64;
    assert!(!miss_distance(centre).0.is_finite(), "centre must be bad");

    // Widen until both sides are finite, then bisect each edge to f64 adjacency.
    let mut lo = centre;
    let mut hi = centre;
    let mut span = 1e-12_f64;
    while miss_distance(centre - span).0.is_nan() || miss_distance(centre + span).0.is_nan() {
        span *= 2.0;
        assert!(span < 1e3, "band is not bounded below 1000 kg");
    }
    let (mut good_lo, mut good_hi) = (centre - span, centre + span);
    for _ in 0..200 {
        let mid = 0.5 * (good_lo + lo);
        if miss_distance(mid).0.is_nan() {
            lo = mid;
        } else {
            good_lo = mid;
        }
        let mid = 0.5 * (hi + good_hi);
        if miss_distance(mid).0.is_nan() {
            hi = mid;
        } else {
            good_hi = mid;
        }
    }
    let width = hi - lo;
    println!("NaN band: [{lo:.17}, {hi:.17}] width={width:e} kg");
    println!(
        "relative width over the 1000 kg bracket: {:e}",
        width / 1000.0
    );
    let (a, e, perigee) = orbit(centre);
    println!("at centre: a={a:.4} km  e={e:.10}  perigee_radius={perigee:.4} km");
    println!("Earth-radius perigee needs rp > 6378 km");
}

/// Which sub-step returns non-finite: the equinoctial element advance, or the
/// equinoctial-to-Cartesian conversion (the Kepler solve)?
#[test]
fn nan_source_is_the_kepler_conversion_not_the_j2_advance() {
    let velocity = new_velocity(31.25);
    let state = [
        TARGET_POS[0],
        TARGET_POS[1],
        TARGET_POS[2],
        velocity[0],
        velocity[1],
        velocity[2],
    ];
    let mut equ = [0.0_f64; 6];
    eci2equinoc_impl(&state, 6, 0.0, 0.0, &mut equ);
    println!("equinoctial at intercept: {equ:?}");
    println!("  a={} e=sqrt(h^2+k^2)={}", equ[0], equ[1].hypot(equ[2]));

    let mut advanced = [0.0_f64; 6];
    satpy_core::advance_equinoc_j2_impl(&equ, TOF_S, &mut advanced);
    let advance_finite = advanced.iter().all(|value| value.is_finite());
    println!("advanced equinoctial after {TOF_S} s: {advanced:?}");
    println!("  advance finite: {advance_finite}");

    let mut eci = [0.0_f64; 6];
    satpy_core::equinoc2eci_impl(&advanced, 6, 0.0, 0.0, &mut eci);
    let conversion_finite = eci.iter().all(|value| value.is_finite());
    println!("eci after conversion: {eci:?}");
    println!("  conversion finite: {conversion_finite}");

    // Both halves of the name, asserted. Until R21 this test printed the two
    // answers and asserted neither, so it stayed green whichever sub-step was
    // the source -- or if neither was.
    assert!(
        advance_finite,
        "the J2 element advance is NOT the source: it returned non-finite"
    );
    assert!(
        !conversion_finite,
        "the equinoctial-to-Cartesian conversion returned all-finite, so the \
         non-finite miss distance comes from somewhere else and this file's \
         attribution is stale"
    );
}

/// Splits the exposure question into its two factors without running a cell.
///
/// The abort needs BOTH a row whose geometry makes the mass bracket
/// pathological AND a bisection sequence that lands in one of the non-finite
/// slivers. This perturbs the captured row and counts how often the second
/// factor still fires, which says whether the NaN is a knife-edge property of
/// one bit pattern or a robust property of the whole geometry class.
#[test]
fn perturbation_sweep_separates_geometry_from_knife_edge() {
    // Deterministic LCG: the point is reproducibility, not statistical quality.
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next_unit = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[expect(
            clippy::cast_precision_loss,
            clippy::as_conversions,
            reason = "xorshift bits to uniform f64; 53-bit mantissa exact"
        )]
        let unit = (state >> 11) as f64 / ((1_u64 << 53) as f64);
        2.0 * unit - 1.0
    };

    let scales = [0.0_f64, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1];
    let mut nan_total = 0_usize;
    let mut widest_empty_domain = 0_usize;
    let mut unperturbed_converged = 0_usize;
    let mut mid_scale_safe = 0_usize;
    let mut mid_scale_converged = 0_usize;
    let mut widest_safe = 0_usize;
    let mut widest_trials = 0_usize;

    println!("rel_scale  trials  nan  empty_domain  safe_by_default  physics_limited  converged");
    for rel_scale in scales {
        let trials = 2000_usize;
        let mut nan = 0_usize;
        let mut empty_domain = 0_usize;
        let mut safe = 0_usize;
        let mut limited = 0_usize;
        let mut converged = 0_usize;
        for _ in 0..trials {
            let jitter = |value: f64, noise: f64| value * noise.mul_add(rel_scale, 1.0);
            let position = [
                jitter(TARGET_POS[0], next_unit()),
                jitter(TARGET_POS[1], next_unit()),
                jitter(TARGET_POS[2], next_unit()),
            ];
            let velocity = [
                jitter(TARGET_VEL[0], next_unit()),
                jitter(TARGET_VEL[1], next_unit()),
                jitter(TARGET_VEL[2], next_unit()),
            ];
            let release = [
                jitter(DV_VEC[0], next_unit()),
                jitter(DV_VEC[1], next_unit()),
                jitter(DV_VEC[2], next_unit()),
            ];
            let conjunction = [
                jitter(OTHER_CONJ[0], next_unit()),
                jitter(OTHER_CONJ[1], next_unit()),
                jitter(OTHER_CONJ[2], next_unit()),
            ];
            let result = solve_single_event_mf_j2_with_status(
                &MfJ2MassSolverEvent::new(
                    position,
                    velocity,
                    release,
                    P_MASS_KG,
                    conjunction,
                    jitter(TOF_S, next_unit()),
                    MIN_MISS_KM,
                    KAPPA,
                ),
                &config(),
            );
            match result.status {
                MfJ2MassSolveStatusCode::SafeByDefault => safe += 1,
                MfJ2MassSolveStatusCode::PhysicsLimited => limited += 1,
                MfJ2MassSolveStatusCode::Converged => converged += 1,
                // Counted apart from the NaN bucket, because it is not one.
                // `AtmosphericLimited` says the perturbed target's perigee is
                // already at or below the reentry interface, so its valid
                // release-mass domain is empty -- an informative physical
                // verdict, reported deliberately. Folding it into `nan` would
                // make this sweep's own assertion message ("reached a
                // non-finite bisection midpoint") false about the rows it fired
                // on.
                MfJ2MassSolveStatusCode::AtmosphericLimited => empty_domain += 1,
                _ => nan += 1,
            }
        }
        println!(
            "{rel_scale:<10.0e} {trials:<7} {nan:<4} {empty_domain:<13} {safe:<16} \
             {limited:<16} {converged}"
        );

        nan_total = nan_total.saturating_add(nan);
        if rel_scale == 0.0 {
            unperturbed_converged = converged;
        }
        if (rel_scale - 1e-6).abs() < f64::EPSILON {
            mid_scale_safe = safe;
            mid_scale_converged = converged;
        }
        widest_safe = safe;
        widest_empty_domain = empty_domain;
        widest_trials = trials;
    }

    // The conclusion this sweep exists to reach, asserted rather than left in
    // the log. 16,000 trials produced zero assertions until R21, so the sweep
    // could have inverted its own answer and still reported green.
    assert_eq!(
        nan_total, 0,
        "a perturbed row reached a non-finite bisection midpoint; the NaN is \
         then a property of the geometry class, not a knife edge, and the \
         exposure argument in this file's header no longer holds"
    );
    assert_eq!(
        unperturbed_converged, 2000,
        "the unperturbed captured row must converge on every trial"
    );
    // Non-degeneracy: the swept range must actually straddle the behaviour
    // change. If every scale reported one status the zero above would mean
    // nothing.
    assert!(
        mid_scale_safe > 0 && mid_scale_converged > 0,
        "rel_scale 1e-6 must show BOTH outcomes ({mid_scale_safe} safe, \
         {mid_scale_converged} converged); the sweep no longer spans the \
         transition and its zero-NaN result is uninformative"
    );
    // At the widest perturbation every row must land on a BENIGN verdict: no
    // converged root, and no iteration or sampling failure.
    //
    // This used to demand that 90% be safe-by-default specifically. That
    // threshold described a solver which returned any finite unrestricted root
    // without ever comparing it against the valid-domain ceiling. Enforcing the
    // ceiling splits the old safe-by-default bucket in two, because at 10%
    // jitter a large share of the perturbed targets have perigee already at or
    // below the reentry interface and therefore no valid release-mass domain at
    // all. Measured 2026-08-26: 1114 safe-by-default, 886 empty-domain.
    //
    // The claim is stated as an exhaustive partition rather than a retuned
    // percentage, so it fails if any row converges or fails numerically -- which
    // is what the sweep is for -- and does not encode a number fitted to one
    // measurement.
    assert_eq!(
        widest_safe.saturating_add(widest_empty_domain),
        widest_trials,
        "at the widest perturbation every row must be safe-by-default or \
         empty-domain, got {widest_safe} safe and {widest_empty_domain} \
         empty of {widest_trials}"
    );
}
