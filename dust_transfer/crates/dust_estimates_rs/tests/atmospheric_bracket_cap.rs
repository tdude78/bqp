//! The atmospheric bracket cap: gate, ledger and poison proof.
//!
//! `solve_single_event_mf_j2_with_status` bisects the dust release mass over
//! `[0, mass_max = 1000 kg]`. For a light target struck nearly head-on that
//! interval is ~99.96% post-decay — releasing that much dust deorbits the
//! target rather than deflecting it — and the midpoint sequence walks through a
//! scatter of sub-milligram masses where the equinoctial-to-ECI Kepler solve
//! fails at eccentricity ~0.997. Two rows from the `nsga2/flower` pop 3-6
//! reproducers abort there with a NaN root despite having ordinary
//! milligram-scale solutions.
//!
//! The fix restricts the retry bracket to `[0, physical_release_mass_ceiling_kg]`,
//! the release mass at which the deflected perigee reaches the 100 km reentry
//! interface. Every test here is written so that it FAILS against the pre-cap
//! solver, which is kept callable as
//! `solve_single_event_mf_j2_unconstrained_bracket`.

use dust_estimates_rs::mass_solver::{
    physical_release_mass_ceiling_kg, solve_single_event_mf_j2_unconstrained_bracket,
    solve_single_event_mf_j2_with_status, MfJ2MassSolveStatusCode, MfJ2MassSolverEvent,
    SolverConfig, REENTRY_INTERFACE_ALT_KM,
};
use satpy_core::{eci2equinoc_impl, RE};

/// `nd_config::CompiledPartAScienceV1::part_a_v1()` production values.
const KAPPA: f64 = 1.0;
const MASS_MAX_KG: f64 = 1000.0;
const MIN_PRACTICAL_KG: f64 = 5.0e-7;

const fn f(bits: u64) -> f64 {
    f64::from_bits(bits)
}

const fn config() -> SolverConfig {
    SolverConfig {
        xtol: 1.0e-6,
        rtol: 1.0e-5,
        maxiter: 50,
        mass_max: MASS_MAX_KG,
    }
}

/// Captured bit-exact with the `nan-probe` feature from `nsga2/flower` seed 41127203.
fn row234() -> MfJ2MassSolverEvent {
    MfJ2MassSolverEvent::new(
        [
            f(0xc09a_5892_299d_d1ae),
            f(0x40b7_4417_077e_4ab0),
            f(0xc0a9_e832_bca6_8e26),
        ],
        [
            f(0x3fef_a9ad_08fc_0d70),
            f(0xc00b_03ed_c7c6_2e0e),
            f(0xc01a_a461_ef71_ee04),
        ],
        [
            f(0xbffb_da06_d347_fc82),
            f(0x4009_2036_56ea_1d30),
            f(0x401a_4deb_7ed2_7223),
        ],
        f(0x403f_0000_0000_0000),
        [
            f(0x4076_7c7d_61aa_d0dc),
            f(0xc092_24dd_1046_0d46),
            f(0xc0bb_2b41_c5db_eb60),
        ],
        f(0x4106_5397_27fe_d800),
        1.0,
        KAPPA,
    )
}

fn row278() -> MfJ2MassSolverEvent {
    MfJ2MassSolverEvent::new(
        [
            f(0x4084_d2d4_fe7b_d8c6),
            f(0xc0a1_7bf5_8c21_6fe6),
            f(0xc0ba_07c9_207b_d58b),
        ],
        [
            f(0x3ffe_f859_8ac4_e464),
            f(0xc01b_334a_618a_2198),
            f(0x4003_d13d_9687_22b4),
        ],
        [
            f(0xc004_581b_be24_1fbd),
            f(0x401c_983d_41f1_e42c),
            f(0xc001_2236_9788_9f8a),
        ],
        f(0x403f_0000_0000_0000),
        [
            f(0x4076_7c7d_61aa_d0dc),
            f(0xc092_24dd_1046_0d46),
            f(0xc0bb_2b41_c5db_eb60),
        ],
        f(0x40e6_d2ad_eb66_8000),
        1.0,
        KAPPA,
    )
}

/// LEDGER, old -> new -> cause.
///
/// old: both rows returned `root_mass_kg = NaN` with status `MidNonFinite`, and
///      the Stage 3 ingress guard substituted `dust_hard_limit_kg = 1000 kg`
///      while Stage 4 reported them infeasible with
///      `REASON_DETERMINISTIC_MASS_INVALID`.
/// new: both converge, to the bit values pinned below, and report FEASIBLE
///      because each root clears `min_practical_deterministic_mass_kg = 5e-7`.
/// cause: the bisection no longer samples the non-finite scatter around
///      `m ~ p_mass = 31 kg`, because the retry bracket stops at the reentry
///      interface — 0.3168 kg for row234, 0.2891 kg for row278.
#[test]
fn captured_rows_converge_to_the_pinned_roots() {
    for (label, event, expected_bits, expected_ceiling) in [
        (
            "row234",
            row234(),
            0x3ed0_78ce_0557_9863_u64,
            0.316_767_185_f64,
        ),
        (
            "row278",
            row278(),
            0x3eee_a43f_b5c4_ad04_u64,
            0.289_071_671_f64,
        ),
    ] {
        let result = solve_single_event_mf_j2_with_status(&event, &config());
        assert_eq!(
            result.status,
            MfJ2MassSolveStatusCode::Converged,
            "{label}: expected a converged solve, got {:?}",
            result.status
        );
        assert!(
            result.root_mass_kg.is_finite() && result.root_mass_kg > 0.0,
            "{label}: root must be finite and positive, got {}",
            result.root_mass_kg
        );
        assert_eq!(
            result.root_mass_kg.to_bits(),
            expected_bits,
            "{label}: root moved. got {:e} ({:#x}), pinned {:e} ({:#x})",
            result.root_mass_kg,
            result.root_mass_kg.to_bits(),
            f64::from_bits(expected_bits),
            expected_bits
        );
        assert!(
            result.root_mass_kg >= MIN_PRACTICAL_KG,
            "{label}: root {} is below min_practical, so the row would still be \
             reported invalid and the fix would be pointless",
            result.root_mass_kg
        );
        let ceiling =
            physical_release_mass_ceiling_kg(&event, MASS_MAX_KG, REENTRY_INTERFACE_ALT_KM);
        assert!(
            (ceiling - expected_ceiling).abs() < 1e-6,
            "{label}: ceiling {ceiling} != {expected_ceiling}"
        );
    }
}

/// POISON PROOF, direction one: the pre-cap solver must still FAIL on these
/// rows. Without this, the test above would pass on a build where the cap does
/// nothing at all and the rows simply never had a problem.
#[test]
fn the_pre_cap_solver_still_fails_on_both_rows() {
    for (label, event) in [("row234", row234()), ("row278", row278())] {
        let pre_cap = solve_single_event_mf_j2_unconstrained_bracket(&event, &config());
        assert_eq!(
            pre_cap.status,
            MfJ2MassSolveStatusCode::MidNonFinite,
            "{label}: the pre-cap solver was expected to abort with MidNonFinite"
        );
        assert!(
            pre_cap.root_mass_kg.is_nan(),
            "{label}: the pre-cap root was expected to be NaN, got {}",
            pre_cap.root_mass_kg
        );
    }
}

/// POISON PROOF, direction two: the cap is what does the work. The ceiling must
/// actually exclude the mass the pre-cap bisection died on — if the ceiling were
/// above 31.25 kg the retry would walk into the same scatter.
#[test]
fn the_ceiling_excludes_the_mass_the_bisection_died_on() {
    const FATAL_MIDPOINT_KG: f64 = 31.25;
    for (label, event) in [("row234", row234()), ("row278", row278())] {
        let ceiling =
            physical_release_mass_ceiling_kg(&event, MASS_MAX_KG, REENTRY_INTERFACE_ALT_KM);
        assert!(
            ceiling < FATAL_MIDPOINT_KG,
            "{label}: ceiling {ceiling} does not exclude the fatal midpoint"
        );
        assert!(
            ceiling > 0.0 && ceiling < MASS_MAX_KG,
            "{label}: ceiling {ceiling} is not a real restriction"
        );
    }
}

/// The cap must be invisible to any row that already solved. A row whose
/// unconstrained bracket yields a finite root must come back bit-identical,
/// because the shipped entry point calls the unconstrained path first and
/// returns it untouched.
#[test]
fn rows_that_already_solved_are_bit_identical() {
    // A benign row: same geometry, but a heavy target, so the momentum-transfer
    // fraction stays small and no bracket restriction can bite.
    let mut benign = row234();
    benign.p_mass = 5000.0;
    let pre_cap = solve_single_event_mf_j2_unconstrained_bracket(&benign, &config());
    assert!(
        pre_cap.root_mass_kg.is_finite(),
        "the control row must solve before the cap, else it proves nothing"
    );
    let shipped = solve_single_event_mf_j2_with_status(&benign, &config());
    assert_eq!(
        shipped.root_mass_kg.to_bits(),
        pre_cap.root_mass_kg.to_bits(),
        "the cap moved a row that already converged"
    );
    assert_eq!(shipped.status, pre_cap.status);
}

/// A target already below the reentry interface has an empty valid domain, and
/// must be reported as such rather than silently solved.
#[test]
fn an_empty_physical_domain_reports_atmospheric_limited() {
    let mut doomed = row234();
    // Drop the intercept to a radius inside the interface.
    let scale = 0.9_f64;
    doomed = MfJ2MassSolverEvent::new(
        [
            doomed.target_pos_intercept[0] * scale,
            doomed.target_pos_intercept[1] * scale,
            doomed.target_pos_intercept[2] * scale,
        ],
        doomed.target_vel_intercept,
        doomed.dv_vec,
        doomed.p_mass,
        doomed.other_conj_pos,
        doomed.tof_s,
        doomed.min_miss_distance_km,
        KAPPA,
    );
    let ceiling = physical_release_mass_ceiling_kg(&doomed, MASS_MAX_KG, REENTRY_INTERFACE_ALT_KM);
    assert_eq!(
        ceiling.to_bits(),
        0.0_f64.to_bits(),
        "a target already inside the interface must have an empty valid domain, got {ceiling}"
    );
}

/// The ceiling must be `mass_max` exactly — not merely close to it — when no
/// release mass reaches the interface, so an unrestricted row re-enters the
/// identical bisection.
#[test]
fn an_unreachable_interface_leaves_the_bracket_untouched() {
    let mut heavy = row234();
    heavy.p_mass = 1.0e9;
    let ceiling = physical_release_mass_ceiling_kg(&heavy, MASS_MAX_KG, REENTRY_INTERFACE_ALT_KM);
    assert_eq!(
        ceiling.to_bits(),
        MASS_MAX_KG.to_bits(),
        "an unreachable interface must leave mass_max exactly, got {ceiling}"
    );
}

/// The ceiling is solved in closed form (two quadratics in the momentum-transfer
/// fraction). Check it against a brute-force bisection of the perigee condition,
/// so an algebra slip in the derivation cannot pass unnoticed.
#[test]
fn the_closed_form_ceiling_matches_a_brute_force_bisection() {
    fn perigee_alt_km(event: &MfJ2MassSolverEvent, mass_kg: f64) -> f64 {
        let factor = KAPPA * mass_kg / (event.p_mass + mass_kg);
        let v = event.target_vel_intercept;
        let w = event.v_rel;
        let state = [
            event.target_pos_intercept[0],
            event.target_pos_intercept[1],
            event.target_pos_intercept[2],
            v[0] + factor * w[0],
            v[1] + factor * w[1],
            v[2] + factor * w[2],
        ];
        let mut equ = [0.0_f64; 6];
        eci2equinoc_impl(&state, 6, 0.0, 0.0, &mut equ);
        let ecc = equ[1].hypot(equ[2]);
        equ[0] * (1.0 - ecc) - RE
    }

    for (label, event) in [("row234", row234()), ("row278", row278())] {
        let closed =
            physical_release_mass_ceiling_kg(&event, MASS_MAX_KG, REENTRY_INTERFACE_ALT_KM);
        let (mut lo, mut hi) = (0.0_f64, MASS_MAX_KG);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if perigee_alt_km(&event, mid) > REENTRY_INTERFACE_ALT_KM {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        assert!(
            (closed - lo).abs() < 1e-6,
            "{label}: closed form {closed} disagrees with brute force {lo}"
        );
        assert!(
            (perigee_alt_km(&event, closed) - REENTRY_INTERFACE_ALT_KM).abs() < 1e-3,
            "{label}: the perigee at the ceiling is not the interface"
        );
    }
}
