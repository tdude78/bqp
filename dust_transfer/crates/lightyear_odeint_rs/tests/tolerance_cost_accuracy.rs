//! Ignored diagnostic: what does the integrator tolerance BUY, in metres, on
//! production-shaped strict-HF arcs, and what does it COST in RHS evaluations.
//!
//! No production authority changes. Run:
//! `cargo test --release -p lightyear_odeint_rs --test tolerance_cost_accuracy -- --ignored --nocapture --test-threads=1`
//!
//! The corpus is four LEO geometries crossed with three arc lengths (2 h,
//! 12 h, and the ~1.3 day intercept->conjunction leg the mass solver walks),
//! alternating the dust (A/m 1.948) and canister (A/m 0.01) ballistic ratios.
//! The reference is the same wrapper at a tolerance two decades below the
//! tightest sample, which is the same instrument as the arc under test and
//! therefore shares its systematic error; read the numbers as "distance from
//! the converged limit of THIS wrapper", not as absolute truth.
//!
//! # A SINGLE `TOL_ROW` MAXIMUM IS A DRAW, NOT A MEASUREMENT
//!
//! The per-arc error is chaotic in the step sequence, so the worst-arc column
//! is a lottery ticket and the winning arc changes from rung to rung. Measured
//! 2026-08-09 by holding the tree fixed and sweeping eps across a +-2x
//! neighbourhood of production (7e-9 .. 2e-8, nine rungs, identical physics):
//!
//! ```text
//! 579a1b6, Vern9   worst-arc min 0.313   median 0.969   max 3.279 m   [ladder unrecorded]
//! c4ea964, Vern9   worst-arc min 0.196   median 0.832   max 2.231 m   [ladder unrecorded]
//! c4ea964, Vern7   worst-arc min 0.680   median 1.831   max 2.847 m   [ladder unrecorded]
//! ```
//!
//! Ten-fold spread with the tree held constant. One arc (`alt800` long) reads
//! 0.076 .. 0.855 m over that neighbourhood at 579a1b6 and 0.037 .. 1.134 m at
//! `c4ea964`, while its MEDIAN barely moves (0.313 -> 0.329 m). Every row above
//! re-runs byte-identical at `c3bc7ba`, which only re-routed the anchor-failure
//! authorisation and moved no propagation bits.
//!
//! So: never quote one rung of this table as "the corpus worst", and never size
//! a tolerance from it. A 3.2x margin was once claimed against a rung that was
//! the minimum of its own neighbourhood -- see `strict_hf_pin::ACCURACY_TOL_M`.
//! Compare medians over the neighbourhood, or compare nothing.
//!
//! ## THE LADDER IS PART OF THE MEASUREMENT, AND THE ROWS ABOVE DO NOT CARRY IT
//!
//! "Nine rungs spanning 7e-9 .. 2e-8" does not name a ladder, and the three rows
//! above cannot be reproduced from what they record. Re-measured 2026-08-10 at
//! `fc222c0`, sweeping the linear ladder
//! `[7, 8, 9, 10, 12, 14, 16, 18, 20] e-9`:
//!
//! ```text
//! fc222c0, Vern7 (RESOLVED)  worst-arc min 0.680  median 2.335  max 2.847 m
//! ```
//!
//! Its min and max land on the recorded `c4ea964, Vern7` row to three decimals
//! and its MEDIAN does not (1.831 -> 2.335), so the two sweeps did not walk the
//! same rungs. Two nine-rung ladders over the same interval, at the SAME commit
//! and the SAME stepper (`c4ea964`, Vern9), read:
//!
//! ```text
//! linear    [7,8,9,10,12,14,16,18,20]e-9   min 0.196  median 0.680  max 1.821 m
//! geometric 7e-9 * (20/7)^(k/8)            min 0.129  median 0.649  max 3.787 m
//! ```
//!
//! The spread this file warns about therefore reaches the neighbourhood
//! STATISTICS too, not just the individual rung: max moves 2.1x on the choice of
//! ladder alone. Quote the rung list with any row added here, or the row is not
//! a measurement either.
//!
//! The Vern7 sweep also re-runs byte-identical -- all nine rungs, all seven
//! columns -- between `c4ea964` and `fc222c0`, which is the positive control for
//! the "bit-identical" claims carried by the eclipse, JB2008 and sqrt-fold
//! commits merged across that interval.
//!
//! The same chaos decides whether this harness RUNS: the reentry arc's failure
//! flips between `Ground` and `Eclipse(Envelope)` with the step sequence, which
//! is why the drop is witness-authorised rather than read off the anchor. It
//! has flipped twice already -- red at 285b641, green at 579a1b6, red again at
//! bce8eaf.

use anyhow::Context;
use std::sync::Arc;

use lightyear_odeint_rs::integrator::{
    integrate_final_checked, FinalPropagationFailure, ScalarGravityAssets,
    ScalarPropagationContext, ScalarPropagationRequest,
};
use lightyear_odeint_rs::probe;
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags, StepperMethod};
use lightyear_odeint_rs::EclipseError;
use satpy_core::{eci2equinoc_impl_f64, equinoc2eci_impl, kep2eci_impl, SEC_PER_DAY};

const JD0: f64 = 2_460_310.5;
const DIR_R6_D15: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

/// Production tolerance ladder.
///
/// `REFERENCE_EPS` is the anchor every error in this file is measured against.
/// It is CALLED converged; nothing here tests that, and the distinction matters
/// because this harness has been used to price physics changes at the sub-metre
/// scale.
///
/// `REFERENCE_CHECK_EPS` exists to give that claim a number. It is LOOSER, not
/// tighter, and deliberately so: the integrator floors tolerance at
/// `eps_eff = eps.max(1e-12)` (six call sites in `integrator.rs`), so a
/// "tighter" control below the floor is the identical computation and can only
/// report a drift of exactly zero. That trap bit once in this repo — the
/// retired `zz_straddle_n11` harness carried a `REF_CHECK = 1e-13` that the
/// floor made vacuous — so the control runs the other way and asks how far the
/// answer moves between 1e-10 and 1e-11.
///
/// READ THE DRIFT BEFORE READING ANY ERROR. A per-arc error smaller than that
/// arc's reference drift is noise: it is measured against an anchor that is
/// still moving by more than the difference being reported.
const EPS_LADDER: [f64; 5] = [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9];
const REFERENCE_EPS: f64 = 1.0e-11;
const REFERENCE_CHECK_EPS: f64 = 1.0e-10;

/// The rung that decides whether an arc is physically infeasible.
///
/// One corpus arc has no converged reference: the 400 km dust particle
/// reenters at roughly 73,000 s into a 111,874 s request. That is a fact about
/// the physics, but the integrator's REPORT of it is tolerance-dependent. At
/// `eps >= 3e-11` (Vern9) the ground event is refined first and the arc comes
/// back `Ground`, which `is_physical_infeasible` recognises. Tighter than that,
/// a step carries the state below the 6,000 km eclipse-envelope floor before
/// the ground event is reported; the envelope guard fires, and
/// `final_propagation_failure` reads `terminal_eclipse_error` before
/// `terminal_event_fired`, so the arc comes back `Eclipse(Envelope)` instead.
/// Both reports describe the same reentry. `REFERENCE_EPS` sits on the wrong
/// side of that flip, which is why the anchor cannot be asked the question.
///
/// So the anchor never classifies an arc by itself — not even when its own
/// report is physically named, because the flip above shows the name at
/// `REFERENCE_EPS` is unreliable in both directions. EVERY anchor failure is
/// accepted as "no reference exists" only when this rung independently
/// returns a physical-infeasibility verdict on the same arc. An anchor that
/// fails on an arc which survives production eps is an instrument failure and
/// still aborts the run -- that control is the point of this file.
const INFEASIBILITY_WITNESS_EPS: f64 = 1.0e-8;

struct ArcCase {
    init_equ: [f64; 6],
    tf_s: f64,
    base_at_tf: [f64; 6],
    config: ForceConfig,
    label: String,
}

/// The compiled stepper, resolved rather than restated.
///
/// This file hardcoded `StepperMethod::Vern9` through the Vern9 -> Vern7 swap
/// at 8ee9fdf and kept reporting, green and silent, a tolerance/cost curve for
/// a stepper the campaign no longer flies -- which matters here more than in a
/// timing harness, because the Vern7 rows of the neighbourhood table above sit
/// three-quarters of a metre ABOVE the Vern9 rows at the same eps. Same defect
/// `prop_timing::authority_stepper` and `v3_accuracy_floor::authority_stepper`
/// were written to close; same fail-closed shape, phrased as an error because
/// every caller here is already fallible.
fn authority_stepper() -> anyhow::Result<StepperMethod> {
    match nd_config::CompiledPartAScienceV1::part_a_v1()
        .hybrid()
        .integrator_method
    {
        "vern7" => Ok(StepperMethod::Vern7),
        "vern9" => Ok(StepperMethod::Vern9),
        other => {
            anyhow::bail!("compiled science selects a stepper this file does not build: {other}")
        }
    }
}

fn production_force_config(am_ratio: f64) -> anyhow::Result<ForceConfig> {
    Ok(ForceConfig {
        sph_order: 5,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        subtract_first_order: true,
        atm_model: 4,
        am_ratio,
        cd: 2.2,
        cr: 1.3,
        target_propagation_mode: 0,
        dt_max: 300.0,
        eps: 1.0e-8,
        integrator_method: authority_stepper()?,
        ..ForceConfig::default()
    })
}

fn corpus() -> anyhow::Result<Vec<ArcCase>> {
    let keplerian = [
        [6_778.137, 0.001, 28.5, 0.0, 10.0, 0.0],
        [6_928.137, 0.010, 53.0, 40.0, 75.0, 90.0],
        [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0],
        [7_378.137, 0.060, 63.4, 275.0, 310.0, 270.0],
    ];
    // 2 h dust free flight, 12 h, and the ~1.3 day intercept->conjunction leg
    // the mass solver actually walks.
    let durations = [7_200.0, 43_200.0, 111_874.0];
    let mut cases = Vec::new();
    for (state_idx, kep) in keplerian.iter().enumerate() {
        let mut init_eci = [0.0; 6];
        kep2eci_impl(kep, true, 0.0, 0.0, true, &mut init_eci);
        let mut init_equ = [0.0; 6];
        eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init_equ);
        for &tf_s in &durations {
            let am_ratio = if state_idx % 2 == 0 { 1.948 } else { 0.01 };
            let config = production_force_config(am_ratio)?
                .with_ephemeris_for_arc(JD0, JD0 + tf_s / SEC_PER_DAY)
                .context("production ephemeris/JB2008 assets must cover diagnostic arc")?;
            let mut base_at_tf = [0.0; 6];
            equinoc2eci_impl(&init_equ, 6, tf_s, 0.0, &mut base_at_tf);
            cases.push(ArcCase {
                init_equ,
                tf_s,
                base_at_tf,
                config,
                label: format!("alt{:.0}_am{am_ratio}_tof{tf_s:.0}", kep[0] - 6_378.137),
            });
        }
    }
    Ok(cases)
}

#[test]
fn only_physical_final_failures_become_diagnostic_skips() {
    assert!(matches!(
        final_delta_or_physical_skip(Err(FinalPropagationFailure::Ground)),
        Ok(None)
    ));
    assert!(
        final_delta_or_physical_skip(Err(FinalPropagationFailure::Census(
            probe::PropagationCensusError::Allocation,
        )))
        .is_err()
    );
    assert!(
        final_delta_or_physical_skip(Err(FinalPropagationFailure::Eclipse(
            EclipseError::Geometry,
        )))
        .is_err()
    );
}

#[test]
fn only_a_physical_witness_excuses_an_anchor_failure() {
    // The verdict is a function of the witness alone: the anchor's own failure
    // name is tolerance-dependent and is never consulted. The case this file
    // exists to keep failing: the anchor died and nothing corroborates it.
    assert_eq!(anchor_failure_verdict(None), AnchorVerdict::Abort);
    assert_eq!(
        anchor_failure_verdict(Some(FinalPropagationFailure::Eclipse(
            EclipseError::Envelope,
        ))),
        AnchorVerdict::Abort
    );
    assert_eq!(
        anchor_failure_verdict(Some(FinalPropagationFailure::Census(
            probe::PropagationCensusError::Allocation,
        ))),
        AnchorVerdict::Abort
    );
    // The reentry the corpus deliberately contains.
    assert_eq!(
        anchor_failure_verdict(Some(FinalPropagationFailure::Ground)),
        AnchorVerdict::Drop
    );
    assert_eq!(
        anchor_failure_verdict(Some(FinalPropagationFailure::LeftEarth)),
        AnchorVerdict::Drop
    );
}

#[test]
fn a_physically_named_anchor_failure_does_not_excuse_itself() {
    // Regression pin for the c3bc7ba routing: the anchor's own failure name
    // is tolerance-dependent and must never authorise the drop. A physically
    // named anchor failure with no corroborating witness is an instrument
    // failure and aborts.
    assert!(
        anchor_or_witnessed_skip("test", Err(FinalPropagationFailure::Ground), || None).is_err()
    );
    assert!(
        anchor_or_witnessed_skip("test", Err(FinalPropagationFailure::Ground), || Some(
            FinalPropagationFailure::Eclipse(EclipseError::Envelope)
        ))
        .is_err()
    );
    // Only a physical witness verdict excuses it.
    assert!(matches!(
        anchor_or_witnessed_skip("test", Err(FinalPropagationFailure::Ground), || Some(
            FinalPropagationFailure::Ground
        )),
        Ok(None)
    ));
}

#[test]
fn a_successful_anchor_never_consults_the_witness() {
    let result = anchor_or_witnessed_skip("test", Ok([0.0; 6]), || {
        panic!("witness must not run when the anchor succeeded")
    });
    assert!(matches!(result, Ok(Some(_))));
}

fn final_delta_or_physical_skip(
    result: Result<[f64; 6], FinalPropagationFailure>,
) -> anyhow::Result<Option<[f64; 6]>> {
    match result {
        Ok(delta) => Ok(Some(delta)),
        Err(failure) if failure.is_physical_infeasible() => Ok(None),
        Err(failure) => {
            Err(anyhow::Error::new(failure)
                .context("tolerance diagnostic final propagation failed"))
        }
    }
}

fn propagate_final(
    case: &ArcCase,
    eps: f64,
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
) -> Result<[f64; 6], FinalPropagationFailure> {
    let mut config = case.config;
    config.eps = eps;
    let gravity = ScalarGravityAssets::new(Arc::clone(packed));
    let context = ScalarPropagationContext::new(JD0, Arc::new(config), gravity);
    let delta = integrate_final_checked(
        ScalarPropagationRequest::new(&context, case.init_equ, &[case.tf_s], 0.0, case.tf_s)
            .with_events(true),
    )?;
    let mut state = case.base_at_tf;
    for (state_component, delta_component) in state.iter_mut().zip(delta) {
        *state_component += delta_component;
    }
    Ok(state)
}

fn propagate(
    case: &ArcCase,
    eps: f64,
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
) -> anyhow::Result<Option<[f64; 6]>> {
    let Some(state) = final_delta_or_physical_skip(propagate_final(case, eps, packed))? else {
        return Ok(None);
    };
    anyhow::ensure!(
        state.iter().all(|value| value.is_finite()),
        "tolerance diagnostic produced a non-finite terminal state"
    );
    Ok(Some(state))
}

/// What an anchor failure means, once the witness rung has spoken.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AnchorVerdict {
    /// The arc is infeasible on its own terms; it carries no accuracy signal.
    Drop,
    /// Nothing corroborates the anchor failure, so it is an instrument failure.
    Abort,
}

/// An anchor failure — whatever its own report says — is only ever excused by
/// a PHYSICAL verdict from [`INFEASIBILITY_WITNESS_EPS`]. A witness that
/// succeeds, or that fails numerically, excuses nothing. The anchor's own
/// failure name is never consulted: it is tolerance-dependent (see the flip
/// documented at the witness constant), so a physically-named anchor failure
/// is no more trustworthy than a numerically-named one.
const fn anchor_failure_verdict(witness: Option<FinalPropagationFailure>) -> AnchorVerdict {
    match witness {
        Some(failure) if failure.is_physical_infeasible() => AnchorVerdict::Drop,
        _ => AnchorVerdict::Abort,
    }
}

/// The routing that [`reference_or_infeasible_skip`] applies, separated from
/// propagation so the routing itself is unit-testable: EVERY anchor failure —
/// regardless of its own failure name — consults the witness, and only a
/// physical witness verdict excuses the drop. The witness closure runs only
/// when the anchor failed.
fn anchor_or_witnessed_skip(
    label: &str,
    anchor: Result<[f64; 6], FinalPropagationFailure>,
    witness: impl FnOnce() -> Option<FinalPropagationFailure>,
) -> anyhow::Result<Option<[f64; 6]>> {
    let failure = match anchor {
        Ok(state) => {
            anyhow::ensure!(
                state.iter().all(|value| value.is_finite()),
                "tolerance diagnostic reference anchor is non-finite"
            );
            return Ok(Some(state));
        }
        Err(failure) => failure,
    };
    let witness = witness();
    match anchor_failure_verdict(witness) {
        AnchorVerdict::Abort => {
            Err(anyhow::Error::new(failure).context("tolerance diagnostic reference anchor failed"))
        }
        AnchorVerdict::Drop => {
            println!(
                "TOL_SKIP_INFEASIBLE {label} witness_eps={INFEASIBILITY_WITNESS_EPS:e} \
                 witness={} anchor_eps={REFERENCE_EPS:e} anchor_report={failure}",
                witness.map_or_else(|| "none".to_string(), |failure| failure.to_string()),
            );
            Ok(None)
        }
    }
}

/// The reference anchor for one arc, or `None` when the arc reenters and
/// therefore has no converged reference to anchor against.
fn reference_or_infeasible_skip(
    case: &ArcCase,
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
) -> anyhow::Result<Option<[f64; 6]>> {
    anchor_or_witnessed_skip(
        &case.label,
        propagate_final(case, REFERENCE_EPS, packed),
        || propagate_final(case, INFEASIBILITY_WITNESS_EPS, packed).err(),
    )
}

fn pos_err_m(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let [a_x, a_y, a_z, ..] = *a;
    let [b_x, b_y, b_z, ..] = *b;
    let dx = a_x - b_x;
    let dy = a_y - b_y;
    let dz = a_z - b_z;
    #[expect(
        clippy::suboptimal_flops,
        reason = "the accuracy metric retains its established non-FMA reduction order"
    )]
    let error_km = (dx * dx + dy * dy + dz * dz).sqrt();
    error_km * 1000.0
}

#[test]
#[ignore = "tolerance cost/accuracy diagnostic; run separately"]
fn tolerance_cost_and_accuracy_over_production_arcs() -> anyhow::Result<()> {
    // Every tolerance row performs direct propagation so its probe census
    // measures that row's actual integration work.
    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("gravity coefficients must load")?;
    let packed =
        lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")?;

    let cases = corpus()?;
    let corpus_arcs = cases.len();
    println!("TOL_ARCS {corpus_arcs}");

    // Arcs whose converged reference does not exist (a high-ballistic dust
    // particle released at 400 km reenters inside 1.3 days) carry no accuracy
    // signal and are dropped rather than scored against a failure. Which
    // failure the anchor reports for that reentry is tolerance-dependent, so
    // the drop is authorised by a witness rung -- see
    // `INFEASIBILITY_WITNESS_EPS`.
    let mut references: Vec<[f64; 6]> = Vec::new();
    let mut kept: Vec<ArcCase> = Vec::new();
    for case in cases {
        match reference_or_infeasible_skip(&case, &packed)? {
            Some(reference) => {
                // The anchor's own convergence, per arc, rather than assumed
                // once for the corpus. Printed even when small: a reader
                // comparing two rows of this table needs to know the floor
                // below which a difference is the anchor moving, not the
                // physics.
                match propagate(&case, REFERENCE_CHECK_EPS, &packed)? {
                    Some(check) => println!(
                        "TOL_REFDRIFT {} drift_m={:.9}",
                        case.label,
                        pos_err_m(&reference, &check)
                    ),
                    None => println!("TOL_REFDRIFT {} UNCHECKED non_finite_at_check", case.label),
                }
                references.push(reference);
                kept.push(case);
            }
            None => println!("TOL_SKIP {}", case.label),
        }
    }
    let cases = kept;
    println!("TOL_SCORED {}", cases.len());

    // Floor, from the corpus size rather than from what survives today. The
    // skip above is authorised per-arc, so an asset or epoch change that made
    // EVERY anchor infeasible would empty `cases` and every loop below it
    // would then iterate nothing: the ladder rows would print with an empty
    // error vector, the p50/max lookups would be the only thing to fail, and
    // a corpus that lost half its arcs would not be visible at all. One arc is
    // documented to drop; three quarters is the loosest bound that still
    // catches a corpus quietly collapsing.
    anyhow::ensure!(
        cases
            .len()
            .checked_mul(4)
            .context("scored-arc floor overflows")?
            >= corpus_arcs
                .checked_mul(3)
                .context("corpus-arc floor overflows")?,
        "only {} of {corpus_arcs} corpus arcs kept a converged reference; \
         every tolerance row below would be computed over that reduced set",
        cases.len()
    );

    println!("TOL_HEADER eps,total_rhs_evals,steps,segments,saturated,pos_err_m_p50,pos_err_m_max");
    let mut evals_at_1e8 = 0u64;

    for &eps in &EPS_LADDER {
        probe::reset()?;
        let mut errors = Vec::with_capacity(cases.len());
        for (case, reference) in cases.iter().zip(references.iter()) {
            match propagate(case, eps, &packed)? {
                Some(state) => errors.push((case.label.clone(), pos_err_m(&state, reference))),
                None => errors.push((case.label.clone(), f64::NAN)),
            }
        }
        let census = probe::snapshot();
        let evals: u64 = census.iter().map(|entry| entry.rhs_evals).sum();
        let steps: u64 = census.iter().map(|entry| entry.steps).sum();
        let segments: u64 = census.iter().map(|entry| entry.segments).sum();
        let saturated: u64 = census.iter().map(|entry| entry.saturated).sum();
        #[expect(
            clippy::float_cmp,
            reason = "the exact production ladder rung controls a discrete baseline row"
        )]
        if eps == 1.0e-8 {
            evals_at_1e8 = evals;
        }
        let mut sorted: Vec<f64> = errors.iter().map(|entry| entry.1).collect();
        if sorted.iter().any(|error| !error.is_finite()) {
            anyhow::bail!("tolerance row at eps={eps:e} contains a non-finite propagation error");
        }
        sorted.sort_by(f64::total_cmp);
        let p50 = *sorted
            .get(sorted.len() / 2)
            .context("tolerance row has no median error")?;
        let max = *sorted
            .last()
            .context("tolerance row has no maximum error")?;
        println!("TOL_ROW {eps:e},{evals},{steps},{segments},{saturated},{p50:.4},{max:.4}");
        for (label, err) in &errors {
            println!("TOL_DETAIL {eps:e},{label},{err:.5}");
        }
    }
    println!("TOL_BASE_EVALS_1e-8 {evals_at_1e8}");

    // Endpoint pins at a converged tolerance. Rebuilding with a different
    // PERTURB_DEVIATION_THRESHOLD_KM and diffing these isolates the Encke
    // rectification error from the tolerance error, which the in-build
    // reference cannot do (it moves with the threshold).
    println!(
        "TOL_PIN_THRESHOLD_KM {}",
        lightyear_odeint_rs::types::PERTURB_DEVIATION_THRESHOLD_KM
    );
    for case in &cases {
        let state = propagate(case, 1.0e-9, &packed)?.context("pin propagation must succeed")?;
        println!(
            "TOL_PIN {},{:.17e},{:.17e},{:.17e}",
            case.label, state[0], state[1], state[2]
        );
    }
    Ok(())
}
