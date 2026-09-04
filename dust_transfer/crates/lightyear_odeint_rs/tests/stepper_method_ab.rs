//! Ignored diagnostic: what does the compiled stepper choice COST and BUY on
//! production-shaped strict-HF arcs?
//!
//! No production authority changes here. Run:
//! `cargo test --release -p lightyear_odeint_rs --test stepper_method_ab -- --ignored --nocapture --test-threads=1`
//!
//! # Why this file exists
//!
//! Integrator method was the one strict-HF knob nobody had ever priced. The
//! first measurement of it (R25) lived in a throwaway worktree and was lost,
//! so the conclusion it reached had no reproducible evidence behind it. This
//! file is that evidence, re-measured at the FLOWN atmosphere model.
//!
//! It does not reproduce the original result, and the differences are the
//! reason it is worth keeping rather than deleting after the decision.
//!
//! **Era `cd330e5`, `atm_model 7` (flown), re-measured 2026-08-11.** The middle
//! column is R25's lost harness at `atm_model 5`; the right column is what this
//! file prints today, on the epoch-FIXED corpus.
//!
//! | quantity | R25 (lost harness, `atm_model 5`) | here (`atm_model 7`, flown) |
//! |---|---|---|
//! | RHS evaluations | -20.0% | **-15.29%** (6,305.1 vs 7,443.3 mean) |
//! | wall | -16.6% | **-10.70%** (per-eval cost 1.054) |
//! | accuracy | Vern7 strictly better | **not rankable on this corpus** (see below) |
//! | worst draw | Vern7 better | RMS 0.1299 vs 0.0300 m, 4/12 draws |
//! | convergence below the sealed eps | Vern7 floored | neither arm floors |
//!
//! The saving is real and the direction holds; the magnitude is roughly three
//! quarters of what was claimed, and the risk that was thought to be the whole
//! catch does not exist.
//!
//! **The accuracy row is a standing law, not a reading.** An earlier era of
//! this header quoted "a wash: RMS 0.0317 vs 0.0323 m, 6 of 12 draws each",
//! which was measured before the epoch axis of this corpus was found to be
//! vacuous and every draw to be flying one epoch. On the fixed corpus the
//! numbers above are what print — but `examples/r43_corpus_floor.rs` then
//! showed that Vern9's RMS on this corpus has a floor spread of 156-171% of its
//! own value, and that the Vern7-minus-Vern9 gap STRADDLES ZERO under a
//! physics-neutral ULP perturbation, on two seeds at 48 draws, with a
//! method-independent anchor. So neither "a wash" nor "Vern7 is 4x worse" is
//! supported. **This corpus prices COST and cannot rank ACCURACY.** The cost
//! side is untouched: evaluation counts are exact integers and reproduce
//! bit-for-bit.
//!
//! # What makes this comparison legal
//!
//! Two things, both of which cost a round to learn:
//!
//! 1. **A single draw cannot rank methods.** Endpoint error at one tolerance is
//!    one sample from a sign-random sum of step-local errors; single draws
//!    scatter over two orders of magnitude and the error column is NOT monotone
//!    in `eps`. Every number here is over `DRAWS` decorrelated arcs -- epoch,
//!    mean anomaly AND orbit are all moved -- scored against a PER-DRAW
//!    reference. See CORPUS SHAPE below for what happens when they are not.
//!
//! 2. **The reference must not be an arm's own floor.** The tempting design is
//!    to score each arm against *its own* method at a converged tolerance, so
//!    the wrapper's systematic error cancels. That is invalid the moment an arm
//!    has a convergence floor: scoring Vern7@1e-8 against Vern7@1e-12 measures
//!    distance to an anchor that is itself still moving, and it flatters the
//!    arm with the floor. This file scores both arms against ONE anchor
//!    (`ANCHOR_METHOD` at `REFERENCE_EPS`) and prints the cross-method anchor
//!    separation next to it, so the reader can see how much of any difference
//!    is the anchor's own method-dependence. Both error columns are reported;
//!    they disagree, and the common-anchor column is the one that means
//!    "distance from truth".
//!
//! # Loop order is not a free choice
//!
//! Interleaving the two ARMS defends against drift. It does nothing about a
//! working set the harness itself inflates, and that confound is not
//! symmetric -- it lands on the arm with the larger per-step workspace, which
//! is exactly what a plausible "bigger tableau thrashes cache" story predicts.
//! The same 12 arcs and the same two binaries gave a 28.3% saving with the
//! repetition loop outermost and 16.6% with the draw loop outermost; 11 points
//! of "speedup" was loop nesting.
//!
//! So: the draw loop is OUTERMOST, every per-draw setup happens before the
//! repetition loop, and the arms are INNERMOST. A measured block touches one
//! draw's configuration, ephemeris and caches and nothing else.
//!
//! The wall ratio is then cross-checked against the RHS-evaluation ratio, which
//! is load-independent. A wall ratio that beats its own counter ratio in the
//! favourable direction is the tell that the harness is flattering an arm.
//!
//! # CORPUS SHAPE DECIDES THE ANSWER -- the reason this file is a keeper
//!
//! The first version of this corpus moved only epoch and mean anomaly, holding
//! the orbit fixed at the pinned arc's 800 km. On that corpus Vern7 measured
//! **1.7x worse** in RMS (0.1197 vs 0.0694 m) and won only 3 of 12 draws --
//! a clear, reproducible, and entirely spurious verdict. Widening the corpus to
//! move altitude and eccentricity as well turned it into a wash (0.0323 vs
//! 0.0317 m, 6 of 12).
//!
//! Twelve phases of one orbit is one sample, not twelve: the step-size
//! controller responds to drag magnitude and how sharply it varies along the
//! arc, so a corpus that fixes altitude and eccentricity holds fixed the only
//! two things that distinguish the steppers. It produced a stable, confident,
//! wrong number -- stable because it was correlated, not because it was right.
//!
//! Sanity check against production, not against taste: **draw 0 is the exact
//! arc the strict-HF V3 pin flies**, and must reproduce that pin's counts --
//! `V3_PINNED_RHS_EVALS` in `tests/strict_hf_pin.rs`, **6,742 evaluations over
//! 666 steps** at the compiled Vern7. If it does not, this harness has drifted
//! off production and no other row here means anything. (The Vern9-era figure
//! this line used to carry, 7,875 evaluations over 474 steps, prices the arc
//! the campaign stopped flying at `8ee9fdf`; draw 0 reads 7,924/478 at Vern9
//! today.)
//!
//! # What the tolerance ladder says
//!
//! Both arms keep converging below the sealed tolerance in COST terms, and
//! neither floors -- see `stepper_ladder_shows_both_arms_keep_converging`. Read
//! the evaluation column: Vern7 goes 5,154.6 -> 6,305.1 -> 8,111.2 -> 11,009.8
//! across 1e-7..1e-10 against Vern9's 7,131.6 -> 7,443.3 -> 9,141.3 ->
//! 12,255.8, so Vern7 is cheaper at every rung, by 27.7% at the loose end and
//! 10.2% at the tight one.
//!
//! **Do not read the ladder's error column as a convergence claim.** It is
//! non-monotone on both arms (Vern9 reads 0.0300 m at 1e-8 and 0.0335 m at
//! 1e-9), which is the floor described above rather than an arm failing to
//! converge. An earlier era of this header claimed Vern7 was "BOTH cheaper and
//! more accurate than Vern9 at the tight end" on the strength of that column;
//! today the same column says the opposite at 1e-10 (0.0343 vs 0.0179 m), and
//! neither reading is worth anything. What flattens near 1e-9 is the ANCHOR.

// Every lossy cast in this file turns a census counter or a corpus size into a
// mean for PRINTING. None of them reaches an assertion, a digest or a returned
// value, and the magnitudes involved -- tens of thousands of evaluations, a
// dozen draws -- are exact in binary64 many orders below the mantissa limit.
// The suppression is file-scoped rather than per-site because there is one
// reason and it is the same at all of them.
#![expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "diagnostic means printed from small integer counters; nothing asserts or digests them"
)]

use anyhow::Context;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lightyear_odeint_rs::integrator::{
    integrate_final_checked, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest,
};
use lightyear_odeint_rs::probe;
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags, StepperMethod};
use satpy_core::{eci2equinoc_impl_f64, equinoc2eci_impl, kep2eci_impl, SEC_PER_DAY};

const DIR_R6_D15: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

/// The arc the strict-HF pins fly, and the arc R25 priced.
const JD0: f64 = 2_460_310.5;
const TOF_S: f64 = 43_200.0;

/// Twelve is not a round number chosen for looks: it is the point at which the
/// per-draw error scatter stopped moving the RMS ranking between reruns.
const DRAWS: usize = 12;

/// Min-of-many-short. Long measured blocks drift with host load by more than
/// the effect; the minimum over many short ones is the estimator that survives
/// a shared machine.
const REPS: usize = 25;

/// The converged anchor.
///
/// `integrator.rs` floors the effective tolerance at `eps.max(1e-12)`, so a
/// "tighter" reference below this is the identical computation and reports a
/// drift of exactly zero -- a vacuous control that has bitten this repo before.
/// 1e-12 is the floor itself; `REFERENCE_CHECK_EPS` is deliberately LOOSER so
/// the convergence claim has a number attached.
const REFERENCE_EPS: f64 = 1.0e-12;
const REFERENCE_CHECK_EPS: f64 = 1.0e-10;

/// The two arms. Vern9 is the incumbent this file exists to compare against;
/// it is kept in the tree for exactly that reason.
const ARMS: [StepperMethod; 2] = [StepperMethod::Vern9, StepperMethod::Vern7];

/// Every explicit one-step method the tree can actually fly, in ascending
/// order of the tableau's order of accuracy.
///
/// This exists because the Vern9-vs-Vern7 comparison above is a two-point
/// sample of a family, and a two-point sample cannot say which direction along
/// the order axis is the winning one. The search that produced Vern7 walked
/// DOWN from Vern9 and stopped; the arms below Vern7 were named as "measured"
/// in `docs/ODE_SOLVER_EVALUATION.md` on the strength of this file, which until
/// now priced neither of them. `every_explicit_arm_is_priced_at_the_sealed_eps`
/// is that missing measurement.
///
/// `Esdirk43` and `Auto` are absent because `validate_scalar_stepper_authority`
/// refuses them under the flown force shape (JB2008 drivers plus binary-eclipse
/// SRP), so there is no production-shaped arc on which they could be priced.
const EXPLICIT_ARMS: [StepperMethod; 6] = [
    StepperMethod::Dopri5Compat,
    StepperMethod::Tsit5,
    StepperMethod::Vern7,
    StepperMethod::Dop853,
    StepperMethod::Rkv98,
    StepperMethod::Vern9,
];

/// The single anchor both arms are scored against.
///
/// Vern9 and not "each arm's own method" -- see the header. Vern9 is the arm
/// that demonstrably keeps converging below the sealed tolerance, so it is the
/// only one of the two whose tight run can stand in for truth.
const ANCHOR_METHOD: StepperMethod = StepperMethod::Vern9;

/// Tolerance rungs for the convergence ladder. Stops at 1e-10 because
/// `REFERENCE_EPS` is 1e-12 and a rung within two decades of its anchor is
/// measuring the anchor.
const LADDER: [f64; 4] = [1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10];

/// The compiled Part A science authority, read rather than restated.
const fn part_a_hybrid() -> &'static nd_config::PartAHybridControls {
    nd_config::CompiledPartAScienceV1::part_a_v1().hybrid()
}

const fn arm_label(method: StepperMethod) -> &'static str {
    match method {
        StepperMethod::Vern7 => "vern7",
        StepperMethod::Vern9 => "vern9",
        StepperMethod::Dopri5Compat => "dopri5",
        StepperMethod::Tsit5 => "tsit5",
        StepperMethod::Dop853 => "dop853",
        StepperMethod::Rkv98 => "rkv98",
        StepperMethod::Esdirk43 => "esdirk43",
        StepperMethod::Auto => "auto",
    }
}

/// Order of accuracy of each arm's tableau, for the cost-at-order column.
const fn arm_order(method: StepperMethod) -> u32 {
    match method {
        StepperMethod::Dopri5Compat | StepperMethod::Tsit5 => 5,
        StepperMethod::Vern7 => 7,
        StepperMethod::Dop853 => 8,
        StepperMethod::Rkv98 | StepperMethod::Vern9 => 9,
        StepperMethod::Esdirk43 => 4,
        StepperMethod::Auto => 0,
    }
}

/// One decorrelated production-shaped arc.
struct Draw {
    init_equ: [f64; 6],
    base_at_tf: [f64; 6],
    /// The arc's OWN epoch, and it must be carried here.
    ///
    /// `with_ephemeris_for_arc` only bounds the catalogue WINDOW. Sun and Moon
    /// resolve dynamically from the JD the propagation context is built on, so
    /// a context built on the shared `JD0` flies the shared epoch however the
    /// ephemeris call was parameterised -- and this harness did exactly that
    /// until 2026-08-10, which made the corpus's epoch axis vacuous while its
    /// doc claimed twelve mutually incommensurate epochs. The orbit axis was
    /// always real; the epoch axis was not.
    epoch: f64,
    config: ForceConfig,
    label: String,
}

/// Production strict-HF force shape, every field read from the sealed
/// authority except the stepper, which is the dimension under test.
///
/// `atm_model` in particular must come from compiled science: this harness
/// exists to price a production decision, and the JB2008 profile changes the
/// RHS cost per evaluation. R25 measured at model 5; production flies 7.
fn production_force_config(method: StepperMethod) -> ForceConfig {
    let controls = part_a_hybrid();
    ForceConfig {
        sph_order: controls.gravity_order,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        subtract_first_order: true,
        atm_model: controls.atmosphere_model,
        am_ratio: controls.dust_am_ratio,
        cd: controls.dust_cd,
        cr: controls.dust_cr,
        target_propagation_mode: 0,
        dt_max: controls.dt_max_s,
        eps: controls.tolerance,
        integrator_method: method,
        ..ForceConfig::default()
    }
}

/// Twelve production-shaped arcs that share the mission's regime but not its
/// geometry, phase or epoch.
///
/// **Draw 0 is deliberately the exact arc the strict-HF V3 pin flies.** That
/// costs one of the twelve samples and buys a check nothing else here provides:
/// if this harness is measuring what production measures, draw 0 must report
/// the pin's own RHS-evaluation and step counts. When it stops doing so, the
/// harness has drifted off production and every other row is suspect.
///
/// Draws 1-11 move epoch, mean anomaly AND the orbit itself. An earlier version
/// of this corpus moved only epoch and mean anomaly, which made it twelve
/// phases of a single orbit rather than twelve arcs: altitude sets the drag
/// magnitude and eccentricity sets how sharply it varies along the arc, and
/// those are the two things the step-size controller is actually responding to.
/// A corpus that holds them fixed cannot distinguish a stepper that handles
/// varying stiffness well from one that got a single easy orbit.
///
/// Every increment is mutually incommensurate with the ~1.6 h orbital period
/// and the 1 d solar/geomagnetic driver cadence, so no two draws share a phase
/// in any coordinate.
fn corpus(method: StepperMethod) -> anyhow::Result<Vec<Draw>> {
    let mut draws = Vec::with_capacity(DRAWS);
    for index in 0..DRAWS {
        let step = index as f64;
        let epoch = JD0 + step * 3.37;
        let kep = if index == 0 {
            [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0]
        } else {
            [
                // 650-1000 km, and eccentricity kept small enough that perigee
                // stays near 580 km at worst. The band is bounded BELOW by
                // physics, not by taste: production dust carries A/m 1.948, and
                // below roughly 600 km that reenters inside the 12 h arc. A
                // corpus that reaches lower does not sample harder arcs, it
                // samples arcs with no endpoint at all.
                7_028.137 + (step * 61.7) % 350.0,
                0.001 + (step * 0.0017) % 0.009,
                (28.5 + step * 13.9) % 180.0,
                (125.0 + step * 47.3) % 360.0,
                (210.0 + step * 71.1) % 360.0,
                (180.0 + step * 29.3) % 360.0,
            ]
        };
        let mean_anomaly = kep.get(5).copied().unwrap_or_default();
        let mut init_eci = [0.0; 6];
        kep2eci_impl(&kep, true, 0.0, 0.0, true, &mut init_eci);
        let mut init_equ = [0.0; 6];
        eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init_equ);
        let config = production_force_config(method)
            .with_ephemeris_for_arc(epoch, epoch + TOF_S / SEC_PER_DAY)
            .context("production ephemeris and JB2008 assets must cover every draw")?;
        let mut base_at_tf = [0.0; 6];
        equinoc2eci_impl(&init_equ, 6, TOF_S, 0.0, &mut base_at_tf);
        draws.push(Draw {
            init_equ,
            base_at_tf,
            epoch,
            config,
            label: format!(
                "draw{index:02}_alt{:.0}_e{:.3}_ma{mean_anomaly:.0}",
                kep.first().copied().unwrap_or_default() - 6_378.137,
                kep.get(1).copied().unwrap_or_default()
            ),
        });
    }
    Ok(draws)
}

struct Run {
    state: [f64; 6],
    rhs_evals: u64,
    steps: u64,
}

/// Probe counters are process-global and libtest runs a binary's tests on
/// parallel threads, so two tests calling `probe::reset()` clobber each
/// other's census. `--test-threads=1` is in the run line for this reason; the
/// lock makes the requirement enforced rather than documented.
static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Strict propagation: any failure, including reentry, is an error.
///
/// Used everywhere AFTER `screen` has removed the draws with no endpoint, so a
/// `Ground` here means the corpus changed underneath the screen rather than
/// that this arc was always infeasible.
fn propagate(
    draw: &Draw,
    method: StepperMethod,
    eps: f64,
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
) -> anyhow::Result<Run> {
    propagate_opt(draw, method, eps, packed)?.with_context(|| {
        format!(
            "{} reentered at {}; screen should have dropped it",
            draw.label,
            arm_label(method)
        )
    })
}

/// Propagation that reports a physically infeasible arc as `None` rather than
/// as a failure. Reentry is an answer about the orbit, not a defect.
fn propagate_opt(
    draw: &Draw,
    method: StepperMethod,
    eps: f64,
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
) -> anyhow::Result<Option<Run>> {
    let mut config = draw.config;
    config.integrator_method = method;
    config.eps = eps;
    let guard = PROBE_LOCK
        .lock()
        .map_err(|_ignored| anyhow::anyhow!("stepper A/B probe lock was poisoned"))?;
    // Repeat identical (draw, method, eps) propagations and read the probe
    // census per run. Direct execution is required for every repetition.
    probe::reset().context("probe census must reset")?;
    let gravity = ScalarGravityAssets::new(Arc::clone(packed));
    let context = ScalarPropagationContext::new(draw.epoch, Arc::new(config), gravity);
    let delta = match integrate_final_checked(
        ScalarPropagationRequest::new(&context, draw.init_equ, &[TOF_S], 0.0, TOF_S)
            .with_events(true),
    ) {
        Ok(delta) => delta,
        Err(failure) if failure.is_physical_infeasible() => {
            drop(guard);
            return Ok(None);
        }
        Err(failure) => {
            drop(guard);
            return Err(anyhow::anyhow!("{failure:?}").context(format!(
                "{} must propagate at {}",
                draw.label,
                arm_label(method)
            )));
        }
    };
    let census = probe::snapshot();
    drop(guard);

    let rhs_evals = census
        .iter()
        .try_fold(0u64, |acc, entry| acc.checked_add(entry.rhs_evals))
        .context("RHS-evaluation census overflow")?;
    let steps = census
        .iter()
        .try_fold(0u64, |acc, entry| acc.checked_add(entry.steps))
        .context("step census overflow")?;

    let mut state = draw.base_at_tf;
    for (component, increment) in state.iter_mut().zip(delta) {
        *component += increment;
    }
    anyhow::ensure!(
        state.iter().all(|value| value.is_finite()),
        "{} produced a non-finite terminal state at {}",
        draw.label,
        arm_label(method)
    );
    Ok(Some(Run {
        state,
        rhs_evals,
        steps,
    }))
}

/// Drop the draws that have no endpoint under EITHER arm.
///
/// Screening on both arms rather than on the anchor alone is deliberate: a draw
/// that reenters under one stepper and not the other would otherwise be scored
/// on an unequal corpus, which is a way to win an A/B by dropping the arm's own
/// hard cases.
fn screen(
    draws: Vec<Draw>,
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
    arms: &[StepperMethod],
) -> anyhow::Result<Vec<Draw>> {
    let mut kept = Vec::with_capacity(draws.len());
    for draw in draws {
        let mut feasible = true;
        for method in arms.iter().copied() {
            if propagate_opt(&draw, method, REFERENCE_EPS, packed)?.is_none() {
                feasible = false;
            }
        }
        if feasible {
            kept.push(draw);
        } else {
            println!("AB_SKIP {} no_endpoint", draw.label);
        }
    }
    println!("AB_SCORED {}", kept.len());
    Ok(kept)
}

fn pos_err_m(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let [a_x, a_y, a_z, ..] = *a;
    let [b_x, b_y, b_z, ..] = *b;
    let dx = a_x - b_x;
    let dy = a_y - b_y;
    let dz = a_z - b_z;
    (dx * dx + dy * dy + dz * dz).sqrt() * 1000.0
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = values.iter().map(|value| value * value).sum();
    (sum / values.len() as f64).sqrt()
}

fn max_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn load_packed() -> anyhow::Result<Arc<satpy_core::PackedGravityCoeffs>> {
    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("gravity coefficients must load")?;
    lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")
}

/// The headline table: cost and accuracy of both arms at the SEALED tolerance,
/// each scored against its own converged limit, over twelve decorrelated draws.
#[test]
#[ignore = "stepper A/B diagnostic; run separately"]
fn stepper_arms_priced_at_the_sealed_tolerance() -> anyhow::Result<()> {
    let packed = load_packed()?;
    let sealed_eps = part_a_hybrid().tolerance;
    println!(
        "AB_SHAPE draws={DRAWS} reps={REPS} sealed_eps={sealed_eps:e} atm_model={} compiled_method={}",
        part_a_hybrid().atmosphere_model,
        part_a_hybrid().integrator_method
    );

    // Every non-arm dimension is resolved BEFORE the repetition loop. One
    // corpus serves both arms: `propagate` sets the stepper on its own copy of
    // the config, so geometry, epochs and ephemeris windows are shared by
    // construction rather than by two builds that have to be kept in step.
    let draws = screen(corpus(ANCHOR_METHOD)?, &packed, &ARMS)?;
    let scored = draws.len();
    anyhow::ensure!(scored > 0, "every draw was screened out");

    let mut errors: Vec<Vec<f64>> = vec![Vec::with_capacity(scored); ARMS.len()];
    let mut own_errors: Vec<Vec<f64>> = vec![Vec::with_capacity(scored); ARMS.len()];
    let mut evals: Vec<u64> = vec![0; ARMS.len()];
    let mut steps: Vec<u64> = vec![0; ARMS.len()];
    let mut walls: Vec<Duration> = vec![Duration::ZERO; ARMS.len()];

    for index in 0..scored {
        // --- per-draw setup, OUTSIDE the repetition loop ---
        let draw = draws.get(index).context("corpus must cover every draw")?;

        // The common anchor, plus its own convergence check, plus how far the
        // OTHER method's tight run sits from it. That last number is what says
        // whether "distance from the converged limit" is a method-independent
        // quantity on this arc or not; without it the common-anchor column
        // could be measuring the anchor's tableau rather than the arm's error.
        let anchor = propagate(draw, ANCHOR_METHOD, REFERENCE_EPS, &packed)?;
        let anchor_check = propagate(draw, ANCHOR_METHOD, REFERENCE_CHECK_EPS, &packed)?;
        println!(
            "AB_REFDRIFT {} anchor drift_m={:.6}",
            draw.label,
            pos_err_m(&anchor.state, &anchor_check.state)
        );

        let mut own_references = Vec::with_capacity(ARMS.len());
        for method in ARMS {
            let own = propagate(draw, method, REFERENCE_EPS, &packed)?;
            println!(
                "AB_ANCHOR_SEP {} {} sep_m={:.6}",
                draw.label,
                arm_label(method),
                pos_err_m(&own.state, &anchor.state)
            );
            own_references.push(own);
        }

        // --- measured block: arms INNERMOST, one draw's data live ---
        let mut best: Vec<Duration> = vec![Duration::MAX; ARMS.len()];
        let mut last: Vec<Option<Run>> = (0..ARMS.len()).map(|_ignored| None).collect();
        for _rep in 0..REPS {
            for (arm, method) in ARMS.iter().copied().enumerate() {
                let started = Instant::now();
                let run = propagate(draw, method, sealed_eps, &packed)?;
                let elapsed = started.elapsed();
                if let Some(slot) = best.get_mut(arm) {
                    *slot = (*slot).min(elapsed);
                }
                if let Some(slot) = last.get_mut(arm) {
                    *slot = Some(run);
                }
            }
        }

        for (arm, method) in ARMS.iter().copied().enumerate() {
            let run = last
                .get(arm)
                .and_then(Option::as_ref)
                .context("every arm must have produced a run")?;
            let own = own_references
                .get(arm)
                .context("every arm must have its own reference")?;
            let error = pos_err_m(&run.state, &anchor.state);
            let own_error = pos_err_m(&run.state, &own.state);
            let wall = best.get(arm).copied().unwrap_or(Duration::MAX);
            println!(
                "AB_ROW {} {} evals={} steps={} err_m={error:.4} own_err_m={own_error:.4} wall_us={}",
                draw.label,
                arm_label(method),
                run.rhs_evals,
                run.steps,
                wall.as_micros()
            );
            if let Some(slot) = errors.get_mut(arm) {
                slot.push(error);
            }
            if let Some(slot) = own_errors.get_mut(arm) {
                slot.push(own_error);
            }
            if let Some(slot) = evals.get_mut(arm) {
                *slot = slot
                    .checked_add(run.rhs_evals)
                    .context("eval total overflow")?;
            }
            if let Some(slot) = steps.get_mut(arm) {
                *slot = slot.checked_add(run.steps).context("step total overflow")?;
            }
            if let Some(slot) = walls.get_mut(arm) {
                *slot = slot.saturating_add(wall);
            }
        }
    }

    println!(
        "AB_HEADER method,mean_evals,mean_steps,rms_err_m,max_err_m,own_rms_err_m,own_max_err_m,wall_ms"
    );
    for (arm, method) in ARMS.iter().copied().enumerate() {
        let arm_errors = errors.get(arm).context("arm errors missing")?;
        let arm_own = own_errors.get(arm).context("arm own errors missing")?;
        let arm_evals = evals.get(arm).copied().unwrap_or_default();
        let arm_steps = steps.get(arm).copied().unwrap_or_default();
        let arm_wall = walls.get(arm).copied().unwrap_or_default();
        println!(
            "AB_SUMMARY {},{:.1},{:.1},{:.4},{:.4},{:.4},{:.4},{:.3}",
            arm_label(method),
            arm_evals as f64 / scored as f64,
            arm_steps as f64 / scored as f64,
            rms(arm_errors),
            max_of(arm_errors),
            rms(arm_own),
            max_of(arm_own),
            arm_wall.as_secs_f64() * 1000.0
        );
    }

    // The cross-check the loop-order confound demands: a wall ratio that beats
    // its own load-independent counter ratio is a harness artefact, not a
    // saving. The residual between the two is the per-evaluation cost
    // difference between the tableaus and should be small and STABLE.
    let (incumbent, candidate) = (0usize, 1usize);
    let eval_ratio = evals.get(candidate).copied().unwrap_or_default() as f64
        / evals.get(incumbent).copied().unwrap_or(1).max(1) as f64;
    let wall_ratio = walls
        .get(candidate)
        .copied()
        .unwrap_or_default()
        .as_secs_f64()
        / walls
            .get(incumbent)
            .copied()
            .unwrap_or(Duration::from_nanos(1))
            .as_secs_f64();
    println!(
        "AB_RATIO eval={eval_ratio:.4} wall={wall_ratio:.4} per_eval_cost={:.4}",
        wall_ratio / eval_ratio
    );

    let favoured = errors
        .get(candidate)
        .context("candidate errors missing")?
        .iter()
        .zip(errors.get(incumbent).context("incumbent errors missing")?)
        .filter(|(candidate_error, incumbent_error)| candidate_error < incumbent_error)
        .count();
    println!("AB_ACCURACY_WINS candidate={favoured}/{scored}");
    Ok(())
}

/// Every explicit one-step arm the tree can fly, priced at the sealed
/// tolerance against one anchor. The Pareto check on Vern7 is the verdict.
///
/// # What this closes
///
/// `docs/ODE_SOLVER_EVALUATION.md` closes the explicit one-step family on a
/// predicate it calls "stage count at order": any explicit method is priced
/// against Vern7's ten evaluations per accepted step. The predicate is sound
/// for methods AT or ABOVE order 7, where more stages buy accuracy this problem
/// does not need. It says nothing about methods BELOW order 7, which take more
/// but cheaper steps, and whose evaluation total is therefore an empirical
/// question rather than a tableau property. The document listed Dopri5, Tsit5,
/// Dop853 and Rkv98 as "measured ... priced by `stepper_method_ab`"; this file
/// priced only Vern9 and Vern7 until this test was added.
///
/// # The verdict is on COST alone, deliberately
///
/// The standing law is that this corpus prices COST and cannot rank ACCURACY:
/// the Vern7-minus-Vern9 RMS gap straddles zero under a physics-neutral ULP
/// perturbation, so no assertion here may rest on the error column. The check
/// below reads RHS evaluations only. They are exact integers and reproduce
/// bit-for-bit across hosts and reruns.
///
/// That makes the gate stronger than a two-axis dominance test, not weaker: it
/// goes red the moment any arm undercuts Vern7 on cost at all, at which point
/// the accuracy question becomes live and has to be answered by an instrument
/// built for it rather than by the RMS column printed here. The error column is
/// reported because a reader needs to see what the cost is buying; nothing
/// asserts on it.
///
/// The margin is not comfortable, which is why the gate is worth keeping.
#[test]
#[ignore = "stepper A/B diagnostic; run separately"]
fn every_explicit_arm_is_priced_at_the_sealed_eps() -> anyhow::Result<()> {
    let packed = load_packed()?;
    let sealed_eps = part_a_hybrid().tolerance;
    println!(
        "ARM_SHAPE draws={DRAWS} sealed_eps={sealed_eps:e} atm_model={} compiled_method={}",
        part_a_hybrid().atmosphere_model,
        part_a_hybrid().integrator_method
    );

    // Screened on the WHOLE arm set, not on the incumbent pair: a draw that
    // reenters under one arm and not another would otherwise let an arm win by
    // sitting out its own hard cases.
    let draws = screen(corpus(ANCHOR_METHOD)?, &packed, &EXPLICIT_ARMS)?;
    let scored = draws.len();
    anyhow::ensure!(scored > 0, "every draw was screened out");

    let mut errors: Vec<Vec<f64>> = vec![Vec::with_capacity(scored); EXPLICIT_ARMS.len()];
    let mut evals: Vec<u64> = vec![0; EXPLICIT_ARMS.len()];
    let mut steps: Vec<u64> = vec![0; EXPLICIT_ARMS.len()];

    for draw in &draws {
        let anchor = propagate(draw, ANCHOR_METHOD, REFERENCE_EPS, &packed)?;
        for (arm, method) in EXPLICIT_ARMS.iter().copied().enumerate() {
            let run = propagate(draw, method, sealed_eps, &packed)?;
            let error = pos_err_m(&run.state, &anchor.state);
            println!(
                "ARM_ROW {} {} order={} evals={} steps={} err_m={error:.4}",
                draw.label,
                arm_label(method),
                arm_order(method),
                run.rhs_evals,
                run.steps
            );
            if let Some(slot) = errors.get_mut(arm) {
                slot.push(error);
            }
            if let Some(slot) = evals.get_mut(arm) {
                *slot = slot
                    .checked_add(run.rhs_evals)
                    .context("eval total overflow")?;
            }
            if let Some(slot) = steps.get_mut(arm) {
                *slot = slot.checked_add(run.steps).context("step total overflow")?;
            }
        }
    }

    let flown = EXPLICIT_ARMS
        .iter()
        .position(|method| *method == StepperMethod::Vern7)
        .context("the flown arm must be in the priced set")?;
    let flown_evals = evals.get(flown).copied().unwrap_or_default();
    let flown_rms = rms(errors.get(flown).context("flown errors missing")?);

    println!("ARM_HEADER method,order,total_evals,total_steps,evals_per_step,rms_err_m,max_err_m,evals_vs_vern7");
    for (arm, method) in EXPLICIT_ARMS.iter().copied().enumerate() {
        let arm_errors = errors.get(arm).context("arm errors missing")?;
        let arm_evals = evals.get(arm).copied().unwrap_or_default();
        let arm_steps = steps.get(arm).copied().unwrap_or_default().max(1);
        println!(
            "ARM_SUMMARY {},{},{arm_evals},{},{:.2},{:.4},{:.4},{:+.2}%",
            arm_label(method),
            arm_order(method),
            steps.get(arm).copied().unwrap_or_default(),
            arm_evals as f64 / arm_steps as f64,
            rms(arm_errors),
            max_of(arm_errors),
            100.0 * (arm_evals as f64 / flown_evals.max(1) as f64 - 1.0)
        );
    }

    // The verdict, on the cost axis alone.
    for (arm, method) in EXPLICIT_ARMS.iter().copied().enumerate() {
        if arm == flown {
            continue;
        }
        let arm_evals = evals.get(arm).copied().unwrap_or_default();
        let arm_rms = rms(errors.get(arm).context("arm errors missing")?);
        anyhow::ensure!(
            arm_evals >= flown_evals,
            "{} undercuts the flown Vern7 arm on cost: {arm_evals} RHS \
             evaluations against {flown_evals}. The explicit one-step family is \
             REOPENED. Do not settle it with the error column printed above \
             ({arm_rms:.4} m against {flown_rms:.4} m) -- this corpus cannot \
             rank accuracy, and a cheaper arm is exactly the case where the \
             ranking would decide the answer.",
            arm_label(method)
        );
    }
    Ok(())
}

/// Does the stepper ranking survive a change of atmosphere model?
///
/// It does not, and that is the single most load-bearing fact in this file.
/// R25 priced these arms at `atm_model 5` and reported Vern7 strictly better on
/// cost AND accuracy. Production flew model 6 then and flies model 7 now.
/// Re-running the identical corpus across the models shows the accuracy ranking
/// INVERTING between 5 and 6: the cheaper JB2008 quadrature changes the drag
/// error the controller is chasing, and the two tableaus do not absorb that
/// change equally.
///
/// The general lesson is worth more than the specific number. An integrator
/// A/B is not a property of the integrator; it is a property of the right-hand
/// side it is integrating. Measuring it against anything other than the flown
/// force model prices a configuration nobody runs.
///
/// The sweep list must therefore CONTAIN the flown model, and the assertion
/// below is what keeps it that way. Without it this test degrades in silence:
/// it has no verdict to fail, so when R31 moved the seal 6 -> 7 the list
/// `[4, 5, 6]` would have kept passing while the `FLOWN` column never printed
/// and production went uncovered.
#[test]
#[ignore = "stepper A/B diagnostic; run separately"]
fn stepper_ranking_depends_on_the_atmosphere_model() -> anyhow::Result<()> {
    const SWEPT: [i32; 4] = [4, 5, 6, 7];

    let packed = load_packed()?;
    let sealed_eps = part_a_hybrid().tolerance;
    let flown = part_a_hybrid().atmosphere_model;
    println!("AB_ATM_HEADER atm_model,method,mean_evals,rms_err_m,max_err_m,flown");
    anyhow::ensure!(
        SWEPT.contains(&flown),
        "compiled science flies atm_model {flown}, which this sweep does not cover; \
         add it to SWEPT before reading any row here as a statement about production"
    );
    for atm_model in SWEPT {
        let mut draws = corpus(ANCHOR_METHOD)?;
        for draw in &mut draws {
            draw.config.atm_model = atm_model;
        }
        // Screened per model: which arcs survive is itself a function of the
        // atmosphere, so a corpus screened once at model 6 would silently
        // compare unequal sets across the sweep.
        let draws = screen(draws, &packed, &ARMS)?;
        let scored = draws.len();
        let mut references = Vec::with_capacity(scored);
        for draw in &draws {
            references.push(propagate(draw, ANCHOR_METHOD, REFERENCE_EPS, &packed)?);
        }
        for method in ARMS {
            let mut row_errors = Vec::with_capacity(scored);
            let mut row_evals = 0u64;
            for (draw, reference) in draws.iter().zip(references.iter()) {
                let run = propagate(draw, method, sealed_eps, &packed)?;
                row_errors.push(pos_err_m(&run.state, &reference.state));
                row_evals = row_evals
                    .checked_add(run.rhs_evals)
                    .context("atmosphere sweep eval overflow")?;
            }
            println!(
                "AB_ATM {atm_model},{},{:.1},{:.4},{:.4},{}",
                arm_label(method),
                row_evals as f64 / scored as f64,
                rms(&row_errors),
                max_of(&row_errors),
                if atm_model == flown { "FLOWN" } else { "-" }
            );
        }
    }
    Ok(())
}

/// The tolerance ladder, measured rather than assumed.
///
/// This exists to answer one question: does either arm stop paying for a
/// tighter tolerance? Neither does. Both RMS columns fall monotonically from
/// 1e-7 to 1e-10, and at the tight end Vern7 is both cheaper and more accurate
/// than Vern9 -- the reverse of the sealed-tolerance rung.
///
/// Read the anchor's own drift before reading any row: what flattens between
/// 1e-9 and 1e-10 is `REFERENCE_EPS` running out of headroom, not an arm
/// hitting a floor. An error smaller than the anchor's drift is noise.
#[test]
#[ignore = "stepper A/B diagnostic; run separately"]
fn stepper_ladder_shows_both_arms_keep_converging() -> anyhow::Result<()> {
    let packed = load_packed()?;
    println!("AB_LADDER_HEADER method,eps,mean_evals,rms_err_m,max_err_m");
    let draws = screen(corpus(ANCHOR_METHOD)?, &packed, &ARMS)?;
    let scored = draws.len();
    let mut references = Vec::with_capacity(scored);
    for draw in &draws {
        references.push(propagate(draw, ANCHOR_METHOD, REFERENCE_EPS, &packed)?);
    }
    for &method in &ARMS {
        for &eps in &LADDER {
            let mut row_errors = Vec::with_capacity(scored);
            let mut row_evals = 0u64;
            for (draw, reference) in draws.iter().zip(references.iter()) {
                let run = propagate(draw, method, eps, &packed)?;
                row_errors.push(pos_err_m(&run.state, &reference.state));
                row_evals = row_evals
                    .checked_add(run.rhs_evals)
                    .context("ladder eval overflow")?;
            }
            println!(
                "AB_LADDER {},{eps:e},{:.1},{:.4},{:.4}",
                arm_label(method),
                row_evals as f64 / scored as f64,
                rms(&row_errors),
                max_of(&row_errors)
            );
        }
    }
    Ok(())
}
