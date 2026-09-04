//! The sampled route rejects arcs the checked route propagates, and calls it
//! `Eclipse(Bracket)`.
//!
//! # What diverges
//!
//! `integrate_final_checked` (the production strict-HF route, which asks for the
//! endpoint alone) returns a state. `integrate_adaptive` (the SAMPLED route),
//! given the SAME arc, the same force config and the same tolerance, returns
//! `EclipseError::Bracket` as soon as the caller asks for enough output times.
//! Nothing eclipse-related has gone wrong on those arcs.
//!
//! # Mechanism, located 2026-08-10
//!
//! `eclipse_coordinator::complete_non_eclipse_segment`, in its
//! `EventType::PerturbDeviation` branch -- the Encke rectification rebase, which
//! fires whenever the osculating deviation crosses
//! `PERTURB_DEVIATION_THRESHOLD_KM` (10 km), i.e. many times on any long arc.
//!
//! The branch selects every requested output time at or before the event with
//! `partition_point(|t| t <= event_t)`, then looks each one up in the times the
//! solver actually returned. The solver stopped its trial AT the event root, so
//! its last emitted sample is `event_t` itself; the requested times that fall
//! strictly INSIDE the step that triggered the event were never emitted. The
//! lookup misses and the branch returns `EclipseError::Bracket`.
//!
//! Instrumented on `alt400_am1.948`, 7,200 s, `eps = 1e-8`, 32 output times:
//!
//! ```text
//! curr_t      299.42341939509913   (segment start, prepended by solver_sample_times)
//! event_t    1612.6484901048807    (PerturbDeviation root)
//! trial.times [299.423, 450, 675, 900, 1125, 1350, 1612.648]   (7 entries)
//! requested   450 .. 1575 in steps of 225                      (6 entries, all <= event_t)
//! missing    1575                  -> EclipseError::Bracket
//! ```
//!
//! `1575` sits between the last emitted sample (`1350`) and the event root, so
//! it is inside the terminating step. It is a legitimately requested output
//! time; recovering it needs dense output WITHIN the event step, which is a
//! solver-level capability the trial does not carry. That is why this file pins
//! the behaviour instead of fixing it.
//!
//! The name is the second defect and the smaller one: `Bracket` is this module's
//! catch-all for every internal bookkeeping miss (28 construction sites), so a
//! sampling-reconstruction hole is reported to callers -- and onward to
//! `QualificationArmRejectCause::EclipseBracket` -- as an eclipse-geometry
//! failure. Same class as the failure-name confusion `tolerance_cost_accuracy`
//! documents at `INFEASIBILITY_WITNESS_EPS`.
//!
//! # What does NOT trigger it, measured
//!
//! | knob | effect |
//! |---|---|
//! | `enable_events = false` | NEVER diverges, at any density, on any arc tried |
//! | `SampledOutputMode::ForceEvaluationTimes` | no protection -- `integrate_adaptive` deliberately overrides `force_eval` under SRP, so the caller's request for solver steps at every sample time is discarded |
//! | single-point `t_eval` | agrees with the checked route, which is why the strict-HF corpus never saw this |
//!
//! So the exposed shape is exactly: SRP on, events ON, more than one output
//! time. The one in-tree caller of that shape is
//! `two_phase_transfer_rs::evaluate::propagate_high_fidelity_target_multi_tof_checked`
//! (`.with_events(true)`, up to `MAX_TOF_SAMPLES = 256` offsets), whose own
//! callers are all inside its crate's `#[cfg(test)] mod tests` at this tip.
//! `batch.rs`'s sampled branch takes the `integrate_adaptive` path with events
//! left off and is therefore not exposed. Nothing in production is failing
//! today; anything that wires that function up starts losing valid arcs.
//!
//! # What this file asserts
//!
//! That the divergence is STILL THERE, not that it is correct. A fix makes this
//! test red on purpose: delete the pin and record the fix. The events-off arm is
//! the control -- without it a pin that only demanded "the sampled route fails"
//! would stay green if the sampled route broke outright.

use anyhow::Context;
use std::sync::Arc;

use lightyear_odeint_rs::integrator::{
    final_propagation_failure, integrate_adaptive, integrate_final_checked,
    FinalPropagationFailure, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest,
};
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags, StepperMethod};
use lightyear_odeint_rs::EclipseError;
use satpy_core::{eci2equinoc_impl_f64, kep2eci_impl, SEC_PER_DAY};

const JD0: f64 = 2_460_310.5;
const DIR_R6_D15: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

/// The `alt400_am1.948` geometry of the `tolerance_cost_accuracy` corpus, at the
/// 2 h length. Chosen because the checked route propagates it cleanly, so the
/// divergence cannot be confused with a genuinely infeasible arc.
const KEPLERIAN: [f64; 6] = [6_778.137, 0.001, 28.5, 0.0, 10.0, 0.0];
const AM_RATIO: f64 = 1.948;
const TF_S: f64 = 7_200.0;
const EPS: f64 = 1.0e-8;

/// Output-time counts swept per arm.
///
/// A range rather than one count: whether a requested time lands inside the
/// rebase step is chaotic in the step sequence, so pinning one density would
/// pin a lottery draw -- the failure mode this repo keeps re-learning. The
/// assertion is over the sweep.
const DENSITIES: [usize; 6] = [8, 16, 32, 64, 128, 256];

/// The compiled stepper, resolved rather than restated -- see
/// `tolerance_cost_accuracy::authority_stepper` for why a literal here rots
/// silently.
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

fn arc_config() -> anyhow::Result<ForceConfig> {
    ForceConfig {
        sph_order: 5,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        subtract_first_order: true,
        atm_model: 4,
        am_ratio: AM_RATIO,
        cd: 2.2,
        cr: 1.3,
        target_propagation_mode: 0,
        dt_max: 300.0,
        eps: EPS,
        integrator_method: authority_stepper()?,
        ..ForceConfig::default()
    }
    .with_ephemeris_for_arc(JD0, JD0 + TF_S / SEC_PER_DAY)
    .context("ephemeris and JB2008 assets must cover the pinned arc")
}

fn initial_equinoctial() -> [f64; 6] {
    let mut init_eci = [0.0; 6];
    kep2eci_impl(&KEPLERIAN, true, 0.0, 0.0, true, &mut init_eci);
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init_equ);
    init_equ
}

/// Requested output times: `count` equally spaced offsets ending at `TF_S`.
fn sample_times(count: usize) -> Vec<f64> {
    (1..=count)
        .map(|index| {
            let numerator = u32::try_from(index).unwrap_or(u32::MAX);
            let denominator = u32::try_from(count).unwrap_or(u32::MAX);
            TF_S * f64::from(numerator) / f64::from(denominator)
        })
        .collect()
}

/// The sampled route's terminal failure for one output-time count, or `None`
/// when it completed.
fn sampled_failure(
    context: &ScalarPropagationContext,
    init_equ: [f64; 6],
    t_eval: &[f64],
    enable_events: bool,
) -> anyhow::Result<Option<FinalPropagationFailure>> {
    let result = integrate_adaptive(
        ScalarPropagationRequest::new(context, init_equ, t_eval, 0.0, TF_S)
            .with_events(enable_events),
    )
    .context("sampled route returned a census failure")?;
    Ok(final_propagation_failure(&result))
}

#[test]
fn sampled_route_still_rejects_an_arc_the_checked_route_propagates() -> anyhow::Result<()> {
    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("gravity coefficients must load")?;
    let packed =
        lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")?;
    let init_equ = initial_equinoctial();
    let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
    let context = ScalarPropagationContext::new(JD0, Arc::new(arc_config()?), gravity);

    // The production route's verdict on this arc. Everything below is only
    // meaningful because this succeeds: the arc is healthy.
    let endpoint = [TF_S];
    let checked = integrate_final_checked(
        ScalarPropagationRequest::new(&context, init_equ, &endpoint, 0.0, TF_S).with_events(true),
    );
    let checked = checked.map_err(|failure| {
        anyhow::anyhow!(
            "the pinned arc must propagate on the checked route; it reported {failure}. \
             The arc changed, or the checked route regressed -- this pin cannot be read \
             until that is resolved."
        )
    })?;
    anyhow::ensure!(
        checked.iter().all(|value| value.is_finite()),
        "the checked route returned a non-finite endpoint on the pinned arc"
    );

    // Events ON: the divergence. Every density that fails must fail as
    // `Eclipse(Bracket)`; a different failure name means the mechanism moved and
    // the header of this file is stale.
    let mut bracketed = Vec::new();
    for count in DENSITIES {
        let t_eval = sample_times(count);
        match sampled_failure(&context, init_equ, &t_eval, true)? {
            None => {}
            Some(FinalPropagationFailure::Eclipse(EclipseError::Bracket)) => {
                bracketed.push(count);
            }
            Some(other) => anyhow::bail!(
                "sampled route failed with {other} at {count} output times, not \
                 Eclipse(Bracket); re-read the mechanism in this file's header"
            ),
        }
    }
    anyhow::ensure!(
        !bracketed.is_empty(),
        "the sampled route no longer diverges from the checked route at any of \
         {DENSITIES:?} output times. If the PerturbDeviation sampling hole was \
         FIXED, delete this pin and record the fix. If it merely moved, widen \
         the sweep -- do not relax the assertion."
    );

    // Control: the same requests with events OFF never diverge. Without this arm
    // the assertion above would also be satisfied by a sampled route that had
    // simply stopped working.
    for count in DENSITIES {
        let t_eval = sample_times(count);
        let failure = sampled_failure(&context, init_equ, &t_eval, false)?;
        anyhow::ensure!(
            failure.is_none(),
            "the events-off control failed at {count} output times ({}); the \
             divergence is then not specific to event handling and this file's \
             mechanism is wrong",
            failure.map_or_else(String::new, |failure| failure.to_string())
        );
    }

    println!("SAMPLED_ROUTE_DIVERGES_AT {bracketed:?} of {DENSITIES:?} output times");
    Ok(())
}
