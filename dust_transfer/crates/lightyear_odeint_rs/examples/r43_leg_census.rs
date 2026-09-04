//! Leg census for the R43 stepper audit: where the arc's accepted steps and
//! RHS evaluations actually live.
//!
//! Three questions this exists to settle, none of which an aggregate answers:
//!
//! 1. **Multistep viability.** An Adams-Bashforth-Moulton method of order k
//!    needs k-1 accepted steps of history before it reaches its order, and
//!    every solver entry here is a cold start (`solver.rs` keeps no controller
//!    or history state across entries, and the h-carry that would have was
//!    refuted at `c546130`). So ABM is decided by the DISTRIBUTION of accepted
//!    steps per entry, not by the mean: what matters is the share of steps
//!    living on entries long enough to amortize a restart.
//! 2. **The root-refinement clamp's price.** `MAX_ROOT_REFINEMENT_STEP_S =
//!    10 s` caps `dt_max` on eclipse root-transaction and bracket-replay legs,
//!    against a production `dt_max` of 300 s. The existing ramp census splits
//!    on SPAN and therefore files the bracket-replay leg — a ~65 s span run at
//!    a 10 s cap — with the unclamped Encke segments. `PROP_LEGCLASS` splits on
//!    the cap instead, which is the axis the clamp lives on.
//! 3. **Whether the step controller is the constraint.** `reject_frac` and the
//!    per-class rejection counts say whether a controller retune has anything
//!    to recover before any gain is designed.
//!
//! The corpus is the twelve draws of `tests/stepper_method_ab.rs`, copied
//! rather than shared because a `tests/` target cannot be imported by an
//! example. Draw 0 is the strict-HF V3 arc, so its counts must reproduce that
//! pin; if they stop doing so this program has drifted off production and no
//! other row here means anything. Force shape is read from the sealed Part A
//! authority for the same reason it is there.
//!
//! Requires `--features prop-census`. Without it `observe_leg` compiles to
//! nothing and the `PROP_LEGCLASS` rows never appear, which is reported rather
//! than silently tolerated.
//!
//! Usage: `r43_leg_census [draws]`

#![expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "diagnostic means printed from small integer counters; nothing asserts or digests them"
)]

use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use lightyear_odeint_rs::integrator::{
    integrate_final_checked, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest,
};
use lightyear_odeint_rs::probe;
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags, StepperMethod};
use satpy_core::{eci2equinoc_impl_f64, kep2eci_impl, SEC_PER_DAY};

const DIR_R6_D15: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

const JD0: f64 = 2_460_310.5;
const TOF_S: f64 = 43_200.0;
const DRAWS: usize = 12;

const fn part_a_hybrid() -> &'static nd_config::PartAHybridControls {
    nd_config::CompiledPartAScienceV1::part_a_v1().hybrid()
}

/// Production strict-HF force shape. Every field from the sealed authority
/// except the stepper, which is the dimension the audit varies.
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

struct Draw {
    init_equ: [f64; 6],
    /// The arc's OWN epoch. Held because the propagation context must be
    /// built on it: `with_ephemeris_for_arc` only bounds the catalogue window,
    /// and Sun/Moon resolve dynamically from the running JD, so a context
    /// built on a shared `JD0` flies a shared epoch no matter what the
    /// ephemeris call was handed.
    epoch: f64,
    config: ForceConfig,
    label: String,
}

/// The `stepper_method_ab` corpus verbatim: twelve arcs that share the
/// mission's regime but not its geometry, phase or epoch. Altitude and
/// eccentricity move because those are what the step-size controller responds
/// to; a corpus holding them fixed is one arc wearing twelve hats.
fn corpus(method: StepperMethod, draws: usize) -> Result<Vec<Draw>> {
    let mut out = Vec::with_capacity(draws);
    for index in 0..draws {
        let step = index as f64;
        let epoch = JD0 + step * 3.37;
        let kep = if index == 0 {
            [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0]
        } else {
            [
                7_028.137 + (step * 61.7) % 350.0,
                0.001 + (step * 0.0017) % 0.009,
                (28.5 + step * 13.9) % 180.0,
                (125.0 + step * 47.3) % 360.0,
                (210.0 + step * 71.1) % 360.0,
                (180.0 + step * 29.3) % 360.0,
            ]
        };
        let mut init_eci = [0.0; 6];
        kep2eci_impl(&kep, true, 0.0, 0.0, true, &mut init_eci);
        let mut init_equ = [0.0; 6];
        eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init_equ);
        let config = production_force_config(method)
            .with_ephemeris_for_arc(epoch, epoch + TOF_S / SEC_PER_DAY)
            .context("production ephemeris and JB2008 assets must cover every draw")?;
        out.push(Draw {
            init_equ,
            epoch,
            config,
            label: format!(
                "draw{index:02}_alt{:.0}_e{:.3}",
                kep.first().copied().unwrap_or_default() - 6_378.137,
                kep.get(1).copied().unwrap_or_default()
            ),
        });
    }
    Ok(out)
}

/// One arc. Reentry is an answer about the orbit, not a defect, so it reports
/// `false` rather than failing — the corpus keeps arcs that have no endpoint
/// out of the census instead of out of the program.
fn run_draw(draw: &Draw, packed: &Arc<satpy_core::PackedGravityCoeffs>) -> Result<bool> {
    let gravity = ScalarGravityAssets::new(Arc::clone(packed));
    let context = ScalarPropagationContext::new(draw.epoch, Arc::new(draw.config), gravity);
    match integrate_final_checked(
        ScalarPropagationRequest::new(&context, draw.init_equ, &[TOF_S], 0.0, TOF_S)
            .with_events(true),
    ) {
        Ok(_delta) => Ok(true),
        Err(failure) if failure.is_physical_infeasible() => Ok(false),
        Err(failure) => {
            Err(anyhow::anyhow!("{failure:?}").context(format!("{} must propagate", draw.label)))
        }
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let draws: usize = args.next().map_or(Ok(DRAWS), |value| {
        value.parse().context("draws must parse as usize")
    })?;
    ensure!(draws > 0 && draws <= DRAWS, "draws must be in 1..={DRAWS}");

    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("gravity coefficients must load")?;
    let packed =
        lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")?;

    // Resolved from the sealed token rather than hardcoded, and matched
    // exhaustively so that a token this program has never been run against
    // stops it instead of silently falling back to the library default (which
    // is Vern9 and is NOT what the campaign flies).
    let token = part_a_hybrid().integrator_method;
    let method = match token {
        "vern7" => StepperMethod::Vern7,
        "vern9" => StepperMethod::Vern9,
        other => anyhow::bail!("sealed integrator_method {other:?} is not a priced arm here"),
    };
    println!(
        "LEG_SHAPE draws={draws} method={token} eps={} dt_max={} atm_model={}",
        part_a_hybrid().tolerance,
        part_a_hybrid().dt_max_s,
        part_a_hybrid().atmosphere_model,
    );
    let corpus = corpus(method, draws)?;

    // Draw 0 alone first: it is the V3 pin arc, and its counts are the check
    // that this program is measuring production. Reported before the corpus
    // aggregate so a drifted harness is visible without reading further.
    for (index, draw) in corpus.iter().enumerate() {
        probe::reset().context("probe census must reset")?;
        let scored = run_draw(draw, &packed)?;
        if !scored {
            println!("LEG_SKIP {} no_endpoint", draw.label);
            continue;
        }
        let report = probe::report().context("per-draw census must render")?;
        for line in report
            .lines()
            .filter(|line| line.starts_with("PROP_LEGCLASS ") || line.starts_with("PROP_CENSUS "))
        {
            println!("LEG_DRAW {index:02} {line}");
        }
    }

    // Corpus aggregate: one reset, every scored draw, one report. The
    // per-draw rows above cannot be summed into this — the last bucket of
    // `PROP_LEGHIST` saturates, so a sum of printed histograms under-reads the
    // tail that the multistep question turns on.
    probe::reset().context("probe census must reset")?;
    let mut scored = 0usize;
    for draw in &corpus {
        if run_draw(draw, &packed)? {
            scored = scored.saturating_add(1);
        }
    }
    println!("LEG_SCORED {scored}");
    print!("{}", probe::report().context("corpus census must render")?);
    Ok(())
}
