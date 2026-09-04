//! What each physics term in the strict-HF force model is WORTH, in metres.
//!
//! # Why this exists
//!
//! The 1.0 m strict-HF accuracy gates difference an arc against ITSELF at a
//! tighter tolerance, so they bound integrator truncation and are structurally
//! blind to force-model bias — a 943x-over-budget density model stayed green
//! through them. Any proposal to simplify a force therefore cannot be scored by
//! those gates; it needs a model-vs-model differential on production-shaped
//! arcs, which is what this program takes.
//!
//! # Method
//!
//! Every arm is the flown configuration with ONE thing changed, propagated over
//! the same corpus, and scored as the terminal position separation from the
//! flown arm on the SAME draw at the SAME tolerance. Both arms of every pair run
//! at the same `eps`, so integrator truncation is common-mode to first order and
//! what survives is the force-model difference plus step-schedule noise.
//!
//! Two tolerance rungs are reported for each arm and they answer different
//! questions:
//!
//! * `eps = 1e-10` — the MODEL rung. Vern7 converges to ~0.007 m RMS here, two
//!   to three orders under anything worth acting on, so a separation at this
//!   rung is the physics and not the stepper.
//! * `eps = 1e-8` — the FLOWN rung, the sealed tolerance. This is what the
//!   change would actually do to a campaign trajectory: model bias plus the
//!   re-timed step schedule that any bit-moving change drags along.
//!
//! A term whose ablation moves the endpoint by less than the 10 m science budget
//! is a candidate for simplification. A term that moves it by kilometres is not,
//! however cheap it looks in a profile.
//!
//! # Corpus
//!
//! Twelve production-shaped LEO arcs, the same generator the sealed stepper A/B
//! flies (`tests/stepper_method_ab.rs`), each draw propagating from its OWN
//! epoch so that the solar/geomagnetic drivers and the Sun geometry differ
//! across draws as well as the orbit.
//!
//! That used to be a deliberate DIFFERENCE from the stepper A/B, which built a
//! per-draw epoch, spent it on `with_ephemeris_for_arc`, and then handed `JD0`
//! to every `ScalarPropagationContext` — and because Sun and Moon resolve
//! dynamically from the running Julian Day, its twelve draws shared one epoch.
//! That bug was fixed in the A/B harness on 2026-08-10, so the two corpora now
//! agree; both fly twelve epochs, and both report 75,661 RHS evaluations for
//! the flown arm at the sealed tolerance.
//!
//! Usage: `force_ablation [timing_reps]` (0 = accuracy only).

use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use lightyear_odeint_rs::integrator::{
    integrate_final_checked, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest,
};
use lightyear_odeint_rs::probe;
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags, StepperMethod};
use num_traits::ToPrimitive;
use satpy_core::{eci2equinoc_impl_f64, equinoc2eci_impl, kep2eci_impl, SEC_PER_DAY};

const DIR_R6_D15: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

const JD0: f64 = 2_460_310.5;
const TOF_S: f64 = 43_200.0;
const DRAWS: usize = 12;

/// The model rung and the flown rung. See the header for what each is for.
const MODEL_EPS: f64 = 1.0e-10;
const FLOWN_EPS: f64 = 1.0e-8;

/// The compiled Part A science authority, read rather than restated.
const fn part_a_hybrid() -> &'static nd_config::PartAHybridControls {
    nd_config::CompiledPartAScienceV1::part_a_v1().hybrid()
}

/// The compiled stepper, resolved rather than restated.
///
/// This file held `StepperMethod::Vern7` as a literal under a doc line that
/// claimed every field came from the seal. The literal happened to agree with
/// compiled science, which is what makes the pattern dangerous rather than
/// merely untidy: it would have kept agreeing, silently, right up until the
/// seal moved, and then this program would have priced force-model ablations
/// against a stepper the campaign does not fly. Four sibling harnesses in this
/// crate carried the same shadow with `Vern9` through the R26 swap.
#[expect(
    clippy::panic,
    reason = "a stepper this file cannot build must stop the run, not silently measure a different one"
)]
fn authority_stepper() -> StepperMethod {
    match part_a_hybrid().integrator_method {
        "vern7" => StepperMethod::Vern7,
        "vern9" => StepperMethod::Vern9,
        other => panic!("compiled science selects a stepper this file does not build: {other}"),
    }
}

/// The flown strict-HF force shape, every field read from sealed science.
fn flown_config() -> ForceConfig {
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
        integrator_method: authority_stepper(),
        ..ForceConfig::default()
    }
}

/// One ablation: a label and the single edit it makes to the flown shape.
struct Arm {
    label: &'static str,
    edit: fn(&mut ForceConfig),
}

const ARMS: &[Arm] = &[
    Arm {
        label: "flown",
        edit: |_config| {},
    },
    // Gravity truncation ladder. R4 refuted the RECURRENCE rewrite; the ORDER
    // itself was never laddered against a science budget.
    Arm {
        label: "grav_order4",
        edit: |config| config.sph_order = 4,
    },
    Arm {
        label: "grav_order3",
        edit: |config| config.sph_order = 3,
    },
    Arm {
        label: "grav_order2",
        edit: |config| config.sph_order = 2,
    },
    // Third-body necessity at the science budget.
    Arm {
        label: "no_moon",
        edit: |config| config.force_flags &= !ForceFlags::MOON_GRAVITY,
    },
    Arm {
        label: "no_sun",
        edit: |config| config.force_flags &= !ForceFlags::SUN_GRAVITY,
    },
    // SRP: the force the whole eclipse machinery exists to gate.
    Arm {
        label: "no_srp",
        edit: |config| config.force_flags &= !ForceFlags::SRP,
    },
    // Drag: what the 21% JB2008 lane buys.
    Arm {
        label: "no_drag",
        edit: |config| {
            config.force_flags &= !ForceFlags::DRAG;
            config.atm_model = 0;
        },
    },
    // The YARDSTICK. Model 7 is an ALREADY-SEALED approximation of model 4, so
    // its endpoint separation is the size of a bias this project has already
    // accepted. Any new lever should be read against this number, not against
    // zero.
    Arm {
        label: "atm_exact_m4",
        edit: |config| config.atm_model = 4,
    },
];

struct Draw {
    init_equ: [f64; 6],
    base_at_tf: [f64; 6],
    epoch: f64,
    config: ForceConfig,
    label: String,
}

/// The stepper A/B's corpus generator, with the epoch actually flown.
fn corpus() -> Result<Vec<Draw>> {
    let mut draws = Vec::with_capacity(DRAWS);
    for index in 0..DRAWS {
        let step = index.to_f64().context("draw index must convert to f64")?;
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
        let config = flown_config()
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
                "draw{index:02}_alt{:.0}_e{:.3}",
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
    wall_ms: f64,
}

/// Probe counters are process-global; keep every census under one lock.
static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn propagate(
    draw: &Draw,
    arm: &Arm,
    eps: f64,
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
) -> Result<Option<Run>> {
    propagate_as(draw, arm.edit, arm.label, eps, packed)
}

fn propagate_as(
    draw: &Draw,
    edit: fn(&mut ForceConfig),
    label: &str,
    eps: f64,
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
) -> Result<Option<Run>> {
    let mut config = draw.config;
    edit(&mut config);
    config.eps = eps;
    // The edit may have removed a body, which changes which bodies must resolve
    // dynamically; re-resolve rather than carrying the flown shape's flags.
    let config = config
        .with_ephemeris_for_arc(draw.epoch, draw.epoch + TOF_S / SEC_PER_DAY)
        .context("ablated config must still resolve its ephemeris")?;

    let guard = PROBE_LOCK
        .lock()
        .map_err(|_ignored| anyhow::anyhow!("probe lock poisoned"))?;
    probe::reset().context("probe census must reset")?;
    let gravity = ScalarGravityAssets::new(Arc::clone(packed));
    let context = ScalarPropagationContext::new(draw.epoch, Arc::new(config), gravity);
    let start = std::time::Instant::now();
    let outcome = integrate_final_checked(
        ScalarPropagationRequest::new(&context, draw.init_equ, &[TOF_S], 0.0, TOF_S)
            .with_events(true),
    );
    let wall_ms = start.elapsed().as_secs_f64() * 1.0e3;
    let delta = match outcome {
        Ok(delta) => delta,
        Err(failure) if failure.is_physical_infeasible() => {
            drop(guard);
            return Ok(None);
        }
        Err(failure) => {
            drop(guard);
            return Err(anyhow::anyhow!("{failure:?}")
                .context(format!("{} must propagate at {label}", draw.label)));
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
    ensure!(
        state.iter().all(|value| value.is_finite()),
        "{} produced a non-finite terminal state at {label}",
        draw.label
    );
    Ok(Some(Run {
        state,
        rhs_evals,
        steps,
        wall_ms,
    }))
}

fn pos_sep_m(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let [a_x, a_y, a_z, ..] = *a;
    let [b_x, b_y, b_z, ..] = *b;
    let dx = a_x - b_x;
    let dy = a_y - b_y;
    let dz = a_z - b_z;
    dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt() * 1_000.0
}

fn report_rung(
    label: &str,
    eps: f64,
    draws: &[Draw],
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
) -> Result<()> {
    let mut reference = Vec::with_capacity(draws.len());
    let flown = ARMS.first().context("the flown arm must be first")?;
    for draw in draws {
        let run = propagate(draw, flown, eps, packed)?
            .with_context(|| format!("{} has no endpoint under the flown model", draw.label))?;
        // Printed so a run under one shadow model can be differenced against a
        // run under another ACROSS PROCESSES: the in-process comparison below
        // scores every arm against the reference built in the same process, and
        // an environment-armed model change moves the reference too.
        let [x_km, y_km, z_km, ..] = run.state;
        println!(
            "ABL_ENDPOINT rung={label} {} x={x_km:.17e} y={y_km:.17e} z={z_km:.17e} evals={} steps={}",
            draw.label, run.rhs_evals, run.steps
        );
        reference.push(run);
    }

    for arm in ARMS {
        let mut worst = 0.0_f64;
        let mut worst_label = "";
        let mut sum_sq = 0.0_f64;
        let mut evals = 0u64;
        let mut steps = 0u64;
        let mut scored = 0usize;
        for (draw, base) in draws.iter().zip(&reference) {
            let Some(run) = propagate(draw, arm, eps, packed)? else {
                println!("ABL_SKIP rung={label} arm={} {}", arm.label, draw.label);
                continue;
            };
            let sep = pos_sep_m(&run.state, &base.state);
            if sep > worst {
                worst = sep;
                worst_label = &draw.label;
            }
            sum_sq = sep.mul_add(sep, sum_sq);
            evals = evals.checked_add(run.rhs_evals).context("eval overflow")?;
            steps = steps.checked_add(run.steps).context("step overflow")?;
            scored = scored.checked_add(1).context("draw count overflow")?;
            println!(
                "ABL_DRAW rung={label} arm={} {} sep_m={sep:.6e} evals={} steps={}",
                arm.label, draw.label, run.rhs_evals, run.steps
            );
        }
        let scored_f64 = scored.to_f64().context("scored count must convert")?;
        let rms = (sum_sq / scored_f64).sqrt();
        let base_evals: u64 = reference.iter().map(|run| run.rhs_evals).sum();
        let eval_ratio = evals.to_f64().context("evals must convert")?
            / base_evals.to_f64().context("baseline evals must convert")?;
        let arm_label = arm.label;
        println!(
            "ABL_ARM rung={label} eps={eps:e} arm={arm_label} n={scored} rms_m={rms:.6e} \
             worst_m={worst:.6e} worst_draw={worst_label} evals={evals} steps={steps} \
             eval_ratio={eval_ratio:.4}"
        );
    }
    Ok(())
}

/// Interleaved min-of-N wall, one arm per round, at the FLOWN tolerance.
fn report_wall(
    reps: usize,
    draws: &[Draw],
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
) -> Result<()> {
    let mut best = vec![f64::INFINITY; ARMS.len()];
    for _rep in 0..reps {
        for (index, arm) in ARMS.iter().enumerate() {
            let mut total = 0.0;
            for draw in draws {
                if let Some(run) = propagate(draw, arm, FLOWN_EPS, packed)? {
                    total += run.wall_ms;
                }
            }
            if let Some(slot) = best.get_mut(index) {
                *slot = slot.min(total);
            }
        }
    }
    let baseline = *best.first().context("the flown arm must be first")?;
    for (index, arm) in ARMS.iter().enumerate() {
        let ms = best.get(index).copied().unwrap_or(f64::NAN);
        println!(
            "ABL_WALL arm={} corpus_ms={ms:.3} ratio={:.4}",
            arm.label,
            ms / baseline
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let timing_reps: usize = std::env::args()
        .nth(1)
        .map_or(Ok(0), |value| value.parse().context("reps must parse"))?;

    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("gravity coefficients must load")?;
    let packed =
        lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")?;

    let draws = corpus()?;
    println!("ABL_CORPUS draws={}", draws.len());

    report_rung("model", MODEL_EPS, &draws, &packed)?;
    report_rung("flown", FLOWN_EPS, &draws, &packed)?;

    if timing_reps > 0 {
        report_wall(timing_reps, &draws, &packed)?;
    }
    Ok(())
}
