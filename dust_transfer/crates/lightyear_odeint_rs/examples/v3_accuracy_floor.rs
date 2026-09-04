//! Measures the noise floor of the accuracy metric both `strict_hf_pin`
//! accuracy gates bound.
//!
//! Those gates bound `|endpoint(production eps) - endpoint(1e-12)|`. Both
//! tolerances are re-propagated and they take different step sequences, so the
//! difference does not shrink to the arc's truncation error -- it carries the
//! decorrelation between two step sequences as well. That is why the floor has
//! to be measured instead of assumed; assuming it is what put a 1 cm bound on a
//! 1 m quantity once already.
//!
//! The floor is measured by perturbing the initial equinoctial state at ULP
//! scale, which changes no physics, and re-reading the same difference.
//!
//! # One ULP of each element is NOT a perturbation of each element
//!
//! The six elements span five orders of magnitude: element 0 is 7178.137 km and
//! element 2 is 0.0227 dimensionless, so one ULP is 9.1e-13 absolute on the
//! first and 3.5e-18 on the third. What reaches the integrator is a state whose
//! position components are ~7e3 km with a ULP of ~9.1e-13 km, and element 2
//! enters position multiplied by the semi-major axis. Its one-ULP bump
//! displaces position by ~2.5e-14 km -- below the representable resolution of
//! the coordinate it perturbs. It rounds away, and the production trajectory
//! comes back BIT IDENTICAL.
//!
//! That is not a small loss of sample size, it is a dead cell that reads as a
//! live one: the metric still moves, because the 1e-12 reference takes a
//! different step sequence and responds where production does not. The element
//! contributes a reading to the distribution while constraining nothing about
//! the quantity the gate bounds.
//!
//! The fix is not a bigger fixed nudge, which would overshoot the five elements
//! that were already live. Each element is CALIBRATED: the program searches for
//! the smallest ULP count on that element alone whose production endpoint
//! differs in bits from the unperturbed one, and every later draw is built from
//! those per-element counts. Calibration that finds no such count is a hard
//! error, and so is a draw whose endpoint comes back bit identical -- an inert
//! perturbation fails the harness rather than quietly reporting a floor
//! measured with a dead axis.
//!
//! # What it reports
//!
//! 1. the arc's own truncation reading, unperturbed, with steps and segments;
//! 2. the calibrated ULP count and absolute nudge per element;
//! 3. the metric and endpoint floors over `FLOOR_DRAWS` draws as a
//!    DISTRIBUTION (min / p50 / p90 / max), under two seeds, because a bound
//!    sized off one draw set is a bound sized off one draw set;
//! 4. an eps ladder, 1e-8 / 1e-10 / 1e-12 -- the standing "re-run it at
//!    eps/1e4" control. It is applied to the ENDPOINT floor, which needs no
//!    reference propagation and so is defined at a single tolerance; the metric
//!    floor cannot take the same ladder because the metric's own reference is
//!    1e-12, so `eps/1e4` collapses it to zero by construction. The rung's own
//!    unperturbed error is printed beside it, so the floor can be read against
//!    the truncation error rather than in isolation;
//! 5. a `dt_max` sweep, which buys back accepted steps at nearly constant
//!    segment count;
//! 6. an events-off arm, which removes eclipse segmentation and so moves
//!    segments at nearly constant tolerance.
//!
//! Items 5 and 6 exist to separate the two candidate drivers of the recorded
//! floor. The hypothesis on record was that the floor is chaos amplification
//! over the step sequence, resting on a 3,844-to-471 fall in accepted steps;
//! the obvious alternative was segments, which the 2026-08-02 log entry shows
//! falling 491 to 66 over the eclipse work. Both counters moved together there,
//! so only a lever that moves one at a time can say which the floor follows.
//! `dt_max` and the events flag do that on today's code, without reverting
//! three landed changes.
//!
//! Item 6 changes physics, unlike item 5: with events off the eclipse boundary
//! is no longer resolved to a root, so the SRP shadow is sampled rather than
//! bracketed. It is read as a mechanism probe on the RATIO between two floors
//! measured the same way, never as an accuracy statement about either arm.
//!
//! Model 4 is run as a CONTROL, not for its own sake: the recorded 1.035 m is a
//! model-4 number, so a model-4 arm is what can reproduce or refute it.
//!
//! # What it found, so a reader knows what to expect
//!
//! The step-count hypothesis is REFUTED, and not by the levers here -- by the
//! history. `6a856aa`, the tree that recorded 1.035 m, runs this arc in 359
//! steps and 64 segments against the tip's 461 and 67. Neither counter moved,
//! and steps went UP. The "3,844 steps" the hypothesis rested on belongs to a
//! transient tree a week later. What did move is accuracy: the arc's own
//! truncation error fell 26.6x and the floor fell 25.7x with it.
//!
//! The two levers here corroborate the direction. `dt_max` moves steps 9.6x at
//! constant segments and the floor falls 17x, so more integration work means a
//! LOWER floor. The eps ladder says the same thing over four decades.
//!
//! The events lever, item 6, turned out to be INERT and is kept only so the
//! next reader does not re-derive it: turning events off moves segments 67 to
//! 57, which is nowhere near the 491-to-66 the history needs. It constrains
//! nothing about segments and must not be quoted as if it did.
//!
//! ```sh
//! cargo run --release -p lightyear_odeint_rs --example v3_accuracy_floor
//! ```

use std::sync::Arc;

use anyhow::{Context, Result};
use lightyear_odeint_rs::integrator::{
    integrate_final_checked, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest,
};
use lightyear_odeint_rs::probe;
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags, StepperMethod};
use satpy_core::{eci2equinoc_impl_f64, equinoc2eci_impl, kep2eci_impl, SEC_PER_DAY};

const JD0: f64 = 2_460_310.5;
const TOF_S: f64 = 43_200.0;
const REFERENCE_EPS: f64 = 1.0e-12;
const DIR_R6_D15: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

/// Draws in the metric arms.
const FLOOR_DRAWS: usize = 48;

/// Draws in the endpoint-only arms, which are half the cost per draw.
const LADDER_DRAWS: usize = 24;

/// Deterministic, so two runs of this program are comparable to each other.
const FLOOR_SEED: u64 = 0x5EED_F100_0000_0001;

/// A second, unrelated seed, so the report can show whether the distribution is
/// a property of the arc or of one draw set.
const FLOOR_SEED_B: u64 = 0x5EED_F100_0000_00B2;

/// The largest ULP count calibration will try before declaring an element
/// unreachable. 2^20 ULP of element 2 is ~3.7e-12 absolute -- still ~1e-10
/// RELATIVE, far below anything that could change physics, and far above what
/// element 2 turns out to need.
const MAX_CALIBRATION_ULPS: u64 = 1 << 20;

/// Read from the sealed authority, never restated, for the reason
/// `strict_hf_pin` gives: a literal copy keeps passing while measuring a
/// configuration nobody runs.
const fn part_a_hybrid() -> &'static nd_config::PartAHybridControls {
    nd_config::CompiledPartAScienceV1::part_a_v1().hybrid()
}

/// One propagation's settings.
///
/// A struct rather than five positional parameters because `atm_model`, `eps`
/// and `dt_max` are all things a caller could plausibly transpose, and three of
/// the arms below differ from production in exactly one field.
#[derive(Clone, Copy)]
struct Setup {
    atm_model: i32,
    eps: f64,
    /// `None` means the compiled production `dt_max`.
    dt_max_s: Option<f64>,
    events: bool,
}

impl Setup {
    const fn production(atm_model: i32, eps: f64) -> Self {
        Self {
            atm_model,
            eps,
            dt_max_s: None,
            events: true,
        }
    }

    const fn at_eps(self, eps: f64) -> Self {
        Self { eps, ..self }
    }
}

/// The same builder `strict_hf_pin` uses, with the atmosphere left open so the
/// authority arm and the exact-profile control differ in that field ALONE.
/// The compiled stepper, resolved rather than restated.
///
/// Replaces a paired `assert_eq!(..., "vern9")` tripwire and hardcoded
/// `StepperMethod::Vern9`. The assert fired correctly on the Vern9 -> Vern7
/// swap, but relaxing it without also editing the literal beside it would have
/// left this harness measuring a stepper the campaign no longer flies, green
/// and silent. Resolving the token removes that second step; the `panic!`
/// keeps the fail-closed property for a stepper this file cannot build.
fn authority_stepper() -> Result<StepperMethod> {
    match part_a_hybrid().integrator_method {
        "vern7" => Ok(StepperMethod::Vern7),
        "vern9" => Ok(StepperMethod::Vern9),
        other => {
            anyhow::bail!("compiled science selects a stepper this file does not build: {other}")
        }
    }
}

fn dust_config(setup: Setup) -> Result<ForceConfig> {
    let controls = part_a_hybrid();
    Ok(ForceConfig {
        sph_order: controls.gravity_order,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        subtract_first_order: true,
        atm_model: setup.atm_model,
        am_ratio: controls.dust_am_ratio,
        cd: controls.dust_cd,
        cr: controls.dust_cr,
        target_propagation_mode: 0,
        dt_max: setup.dt_max_s.unwrap_or(controls.dt_max_s),
        eps: setup.eps,
        integrator_method: authority_stepper()?,
        ..ForceConfig::default()
    })
}

/// One propagation's endpoint and the counters that produced it.
///
/// Steps and segments are both carried because they are the two candidate
/// drivers of the recorded floor and the history moved them together; a reading
/// that reported only one could not distinguish them.
#[derive(Clone, Copy)]
struct Endpoint {
    state: [f64; 6],
    steps: u64,
    segments: u64,
}

fn endpoint(init_equ: [f64; 6], setup: Setup) -> Result<Endpoint> {
    let packed =
        lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")?;

    let config = dust_config(setup)?
        .with_ephemeris_for_arc(JD0, JD0 + TOF_S / SEC_PER_DAY)
        .context("production ephemeris and JB2008 assets must cover the pinned arc")?;

    probe::reset().context("probe census must reset")?;
    let gravity = ScalarGravityAssets::new(packed);
    let context = ScalarPropagationContext::new(JD0, Arc::new(config), gravity);
    let delta = integrate_final_checked(
        ScalarPropagationRequest::new(&context, init_equ, &[TOF_S], 0.0, TOF_S)
            .with_events(setup.events),
    )
    .context("the arc must propagate")?;

    let mut state = [0.0; 6];
    equinoc2eci_impl(&init_equ, 6, TOF_S, 0.0, &mut state);
    for (component, delta_component) in state.iter_mut().zip(delta) {
        *component += delta_component;
    }

    let census = probe::snapshot();
    let steps = census
        .iter()
        .try_fold(0_u64, |acc, entry| acc.checked_add(entry.steps))
        .context("step census overflow")?;
    let segments = census
        .iter()
        .try_fold(0_u64, |acc, entry| acc.checked_add(entry.segments))
        .context("segment census overflow")?;
    Ok(Endpoint {
        state,
        steps,
        segments,
    })
}

/// Position separation in metres.
fn separation_m(left: &[f64; 6], right: &[f64; 6]) -> f64 {
    let mut sum_sq = 0.0;
    for (l, r) in left.iter().zip(right.iter()).take(3) {
        let d = l - r;
        sum_sq += d * d;
    }
    sum_sq.sqrt() * 1000.0
}

/// Bit equality of all six components.
///
/// This, not `separation_m == 0.0`, is what decides whether a perturbation
/// reached the trajectory: only identical bits across every component are
/// evidence that the integration ran on inputs it could not tell apart.
fn bit_identical(left: &[f64; 6], right: &[f64; 6]) -> bool {
    left.iter()
        .zip(right.iter())
        .all(|(l, r)| l.to_bits() == r.to_bits())
}

/// Move one element by `ulps`, leaving the other five untouched.
fn nudge(base: [f64; 6], index: usize, ulps: u64, up: bool) -> [f64; 6] {
    let mut perturbed = base;
    for (slot_index, slot) in perturbed.iter_mut().enumerate() {
        if slot_index == index {
            let bits = slot.to_bits();
            *slot = f64::from_bits(if up {
                bits.wrapping_add(ulps)
            } else {
                bits.wrapping_sub(ulps)
            });
        }
    }
    perturbed
}

/// `SplitMix64`. Seeded and in-file, so the draws are reproducible without
/// taking a dependency for six numbers at a time.
struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ z.wrapping_shr(30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ z.wrapping_shr(27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ z.wrapping_shr(31)
    }

    /// Uniform in `1..=bound`, never zero: a zero draw would be a silently
    /// unperturbed element, which is the defect this file exists to stop.
    fn next_in_range(&mut self, bound: u64) -> u64 {
        let span = bound.max(1);
        self.next_u64()
            .checked_rem(span)
            .unwrap_or(0)
            .wrapping_add(1)
    }
}

/// Smallest ULP count on `index` alone that moves the production endpoint's
/// bits, searched by doubling.
///
/// Doubling rather than a linear scan because the answer spans an order of
/// magnitude across the six elements and every probe is a propagation.
fn calibrate_element(
    init_equ: [f64; 6],
    setup: Setup,
    baseline: &Endpoint,
    index: usize,
) -> Result<u64> {
    let mut ulps = 1_u64;
    while ulps <= MAX_CALIBRATION_ULPS {
        let probe_state = endpoint(nudge(init_equ, index, ulps, true), setup)?;
        if !bit_identical(&probe_state.state, &baseline.state) {
            return Ok(ulps);
        }
        ulps = ulps.wrapping_mul(2);
    }
    anyhow::bail!(
        "element {index} is INERT: {MAX_CALIBRATION_ULPS} ULP leaves the production endpoint bit \
         identical, so it cannot contribute a floor sample. The measurement is abandoned rather \
         than reported with a dead axis."
    )
}

/// Calibrate all six, printing each so a reader can see what was perturbed.
fn calibrate(
    init_equ: [f64; 6],
    setup: Setup,
    baseline: &Endpoint,
    label: &str,
) -> Result<[u64; 6]> {
    let mut ulps = [1_u64; 6];
    for index in 0..6 {
        let count = calibrate_element(init_equ, setup, baseline, index)?;
        let value = init_equ.get(index).copied().unwrap_or(f64::NAN);
        let bumped = f64::from_bits(value.to_bits().wrapping_add(count));
        if let Some(slot) = ulps.get_mut(index) {
            *slot = count;
        }
        println!(
            "V3_FLOOR {label} calibrate elem{index} value={value:.17e} ulps={count} \
             absolute={:.3e}",
            bumped - value
        );
    }
    Ok(ulps)
}

/// A distribution of readings, sorted, with the order statistics worth printing.
struct Spread {
    sorted: Vec<f64>,
}

impl Spread {
    fn new(mut values: Vec<f64>) -> Self {
        values.sort_by(f64::total_cmp);
        Self { sorted: values }
    }

    fn quantile(&self, percent: usize) -> f64 {
        let len = self.sorted.len();
        let rank = percent
            .checked_mul(len)
            .and_then(|scaled| scaled.checked_div(100))
            .unwrap_or(0)
            .min(len.saturating_sub(1));
        self.sorted.get(rank).copied().unwrap_or(f64::NAN)
    }

    fn min(&self) -> f64 {
        self.sorted.first().copied().unwrap_or(f64::NAN)
    }

    fn max(&self) -> f64 {
        self.sorted.last().copied().unwrap_or(f64::NAN)
    }
}

/// Everything one arm produces: the two floors, and the counters of the
/// unperturbed propagation they were measured around.
struct FloorReading {
    metric: Spread,
    endpoint_move: Spread,
    base: Endpoint,
}

/// Run `draws` perturbations and collect the floor readings.
///
/// `reference_eps` is `Some` on the metric arms and `None` on the
/// endpoint-only arms.
fn measure_floor(
    init_equ: [f64; 6],
    setup: Setup,
    reference_eps: Option<f64>,
    ulps: &[u64; 6],
    draws: usize,
    seed: u64,
) -> Result<FloorReading> {
    let base = endpoint(init_equ, setup)?;

    let mut rng = SplitMix64::new(seed);
    let mut metric_values = Vec::with_capacity(draws);
    let mut endpoint_values = Vec::with_capacity(draws);

    for draw_index in 0..draws {
        // All six elements at once: the calibrated counts are exactly where
        // each element becomes visible, so a draw that moves every element by a
        // visible-but-minimal amount is the smallest state perturbation the
        // trajectory can distinguish from no perturbation at all.
        let mut perturbed_equ = init_equ;
        for (index, bound) in ulps.iter().enumerate() {
            let magnitude = rng.next_in_range(*bound);
            let up = rng.next_u64() & 1 == 0;
            perturbed_equ = nudge(perturbed_equ, index, magnitude, up);
        }

        let production = endpoint(perturbed_equ, setup)?;
        anyhow::ensure!(
            !bit_identical(&production.state, &base.state),
            "draw {draw_index} left the production endpoint BIT IDENTICAL. The calibrated ULP \
             counts no longer reach the trajectory, so this run would report a floor built from \
             dead samples."
        );
        endpoint_values.push(separation_m(&production.state, &base.state));

        if let Some(reference_value) = reference_eps {
            let reference = endpoint(perturbed_equ, setup.at_eps(reference_value))?;
            metric_values.push(separation_m(&production.state, &reference.state));
        }
    }

    Ok(FloorReading {
        metric: Spread::new(metric_values),
        endpoint_move: Spread::new(endpoint_values),
        base,
    })
}

fn print_spread(label: &str, spread: &Spread) {
    println!(
        "V3_FLOOR {label} n={} min={:.6} p50={:.6} p90={:.6} max={:.6}",
        spread.sorted.len(),
        spread.min(),
        spread.quantile(50),
        spread.quantile(90),
        spread.max()
    );
}

/// One endpoint-only arm: run it, and print the floor beside the counters and
/// the unperturbed error that produced it.
fn report_endpoint_arm(
    init_equ: [f64; 6],
    setup: Setup,
    reference: &Endpoint,
    ulps: &[u64; 6],
    label: &str,
) -> Result<()> {
    let reading = measure_floor(init_equ, setup, None, ulps, LADDER_DRAWS, FLOOR_SEED)?;
    print_spread(
        &format!(
            "{label} steps={} segments={} unperturbed_err_m={:.6} ENDPOINT_FLOOR_m",
            reading.base.steps,
            reading.base.segments,
            separation_m(&reading.base.state, &reference.state)
        ),
        &reading.endpoint_move,
    );
    Ok(())
}

fn main() -> Result<()> {
    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("gravity coefficients must load")?;

    let kep = [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0];
    let mut init_eci = [0.0; 6];
    kep2eci_impl(&kep, true, 0.0, 0.0, true, &mut init_eci);
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let production_eps = part_a_hybrid().tolerance;
    let authority_model = part_a_hybrid().atmosphere_model;
    println!("V3_FLOOR init_equ={init_equ:?}");
    println!(
        "V3_FLOOR authority_atmosphere_model={authority_model} production_eps={production_eps:.3e} \
         reference_eps={REFERENCE_EPS:.0e} draws={FLOOR_DRAWS} ladder_draws={LADDER_DRAWS}"
    );

    for atm_model in [authority_model, 4] {
        let role = if atm_model == authority_model {
            "AUTHORITY / what the campaign flies"
        } else {
            "exact profile / CONTROL, recorded floor 1.035 m"
        };
        println!("\n===== atm_model {atm_model} ({role}) =====");
        let label = format!("atm={atm_model}");
        let setup = Setup::production(atm_model, production_eps);

        let base_production = endpoint(init_equ, setup)?;
        let base_reference = endpoint(init_equ, setup.at_eps(REFERENCE_EPS))?;
        println!(
            "V3_FLOOR {label} unperturbed err_m={:.6} steps={} segments={} \
             reference_steps={} reference_segments={}",
            separation_m(&base_production.state, &base_reference.state),
            base_production.steps,
            base_production.segments,
            base_reference.steps,
            base_reference.segments
        );

        let ulps = calibrate(init_equ, setup, &base_production, &label)?;

        for seed in [FLOOR_SEED, FLOOR_SEED_B] {
            let reading = measure_floor(
                init_equ,
                setup,
                Some(REFERENCE_EPS),
                &ulps,
                FLOOR_DRAWS,
                seed,
            )?;
            let seeded = format!("{label} seed={seed:#x}");
            print_spread(&format!("{seeded} METRIC_FLOOR_m"), &reading.metric);
            print_spread(
                &format!("{seeded} ENDPOINT_FLOOR_m"),
                &reading.endpoint_move,
            );
        }

        // The eps ladder. If the floor were chaos amplification of the input
        // perturbation it would be a property of the trajectory and should
        // barely move across four decades of tolerance; if it tracks truncation
        // it should collapse with eps, in step with the unperturbed error
        // printed beside it.
        for ladder_eps in [
            production_eps,
            production_eps / 1.0e2,
            production_eps / 1.0e4,
        ] {
            report_endpoint_arm(
                init_equ,
                setup.at_eps(ladder_eps),
                &base_reference,
                &ulps,
                &format!("{label} LADDER eps={ladder_eps:.0e}"),
            )?;
        }
    }

    // Steps at nearly constant segments.
    println!("\n===== dt_max sweep (atm_model 4): steps move, segments do not =====");
    let sweep_setup = Setup::production(4, production_eps);
    let sweep_base = endpoint(init_equ, sweep_setup)?;
    let sweep_reference = endpoint(init_equ, sweep_setup.at_eps(REFERENCE_EPS))?;
    let sweep_ulps = calibrate(init_equ, sweep_setup, &sweep_base, "atm=4 SWEEP")?;
    for dt_max in [300.0_f64, 100.0, 30.0, 10.0] {
        report_endpoint_arm(
            init_equ,
            Setup {
                dt_max_s: Some(dt_max),
                ..sweep_setup
            },
            &sweep_reference,
            &sweep_ulps,
            &format!("atm=4 SWEEP dt_max={dt_max:.0}"),
        )?;
    }

    // Segments at constant tolerance. Physics differs between the two arms, so
    // only the RATIO of the two floors is read, never either arm's accuracy.
    println!("\n===== events sweep (atm_model 4): segments move, tolerance does not =====");
    for events in [true, false] {
        let events_setup = Setup {
            events,
            ..sweep_setup
        };
        let events_reference = endpoint(init_equ, events_setup.at_eps(REFERENCE_EPS))?;
        let events_base = endpoint(init_equ, events_setup)?;
        let events_ulps = calibrate(
            init_equ,
            events_setup,
            &events_base,
            &format!("atm=4 EVENTS events={events}"),
        )?;
        report_endpoint_arm(
            init_equ,
            events_setup,
            &events_reference,
            &events_ulps,
            &format!("atm=4 EVENTS events={events}"),
        )?;
    }

    Ok(())
}
