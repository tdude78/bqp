//! Noise floor of the `stepper_method_ab` CORPUS metric, on the fixed corpus.
//!
//! # Why the existing floor harness does not answer this
//!
//! `v3_accuracy_floor` measures the floor of the `strict_hf_pin` gates, which
//! bound ONE arc's `|endpoint(eps) - endpoint(1e-12)|`. Stepper rankings are
//! not quoted from that number. They are quoted from `AB_SUMMARY`, which is an
//! RMS over twelve arcs, and an RMS over twelve is not twelve times a single
//! arc's floor: averaging suppresses independent scatter and does nothing to a
//! component common to the draws. The quantity that gates a ranking has to be
//! floored as the quantity it is.
//!
//! # Why it is being measured now
//!
//! The corpus flew ONE epoch for all twelve draws until 2026-08-10 (the
//! context was built on the shared `JD0` while `with_ephemeris_for_arc` got the
//! per-draw epoch, and only the latter is a window). Fixing it grew both arms'
//! errors 2-4x and made the tolerance ladder NON-MONOTONE -- Vern9 reads
//! 0.0173 m at 1e-9 and 0.0249 m at 1e-10. A ladder that worsens as the
//! tolerance tightens is not converging, and the harness's own anchor control
//! `AB_REFDRIFT` reaches 0.069 m against a Vern9 corpus RMS of 0.0640 m. So
//! the pre-fix corpus's clean, reproducible, monotone rankings were an artifact
//! of a vacuous epoch axis, and NOTHING may be ranked on the fixed corpus's
//! accuracy column until this floor exists.
//!
//! # Method
//!
//! The physics-neutral perturbation of `v3_accuracy_floor`: move the initial
//! equinoctial state by a calibrated ULP count per element, which changes no
//! physics, and re-read the whole corpus metric.
//!
//! Calibration is PER DRAW and not shared. The twelve arcs span 650-1000 km and
//! e = 0.001-0.025, so the ULP count at which an element becomes visible to the
//! endpoint differs between them; one shared calibration would over-perturb the
//! sensitive arcs and leave dead axes on the stiff ones. A draw whose endpoint
//! comes back bit identical is a hard error here for the same reason it is
//! there -- an inert perturbation reports a floor built from dead samples.
//!
//! Each perturbation draw re-runs the FULL corpus: both arms at the sealed
//! tolerance and the common Vern9 anchor at 1e-12, per arc, exactly as
//! `stepper_method_ab` computes `AB_SUMMARY`. The reported spread is therefore
//! the spread of the number rankings are quoted from.
//!
//! # What it found
//!
//! Measured 2026-08-10 (R43), 24 perturbation draws, two seeds, on `fa83508`:
//!
//! ```text
//!   arm     unperturbed RMS   floor min   floor p50   floor max   spread
//!   vern9          0.064004    0.022736    0.066602    0.135409   0.0942-0.0986
//!   vern7          0.130344    0.128466    0.129566    0.130745   0.0019-0.0022
//! ```
//!
//! **Vern9's corpus RMS is not a measurement.** Its floor spread is 147-153% of
//! its own value, so the number a ranking would quote is one draw of a
//! distribution that covers everything from 0.023 to 0.135 m. Vern7's is a
//! measurement: 1.6% spread.
//!
//! **The ranking quantity straddles zero.** `vern7_minus_vern9_rms_m` runs
//! -0.0059 to +0.1067. A physics-neutral perturbation of the initial state is
//! enough to INVERT the order of the two arms, so this corpus cannot say which
//! stepper is more accurate at the sealed tolerance. Both the R26 "accuracy is
//! a wash" reading and the post-epoch-fix "Vern7 is 2.04x worse" reading are
//! unsupported by it.
//!
//! # The anchor is NOT the cause — that hypothesis was tested and refuted
//!
//! The obvious explanation is that the harness anchors on Vern9@1e-12, so the
//! Vern9 arm is scored against its own method's converged limit while Vern7 is
//! scored across methods (`AB_ANCHOR_SEP` reads exactly 0.000000 for Vern9).
//! Re-running the whole measurement with `ANCHOR = Rkv98`, an independent
//! tableau, moves nothing: vern9 spread 0.0948/0.0977, vern7 spread
//! 0.0024/0.0021, gap still negative at its minimum (-0.0048). The instability
//! is Vern9@1e-8's error being small enough on these arcs to be dominated by
//! step-sequence decorrelation, which the perturbation randomizes completely.
//! A different anchor does not fix that, and neither would a tighter one.
//!
//! ```sh
//! cargo run --release -p lightyear_odeint_rs --example r43_corpus_floor -- [draws]
//! ```

use std::sync::Arc;

use anyhow::{Context, Result};
use lightyear_odeint_rs::integrator::{
    integrate_final_checked, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest,
};
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags, StepperMethod};
use satpy_core::{eci2equinoc_impl_f64, equinoc2eci_impl, kep2eci_impl, SEC_PER_DAY};

const DIR_R6_D15: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

const JD0: f64 = 2_460_310.5;
const TOF_S: f64 = 43_200.0;
const DRAWS: usize = 12;
const REFERENCE_EPS: f64 = 1.0e-12;
const ANCHOR: StepperMethod = StepperMethod::Vern9;
const ARMS: [StepperMethod; 2] = [StepperMethod::Vern9, StepperMethod::Vern7];

/// Perturbation draws. Default 24 rather than `v3_accuracy_floor`'s 48 because
/// each draw here is a whole corpus (12 arcs x 3 propagations, one of them at
/// 1e-12) rather than one arc. The corpus width is the axis that must not
/// shrink; the perturbation count is the one that may.
const FLOOR_DRAWS: usize = 24;
const FLOOR_SEED: u64 = 0x5EED_C025_0000_0001;
const FLOOR_SEED_B: u64 = 0x5EED_C025_0000_00B2;
const MAX_CALIBRATION_ULPS: u64 = 1 << 20;

const fn part_a_hybrid() -> &'static nd_config::PartAHybridControls {
    nd_config::CompiledPartAScienceV1::part_a_v1().hybrid()
}

fn production_force_config(method: StepperMethod, eps: f64) -> ForceConfig {
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
        eps,
        integrator_method: method,
        ..ForceConfig::default()
    }
}

struct Draw {
    init_equ: [f64; 6],
    /// Carried and used to build the propagation context. See the module
    /// header: handing a shared `JD0` here is the defect this file exists
    /// downstream of.
    epoch: f64,
    label: String,
}

/// The `stepper_method_ab` corpus, epoch-correct.
fn corpus() -> Vec<Draw> {
    let mut out = Vec::with_capacity(DRAWS);
    for index in 0..DRAWS {
        let step = f64::from(u32::try_from(index).unwrap_or(u32::MAX));
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
        out.push(Draw {
            init_equ,
            epoch,
            label: format!(
                "draw{index:02}_alt{:.0}_e{:.3}",
                kep.first().copied().unwrap_or_default() - 6_378.137,
                kep.get(1).copied().unwrap_or_default()
            ),
        });
    }
    out
}

/// Terminal ECI state of one arc. `None` when the arc has no endpoint.
fn endpoint(
    init_equ: [f64; 6],
    epoch: f64,
    method: StepperMethod,
    eps: f64,
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
) -> Result<Option<[f64; 6]>> {
    let config = production_force_config(method, eps)
        .with_ephemeris_for_arc(epoch, epoch + TOF_S / SEC_PER_DAY)
        .context("production ephemeris and JB2008 assets must cover the arc")?;
    let gravity = ScalarGravityAssets::new(Arc::clone(packed));
    let context = ScalarPropagationContext::new(epoch, Arc::new(config), gravity);
    let delta = match integrate_final_checked(
        ScalarPropagationRequest::new(&context, init_equ, &[TOF_S], 0.0, TOF_S).with_events(true),
    ) {
        Ok(delta) => delta,
        Err(failure) if failure.is_physical_infeasible() => return Ok(None),
        Err(failure) => return Err(anyhow::anyhow!("{failure:?}").context("arc must propagate")),
    };
    let mut state = [0.0; 6];
    equinoc2eci_impl(&init_equ, 6, TOF_S, 0.0, &mut state);
    for (component, delta_component) in state.iter_mut().zip(delta) {
        *component += delta_component;
    }
    Ok(Some(state))
}

fn separation_m(left: &[f64; 6], right: &[f64; 6]) -> f64 {
    let mut sum = 0.0;
    for (l, r) in left.iter().take(3).zip(right.iter().take(3)) {
        let d = l - r;
        sum += d * d;
    }
    sum.sqrt() * 1000.0
}

fn bit_identical(left: &[f64; 6], right: &[f64; 6]) -> bool {
    left.iter()
        .zip(right.iter())
        .all(|(l, r)| l.to_bits() == r.to_bits())
}

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
    fn next_in_range(&mut self, bound: u64) -> u64 {
        let span = bound.max(1);
        self.next_u64()
            .checked_rem(span)
            .unwrap_or(0)
            .wrapping_add(1)
    }
}

/// Smallest ULP count on one element that moves this arc's production endpoint.
fn calibrate_draw(
    draw: &Draw,
    baseline: &[f64; 6],
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
    sealed_eps: f64,
) -> Result<[u64; 6]> {
    let mut ulps = [1_u64; 6];
    let arm = ARMS[1];
    for index in 0..6 {
        let mut count = 1_u64;
        let found = loop {
            if count > MAX_CALIBRATION_ULPS {
                break None;
            }
            let probe = endpoint(
                nudge(draw.init_equ, index, count, true),
                draw.epoch,
                arm,
                sealed_eps,
                packed,
            )?;
            match probe {
                Some(state) if !bit_identical(&state, baseline) => break Some(count),
                _ => count = count.wrapping_mul(2),
            }
        };
        let Some(count) = found else {
            anyhow::bail!(
                "{} element {index} is INERT at {MAX_CALIBRATION_ULPS} ULP: it cannot contribute \
                 a floor sample, so this measurement is abandoned rather than reported with a \
                 dead axis",
                draw.label
            );
        };
        if let Some(slot) = ulps.get_mut(index) {
            *slot = count;
        }
    }
    println!("FLOOR_CAL {} ulps={ulps:?}", draw.label);
    Ok(ulps)
}

struct Spread {
    sorted: Vec<f64>,
}

impl Spread {
    fn new(mut values: Vec<f64>) -> Self {
        values.sort_by(f64::total_cmp);
        Self { sorted: values }
    }
    /// Nearest-rank order statistic.
    ///
    /// `numerator`/`denominator` rather than an `f64` quantile so the index is
    /// computed in integers: a float quantile needs a `f64 -> usize` cast that
    /// this workspace lints against, and rounding it is the one place an order
    /// statistic can silently pick the wrong sample.
    fn at(&self, numerator: usize, denominator: usize) -> f64 {
        if self.sorted.is_empty() {
            return f64::NAN;
        }
        let last = self.sorted.len().saturating_sub(1);
        let index = last
            .saturating_mul(numerator)
            .checked_div(denominator.max(1))
            .unwrap_or(0);
        self.sorted
            .get(index.min(last))
            .copied()
            .unwrap_or(f64::NAN)
    }
    fn print(&self, label: &str) {
        println!(
            "FLOOR {label} n={} min={:.6} p50={:.6} p90={:.6} max={:.6} spread={:.6}",
            self.sorted.len(),
            self.at(0, 1),
            self.at(1, 2),
            self.at(9, 10),
            self.at(1, 1),
            self.at(1, 1) - self.at(0, 1),
        );
    }
}

/// One corpus pass: per-arm RMS and max over the scored draws.
///
/// `baselines` is `Some` on a perturbed pass and carries each arc's
/// unperturbed endpoint, so a perturbation that failed to reach the trajectory
/// is a hard error rather than a silent zero-variance sample.
fn corpus_metric(
    draws: &[Draw],
    packed: &Arc<satpy_core::PackedGravityCoeffs>,
    baselines: Option<&[[f64; 6]]>,
    arm_eps: f64,
) -> Result<[(f64, f64); 2]> {
    let sealed_eps = arm_eps;
    let mut sums = [0.0_f64; 2];
    let mut maxima = [0.0_f64; 2];
    let mut scored = 0_usize;
    for (draw_index, draw) in draws.iter().enumerate() {
        let init = draw.init_equ;
        let Some(reference) = endpoint(init, draw.epoch, ANCHOR, REFERENCE_EPS, packed)? else {
            continue;
        };
        let mut arm_errors = [0.0_f64; 2];
        let mut all_present = true;
        for (arm_index, method) in ARMS.iter().enumerate() {
            let Some(state) = endpoint(init, draw.epoch, *method, sealed_eps, packed)? else {
                all_present = false;
                break;
            };
            if let Some(base) = baselines.and_then(|rows| rows.get(draw_index)) {
                anyhow::ensure!(
                    !bit_identical(&state, base),
                    "{} came back BIT IDENTICAL under perturbation: the calibrated ULP counts no \
                     longer reach the trajectory and this floor would be built from dead samples",
                    draw.label
                );
            }
            if let Some(slot) = arm_errors.get_mut(arm_index) {
                *slot = separation_m(&state, &reference);
            }
        }
        if !all_present {
            continue;
        }
        scored = scored.saturating_add(1);
        for arm_index in 0..2 {
            let error = arm_errors.get(arm_index).copied().unwrap_or_default();
            if let Some(sum) = sums.get_mut(arm_index) {
                *sum += error * error;
            }
            if let Some(max) = maxima.get_mut(arm_index) {
                *max = max.max(error);
            }
        }
    }
    anyhow::ensure!(scored > 0, "no draw scored");
    let n = f64::from(u32::try_from(scored).unwrap_or(u32::MAX));
    let mut out = [(0.0, 0.0); 2];
    for arm_index in 0..2 {
        if let Some(slot) = out.get_mut(arm_index) {
            *slot = (
                (sums.get(arm_index).copied().unwrap_or_default() / n).sqrt(),
                maxima.get(arm_index).copied().unwrap_or_default(),
            );
        }
    }
    Ok(out)
}

const fn arm_label(method: StepperMethod) -> &'static str {
    match method {
        StepperMethod::Vern7 => "vern7",
        StepperMethod::Vern9 => "vern9",
        _ => "other",
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let floor_draws: usize = args.next().map_or(Ok(FLOOR_DRAWS), |value| {
        value.parse().context("draws must parse as usize")
    })?;

    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("gravity coefficients must load")?;
    let packed =
        lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")?;

    // Second argument overrides the tolerance the ARMS fly. The anchor stays
    // at 1e-12 regardless: a candidate tolerance has to be scored against the
    // same converged reference the sealed one is, or the comparison is between
    // two different questions.
    let arm_eps: f64 = args.next().map_or(Ok(part_a_hybrid().tolerance), |value| {
        value.parse().context("eps must parse as f64")
    })?;
    println!(
        "FLOOR_SHAPE corpus_draws={DRAWS} floor_draws={floor_draws} eps={arm_eps:e} \
         sealed_eps={:e} anchor_eps={REFERENCE_EPS} atm_model={}",
        part_a_hybrid().tolerance,
        part_a_hybrid().atmosphere_model
    );

    let corpus = corpus();

    // Unperturbed pass: the numbers a ranking would be quoted from.
    let base_metric = corpus_metric(&corpus, &packed, None, arm_eps)?;
    for (arm_index, method) in ARMS.iter().enumerate() {
        let (rms, max) = base_metric.get(arm_index).copied().unwrap_or_default();
        println!(
            "FLOOR_BASE {} rms_m={rms:.6} max_m={max:.6}",
            arm_label(*method)
        );
    }

    // Per-draw calibration against each arc's own unperturbed endpoint.
    let mut calibrated = Vec::with_capacity(corpus.len());
    let mut baselines = Vec::with_capacity(corpus.len());
    for draw in &corpus {
        let Some(base) = endpoint(draw.init_equ, draw.epoch, ARMS[1], arm_eps, &packed)? else {
            anyhow::bail!("{} has no endpoint; the corpus screen changed", draw.label);
        };
        calibrated.push(calibrate_draw(draw, &base, &packed, arm_eps)?);
        baselines.push(base);
    }

    // Two seeds, because a spread from one draw set is a spread from one draw
    // set -- the same reason `v3_accuracy_floor` runs two.
    for (seed_label, seed) in [("A", FLOOR_SEED), ("B", FLOOR_SEED_B)] {
        let mut rng = SplitMix64::new(seed);
        let mut vern9_rms = Vec::with_capacity(floor_draws);
        let mut vern7_rms = Vec::with_capacity(floor_draws);
        let mut gap = Vec::with_capacity(floor_draws);
        for _ in 0..floor_draws {
            let perturbed: Vec<Draw> = corpus
                .iter()
                .enumerate()
                .map(|(index, draw)| {
                    let mut init = draw.init_equ;
                    if let Some(bounds) = calibrated.get(index) {
                        for (element, bound) in bounds.iter().enumerate() {
                            let magnitude = rng.next_in_range(*bound);
                            let up = rng.next_u64() & 1 == 0;
                            init = nudge(init, element, magnitude, up);
                        }
                    }
                    Draw {
                        init_equ: init,
                        epoch: draw.epoch,
                        label: draw.label.clone(),
                    }
                })
                .collect();
            let metric = corpus_metric(&perturbed, &packed, Some(&baselines), arm_eps)?;
            let v9 = metric.first().map_or(f64::NAN, |entry| entry.0);
            let v7 = metric.get(1).map_or(f64::NAN, |entry| entry.0);
            vern9_rms.push(v9);
            vern7_rms.push(v7);
            gap.push(v7 - v9);
        }
        Spread::new(vern9_rms).print(&format!("seed{seed_label} vern9_rms_m"));
        Spread::new(vern7_rms).print(&format!("seed{seed_label} vern7_rms_m"));
        // The ranking quantity. A gap distribution straddling zero means the
        // corpus cannot order the two arms at all.
        Spread::new(gap).print(&format!("seed{seed_label} vern7_minus_vern9_rms_m"));
    }
    Ok(())
}
