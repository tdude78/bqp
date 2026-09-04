//! Historical model-4-through-7 quadrature research on one pinned strict-HF arc.
//!
//! This tool does not include current Part A atmosphere model 8. Every output
//! is stamped historical and must not be used as current campaign authority or
//! compared directly with a current-authority timing at a different epoch.
//!
//! # Why this exists
//!
//! Historical perf instruments pinned `atm_model: 4` --- `libm_budget.rs`, a
//! now-retired standalone timing example, and `strict_hf_pin.rs`. (This said FOUR instruments
//! until 2026-08-21; `wallclock_profile.rs` was retired 2026-08-20 with the
//! `wallclock-profile` feature, see `two_phase_transfer_rs/Cargo.toml`. A
//! stated count is a claim, and this one outlived the set it counted.) The
//! campaign moved to `atmosphere_model: 5` at 2df59d4
//! (`nd_config::part_a_science` / `StrictHfForceAuthority::PART_A`), and model 5
//! is the same JB2008 kernel with the three Boole log steps 4x coarser
//! (0.010/0.025/0.075 -> 0.040/0.100/0.300). Every quadrature-cost figure taken
//! on model 4 therefore describes roughly 4x more work per kernel call than the
//! campaign performed in that era, and cannot be composed into a current
//! model-8 cell-level number.
//!
//! This reports, for historical models 4 through 7 on the same arc and build:
//! evaluations, JB2008 kernel calls, ns per RHS evaluation, and the `jb_local_temp`
//! / `atan` call counts implied by the quadrature's own step formulas.
//!
//! `cargo run --release -p lightyear_odeint_rs --example quadrature_profile_arc_ab`

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use lightyear_odeint_rs::integrator::{
    integrate_final_checked, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest,
};
use lightyear_odeint_rs::probe;
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags, StepperMethod};
use satpy_core::{eci2equinoc_impl_f64, kep2eci_impl, SEC_PER_DAY};

const JD0: f64 = 2_460_310.5;
const TOF_S: f64 = 43_200.0;
const HISTORICAL_EPS: f64 = 1.0e-8;

/// The compiled stepper, resolved rather than restated.
///
/// This file hardcoded `StepperMethod::Vern9` through the R26 swap, so the
/// first m6-vs-m7 arc number taken here priced a stepper the campaign had
/// stopped flying. Same defect `prop_timing::authority_stepper` was written to
/// close, and the same shape of fix: resolve the token, and fail closed on a
/// stepper this file cannot build rather than quietly measure another one.
#[expect(
    clippy::panic,
    reason = "a stepper this file cannot build must stop the run, not silently measure a different one"
)]
fn authority_stepper() -> StepperMethod {
    match nd_config::CompiledPartAScienceV1::part_a_v1()
        .hybrid()
        .integrator_method
    {
        "vern7" => StepperMethod::Vern7,
        "vern9" => StepperMethod::Vern9,
        other => panic!("compiled science selects a stepper this file does not build: {other}"),
    }
}

/// Verbatim from `examples/libm_budget.rs`, with `atm_model` lifted to a parameter.
fn dust_config(atm_model: i32) -> ForceConfig {
    ForceConfig {
        sph_order: 5,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        subtract_first_order: true,
        atm_model,
        am_ratio: 1.948,
        cd: 2.2,
        cr: 1.3,
        target_propagation_mode: 0,
        dt_max: 300.0,
        eps: HISTORICAL_EPS,
        integrator_method: authority_stepper(),
        ..ForceConfig::default()
    }
}

fn run_arc(atm_model: i32) -> Result<(u64, u64, u64)> {
    let packed =
        lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")?;

    let kep = [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0];
    let mut init_eci = [0.0; 6];
    kep2eci_impl(&kep, true, 0.0, 0.0, true, &mut init_eci);
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let config = dust_config(atm_model)
        .with_ephemeris_for_arc(JD0, JD0 + TOF_S / SEC_PER_DAY)
        .context("production ephemeris and JB2008 assets must cover the pinned arc")?;

    probe::reset().context("propagation census must reset")?;
    let gravity = ScalarGravityAssets::new(packed);
    let context = ScalarPropagationContext::new(JD0, Arc::new(config), gravity);
    let delta = integrate_final_checked(
        ScalarPropagationRequest::new(&context, init_equ, &[TOF_S], 0.0, TOF_S).with_events(true),
    )
    .context("the pinned strict-HF arc must propagate")?;
    black_box(&delta);

    let evals = probe::snapshot().iter().map(|e| e.rhs_evals).sum::<u64>();
    let (adapter, kernel) = probe::jb_call_census();
    Ok((evals, adapter, kernel))
}

fn usize_as_f64(value: usize) -> Result<f64> {
    value
        .to_string()
        .parse()
        .context("profile count must parse as f64")
}

fn u64_as_f64(value: u64) -> Result<f64> {
    value
        .to_string()
        .parse()
        .context("profile count must parse as f64")
}

fn segment_count(log_span: f64, step: f64) -> Result<usize> {
    ensure!(
        log_span.is_finite() && step.is_finite() && step > 0.0,
        "quadrature segment inputs must be finite with a positive step"
    );
    let floored = (log_span / step).floor();
    ensure!(
        floored.is_finite() && floored >= 0.0,
        "quadrature segment count must be finite and nonnegative"
    );
    let count = floored
        .to_string()
        .parse::<usize>()
        .context("quadrature segment count must fit usize")?;
    count
        .checked_add(1)
        .context("quadrature segment count overflowed")
}

/// `jb_local_temp` calls and `atan`-arm hits per kernel call, from the
/// quadrature's own step formulas. Verbatim structure from
/// `libm_budget::atan_call_count`, with the three log steps lifted to
/// parameters so both profiles can be counted.
fn temp_call_count(
    altitude_km: f64,
    lower: f64,
    middle: f64,
    upper: f64,
) -> Result<(usize, usize)> {
    let mut total = 0usize;
    let mut with_atan = 0usize;

    let mut count_segment = |z_start: f64, z_end: f64, n: usize| -> Result<f64> {
        let zr = ((z_end / z_start).ln() / usize_as_f64(n)?).exp();
        let mut zend = z_start;
        for _ in 0..n {
            let z0 = zend;
            zend = zr * z0;
            let dz = 0.25 * (zend - z0);
            let mut z = z0;
            for _ in 0..4 {
                z += dz;
                total = total
                    .checked_add(1)
                    .context("temperature call count overflowed")?;
                if z - 125.0 > 0.0 {
                    with_atan = with_atan
                        .checked_add(1)
                        .context("atan call count overflowed")?;
                }
            }
        }
        Ok(zend)
    };

    let z1 = 90.0;
    let z2 = altitude_km.min(105.0);
    let n1 = segment_count((z2 / z1).ln(), lower)?;
    let z_after_1 = count_segment(z1, z2, n1)?;
    if altitude_km <= 105.0 {
        return Ok((total, with_atan));
    }

    let al = (altitude_km.min(500.0) / z_after_1).ln();
    let n2 = segment_count(al, middle)?;
    let z_after_2 = count_segment(z_after_1, altitude_km.min(500.0), n2)?;

    let al = (altitude_km.max(500.0) / z_after_2).ln();
    let r = if altitude_km > 500.0 { upper } else { middle };
    let n3 = segment_count(al, r)?;
    count_segment(z_after_2, altitude_km.max(500.0), n3)?;

    Ok((total, with_atan))
}

/// Every profile this program prices, in the order its output slots run.
const MODELS: [i32; 4] = [4, 5, 6, 7];

fn main() -> Result<()> {
    println!(
        "QUADRATURE_PROFILE_IDENTITY historical=true current_part_a_authority=false \
         models=4,5,6,7 epoch_bits={:#018x} stepper={} note=model8-not-profiled",
        JD0.to_bits(),
        nd_config::CompiledPartAScienceV1::part_a_v1()
            .hybrid()
            .integrator_method
    );
    lightyear_odeint_rs::load_constants_from_bytes(
        include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt"),
        5,
    )
    .context("gravity coefficients must load")?;

    // Untimed: loads ephemeris/JB2008 assets and warms every cache, on all models.
    for model in MODELS {
        let _ = run_arc(model)?;
    }

    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);

    let mut per_eval = [0.0_f64; MODELS.len()];
    let mut per_arc = [0.0_f64; MODELS.len()];
    let mut evals_of = [0_u64; MODELS.len()];
    let mut kernel_of = [0_u64; MODELS.len()];
    // Min-of-reps beside the mean, because the mean alone is not usable on a
    // contended host and this box is routinely at load 100+.
    //
    // Interleaving (below) removes DRIFT -- a trend that would otherwise be
    // charged to whichever arm held the machine -- but it does not remove
    // SATURATION bias: every arm is slowed, and a mean over slowed reps
    // compresses the ratio between arms toward 1.0. The minimum is the rep
    // that came closest to running alone, so it is the statistic that survives
    // outside load. This mirrors `prop_timing`'s min-of-block rule, and both
    // are reported so a run whose two statistics disagree announces its own
    // contention instead of quietly shipping the compressed number.
    let mut min_arc = [f64::INFINITY; MODELS.len()];
    // TRULY interleaved: one arc of every model per rep, models innermost.
    //
    // This loop used to run all `reps` of model 4, then all of model 5, and
    // called itself interleaved in a comment. It was not: each model occupied
    // one contiguous block of wall time, so any thermal ramp or change in host
    // tenancy landed on whichever arm happened to hold the machine at the time,
    // and the effect is charged entirely to the model. That is the loop-order
    // confound, and on this host the blocks are seconds long -- easily enough
    // for it to bite. Rotating the models inside the rep bounds the drift any
    // one arm can absorb to a single arc.
    for _ in 0..reps {
        for (slot, model) in MODELS.into_iter().enumerate() {
            let t0 = Instant::now();
            let (evals, _atan, kernel) = run_arc(model)?;
            let elapsed = t0.elapsed().as_secs_f64();
            let (Some(arc_slot), Some(evals_slot), Some(kernel_slot), Some(min_slot)) = (
                per_arc.get_mut(slot),
                evals_of.get_mut(slot),
                kernel_of.get_mut(slot),
                min_arc.get_mut(slot),
            ) else {
                anyhow::bail!("model output slot must fit profile array");
            };
            *min_slot = min_slot.min(elapsed);
            *arc_slot += elapsed;
            *evals_slot = evals;
            *kernel_slot = kernel;
        }
    }
    for (slot, model) in MODELS.into_iter().enumerate() {
        let (Some(arc_slot), Some(&evals), Some(&kernel), Some(eval_slot)) = (
            per_arc.get_mut(slot),
            evals_of.get(slot),
            kernel_of.get(slot),
            per_eval.get_mut(slot),
        ) else {
            anyhow::bail!("model output slot must fit profile array");
        };
        *arc_slot /= usize_as_f64(reps)?;
        let per_arc_s = *arc_slot;
        let ns_per_eval = per_arc_s * 1.0e9 / u64_as_f64(evals)?;
        *eval_slot = ns_per_eval;
        println!(
            "HISTORICAL_MODEL {model}  rhs_evals={evals}  jb_kernel_calls={kernel}  kernel/eval={:.4}  \
             {per_arc_s:.6} s/arc  {ns_per_eval:.2} ns/eval",
            u64_as_f64(kernel)? / u64_as_f64(evals)?
        );
    }
    let model4_per_eval = *per_eval
        .first()
        .context("model 4 profile output must be present")?;
    let model5_per_eval = *per_eval
        .get(1)
        .context("model 5 profile output must be present")?;
    let model6_per_eval = *per_eval
        .get(2)
        .context("model 6 profile output must be present")?;
    println!(
        "HISTORICAL_ARC_COST_RATIO model4/model5 = {:.4}  model4/model6 = {:.4}  model5/model6 = {:.4}",
        model4_per_eval / model5_per_eval,
        model4_per_eval / model6_per_eval,
        model5_per_eval / model6_per_eval
    );
    // The campaign-relevant number is WHOLE ARC, not per evaluation: a coarser
    // quadrature also changes the evaluation count, and R16 measured that half
    // swinging +-3% with random sign. Quoting the per-eval saving alone
    // overstates or understates the win by up to a factor of two.
    let model5_arc = *per_arc.get(1).context("model 5 arc wall must be present")?;
    let model6_arc = *per_arc.get(2).context("model 6 arc wall must be present")?;
    let model7_arc = *per_arc.get(3).context("model 7 arc wall must be present")?;
    println!(
        "HISTORICAL_WHOLE_ARC_WALL model5 = {model5_arc:.6} s  model6 = {model6_arc:.6} s  \
         model6 vs model5 = {:+.2}%",
        (model6_arc - model5_arc) / model5_arc * 100.0
    );
    // The model 7 landing's headline. Model 6 is the incumbent, so model 6 is
    // the denominator; quoting model 7 against model 5 or 4 would fold in a
    // saving that already shipped.
    println!(
        "HISTORICAL_WHOLE_ARC_WALL model6 = {model6_arc:.6} s  model7 = {model7_arc:.6} s  \
         model7 vs model6 = {:+.2}%",
        (model7_arc - model6_arc) / model6_arc * 100.0
    );
    // The same headline on the contention-robust statistic. Quote this one when
    // the run was not taken on a quiet host; quote both when reporting.
    let model6_min = *min_arc
        .get(2)
        .context("model 6 min arc wall must be present")?;
    let model7_min = *min_arc
        .get(3)
        .context("model 7 min arc wall must be present")?;
    println!(
        "HISTORICAL_WHOLE_ARC_MIN model6 = {model6_min:.6} s  model7 = {model7_min:.6} s  \
         model7 vs model6 = {:+.2}%  (min-of-{reps}, contention-robust)",
        (model7_min - model6_min) / model6_min * 100.0
    );

    // Model 7 is deliberately absent from the table below. It uses model 6's
    // log steps exactly, so every count in it would be model 6's count
    // repeated. Its saving is not fewer quadrature steps -- it is not walking
    // the two fixed plans at all -- so a step-count table is the wrong
    // instrument for it and a duplicated column would imply otherwise.
    println!("HISTORICAL_JB_CALLS_PER_KERNEL from-model4-through-7-step-formulas:");
    for alt in [200.0, 400.0, 620.0, 800.0, 980.0] {
        let (t4, a4) = temp_call_count(alt, 0.010, 0.025, 0.075)?;
        let (t5, a5) = temp_call_count(alt, 0.040, 0.100, 0.300)?;
        let (t6, a6) = temp_call_count(alt, 0.040, 0.300, 0.700)?;
        println!(
            "HISTORICAL_ROW alt={alt:>6.0}_km model4_temp={t4:>4} model4_atan={a4:>4}   \
             model5 temp={t5:>4} atan={a5:>4}   model6 temp={t6:>4} atan={a6:>4}   \
             atan m4/m6 {:.3}x  m5/m6 {:.3}x",
            usize_as_f64(a4)? / usize_as_f64(a6)?,
            usize_as_f64(a5)? / usize_as_f64(a6)?
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{segment_count, temp_call_count};

    #[test]
    fn validated_quadrature_count_keeps_the_zero_span_first_segment() -> anyhow::Result<()> {
        assert_eq!(segment_count(0.0, 0.010)?, 1);
        assert_eq!(temp_call_count(90.0, 0.010, 0.025, 0.075)?, (4, 0));
        assert!(segment_count(f64::NAN, 0.010).is_err());
        Ok(())
    }
}
