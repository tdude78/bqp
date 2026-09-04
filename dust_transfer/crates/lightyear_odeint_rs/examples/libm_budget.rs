//! Measures the libm budget of the production strict-HF RHS.
//!
//! Three independent numbers, none of them from a sampling profiler:
//!
//! 1. **Call counts** — JB2008 adapter and kernel calls per RHS evaluation on
//!    the pinned production arc, from `probe::jb_call_census`.
//! 2. **Arc cost** — wall nanoseconds per RHS evaluation, from repeated runs of
//!    the same arc.
//! 3. **Unit cost** — nanoseconds per call for each transcendental, measured on
//!    this machine and this build, with the operand fed forward so the chain is
//!    latency-bound and cannot be hoisted.
//!
//! (1) x (3) / (2) is the share of the RHS attributable to each routine, with
//! no attribution step and therefore no attribution error. It is an upper
//! bound on what removing that routine could buy, because it charges full
//! serial latency to every call.
//!
//! # EVERY SHARE HERE IS PER ATMOSPHERE MODEL, AND THE DEFAULT MOVED
//!
//! Until 2026-08-07 this program hardcoded `atm_model: 4`, the exact JB2008
//! profile, while compiled science had already moved past it. It now follows
//! the authority by default and prints the model on every `ARC` line.
//! **Figures recorded before that date are model-4 figures** and reproduce
//! only with `LIBM_BUDGET_ATM_MODEL=4`. The seal has since moved again, to
//! `atmosphere_model: 7` (R31's fitted kernel), so figures labelled m5 or m6
//! do not reproduce on this tree either.
//!
//! The distinction bites hardest in this program of anywhere in the tree,
//! because the quantity it attributes is exactly the one the profiles disagree
//! about: `ExactOrekitQuadrature` (model 4) runs a 63-step middle quadrature
//! plan where the log-quadrature approximations run 16, so model 4 issues
//! roughly four times the `atan` traffic per kernel call. A transcendental
//! share taken on model 4 is not an upper bound on the same share on the flown
//! model; it is a different number about a different arc.
//!
//! The same trap in a second form was live here until 2026-08-11: every other
//! field came from the seal while `integrator_method` stayed a `Vern9`
//! literal. `authority_stepper` below closes it, as its two siblings already
//! did elsewhere. Figures recorded before that date are **Vern9** figures, and
//! the seal flies Vern7.
//!
//! ```sh
//! cargo run --release -p lightyear_odeint_rs --example libm_budget
//! LIBM_BUDGET_ATM_MODEL=4 cargo run --release -p lightyear_odeint_rs \
//!     --example libm_budget
//! ```

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use lightyear_odeint_rs::integrator::{
    integrate_final_checked, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest,
};
use lightyear_odeint_rs::probe;
use lightyear_odeint_rs::types::StepperMethod;
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags};
use satpy_core::{eci2equinoc_impl_f64, kep2eci_impl, SEC_PER_DAY};

const JD0: f64 = 2_460_310.5;
const TOF_S: f64 = 43_200.0;

/// The compiled Part A science authority, read rather than restated.
const fn part_a_hybrid() -> &'static nd_config::PartAHybridControls {
    nd_config::CompiledPartAScienceV1::part_a_v1().hybrid()
}

/// The stepper the compiled science authority selects, resolved not restated.
///
/// Identical in purpose to `prop_timing::authority_stepper` and
/// `quadrature_profile_arc_ab::authority_stepper`. Those two were introduced
/// when the seal moved Vern9 -> Vern7; this file was the one copy of the
/// pattern that was missed, and it went on reading every other field from the
/// authority while holding `StepperMethod::Vern9` as a literal. A libm budget
/// is a per-stage count, so a harness on the wrong stepper does not merely
/// mis-time -- it attributes the wrong number of stages' worth of calls.
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

/// The strict-HF force config, with the atmosphere model chosen by the caller.
///
/// This was a copy of `strict_hf_pin::production_dust_config` and inherited its
/// defect: `atm_model` was hardcoded to 4 while compiled science had moved on,
/// so every per-routine share this program printed was a share of an arc the
/// campaign does not fly. It matters most HERE of anywhere, because model 4's
/// `ExactOrekitQuadrature` runs a 63-step middle plan against the fitted
/// kernel's and so issues far more `atan` traffic per kernel call -- the exact
/// quantity this program attributes.
///
/// The model now comes from `resolve_atm_model` and is printed on every `ARC`
/// line, so no recorded figure from this program can lose its corpus again.
/// `eps` and `integrator_method` were the two fields that stayed behind when
/// that repair was made, and they are resolved from the authority here too.
fn dust_config(atm_model: i32) -> ForceConfig {
    let controls = part_a_hybrid();
    ForceConfig {
        sph_order: controls.gravity_order,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        subtract_first_order: true,
        atm_model,
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

/// `LIBM_BUDGET_ATM_MODEL=4` reruns this program on the exact profile, which is
/// what every figure recorded before 2026-08-07 was measured on. Unset, it
/// follows the authority — whatever `atmosphere_model` the seal currently
/// reads, which is what the campaign flies.
fn resolve_atm_model() -> Result<i32> {
    std::env::var("LIBM_BUDGET_ATM_MODEL").map_or_else(
        |_| Ok(part_a_hybrid().atmosphere_model),
        |raw| {
            raw.trim()
                .parse()
                .with_context(|| format!("LIBM_BUDGET_ATM_MODEL must be an integer, got {raw:?}"))
        },
    )
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

/// Throughput unit cost over a table of REALISTIC operands.
///
/// The earlier form fed each result into the next call. That converges to a
/// fixed point within a few iterations — `atan2` and `asin` both settled on
/// exactly 0.0 and `powf` diverged to `inf` — so it timed a special-case
/// operand, not the routine. Driving from a table of spread operands and
/// accumulating the results keeps every call on the generic path, and matches
/// how the RHS actually calls these: independent, pipelineable, not serial.
#[expect(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "the timing calculation intentionally retains its established call-count and nanosecond arithmetic"
)]
fn unit_ns(name: &str, xs: &[f64], reps: u64, f: impl Fn(f64) -> f64) -> f64 {
    let mut acc = 0.0_f64;
    for _ in 0..64 {
        for &x in xs {
            acc += f(black_box(x));
        }
    }
    black_box(acc);

    let rounds = reps / xs.len() as u64;
    let mut acc = 0.0_f64;
    let t0 = Instant::now();
    for _ in 0..rounds {
        for &x in xs {
            acc += f(black_box(x));
        }
    }
    let dt = t0.elapsed();
    black_box(acc);
    let calls = rounds * xs.len() as u64;
    let ns = dt.as_secs_f64() * 1.0e9 / calls as f64;
    println!("  {name:<12} {ns:>8.3} ns/call   (acc {acc:.6e}, {calls} calls)");
    ns
}

/// Candidate replacement for the single `atan` at `jb_rs/src/jb2008.rs:504`.
///
/// That call site is the only one that matters and it has a property the
/// general routine cannot assume: **its argument is always strictly positive**
/// (`u = tc[3] * dz * (1 + 4.5e-6 dz^2.5)` with `dz > 0` on the branch that
/// reaches it). Dropping sign handling and the negative half-line removes the
/// branchy front end of a general `atan`.
///
/// Cephes range reduction, degree-4/5 minimax rational core.
///
/// `excessive_precision` fires on every constant here and its suggestion is
/// wrong for the same reason it is wrong in `strict_hf_pin`: these are the
/// published minimax coefficients, and rounding any of them changes the ULP
/// bound this probe exists to measure.
#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::float_cmp,
    reason = "published minimax literals and non-FMA reduction define the candidate accuracy measurement"
)]
#[inline]
fn fast_atan_pos(x: f64) -> f64 {
    const P: [f64; 5] = [
        -8.750608600031904122785e-1,
        -1.615753718733365076637e1,
        -7.500855792314704667340e1,
        -1.228866684490136173410e2,
        -6.485021904942025371773e1,
    ];
    const Q: [f64; 5] = [
        2.485846490142306297962e1,
        1.650270098316988542046e2,
        4.328810604912902668951e2,
        4.853903996359136964868e2,
        1.945506571482613964425e2,
    ];
    const MOREBITS: f64 = 6.123233995736765886130e-17;
    const T3P8: f64 = 2.414213562373095048802e0;
    const TP8: f64 = 4.142135623730950488017e-1;

    let (mut x, y) = if x > T3P8 {
        (-1.0 / x, std::f64::consts::FRAC_PI_2)
    } else if x > TP8 {
        ((x - 1.0) / (x + 1.0), std::f64::consts::FRAC_PI_4)
    } else {
        (x, 0.0)
    };
    let z = x * x;
    let num = ((((P[0] * z + P[1]) * z + P[2]) * z + P[3]) * z + P[4]) * z;
    let den = ((((z + Q[0]) * z + Q[1]) * z + Q[2]) * z + Q[3]) * z + Q[4];
    x += x * (num / den);
    let corr = if y == std::f64::consts::FRAC_PI_2 {
        MOREBITS
    } else if y == std::f64::consts::FRAC_PI_4 {
        0.5 * MOREBITS
    } else {
        0.0
    };
    y + corr + x
}

/// SHAPE PROBES. These do not compute `atan` correctly and are never wired
/// anywhere; the coefficients are arbitrary. They exist to price the *shape* of
/// a candidate before any effort goes into fitting real coefficients, by
/// isolating what the two divisions in `fast_atan_pos` cost against the
/// polynomial work.
///
/// `shape_recip_poly` — one reciprocal for the `x > 1` fold, then a degree-13
/// Horner polynomial in `z = x*x`. This is the cheapest shape that could still
/// reach double accuracy over the whole positive line.
#[expect(
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    reason = "this shape-only microbenchmark keeps compact polynomial notation and non-FMA operation order"
)]
#[inline]
fn shape_recip_poly(x: f64) -> f64 {
    const C: [f64; 14] = [
        -7.6e-2,
        8.3e-2,
        -9.0e-2,
        9.8e-2,
        -1.08e-1,
        1.2e-1,
        -1.36e-1,
        1.538e-1,
        -1.818e-1,
        2.222e-1,
        -2.857e-1,
        4.0e-1,
        -6.666_666_666_666_666e-1,
        1.0,
    ];
    let (r, y, s) = if x > 1.0 {
        (1.0 / x, std::f64::consts::FRAC_PI_2, -1.0)
    } else {
        (x, 0.0, 1.0)
    };
    let z = r * r;
    let mut p = C[0];
    for c in &C[1..] {
        p = p * z + c;
    }
    y + s * r * p
}

/// `shape_poly_only` — the same polynomial with NO division at all. Not a
/// usable `atan` (it diverges for large arguments); it prices the division.
#[inline]
fn shape_poly_only(x: f64) -> f64 {
    const C: [f64; 14] = [
        -7.6e-2,
        8.3e-2,
        -9.0e-2,
        9.8e-2,
        -1.08e-1,
        1.2e-1,
        -1.36e-1,
        1.538e-1,
        -1.818e-1,
        2.222e-1,
        -2.857e-1,
        4.0e-1,
        -6.666_666_666_666_666e-1,
        1.0,
    ];
    let z = x * x;
    let mut p = C[0];
    for c in &C[1..] {
        p = p * z + c;
    }
    x * p
}

/// Max ULP error of `fast_atan_pos` against libm over the operand range the
/// JB2008 temperature profile actually produces, plus its unit cost.
#[expect(
    clippy::many_single_char_names,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    reason = "the ULP sweep keeps its compact mathematical variable names and fixed sample-count conversion"
)]
fn atan_candidate_report() {
    let mut worst_ulp = 0.0_f64;
    let mut worst_x = 0.0_f64;
    let mut worst_rel = 0.0_f64;
    // `u` spans roughly (0, 3000] on the pinned arc: `dz` runs 0 to ~855 km and
    // `tc[3]` is O(3.5e-2). Sample log-uniformly so the small-argument region,
    // where relative error is hardest, is not under-represented.
    let n = 4_000_000;
    for i in 0..n {
        let f = i as f64 / n as f64;
        let x = 1.0e-6 * (3.0e9_f64).powf(f); // 1e-6 .. 3e3
        let a = fast_atan_pos(x);
        let b = x.atan();
        let ulp = (a - b).abs() / (b.abs() * f64::EPSILON * 0.5).max(f64::MIN_POSITIVE);
        let rel = ((a - b) / b).abs();
        if ulp > worst_ulp {
            worst_ulp = ulp;
            worst_x = x;
        }
        worst_rel = worst_rel.max(rel);
    }
    println!(
        "FAST-ATAN  max {:.2} ULP, max rel {:.3e}, worst at x={:.6e}  ({} samples over 1e-6..3e3)",
        worst_ulp, worst_rel, worst_x, n
    );
}

/// How much a relative error in the temperature profile is amplified into
/// density.
///
/// Any faster `atan` perturbs `T(z) = tsubx + tc[2] atan(u)` by a relative
/// amount of order its ULP error. Density is `exp(-integral of g mbar / (R T))`,
/// so that error is amplified by roughly the magnitude of the exponent. Rather
/// than assume the amplification, measure it: perturb the temperature driver by
/// a known relative step and read the resulting relative density change.
///
/// The reported amplification times the routine's relative error is the
/// worst-case (fully coherent) density error the substitution can cause.
#[expect(
    clippy::unreadable_literal,
    clippy::expect_used,
    reason = "the diagnostic uses captured full-precision inputs and requires valid density evaluations"
)]
fn density_temperature_sensitivity() {
    use jb_rs::jb2008::{jb2008_density, Jb2008Input};

    let base = |alt_km: f64, dtc: f64| Jb2008Input {
        mjd_utc: 52_951.003805740744,
        // The sealed Orekit pair, differenced: sat_ra 1.28211886851503 minus
        // sun_ra 3.046653643566772.
        hour_angle_rad: 1.28211886851503 - 3.046653643566772,
        sun_declination_rad: -0.285987757544287,
        sat_geocentric_lat_rad: -1.4877186543999,
        sat_altitude_m: alt_km * 1000.0,
        f10: 91.00,
        f10b: 137.10,
        s10: 108.80,
        s10b: 123.80,
        m10: 116.70,
        m10b: 128.50,
        y10: 168.00,
        y10b: 138.60,
        dst_temperature_correction_k: dtc,
    };

    println!("DENSITY SENSITIVITY TO A TEMPERATURE-PROFILE ERROR:");
    // A 1e-6 relative step on the temperature correction, which enters `tinf`
    // additively exactly as an `atan` error does.
    let dtc0 = 43.0_f64;
    let step = 1.0e-6;
    for alt in [200.0, 400.0, 620.0, 800.0, 980.0] {
        let r0 = jb2008_density(base(alt, dtc0)).expect("density");
        let r1 = jb2008_density(base(alt, dtc0 * (1.0 + step))).expect("density");
        // Relative temperature change actually produced at the top of the
        // profile is `dtc0 * step / tinf`; approximate `tinf` by the driver sum
        // is not needed — use the driver's own relative step as the input.
        let d_ln_rho = ((r1 - r0) / r0).abs();
        // `dtc` enters `tinf` additively in Kelvin, so the absolute temperature
        // perturbation is known exactly and the sensitivity can be quoted per
        // Kelvin without needing `tinf` itself.
        let dt_kelvin = dtc0 * step;
        println!(
            "  alt {:>6.0} km  rho={:.6e}  d(ln rho)/dT = {:.4e} per K",
            alt,
            r0,
            d_ln_rho / dt_kelvin
        );
    }
}

/// Exact `atan` abscissa count per JB2008 kernel call, from the quadrature's own
/// step formulas rather than from a timer.
///
/// `jb_local_temp` takes the `atan` branch only for `z > 125 km`; below that it
/// is a polynomial. Counting the branch exactly is what reconciles the sampling
/// share against the unit cost.
///
/// # The step sizes are per PROFILE, and this used to hardcode the wrong ones
///
/// The three constants below are the quadrature log-step sizes, and the two
/// profiles do not share them: the exact Orekit profile walks 0.010/0.025/0.075
/// where the log-quadrature approximation the campaign flies walks
/// 0.040/0.100/0.300, four times coarser, for about a quarter of the abscissae.
/// This function hardcoded the exact profile's three values while `main` had
/// moved to model 5, so every count it produced was roughly 4x the count of the
/// arc being timed -- and `atan_call_count_report` then multiplied that inflated
/// count by a SCALAR `atan` price, for a routine that has been `f64x4` since
/// `bd06c0e`. Two errors, both upward, and the product was printed as a share of
/// a model-5 evaluation: 68-79%, against a directly measured 4.5-10%.
///
/// The steps are now passed in, and the report names the profile they came
/// from.
#[expect(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "this reproduces the JB2008 quadrature's exact count and endpoint arithmetic"
)]
fn atan_call_count(altitude_km: f64, steps: QuadratureSteps) -> (usize, usize) {
    let QuadratureSteps {
        lower,
        middle,
        upper,
        ..
    } = steps;
    let mut total = 0usize;
    let mut with_atan = 0usize;

    let mut count_segment = |z_start: f64, z_end: f64, n: usize| {
        let zr = ((z_end / z_start).ln() / n as f64).exp();
        let mut zend = z_start;
        for _ in 0..n {
            let z0 = zend;
            zend = zr * z0;
            let dz = 0.25 * (zend - z0);
            let mut z = z0;
            for _ in 0..4 {
                z += dz;
                total += 1;
                if z - 125.0 > 0.0 {
                    with_atan += 1;
                }
            }
        }
        zend
    };

    // Segment 1: 90 km -> min(alt, 105) km.
    let z1 = 90.0;
    let z2 = altitude_km.min(105.0);
    let n1 = ((z2 / z1).ln() / lower).floor() as usize + 1;
    let z_after_1 = count_segment(z1, z2, n1);
    if altitude_km <= 105.0 {
        return (total, with_atan);
    }

    // Segment 2: -> min(alt, 500) km.
    let al = (altitude_km.min(500.0) / z_after_1).ln();
    let n2 = 1 + (al / middle).floor() as usize;
    let z_after_2 = count_segment(z_after_1, altitude_km.min(500.0), n2);

    // Segment 3: -> alt, at the upper step size only above 500 km.
    let al = (altitude_km.max(500.0) / z_after_2).ln();
    let r = if altitude_km > 500.0 { upper } else { middle };
    let n3 = 1 + (al / r).floor() as usize;
    count_segment(z_after_2, altitude_km.max(500.0), n3);

    (total, with_atan)
}

/// The quadrature log-step sizes of one profile, and its name.
#[derive(Clone, Copy)]
struct QuadratureSteps {
    name: &'static str,
    lower: f64,
    middle: f64,
    upper: f64,
}

/// `ExactOrekitQuadrature` and `LogQuadratureX4ApproxV1`, mirroring
/// `jb_rs::jb2008`. Selected by `atm_model`, never assumed.
const fn quadrature_steps(atm_model: i32) -> QuadratureSteps {
    match atm_model {
        4 => QuadratureSteps {
            name: "ExactOrekitQuadrature (atm_model 4)",
            lower: 0.010,
            middle: 0.025,
            upper: 0.075,
        },
        5 => QuadratureSteps {
            name: "LogQuadratureX4ApproxV1 (atm_model 5)",
            lower: 0.040,
            middle: 0.100,
            upper: 0.300,
        },
        6 => QuadratureSteps {
            name: "LogQuadratureX4ApproxV2 (atm_model 6)",
            lower: 0.040,
            middle: 0.300,
            upper: 0.700,
        },
        // Model 7 shares model 6's log steps exactly; the fit replaces the two
        // FIXED PLANS, which contribute no abscissae to the counts below. The
        // steps are therefore genuinely identical and this arm is not a
        // copy-paste of model 6's -- but it must exist, because falling through
        // to the unknown arm would report NaN for a profile whose steps are
        // known exactly, and folding it into model 6's arm would mislabel the
        // output. What model 7 changes is invisible to this instrument.
        7 => QuadratureSteps {
            name: "LogQuadratureFittedV7 (atm_model 7; model 6 steps, fitted fixed plans)",
            lower: 0.040,
            middle: 0.300,
            upper: 0.700,
        },
        // Named, not silently folded into the nearest profile. This was an
        // `if atm_model == 4 { .. } else { model 5 }`, which meant that the day
        // the campaign moved to model 6 this program would have priced a
        // 0.300/0.700 arc using model 5's 0.100/0.300 steps and labelled the
        // output "atm_model 5" -- a wrong abscissa count reported with full
        // confidence. An unknown model must read as unknown.
        _ => QuadratureSteps {
            name: "UNKNOWN PROFILE -- abscissa counts below are NOT this model's",
            lower: f64::NAN,
            middle: f64::NAN,
            upper: f64::NAN,
        },
    }
}

/// Abscissae and `atan` traffic per kernel call, priced at the width the code
/// actually runs.
///
/// `element_ns` must be the PER-ELEMENT cost of the routine in use. The `atan`
/// is `f64x4`, so that is the `wide f64x4` figure above and not the scalar one;
/// quoting a vector routine at its scalar price is half of how the retired
/// version of this report reached 68-79%.
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "reported call counts are deliberately converted with the existing floating cost formula"
)]
fn atan_call_count_report(ns_per_eval: f64, element_ns: f64, steps: QuadratureSteps) {
    println!(
        "ATAN ABSCISSAE PER JB2008 KERNEL CALL (= per RHS evaluation), {}:",
        steps.name
    );
    println!("  priced at {element_ns:.3} ns per ELEMENT (f64x4, not scalar)");
    for alt in [200.0, 400.0, 620.0, 800.0, 980.0] {
        let (total, with_atan) = atan_call_count(alt, steps);
        println!(
            "  alt {:>6.0} km   jb_local_temp={:>4}   of which atan={:>4}   -> {:>6.1} ns = {:>5.1}% of a {:.0} ns evaluation",
            alt,
            total,
            with_atan,
            with_atan as f64 * element_ns,
            100.0 * with_atan as f64 * element_ns / ns_per_eval,
            ns_per_eval
        );
    }
}

#[expect(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::while_float,
    clippy::uninlined_format_args,
    clippy::items_after_statements,
    clippy::redundant_closure_for_method_calls,
    clippy::suboptimal_flops,
    clippy::indexing_slicing,
    reason = "this benchmark intentionally preserves its timed control flow, scalar closures, and non-FMA controls"
)]
fn main() -> Result<()> {
    lightyear_odeint_rs::load_constants_from_bytes(
        include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt"),
        5,
    )
    .context("gravity coefficients must load")?;

    let atm_model = resolve_atm_model()?;

    // Untimed first run: loads ephemeris/JB2008 assets and warms every cache.
    let _ = run_arc(atm_model)?;

    // `op-loop <op> <seconds>`: one libm routine and nothing else, so the
    // unnamed `libsystem_m.dylib` blocks can be attributed to an owner by
    // seeing which single-routine workload reproduces them.
    if std::env::args().nth(1).as_deref() == Some("op-loop") {
        let op = std::env::args().nth(2).unwrap_or_default();
        let secs: f64 = std::env::args()
            .nth(3)
            .and_then(|v| v.parse().ok())
            .unwrap_or(20.0);
        let xs: Vec<f64> = (0..1024)
            .map(|i| ((i as f64 + 0.5) / 1024.0 - 0.5) * 12.0)
            .collect();
        let t0 = Instant::now();
        let mut acc = 0.0;
        while t0.elapsed().as_secs_f64() < secs {
            for _ in 0..10_000 {
                for &x in &xs {
                    let x = black_box(x);
                    acc += match op.as_str() {
                        "atan" => x.atan(),
                        "pow25" => (x.abs() + 0.05).powf(2.5),
                        "exp" => (-x.abs()).exp(),
                        "log" => (x.abs() + 0.05).ln(),
                        "log10" => (x.abs() + 0.05).log10(),
                        "sin" => x.sin(),
                        other => anyhow::bail!("unknown op {other}"),
                    };
                }
            }
        }
        println!("op-loop {op}: acc {acc:.3e}");
        return Ok(());
    }

    // `atan-loop <seconds>`: nothing but `f64::atan` on spread operands.
    // Exists to attribute the unnamed `libsystem_m.dylib` leaves around
    // `+0x252xx`, which `sample` cannot symbolicate. If they appear here they
    // belong to `atan` and must be counted as `atan` time in the arc profile.
    if std::env::args().nth(1).as_deref() == Some("atan-loop") {
        let secs: f64 = std::env::args()
            .nth(2)
            .and_then(|v| v.parse().ok())
            .unwrap_or(30.0);
        let xs: Vec<f64> = (0..1024)
            .map(|i| ((i as f64 + 0.5) / 1024.0 - 0.5) * 12.0)
            .collect();
        let t0 = Instant::now();
        let mut acc = 0.0;
        while t0.elapsed().as_secs_f64() < secs {
            for _ in 0..10_000 {
                for &x in &xs {
                    acc += black_box(x).atan();
                }
            }
        }
        println!("atan-loop: acc {acc:.3e}");
        return Ok(());
    }

    // `arc-loop <seconds>`: propagate the production arc and nothing else, so an
    // external sampling profiler sees only the workload under study.
    if std::env::args().nth(1).as_deref() == Some("arc-loop") {
        let secs: f64 = std::env::args()
            .nth(2)
            .and_then(|v| v.parse().ok())
            .unwrap_or(30.0);
        let t0 = Instant::now();
        let mut n = 0u64;
        while t0.elapsed().as_secs_f64() < secs {
            let _ = run_arc(atm_model)?;
            n += 1;
        }
        println!("arc-loop: {n} arcs in {:.2} s", t0.elapsed().as_secs_f64());
        return Ok(());
    }

    let reps = 20;
    let t0 = Instant::now();
    let mut evals = 0;
    let mut adapter = 0;
    let mut kernel = 0;
    for _ in 0..reps {
        let (e, a, k) = run_arc(atm_model)?;
        evals = e;
        adapter = a;
        kernel = k;
    }
    let per_arc_s = t0.elapsed().as_secs_f64() / reps as f64;
    let ns_per_eval = per_arc_s * 1.0e9 / evals as f64;

    println!(
        "ARC   atm_model={atm_model} rhs_evals={evals} jb_adapter_calls={adapter} jb_kernel_calls={kernel}"
    );
    println!(
        "ARC   {:.6} s/arc, {:.2} ns per RHS evaluation",
        per_arc_s, ns_per_eval
    );
    println!(
        "ARC   adapter/eval={:.4} kernel/eval={:.4}",
        adapter as f64 / evals as f64,
        kernel as f64 / evals as f64
    );

    // Operands spread over the ranges the RHS actually sees: angles across all
    // four quadrants, sines/cosines in [-1, 1], exponent arguments the JB2008
    // barometric terms produce.
    const NX: usize = 1024;
    let mut ang = Vec::with_capacity(NX);
    let mut unit = Vec::with_capacity(NX);
    let mut pos = Vec::with_capacity(NX);
    for i in 0..NX {
        let u = (i as f64 + 0.5) / NX as f64;
        ang.push((u - 0.5) * 12.0);
        unit.push((u - 0.5) * 1.98);
        pos.push(0.05 + u * 6.0);
    }

    println!("UNIT COST (throughput, realistic operands, this machine, this build):");
    let n = 30_000_000;
    unit_ns("atan2", &ang, n, |x| x.atan2(1.234_567_8));
    unit_ns("atan", &ang, n, |x| x.atan());
    unit_ns("asin", &unit, n, |x| (x * 0.5).asin());
    unit_ns("sin", &ang, n, |x| x.sin());
    unit_ns("cos", &ang, n, |x| x.cos());
    unit_ns("sin_cos", &ang, n, |x| {
        let (s, c) = x.sin_cos();
        s + c * 0.25
    });
    unit_ns("exp", &ang, n, |x| (-x.abs()).exp());
    unit_ns("ln", &pos, n, |x| x.ln());
    unit_ns("powf2.5", &pos, n, |x| x.powf(2.5));
    unit_ns("sqrt", &pos, n, |x| x.sqrt());
    unit_ns("hypot", &ang, n, |x| x.hypot(1.234_567_8));
    unit_ns("rem_euclid", &ang, n, |x| {
        x.rem_euclid(std::f64::consts::TAU)
    });
    unit_ns("fmul(ctrl)", &ang, n, |x| x * 0.999_999 + 1.0e-9);

    // The JB2008 operand range, log-uniform, for a like-for-like A/B.
    let jb: Vec<f64> = (0..NX)
        .map(|i| 1.0e-6 * (3.0e9_f64).powf((i as f64 + 0.5) / NX as f64))
        .collect();
    println!("ATAN A/B on the JB2008 operand range:");
    let measured_atan_ns = unit_ns("libm atan", &jb, n, |x| x.atan());
    unit_ns("fast_atan", &jb, n, fast_atan_pos);
    unit_ns("shape recip+p", &jb, n, shape_recip_poly);
    unit_ns("shape poly", &jb, n, shape_poly_only);
    unit_ns("fmul(ctrl2)", &jb, n, |x| x * 0.999_999 + 1.0e-9);

    // Boole's rule evaluates the temperature at FOUR independent abscissae per
    // step, so a 4-wide `atan` is the natural width. Price `wide`'s f64x4
    // against scalar libm per ELEMENT, and check what it costs in accuracy.
    let vector_atan_element_ns = {
        use wide::f64x4;
        let quads: Vec<f64x4> = jb
            .chunks_exact(4)
            .map(|c| f64x4::from([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut acc = f64x4::splat(0.0);
        for _ in 0..64 {
            for q in &quads {
                acc += black_box(*q).atan();
            }
        }
        black_box(acc.to_array());
        let rounds = n / (quads.len() as u64 * 4);
        let mut acc = f64x4::splat(0.0);
        let t0 = Instant::now();
        for _ in 0..rounds {
            for q in &quads {
                acc += black_box(*q).atan();
            }
        }
        let dt = t0.elapsed();
        let elems = rounds * quads.len() as u64 * 4;
        let element_ns = dt.as_secs_f64() * 1.0e9 / elems as f64;
        println!(
            "  {:<12} {:>8.3} ns/element  ({} elements, acc {:?})",
            "wide f64x4",
            element_ns,
            elems,
            acc.to_array()[0]
        );
        // Accuracy of the vector routine against libm on the same operands.
        let mut worst_ulp = 0.0_f64;
        let mut worst_rel = 0.0_f64;
        for q in &quads {
            let got = q.atan().to_array();
            for (g, x) in got.iter().zip(q.to_array()) {
                let want = x.atan();
                worst_ulp = worst_ulp.max((g - want).abs() / (want.abs() * f64::EPSILON * 0.5));
                worst_rel = worst_rel.max(((g - want) / want).abs());
            }
        }
        println!("  wide f64x4 accuracy: max {worst_ulp:.3e} ULP, max rel {worst_rel:.3e}");
        element_ns
    };

    atan_candidate_report();
    // The SCALAR figure is still measured above and still reported there, but it
    // is deliberately NOT what prices the quadrature: that path is `f64x4`.
    let _ = measured_atan_ns;
    atan_call_count_report(
        ns_per_eval,
        vector_atan_element_ns,
        quadrature_steps(atm_model),
    );
    density_temperature_sensitivity();
    Ok(())
}
