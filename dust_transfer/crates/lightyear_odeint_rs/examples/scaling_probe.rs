//! Thread-scaling probe that separates machine effects from code effects.
//!
//! The criterion `rhs_parallel_scaling` bench measures
//! `Instant::now()` -> spawn W threads -> join all -> `elapsed()`. That number is
//! the MAXIMUM over W threads plus the cost of spawning and joining them, and it
//! is charged to every iteration. Two things inflate it as W grows even when
//! every thread runs at exactly the 1-thread speed:
//!
//! 1. `max` over W samples of a right-skewed distribution rises with W. One
//!    descheduled thread moves the reported number for all of them.
//! 2. Thread create/join is inside the timed region and is O(W) serial work in
//!    the kernel.
//!
//! This probe removes both. Threads are spawned once, warm up, sync on a
//! barrier, and then each thread times ONLY its own loop. We report the min,
//! median and max of the per-thread rates, plus the harness-style span
//! (last-finish minus first-start) so the two methods can be compared directly
//! on one run.
//!
//! It also carries controls, because "the RHS scales badly" and "this node
//! scales badly" produce the same criterion number:
//!
//! * `fp`      - dependent FMA chain in registers. Touches no memory, takes no
//!   lock, allocates nothing. If THIS degrades at W=64, the node is the story
//!   and no code change can fix it.
//! * `stream`  - per-thread buffer sweep at a caller-chosen size. Degradation
//!   here prices the shared L3 / memory controllers.
//! * `rhs_const` - the criterion bench's exact workload: one fixed `t`, one
//!   fixed `delta`. Every cache in the RHS keys on `t` or on the resulting
//!   position, so after the first iteration the baseline cache, frame-rotation
//!   cache, and gravity V/W recurrence cache all hit. Gravity still performs
//!   coefficient-dependent summation, so this measures a partial-cache path.
//! * `rhs_sweep` - the same synthetic RHS with `t` swept across 512 distinct
//!   values, forcing cache misses. This probes scaling mechanics only; it is
//!   not current Part A model-8 campaign performance.
//! * `crit_cold` / `crit_warm` - criterion-style spawn/join controls using,
//!   respectively, independently constructed workers per round or persistent
//!   warmed workers. Construction stays outside the timer in both modes.
//!
//! Usage: `scaling_probe <workload> <width> <iters> [stream_kib|rounds]`

use std::sync::{Arc, Barrier};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use lightyear_odeint_rs::{
    rhs::LightyearRHS,
    types::{BodyInvariants, ForceConfig, ForceFlags},
};
use satpy_core::{eci2equinoc_impl, pack_gravity_coeffs};

/// Same coefficient shape the criterion bench uses, so the two are comparable.
fn create_test_coefficients(order: usize) -> Result<(Vec<f64>, Vec<f64>, usize)> {
    let stride = order.checked_add(2).context("gravity stride overflowed")?;
    let coefficient_count = stride
        .checked_mul(stride)
        .context("gravity coefficient count overflowed")?;
    let mut c_coeffs = vec![0.0; coefficient_count];
    let mut s_coeffs = vec![0.0; coefficient_count];
    *c_coeffs
        .get_mut(0)
        .context("gravity coefficient table must have C00")? = 1.0;
    for l in 2..=order {
        let base = l
            .checked_mul(stride)
            .context("gravity row offset overflowed")?;
        let l_f64 = usize_as_f64(l)?;
        *c_coeffs
            .get_mut(base)
            .context("gravity row start must fit coefficient table")? = 1e-3 / l_f64.powi(2);
        for m in 1..=l {
            let denominator = l
                .checked_mul(m)
                .context("gravity coefficient denominator overflowed")?;
            let magnitude = 1e-6 / usize_as_f64(denominator)?;
            let index = base
                .checked_add(m)
                .context("gravity coefficient index overflowed")?;
            *c_coeffs
                .get_mut(index)
                .context("gravity C coefficient index must fit table")? = magnitude;
            *s_coeffs
                .get_mut(index)
                .context("gravity S coefficient index must fit table")? = magnitude * 0.5;
        }
    }
    Ok((c_coeffs, s_coeffs, stride))
}

fn create_synthetic_cache_miss_force_config() -> ForceConfig {
    let sun_pos = [1.495_978_707e8, 1.0e4, -2.0e4];
    let moon_pos = [384_400.0, 2.0e3, -5.0e3];
    ForceConfig {
        sph_order: 5,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        subtract_first_order: true,
        atm_model: 3,
        am_ratio: 0.01,
        cd: 2.2,
        cr: 1.3,
        target_propagation_mode: 0,
        qm_ratio: 0.0,
        r_obj_m: 0.0,
        omega_earth: 7.292_115_0e-5,
        p_sun: 4.56e-6,
        mu_sun: 1.327_124_400_18e11,
        mu_moon: 4_902.800_066,
        mu_jupiter: 1.266_865_34e8,
        mu_venus: 3.248_585_92e5,
        mu_mars: 4.282_837_5e4,
        mu_saturn: 3.793_120_6e7,
        earth_radius: 6378.137,
        sun_pos: Some(sun_pos),
        moon_pos: Some(moon_pos),
        jupiter_pos: None,
        venus_pos: None,
        mars_pos: None,
        saturn_pos: None,
        dynamic_ephemeris_flags: 0,
        sun_invariants: BodyInvariants::precompute(&sun_pos, 1.327_124_400_18e11),
        moon_invariants: BodyInvariants::precompute(&moon_pos, 4_902.800_066),
        jupiter_invariants: None,
        venus_invariants: None,
        mars_invariants: None,
        saturn_invariants: None,
        dt_max: 60.0,
        eps: 1e-8,
        integrator_method: lightyear_odeint_rs::types::StepperMethod::Dopri5Compat,
    }
}

struct RhsFactory {
    init_equ: [f64; 6],
    config: Arc<ForceConfig>,
    packed: Arc<satpy_core::PackedGravityCoeffs>,
}

impl RhsFactory {
    fn build(&self) -> Result<LightyearRHS> {
        LightyearRHS::try_new(
            self.init_equ,
            0.0,
            2_460_000.5,
            Arc::clone(&self.config),
            Arc::clone(&self.packed),
        )
        .map_err(anyhow::Error::msg)
        .context("valid atmosphere model")
    }
}

fn build_rhs_factory() -> Result<RhsFactory> {
    let order = 5;
    let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order)?;
    let packed = pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order)
        .context("valid gravity coefficients")?;
    let init_eci = [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0];
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);
    Ok(RhsFactory {
        init_equ,
        config: Arc::new(create_synthetic_cache_miss_force_config()),
        packed: Arc::new(packed),
    })
}

fn build_rhs() -> Result<LightyearRHS> {
    build_rhs_factory()?.build()
}

fn usize_as_f64(value: usize) -> Result<f64> {
    value
        .to_string()
        .parse()
        .context("probe count must parse as f64")
}

fn u64_as_f64(value: u64) -> Result<f64> {
    value
        .to_string()
        .parse()
        .context("probe count must parse as f64")
}

fn u128_as_f64(value: u128) -> Result<f64> {
    value
        .to_string()
        .parse()
        .context("probe duration must parse as f64")
}

#[expect(
    clippy::suboptimal_flops,
    reason = "the probe keeps the prior add/multiply rounding instead of changing its sampled epochs"
)]
fn ephemeris_probe_jd(phase: f64) -> f64 {
    2_460_000.5 + phase * 0.01
}

/// What one worker thread runs. Returns an accumulator so nothing is optimized
/// away, and is called once for warmup and once under the timer.
enum Work {
    Fp,
    Stream,
    RhsConst(LightyearRHS),
    RhsSweep(LightyearRHS),
    /// `ForceConfig::with_ephemeris_for_arc`, which every HF row and every HF
    /// propagation segment calls, and which reaches an unconditional
    /// `GLOBAL_EPHEMERIS.write()`.
    Ephem(ForceConfig),
    /// The locking call underneath it, isolated from the position interpolation.
    EphemLoad(i32),
}

const DELTA: [f64; 6] = [1.0e-3, -2.0e-3, 7.0e-4, 2.0e-6, -4.0e-6, 1.0e-6];
const CONST_T: f64 = 123.456;

#[derive(Clone, Copy)]
struct RunPlan {
    raw: u64,
    legacy_indexed: usize,
    ephemeris: u32,
}

/// Validate the invariant inputs before either the warmup or timed region.
///
/// The indexed workloads intentionally retain their original modulo/index
/// loops, so this is where their bounds proof lives rather than inside the
/// measured loop.
fn prepare_run(work: &Work, iters: u64, buf: &[f64], times: &[f64]) -> Result<RunPlan> {
    let mut plan = RunPlan {
        raw: iters,
        legacy_indexed: 0,
        ephemeris: 0,
    };
    match work {
        Work::Stream => {
            ensure!(
                !buf.is_empty(),
                "stream workload requires a nonempty buffer"
            );
            ensure!(
                buf.len().checked_rem(8) == Some(0),
                "stream buffer must be an exact cache-line stride multiple"
            );
            let indexed_as_usize =
                usize::try_from(iters).context("indexed probe iterations must fit usize")?;
            ensure!(
                indexed_as_usize <= usize::MAX / 8,
                "stream iteration index times stride must fit usize"
            );
            plan.legacy_indexed = indexed_as_usize;
        }
        Work::RhsSweep(_) => {
            ensure!(!times.is_empty(), "RHS sweep requires at least one time");
            plan.legacy_indexed =
                usize::try_from(iters).context("indexed probe iterations must fit usize")?;
        }
        Work::Ephem(_) => {
            plan.ephemeris =
                u32::try_from(iters).context("ephemeris probe iterations must fit u32")?;
        }
        Work::Fp | Work::RhsConst(_) | Work::EphemLoad(_) => {}
    }
    Ok(plan)
}

enum LegacyIndexedWork<'a> {
    Stream(&'a mut [f64]),
    RhsSweep(&'a mut LightyearRHS, &'a [f64]),
}

#[expect(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    reason = "prevalidated benchmark buffers and iteration count retain the legacy timed modulo/index control path"
)]
#[inline]
fn run_legacy_indexed(work: LegacyIndexedWork<'_>, iterations: usize) -> Result<f64> {
    match work {
        LegacyIndexedWork::Stream(buf) => {
            let mut acc = 0.0f64;
            let n = buf.len();
            for i in 0..iterations {
                // Stride by 8 doubles = one cache line per touch.
                let base = (i * 8) % n;
                acc += buf[base];
                buf[base] = acc * 1e-30;
            }
            Ok(acc)
        }
        LegacyIndexedWork::RhsSweep(rhs, times) => {
            let mut acc = 0.0f64;
            let n = times.len();
            for i in 0..iterations {
                let t = times[i % n];
                let [_, _, _, acceleration_x, _, _] = rhs
                    .compute_internal(std::hint::black_box(&DELTA), std::hint::black_box(t))
                    .context("RHS sweep derivative")?;
                acc += acceleration_x;
            }
            Ok(acc)
        }
    }
}

fn run(work: &mut Work, plan: RunPlan, buf: &mut [f64], times: &[f64]) -> Result<f64> {
    match work {
        // Two independent chains so the loop is latency-bound on the FMA unit
        // rather than on the loop counter, which is what the RHS is too.
        Work::Fp => {
            let mut a = 1.000_000_1_f64;
            let mut b = 0.999_999_9_f64;
            for _ in 0..plan.raw {
                a = a.mul_add(1.000_000_000_1, 1e-12);
                b = b.mul_add(0.999_999_999_9, 1e-12);
            }
            Ok(a + b)
        }
        Work::Stream => run_legacy_indexed(LegacyIndexedWork::Stream(buf), plan.legacy_indexed),
        // `black_box` on both arguments, matching the criterion bench. Without
        // it LLVM sees compile-time-constant inputs and hoists real work out of
        // the loop, which understates the cost by ~3x.
        Work::RhsConst(rhs) => {
            let mut acc = 0.0f64;
            for _ in 0..plan.raw {
                let out = rhs
                    .compute_internal(std::hint::black_box(&DELTA), std::hint::black_box(CONST_T))
                    .context("constant RHS derivative")?;
                let [_, _, _, acceleration_x, _, _] = out;
                acc += acceleration_x;
            }
            Ok(acc)
        }
        Work::Ephem(cfg) => {
            let mut acc = 0.0f64;
            for i in 0..plan.ephemeris {
                // Keep the original add/multiply order; this is a probe, not
                // an authority change.
                let jd = ephemeris_probe_jd(f64::from(i % 64));
                let resolved = cfg
                    .with_ephemeris_for_arc(std::hint::black_box(jd), jd + 0.01)
                    .context("ephemeris coverage")?;
                acc += resolved.sun_pos.map_or(0.0, |[x, _, _]| x);
            }
            Ok(acc)
        }
        Work::EphemLoad(flags) => {
            let mut acc = 0.0f64;
            for _ in 0..plan.raw {
                lightyear_odeint_rs::precomputed_ephem::load_precomputed_ephemeris(
                    std::hint::black_box(*flags),
                )
                .context("catalogues present")?;
                acc += 1.0;
            }
            Ok(acc)
        }
        Work::RhsSweep(rhs) => {
            run_legacy_indexed(LegacyIndexedWork::RhsSweep(rhs, times), plan.legacy_indexed)
        }
    }
}

/// Reproduce the criterion bench's spawn/join measurement shape.
///
/// Cold rounds construct independent workers before timing each round. Warm
/// rounds construct once, warm the same owned workers, then reuse them. Both
/// forms share only immutable force and gravity assets through `Arc`.
fn criterion_style(
    factory: &RhsFactory,
    width: usize,
    iters: u64,
    rounds: usize,
    fresh: bool,
) -> Result<()> {
    ensure!(width > 0, "criterion width must be positive");
    ensure!(rounds > 0, "criterion rounds must be positive");
    let mut persistent = if fresh {
        Vec::new()
    } else {
        (0..width)
            .map(|_| factory.build())
            .collect::<Result<Vec<_>>>()?
    };
    if !fresh {
        for worker in &mut persistent {
            for _ in 0..1000 {
                let derivative = worker
                    .compute_internal(std::hint::black_box(&DELTA), std::hint::black_box(CONST_T))
                    .context("criterion warmup RHS derivative")?;
                std::hint::black_box(derivative);
            }
        }
    }

    let iters_f64 = u64_as_f64(iters)?;
    let mut per_round = Vec::with_capacity(rounds);
    let mut per_thread_max = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let mut round_fresh = if fresh {
            (0..width)
                .map(|_| factory.build())
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let workers = if fresh {
            &mut round_fresh
        } else {
            &mut persistent
        };
        let start = Instant::now();
        let inner_elapsed: Result<Vec<u128>> = std::thread::scope(|scope| {
            let handles: Vec<_> = workers
                .iter_mut()
                .map(|worker| {
                    scope.spawn(move || -> Result<u128> {
                        let own = Instant::now();
                        let mut acc = 0.0f64;
                        for _ in 0..iters {
                            let [_, _, _, acceleration_x, _, _] = worker
                                .compute_internal(
                                    std::hint::black_box(&DELTA),
                                    std::hint::black_box(CONST_T),
                                )
                                .context("criterion worker RHS derivative")?;
                            acc += acceleration_x;
                        }
                        let elapsed_ns = own.elapsed().as_nanos();
                        std::hint::black_box(acc);
                        Ok(elapsed_ns)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("criterion worker panicked"))?
                })
                .collect()
        });
        per_round.push(u128_as_f64(start.elapsed().as_nanos())? / iters_f64);
        per_thread_max.push(
            inner_elapsed?
                .into_iter()
                .map(|elapsed_ns| u128_as_f64(elapsed_ns).map(|ns| ns / iters_f64))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .fold(0.0f64, f64::max),
        );
    }

    per_round.sort_by(f64::total_cmp);
    per_thread_max.sort_by(f64::total_cmp);
    let median = rounds.checked_div(2).context("median index overflowed")?;
    println!(
        "{} width={width} iters={iters} rounds={rounds} span_min={:.1} span_p50={:.1} \
         span_max={:.1} | inner_max_p50={:.1}",
        if fresh { "crit_cold" } else { "crit_warm" },
        per_round
            .first()
            .context("criterion span samples must exist")?,
        per_round
            .get(median)
            .context("criterion median span sample must exist")?,
        per_round
            .last()
            .context("criterion span samples must exist")?,
        per_thread_max
            .get(median)
            .context("criterion median worker sample must exist")?,
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: scaling_probe \
             <fp|stream|rhs_const|rhs_sweep|ephem|ephem_load|crit_cold|crit_warm> \
             <width> <iters> [stream_kib|rounds]"
        );
        std::process::exit(2);
    }
    let workload_arg = args.get(1).context("workload argument must be present")?;
    let width: usize = args
        .get(2)
        .context("width argument must be present")?
        .parse()
        .context("width must parse as usize")?;
    let iters: u64 = args
        .get(3)
        .context("iteration argument must be present")?
        .parse()
        .context("iters must parse as u64")?;
    println!(
        "SCALING_PROBE_IDENTITY authority=synthetic-cache-miss-probe \
         current_part_a_authority=false atmosphere_model=3 spherical_order=5 \
         stepper=Dopri5Compat purpose=cache-miss-and-thread-scaling-only"
    );
    if workload_arg == "crit_cold" || workload_arg == "crit_warm" {
        let rounds: usize = args.get(4).map_or(Ok(21), |value| {
            value.parse().context("rounds must parse as usize")
        })?;
        criterion_style(
            &build_rhs_factory()?,
            width,
            iters,
            rounds,
            workload_arg == "crit_cold",
        )?;
        return Ok(());
    }
    ensure!(width > 0, "probe width must be positive");
    let workload = workload_arg.clone();
    let stream_kib: usize = args.get(4).map_or(Ok(256), |value| {
        value.parse().context("stream_kib must parse as usize")
    })?;
    let stream_len = stream_kib
        .checked_mul(128)
        .context("stream buffer length overflowed")?;
    ensure!(
        stream_len > 0,
        "stream buffer must contain at least one double"
    );

    // Construct each worker's owned RHS before warm-up and timing.
    let mut works: Vec<Work> = (0..width)
        .map(|_| -> Result<Work> {
            let work = match workload.as_str() {
                "fp" => Work::Fp,
                "stream" => Work::Stream,
                "rhs_const" => Work::RhsConst(build_rhs()?),
                "rhs_sweep" => Work::RhsSweep(build_rhs()?),
                // This synthetic atmosphere-3 probe uses drag, which forces
                // SUN_GRAVITY dynamic regardless of sun_pos. Dropping the static
                // positions reproduces the same nonzero dynamic-flag mechanics
                // without dragging in JB2008 driver-arc validation.
                "ephem" => {
                    let mut cfg = create_synthetic_cache_miss_force_config();
                    cfg.sun_pos = None;
                    cfg.moon_pos = None;
                    cfg.sun_invariants = None;
                    cfg.moon_invariants = None;
                    Work::Ephem(cfg)
                }
                "ephem_load" => Work::EphemLoad(ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY),
                other => bail!("unknown workload {other}"),
            };
            Ok(work)
        })
        .collect::<Result<_>>()?;

    let warmup = (iters / 10).max(1000);
    let barrier = Arc::new(Barrier::new(width));
    let t_origin = Instant::now();

    // (start_offset_ns, end_offset_ns, elapsed_ns) per thread.
    let results: Result<Vec<(u128, u128, u128)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = works
            .iter_mut()
            .map(|work| {
                let barrier = Arc::clone(&barrier);
                let t_origin = &t_origin;
                scope.spawn(move || -> Result<(u128, u128, u128)> {
                    // Per-thread buffers, allocated and first-touched by the
                    // thread that uses them, so NUMA first-touch puts them local.
                    let mut buf = vec![0.0f64; stream_len];
                    for (i, slot) in buf.iter_mut().enumerate() {
                        *slot = usize_as_f64(i)?;
                    }
                    let times: Vec<f64> = (0..512).map(|i| f64::from(i) * 0.37).collect();
                    let warmup_plan = prepare_run(work, warmup, &buf, &times);
                    let timed_plan = prepare_run(work, iters, &buf, &times);

                    barrier.wait();
                    std::hint::black_box(run(work, warmup_plan?, &mut buf, &times)?);
                    let start = Instant::now();
                    let acc = run(work, timed_plan?, &mut buf, &times)?;
                    let elapsed = start.elapsed();
                    std::hint::black_box(acc);
                    let start_ns = start.duration_since(*t_origin).as_nanos();
                    let elapsed_ns = elapsed.as_nanos();
                    let end_ns = start_ns
                        .checked_add(elapsed_ns)
                        .context("thread timestamp overflowed")?;
                    Ok((start_ns, end_ns, elapsed_ns))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("probe worker panicked"))?
            })
            .collect::<Result<Vec<_>>>()
    });
    let results = results?;

    let iters_f64 = u64_as_f64(iters)?;
    let mut per_iter: Vec<f64> = results
        .iter()
        .map(|(_, _, elapsed_ns)| Ok(u128_as_f64(*elapsed_ns)? / iters_f64))
        .collect::<Result<_>>()?;
    per_iter.sort_by(f64::total_cmp);

    let first_start = results
        .iter()
        .map(|(start, _, _)| *start)
        .min()
        .context("probe results must contain a start timestamp")?;
    let last_end = results
        .iter()
        .map(|(_, end, _)| *end)
        .max()
        .context("probe results must contain an end timestamp")?;
    let span_ns = last_end
        .checked_sub(first_start)
        .context("probe end timestamp preceded start timestamp")?;
    let span_per_iter = u128_as_f64(span_ns)? / iters_f64;
    let minimum = *per_iter
        .first()
        .context("probe worker results must exist")?;
    let median_index = width.checked_div(2).context("median index overflowed")?;
    let median = *per_iter
        .get(median_index)
        .context("probe median worker result must exist")?;
    let p90_index = width
        .checked_mul(9)
        .context("p90 index multiplication overflowed")?
        .checked_div(10)
        .context("p90 index division overflowed")?;
    let p90 = *per_iter
        .get(p90_index)
        .context("probe p90 worker result must exist")?;
    let maximum = *per_iter.last().context("probe worker results must exist")?;

    // `span` is what the criterion bench reports; `p50` is what each thread
    // actually achieved. The gap between them is harness skew, not code.
    println!(
        "{workload} width={width} iters={iters} stream_kib={stream_kib} \
         min={:.1} p50={:.1} p90={:.1} max={:.1} span={:.1} tail_pct={:.1}",
        minimum,
        median,
        p90,
        maximum,
        span_per_iter,
        100.0 * (maximum / median - 1.0),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        create_synthetic_cache_miss_force_config, create_test_coefficients, prepare_run, run, Work,
    };

    #[test]
    fn synthetic_probe_identity_is_explicit_and_not_part_a_authority() {
        let config = create_synthetic_cache_miss_force_config();
        assert_eq!(config.atm_model, 3);
        assert_eq!(config.sph_order, 5);
        assert_eq!(
            config.integrator_method,
            lightyear_odeint_rs::types::StepperMethod::Dopri5Compat
        );
        assert_ne!(
            config.atm_model,
            nd_config::CompiledPartAScienceV1::part_a_v1()
                .hybrid()
                .atmosphere_model
        );
    }

    #[test]
    fn synthetic_coefficient_shape_stays_square_and_seeded() -> anyhow::Result<()> {
        let (c_coeffs, s_coeffs, stride) = create_test_coefficients(5)?;
        assert_eq!(stride, 7);
        assert_eq!(c_coeffs.len(), 49);
        assert_eq!(s_coeffs.len(), 49);
        let c00 = *c_coeffs
            .first()
            .ok_or_else(|| anyhow::anyhow!("C00 coefficient must exist"))?;
        assert_eq!(c00.to_bits(), 1.0_f64.to_bits());
        Ok(())
    }

    #[test]
    fn stream_stride_repeats_the_original_cache_line_lanes() -> anyhow::Result<()> {
        let mut work = Work::Stream;
        let mut buffer: Vec<f64> = (0..16).map(f64::from).collect();
        let plan = prepare_run(&work, 4, &buffer, &[0.0])?;
        let result = run(&mut work, plan, &mut buffer, &[0.0])?;
        assert_eq!(result.to_bits(), 8.0_f64.to_bits());
        Ok(())
    }

    #[test]
    fn stream_plan_rejects_an_empty_buffer_before_the_timed_loop() {
        let work = Work::Stream;
        assert!(prepare_run(&work, 4, &[], &[0.0]).is_err());
    }
}
