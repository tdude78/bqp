//! Wave 50v — Rayon dispatch overhead micro-bench.
//!
//! Measures the per-task cost of `.into_par_iter().map(...).collect()`
//! at a matrix of synthetic task sizes × batch sizes. Used to determine
//! the minimum per-task work above which 8-thread `par_iter` wins on
//! M-series Macs (and equivalents elsewhere).
//!
//! ## Findings expected
//!
//! Wave 50c/50d showed that fine-grained kernels (~0.4-0.8 µs/call)
//! see ≤ 1.09× scaling on 8 P-cores. This bench quantifies *why* —
//! the rayon work-stealing scheduler has a fixed per-task dispatch
//! cost. Below that cost, parallel overhead exceeds the work
//! itself. Above it, work-stealing amortises.
//!
//! ## Workload shape
//!
//! Task cost (per-iteration synthetic work):
//! - 10 ns  (~10 floating-point ops)
//! - 100 ns (~100 floating-point ops)
//! - 1 µs   (~1k floating-point ops)
//! - 10 µs  (~10k floating-point ops)
//! - 100 µs (~100k floating-point ops)
//!
//! Batch size (total work items dispatched in one `par_iter` call):
//! - 8 (1 work item per P-core on M3 Max)
//! - 64 (8 per core; ample work to amortise)
//! - 512 (64 per core; saturates work-stealing capacity)
//!
//! Each (task × batch) cell is benchmarked twice: serial (.`iter()`)
//! and parallel (.`into_par_iter()`). The ratio is the parallel
//! speedup; below 1.0× means rayon dispatch dominates the synthetic
//! work.
//!
//! ## Running
//!
//! ```bash
//! cargo bench -p satpy_core --profile criterion-fast --bench rayon_overhead_bench
//! ```
//!
//! Output lands under `target/criterion/rayon_overhead_*/`.

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

/// Synthetic work approximating `n` ns of compute via an unrolled
/// arithmetic chain. The chain depends on the input to defeat the
/// optimiser; the final value is `black_box`-returned so dead-code
/// elimination cannot fold it away.
///
/// Empirical calibration (on M3 Max @ ~5 GHz): each loop iteration
/// is ~1.6 ns (a few f64 mul + add + chain dependency). So `iters`
/// of N produces ~1.6N ns of work. The bench passes pre-computed
/// `iters` values for each target cost.
#[inline(never)]
#[expect(
    clippy::suboptimal_flops,
    reason = "the synthetic benchmark preserves its calibrated multiply-then-add dependency chain"
)]
fn synthetic_work(seed: f64, iters: u32) -> f64 {
    let mut acc = seed;
    for i in 0..iters {
        // Mix in a loop counter so the compiler can't constant-fold.
        let x = f64::from(i) * 1.000_000_1 + 0.5;
        acc = acc.mul_add(x, 1.0e-9);
    }
    acc
}

/// Calibrated iter counts for each target task cost on M3 Max.
/// Re-calibrate via the `calibrate_task_cost` bench if running on
/// a different microarchitecture.
const TASK_COSTS_NS: &[(&str, u32)] = &[
    ("10ns", 6),
    ("100ns", 60),
    ("1us", 625),
    ("10us", 6_250),
    ("100us", 62_500),
];

const BATCH_SIZES: &[usize] = &[8, 64, 512];

/// Serial baseline: `.iter().map(...).collect::<Vec<_>>()`.
/// Provides the "no rayon" reference each (task × batch) cell
/// compares against.
fn bench_serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("rayon_overhead/serial");
    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50);

    for &(cost_label, iters) in TASK_COSTS_NS {
        for &batch in BATCH_SIZES {
            group.throughput(Throughput::Elements(
                u64::try_from(batch).unwrap_or(u64::MAX),
            ));
            let id = BenchmarkId::new(cost_label, batch);
            let seeds: Vec<f64> = (0..batch)
                .map(|index| index.to_f64().unwrap_or(f64::NAN) * 0.001)
                .collect();
            group.bench_with_input(id, &seeds, |b, seeds| {
                b.iter(|| {
                    let out: Vec<f64> = seeds.iter().map(|&s| synthetic_work(s, iters)).collect();
                    black_box(out)
                });
            });
        }
    }
    group.finish();
}

/// Parallel: `.into_par_iter().map(...).collect::<Vec<_>>()`.
/// Identical work shape as the serial baseline — only the iterator
/// kind differs. Honors `RAYON_NUM_THREADS` for the global pool.
fn bench_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("rayon_overhead/parallel");
    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50);

    for &(cost_label, iters) in TASK_COSTS_NS {
        for &batch in BATCH_SIZES {
            group.throughput(Throughput::Elements(
                u64::try_from(batch).unwrap_or(u64::MAX),
            ));
            let id = BenchmarkId::new(cost_label, batch);
            let seeds: Vec<f64> = (0..batch)
                .map(|index| index.to_f64().unwrap_or(f64::NAN) * 0.001)
                .collect();
            group.bench_with_input(id, &seeds, |b, seeds| {
                b.iter(|| {
                    let out: Vec<f64> = seeds
                        .par_iter()
                        .map(|&s| synthetic_work(s, iters))
                        .collect();
                    black_box(out)
                });
            });
        }
    }
    group.finish();
}

/// Calibration bench: confirm the per-iter cost of `synthetic_work`.
/// Run once when adapting `TASK_COSTS_NS` constants to a new
/// microarchitecture. Not part of the main matrix.
fn bench_calibration(c: &mut Criterion) {
    let mut group = c.benchmark_group("rayon_overhead/calibrate");
    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(30);

    // Sweep iter counts to map iters → wall-clock cost.
    for &iters in &[10_u32, 100, 1_000, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(iters), &iters, |b, &iters| {
            b.iter(|| black_box(synthetic_work(1.0, iters)));
        });
    }
    group.finish();
}

criterion_group!(serial_benches, bench_serial);
criterion_group!(parallel_benches, bench_parallel);
criterion_group!(calibration_benches, bench_calibration);

// pprof flamegraph capture from the branch version was dropped in the port:
// no pprof dep exists on main, and criterion timings alone back the lint's
// 10 µs threshold (wave 50v).
fn main() {
    serial_benches();
    parallel_benches();
    calibration_benches();
    Criterion::default().configure_from_args().final_summary();
}
