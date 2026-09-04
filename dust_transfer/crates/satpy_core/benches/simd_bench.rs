//! Benchmarks for SIMD vs scalar coordinate transformations.
//!
//! Compares:
//! - `eci2equinoc_simd4` (SIMD) vs 4x `eci2equinoc_impl` (scalar)
//! - `equinoc2eci_simd4` (SIMD) vs 4x `equinoc2eci_impl` (scalar)
//!
//! ## Running Benchmarks
//!
//! ```bash
//! # All SIMD benchmarks
//! cargo bench -p satpy_core --features simd -- simd
//!
//! # Compare with scalar fallback
//! cargo bench -p satpy_core --no-default-features -- simd_comparison
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_traits::ToPrimitive;
use satpy_core::{
    eci2equinoc_impl, eci2equinoc_simd16, eci2equinoc_simd4, equinoc2eci_impl,
    equinoc2eci_impl_f64, equinoc2eci_simd4, equinoc_prop_step_impl, equinoc_prop_step_simd4,
};
use std::time::Duration;

/// Create diverse ECI states for realistic benchmarking
const fn create_test_eci_states() -> [[f64; 6]; 4] {
    [
        // LEO equatorial circular
        [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
        // LEO inclined
        [6778.0, 0.0, 100.0, 0.0, 7.5, 0.5],
        // Different position
        [0.0, 6778.0, 0.0, -7.67, 0.0, 0.0],
        // Near-polar
        [4800.0, 0.0, 4800.0, 0.0, 5.4, 5.4],
    ]
}

/// Create diverse equinoctial states for benchmarking
const fn create_test_equ_states() -> [[f64; 6]; 4] {
    [
        // LEO circular
        [6778.0, 0.001, 0.0, 0.5, 0.0, 0.0],
        // LEO with eccentricity
        [6778.0, 0.05, 0.01, 0.4, 0.1, 1.0],
        // Higher altitude
        [8000.0, 0.02, 0.02, 0.3, 0.2, 2.0],
        // Inclined
        [7000.0, 0.01, 0.03, 0.45, 0.15, 3.0],
    ]
}

fn flatten_four_states(states: [[f64; 6]; 4]) -> [f64; 24] {
    let mut block = [0.0; 24];
    for (output, state) in block.chunks_exact_mut(6).zip(states) {
        output.copy_from_slice(&state);
    }
    block
}

/// Benchmark eci2equinoc: SIMD4 vs 4x scalar
fn bench_eci2equinoc_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("eci2equinoc_simd_vs_scalar");
    group.throughput(Throughput::Elements(4));

    let eci_block = flatten_four_states(create_test_eci_states());

    // SIMD4 path
    group.bench_function("simd4", |b| {
        b.iter(|| {
            let mut out = [0.0; 24];
            eci2equinoc_simd4(
                std::hint::black_box(&eci_block),
                std::hint::black_box(0.0),
                std::hint::black_box(0.0),
                &mut out,
            );
            out
        });
    });

    // Scalar path (4x loop)
    group.bench_function("scalar_4x", |b| {
        b.iter(|| {
            let mut out = [0.0; 24];
            for (input, output) in eci_block.chunks_exact(6).zip(out.chunks_exact_mut(6)) {
                eci2equinoc_impl(
                    std::hint::black_box(input),
                    6,
                    std::hint::black_box(0.0),
                    std::hint::black_box(0.0),
                    output,
                );
            }
            out
        });
    });

    group.finish();
}

/// Benchmark equinoc2eci: SIMD4 vs 4x scalar
fn bench_equinoc2eci_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("equinoc2eci_simd_vs_scalar");
    group.throughput(Throughput::Elements(4));

    let equ_block = flatten_four_states(create_test_equ_states());

    // SIMD4 path
    group.bench_function("simd4", |b| {
        b.iter(|| {
            let mut out = [0.0; 24];
            equinoc2eci_simd4(
                std::hint::black_box(&equ_block),
                std::hint::black_box(0.0),
                std::hint::black_box(0.0),
                &mut out,
            );
            out
        });
    });

    // Scalar path (4x loop)
    group.bench_function("scalar_4x", |b| {
        b.iter(|| {
            let mut out = [0.0; 24];
            for (input, output) in equ_block.chunks_exact(6).zip(out.chunks_exact_mut(6)) {
                equinoc2eci_impl(
                    std::hint::black_box(input),
                    6,
                    std::hint::black_box(0.0),
                    std::hint::black_box(0.0),
                    output,
                );
            }
            out
        });
    });

    group.finish();
}

/// Benchmark SIMD16 (16 states) vs scalar
fn bench_simd16_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd16_vs_scalar");
    group.throughput(Throughput::Elements(16));

    // Create 16 ECI states (repeat 4 patterns)
    let eci_4 = create_test_eci_states();
    let mut eci_block = [0.0; 96];
    for (output, state) in eci_block.chunks_exact_mut(6).zip(eci_4.into_iter().cycle()) {
        output.copy_from_slice(&state);
    }

    // SIMD16 path
    group.bench_function("eci2equinoc_simd16", |b| {
        b.iter(|| {
            let mut out = [0.0; 96];
            eci2equinoc_simd16(
                std::hint::black_box(&eci_block),
                std::hint::black_box(0.0),
                std::hint::black_box(0.0),
                &mut out,
            );
            out
        });
    });

    // Scalar path (16x loop)
    group.bench_function("eci2equinoc_scalar_16x", |b| {
        b.iter(|| {
            let mut out = [0.0; 96];
            for (input, output) in eci_block.chunks_exact(6).zip(out.chunks_exact_mut(6)) {
                eci2equinoc_impl(
                    std::hint::black_box(input),
                    6,
                    std::hint::black_box(0.0),
                    std::hint::black_box(0.0),
                    output,
                );
            }
            out
        });
    });

    group.finish();
}

/// Benchmark roundtrip: ECI -> Equinoctial -> ECI (SIMD vs scalar)
fn bench_roundtrip_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip_simd_vs_scalar");
    group.throughput(Throughput::Elements(4));

    let eci_block = flatten_four_states(create_test_eci_states());

    // SIMD roundtrip
    group.bench_function("simd4_roundtrip", |b| {
        b.iter(|| {
            let mut equ = [0.0; 24];
            let mut eci_out = [0.0; 24];
            eci2equinoc_simd4(&eci_block, 0.0, 0.0, &mut equ);
            equinoc2eci_simd4(&equ, 0.0, 0.0, &mut eci_out);
            eci_out
        });
    });

    // Scalar roundtrip
    group.bench_function("scalar_4x_roundtrip", |b| {
        b.iter(|| {
            let mut equ = [0.0; 24];
            let mut eci_out = [0.0; 24];
            for ((input, equ_output), eci_output) in eci_block
                .chunks_exact(6)
                .zip(equ.chunks_exact_mut(6))
                .zip(eci_out.chunks_exact_mut(6))
            {
                eci2equinoc_impl(input, 6, 0.0, 0.0, equ_output);
                equinoc2eci_impl(equ_output, 6, 0.0, 0.0, eci_output);
            }
            eci_out
        });
    });

    group.finish();
}

/// Benchmark scaling: 4, 16, 64, 256 states
fn bench_simd_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_scaling");

    let eci_4 = create_test_eci_states();

    for n_states in [4_usize, 16, 64, 256] {
        let n_blocks = n_states / 4;

        // Build state vector
        let mut eci_vec: Vec<f64> = Vec::with_capacity(n_states.saturating_mul(6));
        for _ in 0..n_blocks {
            for state in &eci_4 {
                eci_vec.extend_from_slice(state);
            }
        }

        group.throughput(Throughput::Elements(
            u64::try_from(n_states).unwrap_or(u64::MAX),
        ));

        // SIMD path (process in chunks of 4)
        group.bench_with_input(
            BenchmarkId::new("simd4_batched", n_states),
            &eci_vec,
            |b, eci| {
                b.iter(|| {
                    let mut out = vec![0.0; eci.len()];
                    for (input, output) in eci.chunks_exact(24).zip(out.chunks_exact_mut(24)) {
                        if let (Ok(input_array), Ok(output_array)) = (
                            <&[f64; 24]>::try_from(input),
                            <&mut [f64; 24]>::try_from(output),
                        ) {
                            eci2equinoc_simd4(input_array, 0.0, 0.0, output_array);
                        }
                    }
                    out
                });
            },
        );

        // Scalar path
        group.bench_with_input(BenchmarkId::new("scalar", n_states), &eci_vec, |b, eci| {
            b.iter(|| {
                let mut out = vec![0.0; eci.len()];
                for (input, output) in eci.chunks_exact(6).zip(out.chunks_exact_mut(6)) {
                    eci2equinoc_impl(input, 6, 0.0, 0.0, output);
                }
                out
            });
        });
    }

    group.finish();
}

/// Phase H-2: SIMD4 chunk-size threshold sweep for `equinoc_prop_step_impl`.
///
/// Compares scalar `equinoc_prop_step_impl` (calls `equinoc2eci_impl_f64` in a
/// loop) against `equinoc_prop_step_simd4` (one SIMD call per 4-lane chunk +
/// scalar tail) across N ∈ {1, 4, 8, 16, 32, 64, 128} states. Determines the
/// chunk-size threshold where SIMD4 actually wins for the
/// `TargetStateTable::rebuild` call pattern. The prior HF-NEW-04 attempt
/// regressed at production N because compiler already inlines the scalar
/// pattern; this bench documents the boundary.
fn bench_equinoc_prop_step_threshold(c: &mut Criterion) {
    use std::hint::black_box;

    let [_, equinoc, _, _] = create_test_equ_states();
    let t0 = 0.0_f64;

    // Generate dt sweep — evenly spaced over a typical solver horizon.
    let make_t_vals = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|index| 60.0 * index.saturating_add(1).to_f64().unwrap_or(f64::NAN))
            .collect()
    };

    let mut group = c.benchmark_group("equinoc_prop_step_threshold");
    group
        .warm_up_time(Duration::from_millis(400))
        .measurement_time(Duration::from_secs(1))
        .sample_size(30);

    for &n in &[1usize, 4, 8, 16, 32, 64, 128] {
        let t_vals = make_t_vals(n);
        let output_len = n.saturating_mul(6);
        let mut out_scalar = vec![0.0_f64; output_len];
        let mut out_simd = vec![0.0_f64; output_len];
        group.throughput(Throughput::Elements(u64::try_from(n).unwrap_or(u64::MAX)));

        // Scalar: equinoc_prop_step_impl walks t_vals via equinoc2eci_impl_f64.
        group.bench_function(BenchmarkId::new("scalar", n), |b| {
            b.iter(|| {
                equinoc_prop_step_impl(
                    black_box(&equinoc),
                    black_box(&t_vals),
                    black_box(t0),
                    black_box(&mut out_scalar),
                );
            });
        });

        // SIMD4: chunk t_vals into groups of 4, call equinoc_prop_step_simd4
        // per chunk, scalar tail for remainder.
        group.bench_function(BenchmarkId::new("simd4_chunked", n), |b| {
            b.iter(|| {
                let mut time_chunks = t_vals.chunks_exact(4);
                let mut output_chunks = out_simd.chunks_exact_mut(24);
                for (times, output) in time_chunks.by_ref().zip(output_chunks.by_ref()) {
                    let [time0, time1, time2, time3] = times else {
                        continue;
                    };
                    let chunk_t = [*time0, *time1, *time2, *time3];
                    let mut chunk_out = [0.0_f64; 24];
                    equinoc_prop_step_simd4(
                        black_box(&equinoc),
                        black_box(&chunk_t),
                        black_box(t0),
                        black_box(&mut chunk_out),
                    );
                    output.copy_from_slice(&chunk_out);
                }
                // Scalar tail
                for (time, output) in time_chunks
                    .remainder()
                    .iter()
                    .zip(output_chunks.into_remainder().chunks_exact_mut(6))
                {
                    let mut single = [0.0_f64; 6];
                    equinoc2eci_impl_f64(
                        black_box(&equinoc),
                        6,
                        black_box(*time),
                        black_box(t0),
                        &mut single,
                    );
                    output.copy_from_slice(&single);
                }
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = simd_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(30);
    targets =
        bench_eci2equinoc_simd_vs_scalar,
        bench_equinoc2eci_simd_vs_scalar,
        bench_simd16_vs_scalar,
        bench_roundtrip_simd_vs_scalar,
        bench_simd_scaling,
        bench_equinoc_prop_step_threshold,
}

criterion_main!(simd_benches);
