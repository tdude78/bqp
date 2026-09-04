//! Benchmarks for propagation functions: f64 vs `DualVec` comparison.
//!
//! Tests equinoctial element propagation and coordinate transforms to verify
//! SIMD feature doesn't regress `DualVec` performance.
//!
//! ## Expected `DualVec` Overhead
//!
//! `DualVec` carries a value + 3-component gradient (4 x f64 per scalar).
//! Expected overhead: 3-4x compared to f64 due to:
//! - 4x memory bandwidth for gradient storage
//! - Additional arithmetic for chain rule propagation
//! - Trigonometric functions require extra derivative computation
//!
//! ## Running Benchmarks
//!
//! ```bash
//! # All propagation benchmarks
//! cargo bench -p satpy_core -- propagation
//!
//! # Just equinoctial propagation comparison
//! cargo bench -p satpy_core -- equinoc_prop
//!
//! # Test without SIMD feature (regression detection)
//! cargo bench -p satpy_core --no-default-features --features parallel -- propagation
//! ```

#[cfg(feature = "autodiff")]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "autodiff")]
use num_traits::ToPrimitive;
#[cfg(feature = "autodiff")]
use satpy_core::{
    eci2equinoc_impl, equinoc2eci_impl, equinoc_prop_from_impl, equinoc_prop_j2_batch_impl, DualVec,
};
#[cfg(feature = "autodiff")]
use std::time::Duration;

/// Convert f64 slice to `DualVec` array (constants, no gradient seeds)
#[cfg(feature = "autodiff")]
fn to_dualvec_6(v: &[f64; 6]) -> [DualVec; 6] {
    [
        DualVec::constant(v[0]),
        DualVec::constant(v[1]),
        DualVec::constant(v[2]),
        DualVec::constant(v[3]),
        DualVec::constant(v[4]),
        DualVec::constant(v[5]),
    ]
}

/// Create a realistic LEO equinoctial state
#[expect(
    clippy::suboptimal_flops,
    reason = "preserve the benchmark's established floating-point operation order"
)]
fn create_leo_equinoctial() -> [f64; 6] {
    // Convert from Keplerian: a=6778km, e=0.001, i=51.6deg, RAAN=0, argp=0, TA=0
    let semi_major_axis: f64 = 6778.0;
    let eccentricity: f64 = 0.001;
    let inclination: f64 = 51.6_f64.to_radians();
    let raan: f64 = 0.0;
    let argp: f64 = 0.0;
    let ta: f64 = 0.0;

    // Convert to equinoctial: p, f, g, h, k, L
    let semi_latus_rectum = semi_major_axis * (1.0 - eccentricity * eccentricity);
    let omega = raan + argp;
    let eccentricity_f = eccentricity * omega.cos();
    let eccentricity_g = eccentricity * omega.sin();
    let half_inclination = inclination / 2.0;
    let inclination_h = half_inclination.tan() * raan.cos();
    let inclination_k = half_inclination.tan() * raan.sin();
    let longitude = raan + argp + ta;

    [
        semi_latus_rectum,
        eccentricity_f,
        eccentricity_g,
        inclination_h,
        inclination_k,
        longitude,
    ]
}

/// Create a realistic GTO equinoctial state
#[expect(
    clippy::suboptimal_flops,
    reason = "preserve the benchmark's established floating-point operation order"
)]
fn create_gto_equinoctial() -> [f64; 6] {
    // GTO: a=24500km, e=0.73, i=28.5deg
    let semi_major_axis: f64 = 24500.0;
    let eccentricity: f64 = 0.73;
    let inclination: f64 = 28.5_f64.to_radians();
    let raan: f64 = 45.0_f64.to_radians();
    let argp: f64 = 180.0_f64.to_radians();
    let ta: f64 = 0.0;

    let semi_latus_rectum = semi_major_axis * (1.0 - eccentricity * eccentricity);
    let omega = raan + argp;
    let eccentricity_f = eccentricity * omega.cos();
    let eccentricity_g = eccentricity * omega.sin();
    let half_inclination = inclination / 2.0;
    let inclination_h = half_inclination.tan() * raan.cos();
    let inclination_k = half_inclination.tan() * raan.sin();
    let longitude = raan + argp + ta;

    [
        semi_latus_rectum,
        eccentricity_f,
        eccentricity_g,
        inclination_h,
        inclination_k,
        longitude,
    ]
}

/// Benchmark `equinoc_prop_from_impl`: f64 vs `DualVec` (single propagation)
#[cfg(feature = "autodiff")]
fn bench_equinoc_prop_f64_vs_dualvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("equinoc_prop_f64_vs_dualvec");

    let equ_f64 = create_leo_equinoctial();
    let equ_dual = to_dualvec_6(&equ_f64);
    let tof: f64 = 3600.0; // 1 hour in seconds

    // f64
    group.bench_function("f64_leo_1hr", |b| {
        b.iter(|| {
            let mut out = [0.0_f64; 6];
            equinoc_prop_from_impl(
                std::hint::black_box(&equ_f64),
                std::hint::black_box(tof),
                &mut out,
            );
            out
        });
    });

    // DualVec
    let tof_dual = DualVec::constant(tof);
    group.bench_function("dualvec_leo_1hr", |b| {
        b.iter(|| {
            let mut out = [DualVec::constant(0.0); 6];
            equinoc_prop_from_impl(
                std::hint::black_box(&equ_dual),
                std::hint::black_box(tof_dual),
                &mut out,
            );
            out
        });
    });

    group.finish();
}

/// Benchmark `equinoc_prop` for GTO orbit (more eccentric = more complex)
#[cfg(feature = "autodiff")]
fn bench_equinoc_prop_gto(c: &mut Criterion) {
    let mut group = c.benchmark_group("equinoc_prop_gto_f64_vs_dualvec");

    let equ_f64 = create_gto_equinoctial();
    let equ_dual = to_dualvec_6(&equ_f64);
    let tof: f64 = 3600.0;
    let tof_dual = DualVec::constant(tof);

    // f64
    group.bench_function("f64_gto_1hr", |b| {
        b.iter(|| {
            let mut out = [0.0_f64; 6];
            equinoc_prop_from_impl(
                std::hint::black_box(&equ_f64),
                std::hint::black_box(tof),
                &mut out,
            );
            out
        });
    });

    // DualVec
    group.bench_function("dualvec_gto_1hr", |b| {
        b.iter(|| {
            let mut out = [DualVec::constant(0.0); 6];
            equinoc_prop_from_impl(
                std::hint::black_box(&equ_dual),
                std::hint::black_box(tof_dual),
                &mut out,
            );
            out
        });
    });

    group.finish();
}

/// Benchmark batch propagation (100 time steps)
#[cfg(feature = "autodiff")]
fn bench_equinoc_prop_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("equinoc_prop_batch_f64_vs_dualvec");

    let equ_f64 = create_leo_equinoctial();
    let equ_dual = to_dualvec_6(&equ_f64);

    // 100 time steps from 0 to 1 orbital period (~90 min for LEO)
    let tofs: Vec<f64> = (0..100).map(|i| f64::from(i) * 54.0).collect(); // 54s steps

    // f64 batch
    group.bench_function("f64_100_steps", |b| {
        b.iter(|| {
            let mut out = [0.0_f64; 6];
            for &tof in &tofs {
                equinoc_prop_from_impl(&equ_f64, tof, &mut out);
                std::hint::black_box(out);
            }
        });
    });

    // DualVec batch
    group.bench_function("dualvec_100_steps", |b| {
        b.iter(|| {
            let mut out = [DualVec::constant(0.0); 6];
            for &tof in &tofs {
                let tof_dual = DualVec::constant(tof);
                equinoc_prop_from_impl(&equ_dual, tof_dual, &mut out);
                std::hint::black_box(out);
            }
        });
    });

    group.finish();
}

#[cfg(feature = "autodiff")]
fn bench_equinoc_prop_j2_batch_tails(c: &mut Criterion) {
    let mut group = c.benchmark_group("j2_batch_block4_tail");

    for n_states in [4_usize, 5, 32, 33, 128, 129] {
        let mut equinoc_matrix = vec![0.0_f64; n_states.saturating_mul(6)];
        let mut tofs = vec![0.0_f64; n_states];
        for (idx, (state, tof)) in equinoc_matrix
            .chunks_exact_mut(6)
            .zip(tofs.iter_mut())
            .enumerate()
        {
            let idx_f64 = idx.to_f64().unwrap_or_default();
            state.copy_from_slice(&[
                idx_f64.mul_add(10.0, 7000.0),
                idx_f64.mul_add(1.0e-5, 0.001),
                idx_f64.mul_add(1.0e-5, 0.002),
                idx_f64.mul_add(0.001, 0.10),
                idx_f64.mul_add(0.001, 0.20),
                idx_f64.mul_add(0.002, 0.30),
            ]);
            *tof = idx_f64.mul_add(120.0, 60.0);
        }

        group.bench_with_input(
            BenchmarkId::new("states", n_states),
            &(equinoc_matrix, tofs),
            |b, (equinoc_matrix, tofs)| {
                b.iter(|| {
                    let mut out = vec![0.0_f64; tofs.len().saturating_mul(6)];
                    equinoc_prop_j2_batch_impl(
                        std::hint::black_box(equinoc_matrix),
                        std::hint::black_box(tofs),
                        &mut out,
                    );
                    std::hint::black_box(out)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark eci2equinoc coordinate transform: f64 vs `DualVec`
#[cfg(feature = "autodiff")]
fn bench_eci2equinoc_f64_vs_dualvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("eci2equinoc_f64_vs_dualvec");

    // LEO ECI state: ~400km altitude, circular, 51.6 deg inclination
    let eci_f64: [f64; 6] = [6778.0, 0.0, 0.0, 0.0, 5.5, 5.5];
    let eci_dual = to_dualvec_6(&eci_f64);
    let epoch_seconds: f64 = 2_460_000.5 * 86400.0; // JD in seconds
    let reference_seconds: f64 = epoch_seconds;
    let epoch_dual = DualVec::constant(epoch_seconds);
    let reference_dual = DualVec::constant(reference_seconds);

    // f64
    group.bench_function("f64", |b| {
        b.iter(|| {
            let mut out = [0.0_f64; 6];
            eci2equinoc_impl(
                std::hint::black_box(&eci_f64),
                6,
                std::hint::black_box(epoch_seconds),
                std::hint::black_box(reference_seconds),
                &mut out,
            );
            out
        });
    });

    // DualVec
    group.bench_function("dualvec", |b| {
        b.iter(|| {
            let mut out = [DualVec::constant(0.0); 6];
            eci2equinoc_impl(
                std::hint::black_box(&eci_dual),
                6,
                std::hint::black_box(epoch_dual),
                std::hint::black_box(reference_dual),
                &mut out,
            );
            out
        });
    });

    group.finish();
}

/// Benchmark equinoc2eci coordinate transform: f64 vs `DualVec`
#[cfg(feature = "autodiff")]
fn bench_equinoc2eci_f64_vs_dualvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("equinoc2eci_f64_vs_dualvec");

    let equ_f64 = create_leo_equinoctial();
    let equ_dual = to_dualvec_6(&equ_f64);
    let epoch_seconds: f64 = 3600.0; // 1 hour
    let reference_seconds: f64 = 0.0;
    let epoch_dual = DualVec::constant(epoch_seconds);
    let reference_dual = DualVec::constant(reference_seconds);

    // f64
    group.bench_function("f64", |b| {
        b.iter(|| {
            let mut out = [0.0_f64; 6];
            equinoc2eci_impl(
                std::hint::black_box(&equ_f64),
                6,
                std::hint::black_box(epoch_seconds),
                std::hint::black_box(reference_seconds),
                &mut out,
            );
            out
        });
    });

    // DualVec
    group.bench_function("dualvec", |b| {
        b.iter(|| {
            let mut out = [DualVec::constant(0.0); 6];
            equinoc2eci_impl(
                std::hint::black_box(&equ_dual),
                6,
                std::hint::black_box(epoch_dual),
                std::hint::black_box(reference_dual),
                &mut out,
            );
            out
        });
    });

    group.finish();
}

/// Benchmark full propagation step: eci -> equinoc -> propagate -> eci
/// This is representative of what happens in HF propagation
#[cfg(feature = "autodiff")]
fn bench_full_propagation_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_step_f64_vs_dualvec");

    let eci_f64: [f64; 6] = [6778.0, 0.0, 0.0, 0.0, 5.5, 5.5];
    let eci_dual = to_dualvec_6(&eci_f64);
    let t0: f64 = 2_460_000.5 * 86400.0;
    let dt: f64 = 60.0; // 1 minute step
    let t0_dual = DualVec::constant(t0);
    let dt_dual = DualVec::constant(dt);

    // f64 full step
    group.bench_function("f64_full_step", |b| {
        b.iter(|| {
            let mut equ = [0.0_f64; 6];
            let mut eci_out = [0.0_f64; 6];
            // ECI -> Equinoctial
            eci2equinoc_impl(&eci_f64, 6, t0, t0, &mut equ);
            // Propagate
            equinoc_prop_from_impl(&equ, dt, &mut eci_out);
            eci_out
        });
    });

    // DualVec full step
    group.bench_function("dualvec_full_step", |b| {
        b.iter(|| {
            let mut equ = [DualVec::constant(0.0); 6];
            let mut eci_out = [DualVec::constant(0.0); 6];
            // ECI -> Equinoctial
            eci2equinoc_impl(&eci_dual, 6, t0_dual, t0_dual, &mut equ);
            // Propagate
            equinoc_prop_from_impl(&equ, dt_dual, &mut eci_out);
            eci_out
        });
    });

    group.finish();
}

/// Benchmark varying propagation times
#[cfg(feature = "autodiff")]
fn bench_equinoc_prop_times(c: &mut Criterion) {
    let mut group = c.benchmark_group("equinoc_prop_times_f64_vs_dualvec");

    let equ_f64 = create_leo_equinoctial();
    let equ_dual = to_dualvec_6(&equ_f64);

    for tof_hours in &[1.0_f64, 24.0] {
        let tof = tof_hours * 3600.0;
        let tof_dual = DualVec::constant(tof);

        group.bench_with_input(
            BenchmarkId::new("f64", format!("{tof_hours}hr")),
            &tof,
            |b, &tof| {
                b.iter(|| {
                    let mut out = [0.0_f64; 6];
                    equinoc_prop_from_impl(&equ_f64, tof, &mut out);
                    out
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dualvec", format!("{tof_hours}hr")),
            &tof_dual,
            |b, &tof_dual| {
                b.iter(|| {
                    let mut out = [DualVec::constant(0.0); 6];
                    equinoc_prop_from_impl(&equ_dual, tof_dual, &mut out);
                    out
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "autodiff")]
criterion_group! {
    name = propagation_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(20);
    targets =
        bench_equinoc_prop_f64_vs_dualvec,
        bench_equinoc_prop_gto,
        bench_equinoc_prop_batch,
        bench_equinoc_prop_j2_batch_tails,
        bench_equinoc_prop_times,
}

#[cfg(feature = "autodiff")]
criterion_group! {
    name = transform_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(30);
    targets =
        bench_eci2equinoc_f64_vs_dualvec,
        bench_equinoc2eci_f64_vs_dualvec,
        bench_full_propagation_step,
}

#[cfg(feature = "autodiff")]
criterion_main!(propagation_benches, transform_benches);

// When autodiff is not enabled, this benchmark file has no benchmarks to run.
// We still need a main function for cargo to compile it.
#[cfg(not(feature = "autodiff"))]
fn main() {
    eprintln!("propagation_bench requires the 'autodiff' feature. Run with: cargo bench -p satpy_core --features autodiff");
}
