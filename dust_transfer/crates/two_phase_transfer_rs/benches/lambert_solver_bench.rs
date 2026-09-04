//! Benchmark for Lambert solver performance.
//!
//! Run with:
//!   cargo bench -p two_phase_transfer_rs --features bench-internal \
//!     --bench lambert_solver_bench

use criterion::{criterion_group, criterion_main, Criterion};
use two_phase_transfer_rs::{
    izzo2015_batch_m_prograde, izzo2015_batch_tof, izzo2015_best_solution, izzo2015_impl,
    solve_lambert_batch_tof_variable_r2_branch_best_with_scratch, VariableR2LambertScratch,
};

// Was a hand-copied 398600.4418 (7.5 ppm off the production value); point at
// the one shared constant instead of re-typing it.
use satpy_core::MU;

// LEO transfer test case
const R1_LEO: [f64; 3] = [6778.0, 0.0, 0.0]; // km
const R2_LEO: [f64; 3] = [0.0, 7178.0, 0.0]; // km
const TOF_LEO: f64 = 3600.0; // 1 hour

fn leo_state1() -> [f64; 6] {
    let r = 6778.0;
    let v = (MU / r).sqrt();
    [r, 0.0, 0.0, 0.0, v, 0.0]
}

fn leo_state2() -> [f64; 6] {
    let r = 7178.0;
    let v = (MU / r).sqrt();
    [0.0, r, 0.0, -v, 0.0, 0.0]
}

fn bench_single_solve(c: &mut Criterion) {
    c.bench_function("izzo2015_impl_single", |b| {
        b.iter(|| {
            izzo2015_impl(
                std::hint::black_box(MU),
                std::hint::black_box(&R1_LEO),
                std::hint::black_box(&R2_LEO),
                std::hint::black_box(TOF_LEO),
                std::hint::black_box(0),
                std::hint::black_box(true),
                std::hint::black_box(true),
                std::hint::black_box(50),
                std::hint::black_box(1e-9),
                std::hint::black_box(1e-9),
            )
        });
    });
}

fn bench_batch_solve(c: &mut Criterion) {
    c.bench_function("izzo2015_batch_m_prograde_m0", |b| {
        b.iter(|| {
            izzo2015_batch_m_prograde(
                std::hint::black_box(MU),
                std::hint::black_box(&R1_LEO),
                std::hint::black_box(&R2_LEO),
                std::hint::black_box(TOF_LEO),
                std::hint::black_box(0), // m_max = 0 means 2 solves (prograde/retrograde)
                std::hint::black_box(true),
            )
        });
    });

    c.bench_function("izzo2015_batch_m_prograde_m1", |b| {
        b.iter(|| {
            izzo2015_batch_m_prograde(
                std::hint::black_box(MU),
                std::hint::black_box(&R1_LEO),
                std::hint::black_box(&R2_LEO),
                std::hint::black_box(TOF_LEO),
                std::hint::black_box(1), // m_max = 1 means 4 solves
                std::hint::black_box(true),
            )
        });
    });

    c.bench_function("izzo2015_batch_m_prograde_m3", |b| {
        b.iter(|| {
            izzo2015_batch_m_prograde(
                std::hint::black_box(MU),
                std::hint::black_box(&R1_LEO),
                std::hint::black_box(&R2_LEO),
                std::hint::black_box(10800.0), // longer TOF for higher m
                std::hint::black_box(3),       // m_max = 3 means 8 solves
                std::hint::black_box(true),
            )
        });
    });

    c.bench_function("izzo2015_batch_m_prograde_m4_deep", |b| {
        b.iter(|| {
            izzo2015_batch_m_prograde(
                std::hint::black_box(MU),
                std::hint::black_box(&R1_LEO),
                std::hint::black_box(&R2_LEO),
                std::hint::black_box(14400.0), // deep solver max-rev 4 shape
                std::hint::black_box(4),       // m_max = 4 means 10 branches
                std::hint::black_box(true),
            )
        });
    });
}

fn bench_best_solution(c: &mut Criterion) {
    let state1 = leo_state1();
    let state2 = leo_state2();

    c.bench_function("izzo2015_best_solution_m1", |b| {
        b.iter(|| {
            izzo2015_best_solution(
                std::hint::black_box(MU),
                std::hint::black_box(&state1),
                std::hint::black_box(&state2),
                std::hint::black_box(TOF_LEO),
                std::hint::black_box(1),
                std::hint::black_box(true),
            )
        });
    });

    c.bench_function("izzo2015_best_solution_m4_deep", |b| {
        b.iter(|| {
            izzo2015_best_solution(
                std::hint::black_box(MU),
                std::hint::black_box(&state1),
                std::hint::black_box(&state2),
                std::hint::black_box(14400.0),
                std::hint::black_box(4),
                std::hint::black_box(true),
            )
        });
    });
}

#[expect(
    clippy::suboptimal_flops,
    reason = "benchmark inputs preserve the established floating-point sequence"
)]
fn bench_variable_r2_branch_best_m0(c: &mut Criterion) {
    let state1 = leo_state1();
    let state2 = leo_state2();
    let r1 = [state1[0], state1[1], state1[2]];
    let v1_ref = [state1[3], state1[4], state1[5]];
    let mut r2_rows = Vec::with_capacity(65);
    let mut v2_rows = Vec::with_capacity(65);
    let mut tofs = Vec::with_capacity(65);
    for row in 0..65 {
        let angle = 0.01 * f64::from(row);
        let (sin_a, cos_a) = angle.sin_cos();
        r2_rows.push([
            state2[0] * cos_a - state2[1] * sin_a,
            state2[0] * sin_a + state2[1] * cos_a,
            5.0 * f64::from(row % 7),
        ]);
        v2_rows.push([
            state2[3] * cos_a - state2[4] * sin_a,
            state2[3] * sin_a + state2[4] * cos_a,
            state2[5],
        ]);
        tofs.push(600.0 + 15.0 * f64::from(row));
    }
    let mut scratch = VariableR2LambertScratch::default();

    c.bench_function("variable_r2_branch_best_65_m0_winner_max_revs4", |b| {
        b.iter(|| {
            let rows = solve_lambert_batch_tof_variable_r2_branch_best_with_scratch(
                std::hint::black_box(MU),
                std::hint::black_box(&r1),
                std::hint::black_box(&r2_rows),
                std::hint::black_box(&v1_ref),
                std::hint::black_box(&v2_rows),
                std::hint::black_box(&tofs),
                std::hint::black_box(4),
                std::hint::black_box(true),
                std::hint::black_box(None),
                std::hint::black_box(&mut scratch),
            );
            std::hint::black_box(rows.iter().filter(|row| row.valid).count())
        });
    });
}

#[expect(
    clippy::suboptimal_flops,
    reason = "benchmark inputs preserve the established floating-point sequence"
)]
fn bench_variable_r2_selected_branch(c: &mut Criterion) {
    let state1 = leo_state1();
    let state2 = leo_state2();
    let r1 = [state1[0], state1[1], state1[2]];
    let v1_ref = [state1[3], state1[4], state1[5]];
    let mut r2_rows = Vec::with_capacity(65);
    let mut v2_rows = Vec::with_capacity(65);
    let mut tofs = Vec::with_capacity(65);
    for row in 0..65 {
        let angle = 0.01 * f64::from(row);
        let (sin_a, cos_a) = angle.sin_cos();
        r2_rows.push([
            state2[0] * cos_a - state2[1] * sin_a,
            state2[0] * sin_a + state2[1] * cos_a,
            5.0 * f64::from(row % 7),
        ]);
        v2_rows.push([
            state2[3] * cos_a - state2[4] * sin_a,
            state2[3] * sin_a + state2[4] * cos_a,
            state2[5],
        ]);
        tofs.push(43_200.0 + 90.0 * f64::from(row));
    }
    let mut scratch = VariableR2LambertScratch::default();

    c.bench_function("variable_r2_selected_branch", |b| {
        b.iter(|| {
            let rows = solve_lambert_batch_tof_variable_r2_branch_best_with_scratch(
                std::hint::black_box(MU),
                std::hint::black_box(&r1),
                std::hint::black_box(&r2_rows),
                std::hint::black_box(&v1_ref),
                std::hint::black_box(&v2_rows),
                std::hint::black_box(&tofs),
                std::hint::black_box(4),
                std::hint::black_box(true),
                std::hint::black_box(Some((4, true))),
                std::hint::black_box(&mut scratch),
            );
            std::hint::black_box(rows.iter().filter(|row| row.valid).count())
        });
    });
}

/// Benchmark batch TOF solving with seeded iterations.
///
/// This benchmarks the performance of solving Lambert problems for multiple TOFs
/// using the seeded approach, where previous solutions seed the next solve.
#[expect(
    clippy::suboptimal_flops,
    reason = "benchmark inputs preserve the established floating-point sequence"
)]
fn bench_batch_tof(c: &mut Criterion) {
    // Reference velocities for delta-V calculation (circular orbit velocities)
    let v1_ref: [f64; 3] = [0.0, (MU / 6778.0_f64).sqrt(), 0.0];
    let v2_ref: [f64; 3] = [-(MU / 7178.0_f64).sqrt(), 0.0, 0.0];

    // Create 100 TOFs: 1 hour to ~2.67 hours, 1-minute increments
    let tofs_100: Vec<f64> = (0..100).map(|i| 3600.0 + f64::from(i) * 60.0).collect();

    // Benchmark batch_tof with 100 TOFs, m_max=0 (2 solves per TOF: prograde/retrograde)
    // This is a fair comparison with 100 calls to batch_m_prograde(m_max=0)
    c.bench_function("izzo2015_batch_tof_100_m0", |b| {
        b.iter(|| {
            izzo2015_batch_tof(
                std::hint::black_box(MU),
                std::hint::black_box(&R1_LEO),
                std::hint::black_box(&R2_LEO),
                std::hint::black_box(&tofs_100),
                std::hint::black_box(0), // m_max=0 for fair comparison
                std::hint::black_box(Some(&v1_ref)),
                std::hint::black_box(Some(&v2_ref)),
            )
        });
    });

    // Equivalent: 100 separate batch_m_prograde calls (geometry recomputed each time, no seeding)
    // This shows the combined benefit of geometry reuse + seeding
    c.bench_function("izzo2015_batch_m_x100_unseeded", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(100);
            for tof in &tofs_100 {
                let res = izzo2015_batch_m_prograde(
                    std::hint::black_box(MU),
                    std::hint::black_box(&R1_LEO),
                    std::hint::black_box(&R2_LEO),
                    std::hint::black_box(*tof),
                    std::hint::black_box(0), // m_max=0 matches batch_tof
                    std::hint::black_box(true),
                );
                results.push(res);
            }
            results
        });
    });

    // Also benchmark with m_max=1 to show multi-rev handling
    c.bench_function("izzo2015_batch_tof_100_m1", |b| {
        b.iter(|| {
            izzo2015_batch_tof(
                std::hint::black_box(MU),
                std::hint::black_box(&R1_LEO),
                std::hint::black_box(&R2_LEO),
                std::hint::black_box(&tofs_100),
                std::hint::black_box(1), // m_max=1
                std::hint::black_box(Some(&v1_ref)),
                std::hint::black_box(Some(&v2_ref)),
            )
        });
    });

    c.bench_function("izzo2015_batch_tof_256_m4_deep", |b| {
        let tofs_256: Vec<f64> = (0..256).map(|i| 1800.0 + f64::from(i) * 90.0).collect();
        b.iter(|| {
            izzo2015_batch_tof(
                std::hint::black_box(MU),
                std::hint::black_box(&R1_LEO),
                std::hint::black_box(&R2_LEO),
                std::hint::black_box(&tofs_256),
                std::hint::black_box(4),
                std::hint::black_box(Some(&v1_ref)),
                std::hint::black_box(Some(&v2_ref)),
            )
        });
    });

    // Single solve baseline for reference
    c.bench_function("izzo2015_single_x100", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(100);
            for tof in &tofs_100 {
                let res = izzo2015_impl(
                    std::hint::black_box(MU),
                    std::hint::black_box(&R1_LEO),
                    std::hint::black_box(&R2_LEO),
                    std::hint::black_box(*tof),
                    std::hint::black_box(0),
                    std::hint::black_box(true),
                    std::hint::black_box(true),
                    std::hint::black_box(50),
                    std::hint::black_box(1e-9),
                    std::hint::black_box(1e-9),
                );
                results.push(res);
            }
            results
        });
    });
}

criterion_group!(
    benches,
    bench_single_solve,
    bench_batch_solve,
    bench_best_solution,
    bench_batch_tof,
    bench_variable_r2_branch_best_m0,
    bench_variable_r2_selected_branch,
);

criterion_main!(benches);
