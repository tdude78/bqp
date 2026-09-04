use criterion::{criterion_group, criterion_main, Criterion};
use two_phase_transfer_rs::{
    izzo2015_batch_tof_variable_r2, izzo2015_batch_tof_variable_r2_with_scratch,
    VariableR2LambertScratch,
};

// Was a hand-copied 398600.4418 (7.5 ppm off the production value); point at
// the one shared constant instead of re-typing it.
use satpy_core::MU;

fn usize_to_f64(value: usize) -> f64 {
    u32::try_from(value).map_or(f64::INFINITY, f64::from)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "benchmark inputs preserve the established floating-point sequence"
)]
fn make_test_data(n: usize) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<f64>) {
    // Fixed departure in LEO
    let r1: [f64; 3] = [6778.0, 0.0, 0.0];

    // Sweep target positions at different orbital phases
    let r2_vec: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let angle = usize_to_f64(i) * std::f64::consts::PI / usize_to_f64(n);
            let r = 7178.0 + usize_to_f64(i) * 10.0;
            [r * angle.cos(), r * angle.sin(), 0.0]
        })
        .collect();

    // TOF sweep from 30 min to 8 hours
    let tofs: Vec<f64> = (0..n)
        .map(|i| 1800.0 + usize_to_f64(i) * (28800.0 - 1800.0) / usize_to_f64(n.saturating_sub(1)))
        .collect();

    (vec![r1; n], r2_vec, tofs)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "benchmark inputs preserve the established floating-point sequence"
)]
fn bench_n(c: &mut Criterion, n: usize, label: &str) {
    let (r1_vec, r2_vec, tofs) = make_test_data(n);
    let r1 = r1_vec.first().copied().unwrap_or([f64::NAN; 3]);

    let v_circ = (MU / 6778.0_f64).sqrt();
    let v1_ref: [f64; 3] = [0.0, v_circ, 0.0];
    let v2_refs: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let angle = usize_to_f64(i) * std::f64::consts::PI / usize_to_f64(n);
            let r = 7178.0 + usize_to_f64(i) * 10.0;
            let v = (MU / r).sqrt();
            [-v * angle.sin(), v * angle.cos(), 0.0]
        })
        .collect();

    c.bench_function(label, |b| {
        b.iter(|| {
            izzo2015_batch_tof_variable_r2(
                std::hint::black_box(MU),
                std::hint::black_box(&r1),
                std::hint::black_box(&r2_vec),
                std::hint::black_box(&v1_ref),
                std::hint::black_box(&v2_refs),
                std::hint::black_box(&tofs),
                std::hint::black_box(0),
            )
        });
    });
}

#[expect(
    clippy::suboptimal_flops,
    reason = "benchmark inputs preserve the established floating-point sequence"
)]
fn bench_alloc_vs_scratch(c: &mut Criterion, n: usize, m_max: i32) {
    let (r1_vec, r2_vec, tofs) = make_test_data(n);
    let r1 = r1_vec.first().copied().unwrap_or([f64::NAN; 3]);
    let v_circ = (MU / 6778.0_f64).sqrt();
    let v1_ref: [f64; 3] = [0.0, v_circ, 0.0];
    let v2_refs: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let angle = usize_to_f64(i) * std::f64::consts::PI / usize_to_f64(n);
            let r = 7178.0 + usize_to_f64(i) * 10.0;
            let v = (MU / r).sqrt();
            [-v * angle.sin(), v * angle.cos(), 0.0]
        })
        .collect();
    let mut scratch = VariableR2LambertScratch::default();
    let mut group = c.benchmark_group("Lambert Variable R2 Scratch");
    group.bench_function(format!("alloc_n{n}_m{m_max}"), |b| {
        b.iter(|| {
            izzo2015_batch_tof_variable_r2(
                std::hint::black_box(MU),
                std::hint::black_box(&r1),
                std::hint::black_box(&r2_vec),
                std::hint::black_box(&v1_ref),
                std::hint::black_box(&v2_refs),
                std::hint::black_box(&tofs),
                std::hint::black_box(m_max),
            )
            .len()
        });
    });
    group.bench_function(format!("scratch_n{n}_m{m_max}"), |b| {
        b.iter(|| {
            izzo2015_batch_tof_variable_r2_with_scratch(
                std::hint::black_box(MU),
                std::hint::black_box(&r1),
                std::hint::black_box(&r2_vec),
                std::hint::black_box(&v1_ref),
                std::hint::black_box(&v2_refs),
                std::hint::black_box(&tofs),
                std::hint::black_box(m_max),
                std::hint::black_box(&mut scratch),
            )
            .len()
        });
    });
    group.finish();
}

fn bench_batch_tof_variable_r2(c: &mut Criterion) {
    // 16 TOFs: 4 SIMD chunks, no scalar tail
    bench_n(c, 16, "izzo2015_batch_tof_variable_r2_n16_simd");
    // 17 TOFs: 4 SIMD chunks + 1 scalar tail
    bench_n(c, 17, "izzo2015_batch_tof_variable_r2_n17_mixed");
    // 64 TOFs: 16 SIMD chunks, no scalar tail
    bench_n(c, 64, "izzo2015_batch_tof_variable_r2_n64_simd");
    // 65 TOFs: 16 SIMD chunks + 1 scalar tail
    bench_n(c, 65, "izzo2015_batch_tof_variable_r2_n65_mixed");
    // 3 TOFs: pure scalar tail (no full SIMD chunks)
    bench_n(c, 3, "izzo2015_batch_tof_variable_r2_n3_scalar_tail");
    bench_alloc_vs_scratch(c, 64, 2);
    bench_alloc_vs_scratch(c, 256, 4);
}

criterion_group!(benches, bench_batch_tof_variable_r2);
criterion_main!(benches);
