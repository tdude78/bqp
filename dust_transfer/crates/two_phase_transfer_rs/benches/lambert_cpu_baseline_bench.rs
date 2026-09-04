//! Lambert Halley CPU baseline bench.
//!
//! CPU baseline (the `householder_method` path of the absorbed Lambert solver,
//! reached through the `bench-internal` re-export) over a sweep of batch sizes.
//!
//! This lived in `satpy_core/benches/` while the solver was its own crate. It
//! names nothing from `satpy_core`; it was placed there beside the prospective
//! CUDA sibling it is written against.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_traits::ToPrimitive;
use std::hint::black_box;
use std::time::Duration;

// Synthetic LEO Lambert problem set. Avoids hard cases (multi-rev, near-
// parabolic) so all problems converge with the default Householder cap.
// The point of the bench is per-element throughput, not edge-case coverage.
const BATCH_SIZES: &[usize] = &[32, 64, 128, 256, 512, 1024, 2048, 4096, 16384];

fn make_lambert_inputs(n: usize) -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>) {
    // Deterministic LCG mirroring the test seed pattern in gpu/lambert.rs.
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut rng = || -> f64 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 11).to_f64().unwrap_or_default() / (1_u64 << 53).to_f64().unwrap_or(1.0)
    };
    let mut ll = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    for _ in 0..n {
        // ll in (-0.7, 0.7) avoids near-parabolic
        ll.push(rng() * 1.4 - 0.7);
        // t in [2, 8] non-dim — easy single-rev regime
        t.push(2.0 + rng() * 6.0);
    }
    let m = vec![0_i32; n];
    let low_path = vec![0_i32; n];
    (ll, t, m, low_path)
}

// ---------------------------------------------------------------------------
// CPU baseline. Uses izzo2015_impl_with_geom_fast indirectly
// via compute_lambert_geometry + the public function — that path includes
// the geometry computation, which our GPU port skips. Slightly unfair to
// the CPU side but it's what production actually calls.
//
// To bench the inner Householder isolated, we'd need to expose find_xy,
// which the solver module does not publish. Instead, this bench measures
// the full per-Lambert cost on CPU vs the find_xy-only kernel on GPU —
// the GPU's number is therefore a lower bound on its production
// improvement (since geometry stays on host either way).
// ---------------------------------------------------------------------------

fn bench_lambert_cpu(c: &mut Criterion) {
    use two_phase_transfer_rs::izzo2015_impl_with_geom_fast;

    let mut group = c.benchmark_group("lambert_batch");
    for &n in BATCH_SIZES {
        let (ll, t, _m, _low_path) = make_lambert_inputs(n);
        // Build N Lambert geometries with synthetic r1/r2. We can't use
        // compute_lambert_geometry directly (it derives ll from r1/r2),
        // so emulate by handing the kernel ll/t triples it computed
        // earlier. For CPU side, we synthesise a LambertGeometry per
        // problem from the (ll, t) — close enough to the production path
        // for a relative timing comparison.
        let geoms: Vec<two_phase_transfer_rs::LambertGeometry> = ll
            .into_iter()
            .zip(t)
            .map(|(ll_base, t_nd)| two_phase_transfer_rs::LambertGeometry {
                r1_norm: 7000.0,
                r2_norm: 7000.0,
                c_norm: 1.0,
                s: 1.0,
                s_cubed: 1.0,
                ir1: nalgebra::Vector3::new(1.0, 0.0, 0.0),
                ir2: nalgebra::Vector3::new(0.0, 1.0, 0.0),
                it1_base: nalgebra::Vector3::new(0.0, 1.0, 0.0),
                it2_base: nalgebra::Vector3::new(-1.0, 0.0, 0.0),
                ll_base,
                gamma: 1.0,
                rho: 0.0,
                sigma: 1.0,
                t_nd,
                success: true,
            })
            .collect();

        group.throughput(Throughput::Elements(u64::try_from(n).unwrap_or(u64::MAX)));
        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |b, _| {
            b.iter(|| {
                let mut sink = 0.0_f64;
                for geom in geoms.iter().take(n) {
                    let r = izzo2015_impl_with_geom_fast(
                        black_box(geom),
                        0, // m
                        true,
                        false,
                        8,
                        1e-9,
                        1e-9,
                    );
                    sink += r.v1[0] + r.v2[0];
                }
                black_box(sink)
            });
        });
    }
    group.finish();
}

// ===========================================================================
// Phase E1 — coarse-stage realistic comparison (the go/no-go measurement).
//
// Models the production coarse stage of ONE deployer/target pair: `n_seeds`
// seeds, each sweeping `TOF_PER_SEED` times-of-flight. Total N = n_seeds *
// TOF_PER_SEED Lambert problems, all sharing the event epoch.
//
//   Path A (cpu_fragmented): production today. ONE
//     izzo2015_batch_tof_variable_r2_with_scratch SIMD call PER SEED
//     (N≈TOF_PER_SEED≈30 per call — below the GPU crossover). This is the
//     *real* CPU path: geometry computed inside, (prograde,retrograde)
//     branch sweep, best-of selection, all f64x4.
//
//   Path B (gpu_coalesced): proposed Phase E. Gather all N problems'
//     geometry on the host (compute_lambert_geometry), then ONE GPU find_xy
//     launch per branch (prograde ll, retrograde -ll) over the full N —
//     solidly in the GPU-wins regime.
//
// CRITICAL HONESTY: Path B as measured = host geometry + 2 GPU launches. It
// OMITS the host-side velocity reconstruction + best-of-branch selection that
// the CPU's BatchTofResult includes. So the measured A/B ratio is an UPPER
// BOUND on the realizable end-to-end speedup. E1 decision gate:
//   ratio < 2.0  -> definite no-go (even the optimistic bound loses).
//   ratio > 4.0  -> likely go; build the fuller Path B before committing E2.
//   2.0..=4.0    -> ambiguous; add reconstruction/best-of to B and re-measure.
//
// Sizes sweep both sides of the per-pair estimate (~2,160 = 72 seeds × 30)
// so the result is robust to whatever E1.1 instrumentation reports as the
// real per-pair N.
// ===========================================================================

// Earth GM (km^3/s^2). Local const keeps the bench self-contained; only the
// geometry *magnitudes* matter for a relative timing comparison.
const MU_EARTH: f64 = 398_600.441_8;
const COARSE_SEED_COUNTS: &[usize] = &[8, 18, 36, 72, 144];
const TOF_PER_SEED: usize = 30;

struct SeedWork {
    r1: [f64; 3],
    v1_ref: [f64; 3],
    r2_vec: Vec<[f64; 3]>,
    v2_refs: Vec<[f64; 3]>,
    tofs: Vec<f64>,
}

// Synthesise a realistic single-rev LEO coarse grid: each seed is a deployer
// state on a ~7000 km orbit at a fanned-out phase; each TOF sweeps the target
// (a ~7100 km orbit) propagated forward. Magnitudes/ll land in the easy
// single-rev regime so every problem converges under the default cap.
#[expect(
    clippy::suboptimal_flops,
    reason = "preserve the benchmark fixture's established floating-point operation order"
)]
fn make_coarse_workload(n_seeds: usize) -> Vec<SeedWork> {
    let r1_alt = 7000.0_f64;
    let r2_alt = 7100.0_f64;
    let v_circ = |r: f64| (MU_EARTH / r).sqrt();
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt()); // ~2.39996 rad
    let omega = v_circ(r2_alt) / r2_alt; // target mean motion (rad/s)
    let mut seeds = Vec::with_capacity(n_seeds);
    for seed_index in 0..n_seeds {
        let seed_index_f64 = seed_index.to_f64().unwrap_or_default();
        let theta = seed_index_f64 * golden;
        let (ct, st) = (theta.cos(), theta.sin());
        let zc = 0.05 * (seed_index_f64 * 0.7).sin(); // small out-of-plane wobble
        let r1 = [r1_alt * ct, r1_alt * st, r1_alt * zc];
        let vc1 = v_circ(r1_alt);
        let v1_ref = [-vc1 * st, vc1 * ct, 0.0];
        let mut r2_vec = Vec::with_capacity(TOF_PER_SEED);
        let mut v2_refs = Vec::with_capacity(TOF_PER_SEED);
        let mut tofs = Vec::with_capacity(TOF_PER_SEED);
        for tof_index in 0..TOF_PER_SEED {
            let tof = 1500.0 + tof_index.to_f64().unwrap_or_default() * 200.0; // 1500 .. 7300 s
            let phi = theta + 0.6 + omega * tof; // lead phase + propagation
            let (cp, sp) = (phi.cos(), phi.sin());
            r2_vec.push([r2_alt * cp, r2_alt * sp, r2_alt * zc]);
            let vc2 = v_circ(r2_alt);
            v2_refs.push([-vc2 * sp, vc2 * cp, 0.0]);
            tofs.push(tof);
        }
        seeds.push(SeedWork {
            r1,
            v1_ref,
            r2_vec,
            v2_refs,
            tofs,
        });
    }
    seeds
}

fn bench_coarse_cpu_fragmented(c: &mut Criterion) {
    use two_phase_transfer_rs::{
        izzo2015_batch_tof_variable_r2_with_scratch, VariableR2LambertScratch,
    };

    let mut group = c.benchmark_group("lambert_coarse_e1");
    for &n_seeds in COARSE_SEED_COUNTS {
        let work = make_coarse_workload(n_seeds);
        let n = n_seeds.saturating_mul(TOF_PER_SEED);
        group.throughput(Throughput::Elements(u64::try_from(n).unwrap_or(u64::MAX)));
        group.bench_with_input(BenchmarkId::new("cpu_fragmented", n), &n, |b, _| {
            let mut scratch = VariableR2LambertScratch::default();
            b.iter(|| {
                let mut sink = 0.0_f64;
                for sw in &work {
                    let results = izzo2015_batch_tof_variable_r2_with_scratch(
                        black_box(MU_EARTH),
                        black_box(&sw.r1),
                        black_box(&sw.r2_vec),
                        black_box(&sw.v1_ref),
                        black_box(&sw.v2_refs),
                        black_box(&sw.tofs),
                        0, // m_max: single-rev coarse grid
                        &mut scratch,
                    );
                    for r in results {
                        sink += r.v1[0] + r.v2[0];
                    }
                }
                black_box(sink)
            });
        });
    }
    group.finish();
}

// Host geometry gather alone — the irreducible, NON-coalesceable cost that
// BOTH the CPU path and the proposed GPU path must pay (the GPU kernel takes
// post-geometry (ll, t) as input). By Amdahl, geom_time / cpu_fragmented_time
// is the geometry fraction of the coarse stage; 1 / that fraction is the
// hard ceiling on any speedup from offloading only the Lambert *solve*.
// This runs on any platform (no CUDA) and can settle the go/no-go analytically.
fn bench_coarse_geometry_only(c: &mut Criterion) {
    use two_phase_transfer_rs::compute_lambert_geometry;

    let mut group = c.benchmark_group("lambert_coarse_e1");
    for &n_seeds in COARSE_SEED_COUNTS {
        let work = make_coarse_workload(n_seeds);
        let n = n_seeds.saturating_mul(TOF_PER_SEED);
        group.throughput(Throughput::Elements(u64::try_from(n).unwrap_or(u64::MAX)));
        group.bench_with_input(BenchmarkId::new("geometry_only", n), &n, |b, _| {
            b.iter(|| {
                let mut sink = 0.0_f64;
                for sw in &work {
                    for (r2, tof) in sw.r2_vec.iter().zip(&sw.tofs) {
                        let g = compute_lambert_geometry(
                            black_box(MU_EARTH),
                            black_box(&sw.r1),
                            black_box(r2),
                            black_box(*tof),
                        );
                        sink += g.ll_base + g.t_nd;
                    }
                }
                black_box(sink)
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = lambert_cpu;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(20);
    targets = bench_lambert_cpu, bench_coarse_cpu_fragmented, bench_coarse_geometry_only,
}

criterion_main!(lambert_cpu);
