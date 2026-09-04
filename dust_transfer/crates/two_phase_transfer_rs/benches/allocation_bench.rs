use anyhow::{anyhow, ensure, Result};
use criterion::Criterion;
use satpy_core::{eci2equinoc_impl, MU};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use two_phase_transfer_rs::batch_eci::{BatchEciConfiguration, PopulationBatchEciRequest};
use two_phase_transfer_rs::solve::FrontOutputMode;
#[cfg(feature = "bench-internal")]
use two_phase_transfer_rs::solve::{bench_transfer_moo_policy_report, TransferMooBenchPolicy};
use two_phase_transfer_rs::types::SearchDepthPolicy;
use two_phase_transfer_rs::{
    constellation_solve_population_batch_eci_precomputed, solve_plan, ExecutionPolicy, PlanContext,
    SamplingMode, TransferLocalOptimizerChoice, TransferLocalOptimizerConfig, TransferRequest,
};
use two_phase_transfer_rs::{LocalOptimizerKind, TuneLevel};

struct CountingAlloc;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static TIMED_FAILURE: AtomicBool = AtomicBool::new(false);

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[derive(Clone, Copy, Debug)]
struct AllocStats {
    calls: u64,
    bytes: usize,
}

fn measure_allocs_with_value<T>(f: impl FnOnce() -> T) -> (AllocStats, T) {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    let value = f();
    std::hint::black_box(&value);
    let stats = AllocStats {
        calls: ALLOC_CALLS.load(Ordering::Relaxed),
        bytes: ALLOC_BYTES.load(Ordering::Relaxed),
    };
    std::hint::black_box(stats.calls);
    std::hint::black_box(stats.bytes);
    (std::hint::black_box(stats), value)
}

fn make_ctx(max_revs: i32, tof_samples: usize) -> PlanContext {
    let r = 6778.0;
    let v = (MU / r).sqrt();
    let dep_eci = [r, 0.0, 0.0, 0.0, v, 0.0];
    let r_tgt = 6878.0;
    let v_tgt = (MU / r_tgt).sqrt();
    let tgt_eci = [0.0, r_tgt, 0.0, -v_tgt, 0.0, 0.0];

    let mut dep_equ = [0.0; 6];
    eci2equinoc_impl(&dep_eci, 6, 0.0, 0.0, &mut dep_equ);
    let mut tgt_equ = [0.0; 6];
    eci2equinoc_impl(&tgt_eci, 6, 0.0, 0.0, &mut tgt_equ);

    PlanContext::from_request(TransferRequest {
        dep_eci,
        dep_equ,
        epoch_jd: 2_460_000.5,
        tgt_eci,
        tgt_equ,
        max_time_s: 86400.0,
        tof_penalty_weight: 0.1,
        revolution_cap: 1.5,
        max_phase_dv: 1.0,
        max_transfer_dv: 2.0,
        min_perigee: 6500.0,
        max_apogee: 50000.0,
        max_revs,
        sampling_mode: SamplingMode::Fast,
        execution_policy: ExecutionPolicy::default(),
        distance_tol: 0.025,
        deployer_min_distance: 0.12,
        search_depth: SearchDepthPolicy {
            tof_sample_budget: tof_samples,
            ..SearchDepthPolicy::default()
        },
        local_optimizer: TransferLocalOptimizerConfig {
            choice: TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
            tune: TuneLevel::Default,
            seed: 42,
        },
        ..TransferRequest::with_j2_closure_settings(
            two_phase_transfer_rs::J2ClosureSettings::default(),
        )
    })
}

fn circular_eci(radius_km: f64, phase_rad: f64) -> [f64; 6] {
    let v = (MU / radius_km).sqrt();
    [
        radius_km * phase_rad.cos(),
        radius_km * phase_rad.sin(),
        0.0,
        -v * phase_rad.sin(),
        v * phase_rad.cos(),
        0.0,
    ]
}

fn eci_to_equ(state: &[f64; 6]) -> [f64; 6] {
    let mut equ = [0.0; 6];
    eci2equinoc_impl(state, 6, 0.0, 0.0, &mut equ);
    equ
}

fn population_probe_inputs() -> (Vec<[f64; 6]>, Vec<[f64; 6]>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let batch_size = 4;
    let mut sats_eci = Vec::with_capacity(24);
    let mut sats_equ = Vec::with_capacity(24);
    for d in [0.0, 1.0] {
        for b in [0.0, 1.0, 2.0, 3.0] {
            for s in [0.0, 1.0, 2.0] {
                let radius_km = 10.0f64.mul_add(s, 25.0f64.mul_add(d, 6778.0));
                let phase_rad = 0.1f64.mul_add(s, 0.2 * b);
                let state = circular_eci(radius_km, phase_rad);
                sats_equ.push(eci_to_equ(&state));
                sats_eci.push(state);
            }
        }
    }
    let mut targets1 = Vec::with_capacity(24);
    let mut targets2 = Vec::with_capacity(24);
    let mut epochs = Vec::with_capacity(batch_size);
    for b in [0.0, 1.0, 2.0, 3.0] {
        targets1.extend_from_slice(&circular_eci(6878.0, 0.35 * b));
        targets2.extend_from_slice(&circular_eci(6978.0, 0.35f64.mul_add(b, 0.2)));
        epochs.push(b.mul_add(0.01, 2_460_000.5));
    }
    (sats_eci, sats_equ, targets1, targets2, epochs)
}

fn population_probe_allocs() -> Result<AllocStats> {
    let (sats_eci, sats_equ, targets1, targets2, epochs) = population_probe_inputs();
    let (stats, result) = measure_allocs_with_value(|| {
        let target_body_forces = vec![
            [two_phase_transfer_rs::types::BodyForceConfig::j2(
                two_phase_transfer_rs::types::BodyRole::DiagnosticTarget,
            ); 2];
            epochs.len()
        ];
        constellation_solve_population_batch_eci_precomputed(PopulationBatchEciRequest {
            satellite_eci_population: &sats_eci,
            satellite_equinoctial_population: Some(&sats_equ),
            design_count: 2,
            satellite_count: 3,
            configuration: BatchEciConfiguration {
                targets_one_eci: &targets1,
                targets_two_eci: &targets2,
                epoch_jds: &epochs,
                max_time_s: 0.02,
                max_phase_dv: 1.0,
                max_transfer_dv: 2.0,
                max_revs: 1,
                min_perigee: 6500.0,
                max_apogee: 50000.0,
                pairs_to_verify: 2,
                sampling_mode: SamplingMode::Fast,
                search_depth: SearchDepthPolicy {
                    tof_sample_budget: 16,
                    ..SearchDepthPolicy::default()
                },
                distance_tol: 0.025,
                deployer_min_distance: 0.12,
                tof_penalty_weight: 0.1,
                revolution_cap: 1.5,
                target_propagation_authority:
                    two_phase_transfer_rs::types::TargetPropagationAuthority::MfJ2,
                target_body_forces: &target_body_forces,
                force_config: None,
                require_high_fidelity: false,
                j2_closure_settings: two_phase_transfer_rs::solve::J2ClosureSettings::default(),
                packed_coeffs: None,
                local_optimizer: TransferLocalOptimizerConfig {
                    choice: TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
                    tune: TuneLevel::Default,
                    seed: 42,
                },
                warm_starts: None,
                front_output_mode: FrontOutputMode::TransferPareto,
            },
        })
    });
    let fronts = result.map_err(|error| anyhow!("population allocation probe failed: {error}"))?;
    ensure!(
        fronts.len() == 2,
        "population allocation probe returned wrong design count"
    );
    ensure!(
        fronts
            .iter()
            .all(|design| design.len() == 4 && design.iter().all(|front| !front.is_empty())),
        "population allocation probe returned empty or incomplete fronts"
    );
    Ok(stats)
}

#[cfg(feature = "bench-internal")]
fn oxymoo_materialization_probe_allocs(
    policy: TransferMooBenchPolicy,
) -> (
    AllocStats,
    Result<
        two_phase_transfer_rs::solve::TransferMooPolicyBenchReport,
        two_phase_transfer_rs::types::InvalidTargetPropagationAuthorityCode,
    >,
) {
    measure_allocs_with_value(|| bench_transfer_moo_policy_report(policy))
}

fn timed_plan_allocs(max_revs: i32, tof_samples: usize) -> AllocStats {
    let (stats, result) = measure_allocs_with_value(|| {
        let mut ctx = make_ctx(max_revs, tof_samples);
        solve_plan(std::hint::black_box(&mut ctx), None)
    });
    if !matches!(result, Ok(front) if !front.is_empty()) {
        TIMED_FAILURE.store(true, Ordering::Relaxed);
    }
    stats
}

fn timed_population_allocs() -> AllocStats {
    population_probe_allocs().unwrap_or_else(|_| {
        TIMED_FAILURE.store(true, Ordering::Relaxed);
        AllocStats { calls: 0, bytes: 0 }
    })
}

fn bench_transfer_allocations(c: &mut Criterion) -> Result<()> {
    TIMED_FAILURE.store(false, Ordering::Relaxed);
    let population_stats = population_probe_allocs()?;
    for &(max_revs, tof_samples) in &[(2, 64), (4, 256)] {
        let mut ctx = make_ctx(max_revs, tof_samples);
        let front = solve_plan(&mut ctx, None)
            .map_err(|error| anyhow!("allocation benchmark preflight failed: {error}"))?;
        ensure!(
            !front.is_empty(),
            "allocation benchmark preflight returned empty front"
        );
    }
    eprintln!(
        "allocation_probe name=population_precomputed_d2_b4_n3 rows=8 plan_clones=0 alloc_calls={} alloc_bytes={}",
        population_stats.calls, population_stats.bytes
    );
    #[cfg(feature = "bench-internal")]
    {
        let policy = TransferMooBenchPolicy::FastPopulation20Generations3InitialBest1;
        let (stats, report) = oxymoo_materialization_probe_allocs(policy);
        let report = report.map_err(|error| {
            anyhow!("OxyMOO allocation materialization probe failed for {policy:?}: {error}")
        })?;
        ensure!(
            report.front_candidate_count > 0,
            "OxyMOO allocation materialization probe returned empty front"
        );
        eprintln!(
                "allocation_probe name=oxymoo_materialization policy={:?} rows={} materialize_hits={} materialize_misses={} materialize_recompute={} materialize_all_exact={} old_clone_baseline_plan_clones={} plan_clones_after=0 plan_clones_avoided={} alloc_calls={} alloc_bytes={}",
                policy,
                report.front_candidate_count,
                report.materialize_plan_cache_hit_count,
                report.materialize_plan_cache_miss_count,
                report.materialize_recompute_count,
                report.materialize_all_exact_count,
                report.materialize_plan_cache_hit_count,
                report.materialize_plan_cache_hit_count,
                stats.calls,
                stats.bytes
        );
    }

    {
        let mut group = c.benchmark_group("transfer_allocations");
        group.bench_function("solve_plan_fast_m2_tof64", |b| {
            b.iter(|| timed_plan_allocs(2, 64));
        });

        group.bench_function("solve_plan_deep_m4_tof256", |b| {
            b.iter(|| timed_plan_allocs(4, 256));
        });

        group.bench_function("population_precomputed_d2_b4_n3", |b| {
            b.iter(timed_population_allocs);
        });

        group.finish();
    }
    ensure!(
        !TIMED_FAILURE.load(Ordering::Relaxed),
        "allocation benchmark produced an error or empty front"
    );
    Ok(())
}

fn main() -> Result<()> {
    {
        let mut criterion = Criterion::default().configure_from_args();
        bench_transfer_allocations(&mut criterion)?;
        criterion.final_summary();
    }
    Ok(())
}
