use anyhow::{anyhow, ensure, Result};
use criterion::Criterion;
// Import through `two_phase_transfer_rs` re-exports so cargo resolves the
// same `oxymoo` instance as the lib (cargo issue #6313 mitigation).
use satpy_core::{eci2equinoc_impl, kep2eci_impl, MU};
use std::{
    hint::black_box,
    num::NonZeroUsize,
    sync::atomic::{AtomicBool, Ordering},
};
use two_phase_transfer_rs::batch_eci::{
    BatchEciConfiguration, BatchEciRequest, PopulationBatchEciRequest, TargetBodyForceBatchError,
};
use two_phase_transfer_rs::types::SearchDepthPolicy;
use two_phase_transfer_rs::VariableR2LambertScratch;
use two_phase_transfer_rs::{
    constellation_solve_batch_eci_precomputed,
    constellation_solve_population_batch_eci_precomputed,
    evaluate::evaluate_plan_branches_with_scratch, solve::FrontOutputMode, ExecutionPolicy,
    PlanContext, SamplingMode, TransferLocalOptimizerChoice, TransferLocalOptimizerConfig,
    TransferRequest,
};
use two_phase_transfer_rs::{LocalOptimizerKind, TuneLevel};

static TIMED_FAILURE: AtomicBool = AtomicBool::new(false);

/// The campaign's J2 closure settings, read from the config authority.
///
/// NOT `J2ClosureSettings::default()`: the default is gain 0.7 with a cap of 8,
/// the campaign is gain 1.0 with a cap of 5. Timing the default runs the J2
/// block several times more than production does, which inflates its share of
/// this benchmark and deflates every other share to match. Read the values,
/// never restate them, so a config change cannot leave this bench behind.
const fn campaign_j2_settings() -> two_phase_transfer_rs::solve::J2ClosureSettings {
    let controls = nd_config::CompiledPartAScienceV1::part_a_v1().mf_transfer();
    two_phase_transfer_rs::solve::J2ClosureSettings {
        max_iterations: controls.j2_max_iterations,
        endpoint_target_km: controls.j2_endpoint_target_km,
        correction_step_gain: controls.j2_correction_step_gain,
    }
}

fn kep_to_eci(kep: &[f64; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    kep2eci_impl(kep, false, 0.0, 0.0, false, &mut out);
    out
}

fn make_batch(batch_size: usize) -> Option<(Vec<[f64; 6]>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut one_constellation = Vec::with_capacity(15);
    for idx in 0..15 {
        let plane = f64::from(u8::try_from(idx % 5).ok()?);
        let slot = f64::from(u8::try_from(idx / 5).ok()?);
        let sat = [
            7000.0 + slot * 20.0,
            0.001,
            0.2 + plane * 0.01,
            plane * 0.25,
            0.0,
            slot * 0.35,
        ];
        one_constellation.push(kep_to_eci(&sat));
    }
    let satellite_capacity = batch_size.checked_mul(one_constellation.len())?;
    let target_capacity = batch_size.checked_mul(6)?;
    let mut satellites_eci = Vec::with_capacity(satellite_capacity);
    let mut target_one_values = Vec::with_capacity(target_capacity);
    let mut target_two_values = Vec::with_capacity(target_capacity);
    let mut epochs = Vec::with_capacity(batch_size);

    for idx in 0..batch_size {
        satellites_eci.extend_from_slice(&one_constellation);
        let index = f64::from(u32::try_from(idx).ok()?);
        let bias = index * 1.0e-4;
        let target_one_orbit = [7100.0 + bias, 0.002, 0.21, 0.1, 0.0, 0.2];
        let target_two_orbit = [7120.0 + bias, 0.002, 0.21, 0.1, 0.0, 0.25];
        target_one_values.extend_from_slice(&kep_to_eci(&target_one_orbit));
        target_two_values.extend_from_slice(&kep_to_eci(&target_two_orbit));
        epochs.push(index.mul_add(1.0e-5, 2_460_000.5));
    }
    Some((satellites_eci, target_one_values, target_two_values, epochs))
}

fn run_front_batch(
    satellites_eci: &[[f64; 6]],
    targets1: &[f64],
    targets2: &[f64],
    epochs: &[f64],
    front_output_mode: FrontOutputMode,
    max_revs: i32,
    search_depth: SearchDepthPolicy,
) -> Result<usize, TargetBodyForceBatchError> {
    let target_body_forces = vec![
        [two_phase_transfer_rs::types::BodyForceConfig::j2(
            two_phase_transfer_rs::types::BodyRole::DiagnosticTarget,
        ); 2];
        epochs.len()
    ];
    let fronts = constellation_solve_batch_eci_precomputed(BatchEciRequest {
        satellite_eci: satellites_eci,
        satellite_equinoctial: None,
        satellite_count: 15,
        configuration: BatchEciConfiguration {
            targets_one_eci: targets1,
            targets_two_eci: targets2,
            epoch_jds: epochs,
            max_time_s: 172_800.0,
            max_phase_dv: 0.5,
            max_transfer_dv: 2.0,
            max_revs,
            min_perigee: 6498.137,
            max_apogee: 41378.137,
            pairs_to_verify: 5,
            sampling_mode: SamplingMode::Fast,
            search_depth,
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            tof_penalty_weight: 0.1,
            revolution_cap: 2.0,
            target_propagation_authority:
                two_phase_transfer_rs::types::TargetPropagationAuthority::MfJ2,
            target_body_forces: &target_body_forces,
            force_config: None,
            require_high_fidelity: false,
            j2_closure_settings: campaign_j2_settings(),
            packed_coeffs: None,
            local_optimizer: TransferLocalOptimizerConfig {
                choice: TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
                tune: TuneLevel::Aggressive,
                seed: 42,
            },
            warm_starts: None,
            front_output_mode,
        },
    })?;
    if fronts
        .iter()
        .any(two_phase_transfer_rs::ConstellationTransferFront::is_empty)
    {
        return Ok(0);
    }
    Ok(fronts.iter().map(|front| front.candidates.len()).sum())
}

fn run_front_batch_default(
    satellites_eci: &[[f64; 6]],
    targets1: &[f64],
    targets2: &[f64],
    epochs: &[f64],
    front_output_mode: FrontOutputMode,
) -> Result<usize, TargetBodyForceBatchError> {
    run_front_batch(
        satellites_eci,
        targets1,
        targets2,
        epochs,
        front_output_mode,
        0,
        SearchDepthPolicy::default(),
    )
}

fn make_deep_selected_branch_context() -> PlanContext {
    let dep_r = 6_778.137;
    let dep_v = (MU / dep_r).sqrt();
    let dep_eci = [dep_r, 0.0, 0.0, 0.0, dep_v, 0.0];
    let tgt_r = 6_878.137;
    let tgt_v = (MU / tgt_r).sqrt();
    let tgt_eci = [tgt_r, 0.0, 0.0, 0.0, tgt_v, 0.0];
    let mut dep_equ = [0.0; 6];
    eci2equinoc_impl(&dep_eci, 6, 0.0, 0.0, &mut dep_equ);
    let mut tgt_equ = [0.0; 6];
    eci2equinoc_impl(&tgt_eci, 6, 0.0, 0.0, &mut tgt_equ);

    PlanContext::from_request(TransferRequest {
        dep_eci,
        dep_equ,
        epoch_jd: 2_451_545.0,
        tgt_eci,
        tgt_equ,
        max_time_s: 172_800.0,
        tof_penalty_weight: 0.1,
        revolution_cap: 100.0,
        max_phase_dv: 1.0,
        max_transfer_dv: 2.0,
        min_perigee: 6_578.137,
        max_apogee: 50_000.0,
        max_revs: 4,
        sampling_mode: SamplingMode::Fast,
        execution_policy: ExecutionPolicy::default(),
        distance_tol: 0.025,
        deployer_min_distance: 0.12,
        search_depth: SearchDepthPolicy {
            tof_sample_budget: 256,
            coarse_early_stop: false,
            fine_total_limit: 16,
            coarse_reject_margin_km_s: 0.15,
            seed_fine_margin_km_s: 0.15,
            ..SearchDepthPolicy::default()
        },
        local_optimizer: TransferLocalOptimizerConfig {
            choice: TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
            tune: TuneLevel::Aggressive,
            seed: 42,
        },
        ..TransferRequest::with_j2_closure_settings(campaign_j2_settings())
    })
}

fn run_deep_selected_branch(
    ctx: &PlanContext,
) -> Result<Option<NonZeroUsize>, two_phase_transfer_rs::evaluate::EvaluationArithmeticOverflow> {
    let mut lambert_scratch = VariableR2LambertScratch::default();
    let valid_branch_count =
        evaluate_plan_branches_with_scratch(&[0.05, 1.0, 0.05], ctx, false, &mut lambert_scratch)?
            .iter()
            .filter(|plan| plan.valid)
            .count();
    Ok(NonZeroUsize::new(valid_branch_count))
}

fn make_population_batch(
    design_count: usize,
    batch_size: usize,
) -> Option<(Vec<[f64; 6]>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let (base_sats, targets1, targets2, epochs) = make_batch(batch_size)?;
    let population_capacity = design_count.checked_mul(base_sats.len())?;
    let mut population = Vec::with_capacity(population_capacity);
    for design_idx in 0..design_count {
        let design_offset = f64::from(u32::try_from(design_idx).ok()?);
        for state in &base_sats {
            let [x, y, z, velocity_x, velocity_y, velocity_z] = *state;
            population.push([
                design_offset.mul_add(0.25, x),
                design_offset.mul_add(0.05, y),
                z,
                velocity_x,
                velocity_y,
                velocity_z,
            ]);
        }
    }
    Some((population, targets1, targets2, epochs))
}

fn run_population_front_batch(
    satellites_eci_population: &[[f64; 6]],
    design_count: usize,
    targets1: &[f64],
    targets2: &[f64],
    epochs: &[f64],
    max_revs: i32,
    search_depth: SearchDepthPolicy,
) -> Result<usize, TargetBodyForceBatchError> {
    let target_body_forces = vec![
        [two_phase_transfer_rs::types::BodyForceConfig::j2(
            two_phase_transfer_rs::types::BodyRole::DiagnosticTarget,
        ); 2];
        epochs.len()
    ];
    let fronts = constellation_solve_population_batch_eci_precomputed(PopulationBatchEciRequest {
        satellite_eci_population: satellites_eci_population,
        satellite_equinoctial_population: None,
        design_count,
        satellite_count: 15,
        configuration: BatchEciConfiguration {
            targets_one_eci: targets1,
            targets_two_eci: targets2,
            epoch_jds: epochs,
            max_time_s: 172_800.0,
            max_phase_dv: 0.5,
            max_transfer_dv: 2.0,
            max_revs,
            min_perigee: 6498.137,
            max_apogee: 41378.137,
            pairs_to_verify: 5,
            sampling_mode: SamplingMode::Fast,
            search_depth,
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            tof_penalty_weight: 0.1,
            revolution_cap: 2.0,
            target_propagation_authority:
                two_phase_transfer_rs::types::TargetPropagationAuthority::MfJ2,
            target_body_forces: &target_body_forces,
            force_config: None,
            require_high_fidelity: false,
            j2_closure_settings: campaign_j2_settings(),
            packed_coeffs: None,
            local_optimizer: TransferLocalOptimizerConfig {
                choice: TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
                tune: TuneLevel::Aggressive,
                seed: 42,
            },
            warm_starts: None,
            front_output_mode: FrontOutputMode::VerifiedSuperset,
        },
    })?;
    if fronts.iter().any(|design| {
        design
            .iter()
            .any(two_phase_transfer_rs::ConstellationTransferFront::is_empty)
    }) {
        return Ok(0);
    }
    Ok(fronts
        .iter()
        .flatten()
        .map(|front| front.candidates.len())
        .sum())
}

fn fixture(
    label: &str,
    batch_size: usize,
) -> Result<(Vec<[f64; 6]>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    make_batch(batch_size).ok_or_else(|| anyhow!("front-solve fixture overflow for {label}"))
}

fn population_fixture(
    design_count: usize,
    batch_size: usize,
) -> Result<(Vec<[f64; 6]>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    make_population_batch(design_count, batch_size)
        .ok_or_else(|| anyhow!("front-solve population fixture overflow"))
}

fn checked_batch_count(result: Result<usize, TargetBodyForceBatchError>) -> usize {
    match result {
        Ok(count) if count > 0 => count,
        Ok(_) | Err(_) => {
            TIMED_FAILURE.store(true, Ordering::Relaxed);
            0
        }
    }
}

fn preflight_batch(
    label: &str,
    fixture: &(Vec<[f64; 6]>, Vec<f64>, Vec<f64>, Vec<f64>),
    mode: FrontOutputMode,
    max_revs: i32,
    search_depth: SearchDepthPolicy,
) -> Result<()> {
    let count = run_front_batch(
        &fixture.0,
        &fixture.1,
        &fixture.2,
        &fixture.3,
        mode,
        max_revs,
        search_depth,
    )
    .map_err(|error| anyhow!("front-solve preflight failed for {label}: {error}"))?;
    ensure!(count > 0, "front-solve preflight empty for {label}");
    Ok(())
}

fn bench_front_solve_native(c: &mut Criterion) -> Result<()> {
    TIMED_FAILURE.store(false, Ordering::Relaxed);
    let mut group = c.benchmark_group("Front Solve Native");
    let fixture_single = fixture("batch_1", 1)?;
    let fixture_small = fixture("batch_4", 4)?;
    let fixture_medium = fixture("batch_8", 8)?;
    let fixture_large = fixture("batch_18", 18)?;
    let fixture_maximum = fixture("batch_32", 32)?;
    let population_d4_b18 = population_fixture(4, 18)?;
    let deep_selected_branch_ctx = make_deep_selected_branch_context();
    let default_search = SearchDepthPolicy::default();
    preflight_batch(
        "batch_1_verified_superset",
        &fixture_single,
        FrontOutputMode::VerifiedSuperset,
        0,
        default_search,
    )?;
    preflight_batch(
        "batch_4_verified_superset",
        &fixture_small,
        FrontOutputMode::VerifiedSuperset,
        0,
        default_search,
    )?;
    preflight_batch(
        "batch_8_verified_superset",
        &fixture_medium,
        FrontOutputMode::VerifiedSuperset,
        0,
        default_search,
    )?;
    preflight_batch(
        "batch_18_verified_superset",
        &fixture_large,
        FrontOutputMode::VerifiedSuperset,
        0,
        default_search,
    )?;
    preflight_batch(
        "batch_32_verified_superset",
        &fixture_maximum,
        FrontOutputMode::VerifiedSuperset,
        0,
        default_search,
    )?;
    preflight_batch(
        "batch_18_transfer_pareto",
        &fixture_large,
        FrontOutputMode::TransferPareto,
        0,
        default_search,
    )?;
    let deep_search = SearchDepthPolicy {
        tof_sample_budget: 256,
        coarse_early_stop: false,
        fine_total_limit: 16,
        coarse_reject_margin_km_s: 0.15,
        seed_fine_margin_km_s: 0.15,
        ..SearchDepthPolicy::default()
    };
    preflight_batch(
        "batch_18_verified_superset_deep_m4_tof256",
        &fixture_large,
        FrontOutputMode::VerifiedSuperset,
        4,
        deep_search,
    )?;
    preflight_batch(
        "batch_1_verified_superset_deep_m4_tof256",
        &fixture_single,
        FrontOutputMode::VerifiedSuperset,
        4,
        deep_search,
    )?;
    let deep_count = run_deep_selected_branch(&deep_selected_branch_ctx)
        .map_err(|error| anyhow!("front-solve selected-branch preflight failed: {error}"))?;
    ensure!(
        deep_count.is_some(),
        "front-solve selected-branch preflight returned no valid branch"
    );
    let population_count = run_population_front_batch(
        &population_d4_b18.0,
        4,
        &population_d4_b18.1,
        &population_d4_b18.2,
        &population_d4_b18.3,
        4,
        deep_search,
    )
    .map_err(|error| anyhow!("front-solve population preflight failed: {error}"))?;
    ensure!(
        population_count > 0,
        "front-solve population preflight empty"
    );
    group.bench_function("batch_1_verified_superset", |b| {
        b.iter(|| {
            black_box(checked_batch_count(run_front_batch_default(
                black_box(&fixture_single.0),
                black_box(&fixture_single.1),
                black_box(&fixture_single.2),
                black_box(&fixture_single.3),
                FrontOutputMode::VerifiedSuperset,
            )))
        });
    });
    group.bench_function("batch_4_verified_superset", |b| {
        b.iter(|| {
            black_box(checked_batch_count(run_front_batch_default(
                black_box(&fixture_small.0),
                black_box(&fixture_small.1),
                black_box(&fixture_small.2),
                black_box(&fixture_small.3),
                FrontOutputMode::VerifiedSuperset,
            )))
        });
    });
    group.bench_function("batch_8_verified_superset", |b| {
        b.iter(|| {
            black_box(checked_batch_count(run_front_batch_default(
                black_box(&fixture_medium.0),
                black_box(&fixture_medium.1),
                black_box(&fixture_medium.2),
                black_box(&fixture_medium.3),
                FrontOutputMode::VerifiedSuperset,
            )))
        });
    });
    group.bench_function("batch_18_verified_superset", |b| {
        b.iter(|| {
            black_box(checked_batch_count(run_front_batch_default(
                black_box(&fixture_large.0),
                black_box(&fixture_large.1),
                black_box(&fixture_large.2),
                black_box(&fixture_large.3),
                FrontOutputMode::VerifiedSuperset,
            )))
        });
    });
    group.bench_function("batch_32_verified_superset", |b| {
        b.iter(|| {
            black_box(checked_batch_count(run_front_batch_default(
                black_box(&fixture_maximum.0),
                black_box(&fixture_maximum.1),
                black_box(&fixture_maximum.2),
                black_box(&fixture_maximum.3),
                FrontOutputMode::VerifiedSuperset,
            )))
        });
    });
    group.bench_function("batch_18_transfer_pareto", |b| {
        b.iter(|| {
            black_box(checked_batch_count(run_front_batch_default(
                black_box(&fixture_large.0),
                black_box(&fixture_large.1),
                black_box(&fixture_large.2),
                black_box(&fixture_large.3),
                FrontOutputMode::TransferPareto,
            )))
        });
    });
    group.bench_function("batch_18_verified_superset_deep_m4_tof256", |b| {
        b.iter(|| {
            black_box(checked_batch_count(run_front_batch(
                black_box(&fixture_large.0),
                black_box(&fixture_large.1),
                black_box(&fixture_large.2),
                black_box(&fixture_large.3),
                FrontOutputMode::VerifiedSuperset,
                4,
                black_box(deep_search),
            )))
        });
    });
    group.bench_function("batch_1_verified_superset_deep_m4_tof256", |b| {
        b.iter(|| {
            black_box(checked_batch_count(run_front_batch(
                black_box(&fixture_single.0),
                black_box(&fixture_single.1),
                black_box(&fixture_single.2),
                black_box(&fixture_single.3),
                FrontOutputMode::VerifiedSuperset,
                4,
                black_box(deep_search),
            )))
        });
    });
    group.bench_function("prepared_branch_deep_m4_tof256_selected_branch", |b| {
        b.iter(
            || match run_deep_selected_branch(black_box(&deep_selected_branch_ctx)) {
                Ok(Some(count)) => black_box(count.get()),
                Ok(None) | Err(_) => {
                    TIMED_FAILURE.store(true, Ordering::Relaxed);
                    black_box(0)
                }
            },
        );
    });
    group.bench_function(
        "population_d4_b18_n15_verified_superset_deep_m4_tof256",
        |b| {
            b.iter(|| {
                black_box(checked_batch_count(run_population_front_batch(
                    black_box(&population_d4_b18.0),
                    4,
                    black_box(&population_d4_b18.1),
                    black_box(&population_d4_b18.2),
                    black_box(&population_d4_b18.3),
                    4,
                    black_box(deep_search),
                )))
            });
        },
    );
    group.finish();
    ensure!(
        !TIMED_FAILURE.load(Ordering::Relaxed),
        "front-solve benchmark produced an error or empty result"
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut criterion = Criterion::default().configure_from_args();
    bench_front_solve_native(&mut criterion)?;
    criterion.final_summary();
    Ok(())
}
