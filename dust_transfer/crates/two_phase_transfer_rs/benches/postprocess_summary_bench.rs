use criterion::{criterion_group, criterion_main, Criterion};
use satpy_core::{kep2eci_impl, SEC_PER_DAY};
use std::sync::Once;
use two_phase_transfer_rs::batch_eci::{BatchEciConfiguration, BatchEciRequest};
use two_phase_transfer_rs::solve::FrontOutputMode;
use two_phase_transfer_rs::types::{
    BodyForceConfig, BodyRole, SearchDepthPolicy, TargetPropagationAuthority,
};
use two_phase_transfer_rs::{
    batch_postprocess_compact_candidates, constellation_solve_batch_eci_precomputed,
    CompactBatchPostprocessInputs, CompactBatchPostprocessOutputs, CompactBatchTargetPhysics,
    CompactTransferCandidate, PhysicsConfig, PostprocessConfig, SamplingMode,
    TransferLocalOptimizerConfig,
};

static GRAVITY_COEFF_INIT: Once = Once::new();

fn install_bench_gravity_coeffs() {
    GRAVITY_COEFF_INIT.call_once(|| {
        lightyear_odeint_rs::load_constants_from_bytes(
            include_bytes!("../data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt"),
            5,
        )
        .expect("embedded DIR-R6 d15 bench gravity coefficients should load");
    });
}

/// The campaign's J2 closure settings, read from the config authority.
///
/// NOT `J2ClosureSettings::default()`: the default is gain 0.7 with a cap of 8,
/// the campaign is gain 1.0 with a cap of 5. Timing the default runs the J2
/// block several times more than production does, which inflates its share of
/// this benchmark and deflates every other share to match. Read the values,
/// never restate them, so a config change cannot leave this bench behind.
fn campaign_j2_settings() -> two_phase_transfer_rs::solve::J2ClosureSettings {
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

fn make_fixture(
    seed: u64,
) -> (
    CompactTransferCandidate,
    [f64; 6],
    f64,
    f64,
    PhysicsConfig,
    PostprocessConfig,
) {
    let alt_bias = (seed % 3) as f64 * 20.0;
    let inc_bias = (seed % 5) as f64 * 0.0025;
    let satellites = [[7000.0 + alt_bias, 0.001, 0.2 + inc_bias, 0.0, 0.0, 0.0]];
    let target1 = [7100.0 + alt_bias, 0.002, 0.21 + inc_bias, 0.1, 0.0, 0.2];
    let target2 = [7120.0 + alt_bias, 0.002, 0.21 + inc_bias, 0.1, 0.0, 0.25];

    let satellites_eci = [kep_to_eci(&satellites[0])];
    let target1_eci = kep_to_eci(&target1);
    let target2_eci = kep_to_eci(&target2);
    let epoch_jds = [2_460_000.5];
    let target_body_forces = [[BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2]];
    let front = constellation_solve_batch_eci_precomputed(BatchEciRequest {
        satellite_eci: &satellites_eci,
        satellite_equinoctial: None,
        satellite_count: satellites_eci.len(),
        configuration: BatchEciConfiguration {
            targets_one_eci: &target1_eci,
            targets_two_eci: &target2_eci,
            epoch_jds: &epoch_jds,
            max_time_s: 7_200.0,
            max_phase_dv: 0.5,
            max_transfer_dv: 2.0,
            max_revs: 0,
            min_perigee: 6_578.14,
            max_apogee: 41_378.14,
            pairs_to_verify: 1,
            sampling_mode: SamplingMode::Fast,
            search_depth: SearchDepthPolicy::default(),
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            tof_penalty_weight: 0.1,
            revolution_cap: 1.5,
            target_propagation_authority: TargetPropagationAuthority::MfJ2,
            target_body_forces: &target_body_forces,
            force_config: None,
            require_high_fidelity: false,
            j2_closure_settings: campaign_j2_settings(),
            packed_coeffs: None,
            local_optimizer: TransferLocalOptimizerConfig::default(),
            warm_starts: None,
            front_output_mode: FrontOutputMode::TransferPareto,
        },
    })
    .expect("benchmark uses valid target propagation authority")
    .into_iter()
    .next()
    .expect("single-event benchmark must return one front");
    let candidate = front
        .candidates
        .first()
        .expect("expected valid postprocess bench candidate");
    let compact = CompactTransferCandidate::from_constellation_candidate(candidate)
        .expect("expected valid bench candidate to convert to compact payload");
    let intercept_state = compact.target_intercept_state;
    let intercept_jd = compact.solver_intercept_jd;
    let conjunction_jd = intercept_jd + 600.0 / SEC_PER_DAY;

    let physics = PhysicsConfig {
        max_phase_dv: 0.5,
        max_transfer_dv: 2.0,
        max_time_s: 86400.0,
        min_miss_distance_km: 1.0,
        event_rewind_days: 3.0,
        dust_pos_sigma_m: 50.0,
        dust_vel_sigma_mps: 0.05,
        hit_probability: 0.9,
        kappa: 2.0,
        use_high_fidelity: false,
        splitting_criterion: "maxvar".to_string(),
        tof_penalty_weight: 0.1,
        revolution_cap: 1.5,
        distance_tol: 0.025,
        deployer_min_distance: 0.12,
        ..PhysicsConfig::default()
    };
    let postprocess = PostprocessConfig {
        fix_ls_max_nfev: 40,
        fix_ls_tol: 1e-6,
        fix_ls_skip_tol: 0.0,
        dust_intercept_tol_km: 0.01,
        dust_radial_samples: 24,
        dust_angular_samples: 100,
        gmm_components: 3,
        max_physical_dv_kms: 7.5,
        min_practical_dust_mass_kg: 0.01,
        mf_seed_bound_kms: 0.1,
        hf_refine_bound_kms: 1.0,
        mf_seed_reg_weight: 1e-3,
        hf_refine_reg_weight: 1e-4,
        mf_seed_max_bound_expansions: 3,
        hf_refine_max_bound_expansions: 7,
        hybrid_mf_seed_hf_refine: true,
        dust_phase_tof_s: 7200.0,
        canister_tof_fraction: 0.0,
        canister_am: 0.01,
        canister_cd: 2.2,
        canister_cr: 1.3,
    };
    (
        compact,
        intercept_state,
        intercept_jd,
        conjunction_jd,
        physics,
        postprocess,
    )
}

fn run_batch(batch_size: usize, use_high_fidelity: bool) {
    let fixture_a = make_fixture(0);
    let fixture_b = make_fixture(17);
    let mut candidates = Vec::with_capacity(batch_size);
    let mut primary_states = vec![0.0_f64; batch_size * 6];
    let mut secondary_states = vec![0.0_f64; batch_size * 6];
    let mut intercept_jds = vec![0.0_f64; batch_size];
    let mut conjunction_jds = vec![0.0_f64; batch_size];

    for idx in 0..batch_size {
        let (candidate, intercept_state, intercept_jd, conjunction_jd, ..) =
            if idx % 2 == 0 { &fixture_a } else { &fixture_b };
        candidates.push(Some(candidate.clone()));
        let base = idx * 6;
        primary_states[base..base + 6].copy_from_slice(intercept_state);
        secondary_states[base..base + 6].copy_from_slice(intercept_state);
        intercept_jds[idx] = *intercept_jd;
        conjunction_jds[idx] = *conjunction_jd;
    }

    let (_, _, _, _, mut physics, postprocess) = fixture_a;
    physics.use_high_fidelity = use_high_fidelity;
    if use_high_fidelity {
        install_bench_gravity_coeffs();
        physics.sph_order = 5;
        physics.atm_model = 3;
        physics.tolerance = 1e-7;
    }
    let mut corrected_states = vec![f64::NAN; batch_size * 6];
    let mut correction_dvs = vec![f64::NAN; batch_size];
    let mut status_codes = vec![-1_i32; batch_size];
    let corrected = batch_postprocess_compact_candidates(
        CompactBatchPostprocessInputs {
            candidates: &candidates,
            primary_states: &primary_states,
            secondary_states: &secondary_states,
            intercept_jds: &intercept_jds,
            conjunction_jds: &conjunction_jds,
            physics_config: Some(physics),
            postprocess_config: Some(postprocess),
            primary_target: CompactBatchTargetPhysics::default(),
            secondary_target: CompactBatchTargetPhysics::default(),
        },
        CompactBatchPostprocessOutputs {
            corrected_states: &mut corrected_states,
            correction_dvs: &mut correction_dvs,
            status_codes: &mut status_codes,
        },
    )
    .expect("valid postprocess configuration");
    assert!(
        corrected > 0,
        "expected benchmark batch correction to succeed"
    );
}

fn bench_postprocess_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("Postprocess Summary");
    group.bench_function("batch_1", |b| b.iter(|| run_batch(1, false)));
    group.bench_function("batch_8", |b| b.iter(|| run_batch(8, false)));
    group.bench_function("batch_1_hf", |b| b.iter(|| run_batch(1, true)));
    group.bench_function("batch_8_hf", |b| b.iter(|| run_batch(8, true)));
    group.finish();
}

criterion_group!(benches, bench_postprocess_summary);
criterion_main!(benches);
