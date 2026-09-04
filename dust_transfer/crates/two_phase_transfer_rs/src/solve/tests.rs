// The `#[cfg(test)]` gating this file lives on the `mod tests;` declaration in
// `solve.rs`, so a file-content scan reads everything here as production. That
// misclassification blocked a real cut twice in one session. The inner
// attribute below makes the file self-describing to any classifier.
#![cfg(test)]

use super::*;
use satpy_core::MU;

#[expect(
    clippy::suboptimal_flops,
    reason = "test fixture preserves staged non-fused orbital rotation arithmetic"
)]
fn make_circular_state(alt_km: f64, inc_rad: f64, raan_rad: f64, arg_lat_rad: f64) -> [f64; 6] {
    let sma = 6378.137 + alt_km;
    let vel = (MU / sma).sqrt();

    let cos_raan = raan_rad.cos();
    let sin_raan = raan_rad.sin();
    let cos_inc = inc_rad.cos();
    let sin_inc = inc_rad.sin();
    let cos_u = arg_lat_rad.cos();
    let sin_u = arg_lat_rad.sin();

    let x = sma * (cos_raan * cos_u - sin_raan * sin_u * cos_inc);
    let y = sma * (sin_raan * cos_u + cos_raan * sin_u * cos_inc);
    let z = sma * sin_u * sin_inc;

    let vx = vel * (-cos_raan * sin_u - sin_raan * cos_u * cos_inc);
    let vy = vel * (-sin_raan * sin_u + cos_raan * cos_u * cos_inc);
    let vz = vel * cos_u * sin_inc;

    [x, y, z, vx, vy, vz]
}

fn next_deterministic_lcg_unit_interval(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let high_bits =
        u32::try_from(*state >> 33).expect("31-bit deterministic LCG output must fit u32");
    f64::from(high_bits) / f64::from(1_u32 << 31)
}

fn test_mf_configuration() -> ConstellationSolveConfiguration {
    ConstellationSolveConfiguration {
        max_time_s: 86_400.0,
        max_phase_dv: 1.0,
        max_transfer_dv: 2.0,
        max_revs: 1,
        min_perigee: 6500.0,
        max_apogee: 50_000.0,
        pairs_to_verify: 4,
        sampling_mode: SamplingMode::Fast,
        search_depth: SearchDepthPolicy::default(),
        epoch_jd: 0.0,
        distance_tol: 0.025,
        deployer_min_distance: 0.12,
        tof_penalty_weight: 0.1,
        revolution_cap: 2.0,
        target_propagation_authority: TargetPropagationAuthority::MfJ2,
        force_config: None,
        require_high_fidelity: false,
        j2_closure_settings: J2ClosureSettings::default(),
        packed_coeffs: None,
        local_optimizer: TransferLocalOptimizerConfig::default(),
        warm_start: None,
    }
}

#[test]
fn evaluation_diagnostic_merge_overflow_is_atomic_and_preserves_metric_bits() -> anyhow::Result<()>
{
    let mut metrics = VerifiedSupersetStageMetrics {
        lambert_batch_call_count: 7,
        branch_shared_prepare_count: usize::MAX,
        j2_correction_residual_m_sum: 1.5,
        branch_j2_correction_s: 3.5,
        ..VerifiedSupersetStageMetrics::default()
    };
    let before = metrics;
    let before_debug = format!("{metrics:?}");
    let counters = EvaluationDiagnosticCounters {
        lambert_batch_call_count: 5,
        branch_shared_prepare_count: 1,
        j2_correction_residual_m_sum: 2.25,
        branch_j2_correction_s: 4.5,
        ..EvaluationDiagnosticCounters::default()
    };

    let result = add_evaluation_diagnostics_to_stage_metrics(&mut metrics, &counters);

    anyhow::ensure!(
        matches!(
            result,
            Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
        ),
        "metric-count overflow must return ArithmeticOverflow"
    );
    anyhow::ensure!(
        format!("{metrics:?}") == before_debug,
        "metric merge overflow partially mutated metrics"
    );
    anyhow::ensure!(
        metrics.lambert_batch_call_count == before.lambert_batch_call_count
            && metrics.branch_shared_prepare_count == before.branch_shared_prepare_count,
        "metric counts changed after overflow"
    );
    anyhow::ensure!(
        metrics.j2_correction_residual_m_sum.to_bits()
            == before.j2_correction_residual_m_sum.to_bits()
            && metrics.branch_j2_correction_s.to_bits() == before.branch_j2_correction_s.to_bits(),
        "metric timing or residual bits changed after overflow"
    );
    Ok(())
}

#[test]
fn evaluation_diagnostic_merge_accepts_exact_count_boundary() -> anyhow::Result<()> {
    let Some(max_minus_one) = usize::MAX.checked_sub(1) else {
        anyhow::bail!("usize maximum must exceed zero");
    };
    let mut metrics = VerifiedSupersetStageMetrics {
        lambert_batch_call_count: max_minus_one,
        branch_shared_prepare_count: max_minus_one,
        j2_correction_residual_m_sum: 1.5,
        ..VerifiedSupersetStageMetrics::default()
    };
    let counters = EvaluationDiagnosticCounters {
        lambert_batch_call_count: 1,
        branch_shared_prepare_count: 1,
        j2_correction_residual_m_sum: 2.25,
        ..EvaluationDiagnosticCounters::default()
    };

    add_evaluation_diagnostics_to_stage_metrics(&mut metrics, &counters)?;

    anyhow::ensure!(
        metrics.lambert_batch_call_count == usize::MAX
            && metrics.branch_shared_prepare_count == usize::MAX,
        "exact count boundary must merge without overflow"
    );
    anyhow::ensure!(
        metrics.j2_correction_residual_m_sum.to_bits() == 3.75_f64.to_bits(),
        "successful boundary merge must preserve residual sum"
    );
    Ok(())
}

#[test]
fn local_optimizer_errors_preserve_typed_authority_routes() -> anyhow::Result<()> {
    let wrapped_authority =
        anyhow::Error::new(InvalidTargetPropagationAuthorityCode::InvalidCode(7))
            .context("synthetic local-optimizer context");
    anyhow::ensure!(
        local_optimizer_failure_code(&wrapped_authority)
            == InvalidTargetPropagationAuthorityCode::InvalidCode(7),
        "wrapped authority error must remain exact"
    );

    let wrapped_arithmetic = anyhow::Error::new(crate::oxymoo::ArithmeticOverflow)
        .context("synthetic local-optimizer context");
    anyhow::ensure!(
        local_optimizer_failure_code(&wrapped_arithmetic)
            == InvalidTargetPropagationAuthorityCode::ArithmeticOverflow,
        "wrapped OxyMOO arithmetic error must map to ArithmeticOverflow"
    );

    let wrapped_generic = anyhow::anyhow!("synthetic local optimizer failure")
        .context("synthetic local-optimizer context");
    anyhow::ensure!(
        local_optimizer_failure_code(&wrapped_generic)
            == InvalidTargetPropagationAuthorityCode::OptimizerFailure,
        "generic local optimizer error must map to OptimizerFailure"
    );
    Ok(())
}

#[cfg(feature = "bench-internal")]
#[test]
fn bench_local_optimizer_boundary_preserves_typed_error_routes() -> anyhow::Result<()> {
    let authority = map_bench_local_optimizer_result::<()>(Err(anyhow::Error::new(
        InvalidTargetPropagationAuthorityCode::InvalidCode(7),
    )));
    anyhow::ensure!(
        matches!(
            authority,
            Err(InvalidTargetPropagationAuthorityCode::InvalidCode(7))
        ),
        "benchmark local optimizer boundary must preserve authority code"
    );

    let arithmetic = map_bench_local_optimizer_result::<()>(Err(anyhow::Error::new(
        crate::oxymoo::ArithmeticOverflow,
    )));
    anyhow::ensure!(
        matches!(
            arithmetic,
            Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
        ),
        "benchmark local optimizer boundary must preserve arithmetic overflow"
    );

    let generic =
        map_bench_local_optimizer_result::<()>(Err(anyhow::anyhow!("synthetic optimizer failure")));
    anyhow::ensure!(
        matches!(
            generic,
            Err(InvalidTargetPropagationAuthorityCode::OptimizerFailure)
        ),
        "benchmark local optimizer boundary must map generic failure"
    );
    Ok(())
}

#[test]
fn deterministic_grid_propagates_evaluation_counter_overflow() -> anyhow::Result<()> {
    let baseline = evaluation_diagnostic_snapshot();
    let mut poisoned = baseline;
    // The representative grid context enables multi-revolution branch
    // selection, which takes the scalar Lambert path rather than batch R2.
    poisoned.lambert_scalar_tof_count = usize::MAX;
    crate::evaluate::restore_evaluation_diagnostics(&poisoned);

    let mut ctx = make_leo_ctx()?;
    let result = solve_plan_deterministic_grid(&mut ctx);
    let after = evaluation_diagnostic_snapshot();
    crate::evaluate::restore_evaluation_diagnostics(&baseline);

    anyhow::ensure!(
        matches!(
            result,
            Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
        ),
        "deterministic grid must surface evaluator counter overflow"
    );
    anyhow::ensure!(
        after.lambert_scalar_tof_count == usize::MAX,
        "failed evaluator diagnostic update must not partially mutate the counter"
    );
    Ok(())
}

#[test]
fn target_orbit_invariants_use_position_norm_for_unbound_target() {
    let target = [7_000.0, 0.0, 0.0, 12.0, 0.0, 0.0];
    assert!(EciBasicOrbit::from_eci(&target).is_none());

    let invariants = target_orbit_invariants(&target);

    assert!(!invariants.orbit_valid);
    assert_eq!(invariants.sma.to_bits(), 0.0_f64.to_bits());
    assert_eq!(invariants.period.to_bits(), 0.0_f64.to_bits());
    assert_eq!(invariants.period_cached.to_bits(), 0.0_f64.to_bits());
    assert_eq!(invariants.sma_norm.to_bits(), 7_000.0_f64.to_bits());
}

#[test]
fn repaired_transfer_decision_fails_closed_for_short_input() {
    let repaired = repaired_transfer_decision(&[0.2, 1.0]);

    assert!(repaired.iter().all(|value| value.is_nan()));
}

#[test]
fn selected_pair_percentiles_ignore_nonfinite_samples() {
    let mut samples = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        100.0,
        0.0,
        50.0,
        95.0,
    ];

    assert_eq!(
        selected_pair_front_solve_percentiles(&mut samples),
        (95.0, 100.0, 100.0)
    );
}

#[test]
fn percentile_helpers_match_nearest_rank_positions() {
    let mut seconds = (0_u16..=100).map(f64::from).collect::<Vec<_>>();
    let mut rows = (0_usize..=100).collect::<Vec<_>>();

    assert_eq!(
        selected_pair_front_solve_percentiles(&mut seconds),
        (50.0, 95.0, 100.0)
    );
    assert_eq!(branch_rows_per_source_percentiles(&mut rows), (50, 95, 100));
}

#[test]
fn final_polish_bounds_clamp_each_decision_axis() {
    let (lower, upper) = final_candidate_polish_bounds(&[0.0, 1.0, 0.95]);
    let [lower_time, lower_phase, lower_wait] = lower;
    let [upper_time, upper_phase, upper_wait] = upper;

    assert_eq!(lower_time.to_bits(), 0.0_f64.to_bits());
    assert_eq!(lower_phase.to_bits(), 0.98_f64.to_bits());
    assert_eq!(lower_wait.to_bits(), (0.95_f64 - 0.015).to_bits());
    assert_eq!(upper_time.to_bits(), 0.015_f64.to_bits());
    assert_eq!(upper_phase.to_bits(), 1.02_f64.to_bits());
    assert_eq!(upper_wait.to_bits(), 0.95_f64.to_bits());
}

fn make_leo_ctx() -> Result<PlanContext, InvalidTargetPropagationAuthorityCode> {
    let r = 6778.0;
    let v = (MU / r).sqrt();
    let dep_eci = [r, 0.0, 0.0, 0.0, v, 0.0];

    let r_tgt = 6878.0;
    let v_tgt = (MU / r_tgt).sqrt();
    let tgt_eci = [r_tgt, 0.0, 0.0, 0.0, v_tgt, 0.0];

    let mut dep_equ = [0.0; 6];
    satpy_core::eci2equinoc_impl(&dep_eci, 6, 0.0, 0.0, &mut dep_equ);
    let mut tgt_equ = [0.0; 6];
    satpy_core::eci2equinoc_impl(&tgt_eci, 6, 0.0, 0.0, &mut tgt_equ);

    let template = PlanContextTemplate {
        max_time_s: 86_400.0,
        tof_penalty_weight: 0.1,
        revolution_cap: 2.0,
        max_phase_dv: 1.0,
        max_transfer_dv: 2.0,
        min_perigee: 6_500.0,
        max_apogee: 50_000.0,
        max_revs: 2,
        sampling_mode: SamplingMode::Fast,
        execution_policy: ExecutionPolicy {
            use_high_fidelity: false,
            require_high_fidelity: false,
            allow_parallel: true,
            allow_oxymoo_batch_parallel: false,
            allow_branch_expansion_parallel: false,
            allow_polish_parallel: false,
            allow_anchor_parallel: false,
            allow_deterministic_grid_parallel: false,
        },
        j2_closure_settings: J2ClosureSettings::default(),
        search_depth: SearchDepthPolicy::default(),
        distance_tol: 0.025,
        deployer_min_distance: 0.12,
        target_propagation_authority: TargetPropagationAuthority::MfJ2,
        force_config: None,
        packed_coeffs: None,
        local_optimizer: TransferLocalOptimizerConfig::default(),
    };
    build_cached_plan_context(
        &template,
        &PairPlanContextInputs {
            dep_eci,
            dep_equ,
            epoch_jd: 0.0,
            tgt_eci,
            tgt_equ,
            dep_sma: r,
            dep_period: 2.0 * std::f64::consts::PI * ((r * r * r) / MU).sqrt(),
            dep_orbit_cached: EciBasicOrbit::default(),
            dep_orbit_valid: false,
            tgt_period_cached: 0.0,
            tgt_orbit_valid: true,
            tgt_sma: r_tgt,
            tgt_period: 2.0 * std::f64::consts::PI * ((r_tgt * r_tgt * r_tgt) / MU).sqrt(),
        },
    )
}

fn solve_plan_representative(
    ctx: &mut PlanContext,
) -> Result<PlanResult, InvalidTargetPropagationAuthorityCode> {
    Ok(solve_plan(ctx, None)?
        .candidates
        .into_iter()
        .next()
        .unwrap_or_else(PlanResult::invalid))
}

fn synthetic_transfer_candidate(
    total_dv: f64,
    total_time: f64,
    relative_velocity: f64,
    ratios: [f64; 3],
) -> PlanResult {
    let mut plan = PlanResult::invalid();
    plan.valid = true;
    plan.cost = total_dv;
    plan.phase_dv_norm = total_dv;
    plan.transfer_dv_norm = 0.0;
    plan.time2phase = total_time;
    plan.waittime = 0.0;
    plan.tof = 0.0;
    let [_, _, _, payload_velocity, _, _] = &mut plan.payload_intercept_state;
    *payload_velocity = relative_velocity;
    let [_, _, _, target_velocity, _, _] = &mut plan.target_intercept_state;
    *target_velocity = 0.0;
    let [time2phase_ratio, phase_sma_ratio, waittime_ratio] = ratios;
    plan.time2phase_ratio = time2phase_ratio;
    plan.phase_sma_ratio = phase_sma_ratio;
    plan.waittime_ratio = waittime_ratio;
    plan
}

fn assert_transfer_front_posthoc_verified(ctx: &PlanContext, front: &TransferFront) {
    let tolerance = verification_tolerance_for_solve(ctx);
    for candidate in &front.candidates {
        let verification = verify_transfer_result(candidate, ctx, tolerance);
        assert!(
            verification.verified,
            "returned transfer candidate failed post-hoc verification: {verification:?}; candidate={candidate:?}"
        );
    }
}

#[test]
fn delta_v_anchor_policy_flags_match_names() {
    assert!(DeltaVAnchorPolicy::Full.use_cost_anchor());
    assert!(DeltaVAnchorPolicy::Full.use_delta_v_anchor());
    assert!(DeltaVAnchorPolicy::Full.use_probes());

    assert!(DeltaVAnchorPolicy::NoProbes.use_cost_anchor());
    assert!(DeltaVAnchorPolicy::NoProbes.use_delta_v_anchor());
    assert!(!DeltaVAnchorPolicy::NoProbes.use_probes());

    assert!(DeltaVAnchorPolicy::CostOnlyNoProbes.use_cost_anchor());
    assert!(!DeltaVAnchorPolicy::CostOnlyNoProbes.use_delta_v_anchor());
    assert!(!DeltaVAnchorPolicy::CostOnlyNoProbes.use_probes());

    assert!(!DeltaVAnchorPolicy::DvOnlyNoProbes.use_cost_anchor());
    assert!(DeltaVAnchorPolicy::DvOnlyNoProbes.use_delta_v_anchor());
    assert!(!DeltaVAnchorPolicy::DvOnlyNoProbes.use_probes());

    assert!(DeltaVAnchorPolicy::SeedLimit2.use_cost_anchor());
    assert!(DeltaVAnchorPolicy::SeedLimit2.use_delta_v_anchor());
    assert!(DeltaVAnchorPolicy::SeedLimit2.use_probes());
    assert_eq!(DeltaVAnchorPolicy::SeedLimit2.seed_limit(), 2);

    assert!(DeltaVAnchorPolicy::SeedLimit3.use_cost_anchor());
    assert!(DeltaVAnchorPolicy::SeedLimit3.use_delta_v_anchor());
    assert!(DeltaVAnchorPolicy::SeedLimit3.use_probes());
    assert_eq!(DeltaVAnchorPolicy::SeedLimit3.seed_limit(), 3);
}

#[test]
fn fine_cutoffs_keep_historical_limit_margin_and_all_seed_cases() -> anyhow::Result<()> {
    let ranked = [
        (
            SolverSeed {
                x: [0.10, 1.00, 0.10],
                warm_start_used: false,
            },
            synthetic_transfer_candidate(0.10, 5_000.0, 4.0, [0.10, 1.00, 0.10]),
        ),
        (
            SolverSeed {
                x: [0.20, 1.00, 0.10],
                warm_start_used: false,
            },
            synthetic_transfer_candidate(0.20, 5_000.0, 4.0, [0.20, 1.00, 0.10]),
        ),
        (
            SolverSeed {
                x: [0.30, 1.00, 0.10],
                warm_start_used: false,
            },
            synthetic_transfer_candidate(0.40, 5_000.0, 4.0, [0.30, 1.00, 0.10]),
        ),
    ];
    let limited = SearchDepthPolicy {
        fine_total_limit: 2,
        seed_fine_margin_km_s: 0.05,
        ..SearchDepthPolicy::default()
    };
    anyhow::ensure!(
        fine_cost_threshold(&ranked, &limited).to_bits() == 0.25_f64.to_bits(),
        "limited fine cutoff must preserve the historical 0.25 threshold"
    );

    let all = SearchDepthPolicy {
        fine_total_limit: ranked.len(),
        ..limited
    };
    anyhow::ensure!(
        fine_cost_threshold(&ranked, &all).is_infinite(),
        "all-seed fine cutoff must remain unbounded"
    );

    let [first, second, third] = ranked;
    let unordered = [third, first, second];
    anyhow::ensure!(
        provisional_fine_count_for_coarse_early_stop(&unordered, &limited)? == 2,
        "provisional fine count must retain exactly two ranked seeds"
    );
    Ok(())
}

#[test]
fn optimizer_start_seed_selection_preserves_bounded_rank_window_then_dedup() -> anyhow::Result<()> {
    let first = SolverSeed {
        x: [0.10, 1.00, 0.10],
        warm_start_used: false,
    };
    let duplicate = SolverSeed {
        x: first.x,
        warm_start_used: true,
    };
    let second = SolverSeed {
        x: [0.20, 1.00, 0.10],
        warm_start_used: false,
    };
    let third = SolverSeed {
        x: [0.30, 1.00, 0.10],
        warm_start_used: false,
    };
    let ranked = [
        (first, PlanResult::invalid()),
        (duplicate, PlanResult::invalid()),
        (second, PlanResult::invalid()),
        (third, PlanResult::invalid()),
    ];

    let selected = select_optimizer_start_seeds(&ranked, LocalOptimizerKind::NelderMead)?;
    let expected = [first, second];
    anyhow::ensure!(
        selected.len() == expected.len(),
        "Nelder-Mead must deduplicate inside its first three ranked seeds"
    );
    for (index, (actual, expected)) in selected.iter().zip(expected).enumerate() {
        anyhow::ensure!(
            actual.x.map(f64::to_bits) == expected.x.map(f64::to_bits)
                && actual.warm_start_used == expected.warm_start_used,
            "Nelder-Mead selected seed {index} differs from the bounded rank-window oracle"
        );
    }
    Ok(())
}

#[test]
fn delta_v_anchor_start_selection_keeps_first_duplicate_and_strict_score_order() {
    let first = SolverSeed {
        x: [0.10, 1.00, 0.10],
        warm_start_used: false,
    };
    let duplicate = SolverSeed {
        x: first.x,
        warm_start_used: true,
    };
    let second = SolverSeed {
        x: [0.20, 1.00, 0.10],
        warm_start_used: false,
    };
    let third = SolverSeed {
        x: [0.30, 1.00, 0.10],
        warm_start_used: false,
    };
    let ranked = vec![
        (
            first,
            synthetic_transfer_candidate(0.50, 5_000.0, 4.0, first.x),
        ),
        (
            duplicate,
            synthetic_transfer_candidate(0.10, 5_000.0, 4.0, duplicate.x),
        ),
        (
            second,
            synthetic_transfer_candidate(0.20, 5_000.0, 4.0, second.x),
        ),
        (
            third,
            synthetic_transfer_candidate(0.30, 5_000.0, 4.0, third.x),
        ),
    ];

    let selected = select_delta_v_anchor_starts(&ranked, None, DeltaVAnchorPolicy::SeedLimit3);
    let selected = selected.into_iter().flatten().collect::<Vec<_>>();

    let [first_selected, second_selected, third_selected] = selected.as_slice() else {
        panic!("expected exactly three selected delta-v anchor starts");
    };
    assert_eq!(
        first_selected.point.map(f64::to_bits),
        second.x.map(f64::to_bits)
    );
    assert_eq!(
        second_selected.point.map(f64::to_bits),
        third.x.map(f64::to_bits)
    );
    assert_eq!(
        third_selected.point.map(f64::to_bits),
        first.x.map(f64::to_bits)
    );
    assert!(!third_selected.point.iter().any(|value| value.is_nan()));
    assert_eq!(third_selected.score.to_bits(), 0.50_f64.to_bits());
}

#[derive(Clone, Copy)]
struct TestPairContextRequest {
    dep_eci: [f64; 6],
    tgt_eci: [f64; 6],
    max_time_s: f64,
    max_phase_dv: f64,
    max_transfer_dv: f64,
    max_revs: i32,
    min_perigee: f64,
    max_apogee: f64,
    sampling_mode: SamplingMode,
    search_depth: SearchDepthPolicy,
    epoch_jd: f64,
    distance_tol: f64,
    deployer_min_distance: f64,
    tof_penalty_weight: f64,
    revolution_cap: f64,
}

fn make_pair_ctx_for_test(
    request: &TestPairContextRequest,
) -> Result<PlanContext, InvalidTargetPropagationAuthorityCode> {
    let TestPairContextRequest {
        dep_eci,
        tgt_eci,
        max_time_s,
        max_phase_dv,
        max_transfer_dv,
        max_revs,
        min_perigee,
        max_apogee,
        sampling_mode,
        search_depth,
        epoch_jd,
        distance_tol,
        deployer_min_distance,
        tof_penalty_weight,
        revolution_cap,
    } = *request;
    let mut dep_equ = [0.0; 6];
    satpy_core::eci2equinoc_impl(&dep_eci, 6, 0.0, 0.0, &mut dep_equ);
    let mut tgt_equ = [0.0; 6];
    satpy_core::eci2equinoc_impl(&tgt_eci, 6, 0.0, 0.0, &mut tgt_equ);
    let dep_orbit = EciBasicOrbit::from_eci(&dep_eci).unwrap_or_default();
    let tgt_orbit = EciBasicOrbit::from_eci(&tgt_eci).unwrap_or_default();
    let dep_orbit_valid = dep_orbit.sma.is_finite() && dep_orbit.sma > 0.0;
    let tgt_orbit_valid = tgt_orbit.sma.is_finite() && tgt_orbit.sma > 0.0;
    let dep_period = if dep_orbit_valid {
        2.0 * std::f64::consts::PI * ((dep_orbit.sma * dep_orbit.sma * dep_orbit.sma) / MU).sqrt()
    } else {
        0.0
    };
    let tgt_period = if tgt_orbit_valid {
        2.0 * std::f64::consts::PI * ((tgt_orbit.sma * tgt_orbit.sma * tgt_orbit.sma) / MU).sqrt()
    } else {
        0.0
    };

    let template = PlanContextTemplate {
        max_time_s,
        tof_penalty_weight,
        revolution_cap,
        max_phase_dv,
        max_transfer_dv,
        min_perigee,
        max_apogee,
        max_revs,
        sampling_mode,
        execution_policy: ExecutionPolicy {
            use_high_fidelity: false,
            require_high_fidelity: false,
            allow_parallel: true,
            allow_oxymoo_batch_parallel: false,
            allow_branch_expansion_parallel: false,
            allow_polish_parallel: false,
            allow_anchor_parallel: false,
            allow_deterministic_grid_parallel: false,
        },
        j2_closure_settings: J2ClosureSettings::default(),
        search_depth,
        distance_tol,
        deployer_min_distance,
        target_propagation_authority: TargetPropagationAuthority::MfJ2,
        force_config: None,
        packed_coeffs: None,
        local_optimizer: TransferLocalOptimizerConfig::default(),
    };
    build_cached_plan_context(
        &template,
        &PairPlanContextInputs {
            dep_eci,
            dep_equ,
            epoch_jd,
            tgt_eci,
            tgt_equ,
            dep_sma: dep_orbit.sma,
            dep_period,
            dep_orbit_cached: dep_orbit,
            dep_orbit_valid,
            tgt_period_cached: tgt_period,
            tgt_orbit_valid,
            tgt_sma: tgt_orbit.sma,
            tgt_period,
        },
    )
}

fn assert_constellation_front_posthoc_verified(
    satellites: &[[f64; 6]],
    target1: [f64; 6],
    target2: [f64; 6],
    configuration: &ConstellationSolveConfiguration,
    front: &ConstellationTransferFront,
) -> anyhow::Result<()> {
    for candidate in &front.candidates {
        let sat_idx = usize::try_from(candidate.sat_index).map_err(|error| {
            anyhow::anyhow!("test candidate satellite index must be nonnegative: {error}")
        })?;
        let dep_eci = *satellites.get(sat_idx).ok_or_else(|| {
            anyhow::anyhow!("test candidate satellite index must be in range: {sat_idx}")
        })?;
        let target = if candidate.target_index == 0 {
            target1
        } else {
            target2
        };
        let ctx = make_pair_ctx_for_test(&TestPairContextRequest {
            dep_eci,
            tgt_eci: target,
            max_time_s: configuration.max_time_s,
            max_phase_dv: configuration.max_phase_dv,
            max_transfer_dv: configuration.max_transfer_dv,
            max_revs: configuration.max_revs,
            min_perigee: configuration.min_perigee,
            max_apogee: configuration.max_apogee,
            sampling_mode: configuration.sampling_mode,
            search_depth: configuration.search_depth,
            epoch_jd: configuration.epoch_jd,
            distance_tol: configuration.distance_tol,
            deployer_min_distance: configuration.deployer_min_distance,
            tof_penalty_weight: configuration.tof_penalty_weight,
            revolution_cap: configuration.revolution_cap,
        })?;
        let verification = verify_transfer_result(
            &candidate.optimum,
            &ctx,
            verification_tolerance_for_solve(&ctx),
        );
        anyhow::ensure!(
            verification.verified,
            "returned constellation candidate failed post-hoc verification: {verification:?}; candidate={candidate:?}"
        );
    }
    Ok(())
}

#[test]
fn zero_pair_request_means_full_pair_front() {
    assert_eq!(pair_verification_limit(0, 8), 8);
    assert_eq!(pair_verification_limit(3, 8), 3);
    assert_eq!(pair_verification_limit(20, 8), 8);
}

fn pair_proxy_for_test(
    sat_idx: usize,
    tgt_idx: usize,
    dv_proxy: f64,
    time_proxy_s: f64,
    rel_v_proxy: f64,
    cv_proxy: f64,
) -> PairProxyCandidate {
    PairProxyCandidate {
        sat_idx,
        tgt_idx,
        x_hint: [0.1, 1.0, 0.2],
        dv_proxy,
        time_proxy_s,
        rel_v_proxy,
        time_per_rel_v_proxy: pair_proxy_time_per_relative_velocity(time_proxy_s, rel_v_proxy),
        cv_proxy,
    }
}

#[test]
fn pair_proxy_budget_zero_attempts_all_pairs_without_proxy_pruning() -> anyhow::Result<()> {
    let candidates = vec![
        pair_proxy_for_test(0, 0, 0.10, 7200.0, 4.0, 0.0),
        pair_proxy_for_test(1, 0, 0.20, 3600.0, 3.0, 0.0),
        pair_proxy_for_test(2, 0, f64::INFINITY, 1200.0, 8.0, 0.0),
    ];

    let selection = select_pair_proxy_candidates(candidates, 0)?;

    anyhow::ensure!(
        selection.selected.len() == 3,
        "exact mode selected pair count"
    );
    let [first, second, third] = selection.selected.as_slice() else {
        anyhow::bail!("expected all three proxy candidates to remain selected");
    };
    anyhow::ensure!(selection.total_pairs == 3, "exact mode total pair count");
    anyhow::ensure!(
        selection.selected_pairs == 3,
        "exact mode selected metric count"
    );
    anyhow::ensure!(selection.selected_layers == 0, "exact mode layer count");
    anyhow::ensure!(
        selection.omitted_layers == 0,
        "exact mode omitted layer count"
    );
    anyhow::ensure!(first.sat_idx == 0, "first exact mode satellite");
    anyhow::ensure!(second.sat_idx == 1, "second exact mode satellite");
    anyhow::ensure!(third.sat_idx == 2, "third exact mode satellite");
    Ok(())
}

#[test]
fn exact_pair_mode_retains_invalid_proxy_before_selection() {
    let invalid = pair_proxy_for_test(0, 0, INVALID_COST, 1200.0, 0.0, 1.0);

    assert!(retain_pair_proxy_candidate(&invalid, 0));
    assert!(!retain_pair_proxy_candidate(&invalid, 1));
}

#[test]
fn exact_prepare_event_reports_every_input_pair_even_with_invalid_proxies() -> anyhow::Result<()> {
    let satellites = vec![[0.0; 6]];
    let target1 = make_circular_state(500.0, 0.0, 0.0, 0.2);
    let target2 = make_circular_state(620.0, 0.02, 0.0, 0.8);

    let mut configuration = test_mf_configuration();
    configuration.pairs_to_verify = 0;
    let Some(plan) = prepare_event(EventPlanRequest {
        satellites: &satellites,
        satellites_equ_cached: None,
        target1: &target1,
        target2: &target2,
        target_body_forces: [BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2],
        configuration,
        scratch: None,
        front_output_mode: FrontOutputMode::VerifiedSuperset,
    })?
    else {
        anyhow::bail!("one-satellite fixture must prepare an event");
    };

    anyhow::ensure!(plan.selected_pair_count() == 2, "selected pair count");
    anyhow::ensure!(
        plan.screen_metrics.pair_proxy_candidate_count == 2,
        "pair-proxy candidate count"
    );
    anyhow::ensure!(
        plan.screen_metrics.selected_pair_count == 2,
        "selected-pair metric count"
    );
    anyhow::ensure!(
        plan.screen_metrics.pair_proxy_exact_mode,
        "exact pair-proxy mode"
    );
    anyhow::ensure!(
        plan.screen_metrics.selected_pair_target0_count == 1,
        "target zero selected-pair count"
    );
    anyhow::ensure!(
        plan.screen_metrics.selected_pair_target1_count == 1,
        "target one selected-pair count"
    );
    anyhow::ensure!(
        plan.selected_pairs
            .iter()
            .all(|candidate| candidate.dv_proxy >= INVALID_COST),
        "invalid proxy candidate must remain visible in exact mode"
    );
    Ok(())
}

#[test]
fn pair_proxy_capacity_overflow_fails_before_event_construction() -> anyhow::Result<()> {
    // `prepare_event` calls this preflight before metrics, orbit conversion,
    // scratch preparation, selected-pair construction, or EventPlan output.
    let overflow = pair_proxy_capacity_for_satellite_count(usize::MAX);

    anyhow::ensure!(
        matches!(
            overflow,
            Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
        ),
        "pair-proxy capacity overflow must remain typed"
    );
    anyhow::ensure!(
        pair_proxy_capacity_for_satellite_count(1)? == 2,
        "finite pair-proxy capacity changed"
    );
    Ok(())
}

#[test]
fn pair_proxy_scratch_impossible_reservation_is_typed() -> anyhow::Result<()> {
    let mut scratch = PairProxyScratch::new(0);

    let reservation = scratch.prepare(usize::MAX);

    anyhow::ensure!(
        matches!(
            reservation,
            Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
        ),
        "impossible pair-proxy scratch reservation must remain typed"
    );
    Ok(())
}

#[test]
fn pair_proxy_budgeted_selection_uses_pareto_layers_not_scalar_topk() -> anyhow::Result<()> {
    let low_dv_slow = pair_proxy_for_test(0, 0, 0.10, 5000.0, 2.0, 0.0);
    let scalar_second_but_dominated = pair_proxy_for_test(1, 0, 0.20, 7000.0, 1.0, 0.0);
    let fast_high_relative_velocity = pair_proxy_for_test(2, 0, 0.30, 900.0, 8.0, 0.0);

    let selection = select_pair_proxy_candidates(
        vec![
            scalar_second_but_dominated,
            fast_high_relative_velocity,
            low_dv_slow,
        ],
        2,
    )?;

    let selected_pairs: std::collections::BTreeSet<_> = selection
        .selected
        .iter()
        .map(|candidate| (candidate.sat_idx, candidate.tgt_idx))
        .collect();
    anyhow::ensure!(selected_pairs.contains(&(0, 0)), "low-dv pair missing");
    anyhow::ensure!(selected_pairs.contains(&(2, 0)), "fast pair missing");
    anyhow::ensure!(
        !selected_pairs.contains(&(1, 0)),
        "dominated scalar pair unexpectedly selected"
    );
    anyhow::ensure!(selection.selected_layers == 1, "selected layer count");
    anyhow::ensure!(selection.omitted_layers == 1, "omitted layer count");
    Ok(())
}

#[test]
fn pair_proxy_budgeted_selection_preserves_target_diversity_within_cap() -> anyhow::Result<()> {
    let target_zero_best = pair_proxy_for_test(0, 0, 0.10, 1200.0, 3.0, 0.0);
    let target_zero_redundant = pair_proxy_for_test(1, 0, 0.11, 1300.0, 3.0, 0.0);
    let target_one_best = pair_proxy_for_test(2, 1, 0.12, 1000.0, 4.0, 0.0);
    let target_one_redundant = pair_proxy_for_test(3, 1, 0.20, 2000.0, 2.0, 0.0);

    let selection = select_pair_proxy_candidates(
        vec![
            target_zero_redundant,
            target_one_redundant,
            target_zero_best,
            target_one_best,
        ],
        2,
    )?;

    let selected_pairs: std::collections::BTreeSet<_> = selection
        .selected
        .iter()
        .map(|candidate| (candidate.sat_idx, candidate.tgt_idx))
        .collect();
    anyhow::ensure!(selection.selected.len() == 2, "selected pair count");
    anyhow::ensure!(selection.selected_pairs == 2, "selected metric count");
    anyhow::ensure!(selection.selected_by_target == [1, 1], "target diversity");
    anyhow::ensure!(selected_pairs.contains(&(0, 0)), "target zero pair missing");
    anyhow::ensure!(selected_pairs.contains(&(2, 1)), "target one pair missing");
    Ok(())
}

#[test]
fn pair_proxy_nonpositive_relative_velocity_gets_constraint_violation() -> anyhow::Result<()> {
    let satellite = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
    let target = [7100.0, 0.0, 0.0, 0.0, 7.5, 0.0];
    let sat_props = SatOrbitProps {
        sma_est: 7000.0,
        sma_orbit: 7000.0,
        period_orbit: 6000.0,
        orbit_cached: EciBasicOrbit::default(),
        orbit_valid: true,
    };

    let invalid_rel_v = make_pair_proxy_candidate(
        0,
        0,
        &satellite,
        &target,
        &sat_props,
        7100.0,
        6100.0,
        true,
        node_wait_proxy(&satellite, &target),
        pair_x_hint(&satellite, &target, 86_400.0),
        86_400.0,
        PairProxyModel::Sum,
    );
    let valid = pair_proxy_for_test(1, 0, 0.10, 1200.0, 3.0, 0.0);
    let selection = select_pair_proxy_candidates(vec![invalid_rel_v, valid], 1)?;

    anyhow::ensure!(
        invalid_rel_v.rel_v_proxy == 0.0,
        "fixture relative velocity"
    );
    anyhow::ensure!(
        invalid_rel_v.cv_proxy >= 1.0,
        "fixture constraint violation"
    );
    anyhow::ensure!(selection.selected.len() == 1, "selected pair count");
    let [selected] = selection.selected.as_slice() else {
        anyhow::bail!("expected exactly one selected proxy candidate");
    };
    anyhow::ensure!(selected.sat_idx == 1, "finite candidate satellite index");
    Ok(())
}

#[test]
fn pair_proxy_no_finite_candidates_remains_successful_empty() -> anyhow::Result<()> {
    let candidates = vec![
        pair_proxy_for_test(0, 0, f64::INFINITY, 1200.0, 3.0, 0.0),
        pair_proxy_for_test(1, 1, 0.2, f64::NAN, 4.0, 0.0),
    ];

    let selection = select_pair_proxy_candidates(candidates, 1)?;

    anyhow::ensure!(selection.total_pairs == 2, "total pair count");
    anyhow::ensure!(
        selection.selected.is_empty(),
        "non-finite pairs must not be selected"
    );
    anyhow::ensure!(selection.selected_pairs == 0, "selected metric count");
    anyhow::ensure!(selection.selected_layers == 0, "selected layer count");
    anyhow::ensure!(selection.omitted_layers == 0, "omitted layer count");
    anyhow::ensure!(
        selection.selected_by_target == [0, 0],
        "selected target counts"
    );
    Ok(())
}

#[test]
fn pair_x_hint_precomputed_kepler_matches_inline_conversion() {
    let satellite = [7000.0, 0.0, 300.0, 0.0, 7.45, 0.2];
    let target = [7100.0, 120.0, -250.0, -0.1, 7.35, 0.25];

    let inline = pair_x_hint(&satellite, &target, 86_400.0);
    let cached = pair_x_hint_from_kepler(
        &kepler_from_eci(&satellite),
        &kepler_from_eci(&target),
        86_400.0,
    );

    for (inline, cached) in inline.iter().zip(cached) {
        assert!((*inline - cached).abs() < 1.0e-12);
    }
}

#[test]
fn pair_time_proxy_precomputed_node_wait_matches_inline_node_wait() {
    let satellite = [7000.0, 0.0, 300.0, 0.0, 7.45, 0.2];
    let target = [7100.0, 120.0, -250.0, -0.1, 7.35, 0.25];
    let sat_props = compute_sat_orbit_props(&satellite);
    let target_props = compute_sat_orbit_props(&target);
    let inline_node_wait = node_wait_proxy(&satellite, &target);
    let precomputed_node_wait = node_wait_proxy_from_min_times(
        node_wait_min_from_eci(&satellite),
        node_wait_min_from_eci(&target),
    );

    assert!((inline_node_wait - precomputed_node_wait).abs() < 1.0e-9);

    let inline_result = pair_time_proxy_and_cv(
        &satellite,
        &target,
        sat_props.sma_orbit,
        target_props.sma_orbit,
        sat_props.period_orbit,
        target_props.period_orbit,
        inline_node_wait,
        86_400.0,
    );
    let precomputed_result = pair_time_proxy_and_cv(
        &satellite,
        &target,
        sat_props.sma_orbit,
        target_props.sma_orbit,
        sat_props.period_orbit,
        target_props.period_orbit,
        precomputed_node_wait,
        86_400.0,
    );

    assert!((inline_result.0 - precomputed_result.0).abs() < 1.0e-9);
    assert!((inline_result.1 - precomputed_result.1).abs() < 1.0e-12);
}

#[test]
fn constellation_front_archive_matches_full_final_filtering() -> anyhow::Result<()> {
    let kept_low_dv = ConstellationTransferCandidate::from_plan(
        0,
        0,
        0.10,
        [0.1, 1.0, 0.1],
        synthetic_transfer_candidate(0.10, 7200.0, 5.0, [0.1, 1.0, 0.1]),
    )
    .ok_or_else(|| anyhow::anyhow!("low-dv archive fixture must be representable"))?;
    let kept_fast = ConstellationTransferCandidate::from_plan(
        1,
        0,
        0.20,
        [0.2, 1.0, 0.0],
        synthetic_transfer_candidate(0.20, 1800.0, 6.0, [0.2, 1.0, 0.0]),
    )
    .ok_or_else(|| anyhow::anyhow!("fast archive fixture must be representable"))?;
    let dominated = ConstellationTransferCandidate::from_plan(
        2,
        0,
        0.30,
        [0.3, 1.0, 0.2],
        synthetic_transfer_candidate(0.25, 4000.0, 4.0, [0.3, 1.0, 0.2]),
    )
    .ok_or_else(|| anyhow::anyhow!("dominated archive fixture must be representable"))?;
    let duplicate = ConstellationTransferCandidate::from_plan(
        3,
        0,
        0.20,
        [0.4, 1.0, 0.2],
        synthetic_transfer_candidate(0.20, 1800.0, 6.0, [0.4, 1.0, 0.2]),
    )
    .ok_or_else(|| anyhow::anyhow!("duplicate archive fixture must be representable"))?;

    let expected = finalize_constellation_transfer_front(vec![
        dominated.clone(),
        kept_fast.clone(),
        kept_low_dv.clone(),
        duplicate.clone(),
    ]);
    let mut archive = ConstellationFrontArchive::new();
    anyhow::ensure!(
        archive.insert(dominated)?,
        "dominated candidate first enters archive"
    );
    anyhow::ensure!(archive.insert(kept_fast)?, "fast candidate enters archive");
    anyhow::ensure!(
        archive.insert(kept_low_dv)?,
        "low-dv candidate enters archive"
    );
    anyhow::ensure!(
        !archive.insert(duplicate)?,
        "duplicate candidate stays deduplicated"
    );
    let actual = archive.into_front();

    anyhow::ensure!(actual.len() == expected.len(), "archive front size");
    let actual_objectives = actual
        .candidates
        .iter()
        .map(|candidate| candidate.objectives.as_minimization_array())
        .collect::<Vec<_>>();
    let expected_objectives = expected
        .candidates
        .iter()
        .map(|candidate| candidate.objectives.as_minimization_array())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        actual_objectives == expected_objectives,
        "archive objectives must match full filtering"
    );
    Ok(())
}

#[test]
fn constellation_superset_keeps_within_epsilon_and_drops_epsilon_dominated() {
    // Sealed nd-epsilon-membership contract (2026-08-13 reseal, evidence in
    // docs/evidence/front-lane-20260813/): the superset keeps a candidate that
    // is Pareto-dominated but within the epsilon band, and DROPS one beaten by
    // >= 0.05 km/s dv AND >= 5% time.
    let low_dv = ConstellationTransferCandidate::from_plan(
        0,
        0,
        0.10,
        [0.1, 1.0, 0.1],
        synthetic_transfer_candidate(0.10, 7200.0, 5.0, [0.1, 1.0, 0.1]),
    )
    .unwrap();
    let within_epsilon = ConstellationTransferCandidate::from_plan(
        1,
        0,
        0.18,
        [0.3, 1.0, 0.2],
        synthetic_transfer_candidate(0.14, 8000.0, 4.0, [0.3, 1.0, 0.2]),
    )
    .unwrap();
    let epsilon_dominated = ConstellationTransferCandidate::from_plan(
        2,
        0,
        0.30,
        [0.4, 1.0, 0.3],
        synthetic_transfer_candidate(0.25, 8000.0, 4.0, [0.4, 1.0, 0.3]),
    )
    .unwrap();

    let pareto = finalize_constellation_transfer_front(vec![
        low_dv.clone(),
        within_epsilon.clone(),
        epsilon_dominated.clone(),
    ]);
    let superset =
        finalize_constellation_transfer_superset(vec![low_dv, within_epsilon, epsilon_dominated]);

    assert_eq!(pareto.len(), 1);
    assert_eq!(superset.len(), 2);
    assert!(superset
        .candidates
        .iter()
        .all(|kept| kept.optimum.total_dv() < 0.2));
}

#[test]
fn verified_superset_expands_solver_decisions_across_lambert_branches() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let front = solve_plan_oxymoo_front_internal(
        &mut ctx,
        None,
        Some([0.0, 1.0, 0.0]),
        FrontOutputMode::VerifiedSuperset,
        None,
        DeltaVAnchorPolicy::Full,
        TransferMooPolicy::Full,
    )?;
    let branch_ids = front
        .candidates
        .iter()
        .map(|plan| (plan.branch_rev, plan.branch_low_path))
        .collect::<std::collections::BTreeSet<_>>();

    anyhow::ensure!(
        branch_ids.len() > 1,
        "verified superset should preserve branch alternatives, got {branch_ids:?}"
    );
    for plan in &front.candidates {
        anyhow::ensure!(
            plan.best_M <= ctx.max_revs,
            "verified superset emitted best_M={} above max_revs={}",
            plan.best_M,
            ctx.max_revs
        );
        for (axis, (arrival_dv, (target_velocity, payload_velocity))) in plan
            .arrival_dv
            .iter()
            .zip(
                plan.target_intercept_state
                    .iter()
                    .skip(3)
                    .zip(plan.payload_intercept_state.iter().skip(3)),
            )
            .enumerate()
        {
            let expected = *target_velocity - *payload_velocity;
            anyhow::ensure!(
                (*arrival_dv - expected).abs() < 1.0e-9,
                "arrival dV convention mismatch axis {axis}: got {arrival_dv} expected {expected}"
            );
        }
    }
    let metrics = front.verified_superset_metrics;
    anyhow::ensure!(
        metrics.branch_source_count > 0,
        "verified superset should record branch expansion source count"
    );
    anyhow::ensure!(
        metrics.branch_rows_per_source_max > 0,
        "verified superset should record branch rows per source"
    );
    anyhow::ensure!(
        metrics.branch_rows_per_source_p95 >= metrics.branch_rows_per_source_p50,
        "p95 rows/source must be at least p50"
    );
    anyhow::ensure!(
        metrics.branch_eval_call_count >= metrics.branch_source_count,
        "each source should trigger branch evaluation diagnostics"
    );
    Ok(())
}

#[test]
fn verified_superset_fallback_enumerates_deterministic_grid_branches() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let front = verified_superset_deterministic_grid_fallback(&mut ctx, false)?;
    let branch_ids = front
        .candidates
        .iter()
        .map(|plan| (plan.branch_rev, plan.branch_low_path))
        .collect::<std::collections::BTreeSet<_>>();

    anyhow::ensure!(
        branch_ids.len() > 1,
        "verified superset fallback should branch-expand, got {branch_ids:?}"
    );
    let decision_ids = front
        .candidates
        .iter()
        .map(|plan| {
            (
                plan.time2phase_ratio.to_bits(),
                plan.phase_sma_ratio.to_bits(),
                plan.waittime_ratio.to_bits(),
            )
        })
        .collect::<std::collections::BTreeSet<_>>();
    anyhow::ensure!(
        decision_ids.len() > 1,
        "verified superset fallback should enumerate grid decisions, got {}",
        decision_ids.len()
    );
    Ok(())
}

#[test]
fn verified_superset_respects_zero_max_revs() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.max_revs = 0;
    let front = solve_plan_oxymoo_front_internal(
        &mut ctx,
        None,
        Some([0.0, 1.0, 0.0]),
        FrontOutputMode::VerifiedSuperset,
        None,
        DeltaVAnchorPolicy::Full,
        TransferMooPolicy::Full,
    )?;

    anyhow::ensure!(
        !front.candidates.is_empty(),
        "zero-revolution verified superset should still keep feasible M0 transfers"
    );
    anyhow::ensure!(
        front
            .candidates
            .iter()
            .all(|plan| plan.branch_rev == 0 && plan.best_M == 0),
        "max_revs=0 must not emit multi-rev branches: {:?}",
        front
            .candidates
            .iter()
            .map(|plan| (plan.branch_rev, plan.best_M))
            .collect::<Vec<_>>()
    );
    anyhow::ensure!(
        front.candidates.iter().all(|plan| plan.branch_low_path),
        "M0 branches should stay on the canonical low-path flag"
    );
    anyhow::ensure!(
        front.verified_superset_metrics.branch_eval_call_count == 0,
        "max_revs=0 should reuse source plans instead of re-evaluating branch expansion"
    );
    Ok(())
}

#[test]
fn transfer_moo_problem_is_two_objective_with_cv() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    let plan_cache: TransferMooPlanCache = RefCell::new(FxHashMap::default());
    let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let problem = TransferMooProblem::new(ctx, plan_cache, policy)?;
    let mut objectives = [0.0; TRANSFER_MOO_OBJECTIVES];
    let cv = problem.evaluate(&[0.0, 1.0, 0.0], &mut objectives)?;

    anyhow::ensure!(problem.objective_count() == 2, "expected two objectives");
    anyhow::ensure!(
        problem.variable_specs() == TRANSFER_MOO_VARIABLES,
        "unexpected transfer decision variables"
    );
    anyhow::ensure!(cv.is_finite(), "constraint violation must be finite");
    anyhow::ensure!(
        objectives.iter().all(|value| value.is_finite()),
        "objectives must be finite: {objectives:?}"
    );
    anyhow::ensure!(
        !problem.plan_cache_borrow().is_empty(),
        "evaluation must cache its plan"
    );
    Ok(())
}

#[test]
fn transfer_moo_problem_caches_duplicate_repaired_decisions() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    let plan_cache: TransferMooPlanCache = RefCell::new(FxHashMap::default());
    let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let problem = TransferMooProblem::new(ctx, plan_cache, policy)?;
    let mut first_objectives = [0.0; TRANSFER_MOO_OBJECTIVES];
    let mut second_objectives = [0.0; TRANSFER_MOO_OBJECTIVES];
    let decision = [0.0, 1.0, 0.0];

    let first_cv = problem.evaluate(&decision, &mut first_objectives)?;
    let second_cv = problem.evaluate(&decision, &mut second_objectives)?;

    anyhow::ensure!(
        first_cv.to_bits() == second_cv.to_bits(),
        "duplicate decision cv changed"
    );
    anyhow::ensure!(
        first_objectives.map(f64::to_bits) == second_objectives.map(f64::to_bits),
        "duplicate decision objectives changed"
    );
    anyhow::ensure!(problem.eval_cache_hits() == 1, "expected one cache hit");
    Ok(())
}

fn colliding_transfer_decisions(capacity: usize) -> anyhow::Result<([f64; 3], [f64; 3])> {
    let first = repaired_transfer_decision(&[0.08, 0.99, 0.05]);
    let first_key = transfer_decision_key(&first);
    let mut probe = TransferMooEvalCache::new(capacity)?;
    probe.insert(TransferMooEvalCacheEntry {
        key: first_key,
        objectives: [0.0; TRANSFER_MOO_OBJECTIVES],
        cv: 0.0,
    })?;

    for step in 1_u32..=100_000 {
        let phase = 0.98 + 0.04 * (f64::from(step) / 100_000.0);
        let second = repaired_transfer_decision(&[first[0], phase, first[2]]);
        let second_key = transfer_decision_key(&second);
        if second_key == first_key || second[1].to_bits() == 1.0_f64.to_bits() {
            continue;
        }
        probe.insert(TransferMooEvalCacheEntry {
            key: second_key,
            objectives: [1.0; TRANSFER_MOO_OBJECTIVES],
            cv: 1.0,
        })?;
        if probe.get(first_key)?.is_none() {
            return Ok((first, second));
        }
    }

    anyhow::bail!("failed to find a direct-slot collision for capacity {capacity}")
}

#[test]
fn direct_plan_aba_preserves_terminal_order_and_plan_bits() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.max_phase_dv = 1.0e-6;
    prepare_single_pair_context(&mut ctx);
    let cache = RefCell::new(SolveLocalWorkCache::new());
    let a = [0.08, 0.99, 0.05];
    let b = [0.08, 1.01, 0.05];
    let mut plans = Vec::new();
    plans
        .try_reserve_exact(3)
        .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    for decision in [a, b, a] {
        plans.push(evaluate_plan_local(&decision, &ctx, false, &cache)?);
    }
    let [a_first, b_middle, a_last] = plans.as_slice() else {
        anyhow::bail!("A/B/A oracle must produce exactly three plans");
    };
    anyhow::ensure!(
        [
            a_first.timing_failure_reason,
            b_middle.timing_failure_reason,
            a_last.timing_failure_reason,
        ] == [
            crate::types::TimingFailureToken::PhaseDvBoundExceeded,
            crate::types::TimingFailureToken::PhaseDvBoundExceeded,
            crate::types::TimingFailureToken::PhaseDvBoundExceeded,
        ],
        "A/B/A terminal ordering changed: {:?}",
        [
            a_first.timing_failure_reason,
            b_middle.timing_failure_reason,
            a_last.timing_failure_reason,
        ]
    );
    assert_plan_result_fields_are_exhaustive(a_first);
    assert_plan_result_scalar_bits_equal(a_first, a_last);
    assert_plan_result_vector_bits_equal(a_first, a_last);
    assert_plan_result_metadata_equal(a_first, a_last);
    Ok(())
}

#[test]
fn transfer_moo_batch_rejects_malformed_shape_without_output_mutation() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let problem = TransferMooProblem::new(ctx, RefCell::new(FxHashMap::default()), policy)?;

    let mut objectives = [-7.0, 11.0];
    let mut constraint_violations = [-13.0];
    let objective_sentinel = objectives.map(f64::to_bits);
    let constraint_sentinel = constraint_violations.map(f64::to_bits);

    let wrong_width = problem.evaluate_batch(
        &[0.0, 1.0, 0.0],
        2,
        TRANSFER_MOO_OBJECTIVES,
        &mut objectives,
        &mut constraint_violations,
    );
    anyhow::ensure!(
        wrong_width.is_err(),
        "wrong decision width must fail closed"
    );
    anyhow::ensure!(
        objectives.map(f64::to_bits) == objective_sentinel
            && constraint_violations.map(f64::to_bits) == constraint_sentinel,
        "wrong decision width partially mutated caller outputs"
    );

    let short_decision_backing = problem.evaluate_batch(
        &[0.0, 1.0],
        TRANSFER_MOO_VARIABLES.len(),
        TRANSFER_MOO_OBJECTIVES,
        &mut objectives,
        &mut constraint_violations,
    );
    anyhow::ensure!(
        short_decision_backing.is_err(),
        "short decision backing must fail closed"
    );
    anyhow::ensure!(
        objectives.map(f64::to_bits) == objective_sentinel
            && constraint_violations.map(f64::to_bits) == constraint_sentinel,
        "short decision backing partially mutated caller outputs"
    );

    let mut short_objectives = [-17.0];
    let short_objective_sentinel = short_objectives.map(f64::to_bits);
    let short_objective_backing = problem.evaluate_batch(
        &[0.0, 1.0, 0.0],
        TRANSFER_MOO_VARIABLES.len(),
        TRANSFER_MOO_OBJECTIVES,
        &mut short_objectives,
        &mut constraint_violations,
    );
    anyhow::ensure!(
        short_objective_backing.is_err(),
        "short objective backing must fail closed"
    );
    anyhow::ensure!(
        short_objectives.map(f64::to_bits) == short_objective_sentinel
            && constraint_violations.map(f64::to_bits) == constraint_sentinel,
        "short objective backing partially mutated caller outputs"
    );
    Ok(())
}

#[cfg(feature = "bench-internal")]
#[test]
fn transfer_moo_policy_report_materializes_without_recompute() -> anyhow::Result<()> {
    let report = bench_transfer_moo_policy_report(TransferMooBenchPolicy::Full)?;

    anyhow::ensure!(
        report.materialize_plan_cache_hit_count > 0,
        "expected source materialization hits in policy report, got {report:?}"
    );
    anyhow::ensure!(
        report.materialize_plan_cache_miss_count == 0,
        "source materialization should not miss selected rows in policy report"
    );
    anyhow::ensure!(
        report.materialize_recompute_count == 0,
        "source materialization should not recompute selected rows in policy report"
    );
    anyhow::ensure!(
        report.materialize_all_exact_count == 1,
        "policy report should expose the all-source exact materialization pass"
    );
    Ok(())
}

#[test]
fn transfer_moo_preload_hits_eval_cache_for_seed() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    let plan_cache: TransferMooPlanCache = RefCell::new(FxHashMap::default());
    let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let problem = TransferMooProblem::new(ctx.clone(), plan_cache, policy)?;
    let raw_decision = [0.9, 1.0, 0.2];
    let repaired = repaired_transfer_decision(&raw_decision);
    let plan = evaluate_plan_local(
        &repaired,
        &ctx,
        false,
        &RefCell::new(SolveLocalWorkCache::new()),
    )?;
    let mut objectives = [0.0; TRANSFER_MOO_OBJECTIVES];

    let key = transfer_decision_key(&repaired);
    problem.preload_plan(key, &repaired, plan)?;
    let _ = problem.evaluate(&raw_decision, &mut objectives)?;

    anyhow::ensure!(problem.eval_cache_hits() == 1, "preloaded seed cache miss");
    Ok(())
}

#[test]
fn transfer_moo_source_materialization_takes_cached_plan() -> anyhow::Result<()> {
    let ctx = make_leo_ctx()?;
    let key = TransferDecisionKey([1, 2, 3]);
    let mut plan = PlanResult::invalid();
    plan.time2phase_ratio = 0.25;
    let mut map = FxHashMap::default();
    map.insert(key, plan);
    let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let problem = TransferMooProblem::new(ctx, RefCell::new(map), policy)?;

    let Some(taken) = problem.take_cached_source_plan(key) else {
        anyhow::bail!("expected cached plan");
    };

    anyhow::ensure!(
        taken.time2phase_ratio.to_bits() == 0.25_f64.to_bits(),
        "cached plan changed"
    );
    anyhow::ensure!(
        !problem.plan_cache_borrow().contains_key(&key),
        "source materialization should move selected plans out of the cache"
    );
    Ok(())
}

#[test]
fn malformed_oxymoo_population_public_backing_fails_closed() -> anyhow::Result<()> {
    let mut result = run_moo_population()?;
    anyhow::ensure!(
        !result.population.decisions.is_empty(),
        "real OxyMOO run must produce public decision backing to corrupt"
    );
    result.population.decisions.pop();
    let malformed_row = repaired_population_transfer_decision(&result.population, 0);
    anyhow::ensure!(
        matches!(
            malformed_row,
            Err(OxyMooCandidateMaterializationError::MalformedPopulation)
        ),
        "corrupted OxyMOO decision backing must fail closed as malformed"
    );

    let mut retained_seed = PlanResult::invalid();
    retained_seed.time2phase_ratio = 0.1;
    let mut derived_candidate = PlanResult::invalid();
    derived_candidate.time2phase_ratio = 0.2;
    let candidates = [retained_seed, derived_candidate];
    let candidate_bits = candidates
        .iter()
        .map(plan_bit_signature)
        .collect::<Vec<_>>();
    let malformed_boundary = handle_oxymoo_candidate_materialization_result(Err(
        OxyMooCandidateMaterializationError::MalformedPopulation,
    ));
    anyhow::ensure!(
        matches!(
            malformed_boundary,
            Err(InvalidTargetPropagationAuthorityCode::OptimizerFailure)
        ),
        "malformed OxyMOO population must propagate as optimizer failure"
    );
    anyhow::ensure!(
        candidates
            .iter()
            .map(plan_bit_signature)
            .collect::<Vec<_>>()
            == candidate_bits,
        "malformed OxyMOO population must not mutate caller candidates"
    );

    let arithmetic_boundary = handle_oxymoo_candidate_materialization_result(Err(
        OxyMooCandidateMaterializationError::ArithmeticOverflow,
    ));
    anyhow::ensure!(
        matches!(
            arithmetic_boundary,
            Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
        ),
        "OxyMOO arithmetic overflow must propagate as arithmetic overflow"
    );
    anyhow::ensure!(
        candidates
            .iter()
            .map(plan_bit_signature)
            .collect::<Vec<_>>()
            == candidate_bits,
        "OxyMOO arithmetic overflow must not mutate caller candidates"
    );

    let wrapped_arithmetic = anyhow::Error::new(crate::oxymoo::ArithmeticOverflow)
        .context("synthetic OxyMOO validation context");
    anyhow::ensure!(
        classify_oxymoo_optimizer_error(&wrapped_arithmetic)
            == OxyMooCandidateMaterializationError::ArithmeticOverflow,
        "context-wrapped OxyMOO arithmetic cause must retain its distinct route"
    );
    let wrapped_generic = anyhow::anyhow!("synthetic optimizer failure")
        .context("synthetic OxyMOO validation context");
    anyhow::ensure!(
        classify_oxymoo_optimizer_error(&wrapped_generic)
            == OxyMooCandidateMaterializationError::OptimizerFailure,
        "context-wrapped generic OxyMOO error must route to optimizer failure"
    );

    Ok(())
}

#[test]
fn transfer_moo_initial_buffer_overflow_preserves_existing_contents() -> anyhow::Result<()> {
    let mut decisions: Vec<f64> = vec![0.9, 1.0, 0.2];
    let mut seen = vec![TransferDecisionKey([1, 2, 3])];
    let decision_snapshot = decisions
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let seen_snapshot = seen.clone();

    let reservation =
        reserve_transfer_moo_initial_decision_buffers(&mut decisions, &mut seen, usize::MAX);

    anyhow::ensure!(
        reservation.is_err(),
        "overflowing initial seed count must return Err"
    );
    anyhow::ensure!(
        decisions
            .iter()
            .map(|value| value.to_bits())
            .eq(decision_snapshot),
        "overflowing initial seed count mutated decision buffer"
    );
    anyhow::ensure!(
        seen == seen_snapshot,
        "overflowing initial seed count mutated key buffer"
    );
    Ok(())
}

#[test]
fn transfer_moo_initial_decisions_are_repaired_and_deduplicated() -> anyhow::Result<()> {
    let mut duplicate_plan = synthetic_transfer_candidate(0.10, 3600.0, 4.0, [0.9, 1.0, 0.2]);
    duplicate_plan.valid = true;
    let mut unique_plan = synthetic_transfer_candidate(0.12, 4200.0, 5.0, [0.2, 1.0, 0.1]);
    unique_plan.valid = true;
    let ranked = vec![
        (
            SolverSeed {
                x: [0.9, 1.0, 0.2],
                warm_start_used: false,
            },
            duplicate_plan,
        ),
        (
            SolverSeed {
                x: [0.2, 1.0, 0.1],
                warm_start_used: false,
            },
            unique_plan,
        ),
    ];

    let decisions = transfer_moo_initial_decisions(Some([0.9, 1.0, 0.2]), &ranked)?;

    let [time0, phase0, wait0, time1, phase1, wait1] = decisions.as_slice() else {
        anyhow::bail!("expected two repaired OxyMOO decisions");
    };
    anyhow::ensure!((*time0 - 0.9).abs() <= 1e-12, "first time decision changed");
    anyhow::ensure!(
        (*phase0 - 1.0).abs() <= 1e-12,
        "first phase decision changed"
    );
    anyhow::ensure!(
        (*wait0 - 0.08).abs() <= 1e-12,
        "first wait decision changed"
    );
    anyhow::ensure!(
        (*time1 - 0.2).abs() <= 1e-12,
        "second time decision changed"
    );
    anyhow::ensure!(
        (*phase1 - 1.0).abs() <= 1e-12,
        "second phase decision changed"
    );
    anyhow::ensure!(
        (*wait1 - 0.1).abs() <= 1e-12,
        "second wait decision changed"
    );
    Ok(())
}

#[test]
fn warm_start_initial_decision_counter_tracks_retained_warm_seed() -> anyhow::Result<()> {
    let mut warm_plan = synthetic_transfer_candidate(0.10, 3600.0, 4.0, [0.9, 1.0, 0.2]);
    warm_plan.valid = true;
    let mut unique_plan = synthetic_transfer_candidate(0.12, 4200.0, 5.0, [0.2, 1.0, 0.1]);
    unique_plan.valid = true;
    let ranked = vec![
        (
            SolverSeed {
                x: [0.9, 1.0, 0.2],
                warm_start_used: true,
            },
            warm_plan,
        ),
        (
            SolverSeed {
                x: [0.2, 1.0, 0.1],
                warm_start_used: false,
            },
            unique_plan,
        ),
    ];

    let decisions = transfer_moo_initial_decisions(None, &ranked)?;
    anyhow::ensure!(
        count_warm_start_initial_decisions(&ranked, &decisions) == 1,
        "expected retained warm-start seed"
    );

    let [_, ranked_without_warm @ ..] = ranked.as_slice() else {
        anyhow::bail!("test fixture requires a non-warm seed");
    };
    let decisions_without_warm = transfer_moo_initial_decisions(None, ranked_without_warm)?;
    anyhow::ensure!(
        count_warm_start_initial_decisions(&ranked, &decisions_without_warm) == 0,
        "removed warm seed must not be counted"
    );
    Ok(())
}

#[test]
fn transfer_moo_preload_keeps_only_retained_initial_seed_plans() -> anyhow::Result<()> {
    let mut first_plan = synthetic_transfer_candidate(0.10, 3600.0, 4.0, [0.9, 1.0, 0.2]);
    first_plan.valid = true;
    let mut second_plan = synthetic_transfer_candidate(0.12, 4200.0, 5.0, [0.2, 1.0, 0.1]);
    second_plan.valid = true;
    let ranked = vec![
        (
            SolverSeed {
                x: [0.9, 1.0, 0.2],
                warm_start_used: false,
            },
            first_plan,
        ),
        (
            SolverSeed {
                x: [0.2, 1.0, 0.1],
                warm_start_used: false,
            },
            second_plan,
        ),
    ];
    let mut decisions = transfer_moo_initial_decisions(None, &ranked)?;
    truncate_transfer_moo_initial_decisions(&mut decisions, Some(1))?;
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    ctx.search_depth.oxymoo_policy =
        crate::types::OxyMooPolicy::FastPopulation20Generations3InitialBest1;
    let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let problem = TransferMooProblem::new(ctx, RefCell::new(FxHashMap::default()), policy)?;

    let preloaded = preload_retained_transfer_moo_plans(&problem, &ranked, &decisions)?;
    let mut objectives = [0.0; TRANSFER_MOO_OBJECTIVES];

    anyhow::ensure!(preloaded == 1, "expected one preloaded retained seed");
    let [first_ranked, second_ranked] = ranked.as_slice() else {
        anyhow::bail!("test fixture requires two ranked seeds");
    };
    let _ = problem.evaluate(&first_ranked.0.x, &mut objectives)?;
    anyhow::ensure!(problem.eval_cache_hits() == 1, "retained seed cache miss");
    let _ = problem.evaluate(&second_ranked.0.x, &mut objectives)?;
    anyhow::ensure!(
        problem.eval_cache_hits() == 1,
        "unretained seed unexpectedly hit cache"
    );
    Ok(())
}

#[test]
fn branch_expansion_sources_deduplicate_repaired_decisions_before_lambert_work(
) -> anyhow::Result<()> {
    let mut first = synthetic_transfer_candidate(0.10, 3600.0, 4.0, [0.9, 1.0, 0.2]);
    first.func_evals = 11;
    first.optimizer_func_evals = 13;
    first.optimizer_converged = true;
    first.warm_start_used = true;
    let duplicate = synthetic_transfer_candidate(0.11, 3700.0, 4.0, [0.9, 1.0, 0.2]);
    let nonfinite = synthetic_transfer_candidate(0.12, 3800.0, 4.0, [f64::NAN, 1.0, 0.1]);
    let unique = synthetic_transfer_candidate(0.13, 3900.0, 4.0, [0.2, 1.0, 0.1]);

    let sources = branch_expansion_sources_unique_by_repaired_decision(vec![
        first, duplicate, nonfinite, unique,
    ])?;

    anyhow::ensure!(sources.len() == 2, "unique branch source count");
    let [first_source, second_source] = sources.as_slice() else {
        anyhow::bail!("expected two unique branch-expansion sources");
    };
    anyhow::ensure!(
        (first_source.time2phase_ratio - 0.9).abs() <= 1e-12,
        "first source time ratio"
    );
    anyhow::ensure!(
        (first_source.waittime_ratio - 0.08).abs() <= 1e-12,
        "first source wait ratio"
    );
    anyhow::ensure!(
        first_source.func_evals == 11,
        "first source function evaluations"
    );
    anyhow::ensure!(
        first_source.optimizer_func_evals == 13,
        "first source optimizer evaluations"
    );
    anyhow::ensure!(first_source.optimizer_converged, "first source convergence");
    anyhow::ensure!(first_source.warm_start_used, "first source warm start");
    anyhow::ensure!(
        (second_source.time2phase_ratio - 0.2).abs() <= 1e-12,
        "second source time ratio"
    );
    anyhow::ensure!(
        (second_source.waittime_ratio - 0.1).abs() <= 1e-12,
        "second source wait ratio"
    );
    Ok(())
}

#[test]
fn delta_v_anchor_probes_skip_existing_repaired_decisions_before_eval() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    let cache = RefCell::new(SolveLocalWorkCache::new());
    let center = [0.50, 1.0, 0.50];
    let [center_time, center_phase, center_wait] = center;
    let mut duplicate_x = [center_time - 0.02, center_phase, center_wait];
    repair_transfer_decision(&mut duplicate_x);
    let mut candidates = vec![synthetic_transfer_candidate(0.10, 3600.0, 4.0, duplicate_x)];

    push_delta_v_anchor_probe_candidates(&mut candidates, &ctx, &cache, center, false)?;

    let duplicate_count = candidates
        .iter()
        .filter(|candidate| seed_is_duplicate(&transfer_plan_decision(candidate), &duplicate_x))
        .count();
    anyhow::ensure!(duplicate_count == 1, "duplicate probe count changed");
    Ok(())
}

#[test]
fn zero_rev_branch_expansion_does_not_reuse_repaired_source_physics() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.max_revs = 0;
    let canonical_but_repaired = synthetic_transfer_candidate(0.10, 3600.0, 4.0, [0.9, 1.0, 0.2]);
    let repaired = repaired_transfer_decision(&[
        canonical_but_repaired.time2phase_ratio,
        canonical_but_repaired.phase_sma_ratio,
        canonical_but_repaired.waittime_ratio,
    ]);
    let [_, _, repaired_wait] = repaired;
    anyhow::ensure!(
        repaired_wait.to_bits() != canonical_but_repaired.waittime_ratio.to_bits(),
        "test setup must exercise a repaired wait-time decision"
    );

    let mut metrics = VerifiedSupersetStageMetrics::default();
    let (_expanded, _branch_eval_s) = expand_lambert_branch_candidates_for_superset(
        &ctx,
        vec![canonical_but_repaired],
        &mut metrics,
    )?;

    anyhow::ensure!(
        metrics.branch_emitted_count == 0,
        "repaired sources must be recomputed through branch evaluation instead of reusing stale source physics"
    );
    Ok(())
}

#[test]
fn transfer_moo_eval_cache_stores_only_eval_metadata() -> anyhow::Result<()> {
    anyhow::ensure!(
        std::mem::size_of::<TransferMooEvalCacheEntry>()
            == std::mem::size_of::<TransferDecisionKey>()
                + std::mem::size_of::<[f64; TRANSFER_MOO_OBJECTIVES]>()
                + std::mem::size_of::<f64>(),
        "eval-cache entry must contain only key, objectives, and cv"
    );

    let mut cache = TransferMooEvalCache::new(1)?;
    let key = TransferDecisionKey([1, 2, 3]);
    let objectives = [1.0, 2.0];
    cache.insert(TransferMooEvalCacheEntry {
        key,
        objectives,
        cv: 0.0,
    })?;

    let Some(cached) = cache.get(key)? else {
        anyhow::bail!("expected cache hit");
    };

    anyhow::ensure!(
        cached.objectives.map(f64::to_bits) == objectives.map(f64::to_bits),
        "cached objectives changed"
    );
    anyhow::ensure!(
        cached.cv.to_bits() == 0.0_f64.to_bits(),
        "cached constraint violation changed"
    );
    anyhow::ensure!(cache.hits() == 1, "expected cache hit count of one");
    Ok(())
}

#[test]
fn transfer_moo_default_optimizer_is_nsga2() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    ctx.local_optimizer.seed = 123;
    let (population_size, generations) = transfer_moo_population_generations();

    let config = transfer_moo_config_with_initial_decisions(&ctx, Vec::new())?;
    let generation_batches = generations
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("test optimizer generation count must fit usize"))?;
    let expected_max_evaluations = population_size
        .checked_mul(generation_batches)
        .ok_or_else(|| anyhow::anyhow!("test optimizer evaluation count must fit usize"))?;

    anyhow::ensure!(config.population_size == population_size, "population size");
    anyhow::ensure!(config.generations == generations, "generation count");
    anyhow::ensure!(
        config.max_evaluations == Some(expected_max_evaluations),
        "maximum evaluations"
    );
    anyhow::ensure!(config.seed == 123, "optimizer seed");
    Ok(())
}

#[test]
fn oxymoo_policy_full_matches_default_fast_config() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    ctx.local_optimizer.seed = 123;

    let default_config = transfer_moo_config_with_initial_decisions(&ctx, Vec::new())?;
    let policy_config = transfer_moo_config_with_policy(&ctx, Vec::new(), TransferMooPolicy::Full)?;

    anyhow::ensure!(
        policy_config.population_size == default_config.population_size,
        "full policy population size"
    );
    anyhow::ensure!(
        policy_config.generations == default_config.generations,
        "full policy generation count"
    );
    anyhow::ensure!(
        policy_config.max_evaluations == default_config.max_evaluations,
        "full policy maximum evaluations"
    );
    anyhow::ensure!(policy_config.seed == 123, "full policy seed");
    Ok(())
}

#[test]
fn oxymoo_policy_variants_change_canonical_fast_mode() -> anyhow::Result<()> {
    let mut fast = make_leo_ctx()?;
    fast.sampling_mode = SamplingMode::Fast;
    let fast_policy = transfer_moo_config_with_policy(
        &fast,
        Vec::new(),
        TransferMooPolicy::FastPopulation20Generations3InitialBest1,
    )?;
    anyhow::ensure!(
        fast_policy.population_size == 20,
        "fast policy population size"
    );
    anyhow::ensure!(fast_policy.generations == 3, "fast policy generation count");
    let generation_batches = fast_policy
        .generations
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("test optimizer generation count must fit usize"))?;
    let expected_max_evaluations = fast_policy
        .population_size
        .checked_mul(generation_batches)
        .ok_or_else(|| anyhow::anyhow!("test optimizer evaluation count must fit usize"))?;
    anyhow::ensure!(
        fast_policy.max_evaluations == Some(expected_max_evaluations),
        "fast policy maximum evaluations"
    );
    Ok(())
}

#[test]
fn oxymoo_runtime_policy_reaches_production_config() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    ctx.search_depth.oxymoo_policy =
        crate::types::OxyMooPolicy::FastPopulation20Generations3InitialBest1;

    let config = transfer_moo_config_with_initial_decisions(&ctx, Vec::new())?;

    anyhow::ensure!(
        config.population_size == 20,
        "runtime policy population size"
    );
    anyhow::ensure!(config.generations == 3, "runtime policy generation count");
    let generation_batches = config
        .generations
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("test optimizer generation count must fit usize"))?;
    let expected_max_evaluations = config
        .population_size
        .checked_mul(generation_batches)
        .ok_or_else(|| anyhow::anyhow!("test optimizer evaluation count must fit usize"))?;
    anyhow::ensure!(
        config.max_evaluations == Some(expected_max_evaluations),
        "runtime policy maximum evaluations"
    );
    Ok(())
}

#[test]
fn transfer_moo_problem_cache_capacity_follows_runtime_policy() -> anyhow::Result<()> {
    let mut full = make_leo_ctx()?;
    full.sampling_mode = SamplingMode::Fast;
    let full_policy = TransferMooPolicy::from(full.search_depth.oxymoo_policy);
    let full_problem =
        TransferMooProblem::new(full, RefCell::new(FxHashMap::default()), full_policy)?;

    let mut fast = make_leo_ctx()?;
    fast.sampling_mode = SamplingMode::Fast;
    fast.search_depth.oxymoo_policy =
        crate::types::OxyMooPolicy::FastPopulation20Generations3InitialBest1;
    let fast_policy = TransferMooPolicy::from(fast.search_depth.oxymoo_policy);
    let fast_problem =
        TransferMooProblem::new(fast, RefCell::new(FxHashMap::default()), fast_policy)?;

    anyhow::ensure!(
        fast_problem.eval_cache_capacity() == 256,
        "fast policy cache capacity changed"
    );
    anyhow::ensure!(
        fast_problem.eval_cache_capacity() < full_problem.eval_cache_capacity(),
        "fast policy cache must remain smaller than full policy"
    );
    Ok(())
}

#[test]
fn transfer_moo_config_preserves_initial_decisions() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    ctx.local_optimizer.seed = 123;
    let initial = vec![0.1, 1.0, 0.2, 0.4, 1.1, 0.1];

    let config = transfer_moo_config_with_initial_decisions(&ctx, initial.clone())?;

    anyhow::ensure!(
        config.initial_decisions == initial,
        "initial decisions changed"
    );
    Ok(())
}

#[test]
fn objective_aware_seed_retention_uses_combined_time_velocity_objective() {
    let low_dv_slow = (
        SolverSeed {
            x: [0.1, 1.0, 0.1],
            warm_start_used: false,
        },
        synthetic_transfer_candidate(0.10, 7200.0, 4.0, [0.1, 1.0, 0.1]),
    );
    let best_ratio = (
        SolverSeed {
            x: [0.2, 1.0, 0.1],
            warm_start_used: false,
        },
        synthetic_transfer_candidate(0.18, 2000.0, 8.0, [0.2, 1.0, 0.1]),
    );
    let high_rel_but_bad_ratio = (
        SolverSeed {
            x: [0.3, 1.0, 0.1],
            warm_start_used: false,
        },
        synthetic_transfer_candidate(0.20, 9000.0, 9.0, [0.3, 1.0, 0.1]),
    );
    let dominated = (
        SolverSeed {
            x: [0.4, 1.0, 0.1],
            warm_start_used: false,
        },
        synthetic_transfer_candidate(0.30, 8000.0, 2.0, [0.4, 1.0, 0.1]),
    );
    let eligible = vec![
        low_dv_slow.clone(),
        best_ratio.clone(),
        high_rel_but_bad_ratio.clone(),
        dominated.clone(),
    ];
    let mut selected = vec![low_dv_slow];

    retain_objective_aware_seed_candidates(&mut selected, &eligible);

    let selected_x: std::collections::BTreeSet<_> = selected
        .iter()
        .map(|(seed, _)| seed.x.map(f64::to_bits))
        .collect();
    assert!(selected_x.contains(&best_ratio.0.x.map(f64::to_bits)));
    assert!(!selected_x.contains(&high_rel_but_bad_ratio.0.x.map(f64::to_bits)));
    assert!(!selected_x.contains(&dominated.0.x.map(f64::to_bits)));
}

#[test]
fn oxymoo_pair_solver_returns_verified_nondominated_front() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    let raw_oxymoo = run_oxymoo_transfer_candidates(&ctx, false)?;
    anyhow::ensure!(
        raw_oxymoo
            .iter()
            .any(|candidate| candidate.optimizer_func_evals > 0),
        "expected direct OxyMOO population candidates"
    );

    let front = solve_plan_oxymoo_front(&mut ctx, None)?;

    anyhow::ensure!(!front.is_empty(), "expected OxyMOO transfer front");
    for (idx, candidate) in front.candidates.iter().enumerate() {
        anyhow::ensure!(candidate.valid, "OxyMOO front candidate {idx} is invalid");
        anyhow::ensure!(
            candidate.transfer_objectives().is_finite(),
            "OxyMOO front candidate {idx} has non-finite objectives"
        );
        for (other_idx, other) in front.candidates.iter().enumerate() {
            if idx != other_idx {
                anyhow::ensure!(
                    !transfer_candidate_dominates(other, candidate),
                    "OxyMOO front candidate should not be dominated"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn delta_v_anchor_polish_does_not_worsen_best_seed() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    let local_cache = RefCell::new(SolveLocalWorkCache::new());
    let (ranked_seeds, warm_start_consumed, _seed_timing) =
        rank_seed_candidates_for_front(&ctx, None, &local_cache)?;

    let best_seed_dv = ranked_seeds
        .iter()
        .filter_map(|(_, plan)| {
            transfer_candidate_is_objective_finite(plan).then_some(plan.total_dv())
        })
        .fold(f64::INFINITY, f64::min);
    let anchors =
        run_delta_v_anchor_candidates(&ctx, &ranked_seeds, warm_start_consumed, &local_cache)?;
    let best_anchor_dv = anchors
        .iter()
        .filter_map(|plan| transfer_candidate_is_objective_finite(plan).then_some(plan.total_dv()))
        .fold(f64::INFINITY, f64::min);

    anyhow::ensure!(best_seed_dv.is_finite(), "expected a finite seed dV");
    anyhow::ensure!(
        anchors.iter().any(|plan| plan.optimizer_func_evals > 0),
        "expected at least one locally polished dV anchor"
    );
    anyhow::ensure!(
        best_anchor_dv <= best_seed_dv + 1e-9,
        "anchor should match or improve best seed dV: seed={best_seed_dv} anchor={best_anchor_dv}"
    );
    Ok(())
}

#[test]
fn final_candidate_delta_v_polish_keeps_candidates_local_and_nonworse() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    let local_cache = RefCell::new(SolveLocalWorkCache::new());
    let (ranked_seeds, _warm_start_consumed, _seed_timing) =
        rank_seed_candidates_for_front(&ctx, None, &local_cache)?;
    let mut candidates: Vec<PlanResult> = ranked_seeds
        .iter()
        .filter_map(|(_, plan)| {
            transfer_candidate_is_objective_finite(plan).then_some(plan.clone())
        })
        .take(2)
        .collect();
    anyhow::ensure!(
        candidates.len() == 2,
        "test needs two finite seed candidates"
    );

    let before: Vec<([f64; 3], f64)> = candidates
        .iter()
        .map(|plan| (transfer_plan_decision(plan), plan.total_dv()))
        .collect();

    polish_transfer_candidates_delta_v(
        &mut candidates,
        &ctx,
        &local_cache,
        PolishScopePolicy::Full,
    )?;

    anyhow::ensure!(
        candidates.len() == before.len(),
        "polish changed candidate count"
    );
    for (candidate, (start, before_dv)) in candidates.iter().zip(before.iter()) {
        anyhow::ensure!(
            transfer_candidate_is_objective_finite(candidate),
            "polish produced non-finite candidate"
        );
        anyhow::ensure!(
            candidate.total_dv() <= before_dv + 1e-9,
            "candidate polish worsened delta-V: before={before_dv} after={}",
            candidate.total_dv()
        );
        let polished = transfer_plan_decision(candidate);
        for (idx, ((polished_value, start_value), radius)) in polished
            .iter()
            .zip(start.iter())
            .zip(FINAL_CANDIDATE_POLISH_RADIUS)
            .enumerate()
        {
            anyhow::ensure!(
                (*polished_value - *start_value).abs() <= radius + 1e-12,
                "candidate polish moved outside local radius at dim {idx}: start={start:?} polished={polished:?}"
            );
        }
    }
    Ok(())
}

/// The refinement appends; it never rewrites what the pool already ranked,
/// and anything it appends is strictly better than the incumbent it came
/// from and still local to it.
#[test]
fn release_epoch_line_refinement_only_appends_strictly_better_local_candidates(
) -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    ctx.target_propagation_authority = crate::types::TargetPropagationAuthority::MfJ2;
    prepare_single_pair_context(&mut ctx);

    let cache = RefCell::new(SolveLocalWorkCache::new());
    let starts = [
        [0.08, 1.0, 0.05],
        [0.22, 1.0, 0.20],
        [0.40, 1.0, 0.20],
        [0.60, 1.0, 0.05],
        [0.05, 1.0, 0.40],
    ];
    let mut candidates: Vec<PlanResult> = starts
        .iter()
        .map(|x| evaluate_plan_local(x, &ctx, false, &cache))
        .collect::<Result<Vec<_>, _>>()?;
    let before = candidates.clone();

    refine_release_epoch_line(&mut candidates, &ctx, &cache)?;

    anyhow::ensure!(
        candidates.len() >= before.len(),
        "refinement removed an existing candidate"
    );
    for (was, now) in before.iter().zip(candidates.iter()) {
        anyhow::ensure!(
            transfer_plan_decision(was).map(f64::to_bits)
                == transfer_plan_decision(now).map(f64::to_bits),
            "the refinement rewrote an existing candidate"
        );
        anyhow::ensure!(
            was.total_dv().to_bits() == now.total_dv().to_bits(),
            "the refinement rewrote an existing candidate delta-V"
        );
    }

    // Worst incumbent the refinement was allowed to start from.
    let mut incumbent_dv: Vec<f64> = before
        .iter()
        .filter(|plan| transfer_candidate_is_objective_finite(plan))
        .map(PlanResult::total_dv)
        .collect();
    incumbent_dv.sort_by(f64::total_cmp);
    let refined_count = RELEASE_EPOCH_REFINE_TOP_K.min(incumbent_dv.len());
    let Some(worst_start_index) = refined_count.checked_sub(1) else {
        anyhow::bail!("release-epoch fixture must retain at least one finite incumbent");
    };
    let Some(&worst_start) = incumbent_dv.get(worst_start_index) else {
        anyhow::bail!("refinement boundary must resolve to a retained incumbent");
    };
    let [time_radius, _, _] = FINAL_CANDIDATE_POLISH_RADIUS;
    let reach = RELEASE_EPOCH_REFINE_HALF + time_radius;

    let Some(appended_candidates) = candidates.get(before.len()..) else {
        anyhow::bail!("release-epoch refinement must not remove candidates");
    };
    for added in appended_candidates {
        anyhow::ensure!(
            transfer_candidate_is_objective_finite(added),
            "refinement appended non-finite candidate"
        );
        anyhow::ensure!(
            added.total_dv() < worst_start,
            "appended candidate {} is not better than the worst refined start {worst_start}",
            added.total_dv()
        );
        let x = transfer_plan_decision(added);
        anyhow::ensure!(
            before.iter().any(|start| {
                let s = transfer_plan_decision(start);
                let [x_time, _, x_wait] = x;
                let [start_time, _, start_wait] = s;
                (x_time - start_time).abs() <= reach + 1e-12
                    && (x_wait - start_wait).abs() <= reach + 1e-12
            }),
            "appended candidate {x:?} is not local to any pool entry"
        );
    }
    Ok(())
}

#[test]
fn final_candidate_delta_v_polish_skips_duplicate_repaired_decisions() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    let local_cache = RefCell::new(SolveLocalWorkCache::new());
    let (ranked_seeds, _warm_start_consumed, _seed_timing) =
        rank_seed_candidates_for_front(&ctx, None, &local_cache)?;
    let Some(seed) = ranked_seeds
        .iter()
        .find_map(|(_, plan)| transfer_candidate_is_objective_finite(plan).then_some(plan.clone()))
    else {
        anyhow::bail!("test needs one finite seed candidate");
    };
    let mut duplicate = seed.clone();
    duplicate.polish_skipped = false;
    let mut candidates = vec![seed, duplicate];

    polish_transfer_candidates_delta_v(
        &mut candidates,
        &ctx,
        &local_cache,
        PolishScopePolicy::Full,
    )?;

    anyhow::ensure!(candidates.len() == 2, "polish changed candidate count");
    anyhow::ensure!(
        candidates
            .iter()
            .filter(|candidate| candidate.polish_skipped)
            .count()
            == 1,
        "one exact duplicate repaired decision should skip final polish"
    );
    Ok(())
}

/// Perf #12 regression guard: the pre-polish snapshot for the
/// degenerate-front fallback is taken lazily — only when the `NdEpsilon`
/// scope mask actually skips a candidate — and, when taken, it is
/// captured before any mutation so it equals the pre-polish pool
/// byte-for-byte. Scope-skipped candidates themselves are only flagged
/// (`polish_skipped = true`), never otherwise mutated.
#[test]
fn final_candidate_delta_v_polish_snapshot_is_lazy_and_pre_mutation() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    let local_cache = RefCell::new(SolveLocalWorkCache::new());
    let (ranked_seeds, _warm_start_consumed, _seed_timing) =
        rank_seed_candidates_for_front(&ctx, None, &local_cache)?;
    let Some(seed) = ranked_seeds
        .iter()
        .find_map(|(_, plan)| transfer_candidate_is_objective_finite(plan).then_some(plan.clone()))
    else {
        anyhow::bail!("test needs one finite seed candidate");
    };

    // Craft a candidate the epsilon mask must skip: same geometry as the
    // seed but epsilon-dominated on BOTH objectives (dv worse by >> 0.05
    // km/s, time worse by >> 5%) at no-better constraint violation, with
    // a distinct decision key so the duplicate filter does not fire
    // first (the mask check only runs for non-duplicates).
    let mut dominated = seed.clone();
    let [_, lower_phase_sma, _] = SINGLE_PAIR_LOWER_BOUNDS;
    let [_, upper_phase_sma, _] = SINGLE_PAIR_UPPER_BOUNDS;
    dominated.phase_sma_ratio = if seed.phase_sma_ratio <= 0.5 * (lower_phase_sma + upper_phase_sma)
    {
        seed.phase_sma_ratio + 0.013
    } else {
        seed.phase_sma_ratio - 0.013
    };
    dominated.transfer_dv_norm = seed.transfer_dv_norm + 1.0;
    dominated.tof = seed.tof + seed.total_time().max(1.0);
    anyhow::ensure!(
        transfer_candidate_is_objective_finite(&dominated),
        "dominated fixture must remain objective-finite"
    );

    // Zero scope skips => no snapshot is allocated even when requested.
    let mut lone = vec![seed.clone()];
    let (lone_stats, lone_snapshot) = polish_transfer_candidates_delta_v_with_pre_polish_snapshot(
        &mut lone,
        &ctx,
        &RefCell::new(SolveLocalWorkCache::new()),
        PolishScopePolicy::NdEpsilon,
        true,
    )?;
    anyhow::ensure!(
        lone_stats.scope_skipped_count == 0,
        "lone candidate unexpectedly scope-skipped"
    );
    anyhow::ensure!(
        lone_snapshot.is_none(),
        "no scope skips -> the fallback cannot fire -> no snapshot clone"
    );

    // One scope skip => snapshot exists and equals the pre-polish pool.
    let mut candidates = vec![seed.clone(), dominated.clone()];
    let pre_polish: Vec<PlanResult> = candidates.clone();
    let (stats, snapshot) = polish_transfer_candidates_delta_v_with_pre_polish_snapshot(
        &mut candidates,
        &ctx,
        &RefCell::new(SolveLocalWorkCache::new()),
        PolishScopePolicy::NdEpsilon,
        true,
    )?;
    anyhow::ensure!(
        stats.scope_skipped_count == 1,
        "the dominated candidate must be scope-skipped"
    );
    let Some(snapshot) = snapshot else {
        anyhow::bail!("scope skip present -> snapshot must exist");
    };
    anyhow::ensure!(
        snapshot.len() == pre_polish.len(),
        "snapshot changed candidate count"
    );
    for (snap, pre) in snapshot.iter().zip(pre_polish.iter()) {
        anyhow::ensure!(snap.cost.to_bits() == pre.cost.to_bits(), "snapshot cost");
        anyhow::ensure!(
            transfer_plan_decision(snap).map(f64::to_bits)
                == transfer_plan_decision(pre).map(f64::to_bits),
            "snapshot decision"
        );
        anyhow::ensure!(
            snap.total_dv().to_bits() == pre.total_dv().to_bits(),
            "snapshot total dv"
        );
        anyhow::ensure!(snap.tof.to_bits() == pre.tof.to_bits(), "snapshot tof");
        anyhow::ensure!(
            snap.polish_skipped == pre.polish_skipped,
            "snapshot polish_skipped flag must be pre-polish"
        );
    }
    // The scope-skipped candidate is flagged but otherwise unmutated.
    let [_, scope_skipped_candidate] = candidates.as_slice() else {
        anyhow::bail!("scope-skip fixture must retain two candidates");
    };
    anyhow::ensure!(
        scope_skipped_candidate.polish_skipped,
        "scope-skipped candidate must be flagged"
    );
    anyhow::ensure!(
        scope_skipped_candidate.total_dv().to_bits() == dominated.total_dv().to_bits(),
        "scope-skipped candidate total dv changed"
    );
    anyhow::ensure!(
        scope_skipped_candidate.tof.to_bits() == dominated.tof.to_bits(),
        "scope-skipped candidate tof changed"
    );
    anyhow::ensure!(
        transfer_plan_decision(scope_skipped_candidate).map(f64::to_bits)
            == transfer_plan_decision(&dominated).map(f64::to_bits),
        "scope-skipped candidate decision changed"
    );

    // The plain wrapper (no snapshot requested) produces identical stats
    // and an identical polished pool on a fresh cache.
    let mut wrapper_pool = vec![seed, dominated];
    let wrapper_stats = polish_transfer_candidates_delta_v(
        &mut wrapper_pool,
        &ctx,
        &RefCell::new(SolveLocalWorkCache::new()),
        PolishScopePolicy::NdEpsilon,
    )?;
    anyhow::ensure!(
        wrapper_stats.scope_skipped_count == stats.scope_skipped_count,
        "wrapper changed scope-skip count"
    );
    anyhow::ensure!(
        wrapper_stats.dv_improvement_max_km_s.to_bits() == stats.dv_improvement_max_km_s.to_bits(),
        "wrapper changed maximum dv improvement"
    );
    anyhow::ensure!(
        wrapper_pool.len() == candidates.len(),
        "wrapper changed candidate count"
    );
    for (a, b) in wrapper_pool.iter().zip(candidates.iter()) {
        anyhow::ensure!(a.cost.to_bits() == b.cost.to_bits(), "wrapper cost");
        anyhow::ensure!(
            transfer_plan_decision(a).map(f64::to_bits)
                == transfer_plan_decision(b).map(f64::to_bits),
            "wrapper decision"
        );
        anyhow::ensure!(
            a.total_dv().to_bits() == b.total_dv().to_bits(),
            "wrapper total dv"
        );
        anyhow::ensure!(a.polish_skipped == b.polish_skipped, "wrapper polish flag");
    }
    Ok(())
}

/// Design item d machinery: the tuned `nd_epsilon_dv_mps<N>` policy
/// applies the identical mask rule with a token-supplied dv margin.
/// `nd_epsilon_dv_mps50` must reproduce `nd_epsilon` bit-identically;
/// a margin wider than a crafted candidate's dv gap must deactivate the
/// skip (gated path activates AND differs). No default changes: `full`
/// stays the kernel default and `nd_epsilon` keeps its constants.
#[test]
fn polish_scope_tuned_epsilon_token_parses_and_retunes_mask() -> anyhow::Result<()> {
    // Token grammar (round-trips through the same normalizer the Python
    // runtime_compile validation mirrors).
    anyhow::ensure!(
        PolishScopePolicy::from_token("nd_epsilon_dv_mps50")
            == Some(PolishScopePolicy::NdEpsilonTuned { dv_eps_m_per_s: 50 }),
        "50 m/s epsilon token must parse"
    );
    anyhow::ensure!(
        PolishScopePolicy::from_token(" ND-EPSILON-DV-MPS120 ")
            == Some(PolishScopePolicy::NdEpsilonTuned {
                dv_eps_m_per_s: 120
            }),
        "normalized 120 m/s epsilon token must parse"
    );
    anyhow::ensure!(
        PolishScopePolicy::from_token("nd_epsilon_dv_mps0").is_none(),
        "zero epsilon token must fail"
    );
    anyhow::ensure!(
        PolishScopePolicy::from_token("nd_epsilon_dv_mps5001").is_none(),
        "out-of-range epsilon token must fail"
    );
    anyhow::ensure!(
        PolishScopePolicy::from_token("nd_epsilon_dv_mps").is_none(),
        "incomplete epsilon token must fail"
    );
    anyhow::ensure!(
        PolishScopePolicy::from_token("nd_epsilon_dv_mps1.5").is_none(),
        "fractional epsilon token must fail"
    );
    anyhow::ensure!(
        PolishScopePolicy::from_token("nd_epsilon_dv_mps-3").is_none(),
        "negative epsilon token must fail"
    );
    // 50 m/s reproduces the historical constant bit-identically.
    anyhow::ensure!(
        (f64::from(50_u32) / 1000.0).to_bits() == POLISH_SCOPE_ND_EPS_DV_KM_S.to_bits(),
        "50 m/s must preserve the historical epsilon bits"
    );

    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    let local_cache = RefCell::new(SolveLocalWorkCache::new());
    let (ranked_seeds, _warm_start_consumed, _seed_timing) =
        rank_seed_candidates_for_front(&ctx, None, &local_cache)?;
    let Some(seed) = ranked_seeds
        .iter()
        .find_map(|(_, plan)| transfer_candidate_is_objective_finite(plan).then_some(plan.clone()))
    else {
        anyhow::bail!("test needs one finite seed candidate");
    };

    // Same crafted epsilon-dominated candidate as the lazy-snapshot test:
    // ~1 km/s worse dv, >>5% worse time, distinct decision key.
    let [_, phase_lower, _] = SINGLE_PAIR_LOWER_BOUNDS;
    let [_, phase_upper, _] = SINGLE_PAIR_UPPER_BOUNDS;
    let mut dominated = seed.clone();
    dominated.phase_sma_ratio = if seed.phase_sma_ratio <= 0.5 * (phase_lower + phase_upper) {
        seed.phase_sma_ratio + 0.013
    } else {
        seed.phase_sma_ratio - 0.013
    };
    dominated.transfer_dv_norm = seed.transfer_dv_norm + 1.0;
    dominated.tof = seed.tof + seed.total_time().max(1.0);
    anyhow::ensure!(
        transfer_candidate_is_objective_finite(&dominated),
        "dominated fixture must remain objective-finite"
    );
    let dv_gap = dominated.total_dv() - seed.total_dv();
    anyhow::ensure!(
        dv_gap > POLISH_SCOPE_ND_EPS_DV_KM_S && dv_gap < 5.0,
        "fixture dv gap must straddle the baseline and wide epsilons"
    );

    // Baseline nd_epsilon: the dominated candidate is scope-skipped.
    let mut base_pool = vec![seed.clone(), dominated.clone()];
    let base_stats = polish_transfer_candidates_delta_v(
        &mut base_pool,
        &ctx,
        &RefCell::new(SolveLocalWorkCache::new()),
        PolishScopePolicy::NdEpsilon,
    )?;
    anyhow::ensure!(
        base_stats.scope_skipped_count == 1,
        "baseline epsilon must scope-skip the dominated candidate"
    );

    // nd_epsilon_dv_mps50 must be bit-identical to nd_epsilon.
    let mut tuned50_pool = vec![seed.clone(), dominated.clone()];
    let tuned50_stats = polish_transfer_candidates_delta_v(
        &mut tuned50_pool,
        &ctx,
        &RefCell::new(SolveLocalWorkCache::new()),
        PolishScopePolicy::NdEpsilonTuned { dv_eps_m_per_s: 50 },
    )?;
    anyhow::ensure!(
        tuned50_stats.scope_skipped_count == base_stats.scope_skipped_count,
        "50 m/s epsilon changed scope-skip count"
    );
    anyhow::ensure!(
        tuned50_stats.dv_improvement_max_km_s.to_bits()
            == base_stats.dv_improvement_max_km_s.to_bits(),
        "50 m/s epsilon changed maximum dv improvement"
    );
    anyhow::ensure!(
        tuned50_pool.len() == base_pool.len(),
        "50 m/s epsilon changed pool count"
    );
    for (a, b) in tuned50_pool.iter().zip(base_pool.iter()) {
        anyhow::ensure!(a.cost.to_bits() == b.cost.to_bits(), "50 m/s cost");
        anyhow::ensure!(
            transfer_plan_decision(a).map(f64::to_bits)
                == transfer_plan_decision(b).map(f64::to_bits),
            "50 m/s decision"
        );
        anyhow::ensure!(
            a.total_dv().to_bits() == b.total_dv().to_bits(),
            "50 m/s total dv"
        );
        anyhow::ensure!(a.polish_skipped == b.polish_skipped, "50 m/s polish flag");
    }

    // A 5 km/s margin exceeds the crafted ~1 km/s dv gap, so the tuned
    // mask no longer treats the candidate as provably-unrescuable: the
    // skip deactivates and the candidate is polished instead (differs
    // from the nd_epsilon baseline).
    let mut wide_pool = vec![seed, dominated];
    let wide_stats = polish_transfer_candidates_delta_v(
        &mut wide_pool,
        &ctx,
        &RefCell::new(SolveLocalWorkCache::new()),
        PolishScopePolicy::NdEpsilonTuned {
            dv_eps_m_per_s: 5000,
        },
    )?;
    anyhow::ensure!(
        wide_stats.scope_skipped_count == 0,
        "wide epsilon must deactivate scope skip"
    );
    let [_, wide_candidate] = wide_pool.as_slice() else {
        anyhow::bail!("wide-scope fixture must retain two candidates");
    };
    anyhow::ensure!(
        !wide_candidate.polish_skipped,
        "wide margin must polish the previously scope-skipped candidate"
    );
    Ok(())
}

#[test]
fn transfer_objectives_report_combined_time_velocity_objective() {
    let plan = synthetic_transfer_candidate(0.2, 3600.0, 7.5, [0.1, 1.0, 0.0]);
    let objectives = plan.transfer_objectives();

    assert_eq!(objectives.total_dv.to_bits(), 0.2_f64.to_bits());
    assert_eq!(objectives.total_time.to_bits(), 3600.0_f64.to_bits());
    assert_eq!(objectives.relative_velocity.to_bits(), 7.5_f64.to_bits());
    assert_eq!(
        objectives.time_per_relative_velocity_s_per_km_s.to_bits(),
        480.0_f64.to_bits()
    );
    assert_eq!(
        plan.time_per_relative_velocity_s_per_km_s().to_bits(),
        480.0_f64.to_bits()
    );
    assert_eq!(
        objectives.as_minimization_array().map(f64::to_bits),
        [0.2_f64.to_bits(), 480.0_f64.to_bits()]
    );
}

#[test]
fn transfer_objectives_reject_nonpositive_relative_velocity() -> anyhow::Result<()> {
    let plan = synthetic_transfer_candidate(0.2, 3600.0, 0.0, [0.1, 1.0, 0.0]);
    let objectives = plan.transfer_objectives();
    let front = filter_nondominated_transfer_candidates(vec![plan])?;

    anyhow::ensure!(
        !objectives.is_finite(),
        "nonpositive relative velocity is nonfinite"
    );
    anyhow::ensure!(
        objectives.time_per_relative_velocity_s_per_km_s.is_nan(),
        "nonpositive relative velocity ratio is NaN"
    );
    anyhow::ensure!(
        front.candidates.is_empty(),
        "nonfinite row must not enter front"
    );
    Ok(())
}

#[test]
fn transfer_front_filters_two_objective_combined_dominated_candidates() -> anyhow::Result<()> {
    let low_dv_bad_ratio = synthetic_transfer_candidate(0.10, 7200.0, 2.0, [0.1, 1.0, 0.1]);
    let longer_high_rel = synthetic_transfer_candidate(0.20, 3600.0, 9.0, [0.2, 1.0, 0.0]);
    let dominated_by_ratio = synthetic_transfer_candidate(0.25, 4000.0, 4.0, [0.3, 1.0, 0.2]);
    let duplicate_ratio = synthetic_transfer_candidate(0.20, 7200.0, 18.0, [0.4, 1.0, 0.2]);

    let front = filter_nondominated_transfer_candidates(vec![
        dominated_by_ratio,
        longer_high_rel.clone(),
        low_dv_bad_ratio.clone(),
        duplicate_ratio,
    ])?;

    anyhow::ensure!(front.len() == 2, "two-objective front row count");
    anyhow::ensure!(
        front.candidates.iter().any(|p| {
            (p.total_dv() - low_dv_bad_ratio.total_dv()).abs() < 1e-12
                && (p.time_per_relative_velocity_s_per_km_s()
                    - low_dv_bad_ratio.time_per_relative_velocity_s_per_km_s())
                .abs()
                    < 1e-6
        }),
        "low-dv, bad-ratio candidate missing"
    );
    anyhow::ensure!(
        front.candidates.iter().any(|p| {
            (p.total_dv() - longer_high_rel.total_dv()).abs() < 1e-12
                && (p.time_per_relative_velocity_s_per_km_s()
                    - longer_high_rel.time_per_relative_velocity_s_per_km_s())
                .abs()
                    < 1e-6
        }),
        "longer, high-relative-velocity candidate missing"
    );
    Ok(())
}

#[test]
fn auto_local_optimizer_uses_nm_for_easy_and_pso_for_hard() {
    assert_eq!(
        auto_local_optimizer_kind(TransferComplexity::Easy, 0.3),
        LocalOptimizerKind::NelderMead
    );
    assert_eq!(
        auto_local_optimizer_kind(TransferComplexity::Hard, 0.3),
        LocalOptimizerKind::Pso
    );
    assert_eq!(
        auto_local_optimizer_kind(TransferComplexity::Extreme, 0.03),
        LocalOptimizerKind::NelderMead
    );
}

#[test]
fn fixed_local_optimizer_choice_overrides_auto_policy() {
    let config = TransferLocalOptimizerConfig {
        choice: TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::Lbfgs),
        tune: TuneLevel::Conservative,
        seed: 11,
    };

    assert_eq!(
        resolve_local_optimizer_kind(config, TransferComplexity::Hard, 0.4),
        LocalOptimizerKind::Lbfgs
    );
}

#[test]
fn fixed_nm_and_pso_solve_through_same_transfer_config() -> anyhow::Result<()> {
    for kind in [LocalOptimizerKind::NelderMead, LocalOptimizerKind::Pso] {
        let mut ctx = make_leo_ctx()?;
        ctx.local_optimizer = TransferLocalOptimizerConfig {
            choice: TransferLocalOptimizerChoice::Fixed(kind),
            tune: TuneLevel::Default,
            seed: 11,
        };

        let result = solve_plan_representative(&mut ctx)?;

        anyhow::ensure!(result.valid, "{kind:?} result: {result:?}");
        anyhow::ensure!(result.cost < INVALID_COST, "{kind:?} result: {result:?}");
        anyhow::ensure!(
            result.optimizer_func_evals > 0,
            "{kind:?} result: {result:?}"
        );
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OxymooPopulationRowParity {
    decision_bits: [u64; 3],
    objective_bits: [u64; TRANSFER_MOO_OBJECTIVES],
    constraint_violation_bits: u64,
    rank: usize,
    crowding_bits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OxymooMaterializedPlanParity {
    flags: [bool; 4],
    decision_bits: [u64; 3],
    cost_bits: u64,
    total_dv_bits: u64,
    total_time_bits: u64,
    time_per_relative_velocity_bits: u64,
    branch_rev: i32,
    best_m: i32,
    optimizer_func_evals: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OxymooMaterializationParitySnapshot {
    population_rows: Vec<OxymooPopulationRowParity>,
    fronts: Vec<Vec<usize>>,
    materialized_plans: Vec<OxymooMaterializedPlanParity>,
    source_materialized_plans: Vec<OxymooMaterializedPlanParity>,
    population_hash: u64,
    materialized_hash: u64,
    source_materialized_hash: u64,
    source_missing_count: usize,
    nsga_evaluations: usize,
    nsga_generations: usize,
}

fn mix_parity_hash(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(0x100_0000_01b3);
}

fn push_oxymoo_materialized_plan_parity(
    plans: &mut Vec<OxymooMaterializedPlanParity>,
    hash: &mut u64,
    plan: &PlanResult,
) -> anyhow::Result<()> {
    let plan_snapshot = OxymooMaterializedPlanParity {
        flags: [
            plan.valid,
            plan.branch_low_path,
            plan.optimizer_converged,
            plan.warm_start_used,
        ],
        decision_bits: [
            plan.time2phase_ratio.to_bits(),
            plan.phase_sma_ratio.to_bits(),
            plan.waittime_ratio.to_bits(),
        ],
        cost_bits: plan.cost.to_bits(),
        total_dv_bits: plan.total_dv().to_bits(),
        total_time_bits: plan.total_time().to_bits(),
        time_per_relative_velocity_bits: plan.time_per_relative_velocity_s_per_km_s().to_bits(),
        branch_rev: plan.branch_rev,
        best_m: plan.best_M,
        optimizer_func_evals: plan.optimizer_func_evals,
    };
    let [valid, branch_low_path, optimizer_converged, warm_start_used] = plan_snapshot.flags;
    mix_parity_hash(hash, u64::from(valid));
    for bits in plan_snapshot.decision_bits {
        mix_parity_hash(hash, bits);
    }
    mix_parity_hash(hash, plan_snapshot.cost_bits);
    mix_parity_hash(hash, plan_snapshot.total_dv_bits);
    mix_parity_hash(hash, plan_snapshot.total_time_bits);
    mix_parity_hash(hash, plan_snapshot.time_per_relative_velocity_bits);
    let branch_rev = u64::try_from(plan_snapshot.branch_rev)
        .map_err(|_| anyhow::anyhow!("parity snapshot branch revolution is negative"))?;
    let best_m = u64::try_from(plan_snapshot.best_m)
        .map_err(|_| anyhow::anyhow!("parity snapshot Lambert branch is negative"))?;
    mix_parity_hash(hash, branch_rev);
    mix_parity_hash(hash, u64::from(branch_low_path));
    mix_parity_hash(hash, best_m);
    mix_parity_hash(hash, plan_snapshot.optimizer_func_evals);
    mix_parity_hash(hash, u64::from(optimizer_converged));
    mix_parity_hash(hash, u64::from(warm_start_used));
    plans.push(plan_snapshot);
    Ok(())
}

fn oxymoo_population_parity_rows(
    result: &Nsga2Result,
) -> anyhow::Result<(Vec<OxymooPopulationRowParity>, u64)> {
    let mut rows = Vec::with_capacity(result.population.len());
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for row in 0..result.population.len() {
        let decision: &[f64; 3] = result.population.decision(row)?.try_into().map_err(|_| {
            anyhow::anyhow!("population decision row {row} does not have three values")
        })?;
        let objectives: &[f64; 2] =
            result.population.objectives(row)?.try_into().map_err(|_| {
                anyhow::anyhow!("population objective row {row} does not have two values")
            })?;
        let constraint_violation = *result
            .population
            .constraint_violations
            .get(row)
            .ok_or_else(|| {
                anyhow::anyhow!("population constraint-violation row {row} is missing")
            })?;
        let rank = *result
            .population
            .ranks
            .get(row)
            .ok_or_else(|| anyhow::anyhow!("population rank row {row} is missing"))?;
        let crowding = *result
            .population
            .crowding
            .get(row)
            .ok_or_else(|| anyhow::anyhow!("population crowding row {row} is missing"))?;
        let &[time2phase, phase_sma, waittime] = decision;
        let &[total_dv, time_per_relative_velocity] = objectives;
        let row_snapshot = OxymooPopulationRowParity {
            decision_bits: [
                time2phase.to_bits(),
                phase_sma.to_bits(),
                waittime.to_bits(),
            ],
            objective_bits: [total_dv.to_bits(), time_per_relative_velocity.to_bits()],
            constraint_violation_bits: constraint_violation.to_bits(),
            rank,
            crowding_bits: crowding.to_bits(),
        };
        for bits in row_snapshot.decision_bits {
            mix_parity_hash(&mut hash, bits);
        }
        for bits in row_snapshot.objective_bits {
            mix_parity_hash(&mut hash, bits);
        }
        mix_parity_hash(&mut hash, row_snapshot.constraint_violation_bits);
        mix_parity_hash(
            &mut hash,
            u64::try_from(row_snapshot.rank)
                .map_err(|_| anyhow::anyhow!("population rank does not fit parity hash"))?,
        );
        mix_parity_hash(&mut hash, row_snapshot.crowding_bits);
        rows.push(row_snapshot);
    }
    for front in &result.fronts {
        mix_parity_hash(
            &mut hash,
            u64::try_from(front.len())
                .map_err(|_| anyhow::anyhow!("population front length does not fit parity hash"))?,
        );
        for &row in front {
            mix_parity_hash(
                &mut hash,
                u64::try_from(row)
                    .map_err(|_| anyhow::anyhow!("population row does not fit parity hash"))?,
            );
        }
    }
    Ok((rows, hash))
}

fn materialize_oxymoo_population_for_parity(
    result: &Nsga2Result,
    ctx: &PlanContext,
    warm_start_consumed: bool,
) -> anyhow::Result<(Vec<OxymooMaterializedPlanParity>, u64)> {
    let fallback_cache = RefCell::new(SolveLocalWorkCache::new());
    let mut seen_decisions = FxHashSet::default();
    let mut plans = Vec::new();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let evaluations = u64::try_from(result.evaluations)
        .map_err(|_| anyhow::anyhow!("population evaluation count does not fit parity row"))?;
    for row in 0..result.population.len() {
        let constraint_violation = *result
            .population
            .constraint_violations
            .get(row)
            .ok_or_else(|| {
                anyhow::anyhow!("population constraint-violation row {row} is missing")
            })?;
        if constraint_violation > 0.0 {
            continue;
        }
        let x = repaired_population_transfer_decision(&result.population, row)?;
        let key = transfer_decision_key(&x);
        if !seen_decisions.insert(key) {
            continue;
        }
        let mut plan = evaluate_plan_local(&x, ctx, false, &fallback_cache)?;
        if !transfer_candidate_is_objective_finite(&plan) {
            continue;
        }
        plan.func_evals = evaluations;
        plan.optimizer_func_evals = evaluations;
        plan.optimizer_converged = result.generations > 0;
        plan.warm_start_used = warm_start_consumed;
        push_oxymoo_materialized_plan_parity(&mut plans, &mut hash, &plan)?;
    }
    Ok((plans, hash))
}

fn materialize_oxymoo_population_from_sources_for_parity(
    result: &Nsga2Result,
    plan_cache: &TransferMooPlanCache,
    warm_start_consumed: bool,
) -> anyhow::Result<(Vec<OxymooMaterializedPlanParity>, u64, usize)> {
    let mut seen_decisions = FxHashSet::default();
    let mut plans = Vec::new();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut missing_count = 0usize;
    let evaluations = u64::try_from(result.evaluations)
        .map_err(|_| anyhow::anyhow!("population evaluation count does not fit parity row"))?;
    for row in 0..result.population.len() {
        let constraint_violation = *result
            .population
            .constraint_violations
            .get(row)
            .ok_or_else(|| {
                anyhow::anyhow!("population constraint-violation row {row} is missing")
            })?;
        if constraint_violation > 0.0 {
            continue;
        }
        let x = repaired_population_transfer_decision(&result.population, row)?;
        let key = transfer_decision_key(&x);
        if !seen_decisions.insert(key) {
            continue;
        }
        let cached = {
            let guard = plan_cache.borrow();
            guard.get(&key).cloned()
        };
        let Some(cached) = cached else {
            missing_count = checked_usize_add(missing_count, 1)?;
            continue;
        };
        if !transfer_candidate_is_objective_finite(&cached) {
            continue;
        }
        let mut plan = cached;
        plan.func_evals = evaluations;
        plan.optimizer_func_evals = evaluations;
        plan.optimizer_converged = result.generations > 0;
        plan.warm_start_used = warm_start_consumed;
        push_oxymoo_materialized_plan_parity(&mut plans, &mut hash, &plan)?;
    }
    Ok((plans, hash, missing_count))
}

fn materialize_oxymoo_population_from_problem_sources_for_parity(
    result: &Nsga2Result,
    problem: &TransferMooProblem,
    warm_start_consumed: bool,
) -> anyhow::Result<(Vec<OxymooMaterializedPlanParity>, u64, usize)> {
    materialize_oxymoo_population_from_sources_for_parity(
        result,
        &problem.plan_cache,
        warm_start_consumed,
    )
}

fn oxymoo_materialization_parity_snapshot() -> anyhow::Result<OxymooMaterializationParitySnapshot> {
    let mut ctx = make_leo_ctx()?;
    prepare_single_pair_context(&mut ctx);
    let workspace = TransferMooWorkspace::new();
    let (ranked_seeds, warm_start_consumed, _seed_timing) =
        rank_seed_candidates_for_front(&ctx, None, &workspace.seed_cache)?;
    let (population_size, generations) = transfer_moo_population_generations();
    let generation_count = checked_usize_add(generations, 1)?;
    let cache_capacity = checked_usize_mul(population_size, generation_count)?;
    let mut map = FxHashMap::default();
    map.try_reserve(cache_capacity)
        .map_err(|error| anyhow::anyhow!("parity plan-cache reservation failed: {error}"))?;
    let plan_cache: TransferMooPlanCache = RefCell::new(map);
    let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let problem = TransferMooProblem::new(ctx.clone(), plan_cache, policy)?;
    for (seed, plan) in &ranked_seeds {
        if transfer_candidate_is_objective_finite(plan) {
            let x = repaired_transfer_decision(&seed.x);
            problem.preload_plan(transfer_decision_key(&x), &x, plan.clone())?;
        }
    }
    let mut initial_decisions = Vec::new();
    let mut seen_decisions = Vec::new();
    fill_transfer_moo_initial_decisions(
        Some([0.0, 1.0, 0.0]),
        &ranked_seeds,
        &mut initial_decisions,
        &mut seen_decisions,
    )?;
    let config = transfer_moo_config_with_initial_decisions(&ctx, initial_decisions);
    let optimizer = Nsga2::new(problem, config?)?;
    let (problem, result) = optimizer.run_owned_with_problem()?;
    let (population_rows, population_hash) = oxymoo_population_parity_rows(&result)?;
    let (materialized_plans, materialized_hash) =
        materialize_oxymoo_population_for_parity(&result, &ctx, warm_start_consumed)?;
    let (source_materialized_plans, source_materialized_hash, source_missing_count) =
        materialize_oxymoo_population_from_sources_for_parity(
            &result,
            &problem.plan_cache,
            warm_start_consumed,
        )?;

    Ok(OxymooMaterializationParitySnapshot {
        population_rows,
        fronts: result.fronts.clone(),
        materialized_plans,
        source_materialized_plans,
        population_hash,
        materialized_hash,
        source_materialized_hash,
        source_missing_count,
        nsga_evaluations: result.evaluations,
        nsga_generations: result.generations,
    })
}

fn oxymoo_returned_problem_materialization_parity_snapshot(
) -> anyhow::Result<OxymooMaterializationParitySnapshot> {
    let mut ctx = make_leo_ctx()?;
    prepare_single_pair_context(&mut ctx);
    let workspace = TransferMooWorkspace::new();
    let (ranked_seeds, warm_start_consumed, _seed_timing) =
        rank_seed_candidates_for_front(&ctx, None, &workspace.seed_cache)?;
    let (population_size, generations) = transfer_moo_population_generations();
    let generation_count = checked_usize_add(generations, 1)?;
    let cache_capacity = checked_usize_mul(population_size, generation_count)?;
    let mut map = FxHashMap::default();
    map.try_reserve(cache_capacity)
        .map_err(|error| anyhow::anyhow!("parity plan-cache reservation failed: {error}"))?;
    let plan_cache: TransferMooPlanCache = RefCell::new(map);
    let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let problem = TransferMooProblem::new(ctx.clone(), plan_cache, policy)?;
    for (seed, plan) in &ranked_seeds {
        if transfer_candidate_is_objective_finite(plan) {
            let x = repaired_transfer_decision(&seed.x);
            problem.preload_plan(transfer_decision_key(&x), &x, plan.clone())?;
        }
    }
    let mut initial_decisions = Vec::new();
    let mut seen_decisions = Vec::new();
    fill_transfer_moo_initial_decisions(
        Some([0.0, 1.0, 0.0]),
        &ranked_seeds,
        &mut initial_decisions,
        &mut seen_decisions,
    )?;
    let config = transfer_moo_config_with_initial_decisions(&ctx, initial_decisions);
    let optimizer = Nsga2::new(problem, config?)?;
    let (problem, result) = optimizer.run_owned_with_problem()?;
    let (population_rows, population_hash) = oxymoo_population_parity_rows(&result)?;
    let (materialized_plans, materialized_hash) =
        materialize_oxymoo_population_for_parity(&result, &ctx, warm_start_consumed)?;
    let (source_materialized_plans, source_materialized_hash, source_missing_count) =
        materialize_oxymoo_population_from_problem_sources_for_parity(
            &result,
            &problem,
            warm_start_consumed,
        )?;

    Ok(OxymooMaterializationParitySnapshot {
        population_rows,
        fronts: result.fronts.clone(),
        materialized_plans,
        source_materialized_plans,
        population_hash,
        materialized_hash,
        source_materialized_hash,
        source_missing_count,
        nsga_evaluations: result.evaluations,
        nsga_generations: result.generations,
    })
}

fn assert_oxymoo_materialization_parity_harness_covers_rows_fronts_ranks_and_crowding(
) -> anyhow::Result<()> {
    let first = oxymoo_materialization_parity_snapshot()?;
    let second = oxymoo_materialization_parity_snapshot()?;
    anyhow::ensure!(
        second == first,
        "OxyMOO materialization parity snapshot drifted across identical seeded runs"
    );
    anyhow::ensure!(
        !first.population_rows.is_empty(),
        "harness must cover at least one selected OxyMOO population row"
    );
    anyhow::ensure!(
        !first.fronts.is_empty() && first.fronts.iter().all(|front| !front.is_empty()),
        "harness must cover OxyMOO front membership"
    );
    anyhow::ensure!(
        first
            .population_rows
            .iter()
            .all(|row| row.rank != usize::MAX),
        "harness must cover assigned OxyMOO ranks"
    );
    anyhow::ensure!(
        first
            .population_rows
            .iter()
            .any(|row| row.crowding_bits != 0),
        "harness must cover non-zero or infinite crowding metadata"
    );
    anyhow::ensure!(
        !first.materialized_plans.is_empty(),
        "harness must cover materialized verified-superset plans"
    );
    anyhow::ensure!(
        first.nsga_evaluations > 0 && first.nsga_generations > 0,
        "harness must cover a completed NSGA-II run"
    );
    Ok(())
}

#[test]
fn oxymoo_materialization_parity_harness_covers_rows_fronts_ranks_and_crowding(
) -> anyhow::Result<()> {
    assert_oxymoo_materialization_parity_harness_covers_rows_fronts_ranks_and_crowding()
}

#[test]
fn oxymoo_source_materialization_matches_recompute_parity() -> anyhow::Result<()> {
    let snapshot = oxymoo_materialization_parity_snapshot()?;
    anyhow::ensure!(
        snapshot.source_missing_count == 0,
        "source-carrying materialization must cover every required selected row"
    );
    anyhow::ensure!(
        snapshot.source_materialized_plans == snapshot.materialized_plans,
        "source-carried plans must match recompute materialization rows exactly"
    );
    anyhow::ensure!(
        snapshot.source_materialized_hash == snapshot.materialized_hash,
        "source-carried materialization hash must match recompute hash"
    );
    Ok(())
}

#[test]
fn oxymoo_returned_problem_sources_match_recompute_parity() -> anyhow::Result<()> {
    let snapshot = oxymoo_returned_problem_materialization_parity_snapshot()?;
    anyhow::ensure!(
        snapshot.source_missing_count == 0,
        "returned problem source materialization must cover every required selected row"
    );
    anyhow::ensure!(
        snapshot.source_materialized_plans == snapshot.materialized_plans,
        "returned problem source plans must match recompute materialization rows exactly"
    );
    anyhow::ensure!(
        snapshot.source_materialized_hash == snapshot.materialized_hash,
        "returned problem source materialization hash must match recompute hash"
    );
    Ok(())
}

#[test]
fn oxymoo_verified_superset_materializes_from_sources_without_recompute() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;

    let front = solve_plan_oxymoo_front_internal(
        &mut ctx,
        None,
        None,
        FrontOutputMode::VerifiedSuperset,
        None,
        DeltaVAnchorPolicy::Full,
        TransferMooPolicy::Full,
    )?;

    let metrics = front.verified_superset_metrics;
    anyhow::ensure!(!front.is_empty(), "expected a verified OxyMOO front");
    anyhow::ensure!(
        metrics.nsga_materialize_plan_cache_hit_count > 0,
        "source materialization must cover at least one selected row"
    );
    anyhow::ensure!(
        metrics.nsga_materialize_plan_cache_miss_count == 0,
        "source materialization must not miss selected rows"
    );
    anyhow::ensure!(
        metrics.nsga_materialize_recompute_count == 0,
        "source materialization must avoid recomputing selected rows"
    );
    anyhow::ensure!(
        metrics.nsga_materialize_all_exact_count == 1,
        "source materialization should record an all-source exact pass"
    );
    Ok(())
}

#[test]
fn oxymoo_source_materialization_control_is_verified_superset_only() {
    assert!(oxymoo_source_materialization_enabled(
        FrontOutputMode::VerifiedSuperset
    ));
    assert!(!oxymoo_source_materialization_enabled(
        FrontOutputMode::TransferPareto
    ));
}

#[test]
fn fixed_lbfgs_without_a_gradient_fails_without_seed_fallback() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.local_optimizer = TransferLocalOptimizerConfig {
        choice: TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::Lbfgs),
        tune: TuneLevel::Default,
        seed: 11,
    };

    let result = solve_plan(&mut ctx, None);

    anyhow::ensure!(
        matches!(
            result,
            Err(InvalidTargetPropagationAuthorityCode::OptimizerFailure)
        ),
        "fixed L-BFGS without a gradient must return OptimizerFailure"
    );
    Ok(())
}

#[test]
fn pso_local_optimizer_converges_on_a_sphere() -> anyhow::Result<()> {
    // Smoke test: >4 evaluations and cost < 0.1 also hold for a Nelder-Mead
    // run on this problem, so this cannot distinguish the PSO path from the
    // simplex path.
    struct Sphere;
    impl LocalScalarProblem3 for Sphere {
        fn value(&self, x: &[f64; 3]) -> anyhow::Result<f64> {
            Ok(x.iter().map(|value| value * value).sum())
        }
    }

    let result = run_local_optimizer(
        &Sphere,
        [-1.0; 3],
        [1.0; 3],
        [0.9; 3],
        local_config(LocalOptimizerKind::Pso, 16, TuneLevel::Default, 3),
    )?;

    anyhow::ensure!(result.evaluations > 4, "{result:?}");
    anyhow::ensure!(result.cost < 0.1, "{result:?}");
    Ok(())
}

#[test]
fn test_solve_plan_basic() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let result = solve_plan_representative(&mut ctx)?;

    // Should find a valid solution for simple LEO-to-LEO. `make_leo_ctx` is
    // coplanar, so the 0.5 km/s bound is the one a coplanar transfer must meet.
    anyhow::ensure!(result.valid, "Should find valid solution");
    anyhow::ensure!(result.cost < INVALID_COST, "solution cost must be finite");
    anyhow::ensure!(
        result.total_dv() < 0.5,
        "Coplanar 100km altitude change should be < 0.5 km/s"
    );
    Ok(())
}

#[test]
fn test_solve_plan_returns_verified_front() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;

    let front = solve_plan(&mut ctx, None)?;

    anyhow::ensure!(
        !front.is_empty(),
        "expected at least one transfer candidate"
    );
    for candidate in &front.candidates {
        anyhow::ensure!(candidate.valid);
        anyhow::ensure!(candidate.cost < INVALID_COST);
        anyhow::ensure!(candidate.transfer_objectives().is_finite());
    }
    assert_transfer_front_posthoc_verified(&ctx, &front);
    for (idx, candidate) in front.candidates.iter().enumerate() {
        for (other_idx, other) in front.candidates.iter().enumerate() {
            if idx != other_idx {
                anyhow::ensure!(
                    !transfer_candidate_dominates(other, candidate),
                    "front candidate should not be dominated"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn solve_plan_rejects_high_fidelity_candidate_search_before_propagation() -> anyhow::Result<()> {
    let mut request =
        crate::types::TransferRequest::with_j2_closure_settings(J2ClosureSettings::default());
    request.target_propagation_authority = TargetPropagationAuthority::HighFidelity;
    request.target_body_force = BodyForceConfig::gravity_only(BodyRole::DiagnosticTarget);
    request.force_config = Some(std::sync::Arc::new(
        lightyear_odeint_rs::types::ForceConfig {
            sph_order: 5,
            force_flags: lightyear_odeint_rs::types::ForceFlags::DRAG
                | lightyear_odeint_rs::types::ForceFlags::SRP
                | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
                | lightyear_odeint_rs::types::ForceFlags::MOON_GRAVITY,
            atm_model: crate::types::HIGH_FIDELITY_ATM_MODEL,
            target_propagation_mode: TargetPropagationAuthority::HighFidelity
                .as_force_config_code(),
            sun_pos: Some([149_600_000.0, 0.0, 0.0]),
            moon_pos: Some([384_400.0, 0.0, 0.0]),
            ..Default::default()
        },
    ));
    let mut ctx = PlanContext::from_request(request);
    let original_target_sma = ctx.tgt_sma;
    let original_target_cache_valid = ctx.tgt_orbit_valid;

    let Err(error) = solve_plan(&mut ctx, None) else {
        anyhow::bail!("high-fidelity candidate search must fail before solve");
    };

    anyhow::ensure!(
        error
            == crate::types::InvalidTargetPropagationAuthorityCode::CandidateSearch(
                crate::types::CandidateSearchAuthorityError::UnsupportedHighFidelityTransferSearch,
            ),
        "high-fidelity candidate-search rejection mismatch: {error:?}"
    );
    anyhow::ensure!(ctx.tgt_sma.to_bits() == original_target_sma.to_bits());
    anyhow::ensure!(ctx.tgt_orbit_valid == original_target_cache_valid);
    Ok(())
}

#[test]
fn solve_plan_rejects_require_hf_mf_context_before_rhs_work() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    anyhow::ensure!(ctx.target_propagation_authority == TargetPropagationAuthority::MfJ2);
    anyhow::ensure!(ctx.force_config.is_none());
    ctx.execution_policy.require_high_fidelity = true;

    let before = evaluation_diagnostic_snapshot();
    let result = solve_plan(&mut ctx, None);
    let delta = evaluation_diagnostic_snapshot().delta_since(before)?;

    anyhow::ensure!(matches!(
        result,
        Err(
            crate::types::InvalidTargetPropagationAuthorityCode::CandidateSearch(
                crate::types::CandidateSearchAuthorityError::UnsupportedHighFidelityTransferSearch,
            )
        )
    ));
    anyhow::ensure!(delta.j2_propagate_state_count == 0);
    anyhow::ensure!(delta.target_j2_batch_state_count == 0);
    anyhow::ensure!(delta.target_j2_scalar_state_count == 0);
    anyhow::ensure!(delta.branch_shared_prepare_count == 0);
    anyhow::ensure!(delta.lambert_batch_call_count == 0);
    Ok(())
}

#[test]
fn test_constellation_multi_objective_returns_design_vector_front() -> anyhow::Result<()> {
    let satellites = vec![
        make_circular_state(400.0, 0.0, 0.0, 0.0),
        make_circular_state(520.0, 0.02, 0.0, 0.4),
    ];
    let target1 = make_circular_state(500.0, 0.0, 0.0, 0.2);
    let target2 = make_circular_state(620.0, 0.02, 0.0, 0.8);

    let configuration = test_mf_configuration();
    let front = constellation_solve_native_with_front_output_mode(EventPlanRequest {
        satellites: &satellites,
        satellites_equ_cached: None,
        target1: &target1,
        target2: &target2,
        target_body_forces: [BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2],
        configuration: configuration.clone(),
        scratch: None,
        front_output_mode: FrontOutputMode::TransferPareto,
    })?;

    anyhow::ensure!(!front.is_empty(), "expected constellation transfer front");
    assert_constellation_front_posthoc_verified(
        &satellites,
        target1,
        target2,
        &configuration,
        &front,
    )?;
    for candidate in &front.candidates {
        anyhow::ensure!(candidate.valid);
        anyhow::ensure!(candidate.optimum.valid);
        anyhow::ensure!(
            matches!(
                usize::try_from(candidate.sat_index),
                Ok(sat_index) if sat_index < satellites.len()
            ),
            "constellation candidate must reference a fixture satellite"
        );
        anyhow::ensure!(candidate.target_index == 0 || candidate.target_index == 1);
        anyhow::ensure!(candidate.objectives.is_finite());
    }
    for (idx, candidate) in front.candidates.iter().enumerate() {
        for (other_idx, other) in front.candidates.iter().enumerate() {
            if idx != other_idx {
                anyhow::ensure!(
                    !constellation_candidate_dominates(other, candidate),
                    "global front candidate should not be dominated"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn prepare_event_scratch_none_matches_some() -> anyhow::Result<()> {
    let satellites = vec![
        make_circular_state(400.0, 0.0, 0.0, 0.0),
        make_circular_state(520.0, 0.02, 0.0, 0.4),
        make_circular_state(610.0, 0.03, 0.0, 0.9),
        make_circular_state(455.0, 0.01, 0.0, 1.7),
    ];
    let target1 = make_circular_state(500.0, 0.0, 0.0, 0.2);
    let target2 = make_circular_state(620.0, 0.02, 0.0, 0.8);
    let n_sats = satellites.len();

    let build_plan = |scratch: Option<&mut crate::scratch::SolveScratch>| {
        prepare_event(EventPlanRequest {
            satellites: &satellites,
            satellites_equ_cached: None,
            target1: &target1,
            target2: &target2,
            target_body_forces: [BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2],
            configuration: test_mf_configuration(),
            scratch,
            front_output_mode: FrontOutputMode::TransferPareto,
        })
    };

    let Some(plan_none) = build_plan(None)? else {
        anyhow::bail!("fixture must yield EventPlan without scratch");
    };
    let mut fresh_scratch = crate::scratch::SolveScratch::new(n_sats)?;
    let Some(plan_some) = build_plan(Some(&mut fresh_scratch))? else {
        anyhow::bail!("fixture must yield EventPlan with scratch");
    };

    anyhow::ensure!(
        plan_none.selected_pair_count() == plan_some.selected_pair_count(),
        "scratch=None vs Some produced a different selected-pair count"
    );
    anyhow::ensure!(
        plan_none.selected_pair_count() >= 1,
        "fixture must select >= 1 pair to exercise the comparison"
    );
    anyhow::ensure!(
        plan_none
            .selected_pair(plan_none.selected_pair_count())
            .is_none(),
        "an invalid selected-pair slot must be absent"
    );

    for slot in 0..plan_none.selected_pair_count() {
        let a = plan_none.selected_pair(slot);
        let b = plan_some.selected_pair(slot);
        anyhow::ensure!(
            a.is_some() && b.is_some(),
            "selected pair {slot}: validated slot unexpectedly absent"
        );
        let (Some(a), Some(b)) = (a, b) else {
            anyhow::bail!("selected pair {slot}: validated slot unexpectedly absent");
        };
        anyhow::ensure!(
            (
                a.sat_idx,
                a.tgt_idx,
                a.dv_proxy.to_bits(),
                a.time_proxy_s.to_bits()
            ) == (
                b.sat_idx,
                b.tgt_idx,
                b.dv_proxy.to_bits(),
                b.time_proxy_s.to_bits()
            ),
            "selected pair {slot}: scratch=None vs Some differ bit-for-bit"
        );
    }

    // Borrowed satellite state and derived equinoctial state must match
    // exactly; scratch must not perturb either result.
    let sat_bits = |sats: &[[f64; 6]]| -> Vec<u64> {
        sats.iter()
            .flat_map(|s| s.iter().map(|v| v.to_bits()))
            .collect()
    };
    anyhow::ensure!(
        sat_bits(plan_none.satellites) == sat_bits(plan_some.satellites),
        "EventPlan satellite ECI state differs between scratch=None and Some"
    );
    anyhow::ensure!(
        sat_bits(plan_none.satellites_equ.as_ref()) == sat_bits(plan_some.satellites_equ.as_ref()),
        "EventPlan satellite equinoctial state differs between scratch=None and Some"
    );
    Ok(())
}

#[test]
fn internal_batch_dispatch_closes_mismatched_authority_without_panic() -> anyhow::Result<()> {
    let satellites = vec![make_circular_state(400.0, 0.0, 0.0, 0.0)];
    let target1 = make_circular_state(500.0, 0.0, 0.0, 0.2);
    let target2 = make_circular_state(620.0, 0.02, 0.0, 0.8);
    let force_config = lightyear_odeint_rs::types::ForceConfig {
        target_propagation_mode: TargetPropagationAuthority::MfJ2.as_force_config_code(),
        ..Default::default()
    };

    let mut configuration = test_mf_configuration();
    configuration.pairs_to_verify = 1;
    configuration.target_propagation_authority = TargetPropagationAuthority::AnalyticalKepler;
    configuration.force_config = Some(std::sync::Arc::new(force_config));
    let Err(error) = constellation_solve_native_with_front_output_mode(EventPlanRequest {
        satellites: &satellites,
        satellites_equ_cached: None,
        target1: &target1,
        target2: &target2,
        target_body_forces: [BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2],
        configuration,
        scratch: None,
        front_output_mode: FrontOutputMode::TransferPareto,
    }) else {
        anyhow::bail!("mismatched authority must not become an empty or partial front");
    };

    anyhow::ensure!(
        error
            == crate::types::InvalidTargetPropagationAuthorityCode::CandidateSearch(
                crate::types::CandidateSearchAuthorityError::UnsupportedHighFidelityTransferSearch,
            ),
        "mismatched-authority rejection mismatch: {error:?}"
    );
    Ok(())
}

#[test]
fn internal_batch_dispatch_rejects_require_hf() -> anyhow::Result<()> {
    // Asserts the typed error only. It does NOT assert the rejection lands
    // before pair screening; no work counter observes that ordering here.
    let satellites = vec![make_circular_state(400.0, 0.0, 0.0, 0.0)];
    let target1 = make_circular_state(500.0, 0.0, 0.0, 0.2);
    let target2 = make_circular_state(620.0, 0.02, 0.0, 0.8);

    let mut configuration = test_mf_configuration();
    configuration.pairs_to_verify = 1;
    configuration.require_high_fidelity = true;
    let Err(error) = constellation_solve_native_with_front_output_mode(EventPlanRequest {
        satellites: &satellites,
        satellites_equ_cached: None,
        target1: &target1,
        target2: &target2,
        target_body_forces: [BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2],
        configuration,
        scratch: None,
        front_output_mode: FrontOutputMode::VerifiedSuperset,
    }) else {
        anyhow::bail!("require-high-fidelity must not become an empty or partial front");
    };

    anyhow::ensure!(
        error
            == crate::types::InvalidTargetPropagationAuthorityCode::CandidateSearch(
                crate::types::CandidateSearchAuthorityError::UnsupportedHighFidelityTransferSearch,
            ),
        "require-high-fidelity rejection mismatch: {error:?}"
    );
    Ok(())
}

#[test]
fn prepare_event_rejects_require_hf() -> anyhow::Result<()> {
    // Asserts the typed error only; ordering vs pair screening is unobserved.
    let satellites = vec![make_circular_state(400.0, 0.0, 0.0, 0.0)];
    let target1 = make_circular_state(500.0, 0.0, 0.0, 0.2);
    let target2 = make_circular_state(620.0, 0.02, 0.0, 0.8);

    let mut configuration = test_mf_configuration();
    configuration.pairs_to_verify = 1;
    configuration.require_high_fidelity = true;
    let Err(error) = prepare_event(EventPlanRequest {
        satellites: &satellites,
        satellites_equ_cached: None,
        target1: &target1,
        target2: &target2,
        target_body_forces: [BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2],
        configuration,
        scratch: None,
        front_output_mode: FrontOutputMode::TransferPareto,
    }) else {
        anyhow::bail!("require-high-fidelity must not become an empty EventPlan");
    };

    anyhow::ensure!(
        error
            == crate::types::InvalidTargetPropagationAuthorityCode::CandidateSearch(
                crate::types::CandidateSearchAuthorityError::UnsupportedHighFidelityTransferSearch,
            ),
        "prepare-event candidate-search rejection mismatch: {error:?}"
    );
    Ok(())
}

#[test]
fn prepare_event_rejects_mf_gravity_only_target() -> anyhow::Result<()> {
    // Asserts the typed error only; ordering vs the empty fallback is
    // unobserved.
    let target = make_circular_state(500.0, 0.0, 0.0, 0.2);

    let Err(error) = prepare_event(EventPlanRequest {
        satellites: &[],
        satellites_equ_cached: None,
        target1: &target,
        target2: &target,
        target_body_forces: [
            BodyForceConfig::gravity_only(BodyRole::DiagnosticTarget),
            BodyForceConfig::j2(BodyRole::DiagnosticTarget),
        ],
        configuration: test_mf_configuration(),
        scratch: None,
        front_output_mode: FrontOutputMode::TransferPareto,
    }) else {
        anyhow::bail!("invalid MF target force must not become an empty EventPlan");
    };

    anyhow::ensure!(
        error
            == crate::types::InvalidTargetPropagationAuthorityCode::InvalidTargetBodyForce {
                authority: TargetPropagationAuthority::MfJ2,
            },
        "prepare-event MF target-body-force rejection mismatch: {error:?}"
    );
    Ok(())
}

#[test]
fn internal_batch_dispatch_keeps_valid_empty_input_empty() -> anyhow::Result<()> {
    let target = make_circular_state(500.0, 0.0, 0.0, 0.2);

    let mut configuration = test_mf_configuration();
    configuration.pairs_to_verify = 1;
    let front = constellation_solve_native_with_front_output_mode(EventPlanRequest {
        satellites: &[],
        satellites_equ_cached: None,
        target1: &target,
        target2: &target,
        target_body_forces: [BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2],
        configuration,
        scratch: None,
        front_output_mode: FrontOutputMode::TransferPareto,
    })?;

    anyhow::ensure!(front.is_empty(), "valid empty input must remain empty");
    Ok(())
}

#[test]
fn prepare_event_can_borrow_stable_satellite_state_and_remains_send_sync() -> anyhow::Result<()> {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<EventPlan<'static>>();

    let satellites = vec![
        make_circular_state(400.0, 0.0, 0.0, 0.0),
        make_circular_state(520.0, 0.02, 0.0, 0.4),
        make_circular_state(610.0, 0.03, 0.0, 0.9),
        make_circular_state(455.0, 0.01, 0.0, 1.7),
    ];
    let satellites_equ = satellites
        .iter()
        .map(|sat| {
            let mut equ = [0.0_f64; 6];
            satpy_core::eci2equinoc_impl(sat, 6, 0.0, 0.0, &mut equ);
            equ
        })
        .collect::<Vec<_>>();
    let target1 = make_circular_state(500.0, 0.0, 0.0, 0.2);
    let target2 = make_circular_state(620.0, 0.02, 0.0, 0.8);

    let Some(plan) = prepare_event(EventPlanRequest {
        satellites: &satellites,
        satellites_equ_cached: Some(&satellites_equ),
        target1: &target1,
        target2: &target2,
        target_body_forces: [BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2],
        configuration: test_mf_configuration(),
        scratch: None,
        front_output_mode: FrontOutputMode::TransferPareto,
    })?
    else {
        anyhow::bail!("fixture must yield an EventPlan");
    };

    anyhow::ensure!(plan.uses_borrowed_satellite_equ_state());
    Ok(())
}

#[test]
fn walker_constellation_returns_multiple_pairs_and_transfers() -> anyhow::Result<()> {
    let planes = 3_u32;
    let sats_per_plane = 3_u32;
    let total_sats = planes
        .checked_mul(sats_per_plane)
        .ok_or_else(|| anyhow::anyhow!("Walker fixture satellite count overflow"))?;
    let satellite_capacity = usize::try_from(total_sats)
        .map_err(|error| anyhow::anyhow!("Walker fixture capacity conversion failed: {error}"))?;
    let inclination = 53.0_f64.to_radians();
    let phasing = 1.0;
    let mut satellites = Vec::with_capacity(satellite_capacity);
    for plane in 0..planes {
        let raan = std::f64::consts::TAU * f64::from(plane) / f64::from(planes);
        for slot in 0..sats_per_plane {
            let slot_phase = f64::from(slot) / f64::from(sats_per_plane);
            let walker_phase = phasing * f64::from(plane) / f64::from(total_sats);
            let arg_lat = std::f64::consts::TAU * (slot_phase + walker_phase);
            satellites.push(make_circular_state(550.0, inclination, raan, arg_lat));
        }
    }

    let target1 = make_circular_state(565.0, inclination, 0.18, 0.73);
    let target2 = make_circular_state(610.0, 55.0_f64.to_radians(), 1.42, 2.15);

    let mut configuration = test_mf_configuration();
    configuration.max_time_s = 172_800.0;
    configuration.max_phase_dv = 1.2;
    configuration.max_transfer_dv = 2.4;
    configuration.max_revs = 2;
    configuration.pairs_to_verify = 0;
    let front = constellation_solve_native_with_front_output_mode(EventPlanRequest {
        satellites: &satellites,
        satellites_equ_cached: None,
        target1: &target1,
        target2: &target2,
        target_body_forces: [BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2],
        configuration: configuration.clone(),
        scratch: None,
        front_output_mode: FrontOutputMode::TransferPareto,
    })?;

    let mut by_pair = std::collections::BTreeMap::<(i32, i32), Vec<()>>::new();
    assert_constellation_front_posthoc_verified(
        &satellites,
        target1,
        target2,
        &configuration,
        &front,
    )?;
    for candidate in &front.candidates {
        anyhow::ensure!(candidate.valid, "Walker candidate must be valid");
        anyhow::ensure!(candidate.optimum.valid, "Walker optimum must be valid");
        anyhow::ensure!(
            candidate.objectives.is_finite(),
            "Walker candidate objectives must be finite"
        );
        by_pair
            .entry((candidate.sat_index, candidate.target_index))
            .or_default()
            .push(());
    }
    let candidate_count = front.len();
    let unique_pair_count = by_pair.len();
    println!(
        "walker_front_candidates={candidate_count} unique_pairs={unique_pair_count} pair_counts={by_pair:?}"
    );

    anyhow::ensure!(
        by_pair.len() >= 2,
        "expected multiple deployer-object pairs in Walker front: {by_pair:?}"
    );
    anyhow::ensure!(
        by_pair.values().any(|candidates| candidates.len() >= 2),
        "expected at least one pair with multiple transfer solutions: {by_pair:?}"
    );
    for (idx, candidate) in front.candidates.iter().enumerate() {
        for (other_idx, other) in front.candidates.iter().enumerate() {
            if idx != other_idx {
                anyhow::ensure!(
                    !constellation_candidate_dominates(other, candidate),
                    "Walker front candidate should not be dominated"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_finalize_verified_candidate_skips_unverified_cheaper_candidate() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");

    let mut cheaper_invalid = accepted.clone();
    cheaper_invalid.cost = accepted.cost * 0.5;
    let [transfer_x, ..] = &mut cheaper_invalid.transfer_dv;
    *transfer_x += 0.5;
    cheaper_invalid.valid = true;

    let mut candidates = vec![cheaper_invalid, accepted.clone()];
    let selected = finalize_verified_candidate(&ctx, &mut candidates)?
        .ok_or_else(|| anyhow::anyhow!("expected verified fallback candidate"))?;

    anyhow::ensure!(selected.valid);
    anyhow::ensure!(selected.cost.to_bits() == accepted.cost.to_bits());
    anyhow::ensure!(selected.time2phase_ratio.to_bits() == accepted.time2phase_ratio.to_bits());
    anyhow::ensure!(selected.phase_sma_ratio.to_bits() == accepted.phase_sma_ratio.to_bits());
    anyhow::ensure!(selected.waittime_ratio.to_bits() == accepted.waittime_ratio.to_bits());
    Ok(())
}

#[test]
fn posthoc_verification_rejects_physical_constraint_violations() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    let baseline = verify_transfer_result(&accepted, &ctx, tolerance);
    anyhow::ensure!(
        baseline.verified,
        "baseline should verify before constraint tightening: {baseline:?}"
    );

    let mut over_time_ctx = ctx.clone();
    over_time_ctx.max_time_s = accepted.total_time() * 0.5;
    let over_time = verify_transfer_result(&accepted, &over_time_ctx, tolerance);
    anyhow::ensure!(
        !over_time.verified,
        "over-time fixture must fail verification"
    );
    anyhow::ensure!(over_time
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Transfer time exceeds budget"));

    let mut close_deployer_ctx = ctx.clone();
    close_deployer_ctx.deployer_min_distance = accepted.deployer_distance + 1.0;
    let close_deployer = verify_transfer_result(&accepted, &close_deployer_ctx, tolerance);
    anyhow::ensure!(
        !close_deployer.verified,
        "close-deployer fixture must fail verification"
    );
    anyhow::ensure!(close_deployer
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Deployer separation below minimum"));
    Ok(())
}

#[test]
#[expect(
    clippy::suboptimal_flops,
    reason = "stale-state fixture preserves non-fused threshold perturbation arithmetic"
)]
fn posthoc_verification_rejects_stale_intercept_states() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    let [target_x, ..] = &mut accepted.target_intercept_state;
    *target_x += tolerance * 10.0;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(
        !verification.verified,
        "stale-intercept fixture must fail verification"
    );
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("target intercept stored state inconsistent"));
    Ok(())
}

#[test]
fn posthoc_verification_rejects_delta_v_magnitude_mismatch() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    let [transfer_x, ..] = &mut accepted.transfer_dv;
    *transfer_x += 1e-4;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(
        !verification.verified,
        "delta-v-norm fixture must fail verification"
    );
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Transfer delta-V norm mismatch"));
    Ok(())
}

#[test]
fn posthoc_verification_rejects_payload_velocity_as_transfer_impulse() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    let [_, _, _, first, second, third] = accepted.payload_intercept_state;
    accepted.transfer_dv = [first, second, third];
    accepted.transfer_dv_norm = norm3(&accepted.transfer_dv);
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(
        !verification.verified,
        "payload-impulse fixture must fail verification"
    );
    anyhow::ensure!(
        verification
            .propagation_error
            .as_deref()
            .unwrap_or_default()
            .contains("Transfer delta-V exceeds limit")
            || verification
                .propagation_error
                .as_deref()
                .unwrap_or_default()
                .contains("payload intercept stored state inconsistent")
            || verification
                .propagation_error
                .as_deref()
                .unwrap_or_default()
                .contains("Transfer delta-V vector mismatch"),
        "expected transfer impulse semantic mismatch, got {verification:?}"
    );
    Ok(())
}

#[test]
fn posthoc_verification_rejects_arrival_delta_v_magnitude_mismatch() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    accepted.arrival_dv_norm += 1e-4;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(
        !verification.verified,
        "arrival-delta-v-norm fixture must fail verification"
    );
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Arrival delta-V norm mismatch"));
    Ok(())
}

#[test]
fn posthoc_verification_rejects_stale_arrival_delta_v_vector() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    let [arrival_x, arrival_y, arrival_z] = accepted.arrival_dv;
    accepted.arrival_dv = [-arrival_x, -arrival_y, -arrival_z];
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(
        !verification.verified,
        "arrival-delta-v-vector fixture must fail verification"
    );
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Arrival delta-V vector mismatch"));
    Ok(())
}

#[test]
fn posthoc_verification_rejects_lambert_revolution_above_limit() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    accepted.best_M = ctx.max_revs + 1;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(
        !verification.verified,
        "over-revolution fixture must fail verification"
    );
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Lambert revolution count exceeds limit"));
    Ok(())
}

#[test]
fn posthoc_verification_clamps_negative_max_revs_like_evaluator() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    ctx.max_revs = -1;
    accepted.best_M = 1;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(
        !verification.verified,
        "negative-max-revs fixture must fail verification"
    );
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Lambert revolution count exceeds limit"));
    Ok(())
}

#[test]
fn posthoc_verification_rejects_transfer_tof_above_revolution_cap() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    ctx.revolution_cap = accepted.tof / accepted.dep_period * 0.5;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(
        !verification.verified,
        "revolution-cap fixture must fail verification"
    );
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Transfer time of flight exceeds revolution cap"));
    Ok(())
}

#[test]
fn posthoc_verification_rejects_total_time_before_delta_v_replay() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    let transfer_start = accepted.time2phase + accepted.waittime;
    ctx.max_time_s = 0.5 * (transfer_start + crate::types::MIN_TOF + accepted.total_time());
    let [phase_x, ..] = &mut accepted.phase_dv;
    *phase_x += 1e-4;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(!verification.verified);
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Transfer time exceeds budget"));
    Ok(())
}

#[test]
fn posthoc_verification_uses_context_period_for_revolution_cap() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    ctx.revolution_cap = accepted.tof / ctx.dep_period * 0.5;
    accepted.dep_period = ctx.dep_period * 100.0;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(!verification.verified);
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Transfer time of flight exceeds revolution cap"));
    Ok(())
}

#[test]
#[expect(
    clippy::suboptimal_flops,
    reason = "stale-distance fixture preserves non-fused threshold perturbation arithmetic"
)]
fn posthoc_verification_rejects_stale_deployer_distance() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    accepted.deployer_distance += tolerance * 10.0;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(!verification.verified);
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Deployer distance mismatch"));
    Ok(())
}

#[test]
#[expect(
    clippy::suboptimal_flops,
    reason = "stale-distance fixture preserves non-fused threshold perturbation arithmetic"
)]
fn posthoc_verification_rejects_stale_intercept_distance() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    accepted.distance += tolerance * 10.0;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(!verification.verified);
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Intercept distance mismatch"));
    Ok(())
}

#[test]
fn posthoc_verification_rejects_transfer_tof_below_minimum() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    accepted.tof = crate::types::MIN_TOF * 0.5;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(!verification.verified);
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Transfer time of flight below solver minimum"));
    Ok(())
}

#[test]
#[expect(
    clippy::suboptimal_flops,
    reason = "phase-headroom fixture preserves non-fused boundary arithmetic"
)]
fn posthoc_verification_rejects_no_transfer_phase_headroom() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    accepted.time2phase = ctx.max_time_s - crate::types::MIN_TOF * 0.5;
    accepted.waittime = 0.0;
    accepted.tof = crate::types::MIN_TOF;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(!verification.verified);
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Transfer phase has no available overlap"));
    Ok(())
}

#[test]
fn posthoc_verification_rejects_overlapping_phase_timestamps() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    let mut accepted = solve_plan_representative(&mut ctx)?;
    anyhow::ensure!(accepted.valid, "expected baseline solution to be valid");
    let tolerance = verification_tolerance_for_solve(&ctx);

    accepted.tof_jd_start = accepted.waittime_jd_start - 10.0 / satpy_core::SEC_PER_DAY;
    let verification = verify_transfer_result(&accepted, &ctx, tolerance);

    anyhow::ensure!(!verification.verified);
    anyhow::ensure!(verification
        .propagation_error
        .as_deref()
        .unwrap_or_default()
        .contains("Phase timeline overlap"));
    Ok(())
}

#[test]
fn test_solve_plan_returns_invalid_when_all_candidates_violate_budgets() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    ctx.max_phase_dv = 1e-6;
    ctx.max_transfer_dv = 1e-6;

    let result = solve_plan_representative(&mut ctx)?;

    anyhow::ensure!(!result.valid);
    anyhow::ensure!(result.cost >= INVALID_COST);
    Ok(())
}

#[test]
fn test_pair_cache_reset_enabled_from_value() {
    assert!(!pair_cache_reset_enabled_from_value(None));
    assert!(!pair_cache_reset_enabled_from_value(Some("outer")));
    assert!(!pair_cache_reset_enabled_from_value(Some("worker")));
    assert!(pair_cache_reset_enabled_from_value(Some("pair")));
    assert!(!pair_cache_reset_enabled_from_value(Some("legacy")));
    assert!(pair_cache_reset_enabled_from_value(Some("1")));
}

#[test]
fn test_target_plane_from_equinoctial_matches_keplerian_plane() {
    for eci in [
        make_circular_state(800.0, 1.0, 0.4, 1.5),
        make_circular_state(800.0, 0.0, 0.0, 1.5),
    ] {
        let mut equ = [0.0_f64; 6];
        satpy_core::eci2equinoc_impl(&eci, 6, 0.0, 0.0, &mut equ);

        let (inc, raan, valid) = target_plane_from_equinoctial(&equ, &eci);

        let mut kep = [0.0_f64; 6];
        satpy_core::eci2kep_impl(&eci, false, true, &mut kep);
        assert!(valid);
        let [_, _, kep_inc, kep_raan, ..] = kep;
        assert!((inc - kep_inc).abs() < 1e-12);
        let raan_diff = (raan - kep_raan + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        assert!(raan_diff.abs() < 1e-12);
    }
}

#[test]
fn test_target_plane_from_equinoctial_canonicalizes_near_equatorial_plane() {
    let eci = make_circular_state(800.0, 0.0, 0.0, 1.5);
    let mut equ = [7000.0, 0.001, 0.002, 0.0, 0.0, 1.5];
    let [_, _, _, equ_h, equ_k, _] = &mut equ;
    *equ_h = satpy_core::TAN_HALF_INCLINATION_FLOOR * 0.25;
    *equ_k = -satpy_core::TAN_HALF_INCLINATION_FLOOR * 0.25;

    let (inc, raan, valid) = target_plane_from_equinoctial(&equ, &eci);

    assert!(valid);
    assert_eq!(inc.to_bits(), 0.0_f64.to_bits());
    assert_eq!(raan.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn test_seed_prefilter_cost_gap_margin() -> anyhow::Result<()> {
    // Verify the cost-gap margin logic: seeds within 0.2 of the 40th seed
    // should be kept, those beyond should be dropped.
    let cutoff = 40_usize;
    let make_plan = |cost: f64| {
        let mut p = PlanResult::invalid();
        p.cost = cost;
        p
    };

    // 50 seeds with costs 0.1, 0.2, ..., 5.0
    let seeds: Vec<(SolverSeed, PlanResult)> = (1..=50)
        .map(|i| {
            let cost = f64::from(i) * 0.1;
            (
                SolverSeed {
                    x: [0.1, 1.0, 0.1],
                    warm_start_used: false,
                },
                make_plan(cost),
            )
        })
        .collect();

    // Cost of 40th seed (index 39) = 4.0
    let last_index = cutoff
        .checked_sub(1)
        .ok_or_else(|| anyhow::anyhow!("seed prefilter cutoff must be nonzero"))?;
    let cost_of_40th = seeds
        .get(last_index)
        .map(|(_, plan)| plan.cost)
        .ok_or_else(|| anyhow::anyhow!("seed prefilter cutoff must have a seed"))?;
    anyhow::ensure!(
        (cost_of_40th - 4.0).abs() < 1e-10,
        "40th seed must cost 4.0"
    );

    let cost_threshold = cost_of_40th + 0.2; // 4.2

    let filtered_count = seeds
        .iter()
        .take_while(|(_, plan)| plan.cost <= cost_threshold)
        .count();

    // Seeds with cost <= 4.2: indices 0..42 (costs 0.1..4.2), so 42 seeds
    anyhow::ensure!(
        filtered_count == 42,
        "cost-gap prefilter must retain 42 seeds, got {filtered_count}"
    );
    Ok(())
}

#[test]
fn test_two_stage_nm_func_evals_accumulated() -> anyhow::Result<()> {
    // Verify that func_evals is properly accumulated from both NM stages.
    let mut ctx = make_leo_ctx()?;
    ctx.local_optimizer = TransferLocalOptimizerConfig {
        choice: TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
        tune: TuneLevel::Default,
        seed: 11,
    };
    let front = solve_plan(&mut ctx, None)?;

    anyhow::ensure!(
        front
            .candidates
            .iter()
            .any(|candidate| candidate.valid && candidate.func_evals > 0),
        "func_evals should accumulate on at least one NM-produced candidate"
    );
    Ok(())
}

#[test]
fn test_seed_prefilter_all_seeds_within_gap() -> anyhow::Result<()> {
    // Edge case: all seeds have similar cost, gap margin keeps all of them
    let make_plan = |cost: f64| {
        let mut p = PlanResult::invalid();
        p.cost = cost;
        p
    };

    // 20 seeds all with cost 1.0 (well under the 40 cutoff)
    let seeds: Vec<(SolverSeed, PlanResult)> = (0..20)
        .map(|_| {
            (
                SolverSeed {
                    x: [0.1, 1.0, 0.1],
                    warm_start_used: false,
                },
                make_plan(1.0),
            )
        })
        .collect();

    // When fewer seeds than cutoff, threshold is infinity — all kept
    let cutoff = seeds.len().min(40);
    let cost_threshold = if cutoff > 0 && cutoff < seeds.len() {
        let last_index = cutoff
            .checked_sub(1)
            .ok_or_else(|| anyhow::anyhow!("seed prefilter cutoff must be nonzero"))?;
        seeds
            .get(last_index)
            .map(|(_, plan)| plan.cost + 0.2)
            .ok_or_else(|| anyhow::anyhow!("seed prefilter cutoff must have a seed"))?
    } else {
        f64::INFINITY
    };

    let filtered_count = seeds
        .iter()
        .take_while(|(_, plan)| plan.cost <= cost_threshold)
        .count();

    anyhow::ensure!(
        filtered_count == 20,
        "all seeds should pass when count <= 40; got {filtered_count}"
    );
    Ok(())
}

#[test]
fn test_should_stop_coarse_stage_requires_strong_best_and_stalled_recents() {
    let recent = VecDeque::from(vec![0.24, 0.23, 0.22, 0.21]);
    assert!(should_stop_coarse_stage(
        &SearchDepthPolicy::default(),
        16,
        0.09,
        &recent,
        6,
    ));

    assert!(should_stop_coarse_stage(
        &SearchDepthPolicy::default(),
        16,
        0.09,
        &recent,
        6,
    ));
    assert!(!should_stop_coarse_stage(
        &SearchDepthPolicy::default(),
        15,
        0.09,
        &recent,
        6,
    ));
    assert!(!should_stop_coarse_stage(
        &SearchDepthPolicy::default(),
        16,
        0.11,
        &recent,
        6,
    ));
    assert!(!should_stop_coarse_stage(
        &SearchDepthPolicy::default(),
        16,
        0.09,
        &VecDeque::from(vec![0.14, 0.15, 0.13, 0.16]),
        6,
    ));
    assert!(!should_stop_coarse_stage(
        &SearchDepthPolicy::default(),
        16,
        0.09,
        &recent,
        5,
    ));
}

#[test]
fn recent_coarse_cost_window_retains_last_four_without_fifth_growth() {
    let mut recent_costs = VecDeque::new();
    recent_costs
        .try_reserve_exact(4)
        .expect("fixed rolling window capacity must reserve");
    let initial_capacity = recent_costs.capacity();

    for cost in [1.0, 2.0, 3.0, 4.0] {
        push_recent_coarse_cost(&mut recent_costs, cost);
    }
    assert_eq!(
        recent_costs.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    push_recent_coarse_cost(&mut recent_costs, 5.0);

    assert_eq!(
        recent_costs.iter().copied().collect::<Vec<_>>(),
        vec![2.0, 3.0, 4.0, 5.0]
    );
    assert_eq!(recent_costs.capacity(), initial_capacity);
}

/// A long deterministic scratch-warming sequence must not change plan bits.
#[test]
#[expect(
    clippy::suboptimal_flops,
    reason = "scratch-warming fixture preserves its deterministic non-fused LCG envelope"
)]
fn evaluate_plan_local_is_independent_of_cross_call_warming() -> anyhow::Result<()> {
    let ctx = make_leo_ctx()?;
    let x_test = [0.08, 1.00, 0.05];

    // Fresh cache baseline.
    let fresh = RefCell::new(SolveLocalWorkCache::new());
    let baseline = evaluate_plan_local(&x_test, &ctx, false, &fresh)?;

    // Warmed cache — 50 decisions drawn from the production envelope
    // via a deterministic LCG so the warming sequence is reproducible.
    let warmed = RefCell::new(SolveLocalWorkCache::new());
    let mut state: u64 = 0xDEAD_BEEF_C0DE_F00D;
    let mut next = || next_deterministic_lcg_unit_interval(&mut state);
    for _ in 0..50 {
        let x = [
            0.02 + 0.18 * next(),
            0.80 + 0.40 * next(),
            0.02 + 0.10 * next(),
        ];
        evaluate_plan_local(&x, &ctx, false, &warmed)?;
    }

    // Post-warmup eval of x_test must match fresh-cache eval bit-for-bit.
    let warmed_result = evaluate_plan_local(&x_test, &ctx, false, &warmed)?;

    assert_plan_result_fields_are_exhaustive(&baseline);
    assert_plan_result_scalar_bits_equal(&baseline, &warmed_result);
    assert_plan_result_vector_bits_equal(&baseline, &warmed_result);
    assert_plan_result_metadata_equal(&baseline, &warmed_result);
    Ok(())
}

/// Full `PlanResult` scratch-state independence.
///
/// Two independent scratch owners evaluate the same `x_test`; the resulting
/// `PlanResult` structs must be bit-for-bit equal across every field. The
/// destructure-with-no-
/// `..` pattern below is a deliberate compile-time regression guard:
/// any future field added to `PlanResult` will break this test's build
/// until its author has added an assertion here and considered whether
/// that field is cache-state-invariant.
fn assert_plan_result_fields_are_exhaustive(plan: &PlanResult) {
    // Compile-time regression guard. If a new `PlanResult` field is
    // added without updating this test, this destructure (with no `..`)
    // fails to build. Bindings are intentionally unused.
    let PlanResult {
        cost: _,
        valid: _,
        polish_steps: _,
        polish_evals: _,
        polish_time_us: _,
        polish_skipped: _,
        escape_triggered: _,
        time2phase_ratio: _,
        phase_sma_ratio: _,
        waittime_ratio: _,
        time2phase: _,
        waittime: _,
        tof: _,
        distance: _,
        deployer_distance: _,
        phase_sma: _,
        phase_dv: _,
        transfer_dv: _,
        arrival_dv: _,
        phase_dv_norm: _,
        transfer_dv_norm: _,
        arrival_dv_norm: _,
        payload_intercept_state: _,
        target_intercept_state: _,
        deployer_intercept_state: _,
        release_state: _,
        best_M: _,
        prograde: _,
        branch_rev: _,
        branch_low_path: _,
        branch_tof_s: _,
        branch_departure_dv: _,
        branch_arrival_dv: _,
        branch_total_dv: _,
        branch_status: _,
        branch_rejection: _,
        intercept_jd: _,
        waittime_jd_start: _,
        tof_jd_start: _,
        timing_failure_reason: _,
        func_evals: _,
        optimizer_func_evals: _,
        optimizer_converged: _,
        warm_start_used: _,
        dep_period: _,
        j2_iteration_count: _,
        j2_endpoint_residual_m: _,
        post_hf_endpoint_residual_m: _,
        replay_provenance: _,
    } = plan;
}

fn assert_plan_result_scalar_bits_equal(plan_a: &PlanResult, plan_b: &PlanResult) {
    // Floats use bit-exact equality: both values come from the same
    // deterministic compute path, so no tolerance is acceptable.
    assert_eq!(plan_a.valid, plan_b.valid, "valid");
    assert_eq!(plan_a.cost.to_bits(), plan_b.cost.to_bits(), "cost");
    assert_eq!(plan_a.polish_steps, plan_b.polish_steps, "polish_steps");
    assert_eq!(plan_a.polish_evals, plan_b.polish_evals, "polish_evals");
    assert_eq!(
        plan_a.polish_time_us, plan_b.polish_time_us,
        "polish_time_us"
    );
    assert_eq!(
        plan_a.polish_skipped, plan_b.polish_skipped,
        "polish_skipped"
    );
    assert_eq!(
        plan_a.escape_triggered, plan_b.escape_triggered,
        "escape_triggered"
    );
    assert_eq!(
        plan_a.time2phase_ratio.to_bits(),
        plan_b.time2phase_ratio.to_bits(),
        "time2phase_ratio"
    );
    assert_eq!(
        plan_a.phase_sma_ratio.to_bits(),
        plan_b.phase_sma_ratio.to_bits(),
        "phase_sma_ratio"
    );
    assert_eq!(
        plan_a.waittime_ratio.to_bits(),
        plan_b.waittime_ratio.to_bits(),
        "waittime_ratio"
    );
    assert_eq!(
        plan_a.time2phase.to_bits(),
        plan_b.time2phase.to_bits(),
        "time2phase"
    );
    assert_eq!(
        plan_a.waittime.to_bits(),
        plan_b.waittime.to_bits(),
        "waittime"
    );
    assert_eq!(plan_a.tof.to_bits(), plan_b.tof.to_bits(), "tof");
    assert_eq!(
        plan_a.distance.to_bits(),
        plan_b.distance.to_bits(),
        "distance"
    );
    assert_eq!(
        plan_a.deployer_distance.to_bits(),
        plan_b.deployer_distance.to_bits(),
        "deployer_distance"
    );
    assert_eq!(
        plan_a.phase_sma.to_bits(),
        plan_b.phase_sma.to_bits(),
        "phase_sma"
    );
    assert_eq!(
        plan_a.phase_dv_norm.to_bits(),
        plan_b.phase_dv_norm.to_bits(),
        "phase_dv_norm"
    );
    assert_eq!(
        plan_a.transfer_dv_norm.to_bits(),
        plan_b.transfer_dv_norm.to_bits(),
        "transfer_dv_norm"
    );
    assert_eq!(
        plan_a.arrival_dv_norm.to_bits(),
        plan_b.arrival_dv_norm.to_bits(),
        "arrival_dv_norm"
    );
}

fn assert_plan_result_vector_bits_equal(plan_a: &PlanResult, plan_b: &PlanResult) {
    for (index, (value_a, value_b)) in plan_a
        .phase_dv
        .iter()
        .zip(plan_b.phase_dv.iter())
        .enumerate()
    {
        assert_eq!(value_a.to_bits(), value_b.to_bits(), "phase_dv[{index}]");
    }
    for (index, (value_a, value_b)) in plan_a
        .transfer_dv
        .iter()
        .zip(plan_b.transfer_dv.iter())
        .enumerate()
    {
        assert_eq!(value_a.to_bits(), value_b.to_bits(), "transfer_dv[{index}]");
    }
    for (index, (value_a, value_b)) in plan_a
        .arrival_dv
        .iter()
        .zip(plan_b.arrival_dv.iter())
        .enumerate()
    {
        assert_eq!(value_a.to_bits(), value_b.to_bits(), "arrival_dv[{index}]");
    }
    for (index, (value_a, value_b)) in plan_a
        .payload_intercept_state
        .iter()
        .zip(plan_b.payload_intercept_state.iter())
        .enumerate()
    {
        assert_eq!(
            value_a.to_bits(),
            value_b.to_bits(),
            "payload_intercept_state[{index}]"
        );
    }
    for (index, (value_a, value_b)) in plan_a
        .target_intercept_state
        .iter()
        .zip(plan_b.target_intercept_state.iter())
        .enumerate()
    {
        assert_eq!(
            value_a.to_bits(),
            value_b.to_bits(),
            "target_intercept_state[{index}]"
        );
    }
    for (index, (value_a, value_b)) in plan_a
        .deployer_intercept_state
        .iter()
        .zip(plan_b.deployer_intercept_state.iter())
        .enumerate()
    {
        assert_eq!(
            value_a.to_bits(),
            value_b.to_bits(),
            "deployer_intercept_state[{index}]"
        );
    }
    for (index, (value_a, value_b)) in plan_a
        .release_state
        .iter()
        .zip(plan_b.release_state.iter())
        .enumerate()
    {
        assert_eq!(
            value_a.to_bits(),
            value_b.to_bits(),
            "release_state[{index}]"
        );
    }
}

fn assert_plan_result_metadata_equal(plan_a: &PlanResult, plan_b: &PlanResult) {
    assert_eq!(plan_a.best_M, plan_b.best_M, "best_M");
    assert_eq!(plan_a.prograde, plan_b.prograde, "prograde");
    assert_eq!(plan_a.branch_rev, plan_b.branch_rev, "branch_rev");
    assert_eq!(
        plan_a.branch_low_path, plan_b.branch_low_path,
        "branch_low_path"
    );
    assert_eq!(
        plan_a.branch_tof_s.to_bits(),
        plan_b.branch_tof_s.to_bits(),
        "branch_tof_s"
    );
    assert_eq!(
        plan_a.branch_departure_dv.to_bits(),
        plan_b.branch_departure_dv.to_bits(),
        "branch_departure_dv"
    );
    assert_eq!(
        plan_a.branch_arrival_dv.to_bits(),
        plan_b.branch_arrival_dv.to_bits(),
        "branch_arrival_dv"
    );
    assert_eq!(
        plan_a.branch_total_dv.to_bits(),
        plan_b.branch_total_dv.to_bits(),
        "branch_total_dv"
    );
    assert_eq!(plan_a.branch_status, plan_b.branch_status, "branch_status");
    assert_eq!(
        plan_a.branch_rejection, plan_b.branch_rejection,
        "branch_rejection"
    );
    assert_eq!(
        plan_a.intercept_jd.to_bits(),
        plan_b.intercept_jd.to_bits(),
        "intercept_jd"
    );
    assert_eq!(
        plan_a.waittime_jd_start.to_bits(),
        plan_b.waittime_jd_start.to_bits(),
        "waittime_jd_start"
    );
    assert_eq!(
        plan_a.tof_jd_start.to_bits(),
        plan_b.tof_jd_start.to_bits(),
        "tof_jd_start"
    );
    assert_eq!(
        plan_a.timing_failure_reason, plan_b.timing_failure_reason,
        "timing_failure_reason"
    );
    assert_eq!(plan_a.func_evals, plan_b.func_evals, "func_evals");
    assert_eq!(
        plan_a.optimizer_func_evals, plan_b.optimizer_func_evals,
        "optimizer_func_evals"
    );
    assert_eq!(
        plan_a.optimizer_converged, plan_b.optimizer_converged,
        "optimizer_converged"
    );
    assert_eq!(
        plan_a.warm_start_used, plan_b.warm_start_used,
        "warm_start_used"
    );
    assert_eq!(
        plan_a.dep_period.to_bits(),
        plan_b.dep_period.to_bits(),
        "dep_period"
    );
    assert_eq!(
        plan_a.j2_iteration_count, plan_b.j2_iteration_count,
        "j2_iteration_count"
    );
    assert_eq!(
        plan_a.j2_endpoint_residual_m.to_bits(),
        plan_b.j2_endpoint_residual_m.to_bits(),
        "j2_endpoint_residual_m"
    );
    assert_eq!(
        plan_a.post_hf_endpoint_residual_m.to_bits(),
        plan_b.post_hf_endpoint_residual_m.to_bits(),
        "post_hf_endpoint_residual_m"
    );
}

fn evaluate_plan_local_bits_for_test(
    x: &[f64; 3],
    ctx: &PlanContext,
) -> anyhow::Result<(bool, u64)> {
    let cache = RefCell::new(SolveLocalWorkCache::new());
    let plan = evaluate_plan_local(x, ctx, false, &cache)?;
    Ok((plan.valid, plan.cost.to_bits()))
}

/// Pass 31 (Item H6 in B1 audit) — LLVM SIMD reduction order invariance
/// test. The pass 30 (B1) audit identified the prime remaining suspect for
/// pass 15's residual non-determinism: under `-Cllvm-args=-fp-contract=fast`,
/// LLVM is permitted to fuse / reorder floating-point operations. If the
/// reduction order chosen by the optimizer depends on the call-site context
/// (e.g. inlined inside a rayon `par_iter` closure vs inlined inside a serial
/// for-loop), the same `evaluate_plan_local` call could return ULP-different
/// `cost` values, cascading through NSGA-II selection and producing the
/// candidate-count divergence pass 15 hit.
///
/// This test pins the invariant. It runs the SAME 256 evaluations through
/// two iteration shapes: a plain serial `for` loop, and a rayon
/// `par_iter` chain that we constrain to one worker via
/// `ThreadPoolBuilder::new().num_threads(1)`. Single-threaded `par_iter`
/// rules out thread-state interference; any divergence is purely LLVM
/// context-sensitivity.
///
/// If this test FAILS, H6 is confirmed and the fix is one of:
///   - drop `-Cllvm-args=-fp-contract=on` in `scripts/release-policy-common.sh` (perf cost)
///   - add `#[inline(never)]` boundaries on the inner-most fp-sensitive helpers
///   - switch to explicit `f64::mul_add` everywhere fp-fusion was implicit
#[test]
#[expect(
    clippy::suboptimal_flops,
    reason = "serial-parallel parity fixture preserves its deterministic non-fused LCG envelope"
)]
fn fp_contract_par_iter_path_matches_serial_path_single_thread() -> anyhow::Result<()> {
    let ctx = make_leo_ctx()?;

    // 256 deterministic decisions across the production envelope. Same
    // LCG seed used everywhere else for reproducibility.
    let mut state: u64 = 0xDEAD_BEEF_C0DE_F00D;
    let mut next = || next_deterministic_lcg_unit_interval(&mut state);
    let xs: Vec<[f64; 3]> = (0..256)
        .map(|_| {
            let x0 = 0.02 + 0.18 * next();
            let x1 = 0.80 + 0.40 * next();
            let x2 = 0.02 + 0.10 * next();
            [x0, x1, x2]
        })
        .collect();

    let serial: Vec<(bool, u64)> = xs
        .iter()
        .map(|x| evaluate_plan_local_bits_for_test(x, &ctx))
        .collect::<anyhow::Result<_>>()?;

    // Force single-worker rayon for this thread; if a global pool exists,
    // this creates a scoped pool that doesn't disturb other tests.
    let parallel: Vec<(bool, u64)> = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .map_err(|error| anyhow::anyhow!("build single-thread rayon pool: {error}"))?;
        pool.install(|| {
            use rayon::prelude::*;
            xs.par_iter()
                .map(|x| evaluate_plan_local_bits_for_test(x, &ctx))
                .collect::<anyhow::Result<_>>()
        })?
    };

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, [f64; 3], f64, f64)> = None;
    for (i, (x, ((sv, sb), (pv, pb)))) in xs
        .iter()
        .zip(serial.iter().zip(parallel.iter()))
        .enumerate()
    {
        if sv != pv || sb != pb {
            mismatches = mismatches
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("fixed 256-row parity fixture count overflow"))?;
            if first_mismatch.is_none() {
                first_mismatch = Some((i, *x, f64::from_bits(*sb), f64::from_bits(*pb)));
            }
        }
    }
    let (first_index, first_x, first_serial_cost, first_parallel_cost) =
        first_mismatch.unwrap_or((0, [0.0; 3], 0.0, 0.0));
    let evaluation_count = xs.len();

    anyhow::ensure!(
        mismatches == 0,
        "H6 (LLVM fp-contract context sensitivity) detected: {mismatches} of {evaluation_count} costs diverge between \
         serial-for-loop and single-thread par_iter paths. First mismatch: idx={first_index} x={first_x:?} \
         serial_cost={first_serial_cost} par_cost={first_parallel_cost}"
    );
    Ok(())
}

/// Pass 31 follow-on (Item H7) — multi-thread `par_iter` determinism test.
///
/// Pass 31 (H6) ruled out the single-thread LLVM context-sensitivity
/// hypothesis. The remaining suspect for pass 15's residual non-determinism
/// is H7: per-rayon-worker TLS state in `satpy_core` (gravity coefficient
/// caches, propagator scratch, etc.) producing different `PlanResult`s when
/// the same `(x, ctx)` is evaluated by different workers.
///
/// This test forces multi-thread `par_iter` (`num_threads(8)`), runs the
/// same 256-decision envelope, and asserts bit-identical results vs a
/// serial for-loop. If it FAILS, H7 is confirmed and the divergence
/// originates in cross-thread TLS state that pass 23 (C4) did NOT cover.
/// If it PASSES, the divergence must originate inside oxymoo's
/// NSGA-II driver (rank/crowding/selection), not in the Problem evaluator.
#[test]
#[expect(
    clippy::suboptimal_flops,
    reason = "serial-parallel parity fixture preserves its deterministic non-fused LCG envelope"
)]
fn fp_contract_par_iter_path_matches_serial_path_multi_thread() -> anyhow::Result<()> {
    let ctx = make_leo_ctx()?;

    // Same 256-tuple LCG envelope as the single-thread sibling test.
    let mut state: u64 = 0xDEAD_BEEF_C0DE_F00D;
    let mut next = || next_deterministic_lcg_unit_interval(&mut state);
    let xs: Vec<[f64; 3]> = (0..256)
        .map(|_| {
            let x0 = 0.02 + 0.18 * next();
            let x1 = 0.80 + 0.40 * next();
            let x2 = 0.02 + 0.10 * next();
            [x0, x1, x2]
        })
        .collect();

    let serial: Vec<(bool, u64)> = xs
        .iter()
        .map(|x| evaluate_plan_local_bits_for_test(x, &ctx))
        .collect::<anyhow::Result<_>>()?;

    // Force multi-worker rayon for this thread (scoped, doesn't disturb
    // other tests).
    let parallel: Vec<(bool, u64)> = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .map_err(|error| anyhow::anyhow!("build 8-thread rayon pool: {error}"))?;
        pool.install(|| {
            use rayon::prelude::*;
            xs.par_iter()
                .map(|x| evaluate_plan_local_bits_for_test(x, &ctx))
                .collect::<anyhow::Result<_>>()
        })?
    };

    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, [f64; 3], f64, f64)> = None;
    for (i, (x, ((sv, sb), (pv, pb)))) in xs
        .iter()
        .zip(serial.iter().zip(parallel.iter()))
        .enumerate()
    {
        if sv != pv || sb != pb {
            mismatches = mismatches
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("fixed 256-row parity fixture count overflow"))?;
            if first_mismatch.is_none() {
                first_mismatch = Some((i, *x, f64::from_bits(*sb), f64::from_bits(*pb)));
            }
        }
    }
    let (first_index, first_x, first_serial_cost, first_parallel_cost) =
        first_mismatch.unwrap_or((0, [0.0; 3], 0.0, 0.0));
    let evaluation_count = xs.len();

    anyhow::ensure!(
        mismatches == 0,
        "H7 (multi-thread TLS state leak) detected: {mismatches} of {evaluation_count} costs diverge between \
         serial for-loop and 8-thread par_iter paths. First mismatch: idx={first_index} x={first_x:?} \
         serial_cost={first_serial_cost} par_cost={first_parallel_cost}"
    );
    Ok(())
}

#[test]
fn evaluate_plan_local_reuses_phase_state() -> anyhow::Result<()> {
    let ctx = make_leo_ctx()?;
    let cache = RefCell::new(SolveLocalWorkCache::new());
    let x1 = [0.08, 1.00, 0.05];
    let x2 = [0.08, 1.03, 0.05];

    evaluate_plan_local(&x1, &ctx, false, &cache)?;
    evaluate_plan_local(&x2, &ctx, false, &cache)?;

    anyhow::ensure!(
        cache.borrow().phase_state_cache.len() == 1,
        "phase-state cache must retain one entry"
    );
    Ok(())
}

#[test]
fn test_sort_grid_seed_candidates_by_hint_prefers_nearest_then_lexicographic() {
    let mut seeds = vec![
        SolverSeed {
            x: [0.40, 1.10, 0.20],
            warm_start_used: false,
        },
        SolverSeed {
            x: [0.22, 1.02, 0.08],
            warm_start_used: false,
        },
        SolverSeed {
            x: [0.18, 0.99, 0.06],
            warm_start_used: false,
        },
        SolverSeed {
            x: [0.22, 1.02, 0.07],
            warm_start_used: false,
        },
    ];

    sort_grid_seed_candidates_by_hint(&mut seeds, [0.20, 1.00, 0.05]);

    let ordered: Vec<[f64; 3]> = seeds.iter().map(|seed| seed.x).collect();
    assert_eq!(
        ordered,
        vec![
            [0.18, 0.99, 0.06],
            [0.22, 1.02, 0.07],
            [0.22, 1.02, 0.08],
            [0.40, 1.10, 0.20],
        ]
    );
}

#[test]
fn test_search_depth_policy_defaults_pin_historical_constants() {
    let policy = SearchDepthPolicy::default();
    assert_eq!(policy.tof_sample_budget, 64);
    assert!(policy.coarse_early_stop);
    assert_eq!(policy.fine_total_limit, 10);
    assert_eq!(
        policy.coarse_reject_margin_km_s.to_bits(),
        0.05_f64.to_bits()
    );
    assert_eq!(policy.seed_fine_margin_km_s.to_bits(), 0.05_f64.to_bits());
    assert_eq!(policy.oxymoo_policy, OxyMooPolicy::Full);
    assert_eq!(policy.clamped_tof_budget(), 64);
    let oversize = SearchDepthPolicy {
        tof_sample_budget: 100_000,
        ..SearchDepthPolicy::default()
    };
    assert_eq!(oversize.clamped_tof_budget(), crate::types::MAX_TOF_SAMPLES);
    let zero = SearchDepthPolicy {
        tof_sample_budget: 0,
        ..SearchDepthPolicy::default()
    };
    assert_eq!(zero.clamped_tof_budget(), 1);
}

#[test]
fn test_coarse_early_stop_margin_uses_policy_value() {
    use std::collections::VecDeque;
    let recent: VecDeque<f64> = [0.17, 0.18, 0.19, 0.20].into_iter().collect();
    // Default margins: best 0.08 + max(0.08, 0.05) => 0.16 threshold; all
    // recent costs are strictly above it, so the stage stops.
    assert!(should_stop_coarse_stage(
        &SearchDepthPolicy::default(),
        32,
        0.08,
        &recent,
        8,
    ));
    // A widened reject margin (0.15) lifts the threshold past the recent
    // costs, so the coarse stage keeps scanning.
    let widened = SearchDepthPolicy {
        coarse_reject_margin_km_s: 0.15,
        ..SearchDepthPolicy::default()
    };
    assert!(!should_stop_coarse_stage(&widened, 32, 0.08, &recent, 8,));
}

#[test]
fn test_evaluate_plan_valid_across_tof_budgets() -> anyhow::Result<()> {
    let mut ctx = make_pair_ctx_for_test(&TestPairContextRequest {
        dep_eci: make_circular_state(550.0, 0.9, 0.2, 0.1),
        tgt_eci: make_circular_state(620.0, 0.9, 0.2, 0.9),
        max_time_s: 86_400.0,
        max_phase_dv: 1.25,
        max_transfer_dv: 1.25,
        max_revs: 4,
        min_perigee: 6_500.0,
        max_apogee: 50_000.0,
        sampling_mode: SamplingMode::Fast,
        search_depth: SearchDepthPolicy::default(),
        epoch_jd: 0.0,
        distance_tol: 0.025,
        deployer_min_distance: 0.12,
        tof_penalty_weight: 0.1,
        revolution_cap: 2.0,
    })?;
    let x = [0.2, 1.0, 0.1];
    let default_plan = evaluate_plan(&x, &ctx, false)?;
    for budget in [8usize, 64, 256] {
        ctx.search_depth.tof_sample_budget = budget;
        let plan = evaluate_plan(&x, &ctx, false)?;
        anyhow::ensure!(
            plan.cost.is_finite() || plan.cost == INVALID_COST,
            "budget {budget} produced non-finite, non-sentinel cost"
        );
    }
    // Restating the default budget reproduces the default-policy result.
    ctx.search_depth.tof_sample_budget = 64;
    let replay = evaluate_plan(&x, &ctx, false)?;
    anyhow::ensure!(replay.cost.to_bits() == default_plan.cost.to_bits());
    anyhow::ensure!(replay.total_dv().to_bits() == default_plan.total_dv().to_bits());
    Ok(())
}

// ====================================================================
// Representative multi-rev LEO VerifiedSuperset work-count audit. Explicit
// one-thread and nested-worker runs pin serial accounting; isolated global
// width-4 child proves top-level leaf fan-out.
// ====================================================================
fn run_audit_front_solve() -> anyhow::Result<TransferFront> {
    let mut ctx = make_leo_ctx()?;
    Ok(solve_plan_oxymoo_front_internal(
        &mut ctx,
        None,
        Some([0.0, 1.0, 0.0]),
        FrontOutputMode::VerifiedSuperset,
        None,
        DeltaVAnchorPolicy::Full,
        TransferMooPolicy::Full,
    )?)
}

fn audit_work_count_metrics() -> anyhow::Result<VerifiedSupersetStageMetrics> {
    // Deterministic serial reference profile: a single-thread rayon pool
    // drives `current_num_threads() == 1`, so the OxyMOO batch-parallel gate
    // stays off and the recorded serial work counts are reproducible on any
    // host (regardless of the ambient global-pool size).
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    pool.install(|| -> anyhow::Result<VerifiedSupersetStageMetrics> {
        Ok(run_audit_front_solve()?.verified_superset_metrics)
    })
}

fn audit_work_count_metrics_nested() -> anyhow::Result<VerifiedSupersetStageMetrics> {
    // `ThreadPool::install` executes this closure on a rayon worker. Adaptive
    // dispatch therefore keeps every leaf stage serial despite pool width 4.
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    pool.install(|| -> anyhow::Result<VerifiedSupersetStageMetrics> {
        Ok(run_audit_front_solve()?.verified_superset_metrics)
    })
}

#[test]
fn leaf_parallel_gate_requires_top_level_multi_thread_work() {
    assert!(should_use_leaf_parallel(true, 2, 2, 4, true));
    assert!(!should_use_leaf_parallel(true, 2, 2, 4, false));
    assert!(!should_use_leaf_parallel(true, 2, 2, 1, true));
    assert!(!should_use_leaf_parallel(true, 1, 2, 4, true));
    assert!(!should_use_leaf_parallel(false, 2, 2, 4, true));
}

#[test]
fn nested_leaf_fanout_gates_serialize() -> anyhow::Result<()> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let observed = pool.install(|| -> anyhow::Result<_> {
        anyhow::ensure!(rayon::current_thread_index().is_some());
        let mut ctx = make_leo_ctx()?;
        prepare_single_pair_context(&mut ctx);
        let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
        let problem =
            TransferMooProblem::new(ctx.clone(), RefCell::new(FxHashMap::default()), policy)?;
        Ok((
            should_use_polish_parallel(&ctx, POLISH_PARALLEL_MIN_CANDIDATES),
            problem.should_use_oxymoo_batch_parallel(OXYMOO_BATCH_PARALLEL_MIN_ROWS),
            should_use_anchor_parallel(&ctx, 1),
            should_use_branch_expansion_parallel(&ctx, BRANCH_EXPANSION_PARALLEL_MIN_SOURCES),
            should_use_deterministic_grid_parallel(&ctx, DETERMINISTIC_GRID_PARALLEL_MIN_POINTS),
        ))
    })?;

    anyhow::ensure!(observed == (false, false, false, false, false));
    Ok(())
}

#[test]
fn nested_outer_pool_keeps_leaf_fanouts_serial() -> anyhow::Result<()> {
    let m = audit_work_count_metrics_nested()?;
    anyhow::ensure!(m.oxymoo_parallel_batch_count == 0);
    anyhow::ensure!(m.anchor_parallel_count == 0);
    anyhow::ensure!(m.branch_parallel_count == 0);
    anyhow::ensure!(m.polish_parallel_count == 0);
    anyhow::ensure!(
        m.oxymoo_parallel_batch_count + m.oxymoo_serial_batch_count == 6,
        "1 init + 5 generation batches, each classified once"
    );
    anyhow::ensure!(
        m.oxymoo_full_eval_count + m.oxymoo_eval_cache_hit_count == 168,
        "OxyMOO Full policy generation evals: full={} + hit={} should be 168",
        m.oxymoo_full_eval_count,
        m.oxymoo_eval_cache_hit_count
    );

    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let grid_parallel_hits = pool.install(|| -> anyhow::Result<usize> {
        reset_deterministic_grid_parallel_path_hits();
        let mut ctx = make_leo_ctx()?;
        let front = verified_superset_deterministic_grid_fallback(&mut ctx, false)?;
        anyhow::ensure!(!front.candidates.is_empty());
        Ok(deterministic_grid_parallel_path_hits())
    })?;
    anyhow::ensure!(grid_parallel_hits == 0);
    Ok(())
}

// ====================================================================
// OxyMOO identity helpers. Custom `ThreadPool::install` calls are nested by
// definition and therefore exercise leaf-serial behavior at every width.
// Direct helper tests below still validate parallel algorithm identity.
// ====================================================================
fn run_moo_population() -> anyhow::Result<Nsga2Result> {
    let mut ctx = make_leo_ctx()?;
    // Caches orbits AND opts the single-pair context into batch parallelism.
    prepare_single_pair_context(&mut ctx);
    anyhow::ensure!(ctx.execution_policy.allow_oxymoo_batch_parallel);
    let plan_cache: TransferMooPlanCache = RefCell::new(FxHashMap::default());
    let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let problem = TransferMooProblem::new(ctx.clone(), plan_cache, policy)?;
    let config = transfer_moo_config_with_initial_decisions(&ctx, Vec::new())?;
    let optimizer = Nsga2::new(problem, config)?;
    let (_problem, result) = optimizer.run_owned_with_problem()?;
    Ok(result)
}

fn assert_bits_eq(serial: &[f64], parallel: &[f64], label: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        serial.len() == parallel.len(),
        "{label}: length mismatch ({} vs {})",
        serial.len(),
        parallel.len()
    );
    // 0 == 0 satisfies the length check and `zip` then yields nothing, so two
    // empty result vectors would be certified bit-identical having compared no
    // bits. Every caller passes a populated objective/decision vector.
    anyhow::ensure!(
        !serial.is_empty(),
        "{label}: nothing to compare -- both arms returned an empty vector"
    );
    for (i, (a, b)) in serial.iter().zip(parallel.iter()).enumerate() {
        anyhow::ensure!(
            a.to_bits() == b.to_bits(),
            "{label}[{i}] differs: serial={a} parallel={b}"
        );
    }
    Ok(())
}

/// Wall-clock fields vary with scheduling. Keep every discrete diagnostic
/// counter while comparing pool-width accounting exactly.
fn evaluation_diagnostic_counts_only(
    counters: &EvaluationDiagnosticCounters,
) -> EvaluationDiagnosticCounters {
    let mut counters = *counters;
    counters.j2_correction_residual_m_sum = 0.0;
    counters.j2_correction_rejected_residual_m_sum = 0.0;
    counters.branch_shared_prepare_s = 0.0;
    counters.branch_phase_release_s = 0.0;
    counters.branch_target_propagation_s = 0.0;
    counters.branch_lambert_sampling_s = 0.0;
    counters.branch_brent_s = 0.0;
    counters.branch_j2_correction_s = 0.0;
    counters.hf_propagation.target_grid_s = 0.0;
    counters.hf_propagation.brent_s = 0.0;
    counters.hf_propagation.intercept_refinement_s = 0.0;
    counters
}

fn assert_evaluation_diagnostic_accounting_matches(
    serial: &EvaluationDiagnosticCounters,
    parallel: &EvaluationDiagnosticCounters,
    label: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        evaluation_diagnostic_counts_only(serial) == evaluation_diagnostic_counts_only(parallel),
        "{label}: diagnostic counters differ: serial={serial:?} parallel={parallel:?}"
    );
    anyhow::ensure!(
        serial.j2_correction_residual_m_sum.to_bits()
            == parallel.j2_correction_residual_m_sum.to_bits()
            && serial.j2_correction_rejected_residual_m_sum.to_bits()
                == parallel.j2_correction_rejected_residual_m_sum.to_bits(),
        "{label}: deterministic J2 residual sums differ"
    );
    Ok(())
}

fn pool_worker_tls_snapshots(
    pool: &rayon::ThreadPool,
) -> Vec<(WorkCountCounters, EvaluationDiagnosticCounters)> {
    pool.broadcast(|_| (work_count_snapshot(), evaluation_diagnostic_snapshot()))
}

fn assert_pool_worker_tls_restored(
    before: &[(WorkCountCounters, EvaluationDiagnosticCounters)],
    after: &[(WorkCountCounters, EvaluationDiagnosticCounters)],
    label: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        before.len() == after.len(),
        "{label}: worker snapshot count changed ({} vs {})",
        before.len(),
        after.len()
    );
    for (worker, (before, after)) in before.iter().zip(after.iter()).enumerate() {
        anyhow::ensure!(
            before == after,
            "{label}: worker {worker} retained TLS accounting: before={before:?} after={after:?}"
        );
    }
    Ok(())
}

fn run_direct_oxymoo_batch(parallel: bool) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let decisions = vec![
        0.30, 1.00, 0.20, 0.40, 1.10, 0.25, 0.30, 1.00, 0.20, 0.55, 0.95, 0.10,
    ];
    let row_count = decisions.len() / 3;
    let mut ctx = make_leo_ctx()?;
    prepare_single_pair_context(&mut ctx);
    let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let problem = TransferMooProblem::new(ctx, RefCell::new(FxHashMap::default()), policy)?;
    let objective_count = row_count
        .checked_mul(TRANSFER_MOO_OBJECTIVES)
        .ok_or_else(|| anyhow::anyhow!("test OxyMOO objective count must fit usize"))?;
    let mut objectives = vec![0.0; objective_count];
    let mut violations = vec![0.0; row_count];
    if parallel {
        problem.evaluate_batch_parallel(
            &decisions,
            3,
            TRANSFER_MOO_OBJECTIVES,
            &mut objectives,
            &mut violations,
        )?;
    } else {
        problem.evaluate_batch_serial(
            &decisions,
            3,
            TRANSFER_MOO_OBJECTIVES,
            &mut objectives,
            &mut violations,
        )?;
    }
    Ok((objectives, violations))
}

fn run_direct_oxymoo_batch_with_accounting(
    parallel: bool,
) -> anyhow::Result<(
    Vec<f64>,
    Vec<f64>,
    WorkCountCounters,
    EvaluationDiagnosticCounters,
)> {
    let work_before = work_count_snapshot();
    let diagnostics_before = evaluation_diagnostic_snapshot();
    let outcome = (|| {
        let (objectives, violations) = run_direct_oxymoo_batch(parallel)?;
        let work_delta = work_count_snapshot().delta_since(work_before)?;
        let diagnostics_delta = evaluation_diagnostic_snapshot().delta_since(diagnostics_before)?;
        Ok((objectives, violations, work_delta, diagnostics_delta))
    })();
    restore_work_count_snapshot(work_before);
    restore_evaluation_diagnostics(&diagnostics_before);
    outcome
}

fn run_colliding_oxymoo_batch(
    decisions: [[f64; 3]; 3],
    parallel: bool,
) -> anyhow::Result<(
    Vec<f64>,
    Vec<f64>,
    WorkCountCounters,
    usize,
    usize,
    usize,
    bool,
)> {
    let work_before = work_count_snapshot();
    let diagnostics_before = evaluation_diagnostic_snapshot();
    let outcome = (|| {
        let mut ctx = make_leo_ctx()?;
        ctx.max_phase_dv = 1.0e-6;
        prepare_single_pair_context(&mut ctx);
        let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
        let problem = TransferMooProblem::new(ctx, RefCell::new(FxHashMap::default()), policy)?;
        let flat_decisions: Vec<f64> = decisions.into_iter().flatten().collect();
        let mut objectives = vec![0.0; 3 * TRANSFER_MOO_OBJECTIVES];
        let mut violations = vec![0.0; 3];
        if parallel {
            problem.evaluate_batch_parallel(
                &flat_decisions,
                3,
                TRANSFER_MOO_OBJECTIVES,
                &mut objectives,
                &mut violations,
            )?;
        } else {
            problem.evaluate_batch_serial(
                &flat_decisions,
                3,
                TRANSFER_MOO_OBJECTIVES,
                &mut objectives,
                &mut violations,
            )?;
        }
        let work = work_count_snapshot().delta_since(work_before)?;
        let diagnostics = evaluation_diagnostic_snapshot().delta_since(diagnostics_before)?;
        let phase_lookups = diagnostics
            .phase_state_cache_hit_count
            .checked_add(diagnostics.phase_state_cache_miss_count)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        let hits_before_probe = problem.eval_cache_hits();
        let misses_before_probe = problem.eval_cache_misses();
        let mut probe_objectives = [0.0; TRANSFER_MOO_OBJECTIVES];
        let probe_cv = problem.evaluate(&decisions[2], &mut probe_objectives)?;
        let resident_hit = problem.eval_cache_hits()
            == hits_before_probe
                .checked_add(1)
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?
            && problem.eval_cache_misses() == misses_before_probe;
        let expected_probe_objectives = objectives
            .get(4..6)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        let expected_probe_cv = violations
            .get(2)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        anyhow::ensure!(
            probe_objectives
                .iter()
                .copied()
                .map(f64::to_bits)
                .eq(expected_probe_objectives.iter().copied().map(f64::to_bits))
                && probe_cv.to_bits() == expected_probe_cv.to_bits(),
            "resident-key probe changed final A bits"
        );
        Ok((
            objectives,
            violations,
            work,
            phase_lookups,
            hits_before_probe,
            misses_before_probe,
            resident_hit,
        ))
    })();
    restore_work_count_snapshot(work_before);
    restore_evaluation_diagnostics(&diagnostics_before);
    outcome
}

#[test]
fn oxymoo_colliding_batch_serial_parallel_match_state_and_work() -> anyhow::Result<()> {
    let mut ctx = make_leo_ctx()?;
    prepare_single_pair_context(&mut ctx);
    let policy = TransferMooPolicy::from(ctx.search_depth.oxymoo_policy);
    let capacity = TransferMooProblem::new(ctx, RefCell::new(FxHashMap::default()), policy)?
        .eval_cache_capacity();
    let (first, second) = colliding_transfer_decisions(capacity)?;
    let decisions = [first, second, first];
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let serial = pool1.install(|| run_colliding_oxymoo_batch(decisions, false))?;
    let parallel_one = pool1.install(|| run_colliding_oxymoo_batch(decisions, true))?;
    let parallel = pool4.install(|| run_colliding_oxymoo_batch(decisions, true))?;

    assert_bits_eq(
        &serial.0,
        &parallel_one.0,
        "pool-1 colliding batch objectives",
    )?;
    assert_bits_eq(
        &serial.1,
        &parallel_one.1,
        "pool-1 colliding batch violations",
    )?;
    anyhow::ensure!(
        serial.2 == parallel_one.2,
        "pool-1 colliding batch work differs: serial={serial:?} parallel={parallel_one:?}"
    );
    anyhow::ensure!(
        (serial.4, serial.5) == (parallel_one.4, parallel_one.5),
        "pool-1 colliding batch cache differs: serial={serial:?} parallel={parallel_one:?}"
    );
    assert_bits_eq(&serial.0, &parallel.0, "colliding batch objectives")?;
    assert_bits_eq(&serial.1, &parallel.1, "colliding batch violations")?;
    anyhow::ensure!(
        serial.2.plan_full_evaluations == 3
            && parallel_one.2.plan_full_evaluations == 3
            && parallel.2.plan_full_evaluations == 3,
        "hostile A/B/A fixture must execute three full plans at every width: serial={serial:?} parallel_one={parallel_one:?} parallel={parallel:?}"
    );
    anyhow::ensure!(
        (serial.4, serial.5) == (parallel.4, parallel.5),
        "colliding batch cache tallies differ: serial={:?} parallel={:?}",
        (serial.4, serial.5),
        (parallel.4, parallel.5)
    );
    anyhow::ensure!(
        serial.3 == 3 && parallel_one.3 == 3 && parallel.3 == 3,
        "hostile A/B/A fixture must request three phase lookups at every width: serial={serial:?} parallel_one={parallel_one:?} parallel={parallel:?}"
    );
    anyhow::ensure!(
        (serial.4, serial.5) == (0, 3),
        "hostile A/B/A fixture must remain three ordered misses: {serial:?}"
    );
    anyhow::ensure!(
        serial.6 && parallel_one.6 && parallel.6,
        "final direct slot must retain the last A at every width"
    );
    Ok(())
}

#[test]
fn oxymoo_batch_helpers_match_bit_identically() -> anyhow::Result<()> {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let serial = pool1.install(|| run_direct_oxymoo_batch(false))?;
    // Direct helper call intentionally bypasses adaptive runtime gate. This
    // isolates parallel-kernel identity from dispatch-policy behavior.
    let parallel = pool4.install(|| run_direct_oxymoo_batch(true))?;
    assert_bits_eq(&serial.0, &parallel.0, "batch objectives")?;
    assert_bits_eq(&serial.1, &parallel.1, "batch violations")?;
    Ok(())
}

/// A Rayon `par_iter` may execute one row on the calling pool worker. The
/// parallel reducer must restore that worker baseline before replaying its
/// delta, otherwise pool-1 accounting doubles while pool-N may not.
#[test]
fn oxymoo_batch_parallel_accounting_matches_pool_widths() -> anyhow::Result<()> {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let pool1_before = pool_worker_tls_snapshots(&pool1);
    let pool4_before = pool_worker_tls_snapshots(&pool4);
    let serial = pool1.install(|| run_direct_oxymoo_batch_with_accounting(false))?;
    let parallel_one = pool1.install(|| run_direct_oxymoo_batch_with_accounting(true))?;
    let parallel = pool4.install(|| run_direct_oxymoo_batch_with_accounting(true))?;
    assert_pool_worker_tls_restored(
        &pool1_before,
        &pool_worker_tls_snapshots(&pool1),
        "OxyMOO pool-1",
    )?;
    assert_pool_worker_tls_restored(
        &pool4_before,
        &pool_worker_tls_snapshots(&pool4),
        "OxyMOO pool-4",
    )?;

    assert_bits_eq(&serial.0, &parallel_one.0, "OxyMOO pool-1 objective rows")?;
    assert_bits_eq(&serial.1, &parallel_one.1, "OxyMOO pool-1 constraint rows")?;
    anyhow::ensure!(
        serial.2 == parallel_one.2,
        "OxyMOO work counters differ: serial={:?} pool-1={:?}",
        serial.2,
        parallel_one.2
    );
    assert_evaluation_diagnostic_accounting_matches(&serial.3, &parallel_one.3, "OxyMOO pool-1")?;

    assert_bits_eq(&serial.0, &parallel.0, "OxyMOO pool-4 objective rows")?;
    assert_bits_eq(&serial.1, &parallel.1, "OxyMOO pool-4 constraint rows")?;
    anyhow::ensure!(
        serial.2 == parallel.2,
        "OxyMOO work counters differ: serial={:?} pool-4={:?}",
        serial.2,
        parallel.2
    );
    assert_evaluation_diagnostic_accounting_matches(&serial.3, &parallel.3, "OxyMOO pool-4")?;
    anyhow::ensure!(
        serial.2.plan_full_evaluations > 0,
        "OxyMOO accounting fixture must execute at least one full plan"
    );
    Ok(())
}

/// Reinterpret a signed branch/revolution count exactly as the historical
/// sign-extending cast chain did, without a lossy `as` conversion.
fn i32_bit_pattern_as_u64(value: i32) -> u64 {
    u64::from_le_bytes(i64::from(value).to_le_bytes())
}

fn plan_bit_signature(p: &PlanResult) -> Vec<u64> {
    let mut sig = Vec::with_capacity(96);
    for value in [
        p.cost,
        p.time2phase_ratio,
        p.phase_sma_ratio,
        p.waittime_ratio,
        p.time2phase,
        p.waittime,
        p.tof,
        p.distance,
        p.deployer_distance,
        p.phase_sma,
        p.phase_dv[0],
        p.phase_dv[1],
        p.phase_dv[2],
        p.transfer_dv[0],
        p.transfer_dv[1],
        p.transfer_dv[2],
        p.arrival_dv[0],
        p.arrival_dv[1],
        p.arrival_dv[2],
        p.phase_dv_norm,
        p.transfer_dv_norm,
        p.arrival_dv_norm,
    ] {
        sig.push(value.to_bits());
    }
    for state in [
        &p.payload_intercept_state,
        &p.target_intercept_state,
        &p.deployer_intercept_state,
        &p.release_state,
    ] {
        for value in state {
            sig.push(value.to_bits());
        }
    }
    sig.push(u64::from(p.valid));
    sig.push(p.polish_steps);
    sig.push(p.polish_evals);
    sig.push(p.polish_time_us);
    sig.push(u64::from(p.polish_skipped));
    sig.push(u64::from(p.escape_triggered));
    sig.push(i32_bit_pattern_as_u64(p.best_M));
    sig.push(u64::from(p.prograde));
    sig.push(i32_bit_pattern_as_u64(p.branch_rev));
    sig.push(u64::from(p.branch_low_path));
    for value in [
        p.branch_tof_s,
        p.branch_departure_dv,
        p.branch_arrival_dv,
        p.branch_total_dv,
        p.intercept_jd,
        p.waittime_jd_start,
        p.tof_jd_start,
    ] {
        sig.push(value.to_bits());
    }
    sig.push(u64::from(p.branch_status.as_code()));
    sig.push(u64::from(p.branch_rejection.as_code()));
    sig.push(u64::from(p.timing_failure_reason.as_code()));
    sig.push(p.func_evals);
    sig.push(p.optimizer_func_evals);
    sig.push(u64::from(p.optimizer_converged));
    sig.push(u64::from(p.warm_start_used));
    sig.push(p.dep_period.to_bits());
    sig.push(u64::from(p.j2_iteration_count));
    sig.push(p.j2_endpoint_residual_m.to_bits());
    sig.push(p.post_hf_endpoint_residual_m.to_bits());
    for value in p.replay_provenance.launch_pre_impulse_state {
        sig.push(value.to_bits());
    }
    for value in [
        p.replay_provenance.base_epoch_jd,
        p.replay_provenance.max_time_s,
        p.replay_provenance.max_phase_dv,
        p.replay_provenance.max_transfer_dv,
        p.replay_provenance.revolution_cap,
        p.replay_provenance.min_perigee,
        p.replay_provenance.max_apogee,
        p.replay_provenance.distance_tol,
        p.replay_provenance.deployer_min_distance,
    ] {
        sig.push(value.to_bits());
    }
    sig.push(i32_bit_pattern_as_u64(p.replay_provenance.max_revs));
    sig.push(u64::from(p.replay_provenance.target_propagation_mode));
    for value in [
        p.replay_provenance.target_am_ratio,
        p.replay_provenance.target_cd,
        p.replay_provenance.target_cr,
    ] {
        sig.push(value.to_bits());
    }
    sig
}

#[test]
fn plan_bit_signature_covers_full_plan_result() -> anyhow::Result<()> {
    let baseline = PlanResult::invalid();
    let baseline_signature = plan_bit_signature(&baseline);
    let mut variants = Vec::new();

    let mut polished = baseline.clone();
    polished.polish_steps = 1;
    variants.push(("polish", polished));

    let mut payload_state = baseline.clone();
    payload_state.payload_intercept_state[0] = 1.0;
    variants.push(("payload state", payload_state));

    let mut branch = baseline.clone();
    branch.prograde = false;
    branch.branch_departure_dv = 1.0;
    branch.branch_status = crate::types::BranchStatusToken::Accepted;
    variants.push(("branch", branch));

    let mut timestamps = baseline.clone();
    timestamps.intercept_jd = 1.0;
    timestamps.timing_failure_reason = crate::types::TimingFailureToken::InterceptInsufficientLead;
    variants.push(("timestamps", timestamps));

    let mut j2 = baseline.clone();
    j2.dep_period = 1.0;
    j2.j2_iteration_count = 1;
    j2.post_hf_endpoint_residual_m = 1.0;
    variants.push(("J2", j2));

    let mut replay = baseline;
    replay.replay_provenance.launch_pre_impulse_state[0] = 1.0;
    replay.replay_provenance.base_epoch_jd = 1.0;
    replay.replay_provenance.max_revs = 1;
    replay.replay_provenance.target_propagation_mode = 1;
    variants.push(("replay provenance", replay));

    for (label, variant) in variants {
        anyhow::ensure!(
            plan_bit_signature(&variant) != baseline_signature,
            "plan bit signature omits {label} fields"
        );
    }
    Ok(())
}

#[test]
fn oxymoo_population_is_bit_identical_across_nested_pool_widths() -> anyhow::Result<()> {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let serial = pool1.install(run_moo_population)?;
    let parallel = pool4.install(run_moo_population)?;

    anyhow::ensure!(serial.generations == parallel.generations, "generations");
    anyhow::ensure!(serial.evaluations == parallel.evaluations, "evaluations");
    anyhow::ensure!(
        serial.fronts == parallel.fronts,
        "front indices/order must match"
    );
    anyhow::ensure!(
        serial.population.len() == parallel.population.len(),
        "population size"
    );
    assert_bits_eq(
        &serial.population.decisions,
        &parallel.population.decisions,
        "population.decisions",
    )?;
    assert_bits_eq(
        &serial.population.objectives,
        &parallel.population.objectives,
        "population.objectives",
    )?;
    assert_bits_eq(
        &serial.population.constraint_violations,
        &parallel.population.constraint_violations,
        "population.constraint_violations",
    )?;
    Ok(())
}

#[test]
fn oxymoo_front_solve_is_bit_identical_across_nested_pool_widths() -> anyhow::Result<()> {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let serial = pool1.install(run_audit_front_solve)?;
    let parallel = pool4.install(run_audit_front_solve)?;

    // Both calls are nested rayon-worker calls, so both stay leaf-serial.
    anyhow::ensure!(
        serial.verified_superset_metrics.oxymoo_parallel_batch_count == 0,
        "1-thread pool must take the serial reference path"
    );
    anyhow::ensure!(
        parallel
            .verified_superset_metrics
            .oxymoo_parallel_batch_count
            == 0,
        "nested 4-thread call must keep OxyMOO leaf-serial"
    );

    // Logical eval total invariant (population x (1 init + generations) = 168)
    // holds on both paths.
    for m in [
        &serial.verified_superset_metrics,
        &parallel.verified_superset_metrics,
    ] {
        anyhow::ensure!(
            m.oxymoo_full_eval_count + m.oxymoo_eval_cache_hit_count == 168,
            "logical OxyMOO eval total must be 168 (full={} + hit={})",
            m.oxymoo_full_eval_count,
            m.oxymoo_eval_cache_hit_count
        );
    }
    // The final verified front's PlanResults must be bit-identical, in order.
    anyhow::ensure!(
        serial.candidates.len() == parallel.candidates.len(),
        "verified front size must match"
    );
    for (i, (a, b)) in serial
        .candidates
        .iter()
        .zip(parallel.candidates.iter())
        .enumerate()
    {
        anyhow::ensure!(
            plan_bit_signature(a) == plan_bit_signature(b),
            "verified front candidate {i} differs between serial and parallel"
        );
    }
    Ok(())
}

// ====================================================================
// 7.4 branch-expansion identity proof: the parallel Lambert branch
// expansion must be bit-for-bit identical to the serial reference. A
// branch-plan signature extends `plan_bit_signature` with the branch
// discriminators (`branch_rev`, `branch_low_path`, `branch_tof_s`).
// ====================================================================
fn branch_plan_bit_signature(p: &PlanResult) -> Vec<u64> {
    let mut sig = plan_bit_signature(p);
    sig.push(i32_bit_pattern_as_u64(p.branch_rev));
    sig.push(u64::from(p.branch_low_path));
    sig.push(p.branch_tof_s.to_bits());
    sig
}

#[test]
fn branch_front_is_bit_identical_across_nested_pool_widths() -> anyhow::Result<()> {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let serial = pool1.install(run_audit_front_solve)?;
    let parallel = pool4.install(run_audit_front_solve)?;

    // Both calls are nested rayon-worker calls, so both stay leaf-serial.
    anyhow::ensure!(
        serial.verified_superset_metrics.branch_parallel_count == 0,
        "1-thread pool must take the serial branch-expansion path"
    );
    anyhow::ensure!(
        parallel.verified_superset_metrics.branch_parallel_count == 0,
        "nested 4-thread call must keep branch expansion serial (parallel_count={})",
        parallel.verified_superset_metrics.branch_parallel_count
    );

    // Every branch-stage integer counter is invariant across the two paths:
    // the deterministic serial reduction reproduces the serial totals exactly.
    let s = &serial.verified_superset_metrics;
    let p = &parallel.verified_superset_metrics;
    anyhow::ensure!(
        s.branch_source_count == p.branch_source_count,
        "branch_source_count"
    );
    anyhow::ensure!(
        s.branch_full_eval_count == p.branch_full_eval_count,
        "branch_full_eval_count"
    );
    anyhow::ensure!(
        s.branch_shared_prepare_count == p.branch_shared_prepare_count,
        "branch_shared_prepare_count"
    );
    anyhow::ensure!(
        s.branch_eval_call_count == p.branch_eval_call_count,
        "branch_eval_call_count"
    );
    anyhow::ensure!(
        s.branch_emitted_count == p.branch_emitted_count,
        "branch_emitted_count"
    );
    anyhow::ensure!(
        s.branch_rejected_count == p.branch_rejected_count,
        "branch_rejected_count"
    );
    anyhow::ensure!(
        s.branch_brent_call_count == p.branch_brent_call_count,
        "branch_brent_call_count"
    );
    anyhow::ensure!(
        s.branch_j2_correction_call_count == p.branch_j2_correction_call_count,
        "branch_j2_correction_call_count"
    );
    anyhow::ensure!(
        s.post_branch_candidate_count == p.post_branch_candidate_count,
        "post_branch_candidate_count"
    );
    anyhow::ensure!(
        (
            s.branch_rows_per_source_p50,
            s.branch_rows_per_source_p95,
            s.branch_rows_per_source_max
        ) == (
            p.branch_rows_per_source_p50,
            p.branch_rows_per_source_p95,
            p.branch_rows_per_source_max
        ),
        "branch_rows_per_source percentiles"
    );

    // The final verified front's PlanResults must be bit-identical, in order.
    anyhow::ensure!(
        serial.candidates.len() == parallel.candidates.len(),
        "verified front size must match"
    );
    for (i, (a, b)) in serial
        .candidates
        .iter()
        .zip(parallel.candidates.iter())
        .enumerate()
    {
        anyhow::ensure!(
            branch_plan_bit_signature(a) == branch_plan_bit_signature(b),
            "verified front candidate {i} differs between serial and parallel"
        );
    }
    Ok(())
}

#[test]
fn branch_expansion_stage_parallel_matches_serial_bit_identical() -> anyhow::Result<()> {
    // Multi-rev fixture with several distinct branch sources. Direct helper
    // calls isolate algorithm identity from adaptive dispatch policy.
    let build_ctx = || -> anyhow::Result<PlanContext> {
        let mut ctx = make_leo_ctx()?;
        anyhow::ensure!(ctx.max_revs > 0, "fixture must be multi-rev");
        prepare_single_pair_context(&mut ctx);
        anyhow::ensure!(ctx.execution_policy.allow_branch_expansion_parallel);
        Ok(ctx)
    };
    let candidates: Vec<PlanResult> = [
        [0.30, 1.00, 0.20],
        [0.40, 1.10, 0.25],
        [0.50, 0.90, 0.30],
        [0.45, 1.05, 0.28],
        [0.35, 0.95, 0.22],
        [0.55, 1.15, 0.18],
    ]
    .iter()
    .map(|ratios| synthetic_transfer_candidate(0.10, 3600.0, 4.0, *ratios))
    .collect();

    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;

    let parallel_candidates = candidates.clone();
    let ctx_serial = build_ctx()?;
    let mut serial_metrics = VerifiedSupersetStageMetrics::default();
    let serial_sources = branch_expansion_sources_unique_by_repaired_decision_indexed(candidates)?;
    serial_metrics.branch_source_count = serial_sources.len();
    let (serial_expanded, _serial_s) = pool1.install(|| {
        expand_lambert_branch_candidates_serial(&ctx_serial, serial_sources, &mut serial_metrics)
    })?;

    let ctx_parallel = build_ctx()?;
    let mut parallel_metrics = VerifiedSupersetStageMetrics::default();
    let parallel_sources =
        branch_expansion_sources_unique_by_repaired_decision_indexed(parallel_candidates)?;
    parallel_metrics.branch_source_count = parallel_sources.len();
    let (parallel_expanded, _parallel_s) = pool4.install(|| {
        expand_lambert_branch_candidates_parallel(
            &ctx_parallel,
            &parallel_sources,
            &mut parallel_metrics,
        )
    })?;

    // The direct helpers genuinely took the two algorithm paths.
    anyhow::ensure!(
        serial_metrics.branch_parallel_count == 0,
        "1-thread pool must take the serial branch-expansion path"
    );
    anyhow::ensure!(
        parallel_metrics.branch_parallel_count > 1,
        "4-thread pool must fan the branch expansion out (parallel_count={})",
        parallel_metrics.branch_parallel_count
    );
    anyhow::ensure!(
        parallel_metrics.branch_parallel_count == parallel_metrics.branch_source_count,
        "each expanded source classified once"
    );

    // Percentile metrics match across the two paths.
    anyhow::ensure!(
        (
            serial_metrics.branch_rows_per_source_p50,
            serial_metrics.branch_rows_per_source_p95,
            serial_metrics.branch_rows_per_source_max
        ) == (
            parallel_metrics.branch_rows_per_source_p50,
            parallel_metrics.branch_rows_per_source_p95,
            parallel_metrics.branch_rows_per_source_max
        ),
        "branch_rows_per_source percentiles differ"
    );

    // The expanded candidate Vec must be bit-identical, in order.
    anyhow::ensure!(
        serial_expanded.len() == parallel_expanded.len(),
        "expanded candidate count differs: serial={} parallel={}",
        serial_expanded.len(),
        parallel_expanded.len()
    );
    anyhow::ensure!(
        !serial_expanded.is_empty(),
        "fixture should expand at least one candidate"
    );
    for (i, (a, b)) in serial_expanded
        .iter()
        .zip(parallel_expanded.iter())
        .enumerate()
    {
        anyhow::ensure!(
            branch_plan_bit_signature(a) == branch_plan_bit_signature(b),
            "expanded candidate {i} differs between serial and parallel"
        );
    }
    Ok(())
}
// ====================================================================
// Delta-V polish identity proofs. Direct helper calls validate parallel
// algorithm identity; custom-pool front solves validate nested serialization.
// ====================================================================

/// Extended bit signature covering every polished field, including the
/// polish/optimizer metadata the polish stage writes, so a serial-vs-parallel
/// mismatch in ANY field (not just physics) is caught.
fn polished_candidate_signature(p: &PlanResult) -> Vec<u64> {
    let mut sig = plan_bit_signature(p);
    sig.push(p.func_evals);
    sig.push(p.optimizer_func_evals);
    sig.push(u64::from(p.optimizer_converged));
    sig.push(u64::from(p.warm_start_used));
    sig.push(p.polish_steps);
    sig.push(p.polish_evals);
    sig.push(p.polish_time_us);
    sig.push(u64::from(p.polish_skipped));
    sig.push(u64::from(p.escape_triggered));
    sig
}

/// Deterministic multi-candidate polish fixture: rank the LEO front seeds
/// and keep the finite ones. Built once and cloned so the serial and
/// parallel polish runs receive byte-identical input candidates.
fn build_polish_stage_fixture() -> anyhow::Result<(PlanContext, Vec<PlanResult>)> {
    let mut ctx = make_leo_ctx()?;
    ctx.sampling_mode = SamplingMode::Fast;
    prepare_single_pair_context(&mut ctx);
    anyhow::ensure!(
        ctx.execution_policy.allow_polish_parallel,
        "single-pair context must opt polish into the fan-out"
    );
    let cache = RefCell::new(SolveLocalWorkCache::new());
    let (ranked_seeds, _warm_start_consumed, _seed_timing) =
        rank_seed_candidates_for_front(&ctx, None, &cache)?;
    let candidates: Vec<PlanResult> = ranked_seeds
        .iter()
        .filter_map(|(_, plan)| {
            transfer_candidate_is_objective_finite(plan).then_some(plan.clone())
        })
        .take(12)
        .collect();
    anyhow::ensure!(
        candidates.len() >= 2,
        "fixture needs >= 2 finite polish candidates, got {}",
        candidates.len()
    );
    Ok((ctx, candidates))
}

fn run_polish_stage(
    ctx: &PlanContext,
    input: &[PlanResult],
) -> anyhow::Result<(Vec<PlanResult>, PolishScopeStats)> {
    let mut candidates = input.to_vec();
    let cache = RefCell::new(SolveLocalWorkCache::new());
    let stats =
        polish_transfer_candidates_delta_v(&mut candidates, ctx, &cache, PolishScopePolicy::Full);
    Ok((candidates, stats?))
}

fn run_direct_polish_parallel_with_accounting(
    ctx: &PlanContext,
    input: &[PlanResult],
) -> anyhow::Result<(
    Vec<PlanResult>,
    PolishScopeStats,
    WorkCountCounters,
    EvaluationDiagnosticCounters,
)> {
    let work_before = work_count_snapshot();
    let diagnostics_before = evaluation_diagnostic_snapshot();
    let outcome = (|| {
        let mut candidates = input.to_vec();
        let actions = vec![PolishAction::Polish; candidates.len()];
        let mut stats = PolishScopeStats {
            scope_skipped_count: 0,
            dv_improvement_max_km_s: 0.0,
            polish_parallel_count: 0,
        };
        polish_candidates_parallel(&mut candidates, &actions, ctx, &mut stats)?;
        let work_delta = work_count_snapshot().delta_since(work_before)?;
        let diagnostics_delta = evaluation_diagnostic_snapshot().delta_since(diagnostics_before)?;
        Ok((candidates, stats, work_delta, diagnostics_delta))
    })();
    restore_work_count_snapshot(work_before);
    restore_evaluation_diagnostics(&diagnostics_before);
    outcome
}

/// Direct pool-1 vs pool-N proof for the parallel polish reducer. A pool-1
/// worker necessarily executes its own Rayon work, exposing a missing TLS
/// restoration as doubled diagnostics or work counts.
#[test]
fn polish_parallel_accounting_matches_pool_widths() -> anyhow::Result<()> {
    let (ctx, input) = build_polish_stage_fixture()?;
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let pool1_before = pool_worker_tls_snapshots(&pool1);
    let pool4_before = pool_worker_tls_snapshots(&pool4);
    let one = pool1.install(|| run_direct_polish_parallel_with_accounting(&ctx, &input))?;
    let four = pool4.install(|| run_direct_polish_parallel_with_accounting(&ctx, &input))?;
    assert_pool_worker_tls_restored(
        &pool1_before,
        &pool_worker_tls_snapshots(&pool1),
        "polish pool-1",
    )?;
    assert_pool_worker_tls_restored(
        &pool4_before,
        &pool_worker_tls_snapshots(&pool4),
        "polish pool-4",
    )?;

    anyhow::ensure!(one.0.len() == four.0.len(), "polish output length differs");
    for (index, (left, right)) in one.0.iter().zip(four.0.iter()).enumerate() {
        anyhow::ensure!(
            polished_candidate_signature(left) == polished_candidate_signature(right),
            "polish candidate {index} differs between pool widths"
        );
    }
    anyhow::ensure!(
        one.1.scope_skipped_count == four.1.scope_skipped_count
            && one.1.dv_improvement_max_km_s.to_bits() == four.1.dv_improvement_max_km_s.to_bits()
            && one.1.polish_parallel_count == four.1.polish_parallel_count,
        "polish stage accounting differs between pool widths"
    );
    anyhow::ensure!(
        one.1.polish_parallel_count == one.0.len() && one.1.polish_parallel_count > 0,
        "pool-1 direct parallel polish must process every fixture candidate"
    );
    anyhow::ensure!(
        one.2 == four.2,
        "polish work counters differ: pool-1={:?} pool-4={:?}",
        one.2,
        four.2
    );
    assert_evaluation_diagnostic_accounting_matches(&one.3, &four.3, "polish")?;
    anyhow::ensure!(
        one.2.plan_full_evaluations > 0,
        "polish accounting fixture must execute at least one full plan"
    );
    Ok(())
}

#[test]
fn polish_stage_parallel_matches_serial_bit_identical() -> anyhow::Result<()> {
    // Stage-focused identity: the polished candidates Vec — every float
    // field via to_bits, every status flag, IN ORDER — plus the scope-skip
    // tally must be identical between the serial and parallel polish paths.
    let (ctx, input) = build_polish_stage_fixture()?;
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let (serial, serial_stats) = pool1.install(|| run_polish_stage(&ctx, &input))?;
    let mut seen = FxHashSet::default();
    for candidate in &input {
        let key = transfer_decision_key(&repaired_transfer_decision(&transfer_plan_decision(
            candidate,
        )));
        anyhow::ensure!(seen.insert(key), "fixture decisions must be unique");
    }
    let (parallel, parallel_stats) = pool4.install(|| -> anyhow::Result<_> {
        let mut candidates = input.clone();
        let actions = vec![PolishAction::Polish; candidates.len()];
        let mut stats = PolishScopeStats {
            scope_skipped_count: 0,
            dv_improvement_max_km_s: 0.0,
            polish_parallel_count: 0,
        };
        polish_candidates_parallel(&mut candidates, &actions, &ctx, &mut stats)?;
        Ok((candidates, stats))
    })?;

    // Direct helper bypasses adaptive gate solely to test algorithm identity.
    anyhow::ensure!(
        serial_stats.polish_parallel_count == 0,
        "1-thread pool must take the serial reference path"
    );
    anyhow::ensure!(
        parallel_stats.polish_parallel_count > 0,
        "4-thread pool must fan the polish out (parallel_count={})",
        parallel_stats.polish_parallel_count
    );

    // (c) scope_skipped_count equality.
    anyhow::ensure!(
        serial_stats.scope_skipped_count == parallel_stats.scope_skipped_count,
        "scope_skipped_count must match between serial and parallel"
    );
    // dv-improvement reduction is order-invariant and must match to the bit.
    anyhow::ensure!(
        serial_stats.dv_improvement_max_km_s.to_bits()
            == parallel_stats.dv_improvement_max_km_s.to_bits(),
        "polish dv_improvement_max_km_s must match bit-for-bit"
    );

    // (b) polished candidates Vec, in order, all fields bit-identical.
    anyhow::ensure!(
        serial.len() == parallel.len(),
        "polished candidate count must match"
    );
    for (i, (a, b)) in serial.iter().zip(parallel.iter()).enumerate() {
        anyhow::ensure!(
            polished_candidate_signature(a) == polished_candidate_signature(b),
            "polished candidate {i} differs between serial and parallel"
        );
    }
    Ok(())
}

#[test]
fn polish_front_is_bit_identical_across_nested_pool_widths() -> anyhow::Result<()> {
    // Both full solves run on rayon workers and therefore stay leaf-serial.
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let serial = pool1.install(run_audit_front_solve)?;
    let parallel = pool4.install(run_audit_front_solve)?;

    anyhow::ensure!(
        serial.verified_superset_metrics.polish_parallel_count == 0,
        "1-thread pool must take the serial polish reference path"
    );
    anyhow::ensure!(
        parallel.verified_superset_metrics.polish_parallel_count == 0,
        "nested 4-thread call must keep polish serial"
    );

    anyhow::ensure!(
        serial.candidates.len() == parallel.candidates.len(),
        "verified front size must match"
    );
    for (i, (a, b)) in serial
        .candidates
        .iter()
        .zip(parallel.candidates.iter())
        .enumerate()
    {
        anyhow::ensure!(
            polished_candidate_signature(a) == polished_candidate_signature(b),
            "front candidate {i} differs between serial and parallel polish"
        );
    }
    Ok(())
}
// ====================================================================
// Delta-v anchor identity helpers. Custom-pool calls validate nested
// serialization; isolated global-pool child test validates true fan-out.
// ====================================================================

/// Run just the delta-v anchor stage against a fresh cache and report the
/// pushed candidate Vec plus how many anchor NM runs were dispatched in
/// parallel. Zero-arg so it can be handed to `ThreadPool::install` without
/// capturing non-`Send` context; the seed ranking that produces the anchor
/// starts is deterministic and pool-invariant.
fn run_anchor_stage_candidates() -> anyhow::Result<(Vec<PlanResult>, usize)> {
    let mut ctx = make_leo_ctx()?;
    // Caches orbits AND opts the single-pair context into anchor parallelism.
    prepare_single_pair_context(&mut ctx);
    anyhow::ensure!(ctx.execution_policy.allow_anchor_parallel);
    let seed_cache = RefCell::new(SolveLocalWorkCache::new());
    let (ranked_seeds, warm_start_consumed, _timing) =
        rank_seed_candidates_for_front(&ctx, None, &seed_cache)?;
    // Fresh cache so serial and parallel start from the same (empty) state;
    // plan values are cache-neutral regardless.
    let anchor_cache = RefCell::new(SolveLocalWorkCache::new());
    let mut out = Vec::new();
    reset_anchor_parallel_count();
    push_delta_v_anchor_candidates(
        &mut out,
        &ctx,
        &ranked_seeds,
        warm_start_consumed,
        &anchor_cache,
        DeltaVAnchorPolicy::Full,
    )?;
    Ok((out, anchor_parallel_count_snapshot()))
}

#[test]
fn anchor_stage_is_bit_identical_across_nested_pool_widths() -> anyhow::Result<()> {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let (serial, serial_parallel) = pool1.install(run_anchor_stage_candidates)?;
    let (parallel, parallel_parallel) = pool4.install(run_anchor_stage_candidates)?;

    // Both calls run on rayon workers and therefore stay leaf-serial.
    anyhow::ensure!(
        serial_parallel == 0,
        "1-thread pool must take the serial anchor path"
    );
    anyhow::ensure!(
        parallel_parallel == 0,
        "nested 4-thread call must keep anchors serial"
    );

    anyhow::ensure!(!serial.is_empty(), "expected anchor candidates");
    anyhow::ensure!(
        serial.len() == parallel.len(),
        "anchor candidate count must match: serial={} parallel={}",
        serial.len(),
        parallel.len()
    );
    for (i, (a, b)) in serial.iter().zip(parallel.iter()).enumerate() {
        anyhow::ensure!(
            plan_bit_signature(a) == plan_bit_signature(b),
            "anchor candidate {i} differs between serial and parallel"
        );
    }
    Ok(())
}

#[test]
fn anchor_front_is_bit_identical_across_nested_pool_widths() -> anyhow::Result<()> {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let serial = pool1.install(run_audit_front_solve)?;
    let parallel = pool4.install(run_audit_front_solve)?;

    let sm = &serial.verified_superset_metrics;
    let pm = &parallel.verified_superset_metrics;

    // Both calls run on rayon workers and therefore stay leaf-serial.
    anyhow::ensure!(
        sm.anchor_parallel_count == 0,
        "1-thread pool must run anchors serially"
    );
    anyhow::ensure!(
        pm.anchor_parallel_count == 0,
        "nested 4-thread call must keep anchors serial"
    );

    // Cache-independent anchor work is identical on both paths (the NM run /
    // iteration / probe-eval counts do not depend on cache warming).
    anyhow::ensure!(
        sm.anchor_nm_run_count == pm.anchor_nm_run_count,
        "anchor NM run count must match"
    );
    anyhow::ensure!(
        sm.anchor_nm_iteration_count == pm.anchor_nm_iteration_count,
        "anchor NM iteration count must match"
    );
    anyhow::ensure!(
        sm.anchor_probe_eval_count == pm.anchor_probe_eval_count,
        "anchor probe eval count must match"
    );

    // The final verified front's PlanResults must be bit-identical, in order.
    anyhow::ensure!(
        serial.candidates.len() == parallel.candidates.len(),
        "verified front size must match"
    );
    for (i, (a, b)) in serial
        .candidates
        .iter()
        .zip(parallel.candidates.iter())
        .enumerate()
    {
        anyhow::ensure!(
            plan_bit_signature(a) == plan_bit_signature(b),
            "verified front candidate {i} differs between serial and parallel"
        );
    }
    Ok(())
}

const TOP_LEVEL_LEAF_FANOUT_CHILD_ENV: &str = "NASA_DUST_TOP_LEVEL_LEAF_FANOUT_CHILD";
const TOP_LEVEL_LEAF_FANOUT_CHILD_MARKER: &str = "NASA_DUST_TOP_LEVEL_LEAF_FANOUT_CHILD_RAN";
const TOP_LEVEL_LEAF_FANOUT_TEST: &str = "solve::tests::top_level_global_pool_engages_leaf_fanouts";

#[test]
fn top_level_global_pool_engages_leaf_fanouts() -> anyhow::Result<()> {
    if std::env::var_os(TOP_LEVEL_LEAF_FANOUT_CHILD_ENV).is_none() {
        let output = std::process::Command::new(std::env::current_exe()?)
            .arg(TOP_LEVEL_LEAF_FANOUT_TEST)
            .arg("--exact")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(TOP_LEVEL_LEAF_FANOUT_CHILD_ENV, "4")
            .output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::ensure!(
            output.status.success(),
            "isolated child failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        anyhow::ensure!(
            stdout.contains(TOP_LEVEL_LEAF_FANOUT_CHILD_MARKER)
                || stderr.contains(TOP_LEVEL_LEAF_FANOUT_CHILD_MARKER),
            "isolated child marker missing\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        return Ok(());
    }

    let width = std::env::var(TOP_LEVEL_LEAF_FANOUT_CHILD_ENV)?.parse::<usize>()?;
    anyhow::ensure!(nd_sched::init_global_pool(Some(width)) == width);
    println!("{TOP_LEVEL_LEAF_FANOUT_CHILD_MARKER}");
    anyhow::ensure!(rayon::current_num_threads() == width);
    anyhow::ensure!(rayon::current_thread_index().is_none());

    let serial_pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let serial = serial_pool.install(run_audit_front_solve)?;
    let parallel = run_audit_front_solve()?;
    let sm = &serial.verified_superset_metrics;
    let pm = &parallel.verified_superset_metrics;

    anyhow::ensure!(sm.oxymoo_parallel_batch_count == 0);
    anyhow::ensure!(pm.oxymoo_parallel_batch_count > 0);
    anyhow::ensure!(sm.anchor_parallel_count == 0);
    anyhow::ensure!(pm.anchor_parallel_count > 0);
    anyhow::ensure!(pm.anchor_parallel_count == pm.anchor_nm_run_count);
    anyhow::ensure!(sm.branch_parallel_count == 0);
    anyhow::ensure!(pm.branch_parallel_count > 1);
    anyhow::ensure!(sm.polish_parallel_count == 0);
    anyhow::ensure!(pm.polish_parallel_count > 0);

    anyhow::ensure!(sm.anchor_nm_run_count == pm.anchor_nm_run_count);
    anyhow::ensure!(sm.anchor_nm_iteration_count == pm.anchor_nm_iteration_count);
    anyhow::ensure!(sm.anchor_probe_eval_count == pm.anchor_probe_eval_count);
    anyhow::ensure!(sm.branch_source_count == pm.branch_source_count);
    anyhow::ensure!(sm.branch_full_eval_count == pm.branch_full_eval_count);
    anyhow::ensure!(sm.branch_eval_call_count == pm.branch_eval_call_count);
    anyhow::ensure!(sm.branch_emitted_count == pm.branch_emitted_count);
    anyhow::ensure!(sm.branch_rejected_count == pm.branch_rejected_count);
    anyhow::ensure!(sm.post_branch_candidate_count == pm.post_branch_candidate_count);
    anyhow::ensure!(
        sm.oxymoo_full_eval_count + sm.oxymoo_eval_cache_hit_count
            == pm.oxymoo_full_eval_count + pm.oxymoo_eval_cache_hit_count
    );

    anyhow::ensure!(serial.candidates.len() == parallel.candidates.len());
    for (index, (left, right)) in serial
        .candidates
        .iter()
        .zip(parallel.candidates.iter())
        .enumerate()
    {
        anyhow::ensure!(
            polished_candidate_signature(left) == polished_candidate_signature(right),
            "full-front polished candidate {index} differs"
        );
        anyhow::ensure!(
            branch_plan_bit_signature(left) == branch_plan_bit_signature(right),
            "full-front branch candidate {index} differs"
        );
    }

    let (serial_anchors, serial_anchor_parallel_count) =
        serial_pool.install(run_anchor_stage_candidates)?;
    let (parallel_anchors, parallel_anchor_parallel_count) = run_anchor_stage_candidates()?;
    anyhow::ensure!(serial_anchor_parallel_count == 0);
    anyhow::ensure!(parallel_anchor_parallel_count > 0);
    anyhow::ensure!(serial_anchors.len() == parallel_anchors.len());
    for (index, (left, right)) in serial_anchors
        .iter()
        .zip(parallel_anchors.iter())
        .enumerate()
    {
        anyhow::ensure!(
            plan_bit_signature(left) == plan_bit_signature(right),
            "anchor candidate {index} differs"
        );
    }

    let serial_grid = serial_pool.install(|| {
        let mut ctx = make_leo_ctx()?;
        verified_superset_deterministic_grid_fallback(&mut ctx, false)
    })?;
    reset_deterministic_grid_parallel_path_hits();
    let mut parallel_grid_ctx = make_leo_ctx()?;
    let parallel_grid =
        verified_superset_deterministic_grid_fallback(&mut parallel_grid_ctx, false)?;
    anyhow::ensure!(deterministic_grid_parallel_path_hits() > 0);
    anyhow::ensure!(serial_grid.candidates.len() == parallel_grid.candidates.len());
    for (index, (left, right)) in serial_grid
        .candidates
        .iter()
        .zip(parallel_grid.candidates.iter())
        .enumerate()
    {
        anyhow::ensure!(
            branch_plan_bit_signature(left) == branch_plan_bit_signature(right),
            "grid candidate {index} differs"
        );
    }
    Ok(())
}

#[test]
fn oxymoo_full_generation_eval_count_is_168() -> anyhow::Result<()> {
    // GREEN pin: OxyMOO Full policy runs population(28) x (1 init + 5 gens)
    // = 168 objective evaluations. Every `Problem::evaluate` call is either
    // an eval-cache hit or a miss (a full eval), so full + hit == 168 and
    // the serial batch count is generations + 1 = 6.
    let m = audit_work_count_metrics()?;
    anyhow::ensure!(
        m.oxymoo_full_eval_count + m.oxymoo_eval_cache_hit_count == 168,
        "OxyMOO Full policy generation evals: full={} + hit={} should be 168",
        m.oxymoo_full_eval_count,
        m.oxymoo_eval_cache_hit_count
    );
    anyhow::ensure!(
        m.oxymoo_eval_cache_miss_count == m.oxymoo_full_eval_count,
        "every OxyMOO eval-cache miss forces exactly one full eval"
    );
    anyhow::ensure!(
        m.oxymoo_serial_batch_count == 6,
        "OxyMOO Full policy runs 1 init + 5 generation batches serially"
    );
    anyhow::ensure!(
        m.oxymoo_parallel_batch_count == 0,
        "one-thread serial reference must not fan out OxyMOO"
    );
    Ok(())
}

#[test]
fn anchor_and_polish_stage_full_evals_are_recorded() -> anyhow::Result<()> {
    // GREEN pin: the per-stage full-plan-eval attribution is live and the
    // anchor NM/probe work is counted (serial: parallel counters are 0).
    let m = audit_work_count_metrics()?;
    anyhow::ensure!(
        m.anchor_full_eval_count > 0,
        "anchor stage should record full plan evals"
    );
    anyhow::ensure!(
        m.polish_full_eval_count > 0,
        "polish stage should record full plan evals"
    );
    anyhow::ensure!(
        m.anchor_nm_run_count > 0 && m.anchor_nm_iteration_count > 0,
        "anchor NM runs and iterations should be recorded"
    );
    anyhow::ensure!(
        m.anchor_probe_eval_count > 0,
        "anchor probe evals should be recorded"
    );
    anyhow::ensure!(m.anchor_parallel_count == 0, "anchor is serial today");
    anyhow::ensure!(m.polish_parallel_count == 0, "polish is serial today");
    anyhow::ensure!(m.branch_parallel_count == 0, "branch is serial today");
    Ok(())
}

#[test]
fn branch_subtimers_are_bounded_by_branch_eval() -> anyhow::Result<()> {
    // GREEN pin. The task hypothesized branch_brent_s + branch_j2_s would
    // dominate (>= 0.5 x) branch_eval_s on a multi-rev fixture. Measured
    // empirically the Brent + J2-correction sub-phases are a MINORITY of
    // branch eval time (~0.27x on the LEO fixture), so the ">= 0.5"
    // direction does not hold — recorded here as a finding. What IS a
    // stable structural invariant (and the useful pin) is that the two
    // sub-phase timers are non-negative components bounded by the whole
    // branch eval time, on a genuine multi-rev fixture.
    let m = audit_work_count_metrics()?;
    anyhow::ensure!(
        m.branch_source_count > 1,
        "expected a multi-rev branch fixture, got {}",
        m.branch_source_count
    );
    anyhow::ensure!(m.branch_brent_s >= 0.0 && m.branch_j2_correction_s >= 0.0);
    anyhow::ensure!(
        m.branch_brent_s + m.branch_j2_correction_s <= m.branch_eval_s + 1.0e-6,
        "brent ({}) + j2 ({}) subtimers should not exceed branch_eval_s ({})",
        m.branch_brent_s,
        m.branch_j2_correction_s,
        m.branch_eval_s
    );
    Ok(())
}

#[test]
fn polish_scope_fallback_never_fires_on_standard_fixture() -> anyhow::Result<()> {
    // Deliverable #5 (fallback duplication check): the degenerate-front
    // safety net requires verified_front.len() < 2 AND scope_skipped_count
    // > 0. The standard LEO fixture uses the default (Full) polish scope
    // policy, which skips nothing, and finalizes a healthy multi-row front,
    // so the trigger shape is not constructible here. The double-work
    // counter therefore reads zero. If a future change lets scoped polish
    // starve the front, `polish_scope_fallback_full_eval_count` will start
    // reporting the duplicated polish+branch work and this guard flips.
    let m = audit_work_count_metrics()?;
    anyhow::ensure!(
        m.polish_scope_fallback_count == 0,
        "polish-scope fallback should not fire on the standard fixture"
    );
    anyhow::ensure!(
        m.polish_scope_fallback_full_eval_count == 0,
        "no duplicated fallback full evals when the fallback does not fire"
    );
    anyhow::ensure!(
        m.deterministic_fallback_full_eval_count == 0,
        "the standard fixture returns a non-empty front, no grid fallback"
    );
    Ok(())
}

fn run_direct_deterministic_grid(parallel: bool) -> anyhow::Result<TransferFront> {
    let mut ctx = make_leo_ctx()?;
    prepare_single_pair_context(&mut ctx);
    let mut grid_points = Vec::new();
    for &time2phase_ratio in SINGLE_PAIR_TIME_PTS {
        for &phase_sma_ratio in SINGLE_PAIR_PHASE_PTS {
            for &waittime_ratio in SINGLE_PAIR_WAIT_PTS {
                if time2phase_ratio + waittime_ratio <= 0.98 {
                    grid_points.push([time2phase_ratio, phase_sma_ratio, waittime_ratio]);
                }
            }
        }
    }
    let mut candidates = Vec::new();
    if parallel {
        deterministic_grid_fallback_candidates_parallel(
            &ctx,
            &grid_points,
            false,
            &mut candidates,
        )?;
    } else {
        deterministic_grid_fallback_candidates_serial(&ctx, &grid_points, false, &mut candidates)?;
    }
    Ok(finalize_verified_superset(&ctx, &mut candidates)?)
}

/// Test-only path accounting must fail atomically rather than wrap.
#[test]
fn deterministic_grid_parallel_path_hit_counter_rejects_overflow() -> anyhow::Result<()> {
    let original = deterministic_grid_parallel_path_hits();
    DETERMINISTIC_GRID_PARALLEL_PATH_HITS.with(|hits| hits.set(usize::MAX));

    let outcome = record_deterministic_grid_parallel_path_hit();
    let observed = deterministic_grid_parallel_path_hits();
    DETERMINISTIC_GRID_PARALLEL_PATH_HITS.with(|hits| hits.set(original));

    anyhow::ensure!(
        matches!(
            outcome,
            Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
        ),
        "parallel-grid hit counter overflow must stay typed"
    );
    anyhow::ensure!(
        observed == usize::MAX,
        "failed parallel-grid hit increment must not mutate the counter"
    );
    Ok(())
}

/// Direct helper identity proof for deterministic grid fallback. Runtime
/// dispatch behavior is covered separately by global/nested gate tests.
#[test]
fn deterministic_grid_fallback_parallel_matches_serial_bit_identical() -> anyhow::Result<()> {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;

    let serial = pool1.install(|| run_direct_deterministic_grid(false))?;
    let parallel = pool4.install(|| -> anyhow::Result<TransferFront> {
        reset_deterministic_grid_parallel_path_hits();
        let front = run_direct_deterministic_grid(true)?;
        anyhow::ensure!(deterministic_grid_parallel_path_hits() > 0);
        Ok(front)
    })?;

    anyhow::ensure!(
        !serial.candidates.is_empty(),
        "grid fallback fixture must emit candidates"
    );
    anyhow::ensure!(
        serial.candidates.len() == parallel.candidates.len(),
        "grid fallback front size must match across serial and parallel paths"
    );
    for (i, (a, b)) in serial
        .candidates
        .iter()
        .zip(parallel.candidates.iter())
        .enumerate()
    {
        anyhow::ensure!(
            branch_plan_bit_signature(a) == branch_plan_bit_signature(b),
            "grid fallback candidate {i} differs between serial and parallel"
        );
    }
    Ok(())
}
