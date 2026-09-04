//! `bench-internal` policy harness for the delta-V anchor and transfer-MOO
//! policy sweeps.
//!
//! Nothing here is reachable in a default build: the whole module is gated on
//! the `bench-internal` feature. It builds one fixed LEO context, runs the
//! verified-superset front under a chosen policy, and reports timings and
//! candidate counts beside the full-policy reference front.

use super::{
    build_cached_plan_context, checked_stage_metric_count_add, checked_stage_metric_count_delta,
    evaluate_plan_local, local_config, map_bench_local_optimizer_result,
    nm_max_iters_for_complexity, prepare_single_pair_context, push_delta_v_anchor_probe_candidates,
    push_unique_transfer_candidate, rank_seed_candidates_for_front, run_local_optimizer,
    select_delta_v_anchor_starts, solve_plan_oxymoo_front_internal,
    transfer_candidate_is_objective_finite, try_reserve_transfer_capacity, AnchorRunSettings,
    DeltaVAnchorBenchPolicy, DeltaVAnchorPolicy, EciBasicOrbit, ExecutionPolicy, FrontOutputMode,
    InvalidTargetPropagationAuthorityCode, J2ClosureSettings, LocalOptimizerKind,
    PairPlanContextInputs, PlanContext, PlanContextTemplate, PlanResult, RefCell, SamplingMode,
    SearchDepthPolicy, SolveLocalWorkCache, SolverSeed, StageTimer, TargetPropagationAuthority,
    TransferComplexity, TransferDeltaVProblem, TransferFront, TransferLocalOptimizerChoice,
    TransferLocalOptimizerConfig, TransferLocalProblem, TransferMooBenchPolicy, TransferMooPolicy,
    TuneLevel, MU, SINGLE_PAIR_LOWER_BOUNDS, SINGLE_PAIR_UPPER_BOUNDS,
};

#[derive(Clone, Debug)]
pub struct DeltaVAnchorPolicyBenchReport {
    pub policy: DeltaVAnchorBenchPolicy,
    pub anchor_candidate_count: usize,
    pub front_candidate_count: usize,
    pub objective_equivalent_to_full: bool,
    pub cost_anchor_s: f64,
    pub delta_v_anchor_s: f64,
    pub probe_s: f64,
    pub coarse_eval_count: u64,
    pub fine_eval_count: u64,
    pub probe_candidate_count: usize,
    pub polished_candidate_count: usize,
}

#[derive(Clone, Debug)]
pub struct TransferMooPolicyBenchReport {
    pub policy: TransferMooBenchPolicy,
    pub population_size: usize,
    pub generations: usize,
    pub nsga_eval_count: u64,
    pub front_candidate_count: usize,
    pub objective_equivalent_to_full: bool,
    pub oxymoo_s: f64,
    pub nsga_run_s: f64,
    pub nsga_materialize_s: f64,
    pub materialize_plan_cache_hit_count: usize,
    pub materialize_plan_cache_miss_count: usize,
    pub materialize_all_exact_count: usize,
    pub materialize_recompute_count: usize,
    pub pre_oxymoo_candidate_count: usize,
    pub post_oxymoo_candidate_count: usize,
    pub post_branch_candidate_count: usize,
    pub post_finalize_candidate_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct DeltaVAnchorPolicyProfile {
    cost_anchor_s: f64,
    delta_v_anchor_s: f64,
    probe_s: f64,
    coarse_eval_count: u64,
    fine_eval_count: u64,
    probe_candidate_count: usize,
    polished_candidate_count: usize,
}

fn make_leo_ctx_for_anchor_policy_bench(
) -> Result<PlanContext, crate::types::InvalidTargetPropagationAuthorityCode> {
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
    let mut ctx = build_cached_plan_context(
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
    )?;
    ctx.local_optimizer = TransferLocalOptimizerConfig {
        choice: TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
        tune: TuneLevel::Aggressive,
        seed: 42,
    };
    Ok(ctx)
}

fn anchor_profile_iteration_counts(
    max_iters: usize,
) -> Result<(usize, usize), InvalidTargetPropagationAuthorityCode> {
    let doubled = max_iters
        .checked_mul(2)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let coarse_iters = (doubled / 3).max(1);
    let fine_iters = if max_iters <= coarse_iters {
        1
    } else {
        max_iters
            .checked_sub(coarse_iters)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?
    };
    Ok((coarse_iters, fine_iters))
}

fn push_probe_candidates_profiled(
    candidates: &mut Vec<PlanResult>,
    ctx: &PlanContext,
    cache: &RefCell<SolveLocalWorkCache>,
    center: [f64; 3],
    warm_start_consumed: bool,
    profile: &mut DeltaVAnchorPolicyProfile,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let before = candidates.len();
    let start = StageTimer::start();
    push_delta_v_anchor_probe_candidates(candidates, ctx, cache, center, warm_start_consumed)?;
    profile.probe_s += start.elapsed_s();
    let added = checked_stage_metric_count_delta(candidates.len(), before)?;
    checked_stage_metric_count_add(&mut profile.probe_candidate_count, added)?;
    Ok(())
}

fn push_nm_delta_v_anchor_profiled(
    out: &mut Vec<PlanResult>,
    ctx: &PlanContext,
    local_cache: &RefCell<SolveLocalWorkCache>,
    start_x: [f64; 3],
    settings: AnchorRunSettings,
    emit_probes: bool,
    profile: &mut DeltaVAnchorPolicyProfile,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let anchor_start = StageTimer::start();
    let (coarse_iters, fine_iters) = anchor_profile_iteration_counts(settings.max_iters)?;
    let coarse_problem = TransferDeltaVProblem {
        ctx,
        cache: local_cache,
        coarse_mode: true,
    };
    let coarse = map_bench_local_optimizer_result(run_local_optimizer(
        &coarse_problem,
        SINGLE_PAIR_LOWER_BOUNDS,
        SINGLE_PAIR_UPPER_BOUNDS,
        start_x,
        local_config(
            LocalOptimizerKind::NelderMead,
            coarse_iters,
            settings.tune,
            settings.seed,
        ),
    ))
    .inspect_err(|_error| {
        profile.delta_v_anchor_s += anchor_start.elapsed_s();
    })?;
    profile.coarse_eval_count = profile
        .coarse_eval_count
        .checked_add(coarse.evaluations)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;

    let fine_problem = TransferDeltaVProblem {
        ctx,
        cache: local_cache,
        coarse_mode: false,
    };
    let fine = map_bench_local_optimizer_result(run_local_optimizer(
        &fine_problem,
        SINGLE_PAIR_LOWER_BOUNDS,
        SINGLE_PAIR_UPPER_BOUNDS,
        coarse.x,
        local_config(
            LocalOptimizerKind::NelderMead,
            fine_iters,
            settings.tune,
            settings.seed,
        ),
    ))
    .inspect_err(|_error| {
        profile.delta_v_anchor_s += anchor_start.elapsed_s();
    })?;
    profile.fine_eval_count = profile
        .fine_eval_count
        .checked_add(fine.evaluations)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    profile.delta_v_anchor_s += anchor_start.elapsed_s();

    let mut plan = evaluate_plan_local(&fine.x, ctx, false, local_cache)?;
    plan.func_evals = coarse
        .evaluations
        .checked_add(fine.evaluations)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    plan.optimizer_func_evals = plan.func_evals;
    plan.optimizer_converged = fine.converged;
    plan.warm_start_used = settings.warm_start_consumed;
    let before = out.len();
    push_unique_transfer_candidate(out, plan)?;
    let added = checked_stage_metric_count_delta(out.len(), before)?;
    checked_stage_metric_count_add(&mut profile.polished_candidate_count, added)?;
    if emit_probes {
        push_probe_candidates_profiled(
            out,
            ctx,
            local_cache,
            fine.x,
            settings.warm_start_consumed,
            profile,
        )?;
    }
    Ok(())
}

fn push_nm_cost_anchor_profiled(
    out: &mut Vec<PlanResult>,
    ctx: &PlanContext,
    local_cache: &RefCell<SolveLocalWorkCache>,
    start_x: [f64; 3],
    settings: AnchorRunSettings,
    emit_probes: bool,
    profile: &mut DeltaVAnchorPolicyProfile,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let anchor_start = StageTimer::start();
    let (coarse_iters, fine_iters) = anchor_profile_iteration_counts(settings.max_iters)?;
    let coarse_problem = TransferLocalProblem {
        ctx,
        cache: local_cache,
        coarse_mode: true,
        gradient_enabled: false,
    };
    let coarse = map_bench_local_optimizer_result(run_local_optimizer(
        &coarse_problem,
        SINGLE_PAIR_LOWER_BOUNDS,
        SINGLE_PAIR_UPPER_BOUNDS,
        start_x,
        local_config(
            LocalOptimizerKind::NelderMead,
            coarse_iters,
            settings.tune,
            settings.seed,
        ),
    ))
    .inspect_err(|_error| {
        profile.cost_anchor_s += anchor_start.elapsed_s();
    })?;
    profile.coarse_eval_count = profile
        .coarse_eval_count
        .checked_add(coarse.evaluations)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;

    let fine_problem = TransferLocalProblem {
        ctx,
        cache: local_cache,
        coarse_mode: false,
        gradient_enabled: false,
    };
    let fine = map_bench_local_optimizer_result(run_local_optimizer(
        &fine_problem,
        SINGLE_PAIR_LOWER_BOUNDS,
        SINGLE_PAIR_UPPER_BOUNDS,
        coarse.x,
        local_config(
            LocalOptimizerKind::NelderMead,
            fine_iters,
            settings.tune,
            settings.seed,
        ),
    ))
    .inspect_err(|_error| {
        profile.cost_anchor_s += anchor_start.elapsed_s();
    })?;
    profile.fine_eval_count = profile
        .fine_eval_count
        .checked_add(fine.evaluations)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    profile.cost_anchor_s += anchor_start.elapsed_s();

    let mut plan = evaluate_plan_local(&fine.x, ctx, false, local_cache)?;
    plan.func_evals = coarse
        .evaluations
        .checked_add(fine.evaluations)
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    plan.optimizer_func_evals = plan.func_evals;
    plan.optimizer_converged = fine.converged;
    plan.warm_start_used = settings.warm_start_consumed;
    let before = out.len();
    push_unique_transfer_candidate(out, plan)?;
    let added = checked_stage_metric_count_delta(out.len(), before)?;
    checked_stage_metric_count_add(&mut profile.polished_candidate_count, added)?;
    if emit_probes {
        push_probe_candidates_profiled(
            out,
            ctx,
            local_cache,
            fine.x,
            settings.warm_start_consumed,
            profile,
        )?;
    }
    Ok(())
}

fn profile_delta_v_anchor_policy(
    ctx: &PlanContext,
    ranked_seeds: &[(SolverSeed, PlanResult)],
    warm_start_consumed: bool,
    local_cache: &RefCell<SolveLocalWorkCache>,
    policy: DeltaVAnchorPolicy,
) -> Result<(usize, DeltaVAnchorPolicyProfile), InvalidTargetPropagationAuthorityCode> {
    let capacity = policy
        .seed_limit()
        .checked_add(1)
        .and_then(|count| count.checked_mul(5))
        .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
    let mut out = Vec::new();
    try_reserve_transfer_capacity(&mut out, capacity)?;
    let mut profile = DeltaVAnchorPolicyProfile::default();
    let complexity = TransferComplexity::classify_from_ctx(ctx);
    let tune = match ctx.local_optimizer.choice {
        TransferLocalOptimizerChoice::Auto => TuneLevel::Default,
        TransferLocalOptimizerChoice::Fixed(_) => ctx.local_optimizer.tune,
    };
    let max_iters = nm_max_iters_for_complexity(complexity);
    let seed = ctx.local_optimizer.seed;
    let settings = AnchorRunSettings {
        max_iters,
        tune,
        seed,
        warm_start_consumed,
    };
    let cost_anchor = ranked_seeds
        .iter()
        .find(|(_, plan)| transfer_candidate_is_objective_finite(plan))
        .map(|(seed, _)| seed.x);
    if policy.use_cost_anchor() {
        if let Some(start) = cost_anchor {
            push_nm_cost_anchor_profiled(
                &mut out,
                ctx,
                local_cache,
                start,
                settings,
                policy.use_probes(),
                &mut profile,
            )?;
        }
    }

    let starts = select_delta_v_anchor_starts(ranked_seeds, cost_anchor, policy);
    for start in starts.into_iter().flatten() {
        push_nm_delta_v_anchor_profiled(
            &mut out,
            ctx,
            local_cache,
            start.point,
            settings,
            policy.use_probes(),
            &mut profile,
        )?;
    }
    Ok((out.len(), profile))
}

fn transfer_front_objective_equivalent(left: &TransferFront, right: &TransferFront) -> bool {
    left.len() == right.len()
        && left.candidates.iter().all(|candidate| {
            let total_dv = candidate.total_dv();
            let time_per_rel = candidate.time_per_relative_velocity_s_per_km_s();
            right.candidates.iter().any(|other| {
                (other.total_dv() - total_dv).abs() <= 1.0e-9
                    && (other.time_per_relative_velocity_s_per_km_s() - time_per_rel).abs()
                        <= 1.0e-6
            })
        })
}

/// # Errors
///
/// Returns [`InvalidTargetPropagationAuthorityCode::ArithmeticOverflow`] when a
/// benchmark-stage count or reservation cannot be represented.
pub fn bench_verified_superset_leo_with_delta_v_anchor_policy(
    policy: DeltaVAnchorBenchPolicy,
) -> Result<TransferFront, crate::types::InvalidTargetPropagationAuthorityCode> {
    let mut ctx = make_leo_ctx_for_anchor_policy_bench()?;
    solve_plan_oxymoo_front_internal(
        &mut ctx,
        None,
        None,
        FrontOutputMode::VerifiedSuperset,
        None,
        policy.into(),
        TransferMooPolicy::Full,
    )
}

/// # Errors
///
/// Returns [`InvalidTargetPropagationAuthorityCode::ArithmeticOverflow`] when a
/// benchmark-stage count or reservation cannot be represented.
pub fn bench_delta_v_anchor_policy_report(
    policy: DeltaVAnchorBenchPolicy,
) -> Result<DeltaVAnchorPolicyBenchReport, crate::types::InvalidTargetPropagationAuthorityCode> {
    let mut profile_ctx = make_leo_ctx_for_anchor_policy_bench()?;
    prepare_single_pair_context(&mut profile_ctx);
    let local_cache = RefCell::new(SolveLocalWorkCache::new());
    let (ranked_seeds, warm_start_consumed, _) =
        rank_seed_candidates_for_front(&profile_ctx, None, &local_cache)?;
    let (anchor_candidate_count, profile) = profile_delta_v_anchor_policy(
        &profile_ctx,
        &ranked_seeds,
        warm_start_consumed,
        &local_cache,
        policy.into(),
    )?;

    let full_front =
        bench_verified_superset_leo_with_delta_v_anchor_policy(DeltaVAnchorBenchPolicy::Full)?;
    let policy_front = if policy == DeltaVAnchorBenchPolicy::Full {
        full_front.clone()
    } else {
        bench_verified_superset_leo_with_delta_v_anchor_policy(policy)?
    };
    Ok(DeltaVAnchorPolicyBenchReport {
        policy,
        anchor_candidate_count,
        front_candidate_count: policy_front.len(),
        objective_equivalent_to_full: transfer_front_objective_equivalent(
            &full_front,
            &policy_front,
        ),
        cost_anchor_s: profile.cost_anchor_s,
        delta_v_anchor_s: profile.delta_v_anchor_s,
        probe_s: profile.probe_s,
        coarse_eval_count: profile.coarse_eval_count,
        fine_eval_count: profile.fine_eval_count,
        probe_candidate_count: profile.probe_candidate_count,
        polished_candidate_count: profile.polished_candidate_count,
    })
}

/// # Errors
///
/// Returns [`InvalidTargetPropagationAuthorityCode::ArithmeticOverflow`] when a
/// benchmark-stage count or reservation cannot be represented.
pub fn bench_verified_superset_leo_with_transfer_moo_policy(
    policy: TransferMooBenchPolicy,
) -> Result<TransferFront, crate::types::InvalidTargetPropagationAuthorityCode> {
    let mut ctx = make_leo_ctx_for_anchor_policy_bench()?;
    solve_plan_oxymoo_front_internal(
        &mut ctx,
        None,
        None,
        FrontOutputMode::VerifiedSuperset,
        None,
        DeltaVAnchorPolicy::Full,
        policy.into(),
    )
}

/// # Errors
///
/// Returns [`InvalidTargetPropagationAuthorityCode::ArithmeticOverflow`] when a
/// benchmark-stage count or reservation cannot be represented.
pub fn bench_transfer_moo_policy_report(
    policy: TransferMooBenchPolicy,
) -> Result<TransferMooPolicyBenchReport, crate::types::InvalidTargetPropagationAuthorityCode> {
    let full_front =
        bench_verified_superset_leo_with_transfer_moo_policy(TransferMooBenchPolicy::Full)?;
    let policy_front = if policy == TransferMooBenchPolicy::Full {
        full_front.clone()
    } else {
        bench_verified_superset_leo_with_transfer_moo_policy(policy)?
    };
    let metrics = policy_front.verified_superset_metrics;
    let (population_size, generations) = TransferMooPolicy::from(policy).population_generations();
    let nsga_eval_count = policy_front
        .candidates
        .iter()
        .map(|candidate| candidate.optimizer_func_evals)
        .max()
        .unwrap_or(0);
    Ok(TransferMooPolicyBenchReport {
        policy,
        population_size,
        generations,
        nsga_eval_count,
        front_candidate_count: policy_front.len(),
        objective_equivalent_to_full: transfer_front_objective_equivalent(
            &full_front,
            &policy_front,
        ),
        oxymoo_s: metrics.oxymoo_s,
        nsga_run_s: metrics.nsga_run_s,
        nsga_materialize_s: metrics.nsga_materialize_s,
        materialize_plan_cache_hit_count: metrics.nsga_materialize_plan_cache_hit_count,
        materialize_plan_cache_miss_count: metrics.nsga_materialize_plan_cache_miss_count,
        materialize_all_exact_count: metrics.nsga_materialize_all_exact_count,
        materialize_recompute_count: metrics.nsga_materialize_recompute_count,
        pre_oxymoo_candidate_count: metrics.pre_oxymoo_candidate_count,
        post_oxymoo_candidate_count: metrics.post_oxymoo_candidate_count,
        post_branch_candidate_count: metrics.post_branch_candidate_count,
        post_finalize_candidate_count: metrics.post_finalize_candidate_count,
    })
}
