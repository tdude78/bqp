use anyhow::{anyhow, ensure, Result};
use criterion::Criterion;
use satpy_core::{eci2equinoc_impl, MU};
use std::sync::atomic::{AtomicBool, Ordering};
use two_phase_transfer_rs::{
    solve_plan, types::InvalidTargetPropagationAuthorityCode, ExecutionPolicy, PlanContext,
    SamplingMode, TransferFront, TransferLocalOptimizerChoice, TransferLocalOptimizerConfig,
    TransferRequest, INVALID_COST,
};
use two_phase_transfer_rs::{LocalOptimizerKind, TuneLevel};

static TIMED_FAILURE: AtomicBool = AtomicBool::new(false);

fn make_leo_ctx_with(choice: TransferLocalOptimizerChoice, tune: TuneLevel) -> PlanContext {
    let r = 6778.0;
    let v = (MU / r).sqrt();
    let dep_eci = [r, 0.0, 0.0, 0.0, v, 0.0];

    // Target: Higher altitude, phase difference
    let r_tgt = 6878.0;
    let v_tgt = (MU / r_tgt).sqrt();
    // 90 degrees ahead
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
        max_revs: 2,
        sampling_mode: SamplingMode::Fast,
        execution_policy: ExecutionPolicy::default(),
        distance_tol: 0.025,
        deployer_min_distance: 0.12,
        local_optimizer: TransferLocalOptimizerConfig {
            choice,
            tune,
            seed: 42,
        },
        ..TransferRequest::with_j2_closure_settings(
            two_phase_transfer_rs::J2ClosureSettings::default(),
        )
    })
}

fn make_leo_ctx() -> PlanContext {
    make_leo_ctx_with(TransferLocalOptimizerChoice::Auto, TuneLevel::Default)
}

fn solve_benchmark_plan(
    ctx: &mut PlanContext,
) -> Result<TransferFront, InvalidTargetPropagationAuthorityCode> {
    solve_plan(ctx, None)
}

fn front_cost(front: &TransferFront) -> f64 {
    front
        .candidates
        .first()
        .map_or(INVALID_COST, |candidate| candidate.cost)
}

const fn front_valid(front: &TransferFront) -> bool {
    !front.is_empty()
}

fn report_front(label: &str, front: &TransferFront) {
    println!(
        "{label} Cost: {:.6} km/s, Valid: {}",
        front_cost(front),
        front_valid(front)
    );
}

fn preflight(
    label: &str,
    choice: TransferLocalOptimizerChoice,
    tune: TuneLevel,
) -> Result<TransferFront> {
    let mut ctx = make_leo_ctx_with(choice, tune);
    let front = solve_benchmark_plan(&mut ctx)
        .map_err(|error| anyhow!("transfer planning preflight failed for {label}: {error}"))?;
    ensure!(
        !front.is_empty(),
        "transfer planning preflight empty for {label}"
    );
    Ok(front)
}

fn timed_front_len(result: Result<TransferFront, InvalidTargetPropagationAuthorityCode>) -> usize {
    match result {
        Ok(front) if !front.is_empty() => front.len(),
        Ok(_) | Err(_) => {
            TIMED_FAILURE.store(true, Ordering::Relaxed);
            0
        }
    }
}

fn bench_solve_plan(c: &mut Criterion) -> Result<()> {
    TIMED_FAILURE.store(false, Ordering::Relaxed);
    let nm_default = preflight(
        "NM-Default",
        TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
        TuneLevel::Default,
    )?;
    let nm_conservative = preflight(
        "NM-Conservative",
        TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
        TuneLevel::Conservative,
    )?;
    let nm_aggressive = preflight(
        "NM-Aggressive",
        TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
        TuneLevel::Aggressive,
    )?;
    let pso = preflight(
        "PSO",
        TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::Pso),
        TuneLevel::Default,
    )?;
    let auto = preflight(
        "Auto",
        TransferLocalOptimizerChoice::Auto,
        TuneLevel::Default,
    )?;
    report_front("NM-Default", &nm_default);
    report_front("NM-Conservative", &nm_conservative);
    report_front("NM-Aggressive", &nm_aggressive);
    report_front("PSO", &pso);
    report_front("Auto", &auto);

    {
        let mut group = c.benchmark_group("Transfer Planning");

        // ===== NELDER-MEAD TUNING VARIANTS =====
        // NM now skips polish by default (like CMA-ES), no workaround needed

        // 1a. Nelder-Mead (Default) - uses aggressive tuning as new default
        group.bench_function("NM-Default", |b| {
            b.iter(|| {
                let mut ctx = make_leo_ctx_with(
                    TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
                    TuneLevel::Default,
                );
                std::hint::black_box(timed_front_len(solve_benchmark_plan(std::hint::black_box(
                    &mut ctx,
                ))))
            });
        });

        // 1b. Nelder-Mead (Conservative)
        group.bench_function("NM-Conservative", |b| {
            b.iter(|| {
                let mut ctx = make_leo_ctx_with(
                    TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
                    TuneLevel::Conservative,
                );
                std::hint::black_box(timed_front_len(solve_benchmark_plan(std::hint::black_box(
                    &mut ctx,
                ))))
            });
        });

        // 1c. Nelder-Mead (Aggressive)
        group.bench_function("NM-Aggressive", |b| {
            b.iter(|| {
                let mut ctx = make_leo_ctx_with(
                    TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::NelderMead),
                    TuneLevel::Aggressive,
                );
                std::hint::black_box(timed_front_len(solve_benchmark_plan(std::hint::black_box(
                    &mut ctx,
                ))))
            });
        });

        // 2. PSO
        group.bench_function("PSO", |b| {
            b.iter(|| {
                let mut ctx = make_leo_ctx_with(
                    TransferLocalOptimizerChoice::Fixed(LocalOptimizerKind::Pso),
                    TuneLevel::Default,
                );
                std::hint::black_box(timed_front_len(solve_benchmark_plan(std::hint::black_box(
                    &mut ctx,
                ))))
            });
        });

        // 2b. Auto policy.
        group.bench_function("Auto", |b| {
            b.iter(|| {
                let mut ctx = make_leo_ctx();
                std::hint::black_box(timed_front_len(solve_benchmark_plan(std::hint::black_box(
                    &mut ctx,
                ))))
            });
        });

        group.finish();
    }
    ensure!(
        !TIMED_FAILURE.load(Ordering::Relaxed),
        "transfer planning benchmark produced an error or empty front"
    );
    Ok(())
}

fn main() -> Result<()> {
    {
        let mut criterion = Criterion::default().configure_from_args();
        bench_solve_plan(&mut criterion)?;
        criterion.final_summary();
    }
    Ok(())
}
