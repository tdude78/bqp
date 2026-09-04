use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use satpy_core::{eci2equinoc_impl, MU};
use two_phase_transfer_rs::{
    solve_plan, ExecutionPolicy, PlanContext, SamplingMode, TransferFront, TransferRequest,
};

fn make_plan_context(
    dep_eci: [f64; 6],
    tgt_eci: [f64; 6],
    max_phase_dv: f64,
    max_transfer_dv: f64,
    sampling_mode: SamplingMode,
) -> PlanContext {
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
        max_phase_dv,
        max_transfer_dv,
        min_perigee: 6500.0,
        max_apogee: 50000.0,
        max_revs: 2,
        sampling_mode,
        execution_policy: ExecutionPolicy::default(),
        distance_tol: 0.025,
        deployer_min_distance: 0.12,
        ..TransferRequest::with_j2_closure_settings(
            two_phase_transfer_rs::J2ClosureSettings::default(),
        )
    })
}

fn make_easy_transfer() -> PlanContext {
    // LEO to LEO+100km, coplanar, small altitude change
    let r_leo = 6778.0;
    let v_leo = (MU / r_leo).sqrt();
    let r_target = 6878.0;
    let v_target = (MU / r_target).sqrt();

    let dep_eci = [r_leo, 0.0, 0.0, 0.0, v_leo, 0.0];
    // Target 90 degrees ahead
    let tgt_eci = [0.0, r_target, 0.0, -v_target, 0.0, 0.0];
    make_plan_context(dep_eci, tgt_eci, 1.0, 2.0, SamplingMode::Fast)
}

fn make_moderate_transfer() -> PlanContext {
    // 30-degree plane change
    let r_leo = 6778.0;
    let v_leo = (MU / r_leo).sqrt();
    let dep_eci = [r_leo, 0.0, 0.0, 0.0, v_leo, 0.0];

    // Target in different plane (30 degrees inclination)
    let inc = 30.0_f64.to_radians();
    let angle = std::f64::consts::PI / 2.0;
    let tgt_eci = [
        r_leo * angle.cos(),
        r_leo * angle.sin() * inc.cos(),
        r_leo * angle.sin() * inc.sin(),
        -v_leo * angle.sin() * inc.cos(),
        v_leo * angle.cos(),
        0.0,
    ];

    make_plan_context(dep_eci, tgt_eci, 1.5, 3.0, SamplingMode::Fast)
}

fn make_hard_transfer() -> PlanContext {
    // 90-degree plane change (equatorial to polar)
    let r_leo = 6778.0;
    let v_leo = (MU / r_leo).sqrt();
    let dep_eci = [r_leo, 0.0, 0.0, 0.0, v_leo, 0.0]; // Equatorial

    // Polar orbit (90 degrees inclination)
    let tgt_eci = [r_leo, 0.0, 0.0, 0.0, 0.0, v_leo];

    make_plan_context(dep_eci, tgt_eci, 2.0, 5.0, SamplingMode::Fast)
}

fn front_cost(front: &TransferFront) -> f64 {
    front
        .candidates
        .first()
        .map_or(f64::INFINITY, |candidate| candidate.cost)
}

const fn front_valid(front: &TransferFront) -> bool {
    !front.is_empty()
}

fn front_func_evals(front: &TransferFront) -> u64 {
    front
        .candidates
        .first()
        .map_or(0, |candidate| candidate.func_evals)
}

fn profile_lambert_calls(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lambert Call Profiling");

    let cases = vec![
        ("Easy", make_easy_transfer()),
        ("Moderate", make_moderate_transfer()),
        ("Hard", make_hard_transfer()),
    ];

    for (name, mut ctx) in cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| solve_plan(std::hint::black_box(&mut ctx), None));
        });

        // Single run for diagnostics; Criterion timing is measured above.
        let result = match solve_plan(&mut ctx, None) {
            Ok(front) => front,
            Err(authority) => {
                eprintln!(
                    "profile_lambert_calls {name} diagnostic failed target-propagation authority: {authority}"
                );
                std::process::abort()
            }
        };

        let divider = "=".repeat(60);
        println!("\n{divider}");
        println!("PROFILE SUMMARY: {name}");
        println!("{divider}");
        println!("Cost: {:.6} km/s", front_cost(&result));
        println!("Valid: {}", front_valid(&result));
        println!("Func evals: {}", front_func_evals(&result));
        println!();
    }

    group.finish();
}

criterion_group!(benches, profile_lambert_calls);
criterion_main!(benches);
