use anyhow::{anyhow, ensure, Result};
use criterion::{measurement::WallTime, BenchmarkGroup, Criterion};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use two_phase_transfer_rs::solve::{
    bench_transfer_moo_policy_report, bench_verified_superset_leo_with_transfer_moo_policy,
    TransferMooBenchPolicy,
};

const POLICIES: [(TransferMooBenchPolicy, &str); 9] = [
    (TransferMooBenchPolicy::Full, "oxymoo_policy_full"),
    (
        TransferMooBenchPolicy::FastPopulation20,
        "oxymoo_policy_population20",
    ),
    (
        TransferMooBenchPolicy::FastPopulation16,
        "oxymoo_policy_population16",
    ),
    (
        TransferMooBenchPolicy::FastGenerations3,
        "oxymoo_policy_generations3",
    ),
    (
        TransferMooBenchPolicy::FastGenerations2,
        "oxymoo_policy_generations2",
    ),
    (
        TransferMooBenchPolicy::FastPopulation20Generations3,
        "oxymoo_policy_population20_generations3",
    ),
    (
        TransferMooBenchPolicy::FastInitialBest1,
        "oxymoo_policy_initial_best1",
    ),
    (
        TransferMooBenchPolicy::FastPopulation20Generations3InitialBest1,
        "oxymoo_policy_population20_generations3_initial_best1",
    ),
    (
        TransferMooBenchPolicy::FastStableObjectiveStop,
        "oxymoo_policy_stable_objective_stop",
    ),
];

static TIMED_FAILURE: AtomicBool = AtomicBool::new(false);

fn report_policy(policy: TransferMooBenchPolicy) -> Result<()> {
    let report = bench_transfer_moo_policy_report(policy)
        .map_err(|error| anyhow!("OxyMOO policy report failed for {policy:?}: {error}"))?;
    ensure!(
        report.front_candidate_count > 0,
        "OxyMOO policy report returned an empty front for {policy:?}"
    );
    eprintln!(
        "oxymoo_policy={:?} population={} generations={} nsga_evals={} front_candidates={} objective_equivalent={} oxymoo_s={:.6} nsga_run_s={:.6} nsga_materialize_s={:.6} materialize_hits={} materialize_misses={} materialize_all_exact={} materialize_recompute={} pre_ox={} post_ox={} post_branch={} post_finalize={}",
        report.policy,
        report.population_size,
        report.generations,
        report.nsga_eval_count,
        report.front_candidate_count,
        report.objective_equivalent_to_full,
        report.oxymoo_s,
        report.nsga_run_s,
        report.nsga_materialize_s,
        report.materialize_plan_cache_hit_count,
        report.materialize_plan_cache_miss_count,
        report.materialize_all_exact_count,
        report.materialize_recompute_count,
        report.pre_oxymoo_candidate_count,
        report.post_oxymoo_candidate_count,
        report.post_branch_candidate_count,
        report.post_finalize_candidate_count,
    );
    Ok(())
}

fn benchmark_policy(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    policy: TransferMooBenchPolicy,
) -> Result<()> {
    let preflight = bench_verified_superset_leo_with_transfer_moo_policy(policy)
        .map_err(|error| anyhow!("OxyMOO benchmark preflight failed for {policy:?}: {error}"))?;
    ensure!(
        !preflight.is_empty(),
        "OxyMOO benchmark preflight returned an empty front for {policy:?}"
    );

    group.bench_function(name, |bencher| {
        bencher.iter(|| {
            match bench_verified_superset_leo_with_transfer_moo_policy(black_box(policy)) {
                Ok(front) if !front.is_empty() => black_box(front.len()),
                Ok(_) => {
                    TIMED_FAILURE.store(true, Ordering::Relaxed);
                    black_box(0)
                }
                Err(_) => {
                    TIMED_FAILURE.store(true, Ordering::Relaxed);
                    black_box(0)
                }
            }
        });
    });
    Ok(())
}

fn bench_policy(criterion: &mut Criterion) -> Result<()> {
    TIMED_FAILURE.store(false, Ordering::Relaxed);
    for &(policy, _) in &POLICIES {
        report_policy(policy)?;
    }

    let mut group = criterion.benchmark_group("OxyMOO Policy");
    for &(policy, name) in &POLICIES {
        benchmark_policy(&mut group, name, policy)?;
    }
    group.finish();
    ensure!(
        !TIMED_FAILURE.load(Ordering::Relaxed),
        "OxyMOO benchmark produced an error or empty front"
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut criterion = Criterion::default().configure_from_args();
    bench_policy(&mut criterion)?;
    criterion.final_summary();
    Ok(())
}
