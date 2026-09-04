use anyhow::{anyhow, ensure, Result};
use criterion::Criterion;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use two_phase_transfer_rs::solve::{
    bench_delta_v_anchor_policy_report, bench_verified_superset_leo_with_delta_v_anchor_policy,
    DeltaVAnchorBenchPolicy,
};
use two_phase_transfer_rs::types::InvalidTargetPropagationAuthorityCode;

static TIMED_FAILURE: AtomicBool = AtomicBool::new(false);

const POLICIES: [(DeltaVAnchorBenchPolicy, &str); 6] = [
    (DeltaVAnchorBenchPolicy::Full, "delta_v_anchor_policy_full"),
    (
        DeltaVAnchorBenchPolicy::NoProbes,
        "delta_v_anchor_policy_no_probes",
    ),
    (
        DeltaVAnchorBenchPolicy::CostOnlyNoProbes,
        "delta_v_anchor_policy_cost_only_no_probes",
    ),
    (
        DeltaVAnchorBenchPolicy::DvOnlyNoProbes,
        "delta_v_anchor_policy_dv_only_no_probes",
    ),
    (
        DeltaVAnchorBenchPolicy::SeedLimit2,
        "delta_v_anchor_policy_seed_limit_2",
    ),
    (
        DeltaVAnchorBenchPolicy::SeedLimit3,
        "delta_v_anchor_policy_seed_limit_3",
    ),
];

fn report_policy(policy: DeltaVAnchorBenchPolicy) -> Result<()> {
    let report = bench_delta_v_anchor_policy_report(policy)
        .map_err(|error| anyhow!("delta-v report failed for {policy:?}: {error}"))?;
    ensure!(
        report.front_candidate_count > 0,
        "delta-v report returned empty front for {policy:?}"
    );
    eprintln!(
        "delta_v_anchor_policy={:?} anchor_candidates={} front_candidates={} objective_equivalent={} cost_anchor_s={:.6} delta_v_anchor_s={:.6} probe_s={:.6} coarse_evals={} fine_evals={} probe_candidates={} polished_candidates={}",
        report.policy,
        report.anchor_candidate_count,
        report.front_candidate_count,
        report.objective_equivalent_to_full,
        report.cost_anchor_s,
        report.delta_v_anchor_s,
        report.probe_s,
        report.coarse_eval_count,
        report.fine_eval_count,
        report.probe_candidate_count,
        report.polished_candidate_count,
    );
    Ok(())
}

fn timed_front_len(
    result: Result<two_phase_transfer_rs::TransferFront, InvalidTargetPropagationAuthorityCode>,
) -> usize {
    match result {
        Ok(front) if !front.is_empty() => front.len(),
        Ok(_) | Err(_) => {
            TIMED_FAILURE.store(true, Ordering::Relaxed);
            0
        }
    }
}

fn bench_policy(c: &mut Criterion) -> Result<()> {
    TIMED_FAILURE.store(false, Ordering::Relaxed);
    for &(policy, _) in &POLICIES {
        report_policy(policy)?;
        let front = bench_verified_superset_leo_with_delta_v_anchor_policy(policy)
            .map_err(|error| anyhow!("delta-v preflight failed for {policy:?}: {error}"))?;
        ensure!(!front.is_empty(), "delta-v preflight empty for {policy:?}");
    }

    let mut group = c.benchmark_group("DeltaV Anchor Policy");
    for (policy, name) in POLICIES {
        group.bench_function(name, move |b| {
            b.iter(|| {
                black_box(timed_front_len(
                    bench_verified_superset_leo_with_delta_v_anchor_policy(black_box(policy)),
                ))
            });
        });
    }
    group.finish();
    if TIMED_FAILURE.load(Ordering::Relaxed) {
        return Err(anyhow!("delta-v benchmark produced error or empty front"));
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut criterion = Criterion::default().configure_from_args();
    bench_policy(&mut criterion)?;
    criterion.final_summary();
    Ok(())
}
