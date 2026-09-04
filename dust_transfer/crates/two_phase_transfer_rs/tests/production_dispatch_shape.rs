//! Which fan-out regime a PRODUCTION-SHAPED solve actually takes.
//!
//! # Why this file exists
//!
//! `width_identity.rs` proves the fan-out regimes agree. It does not say which
//! one production runs, and it deliberately picks its fixtures to reach both:
//! its `leaf` arm drives one event with `pairs_to_verify` below
//! `TRANSFER_PAIR_PAR_THRESHOLD` precisely because nothing else would exercise
//! the five leaf gates. That is the right shape for an identity proof and the
//! wrong shape for a cost question.
//!
//! The cost question is: at the shape the Part A campaign actually submits —
//! the population entry, several designs by several events, and the production
//! `search_pairs_to_verify = 8` — do the five leaf gates ever fire? This file
//! answers it by counting, in one process, at one pool width.
//!
//! # The three dispatch layers, and how each one is switched off
//!
//! Every gate in this crate reads `rayon::current_thread_index()` and treats a
//! caller that is already a rayon worker as disqualified:
//!
//! * P1, `batch_eci::should_use_outer_batch_parallel_for{,_flat_work_units}` —
//!   the flat `(design × event × pair)` driver. Admits on
//!   `flat_work_units >= 2 * threads`, which a campaign cell clears by two
//!   orders of magnitude.
//! * L2, `solve_policy::should_parallelize_selected_pairs` — the per-event
//!   selected-pair `par_iter`. Requires `!nested`.
//! * L3, the five `solve::should_use_leaf_parallel` gates (`solve.rs`
//!   deterministic grid, `moo.rs` `OxyMOO` batch, `branch_expansion.rs`,
//!   `polish.rs`, `anchor.rs`). Each requires `caller_is_top_level`.
//!
//! So P1 firing puts every pair solve on a rayon worker, which disqualifies L2,
//! which is what would otherwise have disqualified L3 — the two exclusions are
//! not independent, they are the same exclusion applied at the same instant.
//! `population` below is the assertion that this is what a campaign-shaped
//! request does.
//!
//! # Non-vacuity
//!
//! A test that asserts four counters are zero passes just as well if the
//! counters were never wired, if the solve returned nothing, or if the pool is
//! one thread wide. All three are excluded:
//!
//! * `leaf_regime_is_reachable` drives the SAME counters non-zero on the
//!   single-event low-pair shape, so zero in the population arm is a fact about
//!   the shape and not about the instrument.
//! * every arm asserts `rayon_current_num_threads > 1` and a positive
//!   `selected_pair_count`, so a serial or empty run fails rather than passes.
//! * the population arm asserts `outer_batch_parallel_event_count` is positive,
//!   which only the flat driver writes.

use two_phase_transfer_rs::batch_eci::{
    BatchEciConfiguration, BatchEciRequest, PopulationBatchEciRequest,
};
use two_phase_transfer_rs::solve::FrontOutputMode;
use two_phase_transfer_rs::types::{
    BodyForceConfig, BodyRole, SearchDepthPolicy, VerifiedSupersetStageMetrics,
};
use two_phase_transfer_rs::{
    constellation_solve_batch_eci_precomputed,
    constellation_solve_population_batch_eci_precomputed, SamplingMode,
    TransferLocalOptimizerConfig,
};

/// Production `mf_transfer().search_pairs_to_verify`, which is what the MF
/// search submits for every cell of the 36-cohort matrix.
const PRODUCTION_PAIRS_TO_VERIFY: usize = 8;

/// Below `TRANSFER_PAIR_PAR_THRESHOLD`, so the L2 `par_iter` declines and the
/// pair solve runs at top level. The only regime in which L3 can fire.
const LEAF_REGIME_PAIRS_TO_VERIFY: usize = 3;

fn kep_to_eci(kep: &[f64; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    satpy_core::kep2eci_impl(kep, false, 0.0, 0.0, false, &mut out);
    out
}

/// The same 15-satellite LEO constellation `width_identity.rs` uses, so the two
/// files describe the same physics from two directions.
fn constellation() -> Vec<[f64; 6]> {
    let mut sats = Vec::with_capacity(15);
    for index in 0_u8..15 {
        let plane = f64::from(index % 5);
        let slot = f64::from(index / 5);
        sats.push(kep_to_eci(&[
            7000.0 + slot * 20.0,
            0.001,
            0.2 + plane * 0.01,
            plane * 0.25,
            0.0,
            slot * 0.35,
        ]));
    }
    sats
}

fn target_one() -> [f64; 6] {
    kep_to_eci(&[7100.0, 0.002, 0.21, 0.1, 0.0, 0.2])
}

fn target_two() -> [f64; 6] {
    kep_to_eci(&[7120.0, 0.002, 0.21, 0.1, 0.0, 0.25])
}

fn configuration<'a>(
    targets_one_eci: &'a [f64],
    targets_two_eci: &'a [f64],
    epoch_jds: &'a [f64],
    target_body_forces: &'a [[BodyForceConfig; 2]],
    pairs_to_verify: usize,
) -> BatchEciConfiguration<'a> {
    BatchEciConfiguration {
        targets_one_eci,
        targets_two_eci,
        epoch_jds,
        max_time_s: 7_200.0,
        max_phase_dv: 0.5,
        max_transfer_dv: 2.0,
        max_revs: 0,
        min_perigee: 6_578.14,
        max_apogee: 41_378.14,
        pairs_to_verify,
        sampling_mode: SamplingMode::Fast,
        search_depth: SearchDepthPolicy::default(),
        distance_tol: 0.025,
        deployer_min_distance: 0.12,
        tof_penalty_weight: 0.1,
        revolution_cap: 1.5,
        target_propagation_authority:
            two_phase_transfer_rs::types::TargetPropagationAuthority::MfJ2,
        target_body_forces,
        force_config: None,
        require_high_fidelity: false,
        j2_closure_settings: two_phase_transfer_rs::solve::J2ClosureSettings::default(),
        packed_coeffs: None,
        local_optimizer: TransferLocalOptimizerConfig::default(),
        warm_starts: None,
        // Every dispatch tally is only accumulated under `VerifiedSuperset`
        // (`solve::reduce_event`); the Pareto mode would compare against an
        // identically zero metrics struct at every shape.
        front_output_mode: FrontOutputMode::VerifiedSuperset,
    }
}

/// The four leaf tallies with a public counter. `deterministic_grid` is the
/// fifth L3 gate and its only probe is `#[cfg(test)]`, so it is exercised but
/// not separately witnessed here — the same limitation `width_identity.rs`
/// records.
const fn leaf_dispatch_total(metrics: &VerifiedSupersetStageMetrics) -> usize {
    metrics
        .oxymoo_parallel_batch_count
        .saturating_add(metrics.anchor_parallel_count)
        .saturating_add(metrics.branch_parallel_count)
        .saturating_add(metrics.polish_parallel_count)
}

/// Sum of every front's metrics, so an arm reports one number per counter
/// regardless of how many `(design, event)` cells it solved.
#[derive(Default)]
struct DispatchTally {
    fronts: usize,
    selected_pairs: usize,
    threads: usize,
    outer_batch_events: usize,
    selected_pair_parallel_events: usize,
    selected_pair_serial_events: usize,
    leaf: usize,
}

impl DispatchTally {
    fn absorb(&mut self, metrics: &VerifiedSupersetStageMetrics) {
        self.fronts = self.fronts.saturating_add(1);
        self.selected_pairs = self
            .selected_pairs
            .saturating_add(metrics.selected_pair_count);
        self.threads = self.threads.max(metrics.rayon_current_num_threads);
        self.outer_batch_events = self
            .outer_batch_events
            .saturating_add(metrics.outer_batch_parallel_event_count);
        self.selected_pair_parallel_events = self
            .selected_pair_parallel_events
            .saturating_add(metrics.selected_pair_parallel_event_count);
        self.selected_pair_serial_events = self
            .selected_pair_serial_events
            .saturating_add(metrics.selected_pair_serial_event_count);
        self.leaf = self.leaf.saturating_add(leaf_dispatch_total(metrics));
    }

    /// Guards every arm shares: a run that solved nothing, or ran one thread
    /// wide, cannot be read as evidence about which gate fired.
    ///
    /// Also prints the arm's whole tally. Under `--nocapture` that turns each
    /// arm into a readable measurement rather than a bare pass, which is the
    /// point of the file: the assertions below say "zero", and the reader
    /// wants to see what the other counters were while it was zero.
    fn assert_is_a_real_parallel_solve(&self, label: &str) {
        println!(
            "TALLY {label}: fronts={} selected_pairs={} threads={} \
             outer_batch_events={} l2_parallel_events={} l2_serial_events={} l3_leaf={}",
            self.fronts,
            self.selected_pairs,
            self.threads,
            self.outer_batch_events,
            self.selected_pair_parallel_events,
            self.selected_pair_serial_events,
            self.leaf,
        );
        assert!(
            self.fronts > 0,
            "{label}: no fronts came back, so no counter was observed"
        );
        assert!(
            self.selected_pairs > 0,
            "{label}: zero selected pairs across {} fronts — the fixture screened \
             everything out and witnesses no dispatch decision",
            self.fronts
        );
        assert!(
            self.threads > 1,
            "{label}: rayon pool is {} thread(s) wide; every gate in this crate \
             is unreachable at width 1, so zero counters prove nothing",
            self.threads
        );
    }
}

/// Campaign shape: the population entry, four designs by three events, the
/// production pair budget.
///
/// `flat_work_units = 12 cells * 8 pairs = 96`, far past the `2 * threads`
/// admission, so P1 takes it.
#[test]
fn population_shape_suppresses_every_leaf_gate() {
    let satellites = constellation();
    let n_sats = satellites.len();
    let design_count = 4;
    let epoch_jds = [2_460_000.5, 2_460_000.6, 2_460_000.7];
    let event_count = epoch_jds.len();

    // One satellite block per (design, event) cell, in the
    // `design-major, event-minor` order the entry documents. Designs are
    // perturbed apart so they are not four copies of one solve.
    let mut population = Vec::with_capacity(design_count * event_count * n_sats);
    for design in 0..design_count {
        let nudge = f64::from(u32::try_from(design).unwrap_or(0)) * 5.0;
        for _ in 0..event_count {
            for sat in &satellites {
                let mut row = *sat;
                row[0] += nudge;
                population.push(row);
            }
        }
    }

    let target1 = target_one();
    let target2 = target_two();
    let mut targets_one = Vec::with_capacity(event_count * 6);
    let mut targets_two = Vec::with_capacity(event_count * 6);
    for _ in 0..event_count {
        targets_one.extend_from_slice(&target1);
        targets_two.extend_from_slice(&target2);
    }
    let target_body_forces =
        vec![[BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2]; event_count];

    let fronts = constellation_solve_population_batch_eci_precomputed(PopulationBatchEciRequest {
        satellite_eci_population: &population,
        satellite_equinoctial_population: None,
        design_count,
        satellite_count: n_sats,
        configuration: configuration(
            &targets_one,
            &targets_two,
            &epoch_jds,
            &target_body_forces,
            PRODUCTION_PAIRS_TO_VERIFY,
        ),
    });
    assert!(
        fronts.is_ok(),
        "population fixture must solve: {:?}",
        fronts.as_ref().err()
    );
    let Ok(fronts) = fronts else { return };

    let mut tally = DispatchTally::default();
    for design_fronts in &fronts {
        for front in design_fronts {
            tally.absorb(&front.verified_superset_metrics);
        }
    }
    tally.assert_is_a_real_parallel_solve("population");

    assert!(
        tally.outer_batch_events > 0,
        "population: the flat (design x event x pair) driver must have run — only \
         it writes outer_batch_parallel_event_count, and without it this arm is \
         measuring the per-design serial fallback instead of the campaign path"
    );
    assert_eq!(
        tally.selected_pair_parallel_events, 0,
        "population: L2 selected-pair par_iter must stay suppressed under the \
         flat driver (every worker is already a rayon worker)"
    );
    assert_eq!(
        tally.leaf, 0,
        "population: all five L3 leaf gates must stay suppressed under the flat \
         driver; got {} firings across {} fronts",
        tally.leaf, tally.fronts
    );
}

/// Same production pair budget, single-event batch entry: the shape a one-off
/// `nd_pipeline::native_mf::solve_transfer_front_group` takes. P1 declines
/// (one cell, `1 * 8 = 8` flat units,
/// below `2 * threads` at any pool this test runs on), so the caller reaches
/// the L2 gate at top level — and `8 >= TRANSFER_PAIR_PAR_THRESHOLD` fires it,
/// which disqualifies L3 exactly as the population arm does.
#[test]
fn single_event_production_pair_budget_fires_l2_not_l3() {
    let satellites = constellation();
    let target1 = target_one();
    let target2 = target_two();
    let target_body_forces = [[BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2]];

    let fronts = constellation_solve_batch_eci_precomputed(BatchEciRequest {
        satellite_eci: &satellites,
        satellite_equinoctial: None,
        satellite_count: satellites.len(),
        configuration: configuration(
            &target1,
            &target2,
            &[2_460_000.5],
            &target_body_forces,
            PRODUCTION_PAIRS_TO_VERIFY,
        ),
    });
    assert!(
        fronts.is_ok(),
        "single-event fixture must solve: {:?}",
        fronts.as_ref().err()
    );
    let Ok(fronts) = fronts else { return };

    let mut tally = DispatchTally::default();
    for front in &fronts {
        tally.absorb(&front.verified_superset_metrics);
    }
    tally.assert_is_a_real_parallel_solve("single_event");

    assert!(
        tally.selected_pair_parallel_events > 0,
        "single_event: the L2 selected-pair par_iter must fire at the production \
         pair budget; if it did not, this arm is not the regime it claims to be"
    );
    assert_eq!(
        tally.leaf, 0,
        "single_event: L3 must stay suppressed once L2 owns the pool; got {} firings",
        tally.leaf
    );
}

/// The poison arm. Same crate, same pool, same counters — only the pair budget
/// changes, and the four leaf tallies go non-zero. Without this the two
/// assertions above would be satisfied by a counter that is never written.
#[test]
fn leaf_regime_is_reachable() {
    let satellites = constellation();
    let target1 = target_one();
    let target2 = target_two();
    let target_body_forces = [[BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2]];

    let fronts = constellation_solve_batch_eci_precomputed(BatchEciRequest {
        satellite_eci: &satellites,
        satellite_equinoctial: None,
        satellite_count: satellites.len(),
        configuration: configuration(
            &target1,
            &target2,
            &[2_460_000.5],
            &target_body_forces,
            LEAF_REGIME_PAIRS_TO_VERIFY,
        ),
    });
    assert!(
        fronts.is_ok(),
        "leaf fixture must solve: {:?}",
        fronts.as_ref().err()
    );
    let Ok(fronts) = fronts else { return };

    let mut tally = DispatchTally::default();
    for front in &fronts {
        tally.absorb(&front.verified_superset_metrics);
    }
    tally.assert_is_a_real_parallel_solve("leaf");

    assert_eq!(
        tally.selected_pair_parallel_events, 0,
        "leaf: below the pair threshold the L2 par_iter must decline, or this \
         arm is not the leaf regime"
    );
    assert!(
        tally.leaf > 0,
        "leaf: the four public leaf tallies must be writable — if they are zero \
         here they are zero everywhere and the population arm proves nothing"
    );
}
