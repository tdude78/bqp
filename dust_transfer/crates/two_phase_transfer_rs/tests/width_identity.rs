//! Bit identity of a real constellation solve across rayon pool widths.
//!
//! # Why this file exists
//!
//! Five call sites in this crate gate a leaf fan-out on
//! `solve::should_use_leaf_parallel`, and a sixth (`solve_policy::
//! should_parallelize_selected_pairs`) gates the outer selected-pair fan-out.
//! Every one of them reads `rayon::current_num_threads()` and
//! `rayon::current_thread_index()` — AMBIENT EXECUTION CONTEXT, not inputs. The
//! same solve therefore takes different code paths at `--threads 1`, at the
//! campaign's `--threads 8`, and at the top of the probe ladder at 64.
//!
//! That is only sound while every one of those branches produces the same
//! answer, and nothing in the type system says so. Until this file, nothing in
//! this crate measured it: `dust_estimates_rs::parallel_branch_identity`
//! (deleted 2026-08-06 with the GMM dust-mass search) and `lightyear_odeint_rs/tests/
//! batch_width_identity.rs` sweep width for *their* crates, and
//! `sharded_hybrid.rs` makes `rayon_threads` a shard-protocol identity field —
//! that is, cross-width identity is currently handled by REFUSING to shard
//! across widths rather than by proving the branches agree. This is the proof
//! that did not exist.
//!
//! # The two regimes, and why both are needed
//!
//! The outer and leaf fan-outs are MUTUALLY EXCLUSIVE by construction. Once the
//! selected-pair `par_iter` fires, every pair solve runs on a rayon worker, so
//! `caller_is_top_level` is false at all five leaf gates and they stay serial.
//! A single fixture therefore cannot exercise both, and a test that only ran
//! the outer one would leave the five leaf gates unmeasured while looking like
//! a width sweep.
//!
//! So each width runs two solves:
//!
//! * `leaf` — `pairs_to_verify` below `TRANSFER_PAIR_PAR_THRESHOLD`, so the
//!   outer `par_iter` declines and the pair solve runs at top level. This is
//!   the regime in which the five leaf gates fire.
//! * `outer` — `pairs_to_verify` at or above the threshold, so the outer
//!   `par_iter` fires and the leaf gates are nested-serial.
//!
//! # What is compared, and what is deliberately not
//!
//! * The full candidate front, through its derived `Debug`. Rust prints an f64
//!   as the shortest string that round-trips, so this rendering determines the
//!   bits: transfer/phase/arrival delta-V vectors and norms, every intercept
//!   and release state, the Lambert branch tokens, the J2 residuals and the
//!   replay provenance all compare exactly. `PlanResult::polish_time_us` is a
//!   wall clock by name only — production never writes it from a clock — so it
//!   is safe inside this rendering.
//! * Every INTEGER and bool field of `VerifiedSupersetStageMetrics`, selected
//!   by parsing its derived `Debug` rather than by naming fields, so a counter
//!   added later is covered without editing this file.
//! * NOT the f64 metrics. Almost all are wall-clock stage timers.
//!   `j2_correction_residual_m_sum` is instead a diagnostic reduction, but the
//!   per-candidate J2 residuals already compare bit-exactly in the front. This
//!   harness therefore keeps its counter comparison integral rather than
//!   treating one aggregate f64 as a work counter.
//! * NOT the dispatch-shape counters in `DISPATCH_SHAPE_FIELDS`. Those are the
//!   evidence that the branches diverged, and they are asserted to differ.
//! * The candidate front is compared BOTH across widths and across a repeat of
//!   each width. The repeat is what stops the schedule-dependent exemption
//!   below from quietly covering an unreproducible answer.
//!
//! # Why a child process
//!
//! The rayon global pool is process-wide and set once, so the width has to be
//! forced in a re-exec'd child — the same approach, and the same reason, as
//! the since-deleted `dust_estimates_rs::parallel_branch_identity` and the
//! later `batch_width_identity.rs`. Comparing across processes
//! also keeps this harness free of any parallel float reduction of its own.

use std::collections::BTreeMap;
use two_phase_transfer_rs::batch_eci::{BatchEciConfiguration, BatchEciRequest};
use two_phase_transfer_rs::solve::FrontOutputMode;
use two_phase_transfer_rs::types::{BodyForceConfig, BodyRole, SearchDepthPolicy};
use two_phase_transfer_rs::{
    constellation_solve_batch_eci_precomputed, SamplingMode, TransferLocalOptimizerConfig,
};

/// Widths the campaign actually spans: 1 is the serial control (`--threads 1`,
/// and the width at which every gate in this crate is unreachable), 8 is the
/// campaign, 64 is the top of the probe ladder.
const WIDTHS: [usize; 3] = [1, 8, 64];

/// How many times the whole width sweep is run. Two, so every counter can be
/// compared against a SECOND run at its own width before any cross-width claim
/// is made about it — see [`assert_same_width_repeat_is_reproducible`].
const SWEEP_PASSES: usize = 2;

const CHILD_WIDTH_ENV: &str = "ND_TWO_PHASE_WIDTH_IDENTITY_CHILD";
const CHILD_MARKER: &str = "TWO_PHASE_WIDTH_IDENTITY_CHILD_COMPLETED";
const TEST_NAME: &str = "solve_is_bit_identical_across_pool_widths";

/// Below `solve_policy::TRANSFER_PAIR_PAR_THRESHOLD` (4), so the outer
/// selected-pair `par_iter` declines and the leaf gates see a top-level caller.
const LEAF_REGIME_PAIRS: usize = 2;
/// At or above that threshold, so the outer `par_iter` fires instead.
const OUTER_REGIME_PAIRS: usize = 8;

const LEAF_REGIME: &str = "leaf";
const OUTER_REGIME: &str = "outer";

/// Counters that record WHICH branch ran. They are supposed to differ across
/// widths; that is the whole point of the sweep, and they are asserted to
/// differ below rather than compared for equality.
const DISPATCH_SHAPE_FIELDS: [&str; 9] = [
    "rayon_current_num_threads",
    "selected_pair_serial_event_count",
    "selected_pair_parallel_event_count",
    "selected_pair_parallel_policy_enabled_count",
    "oxymoo_serial_batch_count",
    "oxymoo_parallel_batch_count",
    "anchor_parallel_count",
    "branch_parallel_count",
    "polish_parallel_count",
];

/// Counters that are not merely width-dependent but RUN-TO-RUN unstable at a
/// FIXED width, and only where a leaf fan-out actually ran. Exempted there,
/// compared everywhere else — so the exemption cannot spread silently to the
/// outer fan-out, which is measurably stable and stays fully compared.
///
/// # Current mechanism
///
/// `evaluate_plan_local` computes a fresh `PlanResult` on every call. The
/// retired inner plan-result cache no longer controls whether J2, Lambert, or
/// branch work runs. Its former downstream exemptions therefore must not
/// survive here.
///
/// One exact intermediate cache remains relevant: each worker's
/// `phase_state_cache`, keyed by `time2phase_ratio.to_bits()`. Every direct
/// evaluation records one hit or miss; a miss performs the corresponding J2
/// state propagation. Work stealing can change which exact phase keys a worker
/// has already seen, so only that hit/miss split and its propagation work may
/// vary at a fixed parallel width. Every other integer tally is compared.
const SCHEDULE_DEPENDENT_COUNTERS: [&str; 3] = [
    "j2_propagate_state_count",
    "phase_state_cache_hit_count",
    "phase_state_cache_miss_count",
];

/// The four leaf fan-out tallies. `deterministic_grid` is the fifth leaf gate
/// and has no public counter (its only probe is `#[cfg(test)]`, invisible from
/// an integration test), so it is exercised but not separately witnessed.
const LEAF_DISPATCH_COUNTERS: [&str; 4] = [
    "oxymoo_parallel_batch_count",
    "anchor_parallel_count",
    "branch_parallel_count",
    "polish_parallel_count",
];

/// Floors on parsed integer/bool coverage. The current metrics expose 84 such
/// fields. Cross-width starts with 75 after excluding nine asserted dispatch
/// fields, then a leaf comparison may exclude three schedule-dependent scratch
/// fields. Same-width repeats compare 81 fields for leaf work and all 84 when
/// no leaf fan-out ran. A floor rather than equality lets new counters join
/// coverage automatically while a broken parser still fails closed.
const MIN_CROSS_WIDTH_INTEGER_FIELDS: usize = 75;
const MIN_REPEAT_INTEGER_FIELDS: usize = 81;
const MIN_REPEAT_INTEGER_FIELDS_NO_LEAF: usize = 84;

fn leaf_fanout_ran(fields: &BTreeMap<String, String>) -> bool {
    LEAF_DISPATCH_COUNTERS
        .iter()
        .any(|name| metric(fields, name) > 0)
}

fn kep_to_eci(kep: &[f64; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    satpy_core::kep2eci_impl(kep, false, 0.0, 0.0, false, &mut out);
    out
}

/// A 15-satellite constellation over two targets — the same shape the front
/// solve benches use, and large enough that pair screening has a real choice to
/// make. Physics is not invented here: these are ordinary LEO elements pushed
/// through the production `kep2eci_impl`.
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

/// One regime's full result, rendered so the parent can compare it as text.
///
/// Deliberately requests `VerifiedSuperset` through the public batch entry:
/// every stage counter — including all five parallel-dispatch tallies — is only
/// accumulated under `VerifiedSuperset` (`solve::reduce_event`). Driving the
/// Pareto mode would compare candidate fronts against a metrics struct that
/// is identically zero at every width, which reads like a passing sweep and
/// witnesses nothing.
///
/// One event, so `should_use_outer_batch_parallel_for_flat_work_units` declines
/// (`batch_size` below `BATCH_PAR_THRESHOLD`, flat units below `2 * threads`)
/// and the flat event×pair driver — which would put every pair solve on a rayon
/// worker and leave the leaf gates nested-serial at every width — stays out of
/// the way. That is asserted, not assumed.
fn run_regime(label: &str, pairs_to_verify: usize) {
    let satellites = constellation();
    let target1 = kep_to_eci(&[7100.0, 0.002, 0.21, 0.1, 0.0, 0.2]);
    let target2 = kep_to_eci(&[7120.0, 0.002, 0.21, 0.1, 0.0, 0.25]);
    let target_body_forces = [[BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2]];

    let fronts = constellation_solve_batch_eci_precomputed(BatchEciRequest {
        satellite_eci: &satellites,
        satellite_equinoctial: None,
        satellite_count: satellites.len(),
        configuration: BatchEciConfiguration {
            targets_one_eci: &target1,
            targets_two_eci: &target2,
            epoch_jds: &[2_460_000.5],
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
            target_body_forces: &target_body_forces,
            force_config: None,
            require_high_fidelity: false,
            j2_closure_settings: two_phase_transfer_rs::solve::J2ClosureSettings::default(),
            packed_coeffs: None,
            local_optimizer: TransferLocalOptimizerConfig::default(),
            warm_starts: None,
            front_output_mode: FrontOutputMode::VerifiedSuperset,
        },
    });
    assert!(
        fronts.is_ok(),
        "width-identity fixture must solve: {:?}",
        fronts.as_ref().err()
    );
    let Ok(fronts) = fronts else { return };
    assert_eq!(fronts.len(), 1, "one event in, one front out");
    let Some(front) = fronts.into_iter().next() else {
        return;
    };

    // Leading newline on purpose: with `--nocapture` the harness writes
    // `test <name> ... ` without a trailing newline, so the first marker would
    // otherwise share a line with it and fail to parse as a prefix.
    println!("\nREGIME {label}");
    println!("CANDIDATES {:?}", front.candidates);
    println!("METRICS {:?}", front.verified_superset_metrics);
}

/// Split a derived one-line `Debug` of a flat struct into `field -> value`.
///
/// Returns `None` if the string is not of that shape, which the caller treats
/// as a hard failure rather than as an empty comparison.
fn parse_debug_struct(text: &str) -> Option<BTreeMap<String, String>> {
    let body = text.split_once('{')?.1.strip_suffix('}')?.trim();
    let mut fields = BTreeMap::new();
    for entry in body.split(", ") {
        let (name, value) = entry.split_once(": ")?;
        fields.insert(name.trim().to_owned(), value.trim().to_owned());
    }
    Some(fields)
}

/// True for a value that `Debug` rendered as an integer or a bool.
///
/// Rust's `Debug` for f64 always emits `.`, `e`, `inf` or `NaN`, so nothing
/// float-valued can pass this — which is how the float telemetry is excluded
/// without naming a single field.
fn is_integral(value: &str) -> bool {
    value == "true" || value == "false" || value.parse::<i64>().is_ok()
}

/// Read one integer counter. A missing or non-integer field is a hard failure,
/// reported through `assert!` rather than `panic!` so the message survives
/// clippy's production-code restriction lints, which do not recognise a helper
/// in an integration-test crate as test code.
fn metric(fields: &BTreeMap<String, String>, name: &str) -> i64 {
    let parsed = fields.get(name).and_then(|raw| raw.parse::<i64>().ok());
    assert!(
        parsed.is_some(),
        "metrics must carry an integer `{name}` field, got {:?}",
        fields.get(name)
    );
    parsed.unwrap_or_default()
}

struct RegimeOutput {
    candidates: String,
    metrics: BTreeMap<String, String>,
}

/// Pull the two regimes out of one child's stdout.
///
/// Returns `Err` with a description rather than failing in place: the caller is
/// the `#[test]` function and owns the assertion.
fn parse_child_output(stdout: &str) -> anyhow::Result<BTreeMap<String, RegimeOutput>> {
    let mut regimes = BTreeMap::new();
    let mut current: Option<String> = None;
    let mut candidates: Option<String> = None;
    for line in stdout.lines() {
        if let Some(label) = line.strip_prefix("REGIME ") {
            current = Some(label.to_owned());
            candidates = None;
        } else if let Some(rest) = line.strip_prefix("CANDIDATES ") {
            candidates = Some(rest.to_owned());
        } else if let Some(rest) = line.strip_prefix("METRICS ") {
            let label = current
                .clone()
                .ok_or_else(|| anyhow::anyhow!("a METRICS line arrived before any REGIME line"))?;
            let rendered = candidates
                .take()
                .ok_or_else(|| anyhow::anyhow!("regime {label} emitted no CANDIDATES line"))?;
            let metrics = parse_debug_struct(rest).ok_or_else(|| {
                anyhow::anyhow!("regime {label} metrics are not a flat derived Debug rendering")
            })?;
            regimes.insert(
                label,
                RegimeOutput {
                    candidates: rendered,
                    metrics,
                },
            );
        }
    }
    Ok(regimes)
}

/// First differing character, with a window either side. A raw dump of two
/// multi-kilobyte candidate renderings is unreadable and hides the finding.
fn first_difference(lhs: &str, rhs: &str) -> String {
    let mut offset = 0_usize;
    for (index, (a, b)) in lhs.chars().zip(rhs.chars()).enumerate() {
        if a != b {
            offset = index;
            break;
        }
        offset = index.saturating_add(1);
    }
    let start = offset.saturating_sub(160);
    let take = 400_usize;
    let window = |text: &str| -> String { text.chars().skip(start).take(take).collect::<String>() };
    format!(
        "first difference at character {offset}\n  reference: ...{}...\n  candidate: ...{}...",
        window(lhs),
        window(rhs)
    )
}

#[test]
fn solve_is_bit_identical_across_pool_widths() {
    if let Some(raw_width) = std::env::var_os(CHILD_WIDTH_ENV) {
        let width: usize = raw_width
            .to_string_lossy()
            .parse()
            .expect("child pool width must be numeric");
        let installed = nd_sched::init_global_pool_authoritative(width)
            .expect("authoritative global pool must initialise");
        assert_eq!(installed, width, "installed pool width");
        assert_eq!(
            rayon::current_num_threads(),
            width,
            "rayon must report the forced width, or every gate below reads the wrong context"
        );
        run_regime(LEAF_REGIME, LEAF_REGIME_PAIRS);
        run_regime(OUTER_REGIME, OUTER_REGIME_PAIRS);
        println!("{CHILD_MARKER}");
        return;
    }

    let executable = std::env::current_exe().expect("test executable must resolve");
    // Two independent sweeps of the same widths. The first is the cross-width
    // comparison this file has always made; the second exists so that a counter
    // can be asked whether it is reproducible at a FIXED width before any
    // cross-width claim is made about it. Six children at roughly 0.1 s each.
    let mut passes: Vec<BTreeMap<usize, BTreeMap<String, RegimeOutput>>> = Vec::new();
    for _ in 0..SWEEP_PASSES {
        let mut by_width: BTreeMap<usize, BTreeMap<String, RegimeOutput>> = BTreeMap::new();
        for width in WIDTHS {
            let output = std::process::Command::new(&executable)
                .args([TEST_NAME, "--exact", "--nocapture"])
                .env(CHILD_WIDTH_ENV, width.to_string())
                .env("RUST_TEST_THREADS", "1")
                .output()
                .expect("spawning the width-forced child must succeed");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "child failed at width {width}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            // Without this the parent passes whenever the child exits 0 for any
            // reason at all, including having filtered out every test.
            assert!(
                stdout.contains(CHILD_MARKER),
                "child at width {width} exited 0 without running the solve\nstdout:\n{stdout}"
            );
            let parsed = parse_child_output(&stdout);
            assert!(
                parsed.is_ok(),
                "width {width} stdout is not parseable: {:?}",
                parsed.as_ref().err()
            );
            let Ok(regimes) = parsed else { return };
            assert_eq!(
                regimes.len(),
                2,
                "width {width} must report both regimes, got {:?}",
                regimes.keys().collect::<Vec<_>>()
            );
            by_width.insert(width, regimes);
        }
        passes.push(by_width);
    }
    // Assert-then-bind, the idiom every other unwrap in this file uses: a bare
    // `else return` here would let an empty sweep pass green.
    assert!(
        !passes.is_empty(),
        "width sweep produced no passes; nothing was compared"
    );
    let Some(by_width) = passes.first() else {
        return;
    };

    for regime in [LEAF_REGIME, OUTER_REGIME] {
        // Resolve every width once, in sweep order, so the comparison helpers
        // never have to look a width up and never have to panic when one is
        // missing.
        let mut per_width: Vec<(usize, &RegimeOutput)> = Vec::new();
        for width in WIDTHS {
            let found = by_width.get(&width).and_then(|regimes| regimes.get(regime));
            assert!(
                found.is_some(),
                "width {width} must have run regime {regime}"
            );
            if let Some(output) = found {
                per_width.push((width, output));
            }
        }
        assert_eq!(per_width.len(), WIDTHS.len(), "every width must be present");
        assert_branch_divergence(regime, &per_width);
        assert_results_identical(regime, &per_width);
        assert_same_width_repeat_is_reproducible(regime, &passes);
    }
}

/// NON-VACUITY. Prove the widths took DIFFERENT branches before comparing their
/// answers. Without this the comparison below would be two serial runs agreeing
/// with each other, which is the shape of a test that passes forever while
/// measuring nothing.
fn assert_branch_divergence(regime: &str, per_width: &[(usize, &RegimeOutput)]) {
    // The fixture has to be the size this regime claims, or the gate it is
    // aimed at was never approached.
    for (width, output) in per_width {
        let fields = &output.metrics;
        let selected = metric(fields, "selected_pair_count");
        let expected = if regime == LEAF_REGIME {
            i64::try_from(LEAF_REGIME_PAIRS).unwrap_or_default()
        } else {
            i64::try_from(OUTER_REGIME_PAIRS).unwrap_or_default()
        };
        assert_eq!(
            selected, expected,
            "regime {regime} at width {width}: fixture selected {selected} pairs, \
             but the regime is defined by selecting {expected}"
        );
        assert_eq!(
            metric(fields, "rayon_current_num_threads"),
            i64::try_from(*width).unwrap_or_default(),
            "regime {regime}: the solve must observe the forced pool width"
        );
        // If the flat event×pair driver took this batch, every pair solve ran
        // on a rayon worker and BOTH the outer selected-pair gate and all five
        // leaf gates are nested-serial — at which point the sweep below is
        // three serial runs agreeing with each other.
        assert_eq!(
            metric(fields, "outer_batch_parallel_event_count"),
            0,
            "regime {regime} at width {width}: the flat batch driver must decline, \
             or neither fan-out under test is reachable"
        );
    }

    for (width, output) in per_width {
        let width = *width;
        let f = &output.metrics;
        let parallel_event = metric(f, "selected_pair_parallel_event_count");
        let leaf_total: i64 = LEAF_DISPATCH_COUNTERS
            .iter()
            .map(|name| metric(f, name))
            .sum();
        println!(
            "regime {regime} width {width}: outer_parallel={parallel_event} \
             oxymoo_par={} oxymoo_ser={} anchor_par={} branch_par={} polish_par={}",
            metric(f, "oxymoo_parallel_batch_count"),
            metric(f, "oxymoo_serial_batch_count"),
            metric(f, "anchor_parallel_count"),
            metric(f, "branch_parallel_count"),
            metric(f, "polish_parallel_count"),
        );

        match (regime, width) {
            (LEAF_REGIME, 1) => {
                assert_eq!(
                    leaf_total, 0,
                    "regime leaf at width 1: the leaf gates require thread_count > 1, \
                     so every leaf fan-out tally must be zero"
                );
                assert!(
                    metric(f, "oxymoo_serial_batch_count") > 0,
                    "regime leaf at width 1: the OxyMOO batches must have run SERIALLY, \
                     or the width-1 arm never reached the gate at all"
                );
            }
            (LEAF_REGIME, _) => {
                for name in LEAF_DISPATCH_COUNTERS {
                    assert!(
                        metric(f, name) > 0,
                        "regime leaf at width {width}: `{name}` is zero, so that leaf gate \
                         took the SERIAL branch and this width is not testing it"
                    );
                }
                assert_eq!(
                    parallel_event, 0,
                    "regime leaf at width {width}: the outer selected-pair par_iter must \
                     DECLINE here, or the leaf gates are nested and stay serial"
                );
            }
            (_, 1) => {
                assert_eq!(
                    parallel_event, 0,
                    "regime outer at width 1: the outer par_iter requires thread_count > 1"
                );
                assert_eq!(
                    metric(f, "selected_pair_serial_event_count"),
                    1,
                    "regime outer at width 1: the event must be recorded as serially solved"
                );
            }
            (_, _) => {
                assert_eq!(
                    parallel_event, 1,
                    "regime outer at width {width}: the outer selected-pair par_iter must FIRE"
                );
                assert_eq!(
                    leaf_total, 0,
                    "regime outer at width {width}: with the outer par_iter running, every \
                     pair solve is on a rayon worker, so the leaf gates must stay serial. \
                     A nonzero tally here means the two fan-outs are no longer exclusive."
                );
            }
        }
    }
}

/// Integer metric names that differ across widths, and the names of every
/// counter compared, given a reference field set.
fn comparable_counters<'a>(
    fields: &'a BTreeMap<String, String>,
    exempt: &[&str],
) -> Vec<&'a String> {
    fields
        .iter()
        .filter(|(name, value)| is_integral(value) && !exempt.contains(&name.as_str()))
        .map(|(name, _)| name)
        .collect()
}

/// Compare two widths' integer counters, reporting EVERY mismatch.
///
/// Stopping at the first one reports whichever counter happens to sort earliest
/// and hides the rest, which turns a twenty-counter finding into a one-counter
/// one.
fn counter_mismatches(
    reference: &BTreeMap<String, String>,
    other: &BTreeMap<String, String>,
    names: &[&String],
) -> Vec<String> {
    let mut mismatches = Vec::new();
    let show = |value: Option<&String>| -> String {
        value.map_or_else(|| "<missing>".to_owned(), Clone::clone)
    };
    for name in names {
        let expected = reference.get(name.as_str());
        let actual = other.get(name.as_str());
        // A field present on one side and absent on the other is a mismatch,
        // not a lookup failure — comparing the `Option`s says so directly.
        if expected != actual {
            mismatches.push(format!("  {name}: {} -> {}", show(expected), show(actual)));
        }
    }
    mismatches
}

/// What replaces the refuted conservation law: the answer, and every counter
/// outside [`SCHEDULE_DEPENDENT_COUNTERS`], must survive REPEATING a width.
///
/// The exemption above says some counters move with the work-stealing
/// partition. Left alone that is an unfalsifiable excuse — a counter can be
/// dropped from the cross-width comparison by asserting it is schedule noise,
/// and nothing ever checks the claim. This does check it, from the only angle
/// that can: a second sweep at the SAME widths, where the width is held fixed
/// and only the schedule is free to differ.
///
/// Two things are asserted, and between them they bound the exemption from both
/// sides:
///
/// * The candidate front must be bit-identical across the repeat. This is the
///   property the whole file exists to protect, and the cross-width comparison
///   alone never established it — a front that varied run to run at ONE width
///   would still have matched itself across widths whenever the two sweeps
///   happened to land in the same state.
/// * Every integer counter NOT exempted must be bit-identical across the
///   repeat. An exemption that is too narrow shows up here as a red test rather
///   than as a cross-width flake, which is the difference between a finding and
///   a mystery.
///
/// The exempted counters are deliberately not asserted to MOVE; demanding a
/// particular work-stealing partition would install a flake.
fn assert_same_width_repeat_is_reproducible(
    regime: &str,
    passes: &[BTreeMap<usize, BTreeMap<String, RegimeOutput>>],
) {
    let Some(first) = passes.first() else { return };
    let mut compared = 0_usize;
    for later in passes.iter().skip(1) {
        for width in WIDTHS {
            let reference = first.get(&width).and_then(|regimes| regimes.get(regime));
            let repeat = later.get(&width).and_then(|regimes| regimes.get(regime));
            let (Some(reference), Some(repeat)) = (reference, repeat) else {
                continue;
            };

            assert_eq!(
                reference.candidates,
                repeat.candidates,
                "regime {regime} at width {width}: the candidate front is not reproducible \
                 across two runs at the SAME width. The pool width is held fixed here, so \
                 only the work-stealing schedule differs — the answer is a function of the \
                 schedule.\n{}",
                first_difference(&reference.candidates, &repeat.candidates)
            );

            let leaf_ran = leaf_fanout_ran(&reference.metrics) || leaf_fanout_ran(&repeat.metrics);
            let exempt: &[&str] = if leaf_ran {
                &SCHEDULE_DEPENDENT_COUNTERS
            } else {
                &[]
            };
            let minimum = if leaf_ran {
                MIN_REPEAT_INTEGER_FIELDS
            } else {
                MIN_REPEAT_INTEGER_FIELDS_NO_LEAF
            };
            let names = comparable_counters(&reference.metrics, exempt);
            // NON-VACUITY. If the rendering stopped parsing, every comparison
            // above would be an empty loop reporting success.
            assert!(
                names.len() >= minimum,
                "regime {regime} at width {width}: only {} integer metric fields were parsed \
                 out of the derived Debug (floor {minimum})",
                names.len()
            );
            let mismatches = counter_mismatches(&reference.metrics, &repeat.metrics, &names);
            assert!(
                mismatches.is_empty(),
                "regime {regime} at width {width}: {} integer counter(s) differ between two \
                 runs at the SAME width. These are outside SCHEDULE_DEPENDENT_COUNTERS, so \
                 they are claimed to be reproducible; the claim is wrong, and the cross-width \
                 exemption is too narrow to describe what this build actually does.\n{}",
                mismatches.len(),
                mismatches.join("\n")
            );
            compared = compared.saturating_add(1);
        }
    }
    assert!(
        compared > 0,
        "regime {regime}: no width was swept twice, so nothing was checked for reproducibility"
    );
}

/// The identity itself: the ANSWER must not depend on which branch ran.
fn assert_results_identical(regime: &str, per_width: &[(usize, &RegimeOutput)]) {
    let Some((reference_width, reference)) = per_width.first() else {
        return;
    };
    let reference_width = *reference_width;
    assert_eq!(
        reference_width, 1,
        "the first swept width is the serial control and must be 1"
    );

    // Non-vacuity: an empty front would satisfy every comparison below.
    assert!(
        reference.candidates.len() > 200 && reference.candidates != "[]",
        "regime {regime}: the fixture produced no candidates, so there is nothing to compare"
    );

    for (width, output) in per_width {
        assert_eq!(
            metric(&output.metrics, "lambert_branch_valid_count"),
            metric(&output.metrics, "lambert_branch_prograde_count")
                .checked_add(metric(&output.metrics, "lambert_branch_retrograde_count",))
                .unwrap_or(i64::MIN),
            "regime {regime}: width {width} lost a valid Lambert branch classification"
        );
    }

    // Every exemption below must still name a live field. An exemption for a
    // deleted field silently narrows this test and nothing would report it.
    for name in DISPATCH_SHAPE_FIELDS
        .into_iter()
        .chain(SCHEDULE_DEPENDENT_COUNTERS)
    {
        assert!(
            reference.metrics.contains_key(name),
            "regime {regime}: `{name}` is exempted from a comparison below but no longer \
             exists on the metrics struct"
        );
    }

    // The schedule-dependent exemption is keyed on whether a leaf fan-out
    // actually ran, not on the regime name, so it tightens by itself the moment
    // a comparison stops involving the leaf path.
    let across_dispatch = comparable_counters(&reference.metrics, &DISPATCH_SHAPE_FIELDS);
    assert!(
        across_dispatch.len() >= MIN_CROSS_WIDTH_INTEGER_FIELDS,
        "regime {regime}: only {} integer metric fields were parsed out of the derived Debug \
         (floor {MIN_CROSS_WIDTH_INTEGER_FIELDS}). The rendering changed shape and this \
         comparison has quietly stopped covering the counters.",
        across_dispatch.len()
    );

    for (width, other) in per_width.iter().skip(1) {
        let width = *width;

        // THE DELIVERABLE. Selected transfers, every delta-V component and
        // norm, the front's contents and order, the Lambert branch tokens and
        // the replay provenance — all of it, exactly.
        assert_eq!(
            reference.candidates,
            other.candidates,
            "regime {regime}: the candidate front at width {width} differs from width \
             {reference_width}. `--threads` is silently a science parameter.\n{}",
            first_difference(&reference.candidates, &other.candidates)
        );

        let names = if leaf_fanout_ran(&other.metrics) {
            across_dispatch
                .iter()
                .filter(|name| !SCHEDULE_DEPENDENT_COUNTERS.contains(&name.as_str()))
                .copied()
                .collect()
        } else {
            across_dispatch.clone()
        };
        let mismatches = counter_mismatches(&reference.metrics, &other.metrics, &names);
        assert!(
            mismatches.is_empty(),
            "regime {regime}: {} integer counter(s) differ between the SERIAL width \
             {reference_width} and width {width}. Each parallel arm folds its per-unit \
             deltas in index order, so these are exact by construction and a difference \
             is a real divergence, not reduction-order noise.\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );

        // The OxyMOO batches were dispatched differently but the same number of
        // them ran: the search did not change shape, only its execution did.
        let batches = |fields: &BTreeMap<String, String>| -> i64 {
            metric(fields, "oxymoo_parallel_batch_count")
                .saturating_add(metric(fields, "oxymoo_serial_batch_count"))
        };
        assert_eq!(
            batches(&reference.metrics),
            batches(&other.metrics),
            "regime {regime}: width {width} ran a different number of OxyMOO batches than \
             width {reference_width}"
        );
        // Same for the OxyMOO evaluation cache. Individual counters already
        // compare exactly above; this conservation check makes request-shape
        // drift easier to diagnose.
        let requests = |fields: &BTreeMap<String, String>| -> i64 {
            metric(fields, "oxymoo_eval_cache_hit_count")
                .saturating_add(metric(fields, "oxymoo_eval_cache_miss_count"))
        };
        assert_eq!(
            requests(&reference.metrics),
            requests(&other.metrics),
            "regime {regime}: width {width} requested a different number of OxyMOO plan \
             evaluations than width {reference_width}."
        );
    }

    // Among the PARALLEL widths the same branches ran, so only two things are
    // exempt: the reported width itself, and — where a leaf fan-out ran — the
    // phase-state scratch counters that can depend on work stealing. Everything
    // else must agree, which is what stops a real
    // width sensitivity from hiding inside the serial-vs-parallel exemption
    // above. Note what is NOT exempt here: `anchor_full_eval_count` and the
    // three OxyMOO cache counters. They stay compared between parallel widths,
    // so this remains the strictest comparison in the file.
    let Some((first_parallel, base)) = per_width.get(1) else {
        return;
    };
    let first_parallel = *first_parallel;
    let mut strict_exempt = vec!["rayon_current_num_threads"];
    if leaf_fanout_ran(&base.metrics) {
        strict_exempt.extend(SCHEDULE_DEPENDENT_COUNTERS);
    }
    let strict = comparable_counters(&base.metrics, &strict_exempt);
    assert!(
        strict.len() > across_dispatch.len(),
        "the parallel-vs-parallel comparison must be STRICTER than the serial-vs-parallel \
         one, or the exemptions are doing nothing"
    );
    for (width, other) in per_width.iter().skip(2) {
        let mismatches = counter_mismatches(&base.metrics, &other.metrics, &strict);
        assert!(
            mismatches.is_empty(),
            "regime {regime}: {} integer counter(s) differ between parallel widths \
             {first_parallel} and {width}. Both take the same branches, and every counter \
             that a same-width repeat showed to be schedule-dependent is already exempt, \
             so what is left is a real width sensitivity.\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    }
}
