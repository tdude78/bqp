use crate::scratch::{CoordinateScratch, SolveScratch};
use crate::types::PlanContext;
use crate::types::{
    BodyForceConfig, ConstellationTransferFront, SamplingMode, SearchDepthPolicy,
    TargetPropagationAuthority, TransferLocalOptimizerConfig,
};
use crate::{solve, types};
use std::time::Instant;

const BATCH_PAR_THRESHOLD: usize = 4;
// Pass 40B — chunk-size sweep at the trust canary (workers=8) showed:
//   events=128, min_len=1  (pre-40B): CPU% 281 %, wall 69.78 s
//   events=128, min_len=4:            CPU% 256 %, wall 70.45 s (+1.0 %)
//   events=128, min_len=8:            CPU% 299 %, wall 66.06 s (−5.4 %)
//   events=128, min_len=16:           CPU% 287 %, wall 68.75 s (−1.5 %)
//   events=32,  min_len=8:            wall 29.4 s (+84 % vs pre-40B 15.9 s)
//
// A static `min_len=8` improves events=128 / 512 but cripples events=32
// (only 4 chunks → 4 of 8 workers active). Solution: scale `min_len` so
// rayon always produces ≈ `workers × 2` chunks regardless of batch size.
// At batch=32 / workers=8 → min_len=2 → 16 chunks → 8-worker saturation.
// At batch=128 / workers=8 → min_len=8 → 16 chunks → 8-worker saturation
// AND chunks are large enough to amortize rayon sync overhead.
// At batch=512 / workers=8 → min_len=32 → 16 chunks → same shape, larger
// per-chunk work → better cache locality.
const FLAT_DRIVER_SOLVE_CHUNK_SIZE: usize = 8192;

#[inline]
pub(crate) fn clamp_pairs_to_verify(
    pairs_to_verify: usize,
    n_sats: usize,
) -> Result<usize, TargetBodyForceBatchError> {
    if n_sats == 0 || pairs_to_verify == 0 {
        Ok(0)
    } else {
        n_sats
            .checked_mul(2)
            .map(|available_pairs| pairs_to_verify.min(available_pairs))
            .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TargetBodyForceBatchError {
    CandidateSearchAuthority(types::CandidateSearchAuthorityError),
    InvalidForceConfig(types::InvalidTargetPropagationAuthorityCode),
    Shape {
        expected_rows: usize,
        actual_rows: usize,
    },
    InvalidTarget {
        row: usize,
        target: usize,
        authority: TargetPropagationAuthority,
    },
    ArithmeticOverflow,
    ReducerAuthority(types::InvalidTargetPropagationAuthorityCode),
    InvalidInput(String),
}

impl std::fmt::Display for TargetBodyForceBatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CandidateSearchAuthority(error) => error.fmt(formatter),
            Self::InvalidForceConfig(error) | Self::ReducerAuthority(error) => error.fmt(formatter),
            Self::Shape {
                expected_rows,
                actual_rows,
            } => write!(
                formatter,
                "target body-force batch shape mismatch: expected {expected_rows} rows, got {actual_rows}"
            ),
            Self::InvalidTarget {
                row,
                target,
                authority,
            } => write!(
                formatter,
                "target body-force batch row {row} target {target} violates {authority:?} authority"
            ),
            Self::ArithmeticOverflow => formatter.write_str("batch arithmetic overflow"),
            Self::InvalidInput(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for TargetBodyForceBatchError {}

impl From<String> for TargetBodyForceBatchError {
    fn from(message: String) -> Self {
        Self::InvalidInput(message)
    }
}

#[inline]
const fn map_reducer_authority_error(
    error: types::InvalidTargetPropagationAuthorityCode,
) -> TargetBodyForceBatchError {
    match error {
        types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow => {
            TargetBodyForceBatchError::ArithmeticOverflow
        }
        other => TargetBodyForceBatchError::ReducerAuthority(other),
    }
}

#[inline]
fn reserve_exact_or_overflow<T>(
    values: &mut Vec<T>,
    additional: usize,
) -> Result<(), TargetBodyForceBatchError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| TargetBodyForceBatchError::ArithmeticOverflow)
}

#[inline]
const fn shape_error(expected_rows: usize, actual_rows: usize) -> TargetBodyForceBatchError {
    TargetBodyForceBatchError::Shape {
        expected_rows,
        actual_rows,
    }
}

fn empty_fronts(
    count: usize,
) -> Result<Vec<ConstellationTransferFront>, TargetBodyForceBatchError> {
    let mut fronts = Vec::new();
    reserve_exact_or_overflow(&mut fronts, count)?;
    fronts.resize_with(count, ConstellationTransferFront::empty);
    Ok(fronts)
}

/// Shared immutable controls and target rows for one precomputed ECI batch.
#[derive(Clone)]
pub struct BatchEciConfiguration<'a> {
    pub targets_one_eci: &'a [f64],
    pub targets_two_eci: &'a [f64],
    pub epoch_jds: &'a [f64],
    pub max_time_s: f64,
    pub max_phase_dv: f64,
    pub max_transfer_dv: f64,
    pub max_revs: i32,
    pub min_perigee: f64,
    pub max_apogee: f64,
    pub pairs_to_verify: usize,
    pub sampling_mode: SamplingMode,
    pub search_depth: SearchDepthPolicy,
    pub distance_tol: f64,
    pub deployer_min_distance: f64,
    pub tof_penalty_weight: f64,
    pub revolution_cap: f64,
    pub target_propagation_authority: TargetPropagationAuthority,
    pub target_body_forces: &'a [[BodyForceConfig; 2]],
    pub force_config: Option<std::sync::Arc<lightyear_odeint_rs::types::ForceConfig>>,
    pub require_high_fidelity: bool,
    pub j2_closure_settings: solve::J2ClosureSettings,
    pub packed_coeffs: Option<std::sync::Arc<satpy_core::PackedGravityCoeffs>>,
    pub local_optimizer: TransferLocalOptimizerConfig,
    pub warm_starts: Option<&'a [Option<types::WarmStartData>]>,
    pub front_output_mode: solve::FrontOutputMode,
}

impl BatchEciConfiguration<'_> {
    #[inline]
    fn solve_configuration(
        &self,
        epoch_jd: f64,
        warm_start: Option<types::WarmStartData>,
    ) -> solve::ConstellationSolveConfiguration {
        solve::ConstellationSolveConfiguration {
            max_time_s: self.max_time_s,
            max_phase_dv: self.max_phase_dv,
            max_transfer_dv: self.max_transfer_dv,
            max_revs: self.max_revs,
            min_perigee: self.min_perigee,
            max_apogee: self.max_apogee,
            pairs_to_verify: self.pairs_to_verify,
            sampling_mode: self.sampling_mode,
            search_depth: self.search_depth,
            epoch_jd,
            distance_tol: self.distance_tol,
            deployer_min_distance: self.deployer_min_distance,
            tof_penalty_weight: self.tof_penalty_weight,
            revolution_cap: self.revolution_cap,
            target_propagation_authority: self.target_propagation_authority,
            force_config: self.force_config.clone(),
            require_high_fidelity: self.require_high_fidelity,
            j2_closure_settings: self.j2_closure_settings,
            packed_coeffs: self.packed_coeffs.clone(),
            local_optimizer: self.local_optimizer,
            warm_start,
        }
    }
}

/// One precomputed satellite batch plus its common target configuration.
#[derive(Clone)]
pub struct BatchEciRequest<'a> {
    pub satellite_eci: &'a [[f64; 6]],
    pub satellite_equinoctial: Option<&'a [[f64; 6]]>,
    pub satellite_count: usize,
    pub configuration: BatchEciConfiguration<'a>,
}

/// A population of precomputed satellite batches sharing targets and controls.
#[derive(Clone)]
pub struct PopulationBatchEciRequest<'a> {
    pub satellite_eci_population: &'a [[f64; 6]],
    pub satellite_equinoctial_population: Option<&'a [[f64; 6]]>,
    pub design_count: usize,
    pub satellite_count: usize,
    pub configuration: BatchEciConfiguration<'a>,
}

#[inline]
fn validate_candidate_search_authority(
    authority: TargetPropagationAuthority,
    force_config: Option<&lightyear_odeint_rs::types::ForceConfig>,
    require_high_fidelity: bool,
) -> Result<(), TargetBodyForceBatchError> {
    types::validate_candidate_search_authority(authority, force_config, require_high_fidelity)
        .map_err(TargetBodyForceBatchError::CandidateSearchAuthority)
}

#[inline]
fn validate_target_solve_authority(
    authority: TargetPropagationAuthority,
    rows: &[[BodyForceConfig; 2]],
    batch_size: usize,
    force_config: Option<&lightyear_odeint_rs::types::ForceConfig>,
) -> Result<(), TargetBodyForceBatchError> {
    validate_target_body_force_batch(authority, rows, batch_size)?;
    types::validate_target_propagation_force_config(authority, force_config)
        .map_err(TargetBodyForceBatchError::InvalidForceConfig)
}

#[inline]
fn validate_target_body_force_batch(
    authority: TargetPropagationAuthority,
    rows: &[[BodyForceConfig; 2]],
    batch_size: usize,
) -> Result<(), TargetBodyForceBatchError> {
    if rows.len() != batch_size {
        return Err(TargetBodyForceBatchError::Shape {
            expected_rows: batch_size,
            actual_rows: rows.len(),
        });
    }
    for (row, forces) in rows.iter().enumerate() {
        for (target, force) in forces.iter().enumerate() {
            if types::validate_target_body_force(authority, *force).is_err() {
                return Err(TargetBodyForceBatchError::InvalidTarget {
                    row,
                    target,
                    authority,
                });
            }
        }
    }
    Ok(())
}

/// Rayon `min_len` for the two FLAT `(…, event, pair)` drivers: none.
///
/// `batch_par_min_len_for` deliberately caps a `par_iter` at about two chunks
/// per worker, and it is right to do so where it was tuned — the EVENT
/// dimension, where a batch is 32 to 512 units and rayon's split cost is worth
/// amortizing. The flat drivers below iterate the PAIR dimension instead, where
/// one unit is a whole `solve_one_selected_pair`. Applying an amortization rule
/// to units that need no amortizing only throws away the partition.
///
/// Measured on a 16-design x 8-event x 8-pair population solve (1024 units,
/// 8 workers, so the adaptive rule returns `min_len = 64` and admits 16 chunks
/// for 8 workers). Per-unit cost p50 is 22 ms; rayon's split is sub-microsecond,
/// five orders of magnitude below it. Three interleaved paired reps in ONE
/// process, comparing WITHIN-RUN occupancy (`sum of unit time / (workers x
/// section wall)`) because the host's own wall clock is not comparable across
/// arms under load:
///
/// | rep | `min_len = 64` | `min_len = 1` |
/// |-----|----------------|---------------|
/// | 0   | 0.9725         | 0.9957        |
/// | 1   | 0.9768         | 0.9918        |
/// | 2   | 0.9834         | 0.9890        |
///
/// Mean gain 0.0146 against a 0.0088 spread over the three deltas, same sign
/// every rep, and the two arms do not overlap (worst `min_len = 1` rep beats
/// the best `min_len = 64` rep). Greedy-scheduling the same measured cost
/// vectors offline agrees and brackets it: modeled occupancy at granularity 64
/// is 0.892 / 0.949 / 0.959 over three cost vectors against 0.995 / 0.997 /
/// 0.997 at granularity 1.
///
/// Nothing was traded for it. All 86 integer fields of the summed
/// `VerifiedSupersetStageMetrics` are IDENTICAL between the two arms, including
/// `j2_propagate_state_count` (11,146,464 both ways) and the plan-eval and
/// phase-state cache tallies — so the finer partition does not split a warm
/// per-worker cache into cold ones, which was the one cost this change could
/// plausibly have hidden.
///
/// `batch_par_min_len_for` has since lost that caller too -- see
/// [`PHASE_A_PAR_MIN_LEN`] -- and is now retired to test-only, as was the
/// flat `min_len = 1` baseline it replaced.
const FLAT_DRIVER_PAR_MIN_LEN: usize = 1;

/// Rayon `min_len` for the Phase-A cell pre-pass: none, for the same reason.
///
/// This was `batch_par_min_len_for(cell_count)`, which caps the fan at about two
/// chunks per worker. That rule was tuned on the EVENT dimension, where a batch
/// is 32 to 512 units and rayon's split cost is worth amortizing. The Phase-A
/// unit is not an event: it is a whole `prepare_event_precomputed_row`, a plan
/// build and pair selection over `n_sats` satellites for one `(design, event)`
/// cell. Amortizing a sub-microsecond split against that throws away the
/// partition for nothing, which is exactly what R10 measured on the flat pair
/// drivers above.
///
/// A proportional floor is additionally the wrong SHAPE. Split cost is fixed per
/// split, so a floor of `cell_count / (2 * workers)` makes the chunk grow with
/// the batch and the imbalance grow with it. A floor should be a fixed item
/// count or nothing.
///
/// **The benefit here is UNMEASURED and is not claimed.** R10's occupancy A/B
/// was taken on the pair dimension, not this one, and no equivalent reading
/// exists for Phase A. What IS established is the bound and the safety: the
/// 2026-08-07 MF cost map puts Stage 1 at 85.49% of an MF cell and the Lambert
/// entry point at 91.29% of Stage 1's busy CPU, so everything else in Stage 1 --
/// Phase-A prep included -- is at most 7.4% of the cell, and only its imbalance
/// portion is recoverable. This is a structural correction with a small upside,
/// landed because it cannot cost anything, not because a wall was measured.
///
/// Bit-safety is by construction and not by measurement:
/// [`prepare_population_phase_a`] collects through `collect_into_vec`, which is
/// indexed, so canonical `(design * events + event)` order is restored before
/// `Result` selects an error. A partition change cannot reorder the output and
/// cannot change which error escapes.
const PHASE_A_PAR_MIN_LEN: usize = 1;

/// Pass 40B — adaptive chunk size: target ≈ `workers × 2` chunks per
/// batch. Returns a per-call `min_len` derived from the actual batch
/// size. This wins at every scale tested: events=32 keeps 8-worker
/// saturation (16 chunks of 2), events=128 amortizes rayon sync over
/// chunks of 8, events=512 over chunks of 32. The historical flat
/// `min_len = 1` baseline is retired; runtime paths use
/// `batch_par_min_len_for(batch)`.
#[inline]
#[cfg(test)]
fn batch_par_min_len_for(batch_size: usize) -> Result<usize, TargetBodyForceBatchError> {
    let workers = rayon::current_num_threads().max(1);
    let target_chunks = workers
        .checked_mul(2)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    let min_len = batch_size
        .checked_div(target_chunks)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    Ok(min_len.max(1))
}

#[inline]
const fn should_use_outer_batch_parallel_for(
    batch_size: usize,
    is_nested: bool,
    thread_count: usize,
) -> bool {
    !is_nested && batch_size >= BATCH_PAR_THRESHOLD && thread_count > 1
}

#[inline]
fn effective_flat_pair_width(
    pairs_to_verify: usize,
    n_sats: usize,
) -> Result<usize, TargetBodyForceBatchError> {
    let available_pairs = n_sats
        .checked_mul(2)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    if pairs_to_verify == 0 {
        Ok(available_pairs)
    } else {
        Ok(pairs_to_verify.min(available_pairs))
    }
}

#[inline]
fn estimated_flat_pair_work_units(
    batch_size: usize,
    pairs_to_verify: usize,
    n_sats: usize,
) -> Result<usize, TargetBodyForceBatchError> {
    batch_size
        .checked_mul(effective_flat_pair_width(pairs_to_verify, n_sats)?)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)
}

#[inline]
fn should_use_outer_batch_parallel_for_flat_work_units(
    batch_size: usize,
    flat_work_units: usize,
    is_nested: bool,
    thread_count: usize,
) -> Result<bool, TargetBodyForceBatchError> {
    if is_nested || thread_count <= 1 {
        return Ok(false);
    }
    let threshold = thread_count
        .checked_mul(2)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?
        .max(1);
    Ok(batch_size >= BATCH_PAR_THRESHOLD || flat_work_units >= threshold)
}

#[inline]
fn locate_prefix_slot(prefix_offsets: &[usize], global_idx: usize) -> Option<(usize, usize)> {
    let slot_idx = prefix_offsets
        .partition_point(|&offset| offset <= global_idx)
        .checked_sub(1)?;
    let slot_start = *prefix_offsets.get(slot_idx)?;
    global_idx
        .checked_sub(slot_start)
        .map(|pair_slot| (slot_idx, pair_slot))
}

/// Phase B + Phase C of the unified single-level parallelism over
/// `(event × pair)` (P1a, event-batch-local flatten).
///
/// Takes the per-event Phase-A plans (already prepared serially by the caller;
/// `None` = an event that yields an empty front), flattens every selected pair
/// across the batch into ONE rayon `par_iter` (the only rayon boundary — the L2
/// selected-pair `par_iter` stays suppressed, L3 stays OFF), and then reduces
/// each event SERIALLY in fixed `(event_idx, pair_slot)` order via
/// `solve::reduce_event` — the SAME order as the historical per-event serial
/// drain, so the output `Vec<ConstellationTransferFront>` is bit-identical in
/// its transfer rows.
///
/// The exception is `ConstellationTransferFront::verified_superset_metrics`,
/// whose f64 diagnostic fields (`j2_correction_residual_m_sum` in metres, the
/// `*_s` sub-timers) are reduction-order deterministic but not bit-identical
/// to the serial drain: the per-pair contributions are exact either way, but
/// the fold groups them per pair where serial ran one running sum, and `+` on
/// f64 is not associative. Integer counters are exact.
///
/// Per-worker `PlanContext`/`TransferMooWorkspace` are built via `map_init`
/// (nothing crosses threads but the shared immutable `&plans`). No `Python::`/
/// `with_gil`/pyo3 on the worker path.
fn run_flat_event_pair_driver(
    plans: Vec<Option<solve::EventPlan<'_>>>,
) -> Result<Vec<ConstellationTransferFront>, TargetBodyForceBatchError> {
    use rayon::prelude::*;

    let batch_size = plans.len();

    // Phase A result already in `plans`. Prefix offsets address the flat pair
    // dimension without materializing one work-unit struct per selected pair.
    let pair_offset_capacity = batch_size
        .checked_add(1)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    let mut pair_offsets = Vec::new();
    reserve_exact_or_overflow(&mut pair_offsets, pair_offset_capacity)?;
    pair_offsets.push(0);
    let mut total_pairs = 0usize;
    let mut per_event_results: Vec<Vec<Option<solve::PairFrontResult>>> = Vec::new();
    reserve_exact_or_overflow(&mut per_event_results, batch_size)?;
    for plan in &plans {
        let pair_count = plan
            .as_ref()
            .map_or(0, solve::EventPlan::selected_pair_count);
        let mut slots = Vec::new();
        reserve_exact_or_overflow(&mut slots, pair_count)?;
        for _ in 0..pair_count {
            slots.push(None);
        }
        total_pairs = total_pairs
            .checked_add(pair_count)
            .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
        pair_offsets.push(total_pairs);
        per_event_results.push(slots);
    }
    let j2_closure_settings = plans
        .iter()
        .filter_map(Option::as_ref)
        .find(|plan| plan.selected_pair_count() > 0)
        .map_or_else(
            solve::J2ClosureSettings::default,
            solve::EventPlan::j2_closure_settings,
        );

    // Phase B: flat par_iter over selected-pair index ranges. Chunk the range
    // so very large batches do not hold one full solved-result Vec in addition
    // to the per-event result slots.
    let mut chunk_start = 0usize;
    while chunk_start < total_pairs {
        let chunk_end = chunk_start
            .checked_add(FLAT_DRIVER_SOLVE_CHUNK_SIZE)
            .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?
            .min(total_pairs);
        let chunk_len = chunk_end
            .checked_sub(chunk_start)
            .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
        let min_len = FLAT_DRIVER_PAR_MIN_LEN;
        let mut solved: Vec<
            Result<
                Option<(usize, usize, Option<solve::PairFrontResult>)>,
                types::InvalidTargetPropagationAuthorityCode,
            >,
        > = Vec::new();
        reserve_exact_or_overflow(&mut solved, chunk_len)?;
        // Capacity is fallibly reserved to this indexed range's exact length,
        // so `collect_into_vec` does not need its infallible growth path.
        (chunk_start..chunk_end)
            .into_par_iter()
            .with_min_len(min_len)
            .map_init(
                || {
                    (
                        PlanContext::with_j2_closure_settings(j2_closure_settings),
                        solve::TransferMooWorkspace::new(),
                    )
                },
                |(local_ctx, moo_workspace), global_idx| {
                    let (event_idx, pair_slot) = locate_prefix_slot(&pair_offsets, global_idx)
                        .ok_or(types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                    // Only `Some` plans with >=1 selected pair produced work units.
                    let plan = plans
                        .get(event_idx)
                        .and_then(Option::as_ref)
                        .ok_or(types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                    let candidate = plan
                        .selected_pair(pair_slot)
                        .ok_or(types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                    let result = solve::solve_one_selected_pair(
                        plan,
                        candidate,
                        pair_slot,
                        local_ctx,
                        moo_workspace,
                    )?;
                    Ok(Some((event_idx, pair_slot, result)))
                },
            )
            .collect_into_vec(&mut solved);

        // Scatter results by their (event_idx, pair_slot) — deterministic and
        // independent of which worker computed each unit (rayon preserves order,
        // but we index explicitly to make that irrelevant).
        for solved_result in solved {
            let Some((event_idx, pair_slot, result)) =
                solved_result.map_err(map_reducer_authority_error)?
            else {
                continue;
            };
            let slot = per_event_results
                .get_mut(event_idx)
                .and_then(|event_results| event_results.get_mut(pair_slot))
                .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
            *slot = result;
        }
        chunk_start = chunk_end;
    }

    // Phase C: serial reduce per event, in fixed order.
    let flags = solve::EventReduceFlags {
        selected_pair_parallel: false,
        selected_pair_parallel_policy_enabled: false,
        outer_batch_parallel: true,
        rayon_current_num_threads: rayon::current_num_threads(),
    };
    let mut fronts = Vec::new();
    reserve_exact_or_overflow(&mut fronts, batch_size)?;
    for (plan, results) in plans.into_iter().zip(per_event_results) {
        fronts.push(plan.map_or_else(
            || Ok(ConstellationTransferFront::empty()),
            |plan| solve::reduce_event(&plan, results, flags).map_err(map_reducer_authority_error),
        )?);
    }
    Ok(fronts)
}

// Gate must match `build_population_event_pair_work_units` below, which is
// `any(parallel, test)`. This is a plain POD with no rayon dependency, so the
// wider gate costs nothing and lets the serial test target build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PopulationEventPairWork {
    design: usize,
    event: usize,
    pair: usize,
}

fn build_population_event_pair_work_units(
    pair_counts: &[Vec<usize>],
) -> Result<Vec<PopulationEventPairWork>, TargetBodyForceBatchError> {
    let total_pairs = pair_counts
        .iter()
        .flat_map(|design_counts| design_counts.iter())
        .copied()
        .try_fold(0usize, |total, pair_count| {
            total
                .checked_add(pair_count)
                .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)
        })?;
    let mut work_units = Vec::new();
    reserve_exact_or_overflow(&mut work_units, total_pairs)?;
    for (design_index, design_counts) in pair_counts.iter().enumerate() {
        for (event_index, &pair_count) in design_counts.iter().enumerate() {
            for pair_index in 0..pair_count {
                work_units.push(PopulationEventPairWork {
                    design: design_index,
                    event: event_index,
                    pair: pair_index,
                });
            }
        }
    }
    Ok(work_units)
}

type PopulationPairResultSlots = Vec<Vec<Vec<Option<solve::PairFrontResult>>>>;

fn initialize_population_pair_result_slots(
    plans: &[Vec<Option<solve::EventPlan<'_>>>],
) -> Result<PopulationPairResultSlots, TargetBodyForceBatchError> {
    let mut per_cell_results = Vec::new();
    reserve_exact_or_overflow(&mut per_cell_results, plans.len())?;
    for design_plans in plans {
        let mut design_slots = Vec::new();
        reserve_exact_or_overflow(&mut design_slots, design_plans.len())?;
        for plan in design_plans {
            let pair_count = plan
                .as_ref()
                .map_or(0, solve::EventPlan::selected_pair_count);
            // `PairFrontResult` is not `Clone`, so build fixed `None` slots
            // explicitly instead of using `vec![None; pair_count]`.
            let mut slots = Vec::new();
            reserve_exact_or_overflow(&mut slots, pair_count)?;
            for _ in 0..pair_count {
                slots.push(None);
            }
            design_slots.push(slots);
        }
        per_cell_results.push(design_slots);
    }
    Ok(per_cell_results)
}

fn population_pair_counts(
    slots: &PopulationPairResultSlots,
) -> Result<Vec<Vec<usize>>, TargetBodyForceBatchError> {
    let mut pair_counts = Vec::new();
    reserve_exact_or_overflow(&mut pair_counts, slots.len())?;
    for design_slots in slots {
        let mut design_counts = Vec::new();
        reserve_exact_or_overflow(&mut design_counts, design_slots.len())?;
        design_counts.extend(design_slots.iter().map(Vec::len));
        pair_counts.push(design_counts);
    }
    Ok(pair_counts)
}

fn reduce_population_event_pair_results(
    plans: Vec<Vec<Option<solve::EventPlan<'_>>>>,
    per_cell_results: PopulationPairResultSlots,
    flags: solve::EventReduceFlags,
) -> Result<Vec<Vec<ConstellationTransferFront>>, TargetBodyForceBatchError> {
    let mut fronts = Vec::new();
    reserve_exact_or_overflow(&mut fronts, plans.len())?;
    for (design_plans, design_results) in plans.into_iter().zip(per_cell_results) {
        let mut design_fronts = Vec::new();
        reserve_exact_or_overflow(&mut design_fronts, design_plans.len())?;
        for (plan, results) in design_plans.into_iter().zip(design_results) {
            design_fronts.push(plan.map_or_else(
                || Ok(ConstellationTransferFront::empty()),
                |plan| {
                    solve::reduce_event(&plan, results, flags).map_err(map_reducer_authority_error)
                },
            )?);
        }
        fronts.push(design_fronts);
    }
    Ok(fronts)
}

/// Phase B + Phase C of the unified single-level parallelism over the WHOLE
/// population: `(design × event × pair)` (P1, population-batch flatten).
///
/// The one-dimension-wider mirror of [`run_flat_event_pair_driver`]: it takes the
/// per-`(design, event)` Phase-A plans (already prepared by the population
/// pre-pass; `None` = an event that yields an empty front), flattens every
/// selected pair across EVERY `(design, event)` cell of the generation into ONE
/// rayon `par_iter` (the only rayon boundary — the L2 selected-pair `par_iter`
/// stays suppressed on every worker, L3 stays OFF), then reduces each
/// `(design, event)` SERIALLY in fixed `(design_idx, event_idx, pair_slot)` order
/// via `solve::reduce_event` — the SAME order as the historical per-design serial
/// drain, so each design's output `Vec<ConstellationTransferFront>` is
/// bit-identical to the per-design driver's in its transfer rows.
///
/// Same `verified_superset_metrics` caveat as [`run_flat_event_pair_driver`]:
/// the f64 diagnostic fields are reduction-order deterministic, not
/// bit-identical to the serial drain, because the fold groups per pair.
///
/// Per-worker `PlanContext`/`TransferMooWorkspace` are built via `map_init`
/// (so ~`workers` allocations regardless of the ~`design×event×pair` unit count;
/// nothing crosses threads but the shared immutable `&plans`). No Python or
/// direct worker I/O exists on this path. Pool-size-1 collapses the flat
/// `par_iter` to one chunk on the calling thread → byte-identical serial
/// reference (same property the per-design driver relies on).
fn run_flat_pop_event_pair_driver(
    // plans[design_idx][event_idx] — already prepared by the Phase-A pre-pass.
    plans: Vec<Vec<Option<solve::EventPlan<'_>>>>,
) -> Result<Vec<Vec<ConstellationTransferFront>>, TargetBodyForceBatchError> {
    use rayon::prelude::*;

    // Flatten the pair dimension across ALL (design, event) cells. One
    // pre-sized result slot per (design, event, pair) so the scatter below is a
    // pure index write (worker-order-independent). Work units are materialized
    // serially in public artifact order: (design_idx, event_idx, pair_idx).
    let mut per_cell_results = initialize_population_pair_result_slots(&plans)?;
    let pair_counts = population_pair_counts(&per_cell_results)?;
    let work_units = build_population_event_pair_work_units(&pair_counts)?;
    let total_pairs = work_units.len();
    let j2_closure_settings = plans
        .iter()
        .flatten()
        .filter_map(Option::as_ref)
        .find(|plan| plan.selected_pair_count() > 0)
        .map_or_else(
            solve::J2ClosureSettings::default,
            solve::EventPlan::j2_closure_settings,
        );

    // Phase B: flat par_iter over every selected pair in the generation,
    // represented as global index ranges. Chunk result collection to keep peak
    // memory bounded for large population batches.
    let mut chunk_start = 0usize;
    while chunk_start < total_pairs {
        let chunk_end = chunk_start
            .checked_add(FLAT_DRIVER_SOLVE_CHUNK_SIZE)
            .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?
            .min(total_pairs);
        let chunk_len = chunk_end
            .checked_sub(chunk_start)
            .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
        let work_chunk = work_units
            .get(chunk_start..chunk_end)
            .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
        let min_len = FLAT_DRIVER_PAR_MIN_LEN;
        let mut solved: Vec<
            Result<
                Option<(usize, usize, usize, Option<solve::PairFrontResult>)>,
                types::InvalidTargetPropagationAuthorityCode,
            >,
        > = Vec::new();
        reserve_exact_or_overflow(&mut solved, chunk_len)?;
        // Capacity is fallibly reserved to this indexed range's exact length,
        // so `collect_into_vec` does not need its infallible growth path.
        work_chunk
            .par_iter()
            .with_min_len(min_len)
            .map_init(
                || {
                    (
                        PlanContext::with_j2_closure_settings(j2_closure_settings),
                        solve::TransferMooWorkspace::new(),
                    )
                },
                |(local_ctx, moo_workspace), work| {
                    let design_idx = work.design;
                    let event_idx = work.event;
                    let pair_slot = work.pair;
                    // Only `Some` plans with >=1 selected pair produced work units.
                    let plan = plans
                        .get(design_idx)
                        .and_then(|event_plans| event_plans.get(event_idx))
                        .and_then(Option::as_ref)
                        .ok_or(types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                    let candidate = plan
                        .selected_pair(pair_slot)
                        .ok_or(types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
                    let result = solve::solve_one_selected_pair(
                        plan,
                        candidate,
                        pair_slot,
                        local_ctx,
                        moo_workspace,
                    )?;
                    Ok(Some((design_idx, event_idx, pair_slot, result)))
                },
            )
            .collect_into_vec(&mut solved);

        // Scatter results by their (design_idx, event_idx, pair_slot) —
        // deterministic and independent of which worker computed each unit.
        for solved_result in solved {
            let Some((design_idx, event_idx, pair_slot, result)) =
                solved_result.map_err(map_reducer_authority_error)?
            else {
                continue;
            };
            let slot = per_cell_results
                .get_mut(design_idx)
                .and_then(|event_rows| event_rows.get_mut(event_idx))
                .and_then(|pair_slots| pair_slots.get_mut(pair_slot))
                .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
            *slot = result;
        }
        chunk_start = chunk_end;
    }

    // Phase C: serial reduce per (design, event), in fixed order = today's
    // per-design serial drain order.
    let flags = solve::EventReduceFlags {
        selected_pair_parallel: false,
        selected_pair_parallel_policy_enabled: false,
        outer_batch_parallel: true,
        rayon_current_num_threads: rayon::current_num_threads(),
    };
    reduce_population_event_pair_results(plans, per_cell_results, flags)
}

fn prepare_population_phase_a<T, E, F>(
    cell_count: usize,
    n_sats: usize,
    prepare: F,
) -> Result<Vec<T>, E>
where
    T: Send,
    E: From<TargetBodyForceBatchError> + Send,
    F: Fn(usize, &mut SolveScratch) -> Result<T, E> + Send + Sync,
{
    use rayon::prelude::*;

    // Indexed collection restores canonical `(design * events + event)` order
    // before `Result` selects an error, independent of worker completion order.
    let min_len = PHASE_A_PAR_MIN_LEN;
    let mut prepared: Vec<Result<T, E>> = Vec::new();
    prepared
        .try_reserve_exact(cell_count)
        .map_err(|_| E::from(TargetBodyForceBatchError::ArithmeticOverflow))?;
    (0..cell_count)
        .into_par_iter()
        .with_min_len(min_len)
        .map_init(
            || SolveScratch::new(n_sats),
            |solve_scratch, cell_index| {
                let solve_scratch = solve_scratch
                    .as_mut()
                    .map_err(|_| E::from(TargetBodyForceBatchError::ArithmeticOverflow))?;
                prepare(cell_index, solve_scratch)
            },
        )
        .collect_into_vec(&mut prepared);
    // The parallel collector is indexed and has exactly `cell_count` slots,
    // but `Result<Vec<_>, _>::collect()` would allocate a new output Vec
    // through the infallible collection path. Reserve its complete capacity
    // before moving a single prepared value so an allocation failure remains a
    // typed batch-boundary failure with no partial result exposed.
    let mut values = Vec::new();
    values
        .try_reserve_exact(cell_count)
        .map_err(|_| E::from(TargetBodyForceBatchError::ArithmeticOverflow))?;
    for value in prepared {
        values.push(value?);
    }
    Ok(values)
}

fn empty_population_fronts(
    design_count: usize,
    batch_size: usize,
) -> Result<Vec<Vec<ConstellationTransferFront>>, TargetBodyForceBatchError> {
    let mut population = Vec::new();
    reserve_exact_or_overflow(&mut population, design_count)?;
    for _ in 0..design_count {
        population.push(empty_fronts(batch_size)?);
    }
    Ok(population)
}

#[cfg(test)]
#[inline]
fn propagate_satellite_mf_j2_from_kep(
    kep: &[f64; 6],
    dt_s: f64,
    out_eci: &mut [f64; 6],
    out_equ: &mut [f64; 6],
) {
    satpy_core::kep2eci_impl(kep, false, 0.0, 0.0, false, out_eci);
    satpy_core::eci2equinoc_impl(out_eci, 6, 0.0, 0.0, out_equ);
    let mut advanced_equ = [0.0_f64; 6];
    satpy_core::advance_equinoc_j2_impl(out_equ, dt_s, &mut advanced_equ);
    *out_equ = advanced_equ;
    satpy_core::equinoc2eci_impl(out_equ, 6, 0.0, 0.0, out_eci);
}

#[cfg(test)]
#[inline]
fn propagate_satellite_mf_j2_from_equ(
    equ: &[f64; 6],
    dt_s: f64,
    out_eci: &mut [f64; 6],
    out_equ: &mut [f64; 6],
) {
    let mut advanced_equ = [0.0_f64; 6];
    satpy_core::advance_equinoc_j2_impl(equ, dt_s, &mut advanced_equ);
    *out_equ = advanced_equ;
    satpy_core::equinoc2eci_impl(out_equ, 6, 0.0, 0.0, out_eci);
}

#[cfg(test)]
#[inline]
fn propagate_satellite_mf_j2_from_equ_block4(
    equ_block: &[f64; 24],
    dt_s: f64,
    out_eci_block: &mut [f64; 24],
    out_equ_block: &mut [f64; 24],
) {
    satpy_core::advance_equinoc_j2_batch_block4(equ_block, &[dt_s; 4], out_equ_block);
    satpy_core::equinoc2eci_simd4(out_equ_block, 0.0, 0.0, out_eci_block);
}

#[inline]
fn finish_event_timing(
    mut front: ConstellationTransferFront,
    front_output_mode: solve::FrontOutputMode,
    event_start: Instant,
    prep_s: f64,
    propagation_s: f64,
    constellation_solve_s: f64,
) -> Result<ConstellationTransferFront, TargetBodyForceBatchError> {
    if matches!(front_output_mode, solve::FrontOutputMode::VerifiedSuperset) {
        let metrics = &mut front.verified_superset_metrics;
        metrics.batch_event_total_s += event_start.elapsed().as_secs_f64();
        metrics.batch_event_prep_s += prep_s;
        metrics.batch_satellite_propagation_s += propagation_s;
        metrics.batch_constellation_solve_s += constellation_solve_s;
        {
            if rayon::current_thread_index().is_some() {
                metrics.outer_batch_parallel_event_count = metrics
                    .outer_batch_parallel_event_count
                    .checked_add(1)
                    .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
            }
        }
    }
    Ok(front)
}

fn warm_start_for_batch_row(
    warm_start_batch: Option<&[Option<types::WarmStartData>]>,
    row: usize,
) -> Result<Option<types::WarmStartData>, TargetBodyForceBatchError> {
    let Some(batch) = warm_start_batch else {
        return Ok(None);
    };
    let expected_rows = row
        .checked_add(1)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    batch
        .get(row)
        .copied()
        .ok_or_else(|| shape_error(expected_rows, batch.len()))
}

#[inline]
fn target_eci_arr(tgt_eci_vec: &[f64], row: usize) -> Result<&[f64; 6], TargetBodyForceBatchError> {
    let expected_rows = row
        .checked_add(1)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    let start = row
        .checked_mul(6)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    let end = start
        .checked_add(6)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    let values = tgt_eci_vec
        .get(start..end)
        .ok_or_else(|| shape_error(expected_rows, tgt_eci_vec.len() / 6))?;
    <&[f64; 6]>::try_from(values).map_err(|_| shape_error(expected_rows, tgt_eci_vec.len() / 6))
}

/// Phase A for the flat driver, precomputed-satellite variant.
///
/// Mirrors the propagation-free prelude of `batch_process_single_event_eci_precomputed`
/// (slice the per-event satellite states; equinoctial states are either taken
/// from the supplied batch or recomputed inside `solve::prepare_event`, which
/// uses the identical `eci2equinoc_impl`), then runs Phase A. The returned
/// `EventPlan` borrows the stable batch arena and owns its derived state.
/// It produces byte-identical satellite/target state to the serial wrapper.
fn prepare_event_precomputed_row<'a>(
    request: &BatchEciRequest<'a>,
    b: usize,
    solve_scratch: &mut SolveScratch,
) -> Result<Option<solve::EventPlan<'a>>, TargetBodyForceBatchError> {
    let configuration = &request.configuration;
    let expected_rows = b
        .checked_add(1)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    let tgt1_eci_arr = target_eci_arr(configuration.targets_one_eci, b)?;
    let tgt2_eci_arr = target_eci_arr(configuration.targets_two_eci, b)?;
    let epoch_jd = configuration
        .epoch_jds
        .get(b)
        .copied()
        .ok_or_else(|| shape_error(expected_rows, configuration.epoch_jds.len()))?;
    let target_body_forces = configuration
        .target_body_forces
        .get(b)
        .copied()
        .ok_or_else(|| shape_error(expected_rows, configuration.target_body_forces.len()))?;
    let sat_base = b
        .checked_mul(request.satellite_count)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    let sat_end = sat_base
        .checked_add(request.satellite_count)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    let sats_eci = request
        .satellite_eci
        .get(sat_base..sat_end)
        .ok_or_else(|| shape_error(sat_end, request.satellite_eci.len()))?;
    let sats_equ = match request.satellite_equinoctial {
        Some(eq_batch) => Some(
            eq_batch
                .get(sat_base..sat_end)
                .ok_or_else(|| shape_error(sat_end, eq_batch.len()))?,
        ),
        None => None,
    };

    solve::prepare_event(solve::EventPlanRequest {
        satellites: sats_eci,
        satellites_equ_cached: sats_equ,
        target1: tgt1_eci_arr,
        target2: tgt2_eci_arr,
        target_body_forces,
        configuration: configuration.solve_configuration(
            epoch_jd,
            warm_start_for_batch_row(configuration.warm_starts, b)?,
        ),
        scratch: Some(solve_scratch),
        front_output_mode: configuration.front_output_mode,
    })
    .map_err(map_reducer_authority_error)
}

fn batch_process_single_event_eci_precomputed(
    request: &BatchEciRequest<'_>,
    b: usize,
    coords_scratch: &mut CoordinateScratch,
    solve_scratch: &mut SolveScratch,
) -> Result<ConstellationTransferFront, TargetBodyForceBatchError> {
    let event_start = Instant::now();
    let configuration = &request.configuration;
    let expected_rows = b
        .checked_add(1)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    let tgt1_eci_arr = target_eci_arr(configuration.targets_one_eci, b)?;
    let tgt2_eci_arr = target_eci_arr(configuration.targets_two_eci, b)?;
    let epoch_jd = configuration
        .epoch_jds
        .get(b)
        .copied()
        .ok_or_else(|| shape_error(expected_rows, configuration.epoch_jds.len()))?;
    let target_body_forces = configuration
        .target_body_forces
        .get(b)
        .copied()
        .ok_or_else(|| shape_error(expected_rows, configuration.target_body_forces.len()))?;
    let sat_base = b
        .checked_mul(request.satellite_count)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    let sat_end = sat_base
        .checked_add(request.satellite_count)
        .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
    let sats_eci = request
        .satellite_eci
        .get(sat_base..sat_end)
        .ok_or_else(|| shape_error(sat_end, request.satellite_eci.len()))?;

    let sats_equ = if let Some(eq_batch) = request.satellite_equinoctial {
        eq_batch
            .get(sat_base..sat_end)
            .ok_or_else(|| shape_error(sat_end, eq_batch.len()))?
    } else {
        coords_scratch
            .prepare(request.satellite_count)
            .map_err(|_| TargetBodyForceBatchError::ArithmeticOverflow)?;
        let equ_vec = &mut coords_scratch.sats_equ;
        for sat_eci in sats_eci {
            let mut sat_equ = [0.0; 6];
            satpy_core::eci2equinoc_impl(sat_eci, 6, 0.0, 0.0, &mut sat_equ);
            equ_vec.push(sat_equ);
        }
        equ_vec.as_slice()
    };

    let prep_s = event_start.elapsed().as_secs_f64();
    let solve_start = Instant::now();
    let front = solve::constellation_solve_native_with_front_output_mode(solve::EventPlanRequest {
        satellites: sats_eci,
        satellites_equ_cached: Some(sats_equ),
        target1: tgt1_eci_arr,
        target2: tgt2_eci_arr,
        target_body_forces,
        configuration: configuration.solve_configuration(
            epoch_jd,
            warm_start_for_batch_row(configuration.warm_starts, b)?,
        ),
        scratch: Some(solve_scratch),
        front_output_mode: configuration.front_output_mode,
    })
    .map_err(map_reducer_authority_error)?;
    finish_event_timing(
        front,
        configuration.front_output_mode,
        event_start,
        prep_s,
        0.0,
        solve_start.elapsed().as_secs_f64(),
    )
}

/// Native batch solver that accepts precomputed satellite ECI/EQU per event.
///
/// # Errors
///
/// Returns an error when candidate-search, target-force, or propagation
/// authority controls are incompatible with the supplied batch, when a
/// reducer reports an authority failure, or when batch work arithmetic
/// overflows.
pub fn constellation_solve_batch_eci_precomputed(
    request: BatchEciRequest<'_>,
) -> Result<Vec<ConstellationTransferFront>, TargetBodyForceBatchError> {
    let mut request_for_rows = request;
    let batch_size = request_for_rows.configuration.epoch_jds.len();
    validate_candidate_search_authority(
        request_for_rows.configuration.target_propagation_authority,
        request_for_rows.configuration.force_config.as_deref(),
        request_for_rows.configuration.require_high_fidelity,
    )?;
    validate_target_solve_authority(
        request_for_rows.configuration.target_propagation_authority,
        request_for_rows.configuration.target_body_forces,
        batch_size,
        request_for_rows.configuration.force_config.as_deref(),
    )?;
    let fronts = (|| -> Result<Vec<ConstellationTransferFront>, TargetBodyForceBatchError> {
        if batch_size == 0 {
            return empty_fronts(batch_size);
        }
        let Some(target_values) = batch_size.checked_mul(6) else {
            return Err(TargetBodyForceBatchError::ArithmeticOverflow);
        };
        let Some(satellite_rows) = batch_size.checked_mul(request_for_rows.satellite_count) else {
            return Err(TargetBodyForceBatchError::ArithmeticOverflow);
        };
        if request_for_rows.configuration.targets_one_eci.len() < target_values {
            return Err(shape_error(
                batch_size,
                request_for_rows.configuration.targets_one_eci.len() / 6,
            ));
        }
        if request_for_rows.configuration.targets_two_eci.len() < target_values {
            return Err(shape_error(
                batch_size,
                request_for_rows.configuration.targets_two_eci.len() / 6,
            ));
        }
        if request_for_rows.satellite_eci.len() < satellite_rows {
            return Err(shape_error(
                satellite_rows,
                request_for_rows.satellite_eci.len(),
            ));
        }
        if let Some(eq_batch) = request_for_rows.satellite_equinoctial {
            if eq_batch.len() < satellite_rows {
                return Err(shape_error(satellite_rows, eq_batch.len()));
            }
        }
        if let Some(warm_starts) = request_for_rows.configuration.warm_starts {
            if warm_starts.len() < batch_size {
                return Err(shape_error(batch_size, warm_starts.len()));
            }
        }

        if request_for_rows.satellite_count == 0 {
            return empty_fronts(batch_size);
        }

        let pairs_to_verify = clamp_pairs_to_verify(
            request_for_rows.configuration.pairs_to_verify,
            request_for_rows.satellite_count,
        )?;
        request_for_rows.configuration.pairs_to_verify = pairs_to_verify;

        {
            let is_nested = rayon::current_thread_index().is_some();
            let flat_work_units = estimated_flat_pair_work_units(
                batch_size,
                pairs_to_verify,
                request_for_rows.satellite_count,
            )?;
            let use_outer_parallel = should_use_outer_batch_parallel_for_flat_work_units(
                batch_size,
                flat_work_units,
                is_nested,
                rayon::current_num_threads(),
            )?;

            if use_outer_parallel {
                // Unified single-level parallelism over (event × pair): Phase A
                // prepares every event serially (cheap screening/setup), then ONE
                // flat rayon `par_iter` solves every selected pair across the
                // batch, then Phase C reduces each event serially in order. This
                // replaces the event-only `into_par_iter` with a single flat
                // boundary; pool-size-1 still falls through to the serial path
                // below (byte-for-byte unchanged).
                let mut solve_scratch = SolveScratch::new(request_for_rows.satellite_count)
                    .map_err(|_| TargetBodyForceBatchError::ArithmeticOverflow)?;
                let mut plans = Vec::new();
                reserve_exact_or_overflow(&mut plans, batch_size)?;
                for b in 0..batch_size {
                    plans.push(prepare_event_precomputed_row(
                        &request_for_rows,
                        b,
                        &mut solve_scratch,
                    )?);
                }
                return run_flat_event_pair_driver(plans);
            }
        }

        let mut coords_scratch = CoordinateScratch::new(request_for_rows.satellite_count)
            .map_err(|_| TargetBodyForceBatchError::ArithmeticOverflow)?;
        let mut solve_scratch = SolveScratch::new(request_for_rows.satellite_count)
            .map_err(|_| TargetBodyForceBatchError::ArithmeticOverflow)?;
        let mut fronts = Vec::new();
        reserve_exact_or_overflow(&mut fronts, batch_size)?;
        for b in 0..batch_size {
            fronts.push(batch_process_single_event_eci_precomputed(
                &request_for_rows,
                b,
                &mut coords_scratch,
                &mut solve_scratch,
            )?);
        }
        Ok(fronts)
    })()?;
    Ok(fronts)
}

/// Native population batch solver for precomputed ECI/EQU satellites.
///
/// Shape contract:
/// - `satellites_*_population`: `(design_count * batch_size * n_sats)` rows.
/// - target/epoch batches are shared across designs and have length `batch_size`.
///
/// When admitted by the outer Rayon policy, Phase A prepares every
/// `(design,event)` cell, then the existing population flat driver solves one
/// native `(design,event,pair)` batch. Small/nested calls fall back to the
/// per-design batch solver, preserving current production behavior.
///
/// # Errors
///
/// Returns an error when candidate-search, target-force, or propagation
/// authority controls are incompatible with the supplied population batch,
/// when a reducer reports an authority failure, or when batch work arithmetic
/// overflows.
pub fn constellation_solve_population_batch_eci_precomputed(
    request: PopulationBatchEciRequest<'_>,
) -> Result<Vec<Vec<ConstellationTransferFront>>, TargetBodyForceBatchError> {
    let PopulationBatchEciRequest {
        satellite_eci_population: satellites_eci_population,
        satellite_equinoctial_population: satellites_equ_population,
        design_count,
        satellite_count: n_sats,
        configuration,
    } = request;
    let batch_size = configuration.epoch_jds.len();
    validate_candidate_search_authority(
        configuration.target_propagation_authority,
        configuration.force_config.as_deref(),
        configuration.require_high_fidelity,
    )?;
    validate_target_solve_authority(
        configuration.target_propagation_authority,
        configuration.target_body_forces,
        batch_size,
        configuration.force_config.as_deref(),
    )?;
    let configuration_for_serial = configuration.clone();
    let fronts = (|| -> Result<_, TargetBodyForceBatchError> {
        if design_count == 0 {
            return Ok(Vec::new());
        }
        if batch_size == 0 {
            return empty_population_fronts(design_count, batch_size);
        }
        let Some(target_values) = batch_size.checked_mul(6) else {
            return Err(TargetBodyForceBatchError::ArithmeticOverflow);
        };
        if configuration.targets_one_eci.len() < target_values {
            return Err(shape_error(
                batch_size,
                configuration.targets_one_eci.len() / 6,
            ));
        }
        if configuration.targets_two_eci.len() < target_values {
            return Err(shape_error(
                batch_size,
                configuration.targets_two_eci.len() / 6,
            ));
        }
        let Some(rows_per_design) = batch_size.checked_mul(n_sats) else {
            return Err(TargetBodyForceBatchError::ArithmeticOverflow);
        };
        let Some(total_rows) = rows_per_design.checked_mul(design_count) else {
            return Err(TargetBodyForceBatchError::ArithmeticOverflow);
        };
        if satellites_eci_population.len() < total_rows {
            return Err(shape_error(total_rows, satellites_eci_population.len()));
        }
        if let Some(eq_population) = satellites_equ_population {
            if eq_population.len() < total_rows {
                return Err(shape_error(total_rows, eq_population.len()));
            }
        }

        if n_sats == 0 {
            return empty_population_fronts(design_count, batch_size);
        }

        let pairs_to_verify = clamp_pairs_to_verify(configuration.pairs_to_verify, n_sats)?;

        {
            let phase_a_configuration = BatchEciConfiguration {
                pairs_to_verify,
                warm_starts: None,
                ..configuration_for_serial.clone()
            };
            let Some(cell_count) = design_count.checked_mul(batch_size) else {
                return Err(TargetBodyForceBatchError::ArithmeticOverflow);
            };
            let flat_work_units = cell_count
                .checked_mul(effective_flat_pair_width(pairs_to_verify, n_sats)?)
                .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
            let is_nested = rayon::current_thread_index().is_some();
            let use_population_flat = should_use_outer_batch_parallel_for(
                cell_count,
                is_nested,
                rayon::current_num_threads(),
            ) || should_use_outer_batch_parallel_for_flat_work_units(
                cell_count,
                flat_work_units,
                is_nested,
                rayon::current_num_threads(),
            )?;

            if use_population_flat {
                let prepared =
                    prepare_population_phase_a(cell_count, n_sats, |cell_index, solve_scratch| {
                        let design_idx = cell_index
                            .checked_div(batch_size)
                            .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
                        let b = cell_index
                            .checked_rem(batch_size)
                            .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
                        let design_base = design_idx
                            .checked_mul(rows_per_design)
                            .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
                        let design_end = design_base
                            .checked_add(rows_per_design)
                            .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
                        let design_eci = satellites_eci_population
                            .get(design_base..design_end)
                            .ok_or_else(|| {
                                shape_error(total_rows, satellites_eci_population.len())
                            })?;
                        let design_equ = match satellites_equ_population {
                            Some(eq_population) => Some(
                                eq_population
                                    .get(design_base..design_end)
                                    .ok_or_else(|| shape_error(total_rows, eq_population.len()))?,
                            ),
                            None => None,
                        };
                        let batch_request = BatchEciRequest {
                            satellite_eci: design_eci,
                            satellite_equinoctial: design_equ,
                            satellite_count: n_sats,
                            configuration: phase_a_configuration.clone(),
                        };
                        prepare_event_precomputed_row(&batch_request, b, solve_scratch)
                    })?;

                let mut prepared = prepared.into_iter();
                let mut population_plans = Vec::new();
                reserve_exact_or_overflow(&mut population_plans, design_count)?;
                for _ in 0..design_count {
                    let mut design_plans = Vec::new();
                    reserve_exact_or_overflow(&mut design_plans, batch_size)?;
                    for _ in 0..batch_size {
                        design_plans.push(
                            prepared
                                .next()
                                .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?,
                        );
                    }
                    population_plans.push(design_plans);
                }
                return run_flat_pop_event_pair_driver(population_plans);
            }
        }

        let mut fronts = Vec::new();
        reserve_exact_or_overflow(&mut fronts, design_count)?;
        for design_idx in 0..design_count {
            let design_base = design_idx
                .checked_mul(rows_per_design)
                .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
            let design_end = design_base
                .checked_add(rows_per_design)
                .ok_or(TargetBodyForceBatchError::ArithmeticOverflow)?;
            let design_eci = satellites_eci_population
                .get(design_base..design_end)
                .ok_or_else(|| shape_error(total_rows, satellites_eci_population.len()))?;
            let design_equ = match satellites_equ_population {
                Some(eq_population) => Some(
                    eq_population
                        .get(design_base..design_end)
                        .ok_or_else(|| shape_error(total_rows, eq_population.len()))?,
                ),
                None => None,
            };
            fronts.push(constellation_solve_batch_eci_precomputed(
                BatchEciRequest {
                    satellite_eci: design_eci,
                    satellite_equinoctial: design_equ,
                    satellite_count: n_sats,
                    configuration: BatchEciConfiguration {
                        pairs_to_verify,
                        warm_starts: None,
                        ..configuration_for_serial.clone()
                    },
                },
            )?);
        }
        Ok(fronts)
    })()?;
    Ok(fronts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(
        clippy::float_cmp,
        reason = "fixture pins exact MF configuration controls"
    )]
    #[test]
    fn mf_batch_configuration_keeps_typed_fixture_controls() {
        let target_forces = mf_target_body_force_batch(1);
        let configuration =
            mf_batch_configuration(&[1.0; 6], &[2.0; 6], &[2_460_000.5], &target_forces);

        assert_eq!(configuration.max_time_s, 86_400.0);
        assert_eq!(configuration.max_phase_dv, 2.0);
        assert_eq!(configuration.max_transfer_dv, 5.0);
        assert_eq!(configuration.max_revs, 0);
        assert_eq!(configuration.pairs_to_verify, 2);
        assert_eq!(
            configuration.target_propagation_authority,
            TargetPropagationAuthority::MfJ2
        );
        assert!(configuration.force_config.is_none());
        assert!(!configuration.require_high_fidelity);
        assert_eq!(
            configuration.front_output_mode,
            solve::FrontOutputMode::TransferPareto
        );
    }

    #[test]
    fn finish_event_timing_rejects_outer_parallel_metric_overflow() -> anyhow::Result<()> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
        let mut front = ConstellationTransferFront::empty();
        front
            .verified_superset_metrics
            .outer_batch_parallel_event_count = usize::MAX;

        let result = pool.install(|| {
            finish_event_timing(
                front,
                solve::FrontOutputMode::VerifiedSuperset,
                Instant::now(),
                0.0,
                0.0,
                0.0,
            )
        });
        anyhow::ensure!(
            matches!(result, Err(TargetBodyForceBatchError::ArithmeticOverflow)),
            "outer-batch metric overflow must remain a typed batch error"
        );
        Ok(())
    }

    #[test]
    fn result_collection_reservation_rejects_impossible_capacity() {
        let mut results = Vec::<usize>::new();

        assert_eq!(
            reserve_exact_or_overflow(&mut results, usize::MAX),
            Err(TargetBodyForceBatchError::ArithmeticOverflow)
        );
        assert!(results.is_empty());
    }

    fn hf_target_body_force_batch(batch_size: usize) -> Vec<[BodyForceConfig; 2]> {
        vec![
            [BodyForceConfig::high_fidelity(types::BodyRole::DiagnosticTarget, 0.01, 2.2, 1.0,); 2];
            batch_size
        ]
    }

    fn hf_force_config(
        sph_order: usize,
        force_flags: i32,
        atm_model: i32,
    ) -> std::sync::Arc<lightyear_odeint_rs::types::ForceConfig> {
        std::sync::Arc::new(lightyear_odeint_rs::types::ForceConfig {
            sph_order,
            force_flags,
            atm_model,
            target_propagation_mode: TargetPropagationAuthority::HighFidelity
                .as_force_config_code(),
            ..Default::default()
        })
    }

    fn mf_target_body_forces() -> [BodyForceConfig; 2] {
        [BodyForceConfig::j2(types::BodyRole::DiagnosticTarget); 2]
    }

    fn mf_target_body_force_batch(batch_size: usize) -> Vec<[BodyForceConfig; 2]> {
        vec![mf_target_body_forces(); batch_size]
    }

    fn high_fidelity_rejection_configuration<'a>(
        targets_one_eci: &'a [f64],
        targets_two_eci: &'a [f64],
        epoch_jds: &'a [f64],
        target_body_forces: &'a [[BodyForceConfig; 2]],
    ) -> BatchEciConfiguration<'a> {
        BatchEciConfiguration {
            targets_one_eci,
            targets_two_eci,
            epoch_jds,
            max_time_s: 60.0,
            max_phase_dv: 0.5,
            max_transfer_dv: 2.0,
            max_revs: 0,
            min_perigee: 6_578.14,
            max_apogee: 41_378.14,
            pairs_to_verify: 0,
            sampling_mode: SamplingMode::Fast,
            search_depth: SearchDepthPolicy::default(),
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            tof_penalty_weight: 0.1,
            revolution_cap: 2.0,
            target_propagation_authority: TargetPropagationAuthority::HighFidelity,
            target_body_forces,
            force_config: None,
            require_high_fidelity: false,
            j2_closure_settings: solve::J2ClosureSettings::default(),
            packed_coeffs: None,
            local_optimizer: TransferLocalOptimizerConfig::default(),
            warm_starts: None,
            front_output_mode: solve::FrontOutputMode::TransferPareto,
        }
    }

    fn mf_batch_configuration<'a>(
        targets_one_eci: &'a [f64],
        targets_two_eci: &'a [f64],
        epoch_jds: &'a [f64],
        target_body_forces: &'a [[BodyForceConfig; 2]],
    ) -> BatchEciConfiguration<'a> {
        BatchEciConfiguration {
            targets_one_eci,
            targets_two_eci,
            epoch_jds,
            max_time_s: 86_400.0,
            max_phase_dv: 2.0,
            max_transfer_dv: 5.0,
            max_revs: 0,
            min_perigee: 6_578.14,
            max_apogee: 100_000.0,
            pairs_to_verify: 2,
            sampling_mode: SamplingMode::Fast,
            search_depth: SearchDepthPolicy::default(),
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            tof_penalty_weight: 0.1,
            revolution_cap: 2.0,
            target_propagation_authority: TargetPropagationAuthority::MfJ2,
            target_body_forces,
            force_config: None,
            require_high_fidelity: false,
            j2_closure_settings: solve::J2ClosureSettings::default(),
            packed_coeffs: None,
            local_optimizer: TransferLocalOptimizerConfig::default(),
            warm_starts: None,
            front_output_mode: solve::FrontOutputMode::TransferPareto,
        }
    }

    #[test]
    fn prefix_slot_lookup_rejects_empty_offsets() {
        assert_eq!(locate_prefix_slot(&[], 0), None);
    }

    #[test]
    fn target_eci_row_rejects_truncated_input_without_panicking() {
        let lookup = std::panic::catch_unwind(|| target_eci_arr(&[0.0; 5], 0));
        assert!(matches!(
            lookup,
            Ok(Err(TargetBodyForceBatchError::Shape {
                expected_rows: 1,
                actual_rows: 0,
            }))
        ));
    }

    #[test]
    fn target_body_force_batch_reports_typed_shape_and_row_errors() {
        assert_eq!(
            validate_target_body_force_batch(TargetPropagationAuthority::HighFidelity, &[], 1,),
            Err(TargetBodyForceBatchError::Shape {
                expected_rows: 1,
                actual_rows: 0,
            })
        );

        let invalid = [[BodyForceConfig::gravity_only(types::BodyRole::DiagnosticTarget); 2]];
        assert_eq!(
            validate_target_body_force_batch(TargetPropagationAuthority::HighFidelity, &invalid, 1,),
            Err(TargetBodyForceBatchError::InvalidTarget {
                row: 0,
                target: 0,
                authority: TargetPropagationAuthority::HighFidelity,
            })
        );
    }

    #[test]
    fn precomputed_batch_rejects_undersized_satellite_rows_before_emitting_fronts() {
        let target_forces = mf_target_body_force_batch(1);
        let error = constellation_solve_batch_eci_precomputed(BatchEciRequest {
            satellite_eci: &[],
            satellite_equinoctial: None,
            satellite_count: 1,
            configuration: mf_batch_configuration(
                &[7_000.0; 6],
                &[7_100.0; 6],
                &[2_460_000.5],
                &target_forces,
            ),
        })
        .expect_err("undersized satellite rows must not become empty fronts");

        assert_eq!(
            error,
            TargetBodyForceBatchError::Shape {
                expected_rows: 1,
                actual_rows: 0,
            }
        );
    }

    #[test]
    fn precomputed_population_rejects_undersized_satellite_rows_before_emitting_fronts() {
        let target_forces = mf_target_body_force_batch(1);
        let error =
            constellation_solve_population_batch_eci_precomputed(PopulationBatchEciRequest {
                satellite_eci_population: &[],
                satellite_equinoctial_population: None,
                design_count: 1,
                satellite_count: 1,
                configuration: mf_batch_configuration(
                    &[7_000.0; 6],
                    &[7_100.0; 6],
                    &[2_460_000.5],
                    &target_forces,
                ),
            })
            .expect_err("undersized population rows must not become empty fronts");

        assert_eq!(
            error,
            TargetBodyForceBatchError::Shape {
                expected_rows: 1,
                actual_rows: 0,
            }
        );
    }

    #[test]
    fn parallel_population_rejects_undersized_satellite_rows_before_emitting_fronts(
    ) -> anyhow::Result<()> {
        let target_forces = mf_target_body_force_batch(4);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
        let result = pool.install(|| {
            constellation_solve_population_batch_eci_precomputed(PopulationBatchEciRequest {
                satellite_eci_population: &[],
                satellite_equinoctial_population: None,
                design_count: 1,
                satellite_count: 1,
                configuration: mf_batch_configuration(
                    &[7_000.0; 24],
                    &[7_100.0; 24],
                    &[2_460_000.5; 4],
                    &target_forces,
                ),
            })
        });
        let error = result.err().ok_or_else(|| {
            anyhow::anyhow!("undersized parallel population rows must not emit fronts")
        })?;

        anyhow::ensure!(
            error
                == TargetBodyForceBatchError::Shape {
                    expected_rows: 4,
                    actual_rows: 0,
                },
            "undersized parallel population returned unexpected error: {error:?}"
        );
        Ok(())
    }

    #[test]
    fn public_precomputed_batch_rejects_high_fidelity_search_before_empty_front() {
        let error = constellation_solve_batch_eci_precomputed(BatchEciRequest {
            satellite_eci: &[],
            satellite_equinoctial: None,
            satellite_count: 0,
            configuration: high_fidelity_rejection_configuration(&[], &[], &[2_460_000.5], &[]),
        })
        .expect_err("missing target force rows must not become empty scientific fronts");

        assert_eq!(
            error,
            TargetBodyForceBatchError::CandidateSearchAuthority(
                types::CandidateSearchAuthorityError::UnsupportedHighFidelityTransferSearch,
            )
        );
    }

    #[test]
    fn public_precomputed_batch_rejects_four_by_four_hf_target_gravity() {
        let forces = hf_target_body_force_batch(1);
        let result = constellation_solve_batch_eci_precomputed(BatchEciRequest {
            satellite_eci: &[],
            satellite_equinoctial: None,
            satellite_count: 0,
            configuration: BatchEciConfiguration {
                force_config: Some(hf_force_config(4, 3, 3)),
                require_high_fidelity: true,
                ..high_fidelity_rejection_configuration(&[], &[], &[2_460_000.5], &forces)
            },
        });

        assert!(matches!(
            result,
            Err(TargetBodyForceBatchError::CandidateSearchAuthority(
                types::CandidateSearchAuthorityError::UnsupportedHighFidelityTransferSearch,
            ))
        ));
    }

    #[test]
    fn public_population_precomputed_rejects_hf_target_third_body_flags() {
        let forces = hf_target_body_force_batch(1);
        let third_body_flags = 3 | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY;
        let result =
            constellation_solve_population_batch_eci_precomputed(PopulationBatchEciRequest {
                satellite_eci_population: &[],
                satellite_equinoctial_population: None,
                design_count: 1,
                satellite_count: 0,
                configuration: BatchEciConfiguration {
                    force_config: Some(hf_force_config(5, third_body_flags, 3)),
                    require_high_fidelity: true,
                    ..high_fidelity_rejection_configuration(&[], &[], &[2_460_000.5], &forces)
                },
            });

        assert!(matches!(
            result,
            Err(TargetBodyForceBatchError::CandidateSearchAuthority(
                types::CandidateSearchAuthorityError::UnsupportedHighFidelityTransferSearch,
            ))
        ));
    }

    #[test]
    fn public_precomputed_batch_rejects_wrong_hf_target_atmosphere() {
        let forces = hf_target_body_force_batch(1);
        let result = constellation_solve_batch_eci_precomputed(BatchEciRequest {
            satellite_eci: &[],
            satellite_equinoctial: None,
            satellite_count: 0,
            configuration: BatchEciConfiguration {
                force_config: Some(hf_force_config(5, types::HIGH_FIDELITY_FORCE_FLAGS, 3)),
                require_high_fidelity: true,
                ..high_fidelity_rejection_configuration(&[], &[], &[2_460_000.5], &forces)
            },
        });

        assert!(matches!(
            result,
            Err(TargetBodyForceBatchError::CandidateSearchAuthority(
                types::CandidateSearchAuthorityError::UnsupportedHighFidelityTransferSearch,
            ))
        ));
    }

    #[test]
    fn test_propagate_satellite_mf_j2_from_equ_matches_kep_seed_path() {
        let kep = [7000.0, 0.001, 0.6, 0.1, 0.2, 0.3];
        let dt_s = 600.0_f64;
        let mut base_eci = [0.0_f64; 6];
        satpy_core::kep2eci_impl(&kep, false, 0.0, 0.0, false, &mut base_eci);
        let mut base_equ = [0.0_f64; 6];
        satpy_core::eci2equinoc_impl(&base_eci, 6, 0.0, 0.0, &mut base_equ);

        let mut eci_from_kep = [0.0_f64; 6];
        let mut equ_from_kep = [0.0_f64; 6];
        propagate_satellite_mf_j2_from_kep(&kep, dt_s, &mut eci_from_kep, &mut equ_from_kep);

        let mut eci_from_equ = [0.0_f64; 6];
        let mut equ_from_equ = [0.0_f64; 6];
        propagate_satellite_mf_j2_from_equ(&base_equ, dt_s, &mut eci_from_equ, &mut equ_from_equ);

        let tol = 1.0e-6_f64;
        for (lhs, rhs) in eci_from_kep.iter().zip(eci_from_equ.iter()) {
            assert!(
                (*lhs - *rhs).abs() <= tol,
                "eci mismatch: lhs={lhs} rhs={rhs}"
            );
        }
        for (lhs, rhs) in equ_from_kep.iter().zip(equ_from_equ.iter()) {
            assert!(
                (*lhs - *rhs).abs() <= tol,
                "equ mismatch: lhs={lhs} rhs={rhs}"
            );
        }
    }

    #[test]
    fn test_propagate_satellite_mf_j2_from_equ_block4_matches_scalar() {
        let equ_states = [
            [7000.0, 0.001, 0.002, 0.10, 0.20, 0.30],
            [7020.0, 0.002, 0.003, 0.40, 0.50, 0.60],
            [7040.0, 0.003, 0.004, 0.70, 0.80, 0.90],
            [7060.0, 0.004, 0.005, 1.00, 1.10, 1.20],
        ];
        let dt_s = 900.0_f64;
        let mut equ_block = [0.0_f64; 24];
        for (equ_block_lane, equ) in equ_block.chunks_exact_mut(6).zip(&equ_states) {
            equ_block_lane.copy_from_slice(equ);
        }

        let mut block_eci = [0.0_f64; 24];
        let mut block_equ = [0.0_f64; 24];
        propagate_satellite_mf_j2_from_equ_block4(&equ_block, dt_s, &mut block_eci, &mut block_equ);

        let tol = 1.0e-6_f64;
        for (idx, ((block_eci_lane, block_equ_lane), equ)) in block_eci
            .chunks_exact(6)
            .zip(block_equ.chunks_exact(6))
            .zip(&equ_states)
            .enumerate()
        {
            let mut scalar_eci = [0.0_f64; 6];
            let mut scalar_equ = [0.0_f64; 6];
            propagate_satellite_mf_j2_from_equ(equ, dt_s, &mut scalar_eci, &mut scalar_equ);
            for (
                component,
                (
                    (&block_eci_component, &scalar_eci_component),
                    (&block_equ_component, &scalar_equ_component),
                ),
            ) in block_eci_lane
                .iter()
                .zip(scalar_eci.iter())
                .zip(block_equ_lane.iter().zip(scalar_equ.iter()))
                .enumerate()
            {
                assert!(
                    (block_eci_component - scalar_eci_component).abs() <= tol,
                    "eci mismatch lane={idx} component={component} \
                     lhs={block_eci_component} rhs={scalar_eci_component}"
                );
                assert!(
                    (block_equ_component - scalar_equ_component).abs() <= tol,
                    "equ mismatch lane={idx} component={component} \
                     lhs={block_equ_component} rhs={scalar_equ_component}"
                );
            }
        }
    }

    #[test]
    fn test_batch_parallel_policy_enables_dynamic_beta_event_batches() {
        assert!(should_use_outer_batch_parallel_for(16, false, 8));
        assert!(should_use_outer_batch_parallel_for(8, false, 8));
        assert!(!should_use_outer_batch_parallel_for(3, false, 8));
        assert!(!should_use_outer_batch_parallel_for(8, true, 8));
    }

    #[test]
    fn test_batch_parallel_policy_requires_multi_thread_pool() {
        assert!(!should_use_outer_batch_parallel_for(8, false, 1));
    }

    #[test]
    fn test_batch_parallel_policy_threshold_is_inclusive() {
        assert!(!should_use_outer_batch_parallel_for(
            BATCH_PAR_THRESHOLD - 1,
            false,
            8,
        ));
        assert!(should_use_outer_batch_parallel_for(
            BATCH_PAR_THRESHOLD,
            false,
            8,
        ));
    }

    #[test]
    fn test_flat_pair_policy_admits_small_event_batches_with_enough_pair_work() {
        assert!(should_use_outer_batch_parallel_for_flat_work_units(
            BATCH_PAR_THRESHOLD - 1,
            16,
            false,
            8,
        )
        .expect("fixed test work count must not overflow"));
        assert!(!should_use_outer_batch_parallel_for_flat_work_units(
            BATCH_PAR_THRESHOLD - 1,
            15,
            false,
            8,
        )
        .expect("fixed test work count must not overflow"));
        assert!(!should_use_outer_batch_parallel_for_flat_work_units(
            BATCH_PAR_THRESHOLD - 1,
            16,
            true,
            8,
        )
        .expect("fixed test work count must not overflow"));
        assert!(!should_use_outer_batch_parallel_for_flat_work_units(
            BATCH_PAR_THRESHOLD - 1,
            16,
            false,
            1,
        )
        .expect("fixed test work count must not overflow"));
    }

    #[test]
    fn test_batch_parallel_policy_disables_nested_pool() {
        assert!(!should_use_outer_batch_parallel_for(8, true, 8));
    }

    #[test]
    fn test_batch_par_min_len_for_targets_workers_x2_chunks() {
        // Pass 40B — adaptive chunk size returns `batch_size / (workers * 2)`,
        // clamped to ≥ 1. With rayon's default thread pool the workers count
        // is the machine's logical cores; we don't know the exact value here,
        // but we can pin the relationship: the returned min_len, when
        // multiplied by `workers * 2`, should be ≤ batch_size.
        let workers = rayon::current_num_threads().max(1);
        let target_chunks = workers
            .checked_mul(2)
            .expect("test worker count times two must fit usize")
            .max(1);
        for batch in [4_usize, 8, 32, 128, 512] {
            let min_len = batch_par_min_len_for(batch)
                .expect("fixed test batch arithmetic must not overflow");
            assert!(min_len >= 1, "min_len must be at least 1 for batch={batch}");
            // The total chunks rayon will produce is at most `batch_size / min_len`.
            // We expect that to be ~target_chunks (modulo integer rounding).
            assert!(
                min_len
                    .checked_mul(target_chunks)
                    .expect("fixed test work count must not overflow")
                    <= batch.max(target_chunks),
                "min_len={min_len} × workers*2={target_chunks} exceeds batch={batch}"
            );
        }
    }

    #[test]
    fn population_phase_a_parallel_preserves_input_order_and_first_error() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("build four-thread Phase-A pool");

        let ordered = pool
            .install(|| {
                prepare_population_phase_a(32, 1, |ordinal, _scratch| {
                    Ok::<_, TargetBodyForceBatchError>(ordinal)
                })
            })
            .expect("infallible Phase-A fixture");
        assert_eq!(ordered, (0..32).collect::<Vec<_>>());

        let error = pool
            .install(|| {
                prepare_population_phase_a(32, 1, |ordinal, _scratch| {
                    if ordinal == 7 || ordinal == 19 {
                        Err(TargetBodyForceBatchError::InvalidInput(ordinal.to_string()))
                    } else {
                        Ok(ordinal)
                    }
                })
            })
            .expect_err("fixture must return an error");
        assert_eq!(
            error,
            TargetBodyForceBatchError::InvalidInput("7".to_owned()),
            "parallel Phase A must report first input error"
        );
    }

    #[test]
    fn population_phase_a_parallel_uses_multiple_workers() {
        use std::collections::BTreeSet;
        use std::sync::{Arc, Mutex};
        use std::time::Duration;

        let workers = Arc::new(Mutex::new(BTreeSet::new()));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("build four-thread Phase-A pool");
        let observed_workers = Arc::clone(&workers);
        let ordered = pool
            .install(|| {
                prepare_population_phase_a(32, 1, move |ordinal, _scratch| {
                    observed_workers.lock().expect("worker set lock").insert(
                        rayon::current_thread_index()
                            .expect("Phase-A callback must run in Rayon worker"),
                    );
                    std::thread::sleep(Duration::from_millis(2));
                    Ok::<_, TargetBodyForceBatchError>(ordinal)
                })
            })
            .expect("infallible Phase-A fixture");

        assert_eq!(ordered, (0..32).collect::<Vec<_>>());
        assert!(
            workers.lock().expect("worker set lock").len() >= 2,
            "four-thread Phase-A pool must engage multiple workers"
        );
    }

    #[test]
    fn public_population_phase_a_is_bit_exact_w1_w8() {
        const WIDTH_ENV: &str = "ND_PHASE_A_PUBLIC_TEST_WIDTH";

        if std::env::var_os(WIDTH_ENV).is_none() {
            let identities = [1usize, 8]
                .into_iter()
                .map(|width| {
                    let output = std::process::Command::new(
                        std::env::current_exe().expect("locate batch_eci test binary"),
                    )
                    .args([
                        "batch_eci::tests::public_population_phase_a_is_bit_exact_w1_w8",
                        "--exact",
                        "--nocapture",
                    ])
                    .env(WIDTH_ENV, width.to_string())
                    .env("RUST_TEST_THREADS", "1")
                    .output()
                    .expect("run isolated Phase-A width child");
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    assert!(
                        output.status.success(),
                        "public population width {width} failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
                    );
                    stdout
                        .lines()
                        .chain(stderr.lines())
                        .find_map(|line| {
                            line.split_once("PHASE_A_PUBLIC_IDENTITY=")
                                .map(|(_, identity)| identity)
                        })
                        .expect("child must emit public Phase-A identity")
                        .to_owned()
                })
                .collect::<Vec<_>>();
            assert_eq!(identities.first(), identities.get(1));
            return;
        }

        let width = std::env::var(WIDTH_ENV)
            .expect("child width environment")
            .parse::<usize>()
            .expect("numeric child width");
        assert_eq!(nd_sched::init_global_pool(Some(width)), width);
        assert_eq!(rayon::current_num_threads(), width);
        assert!(rayon::current_thread_index().is_none());

        let design_sat_keps: [[[f64; 6]; 2]; 2] = [
            [
                [7000.0, 0.001, 0.45, 0.00, 0.0, 0.00],
                [7050.0, 0.001, 0.45, 0.05, 0.0, 0.20],
            ],
            [
                [6980.0, 0.001, 0.47, 0.03, 0.0, 0.10],
                [7090.0, 0.001, 0.48, 0.04, 0.0, 0.55],
            ],
        ];
        let epochs = [
            2_460_000.0,
            2_460_000.000_5,
            2_460_000.001_0,
            2_460_000.001_5,
        ];
        let [first_design, _] = &design_sat_keps;
        let design_count = design_sat_keps.len();
        let n_sats = first_design.len();
        let batch_size = epochs.len();
        let cell_count = design_count * batch_size;
        let flat_work_units = cell_count
            .checked_mul(
                effective_flat_pair_width(2, n_sats)
                    .expect("fixed test pair width must not overflow"),
            )
            .expect("fixed test work count must not overflow");
        let admitted = should_use_outer_batch_parallel_for(cell_count, false, width)
            || should_use_outer_batch_parallel_for_flat_work_units(
                cell_count,
                flat_work_units,
                false,
                width,
            )
            .expect("fixed test work count must not overflow");
        assert_eq!(
            admitted,
            width > 1,
            "fixture must split W1 fallback from W8 Phase A"
        );

        let eci_from_kep = |kep: &[f64; 6]| {
            let mut eci = [0.0_f64; 6];
            satpy_core::kep2eci_impl(kep, false, 0.0, 0.0, false, &mut eci);
            eci
        };
        let equ_from_eci = |eci: &[f64; 6]| {
            let mut equ = [0.0_f64; 6];
            satpy_core::eci2equinoc_impl(eci, 6, 0.0, 0.0, &mut equ);
            equ
        };
        let target1 = eci_from_kep(&[7100.0, 0.001, 0.45, 0.10, 0.0, 0.30]);
        let target2 = eci_from_kep(&[7125.0, 0.001, 0.45, 0.14, 0.0, 0.60]);

        let mut target_one_rows = Vec::with_capacity(batch_size * 6);
        let mut target_two_rows = Vec::with_capacity(batch_size * 6);
        for _ in epochs {
            target_one_rows.extend_from_slice(&target1);
            target_two_rows.extend_from_slice(&target2);
        }

        let mut satellites_eci = Vec::with_capacity(cell_count * n_sats);
        let mut satellites_equ = Vec::with_capacity(cell_count * n_sats);
        for design in design_sat_keps {
            let design_eci = design.iter().map(&eci_from_kep).collect::<Vec<_>>();
            let design_equ = design_eci.iter().map(&equ_from_eci).collect::<Vec<_>>();
            for _ in epochs {
                satellites_eci.extend_from_slice(&design_eci);
                satellites_equ.extend_from_slice(&design_equ);
            }
        }

        let forces = mf_target_body_force_batch(batch_size);
        let fronts =
            constellation_solve_population_batch_eci_precomputed(PopulationBatchEciRequest {
                satellite_eci_population: &satellites_eci,
                satellite_equinoctial_population: Some(&satellites_equ),
                design_count,
                satellite_count: n_sats,
                configuration: mf_batch_configuration(
                    &target_one_rows,
                    &target_two_rows,
                    &epochs,
                    &forces,
                ),
            })
            .expect("valid public population Phase-A fixture");

        assert_eq!(fronts.len(), design_count);
        assert!(fronts.iter().all(|events| events.len() == batch_size));
        assert!(
            fronts
                .iter()
                .flat_map(|events| events.iter())
                .any(|front| !front.is_empty()),
            "fixture must produce at least one transfer candidate"
        );
        assert_ne!(
            format!("{:?}", fronts.first()),
            format!("{:?}", fronts.get(1)),
            "fixture designs must remain distinguishable"
        );
        println!("PHASE_A_PUBLIC_IDENTITY={fronts:?}");
    }

    #[test]
    fn public_precomputed_population_matches_single_design_batch() {
        fn eci_from_kep(kep: &[f64; 6]) -> [f64; 6] {
            let mut eci = [0.0_f64; 6];
            satpy_core::kep2eci_impl(kep, false, 0.0, 0.0, false, &mut eci);
            eci
        }

        fn equ_from_eci(eci: &[f64; 6]) -> [f64; 6] {
            let mut equ = [0.0_f64; 6];
            satpy_core::eci2equinoc_impl(eci, 6, 0.0, 0.0, &mut equ);
            equ
        }

        let satellites_eci = [
            eci_from_kep(&[7000.0, 0.001, 0.45, 0.00, 0.0, 0.0]),
            eci_from_kep(&[7050.0, 0.001, 0.45, 0.05, 0.0, 0.2]),
        ];
        let satellites_equ = satellites_eci.map(|eci| equ_from_eci(&eci));
        let target1 = eci_from_kep(&[7100.0, 0.001, 0.45, 0.10, 0.0, 0.3]);
        let target2 = eci_from_kep(&[7125.0, 0.001, 0.45, 0.14, 0.0, 0.6]);
        let epochs = [2_460_000.0, 2_460_000.000_5];
        let mut target_one_rows = Vec::with_capacity(epochs.len() * 6);
        let mut target_two_rows = Vec::with_capacity(epochs.len() * 6);
        let mut satellites_eci_batch = Vec::with_capacity(epochs.len() * satellites_eci.len());
        let mut satellites_equ_batch = Vec::with_capacity(epochs.len() * satellites_equ.len());
        for _ in epochs {
            target_one_rows.extend_from_slice(&target1);
            target_two_rows.extend_from_slice(&target2);
            satellites_eci_batch.extend_from_slice(&satellites_eci);
            satellites_equ_batch.extend_from_slice(&satellites_equ);
        }

        for front_output_mode in [
            solve::FrontOutputMode::TransferPareto,
            solve::FrontOutputMode::VerifiedSuperset,
        ] {
            let forces = mf_target_body_force_batch(epochs.len());
            let configuration = BatchEciConfiguration {
                front_output_mode,
                ..mf_batch_configuration(&target_one_rows, &target_two_rows, &epochs, &forces)
            };
            let batch = constellation_solve_batch_eci_precomputed(BatchEciRequest {
                satellite_eci: &satellites_eci_batch,
                satellite_equinoctial: Some(&satellites_equ_batch),
                satellite_count: satellites_eci.len(),
                configuration: configuration.clone(),
            })
            .expect("valid MF precomputed batch");
            let population =
                constellation_solve_population_batch_eci_precomputed(PopulationBatchEciRequest {
                    satellite_eci_population: &satellites_eci_batch,
                    satellite_equinoctial_population: Some(&satellites_equ_batch),
                    design_count: 1,
                    satellite_count: satellites_eci.len(),
                    configuration,
                })
                .expect("valid MF precomputed population");

            assert_eq!(population.len(), 1);
            assert_eq!(batch.len(), epochs.len());
            assert_eq!(population.iter().map(Vec::len).next(), Some(batch.len()));
            assert!(
                batch.iter().any(|front| !front.is_empty()),
                "fixture must retain at least one transfer candidate"
            );
            for (event_idx, (batch_front, population_front)) in
                batch.iter().zip(population.iter().flatten()).enumerate()
            {
                let signature = |front: &ConstellationTransferFront| {
                    front
                        .candidates
                        .iter()
                        .map(|candidate| {
                            (
                                candidate.sat_index,
                                candidate.target_index,
                                candidate.optimum.branch_rev,
                                candidate.optimum.branch_low_path,
                                candidate.objectives.total_dv.to_bits(),
                                candidate.objectives.total_time.to_bits(),
                                candidate.objectives.relative_velocity.to_bits(),
                                candidate
                                    .objectives
                                    .time_per_relative_velocity_s_per_km_s
                                    .to_bits(),
                                candidate.optimum.branch_arrival_dv.to_bits(),
                                candidate.optimum.branch_tof_s.to_bits(),
                                candidate.optimum.branch_total_dv.to_bits(),
                                candidate.optimum.cost.to_bits(),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    signature(batch_front),
                    signature(population_front),
                    "event {event_idx}: public population route changed candidate ordering or values"
                );
            }
        }
    }

    #[test]
    fn public_precomputed_batch_uses_outer_parallelism_for_verified_superset() -> anyhow::Result<()>
    {
        fn eci_from_kep(kep: &[f64; 6]) -> [f64; 6] {
            let mut eci = [0.0_f64; 6];
            satpy_core::kep2eci_impl(kep, false, 0.0, 0.0, false, &mut eci);
            eci
        }

        fn equ_from_eci(eci: &[f64; 6]) -> [f64; 6] {
            let mut equ = [0.0_f64; 6];
            satpy_core::eci2equinoc_impl(eci, 6, 0.0, 0.0, &mut equ);
            equ
        }

        let satellites_eci = [
            eci_from_kep(&[7000.0, 0.001, 0.45, 0.00, 0.0, 0.0]),
            eci_from_kep(&[7050.0, 0.001, 0.45, 0.05, 0.0, 0.2]),
        ];
        let satellites_equ = satellites_eci.map(|eci| equ_from_eci(&eci));
        let target1 = eci_from_kep(&[7100.0, 0.001, 0.45, 0.10, 0.0, 0.3]);
        let target2 = eci_from_kep(&[7125.0, 0.001, 0.45, 0.14, 0.0, 0.6]);
        let epochs = [
            2_460_000.0,
            2_460_000.000_5,
            2_460_000.001_0,
            2_460_000.001_5,
            2_460_000.002_0,
            2_460_000.002_5,
            2_460_000.003_0,
            2_460_000.003_5,
            2_460_000.004_0,
            2_460_000.004_5,
            2_460_000.005_0,
            2_460_000.005_5,
            2_460_000.006_0,
            2_460_000.006_5,
            2_460_000.007_0,
            2_460_000.007_5,
        ];
        let mut target_one_rows = Vec::with_capacity(epochs.len() * 6);
        let mut target_two_rows = Vec::with_capacity(epochs.len() * 6);
        let mut satellites_eci_batch = Vec::with_capacity(epochs.len() * satellites_eci.len());
        let mut satellites_equ_batch = Vec::with_capacity(epochs.len() * satellites_equ.len());
        for _ in epochs {
            target_one_rows.extend_from_slice(&target1);
            target_two_rows.extend_from_slice(&target2);
            satellites_eci_batch.extend_from_slice(&satellites_eci);
            satellites_equ_batch.extend_from_slice(&satellites_equ);
        }

        let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
        let forces = mf_target_body_force_batch(epochs.len());
        let fronts = pool.install(|| {
            constellation_solve_batch_eci_precomputed(BatchEciRequest {
                satellite_eci: &satellites_eci_batch,
                satellite_equinoctial: Some(&satellites_equ_batch),
                satellite_count: satellites_eci.len(),
                configuration: BatchEciConfiguration {
                    front_output_mode: solve::FrontOutputMode::VerifiedSuperset,
                    ..mf_batch_configuration(&target_one_rows, &target_two_rows, &epochs, &forces)
                },
            })
        })?;

        let mut metrics = types::VerifiedSupersetStageMetrics::default();
        for front in &fronts {
            metrics.add_assign(front.verified_superset_metrics)?;
        }
        anyhow::ensure!(
            metrics.outer_batch_parallel_event_count == epochs.len(),
            "expected one outer-batch metric per event, got {} for {} events",
            metrics.outer_batch_parallel_event_count,
            epochs.len()
        );
        anyhow::ensure!(
            metrics.selected_pair_parallel_event_count == 0,
            "outer flat driver must not report selected-pair parallel events"
        );
        Ok(())
    }

    #[test]
    fn flat_pair_driver_matches_serial() -> anyhow::Result<()> {
        fn eci_from_kep(kep: &[f64; 6]) -> [f64; 6] {
            let mut eci = [0.0_f64; 6];
            satpy_core::kep2eci_impl(kep, false, 0.0, 0.0, false, &mut eci);
            eci
        }
        fn equ_from_eci(eci: &[f64; 6]) -> [f64; 6] {
            let mut equ = [0.0_f64; 6];
            satpy_core::eci2equinoc_impl(eci, 6, 0.0, 0.0, &mut equ);
            equ
        }

        // 4 satellites × 2 targets = 8 candidate pairs; pairs_to_verify = 4
        // selects >= 4 (the flatten width per event > the L2 threshold).
        let sat_keps = [
            [7000.0, 0.001, 0.45, 0.00, 0.0, 0.00],
            [7050.0, 0.001, 0.45, 0.05, 0.0, 0.20],
            [7010.0, 0.001, 0.46, 0.02, 0.0, 0.40],
            [7075.0, 0.001, 0.44, 0.08, 0.0, 0.60],
        ];
        let n_sats = sat_keps.len();
        let sat_eci_once = sat_keps.iter().map(eci_from_kep).collect::<Vec<_>>();
        let sat_equ_once = sat_eci_once.iter().map(equ_from_eci).collect::<Vec<_>>();
        let target1 = eci_from_kep(&[7100.0, 0.001, 0.45, 0.10, 0.0, 0.30]);
        let target2 = eci_from_kep(&[7125.0, 0.001, 0.45, 0.14, 0.0, 0.60]);

        // 3 distinct events (distinct epochs so the per-event work differs).
        let epochs = [2_460_000.0, 2_460_000.001_0, 2_460_000.002_0];
        let batch_size = epochs.len();

        // The precomputed row indexes satellites as a `(batch_size * n_sats)`
        // layout, so replicate the constellation per event.
        let mut satellites_eci = Vec::with_capacity(batch_size * n_sats);
        let mut satellites_equ = Vec::with_capacity(batch_size * n_sats);
        for _ in 0..batch_size {
            satellites_eci.extend_from_slice(&sat_eci_once);
            satellites_equ.extend_from_slice(&sat_equ_once);
        }

        // Build the (event × pair) plans for one batch. Each call yields a
        // fresh `Vec<Option<EventPlan>>` because `run_flat_event_pair_driver`
        // consumes its argument; Phase A is pool-independent.
        let build_plans = || -> anyhow::Result<Vec<Option<solve::EventPlan>>> {
            let mut solve_scratch = SolveScratch::new(n_sats)?;
            (0..batch_size)
                .map(|b| -> anyhow::Result<_> {
                    let sat_base = b
                        .checked_mul(n_sats)
                        .ok_or_else(|| anyhow::anyhow!("fixture satellite offset overflow"))?;
                    let sat_end = sat_base
                        .checked_add(n_sats)
                        .ok_or_else(|| anyhow::anyhow!("fixture satellite end overflow"))?;
                    let epoch_jd = epochs
                        .get(b)
                        .copied()
                        .ok_or_else(|| anyhow::anyhow!("fixture epoch must exist"))?;
                    let satellites = satellites_eci
                        .get(sat_base..sat_end)
                        .ok_or_else(|| anyhow::anyhow!("fixture ECI rows must exist"))?;
                    let satellites_equ = satellites_equ
                        .get(sat_base..sat_end)
                        .ok_or_else(|| anyhow::anyhow!("fixture EQU rows must exist"))?;
                    let plan = solve::prepare_event(solve::EventPlanRequest {
                        satellites,
                        satellites_equ_cached: Some(satellites_equ),
                        target1: &target1,
                        target2: &target2,
                        target_body_forces: mf_target_body_forces(),
                        configuration: solve::ConstellationSolveConfiguration {
                            max_time_s: 86_400.0,
                            max_phase_dv: 2.0,
                            max_transfer_dv: 5.0,
                            max_revs: 2,
                            min_perigee: 6_578.14,
                            max_apogee: 100_000.0,
                            pairs_to_verify: 4,
                            sampling_mode: SamplingMode::Fast,
                            search_depth: SearchDepthPolicy::default(),
                            epoch_jd,
                            distance_tol: 0.025,
                            deployer_min_distance: 0.12,
                            tof_penalty_weight: 0.1,
                            revolution_cap: 2.0,
                            target_propagation_authority: TargetPropagationAuthority::MfJ2,
                            force_config: None,
                            require_high_fidelity: false,
                            j2_closure_settings: solve::J2ClosureSettings::default(),
                            packed_coeffs: None,
                            local_optimizer: TransferLocalOptimizerConfig::default(),
                            warm_start: None,
                        },
                        scratch: Some(&mut solve_scratch),
                        front_output_mode: solve::FrontOutputMode::TransferPareto,
                    })?;
                    Ok(plan)
                })
                .collect()
        };

        // Sanity: the fixture must actually exercise >= 4 selected pairs in at
        // least one event, else the flatten width would be trivial.
        let sanity_plans = build_plans()?;
        anyhow::ensure!(
            sanity_plans
                .iter()
                .flatten()
                .all(solve::EventPlan::uses_borrowed_satellite_equ_state),
            "flat-driver fixture should borrow stable satellite arenas"
        );
        let max_pairs = sanity_plans
            .iter()
            .map(|plan| {
                plan.as_ref()
                    .map_or(0, solve::EventPlan::selected_pair_count)
            })
            .max()
            .ok_or_else(|| anyhow::anyhow!("flat-driver fixture must contain an event"))?;
        anyhow::ensure!(
            max_pairs >= 4,
            "fixture must expose >= 4 selected pairs per event (got {max_pairs})"
        );

        let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
        let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;

        let fronts_parallel: Vec<ConstellationTransferFront> = pool8.install(|| {
            let plans = build_plans()?;
            Ok::<_, anyhow::Error>(run_flat_event_pair_driver(plans)?)
        })?;
        let fronts_serial: Vec<ConstellationTransferFront> = pool1.install(|| {
            let plans = build_plans()?;
            Ok::<_, anyhow::Error>(run_flat_event_pair_driver(plans)?)
        })?;

        anyhow::ensure!(
            fronts_parallel.len() == fronts_serial.len(),
            "event count mismatch between 8-thread and size-1 flat driver"
        );
        anyhow::ensure!(
            fronts_serial.iter().any(|front| !front.is_empty()),
            "flat-driver fixture should produce at least one valid transfer"
        );

        // Strict, ordered, bit-for-bit comparison per event / per candidate.
        for (event_idx, (par_front, ser_front)) in
            fronts_parallel.iter().zip(fronts_serial.iter()).enumerate()
        {
            anyhow::ensure!(
                par_front.candidates.len() == ser_front.candidates.len(),
                "event {event_idx}: candidate count differs (8-thread vs size-1)"
            );
            for (slot, (pc, sc)) in par_front
                .candidates
                .iter()
                .zip(ser_front.candidates.iter())
                .enumerate()
            {
                let par_key = (
                    pc.sat_index,
                    pc.target_index,
                    pc.objectives.total_dv.to_bits(),
                    pc.objectives.total_time.to_bits(),
                    pc.objectives.relative_velocity.to_bits(),
                    pc.optimum.branch_tof_s.to_bits(),
                    pc.optimum.branch_total_dv.to_bits(),
                    pc.optimum.cost.to_bits(),
                    pc.estimated_objective.to_bits(),
                );
                let ser_key = (
                    sc.sat_index,
                    sc.target_index,
                    sc.objectives.total_dv.to_bits(),
                    sc.objectives.total_time.to_bits(),
                    sc.objectives.relative_velocity.to_bits(),
                    sc.optimum.branch_tof_s.to_bits(),
                    sc.optimum.branch_total_dv.to_bits(),
                    sc.optimum.cost.to_bits(),
                    sc.estimated_objective.to_bits(),
                );
                anyhow::ensure!(
                    par_key == ser_key,
                    "event {event_idx} candidate {slot}: 8-thread vs size-1 flat driver \
                     produced bit-different candidate"
                );
            }
        }
        Ok(())
    }

    fn assert_population_fronts_bit_identical(
        fronts_parallel: &[Vec<ConstellationTransferFront>],
        fronts_serial: &[Vec<ConstellationTransferFront>],
    ) -> anyhow::Result<()> {
        for (design_idx, (par_design, ser_design)) in
            fronts_parallel.iter().zip(fronts_serial.iter()).enumerate()
        {
            anyhow::ensure!(
                par_design.len() == ser_design.len(),
                "design {design_idx}: event count mismatch (8-thread vs size-1)"
            );
            for (event_idx, (par_front, ser_front)) in
                par_design.iter().zip(ser_design.iter()).enumerate()
            {
                anyhow::ensure!(
                    par_front.candidates.len() == ser_front.candidates.len(),
                    "design {design_idx} event {event_idx}: candidate count differs \
                     (8-thread vs size-1)"
                );
                for (slot, (pc, sc)) in par_front
                    .candidates
                    .iter()
                    .zip(ser_front.candidates.iter())
                    .enumerate()
                {
                    let par_key = (
                        pc.sat_index,
                        pc.target_index,
                        pc.objectives.total_dv.to_bits(),
                        pc.objectives.total_time.to_bits(),
                        pc.objectives.relative_velocity.to_bits(),
                        pc.optimum.branch_tof_s.to_bits(),
                        pc.optimum.branch_total_dv.to_bits(),
                        pc.optimum.cost.to_bits(),
                        pc.estimated_objective.to_bits(),
                    );
                    let ser_key = (
                        sc.sat_index,
                        sc.target_index,
                        sc.objectives.total_dv.to_bits(),
                        sc.objectives.total_time.to_bits(),
                        sc.objectives.relative_velocity.to_bits(),
                        sc.optimum.branch_tof_s.to_bits(),
                        sc.optimum.branch_total_dv.to_bits(),
                        sc.optimum.cost.to_bits(),
                        sc.estimated_objective.to_bits(),
                    );
                    anyhow::ensure!(
                        par_key == ser_key,
                        "design {design_idx} event {event_idx} candidate {slot}: \
                         8-thread vs size-1 population driver produced bit-different candidate"
                    );
                }
            }
        }
        Ok(())
    }

    /// P1 step 2 — parity gate for the POPULATION (design × event × pair) flat
    /// driver. Builds a fixture of 2 DESIGNS (distinct constellations so their
    /// fronts genuinely differ) × 3 events × >= 4 selected pairs and runs the
    /// SAME `run_flat_pop_event_pair_driver` under an 8-thread rayon pool and a
    /// size-1 pool, asserting the two `Vec<Vec<ConstellationTransferFront>>` are
    /// bit-identical per `[design][event]` (candidate
    /// `(sat, tgt, dv.bits, tof.bits, ...)` in order). The size-1 pool collapses
    /// the flat `par_iter` to a single worker (the deterministic serial
    /// reference = today's per-design drain); 8 threads exercises the
    /// cross-(design,event,pair) work-stealing scatter. If the scatter ever
    /// reordered a reduction or leaked state across designs, this fails. The
    /// one-dimension-wider sibling of `flat_pair_driver_matches_serial`.
    #[test]
    fn flat_pop_event_pair_driver_matches_serial() -> anyhow::Result<()> {
        fn eci_from_kep(kep: &[f64; 6]) -> [f64; 6] {
            let mut eci = [0.0_f64; 6];
            satpy_core::kep2eci_impl(kep, false, 0.0, 0.0, false, &mut eci);
            eci
        }
        fn equ_from_eci(eci: &[f64; 6]) -> [f64; 6] {
            let mut equ = [0.0_f64; 6];
            satpy_core::eci2equinoc_impl(eci, 6, 0.0, 0.0, &mut equ);
            equ
        }

        // Two distinct constellations (design 0 = the flat-driver fixture;
        // design 1 = perturbed SMAs/phasing so its fronts differ from design 0,
        // proving the driver keeps designs independent). 4 sats × 2 targets = 8
        // candidate pairs each; pairs_to_verify = 4 selects >= 4 per event.
        let design_sat_keps: [[[f64; 6]; 4]; 2] = [
            [
                [7000.0, 0.001, 0.45, 0.00, 0.0, 0.00],
                [7050.0, 0.001, 0.45, 0.05, 0.0, 0.20],
                [7010.0, 0.001, 0.46, 0.02, 0.0, 0.40],
                [7075.0, 0.001, 0.44, 0.08, 0.0, 0.60],
            ],
            [
                [6980.0, 0.001, 0.47, 0.03, 0.0, 0.10],
                [7035.0, 0.001, 0.47, 0.07, 0.0, 0.35],
                [7090.0, 0.001, 0.48, 0.04, 0.0, 0.55],
                [7120.0, 0.001, 0.46, 0.09, 0.0, 0.80],
            ],
        ];
        let n_sats = 4usize;
        let n_designs = design_sat_keps.len();
        let target1 = eci_from_kep(&[7100.0, 0.001, 0.45, 0.10, 0.0, 0.30]);
        let target2 = eci_from_kep(&[7125.0, 0.001, 0.45, 0.14, 0.0, 0.60]);

        // 3 distinct events (distinct epochs so the per-event work differs).
        let epochs = [2_460_000.0, 2_460_000.001_0, 2_460_000.002_0];
        let batch_size = epochs.len();

        let mut design_satellites_eci = Vec::with_capacity(n_designs);
        let mut design_satellites_equ = Vec::with_capacity(n_designs);
        for design_keps in &design_sat_keps {
            let sat_eci_once = design_keps.iter().map(eci_from_kep).collect::<Vec<_>>();
            let sat_equ_once = sat_eci_once.iter().map(equ_from_eci).collect::<Vec<_>>();
            let mut satellites_eci = Vec::with_capacity(batch_size * n_sats);
            let mut satellites_equ = Vec::with_capacity(batch_size * n_sats);
            for _ in 0..batch_size {
                satellites_eci.extend_from_slice(&sat_eci_once);
                satellites_equ.extend_from_slice(&sat_equ_once);
            }
            design_satellites_eci.push(satellites_eci);
            design_satellites_equ.push(satellites_equ);
        }

        // Build the [design][event] plans. Each call yields a fresh
        // `Vec<Vec<Option<EventPlan>>>` because `run_flat_pop_event_pair_driver`
        // consumes its argument; Phase A is pool-independent here (one
        // SolveScratch per design, mirroring the per-design driver — the
        // parallel Phase-A pre-pass is exercised separately at the pyfunction
        // gate; this test isolates Phase B+C parity of the population driver).
        let build_pop_plans = || -> anyhow::Result<Vec<Vec<Option<solve::EventPlan>>>> {
            design_satellites_eci
                .iter()
                .zip(&design_satellites_equ)
                .map(|(satellites_eci, satellites_equ)| -> anyhow::Result<_> {
                    let mut solve_scratch = SolveScratch::new(n_sats)?;
                    (0..batch_size)
                        .map(|b| -> anyhow::Result<_> {
                            let sat_base = b.checked_mul(n_sats).ok_or_else(|| {
                                anyhow::anyhow!("fixture satellite offset overflow")
                            })?;
                            let sat_end = sat_base
                                .checked_add(n_sats)
                                .ok_or_else(|| anyhow::anyhow!("fixture satellite end overflow"))?;
                            let epoch_jd = epochs
                                .get(b)
                                .copied()
                                .ok_or_else(|| anyhow::anyhow!("fixture epoch must exist"))?;
                            let satellites = satellites_eci
                                .get(sat_base..sat_end)
                                .ok_or_else(|| anyhow::anyhow!("fixture ECI rows must exist"))?;
                            let satellites_equ = satellites_equ
                                .get(sat_base..sat_end)
                                .ok_or_else(|| anyhow::anyhow!("fixture EQU rows must exist"))?;
                            let plan = solve::prepare_event(solve::EventPlanRequest {
                                satellites,
                                satellites_equ_cached: Some(satellites_equ),
                                target1: &target1,
                                target2: &target2,
                                target_body_forces: mf_target_body_forces(),
                                configuration: solve::ConstellationSolveConfiguration {
                                    max_time_s: 86_400.0,
                                    max_phase_dv: 2.0,
                                    max_transfer_dv: 5.0,
                                    max_revs: 2,
                                    min_perigee: 6_578.14,
                                    max_apogee: 100_000.0,
                                    pairs_to_verify: 4,
                                    sampling_mode: SamplingMode::Fast,
                                    search_depth: SearchDepthPolicy::default(),
                                    epoch_jd,
                                    distance_tol: 0.025,
                                    deployer_min_distance: 0.12,
                                    tof_penalty_weight: 0.1,
                                    revolution_cap: 2.0,
                                    target_propagation_authority: TargetPropagationAuthority::MfJ2,
                                    force_config: None,
                                    require_high_fidelity: false,
                                    j2_closure_settings: solve::J2ClosureSettings::default(),
                                    packed_coeffs: None,
                                    local_optimizer: TransferLocalOptimizerConfig::default(),
                                    warm_start: None,
                                },
                                scratch: Some(&mut solve_scratch),
                                front_output_mode: solve::FrontOutputMode::TransferPareto,
                            })?;
                            Ok(plan)
                        })
                        .collect()
                })
                .collect()
        };

        // Sanity: >= 4 selected pairs in at least one (design, event) cell.
        let sanity_pop_plans = build_pop_plans()?;
        anyhow::ensure!(
            sanity_pop_plans
                .iter()
                .flat_map(|design_plans| design_plans.iter())
                .flatten()
                .all(solve::EventPlan::uses_borrowed_satellite_equ_state),
            "population flat-driver fixture should borrow stable satellite arenas"
        );
        let max_pairs = sanity_pop_plans
            .iter()
            .flat_map(|design_plans| design_plans.iter())
            .map(|plan| {
                plan.as_ref()
                    .map_or(0, solve::EventPlan::selected_pair_count)
            })
            .max()
            .ok_or_else(|| anyhow::anyhow!("population fixture must contain an event"))?;
        anyhow::ensure!(
            max_pairs >= 4,
            "fixture must expose >= 4 selected pairs per (design, event) (got {max_pairs})"
        );

        let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
        let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;

        let fronts_parallel: Vec<Vec<ConstellationTransferFront>> = pool8.install(|| {
            let plans = build_pop_plans()?;
            Ok::<_, anyhow::Error>(run_flat_pop_event_pair_driver(plans)?)
        })?;
        let fronts_serial: Vec<Vec<ConstellationTransferFront>> = pool1.install(|| {
            let plans = build_pop_plans()?;
            Ok::<_, anyhow::Error>(run_flat_pop_event_pair_driver(plans)?)
        })?;

        anyhow::ensure!(
            fronts_parallel.len() == n_designs,
            "population driver returned the wrong design count"
        );
        anyhow::ensure!(
            fronts_parallel.len() == fronts_serial.len(),
            "design count mismatch between 8-thread and size-1 population driver"
        );
        anyhow::ensure!(
            fronts_serial
                .iter()
                .flat_map(|d| d.iter())
                .any(|front| !front.is_empty()),
            "population fixture should produce at least one valid transfer"
        );
        // Designs must be distinct (else the test would not prove independence).
        // Fingerprint each design's fronts by its candidates' total_dv bits.
        let design_fingerprint = |design: &[ConstellationTransferFront]| -> Vec<u64> {
            design
                .iter()
                .flat_map(|f| f.candidates.iter())
                .map(|c| c.objectives.total_dv.to_bits())
                .collect()
        };
        anyhow::ensure!(
            fronts_serial
                .first()
                .map(|design| design_fingerprint(design))
                != fronts_serial
                    .get(1)
                    .map(|design| design_fingerprint(design)),
            "the two designs produced identical fronts; fixture fails to exercise \
             cross-design independence"
        );

        // Strict, ordered, bit-for-bit comparison per design / event / candidate.
        assert_population_fronts_bit_identical(&fronts_parallel, &fronts_serial)?;
        Ok(())
    }

    #[test]
    fn population_event_pair_work_units_are_sorted() -> anyhow::Result<()> {
        let units = build_population_event_pair_work_units(&[vec![2, 0, 1], vec![1, 2]])?;
        let coords = units
            .iter()
            .map(|unit| (unit.design, unit.event, unit.pair))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            coords
                == vec![
                    (0, 0, 0),
                    (0, 0, 1),
                    (0, 2, 0),
                    (1, 0, 0),
                    (1, 1, 0),
                    (1, 1, 1)
                ],
            "population event-pair work units changed public artifact order"
        );
        Ok(())
    }

    #[test]
    fn population_event_pair_work_units_reject_count_overflow() {
        let result = build_population_event_pair_work_units(&[vec![usize::MAX, 1]]);
        assert_eq!(result, Err(TargetBodyForceBatchError::ArithmeticOverflow));
    }
}
