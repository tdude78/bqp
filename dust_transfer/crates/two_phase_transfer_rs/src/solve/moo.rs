//! The `OxyMOO` `NSGA-II` problem for the two transfer objectives, its
//! evaluation caches, and the work counters those caches are audited against.
//!
//! `TransferMooProblem` is the `Problem` implementation the optimizer drives.
//! Everything else here exists to make repeated evaluation of the same decision
//! cheap without changing what is computed: the quantized decision key, the
//! objective/plan caches keyed by it, and the batch-class and work counters
//! that let tests assert the serial and parallel batch paths did identical
//! work.

use rayon::prelude::*;

use super::{
    evaluate_plan_local, map_evaluation_arithmetic_overflow, should_use_leaf_parallel,
    transfer_candidate_is_objective_finite, Cell, EvaluationDiagnosticCounters, FxHashMap,
    InvalidTargetPropagationAuthorityCode, Nsga2Config, OxyMooPolicy, PlanContext, PlanResult,
    Problem, RefCell, SolveLocalWorkCache, VariableSpec, INVALID_COST, SINGLE_PAIR_LOWER_BOUNDS,
    SINGLE_PAIR_UPPER_BOUNDS, TRANSFER_MOO_OBJECTIVES, TRANSFER_MOO_VARIABLES,
};

use crate::types::counter_roster;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct TransferDecisionKey(pub(super) [u64; 3]);

#[inline]
pub(super) const fn transfer_decision_key(x: &[f64; 3]) -> TransferDecisionKey {
    let [time, phase, wait] = *x;
    TransferDecisionKey([time.to_bits(), phase.to_bits(), wait.to_bits()])
}

#[inline]
pub(super) fn repaired_transfer_decision(decision: &[f64]) -> [f64; 3] {
    let (Some(&time), Some(&phase), Some(&wait)) =
        (decision.first(), decision.get(1), decision.get(2))
    else {
        return [f64::NAN; 3];
    };
    let [time_lower, phase_lower, wait_lower] = SINGLE_PAIR_LOWER_BOUNDS;
    let [time_upper, phase_upper, wait_upper] = SINGLE_PAIR_UPPER_BOUNDS;
    let mut repaired = [
        time.clamp(time_lower, time_upper),
        phase.clamp(phase_lower, phase_upper),
        wait.clamp(wait_lower, wait_upper),
    ];
    repair_transfer_decision(&mut repaired);
    repaired
}

#[inline]
pub(super) fn repair_transfer_decision(x: &mut [f64; 3]) {
    let [time, phase, wait] = *x;
    let [time_lower, phase_lower, wait_lower] = SINGLE_PAIR_LOWER_BOUNDS;
    let [time_upper, phase_upper, wait_upper] = SINGLE_PAIR_UPPER_BOUNDS;
    let mut time = time.clamp(time_lower, time_upper);
    let phase = phase.clamp(phase_lower, phase_upper);
    let mut wait = wait.clamp(wait_lower, wait_upper);
    if time + wait > 0.98 {
        let headroom = 0.98 - time;
        if headroom > 0.0 {
            wait = wait.min(headroom);
        } else {
            time = 0.95;
            wait = 0.03;
        }
    }
    *x = [time, phase, wait];
}

#[derive(Clone, Copy)]
pub(super) struct TransferMooEvalCacheEntry {
    pub(super) key: TransferDecisionKey,
    pub(super) objectives: [f64; TRANSFER_MOO_OBJECTIVES],
    pub(super) cv: f64,
}

pub(super) struct TransferMooEvalCache {
    entries: Vec<Option<TransferMooEvalCacheEntry>>,
    hits: usize,
    // 7.3 work-count audit: a `get` miss forces a full OxyMOO objective
    // evaluation, so misses == full-eval count for the run. Promoted to a
    // runtime counter (was hit-only, test-gated) so the verified-superset
    // stage metrics can surface oxymoo eval cache hit/miss/full-eval counts.
    misses: usize,
}

impl TransferMooEvalCache {
    pub(super) fn new(capacity: usize) -> Result<Self, InvalidTargetPropagationAuthorityCode> {
        let capacity = capacity
            .max(16)
            .checked_next_power_of_two()
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(capacity)
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        entries.resize(capacity, None);
        Ok(Self {
            entries,
            hits: 0,
            misses: 0,
        })
    }

    pub(super) fn get(
        &mut self,
        key: TransferDecisionKey,
    ) -> Result<Option<TransferMooEvalCacheEntry>, InvalidTargetPropagationAuthorityCode> {
        let slot = self.slot(key)?;
        let entry = self
            .entries
            .get(slot)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?
            .as_ref()
            .filter(|entry| entry.key == key)
            .copied();
        if entry.is_some() {
            self.hits = self
                .hits
                .checked_add(1)
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        } else {
            self.misses = self
                .misses
                .checked_add(1)
                .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        }
        Ok(entry)
    }

    pub(super) fn insert(
        &mut self,
        entry: TransferMooEvalCacheEntry,
    ) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        let slot = self.slot(entry.key)?;
        let entry_slot = self
            .entries
            .get_mut(slot)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        *entry_slot = Some(entry);
        Ok(())
    }

    fn record_hit(&mut self) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        self.hits = self
            .hits
            .checked_add(1)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        Ok(())
    }

    fn record_miss(&mut self) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        self.misses = self
            .misses
            .checked_add(1)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        Ok(())
    }

    #[inline]
    fn slot(
        &self,
        key: TransferDecisionKey,
    ) -> Result<usize, InvalidTargetPropagationAuthorityCode> {
        let mut hash = 0x9e37_79b9_7f4a_7c15u64;
        for value in key.0 {
            hash ^= value
                .wrapping_add(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(hash << 6)
                .wrapping_add(hash >> 2);
        }
        let length = u64::try_from(self.entries.len())
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        let mask = length
            .checked_sub(1)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        usize::try_from(hash & mask)
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
    }

    pub(super) const fn hits(&self) -> usize {
        self.hits
    }

    const fn misses(&self) -> usize {
        self.misses
    }

    #[cfg(test)]
    const fn capacity(&self) -> usize {
        self.entries.len()
    }
}

pub(super) type TransferMooPlanCache = RefCell<FxHashMap<TransferDecisionKey, PlanResult>>;
pub(super) const TRANSFER_MOO_PLAN_CACHE_MAX_ENTRIES: usize = 8192;

// No lock counters here, and nothing to count. `solve.rs` takes no locks on
// any path: `TransferMooPlanCache` and `moo_eval_cache` are `RefCell`s owned by
// one rayon worker, `SolveLocalWorkCache` comes from `map_init`, and the
// evaluation diagnostics live in a `thread_local!`. A `TransferMooLockCounters`
// used to sit here with `eval_cache_*`/`plan_cache_*` wait counters; its two
// recording methods were never called from anywhere, so the four counters could
// only ever read zero and the tests asserting so could not fail. Removed rather
// than left as evidence of a contention problem that does not exist.

static NEXT_TRANSFER_MOO_PROBLEM_GEN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

thread_local! {
    static TLS_TRANSFER_MOO_PROBLEM_GEN: Cell<u64> = const { Cell::new(0) };
    static TLS_TRANSFER_MOO_WORK_SCRATCH: RefCell<SolveLocalWorkCache> =
        RefCell::new(SolveLocalWorkCache::new());
    // 7.3 work-count audit: per-thread tally of full plan evaluations and
    // anchor-stage optimizer/probe work. Snapshotted (`work_count_snapshot`)
    // around each front-solver stage in `solve_plan_oxymoo_front_internal`,
    // then differenced to attribute full-eval / NM-run / probe counts per
    // stage. Thread-local for the same reason as EVALUATION_DIAGNOSTIC_COUNTERS
    // (concurrent pair solves must not contaminate each other's deltas); the
    // `add_delta` reduction primitive folds worker deltas back after the
    // parallel fan-out joins.
    static WORK_COUNT_COUNTERS: Cell<WorkCountCounters> = const {
        Cell::new(WorkCountCounters::ZERO)
    };
}

counter_roster! {
    error = crate::types::InvalidTargetPropagationAuthorityCode;
    overflow = crate::types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow;
    sub = production;
    /// 7.3 work-count audit thread-local tallies (see [`WORK_COUNT_COUNTERS`]).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(super) struct WorkCountCounters {
        /// Full plan evaluations that compute a fresh `PlanResult`.
        count plan_full_evaluations: usize,
        /// Nelder-Mead optimizer runs launched by the delta-v anchor stage
        /// (coarse + fine per anchor candidate).
        count anchor_nm_runs: usize,
        /// Summed optimizer function evaluations across those NM runs.
        count anchor_nm_iterations: usize,
        /// Anchor-stage probe plan evaluations (`push_delta_v_anchor_probe_candidates`).
        count anchor_probe_evaluations: usize,
    }
}

impl WorkCountCounters {
    // A field missed here fails to compile (non-exhaustive struct literal), so
    // this cannot silently drift from the roster.
    const ZERO: Self = Self {
        plan_full_evaluations: 0,
        anchor_nm_runs: 0,
        anchor_nm_iterations: 0,
        anchor_probe_evaluations: 0,
    };

    /// Field-wise per-worker delta reduction, mirroring
    /// [`crate::evaluate::EvaluationDiagnosticCounters::add_delta`]. Folding a
    /// same-thread delta would double-count, so callers must pass a delta the
    /// receiving thread has not already accumulated. Transactional: an
    /// overflow error leaves `self` unchanged.
    #[inline]
    pub(super) fn add_delta(
        &mut self,
        delta: Self,
    ) -> Result<(), crate::types::InvalidTargetPropagationAuthorityCode> {
        let mut merged = *self;
        Self::roster_add(&mut merged, &delta)?;
        *self = merged;
        Ok(())
    }

    /// Field-wise per-worker delta (`self - before`), the inverse of
    /// [`Self::add_delta`]. Used by the 7.4 `OxyMOO` parallel batch path to
    /// capture each rayon worker's plan-eval work so it can be reduced back
    /// into the front-solve thread's tallies after the join.
    #[inline]
    pub(super) fn delta_since(
        self,
        before: Self,
    ) -> Result<Self, crate::types::InvalidTargetPropagationAuthorityCode> {
        Self::roster_delta_since(&self, &before)
    }
}

/// 7.4 `OxyMOO` batch-parallel classification: per-front-thread tally of how many
/// NSGA-II objective batches were evaluated in parallel vs serially. Reset
/// before each optimizer run and read back after (see
/// `push_oxymoo_transfer_candidates`) to attribute the
/// `oxymoo_{parallel,serial}_batch_count` stage metrics. Thread-local because
/// concurrent pair solves must not contaminate each other's counts; the
/// `evaluate_batch` override runs on the front-solve thread, so a same-thread
/// snapshot/diff is exact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct OxymooBatchClass {
    pub(super) parallel: usize,
    pub(super) serial: usize,
}

impl OxymooBatchClass {
    const ZERO: Self = Self {
        parallel: 0,
        serial: 0,
    };
}

thread_local! {
    static TLS_OXYMOO_BATCH_CLASS: Cell<OxymooBatchClass> = const { Cell::new(OxymooBatchClass::ZERO) };
}

#[inline]
pub(super) fn reset_oxymoo_batch_class() {
    TLS_OXYMOO_BATCH_CLASS.with(|cell| cell.set(OxymooBatchClass::ZERO));
}

#[inline]
pub(super) fn oxymoo_batch_class_snapshot() -> OxymooBatchClass {
    TLS_OXYMOO_BATCH_CLASS.with(Cell::get)
}

#[inline]
fn record_oxymoo_serial_batch() -> anyhow::Result<()> {
    TLS_OXYMOO_BATCH_CLASS.with(|cell| {
        let mut class = cell.get();
        class.serial = class
            .serial
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("OxyMOO serial batch count overflow"))?;
        cell.set(class);
        Ok(())
    })
}

#[inline]
fn record_oxymoo_parallel_batch() -> anyhow::Result<()> {
    TLS_OXYMOO_BATCH_CLASS.with(|cell| {
        let mut class = cell.get();
        class.parallel = class
            .parallel
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("OxyMOO parallel batch count overflow"))?;
        cell.set(class);
        Ok(())
    })
}

// 7.4 delta-v anchor-parallel tally: per-front-thread count of anchor NM runs
// dispatched through the parallel anchor fan-out. Reset before the anchor stage
// (see `nd_pipeline::native_mf::solve_transfer_front_group`) and read back
// after to attribute the
// `anchor_parallel_count` stage metric. Incremented on the front thread during
// the serial post-join reduction (one increment per NM run each parallel worker
// performed), so it equals the emitted anchor-NM run metric whenever every anchor runs in
// parallel. Thread-local so concurrent pair solves cannot contaminate each
// other's counts.
thread_local! {
    static TLS_ANCHOR_PARALLEL_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[inline]
pub(super) fn reset_anchor_parallel_count() {
    TLS_ANCHOR_PARALLEL_COUNT.with(|cell| cell.set(0));
}

#[inline]
pub(super) fn anchor_parallel_count_snapshot() -> usize {
    TLS_ANCHOR_PARALLEL_COUNT.with(Cell::get)
}

#[inline]
pub(super) fn record_anchor_parallel_runs(
    runs: usize,
) -> Result<(), crate::types::InvalidTargetPropagationAuthorityCode> {
    TLS_ANCHOR_PARALLEL_COUNT.with(|cell| {
        let updated = cell
            .get()
            .checked_add(runs)
            .ok_or(crate::types::InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        cell.set(updated);
        Ok(())
    })
}

#[inline]
pub(super) fn work_count_snapshot() -> WorkCountCounters {
    WORK_COUNT_COUNTERS.with(Cell::get)
}

#[inline]
pub(super) fn restore_work_count_snapshot(snapshot: WorkCountCounters) {
    WORK_COUNT_COUNTERS.with(|cell| cell.set(snapshot));
}

#[inline]
pub(super) fn record_work_count(
    update: impl FnOnce(
        &mut WorkCountCounters,
    ) -> Result<(), crate::types::InvalidTargetPropagationAuthorityCode>,
) -> Result<(), crate::types::InvalidTargetPropagationAuthorityCode> {
    WORK_COUNT_COUNTERS.with(|cell| {
        let mut counters = cell.get();
        update(&mut counters)?;
        cell.set(counters);
        Ok(())
    })
}

/// Reduce a worker-computed work-count delta into the calling thread's
/// thread-local tallies (per-worker delta reduction). See
/// [`crate::evaluate::merge_evaluation_diagnostics`].
#[inline]
pub(super) fn merge_work_counts(
    delta: WorkCountCounters,
) -> Result<(), crate::types::InvalidTargetPropagationAuthorityCode> {
    record_work_count(|counters| counters.add_delta(delta))
}

/// Evaluate one transfer decision to a full `PlanResult`, off `&self` so the
/// 7.4 parallel batch path can call it from any rayon worker.
///
/// The `TLS_TRANSFER_MOO_WORK_SCRATCH` phase-state scratch is per-thread and
/// keyed by `problem_gen_id`: the first call for a new problem generation on a
/// given worker clears any stale scratch, so each worker uses its own fresh
/// cache. `OxyMOO`'s outer direct-map owns decision reuse and its bounded source
/// cache owns final materialization, so this scratch retains only phase/orbit
/// state and Lambert workspace. Full-plan results have one owner: bounded
/// `OxyMOO` source materialization below.
#[inline]
fn evaluate_transfer_moo_plan(
    ctx: &PlanContext,
    problem_gen_id: u64,
    x: &[f64; 3],
) -> Result<PlanResult, crate::types::InvalidTargetPropagationAuthorityCode> {
    let prev = TLS_TRANSFER_MOO_PROBLEM_GEN.with(std::cell::Cell::get);
    if prev != problem_gen_id {
        TLS_TRANSFER_MOO_PROBLEM_GEN.with(|g| g.set(problem_gen_id));
        TLS_TRANSFER_MOO_WORK_SCRATCH.with(|c| c.borrow_mut().clear());
    }
    TLS_TRANSFER_MOO_WORK_SCRATCH.with(|cache| evaluate_plan_local(x, ctx, false, cache))
}

pub(super) struct TransferMooProblem {
    ctx: PlanContext,
    problem_gen_id: u64,
    moo_eval_cache: RefCell<TransferMooEvalCache>,
    pub(super) plan_cache: TransferMooPlanCache,
}

impl TransferMooProblem {
    pub(super) fn new(
        ctx: PlanContext,
        plan_cache: TransferMooPlanCache,
        policy: TransferMooPolicy,
    ) -> Result<Self, InvalidTargetPropagationAuthorityCode> {
        let (population_size, generations) = policy.population_generations();
        // Perf #2: 2x headroom so the direct-mapped eval cache stays <50% load.
        // population_size*(generations+1) unique keys accumulate over the run;
        // without headroom they exceed the next_power_of_two() capacity and the
        // direct-mapped slots evict still-live parents, forcing full Lambert+prop
        // re-evaluations. Cache hits and misses return identical objective values,
        // so optimization results are unchanged (determinism-neutral).
        let generation_count = generations
            .checked_add(1)
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        let capacity = population_size
            .checked_mul(generation_count)
            .and_then(|value| value.checked_mul(2))
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        let problem_gen_id = NEXT_TRANSFER_MOO_PROBLEM_GEN
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |current| current.checked_add(1),
            )
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        Ok(Self {
            ctx,
            problem_gen_id,
            moo_eval_cache: RefCell::new(TransferMooEvalCache::new(capacity)?),
            plan_cache,
        })
    }

    #[inline]
    fn eval_cache_get(
        &self,
        key: TransferDecisionKey,
    ) -> Result<Option<TransferMooEvalCacheEntry>, InvalidTargetPropagationAuthorityCode> {
        self.moo_eval_cache.borrow_mut().get(key)
    }

    #[inline]
    fn eval_cache_insert(
        &self,
        entry: TransferMooEvalCacheEntry,
    ) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        self.moo_eval_cache.borrow_mut().insert(entry)
    }

    #[inline]
    fn eval_cache_record_hit(&self) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        self.moo_eval_cache.borrow_mut().record_hit()
    }

    #[inline]
    fn eval_cache_record_miss(&self) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        self.moo_eval_cache.borrow_mut().record_miss()
    }

    #[inline]
    #[cfg(test)]
    pub(super) fn plan_cache_borrow(
        &self,
    ) -> std::cell::Ref<'_, FxHashMap<TransferDecisionKey, PlanResult>> {
        self.plan_cache.borrow()
    }

    #[inline]
    fn plan_cache_borrow_mut(
        &self,
    ) -> std::cell::RefMut<'_, FxHashMap<TransferDecisionKey, PlanResult>> {
        self.plan_cache.borrow_mut()
    }

    pub(super) fn into_plan_cache(self) -> TransferMooPlanCache {
        self.plan_cache
    }

    pub(super) fn preload_plan(
        &self,
        key: TransferDecisionKey,
        x: &[f64; 3],
        plan: PlanResult,
    ) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        let mut objective_values = [0.0; TRANSFER_MOO_OBJECTIVES];
        fill_transfer_moo_objectives(&plan, &self.ctx, &mut objective_values);
        let cv = transfer_moo_constraint_violation(x, &plan, &self.ctx);
        if cv <= 0.0 && transfer_candidate_is_objective_finite(&plan) {
            self.insert_plan_cache_entry(key, plan)?;
        }
        self.eval_cache_insert(TransferMooEvalCacheEntry {
            key,
            objectives: objective_values,
            cv,
        })
    }

    // 7.3 work-count audit: runtime-visible so `push_oxymoo_transfer_candidates`
    // can fold the OxyMOO eval-cache hit/miss tallies into the stage metrics.
    pub(super) fn eval_cache_hits(&self) -> usize {
        self.moo_eval_cache.borrow().hits()
    }

    pub(super) fn eval_cache_misses(&self) -> usize {
        self.moo_eval_cache.borrow().misses()
    }

    #[cfg(test)]
    pub(super) fn eval_cache_capacity(&self) -> usize {
        self.moo_eval_cache.borrow().capacity()
    }

    fn insert_plan_cache_entry(
        &self,
        key: TransferDecisionKey,
        plan: PlanResult,
    ) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        let mut cache = self.plan_cache_borrow_mut();
        if let Some(existing) = cache.get_mut(&key) {
            *existing = plan;
            return Ok(());
        }
        if cache.len() < TRANSFER_MOO_PLAN_CACHE_MAX_ENTRIES {
            cache
                .try_reserve(1)
                .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
            cache.insert(key, plan);
        }
        Ok(())
    }

    pub(super) fn take_cached_source_plan(&self, key: TransferDecisionKey) -> Option<PlanResult> {
        self.plan_cache_borrow_mut().remove(&key)
    }
}

/// 7.4: the `parallel` feature threshold below which a batch is not worth
/// fanning out (the front-solve population is 16-28, so real generation batches
/// always clear this).
pub(super) const OXYMOO_BATCH_PARALLEL_MIN_ROWS: usize = 2;

/// Full physics for one transfer decision: plan, objectives, constraint
/// violation, and whether the finite/feasible plan should be admitted to the
/// plan cache. Shared by the serial [`TransferMooProblem::evaluate`] path and
/// the 7.4 parallel batch path so both compute byte-identical values.
#[inline]
fn compute_transfer_moo_row(
    ctx: &PlanContext,
    problem_gen_id: u64,
    x: &[f64; 3],
) -> Result<
    ([f64; TRANSFER_MOO_OBJECTIVES], f64, PlanResult, bool),
    crate::types::InvalidTargetPropagationAuthorityCode,
> {
    let plan = evaluate_transfer_moo_plan(ctx, problem_gen_id, x)?;
    let mut objective_values = [0.0; TRANSFER_MOO_OBJECTIVES];
    fill_transfer_moo_objectives(&plan, ctx, &mut objective_values);
    let cv = transfer_moo_constraint_violation(x, &plan, ctx);
    let insert_plan = cv <= 0.0 && transfer_candidate_is_objective_finite(&plan);
    Ok((objective_values, cv, plan, insert_plan))
}

fn checked_batch_output_row<'a>(
    values: &'a mut [f64],
    row: usize,
    width: usize,
    matrix_name: &str,
) -> anyhow::Result<&'a mut [f64]> {
    let start = row.checked_mul(width).ok_or_else(|| {
        anyhow::anyhow!("{matrix_name} row offset overflow for row {row}, width {width}")
    })?;
    let end = start.checked_add(width).ok_or_else(|| {
        anyhow::anyhow!("{matrix_name} row end overflow for row {row}, width {width}")
    })?;
    let values_len = values.len();
    values.get_mut(start..end).ok_or_else(|| {
        anyhow::anyhow!(
            "{matrix_name} row {row} is outside storage length {values_len} with width {width}",
        )
    })
}

enum OxyMooBatchRowKind {
    Written,
    Duplicate(usize),
}

#[derive(Clone, Copy)]
enum OxyMooBatchVirtualEntry {
    Existing(TransferMooEvalCacheEntry),
    Pending {
        key: TransferDecisionKey,
        unique: usize,
    },
}

#[derive(Clone, Copy)]
struct OxyMooBatchVirtualSlot {
    slot: usize,
    entry: OxyMooBatchVirtualEntry,
}

struct OxyMooBatchUniqueWork {
    x: [f64; 3],
    key: TransferDecisionKey,
    first_row: usize,
}

struct OxyMooBatchUniqueResult {
    objectives: [f64; TRANSFER_MOO_OBJECTIVES],
    cv: f64,
    plan: PlanResult,
    insert_plan: bool,
    diag_delta: EvaluationDiagnosticCounters,
    work_delta: WorkCountCounters,
}

impl TransferMooProblem {
    fn validate_batch_shape(
        &self,
        decisions: &[f64],
        n_variables: usize,
        n_objectives: usize,
        objectives: &[f64],
        constraint_violations: &[f64],
    ) -> anyhow::Result<()> {
        if n_variables != self.variable_specs().len() {
            anyhow::bail!(
                "problem evaluation variable count {n_variables} does not match expected {}",
                self.variable_specs().len()
            );
        }
        if n_objectives != self.objective_count() {
            anyhow::bail!(
                "problem evaluation objective count {n_objectives} does not match expected {}",
                self.objective_count()
            );
        }

        let row_count = constraint_violations.len();
        let expected_decisions = row_count.checked_mul(n_variables).ok_or_else(|| {
            anyhow::anyhow!("decision matrix length overflow for row_count {row_count}")
        })?;
        if decisions.len() != expected_decisions {
            anyhow::bail!(
                "decision matrix length {} does not match row_count * n_variables = {expected_decisions}",
                decisions.len()
            );
        }
        let expected_objectives = row_count.checked_mul(n_objectives).ok_or_else(|| {
            anyhow::anyhow!("objective matrix length overflow for row_count {row_count}")
        })?;
        if objectives.len() != expected_objectives {
            anyhow::bail!(
                "objective matrix length {} does not match row_count * n_objectives = {expected_objectives}",
                objectives.len()
            );
        }
        Ok(())
    }

    /// Serial per-row objective batch — the byte-identical reference path,
    /// mirroring the `crate::oxymoo::Problem::evaluate_batch` provided method. Used as
    /// the fallback whenever the parallel gate is off.
    pub(super) fn evaluate_batch_serial(
        &self,
        decisions: &[f64],
        n_variables: usize,
        n_objectives: usize,
        objectives: &mut [f64],
        constraint_violations: &mut [f64],
    ) -> anyhow::Result<()> {
        for (row, ((decision, objective_row), cv_slot)) in decisions
            .chunks_exact(n_variables)
            .zip(objectives.chunks_exact_mut(n_objectives))
            .zip(constraint_violations.iter_mut())
            .enumerate()
        {
            let cv = self.evaluate(decision, objective_row)?;
            if !cv.is_finite() {
                anyhow::bail!(
                    "problem evaluation returned non-finite constraint violation at row {row}"
                );
            }
            *cv_slot = cv.max(0.0);
            for (col, objective) in objective_row.iter().enumerate() {
                if !objective.is_finite() {
                    anyhow::bail!(
                        "problem evaluation wrote non-finite objective at row {row}, column {col}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Runtime gate for `OxyMOO` batches. Top-level multi-thread calls use global
    /// rayon fan-out; calls already on a rayon worker stay leaf-serial.
    pub(super) fn should_use_oxymoo_batch_parallel(&self, batch_size: usize) -> bool {
        should_use_leaf_parallel(
            self.ctx.execution_policy.allow_oxymoo_batch_parallel,
            batch_size,
            OXYMOO_BATCH_PARALLEL_MIN_ROWS,
            rayon::current_num_threads(),
            rayon::current_thread_index().is_none(),
        )
    }

    /// Parallel objective batch: resolve cache membership + hit/miss tallies in
    /// a serial pre-pass, compute the unique misses across the rayon pool with
    /// per-worker phase/orbit/Lambert scratch and no shared-cache access, then merge cache
    /// entries and per-worker diagnostic deltas back in deterministic row order.
    pub(super) fn evaluate_batch_parallel(
        &self,
        decisions: &[f64],
        n_variables: usize,
        n_objectives: usize,
        objectives: &mut [f64],
        constraint_violations: &mut [f64],
    ) -> anyhow::Result<()> {
        record_oxymoo_parallel_batch()?;
        let n_rows = constraint_violations.len();

        // `Written` rows are already resolved (a pre-existing cache hit) or will
        // be written directly from their unique miss result; `Duplicate(u)` rows
        // are same-batch repeats of unique miss `u`, fanned out after compute.
        let mut row_kinds = Vec::new();
        row_kinds
            .try_reserve(n_rows)
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        let mut unique = Vec::new();
        unique
            .try_reserve(n_rows)
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        let mut virtual_slots: Vec<OxyMooBatchVirtualSlot> = Vec::new();
        virtual_slots
            .try_reserve_exact(n_rows)
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;

        // PASS 1 (serial): classify every row and resolve pre-existing cache
        // hits, driving the hit/miss counters exactly as the serial loop would.
        // A same-batch duplicate decision is a hit here because the serial loop
        // inserts the first occurrence's result before the second is evaluated.
        for (row, ((decision, objective_row), cv_slot)) in decisions
            .chunks_exact(n_variables)
            .zip(objectives.chunks_exact_mut(n_objectives))
            .zip(constraint_violations.iter_mut())
            .enumerate()
        {
            let x = repaired_transfer_decision(decision);
            let key = transfer_decision_key(&x);
            let slot = self.moo_eval_cache.borrow().slot(key)?;
            let virtual_entry = virtual_slots
                .iter()
                .find(|entry| entry.slot == slot)
                .map(|entry| entry.entry)
                .or_else(|| {
                    self.moo_eval_cache
                        .borrow()
                        .entries
                        .get(slot)
                        .copied()
                        .flatten()
                        .map(OxyMooBatchVirtualEntry::Existing)
                });
            match virtual_entry {
                Some(OxyMooBatchVirtualEntry::Existing(entry)) if entry.key == key => {
                    self.eval_cache_record_hit()?;
                    objective_row.copy_from_slice(&entry.objectives);
                    *cv_slot = entry.cv.max(0.0);
                    row_kinds.push(OxyMooBatchRowKind::Written);
                }
                Some(OxyMooBatchVirtualEntry::Pending {
                    key: pending_key,
                    unique: pending_unique,
                }) if pending_key == key => {
                    self.eval_cache_record_hit()?;
                    row_kinds.push(OxyMooBatchRowKind::Duplicate(pending_unique));
                }
                _ => {
                    self.eval_cache_record_miss()?;
                    let pending_unique = unique.len();
                    unique.push(OxyMooBatchUniqueWork {
                        x,
                        key,
                        first_row: row,
                    });
                    let pending = OxyMooBatchVirtualEntry::Pending {
                        key,
                        unique: pending_unique,
                    };
                    if let Some(entry) = virtual_slots.iter_mut().find(|entry| entry.slot == slot) {
                        entry.entry = pending;
                    } else {
                        virtual_slots.push(OxyMooBatchVirtualSlot {
                            slot,
                            entry: pending,
                        });
                    }
                    row_kinds.push(OxyMooBatchRowKind::Written);
                }
            }
        }

        // PASS 2 (parallel): compute the miss physics off-thread with no shared
        // cache access. Each worker captures its own diagnostic-counter delta so
        // the front-solve thread's tallies stay exact after a serial reduction.
        let ctx = &self.ctx;
        let problem_gen_id = self.problem_gen_id;
        let mut computed = Vec::new();
        computed
            .try_reserve(unique.len())
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        unique
            .par_iter()
            .map(|work| {
                // Isolated region: worker-local deltas stay out of the
                // executing thread until the deterministic serial reduction
                // below. See `with_isolated_diag_region`.
                super::with_isolated_diag_region(|| {
                    compute_transfer_moo_row(ctx, problem_gen_id, &work.x)
                })
                .map(
                    |((objective_values, cv, plan, insert_plan), diag_delta, work_delta)| {
                        OxyMooBatchUniqueResult {
                            objectives: objective_values,
                            cv,
                            plan,
                            insert_plan,
                            diag_delta,
                            work_delta,
                        }
                    },
                )
            })
            .collect_into_vec(&mut computed);

        // Reduce per-worker diagnostic deltas back into the front-solve thread
        // so the whole-front lambert/J2/plan-eval stage metrics stay exact.
        let mut diag_total = EvaluationDiagnosticCounters::default();
        let mut work_total = WorkCountCounters::default();
        for res in &computed {
            let res = res.as_ref().map_err(|error| *error)?;
            map_evaluation_arithmetic_overflow(diag_total.add_delta(&res.diag_delta))?;
            work_total.add_delta(res.work_delta)?;
        }
        map_evaluation_arithmetic_overflow(crate::evaluate::merge_evaluation_diagnostics(
            &diag_total,
        ))?;
        merge_work_counts(work_total)?;

        // Lightweight copy of the miss outputs for duplicate fan-out (the plans
        // are consumed by pass 3 below).
        let mut unique_outputs = Vec::new();
        unique_outputs
            .try_reserve(computed.len())
            .map_err(|_| InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?;
        for res in &computed {
            let res = res.as_ref().map_err(|error| *error)?;
            unique_outputs.push((res.objectives, res.cv));
        }

        // PASS 3 (serial, row order): write miss outputs and update the eval /
        // plan caches in the same order the serial loop would, so eval-cache
        // slot collisions and plan-cache contents resolve identically.
        for (res, work) in computed.into_iter().zip(unique.iter()) {
            let res = res?;
            if !res.cv.is_finite() {
                anyhow::bail!(
                    "problem evaluation returned non-finite constraint violation at row {}",
                    work.first_row
                );
            }
            for (col, value) in res.objectives.iter().enumerate() {
                if !value.is_finite() {
                    anyhow::bail!(
                        "problem evaluation wrote non-finite objective at row {}, column {col}",
                        work.first_row
                    );
                }
            }
            checked_batch_output_row(objectives, work.first_row, n_objectives, "objective matrix")?
                .copy_from_slice(&res.objectives);
            let Some(cv_slot) = constraint_violations.get_mut(work.first_row) else {
                anyhow::bail!(
                    "constraint-violation row {} is outside storage length {}",
                    work.first_row,
                    constraint_violations.len()
                );
            };
            *cv_slot = res.cv.max(0.0);
            if res.insert_plan {
                self.insert_plan_cache_entry(work.key, res.plan)?;
            }
            self.eval_cache_insert(TransferMooEvalCacheEntry {
                key: work.key,
                objectives: res.objectives,
                cv: res.cv,
            })?;
        }

        // Fan out same-batch duplicates from their unique's computed result.
        for (row, kind) in row_kinds.iter().enumerate() {
            if let OxyMooBatchRowKind::Duplicate(u) = *kind {
                let Some(&(obj, cv)) = unique_outputs.get(u) else {
                    anyhow::bail!("duplicate row {row} refers to missing unique result {u}");
                };
                checked_batch_output_row(objectives, row, n_objectives, "objective matrix")?
                    .copy_from_slice(&obj);
                let Some(cv_slot) = constraint_violations.get_mut(row) else {
                    anyhow::bail!(
                        "constraint-violation row {row} is outside storage length {}",
                        constraint_violations.len()
                    );
                };
                *cv_slot = cv.max(0.0);
            }
        }

        Ok(())
    }
}

impl Problem for TransferMooProblem {
    fn variable_specs(&self) -> &[VariableSpec] {
        &TRANSFER_MOO_VARIABLES
    }

    fn objective_count(&self) -> usize {
        TRANSFER_MOO_OBJECTIVES
    }

    fn evaluate(&self, decision: &[f64], objectives: &mut [f64]) -> anyhow::Result<f64> {
        if decision.len() != TRANSFER_MOO_VARIABLES.len() {
            anyhow::bail!(
                "transfer decision length {} does not match expected {}",
                decision.len(),
                TRANSFER_MOO_VARIABLES.len()
            );
        }
        if objectives.len() != TRANSFER_MOO_OBJECTIVES {
            anyhow::bail!(
                "transfer objective length {} does not match expected {TRANSFER_MOO_OBJECTIVES}",
                objectives.len()
            );
        }
        let x = repaired_transfer_decision(decision);
        let key = transfer_decision_key(&x);
        if let Some(cached) = self.eval_cache_get(key)? {
            objectives.copy_from_slice(&cached.objectives);
            return Ok(cached.cv);
        }

        let (objective_values, cv, plan, insert_plan) =
            compute_transfer_moo_row(&self.ctx, self.problem_gen_id, &x)?;
        objectives.copy_from_slice(&objective_values);
        if insert_plan {
            self.insert_plan_cache_entry(key, plan)?;
        }
        self.eval_cache_insert(TransferMooEvalCacheEntry {
            key,
            objectives: objective_values,
            cv,
        })?;
        Ok(cv)
    }

    fn evaluate_batch(
        &self,
        decisions: &[f64],
        n_variables: usize,
        n_objectives: usize,
        objectives: &mut [f64],
        constraint_violations: &mut [f64],
    ) -> anyhow::Result<()> {
        self.validate_batch_shape(
            decisions,
            n_variables,
            n_objectives,
            objectives,
            constraint_violations,
        )?;
        {
            if self.should_use_oxymoo_batch_parallel(constraint_violations.len()) {
                return self.evaluate_batch_parallel(
                    decisions,
                    n_variables,
                    n_objectives,
                    objectives,
                    constraint_violations,
                );
            }
        }
        record_oxymoo_serial_batch()?;
        self.evaluate_batch_serial(
            decisions,
            n_variables,
            n_objectives,
            objectives,
            constraint_violations,
        )
    }
}

fn fill_transfer_moo_objectives(plan: &PlanResult, ctx: &PlanContext, objectives: &mut [f64]) {
    let Some((delta_v_slot, trailing)) = objectives.split_first_mut() else {
        return;
    };
    let Some(time_slot) = trailing.first_mut() else {
        return;
    };
    let total_dv = plan.total_dv();
    *delta_v_slot = if plan.valid && total_dv.is_finite() {
        total_dv
    } else {
        transfer_moo_dv_reference(ctx)
    };

    let time_per_rel_v = plan.time_per_relative_velocity_s_per_km_s();
    *time_slot = if plan.valid && time_per_rel_v.is_finite() {
        time_per_rel_v / 86_400.0
    } else {
        transfer_moo_time_per_relative_velocity_reference(ctx)
    };
}

pub(super) fn transfer_moo_constraint_violation(
    x: &[f64; 3],
    plan: &PlanResult,
    ctx: &PlanContext,
) -> f64 {
    let [time_ratio, _, wait_ratio] = *x;
    let mut cv = (time_ratio + wait_ratio - 0.98).max(0.0) * 10.0;
    if !plan.valid || plan.cost >= INVALID_COST {
        cv += 1.0;
    }
    if !plan.time_per_relative_velocity_s_per_km_s().is_finite() {
        cv += 1.0;
    }
    cv += normalized_excess(plan.phase_dv_norm, ctx.max_phase_dv);
    cv += normalized_excess(plan.transfer_dv_norm, ctx.max_transfer_dv);
    cv += normalized_excess(plan.total_time(), ctx.max_time_s);
    cv += normalized_excess(plan.distance, ctx.distance_tol);
    cv += normalized_excess(ctx.deployer_min_distance, plan.deployer_distance);
    cv
}

#[inline]
pub(super) fn normalized_excess(value: f64, limit: f64) -> f64 {
    if value.is_finite() && limit.is_finite() && limit > 0.0 {
        ((value - limit) / limit).max(0.0)
    } else {
        0.0
    }
}

#[inline]
pub(super) const fn transfer_moo_population_generations() -> (usize, usize) {
    (28, 5)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TransferMooPolicy {
    Full,
    FastPopulation20,
    FastPopulation16,
    FastGenerations3,
    FastGenerations2,
    FastPopulation20Generations3,
    FastInitialBest1,
    FastPopulation20Generations3InitialBest1,
    #[cfg(feature = "bench-internal")]
    FastStableObjectiveStop,
}

#[cfg(feature = "bench-internal")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferMooBenchPolicy {
    Full,
    FastPopulation20,
    FastPopulation16,
    FastGenerations3,
    FastGenerations2,
    FastPopulation20Generations3,
    FastInitialBest1,
    FastPopulation20Generations3InitialBest1,
    FastStableObjectiveStop,
}

#[cfg(feature = "bench-internal")]
impl From<TransferMooBenchPolicy> for TransferMooPolicy {
    fn from(policy: TransferMooBenchPolicy) -> Self {
        match policy {
            TransferMooBenchPolicy::Full => Self::Full,
            TransferMooBenchPolicy::FastPopulation20 => Self::FastPopulation20,
            TransferMooBenchPolicy::FastPopulation16 => Self::FastPopulation16,
            TransferMooBenchPolicy::FastGenerations3 => Self::FastGenerations3,
            TransferMooBenchPolicy::FastGenerations2 => Self::FastGenerations2,
            TransferMooBenchPolicy::FastPopulation20Generations3 => {
                Self::FastPopulation20Generations3
            }
            TransferMooBenchPolicy::FastInitialBest1 => Self::FastInitialBest1,
            TransferMooBenchPolicy::FastPopulation20Generations3InitialBest1 => {
                Self::FastPopulation20Generations3InitialBest1
            }
            TransferMooBenchPolicy::FastStableObjectiveStop => Self::FastStableObjectiveStop,
        }
    }
}

impl From<OxyMooPolicy> for TransferMooPolicy {
    fn from(policy: OxyMooPolicy) -> Self {
        match policy {
            OxyMooPolicy::Full => Self::Full,
            OxyMooPolicy::FastPopulation20 => Self::FastPopulation20,
            OxyMooPolicy::FastPopulation16 => Self::FastPopulation16,
            OxyMooPolicy::FastGenerations3 => Self::FastGenerations3,
            OxyMooPolicy::FastGenerations2 => Self::FastGenerations2,
            OxyMooPolicy::FastPopulation20Generations3 => Self::FastPopulation20Generations3,
            OxyMooPolicy::FastInitialBest1 => Self::FastInitialBest1,
            OxyMooPolicy::FastPopulation20Generations3InitialBest1 => {
                Self::FastPopulation20Generations3InitialBest1
            }
        }
    }
}

impl TransferMooPolicy {
    #[inline]
    pub(super) const fn population_generations(self) -> (usize, usize) {
        let (population_size, generations) = transfer_moo_population_generations();
        match self {
            Self::FastPopulation20 => (20, generations),
            Self::FastPopulation16 => (16, generations),
            Self::FastGenerations3 => (population_size, 3),
            Self::FastGenerations2 => (population_size, 2),
            Self::FastPopulation20Generations3 | Self::FastPopulation20Generations3InitialBest1 => {
                (20, 3)
            }
            _ => (population_size, generations),
        }
    }

    #[inline]
    pub(super) const fn initial_decision_limit(self) -> Option<usize> {
        match self {
            Self::FastInitialBest1 | Self::FastPopulation20Generations3InitialBest1 => Some(1),
            _ => None,
        }
    }

    #[cfg(feature = "bench-internal")]
    #[inline]
    pub(super) const fn use_stable_objective_stop(self) -> bool {
        matches!(self, Self::FastStableObjectiveStop)
    }
}

#[inline]
pub(super) fn transfer_moo_dv_reference(ctx: &PlanContext) -> f64 {
    (ctx.max_phase_dv + ctx.max_transfer_dv + 1.0).max(1.0)
}

#[inline]
fn transfer_moo_time_per_relative_velocity_reference(ctx: &PlanContext) -> f64 {
    ((ctx.max_time_s / 86_400.0 + 1.0) * 100.0).max(1.0)
}

/// Test-only entry to the production NSGA-II config, defaulting the policy to
/// whatever `ctx.search_depth` selects. Production builds its config inline at
/// the single `transfer_moo_config_with_policy` call site.
#[cfg(test)]
pub(super) fn transfer_moo_config_with_initial_decisions(
    ctx: &PlanContext,
    initial_decisions: Vec<f64>,
) -> Result<Nsga2Config, InvalidTargetPropagationAuthorityCode> {
    transfer_moo_config_with_policy(
        ctx,
        initial_decisions,
        TransferMooPolicy::from(ctx.search_depth.oxymoo_policy),
    )
}

pub(super) fn transfer_moo_config_with_policy(
    ctx: &PlanContext,
    initial_decisions: Vec<f64>,
    policy: TransferMooPolicy,
) -> Result<Nsga2Config, InvalidTargetPropagationAuthorityCode> {
    let (population_size, generations) = policy.population_generations();
    let max_evaluations = Some(
        population_size
            .checked_mul(
                generations
                    .checked_add(1)
                    .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?,
            )
            .ok_or(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)?,
    );
    Ok(Nsga2Config {
        population_size,
        generations,
        max_evaluations,
        seed: ctx.local_optimizer.seed,
        initial_decisions,
        ..Nsga2Config::default()
    })
}
