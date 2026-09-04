//! Propagation-census instrumentation.
//!
//! MEASUREMENT SCAFFOLDING — not intended to land. Counts, per call-site tag:
//! right-hand-side evaluations, integration invocations, and the total
//! propagated time span. The tag is a thread-local set by whichever caller
//! scope is currently running, so a rayon worker attributes its own work.

use num_traits::ToPrimitive;
use std::cell::Cell;
use std::fmt::Write;
#[cfg(feature = "prop-census")]
use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

pub(crate) const NTAG: usize = 11;

/// Census storage or accounting could no longer represent an exact observation.
///
/// Callers must reject the run rather than turn a failed measurement into a
/// saturated or partial row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PropagationCensusError {
    CounterOverflow,
    MutexPoisoned,
    Allocation,
    CollectionActive,
}

impl std::fmt::Display for PropagationCensusError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CounterOverflow => formatter.write_str("propagation census counter overflow"),
            Self::MutexPoisoned => formatter.write_str("propagation census mutex poisoned"),
            Self::Allocation => formatter.write_str("propagation census allocation failed"),
            Self::CollectionActive => {
                formatter.write_str("propagation census collection is active")
            }
        }
    }
}

impl std::error::Error for PropagationCensusError {}

type CensusResult<T> = Result<T, PropagationCensusError>;

#[inline]
fn census_lock<T>(mutex: &Mutex<T>) -> CensusResult<MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| PropagationCensusError::MutexPoisoned)
}

#[inline]
#[cfg(any(feature = "prop-census", test))]
fn try_push_census_row<T>(rows: &mut Vec<T>, limit: usize, row: T) -> CensusResult<()> {
    let next_len = rows
        .len()
        .checked_add(1)
        .ok_or(PropagationCensusError::CounterOverflow)?;
    if next_len > limit {
        return Err(PropagationCensusError::Allocation);
    }
    rows.try_reserve(1)
        .map_err(|_| PropagationCensusError::Allocation)?;
    rows.push(row);
    Ok(())
}

pub(crate) const TAG_OTHER: usize = 0;
pub const TAG_UKF_SIGMA: usize = 1;
pub const TAG_MASS_MISS: usize = 2;
pub const TAG_RELEASE_CONTROL: usize = 3;
pub const TAG_ZERO_MASS: usize = 4;
pub const TAG_UKF_SIGMA_PC: usize = 5;
// `alignment` and `lf_seed` used to sit here. Nothing ever set either, so the
// census advertised two rows that could not appear and sized twelve static
// arrays two slots wider than it needed. Removed rather than left as
// aspirational instrumentation -- a tag nobody sets is a claim the report
// cannot back.

// Release-control sub-phases. `TAG_RELEASE_CONTROL` is set at exactly one place
// (`two_phase_transfer_rs/src/postprocess/distribution.rs`), so it can say "8,261 propagations, 279 of
// them bit-identical repeats" and nothing at all about which sub-phase produced
// the repeats. These split that scope at the five sites that actually evaluate
// the objective.
//
// They are applied through `census_scope`, which is a no-op unless
// `prop-census` is on — so with the feature off, `release_control` keeps
// counting exactly what it counted before and no existing reader shifts under
// anyone. Turning the feature on moves those counts down into the sub-tags,
// which is the whole point and is why it is opt-in.
//
// THESE TAGS COUNT SCOPE ENTRIES AND PROPAGATIONS. THE TWO ARE NOT THE SAME
// UNIT, AND FOR THESE FIVE THEY ARE NOT EVEN THE SAME ORDER OF MAGNITUDE. A
// `PROP_SCOPE <tag>,entries,N` row with no matching `PROP_CENSUS <tag>` row
// means "N objective calls under this scope started zero propagations". It
// does NOT mean the scope is free, and it does not mean the instrument missed
// them -- see `TAG_RC_FD_JACOBIAN`, which is the worked example.
pub const TAG_RC_SKIP_PROBE: usize = 6;
pub const TAG_RC_LM_ENTRY: usize = 7;
pub const TAG_RC_ZERO_DV: usize = 8;
/// One finite-difference probe of the LM objective, at `x + jac_eps` in one
/// coordinate: `two_phase_transfer_rs/src/intercept.rs`, inside the `else`
/// arm that runs when no model Jacobian was produced.
///
/// READ THE ZERO CORRECTLY. On a p24/G1 cell this tag reports 312,570 scope
/// entries and NO `PROP_CENSUS` row at all -- zero propagations, zero RHS
/// evaluations -- in both recorded measurement arms
/// (docs/evidence/prefix-arc-20260813/p24-g1-on.txt line 190;
/// docs/evidence/dup-memo-20260813/p24-g1-off.txt line 188). That zero is a
/// TRUE MEASUREMENT of propagation cost, not a blind counter, and it is also
/// not a statement that the FD Jacobian is free.
///
/// WHY IT IS TRUE. The FD loop is entered only when `model_jacobian` produced
/// nothing, which in production means the caller was `optimize_intercept_bounded`
/// (model `None`) rather than `optimize_intercept_bounded_hf_with_model`. Both
/// production `optimize_intercept_bounded` sites on the shipped Part A path
/// (`two_phase_transfer_rs/src/postprocess/distribution.rs`, the
/// `hybrid_mf_seed_hf_refine` MF seed and the plain least-squares branch)
/// hand it a CLOSED-FORM objective --
/// `compute_miss_vector_equinoctial` -- which propagates nothing. So the
/// propagation census has nothing to attribute, and says so.
///
/// WHY IT IS NOT "FREE". The 312,570 entries are 312,570 analytic miss
/// evaluations plus their LM bookkeeping. That work is real and this
/// instrument cannot see it: the propagation census counts propagations, and
/// an equinoctial miss vector is not one. Anyone pricing this lane needs a
/// timer, not this counter.
///
/// WHEN IT WOULD BE NON-ZERO. Three ways, all live in the source: the
/// `RealFdFallback` route (a model was supplied and `model_jacobian` returned
/// `None` anyway), the third `optimize_intercept_bounded` site in
/// `two_phase_transfer_rs/src/postprocess/distribution.rs`, whose objective
/// calls `propagate_stamped_checked`, and any future model-free caller with a
/// propagating objective. If a run ever
/// prints a `PROP_CENSUS rc_fd_jacobian` row, one of those fired -- that is
/// signal, not noise.
///
/// THE SAME CAUTION APPLIES TO `TAG_RC_TRIAL_STEP`, WEAKER. It reports 322,920
/// entries against 71,526 propagations on the same cell, because its entries
/// mix the analytic MF pass with the propagating HF pass. Its
/// entries-to-propagations ratio is a route mixture, not a per-entry cost.
pub const TAG_RC_FD_JACOBIAN: usize = 9;
pub const TAG_RC_TRIAL_STEP: usize = 10;

/// Shard count for the counters bumped once per RHS EVALUATION.
///
/// The unsharded form was the leading suspect for the width cliff at 64+
/// threads. Those arrays are indexed by TAG, and a tag is phase-wide, so every
/// worker in a phase performed its `fetch_add` on the SAME eight bytes. At the
/// production rate (`atmosphere_model: 5`, 1838.82 ns/eval => 543,800
/// evaluations/s/thread) that put up to 139 M read-modify-writes per second on
/// a single cache line, shared across 8 NUMA domains and 2 sockets.
///
/// Power of two so the index is a mask rather than a modulo. 128 matches the
/// widest production pool (TC nodes are 128 physical cores, SMT off). A pool
/// wider than `NSHARD` wraps and two workers share a shard -- which is exactly
/// why the increments below stay ATOMIC rather than becoming plain adds. A lost
/// update here would corrupt the counts this campaign reasons about.
pub(crate) const NSHARD: usize = 128;
static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn atomic_add(counter: &[AtomicU64], index: usize, value: u64) {
    if let Some(cell) = counter.get(index) {
        cell.fetch_add(value, Ordering::Relaxed);
    }
}

/// Add to a required diagnostic counter while retaining relaxed, wrapping
/// accounting in builds without the exact census.
#[cfg(not(feature = "prop-census"))]
#[inline]
fn relaxed_atomic_add(counter: &[AtomicU64], index: usize, value: u64) -> CensusResult<()> {
    let cell = counter
        .get(index)
        .ok_or(PropagationCensusError::CounterOverflow)?;
    cell.fetch_add(value, Ordering::Relaxed);
    Ok(())
}

#[cfg(feature = "prop-census")]
#[inline]
fn checked_atomic_add(counter: &[AtomicU64], index: usize, value: u64) -> CensusResult<()> {
    let cell = counter
        .get(index)
        .ok_or(PropagationCensusError::CounterOverflow)?;
    cell.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        current.checked_add(value)
    })
    .map(|_| ())
    .map_err(|_| PropagationCensusError::CounterOverflow)
}

#[inline]
fn atomic_store(counter: &[AtomicU64], index: usize, value: u64) {
    if let Some(cell) = counter.get(index) {
        cell.store(value, Ordering::Relaxed);
    }
}

#[inline]
fn atomic_load(counter: &[AtomicU64], index: usize) -> u64 {
    counter
        .get(index)
        .map_or(0, |cell| cell.load(Ordering::Relaxed))
}

#[inline]
fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[inline]
fn nonnegative_f64_to_u64(value: f64) -> Option<u64> {
    (value.is_finite() && value >= 0.0)
        .then(|| value.to_u64())
        .flatten()
}

#[inline]
fn u64_to_f64(value: u64) -> f64 {
    value.to_f64().unwrap_or(f64::INFINITY)
}

/// One shard of a tag-indexed counter, padded to its own line pair.
///
/// `NTAG * 8` is 88 bytes, so without the alignment two shards would still
/// share a 64-byte line and the sharding would buy nothing. Zen prefetches
/// 64-byte lines in PAIRS, so the padding target is 128 bytes, not 64.
#[repr(align(128))]
pub(crate) struct TagShard([AtomicU64; NTAG]);

/// One shard of a counter that is NOT tag-indexed.
#[repr(align(128))]
pub(crate) struct ScalarShard(AtomicU64);

type TagCounter = [TagShard; NSHARD];
type ScalarCounter = [ScalarShard; NSHARD];

// `const fn`, not `const` items: a `const` holding atomics trips
// `clippy::declare_interior_mutable_const`, because each USE of such a const
// silently materialises an independent copy. Here that is exactly the intent
// -- every static wants its own zeroed array -- but a function says so without
// leaving a lint to be suppressed at each call site.
const fn new_tag_counter() -> TagCounter {
    [const { TagShard([const { AtomicU64::new(0) }; NTAG]) }; NSHARD]
}

const fn new_scalar_counter() -> ScalarCounter {
    [const { ScalarShard(AtomicU64::new(0)) }; NSHARD]
}

/// A sharded `fetch_min` counter. `u64::MAX` is the unset sentinel in every
/// shard, exactly as it was in the single atomic, so [`min_value`]'s fold
/// returns what the unsharded counter would have held.
const fn new_min_counter() -> ScalarCounter {
    [const { ScalarShard(AtomicU64::new(u64::MAX)) }; NSHARD]
}

/// This thread's shard, assigned once on first use and never changed.
///
/// `const`-initialised with a sentinel rather than given a lazy initialiser:
/// a non-const `thread_local!` compiles to a lazy-init guard on EVERY access,
/// and this sits on the per-evaluation path. The sentinel branch is
/// perfectly predicted after the first call.
#[inline]
fn current_shard() -> usize {
    SHARD.with(|cell| {
        let cached = cell.get();
        if cached != usize::MAX {
            return cached;
        }
        let assigned = NEXT_SHARD.fetch_add(1, Ordering::Relaxed) & (NSHARD - 1);
        cell.set(assigned);
        assigned
    })
}

#[inline]
pub(crate) fn tag_add(counter: &TagCounter, tag: usize) {
    if let Some(shard) = counter.get(current_shard()) {
        atomic_add(&shard.0, tag, 1);
    }
}

/// Add `value` rather than 1, for the counters that accumulate a per-call
/// quantity (step counts, spans) instead of an occurrence.
#[inline]
fn tag_add_value(counter: &TagCounter, tag: usize, value: u64) {
    if let Some(shard) = counter.get(current_shard()) {
        atomic_add(&shard.0, tag, value);
    }
}

/// Add to a tag-indexed sharded counter under the census's checked accounting.
///
/// Overflow is now detected per SHARD rather than per tag. That is a narrower
/// trip point, not a weaker one: the sum over shards is what any reader sees,
/// so a shard reaching `u64::MAX` is reported before the total could.
#[cfg(feature = "prop-census")]
#[inline]
fn checked_tag_add(counter: &TagCounter, tag: usize, value: u64) -> CensusResult<()> {
    let shard = counter
        .get(current_shard())
        .ok_or(PropagationCensusError::CounterOverflow)?;
    checked_atomic_add(&shard.0, tag, value)
}

#[cfg(not(feature = "prop-census"))]
#[inline]
fn relaxed_tag_add(counter: &TagCounter, tag: usize, value: u64) -> CensusResult<()> {
    let shard = counter
        .get(current_shard())
        .ok_or(PropagationCensusError::CounterOverflow)?;
    relaxed_atomic_add(&shard.0, tag, value)
}

/// Total across every shard. The sharding is invisible to readers: this returns
/// exactly what the single unsharded counter would have held.
#[inline]
fn tag_sum(counter: &TagCounter, tag: usize) -> u64 {
    counter.iter().map(|shard| atomic_load(&shard.0, tag)).sum()
}

#[inline]
fn tag_clear(counter: &TagCounter, tag: usize) {
    for shard in counter {
        atomic_store(&shard.0, tag, 0);
    }
}

#[inline]
pub(crate) fn scalar_add(counter: &ScalarCounter) {
    if let Some(shard) = counter.get(current_shard()) {
        shard.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
fn scalar_sum(counter: &ScalarCounter) -> u64 {
    counter
        .iter()
        .map(|shard| shard.0.load(Ordering::Relaxed))
        .sum()
}

#[inline]
fn scalar_add_value(counter: &ScalarCounter, value: u64) {
    if let Some(shard) = counter.get(current_shard()) {
        shard.0.fetch_add(value, Ordering::Relaxed);
    }
}

#[inline]
fn scalar_clear(counter: &ScalarCounter) {
    for shard in counter {
        shard.0.store(0, Ordering::Relaxed);
    }
}

#[inline]
fn min_observe(counter: &ScalarCounter, value: u64) {
    if let Some(shard) = counter.get(current_shard()) {
        shard.0.fetch_min(value, Ordering::Relaxed);
    }
}

/// Smallest value in any shard, `u64::MAX` when nothing was ever observed.
#[inline]
fn min_value(counter: &ScalarCounter) -> u64 {
    counter
        .iter()
        .map(|shard| shard.0.load(Ordering::Relaxed))
        .min()
        .unwrap_or(u64::MAX)
}

#[inline]
fn min_clear(counter: &ScalarCounter) {
    for shard in counter {
        shard.0.store(u64::MAX, Ordering::Relaxed);
    }
}

pub(crate) static RHS_EVALS: TagCounter = new_tag_counter();
pub(crate) static PROPAGATIONS: TagCounter = new_tag_counter();
pub(crate) static SPAN_MS: TagCounter = new_tag_counter();
/// A failed census update -- a bounded store that filled, or a multi-counter
/// update that left a preceding atomic in place -- can no longer describe an
/// exact observation. Keep that data quarantined: `report` refuses it until the
/// next successful `reset` establishes a fresh observation.
///
/// THE PHYSICS DOES NOT MOVE WHEN THIS LATCHES. Every census writer that can
/// fail is called from the propagation entry path, so returning the failure to
/// the caller converted an instrumentation limit into a
/// `FinalPropagationFailure::Census`, which
/// `nd_pipeline::solver_qualification` maps straight to a
/// `QualificationOutcomeV2` -- a design verdict. The observation now
/// self-invalidates instead: the propagation runs, and the census refuses to
/// report numbers.
#[cfg(feature = "prop-census")]
static CENSUS_INVALID: AtomicBool = AtomicBool::new(false);

/// Which error the FIRST invalidation carried, as a `PropagationCensusError`
/// discriminant, so `report` names the cause rather than a generic failure.
#[cfg(feature = "prop-census")]
static CENSUS_INVALID_KIND: AtomicU8 = AtomicU8::new(0);

/// Generation of process-global census state. A scoped diagnostic owner may
/// write only while its captured generation remains current.
#[cfg(feature = "prop-census")]
static CENSUS_EPOCH: AtomicU64 = AtomicU64::new(0);

/// One lock for every in-process test that resets or asserts exact global
/// census state. Kept here so probe and integrator tests cannot accidentally
/// serialize on different module-local locks.
#[cfg(all(test, feature = "prop-census"))]
static CENSUS_TESTS: Mutex<()> = Mutex::new(());
#[cfg(all(test, feature = "prop-census"))]
static CENSUS_TEST_OWNER: AtomicU64 = AtomicU64::new(0);
#[cfg(all(test, feature = "prop-census"))]
static NEXT_CENSUS_TEST_THREAD: AtomicU64 = AtomicU64::new(1);
#[cfg(all(test, feature = "prop-census"))]
thread_local! {
    static CENSUS_TEST_THREAD: Cell<u64> = const { Cell::new(0) };
}

#[cfg(all(test, feature = "prop-census"))]
pub(crate) struct CensusTestGuard {
    _lock: MutexGuard<'static, ()>,
}

#[cfg(all(test, feature = "prop-census"))]
impl Drop for CensusTestGuard {
    fn drop(&mut self) {
        CENSUS_TEST_OWNER.store(0, Ordering::Release);
    }
}

#[cfg(all(test, feature = "prop-census"))]
fn census_test_thread() -> u64 {
    CENSUS_TEST_THREAD.with(|thread| {
        let current = thread.get();
        if current != 0 {
            return current;
        }
        let assigned = NEXT_CENSUS_TEST_THREAD.fetch_add(1, Ordering::Relaxed);
        thread.set(assigned);
        assigned
    })
}

#[cfg(all(test, feature = "prop-census"))]
pub(crate) fn test_census_guard() -> CensusTestGuard {
    let lock = match CENSUS_TESTS.lock() {
        Ok(guard) => guard,
        // Poison means an earlier test already failed. Recover only to retain
        // serialization for later tests; no evidence from the failed test is
        // made valid by this.
        Err(poisoned) => poisoned.into_inner(),
    };
    CENSUS_TEST_OWNER.store(census_test_thread(), Ordering::Release);
    CensusTestGuard { _lock: lock }
}

#[cfg(all(test, feature = "prop-census"))]
fn census_test_thread_may_write() -> bool {
    let owner = CENSUS_TEST_OWNER.load(Ordering::Acquire);
    owner == 0 || owner == census_test_thread()
}

#[cfg(feature = "prop-census")]
const fn census_error_code(error: PropagationCensusError) -> u8 {
    match error {
        PropagationCensusError::CounterOverflow => 1,
        PropagationCensusError::MutexPoisoned => 2,
        PropagationCensusError::Allocation => 3,
        PropagationCensusError::CollectionActive => 4,
    }
}

#[cfg(feature = "prop-census")]
const fn census_error_from_code(code: u8) -> PropagationCensusError {
    match code {
        2 => PropagationCensusError::MutexPoisoned,
        3 => PropagationCensusError::Allocation,
        4 => PropagationCensusError::CollectionActive,
        _ => PropagationCensusError::CounterOverflow,
    }
}

/// Latch the census as unusable without touching the worker's I/O path.
///
/// The quiescent owner observes the typed error when it asks for the report.
/// Only the first cause of an epoch is retained; later failures are
/// consequences of the already-invalid diagnostic epoch.
#[cfg(feature = "prop-census")]
#[cold]
fn invalidate_census(error: PropagationCensusError) {
    if !CENSUS_INVALID.swap(true, Ordering::AcqRel) {
        CENSUS_INVALID_KIND.store(census_error_code(error), Ordering::Release);
    }
}

#[cfg(feature = "prop-census")]
#[inline]
fn ensure_census_valid() -> CensusResult<()> {
    if CENSUS_INVALID.load(Ordering::Acquire) {
        Err(census_error_from_code(
            CENSUS_INVALID_KIND.load(Ordering::Acquire),
        ))
    } else {
        Ok(())
    }
}

pub(crate) static STEPS: TagCounter = new_tag_counter();
/// Encke rectification segments: one logical propagation is chopped into
/// `MAX_RECT_SEGMENT`-long pieces, each a separate solver invocation.
pub(crate) static SEGMENTS: TagCounter = new_tag_counter();
/// Accepted steps whose size was clipped by `dt_max` rather than chosen by the
/// error controller.
pub(crate) static SATURATED: TagCounter = new_tag_counter();
/// Steps computed, failed the error test, and thrown away. A full stage sweep
/// each, contributing nothing to the answer.
pub(crate) static REJECTED: TagCounter = new_tag_counter();
#[cfg(feature = "prop-census")]
static SCOPE_ENTRIES: TagCounter = new_tag_counter();
/// Smallest accepted step seen anywhere, in microseconds, `u64::MAX` if unset.
/// Stored as an integer so it can live in an atomic min.
pub(crate) static MIN_ACCEPTED_H_NS: ScalarCounter = new_min_counter();
/// Accepted steps inside the `equinoc2eci` cache-clustering regime, and the
/// subset that is not merely an endpoint-truncated final step.
pub(crate) static CACHE_CLUSTER_STEPS: ScalarCounter = new_scalar_counter();
pub(crate) static CACHE_CLUSTER_STEPS_UNTRUNCATED: ScalarCounter = new_scalar_counter();
/// Steps accepted DESPITE failing the error test, because `h` had already
/// reached `h_min`. Any nonzero value means returned states do not satisfy the
/// requested tolerance, and nothing else reports it.
pub(crate) static UNDERFLOW_ACCEPTS: AtomicU64 = AtomicU64::new(0);

/// Mass-solver sensitivity: `|d(miss_km)/d(mass_kg)|` at the converged root,
/// taken from Brent's own last secant.
///
/// Answers whether an endpoint POSITION error on the free-flight leg reaches
/// the SOLVED MASS, which is the science answer. A propagation bias of
/// `dx` km shifts the root by `dx / |slope|` kg to first order, so a small
/// slope means a large mass error and a large slope means the breach is
/// nominal. Stored in nano-units to live in atomics.
pub(crate) static MASS_SENS_COUNT: ScalarCounter = new_scalar_counter();
/// Sum of |slope| in km/kg, scaled by 1e9.
pub(crate) static MASS_SENS_SUM_NANO: ScalarCounter = new_scalar_counter();
/// Smallest |slope| seen, scaled by 1e9, together with the converged mass it
/// belongs to, scaled by 1e6.
///
/// **One lock, because the PAIRING is the property and no atomic form of it
/// fits.** This used to be a `fetch_min` on one sharded counter followed by a
/// separate `store` on a second, with a comment claiming the sharding made the
/// pair safe. It did not. Shards are handed out per THREAD (`current_shard`
/// returns `NEXT_SHARD & (NSHARD - 1)`, counting every thread ever created, and
/// `nd_sched::run_cells` spawns fresh scoped OS threads per call), so two live
/// threads share a shard as a matter of course, and they interleave as: A
/// `fetch_min`, B `fetch_min` with a smaller slope, B `store`, A `store` --
/// leaving B's slope beside A's mass. Not a lost sample: a wrong number in the
/// diagnostic that sizes the mass-error bar.
///
/// **Packing both into one `AtomicU64` was tried and MEASURED to fail.** Two
/// 32-bit lanes cap the slope at 4.295 km/kg, and a production run reports
/// `min_km_per_kg` of **38.29** against a mean of **2.37e3** -- four orders of
/// magnitude past the lane. Under the packed form the same run printed the
/// saturation ceiling as its minimum and paired it with the wrong row's mass,
/// moving `rel_at_min` from 1.101e-2 to 1.899e1. Both quantities need the full
/// `u64`, so the pair cannot be made indivisible by width.
///
/// A lock is affordable here in a way it would not be one level down:
/// [`record_mass_sensitivity`] runs once per CONVERGED MASS ROW, about 3,400
/// times in a multi-minute campaign cell, not once per RHS evaluation.
/// `u64::MAX` is the unset sentinel, exactly as it was in the min counter.
static MASS_MIN_SENS: Mutex<(u64, u64)> = Mutex::new((u64::MAX, 0));

/// As [`MASS_MIN_SENS`], for the largest |slope| in the corpus, with `0` as the
/// unset sentinel. See [`merge_mass_max_sens`] for why both ends are needed.
static MASS_MAX_SENS: Mutex<(u64, u64)> = Mutex::new((0, 0));

/// `slope_km_per_kg` is `(f(xb) - f(xa)) / (xb - xa)` from the final Brent
/// secant, where `f` is miss distance in km and `x` is dust mass in kg.
/// Sum of every row's converged mass, in micrograms, and the row count.
///
/// The MEAN converged mass over all rows needs no row-matching across runs, so
/// it separates the two channels that an end-to-end mass shift mixes: if
/// per-row solve error is milligram-scale the mean barely moves, and any
/// gram-scale shift in the SELECTED mass is therefore discrete Pareto
/// re-selection rather than per-row inaccuracy.
pub(crate) static MASS_SUM_MICROG: ScalarCounter = new_scalar_counter();

/// Which strict-HF lowering row a recorded mass belongs to.
///
/// **This exists because capture position is not an identity.** The dump used
/// to be a bare `Vec<f64>` documented as "batch/row order", and every consumer
/// therefore joined two runs by index. Two separate things break that, and it
/// is worth keeping them apart because only the first has actually bitten:
///
/// * **The row SET moves.** A solver or integrator change that drops or adds
///   one row -- measured, 3,397 against 3,396 -- shifts every index after it,
///   so index *i* names a different physical row in the two arms. This is
///   silent: both dumps are the right length-ish and every line parses. It has
///   produced two published artifacts on this project, a median 39% "movement"
///   and a median 9.6% one, both noise from a permuted join and both nearly
///   reported as physics.
/// * **Capture order is not fixed across processes.** Within one lowering the
///   flush order is deterministic, but the campaign runs cells concurrently
///   (`nd_sched::run_cells`) and every cell appends to this one buffer, so
///   blocks land in completion order.
///
/// The four fields are the pipeline's own row identity: which design, which
/// event of that design, which transfer candidate, which mass fraction.
///
/// **`design_key` is the CALLER's design identity, not the lowering's own
/// `event_slot`.** The first version of this key used `event_slot` directly and
/// it collided four ways over a four-design harness run, because the production
/// stream lowers designs in GROUPS and every group restarts its slot numbering
/// at zero. A key that looks total and is not is worse than no key: it sorts
/// cleanly and still joins the wrong rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SolvedMassRowKey {
    /// The caller's stable identity for the design, e.g. its population index.
    pub design_key: usize,
    /// Index into that design's event axis.
    pub event_index: usize,
    /// Index into that design-event's transfer front.
    pub candidate_index: usize,
    /// Index into the candidate's mass-fraction sweep.
    pub fraction_index: usize,
}

/// One keyed row of the converged-mass dump.
#[derive(Clone, Copy, Debug)]
pub struct SolvedMassRow {
    pub key: SolvedMassRowKey,
    pub mass_kg: f64,
}

/// Every strict-HF batch row's CONVERGED mass, keyed by [`SolvedMassRowKey`],
/// with the mass held as raw bits.
///
/// `MASS_SUM_MICROG` above is an aggregate and `PROP_MASSSENS` records only
/// what reaches `solve_single_event_hf_internal`'s own Brent loop -- which on
/// the production validate-only path is the LF SEED, not the HF answer. Neither
/// can say how far a GIVEN row's solved mass moves when the integrator or the
/// solver tolerance changes, and that per-row displacement is the only thing
/// that sizes a tolerance floor.
///
/// Written from the strict-HF lowering's batch flush, which is the lowest layer
/// that knows a row's identity, and outside any function that does
/// floating-point work on the answer, so enabling it cannot perturb the
/// arithmetic. Off unless `ND_MASS_ROW_DUMP=1`.
static SOLVED_MASS_ROWS: Mutex<Vec<(SolvedMassRowKey, u64)>> = Mutex::new(Vec::new());
static SOLVED_MASS_DUMP_ON: OnceLock<bool> = OnceLock::new();
/// The diagnostic needs one strict-HF batch, not an unbounded process history.
/// A failed capture invalidates its caller instead of silently retaining a prefix.
const MAX_SOLVED_MASS_ROWS: usize = 1_048_576;

/// Both the limit check and the reservation use the iterator's UPPER size
/// bound, so a caller that filters rows out mid-stream is still checked before
/// anything is appended. An iterator with no upper bound is refused rather than
/// let grow the buffer unbounded.
fn append_solved_mass_rows<I>(
    rows: &mut Vec<(SolvedMassRowKey, u64)>,
    incoming: I,
    limit: usize,
) -> CensusResult<()>
where
    I: Iterator<Item = (SolvedMassRowKey, f64)>,
{
    let bound = incoming
        .size_hint()
        .1
        .ok_or(PropagationCensusError::CounterOverflow)?;
    let next_len = rows
        .len()
        .checked_add(bound)
        .ok_or(PropagationCensusError::CounterOverflow)?;
    if next_len > limit {
        return Err(PropagationCensusError::Allocation);
    }
    rows.try_reserve(bound)
        .map_err(|_| PropagationCensusError::Allocation)?;
    rows.extend(incoming.map(|(key, mass)| (key, mass.to_bits())));
    Ok(())
}

fn solved_mass_dump_enabled() -> bool {
    *SOLVED_MASS_DUMP_ON
        .get_or_init(|| std::env::var("ND_MASS_ROW_DUMP").is_ok_and(|value| value == "1"))
}

/// Append one batch's converged masses, each carrying its own row key.
///
/// The iterator is consumed only when the dump is enabled, so a caller that
/// builds keys lazily pays nothing in a normal run.
///
/// # Errors
///
/// Returns a typed error when the optional evidence cannot retain every row.
pub fn record_solved_mass_rows<I>(rows: I) -> CensusResult<()>
where
    I: IntoIterator<Item = (SolvedMassRowKey, f64)>,
{
    if !solved_mass_dump_enabled() {
        return Ok(());
    }
    let mut stored = census_lock(&SOLVED_MASS_ROWS)?;
    append_solved_mass_rows(&mut stored, rows.into_iter(), MAX_SOLVED_MASS_ROWS)
}

/// Every recorded converged mass so far, in the order it was recorded.
///
/// Capture order is NOT a stable identity -- see [`SolvedMassRowKey`]. Sort by
/// `key` before comparing two runs; never zip two of these by index.
///
/// # Errors
///
/// Returns a typed error when evidence storage is poisoned or cannot be copied.
pub fn solved_mass_rows() -> CensusResult<Vec<SolvedMassRow>> {
    let rows = census_lock(&SOLVED_MASS_ROWS)?;
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(rows.len())
        .map_err(|_| PropagationCensusError::Allocation)?;
    copied.extend(rows.iter().map(|(key, bits)| SolvedMassRow {
        key: *key,
        mass_kg: f64::from_bits(*bits),
    }));
    drop(rows);
    Ok(copied)
}

/// Which front row an event's reported answer was READ FROM.
///
/// `EventResult::mass_kg` is `event_front.first().mass` -- the minimum-rank
/// support point of a normalized front, i.e. a discrete argmin. Such a value can
/// move for two unrelated reasons: the numbers under a FIXED support point
/// changed, or the argmin selected a DIFFERENT support point whose numbers were
/// never perturbed at all. The reported mass alone cannot tell those apart, and
/// this project has twice drawn a numerical conclusion from a re-selection
/// (`docs/PART_A_RESULTS_MATRIX.md`, and the withdrawn "2,588x" of the
/// 2026-08-05 audit §12).
///
/// `rank` is the support point's identity: `candidate_index * fraction_count +
/// fraction_index + 1`, so an unchanged `rank` across two arms means the same
/// row was selected and any mass movement is real propagated signal.
///
/// One record per lowered design-event -- 8 for the standard census -- so this
/// is not a hot path. Off unless `ND_MASS_ROW_DUMP=1`, the same gate as
/// [`SolvedMassRowKey`], because the two dumps only answer the question
/// together.
#[derive(Clone, Debug)]
pub struct EventSupportPoint {
    /// The caller's stable design identity, as in [`SolvedMassRowKey`].
    pub design_key: usize,
    /// Index into that design's event axis.
    pub event_index: usize,
    /// Minimum rank on the normalized front, 1-based.
    pub rank: i64,
    /// The front row's own `candidate_id`, `"{candidate_index}@{fraction}"`.
    pub candidate_id: String,
}

static EVENT_SUPPORT_POINTS: Mutex<Vec<EventSupportPoint>> = Mutex::new(Vec::new());
/// One per lowered design-event. The bound is four orders above the census
/// shape so that tripping it means a caller is recording somewhere unintended.
const MAX_EVENT_SUPPORT_POINTS: usize = 65_536;

/// Record which front row one design-event's answer was read from.
///
/// # Errors
///
/// Returns a typed error when evidence storage is poisoned or full.
pub fn record_event_support_point(point: EventSupportPoint) -> CensusResult<()> {
    if !solved_mass_dump_enabled() {
        return Ok(());
    }
    let mut stored = census_lock(&EVENT_SUPPORT_POINTS)?;
    if stored.len() >= MAX_EVENT_SUPPORT_POINTS {
        return Err(PropagationCensusError::Allocation);
    }
    stored
        .try_reserve(1)
        .map_err(|_| PropagationCensusError::Allocation)?;
    stored.push(point);
    drop(stored);
    Ok(())
}

/// Every recorded support point so far, in the order it was recorded.
///
/// Capture order is not an identity for the same reason [`SolvedMassRowKey`]
/// says so; sort by `(design_key, event_index)` before comparing two runs.
///
/// # Errors
///
/// Returns a typed error when evidence storage is poisoned or cannot be copied.
pub fn event_support_points() -> CensusResult<Vec<EventSupportPoint>> {
    let stored = census_lock(&EVENT_SUPPORT_POINTS)?;
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(stored.len())
        .map_err(|_| PropagationCensusError::Allocation)?;
    copied.extend(stored.iter().cloned());
    drop(stored);
    Ok(copied)
}

pub fn record_mass_sensitivity(mass_kg: f64, slope_km_per_kg: f64) {
    let slope = slope_km_per_kg.abs();
    if !slope.is_finite() || slope <= 0.0 || !mass_kg.is_finite() {
        return;
    }
    let Some(nano) = nonnegative_f64_to_u64(slope * 1.0e9) else {
        return;
    };
    // 1 kg = 1e9 MICROgrams. The `MASS_SUM_MICROG` counter name was always
    // right; the local and the printed field said "nano", which is 1000x off.
    // Anyone dividing the printed value by 1e12 to get kg was 1000x low.
    let Some(mass_micrograms) = nonnegative_f64_to_u64(mass_kg.abs() * 1.0e9) else {
        return;
    };
    scalar_add_value(&MASS_SUM_MICROG, mass_micrograms);
    scalar_add(&MASS_SENS_COUNT);
    scalar_add_value(&MASS_SENS_SUM_NANO, nano);
    // 1 kg = 1e6 MILLIgrams. Round-trip was always correct; the name was not.
    // A mass too large to scale saturates rather than dropping the sample, so
    // the min and the aggregates above always describe the same set of rows.
    let mass_milligrams = nonnegative_f64_to_u64(mass_kg.abs() * 1.0e6).unwrap_or(u64::MAX);
    // Both halves under one lock, so no thread can separate the slope from the
    // mass it arrived with. See [`MASS_MIN_SENS`].
    let mut slot = mass_min_sens_lock();
    *slot = merge_mass_min_sens(*slot, nano, mass_milligrams);
    drop(slot);
    // The MAXIMUM is the binding case for `xtol`, and it is a different row from
    // the minimum. `rtol` converts a mass interval INTO kilometres, so the
    // smallest slope bounds it; `xtol` converts the arc's kilometres BACK into
    // kilograms, so the largest slope is what makes a mass tolerance too coarse.
    // Reporting only the min would answer the wrong direction of the question.
    let mut max_slot = mass_max_sens_lock();
    *max_slot = merge_mass_max_sens(*max_slot, nano, mass_milligrams);
    drop(max_slot);
}

/// Fold one `(slope, mass)` observation into the running pair.
///
/// Split out so the pairing rule is testable without the global lock: the whole
/// point of this fix is that the mass travels WITH the slope, and a test that
/// reimplemented the comparison would prove nothing about production.
const fn merge_mass_min_sens(slot: (u64, u64), nano: u64, mass_milli: u64) -> (u64, u64) {
    if nano < slot.0 {
        (nano, mass_milli)
    } else {
        slot
    }
}

/// The mass-sensitivity pair, recovering from poisoning rather than failing.
///
/// The critical section is one integer compare and one tuple write, so it
/// cannot panic and cannot leave the pair half-written. A poisoned lock here
/// therefore means some OTHER thread panicked while this diagnostic happened to
/// be held, and refusing to record would lose evidence for no gain.
fn mass_min_sens_lock() -> std::sync::MutexGuard<'static, (u64, u64)> {
    MASS_MIN_SENS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `(min_nano, mass_milli)`: the smallest |slope| and the mass recorded with it.
fn mass_min_sens_with_mass() -> (u64, u64) {
    *mass_min_sens_lock()
}

/// Fold one observation into the running maximum, mass carried alongside.
const fn merge_mass_max_sens(slot: (u64, u64), nano: u64, mass_milli: u64) -> (u64, u64) {
    if nano > slot.0 {
        (nano, mass_milli)
    } else {
        slot
    }
}

/// As [`mass_min_sens_lock`], for the maximum. `0` is the unset sentinel here.
fn mass_max_sens_lock() -> std::sync::MutexGuard<'static, (u64, u64)> {
    MASS_MAX_SENS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `(max_nano, mass_milli)`: the largest |slope| and the mass recorded with it.
fn mass_max_sens_with_mass() -> (u64, u64) {
    *mass_max_sens_lock()
}
/// Baseline-state cache outcomes at the `cached_r_state` guard in `rhs.rs`.
///
/// Counted rather than inferred. The Vern9 tableau predicts 2 hits per 16
/// stages -- `c[14]` and `c[15]` are both exactly 1.0 and the next step's
/// `c[0]` lands within Kahan compensation of them -- but that is a read of the
/// coefficients, not a measurement, and it understates the real rate because it
/// does not account for `h` collapsing after a rectification restart.
///
/// **The "7x" this comment used to claim was wrong.** Measured 2026-08-05 over
/// a 4-design x 2-event census: 12.50% predicted against **20.44%** observed,
/// which is **1.63x**, and the excess is fully accounted for -- ~1.86M steps
/// (9.2% of all steps, 42% of the genuinely-small ones) hitting on all 16
/// evaluations once `h` falls below `tol / 0.03462`.
///
/// **These counters do not cover all baseline work.** They sit at the
/// `RHSCache` guard, which is consulted once per RHS EVALUATION. `BaselineCalculator`
/// (`rhs.rs`, exact-bit key, held by `EnckeEventHandler`) is consulted about
/// once per accepted STEP -- roughly 20M further baseline evaluations on the
/// production mass path, nearly all misses, because an exact-bit key at an
/// always-new `next_t` overwrites its single slot every time. So this pair
/// covers ~94% of baseline work by volume: quote it as the `RHSCache` baseline
/// hit rate, never as "the baseline cache hit rate".
///
/// That other half is no longer uncounted -- see [`BASELINE_CALC_HIT`] -- but
/// the naming rule stands, because the two are different caches with different
/// keys, different consult rates and different hit rates. Report them
/// SEPARATELY or report their sum explicitly; never let one wear the other's
/// label.
pub(crate) static BASELINE_HIT: TagCounter = new_tag_counter();
pub(crate) static BASELINE_MISS: TagCounter = new_tag_counter();
/// A THIRD baseline cache, and it wears its own label for the reason the note
/// above gives: `LightyearRHS::stage_baselines`, keyed on exact `tof` bits and
/// filled a whole RK step at a time before the stage loop runs.
///
/// `BASELINE_STAGE_HIT` counts queries the prefilled table answered;
/// `BASELINE_PREFILL` counts the four-lane solves that filled it, so
/// `4 * BASELINE_PREFILL` is the solve budget spent and `BASELINE_STAGE_HIT`
/// is what it bought. They are NOT a hit/miss pair: a query the table misses
/// is counted by `BASELINE_MISS` or `BASELINE_HIT` below as before.
pub(crate) static BASELINE_STAGE_HIT: TagCounter = new_tag_counter();
pub(crate) static BASELINE_PREFILL: TagCounter = new_tag_counter();
/// The other baseline cache: `BaselineCalculator::get_baseline_state`, keyed on
/// the EXACT bits of `tof` and consulted from event detection rather than from
/// the RHS.
///
/// Added because a census that reported only the pair above called its result
/// "the Encke baseline cache hit rate", which is the mislabelling the paragraph
/// above forbids. Both are needed to price the baseline conversion, since a
/// miss in EITHER runs the same `equinoc2eci_impl`.
pub(crate) static BASELINE_CALC_HIT: TagCounter = new_tag_counter();
pub(crate) static BASELINE_CALC_MISS: TagCounter = new_tag_counter();
thread_local! {
    static TAG: Cell<usize> = const { Cell::new(TAG_OTHER) };
    /// This thread's counter shard. `usize::MAX` means "not yet assigned"; see
    /// [`current_shard`].
    static SHARD: Cell<usize> = const { Cell::new(usize::MAX) };
    /// Per-thread running total of mass-miss propagations, so one mass solve
    /// can be measured as a delta without racing other workers.
    static TL_MASS_PROPS: Cell<u64> = const { Cell::new(0) };
    /// Per-thread running total of ALL propagations, so one release-control LM
    /// pass (or one LM iteration inside it) can be measured as a delta without
    /// racing other workers. Census-only; see [`tl_all_props`].
    #[cfg(feature = "prop-census")]
    static TL_ALL_PROPS: Cell<u64> = const { Cell::new(0) };
}

#[cfg(feature = "prop-census")]
/// This thread's running propagation total. Only a DELTA across two reads on
/// the same thread means anything; the absolute value is arbitrary.
///
/// # Errors
///
/// This accessor currently cannot fail. Its result type keeps callers on the
/// same contract as other census accessors.
#[inline]
pub fn tl_all_props() -> CensusResult<u64> {
    Ok(TL_ALL_PROPS.with(Cell::get))
}

#[cfg(not(feature = "prop-census"))]
/// This thread's running propagation total.
///
/// # Errors
///
/// This accessor currently cannot fail. Its result type keeps callers on the
/// same contract as other census accessors.
#[inline]
pub const fn tl_all_props() -> CensusResult<u64> {
    Ok(0)
}

#[inline]
pub(crate) fn current_tag() -> usize {
    TAG.with(Cell::get)
}

#[inline]
pub(crate) fn set_tag(tag: usize) -> usize {
    TAG.with(|cell| cell.replace(tag.min(NTAG - 1)))
}

pub struct TagGuard(usize);

impl Drop for TagGuard {
    fn drop(&mut self) {
        set_tag(self.0);
    }
}

/// Attribute every propagation started in this scope to `tag`, restoring the
/// previous tag on drop. Nested scopes shadow.
#[inline]
#[must_use]
pub fn scope(tag: usize) -> TagGuard {
    TagGuard(set_tag(tag))
}

/// A `scope` that exists only when the duplicate census is compiled in.
///
/// Sub-phase tags reattribute propagations *out of* their parent tag, so
/// applying them unconditionally would silently change what `release_control`
/// means for every existing reader. Gating them keeps the default counts
/// exactly as they were and makes the finer split something you ask for.
///
/// Returns an `Option` so the disabled arm holds no guard and drops to nothing.
#[cfg(feature = "prop-census")]
#[inline]
#[must_use]
pub fn census_scope(tag: usize) -> Option<TagGuard> {
    let guard = scope(tag);
    tag_add(&SCOPE_ENTRIES, current_tag());
    Some(guard)
}

#[cfg(not(feature = "prop-census"))]
#[inline]
#[must_use]
pub const fn census_scope(_tag: usize) -> Option<TagGuard> {
    None
}

/// Census-enabled accounting is exact: all source counters preflight before
/// any global counter changes, and global additions reject wraparound.
///
/// Exhaustion invalidates the CENSUS, never the propagation: the caller gets
/// `Ok` and integrates, and `report` refuses the epoch. See `CENSUS_INVALID`.
///
/// The `Result` is therefore always `Ok` in THIS build. It stays because the
/// `cfg(not(feature))` twin below is genuinely fallible and the call sites in
/// `integrator.rs` are deliberately `cfg`-free.
#[expect(clippy::unnecessary_wraps)]
#[cfg(feature = "prop-census")]
#[inline]
pub(crate) fn bump_propagation(start_time_s: f64, end_time_s: f64) -> CensusResult<()> {
    let result = (|| {
        let tag = current_tag();
        let next_mass_props = if tag == TAG_MASS_MISS {
            Some(
                TL_MASS_PROPS
                    .with(|cell| cell.get().checked_add(1))
                    .ok_or(PropagationCensusError::CounterOverflow)?,
            )
        } else {
            None
        };
        let next_all_props = TL_ALL_PROPS
            .with(|cell| cell.get().checked_add(1))
            .ok_or(PropagationCensusError::CounterOverflow)?;

        // Validate every thread-local counter before a global observation changes.
        // If the second global update fails, the invalidation latch below
        // quarantines the preceding update until reset.
        checked_tag_add(&PROPAGATIONS, tag, 1)?;
        let span = (end_time_s - start_time_s).abs();
        if span.is_finite() {
            if let Some(span_ms) = nonnegative_f64_to_u64(span * 1000.0) {
                checked_tag_add(&SPAN_MS, tag, span_ms)?;
            }
        }
        if let Some(next) = next_mass_props {
            TL_MASS_PROPS.with(|cell| cell.set(next));
        }
        TL_ALL_PROPS.with(|cell| cell.set(next_all_props));
        Ok(())
    })();
    if let Err(error) = result {
        invalidate_census(error);
    }
    Ok(())
}

/// Production fast path keeps the pre-census relaxed accounting operations.
/// These counters feed local diagnostics only when `prop-census` is off; no
/// Part-A receipt or campaign decision reads them. The Result shape preserves
/// one caller contract while optimizing to the same relaxed updates as before.
#[cfg(not(feature = "prop-census"))]
#[inline]
pub(crate) fn bump_propagation(start_time_s: f64, end_time_s: f64) -> CensusResult<()> {
    let tag = current_tag();
    relaxed_tag_add(&PROPAGATIONS, tag, 1)?;
    if tag == TAG_MASS_MISS {
        TL_MASS_PROPS.with(|cell| cell.set(cell.get().saturating_add(1)));
    }
    let span = (end_time_s - start_time_s).abs();
    if span.is_finite() {
        if let Some(span_ms) = nonnegative_f64_to_u64(span * 1000.0) {
            relaxed_tag_add(&SPAN_MS, tag, span_ms)?;
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn bump_steps(steps: usize) {
    tag_add_value(&STEPS, current_tag(), usize_to_u64(steps));
}

#[inline]
pub(crate) fn bump_saturated(saturated: usize) {
    tag_add_value(&SATURATED, current_tag(), usize_to_u64(saturated));
}

#[inline]
pub(crate) fn observe_min_h(min_accepted_h: f64) {
    // Written as a positive test rather than two negations: `!(x > 0.0)` is a
    // NaN trap that reads like a `<=`, and the two forms differ precisely on
    // the input this guard exists to reject.
    if !(min_accepted_h.is_finite() && min_accepted_h > 0.0) {
        return;
    }
    if let Some(nanos) = nonnegative_f64_to_u64(min_accepted_h * 1.0e9) {
        min_observe(&MIN_ACCEPTED_H_NS, nanos);
    }
}

#[inline]
pub(crate) fn observe_underflow(accepts: usize) {
    if accepts > 0 {
        UNDERFLOW_ACCEPTS.fetch_add(usize_to_u64(accepts), Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn observe_cache_cluster(total: usize, untruncated: usize) {
    if total > 0 {
        scalar_add_value(&CACHE_CLUSTER_STEPS, usize_to_u64(total));
        scalar_add_value(&CACHE_CLUSTER_STEPS_UNTRUNCATED, usize_to_u64(untruncated));
    }
}

/// Span at or below which a solver entry is a clamped root-refinement leg
/// rather than an Encke deviation rebase.
///
/// Root legs are clamped to `MAX_ROOT_REFINEMENT_STEP_S = 10 s`; rebases run
/// ~574 s on the production mass arc. 60 s sits an order of magnitude clear of
/// both, so the split is not sensitive to where exactly it is put — but it IS a
/// threshold, and the two populations are reported separately precisely so a
/// reader can see whether it mattered.
#[cfg(feature = "prop-census")]
const RAMP_SHORT_SEGMENT_S: f64 = 60.0;

/// Ramp accumulators: `[boundary][population][slot]`.
///
/// Population 0 = short span, 1 = long span, split at [`RAMP_SHORT_SEGMENT_S`].
/// Boundary is `SegmentBoundary` in the order [`ramp_boundary_index`] fixes.
/// Slots `0..RAMP_PROBE_STEPS` are the opening accepted steps in order; the
/// last slot is the sustained tail.
///
/// **The two axes answer different questions and neither substitutes for the
/// other.** Span says how much work an entry has to do; boundary says whether
/// the restart that opened it discarded anything it could have kept. A short
/// entry can sit on a continuable boundary and a long one on a rebase, so a
/// deficit measured on the span axis alone cannot be attributed to a lever
/// that only the boundary axis can reach — which is exactly the inference the
/// eclipse-leg h-carry proposal rests on.
///
/// Nanoseconds, summed, with a parallel count — a mean cannot be formed from a
/// sum alone and a mean-of-means over entries with different step counts is not
/// the mean this question needs.
#[cfg(feature = "prop-census")]
static RAMP_H_NS: [[[AtomicU64; RAMP_SLOTS]; 2]; RAMP_BOUNDARIES] =
    [const { [const { [const { AtomicU64::new(0) }; RAMP_SLOTS] }; 2] }; RAMP_BOUNDARIES];
#[cfg(feature = "prop-census")]
static RAMP_COUNT: [[[AtomicU64; RAMP_SLOTS]; 2]; RAMP_BOUNDARIES] =
    [const { [const { [const { AtomicU64::new(0) }; RAMP_SLOTS] }; 2] }; RAMP_BOUNDARIES];
/// Accepted steps per `[boundary][population]`, so a slot mean can be weighed
/// against the work its population actually stands for.
#[cfg(feature = "prop-census")]
static RAMP_STEPS: [[AtomicU64; 2]; RAMP_BOUNDARIES] =
    [const { [const { AtomicU64::new(0) }; 2] }; RAMP_BOUNDARIES];
#[cfg(feature = "prop-census")]
const RAMP_SLOTS: usize = crate::odesolve::solver::RAMP_PROBE_STEPS + 1;
#[cfg(feature = "prop-census")]
const RAMP_BOUNDARIES: usize = 3;
#[cfg(feature = "prop-census")]
const RAMP_BOUNDARY_NAMES: [&str; RAMP_BOUNDARIES] = ["arc_start", "rebased", "event_cont"];

/// Solver entries, accepted steps and RHS evaluations split by whether the
/// entry ran under the eclipse root-refinement `dt_max` clamp.
///
/// The ramp instrument above cannot answer this. Its population axis is SPAN,
/// and the bracket-replay leg is a long span running under the clamp — it
/// replays one accepted step of ~65 s at `MAX_ROOT_REFINEMENT_STEP_S = 10 s`.
/// So a span split files bracket-replay work with the unclamped Encke
/// segments and reports the clamp as cheaper than it is.
///
/// Class 0 = unclamped (production `dt_max`, 300 s), class 1 = clamped.
/// Discriminated on the `dt_max` the entry actually ran with rather than on a
/// leg name, because the clamp is applied at two separate coordinator sites
/// and a name would have to be kept in sync with both.
#[cfg(feature = "prop-census")]
const LEG_CLASSES: usize = 2;
#[cfg(feature = "prop-census")]
const LEG_CLASS_NAMES: [&str; LEG_CLASSES] = ["unclamped", "clamped"];
/// Split point between the two `dt_max` populations.
///
/// Root legs clamp to 10 s and production runs 300 s, so 60 s sits an order of
/// magnitude clear of both — the same reasoning, and the same value, as
/// [`RAMP_SHORT_SEGMENT_S`], but on a different quantity: that one splits
/// SPANS, this one splits step CAPS.
#[cfg(feature = "prop-census")]
const LEG_CLAMPED_DT_MAX_S: f64 = 60.0;
#[cfg(feature = "prop-census")]
static LEG_COUNT: [AtomicU64; LEG_CLASSES] = [const { AtomicU64::new(0) }; LEG_CLASSES];
#[cfg(feature = "prop-census")]
static LEG_STEPS: [AtomicU64; LEG_CLASSES] = [const { AtomicU64::new(0) }; LEG_CLASSES];
#[cfg(feature = "prop-census")]
static LEG_EVALS: [AtomicU64; LEG_CLASSES] = [const { AtomicU64::new(0) }; LEG_CLASSES];
#[cfg(feature = "prop-census")]
static LEG_REJECTED: [AtomicU64; LEG_CLASSES] = [const { AtomicU64::new(0) }; LEG_CLASSES];
#[cfg(feature = "prop-census")]
static LEG_SPAN_NS: [AtomicU64; LEG_CLASSES] = [const { AtomicU64::new(0) }; LEG_CLASSES];
/// Accepted steps per solver entry, bucketed, per class.
///
/// A mean cannot answer the multistep question. An Adams-Bashforth-Moulton
/// method of order k needs k-1 steps of history before it reaches its order,
/// and pays a restart at every leg boundary; what decides it is the fraction
/// of steps living on entries LONG enough to amortize that restart, which is a
/// property of the distribution's upper tail and not of its mean. Bucket
/// `i < LEG_HIST_LEN-1` is exactly `i` accepted steps; the last bucket is the
/// saturating tail.
#[cfg(feature = "prop-census")]
const LEG_HIST_LEN: usize = 64;
#[cfg(feature = "prop-census")]
static LEG_STEPS_HIST: [[AtomicU64; LEG_HIST_LEN]; LEG_CLASSES] =
    [const { [const { AtomicU64::new(0) }; LEG_HIST_LEN] }; LEG_CLASSES];

/// Binary-eclipse coordinator site owning one solver entry.
///
/// This axis is deliberately narrower than [`LEG_CLASS_NAMES`]. It answers
/// which coordinator obligation bought clamped work, while the existing axis
/// answers only whether a cap was active. Entries outside the binary-eclipse
/// coordinator are unowned and never appear here.
#[cfg(feature = "prop-census")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EclipseTransactionSite {
    Main,
    Refine,
    Proof,
    Window,
}

#[cfg(feature = "prop-census")]
impl EclipseTransactionSite {
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Main => 0,
            Self::Refine => 1,
            Self::Proof => 2,
            Self::Window => 3,
        }
    }
}

#[cfg(feature = "prop-census")]
const ECLIPSE_TRANSACTION_SITES: usize = 4;
#[cfg(feature = "prop-census")]
const ECLIPSE_TRANSACTION_SITE_NAMES: [&str; ECLIPSE_TRANSACTION_SITES] =
    ["main", "refine", "proof", "window"];
#[cfg(feature = "prop-census")]
static ECLIPSE_TRANSACTION_SITE_LEGS: [AtomicU64; ECLIPSE_TRANSACTION_SITES] =
    [const { AtomicU64::new(0) }; ECLIPSE_TRANSACTION_SITES];
#[cfg(feature = "prop-census")]
static ECLIPSE_TRANSACTION_SITE_STEPS: [AtomicU64; ECLIPSE_TRANSACTION_SITES] =
    [const { AtomicU64::new(0) }; ECLIPSE_TRANSACTION_SITES];
#[cfg(feature = "prop-census")]
static ECLIPSE_TRANSACTION_SITE_EVALS: [AtomicU64; ECLIPSE_TRANSACTION_SITES] =
    [const { AtomicU64::new(0) }; ECLIPSE_TRANSACTION_SITES];
#[cfg(feature = "prop-census")]
static ECLIPSE_TRANSACTION_SITE_REJECTED: [AtomicU64; ECLIPSE_TRANSACTION_SITES] =
    [const { AtomicU64::new(0) }; ECLIPSE_TRANSACTION_SITES];
#[cfg(feature = "prop-census")]
static ECLIPSE_TRANSACTION_SITE_FANTASY_REMOVABLE_STEPS: [AtomicU64; ECLIPSE_TRANSACTION_SITES] =
    [const { AtomicU64::new(0) }; ECLIPSE_TRANSACTION_SITES];
#[cfg(feature = "prop-census")]
static ECLIPSE_TRANSACTION_SITE_FANTASY_REMOVABLE_EVALS: [AtomicU64; ECLIPSE_TRANSACTION_SITES] =
    [const { AtomicU64::new(0) }; ECLIPSE_TRANSACTION_SITES];

#[cfg(feature = "prop-census")]
thread_local! {
    static ECLIPSE_TRANSACTION_SITE: Cell<Option<(u64, EclipseTransactionSite)>> = const { Cell::new(None) };
}

/// Restores the prior site on normal return, `?`, or unwind. Nested scopes
/// shadow rather than double-count because [`observe_leg`] reads one site.
#[cfg(feature = "prop-census")]
pub(crate) struct EclipseTransactionSiteGuard {
    epoch: u64,
    prior: Option<(u64, EclipseTransactionSite)>,
}

#[cfg(feature = "prop-census")]
impl Drop for EclipseTransactionSiteGuard {
    fn drop(&mut self) {
        let current_epoch = CENSUS_EPOCH.load(Ordering::Acquire);
        let restore = if self.epoch == current_epoch
            && self.prior.is_some_and(|(epoch, _)| epoch == current_epoch)
        {
            self.prior
        } else {
            None
        };
        ECLIPSE_TRANSACTION_SITE.with(|site| site.set(restore));
    }
}

#[cfg(feature = "prop-census")]
#[must_use]
pub(crate) fn eclipse_transaction_scope(
    site: EclipseTransactionSite,
) -> EclipseTransactionSiteGuard {
    let epoch = CENSUS_EPOCH.load(Ordering::Acquire);
    let prior = ECLIPSE_TRANSACTION_SITE.with(|current| current.replace(Some((epoch, site))));
    EclipseTransactionSiteGuard { epoch, prior }
}

#[cfg(feature = "prop-census")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct EclipseTransactionSiteCensus {
    pub(crate) legs: u64,
    pub(crate) steps: u64,
    pub(crate) evals: u64,
    pub(crate) rejected: u64,
    pub(crate) fantasy_removable_steps: u64,
    pub(crate) fantasy_removable_evals: u64,
}

#[cfg(feature = "prop-census")]
pub(crate) fn eclipse_transaction_site_snapshot(
) -> [EclipseTransactionSiteCensus; ECLIPSE_TRANSACTION_SITES] {
    let mut out = [EclipseTransactionSiteCensus::default(); ECLIPSE_TRANSACTION_SITES];
    for (index, row) in out.iter_mut().enumerate() {
        *row = EclipseTransactionSiteCensus {
            legs: atomic_load(&ECLIPSE_TRANSACTION_SITE_LEGS, index),
            steps: atomic_load(&ECLIPSE_TRANSACTION_SITE_STEPS, index),
            evals: atomic_load(&ECLIPSE_TRANSACTION_SITE_EVALS, index),
            rejected: atomic_load(&ECLIPSE_TRANSACTION_SITE_REJECTED, index),
            fantasy_removable_steps: atomic_load(
                &ECLIPSE_TRANSACTION_SITE_FANTASY_REMOVABLE_STEPS,
                index,
            ),
            fantasy_removable_evals: atomic_load(
                &ECLIPSE_TRANSACTION_SITE_FANTASY_REMOVABLE_EVALS,
                index,
            ),
        };
    }
    out
}

#[cfg(feature = "prop-census")]
fn observe_eclipse_transaction_site(steps: usize, evals: usize, rejected: usize) {
    #[cfg(test)]
    if !census_test_thread_may_write() {
        return;
    }
    let current_epoch = CENSUS_EPOCH.load(Ordering::Acquire);
    let Some((site_epoch, site)) = ECLIPSE_TRANSACTION_SITE.with(Cell::get) else {
        return;
    };
    if site_epoch != current_epoch {
        return;
    }
    let result = (|| {
        let fantasy_removable_steps = u64::try_from(steps.saturating_sub(1))
            .map_err(|_| PropagationCensusError::CounterOverflow)?;
        let fantasy_removable_evals = u64::try_from(evals.saturating_sub(11))
            .map_err(|_| PropagationCensusError::CounterOverflow)?;
        let steps = u64::try_from(steps).map_err(|_| PropagationCensusError::CounterOverflow)?;
        let evals = u64::try_from(evals).map_err(|_| PropagationCensusError::CounterOverflow)?;
        let rejected =
            u64::try_from(rejected).map_err(|_| PropagationCensusError::CounterOverflow)?;
        let index = site.index();
        checked_atomic_add(&ECLIPSE_TRANSACTION_SITE_LEGS, index, 1)?;
        checked_atomic_add(&ECLIPSE_TRANSACTION_SITE_STEPS, index, steps)?;
        checked_atomic_add(&ECLIPSE_TRANSACTION_SITE_EVALS, index, evals)?;
        checked_atomic_add(&ECLIPSE_TRANSACTION_SITE_REJECTED, index, rejected)?;
        checked_atomic_add(
            &ECLIPSE_TRANSACTION_SITE_FANTASY_REMOVABLE_STEPS,
            index,
            fantasy_removable_steps,
        )?;
        checked_atomic_add(
            &ECLIPSE_TRANSACTION_SITE_FANTASY_REMOVABLE_EVALS,
            index,
            fantasy_removable_evals,
        )
    })();
    if let Err(error) = result {
        invalidate_census(error);
    }
}

/// Record one solver entry's class, work and accepted-step count.
#[cfg(feature = "prop-census")]
pub(crate) fn observe_leg(
    dt_max_s: f64,
    segment_span_s: f64,
    steps: usize,
    evals: usize,
    rejected: usize,
) {
    observe_eclipse_transaction_site(steps, evals, rejected);
    // A non-finite cap is not a clamped leg; treat only a finite small cap as
    // one, so a NaN cannot silently file production work under the clamp.
    let class = usize::from(dt_max_s.is_finite() && dt_max_s <= LEG_CLAMPED_DT_MAX_S);
    let (
        Some(count),
        Some(step_total),
        Some(eval_total),
        Some(reject_total),
        Some(span),
        Some(histogram),
    ) = (
        LEG_COUNT.get(class),
        LEG_STEPS.get(class),
        LEG_EVALS.get(class),
        LEG_REJECTED.get(class),
        LEG_SPAN_NS.get(class),
        LEG_STEPS_HIST.get(class),
    )
    else {
        return;
    };
    count.fetch_add(1, Ordering::Relaxed);
    step_total.fetch_add(usize_to_u64(steps), Ordering::Relaxed);
    eval_total.fetch_add(usize_to_u64(evals), Ordering::Relaxed);
    reject_total.fetch_add(usize_to_u64(rejected), Ordering::Relaxed);
    if let Some(nanos) = nonnegative_f64_to_u64(segment_span_s.abs() * 1.0e9) {
        span.fetch_add(nanos, Ordering::Relaxed);
    }
    if let Some(bucket) = histogram.get(steps.min(LEG_HIST_LEN.saturating_sub(1))) {
        bucket.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "prop-census"))]
#[inline(always)]
pub(crate) const fn observe_leg(
    _dt_max_s: f64,
    _segment_span_s: f64,
    _steps: usize,
    _evals: usize,
    _rejected: usize,
) {
}

/// Position of a boundary kind on the census's boundary axis.
///
/// Written out rather than taken from a discriminant so that reordering the
/// enum cannot silently relabel a landed measurement.
#[cfg(feature = "prop-census")]
const fn ramp_boundary_index(boundary: crate::integrator::SegmentBoundary) -> usize {
    match boundary {
        crate::integrator::SegmentBoundary::ArcStart => 0,
        crate::integrator::SegmentBoundary::Rebased => 1,
        crate::integrator::SegmentBoundary::EventContinuation => 2,
    }
}

/// Record one solver entry's opening step sizes and its sustained tail.
///
/// The question this exists to settle: every solver entry restarts the
/// controller with no memory of the last one, so the opening steps run below
/// whatever the error controller would sustain. That costs ACCEPTED steps,
/// which no rejection counter can see. Two models of how much disagreed by 4.4x
/// (6.2% against 27.2% of all RHS evaluations) and aggregates could not
/// separate them, because `steps/segment` and `sat_frac` implied contradictory
/// step sizes. This measures the ramp and the equilibrium directly instead.
#[cfg(feature = "prop-census")]
pub(crate) fn observe_ramp(
    boundary: crate::integrator::SegmentBoundary,
    segment_span_s: f64,
    first_accepted_h: &[f64],
    tail_h_sum: f64,
    tail_h_count: usize,
) {
    let population = usize::from(segment_span_s > RAMP_SHORT_SEGMENT_S);
    let boundary_index = ramp_boundary_index(boundary);
    let Some(ns_row) = RAMP_H_NS
        .get(boundary_index)
        .and_then(|plane| plane.get(population))
    else {
        return;
    };
    let Some(count_row) = RAMP_COUNT
        .get(boundary_index)
        .and_then(|plane| plane.get(population))
    else {
        return;
    };
    if let Some(steps) = RAMP_STEPS
        .get(boundary_index)
        .and_then(|plane| plane.get(population))
    {
        let opening = first_accepted_h
            .iter()
            .filter(|step_h| step_h.is_finite() && **step_h > 0.0)
            .count();
        steps.fetch_add(
            usize_to_u64(opening).saturating_add(usize_to_u64(tail_h_count)),
            Ordering::Relaxed,
        );
    }
    for (slot, step_h) in first_accepted_h.iter().enumerate() {
        // A zero slot means the entry accepted fewer steps than the probe
        // depth. Counting it would pull the mean toward zero and read as a
        // ramp that never recovers, which is the opposite of the truth.
        if !(step_h.is_finite() && *step_h > 0.0) {
            continue;
        }
        let (Some(ns), Some(count)) = (ns_row.get(slot), count_row.get(slot)) else {
            continue;
        };
        if let Some(nanos) = nonnegative_f64_to_u64(step_h * 1.0e9) {
            ns.fetch_add(nanos, Ordering::Relaxed);
            count.fetch_add(1, Ordering::Relaxed);
        }
    }
    if tail_h_count == 0 || !(tail_h_sum.is_finite() && tail_h_sum > 0.0) {
        return;
    }
    let tail_slot = RAMP_SLOTS.saturating_sub(1);
    let (Some(ns), Some(count)) = (ns_row.get(tail_slot), count_row.get(tail_slot)) else {
        return;
    };
    if let Some(nanos) = nonnegative_f64_to_u64(tail_h_sum * 1.0e9) {
        ns.fetch_add(nanos, Ordering::Relaxed);
        count.fetch_add(usize_to_u64(tail_h_count), Ordering::Relaxed);
    }
}

#[cfg(not(feature = "prop-census"))]
#[inline(always)]
pub(crate) const fn observe_ramp(
    _boundary: crate::integrator::SegmentBoundary,
    _segment_span_s: f64,
    _first_accepted_h: &[f64],
    _tail_h_sum: f64,
    _tail_h_count: usize,
) {
}

/// Calls into the JB2008 adapter, and calls that actually reach the kernel.
///
/// Exists because a self-time SHARE and a nanoseconds-per-CALL figure cannot be
/// compared without this ratio, and two measurements of the JB2008 adapter
/// disagreed 13x while both sides assumed it was 1:1. The adapter has three
/// early returns before the kernel -- missing drivers, a bad UTC JD, and a
/// driver-table lookup miss -- each of which runs the adapter's entry work and
/// calls no kernel. If those fire often the adapter can legitimately rival the
/// kernel's share while costing a fraction of it per call, and both figures are
/// right about different things.
/// Sharded, and these two were the WORST of the four per-evaluation counters:
/// they are not tag-indexed, so before sharding they were two adjacent bare
/// `AtomicU64`s on ONE cache line, taking two read-modify-writes per RHS
/// evaluation from every thread in the process regardless of phase.
pub(crate) static JB_ADAPTER_CALLS: ScalarCounter = new_scalar_counter();
pub(crate) static JB_KERNEL_CALLS: ScalarCounter = new_scalar_counter();

/// `(adapter_calls, kernel_calls)` since `reset`.
#[must_use]
pub fn jb_call_census() -> (u64, u64) {
    (scalar_sum(&JB_ADAPTER_CALLS), scalar_sum(&JB_KERNEL_CALLS))
}

/// Propagation-level RETURN outcomes, counted at the point the value is handed
/// back to the caller.
///
/// Distinct from `PROPAGATIONS`/`SEGMENTS`, which count entries. This pair
/// answers the only question that makes a profile citable: did the propagations
/// whose time was measured actually produce an answer. A harness that reads
/// phase timers cannot tell -- 100% of propagations can return `None` while
/// every timer looks healthy, which is how a full session of numbers was lost.
/// Sharded for the same reason as the JB pair above, which the comment there
/// calls the worst form: neither is tag-indexed, so unsharded they were two
/// adjacent `AtomicU64`s on one cache line taking a read-modify-write from
/// every worker. Per PROPAGATION rather than per evaluation, so roughly 1e4
/// times rarer -- this is hygiene against a shape already proven costly, not a
/// measured win.
static PROP_RETURNS: ScalarCounter = new_scalar_counter();
static PROP_RETURNS_BAD: ScalarCounter = new_scalar_counter();

/// Record one propagation return. `ok` is false for `None`, for an `Err`, and
/// for a nominally successful state carrying a non-finite component.
#[inline]
pub(crate) fn observe_prop_return(ok: bool) {
    scalar_add(&PROP_RETURNS);
    if !ok {
        scalar_add(&PROP_RETURNS_BAD);
    }
}

/// `(returns, bad_returns)` since `reset`.
#[must_use]
pub fn prop_return_census() -> (u64, u64) {
    (scalar_sum(&PROP_RETURNS), scalar_sum(&PROP_RETURNS_BAD))
}

/// Smallest accepted step seen since `reset`, in seconds. `None` if no step was
/// accepted.
///
/// `#[must_use]` became required here when the read moved behind [`min_value`]:
/// clippy stops treating a direct static access as a side effect once the
/// static is only touched by the callee.
#[must_use]
pub(crate) fn min_accepted_h_s() -> Option<f64> {
    let nanos = min_value(&MIN_ACCEPTED_H_NS);
    (nanos != u64::MAX).then(|| u64_to_f64(nanos) / 1.0e9)
}

#[inline]
pub(crate) fn bump_rejected(rejected: usize) {
    tag_add_value(&REJECTED, current_tag(), usize_to_u64(rejected));
}

#[inline]
pub(crate) fn bump_segment() {
    tag_add(&SEGMENTS, current_tag());
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TagCensus {
    pub rhs_evals: u64,
    pub propagations: u64,
    pub segments: u64,
    pub steps: u64,
    pub saturated: u64,
    pub rejected: u64,
    pub baseline_hit: u64,
    pub baseline_miss: u64,
    pub baseline_stage_hit: u64,
    pub baseline_prefill: u64,
    pub baseline_calc_hit: u64,
    pub baseline_calc_miss: u64,
    pub span_s: f64,
}

#[must_use]
pub fn snapshot() -> [TagCensus; NTAG] {
    let mut out = [TagCensus::default(); NTAG];
    for (tag, output) in out.iter_mut().enumerate() {
        *output = TagCensus {
            rhs_evals: tag_sum(&RHS_EVALS, tag),
            propagations: tag_sum(&PROPAGATIONS, tag),
            segments: tag_sum(&SEGMENTS, tag),
            steps: tag_sum(&STEPS, tag),
            saturated: tag_sum(&SATURATED, tag),
            rejected: tag_sum(&REJECTED, tag),
            baseline_hit: tag_sum(&BASELINE_HIT, tag),
            baseline_miss: tag_sum(&BASELINE_MISS, tag),
            baseline_stage_hit: tag_sum(&BASELINE_STAGE_HIT, tag),
            baseline_prefill: tag_sum(&BASELINE_PREFILL, tag),
            baseline_calc_hit: tag_sum(&BASELINE_CALC_HIT, tag),
            baseline_calc_miss: tag_sum(&BASELINE_CALC_MISS, tag),
            span_s: u64_to_f64(tag_sum(&SPAN_MS, tag)) / 1000.0,
        };
    }
    out
}

/// Clear all telemetry and optional census evidence.
///
/// # Errors
///
/// Returns a typed error when a retained-evidence lock was poisoned, leaving
/// telemetry intact.
pub fn reset() -> CensusResult<()> {
    let mut solved_mass_rows = census_lock(&SOLVED_MASS_ROWS)?;
    let mut event_support_points = census_lock(&EVENT_SUPPORT_POINTS)?;
    #[cfg(feature = "prop-census")]
    let mut captured_arcs = census_lock(&CAPTURED_ARCS)?;
    #[cfg(feature = "prop-census")]
    let mut seen_states = census_lock(&SEEN_STATES)?;
    #[cfg(feature = "prop-census")]
    let mut lm_iters = census_lock(&LM_ITERS)?;
    #[cfg(feature = "prop-census")]
    let mut lm_passes = census_lock(&LM_PASSES)?;
    #[cfg(feature = "prop-census")]
    if CENSUS_EPOCH
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |epoch| {
            epoch.checked_add(1)
        })
        .is_err()
    {
        invalidate_census(PropagationCensusError::CounterOverflow);
        return Err(PropagationCensusError::CounterOverflow);
    }

    for tag in 0..NTAG {
        tag_clear(&RHS_EVALS, tag);
        tag_clear(&PROPAGATIONS, tag);
        tag_clear(&SEGMENTS, tag);
        tag_clear(&SATURATED, tag);
        tag_clear(&REJECTED, tag);
        tag_clear(&BASELINE_HIT, tag);
        tag_clear(&BASELINE_MISS, tag);
        tag_clear(&BASELINE_STAGE_HIT, tag);
        tag_clear(&BASELINE_PREFILL, tag);
        tag_clear(&BASELINE_CALC_HIT, tag);
        tag_clear(&BASELINE_CALC_MISS, tag);
        tag_clear(&STEPS, tag);
        tag_clear(&SPAN_MS, tag);
    }
    min_clear(&MIN_ACCEPTED_H_NS);
    scalar_clear(&CACHE_CLUSTER_STEPS);
    scalar_clear(&CACHE_CLUSTER_STEPS_UNTRUNCATED);
    // The ramp accumulators clear with their neighbours. Omitting them made
    // `PROP_RAMP` survive `reset()`, so a harness that resets and re-measures
    // would have read the previous run's samples blended into its own -- and
    // silently, because the numbers stay plausible.
    #[cfg(feature = "prop-census")]
    for (ns_plane, count_plane) in RAMP_H_NS.iter().zip(RAMP_COUNT.iter()) {
        for (ns_row, count_row) in ns_plane.iter().zip(count_plane.iter()) {
            for (ns, count) in ns_row.iter().zip(count_row.iter()) {
                ns.store(0, Ordering::Relaxed);
                count.store(0, Ordering::Relaxed);
            }
        }
    }
    #[cfg(feature = "prop-census")]
    for plane in &RAMP_STEPS {
        for steps in plane {
            steps.store(0, Ordering::Relaxed);
        }
    }
    // Same reason the ramp accumulators clear here: a harness that resets
    // between arms must not read the previous arm's legs blended into its own.
    #[cfg(feature = "prop-census")]
    for class in 0..LEG_CLASSES {
        for counter in [
            LEG_COUNT.get(class),
            LEG_STEPS.get(class),
            LEG_EVALS.get(class),
            LEG_REJECTED.get(class),
            LEG_SPAN_NS.get(class),
        ]
        .into_iter()
        .flatten()
        {
            counter.store(0, Ordering::Relaxed);
        }
        if let Some(histogram) = LEG_STEPS_HIST.get(class) {
            for bucket in histogram {
                bucket.store(0, Ordering::Relaxed);
            }
        }
    }
    #[cfg(feature = "prop-census")]
    for index in 0..ECLIPSE_TRANSACTION_SITES {
        for counter in [
            &ECLIPSE_TRANSACTION_SITE_LEGS,
            &ECLIPSE_TRANSACTION_SITE_STEPS,
            &ECLIPSE_TRANSACTION_SITE_EVALS,
            &ECLIPSE_TRANSACTION_SITE_REJECTED,
            &ECLIPSE_TRANSACTION_SITE_FANTASY_REMOVABLE_STEPS,
            &ECLIPSE_TRANSACTION_SITE_FANTASY_REMOVABLE_EVALS,
        ] {
            atomic_store(counter, index, 0);
        }
    }
    UNDERFLOW_ACCEPTS.store(0, Ordering::Relaxed);
    scalar_clear(&JB_ADAPTER_CALLS);
    scalar_clear(&JB_KERNEL_CALLS);
    scalar_clear(&PROP_RETURNS);
    scalar_clear(&PROP_RETURNS_BAD);
    scalar_clear(&MASS_SENS_COUNT);
    scalar_clear(&MASS_SENS_SUM_NANO);
    *mass_min_sens_lock() = (u64::MAX, 0);
    // The max end needs clearing for the same reason the min end does: it is a
    // running extremum, so a harness that resets between arms and re-measures
    // carries the previous arm's maximum forward into `PROP_MASSSENS`. It was
    // missed when the max was added -- `reset()` cleared only the min.
    *mass_max_sens_lock() = (0, 0);
    scalar_clear(&MASS_SUM_MICROG);
    solved_mass_rows.clear();
    event_support_points.clear();
    for counter in &STAGE_CALLS {
        counter.store(0, Ordering::Relaxed);
    }
    for bucket in &HF_CALLS_HIST {
        bucket.store(0, Ordering::Relaxed);
    }
    #[cfg(feature = "prop-census")]
    {
        for tag in 0..NTAG {
            tag_clear(&SCOPE_ENTRIES, tag);
        }
        captured_arcs.clear();
        *seen_states = None;
        lm_iters.clear();
        lm_passes.clear();
        LM_PASS_SEQ.store(0, Ordering::Relaxed);
        drop(lm_passes);
        drop(lm_iters);
        drop(seen_states);
        drop(captured_arcs);
        CENSUS_INVALID_KIND.store(0, Ordering::Release);
        CENSUS_INVALID.store(false, Ordering::Release);
    }
    drop(event_support_points);
    drop(solved_mass_rows);
    Ok(())
}

pub(crate) const TAG_NAMES: [&str; NTAG] = [
    "other",
    "ukf_sigma_mean_only",
    "mass_miss_hf",
    "release_control",
    "zero_mass_anchor",
    "ukf_sigma_pc_full",
    "rc_skip_probe",
    "rc_lm_entry",
    "rc_zero_dv",
    "rc_fd_jacobian",
    "rc_trial_step",
];

// --- mass-solver stage census (written from dust_estimates_rs) ---

pub(crate) const NSTAGE: usize = 8;
pub const STAGE_VALIDATE_INITIAL: usize = 0;
pub const STAGE_VALIDATE_REPAIR: usize = 1;
pub const STAGE_VALIDATE_REFINE: usize = 2;
pub const STAGE_FULL_BRACKET: usize = 3;
pub const STAGE_FULL_REFINE: usize = 4;
pub const STAGE_ROWS_VALIDATE_ONLY: usize = 5;
pub const STAGE_ROWS_FULL: usize = 6;
pub const STAGE_MEMO_HITS: usize = 7;

pub(crate) const STAGE_NAMES: [&str; NSTAGE] = [
    "validate_initial",
    "validate_repair",
    "validate_refine",
    "full_bracket",
    "full_refine",
    "rows_validate_only",
    "rows_full",
    "memo_hits",
];

pub(crate) static STAGE_CALLS: [AtomicU64; NSTAGE] = [const { AtomicU64::new(0) }; NSTAGE];

/// Histogram of "how many distinct HF miss evaluations did one mass solve take".
pub(crate) const HIST_LEN: usize = 40;
pub(crate) static HF_CALLS_HIST: [AtomicU64; HIST_LEN] = [const { AtomicU64::new(0) }; HIST_LEN];

#[inline]
pub fn bump_stage(stage: usize) {
    atomic_add(&STAGE_CALLS, stage, 1);
}

/// On drop, files "how many HF miss propagations did this one mass solve
/// cost" into the histogram. Covers every early return the solver has.
pub struct MassSolveGuard(u64);

impl Drop for MassSolveGuard {
    fn drop(&mut self) {
        let now = TL_MASS_PROPS.with(Cell::get);
        let calls = usize::try_from(now.saturating_sub(self.0)).unwrap_or(usize::MAX);
        atomic_add(&HF_CALLS_HIST, calls.min(HIST_LEN - 1), 1);
    }
}

#[must_use]
pub fn mass_solve_scope() -> MassSolveGuard {
    MassSolveGuard(TL_MASS_PROPS.with(Cell::get))
}

// --- release-control LM census (written from `two_phase_transfer_rs`) ---
//
// MEASUREMENT SCAFFOLDING. `prop-census`-gated, so with the feature off every
// writer below is an empty inline function and the LM loop is byte-identical
// to what it was.
//
// The tag counters answer "how many propagations did the FD Jacobian cost".
// They cannot answer "how many LM ITERATIONS ran", "was the Jacobian
// well-conditioned", or "is the FD step resolving anything above the arc's own
// noise floor" -- and the propagation count divided by an ASSUMED three per
// iteration is exactly the kind of inference that has been wrong here before.
// These rows are the direct observation instead.

/// Whether the LM census is compiled in.
///
/// A `const`, not a function call, so `if probe::LM_CENSUS { .. }` in
/// `two_phase_transfer_rs` is eliminated entirely when the feature is off. That
/// crate has no `prop-census` feature of its own and does not need one; adding
/// one would mean editing a `Cargo.toml`.
#[cfg(feature = "prop-census")]
pub const LM_CENSUS: bool = true;
/// See the enabled arm.
#[cfg(not(feature = "prop-census"))]
pub const LM_CENSUS: bool = false;

/// One LM iteration of one `lm_solve_bounded` pass.
#[derive(Debug, Clone, Copy)]
pub struct LmIterRow {
    /// Which `lm_solve_bounded` call this iteration belongs to.
    pub pass: u64,
    /// 0-based iteration index within that pass.
    pub iter: u32,
    /// Propagations charged to this iteration (FD columns + trial steps).
    pub props: u64,
    /// `||miss||` at the START of the iteration, km.
    pub miss_km: f64,
    /// Regularised cost at the start of the iteration.
    pub cost: f64,
    /// `||miss(x + eps e_j) - miss(x)||` per FD column, km. This is the SIGNAL
    /// the one-sided difference is built from; compare against the arc's own
    /// reproducibility floor to see whether it is differentiating noise.
    pub fd_dmiss_km: [f64; 3],
    /// Actual ambient coordinate increment `(x[j] + jac_eps) - x[j]`, km/s.
    /// No ball projection occurs; floating representability may make it differ
    /// from `jac_eps`.
    pub fd_step_kms: [f64; 3],
    /// Singular values of the 3x3 miss block of the Jacobian, descending.
    pub fd_svals: [f64; 3],
    /// The 3x3 miss block actually used, row-major, km per km/s. Recorded so
    /// that questions about the Jacobian are answered by comparing matrices
    /// directly rather than by arguing from the singular values.
    pub fd_jac: [f64; 9],
    /// The iterate the Jacobian was built at, km/s.
    pub x_kms: [f64; 3],
    /// How the 3x3 miss Jacobian was actually built.
    pub jacobian_route: LmJacobianRoute,
    /// Trial-step objective evaluations spent in the damping loop (1..=10).
    pub trial_evals: u32,
    /// Whether the damping loop found a descending step.
    pub accepted: bool,
}

/// Source of the 3x3 miss Jacobian recorded for one LM iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmJacobianRoute {
    /// The caller's cheap model produced every usable column.
    Model,
    /// No model was available, so the real objective was differenced.
    RealFdNoModel,
    /// A supplied model declined or produced an unusable column.
    RealFdFallback,
}

impl LmJacobianRoute {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::RealFdNoModel => "real_fd_no_model",
            Self::RealFdFallback => "real_fd_fallback",
        }
    }
}

impl std::fmt::Display for LmJacobianRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One `lm_solve_bounded` call.
#[derive(Debug, Clone, Copy)]
pub struct LmPassRow {
    pub pass: u64,
    /// Propagations charged to the whole pass. Zero identifies the analytic
    /// (MF/equinoctial) passes, which cost no propagation at all.
    pub props: u64,
    pub iters: u32,
    pub nfev: u32,
    pub converged: bool,
    pub bound_kms: f64,
    pub start_miss_km: f64,
    pub best_miss_km: f64,
}

#[cfg(feature = "prop-census")]
static LM_ITERS: Mutex<Vec<LmIterRow>> = Mutex::new(Vec::new());
#[cfg(feature = "prop-census")]
static LM_PASSES: Mutex<Vec<LmPassRow>> = Mutex::new(Vec::new());
#[cfg(feature = "prop-census")]
static LM_PASS_SEQ: AtomicU64 = AtomicU64::new(0);

/// Per-iteration LM rows are an opt-in diagnostic (`ND_LM_CENSUS=1`), matching
/// the other dump stores: production-scale census reads never need them, and at
/// P24 shapes they are the store most likely to hit its bound and latch the
/// whole epoch invalid.
#[cfg(feature = "prop-census")]
fn lm_census_disabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*ENABLED.get_or_init(|| std::env::var("ND_LM_CENSUS").is_ok_and(|value| value == "1"))
}

/// Hard cap for retained LM rows. A census that cannot retain every row is not
/// evidence, so it fails closed before allocating an unbounded log.
// Sized 2026-08-13 for production-scale Phase-A reads: a P24/G1 hybrid cell
// fires ~306k descriptors x ~7 LM iterations. Diagnostic storage only.
#[cfg(feature = "prop-census")]
const MAX_LM_CENSUS_ROWS: usize = 8_388_608;

#[cfg(feature = "prop-census")]
/// Claim a pass id. Ids are unique across threads; they are not ordered.
///
/// # Errors
///
/// Returns [`PropagationCensusError::CounterOverflow`] when the sequence is
/// exhausted.
#[inline]
pub fn lm_next_pass_id() -> CensusResult<u64> {
    LM_PASS_SEQ
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| PropagationCensusError::CounterOverflow)
}

#[cfg(not(feature = "prop-census"))]
/// Claim a pass id in builds without the optional census.
///
/// # Errors
///
/// This implementation cannot fail and always returns the disabled-census id.
pub const fn lm_next_pass_id() -> CensusResult<u64> {
    Ok(0)
}

#[cfg(feature = "prop-census")]
/// Record one LM iteration row, invalidating only the diagnostic epoch if its
/// bounded storage cannot accept the row.
pub fn record_lm_iter(row: LmIterRow) {
    if lm_census_disabled() {
        return;
    }
    let result = census_lock(&LM_ITERS)
        .and_then(|mut rows| try_push_census_row(&mut rows, MAX_LM_CENSUS_ROWS, row));
    if let Err(error) = result {
        invalidate_census(error);
    }
}

#[cfg(not(feature = "prop-census"))]
/// Record one LM iteration row in builds without the optional census.
///
pub const fn record_lm_iter(_row: LmIterRow) {}

#[cfg(feature = "prop-census")]
/// Record one LM pass row, invalidating only the diagnostic epoch if its
/// bounded storage cannot accept the row.
pub fn record_lm_pass(row: LmPassRow) {
    if lm_census_disabled() {
        return;
    }
    let result = census_lock(&LM_PASSES)
        .and_then(|mut rows| try_push_census_row(&mut rows, MAX_LM_CENSUS_ROWS, row));
    if let Err(error) = result {
        invalidate_census(error);
    }
}

#[cfg(not(feature = "prop-census"))]
/// Record one LM pass row in builds without the optional census.
///
pub const fn record_lm_pass(_row: LmPassRow) {}

#[cfg(feature = "prop-census")]
/// Return a copy of retained LM iteration rows.
///
/// # Errors
///
/// Returns a typed error when retained evidence is poisoned or cannot allocate
/// the result copy.
pub fn lm_iter_rows() -> CensusResult<Vec<LmIterRow>> {
    let rows = census_lock(&LM_ITERS)?;
    let mut copy = Vec::new();
    copy.try_reserve_exact(rows.len())
        .map_err(|_| PropagationCensusError::Allocation)?;
    copy.extend(rows.iter().copied());
    drop(rows);
    Ok(copy)
}

#[cfg(not(feature = "prop-census"))]
/// Return retained LM iteration rows in builds without the optional census.
///
/// # Errors
///
/// This implementation cannot fail and returns no retained rows.
pub const fn lm_iter_rows() -> CensusResult<Vec<LmIterRow>> {
    Ok(Vec::new())
}

#[cfg(feature = "prop-census")]
/// Return a copy of retained LM pass rows.
///
/// # Errors
///
/// Returns a typed error when retained evidence is poisoned or cannot allocate
/// the result copy.
pub fn lm_pass_rows() -> CensusResult<Vec<LmPassRow>> {
    let rows = census_lock(&LM_PASSES)?;
    let mut copy = Vec::new();
    copy.try_reserve_exact(rows.len())
        .map_err(|_| PropagationCensusError::Allocation)?;
    copy.extend(rows.iter().copied());
    drop(rows);
    Ok(copy)
}

#[cfg(not(feature = "prop-census"))]
/// Return retained LM pass rows in builds without the optional census.
///
/// # Errors
///
/// This implementation cannot fail and returns no retained rows.
pub const fn lm_pass_rows() -> CensusResult<Vec<LmPassRow>> {
    Ok(Vec::new())
}

// --- exact-duplicate census over propagation initial states ---
//
// GATED, and the gate is the point. Everything below takes a process-wide
// `Mutex` on EVERY propagation, on a path the campaign runs ~8,261 times per
// design-event across parallel rayon workers. It shipped ungated and read by
// nothing: no caller outside this module ever asked for its output. The
// free-threading audit missed it because that audit scoped to per-*evaluation*
// paths and this is per-*propagation*.
//
// Enable with `--features prop-census` when you actually want the census. With
// the feature off the writers below compile to empty inline functions, so the
// call sites in `integrator.rs` need no `cfg` of their own and cannot drift
// out of sync with this module.

#[cfg(feature = "prop-census")]
type StateCensusMap = std::collections::HashMap<[u64; 16], u64>;

/// PROBE (phase-a-measure dedup recount): the science half of the duplicate
/// key. The original 9-word key -- state bits, t0, tf, tag -- omitted every
/// science parameter, so two arcs agreeing on state and times but differing
/// in epoch or ballistics were counted as one duplicate. The 2026-08-05 audit
/// (docs/plans/2026-08-05-hf-hybrid-speedup-audit.md, task #22) named the
/// full-key recount as step one before evaluating duplicate reuse; this widens the key
/// to 16 words so `PROP_DEDUP` reports true exact-arc duplicates.
#[cfg(feature = "prop-census")]
#[derive(Clone, Copy)]
pub(crate) struct ScienceKey {
    pub jd0: f64,
    pub am_ratio: f64,
    pub cd: f64,
    pub cr: f64,
    pub eps: f64,
    pub dt_max: f64,
    pub sph_order: usize,
    pub force_flags: i32,
    pub atm_model: i32,
}

#[cfg(feature = "prop-census")]
impl ScienceKey {
    /// Pack the three small integers into one key word:
    /// `sph_order` high 16, `atm_model` middle 16, `force_flags` low 32.
    fn packed_ints(&self) -> u64 {
        let sph = (usize_to_u64(self.sph_order) & 0xFFFF) << 48;
        let atm = (u64::from(self.atm_model.cast_unsigned()) & 0xFFFF) << 32;
        let flags = u64::from(self.force_flags.cast_unsigned());
        sph | atm | flags
    }
}

#[cfg(feature = "prop-census")]
pub(crate) static SEEN_STATES: Mutex<Option<StateCensusMap>> = Mutex::new(None);

/// A fixed diagnostic ceiling prevents an optional census from becoming an
/// unbounded process-memory sink. Reaching it invalidates the observation.
#[cfg(feature = "prop-census")]
// Sized 2026-08-13 with MAX_LM_CENSUS_ROWS above: rc_trial_step submissions
// scale ~90x from the 4x2 audit corpus at P24/G1.
const MAX_STATE_CENSUS_KEYS: usize = 4_194_304;

/// One retained arc per tag, so this ceiling is a backstop and not a policy:
/// `try_capture_arc` returns early once a line for the tag exists, which alone
/// bounds the vector at `NTAG`. Sized to match so the two bounds cannot
/// disagree -- and so the row-limit branch below stays UNREACHABLE, which is
/// why `capture_arc`'s latch is not documented as a filling story.
#[cfg(feature = "prop-census")]
const MAX_CAPTURED_ARCS: usize = NTAG;

#[cfg(feature = "prop-census")]
#[derive(Clone, Copy)]
pub(crate) struct CensusArc<'a> {
    pub tag: usize,
    pub init_equinoc: &'a [f64; 6],
    pub jd0: f64,
    pub start_time_s: f64,
    pub final_time_s: f64,
    pub eps: f64,
    pub sph_order: usize,
    pub force_flags: i32,
    pub atm_model: i32,
    pub am_ratio: f64,
    pub cd: f64,
    pub cr: f64,
    pub dt_max: f64,
}

/// Record one propagation's exact initial state + arc, so "how many of these
/// N propagations are bit-identical repeats" can be answered by counting
/// rather than argued from field cardinalities.
///
/// INFALLIBLE BY DESIGN. `SEEN_STATES` is bounded, so a long enough process
/// fills it; this used to return `Err` at that point, which the propagation
/// entry turned into `FinalPropagationFailure::Census` and callers absorbed as
/// an infeasible design. Worse, the refusal was selective -- a repeat key takes
/// the `get_mut` branch below and still succeeded -- so the surviving arcs were
/// exactly the un-novel ones. A full map now latches `CENSUS_INVALID` and the
/// propagation proceeds.
#[cfg(feature = "prop-census")]
pub(crate) fn record_state(
    state: &[f64; 6],
    start_time_s: f64,
    final_time_s: f64,
    science: &ScienceKey,
) {
    record_state_within(
        state,
        start_time_s,
        final_time_s,
        science,
        MAX_STATE_CENSUS_KEYS,
    );
}

/// `record_state` with the key ceiling supplied, so a test can saturate it in
/// two calls instead of 262,144.
#[cfg(feature = "prop-census")]
fn record_state_within(
    state: &[f64; 6],
    start_time_s: f64,
    final_time_s: f64,
    science: &ScienceKey,
    limit: usize,
) {
    let recorded = (|| -> CensusResult<()> {
        let mut key = [0u64; 16];
        for (key_component, state_component) in key.iter_mut().take(6).zip(state) {
            *key_component = state_component.to_bits();
        }
        if let Some(time) = key.get_mut(6) {
            *time = start_time_s.to_bits();
        }
        if let Some(time) = key.get_mut(7) {
            *time = final_time_s.to_bits();
        }
        if let Some(tag) = key.get_mut(8) {
            *tag = u64::try_from(current_tag())
                .map_err(|_| PropagationCensusError::CounterOverflow)?;
        }
        // Science words 9..16. Index 8 stays the tag: `state_census_by_tag`
        // reads it positionally and must not move.
        for (slot, bits) in key.iter_mut().skip(9).zip([
            science.jd0.to_bits(),
            science.am_ratio.to_bits(),
            science.cd.to_bits(),
            science.cr.to_bits(),
            science.eps.to_bits(),
            science.dt_max.to_bits(),
            science.packed_ints(),
        ]) {
            *slot = bits;
        }
        let mut states = census_lock(&SEEN_STATES)?;
        let result = insert_state_census_key(&mut states, key, limit);
        drop(states);
        result
    })();
    if let Err(error) = recorded {
        invalidate_census(error);
    }
}

#[cfg(feature = "prop-census")]
fn insert_state_census_key(
    states: &mut Option<StateCensusMap>,
    key: [u64; 16],
    limit: usize,
) -> CensusResult<()> {
    if let Some(map) = states.as_mut() {
        if let Some(count) = map.get_mut(&key) {
            *count = count
                .checked_add(1)
                .ok_or(PropagationCensusError::CounterOverflow)?;
            return Ok(());
        }
        let next_len = map
            .len()
            .checked_add(1)
            .ok_or(PropagationCensusError::CounterOverflow)?;
        if next_len > limit {
            return Err(PropagationCensusError::Allocation);
        }
        map.try_reserve(1)
            .map_err(|_| PropagationCensusError::Allocation)?;
        map.insert(key, 1);
        return Ok(());
    }

    if limit == 0 {
        return Err(PropagationCensusError::Allocation);
    }
    let mut map = StateCensusMap::new();
    map.try_reserve(1)
        .map_err(|_| PropagationCensusError::Allocation)?;
    map.insert(key, 1);
    *states = Some(map);
    Ok(())
}

/// Per-tag `(submitted, unique)` propagation counts.
#[cfg(feature = "prop-census")]
pub(crate) fn state_census_by_tag() -> CensusResult<[(u64, u64); NTAG]> {
    let guard = census_lock(&SEEN_STATES)?;
    let mut out = [(0u64, 0u64); NTAG];
    if let Some(map) = guard.as_ref() {
        for (key, count) in map {
            let tag_bits = key
                .get(8)
                .copied()
                .ok_or(PropagationCensusError::CounterOverflow)?;
            let tag =
                usize::try_from(tag_bits).map_err(|_| PropagationCensusError::CounterOverflow)?;
            let tag = tag.min(
                NTAG.checked_sub(1)
                    .ok_or(PropagationCensusError::CounterOverflow)?,
            );
            let totals = out
                .get_mut(tag)
                .ok_or(PropagationCensusError::CounterOverflow)?;
            totals.0 = totals
                .0
                .checked_add(*count)
                .ok_or(PropagationCensusError::CounterOverflow)?;
            totals.1 = totals
                .1
                .checked_add(1)
                .ok_or(PropagationCensusError::CounterOverflow)?;
        }
    }
    drop(guard);
    Ok(out)
}

/// First propagation captured per tag, verbatim, so a standalone harness can
/// reproduce a real campaign arc instead of constructing one.
///
/// Exists because pairing the campaign dust ballistic coefficient
/// (`am_ratio` 1.948, ~200x a satellite's) with a satellite-like circular LEO
/// state deorbits the object inside the arc: 5,152 of 5,152 propagations
/// diverged. The force config and the state have to come from the same place.
#[cfg(feature = "prop-census")]
pub(crate) static CAPTURED_ARCS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Infallible for the same reason as `record_state`: this is a diagnostic
/// store on the propagation entry path, and a store that cannot take a row
/// must void the observation, not the flight.
///
/// UNLIKE `record_state`, the reason is NOT that the store fills.
/// `MAX_CAPTURED_ARCS` is unreachable by construction (see its comment), so
/// what the latch below actually catches is a poisoned `CAPTURED_ARCS` mutex —
/// i.e. some other thread panicked mid-capture — or a failed `try_reserve` for
/// the row. Both are reachable and neither is tested; the census-side
/// consequence is pinned only through `record_state`'s saturation test.
#[cfg(feature = "prop-census")]
pub(crate) fn capture_arc(arc: CensusArc<'_>) {
    if let Err(error) = try_capture_arc(arc) {
        invalidate_census(error);
    }
}

#[cfg(feature = "prop-census")]
fn try_capture_arc(arc: CensusArc<'_>) -> CensusResult<()> {
    const MAX_CAPTURED_ARC_BYTES: usize = 512;

    let tag_name = TAG_NAMES
        .get(arc.tag)
        .copied()
        .ok_or(PropagationCensusError::CounterOverflow)?;
    let mut captured = census_lock(&CAPTURED_ARCS)?;
    if captured.iter().any(|line| line.starts_with(tag_name)) {
        return Ok(());
    }
    let result = try_push_census_row(&mut captured, MAX_CAPTURED_ARCS, {
        let mut line = String::new();
        line.try_reserve_exact(MAX_CAPTURED_ARC_BYTES)
            .map_err(|_| PropagationCensusError::Allocation)?;
        let [semi_major_axis, equinoctial_h, equinoctial_k, equinoctial_p, equinoctial_q, mean_longitude] =
            *arc.init_equinoc;
        write!(
            line,
            "{tag_name} init_equinoc=[{semi_major_axis:.17e}, {equinoctial_h:.17e}, {equinoctial_k:.17e}, {equinoctial_p:.17e}, {equinoctial_q:.17e}, {mean_longitude:.17e}] \
                 jd0={:.17e} t0_s={:.17e} tf_s={:.17e} eps={:e} sph_order={} force_flags={:#06x} \
                 atm_model={} am_ratio={:.17e} cd={:.17e} cr={:.17e} dt_max={:.17e}",
            arc.jd0,
            arc.start_time_s,
            arc.final_time_s,
            arc.eps,
            arc.sph_order,
            arc.force_flags,
            arc.atm_model,
            arc.am_ratio,
            arc.cd,
            arc.cr,
            arc.dt_max,
        )
        .map_err(|_| PropagationCensusError::Allocation)?;
        line
    });
    drop(captured);
    result
}

/// PROBE (phase-a-measure, lane A prefix-arc census): groups the computed-arc
/// census by the FULL 16-word key MINUS `tf` (word 7), so "how many arcs share
/// everything except their end time" can be counted rather than argued.
///
/// WHAT IT PRICES. A trajectory cached at its accepted steps could serve any
/// shorter member of such a group by dense-output interpolation, and any longer
/// member could resume from the shorter one's endpoint (Encke-form state is
/// checkpointable). Order-obliviously, the ceiling for a group is "integrate
/// the longest span once, serve everything else": savable span =
/// (count-weighted group span) - (longest single span).
///
/// EVAL CONVERSION IS A PROXY. The census does not carry per-arc eval counts,
/// so savable SPAN is converted to evals per tag as
/// `tag_rhs_evals * savable_span / tag_span`. Within a group the members share
/// the same state, epoch and science, so the shorter arc's step sequence is
/// (near-)a prefix of the longer's and span-proportionality is tight there;
/// ACROSS groups of one tag it assumes a common eval density, which is the
/// stated error bar on `est_evals`.
///
/// CURRENT SEMANTICS. Every computed arc enters this map, including exact
/// duplicates. These lines therefore price the full prefix-reuse ceiling.
#[cfg(feature = "prop-census")]
fn write_prefix_census(out: &mut CensusReport) -> CensusResult<()> {
    /// Group-key width: the 16 census words minus the `tf` word.
    const GROUP_WORDS: usize = 15;
    /// `tf` position in the census key.
    const TF_WORD: usize = 7;
    /// Tag position in the GROUP key (census word 8, shifted down by the
    /// removed `tf` word).
    const GROUP_TAG_WORD: usize = 7;
    /// `t0` position in the group key (census word 6, unshifted).
    const GROUP_T0_WORD: usize = 6;

    #[derive(Clone, Copy, Default)]
    struct TagAccumulator {
        arcs: u64,
        groups: u64,
        multi_groups: u64,
        multi_arcs: u64,
        span_s: f64,
        multi_span_s: f64,
        longest_span_s: f64,
    }

    let mut groups: std::collections::HashMap<[u64; GROUP_WORDS], Vec<(u64, u64)>> =
        std::collections::HashMap::new();
    {
        let guard = census_lock(&SEEN_STATES)?;
        let Some(map) = guard.as_ref() else {
            drop(guard);
            return Ok(());
        };
        for (key, count) in map {
            let mut group_key = [0u64; GROUP_WORDS];
            let mut slot = 0usize;
            for (index, word) in key.iter().enumerate() {
                if index == TF_WORD {
                    continue;
                }
                *group_key
                    .get_mut(slot)
                    .ok_or(PropagationCensusError::CounterOverflow)? = *word;
                slot = slot
                    .checked_add(1)
                    .ok_or(PropagationCensusError::CounterOverflow)?;
            }
            let tf_bits = key
                .get(TF_WORD)
                .copied()
                .ok_or(PropagationCensusError::CounterOverflow)?;
            groups.entry(group_key).or_default().push((tf_bits, *count));
        }
        drop(guard);
    }

    // Group-size histogram buckets: 1, 2, 3, 4, 5-8, 9-16, 17-32, 33+.
    let mut size_hist = [0u64; 8];
    let mut per_tag = [TagAccumulator::default(); NTAG];
    for (group_key, members) in &groups {
        let t0_s = f64::from_bits(
            group_key
                .get(GROUP_T0_WORD)
                .copied()
                .ok_or(PropagationCensusError::CounterOverflow)?,
        );
        let tag_bits = group_key
            .get(GROUP_TAG_WORD)
            .copied()
            .ok_or(PropagationCensusError::CounterOverflow)?;
        let tag = usize::try_from(tag_bits)
            .map_err(|_| PropagationCensusError::CounterOverflow)?
            .min(
                NTAG.checked_sub(1)
                    .ok_or(PropagationCensusError::CounterOverflow)?,
            );
        let mut group_span_s = 0.0_f64;
        let mut longest_s = 0.0_f64;
        let mut group_arcs = 0_u64;
        for (tf_bits, count) in members {
            let duration_s = f64::from_bits(*tf_bits) - t0_s;
            group_span_s += u64_to_f64(*count) * duration_s;
            longest_s = longest_s.max(duration_s);
            group_arcs = group_arcs
                .checked_add(*count)
                .ok_or(PropagationCensusError::CounterOverflow)?;
        }
        let bucket = match members.len() {
            0..=4 => members.len().saturating_sub(1),
            5..=8 => 4,
            9..=16 => 5,
            17..=32 => 6,
            _ => 7,
        };
        if let Some(slot) = size_hist.get_mut(bucket) {
            *slot = slot
                .checked_add(1)
                .ok_or(PropagationCensusError::CounterOverflow)?;
        }
        let accumulator = per_tag
            .get_mut(tag)
            .ok_or(PropagationCensusError::CounterOverflow)?;
        accumulator.arcs = accumulator
            .arcs
            .checked_add(group_arcs)
            .ok_or(PropagationCensusError::CounterOverflow)?;
        accumulator.groups = accumulator
            .groups
            .checked_add(1)
            .ok_or(PropagationCensusError::CounterOverflow)?;
        accumulator.span_s += group_span_s;
        if members.len() >= 2 {
            accumulator.multi_groups = accumulator
                .multi_groups
                .checked_add(1)
                .ok_or(PropagationCensusError::CounterOverflow)?;
            accumulator.multi_arcs = accumulator
                .multi_arcs
                .checked_add(group_arcs)
                .ok_or(PropagationCensusError::CounterOverflow)?;
            accumulator.multi_span_s += group_span_s;
            accumulator.longest_span_s += longest_s;
        }
    }

    let census = snapshot();
    let total_rhs = census.iter().try_fold(0_u64, |total, row| {
        total
            .checked_add(row.rhs_evals)
            .ok_or(PropagationCensusError::CounterOverflow)
    })?;
    append_format(
        out,
        format_args!(
            "PROP_PREFIX tag,arcs,groups,multi_groups,multi_arcs,span_s,multi_span_s,\
             longest_span_s,savable_span_s,savable_frac,est_savable_evals\n"
        ),
    )?;
    let mut total_est_evals = 0.0_f64;
    let mut total_savable_span = 0.0_f64;
    for (tag, accumulator) in per_tag.iter().enumerate() {
        if accumulator.groups == 0 {
            continue;
        }
        let savable_span_s = accumulator.multi_span_s - accumulator.longest_span_s;
        let savable_frac = if accumulator.span_s > 0.0 {
            savable_span_s / accumulator.span_s
        } else {
            0.0
        };
        let tag_evals = census.get(tag).map_or(0, |row| row.rhs_evals);
        let est_evals = u64_to_f64(tag_evals) * savable_frac;
        total_est_evals += est_evals;
        total_savable_span += savable_span_s;
        let tag_name = TAG_NAMES
            .get(tag)
            .copied()
            .ok_or(PropagationCensusError::CounterOverflow)?;
        append_format(
            out,
            format_args!(
                "PROP_PREFIX {tag_name},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.5},{:.0}\n",
                accumulator.arcs,
                accumulator.groups,
                accumulator.multi_groups,
                accumulator.multi_arcs,
                accumulator.span_s,
                accumulator.multi_span_s,
                accumulator.longest_span_s,
                savable_span_s,
                savable_frac,
                est_evals,
            ),
        )?;
    }
    let total_pct = if total_rhs > 0 {
        100.0 * total_est_evals / u64_to_f64(total_rhs)
    } else {
        0.0
    };
    append_format(
        out,
        format_args!(
            "PROP_PREFIX TOTAL,savable_span_s,{total_savable_span:.1},\
             est_savable_evals,{total_est_evals:.0},pct_of_total_rhs,{total_pct:.3}\n"
        ),
    )?;
    for (bucket, label) in ["1", "2", "3", "4", "5-8", "9-16", "17-32", "33+"]
        .iter()
        .enumerate()
    {
        let count = size_hist.get(bucket).copied().unwrap_or(0);
        if count > 0 {
            append_format(out, format_args!("PROP_PREFIX_SIZE {label},{count}\n"))?;
        }
    }

    write_prefix_xtag(out, &groups, total_savable_span)?;
    write_quantized_dup_census(out)?;
    Ok(())
}

/// PROBE (lane B, near-miss census): re-runs the exact-duplicate count with the
/// six STATE words quantized -- low `mask_bits` mantissa bits zeroed -- and
/// everything else (t0, tf, tag, science) still exact. The collapse curve over
/// mask widths says whether the 1.49M unique arcs have near-miss structure: how
/// many computed arcs are within a relative half-ulp band of another arc that
/// differs ONLY in state, i.e. arcs a sub-tolerance-accuracy consumer could
/// have been served. Serving one arc another's result IS bit-moving; these
/// lines only price the ceiling, they do not argue the physics.
///
/// Reading: `mask_bits` of 16 zeroes ~2^-36 relative (~0.1 um on a 7000 km
/// axis -- below the eps=1e-8 km tolerance), 26 is ~3e-8 relative (~0.2 mm),
/// 33 is ~4e-6 relative. Only `savable` at masks at or below the tolerance
/// band is even arguable.
#[cfg(feature = "prop-census")]
fn write_quantized_dup_census(out: &mut CensusReport) -> CensusResult<()> {
    let census = snapshot();
    let total_rhs = census.iter().try_fold(0_u64, |total, row| {
        total
            .checked_add(row.rhs_evals)
            .ok_or(PropagationCensusError::CounterOverflow)
    })?;
    for mask_bits in [8u32, 16, 26, 33] {
        let mask = !(1u64
            .checked_shl(mask_bits)
            .ok_or(PropagationCensusError::CounterOverflow)?
            .checked_sub(1)
            .ok_or(PropagationCensusError::CounterOverflow)?);
        let mut quantized: std::collections::HashMap<[u64; 16], (u64, u64, f64)> =
            std::collections::HashMap::new();
        {
            let guard = census_lock(&SEEN_STATES)?;
            let Some(map) = guard.as_ref() else {
                drop(guard);
                return Ok(());
            };
            // Value: (distinct member keys, count-weighted arcs, span sum).
            for (key, count) in map {
                let mut quantized_key = *key;
                for (index, word) in quantized_key.iter_mut().enumerate() {
                    if index < 6 {
                        *word &= mask;
                    }
                }
                let start_s = f64::from_bits(key.get(6).copied().unwrap_or(0));
                let end_s = f64::from_bits(key.get(7).copied().unwrap_or(0));
                let span_s = u64_to_f64(*count) * (end_s - start_s);
                let entry = quantized.entry(quantized_key).or_insert((0, 0, 0.0));
                entry.0 = entry
                    .0
                    .checked_add(1)
                    .ok_or(PropagationCensusError::CounterOverflow)?;
                entry.1 = entry
                    .1
                    .checked_add(*count)
                    .ok_or(PropagationCensusError::CounterOverflow)?;
                entry.2 += span_s;
            }
            drop(guard);
        }
        let mut multi_groups = 0u64;
        let mut savable_arcs = 0u64;
        let mut total_span = 0.0_f64;
        let mut savable_span = 0.0_f64;
        for (distinct, arcs, span_s) in quantized.values() {
            total_span += *span_s;
            if *distinct >= 2 {
                multi_groups = multi_groups
                    .checked_add(1)
                    .ok_or(PropagationCensusError::CounterOverflow)?;
                savable_arcs = savable_arcs
                    .checked_add(arcs.saturating_sub(1))
                    .ok_or(PropagationCensusError::CounterOverflow)?;
                // Members of one quantized group share tf and t0, so span is
                // uniform across them: all but one member's share is savable.
                savable_span += *span_s * (1.0 - 1.0 / u64_to_f64((*arcs).max(1)));
            }
        }
        let est_pct = if total_span > 0.0 && total_rhs > 0 {
            100.0 * (savable_span / total_span)
        } else {
            0.0
        };
        append_format(
            out,
            format_args!(
                "PROP_PREFIX_Q mask_bits,{mask_bits},groups,{},multi_groups,{multi_groups},\
                 savable_arcs,{savable_arcs},savable_span_s,{savable_span:.1},\
                 est_pct_of_span,{est_pct:.3}\n",
                quantized.len(),
            ),
        )?;
    }
    Ok(())
}

/// Cross-tag half of [`write_prefix_census`]: regroup IGNORING the tag word
/// too. The extra savable span over the per-tag sum measures what any proposed
/// cross-tag prefix reuse would additionally reach.
#[cfg(feature = "prop-census")]
fn write_prefix_xtag(
    out: &mut CensusReport,
    groups: &std::collections::HashMap<[u64; 15], Vec<(u64, u64)>>,
    total_savable_span: f64,
) -> CensusResult<()> {
    /// Tag position in the per-tag group key; `t0` position in both keys.
    const GROUP_TAG_WORD: usize = 7;
    /// `t0` position in the group key (census word 6, unshifted).
    const GROUP_T0_WORD: usize = 6;
    let mut xtag_groups: std::collections::HashMap<[u64; 14], Vec<(u64, u64)>> =
        std::collections::HashMap::new();
    for (group_key, members) in groups {
        let mut xtag_key = [0u64; 14];
        let mut slot = 0usize;
        for (index, word) in group_key.iter().enumerate() {
            if index == GROUP_TAG_WORD {
                continue;
            }
            *xtag_key
                .get_mut(slot)
                .ok_or(PropagationCensusError::CounterOverflow)? = *word;
            slot = slot
                .checked_add(1)
                .ok_or(PropagationCensusError::CounterOverflow)?;
        }
        xtag_groups
            .entry(xtag_key)
            .or_default()
            .extend(members.iter().copied());
    }
    let mut xtag_savable_span = 0.0_f64;
    for (xtag_key, members) in &xtag_groups {
        if members.len() < 2 {
            continue;
        }
        let t0_s = f64::from_bits(
            xtag_key
                .get(GROUP_T0_WORD)
                .copied()
                .ok_or(PropagationCensusError::CounterOverflow)?,
        );
        let mut group_span_s = 0.0_f64;
        let mut longest_s = 0.0_f64;
        for (tf_bits, count) in members {
            let duration_s = f64::from_bits(*tf_bits) - t0_s;
            group_span_s += u64_to_f64(*count) * duration_s;
            longest_s = longest_s.max(duration_s);
        }
        xtag_savable_span += group_span_s - longest_s;
    }
    append_format(
        out,
        format_args!(
            "PROP_PREFIX_XTAG savable_span_s_ignoring_tag,{xtag_savable_span:.1},\
             extra_vs_per_tag_s,{:.1}\n",
            xtag_savable_span - total_savable_span
        ),
    )?;
    Ok(())
}

/// Report text has a fixed diagnostic ceiling. A report beyond this size is
/// incomplete evidence, not a reason to grow unboundedly or print a prefix.
const MAX_CENSUS_REPORT_BYTES: usize = 64 * 1024;

struct CensusReport {
    text: String,
}

impl CensusReport {
    fn new() -> CensusResult<Self> {
        let mut text = String::new();
        text.try_reserve_exact(MAX_CENSUS_REPORT_BYTES)
            .map_err(|_| PropagationCensusError::Allocation)?;
        Ok(Self { text })
    }

    #[cfg(test)]
    fn as_str(&self) -> &str {
        &self.text
    }

    fn into_inner(self) -> String {
        self.text
    }
}

impl std::fmt::Write for CensusReport {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        let next_len = self
            .text
            .len()
            .checked_add(value.len())
            .ok_or(std::fmt::Error)?;
        if next_len > MAX_CENSUS_REPORT_BYTES {
            return Err(std::fmt::Error);
        }
        self.text.push_str(value);
        Ok(())
    }
}

fn append_format(out: &mut CensusReport, args: std::fmt::Arguments<'_>) -> CensusResult<()> {
    out.write_fmt(args)
        .map_err(|_| PropagationCensusError::Allocation)
}

fn write_census_table(
    out: &mut CensusReport,
    census: &[TagCensus; NTAG],
    total_rhs: u64,
) -> CensusResult<()> {
    append_format(
        out,
        format_args!(
            "PROP_CENSUS tag,rhs_evals,pct_rhs,propagations,segments,steps,span_s,\
             evals_per_prop,steps_per_prop,evals_per_step,mean_span_s,mean_step_s,sat_frac,\
             rejected,reject_frac,evals_per_attempt\n"
        ),
    )?;
    for (tag, census_row) in census.iter().copied().enumerate() {
        let c = census_row;
        if c.rhs_evals == 0 && c.propagations == 0 {
            continue;
        }
        let pct = if total_rhs > 0 {
            100.0 * u64_to_f64(c.rhs_evals) / u64_to_f64(total_rhs)
        } else {
            0.0
        };
        let props = u64_to_f64(c.propagations.max(1));
        let steps = u64_to_f64(c.steps.max(1));
        let attempts = u64_to_f64(
            c.steps
                .checked_add(c.rejected)
                .ok_or(PropagationCensusError::CounterOverflow)?
                .max(1),
        );
        let tag_name = TAG_NAMES
            .get(tag)
            .copied()
            .ok_or(PropagationCensusError::CounterOverflow)?;
        append_format(
            out,
            format_args!(
                "PROP_CENSUS {},{},{:.2},{},{},{},{:.1},{:.1},{:.1},{:.2},{:.1},{:.1},{:.3},\
             {},{:.4},{:.2}\n",
                tag_name,
                c.rhs_evals,
                pct,
                c.propagations,
                c.segments,
                c.steps,
                c.span_s,
                u64_to_f64(c.rhs_evals) / props,
                u64_to_f64(c.steps) / props,
                u64_to_f64(c.rhs_evals) / steps,
                c.span_s / props,
                c.span_s / steps,
                u64_to_f64(c.saturated) / steps,
                c.rejected,
                u64_to_f64(c.rejected) / attempts,
                u64_to_f64(c.rhs_evals) / attempts,
            ),
        )?;
    }
    append_format(
        out,
        format_args!("PROP_CENSUS TOTAL_RHS_EVALS,{total_rhs}\n"),
    )?;
    for (name, counter) in STAGE_NAMES.iter().zip(STAGE_CALLS.iter()) {
        let n = counter.load(Ordering::Relaxed);
        if n > 0 {
            append_format(out, format_args!("PROP_STAGE {name},{n}\n"))?;
        }
    }
    for (bucket, counter) in HF_CALLS_HIST.iter().enumerate() {
        let n = counter.load(Ordering::Relaxed);
        if n > 0 {
            append_format(out, format_args!("PROP_HFHIST {bucket},{n}\n"))?;
        }
    }
    Ok(())
}

/// Leg-class rows: the clamped/unclamped split and the steps-per-entry shape.
///
/// Separate from `write_runtime_summary` because these rows answer a different
/// question from the ramp rows they sit next to. The ramp asks what a restart
/// COSTS; this asks how the arc's work is DISTRIBUTED over solver entries, and
/// fusing them under one condition is how an instrument goes quietly dark.
#[cfg(feature = "prop-census")]
fn write_leg_census(out: &mut CensusReport) -> CensusResult<()> {
    append_format(
        out,
        format_args!(
            "PROP_LEGCLASS class,legs,steps,evals,rejected,span_s,steps_per_leg,evals_per_leg,mean_h_s\n"
        ),
    )?;
    for (class, name) in LEG_CLASS_NAMES.iter().enumerate() {
        let (Some(count), Some(steps), Some(evals), Some(rejected), Some(span)) = (
            LEG_COUNT.get(class),
            LEG_STEPS.get(class),
            LEG_EVALS.get(class),
            LEG_REJECTED.get(class),
            LEG_SPAN_NS.get(class),
        ) else {
            continue;
        };
        let legs = count.load(Ordering::Relaxed);
        if legs == 0 {
            continue;
        }
        let legs_f = u64_to_f64(legs);
        let steps = steps.load(Ordering::Relaxed);
        let steps_f = u64_to_f64(steps);
        let span_s = u64_to_f64(span.load(Ordering::Relaxed)) / 1.0e9;
        let evals = evals.load(Ordering::Relaxed);
        let rejected = rejected.load(Ordering::Relaxed);
        // Mean step over the class, not over legs: a mean-of-means across
        // entries holding different step counts is not this mean.
        let mean_h_s = if steps == 0 { 0.0 } else { span_s / steps_f };
        append_format(
            out,
            format_args!(
                "PROP_LEGCLASS {name},{legs},{steps},{evals},{rejected},{span_s:.3},{:.3},{:.3},{mean_h_s:.4}\n",
                steps_f / legs_f,
                u64_to_f64(evals) / legs_f,
            ),
        )?;
    }
    append_format(out, format_args!("PROP_LEGHIST class,steps,legs\n"))?;
    for (class, histogram) in LEG_STEPS_HIST.iter().enumerate() {
        let name = LEG_CLASS_NAMES.get(class).copied().unwrap_or("?");
        for (bucket, counter) in histogram.iter().enumerate() {
            let legs = counter.load(Ordering::Relaxed);
            if legs == 0 {
                continue;
            }
            // The last bucket saturates, so it is labelled as a bound rather
            // than as a count. A reader who takes it for an exact value
            // under-reads the tail, which is the half of this distribution the
            // multistep question turns on.
            let label = if bucket == LEG_HIST_LEN.saturating_sub(1) {
                format!("{bucket}+")
            } else {
                bucket.to_string()
            };
            append_format(out, format_args!("PROP_LEGHIST {name},{label},{legs}\n"))?;
        }
    }
    Ok(())
}

#[cfg(feature = "prop-census")]
fn write_eclipse_transaction_site_census(out: &mut CensusReport) -> CensusResult<()> {
    append_format(
        out,
        format_args!(
            "PROP_ECLIPSE_SITE site,legs,steps,evals,rejected,fantasy_removable_steps,fantasy_removable_evals\n"
        ),
    )?;
    let rows = eclipse_transaction_site_snapshot();
    for (name, row) in ECLIPSE_TRANSACTION_SITE_NAMES.iter().zip(rows) {
        if row.legs == 0 {
            continue;
        }
        append_format(
            out,
            format_args!(
                "PROP_ECLIPSE_SITE {name},{},{},{},{},{},{}\n",
                row.legs,
                row.steps,
                row.evals,
                row.rejected,
                row.fantasy_removable_steps,
                row.fantasy_removable_evals,
            ),
        )?;
    }
    Ok(())
}

fn write_runtime_summary(out: &mut CensusReport) -> CensusResult<()> {
    #[cfg(feature = "prop-census")]
    write_leg_census(out)?;
    #[cfg(feature = "prop-census")]
    write_eclipse_transaction_site_census(out)?;
    #[cfg(not(feature = "prop-census"))]
    append_format(
        out,
        format_args!("PROP_ECLIPSE_SITE not-compiled-in (build with --features prop-census)\n"),
    )?;
    // Deliberately OUTSIDE the `min_accepted_h_s()` block below. It was inside
    // at first, which silently made the ramp census conditional on an unrelated
    // diagnostic having a value -- a run that accepted no step suppressed it,
    // and so did anything that cleared `MIN_ACCEPTED_H_NS` without clearing
    // these. Two instruments, two conditions.
    //
    // One row per (population, slot). The last slot of each population is the
    // SUSTAINED tail -- what the controller settles at once the restart ramp is
    // over -- and the rows before it are the ramp. The ratio of the two is the
    // whole measurement, printed unaggregated because "how many opening steps
    // sit below equilibrium" is a shape question a single mean would hide.
    #[cfg(feature = "prop-census")]
    {
        append_format(
            out,
            format_args!("PROP_RAMP boundary,population,slot,count,mean_h_s\n"),
        )?;
        for (boundary, ((ns_plane, count_plane), steps_plane)) in RAMP_H_NS
            .iter()
            .zip(RAMP_COUNT.iter())
            .zip(RAMP_STEPS.iter())
            .enumerate()
        {
            let boundary_name = RAMP_BOUNDARY_NAMES.get(boundary).copied().unwrap_or("?");
            for (population, ((ns_row, count_row), steps)) in ns_plane
                .iter()
                .zip(count_plane.iter())
                .zip(steps_plane.iter())
                .enumerate()
            {
                let name = if population == 0 { "short" } else { "long" };
                for (slot, (ns, count)) in ns_row.iter().zip(count_row.iter()).enumerate() {
                    let samples = count.load(Ordering::Relaxed);
                    if samples == 0 {
                        continue;
                    }
                    let label = if slot == RAMP_SLOTS.saturating_sub(1) {
                        "tail".to_owned()
                    } else {
                        slot.to_string()
                    };
                    let mean_s =
                        u64_to_f64(ns.load(Ordering::Relaxed)) / u64_to_f64(samples) / 1.0e9;
                    append_format(
                        out,
                        format_args!(
                            "PROP_RAMP {boundary_name},{name},{label},{samples},{mean_s:.9}\n"
                        ),
                    )?;
                }
                let accepted = steps.load(Ordering::Relaxed);
                if accepted != 0 {
                    append_format(
                        out,
                        format_args!("PROP_RAMP {boundary_name},{name},steps,{accepted},0.0\n"),
                    )?;
                }
            }
        }
    }
    if let Some(min_h) = min_accepted_h_s() {
        append_format(
            out,
            format_args!(
                "PROP_MINH min_accepted_h_s,{:.9},cache_cluster_steps,{},untruncated,{}\n",
                min_h,
                scalar_sum(&CACHE_CLUSTER_STEPS),
                scalar_sum(&CACHE_CLUSTER_STEPS_UNTRUNCATED)
            ),
        )?;
        let hits = (0..NTAG).try_fold(0_u64, |total, tag| {
            total
                .checked_add(tag_sum(&BASELINE_HIT, tag))
                .ok_or(PropagationCensusError::CounterOverflow)
        })?;
        let misses = (0..NTAG).try_fold(0_u64, |total, tag| {
            total
                .checked_add(tag_sum(&BASELINE_MISS, tag))
                .ok_or(PropagationCensusError::CounterOverflow)
        })?;
        let baseline_total = hits
            .checked_add(misses)
            .ok_or(PropagationCensusError::CounterOverflow)?;
        if baseline_total > 0 {
            append_format(
                out,
                format_args!(
                    "PROP_BASELINE hits,{hits},misses,{misses},hit_rate,{:.4}\n",
                    u64_to_f64(hits) / u64_to_f64(baseline_total)
                ),
            )?;
        }
        // A SEPARATE line, not more fields on `PROP_BASELINE`: the stage table
        // is a different cache with a different key population, and the note on
        // `BASELINE_STAGE_HIT` is explicit that these must not wear the
        // hit/miss pair's label. `prefill_packs` is four-lane solves issued,
        // so `4 * prefill_packs` is what the table cost and `stage_hits` is
        // what it returned.
        let stage_hits = (0..NTAG).try_fold(0_u64, |total, tag| {
            total
                .checked_add(tag_sum(&BASELINE_STAGE_HIT, tag))
                .ok_or(PropagationCensusError::CounterOverflow)
        })?;
        let prefill_packs = (0..NTAG).try_fold(0_u64, |total, tag| {
            total
                .checked_add(tag_sum(&BASELINE_PREFILL, tag))
                .ok_or(PropagationCensusError::CounterOverflow)
        })?;
        if stage_hits > 0 || prefill_packs > 0 {
            append_format(
                out,
                format_args!(
                    "PROP_BASELINE_STAGE stage_hits,{stage_hits},prefill_packs,{prefill_packs}\n"
                ),
            )?;
        }
        // The `BaselineCalculator` half, on its own line for the same reason:
        // a different key, a different consult rate and a different hit rate.
        // It was write-only until R29 -- both counters reached `TagCensus` and
        // nothing printed them -- which left the note on `BASELINE_HIT`
        // claiming this half was "no longer uncounted" while no report carried
        // it. Emitting it is what makes that note true.
        let calc_hit = (0..NTAG).try_fold(0_u64, |total, tag| {
            total
                .checked_add(tag_sum(&BASELINE_CALC_HIT, tag))
                .ok_or(PropagationCensusError::CounterOverflow)
        })?;
        let calc_miss = (0..NTAG).try_fold(0_u64, |total, tag| {
            total
                .checked_add(tag_sum(&BASELINE_CALC_MISS, tag))
                .ok_or(PropagationCensusError::CounterOverflow)
        })?;
        if calc_hit > 0 || calc_miss > 0 {
            append_format(
                out,
                format_args!("PROP_BASELINE_CALC hit,{calc_hit},miss,{calc_miss}\n"),
            )?;
        }
        let sens_n = scalar_sum(&MASS_SENS_COUNT);
        if sens_n > 0 {
            let mean = u64_to_f64(scalar_sum(&MASS_SENS_SUM_NANO)) / 1.0e9 / u64_to_f64(sens_n);
            let (min_nano, mass_milli) = mass_min_sens_with_mass();
            let min = u64_to_f64(min_nano) / 1.0e9;
            let mass_at_min = u64_to_f64(mass_milli) / 1.0e6;
            let (max_nano, max_mass_milli) = mass_max_sens_with_mass();
            let max = u64_to_f64(max_nano) / 1.0e9;
            let mass_at_max = u64_to_f64(max_mass_milli) / 1.0e6;
            // The breach under test: 10.766 m of endpoint error.
            let breach_km = 10.766_135e-3;
            // The `xtol` direction: the arc's DELIVERED accuracy, converted into
            // the mass resolution it supports on the most sensitive row. Below
            // this, a finer mass tolerance is resolving integrator noise.
            let delivered_km = 0.012_342e-3;
            append_format(
                out,
                format_args!(
                    "PROP_MASSSENS rows,{sens_n},mean_km_per_kg,{mean:.6e},min_km_per_kg,{min:.6e},\
                     mass_at_min_kg,{mass_at_min:.6},dmass_at_mean_kg,{:.6e},dmass_at_min_kg,{:.6e},\
                     rel_at_min,{:.6e},mass_sum_micrograms,{},max_km_per_kg,{max:.6e},\
                     mass_at_max_kg,{mass_at_max:.6},xtol_floor_at_max_kg,{:.6e},\
                     xtol_floor_at_mean_kg,{:.6e},xtol_floor_at_min_kg,{:.6e}\n",
                    breach_km / mean,
                    breach_km / min,
                    if mass_at_min > 0.0 {
                        breach_km / min / mass_at_min
                    } else {
                        f64::NAN
                    },
                    scalar_sum(&MASS_SUM_MICROG),
                    if max > 0.0 { delivered_km / max } else { f64::NAN },
                    delivered_km / mean,
                    delivered_km / min
                ),
            )?;
        }
        append_format(
            out,
            format_args!(
                "PROP_UNDERFLOW accepts,{}\n",
                UNDERFLOW_ACCEPTS.load(Ordering::Relaxed)
            ),
        )?;
    }
    Ok(())
}

fn write_census_details(out: &mut CensusReport) -> CensusResult<()> {
    #[cfg(feature = "prop-census")]
    {
        for (tag, name) in TAG_NAMES.iter().enumerate().skip(TAG_RC_SKIP_PROBE) {
            append_format(
                out,
                format_args!(
                    "PROP_SCOPE {name},entries,{}\n",
                    tag_sum(&SCOPE_ENTRIES, tag)
                ),
            )?;
        }
        let captured = census_lock(&CAPTURED_ARCS)?;
        for line in captured.iter() {
            append_format(out, format_args!("PROP_ARC {line}\n"))?;
        }
        drop(captured);
        for (tag, (submitted, unique)) in state_census_by_tag()?.into_iter().enumerate() {
            if submitted > 0 {
                let duplicate_count = submitted
                    .checked_sub(unique)
                    .ok_or(PropagationCensusError::CounterOverflow)?;
                let tag_name = TAG_NAMES
                    .get(tag)
                    .copied()
                    .ok_or(PropagationCensusError::CounterOverflow)?;
                append_format(
                    out,
                    format_args!(
                        "PROP_DEDUP {tag_name},submitted,{submitted},unique,{unique},dup,{duplicate_count}\n"
                    ),
                )?;
            }
        }
        write_prefix_census(out)?;
    }
    // Announce the absence rather than printing a silently shorter report. A
    // reader who sees no PROP_DEDUP lines cannot otherwise tell "no duplicates"
    // from "the census was not compiled in", and those mean opposite things.
    #[cfg(not(feature = "prop-census"))]
    append_format(
        out,
        format_args!("PROP_DEDUP not-compiled-in (build with --features prop-census)\n"),
    )?;
    Ok(())
}

/// Render bounded propagation telemetry and optional census evidence.
///
/// # Errors
///
/// Returns a typed error when census data is invalid, a counter cannot be
/// represented exactly, a retained-evidence lock is poisoned, or bounded
/// output storage cannot be allocated.
pub fn report() -> CensusResult<String> {
    #[cfg(feature = "prop-census")]
    ensure_census_valid()?;
    let census = snapshot();
    let total_rhs = census.iter().try_fold(0_u64, |total, row| {
        total
            .checked_add(row.rhs_evals)
            .ok_or(PropagationCensusError::CounterOverflow)
    })?;
    let mut out = CensusReport::new()?;
    write_census_table(&mut out, &census, total_rhs)?;
    write_runtime_summary(&mut out)?;
    write_census_details(&mut out)?;
    Ok(out.into_inner())
}

#[cfg(test)]
mod shard_tests {
    use super::*;

    #[test]
    fn retired_rhs_timing_surface_is_absent() {
        let probe_source = include_str!("probe.rs");
        let production_probe = probe_source
            .split("\n#[cfg(test)]\nmod shard_tests")
            .next()
            .expect("probe production prefix");
        let integrator_source = include_str!("integrator.rs");

        for token in [
            ["MassPhase", "Timer"].concat(),
            ["PostprocessPhase", "Timer"].concat(),
            ["Prop", "Timer"].concat(),
            ["add_stage", "_ns"].concat(),
            ["mass_phase", "_seconds"].concat(),
            ["MASS_PHASE", "_"].concat(),
            ["PP_PHASE", "_"].concat(),
            ["STAGE", "_NS"].concat(),
            ["PostprocessDescriptor", "Sample"].concat(),
            ["PostprocessDescriptor", "Cost"].concat(),
            ["record_postprocess_descriptor", "_costs"].concat(),
            ["postprocess_descriptor", "_costs"].concat(),
            ["ND_PP_COST", "_DUMP"].concat(),
            ["POSTPROCESS_DESCRIPTOR", "_COSTS"].concat(),
            ["POSTPROCESS_BATCH", "_SEQ"].concat(),
            ["RHS", "_NS"].concat(),
            ["PROP", "_NS"].concat(),
            ["RHS", "_TIMED"].concat(),
        ] {
            assert!(
                !production_probe.contains(&token),
                "probe production source still contains retired timing token `{token}`"
            );
        }
        assert!(
            !integrator_source.contains(&["Prop", "Timer"].concat()),
            "checked final integrator still constructs the retired propagation timer"
        );
        assert!(
            !integrator_source.contains(&["Rhs", "Timer"].concat()),
            "integrator source still cites the retired RHS timer"
        );
    }

    /// Read the cell in THIS THREAD's shard.
    ///
    /// The counters are sharded, so a census assertion must address the shard
    /// `bump_propagation` will write, not shard 0. `current_shard` is assigned
    /// once per thread and never changes, so the two agree by construction --
    /// and the caller asserts the index is in range, so the `map_or` fallback
    /// cannot quietly turn an assertion into a comparison of two zeroes.
    #[cfg(feature = "prop-census")]
    fn own_shard_load(counter: &TagCounter, tag: usize) -> u64 {
        counter
            .get(current_shard())
            .map_or(0, |shard| atomic_load(&shard.0, tag))
    }

    #[cfg(feature = "prop-census")]
    fn own_shard_store(counter: &TagCounter, tag: usize, value: u64) {
        if let Some(shard) = counter.get(current_shard()) {
            atomic_store(&shard.0, tag, value);
        }
    }

    /// Saturate this thread's cell and return what it held.
    #[cfg(feature = "prop-census")]
    fn own_shard_swap(counter: &TagCounter, tag: usize, value: u64) -> u64 {
        counter.get(current_shard()).map_or(0, |shard| {
            shard
                .0
                .get(tag)
                .map_or(0, |cell| cell.swap(value, Ordering::Relaxed))
        })
    }

    #[test]
    fn collection_active_error_is_available_without_census_instrumentation() {
        assert_eq!(
            PropagationCensusError::CollectionActive.to_string(),
            "propagation census collection is active"
        );
    }

    #[cfg(feature = "prop-census")]
    #[test]
    fn collection_active_error_code_roundtrips() {
        assert_eq!(
            census_error_from_code(census_error_code(PropagationCensusError::CollectionActive)),
            PropagationCensusError::CollectionActive
        );
    }

    #[cfg(feature = "prop-census")]
    #[test]
    fn propagation_census_overflow_rejects_before_mutating_any_counter() {
        let _census = test_census_guard();
        let _tag = scope(TAG_OTHER);
        assert!(
            current_shard() < NSHARD && TAG_OTHER < NTAG,
            "shard/tag accessors below must address a real cell"
        );
        let before_props = own_shard_load(&PROPAGATIONS, TAG_OTHER);
        let before_span = own_shard_load(&SPAN_MS, TAG_OTHER);

        TL_ALL_PROPS.with(|cell| {
            let before_tls = cell.replace(u64::MAX);
            assert_eq!(
                bump_propagation(0.0, 1.0),
                Ok(()),
                "census exhaustion must not be reported to the propagation caller"
            );
            assert_eq!(cell.get(), u64::MAX);
            cell.set(before_tls);
        });

        assert_eq!(own_shard_load(&PROPAGATIONS, TAG_OTHER), before_props);
        assert_eq!(own_shard_load(&SPAN_MS, TAG_OTHER), before_span);
        assert_eq!(
            ensure_census_valid(),
            Err(PropagationCensusError::CounterOverflow),
            "census exhaustion must still latch the observation invalid"
        );
        clear_census_invalidation();
    }

    #[cfg(feature = "prop-census")]
    #[test]
    fn span_overflow_quarantines_partial_census_before_report() {
        let _census = test_census_guard();
        let _tag = scope(TAG_OTHER);
        // Saturate THIS THREAD's shard: sharded overflow is per shard, and this
        // thread's `bump_propagation` writes the shard addressed here.
        assert!(
            current_shard() < NSHARD && TAG_OTHER < NTAG,
            "shard/tag accessors below must address a real cell"
        );
        let before_props = own_shard_load(&PROPAGATIONS, TAG_OTHER);
        let before_span = own_shard_swap(&SPAN_MS, TAG_OTHER, u64::MAX);
        let before_all = TL_ALL_PROPS.with(Cell::get);

        assert_eq!(
            bump_propagation(0.0, 1.0),
            Ok(()),
            "a second global-counter overflow must quarantine the measurement, not the arc"
        );
        assert_eq!(
            report(),
            Err(PropagationCensusError::CounterOverflow),
            "report must not seal partial multi-counter telemetry"
        );

        own_shard_store(&PROPAGATIONS, TAG_OTHER, before_props);
        own_shard_store(&SPAN_MS, TAG_OTHER, before_span);
        TL_ALL_PROPS.with(|cell| cell.set(before_all));
        clear_census_invalidation();
    }

    /// Restore a fresh census epoch without the full `reset()` -- these tests
    /// run beside others in one process and must not clear their counters.
    #[cfg(feature = "prop-census")]
    fn clear_census_invalidation() {
        CENSUS_INVALID_KIND.store(0, Ordering::Release);
        CENSUS_INVALID.store(false, Ordering::Release);
    }

    /// The defect this feature shipped with: past `MAX_STATE_CENSUS_KEYS` a
    /// NOVEL key returned `Err`, the propagation entry mapped it to
    /// `FinalPropagationFailure::Census`, and callers scored it as an
    /// infeasible design -- while REPEAT keys kept propagating, so the arcs
    /// that survived were the un-novel ones. Saturation must now be visible
    /// only in the census.
    #[cfg(feature = "prop-census")]
    #[test]
    fn state_census_saturation_invalidates_the_census_not_the_propagation() {
        let _census = test_census_guard();
        let _tag = scope(TAG_OTHER);
        let mut states = census_lock(&SEEN_STATES).expect("census lock");
        let before = states.take();
        drop(states);
        clear_census_invalidation();

        let science = ScienceKey {
            jd0: 0.0,
            am_ratio: 0.0,
            cd: 0.0,
            cr: 0.0,
            eps: 0.0,
            dt_max: 0.0,
            sph_order: 0,
            force_flags: 0,
            atm_model: 0,
        };
        // Limit 1: the first key fits, the second is novel and does not.
        record_state_within(&[1.0; 6], 0.0, 1.0, &science, 1);
        assert_eq!(
            ensure_census_valid(),
            Ok(()),
            "a key that fits must leave the census usable"
        );

        record_state_within(&[2.0; 6], 0.0, 1.0, &science, 1);
        assert_eq!(
            ensure_census_valid(),
            Err(PropagationCensusError::Allocation),
            "a novel key past the ceiling must latch the census invalid"
        );
        assert_eq!(
            report(),
            Err(PropagationCensusError::Allocation),
            "report must refuse numbers taken across a saturated census"
        );

        // A repeat of the seen key still counts, and the map stays bounded --
        // the ceiling is still a ceiling.
        record_state_within(&[1.0; 6], 0.0, 1.0, &science, 1);
        let mut states = census_lock(&SEEN_STATES).expect("census lock");
        assert_eq!(
            states.as_ref().map(std::collections::HashMap::len),
            Some(1),
            "the bounded map must not have grown past its limit"
        );
        assert_eq!(
            states.as_ref().map(|map| map.values().sum::<u64>()),
            Some(2),
            "a repeat of a seen key must still be counted after saturation"
        );
        *states = before;
        drop(states);
        clear_census_invalidation();
    }

    #[cfg(feature = "prop-census")]
    #[test]
    fn lm_pass_id_overflow_is_typed() {
        let _census = test_census_guard();
        let before = LM_PASS_SEQ.swap(u64::MAX, Ordering::Relaxed);
        assert_eq!(
            lm_next_pass_id(),
            Err(PropagationCensusError::CounterOverflow)
        );
        LM_PASS_SEQ.store(before, Ordering::Relaxed);
    }

    #[test]
    fn bounded_census_row_insert_rejects_capacity_without_partial_row() {
        let mut rows = vec![1_u64];
        assert_eq!(
            try_push_census_row(&mut rows, 1, 2),
            Err(PropagationCensusError::Allocation)
        );
        assert_eq!(rows, vec![1]);
    }

    #[cfg(not(feature = "prop-census"))]
    #[test]
    fn relaxed_census_add_rejects_a_missing_counter() {
        let counters: [AtomicU64; 0] = [];
        assert_eq!(
            relaxed_atomic_add(&counters, 0, 1),
            Err(PropagationCensusError::CounterOverflow)
        );
    }

    const fn row_key(design_key: usize, candidate_index: usize) -> SolvedMassRowKey {
        SolvedMassRowKey {
            design_key,
            event_index: 0,
            candidate_index,
            fraction_index: 0,
        }
    }

    #[test]
    fn solved_mass_capture_rejects_limit_without_partial_append() {
        let mut rows = vec![(row_key(0, 0), 1.0_f64.to_bits())];
        assert_eq!(
            append_solved_mass_rows(&mut rows, [(row_key(0, 1), 2.0)].into_iter(), 1),
            Err(PropagationCensusError::Allocation)
        );
        assert_eq!(rows, vec![(row_key(0, 0), 1.0_f64.to_bits())]);
    }

    /// Two runs that produce the same rows in different CAPTURE orders must
    /// become identical once joined on the key. Positional zipping of these two
    /// vectors disagrees on every row, which is the whole point: without the
    /// key the same evidence reads as total movement.
    #[test]
    fn solved_mass_rows_agree_across_capture_orders_only_when_joined_on_the_key() {
        let masses = [
            (row_key(0, 0), 1.0_f64),
            (row_key(0, 1), 2.0),
            (row_key(1, 0), 3.0),
        ];
        let mut first = Vec::new();
        append_solved_mass_rows(&mut first, masses.into_iter(), 16).expect("first capture");

        // A different flush order for the same physical rows.
        let mut second = Vec::new();
        let mut permuted = masses;
        permuted.swap(0, 2);
        append_solved_mass_rows(&mut second, permuted.into_iter(), 16).expect("second capture");

        assert_ne!(
            first, second,
            "the permutation must actually change capture order, or this proves nothing"
        );
        assert!(
            first
                .iter()
                .zip(second.iter())
                .any(|((_, lhs), (_, rhs))| lhs != rhs),
            "a positional join must disagree, or the key is not what is fixing it"
        );

        first.sort_by_key(|(key, _)| *key);
        second.sort_by_key(|(key, _)| *key);
        assert_eq!(first, second);
    }

    /// The pairing property the old two-atomic form could not provide: the
    /// retained mass is the one submitted WITH the retained slope, whatever the
    /// arrival order.
    ///
    /// The masses are deliberately anti-correlated with the slopes, so a
    /// `fetch_min`-then-separate-`store` sequence would leave the largest mass
    /// beside the smallest slope on at least one ordering.
    ///
    /// Also pins the RANGE that killed the packed-into-one-`AtomicU64` variant:
    /// a production `min_km_per_kg` of 38.29 is 3.829e10 in nano units, nine
    /// times past what a 32-bit lane holds.
    #[test]
    fn mass_min_sens_keeps_the_mass_that_arrived_with_the_slope() {
        let samples = [
            (2_370_507_000_000_u64, 100_u64),
            (38_292_770_000, 25_530),
            (500_000_000_000, 900_000),
        ];
        for rotation in 0..samples.len() {
            let mut slot = (u64::MAX, 0_u64);
            for (slope, mass) in samples.iter().cycle().skip(rotation).take(samples.len()) {
                slot = merge_mass_min_sens(slot, *slope, *mass);
            }
            assert_eq!(
                slot,
                (38_292_770_000, 25_530),
                "arrival order {rotation} changed the retained pair"
            );
            assert!(
                slot.0 > u64::from(u32::MAX),
                "the measured minimum must not fit a 32-bit lane, or this pins nothing"
            );
        }
    }

    #[cfg(feature = "prop-census")]
    #[test]
    fn state_census_rejects_new_map_before_committing_it() {
        let mut states = None;
        assert_eq!(
            insert_state_census_key(&mut states, [0; 16], 0),
            Err(PropagationCensusError::Allocation)
        );
        assert!(states.is_none());
    }

    #[test]
    fn poisoned_census_lock_is_typed() {
        let lock = std::sync::Mutex::new(());
        std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                let _guard = lock.lock();
                panic!("test-only census lock poison");
            });
            assert!(handle.join().is_err());
        });
        assert_eq!(
            census_lock(&lock).map(|_| ()),
            Err(PropagationCensusError::MutexPoisoned)
        );
    }

    #[test]
    fn bounded_report_rejects_overflow_without_partial_output() -> CensusResult<()> {
        let mut report = CensusReport::new()?;
        append_format(&mut report, format_args!("PROP_TEST ok\n"))?;
        let before = report.as_str().to_owned();
        let mut overflow = String::new();
        overflow
            .try_reserve_exact(
                MAX_CENSUS_REPORT_BYTES
                    .checked_add(1)
                    .ok_or(PropagationCensusError::CounterOverflow)?,
            )
            .map_err(|_| PropagationCensusError::Allocation)?;
        for _ in 0..=MAX_CENSUS_REPORT_BYTES {
            overflow.push('x');
        }
        match append_format(&mut report, format_args!("{overflow}")) {
            Err(PropagationCensusError::Allocation) if report.as_str() == before => Ok(()),
            Err(error) => Err(error),
            Ok(()) => Err(PropagationCensusError::CounterOverflow),
        }
    }

    /// Sharding is only safe if the SUMMED value is identical to what one
    /// unsharded counter would have held. A lost update would silently corrupt
    /// exactly the diagnostics this campaign uses to reason about itself, and
    /// `strict_hf_pin` only exercises one thread, so it cannot see this.
    ///
    /// Deliberately runs MORE threads than `NSHARD` so shard indices wrap and
    /// two threads share a slot -- the only case in which the `fetch_add` (as
    /// opposed to a plain `+= 1`) is load bearing.
    #[test]
    fn sharded_tag_counter_total_is_exact_under_shard_wraparound() {
        static COUNTER: TagCounter = new_tag_counter();
        const THREADS: usize = NSHARD + 37;
        const PER_THREAD: u64 = 5_000;
        const TAG: usize = 3;

        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    for _ in 0..PER_THREAD {
                        tag_add(&COUNTER, TAG);
                    }
                });
            }
        });

        let expected_total = u64::try_from(THREADS)
            .ok()
            .and_then(|threads| threads.checked_mul(PER_THREAD));
        assert_eq!(
            Some(tag_sum(&COUNTER, TAG)),
            expected_total,
            "sharded total must equal the unsharded total exactly"
        );
        for other in (0..NTAG).filter(|t| *t != TAG) {
            assert_eq!(
                tag_sum(&COUNTER, other),
                0,
                "tag {other} must not receive writes aimed at tag {TAG}"
            );
        }
        tag_clear(&COUNTER, TAG);
        assert_eq!(tag_sum(&COUNTER, TAG), 0, "clear must zero every shard");
    }

    #[test]
    fn sharded_scalar_counter_total_is_exact_under_shard_wraparound() {
        static COUNTER: ScalarCounter = new_scalar_counter();
        const THREADS: usize = NSHARD + 37;
        const PER_THREAD: u64 = 5_000;

        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    for _ in 0..PER_THREAD {
                        scalar_add(&COUNTER);
                    }
                });
            }
        });

        let expected_total = u64::try_from(THREADS)
            .ok()
            .and_then(|threads| threads.checked_mul(PER_THREAD));
        assert_eq!(Some(scalar_sum(&COUNTER)), expected_total);
        scalar_clear(&COUNTER);
        assert_eq!(scalar_sum(&COUNTER), 0, "clear must zero every shard");
    }

    /// The padding is the whole point: `NTAG * 8` is 88 bytes, so without
    /// `repr(align(128))` two shards would share a 64-byte line and the
    /// sharding would buy nothing.
    #[test]
    fn shards_do_not_share_a_cache_line() {
        assert_eq!(std::mem::align_of::<TagShard>(), 128);
        assert_eq!(std::mem::size_of::<TagShard>(), 128);
        assert_eq!(std::mem::align_of::<ScalarShard>(), 128);
        assert_eq!(std::mem::size_of::<ScalarShard>(), 128);
        let first = std::ptr::from_ref(&RHS_EVALS[0]).addr();
        let second = std::ptr::from_ref(&RHS_EVALS[1]).addr();
        assert_eq!(
            second.checked_sub(first),
            Some(128),
            "adjacent shards must be 128 bytes apart"
        );
    }

    /// One thread keeps one shard for its whole life.
    #[test]
    fn shard_assignment_is_stable_per_thread() {
        let first = current_shard();
        for _ in 0..1_000 {
            assert_eq!(current_shard(), first);
        }
        assert!(first < NSHARD);
    }

    /// `reset()` must clear the ramp accumulators, not just their neighbours.
    ///
    /// This shipped broken: `RAMP_H_NS`/`RAMP_COUNT` were added beside
    /// `CACHE_CLUSTER_STEPS` but not added to `reset()`, so `PROP_RAMP` carried
    /// the previous measurement into the next one. The failure mode is the
    /// dangerous kind -- the blended numbers stay plausible, so nothing looks
    /// wrong.
    ///
    /// Asserts on the RENDERED report rather than on the statics, because the
    /// report is what a measurement is read from and it is the thing that was
    /// actually wrong. The shared census-test guard serializes this reset and
    /// assertion against every other in-process census test.
    #[cfg(feature = "prop-census")]
    #[test]
    fn reset_clears_the_ramp_accumulators() {
        fn ramp_rows(report: &str) -> usize {
            report
                .lines()
                .filter(|line| line.starts_with("PROP_RAMP ") && !line.contains("population,slot"))
                .count()
        }
        let _census = test_census_guard();

        reset().expect("census reset");
        observe_ramp(
            crate::integrator::SegmentBoundary::Rebased,
            1.0,
            &[0.25, 0.5],
            4.0,
            2,
        );
        observe_ramp(
            crate::integrator::SegmentBoundary::EventContinuation,
            600.0,
            &[10.0],
            90.0,
            3,
        );

        // Non-vacuity FIRST: if `observe_ramp` recorded nothing, the assertion
        // below would pass against a report that was empty for the wrong
        // reason. Two populations, so two distinct rows at minimum.
        let populated = report().expect("report must render");
        assert!(
            ramp_rows(&populated) >= 2,
            "observe_ramp must produce rows before reset can be shown to clear them:\n{populated}"
        );

        reset().expect("census reset");
        let cleared = report().expect("report must render");
        assert_eq!(
            ramp_rows(&cleared),
            0,
            "reset() must clear PROP_RAMP; it survived:\n{cleared}"
        );
    }

    /// Site rows are report-owned evidence, not a second diagnostic printer.
    /// The state accessor exists so coordinator tests can reconcile every
    /// accepted step and RHS evaluation against the production result.
    #[cfg(feature = "prop-census")]
    #[test]
    fn eclipse_transaction_site_report_exposes_exact_owned_counters() {
        let _census = test_census_guard();
        reset().expect("census reset");
        {
            let _site = eclipse_transaction_scope(EclipseTransactionSite::Refine);
            observe_leg(10.0, 25.0, 3, 31, 0);
        }

        let sites = eclipse_transaction_site_snapshot();
        let refine = sites
            .get(EclipseTransactionSite::Refine.index())
            .copied()
            .expect("refine site");
        assert_eq!(refine.legs, 1);
        assert_eq!(refine.steps, 3);
        assert_eq!(refine.evals, 31);
        assert_eq!(refine.rejected, 0);
        assert_eq!(refine.fantasy_removable_steps, 2);
        assert_eq!(refine.fantasy_removable_evals, 20);
        assert!(sites
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != EclipseTransactionSite::Refine.index())
            .all(|(_, row)| *row == EclipseTransactionSiteCensus::default()));

        let rendered = report().expect("site census must render through probe::report");
        assert!(rendered.contains("PROP_ECLIPSE_SITE refine,1,3,31,0,2,20\n"));
    }

    #[cfg(feature = "prop-census")]
    #[test]
    fn eclipse_transaction_site_scope_nests_and_restores_without_leakage() {
        let _census = test_census_guard();
        reset().expect("census reset");
        {
            let _main = eclipse_transaction_scope(EclipseTransactionSite::Main);
            observe_leg(300.0, 20.0, 2, 21, 0);
            {
                let _proof = eclipse_transaction_scope(EclipseTransactionSite::Proof);
                observe_leg(10.0, 5.0, 1, 11, 0);
            }
            observe_leg(300.0, 20.0, 3, 31, 0);
        }
        observe_leg(300.0, 20.0, 4, 41, 0);

        let sites = eclipse_transaction_site_snapshot();
        let site_row =
            |site: EclipseTransactionSite| sites.get(site.index()).copied().expect("known site");
        assert_eq!(
            site_row(EclipseTransactionSite::Main),
            EclipseTransactionSiteCensus {
                legs: 2,
                steps: 5,
                evals: 52,
                rejected: 0,
                fantasy_removable_steps: 3,
                fantasy_removable_evals: 30,
            }
        );
        assert_eq!(
            site_row(EclipseTransactionSite::Proof),
            EclipseTransactionSiteCensus {
                legs: 1,
                steps: 1,
                evals: 11,
                rejected: 0,
                fantasy_removable_steps: 0,
                fantasy_removable_evals: 0,
            }
        );
        assert_eq!(site_row(EclipseTransactionSite::Refine).legs, 0);
        assert_eq!(site_row(EclipseTransactionSite::Window).legs, 0);
        assert_eq!(
            sites.iter().map(|row| row.legs).sum::<u64>(),
            3,
            "unscoped leg must stay unowned"
        );
    }

    #[cfg(feature = "prop-census")]
    #[test]
    fn eclipse_transaction_site_scope_does_not_cross_reset_epoch() {
        let _census = test_census_guard();
        reset().expect("census reset");
        {
            let _stale = eclipse_transaction_scope(EclipseTransactionSite::Main);
            observe_leg(300.0, 20.0, 2, 21, 0);
            reset().expect("census reset inside scope");
            observe_leg(300.0, 20.0, 3, 31, 0);
        }
        observe_leg(300.0, 20.0, 4, 41, 0);
        {
            let _fresh = eclipse_transaction_scope(EclipseTransactionSite::Window);
            observe_leg(10.0, 5.0, 1, 11, 0);
        }

        let sites = eclipse_transaction_site_snapshot();
        let site_row =
            |site: EclipseTransactionSite| sites.get(site.index()).copied().expect("known site");
        assert_eq!(site_row(EclipseTransactionSite::Main).legs, 0);
        assert_eq!(site_row(EclipseTransactionSite::Refine).legs, 0);
        assert_eq!(site_row(EclipseTransactionSite::Proof).legs, 0);
        assert_eq!(site_row(EclipseTransactionSite::Window).legs, 1);
    }

    #[cfg(feature = "prop-census")]
    #[test]
    fn eclipse_transaction_site_fantasy_ceiling_sums_per_leg() {
        let _census = test_census_guard();
        reset().expect("census reset");
        {
            let _site = eclipse_transaction_scope(EclipseTransactionSite::Refine);
            for (steps, evals) in [(0, 0), (2, 20), (1, 5), (4, 30)] {
                observe_leg(10.0, 1.0, steps, evals, 0);
            }
        }

        let sites = eclipse_transaction_site_snapshot();
        let refine = sites
            .get(EclipseTransactionSite::Refine.index())
            .copied()
            .expect("refine site");
        assert_eq!(refine.legs, 4);
        assert_eq!(refine.steps, 7);
        assert_eq!(refine.evals, 55);
        assert_eq!(refine.fantasy_removable_steps, 4);
        assert_eq!(refine.fantasy_removable_evals, 28);
        assert_ne!(
            refine.fantasy_removable_steps,
            refine.steps.saturating_sub(refine.legs)
        );
        assert_ne!(
            refine.fantasy_removable_evals,
            refine.evals.saturating_sub(11 * refine.legs)
        );
        let rendered = report().expect("census report");
        assert!(rendered.contains("PROP_ECLIPSE_SITE refine,4,7,55,0,4,28\n"));
    }

    /// `reset()` must clear BOTH mass-sensitivity extrema, not only the min.
    ///
    /// Same shape as the ramp bug above and shipped the same way: the max end
    /// was added beside the min and `reset()` only ever cleared the min, so a
    /// harness that resets between arms carried the previous arm's maximum
    /// into `PROP_MASSSENS`. A running extremum fails silently -- the number
    /// stays plausible, it just belongs to the wrong arm.
    ///
    /// Asserts on the rendered report, because that is what a measurement is
    /// read from. The two slopes are four orders apart so the surviving value
    /// cannot be mistaken for the fresh one.
    #[cfg(feature = "prop-census")]
    #[test]
    fn reset_clears_both_mass_sensitivity_extrema() {
        fn max_km_per_kg(report: &str) -> Option<f64> {
            let line = report
                .lines()
                .find(|line| line.starts_with("PROP_MASSSENS "))?;
            let fields: Vec<&str> = line.split(',').collect();
            let at = fields.iter().position(|f| *f == "max_km_per_kg")?;
            fields.get(at + 1)?.trim().parse().ok()
        }
        let _census = test_census_guard();

        // `PROP_MASSSENS` renders only inside the `min_accepted_h_s()` block,
        // so an arm that accepted no step prints nothing at all.
        reset().expect("census reset");
        observe_min_h(1.0);
        record_mass_sensitivity(1.0, 1.0e4);

        // Non-vacuity FIRST: if the big sample never landed, the assertion
        // below would pass against a report that never carried it at all.
        let populated = report().expect("report must render");
        let big = max_km_per_kg(&populated)
            .expect("PROP_MASSSENS must carry max_km_per_kg before reset is tested");
        assert!(
            big > 1.0e3,
            "the large slope must reach the max slot, got {big:e}:\n{populated}"
        );

        reset().expect("census reset");
        observe_min_h(1.0);
        record_mass_sensitivity(1.0, 1.0);
        let fresh = report().expect("report must render");
        let after =
            max_km_per_kg(&fresh).expect("PROP_MASSSENS must render after the second sample");
        assert!(
            after < 1.0e3,
            "reset() must clear the mass-sensitivity MAX; the previous arm's \
             {big:e} survived as {after:e}:\n{fresh}"
        );
    }
}
