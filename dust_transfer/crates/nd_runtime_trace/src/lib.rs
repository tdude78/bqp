//! Fixed-lane, lossy runtime tracing for long-running NASA Dust commands.
//!
//! This crate is deliberately outside the scientific evidence graph. Producers
//! move fixed-layout records into one SPSC lane each; only the dedicated writer
//! thread formats or touches the filesystem. Trace failure never changes a
//! command result.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write as _};
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Args;
use rtrb::{Consumer, PopError, Producer, PushError, RingBuffer};
use serde::Serialize;

/// Schema carried by every trace header and record.
pub const TRACE_SCHEMA: &str = "nasa_dust.runtime_trace.v2";
/// Fixed production capacity of every SPSC lane.
pub const DEFAULT_QUEUE_CAPACITY: usize = 256;
/// Hard process lane ceiling, including the controller lane.
pub const MAX_LANES: usize = 512;
/// Maximum bytes occupied by all fixed transport records.
pub const MAX_TRANSPORT_BYTES: usize = 32 * 1024 * 1024;
/// Maximum encoded bytes in one event line, including its newline.
pub const MAX_EVENT_LINE_BYTES: usize = 4 * 1024;
/// Default maximum trace file size.
pub const MAX_TRACE_BYTES: usize = 64 * 1024 * 1024;
/// Space held back for per-lane and terminal accounting.
pub const TERMINAL_RESERVE_BYTES: usize = 256 * 1024;
/// Production heartbeat interval.
pub const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

const IDLE_POLL: Duration = Duration::from_millis(10);
const FINISH_TIMEOUT: Duration = Duration::from_secs(5);
const DRAIN_BURST: usize = 32;

/// Flattened trace flags shared by the four trace-capable command families.
#[derive(Debug, Clone, Default, Args)]
pub struct TraceArgs {
    /// Disable the default runtime trace (for reference timing runs).
    #[arg(long, conflicts_with = "trace_dir")]
    pub no_runtime_trace: bool,
    /// Absolute external directory for this process's runtime trace.
    #[arg(long, value_name = "ABSOLUTE_PATH", value_parser = absolute_trace_dir)]
    pub trace_dir: Option<PathBuf>,
}

fn absolute_trace_dir(value: &str) -> anyhow::Result<PathBuf> {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        Ok(path)
    } else {
        Err(anyhow::anyhow!("--trace-dir must be an absolute path"))
    }
}

impl TraceArgs {
    /// Whether the command should start its default trace.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        !self.no_runtime_trace
    }
}

/// A compact pre-registered context key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ContextId(u16);

impl ContextId {
    /// Process-wide context, always registered as zero.
    pub const PROCESS: Self = Self(0);

    /// Construct a compact context key.
    #[must_use]
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    /// Canonical pre-registration key for one input cell index.
    #[must_use]
    pub fn for_cell_index(index: usize) -> Option<Self> {
        u16::try_from(index)
            .ok()
            .and_then(|value| value.checked_add(1))
            .map(Self)
    }
}

/// Writer-side expansion for a compact producer context.
#[derive(Debug, Clone, Serialize)]
pub struct TraceContext {
    pub id: ContextId,
    pub kind: String,
    pub identity: String,
}

/// Static command/process role recorded in the filename and header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandRole {
    Matrix,
    PartACanary,
    ShardWorker,
    /// Local producer-cost diagnostic; never selected by the `nd` CLI.
    EmitBench,
}

impl CommandRole {
    const fn file_label(self) -> &'static str {
        match self {
            Self::Matrix => "matrix",
            Self::PartACanary => "part-a-canary",
            Self::ShardWorker => "shard-worker",
            Self::EmitBench => "emit-bench",
        }
    }
}

/// Scheduler identity is diagnostic header data only.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SchedulerIdentity {
    pub job: Option<String>,
    pub array_task: Option<String>,
    pub step: Option<String>,
    pub node: Option<String>,
    pub rank: Option<String>,
}

impl SchedulerIdentity {
    /// Read bounded scheduler identity fields. They never affect execution.
    #[must_use]
    pub fn from_environment() -> Self {
        Self {
            job: bounded_env("SLURM_JOB_ID"),
            array_task: bounded_env("SLURM_ARRAY_TASK_ID"),
            step: bounded_env("SLURM_STEP_ID"),
            node: bounded_env("SLURMD_NODENAME"),
            rank: bounded_env("SLURM_PROCID"),
        }
    }
}

fn bounded_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().chars().take(96).collect::<String>())
        .filter(|value| !value.is_empty())
}

/// Default matrix trace directory, always outside matrix authority roots.
#[must_use]
pub fn matrix_default_trace_dir(
    run_root: &Path,
    run_id: &str,
    canonical: bool,
    current_dir: &Path,
) -> PathBuf {
    if canonical {
        external_sibling_trace_root(run_root).join(run_id)
    } else {
        current_dir
            .join("build")
            .join("runtime-traces")
            .join(run_id)
    }
}

/// Default canary trace directory.
#[must_use]
pub fn canary_default_trace_dir(canary_run_directory: &Path) -> PathBuf {
    external_sibling_trace_root(canary_run_directory)
}

/// Default ad hoc shard-worker trace directory.
#[must_use]
pub fn shard_worker_default_trace_dir(rendezvous: &Path) -> PathBuf {
    external_sibling_trace_root(rendezvous)
}

fn external_sibling_trace_root(path: &Path) -> PathBuf {
    let mut suffixed = path.as_os_str().to_os_string();
    suffixed.push(".traces");
    PathBuf::from(suffixed)
}

/// Fixed startup configuration. All owned strings are consumed by the writer,
/// never by producer calls.
#[derive(Debug, Clone)]
pub struct TraceConfig {
    pub output_dir: PathBuf,
    pub command_role: CommandRole,
    pub binary_sha256: [u8; 32],
    /// Hash of the source revision when the caller has a stamped authority for
    /// it. `None` is serialized explicitly instead of inventing provenance.
    pub revision_sha256: Option<[u8; 32]>,
    pub scheduler: SchedulerIdentity,
    pub contexts: Vec<TraceContext>,
    pub driver_lanes: usize,
    pub heartbeat_interval: Duration,
    pub queue_capacity: usize,
    pub byte_cap: usize,
    /// Writer-thread stderr echo for Slurm `.out` / test harnesses. Crate-unit
    /// traces turn this off because cargo does not capture writer-thread stdio.
    pub echo_live: bool,
    #[cfg(test)]
    test_writer_delay: Duration,
    #[cfg(test)]
    test_file_stem: Option<String>,
    #[cfg(test)]
    test_fault: TestFault,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
enum TestFault {
    #[default]
    None,
    WriteAfter(usize),
    PartialWriteAfter(usize),
    Flush,
    Sync,
}

impl TraceConfig {
    /// Construct the production fixed-resource configuration.
    #[must_use]
    pub const fn new(
        output_dir: PathBuf,
        command_role: CommandRole,
        binary_sha256: [u8; 32],
        revision_sha256: Option<[u8; 32]>,
        scheduler: SchedulerIdentity,
        contexts: Vec<TraceContext>,
        driver_lanes: usize,
    ) -> Self {
        Self {
            output_dir,
            command_role,
            binary_sha256,
            revision_sha256,
            scheduler,
            contexts,
            driver_lanes,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            byte_cap: MAX_TRACE_BYTES,
            echo_live: true,
            #[cfg(test)]
            test_writer_delay: Duration::ZERO,
            #[cfg(test)]
            test_file_stem: None,
            #[cfg(test)]
            test_fault: TestFault::None,
        }
    }

    #[cfg(test)]
    fn test(
        output_dir: &Path,
        driver_lanes: usize,
        queue_capacity: usize,
        byte_cap: usize,
    ) -> Self {
        Self {
            output_dir: output_dir.to_path_buf(),
            command_role: CommandRole::Matrix,
            binary_sha256: [0x11; 32],
            revision_sha256: Some([0x22; 32]),
            scheduler: SchedulerIdentity::default(),
            contexts: Vec::new(),
            driver_lanes,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            queue_capacity,
            byte_cap,
            echo_live: false,
            test_writer_delay: Duration::ZERO,
            test_file_stem: None,
            test_fault: TestFault::None,
        }
    }

    #[cfg(test)]
    const fn with_test_writer_delay(mut self, delay: Duration) -> Self {
        self.test_writer_delay = delay;
        self
    }

    #[cfg(test)]
    fn with_file_stem(mut self, stem: &str) -> Self {
        self.test_file_stem = Some(stem.to_owned());
        self
    }

    #[cfg(test)]
    const fn with_fault(mut self, fault: TestFault) -> Self {
        self.test_fault = fault;
        self
    }
}

/// Bounded, JSON-safe scalar summary. Non-finite values remain distinguishable
/// by class and raw bits without ever becoming invalid JSON numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarSummary {
    pub class: ScalarClass,
    pub bits: u64,
}

impl ScalarSummary {
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        let class = if value.is_nan() {
            ScalarClass::Nan
        } else if value == f64::INFINITY {
            ScalarClass::PositiveInfinity
        } else if value == f64::NEG_INFINITY {
            ScalarClass::NegativeInfinity
        } else {
            ScalarClass::Finite
        };
        Self {
            class,
            bits: value.to_bits(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarClass {
    Finite,
    Nan,
    PositiveInfinity,
    NegativeInfinity,
}

/// Closed fixed-layout producer payload.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "payload_kind", rename_all = "snake_case")]
pub enum TraceEvent {
    ProcessStarted {
        context: ContextId,
    },
    MatrixUnitSelected {
        context: ContextId,
        unit_index: u32,
    },
    AuthorityValidated {
        context: ContextId,
        authority_hash: [u8; 32],
    },
    SchedulerPoolReady {
        context: ContextId,
        width: u16,
    },
    CellStarted {
        context: ContextId,
        cell_index: usize,
    },
    CohortStarted {
        context: ContextId,
        cohort_index: usize,
    },
    OptimizerStarted {
        context: ContextId,
        optimizer_id: u16,
    },
    ObjectiveStarted {
        context: ContextId,
        generation: usize,
        objective_ordinal: u64,
        event_offset: usize,
        event_count: usize,
        active_designs: usize,
        candidate_count: usize,
        view_hash: [u8; 32],
    },
    ObjectiveFinished {
        context: ContextId,
        generation: usize,
        objective_ordinal: u64,
        physics_trials: u64,
        returned_designs: usize,
        status: u16,
    },
    OptimizerFinished {
        context: ContextId,
        status: u16,
    },
    CellFinished {
        context: ContextId,
        status: u16,
        recovered_shards: u64,
    },
    CohortFinished {
        context: ContextId,
        status: u16,
    },
    AdaptiveStageStarted {
        context: ContextId,
        generation: usize,
        event_offset: usize,
        event_count: usize,
        active_designs: usize,
    },
    AdaptiveStageFinished {
        context: ContextId,
        generation: usize,
        event_offset: usize,
        event_count: usize,
        active_designs: usize,
        physics_trials: u64,
        candidate_count: usize,
        status: u16,
    },
    ProfileSummary {
        context: ContextId,
        profile_kind: u16,
        calls: u64,
        elapsed_ns: u64,
        minimum: ScalarSummary,
        maximum: ScalarSummary,
    },
    K3BarrierCommitted {
        context: ContextId,
        barrier_index: u32,
        checkpoint_hash: [u8; 32],
    },
    CapRelaxation {
        context: ContextId,
        count: u64,
    },
    ReceiptSealed {
        context: ContextId,
        receipt_hash: [u8; 32],
    },
    ArtifactPublished {
        context: ContextId,
        publication_receipt_hash: [u8; 32],
    },
    ShardSessionReady {
        context: ContextId,
        worker_count: u16,
    },
    ShardWorkerJoined {
        context: ContextId,
        worker: u16,
    },
    ShardRequestStarted {
        context: ContextId,
        sequence: u64,
        design_count: usize,
    },
    ShardRequestFinished {
        context: ContextId,
        sequence: u64,
        status: u16,
    },
    ShardBatchRecovered {
        context: ContextId,
        sequence: u64,
        recovered_shards: u16,
    },
    ShardWorkerFinished {
        context: ContextId,
        batches: u64,
        status: u16,
    },
    Notice {
        context: ContextId,
        code: u16,
        detail: u64,
    },
}

impl TraceEvent {
    const fn name(self) -> &'static str {
        match self {
            Self::ProcessStarted { .. } => "process.started",
            Self::MatrixUnitSelected { .. } => "matrix.unit_selected",
            Self::AuthorityValidated { .. } => "authority.validated",
            Self::SchedulerPoolReady { .. } => "scheduler.pool.ready",
            Self::CellStarted { .. } => "cell.started",
            Self::CohortStarted { .. } => "cohort.started",
            Self::OptimizerStarted { .. } => "optimizer.started",
            Self::ObjectiveStarted { .. } => "objective.started",
            Self::ObjectiveFinished { .. } => "objective.finished",
            Self::OptimizerFinished { .. } => "optimizer.finished",
            Self::CellFinished { .. } => "cell.finished",
            Self::CohortFinished { .. } => "cohort.finished",
            Self::AdaptiveStageStarted { .. } => "adaptive_stage.started",
            Self::AdaptiveStageFinished { .. } => "adaptive_stage.finished",
            Self::ProfileSummary { .. } => "profile.summary",
            Self::K3BarrierCommitted { .. } => "k3.barrier.committed",
            Self::CapRelaxation { .. } => "cap_relaxation",
            Self::ReceiptSealed { .. } => "receipt.sealed",
            Self::ArtifactPublished { .. } => "artifact.published",
            Self::ShardSessionReady { .. } => "shard.session.ready",
            Self::ShardWorkerJoined { .. } => "shard.worker.joined",
            Self::ShardRequestStarted { .. } => "shard.request.started",
            Self::ShardRequestFinished { .. } => "shard.request.finished",
            Self::ShardBatchRecovered { .. } => "shard.batch.recovered",
            Self::ShardWorkerFinished { .. } => "shard.worker.finished",
            Self::Notice { .. } => "notice",
        }
    }

    const fn context(self) -> ContextId {
        match self {
            Self::ProcessStarted { context }
            | Self::MatrixUnitSelected { context, .. }
            | Self::AuthorityValidated { context, .. }
            | Self::SchedulerPoolReady { context, .. }
            | Self::CellStarted { context, .. }
            | Self::CohortStarted { context, .. }
            | Self::OptimizerStarted { context, .. }
            | Self::ObjectiveStarted { context, .. }
            | Self::ObjectiveFinished { context, .. }
            | Self::OptimizerFinished { context, .. }
            | Self::CellFinished { context, .. }
            | Self::CohortFinished { context, .. }
            | Self::AdaptiveStageStarted { context, .. }
            | Self::AdaptiveStageFinished { context, .. }
            | Self::ProfileSummary { context, .. }
            | Self::K3BarrierCommitted { context, .. }
            | Self::CapRelaxation { context, .. }
            | Self::ReceiptSealed { context, .. }
            | Self::ArtifactPublished { context, .. }
            | Self::ShardSessionReady { context, .. }
            | Self::ShardWorkerJoined { context, .. }
            | Self::ShardRequestStarted { context, .. }
            | Self::ShardRequestFinished { context, .. }
            | Self::ShardBatchRecovered { context, .. }
            | Self::ShardWorkerFinished { context, .. }
            | Self::Notice { context, .. } => context,
        }
    }
}

const _: () = assert!(std::mem::size_of::<TraceEvent>() <= 192);

#[derive(Debug, Clone, Copy)]
struct TraceRecord {
    lane_sequence: u64,
    elapsed_ns: u64,
    event: TraceEvent,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "payload_kind", rename_all = "snake_case")]
enum TerminalEvent {
    ProcessFinished {
        context: ContextId,
        status: u16,
    },
    ProcessFailed {
        context: ContextId,
        status: u16,
        error_hash: Option<Sha256Hex>,
    },
}

impl TerminalEvent {
    const fn new(command_outcome: CommandOutcome, error_hash: Option<[u8; 32]>) -> Self {
        match command_outcome {
            CommandOutcome::Succeeded => Self::ProcessFinished {
                context: ContextId::PROCESS,
                status: 0,
            },
            CommandOutcome::Failed => Self::ProcessFailed {
                context: ContextId::PROCESS,
                status: 1,
                error_hash: match error_hash {
                    Some(hash) => Some(Sha256Hex(hash)),
                    None => None,
                },
            },
        }
    }

    const fn command_outcome(self) -> CommandOutcome {
        match self {
            Self::ProcessFinished { .. } => CommandOutcome::Succeeded,
            Self::ProcessFailed { .. } => CommandOutcome::Failed,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::ProcessFinished { .. } => "process.finished",
            Self::ProcessFailed { .. } => "process.failed",
        }
    }

    const fn severity(self) -> &'static str {
        match self {
            Self::ProcessFinished { .. } => "info",
            Self::ProcessFailed { .. } => "error",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Sha256Hex([u8; 32]);

impl Serialize for Sha256Hex {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use std::fmt::Write as _;

        let mut encoded = String::with_capacity(64);
        for byte in self.0 {
            write!(&mut encoded, "{byte:02x}").map_err(serde::ser::Error::custom)?;
        }
        serializer.serialize_str(&encoded)
    }
}

#[derive(Debug, Clone, Copy)]
struct TerminalRecord {
    elapsed_ns: u64,
    event: TerminalEvent,
}

/// Fixed bytes reserved by one transport slot.
pub const TRACE_RECORD_SIZE_BYTES: usize = std::mem::size_of::<TraceRecord>();

#[repr(align(128))]
#[derive(Debug, Default)]
struct LaneCounters {
    full: AtomicU64,
    disconnected: AtomicU64,
    reentrant: AtomicU64,
}

#[derive(Debug)]
struct TraceLaneProducer {
    producer: Producer<TraceRecord>,
    counters: Arc<LaneCounters>,
    started: Instant,
    sequence: u64,
}

thread_local! {
    static TRACE_LANE: RefCell<Option<TraceLaneProducer>> = const { RefCell::new(None) };
    static TRACE_COUNTERS: RefCell<Option<Arc<LaneCounters>>> = const { RefCell::new(None) };
    static TRACE_CONTEXT: Cell<ContextId> = const { Cell::new(ContextId::PROCESS) };
    static IN_EMIT: Cell<bool> = const { Cell::new(false) };
}

/// Current scheduler context for the calling coordinator thread.
#[must_use]
pub fn current_context() -> ContextId {
    TRACE_CONTEXT.get()
}

/// Temporarily select a pre-registered context on the current controller lane.
#[must_use]
pub fn enter_context(context: ContextId) -> TraceContextGuard {
    let previous = TRACE_CONTEXT.replace(context);
    TraceContextGuard { previous }
}

/// Restores the controller context after serial/width-one work returns.
#[derive(Debug)]
pub struct TraceContextGuard {
    previous: ContextId,
}

impl Drop for TraceContextGuard {
    fn drop(&mut self) {
        TRACE_CONTEXT.set(self.previous);
    }
}

/// Decode a lowercase or uppercase hexadecimal SHA-256 without allocation.
#[must_use]
pub fn sha256_from_hex(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let mut output = [0u8; 32];
    for (slot, pair) in output.iter_mut().zip(hex.as_bytes().chunks_exact(2)) {
        let pair = std::str::from_utf8(pair).ok()?;
        *slot = u8::from_str_radix(pair, 16).ok()?;
    }
    Some(output)
}

/// Attempt to emit one fixed record on the current thread's bound lane.
///
/// The disabled/unbound path and every producer outcome return immediately.
/// No producer path formats, allocates, locks, blocks, or performs file I/O.
pub fn emit(event: TraceEvent) {
    IN_EMIT.with(|in_emit| {
        if in_emit.replace(true) {
            TRACE_COUNTERS.with_borrow(|counters| {
                if let Some(counters) = counters {
                    counters.reentrant.fetch_add(1, Ordering::Relaxed);
                }
            });
            return;
        }
        TRACE_LANE.with_borrow_mut(|slot| {
            if let Some(lane) = slot {
                lane.sequence = lane.sequence.saturating_add(1);
                let elapsed_ns = duration_ns(lane.started.elapsed());
                let record = TraceRecord {
                    lane_sequence: lane.sequence,
                    elapsed_ns,
                    event,
                };
                match lane.producer.push(record) {
                    Ok(()) => {}
                    Err(PushError::Full(_)) if lane.producer.is_abandoned() => {
                        lane.counters.disconnected.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(PushError::Full(_)) => {
                        lane.counters.full.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });
        in_emit.set(false);
    });
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// One non-cloneable producer which can be bound to exactly one thread.
#[derive(Debug)]
pub struct TraceLaneBinding {
    lane: Option<TraceLaneProducer>,
}

impl TraceLaneBinding {
    /// Move this lane into the calling thread's TLS.
    #[must_use]
    pub fn bind(mut self, context: ContextId) -> TraceBindingGuard {
        let lane = self.lane.take();
        if let Some(lane) = lane {
            let counters = Arc::clone(&lane.counters);
            TRACE_LANE.with_borrow_mut(|slot| {
                debug_assert!(slot.is_none());
                *slot = Some(lane);
            });
            TRACE_COUNTERS.with_borrow_mut(|slot| *slot = Some(counters));
            TRACE_CONTEXT.set(context);
            TraceBindingGuard {
                context,
                bound: true,
            }
        } else {
            TraceBindingGuard {
                context,
                bound: false,
            }
        }
    }
}

/// Concrete scheduler-owned driver lane collection.
#[derive(Debug, Default)]
pub struct TraceDriverLanes {
    lanes: Vec<TraceLaneBinding>,
    unbound_defects: Option<Arc<AtomicU64>>,
}

impl TraceDriverLanes {
    #[must_use]
    pub fn disabled() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Record scheduler call sites that could not bind their required lane.
    pub fn note_unbound(&self, count: usize) {
        if count == 0 {
            return;
        }
        if let Some(defects) = &self.unbound_defects {
            defects.fetch_add(u64::try_from(count).unwrap_or(u64::MAX), Ordering::Relaxed);
        }
    }

    /// Temporarily move at most `count` lanes into one scheduler invocation.
    #[doc(hidden)]
    pub fn take_for_scheduler(&mut self, count: usize) -> Vec<TraceLaneBinding> {
        let keep_from = self.lanes.len().saturating_sub(count);
        self.lanes.split_off(keep_from)
    }

    /// Restore quiescent driver lanes for a later scheduler invocation.
    #[doc(hidden)]
    pub fn restore_from_scheduler(&mut self, mut lanes: Vec<TraceLaneBinding>) {
        self.lanes.append(&mut lanes);
    }
}

impl IntoIterator for TraceDriverLanes {
    type Item = TraceLaneBinding;
    type IntoIter = std::vec::IntoIter<TraceLaneBinding>;

    fn into_iter(self) -> Self::IntoIter {
        self.lanes.into_iter()
    }
}

/// Bound producer lifetime on one coordinator thread.
#[derive(Debug)]
pub struct TraceBindingGuard {
    context: ContextId,
    bound: bool,
}

impl TraceBindingGuard {
    pub fn set_context(&mut self, context: ContextId) {
        self.context = context;
        TRACE_CONTEXT.set(context);
    }

    /// Remove the producer from TLS after a driver has stopped taking work.
    #[must_use]
    pub fn unbind(mut self) -> TraceLaneBinding {
        let lane = if self.bound {
            self.bound = false;
            TRACE_COUNTERS.with_borrow_mut(|slot| {
                slot.take();
            });
            TRACE_CONTEXT.set(ContextId::PROCESS);
            TRACE_LANE.with_borrow_mut(Option::take)
        } else {
            None
        };
        TraceLaneBinding { lane }
    }
}

impl Drop for TraceBindingGuard {
    fn drop(&mut self) {
        if self.bound {
            TRACE_LANE.with_borrow_mut(|slot| {
                slot.take();
            });
            TRACE_COUNTERS.with_borrow_mut(|slot| {
                slot.take();
            });
            TRACE_CONTEXT.set(ContextId::PROCESS);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum CommandOutcome {
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceDisposition {
    Complete,
    Lossy,
    Incomplete,
}

/// Final trace accounting, independent of command success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceFinishReport {
    pub command_outcome: CommandOutcome,
    pub disposition: TraceDisposition,
    pub path: Option<PathBuf>,
    pub attempted: u64,
    pub accepted: u64,
    pub written: u64,
    pub dropped_full: u64,
    pub dropped_disconnected: u64,
    pub dropped_reentrant: u64,
    pub failed_or_discarded: u64,
    pub truncated_lines: u64,
    pub byte_cap_reached: bool,
    pub sink_error: bool,
    pub finish_timed_out: bool,
    pub unbound_defects: u64,
}

impl TraceFinishReport {
    #[must_use]
    pub const fn disabled(command_outcome: CommandOutcome) -> Self {
        Self {
            command_outcome,
            disposition: TraceDisposition::Incomplete,
            path: None,
            attempted: 0,
            accepted: 0,
            written: 0,
            dropped_full: 0,
            dropped_disconnected: 0,
            dropped_reentrant: 0,
            failed_or_discarded: 0,
            truncated_lines: 0,
            byte_cap_reached: false,
            sink_error: false,
            finish_timed_out: false,
            unbound_defects: 0,
        }
    }
}

/// Successfully allocated producer bindings plus the finish controller.
#[derive(Debug)]
pub struct StartedTrace {
    pub controller: TraceLaneBinding,
    pub drivers: TraceDriverLanes,
    pub session: TraceSession,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceStartErrorKind {
    ResourceLimit,
    DirectoryCreate,
    FileCreate,
    WriterSpawn,
}

/// Trace startup failure. The caller should warn once and continue untraced.
#[derive(Debug)]
pub struct TraceStartError {
    kind: TraceStartErrorKind,
    source: Option<io::Error>,
}

impl TraceStartError {
    #[must_use]
    pub const fn kind(&self) -> TraceStartErrorKind {
        self.kind
    }
}

impl fmt::Display for TraceStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "runtime trace startup failed: {:?}", self.kind)?;
        if let Some(source) = &self.source {
            write!(formatter, ": {source}")?;
        }
        Ok(())
    }
}

impl std::error::Error for TraceStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|error| -> &(dyn std::error::Error + 'static) { error })
    }
}

struct WriterLane {
    id: usize,
    consumer: Consumer<TraceRecord>,
    counters: Arc<LaneCounters>,
    last: Option<TraceRecord>,
    written: u64,
    failed_or_discarded: u64,
}

/// Finish controller. Drop only abandons the writer; it never joins.
#[derive(Debug)]
pub struct TraceSession {
    terminal: Option<mpsc::SyncSender<TerminalRecord>>,
    abandon: Arc<AtomicBool>,
    completion: mpsc::Receiver<TraceFinishReport>,
    writer: Option<JoinHandle<()>>,
    open_path: PathBuf,
    started: Instant,
}

impl TraceSession {
    /// Allocate all lanes, securely create the open file, and start one writer.
    ///
    /// # Errors
    ///
    /// Returns an error retaining a typed [`TraceStartError`] cause. Scientific
    /// callers should warn once and continue with tracing disabled.
    pub fn start(config: TraceConfig) -> anyhow::Result<StartedTrace> {
        let lane_count = config.driver_lanes.checked_add(1).ok_or(TraceStartError {
            kind: TraceStartErrorKind::ResourceLimit,
            source: None,
        })?;
        let transport_bytes = lane_count
            .checked_mul(config.queue_capacity)
            .and_then(|records| records.checked_mul(std::mem::size_of::<TraceRecord>()))
            .ok_or(TraceStartError {
                kind: TraceStartErrorKind::ResourceLimit,
                source: None,
            })?;
        if lane_count > MAX_LANES
            || transport_bytes > MAX_TRANSPORT_BYTES
            || config.queue_capacity == 0
            || config.byte_cap <= TERMINAL_RESERVE_BYTES
        {
            return Err(TraceStartError {
                kind: TraceStartErrorKind::ResourceLimit,
                source: None,
            }
            .into());
        }

        create_private_directories(&config.output_dir).map_err(|source| TraceStartError {
            kind: TraceStartErrorKind::DirectoryCreate,
            source: Some(source),
        })?;
        let open_path = open_path(&config);
        let file = open_trace_file(&open_path).map_err(|source| TraceStartError {
            kind: TraceStartErrorKind::FileCreate,
            source: Some(source),
        })?;

        let started = Instant::now();
        let mut bindings = Vec::with_capacity(lane_count);
        let mut writer_lanes = Vec::with_capacity(lane_count);
        for id in 0..lane_count {
            let (producer, consumer) = RingBuffer::new(config.queue_capacity);
            let counters = Arc::new(LaneCounters::default());
            bindings.push(TraceLaneBinding {
                lane: Some(TraceLaneProducer {
                    producer,
                    counters: Arc::clone(&counters),
                    started,
                    sequence: 0,
                }),
            });
            writer_lanes.push(WriterLane {
                id,
                consumer,
                counters,
                last: None,
                written: 0,
                failed_or_discarded: 0,
            });
        }
        let mut binding_iter = bindings.into_iter();
        let controller = binding_iter.next().ok_or(TraceStartError {
            kind: TraceStartErrorKind::ResourceLimit,
            source: None,
        })?;
        let drivers = TraceDriverLanes {
            lanes: binding_iter.collect(),
            unbound_defects: Some(Arc::new(AtomicU64::new(0))),
        };
        let unbound_defects = drivers
            .unbound_defects
            .as_ref()
            .map_or_else(|| Arc::new(AtomicU64::new(0)), Arc::clone);

        let abandon = Arc::new(AtomicBool::new(false));
        let writer_abandon = Arc::clone(&abandon);
        let writer_open_path = open_path.clone();
        let (terminal_tx, terminal_rx) = mpsc::sync_channel(1);
        let (completion_tx, completion_rx) = mpsc::channel();
        let writer = thread::Builder::new()
            .name("nd-trace-writer".to_owned())
            .spawn(move || {
                let report = writer_main(
                    file,
                    &writer_open_path,
                    &config,
                    writer_lanes,
                    &terminal_rx,
                    &writer_abandon,
                    &unbound_defects,
                    started,
                );
                let _ = completion_tx.send(report);
            })
            .map_err(|source| TraceStartError {
                kind: TraceStartErrorKind::WriterSpawn,
                source: Some(source),
            })?;

        Ok(StartedTrace {
            controller,
            drivers,
            session: Self {
                terminal: Some(terminal_tx),
                abandon,
                completion: completion_rx,
                writer: Some(writer),
                open_path,
                started,
            },
        })
    }

    /// Live `.open.ndjson` path. Named so a matrix `.out` can point at it.
    #[must_use]
    pub fn open_path(&self) -> &Path {
        &self.open_path
    }

    /// Request bounded shutdown and return conservation accounting.
    #[must_use]
    pub fn finish(
        mut self,
        command_outcome: CommandOutcome,
        error_hash: Option<[u8; 32]>,
    ) -> TraceFinishReport {
        let terminal = TerminalRecord {
            elapsed_ns: duration_ns(self.started.elapsed()),
            event: TerminalEvent::new(command_outcome, error_hash),
        };
        let sent = self
            .terminal
            .take()
            .is_some_and(|sender| sender.send(terminal).is_ok());
        if !sent {
            self.abandon.store(true, Ordering::Release);
        }
        if let Ok(mut report) = self.completion.recv_timeout(FINISH_TIMEOUT) {
            if let Some(writer) = self.writer.take() {
                if writer.join().is_err() {
                    report.disposition = TraceDisposition::Incomplete;
                    report.sink_error = true;
                    report.path = Some(self.open_path.clone());
                }
            }
            report
        } else {
            self.abandon.store(true, Ordering::Release);
            self.writer.take();
            let mut report = TraceFinishReport::disabled(command_outcome);
            report.path = Some(self.open_path.clone());
            report.finish_timed_out = true;
            report
        }
    }
}

impl Drop for TraceSession {
    fn drop(&mut self) {
        self.abandon.store(true, Ordering::Release);
        self.writer.take();
    }
}

fn create_private_directories(path: &Path) -> io::Result<()> {
    if path.exists() {
        let metadata = fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "runtime trace output path is not a real directory",
            ));
        }
        return Ok(());
    }
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true).mode(0o700).create(path)
}

static FILE_NONCE: AtomicU64 = AtomicU64::new(0);

fn open_path(config: &TraceConfig) -> PathBuf {
    #[cfg(test)]
    if let Some(stem) = &config.test_file_stem {
        return config.output_dir.join(format!("{stem}.open.ndjson"));
    }
    let nonce = FILE_NONCE.fetch_add(1, Ordering::Relaxed);
    let epoch_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let mut identity = String::new();
    for value in [
        config.scheduler.job.as_deref(),
        config.scheduler.array_task.as_deref(),
        config.scheduler.step.as_deref(),
        config.scheduler.rank.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        identity.push('-');
        identity.push_str(&sanitize_component(value));
    }
    let stem = format!(
        "{}{}-pid{}-{epoch_ns}-{nonce}",
        config.command_role.file_label(),
        identity,
        std::process::id()
    );
    config.output_dir.join(format!("{stem}.open.ndjson"))
}

fn sanitize_component(input: &str) -> String {
    input
        .chars()
        .take(64)
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn open_trace_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
}

#[derive(Serialize)]
struct HeaderLine<'a> {
    schema: &'static str,
    class: &'static str,
    command_role: CommandRole,
    pid: u32,
    binary_sha256: [u8; 32],
    revision_sha256: Option<[u8; 32]>,
    scheduler: &'a SchedulerIdentity,
    queue_capacity: usize,
    lane_count: usize,
    heartbeat_interval_ns: u64,
    byte_cap: usize,
    contexts: &'a [TraceContext],
}

#[derive(Serialize)]
struct EventLine<'a, T> {
    schema: &'static str,
    class: &'static str,
    writer_sequence: u64,
    lane_id: usize,
    lane_sequence: u64,
    elapsed_ns: u64,
    context: ContextId,
    event_name: &'static str,
    severity: &'static str,
    payload: &'a T,
}

#[derive(Serialize)]
struct HeartbeatLane {
    lane_id: usize,
    last_phase: Option<&'static str>,
    record_age_ns: Option<u64>,
    uncertain_after_loss: bool,
}

#[derive(Serialize)]
struct HeartbeatLine<'a> {
    schema: &'static str,
    class: &'static str,
    elapsed_ns: u64,
    lossy: bool,
    lanes: &'a [HeartbeatLane],
}

#[derive(Serialize)]
struct FooterLaneLine {
    schema: &'static str,
    class: &'static str,
    lane_id: usize,
    attempted: u64,
    accepted: u64,
    written: u64,
    dropped_full: u64,
    dropped_disconnected: u64,
    dropped_reentrant: u64,
    failed_or_discarded: u64,
}

#[derive(Serialize)]
struct FooterLine {
    schema: &'static str,
    class: &'static str,
    command_outcome: CommandOutcome,
    disposition: TraceDisposition,
    attempted: u64,
    accepted: u64,
    written: u64,
    dropped_full: u64,
    dropped_disconnected: u64,
    dropped_reentrant: u64,
    failed_or_discarded: u64,
    truncated_lines: u64,
    byte_cap_reached: bool,
    sink_error: bool,
    unbound_defects: u64,
}

struct Sink {
    file: Option<File>,
    buffer: Vec<u8>,
    bytes_written: usize,
    byte_cap: usize,
    byte_cap_reached: bool,
    failed: bool,
    truncated_lines: u64,
    #[cfg(test)]
    test_fault: TestFault,
    #[cfg(test)]
    successful_writes: usize,
}

impl Sink {
    fn new(file: File, byte_cap: usize, config: &TraceConfig) -> Self {
        #[cfg(not(test))]
        let _ = config;
        Self {
            file: Some(file),
            buffer: Vec::with_capacity(MAX_EVENT_LINE_BYTES),
            bytes_written: 0,
            byte_cap,
            byte_cap_reached: false,
            failed: false,
            truncated_lines: 0,
            #[cfg(test)]
            test_fault: config.test_fault,
            #[cfg(test)]
            successful_writes: 0,
        }
    }

    fn write<T: Serialize>(&mut self, value: &T, event_line: bool, terminal: bool) -> bool {
        if self.failed {
            return false;
        }
        self.buffer.clear();
        if serde_json::to_writer(&mut self.buffer, value).is_err() {
            self.failed = true;
            self.file.take();
            return false;
        }
        self.buffer.push(b'\n');
        if event_line && self.buffer.len() > MAX_EVENT_LINE_BYTES {
            self.truncated_lines = self.truncated_lines.saturating_add(1);
            return false;
        }
        let allowed = if terminal {
            self.byte_cap
        } else {
            self.byte_cap.saturating_sub(TERMINAL_RESERVE_BYTES)
        };
        if self.bytes_written.saturating_add(self.buffer.len()) > allowed {
            self.byte_cap_reached = true;
            return false;
        }
        let Some(file) = self.file.as_mut() else {
            self.failed = true;
            return false;
        };
        #[cfg(test)]
        match self.test_fault {
            TestFault::WriteAfter(limit) if self.successful_writes >= limit => {
                self.failed = true;
                self.file.take();
                return false;
            }
            TestFault::PartialWriteAfter(limit) if self.successful_writes >= limit => {
                let partial = self.buffer.len().saturating_div(2).max(1);
                if let Some(bytes) = self.buffer.get(..partial) {
                    if file.write(bytes).is_ok() {
                        self.bytes_written = self.bytes_written.saturating_add(partial);
                    }
                }
                self.failed = true;
                self.file.take();
                return false;
            }
            _ => {}
        }
        if file.write_all(&self.buffer).is_err() {
            self.failed = true;
            self.file.take();
            return false;
        }
        self.bytes_written = self.bytes_written.saturating_add(self.buffer.len());
        #[cfg(test)]
        {
            self.successful_writes = self.successful_writes.saturating_add(1);
        }
        true
    }

    fn finish_io(&mut self) -> bool {
        let Some(file) = self.file.as_mut() else {
            return false;
        };
        #[cfg(test)]
        if matches!(self.test_fault, TestFault::Flush) {
            self.failed = true;
            self.file.take();
            return false;
        }
        if file.flush().is_err() {
            self.failed = true;
            self.file.take();
            return false;
        }
        #[cfg(test)]
        if matches!(self.test_fault, TestFault::Sync) {
            self.failed = true;
            self.file.take();
            return false;
        }
        if file.sync_data().is_err() {
            self.failed = true;
            self.file.take();
            return false;
        }
        true
    }

    /// Push drained bytes to the kernel without fsync. Producers never wait
    /// here; this is the writer thread making the still-open file readable.
    fn publish_live(&mut self) {
        if self.failed {
            return;
        }
        let Some(file) = self.file.as_mut() else {
            return;
        };
        if file.flush().is_err() {
            self.failed = true;
            self.file.take();
        }
    }
}

/// Compact live line for Slurm `.out` and test harnesses. Writer thread only;
/// producers never wait on stderr. Skip crate-unit traces (`echo_live = false`)
/// and the emit-bench role so cargo tests stay silent and the bench machine
/// result stays on stdout with empty stderr.
fn echo_elapsed_secs(elapsed_ns: u64) -> f64 {
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "elapsed_ns is a diagnostic second display; the NDJSON file keeps the integer"
    )]
    {
        elapsed_ns as f64 / 1_000_000_000.0
    }
}

fn write_live_event_line(
    stderr: &mut impl io::Write,
    secs: f64,
    heartbeat_age_s: Option<f64>,
    event: &TraceEvent,
) -> io::Result<()> {
    match heartbeat_age_s {
        Some(age) => write!(stderr, "nd-trace {secs:.3}s heartbeat age={age:.1}s ")?,
        None => write!(stderr, "nd-trace {secs:.3}s ")?,
    }
    write_live_event_body(stderr, event)
}

fn write_live_event_body(stderr: &mut impl io::Write, event: &TraceEvent) -> io::Result<()> {
    match *event {
        TraceEvent::ObjectiveStarted {
            generation,
            objective_ordinal,
            event_count,
            active_designs,
            candidate_count,
            ..
        } => writeln!(
            stderr,
            "objective.started gen={generation} ord={objective_ordinal} events={event_count} designs={active_designs}/{candidate_count}"
        ),
        TraceEvent::ObjectiveFinished {
            generation,
            objective_ordinal,
            physics_trials,
            returned_designs,
            status,
            ..
        } => writeln!(
            stderr,
            "objective.finished gen={generation} ord={objective_ordinal} trials={physics_trials} designs={returned_designs} status={status}"
        ),
        TraceEvent::AdaptiveStageStarted {
            generation,
            event_offset,
            event_count,
            active_designs,
            ..
        } => writeln!(
            stderr,
            "adaptive_stage.started gen={generation} events={event_offset}+{event_count} designs={active_designs}"
        ),
        TraceEvent::AdaptiveStageFinished {
            generation,
            event_offset,
            event_count,
            physics_trials,
            candidate_count,
            status,
            ..
        } => writeln!(
            stderr,
            "adaptive_stage.finished gen={generation} events={event_offset}+{event_count} trials={physics_trials} candidates={candidate_count} status={status}"
        ),
        TraceEvent::OptimizerStarted { optimizer_id, .. } => {
            writeln!(stderr, "optimizer.started id={optimizer_id}")
        }
        TraceEvent::OptimizerFinished { status, .. } => {
            writeln!(stderr, "optimizer.finished status={status}")
        }
        TraceEvent::CellStarted { cell_index, .. } => {
            writeln!(stderr, "cell.started index={cell_index}")
        }
        TraceEvent::MatrixUnitSelected { unit_index, .. } => {
            writeln!(stderr, "matrix.unit_selected index={unit_index}")
        }
        TraceEvent::CohortStarted { cohort_index, .. } => {
            writeln!(stderr, "cohort.started index={cohort_index}")
        }
        TraceEvent::K3BarrierCommitted {
            barrier_index,
            checkpoint_hash,
            ..
        } => {
            write!(stderr, "k3.barrier.committed barrier={barrier_index} ")?;
            write_hash8(stderr, &checkpoint_hash)?;
            writeln!(stderr)
        }
        TraceEvent::ShardRequestStarted {
            sequence,
            design_count,
            ..
        } => writeln!(
            stderr,
            "shard.request.started seq={sequence} designs={design_count}"
        ),
        TraceEvent::ShardRequestFinished {
            sequence, status, ..
        } => writeln!(
            stderr,
            "shard.request.finished seq={sequence} status={status}"
        ),
        TraceEvent::Notice { code, detail, .. } => {
            writeln!(stderr, "notice code={code} detail={detail}")
        }
        TraceEvent::CapRelaxation { count, .. } => {
            writeln!(stderr, "cap_relaxation count={count}")
        }
        TraceEvent::ProfileSummary {
            profile_kind,
            calls,
            elapsed_ns,
            ..
        } => writeln!(
            stderr,
            "profile.summary kind={profile_kind} calls={calls} elapsed_ns={elapsed_ns}"
        ),
        TraceEvent::SchedulerPoolReady { width, .. } => {
            writeln!(stderr, "scheduler.pool.ready width={width}")
        }
        TraceEvent::CellFinished {
            status,
            recovered_shards,
            ..
        } => writeln!(
            stderr,
            "cell.finished status={status} recovered={recovered_shards}"
        ),
        TraceEvent::CohortFinished { status, .. } => {
            writeln!(stderr, "cohort.finished status={status}")
        }
        TraceEvent::ShardSessionReady { worker_count, .. } => {
            writeln!(stderr, "shard.session.ready workers={worker_count}")
        }
        TraceEvent::ShardWorkerJoined { worker, .. } => {
            writeln!(stderr, "shard.worker.joined worker={worker}")
        }
        TraceEvent::ShardBatchRecovered {
            sequence,
            recovered_shards,
            ..
        } => writeln!(
            stderr,
            "shard.batch.recovered seq={sequence} recovered={recovered_shards}"
        ),
        TraceEvent::ShardWorkerFinished { batches, status, .. } => writeln!(
            stderr,
            "shard.worker.finished batches={batches} status={status}"
        ),
        TraceEvent::AuthorityValidated {
            authority_hash, ..
        } => {
            write!(stderr, "authority.validated ")?;
            write_hash8(stderr, &authority_hash)?;
            writeln!(stderr)
        }
        TraceEvent::ReceiptSealed { receipt_hash, .. } => {
            write!(stderr, "receipt.sealed ")?;
            write_hash8(stderr, &receipt_hash)?;
            writeln!(stderr)
        }
        TraceEvent::ArtifactPublished {
            publication_receipt_hash,
            ..
        } => {
            write!(stderr, "artifact.published ")?;
            write_hash8(stderr, &publication_receipt_hash)?;
            writeln!(stderr)
        }
        TraceEvent::ProcessStarted { .. } => writeln!(stderr, "process.started"),
    }
}

fn write_hash8(stderr: &mut impl io::Write, bytes: &[u8; 32]) -> io::Result<()> {
    for byte in bytes.get(..8).unwrap_or(&[]) {
        write!(stderr, "{byte:02x}")?;
    }
    Ok(())
}

fn write_live_terminal_line(
    stderr: &mut impl io::Write,
    secs: f64,
    event: TerminalEvent,
) -> io::Result<()> {
    write!(stderr, "nd-trace {secs:.3}s ")?;
    match event {
        TerminalEvent::ProcessFinished { status, .. } => {
            writeln!(stderr, "process.finished status={status}")
        }
        TerminalEvent::ProcessFailed {
            status, error_hash, ..
        } => {
            write!(stderr, "process.failed status={status}")?;
            if let Some(hash) = error_hash.as_ref() {
                write!(stderr, " ")?;
                write_hash8(stderr, &hash.0)?;
            }
            writeln!(stderr)
        }
    }
}

fn echo_live_event(echo_live: bool, role: CommandRole, elapsed_ns: u64, event: &TraceEvent) {
    if !echo_live || role == CommandRole::EmitBench {
        return;
    }
    let secs = echo_elapsed_secs(elapsed_ns);
    let stderr = io::stderr();
    let mut stderr = stderr.lock();
    let _ = write_live_event_line(&mut stderr, secs, None, event);
}

fn echo_live_terminal(echo_live: bool, role: CommandRole, elapsed_ns: u64, event: TerminalEvent) {
    if !echo_live || role == CommandRole::EmitBench {
        return;
    }
    let secs = echo_elapsed_secs(elapsed_ns);
    let stderr = io::stderr();
    let mut stderr = stderr.lock();
    let _ = write_live_terminal_line(&mut stderr, secs, event);
}

fn echo_live_heartbeat(
    echo_live: bool,
    role: CommandRole,
    elapsed_ns: u64,
    last: Option<&TraceRecord>,
) {
    if !echo_live || role == CommandRole::EmitBench {
        return;
    }
    let secs = echo_elapsed_secs(elapsed_ns);
    let stderr = io::stderr();
    let mut stderr = stderr.lock();
    if let Some(record) = last {
        let age = echo_elapsed_secs(elapsed_ns.saturating_sub(record.elapsed_ns));
        let _ = write_live_event_line(&mut stderr, secs, Some(age), &record.event);
    } else {
        let _ = writeln!(stderr, "nd-trace {secs:.3}s heartbeat idle");
    }
}

fn writer_main(
    file: File,
    open_path: &Path,
    config: &TraceConfig,
    mut lanes: Vec<WriterLane>,
    terminal_rx: &mpsc::Receiver<TerminalRecord>,
    abandon: &AtomicBool,
    unbound_defects: &AtomicU64,
    started: Instant,
) -> TraceFinishReport {
    #[cfg(test)]
    if !config.test_writer_delay.is_zero() {
        thread::sleep(config.test_writer_delay);
    }
    let mut sink = Sink::new(file, config.byte_cap, config);
    let header = HeaderLine {
        schema: TRACE_SCHEMA,
        class: "header",
        command_role: config.command_role,
        pid: std::process::id(),
        binary_sha256: config.binary_sha256,
        revision_sha256: config.revision_sha256,
        scheduler: &config.scheduler,
        queue_capacity: config.queue_capacity,
        lane_count: lanes.len(),
        heartbeat_interval_ns: duration_ns(config.heartbeat_interval),
        byte_cap: config.byte_cap,
        contexts: &config.contexts,
    };
    sink.write(&header, false, false);
    sink.publish_live();

    let mut writer_sequence = 0u64;
    let mut heartbeat_states = Vec::with_capacity(lanes.len());
    let mut next_heartbeat = Instant::now()
        .checked_add(config.heartbeat_interval)
        .unwrap_or_else(Instant::now);
    let mut terminal = None;
    loop {
        if abandon.load(Ordering::Acquire) {
            break;
        }
        let mut drained = false;
        for lane in &mut lanes {
            for _ in 0..DRAIN_BURST {
                let record = match lane.consumer.pop() {
                    Ok(record) => record,
                    Err(PopError::Empty) => break,
                };
                drained = true;
                writer_sequence = writer_sequence.saturating_add(1);
                let line = EventLine {
                    schema: TRACE_SCHEMA,
                    class: "event",
                    writer_sequence,
                    lane_id: lane.id,
                    lane_sequence: record.lane_sequence,
                    elapsed_ns: record.elapsed_ns,
                    context: record.event.context(),
                    event_name: record.event.name(),
                    severity: "info",
                    payload: &record.event,
                };
                if sink.write(&line, true, false) {
                    lane.written = lane.written.saturating_add(1);
                    echo_live_event(
                        config.echo_live,
                        config.command_role,
                        record.elapsed_ns,
                        &record.event,
                    );
                } else {
                    lane.failed_or_discarded = lane.failed_or_discarded.saturating_add(1);
                }
                lane.last = Some(record);
            }
        }

        let now = Instant::now();
        if drained {
            sink.publish_live();
        }
        if terminal.is_none() {
            match terminal_rx.try_recv() {
                Ok(record) => terminal = Some(record),
                Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => {}
            }
        }
        if now >= next_heartbeat && terminal.is_none() {
            let elapsed = started.elapsed();
            write_heartbeat(&mut sink, &lanes, elapsed, &mut heartbeat_states);
            let last = lanes
                .iter()
                .filter_map(|lane| lane.last.as_ref())
                .max_by_key(|record| record.elapsed_ns);
            echo_live_heartbeat(
                config.echo_live,
                config.command_role,
                duration_ns(elapsed),
                last,
            );
            sink.publish_live();
            next_heartbeat = now.checked_add(config.heartbeat_interval).unwrap_or(now);
        }
        if terminal.is_some() && lanes.iter().all(|lane| lane.consumer.slots() == 0) {
            break;
        }
        if !drained {
            thread::sleep(IDLE_POLL);
        }
    }

    if abandon.load(Ordering::Acquire) {
        let mut report =
            TraceFinishReport::disabled(terminal.map_or(CommandOutcome::Failed, |record| {
                record.event.command_outcome()
            }));
        report.path = Some(open_path.to_path_buf());
        report.sink_error = true;
        return report;
    }

    let Some(terminal) = terminal else {
        let mut report = TraceFinishReport::disabled(CommandOutcome::Failed);
        report.path = Some(open_path.to_path_buf());
        report.sink_error = true;
        return report;
    };

    finish_writer(
        sink,
        &lanes,
        open_path,
        terminal,
        writer_sequence,
        config.echo_live,
        config.command_role,
        unbound_defects.load(Ordering::Relaxed),
    )
}

fn write_heartbeat(
    sink: &mut Sink,
    lanes: &[WriterLane],
    elapsed: Duration,
    states: &mut Vec<HeartbeatLane>,
) {
    let elapsed_ns = duration_ns(elapsed);
    states.clear();
    for lane in lanes {
        let dropped = lane
            .counters
            .full
            .load(Ordering::Relaxed)
            .saturating_add(lane.counters.disconnected.load(Ordering::Relaxed))
            .saturating_add(lane.counters.reentrant.load(Ordering::Relaxed));
        states.push(HeartbeatLane {
            lane_id: lane.id,
            last_phase: lane.last.map(|record| record.event.name()),
            record_age_ns: lane
                .last
                .map(|record| elapsed_ns.saturating_sub(record.elapsed_ns)),
            uncertain_after_loss: dropped != 0,
        });
    }
    let line = HeartbeatLine {
        schema: TRACE_SCHEMA,
        class: "heartbeat",
        elapsed_ns,
        lossy: states.iter().any(|state| state.uncertain_after_loss),
        lanes: states,
    };
    sink.write(&line, false, false);
}

fn finish_writer(
    mut sink: Sink,
    lanes: &[WriterLane],
    open_path: &Path,
    terminal: TerminalRecord,
    mut writer_sequence: u64,
    echo_live: bool,
    command_role: CommandRole,
    unbound_defects: u64,
) -> TraceFinishReport {
    let controller_attempted = lanes.first().map_or(0, |lane| {
        lane.written
            .saturating_add(lane.failed_or_discarded)
            .saturating_add(lane.counters.full.load(Ordering::Relaxed))
            .saturating_add(lane.counters.disconnected.load(Ordering::Relaxed))
            .saturating_add(lane.counters.reentrant.load(Ordering::Relaxed))
    });
    writer_sequence = writer_sequence.saturating_add(1);
    let terminal_line = EventLine {
        schema: TRACE_SCHEMA,
        class: "event",
        writer_sequence,
        lane_id: 0,
        lane_sequence: controller_attempted.saturating_add(1),
        elapsed_ns: terminal.elapsed_ns,
        context: ContextId::PROCESS,
        event_name: terminal.event.name(),
        severity: terminal.event.severity(),
        payload: &terminal.event,
    };
    let terminal_written = sink.write(&terminal_line, true, true);
    if terminal_written {
        echo_live_terminal(echo_live, command_role, terminal.elapsed_ns, terminal.event);
    }

    let mut report = TraceFinishReport::disabled(terminal.event.command_outcome());
    report.path = Some(open_path.to_path_buf());
    report.unbound_defects = unbound_defects;
    for lane in lanes {
        let dropped_full = lane.counters.full.load(Ordering::Relaxed);
        let dropped_disconnected = lane.counters.disconnected.load(Ordering::Relaxed);
        let dropped_reentrant = lane.counters.reentrant.load(Ordering::Relaxed);
        let terminal_accepted = u64::from(lane.id == 0);
        let terminal_written_count = u64::from(lane.id == 0 && terminal_written);
        let terminal_failed = terminal_accepted.saturating_sub(terminal_written_count);
        let accepted = lane
            .written
            .saturating_add(lane.failed_or_discarded)
            .saturating_add(terminal_accepted);
        let attempted = accepted
            .saturating_add(dropped_full)
            .saturating_add(dropped_disconnected)
            .saturating_add(dropped_reentrant);
        let line = FooterLaneLine {
            schema: TRACE_SCHEMA,
            class: "footer_lane",
            lane_id: lane.id,
            attempted,
            accepted,
            written: lane.written.saturating_add(terminal_written_count),
            dropped_full,
            dropped_disconnected,
            dropped_reentrant,
            failed_or_discarded: lane.failed_or_discarded.saturating_add(terminal_failed),
        };
        sink.write(&line, false, true);
        report.attempted = report.attempted.saturating_add(attempted);
        report.accepted = report.accepted.saturating_add(accepted);
        report.written = report
            .written
            .saturating_add(lane.written)
            .saturating_add(terminal_written_count);
        report.dropped_full = report.dropped_full.saturating_add(dropped_full);
        report.dropped_disconnected = report
            .dropped_disconnected
            .saturating_add(dropped_disconnected);
        report.dropped_reentrant = report.dropped_reentrant.saturating_add(dropped_reentrant);
        report.failed_or_discarded = report
            .failed_or_discarded
            .saturating_add(lane.failed_or_discarded)
            .saturating_add(terminal_failed);
    }
    report.truncated_lines = sink.truncated_lines;
    report.byte_cap_reached = sink.byte_cap_reached;
    report.sink_error = sink.failed;
    report.disposition = if report.sink_error || report.unbound_defects != 0 || !terminal_written {
        TraceDisposition::Incomplete
    } else if report.dropped_full != 0
        || report.dropped_disconnected != 0
        || report.dropped_reentrant != 0
        || report.failed_or_discarded != 0
        || report.truncated_lines != 0
        || report.byte_cap_reached
    {
        TraceDisposition::Lossy
    } else {
        TraceDisposition::Complete
    };
    let footer = FooterLine {
        schema: TRACE_SCHEMA,
        class: "footer",
        command_outcome: report.command_outcome,
        disposition: report.disposition,
        attempted: report.attempted,
        accepted: report.accepted,
        written: report.written,
        dropped_full: report.dropped_full,
        dropped_disconnected: report.dropped_disconnected,
        dropped_reentrant: report.dropped_reentrant,
        failed_or_discarded: report.failed_or_discarded,
        truncated_lines: report.truncated_lines,
        byte_cap_reached: report.byte_cap_reached,
        sink_error: report.sink_error,
        unbound_defects: report.unbound_defects,
    };
    if !sink.write(&footer, false, true) || !sink.finish_io() {
        report.disposition = TraceDisposition::Incomplete;
        report.sink_error = true;
        return report;
    }

    let suffix = match report.disposition {
        TraceDisposition::Complete => "complete.ndjson",
        TraceDisposition::Lossy => "lossy.ndjson",
        TraceDisposition::Incomplete => return report,
    };
    let final_path = replace_open_suffix(open_path, suffix);
    if rename_no_replace(open_path, &final_path).is_err() {
        report.disposition = TraceDisposition::Incomplete;
        report.sink_error = true;
        return report;
    }
    report.path = Some(final_path);
    report
}

fn rename_no_replace(source: &Path, target: &Path) -> io::Result<()> {
    let source = std::ffi::CString::new(source.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "trace source contains NUL"))?;
    let target = std::ffi::CString::new(target.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "trace target contains NUL"))?;
    #[cfg(target_os = "linux")]
    // SAFETY: both C strings are NUL-terminated and remain alive for the call.
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            target.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    #[cfg(target_os = "macos")]
    // SAFETY: both C strings are NUL-terminated and remain alive for the call.
    let result = unsafe { libc::renamex_np(source.as_ptr(), target.as_ptr(), libc::RENAME_EXCL) };
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let result = -1;
    if result == 0 {
        return Ok(());
    }
    let rename_error = io::Error::last_os_error();
    // NFS mounts below 4.2 reject RENAME_NOREPLACE with EINVAL (observed live
    // on TinkerCliffs /home). For regular trace files the portable no-replace
    // sequence is a hard link (atomically fails with EEXIST if the target
    // exists) followed by removing the source name.
    #[cfg(target_os = "linux")]
    {
        let errno = rename_error.raw_os_error();
        if errno == Some(libc::EINVAL) || errno == Some(libc::ENOSYS) {
            // SAFETY: both C strings are NUL-terminated and alive for the call.
            let linked = unsafe {
                libc::linkat(
                    libc::AT_FDCWD,
                    source.as_ptr(),
                    libc::AT_FDCWD,
                    target.as_ptr(),
                    0,
                )
            };
            if linked != 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: NUL-terminated C string alive for the call.
            let unlinked = unsafe { libc::unlinkat(libc::AT_FDCWD, source.as_ptr(), 0) };
            if unlinked != 0 {
                return Err(io::Error::last_os_error());
            }
            return Ok(());
        }
    }
    Err(rename_error)
}

fn replace_open_suffix(path: &Path, suffix: &str) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("runtime-trace.open.ndjson");
    let stem = name.strip_suffix("open.ndjson").unwrap_or("runtime-trace.");
    path.with_file_name(format!("{stem}{suffix}"))
}

#[cfg(test)]
mod tests;
