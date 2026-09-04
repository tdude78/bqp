use super::*;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::os::unix::io::AsRawFd as _;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

struct CountingAllocator;

thread_local! {
    static COUNT_THIS_THREAD: Cell<bool> = const { Cell::new(false) };
}

static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

fn trace_start_error(result: anyhow::Result<StartedTrace>) -> TraceStartError {
    result
        .expect_err("trace startup must fail")
        .downcast::<TraceStartError>()
        .expect("trace startup must retain its typed cause")
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT_THIS_THREAD.with(|enabled| {
            if enabled.get() {
                ALLOCATION_COUNT.fetch_add(1, AtomicOrdering::Relaxed);
            }
        });
        // SAFETY: Delegates the exact allocation request to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` and `layout` are the pair supplied by the caller.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        COUNT_THIS_THREAD.with(|enabled| {
            if enabled.get() {
                ALLOCATION_COUNT.fetch_add(1, AtomicOrdering::Relaxed);
            }
        });
        // SAFETY: Delegates the exact reallocation request to the system allocator.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static TEST_ALLOCATOR: CountingAllocator = CountingAllocator;

fn count_current_thread_allocations(action: impl FnOnce()) -> usize {
    COUNT_THIS_THREAD.with(|enabled| enabled.set(false));
    ALLOCATION_COUNT.store(0, AtomicOrdering::Relaxed);
    COUNT_THIS_THREAD.with(|enabled| enabled.set(true));
    action();
    COUNT_THIS_THREAD.with(|enabled| enabled.set(false));
    ALLOCATION_COUNT.load(AtomicOrdering::Relaxed)
}

#[test]
fn command_role_has_no_unused_probe_variant() {
    let source = include_str!("lib.rs");
    let role = source
        .split_once("pub enum CommandRole")
        .and_then(|(_, tail)| tail.split_once("impl CommandRole"))
        .map(|(body, _)| body)
        .expect("CommandRole source window");
    assert!(
        !role.contains("PartAProbeRun") && !role.contains("part-a-probe-run"),
        "CommandRole::PartAProbeRun has no production constructor; drop the unused variant: {role}"
    );
    assert!(
        role.contains("PartACanary")
            && role.contains("ShardWorker")
            && role.contains("EmitBench")
            && role.contains("Matrix"),
        "live command roles must remain: {role}"
    );
}

#[test]
fn resource_contract_is_fixed_and_non_vacuous() {
    assert_eq!(DEFAULT_QUEUE_CAPACITY, 256);
    assert_eq!(MAX_LANES, 512);
    assert_eq!(MAX_TRANSPORT_BYTES, 32 * 1024 * 1024);
    assert_eq!(MAX_EVENT_LINE_BYTES, 4 * 1024);
    assert_eq!(MAX_TRACE_BYTES, 64 * 1024 * 1024);
    assert!(std::mem::size_of::<TraceEvent>() > 0);
    assert!(std::mem::size_of::<TraceEvent>() <= 192);
    assert!(
        DEFAULT_QUEUE_CAPACITY * MAX_LANES * std::mem::size_of::<TraceRecord>()
            <= MAX_TRANSPORT_BYTES
    );
}

#[test]
fn start_failures_use_anyhow_and_keep_typed_causes() {
    fn assert_anyhow_result(_: anyhow::Result<StartedTrace>) {}

    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), MAX_LANES, 1, MAX_TRACE_BYTES);
    assert_anyhow_result(TraceSession::start(config));
    let error = trace_start_error(TraceSession::start(TraceConfig::test(
        temp.path(),
        MAX_LANES,
        1,
        MAX_TRACE_BYTES,
    )));
    assert_eq!(error.kind(), TraceStartErrorKind::ResourceLimit);
}

#[test]
fn successful_emit_uses_writer_derived_accounting() {
    let source = include_str!("lib.rs");
    let emit = source
        .split_once("pub fn emit(event: TraceEvent)")
        .and_then(|(_, tail)| tail.split_once("fn duration_ns"))
        .map(|(body, _)| body)
        .expect("emit source window");
    assert!(
        !emit.contains("counters.attempted") && !emit.contains("counters.accepted"),
        "successful emit accounting must be derived by the writer, not updated atomically"
    );
    assert!(
        !emit.contains("Mutex")
            && !emit.contains("RwLock")
            && !emit.contains("lock(")
            && !emit.contains("BufWriter")
            && !emit.contains("write_all")
            && !emit.contains("File::")
            && !emit.contains("fs::"),
        "emit must stay lock-free and off the filesystem: {emit}"
    );
}

/// The writer owns the file. Producers never wait for it. Live readers of the
/// still-open `.open.ndjson` file must see drained events without waiting for
/// `finish` or an fsync.
#[test]
fn writer_publishes_drained_events_before_finish() {
    let source = include_str!("lib.rs");
    let writer = source
        .split_once("fn writer_main(")
        .and_then(|(_, tail)| tail.split_once("fn write_heartbeat"))
        .map(|(body, _)| body)
        .expect("writer_main source window");
    assert!(
        writer.contains("publish_live"),
        "writer_main must publish drained events to the open file before idle-sleep or finish"
    );
    assert!(
        writer.contains("echo_live_event"),
        "writer_main must echo drained events with payload fields for matrix job logs"
    );
    assert!(
        writer.contains("config.echo_live"),
        "writer_main must honour TraceConfig.echo_live so crate tests stay silent"
    );
    assert!(
        writer.contains("echo_live_heartbeat") && writer.contains("lane.last"),
        "heartbeat stderr echo must replay the last event payload (gen/stage), not only its class name"
    );
}

#[test]
fn live_event_echo_names_generation_and_stage() {
    let source = include_str!("lib.rs");
    let echo = source
        .split_once("fn write_live_event_line(")
        .and_then(|(_, tail)| tail.split_once("fn echo_live_event("))
        .map(|(body, _)| body)
        .expect("write_live_event_line source window");
    assert!(
        echo.contains("gen={generation}")
            && echo.contains("objective.started")
            && echo.contains("adaptive_stage.started"),
        "90h Exact36 .out must name generation and adaptive stage, not only event class: {echo}"
    );
    assert!(
        echo.contains("notice code=")
            && echo.contains("cap_relaxation count=")
            && echo.contains("profile.summary"),
        "writer-side live echo must name notice/cap/profile scalars without growing TraceEvent: {echo}"
    );
    assert!(
        echo.contains("scheduler.pool.ready width=")
            && echo.contains("shard.session.ready workers=")
            && echo.contains("cell.finished status=")
            && echo.contains("shard.worker.finished batches="),
        "writer-side live echo must name remaining scalar waits without growing TraceEvent: {echo}"
    );
    assert!(
        echo.contains("receipt.sealed")
            && echo.contains("authority.validated")
            && echo.contains("{byte:02x}"),
        "writer-side live echo must print short hashes without allocating a String: {echo}"
    );
    let terminal = source
        .split_once("fn write_live_terminal_line(")
        .and_then(|(_, tail)| tail.split_once("fn echo_live_event("))
        .map(|(body, _)| body)
        .expect("private terminal live-echo source window");
    assert!(
        terminal.contains("process.finished status=")
            && terminal.contains("process.failed status=")
            && terminal.contains("write_hash8"),
        "private terminal echo must name outcome and failure hash: {terminal}"
    );
    let producer = source
        .split_once("pub enum TraceEvent")
        .and_then(|(_, tail)| tail.split_once("impl TraceEvent"))
        .map(|(body, _)| body)
        .expect("TraceEvent source window");
    assert!(
        !producer.contains("ProcessFinished") && !producer.contains("ProcessFailed"),
        "terminal outcomes must not share the lossy producer queue: {producer}"
    );
    let k3 = echo
        .split_once("TraceEvent::K3BarrierCommitted")
        .and_then(|(_, tail)| tail.split_once("TraceEvent::ShardRequestStarted"))
        .map(|(body, _)| body)
        .expect("K3BarrierCommitted live-echo arm");
    assert!(
        k3.contains("write_hash8"),
        "k3.barrier.committed must print the checkpoint hash prefix: {k3}"
    );
}

#[test]
fn crate_test_traces_do_not_echo_to_stderr() {
    let source = include_str!("lib.rs");
    assert!(
        !source.contains("fn echo_live_progress("),
        "echo_live_progress is a leftover shim; idle heartbeat must writeln directly"
    );
    let echo = source
        .split_once("fn echo_live_event(")
        .and_then(|(_, tail)| tail.split_once("fn writer_main("))
        .map(|(body, _)| body)
        .expect("echo_live_event source window");
    assert!(
        echo.contains("!echo_live") && echo.contains("EmitBench"),
        "live stderr echo must be skippable for crate tests and emit-bench: {echo}"
    );
    let test_ctor = source
        .split_once("fn test(")
        .and_then(|(_, tail)| tail.split_once("fn with_test_writer_delay"))
        .map(|(body, _)| body)
        .expect("TraceConfig::test source window");
    assert!(
        test_ctor.contains("echo_live: false"),
        "crate-test traces must disable live stderr echo: {test_ctor}"
    );
    let production_ctor = source
        .split_once("/// Construct the production fixed-resource configuration.")
        .and_then(|(_, tail)| tail.split_once("#[cfg(test)]\n    fn test("))
        .map(|(body, _)| body)
        .expect("TraceConfig::new source window");
    assert!(
        production_ctor.contains("echo_live: true"),
        "production traces must enable live stderr echo for matrix job logs: {production_ctor}"
    );
}

#[test]
fn open_trace_is_readable_before_finish() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), 0, 8, MAX_TRACE_BYTES).with_file_stem("live");
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");
    drop(drivers);
    let binding = controller.bind(ContextId::PROCESS);
    emit(TraceEvent::ProcessStarted {
        context: ContextId::PROCESS,
    });
    let open_path = temp.path().join("live.open.ndjson");
    let deadline = Instant::now() + Duration::from_secs(1);
    let mut saw_event = false;
    while Instant::now() < deadline {
        if fs::read(&open_path).ok().is_some_and(|bytes| {
            bytes
                .windows(b"process.started".len())
                .any(|window| window == b"process.started")
        }) {
            saw_event = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(binding);
    let report = session.finish(CommandOutcome::Succeeded, None);
    assert!(
        saw_event,
        "open trace was not readable before finish; path existed={} report={report:?}",
        open_path.exists()
    );
}

#[test]
fn paused_capacity_four_accepts_four_and_accounts_remaining_full_drops() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), 0, 4, MAX_TRACE_BYTES)
        .with_test_writer_delay(Duration::from_secs(1));
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");
    assert!(drivers.is_empty());
    let binding = controller.bind(ContextId::PROCESS);
    for ordinal in 0..10 {
        emit(TraceEvent::ObjectiveStarted {
            context: ContextId::PROCESS,
            generation: 7,
            objective_ordinal: ordinal,
            event_offset: 0,
            event_count: 8,
            active_designs: 2,
            candidate_count: 2,
            view_hash: [0x5a; 32],
        });
    }
    drop(binding);
    let report = session.finish(CommandOutcome::Succeeded, None);
    assert_eq!(report.disposition, TraceDisposition::Lossy);
    assert_eq!(report.attempted, 11);
    assert_eq!(report.accepted, 5);
    assert_eq!(report.written, 5);
    assert_eq!(report.dropped_full, 6);
    assert_eq!(
        report.attempted,
        report.accepted
            + report.dropped_full
            + report.dropped_disconnected
            + report.dropped_reentrant
    );
}

#[test]
fn eight_producers_have_distinct_fifo_lane_sequences() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), 8, 16, MAX_TRACE_BYTES);
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");
    drop(controller);
    std::thread::scope(|scope| {
        for (index, lane) in drivers.into_iter().enumerate() {
            scope.spawn(move || {
                let context = ContextId::new(
                    u16::try_from(index)
                        .expect("test lane fits u16")
                        .checked_add(1)
                        .expect("test lane context does not overflow"),
                );
                let binding = lane.bind(context);
                for ordinal in 0..8 {
                    emit(TraceEvent::CellStarted {
                        context,
                        cell_index: ordinal,
                    });
                }
                drop(binding);
            });
        }
    });
    let report = session.finish(CommandOutcome::Succeeded, None);
    assert_eq!(report.disposition, TraceDisposition::Complete);
    assert_eq!(report.accepted, 65);
    assert_eq!(report.written, 65);
    let text = fs::read_to_string(report.path.expect("trace path")).expect("read trace");
    let mut sequences = vec![Vec::new(); 9];
    for line in text.lines() {
        let value: serde_json::Value = serde_json::from_str(line).expect("valid NDJSON line");
        if value.get("class").and_then(serde_json::Value::as_str) == Some("event") {
            let lane = usize::try_from(
                value
                    .get("lane_id")
                    .and_then(serde_json::Value::as_u64)
                    .expect("lane id"),
            )
            .expect("lane id fits usize");
            sequences.get_mut(lane).expect("registered lane").push(
                value
                    .get("lane_sequence")
                    .and_then(serde_json::Value::as_u64)
                    .expect("lane sequence"),
            );
        }
    }
    for lane in sequences.get(1..).expect("driver lanes") {
        assert_eq!(lane, &[1, 2, 3, 4, 5, 6, 7, 8]);
    }
}

#[test]
fn clean_finish_renames_complete_and_creates_private_file() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), 0, 8, MAX_TRACE_BYTES);
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");
    assert!(drivers.is_empty());
    let binding = controller.bind(ContextId::PROCESS);
    emit(TraceEvent::ProcessStarted {
        context: ContextId::PROCESS,
    });
    drop(binding);
    let report = session.finish(CommandOutcome::Failed, None);
    assert_eq!(report.disposition, TraceDisposition::Complete);
    assert_eq!(report.command_outcome, CommandOutcome::Failed);
    let path = report.path.expect("trace path");
    assert!(path.to_string_lossy().ends_with(".complete.ndjson"));
    assert_eq!(
        fs::metadata(&path).expect("metadata").permissions().mode() & 0o777,
        0o600
    );
    assert!(!path.with_extension("open.ndjson").exists());
}

#[test]
fn exclusive_creation_never_overwrites_an_existing_open_file() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), 0, 8, MAX_TRACE_BYTES).with_file_stem("fixed-name");
    let existing = temp.path().join("fixed-name.open.ndjson");
    fs::write(&existing, b"sentinel").expect("seed existing file");
    let error = trace_start_error(TraceSession::start(config));
    assert_eq!(error.kind(), TraceStartErrorKind::FileCreate);
    assert_eq!(fs::read(existing).expect("read sentinel"), b"sentinel");
}

#[test]
fn trace_root_rejects_symlinks_and_new_directories_are_private() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let real = temp.path().join("real");
    fs::create_dir(&real).expect("real directory");
    let link = temp.path().join("link");
    std::os::unix::fs::symlink(&real, &link).expect("trace-root symlink");
    let error = trace_start_error(TraceSession::start(TraceConfig::test(
        &link,
        0,
        8,
        MAX_TRACE_BYTES,
    )));
    assert_eq!(error.kind(), TraceStartErrorKind::DirectoryCreate);

    let nested = temp.path().join("private").join("trace");
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(TraceConfig::test(&nested, 0, 8, MAX_TRACE_BYTES))
        .expect("private trace root");
    drop(controller);
    drop(drivers);
    let report = session.finish(CommandOutcome::Succeeded, None);
    assert_eq!(
        fs::metadata(&nested)
            .expect("trace directory metadata")
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    assert_eq!(report.disposition, TraceDisposition::Complete);
}

#[test]
fn open_file_is_close_on_exec() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let path = temp.path().join("direct.open.ndjson");
    let file = open_trace_file(&path).expect("secure open");
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFD) };
    assert!(flags >= 0);
    assert_ne!(flags & libc::FD_CLOEXEC, 0);
}

#[test]
fn finalization_never_replaces_an_existing_final_file() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let config =
        TraceConfig::test(temp.path(), 0, 8, MAX_TRACE_BYTES).with_file_stem("fixed-final");
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");
    drop(controller);
    drop(drivers);
    let final_path = temp.path().join("fixed-final.complete.ndjson");
    fs::write(&final_path, b"sentinel").expect("seed final path");
    let report = session.finish(CommandOutcome::Succeeded, None);
    assert_eq!(report.disposition, TraceDisposition::Incomplete);
    assert!(report.sink_error);
    assert_eq!(fs::read(final_path).expect("final sentinel"), b"sentinel");
    assert!(temp.path().join("fixed-final.open.ndjson").exists());
}

fn run_sink_fault(fault: TestFault) -> (TraceFinishReport, String) {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), 0, 8, MAX_TRACE_BYTES)
        .with_file_stem("fault")
        .with_fault(fault);
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start fault trace");
    drop(drivers);
    let binding = controller.bind(ContextId::PROCESS);
    emit(TraceEvent::ProcessStarted {
        context: ContextId::PROCESS,
    });
    drop(binding);
    let report = session.finish(CommandOutcome::Succeeded, None);
    let path = report.path.clone().expect("fault trace path");
    let bytes = fs::read(path).expect("read incomplete trace");
    (report, String::from_utf8_lossy(&bytes).into_owned())
}

#[test]
fn write_flush_and_sync_faults_leave_open_incomplete_without_panicking() {
    for fault in [TestFault::WriteAfter(1), TestFault::Flush, TestFault::Sync] {
        let (report, _text) = run_sink_fault(fault);
        assert_eq!(report.disposition, TraceDisposition::Incomplete);
        assert!(report.sink_error);
        assert!(report
            .path
            .as_deref()
            .expect("fault path")
            .to_string_lossy()
            .ends_with(".open.ndjson"));
    }
}

#[test]
fn partial_write_is_only_a_malformed_trailing_line_and_stays_incomplete() {
    let (report, text) = run_sink_fault(TestFault::PartialWriteAfter(1));
    assert_eq!(report.disposition, TraceDisposition::Incomplete);
    assert!(report.sink_error);
    let mut lines = text.split('\n');
    let header = lines.next().expect("complete header");
    assert!(serde_json::from_str::<serde_json::Value>(header).is_ok());
    let trailing = lines.next().expect("partial trailing event");
    assert!(!trailing.is_empty());
    assert!(serde_json::from_str::<serde_json::Value>(trailing).is_err());
    assert!(lines.all(str::is_empty));
}

#[test]
fn stalled_writer_does_not_stall_producer() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), 0, 4, MAX_TRACE_BYTES)
        .with_test_writer_delay(Duration::from_secs(1));
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start stalled trace");
    drop(drivers);
    let binding = controller.bind(ContextId::PROCESS);
    let started = Instant::now();
    for index in 0..1_000 {
        emit(TraceEvent::CellStarted {
            context: ContextId::PROCESS,
            cell_index: index,
        });
    }
    assert!(started.elapsed() < Duration::from_millis(100));
    drop(binding);
    let report = session.finish(CommandOutcome::Succeeded, None);
    assert_eq!(report.attempted, 1_001);
    assert_eq!(report.accepted, 5);
}

#[test]
fn footer_records_the_command_outcome_given_to_finish() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), 0, 8, MAX_TRACE_BYTES);
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");
    drop(controller);
    drop(drivers);
    let report = session.finish(CommandOutcome::Succeeded, None);
    let text = fs::read_to_string(report.path.expect("trace path")).expect("read trace");
    let footer: serde_json::Value =
        serde_json::from_str(text.lines().last().expect("footer line")).expect("valid footer");
    assert_eq!(footer.get("class").expect("footer class"), "footer");
    assert_eq!(
        footer.get("command_outcome").expect("command outcome"),
        "succeeded"
    );
}

#[test]
fn disabled_and_enabled_emit_allocate_nothing_after_binding() {
    emit(TraceEvent::ProcessStarted {
        context: ContextId::PROCESS,
    });
    let disabled_allocations = count_current_thread_allocations(|| {
        for _ in 0..100 {
            emit(TraceEvent::ProcessStarted {
                context: ContextId::PROCESS,
            });
        }
    });
    assert_eq!(disabled_allocations, 0);

    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), 0, 256, MAX_TRACE_BYTES);
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");
    assert!(drivers.is_empty());
    let binding = controller.bind(ContextId::PROCESS);
    emit(TraceEvent::ProcessStarted {
        context: ContextId::PROCESS,
    });
    let enabled_allocations = count_current_thread_allocations(|| {
        for _ in 0..100 {
            emit(TraceEvent::ProcessStarted {
                context: ContextId::PROCESS,
            });
        }
    });
    assert_eq!(enabled_allocations, 0);
    drop(binding);
    let report = session.finish(CommandOutcome::Succeeded, None);
    assert_eq!(report.accepted, 102);
}

#[test]
fn lane_ceiling_refuses_trace_without_creating_output() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let output = temp.path().join("too-many");
    let config = TraceConfig::test(&output, MAX_LANES, 1, MAX_TRACE_BYTES);
    let error = trace_start_error(TraceSession::start(config));
    assert_eq!(error.kind(), TraceStartErrorKind::ResourceLimit);
    assert!(!output.exists());
}

#[test]
fn missing_driver_lane_is_counted_and_makes_trace_incomplete() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(TraceConfig::test(temp.path(), 0, 8, MAX_TRACE_BYTES))
        .expect("start trace");
    drop(controller);
    drivers.note_unbound(2);
    drop(drivers);
    let report = session.finish(CommandOutcome::Succeeded, None);
    assert_eq!(report.unbound_defects, 2);
    assert_eq!(report.disposition, TraceDisposition::Incomplete);
    assert!(report
        .path
        .as_deref()
        .expect("incomplete path")
        .to_string_lossy()
        .ends_with(".open.ndjson"));
}

#[test]
fn heartbeat_marks_last_phase_uncertain_after_a_sequence_gap() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let mut config = TraceConfig::test(temp.path(), 0, 1, MAX_TRACE_BYTES)
        .with_test_writer_delay(Duration::from_millis(30));
    config.heartbeat_interval = Duration::from_millis(5);
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");
    assert!(drivers.is_empty());
    let binding = controller.bind(ContextId::PROCESS);
    emit(TraceEvent::ProcessStarted {
        context: ContextId::PROCESS,
    });
    emit(TraceEvent::Notice {
        context: ContextId::PROCESS,
        code: 7,
        detail: 11,
    });
    std::thread::sleep(Duration::from_millis(80));
    drop(binding);
    let report = session.finish(CommandOutcome::Succeeded, None);
    assert_eq!(report.dropped_full, 1);
    let text = fs::read_to_string(report.path.expect("trace path")).expect("read trace");
    let heartbeat = text
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|value| value.get("class").and_then(serde_json::Value::as_str) == Some("heartbeat"))
        .expect("heartbeat record");
    assert_eq!(
        heartbeat.get("lossy").and_then(serde_json::Value::as_bool),
        Some(true)
    );
}

#[test]
fn terminal_record_survives_saturated_queue() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), 0, 1, MAX_TRACE_BYTES)
        .with_test_writer_delay(Duration::from_millis(30));
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");
    assert!(drivers.is_empty());
    let binding = controller.bind(ContextId::PROCESS);
    emit(TraceEvent::ProcessStarted {
        context: ContextId::PROCESS,
    });
    emit(TraceEvent::Notice {
        context: ContextId::PROCESS,
        code: 7,
        detail: 11,
    });
    drop(binding);

    let report = session.finish(CommandOutcome::Succeeded, None);
    assert_eq!(report.dropped_full, 1);
    let text = fs::read_to_string(report.path.expect("trace path")).expect("read trace");
    let records = text
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("valid NDJSON line"))
        .collect::<Vec<_>>();
    let terminal = records
        .iter()
        .position(|record| {
            record.get("event_name").and_then(serde_json::Value::as_str) == Some("process.finished")
        })
        .expect("reserved terminal process.finished record");
    let first_footer = records
        .iter()
        .position(|record| {
            matches!(
                record.get("class").and_then(serde_json::Value::as_str),
                Some("footer_lane" | "footer")
            )
        })
        .expect("terminal accounting footer");
    assert!(terminal < first_footer);
    assert_eq!(
        records
            .get(terminal)
            .and_then(|record| record.pointer("/payload/status"))
            .and_then(serde_json::Value::as_u64),
        Some(0)
    );
}

#[test]
fn terminal_record_follows_multi_lane_backlog_larger_than_drain_burst() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let backlog = DRAIN_BURST + 8;
    let config = TraceConfig::test(temp.path(), 2, backlog, MAX_TRACE_BYTES)
        .with_test_writer_delay(Duration::from_millis(100));
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");

    let binding = controller.bind(ContextId::PROCESS);
    for detail in 0..backlog {
        emit(TraceEvent::Notice {
            context: ContextId::PROCESS,
            code: 7,
            detail: u64::try_from(detail).expect("detail fits u64"),
        });
    }
    drop(binding);
    std::thread::scope(|scope| {
        for (index, lane) in drivers.into_iter().enumerate() {
            scope.spawn(move || {
                let context = ContextId::new(u16::try_from(index + 1).expect("context fits u16"));
                let binding = lane.bind(context);
                for cell_index in 0..backlog {
                    emit(TraceEvent::CellStarted {
                        context,
                        cell_index,
                    });
                }
                drop(binding);
            });
        }
    });

    let report = session.finish(CommandOutcome::Succeeded, None);
    let ordinary_events = u64::try_from(backlog * 3).expect("event count fits u64");
    assert_eq!(report.attempted, ordinary_events + 1);
    assert_eq!(report.accepted, ordinary_events + 1);
    assert_eq!(report.written, ordinary_events + 1);
    let text = fs::read_to_string(report.path.expect("trace path")).expect("read trace");
    let records = text
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("valid NDJSON line"))
        .collect::<Vec<_>>();
    let events = records
        .iter()
        .filter(|record| record.get("class").and_then(serde_json::Value::as_str) == Some("event"))
        .collect::<Vec<_>>();
    assert_eq!(events.len(), backlog * 3 + 1);
    let (terminal, ordinary_events) = events.split_last().expect("terminal event");
    assert_eq!(
        terminal
            .get("event_name")
            .and_then(serde_json::Value::as_str),
        Some("process.finished")
    );
    let terminal_sequence = terminal
        .get("writer_sequence")
        .and_then(serde_json::Value::as_u64)
        .expect("terminal writer sequence");
    assert!(ordinary_events.iter().all(|event| {
        event
            .get("writer_sequence")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|sequence| sequence < terminal_sequence)
    }));
    let terminal_index = records
        .iter()
        .position(|record| {
            record.get("event_name").and_then(serde_json::Value::as_str) == Some("process.finished")
        })
        .expect("terminal record index");
    assert!(records.iter().skip(terminal_index).skip(1).all(|record| {
        record.get("class").and_then(serde_json::Value::as_str) != Some("event")
    }));
}

#[test]
fn terminal_only_trace_conserves_lane_and_footer_accounting() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(TraceConfig::test(temp.path(), 0, 1, MAX_TRACE_BYTES))
        .expect("start trace");
    drop(controller);
    drop(drivers);

    let report = session.finish(CommandOutcome::Succeeded, None);
    assert_eq!(
        (report.attempted, report.accepted, report.written),
        (1, 1, 1)
    );
    assert_eq!(report.failed_or_discarded, 0);
    let text = fs::read_to_string(report.path.expect("trace path")).expect("read trace");
    let records = text
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("valid NDJSON line"))
        .collect::<Vec<_>>();
    let lane = records
        .iter()
        .find(|record| {
            record.get("class").and_then(serde_json::Value::as_str) == Some("footer_lane")
        })
        .expect("lane footer");
    let footer = records.last().expect("trace footer");
    for record in [lane, footer] {
        assert_eq!(
            record.get("attempted").and_then(serde_json::Value::as_u64),
            Some(1)
        );
        assert_eq!(
            record.get("accepted").and_then(serde_json::Value::as_u64),
            Some(1)
        );
        assert_eq!(
            record.get("written").and_then(serde_json::Value::as_u64),
            Some(1)
        );
        assert_eq!(
            record
                .get("failed_or_discarded")
                .and_then(serde_json::Value::as_u64),
            Some(0)
        );
    }
}

#[test]
fn terminal_failure_preserves_error_hash_and_footer() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let config = TraceConfig::test(temp.path(), 0, 1, MAX_TRACE_BYTES);
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");
    drop(controller);
    drop(drivers);
    let error_hash = [0xab; 32];
    let report = session.finish(CommandOutcome::Failed, Some(error_hash));
    assert_eq!(report.command_outcome, CommandOutcome::Failed);

    let text = fs::read_to_string(report.path.expect("trace path")).expect("read trace");
    let records = text
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("valid NDJSON line"))
        .collect::<Vec<_>>();
    assert!(records.iter().all(|record| {
        record.get("schema").and_then(serde_json::Value::as_str)
            == Some("nasa_dust.runtime_trace.v2")
    }));
    let failed = records
        .iter()
        .filter(|record| {
            record.get("event_name").and_then(serde_json::Value::as_str) == Some("process.failed")
        })
        .collect::<Vec<_>>();
    assert_eq!(failed.len(), 1);
    let failed = failed.first().expect("one failed terminal record");
    assert_eq!(
        failed.get("context").and_then(serde_json::Value::as_u64),
        Some(0)
    );
    assert_eq!(
        failed.get("severity").and_then(serde_json::Value::as_str),
        Some("error")
    );
    assert_eq!(
        failed
            .pointer("/payload/error_hash")
            .and_then(serde_json::Value::as_str),
        Some("abababababababababababababababababababababababababababababababab")
    );
    let footer = records.last().expect("footer record");
    assert_eq!(
        footer
            .get("command_outcome")
            .and_then(serde_json::Value::as_str),
        Some("failed")
    );
}

#[test]
fn live_echo_writes_each_line_under_one_stderr_lock() {
    let source = include_str!("lib.rs");
    let event = source
        .split_once("fn echo_live_event(")
        .and_then(|(_, tail)| tail.split_once("fn echo_live_terminal("))
        .map(|(body, _)| body)
        .expect("event echo source window");
    assert!(
        event.matches(".lock()").count() == 1
            && event.contains("let mut stderr = stderr.lock();")
            && event.contains("write_live_event_line"),
        "event echo must hold one StderrLock across its complete line: {event}"
    );
    let terminal = source
        .split_once("fn echo_live_terminal(")
        .and_then(|(_, tail)| tail.split_once("fn echo_live_heartbeat("))
        .map(|(body, _)| body)
        .expect("terminal echo source window");
    assert!(
        terminal.matches(".lock()").count() == 1
            && terminal.contains("let mut stderr = stderr.lock();")
            && terminal.contains("write_live_terminal_line"),
        "terminal echo must hold one StderrLock across its complete line: {terminal}"
    );
    let heartbeat = source
        .split_once("fn echo_live_heartbeat(")
        .and_then(|(_, tail)| tail.split_once("fn writer_main("))
        .map(|(body, _)| body)
        .expect("heartbeat echo source window");
    assert!(
        heartbeat.matches(".lock()").count() == 1
            && heartbeat.contains("let mut stderr = stderr.lock();")
            && heartbeat.contains("write_live_event_line")
            && heartbeat.contains("heartbeat idle"),
        "heartbeat echo must hold one StderrLock across either complete line: {heartbeat}"
    );
    for echo in [event, terminal, heartbeat] {
        assert!(
            !echo.contains("let mut stderr = io::stderr();"),
            "unlocked stderr permits concurrent diagnostics to splice into trace lines: {echo}"
        );
    }
}

#[test]
fn byte_cap_preserves_terminal_accounting_and_marks_trace_lossy() {
    let temp = tempfile::tempdir().expect("temporary trace root");
    let byte_cap = TERMINAL_RESERVE_BYTES
        .checked_add(2 * 1024)
        .expect("test byte cap");
    let config = TraceConfig::test(temp.path(), 0, 256, byte_cap);
    let StartedTrace {
        controller,
        drivers,
        session,
    } = TraceSession::start(config).expect("start trace");
    assert!(drivers.is_empty());
    let binding = controller.bind(ContextId::PROCESS);
    for ordinal in 0..200 {
        emit(TraceEvent::ObjectiveStarted {
            context: ContextId::PROCESS,
            generation: 7,
            objective_ordinal: ordinal,
            event_offset: 0,
            event_count: 8,
            active_designs: 2,
            candidate_count: 2,
            view_hash: [0x7e; 32],
        });
    }
    drop(binding);
    let report = session.finish(CommandOutcome::Succeeded, None);
    assert_eq!(report.disposition, TraceDisposition::Lossy);
    assert!(report.byte_cap_reached);
    assert_eq!(report.accepted, report.written + report.failed_or_discarded);
    let path = report.path.expect("trace path");
    assert!(path.to_string_lossy().ends_with(".lossy.ndjson"));
    let text = fs::read_to_string(path).expect("read trace");
    let footer: serde_json::Value =
        serde_json::from_str(text.lines().last().expect("footer line")).expect("valid footer");
    assert_eq!(footer.get("class").expect("footer class"), "footer");
    assert_eq!(
        footer
            .get("byte_cap_reached")
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
}

#[test]
fn default_trace_roots_are_external_siblings_or_local_build_paths() {
    let cwd = std::path::Path::new("/work/repo");
    assert_eq!(
        matrix_default_trace_dir(std::path::Path::new("/authority/runs"), "leaf-7", true, cwd),
        std::path::Path::new("/authority/runs.traces/leaf-7")
    );
    assert_eq!(
        matrix_default_trace_dir(std::path::Path::new("/ignored"), "local-9", false, cwd),
        std::path::Path::new("/work/repo/build/runtime-traces/local-9")
    );
    assert_eq!(
        canary_default_trace_dir(std::path::Path::new("/canary/run")),
        std::path::Path::new("/canary/run.traces")
    );
    assert_eq!(
        shard_worker_default_trace_dir(std::path::Path::new("/rdv/cell.socket")),
        std::path::Path::new("/rdv/cell.socket.traces")
    );
}
