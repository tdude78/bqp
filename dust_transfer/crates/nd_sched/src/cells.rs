//! Cross-cell backlog driver.
//!
//! The optimizer runs many independent "cells" (e.g. one descriptive
//! configuration each), every one of which internally fans out over the flat
//! `WorkUnit` graph on the SAME global pool. Cross-cell drivers therefore run on
//! scoped OS threads, outside that pool. Rather than a batch barrier plus a
//! concurrency semaphore that parks workers, cells flow through one bounded set
//! of drivers pulling from a shared backlog. There is no batch barrier and no
//! per-cell pool.
//!
//! Determinism: each cell carries its input index, and results are placed by
//! that index, so the returned vector is in cell-index order regardless of
//! which driver ran which cell or in what order they finished.

use crate::pool;
use nd_runtime_trace::{emit, ContextId, TraceDriverLanes, TraceEvent};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::thread;

#[derive(Clone, Copy)]
enum DriverStart {
    Wait,
    Run,
    Stop,
}

/// Run `driver` over every cell.
///
/// Width one and reentrant calls from a Rayon worker stay on the calling thread
/// so an inner scientific kernel can own its current pool. Wider external calls use
/// `W = min(max_concurrent, cell_count, scientific_pool_width)` scoped
/// OS-thread backlog drivers.
///
/// Returns one result per input cell, in input (cell-index) order.
pub fn run_cells<C, R, F, S>(
    cells: Vec<C>,
    max_concurrent: usize,
    trace_lanes: &mut TraceDriverLanes,
    driver: F,
    status_of: S,
) -> Vec<R>
where
    C: Send,
    R: Send,
    F: Fn(C) -> R + Sync + Send,
    S: Fn(&R) -> u16 + Sync + Send,
{
    let n = cells.len();
    if n == 0 {
        return Vec::new();
    }
    let _ = pool::init_global_pool(None);
    // A Rayon worker must never block waiting for OS drivers whose nested
    // scientific fanout needs that same pool. Serial reentry keeps the worker
    // active and lets nested Rayon execute normally, including width-one pools.
    if rayon::current_thread_index().is_some() {
        trace_lanes.note_unbound(n);
        return cells.into_iter().map(driver).collect();
    }
    if max_concurrent <= 1 {
        return cells
            .into_iter()
            .enumerate()
            .map(|(index, cell)| {
                let context = ContextId::for_cell_index(index).unwrap_or(ContextId::PROCESS);
                let _context_guard = nd_runtime_trace::enter_context(context);
                emit(TraceEvent::CellStarted {
                    context,
                    cell_index: index,
                });
                let output = driver(cell);
                emit(TraceEvent::CellFinished {
                    context,
                    status: status_of(&output),
                    recovered_shards: 0,
                });
                output
            })
            .collect();
    }

    let backlog = Mutex::new(cells.into_iter().enumerate().collect::<VecDeque<_>>());
    let results = Mutex::new(Vec::with_capacity(n));
    let start = (Mutex::new(DriverStart::Wait), Condvar::new());

    // Preserve one configured global scientific pool before drivers begin.
    let scientific_pool_width = pool::num_threads();
    let workers = max_concurrent.min(n).min(scientific_pool_width);
    trace_lanes.note_unbound(workers.saturating_sub(trace_lanes.len()));
    let driver_ref = &driver;
    let status_ref = &status_of;
    let mut available_trace_lanes = trace_lanes.take_for_scheduler(workers).into_iter();
    let returned_trace_lanes = Mutex::new(Vec::with_capacity(workers));
    let mut spawn_defects = 0usize;

    // Scoped OS threads leave every driver outside Rayon, so inner scientific
    // fanout can fully use the one global Rayon pool. `scope` joins all
    // drivers and re-propagates a driver panic before results are collected.
    thread::scope(|s| {
        let mut spawn_failed = false;
        for worker_idx in 0..workers {
            let trace_lane = available_trace_lanes.next();
            let backlog_ref = &backlog;
            let results_ref = &results;
            let start_ref = &start;
            let returned_trace_lanes_ref = &returned_trace_lanes;
            let spawn = thread::Builder::new()
                .name(format!("nd-cell-driver-{worker_idx}"))
                .stack_size(crate::WORKER_STACK_BYTES)
                .spawn_scoped(s, move || {
                    let mut trace_binding = trace_lane.map(|lane| lane.bind(ContextId::PROCESS));
                    let (start_state, start_changed) = start_ref;
                    let mut state = start_state.lock();
                    loop {
                        match *state {
                            DriverStart::Wait => start_changed.wait(&mut state),
                            DriverStart::Run => break,
                            DriverStart::Stop => {
                                if let Some(binding) = trace_binding.take() {
                                    returned_trace_lanes_ref
                                        .lock()
                                        .push((worker_idx, binding.unbind()));
                                }
                                return;
                            }
                        }
                    }
                    drop(state);

                    loop {
                        let Some((idx, cell)) = backlog_ref.lock().pop_front() else {
                            break;
                        };
                        let context = ContextId::for_cell_index(idx).unwrap_or(ContextId::PROCESS);
                        if let Some(binding) = &mut trace_binding {
                            binding.set_context(context);
                        }
                        emit(TraceEvent::CellStarted {
                            context,
                            cell_index: idx,
                        });
                        let out = driver_ref(cell);
                        emit(TraceEvent::CellFinished {
                            context,
                            status: status_ref(&out),
                            recovered_shards: 0,
                        });
                        results_ref.lock().push((idx, out));
                    }
                    if let Some(binding) = trace_binding {
                        returned_trace_lanes_ref
                            .lock()
                            .push((worker_idx, binding.unbind()));
                    }
                });
            if let Err(spawn_error) = spawn {
                spawn_failed = true;
                spawn_defects = spawn_defects.saturating_add(1);
                // The serial fallback below is deliberate and correct, but it is
                // also invisible: a width-64 leaf silently becomes a width-1 leaf
                // and the only symptom is wall time. Say so on stderr, because
                // this is the one degradation the operator cannot otherwise see
                // -- `note_unbound` records a count with no reason, the trace may
                // itself be disabled, and EAGAIN (raise the thread/process limit)
                // and ENOMEM (reduce width or stack size) demand opposite
                // responses. Naming the errno is the whole value of this line.
                eprintln!(
                    "nd_sched: worker thread spawn failed after {worker_idx} of \
                     {workers} workers ({spawn_error}); the remaining backlog runs \
                     SERIALLY in this process, so this leaf is now effectively \
                     width 1"
                );
                break;
            }
        }

        {
            let (start_state, start_changed) = &start;
            *start_state.lock() = if spawn_failed {
                DriverStart::Stop
            } else {
                DriverStart::Run
            };
            start_changed.notify_all();
        }

        // Resource exhaustion must not turn a valid campaign into a process
        // panic. Waiting drivers stop before taking work, and the caller drains
        // the same deterministic backlog serially.
        if spawn_failed {
            loop {
                let Some((idx, cell)) = backlog.lock().pop_front() else {
                    break;
                };
                let context = ContextId::for_cell_index(idx).unwrap_or(ContextId::PROCESS);
                let _context_guard = nd_runtime_trace::enter_context(context);
                emit(TraceEvent::CellStarted {
                    context,
                    cell_index: idx,
                });
                let output = driver_ref(cell);
                emit(TraceEvent::CellFinished {
                    context,
                    status: status_ref(&output),
                    recovered_shards: 0,
                });
                results.lock().push((idx, output));
            }
        }
    });

    trace_lanes.note_unbound(spawn_defects);

    let mut returned_trace_lanes = returned_trace_lanes.into_inner();
    returned_trace_lanes.sort_unstable_by_key(|(worker_idx, _)| *worker_idx);
    trace_lanes.restore_from_scheduler(
        returned_trace_lanes
            .into_iter()
            .map(|(_, lane)| lane)
            .chain(available_trace_lanes)
            .collect(),
    );

    let mut indexed = results.into_inner();
    indexed.sort_unstable_by_key(|(idx, _)| *idx);
    indexed.into_iter().map(|(_, output)| output).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nd_runtime_trace::{
        CommandOutcome, CommandRole, ContextId, SchedulerIdentity, TraceConfig, TraceContext,
        TraceDisposition, TraceSession,
    };
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn scheduler_test_traces_do_not_echo_to_stderr() {
        let source = include_str!("cells.rs");
        let mut remaining = source
            .split_once("#[cfg(test)]\nmod tests")
            .map(|(_, tail)| tail)
            .expect("cells test-module boundary");
        let mut sites = 0usize;
        while let Some((_, tail)) =
            remaining.split_once("TraceConfig::new(\n            temp.path()")
        {
            sites = sites.saturating_add(1);
            let until_start = tail
                .split_once("TraceSession::start")
                .map(|(body, _)| body)
                .expect("test TraceConfig must be started");
            assert!(
                until_start.contains("echo_live = false"),
                "scheduler test traces must disable live stderr echo; cargo does not capture writer-thread stdio: {until_start}"
            );
            remaining = tail;
        }
        assert!(
            sites >= 2,
            "scheduler tests must still construct production TraceConfig"
        );
    }

    #[test]
    fn driver_lanes_emit_cell_boundaries_without_moving_result_order() {
        let temp = tempfile::tempdir().expect("temporary trace root");
        let contexts = (0..4)
            .map(|index| TraceContext {
                id: ContextId::new(u16::try_from(index + 1).expect("test context fits")),
                kind: "cell".to_owned(),
                identity: format!("cell-{index}"),
            })
            .collect();
        let mut config = TraceConfig::new(
            temp.path().to_path_buf(),
            CommandRole::Matrix,
            [0x11; 32],
            Some([0x22; 32]),
            SchedulerIdentity::default(),
            contexts,
            2,
        );
        config.echo_live = false;
        let mut started = TraceSession::start(config).expect("start trace");
        let controller = started.controller.bind(ContextId::PROCESS);
        let out = run_cells(
            (0..4usize).collect(),
            2,
            &mut started.drivers,
            |cell| {
                std::thread::sleep(std::time::Duration::from_millis(2));
                cell * 3
            },
            |_| 0,
        );
        assert_eq!(out, vec![0, 3, 6, 9]);
        drop(controller);
        let report = started.session.finish(CommandOutcome::Succeeded, None);
        assert_eq!(report.disposition, TraceDisposition::Complete);
        let text = fs::read_to_string(report.path.expect("trace path")).expect("read trace");
        let event_names: Vec<String> = text
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter_map(|value| {
                value
                    .get("event_name")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .collect();
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "cell.started")
                .count(),
            4
        );
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "cell.finished")
                .count(),
            4
        );
    }

    #[test]
    fn run_cells_returns_results_in_cell_index_order() {
        let cells: Vec<usize> = (0..200).collect();
        let out = run_cells(
            cells.clone(),
            4,
            &mut TraceDriverLanes::disabled(),
            |c| c * 10 + 1,
            |_| 0,
        );
        assert_eq!(out.len(), cells.len());
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i * 10 + 1, "results must be in cell-index order");
        }
    }

    #[test]
    fn run_cells_runs_every_cell_exactly_once() {
        static SEEN: AtomicUsize = AtomicUsize::new(0);
        SEEN.store(0, Ordering::SeqCst);
        let cells: Vec<usize> = (0..500).collect();
        let out = run_cells(
            cells,
            8,
            &mut TraceDriverLanes::disabled(),
            |c| {
                SEEN.fetch_add(1, Ordering::SeqCst);
                c
            },
            |_| 0,
        );
        assert_eq!(SEEN.load(Ordering::SeqCst), 500, "every cell runs once");
        assert_eq!(out, (0..500).collect::<Vec<_>>());
    }

    #[test]
    fn run_cells_handles_empty_and_singleton() {
        let empty = run_cells(
            Vec::<u32>::new(),
            4,
            &mut TraceDriverLanes::disabled(),
            |c| c,
            |_| 0,
        );
        assert!(empty.is_empty());
        // Width one stays on the calling thread so nested scientific kernels
        // see an external caller and can use their flat population driver.
        let single = run_cells(
            vec![42u32],
            1,
            &mut TraceDriverLanes::disabled(),
            |c| c + 1,
            |_| 0,
        );
        assert_eq!(single, vec![43]);
    }

    #[test]
    fn width_one_runs_driver_outside_rayon_pool() {
        let outside_pool = run_cells(
            vec![7u32],
            1,
            &mut TraceDriverLanes::disabled(),
            |_| rayon::current_thread_index().is_none(),
            |_| 0,
        );
        assert_eq!(outside_pool, vec![true]);
    }

    #[test]
    fn multi_cell_drivers_stay_outside_rayon_while_inner_work_uses_it() {
        let driver_outside = AtomicUsize::new(0);
        let inner_pool_work = AtomicUsize::new(0);
        let cells: Vec<usize> = (0..4).collect();

        let out = run_cells(
            cells,
            2,
            &mut TraceDriverLanes::disabled(),
            |_| {
                if rayon::current_thread_index().is_none() {
                    driver_outside.fetch_add(1, Ordering::SeqCst);
                }

                let units: Vec<usize> = (0..128).collect();
                crate::flat::flat_eval(&units, |_| {
                    if rayon::current_thread_index().is_some() {
                        inner_pool_work.fetch_add(1, Ordering::SeqCst);
                    }
                    1usize
                })
                .into_iter()
                .sum::<usize>()
            },
            |_| 0,
        );

        assert_eq!(out, vec![128; 4]);
        assert_eq!(driver_outside.load(Ordering::SeqCst), 4);
        assert_eq!(inner_pool_work.load(Ordering::SeqCst), 4 * 128);
    }

    #[test]
    fn reentrant_run_cells_stays_on_current_rayon_worker() {
        let local = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("build one-thread local pool");

        let out = local.install(|| {
            run_cells(
                (0..4usize).collect(),
                4,
                &mut TraceDriverLanes::disabled(),
                |cell| {
                    let driver_worker = rayon::current_thread_index();
                    let inner_units: Vec<usize> = (0..16).collect();
                    let inner_workers =
                        crate::flat::flat_eval(&inner_units, |_| rayon::current_thread_index());
                    (cell, driver_worker, inner_workers)
                },
                |_| 0,
            )
        });

        // Nothing below runs if `run_cells` returns no rows, and the inner
        // `all` is vacuously true if `flat_eval` returns no workers. Either
        // collapse would leave this test green while checking no worker
        // affinity at all. Floors come from the fixed fan-out: 4 cells above,
        // 16 inner units each.
        assert_eq!(out.len(), 4, "run_cells must return one row per cell");
        for (expected, (cell, driver_worker, inner_workers)) in out.into_iter().enumerate() {
            assert_eq!(cell, expected);
            assert_eq!(driver_worker, Some(0));
            assert_eq!(
                inner_workers.len(),
                16,
                "flat_eval must return one worker per inner unit"
            );
            assert!(inner_workers.iter().all(|worker| *worker == Some(0)));
        }
    }

    #[test]
    fn reentrant_rayon_path_never_emits_and_marks_trace_incomplete() {
        let temp = tempfile::tempdir().expect("temporary trace root");
        let mut config = TraceConfig::new(
            temp.path().to_path_buf(),
            CommandRole::Matrix,
            [0x11; 32],
            Some([0x22; 32]),
            SchedulerIdentity::default(),
            Vec::new(),
            0,
        );
        config.echo_live = false;
        let mut started = TraceSession::start(config).expect("start trace");
        let controller = started.controller.bind(ContextId::PROCESS);
        let local = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("build one-thread local pool");

        let out = local.install(|| {
            run_cells(
                vec![1usize, 2, 3],
                3,
                &mut started.drivers,
                |cell| cell * 2,
                |_| 0,
            )
        });
        assert_eq!(out, vec![2, 4, 6]);
        drop(controller);
        let report = started.session.finish(CommandOutcome::Succeeded, None);
        assert_eq!(report.disposition, TraceDisposition::Incomplete);
        assert_eq!(report.unbound_defects, 3);
        assert_eq!(report.attempted, 1);
    }

    #[test]
    fn run_cells_supports_nested_parallelism() {
        // Each OS driver fans out through the one global Rayon pool.
        let cells: Vec<usize> = (0..16).collect();
        let out = run_cells(
            cells,
            4,
            &mut TraceDriverLanes::disabled(),
            |c| {
                let inner: Vec<usize> = (0..100).collect();
                let mapped = crate::flat::flat_eval(&inner, |&x| x + c);
                mapped.iter().copied().max().unwrap()
            },
            |_| 0,
        );
        for (c, &v) in out.iter().enumerate() {
            assert_eq!(v, 99 + c);
        }
    }
}
