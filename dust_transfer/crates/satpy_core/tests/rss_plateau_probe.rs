//! Fixed-worker RSS plateau probe for bounded gravity-cache heap authority.
//!
//! LOUD SCOPE WARNING: this whole file is gated on
//! `cfg(all(target_os = "linux", feature = "autodiff"))`. On macOS — and on
//! Linux without `--features autodiff` — it compiles to ZERO tests and the
//! runner still exits 0 ("running 0 tests"). A green run here proves nothing
//! unless the output shows the test actually ran.
//!
//! Proves that repeated construct/touch/drop cycles of `GravityCacheGeneric<DualVec>`
//! on a fixed pool of reused worker threads reach a resident-set plateau instead of
//! unbounded heap growth. Reads `/proc/self/status` `VmRSS` (not monotonic
//! `ru_maxrss`), so the probe is Linux-only and must run in its own test process:
//!
//! ```bash
//! cargo test -p satpy_core --features autodiff --test rss_plateau_probe -- --nocapture --test-threads=1
//! ```
//!
//! Acceptance. The 2026-07-14 handoff suggested `terminal <= first-wave peak +
//! 8 MiB` and allowed calibration only against proven fixed physical allocation.
//! The first TC run (tc028, HEAD 10f11408) proved that bound physically
//! inappropriate for this workload: each recurrence matrix is one 549 152-byte
//! chunk, and glibc's dynamic mmap-threshold adaptation migrates freed chunks of
//! that size from mmap into per-thread arenas over the first few waves (observed
//! 20.4 -> 67.5 MiB across waves 2-5), after which RSS held byte-flat for 28
//! consecutive waves (67 476 -> 67 488 KiB) and dropped to 51.4 MiB at terminal
//! free. That is bounded allocator retention, not growth. Calibrated criteria:
//!
//! 1. first-wave alive delta <= 32 MiB (unchanged);
//! 2. steady state: max alive over the last half of reuse waves may exceed the
//!    max over the first half (waves 2..) by at most 2 MiB — a real per-wave
//!    leak of even ~128 KiB would add ~2 MiB across 16 waves and fail;
//! 3. absolute ceiling: max alive <= baseline + 96 MiB, ~5x the touched
//!    working set (16 caches x 1.05 MiB) plus observed arena steady state.
#![cfg(all(target_os = "linux", feature = "autodiff"))]

use std::sync::mpsc;

use satpy_core::gravity::GravityCacheGeneric;
use satpy_core::DualVec;

const WORKERS: usize = 8;
const CACHES_PER_WORKER_PER_WAVE: usize = 2;
/// Create/drop waves executed on the same threads after the first wave.
const REUSE_WAVES: usize = 32;
const FIRST_WAVE_MAX_DELTA_KIB: i64 = 32 * 1024;
const SECOND_HALF_MAX_GROWTH_KIB: i64 = 2 * 1024;
const ABSOLUTE_CEILING_ABOVE_BASELINE_KIB: i64 = 96 * 1024;

enum Command {
    BuildAndTouch,
    Drop,
    Exit,
}

#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test-only probe helper: a missing or unparsable /proc/self/status must abort loudly; clippy's allow-expect-in-tests covers #[test] fns, not free helpers"
)]
fn vm_rss_kib() -> i64 {
    let status =
        std::fs::read_to_string("/proc/self/status").expect("probe must read /proc/self/status");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .expect("VmRSS must parse as integer KiB");
        }
    }
    panic!("VmRSS line missing from /proc/self/status");
}

/// Allocate and write every recurrence page.
///
/// Deliberately `prime_storage`, not `reset`: `reset` clears only the live
/// recurrence prefix (7x7 of 131x131 at the sealed order), so a probe built on
/// it would measure an allocation it never touched.
fn build_touched_cache() -> GravityCacheGeneric<DualVec> {
    let mut cache = GravityCacheGeneric::<DualVec>::new();
    cache.prime_storage();
    std::hint::black_box(&cache);
    cache
}

#[test]
fn fixed_worker_dual_cache_rss_plateaus_across_create_drop_waves() {
    let baseline_kib = vm_rss_kib();

    let mut cmd_txs = Vec::with_capacity(WORKERS);
    let mut ack_rxs = Vec::with_capacity(WORKERS);
    let mut handles = Vec::with_capacity(WORKERS);
    for worker_id in 0..WORKERS {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
        let (ack_tx, ack_rx) = mpsc::channel::<()>();
        handles.push(
            std::thread::Builder::new()
                .name(format!("rss-probe-worker-{worker_id}"))
                .spawn(move || {
                    let mut held: Vec<GravityCacheGeneric<DualVec>> = Vec::new();
                    std::hint::black_box(held.len());
                    while let Ok(cmd) = cmd_rx.recv() {
                        match cmd {
                            Command::BuildAndTouch => {
                                held = (0..CACHES_PER_WORKER_PER_WAVE)
                                    .map(|_| build_touched_cache())
                                    .collect();
                            }
                            Command::Drop => held = Vec::new(),
                            Command::Exit => break,
                        }
                        // `held`'s only job is to own resident memory between
                        // commands; read it so the ownership stays live under
                        // the optimizer instead of being a dead store.
                        std::hint::black_box(held.len());
                        ack_tx.send(()).expect("probe main thread must stay alive");
                    }
                })
                .expect("fixed probe worker must spawn"),
        );
        cmd_txs.push(cmd_tx);
        ack_rxs.push(ack_rx);
    }

    let run_wave = || -> i64 {
        for tx in &cmd_txs {
            tx.send(Command::BuildAndTouch)
                .expect("probe worker must accept build command");
        }
        for rx in &ack_rxs {
            rx.recv().expect("probe worker must ack build");
        }
        let alive_kib = vm_rss_kib();
        for tx in &cmd_txs {
            tx.send(Command::Drop)
                .expect("probe worker must accept drop command");
        }
        for rx in &ack_rxs {
            rx.recv().expect("probe worker must ack drop");
        }
        alive_kib
    };

    let first_wave_alive_kib = run_wave();
    let first_wave_delta_kib = first_wave_alive_kib - baseline_kib;
    println!(
        "rss-probe baseline={baseline_kib} KiB first_wave_alive={first_wave_alive_kib} KiB \
         first_wave_delta={first_wave_delta_kib} KiB"
    );

    let mut reuse_alive_kib = Vec::with_capacity(REUSE_WAVES);
    for wave in 1..=REUSE_WAVES {
        let alive_kib = run_wave();
        reuse_alive_kib.push(alive_kib);
        println!("rss-probe wave={wave} alive={alive_kib} KiB");
    }
    let terminal_dropped_kib = vm_rss_kib();
    let max_reuse_alive_kib = *reuse_alive_kib.iter().max().expect("reuse waves ran");
    let (first_half, second_half) = reuse_alive_kib.split_at(REUSE_WAVES / 2);
    let first_half_max_kib = *first_half.iter().max().expect("first half ran");
    let second_half_max_kib = *second_half.iter().max().expect("second half ran");
    println!(
        "rss-probe max_reuse_alive={max_reuse_alive_kib} KiB \
         first_half_max={first_half_max_kib} KiB second_half_max={second_half_max_kib} KiB \
         terminal_dropped={terminal_dropped_kib} KiB"
    );

    for tx in &cmd_txs {
        tx.send(Command::Exit)
            .expect("probe worker must accept exit");
    }
    for handle in handles {
        handle.join().expect("probe worker must exit cleanly");
    }

    assert!(
        first_wave_delta_kib <= FIRST_WAVE_MAX_DELTA_KIB,
        "first-wave RSS delta {first_wave_delta_kib} KiB exceeds preregistered \
         {FIRST_WAVE_MAX_DELTA_KIB} KiB bound"
    );
    assert!(
        second_half_max_kib <= first_half_max_kib + SECOND_HALF_MAX_GROWTH_KIB,
        "second-half RSS max {second_half_max_kib} KiB exceeds first-half max \
         {first_half_max_kib} KiB + {SECOND_HALF_MAX_GROWTH_KIB} KiB steady-state bound: \
         RSS is still growing after allocator arenas should have stabilized"
    );
    assert!(
        max_reuse_alive_kib <= baseline_kib + ABSOLUTE_CEILING_ABOVE_BASELINE_KIB,
        "reuse-wave RSS {max_reuse_alive_kib} KiB exceeds baseline {baseline_kib} KiB + \
         {ABSOLUTE_CEILING_ABOVE_BASELINE_KIB} KiB absolute physical ceiling"
    );
    assert!(
        terminal_dropped_kib <= max_reuse_alive_kib,
        "terminal dropped RSS {terminal_dropped_kib} KiB exceeds max alive \
         {max_reuse_alive_kib} KiB: freeing all caches must not grow RSS"
    );
}
