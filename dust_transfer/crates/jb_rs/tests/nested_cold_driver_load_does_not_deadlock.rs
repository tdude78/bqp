//! A cold `compiled_part_a_v3_drivers()` must return, not hang.
//!
//! Its own test binary, and the reason is the defect it guards. The Part A v3
//! drivers are built FROM the compiled drivers:
//! `compiled_part_a_v3_drivers` -> `build_part_a_v3_drivers` ->
//! `compiled_drivers`. Both go through `success_only_cached`. Serialising the
//! cold path with ONE mutex shared by every cache in the module therefore takes
//! the same non-reentrant lock twice on one thread and deadlocks outright --
//! only on a COLD load, so any test that ran after something else had already
//! populated `COMPILED_DRIVERS` would pass and see nothing.
//!
//! TWO fixes, applied in that order, and the second supersedes the first:
//!
//!   1. `ad483a6e` gave each cache its own lock, so the two acquisitions are no
//!      longer the same mutex.
//!   2. This one removes the NESTING. `compiled_part_a_v3_drivers` warms the
//!      parent before taking its own cold-load lock, so after that returns Ok
//!      the call inside `build_part_a_v3_drivers` hits the `get()` fast path and
//!      takes no lock at all.
//!
//! NOTE FOR ANYONE RE-RUNNING THE ORIGINAL POISON PROOF: re-sharing the two
//! locks NO LONGER reproduces the hang, and that is a strengthening, not a
//! weakness in this test. Measured after (2): shared locks, 0.02 s. The fix no
//! longer depends on getting a lock ORDERING right, which is the property worth
//! having -- an ordering nothing enforces is a deadlock waiting for its second
//! edge. To reproduce the original defect you must restore the nesting as well.
//!
//! This test exists because the deadlock was shipped, and a hang is the one
//! failure mode a test suite reports as a timeout rather than a diagnosis.
use std::sync::mpsc;
use std::time::Duration;

/// Generous: the load parses and validates the compiled SET plus the v3 scenario
/// rows. It is a deadlock detector, not a performance bound -- a deadlock never
/// completes, so any finite budget separates the two.
const BUDGET: Duration = Duration::from_mins(1);

#[test]
fn cold_part_a_v3_driver_load_completes() {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let outcome = jb_rs::drivers::compiled_part_a_v3_drivers().map(|drivers| {
            // Touch the value so the load cannot be optimised into nothing.
            std::ptr::from_ref(drivers.as_ref()).addr()
        });
        let _ = tx.send(outcome.is_ok());
    });

    match rx.recv_timeout(BUDGET) {
        Ok(true) => {}
        Ok(false) => panic!("cold Part A v3 driver load failed"),
        Err(mpsc::RecvTimeoutError::Timeout) => panic!(
            "cold Part A v3 driver load did not finish within {BUDGET:?}: the nested load \
             (compiled_part_a_v3_drivers -> compiled_drivers) is deadlocking on a shared \
             cold-path lock"
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("cold Part A v3 driver load panicked")
        }
    }
}
