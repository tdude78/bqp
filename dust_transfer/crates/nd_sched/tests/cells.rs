use nd_runtime_trace::TraceDriverLanes;
use nd_sched::{init_global_pool, run_cells};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Cell execution is capped at the scientific pool width — **and reaches it**.
///
/// The reaching half was added 2026-08-06, and the release condition changed
/// with it. The releaser used to wait for `active == requested`, which is
/// `pool_width + 2` and therefore unreachable BY CONSTRUCTION: the wait always
/// ran to its 250 ms deadline, and if the workers had been slow to start it
/// would have released with `peak == 1` and passed. A cap test that observes a
/// concurrency of one has not seen the cap.
///
/// Waiting for `active == pool_width` is the largest reachable value, so the
/// wait now ends when the property is actually demonstrated rather than on a
/// timer, and `peak == pool_width` below is a real observation. The deadline
/// stays as a backstop that fails loudly instead of hanging.
#[test]
fn os_driver_count_never_exceeds_scientific_pool_width_and_reaches_it() {
    let pool_width = init_global_pool(Some(2));
    let requested = pool_width + 2;
    let active = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);
    let release = AtomicBool::new(false);

    std::thread::scope(|scope| {
        scope.spawn(|| {
            // Generous, because it is a backstop against a hang and not a
            // measurement: `pool_width` workers each claim a cell in
            // microseconds once the pool exists.
            let deadline = Instant::now() + Duration::from_secs(30);
            while active.load(Ordering::SeqCst) < pool_width && Instant::now() < deadline {
                std::thread::yield_now();
            }
            release.store(true, Ordering::SeqCst);
        });

        let out = run_cells(
            (0..requested).collect(),
            requested,
            &mut TraceDriverLanes::disabled(),
            |cell| {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                while !release.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
                active.fetch_sub(1, Ordering::SeqCst);
                cell
            },
            |_| 0,
        );
        assert_eq!(out, (0..requested).collect::<Vec<_>>());
    });

    let observed = peak.load(Ordering::SeqCst);
    assert!(
        observed <= pool_width,
        "OS driver concurrency peaked at {observed}, above the scientific pool \
         width {pool_width}"
    );
    // Non-vacuity. Without this the cap holds trivially at a peak of 1, which
    // is what a serial `run_cells` would produce.
    assert_eq!(
        observed, pool_width,
        "cell execution never reached the pool width ({observed} of \
         {pool_width}), so the cap above was not exercised. Either `run_cells` \
         is running serially, or the 30 s release backstop fired -- both are \
         findings, neither is a flake to be tolerated"
    );
}
