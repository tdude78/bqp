//! Parallelization decision utilities.
//!
//! Provides reusable helpers for determining when to enable parallel execution,
//! reducing code duplication across modules.

/// Determine if parallelization should be enabled based on work size and context.
///
/// Checks two conditions: not in a nested Rayon context and sufficient work size.
///
/// **Performance:** Prevents overhead on small workloads while enabling parallelism
/// for large batches. Automatically disables when nested to avoid deadlocks.
///
/// # Arguments
/// * `work_size` - Number of work items to process
/// * `threshold` - Minimum work size to enable parallelism
///
/// # Returns
/// `true` if parallelization should be used, `false` otherwise
///
/// # Example
/// ```rust,ignore
/// use satpy_core::parallel_utils::should_parallelize;
///
/// const BATCH_THRESHOLD: usize = 512;
///
/// if should_parallelize(n_particles, BATCH_THRESHOLD) {
///     particles.par_iter_mut().for_each(|p| process(p));
/// } else {
///     particles.iter_mut().for_each(|p| process(p));
/// }
/// ```
#[inline]
#[must_use]
pub fn should_parallelize(work_size: usize, threshold: usize) -> bool {
    #[cfg(feature = "parallel")]
    let _ = nd_sched::init_global_pool(None);

    // Check for nested Rayon context
    #[cfg(feature = "parallel")]
    let is_nested = rayon::current_thread_index().is_some();
    #[cfg(not(feature = "parallel"))]
    let is_nested = false;

    // No benefit if only a single Rayon worker is available
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() <= 1 {
        return false;
    }

    // Check sufficient work size
    let sufficient_work = work_size >= threshold;

    !is_nested && sufficient_work
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;

    #[test]
    fn nested_context_stays_sequential() {
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let results: Vec<bool> = pool.install(|| {
            (0..2)
                .into_par_iter()
                .map(|_| should_parallelize(1024, 1))
                .collect()
        });
        // `all` is vacuously true on an empty vector, so a par_iter that
        // produced nothing would leave this green having probed no nested
        // context. N = 2 is the fixed `(0..2)` fan-out above.
        assert_eq!(results.len(), 2, "both nested probes must report");
        assert!(results.iter().all(|v| !*v), "nested rayon stays sequential");
    }

    #[test]
    fn outer_call_parallelizes_when_work_is_large() {
        let r = should_parallelize(1024, 1);
        assert!(r, "outer calls parallelize when work exceeds threshold");
    }

    #[test]
    fn satpy_first_touch_uses_scheduler_global_pool() {
        const CHILD_ENV: &str = "NASA_DUST_SATPY_SCHED_POOL_CHILD";
        const CHILD_MARKER: &str = "NASA_DUST_SATPY_SCHED_POOL_CHILD_EXECUTED";
        const TEST_NAME: &str =
            "parallel_utils::tests::satpy_first_touch_uses_scheduler_global_pool";

        if std::env::var_os(CHILD_ENV).is_none() {
            let output = std::process::Command::new(
                std::env::current_exe().expect("current Rust test executable"),
            )
            .args([TEST_NAME, "--exact", "--nocapture"])
            .env(CHILD_ENV, "1")
            .env("RUST_TEST_THREADS", "1")
            .output()
            .expect("spawn isolated first-touch child test");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "child failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            assert!(
                stdout.contains(CHILD_MARKER),
                "child reported success without executing test\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            return;
        }

        println!("{CHILD_MARKER}");
        assert_eq!(nd_sched::init_global_pool(Some(2)), 2);
        assert!(should_parallelize(1024, 1));
        let mut worker_names = rayon::broadcast(|_| {
            std::thread::current()
                .name()
                .unwrap_or("<unnamed>")
                .to_owned()
        });
        worker_names.sort();
        assert_eq!(worker_names.len(), 2, "configured scheduler width must win");
        assert!(
            worker_names
                .iter()
                .all(|name| name.starts_with("nd-sched-")),
            "satpy first touch must use nd_sched pool, got {worker_names:?}"
        );
    }
}
