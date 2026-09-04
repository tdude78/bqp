//! Index-order-preserving parallel evaluation on the single global pool.

use rayon::prelude::*;

/// Evaluate `f` over every unit in `units`, in parallel, PRESERVING input index
/// order in the returned vector.
///
/// This is the one rayon boundary for the flat graph. `par_iter().map().collect()`
/// on a slice is an indexed parallel iterator, so `out[i]` is always `f(&units[i])`
/// no matter which worker computed it — the determinism contract. Do NOT replace
/// the `collect` with a parallel `sum`/`reduce`/`fold`: floating-point reductions
/// are order-sensitive (see the `no_par_float_reduce` lint).
#[inline]
pub fn flat_eval<T, R, F>(units: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    let _ = crate::pool::init_global_pool(None);
    units.par_iter().map(f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_eval_preserves_input_order_under_parallelism() {
        // Enough units to actually spread across workers; the closure returns a
        // value derived from the input so any reordering is detectable.
        let units: Vec<usize> = (0..10_000).collect();
        let out = flat_eval(&units, |&x| x * 3 + 1);
        assert_eq!(out.len(), units.len());
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i * 3 + 1, "flat_eval must preserve index order");
        }
    }

    #[test]
    fn flat_eval_first_touch_uses_scheduler_pool() {
        const CHILD_ENV: &str = "NASA_DUST_FLAT_SCHED_POOL_CHILD";
        const CHILD_MARKER: &str = "NASA_DUST_FLAT_SCHED_POOL_CHILD_EXECUTED";
        const TEST_NAME: &str = "flat::tests::flat_eval_first_touch_uses_scheduler_pool";

        if std::env::var_os(CHILD_ENV).is_none() {
            let output = std::process::Command::new(
                std::env::current_exe().expect("current Rust test executable"),
            )
            .args([TEST_NAME, "--exact", "--nocapture"])
            .env(CHILD_ENV, "1")
            .env("RUST_TEST_THREADS", "1")
            .output()
            .expect("spawn isolated flat first-touch child test");
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
        assert_eq!(crate::pool::init_global_pool(Some(2)), 2);
        let input = [1_u8, 2_u8];
        assert_eq!(flat_eval(&input, |&value| value), input);
        let mut worker_names = rayon::broadcast(|_| {
            std::thread::current()
                .name()
                .unwrap_or("<unnamed>")
                .to_owned()
        });
        worker_names.sort();
        assert_eq!(worker_names.len(), 2, "explicit generic width must win");
        assert!(
            worker_names
                .iter()
                .all(|name| name.starts_with("nd-sched-")),
            "flat evaluator first touch must use scheduler pool, got {worker_names:?}"
        );
    }
}
