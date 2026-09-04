//! The single process-wide rayon thread pool.
//!
//! There is exactly ONE pool for the whole process, built once from config.
//! This crate deliberately provides NO scoped/per-batch pools: the oracle's
//! `build_scoped_pool` / `scoped_pool` pattern (a fresh N-worker pool per
//! population batch) is the oversubscription anti-pattern being removed. All
//! data-parallel work runs on the global pool via rayon's global `par_iter` /
//! `scope` / `spawn`.

use std::fmt;
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GlobalPoolOrigin {
    Explicit,
    Generic,
    Foreign,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GlobalPoolState {
    width: usize,
    origin: GlobalPoolOrigin,
}

/// Immutable first-touch scheduler authority. `OnceLock` serializes the one
/// Rayon build attempt and publishes origin plus actual width together.
static GLOBAL_POOL_STATE: OnceLock<Result<GlobalPoolState, Arc<rayon::ThreadPoolBuildError>>> =
    OnceLock::new();

/// Failure to establish Part A as the authority that creates Rayon global pool.
#[derive(Debug)]
pub enum GlobalPoolAuthorityError {
    ZeroThreads,
    PoolBuild(Arc<rayon::ThreadPoolBuildError>),
    NonAuthoritativePool { actual: usize },
    WidthMismatch { requested: usize, actual: usize },
}

impl fmt::Display for GlobalPoolAuthorityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroThreads => f.write_str("Part A Rayon width must be positive"),
            Self::PoolBuild(_) => f.write_str("building Part A Rayon global pool failed"),
            Self::NonAuthoritativePool { actual } => write!(
                f,
                "Part A cannot adopt a generic or foreign global Rayon pool at width {actual}"
            ),
            Self::WidthMismatch { requested, actual } => write!(
                f,
                "Part A requested Rayon width {requested} but global pool started at {actual}"
            ),
        }
    }
}

impl std::error::Error for GlobalPoolAuthorityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PoolBuild(source) => Some(source.as_ref()),
            Self::ZeroThreads | Self::NonAuthoritativePool { .. } | Self::WidthMismatch { .. } => {
                None
            }
        }
    }
}

// macOS: request P-core-first scheduling on each rayon worker. Rust std does
// not propagate the parent thread's QoS, so workers default to
// QOS_CLASS_DEFAULT and the AMP scheduler may place hot solver work on E-cores.
// Mirrors the oracle thread_pool start handler.
#[cfg(target_os = "macos")]
#[inline]
fn set_worker_qos_user_initiated() {
    // QOS_CLASS_USER_INITIATED = 0x19 (<sys/qos.h>). libc binds the fn but does
    // not re-export the constant at crate root, so bind the libSystem symbol
    // directly; qos_class_t is `unsigned int` (u32) at the C ABI.
    const QOS_CLASS_USER_INITIATED: u32 = 0x19;
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    // SAFETY: sets a thread-local QoS hint on the calling thread; always safe.
    unsafe {
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    }
}

/// Resolve pool width from caller authority or host parallelism.
#[inline]
fn resolve_worker_threads(worker_threads: Option<usize>) -> usize {
    if let Some(n) = worker_threads {
        return n.max(1);
    }
    std::thread::available_parallelism()
        .map_or(1, usize::from)
        .max(1)
}

fn build_global_pool(desired: usize) -> Result<(), rayon::ThreadPoolBuildError> {
    let builder = rayon::ThreadPoolBuilder::new()
        .num_threads(desired)
        .stack_size(crate::WORKER_STACK_BYTES)
        .thread_name(|i| format!("nd-sched-{i}"));

    #[cfg(target_os = "macos")]
    let builder = builder.start_handler(move |_thread_index| {
        set_worker_qos_user_initiated();
    });

    builder.build_global()
}

fn classify_global_pool_build_error(
    error: rayon::ThreadPoolBuildError,
) -> Result<(), Arc<rayon::ThreadPoolBuildError>> {
    if std::error::Error::source(&error).is_some() {
        Err(Arc::new(error))
    } else {
        Ok(())
    }
}

fn initialize_global_pool_state(
    desired: usize,
    owner: GlobalPoolOrigin,
) -> Result<GlobalPoolState, Arc<rayon::ThreadPoolBuildError>> {
    match build_global_pool(desired) {
        Ok(()) => Ok(GlobalPoolState {
            width: desired,
            origin: owner,
        }),
        // This builder never uses Rayon's current-thread mode. Under pinned
        // Rayon, its only source-less build failure is an existing global
        // pool. Spawn failures retain their io::Error source and must not be
        // mislabeled as foreign-pool authority failures.
        Err(error) => {
            classify_global_pool_build_error(error)?;
            Ok(GlobalPoolState {
                width: rayon::current_num_threads().max(1),
                origin: GlobalPoolOrigin::Foreign,
            })
        }
    }
}

fn generic_global_pool_width(
    initialization: &Result<GlobalPoolState, Arc<rayon::ThreadPoolBuildError>>,
) -> usize {
    initialization.as_ref().map_or_else(
        |source| {
            // Classifier creates `Err` only for sourceful spawn failures, so
            // this assertion is false on every valid path: it invokes the
            // panic hook with full source text. Abort is reachable only if
            // that invariant drifts; neither branch can return a fake width.
            let source_missing = std::error::Error::source(source.as_ref()).is_none();
            assert!(
                source_missing,
                "building Part A Rayon global pool failed: {source}"
            );
            std::process::abort()
        },
        |state| state.width,
    )
}

/// Build the single global rayon pool (idempotent).
///
/// The first call builds the process-global pool with 16 MiB worker stacks and,
/// on macOS, the user-initiated `QoS` start handler; subsequent calls are no-ops
/// that return the actual already-latched width. Sizing follows
/// [`resolve_worker_threads`]: explicit `worker_threads`, then host
/// `available_parallelism`. This generic path can never become Part A
/// authority, even if its width later matches an authoritative request.
///
/// Returns the latched pool width.
///
/// # Panics
///
/// Panics if Rayon cannot spawn the global pool. Part A callers that require
/// recoverable diagnostics must use [`init_global_pool_authoritative`].
#[must_use]
pub fn init_global_pool(worker_threads: Option<usize>) -> usize {
    generic_global_pool_width(GLOBAL_POOL_STATE.get_or_init(|| {
        initialize_global_pool_state(
            resolve_worker_threads(worker_threads),
            GlobalPoolOrigin::Generic,
        )
    }))
}

/// Create Rayon global pool under explicit Part A thread authority.
///
/// Unlike [`init_global_pool`], this path accepts only a pool created by a
/// prior authoritative call with the same width. Generic and foreign pools are
/// rejected even when their observed width matches `worker_threads`.
///
/// # Errors
///
/// Returns an error if width is zero, Rayon cannot spawn the pool, a
/// generic/foreign pool already exists, or an authoritative pool has a
/// different width.
pub fn init_global_pool_authoritative(worker_threads: usize) -> anyhow::Result<usize> {
    if worker_threads == 0 {
        return Err(GlobalPoolAuthorityError::ZeroThreads.into());
    }
    let state = match GLOBAL_POOL_STATE
        .get_or_init(|| initialize_global_pool_state(worker_threads, GlobalPoolOrigin::Explicit))
    {
        Ok(state) => state,
        Err(source) => {
            return Err(GlobalPoolAuthorityError::PoolBuild(Arc::clone(source)).into());
        }
    };
    match state.origin {
        GlobalPoolOrigin::Explicit if state.width == worker_threads => Ok(state.width),
        GlobalPoolOrigin::Explicit => Err(GlobalPoolAuthorityError::WidthMismatch {
            requested: worker_threads,
            actual: state.width,
        }
        .into()),
        GlobalPoolOrigin::Generic | GlobalPoolOrigin::Foreign => {
            Err(GlobalPoolAuthorityError::NonAuthoritativePool {
                actual: state.width,
            }
            .into())
        }
    }
}

/// Return the configured global Rayon-pool width without first-touching it.
///
/// Returns `Some` only for explicit scheduler authority. Generic or foreign
/// pools remain unconfigured for Part A. This never queries or initializes
/// Rayon.
#[inline]
#[must_use]
pub fn configured_global_pool_threads() -> Option<usize> {
    GLOBAL_POOL_STATE
        .get()
        .and_then(|initialization| match initialization {
            Ok(state) if state.origin == GlobalPoolOrigin::Explicit => Some(state.width),
            Ok(_) | Err(_) => None,
        })
}

/// Number of workers in the global pool, initializing it with defaults first if
/// it has not been built yet.
#[inline]
#[must_use]
pub fn num_threads() -> usize {
    init_global_pool(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_authority_error(result: anyhow::Result<usize>) -> GlobalPoolAuthorityError {
        result
            .expect_err("authoritative pool operation must fail")
            .downcast::<GlobalPoolAuthorityError>()
            .expect("scheduler error must retain its typed cause")
    }

    /// Re-run one of this module's tests in a fresh process, so it gets an
    /// unlatched global Rayon pool.
    ///
    /// Returns `true` in the child (run the assertions) and `false` in the
    /// parent (the child already did).
    ///
    /// `test_name` is an `--exact` filter, so it is a string that must keep
    /// matching a `#[test]` fn below. A stale name is NOT a failure by default:
    /// libtest reports "0 passed; 1 filtered out" and exits 0, the parent's
    /// `status.success()` holds, the helper returns `false`, and all three
    /// callers take their early return — so every case passes while asserting
    /// nothing, in BOTH processes. Requiring the child's own report binds the
    /// filter to a real test.
    fn run_pool_authority_child(test_name: &str, case: &str) -> bool {
        const CHILD_ENV: &str = "NASA_DUST_POOL_AUTHORITY_CHILD";

        if let Some(child_case) = std::env::var_os(CHILD_ENV) {
            assert_eq!(child_case, case);
            return true;
        }

        let output = std::process::Command::new(
            std::env::current_exe().expect("current Rust test executable"),
        )
        .args([test_name, "--exact", "--nocapture"])
        .env(CHILD_ENV, case)
        .env("RUST_TEST_THREADS", "1")
        .output()
        .expect("spawn isolated pool-authority child test");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "child failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        assert!(
            stdout.contains("running 1 test"),
            "child ran no test: `{test_name}` matched nothing under `--exact`, \
             so this case asserts nothing in either process\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        false
    }

    #[test]
    fn configured_pool_query_does_not_latch_before_explicit_initialization() {
        if !run_pool_authority_child(
            "pool::tests::configured_pool_query_does_not_latch_before_explicit_initialization",
            "configured-query",
        ) {
            return;
        }

        assert_eq!(configured_global_pool_threads(), None);
        assert_eq!(init_global_pool(Some(4)), 4);
        assert_eq!(configured_global_pool_threads(), None);
    }

    #[test]
    fn authoritative_failures_use_anyhow_and_keep_typed_causes() {
        fn assert_anyhow_result(_: anyhow::Result<usize>) {}

        assert_anyhow_result(init_global_pool_authoritative(0));
        let error = pool_authority_error(init_global_pool_authoritative(0));
        assert!(matches!(error, GlobalPoolAuthorityError::ZeroThreads));
    }

    #[test]
    fn rayon_spawn_failure_is_not_classified_as_foreign_pool() {
        let build_error = rayon::ThreadPoolBuilder::new()
            .spawn_handler(|_| Err(std::io::Error::other("hostile spawn failure")))
            .build()
            .expect_err("hostile spawn must fail");

        let retained = classify_global_pool_build_error(build_error)
            .expect_err("spawn failure must remain a build failure");
        let source = std::error::Error::source(&retained)
            .expect("Rayon build failure must retain spawn I/O source");
        assert_eq!(source.to_string(), "hostile spawn failure");
        assert_eq!(
            source
                .downcast_ref::<std::io::Error>()
                .map(std::io::Error::kind),
            Some(std::io::ErrorKind::Other)
        );

        let error = anyhow::Error::new(GlobalPoolAuthorityError::PoolBuild(retained));
        assert!(error.chain().any(|cause| cause
            .downcast_ref::<rayon::ThreadPoolBuildError>()
            .is_some()));
        assert!(error
            .chain()
            .any(|cause| cause.downcast_ref::<std::io::Error>().is_some()));
    }

    #[test]
    fn generic_pool_spawn_failure_panics_with_retained_source() {
        let build_error = rayon::ThreadPoolBuilder::new()
            .spawn_handler(|_| Err(std::io::Error::other("generic hostile spawn failure")))
            .build()
            .expect_err("hostile spawn must fail");
        let retained = classify_global_pool_build_error(build_error)
            .expect_err("spawn failure must remain a build failure");
        let initialization = Err(Arc::clone(&retained));

        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            generic_global_pool_width(&initialization)
        }))
        .expect_err("generic pool build failure must not return a width");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&'static str>().copied())
            .expect("panic payload must be a visible diagnostic string");
        assert!(message.contains("building Part A Rayon global pool failed"));
        assert!(
            message.contains("generic hostile spawn failure"),
            "nested spawn source missing from panic: {message}"
        );
    }

    #[test]
    fn authoritative_pool_rejects_prior_global_even_when_width_matches() {
        if !run_pool_authority_child(
            "pool::tests::authoritative_pool_rejects_prior_global_even_when_width_matches",
            "foreign",
        ) {
            return;
        }

        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build_global()
            .expect("seed foreign global pool");
        assert!(matches!(
            pool_authority_error(init_global_pool_authoritative(2)),
            GlobalPoolAuthorityError::NonAuthoritativePool { actual: 2 }
        ));
    }

    #[test]
    fn authoritative_pool_rejects_generic_latched_pool_even_when_width_matches() {
        if !run_pool_authority_child(
            "pool::tests::authoritative_pool_rejects_generic_latched_pool_even_when_width_matches",
            "generic",
        ) {
            return;
        }

        assert_eq!(init_global_pool(Some(2)), 2);
        assert!(matches!(
            pool_authority_error(init_global_pool_authoritative(2)),
            GlobalPoolAuthorityError::NonAuthoritativePool { actual: 2 }
        ));
    }

    #[test]
    fn authoritative_pool_builds_requested_width_in_fresh_process() {
        if !run_pool_authority_child(
            "pool::tests::authoritative_pool_builds_requested_width_in_fresh_process",
            "fresh",
        ) {
            return;
        }

        assert!(matches!(init_global_pool_authoritative(2), Ok(2)));
        assert!(matches!(init_global_pool_authoritative(2), Ok(2)));
        assert_eq!(num_threads(), 2);
    }

    #[test]
    fn global_pool_reports_positive_width_and_is_idempotent() {
        let w1 = init_global_pool(None);
        let w2 = init_global_pool(Some(1)); // second call cannot change the width
        assert!(w1 >= 1, "pool width must be positive");
        assert_eq!(w1, w2, "the global pool width latches on first init");
        assert_eq!(num_threads(), w1);
    }

    #[test]
    fn explicit_argument_or_available_parallelism_resolves_width() {
        assert_eq!(resolve_worker_threads(Some(4)), 4);
        assert!(resolve_worker_threads(None) >= 1);
    }
}
