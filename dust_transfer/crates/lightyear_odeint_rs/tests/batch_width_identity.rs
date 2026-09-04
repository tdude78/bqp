//! Bit identity across the two arms of `should_use_parallel_batch`.
//!
//! # Why this file exists
//!
//! `batch::should_use_parallel_batch` selects a branch from AMBIENT EXECUTION
//! CONTEXT, not from inputs. All three of its conditions have to hold at once:
//! at least `LIGHTYEAR_PAR_THRESHOLD` items, a caller that is NOT already on a
//! Rayon worker, and an explicitly configured global pool wider than one
//! thread. Miss any one and the same call takes the serial arm instead.
//!
//! That is only safe while the two arms are numerically identical, and nothing
//! in the type system says they are. The arms are not merely a reschedule of
//! each other: the parallel one runs `par_chunks_exact` through
//! `try_for_each_init`, which constructs a **separate**
//! `ReusableFinalNoEventIntegrator` per split job — so the number of integrator
//! instances, and therefore which propagation lands on a warm versus a freshly
//! built one, is a runtime work-stealing decision. The serial arm builds one
//! integrator for the whole batch. Reuse is supposed to be invisible
//! (`propagate` calls `reset_for_propagation` on entry), and this test is what
//! makes that a measurement rather than an argument.
//!
//! Before this file, no test in the workspace constructed a thread pool for
//! this gate at all: `ThreadPoolBuilder` appears in ten files and none of them
//! exercise `integrate_batch_native`. The parallel arm was unexecuted.
//!
//! # Why a child process
//!
//! The gate reads `nd_sched::configured_global_pool_threads()`, and the global
//! pool is process-wide and set once. Forcing a width in-process would either
//! fight whatever another test installed first or silently no-op, so the width
//! is set in a re-exec'd child. `dust_estimates_rs::parallel_branch_identity`
//! took the same approach for the same reason; it was deleted on 2026-08-06
//! with the probabilistic GMM dust-mass search (`docs/REFACTOR_BLOCKLIST.md`
//! B4). `two_phase_transfer_rs/tests/width_identity.rs` re-execs children the
//! same way (and predates this file), so the pattern lives in two places --
//! an earlier revision of this note claimed it was the last one standing.

use anyhow::Context as _;
use lightyear_odeint_rs::batch::should_use_parallel_batch;
use lightyear_odeint_rs::types::ForceConfig;
use satpy_core::{eci2equinoc_impl_f64, kep2eci_impl};

/// Width forced in the child. Above `LIGHTYEAR_PAR_THRESHOLD`'s pool
/// requirement and small enough to be sane on a laptop or a CI box.
const CHILD_POOL_WIDTH: &str = "4";
const CHILD_WIDTH_ENV: &str = "ND_BATCH_WIDTH_IDENTITY_CHILD";
const CHILD_MARKER: &str = "BATCH_WIDTH_IDENTITY_CHILD_RAN_BOTH_ARMS";

/// Sigma-like arcs. Must be at least `LIGHTYEAR_PAR_THRESHOLD` (32) or the gate
/// stays serial regardless of pool width.
const ARCS: usize = 40;
const JD0: f64 = 2_460_310.5;
const TOF_S: f64 = 600.0;
const EPS: f64 = 1e-9;

/// The compiled stepper, resolved rather than restated.
///
/// This gate proves the parallel and serial batch arms are BIT-IDENTICAL. A
/// hardcoded stepper proves it for a stepper the campaign may not fly: this
/// file held `Vern9` across the Vern9 -> Vern7 swap at 8ee9fdf, so the arm
/// identity production actually depends on went unproven while the gate stayed
/// green. Same defect `prop_timing::authority_stepper` and
/// `tolerance_cost_accuracy::authority_stepper` close. Nothing here is pinned
/// to a value -- the two arms are compared to each other -- so following the
/// token changes what is proven, never a digest.
fn authority_stepper() -> anyhow::Result<lightyear_odeint_rs::types::StepperMethod> {
    use lightyear_odeint_rs::types::StepperMethod;
    match nd_config::CompiledPartAScienceV1::part_a_v1()
        .hybrid()
        .integrator_method
    {
        "vern7" => Ok(StepperMethod::Vern7),
        "vern9" => Ok(StepperMethod::Vern9),
        other => {
            anyhow::bail!("compiled science selects a stepper this file does not build: {other}")
        }
    }
}

fn dust_config() -> anyhow::Result<ForceConfig> {
    let flags = lightyear_odeint_rs::types::ForceFlags::DRAG
        | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
        | lightyear_odeint_rs::types::ForceFlags::MOON_GRAVITY;
    let config = ForceConfig {
        sph_order: 5,
        force_flags: flags,
        subtract_first_order: true,
        atm_model: 4,
        am_ratio: 0.02,
        cd: 2.2,
        cr: 1.3,
        dt_max: 60.0,
        eps: EPS,
        integrator_method: authority_stepper()?,
        ..ForceConfig::default()
    }
    .with_ephemeris_for_arc(JD0, JD0 + TOF_S / satpy_core::SEC_PER_DAY)
    .context("ephemeris and JB2008 assets must cover the test arc")?;
    Ok(config)
}

/// `ARCS` distinct one-ULP neighbours of one orbit: distinct arcs, same orbit.
fn sigma_states() -> anyhow::Result<Vec<f64>> {
    let kep = [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0];
    let mut init_eci = [0.0; 6];
    kep2eci_impl(&kep, true, 0.0, 0.0, true, &mut init_eci);
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let mut states = Vec::with_capacity(ARCS * 6);
    for k in 0..ARCS {
        let mut equ = init_equ;
        let offset = u64::try_from(k).context("sigma index must fit u64")?;
        let bits = equ[0]
            .to_bits()
            .checked_add(offset)
            .context("sigma neighbour overflows")?;
        equ[0] = f64::from_bits(bits);
        states.extend_from_slice(&equ);
    }
    Ok(states)
}

fn run_batch(states: &[f64], config: &ForceConfig) -> anyhow::Result<Vec<f64>> {
    lightyear_odeint_rs::integrate_batch_native(lightyear_odeint_rs::BatchPropagationRequest {
        initial_equinoc_states: states,
        t_eval: &[TOF_S],
        t0_s: 0.0,
        t_final_s: TOF_S,
        epoch_jd: JD0,
        force_config: *config,
        ballistics: lightyear_odeint_rs::BatchBallistics {
            am_ratio: None,
            cd: None,
            cr: None,
        },
    })
    .context("batch propagation must succeed")
}

/// Run `f` on a Rayon worker so `current_thread_index()` is `Some` and the gate
/// returns false: the SERIAL arm.
fn run_nested<T: Send>(f: impl FnOnce() -> T + Send) -> Option<T> {
    let mut slot = None;
    rayon::scope(|scope| {
        scope.spawn(|_| {
            slot = Some(f());
        });
    });
    slot
}

#[test]
fn batch_parallel_and_serial_arms_are_bit_identical() {
    let Some(width) = std::env::var_os(CHILD_WIDTH_ENV) else {
        // Parent: re-exec ourselves with the width forced.
        let executable = std::env::current_exe().expect("test executable must resolve");
        let output = std::process::Command::new(executable)
            .args([
                "batch_parallel_and_serial_arms_are_bit_identical",
                "--exact",
                "--nocapture",
            ])
            .env(CHILD_WIDTH_ENV, CHILD_POOL_WIDTH)
            .env("RUST_TEST_THREADS", "1")
            .output()
            .expect("spawning the width-forced child must succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "child failed at width {CHILD_POOL_WIDTH}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        // Without this the parent passes whenever the child exits 0 for any
        // reason at all, including having filtered out every test.
        assert!(
            stdout.contains(CHILD_MARKER),
            "child exited 0 without running both arms\nstdout:\n{stdout}"
        );
        return;
    };

    let width: usize = width
        .to_string_lossy()
        .parse()
        .expect("child width must be numeric");
    // `init_global_pool` is NOT enough: it creates a pool with `Generic`
    // origin, and `configured_global_pool_threads()` returns `None` for those,
    // so the gate stays serial no matter how wide the pool is. Only the
    // authoritative entry point -- the one the Part A scheduler uses -- marks
    // the pool `Explicit`. That subtlety is a large part of why this arm went
    // unexecuted: a test that merely builds a wide pool still misses it.
    let installed = nd_sched::init_global_pool_authoritative(width)
        .expect("authoritative global pool must initialise");
    assert_eq!(installed, width);
    assert_eq!(rayon::current_num_threads(), width);
    assert_eq!(
        nd_sched::configured_global_pool_threads(),
        Some(width),
        "the pool must register as explicitly configured, or the gate cannot fire"
    );

    // Both arms execute direct propagation. The census below proves the
    // comparison is non-vacuous rather than observing a reused final result.
    lightyear_odeint_rs::load_constants_from_bytes(
        include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt"),
        5,
    )
    .expect("gravity coefficients must load");
    let config = dust_config().expect("force config must build");
    let states = sigma_states().expect("sigma states must build");

    // Non-vacuity, checked BEFORE the comparison: the gate must actually
    // disagree between these two contexts, or the assertion below is comparing
    // one arm with itself and would pass forever while proving nothing.
    assert!(
        should_use_parallel_batch(ARCS),
        "top level at width {width} must select the PARALLEL arm"
    );
    assert_eq!(
        run_nested(|| should_use_parallel_batch(ARCS)),
        Some(false),
        "a nested call must select the SERIAL arm"
    );

    let parallel = run_batch(&states, &config).expect("parallel arm must propagate");
    let serial = run_nested(|| run_batch(&states, &config))
        .expect("nested batch must return")
        .expect("nested batch must propagate");

    assert_eq!(
        parallel.len(),
        serial.len(),
        "the two arms must produce the same output shape"
    );
    assert_eq!(parallel.len(), ARCS * 6, "one final state per arc");
    for (index, (parallel_value, serial_value)) in parallel.iter().zip(&serial).enumerate() {
        assert_eq!(
            parallel_value.to_bits(),
            serial_value.to_bits(),
            "element {index} differs between the parallel and serial arms \
             (parallel={parallel_value:e}, serial={serial_value:e}). The two arms of \
             `should_use_parallel_batch` must be bit-identical, or `--threads` is \
             silently a science parameter."
        );
    }
    // Non-vacuity: an all-zero or empty result would satisfy the loop above.
    assert!(
        parallel.iter().all(|value| value.is_finite()),
        "the propagation must have produced finite states"
    );
    assert!(
        parallel.iter().any(|value| *value != 0.0),
        "the propagation must have produced non-trivial states"
    );

    println!("{CHILD_MARKER}");
}
