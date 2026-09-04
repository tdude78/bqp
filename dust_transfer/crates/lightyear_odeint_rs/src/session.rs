//! Stateful session for reduced FFI overhead in batch integration.
//!
//! This module provides `LightyearSession`, which caches integrator configuration
//! and spherical harmonics coefficients to avoid repeated lookups and parsing.

use rayon::prelude::*;
use satpy_core::{cross3, eci2equinoc_impl_f64, equinoc_prop_from_impl, norm3};
#[cfg(test)]
use std::collections::HashSet;
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;

use crate::batch::should_use_parallel_batch;
#[cfg(test)]
use crate::integrator::ScalarGravityAssets;
use crate::integrator::{
    integrate_final_checked, validate_scalar_stepper_authority, FinalPropagationFailure,
    ReusableFinalCheckedIntegrator, ReusableFinalNoEventIntegrator, ScalarPropagationContext,
    ScalarPropagationRequest,
};
#[cfg(feature = "scalar-leg-observer")]
use crate::integrator::{
    ObservedFinalLeg, ObservedFinalMetricError, ObservedFinalMetrics, ObservedSolverTerminalStatus,
};
use crate::types::ForceConfig;

/// A stateful session that caches integrator configuration and spherical harmonics coefficients.
///
/// This reduces per-call overhead by avoiding repeated config parsing and coefficient lookups.
/// The session is immutable after construction (except for per-call sun/moon positions).
///
/// # Workspace Management
/// Use `create_workspace()` to pre-allocate output buffers, then pass them to
/// `integrate_batch_into()` to avoid allocation overhead in UKF loops.
pub struct LightyearSession {
    /// Immutable force authority and gravity assets for every session propagation.
    context: ScalarPropagationContext,
}

impl LightyearSession {
    /// Construct a session from one immutable scalar propagation authority.
    ///
    /// The context owns only immutable force and gravity assets. Per-propagation
    /// RHS state remains inside the propagation path, so Encke history is never
    /// shared or retained by this session.
    #[must_use]
    pub const fn from_context(context: ScalarPropagationContext) -> Self {
        Self { context }
    }

    /// Get the spherical harmonics order.
    #[must_use]
    pub fn sph_order(&self) -> usize {
        self.context.config.sph_order
    }

    /// Get the Julian date at epoch.
    #[must_use]
    pub const fn jd0(&self) -> f64 {
        self.context.jd0
    }

    /// Get configured maximum integrator step size in seconds.
    #[must_use]
    pub fn dt_max(&self) -> f64 {
        self.context.config.dt_max
    }

    /// Get the integration error tolerance.
    #[must_use]
    pub fn eps(&self) -> f64 {
        self.context.config.eps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GlobalCoeffs, GLOBAL_COEFFS};
    use crate::types::StepperMethod;
    use satpy_core::{pack_gravity_coeffs, PackedGravityCoeffs};
    use std::process::Command;
    use std::sync::Arc;

    const TEST_STACK_SIZE: usize = 64 * 1024 * 1024;
    const GLOBAL_POOL_CHILD_ENV: &str = "NASA_DUST_GLOBAL_POOL_TEST_CHILD";
    const GLOBAL_POOL_CHILD_MARKER: &str = "NASA_DUST_GLOBAL_POOL_TEST_CHILD_RAN";
    const GLOBAL_POOL_CHILD_TEST: &str =
        "session::tests::variable_final_native_batch_uses_global_rayon_pool_without_semantic_drift";
    const PARALLEL_BATCH_CHILD_ENV: &str = "NASA_DUST_PARALLEL_BATCH_TEST_CHILD";
    const PARALLEL_BATCH_CHILD_MARKER: &str = "NASA_DUST_PARALLEL_BATCH_TEST_CHILD_RAN";
    const PARALLEL_BATCH_CHILD_TEST: &str =
        "session::tests::variable_final_native_parallel_batch_matches_serial_bits_and_uses_width";

    #[derive(Debug)]
    struct VariableFinalMarker;

    impl std::fmt::Display for VariableFinalMarker {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("variable-final marker")
        }
    }

    impl std::error::Error for VariableFinalMarker {}

    #[test]
    fn variable_final_authority_error_keeps_source_and_diagnostic() -> anyhow::Result<()> {
        let error = anyhow::Error::new(VariableFinalNativeError::UnsupportedStepper(
            anyhow::Error::new(VariableFinalMarker),
        ));
        anyhow::ensure!(
            error
                .chain()
                .any(|cause| cause.downcast_ref::<VariableFinalMarker>().is_some()),
            "variable-final source missing from error chain"
        );
        anyhow::ensure!(error.to_string() == "variable-final marker");
        Ok(())
    }

    fn run_with_stack<F: FnOnce() + Send + 'static>(f: F) {
        std::thread::Builder::new()
            .stack_size(TEST_STACK_SIZE)
            .spawn(f)
            .expect("failed to spawn test thread")
            .join()
            .expect("test thread panicked");
    }

    fn test_usize_as_f64(value: usize, context: &str) -> f64 {
        f64::from(u32::try_from(value).expect(context))
    }

    fn test_state_row(values: &[f64], row: usize) -> &[f64] {
        let start = row
            .checked_mul(6)
            .expect("test state-row offset must not overflow");
        let end = start
            .checked_add(6)
            .expect("test state-row end must not overflow");
        values
            .get(start..end)
            .expect("test fixture must contain a complete state row")
    }

    fn test_state_row_mut(values: &mut [f64], row: usize) -> &mut [f64] {
        let start = row
            .checked_mul(6)
            .expect("test state-row offset must not overflow");
        let end = start
            .checked_add(6)
            .expect("test state-row end must not overflow");
        values
            .get_mut(start..end)
            .expect("test fixture must contain a complete state row")
    }

    #[test]
    fn variable_final_model4_and_model5_reject_auto_and_implicit_before_empty_batch_returns() {
        let _coeffs = install_test_coeffs(0);
        for model in [4, 5] {
            for stepper in [StepperMethod::Esdirk43, StepperMethod::Auto] {
                let config = ForceConfig {
                    atm_model: model,
                    integrator_method: stepper,
                    eps: 1.0e-8,
                    ..Default::default()
                };
                let session = test_session(2_460_310.5, config);
                let mut output = [];
                let error = session
                    .integrate_variable_final_into(
                        VariableFinalBatchRequest {
                            initial_eci_states: &[],
                            epoch_jd: &[],
                            final_time_s: &[],
                            t0_s: 0.0,
                            ballistics: VariableFinalBallistics::default(),
                        },
                        &mut output,
                    )
                    .expect_err("guarded HF method must fail before zero-row return");
                assert!(
                    error
                        .to_string()
                        .contains("requires explicit scalar method"),
                    "model={model} stepper={stepper:?}: {error}"
                );
            }
        }
    }

    /// Publishes a synthetic pack to `GLOBAL_COEFFS` and returns it TOGETHER
    /// with the guard that serialises this test against every other publishing
    /// test in this binary (see `config::lock_global_coeffs_for_test`). Bind
    /// the result to a named variable so the guard lives for the whole test:
    /// `test_session` reads the global back and must see THIS install.
    fn install_test_coeffs(
        order: usize,
    ) -> (std::sync::MutexGuard<'static, ()>, Arc<PackedGravityCoeffs>) {
        let guard = crate::config::lock_global_coeffs_for_test();
        let stride = order
            .checked_add(2)
            .expect("test coefficient stride must not overflow");
        let total_size = stride
            .checked_mul(stride)
            .expect("test coefficient array length must not overflow");
        let mut c_coeffs = vec![0.0; total_size];
        let mut s_coeffs = vec![0.0; total_size];
        *c_coeffs
            .get_mut(0)
            .expect("test coefficient array must contain C[0,0]") = 1.0;
        for l in 2..=order {
            let base = l
                .checked_mul(stride)
                .expect("test coefficient row offset must not overflow");
            *c_coeffs
                .get_mut(base)
                .expect("test coefficient array must contain degree term") =
                1e-3 / test_usize_as_f64(l, "test degree must fit u32").powi(2);
            for m in 1..=l {
                let degree_order = l
                    .checked_mul(m)
                    .expect("test coefficient degree-order product must not overflow");
                let magnitude =
                    1e-6 / test_usize_as_f64(degree_order, "test degree-order must fit u32");
                let coefficient_index = base
                    .checked_add(m)
                    .expect("test coefficient index must not overflow");
                *c_coeffs
                    .get_mut(coefficient_index)
                    .expect("test coefficient array must contain cosine term") = magnitude;
                *s_coeffs
                    .get_mut(coefficient_index)
                    .expect("test coefficient array must contain sine term") = magnitude * 0.5;
            }
        }
        let packed = Arc::new(
            pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order)
                .expect("session test gravity coefficients must pack"),
        );
        GLOBAL_COEFFS.store(Arc::new(GlobalCoeffs::Loaded(Arc::clone(&packed))));
        (guard, packed)
    }

    fn test_session(jd0: f64, mut config: ForceConfig) -> LightyearSession {
        config.subtract_first_order |= config.sph_order > 0;
        let coefficients = GLOBAL_COEFFS.load();
        let packed = coefficients
            .loaded_snapshot()
            .expect("session test must install gravity coefficients first");
        LightyearSession::from_context(ScalarPropagationContext::new(
            jd0,
            Arc::new(config),
            ScalarGravityAssets::new(packed),
        ))
    }

    #[test]
    fn session_from_scalar_context_keeps_force_authority() {
        let (_global_coeffs_lock, packed) = install_test_coeffs(5);
        let jd0 = 2_460_310.5;
        let config = ForceConfig {
            sph_order: 5,
            ..ForceConfig::default()
        };
        let context =
            ScalarPropagationContext::new(jd0, Arc::new(config), ScalarGravityAssets::new(packed));

        let session = LightyearSession::from_context(context);

        assert_eq!(session.jd0().to_bits(), jd0.to_bits());
        assert_eq!(session.sph_order(), 5);
    }

    #[test]
    fn variable_final_context_binds_row_authority_directly() {
        let (_global_coeffs_lock, packed) = install_test_coeffs(5);
        let session = LightyearSession::from_context(ScalarPropagationContext::new(
            2_460_310.5,
            Arc::new(ForceConfig {
                sph_order: 5,
                ..ForceConfig::default()
            }),
            ScalarGravityAssets::new(Arc::clone(&packed)),
        ));
        let row_epoch_jd = 2_460_311.25;
        let row_config = Arc::new(ForceConfig {
            sph_order: 3,
            ..ForceConfig::default()
        });

        let row_context = session.scalar_propagation_context(row_epoch_jd, Arc::clone(&row_config));

        assert_eq!(row_context.jd0.to_bits(), row_epoch_jd.to_bits());
        assert!(Arc::ptr_eq(&row_context.config, &row_config));
        assert!(Arc::ptr_eq(&row_context.gravity.packed, &packed));
    }

    fn create_test_eci_points(n_sigma: usize) -> Vec<f64> {
        let base_states = [
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            [7000.0, 100.0, -50.0, -0.1, 7.45, 0.2],
            [7200.0, -120.0, 80.0, 0.05, 7.30, -0.15],
        ];

        let state_values = n_sigma
            .checked_mul(6)
            .expect("test state vector length must not overflow");
        let mut init_states_flat = Vec::with_capacity(state_values);
        for (idx, base_state) in base_states.iter().cycle().take(n_sigma).enumerate() {
            let mut eci = *base_state;
            let index_value = test_usize_as_f64(idx, "test state index must fit u32");
            let count_value = test_usize_as_f64(n_sigma, "test state count must fit u32");
            let half_count = count_value * 0.5;
            let centered_index = index_value - half_count;
            let scale = 1e-6 * centered_index;
            for (axis, value) in eci.iter_mut().enumerate() {
                *value += scale * (test_usize_as_f64(axis, "test axis must fit u32") + 1.0);
            }
            init_states_flat.extend_from_slice(&eci);
        }
        init_states_flat
    }

    #[test]
    fn session_nonzero_harmonics_always_subtract_two_body_baseline() {
        run_with_stack(|| {
            let _coeffs = install_test_coeffs(5);
            let session = test_session(
                2_460_000.25,
                ForceConfig {
                    sph_order: 5,
                    eps: 1.0e-8,
                    integrator_method: StepperMethod::Dopri5Compat,
                    ..ForceConfig::default()
                },
            );
            assert!(session.context.config.subtract_first_order);
        });
    }

    #[test]
    fn session_dt_max_getter_returns_the_configured_value() {
        // Pins the getter round trip only. It does NOT exercise the per-epoch
        // rebuild (`config_for_jd_mid`), which is where a variable-final
        // propagation could drop the requested dt_max.
        run_with_stack(|| {
            let _coeffs = install_test_coeffs(5);
            let requested_dt_max_s = 17.25;
            let session = test_session(
                2_460_000.25,
                ForceConfig {
                    sph_order: 0,
                    eps: 1.0e-8,
                    dt_max: requested_dt_max_s,
                    integrator_method: StepperMethod::Dopri5Compat,
                    ..ForceConfig::default()
                },
            );
            assert_eq!(session.dt_max().to_bits(), requested_dt_max_s.to_bits());
        });
    }

    #[test]
    fn variable_final_ephemeris_preflight_uses_each_rows_absolute_endpoints() {
        run_with_stack(|| {
            let _ephem_guard = crate::precomputed_ephem::ephemeris_test_guard();
            let _coeffs = install_test_coeffs(5);
            let flags = crate::types::ForceFlags::SUN_GRAVITY;
            crate::precomputed_ephem::load_precomputed_ephemeris(flags)
                .expect("typed Sun ephemeris catalogue must load");
            let ephem = crate::precomputed_ephem::get_precomputed_ephemeris()
                .expect("typed Sun ephemeris catalogue must publish");
            let (start, end) = ephem
                .get(crate::precomputed_ephem::Body::Sun)
                .expect("sun catalogue")
                .jd_range();
            let session = test_session(
                start,
                ForceConfig {
                    sph_order: 5,
                    force_flags: flags,
                    eps: 1.0e-8,
                    integrator_method: StepperMethod::Dopri5Compat,
                    ..ForceConfig::default()
                },
            );
            let states = create_test_eci_points(2);
            let mut output = [f64::NAN; 12];
            session
                .preflight_variable_final_ephemeris(&VariableFinalBatchRequest {
                    initial_eci_states: &states,
                    epoch_jd: &[start + 1.0, end],
                    final_time_s: &[-86_400.0, 0.0],
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                })
                .expect("inclusive backward and zero-duration rows must pass");

            let one_ulp_after_end = f64::from_bits(end.to_bits() + 1);
            let error = session
                .integrate_variable_final_into(
                    VariableFinalBatchRequest {
                        initial_eci_states: &states,
                        epoch_jd: &[start + 1.0, one_ulp_after_end],
                        final_time_s: &[-86_400.0, 0.0],
                        t0_s: 0.0,
                        ballistics: VariableFinalBallistics::default(),
                    },
                    &mut output,
                )
                .expect_err("second row starts one ULP outside catalogue");
            let VariableFinalNativeError::Ephemeris { row, source } = error else {
                panic!("second row must fail ephemeris preflight")
            };
            assert_eq!(row, 1);
            assert!(source.to_string().contains("coverage failure"), "{source}");
        });
    }

    #[test]
    fn variable_final_native_batch_rejects_any_hidden_ground_crossing() {
        run_with_stack(|| {
            let _coeffs = install_test_coeffs(5);
            let jd0 = 2_460_310.5;
            let eps = 1.0e-8;
            let session = test_session(
                jd0,
                ForceConfig {
                    sph_order: 0,
                    eps,
                    integrator_method: StepperMethod::Vern9,
                    ..ForceConfig::default()
                },
            );

            let period_s = 4_920.0;
            let mu = crate::types::MU;
            let safe_radius_km = 7_000.0;
            let safe_speed_km_s = (mu / safe_radius_km).sqrt();
            let semi_major_km = (mu * (period_s / (2.0 * std::f64::consts::PI)).powi(2)).cbrt();
            let impact_speed_km_s = (mu * (2.0 / safe_radius_km - 1.0 / semi_major_km)).sqrt();
            let twice_semi_major_km = 2.0 * semi_major_km;
            let perigee_km = twice_semi_major_km - safe_radius_km;
            assert!(perigee_km < crate::types::RE + crate::types::GROUND_ALTITUDE);

            let states = [
                safe_radius_km,
                0.0,
                0.0,
                0.0,
                safe_speed_km_s,
                0.0,
                safe_radius_km,
                0.0,
                0.0,
                0.0,
                impact_speed_km_s,
                0.0,
            ];
            let epochs = [jd0, jd0];
            let tofs = [period_s, period_s];
            let mut out = [f64::NAN; 12];

            let failure = session
                .integrate_variable_final_into(
                    VariableFinalBatchRequest {
                        initial_eci_states: &states,
                        epoch_jd: &epochs,
                        final_time_s: &tofs,
                        t0_s: 0.0,
                        ballistics: VariableFinalBallistics::default(),
                    },
                    &mut out,
                )
                .expect_err("one impacting sigma trajectory must fail the whole native batch");

            assert!(
                matches!(
                    failure,
                    VariableFinalNativeError::Row {
                        row: 1,
                        failure: VariableFinalRowFailure::MinimumRadiusViolation,
                    }
                ),
                "unexpected typed failure: {failure:?}"
            );
            let message = failure.to_string();
            assert!(message.contains("perigee altitude"), "{message}");
            assert!(message.contains("minimum"), "{message}");
            assert!(out
                .get(..6)
                .expect("test output must contain first state row")
                .iter()
                .all(|value| value.is_finite()));
            assert!(out
                .get(6..)
                .expect("test output must contain second state row")
                .iter()
                .all(|value| value.is_nan()));
        });
    }

    #[test]
    fn variable_final_native_parallel_batch_matches_serial_bits_and_uses_width() {
        if std::env::var_os(PARALLEL_BATCH_CHILD_ENV).is_none() {
            let output =
                Command::new(std::env::current_exe().expect("current Rust test executable"))
                    .args([PARALLEL_BATCH_CHILD_TEST, "--exact", "--nocapture"])
                    .env(PARALLEL_BATCH_CHILD_ENV, "1")
                    .env("RUST_TEST_THREADS", "1")
                    .output()
                    .expect("spawn isolated parallel-batch child test");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "parallel-batch child failed\nstdout:\n{stdout}\nstderr:\n{stderr}",
            );
            assert!(
                stdout.contains(PARALLEL_BATCH_CHILD_MARKER),
                "parallel-batch child filter did not execute target test\nstdout:\n{stdout}\nstderr:\n{stderr}",
            );
            eprintln!("verified child marker: {PARALLEL_BATCH_CHILD_MARKER}");
            return;
        }

        println!("{PARALLEL_BATCH_CHILD_MARKER}");
        run_with_stack(|| {
            assert_eq!(
                nd_sched::init_global_pool_authoritative(2)
                    .expect("isolated test must establish authoritative W2 global pool"),
                2
            );
            let _ephem_guard = crate::precomputed_ephem::ephemeris_test_guard();
            let _coeffs = install_test_coeffs(5);
            crate::precomputed_ephem::load_precomputed_ephemeris(
                crate::types::ForceFlags::SUN_GRAVITY
                    | crate::types::ForceFlags::MOON_GRAVITY
                    | crate::types::ForceFlags::JUPITER_GRAVITY
                    | crate::types::ForceFlags::VENUS_GRAVITY,
            )
            .expect("all default typed ephemeris catalogues must load");
            let jd0 = 2_460_310.5;
            let n_rows = crate::batch::LIGHTYEAR_PAR_THRESHOLD;
            assert!(n_rows >= 2);
            assert!(nd_sched::num_threads() >= 2);
            let session = test_session(
                jd0,
                ForceConfig {
                    sph_order: 5,
                    force_flags: crate::types::ForceFlags::DRAG | crate::types::ForceFlags::SRP,
                    atm_model: 3,
                    am_ratio: 0.01,
                    cd: 2.2,
                    cr: 1.3,
                    eps: 1.0e-8,
                    dt_max: 60.0,
                    integrator_method: StepperMethod::Dopri5Compat,
                    ..ForceConfig::default()
                },
            );
            assert_eq!(session.sph_order(), 5);
            assert_eq!(
                session.context.config.force_flags,
                crate::types::ForceFlags::DRAG | crate::types::ForceFlags::SRP
            );

            let states = create_test_eci_points(n_rows);
            let epochs = vec![jd0; n_rows];
            let tofs = vec![300.0; n_rows];
            let am = vec![0.01; n_rows];
            let cd = vec![2.2; n_rows];
            let cr = vec![1.3; n_rows];
            let mut serial = vec![f64::NAN; n_rows * 6];
            for row in 0..n_rows {
                let state = test_state_row(&states, row);
                let epoch = epochs
                    .get(row)
                    .expect("test epoch vector must contain every row");
                let tof = tofs
                    .get(row)
                    .expect("test TOF vector must contain every row");
                let area_mass = am.get(row).expect("test AM vector must contain every row");
                let drag = cd
                    .get(row)
                    .expect("test drag vector must contain every row");
                let reflectivity = cr
                    .get(row)
                    .expect("test reflectivity vector must contain every row");
                let serial_row = test_state_row_mut(&mut serial, row);
                session
                    .integrate_variable_final_into(
                        VariableFinalBatchRequest {
                            initial_eci_states: state,
                            epoch_jd: std::slice::from_ref(epoch),
                            final_time_s: std::slice::from_ref(tof),
                            t0_s: 0.0,
                            ballistics: VariableFinalBallistics {
                                am_ratio: Some(std::slice::from_ref(area_mass)),
                                cd: Some(std::slice::from_ref(drag)),
                                cr: Some(std::slice::from_ref(reflectivity)),
                            },
                        },
                        serial_row,
                    )
                    .expect("serial row");
            }

            let mut parallel = vec![f64::NAN; n_rows * 6];
            variable_final_native_test_observer_begin(parallel.as_ptr().addr());
            session
                .integrate_variable_final_into(
                    VariableFinalBatchRequest {
                        initial_eci_states: &states,
                        epoch_jd: &epochs,
                        final_time_s: &tofs,
                        t0_s: 0.0,
                        ballistics: VariableFinalBallistics {
                            am_ratio: Some(&am),
                            cd: Some(&cd),
                            cr: Some(&cr),
                        },
                    },
                    &mut parallel,
                )
                .expect("parallel batch");
            let observed = variable_final_native_test_observer_take();

            assert!(
                observed.parallel_branch_entered,
                "native batch stayed serial: {:?}",
                observed.threads
            );
            assert_eq!(
                parallel
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                serial
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        });
    }

    #[test]
    fn variable_final_native_parallel_batch_returns_lowest_index_typed_failure() {
        run_with_stack(|| {
            let _coeffs = install_test_coeffs(5);
            let jd0 = 2_460_310.5;
            let n_rows = crate::batch::LIGHTYEAR_PAR_THRESHOLD;
            let session = test_session(
                jd0,
                ForceConfig {
                    sph_order: 0,
                    eps: 1.0e-8,
                    integrator_method: StepperMethod::Vern9,
                    ..ForceConfig::default()
                },
            );
            let mut states = create_test_eci_points(n_rows);
            let hostile_state = test_state_row_mut(&mut states, 2);
            *hostile_state
                .get_mut(0)
                .expect("test state row must have an x component") = 6_300.0;
            hostile_state
                .get_mut(1..)
                .expect("test state row must have velocity components")
                .fill(0.0);
            let epochs = vec![jd0; n_rows];
            let tofs = vec![60.0; n_rows];
            let mut am = vec![0.0; n_rows];
            *am.get_mut(7)
                .expect("test AM vector must contain hostile row") = f64::NAN;
            let mut out = vec![f64::NAN; n_rows * 6];

            let failure = session
                .integrate_variable_final_into(
                    VariableFinalBatchRequest {
                        initial_eci_states: &states,
                        epoch_jd: &epochs,
                        final_time_s: &tofs,
                        t0_s: 0.0,
                        ballistics: VariableFinalBallistics {
                            am_ratio: Some(&am),
                            cd: None,
                            cr: None,
                        },
                    },
                    &mut out,
                )
                .expect_err("two hostile rows must fail batch");

            assert!(matches!(
                failure,
                VariableFinalNativeError::Row {
                    row: 2,
                    failure: VariableFinalRowFailure::MinimumRadiusViolation,
                }
            ));
        });
    }

    #[test]
    fn variable_final_session_preserves_typed_eclipse_failure() {
        run_with_stack(|| {
            let _ephem_guard = crate::precomputed_ephem::ephemeris_test_guard();
            let _coeffs = install_test_coeffs(5);
            crate::precomputed_ephem::load_precomputed_ephemeris(
                crate::types::ForceFlags::SUN_GRAVITY,
            )
            .expect("typed Sun ephemeris catalogue must load");
            let jd0 = 2_460_310.5;
            let session = test_session(
                jd0,
                ForceConfig {
                    sph_order: 5,
                    force_flags: crate::types::ForceFlags::SRP,
                    am_ratio: 0.02,
                    cd: 0.0,
                    cr: 1.3,
                    eps: 1.0e-8,
                    dt_max: 60.0,
                    integrator_method: StepperMethod::Dopri5Compat,
                    ..ForceConfig::default()
                },
            );
            let radius_km = 60_000.0;
            let speed_km_s = (crate::types::MU / radius_km).sqrt();
            let state = [radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0];
            let mut output = [0.0; 6];
            let error = session
                .integrate_variable_final_into(
                    VariableFinalBatchRequest {
                        initial_eci_states: &state,
                        epoch_jd: &[jd0],
                        final_time_s: &[60.0],
                        t0_s: 0.0,
                        ballistics: VariableFinalBallistics::default(),
                    },
                    &mut output,
                )
                .expect_err("outside Part A eclipse envelope must fail session");
            assert!(matches!(
                error,
                VariableFinalNativeError::Row {
                    row: 0,
                    failure: VariableFinalRowFailure::Propagation(
                        FinalPropagationFailure::Eclipse(crate::eclipse::EclipseError::Envelope)
                    ),
                }
            ));
            assert!(output.iter().all(|value| value.is_nan()));
        });
    }

    #[test]
    fn variable_final_native_global_rayon_nested_invocation_uses_serial_branch() {
        run_with_stack(|| {
            let _ephem_guard = crate::precomputed_ephem::ephemeris_test_guard();
            let _coeffs = install_test_coeffs(5);
            let jd0 = 2_460_310.5;
            let n_rows = crate::batch::LIGHTYEAR_PAR_THRESHOLD;
            let session = test_session(
                jd0,
                ForceConfig {
                    sph_order: 0,
                    eps: 1.0e-8,
                    integrator_method: StepperMethod::Vern9,
                    ..ForceConfig::default()
                },
            );
            let states = create_test_eci_points(n_rows);
            let epochs = vec![jd0; n_rows];
            let tofs = vec![1.0; n_rows];

            let mut serial = vec![f64::NAN; n_rows * 6];
            session
                .integrate_variable_final_global_into(
                    VariableFinalBatchRequest {
                        initial_eci_states: &states,
                        epoch_jd: &epochs,
                        final_time_s: &tofs,
                        t0_s: 0.0,
                        ballistics: VariableFinalBallistics::default(),
                    },
                    &mut serial,
                    1,
                    2,
                )
                .expect("direct serial batch");

            let outer = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .stack_size(TEST_STACK_SIZE)
                .build()
                .expect("test outer pool");
            let mut nested = vec![f64::NAN; n_rows * 6];
            variable_final_native_test_observer_begin(nested.as_ptr().addr());
            outer
                .install(|| {
                    session.integrate_variable_final_global_into(
                        VariableFinalBatchRequest {
                            initial_eci_states: &states,
                            epoch_jd: &epochs,
                            final_time_s: &tofs,
                            t0_s: 0.0,
                            ballistics: VariableFinalBallistics::default(),
                        },
                        &mut nested,
                        2,
                        2,
                    )
                })
                .expect("nested batch must use serial branch");

            let observed = variable_final_native_test_observer_take();
            assert!(
                !observed.parallel_branch_entered,
                "nested batch must take the serial branch, not fan out again"
            );
            assert_eq!(
                observed.threads.len(),
                1,
                "nested batch must stay on current outer-pool worker"
            );
            assert_eq!(
                nested
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                serial
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        });
    }

    #[test]
    fn variable_final_nested_uniform_batch_constructs_one_checked_rhs() {
        run_with_stack(|| {
            let _coeffs = install_test_coeffs(5);
            let jd0 = 2_460_310.5;
            let n_rows = 39;
            let session = test_session(
                jd0,
                ForceConfig {
                    sph_order: 0,
                    eps: 1.0e-8,
                    integrator_method: StepperMethod::Vern9,
                    ..ForceConfig::default()
                },
            );
            let states = create_test_eci_points(n_rows);
            let epochs = vec![jd0; n_rows];
            let tofs = vec![60.0; n_rows];
            let mut out = vec![f64::NAN; n_rows * 6];
            let outer = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .stack_size(TEST_STACK_SIZE)
                .build()
                .expect("outer pool");

            let constructions = outer.install(|| {
                crate::rhs::reset_test_rhs_constructions();
                session
                    .integrate_variable_final_into(
                        VariableFinalBatchRequest {
                            initial_eci_states: &states,
                            epoch_jd: &epochs,
                            final_time_s: &tofs,
                            t0_s: 0.0,
                            ballistics: VariableFinalBallistics::default(),
                        },
                        &mut out,
                    )
                    .expect("nested checked batch");
                crate::rhs::test_rhs_constructions()
            });

            assert_eq!(
                constructions, 1,
                "uniform nested rows must reuse one checked lane RHS"
            );
            assert!(out.iter().all(|value| value.is_finite()));
        });
    }

    #[test]
    fn variable_final_native_batch_uses_global_rayon_pool_without_semantic_drift() {
        if std::env::var_os(GLOBAL_POOL_CHILD_ENV).is_none() {
            for scenario in [
                "invalid",
                "bad-state",
                "bad-ballistic",
                "unsupported-stepper",
                "ephemeris-precedence",
                "perigee",
                "invalid-equinoctial",
                "normal-invalid",
                "normal-small",
                "normal-unconfigured",
                "normal-authorized",
                "normal-nested",
                "width-one",
                "nested",
                "foreign-same-width",
                "generic-same-width",
                "repeated-explicit",
                "semantics",
            ] {
                let output =
                    Command::new(std::env::current_exe().expect("current Rust test executable"))
                        .args([GLOBAL_POOL_CHILD_TEST, "--exact", "--nocapture"])
                        .env(GLOBAL_POOL_CHILD_ENV, scenario)
                        .env("RUST_TEST_THREADS", "1")
                        .output()
                        .expect("spawn isolated global-pool child test");
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(
                    output.status.success(),
                    "{scenario} child failed\nstdout:\n{stdout}\nstderr:\n{stderr}",
                );
                let marker = format!("{GLOBAL_POOL_CHILD_MARKER}:{scenario}");
                assert!(
                    stdout.contains(&marker),
                    "{scenario} child filter did not execute target test\nstdout:\n{stdout}\nstderr:\n{stderr}",
                );
            }
            eprintln!("verified child marker: {GLOBAL_POOL_CHILD_MARKER}");
            return;
        }

        let scenario = std::env::var(GLOBAL_POOL_CHILD_ENV)
            .expect("global-pool child must declare its first-touch scenario");
        println!("{GLOBAL_POOL_CHILD_MARKER}:{scenario}");
        run_with_stack(move || run_global_pool_child(&scenario));
    }

    fn run_global_pool_child(scenario: &str) {
        let _coeffs = install_test_coeffs(5);
        let fixture = GlobalPoolTestFixture::new();
        match scenario {
            "invalid" => {
                assert_global_pool_invalid_requests(&fixture.session);
                assert_empty_top_level_width_four_initializes_pool(&fixture.session);
            }
            "bad-state" => {
                let session = row_preflight_session();
                assert_bad_state_request(&session);
                assert_empty_top_level_width_four_initializes_pool(&session);
            }
            "bad-ballistic" => {
                let session = row_preflight_session();
                assert_bad_ballistic_request(&session);
                assert_empty_top_level_width_four_initializes_pool(&session);
            }
            "unsupported-stepper" => {
                let unsupported = unsupported_stepper_session();
                assert_unsupported_stepper_request(&unsupported);
                let valid = row_preflight_session();
                assert_empty_top_level_width_four_initializes_pool(&valid);
            }
            "ephemeris-precedence" => {
                let ephemeris = ephemeris_precedence_session();
                assert_ephemeris_precedes_invalid_row(&ephemeris);
                let valid = row_preflight_session();
                assert_empty_top_level_width_four_initializes_pool(&valid);
            }
            "perigee" => {
                let session = row_preflight_session();
                assert_bad_perigee_request(&session);
                assert_empty_top_level_width_four_initializes_pool(&session);
            }
            "invalid-equinoctial" => {
                let session = row_preflight_session();
                assert_invalid_equinoctial_request(&session);
                assert_empty_top_level_width_four_initializes_pool(&session);
            }
            "normal-invalid" => {
                let session = row_preflight_session();
                assert_normal_invalid_request(&session);
                assert_empty_top_level_width_four_initializes_pool(&session);
            }
            "normal-small" => {
                let session = row_preflight_session();
                run_normal_small_batch(&session);
                assert_empty_top_level_width_four_initializes_pool(&session);
            }
            "normal-unconfigured" => {
                let session = row_preflight_session();
                run_normal_unconfigured_batch(&session);
                assert_empty_top_level_width_four_initializes_pool(&session);
            }
            "normal-authorized" => {
                assert_normal_authorized_batch_uses_configured_global_pool(&fixture);
            }
            "normal-nested" => {
                let session = row_preflight_session();
                run_normal_nested_unconfigured_batch(&session);
                assert_empty_top_level_width_four_initializes_pool(&session);
            }
            "width-one" => {
                run_empty_width_one_batch(&fixture.session);
                assert_empty_top_level_width_four_initializes_pool(&fixture.session);
            }
            "nested" => {
                run_nested_local_empty_width_four_batch(&fixture.session);
                assert_empty_top_level_width_four_initializes_pool(&fixture.session);
            }
            "foreign-same-width" => {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(4)
                    .build_global()
                    .expect("seed foreign global pool");
                assert_top_level_width_four_rejects_non_authoritative_pool(&fixture.session);
            }
            "generic-same-width" => {
                assert_eq!(nd_sched::init_global_pool(Some(4)), 4);
                assert_top_level_width_four_rejects_non_authoritative_pool(&fixture.session);
            }
            "repeated-explicit" => {
                assert_empty_top_level_width_four_initializes_pool(&fixture.session);
                run_empty_top_level_width_four_batch(&fixture.session)
                    .expect("repeated explicit W4 initialization must be idempotent");
            }
            "semantics" => {
                let _ephem_guard = crate::precomputed_ephem::ephemeris_test_guard();
                crate::precomputed_ephem::load_precomputed_ephemeris(
                    crate::types::ForceFlags::SUN_GRAVITY
                        | crate::types::ForceFlags::MOON_GRAVITY
                        | crate::types::ForceFlags::JUPITER_GRAVITY
                        | crate::types::ForceFlags::VENUS_GRAVITY,
                )
                .expect("all default typed ephemeris catalogues must load");
                // Raw Rayon observations stay inside this helper, after its
                // first top-level W4 variable-final API call establishes pool.
                assert_global_pool_result_bits(&fixture);
                assert_global_pool_rejects_width_mismatch(&fixture.session);
                assert_global_pool_failure_semantics(&fixture);
            }
            _ => panic!("unknown global-pool child scenario: {scenario}"),
        }
    }

    struct GlobalPoolTestFixture {
        session: LightyearSession,
        states: Vec<f64>,
        epochs: Vec<f64>,
        tofs: Vec<f64>,
        am: Vec<f64>,
        cd: Vec<f64>,
        cr: Vec<f64>,
    }

    impl GlobalPoolTestFixture {
        fn new() -> Self {
            let jd0 = 2_460_310.5;
            let n_rows = crate::batch::LIGHTYEAR_PAR_THRESHOLD;
            let session = test_session(
                jd0,
                ForceConfig {
                    sph_order: 5,
                    force_flags: crate::types::ForceFlags::DRAG | crate::types::ForceFlags::SRP,
                    atm_model: 3,
                    am_ratio: 0.01,
                    cd: 2.2,
                    cr: 1.3,
                    eps: 1.0e-8,
                    dt_max: 60.0,
                    integrator_method: StepperMethod::Dopri5Compat,
                    ..ForceConfig::default()
                },
            );
            Self {
                session,
                states: create_test_eci_points(n_rows),
                epochs: vec![jd0; n_rows],
                tofs: vec![1.0; n_rows],
                am: vec![0.01; n_rows],
                cd: vec![2.2; n_rows],
                cr: vec![1.3; n_rows],
            }
        }

        fn row_count(&self) -> usize {
            self.epochs.len()
        }
    }

    fn row_preflight_session() -> LightyearSession {
        test_session(
            2_460_310.5,
            ForceConfig {
                sph_order: 0,
                eps: 1.0e-8,
                dt_max: 60.0,
                integrator_method: StepperMethod::Dopri5Compat,
                ..ForceConfig::default()
            },
        )
    }

    fn unsupported_stepper_session() -> LightyearSession {
        test_session(
            2_460_310.5,
            ForceConfig {
                sph_order: 0,
                atm_model: 5,
                eps: 1.0e-8,
                dt_max: 60.0,
                integrator_method: StepperMethod::Auto,
                ..ForceConfig::default()
            },
        )
    }

    fn ephemeris_precedence_session() -> LightyearSession {
        test_session(
            2_460_310.5,
            ForceConfig {
                sph_order: 0,
                force_flags: crate::types::ForceFlags::SUN_GRAVITY,
                eps: 1.0e-8,
                dt_max: 60.0,
                integrator_method: StepperMethod::Dopri5Compat,
                ..ForceConfig::default()
            },
        )
    }

    fn assert_global_pool_rejects_width_mismatch(session: &LightyearSession) {
        let mut empty_out = [];
        let mismatch = session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &[],
                    epoch_jd: &[],
                    final_time_s: &[],
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut empty_out,
                2,
                4,
            )
            .expect_err("requested W2 must reject latched global W4");
        assert_eq!(
            mismatch.to_string(),
            "Part A requested Rayon width 2 but global pool started at 4"
        );
        session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &[],
                    epoch_jd: &[],
                    final_time_s: &[],
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut empty_out,
                4,
                4,
            )
            .expect("valid W4 request must retain the named global pool after rejected W2");
        canonical_global_worker_ids();
    }

    fn assert_bad_state_request(session: &LightyearSession) {
        let states = [f64::NAN, 0.0, 0.0, 0.0, 7.5, 0.0];
        let epochs = [2_460_310.5];
        let final_time = [1.0];
        let mut output = [0.0; 6];
        let error = session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &states,
                    epoch_jd: &epochs,
                    final_time_s: &final_time,
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut output,
                2,
                4,
            )
            .expect_err("nonfinite row must fail before global-pool initialization");
        assert!(matches!(
            error,
            VariableFinalNativeError::Row {
                row: 0,
                failure: VariableFinalRowFailure::InvalidInput,
            }
        ));
        assert!(output.iter().all(|value| value.is_nan()));
    }

    fn assert_bad_ballistic_request(session: &LightyearSession) {
        let states = [7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let epochs = [2_460_310.5];
        let final_time = [1.0];
        let am_ratio = [f64::NAN];
        let mut output = [0.0; 6];
        let error = session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &states,
                    epoch_jd: &epochs,
                    final_time_s: &final_time,
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics {
                        am_ratio: Some(&am_ratio),
                        cd: None,
                        cr: None,
                    },
                },
                &mut output,
                2,
                4,
            )
            .expect_err("nonfinite ballistic row must fail before global-pool initialization");
        assert!(matches!(
            error,
            VariableFinalNativeError::Row {
                row: 0,
                failure: VariableFinalRowFailure::InvalidBallisticCoefficient,
            }
        ));
        assert!(output.iter().all(|value| value.is_nan()));
    }

    fn assert_unsupported_stepper_request(session: &LightyearSession) {
        let mut output = [];
        let error = session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &[],
                    epoch_jd: &[],
                    final_time_s: &[],
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut output,
                2,
                4,
            )
            .expect_err("unsupported scalar stepper must fail before global-pool initialization");
        assert!(
            matches!(&error, VariableFinalNativeError::UnsupportedStepper(_)),
            "unexpected unsupported-stepper result: {error:?}"
        );
    }

    fn assert_ephemeris_precedes_invalid_row(session: &LightyearSession) {
        let states = [f64::NAN, 0.0, 0.0, 0.0, 7.5, 0.0];
        let epochs = [1.0];
        let final_time = [1.0];
        let mut output = [7.0; 6];
        let error = session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &states,
                    epoch_jd: &epochs,
                    final_time_s: &final_time,
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut output,
                2,
                4,
            )
            .expect_err("dynamic ephemeris failure must precede invalid-row preflight");
        assert!(
            matches!(&error, VariableFinalNativeError::Ephemeris { row: 0, .. }),
            "ephemeris preflight must retain precedence: {error:?}"
        );
        assert!(
            output
                .iter()
                .all(|value| value.to_bits() == 7.0_f64.to_bits()),
            "ephemeris failure must precede row-output mutation: {output:?}"
        );
    }

    fn assert_bad_perigee_request(session: &LightyearSession) {
        let safe_radius_km = 7_000.0;
        let period_s = 4_920.0;
        let mu = crate::types::MU;
        let semi_major_km = (mu * (period_s / (2.0 * std::f64::consts::PI)).powi(2)).cbrt();
        let impact_speed_km_s = (mu * (2.0 / safe_radius_km - 1.0 / semi_major_km)).sqrt();
        let protected_radius_km = crate::types::RE + crate::types::GROUND_ALTITUDE;
        let hidden_perigee_km = 2.0_f64.mul_add(semi_major_km, -safe_radius_km);
        assert!(safe_radius_km >= protected_radius_km);
        assert!(hidden_perigee_km < protected_radius_km);
        let states = [safe_radius_km, 0.0, 0.0, 0.0, impact_speed_km_s, 0.0];
        let epochs = [2_460_310.5];
        let final_time = [1.0];
        let mut output = [0.0; 6];
        let error = session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &states,
                    epoch_jd: &epochs,
                    final_time_s: &final_time,
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut output,
                2,
                4,
            )
            .expect_err("below-minimum perigee must fail before global-pool initialization");
        assert!(
            matches!(
                &error,
                VariableFinalNativeError::Row {
                    row: 0,
                    failure: VariableFinalRowFailure::MinimumRadiusViolation,
                }
            ),
            "unexpected below-perigee result: {error:?}"
        );
        assert!(output.iter().all(|value| value.is_nan()));
    }

    fn assert_invalid_equinoctial_request(session: &LightyearSession) {
        let states = [7_000.0, 0.0, 0.0, 0.0, 20.0, 0.0];
        let mut equinoctial = [0.0; 6];
        eci2equinoc_impl_f64(&states, 6, 0.0, 0.0, &mut equinoctial);
        assert!(
            equinoctial.iter().any(|value| !value.is_finite()),
            "finite unbound fixture must reach ECI-to-equinoctial rejection: {equinoctial:?}"
        );
        let epochs = [2_460_310.5];
        let final_time = [1.0];
        let mut output = [0.0; 6];
        let error = session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &states,
                    epoch_jd: &epochs,
                    final_time_s: &final_time,
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut output,
                2,
                4,
            )
            .expect_err(
                "finite invalid equinoctial row must fail before global-pool initialization",
            );
        assert!(
            matches!(
                &error,
                VariableFinalNativeError::Row {
                    row: 0,
                    failure: VariableFinalRowFailure::InvalidEquinoctialState,
                }
            ),
            "unexpected invalid-equinoctial result: {error:?}"
        );
        assert!(output.iter().all(|value| value.is_nan()));
    }

    fn assert_normal_invalid_request(session: &LightyearSession) {
        let states = [f64::NAN, 0.0, 0.0, 0.0, 7.5, 0.0];
        let epochs = [2_460_310.5];
        let final_time = [1.0];
        let mut output = [0.0; 6];
        let error = session
            .integrate_variable_final_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &states,
                    epoch_jd: &epochs,
                    final_time_s: &final_time,
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut output,
            )
            .expect_err("normal nonfinite row must fail without global-pool initialization");
        assert!(matches!(
            error,
            VariableFinalNativeError::Row {
                row: 0,
                failure: VariableFinalRowFailure::InvalidInput,
            }
        ));
        assert!(output.iter().all(|value| value.is_nan()));
    }

    fn run_normal_small_batch(session: &LightyearSession) {
        let states = [7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let epochs = [2_460_310.5];
        let final_time = [1.0];
        let mut output = [f64::NAN; 6];
        session
            .integrate_variable_final_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &states,
                    epoch_jd: &epochs,
                    final_time_s: &final_time,
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut output,
            )
            .expect("normal small batch");
        assert!(output.iter().all(|value| value.is_finite()));
    }

    fn run_normal_unconfigured_batch(session: &LightyearSession) {
        let n_rows = crate::batch::LIGHTYEAR_PAR_THRESHOLD;
        let states = create_test_eci_points(n_rows);
        let epochs = vec![2_460_310.5; n_rows];
        let final_time = vec![1.0; n_rows];
        let mut output = vec![f64::NAN; n_rows * 6];
        variable_final_native_test_observer_begin(output.as_ptr().addr());
        session
            .integrate_variable_final_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &states,
                    epoch_jd: &epochs,
                    final_time_s: &final_time,
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut output,
            )
            .expect("normal unconfigured batch");
        let observed = variable_final_native_test_observer_take();
        assert!(
            !observed.parallel_branch_entered,
            "unconfigured normal batch must stay serial"
        );
        assert!(output.iter().all(|value| value.is_finite()));
    }

    fn run_normal_nested_unconfigured_batch(session: &LightyearSession) {
        let n_rows = crate::batch::LIGHTYEAR_PAR_THRESHOLD;
        let states = create_test_eci_points(n_rows);
        let epochs = vec![2_460_310.5; n_rows];
        let final_time = vec![1.0; n_rows];
        let mut output = vec![f64::NAN; n_rows * 6];
        let local_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build local normal nested Rayon pool");
        variable_final_native_test_observer_begin(output.as_ptr().addr());
        local_pool
            .install(|| {
                session.integrate_variable_final_into(
                    VariableFinalBatchRequest {
                        initial_eci_states: &states,
                        epoch_jd: &epochs,
                        final_time_s: &final_time,
                        t0_s: 0.0,
                        ballistics: VariableFinalBallistics::default(),
                    },
                    &mut output,
                )
            })
            .expect("normal nested batch");
        let observed = variable_final_native_test_observer_take();
        assert!(
            !observed.parallel_branch_entered,
            "nested normal batch must stay serial"
        );
        assert!(output.iter().all(|value| value.is_finite()));
    }

    fn assert_normal_authorized_batch_uses_configured_global_pool(fixture: &GlobalPoolTestFixture) {
        let serial_width_one = run_width_one_batch(fixture);
        assert_empty_top_level_width_four_initializes_pool(&fixture.session);

        let mut normal_output = vec![f64::NAN; output_values(fixture)];
        variable_final_native_test_observer_begin(normal_output.as_ptr().addr());
        fixture
            .session
            .integrate_variable_final_into(fixture_request(fixture), &mut normal_output)
            .expect("normal batch after explicit W4 configuration");
        let observed = variable_final_native_test_observer_take();
        assert!(
            observed.parallel_branch_entered,
            "authorized normal batch stayed serial: {:?}",
            observed.threads
        );
        let global_worker_ids = canonical_global_worker_ids();
        assert!(
            observed.threads.is_subset(&global_worker_ids),
            "authorized normal batch used non-global Rayon workers: {:?}",
            observed.threads
        );
        assert_eq!(
            normal_output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            serial_width_one
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "normal W4 batch changed W1 output bits or input order"
        );
    }

    fn fixture_request(fixture: &GlobalPoolTestFixture) -> VariableFinalBatchRequest<'_> {
        VariableFinalBatchRequest {
            initial_eci_states: &fixture.states,
            epoch_jd: &fixture.epochs,
            final_time_s: &fixture.tofs,
            t0_s: 0.0,
            ballistics: VariableFinalBallistics {
                am_ratio: Some(&fixture.am),
                cd: Some(&fixture.cd),
                cr: Some(&fixture.cr),
            },
        }
    }

    fn output_values(fixture: &GlobalPoolTestFixture) -> usize {
        fixture
            .row_count()
            .checked_mul(6)
            .expect("global-pool test output length must not overflow")
    }

    fn run_width_one_batch(fixture: &GlobalPoolTestFixture) -> Vec<f64> {
        let mut output = vec![f64::NAN; output_values(fixture)];
        variable_final_native_test_observer_begin(output.as_ptr().addr());
        fixture
            .session
            .integrate_variable_final_global_into(fixture_request(fixture), &mut output, 1, 4)
            .expect("serial width-one batch");
        let observed = variable_final_native_test_observer_take();
        assert!(
            !observed.parallel_branch_entered,
            "width-one baseline must stay serial"
        );
        let worker_count = observed.threads.len();
        assert_eq!(
            worker_count, 1,
            "width-one baseline ran on {worker_count} threads"
        );
        output
    }

    fn run_empty_width_one_batch(session: &LightyearSession) {
        let mut empty_out = [];
        session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &[],
                    epoch_jd: &[],
                    final_time_s: &[],
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut empty_out,
                1,
                4,
            )
            .expect("valid empty W1 batch");
    }

    fn run_nested_local_empty_width_four_batch(session: &LightyearSession) {
        let local_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build local nested Rayon pool");
        local_pool.install(|| {
            let mut empty_out = [];
            session
                .integrate_variable_final_global_into(
                    VariableFinalBatchRequest {
                        initial_eci_states: &[],
                        epoch_jd: &[],
                        final_time_s: &[],
                        t0_s: 0.0,
                        ballistics: VariableFinalBallistics::default(),
                    },
                    &mut empty_out,
                    4,
                    4,
                )
                .expect("valid nested local W4 batch");
        });
    }

    fn assert_empty_top_level_width_four_initializes_pool(session: &LightyearSession) {
        assert_eq!(
            nd_sched::configured_global_pool_threads(),
            None,
            "prior request must not latch the global Rayon pool"
        );
        run_empty_top_level_width_four_batch(session).expect("valid top-level W4 batch");
        canonical_global_worker_ids();
    }

    fn run_empty_top_level_width_four_batch(
        session: &LightyearSession,
    ) -> Result<(), VariableFinalNativeError> {
        let mut empty_out = [];
        session.integrate_variable_final_global_into(
            VariableFinalBatchRequest {
                initial_eci_states: &[],
                epoch_jd: &[],
                final_time_s: &[],
                t0_s: 0.0,
                ballistics: VariableFinalBallistics::default(),
            },
            &mut empty_out,
            4,
            4,
        )
    }

    fn assert_top_level_width_four_rejects_non_authoritative_pool(session: &LightyearSession) {
        let error = run_empty_top_level_width_four_batch(session)
            .expect_err("non-authoritative same-width global pool must be rejected");
        assert!(
            matches!(error, VariableFinalNativeError::RayonConfig(_)),
            "unexpected error: {error}"
        );
        assert_eq!(
            nd_sched::configured_global_pool_threads(),
            None,
            "non-authoritative pool must never become Part A scheduler authority"
        );
    }

    fn canonical_global_worker_ids() -> HashSet<std::thread::ThreadId> {
        let global_workers = rayon::broadcast(|_| {
            let worker = std::thread::current();
            (worker.id(), worker.name().unwrap_or_default().to_owned())
        });
        assert_eq!(
            rayon::current_num_threads(),
            4,
            "API must establish the requested four-worker canonical global pool"
        );
        assert_eq!(
            global_workers.len(),
            4,
            "canonical global-pool worker identity drift: {global_workers:?}"
        );
        assert!(
            global_workers
                .iter()
                .all(|(_, name)| name.starts_with("nd-sched-")),
            "variable-final API must create named nd-sched workers: {global_workers:?}"
        );
        global_workers.into_iter().map(|(id, _)| id).collect()
    }

    fn run_global_width_four_batch(fixture: &GlobalPoolTestFixture) -> Vec<f64> {
        let mut output = vec![f64::NAN; output_values(fixture)];
        variable_final_native_test_observer_begin(output.as_ptr().addr());
        fixture
            .session
            .integrate_variable_final_global_into(fixture_request(fixture), &mut output, 4, 4)
            .expect("global-pool width-four batch");
        let observed = variable_final_native_test_observer_take();
        assert!(
            observed.parallel_branch_entered,
            "global-pool batch stayed serial: {:?}",
            observed.threads
        );
        let global_worker_ids = canonical_global_worker_ids();
        assert!(
            observed.threads.is_subset(&global_worker_ids),
            "batch used non-global Rayon workers: {:?}",
            observed.threads
        );
        output
    }

    fn assert_global_pool_result_bits(fixture: &GlobalPoolTestFixture) {
        let global_width_four = run_global_width_four_batch(fixture);
        let serial_width_one = run_width_one_batch(fixture);
        assert_eq!(
            global_width_four
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            serial_width_one
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "W1/W4 outputs changed bits or input order"
        );
    }

    fn assert_global_pool_failure_semantics(fixture: &GlobalPoolTestFixture) {
        let output_values = fixture
            .row_count()
            .checked_mul(6)
            .expect("global-pool test output length must not overflow");
        let mut hostile_states = fixture.states.clone();
        let hostile_state = test_state_row_mut(&mut hostile_states, 2);
        *hostile_state
            .get_mut(0)
            .expect("test state row must have an x component") = 6_300.0;
        hostile_state
            .get_mut(1..)
            .expect("test state row must have velocity components")
            .fill(0.0);
        let mut hostile_am = fixture.am.clone();
        *hostile_am
            .get_mut(7)
            .expect("test AM vector must contain hostile row") = f64::NAN;
        let mut hostile_w1_out = vec![f64::NAN; output_values];
        let hostile_w1_failure = fixture
            .session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &hostile_states,
                    epoch_jd: &fixture.epochs,
                    final_time_s: &fixture.tofs,
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics {
                        am_ratio: Some(&hostile_am),
                        cd: Some(&fixture.cd),
                        cr: Some(&fixture.cr),
                    },
                },
                &mut hostile_w1_out,
                1,
                4,
            )
            .expect_err("two hostile rows must fail serial W1 batch");
        let mut hostile_w4_out = vec![f64::NAN; output_values];
        let hostile_w4_failure = fixture
            .session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &hostile_states,
                    epoch_jd: &fixture.epochs,
                    final_time_s: &fixture.tofs,
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics {
                        am_ratio: Some(&hostile_am),
                        cd: Some(&fixture.cd),
                        cr: Some(&fixture.cr),
                    },
                },
                &mut hostile_w4_out,
                4,
                4,
            )
            .expect_err("two hostile rows must fail global W4 batch");
        for (width, failure) in [(1, hostile_w1_failure), (4, hostile_w4_failure)] {
            match failure {
                VariableFinalNativeError::Row { row, failure } => {
                    assert_eq!(row, 2, "W{width} did not select lowest failure");
                    assert_eq!(
                        failure,
                        VariableFinalRowFailure::MinimumRadiusViolation,
                        "W{width} changed typed failure identity"
                    );
                }
                other => panic!("W{width} changed error identity: {other:?}"),
            }
        }
    }

    fn assert_global_pool_invalid_requests(session: &LightyearSession) {
        let too_many_cores = satpy_core::parallel_budget::available_cores()
            .checked_add(1)
            .expect("available-core count must not be usize::MAX");
        for (width, budget, expected) in [
            (0, 1, "positive"),
            (2, 1, "budget"),
            (1, too_many_cores, "available"),
        ] {
            let mut empty_out = [];
            let error = session
                .integrate_variable_final_global_into(
                    VariableFinalBatchRequest {
                        initial_eci_states: &[],
                        epoch_jd: &[],
                        final_time_s: &[],
                        t0_s: 0.0,
                        ballistics: VariableFinalBallistics::default(),
                    },
                    &mut empty_out,
                    width,
                    budget,
                )
                .expect_err("invalid Rayon width or budget must fail");
            assert!(error.to_string().contains(expected), "{error}");
        }
        let epoch = [0.0];
        let final_time = [1.0];
        let mut empty_out = [];
        let error = session
            .integrate_variable_final_global_into(
                VariableFinalBatchRequest {
                    initial_eci_states: &[],
                    epoch_jd: &epoch,
                    final_time_s: &final_time,
                    t0_s: 0.0,
                    ballistics: VariableFinalBallistics::default(),
                },
                &mut empty_out,
                4,
                4,
            )
            .expect_err("malformed batch shape must fail before global-pool initialization");
        assert!(matches!(error, VariableFinalNativeError::InputContract));
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VariableFinalRowFailure {
    #[default]
    None,
    InvalidInput,
    InvalidEquinoctialState,
    InvalidBallisticCoefficient,
    MinimumRadiusViolation,
    Propagation(FinalPropagationFailure),
    NonFiniteOutput,
}

#[derive(Debug)]
pub enum VariableFinalNativeError {
    InputContract,
    ArithmeticOverflow,
    UnsupportedStepper(anyhow::Error),
    RayonConfig(anyhow::Error),
    Ephemeris {
        row: usize,
        source: anyhow::Error,
    },
    Row {
        row: usize,
        failure: VariableFinalRowFailure,
    },
}

/// What a batch call did, as observed from inside it.
///
/// `parallel_branch_entered` is recorded by the parallel arm itself. Counting
/// distinct worker threads cannot substitute for it: Rayon is free to run a
/// small batch entirely on one worker, so `threads.len() == 1` is a legal
/// outcome of the parallel branch and the batch-stayed-serial assertion built
/// on it failed at random. The thread set is still collected, but only the
/// global-pool membership check reads it -- that question really is about
/// threads.
#[cfg(test)]
#[derive(Default)]
struct VariableFinalNativeObservation {
    output_address: usize,
    parallel_branch_entered: bool,
    threads: HashSet<std::thread::ThreadId>,
}

#[cfg(test)]
static VARIABLE_FINAL_NATIVE_TEST_OBSERVATION: Mutex<Option<VariableFinalNativeObservation>> =
    Mutex::new(None);

#[cfg(test)]
fn variable_final_native_test_observer_begin(output_address: usize) {
    *VARIABLE_FINAL_NATIVE_TEST_OBSERVATION
        .lock()
        .expect("native test observer lock") = Some(VariableFinalNativeObservation {
        output_address,
        ..VariableFinalNativeObservation::default()
    });
}

#[cfg(test)]
fn variable_final_native_test_observer_with(
    output_address: usize,
    apply: impl FnOnce(&mut VariableFinalNativeObservation),
) {
    if let Some(observation) = VARIABLE_FINAL_NATIVE_TEST_OBSERVATION
        .lock()
        .expect("native test observer lock")
        .as_mut()
    {
        if observation.output_address == output_address {
            apply(observation);
        }
    }
}

#[cfg(test)]
fn variable_final_native_test_observer_record_parallel_entry(output_address: usize) {
    variable_final_native_test_observer_with(output_address, |observation| {
        observation.parallel_branch_entered = true;
    });
}

#[cfg(test)]
fn variable_final_native_test_observer_record(output_address: usize) {
    variable_final_native_test_observer_with(output_address, |observation| {
        observation.threads.insert(std::thread::current().id());
    });
}

#[cfg(test)]
fn variable_final_native_test_observer_take() -> VariableFinalNativeObservation {
    VARIABLE_FINAL_NATIVE_TEST_OBSERVATION
        .lock()
        .expect("native test observer lock")
        .take()
        .expect("native test observer active")
}

impl std::fmt::Display for VariableFinalNativeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputContract => write!(formatter, "variable-final native input contract failed"),
            Self::ArithmeticOverflow => write!(formatter, "variable-final native counter overflow"),
            Self::UnsupportedStepper(source) | Self::RayonConfig(source) => {
                std::fmt::Display::fmt(source, formatter)
            }
            Self::Ephemeris { row, source } => write!(
                formatter,
                "variable-final row {row} ephemeris preflight failed: {source}"
            ),
            Self::Row {
                row,
                failure: VariableFinalRowFailure::MinimumRadiusViolation,
            } => write!(
                formatter,
                "variable-final row {row} failed: perigee altitude is below minimum protected radius"
            ),
            Self::Row { row, failure } => {
                write!(formatter, "variable-final row {row} failed: {failure:?}")
            }
        }
    }
}

impl std::error::Error for VariableFinalNativeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UnsupportedStepper(source)
            | Self::RayonConfig(source)
            | Self::Ephemeris { source, .. } => Some(source.as_ref()),
            Self::InputContract | Self::ArithmeticOverflow | Self::Row { .. } => None,
        }
    }
}

#[inline]
fn variable_final_osculating_perigee_km(state: &[f64]) -> Option<f64> {
    let state: &[f64; 6] = state.get(..6)?.try_into().ok()?;
    if !state.iter().all(|value| value.is_finite()) {
        return None;
    }
    let [rx, ry, rz, vx, vy, vz] = *state;
    let r = [rx, ry, rz];
    let v = [vx, vy, vz];
    let r_norm = norm3(&r);
    let h = cross3(&r, &v);
    let h_sq = h[0].mul_add(h[0], h[1].mul_add(h[1], h[2] * h[2]));
    if !(r_norm > 0.0 && h_sq > 0.0) {
        return None;
    }
    let vxh = cross3(&v, &h);
    let mu = crate::types::MU;
    let eccentricity = norm3(&[
        vxh[0] / mu - r[0] / r_norm,
        vxh[1] / mu - r[1] / r_norm,
        vxh[2] / mu - r[2] / r_norm,
    ]);
    let perigee = (h_sq / mu) / (1.0 + eccentricity);
    (perigee.is_finite() && perigee > 0.0).then_some(perigee)
}

#[inline]
fn variable_final_state_clears_min_radius(state: &[f64], min_radius_km: f64) -> bool {
    let Some(position) = state
        .get(..3)
        .and_then(|values| <&[f64; 3]>::try_from(values).ok())
    else {
        return false;
    };
    min_radius_km.is_finite()
        && min_radius_km > 0.0
        && norm3(position) >= min_radius_km
        && variable_final_osculating_perigee_km(state)
            .is_some_and(|perigee| perigee >= min_radius_km)
}

/// One terminal-event-aware ECI row observed by the bounded qualification
/// diagnostic.
///
/// This type is feature-only. `outcome` is the reconstructed ECI endpoint,
/// while solver counters and terminal status come from the exact scalar leg
/// which produced it. A missing slot in the caller-owned output means no
/// scalar leg began, so a qualification trace must fail closed rather than
/// infer a zero-work success.
#[cfg(feature = "scalar-leg-observer")]
#[derive(Clone, Debug)]
pub struct ObservedVariableFinalRow {
    pub outcome: Result<[f64; 6], VariableFinalRowFailure>,
    pub metrics: Result<ObservedFinalMetrics, ObservedFinalMetricError>,
    pub terminal_status: ObservedSolverTerminalStatus,
}

/// Optional row-local ballistic coefficients for variable-final propagation.
///
/// Omitted arrays retain the immutable session force authority's value.
#[derive(Clone, Copy, Default)]
pub struct VariableFinalBallistics<'a> {
    pub am_ratio: Option<&'a [f64]>,
    pub cd: Option<&'a [f64]>,
    pub cr: Option<&'a [f64]>,
}

/// Immutable inputs for an ECI variable-final batch propagation.
///
/// Tolerance and integrator method remain exclusively in the session's scalar
/// force authority; callers cannot override either per batch.
#[derive(Clone, Copy)]
pub struct VariableFinalBatchRequest<'a> {
    pub initial_eci_states: &'a [f64],
    pub epoch_jd: &'a [f64],
    pub final_time_s: &'a [f64],
    pub t0_s: f64,
    pub ballistics: VariableFinalBallistics<'a>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct VariableFinalIntegratorKey {
    epoch_jd: u64,
    final_time_s: u64,
    start_time_s: u64,
    area_mass_ratio: u64,
    drag_coefficient: u64,
    reflectivity_coefficient: u64,
}

impl VariableFinalIntegratorKey {
    const fn new(propagation: &VariableFinalPropagation<'_>) -> Self {
        Self {
            epoch_jd: propagation.epoch_jd.to_bits(),
            final_time_s: propagation.final_time_s.to_bits(),
            start_time_s: propagation.start_time_s.to_bits(),
            area_mass_ratio: propagation.area_mass_ratio.to_bits(),
            drag_coefficient: propagation.drag_coefficient.to_bits(),
            reflectivity_coefficient: propagation.reflectivity_coefficient.to_bits(),
        }
    }
}

#[derive(Default)]
struct VariableFinalReusableCache {
    key: Option<VariableFinalIntegratorKey>,
    integrator: Option<ReusableFinalNoEventIntegrator>,
    checked_integrator: Option<ReusableFinalCheckedIntegrator>,
}

#[derive(Clone, Copy)]
struct VariableFinalPropagation<'a> {
    initial_equinoctial: [f64; 6],
    epoch_jd: f64,
    start_time_s: f64,
    final_time_s: f64,
    area_mass_ratio: f64,
    drag_coefficient: f64,
    reflectivity_coefficient: f64,
    ephemeris: &'a crate::precomputed_ephem::AllPrecomputedEphemeris,
}

#[derive(Clone, Copy)]
struct VariableFinalExecution<'a> {
    request: VariableFinalBatchRequest<'a>,
    ephemeris: &'a crate::precomputed_ephem::AllPrecomputedEphemeris,
    enforce_terminal_events: bool,
}

#[derive(Clone, Copy)]
struct VariableFinalPreparedRow {
    initial_equinoctial: [f64; 6],
    epoch_jd: f64,
    final_time_s: f64,
    area_mass_ratio: f64,
    drag_coefficient: f64,
    reflectivity_coefficient: f64,
}

#[derive(Debug)]
struct VariableFinalEphemerisError {
    row: usize,
    source: crate::precomputed_ephem::EphemerisCoverageError,
}

impl std::fmt::Display for VariableFinalEphemerisError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "variable-final row {} dynamic ephemeris preflight failed: {}",
            self.row, self.source
        )
    }
}

impl std::error::Error for VariableFinalEphemerisError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl LightyearSession {
    fn prepare_variable_final_row(
        &self,
        idx: usize,
        init_chunk: &[f64],
        request: &VariableFinalBatchRequest<'_>,
        enforce_terminal_events: bool,
    ) -> Result<VariableFinalPreparedRow, VariableFinalRowFailure> {
        if init_chunk.len() != 6 {
            return Err(VariableFinalRowFailure::InvalidInput);
        }
        let Some((&epoch_jd, &final_time_s)) =
            request.epoch_jd.get(idx).zip(request.final_time_s.get(idx))
        else {
            return Err(VariableFinalRowFailure::InvalidInput);
        };
        if !epoch_jd.is_finite()
            || !final_time_s.is_finite()
            || !request.t0_s.is_finite()
            || !init_chunk.iter().all(|value| value.is_finite())
        {
            return Err(VariableFinalRowFailure::InvalidInput);
        }
        let min_radius_km = self.context.config.earth_radius + crate::types::GROUND_ALTITUDE;
        if enforce_terminal_events
            && !variable_final_state_clears_min_radius(init_chunk, min_radius_km)
        {
            return Err(VariableFinalRowFailure::MinimumRadiusViolation);
        }

        let mut initial_equinoctial = [0.0f64; 6];
        eci2equinoc_impl_f64(init_chunk, 6, 0.0, 0.0, &mut initial_equinoctial);
        if !initial_equinoctial.iter().all(|value| value.is_finite()) {
            return Err(VariableFinalRowFailure::InvalidEquinoctialState);
        }

        let area_mass_ratio = request
            .ballistics
            .am_ratio
            .and_then(|values| values.get(idx))
            .copied()
            .unwrap_or(self.context.config.am_ratio);
        let drag_coefficient = request
            .ballistics
            .cd
            .and_then(|values| values.get(idx))
            .copied()
            .unwrap_or(self.context.config.cd);
        let reflectivity_coefficient = request
            .ballistics
            .cr
            .and_then(|values| values.get(idx))
            .copied()
            .unwrap_or(self.context.config.cr);
        if !area_mass_ratio.is_finite()
            || !drag_coefficient.is_finite()
            || !reflectivity_coefficient.is_finite()
        {
            return Err(VariableFinalRowFailure::InvalidBallisticCoefficient);
        }

        Ok(VariableFinalPreparedRow {
            initial_equinoctial,
            epoch_jd,
            final_time_s,
            area_mass_ratio,
            drag_coefficient,
            reflectivity_coefficient,
        })
    }

    fn preflight_variable_final_rows(
        &self,
        request: &VariableFinalBatchRequest<'_>,
        states_out: &mut [f64],
        enforce_terminal_events: bool,
    ) -> Result<(), VariableFinalNativeError> {
        for (row, (init_chunk, out_chunk)) in request
            .initial_eci_states
            .chunks_exact(6)
            .zip(states_out.chunks_exact_mut(6))
            .enumerate()
        {
            if let Err(failure) =
                self.prepare_variable_final_row(row, init_chunk, request, enforce_terminal_events)
            {
                out_chunk.fill(f64::NAN);
                return Err(VariableFinalNativeError::Row { row, failure });
            }
        }
        Ok(())
    }

    fn preflight_variable_final_ephemeris(
        &self,
        request: &VariableFinalBatchRequest<'_>,
    ) -> Result<Arc<crate::precomputed_ephem::AllPrecomputedEphemeris>, VariableFinalEphemerisError>
    {
        debug_assert_eq!(request.epoch_jd.len(), request.final_time_s.len());
        for (row, (&jd0, &tf_s)) in request
            .epoch_jd
            .iter()
            .zip(request.final_time_s)
            .enumerate()
        {
            let jd_a = jd0 + request.t0_s / satpy_core::SEC_PER_DAY;
            let jd_b = jd0 + tf_s / satpy_core::SEC_PER_DAY;
            self.context
                .config
                .with_ephemeris_for_arc(jd_a, jd_b)
                .map_err(|source| VariableFinalEphemerisError { row, source })?;
        }
        Ok(
            crate::precomputed_ephem::get_precomputed_ephemeris().unwrap_or_else(|| {
                Arc::new(crate::precomputed_ephem::AllPrecomputedEphemeris::default())
            }),
        )
    }

    /// Run the existing variable-final checked core serially with one
    /// caller-owned observation slot per row.
    ///
    /// Qualification invokes this only from the nested W1 scalar boundary:
    /// the canonical batch is serial there too, and the one reusable RHS cache
    /// remains local to this exact batch. It is feature-only and accepts no
    /// solver, force, or asset override. Every occupied or missing slot is a
    /// caller contract failure; this method never overwrites prior evidence.
    ///
    /// # Errors
    ///
    /// Returns the same typed input, authority, ephemeris, or first-row error
    /// as the canonical serial batch. A scalar leg that started always writes
    /// its observed outcome before its row failure is returned.
    #[cfg(feature = "scalar-leg-observer")]
    pub fn integrate_variable_final_observed_into(
        &self,
        request: VariableFinalBatchRequest<'_>,
        states_out: &mut [f64],
        observations_out: &mut [Option<ObservedVariableFinalRow>],
    ) -> Result<(), VariableFinalNativeError> {
        let n_rows = request.epoch_jd.len();
        let Some(state_values) = n_rows.checked_mul(6) else {
            return Err(VariableFinalNativeError::InputContract);
        };
        if request.initial_eci_states.len() != state_values
            || request.final_time_s.len() != n_rows
            || states_out.len() != state_values
            || observations_out.len() != n_rows
            || observations_out.iter().any(Option::is_some)
            || request
                .ballistics
                .am_ratio
                .is_some_and(|values| values.len() != n_rows)
            || request
                .ballistics
                .cd
                .is_some_and(|values| values.len() != n_rows)
            || request
                .ballistics
                .cr
                .is_some_and(|values| values.len() != n_rows)
        {
            return Err(VariableFinalNativeError::InputContract);
        }
        validate_scalar_stepper_authority(&self.context.config, "scalar")
            .map_err(VariableFinalNativeError::UnsupportedStepper)?;
        let ephem = self
            .preflight_variable_final_ephemeris(&request)
            .map_err(|error| VariableFinalNativeError::Ephemeris {
                row: error.row,
                source: anyhow::Error::new(error.source),
            })?;
        let execution = VariableFinalExecution {
            request,
            ephemeris: ephem.as_ref(),
            enforce_terminal_events: true,
        };
        let mut cache = VariableFinalReusableCache::default();
        for (row, ((init_chunk, out_chunk), observation_slot)) in request
            .initial_eci_states
            .chunks_exact(6)
            .zip(states_out.chunks_exact_mut(6))
            .zip(observations_out.iter_mut())
            .enumerate()
        {
            let failure = self.propagate_variable_final_row_observed_with_cache_into(
                row,
                init_chunk,
                out_chunk,
                execution,
                &mut cache,
                observation_slot,
            );
            if failure != VariableFinalRowFailure::None {
                return Err(VariableFinalNativeError::Row { row, failure });
            }
        }
        Ok(())
    }

    /// ECI variable-final propagation entry (native in-process kernel).
    /// Used by parity diagnostics that must exercise the exact ECI
    /// variable-final propagation kernel.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the input shape, stepper authority,
    /// ephemeris coverage, or any row propagation is invalid.
    pub fn integrate_variable_final_into(
        &self,
        request: VariableFinalBatchRequest<'_>,
        states_out: &mut [f64],
    ) -> Result<(), VariableFinalNativeError> {
        let n_rows = request.epoch_jd.len();
        let Some(state_values) = n_rows.checked_mul(6) else {
            return Err(VariableFinalNativeError::InputContract);
        };
        if request.initial_eci_states.len() != state_values
            || request.final_time_s.len() != n_rows
            || states_out.len() != state_values
            || request
                .ballistics
                .am_ratio
                .is_some_and(|values| values.len() != n_rows)
            || request
                .ballistics
                .cd
                .is_some_and(|values| values.len() != n_rows)
            || request
                .ballistics
                .cr
                .is_some_and(|values| values.len() != n_rows)
        {
            return Err(VariableFinalNativeError::InputContract);
        }
        validate_scalar_stepper_authority(&self.context.config, "scalar")
            .map_err(VariableFinalNativeError::UnsupportedStepper)?;
        let ephem = self
            .preflight_variable_final_ephemeris(&request)
            .map_err(|error| VariableFinalNativeError::Ephemeris {
                row: error.row,
                source: anyhow::Error::new(error.source),
            })?;
        let execution = VariableFinalExecution {
            request,
            ephemeris: ephem.as_ref(),
            enforce_terminal_events: true,
        };
        let row_results = if should_use_parallel_batch(n_rows) {
            #[cfg(test)]
            let test_output_address = states_out.as_ptr().addr();
            #[cfg(test)]
            variable_final_native_test_observer_record_parallel_entry(test_output_address);
            request
                .initial_eci_states
                .par_chunks(6)
                .zip(states_out.par_chunks_mut(6))
                .enumerate()
                .map_init(
                    VariableFinalReusableCache::default,
                    |cache, (idx, (init_chunk, out_chunk))| {
                        #[cfg(test)]
                        variable_final_native_test_observer_record(test_output_address);
                        self.propagate_variable_final_row_with_cache_into(
                            idx,
                            init_chunk,
                            out_chunk,
                            execution,
                            Some(cache),
                        )
                    },
                )
                .collect::<Vec<_>>()
        } else {
            let mut rows = Vec::with_capacity(n_rows);
            let mut cache = VariableFinalReusableCache::default();
            #[cfg(test)]
            let test_output_address = states_out.as_ptr().addr();
            for (idx, (init_chunk, out_chunk)) in request
                .initial_eci_states
                .chunks_exact(6)
                .zip(states_out.chunks_exact_mut(6))
                .enumerate()
            {
                #[cfg(test)]
                variable_final_native_test_observer_record(test_output_address);
                let row = self.propagate_variable_final_row_with_cache_into(
                    idx,
                    init_chunk,
                    out_chunk,
                    execution,
                    Some(&mut cache),
                );
                rows.push(row);
                if row != VariableFinalRowFailure::None {
                    break;
                }
            }
            rows
        };
        if let Some((row, failure)) = row_results
            .iter()
            .enumerate()
            .find(|(_, failure)| **failure != VariableFinalRowFailure::None)
        {
            return Err(VariableFinalNativeError::Row {
                row,
                failure: *failure,
            });
        }
        Ok(())
    }

    /// Execute one variable-final ECI batch on the process-global Rayon pool.
    /// Top-level widths above one must match the latched global pool width.
    /// Width one and nested Rayon calls stay serial.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the Rayon request, input shape, stepper
    /// authority, ephemeris coverage, or any row propagation is invalid.
    pub fn integrate_variable_final_global_into(
        &self,
        request: VariableFinalBatchRequest<'_>,
        states_out: &mut [f64],
        rayon_threads: usize,
        rayon_thread_budget: usize,
    ) -> Result<(), VariableFinalNativeError> {
        if rayon_threads == 0 || rayon_thread_budget == 0 {
            return Err(VariableFinalNativeError::RayonConfig(anyhow::anyhow!(
                "variable-final Rayon width and budget must be positive"
            )));
        }
        if rayon_threads > rayon_thread_budget {
            return Err(VariableFinalNativeError::RayonConfig(anyhow::anyhow!(
                "variable-final Rayon width {rayon_threads} exceeds budget {rayon_thread_budget}"
            )));
        }
        let available_cores = satpy_core::parallel_budget::available_cores();
        if rayon_thread_budget > available_cores {
            return Err(VariableFinalNativeError::RayonConfig(anyhow::anyhow!(
                "variable-final Rayon budget {rayon_thread_budget} exceeds available cores {available_cores}"
            )));
        }
        let n_rows = request.epoch_jd.len();
        let Some(state_values) = n_rows.checked_mul(6) else {
            return Err(VariableFinalNativeError::InputContract);
        };
        if request.initial_eci_states.len() != state_values
            || request.final_time_s.len() != n_rows
            || states_out.len() != state_values
            || request
                .ballistics
                .am_ratio
                .is_some_and(|values| values.len() != n_rows)
            || request
                .ballistics
                .cd
                .is_some_and(|values| values.len() != n_rows)
            || request
                .ballistics
                .cr
                .is_some_and(|values| values.len() != n_rows)
        {
            return Err(VariableFinalNativeError::InputContract);
        }
        validate_scalar_stepper_authority(&self.context.config, "scalar")
            .map_err(VariableFinalNativeError::UnsupportedStepper)?;
        let ephem = self
            .preflight_variable_final_ephemeris(&request)
            .map_err(|error| VariableFinalNativeError::Ephemeris {
                row: error.row,
                source: anyhow::Error::new(error.source),
            })?;
        self.preflight_variable_final_rows(&request, states_out, true)?;
        // `current_thread_index` is thread-local nested-call detection; it
        // does not construct Rayon’s global pool. Only a valid top-level W>1
        // request is allowed to latch the canonical scheduler configuration.
        let top_level_parallel = rayon_threads > 1 && rayon::current_thread_index().is_none();
        if top_level_parallel {
            nd_sched::init_global_pool_authoritative(rayon_threads)
                .map_err(VariableFinalNativeError::RayonConfig)?;
        }
        let execution = VariableFinalExecution {
            request,
            ephemeris: ephem.as_ref(),
            enforce_terminal_events: true,
        };

        #[cfg(test)]
        let test_output_address = states_out.as_ptr().addr();
        let failures = if top_level_parallel && n_rows > 1 {
            #[cfg(test)]
            variable_final_native_test_observer_record_parallel_entry(test_output_address);
            request
                .initial_eci_states
                .par_chunks(6)
                .zip(states_out.par_chunks_mut(6))
                .enumerate()
                .map(|(idx, (init_chunk, out_chunk))| {
                    #[cfg(test)]
                    variable_final_native_test_observer_record(test_output_address);
                    self.propagate_variable_final_row_with_cache_into(
                        idx, init_chunk, out_chunk, execution, None,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            let mut failures = Vec::with_capacity(n_rows);
            for (idx, (init_chunk, out_chunk)) in request
                .initial_eci_states
                .chunks_exact(6)
                .zip(states_out.chunks_exact_mut(6))
                .enumerate()
            {
                #[cfg(test)]
                variable_final_native_test_observer_record(test_output_address);
                failures.push(self.propagate_variable_final_row_with_cache_into(
                    idx, init_chunk, out_chunk, execution, None,
                ));
            }
            failures
        };
        if let Some((row, failure)) = failures
            .into_iter()
            .enumerate()
            .find(|(_, failure)| *failure != VariableFinalRowFailure::None)
        {
            return Err(VariableFinalNativeError::Row { row, failure });
        }
        Ok(())
    }

    fn scalar_propagation_context(
        &self,
        epoch_jd: f64,
        config: Arc<ForceConfig>,
    ) -> ScalarPropagationContext {
        ScalarPropagationContext::new(epoch_jd, config, self.context.gravity.clone())
    }

    #[cfg(feature = "scalar-leg-observer")]
    fn propagate_variable_final_checked_observed(
        &self,
        request: VariableFinalPropagation<'_>,
        reuse_cache: &mut VariableFinalReusableCache,
    ) -> Option<ObservedFinalLeg> {
        let key = VariableFinalIntegratorKey::new(&request);
        let VariableFinalPropagation {
            initial_equinoctial,
            epoch_jd,
            start_time_s,
            final_time_s,
            area_mass_ratio,
            drag_coefficient,
            reflectivity_coefficient,
            ephemeris,
        } = request;
        let midpoint_jd = epoch_jd + 0.5 * (start_time_s + final_time_s) / satpy_core::SEC_PER_DAY;
        if reuse_cache.key != Some(key) || reuse_cache.checked_integrator.is_none() {
            let config = Arc::new(self.config_for_jd_mid(
                ephemeris,
                midpoint_jd,
                area_mass_ratio,
                drag_coefficient,
                reflectivity_coefficient,
            ));
            let context = self.scalar_propagation_context(epoch_jd, config);
            let Ok(integrator) = ReusableFinalCheckedIntegrator::new(context) else {
                return None;
            };
            reuse_cache.checked_integrator = Some(integrator);
            reuse_cache.integrator = None;
            reuse_cache.key = Some(key);
        }
        let integrator = reuse_cache.checked_integrator.as_mut()?;
        Some(integrator.propagate_checked_observed(initial_equinoctial, start_time_s, final_time_s))
    }

    fn propagate_variable_final_checked(
        &self,
        request: VariableFinalPropagation<'_>,
        reuse_cache: Option<&mut VariableFinalReusableCache>,
    ) -> Result<[f64; 6], FinalPropagationFailure> {
        let key = VariableFinalIntegratorKey::new(&request);
        let VariableFinalPropagation {
            initial_equinoctial,
            epoch_jd,
            start_time_s,
            final_time_s,
            area_mass_ratio,
            drag_coefficient,
            reflectivity_coefficient,
            ephemeris,
        } = request;
        let midpoint_jd = epoch_jd + 0.5 * (start_time_s + final_time_s) / satpy_core::SEC_PER_DAY;
        if let Some(cache) = reuse_cache {
            if cache.key != Some(key) || cache.checked_integrator.is_none() {
                let config = Arc::new(self.config_for_jd_mid(
                    ephemeris,
                    midpoint_jd,
                    area_mass_ratio,
                    drag_coefficient,
                    reflectivity_coefficient,
                ));
                let context = self.scalar_propagation_context(epoch_jd, config);
                let integrator = ReusableFinalCheckedIntegrator::new(context)?;
                cache.checked_integrator = Some(integrator);
                cache.integrator = None;
                cache.key = Some(key);
            }
            let Some(integrator) = cache.checked_integrator.as_mut() else {
                return Err(FinalPropagationFailure::IntegrationFailure);
            };
            return integrator.propagate_checked(initial_equinoctial, start_time_s, final_time_s);
        }

        let config = Arc::new(self.config_for_jd_mid(
            ephemeris,
            midpoint_jd,
            area_mass_ratio,
            drag_coefficient,
            reflectivity_coefficient,
        ));
        let context = self.scalar_propagation_context(epoch_jd, config);
        let final_times = [final_time_s];
        integrate_final_checked(
            ScalarPropagationRequest::new(
                &context,
                initial_equinoctial,
                &final_times,
                start_time_s,
                final_time_s,
            )
            .with_events(true),
        )
    }

    fn propagate_variable_final_no_event(
        &self,
        request: VariableFinalPropagation<'_>,
        reuse_cache: Option<&mut VariableFinalReusableCache>,
    ) -> Result<[f64; 6], FinalPropagationFailure> {
        let key = VariableFinalIntegratorKey::new(&request);
        let VariableFinalPropagation {
            initial_equinoctial,
            epoch_jd,
            start_time_s,
            final_time_s,
            area_mass_ratio,
            drag_coefficient,
            reflectivity_coefficient,
            ephemeris,
        } = request;
        let midpoint_jd = epoch_jd + 0.5 * (start_time_s + final_time_s) / satpy_core::SEC_PER_DAY;
        if let Some(cache) = reuse_cache {
            if cache.key == Some(key) {
                if let Some(integrator) = cache.integrator.as_mut() {
                    return integrator.propagate(initial_equinoctial, start_time_s, final_time_s);
                }
            }
            let config = Arc::new(self.config_for_jd_mid(
                ephemeris,
                midpoint_jd,
                area_mass_ratio,
                drag_coefficient,
                reflectivity_coefficient,
            ));
            let context = self.scalar_propagation_context(epoch_jd, config);
            let integrator = ReusableFinalNoEventIntegrator::new(context);
            let Ok(integrator) = integrator else {
                return Err(FinalPropagationFailure::IntegrationFailure);
            };
            cache.integrator = Some(integrator);
            cache.key = Some(key);
            let Some(integrator) = cache.integrator.as_mut() else {
                return Err(FinalPropagationFailure::IntegrationFailure);
            };
            return integrator.propagate(initial_equinoctial, start_time_s, final_time_s);
        }

        let config = Arc::new(self.config_for_jd_mid(
            ephemeris,
            midpoint_jd,
            area_mass_ratio,
            drag_coefficient,
            reflectivity_coefficient,
        ));
        let context = self.scalar_propagation_context(epoch_jd, config);
        let integrator = ReusableFinalNoEventIntegrator::new(context);
        let Ok(mut integrator) = integrator else {
            return Err(FinalPropagationFailure::IntegrationFailure);
        };
        integrator.propagate(initial_equinoctial, start_time_s, final_time_s)
    }

    #[cfg(feature = "scalar-leg-observer")]
    fn propagate_variable_final_row_observed_with_cache_into(
        &self,
        idx: usize,
        init_chunk: &[f64],
        out_chunk: &mut [f64],
        execution: VariableFinalExecution<'_>,
        reuse_cache: &mut VariableFinalReusableCache,
        observation_slot: &mut Option<ObservedVariableFinalRow>,
    ) -> VariableFinalRowFailure {
        let VariableFinalExecution {
            request,
            ephemeris: ephem,
            enforce_terminal_events,
        } = execution;
        if init_chunk.len() != 6 || out_chunk.len() != 6 {
            return VariableFinalRowFailure::InvalidInput;
        }
        let prepared = match self.prepare_variable_final_row(
            idx,
            init_chunk,
            &request,
            enforce_terminal_events,
        ) {
            Ok(prepared) => prepared,
            Err(failure) => {
                out_chunk.fill(f64::NAN);
                return failure;
            }
        };
        let propagation = VariableFinalPropagation {
            initial_equinoctial: prepared.initial_equinoctial,
            epoch_jd: prepared.epoch_jd,
            start_time_s: request.t0_s,
            final_time_s: prepared.final_time_s,
            area_mass_ratio: prepared.area_mass_ratio,
            drag_coefficient: prepared.drag_coefficient,
            reflectivity_coefficient: prepared.reflectivity_coefficient,
            ephemeris: ephem,
        };
        let observed = self.propagate_variable_final_checked_observed(propagation, reuse_cache);
        let Some(observed) = observed else {
            out_chunk.fill(f64::NAN);
            return VariableFinalRowFailure::Propagation(
                FinalPropagationFailure::IntegrationFailure,
            );
        };
        let ObservedFinalLeg {
            outcome: delta,
            metrics,
            terminal_status,
        } = observed;
        let outcome = match delta {
            Ok(delta_state) => {
                let mut baseline = [0.0f64; 6];
                equinoc_prop_from_impl(
                    &prepared.initial_equinoctial,
                    prepared.final_time_s - request.t0_s,
                    &mut baseline,
                );
                for ((output, baseline_value), delta_value) in
                    out_chunk.iter_mut().zip(baseline).zip(delta_state)
                {
                    *output = baseline_value + delta_value;
                }
                if out_chunk.iter().all(|value| value.is_finite()) {
                    let mut endpoint = [0.0; 6];
                    endpoint.copy_from_slice(out_chunk);
                    Ok(endpoint)
                } else {
                    out_chunk.fill(f64::NAN);
                    Err(VariableFinalRowFailure::NonFiniteOutput)
                }
            }
            Err(error) => {
                out_chunk.fill(f64::NAN);
                Err(VariableFinalRowFailure::Propagation(error))
            }
        };
        let failure = outcome
            .as_ref()
            .map_or_else(|failure| *failure, |_| VariableFinalRowFailure::None);
        *observation_slot = Some(ObservedVariableFinalRow {
            outcome,
            metrics,
            terminal_status,
        });
        failure
    }

    fn propagate_variable_final_row_with_cache_into(
        &self,
        idx: usize,
        init_chunk: &[f64],
        out_chunk: &mut [f64],
        execution: VariableFinalExecution<'_>,
        mut reuse_cache: Option<&mut VariableFinalReusableCache>,
    ) -> VariableFinalRowFailure {
        let VariableFinalExecution {
            request,
            ephemeris: ephem,
            enforce_terminal_events,
        } = execution;
        if init_chunk.len() != 6 || out_chunk.len() != 6 {
            return VariableFinalRowFailure::InvalidInput;
        }
        let prepared = match self.prepare_variable_final_row(
            idx,
            init_chunk,
            &request,
            enforce_terminal_events,
        ) {
            Ok(prepared) => prepared,
            Err(failure) => {
                out_chunk.fill(f64::NAN);
                return failure;
            }
        };

        let propagation = VariableFinalPropagation {
            initial_equinoctial: prepared.initial_equinoctial,
            epoch_jd: prepared.epoch_jd,
            start_time_s: request.t0_s,
            final_time_s: prepared.final_time_s,
            area_mass_ratio: prepared.area_mass_ratio,
            drag_coefficient: prepared.drag_coefficient,
            reflectivity_coefficient: prepared.reflectivity_coefficient,
            ephemeris: ephem,
        };
        let delta = if enforce_terminal_events {
            self.propagate_variable_final_checked(propagation, reuse_cache.as_deref_mut())
        } else {
            self.propagate_variable_final_no_event(propagation, reuse_cache)
        };
        let delta_state = match delta {
            Ok(delta_state) => delta_state,
            Err(error) => {
                out_chunk.fill(f64::NAN);
                return VariableFinalRowFailure::Propagation(error);
            }
        };

        let mut baseline = [0.0f64; 6];
        equinoc_prop_from_impl(
            &prepared.initial_equinoctial,
            prepared.final_time_s - request.t0_s,
            &mut baseline,
        );
        for ((output, baseline_value), delta_value) in
            out_chunk.iter_mut().zip(baseline).zip(delta_state)
        {
            *output = baseline_value + delta_value;
        }
        if !out_chunk.iter().all(|value| value.is_finite()) {
            out_chunk.fill(f64::NAN);
            return VariableFinalRowFailure::NonFiniteOutput;
        }
        VariableFinalRowFailure::None
    }

    // `jd_mid` is built at the call site as
    // `jd0 + 0.5 * (t0_s + tf_s) / SEC_PER_DAY`, which is one of the five sites
    // in task #27. Typing it as UTC records the SCALE; it does
    // nothing about the fixed-86400 day length. The two are orthogonal.
    fn config_for_jd_mid(
        &self,
        ephem: &crate::precomputed_ephem::AllPrecomputedEphemeris,
        jd_mid: f64,
        am_ratio: f64,
        cd: f64,
        cr: f64,
    ) -> ForceConfig {
        use crate::precomputed_ephem::Body;
        let base = self.context.config.as_ref();
        let dynamic_ephemeris_flags = base.required_dynamic_ephemeris_flags();
        let utc_mid = jb_rs::drivers::UtcJulianDay::new(jd_mid).ok();
        let position = |body: Body, flag: i32, fixed: Option<[f64; 3]>| {
            if (dynamic_ephemeris_flags & flag) != 0 {
                utc_mid.and_then(|utc| {
                    ephem
                        .get(body)
                        .and_then(|table| table.position_at_part_a_utc_jd(utc).ok())
                })
            } else {
                fixed
            }
        };
        let sun_pos = position(
            Body::Sun,
            crate::types::ForceFlags::SUN_GRAVITY,
            base.sun_pos,
        );
        let moon_pos = position(
            Body::Moon,
            crate::types::ForceFlags::MOON_GRAVITY,
            base.moon_pos,
        );
        let jupiter_pos = position(
            Body::Jupiter,
            crate::types::ForceFlags::JUPITER_GRAVITY,
            base.jupiter_pos,
        );
        let venus_pos = position(
            Body::Venus,
            crate::types::ForceFlags::VENUS_GRAVITY,
            base.venus_pos,
        );
        let mars_pos = position(
            Body::Mars,
            crate::types::ForceFlags::MARS_GRAVITY,
            base.mars_pos,
        );
        let saturn_pos = position(
            Body::Saturn,
            crate::types::ForceFlags::SATURN_GRAVITY,
            base.saturn_pos,
        );
        let invariant = |flag: i32, value: Option<[f64; 3]>, mu: f64| {
            if (dynamic_ephemeris_flags & flag) != 0 {
                None
            } else {
                value.and_then(|pos| crate::types::BodyInvariants::precompute(&pos, mu))
            }
        };

        ForceConfig {
            am_ratio,
            cd,
            cr,
            sun_pos,
            moon_pos,
            jupiter_pos,
            venus_pos,
            mars_pos,
            saturn_pos,
            dynamic_ephemeris_flags,
            sun_invariants: invariant(crate::types::ForceFlags::SUN_GRAVITY, sun_pos, base.mu_sun),
            moon_invariants: invariant(
                crate::types::ForceFlags::MOON_GRAVITY,
                moon_pos,
                base.mu_moon,
            ),
            jupiter_invariants: invariant(
                crate::types::ForceFlags::JUPITER_GRAVITY,
                jupiter_pos,
                base.mu_jupiter,
            ),
            venus_invariants: invariant(
                crate::types::ForceFlags::VENUS_GRAVITY,
                venus_pos,
                base.mu_venus,
            ),
            mars_invariants: invariant(
                crate::types::ForceFlags::MARS_GRAVITY,
                mars_pos,
                base.mu_mars,
            ),
            saturn_invariants: invariant(
                crate::types::ForceFlags::SATURN_GRAVITY,
                saturn_pos,
                base.mu_saturn,
            ),
            dt_max: base.dt_max,
            ..*self.context.config
        }
    }
}
