//! Batch processing for parallel integration of multiple trajectories.
//!
//! This module provides functions for integrating multiple initial states
//! in parallel, optimized for UKF sigma point propagation.

use anyhow::{anyhow, ensure, Context, Result};
use rayon::prelude::*;
use std::sync::Arc;

use crate::config::get_global_coeffs_packed;
use crate::integrator::{
    integrate_adaptive, integrate_final_checked, validate_scalar_stepper_authority,
    ReusableFinalNoEventIntegrator, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest,
};
use crate::types::{ForceConfig, StepperMethod};
use crate::utils::write_batch_output_aligned;
use crate::{
    init_states_shape_message, output_shape_mismatch_message, CONSTANTS_NOT_LOADED_MESSAGE,
};
use satpy_core::SEC_PER_DAY;

/// Default minimum states for parallel batch integration.
/// Raised to 32 so Rayon stays dormant for typical UKF batch sizes (13-15).
/// Event-level parallelism is handled by the caller instead.
pub(crate) const LIGHTYEAR_PAR_THRESHOLD: usize = 32;
const STATE_WIDTH: usize = 6;

#[derive(Clone, Copy)]
struct BatchDimensions {
    input_len: usize,
    output_len: usize,
    row_len: usize,
    n_times: usize,
}

impl BatchDimensions {
    fn new(n_sigma: usize, n_times: usize) -> Result<Self> {
        let input_len = n_sigma
            .checked_mul(STATE_WIDTH)
            .ok_or_else(|| anyhow!("batch initial-state length overflows usize"))?;
        let row_len = n_times
            .checked_mul(STATE_WIDTH)
            .ok_or_else(|| anyhow!("batch output-row length overflows usize"))?;
        let output_len = n_sigma
            .checked_mul(row_len)
            .ok_or_else(|| anyhow!("batch output length overflows usize"))?;
        Ok(Self {
            input_len,
            output_len,
            row_len,
            n_times,
        })
    }
}

/// Optional row-local ballistic coefficients for a batch propagation.
///
/// Omitted arrays retain the immutable force authority's corresponding value.
#[derive(Clone, Copy, Default)]
pub struct BatchBallistics<'a> {
    pub am_ratio: Option<&'a [f64]>,
    pub cd: Option<&'a [f64]>,
    pub cr: Option<&'a [f64]>,
}

impl BatchBallistics<'_> {
    #[must_use]
    const fn has_any(self) -> bool {
        self.am_ratio.is_some() || self.cd.is_some() || self.cr.is_some()
    }

    fn validate_lengths(self, n_sigma: usize) -> Result<()> {
        for (name, values) in [
            ("am_ratio", self.am_ratio),
            ("cd", self.cd),
            ("cr", self.cr),
        ] {
            if let Some(values) = values {
                ensure!(
                    values.len() >= n_sigma,
                    "per-state {name} values have length {}, need at least {n_sigma}",
                    values.len()
                );
            }
        }
        Ok(())
    }

    fn value_or_default(
        values: Option<&[f64]>,
        index: usize,
        default: f64,
        name: &str,
    ) -> Result<f64> {
        values.map_or_else(
            || Ok(default),
            |values| {
                values
                    .get(index)
                    .copied()
                    .ok_or_else(|| anyhow!("per-state {name} value missing for row {index}"))
            },
        )
    }

    fn config_for_state(self, index: usize, base: &Arc<ForceConfig>) -> Result<Arc<ForceConfig>> {
        if !self.has_any() {
            return Ok(Arc::clone(base));
        }
        Ok(Arc::new(ForceConfig {
            am_ratio: Self::value_or_default(self.am_ratio, index, base.am_ratio, "am_ratio")?,
            cd: Self::value_or_default(self.cd, index, base.cd, "cd")?,
            cr: Self::value_or_default(self.cr, index, base.cr, "cr")?,
            ..(**base)
        }))
    }
}

/// Immutable inputs for one native equinoctial batch propagation.
///
/// The force configuration is the sole authority for tolerance and integrator
/// method. Row-local ballistics may vary only the three physical coefficients.
#[derive(Clone, Copy)]
pub struct BatchPropagationRequest<'a> {
    pub initial_equinoc_states: &'a [f64],
    pub t_eval: &'a [f64],
    pub t0_s: f64,
    pub t_final_s: f64,
    pub epoch_jd: f64,
    pub force_config: ForceConfig,
    pub ballistics: BatchBallistics<'a>,
}

/// Check if parallel batch processing should be used.
///
/// Returns false if batch is small, nested, or no explicit global pool exists.
/// A normal batch must never create Rayon’s global pool: only the explicit
/// variable-final global entry point owns first-touch scheduler authority.
#[inline]
#[must_use]
pub fn should_use_parallel_batch(n_items: usize) -> bool {
    n_items >= LIGHTYEAR_PAR_THRESHOLD
        && rayon::current_thread_index().is_none()
        && nd_sched::configured_global_pool_threads().is_some_and(|threads| threads > 1)
}

/// Final-state integrator reuse requires one shared force config across the
/// batch, so any per-state ballistic override disables it.
#[inline]
const fn reuse_final_only_enabled(use_final_only: bool, has_per_state_config: bool) -> bool {
    use_final_only && !has_per_state_config
}

fn state_from_chunk(values: &[f64]) -> Result<crate::types::StateType> {
    let state: &[f64; STATE_WIDTH] = values
        .try_into()
        .map_err(|_| anyhow!("batch initial-state row has invalid width"))?;
    Ok(*state)
}

fn write_final_state(out: &mut [f64], state: &crate::types::StateType) -> Result<()> {
    let destination = out
        .get_mut(..STATE_WIDTH)
        .ok_or_else(|| anyhow!("batch final-state row has invalid width"))?;
    destination.copy_from_slice(state);
    Ok(())
}

fn prepare_batch_config(
    mut config: ForceConfig,
    jd0: f64,
    start_time_s: f64,
    end_time_s: f64,
) -> Result<Arc<ForceConfig>> {
    // Encke already carries two-body gravity in its reference orbit. Any
    // nonzero spherical-harmonic model must remove that central term at every
    // entry boundary, matching the session API.
    config.subtract_first_order = config.subtract_first_order || config.sph_order > 0;
    let jd_start = jd0 + start_time_s / SEC_PER_DAY;
    let jd_end = jd0 + end_time_s / SEC_PER_DAY;
    config
        .with_ephemeris_for_arc(jd_start, jd_end)
        .map(Arc::new)
        .map_err(anyhow::Error::new)
}

fn write_complete_sampled_output(
    out: &mut [f64],
    t_eval: &[f64],
    result: &crate::types::IntegrationResult,
    direction: f64,
) -> Result<()> {
    if let Some(error) = result.terminal_eclipse_error {
        out.fill(f64::NAN);
        return Err(anyhow::Error::new(error));
    }
    if result.terminal_event_fired {
        out.fill(f64::NAN);
        return Err(anyhow!(
            "batch sampled propagation failed: {}",
            result.terminal_event_name
        ));
    }
    if !write_batch_output_aligned(out, t_eval, result, direction) {
        return Err(anyhow!(
            "batch sampled propagation did not return every requested output"
        ));
    }
    if !out.iter().all(|value| value.is_finite()) {
        out.fill(f64::NAN);
        return Err(anyhow!(
            "batch sampled propagation returned incomplete or non-finite output"
        ));
    }
    Ok(())
}

// ============================================================================
// Batch Integration Functions
// ============================================================================

/// Batch integrate multiple trajectories under one immutable force authority.
///
/// # Errors
///
/// Returns an error when batch shapes, force authority, ephemeris coverage, or
/// any propagation row is invalid.
pub fn integrate_batch_native(request: BatchPropagationRequest<'_>) -> Result<Vec<f64>> {
    let n_sigma = request.initial_equinoc_states.len() / STATE_WIDTH;
    let dimensions = BatchDimensions::new(n_sigma, request.t_eval.len())?;
    let mut states_flat = Vec::new();
    states_flat
        .try_reserve_exact(dimensions.output_len)
        .context("batch output allocation failed")?;
    states_flat.resize(dimensions.output_len, 0.0);
    integrate_batch_native_into(request, &mut states_flat)?;
    Ok(states_flat)
}

/// Batch integrate trajectories into a caller-owned flat buffer.
///
/// # Errors
///
/// Returns an error when batch shapes, per-state overrides, force authority,
/// ephemeris coverage, or any propagation row is invalid.
pub fn integrate_batch_native_into(
    request: BatchPropagationRequest<'_>,
    states_out: &mut [f64],
) -> Result<()> {
    let BatchPropagationRequest {
        initial_equinoc_states,
        t_eval,
        t0_s: start_time_s,
        t_final_s: end_time_s,
        epoch_jd: jd0,
        force_config: config,
        ballistics,
    } = request;
    let stepper = config.integrator_method;
    crate::rhs::validate_atmosphere_model_code(config.atm_model)?;
    validate_scalar_stepper_authority(&config, "batch")?;
    if matches!(stepper, StepperMethod::Esdirk43) {
        crate::rhs_dual::validate_dual_force_config(&config)?;
    }
    ensure!(
        initial_equinoc_states.len() % STATE_WIDTH == 0,
        "batch initial-state length {} is not divisible by {STATE_WIDTH}",
        initial_equinoc_states.len()
    );
    let n_sigma = initial_equinoc_states.len() / STATE_WIDTH;
    let dimensions = BatchDimensions::new(n_sigma, t_eval.len())?;
    ensure!(
        initial_equinoc_states.len() == dimensions.input_len,
        "{}",
        init_states_shape_message(&[n_sigma, STATE_WIDTH])
    );
    ensure!(
        states_out.len() == dimensions.output_len,
        "{}",
        output_shape_mismatch_message(n_sigma, dimensions.n_times, &[states_out.len()])
    );
    if dimensions.n_times == 0 || n_sigma == 0 {
        return Ok(());
    }
    ballistics.validate_lengths(n_sigma)?;
    // A failed row must never leave plausible zero-filled science behind.
    states_out.fill(f64::NAN);

    let config_arc = prepare_batch_config(config, jd0, start_time_s, end_time_s)?;
    let direction = if end_time_s >= start_time_s {
        1.0
    } else {
        -1.0
    };

    let packed = get_global_coeffs_packed().ok_or_else(|| anyhow!(CONSTANTS_NOT_LOADED_MESSAGE))?;
    let gravity = ScalarGravityAssets::new(packed);

    let use_parallel = should_use_parallel_batch(n_sigma);
    let use_final_only = dimensions.n_times == 1
        && t_eval
            .first()
            .is_some_and(|time| (*time - end_time_s).abs() < 1e-9);
    let can_reuse_final_only = reuse_final_only_enabled(use_final_only, ballistics.has_any());
    if can_reuse_final_only {
        let context = ScalarPropagationContext::new(jd0, config_arc, gravity);
        if use_parallel {
            initial_equinoc_states
                .par_chunks_exact(STATE_WIDTH)
                .zip(states_out.par_chunks_exact_mut(dimensions.row_len))
                .try_for_each_init(
                    || None,
                    |reusable, (init_chunk, out_chunk)| -> Result<()> {
                        let reusable = match reusable {
                            Some(reusable) => reusable,
                            None => reusable
                                .insert(ReusableFinalNoEventIntegrator::new(context.clone())?),
                        };
                        let state = reusable
                            .propagate(state_from_chunk(init_chunk)?, start_time_s, end_time_s)
                            .map_err(anyhow::Error::new)?;
                        write_final_state(out_chunk, &state)
                    },
                )?;
        } else {
            let mut reusable = ReusableFinalNoEventIntegrator::new(context)?;
            initial_equinoc_states
                .chunks_exact(STATE_WIDTH)
                .zip(states_out.chunks_exact_mut(dimensions.row_len))
                .try_for_each(|(init_chunk, out_chunk)| -> Result<()> {
                    let state = reusable
                        .propagate(state_from_chunk(init_chunk)?, start_time_s, end_time_s)
                        .map_err(anyhow::Error::new)?;
                    write_final_state(out_chunk, &state)
                })?;
        }
    } else {
        let propagate_one =
            |index: usize, init_chunk: &[f64], out_chunk: &mut [f64]| -> Result<()> {
                let config_for_state = ballistics.config_for_state(index, &config_arc)?;
                let init_state = state_from_chunk(init_chunk)?;
                let context = ScalarPropagationContext::new(jd0, config_for_state, gravity.clone());
                if use_final_only {
                    let state = integrate_final_checked(ScalarPropagationRequest::new(
                        &context,
                        init_state,
                        t_eval,
                        start_time_s,
                        end_time_s,
                    ))
                    .map_err(anyhow::Error::new)?;
                    write_final_state(out_chunk, &state)
                } else {
                    let result = integrate_adaptive(ScalarPropagationRequest::new(
                        &context,
                        init_state,
                        t_eval,
                        start_time_s,
                        end_time_s,
                    ))
                    .map_err(anyhow::Error::new)?;
                    write_complete_sampled_output(out_chunk, t_eval, &result, direction)
                }
            };
        if use_parallel {
            initial_equinoc_states
                .par_chunks_exact(STATE_WIDTH)
                .enumerate()
                .zip(states_out.par_chunks_exact_mut(dimensions.row_len))
                .try_for_each(|((index, init_chunk), out_chunk)| {
                    propagate_one(index, init_chunk, out_chunk)
                })?;
        } else {
            initial_equinoc_states
                .chunks_exact(STATE_WIDTH)
                .enumerate()
                .zip(states_out.chunks_exact_mut(dimensions.row_len))
                .try_for_each(|((index, init_chunk), out_chunk)| {
                    propagate_one(index, init_chunk, out_chunk)
                })?;
        }
    }

    ensure!(
        states_out.iter().all(|value| value.is_finite()),
        "batch propagation failed without a complete finite result"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GlobalCoeffs, GLOBAL_COEFFS};
    use crate::integrator::{
        integrate_adaptive, integrate_final_checked, ScalarGravityAssets, ScalarPropagationContext,
        ScalarPropagationRequest,
    };
    use crate::types::{ForceFlags, StepperMethod};
    use satpy_core::{eci2equinoc_impl, pack_gravity_coeffs, PackedGravityCoeffs};

    const TEST_STACK_SIZE: usize = 64 * 1024 * 1024;

    #[test]
    fn reusable_final_lane_requires_single_time_and_shared_config() {
        assert!(reuse_final_only_enabled(true, false));
        assert!(!reuse_final_only_enabled(false, false));
        assert!(!reuse_final_only_enabled(true, true));
    }

    #[test]
    fn incomplete_sampled_row_returns_error_and_remains_nan() {
        let result = crate::types::IntegrationResult {
            times: vec![0.0, 1.0],
            states: vec![[1.0; 6], [2.0; 6]],
            ..crate::types::IntegrationResult::default()
        };
        let mut output = vec![42.0; 3 * 6];

        let outcome = write_complete_sampled_output(&mut output, &[0.0, 1.0, 2.0], &result, 1.0);

        assert!(matches!(
            outcome,
            Err(error) if error.to_string().contains("did not return every requested output")
        ));
        assert!(output.iter().all(|value| value.is_nan()));
    }

    fn run_with_stack<F: FnOnce() + Send + 'static>(f: F) {
        std::thread::Builder::new()
            .stack_size(TEST_STACK_SIZE)
            .spawn(f)
            .expect("failed to spawn test thread")
            .join()
            .expect("test thread panicked");
    }

    fn test_usize_as_f64(value: usize) -> f64 {
        u32::try_from(value).map_or(f64::NAN, f64::from)
    }

    /// Publishes a synthetic pack to `GLOBAL_COEFFS` and returns it TOGETHER
    /// with the guard that serialises this test against every other publishing
    /// test in this binary (see `config::lock_global_coeffs_for_test`). Bind
    /// the result to a named variable so the guard lives for the whole test:
    /// the batch entry points read the global back and must see THIS install.
    fn install_test_coeffs(
        order: usize,
    ) -> (std::sync::MutexGuard<'static, ()>, Arc<PackedGravityCoeffs>) {
        let guard = crate::config::lock_global_coeffs_for_test();
        let stride = order.saturating_add(2);
        let total_size = stride.saturating_mul(stride);
        let mut c_coeffs = vec![0.0; total_size];
        let mut s_coeffs = vec![0.0; total_size];
        if let Some(c00) = c_coeffs.first_mut() {
            *c00 = 1.0;
        }
        for (degree, (c_row, s_row)) in c_coeffs
            .chunks_exact_mut(stride)
            .zip(s_coeffs.chunks_exact_mut(stride))
            .enumerate()
        {
            if !(2..=order).contains(&degree) {
                continue;
            }
            if let Some(c_zero_order) = c_row.first_mut() {
                *c_zero_order = 1e-3 / test_usize_as_f64(degree).powi(2);
            }
            for (degree_order, (cosine, sine)) in c_row
                .iter_mut()
                .zip(s_row.iter_mut())
                .enumerate()
                .skip(1)
                .take(degree)
            {
                let magnitude = 1e-6 / test_usize_as_f64(degree.saturating_mul(degree_order));
                *cosine = magnitude;
                *sine = magnitude * 0.5;
            }
        }
        let packed = Arc::new(
            pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order)
                .expect("batch test gravity coefficients must pack"),
        );
        GLOBAL_COEFFS.store(Arc::new(GlobalCoeffs::Loaded(Arc::clone(&packed))));
        (guard, packed)
    }

    fn create_test_config() -> ForceConfig {
        ForceConfig {
            sph_order: 5,
            force_flags: 0,
            dt_max: 60.0,
            eps: 1e-8,
            ..ForceConfig::default()
        }
    }

    fn create_test_sigma_points(n_sigma: usize) -> Vec<f64> {
        let base_states = [
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            [7000.0, 100.0, -50.0, -0.1, 7.45, 0.2],
            [7200.0, -120.0, 80.0, 0.05, 7.30, -0.15],
        ];

        let mut init_states_flat = Vec::new();
        for (index, eci) in base_states
            .iter()
            .copied()
            .cycle()
            .take(n_sigma)
            .enumerate()
        {
            let scale = 1e-6 * (test_usize_as_f64(index) - test_usize_as_f64(n_sigma) * 0.5);
            let mut equ = [0.0; 6];
            eci2equinoc_impl(&eci, 6, 0.0, 0.0, &mut equ);
            for (axis, &value) in equ.iter().enumerate() {
                init_states_flat.push(value + scale * (test_usize_as_f64(axis) + 1.0));
            }
        }
        init_states_flat
    }

    fn assert_flat_close(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len(), "vector length mismatch");
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= tol,
                "component {idx} differs: lhs={a} rhs={b} diff={diff} tol={tol}"
            );
        }
    }

    #[test]
    fn batch_rejects_model4_and_model5_auto_and_implicit_before_empty_batch_returns() {
        for model in [4, 5] {
            for stepper in [StepperMethod::Esdirk43, StepperMethod::Auto] {
                let mut config = ForceConfig {
                    atm_model: model,
                    integrator_method: stepper,
                    ..ForceConfig::default()
                };
                config.eps = 1e-8;
                let error = integrate_batch_native_into(
                    BatchPropagationRequest {
                        initial_equinoc_states: &[],
                        t_eval: &[],
                        t0_s: 0.0,
                        t_final_s: 0.0,
                        epoch_jd: 2_460_310.5,
                        force_config: config,
                        ballistics: BatchBallistics::default(),
                    },
                    &mut [],
                )
                .expect_err("guarded HF method must fail before zero-row return");
                assert!(
                    error.to_string().contains("requires explicit batch method"),
                    "model={model} stepper={stepper:?}: {error:#}"
                );
            }
        }
    }

    #[test]
    fn batch_rejects_short_per_state_config_before_propagation() {
        let _coefficients = install_test_coeffs(5);
        let initial_state = [6_778.0, 0.0, 0.0, 0.0, 7.67, 0.0];
        let mut output = [0.0; 6];

        let mut config = create_test_config();
        config.eps = 1e-8;
        config.integrator_method = StepperMethod::Dopri5Compat;
        let result = integrate_batch_native_into(
            BatchPropagationRequest {
                initial_equinoc_states: &initial_state,
                t_eval: &[0.0],
                t0_s: 0.0,
                t_final_s: 0.0,
                epoch_jd: 2_460_310.5,
                force_config: config,
                ballistics: BatchBallistics {
                    am_ratio: Some(&[]),
                    cd: None,
                    cr: None,
                },
            },
            &mut output,
        );

        assert!(matches!(result, Err(error) if error.to_string().contains("am_ratio")));
    }

    #[test]
    fn batch_final_only_matches_individual_final_only() {
        let (_global_coeffs_lock, packed) = install_test_coeffs(5);
        let jd0 = 2_460_310.5;
        let t_eval = [600.0f64];
        let tf_s = 600.0;
        let eps = 1e-8;

        let eci_states = [
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            [7000.0, 100.0, -50.0, -0.1, 7.45, 0.2],
            [7200.0, -120.0, 80.0, 0.05, 7.30, -0.15],
        ];

        let mut init_states_flat = Vec::new();
        for eci in eci_states {
            let mut equ = [0.0; 6];
            eci2equinoc_impl(&eci, 6, 0.0, 0.0, &mut equ);
            init_states_flat.extend_from_slice(&equ);
        }

        let mut batch_config = create_test_config();
        batch_config.eps = eps;
        batch_config.integrator_method = StepperMethod::Dopri5Compat;
        let batch = integrate_batch_native(BatchPropagationRequest {
            initial_equinoc_states: &init_states_flat,
            t_eval: &t_eval,
            t0_s: 0.0,
            t_final_s: tf_s,
            epoch_jd: jd0,
            force_config: batch_config,
            ballistics: BatchBallistics::default(),
        })
        .expect("batch final-only propagation failed");

        for (state_index, (initial_state, output_state)) in init_states_flat
            .chunks_exact(STATE_WIDTH)
            .zip(batch.chunks_exact(STATE_WIDTH))
            .enumerate()
        {
            let mut expected = [0.0; 6];
            let mut individual_config = create_test_config();
            individual_config.subtract_first_order = true;
            individual_config.eps = eps;
            individual_config.integrator_method = StepperMethod::Dopri5Compat;
            let context = ScalarPropagationContext::new(
                jd0,
                Arc::new(individual_config),
                ScalarGravityAssets::new(Arc::clone(&packed)),
            );
            let individual = integrate_final_checked(ScalarPropagationRequest::new(
                &context,
                state_from_chunk(initial_state).expect("test state row has fixed width"),
                &t_eval,
                0.0,
                tf_s,
            ))
            .expect("individual final-only propagation failed");
            expected.copy_from_slice(&individual);
            for (component, (&got, &expected_component)) in
                output_state.iter().zip(expected.iter()).enumerate()
            {
                let diff = (got - expected_component).abs();
                assert!(
                    diff <= 1e-7,
                    "state {state_index} component {component} mismatch: got={got} expected={expected_component} diff={diff}"
                );
            }
        }
    }

    #[test]
    fn batch_singleton_interior_time_uses_sampled_output() {
        run_with_stack(|| {
            let (_global_coeffs_lock, packed) = install_test_coeffs(5);
            let jd0 = 2_460_310.5;
            let t_eval = [300.0];
            let t_final_s = 600.0;
            let mut initial = [0.0; STATE_WIDTH];
            eci2equinoc_impl(
                &[6_778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
                6,
                0.0,
                0.0,
                &mut initial,
            );
            let mut config = create_test_config();
            config.integrator_method = StepperMethod::Dopri5Compat;

            let expected_context = ScalarPropagationContext::new(
                jd0,
                Arc::new(ForceConfig {
                    subtract_first_order: true,
                    ..config
                }),
                ScalarGravityAssets::new(packed),
            );
            let expected = integrate_adaptive(ScalarPropagationRequest::new(
                &expected_context,
                initial,
                &t_eval,
                0.0,
                t_final_s,
            ))
            .expect("scalar interior sampled propagation")
            .states
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

            let actual = integrate_batch_native(BatchPropagationRequest {
                initial_equinoc_states: &initial,
                t_eval: &t_eval,
                t0_s: 0.0,
                t_final_s,
                epoch_jd: jd0,
                force_config: config,
                ballistics: BatchBallistics::default(),
            })
            .expect("batch interior sampled propagation");

            assert_flat_close(&actual, &expected, 1.0e-7);
        });
    }

    #[test]
    fn batch_final_only_repeated_calls_remain_stable_for_dopri5_and_auto() {
        run_with_stack(|| {
            let _coeffs = install_test_coeffs(5);
            let jd0 = 2_460_310.5;
            let t_eval = [600.0f64];
            let tf_s = 600.0;
            let eps = 1e-8;
            let n_sigma = 65;
            let init_states_flat = create_test_sigma_points(n_sigma);

            for stepper in [StepperMethod::Dopri5Compat, StepperMethod::Auto] {
                let mut config = create_test_config();
                config.eps = eps;
                config.integrator_method = stepper;
                let mut runs: Vec<Vec<f64>> = Vec::new();
                for _ in 0..3 {
                    let out = integrate_batch_native(BatchPropagationRequest {
                        initial_equinoc_states: &init_states_flat,
                        t_eval: &t_eval,
                        t0_s: 0.0,
                        t_final_s: tf_s,
                        epoch_jd: jd0,
                        force_config: config,
                        ballistics: BatchBallistics::default(),
                    })
                    .expect("batch final-only propagation failed");
                    runs.push(out);
                }

                let first = runs.first().expect("fixed repeat count has first result");
                for repeat in runs.iter().skip(1) {
                    assert_flat_close(first, repeat, 1e-7);
                }
            }
        });
    }

    #[test]
    fn batch_native_into_matches_allocating_native_for_final_only() {
        run_with_stack(|| {
            let _coeffs = install_test_coeffs(5);
            let jd0 = 2_460_310.5;
            let t_eval = [600.0f64];
            let tf_s = 600.0;
            let eps = 1e-8;
            let n_sigma = 65;
            let init_states_flat = create_test_sigma_points(n_sigma);

            let mut config = create_test_config();
            config.eps = eps;
            config.integrator_method = StepperMethod::Dopri5Compat;
            let expected = integrate_batch_native(BatchPropagationRequest {
                initial_equinoc_states: &init_states_flat,
                t_eval: &t_eval,
                t0_s: 0.0,
                t_final_s: tf_s,
                epoch_jd: jd0,
                force_config: config,
                ballistics: BatchBallistics::default(),
            })
            .expect("batch native reference propagation failed");

            let mut out = vec![f64::NAN; expected.len()];
            integrate_batch_native_into(
                BatchPropagationRequest {
                    initial_equinoc_states: &init_states_flat,
                    t_eval: &t_eval,
                    t0_s: 0.0,
                    t_final_s: tf_s,
                    epoch_jd: jd0,
                    force_config: config,
                    ballistics: BatchBallistics::default(),
                },
                &mut out,
            )
            .expect("batch native into propagation failed");

            assert_flat_close(&out, &expected, 1e-7);

            let mut too_short = vec![0.0; expected.len().saturating_sub(1)];
            assert!(integrate_batch_native_into(
                BatchPropagationRequest {
                    initial_equinoc_states: &init_states_flat,
                    t_eval: &t_eval,
                    t0_s: 0.0,
                    t_final_s: tf_s,
                    epoch_jd: jd0,
                    force_config: config,
                    ballistics: BatchBallistics::default(),
                },
                &mut too_short,
            )
            .is_err());
        });
    }

    #[test]
    fn multi_time_binary_srp_batch_matches_scalar_and_preserves_typed_failure() {
        run_with_stack(|| {
            let (_global_coeffs_lock, packed) = install_test_coeffs(5);
            let jd0 = 2_460_310.5;
            let t_eval = [0.0, 300.0, 600.0];
            let mut config = create_test_config();
            config.force_flags = ForceFlags::SRP;
            config.am_ratio = 0.02;
            config.cr = 1.3;
            config.sun_pos = Some([149_597_870.7, 0.0, 0.0]);
            config.subtract_first_order = true;
            config.eps = 1.0e-8;
            config.integrator_method = StepperMethod::Vern9;

            let mut init = [0.0; 6];
            eci2equinoc_impl(&[7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0], 6, 0.0, 0.0, &mut init);
            let batch = integrate_batch_native(BatchPropagationRequest {
                initial_equinoc_states: &init,
                t_eval: &t_eval,
                t0_s: 0.0,
                t_final_s: 600.0,
                epoch_jd: jd0,
                force_config: config,
                ballistics: BatchBallistics::default(),
            })
            .expect("multi-time binary SRP batch");
            let context = ScalarPropagationContext::new(
                jd0,
                Arc::new(config),
                ScalarGravityAssets::new(packed),
            );
            let scalar = integrate_adaptive(ScalarPropagationRequest::new(
                &context, init, &t_eval, 0.0, 600.0,
            ))
            .expect("scalar binary SRP propagation census");
            assert!(!scalar.terminal_event_fired, "{scalar:?}");
            let expected = scalar.states.into_iter().flatten().collect::<Vec<_>>();
            assert_flat_close(&batch, &expected, 0.0);

            let mut outside = [0.0; 6];
            eci2equinoc_impl(
                &[60_000.0, 0.0, 0.0, 0.0, 2.5, 0.0],
                6,
                0.0,
                0.0,
                &mut outside,
            );
            let mut output = vec![0.0; t_eval.len().saturating_mul(STATE_WIDTH)];
            let error = integrate_batch_native_into(
                BatchPropagationRequest {
                    initial_equinoc_states: &outside,
                    t_eval: &t_eval,
                    t0_s: 0.0,
                    t_final_s: 600.0,
                    epoch_jd: jd0,
                    force_config: config,
                    ballistics: BatchBallistics::default(),
                },
                &mut output,
            )
            .expect_err("outside eclipse envelope must fail batch");
            assert_eq!(
                error.downcast_ref::<crate::eclipse::EclipseError>(),
                Some(&crate::eclipse::EclipseError::Envelope)
            );
            assert!(output.iter().all(|value| value.is_nan()));
        });
    }
}
