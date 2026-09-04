//! Stiffness-adaptive meta-solver that switches between Tsit5 (explicit)
//! and ESDIRK4(3) (implicit) based on runtime step rejection patterns.
//!
//! One entry point, `integrate_auto_final`, which is what `StepperMethod::Auto`
//! dispatches to. A sampled-output twin used to sit beside it, reachable only
//! from its own test; it is gone rather than kept in step by hand.
//!
//! The current implementation uses a pragmatic try-explicit-then-fallback-to-implicit
//! approach: it first attempts integration with the explicit solver, and if the
//! result shows signs of excessive step rejection (high step count relative to
//! output points, or integration failure), it retries with the implicit ESDIRK solver.
//!
//! A more sophisticated mid-integration switching strategy can be layered on later.

use crate::integrator::LightyearSystem;
use crate::jacobian::compute_jacobian_unlatched;
use crate::rhs::LightyearRHS;
use crate::rhs_dual::LightyearDualRHS;

use crate::odesolve::{
    integrate_final, integrate_final_esdirk, ErrorControl, IntegrationResult, IntegrationStats,
    IntegrationStatus, IntegratorConfig, JacobianProvider, Method as OdeMethod,
};
use num_traits::ToPrimitive;
use satpy_core::GravityError;

// ============================================================================
// DualVec Jacobian Adapter
// ============================================================================

/// Adapter that bridges `crate::odesolve::JacobianProvider` with the
/// `lightyear_odeint_rs::jacobian::compute_jacobian_unlatched` function.
///
/// This allows the ESDIRK solver (which expects a generic `JacobianProvider`)
/// to use the `DualVec` automatic differentiation-based Jacobian computation.
pub struct DualVecJacobian<'a> {
    dual_rhs: &'a LightyearDualRHS,
}

impl<'a> DualVecJacobian<'a> {
    #[must_use]
    pub(crate) const fn new(dual_rhs: &'a LightyearDualRHS) -> Self {
        Self { dual_rhs }
    }
}

impl JacobianProvider for DualVecJacobian<'_> {
    fn jacobian(&self, t: f64, y: &[f64], jac: &mut [[f64; 6]; 6]) {
        let Some(delta) = y
            .get(..6)
            .and_then(|state| <&[f64; 6]>::try_from(state).ok())
        else {
            jac.fill([f64::NAN; 6]);
            return;
        };
        if compute_jacobian_unlatched(self.dual_rhs, delta, t, jac).is_err() {
            // `LightyearDualRHS::compute_internal` has retained the first exact
            // GravityError in its owned latch. This void trait has no typed error
            // channel, so NaN is only the stop signal; the public solver boundary
            // consumes and returns the exact failure before inspecting status.
            jac.fill([f64::NAN; 6]);
        }
    }
}

// ============================================================================
// Auto-switching meta-solver
// ============================================================================

/// Heuristic threshold: if the explicit solver uses more than this multiple
/// of `t_eval.len()` accepted steps, it is likely struggling with stiffness.
const STIFFNESS_STEP_RATIO: usize = 10;

fn invalid_final_result(y0: &[f64], t0: f64) -> IntegrationResult {
    IntegrationResult {
        t: t0,
        y: y0.to_vec(),
        status: IntegrationStatus::InvalidInput,
        stats: IntegrationStats::default(),
        event: None,
    }
}

/// Consume both RHS latches before returning a gravity failure.
///
/// The scalar lane takes precedence when both solver paths reported an error,
/// but the `DualVec` latch must still be drained so it cannot leak into a later
/// invocation on the same RHS instance.
fn consume_solver_gravity_errors(
    rhs: &LightyearRHS,
    dual_rhs: &LightyearDualRHS,
) -> Result<(), GravityError> {
    let scalar_error = rhs.take_gravity_error();
    let dual_error = dual_rhs.take_gravity_error();
    match (scalar_error, dual_error) {
        (Some(error), _) | (None, Some(error)) => Err(error),
        (None, None) => Ok(()),
    }
}

/// Integrate using the auto-switching meta-solver (final state only).
///
/// Same strategy as `integrate_auto_sampled` but returns only the final state.
///
/// # Errors
///
/// Returns an exact scalar or `DualVec` packed-gravity evaluation failure before
/// interpreting an ODE status or selecting a fallback result.
pub fn integrate_auto_final(
    rhs: &LightyearRHS,
    dual_rhs: &LightyearDualRHS,
    y0: &[f64],
    t0: f64,
    tf: f64,
    eps: f64,
    dt_max: f64,
    max_steps: usize,
    max_rejects: usize,
) -> Result<IntegrationResult, GravityError> {
    // Both RHS latches are scoped to this public solve boundary. The scalar
    // system runs first; the DualVec system is only evaluated by ESDIRK.
    rhs.clear_gravity_error();
    dual_rhs.reset_gravity_error();

    let Some(y0_arr) = y0
        .get(..6)
        .and_then(|state| <&[f64; 6]>::try_from(state).ok())
    else {
        consume_solver_gravity_errors(rhs, dual_rhs)?;
        return Ok(invalid_final_result(y0, t0));
    };
    let system = LightyearSystem { rhs };
    let eps_eff = eps.max(1e-12);

    let cfg = IntegratorConfig {
        error_control: ErrorControl::Absolute { eps: eps_eff },
        h0: None,
        h_min: 1e-12,
        h_max: dt_max,
        max_steps,
        max_rejects,
        force_eval: false,
    };

    // Step 1: Try explicit Tsit5
    let result_explicit = integrate_final(&system, OdeMethod::Tsit5, y0, t0, tf, cfg);

    // A scalar RHS failure must not be reclassified as an ODE status or trigger
    // a DualVec fallback.
    consume_solver_gravity_errors(rhs, dual_rhs)?;

    // Step 2: Check if explicit succeeded cleanly
    let explicit_ok = matches!(
        result_explicit.status,
        IntegrationStatus::Success | IntegrationStatus::EventTriggered
    );
    // For final-only, use absolute step count threshold since we have no t_eval length
    let expected_steps = ((tf - t0).abs() / dt_max)
        .ceil()
        .to_usize()
        .unwrap_or(usize::MAX);
    let step_count_ok =
        result_explicit.stats.steps < STIFFNESS_STEP_RATIO.saturating_mul(expected_steps.max(10));

    if explicit_ok && step_count_ok {
        return Ok(result_explicit);
    }

    // Step 3: Fallback to ESDIRK4(3)
    let jac = DualVecJacobian::new(dual_rhs);
    dual_rhs.reset_gravity_error();
    let esdirk_result = integrate_final_esdirk(&system, &jac, y0_arr, t0, tf, cfg);

    // ESDIRK evaluates both scalar RHS values and the DualVec Jacobian. Consume
    // their latches before its status can select either fallback result.
    consume_solver_gravity_errors(rhs, dual_rhs)?;

    let esdirk_ok = matches!(
        esdirk_result.status,
        IntegrationStatus::Success | IntegrationStatus::EventTriggered
    );

    if esdirk_ok {
        Ok(esdirk_result)
    } else if explicit_ok {
        Ok(result_explicit)
    } else {
        Ok(esdirk_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ForceConfig;
    use satpy_core::{pack_gravity_coeffs, GravityError, PackedGravityCoeffs};
    use std::sync::Arc;

    const TEST_STACK_SIZE: usize = 16 * 1024 * 1024;

    fn run_with_stack<F>(f: F) -> anyhow::Result<()>
    where
        F: FnOnce() -> anyhow::Result<()> + Send + 'static,
    {
        let handle = std::thread::Builder::new()
            .stack_size(TEST_STACK_SIZE)
            .spawn(f)
            .map_err(|error| {
                anyhow::anyhow!("failed to spawn adaptive-solver test thread: {error}")
            })?;
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("adaptive-solver test thread panicked"))?
    }

    fn packed_coefficients() -> Arc<PackedGravityCoeffs> {
        Arc::new(
            pack_gravity_coeffs(&[1.0, 0.0, 0.0, 0.0], &[0.0; 4], 2, 1)
                .expect("adaptive-solver test gravity coefficients must pack"),
        )
    }

    fn invalid_time_dual_rhs() -> anyhow::Result<LightyearDualRHS> {
        LightyearDualRHS::new(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            f64::NAN,
            Arc::new(ForceConfig {
                sph_order: 1,
                ..ForceConfig::default()
            }),
            packed_coefficients(),
        )
    }

    fn invalid_time_scalar_rhs() -> anyhow::Result<LightyearRHS> {
        LightyearRHS::try_new(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            f64::NAN,
            Arc::new(ForceConfig {
                sph_order: 1,
                ..ForceConfig::default()
            }),
            packed_coefficients(),
        )
    }

    #[test]
    fn jacobian_trait_latches_invalid_time_before_emitting_nan() {
        run_with_stack(|| {
            let dual_rhs = invalid_time_dual_rhs()?;
            let jacobian = DualVecJacobian::new(&dual_rhs);
            let mut matrix = [[0.0; 6]; 6];

            jacobian.jacobian(0.0, &[0.0; 6], &mut matrix);

            if !matrix.iter().flatten().all(|value| value.is_nan()) {
                return Err(anyhow::anyhow!(
                    "void Jacobian trait must emit NaN after its exact error"
                ));
            }
            if dual_rhs.take_gravity_error() != Some(GravityError::InvalidTime) {
                return Err(anyhow::anyhow!(
                    "Jacobian trait must retain InvalidTime for its public boundary"
                ));
            }
            Ok(())
        })
        .expect("invalid-time Jacobian trait path must retain its typed gravity error");
    }

    #[test]
    fn auto_final_returns_scalar_invalid_time_before_status_or_fallback() {
        run_with_stack(|| {
            let rhs = invalid_time_scalar_rhs()?;
            let dual_rhs = invalid_time_dual_rhs()?;

            let error =
                integrate_auto_final(&rhs, &dual_rhs, &[0.0; 6], 0.0, 1.0, 1e-8, 1.0, 32, 8)
                    .expect_err(
                        "scalar InvalidTime must outrank solver status and ESDIRK fallback",
                    );

            if error != GravityError::InvalidTime {
                return Err(anyhow::anyhow!("expected InvalidTime, got {error:?}"));
            }
            if dual_rhs.take_gravity_error().is_some() {
                return Err(anyhow::anyhow!(
                    "scalar gravity failure must prevent Dual fallback"
                ));
            }
            Ok(())
        })
        .expect("auto final must preserve scalar InvalidTime");
    }
}
