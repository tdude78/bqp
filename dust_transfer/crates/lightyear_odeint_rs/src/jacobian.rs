//! Jacobian extraction from `DualVec` automatic differentiation.
//!
//! Computes the 6x6 Jacobian `df/dy` by seeding 3 partial derivatives at a time
//! through the `DualVec` RHS, requiring exactly 2 RHS evaluations for the full matrix.

use crate::rhs_dual::LightyearDualRHS;
use nalgebra::Vector3;
use satpy_core::{DualVec, GravityError};

/// Compute the 6x6 Jacobian `df/dy` at state `delta` and time `t`.
///
/// Uses `LightyearDualRHS` (forward-mode AD) in 2 passes:
///   Pass 1: seed positions (`dx`, `dy`, `dz`) -> columns 0, 1, 2
///   Pass 2: seed velocities (`dvx`, `dvy`, `dvz`) -> columns 3, 4, 5
///
/// # Arguments
/// * `dual_rhs` - The `DualVec` force model
/// * `delta`    - Current delta-state (6D)
/// * `t`        - Current time (seconds from epoch)
/// * `jac`      - Output: `jac[row][col] = df_row / dy_col`
///
/// # Errors
///
/// Returns the exact packed-gravity evaluation failure. The output matrix is
/// unchanged when either AD pass fails.
/// Production takes [`compute_jacobian_unlatched`] instead: the trait adapters
/// need the latch left intact so their own outer boundary can report the exact
/// gravity error. This latched wrapper is the boundary the four Jacobian tests
/// below exercise, and it has no other caller.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "latched boundary; production uses the unlatched form, tests use this one"
    )
)]
pub fn compute_jacobian(
    dual_rhs: &LightyearDualRHS,
    delta: &[f64; 6],
    t: f64,
    jac: &mut [[f64; 6]; 6],
) -> Result<(), GravityError> {
    dual_rhs.reset_gravity_error();
    let result = compute_jacobian_unlatched(dual_rhs, delta, t, jac);
    dual_rhs.take_gravity_error().map_or(result, Err)
}

/// Compute a Jacobian without resetting or consuming the RHS error latch.
///
/// Trait adapters use this so their outer public boundary can report the exact
/// gravity error after forcing the non-fallible solver to stop.
///
/// # Errors
///
/// Returns the exact packed-gravity evaluator failure while leaving the owned
/// RHS latch intact for the outer boundary.
pub fn compute_jacobian_unlatched(
    dual_rhs: &LightyearDualRHS,
    delta: &[f64; 6],
    t: f64,
    jac: &mut [[f64; 6]; 6],
) -> Result<(), GravityError> {
    // Pass 1: Seed position components (columns 0, 1, 2)
    // DualVec carries 3 partial derivatives simultaneously via f64x4 SIMD.
    // Seed delta[0] with d/dy0=1, delta[1] with d/dy1=1, delta[2] with d/dy2=1.
    // Velocity components get zero derivatives (DualVec::constant).
    let [delta_x, delta_y, delta_z, velocity_x, velocity_y, velocity_z] = *delta;
    let delta_dual = [
        DualVec::new(delta_x, Vector3::new(1.0, 0.0, 0.0)),
        DualVec::new(delta_y, Vector3::new(0.0, 1.0, 0.0)),
        DualVec::new(delta_z, Vector3::new(0.0, 0.0, 1.0)),
        DualVec::constant(velocity_x),
        DualVec::constant(velocity_y),
        DualVec::constant(velocity_z),
    ];

    let f1 = dual_rhs.compute_internal(&delta_dual, t)?;

    // Pass 2: Seed velocity components (columns 3, 4, 5)
    // Position components get zero derivatives, velocity components get unit seeds.
    let delta_dual = [
        DualVec::constant(delta_x),
        DualVec::constant(delta_y),
        DualVec::constant(delta_z),
        DualVec::new(velocity_x, Vector3::new(1.0, 0.0, 0.0)),
        DualVec::new(velocity_y, Vector3::new(0.0, 1.0, 0.0)),
        DualVec::new(velocity_z, Vector3::new(0.0, 0.0, 1.0)),
    ];

    let f2 = dual_rhs.compute_internal(&delta_dual, t)?;

    // Extract columns 0,1,2 from the derivative lanes only after both AD
    // passes succeed, so a typed gravity error cannot leave a partial matrix.
    for (jacobian_row, output) in jac.iter_mut().zip(f1) {
        let [jacobian_x, jacobian_y, jacobian_z, ..] = jacobian_row;
        let [derivative_x, derivative_y, derivative_z] = output.d();
        *jacobian_x = derivative_x;
        *jacobian_y = derivative_y;
        *jacobian_z = derivative_z;
    }

    // Extract columns 3,4,5 from the derivative lanes
    for (jacobian_row, output) in jac.iter_mut().zip(f2) {
        let [_, _, _, jacobian_velocity_x, jacobian_velocity_y, jacobian_velocity_z] = jacobian_row;
        let [derivative_velocity_x, derivative_velocity_y, derivative_velocity_z] = output.d();
        *jacobian_velocity_x = derivative_velocity_x;
        *jacobian_velocity_y = derivative_velocity_y;
        *jacobian_velocity_z = derivative_velocity_z;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ForceConfig;
    use satpy_core::PackedGravityCoeffs;
    use std::sync::Arc;

    /// Stack size for test threads: 16 MB.
    ///
    /// `LightyearDualRHS` embeds `GravityCacheGeneric<DualVec>` which is ~1.1 MB
    /// (2 x 131 x 131 x 32 bytes). In debug builds without optimizations the
    /// compiler does not elide stack temporaries, and the deep gravity recursion
    /// call chain with `DualVec` (4x larger than `f64`) compounds the usage. 16 MB
    /// provides robust headroom for all force configurations.
    const TEST_STACK_SIZE: usize = 16 * 1024 * 1024;

    /// Run a closure on a thread with a larger stack to avoid overflow from
    /// the ~1.1 MB `GravityCacheGeneric<DualVec>` in `LightyearDualRHS`.
    fn run_with_stack<F>(f: F) -> anyhow::Result<()>
    where
        F: FnOnce() -> anyhow::Result<()> + Send + 'static,
    {
        let handle = std::thread::Builder::new()
            .stack_size(TEST_STACK_SIZE)
            .spawn(f)
            .map_err(|error| anyhow::anyhow!("failed to spawn test thread: {error}"))?;
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("test thread panicked"))?
    }

    /// Create minimal mock spherical harmonic coefficients (mirrors `lib.rs` tests).
    fn mock_coefficients(order: usize) -> anyhow::Result<Arc<PackedGravityCoeffs>> {
        let stride = order
            .checked_add(2)
            .ok_or_else(|| anyhow::anyhow!("coefficient order overflow"))?;
        let total_size = stride
            .checked_mul(stride)
            .ok_or_else(|| anyhow::anyhow!("coefficient table size overflow"))?;
        let mut c_coeffs = vec![0.0; total_size];
        let s_coeffs = vec![0.0; total_size];

        // C[0,0] = 1.0 is the point-mass gravity term
        let Some(c00) = c_coeffs.first_mut() else {
            return Err(anyhow::anyhow!("coefficient table unexpectedly empty"));
        };
        *c00 = 1.0;

        // J2 term (if order >= 2)
        if order >= 2 {
            let j2_index = 2usize
                .checked_mul(stride)
                .ok_or_else(|| anyhow::anyhow!("J2 coefficient index overflow"))?;
            let Some(c20) = c_coeffs.get_mut(j2_index) else {
                return Err(anyhow::anyhow!("J2 coefficient index outside table"));
            };
            *c20 = -1.08263e-3;
        }

        let packed = satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order).map_err(
            |error| anyhow::anyhow!("Jacobian test gravity coefficients must pack: {error}"),
        )?;

        Ok(Arc::new(packed))
    }

    fn build_force_config(force_flags: i32) -> Arc<ForceConfig> {
        Arc::new(ForceConfig {
            sph_order: 0,
            force_flags,
            eps: 1e-10,
            integrator_method: crate::types::StepperMethod::Dopri5Compat,
            ..ForceConfig::default()
        })
    }

    /// Compute the 6x6 Jacobian via central finite differences on `LightyearDualRHS`.
    ///
    /// Uses the `DualVec` RHS value lane (`.v()`) for finite differences so that
    /// both the AD and FD Jacobians are computed from the same function, avoiding
    /// discrepancies from the optimized f64 RHS code path (SIMD third-body,
    /// packed-sincos gravity specializations).
    fn jacobian_finite_diff_dual(
        rhs: &LightyearDualRHS,
        delta: &[f64; 6],
        t: f64,
        eps: f64,
    ) -> anyhow::Result<[[f64; 6]; 6]> {
        // Helper: evaluate dual RHS with zero-derivative seeds and extract values
        let eval_values = |state: &[f64; 6]| -> Result<[f64; 6], GravityError> {
            let &[state_x, state_y, state_z, velocity_x, velocity_y, velocity_z] = state;
            let d_dual = [
                DualVec::constant(state_x),
                DualVec::constant(state_y),
                DualVec::constant(state_z),
                DualVec::constant(velocity_x),
                DualVec::constant(velocity_y),
                DualVec::constant(velocity_z),
            ];
            let [derivative_x, derivative_y, derivative_z, velocity_x, velocity_y, velocity_z] =
                rhs.compute_internal(&d_dual, t)?;
            Ok([
                derivative_x.v(),
                derivative_y.v(),
                derivative_z.v(),
                velocity_x.v(),
                velocity_y.v(),
                velocity_z.v(),
            ])
        };

        let mut finite_difference_jacobian = [[0.0f64; 6]; 6];
        for (column, _) in delta.iter().enumerate() {
            let mut delta_plus = *delta;
            let mut delta_minus = *delta;
            let plus_component = delta_plus.get_mut(column).ok_or_else(|| {
                anyhow::anyhow!("finite-difference plus column outside fixed state shape")
            })?;
            let minus_component = delta_minus.get_mut(column).ok_or_else(|| {
                anyhow::anyhow!("finite-difference minus column outside fixed state shape")
            })?;
            *plus_component += eps;
            *minus_component -= eps;
            let f_plus = eval_values(&delta_plus).map_err(|error| {
                anyhow::anyhow!("finite-difference plus evaluation failed: {error}")
            })?;
            let f_minus = eval_values(&delta_minus).map_err(|error| {
                anyhow::anyhow!("finite-difference minus evaluation failed: {error}")
            })?;
            let inv_2eps = 1.0 / (2.0 * eps);
            for ((jacobian_row, f_plus_value), f_minus_value) in finite_difference_jacobian
                .iter_mut()
                .zip(f_plus)
                .zip(f_minus)
            {
                let jacobian_value = jacobian_row.get_mut(column).ok_or_else(|| {
                    anyhow::anyhow!("finite-difference column outside fixed Jacobian shape")
                })?;
                *jacobian_value = (f_plus_value - f_minus_value) * inv_2eps;
            }
        }
        Ok(finite_difference_jacobian)
    }

    /// Assert two Jacobians match within combined relative + absolute tolerance.
    fn assert_jacobians_close(
        label: &str,
        automatic_jacobian: &[[f64; 6]; 6],
        finite_difference_jacobian: &[[f64; 6]; 6],
        rel_tol: f64,
        abs_tol: f64,
    ) {
        for (row, (automatic_row, finite_difference_row)) in automatic_jacobian
            .iter()
            .zip(finite_difference_jacobian)
            .enumerate()
        {
            for (col, (&automatic_value, &finite_difference_value)) in
                automatic_row.iter().zip(finite_difference_row).enumerate()
            {
                let abs_diff = (automatic_value - finite_difference_value).abs();
                let scale = finite_difference_value.abs().max(automatic_value.abs());
                let ok = abs_diff < abs_tol || abs_diff < rel_tol * scale;
                assert!(
                    ok,
                    "{label} mismatch at [{row}][{col}]: AD={automatic_value:.10e}, FD={finite_difference_value:.10e}, \
                     abs_diff={abs_diff:.4e}, rel_diff={:.4e}",
                    abs_diff / scale.max(1e-30)
                );
            }
        }
    }

    #[test]
    fn test_jacobian_vs_finite_diff_gravity_only() -> anyhow::Result<()> {
        run_with_stack(|| {
            let packed = mock_coefficients(0)?;
            let config = build_force_config(0);

            let init_equinoc_state = [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let jd0 = 2_460_000.0;
            let t0_s = 0.0;
            let t = 0.0;
            let delta = [0.01, 0.02, -0.01, 1e-5, -1e-5, 2e-5];

            let dual_rhs = LightyearDualRHS::new(init_equinoc_state, t0_s, jd0, config, packed)?;

            let mut automatic_jacobian = [[0.0f64; 6]; 6];
            compute_jacobian(&dual_rhs, &delta, t, &mut automatic_jacobian).map_err(|error| {
                anyhow::anyhow!("automatic gravity-only Jacobian failed: {error}")
            })?;

            let finite_difference_jacobian = jacobian_finite_diff_dual(&dual_rhs, &delta, t, 1e-7)?;

            assert_jacobians_close(
                "Gravity-only Jacobian",
                &automatic_jacobian,
                &finite_difference_jacobian,
                1e-5,
                1e-10,
            );
            Ok(())
        })
    }

    #[test]
    fn test_jacobian_vs_finite_diff_j2_gravity() -> anyhow::Result<()> {
        run_with_stack(|| {
            let packed = mock_coefficients(2)?;
            let config = Arc::new(ForceConfig {
                sph_order: 2,
                eps: 1e-10,
                integrator_method: crate::types::StepperMethod::Dopri5Compat,
                ..ForceConfig::default()
            });

            let init_equinoc_state = [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let jd0 = 2_460_000.0;
            let t0_s = 0.0;
            let t = 0.0;
            let delta = [0.05, -0.03, 0.04, 2e-5, -1e-5, 3e-5];

            let dual_rhs = LightyearDualRHS::new(init_equinoc_state, t0_s, jd0, config, packed)?;

            let mut automatic_jacobian = [[0.0f64; 6]; 6];
            compute_jacobian(&dual_rhs, &delta, t, &mut automatic_jacobian)
                .map_err(|error| anyhow::anyhow!("automatic J2 Jacobian failed: {error}"))?;

            let finite_difference_jacobian = jacobian_finite_diff_dual(&dual_rhs, &delta, t, 1e-7)?;

            assert_jacobians_close(
                "J2 Jacobian",
                &automatic_jacobian,
                &finite_difference_jacobian,
                1e-5,
                1e-10,
            );
            Ok(())
        })
    }

    #[test]
    fn test_jacobian_propagates_invalid_time_without_partial_output() -> anyhow::Result<()> {
        run_with_stack(|| {
            let packed = mock_coefficients(2)?;
            let mut config = *build_force_config(0);
            config.sph_order = 2;
            let dual_rhs = LightyearDualRHS::new(
                [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                0.0,
                f64::NAN,
                Arc::new(config),
                packed,
            )?;
            let mut jacobian = [[7.0; 6]; 6];

            let error = compute_jacobian(
                &dual_rhs,
                &[0.01, 0.02, -0.01, 1e-5, -1e-5, 2e-5],
                0.0,
                &mut jacobian,
            )
            .err()
            .ok_or_else(|| {
                anyhow::anyhow!("invalid gravity time must propagate through the Jacobian boundary")
            })?;
            if error != GravityError::InvalidTime {
                return Err(anyhow::anyhow!("expected InvalidTime, got {error}"));
            }
            if jacobian != [[7.0; 6]; 6] {
                return Err(anyhow::anyhow!(
                    "a typed Jacobian failure must not leave a partial output"
                ));
            }
            if dual_rhs.take_gravity_error().is_some() {
                return Err(anyhow::anyhow!(
                    "the public Jacobian boundary consumes its error latch"
                ));
            }
            Ok(())
        })
    }

    #[test]
    fn test_jacobian_identity_block_structure() {
        let result = run_with_stack(|| {
            // The delta-state ODE has the form:
            //   f[0..3] = delta[3..6]   (velocity passthrough)
            //   f[3..6] = accelerations(delta)
            //
            // So the Jacobian should have:
            //   Rows 0-2: identity block in columns 3-5 (df_pos/dv = I)
            //   Rows 0-2: zero block in columns 0-2 (df_pos/dpos = 0)
            let packed = mock_coefficients(0)?;
            let config = build_force_config(0);

            let init_equinoc_state = [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let jd0 = 2_460_000.0;
            let t0_s = 0.0;
            let t = 0.0;
            let delta = [0.01, 0.02, -0.01, 1e-5, -1e-5, 2e-5];

            let dual_rhs = LightyearDualRHS::new(init_equinoc_state, t0_s, jd0, config, packed)?;

            let mut jac = [[0.0f64; 6]; 6];
            compute_jacobian(&dual_rhs, &delta, t, &mut jac).map_err(|error| {
                anyhow::anyhow!("automatic identity-block Jacobian failed: {error}")
            })?;

            // Check upper-left 3x3 is zero (df_pos/dpos = 0)
            for (row, jacobian_row) in jac.iter().take(3).enumerate() {
                for (col, &value) in jacobian_row.iter().take(3).enumerate() {
                    assert!(
                        value.abs() < 1e-14,
                        "Expected zero at [{row}][{col}], got {value:.4e}"
                    );
                }
            }

            // Check upper-right 3x3 is identity (df_pos/dvel = I)
            for (row, jacobian_row) in jac.iter().take(3).enumerate() {
                for (col, &value) in jacobian_row.iter().enumerate().skip(3) {
                    let expected = if col.checked_sub(3) == Some(row) {
                        1.0
                    } else {
                        0.0
                    };
                    assert!(
                        (value - expected).abs() < 1e-14,
                        "Expected {expected} at [{row}][{col}], got {value:.4e}"
                    );
                }
            }
            Ok(())
        });
        assert!(result.is_ok(), "identity-block check failed: {result:?}");
    }
}
