//! `Tau_max` computation for Gaussian splitting.
//!
//! Computes the maximum variance reduction τ such that P - τ*u*u^T remains PSD.

use crate::linalg::cholesky_solve6;
use common_rs::DIM;

/// Compute `tau_max`: the largest τ such that `P - τ·u·uᵀ` remains positive semi-definite.
///
/// For positive-definite P and nonzero u:
///   `τ_max = 1 / (uᵀ P⁻¹ u)`
///
/// Uses Cholesky decomposition to solve `P*x = u`, then `τ_max = 1 / (uᵀ x)`.
///
/// # Arguments
/// * `p` - 6x6 positive semi-definite covariance matrix (row-major)
/// * `u` - 6-element unit direction vector
///
/// # Returns
/// * `tau_max` - Maximum allowable variance reduction, or 0.0 if `P` is singular/`u` is degenerate
///
/// # Algorithm
/// 1. Check that u has finite positive norm
/// 2. Solve P*x = u via Cholesky (or fallback to eigendecomposition)
/// 3. Compute `τ_max = 1 / (uᵀ x)`
/// 4. Return 0.0 if result is non-positive or non-finite
#[inline]
#[must_use]
pub fn tau_max6(p: &[[f64; DIM]; DIM], u: &[f64; DIM]) -> f64 {
    const EPS: f64 = 1e-12;

    // Check axis validity
    let norm_sq = u[0].mul_add(
        u[0],
        u[1].mul_add(
            u[1],
            u[2].mul_add(u[2], u[3].mul_add(u[3], u[4].mul_add(u[4], u[5] * u[5]))),
        ),
    );
    if norm_sq <= 0.0 || !norm_sq.is_finite() {
        return 0.0;
    }

    // Solve P * x = u
    let Some(x) = cholesky_solve6(p, u) else {
        return 0.0;
    };

    // Compute u^T * x = u^T * P^{-1} * u
    let denom = u[0].mul_add(
        x[0],
        u[1].mul_add(
            x[1],
            u[2].mul_add(x[2], u[3].mul_add(x[3], u[4].mul_add(x[4], u[5] * x[5]))),
        ),
    );

    if denom <= EPS || !denom.is_finite() {
        return 0.0;
    }

    1.0 / denom
}

/// Compute `alpha_max` from `tau_max` and variance along axis.
///
/// `α_max = √(τ_max / var_ax)`
///
/// This is the maximum separation factor that keeps the downdated covariance PSD.
///
/// # Arguments
/// * `tau_max` - Maximum tau from `tau_max6`
/// * `var_ax` - Variance along split axis (`u^T P u`)
///
/// # Returns
/// * `alpha_max` - Maximum separation factor
#[inline]
#[must_use]
#[expect(
    clippy::redundant_pub_crate,
    reason = "sibling split module consumes this internal kernel"
)]
pub(super) fn alpha_max_from_tau(tau_max: f64, var_ax: f64) -> f64 {
    if tau_max <= 0.0 || var_ax <= 0.0 {
        return 0.0;
    }
    (tau_max / var_ax).max(1e-12).sqrt()
}

/// Choose alpha based on H4 heuristic.
///
/// `H4`: Default separation is 0.6 x `alpha_max`.
///
/// If user provides `alpha < 0`, use default. Otherwise clamp to `[0, alpha_max]`.
///
/// # Arguments
/// * `alpha_input` - User-provided alpha (negative means use default)
/// * `alpha_max` - Maximum allowed alpha
/// * `default_fraction` - Fraction of `alpha_max` to use as default (typically 0.6)
///
/// # Returns
/// * Chosen alpha value clamped to valid range
#[inline]
#[must_use]
#[expect(
    clippy::redundant_pub_crate,
    reason = "sibling split module consumes this internal kernel"
)]
pub(super) fn choose_alpha(alpha_input: f64, alpha_max: f64, default_fraction: f64) -> f64 {
    let mut alpha = if alpha_input < 0.0 {
        // H4: Use default fraction of alpha_max
        default_fraction * alpha_max
    } else {
        alpha_input
    };

    // Clamp to valid range
    if !alpha.is_finite() {
        alpha = 0.0;
    }
    if alpha < 0.0 {
        alpha = 0.0;
    }
    if alpha > alpha_max {
        alpha = alpha_max;
    }

    alpha
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::quadratic_form6;
    use approx::assert_relative_eq;

    fn identity_cov() -> [[f64; DIM]; DIM] {
        let mut cov = [[0.0; DIM]; DIM];
        for (index, row) in cov.iter_mut().enumerate() {
            if let Some(diagonal) = row.get_mut(index) {
                *diagonal = 1.0;
            }
        }
        cov
    }

    #[test]
    fn test_tau_max_identity() {
        let p = identity_cov();
        let u = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let tau = tau_max6(&p, &u);

        // For identity matrix and unit vector, tau_max = 1 / (u^T I^{-1} u) = 1
        assert_relative_eq!(tau, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tau_max_scaled_identity() {
        // P = 4*I
        let mut p = [[0.0; DIM]; DIM];
        for (index, row) in p.iter_mut().enumerate() {
            if let Some(diagonal) = row.get_mut(index) {
                *diagonal = 4.0;
            }
        }
        let u = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let tau = tau_max6(&p, &u);

        // tau_max = 1 / (u^T (4I)^{-1} u) = 1 / (1/4) = 4
        assert_relative_eq!(tau, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tau_max_zero_axis() {
        let p = identity_cov();
        let u = [0.0; DIM];

        let tau = tau_max6(&p, &u);
        assert_eq!(tau.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn test_alpha_max_from_tau() {
        let alpha_max = alpha_max_from_tau(4.0, 1.0);
        assert_relative_eq!(alpha_max, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_choose_alpha_default() {
        let alpha = choose_alpha(-1.0, 1.0, 0.6);
        assert_relative_eq!(alpha, 0.6, epsilon = 1e-10);
    }

    #[test]
    fn test_choose_alpha_clamped() {
        let alpha = choose_alpha(2.0, 1.0, 0.6);
        assert_relative_eq!(alpha, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_choose_alpha_user_provided() {
        let alpha = choose_alpha(0.3, 1.0, 0.6);
        assert_relative_eq!(alpha, 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_tau_max_matches_variance() {
        // Test that tau_max * (u^T P u) gives reasonable separation
        let p = identity_cov();
        let u = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let var_ax = quadratic_form6(&p, &u);
        let tau = tau_max6(&p, &u);
        let alpha_max = alpha_max_from_tau(tau, var_ax);

        // For identity, var_ax = 1, tau_max = 1, so alpha_max = 1
        assert_relative_eq!(var_ax, 1.0, epsilon = 1e-10);
        assert_relative_eq!(tau, 1.0, epsilon = 1e-10);
        assert_relative_eq!(alpha_max, 1.0, epsilon = 1e-10);
    }
}
