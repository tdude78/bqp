//! Core Gaussian splitting algorithms.
//!
//! Implements the main splitting functions for decomposing a Gaussian distribution
//! into a weighted mixture of Gaussians using Gauss-Hermite quadrature.

use crate::downdate::downdate_single_axis;
use crate::linalg::{dominant_eigenvector6, normalize_inplace6, quadratic_form6};
use crate::quadrature::hermgauss_std_normal;
use crate::tau::{alpha_max_from_tau, choose_alpha, tau_max6};
use crate::types::{SplitConfig, SplitResult, MAX_COMPONENTS};
use common_rs::{symmetrize_array, DIM};

/// Split a Gaussian distribution along a specified axis.
///
/// Given a Gaussian `N(m, P)` and unit direction `u`, produces a mixture of `K` Gaussians
/// with means offset along `u` and shared (downdated) covariance.
///
/// # Arguments
/// * `mean` - Original mean vector (6 elements)
/// * `cov` - Original covariance matrix (6x6, row-major)
/// * `axis` - Split direction (will be normalized)
/// * `num_components` - Number of mixture components (1-7)
/// * `alpha` - Separation scale factor. If negative, uses `H4` default (`0.6 * alpha_max`)
/// * `config` - Configuration parameters
///
/// # Returns
/// * [`SplitResult`] with `K` means, `K` covariances (all identical), and `K` weights
///
/// # Algorithm
/// 1. Normalize axis to unit vector `u`
/// 2. Compute variance along axis: `σ² = uᵀ P u`
/// 3. Compute `τ_max` and `α_max`
/// 4. Choose α (clamped to `[0, α_max]`)
/// 5. Get Gauss-Hermite nodes `z_k` and weights `w_k`
/// 6. Compute offsets: `δ_k = α x σ x z_k`
/// 7. Compute `τ = Σ w_k x δ_k²`
/// 8. Downdate covariance: `C = P - τ x uuᵀ`
/// 9. Compute means: `m_k = m + δ_k x u`
/// 10. Normalize weights
#[must_use]
pub fn split_gaussian_along_axis(
    mean: &[f64; DIM],
    cov: &[[f64; DIM]; DIM],
    axis: &[f64; DIM],
    num_components: usize,
    alpha: f64,
    config: &SplitConfig,
) -> SplitResult {
    // Clamp num_components
    let k = num_components.clamp(1, MAX_COMPONENTS);

    // Normalize axis
    let mut u = *axis;
    if !normalize_inplace6(&mut u) {
        // Zero axis - return unsplit
        return SplitResult::unsplit(mean, cov);
    }

    // Trivial case
    if k == 1 {
        return SplitResult::unsplit(mean, cov);
    }

    // Symmetrize covariance
    let mut p = *cov;
    symmetrize_array(&mut p);

    // Compute variance along axis
    let var_ax = quadratic_form6(&p, &u);
    if var_ax <= 0.0 || !var_ax.is_finite() {
        return SplitResult::unsplit(mean, &p);
    }

    // Compute tau_max
    let tau_limit = tau_max6(&p, &u);
    if tau_limit <= 0.0 || !tau_limit.is_finite() {
        return SplitResult::unsplit(mean, &p);
    }

    // Compute alpha_max and choose alpha
    let alpha_max = alpha_max_from_tau(tau_limit, var_ax);
    let alpha_chosen = choose_alpha(alpha, alpha_max, config.default_alpha_fraction);

    // Get Gauss-Hermite quadrature
    let Some(hg) = hermgauss_std_normal(k) else {
        return SplitResult::unsplit(mean, &p);
    };
    let sqrt_var_ax = var_ax.sqrt();

    // Compute offsets
    let mut offsets = [0.0; MAX_COMPONENTS];
    for (offset, &node) in offsets.iter_mut().zip(hg.nodes).take(k) {
        *offset = alpha_chosen * sqrt_var_ax * node;
    }

    // Compute tau for downdating
    let mut tau = 0.0;
    for (&off, &weight) in offsets.iter().zip(hg.weights).take(k) {
        tau += weight * off * off;
    }

    // Downdate covariance: C = P - tau * u * u^T
    let c = downdate_single_axis(&p, tau, &u);

    // Construct result
    let mut result = SplitResult::new(k);

    // Compute means: m_k = m + offset_k * u
    for (component_mean, &offset) in result.means.iter_mut().zip(&offsets) {
        for ((output, &center), &axis_component) in component_mean.iter_mut().zip(mean).zip(&u) {
            *output = center + offset * axis_component;
        }
    }

    // All covariances are identical
    result.covariances.fill(c);

    // Normalize weights
    let w_sum: f64 = hg.weights.get(..k).unwrap_or(&[]).iter().sum();
    let inv_w_sum = if w_sum > 0.0 && w_sum.is_finite() {
        1.0 / w_sum
    } else {
        u32::try_from(k).map_or(1.0, |count| 1.0 / f64::from(count))
    };
    for (output, &weight) in result.weights.iter_mut().zip(hg.weights) {
        *output = weight * inv_w_sum;
    }

    result
}

/// Split a Gaussian using its dominant eigenvector.
///
/// Automatically selects the direction of maximum variance for splitting.
///
/// # Arguments
/// * `mean` - Original mean vector
/// * `cov` - Original covariance matrix
/// * `num_components` - Number of mixture components
/// * `alpha` - Separation factor (None for default)
/// * `config` - Configuration parameters
///
/// # Returns
/// * `SplitResult` with K components
#[must_use]
pub fn split_gaussian_no_axis(
    mean: &[f64; DIM],
    cov: &[[f64; DIM]; DIM],
    num_components: usize,
    alpha: Option<f64>,
    config: &SplitConfig,
) -> SplitResult {
    // Symmetrize covariance
    let mut p = *cov;
    symmetrize_array(&mut p);

    let k = num_components.clamp(1, MAX_COMPONENTS);

    if k == 1 {
        return SplitResult::unsplit(mean, &p);
    }

    // Find dominant eigenvector
    let Some(u) = dominant_eigenvector6(&p) else {
        return SplitResult::unsplit(mean, &p);
    };

    // Use single-axis split along dominant direction
    split_gaussian_along_axis(mean, &p, &u, k, alpha.unwrap_or(-1.0), config)
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::suboptimal_flops,
        reason = "tests preserve reference floating-point accumulation order"
    )]

    use super::*;
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

    fn mean_at(result: &SplitResult, component: usize, coordinate: usize) -> f64 {
        result
            .means
            .get(component)
            .and_then(|mean| mean.get(coordinate))
            .copied()
            .unwrap_or(f64::NAN)
    }

    fn covariance_at(covariance: &[[f64; DIM]; DIM], row: usize, column: usize) -> f64 {
        covariance
            .get(row)
            .and_then(|values| values.get(column))
            .copied()
            .unwrap_or(f64::NAN)
    }

    #[test]
    fn test_split_along_axis_k1() {
        let mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cov = identity_cov();
        let axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = SplitConfig::default();

        let result = split_gaussian_along_axis(&mean, &cov, &axis, 1, -1.0, &config);

        assert_eq!(result.num_components(), 1);
        assert_eq!(
            result.means.first().map(|value| value.map(f64::to_bits)),
            Some(mean.map(f64::to_bits))
        );
        assert_relative_eq!(
            result.weights.first().copied().unwrap_or(f64::NAN),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_split_along_axis_k3() {
        let mean = [0.0; DIM];
        let cov = identity_cov();
        let axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = SplitConfig::default();

        let result = split_gaussian_along_axis(&mean, &cov, &axis, 3, -1.0, &config);

        assert_eq!(result.num_components(), 3);

        // Weights should sum to 1
        let w_sum: f64 = result.weights.iter().sum();
        assert_relative_eq!(w_sum, 1.0, epsilon = 1e-10);

        // Middle component should be at origin (for symmetric quadrature)
        assert_relative_eq!(mean_at(&result, 1, 0), 0.0, epsilon = 1e-10);

        // First and last should be symmetric
        assert_relative_eq!(
            mean_at(&result, 0, 0),
            -mean_at(&result, 2, 0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_split_along_axis_preserves_mean() {
        let mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cov = identity_cov();
        let axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = SplitConfig::default();

        let result = split_gaussian_along_axis(&mean, &cov, &axis, 5, -1.0, &config);

        // Weighted mean should equal original mean
        let mut mix_mean = [0.0; DIM];
        for (m, &weight) in result.means.iter().zip(&result.weights) {
            for (mixture, &component) in mix_mean.iter_mut().zip(m) {
                *mixture += weight * component;
            }
        }

        for (&mixture, &expected) in mix_mean.iter().zip(&mean) {
            assert_relative_eq!(mixture, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_three_component_split_preserves_full_covariance() {
        let mean = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let mut cov = identity_cov();
        if let Some(diagonal) = cov.first_mut().and_then(|row| row.first_mut()) {
            *diagonal = 4.0;
        }
        if let Some(diagonal) = cov.get_mut(1).and_then(|row| row.get_mut(1)) {
            *diagonal = 2.0;
        }
        let axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = SplitConfig::default();
        let result = split_gaussian_along_axis(&mean, &cov, &axis, 3, -1.0, &config);

        let mut recovered_mean = [0.0; DIM];
        for (component, weight) in result.means.iter().zip(result.weights.iter()) {
            for (recovered, &value) in recovered_mean.iter_mut().zip(component) {
                *recovered += weight * value;
            }
        }

        let mut recovered_cov = [[0.0; DIM]; DIM];
        for ((component_mean, component_covariance), &weight) in result
            .means
            .iter()
            .zip(&result.covariances)
            .zip(&result.weights)
        {
            for (row_index, recovered_row) in recovered_cov.iter_mut().enumerate() {
                let delta_i = component_mean.get(row_index).copied().unwrap_or(f64::NAN)
                    - recovered_mean.get(row_index).copied().unwrap_or(f64::NAN);
                for (column_index, recovered_value) in recovered_row.iter_mut().enumerate() {
                    let delta_j = component_mean
                        .get(column_index)
                        .copied()
                        .unwrap_or(f64::NAN)
                        - recovered_mean
                            .get(column_index)
                            .copied()
                            .unwrap_or(f64::NAN);
                    *recovered_value += weight
                        * (covariance_at(component_covariance, row_index, column_index)
                            + delta_i * delta_j);
                }
            }
        }

        for (row_index, (&actual_mean, &expected_mean)) in
            recovered_mean.iter().zip(&mean).enumerate()
        {
            assert_relative_eq!(actual_mean, expected_mean, epsilon = 1.0e-10);
            for column_index in 0..DIM {
                assert_relative_eq!(
                    covariance_at(&recovered_cov, row_index, column_index),
                    covariance_at(&cov, row_index, column_index),
                    epsilon = 1.0e-10
                );
            }
        }
    }

    #[test]
    fn test_split_no_axis_uses_dominant_eigenvector() {
        // Create covariance with dominant direction
        let mut cov = [[0.0; DIM]; DIM];
        if let Some(diagonal) = cov.first_mut().and_then(|row| row.first_mut()) {
            *diagonal = 10.0;
        }
        for (index, row) in cov.iter_mut().enumerate().skip(1) {
            if let Some(diagonal) = row.get_mut(index) {
                *diagonal = 1.0;
            }
        }

        let mean = [0.0; DIM];
        let config = SplitConfig::default();

        let result = split_gaussian_no_axis(&mean, &cov, 3, None, &config);

        assert_eq!(result.num_components(), 3);

        // Components should be spread along first dimension
        let first = mean_at(&result, 0, 0);
        let last = mean_at(&result, 2, 0);
        assert!(first.abs() > 0.1);
        assert!(last.abs() > 0.1);
        assert!(first * last < 0.0); // Opposite signs
    }

    #[test]
    fn test_zero_axis_returns_unsplit() {
        let mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cov = identity_cov();
        let axis = [0.0; DIM];
        let config = SplitConfig::default();

        let result = split_gaussian_along_axis(&mean, &cov, &axis, 5, -1.0, &config);

        assert_eq!(result.num_components(), 1);
        assert_eq!(
            result.means.first().map(|value| value.map(f64::to_bits)),
            Some(mean.map(f64::to_bits))
        );
    }
}
