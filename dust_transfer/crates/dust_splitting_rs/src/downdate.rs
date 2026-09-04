//! Single-axis covariance downdating.
//!
//! When splitting a Gaussian, the new (shared) covariance is obtained by
//! "downdating" the original: C = P - τ·u·uᵀ.

use crate::linalg::{matrix_sub6, outer_product_scaled6};
use common_rs::DIM;

/// Single-axis covariance downdate (no PSD repair needed for single axis).
///
/// Computes C = P - τ·u·uᵀ directly.
///
/// # Arguments
/// * `p` - Original covariance matrix
/// * `tau` - Variance reduction factor
/// * `u` - Unit axis direction
///
/// # Returns
/// * Downdated covariance matrix
#[inline]
#[must_use]
#[expect(
    clippy::redundant_pub_crate,
    reason = "sibling split module consumes this internal kernel"
)]
pub(super) fn downdate_single_axis(
    p: &[[f64; DIM]; DIM],
    tau: f64,
    u: &[f64; DIM],
) -> [[f64; DIM]; DIM] {
    let outer = outer_product_scaled6(u, tau);
    matrix_sub6(p, &outer)
}

#[cfg(test)]
mod tests {
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

    fn matrix_at(matrix: &[[f64; DIM]; DIM], row: usize, column: usize) -> f64 {
        matrix
            .get(row)
            .and_then(|values| values.get(column))
            .copied()
            .unwrap_or(f64::NAN)
    }

    #[test]
    fn test_downdate_single_axis() {
        let p = identity_cov();
        let u = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let tau = 0.5;

        let c = downdate_single_axis(&p, tau, &u);

        // C[0][0] should be 1 - 0.5 = 0.5
        assert_relative_eq!(matrix_at(&c, 0, 0), 0.5, epsilon = 1e-10);
        // Other diagonals unchanged
        for (index, row) in c.iter().enumerate().skip(1) {
            assert_relative_eq!(
                row.get(index).copied().unwrap_or(f64::NAN),
                1.0,
                epsilon = 1e-10
            );
        }
    }
}
