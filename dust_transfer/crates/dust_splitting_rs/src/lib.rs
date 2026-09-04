//! Gaussian Mixture Splitting for Dust Calculations
//!
//! This crate provides Rust implementations of Gaussian splitting algorithms
//! for uncertainty propagation in orbital mechanics dust calculations.
//!
//! # Key Functions
//!
//! - [`split_gaussian_along_axis`]: Split a Gaussian along a specified axis
//! - [`split_gaussian_no_axis`]: Auto-select dominant eigenvector for splitting
//!
//! # Heuristics
//!
//! The following engineering heuristics are documented throughout:
//!
//! - **H4**: 0.6 x `alpha_max` default separation factor

pub mod linalg;
mod quadrature;
mod types;

mod downdate;
mod split;
mod tau;

// Re-export main types and functions
#[cfg(test)]
use common_rs::DIM;
pub use linalg::{cholesky_solve6, principal_covariance_axis6};
pub use split::{split_gaussian_along_axis, split_gaussian_no_axis};
pub use types::{SplitConfig, SplitResult, MAX_COMPONENTS};

/// Flatten a [`SplitResult`] into caller-provided row-major buffers.
///
/// Test-only: the shipping callers write their own packed layouts directly;
/// this exists so the tests can pin that layout independently.
#[cfg(test)]
fn flatten_result_into(
    result: &SplitResult,
    means: &mut [f64],
    covs: &mut [f64],
    weights: &mut [f64],
) {
    let k = result.num_components();
    debug_assert_eq!(means.len(), k.saturating_mul(DIM));
    debug_assert_eq!(covs.len(), k.saturating_mul(DIM).saturating_mul(DIM));
    debug_assert_eq!(weights.len(), k);
    for (
        ((component_mean, component_covariance), component_weight),
        ((mean_out, covariance_out), weight_out),
    ) in result
        .means
        .iter()
        .zip(&result.covariances)
        .zip(&result.weights)
        .zip(
            means
                .chunks_exact_mut(DIM)
                .zip(covs.chunks_exact_mut(DIM.saturating_mul(DIM)))
                .zip(weights),
        )
    {
        mean_out.copy_from_slice(component_mean);
        for (row_out, row) in covariance_out
            .chunks_exact_mut(DIM)
            .zip(component_covariance)
        {
            row_out.copy_from_slice(row);
        }
        *weight_out = *component_weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the judgment-call default (`types.rs`, marked H4). It is a bare
    /// literal in `SplitConfig::default`, so a copy here notices it moving.
    #[test]
    fn split_config_default_pins_the_h4_judgment_call() {
        let config = SplitConfig::default();
        assert!((config.default_alpha_fraction - 0.6).abs() < 1e-10);
    }

    #[test]
    fn flatten_result_into_matches_component_layout() {
        let result = SplitResult::new(2);
        let mut means = vec![0.0; result.num_components() * DIM];
        let mut covs = vec![0.0; result.num_components() * DIM * DIM];
        let mut weights = vec![0.0; result.num_components()];

        flatten_result_into(&result, &mut means, &mut covs, &mut weights);

        assert_eq!(
            means.get(..DIM),
            result.means.first().map(<[f64; DIM]>::as_slice)
        );
        assert_eq!(weights.as_slice(), result.weights.as_slice());
    }
}
