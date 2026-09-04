//! Core types for Gaussian splitting.
//!
//! This module defines the data structures used throughout the splitting algorithms.

use smallvec::SmallVec;

use common_rs::DIM;

/// Maximum number of Gauss-Hermite components supported.
pub const MAX_COMPONENTS: usize = 7;

/// Result of Gaussian splitting operation.
///
/// Uses `SmallVec` with capacity 8 to avoid heap allocation for typical cases
/// (`MAX_COMPONENTS=7`), while still supporting larger splits if needed.
#[derive(Clone, Debug)]
pub struct SplitResult {
    /// Component means, shape (K, 6). Stack-allocated for K ≤ 8.
    pub means: SmallVec<[[f64; DIM]; 8]>,
    /// Component covariances, shape (K, 6, 6). Stack-allocated for K ≤ 8.
    pub covariances: SmallVec<[[[f64; DIM]; DIM]; 8]>,
    /// Component weights, shape (K,), sum to 1. Stack-allocated for K ≤ 8.
    pub weights: SmallVec<[f64; 8]>,
}

impl SplitResult {
    /// Create an "unsplit" result (single component with original distribution).
    #[must_use]
    pub fn unsplit(mean: &[f64; DIM], cov: &[[f64; DIM]; DIM]) -> Self {
        let mut means = SmallVec::new();
        means.push(*mean);
        let mut covariances = SmallVec::new();
        covariances.push(*cov);
        let mut weights = SmallVec::new();
        weights.push(1.0);
        Self {
            means,
            covariances,
            weights,
        }
    }

    /// Create a new split result with K components.
    #[must_use]
    pub fn new(k: usize) -> Self {
        let mut means = SmallVec::with_capacity(k);
        let mut covariances = SmallVec::with_capacity(k);
        let mut weights = SmallVec::with_capacity(k);
        means.resize(k, [0.0; DIM]);
        covariances.resize(k, [[0.0; DIM]; DIM]);
        weights.resize(k, 0.0);

        Self {
            means,
            covariances,
            weights,
        }
    }

    /// Number of components.
    #[must_use]
    pub fn num_components(&self) -> usize {
        self.means.len()
    }
}

/// Configuration for splitting operations.
#[derive(Clone, Debug)]
pub struct SplitConfig {
    /// Default alpha as fraction of `alpha_max` (`H4`: 0.6).
    pub default_alpha_fraction: f64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            default_alpha_fraction: 0.6, // H4: Engineering judgment
        }
    }
}

// The six conversion helpers (`array_to_matrix6`, `matrix6_to_array`,
// `slice_to_vector6`, `vector6_to_array`, `symmetrize`, `symmetrize_array`)
// live in `common_rs` and are imported there by the modules that use them.
// This module used to `pub use` all six, so `linalg.rs` and `downdate.rs`
// reached `common_rs` through `crate::types` -- a hop that published nine
// names as `dust_splitting_rs` API that no crate outside this one ever read.
//
// They are TESTED in common_rs, which owns them. This module carried a
// `test_symmetrize_array` that was a strictly weaker copy of
// `common_rs::types6`'s test of the same name — same function, one
// off-diagonal pair instead of two — so it could only
// ever fail alongside the original. Do not re-add a local copy; a test here
// would have to exercise something this crate does that common_rs does not.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_result_unsplit() {
        let mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut cov = [[0.0; DIM]; DIM];
        for (index, row) in cov.iter_mut().enumerate() {
            if let Some(diagonal) = row.get_mut(index) {
                *diagonal = 1.0;
            }
        }

        let result = SplitResult::unsplit(&mean, &cov);
        assert_eq!(result.num_components(), 1);
        assert_eq!(
            result.means.first().map(|value| value.map(f64::to_bits)),
            Some(mean.map(f64::to_bits))
        );
        let weight = result.weights.first().copied().unwrap_or(f64::NAN);
        assert!((weight - 1.0).abs() < 1e-10);
    }
}
