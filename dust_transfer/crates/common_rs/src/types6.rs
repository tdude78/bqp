//! Common 6D types for orbital mechanics.
//!
//! Provides type aliases and conversion utilities for 6-dimensional state vectors
//! and covariance matrices used throughout the orbital mechanics crates.

use nalgebra::{SMatrix, SVector};

/// Dimensionality of state space (6D for orbital mechanics).
pub const DIM: usize = 6;

/// Fixed-size 6x6 matrix type.
pub type Matrix6x6 = SMatrix<f64, DIM, DIM>;

/// Fixed-size 6-element vector type.
pub type Vector6D = SVector<f64, DIM>;

// ============================================================================
// Conversion: Array <-> Nalgebra
// ============================================================================

/// Convert a 2D array to [`Matrix6x6`].
///
/// Input is row-major (Rust/C convention).
///
/// # Arguments
/// * `arr` - 6x6 array in row-major order
///
/// # Returns
/// * Nalgebra [`Matrix6x6`]
#[inline]
#[must_use]
pub fn array_to_matrix6(arr: &[[f64; DIM]; DIM]) -> Matrix6x6 {
    Matrix6x6::from_row_iterator(arr.iter().flatten().copied())
}

/// Convert a fixed-size array to [`Vector6D`].
///
/// # Arguments
/// * `s` - 6-element array
///
/// # Returns
/// * Nalgebra [`Vector6D`]
#[inline]
#[must_use]
pub fn slice_to_vector6(s: &[f64; DIM]) -> Vector6D {
    Vector6D::from_row_slice(s)
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Symmetrize a 2D array in-place: A = (A + A^T) / 2.
///
/// Array version for use with raw arrays instead of nalgebra types.
///
/// # Arguments
/// * `arr` - Array to symmetrize (modified in-place)
#[inline]
pub fn symmetrize_array(arr: &mut [[f64; DIM]; DIM]) {
    for i in 0..DIM {
        let Some(first_upper) = i.checked_add(1) else {
            continue;
        };
        let (head, tail) = arr.split_at_mut(first_upper);
        let Some(row_i) = head.get_mut(i) else {
            continue;
        };
        for (offset, row_j) in tail.iter_mut().enumerate() {
            let Some(j) = first_upper.checked_add(offset) else {
                continue;
            };
            let (Some(&upper), Some(&lower)) = (row_i.get(j), row_j.get(i)) else {
                continue;
            };
            let avg = 0.5 * (upper + lower);
            if let Some(upper) = row_i.get_mut(j) {
                *upper = avg;
            }
            if let Some(lower) = row_j.get_mut(i) {
                *lower = avg;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn identity_array() -> [[f64; DIM]; DIM] {
        let mut arr = [[0.0; DIM]; DIM];
        for (i, row) in arr.iter_mut().enumerate() {
            if let Some(value) = row.get_mut(i) {
                *value = 1.0;
            }
        }
        arr
    }

    #[test]
    fn test_array_to_matrix6_identity() {
        let arr = identity_array();
        let mat = array_to_matrix6(&arr);

        for (i, row) in arr.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let actual = mat.get((i, j)).copied().unwrap_or(f64::NAN);
                if i == j {
                    assert_relative_eq!(actual, 1.0, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(actual, *value, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_slice_to_vector6() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = slice_to_vector6(&arr);

        for (value, expected) in v.iter().zip(arr.iter()) {
            assert_relative_eq!(*value, *expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_symmetrize_array() {
        let mut arr = [[0.0; DIM]; DIM];
        arr[0][1] = 2.0;
        arr[1][0] = 4.0;
        arr[2][5] = 10.0;
        arr[5][2] = 20.0;

        symmetrize_array(&mut arr);

        assert_relative_eq!(arr[0][1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(arr[1][0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(arr[2][5], 15.0, epsilon = 1e-10);
        assert_relative_eq!(arr[5][2], 15.0, epsilon = 1e-10);
    }
}
