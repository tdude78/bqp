//! Linear algebra operations for 6x6 matrices.
//!
//! Uses nalgebra for Cholesky decomposition and eigenvalue computation.
//! SIMD-accelerated hot paths available with the `simd` feature.

use nalgebra::{Cholesky, SymmetricEigen};
use std::fmt;

use wide::f64x4;

use common_rs::{array_to_matrix6, slice_to_vector6, Vector6D, DIM};

const MIN_EIGENVALUE: f64 = 1e-12;
const DIM_F64: f64 = 6.0;
const POWER_MAX_ITER: usize = 10;
const POWER_CONVERGE_TOL: f64 = 1e-10;

/// Validation failures for authoritative covariance-axis selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CovarianceAxisError {
    NonFinite,
    NonSymmetric,
    DegenerateAxis,
}

impl fmt::Display for CovarianceAxisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonFinite => "covariance must contain only finite values",
            Self::NonSymmetric => "covariance must be symmetric",
            Self::DegenerateAxis => "principal covariance axis could not be resolved",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for CovarianceAxisError {}

/// Solve P * x = b via Cholesky decomposition.
///
/// Returns `Some(x)` if P is positive definite, `None` otherwise.
/// Falls back to LDLT if Cholesky fails.
///
/// # Arguments
/// * `p` - 6x6 positive semi-definite matrix (row-major)
/// * `b` - 6-element right-hand side vector
///
/// # Returns
/// * `Some([f64; 6])` - Solution vector x
/// * `None` - If decomposition fails
#[inline]
#[must_use]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "fixed-size eigensolve applies IEEE matrix arithmetic"
)]
pub fn cholesky_solve6(p: &[[f64; DIM]; DIM], b: &[f64; DIM]) -> Option<[f64; DIM]> {
    let p_mat = array_to_matrix6(p);
    let b_vec = slice_to_vector6(b);

    // Try Cholesky first (faster, requires strict positive definiteness)
    if let Some(chol) = Cholesky::new(p_mat) {
        let x = chol.solve(&b_vec);
        return Some(vector6_to_array_inline(&x));
    }

    // Fall back to eigendecomposition-based solve for near-singular matrices
    // This is more robust but slower
    let eigen = SymmetricEigen::new(p_mat);
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    // Check for near-zero eigenvalues
    if eigenvalues
        .iter()
        .any(|eigenvalue| eigenvalue.abs() < MIN_EIGENVALUE)
    {
        return None;
    }

    // Solve via eigendecomposition: P = V * D * V^T
    // x = V * D^{-1} * V^T * b
    let vt_b = eigenvectors.transpose() * b_vec;
    let mut d_inv_vt_b = Vector6D::zeros();
    for ((output, &projected), &eigenvalue) in d_inv_vt_b
        .iter_mut()
        .zip(vt_b.iter())
        .zip(eigenvalues.iter())
    {
        *output = projected / eigenvalue;
    }
    let x = eigenvectors * d_inv_vt_b;

    Some(vector6_to_array_inline(&x))
}

/// Compute symmetric eigendecomposition of a 6x6 matrix.
///
/// Returns (eigenvalues, eigenvectors) sorted by eigenvalue magnitude.
/// Eigenvectors are column vectors.
///
/// # Arguments
/// * `p` - 6x6 symmetric matrix (row-major)
///
/// # Returns
/// * `([f64; 6], [[f64; 6]; 6])` - (eigenvalues, eigenvectors as columns)
#[inline]
#[must_use]
pub(crate) fn symmetric_eigen6(p: &[[f64; DIM]; DIM]) -> ([f64; DIM], [[f64; DIM]; DIM]) {
    let p_mat = array_to_matrix6(p);
    let eigen = SymmetricEigen::new(p_mat);

    let eigenvalues =
        std::array::from_fn(|index| eigen.eigenvalues.get(index).copied().unwrap_or(0.0));
    let eigenvectors = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            eigen
                .eigenvectors
                .get((row, column))
                .copied()
                .unwrap_or(0.0)
        })
    });

    (eigenvalues, eigenvectors)
}

/// Select the principal covariance axis with deterministic tie and sign authority.
///
/// Eigenvalues are ordered descending. Repeated eigenvalues use a spectral-projector
/// basis seeded by coordinate order, making the result independent of the arbitrary
/// eigenvectors returned for a degenerate eigenspace. Each axis is signed so its
/// largest-magnitude component (lowest index on ties) is non-negative.
///
/// # Errors
///
/// Returns [`CovarianceAxisError`] for non-finite or asymmetric covariance, or
/// a numerically degenerate projector basis.
pub fn principal_covariance_axis6(
    covariance: &[[f64; DIM]; DIM],
) -> Result<[f64; DIM], CovarianceAxisError> {
    let mut symmetric = [[0.0; DIM]; DIM];
    let mut scale = 1.0_f64;
    for row in covariance {
        for &value in row {
            if !value.is_finite() {
                return Err(CovarianceAxisError::NonFinite);
            }
            scale = scale.max(value.abs());
        }
    }
    let symmetry_tolerance = 128.0 * f64::EPSILON * scale;
    for (row_index, symmetric_row) in symmetric.iter_mut().enumerate() {
        for (column_index, symmetric_value) in symmetric_row.iter_mut().enumerate() {
            let direct = covariance
                .get(row_index)
                .and_then(|row| row.get(column_index))
                .copied()
                .unwrap_or(f64::NAN);
            let transpose = covariance
                .get(column_index)
                .and_then(|row| row.get(row_index))
                .copied()
                .unwrap_or(f64::NAN);
            if (direct - transpose).abs() > symmetry_tolerance {
                return Err(CovarianceAxisError::NonSymmetric);
            }
            *symmetric_value = 0.5 * (direct + transpose);
        }
    }

    let (eigenvalues, eigenvectors) = symmetric_eigen6(&symmetric);
    if eigenvalues.iter().any(|value| !value.is_finite())
        || eigenvectors
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(CovarianceAxisError::NonFinite);
    }

    let mut order = [0, 1, 2, 3, 4, 5];
    order.sort_by(|left, right| {
        let right_value = eigenvalues.get(*right).copied().unwrap_or(f64::NAN);
        let left_value = eigenvalues.get(*left).copied().unwrap_or(f64::NAN);
        right_value.total_cmp(&left_value)
    });

    let eigenvalue_scale = eigenvalues
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    let tie_tolerance = 128.0 * f64::EPSILON * eigenvalue_scale;
    let basis_tolerance = 256.0 * f64::EPSILON;
    let leading_index = order.first().copied().unwrap_or(0);
    let leading_value = eigenvalues.get(leading_index).copied().unwrap_or(f64::NAN);
    let mut group_end = 1;
    while group_end < DIM
        && order
            .get(group_end)
            .and_then(|&index| eigenvalues.get(index))
            .is_some_and(|&value| (value - leading_value).abs() <= tie_tolerance)
    {
        group_end = group_end.saturating_add(1);
    }

    let mut projector = [[0.0; DIM]; DIM];
    for &eigen_index in order.get(..group_end).unwrap_or(&[]) {
        for (row_index, projector_row) in projector.iter_mut().enumerate() {
            let left = eigenvectors
                .get(row_index)
                .and_then(|row| row.get(eigen_index))
                .copied()
                .unwrap_or(0.0);
            for (column_index, projector_value) in projector_row.iter_mut().enumerate() {
                let right = eigenvectors
                    .get(column_index)
                    .and_then(|row| row.get(eigen_index))
                    .copied()
                    .unwrap_or(0.0);
                *projector_value += left * right;
            }
        }
    }

    for coordinate in 0..DIM {
        let mut candidate = [0.0; DIM];
        for (candidate_value, projector_row) in candidate.iter_mut().zip(&projector) {
            *candidate_value = projector_row.get(coordinate).copied().unwrap_or(0.0);
        }
        let norm = candidate
            .iter()
            .map(|component| component * component)
            .sum::<f64>()
            .sqrt();
        if norm <= basis_tolerance {
            continue;
        }
        for component in &mut candidate {
            *component /= norm;
        }
        canonicalize_axis_sign(&mut candidate);
        return Ok(candidate);
    }

    Err(CovarianceAxisError::DegenerateAxis)
}

fn canonicalize_axis_sign(axis: &mut [f64; DIM]) {
    let mut pivot_value = axis.first().copied().unwrap_or(0.0);
    for &component in axis.iter().skip(1) {
        if component.abs() > pivot_value.abs() {
            pivot_value = component;
        }
    }
    if pivot_value.is_sign_negative() {
        for component in axis {
            *component = -*component;
        }
    }
}

/// Get the dominant eigenvector (corresponding to largest eigenvalue) of a 6x6 symmetric matrix.
///
/// Uses power iteration (10 steps) which is 10-50x cheaper than full eigendecomposition
/// for extracting a single dominant eigenvector from a 6x6 matrix.
/// Falls back to full [`SymmetricEigen`] if the matrix is near-zero.
///
/// Returns `None` if the eigenvector has zero norm.
///
/// # Arguments
/// * `p` - 6x6 symmetric matrix (row-major)
///
/// # Returns
/// * `Some([f64; 6])` - Unit eigenvector corresponding to largest eigenvalue
/// * `None` - If eigenvector has zero norm
#[inline]
#[must_use]
pub fn dominant_eigenvector6(p: &[[f64; DIM]; DIM]) -> Option<[f64; DIM]> {
    // Check for near-zero matrix by inspecting diagonal (Frobenius norm lower bound)
    let diag_sq: f64 = p
        .iter()
        .enumerate()
        .filter_map(|(index, row)| row.get(index))
        .map(|value| value * value)
        .sum();
    if diag_sq <= 0.0 || !diag_sq.is_finite() {
        return None;
    }

    // Shift matrix by a small multiple of the diagonal sum to ensure positive definiteness,
    // which improves power iteration convergence for indefinite matrices.
    // shift = trace(P) / DIM keeps the dominant eigenvector unchanged.
    let trace: f64 = p
        .iter()
        .enumerate()
        .filter_map(|(index, row)| row.get(index))
        .sum();
    let shift = trace.abs() / DIM_F64;

    // Initial vector: use [1,1,1,1,1,1]/sqrt(6) as seed
    let inv_sqrt6 = 1.0 / DIM_F64.sqrt();
    let mut v = [inv_sqrt6; DIM];

    // Power iteration with convergence check.
    // Falls back to full SymmetricEigen when convergence is slow (nearly-equal eigenvalues).
    let mut converged = false;
    for _ in 0..POWER_MAX_ITER {
        let mut w = [0.0; DIM];
        for ((output, row), &vector_component) in w.iter_mut().zip(p).zip(&v) {
            let mut sum = shift * vector_component;
            for (&matrix_component, &input_component) in row.iter().zip(&v) {
                sum += matrix_component * input_component;
            }
            *output = sum;
        }
        let norm_sq: f64 = w.iter().map(|x| x * x).sum();
        if norm_sq <= 0.0 || !norm_sq.is_finite() {
            break;
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        let mut diff_sq = 0.0;
        for (vector_component, &work_component) in v.iter_mut().zip(&w) {
            let new_value = work_component * inv_norm;
            let difference = new_value - *vector_component;
            diff_sq += difference * difference;
            *vector_component = new_value;
        }
        if diff_sq < POWER_CONVERGE_TOL * POWER_CONVERGE_TOL {
            converged = true;
            break;
        }
    }

    if !converged {
        // Slow convergence (nearly-equal eigenvalues) — use full decomposition
        let p_mat = array_to_matrix6(p);
        let eigen = SymmetricEigen::new(p_mat);
        let max_index = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 {
                    (index, value)
                } else {
                    best
                }
            })
            .0;
        let out = std::array::from_fn(|row| {
            eigen
                .eigenvectors
                .get((row, max_index))
                .copied()
                .unwrap_or(0.0)
        });
        return Some(out);
    }

    // Final norm check
    let norm_sq: f64 = v.iter().map(|x| x * x).sum();
    if norm_sq <= 0.0 || !norm_sq.is_finite() {
        return None;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    for val in &mut v {
        *val *= inv_norm;
    }

    Some(v)
}

/// Compute quadratic form u^T * P * u.
///
/// # Arguments
/// * `p` - 6x6 symmetric matrix (row-major)
/// * `u` - 6-element vector
///
/// # Returns
/// * Scalar value u^T * P * u
#[inline]
#[must_use]
pub fn quadratic_form6(p: &[[f64; DIM]; DIM], u: &[f64; DIM]) -> f64 {
    quadratic_form6_simd(p, u)
}

/// Scalar implementation of quadratic form.
#[cfg(test)]
#[inline]
fn quadratic_form6_scalar(p: &[[f64; DIM]; DIM], u: &[f64; DIM]) -> f64 {
    let mut result = 0.0;
    for (row, &left) in p.iter().zip(u) {
        let row_dot = row
            .iter()
            .zip(u)
            .fold(0.0, |sum, (&matrix, &right)| matrix.mul_add(right, sum));
        result = left.mul_add(row_dot, result);
    }
    result
}

/// SIMD implementation of quadratic form using f64x4.
///
/// Processes 4 elements at a time for the inner loop, handling the remaining
/// 2 elements with scalar ops. For a 6x6 matrix, this gives ~2x speedup.
#[inline]
fn quadratic_form6_simd(p: &[[f64; DIM]; DIM], u: &[f64; DIM]) -> f64 {
    // For each row i, compute u[i] * sum_j(p[i][j] * u[j])
    // Process columns 0-3 with SIMD, columns 4-5 with scalar

    let [u0, u1, u2, u3, u4, u5] = *u;
    let u_vec4 = f64x4::new([u0, u1, u2, u3]);
    let mut total = 0.0;

    for (row, &u_i) in p.iter().zip(u) {
        let [p0, p1, p2, p3, p4, p5] = *row;

        // SIMD: columns 0-3
        let p_row4 = f64x4::new([p0, p1, p2, p3]);
        let prod4 = p_row4 * u_vec4;
        let sum4 = prod4.reduce_add();

        // Scalar: columns 4-5
        let sum_tail = p4 * u4 + p5 * u5;

        total += u_i * (sum4 + sum_tail);
    }

    total
}

/// Compute outer product u * u^T and scale by factor.
///
/// Returns factor * (u * u^T) as a 6x6 matrix.
///
/// # Arguments
/// * `u` - 6-element vector
/// * `factor` - Scalar multiplier
///
/// # Returns
/// * `[[f64; 6]; 6]` - Resulting matrix
#[inline]
#[must_use]
pub(crate) fn outer_product_scaled6(u: &[f64; DIM], factor: f64) -> [[f64; DIM]; DIM] {
    outer_product_scaled6_simd(u, factor)
}

/// Scalar implementation of outer product.
#[cfg(test)]
#[inline]
fn outer_product_scaled6_scalar(u: &[f64; DIM], factor: f64) -> [[f64; DIM]; DIM] {
    let mut result = [[0.0; DIM]; DIM];
    for (output_row, &left) in result.iter_mut().zip(u) {
        for (output, &right) in output_row.iter_mut().zip(u) {
            *output = factor * left * right;
        }
    }
    result
}

/// SIMD implementation of outer product using f64x4.
///
/// For each row i, computes factor * u[i] * [u[0..4]] using SIMD,
/// then handles columns 4-5 with scalar ops.
#[inline]
fn outer_product_scaled6_simd(u: &[f64; DIM], factor: f64) -> [[f64; DIM]; DIM] {
    let mut result = [[0.0; DIM]; DIM];

    // Pre-compute SIMD vector for columns 0-3
    let [u0, u1, u2, u3, u4, u5] = *u;
    let u_vec4 = f64x4::new([u0, u1, u2, u3]);

    for (output_row, &input) in result.iter_mut().zip(u) {
        let scale = factor * input;
        let scale_vec = f64x4::splat(scale);

        // SIMD: columns 0-3
        let [out0, out1, out2, out3] = (scale_vec * u_vec4).to_array();
        *output_row = [out0, out1, out2, out3, scale * u4, scale * u5];
    }

    result
}

/// Subtract one 6x6 matrix from another: result = a - b.
#[inline]
#[must_use]
pub(crate) fn matrix_sub6(a: &[[f64; DIM]; DIM], b: &[[f64; DIM]; DIM]) -> [[f64; DIM]; DIM] {
    matrix_sub6_simd(a, b)
}

/// Scalar implementation of matrix subtraction.
#[cfg(test)]
#[inline]
fn matrix_sub6_scalar(a: &[[f64; DIM]; DIM], b: &[[f64; DIM]; DIM]) -> [[f64; DIM]; DIM] {
    let mut result = [[0.0; DIM]; DIM];
    for ((output_row, a_row), b_row) in result.iter_mut().zip(a).zip(b) {
        for ((output, &left), &right) in output_row.iter_mut().zip(a_row).zip(b_row) {
            *output = left - right;
        }
    }
    result
}

/// SIMD implementation of matrix subtraction using f64x4.
///
/// Processes 4 elements at a time per row, handles remaining 2 with scalar.
#[inline]
fn matrix_sub6_simd(a: &[[f64; DIM]; DIM], b: &[[f64; DIM]; DIM]) -> [[f64; DIM]; DIM] {
    let mut result = [[0.0; DIM]; DIM];

    for ((output_row, a_row), b_row) in result.iter_mut().zip(a).zip(b) {
        let [a0, a1, a2, a3, a4, a5] = *a_row;
        let [b0, b1, b2, b3, b4, b5] = *b_row;
        // SIMD: columns 0-3
        let a_simd = f64x4::new([a0, a1, a2, a3]);
        let b_simd = f64x4::new([b0, b1, b2, b3]);
        let [out0, out1, out2, out3] = (a_simd - b_simd).to_array();
        *output_row = [out0, out1, out2, out3, a4 - b4, a5 - b5];
    }

    result
}

/// Normalize a vector to unit length in-place.
///
/// Returns `true` if normalization succeeded, `false` if vector has zero norm.
#[inline]
pub(crate) fn normalize_inplace6(v: &mut [f64; DIM]) -> bool {
    let norm_sq = v[0].mul_add(
        v[0],
        v[1].mul_add(
            v[1],
            v[2].mul_add(v[2], v[3].mul_add(v[3], v[4].mul_add(v[4], v[5] * v[5]))),
        ),
    );
    if norm_sq <= 0.0 || !norm_sq.is_finite() {
        return false;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    for val in v.iter_mut() {
        *val *= inv_norm;
    }
    true
}

/// Helper to convert `Vector6D` to array (inline version to avoid import)
#[inline]
fn vector6_to_array_inline(v: &Vector6D) -> [f64; DIM] {
    let mut arr = [0.0; DIM];
    for (output, &input) in arr.iter_mut().zip(v.iter()) {
        *output = input;
    }
    arr
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "dense fixed-size test fixtures preserve explicit reference arithmetic"
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

    fn matrix_at(matrix: &[[f64; DIM]; DIM], row: usize, column: usize) -> f64 {
        matrix
            .get(row)
            .and_then(|values| values.get(column))
            .copied()
            .unwrap_or(f64::NAN)
    }

    #[test]
    fn test_cholesky_solve_identity() {
        let p = identity_cov();
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let x = cholesky_solve6(&p, &b).expect("should solve");

        for (&actual, &expected) in x.iter().zip(&b) {
            assert_relative_eq!(actual, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cholesky_solve_diagonal() {
        let mut p = [[0.0; DIM]; DIM];
        for (index, row) in p.iter_mut().enumerate() {
            if let Some(diagonal) = row.get_mut(index) {
                *diagonal = (index + 1) as f64;
            }
        }
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let x = cholesky_solve6(&p, &b).expect("should solve");

        for (index, (&actual, &rhs)) in x.iter().zip(&b).enumerate() {
            assert_relative_eq!(actual, rhs / (index + 1) as f64, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_symmetric_eigen_identity() {
        let p = identity_cov();
        let (eigenvalues, eigenvectors) = symmetric_eigen6(&p);

        // All eigenvalues should be 1.0
        for ev in &eigenvalues {
            assert_relative_eq!(*ev, 1.0, epsilon = 1e-10);
        }

        // Eigenvectors should be orthonormal
        for column in 0..DIM {
            let mut norm_sq = 0.0;
            for row in &eigenvectors {
                let component = row.get(column).copied().unwrap_or(f64::NAN);
                norm_sq += component * component;
            }
            assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn principal_covariance_axis_selects_largest_distinct_diagonal_eigenvalue() {
        let mut covariance = [[0.0; DIM]; DIM];
        for (index, value) in [1.0, 7.0, 2.0, 6.0, 3.0, 5.0].into_iter().enumerate() {
            if let Some(diagonal) = covariance.get_mut(index).and_then(|row| row.get_mut(index)) {
                *diagonal = value;
            }
        }

        let axis = principal_covariance_axis6(&covariance).expect("valid covariance");

        assert_eq!(
            axis.map(f64::to_bits),
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0].map(f64::to_bits)
        );
    }

    #[test]
    fn principal_covariance_axis_rejects_non_finite_covariance() {
        let mut covariance = identity_cov();
        if let Some(value) = covariance.get_mut(2).and_then(|row| row.get_mut(4)) {
            *value = f64::NAN;
        }
        if let Some(value) = covariance.get_mut(4).and_then(|row| row.get_mut(2)) {
            *value = f64::NAN;
        }

        assert!(principal_covariance_axis6(&covariance).is_err());
    }

    #[test]
    fn principal_covariance_axis_resolves_eigenvalue_ties_deterministically() {
        let covariance = identity_cov();

        let first = principal_covariance_axis6(&covariance).expect("valid covariance");
        let second = principal_covariance_axis6(&covariance).expect("valid covariance");

        assert_eq!(first.map(f64::to_bits), second.map(f64::to_bits));
        assert_eq!(
            first.map(f64::to_bits),
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0].map(f64::to_bits)
        );
    }

    #[test]
    fn test_dominant_eigenvector() {
        let mut p = [[0.0; DIM]; DIM];
        for (index, row) in p.iter_mut().enumerate() {
            if let Some(diagonal) = row.get_mut(index) {
                *diagonal = (index + 1) as f64;
            }
        }
        // Largest eigenvalue is 6.0, eigenvector should point in direction of last coordinate

        let v = dominant_eigenvector6(&p).expect("should find eigenvector");

        // Should be approximately [0, 0, 0, 0, 0, 1] (or its negative)
        assert!(v.last().copied().unwrap_or(f64::NAN).abs() > 0.99);
        for &component in v.iter().take(5) {
            assert!(component.abs() < 0.01);
        }
    }

    #[test]
    fn test_quadratic_form() {
        let p = identity_cov();
        let u = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let result = quadratic_form6(&p, &u);
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_outer_product_scaled() {
        let u = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let result = outer_product_scaled6(&u, 2.0);

        assert_relative_eq!(matrix_at(&result, 0, 0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(matrix_at(&result, 0, 1), 4.0, epsilon = 1e-10);
        assert_relative_eq!(matrix_at(&result, 1, 0), 4.0, epsilon = 1e-10);
        assert_relative_eq!(matrix_at(&result, 1, 1), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_inplace() {
        let mut v = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        assert!(normalize_inplace6(&mut v));
        assert_relative_eq!(v.first().copied().unwrap_or(f64::NAN), 0.6, epsilon = 1e-10);
        assert_relative_eq!(v.get(1).copied().unwrap_or(f64::NAN), 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = [0.0; DIM];
        assert!(!normalize_inplace6(&mut v));
    }

    // ---------------------------------------------------------------------
    // Scalar-vs-SIMD equivalence.
    //
    // The public wrappers dispatch unconditionally to the `_simd` kernels,
    // which split a 6-wide problem into an f64x4 body (columns 0-3) plus a
    // 2-wide scalar tail (columns 4-5). The value-based tests above use
    // identity matrices and unit basis vectors, so they never load a
    // distinguishable value into the tail columns and cannot detect a wrong
    // tail. These tests pin each SIMD kernel against its scalar twin on dense
    // inputs where every one of the 36 (or 6) slots is distinct.
    //
    // Tolerances are relative, not exact: the two kernels accumulate in
    // different orders, so they are not required to be bit-identical.
    // ---------------------------------------------------------------------

    /// Dense 6x6 with every entry distinct and asymmetric magnitudes, so a
    /// transposed, truncated, or tail-dropping kernel cannot coincide.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "dense test fixture uses checked small indices and IEEE values"
    )]
    fn dense_matrix() -> [[f64; DIM]; DIM] {
        let mut m = [[0.0; DIM]; DIM];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = 1.0 + (i * DIM + j) as f64 * 0.37 - if (i + j) % 3 == 0 { 2.9 } else { 0.0 };
            }
        }
        m
    }

    /// Symmetric dense matrix, for the quadratic form (which assumes symmetry).
    fn dense_symmetric() -> [[f64; DIM]; DIM] {
        let m = dense_matrix();
        let mut s = [[0.0; DIM]; DIM];
        for (row_index, row) in s.iter_mut().enumerate() {
            for (column_index, value) in row.iter_mut().enumerate() {
                *value = 0.5
                    * (matrix_at(&m, row_index, column_index)
                        + matrix_at(&m, column_index, row_index));
            }
        }
        s
    }

    fn dense_vector() -> [f64; DIM] {
        [1.7, -0.4, 3.25, -2.125, 0.875, -4.5]
    }

    #[test]
    fn quadratic_form6_simd_matches_scalar_on_dense_input() {
        let p = dense_symmetric();
        let u = dense_vector();
        assert_relative_eq!(
            quadratic_form6_simd(&p, &u),
            quadratic_form6_scalar(&p, &u),
            max_relative = 1e-12
        );
    }

    #[test]
    fn outer_product_scaled6_simd_matches_scalar_on_dense_input() {
        let u = dense_vector();
        let simd = outer_product_scaled6_simd(&u, -1.75);
        let scalar = outer_product_scaled6_scalar(&u, -1.75);
        for (row_index, row) in simd.iter().enumerate() {
            for (column_index, &actual) in row.iter().enumerate() {
                assert_relative_eq!(
                    actual,
                    matrix_at(&scalar, row_index, column_index),
                    max_relative = 1e-12
                );
            }
        }
    }

    #[test]
    fn matrix_sub6_simd_matches_scalar_on_dense_input() {
        let a = dense_matrix();
        let b = dense_symmetric();
        let simd = matrix_sub6_simd(&a, &b);
        let scalar = matrix_sub6_scalar(&a, &b);
        for (row_index, row) in simd.iter().enumerate() {
            for (column_index, &actual) in row.iter().enumerate() {
                assert_relative_eq!(
                    actual,
                    matrix_at(&scalar, row_index, column_index),
                    max_relative = 1e-12
                );
            }
        }
    }

    /// The tail columns (4-5) are the half the SIMD body does not cover.
    /// Zeroing everything else makes quadratic-form correctness depend on
    /// reading those scalar tail columns.
    #[test]
    fn quadratic_form_simd_reads_the_scalar_tail_columns() {
        let mut q = [[0.0; DIM]; DIM];
        q[4][4] = 2.0;
        q[5][5] = 3.0;
        let u = [0.0, 0.0, 0.0, 0.0, 2.0, 3.0];
        assert_relative_eq!(quadratic_form6_simd(&q, &u), 35.0, epsilon = 1e-12);
    }
}
