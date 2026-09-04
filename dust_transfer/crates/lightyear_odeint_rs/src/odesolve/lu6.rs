//! Compact 6×6 LU decomposition with partial pivoting.
//!
//! This is the linear algebra kernel used by Newton iterations in the
//! ESDIRK implicit solver. The matrix size is hardcoded at 6 because
//! the orbital state vector is always 6-dimensional (position + velocity).

/// LU factorization of a 6×6 matrix with partial pivoting.
///
/// Stores L (unit lower triangular, below diagonal) and U (upper triangular,
/// on and above diagonal) in a single 6×6 array. The permutation is stored
/// in `piv`.
#[derive(Debug, Clone)]
pub struct Lu6 {
    /// Combined L/U factors stored in-place.
    pub a: [[f64; 6]; 6],
    /// Pivot indices: row i was swapped with row piv[i].
    pub piv: [usize; 6],
    /// True if the matrix is singular (zero or near-zero pivot encountered).
    pub singular: bool,
}

impl Lu6 {
    /// Factor the matrix `mat` into PA = LU with partial pivoting.
    ///
    /// The input matrix is copied; the original is not modified.
    /// If a zero pivot is encountered, `singular` is set to true and
    /// the factorization is left in an incomplete but safe state.
    #[must_use]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "preserve the established IEEE operation order in LU factorization"
    )]
    pub fn factor(mat: &[[f64; 6]; 6]) -> Self {
        let mut a = *mat;
        let mut piv = [0usize; 6];
        let mut singular = false;

        for k in 0..6 {
            // Find pivot: largest |a[i][k]| for i >= k
            let mut max_val = 0.0;
            let mut max_row = k;
            for (i, row) in a.iter().enumerate().skip(k) {
                let Some(&entry) = row.get(k) else {
                    singular = true;
                    continue;
                };
                let v = entry.abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }
            let Some(pivot_slot) = piv.get_mut(k) else {
                singular = true;
                continue;
            };
            *pivot_slot = max_row;

            if max_val < 1e-30 {
                singular = true;
                // Leave a[k][k] as is; continuing would divide by near-zero
                // but we mark singular so callers know not to trust the result.
                continue;
            }

            // Swap rows k and max_row
            if max_row != k {
                a.swap(k, max_row);
            }

            let Some(pivot) = a.get(k).and_then(|row| row.get(k)).copied() else {
                singular = true;
                continue;
            };
            let pivot_inv = 1.0 / pivot;

            // Eliminate below
            let Some(pivot_row) = a.get(k).copied() else {
                singular = true;
                continue;
            };
            for row in a.iter_mut().skip(k.saturating_add(1)) {
                let Some(factor_entry) = row.get_mut(k) else {
                    singular = true;
                    continue;
                };
                *factor_entry *= pivot_inv; // L factor
                let factor = *factor_entry;
                for (j, value) in row.iter_mut().enumerate().skip(k + 1) {
                    let Some(&upper) = pivot_row.get(j) else {
                        singular = true;
                        continue;
                    };
                    *value -= factor * upper; // U factor
                }
            }
        }

        Self { a, piv, singular }
    }

    /// Solve Ax = b in-place: `b` is overwritten with the solution x.
    ///
    /// Must be called after `factor()`. If the matrix was singular,
    /// the result is undefined but will not panic.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "preserve the established IEEE operation order in triangular solves"
    )]
    pub fn solve(&self, b: &mut [f64; 6]) {
        if self.piv.iter().any(|&pivot| pivot >= b.len()) {
            b.fill(0.0);
            return;
        }
        // Apply row permutation (forward)
        for (k, &p) in self.piv.iter().enumerate() {
            if p != k {
                b.swap(k, p);
            }
        }

        // Forward substitution: L * z = Pb
        for (i, row) in self.a.iter().enumerate().skip(1) {
            let sum: f64 = row
                .iter()
                .take(i)
                .zip(b.iter())
                .map(|(a_ij, b_j)| a_ij * b_j)
                .sum();
            let Some(value) = b.get_mut(i) else {
                b.fill(0.0);
                return;
            };
            *value -= sum;
        }

        // Back substitution: U * x = z
        for i in (0..6).rev() {
            let Some(row) = self.a.get(i) else {
                b.fill(0.0);
                return;
            };
            let sum: f64 = row
                .iter()
                .skip(i + 1)
                .zip(b.iter().skip(i + 1))
                .map(|(a_ij, b_j)| a_ij * b_j)
                .sum();
            let Some(diagonal) = row.get(i).copied() else {
                b.fill(0.0);
                return;
            };
            let Some(value) = b.get_mut(i) else {
                b.fill(0.0);
                return;
            };
            *value = if diagonal.abs() > 1e-30 {
                (*value - sum) / diagonal
            } else {
                0.0 // singular pivot: return zero rather than inf/nan
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let eye: [[f64; 6]; 6] = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let lu = Lu6::factor(&eye);
        assert!(!lu.singular);

        let mut b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_orig = b;
        lu.solve(&mut b);
        for (i, (&value, &expected)) in b.iter().zip(b_orig.iter()).enumerate() {
            assert!(
                (value - expected).abs() < 1e-14,
                "Identity solve failed at index {i}"
            );
        }
    }

    #[test]
    fn test_known_system() {
        // Solve:
        //   2x + 1y + 1z + 0 + 0 + 0 = 7
        //   4x + 3y + 3z + 1 + 0 + 0 = 23
        //   8x + 7y + 9z + 5 + 1 + 0 = 45
        //   6x + 7y + 9z + 8 + 2 + 1 = 40
        //   0  + 0  + 0  + 1 + 2 + 3 = 6
        //   0  + 0  + 0  + 0 + 1 + 4 = 5
        //
        // Solution: x=[1,1,1,1,1,1]
        let a: [[f64; 6]; 6] = [
            [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [4.0, 3.0, 3.0, 1.0, 0.0, 0.0],
            [8.0, 7.0, 9.0, 5.0, 1.0, 0.0],
            [6.0, 7.0, 9.0, 8.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 4.0],
        ];
        let lu = Lu6::factor(&a);
        assert!(!lu.singular);

        let mut b = [4.0, 11.0, 30.0, 33.0, 6.0, 5.0];
        lu.solve(&mut b);
        for (i, &value) in b.iter().enumerate() {
            assert!(
                (value - 1.0).abs() < 1e-12,
                "Index {i}: expected 1.0, got {value}"
            );
        }
    }

    #[test]
    fn test_random_system() {
        // A = rotation-ish dense matrix, known solution
        let a: [[f64; 6]; 6] = [
            [3.0, 1.0, -1.0, 0.5, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0, 0.5, 0.0],
            [-1.0, 1.0, 5.0, 0.0, 0.0, 0.5],
            [0.5, 0.0, 0.0, 3.0, 1.0, -1.0],
            [0.0, 0.5, 0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 0.5, -1.0, 1.0, 5.0],
        ];
        let x_true = [1.0, -2.0, 3.0, -1.0, 2.0, -3.0];

        // Compute b = A * x_true
        let mut b = [0.0f64; 6];
        for (row, out) in a.iter().zip(b.iter_mut()) {
            *out = row
                .iter()
                .zip(x_true.iter())
                .map(|(a_ij, x_j)| a_ij * x_j)
                .sum();
        }

        let lu = Lu6::factor(&a);
        assert!(!lu.singular);
        lu.solve(&mut b);

        for (i, (&value, &expected)) in b.iter().zip(x_true.iter()).enumerate() {
            assert!(
                (value - expected).abs() < 1e-12,
                "Index {}: expected {}, got {} (err = {:e})",
                i,
                expected,
                value,
                (value - expected).abs()
            );
        }
    }

    #[test]
    fn test_singular_detection() {
        let zero: [[f64; 6]; 6] = [[0.0; 6]; 6];
        let lu = Lu6::factor(&zero);
        assert!(lu.singular);
    }

    #[test]
    fn test_iteration_matrix() {
        // Test solving (I - h*gamma*J)*dk = -G
        // where J is a simple Jacobian, simulating what the ESDIRK Newton iteration does
        let h = 0.1;
        let gamma = 0.25;

        // Simple Jacobian: harmonic oscillator [0 1; -1 0] extended to 6D
        let identity = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let mut w = identity;
        // I - h*gamma*J
        // Position-velocity coupling (first 3 are position, last 3 are velocity)
        // dy/dt = v => J[0][3] = 1, J[1][4] = 1, J[2][5] = 1
        // dv/dt = -y => J[3][0] = -1, J[4][1] = -1, J[5][2] = -1
        w[0][3] -= h * gamma * 1.0;
        w[1][4] -= h * gamma * 1.0;
        w[2][5] -= h * gamma * 1.0;
        w[3][0] -= -(h * gamma);
        w[4][1] -= -(h * gamma);
        w[5][2] -= -(h * gamma);

        let lu = Lu6::factor(&w);
        assert!(!lu.singular);

        let mut rhs = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        lu.solve(&mut rhs);

        // Verify: W * solution = original rhs
        let mut check = [0.0f64; 6];
        // Reconstruct W
        let mut w2 = identity;
        w2[0][3] -= h * gamma * 1.0;
        w2[1][4] -= h * gamma * 1.0;
        w2[2][5] -= h * gamma * 1.0;
        w2[3][0] -= -(h * gamma);
        w2[4][1] -= -(h * gamma);
        w2[5][2] -= -(h * gamma);

        for (row, out) in w2.iter().zip(check.iter_mut()) {
            *out = row
                .iter()
                .zip(rhs.iter())
                .map(|(a_ij, x_j)| a_ij * x_j)
                .sum();
        }

        let orig = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for (i, (&value, &expected)) in check.iter().zip(orig.iter()).enumerate() {
            assert!(
                (value - expected).abs() < 1e-14,
                "W*x check failed at {i}: {value} vs {expected}"
            );
        }
    }
}
