//! Kennedy–Carpenter ESDIRK4(3)6L[2]SA tableau.
//!
//! 6-stage, order 4 method with embedded order 3 error estimate.
//! L-stable, stiffly accurate (last row of A equals b).
//!
//! Reference: C.A. Kennedy & M.H. Carpenter, "Additive Runge–Kutta schemes
//! for convection–diffusion–reaction equations", Appl. Numer. Math. 44 (2003) 139–181.
//! Coefficients taken from SUNDIALS/ARKode (`ARK436L2SA_DIRK_6_3_4`).

/// Butcher tableau for an ESDIRK method with fixed 6×6 arrays.
#[derive(Debug)]
pub struct EsdirkTableau {
    /// Number of stages.
    pub stages: usize,
    /// Lower-triangular coefficient matrix A[i][j], i=0..5, j=0..5.
    /// Diagonal entries are all equal to `gamma`.
    /// A[0][j] = 0 for all j (explicit first stage).
    pub a: [[f64; 6]; 6],
    /// Abscissa (node) vector c[i] = `sum_j` A[i][j].
    pub c: [f64; 6],
    /// Solution weights (order 4).
    pub b: [f64; 6],
    /// Embedded weights (order 3) for error estimation.
    pub b_hat: [f64; 6],
    /// Diagonal element (constant across implicit stages).
    pub gamma: f64,
    /// Order of the embedded method (for error estimation).
    pub order_err: usize,
}

/// ESDIRK4(3)6L[2]SA tableau (Kennedy–Carpenter), zero-cost static.
///
/// Private: `esdirk43_tableau()` is the single route to it, so there is no
/// second name for the same bytes to drift against.
static ESDIRK43_TABLEAU: EsdirkTableau = EsdirkTableau {
    stages: 6,
    gamma: 1.0 / 4.0,
    order_err: 3,
    a: [
        // Stage 0: explicit
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        // Stage 1
        [1.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
        // Stage 2
        [
            8_611.0 / 62_500.0,
            -1_743.0 / 31_250.0,
            1.0 / 4.0,
            0.0,
            0.0,
            0.0,
        ],
        // Stage 3
        [
            5_012_029.0 / 34_652_500.0,
            -654_441.0 / 2_922_500.0,
            174_375.0 / 388_108.0,
            1.0 / 4.0,
            0.0,
            0.0,
        ],
        // Stage 4
        [
            15_267_082_809.0 / 155_376_265_600.0,
            -71_443_401.0 / 120_774_400.0,
            730_878_875.0 / 902_184_768.0,
            2_285_395.0 / 8_070_912.0,
            1.0 / 4.0,
            0.0,
        ],
        // Stage 5 (stiffly accurate: a[5] == b)
        [
            82_889.0 / 524_892.0,
            0.0,
            15_625.0 / 83_664.0,
            69_875.0 / 102_672.0,
            -2_260.0 / 8_211.0,
            1.0 / 4.0,
        ],
    ],
    c: [0.0, 1.0 / 2.0, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0],
    b: [
        82_889.0 / 524_892.0,
        0.0,
        15_625.0 / 83_664.0,
        69_875.0 / 102_672.0,
        -2_260.0 / 8_211.0,
        1.0 / 4.0,
    ],
    b_hat: [
        4_586_570_599.0 / 29_645_900_160.0,
        0.0,
        178_811_875.0 / 945_068_544.0,
        814_220_225.0 / 1_159_782_912.0,
        -3_700_637.0 / 11_593_932.0,
        61_727.0 / 225_920.0,
    ],
};

/// Return the ESDIRK4(3)6L[2]SA tableau (Kennedy–Carpenter).
#[inline]
#[must_use]
pub fn esdirk43_tableau() -> &'static EsdirkTableau {
    &ESDIRK43_TABLEAU
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tableau_consistency() {
        let tab = esdirk43_tableau();
        let [first_row, _, _, _, _, last_row] = &tab.a;
        let [first_node, _, _, _, _, last_node] = tab.c;

        // Check dimensions
        assert_eq!(tab.stages, 6);
        assert_eq!(tab.order_err, 3);
        assert!((tab.gamma - 0.25).abs() < 1e-15);

        // Row sums of A should equal c
        for (row_index, (row, &node)) in tab.a.iter().zip(tab.c.iter()).enumerate() {
            let row_sum: f64 = row.iter().sum();
            let difference = (row_sum - node).abs();
            assert!(
                difference < 1e-14,
                "Row {row_index} sum {row_sum} != c {node} (diff {difference:e})",
            );
        }

        // b weights sum to 1
        let b_sum: f64 = tab.b.iter().sum();
        assert!((b_sum - 1.0).abs() < 1e-14, "b sum {b_sum} != 1.0");

        // b_hat weights sum to 1
        let bhat_sum: f64 = tab.b_hat.iter().sum();
        assert!(
            (bhat_sum - 1.0).abs() < 1e-14,
            "b_hat sum {bhat_sum} != 1.0"
        );

        // Stiffly accurate: last row of A equals b
        for (j, (&last_coefficient, &weight)) in last_row.iter().zip(tab.b.iter()).enumerate() {
            assert!(
                (last_coefficient - weight).abs() < 1e-15,
                "a[5][{j}] = {last_coefficient} != b[{j}] = {weight}",
            );
        }

        // Stage 0 is explicit: a[0] is all zeros
        for (j, &coefficient) in first_row.iter().enumerate() {
            assert_eq!(
                coefficient.to_bits(),
                0.0_f64.to_bits(),
                "a[0][{j}] should be 0"
            );
        }

        // Diagonal is gamma for stages 1..5
        for (i, row) in tab.a.iter().enumerate().skip(1) {
            let diagonal = row.get(i).copied();
            assert!(
                diagonal.is_some_and(|coefficient| (coefficient - tab.gamma).abs() < 1e-15),
                "a[{i}][{i}] = {diagonal:?} != gamma = {}",
                tab.gamma
            );
        }

        // Upper triangle is zero (lower-triangular)
        for (i, row) in tab.a.iter().enumerate() {
            for (j, &coefficient) in row.iter().enumerate().skip(i + 1) {
                assert_eq!(
                    coefficient.to_bits(),
                    0.0_f64.to_bits(),
                    "a[{i}][{j}] should be 0 (upper triangle)"
                );
            }
        }

        // c[0] = 0, c[5] = 1
        assert_eq!(first_node.to_bits(), 0.0_f64.to_bits());
        assert!((last_node - 1.0).abs() < 1e-15);
    }
}
