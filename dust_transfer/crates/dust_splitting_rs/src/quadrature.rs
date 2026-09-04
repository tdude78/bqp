//! Gauss-Hermite quadrature nodes and weights.
//!
//! Pre-tabulated values for standard normal distribution (adapted from physicists' Hermite).
//! Supports n = 1 to 7 components.

/// Gauss-Hermite nodes and weights for standard normal distribution.
#[derive(Clone, Copy, Debug)]
pub struct HermiteGauss {
    /// Nodes (z values).
    pub nodes: &'static [f64],
    /// Weights (w values).
    pub weights: &'static [f64],
}

/// Get Gauss-Hermite quadrature for n points adapted to N(0,1).
///
/// Returns `None` unless `n` is in the supported range `1..=7`.
#[must_use]
pub fn hermgauss_std_normal(n: usize) -> Option<HermiteGauss> {
    match n {
        1 => Some(HermiteGauss {
            nodes: &NODES_1,
            weights: &WEIGHTS_1,
        }),
        2 => Some(HermiteGauss {
            nodes: &NODES_2,
            weights: &WEIGHTS_2,
        }),
        3 => Some(HermiteGauss {
            nodes: &NODES_3,
            weights: &WEIGHTS_3,
        }),
        4 => Some(HermiteGauss {
            nodes: &NODES_4,
            weights: &WEIGHTS_4,
        }),
        5 => Some(HermiteGauss {
            nodes: &NODES_5,
            weights: &WEIGHTS_5,
        }),
        6 => Some(HermiteGauss {
            nodes: &NODES_6,
            weights: &WEIGHTS_6,
        }),
        7 => Some(HermiteGauss {
            nodes: &NODES_7,
            weights: &WEIGHTS_7,
        }),
        _ => None,
    }
}

// Pre-tabulated Gauss-Hermite nodes/weights adapted to N(0,1)
// These match the C++ implementation exactly.

static NODES_1: [f64; 1] = [0.0];
static WEIGHTS_1: [f64; 1] = [1.0];

#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static NODES_2: [f64; 2] = [-1.000000000000000000e+00, 1.000000000000000000e+00];
#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static WEIGHTS_2: [f64; 2] = [5.000000000000000000e-01, 5.000000000000000000e-01];

#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static NODES_3: [f64; 3] = [
    -1.732050807568877193e+00,
    0.000000000000000000e+00,
    1.732050807568877193e+00,
];
#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static WEIGHTS_3: [f64; 3] = [
    1.666666666666666574e-01,
    6.666666666666666297e-01,
    1.666666666666666574e-01,
];

#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static NODES_4: [f64; 4] = [
    -2.334414218338977332e+00,
    -7.419637843027259150e-01,
    7.419637843027259150e-01,
    2.334414218338977332e+00,
];
#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static WEIGHTS_4: [f64; 4] = [
    4.587585476806853302e-02,
    4.541241452319315086e-01,
    4.541241452319315086e-01,
    4.587585476806853302e-02,
];

#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static NODES_5: [f64; 5] = [
    -2.856970013872805580e+00,
    -1.355626179974265932e+00,
    0.000000000000000000e+00,
    1.355626179974265932e+00,
    2.856970013872805580e+00,
];
#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static WEIGHTS_5: [f64; 5] = [
    1.125741132772069449e-02,
    2.220759220056126304e-01,
    5.333333333333334370e-01,
    2.220759220056126304e-01,
    1.125741132772069449e-02,
];

#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static NODES_6: [f64; 6] = [
    -3.324257433552119334e+00,
    -1.889175877753710875e+00,
    -6.167065901925942173e-01,
    6.167065901925942173e-01,
    1.889175877753710875e+00,
    3.324257433552119334e+00,
];
#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static WEIGHTS_6: [f64; 6] = [
    2.555784402056241345e-03,
    8.861574604191445326e-02,
    4.088284695560292503e-01,
    4.088284695560292503e-01,
    8.861574604191445326e-02,
    2.555784402056241345e-03,
];

#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static NODES_7: [f64; 7] = [
    -3.750439717725742472e+00,
    -2.366759410734541547e+00,
    -1.154405394739968171e+00,
    0.000000000000000000e+00,
    1.154405394739968171e+00,
    2.366759410734541547e+00,
    3.750439717725742472e+00,
];
#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "published quadrature constants retain source precision and spelling"
)]
static WEIGHTS_7: [f64; 7] = [
    5.482688559722184284e-04,
    3.075712396758651518e-02,
    2.401231786050126160e-01,
    4.571428571428572396e-01,
    2.401231786050126160e-01,
    3.075712396758651518e-02,
    5.482688559722184284e-04,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hermgauss_weights_sum_to_one() {
        for n in 1..=7 {
            let hg = hermgauss_std_normal(n).expect("supported quadrature order");
            let sum: f64 = hg.weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "n={n}: weights sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_hermgauss_nodes_symmetric() {
        for n in 1..=7 {
            let hg = hermgauss_std_normal(n).expect("supported quadrature order");
            for (&left, &right) in hg
                .nodes
                .iter()
                .zip(hg.nodes.iter().rev())
                .take(hg.nodes.len() / 2)
            {
                assert!((left + right).abs() < 1e-10, "n={n}: nodes not symmetric");
            }
        }
    }

    #[test]
    fn test_three_point_rule_recovers_standard_normal_moments() {
        let hg = hermgauss_std_normal(3).expect("supported quadrature order");
        let expected_weights = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
        for (actual, expected) in hg.weights.iter().zip(expected_weights) {
            assert!((actual - expected).abs() < 1.0e-15);
        }

        let mean: f64 = hg
            .nodes
            .iter()
            .zip(hg.weights.iter())
            .map(|(node, weight)| node * weight)
            .sum();
        let variance: f64 = hg
            .nodes
            .iter()
            .zip(hg.weights.iter())
            .map(|(node, weight)| weight * (node - mean).powi(2))
            .sum();

        assert!(mean.abs() < 1.0e-15);
        assert!((variance - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn test_hermgauss_weights_symmetric() {
        for n in 1..=7 {
            let hg = hermgauss_std_normal(n).expect("supported quadrature order");
            for (&left, &right) in hg
                .weights
                .iter()
                .zip(hg.weights.iter().rev())
                .take(hg.nodes.len() / 2)
            {
                assert!((left - right).abs() < 1e-10, "n={n}: weights not symmetric");
            }
        }
    }

    #[test]
    fn test_hermgauss_invalid_n_is_rejected() {
        assert!(hermgauss_std_normal(8).is_none());
    }
}
