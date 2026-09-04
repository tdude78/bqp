use super::Tableau;

// Coefficients extracted from ode_solvers dopri853 tableau.
//
// PERF: The canonical DOP853 tableau has 16 stages, but stages 13-16
// (indices 12-15) have ALL-ZERO B and E coefficients — they contribute
// nothing to the 8th-order solution or the embedded error estimate.
// We truncate to 12 active stages, eliminating 4 wasted RHS evaluations
// per step (~25% ODE speedup) with zero output change.
#[expect(clippy::excessive_precision)]
mod coeffs {
    include!("dop853_coeffs.rs");
}

const A0: [f64; 0] = [];
const A_ROWS: [&[f64]; 12] = [
    &A0,
    &coeffs::A.0,
    &coeffs::A.1,
    &coeffs::A.2,
    &coeffs::A.3,
    &coeffs::A.4,
    &coeffs::A.5,
    &coeffs::A.6,
    &coeffs::A.7,
    &coeffs::A.8,
    &coeffs::A.9,
    &coeffs::A.10,
];

// C (time fractions) truncated to 12 active stages.
const C_ACTIVE: [f64; 12] = [
    coeffs::C[0],
    coeffs::C[1],
    coeffs::C[2],
    coeffs::C[3],
    coeffs::C[4],
    coeffs::C[5],
    coeffs::C[6],
    coeffs::C[7],
    coeffs::C[8],
    coeffs::C[9],
    coeffs::C[10],
    coeffs::C[11],
];

// E (error coefficients) truncated to 12 active stages.
// Original E[12..15] are all 0.0 — verified in dop853_coeffs.rs.
const E_ACTIVE: [f64; 12] = [
    coeffs::E[0],
    coeffs::E[1],
    coeffs::E[2],
    coeffs::E[3],
    coeffs::E[4],
    coeffs::E[5],
    coeffs::E[6],
    coeffs::E[7],
    coeffs::E[8],
    coeffs::E[9],
    coeffs::E[10],
    coeffs::E[11],
];

static TABLEAU: Tableau = Tableau {
    stages: 12,
    c: &C_ACTIVE,
    a: &A_ROWS,
    b: &coeffs::B,
    b_hat: None,
    err: Some(&E_ACTIVE),
    err3: Some(&coeffs::E3),
    // Blend the third-order estimate into the error norm only at loose
    // tolerances.
    //
    // Hairer's DOP853 combines the fifth- and third-order embedded estimates as
    // `sqrt(err5^2 + 0.01*err3^2)`. The combination is norm-independent — it
    // mixes the two scalar norms, not per-component values — so it applies to
    // both the WRMS and the max-norm forms.
    //
    // At tight tolerances E3's sparse coefficients produce a noisy estimate
    // that destabilises the step-size controller; dop853 breaks at eps=5e-8
    // without this gate. 1e-6 is the measured floor below which the E3 signal
    // stops being informative for this tableau.
    //
    // This threshold belongs to DOP853, not to the solver. E3 is the only
    // third-order embedded estimate in the tree, so this is the only tableau
    // that sets it — see `Tableau::err3_min_eps`.
    err3_min_eps: Some(1e-6),
    order: 8,
    order_err: 5,
    fsal: false,
};

#[must_use]
pub fn tableau() -> &'static Tableau {
    &TABLEAU
}
