use super::Tableau;

// Coefficients extracted from satkit RKV98 efficient table.
#[expect(clippy::excessive_precision)]
mod coeffs {
    include!("rkv98_coeffs.rs");
}

const A_ROWS: [&[f64]; 26] = [
    &coeffs::A[0],
    &coeffs::A[1],
    &coeffs::A[2],
    &coeffs::A[3],
    &coeffs::A[4],
    &coeffs::A[5],
    &coeffs::A[6],
    &coeffs::A[7],
    &coeffs::A[8],
    &coeffs::A[9],
    &coeffs::A[10],
    &coeffs::A[11],
    &coeffs::A[12],
    &coeffs::A[13],
    &coeffs::A[14],
    &coeffs::A[15],
    &coeffs::A[16],
    &coeffs::A[17],
    &coeffs::A[18],
    &coeffs::A[19],
    &coeffs::A[20],
    &coeffs::A[21],
    &coeffs::A[22],
    &coeffs::A[23],
    &coeffs::A[24],
    &coeffs::A[25],
];

static TABLEAU: Tableau = Tableau {
    stages: 26,
    c: &coeffs::C,
    a: &A_ROWS,
    b: &coeffs::B,
    b_hat: Some(&coeffs::BHAT),
    err: None,
    err3: None,
    err3_min_eps: None,
    order: 9,
    order_err: 8,
    fsal: false,
};

#[must_use]
pub fn tableau() -> &'static Tableau {
    &TABLEAU
}
