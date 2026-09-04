use super::Tableau;

// Coefficients from Tsitouras 5(4) (rkt54) as used in RKLIB.
// Source: rklib_variable_steps.f90 (Tsitouras 5(4) Embedded Runge-Kutta method)

const A2: f64 = 0.161;
const A3: f64 = 0.327;
const A4: f64 = 0.9;
const A5: f64 = 0.980_025_540_904_509_7;
const A6: f64 = 1.0;

const B21: f64 = 0.161;
const B31: f64 = -0.008_480_655_492_356_989;
const B32: f64 = 0.335_480_655_492_357;
const B41: f64 = 2.897_153_057_105_493_5;
const B42: f64 = -6.359_448_489_975_075;
const B43: f64 = 4.362_295_432_869_582;
const B51: f64 = 5.325_864_828_439_257;
const B52: f64 = -11.748_883_564_062_828;
const B53: f64 = 7.495_539_342_889_836_5;
const B54: f64 = -0.092_495_066_361_755_25;
const B61: f64 = 5.861_455_442_946_42;
const B62: f64 = -12.920_969_317_847_11;
const B63: f64 = 8.159_367_898_576_159;
const B64: f64 = -0.071_584_973_281_401;
const B65: f64 = -0.028_269_050_394_068_38;

const C1: f64 = 0.096_460_766_818_065_23;
const C2: f64 = 0.01;
const C3: f64 = 0.479_889_650_414_499_6;
const C4: f64 = 1.379_008_574_103_742;
const C5: f64 = -3.290_069_515_436_081;
const C6: f64 = 2.324_710_524_099_774;

const D1: f64 = 0.094_680_755_765_839_46;
const D2: f64 = 0.009_183_565_540_343_253;
const D3: f64 = 0.487_770_528_424_761_6;
const D4: f64 = 1.234_297_566_930_479;
const D5: f64 = -2.707_712_349_983_525_5;
const D6: f64 = 1.866_628_418_170_587;
const D7: f64 = 1.0 / 66.0;

const C: [f64; 7] = [0.0, A2, A3, A4, A5, A6, 1.0];

const A1: [f64; 0] = [];
const A2_ROW: [f64; 1] = [B21];
const A3_ROW: [f64; 2] = [B31, B32];
const A4_ROW: [f64; 3] = [B41, B42, B43];
const A5_ROW: [f64; 4] = [B51, B52, B53, B54];
const A6_ROW: [f64; 5] = [B61, B62, B63, B64, B65];
const A7_ROW: [f64; 6] = [C1, C2, C3, C4, C5, C6];

const A: [&[f64]; 7] = [&A1, &A2_ROW, &A3_ROW, &A4_ROW, &A5_ROW, &A6_ROW, &A7_ROW];

const B: [f64; 7] = [C1, C2, C3, C4, C5, C6, 0.0];
const ERR: [f64; 7] = [
    C1 - D1,
    C2 - D2,
    C3 - D3,
    C4 - D4,
    C5 - D5,
    C6 - D6,
    0.0 - D7,
];

static TABLEAU: Tableau = Tableau {
    stages: 7,
    c: &C,
    a: &A,
    b: &B,
    b_hat: None,
    err: Some(&ERR),
    err3: None,
    err3_min_eps: None,
    order: 5,
    order_err: 4,
    fsal: true,
};

#[must_use]
pub fn tableau() -> &'static Tableau {
    &TABLEAU
}
