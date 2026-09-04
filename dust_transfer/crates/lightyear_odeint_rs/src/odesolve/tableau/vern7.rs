//! Verner 7(6) "numerically optimal" embedded Runge-Kutta pair.
//!
//! 10-stage, order 7 method with embedded order 6 error estimate.
//! Reference: J.H. Verner, "Numerically optimal Runge–Kutta pairs with
//! interpolants", Numerical Algorithms 53 (2010), 383–396.
//! doi:10.100_7/s11075-009-9290-3
//!
//! Coefficients from Julia `OrdinaryDiffEqVerner` (MIT license).

use super::Tableau;

// --- c nodes (10 stages) ---
const C: [f64; 10] = [
    0.0,
    0.005,
    0.108_888_888_888_888_88,
    0.163_333_333_333_333_33,
    0.455_5,
    0.609_509_448_997_838_1,
    0.884,
    0.925,
    1.0,
    1.0,
];

// --- A matrix rows (lower triangular) ---
const A0: [f64; 0] = [];
const A1: [f64; 1] = [0.005];
const A2: [f64; 2] = [-1.076_790_123_456_79, 1.185_679_012_345_679];
const A3: [f64; 3] = [0.040_833_333_333_333_33, 0.0, 0.122_5];
const A4: [f64; 4] = [
    0.638_913_923_625_572_6,
    0.0,
    -2.455_672_638_223_657,
    2.272_258_714_598_084,
];
const A5: [f64; 5] = [
    -2.661_577_375_018_757_2,
    0.0,
    10.804_513_886_456_137,
    -8.353_914_657_396_2,
    0.820_487_594_956_657,
];
const A6: [f64; 6] = [
    6.067_741_434_696_772,
    0.0,
    -24.711_273_635_911_088,
    20.427_517_930_788_895,
    -1.906_157_978_816_647_2,
    1.006_172_249_242_068,
];
const A7: [f64; 7] = [
    12.054_670_076_253_203,
    0.0,
    -49.754_784_950_468_99,
    41.142_888_638_604_674,
    -4.461_760_149_974_004,
    2.042_334_822_239_175,
    -0.098_348_436_654_061_07,
];
const A8: [f64; 8] = [
    10.138_146_522_881_808,
    0.0,
    -42.641_136_031_717_5,
    35.763_840_039_922_57,
    -4.348_022_840_392_907_5,
    2.009_862_268_377_035_7,
    0.348_749_046_033_827_2,
    -0.271_439_005_104_831_27,
];
// Stage 10 (error estimation only, depends on stages 0–6)
const A9: [f64; 7] = [
    -45.030_072_034_298_676,
    0.0,
    187.327_243_765_458_9,
    -154.028_823_693_501_86,
    18.564_653_063_475_36,
    -7.141_809_679_295_079,
    1.308_808_578_161_378_7,
];

const A_ROWS: [&[f64]; 10] = [&A0, &A1, &A2, &A3, &A4, &A5, &A6, &A7, &A8, &A9];

// --- b weights (order 7, main method) ---
const B: [f64; 10] = [
    0.047_155_618_486_272_22,
    0.0,
    0.0,
    0.257_505_642_984_341_53,
    0.262_166_539_774_126_24,
    0.152_160_926_567_385_58,
    0.493_996_917_003_248_5,
    -0.294_303_117_140_325_03,
    0.081_317_472_324_951_11,
    0.0,
];

// --- Error coefficients: btilde = b - b_hat (Verner 2010) ---
const ERR: [f64; 10] = [
    0.002_547_011_879_931_045,
    0.0,
    0.0,
    -0.009_658_394_872_795_75,
    0.042_064_709_756_396_91,
    -0.066_682_243_746_930_1,
    0.265_009_746_462_128_1,
    -0.294_303_117_140_325_03,
    0.081_317_472_324_951_11,
    -0.020_295_184_663_356_28,
];

static TABLEAU: Tableau = Tableau {
    stages: 10,
    c: &C,
    a: &A_ROWS,
    b: &B,
    b_hat: None,
    err: Some(&ERR),
    err3: None,
    err3_min_eps: None,
    order: 7,
    order_err: 6,
    fsal: false,
};

#[must_use]
pub fn tableau() -> &'static Tableau {
    &TABLEAU
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(
        clippy::indexing_slicing,
        reason = "the static-tableau authority test intentionally direct-indexes the pinned stage"
    )]
    #[test]
    fn stage_one_is_static_density_carry_authority() {
        let tab = &TABLEAU;
        let stage_one_bits = 0.005_f64.to_bits();
        assert_eq!(tab.c[1].to_bits(), stage_one_bits);
        assert_eq!(tab.a[1].len(), 1);
        assert_eq!(tab.a[1][0].to_bits(), stage_one_bits);
        assert_eq!(tab.b[1].to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            tab.err.expect("Vern7 error weights")[1].to_bits(),
            0.0_f64.to_bits()
        );
        assert!(tab.b_hat.is_none());
        assert!(tab.err3.is_none());
    }

    #[expect(
        clippy::float_cmp,
        clippy::indexing_slicing,
        reason = "the static-tableau invariant test intentionally direct-indexes entries and checks exact zero coefficients"
    )]
    #[test]
    fn test_vern7_consistency() {
        let tab = tableau();
        assert_eq!(tab.stages, 10);
        assert_eq!(tab.order, 7);
        assert_eq!(tab.order_err, 6);
        assert!(!tab.fsal);

        // c values: c[0] = 0, c[9] = 1
        assert_eq!(tab.c[0], 0.0);
        assert!((tab.c[9] - 1.0).abs() < 1e-14);

        // Row sums of A equal c
        for s in 0..10 {
            let row_sum: f64 = tab.a[s].iter().sum();
            assert!(
                (row_sum - tab.c[s]).abs() < 1e-12,
                "Row {} sum {} != c {} (diff = {:e})",
                s,
                row_sum,
                tab.c[s],
                (row_sum - tab.c[s]).abs()
            );
        }

        // b weights sum to 1
        let b_sum: f64 = tab.b.iter().sum();
        assert!(
            (b_sum - 1.0).abs() < 1e-14,
            "b sum {} != 1.0 (diff = {:e})",
            b_sum,
            (b_sum - 1.0).abs()
        );

        // err weights should not all be zero
        let err_sum: f64 = tab.err.unwrap().iter().map(|e| e.abs()).sum();
        assert!(err_sum > 0.0, "err weights are all zero");
    }
}
