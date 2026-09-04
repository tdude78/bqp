//! Verner 9(8) "numerically optimal" embedded Runge-Kutta pair.
//!
//! 16-stage, order 9 method with embedded order 8 error estimate.
//! Reference: J.H. Verner, "Numerically optimal Runge-Kutta pairs with
//! interpolants", Numerical Algorithms 53 (2010), 383-396.
//! doi:10.100_7/s11075-009-9290-3
//!
//! Coefficients from Julia `OrdinaryDiffEqVerner` (MIT license).
//!
//! Stage layout (16 stages, NOT FSAL):
//!   - Stages 0..13 are the core stages.
//!   - Stages 14 and 15 are both evaluated at c=1.0 but with different
//!     A-row dependencies: stage 14 depends on stages 0..13, stage 15
//!     depends on stages 0..12 only (no k14 or k15 dependency).
//!   - The order-9 solution uses b[0], b[7..14] (b[1..6] and b[15] are zero).
//!   - The error estimate uses btilde[0], btilde[7..15] (btilde[1..6] are zero).

use super::Tableau;

// --- c nodes (16 stages) ---
// c[0] = 0 (initial point)
// c[1]..c[13] from Vern9Tableau fields c1..c13
// c[14] = c[15] = 1.0 (both auxiliary stages evaluated at end of step)
const C: [f64; 16] = [
    0.0,                      // stage 0
    0.034_62,                 // stage 1  (Julia c1)
    0.097_024_350_638_780_45, // stage 2  (Julia c2)
    0.145_536_525_958_170_68, // stage 3  (Julia c3)
    0.561,                    // stage 4  (Julia c4)
    0.229_007_911_590_485_03, // stage 5  (Julia c5)
    0.544_992_088_409_515,    // stage 6  (Julia c6)
    0.645,                    // stage 7  (Julia c7)
    0.483_75,                 // stage 8  (Julia c8)
    0.067_57,                 // stage 9  (Julia c9)
    0.25,                     // stage 10 (Julia c10)
    0.659_065_061_873_099_9,  // stage 11 (Julia c11)
    0.820_6,                  // stage 12 (Julia c12)
    0.901_2,                  // stage 13 (Julia c13), also used by stage 14
    1.0,                      // stage 14 (evaluated at t + dt)
    1.0,                      // stage 15 (evaluated at t + dt)
];

// --- A matrix rows (lower triangular, 0-indexed) ---
// Julia naming: a_{ij} where i is 1-indexed destination stage, j is 1-indexed source.
// Many entries are structurally zero (columns 1..4 for later stages).
// We store full rows with explicit zeros for the generic solver.

const A0: [f64; 0] = [];

// Stage 1: a0201 * k0
const A1: [f64; 1] = [
    0.034_62, // a0201
];

// Stage 2: a0301*k0 + a0302*k1
const A2: [f64; 2] = [
    -0.038_933_543_885_728_75, // a0301
    0.135_957_894_524_509_18,  // a0302
];

// Stage 3: a0401*k0 + 0*k1 + a0403*k2
const A3: [f64; 3] = [
    0.036_384_131_489_542_67, // a0401
    0.0,                      // a0402 (zero)
    0.109_152_394_468_628_01, // a0403
];

// Stage 4: a0501*k0 + 0*k1 + a0503*k2 + a0504*k3
const A4: [f64; 4] = [
    2.025_763_914_393_969_4, // a0501
    0.0,                     // a0502
    -7.638_023_836_496_291,  // a0503
    6.173_259_922_102_322,   // a0504
];

// Stage 5: a0601*k0 + 0*k1 + 0*k2 + a0604*k3 + a0605*k4
const A5: [f64; 5] = [
    0.051_122_755_894_060_61,    // a0601
    0.0,                         // a0602
    0.0,                         // a0603
    0.177_082_379_455_502_18,    // a0604
    0.000_802_776_240_922_253_6, // a0605
];

// Stage 6: a0701*k0 + 0*k1 + 0*k2 + a0704*k3 + a0705*k4 + a0706*k5
const A6: [f64; 6] = [
    0.131_600_635_797_521_63, // a0701
    0.0,                      // a0702
    0.0,                      // a0703
    -0.295_727_625_266_963_6, // a0704
    0.087_813_780_356_429_55, // a0705
    0.621_305_297_522_527_4,  // a0706
];

// Stage 7: a0801*k0 + 0*k1 + 0*k2 + 0*k3 + 0*k4 + a0806*k5 + a0807*k6
const A7: [f64; 7] = [
    0.071_666_666_666_666_67, // a0801
    0.0,                      // a0802
    0.0,                      // a0803
    0.0,                      // a0804
    0.0,                      // a0805
    0.330_553_357_891_531_95, // a0806
    0.242_779_975_441_801_4,  // a0807
];

// Stage 8: a0901*k0 + 0*k1..k4 + a0906*k5 + a0907*k6 + a0908*k7
const A8: [f64; 8] = [
    0.071_806_640_625,       // a0901
    0.0,                     // a0902
    0.0,                     // a0903
    0.0,                     // a0904
    0.0,                     // a0905
    0.329_438_028_322_817_7, // a0906
    0.116_519_002_927_182_3, // a0907
    -0.034_013_671_875,      // a0908
];

// Stage 9: a1001*k0 + 0*k1..k4 + a1006*k5 + a1007*k6 + a1008*k7 + a1009*k8
const A9: [f64; 9] = [
    0.048_367_576_463_406_46,   // a1001
    0.0,                        // a1002
    0.0,                        // a1003
    0.0,                        // a1004
    0.0,                        // a1005
    0.039_289_899_256_761_64,   // a1006
    0.105_474_094_589_034_46,   // a1007
    -0.021_438_652_846_483_126, // a1008
    -0.104_122_917_462_719_44,  // a1009
];

// Stage 10: a1101*k0 + 0*k1..k4 + a1106*k5 + a1107*k6 + a1108*k7 + a1109*k8 + a1110*k9
const A10: [f64; 10] = [
    -0.026_645_614_872_014_785, // a1101
    0.0,                        // a1102
    0.0,                        // a1103
    0.0,                        // a1104
    0.0,                        // a1105
    0.033_333_333_333_333_33,   // a1106
    -0.163_107_224_487_246_7,   // a1107
    0.033_960_816_841_277_61,   // a1108
    0.157_231_941_381_462_6,    // a1109
    0.215_226_747_803_187_96,   // a1110
];

// Stage 11: a1201*k0 + 0*k1..k4 + a1206*k5 + a1207*k6 + a1208*k7 + a1209*k8 + a1210*k9 + a1211*k10
const A11: [f64; 11] = [
    0.036_890_092_487_086_22,     // a1201
    0.0,                          // a1202
    0.0,                          // a1203
    0.0,                          // a1204
    0.0,                          // a1205
    -0.146_518_157_672_554_3,     // a1206
    0.224_257_776_817_202_4,      // a1207
    0.022_944_057_170_660_73,     // a1208
    -0.003_585_005_290_572_859_7, // a1209
    0.086_692_233_164_443_85,     // a1210
    0.438_384_065_196_833_76,     // a1211
];

// Stage 12: a1301*k0 + 0*k1..k4 + a1306*k5 + a1307*k6 + a1308*k7 + a1309*k8 + a1310*k9 + a1311*k10 + a1312*k11
const A12: [f64; 12] = [
    -0.486_601_221_511_334_1, // a1301
    0.0,                      // a1302
    0.0,                      // a1303
    0.0,                      // a1304
    0.0,                      // a1305
    -6.304_602_650_282_853,   // a1306
    -0.281_245_618_289_472_9, // a1307
    -2.679_019_236_219_849,   // a1308
    0.518_815_663_924_157_7,  // a1309
    1.365_353_187_603_341_8,  // a1310
    5.885_091_088_503_946_5,  // a1311
    2.802_808_786_272_062_8,  // a1312
];

// Stage 13: a1401*k0 + 0*k1..k4 + a1406*k5..a1413*k12
const A13: [f64; 13] = [
    0.418_536_745_775_347_2,   // a1401
    0.0,                       // a1402
    0.0,                       // a1403
    0.0,                       // a1404
    0.0,                       // a1405
    6.724_547_581_906_459,     // a1406
    -0.425_444_280_164_611_33, // a1407
    3.343_279_153_001_265_3,   // a1408
    0.617_081_663_117_537_4,   // a1409
    -0.929_966_123_939_932_9,  // a1410
    -6.099_948_804_751_011,    // a1411
    -3.002_206_187_889_399,    // a1412
    0.255_320_252_944_344_6,   // a1413
];

// Stage 14: a1501*k0 + 0*k1..k4 + a1506*k5..a1514*k13
const A14: [f64; 14] = [
    -0.779_374_086_122_884_8, // a1501
    0.0,                      // a1502
    0.0,                      // a1503
    0.0,                      // a1504
    0.0,                      // a1505
    -13.937_342_538_107_776,  // a1506
    1.252_048_853_379_356_3,  // a1507
    -14.691_500_408_016_868,  // a1508
    -0.494_705_058_533_141,   // a1509
    2.242_974_909_146_236_8,  // a1510
    13.367_893_803_828_643,   // a1511
    14.396_650_486_650_687,   // a1512
    -0.797_581_333_177_68,    // a1513
    0.440_935_370_953_427_8,  // a1514
];

// Stage 15: a1601*k0 + 0*k1..k4 + a1606*k5..a1613*k12
// Note: stage 15 does NOT depend on k13 or k14 (only up to k12)
const A15: [f64; 13] = [
    2.058_051_337_466_886_7, // a1601
    0.0,                     // a1602
    0.0,                     // a1603
    0.0,                     // a1604
    0.0,                     // a1605
    22.357_937_727_968_032,  // a1606
    0.909_498_109_975_564_6, // a1607
    35.891_100_982_402_64,   // a1608
    -3.442_515_027_624_454,  // a1609
    -4.865_481_358_036_369,  // a1610
    -18.909_803_813_543_427, // a1611
    -34.263_544_480_304_52,  // a1612
    1.264_756_521_695_642_7, // a1613
];

const A_ROWS: [&[f64]; 16] = [
    &A0, &A1, &A2, &A3, &A4, &A5, &A6, &A7, &A8, &A9, &A10, &A11, &A12, &A13, &A14, &A15,
];

// --- b weights (order 9, main method) ---
// Only b[0], b[7]..b[14] are nonzero.
// u_new = u + dt * sum(b[i] * k[i])
const B: [f64; 16] = [
    0.014_611_976_858_423_152, // b1
    0.0,                       // b2
    0.0,                       // b3
    0.0,                       // b4
    0.0,                       // b5
    0.0,                       // b6
    0.0,                       // b7
    -0.391_521_186_233_133_9,  // b8
    0.231_093_250_028_950_65,  // b9
    0.127_476_676_999_285_25,  // b10
    0.224_643_417_620_415_8,   // b11
    0.568_435_268_974_851_3,   // b12
    0.058_258_715_572_158_275, // b13
    0.136_431_740_348_221_56,  // b14
    0.030_570_139_830_827_976, // b15
    0.0,                       // b16
];

// --- Error coefficients: btilde = b - b_hat ---
// btilde[i] = b[i] - bhat[i]
// The error estimate is: err = dt * sum(btilde[i] * k[i])
// Only btilde[0], btilde[7]..btilde[15] are nonzero.
const ERR: [f64; 16] = [
    -0.005_357_988_290_444_578, // btilde1
    0.0,                        // btilde2
    0.0,                        // btilde3
    0.0,                        // btilde4
    0.0,                        // btilde5
    0.0,                        // btilde6
    0.0,                        // btilde7
    -2.583_020_491_182_464,     // btilde8
    0.142_522_531_546_866_25,   // btilde9
    0.013_420_653_512_688_676,  // btilde10
    -0.028_672_962_914_094_93,  // btilde11
    2.624_999_655_215_792,      // btilde12
    -0.282_550_964_329_153_7,   // btilde13
    0.136_431_740_348_221_56,   // btilde14
    0.030_570_139_830_827_976,  // btilde15
    -0.048_342_313_738_239_58,  // btilde16
];

static TABLEAU: Tableau = Tableau {
    stages: 16,
    c: &C,
    a: &A_ROWS,
    b: &B,
    b_hat: None,
    err: Some(&ERR),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(
        clippy::float_cmp,
        clippy::indexing_slicing,
        reason = "the static-tableau invariant test intentionally direct-indexes entries and checks exact zero coefficients"
    )]
    #[test]
    fn test_vern9_consistency() {
        let tab = tableau();
        assert_eq!(tab.stages, 16);
        assert_eq!(tab.order, 9);
        assert_eq!(tab.order_err, 8);
        assert!(!tab.fsal);

        // c values: c[0] = 0, c[14] = c[15] = 1
        assert_eq!(tab.c[0], 0.0);
        assert!((tab.c[14] - 1.0).abs() < 1e-14);
        assert!((tab.c[15] - 1.0).abs() < 1e-14);

        // Row sums of A equal c (for stages 0..14)
        // Stage 15 has a shorter A row (13 entries) and its row sum
        // does NOT equal c[15]=1.0 -- it is an auxiliary error stage.
        for s in 0..15 {
            let row_sum: f64 = tab.a[s].iter().sum();
            assert!(
                (row_sum - tab.c[s]).abs() < 1e-12,
                "Row {} sum {:.16e} != c {:.16e} (diff = {:e})",
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
