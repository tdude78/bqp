use dust_estimates_rs::fraction_finalize::{
    fraction_grid_finalize_mf_row, LABEL_OTHER, LABEL_PHYSICS_LIMITED, LABEL_SAFE_BY_DEFAULT,
    REASON_DETERMINISTIC_MASS_INVALID, REASON_OK, REASON_PROB_MASS_INVALID, REASON_SAFE_BY_DEFAULT,
};

// Private indices into the frozen REASON_CODES table (fraction_finalize.rs;
// mirrored in nd_pipeline::physics::reason). Hand-copied here because the
// crate does not export them; the table is frozen, so a drift is a real
// contract break this test should catch.
const REASON_DETERMINISTIC_MASS_PHYSICS_LIMITED: i32 = 2;
const REASON_PROB_MASS_FLOOR_HARD_LIMIT: i32 = 4;
const REASON_PROB_MASS_HARD_LIMIT: i32 = 5;

#[test]
fn every_mf_outcome_preserves_one_executed_dv() {
    // One case per reason code `fraction_grid_finalize_mf_row` can emit
    // (hard limit 100.0, min_practical 1.0): all seven outcomes must pass
    // `executed_dv` through bit-identically.
    let executed_dv = 7.125_f64;
    let cases = [
        (
            5.0,
            LABEL_SAFE_BY_DEFAULT,
            1.0,
            3.0,
            true,
            REASON_SAFE_BY_DEFAULT,
        ),
        (5.0, LABEL_OTHER, 1.0, 3.0, true, REASON_OK),
        (
            f64::NAN,
            LABEL_OTHER,
            1.0,
            3.0,
            true,
            REASON_DETERMINISTIC_MASS_INVALID,
        ),
        // Physics-limited label with a valid deterministic mass.
        (
            5.0,
            LABEL_PHYSICS_LIMITED,
            1.0,
            3.0,
            true,
            REASON_DETERMINISTIC_MASS_PHYSICS_LIMITED,
        ),
        // Deterministic floor at/above the hard limit short-circuits Pc.
        (
            5.0,
            LABEL_OTHER,
            150.0,
            3.0,
            true,
            REASON_PROB_MASS_FLOOR_HARD_LIMIT,
        ),
        // Caller-invalidated precomputed Pc.
        (5.0, LABEL_OTHER, 1.0, 3.0, false, REASON_PROB_MASS_INVALID),
        // Probabilistic total mass at/above the hard limit.
        (
            5.0,
            LABEL_OTHER,
            1.0,
            150.0,
            true,
            REASON_PROB_MASS_HARD_LIMIT,
        ),
    ];

    for (det_mass, label, floor_mass, pc_total_mass, pc_valid, expected_reason) in cases {
        let verdict = fraction_grid_finalize_mf_row(
            executed_dv,
            det_mass,
            label,
            floor_mass,
            pc_total_mass,
            pc_valid,
            100.0,
            1.0,
        )
        .unwrap();

        assert_eq!(verdict.executed_dv.to_bits(), executed_dv.to_bits());
        assert_eq!(verdict.reason_code, expected_reason);
    }
}
