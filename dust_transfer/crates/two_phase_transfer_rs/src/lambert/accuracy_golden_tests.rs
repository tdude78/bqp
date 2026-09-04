//! Golden reference tests for Lambert solver accuracy.
//!
//! These tests ensure delta-V values don't regress by more than TOLERANCE per component.
//!
//! REQUIREMENT: delta-V must not worsen by more than 1e-9 per component

use super::izzo2015_impl;
use satpy_core::MU;

/// Maximum allowed deviation from golden values per velocity component (km/s)
const TOLERANCE: f64 = 1e-9;

// =============================================================================
// Golden values: per-pair provenance.
//
// Every constant below is a SELF-REFERENTIAL golden captured from this
// crate's own Rust Izzo2015 implementation — a regression pin, not an
// independent oracle.
//
// - LEO_M0_PROGRADE: regenerated 2026-01-03 from the then-current
//   implementation.
// - LEO_M0_RETROGRADE, LEO_M1_PROGRADE, GEO_M0_PROGRADE, HIGH_ECC_M0: carried
//   forward from the original 2024-01-01 capture. Whether they were also
//   re-verified on 2026-01-03 is unrecorded, so the implementation revision
//   behind these four pairs is UNRESOLVED. They still pin today's behavior,
//   but the 2024 date must not be read as verified provenance.
// =============================================================================

// LEO_M0_PROGRADE (m=0, prograde=true)
// Regenerated 2026-01-03 from current Rust Izzo2015 implementation
const GOLDEN_LEO_M0_PROGRADE_V1: [f64; 3] = [4.961_263_019_087_933, 5.690_412_001_212_047, 0.0];
const GOLDEN_LEO_M0_PROGRADE_V2: [f64; 3] = [-5.373_309_075_538_486_5, -4.644_160_093_414_372, 0.0];

// LEO_M0_RETROGRADE (m=0, prograde=false)
const GOLDEN_LEO_M0_RETROGRADE_V1: [f64; 3] =
    [-0.876_713_668_015_866, -7.441_088_427_080_653, -0.0];
const GOLDEN_LEO_M0_RETROGRADE_V2: [f64; 3] = [7.026_427_606_401_876, 0.462_052_847_337_089_1, 0.0];

// LEO_M1_PROGRADE (m=1, prograde=true)
const GOLDEN_LEO_M1_PROGRADE_V1: [f64; 3] = [-1.381_437_145_496_800_6, 8.656_981_453_818_297, 0.0];
const GOLDEN_LEO_M1_PROGRADE_V2: [f64; 3] = [-8.174_563_986_344_443, 1.863_854_612_970_654, 0.0];

// GEO_M0_PROGRADE (m=0, prograde=true)
const GOLDEN_GEO_M0_PROGRADE_V1: [f64; 3] = [1.604_430_543_715_305_7, 2.375_381_633_673_021_6, 0.0];
const GOLDEN_GEO_M0_PROGRADE_V2: [f64; 3] =
    [-2.375_381_633_673_021_6, -1.604_430_543_715_305_7, 0.0];

// HIGH_ECC_M0 (m=0, prograde=true)
const GOLDEN_HIGH_ECC_M0_V1: [f64; 3] = [7.614_304_564_606_675, 7.363_474_032_728_440_5, 0.0];
const GOLDEN_HIGH_ECC_M0_V2: [f64; 3] = [-0.491_732_796_071_909_1, -0.742_563_328_652_940_8, 0.0];

// =============================================================================
// Test cases
// =============================================================================

// LEO transfer test data
const R1_LEO: [f64; 3] = [6778.0, 0.0, 0.0];
const R2_LEO: [f64; 3] = [0.0, 7178.0, 0.0];
const TOF_LEO: f64 = 3600.0; // 1 hour
const TOF_LEO_MULTI: f64 = 10800.0; // 3 hours

// GEO transfer test data
const R1_GEO: [f64; 3] = [42164.0, 0.0, 0.0];
const R2_GEO: [f64; 3] = [0.0, 42164.0, 0.0];
const TOF_GEO: f64 = 43200.0; // 12 hours

// High eccentricity test data
const R1_HIGH_ECC: [f64; 3] = [6678.0, 0.0, 0.0];
const R2_HIGH_ECC: [f64; 3] = [0.0, 100_000.0, 0.0];
const TOF_HIGH_ECC: f64 = 86400.0; // 24 hours

/// Helper function to check velocity against golden value
fn check_velocity(actual: &[f64; 3], golden: &[f64; 3], label: &str) {
    for (i, (actual_component, golden_component)) in actual.iter().zip(golden).enumerate() {
        let diff = (actual_component - golden_component).abs();
        assert!(
            diff < TOLERANCE,
            "{label} component {i} differs by {diff:.2e}, exceeds tolerance {TOLERANCE:.2e}\n  actual:  {actual:?}\n  golden:  {golden:?}"
        );
    }
}

#[test]
fn golden_leo_m0_prograde() {
    let result = izzo2015_impl(MU, &R1_LEO, &R2_LEO, TOF_LEO, 0, true, true, 25, 1e-9, 1e-9);
    assert!(result.success, "LEO m=0 prograde solve failed");
    check_velocity(&result.v1, &GOLDEN_LEO_M0_PROGRADE_V1, "v1");
    check_velocity(&result.v2, &GOLDEN_LEO_M0_PROGRADE_V2, "v2");
}

#[test]
fn golden_leo_m0_retrograde() {
    let result = izzo2015_impl(
        MU, &R1_LEO, &R2_LEO, TOF_LEO, 0, false, true, 25, 1e-9, 1e-9,
    );
    assert!(result.success, "LEO m=0 retrograde solve failed");
    check_velocity(&result.v1, &GOLDEN_LEO_M0_RETROGRADE_V1, "v1");
    check_velocity(&result.v2, &GOLDEN_LEO_M0_RETROGRADE_V2, "v2");
}

#[test]
fn golden_leo_m1_prograde() {
    let result = izzo2015_impl(
        MU,
        &R1_LEO,
        &R2_LEO,
        TOF_LEO_MULTI,
        1,
        true,
        true,
        25,
        1e-9,
        1e-9,
    );
    assert!(result.success, "LEO m=1 prograde solve failed");
    check_velocity(&result.v1, &GOLDEN_LEO_M1_PROGRADE_V1, "v1");
    check_velocity(&result.v2, &GOLDEN_LEO_M1_PROGRADE_V2, "v2");
}

#[test]
fn golden_geo_m0_prograde() {
    let result = izzo2015_impl(MU, &R1_GEO, &R2_GEO, TOF_GEO, 0, true, true, 25, 1e-9, 1e-9);
    assert!(result.success, "GEO m=0 prograde solve failed");
    check_velocity(&result.v1, &GOLDEN_GEO_M0_PROGRADE_V1, "v1");
    check_velocity(&result.v2, &GOLDEN_GEO_M0_PROGRADE_V2, "v2");
}

#[test]
fn golden_high_eccentricity() {
    let result = izzo2015_impl(
        MU,
        &R1_HIGH_ECC,
        &R2_HIGH_ECC,
        TOF_HIGH_ECC,
        0,
        true,
        true,
        25,
        1e-9,
        1e-9,
    );
    assert!(result.success, "High eccentricity m=0 solve failed");
    check_velocity(&result.v1, &GOLDEN_HIGH_ECC_M0_V1, "v1");
    check_velocity(&result.v2, &GOLDEN_HIGH_ECC_M0_V2, "v2");
}
