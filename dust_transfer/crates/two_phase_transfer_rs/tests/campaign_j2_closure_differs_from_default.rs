//! Pins the fact that the campaign's J2 closure settings are NOT
//! `J2ClosureSettings::default()`.
//!
//! The two disagree on `max_iterations` (library default 8 vs campaign 5) and
//! `correction_step_gain` (default 0.7 vs campaign 1.0).
//! Gain 1.0 is the exact Newton step; 0.7 undershoots and buys
//! several extra iterations, so anything timed or profiled on the default runs
//! the J2 block far more than production does. That is why the MF benches read
//! `nd_config` instead of calling `default()`.
//!
//! If the two ever converge this test fails. That is deliberate: the
//! difference is a known fact, and collapsing it is a decision someone should
//! have to make on purpose — updating this test — rather than a silent drift
//! that quietly re-points every bench.

use two_phase_transfer_rs::solve::J2ClosureSettings;

const fn campaign_j2_settings() -> J2ClosureSettings {
    let controls = nd_config::CompiledPartAScienceV1::part_a_v1().mf_transfer();
    J2ClosureSettings {
        max_iterations: controls.j2_max_iterations,
        endpoint_target_km: controls.j2_endpoint_target_km,
        correction_step_gain: controls.j2_correction_step_gain,
    }
}

#[test]
fn campaign_j2_closure_is_not_the_default() {
    let campaign = campaign_j2_settings();
    let default = J2ClosureSettings::default();

    // Literal pins of BOTH sides. A bare != between two production-derived
    // values would still pass if both drifted together; these fail the moment
    // either side moves.
    assert_eq!(campaign.max_iterations, 5);
    assert_eq!(default.max_iterations, 8);
    assert_eq!(campaign.correction_step_gain.to_bits(), 1.0_f64.to_bits());
    assert_eq!(default.correction_step_gain.to_bits(), 0.7_f64.to_bits());

    assert_ne!(
        campaign.max_iterations, default.max_iterations,
        "campaign and default J2 closure iteration caps have converged \
         (both {}); benches read the campaign value from nd_config on the \
         premise that they differ. If this is intended, update this test.",
        campaign.max_iterations
    );

    assert_ne!(
        campaign.correction_step_gain.to_bits(),
        default.correction_step_gain.to_bits(),
        "campaign and default J2 correction step gains have converged \
         (both {}); benches read the campaign value from nd_config on the \
         premise that they differ. If this is intended, update this test.",
        campaign.correction_step_gain
    );

    // The endpoint target is the one field the two agree on. Stated so a
    // reader does not mistake its absence above for an oversight.
    assert_eq!(
        campaign.endpoint_target_km.to_bits(),
        default.endpoint_target_km.to_bits()
    );
}
