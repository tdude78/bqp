//! Pins the audit finding (2026-08-16) that the finite-packet bound's
//! `target_probability` knob is LOAD-FREE at the production seal
//! `grains_per_independent_packet = 1`: at ~0.3 kg deterministic mass and the
//! sealed 6.45e-10 kg grain (~4.65e8 "independent" trials), moving the target
//! confidence from 0.50 to 0.99 moves the released mass by ~1e-4 relative.
//! The released mass is therefore effectively `det_mass / capture_probability`
//! and must be quoted as model-conditional, never as "P(hit) >= 0.99".
//! This test goes red the day the bound's arithmetic (or the packet-size
//! seal's role in it) changes enough to make the confidence knob load-bearing.

use dust_estimates_rs::finite_packet_release_mass_bound_core;

const GRAIN_MASS_KG: f64 = 6.45e-10;

#[test]
fn confidence_knob_is_load_free_at_production_scale() -> anyhow::Result<()> {
    let det_mass_kg = 0.3;
    let capture_probability = 0.8;
    let low = finite_packet_release_mass_bound_core(
        capture_probability,
        0.50,
        det_mass_kg,
        GRAIN_MASS_KG,
        1,
    )?;
    let high = finite_packet_release_mass_bound_core(
        capture_probability,
        0.99,
        det_mass_kg,
        GRAIN_MASS_KG,
        1,
    )?;
    let relative = (high.release_mass_kg - low.release_mass_kg) / low.release_mass_kg;
    anyhow::ensure!(
        relative >= 0.0,
        "higher confidence must never lower the bound (got {relative})"
    );
    anyhow::ensure!(
        relative < 1e-3,
        "confidence 0.50 -> 0.99 moved the mass by {relative} relative; the \
         load-free property no longer holds and every model-conditional label \
         citing it must be revisited"
    );
    // The bound at production scale is dominated by det_mass / capture.
    let expected = det_mass_kg / capture_probability;
    let dominance = (low.release_mass_kg - expected).abs() / expected;
    anyhow::ensure!(
        dominance < 1e-3,
        "released mass {} is not det/capture-dominated (expected ~{expected}, \
         relative gap {dominance})",
        low.release_mass_kg
    );
    Ok(())
}

#[test]
fn confidence_knob_is_load_bearing_at_small_packet_counts() -> anyhow::Result<()> {
    // Control arm: with few independent packets the Chernoff term MUST matter,
    // proving this suite measures the knob and not a constant function.
    let low =
        finite_packet_release_mass_bound_core(0.8, 0.50, 10.0 * GRAIN_MASS_KG, GRAIN_MASS_KG, 1)?;
    let high =
        finite_packet_release_mass_bound_core(0.8, 0.99, 10.0 * GRAIN_MASS_KG, GRAIN_MASS_KG, 1)?;
    anyhow::ensure!(
        high.release_mass_kg > low.release_mass_kg * 1.5,
        "confidence knob failed to move a 10-packet bound ({} -> {})",
        low.release_mass_kg,
        high.release_mass_kg
    );
    Ok(())
}
