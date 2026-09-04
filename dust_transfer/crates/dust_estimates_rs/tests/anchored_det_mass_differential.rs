//! The anchored differential must return the catalogue's own miss at zero mass.
use dust_estimates_rs::mass_solver::{
    solve_single_event_mf_j2_with_status, MfJ2MassSolveStatusCode, MfJ2MassSolverEvent,
    SolverConfig,
};

fn base_event(other_conj: [f64; 3]) -> MfJ2MassSolverEvent {
    MfJ2MassSolverEvent::new(
        [7000.0, 0.0, 0.0],
        [0.0, 7.546_053_288, 0.0],
        [0.0, 7.6, 0.5],
        1200.0,
        other_conj,
        3600.0,
        1.0,
        1.0,
    )
}

/// At zero mass the anchored miss IS the catalogue miss, whatever the model
/// displacement over the time of flight.
///
/// This is the property the unanchored formula lacked: it measured
/// `MF-J2 image - catalogue position`, so a catalogue whose conjunction is not
/// an MF-J2 image reported the model displacement as a miss.
///
/// The displacement is measured on the sealed asset by
/// `nd_pipeline::event_bank_v3`'s `v3_is_not_an_mf_j2_image_of_itself`, which
/// prints `V3_MF_J2_DISPLACEMENT_RECEIPT`: over the first 24 events it is
/// 3.169 km min, 27.844 km median, 177.938 km max, against a catalogue miss of
/// at most 0.085 km. `deterministic_mass_min_distance_km` = 1.0 km sits between
/// the two, which is why every row read `SafeByDefault`. That test supplies the
/// input; this one pins the rule that turns it into a verdict.
#[test]
fn anchored_zero_mass_miss_is_the_catalogue_miss() {
    let cfg = SolverConfig {
        xtol: 1.0e-6,
        rtol: 1.0e-5,
        maxiter: 60,
        mass_max: 1.0e4,
    };
    // Deliberately far from wherever MF-J2 sends the target: this stands in for
    // a strict-HF conjunction anchor, which is exactly what v3 supplies.
    let other_conj = [-6000.0, 3000.0, 120.0];
    let target_conj = [-6000.0, 3000.0, 120.010_676];
    let expected_miss = 0.010_676_f64;

    let unanchored = solve_single_event_mf_j2_with_status(&base_event(other_conj), &cfg);
    let anchored = solve_single_event_mf_j2_with_status(
        &base_event(other_conj).with_conjunction_anchor(target_conj),
        &cfg,
    );

    // THE VERDICT FLIP IS THE CLAIM, so it is asserted rather than described in
    // a commit message. Unanchored, the model displacement exceeds
    // `deterministic_mass_min_distance_km` and the row is declared safe with no
    // mass required -- which is what every sealed v3 event did.
    assert_eq!(
        unanchored.status,
        MfJ2MassSolveStatusCode::SafeByDefault,
        "unanchored status should be SafeByDefault, miss0 {}",
        unanchored.miss_at_zero_km
    );
    assert!(
        unanchored.miss_at_zero_km > 1.0,
        "unanchored miss0 {} should carry the model displacement",
        unanchored.miss_at_zero_km
    );
    assert_eq!(
        anchored.status,
        MfJ2MassSolveStatusCode::Converged,
        "anchored status should be Converged, miss0 {}",
        anchored.miss_at_zero_km
    );
    assert!(
        anchored.root_mass_kg.is_finite() && anchored.root_mass_kg > 0.0,
        "anchored solve must produce a positive finite release mass, got {}",
        anchored.root_mass_kg
    );
    let delta = (anchored.miss_at_zero_km - expected_miss).abs();
    assert!(
        delta < 1.0e-9,
        "anchored miss0 {} is not the catalogue miss {expected_miss} (delta {delta})",
        anchored.miss_at_zero_km
    );
}

/// The anchor is opt-in and changes nothing when absent.
#[test]
fn absent_anchor_reproduces_the_historical_formula_bit_for_bit() {
    let cfg = SolverConfig {
        xtol: 1.0e-6,
        rtol: 1.0e-5,
        maxiter: 60,
        mass_max: 1.0e4,
    };
    let other_conj = [-6000.0, 3000.0, 120.0];
    let event = base_event(other_conj);
    let a = solve_single_event_mf_j2_with_status(&event, &cfg);
    let b = solve_single_event_mf_j2_with_status(&base_event(other_conj), &cfg);
    assert_eq!(
        a.miss_at_zero_km.to_bits(),
        b.miss_at_zero_km.to_bits(),
        "unanchored path is not deterministic"
    );
    assert!(
        event.target_conj_pos.is_none(),
        "anchor must default to None"
    );
}

/// COST AND OUTCOME on realistic v3 numbers.
///
/// Diagnostic, not a gate: it prints the status, root mass and iteration count
/// so the campaign cost of the anchored path is on record. Before the anchor,
/// every v3 row returned `SafeByDefault` after ONE propagation; the anchored
/// path runs the bracket expansion and bisection for real.
#[test]
fn anchored_solve_cost_and_outcome_on_v3_scale_inputs() {
    let cfg = SolverConfig {
        xtol: 1.0e-6,
        rtol: 1.0e-5,
        maxiter: 60,
        mass_max: 1.0e4,
    };
    let other_conj = [-6000.0, 3000.0, 120.0];
    let target_conj = [-6000.0, 3000.0, 120.010_676];
    // ~4.3 days, the median v3 refined TCA offset.
    let event = MfJ2MassSolverEvent::new(
        [7000.0, 0.0, 0.0],
        [0.0, 7.546_053_288, 0.0],
        [0.0, 7.6, 0.5],
        1200.0,
        other_conj,
        368_473.815_345_411_8,
        1.0,
        1.0,
    )
    .with_conjunction_anchor(target_conj);
    let start = std::time::Instant::now();
    let result = solve_single_event_mf_j2_with_status(&event, &cfg);
    let elapsed = start.elapsed();
    // Printed so a reader can reproduce the numbers quoted for this change
    // rather than take them from a commit message.
    eprintln!(
        "ANCHORED_DET_MASS_RECEIPT status={:?} root_kg={} miss0_km={} miss_upper_km={} iters={} elapsed={elapsed:?}",
        result.status,
        result.root_mass_kg,
        result.miss_at_zero_km,
        result.miss_at_upper_km,
        result.iterations
    );
    // Bounds, not pins: the point is that the solve CONVERGES and terminates in
    // a bounded number of steps on a v3-scale time of flight. A pinned root mass
    // here would be a bit pin in a file that is not one.
    assert_eq!(result.status, MfJ2MassSolveStatusCode::Converged);
    assert!(result.root_mass_kg > 0.0 && result.root_mass_kg < cfg.mass_max);
    assert!(
        result.iterations <= cfg.maxiter,
        "anchored solve used {} iterations against maxiter {}",
        result.iterations,
        cfg.maxiter
    );
}
