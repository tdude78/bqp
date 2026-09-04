use anyhow::ensure;
use dust_estimates_rs::{
    mass_solver::{
        solve_batch_events_mf_j2_with_evidence, solve_batch_events_mf_j2_with_status_into,
        MfJ2MassSolveStatusCode, MfJ2MassSolverEvent, OperationalDeterministicMass, SolverConfig,
    },
    project_shared_target_bplane_components, shared_target_contact_mass_requirement, DustMassClaim,
    DustScenarioIdentity, SharedTargetBplaneProjectionInputs, SharedTargetMassEstimate,
    SharedTargetMassInputs, SharedTargetPacketCountGovernor, SharedTargetPositionTreatment,
    SharedTargetQuadrature, SharedTargetScenario, MAX_EXACT_BINARY64_PACKET_COUNT,
    SHARED_TARGET_CLAIM_ID, SHARED_TARGET_COUNT_CERTIFICATE_ID, SHARED_TARGET_METHOD_ID,
};

fn deterministic_event(kappa: f64) -> MfJ2MassSolverEvent {
    MfJ2MassSolverEvent::new(
        [
            666.354_001_014_283_1,
            -2_237.979_584_736_737_7,
            -6_663.785_651_912_341,
        ],
        [
            1.935_632_268_969_379_2,
            -6.800_088_428_550_588,
            2.477_168_251_036_653_5,
        ],
        [
            -2.543_021_665_074_745_5,
            7.148_671_179_188_607,
            -2.141_705_688_340_100_4,
        ],
        31.0,
        [
            359.780_610_720_872_35,
            -1_161.215_882_391_510_1,
            -6_955.256_925_339_665,
        ],
        46_741.434_985_399_246,
        1.0,
        kappa,
    )
}

const fn deterministic_config() -> SolverConfig {
    SolverConfig {
        xtol: 1.0e-6,
        rtol: 1.0e-5,
        maxiter: 50,
        mass_max: 1000.0,
    }
}

fn quadrature(
    target_radial_samples: usize,
    target_angular_samples: usize,
    convergence_tolerance: f64,
) -> anyhow::Result<SharedTargetQuadrature> {
    SharedTargetQuadrature::new(
        target_radial_samples,
        target_angular_samples,
        convergence_tolerance,
    )
}

fn named_scenario(
    identifier: &'static str,
    kappa: f64,
    packet_correlation_grains: u64,
    target_position_sigma_m: f64,
    quadrature: SharedTargetQuadrature,
) -> anyhow::Result<SharedTargetScenario> {
    SharedTargetScenario::new(
        DustScenarioIdentity::named(identifier)?,
        kappa,
        packet_correlation_grains,
        target_position_sigma_m,
        SharedTargetPositionTreatment::AssumedNotObservedIsotropicEncounterBplaneSharedDraw,
        quadrature,
        DustMassClaim::ModelConditionedConservativeContactRequirement,
    )
}

fn scenario(packet_correlation_grains: u64) -> anyhow::Result<SharedTargetScenario> {
    named_scenario(
        "shared-target-binary64-integration-test",
        1.0,
        packet_correlation_grains,
        100.0,
        quadrature(32, 16, 2.0e-2)?,
    )
}

fn deterministic_operational_mass(kappa: f64) -> anyhow::Result<OperationalDeterministicMass> {
    let (outcomes, _) = solve_batch_events_mf_j2_with_evidence(
        &[deterministic_event(kappa)],
        &deterministic_config(),
    );
    outcomes
        .first()
        .ok_or_else(|| anyhow::anyhow!("single-event MF batch returned no outcome"))?
        .operational_mass()?
        .ok_or_else(|| anyhow::anyhow!("converged MF row issued no operational mass"))
}

fn contact_estimate(
    scenario: SharedTargetScenario,
    operational_mass: OperationalDeterministicMass,
    means: &[f64],
    covariances: &[f64],
    weights: &[f64],
    area_km2: f64,
    target_hit_probability: f64,
    grain_mass_kg: f64,
) -> anyhow::Result<SharedTargetMassEstimate> {
    shared_target_contact_mass_requirement(&SharedTargetMassInputs {
        deterministic_mass: scenario.bind_deterministic_mass(operational_mass)?,
        projected_means_2d: means,
        projected_covariances_2d: covariances,
        mixture_weights: weights,
        area_km2,
        target_hit_probability,
        grain_mass_kg,
        covariance_minimum: 1.0e-12,
        covariance_maximum: 1.0e12,
    })
}

#[test]
fn singular_binary64_contact_requirement_binds_minimum_and_governor() -> anyhow::Result<()> {
    let scenario = scenario(1)?;
    let operational_mass = deterministic_operational_mass(1.0)?;
    let deterministic_mass = scenario.bind_deterministic_mass(operational_mass)?;
    let estimate = shared_target_contact_mass_requirement(&SharedTargetMassInputs {
        deterministic_mass,
        projected_means_2d: &[0.0, 0.0],
        projected_covariances_2d: &[0.0025, 0.0, 0.0, 0.0025],
        mixture_weights: &[7.0],
        area_km2: std::f64::consts::PI * 0.00125_f64.powi(2),
        target_hit_probability: 0.5,
        grain_mass_kg: 1.0,
        covariance_minimum: 1.0e-12,
        covariance_maximum: 1.0,
    })?;
    let witness = estimate.witness();
    ensure!(witness.method_id() == SHARED_TARGET_METHOD_ID);
    ensure!(witness.claim_id() == SHARED_TARGET_CLAIM_ID);
    ensure!(witness.count_certificate_id() == SHARED_TARGET_COUNT_CERTIFICATE_ID);
    ensure!(estimate.effective_packet_count() == witness.final_packet_count());
    ensure!(
        witness.final_packet_count()
            == witness
                .probability_packet_count()
                .max(witness.deterministic_floor_packet_count())
    );
    if witness.governor() == SharedTargetPacketCountGovernor::Probability {
        ensure!(witness.governed_predecessor_log_no_hit_bits().is_some());
        ensure!(
            f64::from_bits(witness.probability_predecessor_log_no_hit_bits())
                > f64::from_bits(witness.policy_threshold_log_bits())
        );
    }
    ensure!(
        f64::from_bits(witness.selected_log_no_hit_bits())
            <= f64::from_bits(witness.policy_threshold_log_bits())
    );
    Ok(())
}

#[test]
fn scenario_rejects_packet_correlation_outside_exact_binary64_domain() {
    let error = scenario(MAX_EXACT_BINARY64_PACKET_COUNT + 1)
        .expect_err("2^53 packet correlation must be rejected before multiplication");
    assert!(error.to_string().contains("exact binary64 integer domain"));
}

#[test]
fn shared_target_expectation_exceeds_independent_convolution_mass() -> anyhow::Result<()> {
    let scenario = named_scenario(
        "isotropic-target-draw",
        1.0,
        1,
        100.0,
        quadrature(192, 32, 1.0e-4)?,
    )?;
    let grain_variance = 0.03_f64.powi(2);
    let radius_km = 0.00125_f64;
    let estimate = contact_estimate(
        scenario,
        deterministic_operational_mass(1.0)?,
        &[0.0, 0.0],
        &[grain_variance, 0.0, 0.0, grain_variance],
        &[1.0],
        std::f64::consts::PI * radius_km * radius_km,
        0.9,
        1.0,
    )?;
    let target_variance = 0.1_f64.powi(2);
    let analytic_capture =
        -(-radius_km.powi(2) / (2.0 * (grain_variance + target_variance))).exp_m1();
    let independent_trials = ((1.0_f64 - 0.9).ln() / (-analytic_capture).ln_1p()).ceil();
    let relative = ((estimate.expected_conditional_capture_probability() - analytic_capture)
        / analytic_capture)
        .abs();
    let witness = estimate.witness();
    ensure!(
        relative <= 1.0e-4,
        "isotropic target draw differs from closed form: measured={} analytic={analytic_capture} relative={relative}",
        estimate.expected_conditional_capture_probability()
    );
    ensure!(
        independent_trials.is_finite()
            && independent_trials > 0.0
            && estimate.release_mass_kg() > independent_trials
            && witness.target_refinement_level() == 0
            && estimate.no_hit_probability()
                <= scenario
                    .quadrature()
                    .conservative_failure_probability(0.9)?,
        "shared target draw lost dependence or base-grid threshold authority"
    );
    Ok(())
}

#[test]
fn target_draw_refines_once_and_then_fails_closed() -> anyhow::Result<()> {
    let operational_mass = deterministic_operational_mass(1.0)?;
    let variance = 0.1_f64.powi(2);
    let means = [0.0, 0.0];
    let covariances = [variance, 0.0, 0.0, 9.0 * variance];
    let area_km2 = std::f64::consts::PI * 0.00125_f64.powi(2);
    let refined_scenario = named_scenario(
        "single-target-refinement",
        1.0,
        1,
        100.0,
        quadrature(16, 16, 1.0e-4)?,
    )?;
    let refined = contact_estimate(
        refined_scenario,
        operational_mass,
        &means,
        &covariances,
        &[1.0],
        area_km2,
        0.99,
        1.0,
    )?;
    let witness = refined.witness();
    ensure!(
        witness.target_refinement_level() == 1
            && witness.target_radial_samples() == 32
            && witness.target_angular_samples() == 16
            && witness.base_target_quadrature_delta() > 1.0e-4
            && witness.target_quadrature_delta() <= 1.0e-4,
        "bounded target refinement witness is incomplete or nonminimal"
    );

    let hostile_scenario = named_scenario(
        "exhausted-target-refinement",
        1.0,
        1,
        100.0,
        quadrature(16, 16, f64::MIN_POSITIVE)?,
    )?;
    let error = contact_estimate(
        hostile_scenario,
        operational_mass,
        &means,
        &covariances,
        &[1.0],
        area_km2,
        0.99,
        1.0,
    )
    .expect_err("second target-grid disagreement must fail closed");
    ensure!(
        error.to_string().contains("bounded refinement"),
        "target-refinement failure lost context: {error}"
    );
    Ok(())
}

#[test]
fn packet_correlation_is_probability_independent_and_floor_governs() -> anyhow::Result<()> {
    let operational_mass = deterministic_operational_mass(0.8)?;
    let independent_scenario = named_scenario(
        "packet-correlation-one",
        0.8,
        1,
        400.0,
        quadrature(32, 16, 2.0e-2)?,
    )?;
    let correlated_scenario = named_scenario(
        "packet-correlation-four",
        0.8,
        4,
        400.0,
        quadrature(32, 16, 2.0e-2)?,
    )?;
    let means = [0.0, 0.0];
    let covariances = [0.25, 0.0, 0.0, 0.25];
    let weights = [1.0];
    let area_km2 = std::f64::consts::PI * 0.00125_f64.powi(2);
    let independent = contact_estimate(
        independent_scenario,
        operational_mass,
        &means,
        &covariances,
        &weights,
        area_km2,
        0.8,
        1.0,
    )?;
    let correlated = contact_estimate(
        correlated_scenario,
        operational_mass,
        &means,
        &covariances,
        &weights,
        area_km2,
        0.8,
        1.0,
    )?;
    ensure!(
        independent.witness().probability_packet_count()
            == correlated.witness().probability_packet_count()
            && independent.witness().governor() == SharedTargetPacketCountGovernor::Probability
            && correlated.witness().governor() == SharedTargetPacketCountGovernor::Probability
            && independent.effective_packet_count() == correlated.effective_packet_count(),
        "packet correlation changed binary64 probability minimum"
    );
    ensure!(
        correlated.release_mass_kg().to_bits() == (4.0 * independent.release_mass_kg()).to_bits(),
        "four-grain packets did not scale released mass exactly"
    );

    let floor_grain_mass =
        independent.deterministic_required_mass_kg() / (2.0 * independent.release_mass_kg());
    let floor = contact_estimate(
        independent_scenario,
        operational_mass,
        &means,
        &covariances,
        &weights,
        area_km2,
        0.8,
        floor_grain_mass,
    )?;
    let witness = floor.witness();
    ensure!(
        witness.governor() == SharedTargetPacketCountGovernor::DeterministicFloor
            && witness.deterministic_floor_packet_count() > witness.probability_packet_count()
            && witness.final_packet_count() == witness.deterministic_floor_packet_count()
            && witness.governed_predecessor_log_no_hit_bits().is_none()
            && floor.release_mass_kg() >= floor.deterministic_required_mass_kg(),
        "deterministic floor did not govern through the public solver"
    );
    Ok(())
}

#[test]
fn mf_batch_evidence_binds_executed_mass_and_kappa() -> anyhow::Result<()> {
    let converged = deterministic_event(1.0);
    let mut rejected = converged;
    rejected.min_miss_distance_km = 0.0;
    let events = [converged, rejected];
    let (outcomes, dispatched_parallel) =
        solve_batch_events_mf_j2_with_evidence(&events, &deterministic_config());
    let mut masses = [0.0; 2];
    let mut statuses = [MfJ2MassSolveStatusCode::MissAtZeroNonFinite; 2];
    let mut miss_zero = [0.0; 2];
    let mut miss_root = [0.0; 2];
    let mut miss_upper = [0.0; 2];
    let mut iterations = [0; 2];
    let raw_dispatched_parallel = solve_batch_events_mf_j2_with_status_into(
        &events,
        &deterministic_config(),
        &mut masses,
        &mut statuses,
        &mut miss_zero,
        &mut miss_root,
        &mut miss_upper,
        &mut iterations,
    );
    ensure!(
        outcomes.len() == 2
            && dispatched_parallel == raw_dispatched_parallel
            && outcomes.iter().zip(masses.iter().zip(statuses)).all(
                |(outcome, (&mass, status))| {
                    outcome.mass_kg().to_bits() == mass.to_bits() && outcome.status() == status
                }
            ),
        "MF evidence batch drifted from executed deterministic solver"
    );
    let converged = outcomes
        .first()
        .ok_or_else(|| anyhow::anyhow!("MF batch returned no converged row"))?;
    let issued = converged
        .operational_mass()?
        .ok_or_else(|| anyhow::anyhow!("converged MF row issued no operational mass"))?;
    let rejected = outcomes
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("MF batch returned no rejected row"))?;
    ensure!(
        rejected.operational_mass()?.is_none(),
        "rejected MF row forged operational mass"
    );
    let matching = named_scenario(
        "mf-batch-evidence-match",
        1.0,
        1,
        100.0,
        quadrature(32, 16, 2.0e-2)?,
    )?
    .bind_deterministic_mass(issued)?;
    ensure!(
        issued.raw_solver_mass_kg().to_bits() == converged.mass_kg().to_bits()
            && matching.required_mass_kg().to_bits()
                == issued.commanded_required_mass_kg().to_bits(),
        "operational binding changed raw solver evidence or commanded mass"
    );
    let mismatch = named_scenario(
        "mf-batch-evidence-kappa-mismatch",
        0.65,
        1,
        100.0,
        quadrature(32, 16, 2.0e-2)?,
    )?
    .bind_deterministic_mass(issued)
    .expect_err("scenario must reject operational mass issued under another kappa");
    ensure!(
        mismatch.to_string().contains("kappa") && mismatch.to_string().contains("does not match"),
        "kappa mismatch lost binding context: {mismatch}"
    );
    Ok(())
}

#[test]
fn production_projection_keeps_target_covariance_outside_per_grain_covariance() -> anyhow::Result<()>
{
    let component_mean_6d = [11.0, 22.0, 33.0, 0.0, 0.0, 1.0];
    let mut component_covariance_6d = [0.0_f64; 36];
    component_covariance_6d[0] = 4.0;
    component_covariance_6d[7] = 9.0;
    component_covariance_6d[14] = 16.0;
    let projection =
        project_shared_target_bplane_components(&SharedTargetBplaneProjectionInputs {
            component_means_6d: &component_mean_6d,
            component_covariances_6d: &component_covariance_6d,
            target_state: &[10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
            hf_velocity_mean: &[0.0, 0.0, 1.0],
            covariance_minimum: 1.0e-12,
            covariance_maximum: 1.0e12,
        })?;
    ensure!(
        projection
            .projected_covariances_2d()
            .iter()
            .copied()
            .map(f64::to_bits)
            .eq([9.0, 0.0, 0.0, 4.0].map(f64::to_bits)),
        "B-plane projection convolved target covariance into per-grain covariance"
    );
    ensure!(
        projection.projection_clamped() == 0
            && projection
                .projected_means_2d()
                .iter()
                .all(|value| value.is_finite()),
        "ordinary projection clamped or produced nonfinite means"
    );
    let target_covariance = scenario(1)?.target_covariance_2d_km2();
    ensure!(
        (target_covariance[0] - 0.01).abs() <= 4.0 * f64::EPSILON
            && target_covariance[1].to_bits() == 0.0_f64.to_bits()
            && target_covariance[2].to_bits() == 0.0_f64.to_bits()
            && (target_covariance[3] - 0.01).abs() <= 4.0 * f64::EPSILON,
        "assumed target covariance did not remain separate"
    );
    Ok(())
}
