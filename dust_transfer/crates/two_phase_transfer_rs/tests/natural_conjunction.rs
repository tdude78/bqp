use two_phase_transfer_rs::types::{BodyForceConfig, BodyRole};
use two_phase_transfer_rs::{
    NaturalConjunctionEnclosure, NaturalConjunctionFatalError, NaturalConjunctionInputError,
    NaturalConjunctionOutcome, NaturalObjectInput, TransferPostprocessSessionCore,
};

mod support;
use support::{strict_physics, strict_session, T0_JD_UTC};

const PRIMARY_SOURCE: [u8; 32] = [0x11; 32];
const PRIMARY_BODY: [u8; 32] = [0x12; 32];
const SECONDARY_SOURCE: [u8; 32] = [0x21; 32];
const SECONDARY_BODY: [u8; 32] = [0x22; 32];

fn object(norad: u64, state: [f64; 6], source: [u8; 32], body: [u8; 32]) -> NaturalObjectInput {
    object_with_force(
        norad,
        state,
        source,
        body,
        BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 0.01, 2.2, 1.3),
    )
}

#[expect(
    clippy::expect_used,
    reason = "test-only setup helper: a refused Part A authority must abort \
              the test loudly; clippy's allow-expect-in-tests covers \
              `#[test]` fns, not free helpers"
)]
fn object_with_force(
    norad: u64,
    state: [f64; 6],
    source: [u8; 32],
    body: [u8; 32],
    body_force: BodyForceConfig,
) -> NaturalObjectInput {
    NaturalObjectInput::new(norad, source, body, T0_JD_UTC, state, body_force)
        .expect("valid natural-object authority")
}

#[test]
fn natural_object_rejects_missing_identity() {
    let error = NaturalObjectInput::new(
        40_001,
        [0; 32],
        PRIMARY_BODY,
        T0_JD_UTC,
        circular_state(),
        BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 0.01, 2.2, 1.3),
    )
    .expect_err("zero source digest is missing identity authority");
    assert!(matches!(
        error,
        NaturalConjunctionFatalError::InvalidInput(NaturalConjunctionInputError::Identity)
    ));
    let error = NaturalObjectInput::new(
        40_001,
        PRIMARY_SOURCE,
        PRIMARY_BODY,
        T0_JD_UTC,
        circular_state(),
        BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 0.01, 2.2, 0.0),
    )
    .expect_err("zero reflectivity is invalid body authority");
    assert!(matches!(
        error,
        NaturalConjunctionFatalError::InvalidInput(NaturalConjunctionInputError::BodyForce)
    ));
}

fn circular_state() -> [f64; 6] {
    let speed = (398_600.441_8_f64 / 7_000.0).sqrt();
    [7_000.0, 0.0, 0.0, 0.0, speed, 0.0]
}

#[test]
fn refinement_fails_closed_on_force_or_enclosure_authority() {
    let primary = object(10_001, circular_state(), PRIMARY_SOURCE, PRIMARY_BODY);
    let secondary = object(10_002, circular_state(), SECONDARY_SOURCE, SECONDARY_BODY);

    let mut wrong = strict_physics();
    wrong.dt_max = f64::from_bits(wrong.dt_max.to_bits() + 1);
    let wrong_session = TransferPostprocessSessionCore::try_new(Some(wrong), None)
        .expect("generic strict session accepts non-Part-A tuning");
    assert!(matches!(
        wrong_session.refine_natural_conjunction(
            &primary,
            &secondary,
            NaturalConjunctionEnclosure::new(0.0, 1.0),
        ),
        Err(NaturalConjunctionFatalError::ForceAuthorityMismatch)
    ));

    assert!(matches!(
        strict_session().refine_natural_conjunction(
            &primary,
            &secondary,
            NaturalConjunctionEnclosure::new(0.0, 120.0 + f64::EPSILON * 128.0),
        ),
        Err(NaturalConjunctionFatalError::InvalidInput(
            NaturalConjunctionInputError::WorkLimit
        ))
    ));

    assert!(matches!(
        strict_session().refine_natural_conjunction(
            &primary,
            &secondary,
            NaturalConjunctionEnclosure::new(0.0, 14.0 * 86_400.0 + 1.0),
        ),
        Err(NaturalConjunctionFatalError::InvalidInput(
            NaturalConjunctionInputError::Enclosure
        ))
    ));
}

#[test]
fn closed_endpoint_safe_candidate_is_typed_infeasible() {
    let primary_state = circular_state();
    let [x, y, z, vx, vy, vz] = primary_state;
    let secondary_state = [x + 2.0, y, z, vx, vy, vz];
    let primary = object(20_001, primary_state, PRIMARY_SOURCE, PRIMARY_BODY);
    let secondary = object(20_002, secondary_state, SECONDARY_SOURCE, SECONDARY_BODY);

    let outcome = strict_session()
        .refine_natural_conjunction(
            &primary,
            &secondary,
            NaturalConjunctionEnclosure::new(0.0, 0.0),
        )
        .expect("intact authority must classify a physical safe candidate");
    let NaturalConjunctionOutcome::CandidateInfeasible(infeasible) = outcome else {
        panic!("2 km closed endpoint must be physically infeasible");
    };
    assert_eq!(infeasible.closest_offset_s().to_bits(), 0.0_f64.to_bits());
    assert!(infeasible.miss_distance_km() >= 1.0);
    assert_eq!(infeasible.primary_identity(), primary.identity());
    assert_eq!(infeasible.secondary_identity(), secondary.identity());
}

#[test]
fn interior_conjunction_is_opaque_authority_bound_and_independently_witnessed() {
    let primary_state = circular_state();
    let [x, y, z, vx, vy, vz] = primary_state;
    let secondary_state = [x + 0.5, y, z, vx - 1.0, vy, vz];
    let primary = object(30_001, primary_state, PRIMARY_SOURCE, PRIMARY_BODY);
    let secondary_force =
        BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 0.012, 2.1, 1.2);
    let secondary = object_with_force(
        30_002,
        secondary_state,
        SECONDARY_SOURCE,
        SECONDARY_BODY,
        secondary_force,
    );
    let substituted = object_with_force(
        30_002,
        secondary_state,
        SECONDARY_SOURCE,
        SECONDARY_BODY,
        BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 0.013, 2.1, 1.2),
    );
    assert_ne!(secondary.identity(), substituted.identity());
    let [x, y, z, vx, vy, vz] = secondary_state;
    let substituted_state = [f64::from_bits(x.to_bits() + 1), y, z, vx, vy, vz];
    let substituted = object_with_force(
        30_002,
        substituted_state,
        SECONDARY_SOURCE,
        SECONDARY_BODY,
        secondary_force,
    );
    assert_ne!(secondary.identity(), substituted.identity());
    let enclosure = NaturalConjunctionEnclosure::new(0.0, 1.0);

    let outcome = strict_session()
        .refine_natural_conjunction(&primary, &secondary, enclosure)
        .expect("short intact strict-HF enclosure must refine");
    let NaturalConjunctionOutcome::Verified(verified) = outcome else {
        panic!("closing half-kilometre pair must verify");
    };

    assert_eq!(verified.primary_identity(), primary.identity());
    assert_eq!(verified.secondary_identity(), secondary.identity());
    assert_eq!(
        verified.enclosure_lower_offset_s().to_bits(),
        enclosure.lower_offset_s().to_bits()
    );
    assert_eq!(
        verified.enclosure_upper_offset_s().to_bits(),
        enclosure.upper_offset_s().to_bits()
    );
    assert!((enclosure.lower_offset_s()..=enclosure.upper_offset_s())
        .contains(&verified.refined_offset_s()));
    assert!(verified.refined_offset_s() > 0.0);
    assert!(verified.refined_offset_s() < 1.0);
    assert!(verified.miss_distance_km() < 1.0);
    assert_eq!(
        verified.conjunction_jd_utc().to_bits(),
        (T0_JD_UTC + verified.refined_offset_s() / 86_400.0).to_bits()
    );
    assert!(verified
        .primary_conjunction_state()
        .iter()
        .chain(verified.secondary_conjunction_state().iter())
        .all(|component| component.is_finite()));
    assert!(verified
        .primary_independent_witness_state()
        .iter()
        .chain(verified.secondary_independent_witness_state().iter())
        .all(|component| component.is_finite()));
    assert!(verified.primary_position_residual_km() <= 0.025);
    assert!(verified.secondary_position_residual_km() <= 0.025);
    assert!(verified.primary_velocity_residual_km_s() <= 2.0e-5);
    assert!(verified.secondary_velocity_residual_km_s() <= 2.0e-5);
    assert_ne!(verified.force_authority_sha256(), [0; 32]);
    assert_ne!(verified.ephemeris_authority_sha256(), [0; 32]);
    assert_ne!(verified.gravity_source_sha256(), [0; 32]);
    assert_ne!(verified.gravity_packed_semantic_sha256(), [0; 32]);
    assert_ne!(
        verified.gravity_source_sha256(),
        verified.gravity_packed_semantic_sha256()
    );
    assert!(verified.verify_digest());
}
