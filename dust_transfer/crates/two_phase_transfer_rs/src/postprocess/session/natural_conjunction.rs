use super::{StrictHfForceAuthority, TransferPostprocessSessionCore};
use crate::evaluate::{
    propagate_high_fidelity_state_independent_witness,
    propagate_high_fidelity_target_at_authoritative_offset_checked,
    propagate_high_fidelity_target_dense_grid_checked,
    propagate_high_fidelity_target_multi_tof_checked, TransferPropagationFailure,
};
use crate::types::{
    all_finite, BodyForceConfig, BodyRole, ExecutionPolicy, PlanContext, PropagationFidelity,
    TargetPropagationAuthority, MAX_TOF_SAMPLES,
};
use lightyear_odeint_rs::precomputed_ephem::{
    embedded_catalogue_sha256_hex, part_a_ephemeris_authority, Body,
};
use lightyear_odeint_rs::types::{ForceConfig, StepperMethod};
use satpy_core::{eci2equinoc_impl, SEC_PER_DAY};
use sha2::{Digest, Sha256};
use std::fmt;
use std::sync::Arc;

const MAX_ENCLOSURE_S: f64 = 14.0 * SEC_PER_DAY;
const MAX_NON_POINT_ENCLOSURE_S: f64 = 120.0;
/// How far before its enclosure a scan anchor may sit.
///
/// Production anchors on the dense screening node the candidate's 120 s slab
/// opens on, so the lead is zero; one slab of slack lets a caller anchor on the
/// preceding node without needing a new contract. Together with the 120 s
/// enclosure cap this bounds an anchored scan's integration at 240 s, against
/// up to fourteen days from the object epoch.
const MAX_ANCHOR_LEAD_S: f64 = 120.0;
const GRID_INTERVALS: usize = 32;
/// Golden-section depth per bracket.
///
/// Resolution is `bracket_s * 0.618^REFINE_ITERATIONS`. On the sealed 3.75 s
/// grid a bracket is 7.5 s, so 14 iterations resolve ~9e-4 s, about 6.7 m of
/// relative motion; because miss distance is second order at a minimum that is
/// sub-metre miss error. The acceptance decision is a 1 km threshold
/// (`UNSAFE_MISS_KM`) and this function already admits 25 m of state error
/// (`WITNESS_POSITION_KM`), so deeper iteration is invisible to every consumer.
///
/// 14 is a FLOOR, not a preference: `all_observed_local_brackets_recover_lower_
/// basin_missed_by_one_best_route` needs the refined root inside 0.012 s of
/// 11.2 s, i.e. `7.5 * 0.618^k < 0.012`, which requires `k >= 14`.
const REFINE_ITERATIONS: usize = 14;

/// Interior brackets available on the sealed grid: a strict interior minimum
/// needs a `windows(3)` centre, so at most `GRID_INTERVALS / 2` of them.
const MAX_INTERIOR_BRACKETS: usize = GRID_INTERVALS / 2;

/// One-sided brackets: the two grid endpoints, which `windows(3)` can never
/// centre. See `refine_every_local_bracket`.
const MAX_ENDPOINT_BRACKETS: usize = 2;

/// Sealed worst-case pair-evaluation budget, DERIVED so it cannot drift from
/// the two constants that actually determine it. Each bracket costs
/// `REFINE_ITERATIONS + 2` evaluations (two golden-section seeds plus one per
/// iteration), and the grid itself costs `GRID_INTERVALS + 1`.
const MAX_TOTAL_PAIR_EVALUATIONS: usize =
    (MAX_INTERIOR_BRACKETS + MAX_ENDPOINT_BRACKETS) * (REFINE_ITERATIONS + 2) + GRID_INTERVALS + 1;

/// Conservative bound on relative acceleration between two Part A objects,
/// km/s^2, used ONLY to decide whether an endpoint minimum could reach
/// `UNSAFE_MISS_KM` within one grid step. Two-body acceleration at the 6578 km
/// radial floor is `mu / r^2` = 9.2e-3 km/s^2; the RELATIVE acceleration of two
/// nearby objects is tidal and smaller still, so twice the single-body value is
/// a safe envelope for a 3.75 s extrapolation.
const ENDPOINT_GATE_REL_ACCEL_KM_S2: f64 = 1.84e-2;
const UNSAFE_MISS_KM: f64 = 1.0;
/// Grid nodes per dense-arc segment. Ten 60 s nodes is a 600 s segment, short
/// enough for the eclipse scanner's bracket bound on every probed LEO arc.
const DENSE_ARC_SEGMENT_NODES: usize = 10;
/// Bisection depth for a segment that still fails a recoverable propagation.
/// Ten halvings reduce a segment to a single node well before the cap.
const DENSE_ARC_MAX_SPLIT_DEPTH: u32 = 10;
/// How many grid nodes the restart anchor may walk back when a segment fails.
///
/// Bisecting a failing segment keeps its START fixed, so it cannot help when
/// the restart instant itself is the problem -- an eclipse root committed at or
/// next to a segment boundary leaves the boundary geometry numerically
/// ambiguous, and the scanner reports chatter. Restarting from an earlier node
/// that is already solved moves the restart instant by a multiple of the grid
/// step without moving a single output node off the grid.
const DENSE_ARC_MAX_ANCHOR_BACKOFF_NODES: usize = 5;

/// Ceiling on how far a chained dense-arc node may sit from the from-epoch
/// propagation of the same instant, in kilometres.
///
/// The screening cache restarts the Encke baseline at every segment boundary,
/// so it is not bit-identical to the authority. Measured worst case over the
/// probe set is 1.2e-4 km at the far end of a full fourteen-day arc; this
/// ceiling is ~80x that.
///
/// Public because it is a SHARED constant, not an internal one: the
/// `natural_dense_arc` suite enforces it, and `nd_part_a_v3_generator` derives
/// its narrowphase margin against it. Both previously carried their own copy of
/// the literal with a comment saying the two must agree and nothing making them
/// agree. One definition, two readers.
pub const NATURAL_DENSE_ARC_AUTHORITY_CEILING_KM: f64 = 0.01;

const WITNESS_POSITION_KM: f64 = 0.025;
const WITNESS_VELOCITY_KM_S: f64 = 2.0e-5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NaturalObjectIdentity([u8; 32]);

#[derive(Clone, Debug)]
pub struct NaturalObjectInput {
    identity: NaturalObjectIdentity,
    catalogue_number: u64,
    epoch_jd_utc: f64,
    state: [f64; 6],
    body_force: BodyForceConfig,
}

impl NaturalObjectInput {
    /// Stamp one natural object with its source and property authorities.
    ///
    /// # Errors
    ///
    /// Returns [`NaturalConjunctionFatalError::InvalidInput`] with
    /// [`NaturalConjunctionInputError::Identity`] for a zero catalogue number
    /// or an all-zero authority digest, and with the state or force variant
    /// when the stamped state or body force is not usable.
    pub fn new(
        catalogue_number: u64,
        source_authority_sha256: [u8; 32],
        property_authority_sha256: [u8; 32],
        epoch_jd_utc: f64,
        state: [f64; 6],
        body_force: BodyForceConfig,
    ) -> Result<Self, NaturalConjunctionFatalError> {
        if catalogue_number == 0
            || source_authority_sha256 == [0; 32]
            || property_authority_sha256 == [0; 32]
        {
            return Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::Identity,
            ));
        }
        if !epoch_jd_utc.is_finite() {
            return Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::Epoch,
            ));
        }
        if !all_finite(&state) {
            return Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::State,
            ));
        }
        if !natural_body_force_is_valid(body_force) {
            return Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::BodyForce,
            ));
        }
        let mut digest = Sha256::new();
        digest.update(b"nasa-dust/natural-object-identity/v1\0");
        digest.update(catalogue_number.to_le_bytes());
        digest.update(source_authority_sha256);
        digest.update(property_authority_sha256);
        digest.update(epoch_jd_utc.to_bits().to_le_bytes());
        for component in state {
            digest.update(component.to_bits().to_le_bytes());
        }
        update_body_force_identity(&mut digest, body_force);
        Ok(Self {
            identity: NaturalObjectIdentity(digest.finalize().into()),
            catalogue_number,
            epoch_jd_utc,
            state,
            body_force,
        })
    }

    #[must_use]
    pub const fn identity(&self) -> NaturalObjectIdentity {
        self.identity
    }

    /// NORAD catalogue number, so a propagation failure can name the object.
    #[must_use]
    pub const fn catalogue_number(&self) -> u64 {
        self.catalogue_number
    }

    /// The sealed epoch state this object's arcs start from.
    #[must_use]
    pub const fn state(&self) -> [f64; 6] {
        self.state
    }

    #[must_use]
    pub const fn epoch_jd_utc(&self) -> f64 {
        self.epoch_jd_utc
    }
}

fn natural_body_force_is_valid(body_force: BodyForceConfig) -> bool {
    body_force.role == BodyRole::DiagnosticTarget
        && body_force.fidelity == PropagationFidelity::HighFidelity
        && body_force.am_ratio.is_finite()
        && body_force.am_ratio > 0.0
        && body_force.cd.is_finite()
        && body_force.cd > 0.0
        && body_force.cr.is_finite()
        && body_force.cr > 0.0
}

fn update_body_force_identity(digest: &mut Sha256, body_force: BodyForceConfig) {
    digest.update([match body_force.role {
        BodyRole::TransferVehicle => 0,
        BodyRole::Canister => 1,
        BodyRole::Dust => 2,
        BodyRole::DiagnosticTarget => 3,
    }]);
    digest.update([match body_force.fidelity {
        PropagationFidelity::J2 => 0,
        PropagationFidelity::HighFidelity => 1,
    }]);
    for value in [body_force.am_ratio, body_force.cd, body_force.cr] {
        digest.update(value.to_bits().to_le_bytes());
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NaturalConjunctionEnclosure {
    lower_offset_s: f64,
    upper_offset_s: f64,
}

impl NaturalConjunctionEnclosure {
    #[must_use]
    pub const fn new(lower_offset_s: f64, upper_offset_s: f64) -> Self {
        Self {
            lower_offset_s,
            upper_offset_s,
        }
    }

    #[must_use]
    pub const fn lower_offset_s(self) -> f64 {
        self.lower_offset_s
    }

    #[must_use]
    pub const fn upper_offset_s(self) -> f64 {
        self.upper_offset_s
    }
}

/// Cached strict-HF states for both objects of one candidate at a single
/// offset from their shared epoch, used to re-base the sealed scan.
///
/// `offset_s` is measured from the objects' shared epoch, exactly like an
/// enclosure offset. It must open at or before the enclosure and lead it by at
/// most [`MAX_ANCHOR_LEAD_S`], which bounds how much integration an anchored
/// scan can hide behind a cached state.
///
/// An anchor is a screening value, not an authority. A state read from the
/// dense screening cache restarts the Encke baseline at every segment
/// boundary, so it is not bit-identical to the from-epoch propagation of the
/// same instant. What binds an accepted conjunction is the from-epoch
/// independent witness, which no anchor touches.
#[derive(Clone, Copy, Debug)]
pub struct NaturalConjunctionScanAnchor {
    offset_s: f64,
    primary_state: [f64; 6],
    secondary_state: [f64; 6],
}

impl NaturalConjunctionScanAnchor {
    /// # Errors
    ///
    /// Returns `InvalidInput(Anchor)` when the offset is not finite and
    /// non-negative, or either state is not finite.
    pub fn new(
        offset_s: f64,
        primary_state: [f64; 6],
        secondary_state: [f64; 6],
    ) -> Result<Self, NaturalConjunctionFatalError> {
        if !offset_s.is_finite()
            || offset_s < 0.0
            || !all_finite(&primary_state)
            || !all_finite(&secondary_state)
        {
            return Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::Anchor,
            ));
        }
        Ok(Self {
            offset_s,
            primary_state,
            secondary_state,
        })
    }

    #[must_use]
    pub const fn offset_s(self) -> f64 {
        self.offset_s
    }

    #[must_use]
    pub const fn primary_state(self) -> [f64; 6] {
        self.primary_state
    }

    #[must_use]
    pub const fn secondary_state(self) -> [f64; 6] {
        self.secondary_state
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NaturalConjunctionInputError {
    Identity,
    Epoch,
    State,
    BodyForce,
    Enclosure,
    WorkLimit,
    DuplicateObject,
    Anchor,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NaturalConjunctionFatalError {
    /// A dense-arc segment failed, with enough context to reproduce it: which
    /// segment, which grid node it starts at, the epoch it restarted from, and
    /// how many nodes the restart anchor had already been walked back.
    DenseArcSegment {
        segment_index: usize,
        first_node: usize,
        segment_epoch_jd_utc: f64,
        anchor_backoff_nodes: usize,
        inner: Box<Self>,
    },
    InvalidInput(NaturalConjunctionInputError),
    ForceAuthorityMismatch,
    AuthorityUnavailable,
    Propagation(TransferPropagationFailure),
}

impl fmt::Display for NaturalConjunctionFatalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "natural conjunction fatal error: {self:?}")
    }
}

impl std::error::Error for NaturalConjunctionFatalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Propagation(error) => Some(error),
            Self::DenseArcSegment { inner, .. } => Some(inner.as_ref()),
            Self::InvalidInput(_) | Self::ForceAuthorityMismatch | Self::AuthorityUnavailable => {
                None
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NaturalConjunctionInfeasible {
    primary_identity: NaturalObjectIdentity,
    secondary_identity: NaturalObjectIdentity,
    closest_offset_s: f64,
    miss_distance_km: f64,
}

impl NaturalConjunctionInfeasible {
    #[must_use]
    pub const fn primary_identity(&self) -> NaturalObjectIdentity {
        self.primary_identity
    }
    #[must_use]
    pub const fn secondary_identity(&self) -> NaturalObjectIdentity {
        self.secondary_identity
    }
    #[must_use]
    pub const fn closest_offset_s(&self) -> f64 {
        self.closest_offset_s
    }
    #[must_use]
    pub const fn miss_distance_km(&self) -> f64 {
        self.miss_distance_km
    }
}

/// One candidate whose two independent integrations of the SAME arc disagreed
/// by more than the witness gate.
///
/// This is a property of one arc, not of the authority: the force model, the
/// ephemeris and the gravity field are all intact, and every other candidate in
/// the pool is unaffected. The residual grows monotonically with arc length, so
/// a stiff, low-perigee object at the far end of a fourteen-day span can cross
/// the gate while a benign one at the same offset does not.
///
/// It is NOT a pass. The consumer cannot vouch for either state, so the honest
/// response is to taint the whole pair -- suppress, never admit. The gate stays
/// where it is; only the blast radius changes, from the run to the pair.
#[derive(Clone, Copy, Debug)]
pub struct NaturalConjunctionWitnessResidual {
    primary_identity: NaturalObjectIdentity,
    secondary_identity: NaturalObjectIdentity,
    closest_offset_s: f64,
    miss_distance_km: f64,
    primary_position_residual_km: f64,
    secondary_position_residual_km: f64,
    primary_velocity_residual_km_s: f64,
    secondary_velocity_residual_km_s: f64,
}

impl NaturalConjunctionWitnessResidual {
    #[must_use]
    pub const fn primary_identity(&self) -> NaturalObjectIdentity {
        self.primary_identity
    }
    #[must_use]
    pub const fn secondary_identity(&self) -> NaturalObjectIdentity {
        self.secondary_identity
    }
    #[must_use]
    pub const fn closest_offset_s(&self) -> f64 {
        self.closest_offset_s
    }
    #[must_use]
    pub const fn miss_distance_km(&self) -> f64 {
        self.miss_distance_km
    }
    #[must_use]
    pub const fn primary_position_residual_km(&self) -> f64 {
        self.primary_position_residual_km
    }
    #[must_use]
    pub const fn secondary_position_residual_km(&self) -> f64 {
        self.secondary_position_residual_km
    }
    #[must_use]
    pub const fn primary_velocity_residual_km_s(&self) -> f64 {
        self.primary_velocity_residual_km_s
    }
    #[must_use]
    pub const fn secondary_velocity_residual_km_s(&self) -> f64 {
        self.secondary_velocity_residual_km_s
    }
}

/// The witness gate itself, as a predicate over the four residuals.
///
/// Extracted so the gate has exactly one definition and so each of its four
/// arms can be driven independently by a test: an OR with four arms passes for
/// three wrong reasons if only one arm is ever exercised.
#[must_use]
fn witness_residual_over_gate(
    primary_position_residual_km: f64,
    secondary_position_residual_km: f64,
    primary_velocity_residual_km_s: f64,
    secondary_velocity_residual_km_s: f64,
) -> bool {
    primary_position_residual_km > WITNESS_POSITION_KM
        || secondary_position_residual_km > WITNESS_POSITION_KM
        || primary_velocity_residual_km_s > WITNESS_VELOCITY_KM_S
        || secondary_velocity_residual_km_s > WITNESS_VELOCITY_KM_S
}

#[derive(Clone, Debug)]
pub struct VerifiedNaturalConjunction {
    primary_identity: NaturalObjectIdentity,
    secondary_identity: NaturalObjectIdentity,
    enclosure: NaturalConjunctionEnclosure,
    refined_offset_s: f64,
    conjunction_jd_utc: f64,
    miss_distance_km: f64,
    primary_state: [f64; 6],
    secondary_state: [f64; 6],
    primary_independent_witness_state: [f64; 6],
    secondary_independent_witness_state: [f64; 6],
    primary_position_residual_km: f64,
    secondary_position_residual_km: f64,
    primary_velocity_residual_km_s: f64,
    secondary_velocity_residual_km_s: f64,
    force_authority_sha256: [u8; 32],
    ephemeris_authority_sha256: [u8; 32],
    gravity_source_sha256: [u8; 32],
    gravity_packed_semantic_sha256: [u8; 32],
    digest: [u8; 32],
}

impl VerifiedNaturalConjunction {
    #[must_use]
    pub const fn primary_identity(&self) -> NaturalObjectIdentity {
        self.primary_identity
    }
    #[must_use]
    pub const fn secondary_identity(&self) -> NaturalObjectIdentity {
        self.secondary_identity
    }
    #[must_use]
    pub const fn enclosure_lower_offset_s(&self) -> f64 {
        self.enclosure.lower_offset_s
    }
    #[must_use]
    pub const fn enclosure_upper_offset_s(&self) -> f64 {
        self.enclosure.upper_offset_s
    }
    #[must_use]
    pub const fn refined_offset_s(&self) -> f64 {
        self.refined_offset_s
    }
    #[must_use]
    pub const fn conjunction_jd_utc(&self) -> f64 {
        self.conjunction_jd_utc
    }
    #[must_use]
    pub const fn miss_distance_km(&self) -> f64 {
        self.miss_distance_km
    }
    #[must_use]
    pub const fn primary_conjunction_state(&self) -> &[f64; 6] {
        &self.primary_state
    }
    #[must_use]
    pub const fn secondary_conjunction_state(&self) -> &[f64; 6] {
        &self.secondary_state
    }
    #[must_use]
    pub const fn primary_independent_witness_state(&self) -> &[f64; 6] {
        &self.primary_independent_witness_state
    }
    #[must_use]
    pub const fn secondary_independent_witness_state(&self) -> &[f64; 6] {
        &self.secondary_independent_witness_state
    }
    #[must_use]
    pub const fn primary_position_residual_km(&self) -> f64 {
        self.primary_position_residual_km
    }
    #[must_use]
    pub const fn secondary_position_residual_km(&self) -> f64 {
        self.secondary_position_residual_km
    }
    #[must_use]
    pub const fn primary_velocity_residual_km_s(&self) -> f64 {
        self.primary_velocity_residual_km_s
    }
    #[must_use]
    pub const fn secondary_velocity_residual_km_s(&self) -> f64 {
        self.secondary_velocity_residual_km_s
    }
    #[must_use]
    pub const fn force_authority_sha256(&self) -> [u8; 32] {
        self.force_authority_sha256
    }
    #[must_use]
    pub const fn ephemeris_authority_sha256(&self) -> [u8; 32] {
        self.ephemeris_authority_sha256
    }
    #[must_use]
    pub const fn gravity_source_sha256(&self) -> [u8; 32] {
        self.gravity_source_sha256
    }
    #[must_use]
    pub const fn gravity_packed_semantic_sha256(&self) -> [u8; 32] {
        self.gravity_packed_semantic_sha256
    }
    #[must_use]
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }
    #[must_use]
    pub fn verify_digest(&self) -> bool {
        verified_digest(self) == self.digest
    }
}

#[derive(Clone, Debug)]
pub enum NaturalConjunctionOutcome {
    CandidateInfeasible(NaturalConjunctionInfeasible),
    CandidatePropagationInfeasible(TransferPropagationFailure),
    /// The independent witness disagreed with the refined state by more than
    /// the witness gate on THIS candidate. Per-candidate, never fatal: see
    /// `NaturalConjunctionWitnessResidual`.
    CandidateWitnessResidual(NaturalConjunctionWitnessResidual),
    Verified(Box<VerifiedNaturalConjunction>),
}

struct NaturalPropagator {
    context: PlanContext,
}

impl NaturalPropagator {
    fn new(
        session: &TransferPostprocessSessionCore,
        object: &NaturalObjectInput,
    ) -> Result<Self, NaturalConjunctionFatalError> {
        Self::at(
            session,
            object.epoch_jd_utc,
            object.state,
            object.body_force,
        )
    }

    /// Build a propagator for one epoch and state under the session's force
    /// authority.
    ///
    /// Every field is rederived from the session and the two arguments, so an
    /// arc is a pure function of its own epoch and state. Nothing is carried
    /// across a segment boundary except the state itself -- in particular the
    /// force config is rebuilt, because it is stamped against the epoch it was
    /// built for and reusing one across a moving epoch walks the atmosphere
    /// driver off its authorized coverage.
    fn at(
        session: &TransferPostprocessSessionCore,
        epoch_jd_utc: f64,
        state: [f64; 6],
        body_force: BodyForceConfig,
    ) -> Result<Self, NaturalConjunctionFatalError> {
        if !epoch_jd_utc.is_finite() || !all_finite(&state) {
            return Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::State,
            ));
        }
        let mut equinoctial = [0.0; 6];
        eci2equinoc_impl(&state, 6, 0.0, 0.0, &mut equinoctial);
        if !all_finite(&equinoctial) {
            return Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::State,
            ));
        }
        let force_config = super::super::build_force_config(
            &session.physics_config,
            body_force.am_ratio,
            body_force.cd,
            body_force.cr,
        )
        .map_err(|_| NaturalConjunctionFatalError::ForceAuthorityMismatch)?;
        if !force_config_matches_body(&force_config, body_force) {
            return Err(NaturalConjunctionFatalError::ForceAuthorityMismatch);
        }
        let packed = session
            .coeffs
            .packed
            .clone()
            .ok_or(NaturalConjunctionFatalError::AuthorityUnavailable)?;
        let context = PlanContext {
            epoch_jd: epoch_jd_utc,
            tgt_eci: state,
            tgt_equ: equinoctial,
            execution_policy: ExecutionPolicy {
                use_high_fidelity: true,
                require_high_fidelity: true,
                allow_parallel: false,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            target_propagation_authority: TargetPropagationAuthority::HighFidelity,
            target_body_force: body_force,
            force_config: Some(Arc::new(force_config)),
            packed_coeffs: Some(packed),
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };
        Ok(Self { context })
    }

    fn state(&self, offset_s: f64) -> Result<[f64; 6], NaturalConjunctionFatalError> {
        propagate_high_fidelity_target_at_authoritative_offset_checked(&self.context, offset_s)
            .map_err(NaturalConjunctionFatalError::Propagation)
    }

    fn grid(&self, offsets: &[f64]) -> Result<Vec<[f64; 6]>, NaturalConjunctionFatalError> {
        let mut states = vec![[f64::NAN; 6]; offsets.len()];
        let mut positive = Vec::with_capacity(offsets.len());
        let mut positive_indices = Vec::with_capacity(offsets.len());
        for (index, &offset) in offsets.iter().enumerate() {
            if offset == 0.0 {
                *states
                    .get_mut(index)
                    .ok_or(NaturalConjunctionFatalError::InvalidInput(
                        NaturalConjunctionInputError::WorkLimit,
                    ))? = self.state(offset)?;
            } else {
                positive.push(offset);
                positive_indices.push(index);
            }
        }
        if !positive.is_empty() {
            for (offset_chunk, index_chunk) in positive
                .chunks(MAX_TOF_SAMPLES)
                .zip(positive_indices.chunks(MAX_TOF_SAMPLES))
            {
                let mut propagated = vec![[f64::NAN; 6]; offset_chunk.len()];
                propagate_high_fidelity_target_multi_tof_checked(
                    &self.context,
                    offset_chunk,
                    &mut propagated,
                )
                .map_err(NaturalConjunctionFatalError::Propagation)?;
                for (&index, state) in index_chunk.iter().zip(propagated) {
                    *states
                        .get_mut(index)
                        .ok_or(NaturalConjunctionFatalError::InvalidInput(
                            NaturalConjunctionInputError::WorkLimit,
                        ))? = state;
                }
            }
        }
        Ok(states)
    }

    /// Fill a dense, strictly increasing offset grid from one integration.
    ///
    /// A leading `0.0` node is the propagator's own epoch state; every later
    /// node comes from a single forward integration, not one per node.
    fn dense_grid(&self, offsets_s: &[f64]) -> Result<Vec<[f64; 6]>, NaturalConjunctionFatalError> {
        let mut states = vec![[f64::NAN; 6]; offsets_s.len()];
        let (epoch_nodes, positive_offsets) = match offsets_s.split_first() {
            Some((first, rest)) if *first == 0.0 => (1_usize, rest),
            _ => (0_usize, offsets_s),
        };
        if epoch_nodes == 1 {
            *states
                .first_mut()
                .ok_or(NaturalConjunctionFatalError::InvalidInput(
                    NaturalConjunctionInputError::WorkLimit,
                ))? = self.state(0.0)?;
        }
        if !positive_offsets.is_empty() {
            let out =
                states
                    .get_mut(epoch_nodes..)
                    .ok_or(NaturalConjunctionFatalError::InvalidInput(
                        NaturalConjunctionInputError::WorkLimit,
                    ))?;
            propagate_high_fidelity_target_dense_grid_checked(&self.context, positive_offsets, out)
                .map_err(NaturalConjunctionFatalError::Propagation)?;
        }
        Ok(states)
    }

    fn witness(&self, offset_s: f64) -> Result<[f64; 6], NaturalConjunctionFatalError> {
        propagate_high_fidelity_state_independent_witness(
            &self.context.tgt_equ,
            offset_s,
            self.context.epoch_jd,
            self.context.target_body_force,
            &self.context,
            30.0,
            1.0e-10,
        )
        .map_err(NaturalConjunctionFatalError::Propagation)
    }
}

#[derive(Clone, Copy)]
struct Evaluation {
    offset_s: f64,
    primary: [f64; 6],
    secondary: [f64; 6],
    distance_squared: f64,
}

impl TransferPostprocessSessionCore {
    /// Deterministic sealed-resolution model refinement for approved concept-study scope.
    ///
    /// Every non-point adaptive mean-J2 root enclosure is at most 120 seconds and gets
    /// exactly 32 sealed intervals, including both endpoints. Every observed strict
    /// local-minimum bracket receives the same fixed refinement work. Results use a
    /// deterministic total order across all samples and refinements. It does not claim
    /// detection of minima below the at-most-3.75-second sealed grid resolution and is
    /// not a formal flowpipe.
    /// Build one object's dense fixed-grid ephemeris across the sealed arc.
    ///
    /// Returns states at offsets `0, step, 2*step, ...` from the object's
    /// epoch, node zero being the object's own epoch state.
    ///
    /// MAY RETURN FEWER THAN `node_count` STATES. Some arcs cannot be
    /// propagated past a particular instant under the sealed authority -- an
    /// eclipse boundary the scanner cannot resolve, reported as chatter. That
    /// is a property of the arc, not of this entry point: the same instant
    /// defeats a single-endpoint propagation from the object's own epoch, which
    /// is exactly what exact refinement performs. Rather than fail the whole
    /// object, the longest solved prefix is returned, and the caller must treat
    /// the missing tail as "no information" rather than as "no conjunction".
    ///
    /// The arc is walked as a chain of short segments, each restarted from the
    /// previous segment's end state. A single fourteen-day integration carrying
    /// output samples throughout cannot be used: the eclipse event scanner
    /// requires a certified relative-motion bound between consecutive scan
    /// endpoints, and requesting samples before the first shadow crossing on a
    /// long arc fails the bracket outright. Segments short enough to stay
    /// inside that bound do work, and the total integrated time is unchanged --
    /// the whole grid costs about one arc.
    ///
    /// A segment that still fails a recoverable propagation is bisected and
    /// retried, down to a single node. The split depends only on propagation
    /// results, so it is reproducible.
    ///
    /// This is a screening cache, not an acceptance authority. It runs the same
    /// sealed strict-HF force model, but restarting the Encke baseline at each
    /// segment boundary means it is not bit-identical to a from-epoch
    /// propagation, and nothing it returns is bound into a verified conjunction
    /// digest. `natural_dense_arc_matches_from_epoch_authority` measures the
    /// discrepancy.
    ///
    /// # Errors
    ///
    /// Returns an error when the session force authority is not the sealed
    /// Part A strict-HF model, when the grid is invalid or leaves the sealed
    /// fourteen-day arc, or when a single-node segment cannot be propagated.
    pub fn natural_dense_ephemeris_arc(
        &self,
        object: &NaturalObjectInput,
        node_step_s: f64,
        node_count: usize,
    ) -> Result<Vec<[f64; 6]>, NaturalConjunctionFatalError> {
        validate_session_authority(self)?;
        if !natural_body_force_is_valid(object.body_force) {
            return Err(NaturalConjunctionFatalError::ForceAuthorityMismatch);
        }
        let enclosure_error =
            NaturalConjunctionFatalError::InvalidInput(NaturalConjunctionInputError::Enclosure);
        let span_nodes = node_count
            .checked_sub(1)
            .ok_or_else(|| enclosure_error.clone())?;
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "usize has no infallible f64 conversion; the horizon check on \
                      the next line rejects any node count the sealed arc cannot \
                      hold, which is far under 2^53"
        )]
        let horizon_s = node_step_s * span_nodes as f64;
        if !node_step_s.is_finite()
            || node_step_s <= 0.0
            || node_count == 0
            || !horizon_s.is_finite()
            || horizon_s > MAX_ENCLOSURE_S
        {
            return Err(enclosure_error);
        }

        let mut states = Vec::with_capacity(node_count);
        states.push(NaturalPropagator::new(self, object)?.state(0.0)?);
        if span_nodes == 0 {
            return Ok(states);
        }

        // Epoch of grid node `node`, always derived from the object epoch times
        // the node index, never by accumulating an increment: 2,016 additions
        // of the same increment drift, and the final node sits exactly on the
        // authorized arc boundary where drift is fatal.
        let node_epoch = |node: usize| {
            #[expect(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                reason = "usize has no infallible f64 conversion; the node index is \
                          bounded by the arc horizon validated above, far under 2^53"
            )]
            let advanced_s = node_step_s * node as f64;
            object.epoch_jd_utc + advanced_s / SEC_PER_DAY
        };

        let mut filled = 0_usize;
        let mut segment_index = 0_usize;
        while filled < span_nodes {
            let remaining = span_nodes.saturating_sub(filled);
            let take = remaining.min(DENSE_ARC_SEGMENT_NODES);
            let before = states.len();

            // Walk the restart anchor back a node at a time until the segment
            // solves. Backoff zero is the ordinary case. Every anchor is a node
            // already solved on this same grid, and the requested offsets are
            // shifted to compensate, so the output nodes are identical whichever
            // anchor succeeds -- only the instant the Encke baseline restarts at
            // moves, which is the thing an eclipse root next to a segment
            // boundary makes unusable.
            let mut attempt = 0_usize;
            let outcome = loop {
                let backoff = attempt.min(filled);
                let anchor_node = filled.saturating_sub(backoff);
                let anchor_state =
                    *states
                        .get(anchor_node)
                        .ok_or(NaturalConjunctionFatalError::InvalidInput(
                            NaturalConjunctionInputError::WorkLimit,
                        ))?;
                let anchor = NaturalPropagator::at(
                    self,
                    node_epoch(anchor_node),
                    anchor_state,
                    object.body_force,
                )?;
                let offsets = (1..=take)
                    .map(|step| {
                        let node = backoff.checked_add(step).ok_or(
                            NaturalConjunctionFatalError::InvalidInput(
                                NaturalConjunctionInputError::WorkLimit,
                            ),
                        )?;
                        #[expect(
                            clippy::as_conversions,
                            clippy::cast_precision_loss,
                            reason = "usize has no infallible f64 conversion; segment \
                                      node counts are small integers"
                        )]
                        let node = node as f64;
                        Ok(node_step_s * node)
                    })
                    .collect::<Result<Vec<_>, NaturalConjunctionFatalError>>()?;
                states.truncate(before);
                match chain_dense_segment(
                    self,
                    &anchor,
                    object.body_force,
                    &offsets,
                    &mut states,
                    0,
                ) {
                    Ok(()) => break Ok(()),
                    Err(error) => {
                        let recoverable = matches!(
                            &error,
                            NaturalConjunctionFatalError::Propagation(failure)
                                if failure.is_candidate_infeasible_under_valid_authority()
                        );
                        if !recoverable || backoff >= DENSE_ARC_MAX_ANCHOR_BACKOFF_NODES.min(filled)
                        {
                            break Err(NaturalConjunctionFatalError::DenseArcSegment {
                                segment_index,
                                first_node: filled,
                                segment_epoch_jd_utc: node_epoch(anchor_node),
                                anchor_backoff_nodes: backoff,
                                inner: Box::new(error),
                            });
                        }
                        attempt = attempt.saturating_add(1);
                    }
                }
            };
            // A segment the sealed authority cannot propagate is a property of
            // the arc, not a failure of this call: the same instant defeats a
            // single-endpoint propagation from the object's own epoch, which is
            // what exact refinement uses. Return the longest solved prefix and
            // let the caller decide. Authority failures still propagate.
            if let Err(error) = outcome {
                let recoverable = matches!(
                    &error,
                    NaturalConjunctionFatalError::DenseArcSegment { inner, .. }
                        if matches!(
                            inner.as_ref(),
                            NaturalConjunctionFatalError::Propagation(failure)
                                if failure.is_candidate_infeasible_under_valid_authority()
                        )
                );
                if !recoverable {
                    return Err(error);
                }
                states.truncate(before);
                return Ok(states);
            }
            if states.len() != before.saturating_add(take) {
                return Err(NaturalConjunctionFatalError::InvalidInput(
                    NaturalConjunctionInputError::WorkLimit,
                ));
            }
            filled = filled.saturating_add(take);
            segment_index = segment_index.saturating_add(1);
        }
        Ok(states)
    }

    /// Propagate one natural object onto a dense offset grid, once.
    ///
    /// Runs the same sealed strict-HF force authority as
    /// [`Self::refine_natural_conjunction`], under one forward integration for
    /// the whole grid rather than one integration per evaluated offset. It is
    /// a screening cache, not an acceptance authority: nothing it returns is
    /// bound into a verified conjunction digest.
    ///
    /// `offsets_s` must be finite, strictly increasing and non-negative, and
    /// must not exceed the fourteen-day authorized arc.
    ///
    /// # Errors
    ///
    /// Returns an error when the session force authority is not the sealed
    /// Part A strict-HF model, when the object or grid is invalid, or when the
    /// integration fails.
    pub fn natural_dense_ephemeris_grid(
        &self,
        object: &NaturalObjectInput,
        offsets_s: &[f64],
    ) -> Result<Vec<[f64; 6]>, NaturalConjunctionFatalError> {
        validate_session_authority(self)?;
        if !natural_body_force_is_valid(object.body_force) {
            return Err(NaturalConjunctionFatalError::ForceAuthorityMismatch);
        }
        if offsets_s.is_empty()
            || offsets_s
                .first()
                .is_none_or(|first| !first.is_finite() || *first < 0.0)
            || offsets_s
                .last()
                .is_none_or(|last| !last.is_finite() || *last > MAX_ENCLOSURE_S)
            || offsets_s.windows(2).any(|window| {
                let [lower, upper] = window else {
                    return true;
                };
                !upper.is_finite() || *upper <= *lower
            })
        {
            return Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::Enclosure,
            ));
        }
        NaturalPropagator::new(self, object)?.dense_grid(offsets_s)
    }

    /// Refine one candidate over the sealed scan from the objects' own epoch.
    ///
    /// # Errors
    ///
    /// Returns [`NaturalConjunctionFatalError::ForceAuthorityMismatch`] or
    /// [`NaturalConjunctionFatalError::AuthorityUnavailable`] when the session
    /// is not running the sealed Part A strict-HF authority,
    /// [`NaturalConjunctionFatalError::InvalidInput`] for an unusable
    /// enclosure, and [`NaturalConjunctionFatalError::Propagation`] when a
    /// propagation fails under a valid authority.
    pub fn refine_natural_conjunction(
        &self,
        primary: &NaturalObjectInput,
        secondary: &NaturalObjectInput,
        enclosure: NaturalConjunctionEnclosure,
    ) -> Result<NaturalConjunctionOutcome, NaturalConjunctionFatalError> {
        self.refine_natural_conjunction_with_anchor(primary, secondary, enclosure, None)
    }

    /// Refine one candidate with the sealed scan re-based on a cached state.
    ///
    /// [`Self::refine_natural_conjunction`] integrates the sealed scan samples
    /// from the objects' own epoch, so a slab fourteen days out costs two full
    /// fourteen-day arcs per candidate BEFORE any refinement happens. `anchor`
    /// carries a state for both objects at one earlier offset -- in production
    /// the dense screening node the candidate's slab opens on, which the
    /// narrowphase already propagated -- and the scan restarts there instead,
    /// integrating at most `MAX_ANCHOR_LEAD_S + MAX_NON_POINT_ENCLOSURE_S`.
    ///
    /// This moves only WHERE the scan's integration starts. Every offset it
    /// records -- the sample offsets, the refined root, the conjunction epoch,
    /// and every digest input -- stays absolute from the object epoch, so event
    /// identity arithmetic is unchanged in form.
    ///
    /// The anchor is a screening value, not an authority: it is not
    /// bit-identical to the from-epoch propagation of the same instant, and
    /// event bits move. What binds an accepted state is the from-epoch
    /// independent witness below, at `WITNESS_POSITION_KM`, which is computed
    /// from the object epoch whether or not an anchor was supplied.
    ///
    /// # Errors
    ///
    /// As [`Self::refine_natural_conjunction`], plus `InvalidInput(Anchor)`
    /// when the anchor is not finite, opens after the enclosure does, or leads
    /// it by more than `MAX_ANCHOR_LEAD_S`.
    pub fn refine_natural_conjunction_from_scan_anchor(
        &self,
        primary: &NaturalObjectInput,
        secondary: &NaturalObjectInput,
        enclosure: NaturalConjunctionEnclosure,
        anchor: NaturalConjunctionScanAnchor,
    ) -> Result<NaturalConjunctionOutcome, NaturalConjunctionFatalError> {
        self.refine_natural_conjunction_with_anchor(primary, secondary, enclosure, Some(anchor))
    }

    fn refine_natural_conjunction_with_anchor(
        &self,
        primary: &NaturalObjectInput,
        secondary: &NaturalObjectInput,
        enclosure: NaturalConjunctionEnclosure,
        anchor: Option<NaturalConjunctionScanAnchor>,
    ) -> Result<NaturalConjunctionOutcome, NaturalConjunctionFatalError> {
        match self.refine_natural_conjunction_inner(primary, secondary, enclosure, anchor) {
            Err(NaturalConjunctionFatalError::Propagation(failure)) => {
                classify_propagation_failure(failure)
            }
            result => result,
        }
    }

    fn refine_natural_conjunction_inner(
        &self,
        primary: &NaturalObjectInput,
        secondary: &NaturalObjectInput,
        enclosure: NaturalConjunctionEnclosure,
        anchor: Option<NaturalConjunctionScanAnchor>,
    ) -> Result<NaturalConjunctionOutcome, NaturalConjunctionFatalError> {
        validate_request(self, primary, secondary, enclosure)?;
        if let Some(anchor) = anchor {
            validate_scan_anchor(anchor, enclosure)?;
        }
        let primary_propagator = NaturalPropagator::new(self, primary)?;
        let secondary_propagator = NaturalPropagator::new(self, secondary)?;
        let offsets = sealed_sample_offsets(enclosure)?;
        // Both objects are validated to share an epoch, so ONE anchor instant
        // serves both propagators. Only the offsets handed to the propagators
        // become relative to it; `offsets` itself stays absolute and is what
        // every downstream evaluation, root and digest reads.
        let (primary_states, secondary_states) = match anchor {
            None => (
                primary_propagator.grid(&offsets)?,
                secondary_propagator.grid(&offsets)?,
            ),
            Some(anchor) => {
                let anchor_epoch_jd = primary.epoch_jd_utc + anchor.offset_s / SEC_PER_DAY;
                let relative = offsets
                    .iter()
                    .map(|offset| offset - anchor.offset_s)
                    .collect::<Vec<_>>();
                let primary_anchored = NaturalPropagator::at(
                    self,
                    anchor_epoch_jd,
                    anchor.primary_state,
                    primary.body_force,
                )?;
                let secondary_anchored = NaturalPropagator::at(
                    self,
                    anchor_epoch_jd,
                    anchor.secondary_state,
                    secondary.body_force,
                )?;
                (
                    primary_anchored.grid(&relative)?,
                    secondary_anchored.grid(&relative)?,
                )
            }
        };
        let samples = offsets
            .into_iter()
            .zip(primary_states)
            .zip(secondary_states)
            .map(|((offset, primary_state), secondary_state)| {
                evaluation(offset, primary_state, secondary_state)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Refinement probes re-base on the grid sample anchoring the bracket
        // instead of re-integrating from the object epoch, so a probe spans one
        // bracket -- at most a grid step -- instead of up to fourteen days.
        //
        // The state they re-base on is whatever the scan above produced: the
        // from-epoch value when no anchor was supplied, and the anchored scan's
        // value when one was. Either way this introduces no new source of
        // state, it reuses the sample already computed, and fewer steps with
        // one Encke restart accumulate strictly less deviation than the long
        // arc the probe replaces. What binds the accepted state in both cases
        // is the from-epoch independent witness below, at WITNESS_POSITION_KM.
        //
        // Both objects are validated to share an epoch, so one anchor offset
        // serves both propagators.
        let best = refine_every_local_bracket(&samples, |anchor| {
            let anchor_epoch_jd = primary.epoch_jd_utc + anchor.offset_s / SEC_PER_DAY;
            let anchor_offset_s = anchor.offset_s;
            let primary_anchored =
                NaturalPropagator::at(self, anchor_epoch_jd, anchor.primary, primary.body_force)?;
            let secondary_anchored = NaturalPropagator::at(
                self,
                anchor_epoch_jd,
                anchor.secondary,
                secondary.body_force,
            )?;
            Ok(move |offset_s: f64| {
                let span_s = offset_s - anchor_offset_s;
                evaluation(
                    offset_s,
                    primary_anchored.state(span_s)?,
                    secondary_anchored.state(span_s)?,
                )
            })
        })?;
        let miss_distance_km = best.distance_squared.sqrt();
        if miss_distance_km >= UNSAFE_MISS_KM {
            return Ok(NaturalConjunctionOutcome::CandidateInfeasible(
                NaturalConjunctionInfeasible {
                    primary_identity: primary.identity,
                    secondary_identity: secondary.identity,
                    closest_offset_s: best.offset_s,
                    miss_distance_km,
                },
            ));
        }
        let primary_witness = primary_propagator.witness(best.offset_s)?;
        let secondary_witness = secondary_propagator.witness(best.offset_s)?;
        let (primary_position_residual_km, primary_velocity_residual_km_s) =
            residuals(best.primary, primary_witness);
        let (secondary_position_residual_km, secondary_velocity_residual_km_s) =
            residuals(best.secondary, secondary_witness);
        if witness_residual_over_gate(
            primary_position_residual_km,
            secondary_position_residual_km,
            primary_velocity_residual_km_s,
            secondary_velocity_residual_km_s,
        ) {
            return Ok(NaturalConjunctionOutcome::CandidateWitnessResidual(
                NaturalConjunctionWitnessResidual {
                    primary_identity: primary.identity,
                    secondary_identity: secondary.identity,
                    closest_offset_s: best.offset_s,
                    miss_distance_km,
                    primary_position_residual_km,
                    secondary_position_residual_km,
                    primary_velocity_residual_km_s,
                    secondary_velocity_residual_km_s,
                },
            ));
        }
        let primary_force_config = primary_propagator
            .context
            .force_config
            .as_deref()
            .ok_or(NaturalConjunctionFatalError::AuthorityUnavailable)?;
        let secondary_force_config = secondary_propagator
            .context
            .force_config
            .as_deref()
            .ok_or(NaturalConjunctionFatalError::AuthorityUnavailable)?;
        let force_authority_sha256 = combined_force_identity(
            primary_force_config,
            primary.body_force,
            secondary_force_config,
            secondary.body_force,
        );
        let ephemeris_authority_sha256 = ephemeris_identity()?;
        let gravity_identity = self
            .strict_hf_gravity_identity
            .ok_or(NaturalConjunctionFatalError::AuthorityUnavailable)?;
        let gravity_source_sha256 = gravity_identity.source_sha256();
        let gravity_packed_semantic_sha256 = gravity_identity.packed_semantic_sha256();
        let mut verified = VerifiedNaturalConjunction {
            primary_identity: primary.identity,
            secondary_identity: secondary.identity,
            enclosure,
            refined_offset_s: best.offset_s,
            conjunction_jd_utc: primary.epoch_jd_utc + best.offset_s / SEC_PER_DAY,
            miss_distance_km,
            primary_state: best.primary,
            secondary_state: best.secondary,
            primary_independent_witness_state: primary_witness,
            secondary_independent_witness_state: secondary_witness,
            primary_position_residual_km,
            secondary_position_residual_km,
            primary_velocity_residual_km_s,
            secondary_velocity_residual_km_s,
            force_authority_sha256,
            ephemeris_authority_sha256,
            gravity_source_sha256,
            gravity_packed_semantic_sha256,
            digest: [0; 32],
        };
        verified.digest = verified_digest(&verified);
        Ok(NaturalConjunctionOutcome::Verified(Box::new(verified)))
    }
}

/// Fill `offsets` from `propagator`, bisecting the segment on a recoverable
/// propagation failure and restarting the second half from the split state.
///
/// Only a candidate-level propagation failure is retried. An authority failure
/// -- missing assets, a force-model mismatch, an invalid input -- is returned
/// untouched: a shorter segment must never be able to paper over one.
fn chain_dense_segment(
    session: &TransferPostprocessSessionCore,
    propagator: &NaturalPropagator,
    body_force: BodyForceConfig,
    offsets: &[f64],
    states: &mut Vec<[f64; 6]>,
    depth: u32,
) -> Result<(), NaturalConjunctionFatalError> {
    let error = match propagator.dense_grid(offsets) {
        Ok(filled) => {
            states.extend(filled);
            return Ok(());
        }
        Err(error) => error,
    };
    let recoverable = matches!(
        &error,
        NaturalConjunctionFatalError::Propagation(failure)
            if failure.is_candidate_infeasible_under_valid_authority()
    );
    if !recoverable || offsets.len() < 2 || depth >= DENSE_ARC_MAX_SPLIT_DEPTH {
        return Err(error);
    }
    let (head, tail) = offsets.split_at(offsets.len() / 2);
    let split_offset_s = *head
        .last()
        .ok_or(NaturalConjunctionFatalError::InvalidInput(
            NaturalConjunctionInputError::Enclosure,
        ))?;
    chain_dense_segment(
        session,
        propagator,
        body_force,
        head,
        states,
        depth.saturating_add(1),
    )?;
    let carried = *states
        .last()
        .ok_or(NaturalConjunctionFatalError::InvalidInput(
            NaturalConjunctionInputError::WorkLimit,
        ))?;
    let restarted = NaturalPropagator::at(
        session,
        propagator.context.epoch_jd + split_offset_s / SEC_PER_DAY,
        carried,
        body_force,
    )?;
    let shifted = tail
        .iter()
        .map(|offset| offset - split_offset_s)
        .collect::<Vec<_>>();
    chain_dense_segment(
        session,
        &restarted,
        body_force,
        &shifted,
        states,
        depth.saturating_add(1),
    )
}

fn classify_propagation_failure(
    failure: TransferPropagationFailure,
) -> Result<NaturalConjunctionOutcome, NaturalConjunctionFatalError> {
    if failure.is_candidate_infeasible_under_valid_authority() {
        Ok(NaturalConjunctionOutcome::CandidatePropagationInfeasible(
            failure,
        ))
    } else {
        debug_assert!(failure.is_authority_failure());
        Err(NaturalConjunctionFatalError::Propagation(failure))
    }
}

/// Session-level strict-HF authority, shared by pair refinement and the dense
/// ephemeris grid. Both run under the identical sealed force model.
fn validate_session_authority(
    session: &TransferPostprocessSessionCore,
) -> Result<(), NaturalConjunctionFatalError> {
    let authority = StrictHfForceAuthority::PART_A;
    let physics = &session.physics_config;
    // One row per sealed field. The two structs name the same quantities
    // differently -- `sph_order`/`gravity_order`, `atm_model`/`atmosphere_model`,
    // `method`/`integrator_method`, `dt_max`/`dt_max_s` -- so a `&&` chain of
    // these reads like a copy-paste slip on every line. Listing them pairs each
    // config field with its authority field on one line and keeps the check
    // exhaustive. Each row is a pure field comparison, so evaluating all six
    // rather than short-circuiting changes nothing but a few loads.
    let sealed_fields_match = [
        physics.sph_order == authority.gravity_order,
        physics.force_flags == authority.force_flags,
        physics.atm_model == authority.atmosphere_model,
        physics.method == authority.integrator_method,
        physics.dt_max.to_bits() == authority.dt_max_s.to_bits(),
        physics.tolerance.to_bits() == authority.tolerance.to_bits(),
    ];
    if !physics.use_high_fidelity
        || !physics.require_hf_transfer_correction
        || !sealed_fields_match.iter().copied().all(|matched| matched)
    {
        return Err(NaturalConjunctionFatalError::ForceAuthorityMismatch);
    }
    let identity = session
        .strict_hf_gravity_identity
        .ok_or(NaturalConjunctionFatalError::AuthorityUnavailable)?;
    if identity
        != super::canonical_strict_hf_gravity_identity()
            .map_err(|_| NaturalConjunctionFatalError::AuthorityUnavailable)?
    {
        return Err(NaturalConjunctionFatalError::AuthorityUnavailable);
    }
    part_a_ephemeris_authority().map_err(|_| NaturalConjunctionFatalError::AuthorityUnavailable)?;
    Ok(())
}

fn validate_request(
    session: &TransferPostprocessSessionCore,
    primary: &NaturalObjectInput,
    secondary: &NaturalObjectInput,
    enclosure: NaturalConjunctionEnclosure,
) -> Result<(), NaturalConjunctionFatalError> {
    validate_session_authority(session)?;
    if !natural_body_force_is_valid(primary.body_force)
        || !natural_body_force_is_valid(secondary.body_force)
    {
        return Err(NaturalConjunctionFatalError::ForceAuthorityMismatch);
    }
    if primary.identity == secondary.identity {
        return Err(NaturalConjunctionFatalError::InvalidInput(
            NaturalConjunctionInputError::DuplicateObject,
        ));
    }
    if primary.epoch_jd_utc.to_bits() != secondary.epoch_jd_utc.to_bits() {
        return Err(NaturalConjunctionFatalError::InvalidInput(
            NaturalConjunctionInputError::Epoch,
        ));
    }
    if !enclosure.lower_offset_s.is_finite()
        || !enclosure.upper_offset_s.is_finite()
        || enclosure.lower_offset_s < 0.0
        || enclosure.lower_offset_s > enclosure.upper_offset_s
        || enclosure.upper_offset_s > MAX_ENCLOSURE_S
    {
        return Err(NaturalConjunctionFatalError::InvalidInput(
            NaturalConjunctionInputError::Enclosure,
        ));
    }
    if enclosure.lower_offset_s.to_bits() != enclosure.upper_offset_s.to_bits()
        && enclosure.upper_offset_s - enclosure.lower_offset_s > MAX_NON_POINT_ENCLOSURE_S
    {
        return Err(NaturalConjunctionFatalError::InvalidInput(
            NaturalConjunctionInputError::WorkLimit,
        ));
    }
    Ok(())
}

/// Fail closed on an anchor that would let the scan integrate further than
/// the short span its accuracy argument covers.
///
/// The anchor must open at or before the enclosure -- a later anchor would ask
/// the propagator for negative offsets -- and lead it by at most
/// `MAX_ANCHOR_LEAD_S`. A caller that cannot supply such an anchor must use the
/// from-epoch path, never a wider anchor.
fn validate_scan_anchor(
    anchor: NaturalConjunctionScanAnchor,
    enclosure: NaturalConjunctionEnclosure,
) -> Result<(), NaturalConjunctionFatalError> {
    let anchor_error =
        NaturalConjunctionFatalError::InvalidInput(NaturalConjunctionInputError::Anchor);
    if !anchor.offset_s.is_finite()
        || anchor.offset_s < 0.0
        || !all_finite(&anchor.primary_state)
        || !all_finite(&anchor.secondary_state)
    {
        return Err(anchor_error);
    }
    // A NaN or infinite lead falls outside the range, so this single test
    // covers the non-finite cases as well as the ordering and the cap.
    let lead_s = enclosure.lower_offset_s - anchor.offset_s;
    if !(0.0..=MAX_ANCHOR_LEAD_S).contains(&lead_s) {
        return Err(anchor_error);
    }
    Ok(())
}

fn sealed_sample_offsets(
    enclosure: NaturalConjunctionEnclosure,
) -> Result<Vec<f64>, NaturalConjunctionFatalError> {
    if enclosure.lower_offset_s.to_bits() == enclosure.upper_offset_s.to_bits() {
        return Ok(vec![enclosure.lower_offset_s]);
    }
    let sample_interval_count = GRID_INTERVALS;
    let sample_count = sample_interval_count + 1;
    let maximum_local_brackets = sample_interval_count / 2;
    let required_evaluations = maximum_local_brackets
        .checked_mul(REFINE_ITERATIONS + 2)
        .and_then(|refinement| sample_count.checked_add(refinement))
        .ok_or(NaturalConjunctionFatalError::InvalidInput(
            NaturalConjunctionInputError::WorkLimit,
        ))?;
    if required_evaluations > MAX_TOTAL_PAIR_EVALUATIONS {
        return Err(NaturalConjunctionFatalError::InvalidInput(
            NaturalConjunctionInputError::WorkLimit,
        ));
    }
    let step = (enclosure.upper_offset_s - enclosure.lower_offset_s) / 32.0;
    let mut offsets = Vec::with_capacity(sample_count);
    let mut offset = enclosure.lower_offset_s;
    offsets.push(offset);
    for _ in 1..sample_interval_count {
        offset += step;
        offsets.push(offset);
    }
    offsets.push(enclosure.upper_offset_s);
    Ok(offsets)
}

fn evaluation(
    offset_s: f64,
    primary: [f64; 6],
    secondary: [f64; 6],
) -> Result<Evaluation, NaturalConjunctionFatalError> {
    let [primary_x, primary_y, primary_z, ..] = primary;
    let [secondary_x, secondary_y, secondary_z, ..] = secondary;
    let distance_squared = (primary_x - secondary_x).powi(2)
        + (primary_y - secondary_y).powi(2)
        + (primary_z - secondary_z).powi(2);
    if !distance_squared.is_finite() {
        return Err(NaturalConjunctionFatalError::Propagation(
            TransferPropagationFailure::NonFiniteOutput,
        ));
    }
    Ok(Evaluation {
        offset_s,
        primary,
        secondary,
        distance_squared,
    })
}

fn evaluation_order(left: &Evaluation, right: &Evaluation) -> std::cmp::Ordering {
    left.distance_squared
        .total_cmp(&right.distance_squared)
        .then_with(|| left.offset_s.total_cmp(&right.offset_s))
}

/// Relative speed between the two bodies of one evaluation, km/s.
fn relative_speed_km_s(sample: &Evaluation) -> f64 {
    let [_, _, _, pvx, pvy, pvz] = sample.primary;
    let [_, _, _, svx, svy, svz] = sample.secondary;
    ((pvx - svx).powi(2) + (pvy - svy).powi(2) + (pvz - svz).powi(2)).sqrt()
}

/// Can the separation at `sample` fall to `UNSAFE_MISS_KM` within `span_s`?
///
/// Lower bound on the separation over the span is
/// `d0 - v_rel * span - 0.5 * a_max * span^2`. When that is still above the
/// acceptance threshold the interval provably cannot produce an accepted
/// conjunction, so its refinement is pure cost and is skipped.
///
/// This gate exists because endpoint minima are the COMMON case: any candidate
/// whose true closest approach lies in a neighbouring slab has monotone
/// separation across this one. Refining every such endpoint unconditionally
/// would add work to most candidates in the pool.
fn endpoint_span_can_reach_threshold(sample: &Evaluation, span_s: f64) -> bool {
    let span = span_s.abs();
    let floor_km = sample.distance_squared.sqrt()
        - relative_speed_km_s(sample) * span
        - 0.5 * ENDPOINT_GATE_REL_ACCEL_KM_S2 * span * span;
    floor_km <= UNSAFE_MISS_KM
}

/// Refine every local minimum of the sampled grid.
///
/// `make_evaluator` is handed the sample the refinement is ANCHORED on and
/// returns a closure that evaluates absolute offsets. Anchoring lets the caller
/// re-base propagation on a state the grid sweep already computed, so a probe
/// integrates the width of one bracket instead of the whole arc from epoch.
///
/// Interior minima come from `windows(3)`. The two grid ENDPOINTS can never be
/// a `windows(3)` centre, so they are handled explicitly: slabs are fixed and
/// adjacent, which means a true closest approach lying within half a grid step
/// of a slab boundary is an endpoint minimum in BOTH neighbouring slabs. Before
/// this was handled, such a conjunction was reported at its coarse grid value
/// -- kilometres away from the truth -- and typed infeasible in both slabs,
/// losing a real sub-kilometre event. Golden section needs only unimodality on
/// the interval, not an interior seed, so a one-sided refine is well posed.
fn refine_every_local_bracket<F, E>(
    samples: &[Evaluation],
    mut make_evaluator: F,
) -> Result<Evaluation, NaturalConjunctionFatalError>
where
    F: FnMut(&Evaluation) -> Result<E, NaturalConjunctionFatalError>,
    E: FnMut(f64) -> Result<Evaluation, NaturalConjunctionFatalError>,
{
    let mut best = *samples
        .first()
        .ok_or(NaturalConjunctionFatalError::InvalidInput(
            NaturalConjunctionInputError::Enclosure,
        ))?;
    for candidate in samples.iter().copied().skip(1) {
        if evaluation_order(&candidate, &best).is_lt() {
            best = candidate;
        }
    }
    for bracket in samples.windows(3) {
        let [lower, center, upper] = bracket else {
            return Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::Enclosure,
            ));
        };
        if center.distance_squared < lower.distance_squared
            && center.distance_squared < upper.distance_squared
        {
            let mut evaluate = make_evaluator(lower)?;
            let refined = refine_bracket(&mut evaluate, lower.offset_s, upper.offset_s, *center)?;
            if evaluation_order(&refined, &best).is_lt() {
                best = refined;
            }
        }
    }

    // Endpoint minima, gated. Destructured rather than indexed so the bounds
    // are proved by the pattern: a grid with fewer than two samples is the
    // point-enclosure case and has no interval to refine.
    if let ([leading, second, ..], [.., penultimate, trailing]) = (samples, samples) {
        if evaluation_order(leading, &best).is_le()
            && endpoint_span_can_reach_threshold(leading, second.offset_s - leading.offset_s)
        {
            let mut evaluate = make_evaluator(leading)?;
            let refined =
                refine_bracket(&mut evaluate, leading.offset_s, second.offset_s, *leading)?;
            if evaluation_order(&refined, &best).is_lt() {
                best = refined;
            }
        }
        if evaluation_order(trailing, &best).is_le()
            && endpoint_span_can_reach_threshold(trailing, trailing.offset_s - penultimate.offset_s)
        {
            let mut evaluate = make_evaluator(penultimate)?;
            let refined = refine_bracket(
                &mut evaluate,
                penultimate.offset_s,
                trailing.offset_s,
                *trailing,
            )?;
            if evaluation_order(&refined, &best).is_lt() {
                best = refined;
            }
        }
    }
    Ok(best)
}

fn refine_bracket(
    evaluate: &mut impl FnMut(f64) -> Result<Evaluation, NaturalConjunctionFatalError>,
    mut lower: f64,
    mut upper: f64,
    mut best: Evaluation,
) -> Result<Evaluation, NaturalConjunctionFatalError> {
    const GOLDEN: f64 = 0.618_033_988_749_894_9;
    if lower.to_bits() == upper.to_bits() {
        return Ok(best);
    }
    let mut left = evaluate(upper - GOLDEN * (upper - lower))?;
    let mut right = evaluate(lower + GOLDEN * (upper - lower))?;
    for _ in 0..REFINE_ITERATIONS {
        for candidate in [left, right] {
            if evaluation_order(&candidate, &best).is_lt() {
                best = candidate;
            }
        }
        if evaluation_order(&left, &right).is_le() {
            upper = right.offset_s;
            right = left;
            left = evaluate(upper - GOLDEN * (upper - lower))?;
        } else {
            lower = left.offset_s;
            left = right;
            right = evaluate(lower + GOLDEN * (upper - lower))?;
        }
    }
    for candidate in [left, right] {
        if evaluation_order(&candidate, &best).is_lt() {
            best = candidate;
        }
    }
    Ok(best)
}

/// Canonical position residual between a strict-HF state and its independent witness.
///
/// This is the only implementation of the quantity. Downstream revalidation
/// recomputes the residual from the banked states and compares it bit-for-bit
/// against what refinement produced, so a second spelling of the same formula
/// (a `hypot` chain, an FMA contraction, a different association order) is a
/// last-bit disagreement waiting to fire. Call this; do not re-derive it.
#[must_use]
pub fn natural_state_position_residual_km(actual: &[f64; 6], witness: &[f64; 6]) -> f64 {
    ((actual[0] - witness[0]).powi(2)
        + (actual[1] - witness[1]).powi(2)
        + (actual[2] - witness[2]).powi(2))
    .sqrt()
}

/// Canonical velocity residual between a strict-HF state and its independent witness.
///
/// See [`natural_state_position_residual_km`] for why this must stay single-sourced.
#[must_use]
pub fn natural_state_velocity_residual_km_s(actual: &[f64; 6], witness: &[f64; 6]) -> f64 {
    ((actual[3] - witness[3]).powi(2)
        + (actual[4] - witness[4]).powi(2)
        + (actual[5] - witness[5]).powi(2))
    .sqrt()
}

fn residuals(actual: [f64; 6], witness: [f64; 6]) -> (f64, f64) {
    (
        natural_state_position_residual_km(&actual, &witness),
        natural_state_velocity_residual_km_s(&actual, &witness),
    )
}

const fn force_config_matches_body(config: &ForceConfig, body_force: BodyForceConfig) -> bool {
    config.am_ratio.to_bits() == body_force.am_ratio.to_bits()
        && config.cd.to_bits() == body_force.cd.to_bits()
        && config.cr.to_bits() == body_force.cr.to_bits()
}

fn update_actual_force_coefficients(digest: &mut Sha256, config: &ForceConfig) {
    for value in [config.am_ratio, config.cd, config.cr] {
        digest.update(value.to_bits().to_le_bytes());
    }
}

fn global_force_identity(config: &ForceConfig) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"nasa-dust/strict-hf-global-force-authority/v1\0");
    digest.update(config.sph_order.to_le_bytes());
    digest.update(config.force_flags.to_le_bytes());
    digest.update([u8::from(config.subtract_first_order)]);
    digest.update(config.atm_model.to_le_bytes());
    for value in [config.dt_max, config.eps] {
        digest.update(value.to_bits().to_le_bytes());
    }
    digest.update([match config.integrator_method {
        StepperMethod::Dopri5Compat => 0,
        StepperMethod::Tsit5 => 1,
        StepperMethod::Dop853 => 2,
        StepperMethod::Rkv98 => 3,
        StepperMethod::Vern7 => 4,
        StepperMethod::Vern9 => 5,
        StepperMethod::Esdirk43 => 6,
        StepperMethod::Auto => 7,
    }]);
    digest.finalize().into()
}

fn combined_force_identity(
    primary_config: &ForceConfig,
    primary_body_force: BodyForceConfig,
    secondary_config: &ForceConfig,
    secondary_body_force: BodyForceConfig,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"nasa-dust/natural-conjunction-force-authority/v1\0");
    digest.update(global_force_identity(primary_config));
    update_actual_force_coefficients(&mut digest, primary_config);
    update_body_force_identity(&mut digest, primary_body_force);
    digest.update(global_force_identity(secondary_config));
    update_actual_force_coefficients(&mut digest, secondary_config);
    update_body_force_identity(&mut digest, secondary_body_force);
    digest.finalize().into()
}

fn ephemeris_identity() -> Result<[u8; 32], NaturalConjunctionFatalError> {
    let authority = part_a_ephemeris_authority()
        .map_err(|_| NaturalConjunctionFatalError::AuthorityUnavailable)?;
    let mut bundle = Sha256::new();
    for body in [Body::Sun, Body::Moon] {
        let identity = embedded_catalogue_sha256_hex(body)
            .ok_or(NaturalConjunctionFatalError::AuthorityUnavailable)?;
        bundle.update(body.name().as_bytes());
        bundle.update(b"=");
        bundle.update(identity.as_bytes());
        bundle.update(b"\n");
    }
    let bundle_hex = hex(bundle.finalize());
    let mut digest = Sha256::new();
    digest.update(b"nasa-dust/strict-hf-ephemeris-authority/v1\0");
    for record in [authority.manifest_sha256(), bundle_hex.as_str()] {
        digest.update(record.len().to_le_bytes());
        digest.update(record.as_bytes());
    }
    Ok(digest.finalize().into())
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    // Capacity hint only: a saturated value changes no output, just a realloc.
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        output.push(hex_digit(byte >> 4));
        output.push(hex_digit(byte & 0x0f));
    }
    output
}

fn hex_digit(nibble: u8) -> char {
    // Table lookup rather than `b'0' + nibble`: same output for every nibble in
    // 0..=15, same replacement character outside it, and no arithmetic to price.
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    DIGITS
        .get(usize::from(nibble))
        .map_or(char::REPLACEMENT_CHARACTER, |digit| char::from(*digit))
}

fn verified_digest(value: &VerifiedNaturalConjunction) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"nasa-dust/verified-natural-conjunction/v1\0");
    digest.update(value.primary_identity.0);
    digest.update(value.secondary_identity.0);
    for scalar in [
        value.enclosure.lower_offset_s,
        value.enclosure.upper_offset_s,
        value.refined_offset_s,
        value.conjunction_jd_utc,
        value.miss_distance_km,
    ] {
        digest.update(scalar.to_bits().to_le_bytes());
    }
    for scalar in value.primary_state.into_iter().chain(value.secondary_state) {
        digest.update(scalar.to_bits().to_le_bytes());
    }
    for scalar in value
        .primary_independent_witness_state
        .into_iter()
        .chain(value.secondary_independent_witness_state)
    {
        digest.update(scalar.to_bits().to_le_bytes());
    }
    for scalar in [
        value.primary_position_residual_km,
        value.secondary_position_residual_km,
        value.primary_velocity_residual_km_s,
        value.secondary_velocity_residual_km_s,
    ] {
        digest.update(scalar.to_bits().to_le_bytes());
    }
    digest.update(value.force_authority_sha256);
    digest.update(value.ephemeris_authority_sha256);
    digest.update(value.gravity_source_sha256);
    digest.update(value.gravity_packed_semantic_sha256);
    digest.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every arm of the witness gate must be able to fire on its own.
    ///
    /// A four-arm OR passes for three wrong reasons if one arm is the only one
    /// ever driven, so each residual is pushed over its own ceiling in turn
    /// while the other three sit under theirs, and a below-gate control proves
    /// the predicate is not simply always true.
    #[test]
    fn witness_gate_fires_on_each_arm_alone() {
        let under_position = WITNESS_POSITION_KM * 0.5;
        let under_velocity = WITNESS_VELOCITY_KM_S * 0.5;
        let over_position = WITNESS_POSITION_KM * 1.5;
        let over_velocity = WITNESS_VELOCITY_KM_S * 1.5;

        assert!(
            !witness_residual_over_gate(
                under_position,
                under_position,
                under_velocity,
                under_velocity
            ),
            "control: four residuals inside their ceilings must not trip the gate"
        );
        assert!(witness_residual_over_gate(
            over_position,
            under_position,
            under_velocity,
            under_velocity
        ));
        assert!(witness_residual_over_gate(
            under_position,
            over_position,
            under_velocity,
            under_velocity
        ));
        assert!(witness_residual_over_gate(
            under_position,
            under_position,
            over_velocity,
            under_velocity
        ));
        assert!(witness_residual_over_gate(
            under_position,
            under_position,
            under_velocity,
            over_velocity
        ));
    }

    /// The gate value itself is a science choice, not a tuning knob. Pin it so
    /// widening it to make a run finish is a red test rather than a diff nobody
    /// reads.
    #[test]
    fn witness_gate_values_are_pinned() {
        assert!(
            (WITNESS_POSITION_KM - 0.025).abs() < f64::EPSILON,
            "witness position gate moved; it is a science choice, not a knob"
        );
        assert!(
            (WITNESS_VELOCITY_KM_S - 2.0e-5).abs() < f64::EPSILON,
            "witness velocity gate moved; it is a science choice, not a knob"
        );
    }

    /// A witness disagreement is a numerical property of ONE arc, not authority
    /// corruption, so it must not be reachable as a fatal error at all.
    #[test]
    fn no_fatal_variant_carries_a_witness_residual() {
        // Exhaustive: adding a fatal variant forces this match to be revisited,
        // which is the point -- a future witness-shaped fatal would be caught
        // here rather than in a cluster log after thirty minutes of refinement.
        fn is_authority_shaped(error: &NaturalConjunctionFatalError) -> bool {
            match error {
                NaturalConjunctionFatalError::DenseArcSegment { .. }
                | NaturalConjunctionFatalError::InvalidInput(_)
                | NaturalConjunctionFatalError::ForceAuthorityMismatch
                | NaturalConjunctionFatalError::AuthorityUnavailable
                | NaturalConjunctionFatalError::Propagation(_) => true,
            }
        }
        assert!(is_authority_shaped(
            &NaturalConjunctionFatalError::AuthorityUnavailable
        ));
    }

    #[test]
    fn propagation_partition_never_hides_fatal_inputs() {
        use lightyear_odeint_rs::integrator::FinalPropagationFailure as Final;

        assert!(matches!(
            classify_propagation_failure(TransferPropagationFailure::Final(
                Final::IntegrationFailure
            )),
            Ok(NaturalConjunctionOutcome::CandidatePropagationInfeasible(_))
        ));
        assert!(matches!(
            classify_propagation_failure(TransferPropagationFailure::InvalidInput),
            Err(NaturalConjunctionFatalError::Propagation(
                TransferPropagationFailure::InvalidInput
            ))
        ));
        assert!(matches!(
            classify_propagation_failure(TransferPropagationFailure::Authority),
            Err(NaturalConjunctionFatalError::Propagation(
                TransferPropagationFailure::Authority
            ))
        ));
    }

    fn metric(offset_s: f64, distance_squared: f64) -> Evaluation {
        Evaluation {
            offset_s,
            primary: [0.0; 6],
            secondary: [0.0; 6],
            distance_squared,
        }
    }

    fn old_one_best_bracket(
        samples: &[Evaluation],
        evaluate: &mut impl FnMut(f64) -> Result<Evaluation, NaturalConjunctionFatalError>,
    ) -> Result<Evaluation, NaturalConjunctionFatalError> {
        let best = samples.iter().copied().min_by(evaluation_order).ok_or(
            NaturalConjunctionFatalError::InvalidInput(NaturalConjunctionInputError::Enclosure),
        )?;
        for bracket in samples.windows(3) {
            let [lower, center, upper] = bracket else {
                continue;
            };
            if center.offset_s.to_bits() == best.offset_s.to_bits() {
                return refine_bracket(evaluate, lower.offset_s, upper.offset_s, *center);
            }
        }
        Ok(best)
    }

    #[expect(
        clippy::unnecessary_wraps,
        reason = "stands in for the fallible evaluator: passed as \
                  `&mut impl FnMut(f64) -> Result<Evaluation, _>`"
    )]
    fn two_basin(offset_s: f64) -> Result<Evaluation, NaturalConjunctionFatalError> {
        let narrow_lower = 1.0e-4 + ((offset_s - 11.2) / 0.4).powi(2);
        let broad_upper = 1.0e-2 + ((offset_s - 90.0) / 5.0).powi(2);
        Ok(Evaluation {
            offset_s,
            primary: [0.0; 6],
            secondary: [0.0; 6],
            distance_squared: narrow_lower.min(broad_upper),
        })
    }

    #[test]
    fn all_observed_local_brackets_recover_lower_basin_missed_by_one_best_route() {
        let enclosure = NaturalConjunctionEnclosure::new(0.0, 120.0);
        let offsets = sealed_sample_offsets(enclosure).expect("sealed grid");
        assert_eq!(offsets.len(), 33);
        let samples = offsets
            .into_iter()
            .map(two_basin)
            .collect::<Result<Vec<_>, _>>()
            .expect("finite hostile samples");
        let old = old_one_best_bracket(&samples, &mut two_basin)
            .expect("old sampled-best bracket refines");
        assert!(old.offset_s > 60.0, "old route must choose broad basin");

        let refined = refine_every_local_bracket(&samples, |_anchor| Ok(two_basin))
            .expect("all observed local brackets refine");

        assert!(refined.offset_s < 60.0);
        assert!(refined.distance_squared < 1.0e-3);
    }

    #[test]
    fn force_config_body_match_is_bit_exact() {
        let body = BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 0.01, 2.2, 1.3);
        let mut config = ForceConfig {
            am_ratio: body.am_ratio,
            cd: body.cd,
            cr: body.cr,
            ..ForceConfig::default()
        };
        assert!(force_config_matches_body(&config, body));
        config.cr = f64::from_bits(config.cr.to_bits() + 1);
        assert!(!force_config_matches_body(&config, body));
    }

    #[test]
    fn equal_distance_ties_do_not_create_brackets_but_endpoints_refine_when_reachable() {
        // A flat grid has no strict interior minimum, and every sample is the
        // joint best, so the endpoint arms see a separation that already sits
        // at the acceptance threshold. Nothing to refine either way.
        let equal = [metric(0.0, 1.0), metric(1.0, 1.0), metric(2.0, 1.0)];
        let callbacks = std::cell::Cell::new(0_usize);
        let winner = refine_every_local_bracket(&equal, |_anchor| {
            Ok(|offset: f64| {
                callbacks.set(callbacks.get() + 1);
                Ok(metric(offset, 1.0))
            })
        })
        .expect("equal samples classify");
        assert_eq!(winner.offset_s.to_bits(), 0.0_f64.to_bits());

        // An ENDPOINT winner must now be refined, not returned at its coarse
        // grid value. Slabs are fixed and adjacent, so a true closest approach
        // just past a boundary is an endpoint minimum in both neighbours; the
        // old behaviour reported it kilometres away and typed it infeasible in
        // both, losing a real sub-kilometre event.
        let endpoint = [metric(0.0, 2.0), metric(1.0, 1.0), metric(2.0, 0.0)];
        let endpoint_calls = std::cell::Cell::new(0_usize);
        let winner = refine_every_local_bracket(&endpoint, |_anchor| {
            Ok(|offset: f64| {
                endpoint_calls.set(endpoint_calls.get() + 1);
                Ok(metric(offset, 0.0))
            })
        })
        .expect("endpoint samples classify");
        assert_eq!(
            endpoint_calls.get(),
            REFINE_ITERATIONS + 2,
            "trailing endpoint minimum must be refined exactly once"
        );
        // The stub evaluator reports zero separation everywhere, so a probe
        // wins; what matters is that the winner came from inside the endpoint
        // bracket rather than being the coarse grid sample handed back unrefined.
        assert!(
            winner.offset_s > 1.0 && winner.offset_s <= 2.0,
            "endpoint winner must come from the refined bracket, got {}",
            winner.offset_s
        );

        // ... but only when it could actually reach the threshold. A far
        // endpoint is skipped, which is what keeps this off the common path:
        // any candidate whose closest approach lies in another slab has
        // monotone separation here and would otherwise refine for nothing.
        let far = [metric(0.0, 400.0), metric(1.0, 300.0), metric(2.0, 200.0)];
        let far_calls = std::cell::Cell::new(0_usize);
        refine_every_local_bracket(&far, |_anchor| {
            Ok(|offset: f64| {
                far_calls.set(far_calls.get() + 1);
                Ok(metric(offset, 0.0))
            })
        })
        .expect("far samples classify");
        assert_eq!(
            far_calls.get(),
            0,
            "an endpoint that cannot reach UNSAFE_MISS_KM must not be refined"
        );
    }

    #[test]
    fn alternating_grid_reaches_exact_sealed_work_cap() {
        let mut samples = Vec::with_capacity(GRID_INTERVALS + 1);
        let mut offset = 0.0;
        let mut high = true;
        for _ in 0..=GRID_INTERVALS {
            samples.push(metric(offset, if high { 1.0 } else { 0.0 }));
            offset += 1.0;
            high = !high;
        }
        let callbacks = std::cell::Cell::new(0_usize);
        refine_every_local_bracket(&samples, |_anchor| {
            Ok(|offset: f64| {
                callbacks.set(callbacks.get() + 1);
                Ok(metric(offset, 0.0))
            })
        })
        .expect("alternating local brackets refine");
        // Alternating highs put both grid endpoints at the HIGH value, so the
        // endpoint arms never fire here and this measures the interior worst
        // case exactly.
        assert_eq!(
            callbacks.get(),
            MAX_INTERIOR_BRACKETS * (REFINE_ITERATIONS + 2),
            "every interior minimum must refine exactly once"
        );
        assert!(
            samples.len() + callbacks.get() <= MAX_TOTAL_PAIR_EVALUATIONS,
            "interior worst case must fit the sealed budget"
        );
        // The sealed budget must also cover the two endpoint arms on top.
        assert_eq!(
            MAX_TOTAL_PAIR_EVALUATIONS,
            (MAX_INTERIOR_BRACKETS + MAX_ENDPOINT_BRACKETS) * (REFINE_ITERATIONS + 2)
                + GRID_INTERVALS
                + 1,
            "budget must stay derived from the constants that determine it"
        );
    }
}
