//! Shared-target dust-hit uncertainty for the Part A v3 mass lane.
//!
//! CLAIM CLASS: every mass this module issues is the smallest representable
//! effective packet count satisfying the compiled conservative no-hit policy
//! of the named binary64 C12/E14 approximation, with released mass floored by
//! the deterministic requirement. This is a model-conditioned contact
//! requirement under sealed assumptions. It is not an ideal-real Gaussian
//! minimum, a real-world probability calibration, a deflection result, or a
//! mathematical lower bound on required released mass.
//!
//! ASSUMPTION SENSITIVITY, most load-bearing first:
//! 1. `target_position_sigma_m` — knife edge. Production seals 100 m; the
//!    bound moves ~1e5x in mass per 50 m of sigma, and at 200 m the required
//!    contact confidence exceeds the exact binary64 packet-count domain (see
//!    [`Binary64PacketCountUnrepresentable`]).
//! 2. `packet_correlation_grains` — LINEAR in released mass and unverified by
//!    any measurement; production seals 1 (all grains independent).
//! 3. `momentum_coupling_kappa` — inert for this contact bound. It moves
//!    nothing here; it exists only to bind the deterministic-mass evidence
//!    identity. The sealed assumption-id string happens to lead with `kappa`
//!    for historical reasons; that lexical order is NOT the sensitivity order.
//!
//! Target position is one shared random draw for every released packet. It is
//! therefore integrated outside each grain's covariance and outside the
//! conditional packet-failure exponent:
//!
//! `P(no hit) = E_target[(1 - p_conditional(target))^N_eff]`.
//!
//! Part A production binds this solver to opaque deterministic-mass evidence.
//! The historical schema-2 Pc entry point remains separate for qualification
//! and unrelated compatibility tests; it is not a production mass authority.

use anyhow::{anyhow, bail, ensure};

use crate::mass_solver::OperationalDeterministicMass;

/// Sealed V1 ceiling for packet counts admitted to binary64 arithmetic.
pub const MAX_EXACT_BINARY64_PACKET_COUNT: u64 = (1_u64 << 53) - 1;

/// Singular production method identity.
pub const SHARED_TARGET_METHOD_ID: &str = "c12-e14-validated-binary64-log-minimum-v1";
/// Honest public claim identity.
pub const SHARED_TARGET_CLAIM_ID: &str = "model-conditioned-conservative-contact-requirement-v1";
/// Exact machine-count certificate identity.
pub const SHARED_TARGET_COUNT_CERTIFICATE_ID: &str = "binary64-reserved-threshold-minimum-v1";

const MINIMUM_SEPARABLE_ANISOTROPY: f64 = 2.0;
const TARGET_TAIL_ABSOLUTE_ERROR_RESERVE: f64 = 2.0e-8;

/// Inputs for projecting per-grain six-dimensional GMM components onto the
/// encounter B-plane used by the shared-target solver.
///
/// Target uncertainty is deliberately absent. It belongs to
/// [`SharedTargetScenario`] and is integrated once outside the packet-failure
/// exponent.
#[derive(Clone, Copy, Debug)]
pub struct SharedTargetBplaneProjectionInputs<'a> {
    pub component_means_6d: &'a [f64],
    pub component_covariances_6d: &'a [f64],
    pub target_state: &'a [f64],
    pub hf_velocity_mean: &'a [f64],
    pub covariance_minimum: f64,
    pub covariance_maximum: f64,
}

/// Per-grain B-plane projection with no target-covariance convolution.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedTargetBplaneProjection {
    projected_means_2d: Vec<f64>,
    projected_covariances_2d: Vec<f64>,
    projection_clamped: usize,
}

impl SharedTargetBplaneProjection {
    #[must_use]
    pub fn projected_means_2d(&self) -> &[f64] {
        &self.projected_means_2d
    }

    #[must_use]
    pub fn projected_covariances_2d(&self) -> &[f64] {
        &self.projected_covariances_2d
    }

    #[must_use]
    pub const fn projection_clamped(&self) -> usize {
        self.projection_clamped
    }
}

fn project_shared_target_position_components<I>(
    rows: I,
    component_count: usize,
    target_state: &[f64],
    hf_velocity_mean: &[f64],
    covariance_minimum: f64,
    covariance_maximum: f64,
) -> anyhow::Result<SharedTargetBplaneProjection>
where
    I: IntoIterator<Item = anyhow::Result<([f64; 3], [f64; 9])>>,
{
    let target_state = <[f64; 6]>::try_from(target_state)
        .map_err(|_| anyhow!("shared-target target state must contain 6 values"))?;
    ensure!(
        target_state.iter().all(|value| value.is_finite()),
        "shared-target target state must be finite"
    );
    let plane = super::pc_bplane_basis_from_states(hf_velocity_mean, &target_state)?;
    let target_position = [target_state[0], target_state[1], target_state[2]];
    let mut projected_means_2d = Vec::with_capacity(
        component_count
            .checked_mul(2)
            .ok_or_else(|| anyhow!("shared-target projected mean length overflow"))?,
    );
    let mut projected_covariances_2d = Vec::with_capacity(
        component_count
            .checked_mul(4)
            .ok_or_else(|| anyhow!("shared-target projected covariance length overflow"))?,
    );
    let mut projection_clamped = 0usize;

    for row in rows {
        let (position_mean, position_covariance) = row?;
        let delta = [
            position_mean[0] - target_position[0],
            position_mean[1] - target_position[1],
            position_mean[2] - target_position[2],
        ];
        let first = [plane[0], plane[1], plane[2]];
        let second = [plane[3], plane[4], plane[5]];
        projected_means_2d.push(super::dot3(first, delta));
        projected_means_2d.push(super::dot3(second, delta));
        let [raw00, raw01, raw10, raw11] =
            super::project_covariance_to_bplane(position_covariance, plane);
        let (sanitized, clamped) = super::sanitize_covariance_2d_values(
            raw00,
            raw01,
            raw10,
            raw11,
            covariance_minimum,
            covariance_maximum,
        )?;
        projection_clamped = projection_clamped
            .checked_add(usize::from(clamped))
            .ok_or_else(|| anyhow!("shared-target projection clamp count overflow"))?;
        projected_covariances_2d.extend(sanitized);
    }

    Ok(SharedTargetBplaneProjection {
        projected_means_2d,
        projected_covariances_2d,
        projection_clamped,
    })
}

fn position_projection_row(
    state_mean_6d: &[f64],
    state_covariance_6x6: &[f64],
) -> anyhow::Result<([f64; 3], [f64; 9])> {
    let &[mean_x, mean_y, mean_z, ..] = state_mean_6d else {
        anyhow::bail!("shared-target component mean is missing position values");
    };
    let &[cov00, cov01, cov02, _, _, _, cov10, cov11, cov12, _, _, _, cov20, cov21, cov22, ..] =
        state_covariance_6x6
    else {
        anyhow::bail!("shared-target component covariance position block is incomplete");
    };
    Ok((
        [mean_x, mean_y, mean_z],
        [
            cov00, cov01, cov02, cov10, cov11, cov12, cov20, cov21, cov22,
        ],
    ))
}

/// Project only the dust component uncertainty into the encounter B-plane.
///
/// # Errors
///
/// Returns an error for malformed/non-finite rows, degenerate encounter
/// geometry, or a covariance that cannot be sanitized inside the configured
/// eigenvalue bounds.
pub fn project_shared_target_bplane_components(
    inputs: &SharedTargetBplaneProjectionInputs<'_>,
) -> anyhow::Result<SharedTargetBplaneProjection> {
    ensure!(
        inputs.component_means_6d.len().is_multiple_of(6),
        "shared-target component means length must be divisible by 6"
    );
    let component_count = inputs.component_means_6d.len() / 6;
    ensure!(
        component_count > 0,
        "shared-target GMM must contain a component"
    );
    ensure!(
        inputs.component_covariances_6d.len()
            == component_count
                .checked_mul(36)
                .ok_or_else(|| anyhow!("shared-target covariance length overflow"))?,
        "shared-target component covariance length must equal component_count * 36"
    );
    let rows = inputs
        .component_means_6d
        .chunks_exact(6)
        .zip(inputs.component_covariances_6d.chunks_exact(36))
        .map(|(mean, covariance)| {
            ensure!(
                mean.iter().all(|value| value.is_finite())
                    && covariance.iter().all(|value| value.is_finite()),
                "shared-target component row must be finite"
            );
            position_projection_row(mean, covariance)
        });
    project_shared_target_position_components(
        rows,
        component_count,
        inputs.target_state,
        inputs.hf_velocity_mean,
        inputs.covariance_minimum,
        inputs.covariance_maximum,
    )
}

/// Non-forgeable identity sealed with a shared-target dust-mass scenario.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DustScenarioIdentity {
    assumption_id: &'static str,
}

impl DustScenarioIdentity {
    /// Creates a named assumption set. Exact physical/numerical content lives
    /// beside this identifier in [`SharedTargetScenario`].
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, non-canonical, or reserved identifier.
    pub fn named(identifier: &'static str) -> anyhow::Result<Self> {
        ensure!(
            !identifier.is_empty() && identifier.trim() == identifier,
            "named dust scenario identity must be non-empty and canonical"
        );
        Ok(Self {
            assumption_id: identifier,
        })
    }

    /// Stable assumption identifier for receipts and result labels.
    #[must_use]
    pub const fn assumption_id(self) -> &'static str {
        self.assumption_id
    }
}

/// Claim class carried by every result from this lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DustMassClaim {
    /// Model-conditioned conservative released-mass requirement for
    /// >=1-grain contact with the target at the scenario confidence.
    ///
    /// NOT a deflection or momentum-delivery bound: the capture fraction is
    /// 1.1e-5 to 5.3e-5 on the production corpus, so momentum delivery would
    /// need ~7e5 to ~2.2e8 times this mass.
    ///
    /// The confidence is NOT a property of this claim — it is the scenario's
    /// `target_hit_probability` (0.99 in production science, but other values
    /// are set by callers and tests). Any report rendering this claim must
    /// read the confidence from
    /// [`SharedTargetMassEstimate::target_hit_probability`] rather than
    /// restating a literal.
    ModelConditionedConservativeContactRequirement,
}

impl DustMassClaim {
    /// Serialized claim token.
    ///
    /// SEALED: this exact string is streamed into H64 evidence receipts and
    /// mirrors `nd_config::PartASharedTargetClaim`, whose serde variant name
    /// and hash tag are folded into the golden science SHA-256. Changing the
    /// token moves sealed digests; the contact-bound relabel (2026-08-18) is
    /// therefore doc-level on the variant itself, not in the token.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelConditionedConservativeContactRequirement => SHARED_TARGET_CLAIM_ID,
        }
    }
}

/// Typed treatment of one shared target-position draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedTargetPositionTreatment {
    /// Isotropic Gaussian uncertainty in the encounter B-plane. No RIC
    /// radial/cross-track axis ratios enter this model.
    AssumedNotObservedIsotropicEncounterBplaneSharedDraw,
}

impl SharedTargetPositionTreatment {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AssumedNotObservedIsotropicEncounterBplaneSharedDraw => {
                "assumed-not-observed-isotropic-encounter-bplane-shared-draw"
            }
        }
    }
}

/// Typed identity and physical assumptions for one shared-target scenario.
///
/// Assumption sensitivity, most load-bearing first (see the module doc):
/// `target_position_sigma_m` (knife edge: ~1e5x mass per 50 m, beyond the
/// finite `u64` packet-count domain at 200 m), then
/// `packet_correlation_grains` (linear in mass, unverified),
/// then `momentum_coupling_kappa` last (inert for the contact bound; it only
/// binds deterministic-mass evidence identity). Field order below is the
/// historical constructor order, not the sensitivity order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedTargetScenario {
    identity: DustScenarioIdentity,
    momentum_coupling_kappa: f64,
    packet_correlation_grains: u64,
    target_position_sigma_m: f64,
    target_position_treatment: SharedTargetPositionTreatment,
    quadrature: SharedTargetQuadrature,
    claim: DustMassClaim,
}

impl SharedTargetScenario {
    /// Builds a scenario while enforcing its reportable parameter domain.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty identity, coupling outside `(0, 1]`, zero
    /// packet correlation, invalid target sigma, or misuse of the reserved
    /// optimistic-baseline identity.
    pub fn new(
        identity: DustScenarioIdentity,
        momentum_coupling_kappa: f64,
        packet_correlation_grains: u64,
        target_position_sigma_m: f64,
        target_position_treatment: SharedTargetPositionTreatment,
        quadrature: SharedTargetQuadrature,
        claim: DustMassClaim,
    ) -> anyhow::Result<Self> {
        ensure!(
            momentum_coupling_kappa.is_finite()
                && momentum_coupling_kappa > 0.0
                && momentum_coupling_kappa <= 1.0,
            "kappa must lie in (0, 1], got {momentum_coupling_kappa}"
        );
        ensure!(
            (1..=MAX_EXACT_BINARY64_PACKET_COUNT).contains(&packet_correlation_grains),
            "packet correlation must lie in the exact binary64 integer domain"
        );
        ensure!(
            target_position_sigma_m.is_finite() && target_position_sigma_m > 0.0,
            "target position sigma must be finite and positive"
        );
        let target_sigma_km = target_position_sigma_m * 1.0e-3;
        let target_variance_km2 = target_sigma_km * target_sigma_km;
        ensure!(
            target_variance_km2.is_finite() && target_variance_km2 > 0.0,
            "target position sigma does not produce finite positive covariance"
        );
        Ok(Self {
            identity,
            momentum_coupling_kappa,
            packet_correlation_grains,
            target_position_sigma_m,
            target_position_treatment,
            quadrature,
            claim,
        })
    }

    #[must_use]
    pub const fn identity(self) -> DustScenarioIdentity {
        self.identity
    }

    #[must_use]
    pub const fn assumption_id(self) -> &'static str {
        self.identity.assumption_id()
    }

    #[must_use]
    pub const fn momentum_coupling_kappa(self) -> f64 {
        self.momentum_coupling_kappa
    }

    #[must_use]
    pub const fn packet_correlation_grains(self) -> u64 {
        self.packet_correlation_grains
    }

    #[must_use]
    pub const fn target_position_sigma_m(self) -> f64 {
        self.target_position_sigma_m
    }

    #[must_use]
    pub const fn target_position_treatment(self) -> SharedTargetPositionTreatment {
        self.target_position_treatment
    }

    #[must_use]
    pub const fn quadrature(self) -> SharedTargetQuadrature {
        self.quadrature
    }

    #[must_use]
    pub const fn content_identity(
        self,
        target_hit_probability: f64,
    ) -> SharedTargetScenarioContentIdentity {
        SharedTargetScenarioContentIdentity {
            identity: self.identity,
            momentum_coupling_kappa_bits: self.momentum_coupling_kappa.to_bits(),
            packet_correlation_grains: self.packet_correlation_grains,
            target_position_sigma_m_bits: self.target_position_sigma_m.to_bits(),
            target_position_treatment: self.target_position_treatment,
            quadrature: self.quadrature,
            claim: self.claim,
            target_hit_probability_bits: target_hit_probability.to_bits(),
        }
    }

    /// Isotropic B-plane covariance derived from the scenario sigma.
    #[must_use]
    pub const fn target_covariance_2d_km2(self) -> [f64; 4] {
        match self.target_position_treatment {
            SharedTargetPositionTreatment::AssumedNotObservedIsotropicEncounterBplaneSharedDraw => {
                let sigma_km = self.target_position_sigma_m * 1.0e-3;
                let variance_km2 = sigma_km * sigma_km;
                [variance_km2, 0.0, 0.0, variance_km2]
            }
        }
    }

    /// Binds operational deterministic mass to this scenario's exact kappa.
    ///
    /// # Errors
    ///
    /// Returns an error when the deterministic solve used another kappa.
    pub fn bind_deterministic_mass(
        self,
        operational_mass: OperationalDeterministicMass,
    ) -> anyhow::Result<ScenarioBoundDeterministicMass> {
        ensure!(
            self.momentum_coupling_kappa.to_bits()
                == operational_mass.momentum_coupling_kappa().to_bits(),
            "deterministic mass kappa {} does not match scenario kappa {}",
            operational_mass.momentum_coupling_kappa(),
            self.momentum_coupling_kappa
        );
        Ok(ScenarioBoundDeterministicMass {
            scenario: self,
            operational_mass,
        })
    }

    #[must_use]
    pub const fn claim(self) -> DustMassClaim {
        self.claim
    }
}

/// Operational deterministic mass proven to use the enclosing scenario's kappa.
///
/// Raw solver evidence remains immutable inside the opaque operational token;
/// shared-target physics consumes only its commanded mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScenarioBoundDeterministicMass {
    scenario: SharedTargetScenario,
    operational_mass: OperationalDeterministicMass,
}

impl ScenarioBoundDeterministicMass {
    #[must_use]
    pub const fn scenario(self) -> SharedTargetScenario {
        self.scenario
    }

    /// Commanded operational mass, including the compiled practical floor.
    #[must_use]
    pub const fn required_mass_kg(self) -> f64 {
        self.operational_mass.commanded_required_mass_kg()
    }

    #[must_use]
    pub const fn mass_authority_id(self) -> &'static str {
        self.operational_mass.mass_authority_id()
    }
}

/// Singular solver identity for integrating the shared Gaussian target draw.
pub const SHARED_TARGET_DRAW_INTEGRATION_ID: &str =
    "rotated-cartesian-bounded-radial-refinement-polar-below-2-v2";

/// Deterministic numerical authority for the shared target draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SharedTargetQuadrature {
    target_radial_samples: usize,
    target_angular_samples: usize,
    convergence_tolerance_bits: u64,
}

impl SharedTargetQuadrature {
    /// Creates an even fine grid; convergence is checked against its half grid.
    ///
    /// # Errors
    ///
    /// Returns an error for unusable counts or tolerance.
    pub fn new(
        target_radial_samples: usize,
        target_angular_samples: usize,
        convergence_tolerance: f64,
    ) -> anyhow::Result<Self> {
        ensure!(
            [target_radial_samples, target_angular_samples]
                .into_iter()
                .all(|count| count >= 8 && count.is_multiple_of(2)),
            "fine target quadrature counts must be even and at least eight"
        );
        ensure!(
            [target_radial_samples, target_angular_samples]
                .into_iter()
                .all(|count| u32::try_from(count).is_ok()),
            "quadrature sample count exceeds u32"
        );
        ensure!(
            convergence_tolerance.is_finite()
                && convergence_tolerance > 0.0
                && convergence_tolerance < 1.0,
            "quadrature convergence tolerance must lie in (0, 1)"
        );
        Ok(Self {
            target_radial_samples,
            target_angular_samples,
            convergence_tolerance_bits: convergence_tolerance.to_bits(),
        })
    }

    #[must_use]
    pub const fn target_integration_id(self) -> &'static str {
        SHARED_TARGET_DRAW_INTEGRATION_ID
    }

    #[must_use]
    pub const fn target_radial_samples(self) -> usize {
        self.target_radial_samples
    }

    #[must_use]
    pub const fn target_angular_samples(self) -> usize {
        self.target_angular_samples
    }

    #[must_use]
    pub const fn convergence_tolerance(self) -> f64 {
        f64::from_bits(self.convergence_tolerance_bits)
    }

    /// No-hit probability available to the discrete solve after reserving both
    /// numerical axes and the bound six-sigma truncation error.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested hit probability leaves no positive
    /// numerical budget.
    pub fn conservative_failure_probability(
        self,
        target_hit_probability: f64,
    ) -> anyhow::Result<f64> {
        let raw_no_hit = 1.0 - target_hit_probability;
        let error_reserve = 2.0 * self.convergence_tolerance() + TARGET_TAIL_ABSOLUTE_ERROR_RESERVE;
        let policy_threshold = raw_no_hit - error_reserve;
        ensure!(
            policy_threshold.is_finite() && policy_threshold > 0.0 && policy_threshold < 1.0,
            "shared-target hit probability leaves no positive quadrature error budget"
        );
        Ok(policy_threshold)
    }
}

/// Exact scenario identity used by receipts and numerical witnesses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SharedTargetScenarioContentIdentity {
    identity: DustScenarioIdentity,
    momentum_coupling_kappa_bits: u64,
    packet_correlation_grains: u64,
    target_position_sigma_m_bits: u64,
    target_position_treatment: SharedTargetPositionTreatment,
    quadrature: SharedTargetQuadrature,
    claim: DustMassClaim,
    target_hit_probability_bits: u64,
}

impl SharedTargetScenarioContentIdentity {
    #[must_use]
    pub const fn assumption_id(self) -> &'static str {
        self.identity.assumption_id()
    }

    #[must_use]
    pub const fn momentum_coupling_kappa_bits(self) -> u64 {
        self.momentum_coupling_kappa_bits
    }

    #[must_use]
    pub const fn packet_correlation_grains(self) -> u64 {
        self.packet_correlation_grains
    }

    #[must_use]
    pub const fn target_position_sigma_m_bits(self) -> u64 {
        self.target_position_sigma_m_bits
    }

    #[must_use]
    pub const fn target_position_treatment(self) -> SharedTargetPositionTreatment {
        self.target_position_treatment
    }

    #[must_use]
    pub const fn quadrature(self) -> SharedTargetQuadrature {
        self.quadrature
    }

    #[must_use]
    pub const fn claim(self) -> DustMassClaim {
        self.claim
    }

    #[must_use]
    pub const fn target_hit_probability_bits(self) -> u64 {
        self.target_hit_probability_bits
    }
}

/// Inputs for one projected shared-target mass row.
///
/// Component means and covariances describe per-grain dust uncertainty only.
/// Target covariance is derived from `deterministic_mass.scenario()` and is
/// sampled once per scenario realization, outside the packet exponent.
#[derive(Clone, Copy, Debug)]
pub struct SharedTargetMassInputs<'a> {
    pub deterministic_mass: ScenarioBoundDeterministicMass,
    pub projected_means_2d: &'a [f64],
    pub projected_covariances_2d: &'a [f64],
    pub mixture_weights: &'a [f64],
    pub area_km2: f64,
    pub target_hit_probability: f64,
    pub grain_mass_kg: f64,
    pub covariance_minimum: f64,
    pub covariance_maximum: f64,
}

/// Everything the C12 model reads, with the deterministic limb as plain data.
///
/// This is the shape a REPLAY can supply. `SharedTargetMassInputs` carries an
/// unforgeable `ScenarioBoundDeterministicMass`, which a persisted record
/// cannot reconstruct and must not be able to. But the deterministic limb
/// contributes exactly two things to the model -- the scenario and the required
/// mass -- and both are separately witnessed in the persisted record and
/// checked there. So replay recomputes the probability, count, convergence and
/// refinement limbs from raw inputs without re-deriving, or forging, the
/// deterministic evidence.
#[derive(Clone, Copy)]
pub struct SharedTargetReplayInputs<'a> {
    /// The compiled scenario the witness claims it was solved under.
    pub scenario: SharedTargetScenario,
    /// The separately witnessed deterministic mass requirement, in kg.
    pub deterministic_required_mass_kg: f64,
    pub projected_means_2d: &'a [f64],
    pub projected_covariances_2d: &'a [f64],
    pub mixture_weights: &'a [f64],
    pub area_km2: f64,
    pub target_hit_probability: f64,
    pub grain_mass_kg: f64,
    pub covariance_minimum: f64,
    pub covariance_maximum: f64,
}

/// One pure replay result: every quantity a persisted verifier bit-compares.
///
/// Non-minting by construction -- it carries no token and no authority, so
/// obtaining one proves nothing on its own. Only the comparison does.
#[derive(Clone, Copy)]
pub struct SharedTargetReplay {
    /// The recomputed C12 witness.
    pub witness: C12Binary64LogMinimumWitnessV1,
    /// The recomputed released mass, in kg.
    pub release_mass_kg: f64,
    /// The recomputed selected no-hit probability.
    pub no_hit_probability: f64,
    /// The recomputed probability-limb predecessor no-hit probability.
    pub probability_predecessor_no_hit_probability: f64,
    /// The recomputed expected conditional capture probability.
    pub expected_conditional_capture_probability: f64,
}

/// Which exact limb selected the final effective packet count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedTargetPacketCountGovernor {
    Probability,
    DeterministicFloor,
}

impl SharedTargetPacketCountGovernor {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Probability => "probability",
            Self::DeterministicFloor => "deterministic-floor",
        }
    }
}

/// Why a required packet count cannot be represented exactly in binary64.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Binary64PacketCountUnrepresentableReason {
    ProbabilityThreshold {
        best_attained_log_no_hit_bits: u64,
        policy_threshold_log_bits: u64,
    },
    DeterministicFloor {
        requested_count: u64,
    },
}

/// Typed boundary error for the sealed exact binary64 integer domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Binary64PacketCountUnrepresentable {
    maximum_exact_count: u64,
    reason: Binary64PacketCountUnrepresentableReason,
}

impl Binary64PacketCountUnrepresentable {
    const fn probability_threshold(best_log: f64, threshold_log: f64) -> Self {
        Self {
            maximum_exact_count: MAX_EXACT_BINARY64_PACKET_COUNT,
            reason: Binary64PacketCountUnrepresentableReason::ProbabilityThreshold {
                best_attained_log_no_hit_bits: best_log.to_bits(),
                policy_threshold_log_bits: threshold_log.to_bits(),
            },
        }
    }

    const fn deterministic_floor(requested_count: u64) -> Self {
        Self {
            maximum_exact_count: MAX_EXACT_BINARY64_PACKET_COUNT,
            reason: Binary64PacketCountUnrepresentableReason::DeterministicFloor {
                requested_count,
            },
        }
    }

    #[must_use]
    pub const fn maximum_exact_count(self) -> u64 {
        self.maximum_exact_count
    }

    #[must_use]
    pub const fn reason(self) -> Binary64PacketCountUnrepresentableReason {
        self.reason
    }
}

impl std::fmt::Display for Binary64PacketCountUnrepresentable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.reason {
            Binary64PacketCountUnrepresentableReason::ProbabilityThreshold {
                best_attained_log_no_hit_bits,
                policy_threshold_log_bits,
            } => write!(
                formatter,
                "binary64 threshold minimum exceeds exact count {}: best_log_bits={best_attained_log_no_hit_bits:016x}, threshold_log_bits={policy_threshold_log_bits:016x}",
                self.maximum_exact_count
            ),
            Binary64PacketCountUnrepresentableReason::DeterministicFloor { requested_count } => {
                write!(
                    formatter,
                    "deterministic packet floor {requested_count} exceeds exact binary64 count {}",
                    self.maximum_exact_count
                )
            }
        }
    }
}

impl std::error::Error for Binary64PacketCountUnrepresentable {}

/// Singular machine-policy witness for the production C12/log-min route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct C12Binary64LogMinimumWitnessV1 {
    scenario_content_identity: SharedTargetScenarioContentIdentity,
    policy_threshold_probability_bits: u64,
    policy_threshold_log_bits: u64,
    probability_packet_count: u64,
    deterministic_floor_packet_count: u64,
    final_packet_count: u64,
    governor: SharedTargetPacketCountGovernor,
    selected_log_no_hit_bits: u64,
    probability_predecessor_log_no_hit_bits: u64,
    governed_predecessor_log_no_hit_bits: Option<u64>,
    target_normalization_log_bits: u64,
    base_target_quadrature_delta_bits: u64,
    target_quadrature_delta_bits: u64,
    target_refinement_level: u8,
    target_radial_samples: usize,
    target_angular_samples: usize,
    maximum_disk_scale_bits: u64,
    maximum_fourth_order_bits: u64,
    maximum_e14_indicator_bits: u64,
    c12_component_evaluations: u64,
}

impl C12Binary64LogMinimumWitnessV1 {
    #[must_use]
    pub const fn method_id(self) -> &'static str {
        SHARED_TARGET_METHOD_ID
    }

    #[must_use]
    pub const fn claim_id(self) -> &'static str {
        SHARED_TARGET_CLAIM_ID
    }

    #[must_use]
    pub const fn count_certificate_id(self) -> &'static str {
        SHARED_TARGET_COUNT_CERTIFICATE_ID
    }

    #[must_use]
    pub const fn scenario_content_identity(self) -> SharedTargetScenarioContentIdentity {
        self.scenario_content_identity
    }

    #[must_use]
    pub const fn policy_threshold_probability_bits(self) -> u64 {
        self.policy_threshold_probability_bits
    }

    #[must_use]
    pub const fn policy_threshold_log_bits(self) -> u64 {
        self.policy_threshold_log_bits
    }

    #[must_use]
    pub const fn probability_packet_count(self) -> u64 {
        self.probability_packet_count
    }

    #[must_use]
    pub const fn deterministic_floor_packet_count(self) -> u64 {
        self.deterministic_floor_packet_count
    }

    #[must_use]
    pub const fn final_packet_count(self) -> u64 {
        self.final_packet_count
    }

    #[must_use]
    pub const fn governor(self) -> SharedTargetPacketCountGovernor {
        self.governor
    }

    #[must_use]
    pub const fn selected_log_no_hit_bits(self) -> u64 {
        self.selected_log_no_hit_bits
    }

    #[must_use]
    pub const fn probability_predecessor_log_no_hit_bits(self) -> u64 {
        self.probability_predecessor_log_no_hit_bits
    }

    #[must_use]
    pub const fn governed_predecessor_log_no_hit_bits(self) -> Option<u64> {
        self.governed_predecessor_log_no_hit_bits
    }

    #[must_use]
    pub const fn target_normalization_log_bits(self) -> u64 {
        self.target_normalization_log_bits
    }

    #[must_use]
    pub const fn base_target_quadrature_delta(self) -> f64 {
        f64::from_bits(self.base_target_quadrature_delta_bits)
    }

    #[must_use]
    pub const fn target_quadrature_delta(self) -> f64 {
        f64::from_bits(self.target_quadrature_delta_bits)
    }

    #[must_use]
    pub const fn target_refinement_level(self) -> u8 {
        self.target_refinement_level
    }

    #[must_use]
    pub const fn target_radial_samples(self) -> usize {
        self.target_radial_samples
    }

    #[must_use]
    pub const fn target_angular_samples(self) -> usize {
        self.target_angular_samples
    }

    #[must_use]
    pub const fn maximum_disk_scale(self) -> f64 {
        f64::from_bits(self.maximum_disk_scale_bits)
    }

    #[must_use]
    pub const fn maximum_fourth_order(self) -> f64 {
        f64::from_bits(self.maximum_fourth_order_bits)
    }

    #[must_use]
    pub const fn maximum_e14_indicator(self) -> f64 {
        f64::from_bits(self.maximum_e14_indicator_bits)
    }

    #[must_use]
    pub const fn c12_component_evaluations(self) -> u64 {
        self.c12_component_evaluations
    }
}

/// Model-conditioned conservative contact-mass requirement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedTargetMassEstimate {
    scenario: SharedTargetScenario,
    target_area_m2: f64,
    target_disk_radius_m: f64,
    target_hit_probability: f64,
    grain_mass_kg: f64,
    deterministic_required_mass_kg: f64,
    deterministic_mass_authority_id: &'static str,
    release_mass_kg: f64,
    effective_packet_count: u64,
    no_hit_probability: f64,
    probability_predecessor_no_hit_probability: f64,
    expected_conditional_capture_probability: f64,
    witness: C12Binary64LogMinimumWitnessV1,
}

impl Eq for SharedTargetMassEstimate {}

impl SharedTargetMassEstimate {
    #[must_use]
    pub const fn scenario(self) -> SharedTargetScenario {
        self.scenario
    }
    #[must_use]
    pub const fn claim(self) -> DustMassClaim {
        self.scenario.claim()
    }
    #[must_use]
    pub const fn target_area_m2(self) -> f64 {
        self.target_area_m2
    }
    #[must_use]
    pub const fn target_disk_radius_m(self) -> f64 {
        self.target_disk_radius_m
    }
    #[must_use]
    pub const fn target_hit_probability(self) -> f64 {
        self.target_hit_probability
    }
    #[must_use]
    pub const fn grain_mass_kg(self) -> f64 {
        self.grain_mass_kg
    }
    #[must_use]
    pub const fn deterministic_required_mass_kg(self) -> f64 {
        self.deterministic_required_mass_kg
    }
    #[must_use]
    pub const fn deterministic_mass_authority_id(self) -> &'static str {
        self.deterministic_mass_authority_id
    }
    #[must_use]
    pub const fn release_mass_kg(self) -> f64 {
        self.release_mass_kg
    }
    #[must_use]
    pub const fn effective_packet_count(self) -> u64 {
        self.effective_packet_count
    }
    #[must_use]
    pub const fn no_hit_probability(self) -> f64 {
        self.no_hit_probability
    }
    #[must_use]
    pub const fn probability_predecessor_no_hit_probability(self) -> f64 {
        self.probability_predecessor_no_hit_probability
    }
    #[must_use]
    pub const fn expected_conditional_capture_probability(self) -> f64 {
        self.expected_conditional_capture_probability
    }
    #[must_use]
    pub const fn witness(self) -> C12Binary64LogMinimumWitnessV1 {
        self.witness
    }
}

/// Inputs for a non-authoritative conditional-capture source.
pub struct SharedTargetConditionalCaptureSourceInputs<'a> {
    pub component_means_6d: &'a [f64],
    pub component_covariances_6d: &'a [f64],
    pub mixture_weights: &'a [f64],
    pub hf_velocity_mean: [f64; 3],
    pub area_km2: f64,
    pub released_mass_kg: f64,
    pub covariance_minimum: f64,
    pub covariance_maximum: f64,
}

/// Retained position block derived from one validated full-state GMM component.
///
/// This is numerical input, not persisted evidence or trajectory authority.
struct ConditionalSourceComponent {
    position_mean: [f64; 3],
    position_covariance: [f64; 9],
    log_weight: f64,
}

pub struct PreparedConditionalCaptureSource {
    components: Vec<ConditionalSourceComponent>,
    hf_velocity_mean: [f64; 3],
    area_km2: f64,
    released_mass_kg_bits: u64,
    covariance_minimum: f64,
    covariance_maximum: f64,
}

impl PreparedConditionalCaptureSource {
    /// Reproject retained position blocks using fixed HF dust velocity and one target draw.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid encounter geometry or C12/probability/mass.
    pub fn evaluate_target_state(
        &self,
        sampled_target_state_gcrs_km_km_s: [f64; 6],
    ) -> anyhow::Result<ConditionalCaptureEstimate> {
        let (model, projection_clamped) =
            prepare_conditional_mixture_for_target_state(self, sampled_target_state_gcrs_km_km_s)?;
        let capture = conditional_mixture_capture_c12(&model, 0.0, 0.0)?;
        let probability = probability_from_log(capture.log_probability, "conditional capture")?;
        let expected_mass = conditional_expected_hit_mass_kg(
            f64::from_bits(self.released_mass_kg_bits),
            probability,
        )?;
        Ok(ConditionalCaptureEstimate {
            projection_clamped,
            conditional_capture_probability_bits: probability.to_bits(),
            conditional_expected_hit_mass_kg_bits: expected_mass.to_bits(),
            maximum_disk_scale_bits: capture.maximum_disk_scale.to_bits(),
            maximum_fourth_order_bits: capture.maximum_fourth_order.to_bits(),
            maximum_e14_indicator_bits: capture.maximum_e14_indicator.to_bits(),
        })
    }
}

/// Validate a full-state GMM, then retain its position blocks and fixed HF dust velocity.
///
/// # Errors
///
/// Returns an error for malformed/non-finite GMM data or invalid fixed inputs.
pub fn prepare_shared_target_conditional_capture_source(
    inputs: &SharedTargetConditionalCaptureSourceInputs<'_>,
) -> anyhow::Result<PreparedConditionalCaptureSource> {
    ensure!(
        inputs.component_means_6d.len().is_multiple_of(6),
        "shared-target component means length must be divisible by 6"
    );
    let component_count = inputs.component_means_6d.len() / 6;
    ensure!(
        component_count > 0,
        "shared-target GMM must contain a component"
    );
    ensure!(
        inputs.component_covariances_6d.len()
            == component_count
                .checked_mul(36)
                .ok_or_else(|| anyhow!("shared-target covariance length overflow"))?
            && inputs.mixture_weights.len() == component_count,
        "shared-target raw GMM arrays have inconsistent lengths"
    );
    ensure!(
        inputs
            .component_means_6d
            .iter()
            .all(|value| value.is_finite())
            && inputs
                .component_covariances_6d
                .iter()
                .all(|value| value.is_finite())
            && inputs.hf_velocity_mean.into_iter().all(f64::is_finite),
        "shared-target raw GMM and HF velocity must be finite"
    );
    let weight_sum = inputs.mixture_weights.iter().sum::<f64>();
    ensure!(
        inputs
            .mixture_weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0)
            && weight_sum.is_finite()
            && weight_sum > 0.0,
        "shared-target mixture weights are invalid"
    );
    ensure!(
        inputs.area_km2.is_finite() && inputs.area_km2 > 0.0,
        "shared-target area must be finite and positive"
    );
    ensure!(
        inputs.released_mass_kg.is_finite() && inputs.released_mass_kg >= 0.0,
        "released mass must be finite and nonnegative"
    );
    ensure!(
        inputs.covariance_minimum.is_finite()
            && inputs.covariance_minimum > 0.0
            && inputs.covariance_maximum.is_finite()
            && inputs.covariance_maximum >= inputs.covariance_minimum,
        "shared-target covariance bounds are invalid"
    );
    let mut components = Vec::with_capacity(component_count);
    for ((mean, covariance), &weight) in inputs
        .component_means_6d
        .chunks_exact(6)
        .zip(inputs.component_covariances_6d.chunks_exact(36))
        .zip(inputs.mixture_weights)
    {
        if weight == 0.0 {
            continue;
        }
        let (position_mean, position_covariance) = position_projection_row(mean, covariance)?;
        components.push(ConditionalSourceComponent {
            position_mean,
            position_covariance,
            log_weight: (weight / weight_sum).ln(),
        });
    }
    Ok(PreparedConditionalCaptureSource {
        components,
        hf_velocity_mean: inputs.hf_velocity_mean,
        area_km2: inputs.area_km2,
        released_mass_kg_bits: inputs.released_mass_kg.to_bits(),
        covariance_minimum: inputs.covariance_minimum,
        covariance_maximum: inputs.covariance_maximum,
    })
}

/// Non-authoritative conditional probability and expected mass at one draw.
///
/// Expected mass is not realized captured mass and grants no trajectory claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditionalCaptureEstimate {
    projection_clamped: usize,
    conditional_capture_probability_bits: u64,
    conditional_expected_hit_mass_kg_bits: u64,
    maximum_disk_scale_bits: u64,
    maximum_fourth_order_bits: u64,
    maximum_e14_indicator_bits: u64,
}

impl ConditionalCaptureEstimate {
    #[must_use]
    pub const fn projection_clamped(self) -> usize {
        self.projection_clamped
    }

    #[must_use]
    pub const fn conditional_capture_probability(self) -> f64 {
        f64::from_bits(self.conditional_capture_probability_bits)
    }

    #[must_use]
    pub const fn conditional_expected_hit_mass_kg(self) -> f64 {
        f64::from_bits(self.conditional_expected_hit_mass_kg_bits)
    }

    #[must_use]
    pub const fn maximum_disk_scale(self) -> f64 {
        f64::from_bits(self.maximum_disk_scale_bits)
    }

    #[must_use]
    pub const fn maximum_fourth_order(self) -> f64 {
        f64::from_bits(self.maximum_fourth_order_bits)
    }

    #[must_use]
    pub const fn maximum_e14_indicator(self) -> f64 {
        f64::from_bits(self.maximum_e14_indicator_bits)
    }
}

fn conditional_expected_hit_mass_kg(
    released_mass_kg: f64,
    conditional_capture_probability: f64,
) -> anyhow::Result<f64> {
    ensure!(
        released_mass_kg.is_finite() && released_mass_kg >= 0.0,
        "released mass must be finite and nonnegative"
    );
    ensure!(
        conditional_capture_probability.is_finite()
            && (0.0..=1.0).contains(&conditional_capture_probability),
        "conditional capture probability must lie in [0, 1]"
    );
    let expected = if conditional_capture_probability == 0.0 {
        0.0
    } else if conditional_capture_probability.to_bits() == 1.0_f64.to_bits() {
        released_mass_kg
    } else {
        released_mass_kg * conditional_capture_probability
    };
    ensure!(
        expected.is_finite() && (0.0..=released_mass_kg).contains(&expected),
        "conditional expected hit mass must lie in [0, released mass]"
    );
    Ok(expected)
}

#[derive(Clone, Copy)]
struct PreparedComponent {
    mean_x: f64,
    mean_y: f64,
    cov00: f64,
    cov01: f64,
    cov11: f64,
    inv00: f64,
    inv01: f64,
    inv11: f64,
    log_normalization: f64,
    log_weight: f64,
    major_precision: f64,
    minor_precision: f64,
    // Hoisted once per component: the C12 point kernel consumes these on
    // every grid point (7,680+ per row), and recomputing sqrt there was
    // measured in the 2026-08-27 deep profile. Same values, same operations,
    // computed once — bit-identical by construction.
    major_precision_sqrt: f64,
    minor_precision_sqrt: f64,
    major_axis_x: f64,
    major_axis_y: f64,
}

struct PreparedConditionalMixture {
    components: Vec<PreparedComponent>,
    radius_km: f64,
    log_area_km2: f64,
}

struct PreparedModel {
    conditional: PreparedConditionalMixture,
    target_cholesky: [f64; 3],
    target_frame: [f64; 4],
    target_separability: f64,
    deterministic_floor_count: u64,
    packet_mass_kg: f64,
    policy_threshold_probability: f64,
    policy_threshold_log: f64,
}

struct PreparedTargetGrid {
    log_packet_failures: Vec<f64>,
    weights: PreparedBinary64TargetWeights,
    expected_conditional_capture_probability: f64,
    maximum_disk_scale: f64,
    maximum_fourth_order: f64,
    maximum_e14_indicator: f64,
    c12_component_evaluations: u64,
}

struct PreparedTargetSolution {
    fine: PreparedTargetGrid,
    decision: Binary64PacketCountDecision,
    selected_log: f64,
    base_target_quadrature_delta: f64,
    target_quadrature_delta: f64,
    target_refinement_level: u8,
    target_radial_samples: usize,
    target_angular_samples: usize,
    maximum_disk_scale: f64,
    maximum_fourth_order: f64,
    maximum_e14_indicator: f64,
    c12_component_evaluations: u64,
}

#[derive(Clone, Debug, PartialEq)]
struct PreparedBinary64TargetWeights {
    log_weights: Vec<f64>,
    normalization_log_bits: u64,
}

impl PreparedBinary64TargetWeights {
    fn new(weights: &[f64]) -> anyhow::Result<Self> {
        ensure!(!weights.is_empty(), "binary64 target weights are empty");
        let mut log_weights = Vec::with_capacity(weights.len());
        let mut normalization_log = f64::NEG_INFINITY;
        for &weight in weights {
            ensure!(
                weight.is_finite() && weight >= 0.0,
                "binary64 target weight is invalid"
            );
            let log_weight = if weight == 0.0 {
                f64::NEG_INFINITY
            } else {
                weight.ln()
            };
            log_weights.push(log_weight);
            normalization_log = binary64_logaddexp(normalization_log, log_weight);
        }
        ensure!(
            normalization_log.is_finite(),
            "binary64 target weights have no positive finite sum"
        );
        Ok(Self {
            log_weights,
            normalization_log_bits: normalization_log.to_bits(),
        })
    }

    const fn normalization_log_bits(&self) -> u64 {
        self.normalization_log_bits
    }

    /// Cheap lower bound on `normalized_log_no_hit` at `count`: the largest
    /// single term of the log-sum-exp, normalized with the same subtraction.
    ///
    /// The true fold satisfies `numerator >= max_term` exactly in binary64
    /// (`binary64_logaddexp(a, b) >= max(a, b)` by construction, including
    /// the underflow shortcut), and `x - Z` is monotone in `x` under any
    /// rounding, so `fl(numerator - Z) >= fl(max_term - Z)` — the returned
    /// value is a rigorous lower bound on the exact bits the fold produces.
    /// Returns None when a term is NaN; the caller then takes the full fold,
    /// whose validation reports the defect.
    fn normalized_max_term(&self, failures: &[f64], count_f64: f64) -> Option<f64> {
        let mut max_term = f64::NEG_INFINITY;
        for (&log_weight, &failure) in self.log_weights.iter().zip(failures) {
            let term = log_weight + count_f64 * failure;
            if term.is_nan() {
                return None;
            }
            if term > max_term {
                max_term = term;
            }
        }
        Some(max_term - f64::from_bits(self.normalization_log_bits))
    }

    fn normalized_log_no_hit(&self, failures: &[f64], count: u64) -> anyhow::Result<f64> {
        ensure!(
            !failures.is_empty() && failures.len() == self.log_weights.len(),
            "binary64 target failure grid is empty or mismatched"
        );
        ensure!(
            failures
                .iter()
                .all(|value| *value == f64::NEG_INFINITY || (value.is_finite() && *value <= 0.0)),
            "binary64 target failure log is invalid"
        );
        let count_f64 = u64_to_exact_binary64(count)?;
        if count == 0 {
            return Ok(0.0);
        }
        // Two-pass max-then-sum log-sum-exp (2026-08-27, bit-moving with the
        // physics intact). The sequential logaddexp fold chained every `exp`
        // through the running accumulator, serializing the row's dominant
        // transcendental work; anchoring on the maximum first makes every
        // remaining exponential independent and better-conditioned (each
        // argument is <= 0, the leading term is exact). The result still
        // satisfies `numerator >= max_term` bit-for-bit (`ln_1p(rest >= 0)
        // >= 0`), which the pruned threshold search's lower bound relies on.
        // `two_pass_no_hit_fold_matches_sequential_fold_within_ulps`
        // characterizes the move against the retired sequential fold.
        let mut max_term = f64::NEG_INFINITY;
        let mut anchor = usize::MAX;
        for (index, (&log_weight, &failure)) in self.log_weights.iter().zip(failures).enumerate() {
            let term = log_weight + count_f64 * failure;
            ensure!(!term.is_nan(), "binary64 no-hit term is invalid");
            if term > max_term {
                max_term = term;
                anchor = index;
            }
        }
        if max_term == f64::NEG_INFINITY {
            return Ok(f64::NEG_INFINITY);
        }
        let mut rest = 0.0_f64;
        for (index, (&log_weight, &failure)) in self.log_weights.iter().zip(failures).enumerate() {
            if index == anchor {
                continue;
            }
            let term = log_weight + count_f64 * failure;
            let delta = term - max_term;
            // exp underflows to exactly +0.0 at or below the bound; adding
            // +0.0 is a no-op, so skipping it is value-preserving.
            if delta > LOGADDEXP_EXP_UNDERFLOW_BOUND {
                rest += delta.exp();
            }
        }
        let numerator = max_term + rest.ln_1p();
        let normalized = numerator - f64::from_bits(self.normalization_log_bits);
        if normalized.is_finite() && normalized <= 0.0 {
            return Ok(normalized);
        }
        // The sequential fold is termwise monotone under rounding, so its
        // numerator provably never exceeds the sequentially-folded
        // normalization; the two-pass sum carries no such proof when the
        // true value sits within rounding distance of zero. Retake the
        // proven path instead of failing closed on a conditioning artifact.
        self.sequential_normalized_log_no_hit(failures, count_f64)
    }

    fn sequential_normalized_log_no_hit(
        &self,
        failures: &[f64],
        count_f64: f64,
    ) -> anyhow::Result<f64> {
        let mut numerator = f64::NEG_INFINITY;
        for (&log_weight, &failure) in self.log_weights.iter().zip(failures) {
            let term = log_weight + count_f64 * failure;
            ensure!(!term.is_nan(), "binary64 no-hit term is invalid");
            numerator = binary64_logaddexp(numerator, term);
        }
        if numerator == f64::NEG_INFINITY {
            return Ok(f64::NEG_INFINITY);
        }
        let normalized = numerator - f64::from_bits(self.normalization_log_bits);
        ensure!(
            normalized.is_finite() && normalized <= 0.0,
            "binary64 normalized no-hit log is invalid"
        );
        Ok(normalized)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Binary64ThresholdMinimum {
    probability_count: u64,
    selected_log_no_hit_bits: u64,
    predecessor_log_no_hit_bits: u64,
}

impl Binary64ThresholdMinimum {
    const fn probability_count(self) -> u64 {
        self.probability_count
    }
    const fn selected_log_no_hit_bits(self) -> u64 {
        self.selected_log_no_hit_bits
    }
    const fn predecessor_log_no_hit_bits(self) -> u64 {
        self.predecessor_log_no_hit_bits
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Binary64PacketCountDecision {
    probability_minimum: Binary64ThresholdMinimum,
    final_count: u64,
    governor: SharedTargetPacketCountGovernor,
}

fn u64_to_exact_binary64(count: u64) -> anyhow::Result<f64> {
    ensure!(
        count <= MAX_EXACT_BINARY64_PACKET_COUNT,
        "packet count exceeds the exact binary64 integer domain"
    );
    Ok(super::u64_to_f64(count))
}

fn deterministic_floor_packet_count(
    required_mass_kg: f64,
    packet_mass_kg: f64,
) -> anyhow::Result<u64> {
    let mut count = super::checked_ceil_packet_count(
        (required_mass_kg / packet_mass_kg).max(1.0),
        "shared-target deterministic floor",
    )?;
    if count > MAX_EXACT_BINARY64_PACKET_COUNT {
        return Err(anyhow::Error::new(
            Binary64PacketCountUnrepresentable::deterministic_floor(count),
        ));
    }
    if u64_to_exact_binary64(count)? * packet_mass_kg < required_mass_kg {
        count = count
            .checked_add(1)
            .ok_or_else(|| anyhow!("shared-target deterministic floor count overflow"))?;
    }
    if count > MAX_EXACT_BINARY64_PACKET_COUNT {
        return Err(anyhow::Error::new(
            Binary64PacketCountUnrepresentable::deterministic_floor(count),
        ));
    }
    Ok(count)
}

fn ensure_binary64_count_logs_nonincreasing(
    lower_count: u64,
    lower_log: f64,
    upper_count: u64,
    upper_log: f64,
) -> anyhow::Result<()> {
    ensure!(
        lower_count < upper_count
            && !lower_log.is_nan()
            && !upper_log.is_nan()
            && lower_log <= 0.0
            && upper_log <= 0.0
            && lower_log >= upper_log,
        "binary64 no-hit functional is nonmonotone between counts {lower_count} and {upper_count}"
    );
    Ok(())
}

/// Margin, in nats, that the cheap upper bound must clear beyond `ln(N)`
/// before a full fold is skipped. The fold's accumulated rounding is under
/// `3 * N * eps * |log|` — a few 1e-9 at the sealed grid sizes — so 1e-6
/// leaves three orders of magnitude of slack, and an uncertain classification
/// only costs one true fold, never a wrong answer.
const THRESHOLD_SEARCH_CERTAINTY_GUARD_NATS: f64 = 1.0e-6;

fn minimum_binary64_threshold_count(
    failures: &[f64],
    weights: &PreparedBinary64TargetWeights,
    threshold_log: f64,
) -> anyhow::Result<Binary64ThresholdMinimum> {
    // Pruned front end for the exhaustive search below. The bracket update
    // decisions are exactly the true-fold comparisons: a count is classified
    // without its fold only when a rigorous bound already decides the same
    // `<= threshold` comparison the fold would (lower bound is exact-side —
    // the fold result is >= `normalized_max_term` bit-for-bit; upper bound is
    // `max_term + ln(N)` plus a guard that dwarfs fold rounding). The integer
    // bracket trajectory is therefore identical to the exhaustive search, and
    // the returned selected/predecessor logs are re-evaluated with true folds
    // at the same two counts — bit-identical output, measured 2026-08-27 to
    // remove most of the count search's log1p/exp volume (the deep-profile
    // headline: that fold machinery was ~2/3 of the MF24 active CPU).
    // Any straddle disagreement falls back to the exhaustive search, whose
    // per-step monotonicity diagnostics then run in full.
    ensure!(
        threshold_log.is_finite() && threshold_log < 0.0,
        "binary64 no-hit threshold is invalid"
    );
    ensure!(
        !failures.is_empty() && failures.len() == weights.log_weights.len(),
        "binary64 target failure grid is empty or mismatched"
    );
    let terms_log = {
        let count_f64 = u64_to_exact_binary64(u64::try_from(failures.len())?)?;
        count_f64.ln()
    };
    // Some(true): certainly <= threshold. Some(false): certainly > threshold.
    // None: undecided — take the true fold.
    let classify = |count: u64| -> anyhow::Result<Option<bool>> {
        let count_f64 = u64_to_exact_binary64(count)?;
        if count == 0 {
            return Ok(Some(false));
        }
        let Some(lower_bound) = weights.normalized_max_term(failures, count_f64) else {
            return Ok(None);
        };
        if lower_bound > threshold_log {
            return Ok(Some(false));
        }
        if lower_bound + terms_log + THRESHOLD_SEARCH_CERTAINTY_GUARD_NATS <= threshold_log {
            return Ok(Some(true));
        }
        Ok(None)
    };
    let evaluate = |count| weights.normalized_log_no_hit(failures, count);
    let at_or_below = |count: u64| -> anyhow::Result<bool> {
        match classify(count)? {
            Some(decision) => Ok(decision),
            None => Ok(evaluate(count)? <= threshold_log),
        }
    };

    let mut lower = 0_u64;
    let mut upper = 1_u64;
    while !at_or_below(upper)? {
        lower = upper;
        let next = upper
            .checked_mul(2)
            .unwrap_or(MAX_EXACT_BINARY64_PACKET_COUNT)
            .min(MAX_EXACT_BINARY64_PACKET_COUNT);
        if next == upper {
            return Err(anyhow::Error::new(
                Binary64PacketCountUnrepresentable::probability_threshold(
                    evaluate(upper)?,
                    threshold_log,
                ),
            ));
        }
        upper = next;
    }
    while upper.saturating_sub(lower) > 1 {
        let midpoint = lower.saturating_add(upper.saturating_sub(lower) / 2);
        if at_or_below(midpoint)? {
            upper = midpoint;
        } else {
            lower = midpoint;
        }
    }
    let upper_log = evaluate(upper)?;
    let lower_log = evaluate(lower)?;
    if upper_log <= threshold_log && lower_log > threshold_log {
        ensure_binary64_count_logs_nonincreasing(lower, lower_log, upper, upper_log)?;
        return Ok(Binary64ThresholdMinimum {
            probability_count: upper,
            selected_log_no_hit_bits: upper_log.to_bits(),
            predecessor_log_no_hit_bits: lower_log.to_bits(),
        });
    }
    minimum_binary64_threshold_count_exhaustive(failures, weights, threshold_log)
}

fn minimum_binary64_threshold_count_exhaustive(
    failures: &[f64],
    weights: &PreparedBinary64TargetWeights,
    threshold_log: f64,
) -> anyhow::Result<Binary64ThresholdMinimum> {
    ensure!(
        threshold_log.is_finite() && threshold_log < 0.0,
        "binary64 no-hit threshold is invalid"
    );
    let evaluate = |count| weights.normalized_log_no_hit(failures, count);
    let mut lower = 0_u64;
    let mut lower_log = evaluate(lower)?;
    let mut upper = 1_u64;
    let mut upper_log = evaluate(upper)?;
    ensure_binary64_count_logs_nonincreasing(lower, lower_log, upper, upper_log)?;
    // The doubling search is over a LOG PROBABILITY, so its condition is a
    // float comparison by construction. `evaluate` is monotone and the loop
    // exits on the exact-count ceiling below, so this cannot spin.
    #[expect(clippy::while_float, reason = "the search bound is a log probability")]
    while upper_log > threshold_log {
        lower = upper;
        lower_log = upper_log;
        let next = upper
            .checked_mul(2)
            .unwrap_or(MAX_EXACT_BINARY64_PACKET_COUNT)
            .min(MAX_EXACT_BINARY64_PACKET_COUNT);
        if next == upper {
            return Err(anyhow::Error::new(
                Binary64PacketCountUnrepresentable::probability_threshold(upper_log, threshold_log),
            ));
        }
        let next_log = evaluate(next)?;
        ensure_binary64_count_logs_nonincreasing(upper, upper_log, next, next_log)?;
        upper = next;
        upper_log = next_log;
    }
    while upper.saturating_sub(lower) > 1 {
        // `upper > lower` is this loop's own condition, so the saturating forms
        // are exact here; they state the invariant rather than assume it.
        let midpoint = lower.saturating_add(upper.saturating_sub(lower) / 2);
        let midpoint_log = evaluate(midpoint)?;
        ensure_binary64_count_logs_nonincreasing(lower, lower_log, midpoint, midpoint_log)?;
        ensure_binary64_count_logs_nonincreasing(midpoint, midpoint_log, upper, upper_log)?;
        if midpoint_log <= threshold_log {
            upper = midpoint;
            upper_log = midpoint_log;
        } else {
            lower = midpoint;
            lower_log = midpoint_log;
        }
    }
    ensure!(
        upper_log <= threshold_log && lower_log > threshold_log,
        "binary64 threshold search lost selected/predecessor minimality"
    );
    Ok(Binary64ThresholdMinimum {
        probability_count: upper,
        selected_log_no_hit_bits: upper_log.to_bits(),
        predecessor_log_no_hit_bits: lower_log.to_bits(),
    })
}

const fn select_shared_target_packet_count(
    probability_count: u64,
    deterministic_floor_count: u64,
) -> (u64, SharedTargetPacketCountGovernor) {
    if probability_count >= deterministic_floor_count {
        (
            probability_count,
            SharedTargetPacketCountGovernor::Probability,
        )
    } else {
        (
            deterministic_floor_count,
            SharedTargetPacketCountGovernor::DeterministicFloor,
        )
    }
}

fn decide_binary64_packet_count(
    failures: &[f64],
    weights: &PreparedBinary64TargetWeights,
    threshold_log: f64,
    deterministic_floor_count: u64,
) -> anyhow::Result<Binary64PacketCountDecision> {
    ensure!(
        deterministic_floor_count > 0,
        "deterministic packet floor must be positive"
    );
    if deterministic_floor_count > MAX_EXACT_BINARY64_PACKET_COUNT {
        return Err(anyhow::Error::new(
            Binary64PacketCountUnrepresentable::deterministic_floor(deterministic_floor_count),
        ));
    }
    let probability_minimum = minimum_binary64_threshold_count(failures, weights, threshold_log)?;
    let (final_count, governor) = select_shared_target_packet_count(
        probability_minimum.probability_count(),
        deterministic_floor_count,
    );
    Ok(Binary64PacketCountDecision {
        probability_minimum,
        final_count,
        governor,
    })
}

fn final_selected_log_no_hit(
    failures: &[f64],
    weights: &PreparedBinary64TargetWeights,
    decision: Binary64PacketCountDecision,
) -> anyhow::Result<f64> {
    match decision.governor {
        SharedTargetPacketCountGovernor::Probability => Ok(f64::from_bits(
            decision.probability_minimum.selected_log_no_hit_bits(),
        )),
        SharedTargetPacketCountGovernor::DeterministicFloor => {
            weights.normalized_log_no_hit(failures, decision.final_count)
        }
    }
}

#[cfg(test)]
std::thread_local! {
    static PREPARE_MODEL_CALL_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reset_prepare_model_call_count() {
    PREPARE_MODEL_CALL_COUNT.set(0);
}

#[cfg(test)]
fn prepare_model_call_count() -> usize {
    PREPARE_MODEL_CALL_COUNT.get()
}

fn prepare_conditional_component(
    mean_x: f64,
    mean_y: f64,
    covariance: [f64; 4],
    log_weight: f64,
) -> anyhow::Result<PreparedComponent> {
    let (determinant, inv00, inv01, inv11) = super::det_inv_symmetric_2x2(&covariance);
    ensure!(
        determinant.is_finite()
            && determinant > 0.0
            && [inv00, inv01, inv11].into_iter().all(f64::is_finite),
        "shared-target per-grain covariance is invalid"
    );
    let difference = inv00 - inv11;
    let discriminant = difference.hypot(2.0 * inv01);
    let major_precision = 0.5 * (inv00 + inv11 + discriminant);
    let minor_precision = 0.5 * (inv00 + inv11 - discriminant);
    let (axis_x, axis_y) = if discriminant == 0.0 {
        (1.0, 0.0)
    } else if difference >= 0.0 {
        (difference + discriminant, 2.0 * inv01)
    } else {
        (2.0 * inv01, discriminant - difference)
    };
    let axis_norm = axis_x.hypot(axis_y);
    let (major_axis_x, major_axis_y) = (axis_x / axis_norm, axis_y / axis_norm);
    ensure!(
        major_precision.is_finite()
            && minor_precision.is_finite()
            && minor_precision > 0.0
            && major_axis_x.is_finite()
            && major_axis_y.is_finite(),
        "shared-target component eigensystem is invalid"
    );
    Ok(PreparedComponent {
        mean_x,
        mean_y,
        cov00: covariance[0],
        cov01: covariance[1],
        cov11: covariance[3],
        inv00,
        inv01,
        inv11,
        log_normalization: -std::f64::consts::TAU.ln() - 0.5 * determinant.ln(),
        log_weight,
        major_precision,
        minor_precision,
        major_precision_sqrt: major_precision.sqrt(),
        minor_precision_sqrt: minor_precision.sqrt(),
        major_axis_x,
        major_axis_y,
    })
}

fn prepare_conditional_mixture_for_target_state(
    source: &PreparedConditionalCaptureSource,
    target_state: [f64; 6],
) -> anyhow::Result<(PreparedConditionalMixture, usize)> {
    ensure!(
        target_state.iter().all(|value| value.is_finite()),
        "shared-target target state must be finite"
    );
    let plane = super::pc_bplane_basis_from_states(&source.hf_velocity_mean, &target_state)?;
    let [target_x, target_y, target_z, _, _, _] = target_state;
    let mut components = Vec::with_capacity(source.components.len());
    let mut projection_clamped = 0usize;

    for component in &source.components {
        let position_mean = component.position_mean;
        let position_covariance = component.position_covariance;
        let delta = [
            position_mean[0] - target_x,
            position_mean[1] - target_y,
            position_mean[2] - target_z,
        ];
        let first = [plane[0], plane[1], plane[2]];
        let second = [plane[3], plane[4], plane[5]];
        let mean_x = super::dot3(first, delta);
        let mean_y = super::dot3(second, delta);
        let [raw00, raw01, raw10, raw11] =
            super::project_covariance_to_bplane(position_covariance, plane);
        let (covariance, clamped) = super::sanitize_covariance_2d_values(
            raw00,
            raw01,
            raw10,
            raw11,
            source.covariance_minimum,
            source.covariance_maximum,
        )?;
        projection_clamped = projection_clamped
            .checked_add(usize::from(clamped))
            .ok_or_else(|| anyhow!("shared-target projection clamp count overflow"))?;
        components.push(prepare_conditional_component(
            mean_x,
            mean_y,
            covariance,
            component.log_weight,
        )?);
    }

    ensure!(
        !components.is_empty(),
        "shared-target mixture has no positive finite weight"
    );
    let radius_km = (source.area_km2 / std::f64::consts::PI).sqrt();
    ensure!(
        radius_km.is_finite() && radius_km > 0.0,
        "shared-target hard-body radius is invalid"
    );
    Ok((
        PreparedConditionalMixture {
            components,
            radius_km,
            log_area_km2: source.area_km2.ln(),
        },
        projection_clamped,
    ))
}

fn prepare_conditional_mixture(
    projected_means_2d: &[f64],
    projected_covariances_2d: &[f64],
    mixture_weights: &[f64],
    area_km2: f64,
    covariance_minimum: f64,
    covariance_maximum: f64,
) -> anyhow::Result<PreparedConditionalMixture> {
    ensure!(
        area_km2.is_finite() && area_km2 > 0.0,
        "shared-target area must be finite and positive"
    );
    ensure!(
        covariance_minimum.is_finite()
            && covariance_minimum > 0.0
            && covariance_maximum.is_finite()
            && covariance_maximum >= covariance_minimum,
        "shared-target covariance bounds are invalid"
    );
    let component_count = mixture_weights.len();
    ensure!(
        component_count > 0,
        "shared-target mixture must not be empty"
    );
    ensure!(
        projected_means_2d.len()
            == component_count
                .checked_mul(2)
                .ok_or_else(|| anyhow!("shared-target mean length overflow"))?
            && projected_covariances_2d.len()
                == component_count
                    .checked_mul(4)
                    .ok_or_else(|| anyhow!("shared-target covariance length overflow"))?,
        "shared-target component arrays have inconsistent lengths"
    );
    ensure!(
        projected_means_2d.iter().all(|value| value.is_finite()),
        "shared-target component mean is non-finite"
    );
    ensure!(
        mixture_weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0),
        "shared-target mixture weight is invalid"
    );
    let weight_sum = mixture_weights.iter().sum::<f64>();
    ensure!(
        weight_sum.is_finite() && weight_sum > 0.0,
        "shared-target mixture has no positive finite weight"
    );
    let mut components = Vec::with_capacity(component_count);
    for ((mean, covariance), &weight) in projected_means_2d
        .chunks_exact(2)
        .zip(projected_covariances_2d.chunks_exact(4))
        .zip(mixture_weights)
    {
        if weight == 0.0 {
            continue;
        }
        let [mean_x, mean_y] = <[f64; 2]>::try_from(mean)
            .map_err(|_| anyhow!("shared-target component mean must contain two values"))?;
        let [cov00, cov01, cov10, cov11] = <[f64; 4]>::try_from(covariance)
            .map_err(|_| anyhow!("shared-target component covariance must contain four values"))?;
        let (covariance, _) = super::sanitize_covariance_2d_values(
            cov00,
            cov01,
            cov10,
            cov11,
            covariance_minimum,
            covariance_maximum,
        )?;
        components.push(prepare_conditional_component(
            mean_x,
            mean_y,
            covariance,
            (weight / weight_sum).ln(),
        )?);
    }
    ensure!(
        !components.is_empty(),
        "shared-target mixture has no positive finite weight"
    );
    let radius_km = (area_km2 / std::f64::consts::PI).sqrt();
    ensure!(
        radius_km.is_finite() && radius_km > 0.0,
        "shared-target hard-body radius is invalid"
    );
    Ok(PreparedConditionalMixture {
        components,
        radius_km,
        log_area_km2: area_km2.ln(),
    })
}

fn prepare_model(inputs: &SharedTargetReplayInputs<'_>) -> anyhow::Result<PreparedModel> {
    #[cfg(test)]
    PREPARE_MODEL_CALL_COUNT.set(PREPARE_MODEL_CALL_COUNT.get().saturating_add(1));
    ensure!(
        inputs.area_km2.is_finite() && inputs.area_km2 > 0.0,
        "shared-target area must be finite and positive"
    );
    ensure!(
        inputs.target_hit_probability.is_finite()
            && (0.0..1.0).contains(&inputs.target_hit_probability),
        "shared-target hit probability must lie in (0, 1)"
    );
    ensure!(
        inputs.grain_mass_kg.is_finite() && inputs.grain_mass_kg > 0.0,
        "grain mass must be finite and positive"
    );
    let conditional = prepare_conditional_mixture(
        inputs.projected_means_2d,
        inputs.projected_covariances_2d,
        inputs.mixture_weights,
        inputs.area_km2,
        inputs.covariance_minimum,
        inputs.covariance_maximum,
    )?;
    let scenario = inputs.scenario;
    let quadrature = scenario.quadrature();
    let target_cholesky = target_cholesky(scenario.target_covariance_2d_km2())?;
    let (target_frame, target_separability) =
        target_frame_and_separability(&conditional.components, target_cholesky)?;
    let correlation = u64_to_exact_binary64(scenario.packet_correlation_grains())?;
    let packet_mass_kg = inputs.grain_mass_kg * correlation;
    ensure!(
        packet_mass_kg.is_finite() && packet_mass_kg > 0.0,
        "shared-target packet mass is invalid"
    );
    let deterministic_floor_count =
        deterministic_floor_packet_count(inputs.deterministic_required_mass_kg, packet_mass_kg)?;
    let policy_threshold_probability =
        quadrature.conservative_failure_probability(inputs.target_hit_probability)?;
    let policy_threshold_log = policy_threshold_probability.ln();
    ensure!(
        policy_threshold_log.is_finite() && policy_threshold_log < 0.0,
        "shared-target failure probability is invalid"
    );
    Ok(PreparedModel {
        conditional,
        target_cholesky,
        target_frame,
        target_separability,
        deterministic_floor_count,
        packet_mass_kg,
        policy_threshold_probability,
        policy_threshold_log,
    })
}

fn prepare_target_solution(
    model: &PreparedModel,
    quadrature: SharedTargetQuadrature,
) -> anyhow::Result<PreparedTargetSolution> {
    let mut base_delta = 0.0;
    for refinement_level in 0_u8..=1 {
        let radial = quadrature
            .target_radial_samples()
            .checked_mul(usize::from(refinement_level) + 1)
            .ok_or_else(|| anyhow!("target radial refinement count overflow"))?;
        let angular = quadrature.target_angular_samples();
        let fine = prepare_target_grid(model, radial, angular)?;
        let half = prepare_target_grid(model, radial / 2, angular / 2)?;
        let decision = decide_binary64_packet_count(
            &fine.log_packet_failures,
            &fine.weights,
            model.policy_threshold_log,
            model.deterministic_floor_count,
        )?;
        let selected_log =
            final_selected_log_no_hit(&fine.log_packet_failures, &fine.weights, decision)?;
        let half_log = half
            .weights
            .normalized_log_no_hit(&half.log_packet_failures, decision.final_count)?;
        let fine_probability = probability_from_log(selected_log, "fine shared-target no-hit")?;
        let half_probability = probability_from_log(half_log, "half-target shared-target no-hit")?;
        let delta = next_up((fine_probability - half_probability).abs());
        if delta > quadrature.convergence_tolerance() && refinement_level == 0 {
            base_delta = delta;
            continue;
        }
        ensure!(
            delta <= quadrature.convergence_tolerance(),
            "shared-target target quadrature failed after bounded refinement: level={refinement_level}, delta={delta}, tolerance={}{}",
            quadrature.convergence_tolerance(),
            target_envelope_note(model.target_separability)
        );
        return Ok(PreparedTargetSolution {
            maximum_disk_scale: fine.maximum_disk_scale.max(half.maximum_disk_scale),
            maximum_fourth_order: fine.maximum_fourth_order.max(half.maximum_fourth_order),
            maximum_e14_indicator: fine.maximum_e14_indicator.max(half.maximum_e14_indicator),
            c12_component_evaluations: fine
                .c12_component_evaluations
                .checked_add(half.c12_component_evaluations)
                .ok_or_else(|| anyhow!("C12 component count overflow"))?,
            fine,
            decision,
            selected_log,
            base_target_quadrature_delta: if refinement_level == 0 {
                delta
            } else {
                base_delta
            },
            target_quadrature_delta: delta,
            target_refinement_level: refinement_level,
            target_radial_samples: radial,
            target_angular_samples: angular,
        });
    }
    bail!("shared-target bounded target refinement exhausted without a verdict")
}

/// Computes one model-conditioned conservative contact-mass requirement.
///
/// # Errors
///
/// Returns an error for invalid authority inputs, C12 domain failure,
/// unconverged target quadrature, or an exact-count-domain overflow.
/// Recompute the C12 probability, count, convergence and refinement limbs.
///
/// This is the single arithmetic path. `shared_target_contact_mass_requirement`
/// calls it too, so a persisted replay cannot drift from live issuance by
/// running a second implementation -- there is only one.
///
/// It is non-minting: the returned value carries no authority token, so it
/// proves nothing until a caller bit-compares it against a stored witness.
///
/// # Errors
///
/// Returns an error for invalid authority inputs, C12 domain failure,
/// unconverged target quadrature, or an exact-count-domain overflow.
fn replay_shared_target_contact_mass_from_prepared(
    inputs: &SharedTargetReplayInputs<'_>,
    model: &PreparedModel,
) -> anyhow::Result<SharedTargetReplay> {
    let quadrature = inputs.scenario.quadrature();
    let solution = prepare_target_solution(model, quadrature)?;
    let decision = solution.decision;
    let minimum = decision.probability_minimum;
    let selected = decision.final_count;
    let probability_predecessor_log = f64::from_bits(minimum.predecessor_log_no_hit_bits());
    let governed_predecessor_log = (decision.governor
        == SharedTargetPacketCountGovernor::Probability)
        .then_some(minimum.predecessor_log_no_hit_bits());
    let release_mass_kg = u64_to_exact_binary64(selected)? * model.packet_mass_kg;
    ensure!(
        release_mass_kg.is_finite() && release_mass_kg > 0.0,
        "shared-target released mass exceeds finite f64"
    );
    let witness = C12Binary64LogMinimumWitnessV1 {
        scenario_content_identity: inputs
            .scenario
            .content_identity(inputs.target_hit_probability),
        policy_threshold_probability_bits: model.policy_threshold_probability.to_bits(),
        policy_threshold_log_bits: model.policy_threshold_log.to_bits(),
        probability_packet_count: minimum.probability_count(),
        deterministic_floor_packet_count: model.deterministic_floor_count,
        final_packet_count: selected,
        governor: decision.governor,
        selected_log_no_hit_bits: solution.selected_log.to_bits(),
        probability_predecessor_log_no_hit_bits: minimum.predecessor_log_no_hit_bits(),
        governed_predecessor_log_no_hit_bits: governed_predecessor_log,
        target_normalization_log_bits: solution.fine.weights.normalization_log_bits(),
        base_target_quadrature_delta_bits: solution.base_target_quadrature_delta.to_bits(),
        target_quadrature_delta_bits: solution.target_quadrature_delta.to_bits(),
        target_refinement_level: solution.target_refinement_level,
        target_radial_samples: solution.target_radial_samples,
        target_angular_samples: solution.target_angular_samples,
        maximum_disk_scale_bits: solution.maximum_disk_scale.to_bits(),
        maximum_fourth_order_bits: solution.maximum_fourth_order.to_bits(),
        maximum_e14_indicator_bits: solution.maximum_e14_indicator.to_bits(),
        c12_component_evaluations: solution.c12_component_evaluations,
    };
    Ok(SharedTargetReplay {
        witness,
        release_mass_kg,
        no_hit_probability: probability_from_log(
            solution.selected_log,
            "selected shared-target no-hit",
        )?,
        probability_predecessor_no_hit_probability: probability_from_log(
            probability_predecessor_log,
            "probability predecessor shared-target no-hit",
        )?,
        expected_conditional_capture_probability: solution
            .fine
            .expected_conditional_capture_probability,
    })
}

/// Replays the exact shared-target contact-mass solve and returns its witness.
///
/// # Errors
///
/// Returns an error when input authority, model preparation, or the exact
/// shared-target solve fails validation.
pub fn replay_shared_target_contact_mass(
    inputs: &SharedTargetReplayInputs<'_>,
) -> anyhow::Result<SharedTargetReplay> {
    let model = prepare_model(inputs)?;
    replay_shared_target_contact_mass_from_prepared(inputs, &model)
}

const fn shared_target_replay_inputs<'a>(
    inputs: &SharedTargetMassInputs<'a>,
) -> SharedTargetReplayInputs<'a> {
    SharedTargetReplayInputs {
        scenario: inputs.deterministic_mass.scenario(),
        deterministic_required_mass_kg: inputs.deterministic_mass.required_mass_kg(),
        projected_means_2d: inputs.projected_means_2d,
        projected_covariances_2d: inputs.projected_covariances_2d,
        mixture_weights: inputs.mixture_weights,
        area_km2: inputs.area_km2,
        target_hit_probability: inputs.target_hit_probability,
        grain_mass_kg: inputs.grain_mass_kg,
        covariance_minimum: inputs.covariance_minimum,
        covariance_maximum: inputs.covariance_maximum,
    }
}

fn shared_target_mass_estimate(
    inputs: &SharedTargetMassInputs<'_>,
    solved: SharedTargetReplay,
) -> anyhow::Result<SharedTargetMassEstimate> {
    let scenario = inputs.deterministic_mass.scenario();
    let target_area_m2 = inputs.area_km2 * 1.0e6;
    let target_disk_radius_m = (inputs.area_km2 / std::f64::consts::PI).sqrt() * 1.0e3;
    ensure!(
        target_area_m2.is_finite() && target_disk_radius_m.is_finite(),
        "shared-target capture geometry exceeds finite SI archive fields"
    );
    Ok(SharedTargetMassEstimate {
        scenario,
        target_area_m2,
        target_disk_radius_m,
        target_hit_probability: inputs.target_hit_probability,
        grain_mass_kg: inputs.grain_mass_kg,
        deterministic_required_mass_kg: inputs.deterministic_mass.required_mass_kg(),
        deterministic_mass_authority_id: inputs.deterministic_mass.mass_authority_id(),
        release_mass_kg: solved.release_mass_kg,
        effective_packet_count: solved.witness.final_packet_count(),
        no_hit_probability: solved.no_hit_probability,
        probability_predecessor_no_hit_probability: solved
            .probability_predecessor_no_hit_probability,
        expected_conditional_capture_probability: solved.expected_conditional_capture_probability,
        witness: solved.witness,
    })
}

/// Computes one model-conditioned conservative contact-mass requirement.
///
/// Wraps [`replay_shared_target_contact_mass`] with the deterministic evidence
/// the live path carries, so live issuance and persisted replay run one
/// arithmetic path rather than two that could drift.
///
/// # Errors
///
/// Returns an error for invalid authority inputs, C12 domain failure,
/// unconverged target quadrature, or an exact-count-domain overflow.
pub fn shared_target_contact_mass_requirement(
    inputs: &SharedTargetMassInputs<'_>,
) -> anyhow::Result<SharedTargetMassEstimate> {
    let replay_inputs = shared_target_replay_inputs(inputs);
    let model = prepare_model(&replay_inputs)?;
    let solved = replay_shared_target_contact_mass_from_prepared(&replay_inputs, &model)?;
    shared_target_mass_estimate(inputs, solved)
}

/// Applies the exact hard-mass proof, then solves an unresolved contact row.
///
/// Returns `None` only when the deterministic floor or mass-free geometric
/// proof establishes that no release below `release_mass_limit_kg` can satisfy
/// the contact requirement. An unresolved row runs the same arithmetic path as
/// [`shared_target_contact_mass_requirement`] without preparing its model a
/// second time.
///
/// # Errors
///
/// Returns an error for an invalid hard limit, invalid authority inputs, C12
/// domain failure, unconverged target quadrature, or exact-count-domain
/// overflow.
pub fn shared_target_contact_mass_requirement_under_limit(
    inputs: &SharedTargetMassInputs<'_>,
    release_mass_limit_kg: f64,
) -> anyhow::Result<Option<SharedTargetMassEstimate>> {
    ensure!(
        release_mass_limit_kg.is_finite() && release_mass_limit_kg > 0.0,
        "shared-target release-mass proof limit must be finite and positive"
    );
    let replay_inputs = shared_target_replay_inputs(inputs);
    let model = prepare_model(&replay_inputs)?;
    if shared_target_mass_limit_is_provably_infeasible_from_prepared(
        inputs,
        &model,
        release_mass_limit_kg,
    )? {
        return Ok(None);
    }
    let solved = replay_shared_target_contact_mass_from_prepared(&replay_inputs, &model)?;
    shared_target_mass_estimate(inputs, solved).map(Some)
}

fn binary64_logaddexp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    let minimum = left.min(right);
    let delta = minimum - maximum;
    // BIT-IDENTICAL shortcut, not an approximation: binary64 `exp` underflows
    // to exactly +0.0 for every argument at or below -746 (the last nonzero
    // result is at ln(2^-1075) = -745.13..), and `ln_1p(+0.0)` is +0.0, so the
    // full expression is exactly `maximum + 0.0` -- kept as that same final
    // addition so a negative-zero maximum normalizes identically. The
    // large-count threshold search spends most of its grid terms this far
    // below the running maximum; `logaddexp_shortcut_agrees_with_full_path`
    // pins the boundary.
    if delta <= LOGADDEXP_EXP_UNDERFLOW_BOUND {
        return maximum + 0.0;
    }
    maximum + delta.exp().ln_1p()
}

/// Below this, `exp` is +0.0 in binary64: the true value of `exp(-746)` is
/// under 2^-1076, less than half the smallest subnormal, on any faithfully
/// rounded libm. One full nat of margin below the exact bound of -745.13.
const LOGADDEXP_EXP_UNDERFLOW_BOUND: f64 = -746.0;

fn binary64_weighted_logsumexp(values: &[f64], weights: &[f64]) -> anyhow::Result<f64> {
    ensure!(
        !values.is_empty() && values.len() == weights.len(),
        "binary64 weighted log-sum-exp inputs are empty or mismatched"
    );
    let prepared = PreparedBinary64TargetWeights::new(weights)?;
    // Two-pass max-then-sum, mirroring `normalized_log_no_hit`: every
    // exponential is independent of the accumulator, unlike the retired
    // sequential fold that chained them. Falls back to the sequential fold
    // when the two-pass sum lands outside the sequential path's provable
    // range (finite result), which only happens within rounding distance of
    // the boundaries.
    let mut max_term = f64::NEG_INFINITY;
    let mut anchor = usize::MAX;
    for (index, (&value, &log_weight)) in values.iter().zip(&prepared.log_weights).enumerate() {
        ensure!(
            value == f64::NEG_INFINITY || value.is_finite(),
            "binary64 log-sum-exp value is invalid"
        );
        let term = log_weight + value;
        if term > max_term {
            max_term = term;
            anchor = index;
        }
    }
    if max_term == f64::NEG_INFINITY {
        return Ok(f64::NEG_INFINITY);
    }
    let mut rest = 0.0_f64;
    for (index, (&value, &log_weight)) in values.iter().zip(&prepared.log_weights).enumerate() {
        if index == anchor {
            continue;
        }
        let delta = (log_weight + value) - max_term;
        if delta > LOGADDEXP_EXP_UNDERFLOW_BOUND {
            rest += delta.exp();
        }
    }
    let normalized = (max_term + rest.ln_1p()) - f64::from_bits(prepared.normalization_log_bits);
    if normalized.is_finite() {
        return Ok(normalized);
    }
    let mut numerator = f64::NEG_INFINITY;
    for (&value, &log_weight) in values.iter().zip(&prepared.log_weights) {
        numerator = binary64_logaddexp(numerator, log_weight + value);
    }
    if numerator == f64::NEG_INFINITY {
        return Ok(f64::NEG_INFINITY);
    }
    let normalized = numerator - f64::from_bits(prepared.normalization_log_bits);
    ensure!(
        normalized.is_finite(),
        "binary64 weighted log-sum-exp is invalid"
    );
    Ok(normalized)
}

fn log_one_minus_exp(log_probability: f64) -> anyhow::Result<f64> {
    ensure!(
        log_probability == f64::NEG_INFINITY
            || (log_probability.is_finite() && log_probability <= 0.0),
        "conditional capture log probability is invalid"
    );
    if log_probability == f64::NEG_INFINITY {
        return Ok(-0.0);
    }
    if log_probability == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    let result = if log_probability < -std::f64::consts::LN_2 {
        (-log_probability.exp()).ln_1p()
    } else {
        (-log_probability.exp_m1()).ln()
    };
    ensure!(
        result.is_finite() && result <= 0.0,
        "conditional packet failure probability is invalid"
    );
    Ok(result)
}

fn probability_from_log(log_probability: f64, label: &str) -> anyhow::Result<f64> {
    ensure!(
        log_probability == f64::NEG_INFINITY
            || (log_probability.is_finite() && log_probability <= 0.0),
        "{label} log probability is invalid"
    );
    let probability = log_probability.exp();
    ensure!(
        probability.is_finite() && (0.0..=1.0).contains(&probability),
        "{label} probability is invalid"
    );
    Ok(probability)
}

fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    if value > 0.0 {
        f64::from_bits(value.to_bits().saturating_add(1))
    } else {
        f64::from_bits(value.to_bits().saturating_sub(1))
    }
}

fn target_cholesky(covariance: [f64; 4]) -> anyhow::Result<[f64; 3]> {
    let [cov00, cov01, cov10, cov11] = covariance;
    ensure!(
        covariance.into_iter().all(f64::is_finite),
        "shared target covariance is non-finite"
    );
    let scale = cov01.abs().max(cov10.abs()).max(1.0);
    ensure!(
        (cov01 - cov10).abs() <= 1.0e-12 * scale,
        "shared target covariance is not symmetric"
    );
    let off_diagonal = 0.5 * (cov01 + cov10);
    let l00 = cov00.sqrt();
    let l10 = off_diagonal / l00;
    let l11_squared = cov11 - l10 * l10;
    ensure!(
        l00.is_finite() && l00 > 0.0 && l11_squared.is_finite() && l11_squared > 0.0,
        "shared target covariance is not positive definite"
    );
    Ok([l00, l10, l11_squared.sqrt()])
}

fn standard_normal_composite_rule(count: usize) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    const HALF_WIDTH: f64 = 6.0;
    ensure!(
        count >= 2,
        "shared-target composite rule needs at least two nodes"
    );
    let order =
        u32::try_from(count).map_err(|_| anyhow!("target quadrature sample count exceeds u32"))?;
    let spacing = 2.0 * HALF_WIDTH / f64::from(order);
    let mut nodes = Vec::with_capacity(count);
    let mut weights = Vec::with_capacity(count);
    for index in 0..order {
        let node = (f64::from(index) + 0.5).mul_add(spacing, -HALF_WIDTH);
        nodes.push(node);
        weights.push((-0.5 * node * node).exp());
    }
    normalize_positive_weights(&mut weights)?;
    Ok((nodes, weights))
}

fn normalize_positive_weights(weights: &mut [f64]) -> anyhow::Result<()> {
    ensure!(
        !weights.is_empty()
            && weights
                .iter()
                .all(|weight| weight.is_finite() && *weight > 0.0),
        "shared-target quadrature weight is invalid"
    );
    let sum = weights.iter().sum::<f64>();
    ensure!(
        sum.is_finite() && sum > 0.0,
        "shared-target weights have invalid sum"
    );
    for weight in weights {
        *weight /= sum;
    }
    Ok(())
}

fn target_frame_and_separability(
    components: &[PreparedComponent],
    target_cholesky: [f64; 3],
) -> anyhow::Result<([f64; 4], f64)> {
    let [l00, l10, l11] = target_cholesky;
    let mut inv00 = 0.0;
    let mut inv01 = 0.0;
    let mut inv11 = 0.0;
    for component in components {
        let weight = component.log_weight.exp();
        inv00 = component.inv00.mul_add(weight, inv00);
        inv01 = component.inv01.mul_add(weight, inv01);
        inv11 = component.inv11.mul_add(weight, inv11);
    }
    let upper_left = l00.mul_add(
        l00 * inv00,
        l10.mul_add(2.0 * l00 * inv01, l10 * l10 * inv11),
    );
    let off_diagonal = l11 * l10.mul_add(inv11, l00 * inv01);
    let lower_right = l11 * l11 * inv11;
    let difference = upper_left - lower_right;
    let discriminant = difference.hypot(2.0 * off_diagonal);
    let (vector_x, vector_y) = if discriminant <= 0.0 {
        (1.0, 0.0)
    } else if difference >= 0.0 {
        (difference + discriminant, 2.0 * off_diagonal)
    } else {
        (2.0 * off_diagonal, discriminant - difference)
    };
    let norm = vector_x.hypot(vector_y);
    let (q00, q10) = (vector_x / norm, vector_y / norm);
    ensure!(
        q00.is_finite() && q10.is_finite(),
        "shared-target frame is invalid"
    );
    let frame = [q00, q10, -q10, q00];
    let mut minimum = f64::INFINITY;
    for component in components {
        let sharp_x = l00 * q00;
        let sharp_y = l10.mul_add(q00, l11 * q10);
        let slow_x = -l00 * q10;
        let slow_y = (-l10).mul_add(q10, l11 * q00);
        let sharp = super::mahalanobis_2x2(
            sharp_x,
            sharp_y,
            component.inv00,
            component.inv01,
            component.inv11,
        );
        let slow = super::mahalanobis_2x2(
            slow_x,
            slow_y,
            component.inv00,
            component.inv01,
            component.inv11,
        );
        ensure!(
            sharp.is_finite() && sharp > 0.0 && slow.is_finite() && slow > 0.0,
            "shared-target frame curvature is invalid"
        );
        minimum = minimum.min((sharp / slow).sqrt());
    }
    ensure!(minimum.is_finite(), "shared-target separability is invalid");
    Ok((frame, minimum))
}

fn target_envelope_note(separability: f64) -> String {
    if separability >= MINIMUM_SEPARABLE_ANISOTROPY {
        String::new()
    } else {
        format!("; polar target rule also failed for separability {separability:.3}")
    }
}

fn gaussian_even_derivatives(standard_offset: f64, precision: f64) -> [f64; 7] {
    let mut result = [0.0; 7];
    result[0] = 1.0;
    let mut previous = 1.0;
    let mut current = standard_offset;
    let mut precision_power = 1.0;
    for order in 2_u32..=12 {
        let next = standard_offset.mul_add(current, -f64::from(order.saturating_sub(1)) * previous);
        if order.is_multiple_of(2) {
            precision_power *= precision;
            if let Some(slot) = usize::try_from(order / 2)
                .ok()
                .and_then(|half| result.get_mut(half))
            {
                *slot = precision_power * next;
            }
        }
        previous = current;
        current = next;
    }
    result
}

fn binomial(row: usize, column: usize) -> f64 {
    const VALUES: [[u32; 7]; 7] = [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 2, 1, 0, 0, 0, 0],
        [1, 3, 3, 1, 0, 0, 0],
        [1, 4, 6, 4, 1, 0, 0],
        [1, 5, 10, 10, 5, 1, 0],
        [1, 6, 15, 20, 15, 6, 1],
    ];
    // Outside the triangle the binomial coefficient IS zero, so the checked
    // lookup returns the right answer rather than papering over a bad index.
    f64::from(
        VALUES
            .get(row)
            .and_then(|values| values.get(column))
            .copied()
            .unwrap_or(0),
    )
}

fn twelfth_order_taylor_coefficients(
    component: &PreparedComponent,
    dx: f64,
    dy: f64,
    radius_squared: f64,
) -> anyhow::Result<[f64; 7]> {
    let major_offset = component
        .major_axis_x
        .mul_add(dx, component.major_axis_y * dy);
    let minor_offset = (-component.major_axis_y).mul_add(dx, component.major_axis_x * dy);
    let major = gaussian_even_derivatives(
        component.major_precision_sqrt * major_offset,
        component.major_precision,
    );
    let minor = gaussian_even_derivatives(
        component.minor_precision_sqrt * minor_offset,
        component.minor_precision,
    );
    let mut coefficients = [0.0; 7];
    coefficients[0] = 1.0;
    let mut radius_power = 1.0;
    let mut denominator = 1.0;
    for order in 1_usize..=6 {
        let mut laplacian_power = 0.0;
        for major_order in 0..=order {
            let minor_order = order.saturating_sub(major_order);
            let (Some(major_term), Some(minor_term)) =
                (major.get(major_order), minor.get(minor_order))
            else {
                bail!("C12 disk recurrence index outside its coefficient table");
            };
            laplacian_power += binomial(order, major_order) * major_term * minor_term;
        }
        radius_power *= radius_squared;
        let order_f = f64::from(u32::try_from(order)?);
        let next_f = f64::from(u32::try_from(order.saturating_add(1))?);
        denominator *= 4.0 * order_f * next_f;
        let Some(slot) = coefficients.get_mut(order) else {
            bail!("C12 disk recurrence coefficient index outside its table");
        };
        *slot = radius_power * laplacian_power / denominator;
    }
    ensure!(
        coefficients.into_iter().all(f64::is_finite),
        "C12 disk recurrence is non-finite"
    );
    Ok(coefficients)
}

fn twelfth_order_disk_tail_upper(x: f64) -> Option<f64> {
    if !(x.is_finite() && x >= 0.0) {
        return None;
    }
    let mut exponential_term = x.powi(7) / 5_040.0;
    let mut tail = exponential_term / 8.0;
    for order in 8_u32..=128 {
        exponential_term *= x / f64::from(order);
        let addition = exponential_term / f64::from(order.saturating_add(1));
        tail += addition;
        if !tail.is_finite() {
            return None;
        }
        if addition <= f64::EPSILON * tail.max(f64::MIN_POSITIVE) {
            break;
        }
    }
    Some(next_up(tail))
}

#[derive(Clone, Copy)]
struct C12ComponentCapture {
    log_probability: f64,
    disk_scale: f64,
    fourth_order: f64,
    e14_indicator: f64,
}

fn conditional_component_capture_c12(
    component: &PreparedComponent,
    model: &PreparedConditionalMixture,
    target_x: f64,
    target_y: f64,
) -> anyhow::Result<C12ComponentCapture> {
    let dx = target_x - component.mean_x;
    let dy = target_y - component.mean_y;
    let quadratic =
        super::mahalanobis_2x2(dx, dy, component.inv00, component.inv01, component.inv11);
    ensure!(quadratic.is_finite(), "C12 disk geometry is non-finite");
    let radius_squared = model.radius_km * model.radius_km;
    let coefficients = twelfth_order_taylor_coefficients(component, dx, dy, radius_squared)?;
    let correction = coefficients.iter().sum::<f64>();
    let disk_scale = radius_squared * component.major_precision;
    let fourth_order = coefficients[2];
    ensure!(disk_scale.is_finite(), "C12 disk scale is non-finite");
    ensure!(
        fourth_order.is_finite(),
        "C12 fourth-order contribution is non-finite"
    );
    ensure!(
        correction.is_finite() && correction > 0.0,
        "C12 disk correction is invalid"
    );
    let gradient_x = component.inv00.mul_add(dx, component.inv01 * dy);
    let gradient_y = component.inv01.mul_add(dx, component.inv11 * dy);
    let rho = 0.5 * radius_squared;
    let m00 = rho * (component.inv00 + gradient_x * gradient_x);
    let m01 = rho * (component.inv01 + gradient_x * gradient_y);
    let m11 = rho * (component.inv11 + gradient_y * gradient_y);
    let tail_argument = (m00 + m01.abs()).max(m11 + m01.abs());
    let e14_indicator = twelfth_order_disk_tail_upper(tail_argument).unwrap_or(f64::MAX);
    let log_probability =
        model.log_area_km2 + component.log_normalization - 0.5 * quadratic + correction.ln();
    ensure!(
        log_probability.is_finite() && log_probability <= 0.0,
        "C12 component capture probability is invalid"
    );
    Ok(C12ComponentCapture {
        log_probability,
        disk_scale,
        fourth_order,
        e14_indicator,
    })
}

#[derive(Clone, Copy)]
struct C12MixtureCapture {
    log_probability: f64,
    maximum_disk_scale: f64,
    maximum_fourth_order: f64,
    maximum_e14_indicator: f64,
    component_evaluations: u64,
}

fn conditional_mixture_capture_c12(
    model: &PreparedConditionalMixture,
    target_x: f64,
    target_y: f64,
) -> anyhow::Result<C12MixtureCapture> {
    let mut log_probability = f64::NEG_INFINITY;
    let mut maximum_disk_scale: f64 = 0.0;
    let mut maximum_fourth_order: f64 = 0.0;
    let mut maximum_e14_indicator: f64 = 0.0;
    let mut component_evaluations = 0_u64;
    for component in &model.components {
        let capture = conditional_component_capture_c12(component, model, target_x, target_y)?;
        log_probability = binary64_logaddexp(
            log_probability,
            component.log_weight + capture.log_probability,
        );
        maximum_disk_scale = maximum_disk_scale.max(capture.disk_scale);
        maximum_fourth_order = maximum_fourth_order.max(capture.fourth_order.abs());
        maximum_e14_indicator = maximum_e14_indicator.max(capture.e14_indicator);
        component_evaluations = component_evaluations
            .checked_add(1)
            .ok_or_else(|| anyhow!("C12 component count overflow"))?;
    }
    ensure!(
        log_probability.is_finite() && log_probability <= 0.0,
        "C12 mixture capture probability is invalid"
    );
    Ok(C12MixtureCapture {
        log_probability,
        maximum_disk_scale,
        maximum_fourth_order,
        maximum_e14_indicator,
        component_evaluations,
    })
}

struct TargetGridRows {
    log_captures: Vec<f64>,
    log_packet_failures: Vec<f64>,
    weights: Vec<f64>,
    maximum_disk_scale: f64,
    maximum_fourth_order: f64,
    maximum_e14_indicator: f64,
    c12_component_evaluations: u64,
}

impl TargetGridRows {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            log_captures: Vec::with_capacity(capacity),
            log_packet_failures: Vec::with_capacity(capacity),
            weights: Vec::with_capacity(capacity),
            maximum_disk_scale: 0.0,
            maximum_fourth_order: 0.0,
            maximum_e14_indicator: 0.0,
            c12_component_evaluations: 0,
        }
    }

    fn push(
        &mut self,
        model: &PreparedModel,
        target_x: f64,
        target_y: f64,
        weight: f64,
    ) -> anyhow::Result<()> {
        ensure!(
            weight.is_finite() && weight > 0.0,
            "target grid weight is invalid"
        );
        let capture = conditional_mixture_capture_c12(&model.conditional, target_x, target_y)?;
        self.log_captures.push(capture.log_probability);
        self.log_packet_failures
            .push(log_one_minus_exp(capture.log_probability)?);
        self.weights.push(weight);
        self.maximum_disk_scale = self.maximum_disk_scale.max(capture.maximum_disk_scale);
        self.maximum_fourth_order = self.maximum_fourth_order.max(capture.maximum_fourth_order);
        self.maximum_e14_indicator = self
            .maximum_e14_indicator
            .max(capture.maximum_e14_indicator);
        self.c12_component_evaluations = self
            .c12_component_evaluations
            .checked_add(capture.component_evaluations)
            .ok_or_else(|| anyhow!("C12 component count overflow"))?;
        Ok(())
    }

    fn finish(self) -> anyhow::Result<PreparedTargetGrid> {
        let expected_log = binary64_weighted_logsumexp(&self.log_captures, &self.weights)?;
        let expected = probability_from_log(expected_log, "expected conditional capture")?;
        Ok(PreparedTargetGrid {
            weights: PreparedBinary64TargetWeights::new(&self.weights)?,
            log_packet_failures: self.log_packet_failures,
            expected_conditional_capture_probability: expected,
            maximum_disk_scale: self.maximum_disk_scale,
            maximum_fourth_order: self.maximum_fourth_order,
            maximum_e14_indicator: self.maximum_e14_indicator,
            c12_component_evaluations: self.c12_component_evaluations,
        })
    }
}

fn prepare_target_grid(
    model: &PreparedModel,
    axis_0_samples: usize,
    axis_1_samples: usize,
) -> anyhow::Result<PreparedTargetGrid> {
    if model.target_separability < MINIMUM_SEPARABLE_ANISOTROPY {
        prepare_target_polar_grid(model, axis_0_samples, axis_1_samples)
    } else {
        prepare_target_cartesian_grid(model, axis_0_samples, axis_1_samples)
    }
}

fn prepare_target_cartesian_grid(
    model: &PreparedModel,
    axis_0_samples: usize,
    axis_1_samples: usize,
) -> anyhow::Result<PreparedTargetGrid> {
    let point_count = axis_0_samples
        .checked_mul(axis_1_samples)
        .ok_or_else(|| anyhow!("target quadrature point count overflow"))?;
    let (axis_0_nodes, axis_0_weights) = standard_normal_composite_rule(axis_0_samples)?;
    let (axis_1_nodes, axis_1_weights) = standard_normal_composite_rule(axis_1_samples)?;
    let [q00, q10, q01, q11] = model.target_frame;
    let [l00, l10, l11_cholesky] = model.target_cholesky;
    let mut rows = TargetGridRows::with_capacity(point_count);
    for (&standard_x, &axis_0_weight) in axis_0_nodes.iter().zip(&axis_0_weights) {
        for (&standard_y, &axis_1_weight) in axis_1_nodes.iter().zip(&axis_1_weights) {
            let rotated_x = q00.mul_add(standard_x, q01 * standard_y);
            let rotated_y = q10.mul_add(standard_x, q11 * standard_y);
            let target_x = l00 * rotated_x;
            let target_y = l10.mul_add(rotated_x, l11_cholesky * rotated_y);
            rows.push(model, target_x, target_y, axis_0_weight * axis_1_weight)?;
        }
    }
    rows.finish()
}

fn prepare_target_polar_grid(
    model: &PreparedModel,
    radial_samples: usize,
    angular_samples: usize,
) -> anyhow::Result<PreparedTargetGrid> {
    const RADIAL_HALF_WIDTH: f64 = 6.0;
    ensure!(
        radial_samples >= 2 && radial_samples.is_multiple_of(2),
        "shared-target polar radial count must be positive and even"
    );
    let radial_count = u32::try_from(radial_samples)
        .map_err(|_| anyhow!("target polar radial sample count exceeds u32"))?;
    let angular_count = u32::try_from(angular_samples)
        .map_err(|_| anyhow!("target polar angular sample count exceeds u32"))?;
    let point_count = radial_samples
        .checked_mul(angular_samples)
        .ok_or_else(|| anyhow!("target polar point count overflow"))?;
    let radial_bin_count = radial_count / 2;
    let radial_spacing = RADIAL_HALF_WIDTH / f64::from(radial_bin_count);
    let angular_spacing = std::f64::consts::TAU / f64::from(angular_count);
    let gauss_offset = 1.0 / 3.0_f64.sqrt();
    let mut radii = Vec::with_capacity(radial_samples);
    let mut radial_weights = Vec::with_capacity(radial_samples);
    for radial_bin in 0..radial_bin_count {
        let midpoint = (f64::from(radial_bin) + 0.5) * radial_spacing;
        let half_spacing = 0.5 * radial_spacing;
        for signed_offset in [-gauss_offset, gauss_offset] {
            let radius = midpoint + half_spacing * signed_offset;
            radii.push(radius);
            radial_weights.push(half_spacing * radius * (-0.5 * radius * radius).exp());
        }
    }
    normalize_positive_weights(&mut radial_weights)?;
    let [q00, q10, q01, q11] = model.target_frame;
    let [l00, l10, l11_cholesky] = model.target_cholesky;
    let angular_weight = 1.0 / f64::from(angular_count);
    let mut rows = TargetGridRows::with_capacity(point_count);
    for (&radius, &radial_weight) in radii.iter().zip(&radial_weights) {
        for angular_index in 0..angular_count {
            let angle = (f64::from(angular_index) + 0.5) * angular_spacing;
            let standard_x = radius * angle.cos();
            let standard_y = radius * angle.sin();
            let rotated_x = q00.mul_add(standard_x, q01 * standard_y);
            let rotated_y = q10.mul_add(standard_x, q11 * standard_y);
            let target_x = l00 * rotated_x;
            let target_y = l10.mul_add(rotated_x, l11_cholesky * rotated_y);
            rows.push(model, target_x, target_y, radial_weight * angular_weight)?;
        }
    }
    rows.finish()
}

/// Mass-independent inputs to the private geometric packet-cap proof.
#[derive(Clone, Copy)]
struct MassFreeProofInputs {
    scenario: SharedTargetScenario,
    area_km2: f64,
    target_hit_probability: f64,
}

fn mass_free_infeasibility_direction(
    inputs: MassFreeProofInputs,
    model: &PreparedModel,
    release_mass_limit_kg: f64,
) -> anyhow::Result<Option<u32>> {
    const DIRECTION_COUNT: u32 = 32;
    const TAIL_SIGMA: f64 = 2.0;
    // The mathematical sufficient condition is strict at zero. Requiring a
    // 0.05 natural-log unit beyond it keeps a row out of the proof arm when
    // floating evaluation is anywhere near the decision boundary. Underflow
    // and overflow already abandon the direction below.
    const NUMERIC_LOG_MARGIN: f64 = 0.05;

    let packet_count_upper = release_mass_limit_kg / model.packet_mass_kg * (1.0 + 1.0e-12);
    ensure!(
        packet_count_upper.is_finite() && packet_count_upper > 0.0,
        "shared-target release-mass proof packet count is invalid"
    );
    let target_covariance = inputs.scenario.target_covariance_2d_km2();
    let radius_km = (inputs.area_km2 / std::f64::consts::PI).sqrt();
    // Mills' lower bound at two sigma is >0.0215. Use the smaller 0.021
    // constant so transcendental evaluation cannot inflate the region weight.
    let tail_weight_lower = 0.021_f64;
    // This analytic proof has no quadrature error to reserve. It must prove the
    // physical requested no-hit requirement itself; using the denser numerical
    // solver's reserved budget can reject a row that still satisfies raw 1-p.
    let target_failure_log = (-inputs.target_hit_probability).ln_1p();
    ensure!(
        target_failure_log.is_finite() && target_failure_log < 0.0,
        "shared-target release-mass proof probability is invalid"
    );

    for direction_index in 0..DIRECTION_COUNT {
        let angle = std::f64::consts::TAU * f64::from(direction_index) / f64::from(DIRECTION_COUNT);
        let unit_x = angle.cos();
        let unit_y = angle.sin();
        let target_variance = unit_x.mul_add(
            target_covariance[0] * unit_x + target_covariance[1] * unit_y,
            unit_y * (target_covariance[2] * unit_x + target_covariance[3] * unit_y),
        );
        ensure!(
            target_variance.is_finite() && target_variance > 0.0,
            "shared-target release-mass proof target variance is invalid"
        );
        let grain_halfspace_boundary = TAIL_SIGMA * target_variance.sqrt() - radius_km;
        let mut density_upper = 0.0_f64;
        let mut density_bound_representable = true;
        for component in &model.conditional.components {
            let mean_projection = unit_x.mul_add(component.mean_x, unit_y * component.mean_y);
            let directional_variance = unit_x.mul_add(
                component.cov00 * unit_x + component.cov01 * unit_y,
                unit_y * (component.cov01 * unit_x + component.cov11 * unit_y),
            );
            ensure!(
                directional_variance.is_finite() && directional_variance > 0.0,
                "shared-target release-mass proof grain variance is invalid"
            );
            let separation = (grain_halfspace_boundary - mean_projection).max(0.0);
            let minimum_quadratic = separation * separation / directional_variance;
            let log_density_upper =
                component.log_normalization + component.log_weight - 0.5 * minimum_quadratic;
            ensure!(
                !log_density_upper.is_nan(),
                "shared-target release-mass proof density bound is invalid"
            );
            let component_density_upper = log_density_upper.exp();
            // Zero from `exp` underflow is not an upper bound on a positive
            // Gaussian density. Replace it with the smallest positive NORMAL
            // value, which is vastly larger than any value that underflowed to
            // zero and therefore remains conservative. Infinity is an upper
            // bound but cannot prove a useful packet probability.
            let component_density_upper = if component_density_upper == 0.0 {
                f64::MIN_POSITIVE
            } else if component_density_upper.is_infinite() {
                density_bound_representable = false;
                break;
            } else {
                component_density_upper
            };
            ensure!(
                component_density_upper.is_finite() && component_density_upper > 0.0,
                "shared-target release-mass proof density bound is invalid"
            );
            density_upper += component_density_upper;
        }
        if !density_bound_representable {
            continue;
        }
        let packet_capture_upper =
            (inputs.area_km2 * density_upper * (1.0 + 1.0e-12)).max(f64::MIN_POSITIVE);
        if !(packet_capture_upper.is_finite() && (0.0..1.0).contains(&packet_capture_upper)) {
            continue;
        }
        let region_no_hit_log =
            tail_weight_lower.ln() + packet_count_upper * (-packet_capture_upper).ln_1p();
        if region_no_hit_log > target_failure_log + NUMERIC_LOG_MARGIN {
            return Ok(Some(direction_index));
        }
    }
    Ok(None)
}

fn shared_target_mass_limit_is_provably_infeasible_from_prepared(
    inputs: &SharedTargetMassInputs<'_>,
    model: &PreparedModel,
    release_mass_limit_kg: f64,
) -> anyhow::Result<bool> {
    if inputs.deterministic_mass.required_mass_kg() >= release_mass_limit_kg {
        return Ok(true);
    }
    let mass_free_inputs = MassFreeProofInputs {
        scenario: inputs.deterministic_mass.scenario(),
        area_km2: inputs.area_km2,
        target_hit_probability: inputs.target_hit_probability,
    };
    Ok(
        mass_free_infeasibility_direction(mass_free_inputs, model, release_mass_limit_kg)?
            .is_some(),
    )
}

#[cfg(test)]
mod target_rule_tests {
    use super::*;

    fn fused_test_mass_inputs<'a>(
        scenario: SharedTargetScenario,
        projected_means_2d: &'a [f64],
        projected_covariances_2d: &'a [f64],
        mixture_weights: &'a [f64],
    ) -> SharedTargetMassInputs<'a> {
        let event = crate::mass_solver::MfJ2MassSolverEvent::new(
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
            1.0,
        );
        let config = crate::mass_solver::SolverConfig {
            xtol: 1.0e-6,
            rtol: 1.0e-5,
            maxiter: 50,
            mass_max: 1_000.0,
        };
        let (outcomes, _) =
            crate::mass_solver::solve_batch_events_mf_j2_with_evidence(&[event], &config);
        let operational_mass = outcomes
            .first()
            .expect("single-event batch must return one outcome")
            .operational_mass()
            .unwrap()
            .expect("converged MF-J2 row must issue operational mass");
        SharedTargetMassInputs {
            deterministic_mass: scenario.bind_deterministic_mass(operational_mass).unwrap(),
            projected_means_2d,
            projected_covariances_2d,
            mixture_weights,
            area_km2: std::f64::consts::PI * 0.00125_f64.powi(2),
            target_hit_probability: 0.5,
            grain_mass_kg: 1.0,
            covariance_minimum: 1.0e-12,
            covariance_maximum: 1.0,
        }
    }

    fn fused_test_scenario(identifier: &'static str) -> SharedTargetScenario {
        SharedTargetScenario::new(
            DustScenarioIdentity::named(identifier).unwrap(),
            1.0,
            1,
            100.0,
            SharedTargetPositionTreatment::AssumedNotObservedIsotropicEncounterBplaneSharedDraw,
            SharedTargetQuadrature::new(32, 16, 2.0e-2).unwrap(),
            DustMassClaim::ModelConditionedConservativeContactRequirement,
        )
        .unwrap()
    }

    #[test]
    fn conditional_capture_full_state_draw_reprojects_bplane_without_mass_replay() {
        let means = [1.0, 2.0, 0.5, 0.0, 0.0, 0.0];
        let covariances = [
            0.0025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0e-6,
        ];
        let weights = [1.0];
        let inputs = SharedTargetConditionalCaptureSourceInputs {
            component_means_6d: &means,
            component_covariances_6d: &covariances,
            mixture_weights: &weights,
            hf_velocity_mean: [3.0, 4.0, 5.0],
            area_km2: std::f64::consts::PI * 0.00125_f64.powi(2),
            released_mass_kg: 17.0,
            covariance_minimum: 1.0e-12,
            covariance_maximum: 1.0,
        };
        reset_prepare_model_call_count();
        let source = prepare_shared_target_conditional_capture_source(&inputs).unwrap();
        let [component] = source.components.as_slice() else {
            panic!("prepared conditional source retained wrong component count");
        };
        assert_eq!(
            component.position_mean.map(f64::to_bits),
            [1.0, 2.0, 0.5].map(f64::to_bits)
        );
        assert_eq!(
            component.position_covariance.map(f64::to_bits),
            [0.0025, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.04].map(f64::to_bits)
        );
        assert_eq!(component.log_weight.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            prepare_model_call_count(),
            0,
            "conditional source ran the minimum-mass preparation"
        );
        let draw = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let capture = source.evaluate_target_state(draw).unwrap();
        assert!((0.0..=1.0).contains(&capture.conditional_capture_probability()));
        assert!(
            (0.0..=inputs.released_mass_kg).contains(&capture.conditional_expected_hit_mass_kg())
        );
        assert_eq!(capture.projection_clamped(), 0);
        assert!((0.0..=f64::MAX).contains(&capture.maximum_disk_scale()));
        assert!((0.0..=f64::MAX).contains(&capture.maximum_fourth_order()));
        assert!((0.0..=f64::MAX).contains(&capture.maximum_e14_indicator()));

        let rotated_draw = [0.0, 0.0, 0.0, 2.0, -1.0, 0.5];
        let rotated = source.evaluate_target_state(rotated_draw).unwrap();
        assert_ne!(
            capture.conditional_capture_probability().to_bits(),
            rotated.conditional_capture_probability().to_bits(),
            "target velocity draw did not rotate and reproject the B-plane"
        );
        let translated_draw = [0.25, -0.5, 0.75, 0.0, 0.0, 0.0];
        let translated = source.evaluate_target_state(translated_draw).unwrap();
        assert_ne!(
            capture.conditional_capture_probability().to_bits(),
            translated.conditional_capture_probability().to_bits(),
            "target position draw did not shift and reproject the B-plane"
        );
        assert!(
            source
                .evaluate_target_state([0.0, 0.0, 0.0, 3.0, 4.0, 5.0])
                .is_err(),
            "zero-relative-velocity draw did not reject invalid encounter geometry"
        );
        assert_eq!(
            prepare_model_call_count(),
            0,
            "per-draw conditional evaluation ran the minimum-mass preparation"
        );
        assert!(source
            .evaluate_target_state([f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0])
            .is_err());

        let hostile_means = [f64::NAN, 2.0, 0.5, 0.0, 0.0, 0.0];
        let hostile_inputs = SharedTargetConditionalCaptureSourceInputs {
            component_means_6d: &hostile_means,
            ..inputs
        };
        assert!(prepare_shared_target_conditional_capture_source(&hostile_inputs).is_err());
        let hostile_weights = [-1.0];
        let hostile_inputs = SharedTargetConditionalCaptureSourceInputs {
            mixture_weights: &hostile_weights,
            ..inputs
        };
        assert!(prepare_shared_target_conditional_capture_source(&hostile_inputs).is_err());
        assert_eq!(
            prepare_model_call_count(),
            0,
            "hostile raw GMM reached minimum-mass preparation"
        );

        let released_mass = inputs.released_mass_kg;
        for probability in [
            0.0,
            f64::from_bits(1),
            f64::from_bits(1.0_f64.to_bits() - 1),
            1.0,
        ] {
            let expected = conditional_expected_hit_mass_kg(released_mass, probability).unwrap();
            assert!((0.0..=released_mass).contains(&expected));
        }
        assert_eq!(
            conditional_expected_hit_mass_kg(released_mass, 0.0)
                .unwrap()
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            conditional_expected_hit_mass_kg(released_mass, 1.0)
                .unwrap()
                .to_bits(),
            released_mass.to_bits()
        );
        assert!(conditional_expected_hit_mass_kg(released_mass, -f64::from_bits(1)).is_err());
        assert!(conditional_expected_hit_mass_kg(
            released_mass,
            f64::from_bits(1.0_f64.to_bits() + 1),
        )
        .is_err());
    }

    #[test]
    fn conditional_capture_fused_projection_matches_projected_c12_bits() {
        let means = [
            1.0, 2.0, 0.5, 9.0, 8.0, 7.0, 99.0, 98.0, 97.0, 3.0, 2.0, 1.0, -0.5, 0.75, 1.25, 6.0,
            5.0, 4.0,
        ];
        let mut covariances = [0.0; 108];
        covariances[0] = 0.0025;
        covariances[1] = 0.0005;
        covariances[6] = 0.0005;
        covariances[7] = 0.01;
        covariances[14] = 0.04;
        covariances[21] = 1.0e-6;
        covariances[28] = 1.0e-6;
        covariances[35] = 1.0e-6;
        covariances[36] = 0.006;
        covariances[43] = 0.015;
        covariances[50] = 0.025;
        covariances[57] = 1.0e-6;
        covariances[64] = 1.0e-6;
        covariances[71] = 1.0e-6;
        covariances[72] = 0.005;
        covariances[73] = -0.0007;
        covariances[78] = -0.0007;
        covariances[79] = 0.02;
        covariances[86] = 0.03;
        covariances[93] = 2.0e-6;
        covariances[100] = 2.0e-6;
        covariances[107] = 2.0e-6;
        let weights = [0.25, 0.0, 0.75];
        let hf_velocity_mean = [3.0, 4.0, 5.0];
        let area_km2 = std::f64::consts::PI * 0.00125_f64.powi(2);
        let released_mass_kg = 17.0;
        let covariance_minimum = 1.0e-12;
        let covariance_maximum = 1.0;
        let draw = [0.2, -0.1, 0.3, 0.5, -1.0, 2.0];
        let source = prepare_shared_target_conditional_capture_source(
            &SharedTargetConditionalCaptureSourceInputs {
                component_means_6d: &means,
                component_covariances_6d: &covariances,
                mixture_weights: &weights,
                hf_velocity_mean,
                area_km2,
                released_mass_kg,
                covariance_minimum,
                covariance_maximum,
            },
        )
        .unwrap();

        let projection =
            project_shared_target_bplane_components(&SharedTargetBplaneProjectionInputs {
                component_means_6d: &means,
                component_covariances_6d: &covariances,
                target_state: &draw,
                hf_velocity_mean: &hf_velocity_mean,
                covariance_minimum,
                covariance_maximum,
            })
            .unwrap();
        let weight_sum = weights.iter().sum::<f64>();
        let projected_components = projection
            .projected_means_2d()
            .chunks_exact(2)
            .zip(projection.projected_covariances_2d().chunks_exact(4))
            .zip(weights)
            .filter_map(|((mean, covariance), weight)| {
                if weight == 0.0 {
                    return None;
                }
                let [mean_x, mean_y] = <[f64; 2]>::try_from(mean).unwrap();
                let covariance = <[f64; 4]>::try_from(covariance).unwrap();
                Some(prepare_conditional_component(
                    mean_x,
                    mean_y,
                    covariance,
                    (weight / weight_sum).ln(),
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .unwrap();
        let projected = PreparedConditionalMixture {
            components: projected_components,
            radius_km: (area_km2 / std::f64::consts::PI).sqrt(),
            log_area_km2: area_km2.ln(),
        };
        let projected_capture = conditional_mixture_capture_c12(&projected, 0.0, 0.0).unwrap();

        let (fused, fused_projection_clamped) =
            prepare_conditional_mixture_for_target_state(&source, draw).unwrap();
        let fused_capture = conditional_mixture_capture_c12(&fused, 0.0, 0.0).unwrap();
        assert_eq!(fused_projection_clamped, projection.projection_clamped());
        assert_eq!(
            fused_capture.log_probability.to_bits(),
            projected_capture.log_probability.to_bits()
        );
        assert_eq!(
            fused_capture.maximum_disk_scale.to_bits(),
            projected_capture.maximum_disk_scale.to_bits()
        );
        assert_eq!(
            fused_capture.maximum_fourth_order.to_bits(),
            projected_capture.maximum_fourth_order.to_bits()
        );
        assert_eq!(
            fused_capture.maximum_e14_indicator.to_bits(),
            projected_capture.maximum_e14_indicator.to_bits()
        );
        assert_eq!(
            fused_capture.component_evaluations,
            projected_capture.component_evaluations
        );
        let projected_probability =
            probability_from_log(projected_capture.log_probability, "projected capture").unwrap();
        let fused_probability =
            probability_from_log(fused_capture.log_probability, "fused capture").unwrap();
        assert_eq!(fused_probability.to_bits(), projected_probability.to_bits());
        assert_eq!(
            conditional_expected_hit_mass_kg(released_mass_kg, fused_probability)
                .unwrap()
                .to_bits(),
            conditional_expected_hit_mass_kg(released_mass_kg, projected_probability)
                .unwrap()
                .to_bits()
        );
        let estimate = source.evaluate_target_state(draw).unwrap();
        assert_eq!(
            estimate.projection_clamped(),
            projection.projection_clamped()
        );
        assert_eq!(
            estimate.conditional_capture_probability().to_bits(),
            projected_probability.to_bits()
        );
        assert_eq!(
            estimate.conditional_expected_hit_mass_kg().to_bits(),
            conditional_expected_hit_mass_kg(released_mass_kg, projected_probability)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            estimate.maximum_disk_scale().to_bits(),
            projected_capture.maximum_disk_scale.to_bits()
        );
        assert_eq!(
            estimate.maximum_fourth_order().to_bits(),
            projected_capture.maximum_fourth_order.to_bits()
        );
        assert_eq!(
            estimate.maximum_e14_indicator().to_bits(),
            projected_capture.maximum_e14_indicator.to_bits()
        );
    }

    #[test]
    fn fused_limit_solve_prepares_once_and_preserves_estimate_bits() {
        let means = [0.0, 0.0];
        let covariances = [0.0025, 0.0, 0.0, 0.0025];
        let weights = [1.0];
        let inputs = fused_test_mass_inputs(
            fused_test_scenario("fused-limit-solve-parity"),
            &means,
            &covariances,
            &weights,
        );
        let expected = shared_target_contact_mass_requirement(&inputs).unwrap();

        reset_prepare_model_call_count();
        let actual = shared_target_contact_mass_requirement_under_limit(&inputs, 1_000.0)
            .unwrap()
            .expect("near row must continue through exact solve");

        assert_eq!(prepare_model_call_count(), 1);
        assert_eq!(actual.witness(), expected.witness());
        for (actual, expected) in [
            (actual.target_area_m2(), expected.target_area_m2()),
            (
                actual.target_disk_radius_m(),
                expected.target_disk_radius_m(),
            ),
            (
                actual.target_hit_probability(),
                expected.target_hit_probability(),
            ),
            (actual.grain_mass_kg(), expected.grain_mass_kg()),
            (
                actual.deterministic_required_mass_kg(),
                expected.deterministic_required_mass_kg(),
            ),
            (actual.release_mass_kg(), expected.release_mass_kg()),
            (actual.no_hit_probability(), expected.no_hit_probability()),
            (
                actual.probability_predecessor_no_hit_probability(),
                expected.probability_predecessor_no_hit_probability(),
            ),
            (
                actual.expected_conditional_capture_probability(),
                expected.expected_conditional_capture_probability(),
            ),
        ] {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(
            actual.deterministic_mass_authority_id(),
            expected.deterministic_mass_authority_id()
        );
        assert_eq!(
            actual.effective_packet_count(),
            expected.effective_packet_count()
        );
    }

    #[test]
    fn fused_limit_rejection_prepares_once_without_solving() {
        let means = [0.1, 0.0];
        let covariances = [1.0e-4, 0.0, 0.0, 1.0e-4];
        let weights = [1.0];
        let scenario = SharedTargetScenario::new(
            DustScenarioIdentity::named("fused-limit-rejection").unwrap(),
            1.0,
            1,
            100.0,
            SharedTargetPositionTreatment::AssumedNotObservedIsotropicEncounterBplaneSharedDraw,
            SharedTargetQuadrature::new(192, 32, 1.0e-4).unwrap(),
            DustMassClaim::ModelConditionedConservativeContactRequirement,
        )
        .unwrap();
        let mut inputs = fused_test_mass_inputs(scenario, &means, &covariances, &weights);
        inputs.target_hit_probability = 0.99;
        inputs.grain_mass_kg = 6.450_736_915_371_043e-10;
        inputs.covariance_maximum = 1.0e12;

        reset_prepare_model_call_count();
        let actual = shared_target_contact_mass_requirement_under_limit(&inputs, 1_000.0)
            .expect("proof inputs must be valid");

        assert!(actual.is_none(), "far row escaped hard-limit proof");
        assert_eq!(prepare_model_call_count(), 1);
    }

    #[test]
    fn private_hard_limit_proof_keeps_exact_floor_and_raw_probability_seams() {
        let means = [0.0, 0.0];
        let covariances = [0.04, 0.0, 0.0, 0.04];
        let weights = [1.0];
        let mut inputs = fused_test_mass_inputs(
            fused_test_scenario("private-hard-limit-seams"),
            &means,
            &covariances,
            &weights,
        );
        inputs.target_hit_probability = 1.0e-6;
        inputs.covariance_maximum = 1.0e12;
        let replay_inputs = shared_target_replay_inputs(&inputs);
        let model = prepare_model(&replay_inputs).unwrap();
        let floor = inputs.deterministic_mass.required_mass_kg();
        let below = f64::from_bits(floor.to_bits() - 1);
        let above = f64::from_bits(floor.to_bits() + 1);

        assert!(
            shared_target_mass_limit_is_provably_infeasible_from_prepared(&inputs, &model, below)
                .unwrap()
        );
        assert!(
            shared_target_mass_limit_is_provably_infeasible_from_prepared(&inputs, &model, floor)
                .unwrap()
        );
        assert!(
            !shared_target_mass_limit_is_provably_infeasible_from_prepared(&inputs, &model, above)
                .unwrap()
        );

        inputs.target_hit_probability = 0.799;
        let replay_inputs = shared_target_replay_inputs(&inputs);
        let model = prepare_model(&replay_inputs).unwrap();
        assert!(
            !shared_target_mass_limit_is_provably_infeasible_from_prepared(
                &inputs, &model, 1_000.0
            )
            .unwrap(),
            "analytic proof consumed numerical reserve instead of raw 1-p"
        );
    }

    fn dense_disk_reference(
        component: &PreparedComponent,
        radius: f64,
        target_x: f64,
        target_y: f64,
        radial_count: u32,
        angular_count: u32,
    ) -> f64 {
        let dr = radius / f64::from(radial_count);
        let dtheta = std::f64::consts::TAU / f64::from(angular_count);
        let mut total = 0.0;
        for radial_index in 0..radial_count {
            let disk_radius = (f64::from(radial_index) + 0.5) * dr;
            for angular_index in 0..angular_count {
                let angle = (f64::from(angular_index) + 0.5) * dtheta;
                let dx = disk_radius.mul_add(angle.cos(), target_x - component.mean_x);
                let dy = disk_radius.mul_add(angle.sin(), target_y - component.mean_y);
                let quadratic = super::super::mahalanobis_2x2(
                    dx,
                    dy,
                    component.inv00,
                    component.inv01,
                    component.inv11,
                );
                total += (component.log_normalization - 0.5 * quadratic).exp()
                    * disk_radius
                    * dr
                    * dtheta;
            }
        }
        total
    }

    fn production_envelope_component(sigma_0: f64, sigma_1: f64, angle: f64) -> PreparedComponent {
        let (sin, cos) = angle.sin_cos();
        let precision_0 = 1.0 / (sigma_0 * sigma_0);
        let precision_1 = 1.0 / (sigma_1 * sigma_1);
        let inv00 = cos.mul_add(cos * precision_0, sin * sin * precision_1);
        let inv01 = sin * cos * (precision_0 - precision_1);
        let inv11 = sin.mul_add(sin * precision_0, cos * cos * precision_1);
        let determinant = (-inv01).mul_add(inv01, inv00 * inv11);
        let (major_precision, minor_precision, major_axis_x, major_axis_y) =
            if precision_0 >= precision_1 {
                (precision_0, precision_1, cos, sin)
            } else {
                (precision_1, precision_0, -sin, cos)
            };
        PreparedComponent {
            mean_x: 0.0,
            mean_y: 0.0,
            cov00: inv11 / determinant,
            cov01: -inv01 / determinant,
            cov11: inv00 / determinant,
            inv00,
            inv01,
            inv11,
            log_normalization: -std::f64::consts::TAU.ln() - (sigma_0 * sigma_1).ln(),
            log_weight: 0.0,
            major_precision,
            minor_precision,
            major_precision_sqrt: major_precision.sqrt(),
            minor_precision_sqrt: minor_precision.sqrt(),
            major_axis_x,
            major_axis_y,
        }
    }

    fn disk_only_model(component: PreparedComponent, radius_km: f64) -> PreparedModel {
        PreparedModel {
            conditional: PreparedConditionalMixture {
                components: vec![component],
                radius_km,
                log_area_km2: (std::f64::consts::PI * radius_km * radius_km).ln(),
            },
            target_cholesky: [1.0, 0.0, 1.0],
            target_frame: [1.0, 0.0, -0.0, 1.0],
            target_separability: 1.0,
            deterministic_floor_count: 1,
            packet_mass_kg: 1.0,
            policy_threshold_probability: 0.5,
            policy_threshold_log: 0.5_f64.ln(),
        }
    }

    #[test]
    fn mass_free_hard_limit_proof_ignores_deterministic_floor() {
        let scenario = SharedTargetScenario::new(
            DustScenarioIdentity::named("mass-free-private-proof-test").unwrap(),
            1.0,
            1,
            100.0,
            SharedTargetPositionTreatment::AssumedNotObservedIsotropicEncounterBplaneSharedDraw,
            SharedTargetQuadrature::new(192, 32, 1.0e-4).unwrap(),
            DustMassClaim::ModelConditionedConservativeContactRequirement,
        )
        .unwrap();
        let means = [0.1, 0.0];
        let covariances = [1.0e-4, 0.0, 0.0, 1.0e-4];
        let weights = [1.0];
        let replay = SharedTargetReplayInputs {
            scenario,
            deterministic_required_mass_kg: 1.0,
            projected_means_2d: &means,
            projected_covariances_2d: &covariances,
            mixture_weights: &weights,
            area_km2: std::f64::consts::PI * 0.00125_f64.powi(2),
            target_hit_probability: 0.99,
            grain_mass_kg: 6.450_736_915_371_043e-10,
            covariance_minimum: 1.0e-12,
            covariance_maximum: 1.0e12,
        };
        let low_floor_model = prepare_model(&replay).unwrap();
        let high_floor_model = prepare_model(&SharedTargetReplayInputs {
            deterministic_required_mass_kg: 999.0,
            ..replay
        })
        .unwrap();
        let proof_inputs = MassFreeProofInputs {
            scenario,
            area_km2: replay.area_km2,
            target_hit_probability: replay.target_hit_probability,
        };
        let low =
            mass_free_infeasibility_direction(proof_inputs, &low_floor_model, 1_000.0).unwrap();
        let high =
            mass_free_infeasibility_direction(proof_inputs, &high_floor_model, 1_000.0).unwrap();
        assert_eq!(low, high, "mass-free proof read deterministic floor state");
        assert!(low.is_some(), "far cloud escaped mass-free proof");
    }

    #[test]
    fn binary64_authority_tokens_and_single_witness_shape_are_exact() {
        let source = include_str!("shared_target.rs");
        let method = ["c12-e14-validated-binary64-", "log-minimum-v1"].concat();
        let claim = ["model-conditioned-conservative-", "contact-requirement-v1"].concat();
        let certificate = ["binary64-reserved-threshold-", "minimum-v1"].concat();
        let witness = ["pub struct C12Binary64Log", "MinimumWitnessV1"].concat();
        assert_eq!(source.matches(&method).count(), 1);
        assert_eq!(source.matches(&claim).count(), 1);
        assert_eq!(source.matches(&certificate).count(), 1);
        assert_eq!(source.matches(&witness).count(), 1);
        let production = source
            .split("#[cfg(test)]\nmod target_rule_tests")
            .next()
            .expect("production source precedes tests");
        for forbidden in [
            "DoubleDouble",
            "OutwardInterval",
            "NumericalQuadratureV1",
            "TaylorRouteUnavailable",
            "SharedTargetDiskIntegrationWitness",
            "SharedTargetQuadratureWitness",
            "shared_target_release_mass_lower_bound",
            "numerical_disk_points_for_route",
        ] {
            assert!(
                !production.contains(forbidden),
                "obsolete production token: {forbidden}"
            );
        }
        let root = include_str!("lib.rs");
        assert!(!root.contains("shared_target_release_mass_lower_bound"));
        assert!(!root.contains("SharedTargetDiskIntegration"));
    }

    #[test]
    fn pruned_threshold_search_matches_exhaustive_search_bit_for_bit() {
        // The pruned front end must return the exact struct the exhaustive
        // search returns — same count, same selected/predecessor log BITS —
        // across grids spanning tiny and huge answer counts, flat and spiky
        // weight profiles, and thresholds close to and far from the terms.
        let mut cases = 0_u64;
        for scale in [1.0e-9_f64, 1.0e-6, 1.0e-3, 0.5, 5.0] {
            for spread in [1.0_f64, 4.0, 64.0] {
                for threshold in [-0.1_f64, -1.0, -6.0, -40.0, -700.0] {
                    let mut failures = Vec::new();
                    let mut weights = Vec::new();
                    for index in 0..96_u32 {
                        let depth = f64::from(index + 1) / 96.0;
                        failures.push(-scale * (1.0 + spread * depth * depth));
                        weights.push(1.0 + depth * 3.0);
                    }
                    let prepared = PreparedBinary64TargetWeights::new(&weights).unwrap();
                    let fast = minimum_binary64_threshold_count(&failures, &prepared, threshold);
                    let slow = minimum_binary64_threshold_count_exhaustive(
                        &failures, &prepared, threshold,
                    );
                    match (fast, slow) {
                        (Ok(fast), Ok(slow)) => {
                            assert_eq!(fast, slow, "scale={scale} spread={spread} thr={threshold}");
                            cases += 1;
                        }
                        (Err(fast), Err(slow)) => {
                            assert_eq!(
                                fast.to_string(),
                                slow.to_string(),
                                "scale={scale} spread={spread} thr={threshold}"
                            );
                        }
                        (fast, slow) => panic!(
                            "pruned/exhaustive disagree on outcome kind: \
                             scale={scale} spread={spread} thr={threshold} \
                             fast_ok={} slow_ok={}",
                            fast.is_ok(),
                            slow.is_ok()
                        ),
                    }
                }
            }
        }
        assert!(
            cases >= 60,
            "equivalence sweep degenerated: {cases} Ok cases"
        );
    }

    #[test]
    fn two_pass_no_hit_fold_matches_sequential_fold_within_ulps() {
        // Characterize the 2026-08-27 bit move: the two-pass max-then-sum
        // fold must stay within a few ulps of the retired sequential fold on
        // production-shaped grids, and agree exactly on the edges the seal
        // reasons about (all-underflow, single term, ties at the maximum).
        // Two metrics, either may pass: a few-ulp relative envelope away from
        // zero, or an absolute 1e-12-nat envelope near it (ulp counts explode
        // across exponents there while the disagreement stays in the same
        // rounding class — these are logs, so nats are the natural absolute
        // unit).
        #[expect(
            clippy::float_cmp,
            reason = "exact bit agreement is the first, strongest tier of the envelope"
        )]
        fn agree(a: f64, b: f64) -> bool {
            a == b
                || (a - b).abs() <= 1.0e-12
                || (a.signum() == b.signum() && a.to_bits().abs_diff(b.to_bits()) <= 128)
        }
        for scale in [1.0e-12_f64, 1.0e-6, 1.0e-2, 1.0, 30.0] {
            for spread in [1.0_f64, 8.0, 128.0] {
                for count in [1_u64, 7, 1_000, 1_000_000, 1_000_000_000_000] {
                    let mut failures = Vec::new();
                    let mut weights = Vec::new();
                    for index in 0..192_u32 {
                        let depth = f64::from(index + 1) / 192.0;
                        failures.push(-scale * (1.0 + spread * depth * depth) / 1.0e6);
                        weights.push(1.0 + depth * 2.0);
                    }
                    let prepared = PreparedBinary64TargetWeights::new(&weights).unwrap();
                    let count_f64 = u64_to_exact_binary64(count).unwrap();
                    let fast = prepared.normalized_log_no_hit(&failures, count).unwrap();
                    let slow = prepared
                        .sequential_normalized_log_no_hit(&failures, count_f64)
                        .unwrap();
                    assert!(
                        agree(fast, slow),
                        "two-pass fold drifted: scale={scale} spread={spread} \
                         count={count} fast={fast} slow={slow}"
                    );
                }
            }
        }

        let prepared = PreparedBinary64TargetWeights::new(&[1.0, 2.0, 1.0]).unwrap();
        // All terms underflow the normalization by more than the exp bound:
        // both paths must agree the sum is the anchor exactly.
        let deep = prepared
            .normalized_log_no_hit(&[-900.0, -0.0, -900.0], 1)
            .unwrap();
        let deep_sequential = prepared
            .sequential_normalized_log_no_hit(&[-900.0, -0.0, -900.0], 1.0)
            .unwrap();
        assert_eq!(deep.to_bits(), deep_sequential.to_bits());
        // Single live term: two-pass is anchor + ln_1p(0.0) — exact.
        let single = PreparedBinary64TargetWeights::new(&[3.0]).unwrap();
        assert_eq!(
            single.normalized_log_no_hit(&[-0.25], 4).unwrap().to_bits(),
            single
                .sequential_normalized_log_no_hit(&[-0.25], 4.0)
                .unwrap()
                .to_bits()
        );
        // Ties at the maximum contribute exp(0) = 1 each through `rest`.
        let tied = PreparedBinary64TargetWeights::new(&[1.0, 1.0]).unwrap();
        let tied_fast = tied.normalized_log_no_hit(&[-0.5, -0.5], 2).unwrap();
        let tied_slow = tied
            .sequential_normalized_log_no_hit(&[-0.5, -0.5], 2.0)
            .unwrap();
        assert!(agree(tied_fast, tied_slow));
    }

    #[test]
    fn logaddexp_shortcut_agrees_with_full_path() {
        // Sweep the underflow boundary on both sides: the shortcut must be
        // bit-identical to the full exp/ln_1p expression everywhere, and the
        // guarded region must actually be where exp underflows on this libm.
        let full = |left: f64, right: f64| -> f64 {
            let maximum = left.max(right);
            let minimum = left.min(right);
            maximum + (minimum - maximum).exp().ln_1p()
        };
        for step in 0..=3200_u32 {
            let delta = -800.0 + f64::from(step) * 0.03125;
            for base in [-1.0e-3_f64, -0.0, -1.0, -650.0] {
                let a = base;
                let b = base + delta;
                assert_eq!(
                    binary64_logaddexp(a, b).to_bits(),
                    full(a, b).to_bits(),
                    "logaddexp diverges at base {base} delta {delta}"
                );
            }
        }
        assert_eq!(
            LOGADDEXP_EXP_UNDERFLOW_BOUND.exp().to_bits(),
            0.0_f64.to_bits()
        );
        // Two nats above the bound exp is a NONZERO subnormal -- the margin
        // is load-bearing, so the guard must sit at -746, not at -744.
        assert!((LOGADDEXP_EXP_UNDERFLOW_BOUND + 2.0).exp() > 0.0);
    }

    #[test]
    fn stable_log_primitives_cover_infinities_and_normalization() {
        // Bit equality, not float equality: these are exact identities of the
        // log primitives, and `to_bits` says so without tripping a lint that
        // exists to catch approximate comparisons written as exact ones.
        assert_eq!(
            binary64_logaddexp(f64::NEG_INFINITY, f64::NEG_INFINITY).to_bits(),
            f64::NEG_INFINITY.to_bits()
        );
        assert_eq!(
            binary64_logaddexp(f64::NEG_INFINITY, -3.0).to_bits(),
            (-3.0_f64).to_bits()
        );
        assert_eq!(
            binary64_weighted_logsumexp(&[f64::NEG_INFINITY, f64::NEG_INFINITY], &[1.0, 2.0])
                .expect("all-zero functional")
                .to_bits(),
            f64::NEG_INFINITY.to_bits()
        );
        let base =
            binary64_weighted_logsumexp(&[-2.0, -5.0], &[1.0, 3.0]).expect("base weighted LSE");
        let scaled =
            binary64_weighted_logsumexp(&[-2.0, -5.0], &[10.0, 30.0]).expect("scaled weighted LSE");
        assert!((base - scaled).abs() <= f64::EPSILON * base.abs());
        let base_weights = PreparedBinary64TargetWeights::new(&[1.0, 3.0]).unwrap();
        let scaled_weights = PreparedBinary64TargetWeights::new(&[10.0, 30.0]).unwrap();
        let base_count =
            minimum_binary64_threshold_count(&[-0.2, -0.4], &base_weights, -1.0).unwrap();
        let scaled_count =
            minimum_binary64_threshold_count(&[-0.2, -0.4], &scaled_weights, -1.0).unwrap();
        assert_eq!(
            base_count.probability_count(),
            scaled_count.probability_count()
        );
        assert_eq!(
            log_one_minus_exp(f64::NEG_INFINITY).unwrap().to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            log_one_minus_exp(0.0).unwrap().to_bits(),
            f64::NEG_INFINITY.to_bits()
        );
    }

    #[test]
    fn binary64_count_search_is_minimal_and_reaches_exact_ceiling() {
        let weights = PreparedBinary64TargetWeights::new(&[7.0]).unwrap();
        let half_log = 0.5_f64.ln();
        let minimum = minimum_binary64_threshold_count(&[half_log], &weights, 0.25_f64.ln())
            .expect("inclusive selected threshold");
        assert_eq!(minimum.probability_count(), 2);
        assert!(f64::from_bits(minimum.selected_log_no_hit_bits) <= 0.25_f64.ln());
        assert!(f64::from_bits(minimum.predecessor_log_no_hit_bits()) > 0.25_f64.ln());

        let one_ulp_log = -2.0_f64.powi(-52);
        let ceiling_log =
            one_ulp_log * u64_to_exact_binary64(MAX_EXACT_BINARY64_PACKET_COUNT).unwrap();
        let ceiling = minimum_binary64_threshold_count(&[one_ulp_log], &weights, ceiling_log)
            .expect("exact ceiling remains searchable");
        assert_eq!(ceiling.probability_count(), MAX_EXACT_BINARY64_PACKET_COUNT);
        let error =
            minimum_binary64_threshold_count(&[one_ulp_log], &weights, ceiling_log - f64::EPSILON)
                .expect_err("threshold beyond exact count must be typed");
        assert!(error
            .downcast_ref::<Binary64PacketCountUnrepresentable>()
            .is_some());
    }

    #[test]
    fn packet_count_governor_uses_independent_probability_minimum() {
        assert_eq!(
            select_shared_target_packet_count(4, 5),
            (5, SharedTargetPacketCountGovernor::DeterministicFloor)
        );
        assert_eq!(
            select_shared_target_packet_count(5, 5),
            (5, SharedTargetPacketCountGovernor::Probability)
        );
        assert_eq!(
            select_shared_target_packet_count(6, 5),
            (6, SharedTargetPacketCountGovernor::Probability)
        );
        let weights = PreparedBinary64TargetWeights::new(&[1.0]).unwrap();
        let error = decide_binary64_packet_count(&[-0.0], &weights, -1.0, 1)
            .expect_err("zero binary64 contact exceeds exact count domain");
        assert!(error
            .downcast_ref::<Binary64PacketCountUnrepresentable>()
            .is_some());

        let probability_minimum = Binary64ThresholdMinimum {
            probability_count: 5,
            selected_log_no_hit_bits: (-2.0_f64).to_bits(),
            predecessor_log_no_hit_bits: (-1.0_f64).to_bits(),
        };
        let probability_decision = Binary64PacketCountDecision {
            probability_minimum,
            final_count: 5,
            governor: SharedTargetPacketCountGovernor::Probability,
        };
        assert_eq!(
            final_selected_log_no_hit(&[], &weights, probability_decision)
                .expect("probability governor reuses search terminal")
                .to_bits(),
            (-2.0_f64).to_bits()
        );
        let floor_decision = Binary64PacketCountDecision {
            probability_minimum,
            final_count: 6,
            governor: SharedTargetPacketCountGovernor::DeterministicFloor,
        };
        assert_eq!(
            final_selected_log_no_hit(&[-0.5], &weights, floor_decision)
                .expect("floor governor evaluates its larger final count")
                .to_bits(),
            (-3.0_f64).to_bits()
        );
    }

    #[test]
    fn deterministic_floor_corrects_rounded_integer_quotient() {
        let packet_mass = f64::from_bits(0x3dd0_d714_27a0_0000);
        let required_mass = f64::from_bits(0x3f35_9412_7e1a_fec3);
        let naive = super::super::checked_ceil_packet_count(
            required_mass / packet_mass,
            "hostile deterministic floor",
        )
        .unwrap();
        assert_eq!(naive, 5_374_441);
        assert_eq!(
            (u64_to_exact_binary64(naive).unwrap() * packet_mass).to_bits(),
            0x3f35_9412_7e1a_fec2
        );
        let corrected = deterministic_floor_packet_count(required_mass, packet_mass).unwrap();
        assert_eq!(corrected, naive + 1);
        assert!(u64_to_exact_binary64(corrected).unwrap() * packet_mass >= required_mass);
    }

    #[test]
    fn c12_matches_dense_disk_oracle_for_all_693_production_envelope_cases() {
        let radius_km = 0.00125_f64;
        let mut compared = 0_usize;
        for (sigma_0, sigma_1) in [(0.032, 0.064), (0.040, 0.160), (0.064, 0.032)] {
            for angle in [0.0_f64, 0.37, 0.91] {
                let component = production_envelope_component(sigma_0, sigma_1, angle);
                let model = disk_only_model(component, radius_km);
                let (sin, cos) = angle.sin_cos();
                for standard_0 in [
                    -20.0_f64, -16.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 16.0, 20.0,
                ] {
                    for standard_1 in [-20.0_f64, -12.0, -6.0, 0.0, 6.0, 12.0, 20.0] {
                        let target_x =
                            cos.mul_add(sigma_0 * standard_0, -sin * sigma_1 * standard_1);
                        let target_y =
                            sin.mul_add(sigma_0 * standard_0, cos * sigma_1 * standard_1);
                        let c12 = conditional_component_capture_c12(
                            &component,
                            &model.conditional,
                            target_x,
                            target_y,
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "C12 rejected production-envelope case: sigmas=({sigma_0},{sigma_1}) \
                                 angle={angle} z=({standard_0},{standard_1}): {error}"
                            )
                        })
                        .log_probability
                        .exp();
                        let dense = dense_disk_reference(
                            &component, radius_km, target_x, target_y, 64, 256,
                        );
                        let relative = ((c12 - dense) / dense).abs();
                        assert!(
                            relative <= 1.0e-4,
                            "C12 relative error {relative} exceeds envelope budget: \
                             sigmas=({sigma_0},{sigma_1}) angle={angle} \
                             z=({standard_0},{standard_1})"
                        );
                        compared += 1;
                    }
                }
            }
        }
        assert_eq!(compared, 693, "production-envelope case count drifted");
    }

    #[test]
    fn c12_total_route_accepts_scale_above_retired_guard() {
        let radius_km = 0.00125;
        let component = production_envelope_component(0.020, 0.050, 0.37);
        let model = disk_only_model(component, radius_km);
        let capture =
            conditional_component_capture_c12(&component, &model.conditional, 0.040, -0.025)
                .expect("finite C12 geometry is total");
        assert!(capture.disk_scale > 0.002);
        let c12 = capture.log_probability.exp();
        let dense = dense_disk_reference(&component, radius_km, 0.040, -0.025, 128, 512);
        assert!(((c12 - dense) / dense).abs() <= 1.0e-4);
    }

    #[test]
    fn weights_stay_outside_packet_exponent_and_c12_matches_isotropic_identity() {
        let weights = PreparedBinary64TargetWeights::new(&[1.0, 3.0]).unwrap();
        let no_hit = weights
            .normalized_log_no_hit(&[0.5_f64.ln(), 0.25_f64.ln()], 2)
            .unwrap()
            .exp();
        assert!(no_hit >= 0.109_375 && no_hit <= next_up(0.109_375));

        let precision = 400.0;
        let component = PreparedComponent {
            mean_x: 0.0,
            mean_y: 0.0,
            cov00: 1.0 / precision,
            cov01: 0.0,
            cov11: 1.0 / precision,
            inv00: precision,
            inv01: 0.0,
            inv11: precision,
            log_normalization: precision.ln() - std::f64::consts::TAU.ln(),
            log_weight: 0.0,
            major_precision: precision,
            minor_precision: precision,
            major_precision_sqrt: precision.sqrt(),
            minor_precision_sqrt: precision.sqrt(),
            major_axis_x: 1.0,
            major_axis_y: 0.0,
        };
        let radius_squared = 0.00125_f64.powi(2);
        let coefficients =
            twelfth_order_taylor_coefficients(&component, 0.0, 0.0, radius_squared).unwrap();
        let c12 = std::f64::consts::PI
            * radius_squared
            * component.log_normalization.exp()
            * coefficients.iter().sum::<f64>();
        let exact = -(-0.5 * precision * radius_squared).exp_m1();
        assert!((c12 - exact).abs() <= 8.0 * f64::EPSILON * exact);
        let dense = dense_disk_reference(&component, radius_squared.sqrt(), 0.0, 0.0, 128, 256);
        assert!((c12 - dense).abs() <= 2.0e-8 * exact);
    }
}
