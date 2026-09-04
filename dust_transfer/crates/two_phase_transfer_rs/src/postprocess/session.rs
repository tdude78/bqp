use crate::py_config::PhysicsConfigError;
use crate::py_config::{PhysicsConfig, PostprocessConfig};
use crate::types::{CompactTransferCandidate, PlanContext, TargetPropagationAuthority};

use dust_estimates_rs::mass_solver::HfContext;
use lightyear_odeint_rs::precomputed_ephem::EphemerisCoverageError;
use lightyear_odeint_rs::types::ForceFlags;
use sha2::{Digest, Sha256};
#[cfg(any(test, feature = "bench-internal"))]
use smallvec::SmallVec;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

mod natural_conjunction;
pub use natural_conjunction::{
    natural_state_position_residual_km, natural_state_velocity_residual_km_s,
    NaturalConjunctionEnclosure, NaturalConjunctionFatalError, NaturalConjunctionInfeasible,
    NaturalConjunctionInputError, NaturalConjunctionOutcome, NaturalConjunctionScanAnchor,
    NaturalConjunctionWitnessResidual, NaturalObjectIdentity, NaturalObjectInput,
    VerifiedNaturalConjunction, NATURAL_DENSE_ARC_AUTHORITY_CEILING_KM,
};

#[cfg(any(test, feature = "bench-internal"))]
use super::distribution::compute_corrected_dust_state_summary_with_runtime;
#[cfg(feature = "solver-qualification")]
use super::distribution::{
    build_ctx, build_release_control_at_fraction_observed,
    materialize_corrected_dust_distribution_observed, ObservedMaterializationRequest,
    ObservedReleaseControlCoreRequest,
};
use super::distribution::{
    build_release_control_at_fraction, compute_corrected_dust_state,
    materialize_corrected_dust_distribution, AuthoritativeReleaseDistribution,
    ConjunctionDiagnostic, CorrectedDustStateRequest, PostprocessControl, PostprocessControlStatus,
    PostprocessDistributionStatus, PostprocessDustDistribution, SummaryPlanInputs,
};
use super::observer::UnobservedPostprocessLeg;
#[cfg(feature = "solver-qualification")]
use super::QualificationLegTrace;
use super::{compact_candidate_is_postprocess_coherent, DEFAULT_NUM_DISTS, MAX_DUST_COMPONENTS};

/// Failure while creating a reusable postprocess session.
///
/// This is intentionally finite: callers must retain the session failure
/// domain instead of collapsing it into an opaque diagnostic string.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PostprocessSessionError {
    /// Postprocess configuration did not satisfy session policy.
    PostprocessConfiguration,
    /// Physics configuration could not select a supported integrator.
    PhysicsConfiguration(PhysicsConfigError),
    /// Strict-HF gravity coefficients were not supplied.
    StrictHfGravityMissing,
    /// Strict-HF gravity coefficients were malformed or inconsistent.
    StrictHfGravityInvalid,
    /// Embedded strict-HF gravity bytes could not be loaded.
    StrictHfEmbeddedGravityUnavailable,
    /// Required strict-HF JB2008 drivers were unavailable.
    StrictHfJb2008Unavailable,
}

impl fmt::Display for PostprocessSessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PostprocessConfiguration => {
                formatter.write_str("postprocess session configuration is invalid")
            }
            Self::PhysicsConfiguration(error) => {
                write!(
                    formatter,
                    "postprocess physics configuration is invalid: {error}"
                )
            }
            Self::StrictHfGravityMissing => {
                formatter.write_str("strict HF gravity coefficients are missing")
            }
            Self::StrictHfGravityInvalid => {
                formatter.write_str("strict HF gravity coefficients are unusable")
            }
            Self::StrictHfEmbeddedGravityUnavailable => {
                formatter.write_str("embedded strict HF gravity coefficients are unavailable")
            }
            Self::StrictHfJb2008Unavailable => {
                formatter.write_str("strict HF JB2008 assets are unavailable")
            }
        }
    }
}

impl std::error::Error for PostprocessSessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PhysicsConfiguration(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(any(test, feature = "bench-internal"))]
#[derive(Clone, Copy, Debug)]
pub(super) struct ResolvedPostprocessRuntimeSettings {
    pub(super) num_dists: usize,
}

#[cfg(any(test, feature = "bench-internal"))]
impl Default for ResolvedPostprocessRuntimeSettings {
    fn default() -> Self {
        Self {
            num_dists: DEFAULT_NUM_DISTS.clamp(1, MAX_DUST_COMPONENTS),
        }
    }
}

#[cfg(any(test, feature = "bench-internal"))]
impl ResolvedPostprocessRuntimeSettings {
    pub(super) fn from_postprocess_config(post: &PostprocessConfig) -> Self {
        Self {
            num_dists: post.gmm_components.clamp(1, MAX_DUST_COMPONENTS),
        }
    }
}

fn validate_postprocess_config(post: &PostprocessConfig) -> Result<(), PostprocessSessionError> {
    if !(1..=MAX_DUST_COMPONENTS).contains(&post.gmm_components) {
        return Err(PostprocessSessionError::PostprocessConfiguration);
    }
    for (_, value) in [
        ("canister_am", post.canister_am),
        ("canister_cd", post.canister_cd),
        ("dust_phase_tof_s", post.dust_phase_tof_s),
        ("max_physical_dv_kms", post.max_physical_dv_kms),
        ("fix_ls_tol", post.fix_ls_tol),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(PostprocessSessionError::PostprocessConfiguration);
        }
    }
    for (_, value) in [
        ("canister_cr", post.canister_cr),
        ("fix_ls_skip_tol", post.fix_ls_skip_tol),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(PostprocessSessionError::PostprocessConfiguration);
        }
    }
    Ok(())
}

#[cfg(any(test, feature = "bench-internal"))]
const POSTPROCESS_STATUS_OK: i32 = 0;
#[cfg(any(test, feature = "bench-internal"))]
const POSTPROCESS_STATUS_MISSING_CANDIDATE: i32 = 1;
#[cfg(any(test, feature = "bench-internal"))]
const POSTPROCESS_STATUS_INVALID_CANDIDATE: i32 = 2;
#[cfg(any(test, feature = "bench-internal"))]
const POSTPROCESS_STATUS_NO_CORRECTION: i32 = 3;
#[cfg(any(test, feature = "bench-internal"))]
const POSTPROCESS_STATUS_STALE_INTERCEPT_INPUT: i32 = 4;
#[cfg(any(test, feature = "bench-internal"))]
const POSTPROCESS_STATUS_HF_REFINE_FAILED: i32 = 5;

/// Sealed Part A high-fidelity force identity for native callers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrictHfForceAuthority {
    pub gravity_order: usize,
    pub force_flags: i32,
    pub atmosphere_model: i32,
    pub integrator_method: &'static str,
    pub dt_max_s: f64,
    pub tolerance: f64,
    pub transfer_body_force: crate::types::BodyForceConfig,
}

impl StrictHfForceAuthority {
    pub const PART_A: Self = Self {
        gravity_order: 5,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        // Must equal `PartAHybridControls::atmosphere_model` in
        // `nd_config::part_a_science`. `nd_pipeline::hybrid` fails CLOSED on any
        // disagreement, in two places: `validate_runtime_constants` (the
        // `force_authority` arm) and `validate_part_a_force_controls`. Moving one
        // without the other does not degrade the run, it refuses to start it.
        //
        // 8 = Part A v3 JB2008 persistence authority with model 7's fitted
        // density kernel. Codes 4--7 retain the historical compiled SET
        // provider; code 8 alone selects the sealed model-conditioned v3
        // persistence scenario. Frame/time and fitted-kernel behavior stay
        // identical to model 7.
        //
        // The fit covers Texo in [500, 2600] K. OUTSIDE that window the fitted
        // accessor falls back to walking the real plan, so the domain is a
        // speed boundary and not a validity one -- model 7 is defined
        // everywhere model 4 is. Below 105 km neither fixed plan runs and
        // model 7 is model 6 bit for bit.
        //
        // Bounded by `v7_broad_grid_density_error_stays_within_rescoped_bound`
        // (jb_rs) at 1.0e-4 over an 1,800-row lattice, NOT by the strict-HF
        // 1.0 m accuracy gates -- those difference an arc against itself at a
        // tighter tolerance, so the quadrature bias cancels and they cannot see
        // this constant at all. That gate's non-vacuity floor is STRUCTURAL,
        // and has to be: the fit residual (worst scalar 7.434e-6) sits an order
        // of magnitude below the quadrature bias model 7 inherits, so model 7's
        // error equals model 6's to four significant digits and no assertion on
        // the error's MAGNITUDE could tell the two apart. The floor counts rows
        // where the profiles disagree instead.
        //
        // Models 4, 5 and 6 are unchanged and all still selectable; model 6 is
        // the profile model 7 was measured against: -10.3% of strict-HF arc
        // wall, with RHS evaluations flat (6,752 -> 6,742, -0.15%), so the
        // saving is per-evaluation cost rather than a shorter arc. Method,
        // three-run spread and the reason the MIN rather than the mean is
        // quoted are recorded on the science seal in
        // `nd_config/tests/part_a_science.rs`.
        //
        // Carried over from the 5 -> 6 move and still true: nd_pipeline/src/
        // hybrid.rs stamps receipts with the V1 model-name/transform strings
        // for every profile, so the human-readable label inside the preimage is
        // stale. Discrimination is carried by this INTEGER, which is hashed;
        // receipts stay distinct. Adding per-profile name constants changes
        // future receipt hashes and is deferred post-campaign.
        atmosphere_model: 8,
        // Must equal `PartAHybridControls::integrator_method`. This crate
        // cannot see `nd_config`, so the token is mirrored rather than read;
        // `nd_pipeline::hybrid::validate_runtime_constants` compares the
        // snapshot's method against compiled science and refuses to start on
        // disagreement, and production itself never reads THIS copy --
        // `part_a_physics_from_controls` flows `controls.integrator_method`
        // straight through.
        //
        // The mirror exists so the strict-HF test helpers fly what the
        // campaign flies. It used to be a bare `"vern9"` literal inside
        // `strict_physics()` below, under a comment forbidding exactly that,
        // and it survived the Vern9 -> Vern7 swap unnoticed because nothing
        // compared it to anything.
        integrator_method: "vern7",
        dt_max_s: 300.0,
        tolerance: 1.0e-8,
        transfer_body_force: crate::types::BodyForceConfig::high_fidelity(
            crate::types::BodyRole::TransferVehicle,
            0.01,
            2.2,
            1.3,
        ),
    };
}

/// Immutable identity of strict-HF gravity source bytes and packed semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StrictHfGravityIdentity {
    source_sha256: [u8; 32],
    packed_semantic_sha256: [u8; 32],
}

impl StrictHfGravityIdentity {
    /// SHA-256 of exact embedded DIR-R6 source bytes.
    #[must_use]
    pub const fn source_sha256(self) -> [u8; 32] {
        self.source_sha256
    }

    /// SHA-256 of validated packed coefficient semantics.
    #[must_use]
    pub const fn packed_semantic_sha256(self) -> [u8; 32] {
        self.packed_semantic_sha256
    }
}

/// Typed readiness failure for a strict native HF mass-solver context.
#[derive(Clone, Debug, PartialEq)]
pub enum StrictHfContextStatus {
    HighFidelityDisabled,
    StrictCorrectionRequired,
    Configuration(PhysicsConfigError),
    MissingGravityCoefficients,
    InvalidGravityCoefficients,
    Ephemeris(EphemerisCoverageError),
}

impl fmt::Display for StrictHfContextStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HighFidelityDisabled => formatter.write_str("high-fidelity context is disabled"),
            Self::StrictCorrectionRequired => {
                formatter.write_str("strict HF transfer correction is required")
            }
            Self::Configuration(error) => {
                write!(
                    formatter,
                    "strict HF physics configuration is invalid: {error}"
                )
            }
            Self::MissingGravityCoefficients => {
                formatter.write_str("strict HF gravity coefficients are missing")
            }
            Self::InvalidGravityCoefficients => {
                formatter.write_str("strict HF gravity coefficients are unusable")
            }
            Self::Ephemeris(error) => write!(formatter, "strict HF ephemeris unavailable: {error}"),
        }
    }
}

impl std::error::Error for StrictHfContextStatus {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Configuration(error) => Some(error),
            Self::Ephemeris(error) => Some(error),
            _ => None,
        }
    }
}

const INPUT_INTERCEPT_JD_TOL: f64 = 1.0e-9;
const EMBEDDED_DIR_R6_D15: &[u8] =
    include_bytes!("../../data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

fn embedded_source_sha256() -> [u8; 32] {
    static SOURCE_SHA256: OnceLock<[u8; 32]> = OnceLock::new();
    *SOURCE_SHA256.get_or_init(|| Sha256::digest(EMBEDDED_DIR_R6_D15).into())
}

fn build_strict_hf_gravity_pack(
    order: usize,
) -> Result<
    (
        Arc<satpy_core::PackedGravityCoeffs>,
        StrictHfGravityIdentity,
    ),
    PostprocessSessionError,
> {
    let packed = lightyear_odeint_rs::packed_constants_from_bytes(EMBEDDED_DIR_R6_D15, order)
        .map_err(|_| PostprocessSessionError::StrictHfEmbeddedGravityUnavailable)?;
    let packed_semantic_sha256 = packed
        .authority_sha256()
        .map_err(|_| PostprocessSessionError::StrictHfGravityInvalid)?;
    let identity = StrictHfGravityIdentity {
        source_sha256: embedded_source_sha256(),
        packed_semantic_sha256,
    };
    Ok((packed, identity))
}

struct CanonicalStrictHfGravity {
    packed: Arc<satpy_core::PackedGravityCoeffs>,
    identity: StrictHfGravityIdentity,
}

fn success_only_cached<'cache, T, E>(
    cache: &'cache OnceLock<T>,
    cold_load: &Mutex<()>,
    initialize: impl FnOnce() -> Result<T, E>,
) -> Result<&'cache T, E> {
    if let Some(value) = cache.get() {
        return Ok(value);
    }

    let _cold_load_guard = cold_load
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    if let Some(value) = cache.get() {
        return Ok(value);
    }

    let candidate = initialize()?;
    Ok(cache.get_or_init(|| candidate))
}

fn canonical_strict_hf_gravity(
) -> Result<&'static CanonicalStrictHfGravity, PostprocessSessionError> {
    static GRAVITY: OnceLock<CanonicalStrictHfGravity> = OnceLock::new();
    static GRAVITY_LOAD: Mutex<()> = Mutex::new(());
    success_only_cached(&GRAVITY, &GRAVITY_LOAD, || {
        build_strict_hf_gravity_pack(StrictHfForceAuthority::PART_A.gravity_order)
            .map(|(packed, identity)| CanonicalStrictHfGravity { packed, identity })
    })
}

/// Canonical strict-HF gravity identity for downstream authority revalidation.
///
/// # Errors
///
/// Returns a typed session error if embedded bytes cannot produce the sealed
/// Part A pack.
pub fn canonical_strict_hf_gravity_identity(
) -> Result<StrictHfGravityIdentity, PostprocessSessionError> {
    Ok(canonical_strict_hf_gravity()?.identity)
}

#[inline]
fn finite_close(lhs: f64, rhs: f64, abs_tol: f64) -> bool {
    lhs.is_finite() && rhs.is_finite() && (lhs - rhs).abs() <= abs_tol
}

#[inline]
fn candidate_matches_requested_intercept(
    candidate: &CompactTransferCandidate,
    intercept_jd: f64,
) -> bool {
    finite_close(
        candidate.solver_intercept_jd,
        intercept_jd,
        INPUT_INTERCEPT_JD_TOL,
    )
}

#[derive(Clone)]
pub(super) struct GlobalCoeffs {
    pub(super) packed: Option<Arc<satpy_core::PackedGravityCoeffs>>,
    pub(super) missing: bool,
}

pub(super) fn load_global_coeffs() -> GlobalCoeffs {
    let packed = lightyear_odeint_rs::get_global_coeffs_packed();
    GlobalCoeffs {
        missing: packed.is_none(),
        packed,
    }
}

fn validate_strict_hf_static_assets(
    physics_config: &PhysicsConfig,
    coeffs: &GlobalCoeffs,
    identity: Option<StrictHfGravityIdentity>,
) -> Result<(), PostprocessSessionError> {
    let Some(packed_coeffs) = coeffs.packed.as_ref() else {
        return Err(PostprocessSessionError::StrictHfGravityMissing);
    };
    if coeffs.missing || physics_config.sph_order > packed_coeffs.max_order() {
        return Err(PostprocessSessionError::StrictHfGravityInvalid);
    }
    let Some(identity) = identity else {
        return Err(PostprocessSessionError::StrictHfGravityInvalid);
    };
    if identity.source_sha256 != embedded_source_sha256()
        || packed_coeffs
            .authority_sha256()
            .map_err(|_| PostprocessSessionError::StrictHfGravityInvalid)?
            != identity.packed_semantic_sha256
        || identity != canonical_strict_hf_gravity_identity()?
    {
        return Err(PostprocessSessionError::StrictHfGravityInvalid);
    }

    let jb2008_drag =
        lightyear_odeint_rs::rhs::atm_model_uses_jb2008_drivers(physics_config.atm_model)
            && (physics_config.force_flags & ForceFlags::DRAG) != 0;
    if jb2008_drag {
        lightyear_odeint_rs::rhs::jb2008_driver_authority(physics_config.atm_model)
            .ok_or(PostprocessSessionError::StrictHfJb2008Unavailable)?
            .load()
            .map_err(|_| PostprocessSessionError::StrictHfJb2008Unavailable)?;
    }
    Ok(())
}

struct ResolvedSessionCoeffs {
    coeffs: GlobalCoeffs,
    strict_hf_gravity_identity: Option<StrictHfGravityIdentity>,
}

fn resolve_session_coeffs(
    physics_config: &PhysicsConfig,
) -> Result<ResolvedSessionCoeffs, PostprocessSessionError> {
    let strict_hf =
        physics_config.use_high_fidelity && physics_config.require_hf_transfer_correction;
    if strict_hf {
        if physics_config.sph_order != StrictHfForceAuthority::PART_A.gravity_order {
            return Err(PostprocessSessionError::StrictHfGravityInvalid);
        }
        let gravity = canonical_strict_hf_gravity()?;
        return Ok(ResolvedSessionCoeffs {
            coeffs: GlobalCoeffs {
                packed: Some(Arc::clone(&gravity.packed)),
                missing: false,
            },
            strict_hf_gravity_identity: Some(gravity.identity),
        });
    }

    Ok(ResolvedSessionCoeffs {
        coeffs: load_global_coeffs(),
        strict_hf_gravity_identity: None,
    })
}

pub(super) const fn default_postprocess_config() -> PostprocessConfig {
    PostprocessConfig {
        fix_ls_max_nfev: 100,
        fix_ls_tol: 1e-5,
        fix_ls_skip_tol: 1.0,
        dust_intercept_tol_km: 0.01,
        dust_radial_samples: 24,
        dust_angular_samples: 100,
        gmm_components: DEFAULT_NUM_DISTS,
        max_physical_dv_kms: 7.5,
        min_practical_dust_mass_kg: 0.01,
        mf_seed_bound_kms: 0.1,
        // <= 0.0 uses the solver's default HF bound policy.
        hf_refine_bound_kms: 0.0,
        mf_seed_reg_weight: 1e-3,
        hf_refine_reg_weight: 1e-3,
        mf_seed_max_bound_expansions: 7,
        hf_refine_max_bound_expansions: 7,
        hybrid_mf_seed_hf_refine: false,
        dust_phase_tof_s: 7200.0,
        canister_tof_fraction: 0.0,
        canister_am: 0.01,
        canister_cd: 2.2,
        canister_cr: 1.3,
    }
}

#[derive(Clone)]
pub struct TransferPostprocessSessionCore {
    physics_config: PhysicsConfig,
    postprocess_config: PostprocessConfig,
    coeffs: GlobalCoeffs,
    strict_hf_gravity_identity: Option<StrictHfGravityIdentity>,
    hf_missing_coeffs_strict: bool,
    #[cfg(any(test, feature = "bench-internal"))]
    runtime_settings: ResolvedPostprocessRuntimeSettings,
}

/// Closed inputs for one observed release-control replay.
///
/// This exists only with the qualification feature because the canonical
/// production entry has no trace owner or observer surface.
#[cfg(feature = "solver-qualification")]
pub struct QualificationReleaseControlRequest<'a> {
    pub candidate: Option<&'a CompactTransferCandidate>,
    pub intercept_jd: f64,
    pub conjunction_jd: f64,
    pub target_am: Option<f64>,
    pub target_drag_coefficient: Option<f64>,
    pub target_reflectivity_coefficient: Option<f64>,
    pub fraction: f64,
}

/// Closed inputs for one observed UKF distribution replay.
#[cfg(feature = "solver-qualification")]
pub struct QualificationDistributionRequest<'a> {
    pub control: &'a PostprocessControl,
    pub dust_ctx: &'a PlanContext,
    pub fraction: Option<f64>,
    pub split_alpha: Option<f64>,
    pub split_axis: Option<[f64; 6]>,
    pub release_covariance: Option<[[f64; 6]; 6]>,
    pub release_distribution: Option<AuthoritativeReleaseDistribution>,
}

impl TransferPostprocessSessionCore {
    #[must_use]
    pub const fn physics_config(&self) -> &PhysicsConfig {
        &self.physics_config
    }

    #[must_use]
    pub const fn postprocess_config(&self) -> &PostprocessConfig {
        &self.postprocess_config
    }

    /// Build the sealed no-override session for natural Part A refinement.
    ///
    /// # Errors
    ///
    /// Returns an error when compiled strict-HF assets cannot establish the
    /// reviewed Part A runtime.
    pub fn try_new_natural_part_a_refinement() -> Result<Self, PostprocessSessionError> {
        let authority = StrictHfForceAuthority::PART_A;
        let physics = PhysicsConfig {
            use_high_fidelity: true,
            require_hf_transfer_correction: true,
            sph_order: authority.gravity_order,
            force_flags: authority.force_flags,
            atm_model: authority.atmosphere_model,
            method: authority.integrator_method.to_owned(),
            dt_max: authority.dt_max_s,
            tolerance: authority.tolerance,
            transfer_am_ratio: authority.transfer_body_force.am_ratio,
            transfer_cd: authority.transfer_body_force.cd,
            transfer_cr: authority.transfer_body_force.cr,
            ..PhysicsConfig::default()
        };
        Self::try_new(Some(physics), None)
    }

    /// Build one validated postprocess session.
    ///
    /// # Errors
    ///
    /// Returns an error when configuration, strict-HF assets, or required JB
    /// drivers cannot establish one valid runtime.
    pub fn try_new(
        physics_config: Option<PhysicsConfig>,
        postprocess_config: Option<PostprocessConfig>,
    ) -> Result<Self, PostprocessSessionError> {
        let physics_config = physics_config.unwrap_or_default();
        physics_config
            .integrator_method()
            .map_err(PostprocessSessionError::PhysicsConfiguration)?;
        let resolved = resolve_session_coeffs(&physics_config)?;
        Self::try_new_with_resolved_coeffs(
            Some(physics_config),
            postprocess_config,
            resolved.coeffs,
            resolved.strict_hf_gravity_identity,
        )
    }

    #[cfg(test)]
    fn try_new_with_coeffs(
        physics_config: Option<PhysicsConfig>,
        postprocess_config: Option<PostprocessConfig>,
        coeffs: GlobalCoeffs,
    ) -> Result<Self, PostprocessSessionError> {
        Self::try_new_with_resolved_coeffs(physics_config, postprocess_config, coeffs, None)
    }

    fn try_new_with_resolved_coeffs(
        physics_config: Option<PhysicsConfig>,
        postprocess_config: Option<PostprocessConfig>,
        coeffs: GlobalCoeffs,
        strict_hf_gravity_identity: Option<StrictHfGravityIdentity>,
    ) -> Result<Self, PostprocessSessionError> {
        let physics_config = physics_config.unwrap_or_default();
        physics_config
            .integrator_method()
            .map_err(PostprocessSessionError::PhysicsConfiguration)?;
        let postprocess_config = postprocess_config.unwrap_or_else(default_postprocess_config);
        validate_postprocess_config(&postprocess_config)?;
        #[cfg(any(test, feature = "bench-internal"))]
        let runtime_settings =
            ResolvedPostprocessRuntimeSettings::from_postprocess_config(&postprocess_config);
        let strict_hf =
            physics_config.use_high_fidelity && physics_config.require_hf_transfer_correction;
        if strict_hf {
            validate_strict_hf_static_assets(&physics_config, &coeffs, strict_hf_gravity_identity)?;
        }
        let hf_missing_coeffs_strict = strict_hf && coeffs.missing;
        Ok(Self {
            physics_config,
            postprocess_config,
            coeffs,
            strict_hf_gravity_identity,
            hf_missing_coeffs_strict,
            #[cfg(any(test, feature = "bench-internal"))]
            runtime_settings,
        })
    }

    /// Strict-HF gravity identity retained by this session, when enabled.
    #[must_use]
    pub const fn strict_hf_gravity_identity(&self) -> Option<StrictHfGravityIdentity> {
        self.strict_hf_gravity_identity
    }

    /// Build one strict native HF mass-solver context for an exact event arc.
    ///
    /// Coefficients belong to this postprocess session; ephemeris is resolved
    /// against the requested arc before returning.  No caller may downgrade
    /// this context to MF when an asset or arc check fails.
    ///
    /// # Errors
    ///
    /// Returns a typed failure when strict-HF is disabled, its assets are
    /// unavailable, or the requested ephemeris arc is uncovered.
    pub fn strict_hf_context_for_arc(
        &self,
        epoch_jd: f64,
        arc_end_jd: f64,
    ) -> Result<HfContext, StrictHfContextStatus> {
        if !self.physics_config.use_high_fidelity {
            return Err(StrictHfContextStatus::HighFidelityDisabled);
        }
        if !self.physics_config.require_hf_transfer_correction {
            return Err(StrictHfContextStatus::StrictCorrectionRequired);
        }

        let packed_coeffs = self
            .coeffs
            .packed
            .clone()
            .ok_or(StrictHfContextStatus::MissingGravityCoefficients)?;
        if self.hf_missing_coeffs_strict
            || self.physics_config.sph_order > packed_coeffs.max_order()
        {
            return Err(StrictHfContextStatus::InvalidGravityCoefficients);
        }
        let Some(identity) = self.strict_hf_gravity_identity else {
            return Err(StrictHfContextStatus::InvalidGravityCoefficients);
        };
        let canonical_identity = canonical_strict_hf_gravity_identity()
            .map_err(|_| StrictHfContextStatus::InvalidGravityCoefficients)?;
        if identity != canonical_identity {
            return Err(StrictHfContextStatus::InvalidGravityCoefficients);
        }

        let force_config = super::build_force_config(
            &self.physics_config,
            self.physics_config.am_ratio,
            self.physics_config.cd,
            self.physics_config.cr,
        )
        .map_err(StrictHfContextStatus::Configuration)?
        .with_ephemeris_for_arc(epoch_jd, arc_end_jd)
        .map_err(StrictHfContextStatus::Ephemeris)?;

        Ok(HfContext {
            use_high_fidelity: true,
            epoch_jd,
            force_config: Some(Arc::new(force_config)),
            packed_coeffs: Some(packed_coeffs),
            hf_validate_only: self.postprocess_config.hybrid_mf_seed_hf_refine,
            hf_strict: true,
        })
    }

    #[cfg(any(test, feature = "bench-internal"))]
    pub(super) fn correct_one(
        &self,
        candidate: Option<&CompactTransferCandidate>,
        _primary_at_intercept: &[f64; 6],
        _secondary_at_intercept: &[f64; 6],
        intercept_jd: f64,
        conjunction_jd: f64,
        target_am: Option<f64>,
        target_drag_coefficient: Option<f64>,
        target_reflectivity_coefficient: Option<f64>,
        scratch: Option<&mut TransferPostprocessScratch>,
    ) -> Result<Option<([f64; 6], f64)>, PostprocessDistributionStatus> {
        let Some(candidate) = candidate else {
            return Ok(None);
        };
        if !candidate.valid {
            return Ok(None);
        }
        if !compact_candidate_is_postprocess_coherent(candidate, conjunction_jd) {
            return Ok(None);
        }
        if !candidate_matches_requested_intercept(candidate, intercept_jd) {
            return Ok(None);
        }
        if self.hf_missing_coeffs_strict {
            return Err(PostprocessDistributionStatus::StrictHfAssetsUnavailable);
        }

        let target_intercept = &candidate.target_intercept_state;
        let summary_inputs = SummaryPlanInputs {
            valid: true,
            release_state: candidate.transfer_burn_pre_state,
            transfer_dv: candidate.transfer_dv,
            tof_jd_start: candidate.tof_jd_start,
            min_radius_km: candidate.replay_policy.min_perigee,
        };
        compute_corrected_dust_state_summary_with_runtime(
            &summary_inputs,
            target_intercept,
            candidate.solver_intercept_jd,
            conjunction_jd,
            &self.physics_config,
            &self.postprocess_config,
            &self.coeffs,
            self.runtime_settings,
            target_am,
            target_drag_coefficient,
            target_reflectivity_coefficient,
            scratch,
        )
        .map(|summary| summary.map(|summary| (summary.dust_mean, summary.correction_dv_norm)))
    }

    /// Recompute one authoritative control for an explicit v2 fraction.
    /// The candidate is replayed once upstream; this path only repartitions
    /// its exact L->I interval and never reuses a control from another row.
    ///
    /// # Errors
    ///
    /// Returns a typed status when the candidate, fraction, propagation
    /// authority, or release control is invalid.
    pub fn release_control_one_at_fraction(
        &self,
        candidate: Option<&CompactTransferCandidate>,
        intercept_jd: f64,
        conjunction_jd: f64,
        target_am: Option<f64>,
        target_drag_coefficient: Option<f64>,
        target_reflectivity_coefficient: Option<f64>,
        fraction: f64,
    ) -> Result<(PostprocessControl, PlanContext), PostprocessControlStatus> {
        let candidate = candidate.ok_or(PostprocessControlStatus::InvalidTimeline)?;
        if !candidate.valid
            || !compact_candidate_is_postprocess_coherent(candidate, conjunction_jd)
            || !candidate_matches_requested_intercept(candidate, intercept_jd)
            || self.hf_missing_coeffs_strict
        {
            return Err(PostprocessControlStatus::InvalidTimeline);
        }
        let inputs = SummaryPlanInputs {
            valid: true,
            release_state: candidate.transfer_burn_pre_state,
            transfer_dv: candidate.transfer_dv,
            tof_jd_start: candidate.tof_jd_start,
            min_radius_km: candidate.replay_policy.min_perigee,
        };
        let target_propagation_authority =
            TargetPropagationAuthority::try_from(candidate.replay_policy.target_propagation_mode)
                .map_err(|_| PostprocessControlStatus::InvalidTimeline)?;
        build_release_control_at_fraction(
            &inputs,
            &candidate.target_intercept_state,
            candidate.solver_intercept_jd,
            conjunction_jd,
            &self.physics_config,
            &self.postprocess_config,
            &self.coeffs,
            target_am,
            target_drag_coefficient,
            target_reflectivity_coefficient,
            fraction,
            target_propagation_authority,
            // No production reader consumes `conjunction_separation_km` -- not
            // this MF route through `nd_pipeline/src/physics/release_control.rs`,
            // nor the strict-HF route through `nd_pipeline/src/native_hybrid.rs`
            // (both in `nd_pipeline`, not this crate). Computing it costs
            // two full intercept->conjunction propagations over a ~2.25 day
            // leg, profiled at ~21% of ALL Part A campaign work, and it is
            // then discarded. Both fidelities skip it here.
            ConjunctionDiagnostic::Skip,
        )
        .map(|(control, dust_ctx, _)| (control, dust_ctx))
    }

    /// Recompute one strict diagnostic control and record every actual scalar
    /// release leg in caller-owned bounded storage.
    ///
    /// # Errors
    ///
    /// Returns a typed status when request inputs cannot describe one valid
    /// strict replay or an observed scalar propagation fails.
    #[cfg(feature = "solver-qualification")]
    pub fn release_control_one_at_fraction_observed(
        &self,
        request: &QualificationReleaseControlRequest<'_>,
        trace: &mut QualificationLegTrace,
    ) -> Result<(PostprocessControl, PlanContext), PostprocessControlStatus> {
        let candidate = request
            .candidate
            .ok_or(PostprocessControlStatus::InvalidTimeline)?;
        if !candidate.valid
            || !compact_candidate_is_postprocess_coherent(candidate, request.conjunction_jd)
            || !candidate_matches_requested_intercept(candidate, request.intercept_jd)
            || self.hf_missing_coeffs_strict
        {
            return Err(PostprocessControlStatus::InvalidTimeline);
        }
        let inputs = SummaryPlanInputs {
            valid: true,
            release_state: candidate.transfer_burn_pre_state,
            transfer_dv: candidate.transfer_dv,
            tof_jd_start: candidate.tof_jd_start,
            min_radius_km: candidate.replay_policy.min_perigee,
        };
        let _target_propagation_authority =
            TargetPropagationAuthority::try_from(candidate.replay_policy.target_propagation_mode)
                .map_err(|_| PostprocessControlStatus::InvalidTimeline)?;
        let (control, dust_ctx) = build_release_control_at_fraction_observed(
            &ObservedReleaseControlCoreRequest {
                plan: &inputs,
                target_intercept_state: &candidate.target_intercept_state,
                intercept_jd: candidate.solver_intercept_jd,
                conf: &self.physics_config,
                post: &self.postprocess_config,
                coeffs: &self.coeffs,
                fraction: request.fraction,
            },
            trace,
        )?;
        let target_am = request.target_am.unwrap_or(0.0);
        let target_drag_coefficient = request
            .target_drag_coefficient
            .unwrap_or(self.physics_config.cd);
        let target_reflectivity_coefficient = request
            .target_reflectivity_coefficient
            .unwrap_or(self.physics_config.cr);
        build_ctx(
            candidate.solver_intercept_jd,
            &self.physics_config,
            &self.coeffs,
            target_am,
            target_drag_coefficient,
            target_reflectivity_coefficient,
        )
        .map_err(PostprocessControlStatus::Configuration)?;
        Ok((control, dust_ctx))
    }

    /// Materialize a corrected distribution from an ALREADY-BUILT release
    /// control and its dust context.
    ///
    /// `full_corrected_distribution_one` rebuilds the control internally. The
    /// strict-HF descriptor path has to resolve R first anyway, to seal the
    /// shared mixture it passes back in as `release_distribution`, so going
    /// through that entry point ran `build_physical_release_control` -- an LM
    /// loop whose every iteration is an HF R->I propagation -- twice per row
    /// with identical arguments. Verified over all 268 candidate rows of a
    /// fixture event: the two results agree bit-for-bit on all six stamped
    /// states, both JDs, `release_control_dv` and its norm,
    /// `canister_tof_fraction`, `canister_coast_s` and `dust_free_flight_s`.
    ///
    /// # Errors
    ///
    /// Returns a typed status when supplied controls, fraction, distribution,
    /// or propagation inputs are invalid.
    pub fn corrected_distribution_from_control(
        &self,
        control: &PostprocessControl,
        dust_ctx: &PlanContext,
        fraction: Option<f64>,
        split_alpha: Option<f64>,
        split_axis: Option<[f64; 6]>,
        release_covariance: Option<[[f64; 6]; 6]>,
        release_distribution: Option<AuthoritativeReleaseDistribution>,
    ) -> Result<PostprocessDustDistribution, PostprocessDistributionStatus> {
        let mut postprocess_config = self.postprocess_config.clone();
        if let Some(fraction) = fraction {
            if !fraction.is_finite() || !(0.0..1.0).contains(&fraction) {
                return Err(PostprocessDistributionStatus::InvalidFraction);
            }
            postprocess_config.canister_tof_fraction = fraction;
        }
        materialize_corrected_dust_distribution(
            control,
            dust_ctx,
            &self.physics_config,
            &postprocess_config,
            split_alpha,
            split_axis,
            release_covariance.as_ref(),
            release_distribution,
            &mut UnobservedPostprocessLeg,
        )
    }

    /// Materialize a strict diagnostic distribution while recording every
    /// actual UKF scalar propagation in the supplied bounded trace.
    ///
    /// # Errors
    ///
    /// Returns a typed status when the observed distribution inputs are
    /// invalid or an observed UKF propagation fails.
    #[cfg(feature = "solver-qualification")]
    pub fn corrected_distribution_from_control_observed(
        &self,
        request: QualificationDistributionRequest<'_>,
        trace: &mut QualificationLegTrace,
    ) -> Result<PostprocessDustDistribution, PostprocessDistributionStatus> {
        let QualificationDistributionRequest {
            control,
            dust_ctx,
            fraction,
            split_alpha,
            split_axis,
            release_covariance,
            release_distribution,
        } = request;
        let mut postprocess_config = self.postprocess_config.clone();
        if let Some(fraction) = fraction {
            if !fraction.is_finite() || !(0.0..1.0).contains(&fraction) {
                return Err(PostprocessDistributionStatus::InvalidFraction);
            }
            postprocess_config.canister_tof_fraction = fraction;
        }
        materialize_corrected_dust_distribution_observed(
            ObservedMaterializationRequest {
                control,
                ctx_dust: dust_ctx,
                conf: &self.physics_config,
                post: &postprocess_config,
                split_alpha,
                split_axis,
                release_covariance: release_covariance.as_ref(),
                release_distribution,
            },
            trace,
        )
    }

    /// Materialize one native corrected distribution with its failure status.
    /// Strict callers must not convert an error into an MF fallback.
    ///
    /// # Errors
    ///
    /// Returns a typed status when the candidate, fraction, release control,
    /// distribution, or propagation cannot be accepted.
    pub fn full_corrected_distribution_one(
        &self,
        candidate: Option<&CompactTransferCandidate>,
        conjunction_jd: f64,
        fraction: Option<f64>,
        split_alpha: Option<f64>,
        split_axis: Option<[f64; 6]>,
        release_covariance: Option<[[f64; 6]; 6]>,
        release_distribution: Option<AuthoritativeReleaseDistribution>,
    ) -> Result<PostprocessDustDistribution, PostprocessDistributionStatus> {
        let candidate = candidate.ok_or(PostprocessDistributionStatus::MissingCandidate)?;
        if !candidate.valid {
            return Err(PostprocessDistributionStatus::InvalidCandidate);
        }
        if !compact_candidate_is_postprocess_coherent(candidate, conjunction_jd) {
            return Err(PostprocessDistributionStatus::InvalidTimeline);
        }
        if self.hf_missing_coeffs_strict {
            return Err(PostprocessDistributionStatus::StrictHfAssetsUnavailable);
        }
        let inputs = SummaryPlanInputs {
            valid: true,
            release_state: candidate.transfer_burn_pre_state,
            transfer_dv: candidate.transfer_dv,
            tof_jd_start: candidate.tof_jd_start,
            min_radius_km: candidate.replay_policy.min_perigee,
        };
        let mut postprocess_config = self.postprocess_config.clone();
        if let Some(fraction) = fraction {
            if !fraction.is_finite() || !(0.0..1.0).contains(&fraction) {
                return Err(PostprocessDistributionStatus::InvalidFraction);
            }
            postprocess_config.canister_tof_fraction = fraction;
        }
        compute_corrected_dust_state(CorrectedDustStateRequest {
            plan: &inputs,
            target_intercept_state: &candidate.target_intercept_state,
            intercept_jd: candidate.solver_intercept_jd,
            conjunction_jd,
            conf: &self.physics_config,
            post: &postprocess_config,
            coeffs: &self.coeffs,
            split_alpha,
            split_axis,
            release_covariance: release_covariance.as_ref(),
            release_distribution,
        })
    }

    #[cfg(any(test, feature = "bench-internal"))]
    #[inline]
    const fn hf_refine_failure_mode(&self) -> bool {
        self.physics_config.use_high_fidelity && self.postprocess_config.hybrid_mf_seed_hf_refine
    }
}

#[derive(Default)]
#[cfg(any(test, feature = "bench-internal"))]
pub(super) struct TransferPostprocessScratch {
    #[cfg(any(test, feature = "bench-internal"))]
    pub(super) last_batch_len: usize,
    #[cfg(any(test, feature = "bench-internal"))]
    pub(super) weights: SmallVec<[f64; MAX_DUST_COMPONENTS]>,
    pub(super) comp_means: SmallVec<[[f64; 6]; MAX_DUST_COMPONENTS]>,
    #[cfg(any(test, feature = "bench-internal"))]
    pub(super) corrected_component_means: SmallVec<[[f64; 6]; MAX_DUST_COMPONENTS]>,
    pub(super) sigma_states: Vec<f64>,
    pub(super) sigma_equinoc: Vec<f64>,
    pub(super) sigma_propagated: Vec<f64>,
    pub(super) sigma_tofs: Vec<f64>,
    pub(super) component_sigma_offsets: SmallVec<[usize; MAX_DUST_COMPONENTS]>,
}

/// Per-row target physical properties for one compact batch side.
#[cfg(any(test, feature = "bench-internal"))]
#[derive(Clone, Copy, Debug, Default)]
pub struct CompactBatchTargetPhysics<'a> {
    pub am_ratio: Option<&'a [f64]>,
    pub drag_coefficient: Option<&'a [f64]>,
    pub reflectivity_coefficient: Option<&'a [f64]>,
}

/// Immutable compact-batch inputs.
#[cfg(any(test, feature = "bench-internal"))]
pub struct CompactBatchPostprocessInputs<'a> {
    pub candidates: &'a [Option<CompactTransferCandidate>],
    pub primary_states: &'a [f64],
    pub secondary_states: &'a [f64],
    pub intercept_jds: &'a [f64],
    pub conjunction_jds: &'a [f64],
    pub physics_config: Option<PhysicsConfig>,
    pub postprocess_config: Option<PostprocessConfig>,
    pub primary_target: CompactBatchTargetPhysics<'a>,
    pub secondary_target: CompactBatchTargetPhysics<'a>,
}

/// Mutable compact-batch output buffers.
#[cfg(any(test, feature = "bench-internal"))]
pub struct CompactBatchPostprocessOutputs<'a> {
    pub corrected_states: &'a mut [f64],
    pub correction_dvs: &'a mut [f64],
    pub status_codes: &'a mut [i32],
}

/// Failure from compact postprocess batch validation or correction.
///
/// Session and correction failures preserve their exact typed cause. Shape,
/// arithmetic, and staging-allocation failures are detected before any caller
/// output buffer is committed.
#[cfg(any(test, feature = "bench-internal"))]
#[derive(Clone, Debug, PartialEq)]
pub enum CompactBatchPostprocessError {
    /// Session construction failed with its exact typed status.
    Session(PostprocessSessionError),
    /// Parallel batch input or output slices do not have the required shape.
    Shape,
    /// Checked batch width or result accounting overflowed.
    ArithmeticOverflow,
    /// Transactional output staging could not reserve memory.
    Allocation,
    /// A row correction failed with its exact typed postprocess status.
    Distribution(PostprocessDistributionStatus),
}

#[cfg(any(test, feature = "bench-internal"))]
impl fmt::Display for CompactBatchPostprocessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Session(status) => status.fmt(formatter),
            Self::Shape => formatter.write_str("postprocess batch input/output shape is invalid"),
            Self::ArithmeticOverflow => {
                formatter.write_str("postprocess batch arithmetic overflow")
            }
            Self::Allocation => formatter.write_str("postprocess batch staging allocation failed"),
            Self::Distribution(status) => status.fmt(formatter),
        }
    }
}

#[cfg(any(test, feature = "bench-internal"))]
impl std::error::Error for CompactBatchPostprocessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Shape | Self::ArithmeticOverflow | Self::Allocation => None,
            Self::Session(status) => Some(status),
            Self::Distribution(status) => Some(status),
        }
    }
}

#[cfg(any(test, feature = "bench-internal"))]
fn validate_batch_target_slice_lengths(
    target: &CompactBatchTargetPhysics<'_>,
    batch_len: usize,
) -> Result<(), CompactBatchPostprocessError> {
    for (_, values) in [
        ("am_ratio", target.am_ratio),
        ("drag_coefficient", target.drag_coefficient),
        ("reflectivity_coefficient", target.reflectivity_coefficient),
    ] {
        if let Some(values) = values {
            if values.len() != batch_len {
                return Err(CompactBatchPostprocessError::Shape);
            }
        }
    }
    Ok(())
}

#[cfg(any(test, feature = "bench-internal"))]
fn validate_compact_batch_input_lengths(
    candidates: &[Option<CompactTransferCandidate>],
    primary_slice: &[f64],
    secondary_slice: &[f64],
    intercept_slice: &[f64],
    conjunction_slice: &[f64],
    corrected_slice: &[f64],
    correction_slice: &[f64],
    status_slice: &[i32],
    primary_target: &CompactBatchTargetPhysics<'_>,
    secondary_target: &CompactBatchTargetPhysics<'_>,
) -> Result<(), CompactBatchPostprocessError> {
    let batch_len = candidates.len();
    let state_len = batch_len
        .checked_mul(6)
        .ok_or(CompactBatchPostprocessError::ArithmeticOverflow)?;
    for (_, actual_len, expected_len) in [
        ("primary state", primary_slice.len(), state_len),
        ("secondary state", secondary_slice.len(), state_len),
        ("intercept", intercept_slice.len(), batch_len),
        ("conjunction", conjunction_slice.len(), batch_len),
        ("corrected state", corrected_slice.len(), state_len),
        ("correction", correction_slice.len(), batch_len),
        ("status", status_slice.len(), batch_len),
    ] {
        if actual_len != expected_len {
            return Err(CompactBatchPostprocessError::Shape);
        }
    }
    validate_batch_target_slice_lengths(primary_target, batch_len)?;
    validate_batch_target_slice_lengths(secondary_target, batch_len)
}

#[cfg(any(test, feature = "bench-internal"))]
fn correct_compact_candidate_batch(
    core: &TransferPostprocessSessionCore,
    candidates: &[Option<CompactTransferCandidate>],
    primary_slice: &[f64],
    secondary_slice: &[f64],
    intercept_slice: &[f64],
    conjunction_slice: &[f64],
    corrected_slice: &mut [f64],
    correction_slice: &mut [f64],
    status_slice: &mut [i32],
    primary_target: &CompactBatchTargetPhysics<'_>,
    secondary_target: &CompactBatchTargetPhysics<'_>,
    scratch: &mut TransferPostprocessScratch,
) -> Result<usize, CompactBatchPostprocessError> {
    validate_compact_batch_input_lengths(
        candidates,
        primary_slice,
        secondary_slice,
        intercept_slice,
        conjunction_slice,
        corrected_slice,
        correction_slice,
        status_slice,
        primary_target,
        secondary_target,
    )?;

    let batch_len = candidates.len();
    let mut success_count = 0usize;
    scratch.last_batch_len = batch_len;
    let mut staged_corrected = Vec::new();
    staged_corrected
        .try_reserve_exact(corrected_slice.len())
        .map_err(|_| CompactBatchPostprocessError::Allocation)?;
    staged_corrected.extend_from_slice(corrected_slice);
    let mut staged_correction = Vec::new();
    staged_correction
        .try_reserve_exact(correction_slice.len())
        .map_err(|_| CompactBatchPostprocessError::Allocation)?;
    staged_correction.extend_from_slice(correction_slice);
    let mut staged_status = Vec::new();
    staged_status
        .try_reserve_exact(status_slice.len())
        .map_err(|_| CompactBatchPostprocessError::Allocation)?;
    staged_status.extend_from_slice(status_slice);
    let inputs = candidates
        .iter()
        .zip(primary_slice.chunks_exact(6))
        .zip(secondary_slice.chunks_exact(6))
        .zip(intercept_slice.iter())
        .zip(conjunction_slice.iter());
    let outputs = staged_corrected
        .chunks_exact_mut(6)
        .zip(staged_correction.iter_mut())
        .zip(staged_status.iter_mut());
    for (idx, (input, output)) in inputs.zip(outputs).enumerate() {
        let ((((pick_opt, primary_chunk), secondary_chunk), &intercept_jd), &conjunction_jd) =
            input;
        let ((corrected_state, correction_output), status_output) = output;
        let primary_state: &[f64; 6] = primary_chunk
            .try_into()
            .map_err(|_| CompactBatchPostprocessError::Shape)?;
        let secondary_state: &[f64; 6] = secondary_chunk
            .try_into()
            .map_err(|_| CompactBatchPostprocessError::Shape)?;

        let target_slices = if pick_opt
            .as_ref()
            .is_some_and(|candidate| candidate.target_index > 0)
        {
            secondary_target
        } else {
            primary_target
        };
        let target_am = target_slices
            .am_ratio
            .and_then(|values| values.get(idx))
            .copied();
        let target_drag_coefficient = target_slices
            .drag_coefficient
            .and_then(|values| values.get(idx))
            .copied();
        let target_reflectivity_coefficient = target_slices
            .reflectivity_coefficient
            .and_then(|values| values.get(idx))
            .copied();

        let status = match pick_opt {
            None => POSTPROCESS_STATUS_MISSING_CANDIDATE,
            Some(candidate)
                if !compact_candidate_is_postprocess_coherent(candidate, conjunction_jd) =>
            {
                POSTPROCESS_STATUS_INVALID_CANDIDATE
            }
            Some(candidate) if !candidate_matches_requested_intercept(candidate, intercept_jd) => {
                POSTPROCESS_STATUS_STALE_INTERCEPT_INPUT
            }
            Some(candidate) => {
                match core.correct_one(
                    Some(candidate),
                    primary_state,
                    secondary_state,
                    intercept_jd,
                    conjunction_jd,
                    target_am,
                    target_drag_coefficient,
                    target_reflectivity_coefficient,
                    Some(scratch),
                ) {
                    Ok(Some((state, correction_dv))) => {
                        corrected_state.copy_from_slice(&state);
                        *correction_output = correction_dv;
                        success_count = success_count
                            .checked_add(1)
                            .ok_or(CompactBatchPostprocessError::ArithmeticOverflow)?;
                        POSTPROCESS_STATUS_OK
                    }
                    Ok(None) if core.hf_refine_failure_mode() => {
                        POSTPROCESS_STATUS_HF_REFINE_FAILED
                    }
                    Ok(None) => POSTPROCESS_STATUS_NO_CORRECTION,
                    Err(error) => {
                        return Err(CompactBatchPostprocessError::Distribution(error));
                    }
                }
            }
        };
        *status_output = status;
    }
    corrected_slice.copy_from_slice(&staged_corrected);
    correction_slice.copy_from_slice(&staged_correction);
    status_slice.copy_from_slice(&staged_status);
    Ok(success_count)
}

/// Batch-correct compact candidates without exposing a panic-capable buffer API.
///
/// # Errors
///
/// Returns a typed configuration, shape, arithmetic, allocation, or row
/// correction failure. Typed row failures leave all output buffers unchanged.
#[cfg(any(test, feature = "bench-internal"))]
pub fn batch_postprocess_compact_candidates(
    inputs: CompactBatchPostprocessInputs<'_>,
    outputs: CompactBatchPostprocessOutputs<'_>,
) -> Result<usize, CompactBatchPostprocessError> {
    let CompactBatchPostprocessInputs {
        candidates,
        primary_states,
        secondary_states,
        intercept_jds,
        conjunction_jds,
        physics_config,
        postprocess_config,
        primary_target,
        secondary_target,
    } = inputs;
    let CompactBatchPostprocessOutputs {
        corrected_states,
        correction_dvs,
        status_codes,
    } = outputs;
    // Invalid configuration is a caller error and must surface as a Result,
    // never as an uncatchable panic across the Python boundary.
    let core = TransferPostprocessSessionCore::try_new(physics_config, postprocess_config)
        .map_err(CompactBatchPostprocessError::Session)?;
    let mut scratch = TransferPostprocessScratch::default();
    let success_count = correct_compact_candidate_batch(
        &core,
        candidates,
        primary_states,
        secondary_states,
        intercept_jds,
        conjunction_jds,
        corrected_states,
        correction_dvs,
        status_codes,
        &primary_target,
        &secondary_target,
        &mut scratch,
    )?;
    Ok(success_count)
}

#[cfg(test)]
const fn propagation_fidelity_token(fidelity: crate::types::PropagationFidelity) -> &'static str {
    match fidelity {
        crate::types::PropagationFidelity::HighFidelity => "high_fidelity",
        crate::types::PropagationFidelity::J2 => "analytical_j2",
    }
}

#[cfg(test)]
const fn propagation_force_scope_token(
    fidelity: crate::types::PropagationFidelity,
    strict_hybrid: bool,
) -> &'static str {
    match fidelity {
        crate::types::PropagationFidelity::HighFidelity if strict_hybrid => {
            "spherical_harmonics_5x5_drag_srp_sun_moon"
        }
        crate::types::PropagationFidelity::HighFidelity => "unsealed_high_fidelity",
        crate::types::PropagationFidelity::J2 => "analytical_j2_only",
    }
}

#[cfg(test)]
const fn release_control_fidelity_error(
    use_high_fidelity: bool,
    require_hf_transfer_correction: bool,
    analytical_j2_only: bool,
    target_propagation_mode: u8,
) -> Option<&'static str> {
    let strict_hf = use_high_fidelity && require_hf_transfer_correction;
    let strict_mf_j2 = !use_high_fidelity && !require_hf_transfer_correction;
    if analytical_j2_only && (!strict_mf_j2 || target_propagation_mode != 1) {
        Some("release_control_fraction_batch analytical J2 mode requires MfJ2 target propagation")
    } else if !analytical_j2_only && !strict_hf {
        Some("release_control_fraction_batch requires high-fidelity transfer propagation")
    } else {
        None
    }
}

#[cfg(test)]
mod fixed_canister_batch_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn strict_gravity_cache_retries_failures_and_reuses_success() {
        let cache = OnceLock::new();
        let cold_load = Mutex::new(());
        let attempts = AtomicUsize::new(0);
        let first = success_only_cached(&cache, &cold_load, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Err::<u8, _>(PostprocessSessionError::StrictHfEmbeddedGravityUnavailable)
        });
        assert!(first.is_err());
        assert!(cache.get().is_none());

        let second = success_only_cached(&cache, &cold_load, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok::<u8, PostprocessSessionError>(7)
        })
        .expect("retry succeeds");
        let third = success_only_cached(&cache, &cold_load, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok::<u8, PostprocessSessionError>(9)
        })
        .expect("success is cached");
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
        assert!(std::ptr::eq(second, third));
        assert_eq!(*third, 7);
    }

    #[test]
    fn strict_gravity_cache_serializes_successful_cold_load() {
        use std::sync::mpsc;
        use std::time::Duration;

        let cache = OnceLock::new();
        let cold_load = Mutex::new(());
        let attempts = AtomicUsize::new(0);
        let (first_entered_tx, first_entered_rx) = mpsc::channel();
        let (release_first_tx, release_first_rx) = mpsc::channel();
        let (second_started_tx, second_started_rx) = mpsc::channel();
        let (second_initialized_tx, second_initialized_rx) = mpsc::channel();

        let (second_raced, first_value, second_value) = std::thread::scope(|scope| {
            let cache_ref = &cache;
            let cold_load_ref = &cold_load;
            let attempts_ref = &attempts;
            let first = scope.spawn(move || {
                *success_only_cached(cache_ref, cold_load_ref, || {
                    attempts_ref.fetch_add(1, Ordering::SeqCst);
                    first_entered_tx.send(()).expect("signal first initializer");
                    release_first_rx.recv().expect("release first initializer");
                    Ok::<u8, PostprocessSessionError>(7)
                })
                .expect("first cache load")
            });

            first_entered_rx
                .recv()
                .expect("first initializer must enter");

            let cache_ref = &cache;
            let cold_load_ref = &cold_load;
            let attempts_ref = &attempts;
            let second = scope.spawn(move || {
                second_started_tx.send(()).expect("signal second caller");
                *success_only_cached(cache_ref, cold_load_ref, || {
                    attempts_ref.fetch_add(1, Ordering::SeqCst);
                    second_initialized_tx
                        .send(())
                        .expect("signal duplicate initializer");
                    Ok::<u8, PostprocessSessionError>(9)
                })
                .expect("second cache load")
            });

            second_started_rx.recv().expect("second caller must start");
            let second_raced = second_initialized_rx
                .recv_timeout(Duration::from_millis(250))
                .is_ok();

            release_first_tx
                .send(())
                .expect("release first initializer");

            let first_value = first.join().expect("first cache thread");
            let second_value = second.join().expect("second cache thread");
            (second_raced, first_value, second_value)
        });

        assert!(!second_raced, "second cold initializer ran concurrently");
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(first_value, 7);
        assert_eq!(second_value, 7);
    }

    #[test]
    fn release_control_fidelity_modes_reject_contradictory_flags() {
        assert_eq!(release_control_fidelity_error(true, true, false, 0), None);
        assert_eq!(release_control_fidelity_error(false, false, true, 1), None);
        assert!(release_control_fidelity_error(true, true, true, 0)
            .expect("HF config must fail analytical mode")
            .contains("analytical J2"));
        assert!(release_control_fidelity_error(false, false, false, 1)
            .expect("MF config must fail strict HF mode")
            .contains("high-fidelity"));
        assert!(release_control_fidelity_error(false, false, true, 2).is_some());
        assert_eq!(release_control_fidelity_error(true, true, false, 1), None);
        assert_eq!(release_control_fidelity_error(true, true, false, 2), None);
        assert!(release_control_fidelity_error(true, false, true, 1).is_some());
        assert!(release_control_fidelity_error(false, true, false, 0).is_some());
    }

    #[test]
    fn release_control_provenance_separates_fidelity_from_force_scope() {
        assert_eq!(
            propagation_fidelity_token(crate::types::PropagationFidelity::HighFidelity),
            "high_fidelity"
        );
        assert_eq!(
            propagation_force_scope_token(crate::types::PropagationFidelity::HighFidelity, true,),
            "spherical_harmonics_5x5_drag_srp_sun_moon"
        );
        assert_eq!(
            propagation_fidelity_token(crate::types::PropagationFidelity::J2),
            "analytical_j2"
        );
        assert_eq!(
            propagation_force_scope_token(crate::types::PropagationFidelity::J2, false),
            "analytical_j2_only"
        );
    }

    #[test]
    fn batch_postprocess_rejects_invalid_config_as_error_not_panic() {
        let post = PostprocessConfig {
            canister_cd: 0.0,
            ..default_postprocess_config()
        };
        let mut corrected = [f64::NAN; 6];
        let mut correction = [f64::NAN; 1];
        let mut status = [-1_i32; 1];
        let error: crate::CompactBatchPostprocessError = batch_postprocess_compact_candidates(
            CompactBatchPostprocessInputs {
                candidates: &[None],
                primary_states: &[0.0; 6],
                secondary_states: &[0.0; 6],
                intercept_jds: &[2_460_000.5],
                conjunction_jds: &[2_460_000.6],
                physics_config: None,
                postprocess_config: Some(post),
                primary_target: CompactBatchTargetPhysics::default(),
                secondary_target: CompactBatchTargetPhysics::default(),
            },
            CompactBatchPostprocessOutputs {
                corrected_states: &mut corrected,
                correction_dvs: &mut correction,
                status_codes: &mut status,
            },
        )
        .expect_err("invalid postprocess configuration must return an error");
        assert_eq!(
            error,
            CompactBatchPostprocessError::Session(
                PostprocessSessionError::PostprocessConfiguration
            )
        );
    }

    #[test]
    fn batch_postprocess_preserves_unknown_integrator_status_without_output_mutation() {
        let physics = PhysicsConfig {
            method: "unknown-integrator".to_owned(),
            ..PhysicsConfig::default()
        };
        let mut corrected = [-17.0; 6];
        let mut correction = [-18.0; 1];
        let mut status = [-19_i32; 1];

        let error = batch_postprocess_compact_candidates(
            CompactBatchPostprocessInputs {
                candidates: &[None],
                primary_states: &[0.0; 6],
                secondary_states: &[0.0; 6],
                intercept_jds: &[2_460_000.5],
                conjunction_jds: &[2_460_000.6],
                physics_config: Some(physics),
                postprocess_config: None,
                primary_target: CompactBatchTargetPhysics::default(),
                secondary_target: CompactBatchTargetPhysics::default(),
            },
            CompactBatchPostprocessOutputs {
                corrected_states: &mut corrected,
                correction_dvs: &mut correction,
                status_codes: &mut status,
            },
        )
        .expect_err("unknown integrator must retain its typed session status");

        assert_eq!(
            error,
            CompactBatchPostprocessError::Session(PostprocessSessionError::PhysicsConfiguration(
                PhysicsConfigError::UnsupportedIntegratorMethod
            ))
        );
        assert_eq!(corrected.map(f64::to_bits), [(-17.0_f64).to_bits(); 6]);
        assert_eq!(correction.map(f64::to_bits), [(-18.0_f64).to_bits(); 1]);
        assert_eq!(status, [-19_i32; 1]);
    }

    #[test]
    fn batch_postprocess_rejects_misaligned_parallel_slices_without_mutation() {
        let mut corrected = [f64::NAN; 6];
        let mut correction = [f64::NAN; 1];
        let mut status = [-1_i32; 1];

        let result = batch_postprocess_compact_candidates(
            CompactBatchPostprocessInputs {
                candidates: &[None],
                primary_states: &[0.0; 5],
                secondary_states: &[0.0; 6],
                intercept_jds: &[2_460_000.5],
                conjunction_jds: &[2_460_000.6],
                physics_config: None,
                postprocess_config: None,
                primary_target: CompactBatchTargetPhysics::default(),
                secondary_target: CompactBatchTargetPhysics::default(),
            },
            CompactBatchPostprocessOutputs {
                corrected_states: &mut corrected,
                correction_dvs: &mut correction,
                status_codes: &mut status,
            },
        );

        assert_eq!(result, Err(CompactBatchPostprocessError::Shape));
        assert!(corrected.iter().all(|value| value.is_nan()));
        assert!(correction.iter().all(|value| value.is_nan()));
        assert_eq!(status, [-1_i32; 1]);
    }

    fn coherent_compact_candidate(intercept_jd: f64) -> CompactTransferCandidate {
        let transfer_tof_s = 60.0;
        CompactTransferCandidate {
            valid: true,
            target_index: 0,
            total_dv: 0.0,
            phase_dv_norm: 0.0,
            transfer_dv_norm: 0.0,
            transfer_tof_s,
            total_time_s: transfer_tof_s,
            relative_velocity_km_s: 1.0,
            time_per_relative_velocity_s_per_km_s: transfer_tof_s,
            solver_intercept_jd: intercept_jd,
            tof_jd_start: intercept_jd - transfer_tof_s / satpy_core::SEC_PER_DAY,
            payload_intercept_state: [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            target_intercept_state: [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            transfer_burn_pre_state: [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            transfer_dv: [0.0; 3],
            ..CompactTransferCandidate::default()
        }
    }

    fn strict_missing_assets_batch_core() -> TransferPostprocessSessionCore {
        let physics_config = PhysicsConfig {
            use_high_fidelity: true,
            require_hf_transfer_correction: true,
            ..PhysicsConfig::default()
        };
        let postprocess_config = default_postprocess_config();
        TransferPostprocessSessionCore {
            physics_config,
            runtime_settings: ResolvedPostprocessRuntimeSettings::from_postprocess_config(
                &postprocess_config,
            ),
            postprocess_config,
            coeffs: GlobalCoeffs {
                packed: None,
                missing: true,
            },
            strict_hf_gravity_identity: None,
            hf_missing_coeffs_strict: true,
        }
    }

    #[test]
    fn compact_batch_propagates_typed_row_error_without_partial_output() {
        let intercept_jd = 2_460_000.5;
        let conjunction_jd = intercept_jd + 60.0 / satpy_core::SEC_PER_DAY;
        let candidates = [None, Some(coherent_compact_candidate(intercept_jd))];
        let mut corrected = [-91.0; 12];
        let mut correction = [-92.0; 2];
        let mut status = [-93_i32; 2];
        let target = CompactBatchTargetPhysics {
            am_ratio: None,
            drag_coefficient: None,
            reflectivity_coefficient: None,
        };
        let mut scratch = TransferPostprocessScratch::default();

        let error = correct_compact_candidate_batch(
            &strict_missing_assets_batch_core(),
            &candidates,
            &[0.0; 12],
            &[0.0; 12],
            &[intercept_jd; 2],
            &[conjunction_jd; 2],
            &mut corrected,
            &mut correction,
            &mut status,
            &target,
            &target,
            &mut scratch,
        )
        .expect_err("typed row failure must escape the compact batch");

        assert_eq!(
            error,
            CompactBatchPostprocessError::Distribution(
                PostprocessDistributionStatus::StrictHfAssetsUnavailable,
            )
        );
        assert_eq!(corrected.map(f64::to_bits), [(-91.0_f64).to_bits(); 12]);
        assert_eq!(correction.map(f64::to_bits), [(-92.0_f64).to_bits(); 2]);
        assert_eq!(status, [-93_i32; 2]);
    }
}

#[cfg(test)]
mod fixed_canister_validation_tests {
    use super::*;

    fn assert_invalid(field: &str, mutate: fn(&mut PostprocessConfig)) {
        let mut post = default_postprocess_config();
        mutate(&mut post);
        let Err(error) = TransferPostprocessSessionCore::try_new(None, Some(post)) else {
            panic!("invalid {field} must fail at postprocess core construction");
        };
        assert_eq!(
            error,
            PostprocessSessionError::PostprocessConfiguration,
            "{field}"
        );
    }

    #[test]
    fn fixed_canister_core_rejects_nonphysical_or_nonfinite_tuple() {
        let invalid: [(&str, fn(&mut PostprocessConfig)); 6] = [
            ("canister_am", |post| {
                post.canister_am = 0.0;
            }),
            ("canister_am", |post: &mut PostprocessConfig| {
                post.canister_am = f64::NAN;
            }),
            ("canister_cd", |post: &mut PostprocessConfig| {
                post.canister_cd = 0.0;
            }),
            ("canister_cd", |post: &mut PostprocessConfig| {
                post.canister_cd = f64::INFINITY;
            }),
            ("canister_cr", |post: &mut PostprocessConfig| {
                post.canister_cr = -1.0e-12;
            }),
            ("canister_cr", |post: &mut PostprocessConfig| {
                post.canister_cr = f64::NAN;
            }),
        ];
        for (field, mutate) in invalid {
            assert_invalid(field, mutate);
        }

        let mut zero_cr = default_postprocess_config();
        zero_cr.canister_cr = 0.0;
        assert!(TransferPostprocessSessionCore::try_new(None, Some(zero_cr)).is_ok());
    }

    #[test]
    fn fixed_canister_core_rejects_invalid_timing_dv_and_tolerances() {
        let invalid: [(&str, fn(&mut PostprocessConfig)); 6] = [
            ("dust_phase_tof_s", |post| {
                post.dust_phase_tof_s = 0.0;
            }),
            ("dust_phase_tof_s", |post: &mut PostprocessConfig| {
                post.dust_phase_tof_s = f64::NAN;
            }),
            ("max_physical_dv_kms", |post: &mut PostprocessConfig| {
                post.max_physical_dv_kms = 0.0;
            }),
            ("max_physical_dv_kms", |post: &mut PostprocessConfig| {
                post.max_physical_dv_kms = f64::INFINITY;
            }),
            ("fix_ls_tol", |post: &mut PostprocessConfig| {
                post.fix_ls_tol = 0.0;
            }),
            ("fix_ls_skip_tol", |post: &mut PostprocessConfig| {
                post.fix_ls_skip_tol = -1.0e-12;
            }),
        ];
        for (field, mutate) in invalid {
            assert_invalid(field, mutate);
        }
    }
}

#[cfg(test)]
mod strict_hybrid_api_tests {
    use super::*;

    fn gravity_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .expect("gravity test lock")
    }

    struct RestoreGlobalCoeffs(std::sync::Arc<lightyear_odeint_rs::config::GlobalCoeffs>);

    impl Drop for RestoreGlobalCoeffs {
        fn drop(&mut self) {
            lightyear_odeint_rs::config::GLOBAL_COEFFS.store(self.0.clone());
        }
    }

    fn strict_physics() -> PhysicsConfig {
        PhysicsConfig {
            use_high_fidelity: true,
            require_hf_transfer_correction: true,
            sph_order: 5,
            force_flags: ForceFlags::DRAG
                | ForceFlags::SRP
                | ForceFlags::SUN_GRAVITY
                | ForceFlags::MOON_GRAVITY,
            // Read from the authority, never a literal. This helper is named
            // for PRODUCTION physics, so a hardcoded copy silently stops
            // representing production the moment the compiled science moves --
            // which is exactly what happened when `atmosphere_model` went 4 ->
            // 5: this said 4, the authority said 5, and the only thing that
            // noticed was the one assertion below that compares the two.
            atm_model: StrictHfForceAuthority::PART_A.atmosphere_model,
            am_ratio: 1.948,
            cd: 2.2,
            cr: 1.3,
            method: StrictHfForceAuthority::PART_A.integrator_method.to_owned(),
            dt_max: 300.0,
            tolerance: 1.0e-8,
            ..PhysicsConfig::default()
        }
    }

    fn part_a_v3_test_arc() -> (f64, f64) {
        let start = jb_rs::drivers::compiled_part_a_v3_identity()
            .expect("compiled Part A v3 atmosphere identity")
            .t0_utc_jd;
        (start, start + 0.1)
    }

    #[test]
    fn part_a_v3_strict_static_preflight_selects_persistence_authority() {
        let _guard = gravity_test_guard();
        let core = TransferPostprocessSessionCore::try_new(
            Some(strict_physics()),
            Some(default_postprocess_config()),
        )
        .expect("Part A v3 strict session");

        assert_eq!(core.physics_config.atm_model, 8);
        let selected =
            lightyear_odeint_rs::rhs::jb2008_driver_authority(core.physics_config.atm_model)
                .expect("Part A v3 atmosphere provider");
        assert_eq!(
            selected,
            lightyear_odeint_rs::rhs::Jb2008DriverAuthority::PartAV3PersistenceV1
        );
    }

    fn strict_core_with_coeffs(coeffs: GlobalCoeffs) -> TransferPostprocessSessionCore {
        let physics_config = strict_physics();
        let postprocess_config = default_postprocess_config();
        let runtime_settings =
            ResolvedPostprocessRuntimeSettings::from_postprocess_config(&postprocess_config);
        TransferPostprocessSessionCore {
            hf_missing_coeffs_strict: coeffs.missing,
            physics_config,
            postprocess_config,
            coeffs,
            strict_hf_gravity_identity: None,
            runtime_settings,
        }
    }

    #[test]
    fn runtime_settings_take_gmm_component_count_from_postprocess_authority() {
        let mut post = default_postprocess_config();
        post.gmm_components = 4;
        assert_eq!(
            ResolvedPostprocessRuntimeSettings::from_postprocess_config(&post).num_dists,
            4
        );
    }

    #[test]
    fn strict_constructor_rejects_missing_gravity_without_changing_mf() {
        let missing = GlobalCoeffs {
            packed: None,
            missing: true,
        };
        let result = TransferPostprocessSessionCore::try_new_with_coeffs(
            Some(strict_physics()),
            Some(default_postprocess_config()),
            missing.clone(),
        );
        let Err(error) = result else {
            panic!("strict HF session must reject missing gravity at construction");
        };
        assert_eq!(error, PostprocessSessionError::StrictHfGravityMissing);
        assert!(TransferPostprocessSessionCore::try_new_with_coeffs(None, None, missing).is_ok());
    }

    #[test]
    fn constructor_rejects_unknown_integrator_before_strict_asset_validation() {
        let mut physics = strict_physics();
        physics.method = "unknown-integrator".to_owned();
        let missing = GlobalCoeffs {
            packed: None,
            missing: true,
        };

        let Err(error) = TransferPostprocessSessionCore::try_new_with_coeffs(
            Some(physics),
            Some(default_postprocess_config()),
            missing,
        ) else {
            panic!("unknown integrator must reject before strict gravity assets");
        };

        assert_eq!(
            error,
            PostprocessSessionError::PhysicsConfiguration(
                PhysicsConfigError::UnsupportedIntegratorMethod
            )
        );
    }

    #[test]
    fn public_constructor_rejects_unknown_integrator_before_strict_asset_resolution() {
        let _guard = gravity_test_guard();
        let previous = lightyear_odeint_rs::config::GLOBAL_COEFFS.load_full();
        let _restore_coeffs = RestoreGlobalCoeffs(previous);
        lightyear_odeint_rs::config::GLOBAL_COEFFS.store(std::sync::Arc::new(
            lightyear_odeint_rs::config::GlobalCoeffs::Unloaded,
        ));
        let mut physics = strict_physics();
        physics.method = "unknown-integrator".to_owned();

        let Err(error) = TransferPostprocessSessionCore::try_new(
            Some(physics),
            Some(default_postprocess_config()),
        ) else {
            panic!("unknown integrator must reject before strict gravity resolution");
        };

        assert_eq!(
            error,
            PostprocessSessionError::PhysicsConfiguration(
                PhysicsConfigError::UnsupportedIntegratorMethod
            )
        );
        assert!(lightyear_odeint_rs::get_global_coeffs_packed().is_none());
    }

    #[test]
    fn strict_hf_context_reports_missing_gravity_coefficients() {
        let core = strict_core_with_coeffs(GlobalCoeffs {
            packed: None,
            missing: true,
        });

        assert!(matches!(
            core.strict_hf_context_for_arc(2_460_000.5, 2_460_000.6),
            Err(StrictHfContextStatus::MissingGravityCoefficients)
        ));
    }

    #[test]
    fn strict_hf_ephemeris_status_exposes_typed_source() {
        let status = StrictHfContextStatus::Ephemeris(EphemerisCoverageError::CatalogueLoad {
            requested_flags: ForceFlags::SUN_GRAVITY,
            message: "hostile catalogue".to_owned(),
        });

        let source = std::error::Error::source(&status)
            .expect("ephemeris status must retain its typed source");
        let ephemeris = source
            .downcast_ref::<EphemerisCoverageError>()
            .expect("status source must remain an ephemeris coverage error");
        assert!(matches!(
            ephemeris,
            EphemerisCoverageError::CatalogueLoad {
                requested_flags,
                message,
            } if *requested_flags == ForceFlags::SUN_GRAVITY && message == "hostile catalogue"
        ));
    }

    #[test]
    fn strict_session_pack_and_identity_ignore_preloaded_global() -> anyhow::Result<()> {
        let _guard = gravity_test_guard();
        let physics = strict_physics();
        let previous = lightyear_odeint_rs::config::GLOBAL_COEFFS.load_full();
        let _restore_coeffs = RestoreGlobalCoeffs(previous);

        let arbitrary = lightyear_odeint_rs::packed_constants_from_bytes(
            b"2 0 1.0e-3 0.0\n",
            physics.sph_order,
        )?;
        let arbitrary_semantic_sha256 = arbitrary.authority_sha256()?;
        lightyear_odeint_rs::config::GLOBAL_COEFFS.store(std::sync::Arc::new(
            lightyear_odeint_rs::config::GlobalCoeffs::Loaded(Arc::clone(&arbitrary)),
        ));

        let core = TransferPostprocessSessionCore::try_new(
            Some(physics),
            Some(default_postprocess_config()),
        )?;
        let identity = core
            .strict_hf_gravity_identity()
            .ok_or_else(|| anyhow::anyhow!("strict session must retain its gravity identity"))?;
        let session_pack = core
            .coeffs
            .packed
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("strict session must retain its gravity pack"))?;

        anyhow::ensure!(
            !Arc::ptr_eq(session_pack, &arbitrary),
            "strict session reused hostile global gravity pack"
        );
        anyhow::ensure!(
            identity.packed_semantic_sha256() != arbitrary_semantic_sha256,
            "strict session retained hostile global gravity identity"
        );
        anyhow::ensure!(
            session_pack.authority_sha256()? == identity.packed_semantic_sha256(),
            "strict session pack and retained gravity identity differ"
        );
        anyhow::ensure!(
            identity == canonical_strict_hf_gravity_identity()?,
            "strict session retained noncanonical gravity identity"
        );
        let installed_semantic_sha256 = lightyear_odeint_rs::get_global_coeffs_packed()
            .ok_or_else(|| anyhow::anyhow!("hostile global must remain installed"))?
            .authority_sha256()?;
        anyhow::ensure!(
            installed_semantic_sha256 == arbitrary_semantic_sha256,
            "strict session mutated hostile global gravity pack"
        );
        Ok(())
    }

    #[test]
    fn strict_sessions_share_the_cached_canonical_gravity_pack() -> anyhow::Result<()> {
        let first = TransferPostprocessSessionCore::try_new(
            Some(strict_physics()),
            Some(default_postprocess_config()),
        )?;
        let second = TransferPostprocessSessionCore::try_new(
            Some(strict_physics()),
            Some(default_postprocess_config()),
        )?;
        let first_pack = first
            .coeffs
            .packed
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("first strict session lacks gravity"))?;
        let second_pack = second
            .coeffs
            .packed
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("second strict session lacks gravity"))?;

        anyhow::ensure!(
            Arc::ptr_eq(first_pack, second_pack),
            "strict sessions reparsed canonical gravity instead of cloning its immutable pack"
        );
        anyhow::ensure!(
            first.strict_hf_gravity_identity() == second.strict_hf_gravity_identity(),
            "strict sessions disagreed on canonical gravity identity"
        );
        Ok(())
    }

    #[test]
    fn strict_hf_context_keeps_session_force_and_ephemeris_authority() {
        let _guard = gravity_test_guard();
        let core = TransferPostprocessSessionCore::try_new(
            Some(strict_physics()),
            Some(default_postprocess_config()),
        )
        .expect("strict production session");

        let (start, end) = part_a_v3_test_arc();
        let context = core
            .strict_hf_context_for_arc(start, end)
            .expect("native HF assets must prepare context");
        let force = context
            .force_config
            .as_ref()
            .expect("strict context force config");

        assert!(context.is_hf_ready());
        assert!(context.hf_strict);
        assert_eq!(force.sph_order, 5);
        // Against the authority, not a literal, for the reason recorded on
        // `strict_physics`. A literal here would go green again the moment
        // someone "fixed" the helper to match it, which is the drift this pair
        // of assertions exists to catch.
        assert_eq!(
            force.atm_model,
            StrictHfForceAuthority::PART_A.atmosphere_model
        );
        assert_eq!(
            force.force_flags,
            ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY
        );
        assert_eq!(force.am_ratio.to_bits(), 1.948_f64.to_bits());
        assert_eq!(force.cd.to_bits(), 2.2_f64.to_bits());
        assert_eq!(force.cr.to_bits(), 1.3_f64.to_bits());
        assert_eq!(
            force.dynamic_ephemeris_flags,
            ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY
        );
        assert_eq!(
            StrictHfForceAuthority::PART_A.gravity_order,
            force.sph_order
        );
        assert_eq!(
            StrictHfForceAuthority::PART_A.force_flags,
            force.force_flags
        );
        assert_eq!(
            StrictHfForceAuthority::PART_A.atmosphere_model,
            force.atm_model
        );

        let mut invalid_config_core = core;
        invalid_config_core.physics_config.method = "unknown-integrator".to_owned();
        let Err(error) = invalid_config_core.strict_hf_context_for_arc(start, end) else {
            panic!("strict context must retain an unsupported integrator");
        };
        assert_eq!(
            error,
            StrictHfContextStatus::Configuration(PhysicsConfigError::UnsupportedIntegratorMethod)
        );
    }

    #[test]
    fn full_distribution_returns_typed_missing_candidate_status() {
        let core = TransferPostprocessSessionCore::try_new(None, None).expect("MF session");

        assert_eq!(
            core.full_corrected_distribution_one(None, 2_460_000.6, None, None, None, None, None,)
                .expect_err("missing candidate must remain observable"),
            PostprocessDistributionStatus::MissingCandidate
        );
    }
}
