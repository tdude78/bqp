//! Compiled-only Part A scientific authority.
//!
//! Canonical Part A YAML owns matrix/control-plane semantics only. Every
//! numeric physics/objective/replay value below is compiled, hashed, and fed
//! to native callers from this one immutable V1 value.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PART_A_SCIENCE_SCHEMA_VERSION: u32 = 12;

/// Maximum complete exact-design outcomes retained by one Part A objective.
///
/// Runtime admission and receipt validation share this bound; it is execution
/// evidence policy and therefore does not enter the serialized science digest.
pub const PART_A_EXACT_DESIGN_CACHE_MAX_ENTRIES: usize = 16_384;

/// Base of the SYNTHETIC identifier space for Part A deployer spacecraft.
///
/// A deployer is a *designed* spacecraft produced by the constellation decoder.
/// It is not a catalogued object and has no NORAD ID. Stage-1 admission still
/// needs a non-zero identifier that cannot collide with a target's catalogue
/// number, so satellite `i` of a design is identified as
/// `PART_A_DEPLOYER_OBJECT_ID_BASE + i`.
///
/// **This is not a catalogue reference and must never be reported as one.** The
/// base sits far outside every real range: catalogued objects are currently
/// below ~70,000 and Space-Track analyst objects occupy 80,000-89,999, so no
/// value here can be mistaken for an observed object.
///
/// Execution identity only: it is validated, never folded into the serialized
/// science digest.
pub const PART_A_DEPLOYER_OBJECT_ID_BASE: u64 = 900_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PartACredibleIntervalMethod {
    Central,
    Hpd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PartASuccessEstimator {
    PosteriorMean,
    Lcb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PartAObjectiveAggregation {
    Max,
    Quantile,
    Cvar,
    Mean,
}

/// Physics model used while searching one Part A design space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartASearchModel {
    MfJ2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PartACoverageFrontMinBand {
    Qualified,
    MidBand,
}

/// How one shared target-position draw enters the dust-hit model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PartASharedTargetPositionTreatment {
    /// Assumed, not observed: one isotropic Gaussian draw in the encounter
    /// B-plane. Radial and cross-track receive the same 100 m sigma; no old
    /// anisotropic RIC scaling is inherited.
    AssumedNotObservedIsotropicEncounterBplaneSharedDraw,
}

/// Numerical authority for integrating the shared Gaussian target draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PartASharedTargetDrawIntegration {
    /// Rotate the whitened Cartesian tensor rule onto the sharp grain axis;
    /// switch to a normalized six-sigma polar rule when no slow axis exists.
    RotatedCartesianWithBoundedRadialRefinementAndPolarBelow2V2,
}

impl PartASharedTargetDrawIntegration {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RotatedCartesianWithBoundedRadialRefinementAndPolarBelow2V2 => {
                "rotated-cartesian-bounded-radial-refinement-polar-below-2-v2"
            }
        }
    }
}

/// Reportable claim attached to the compiled shared-target scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PartASharedTargetClaim {
    /// Model-conditioned conservative released-mass requirement for >=1-grain
    /// contact at the scenario's `target_hit_probability`. Not a deflection,
    /// real-world calibration, or mathematical lower bound on required mass.
    ModelConditionedConservativeContactRequirement,
}

// TWELVE one-variant MF policy enums were deleted here on 2026-08-06, together
// with the `PartAMfTransferControls` / `PartAMfLoweringControls` fields that
// held them and their hash-tag bytes: `PartAMfSamplingMode::Fast`,
// `PartAMfPairProxyModel::Sum`,
// `PartAMfOxyMooPolicy::FastPopulation20Generations3InitialBest1`,
// `PartAMfDeltaVAnchorPolicy::Full`, `PartAMfPolishScopePolicy::NdEpsilon`,
// `PartAMfTargetPropagationAuthority::MfJ2`,
// `PartAMfTargetBodyForce::J2DiagnosticTarget`,
// `PartAMfLocalOptimizer::NelderMead`, `PartAMfLocalTune::Aggressive`,
// `PartAMfFrontOutputMode::VerifiedSuperset`,
// `PartAMfSplittingCriterion::MaxVariance` and
// `PartAMfSplitAlphaPolicy::ScaledTofSchedule`.
//
// Each carried exactly one variant, so the compiled authority could express
// exactly one value and every consumer's `match` had exactly one arm. They
// encoded no choice: what they actually did was restate, in a second vocabulary
// that had to be kept in sync, a native constant that the caller then
// immediately re-derived. The native selection now lives at the one boundary
// that consumes it -- `nd_pipeline/src/physics/transfer.rs` (solver, sampling,
// target-propagation and front-output policy) and `nd_pipeline/src/hybrid.rs`
// (`"maxvar"` / `"scaled"`).
//
// Re-introducing a genuine choice means adding a real multi-variant enum here,
// not resurrecting these. The remaining enums below carry real choices.
//
// This deletion moves the sealed science digest, because `sha256()` folds
// `serde_json::to_vec(self)` and the twelve JSON keys and variant-name strings
// leave the serialized stream with the fields.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PartATaiEpoch {
    seconds_since_1958_01_01: i64,
    nanosecond: u32,
}

impl PartATaiEpoch {
    /// Create a validated TAI epoch.
    ///
    /// # Errors
    ///
    /// Returns [`PartAAuthorityError`] when `nanosecond` is outside one second.
    pub fn new(
        seconds_since_1958_01_01: i64,
        nanosecond: u32,
    ) -> Result<Self, PartAAuthorityError> {
        let epoch = Self {
            seconds_since_1958_01_01,
            nanosecond,
        };
        epoch.validate()?;
        Ok(epoch)
    }

    const fn validate(self) -> Result<(), PartAAuthorityError> {
        if self.nanosecond >= 1_000_000_000 {
            return Err(PartAAuthorityError::invalid(
                "TAI epoch",
                "nanosecond must be below 1000000000",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PartAEarthOrientationConvention {
    /// Polar motion and UT1-UTC taken as identically zero: no EOP realization
    /// at all. Retained because it is a truthful label for a lineage that never
    /// established one, and REFUSED by
    /// [`PartAVerifiedEventAnchor::validate`] for exactly that reason.
    ZeroEop,
    #[serde(rename = "iers_finals2000a_definitive")]
    IersFinals2000ADefinitive,
}

impl PartAEarthOrientationConvention {
    /// Whether this convention names an Earth-orientation realization that was
    /// actually established.
    const fn is_realized(self) -> bool {
        match self {
            Self::ZeroEop => false,
            Self::IersFinals2000ADefinitive => true,
        }
    }
}

/// Compiled constellation-geometry control.
///
/// The Part A family decoders certify, in closed form, that the all-time
/// minimum pairwise separation of a decoded constellation is at least
/// `min_separation_km`. The certificate is exact under the compiled
/// propagator (first-order secular J2 on mean elements), which holds `a`,
/// `e` and `i` invariant, so no sampling horizon, sample count or phase
/// origin enters the decision.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PartAConstellationControls {
    min_separation_km: f64,
}

impl PartAConstellationControls {
    #[must_use]
    pub const fn min_separation_km(&self) -> f64 {
        self.min_separation_km
    }

    fn validate(self) -> Result<(), PartAAuthorityError> {
        require_positive_finite("constellation min separation", self.min_separation_km)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PartAEventAnchorAuthority {
    Unresolved,
    Verified(PartAVerifiedEventAnchor),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PartAVerifiedEventAnchor {
    source_frame: &'static str,
    time_scale: &'static str,
    realization: &'static str,
    leap_second_table_sha256: &'static str,
    leap_second_table_span: &'static str,
    tai_minus_utc_source: &'static str,
    tt_minus_tai_nanoseconds: i64,
    earth_orientation: PartAEarthOrientationConvention,
    reference_epoch_tai: PartATaiEpoch,
    manifest_sha256: &'static str,
}

/// Typed event-anchor evidence required to construct a verified authority.
#[derive(Debug, Clone, Copy)]
pub struct PartAVerifiedEventAnchorInput {
    pub source_frame: &'static str,
    pub time_scale: &'static str,
    pub realization: &'static str,
    pub leap_second_table_sha256: &'static str,
    pub leap_second_table_span: &'static str,
    pub tai_minus_utc_source: &'static str,
    pub tt_minus_tai_nanoseconds: i64,
    pub earth_orientation: PartAEarthOrientationConvention,
    pub reference_epoch_tai: PartATaiEpoch,
    pub manifest_sha256: &'static str,
}

impl PartAVerifiedEventAnchor {
    /// Create a verified event-anchor authority from complete typed evidence.
    ///
    /// # Errors
    ///
    /// Returns [`PartAAuthorityError`] when any required lineage or time field
    /// is malformed.
    pub fn new(input: PartAVerifiedEventAnchorInput) -> Result<Self, PartAAuthorityError> {
        let authority = Self {
            source_frame: input.source_frame,
            time_scale: input.time_scale,
            realization: input.realization,
            leap_second_table_sha256: input.leap_second_table_sha256,
            leap_second_table_span: input.leap_second_table_span,
            tai_minus_utc_source: input.tai_minus_utc_source,
            tt_minus_tai_nanoseconds: input.tt_minus_tai_nanoseconds,
            earth_orientation: input.earth_orientation,
            reference_epoch_tai: input.reference_epoch_tai,
            manifest_sha256: input.manifest_sha256,
        };
        authority.validate()?;
        Ok(authority)
    }

    fn validate(self) -> Result<(), PartAAuthorityError> {
        for (name, value) in [
            ("event source frame", self.source_frame),
            ("event time scale", self.time_scale),
            ("event realization", self.realization),
            ("leap-second table span", self.leap_second_table_span),
            ("TAI-minus-UTC source", self.tai_minus_utc_source),
        ] {
            require_nonempty(name, value)?;
        }
        require_sha256("leap-second table", self.leap_second_table_sha256)?;
        if self.tt_minus_tai_nanoseconds != 32_184_000_000 {
            return Err(PartAAuthorityError::invalid(
                "event time authority",
                "TT minus TAI must equal 32184000000 nanoseconds",
            ));
        }
        // `earth_orientation` used to be the one provenance field this function
        // skipped, so an anchor declaring `ZeroEop` -- no Earth-orientation
        // realization at all -- validated clean and presented itself as
        // verified lineage. Every sibling above is checked for being
        // well-formed AND compatible with the claim of verification; this is
        // that same check for the one field whose only malformed value is a
        // semantic one.
        //
        // Scope, stated so it is not overread: this binds the compiled LABEL to
        // being a realization. It does not compare the label against the rich
        // EOP object in the sealed event manifest -- nothing in this process
        // parses that manifest, only its SHA-256 -- so label/manifest
        // correspondence remains asserted by the sealing procedure, not here.
        if !self.earth_orientation.is_realized() {
            return Err(PartAAuthorityError::invalid(
                "event Earth-orientation authority",
                "a verified event anchor must declare a realized Earth-orientation convention",
            ));
        }
        self.reference_epoch_tai.validate()?;
        require_sha256("event frame/time manifest", self.manifest_sha256)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PartAGravityAuthority {
    Unresolved,
    Verified(PartAVerifiedGravity),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartAAuthorityError {
    authority: &'static str,
    reason: Option<&'static str>,
}

impl PartAAuthorityError {
    const fn unresolved(authority: &'static str) -> Self {
        Self {
            authority,
            reason: None,
        }
    }

    const fn invalid(authority: &'static str, reason: &'static str) -> Self {
        Self {
            authority,
            reason: Some(reason),
        }
    }
}

impl std::fmt::Display for PartAAuthorityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.reason {
            None => write!(
                formatter,
                "Part A production Hybrid authority unresolved: {}",
                self.authority
            ),
            Some(reason) => write!(
                formatter,
                "Part A production Hybrid authority invalid: {}: {reason}",
                self.authority
            ),
        }
    }
}

impl std::error::Error for PartAAuthorityError {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PartAVerifiedGravity {
    source_model: &'static str,
    normalization: &'static str,
    tide_system: &'static str,
    source_gm_km3_s2: f64,
    source_reference_radius_km: f64,
    source_max_degree: usize,
    source_max_order: usize,
    stored_degree: usize,
    stored_order: usize,
    runtime_degree: usize,
    runtime_order: usize,
    coefficient_sha256: &'static str,
    manifest_sha256: &'static str,
}

/// Typed gravity evidence required to construct a verified authority.
#[derive(Debug, Clone, Copy)]
pub struct PartAVerifiedGravityInput {
    pub source_model: &'static str,
    pub normalization: &'static str,
    pub tide_system: &'static str,
    pub source_gm_km3_s2: f64,
    pub source_reference_radius_km: f64,
    pub source_max_degree: usize,
    pub source_max_order: usize,
    pub stored_degree: usize,
    pub stored_order: usize,
    pub runtime_degree: usize,
    pub runtime_order: usize,
    pub coefficient_sha256: &'static str,
    pub manifest_sha256: &'static str,
}

impl PartAVerifiedGravity {
    /// Create a verified gravity authority from complete typed evidence.
    ///
    /// # Errors
    ///
    /// Returns [`PartAAuthorityError`] when provenance, physical constants, or
    /// resolution bounds are invalid.
    pub fn new(input: PartAVerifiedGravityInput) -> Result<Self, PartAAuthorityError> {
        let authority = Self {
            source_model: input.source_model,
            normalization: input.normalization,
            tide_system: input.tide_system,
            source_gm_km3_s2: input.source_gm_km3_s2,
            source_reference_radius_km: input.source_reference_radius_km,
            source_max_degree: input.source_max_degree,
            source_max_order: input.source_max_order,
            stored_degree: input.stored_degree,
            stored_order: input.stored_order,
            runtime_degree: input.runtime_degree,
            runtime_order: input.runtime_order,
            coefficient_sha256: input.coefficient_sha256,
            manifest_sha256: input.manifest_sha256,
        };
        authority.validate()?;
        Ok(authority)
    }

    fn validate(self) -> Result<(), PartAAuthorityError> {
        for (name, value) in [
            ("gravity source model", self.source_model),
            ("gravity normalization", self.normalization),
            ("gravity tide system", self.tide_system),
        ] {
            require_nonempty(name, value)?;
        }
        require_positive_finite("gravity source GM", self.source_gm_km3_s2)?;
        require_positive_finite(
            "gravity source reference radius",
            self.source_reference_radius_km,
        )?;
        if self.source_max_degree == 0 || self.source_max_order > self.source_max_degree {
            return Err(PartAAuthorityError::invalid(
                "gravity authority",
                "source degree must be positive and source order must not exceed source degree",
            ));
        }
        if self.stored_degree == 0
            || self.stored_order > self.stored_degree
            || self.stored_degree > self.source_max_degree
            || self.stored_order > self.source_max_order
        {
            return Err(PartAAuthorityError::invalid(
                "gravity authority",
                "stored degree/order must be positive, valid, and within source bounds",
            ));
        }
        if self.runtime_degree == 0
            || self.runtime_order > self.runtime_degree
            || self.runtime_degree > self.stored_degree
            || self.runtime_order > self.stored_order
        {
            return Err(PartAAuthorityError::invalid(
                "gravity authority",
                "runtime degree/order must be positive, valid, and within stored bounds",
            ));
        }
        require_sha256("gravity coefficients", self.coefficient_sha256)?;
        require_sha256("gravity manifest", self.manifest_sha256)
    }
}

fn require_nonempty(authority: &'static str, value: &str) -> Result<(), PartAAuthorityError> {
    if value.trim().is_empty() {
        return Err(PartAAuthorityError::invalid(
            authority,
            "value must be nonempty",
        ));
    }
    Ok(())
}

fn require_positive_finite(authority: &'static str, value: f64) -> Result<(), PartAAuthorityError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PartAAuthorityError::invalid(
            authority,
            "value must be finite and positive",
        ));
    }
    Ok(())
}

fn require_sha256(authority: &'static str, value: &str) -> Result<(), PartAAuthorityError> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err(PartAAuthorityError::invalid(
            authority,
            "value must be a lowercase SHA-256 hex digest",
        ));
    }
    Ok(())
}

impl PartAEventAnchorAuthority {
    /// Return verified event-anchor evidence.
    ///
    /// # Errors
    ///
    /// Returns [`PartAAuthorityError`] when authority is unresolved or invalid.
    pub fn require_verified(&self) -> Result<&PartAVerifiedEventAnchor, PartAAuthorityError> {
        match self {
            Self::Unresolved => Err(PartAAuthorityError::unresolved("event_anchor_authority")),
            Self::Verified(authority) => {
                authority.validate()?;
                Ok(authority)
            }
        }
    }
}

impl PartAGravityAuthority {
    /// Return verified gravity evidence.
    ///
    /// # Errors
    ///
    /// Returns [`PartAAuthorityError`] when authority is unresolved or invalid.
    pub fn require_verified(&self) -> Result<&PartAVerifiedGravity, PartAAuthorityError> {
        match self {
            Self::Unresolved => Err(PartAAuthorityError::unresolved("gravity_authority")),
            Self::Verified(authority) => {
                authority.validate()?;
                Ok(authority)
            }
        }
    }
}

/// MF event-bank, adaptive-evaluation, and row-objective controls.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PartAMfControls {
    pub solver_lookback_s: f64,
    pub intercept_offset_s: f64,
    pub event_window_margin_days: f64,
    pub event_sample_seed: i64,
    pub b500_event_count: usize,
    pub event_rotations: [usize; 3],
    pub adaptive_initial_events: usize,
    pub adaptive_event_step: usize,
    pub adaptive_stage_count: usize,
    pub prior_alpha: f64,
    pub prior_beta: f64,
    pub quantile: f64,
    pub confidence: f64,
    pub credible_interval_method: PartACredibleIntervalMethod,
    pub success_rate_estimator: PartASuccessEstimator,
    pub objective_aggregation: PartAObjectiveAggregation,
    pub hpd_grid_size: usize,
    pub hard_fail_success_threshold: f64,
    pub dv_log_floor_km_s: f64,
    pub mass_log_floor_kg: f64,
    pub constellation_size_penalty_alpha: f64,
    pub hard_dv_km_s: f64,
    pub hard_mass_kg: f64,
    pub max_mass_kg: f64,
    pub coverage_front_min_band: PartACoverageFrontMinBand,
    pub convergence_pdf_threshold: f64,
    pub stop_band_confidence: f64,
}

impl PartAMfControls {
    /// Return the zero-based adaptive stage for one exact sealed prefix count.
    #[must_use]
    pub fn adaptive_stage_index(&self, count: usize) -> Option<usize> {
        if count > self.b500_event_count || self.adaptive_event_step == 0 {
            return None;
        }
        let delta = count.checked_sub(self.adaptive_initial_events)?;
        if delta.checked_rem(self.adaptive_event_step)? != 0 {
            return None;
        }
        let stage = delta.checked_div(self.adaptive_event_step)?;
        (stage < self.adaptive_stage_count).then_some(stage)
    }
}

/// MF Stage 1 transfer recipe. Every native solver choice and numeric knob is
/// explicit; no runtime `Default` participates in canonical Part A.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "immutable compiled authority preserves one sealed field layout"
)]
pub struct PartAMfTransferControls {
    pub max_time_s: f64,
    pub max_phase_dv: f64,
    pub max_transfer_dv: f64,
    pub max_revs: i32,
    pub min_perigee_km: f64,
    pub max_apogee_km: f64,
    pub search_pairs_to_verify: usize,
    pub tof_sample_budget: usize,
    pub coarse_early_stop: bool,
    pub fine_total_limit: usize,
    pub coarse_reject_margin_km_s: f64,
    pub seed_fine_margin_km_s: f64,
    pub distance_tol_km: f64,
    pub deployer_min_distance_km: f64,
    pub tof_penalty_weight: f64,
    pub revolution_cap: f64,
    pub require_high_fidelity: bool,
    pub force_config_enabled: bool,
    pub j2_max_iterations: usize,
    pub j2_endpoint_target_km: f64,
    pub j2_correction_step_gain: f64,
    pub gravity_coefficients_enabled: bool,
    pub local_optimizer_seed: u64,
    pub warm_start_enabled: bool,
    /// Convert catalogue target states from osculating to MEAN equinoctial
    /// elements before the secular-J2 lane propagates them.
    ///
    /// The lane requires mean elements; catalogue targets are SGP4 outputs, hence
    /// osculating. Feeding them raw biases the semi-major axis by the J2
    /// short-period offset and drifts along-track without bound (738 km over the
    /// 18.6-rev horizon; `satpy_core::mean_elements`).
    ///
    /// DEFAULT FALSE, deliberately. Enabling it moves every reported `dv` by a
    /// large and sign-varying amount — measured +17.1%, -23.6% and -52.1% on the
    /// three fixture events — so every published Part A number would have to be
    /// regenerated by a full campaign re-run in the same change. Until that run
    /// happens, shipping the corrected physics against the old published results
    /// would make code and documentation disagree, which is worse than the
    /// documented defect. See `docs/ELEMENT_THEORY_AND_FIDELITY_AUDIT.md`.
    ///
    /// # 2026-07-25: this must stay FALSE, and the reason is registration, not
    /// # publication timing
    ///
    /// It was flipped to `true` and that BROKE THE MF LANE OUTRIGHT. Under
    /// `part_a_v1()` all three vertical-slice events fail with
    /// `det_mass must be positive and finite, got 0` — a hard `?` propagation at
    /// `nd_pipeline/src/native_mf.rs:633-644`, not a bias and not an
    /// infeasible-marking.
    ///
    /// The cause is that **the event catalogue is not an independent ground
    /// truth**. Its conjunction anchors ARE the raw secular-J2 images of its
    /// start anchors: over all 500 events in `assets/part_a/search_b500_v2.json`,
    /// `equinoc_prop_j2_from_impl(primary_state_start_equinoctial, 2.625 d)`
    /// reproduces `primary_state_conj_eci` to a maximum of 2.07e-10 km, median
    /// exactly 0.0, across 38 revolutions. The 9.9 m `miss_distance_km` that
    /// DEFINES each event is therefore a property of the reduced model, not of
    /// the orbit.
    ///
    /// So converting the target to mean elements does not aim the solver more
    /// accurately at the conjunction — it aims it at a DIFFERENT POINT than the
    /// one the event is defined by. Measured over the production leg for all 500
    /// events, the corrected target sits p50 1704 km, p95 2803 km from the
    /// catalogue anchor. `orchestrate.rs` then hands the RAW catalogue
    /// `other_conj_pos_km` to the det-mass solver while the intercept state is
    /// corrected, `compute_miss_distance_mf_j2` applied no anchored differential
    /// (it does now -- option 2 below LANDED 2026-08-19 as `6a0d023b`, because the
    /// v3 catalogue is not self-referential and forced it),
    /// and `miss0` >> `deterministic_mass_min_distance_km = 1.0` yields
    /// `root_mass_kg: 0.0` / `SafeByDefault`, which is rejected.
    ///
    /// Turning this off is therefore RESTORING INTERNAL CONSISTENCY, not
    /// abandoning a correction. The conversion is only meaningful if the
    /// catalogue is regenerated under the same convention.
    ///
    /// The conversion code itself is correct and is kept: `satpy_core`'s
    /// `mean_equinoctial_from_osculating_state` is exercised by its own tests and
    /// by `part_a_v1_legacy_target_elements()`'s counterpart. What is wrong is
    /// applying it to one side of a self-referential catalogue.
    ///
    /// To enable it properly, one of these had to happen first:
    ///   1. regenerate the event catalogue under the mean convention, or
    ///   2. extend the anchored-differential construction that already protects
    ///      the HF lane to the MF det-mass path, so a model displacement cancels
    ///      instead of entering the miss distance.
    ///
    /// # Option 2 LANDED 2026-08-19 (`6a0d023b`), forced by v3
    ///
    /// It was never optional in the end. The v3 catalogue's conjunctions come
    /// from strict-HF refinement rather than being secular-J2 images of their
    /// own start anchors, so the self-referentiality this whole note rests on is
    /// gone by construction, and the raw difference carried the J2-vs-HF
    /// displacement. Measured: 0 of the first 24 sealed v3 events feasible, all
    /// `SafeByDefault`, against a catalogue miss of 0.0107-0.3733 km.
    /// `MfJ2MassSolverEvent::with_conjunction_anchor` now supplies the anchor
    /// and `orchestrate.rs` passes it; the unanchored path is retained
    /// bit-for-bit for the captured v2-era fixtures.
    ///
    /// THIS FLAG IS STILL FALSE, and the reason has NOT changed: the anchored
    /// differential fixes which POINT the miss is measured from, not which
    /// element set the lane propagates. Enabling the conversion would still move
    /// every published `dv` (+17.1%, -23.6%, -52.1% on the three fixture
    /// events), so it still needs a full campaign re-run in the same change.
    pub target_mean_element_conversion_enabled: bool,
    /// The native MF solver-policy identity, sealed as data.
    ///
    /// # Why this exists, having just been deleted in another shape
    ///
    /// The 2026-08-06 collapse removed twelve one-variant policy enums from
    /// this struct, on the correct observation that they encoded no choice.
    /// What that observation missed — caught by stop-time review the same
    /// day — is that their variant names sat in the serde stream `sha256()`
    /// folds, so they were the SEAL on the live solver policy: after the
    /// collapse, editing the bare constants the consumers kept would have
    /// changed campaign behaviour with zero digest movement.
    ///
    /// This block restores the seal without restoring the twelve enums: the
    /// policy is DATA here (eleven tokens, in the hash), and each consumer
    /// binds its implemented variant to the sealed token with a CONST
    /// assertion — an unknown or edited token on either side fails the
    /// BUILD, which is stronger than the enums ever were (they failed
    /// nothing; an edit to one silently resealed). Consumers:
    /// `nd_pipeline/src/physics/transfer.rs` (nine) and
    /// `nd_pipeline/src/hybrid.rs` (two).
    pub native_policy: PartAMfNativePolicyV1,
}

/// Eleven sealed policy tokens of the native MF lane. See the field doc on
/// [`PartAMfTransferControls::native_policy`] for why these are strings bound
/// by const assertions rather than enums.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PartAMfNativePolicyV1 {
    pub pair_proxy_model: &'static str,
    pub oxymoo_policy: &'static str,
    pub delta_v_anchor_policy: &'static str,
    pub polish_scope_policy: &'static str,
    pub sampling_mode: &'static str,
    pub target_propagation_authority: &'static str,
    pub front_output_mode: &'static str,
    pub local_optimizer: &'static str,
    pub local_optimizer_tune: &'static str,
    pub splitting_criterion: &'static str,
    pub split_alpha_policy: &'static str,
}

/// Const string equality, so a consumer can bind its implemented variant to a
/// sealed token at compile time: `const _: () = assert!(token_eq(a, b));`
/// makes an edited token on either side a BUILD failure.
#[must_use]
#[expect(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    reason = "const fn cannot use iterators; `i < a.len() == b.len()` bounds both \
              indexes, and the increment cannot overflow a length-bounded counter. \
              Every call site is a compile-time `const` assertion, so a panic here \
              would be a BUILD error, which is this function's entire purpose."
)]
pub const fn token_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// MF Stage 2 release-covariance and GMM-split recipe.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PartAMfLoweringControls {
    pub dust_pos_sigma_m: f64,
    pub dust_pos_sigma_radial_cross_track_m: f64,
    pub dust_vel_sigma_mps: f64,
    pub dust_vel_sigma_radial_cross_track_mps: f64,
    pub split_rank: usize,
    pub gmm_components: usize,
    pub split_tof_short_s: f64,
    pub split_alpha_scale_cov: f64,
    pub split_alpha_scale_cov_low: f64,
    pub split_jitter: f64,
    pub split_psd_tol: f64,
    pub split_max_psd_iter: usize,
    pub split_scale_decay: f64,
    pub split_default_alpha_fraction: f64,
    pub dust_phase_tof_s: f64,
}

/// Strict Hybrid physics and postprocess controls. `false`/zero/`None`-like
/// values are explicit because they are effective runtime physics inputs too.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "immutable compiled authority preserves one sealed field layout"
)]
pub struct PartAHybridControls {
    pub use_high_fidelity: bool,
    pub require_hf_transfer_correction: bool,
    pub target_corroboration_position_km: f64,
    pub target_corroboration_velocity_km_s: f64,
    pub session_max_time_s: f64,
    pub event_rewind_days: f64,
    pub session_tof_penalty_weight: f64,
    pub gravity_order: usize,
    pub force_drag: bool,
    pub force_srp: bool,
    pub force_sun: bool,
    pub force_moon: bool,
    pub atmosphere_model: i32,
    pub dust_am_ratio: f64,
    pub dust_cd: f64,
    pub dust_cr: f64,
    pub dt_max_s: f64,
    pub tolerance: f64,
    pub integrator_method: &'static str,
    pub fix_ls_max_nfev: usize,
    pub fix_ls_tol: f64,
    pub fix_ls_skip_tol: f64,
    pub dust_intercept_tol_km: f64,
    pub max_physical_dv_kms: f64,
    pub mf_seed_bound_kms: f64,
    pub hf_refine_bound_kms: f64,
    pub mf_seed_reg_weight: f64,
    pub hf_refine_reg_weight: f64,
    pub mf_seed_max_bound_expansions: usize,
    pub hf_refine_max_bound_expansions: usize,
    pub hybrid_mf_seed_hf_refine: bool,
    pub canister_am: f64,
    pub canister_cd: f64,
    pub canister_cr: f64,
    pub transfer_am_ratio: f64,
    pub transfer_cd: f64,
    pub transfer_cr: f64,
    pub min_practical_dust_mass_kg: f64,
}

/// Strict-HF runtime bindings not already owned by [`PartAHybridControls`].
///
/// Numeric force inputs remain singular in `hybrid`; this value independently
/// seals frame identity plus policies needed to turn those inputs into one
/// canonical production `ForceConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PartAStrictHfRuntimeAuthority {
    frame_authority_sha256: [u8; 32],
    target_propagation_authority: &'static str,
    subtract_first_order: bool,
    dynamic_sun_ephemeris: bool,
    dynamic_moon_ephemeris: bool,
}

impl PartAStrictHfRuntimeAuthority {
    #[must_use]
    pub const fn frame_authority_sha256(&self) -> [u8; 32] {
        self.frame_authority_sha256
    }

    #[must_use]
    pub const fn target_propagation_authority(&self) -> &'static str {
        self.target_propagation_authority
    }

    #[must_use]
    pub const fn subtract_first_order(&self) -> bool {
        self.subtract_first_order
    }

    #[must_use]
    pub const fn dynamic_sun_ephemeris(&self) -> bool {
        self.dynamic_sun_ephemeris
    }

    #[must_use]
    pub const fn dynamic_moon_ephemeris(&self) -> bool {
        self.dynamic_moon_ephemeris
    }
}

/// Native strict-HF lowering controls not represented by transfer config.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PartANativeHybridControls {
    /// The UKF sigma set the campaign flies, sealed as data.
    ///
    /// This replaced the `ukf_alpha`/`ukf_beta`/`ukf_kappa` triple on
    /// 2026-08-09 when the Julier minimal-skew simplex at `W0 = 0` landed. The
    /// triple was the Van der Merwe SCALING tuning; a simplex has no centre
    /// point for `beta` to inflate and no `lambda` to scale by, so all three
    /// became inert. Leaving them would have sealed three numbers that no
    /// longer described anything, and deleting them without a replacement would
    /// have left a worse hole: the compiled sigma set would have flipped with
    /// ZERO digest movement, which is the same defect stop-time review caught
    /// on the MF policy tokens (see
    /// [`PartAMfTransferControls::native_policy`]).
    ///
    /// So the set is a token, in the hash, and `nd_pipeline` binds it to the
    /// implemented generator with a `const` assertion via [`token_eq`] — an
    /// edited token on EITHER side is a BUILD failure, not a silent reseal.
    /// The compiled generator is `dust_ukf_rs::get_sigmas_ukf` and it publishes
    /// its own name as `dust_ukf_rs::SIGMA_SET_TOKEN`.
    pub ukf_sigma_set: &'static str,
    /// Post-impact target-body dynamics used by every strict-HF mass trial.
    ///
    /// The implementation token is compile-time bound in `nd_pipeline`; this
    /// field keeps a change to retained area or collision mechanics inside the
    /// serialized Part A science identity.
    pub retained_mass_dynamics: &'static str,
    /// Deterministic-mass root policy used by canonical MF/J2 search and
    /// strict-HF validation routes.
    ///
    /// This token seals immutable raw solve/evidence identity, a separate
    /// commanded mass equal to `max(raw, compiled floor)`, the one-sided safe
    /// bracket endpoint, and bounded nonfinite recovery into the serialized
    /// science identity.
    pub deterministic_mass_numerical_policy: &'static str,
    pub grain_mass_kg: f64,
    pub cov_min_eig: f64,
    pub cov_max_eig: f64,
    pub small_area_eta_max: f64,
    pub dust_hard_limit_kg: f64,
    pub min_practical_deterministic_mass_kg: f64,
    pub deterministic_mass_xtol_kg: f64,
    pub deterministic_mass_rtol: f64,
    pub deterministic_mass_maxiter: usize,
    pub deterministic_mass_max_kg: f64,
    /// UNPROVENANCE SEAL (audit 2026-08-16): 1.0 km miss goal traces to no
    /// requirement document; deterministic mass is ~linear in it, so a 5-10 km
    /// operational screening threshold moves the mass floor ~5-10x.
    pub deterministic_mass_min_distance_km: f64,
    pub tof_fractions: [f64; 2],
    pub hf_mass_rows_per_batch: usize,
}

/// Exact compiled scenario consumed by both MF and Hybrid shared-target mass.
///
/// The mass it seals is a CONTACT bound (>=1 grain reaches the target at
/// `target_hit_probability` confidence), not a deflection/momentum-delivery
/// bound. Assumption sensitivity, most load-bearing first:
///
/// 1. [`Self::target_position_sigma_m`] — knife edge (~1e5x mass per 50 m;
///    contact confidence unreachable at any mass by 200 m).
/// 2. [`Self::packet_correlation_grains`] — LINEAR in released mass,
///    unverified by any measurement.
/// 3. [`Self::momentum_coupling_kappa`] — inert for the contact bound.
///
/// Field order and the `assumption_id` string are sealed by the compact JSON
/// science digest, so neither
/// can be reordered to match; this doc is the sensitivity order.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PartASharedTargetControls {
    /// Stable scenario identifier carried into receipts and evidence.
    ///
    /// SEALED STRING: its bytes are hashed into the golden science SHA-256
    /// and it is streamed verbatim into H64 evidence receipts, so it cannot
    /// be rewritten. Its lexical order (`kappa1` first, sigma in the middle)
    /// is historical and inverted relative to the true sensitivity order in
    /// the struct doc above: the 100 m sigma is the knife edge, the one-grain
    /// packet correlation is linear and unverified, and kappa is inert.
    pub assumption_id: &'static str,
    /// Assumed, not observed, one-sigma target position uncertainty.
    ///
    /// KNIFE EDGE of the whole contact bound: the released mass moves ~1e5x
    /// per 50 m of sigma, and at 200 m the required contact confidence is
    /// unreachable at any mass.
    pub target_position_sigma_m: f64,
    pub target_position_treatment: PartASharedTargetPositionTreatment,
    /// Inert for the contact bound (moves nothing in it); binds the
    /// deterministic-mass evidence identity only.
    pub momentum_coupling_kappa: f64,
    /// Grains per independent packet. LINEAR in the released contact mass and
    /// unverified by any measurement; production seals 1.
    pub packet_correlation_grains: u64,
    pub target_hit_probability: f64,
    /// Live release-Pc/postprocess disk quadrature. The singular shared-target
    /// C12/log-min solver does not consume these counts.
    pub disk_radial_samples: usize,
    /// Angular half of the live release-Pc/postprocess disk quadrature.
    pub disk_angular_samples: usize,
    pub target_integration: PartASharedTargetDrawIntegration,
    /// Dense target-draw count: Cartesian sharp-axis nodes normally, polar
    /// radial nodes for near-isotropic recovery.
    pub target_radial_samples: usize,
    /// Cartesian slow-axis count and base polar angular count. The named polar
    /// recovery method doubles it for a full revolution; that multiplier is
    /// part of the method identity, not a runtime knob.
    pub target_angular_samples: usize,
    pub convergence_tolerance: f64,
    pub claim: PartASharedTargetClaim,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PartAReportingControls {
    pub physical_hv_reference: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PartAReferenceEvidence {
    pub event_authority_manifest_sha256: &'static str,
    pub event_source_catalogue_sha256: &'static str,
    pub event_source_manifest_sha256: &'static str,
    pub event_exact_wrapper_sha256: &'static str,
    pub event_kernel_source_sha256: &'static str,
    pub gravity_coefficient_sha256: &'static str,
    pub gravity_manifest_sha256: &'static str,
}

/// Frozen holdout semantics. Chunk widths intentionally absent: existing tests
/// prove them output-invariant execution bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PartAH64Controls {
    pub event_count: usize,
    pub source_slice_start: usize,
    pub source_slice_end: usize,
    pub strict_canonical: bool,
    pub required_for_reporting: bool,
}

/// Migration epochs in one sealed K3 campaign.
///
/// This is the single spelling of the count: [`PartAK3Controls::barriers`] is
/// declared with it, so every consumer that binds its own barrier-array width
/// to this constant is checked by the compiler rather than by agreement.
pub const PART_A_K3_BARRIER_COUNT: usize = 4;

/// Sealed first stage of the adaptive event ladder, `X` in `X + Y*n`.
///
/// Named rather than written inline because a second sealed number is DERIVED
/// from it: the K3 all-view union floor is three minimal island prefixes. That
/// floor sat at 72 while this was 24 and silently became unsatisfiable the
/// moment this moved, so the two are now bound at const-eval instead of by
/// comment.
const ADAPTIVE_INITIAL_EVENTS: usize = 8;
const ADAPTIVE_EVENT_STEP: usize = 4;
const ADAPTIVE_STAGE_COUNT: usize = (500 - ADAPTIVE_INITIAL_EVENTS) / ADAPTIVE_EVENT_STEP + 1;

/// Number of K3 all-view rotations (`event_rotations`), and the multiplier on
/// [`ADAPTIVE_INITIAL_EVENTS`] that gives the sealed all-view union floor.
const K3_FINAL_VIEW_COUNT: usize = 3;

/// K3 campaign controls shared by semantic config, cohort construction, and
/// final all-view replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PartAK3Controls {
    pub seeds: [u64; 3],
    pub islands: usize,
    pub rotations: [usize; 3],
    pub barriers: [u32; PART_A_K3_BARRIER_COUNT],
    pub exact36_generations: u32,
    /// Bounded runtime-measurement horizon for the exact36 scope: a
    /// projection lane runs this many generations of the exact campaign
    /// workload and projects the full wall from the measured rate. The three
    /// measurement horizons are pairwise distinct AND distinct from every
    /// campaign depth, because generation depth is what disambiguates the MF
    /// sensitivity scopes.
    pub exact36_measurement_generations: u32,
    pub mf18g500_sensitivity_generations: u32,
    pub mf18g500_sensitivity_measurement_generations: u32,
    pub mf18g1000_sensitivity_generations: u32,
    pub mf18g1000_sensitivity_measurement_generations: u32,
    pub intersect108_generations: u32,
    pub archive_max_size: usize,
    pub k3_final_view_count: usize,
    pub k3_final_min_union_events: usize,
}

/// One immutable compiled Part A scientific authority. Fields stay private so
/// no caller can supply a near-equivalent science profile to canonical paths.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CompiledPartAScienceV1 {
    schema_version: u32,
    hybrid_search_model: PartASearchModel,
    balanced_oa8_bank_indices: [[usize; 8]; 3],
    constellation: PartAConstellationControls,
    event_anchor_authority: PartAEventAnchorAuthority,
    gravity_authority: PartAGravityAuthority,
    mf: PartAMfControls,
    mf_transfer: PartAMfTransferControls,
    mf_lowering: PartAMfLoweringControls,
    hybrid: PartAHybridControls,
    strict_hf_runtime: PartAStrictHfRuntimeAuthority,
    native_hybrid: PartANativeHybridControls,
    shared_target: PartASharedTargetControls,
    reporting: PartAReportingControls,
    reference_evidence: PartAReferenceEvidence,
    h64: PartAH64Controls,
    k3: PartAK3Controls,
}

const PART_A_V1: CompiledPartAScienceV1 = CompiledPartAScienceV1 {
    schema_version: PART_A_SCIENCE_SCHEMA_VERSION,
    hybrid_search_model: PartASearchModel::MfJ2,
    balanced_oa8_bank_indices: [
        [215, 30, 367, 394, 226, 134, 479, 258],
        [187, 251, 292, 343, 168, 66, 356, 325],
        [98, 127, 261, 497, 121, 54, 376, 432],
    ],
    constellation: PartAConstellationControls {
        min_separation_km: 1.0,
    },
    event_anchor_authority: PartAEventAnchorAuthority::Verified(PartAVerifiedEventAnchor {
        source_frame: "GCRS geocentric via astropy 7.2.0 TEME->ITRS->CIRS->GCRS; SGP4-successful source states only; the SGP4-failure equinoctial fallback and zero-vector paths are recorded as an unfalsifiable per-anchor uniformity caveat",
        time_scale: "UTC (JD input, astropy Time scale=utc)",
        realization: "IAU 2006/2000A CIO-based via ERFA 2.0.1 (pyerfa 2.0.1.5): gmst82(UT1)+pom00(xp,yp,0); transpose(pom00(xp,yp,sp00(TT))+era00(UT1)); transpose(c2i06a(TT)); finite-difference velocity transforms",
        leap_second_table_sha256:
            "6f7bc6a25841bc394f82bdfd5d7bb22ffcd4548ee28e9822f2927a909e4f912f",
        leap_second_table_span: "IERS Leap_Second.dat through Bulletin C 71, expires 2026-12-28; last leap 2017-01-01; TAI-UTC 37 s across the 2021-2022 window; provenance-only, ERFA eraDat is the executing table",
        tai_minus_utc_source: "ERFA eraDat via pyerfa 2.0.1.5",
        tt_minus_tai_nanoseconds: 32_184_000_000,
        earth_orientation: PartAEarthOrientationConvention::IersFinals2000ADefinitive,
        reference_epoch_tai: PartATaiEpoch {
            seconds_since_1958_01_01: 2_031_141_337,
            nanosecond: 0,
        },
        manifest_sha256: "a579175193922652044d26be8ecc86f49b0032c092133a95f05fd06d088a3897",
    }),
    gravity_authority: PartAGravityAuthority::Verified(PartAVerifiedGravity {
        source_model: "GO_CONS_EGM_GOC_2__20091009T000000_20131020T235959_0201",
        normalization: "fully_normalized",
        tide_system: "tide_free",
        source_gm_km3_s2: 398_600.441_5,
        source_reference_radius_km: 6_378.136_46,
        source_max_degree: 300,
        source_max_order: 300,
        stored_degree: 15,
        stored_order: 15,
        runtime_degree: 5,
        runtime_order: 5,
        coefficient_sha256: "983f035818399f9cb27f1e8c604cb62b3e72d650aa4cbfadb31b1e7c4fe61f09",
        manifest_sha256: "681bfabf3b43e342f9e7d6b3dfd49a1e05298e8059c55645ce3001fd0c9cae33",
    }),
    mf: PartAMfControls {
        solver_lookback_s: 194_400.0,
        intercept_offset_s: -21_600.0,
        event_window_margin_days: 0.05,
        event_sample_seed: 41_127_203,
        b500_event_count: 500,
        event_rotations: [0, 166, 333],
        adaptive_initial_events: ADAPTIVE_INITIAL_EVENTS,
        adaptive_event_step: ADAPTIVE_EVENT_STEP,
        adaptive_stage_count: ADAPTIVE_STAGE_COUNT,
        prior_alpha: 1.0,
        prior_beta: 1.0,
        quantile: 0.9,
        confidence: 0.95,
        credible_interval_method: PartACredibleIntervalMethod::Hpd,
        success_rate_estimator: PartASuccessEstimator::PosteriorMean,
        objective_aggregation: PartAObjectiveAggregation::Max,
        hpd_grid_size: 401,
        hard_fail_success_threshold: 0.5,
        dv_log_floor_km_s: 1.0e-4,
        mass_log_floor_kg: 1.0e-3,
        constellation_size_penalty_alpha: 0.10,
        hard_dv_km_s: 2.5,
        hard_mass_kg: 1000.0,
        max_mass_kg: 1000.0,
        coverage_front_min_band: PartACoverageFrontMinBand::Qualified,
        convergence_pdf_threshold: 4.0,
        stop_band_confidence: 0.95,
    },
    mf_transfer: PartAMfTransferControls {
        max_time_s: 172_800.0,
        max_phase_dv: 1.25,
        max_transfer_dv: 1.25,
        max_revs: 4,
        min_perigee_km: 6578.137,
        max_apogee_km: 41378.137,
        search_pairs_to_verify: 8,
        tof_sample_budget: 256,
        coarse_early_stop: false,
        fine_total_limit: 10,
        coarse_reject_margin_km_s: 0.15,
        seed_fine_margin_km_s: 0.15,
        distance_tol_km: 0.025,
        deployer_min_distance_km: 0.12,
        tof_penalty_weight: 0.1,
        revolution_cap: 2.0,
        require_high_fidelity: false,
        force_config_enabled: false,
        j2_max_iterations: 5,
        j2_endpoint_target_km: 0.01,
        j2_correction_step_gain: 1.0,
        gravity_coefficients_enabled: false,
        local_optimizer_seed: 42,
        warm_start_enabled: false,
        // OFF, and this is a consistency requirement rather than a retreat from
        // the physics. See the long note on the field declaration.
        target_mean_element_conversion_enabled: false,
        native_policy: PartAMfNativePolicyV1 {
            pair_proxy_model: "sum",
            oxymoo_policy: "fast-population20-generations3-initial-best1",
            delta_v_anchor_policy: "full",
            polish_scope_policy: "nd-epsilon-membership",
            sampling_mode: "fast",
            target_propagation_authority: "mf-j2",
            front_output_mode: "verified-superset",
            local_optimizer: "nelder-mead",
            local_optimizer_tune: "aggressive",
            splitting_criterion: "maxvar",
            split_alpha_policy: "scaled",
        },
    },
    mf_lowering: PartAMfLoweringControls {
        dust_pos_sigma_m: 2.5,
        dust_pos_sigma_radial_cross_track_m: 1.25,
        dust_vel_sigma_mps: 0.082_724_851_417_135_86,
        dust_vel_sigma_radial_cross_track_mps: 0.040_894_506_197_059_824,
        split_rank: 1,
        // 3 -> 1, 2026-08-03. The split is indistinguishable from no split under
        // strict-HF dynamics once the integrator's own truncation error is
        // controlled for (converged gap 2.19e-9 relative, 1.5e5x below the 25 m
        // corroboration threshold); dropping it removes two thirds of the UKF
        // sigma lane's propagations for a 1.2010x whole-cell RHS-count win. Full
        // evidence: docs/plans/2026-07-31-part-a-200-generation-fast-hybrid.md,
        // "2026-08-03 GMM K=1 evidence under strict HF".
        gmm_components: 1,
        split_tof_short_s: 7200.0,
        split_alpha_scale_cov: 0.6,
        split_alpha_scale_cov_low: 0.35,
        split_jitter: 0.0,
        split_psd_tol: 1.0e-12,
        split_max_psd_iter: 10,
        split_scale_decay: 0.7,
        split_default_alpha_fraction: 0.6,
        dust_phase_tof_s: 7200.0,
    },
    hybrid: PartAHybridControls {
        use_high_fidelity: true,
        require_hf_transfer_correction: true,
        target_corroboration_position_km: 0.025,
        target_corroboration_velocity_km_s: 2.0e-5,
        session_max_time_s: 0.0,
        event_rewind_days: 3.0,
        session_tof_penalty_weight: 0.0,
        gravity_order: 5,
        force_drag: true,
        force_srp: true,
        force_sun: true,
        force_moon: true,
        // Serialized into the compact science digest, so moving this integer invalidates
        // every sealed receipt -- a seal move, not a tuning knob. It is also
        // the ONLY thing in that hash describing the atmosphere: the quadrature
        // plans live in `jb_rs` and are hashed by nothing, which is why each
        // profile gets its own code instead of being redefined in place.
        //
        // Rationale, bound and measured effect are recorded once, beside the
        // mirror in `StrictHfForceAuthority::PART_A`
        // (two_phase_transfer_rs/src/postprocess/session.rs). `nd_pipeline`
        // fails CLOSED if the two disagree, so they move together.
        atmosphere_model: 8,
        dust_am_ratio: 1.948,
        dust_cd: 2.2,
        dust_cr: 1.3,
        dt_max_s: 300.0,
        tolerance: 1.0e-8,
        // Vern7 since 2026-08-09 (R26); Vern9 before that. Hashed at
        // Serialized into the compact science digest, so moving this token invalidates every sealed
        // receipt -- it is a seal move, not a tuning knob.
        //
        // Measured over 12 production-shaped arcs at the FLOWN atmosphere
        // model, sealed eps, common Vern9@1e-12 anchor, by
        // `lightyear_odeint_rs/tests/stepper_method_ab.rs` (re-runnable; do not
        // re-decide this from a single arc, which scatters two orders).
        //
        // Re-measured 2026-08-11 at era `c9c9f7d`. Quote the TOTALS, not the
        // ratio: the ratio has been written down honestly in this repo as
        // -20.0%, -13.9%, -15.6% and -15.29% on four different corpora, and it
        // is the fragile half of the claim.
        //
        //   vern9  89,320 evals over 5,449 steps
        //   vern7  75,661 evals over 7,482 steps    -> -15.29% evaluations
        //
        // Vern7 takes MORE and CHEAPER steps, so the wall saving (-10.70%) is
        // smaller than the evaluation saving. The numbers this comment used to
        // carry (7,350.7 and 6,326.5 per-draw means) were taken before the
        // corpus was found to be flying one epoch for all twelve draws; they
        // were honest and they are superseded.
        //
        // THE DECISION IS NOT A TWO-POINT COMPARISON ANY MORE. Every explicit
        // arm the tree can fly is now priced against the same anchor by
        // `every_explicit_arm_is_priced_at_the_sealed_eps`, and Vern7 is the
        // COST MINIMUM of the whole set -- dopri5 +21.54%, tsit5 +0.81%,
        // dop853 +30.69%, rkv98 +87.93%, vern9 +18.05%. Cost is not monotone in
        // order, so walking down from Vern9 and stopping was never by itself
        // evidence that Vern7 was the bottom. That test now asserts on it.
        //
        // NO ACCURACY CLAIM IS MADE HERE, and the ones this comment used to
        // make are withdrawn. `lightyear_odeint_rs/examples/r43_corpus_floor.rs`
        // showed the
        // Vern7-minus-Vern9 RMS gap STRADDLES ZERO under a physics-neutral ULP
        // perturbation, so this corpus prices COST and cannot rank ACCURACY.
        // In particular the old "at 1e-10 Vern7 is BOTH cheaper and more
        // accurate" line is dead: on the current ladder that same column reads
        // 0.0343 m against Vern9's 0.0179 m, and neither reading is worth
        // anything. What flattens near 1e-9 is the ANCHOR, not either arm.
        //
        // THE TOLERANCE KNOB IS STILL NOT DEAD, but on the cost axis only:
        // Vern7 is cheaper than Vern9 at every rung of 1e-7..1e-10 (5,154.6 /
        // 6,305.1 / 8,111.2 / 11,009.8 against 7,131.6 / 7,443.3 / 9,141.3 /
        // 12,255.8 mean evaluations), by 27.7% at the loose end and 10.2% at
        // the tight one. So a future sub-0.1 m requirement is bought by
        // tightening `tolerance` rather than by reverting to Vern9 -- but if
        // that requirement ever arrives, rank the two arms with an instrument
        // built to rank accuracy, not with this corpus.
        integrator_method: "vern7",
        fix_ls_max_nfev: 100,
        fix_ls_tol: 1.0e-5,
        fix_ls_skip_tol: 1.0,
        dust_intercept_tol_km: 0.01,
        max_physical_dv_kms: 7.5,
        mf_seed_bound_kms: 0.1,
        hf_refine_bound_kms: 1.0,
        mf_seed_reg_weight: 1.0e-3,
        hf_refine_reg_weight: 1.0e-4,
        mf_seed_max_bound_expansions: 3,
        hf_refine_max_bound_expansions: 7,
        hybrid_mf_seed_hf_refine: true,
        canister_am: 0.01,
        canister_cd: 2.2,
        canister_cr: 1.3,
        transfer_am_ratio: 0.01,
        transfer_cd: 2.2,
        transfer_cr: 1.3,
        min_practical_dust_mass_kg: 0.01,
    },
    strict_hf_runtime: PartAStrictHfRuntimeAuthority {
        // Independent seal of `satpy_core::frame_time::frame_authority()`.
        // Generated from its immutable EOP/leap/grid/version bytes; never read
        // back from the live singleton when expected authority is constructed.
        frame_authority_sha256: [
            0xc6, 0x26, 0x1e, 0xc7, 0x0a, 0x03, 0x75, 0x31, 0x9f, 0x9c, 0x2b, 0x1a, 0xe1, 0x2f,
            0xf3, 0x31, 0x6d, 0xb3, 0x0e, 0x9f, 0x90, 0x46, 0xb0, 0xe3, 0x4f, 0xf3, 0xf4, 0xe6,
            0x9c, 0xb8, 0x60, 0x69,
        ],
        target_propagation_authority: "strict-hf-v3-fixed-ic",
        subtract_first_order: true,
        dynamic_sun_ephemeris: true,
        dynamic_moon_ephemeris: true,
    },
    native_hybrid: PartANativeHybridControls {
        // Merwe-13 -> Julier-7 (2026-08-09). The simplex propagates 7 arcs per
        // sigma row against the Merwe set's 12 (13 points, one of which the R18
        // endpoint reuse already elided), and pays for it with a loss of
        // third-degree exactness that no choice of `W0` recovers. Derivation
        // and the measured trade: `dust_ukf_rs::NUM_SIGMA` and
        // `docs/plans/2026-08-05-hf-hybrid-speedup-audit.md` §15b/§15c.
        ukf_sigma_set: "julier7-w0-zero",
        retained_mass_dynamics: "perfectly-inelastic-fixed-area-retention-v1",
        deterministic_mass_numerical_policy: "practical-floor-safe-bracket-v1",
        grain_mass_kg: 6.450_736_915_371_043e-10,
        cov_min_eig: 1.0e-6,
        cov_max_eig: 1.0e6,
        small_area_eta_max: 1.0e-2,
        dust_hard_limit_kg: 1000.0,
        min_practical_deterministic_mass_kg: 5.0e-7,
        deterministic_mass_xtol_kg: 1.0e-6,
        // 1e-6 -> 1e-5 (2026-08-06). Kept in step with
        // `dust_estimates_rs::mass_solver::SolverConfig::default`, which
        // carries the derivation: `rtol` is an absolute kilometre tolerance on
        // this corpus, and 1e-6 km is one millimetre of miss distance under a
        // trajectory accurate to 1.2 cm.
        deterministic_mass_rtol: 1.0e-5,
        deterministic_mass_maxiter: 50,
        deterministic_mass_max_kg: 1000.0,
        deterministic_mass_min_distance_km: 1.0,
        tof_fractions: [0.0, 0.95],
        hf_mass_rows_per_batch: 4096,
    },
    shared_target: PartASharedTargetControls {
        // SEALED bytes (hashed + streamed into receipts); cannot be reworded.
        // Read it in sensitivity order, not lexical order: `100m` (target
        // position sigma) is the knife edge, `one-grain` (packet correlation)
        // is linear and unverified, `kappa1` is inert. The mass it labels is
        // a >=1-grain CONTACT bound, not a deflection bound.
        assumption_id: "part-a-v3-kappa1-one-grain-100m-optimistic-model-conditioned",
        target_position_sigma_m: 100.0,
        target_position_treatment:
            PartASharedTargetPositionTreatment::AssumedNotObservedIsotropicEncounterBplaneSharedDraw,
        momentum_coupling_kappa: 1.0,
        packet_correlation_grains: 1,
        target_hit_probability: 0.99,
        // 24x100 -> 16x64 (2026-08-18). This is the separate release-Pc/
        // postprocess disk quadrature; it neither selects nor backstops the
        // singular shared-target C12/log-min method.
        //
        // It runs once per target node, so it multiplies the release-Pc path's
        // cost. Across the production corpus, where the narrow grain sigma is
        // 32-64 m, the 1.25 m disk is at most a fortieth of a sigma wide and
        // the Gaussian is nearly flat across it: the fine-vs-half-disk gate
        // reads 4.4e-8 at 24x100 and 9.8e-8 at 16x64, against a 1e-4 tolerance.
        //
        // The binding case is `cov_min_eig`, which clamps the grain sigma at
        // 1 m, where the 1.25 m disk is wider than the sigma and the Gaussian
        // varies steeply. Measured there, with target sigma scaled to keep the
        // row reachable:
        //
        //     24x100 -> 2.4e-5    16x64 -> 5.4e-5
        //     12x48  -> 9.5e-5    8x32  -> 2.1e-4 (fails)
        //
        // 16x64 is the smallest grid that clears that gate with real margin
        // (1.9x); 12x48 clears it by only 1.05x and 8x32 fails. This measured
        // release-Pc constraint is independent of the singular C12 domain.
        disk_radial_samples: 16,
        disk_angular_samples: 64,
        target_integration:
            PartASharedTargetDrawIntegration::RotatedCartesianWithBoundedRadialRefinementAndPolarBelow2V2,
        // 64x64 Gauss-Hermite -> 384x32 composite, rotated (2026-08-18), then
        // 192x32 after the bounded fourth-order disk cutover (2026-08-24).
        //
        // On the normal Cartesian route these are per-axis node counts in the
        // whitened B-plane frame, rotated so axis 0 varies fastest. On the
        // near-isotropic recovery route they are radial/angular polar counts.
        //
        // Gauss-Hermite fixed the tail truncation of the original polar rule
        // but is exact only for polynomials, and this integrand is a smoothed
        // step: `(1-p)^N` drops from one to zero across a layer roughly a
        // tenth of a target sigma wide. Gauss nodes approximate that worst.
        // Across the 25-row production corpus, 64x64 vs 32x32 failed on 9
        // rows, worst delta 1.607e-3 -- 16x over tolerance. Raising the Gauss
        // order does not fix it: on the worst row the error still oscillates
        // at 2.3e-4 with 256 nodes per axis (65536 nodes).
        //
        // An equally spaced rule converges geometrically on the whole line, so
        // once the spacing resolves the layer the error collapses: on that row
        // 192 -> 4.6e-6, 384 -> 2.3e-8, 512 -> 1.0e-9. Rotating first means
        // only the sharp axis needs the dense count. The recovery probe found
        // 128x8 fails the anchor at 5.607e-4. 192x8 clears that anchor but fails
        // the next live Walker row at 1.243e-4. 192x16 then failed one retained
        // event at anisotropy 1.341. The same geometry terminated live job
        // 7259712 after 10,311 s, disproving the presumed 7.6--49 corpus
        // envelope. Schema 6 bound a normalized six-sigma polar recovery for
        // that geometry; its omitted normal mass is below 1.6e-8, but its
        // 192x16 Cartesian arm failed later live job 7264366 at 1.135e-4.
        // Schema 7 restores the common 192x32 grid and binds one bounded
        // sharp-axis refinement to 384x32 when the base pair is mixed or
        // nonconverged. No second refinement or tolerance change is allowed.
        // The exact method and counts below are work-ratcheted.
        target_radial_samples: 192,
        target_angular_samples: 32,
        convergence_tolerance: 1.0e-4,
        claim: PartASharedTargetClaim::ModelConditionedConservativeContactRequirement,
    },
    reporting: PartAReportingControls {
        physical_hv_reference: [2.5, 1000.0],
    },
    reference_evidence: PartAReferenceEvidence {
        event_authority_manifest_sha256:
            "a579175193922652044d26be8ecc86f49b0032c092133a95f05fd06d088a3897",
        event_source_catalogue_sha256:
            "1d6e4e86b64064c2d476b0a8bcad13ae783179a1420fec7eae3fca4ebbb118f8",
        event_source_manifest_sha256:
            "9d846ceec76f7cb74279a14fcd44eac4303016d5469ef764b09708a5e0df4477",
        event_exact_wrapper_sha256:
            "78b7478db9c83974a74871e163b811fc10fc5cf55113e0e7e603066dd27827d9",
        event_kernel_source_sha256:
            "be051ba96900c9f61e2f99183b6b2f1cd3491e21d4adc6444d47426898d3faf7",
        gravity_coefficient_sha256:
            "983f035818399f9cb27f1e8c604cb62b3e72d650aa4cbfadb31b1e7c4fe61f09",
        gravity_manifest_sha256: "681bfabf3b43e342f9e7d6b3dfd49a1e05298e8059c55645ce3001fd0c9cae33",
    },
    h64: PartAH64Controls {
        event_count: 64,
        source_slice_start: 500,
        source_slice_end: 564,
        strict_canonical: true,
        required_for_reporting: true,
    },
    k3: PartAK3Controls {
        seeds: [41_127_203, 41_127_204, 41_127_205],
        islands: 3,
        rotations: [0, 166, 333],
        barriers: [100, 200, 300, 400],
        exact36_generations: 200,
        exact36_measurement_generations: 4,
        mf18g500_sensitivity_generations: 500,
        mf18g500_sensitivity_measurement_generations: 13,
        mf18g1000_sensitivity_generations: 1000,
        mf18g1000_sensitivity_measurement_generations: 14,
        intersect108_generations: 400,
        archive_max_size: 4096,
        k3_final_view_count: K3_FINAL_VIEW_COUNT,
        // Three minimal island prefixes, and nothing weaker: the compiled OA8
        // openings are pairwise disjoint, so the smallest union a genuine
        // three-view cohort can produce is 3 * `adaptive_initial_events`. It is
        // DERIVED here rather than written
        // out, because it was 72 while X was 24 and silently became wrong the
        // moment X moved: leaving a stale floor refuses an all-view claim from
        // exactly the cohorts the Beta stop decided FASTEST, which is backwards
        // -- decisiveness would cost a cohort its union. The const-eval below is
        // what stops the two shearing apart again.
        k3_final_min_union_events: K3_FINAL_VIEW_COUNT * ADAPTIVE_INITIAL_EVENTS,
    },
};

/// An adaptive event schedule `X + Y*n` was rejected before it could build a
/// plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartAAdaptiveEventsError {
    /// `initial` (X), `step` (Y) or `stages` was zero.
    NotPositive { field: &'static str },
    /// The last stage `X + Y*(stages-1)` exceeds the sealed B500 bank.
    ExceedsEventBank {
        last_stage: usize,
        bank_events: usize,
    },
    /// `X + Y*(stages-1)` overflowed `usize`.
    Overflow,
}

impl std::fmt::Display for PartAAdaptiveEventsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPositive { field } => {
                write!(formatter, "adaptive event schedule {field} must be positive")
            }
            Self::ExceedsEventBank {
                last_stage,
                bank_events,
            } => write!(
                formatter,
                "adaptive event schedule reaches {last_stage} events, beyond the sealed {bank_events}-event bank"
            ),
            Self::Overflow => {
                formatter.write_str("adaptive event schedule stage count overflows usize")
            }
        }
    }
}

impl std::error::Error for PartAAdaptiveEventsError {}

impl CompiledPartAScienceV1 {
    #[must_use]
    pub const fn part_a_v1() -> &'static Self {
        &PART_A_V1
    }

    /// Singular model used by every compiled Part A B500 search lane.
    #[must_use]
    pub const fn search_model(&self) -> PartASearchModel {
        self.hybrid_search_model
    }

    /// Exact B500 row identities for the three disjoint balanced OA8 openings.
    #[must_use]
    pub const fn balanced_oa8_bank_indices(&self) -> &[[usize; 8]; 3] {
        &self.balanced_oa8_bank_indices
    }

    /// The compiled authority with the adaptive schedule `X + Y*n` replaced.
    ///
    /// `initial` is X (events the first stage assesses for Beta convergence),
    /// `step` is Y (events each follow-up round adds), `stages` bounds the
    /// ladder. The Beta stop itself is NOT a parameter here and is not
    /// reachable from this constructor -- only how many events each round buys.
    ///
    /// This returns an OWNED authority, and that is the point: X, Y and
    /// `stages` are folded into [`sha256`](Self::sha256), so an overridden
    /// schedule carries a different science digest than
    /// [`part_a_v1`](Self::part_a_v1) and cannot be mistaken for it. Canonical
    /// Part A refuses any non-default schedule
    /// (`Config::validate_part_a_semantics`); the receipt digests
    /// (`nd_part_a_evidence`) read the compiled default directly, so a canonical
    /// run can never record a digest for a schedule it did not execute.
    ///
    /// # Errors
    ///
    /// Returns [`PartAAdaptiveEventsError`] when a component is zero or the
    /// final stage would run past the sealed B500 event bank.
    pub fn with_adaptive_events(
        initial: usize,
        step: usize,
        stages: usize,
    ) -> Result<Self, PartAAdaptiveEventsError> {
        for (field, value) in [("initial", initial), ("step", step), ("stages", stages)] {
            if value == 0 {
                return Err(PartAAdaptiveEventsError::NotPositive { field });
            }
        }
        let last_stage = stages
            .checked_sub(1)
            .and_then(|extra| step.checked_mul(extra))
            .and_then(|increase| initial.checked_add(increase))
            .ok_or(PartAAdaptiveEventsError::Overflow)?;
        if last_stage > PART_A_V1.mf.b500_event_count {
            return Err(PartAAdaptiveEventsError::ExceedsEventBank {
                last_stage,
                bank_events: PART_A_V1.mf.b500_event_count,
            });
        }
        let mut science = PART_A_V1;
        science.mf.adaptive_initial_events = initial;
        science.mf.adaptive_event_step = step;
        science.mf.adaptive_stage_count = stages;
        Ok(science)
    }

    #[must_use]
    pub const fn mf(&self) -> &PartAMfControls {
        &self.mf
    }

    #[must_use]
    pub const fn mf_transfer(&self) -> &PartAMfTransferControls {
        &self.mf_transfer
    }

    #[must_use]
    pub const fn mf_lowering(&self) -> &PartAMfLoweringControls {
        &self.mf_lowering
    }

    #[must_use]
    pub const fn hybrid(&self) -> &PartAHybridControls {
        &self.hybrid
    }

    #[must_use]
    pub const fn strict_hf_runtime(&self) -> &PartAStrictHfRuntimeAuthority {
        &self.strict_hf_runtime
    }

    #[must_use]
    pub const fn native_hybrid(&self) -> &PartANativeHybridControls {
        &self.native_hybrid
    }

    #[must_use]
    pub const fn shared_target(&self) -> &PartASharedTargetControls {
        &self.shared_target
    }

    #[must_use]
    pub const fn h64(&self) -> &PartAH64Controls {
        &self.h64
    }

    #[must_use]
    pub const fn k3(&self) -> &PartAK3Controls {
        &self.k3
    }

    #[must_use]
    pub const fn reference_evidence(&self) -> &PartAReferenceEvidence {
        &self.reference_evidence
    }

    #[must_use]
    pub const fn event_anchor_authority(&self) -> &PartAEventAnchorAuthority {
        &self.event_anchor_authority
    }

    #[must_use]
    pub const fn constellation(&self) -> &PartAConstellationControls {
        &self.constellation
    }

    /// Sealed reporting controls. Read-only view of already-hashed bytes
    /// (`sha256` folds `reporting.physical_hv_reference` at :1342), so exposing
    /// them cannot change `PART_A_SCIENCE_SCHEMA_VERSION` or the schema digest.
    #[must_use]
    pub const fn reporting(&self) -> &PartAReportingControls {
        &self.reporting
    }

    /// Validate all production Hybrid authority bindings.
    ///
    /// # Errors
    ///
    /// Returns [`PartAAuthorityError`] when any authority is unresolved,
    /// malformed, or inconsistent with compiled reference evidence.
    pub fn require_production_hybrid_authority(&self) -> Result<(), PartAAuthorityError> {
        self.constellation.validate()?;
        let event = self.event_anchor_authority.require_verified()?;
        if event.manifest_sha256 != self.reference_evidence.event_authority_manifest_sha256 {
            return Err(PartAAuthorityError::invalid(
                "event_anchor_authority",
                "manifest hash differs from compiled reference evidence",
            ));
        }
        let gravity = self.gravity_authority.require_verified()?;
        if gravity.runtime_degree != self.hybrid.gravity_order
            || gravity.runtime_order != self.hybrid.gravity_order
            || gravity.coefficient_sha256 != self.reference_evidence.gravity_coefficient_sha256
            || gravity.manifest_sha256 != self.reference_evidence.gravity_manifest_sha256
        {
            return Err(PartAAuthorityError::invalid(
                "gravity_authority",
                "model order or hashes differ from compiled Hybrid/reference evidence",
            ));
        }
        let runtime = self.strict_hf_runtime;
        if runtime.frame_authority_sha256 == [0; 32]
            || runtime.target_propagation_authority != "strict-hf-v3-fixed-ic"
            || !runtime.subtract_first_order
            || runtime.dynamic_sun_ephemeris != self.hybrid.force_sun
            || runtime.dynamic_moon_ephemeris != self.hybrid.force_moon
        {
            return Err(PartAAuthorityError::invalid(
                "strict_hf_runtime",
                "frame or runtime policy differs from compiled Hybrid authority",
            ));
        }
        if self.native_hybrid.retained_mass_dynamics
            != "perfectly-inelastic-fixed-area-retention-v1"
        {
            return Err(PartAAuthorityError::invalid(
                "retained_mass_dynamics",
                "post-impact target dynamics differ from compiled Hybrid authority",
            ));
        }
        if self.native_hybrid.deterministic_mass_numerical_policy
            != "practical-floor-safe-bracket-v1"
        {
            return Err(PartAAuthorityError::invalid(
                "deterministic_mass_numerical_policy",
                "deterministic-mass numerical policy differs from compiled Hybrid authority",
            ));
        }
        Ok(())
    }

    /// Return exact digest of compiled science bytes.
    ///
    /// Serialization failure returns an all-zero non-authority digest, so every
    /// sealed comparison fails closed without a production panic.
    #[must_use]
    pub fn sha256(&self) -> [u8; 32] {
        let Ok(serialized) = serde_json::to_vec(self) else {
            return [0; 32];
        };
        Sha256::digest(serialized).into()
    }

    /// The digest of [`Self::part_a_v1`], computed once per process.
    ///
    /// [`Self::sha256`] serializes the whole compiled authority to JSON before
    /// hashing it, and strict-HF enclosure issuance takes that digest twice for
    /// every RHS it builds -- roughly 2,016 RHS constructions per dense
    /// ephemeris arc, per object. `PART_A_V1` is a compiled constant, so the
    /// digest is a pure function of immutable bytes and caching it returns the
    /// same value `sha256` would have recomputed.
    ///
    /// Only the `PART_A_V1` instance is cached. Derived authorities
    /// ([`Self::part_a_v1_legacy_target_elements`] and the schedule overrides)
    /// hash different bytes and must keep calling [`Self::sha256`].
    #[must_use]
    pub fn part_a_v1_sha256() -> [u8; 32] {
        static DIGEST: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();
        *DIGEST.get_or_init(|| PART_A_V1.sha256())
    }

    #[must_use]
    pub fn sha256_hex(&self) -> String {
        hex(&self.sha256())
    }

    #[must_use]
    pub fn matches_sha256(&self, value: &str) -> bool {
        value == self.sha256_hex()
    }

    /// The compiled authority with the target mean-element correction DISABLED.
    ///
    /// Exists for one purpose: the captured Python oracle propagates catalogue
    /// targets as osculating elements through the secular-J2 lane, and the
    /// port-fidelity fixtures were captured from it. A test that wants to prove
    /// the Rust port still reproduces the oracle asks for the legacy convention
    /// explicitly rather than inheriting whatever production happens to hold.
    /// Never use this for science.
    ///
    /// # It is BYTE-IDENTICAL to production today, and that is not a bug
    ///
    /// Production also holds `target_mean_element_conversion_enabled = false`,
    /// for the reasons set out on the field itself, so this accessor currently
    /// returns the same authority. An earlier version of this doc said
    /// "production now corrects that", which was never true and made every
    /// reader expect a distinction that does not exist -- and any A/B that
    /// compared the two authorities was therefore comparing one authority with
    /// itself.
    ///
    /// The accessor is kept because its VALUE is pinning the convention, not
    /// differing from production: if production ever enables the conversion,
    /// the oracle fixtures must keep the legacy value rather than silently
    /// follow. `legacy_target_elements_is_pinned_not_inherited` asserts exactly
    /// that, in both directions.
    #[must_use]
    pub fn part_a_v1_legacy_target_elements() -> &'static Self {
        // A field-flip on the `PART_A_V1` const is itself a compile-time
        // constant; the OnceLock this replaced was a runtime lock around a
        // value the compiler already had.
        static LEGACY: CompiledPartAScienceV1 = {
            let mut science = PART_A_V1;
            science.mf_transfer.target_mean_element_conversion_enabled = false;
            science
        };
        &LEGACY
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        output.push(hex_digit(byte >> 4));
        output.push(hex_digit(byte & 0x0f));
    }
    output
}

const fn hex_digit(nibble: u8) -> char {
    match nibble {
        0 => '0',
        1 => '1',
        2 => '2',
        3 => '3',
        4 => '4',
        5 => '5',
        6 => '6',
        7 => '7',
        8 => '8',
        9 => '9',
        10 => 'a',
        11 => 'b',
        12 => 'c',
        13 => 'd',
        14 => 'e',
        15 => 'f',
        _ => '\0',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use common_rs::{require_err, require_ok};

    #[test]
    fn canonical_full_control_serialization_is_hash_bound() {
        let baseline = *CompiledPartAScienceV1::part_a_v1();
        let canonical = require_ok!(serde_json::to_vec(&baseline));
        assert!(canonical
            .windows(b"hard_fail_success_threshold".len())
            .any(|window| { window == b"hard_fail_success_threshold" }));
        assert!(canonical
            .windows(b"gmm_components".len())
            .any(|window| window == b"gmm_components"));

        // The sigma set is a token, so this poison also proves the STRING
        // reaches the digest — the failure mode a float field cannot exhibit.
        // `merwe13` is deliberately the name of the set this one replaced.
        let mut changed = baseline;
        changed.native_hybrid.ukf_sigma_set = "merwe13";
        assert_ne!(baseline.sha256(), changed.sha256());

        let mut changed = baseline;
        changed.mf.quantile = 0.8;
        assert_ne!(baseline.sha256(), changed.sha256());

        let mut changed = baseline;
        changed.mf.hard_fail_success_threshold = 0.4;
        assert_ne!(baseline.sha256(), changed.sha256());

        let mut changed = baseline;
        changed.mf_transfer.tof_penalty_weight = 0.2;
        assert_ne!(baseline.sha256(), changed.sha256());

        let mut changed = baseline;
        changed.mf_lowering.split_rank = 2;
        assert_ne!(baseline.sha256(), changed.sha256());

        let mut changed = baseline;
        changed.hybrid.session_max_time_s = 1.0;
        assert_ne!(baseline.sha256(), changed.sha256());

        let mut changed = baseline;
        changed.strict_hf_runtime.frame_authority_sha256[0] ^= 1;
        assert_ne!(baseline.sha256(), changed.sha256());

        let mut changed = baseline;
        changed.h64.event_count = 63;
        assert_ne!(baseline.sha256(), changed.sha256());

        let mut changed = baseline;
        let first_barrier = changed.k3.barriers.first_mut();
        assert!(
            first_barrier.is_some(),
            "compiled K3 barriers must be nonempty"
        );
        let Some(first_barrier) = first_barrier else {
            return;
        };
        *first_barrier = 99;
        assert_ne!(baseline.sha256(), changed.sha256());

        let mut changed = baseline;
        changed.constellation.min_separation_km = 2.0;
        assert_ne!(baseline.sha256(), changed.sha256());
    }

    fn syntactically_verified_science() -> Result<CompiledPartAScienceV1, PartAAuthorityError> {
        let mut science = *CompiledPartAScienceV1::part_a_v1();
        let evidence = science.reference_evidence;
        science.event_anchor_authority = PartAEventAnchorAuthority::Verified(
            PartAVerifiedEventAnchor::new(PartAVerifiedEventAnchorInput {
                source_frame: "frame",
                time_scale: "TAI",
                realization: "realization",
                leap_second_table_sha256:
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                leap_second_table_span: "1972-01-01..present",
                tai_minus_utc_source: "sealed leap-second table",
                tt_minus_tai_nanoseconds: 32_184_000_000,
                earth_orientation: PartAEarthOrientationConvention::IersFinals2000ADefinitive,
                reference_epoch_tai: PartATaiEpoch::new(0, 0)?,
                manifest_sha256: evidence.event_authority_manifest_sha256,
            })?,
        );
        science.gravity_authority = PartAGravityAuthority::Verified(PartAVerifiedGravity::new(
            PartAVerifiedGravityInput {
                source_model: "model",
                normalization: "fully_normalized",
                tide_system: "tide_free",
                source_gm_km3_s2: 398_600.441_8,
                source_reference_radius_km: 6_378.136_3,
                source_max_degree: 300,
                source_max_order: 300,
                stored_degree: 15,
                stored_order: 15,
                runtime_degree: science.hybrid.gravity_order,
                runtime_order: science.hybrid.gravity_order,
                coefficient_sha256: evidence.gravity_coefficient_sha256,
                manifest_sha256: evidence.gravity_manifest_sha256,
            },
        )?);
        Ok(science)
    }

    /// Every member of the production Hybrid gate must fail closed on its own.
    ///
    /// No member of `PART_A_V1` is unresolved today, so the CLI entry points
    /// that call `require_production_hybrid_authority` can no longer be driven
    /// into an unresolved state end to end. This test is the replacement
    /// proof: it mutates one member at a time and shows the composite gate
    /// still rejects, naming the offending member. It is a type-level
    /// statement about the gate, not an end-to-end statement about the CLI.
    #[test]
    fn production_gate_rejects_each_member_independently() {
        let mut unresolved_event = require_ok!(syntactically_verified_science());
        unresolved_event.event_anchor_authority = PartAEventAnchorAuthority::Unresolved;
        assert_eq!(
            require_err!(unresolved_event.require_production_hybrid_authority()).to_string(),
            "Part A production Hybrid authority unresolved: event_anchor_authority"
        );

        let mut unresolved_gravity = require_ok!(syntactically_verified_science());
        unresolved_gravity.gravity_authority = PartAGravityAuthority::Unresolved;
        assert_eq!(
            require_err!(unresolved_gravity.require_production_hybrid_authority()).to_string(),
            "Part A production Hybrid authority unresolved: gravity_authority"
        );

        // The constellation control replaced an authority enum, so its
        // fail-closed path is a validity check rather than an Unresolved
        // variant. It must still gate.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut invalid = require_ok!(syntactically_verified_science());
            invalid.constellation.min_separation_km = bad;
            assert!(
                require_err!(invalid.require_production_hybrid_authority())
                    .to_string()
                    .contains("constellation min separation"),
                "min_separation_km {bad} must fail the production gate"
            );
        }

        let mut invalid_retention = require_ok!(syntactically_verified_science());
        invalid_retention.native_hybrid.retained_mass_dynamics = "fixed-am-ratio";
        assert!(
            require_err!(invalid_retention.require_production_hybrid_authority())
                .to_string()
                .contains("retained_mass_dynamics")
        );

        let mut invalid_mass_policy = require_ok!(syntactically_verified_science());
        invalid_mass_policy
            .native_hybrid
            .deterministic_mass_numerical_policy = "legacy-endpoint-v0";
        assert!(
            require_err!(invalid_mass_policy.require_production_hybrid_authority())
                .to_string()
                .contains("deterministic_mass_numerical_policy")
        );

        // The unmutated value passes, so each rejection above is attributable
        // to the mutation and not to the fixture.
        assert!(require_ok!(syntactically_verified_science())
            .require_production_hybrid_authority()
            .is_ok());
    }

    #[test]
    fn production_gate_cross_binds_verified_payloads_to_compiled_evidence() {
        let mut event_mismatch = require_ok!(syntactically_verified_science());
        event_mismatch.event_anchor_authority = PartAEventAnchorAuthority::Verified(require_ok!(
            PartAVerifiedEventAnchor::new(PartAVerifiedEventAnchorInput {
                source_frame: "frame",
                time_scale: "TAI",
                realization: "realization",
                leap_second_table_sha256:
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                leap_second_table_span: "1972-01-01..present",
                tai_minus_utc_source: "sealed leap-second table",
                tt_minus_tai_nanoseconds: 32_184_000_000,
                earth_orientation: PartAEarthOrientationConvention::IersFinals2000ADefinitive,
                reference_epoch_tai: require_ok!(PartATaiEpoch::new(0, 0)),
                manifest_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            })
        ));
        assert!(
            require_err!(event_mismatch.require_production_hybrid_authority())
                .to_string()
                .contains("manifest hash")
        );

        let mut gravity_mismatch = require_ok!(syntactically_verified_science());
        let evidence = gravity_mismatch.reference_evidence;
        gravity_mismatch.gravity_authority = PartAGravityAuthority::Verified(require_ok!(
            PartAVerifiedGravity::new(PartAVerifiedGravityInput {
                source_model: "model",
                normalization: "fully_normalized",
                tide_system: "tide_free",
                source_gm_km3_s2: 398_600.441_8,
                source_reference_radius_km: 6_378.136_3,
                source_max_degree: 300,
                source_max_order: 300,
                stored_degree: 15,
                stored_order: 15,
                runtime_degree: 6,
                runtime_order: 5,
                coefficient_sha256: evidence.gravity_coefficient_sha256,
                manifest_sha256: evidence.gravity_manifest_sha256,
            })
        ));
        assert!(
            require_err!(gravity_mismatch.require_production_hybrid_authority())
                .to_string()
                .contains("model order or hashes")
        );
    }
}
