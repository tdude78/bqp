//! Mass solver for dust deflection optimization.
//!
//! Finds the exact dust mass required to achieve a target miss distance
//! using Brent's method root-finding with native Rust physics computations.
//!
//! # Physics Model
//!
//! The momentum transfer formula is based on inelastic collision mechanics:
//!
//! ```text
//! Δv = κ × (m_dust / (m_primary + m_dust)) × v_relative
//! ```
//!
//! The κ (kappa) parameter accounts for momentum enhancement from ejecta.
//! Production uses κ = 1.0 as conservative only within the current monotone
//! direct-momentum model. This does not guarantee real dust interaction has
//! effective transfer κ >= 1; values above unity remain parametric enhancement.
//! Closest concept references are Ganguli et al. (2014), doi:10.1063/1.4865347,
//! and Crabtree et al. (2014), doi:10.1109/USNC-URSI-NRSM.2014.6928101.
//! Stronger upper-context momentum-enhancement measurements come from DART and
//! rock-target experiments: Cheng et al. (2023), doi:10.1038/s41586-023-05878-z,
//! and Flynn et al. (2024), doi:10.1115/HVIS2024-032.
//!
//! # High-Fidelity Mode
//!
//! When `use_high_fidelity` is enabled, the solver uses Lightyear ODE integration
//! with perturbations (J2, drag, SRP) instead of two-body Keplerian propagation.
//! This provides physically accurate miss distance computations for conjunction
//! remediation assessment.
//!
//! # References
//!
//! - Cheng, A.F. et al. (2023). "Momentum Transfer from the DART Mission
//!   Kinetic Impact on Asteroid Dimorphos." Nature 616, 457-460.
//!   <https://doi.org/10.1038/s41586-023-05878-z>
//!
//! - Battin, R.H. (1999). "An Introduction to the Mathematics and Methods
//!   of Astrodynamics." AIAA Education Series.

use lightyear_odeint_rs::{
    integrator::FinalPropagationFailure,
    types::{ForceConfig, StepperMethod},
    ScalarGravityAssets, ScalarPropagationContext, ScalarPropagationRequest,
};
use satpy_core::{
    cross3, eci2equinoc_impl, equinoc2eci_impl, equinoc_prop_from_impl, equinoc_prop_j2_from_impl,
    norm3, PackedGravityCoeffs, MU, RE, SEC_PER_DAY,
};
use std::collections::HashMap;

/// Fraction of the local energy scale `mu/r` below which a marginally positive
/// specific energy is still treated as bound.
///
/// Derived, not chosen: the previous absolute margin was 1e-6 km^2/s^2,
/// documented as eccentricity ~1.0001 "at LEO altitudes". At a 400 km LEO
/// (r = 6778 km) the energy scale is `satpy_core::MU / 6778`, so
/// `1e-6 / (MU / 6778)` reproduces that calibration EXACTLY at the radius it
/// was written for, while scaling correctly everywhere else.
///
/// Derive it from THIS crate's `MU` (398600.4415), not from a textbook value:
/// using 398600.4418 puts the reconstructed margin 7.5e-10 off, which the
/// companion test catches.
const ESCAPE_ENERGY_RELATIVE_TOL: f64 = 1.700_449_697_068_386e-8;

/// The one escape predicate, shared by LF, MF-J2 and HF.
///
/// The three fidelities used to disagree. LF scaled the margin by the local
/// energy `mu/r`; MF-J2 and HF kept a bare absolute `1e-6 km^2/s^2`. They
/// coincide only near the 400 km LEO radius the absolute number was calibrated
/// at, so a marginal orbit could be called bound at one fidelity and escaped at
/// another purely because of its altitude -- a fidelity-dependent verdict about
/// a property of the orbit.
///
/// Operation order is fixed here rather than left to each call site, so the
/// three cannot drift apart again in the last bit.
#[inline]
fn specific_energy_is_escaped(specific_energy: f64, mu_over_r: f64) -> bool {
    specific_energy > ESCAPE_ENERGY_RELATIVE_TOL * mu_over_r
}
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use wide::f64x4;

mod observer;
mod profile;
#[cfg(feature = "solver-qualification")]
mod qualification;
mod status;
use observer::{
    MassBatchObserver, MassLegIdentity, MassLegRole, MassLegTag, MassRowObservation,
    MassSolveObserver, UnobservedMassSolve,
};
pub use profile::HfMassSolveProfile;
use profile::{
    hf_counters_enabled, hf_profile_inc_full_refine_iteration, hf_profile_inc_lf_fallback,
    hf_profile_inc_upper_bracket_shrink, hf_profile_inc_validate_refine_iteration,
    hf_profile_record, hf_profile_reset, hf_profile_set_anchor_diagnostics, hf_profile_snapshot,
    HfProfileStage,
};
#[cfg(feature = "solver-qualification")]
pub use qualification::{
    QualificationMassBatchObservation, QualificationMassBatchRow, QualificationMassLeg,
    QualificationMassLegRole, QualificationMassObservationError, MAX_QUALIFICATION_MASS_BATCH_ROWS,
    MAX_QUALIFICATION_MASS_LEGS,
};
use status::{converged_status_for_mass, write_mf_j2_result};
pub use status::{MassSolveStatusCode, MfJ2MassSolveResult, MfJ2MassSolveStatusCode};

const V3_STRICT_HF: bool = false;
static HF_PREFLIGHT_FAILURES: AtomicU64 = AtomicU64::new(0);

/// Deterministic-mass numerical policy: shared practical-mass floor,
/// one-sided safe-bracket convergence, and bounded nonfinite repair within the
/// existing strict-HF evaluation budget.
pub const DETERMINISTIC_MASS_NUMERICAL_POLICY_ID: &str = "practical-floor-safe-bracket-v1";

/// Commanded deterministic mass floor bound to
/// [`DETERMINISTIC_MASS_NUMERICAL_POLICY_ID`].
pub const MINIMUM_OPERATIONAL_DETERMINISTIC_MASS_KG: f64 = 5.0e-7;

/// The sealed deterministic-mass route that issued a witness.
///
/// Closed on purpose. `from_authority_id` returns `None` for anything it does
/// not recognise rather than defaulting, and every match on this enum must
/// name both arms: a `_ =>` arm here would silently decode a future route as
/// an existing one, which is exactly the failure a single free-string
/// comparison already caused once (a genuine strict-HF witness could not
/// round-trip through an MF-J2-only persisted verifier).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeterministicMassRoute {
    /// MF-J2 deterministic mass, issued by the MF-J2 batch solver.
    MfJ2,
    /// Strict-HF deterministic mass, issued by the strict-HF batch solver.
    StrictHf,
}

impl DeterministicMassRoute {
    /// The sealed authority identifier this route issues.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MfJ2 => "mf-j2-deterministic-mass-v1",
            Self::StrictHf => "strict-hf-deterministic-mass-v1",
        }
    }

    /// Resolve a persisted authority identifier, failing closed when unknown.
    #[must_use]
    pub fn from_authority_id(authority_id: &str) -> Option<Self> {
        if authority_id == Self::MfJ2.as_str() {
            Some(Self::MfJ2)
        } else if authority_id == Self::StrictHf.as_str() {
            Some(Self::StrictHf)
        } else {
            None
        }
    }
}

/// Opaque proof that a deterministic solver executed with the carried coupling.
///
/// Fields have no public or crate-visible constructor. Only solver execution in
/// this module can issue this evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DeterministicMassEvidence {
    required_mass_kg: f64,
    momentum_coupling_kappa: f64,
    mass_authority_id: &'static str,
}

impl DeterministicMassEvidence {
    pub(crate) const fn required_mass_kg(self) -> f64 {
        self.required_mass_kg
    }

    pub(crate) const fn momentum_coupling_kappa(self) -> f64 {
        self.momentum_coupling_kappa
    }

    pub(crate) const fn mass_authority_id(self) -> &'static str {
        self.mass_authority_id
    }
}

/// Opaque operational mass policy applied to immutable raw solver evidence.
///
/// Construction is restricted to [`DeterministicMassSolveOutcome::operational_mass`].
/// The raw solver mass remains unchanged; `commanded_required_mass_kg` carries
/// the fixed policy floor independently.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OperationalDeterministicMass {
    raw_evidence: DeterministicMassEvidence,
    commanded_required_mass_kg: f64,
}

impl OperationalDeterministicMass {
    #[must_use]
    pub const fn commanded_required_mass_kg(self) -> f64 {
        self.commanded_required_mass_kg
    }

    #[must_use]
    pub const fn raw_solver_mass_kg(self) -> f64 {
        self.raw_evidence.required_mass_kg()
    }

    #[must_use]
    pub(crate) const fn momentum_coupling_kappa(self) -> f64 {
        self.raw_evidence.momentum_coupling_kappa()
    }

    #[must_use]
    pub(crate) const fn mass_authority_id(self) -> &'static str {
        self.raw_evidence.mass_authority_id()
    }
}

/// One executed deterministic-mass row and any evidence that execution was
/// eligible to issue.
///
/// Evidence is present only for a positive finite converged result. The type
/// has no public constructor, so production callers cannot attach evidence to
/// a caller-supplied mass or status.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeterministicMassSolveOutcome<S> {
    mass_kg: f64,
    status: S,
    evidence: Option<DeterministicMassEvidence>,
}

impl<S: Copy> DeterministicMassSolveOutcome<S> {
    #[must_use]
    pub const fn mass_kg(self) -> f64 {
        self.mass_kg
    }

    #[must_use]
    pub const fn status(self) -> S {
        self.status
    }

    /// Bind immutable raw evidence to the fixed operational mass policy.
    ///
    /// Non-evidence outcomes return `None`. Evidence-bearing outcomes must
    /// preserve the solver-issued mass/evidence identity. The returned token
    /// carries the commanded floor without rewriting either raw value.
    ///
    /// # Errors
    ///
    /// Returns an error when an evidence-bearing outcome does not contain
    /// matching finite positive raw mass bits.
    pub fn operational_mass(self) -> anyhow::Result<Option<OperationalDeterministicMass>> {
        let Some(evidence) = self.evidence else {
            return Ok(None);
        };
        anyhow::ensure!(
            self.mass_kg.is_finite()
                && self.mass_kg > 0.0
                && evidence.required_mass_kg.is_finite()
                && evidence.required_mass_kg > 0.0,
            "converged deterministic mass evidence must be finite and positive",
        );
        anyhow::ensure!(
            self.mass_kg.to_bits() == evidence.required_mass_kg.to_bits(),
            "deterministic mass and evidence disagree",
        );
        Ok(Some(OperationalDeterministicMass {
            raw_evidence: evidence,
            commanded_required_mass_kg: self.mass_kg.max(MINIMUM_OPERATIONAL_DETERMINISTIC_MASS_KG),
        }))
    }
}

fn deterministic_mass_outcome<S: Copy>(
    mass_kg: f64,
    status: S,
    momentum_coupling_kappa: f64,
    mass_authority_id: &'static str,
    converged: bool,
) -> DeterministicMassSolveOutcome<S> {
    let evidence =
        (converged && mass_kg.is_finite() && mass_kg > 0.0).then_some(DeterministicMassEvidence {
            required_mass_kg: mass_kg,
            momentum_coupling_kappa,
            mass_authority_id,
        });
    DeterministicMassSolveOutcome {
        mass_kg,
        status,
        evidence,
    }
}

fn latch_hf_preflight_failure() {
    let _ = HF_PREFLIGHT_FAILURES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
        Some(count.saturating_add(1))
    });
}

/// Consume the bounded HF-preflight failure count at a quiescent owner seam.
/// Numerical workers only latch this fact; they never format or write it.
#[must_use]
pub fn take_hf_preflight_failure_count() -> u64 {
    HF_PREFLIGHT_FAILURES.swap(0, Ordering::AcqRel)
}
const HF_BATCH_PAR_THRESHOLD: usize = HF_BATCH_PAR_THRESHOLD_DEFAULT;
const MF_J2_BATCH_PAR_THRESHOLD: usize = MF_J2_BATCH_PAR_THRESHOLD_DEFAULT;

const HF_BATCH_PAR_THRESHOLD_DEFAULT: usize = 32;
/// Largest MF-J2 batch that stays serial.
///
/// This has now been lowered twice for the same reason, and the second time is
/// why the value is 32 rather than something rounder.
///
/// 2026-08-07: was 1024, which none of the 384 captured calls reached. Measured
/// over a p16/e24 Part A MF population: rows per batch min
/// 366, p50 520, p90 696, max 864, so the gate fired 0 of 384 times and Stage 3
/// deterministic mass ran wholly on the calling thread while every Rayon worker
/// slept -- 9.07% of an MF cell's wall at the sealed p64/e24 shape. Lowered to
/// 256, below that measured minimum with margin.
///
/// 2026-08-13: the `nd-epsilon-membership` reseal cut mass rows 284k -> 37k
/// (7.7x) while leaving the batch COUNT untouched -- epsilon narrows the front,
/// not the number of design-events -- so the mean rows per captured batch fell
/// by the same aggregate factor and 256 went back above the sampled distribution.
/// Same never-firing gate, second occurrence.
///
/// Post-epsilon rows per captured call were measured 2026-08-13 by temporarily
/// instrumenting `det_mass_rows.len()` at the MF call site
/// (`nd_pipeline::physics::orchestrate`). Aggregate provenance and its limits
/// are retained in `docs/PART_A_RESULTS_MATRIX.md`, under "Historical MF-J2
/// batch-row distribution — commit-recorded summary". That historical sample
/// has p10 = 34 and p50 = 56 in both recorded run shapes.
///
/// An earlier revision of this block INFERRED min 48 by applying the aggregate
/// 7.7x row cut to the 2026-08-07 minimum. That was wrong: epsilon does not cut
/// uniformly (p50 fell 9.3x, the minimum fell 36.6x), and an aggregate ratio
/// cannot be pushed through a distribution's extreme.
///
/// 32 therefore did NOT sit below every sampled population call: 76 of 888
/// (8.6%) had at most 32 rows and stayed serial. A separate `X_SMALL` anchor
/// call made the original captured total 889 and was included in its quantiles.
/// The 33-row dispatch cutoff sits just below that historical p10, so about 90% of the
/// sampled population calls dispatch. Parallelising the small tail was not
/// obviously worth it: a row cost ~10.3 us (0.54 s / 52,378 rows), so a 10-row
/// batch was ~103 us against tens of microseconds of `par_iter` split overhead,
/// while a 33-row batch was ~340 us. Dispatching everything would have needed
/// the threshold at <= 9 and was declined without a wall A/B.
///
/// The former "128-row batches keep their pinned serial dispatch" constraint is
/// retired: it was chosen when the range was 366..864, where 128 sat below
/// every captured call. The post-epsilon recorded max was 142, so the gate
/// could still fire in a rare upper tail. The contemporaneous transcript's
/// anchor-inclusive aggregate counted 9 calls above 128; its separate `X_SMALL`
/// anchor had 54 rows, so all 9 were population calls: 9 of 888 (1.0%). The raw
/// row log is gone, so this is historical transcript evidence, not a
/// reproducible current measurement; the results matrix retains its provenance.
/// With p90 = 84, 128 would leave the overwhelming majority serial.
const MF_J2_BATCH_PAR_THRESHOLD_DEFAULT: usize = 32;

#[inline]
const fn v3_strict_hf_enabled() -> bool {
    V3_STRICT_HF
}

#[inline]
const fn hf_batch_dispatch_parallel_threshold(base_threshold: usize) -> usize {
    // Treat configured threshold as the largest serial batch size.
    // `should_parallelize` uses `>=`, so add one to encode strict `>`.
    base_threshold.saturating_add(1)
}

#[inline]
const fn mf_j2_batch_dispatch_parallel_threshold(base_threshold: usize) -> usize {
    base_threshold.saturating_add(1)
}

#[inline]
fn should_use_mf_j2_batch_parallel_dispatch(events_len: usize) -> bool {
    satpy_core::parallel_utils::should_parallelize(
        events_len,
        mf_j2_batch_dispatch_parallel_threshold(MF_J2_BATCH_PAR_THRESHOLD),
    )
}

/// High-fidelity context for physics computations.
///
/// When provided to the solver, enables Lightyear ODE integration with
/// perturbations (J2-J5, drag, SRP) instead of two-body Keplerian propagation.
#[derive(Clone)]
pub struct HfContext {
    /// Whether to use high-fidelity propagation (vs two-body Keplerian).
    pub use_high_fidelity: bool,
    /// Propagation epoch as Julian date (days since `J2000` epoch = `JD 2451545.0`).
    pub epoch_jd: f64,
    /// Force model configuration (gravity order, perturbation flags, tolerances).
    pub force_config: Option<Arc<ForceConfig>>,
    /// Packed gravity coefficients for scalar SIMD evaluation.
    pub packed_coeffs: Option<Arc<PackedGravityCoeffs>>,
    /// Use LF seed + HF validate/repair path (true) vs full HF bisection (false).
    pub hf_validate_only: bool,
    /// Strict HF mode: do not silently fall back to LF when HF is requested.
    pub hf_strict: bool,
}

/// Pre-computed HF configuration for a single event's root-finding loop.
///
/// PERF2 optimization: [`ForceConfig`] clone, per-event overrides (`am_ratio`, `cd`, `cr`),
/// and ephemeris resolution (`with_ephemeris`) are invariant across solver
/// iterations. This struct caches those results so the inner loop
/// only does physics (velocity change + propagation + miss distance).
struct PreparedHfConfig {
    force_config: Arc<ForceConfig>,
    epoch_jd: f64,
    packed_coeffs: Arc<PackedGravityCoeffs>,
}

impl PreparedHfConfig {
    fn scalar_propagation_context(&self) -> ScalarPropagationContext {
        let gravity = ScalarGravityAssets::new(Arc::clone(&self.packed_coeffs));
        ScalarPropagationContext::new(self.epoch_jd, Arc::clone(&self.force_config), gravity)
    }

    /// Apply the production retained-body model without repeating ephemeris
    /// resolution. The collision is perfectly inelastic and retains the
    /// catalogue target's physical area, so only area-to-mass changes:
    /// `A / (M + m) = (A / M) * M / (M + m)`.
    fn for_retained_mass(&self, retained_mass_kg: f64, event: &MassSolverEvent) -> Option<Self> {
        if !retained_mass_kg.is_finite()
            || retained_mass_kg < 0.0
            || !event.p_mass.is_finite()
            || event.p_mass <= 0.0
        {
            return None;
        }
        let force_config = if retained_mass_kg == 0.0 {
            Arc::clone(&self.force_config)
        } else {
            let postimpact_mass_kg = event.p_mass + retained_mass_kg;
            if !postimpact_mass_kg.is_finite() || postimpact_mass_kg <= 0.0 {
                return None;
            }
            let retained_am_ratio =
                self.force_config.am_ratio * (event.p_mass / postimpact_mass_kg);
            if !retained_am_ratio.is_finite() || retained_am_ratio < 0.0 {
                return None;
            }
            let mut config = *self.force_config;
            config.am_ratio = retained_am_ratio;
            Arc::new(config)
        };
        Some(Self {
            force_config,
            epoch_jd: self.epoch_jd,
            packed_coeffs: Arc::clone(&self.packed_coeffs),
        })
    }
}

/// Sealed production retained-body dynamics implemented by the HF mass solver.
pub const RETAINED_MASS_DYNAMICS_ID: &str = "perfectly-inelastic-fixed-area-retention-v1";

/// Checked immutable inputs for one fixed retained-body strict-HF propagation.
///
/// Fields stay private. Callers must pass the validation in [`Self::try_new`]
/// and cannot construct a raw request whose retained mass or numeric inputs
/// bypass the fixed-impact authority.
#[derive(Clone)]
pub struct FixedImpactHfRequest {
    event: MassSolverEvent,
    context: HfContext,
    sampled_retained_mass_kg: f64,
}

impl std::fmt::Debug for FixedImpactHfRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FixedImpactHfRequest")
            .field("sampled_retained_mass_kg", &self.sampled_retained_mass_kg)
            .finish_non_exhaustive()
    }
}

impl FixedImpactHfRequest {
    /// Validate one fixed retained-body strict-HF request.
    ///
    /// # Errors
    ///
    /// Returns a typed failure for an invalid retained mass, a non-finite
    /// numeric input, or a context that cannot execute strict HF.
    pub fn try_new(
        event: MassSolverEvent,
        context: HfContext,
        sampled_retained_mass_kg: f64,
    ) -> Result<Self, FixedImpactHfFailure> {
        if !sampled_retained_mass_kg.is_finite() || sampled_retained_mass_kg < 0.0 {
            return Err(FixedImpactHfFailure::InvalidMass);
        }
        let required_scalars = [
            event.p_mass,
            event.tof_s,
            event.min_miss_distance_km,
            event.kappa,
            context.epoch_jd,
        ];
        let required_vectors = [
            event.p_momentum.as_slice(),
            event.dv_vec.as_slice(),
            event.p_pos_intercept.as_slice(),
            event.secondary_conj_pos.as_slice(),
            event.p_pos_conj_truth.as_slice(),
            event.p_pos_conj_equ_0.as_slice(),
            event.p_velocity.as_slice(),
            event.v_rel.as_slice(),
            event.p_equ_intercept.as_slice(),
        ];
        let optional_scalars = [
            event.p_am_ratio,
            event.p_cd,
            event.p_cr,
            event.p_qm_ratio,
            event.p_r_obj_m,
        ];
        if !required_scalars.iter().all(|value| value.is_finite())
            || !required_vectors
                .iter()
                .flat_map(|values| values.iter())
                .all(|value| value.is_finite())
            || !optional_scalars
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        {
            return Err(FixedImpactHfFailure::NonFinite);
        }
        if event.p_mass <= 0.0
            || event.tof_s < 0.0
            || event.min_miss_distance_km <= 0.0
            || event.kappa < 0.0
        {
            return Err(FixedImpactHfFailure::InvalidInput);
        }
        if !context.use_high_fidelity
            || !context.hf_strict
            || context.force_config.is_none()
            || context.packed_coeffs.is_none()
        {
            return Err(FixedImpactHfFailure::HighFidelityContextRequired);
        }
        Ok(Self {
            event,
            context,
            sampled_retained_mass_kg,
        })
    }
}

/// Binary collision-threshold classification for one fixed-impact result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedImpactHfVerdict {
    /// Strict separation exceeds the compiled threshold.
    Safe,
    /// Separation is below or exactly equal to the compiled threshold.
    Miss,
}

/// Typed fail-closed reason a fixed retained-body strict-HF evaluation stopped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedImpactHfFailure {
    InvalidMass,
    InvalidInput,
    NonFinite,
    HighFidelityContextRequired,
    HighFidelityPreparationFailed,
    InvalidOrbit,
    Ground,
    Escape,
    Integration(FinalPropagationFailure),
}

impl FixedImpactHfFailure {
    /// Stable closed identifier for canonical evidence serialization.
    #[must_use]
    pub const fn evidence_id(self) -> &'static str {
        match self {
            Self::InvalidMass => "fixed-impact:invalid-mass",
            Self::InvalidInput => "fixed-impact:invalid-input",
            Self::NonFinite => "fixed-impact:nonfinite",
            Self::HighFidelityContextRequired => "fixed-impact:hf-context-required",
            Self::HighFidelityPreparationFailed => "fixed-impact:hf-preparation-failed",
            Self::InvalidOrbit => "fixed-impact:invalid-orbit",
            Self::Ground => "fixed-impact:ground",
            Self::Escape => "fixed-impact:escape",
            Self::Integration(failure) => failure.evidence_id(),
        }
    }
}

/// Whether `evidence_id` is one of the closed fixed-impact failure identifiers.
#[must_use]
pub fn is_fixed_impact_hf_failure_evidence_id(evidence_id: &str) -> bool {
    matches!(
        evidence_id,
        "fixed-impact:invalid-mass"
            | "fixed-impact:invalid-input"
            | "fixed-impact:nonfinite"
            | "fixed-impact:hf-context-required"
            | "fixed-impact:hf-preparation-failed"
            | "fixed-impact:invalid-orbit"
            | "fixed-impact:ground"
            | "fixed-impact:escape"
    ) || lightyear_odeint_rs::integrator::is_final_propagation_failure_evidence_id(evidence_id)
}

/// Finite result of one checked fixed retained-body strict-HF propagation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedImpactHfOutcome {
    verdict: FixedImpactHfVerdict,
    miss_distance_km: f64,
    sampled_retained_mass_kg: f64,
    retained_area_to_mass_ratio_m2_per_kg: f64,
    deterministic_mass_min_distance_km: f64,
    impact_state_eci: [f64; 6],
    conjunction_state_eci: [f64; 6],
}

impl FixedImpactHfOutcome {
    #[must_use]
    pub const fn verdict(&self) -> FixedImpactHfVerdict {
        self.verdict
    }

    #[must_use]
    pub const fn miss_distance_km(&self) -> f64 {
        self.miss_distance_km
    }

    #[must_use]
    pub const fn sampled_retained_mass_kg(&self) -> f64 {
        self.sampled_retained_mass_kg
    }

    #[must_use]
    pub const fn retained_area_to_mass_ratio_m2_per_kg(&self) -> f64 {
        self.retained_area_to_mass_ratio_m2_per_kg
    }

    #[must_use]
    pub const fn deterministic_mass_min_distance_km(&self) -> f64 {
        self.deterministic_mass_min_distance_km
    }

    #[must_use]
    pub const fn impact_state_eci(&self) -> [f64; 6] {
        self.impact_state_eci
    }

    #[must_use]
    pub const fn conjunction_state_eci(&self) -> [f64; 6] {
        self.conjunction_state_eci
    }
}

/// Word capacity of [`append_force_config_authority_bits`]: 19 unconditional
/// scalar words, six tagged optional vec3s (4 words each), the
/// dynamic-ephemeris flags word, six tagged optional `BodyInvariants`
/// (6 words each), then `dt_max`, `eps` and the integrator method.
const FORCE_CONFIG_AUTHORITY_WORDS: usize = 19 + 6 * 4 + 1 + 6 * 6 + 3;

/// Word capacity of [`append_hf_context_authority_bits`]: five tagged
/// per-event override scalars (2 words each), `use_high_fidelity`,
/// `epoch_jd`, `hf_strict`, the force-config presence tag, then the
/// (zero-padded when absent) force-config block.
const HF_CONTEXT_AUTHORITY_WORDS: usize = 5 * 2 + 3 + 1 + FORCE_CONFIG_AUTHORITY_WORDS;

/// Word capacity of [`ZeroMassAuthorityKey`]: 27 event words (six vec3s, a
/// six-element equinoctial state, and three scalars) plus the context block.
const ZERO_MASS_AUTHORITY_WORDS: usize = 27 + HF_CONTEXT_AUTHORITY_WORDS;

/// Word capacity of [`AnchorAuthorityKey`]: the six-element zero-mass ECI
/// state plus `tof_s`, plus the context block.
const ANCHOR_AUTHORITY_WORDS: usize = 6 + 1 + HF_CONTEXT_AUTHORITY_WORDS;

/// Fixed-capacity, append-only word buffer for authority cache keys.
///
/// Replaces the per-row heap `Vec<u64>` keys: every key layout is statically
/// sized (each `Option` arm appends a fixed word count, zero-padded when
/// absent), so the key material lives inline in the `HashMap` entry with no
/// per-row allocation. [`Self::finish`] asserts the buffer was filled
/// EXACTLY, so a mis-declared capacity or a forgotten padding arm is a loud
/// panic at key construction, not a silently truncated or collision-prone key.
struct AuthorityBitsWriter<const WORDS: usize> {
    words: [u64; WORDS],
    filled: usize,
}

impl<const WORDS: usize> AuthorityBitsWriter<WORDS> {
    const fn new() -> Self {
        Self {
            words: [0; WORDS],
            filled: 0,
        }
    }

    fn push(&mut self, word: u64) {
        assert!(
            self.filled < WORDS,
            "authority key overflows its declared {WORDS}-word capacity"
        );
        if let Some(slot) = self.words.get_mut(self.filled) {
            *slot = word;
        }
        self.filled = self.filled.saturating_add(1);
    }

    fn push_f64(&mut self, value: f64) {
        self.push(value.to_bits());
    }

    fn push_f64s<const M: usize>(&mut self, values: [f64; M]) {
        for value in values {
            self.push_f64(value);
        }
    }

    fn pad_zero(&mut self, count: usize) {
        for _ in 0..count {
            self.push(0);
        }
    }

    fn push_optional_f64(&mut self, value: Option<f64>) {
        if let Some(value) = value {
            self.push(1);
            self.push_f64(value);
        } else {
            self.push(0);
            self.push(0);
        }
    }

    fn push_optional_vec3(&mut self, value: Option<[f64; 3]>) {
        if let Some(value) = value {
            self.push(1);
            self.push_f64s(value);
        } else {
            self.push(0);
            self.pad_zero(3);
        }
    }

    fn push_optional_body_invariants(
        &mut self,
        value: Option<lightyear_odeint_rs::types::BodyInvariants>,
    ) {
        if let Some(value) = value {
            // Exhaustive destructure, no rest pattern: a new BodyInvariants
            // field is a compile error here until the key includes it.
            let lightyear_odeint_rs::types::BodyInvariants {
                body_norm,
                inv_body_dist,
                mu_coef,
            } = value;
            self.push(1);
            self.push_f64s(body_norm);
            self.push_f64(inv_body_dist);
            self.push_f64(mu_coef);
        } else {
            self.push(0);
            self.pad_zero(5);
        }
    }

    fn finish(self) -> [u64; WORDS] {
        assert_eq!(
            self.filled, WORDS,
            "authority key underfills its declared {WORDS}-word capacity"
        );
        self.words
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ZeroMassAuthorityKey {
    bits: [u64; ZERO_MASS_AUTHORITY_WORDS],
    packed_coeffs: Option<*const PackedGravityCoeffs>,
}

impl ZeroMassAuthorityKey {
    fn new(event: &MassSolverEvent, context: &HfContext) -> Self {
        // Exhaustive destructure, no rest pattern: adding a MassSolverEvent
        // field without deciding its key membership is a compile error here,
        // not a silent cache-collision route to wrong-bits-from-cache.
        let MassSolverEvent {
            p_momentum: _, // keyed via its derived p_velocity below
            dv_vec: _,     // keyed via its derived v_rel below
            p_mass,
            p_pos_intercept,
            tof_s,
            secondary_conj_pos,
            min_miss_distance_km: _, // root-finding target, not a zero-mass input
            kappa,
            p_pos_conj_truth,
            p_pos_conj_equ_0,
            p_velocity,
            v_rel,
            p_equ_intercept,
            // The five per-event overrides are keyed in the shared HF-context
            // block appended below.
            p_am_ratio: _,
            p_cd: _,
            p_cr: _,
            p_qm_ratio: _,
            p_r_obj_m: _,
        } = event;
        let mut bits = AuthorityBitsWriter::<ZERO_MASS_AUTHORITY_WORDS>::new();
        bits.push_f64s(*p_pos_intercept);
        bits.push_f64s(*p_velocity);
        bits.push_f64(*p_mass);
        bits.push_f64s(*v_rel);
        bits.push_f64(*kappa);
        bits.push_f64(*tof_s);
        bits.push_f64s(*secondary_conj_pos);
        bits.push_f64s(*p_pos_conj_truth);
        bits.push_f64s(*p_pos_conj_equ_0);
        bits.push_f64s(*p_equ_intercept);
        append_hf_context_authority_bits(&mut bits, event, context);
        Self {
            bits: bits.finish(),
            packed_coeffs: context.packed_coeffs.as_ref().map(Arc::as_ptr),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AnchorAuthorityKey {
    // Exact zero-mass ECI state, time, force, asset, integrator authority.
    // Deliberately target-only: it does NOT destructure the whole event the
    // way ZeroMassAuthorityKey does, because the anchor's authority is the
    // zero-mass state, not the event that produced it.
    bits: [u64; ANCHOR_AUTHORITY_WORDS],
    packed_coeffs: Option<*const PackedGravityCoeffs>,
}

impl AnchorAuthorityKey {
    fn new(event: &MassSolverEvent, context: &HfContext) -> Self {
        let mut bits = AuthorityBitsWriter::<ANCHOR_AUTHORITY_WORDS>::new();
        bits.push_f64s(zero_mass_eci_state(event));
        bits.push_f64(event.tof_s);
        append_hf_context_authority_bits(&mut bits, event, context);
        Self {
            bits: bits.finish(),
            packed_coeffs: context.packed_coeffs.as_ref().map(Arc::as_ptr),
        }
    }
}

fn append_hf_context_authority_bits<const WORDS: usize>(
    bits: &mut AuthorityBitsWriter<WORDS>,
    event: &MassSolverEvent,
    context: &HfContext,
) {
    // Exhaustive destructure, no rest pattern (see the key constructors).
    let HfContext {
        use_high_fidelity,
        epoch_jd,
        force_config,
        // Keyed by pointer identity on the key structs themselves.
        packed_coeffs: _,
        // Not keyed before this refactor either; the key contents are
        // preserved verbatim, this line only makes the omission visible.
        hf_validate_only: _,
        hf_strict,
    } = context;
    for value in [
        event.p_am_ratio,
        event.p_cd,
        event.p_cr,
        event.p_qm_ratio,
        event.p_r_obj_m,
    ] {
        bits.push_optional_f64(value);
    }
    bits.push((*use_high_fidelity).into());
    bits.push_f64(*epoch_jd);
    bits.push((*hf_strict).into());
    if let Some(force) = force_config.as_deref() {
        bits.push(1);
        append_force_config_authority_bits(bits, force);
    } else {
        // Zero-pad the absent block so every key has the same fixed width;
        // the presence tag keeps an absent config distinct from an all-zero
        // one.
        bits.push(0);
        bits.pad_zero(FORCE_CONFIG_AUTHORITY_WORDS);
    }
}

fn usize_authority_bits(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn signed_authority_bits(value: i32) -> u64 {
    u64::from(u32::from_ne_bytes(value.to_ne_bytes()))
}

const fn stepper_method_authority_bits(method: StepperMethod) -> u64 {
    match method {
        StepperMethod::Dopri5Compat => 0,
        StepperMethod::Tsit5 => 1,
        StepperMethod::Dop853 => 2,
        StepperMethod::Rkv98 => 3,
        StepperMethod::Vern7 => 4,
        StepperMethod::Vern9 => 5,
        StepperMethod::Esdirk43 => 6,
        StepperMethod::Auto => 7,
    }
}

fn append_force_config_authority_bits<const WORDS: usize>(
    bits: &mut AuthorityBitsWriter<WORDS>,
    force: &ForceConfig,
) {
    // Exhaustive destructure, no rest pattern: a NEW ForceConfig field is a
    // compile error here until the key includes it, instead of a silent
    // cache-collision route where two configs differing only in the new
    // field share a slot and serve each other's cached values.
    let ForceConfig {
        sph_order,
        force_flags,
        subtract_first_order,
        atm_model,
        am_ratio,
        cd,
        cr,
        target_propagation_mode,
        qm_ratio,
        r_obj_m,
        omega_earth,
        p_sun,
        mu_sun,
        mu_moon,
        mu_jupiter,
        mu_venus,
        mu_mars,
        mu_saturn,
        earth_radius,
        sun_pos,
        moon_pos,
        jupiter_pos,
        venus_pos,
        mars_pos,
        saturn_pos,
        dynamic_ephemeris_flags,
        sun_invariants,
        moon_invariants,
        jupiter_invariants,
        venus_invariants,
        mars_invariants,
        saturn_invariants,
        dt_max,
        eps,
        integrator_method,
    } = *force;
    bits.push(usize_authority_bits(sph_order));
    bits.push(signed_authority_bits(force_flags));
    bits.push(subtract_first_order.into());
    bits.push(signed_authority_bits(atm_model));
    bits.push_f64(am_ratio);
    bits.push_f64(cd);
    bits.push_f64(cr);
    bits.push(u64::from(target_propagation_mode));
    bits.push_f64(qm_ratio);
    bits.push_f64(r_obj_m);
    bits.push_f64(omega_earth);
    bits.push_f64(p_sun);
    bits.push_f64(mu_sun);
    bits.push_f64(mu_moon);
    bits.push_f64(mu_jupiter);
    bits.push_f64(mu_venus);
    bits.push_f64(mu_mars);
    bits.push_f64(mu_saturn);
    bits.push_f64(earth_radius);
    for value in [
        sun_pos,
        moon_pos,
        jupiter_pos,
        venus_pos,
        mars_pos,
        saturn_pos,
    ] {
        bits.push_optional_vec3(value);
    }
    bits.push(signed_authority_bits(dynamic_ephemeris_flags));
    for value in [
        sun_invariants,
        moon_invariants,
        jupiter_invariants,
        venus_invariants,
        mars_invariants,
        saturn_invariants,
    ] {
        bits.push_optional_body_invariants(value);
    }
    bits.push_f64(dt_max);
    bits.push_f64(eps);
    bits.push(stepper_method_authority_bits(integrator_method));
}

#[derive(Clone, Copy)]
struct ZeroMassCacheView<'a> {
    anchor_reference: &'a OnceLock<ZeroMassReference>,
    miss_at_zero: &'a OnceLock<f64>,
}

#[derive(Clone, Copy, Debug)]
struct ZeroMassReference {
    position: Option<[f64; 3]>,
    exact_hf: bool,
}

struct ZeroMassBatchCache {
    // Anchor and miss have different authorities: miss retains every IEEE
    // input read by compute_new_velocity(0), while anchor is target-only.
    anchor_slots: Vec<OnceLock<ZeroMassReference>>,
    miss_slots: Vec<OnceLock<f64>>,
    row_to_anchor_slot: Vec<usize>,
    row_to_miss_slot: Vec<usize>,
}

impl ZeroMassBatchCache {
    fn from_rows(rows: &[(MassSolverEvent, HfContext)]) -> Self {
        let mut anchor_key_to_slot = HashMap::with_capacity(rows.len());
        let mut miss_key_to_slot = HashMap::with_capacity(rows.len());
        let mut row_to_anchor_slot = Vec::with_capacity(rows.len());
        let mut row_to_miss_slot = Vec::with_capacity(rows.len());
        for (event, context) in rows {
            let next_anchor_slot = anchor_key_to_slot.len();
            let anchor_slot = *anchor_key_to_slot
                .entry(AnchorAuthorityKey::new(event, context))
                .or_insert(next_anchor_slot);
            row_to_anchor_slot.push(anchor_slot);
            let next_miss_slot = miss_key_to_slot.len();
            let miss_slot = *miss_key_to_slot
                .entry(ZeroMassAuthorityKey::new(event, context))
                .or_insert(next_miss_slot);
            row_to_miss_slot.push(miss_slot);
        }
        let anchor_slots = (0..anchor_key_to_slot.len())
            .map(|_| OnceLock::new())
            .collect();
        let miss_slots = (0..miss_key_to_slot.len())
            .map(|_| OnceLock::new())
            .collect();
        Self {
            anchor_slots,
            miss_slots,
            row_to_anchor_slot,
            row_to_miss_slot,
        }
    }

    /// Pre-initialize the distinct HF anchor slots slot-major, in parallel,
    /// before a parallel row pass.
    ///
    /// `AnchorAuthorityKey` is target-only, so every row of one event shares
    /// ONE anchor slot; a flat row `par_iter` then convoys at batch start —
    /// the first thread into `OnceLock::get_or_init` runs a full HF target
    /// propagation while std blocks every other thread parked on the same
    /// slot. Running the SAME pure initializer (same representative inputs,
    /// same `zero_mass_reference_for_event_uncached`) once per DISTINCT slot
    /// first keeps the row pass wait-free, and any slot this pass declines is
    /// simply initialized by the row pass exactly as before, so values are
    /// bit-identical by construction either way.
    ///
    /// Declined slots, mirroring `solve_single_event_hf_internal`'s
    /// pre-anchor prefix: non-HF-ready contexts (their LF anchor init is a
    /// single analytic conversion — no convoy worth hoisting),
    /// `hf_validate_only` rows (they pick their own route), and failed HF
    /// preparation (strict rows abort before the anchor; non-strict rows
    /// degrade to the LF anchor). Preparation here is deliberately unlatched:
    /// only the row-owned preparation may publish a preflight failure, keeping
    /// its count exactly once without moving preparation after the row's next
    /// failure gate.
    fn preinitialize_hf_anchor_slots_parallel(&self, rows: &[(MassSolverEvent, HfContext)]) {
        use rayon::prelude::*;

        let mut representative_rows: Vec<Option<usize>> = vec![None; self.anchor_slots.len()];
        for (row, &slot) in self.row_to_anchor_slot.iter().enumerate() {
            if let Some(representative) = representative_rows.get_mut(slot) {
                if representative.is_none() {
                    *representative = Some(row);
                }
            }
        }
        representative_rows
            .par_iter()
            .zip(self.anchor_slots.par_iter())
            .for_each(|(representative, slot)| {
                let Some(row) = *representative else {
                    return;
                };
                let Some((event, hf_ctx)) = rows.get(row) else {
                    return;
                };
                if !hf_ctx.is_hf_ready() || hf_ctx.hf_validate_only {
                    return;
                }
                let Some(Ok(prepared)) = prepare_hf_for_event_unlatched(event, hf_ctx) else {
                    return;
                };
                slot.get_or_init(|| {
                    zero_mass_reference_for_event_uncached(
                        event,
                        Some(&prepared),
                        &mut UnobservedMassSolve,
                    )
                });
            });
    }

    #[inline]
    fn slot_for_row(&self, row: usize) -> Option<ZeroMassCacheView<'_>> {
        let anchor_slot = *self.row_to_anchor_slot.get(row)?;
        let miss_slot = *self.row_to_miss_slot.get(row)?;
        Some(ZeroMassCacheView {
            anchor_reference: self.anchor_slots.get(anchor_slot)?,
            miss_at_zero: self.miss_slots.get(miss_slot)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetPropagationAuthority {
    HighFidelity,
    MfJ2,
    AnalyticalKepler,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetMassPropagationError {
    HighFidelityContextRequired,
    InvalidOrbit,
    NonFinite,
    IntegrationFailed(FinalPropagationFailure),
}

fn propagate_target_for_mass_authority<O: MassSolveObserver>(
    state: &[f64; 6],
    tof_s: f64,
    authority: TargetPropagationAuthority,
    prepared_hf: Option<&PreparedHfConfig>,
    observer: &mut O,
    leg: MassLegTag,
) -> Result<[f64; 6], TargetMassPropagationError> {
    let mut equ = [0.0; 6];
    eci2equinoc_impl(state, 6, 0.0, 0.0, &mut equ);
    if !equ[0].is_finite() || equ[0] <= 0.0 {
        return Err(TargetMassPropagationError::InvalidOrbit);
    }

    let mut propagated = [0.0; 6];
    match authority {
        TargetPropagationAuthority::AnalyticalKepler => {
            equinoc_prop_from_impl(&equ, tof_s, &mut propagated);
        }
        TargetPropagationAuthority::MfJ2 => {
            equinoc_prop_j2_from_impl(&equ, tof_s, &mut propagated);
        }
        TargetPropagationAuthority::HighFidelity => {
            let prepared =
                prepared_hf.ok_or(TargetMassPropagationError::HighFidelityContextRequired)?;
            let context = prepared.scalar_propagation_context();
            let final_times = [tof_s];
            let request = ScalarPropagationRequest::new(&context, equ, &final_times, 0.0, tof_s)
                .with_events(true);
            let final_delta = observer
                .integrate_final_leg(
                    request,
                    &MassLegIdentity {
                        tag: leg,
                        source_jd_bits: prepared.epoch_jd.to_bits(),
                        initial_eci_bits: state.map(f64::to_bits),
                        initial_equinoctial_bits: equ.map(f64::to_bits),
                        t0_s_bits: 0.0_f64.to_bits(),
                        t_final_s_bits: tof_s.to_bits(),
                    },
                )
                .map_err(TargetMassPropagationError::IntegrationFailed)?;

            equinoc_prop_from_impl(&equ, tof_s, &mut propagated);
            for (value, delta) in propagated.iter_mut().zip(final_delta) {
                *value += delta;
            }
        }
    }

    if propagated.iter().all(|value| value.is_finite()) {
        Ok(propagated)
    } else {
        Err(TargetMassPropagationError::NonFinite)
    }
}

const EXACT_MASS_MEMO_CAPACITY: usize = 128;
const EXACT_MASS_MEMO_TABLE_SIZE: usize = 256;
const EXACT_MASS_MEMO_TABLE_MASK: usize = EXACT_MASS_MEMO_TABLE_SIZE - 1;
const EXACT_MASS_MEMO_SLOT_EMPTY: u8 = 0;
const EXACT_MASS_MEMO_SLOT_OCCUPIED: u8 = 1;
const EXACT_MASS_MEMO_SLOT_TOMBSTONE: u8 = 2;

#[inline]
fn exact_mass_memo_hash_index(key: u64) -> usize {
    debug_assert!(EXACT_MASS_MEMO_TABLE_SIZE.is_power_of_two());
    let mixed = key ^ (key >> 33);
    usize::try_from(mixed.wrapping_mul(0x9E37_79B1_85EB_CA87)).unwrap_or_default()
        & EXACT_MASS_MEMO_TABLE_MASK
}

#[derive(Clone)]
struct ExactMassMissMemo {
    slot_states: [u8; EXACT_MASS_MEMO_TABLE_SIZE],
    mass_bits: [u64; EXACT_MASS_MEMO_TABLE_SIZE],
    miss_values: [f64; EXACT_MASS_MEMO_TABLE_SIZE],
    insertion_slots: [usize; EXACT_MASS_MEMO_CAPACITY],
    len: usize,
    next_slot: usize,
}

impl Default for ExactMassMissMemo {
    fn default() -> Self {
        Self {
            slot_states: [EXACT_MASS_MEMO_SLOT_EMPTY; EXACT_MASS_MEMO_TABLE_SIZE],
            mass_bits: [0; EXACT_MASS_MEMO_TABLE_SIZE],
            miss_values: [0.0; EXACT_MASS_MEMO_TABLE_SIZE],
            insertion_slots: [0; EXACT_MASS_MEMO_CAPACITY],
            len: 0,
            next_slot: 0,
        }
    }
}

impl ExactMassMissMemo {
    #[inline]
    fn find_existing_slot(&self, key: u64) -> Option<usize> {
        let start = exact_mass_memo_hash_index(key);
        for probe in 0..EXACT_MASS_MEMO_TABLE_SIZE {
            let idx = start.wrapping_add(probe) & EXACT_MASS_MEMO_TABLE_MASK;
            match self.slot_states.get(idx).copied() {
                Some(EXACT_MASS_MEMO_SLOT_EMPTY) => return None,
                Some(EXACT_MASS_MEMO_SLOT_OCCUPIED)
                    if self.mass_bits.get(idx).is_some_and(|stored| *stored == key) =>
                {
                    return Some(idx);
                }
                _ => {}
            }
        }
        None
    }

    #[inline]
    fn find_vacant_slot(&self, key: u64) -> usize {
        let start = exact_mass_memo_hash_index(key);
        let mut first_tombstone: Option<usize> = None;

        for probe in 0..EXACT_MASS_MEMO_TABLE_SIZE {
            let idx = start.wrapping_add(probe) & EXACT_MASS_MEMO_TABLE_MASK;
            match self.slot_states.get(idx).copied() {
                Some(EXACT_MASS_MEMO_SLOT_EMPTY) => return first_tombstone.unwrap_or(idx),
                Some(EXACT_MASS_MEMO_SLOT_TOMBSTONE) if first_tombstone.is_none() => {
                    first_tombstone = Some(idx);
                }
                _ => {}
            }
        }
        // Bounded fallback when table consists entirely of occupied/tombstone slots.
        first_tombstone.unwrap_or(start)
    }

    #[inline]
    fn get(&self, mass: f64) -> Option<f64> {
        let key = mass.to_bits();
        self.find_existing_slot(key)
            .and_then(|idx| self.miss_values.get(idx).copied())
    }

    #[inline]
    fn insert(&mut self, mass: f64, miss_distance: f64) {
        let key = mass.to_bits();
        if let Some(value) = self
            .find_existing_slot(key)
            .and_then(|idx| self.miss_values.get_mut(idx))
        {
            *value = miss_distance;
            return;
        }

        let replacement_order_slot = if self.len == EXACT_MASS_MEMO_CAPACITY {
            let order_slot = self.next_slot;
            let Some(&evict_idx) = self.insertion_slots.get(order_slot) else {
                return;
            };
            debug_assert_eq!(
                self.slot_states.get(evict_idx).copied(),
                Some(EXACT_MASS_MEMO_SLOT_OCCUPIED),
                "FIFO order must only point at occupied memo entries"
            );
            let Some(evicted_state) = self.slot_states.get_mut(evict_idx) else {
                return;
            };
            *evicted_state = EXACT_MASS_MEMO_SLOT_TOMBSTONE;
            Some(order_slot)
        } else {
            None
        };

        let idx = self.find_vacant_slot(key);
        let Some(slot_state) = self.slot_states.get_mut(idx) else {
            return;
        };
        *slot_state = EXACT_MASS_MEMO_SLOT_OCCUPIED;
        let Some(stored_mass) = self.mass_bits.get_mut(idx) else {
            return;
        };
        *stored_mass = key;
        let Some(stored_miss_distance) = self.miss_values.get_mut(idx) else {
            return;
        };
        *stored_miss_distance = miss_distance;

        if let Some(order_slot) = replacement_order_slot {
            let Some(stored_slot) = self.insertion_slots.get_mut(order_slot) else {
                return;
            };
            *stored_slot = idx;
            self.next_slot = order_slot.wrapping_add(1) % EXACT_MASS_MEMO_CAPACITY;
        } else {
            let Some(stored_slot) = self.insertion_slots.get_mut(self.len) else {
                return;
            };
            *stored_slot = idx;
            self.len = self.len.saturating_add(1);
        }
    }
}

#[inline]
fn memoized_exact_mass_eval<F>(mass: f64, memo: &mut ExactMassMissMemo, eval: F) -> f64
where
    F: FnOnce() -> f64,
{
    if let Some(cached) = memo.get(mass) {
        lightyear_odeint_rs::probe::bump_stage(lightyear_odeint_rs::probe::STAGE_MEMO_HITS);
        return cached;
    }
    let computed = eval();
    memo.insert(mass, computed);
    computed
}

impl Default for HfContext {
    fn default() -> Self {
        Self {
            use_high_fidelity: false,
            epoch_jd: 0.0,
            force_config: None,
            packed_coeffs: None,
            hf_validate_only: dust_hf_validate_only_enabled(),
            hf_strict: v3_strict_hf_enabled(),
        }
    }
}

impl HfContext {
    /// Check if HF mode is enabled and all required components are present.
    #[inline]
    #[must_use]
    pub const fn is_hf_ready(&self) -> bool {
        self.use_high_fidelity && self.force_config.is_some() && self.packed_coeffs.is_some()
    }
}

/// Event data for mass solving.
///
/// All positions are in **km** (`ECI J2000`), velocities in **km/s**,
/// masses in **kg**, times in **s**. Momentum uses **kg·km/s** (mass × velocity).
#[derive(Clone, Debug)]
pub struct MassSolverEvent {
    /// Primary momentum vector `[px, py, pz]` in kg·km/s
    /// (= `p_mass × velocity_km_s`).
    pub p_momentum: [f64; 3],
    /// Dust cloud velocity vector [vx, vy, vz] in km/s (ECI).
    pub dv_vec: [f64; 3],
    /// Primary object mass in kg.
    pub p_mass: f64,
    /// Primary ECI position at dust intercept [x, y, z] in km.
    pub p_pos_intercept: [f64; 3],
    /// Time of flight from dust intercept to conjunction epoch in s.
    pub tof_s: f64,
    /// Secondary object ECI position at conjunction [x, y, z] in km.
    pub secondary_conj_pos: [f64; 3],
    /// Target minimum miss distance (root-finding goal) in km.
    pub min_miss_distance_km: f64,
    /// Parametric momentum-transfer factor (dimensionless). D3 bounds κ = 1
    /// conservatism to the current monotone model; larger values are parametric.
    pub kappa: f64,
    /// Truth-dynamics baseline primary position at conjunction [x, y, z] in km.
    /// From HF or interpolated state table; used for anchored differential correction.
    pub p_pos_conj_truth: [f64; 3],
    /// Fallback zero-mass baseline position at conjunction [x, y, z] in km.
    /// Used only when checked internal propagation cannot produce an anchor.
    pub p_pos_conj_equ_0: [f64; 3],
    /// Precomputed primary velocity `[vx, vy, vz]` in km/s (= `p_momentum / p_mass`).
    /// Cached to avoid repeated division in bisection hot loop.
    pub p_velocity: [f64; 3],
    /// Precomputed relative velocity `[vx, vy, vz]` in km/s (= `dv_vec - p_velocity`).
    /// Cached to avoid repeated subtraction in bisection hot loop.
    pub v_rel: [f64; 3],
    /// Primary equinoctial state at intercept `[a, h, k, p, q, L]`.
    /// Units: `a` in km, `h/k/p/q` dimensionless, `L` in rad. Used for Encke HF propagation.
    pub p_equ_intercept: [f64; 6],
    /// Per-event area-to-mass ratio override in m²/kg (optional; overrides [`ForceConfig`]).
    pub p_am_ratio: Option<f64>,
    /// Per-event drag coefficient override (dimensionless; optional).
    pub p_cd: Option<f64>,
    /// Per-event SRP reflectivity coefficient override (dimensionless; optional).
    pub p_cr: Option<f64>,
    /// Per-event charge-to-mass ratio override in C/kg (optional).
    pub p_qm_ratio: Option<f64>,
    /// Per-event characteristic object radius override in meters (optional).
    pub p_r_obj_m: Option<f64>,
}

/// Solver configuration for dust mass root-finding (Brent's method).
pub struct SolverConfig {
    /// Absolute convergence tolerance for mass in kg.
    /// Solver stops when bracket width < xtol.
    pub xtol: f64,
    /// Convergence tolerance for miss distance. Solver stops when
    /// `|f(b)| < rtol * target.max(1.0)`, with `target` in km.
    ///
    /// NOT dimensionless in practice, and this doc said it was until 2026-08-09.
    /// Every miss-distance target in the production corpus is under the 1.0 km
    /// floor, so `max` returns the floor on every row and `rtol` reaches the
    /// comparison as an ABSOLUTE tolerance in kilometres: the shipped 1e-5 is
    /// one centimetre, and the 1e-6 it replaced was one millimetre. The two
    /// budgets that fix the value are on the `Default` impl below; read them
    /// before moving it, and read them as kilometres.
    pub rtol: f64,
    /// Maximum number of solver iterations.
    pub maxiter: usize,
    /// Upper bound for mass search in kg.
    /// Physics-limited events are capped at this value.
    pub mass_max: f64,
}

/// Event data for the Python MF-J2 deterministic-mass seed path.
///
/// This intentionally mirrors the current Python `_j2_miss_distance_for_mass`
/// semantics rather than the anchored LF/HF solver event model.
#[derive(Clone, Copy, Debug)]
pub struct MfJ2MassSolverEvent {
    pub target_pos_intercept: [f64; 3],
    pub target_vel_intercept: [f64; 3],
    pub dv_vec: [f64; 3],
    pub p_mass: f64,
    pub other_conj_pos: [f64; 3],
    pub tof_s: f64,
    pub min_miss_distance_km: f64,
    pub kappa: f64,
    pub v_rel: [f64; 3],
    pub mu_over_r_intercept: f64,
    /// The INTERCEPTED target's own catalogue position at the conjunction
    /// epoch, when the caller can supply it.
    ///
    /// `None` reproduces the historical formula bit-for-bit. `Some` switches
    /// `compute_miss_distance_mf_j2` to an anchored differential -- see that
    /// function -- which is required once the catalogue's conjunction is not
    /// the MF-J2 image of its own intercept state.
    pub target_conj_pos: Option<[f64; 3]>,
}

impl MfJ2MassSolverEvent {
    /// Supply the intercepted target's own catalogue conjunction position,
    /// switching the miss distance to the anchored differential.
    #[must_use]
    pub const fn with_conjunction_anchor(mut self, target_conj_pos: [f64; 3]) -> Self {
        self.target_conj_pos = Some(target_conj_pos);
        self
    }

    #[must_use]
    pub fn new(
        target_pos_intercept: [f64; 3],
        target_vel_intercept: [f64; 3],
        dv_vec: [f64; 3],
        p_mass: f64,
        other_conj_pos: [f64; 3],
        tof_s: f64,
        min_miss_distance_km: f64,
        kappa: f64,
    ) -> Self {
        let v_rel = [
            dv_vec[0] - target_vel_intercept[0],
            dv_vec[1] - target_vel_intercept[1],
            dv_vec[2] - target_vel_intercept[2],
        ];
        let r_norm = (target_pos_intercept[0].mul_add(
            target_pos_intercept[0],
            target_pos_intercept[1].mul_add(
                target_pos_intercept[1],
                target_pos_intercept[2] * target_pos_intercept[2],
            ),
        ))
        .sqrt();
        let mu_over_r_intercept = if r_norm.is_finite() && r_norm > 1e-12 {
            MU / r_norm
        } else {
            f64::NAN
        };
        Self {
            target_pos_intercept,
            target_vel_intercept,
            dv_vec,
            p_mass,
            other_conj_pos,
            tof_s,
            min_miss_distance_km,
            kappa,
            v_rel,
            mu_over_r_intercept,
            target_conj_pos: None,
        }
    }
}

/// Last underlying failure behind a `MissAtZeroHfIntegrateFailure`, kept only
/// so an error report can say WHICH integrator failure the compact status code
/// stands for. The status code is the sole production channel and stays
/// unchanged; this slot is written on that failure path alone, is
/// last-writer-wins under parallel solves, and must never gate any decision.
static LAST_MISS_AT_ZERO_HF_INTEGRATE_FAILURE: std::sync::Mutex<Option<String>> =
    std::sync::Mutex::new(None);

fn record_miss_at_zero_hf_integrate_failure(
    failure: lightyear_odeint_rs::integrator::FinalPropagationFailure,
) {
    if let Ok(mut slot) = LAST_MISS_AT_ZERO_HF_INTEGRATE_FAILURE.lock() {
        *slot = Some(format!("{failure:?}"));
    }
}

/// Take (and clear) the diagnostic detail behind the most recent
/// `MissAtZeroHfIntegrateFailure` in this process.
///
/// Diagnostic context only — last-writer-wins, so callers must pair it with
/// the status code they just observed and treat a stale or absent entry as
/// "no detail".
pub fn take_last_miss_at_zero_hf_integrate_failure() -> Option<String> {
    LAST_MISS_AT_ZERO_HF_INTEGRATE_FAILURE
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
}

#[inline]
fn diagnose_miss_at_zero_failure<O: MassSolveObserver>(
    event: &MassSolverEvent,
    prepared_hf: Option<&PreparedHfConfig>,
    observer: &mut O,
) -> MassSolveStatusCode {
    let new_vel = compute_new_velocity(0.0, event);
    if !new_vel[0].is_finite() || !new_vel[1].is_finite() || !new_vel[2].is_finite() {
        return MassSolveStatusCode::MissAtZeroInvalidVelocity;
    }

    let eci_state = [
        event.p_pos_intercept[0],
        event.p_pos_intercept[1],
        event.p_pos_intercept[2],
        new_vel[0],
        new_vel[1],
        new_vel[2],
    ];

    if let Some(prepared) = prepared_hf {
        let min_radius_km =
            prepared.force_config.earth_radius + lightyear_odeint_rs::types::GROUND_ALTITUDE;
        if !state_clears_min_radius(&eci_state, min_radius_km) {
            return MassSolveStatusCode::HfTrajectoryPhysicallyInfeasible;
        }
    }

    let mut equ = [0.0; 6];
    eci2equinoc_impl(&eci_state, 6, 0.0, 0.0, &mut equ);
    if !equ[0].is_finite() || equ[0] <= 0.0 {
        return MassSolveStatusCode::MissAtZeroInvalidOrbit;
    }

    if let Some(prepared) = prepared_hf {
        let context = prepared.scalar_propagation_context();
        let final_times = [event.tof_s];
        let request = ScalarPropagationRequest::new(&context, equ, &final_times, 0.0, event.tof_s)
            .with_events(true);
        // This is the distinct production zero-mass diagnostic final call.
        let final_delta_result = observer.integrate_final_leg(
            request,
            &MassLegIdentity {
                tag: MassLegTag {
                    role: MassLegRole::ZeroMassDiagnostic,
                    mass_kg_bits: 0.0_f64.to_bits(),
                },
                source_jd_bits: prepared.epoch_jd.to_bits(),
                initial_eci_bits: eci_state.map(f64::to_bits),
                initial_equinoctial_bits: equ.map(f64::to_bits),
                t0_s_bits: 0.0_f64.to_bits(),
                t_final_s_bits: event.tof_s.to_bits(),
            },
        );
        let final_delta = match final_delta_result {
            Ok(d) => d,
            Err(failure) if failure.is_physical_infeasible() => {
                return MassSolveStatusCode::HfTrajectoryPhysicallyInfeasible;
            }
            // Checked BEFORE the catch-all, because the catch-all is what lost
            // it: an authority refusal means nothing was integrated, so calling
            // it an integrate failure describes a propagation that never ran.
            Err(failure) if failure.is_authority_refusal() => {
                return MassSolveStatusCode::HfAuthorityRefused;
            }
            Err(failure) => {
                record_miss_at_zero_hf_integrate_failure(failure);
                return MassSolveStatusCode::MissAtZeroHfIntegrateFailure;
            }
        };

        let mut baseline = [0.0; 6];
        equinoc_prop_from_impl(&equ, event.tof_s, &mut baseline);
        let propagated = [
            baseline[0] + final_delta[0],
            baseline[1] + final_delta[1],
            baseline[2] + final_delta[2],
            baseline[3] + final_delta[3],
            baseline[4] + final_delta[4],
            baseline[5] + final_delta[5],
        ];
        if !propagated[0].is_finite() {
            return MassSolveStatusCode::MissAtZeroPropagateNonFinite;
        }
    } else {
        let mut propagated = [0.0; 6];
        equinoc2eci_impl(&equ, 6, event.tof_s, 0.0, &mut propagated);
        if !propagated[0].is_finite() {
            return MassSolveStatusCode::MissAtZeroPropagateNonFinite;
        }
    }
    MassSolveStatusCode::MissAtZeroNonFinite
}

const DETMASS_ANCHOR_CONTRACT_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug)]
struct EventDerived {
    /// Precomputed μ/r at intercept (km²/s²) for hyperbolic-energy checks.
    mu_over_r_intercept: f64,
    /// Constant anchored baseline shift: truth - `internal_zero_mass_reference` (km).
    anchor_shift: [f64; 3],
    /// Whether anchored correction is materially non-zero.
    apply_anchor_shift: bool,
    /// Anchor shift L2 norm in km (diagnostic).
    anchor_shift_norm_km: f64,
    /// Anchor-contract version token for diagnostics.
    anchor_contract_version: u32,
    /// True when internal zero-mass reference was available.
    anchor_internal_reference_used: bool,
}

#[inline]
fn derive_event_invariants(
    event: &MassSolverEvent,
    zero_mass_reference: Option<[f64; 3]>,
) -> EventDerived {
    let r_sq = vec3_norm_sq(&event.p_pos_intercept);
    let mu_over_r_intercept = if r_sq > 0.0 {
        MU / r_sq.sqrt()
    } else {
        f64::NAN
    };
    let baseline = zero_mass_reference.unwrap_or(event.p_pos_conj_equ_0);
    let shift = [
        event.p_pos_conj_truth[0] - baseline[0],
        event.p_pos_conj_truth[1] - baseline[1],
        event.p_pos_conj_truth[2] - baseline[2],
    ];
    let shift_sq = shift[0].mul_add(shift[0], shift[1].mul_add(shift[1], shift[2] * shift[2]));
    let shift_norm = shift_sq.sqrt();
    EventDerived {
        mu_over_r_intercept,
        anchor_shift: shift,
        apply_anchor_shift: shift_sq > 1e-12,
        anchor_shift_norm_km: shift_norm,
        anchor_contract_version: DETMASS_ANCHOR_CONTRACT_VERSION,
        anchor_internal_reference_used: zero_mass_reference.is_some(),
    }
}

#[inline]
fn zero_mass_reference_lf_from_intercept_equ(event: &MassSolverEvent) -> Option<[f64; 3]> {
    if !event.p_equ_intercept[0].is_finite() || event.p_equ_intercept[0] <= 0.0 {
        return None;
    }
    let mut propagated = [0.0; 6];
    equinoc2eci_impl(&event.p_equ_intercept, 6, event.tof_s, 0.0, &mut propagated);
    if !propagated[0].is_finite() || !propagated[1].is_finite() || !propagated[2].is_finite() {
        return None;
    }
    Some([propagated[0], propagated[1], propagated[2]])
}

#[inline]
fn zero_mass_reference_hf_from_intercept_eci<O: MassSolveObserver>(
    event: &MassSolverEvent,
    prepared: &PreparedHfConfig,
    observer: &mut O,
) -> Option<[f64; 3]> {
    let state = zero_mass_eci_state(event);
    let min_radius_km =
        prepared.force_config.earth_radius + lightyear_odeint_rs::types::GROUND_ALTITUDE;
    if !state_clears_min_radius(&state, min_radius_km) {
        return None;
    }
    let propagated = propagate_target_for_mass_authority(
        &state,
        event.tof_s,
        TargetPropagationAuthority::HighFidelity,
        Some(prepared),
        observer,
        MassLegTag {
            role: MassLegRole::ZeroMassAnchor,
            mass_kg_bits: 0.0_f64.to_bits(),
        },
    )
    .ok()?;
    Some([propagated[0], propagated[1], propagated[2]])
}

#[inline]
fn zero_mass_reference_for_event<O: MassSolveObserver>(
    event: &MassSolverEvent,
    prepared: Option<&PreparedHfConfig>,
    cache: Option<ZeroMassCacheView<'_>>,
    observer: &mut O,
) -> ZeroMassReference {
    let Some(slot) = cache else {
        return zero_mass_reference_for_event_uncached(event, prepared, observer);
    };
    *slot
        .anchor_reference
        .get_or_init(|| zero_mass_reference_for_event_uncached(event, prepared, observer))
}

fn zero_mass_reference_for_event_uncached<O: MassSolveObserver>(
    event: &MassSolverEvent,
    prepared: Option<&PreparedHfConfig>,
    observer: &mut O,
) -> ZeroMassReference {
    prepared.map_or_else(
        || ZeroMassReference {
            position: zero_mass_reference_lf_from_intercept_equ(event),
            exact_hf: false,
        },
        |prepared| {
            let probed = {
                let _probe =
                    lightyear_odeint_rs::probe::scope(lightyear_odeint_rs::probe::TAG_ZERO_MASS);
                zero_mass_reference_hf_from_intercept_eci(event, prepared, observer)
            };
            probed.map_or(
                ZeroMassReference {
                    position: None,
                    exact_hf: false,
                },
                |position| ZeroMassReference {
                    position: Some(position),
                    exact_hf: true,
                },
            )
        },
    )
}

#[inline]
fn exact_zero_mass_miss(
    reference: ZeroMassReference,
    derived: &EventDerived,
    secondary_conj_pos: &[f64; 3],
) -> Option<f64> {
    if !reference.exact_hf {
        return None;
    }
    reference.position.map(|position| {
        vec3_distance(
            &apply_anchored_adjustment(position, derived),
            secondary_conj_pos,
        )
    })
}

#[inline]
const fn classify_fixed_impact_hf(
    miss_distance_km: f64,
    threshold_km: f64,
) -> FixedImpactHfVerdict {
    if miss_distance_km > threshold_km {
        FixedImpactHfVerdict::Safe
    } else {
        FixedImpactHfVerdict::Miss
    }
}

#[inline]
const fn fixed_impact_failure_from_target(
    failure: TargetMassPropagationError,
) -> FixedImpactHfFailure {
    match failure {
        TargetMassPropagationError::HighFidelityContextRequired => {
            FixedImpactHfFailure::HighFidelityContextRequired
        }
        TargetMassPropagationError::InvalidOrbit => FixedImpactHfFailure::InvalidOrbit,
        TargetMassPropagationError::NonFinite => FixedImpactHfFailure::NonFinite,
        TargetMassPropagationError::IntegrationFailed(failure) => match failure {
            FinalPropagationFailure::Ground => FixedImpactHfFailure::Ground,
            FinalPropagationFailure::LeftEarth | FinalPropagationFailure::Eccentricity => {
                FixedImpactHfFailure::Escape
            }
            FinalPropagationFailure::NanState => FixedImpactHfFailure::NonFinite,
            failure => FixedImpactHfFailure::Integration(failure),
        },
    }
}

/// Propagate one exact sampled retained mass directly from its drawn state.
///
/// Uses the same retained-body impulse, fixed-area area-to-mass adjustment,
/// and strict-HF target propagation authority as the deterministic mass solver.
/// The request's sampled mass is never interpreted as an expectation or
/// resampled inside this function.
///
/// # Errors
///
/// Returns a typed fail-closed physical, numerical, or authority failure.
pub fn evaluate_fixed_impact_hf(
    request: &FixedImpactHfRequest,
) -> Result<FixedImpactHfOutcome, FixedImpactHfFailure> {
    evaluate_fixed_impact_hf_with_observer(request, &mut UnobservedMassSolve)
}

fn evaluate_fixed_impact_hf_with_observer<O: MassSolveObserver>(
    request: &FixedImpactHfRequest,
    observer: &mut O,
) -> Result<FixedImpactHfOutcome, FixedImpactHfFailure> {
    let event = &request.event;
    let prepared = match prepare_hf_for_event(event, &request.context) {
        Some(Ok(prepared)) => prepared,
        Some(Err(())) => return Err(FixedImpactHfFailure::HighFidelityPreparationFailed),
        None => return Err(FixedImpactHfFailure::HighFidelityContextRequired),
    };
    let sampled_mass_kg = request.sampled_retained_mass_kg;
    let new_velocity = compute_new_velocity(sampled_mass_kg, event);
    if !new_velocity.iter().all(|value| value.is_finite()) {
        return Err(FixedImpactHfFailure::NonFinite);
    }
    let retained_prepared = prepared
        .for_retained_mass(sampled_mass_kg, event)
        .ok_or(FixedImpactHfFailure::NonFinite)?;

    let radius_km = norm3(&event.p_pos_intercept);
    if !radius_km.is_finite() || radius_km <= 0.0 {
        return Err(FixedImpactHfFailure::InvalidOrbit);
    }
    let mu_over_r = MU / radius_km;
    let specific_energy = 0.5 * vec3_norm_sq(&new_velocity) - mu_over_r;
    if !specific_energy.is_finite() {
        return Err(FixedImpactHfFailure::NonFinite);
    }
    if specific_energy_is_escaped(specific_energy, mu_over_r) {
        return Err(FixedImpactHfFailure::Escape);
    }

    let impact_state_eci = [
        event.p_pos_intercept[0],
        event.p_pos_intercept[1],
        event.p_pos_intercept[2],
        new_velocity[0],
        new_velocity[1],
        new_velocity[2],
    ];
    let min_radius_km =
        retained_prepared.force_config.earth_radius + lightyear_odeint_rs::types::GROUND_ALTITUDE;
    if !min_radius_km.is_finite() {
        return Err(FixedImpactHfFailure::NonFinite);
    }
    if !state_clears_min_radius(&impact_state_eci, min_radius_km) {
        return Err(FixedImpactHfFailure::Ground);
    }

    let conjunction_state_eci = propagate_target_for_mass_authority(
        &impact_state_eci,
        event.tof_s,
        TargetPropagationAuthority::HighFidelity,
        Some(&retained_prepared),
        observer,
        MassLegTag {
            role: MassLegRole::MassEvaluation,
            mass_kg_bits: sampled_mass_kg.to_bits(),
        },
    )
    .map_err(fixed_impact_failure_from_target)?;
    if !conjunction_state_eci.iter().all(|value| value.is_finite()) {
        return Err(FixedImpactHfFailure::NonFinite);
    }
    let conjunction_position = [
        conjunction_state_eci[0],
        conjunction_state_eci[1],
        conjunction_state_eci[2],
    ];
    let miss_distance_km = vec3_distance(&conjunction_position, &event.secondary_conj_pos);
    if !miss_distance_km.is_finite() {
        return Err(FixedImpactHfFailure::NonFinite);
    }

    Ok(FixedImpactHfOutcome {
        verdict: classify_fixed_impact_hf(miss_distance_km, event.min_miss_distance_km),
        miss_distance_km,
        sampled_retained_mass_kg: sampled_mass_kg,
        retained_area_to_mass_ratio_m2_per_kg: retained_prepared.force_config.am_ratio,
        deterministic_mass_min_distance_km: event.min_miss_distance_km,
        impact_state_eci,
        conjunction_state_eci,
    })
}

impl Default for SolverConfig {
    fn default() -> Self {
        // Defaults aligned with physics.deterministic_mass_solver in
        // dissertation_production.yaml (single source of truth).
        Self {
            xtol: 1e-6,
            // 1e-6 -> 1e-5 (2026-08-06). `rtol` scales `target.max(1.0)` and
            // every miss-distance target in the production corpus is under
            // 3.16 km, so the floor binds on every row and `rtol` is really an
            // ABSOLUTE tolerance in kilometres: the shipped 1e-6 asked the
            // solver to resolve a miss distance to one millimetre. Two
            // independent budgets put the floor three to four orders of
            // magnitude above that, and they agree with each other:
            //
            //   - Arc accuracy. `strict_hf_production_arc_accuracy` commits to
            //     3 m and delivers 1.2 cm. A residual below the delivered
            //     accuracy of the trajectory that produced it is not a
            //     measurement of anything. -> 1.2e-5 km.
            //   - `xtol` consistency. Brent already declares itself indifferent
            //     to masses within 0.5 * xtol = 5e-7 kg. Converting that to a
            //     miss distance through the WORST measured sensitivity in the
            //     corpus (`PROP_MASSSENS min_km_per_kg` = 38.34 km/kg) gives
            //     5e-7 * 38.34 = 1.9e-5 km. At the mean sensitivity
            //     (2.373e3 km/kg) the same argument allows 1.2e-3 km.
            //
            // 1e-5 is the tighter of the two anchors rounded down, so no row in
            // the corpus can be retired on a residual that admits a mass error
            // larger than the interval `xtol` already accepts. Measured cost of
            // the shipped value: 354 of 3,841 validate-refine propagations
            // existed only to drive the residual from 1e-5 km to 1e-6 km.
            // Measured on the short-span `h0` base; §14 of the audit carries
            // the paired before/after and the accuracy displacement.
            rtol: 1e-5,
            maxiter: 80,
            mass_max: 1.0e6,
        }
    }
}

/// Compute new velocity after dust momentum transfer.
///
/// Implements momentum transfer:
/// `v_new = v_old + κ × (m_dust / (m_p + m_d)) × v_rel`.
///
/// This follows from conservation of momentum in an inelastic collision,
/// with κ accounting for momentum enhancement from ejecta. See DART mission
/// results: Cheng et al. (2023), Nature 616, 457-460.
///
/// Uses precomputed `p_velocity` and `v_rel` from [`MassSolverEvent`] to avoid
/// repeated division in bisection loop (Phase 3 optimization).
#[inline]
fn compute_new_velocity(mass: f64, event: &MassSolverEvent) -> [f64; 3] {
    let total_mass = event.p_mass + mass;
    if total_mass < 1e-9 {
        return [f64::NAN; 3];
    }

    let factor = event.kappa * mass / total_mass;
    [
        event.p_velocity[0] + factor * event.v_rel[0],
        event.p_velocity[1] + factor * event.v_rel[1],
        event.p_velocity[2] + factor * event.v_rel[2],
    ]
}

#[inline]
fn zero_mass_eci_state(event: &MassSolverEvent) -> [f64; 6] {
    let velocity = compute_new_velocity(0.0, event);
    [
        event.p_pos_intercept[0],
        event.p_pos_intercept[1],
        event.p_pos_intercept[2],
        velocity[0],
        velocity[1],
        velocity[2],
    ]
}

#[inline]
fn osculating_perigee_km(state: &[f64; 6]) -> Option<f64> {
    if !state.iter().all(|value| value.is_finite()) {
        return None;
    }
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let r_norm = norm3(&r);
    let h = cross3(&r, &v);
    let h_sq = h[0].mul_add(h[0], h[1].mul_add(h[1], h[2] * h[2]));
    if !(r_norm > 0.0 && h_sq > 0.0) {
        return None;
    }
    let vxh = cross3(&v, &h);
    let e_vec = [
        vxh[0] / MU - r[0] / r_norm,
        vxh[1] / MU - r[1] / r_norm,
        vxh[2] / MU - r[2] / r_norm,
    ];
    let perigee = (h_sq / MU) / (1.0 + norm3(&e_vec));
    (perigee.is_finite() && perigee > 0.0).then_some(perigee)
}

#[inline]
fn state_clears_min_radius(state: &[f64; 6], min_radius_km: f64) -> bool {
    min_radius_km.is_finite()
        && min_radius_km > 0.0
        && norm3(&[state[0], state[1], state[2]]) >= min_radius_km
        && osculating_perigee_km(state).is_some_and(|rp| rp >= min_radius_km)
}

#[inline]
fn compute_new_velocity_mf_j2(mass: f64, event: &MfJ2MassSolverEvent) -> [f64; 3] {
    let total_mass = event.p_mass + mass;
    if total_mass < 1e-9 {
        return [f64::NAN; 3];
    }

    let factor = event.kappa * mass / total_mass;
    [
        event.target_vel_intercept[0] + factor * event.v_rel[0],
        event.target_vel_intercept[1] + factor * event.v_rel[1],
        event.target_vel_intercept[2] + factor * event.v_rel[2],
    ]
}

#[inline]
fn vec3_norm_sq(v: &[f64; 3]) -> f64 {
    {
        let vv = f64x4::new([v[0], v[1], v[2], 0.0]);
        (vv * vv).reduce_add()
    }
}

#[inline]
fn vec3_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx.mul_add(dx, dy.mul_add(dy, dz * dz))).sqrt()
}

/// `baseline` is the zero-mass anchor, used only when the event carries a
/// conjunction anchor. It is constant in mass, so the bisection computes it ONCE
/// and passes it in rather than paying for it on each of its ~30 evaluations;
/// `None` recomputes it on demand.
fn compute_miss_distance_mf_j2(
    mass: f64,
    event: &MfJ2MassSolverEvent,
    baseline: Option<[f64; 3]>,
) -> f64 {
    if !mass.is_finite() || mass < 0.0 || event.tof_s < 0.0 {
        return f64::NAN;
    }

    let new_vel = compute_new_velocity_mf_j2(mass, event);
    if !new_vel[0].is_finite() || !new_vel[1].is_finite() || !new_vel[2].is_finite() {
        return f64::NAN;
    }

    let v_sq = vec3_norm_sq(&new_vel);
    if event.mu_over_r_intercept.is_finite() {
        let specific_energy = 0.5 * v_sq - event.mu_over_r_intercept;
        if specific_energy_is_escaped(specific_energy, event.mu_over_r_intercept) {
            return f64::NAN;
        }
    }

    let eci_state = [
        event.target_pos_intercept[0],
        event.target_pos_intercept[1],
        event.target_pos_intercept[2],
        new_vel[0],
        new_vel[1],
        new_vel[2],
    ];
    let Ok(propagated) = propagate_target_for_mass_authority(
        &eci_state,
        event.tof_s,
        TargetPropagationAuthority::MfJ2,
        None,
        &mut UnobservedMassSolve,
        MassLegTag {
            role: MassLegRole::MassEvaluation,
            mass_kg_bits: mass.to_bits(),
        },
    ) else {
        return f64::NAN;
    };

    let new_pos = [propagated[0], propagated[1], propagated[2]];
    let Some(target_conj_pos) = event.target_conj_pos else {
        // Historical formula, bit-for-bit. Correct ONLY when the catalogue's
        // conjunction is itself the MF-J2 image of the intercept state, which
        // is true of the v2 catalogue and false of v3.
        return vec3_distance(&new_pos, &event.other_conj_pos);
    };

    // ANCHORED DIFFERENTIAL.
    //
    // `new_pos` is an MF-J2 image; `other_conj_pos` is a catalogue position. In
    // v2 those two lived in the same model -- the catalogue's conjunction
    // anchors WERE the secular-J2 images of its start anchors, to 2.07e-10 km --
    // so subtracting them measured the deflection. v3 conjunctions come from
    // strict-HF refinement over the full force model, so the raw difference is
    // dominated by the J2-vs-HF displacement over the time of flight, and every
    // v3 event reported `miss0 >= 1 km` -> `SafeByDefault` (measured
    // 2026-08-19: 0 of 24 events feasible).
    //
    // The deflection is `new_pos - baseline`, where `baseline` is the SAME
    // propagation at zero mass. Both carry the same model displacement, so it
    // cancels in the difference. Adding that deflection to the catalogue's own
    // relative position gives a miss distance in the catalogue's frame:
    //
    //     miss(m) = || (target_conj - other_conj) + (prop(m) - prop(0)) ||
    //
    // At m = 0 this is exactly the catalogue miss distance, by construction.
    let Some(baseline) = baseline.or_else(|| mf_j2_zero_mass_position(event)) else {
        return f64::NAN;
    };
    let anchored = [
        (target_conj_pos[0] - event.other_conj_pos[0]) + (new_pos[0] - baseline[0]),
        (target_conj_pos[1] - event.other_conj_pos[1]) + (new_pos[1] - baseline[1]),
        (target_conj_pos[2] - event.other_conj_pos[2]) + (new_pos[2] - baseline[2]),
    ];
    norm3(&anchored)
}

/// The zero-mass MF-J2 image of the intercept state: the anchor the deflection
/// is measured against.
///
/// Recomputed per evaluation rather than cached on the event, because the event
/// is `Copy` and constructing it must not propagate. The bisection evaluates
/// ~30 times, so this doubles the propagation count on the anchored path; that
/// is the price of the model displacement cancelling, and the path was
/// returning `SafeByDefault` for every row without it.
fn mf_j2_zero_mass_position(event: &MfJ2MassSolverEvent) -> Option<[f64; 3]> {
    let state = [
        event.target_pos_intercept[0],
        event.target_pos_intercept[1],
        event.target_pos_intercept[2],
        event.target_vel_intercept[0],
        event.target_vel_intercept[1],
        event.target_vel_intercept[2],
    ];
    let propagated = propagate_target_for_mass_authority(
        &state,
        event.tof_s,
        TargetPropagationAuthority::MfJ2,
        None,
        &mut UnobservedMassSolve,
        MassLegTag {
            role: MassLegRole::MassEvaluation,
            mass_kg_bits: 0.0_f64.to_bits(),
        },
    )
    .ok()?;
    Some([propagated[0], propagated[1], propagated[2]])
}

/// Conventional reentry interface, in km.
///
/// Above this the deflected target is in orbit; at or below it the "miss
/// distance at conjunction" is a fiction, because the target does not survive
/// to fly the arc being scored.
pub const REENTRY_INTERFACE_ALT_KM: f64 = 100.0;

/// Smallest root of `a x^2 + b x + c` that lies strictly inside `(0, limit)`.
///
/// Returns `None` when the polynomial has no such root — which is the common
/// case here and means "this condition is never reached at any release mass".
fn smallest_root_in_open_interval(a: f64, b: f64, c: f64, limit: f64) -> Option<f64> {
    let mut best = f64::INFINITY;
    let mut consider = |x: f64| {
        if x > 0.0 && x < limit && x < best {
            best = x;
        }
    };
    if a.abs() <= f64::MIN_POSITIVE {
        if b != 0.0 {
            consider(-c / b);
        }
        return best.is_finite().then_some(best);
    }
    let discriminant = b.mul_add(b, -(4.0 * a * c));
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    // Numerically stable pair: one root via the standard formula, the other by
    // Vieta, so the near-cancelling branch is never evaluated directly.
    let stable = -0.5 * (b + b.signum() * root);
    if stable == 0.0 {
        consider((-b + root) / (2.0 * a));
        consider((-b - root) / (2.0 * a));
    } else {
        consider(stable / a);
        consider(c / stable);
    }
    best.is_finite().then_some(best)
}

/// The largest dust release mass (kg) whose momentum transfer still leaves the
/// deflected target in a usable orbit — bounded above by `mass_max`.
///
/// # Why the search needs this
///
/// The deflected target leaves the intercept at
/// `v(m) = v_t + f(m) * v_rel` with `f(m) = kappa*m/(p_mass+m)`, so `f` rises
/// monotonically from 0 to `kappa`. Two things can go wrong as it does, and
/// both make the miss distance at conjunction meaningless rather than large:
/// the orbit can become unbound, or its perigee can drop to the reentry
/// interface. `solve_single_event_mf_j2_with_status` used to bisect over the
/// whole of `[0, mass_max]` regardless. For a light target struck nearly
/// head-on that interval is ~99.96% post-decay, and bisecting it walks the
/// midpoint sequence straight through the region where the equinoctial-to-ECI
/// Kepler solve fails at eccentricity ~0.997 — aborting the solve on rows that
/// have a perfectly good milligram-scale root.
///
/// # How it is computed
///
/// Both conditions are closed-form quadratics in `f`, so no search is needed.
/// Writing `w = v_rel`, `h0 = r x v_t`, `hw = r x w`:
///
/// * unbound: `E(f) = E0 + f (v_t . w) + f^2 |w|^2 / 2 >= 0`;
/// * perigee at `r_p`: substituting `a = -mu/(2E)` and `e^2 = 1 + 2E|h|^2/mu^2`
///   into `a(1-e) = r_p` and clearing the `mu^2/(4E^2)` terms leaves
///   `|h(f)|^2 - 2 r_p^2 E(f) - 2 mu r_p = 0`.
///
/// The ceiling is whichever condition `f` reaches first, converted back by
/// `m = p_mass * f / (kappa - f)`.
///
/// Returns `mass_max` when neither condition is reached within `f < kappa`,
/// and `0.0` when the target is already at or below the interface unperturbed.
#[must_use]
pub fn physical_release_mass_ceiling_kg(
    event: &MfJ2MassSolverEvent,
    mass_max: f64,
    reentry_alt_km: f64,
) -> f64 {
    let position = event.target_pos_intercept;
    let velocity = event.target_vel_intercept;
    let w = event.v_rel;
    let r_norm = vec3_norm_sq(&position).sqrt();
    if !r_norm.is_finite() || r_norm <= 0.0 || !event.kappa.is_finite() || event.kappa <= 0.0 {
        return 0.0;
    }
    let perigee_radius = reentry_alt_km + RE;

    // E(f) = e0 + e1 f + e2 f^2
    let e0 = 0.5f64.mul_add(vec3_norm_sq(&velocity), -(MU / r_norm));
    let e1 = velocity[0].mul_add(w[0], velocity[1].mul_add(w[1], velocity[2] * w[2]));
    let e2 = 0.5 * vec3_norm_sq(&w);
    if e0 >= 0.0 {
        return 0.0;
    }

    // |h(f)|^2 = |h0|^2 + 2 (h0 . hw) f + |hw|^2 f^2
    let h0 = cross3(&position, &velocity);
    let hw = cross3(&position, &w);
    let hh0 = vec3_norm_sq(&h0);
    let hh1 = 2.0 * h0[0].mul_add(hw[0], h0[1].mul_add(hw[1], h0[2] * hw[2]));
    let hh2 = vec3_norm_sq(&hw);

    // Unbound: E(f) = 0.
    let f_unbound = smallest_root_in_open_interval(e2, e1, e0, event.kappa);
    // The unperturbed orbit must itself clear the interface, or the valid domain
    // is empty before any dust is released. This has to be a direct perigee
    // comparison: the quadratic below vanishes wherever the interface radius is
    // an APSIS, which is equally true of an orbit sitting entirely beneath it.
    let semi_major_0 = -MU / (2.0 * e0);
    let mu_squared = MU * MU;
    let eccentricity_0 = 2.0f64.mul_add(e0 * hh0 / mu_squared, 1.0).max(0.0).sqrt();
    let perigee_0 = semi_major_0 * (1.0 - eccentricity_0);
    if perigee_0.is_nan() || perigee_0 <= perigee_radius {
        return 0.0;
    }

    // Perigee: G(f) = |h(f)|^2 - 2 rp^2 E(f) - 2 mu rp = 0.
    //
    // `r_p` and `r_a` are the roots of `2E r^2 + 2 mu r - |h|^2` (the apsides,
    // where the radial rate vanishes). Given the guard above, the orbit starts
    // with its perigee clear of the interface, so the first positive root as the
    // kick grows is the perigee crossing.
    let rp_sq = perigee_radius * perigee_radius;
    let f_perigee = smallest_root_in_open_interval(
        2.0f64.mul_add(-(rp_sq * e2), hh2),
        2.0f64.mul_add(-(rp_sq * e1), hh1),
        2.0f64.mul_add(-(rp_sq * e0), hh0) - 2.0 * MU * perigee_radius,
        event.kappa,
    );

    let fraction = match (f_unbound, f_perigee) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => return mass_max,
    };
    if !fraction.is_finite() || fraction <= 0.0 {
        return 0.0;
    }
    let ceiling = event.p_mass * fraction / (event.kappa - fraction);
    if !ceiling.is_finite() || ceiling <= 0.0 {
        return 0.0;
    }
    ceiling.min(mass_max)
}

/// Solve one event, retrying inside the physically valid release-mass domain if
/// the unconstrained search comes back non-finite.
///
/// # Why the retry is conditional
///
/// The unconstrained bracket is `[0, mass_max = 1000 kg]`. For a light target
/// struck nearly head-on that interval is ~99.96% post-decay, and bisecting it
/// walks the midpoint sequence through a scatter of sub-milligram masses where
/// the equinoctial-to-ECI Kepler solve fails at eccentricity ~0.997. Those rows
/// abort with a non-finite root even though a perfectly good milligram-scale
/// root exists — `nsga2/flower` pop 3-6 produce two of them, whose true roots
/// are 3.81e-6 kg and 1.45e-5 kg at an essentially unchanged 601.8 km perigee.
/// Restricting the bracket to `[0, physical_release_mass_ceiling_kg]` excludes
/// the scatter and leaves the search in the monotone small-mass regime.
///
/// The cap is applied ONLY after the unconstrained attempt fails, and that is a
/// deliberate constraint rather than an optimisation. The ceiling lies below
/// `mass_max` on 93.2% of production rows (535,565 of 574,943 measured across
/// three families), so capping unconditionally would change the bisection
/// midpoint sequence — and therefore the converged root inside `xtol`/`rtol` —
/// on almost every row in the campaign, while changing no row's physics: zero
/// of those 574,943 converged roots lie above their own ceiling. Gating the cap
/// on failure keeps every row that converges today bit-identical, because it
/// re-enters the same code with the same `mass_max`, and confines the new
/// behaviour to rows that previously produced no answer at all.
#[must_use]
pub fn solve_single_event_mf_j2_with_status(
    event: &MfJ2MassSolverEvent,
    config: &SolverConfig,
) -> MfJ2MassSolveResult {
    let unconstrained = solve_mf_j2_over_bracket(event, config, config.mass_max);
    let ceiling =
        physical_release_mass_ceiling_kg(event, config.mass_max, REENTRY_INTERFACE_ALT_KM);
    if unconstrained.root_mass_kg.is_finite() {
        // POST-CHECK, not a pre-filter. The unrestricted solve used to return
        // here unconditionally, so any finite result -- a converged root or a
        // `PhysicsLimited` endpoint -- was published without ever being compared
        // against the valid-domain ceiling. A committed census found no
        // converged root above its own ceiling in 574,943 rows, which is
        // evidence that it has not happened, not an invariant that it cannot.
        if !ceiling.is_finite() || ceiling <= 0.0 {
            return MfJ2MassSolveResult {
                status: MfJ2MassSolveStatusCode::AtmosphericLimited,
                root_mass_kg: f64::NAN,
                ..unconstrained
            };
        }
        if unconstrained.root_mass_kg <= ceiling {
            return unconstrained;
        }
        // Above the ceiling: resolve inside the safe domain rather than publish
        // a mass that leaves the target unbound or below the reentry interface.
        // The restricted solve reports its OWN status, for the same reason the
        // non-finite path below does.
        return solve_mf_j2_over_bracket(event, config, ceiling);
    }
    if !ceiling.is_finite() || ceiling <= 0.0 {
        // The target is already unbound or at/below the interface before any
        // dust is released, so no release mass leaves it in a usable orbit.
        return MfJ2MassSolveResult {
            status: MfJ2MassSolveStatusCode::AtmosphericLimited,
            ..unconstrained
        };
    }
    // If the restricted bisection still fails it reports its OWN reason —
    // `MaxIterReached` is an iteration-budget failure and `MidNonFinite` a
    // sampling one, and neither becomes an atmospheric verdict just because the
    // bracket happened to be capped. `AtmosphericLimited` is reserved for the
    // one condition it actually names: an empty physical domain.
    solve_mf_j2_over_bracket(event, config, ceiling)
}

/// The pre-cap solver behaviour, retained as the regression oracle.
///
/// `solve_single_event_mf_j2_with_status` must agree with this bit-for-bit on
/// every row where this returns a finite root — that is the whole content of
/// the claim that the atmospheric bracket cap touches only rows which
/// previously produced no answer. Exposed so the invariant can be asserted
/// against real production rows rather than argued from the control flow.
#[must_use]
pub fn solve_single_event_mf_j2_unconstrained_bracket(
    event: &MfJ2MassSolverEvent,
    config: &SolverConfig,
) -> MfJ2MassSolveResult {
    solve_mf_j2_over_bracket(event, config, config.mass_max)
}

/// The bisection itself, over an explicit upper bracket.
#[must_use]
fn solve_mf_j2_over_bracket(
    event: &MfJ2MassSolverEvent,
    config: &SolverConfig,
    bracket_upper: f64,
) -> MfJ2MassSolveResult {
    let target = event.min_miss_distance_km;
    let mut upper = bracket_upper;
    let tol = config.xtol.max(1e-12);

    let anchor_baseline = event
        .target_conj_pos
        .and_then(|_| mf_j2_zero_mass_position(event));
    if event.target_conj_pos.is_some() && anchor_baseline.is_none() {
        return MfJ2MassSolveResult {
            root_mass_kg: f64::NAN,
            miss_at_root_km: f64::NAN,
            miss_at_zero_km: f64::NAN,
            miss_at_upper_km: f64::NAN,
            iterations: 0,
            status: MfJ2MassSolveStatusCode::MissAtZeroNonFinite,
        };
    }
    let miss0 = compute_miss_distance_mf_j2(0.0, event, anchor_baseline);
    if !miss0.is_finite() {
        return MfJ2MassSolveResult {
            root_mass_kg: f64::NAN,
            miss_at_root_km: f64::NAN,
            miss_at_zero_km: miss0,
            miss_at_upper_km: f64::NAN,
            iterations: 0,
            status: MfJ2MassSolveStatusCode::MissAtZeroNonFinite,
        };
    }
    if miss0 >= target {
        return MfJ2MassSolveResult {
            root_mass_kg: 0.0,
            miss_at_root_km: miss0,
            miss_at_zero_km: miss0,
            miss_at_upper_km: miss0,
            iterations: 0,
            status: MfJ2MassSolveStatusCode::SafeByDefault,
        };
    }

    let mut miss_hi = compute_miss_distance_mf_j2(upper, event, anchor_baseline);
    let mut upper_shrink_iters = 0usize;
    while !miss_hi.is_finite() && upper > tol && upper_shrink_iters < 30 {
        upper *= 0.5;
        miss_hi = compute_miss_distance_mf_j2(upper, event, anchor_baseline);
        upper_shrink_iters = upper_shrink_iters.saturating_add(1);
    }
    if !miss_hi.is_finite() {
        return MfJ2MassSolveResult {
            root_mass_kg: f64::NAN,
            miss_at_root_km: f64::NAN,
            miss_at_zero_km: miss0,
            miss_at_upper_km: miss_hi,
            iterations: upper_shrink_iters,
            status: MfJ2MassSolveStatusCode::UpperNonFinite,
        };
    }
    if miss_hi < target {
        return MfJ2MassSolveResult {
            root_mass_kg: upper,
            miss_at_root_km: miss_hi,
            miss_at_zero_km: miss0,
            miss_at_upper_km: miss_hi,
            iterations: 0,
            status: MfJ2MassSolveStatusCode::PhysicsLimited,
        };
    }

    let mut lo = 0.0;
    let mut hi = upper;
    let mut miss_mid = miss_hi;
    for iteration in 1..=config.maxiter.max(1) {
        let mid = 0.5 * (lo + hi);
        miss_mid = compute_miss_distance_mf_j2(mid, event, anchor_baseline);
        if !miss_mid.is_finite() {
            return MfJ2MassSolveResult {
                root_mass_kg: f64::NAN,
                miss_at_root_km: f64::NAN,
                miss_at_zero_km: miss0,
                miss_at_upper_km: miss_hi,
                iterations: iteration,
                status: MfJ2MassSolveStatusCode::MidNonFinite,
            };
        }
        let residual = miss_mid - target;
        // `target.max(1.0)` floors on every production row, so this reads as an
        // absolute kilometre tolerance, not a relative one. See `SolverConfig`.
        let residual_converged = config.rtol.is_finite()
            && config.rtol > 0.0
            && residual.abs() < config.rtol * target.abs().max(1.0);
        if (hi - lo).abs() <= tol || residual_converged {
            return MfJ2MassSolveResult {
                root_mass_kg: mid,
                miss_at_root_km: miss_mid,
                miss_at_zero_km: miss0,
                miss_at_upper_km: miss_hi,
                iterations: iteration,
                status: MfJ2MassSolveStatusCode::Converged,
            };
        }
        if miss_mid >= target {
            hi = mid;
            miss_hi = miss_mid;
        } else {
            lo = mid;
        }
    }

    MfJ2MassSolveResult {
        root_mass_kg: f64::NAN,
        miss_at_root_km: miss_mid,
        miss_at_zero_km: miss0,
        miss_at_upper_km: miss_hi,
        iterations: config.maxiter.max(1),
        status: MfJ2MassSolveStatusCode::MaxIterReached,
    }
}

/// Use the cheap MF-J2 solver to estimate the dust mass, then inflate by 1.5×
/// to produce a tighter upper bracket for the expensive HF Brent's solver.
///
/// Returns `None` if the MF solver doesn't converge or returns a non-positive root.
///
/// This deliberately calls [`solve_mf_j2_over_bracket`] with the unconstrained
/// `mass_max` rather than [`solve_single_event_mf_j2_with_status`], so the
/// atmospheric bracket cap does NOT reach the HF lane. The cap turns some
/// previously non-finite MF solves into converged ones; through this function
/// that would flip the preseed from `None` to `Some(..)`, changing the HF
/// Brent bracket and moving strict-HF numbers that no measurement here covers.
/// The evidence and the ruling behind the cap are about the MF deterministic-mass
/// lane, so the HF preseed keeps its existing behaviour bit-for-bit.
fn mf_j2_preseed_upper_bound(event: &MassSolverEvent, config: &SolverConfig) -> Option<f64> {
    let mf_event = MfJ2MassSolverEvent::new(
        event.p_pos_intercept,
        event.p_velocity,
        event.dv_vec,
        event.p_mass,
        event.secondary_conj_pos,
        event.tof_s,
        event.min_miss_distance_km,
        event.kappa,
    );
    let result = solve_mf_j2_over_bracket(&mf_event, config, config.mass_max);
    if result.status == MfJ2MassSolveStatusCode::Converged && result.root_mass_kg > 0.0 {
        Some((1.5 * result.root_mass_kg).min(config.mass_max))
    } else {
        None
    }
}

#[inline]
fn apply_anchored_adjustment(mut pos: [f64; 3], derived: &EventDerived) -> [f64; 3] {
    if derived.apply_anchor_shift {
        pos[0] += derived.anchor_shift[0];
        pos[1] += derived.anchor_shift[1];
        pos[2] += derived.anchor_shift[2];
    }
    pos
}

/// Compute miss distance for a given mass (Low-Fidelity Keplerian)
///
/// This is the objective function for bisection when HF mode is disabled.
fn compute_miss_distance_lf(mass: f64, event: &MassSolverEvent, derived: &EventDerived) -> f64 {
    if !mass.is_finite() || mass < 0.0 {
        return f64::INFINITY;
    }

    // Step 1: Compute new velocity after dust impact
    // Uses precomputed p_velocity and v_rel from event (Phase 3 optimization)
    let new_vel = compute_new_velocity(mass, event);
    if !new_vel[0].is_finite() || !new_vel[1].is_finite() || !new_vel[2].is_finite() {
        return f64::INFINITY;
    }

    // Step 1b: Early hyperbolic detection (Phase 3 optimization)
    // Check if orbit would be hyperbolic before expensive coordinate transform.
    // Specific energy: e = v²/2 - μ/r. For bound orbit e < 0, hyperbolic e >= 0.
    let v_sq = vec3_norm_sq(&new_vel);
    if derived.mu_over_r_intercept.is_finite() {
        let specific_energy = 0.5 * v_sq - derived.mu_over_r_intercept;
        // DOC-M2: Hyperbolic orbit detection threshold, RELATIVE to the local
        // energy scale.
        //
        // Physics basis: specific orbital energy e = -mu/(2a), so e > 0 means
        // a < 0 (hyperbolic). The small positive margin admits near-parabolic
        // states that are bound but sit within numerical reach of zero.
        //
        // The margin used to be the absolute 1e-6 km^2/s^2, and its own comment
        // said what was wrong with it: "corresponds to eccentricity e ~ 1.0001
        // AT LEO ALTITUDES". Specific energy is computed as v^2/2 - mu/r, whose
        // terms scale as mu/r -- 58.8 km^2/s^2 in LEO, 9.45 at GEO, 1.04 at
        // lunar distance. A fixed absolute margin therefore means a different
        // eccentricity tolerance at every altitude: calibrated for LEO, it is
        // ~6x looser at GEO and ~57x looser at lunar distance, so the same
        // orbit shape is classified differently depending only on how far out
        // it happens to be.
        //
        // Scaling by `mu_over_r_intercept` -- the energy scale already computed
        // here -- makes the margin mean one thing everywhere. The constant is
        // chosen to reproduce the LEO calibration EXACTLY at r = 6778 km, so
        // this is the same test in the regime it was designed for and a
        // tighter, better-posed one outside it.
        if specific_energy_is_escaped(specific_energy, derived.mu_over_r_intercept) {
            return f64::INFINITY; // Hyperbolic: skip expensive transform
        }
    }

    // Step 2: Build ECI state [pos, vel]
    let eci_state = [
        event.p_pos_intercept[0],
        event.p_pos_intercept[1],
        event.p_pos_intercept[2],
        new_vel[0],
        new_vel[1],
        new_vel[2],
    ];
    let Ok(propagated) = propagate_target_for_mass_authority(
        &eci_state,
        event.tof_s,
        TargetPropagationAuthority::AnalyticalKepler,
        None,
        &mut UnobservedMassSolve,
        MassLegTag {
            role: MassLegRole::MassEvaluation,
            mass_kg_bits: mass.to_bits(),
        },
    ) else {
        return f64::INFINITY;
    };

    // Step 5: Extract position
    let new_pos = [propagated[0], propagated[1], propagated[2]];

    // Step 6: Apply anchored differential adjustment (with degeneracy guard)
    //
    // Error budget for anchored differential:
    //   truth baseline = pre-interpolated state (LF or HF, from state table)
    //   equ_0 baseline = fresh Keplerian propagation at mass=0
    //   delta = propagated(mass) - equ_0
    //   result = truth + delta
    //
    // The anchored differential corrects for the difference between the
    // interpolated (possibly HF) truth and the Keplerian reference, ensuring
    // the bisection operates on physically consistent miss distances.
    // Degeneracy guard (1e-12 km^2 = 1 µm separation) disables correction
    // when both baselines are identical (e.g., pure LF with no interpolation
    // difference), preventing numerical noise from dominating.
    let new_pos = apply_anchored_adjustment(new_pos, derived);

    // Step 7: Compute L2 distance to secondary
    vec3_distance(&new_pos, &event.secondary_conj_pos)
}

/// Pre-compute the HF force configuration for a single event.
///
/// PERF2: This hoists [`ForceConfig`] clone, per-event overrides (`am_ratio`, `cd`, `cr`),
/// and full-arc ephemeris resolution out of the bisection loop.
/// These are invariant across the 30-60+ bisection iterations per event.
///
/// Returns None if HF context is incomplete (caller should fall back to LF).
/// Returns Some(Err(())) if ephemeris is missing and strict mode is enabled.
fn prepare_hf_for_event(
    event: &MassSolverEvent,
    hf_ctx: &HfContext,
) -> Option<Result<PreparedHfConfig, ()>> {
    prepare_hf_for_event_impl(event, hf_ctx, true)
}

/// Preparation probe for speculative acceleration. It returns the same value
/// as [`prepare_hf_for_event`] but cannot publish the row-owned failure latch.
fn prepare_hf_for_event_unlatched(
    event: &MassSolverEvent,
    hf_ctx: &HfContext,
) -> Option<Result<PreparedHfConfig, ()>> {
    prepare_hf_for_event_impl(event, hf_ctx, false)
}

fn prepare_hf_for_event_impl(
    event: &MassSolverEvent,
    hf_ctx: &HfContext,
    latch_failure: bool,
) -> Option<Result<PreparedHfConfig, ()>> {
    let (Some(force_config), Some(packed)) = (&hf_ctx.force_config, &hf_ctx.packed_coeffs) else {
        if hf_ctx.hf_strict {
            return Some(Err(()));
        }
        return None;
    };

    let mut config = *force_config.as_ref();
    if let Some(am_ratio) = event.p_am_ratio {
        config.am_ratio = am_ratio;
    }
    if let Some(cd) = event.p_cd {
        config.cd = cd;
    }
    if let Some(cr) = event.p_cr {
        config.cr = cr;
    }
    if let Some(qm_ratio) = event.p_qm_ratio {
        config.qm_ratio = qm_ratio;
    }
    if let Some(r_obj_m) = event.p_r_obj_m {
        config.r_obj_m = r_obj_m;
    }

    let config = match config
        .with_ephemeris_for_arc(hf_ctx.epoch_jd, hf_ctx.epoch_jd + event.tof_s / SEC_PER_DAY)
    {
        Ok(config) => config,
        Err(_error) => {
            if latch_failure {
                latch_hf_preflight_failure();
            }
            return Some(Err(()));
        }
    };

    Some(Ok(PreparedHfConfig {
        force_config: Arc::new(config),
        epoch_jd: hf_ctx.epoch_jd,
        packed_coeffs: Arc::clone(packed),
    }))
}

/// Compute miss distance for a given mass (High-Fidelity Lightyear) using pre-computed config.
///
/// Uses Encke method with Lightyear ODE integration for perturbations (J2, drag, SRP).
/// The `prepared` config contains pre-resolved [`ForceConfig`] with ephemeris positions,
/// avoiding redundant work inside the bisection loop (PERF2).
fn compute_miss_distance_hf_prepared<O: MassSolveObserver>(
    mass: f64,
    event: &MassSolverEvent,
    prepared: &PreparedHfConfig,
    derived: &EventDerived,
    observer: &mut O,
) -> f64 {
    if !mass.is_finite() || mass < 0.0 {
        return f64::INFINITY;
    }

    // Step 1: Compute new velocity after dust impact
    let new_vel = compute_new_velocity(mass, event);
    if !new_vel[0].is_finite() || !new_vel[1].is_finite() || !new_vel[2].is_finite() {
        return f64::INFINITY;
    }

    let Some(retained_prepared) = prepared.for_retained_mass(mass, event) else {
        return f64::INFINITY;
    };

    // Step 1b: Early hyperbolic detection (see DOC-M2 comment in miss_distance_from_mass)
    let v_sq = vec3_norm_sq(&new_vel);
    if derived.mu_over_r_intercept.is_finite() {
        let specific_energy = 0.5 * v_sq - derived.mu_over_r_intercept;
        if specific_energy_is_escaped(specific_energy, derived.mu_over_r_intercept) {
            return f64::INFINITY; // Hyperbolic: physically meaningless trajectory
        }
    }

    // Step 2: Build ECI state [pos, vel]
    let eci_state = [
        event.p_pos_intercept[0],
        event.p_pos_intercept[1],
        event.p_pos_intercept[2],
        new_vel[0],
        new_vel[1],
        new_vel[2],
    ];
    let min_radius_km =
        retained_prepared.force_config.earth_radius + lightyear_odeint_rs::types::GROUND_ALTITUDE;
    if !state_clears_min_radius(&eci_state, min_radius_km) {
        return f64::INFINITY;
    }
    let propagated = match propagate_target_for_mass_authority(
        &eci_state,
        event.tof_s,
        TargetPropagationAuthority::HighFidelity,
        Some(&retained_prepared),
        observer,
        MassLegTag {
            role: MassLegRole::MassEvaluation,
            mass_kg_bits: mass.to_bits(),
        },
    ) {
        Ok(propagated) => propagated,
        Err(
            TargetMassPropagationError::IntegrationFailed(_)
            | TargetMassPropagationError::NonFinite,
        ) => return f64::NAN,
        Err(_) => return f64::INFINITY,
    };

    // Step 6: Extract position
    let new_pos = [propagated[0], propagated[1], propagated[2]];

    // Step 7: Apply anchored differential adjustment (with degeneracy guard)
    // NOTE: For HF mode, the truth baseline should already be HF-propagated
    let new_pos = apply_anchored_adjustment(new_pos, derived);

    // Step 8: Compute L2 distance to secondary
    vec3_distance(&new_pos, &event.secondary_conj_pos)
}

#[inline]
fn compute_miss_distance_hf_prepared_cached<O: MassSolveObserver>(
    mass: f64,
    event: &MassSolverEvent,
    prepared: &PreparedHfConfig,
    derived: &EventDerived,
    zero_mass_cache: Option<ZeroMassCacheView<'_>>,
    observer: &mut O,
) -> f64 {
    if mass.to_bits() != 0 {
        return compute_miss_distance_hf_prepared(mass, event, prepared, derived, observer);
    }
    let Some(slot) = zero_mass_cache else {
        return compute_miss_distance_hf_prepared(mass, event, prepared, derived, observer);
    };
    *slot
        .miss_at_zero
        .get_or_init(|| compute_miss_distance_hf_prepared(mass, event, prepared, derived, observer))
}

#[inline]
const fn dust_hf_validate_only_enabled() -> bool {
    // Default true: LF bisection + HF validation/repair gives defensible HF accuracy
    // with O(5-10) HF calls instead of O(30+) full HF bisection.
    true
}

/// EIGHT ADJACENT `f64` PARAMETERS, AND THE UNITS INTERLEAVE.
///
/// The `_m` suffix here means MASS IN KILOGRAMS, not metres: `lo_m`, `hi_m`,
/// `xtol` and `mass_max` are kg, while `lo_miss`, `hi_miss`, `miss_zero_est` and
/// `target` are miss distances in KILOMETRES. So the sequence is kg, km, kg, km,
/// km, km, kg, kg — every one an order-1-or-larger positive f64 with nothing in
/// the type to separate them.
///
/// The suffix is not renamed on purpose: `_m` is used this way throughout the
/// solver and changing it here alone would make this the odd one out.
#[inline]
fn next_validate_repair_mass_candidate(
    lo_m: f64,
    lo_miss: f64,
    hi_m: f64,
    hi_miss: f64,
    miss_zero_est: f64,
    target: f64,
    xtol: f64,
    mass_max: f64,
) -> f64 {
    if !hi_m.is_finite() || !hi_miss.is_finite() || hi_miss <= 0.0 || !target.is_finite() {
        return mass_max;
    }

    // First prefer a secant-style extrapolation from the two latest below-target
    // points. This usually brackets in one fewer HF call than multiplicative jumps.
    if lo_m.is_finite() && lo_miss.is_finite() && hi_m > lo_m + xtol && hi_miss > lo_miss + 1e-12 {
        let slope = (hi_miss - lo_miss) / (hi_m - lo_m);
        if slope.is_finite() && slope > 0.0 {
            let mut next = hi_m + (target - hi_miss).max(0.0) / slope;
            let min_step = (hi_m + xtol).max(hi_m * 1.05);
            let max_step = (hi_m * 16.0).min(mass_max);
            if max_step > min_step {
                next = next.clamp(min_step, max_step);
                if next > hi_m + xtol {
                    return next;
                }
            }
        }
    }

    // Adaptive multiplicative jump based on remaining miss-distance deficit.
    // Faster than fixed x2 growth for large deficits while preserving monotonic mass increase.
    let ratio = if miss_zero_est.is_finite()
        && target > miss_zero_est + 1e-9
        && hi_miss > miss_zero_est + 1e-9
    {
        ((target - miss_zero_est) / (hi_miss - miss_zero_est)).clamp(1.25, 8.0)
    } else {
        (target / hi_miss).clamp(1.25, 8.0)
    };
    let mut next = (hi_m * ratio).min(mass_max);
    if next <= hi_m + xtol {
        // Guarantee strict forward progress when ratio clamps near 1.0.
        next = (hi_m * 1.5).min(mass_max);
    }
    if next <= hi_m + xtol {
        mass_max
    } else {
        next
    }
}

/// The affine estimate of the MINIMUM mass, from two points that are ON the
/// curve: `(0, miss_at_zero)` and `(seed_mass, miss_at_seed)`. Only meaningful
/// when the zero-mass anchor is the exact HF propagation -- see
/// `VALIDATE_CORRECTION_SCALE_MIN_EXACT` for why that is what puts the first
/// point on the curve.
#[inline]
fn validate_affine_root(
    seed_mass: f64,
    miss_at_seed: f64,
    miss_at_zero: f64,
    target: f64,
) -> Option<f64> {
    let span = miss_at_seed - miss_at_zero;
    if !span.is_finite() || span <= 1e-9 || !target.is_finite() || target <= miss_at_zero {
        return None;
    }
    let root = seed_mass * (target - miss_at_zero) / span;
    (root.is_finite() && root >= 0.0).then_some(root)
}

/// Whether a SAFE row (`miss(seed) >= target`) can retire at `seed_mass`
/// without a refinement: the affine minimum sits within `xtol` of the seed, so
/// the refinement would walk to a mass the tolerance already declares identical.
///
/// Conservative by construction. `seed_mass` is an HF-evaluated point with
/// `miss >= target`, and the root is BELOW it, so this overstates the minimum by
/// at most `xtol` and can never understate it.
#[inline]
fn validate_safe_row_retires_at_seed(
    seed_mass: f64,
    miss_at_seed: f64,
    miss_at_zero: f64,
    target: f64,
    xtol: f64,
) -> bool {
    validate_affine_root(seed_mass, miss_at_seed, miss_at_zero, target)
        .is_some_and(|root| seed_mass - root <= xtol)
}

/// Clamp on the one-shot affine correction scale, and why the floor has two
/// settings.
///
/// `miss_zero_est` is `||p_pos_conj_truth - secondary_conj_pos||`, and
/// `derive_event_invariants` sets `anchor_shift = p_pos_conj_truth -
/// zero_mass_reference` so that `apply_anchored_adjustment` maps the zero-mass
/// propagation ONTO `p_pos_conj_truth`. When the reference is the exact HF
/// propagation (`ZeroMassReference::exact_hf`) those two facts compose:
/// `miss_zero_est` IS `miss(0)` of the objective this loop is solving. The
/// affine model through `(0, miss_zero_est)` and `(seed_mass, miss)` then
/// interpolates two points that are ON the curve, and the only error left is
/// curvature -- measured at 3.1e-6 of the target miss over the sub-1% step this
/// branch actually takes (4 designs x 2 events, 395 correcting rows,
/// 2026-08-13). A floor of `1.2` on an estimate that accurate does not add
/// safety, it discards the estimate: it fired on 395 of 395 rows and replaced a
/// 1.002x correction with a 1.2x one, which then cost a Brent refinement to
/// walk back down.
///
/// The floor cannot LOWER an estimate: `raw_scale > 1` is an invariant of this
/// branch, which is reached only when `miss < target`, and (in the anchored
/// form) `miss > miss_zero_est`.
///
/// With the LF fallback reference the first point is not on the HF curve at all,
/// the chord slope can be arbitrarily wrong, and the wide floor is what stops it
/// from throwing the bracket.
const VALIDATE_CORRECTION_SCALE_MIN: f64 = 1.2;
const VALIDATE_CORRECTION_SCALE_MIN_EXACT: f64 = 1.0;
const VALIDATE_CORRECTION_SCALE_MAX: f64 = 32.0;

/// Overshoot applied to a sub-`xtol` affine step so the single evaluation lands
/// ABOVE the root rather than on a coin toss, and the ceiling that keeps the
/// bracket it leaves inside the absolute-width arm even after `hi_m - lo_m`
/// rounds. `0.999` rather than `1.0` because that arm tests `width <= xtol` and
/// a one-ulp excess would fail it.
const VALIDATE_NARROW_STEP_MARGIN: f64 = 1.01;
const VALIDATE_NARROW_STEP_CAP: f64 = 0.999;

/// Where to evaluate the one-shot HF correction, and the step below which
/// evaluating it is not worth an arc.
///
/// Returns `(target_mass, fire_floor)`; the caller evaluates when
/// `target_mass > seed_mass + fire_floor`.
///
/// The narrow arm is the whole point. `affine_m - seed_mass` is the mass the
/// affine estimate adds, and on most production rows it is SMALLER THAN `xtol`:
/// the LF seed already sits inside the interval the solver declares itself
/// indifferent over (316 of 395 rows in the corpus above; median step 0.43
/// `xtol`). The solver could not say so, because it held no HF-evaluated point
/// at or above `target` -- so it manufactured one far away (1.2x here, or 1.25x
/// from `next_validate_repair_mass_candidate` when the step was too small to
/// fire this branch at all) and then spent a Brent refinement walking back down.
/// Aiming the SAME single evaluation just above the affine root instead leaves a
/// bracket no wider than `xtol`, which `validate_bracket_arm`'s absolute-width
/// arm already accepts: the evaluation that establishes the upper endpoint also
/// retires the row.
///
/// This does not relax acceptance. The returned mass is still an HF-evaluated
/// point with `miss >= target`, which is strictly stronger than what the
/// refinement it replaces returns -- that one retires on `|residual| < rtol`
/// and so may return a mass whose HF miss falls up to `rtol * target` SHORT.
#[inline]
fn validate_one_shot_target(
    seed_mass: f64,
    correction_scale: f64,
    xtol: f64,
    mass_max: f64,
    anchor_is_exact_hf: bool,
) -> (f64, f64) {
    let affine_m = (seed_mass * correction_scale).min(mass_max);
    let affine_step = affine_m - seed_mass;
    if anchor_is_exact_hf && affine_step > 0.0 && affine_step < xtol {
        let step = (affine_step * VALIDATE_NARROW_STEP_MARGIN)
            .clamp(0.5 * xtol, VALIDATE_NARROW_STEP_CAP * xtol);
        (seed_mass + step, 0.0)
    } else {
        (affine_m, xtol)
    }
}

/// Which arm of the validate convergence gate retires the bracket: absolute
/// width vs `xtol`, relative width vs `xtol`, residual vs `rtol * target`, or
/// no arm fired and the row falls through to refine.
///
/// The arm constants lived in `lightyear_odeint_rs::probe` while the validate
/// tolerance census was running; the census closed (`xtol` and `rtol` are both
/// at their measured floors) and the classifier stayed, because the production
/// call site reads the arm directly.
const VALIDATE_GATE_ARM_ABS_WIDTH: usize = 0;
const VALIDATE_GATE_ARM_RESIDUAL: usize = 2;
const VALIDATE_GATE_ARM_FELL_THROUGH: usize = 3;

#[inline]
fn validate_bracket_arm(
    lo_m: f64,
    hi_m: f64,
    hi_residual: f64,
    xtol: f64,
    rtol: f64,
    target: f64,
) -> usize {
    let width = (hi_m - lo_m).abs();
    if width <= xtol.max(1e-12) {
        return VALIDATE_GATE_ARM_ABS_WIDTH;
    }
    // The relative-width arm is DELETED, not repaired.
    //
    // It divided a kilogram bracket width by `max(|hi|, 1)` and compared the
    // resulting dimensionless number against `xtol`, which is documented in
    // kilograms. Below 1 kg the divisor is 1, so the arm was an exact duplicate
    // of the absolute arm above -- measured firing 0 times in 3,395 gate
    // entries. Above 1 kg it was a unit error that accepted a bracket up to
    // `xtol * hi` wide: with the compiled 1e-6 kg and a 1000 kg maximum, up to
    // 1e-3 kg.
    //
    // Deleting rather than introducing a dimensionless `mass_rtol`, because
    // nothing asked for a relative mass tolerance: the arm existed as an
    // accidental duplicate and only became reachable when `rtol` relaxed away
    // from `xtol`. A separately typed tolerance would be a new knob with no
    // caller.
    // `target.max(1.0)` floors on every production row, so this reads as an
    // absolute kilometre tolerance, not a relative one. See `SolverConfig`.
    if hi_residual.abs() <= rtol.max(1e-12) * target.max(1.0) {
        return VALIDATE_GATE_ARM_RESIDUAL;
    }
    VALIDATE_GATE_ARM_FELL_THROUGH
}

/// The gate as a predicate. The production call site reads the arm directly, so
/// this survives as the shape the convergence tests assert against.
#[cfg(test)]
fn validate_bracket_is_converged(
    lo_m: f64,
    hi_m: f64,
    hi_residual: f64,
    xtol: f64,
    rtol: f64,
    target: f64,
) -> bool {
    validate_bracket_arm(lo_m, hi_m, hi_residual, xtol, rtol, target)
        != VALIDATE_GATE_ARM_FELL_THROUGH
}

#[derive(Clone, Copy, Debug)]
enum UpperBoundSearch {
    Safe { mass: f64, miss: f64 },
    PhysicsLimited { mass: f64, miss: f64 },
    NonFinite,
}

/// Find a finite HF mass whose miss distance reaches `target`.
///
/// A mixed-fidelity preseed is only a first probe. It cannot define the HF
/// physical ceiling: if unsafe, search expands toward `mass_max`; if a probe
/// becomes non-finite, bisection searches the remaining finite interval for a
/// safe bracket before classifying the event as physics-limited.
fn find_finite_safe_upper_bound(
    initial_mass: f64,
    mass_max: f64,
    xtol: f64,
    target: f64,
    mut eval_miss: impl FnMut(f64) -> f64,
) -> UpperBoundSearch {
    if !mass_max.is_finite() || mass_max <= 0.0 || !target.is_finite() {
        return UpperBoundSearch::NonFinite;
    }
    let tol = xtol.max(1e-12);
    let mut probe = if initial_mass.is_finite() && initial_mass > 0.0 {
        initial_mass.min(mass_max)
    } else {
        mass_max
    };
    probe = probe.max(tol.min(mass_max));
    let mut best_finite_unsafe: Option<(f64, f64)> = None;
    let mut nonfinite_ceiling: Option<f64> = None;

    for _ in 0..80 {
        let miss = eval_miss(probe);
        if miss.is_finite() {
            if miss >= target {
                return UpperBoundSearch::Safe { mass: probe, miss };
            }
            if best_finite_unsafe.is_none_or(|(mass, _)| probe > mass) {
                best_finite_unsafe = Some((probe, miss));
            }
            if probe >= mass_max - tol {
                return UpperBoundSearch::PhysicsLimited { mass: probe, miss };
            }
        } else {
            hf_profile_inc_upper_bracket_shrink();
            nonfinite_ceiling = Some(nonfinite_ceiling.map_or(probe, |old| old.min(probe)));
        }

        let finite_floor = best_finite_unsafe.map_or(0.0, |(mass, _)| mass);
        let next = nonfinite_ceiling.map_or_else(
            || (probe * 2.0).max(probe + tol).min(mass_max),
            |ceiling| 0.5 * (finite_floor + ceiling),
        );
        if (next - probe).abs() <= tol || next <= finite_floor {
            return best_finite_unsafe.map_or(UpperBoundSearch::NonFinite, |(mass, miss)| {
                UpperBoundSearch::PhysicsLimited { mass, miss }
            });
        }
        probe = next;
    }

    best_finite_unsafe.map_or(UpperBoundSearch::NonFinite, |(mass, miss)| {
        UpperBoundSearch::PhysicsLimited { mass, miss }
    })
}

/// Search only the unresolved interval below one non-finite validate probe.
///
/// The lower endpoint is already a finite unsafe HF evaluation; the upper
/// endpoint is already known non-finite. This helper shares the caller's
/// existing HF budget and returns only a newly evaluated finite safe point.
fn recover_finite_safe_validate_probe(
    eval_hf: &mut impl FnMut(f64, HfProfileStage) -> f64,
    hf_calls: &mut usize,
    hf_budget: usize,
    finite_unsafe_mass: f64,
    nonfinite_mass: f64,
    xtol: f64,
    target: f64,
) -> Option<(f64, f64)> {
    if !finite_unsafe_mass.is_finite()
        || !nonfinite_mass.is_finite()
        || !target.is_finite()
        || finite_unsafe_mass < 0.0
        || nonfinite_mass <= finite_unsafe_mass
    {
        return None;
    }
    let tolerance = xtol.max(1.0e-12);
    let mut unsafe_mass = finite_unsafe_mass;
    let mut nonfinite_ceiling = nonfinite_mass;
    while *hf_calls < hf_budget && nonfinite_ceiling - unsafe_mass > tolerance {
        let probe = unsafe_mass + 0.5 * (nonfinite_ceiling - unsafe_mass);
        if probe <= unsafe_mass || probe >= nonfinite_ceiling {
            break;
        }
        let miss = eval_hf(probe, HfProfileStage::ValidateRepair);
        *hf_calls = hf_calls.saturating_add(1);
        if miss.is_finite() {
            if miss >= target {
                return Some((probe, miss));
            }
            unsafe_mass = probe;
        } else {
            nonfinite_ceiling = probe;
        }
    }
    None
}

fn refine_validate_hf_bracket(
    mut eval_hf: impl FnMut(f64, HfProfileStage) -> f64,
    mut hf_calls: usize,
    hf_budget: usize,
    lo_m: f64,
    lo_miss: f64,
    hi_m: f64,
    hi_miss: f64,
    config: &SolverConfig,
    target: f64,
) -> (f64, MassSolveStatusCode) {
    let mut previous_mass = lo_m;
    let mut best_mass = hi_m;
    let mut previous_residual = lo_miss - target;
    let mut best_residual = hi_miss - target;
    let mut bracket_mass = previous_mass;
    let mut bracket_residual = previous_residual;
    let mut step = best_mass - previous_mass;
    let mut previous_step = step;
    let mut safe_mass = hi_m;
    let mut safe_residual = hi_miss - target;

    while hf_calls < hf_budget {
        if (best_residual > 0.0 && bracket_residual > 0.0)
            || (best_residual < 0.0 && bracket_residual < 0.0)
        {
            bracket_mass = previous_mass;
            bracket_residual = previous_residual;
            step = best_mass - previous_mass;
            previous_step = step;
        }
        if bracket_residual.abs() < best_residual.abs() {
            previous_mass = best_mass;
            best_mass = bracket_mass;
            bracket_mass = previous_mass;
            previous_residual = best_residual;
            best_residual = bracket_residual;
            bracket_residual = previous_residual;
        }

        let tolerance = 2.0 * f64::EPSILON * best_mass.abs() + 0.5 * config.xtol;
        let midpoint_delta = 0.5 * (bracket_mass - best_mass);
        // `target.max(1.0)` floors on every production row, so the residual arm
        // is an absolute kilometre tolerance. See `SolverConfig`.
        if midpoint_delta.abs() <= tolerance || best_residual.abs() < config.rtol * target.max(1.0)
        {
            // The SAME secant the full-Brent site records, at the return
            // production actually takes. `hf_validate_only` is true in the
            // sealed controls, so every strict row converges HERE and the
            // full-Brent loop below is not reached with HF at all -- which is
            // why `PROP_MASSSENS` carried only LF seeds until this line existed.
            // `eval_hf` is HF unconditionally: the one caller (`:2170`) sits
            // inside `solve_single_event_hf_validate_only`, which takes a
            // non-optional `&HfContext`.
            let (converged_mass, converged_residual, comparison_mass, comparison_residual) =
                if best_residual >= 0.0 {
                    (best_mass, best_residual, previous_mass, previous_residual)
                } else {
                    (safe_mass, safe_residual, best_mass, best_residual)
                };
            if (converged_mass - comparison_mass).abs() > 0.0 {
                lightyear_odeint_rs::probe::record_mass_sensitivity(
                    converged_mass,
                    (converged_residual - comparison_residual) / (converged_mass - comparison_mass),
                );
            }
            return (converged_mass, converged_status_for_mass(converged_mass));
        }

        if previous_step.abs() >= tolerance && previous_residual.abs() > best_residual.abs() {
            let residual_ratio = best_residual / previous_residual;
            let (mut interpolation_numerator, mut interpolation_denominator);
            if (previous_mass - bracket_mass).abs() < f64::EPSILON * previous_mass.abs().max(1.0) {
                interpolation_numerator = 2.0 * midpoint_delta * residual_ratio;
                interpolation_denominator = 1.0 - residual_ratio;
            } else {
                interpolation_denominator = previous_residual / bracket_residual;
                let bracket_ratio = best_residual / bracket_residual;
                interpolation_numerator = residual_ratio
                    * (2.0
                        * midpoint_delta
                        * interpolation_denominator
                        * (interpolation_denominator - bracket_ratio)
                        - (best_mass - previous_mass) * (bracket_ratio - 1.0));
                interpolation_denominator = (interpolation_denominator - 1.0)
                    * (bracket_ratio - 1.0)
                    * (residual_ratio - 1.0);
            }
            if interpolation_numerator > 0.0 {
                interpolation_denominator = -interpolation_denominator;
            } else {
                interpolation_numerator = -interpolation_numerator;
            }
            if 2.0 * interpolation_numerator
                < 3.0 * midpoint_delta * interpolation_denominator
                    - (tolerance * interpolation_denominator).abs()
                && 2.0 * interpolation_numerator < (previous_step * interpolation_denominator).abs()
            {
                previous_step = step;
                step = interpolation_numerator / interpolation_denominator;
            } else {
                step = midpoint_delta;
                previous_step = step;
            }
        } else {
            step = midpoint_delta;
            previous_step = step;
        }

        previous_mass = best_mass;
        previous_residual = best_residual;
        if step.abs() > tolerance {
            best_mass += step;
        } else {
            best_mass += tolerance.copysign(midpoint_delta);
        }

        hf_profile_inc_validate_refine_iteration();
        let next_best_residual = eval_hf(best_mass, HfProfileStage::ValidateRefine) - target;
        hf_calls = hf_calls.saturating_add(1);
        if !next_best_residual.is_finite() {
            return (best_mass, MassSolveStatusCode::HfValidateBrentNonFinite);
        }
        best_residual = next_best_residual;
        if best_residual >= 0.0 {
            safe_mass = best_mass;
            safe_residual = best_residual;
        }
    }

    (best_mass, MassSolveStatusCode::HfValidateBudgetExhausted)
}

/// Fast hybrid HF solve: use LF bisection to get a mass guess, then validate/repair with a
/// small number of HF evaluations.
///
/// This is intended for optimization loops where we need defensible HF behavior but cannot
/// afford 60+ HF propagations per event.
#[expect(
    clippy::arithmetic_side_effects,
    clippy::too_many_lines,
    reason = "linear validate-repair authority keeps status exits beside each physics gate"
)]
fn solve_single_event_hf_validate_only<O: MassSolveObserver>(
    event: &MassSolverEvent,
    config: &SolverConfig,
    hf_ctx: &HfContext,
    zero_mass_cache: Option<ZeroMassCacheView<'_>>,
    observer: &mut O,
) -> (f64, MassSolveStatusCode) {
    let target = event.min_miss_distance_km;
    lightyear_odeint_rs::probe::bump_stage(lightyear_odeint_rs::probe::STAGE_ROWS_VALIDATE_ONLY);
    let run_full_hf = |observer: &mut O| {
        let mut full_hf_ctx = hf_ctx.clone();
        full_hf_ctx.hf_validate_only = false;
        let outcome = solve_single_event_hf_internal(
            event,
            config,
            Some(&full_hf_ctx),
            zero_mass_cache,
            observer,
        );
        // AFTER the nested solve, not before it. `solve_single_event_hf_internal`
        // opens with `hf_profile_reset()`, so an increment on the near side of
        // this call is erased before any caller can read it -- which is what the
        // counter did from the day it was added, making every assertion on it a
        // comparison of two zeroes. The LF seed above resets the profile for the
        // same reason, so this is the only position on this path that survives.
        hf_profile_inc_lf_fallback();
        outcome
    };

    // 1) Cheap LF solve (no HF in the bisection loop).
    let lf_seed_config = SolverConfig {
        xtol: config.xtol,
        rtol: config.rtol,
        maxiter: config.maxiter.max(30),
        mass_max: config.mass_max,
    };
    let (mut seed_mass, lf_status) =
        solve_single_event_hf_with_status(event, &lf_seed_config, None);
    match lf_status {
        MassSolveStatusCode::SafeByDefault | MassSolveStatusCode::ConvergedNonPositive => {
            // LF can only seed strict HF.  A nonpositive LF boundary must still be
            // checked at mass=0 with the authoritative HF force model below.
            seed_mass = 0.0;
        }
        MassSolveStatusCode::Converged => {}
        // LF is only a performance seed.  If it cannot establish a usable
        // seed, strict mode must run the authoritative full-HF bracket instead
        // of classifying the event from lower-fidelity dynamics.
        _ => return run_full_hf(observer),
    }
    if !seed_mass.is_finite() || seed_mass < 0.0 {
        return run_full_hf(observer);
    }
    seed_mass = seed_mass.min(config.mass_max);

    // 2) Prepare HF context once and reuse in all HF evaluations.
    let prepared_result = prepare_hf_for_event(event, hf_ctx);
    let prepared = match prepared_result {
        Some(Ok(p)) => p,
        Some(Err(())) => return (f64::NAN, MassSolveStatusCode::HfValidateStrictPrepFailed),
        None => {
            if hf_ctx.hf_strict {
                return (f64::NAN, MassSolveStatusCode::HfValidateStrictPrepFailed);
            }
            // In non-strict mode, gracefully degrade to LF seed.
            return (seed_mass, MassSolveStatusCode::HfValidateFallbackLfSeed);
        }
    };
    let validated_velocity = compute_new_velocity(seed_mass, event);
    let validated_state = [
        event.p_pos_intercept[0],
        event.p_pos_intercept[1],
        event.p_pos_intercept[2],
        validated_velocity[0],
        validated_velocity[1],
        validated_velocity[2],
    ];
    let min_radius_km =
        prepared.force_config.earth_radius + lightyear_odeint_rs::types::GROUND_ALTITUDE;
    if !state_clears_min_radius(&validated_state, min_radius_km) {
        return (
            f64::NAN,
            MassSolveStatusCode::HfTrajectoryPhysicallyInfeasible,
        );
    }
    let zero_ref = zero_mass_reference_for_event(event, Some(&prepared), zero_mass_cache, observer);
    if hf_ctx.hf_strict && zero_ref.position.is_none() {
        hf_profile_set_anchor_diagnostics(DETMASS_ANCHOR_CONTRACT_VERSION, 0.0, false);
        return (
            f64::NAN,
            diagnose_miss_at_zero_failure(event, Some(&prepared), observer),
        );
    }
    let derived = derive_event_invariants(event, zero_ref.position);
    hf_profile_set_anchor_diagnostics(
        derived.anchor_contract_version,
        derived.anchor_shift_norm_km,
        derived.anchor_internal_reference_used,
    );
    let hf_timing_enabled = hf_counters_enabled();
    {
        let mut miss_memo = ExactMassMissMemo::default();
        let miss_zero_est = vec3_distance(&event.p_pos_conj_truth, &event.secondary_conj_pos);

        // 3) Validate with HF miss-distance evaluation.
        let mut hf_calls = 0usize;
        let hf_budget = config.maxiter.max(4);
        let mut eval_hf = |mass: f64, stage: HfProfileStage| {
            memoized_exact_mass_eval(mass, &mut miss_memo, || {
                let _probe =
                    lightyear_odeint_rs::probe::scope(lightyear_odeint_rs::probe::TAG_MASS_MISS);
                if hf_timing_enabled {
                    let started = Instant::now();
                    let miss = compute_miss_distance_hf_prepared_cached(
                        mass,
                        event,
                        &prepared,
                        &derived,
                        zero_mass_cache,
                        observer,
                    );
                    hf_profile_record(stage, started.elapsed().as_secs_f64());
                    miss
                } else {
                    compute_miss_distance_hf_prepared_cached(
                        mass,
                        event,
                        &prepared,
                        &derived,
                        zero_mass_cache,
                        observer,
                    )
                }
            })
        };

        let miss = eval_hf(seed_mass, HfProfileStage::ValidateInitial);
        hf_calls += 1;
        if !miss.is_finite() {
            return (
                f64::NAN,
                MassSolveStatusCode::HfValidateInitialMissNonFinite,
            );
        }
        let (lo_m, lo_miss, hi_m, hi_miss) = if miss >= target {
            if seed_mass <= 0.0 {
                return (0.0, MassSolveStatusCode::SafeByDefault);
            }
            // A safe LF seed is not proof of the minimum HF mass.  Establish
            // an HF lower endpoint before declaring a converged HF root.
            let miss_at_zero = exact_zero_mass_miss(zero_ref, &derived, &event.secondary_conj_pos)
                .unwrap_or_else(|| eval_hf(0.0, HfProfileStage::ValidateRepair));
            hf_calls += 1;
            if !miss_at_zero.is_finite() {
                return (f64::NAN, MassSolveStatusCode::HfValidateRepairMissNonFinite);
            }
            if miss_at_zero >= target {
                return (0.0, MassSolveStatusCode::SafeByDefault);
            }
            // The safe branch's bracket lower end is TRUE but useless: `[0,
            // seed_mass]` is ~313 `xtol` wide at the corpus median, so it
            // retires on neither the width arm nor the residual arm and every
            // safe row buys one Brent arc. The same two exact points that price
            // the correcting branch also locate the MINIMUM mass here, and on
            // 655 of 849 safe rows (p24, 2026-08-13) that root sits within one
            // `xtol` of the seed -- the refinement walks to a mass the tolerance
            // already declares identical to the one already in hand.
            if zero_ref.exact_hf
                && validate_safe_row_retires_at_seed(
                    seed_mass,
                    miss,
                    miss_at_zero,
                    target,
                    config.xtol,
                )
            {
                return (seed_mass, converged_status_for_mass(seed_mass));
            }
            (0.0, miss_at_zero, seed_mass, miss)
        } else {
            // 4) One-shot HF correction based on miss-distance deficit.
            // For small perturbation regimes, miss distance is approximately affine in mass.
            // Use an anchored affine estimate when the baseline miss is available.
            let raw_scale = if miss_zero_est.is_finite()
                && target > miss_zero_est + 1e-9
                && miss > miss_zero_est + 1e-9
            {
                (target - miss_zero_est) / (miss - miss_zero_est)
            } else {
                target / miss.max(1e-9)
            };
            let scale_min = if zero_ref.exact_hf {
                VALIDATE_CORRECTION_SCALE_MIN_EXACT
            } else {
                VALIDATE_CORRECTION_SCALE_MIN
            };
            let correction_scale = raw_scale.clamp(scale_min, VALIDATE_CORRECTION_SCALE_MAX);
            let (corrected_m, one_shot_floor) = validate_one_shot_target(
                seed_mass,
                correction_scale,
                config.xtol,
                config.mass_max,
                zero_ref.exact_hf,
            );
            let mut lo_m = seed_mass;
            let mut lo_miss = miss;
            let mut hi_m = seed_mass;
            let mut hi_miss = miss;
            if corrected_m > seed_mass + one_shot_floor && hf_calls < hf_budget {
                hi_m = corrected_m;
                hi_miss = eval_hf(hi_m, HfProfileStage::ValidateRepair);
                hf_calls += 1;
                if !hi_miss.is_finite() {
                    let Some((safe_probe_kg, safe_distance_km)) =
                        recover_finite_safe_validate_probe(
                            &mut eval_hf,
                            &mut hf_calls,
                            hf_budget,
                            lo_m,
                            hi_m,
                            config.xtol,
                            target,
                        )
                    else {
                        return (f64::NAN, MassSolveStatusCode::HfValidateRepairMissNonFinite);
                    };
                    hi_m = safe_probe_kg;
                    hi_miss = safe_distance_km;
                }
                if hi_miss < target {
                    lo_m = hi_m;
                    lo_miss = hi_miss;
                }
            }

            // 5) Repair: grow mass until safe (bounded number of HF calls).
            // Multiplicative growth keeps iteration count small when correction still undershoots.
            while hi_miss < target && hf_calls < hf_budget {
                let next_m = next_validate_repair_mass_candidate(
                    lo_m,
                    lo_miss,
                    hi_m,
                    hi_miss,
                    miss_zero_est,
                    target,
                    config.xtol,
                    config.mass_max,
                );
                if next_m <= hi_m + config.xtol {
                    break;
                }
                hi_m = next_m;
                hi_miss = eval_hf(hi_m, HfProfileStage::ValidateRepair);
                hf_calls += 1;
                if !hi_miss.is_finite() {
                    let Some((safe_probe_kg, safe_distance_km)) =
                        recover_finite_safe_validate_probe(
                            &mut eval_hf,
                            &mut hf_calls,
                            hf_budget,
                            lo_m,
                            hi_m,
                            config.xtol,
                            target,
                        )
                    else {
                        return (f64::NAN, MassSolveStatusCode::HfValidateRepairMissNonFinite);
                    };
                    hi_m = safe_probe_kg;
                    hi_miss = safe_distance_km;
                }
                if hi_miss < target {
                    lo_m = hi_m;
                    lo_miss = hi_miss;
                }
            }
            (lo_m, lo_miss, hi_m, hi_miss)
        };

        if hi_miss < target {
            if hi_m >= config.mass_max - config.xtol.max(1e-12) {
                // Physics-limited: finite maximum mass cannot reach target.
                return (
                    config.mass_max,
                    MassSolveStatusCode::HfValidatePhysicsLimited,
                );
            }
            return (hi_m, MassSolveStatusCode::HfValidateBudgetExhausted);
        }

        let gate_arm = validate_bracket_arm(
            lo_m,
            hi_m,
            hi_miss - target,
            config.xtol,
            config.rtol,
            target,
        );
        if gate_arm != VALIDATE_GATE_ARM_FELL_THROUGH {
            return (hi_m, converged_status_for_mass(hi_m));
        }

        refine_validate_hf_bracket(
            &mut eval_hf,
            hf_calls,
            hf_budget,
            lo_m,
            lo_miss,
            hi_m,
            hi_miss,
            config,
            target,
        )
    }
}

/// Solve for exact dust mass using Brent's method (with optional HF context).
#[expect(
    clippy::too_many_lines,
    reason = "Brent IEEE operation order is strict-HF result authority"
)]
fn solve_single_event_hf_internal<O: MassSolveObserver>(
    event: &MassSolverEvent,
    config: &SolverConfig,
    hf_ctx: Option<&HfContext>,
    zero_mass_cache: Option<ZeroMassCacheView<'_>>,
    observer: &mut O,
) -> (f64, MassSolveStatusCode) {
    hf_profile_reset();
    let _probe_solve = lightyear_odeint_rs::probe::mass_solve_scope();
    lightyear_odeint_rs::probe::bump_stage(lightyear_odeint_rs::probe::STAGE_ROWS_FULL);
    // Internal invariant: public native boundaries validate the configured study range.
    assert!(
        event.kappa > 0.0 && event.kappa <= 4.0,
        "kappa must be in (0.0, 4.0], got {}",
        event.kappa
    );
    if let Some(ctx) = hf_ctx {
        if ctx.is_hf_ready() && ctx.hf_validate_only {
            return solve_single_event_hf_validate_only(
                event,
                config,
                ctx,
                zero_mass_cache,
                observer,
            );
        }
    }

    // PERF2: Pre-compute HF force config once per event (outside root-finding loop).
    // This hoists ForceConfig clone, per-event overrides, and ephemeris resolution
    // out of the solver iterations.
    let mut strict_hf_prep_failed = false;
    let prepared_hf: Option<PreparedHfConfig> =
        hf_ctx.filter(|ctx| ctx.is_hf_ready()).and_then(|ctx| {
            match prepare_hf_for_event(event, ctx) {
                Some(Ok(p)) => Some(p),
                Some(Err(())) => {
                    strict_hf_prep_failed = true;
                    None
                }
                None => None, // incomplete HF context → LF fallback
            }
        });
    if let Some(ctx) = hf_ctx {
        if ctx.hf_strict
            && (strict_hf_prep_failed || (ctx.use_high_fidelity && prepared_hf.is_none()))
        {
            return (f64::NAN, MassSolveStatusCode::HfStrictPrepFailed);
        }
    }
    let prepared_ref = prepared_hf.as_ref();
    let anchor_reference =
        zero_mass_reference_for_event(event, prepared_ref, zero_mass_cache, observer);
    // Same guard as `solve_single_event_hf_validate_only`, for the same reason.
    // `derive_event_invariants` resolves a missing anchor with
    // `unwrap_or(event.p_pos_conj_equ_0)` -- the ANALYTIC equinoctial
    // conjunction position. That substitution is finite and plausible, so a
    // failed zero-mass anchor would otherwise leave a strict-HF row reporting a
    // mass that was solved against a different reference than the one strict
    // mode asked for, with nothing in the output saying so. In strict mode the
    // only honest answer is a typed failure.
    if let Some(ctx) = hf_ctx {
        if ctx.hf_strict && anchor_reference.position.is_none() {
            hf_profile_set_anchor_diagnostics(DETMASS_ANCHOR_CONTRACT_VERSION, 0.0, false);
            return (
                f64::NAN,
                diagnose_miss_at_zero_failure(event, prepared_ref, observer),
            );
        }
    }
    let derived = derive_event_invariants(event, anchor_reference.position);
    hf_profile_set_anchor_diagnostics(
        derived.anchor_contract_version,
        derived.anchor_shift_norm_km,
        derived.anchor_internal_reference_used,
    );
    let hf_timing_enabled = hf_counters_enabled();
    {
        let mut miss_memo = ExactMassMissMemo::default();
        let mut eval_miss = |mass: f64, stage: Option<HfProfileStage>| -> f64 {
            memoized_exact_mass_eval(mass, &mut miss_memo, || {
                prepared_ref.map_or_else(
                    || compute_miss_distance_lf(mass, event, &derived),
                    |prepared| {
                        let _probe = lightyear_odeint_rs::probe::scope(
                            lightyear_odeint_rs::probe::TAG_MASS_MISS,
                        );
                        if hf_timing_enabled {
                            let started = Instant::now();
                            let miss = compute_miss_distance_hf_prepared_cached(
                                mass,
                                event,
                                prepared,
                                &derived,
                                zero_mass_cache,
                                observer,
                            );
                            if let Some(s) = stage {
                                hf_profile_record(s, started.elapsed().as_secs_f64());
                            }
                            miss
                        } else {
                            compute_miss_distance_hf_prepared_cached(
                                mass,
                                event,
                                prepared,
                                &derived,
                                zero_mass_cache,
                                observer,
                            )
                        }
                    },
                )
            })
        };

        let target = event.min_miss_distance_km;

        // Bounds check
        let miss_at_zero = eval_miss(0.0, Some(HfProfileStage::FullBracket));
        if !miss_at_zero.is_finite() {
            let diag = diagnose_miss_at_zero_failure(event, prepared_ref, observer);
            return (f64::NAN, diag);
        }

        if miss_at_zero >= target {
            return (0.0, MassSolveStatusCode::SafeByDefault); // Safe by default
        }

        // Find a finite upper bound for miss distance.
        // Large masses can drive hyperbolic trajectories, which make equinoctial
        // propagation fail and return non-finite miss distances.
        // The MF-J2 preseed is wired to THIS path only -- the full-HF bracket,
        // which strict mode reaches only when the LF seed fails. Production
        // takes the validate-only path (`dust_hf_validate_only_enabled`), so
        // the preseed almost never runs where the cost actually is. That is
        // deliberate, not half-finished: extending it to the validate-only
        // path was measured and is worth 3-5% of a cell, but it MOVES THE
        // ANSWER, so it was rejected. See the plan file. Do not "finish" this
        // by hoisting it into the validate-only seed.
        let initial_upper_mass =
            mf_j2_preseed_upper_bound(event, config).unwrap_or(config.mass_max);
        let upper_search = find_finite_safe_upper_bound(
            initial_upper_mass,
            config.mass_max,
            config.xtol,
            target,
            |mass| eval_miss(mass, Some(HfProfileStage::FullBracket)),
        );
        let (upper_mass, miss_at_max) = match upper_search {
            UpperBoundSearch::Safe { mass, miss } => (mass, miss),
            UpperBoundSearch::PhysicsLimited { mass, miss } => {
                debug_assert!(miss < target);
                return (mass, MassSolveStatusCode::PhysicsLimitedUpperBound);
            }
            UpperBoundSearch::NonFinite => {
                return (f64::NAN, MassSolveStatusCode::UpperBoundNonFinite);
            }
        };

        // Brent's method root-finding (following SciPy brentq reference implementation).
        let mut previous_mass = 0.0_f64;
        let mut best_mass = upper_mass;
        let mut bracket_mass: f64;
        let mut previous_residual = miss_at_zero - target;
        let mut best_residual = miss_at_max - target;
        let mut bracket_residual: f64;
        let mut step: f64;
        let mut previous_step: f64;

        // Initialize: c is the contra-point of b (opposite sign).
        bracket_mass = previous_mass;
        bracket_residual = previous_residual;
        step = best_mass - previous_mass;
        previous_step = step;

        for _ in 0..config.maxiter {
            // Maintain bracket: ensure xc is contra-point of xb (opposite sign of fb).
            if (best_residual > 0.0 && bracket_residual > 0.0)
                || (best_residual < 0.0 && bracket_residual < 0.0)
            {
                bracket_mass = previous_mass;
                bracket_residual = previous_residual;
                step = best_mass - previous_mass;
                previous_step = step;
            }
            // Make xb the best approximation (smallest |f|).
            if bracket_residual.abs() < best_residual.abs() {
                previous_mass = best_mass;
                best_mass = bracket_mass;
                bracket_mass = previous_mass;
                previous_residual = best_residual;
                best_residual = bracket_residual;
                bracket_residual = previous_residual;
            }

            let tolerance = 2.0 * f64::EPSILON * best_mass.abs() + 0.5 * config.xtol;
            let midpoint_delta = 0.5 * (bracket_mass - best_mass);

            // `target.max(1.0)` floors on every production row, so the residual
            // arm is an absolute kilometre tolerance. See `SolverConfig`.
            if midpoint_delta.abs() <= tolerance
                || best_residual.abs() < config.rtol * target.max(1.0)
            {
                // Brent's own last secant is d(miss)/d(mass) at the root, free.
                // It converts an endpoint POSITION error on the free-flight leg
                // into a SOLVED MASS error, which is the only form of the
                // question that matters to the science answer.
                //
                // `prepared_ref.is_some()` is the whole point of this guard, and
                // it is not defensive. `prepared_hf` is `hf_ctx.filter(..)`
                // (`:2225-2235`), so it is `None` exactly when this Brent loop is
                // evaluating the ANALYTIC Kepler miss rather than the HF one --
                // which is what the LF seed at `:1933` does, once per row. Without
                // the guard this site published a two-body slope under an HF
                // label: measured 2026-08-05, `PROP_MASSSENS rows` was 3,397,
                // exactly `PROP_HFHIST` bucket 0, i.e. 100% LF seeds and not one
                // HF root. An endpoint error on the free-flight leg cannot be
                // converted by a slope the free-flight leg never produced.
                if prepared_ref.is_some() && (best_mass - previous_mass).abs() > 0.0 {
                    lightyear_odeint_rs::probe::record_mass_sensitivity(
                        best_mass,
                        (best_residual - previous_residual) / (best_mass - previous_mass),
                    );
                }
                return (best_mass, converged_status_for_mass(best_mass));
            }

            if previous_step.abs() >= tolerance && previous_residual.abs() > best_residual.abs() {
                // Try interpolation
                let residual_ratio = best_residual / previous_residual;
                let (mut interpolation_numerator, mut interpolation_denominator);
                if (previous_mass - bracket_mass).abs()
                    < f64::EPSILON * previous_mass.abs().max(1.0)
                {
                    // Secant (two equal points)
                    interpolation_numerator = 2.0 * midpoint_delta * residual_ratio;
                    interpolation_denominator = 1.0 - residual_ratio;
                } else {
                    // Inverse quadratic interpolation (three distinct points)
                    interpolation_denominator = previous_residual / bracket_residual;
                    let bracket_ratio = best_residual / bracket_residual;
                    interpolation_numerator = residual_ratio
                        * (2.0
                            * midpoint_delta
                            * interpolation_denominator
                            * (interpolation_denominator - bracket_ratio)
                            - (best_mass - previous_mass) * (bracket_ratio - 1.0));
                    interpolation_denominator = (interpolation_denominator - 1.0)
                        * (bracket_ratio - 1.0)
                        * (residual_ratio - 1.0);
                }
                // Normalize sign: ensure q > 0 so p/q gives correct direction
                if interpolation_numerator > 0.0 {
                    interpolation_denominator = -interpolation_denominator;
                } else {
                    interpolation_numerator = -interpolation_numerator;
                }
                // Accept interpolation only if it stays well within the bracket
                if 2.0 * interpolation_numerator
                    < 3.0 * midpoint_delta * interpolation_denominator
                        - (tolerance * interpolation_denominator).abs()
                    && 2.0 * interpolation_numerator
                        < (previous_step * interpolation_denominator).abs()
                {
                    previous_step = step;
                    step = interpolation_numerator / interpolation_denominator;
                } else {
                    step = midpoint_delta;
                    previous_step = step;
                }
            } else {
                // Bisection step
                step = midpoint_delta;
                previous_step = step;
            }

            // Move best guess
            previous_mass = best_mass;
            previous_residual = best_residual;
            if step.abs() > tolerance {
                best_mass += step;
            } else {
                best_mass += tolerance.copysign(midpoint_delta);
            }

            if prepared_ref.is_some() {
                hf_profile_inc_full_refine_iteration();
            }
            let next_best_residual =
                eval_miss(best_mass, Some(HfProfileStage::FullRefine)) - target;
            if !next_best_residual.is_finite() {
                let fallback = if previous_residual.abs() < best_residual.abs() {
                    previous_mass
                } else {
                    best_mass
                };
                return (fallback, MassSolveStatusCode::NonFiniteDuringBrent);
            }
            best_residual = next_best_residual;
        }

        (f64::NAN, MassSolveStatusCode::MaxIterReached)
    }
}

/// Solve for exact dust mass using Brent's method (with optional HF context).
#[must_use]
pub fn solve_single_event_hf(
    event: &MassSolverEvent,
    config: &SolverConfig,
    hf_ctx: Option<&HfContext>,
) -> f64 {
    let (mass, status) =
        solve_single_event_hf_internal(event, config, hf_ctx, None, &mut UnobservedMassSolve);
    if status == MassSolveStatusCode::Converged || status == MassSolveStatusCode::SafeByDefault {
        mass
    } else {
        f64::NAN
    }
}

/// Solve for exact dust mass and status code (with optional HF context).
#[must_use]
pub fn solve_single_event_hf_with_status(
    event: &MassSolverEvent,
    config: &SolverConfig,
    hf_ctx: Option<&HfContext>,
) -> (f64, MassSolveStatusCode) {
    solve_single_event_hf_internal(event, config, hf_ctx, None, &mut UnobservedMassSolve)
}

/// Run one authoritative W1 strict-HF batch with bounded actual-leg capture.
///
/// This is feature-only and deliberately serial: the fixed qualification
/// runner establishes the one global Rayon worker before entering this path.
/// The numerical loop is the production serial batch loop with its one
/// [`ZeroMassBatchCache`]; the observer sees only scalar calls that actually
/// initialize or evaluate that cache, never fabricated cache hits.
///
/// # Errors
///
/// Returns before any row work when the scheduler authority, row shapes,
/// strict-HF contexts, or caller-owned bounded storage are invalid. A
/// post-solve storage failure is returned before a caller can publish
/// incomplete evidence.
///
/// `expected_pool_threads` is the caller mode's compiled Rayon width. The rows
/// themselves always run serially through `solve_hf_rows_serial_cached`
/// regardless of that width, so a wider pool changes which events may run
/// concurrently, never how one batch is evaluated.
#[cfg(feature = "solver-qualification")]
pub fn solve_qualification_hf_batch_serial(
    events: &[MassSolverEvent],
    contexts: &[HfContext],
    config: &SolverConfig,
    expected_pool_threads: usize,
    observation: &mut QualificationMassBatchObservation<'_, '_>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        nd_sched::configured_global_pool_threads() == Some(expected_pool_threads),
        "qualification mass batch requires an authoritative global pool at its compiled width {expected_pool_threads}"
    );
    anyhow::ensure!(
        events.len() == contexts.len(),
        "qualification HF mass batch contexts length {} != events length {}",
        contexts.len(),
        events.len()
    );
    observation.preflight_rows(events.len())?;
    anyhow::ensure!(
        contexts
            .iter()
            .all(|context| context.is_hf_ready() && context.hf_strict),
        "qualification mass batch requires ready strict-HF contexts"
    );

    let rows = events
        .iter()
        .cloned()
        .zip(contexts.iter().cloned())
        .collect::<Vec<_>>();
    let mut masses = vec![0.0; rows.len()];
    let mut statuses = vec![MassSolveStatusCode::MissAtZeroNonFinite; rows.len()];
    let mut profiles = vec![HfMassSolveProfile::default(); rows.len()];
    solve_hf_rows_serial_cached(
        &rows,
        config,
        &mut masses,
        &mut statuses,
        &mut profiles,
        observation,
    )
}

/// Return telemetry for the most recent HF solve on the current thread.
pub(crate) fn last_hf_mass_solve_profile() -> HfMassSolveProfile {
    hf_profile_snapshot()
}

/// Execute ordered strict-HF rows through one shared zero-mass cache.
///
/// Normal serial batches pass no observer. Feature-only qualification passes
/// bounded caller-owned observation storage, but both routes use the exact
/// same cache construction, row timer, solver call, and profile capture.
fn solve_hf_rows_serial_cached<B: MassBatchObserver>(
    rows: &[(MassSolverEvent, HfContext)],
    config: &SolverConfig,
    masses: &mut [f64],
    statuses: &mut [MassSolveStatusCode],
    profiles: &mut [HfMassSolveProfile],
    observation: &mut B,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        masses.len() == rows.len(),
        "masses length {} != strict-HF rows length {}",
        masses.len(),
        rows.len()
    );
    anyhow::ensure!(
        statuses.len() == rows.len(),
        "statuses length {} != strict-HF rows length {}",
        statuses.len(),
        rows.len()
    );
    anyhow::ensure!(
        profiles.len() == rows.len(),
        "profiles length {} != strict-HF rows length {}",
        profiles.len(),
        rows.len()
    );
    observation.preflight_batch(rows.len())?;

    let zero_mass_cache = ZeroMassBatchCache::from_rows(rows);
    for (row_index, (((mass_out, status_out), profile_out), (event, hf_ctx))) in masses
        .iter_mut()
        .zip(statuses.iter_mut())
        .zip(profiles.iter_mut())
        .zip(rows.iter())
        .enumerate()
    {
        let (mass, status, leg_count) = {
            let mut row_observation = observation.open_row(row_index)?;
            let (mass, status) = solve_single_event_hf_internal(
                event,
                config,
                Some(hf_ctx),
                zero_mass_cache.slot_for_row(row_index),
                &mut row_observation,
            );
            let leg_count = row_observation.retained_legs();
            row_observation.seal()?;
            (mass, status, leg_count)
        };
        observation.seal_row(row_index, mass, status, leg_count)?;
        *mass_out = mass;
        *status_out = status;
        *profile_out = last_hf_mass_solve_profile();
    }
    Ok(())
}

fn zip_mass_statuses(
    masses: Vec<f64>,
    statuses: Vec<MassSolveStatusCode>,
) -> Vec<(f64, MassSolveStatusCode)> {
    let mut out = Vec::with_capacity(masses.len().min(statuses.len()));
    for (mass, status) in masses.into_iter().zip(statuses) {
        out.push((mass, status));
    }
    out
}

/// Solve strict-HF mass rows with their exact per-row force contexts.
///
/// Output preserves input order.  Top-level callers use process-global Rayon
/// only above the sealed batch threshold; nested callers stay serial.  Every
/// row retains its own `HfContext`, so separate exact arcs never share force
/// authority.  Internal zero-mass cache reuse is keyed by all authority inputs.
/// # Errors
///
/// Returns an error when row/context lengths or global Rayon width authority differ.
pub fn solve_batch_events_hf_with_status_per_context_global_rayon(
    events: &[MassSolverEvent],
    contexts: &[HfContext],
    config: &SolverConfig,
) -> anyhow::Result<Vec<(f64, MassSolveStatusCode)>> {
    anyhow::ensure!(
        events.len() == contexts.len(),
        "HF mass batch contexts length {} != events length {}",
        contexts.len(),
        events.len()
    );
    if events.is_empty() {
        return Ok(Vec::new());
    }

    let count = events.len();
    let mut masses = vec![0.0; count];
    let mut statuses = vec![MassSolveStatusCode::MissAtZeroNonFinite; count];
    let mut profiles = vec![HfMassSolveProfile::default(); count];
    let (rayon_threads, rayon_thread_budget) = (
        rayon::current_num_threads().max(1),
        satpy_core::parallel_budget::available_cores().max(1),
    );

    solve_batch_events_hf_with_status_and_profiles_from_builder_global_rayon(
        count,
        config,
        |row| {
            events
                .get(row)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("event row {row} is out of bounds"))
        },
        |row| {
            contexts
                .get(row)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("context row {row} is out of bounds"))
        },
        &mut masses,
        &mut statuses,
        &mut profiles,
        rayon_threads,
        rayon_thread_budget,
    )?;
    // The converged-mass dump used to be written here, from the only layer that
    // does NOT know which lowering row each mass belongs to -- which is why it
    // was keyed by capture position. It now happens at the strict-HF flush in
    // `nd_pipeline`, where the row identity exists.
    Ok(zip_mass_statuses(masses, statuses))
}

/// Solve one strict-HF batch and issue opaque evidence directly from each
/// eligible executed result.
///
/// # Errors
///
/// Returns before execution when any row lacks complete strict-HF authority,
/// or when the underlying batch authority rejects its shape or Rayon width.
pub fn solve_batch_events_hf_with_evidence_per_context_global_rayon(
    events: &[MassSolverEvent],
    contexts: &[HfContext],
    config: &SolverConfig,
) -> anyhow::Result<Vec<DeterministicMassSolveOutcome<MassSolveStatusCode>>> {
    anyhow::ensure!(
        events.len() == contexts.len(),
        "HF mass batch contexts length {} != events length {}",
        contexts.len(),
        events.len()
    );
    anyhow::ensure!(
        contexts
            .iter()
            .all(|context| context.hf_strict && context.is_hf_ready()),
        "strict-HF deterministic mass evidence requires complete strict-HF authority"
    );
    let solved =
        solve_batch_events_hf_with_status_per_context_global_rayon(events, contexts, config)?;
    Ok(solved
        .into_iter()
        .zip(events)
        .map(|((mass_kg, status), event)| {
            deterministic_mass_outcome(
                mass_kg,
                status,
                event.kappa,
                DeterministicMassRoute::StrictHf.as_str(),
                status == MassSolveStatusCode::Converged,
            )
        })
        .collect())
}

/// Execute one strict-HF batch on the process-global Rayon pool.
///
/// Top-level widths above one must match the latched global pool width. Width
/// one and nested invocations use the serial path. Indexed row collection and
/// slot-zipped parallel writes preserve input order.
pub(crate) fn solve_batch_events_hf_with_status_and_profiles_from_builder_global_rayon<F, C>(
    n_events: usize,
    config: &SolverConfig,
    build_event: F,
    build_context: C,
    masses: &mut [f64],
    statuses: &mut [MassSolveStatusCode],
    profiles: &mut [HfMassSolveProfile],
    rayon_threads: usize,
    rayon_thread_budget: usize,
) -> anyhow::Result<()>
where
    F: Fn(usize) -> anyhow::Result<MassSolverEvent> + Sync,
    C: Fn(usize) -> anyhow::Result<HfContext> + Sync,
{
    use rayon::prelude::*;

    anyhow::ensure!(rayon_threads > 0, "rayon_threads must be positive");
    anyhow::ensure!(
        rayon_thread_budget > 0,
        "rayon_thread_budget must be positive"
    );
    anyhow::ensure!(
        rayon_threads <= rayon_thread_budget,
        "rayon_threads {rayon_threads} exceeds rayon_thread_budget {rayon_thread_budget}"
    );
    let available_cores = satpy_core::parallel_budget::available_cores();
    anyhow::ensure!(
        rayon_thread_budget <= available_cores,
        "rayon_thread_budget {rayon_thread_budget} exceeds available cores {available_cores}"
    );
    if rayon_threads > 1 && rayon::current_thread_index().is_none() {
        let global_width = rayon::current_num_threads();
        anyhow::ensure!(
            rayon_threads == global_width,
            "requested Rayon width {rayon_threads} does not match global Rayon pool width {global_width}"
        );
    }
    anyhow::ensure!(
        masses.len() == n_events,
        "masses length {} != events length {n_events}",
        masses.len()
    );
    anyhow::ensure!(
        statuses.len() == n_events,
        "statuses length {} != events length {n_events}",
        statuses.len()
    );
    anyhow::ensure!(
        profiles.len() == n_events,
        "profiles length {} != events length {n_events}",
        profiles.len()
    );

    let use_parallel = rayon_threads > 1
        && rayon::current_thread_index().is_none()
        && n_events >= hf_batch_dispatch_parallel_threshold(HF_BATCH_PAR_THRESHOLD);
    if !use_parallel {
        let rows: Vec<_> = (0..n_events)
            .map(|i| Ok((build_event(i)?, build_context(i)?)))
            .collect::<anyhow::Result<_>>()?;
        solve_hf_rows_serial_cached(
            &rows,
            config,
            masses,
            statuses,
            profiles,
            &mut UnobservedMassSolve,
        )?;
        return Ok(());
    }

    let rows: Vec<_> = (0..n_events)
        .into_par_iter()
        .map(|i| Ok((build_event(i)?, build_context(i)?)))
        .collect::<anyhow::Result<_>>()?;
    let zero_mass_cache = ZeroMassBatchCache::from_rows(&rows);
    // Slot-major anchor pre-initialization: without it the flat row pass
    // below convoys on `OnceLock::get_or_init` at batch start (all rows of
    // one event share a single target-only anchor slot). Same pure
    // initializer, same inputs — bit-identical values; see the method doc.
    zero_mass_cache.preinitialize_hf_anchor_slots_parallel(&rows);
    masses
        .par_iter_mut()
        .zip(statuses.par_iter_mut())
        .zip(profiles.par_iter_mut())
        .zip(rows.par_iter())
        .enumerate()
        .for_each(
            |(i, (((mass_out, status_out), profile_out), (event, hf_ctx)))| {
                let (mass, status) = solve_single_event_hf_internal(
                    event,
                    config,
                    Some(hf_ctx),
                    zero_mass_cache.slot_for_row(i),
                    &mut UnobservedMassSolve,
                );
                *mass_out = mass;
                *status_out = status;
                *profile_out = last_hf_mass_solve_profile();
            },
        );
    Ok(())
}

#[cfg(test)]
fn solve_batch_events_mf_j2_with_status_into_from_builder<F>(
    n_events: usize,
    config: &SolverConfig,
    build_event: F,
    masses: &mut [f64],
    statuses: &mut [MfJ2MassSolveStatusCode],
    miss_zero: &mut [f64],
    miss_root: &mut [f64],
    miss_upper: &mut [f64],
    iterations: &mut [usize],
) where
    F: Fn(usize) -> MfJ2MassSolverEvent + Sync,
{
    use rayon::prelude::*;

    assert_eq!(masses.len(), n_events);
    assert_eq!(statuses.len(), n_events);
    assert_eq!(miss_zero.len(), n_events);
    assert_eq!(miss_root.len(), n_events);
    assert_eq!(miss_upper.len(), n_events);
    assert_eq!(iterations.len(), n_events);

    let use_parallel = should_use_mf_j2_batch_parallel_dispatch(n_events);
    if use_parallel {
        masses
            .par_iter_mut()
            .zip(statuses.par_iter_mut())
            .zip(miss_zero.par_iter_mut())
            .zip(miss_root.par_iter_mut())
            .zip(miss_upper.par_iter_mut())
            .zip(iterations.par_iter_mut())
            .enumerate()
            .for_each(
                |(
                    i,
                    (
                        ((((mass_out, status_out), miss_zero_out), miss_root_out), miss_upper_out),
                        iterations_out,
                    ),
                )| {
                    let event = build_event(i);
                    let result = solve_single_event_mf_j2_with_status(&event, config);
                    write_mf_j2_result(
                        result,
                        mass_out,
                        status_out,
                        miss_zero_out,
                        miss_root_out,
                        miss_upper_out,
                        iterations_out,
                    );
                },
            );
    } else {
        for (
            i,
            (
                ((((mass_out, status_out), miss_zero_out), miss_root_out), miss_upper_out),
                iterations_out,
            ),
        ) in masses
            .iter_mut()
            .zip(statuses.iter_mut())
            .zip(miss_zero.iter_mut())
            .zip(miss_root.iter_mut())
            .zip(miss_upper.iter_mut())
            .zip(iterations.iter_mut())
            .enumerate()
        {
            let event = build_event(i);
            let result = solve_single_event_mf_j2_with_status(&event, config);
            write_mf_j2_result(
                result,
                mass_out,
                status_out,
                miss_zero_out,
                miss_root_out,
                miss_upper_out,
                iterations_out,
            );
        }
    }
}

/// Returns `true` when the batch took the Rayon dispatch. The flag reports the
/// branch actually executed, not a re-evaluation of the predicate: an outer
/// `par_iter` makes `should_parallelize` refuse here, and the only way to see
/// that from outside is to observe the branch.
#[doc(hidden)]
pub fn solve_batch_events_mf_j2_with_status_into(
    events: &[MfJ2MassSolverEvent],
    config: &SolverConfig,
    masses: &mut [f64],
    statuses: &mut [MfJ2MassSolveStatusCode],
    miss_zero: &mut [f64],
    miss_root: &mut [f64],
    miss_upper: &mut [f64],
    iterations: &mut [usize],
) -> bool {
    let n_events = events.len();
    assert_eq!(masses.len(), n_events);
    assert_eq!(statuses.len(), n_events);
    assert_eq!(miss_zero.len(), n_events);
    assert_eq!(miss_root.len(), n_events);
    assert_eq!(miss_upper.len(), n_events);
    assert_eq!(iterations.len(), n_events);

    {
        use rayon::prelude::*;
        if should_use_mf_j2_batch_parallel_dispatch(n_events) {
            events
                .par_iter()
                .zip(masses.par_iter_mut())
                .zip(statuses.par_iter_mut())
                .zip(miss_zero.par_iter_mut())
                .zip(miss_root.par_iter_mut())
                .zip(miss_upper.par_iter_mut())
                .zip(iterations.par_iter_mut())
                .for_each(
                    |(
                        (
                            ((((event, mass_out), status_out), miss_zero_out), miss_root_out),
                            miss_upper_out,
                        ),
                        iterations_out,
                    )| {
                        write_mf_j2_result(
                            solve_single_event_mf_j2_with_status(event, config),
                            mass_out,
                            status_out,
                            miss_zero_out,
                            miss_root_out,
                            miss_upper_out,
                            iterations_out,
                        );
                    },
                );
            return true;
        }
    }

    for (
        (((((event, mass_out), status_out), miss_zero_out), miss_root_out), miss_upper_out),
        iterations_out,
    ) in events
        .iter()
        .zip(masses.iter_mut())
        .zip(statuses.iter_mut())
        .zip(miss_zero.iter_mut())
        .zip(miss_root.iter_mut())
        .zip(miss_upper.iter_mut())
        .zip(iterations.iter_mut())
    {
        write_mf_j2_result(
            solve_single_event_mf_j2_with_status(event, config),
            mass_out,
            status_out,
            miss_zero_out,
            miss_root_out,
            miss_upper_out,
            iterations_out,
        );
    }
    false
}

/// Execute an MF-J2 batch and issue opaque deterministic-mass evidence from
/// each eligible result produced by that exact batch.
///
/// The returned flag records whether the existing batch authority dispatched
/// through Rayon.
#[must_use]
pub fn solve_batch_events_mf_j2_with_evidence(
    events: &[MfJ2MassSolverEvent],
    config: &SolverConfig,
) -> (
    Vec<DeterministicMassSolveOutcome<MfJ2MassSolveStatusCode>>,
    bool,
) {
    let count = events.len();
    let mut masses = vec![0.0_f64; count];
    let mut statuses = vec![MfJ2MassSolveStatusCode::MissAtZeroNonFinite; count];
    let mut miss_zero = vec![0.0_f64; count];
    let mut miss_root = vec![0.0_f64; count];
    let mut miss_upper = vec![0.0_f64; count];
    let mut iterations = vec![0_usize; count];
    let dispatched_parallel = solve_batch_events_mf_j2_with_status_into(
        events,
        config,
        &mut masses,
        &mut statuses,
        &mut miss_zero,
        &mut miss_root,
        &mut miss_upper,
        &mut iterations,
    );
    let outcomes = masses
        .into_iter()
        .zip(statuses)
        .zip(events)
        .map(|((mass_kg, status), event)| {
            deterministic_mass_outcome(
                mass_kg,
                status,
                event.kappa,
                DeterministicMassRoute::MfJ2.as_str(),
                status == MfJ2MassSolveStatusCode::Converged,
            )
        })
        .collect();
    (outcomes, dispatched_parallel)
}

#[cfg(test)]
mod tests {

    /// The relative margin must still reproduce the LEO calibration it replaced.
    ///
    /// `ESCAPE_ENERGY_RELATIVE_TOL` is not a taste constant: it is
    /// `1e-6 / (MU / 6778 km)`, chosen so the hyperbolic test is UNCHANGED at
    /// the altitude the original absolute 1e-6 km^2/s^2 was calibrated for.
    /// Round it and you silently move a physics threshold at every altitude,
    /// including the one it was supposed to leave alone.
    #[test]
    fn escape_energy_relative_tol_reproduces_the_leo_calibration() {
        let leo_radius_km = 6778.0_f64;
        let leo_energy_scale = satpy_core::MU / leo_radius_km;
        let absolute_at_leo = super::ESCAPE_ENERGY_RELATIVE_TOL * leo_energy_scale;
        let relative_error = (absolute_at_leo - 1.0e-6).abs() / 1.0e-6;
        assert!(
            relative_error < 1.0e-12,
            "the margin at 400 km LEO is {absolute_at_leo} km^2/s^2, not the \
             1e-6 it replaced (relative error {relative_error})"
        );
    }
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn hf_preflight_failures_are_latched_for_owner_consumption() {
        let _ = take_hf_preflight_failure_count();
        latch_hf_preflight_failure();
        latch_hf_preflight_failure();
        assert!(take_hf_preflight_failure_count() >= 2);
    }

    fn fixture_usize_as_f64(value: usize) -> f64 {
        f64::from(u32::try_from(value).expect("fixture index fits u32"))
    }

    fn next_fixture_f64(value: f64) -> f64 {
        f64::from_bits(
            value
                .to_bits()
                .checked_add(1)
                .expect("fixture f64 bit pattern must not overflow"),
        )
    }

    fn same_or_both_nan(lhs: f64, rhs: f64) -> bool {
        lhs.to_bits() == rhs.to_bits() || (lhs.is_nan() && rhs.is_nan())
    }

    #[test]
    fn exact_float_gate_rejects_signed_zero_drift() {
        assert!(
            !same_or_both_nan(0.0, -0.0),
            "the exact serial/parallel gate must distinguish signed zero"
        );
    }

    #[test]
    fn operational_mass_policy_preserves_raw_bits_below_equal_and_above_floor() {
        let floor = MINIMUM_OPERATIONAL_DETERMINISTIC_MASS_KG;
        for (raw_mass_kg, expected_commanded_kg) in [
            (0.5 * floor, floor),
            (floor, floor),
            (2.0 * floor, 2.0 * floor),
        ] {
            let raw = deterministic_mass_outcome(
                raw_mass_kg,
                MfJ2MassSolveStatusCode::Converged,
                2.0,
                DeterministicMassRoute::MfJ2.as_str(),
                true,
            );
            let operational = raw
                .operational_mass()
                .expect("authentic raw evidence accepts fixed policy")
                .expect("converged raw evidence issues operational token");

            assert_eq!(
                operational.commanded_required_mass_kg().to_bits(),
                expected_commanded_kg.to_bits(),
            );
            assert_eq!(
                operational.raw_solver_mass_kg().to_bits(),
                raw_mass_kg.to_bits()
            );
            assert_eq!(raw.mass_kg().to_bits(), raw_mass_kg.to_bits());
            assert_eq!(
                raw.operational_mass()
                    .expect("raw outcome remains authentic after policy binding")
                    .expect("converged raw outcome still issues a token")
                    .raw_solver_mass_kg()
                    .to_bits(),
                raw_mass_kg.to_bits(),
                "operational policy must never rewrite raw evidence",
            );
        }

        let no_evidence = deterministic_mass_outcome(
            f64::NAN,
            MfJ2MassSolveStatusCode::MissAtZeroNonFinite,
            2.0,
            DeterministicMassRoute::MfJ2.as_str(),
            false,
        );
        assert_eq!(
            no_evidence
                .operational_mass()
                .expect("non-evidence outcome is not malformed"),
            None,
        );
    }

    #[test]
    fn operational_mass_policy_rejects_forged_internal_bits() {
        let forged = DeterministicMassSolveOutcome {
            mass_kg: 0.25,
            status: MfJ2MassSolveStatusCode::Converged,
            evidence: Some(DeterministicMassEvidence {
                required_mass_kg: 0.5,
                momentum_coupling_kappa: 2.0,
                mass_authority_id: DeterministicMassRoute::MfJ2.as_str(),
            }),
        };
        assert!(
            forged.operational_mass().is_err(),
            "raw outcome and opaque evidence must agree bit-for-bit",
        );
    }

    fn declares_a_submodule(line: &str) -> bool {
        let mut rest = line.trim_start();
        if let Some(after_pub) = rest.strip_prefix("pub") {
            let after_pub = after_pub
                .strip_prefix('(')
                .and_then(|restriction| restriction.split_once(')'))
                .map_or(after_pub, |(_, tail)| tail);
            if after_pub.starts_with(char::is_whitespace) {
                rest = after_pub.trim_start();
            }
        }
        let Some(after_mod) = rest.strip_prefix("mod") else {
            return false;
        };
        if !after_mod.starts_with(char::is_whitespace) {
            return false;
        }
        let name: String = after_mod
            .trim_start()
            .chars()
            .take_while(|character| character.is_alphanumeric() || *character == '_')
            .collect();
        !name.is_empty() && name != "tests"
    }

    #[test]
    fn production_source_has_no_unbounded_mass_batch_capture_surface() {
        // The mass solver is five files. Scanning only this one let a capture
        // surface land in any submodule and still read as green.
        let sources = [
            include_str!("mass_solver.rs"),
            include_str!("mass_solver/observer.rs"),
            include_str!("mass_solver/profile.rs"),
            include_str!("mass_solver/qualification.rs"),
            include_str!("mass_solver/status.rs"),
        ];
        // The list above is hand-copied, which is the next version of the same
        // bug: add a sixth submodule and it is silently unscanned. Count the
        // parent's own `mod` declarations instead of trusting it.
        let declared = sources[0]
            .lines()
            .map(declares_a_submodule)
            .filter(|declared| *declared)
            .count();
        assert_eq!(
            sources.len(),
            declared + 1,
            "capture scan reads {} files but the module declares {declared} submodules",
            sources.len()
        );
        let source = sources.concat();
        let forbidden = [
            ["ND_MASS", "_CAPTURE"].concat(),
            ["CAPTURED_HF", "_BATCH"].concat(),
            ["CAPTURE_HF_BATCH", "_ON"].concat(),
            ["capture_hf_batch", "_enabled"].concat(),
            ["captured_hf", "_batch"].concat(),
            ["clear_captured_hf", "_batch"].concat(),
            ["MassPhase", "Timer"].concat(),
            ["MASS_PHASE", "_"].concat(),
            ["add_stage", "_ns"].concat(),
        ];

        for symbol in forbidden {
            assert!(
                !source.contains(&symbol),
                "production mass solver still contains forbidden retired surface `{symbol}`"
            );
        }
    }

    fn assert_hf_profiles_equal(
        row: usize,
        actual: &HfMassSolveProfile,
        expected: &HfMassSolveProfile,
    ) {
        assert_eq!(
            (
                actual.hf_miss_calls_total,
                actual.hf_validate_initial_calls,
                actual.hf_validate_repair_calls,
                actual.hf_validate_refine_calls,
                actual.hf_validate_refine_iterations,
                actual.hf_full_bracket_calls,
                actual.hf_full_refine_calls,
                actual.hf_full_refine_iterations,
                actual.hf_upper_bracket_shrink_iterations,
                actual.hf_lf_fallback_count,
                actual.detmass_anchor_contract_version,
                actual.detmass_anchor_internal_reference_used,
            ),
            (
                expected.hf_miss_calls_total,
                expected.hf_validate_initial_calls,
                expected.hf_validate_repair_calls,
                expected.hf_validate_refine_calls,
                expected.hf_validate_refine_iterations,
                expected.hf_full_bracket_calls,
                expected.hf_full_refine_calls,
                expected.hf_full_refine_iterations,
                expected.hf_upper_bracket_shrink_iterations,
                expected.hf_lf_fallback_count,
                expected.detmass_anchor_contract_version,
                expected.detmass_anchor_internal_reference_used,
            ),
            "row {row} profile counter drift"
        );
        assert!(
            same_or_both_nan(
                actual.detmass_anchor_shift_norm_km,
                expected.detmass_anchor_shift_norm_km,
            ),
            "row {row} profile anchor drift"
        );
    }

    fn assert_global_hf_width_mismatch_is_rejected(config: &SolverConfig) {
        let mut empty_masses = [];
        let mut empty_statuses = [];
        let mut empty_profiles = [];
        let mismatch = solve_batch_events_hf_with_status_and_profiles_from_builder_global_rayon(
            0,
            config,
            |_| panic!("width mismatch must fail before event construction"),
            |_| panic!("width mismatch must fail before context construction"),
            &mut empty_masses,
            &mut empty_statuses,
            &mut empty_profiles,
            2,
            4,
        )
        .expect_err("requested W2 must reject latched global W4");
        assert!(
            mismatch.to_string().contains("global Rayon pool width 4"),
            "{mismatch}"
        );
    }

    fn assert_four_global_rayon_workers() {
        let global_worker_mask = std::sync::atomic::AtomicU64::new(0);
        let global_pool_width = std::sync::atomic::AtomicU64::new(0);
        let global_worker_barrier = std::sync::Barrier::new(4);
        (0..4_usize).into_par_iter().for_each(|_| {
            global_pool_width.store(
                u64::try_from(rayon::current_num_threads()).expect("Rayon width fits u64"),
                std::sync::atomic::Ordering::Relaxed,
            );
            if let Some(worker) = rayon::current_thread_index() {
                global_worker_mask.fetch_or(1_u64 << worker, std::sync::atomic::Ordering::Relaxed);
            }
            global_worker_barrier.wait();
        });
        assert_eq!(
            global_pool_width.load(std::sync::atomic::Ordering::Relaxed),
            4,
            "direct global par_iter must observe four workers"
        );
        assert!(
            global_worker_mask
                .load(std::sync::atomic::Ordering::Relaxed)
                .count_ones()
                > 1,
            "direct global par_iter must execute on multiple workers"
        );
    }

    const GLOBAL_DETMASS_TEST_CHILD_ENV: &str = "NASA_DUST_GLOBAL_DETMASS_TEST_CHILD";
    const GLOBAL_DETMASS_TEST_CHILD_MARKER: &str = "NASA_DUST_GLOBAL_DETMASS_CHILD_EXECUTED";

    fn spawn_four_global_pool_child_if_parent() -> bool {
        if let Some(width) = std::env::var_os(GLOBAL_DETMASS_TEST_CHILD_ENV) {
            let width = width
                .to_string_lossy()
                .parse::<usize>()
                .expect("child width must parse");
            assert_eq!(nd_sched::init_global_pool(Some(width)), width);
            assert_eq!(rayon::current_num_threads(), width);
            println!("{GLOBAL_DETMASS_TEST_CHILD_MARKER}");
            return false;
        }

        let output = std::process::Command::new(
            std::env::current_exe().expect("current Rust test executable"),
        )
        .args([
            "mass_solver::tests::hf_batch_uses_four_thread_global_pool_without_reordering",
            "--exact",
            "--nocapture",
        ])
        .env(GLOBAL_DETMASS_TEST_CHILD_ENV, "4")
        .env("RUST_TEST_THREADS", "1")
        .output()
        .expect("spawn isolated four-global-thread child test");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "child failed\nstdout:\n{stdout}\nstderr:\n{stderr}",
        );
        assert!(
            stdout.contains(GLOBAL_DETMASS_TEST_CHILD_MARKER),
            "child reported success without executing test\nstdout:\n{stdout}\nstderr:\n{stderr}",
        );
        true
    }

    fn push_one_bit_array_variants<const N: usize>(
        variants: &mut Vec<MassSolverEvent>,
        event: &MassSolverEvent,
        field: for<'event> fn(&'event mut MassSolverEvent) -> &'event mut [f64; N],
    ) {
        for index in 0..N {
            let mut changed = event.clone();
            let value = field(&mut changed).get_mut(index);
            assert!(value.is_some(), "fixture component {index} must exist");
            let Some(value) = value else {
                return;
            };
            *value = next_fixture_f64(*value);
            variants.push(changed);
        }
    }

    #[test]
    fn target_propagation_authority_dispatches_kepler_j2_and_hf_exactly() {
        let state = [7000.0, 0.0, 0.0, 0.0, 7.45, 1.0];
        let tof_s = 86_400.0;
        let kepler = propagate_target_for_mass_authority(
            &state,
            tof_s,
            TargetPropagationAuthority::AnalyticalKepler,
            None,
            &mut UnobservedMassSolve,
            MassLegTag {
                role: MassLegRole::MassEvaluation,
                mass_kg_bits: 0.0_f64.to_bits(),
            },
        )
        .expect("Kepler authority");
        let j2 = propagate_target_for_mass_authority(
            &state,
            tof_s,
            TargetPropagationAuthority::MfJ2,
            None,
            &mut UnobservedMassSolve,
            MassLegTag {
                role: MassLegRole::MassEvaluation,
                mass_kg_bits: 0.0_f64.to_bits(),
            },
        )
        .expect("J2 authority");

        assert!(kepler.iter().all(|value| value.is_finite()));
        assert!(j2.iter().all(|value| value.is_finite()));
        assert!(
            kepler
                .iter()
                .zip(j2.iter())
                .any(|(lhs, rhs)| lhs.to_bits() != rhs.to_bits()),
            "Kepler and secular J2 target endpoints must remain distinct"
        );
        assert_eq!(
            propagate_target_for_mass_authority(
                &state,
                tof_s,
                TargetPropagationAuthority::HighFidelity,
                None,
                &mut UnobservedMassSolve,
                MassLegTag {
                    role: MassLegRole::MassEvaluation,
                    mass_kg_bits: 0.0_f64.to_bits(),
                },
            ),
            Err(TargetMassPropagationError::HighFidelityContextRequired),
            "HF authority must fail closed instead of falling back"
        );
    }
    use std::cell::Cell;

    /// Drive ONE production solve twice over an anchor whose POSITION is
    /// byte-identical and whose `exact_hf` flag is the only thing that differs,
    /// and pin that the floor selector at the one-shot reads it.
    ///
    /// The first version of this test clamped the constants itself and asserted
    /// on the result. That is a test-local shadow: it re-implemented the
    /// production expression instead of running it, so deleting the selector
    /// entirely -- hardcoding the old 1.2 floor for every anchor -- left the
    /// suite 95/95 green while the headline lever (the floor fired on 395 of
    /// 395 correcting rows) went unguarded. Everything below therefore goes
    /// through `solve_single_event_hf_internal`, and the arms differ by one
    /// cached boolean.
    #[test]
    fn the_one_shot_floor_selector_reads_the_anchors_exact_hf_flag() {
        let config = SolverConfig {
            xtol: 1e-6,
            rtol: 1e-5,
            maxiter: 80,
            mass_max: 1_000.0,
        };
        let mut context = zero_mass_test_context();
        context.hf_validate_only = true;

        // `(mass, status, repair_calls)` for one anchor-flag setting.
        let solve_with_anchor_flag = |event: &MassSolverEvent, position, exact_hf| {
            let anchor: OnceLock<ZeroMassReference> = OnceLock::new();
            anchor
                .set(ZeroMassReference { position, exact_hf })
                .expect("a fresh anchor slot accepts one write");
            let miss_at_zero: OnceLock<f64> = OnceLock::new();
            let view = ZeroMassCacheView {
                anchor_reference: &anchor,
                miss_at_zero: &miss_at_zero,
            };
            let (mass, status) = solve_single_event_hf_internal(
                event,
                &config,
                Some(&context),
                Some(view),
                &mut UnobservedMassSolve,
            );
            (
                mass,
                status,
                last_hf_mass_solve_profile().hf_validate_repair_calls,
            )
        };

        let correcting_event = |offset_km: f64, target_km: f64| {
            let mut event = valid_zero_mass_test_event();
            event.secondary_conj_pos[0] = event.p_pos_conj_truth[0] + offset_km;
            event.secondary_conj_pos[1] = event.p_pos_conj_truth[1];
            event.secondary_conj_pos[2] = event.p_pos_conj_truth[2];
            event.min_miss_distance_km = target_km;
            event
        };

        // The anchor POSITION both arms see: the real HF zero-mass propagation.
        // Taking it once and feeding it to both arms is what makes `exact_hf`
        // the only variable -- an arm that recomputed its own anchor would also
        // be testing the propagation.
        let truth_position = |event: &MassSolverEvent| {
            let Some(Ok(prepared)) = prepare_hf_for_event(event, &context) else {
                panic!("the strict-HF fixture must prepare");
            };
            let reference = zero_mass_reference_for_event_uncached(
                event,
                Some(&prepared),
                &mut UnobservedMassSolve,
            );
            assert!(
                reference.exact_hf && reference.position.is_some(),
                "the fixture's own anchor must be the exact HF propagation"
            );
            reference.position
        };

        // Both arms buy the one repair arc, but they spend it in different
        // places: the exact anchor aims at the affine root (`used_scale` 1.0,
        // landing 2.0e-7 of the target away) and the LF anchor at the floor
        // (`used_scale` 1.2, landing 2.4e-1 away). The converged masses
        // therefore differ. Delete the selector and the two arms become the
        // same solve, which is what the bit inequality below detects. Their
        // separation is intentionally not bounded: one-sided convergence now
        // returns each arm's proven-safe endpoint rather than an unsafe closer
        // endpoint merely because its residual is smaller.
        //
        // Deliberately NOT asserted here: an arc count. On this fixture the LF
        // seed's own residual is already inside `rtol`, so the refinement
        // retires on its entry check and BOTH arms spend zero refine arcs --
        // the production saving needs a corpus whose seeds are not already
        // exact, which is what the harness measurement covers. An arc
        // assertion on this fixture would pass for a reason unrelated to the
        // selector; a first draft of this test carried one and survived the
        // selector-deletion poison because of it.
        let event = correcting_event(0.1, 0.5);
        let position = truth_position(&event);
        let (exact_mass, exact_status, exact_repair) =
            solve_with_anchor_flag(&event, position, true);
        let (lf_mass, lf_status, lf_repair) = solve_with_anchor_flag(&event, position, false);
        assert_eq!(exact_status, MassSolveStatusCode::Converged);
        assert_eq!(lf_status, MassSolveStatusCode::Converged);
        assert_eq!((exact_repair, lf_repair), (1, 1));
        assert!(
            exact_mass.to_bits() != lf_mass.to_bits(),
            "the anchor flag must reach the aim: both arms landed on {exact_mass:.17e}"
        );
    }

    /// A sub-`xtol` affine step must leave a bracket the absolute-width arm
    /// accepts, so the evaluation that establishes the upper endpoint also
    /// retires the row. This is the whole saving: without it the row falls
    /// through to a Brent refinement it does not need.
    #[test]
    fn a_sub_xtol_affine_step_lands_inside_the_absolute_width_arm() {
        let xtol = 1e-6;
        let seed = 1.0e-3;
        // 0.43 xtol is the corpus median step; 0.02 and 0.99 are its extremes.
        for step_in_xtol in [0.02, 0.43, 0.99] {
            let scale = 1.0 + step_in_xtol * xtol / seed;
            let (target_m, floor) = validate_one_shot_target(seed, scale, xtol, 1.0e6, true);
            let width = target_m - seed;
            assert!(
                width > 0.0 && width <= xtol,
                "step {step_in_xtol} xtol left a bracket of {width}, outside (0, {xtol}]"
            );
            assert!(
                validate_bracket_arm(seed, target_m, 1.0, xtol, 1e-5, 1.0)
                    == VALIDATE_GATE_ARM_ABS_WIDTH,
                "step {step_in_xtol} xtol must retire on the absolute-width arm"
            );
            assert!(
                target_m > seed + floor,
                "the one-shot must still fire: {target_m} vs {}",
                seed + floor
            );
            // Above the affine root, not merely near it: the endpoint has to be
            // the SAFE side or the growth loop pays for another arc.
            assert!(
                target_m > seed * scale,
                "step {step_in_xtol} xtol aimed below the affine root"
            );
        }
    }

    /// The narrow arm is reachable only through the exact-HF anchor, and only
    /// for a step the tolerance already covers. Everything else keeps the
    /// shipped aim and the shipped `xtol` firing floor.
    #[test]
    fn the_narrow_step_does_not_capture_lf_anchors_or_ordinary_corrections() {
        let xtol = 1e-6;
        let seed = 1.0e-3;
        let narrow_scale = 1.0 + 0.4 * xtol / seed;

        let (lf_m, lf_floor) = validate_one_shot_target(seed, narrow_scale, xtol, 1.0e6, false);
        assert!(
            (lf_m - seed * narrow_scale).abs() < f64::EPSILON * seed
                && (lf_floor - xtol).abs() < f64::EPSILON,
            "an LF anchor must keep the shipped aim and firing floor"
        );

        // A correction well above `xtol` is an ordinary bracket-widening step
        // and must not be pulled down into the tolerance interval.
        let (wide_m, wide_floor) = validate_one_shot_target(seed, 1.2, xtol, 1.0e6, true);
        assert!(
            (wide_m - seed * 1.2).abs() < f64::EPSILON && (wide_floor - xtol).abs() < f64::EPSILON,
            "a supra-xtol correction must keep the shipped aim, got {wide_m}"
        );

        // `mass_max` still caps, and a capped estimate that cannot advance must
        // not be rewritten into a narrow step.
        let (capped_m, _) = validate_one_shot_target(seed, 1.2, xtol, seed, true);
        assert!(
            (capped_m - seed).abs() < f64::EPSILON,
            "mass_max must still cap the one-shot, got {capped_m}"
        );
    }

    /// The safe branch's bracket lower end is TRUE but useless, and the affine
    /// minimum is what says so. A root inside `xtol` of the seed means the
    /// refinement would return a mass the tolerance calls identical; a root
    /// well below it is real work that must still be done.
    #[test]
    fn a_safe_row_retires_at_the_seed_only_when_the_affine_minimum_is_within_xtol() {
        let xtol = 1e-6;
        let seed = 1.0e-3;
        let target = 1.0;
        let miss_at_zero = 2.2e-2;
        // Place the seed's miss so the affine root sits a chosen distance below
        // it: root = seed * (target - mz) / (miss - mz).
        let miss_for_gap = |gap_in_xtol: f64| {
            let root = seed - gap_in_xtol * xtol;
            miss_at_zero + (target - miss_at_zero) * seed / root
        };

        for gap in [0.03, 0.36, 0.99] {
            let miss = miss_for_gap(gap);
            assert!(miss > target, "a safe row must overshoot the target");
            assert!(
                validate_safe_row_retires_at_seed(seed, miss, miss_at_zero, target, xtol),
                "gap {gap} xtol is inside the tolerance and must retire at the seed"
            );
        }
        for gap in [1.5, 10.0, 1.0e3] {
            let miss = miss_for_gap(gap);
            assert!(
                !validate_safe_row_retires_at_seed(seed, miss, miss_at_zero, target, xtol),
                "gap {gap} xtol is real work and must still be refined"
            );
        }

        // The retired mass never understates the minimum: the root is below the
        // seed, so `seed_mass` is the conservative side.
        let root = validate_affine_root(seed, miss_for_gap(0.36), miss_at_zero, target)
            .expect("a safe row with a finite span has an affine root");
        assert!(root < seed && seed - root <= xtol);

        // Degenerate spans produce no root rather than a bogus one.
        assert!(validate_affine_root(seed, miss_at_zero, miss_at_zero, target).is_none());
        assert!(validate_affine_root(seed, 1.5, miss_at_zero, miss_at_zero).is_none());
    }

    /// The safe-branch early accept, driven through production over an anchor
    /// whose POSITION is byte-identical and whose `exact_hf` flag is the only
    /// difference -- the same two-arm harness as the floor selector above, and
    /// for the same reason: a pure-function test of
    /// `validate_safe_row_retires_at_seed` cannot show that the solver consults
    /// the anchor before trusting it.
    ///
    /// The tolerances are chosen, not inherited. `rtol` is tight enough that the
    /// refinement's entry check does NOT retire this row on residual, so a row
    /// that reaches the refinement really does spend an arc there; `xtol` is
    /// loose enough that the affine minimum falls inside it, which is the
    /// condition the early accept tests. At production tolerances this fixture's
    /// seed is already residual-exact and both arms would spend zero refine arcs
    /// for a reason that has nothing to do with the anchor.
    #[test]
    fn only_an_exact_anchor_retires_a_safe_row_before_the_refinement() {
        let config = SolverConfig {
            xtol: 1e-3,
            rtol: 1e-12,
            maxiter: 80,
            mass_max: 1_000.0,
        };
        let mut context = zero_mass_test_context();
        context.hf_validate_only = true;

        let mut event = valid_zero_mass_test_event();
        event.secondary_conj_pos[0] = event.p_pos_conj_truth[0] + 0.05;
        event.secondary_conj_pos[1] = event.p_pos_conj_truth[1];
        event.secondary_conj_pos[2] = event.p_pos_conj_truth[2];
        event.min_miss_distance_km = 0.5;

        let Some(Ok(prepared)) = prepare_hf_for_event(&event, &context) else {
            panic!("the strict-HF fixture must prepare");
        };
        let truth = zero_mass_reference_for_event_uncached(
            &event,
            Some(&prepared),
            &mut UnobservedMassSolve,
        );
        assert!(
            truth.exact_hf && truth.position.is_some(),
            "the fixture's own anchor must be the exact HF propagation"
        );

        let solve = |exact_hf: bool| {
            let anchor: OnceLock<ZeroMassReference> = OnceLock::new();
            anchor
                .set(ZeroMassReference {
                    position: truth.position,
                    exact_hf,
                })
                .expect("a fresh anchor slot accepts one write");
            let miss_at_zero: OnceLock<f64> = OnceLock::new();
            let view = ZeroMassCacheView {
                anchor_reference: &anchor,
                miss_at_zero: &miss_at_zero,
            };
            let (mass, status) = solve_single_event_hf_internal(
                &event,
                &config,
                Some(&context),
                Some(view),
                &mut UnobservedMassSolve,
            );
            (
                mass,
                status,
                last_hf_mass_solve_profile().hf_validate_refine_calls,
            )
        };

        let (exact_mass, exact_status, exact_refine) = solve(true);
        let (lf_mass, lf_status, lf_refine) = solve(false);

        assert_eq!(exact_status, MassSolveStatusCode::Converged);
        assert_eq!(lf_status, MassSolveStatusCode::Converged);
        assert_eq!(
            exact_refine, 0,
            "an exact anchor whose affine minimum is inside xtol must retire the \
             safe row before the refinement, not after it"
        );
        assert_eq!(
            lf_refine, 1,
            "the LF anchor has no trustworthy minimum, so it must still pay the \
             refinement arc; both arms retiring early means the anchor flag is \
             not being consulted"
        );
        assert!(
            (exact_mass - lf_mass).abs() <= config.xtol,
            "the skipped refinement must not move the answer outside xtol: \
             exact={exact_mass:.17e} lf={lf_mass:.17e}"
        );
        // Conservative direction: the early accept returns the HF-evaluated
        // seed, which is at or above the refined minimum, never below it.
        assert!(
            exact_mass >= lf_mass,
            "the early accept must never understate the minimum mass"
        );
    }

    #[test]
    fn test_validate_repair_growth_is_adaptive_and_bounded() {
        // Large miss deficit should trigger growth stronger than fixed doubling
        // while still respecting mass_max.
        let next = next_validate_repair_mass_candidate(
            5.0,   // lo_m
            0.10,  // lo_miss
            10.0,  // hi_m
            0.20,  // hi_miss
            0.05,  // miss_zero_est
            5.0,   // target
            1e-6,  // xtol
            500.0, // mass_max
        );
        assert!(next > 20.0, "expected adaptive growth stronger than x2");
        assert!(next <= 500.0);

        // Near target should still advance monotonically above hi_m.
        let near = next_validate_repair_mass_candidate(
            9.0,   // lo_m
            4.7,   // lo_miss
            10.0,  // hi_m
            4.9,   // hi_miss
            4.0,   // miss_zero_est
            5.0,   // target
            1e-6,  // xtol
            500.0, // mass_max
        );
        assert!(near > 10.0);
        assert!(near <= 500.0);
    }

    #[test]
    fn test_validate_bracket_convergence_short_circuit() {
        // Tight mass bracket should be considered converged without Brent.
        assert!(validate_bracket_is_converged(
            10.0,        // lo_m
            10.0 + 1e-7, // hi_m
            0.1,         // hi_residual
            1e-6,        // xtol
            1e-6,        // rtol
            5.0,         // target
        ));

        // Large residual and wide bracket should not short-circuit.
        assert!(!validate_bracket_is_converged(
            10.0, // lo_m
            15.0, // hi_m
            2.0,  // hi_residual
            1e-6, // xtol
            1e-6, // rtol
            5.0,  // target
        ));
    }

    #[test]
    fn validate_refinement_convergence_returns_the_safe_endpoint() {
        let config = SolverConfig {
            xtol: 4.0,
            rtol: 1.0e-12,
            maxiter: 4,
            mass_max: 100.0,
        };
        let mut calls = Vec::new();
        let (mass, status) = refine_validate_hf_bracket(
            |probe, _stage| {
                calls.push(probe);
                if probe >= 11.0 {
                    2.0
                } else {
                    0.99
                }
            },
            0,
            4,
            10.0,
            0.99,
            11.0,
            2.0,
            &config,
            1.0,
        );

        assert_eq!(status, MassSolveStatusCode::Converged);
        assert_eq!(
            mass.to_bits(),
            11.0_f64.to_bits(),
            "convergence must return the finite endpoint proven safe",
        );
        assert!(calls.is_empty(), "entry convergence needs no new HF arc");
    }

    #[test]
    fn validate_nonfinite_repair_recovers_only_a_new_finite_safe_point() {
        let mut calls = Vec::new();
        let mut hf_calls = 1;
        let recovered = recover_finite_safe_validate_probe(
            &mut |mass, stage| {
                assert!(matches!(stage, HfProfileStage::ValidateRepair));
                calls.push(mass);
                if mass > 8.0 {
                    f64::NAN
                } else if mass >= 7.0 {
                    1.25
                } else {
                    0.75
                }
            },
            &mut hf_calls,
            4,
            5.0,
            10.0,
            1.0e-6,
            1.0,
        );

        let (safe_probe_kg, safe_distance_km) =
            recovered.expect("finite safe point exists below nonfinite probe");
        assert_eq!(safe_probe_kg.to_bits(), 7.5_f64.to_bits());
        assert_eq!(safe_distance_km.to_bits(), 1.25_f64.to_bits());
        assert_eq!(hf_calls, 2);
        assert_eq!(calls, vec![7.5]);
    }

    #[test]
    fn validate_nonfinite_repair_stays_bounded_without_a_safe_point() {
        let mut calls = Vec::new();
        let mut hf_calls = 1;
        let recovered = recover_finite_safe_validate_probe(
            &mut |mass, stage| {
                assert!(matches!(stage, HfProfileStage::ValidateRepair));
                calls.push(mass);
                if mass > 7.0 {
                    f64::NAN
                } else {
                    0.75
                }
            },
            &mut hf_calls,
            4,
            5.0,
            10.0,
            1.0e-6,
            1.0,
        );

        assert_eq!(recovered, None);
        assert_eq!(hf_calls, 4, "repair must share the existing HF budget");
        assert_eq!(calls.len(), 3);
    }

    /// `rtol` must not be able to retire a bracket on its WIDTH.
    ///
    /// The two tolerances were interchangeable in the width arms for as long as
    /// both held 1e-6, so nothing distinguished them and relaxing `rtol` would
    /// have widened the accepted mass bracket by the same factor as the
    /// miss-distance residual. The bracket below is 100x wider than `xtol` and
    /// 10x narrower than `rtol`, which is exactly the band that opened up when
    /// `rtol` moved to 1e-5; its residual is far too large for the residual arm,
    /// so a fired arm here can only be a width arm reading `rtol`.
    #[test]
    fn validate_bracket_width_arms_ignore_rtol() {
        let arm = validate_bracket_arm(
            0.5,        // lo_m
            0.5 + 1e-4, // hi_m -- 100x xtol, 10x under rtol
            7.0,        // hi_residual, km: no residual arm can fire on this
            1e-6,       // xtol
            1e-3,       // rtol
            5.0,        // target
        );
        assert_eq!(
            arm, VALIDATE_GATE_ARM_FELL_THROUGH,
            "a bracket wider than xtol must reach refine however loose rtol is"
        );

        // The same bracket against a matching xtol DOES retire, so the
        // assertion above is about which tolerance is read, not about the arm
        // being unreachable.
        assert_eq!(
            validate_bracket_arm(0.5, 0.5 + 1e-4, 7.0, 1e-3, 1e-3, 5.0),
            VALIDATE_GATE_ARM_ABS_WIDTH,
        );
    }

    #[test]
    fn full_hf_upper_search_expands_beyond_unsafe_mf_preseed() {
        let mut probes = Vec::new();
        let result = find_finite_safe_upper_bound(10.0, 100.0, 1e-8, 50.0, |mass| {
            probes.push(mass);
            mass
        });

        match result {
            UpperBoundSearch::Safe { mass, miss } => {
                assert!(mass >= 50.0);
                assert!(miss >= 50.0);
            }
            other => panic!("expected safe expanded bracket, got {other:?}"),
        }
        assert!(probes.iter().any(|mass| *mass > 10.0));
    }

    #[test]
    fn full_hf_upper_search_finds_safe_mass_below_nonfinite_ceiling() {
        let result = find_finite_safe_upper_bound(100.0, 100.0, 1e-8, 70.0, |mass| {
            if mass > 80.0 {
                f64::NAN
            } else {
                mass
            }
        });

        assert!(matches!(result, UpperBoundSearch::Safe { mass, .. } if mass >= 70.0));
    }

    #[test]
    fn test_exact_mass_memo_uses_bit_pattern_and_hits_repeat() {
        let mut memo = ExactMassMissMemo::default();
        let eval_calls = Cell::new(0usize);

        let first = memoized_exact_mass_eval(12.5, &mut memo, || {
            eval_calls.set(eval_calls.get() + 1);
            42.0
        });
        let second = memoized_exact_mass_eval(12.5, &mut memo, || {
            eval_calls.set(eval_calls.get() + 1);
            99.0
        });
        assert_eq!(first.to_bits(), 42.0_f64.to_bits());
        assert_eq!(second.to_bits(), 42.0_f64.to_bits());
        assert_eq!(eval_calls.get(), 1, "repeat mass should hit memo");

        // +0.0 and -0.0 are numerically equal but have different bit patterns.
        let pos_zero = memoized_exact_mass_eval(0.0, &mut memo, || {
            eval_calls.set(eval_calls.get() + 1);
            1.0
        });
        let neg_zero = memoized_exact_mass_eval(-0.0, &mut memo, || {
            eval_calls.set(eval_calls.get() + 1);
            2.0
        });
        assert_eq!(pos_zero.to_bits(), 1.0_f64.to_bits());
        assert_eq!(neg_zero.to_bits(), 2.0_f64.to_bits());
        assert_eq!(
            eval_calls.get(),
            3,
            "bit-distinct masses must be stored independently"
        );
    }

    #[test]
    fn test_exact_mass_memo_capacity_is_bounded_and_fifo() {
        let mut memo = ExactMassMissMemo::default();
        for i in 0..EXACT_MASS_MEMO_CAPACITY {
            let m = fixture_usize_as_f64(i);
            memo.insert(m, m + 0.5);
        }

        assert_eq!(memo.get(0.0), Some(0.5));
        assert_eq!(
            memo.get(fixture_usize_as_f64(EXACT_MASS_MEMO_CAPACITY - 1)),
            Some(fixture_usize_as_f64(EXACT_MASS_MEMO_CAPACITY) - 0.5)
        );

        memo.insert(10_000.0, 1.0);
        assert_eq!(memo.get(0.0), None, "first slot should be evicted first");
        assert_eq!(memo.get(1.0), Some(1.5));

        memo.insert(20_000.0, 2.0);
        assert_eq!(memo.get(1.0), None, "second slot should be evicted second");
        assert_eq!(memo.get(2.0), Some(2.5));
    }

    #[test]
    fn test_exact_mass_memo_hash_probing_handles_collisions() {
        let mut memo = ExactMassMissMemo::default();
        let mut colliding_masses: Vec<f64> = Vec::new();
        let mut target_bucket: Option<usize> = None;

        for i in 0..20_000usize {
            let mass = fixture_usize_as_f64(i) + 0.25;
            let bucket = exact_mass_memo_hash_index(mass.to_bits());
            if target_bucket.is_none() {
                target_bucket = Some(bucket);
            }
            if Some(bucket) == target_bucket {
                colliding_masses.push(mass);
                if colliding_masses.len() == 4 {
                    break;
                }
            }
        }

        assert_eq!(
            colliding_masses.len(),
            4,
            "expected enough values to share one hash bucket"
        );

        for (i, &m) in colliding_masses.iter().enumerate() {
            memo.insert(m, fixture_usize_as_f64(i) + 10.0);
        }
        for (i, &m) in colliding_masses.iter().enumerate() {
            assert_eq!(memo.get(m), Some(fixture_usize_as_f64(i) + 10.0));
        }
    }

    #[test]
    fn test_exact_mass_memo_update_does_not_reset_fifo_age() {
        let mut memo = ExactMassMissMemo::default();
        for i in 0..EXACT_MASS_MEMO_CAPACITY {
            let m = fixture_usize_as_f64(i);
            memo.insert(m, m + 1.0);
        }

        // Updating an existing key should not refresh insertion age.
        memo.insert(0.0, 123.0);
        memo.insert(99_999.0, 7.0);

        assert_eq!(memo.get(0.0), None, "oldest key should still be evicted");
        assert_eq!(memo.get(1.0), Some(2.0));
        assert_eq!(memo.get(99_999.0), Some(7.0));
    }

    #[test]
    fn test_hf_batch_dispatch_parallel_threshold_is_strict() {
        assert_eq!(
            hf_batch_dispatch_parallel_threshold(HF_BATCH_PAR_THRESHOLD_DEFAULT),
            HF_BATCH_PAR_THRESHOLD_DEFAULT + 1
        );
        assert_eq!(hf_batch_dispatch_parallel_threshold(8), 9);
    }

    #[test]
    fn test_hf_batch_dispatch_parallel_threshold_saturates() {
        assert_eq!(hf_batch_dispatch_parallel_threshold(usize::MAX), usize::MAX);
    }

    #[test]
    fn test_mf_j2_batch_dispatch_uses_large_default_serial_cutoff() {
        // Pre-epsilon this was a strict `>`: MF-J2 events are cheaper than HF
        // ones, so the MF gate could afford to fan out later. The epsilon
        // reseal pushed MF rows per batch below that headroom and the two
        // cutoffs have now converged, so the surviving claim is that MF never
        // fans out EARLIER than HF.
        assert!(
            std::hint::black_box(MF_J2_BATCH_PAR_THRESHOLD_DEFAULT)
                >= HF_BATCH_PAR_THRESHOLD_DEFAULT,
            "MF-J2 must not fan out earlier than the HF threshold",
        );
        assert!(
            !should_use_mf_j2_batch_parallel_dispatch(32),
            "32-event MF-J2 batches should stay serial by default",
        );
        assert_eq!(
            hf_batch_dispatch_parallel_threshold(HF_BATCH_PAR_THRESHOLD_DEFAULT),
            HF_BATCH_PAR_THRESHOLD_DEFAULT + 1,
            "HF threshold helper behavior should remain unchanged",
        );
    }

    /// The other half of the cutoff: retain the threshold selected from the
    /// historical row sample. Between the 1024 default and 2026-08-07 the gate
    /// fired 0 of 384 times on a Part A MF population while Stage 3 mass ran
    /// single-threaded.
    ///
    /// Pinned against the threshold helper rather than
    /// `should_use_mf_j2_batch_parallel_dispatch`, because that also consults
    /// `rayon::current_num_threads()` and would make this assertion depend on
    /// the host's core count.
    ///
    /// Pinned against the p10 and NOT against the minimum. A distribution's
    /// minimum is an extreme order statistic: it drifts down as the sample
    /// grows, so pinning it makes this test re-red on every larger run without
    /// anything having regressed. The commit-recorded 2026-08-13 aggregates and
    /// their evidence limits are in `docs/PART_A_RESULTS_MATRIX.md`, under
    /// "Historical MF-J2 batch-row distribution — commit-recorded summary".
    /// Both captured run shapes had p10 = 34; that threshold still trips
    /// instantly on a 256-class regression. This test does not make the sample
    /// current authority.
    #[test]
    fn test_mf_j2_cutoff_does_not_exceed_the_historical_recorded_p10() {
        // 2026-08-07, measured over the 384 batches of a p16/e24 Part A MF
        // population: min 366, p50 520, p90 696, max 864.
        //
        // 2026-08-13, measured post-epsilon (not inferred). Aggregate provenance,
        // evidence limits, and both aggregate distributions are recorded in
        // `docs/PART_A_RESULTS_MATRIX.md`; the anchor-inclusive p10 was 34 rows.
        const HISTORICAL_RECORDED_P10_ROWS: usize = 34;

        let cutoff = mf_j2_batch_dispatch_parallel_threshold(MF_J2_BATCH_PAR_THRESHOLD);
        assert!(
            cutoff <= HISTORICAL_RECORDED_P10_ROWS,
            "MF-J2 parallel cutoff {cutoff} exceeds the historical recorded p10 \
             ({HISTORICAL_RECORDED_P10_ROWS} rows); revisit the threshold provenance",
        );
    }

    #[test]
    fn test_mf_j2_batch_builder_stays_serial_at_default_32() {
        let event = sample_mf_j2_batch_event(0.0);
        let config = SolverConfig {
            xtol: 1e-6,
            rtol: 1e-8,
            maxiter: 4,
            mass_max: 1.0e3,
        };
        let n_events = 32_usize;
        let mut masses = vec![f64::NAN; n_events];
        let mut statuses = vec![MfJ2MassSolveStatusCode::MissAtZeroNonFinite; n_events];
        let mut miss_zero = vec![f64::NAN; n_events];
        let mut miss_root = vec![f64::NAN; n_events];
        let mut miss_upper = vec![f64::NAN; n_events];
        let mut iterations = vec![usize::MAX; n_events];
        let caller_thread_id = std::thread::current().id();
        let worker_thread_seen = std::sync::atomic::AtomicBool::new(false);

        solve_batch_events_mf_j2_with_status_into_from_builder(
            n_events,
            &config,
            |_| {
                if std::thread::current().id() != caller_thread_id {
                    worker_thread_seen.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                event
            },
            &mut masses,
            &mut statuses,
            &mut miss_zero,
            &mut miss_root,
            &mut miss_upper,
            &mut iterations,
        );

        assert_eq!(masses.len(), n_events);
        assert!(
            !worker_thread_seen.load(std::sync::atomic::Ordering::SeqCst),
            "32-event MF-J2 direct-fill batches should stay serial by default",
        );
    }

    #[test]
    fn test_prepare_hf_for_event_threads_qm_ratio_and_radius_with_packed_assets() {
        let c_coeffs = vec![0.0];
        let s_coeffs = vec![0.0];
        let packed = Arc::new(
            satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, 1, 0)
                .expect("test gravity coefficients are valid"),
        );
        let force_config = ForceConfig {
            qm_ratio: 0.001,
            r_obj_m: 0.05,
            ..ForceConfig::default()
        };
        let hf_ctx = HfContext {
            use_high_fidelity: true,
            epoch_jd: 2_460_000.5,
            force_config: Some(std::sync::Arc::new(force_config)),
            packed_coeffs: Some(Arc::clone(&packed)),
            hf_validate_only: false,
            hf_strict: true,
        };
        let event = MassSolverEvent {
            p_momentum: [1000.0, 0.0, 0.0],
            dv_vec: [10.0, 0.0, 0.0],
            p_mass: 100.0,
            p_pos_intercept: [7000.0, 0.0, 0.0],
            tof_s: 120.0,
            secondary_conj_pos: [7010.0, 0.0, 0.0],
            min_miss_distance_km: 5.0,
            kappa: 2.0,
            p_pos_conj_truth: [7000.0, 0.0, 0.0],
            p_pos_conj_equ_0: [7000.0, 0.0, 0.0],
            p_velocity: [10.0, 0.0, 0.0],
            v_rel: [0.0, 0.0, 0.0],
            p_equ_intercept: [0.0; 6],
            p_am_ratio: None,
            p_cd: None,
            p_cr: None,
            p_qm_ratio: Some(0.007),
            p_r_obj_m: Some(0.42),
        };

        let prepared = prepare_hf_for_event(&event, &hf_ctx)
            .expect("hf context should be complete")
            .expect("hf context should prepare without ephemeris requirements");

        assert!((prepared.force_config.qm_ratio - 0.007).abs() < 1e-15);
        assert!((prepared.force_config.r_obj_m - 0.42).abs() < 1e-15);
        assert!(Arc::ptr_eq(&prepared.packed_coeffs, &packed));
    }

    #[test]
    fn test_prepare_hf_for_event_threads_all_force_overrides() {
        let c_coeffs = vec![0.0];
        let s_coeffs = vec![0.0];
        let packed = satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, 1, 0)
            .expect("test gravity coefficients are valid");
        let force_config = ForceConfig {
            am_ratio: 0.001,
            cd: 2.0,
            cr: 1.0,
            qm_ratio: 0.0,
            r_obj_m: 0.0,
            ..ForceConfig::default()
        };
        let hf_ctx = HfContext {
            use_high_fidelity: true,
            epoch_jd: 2_460_000.5,
            force_config: Some(std::sync::Arc::new(force_config)),
            packed_coeffs: Some(std::sync::Arc::new(packed)),
            hf_validate_only: false,
            hf_strict: true,
        };
        let event = MassSolverEvent {
            p_momentum: [1000.0, 0.0, 0.0],
            dv_vec: [10.0, 0.0, 0.0],
            p_mass: 100.0,
            p_pos_intercept: [7000.0, 0.0, 0.0],
            tof_s: 120.0,
            secondary_conj_pos: [7010.0, 0.0, 0.0],
            min_miss_distance_km: 5.0,
            kappa: 2.0,
            p_pos_conj_truth: [7000.0, 0.0, 0.0],
            p_pos_conj_equ_0: [7000.0, 0.0, 0.0],
            p_velocity: [10.0, 0.0, 0.0],
            v_rel: [0.0, 0.0, 0.0],
            p_equ_intercept: [0.0; 6],
            p_am_ratio: Some(0.015),
            p_cd: Some(2.35),
            p_cr: Some(1.25),
            p_qm_ratio: Some(0.006),
            p_r_obj_m: Some(0.37),
        };

        let prepared = prepare_hf_for_event(&event, &hf_ctx)
            .expect("hf context should be complete")
            .expect("hf context should prepare without ephemeris requirements");

        assert!((prepared.force_config.am_ratio - 0.015).abs() < 1e-15);
        assert!((prepared.force_config.cd - 2.35).abs() < 1e-15);
        assert!((prepared.force_config.cr - 1.25).abs() < 1e-15);
        assert!((prepared.force_config.qm_ratio - 0.006).abs() < 1e-15);
        assert!((prepared.force_config.r_obj_m - 0.37).abs() < 1e-15);
    }

    #[test]
    fn perfectly_inelastic_fixed_area_retention_scales_am_with_postimpact_mass() {
        let mut event = valid_zero_mass_test_event();
        event.p_mass = 10.0;
        let base = Arc::new(ForceConfig {
            am_ratio: 1.0,
            cd: 2.35,
            cr: 1.25,
            qm_ratio: 0.0,
            r_obj_m: 0.37,
            ..ForceConfig::default()
        });
        let prepared = PreparedHfConfig {
            force_config: Arc::clone(&base),
            epoch_jd: 2_460_000.5,
            packed_coeffs: Arc::new(
                satpy_core::pack_gravity_coeffs(&[0.0], &[0.0], 1, 0)
                    .expect("test gravity coefficients are valid"),
            ),
        };

        for (retained_mass_kg, expected_am_ratio) in
            [(0.0_f64, 1.0_f64), (10.0, 0.5), (100.0, 1.0 / 11.0)]
        {
            let retained = prepared
                .for_retained_mass(retained_mass_kg, &event)
                .expect("analytic retained mass is physical");
            assert_eq!(
                retained.force_config.am_ratio.to_bits(),
                expected_am_ratio.to_bits()
            );
            assert_eq!(retained.force_config.cd.to_bits(), base.cd.to_bits());
            assert_eq!(retained.force_config.cr.to_bits(), base.cr.to_bits());
            assert_eq!(retained.force_config.qm_ratio.to_bits(), 0.0_f64.to_bits());
            assert_eq!(
                retained.force_config.r_obj_m.to_bits(),
                base.r_obj_m.to_bits()
            );
        }
    }

    #[test]
    fn prepare_hf_rejects_ephemeris_arc_one_ulp_outside_before_rhs() {
        let flags = lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY;
        let ephem = lightyear_odeint_rs::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("sun ephemeris must load");
        let (_, end) = ephem
            .get(lightyear_odeint_rs::precomputed_ephem::Body::Sun)
            .expect("sun catalogue")
            .jd_range();
        let c_coeffs = vec![0.0];
        let s_coeffs = vec![0.0];
        let packed = satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, 1, 0)
            .expect("test gravity coefficients are valid");
        let hf_ctx = HfContext {
            use_high_fidelity: true,
            epoch_jd: next_fixture_f64(end),
            force_config: Some(Arc::new(ForceConfig {
                force_flags: flags,
                ..ForceConfig::default()
            })),
            packed_coeffs: Some(Arc::new(packed)),
            hf_validate_only: false,
            hf_strict: false,
        };
        let mut event = sample_batch_event(1.0, 1.0);
        event.tof_s = 0.0;

        assert!(matches!(
            prepare_hf_for_event(&event, &hf_ctx),
            Some(Err(()))
        ));
    }

    #[test]
    fn test_compute_new_velocity() {
        let p_momentum = [1000.0, 0.0, 0.0]; // 1000 kg·km/s
        let dv_vec = [10.0, 0.0, 0.0]; // 10 km/s dust
        let p_mass = 100.0; // 100 kg primary
        let mass = 10.0; // 10 kg dust
        let kappa = 2.0;
        // Precompute velocity and relative velocity
        let p_velocity = [
            p_momentum[0] / p_mass,
            p_momentum[1] / p_mass,
            p_momentum[2] / p_mass,
        ];
        let v_rel = [
            dv_vec[0] - p_velocity[0],
            dv_vec[1] - p_velocity[1],
            dv_vec[2] - p_velocity[2],
        ];

        // Create minimal event for testing
        // Bug #8 note: p_pos_conj_truth == p_pos_conj_equ_0 means baseline correction
        // is zero. This is acceptable for testing compute_new_velocity() which doesn't
        // use baseline correction, but production tests should use different values.
        let event = MassSolverEvent {
            p_momentum,
            dv_vec,
            p_mass,
            p_pos_intercept: [7000.0, 0.0, 0.0],
            tof_s: 3600.0,
            secondary_conj_pos: [7010.0, 0.0, 0.0],
            min_miss_distance_km: 5.0,
            kappa,
            p_pos_conj_truth: [7000.0, 0.0, 0.0],
            p_pos_conj_equ_0: [7000.0, 0.0, 0.0],
            p_velocity,
            v_rel,
            p_equ_intercept: [0.0; 6], // Not used in LF mode
            p_am_ratio: None,
            p_cd: None,
            p_cr: None,
            p_qm_ratio: None,
            p_r_obj_m: None,
        };

        let new_vel = compute_new_velocity(mass, &event);

        // Expected: p_vel = [10, 0, 0], v_rel = [0, 0, 0], so no change
        assert!((new_vel[0] - 10.0).abs() < 1e-6);
        assert!(new_vel[1].abs() < 1e-6);
        assert!(new_vel[2].abs() < 1e-6);
    }

    #[test]
    fn detmass_orbit_gate_rejects_endpoint_closed_earth_crossing_state() {
        let radius = 7000.0;
        let circular = [radius, 0.0, 0.0, 0.0, (MU / radius).sqrt(), 0.0];
        assert!(state_clears_min_radius(&circular, 6478.137));

        let crossing = [radius, 0.0, 0.0, 0.0, 7.080_443_746_84, 0.0];
        assert!(!state_clears_min_radius(&crossing, 6478.137));
        assert!(osculating_perigee_km(&crossing).unwrap() < 5505.0);
    }

    #[test]
    fn ordinary_hf_solver_types_endpoint_closed_ground_crossing_as_physical_infeasible() {
        let radius_km = 7_000.0;
        let period_s = 4_920.0;
        let semi_major_km = (MU * (period_s / (2.0 * std::f64::consts::PI)).powi(2)).cbrt();
        let speed_km_s = (MU * (2.0 / radius_km - 1.0 / semi_major_km)).sqrt();
        let state = [radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0];
        assert!(!state_clears_min_radius(&state, 6_478.137));

        let mut equ = [0.0; 6];
        eci2equinoc_impl(&state, 6, 0.0, 0.0, &mut equ);
        let primary_mass_kg = 500.0;
        let event = MassSolverEvent {
            p_momentum: [0.0, primary_mass_kg * speed_km_s, 0.0],
            dv_vec: [0.0, speed_km_s, 0.0],
            p_mass: primary_mass_kg,
            p_pos_intercept: [radius_km, 0.0, 0.0],
            tof_s: period_s,
            secondary_conj_pos: [radius_km + 100.0, 0.0, 0.0],
            min_miss_distance_km: 1.0,
            kappa: 1.1,
            p_pos_conj_truth: [radius_km, 0.0, 0.0],
            p_pos_conj_equ_0: [radius_km, 0.0, 0.0],
            p_velocity: [0.0, speed_km_s, 0.0],
            v_rel: [0.0; 3],
            p_equ_intercept: equ,
            p_am_ratio: None,
            p_cd: None,
            p_cr: None,
            p_qm_ratio: None,
            p_r_obj_m: None,
        };
        let config = SolverConfig {
            xtol: 1.0e-6,
            rtol: 1.0e-6,
            maxiter: 8,
            mass_max: 1_000.0,
        };
        let c_coeffs = vec![0.0];
        let s_coeffs = vec![0.0];
        let packed = satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, 1, 0)
            .expect("test gravity coefficients are valid");
        let hf_ctx = HfContext {
            use_high_fidelity: true,
            epoch_jd: 2_460_310.5,
            force_config: Some(Arc::new(ForceConfig {
                sph_order: 0,
                force_flags: 0,
                subtract_first_order: false,
                eps: 1.0e-9,
                dt_max: 60.0,
                ..ForceConfig::default()
            })),
            packed_coeffs: Some(Arc::new(packed)),
            hf_validate_only: false,
            hf_strict: false,
        };

        let (mass, status) = solve_single_event_hf_with_status(&event, &config, Some(&hf_ctx));

        assert!(
            mass.is_nan(),
            "physical infeasibility must not expose a mass"
        );
        assert_eq!(status.as_str(), "hf_trajectory_physically_infeasible");

        let mut validate_only_ctx = hf_ctx;
        validate_only_ctx.hf_validate_only = true;
        let (validate_mass, validate_status) =
            solve_single_event_hf_with_status(&event, &config, Some(&validate_only_ctx));
        assert!(
            validate_mass.is_nan(),
            "validate-only physical infeasibility must not expose a mass"
        );
        assert_eq!(
            validate_status.as_str(),
            "hf_trajectory_physically_infeasible"
        );
    }

    fn sample_batch_event(miss_factor: f64, tof_hours: f64) -> MassSolverEvent {
        let r = 6378.137 + 500.0;
        let v = (MU / r).sqrt();
        let p_mass = 500.0;
        let p_momentum = [0.0, p_mass * v, 0.0];
        let p_velocity = [0.0, v, 0.0];
        let dv_vec = [0.5, v, 0.0];
        let v_rel = [
            dv_vec[0] - p_velocity[0],
            dv_vec[1] - p_velocity[1],
            dv_vec[2] - p_velocity[2],
        ];
        let tof_s = tof_hours * 3600.0;
        let omega = v / r;
        let angle = omega * tof_s;
        let prop_x = r * angle.cos();
        let prop_y = r * angle.sin();

        MassSolverEvent {
            p_momentum,
            dv_vec,
            p_mass,
            p_pos_intercept: [r, 0.0, 0.0],
            tof_s,
            secondary_conj_pos: [prop_x + miss_factor, prop_y, 0.0],
            min_miss_distance_km: 5.0,
            kappa: 1.10,
            p_pos_conj_truth: [prop_x, prop_y, 0.0],
            p_pos_conj_equ_0: [prop_x, prop_y, 0.0],
            p_velocity,
            v_rel,
            p_equ_intercept: [0.0; 6],
            p_am_ratio: None,
            p_cd: None,
            p_cr: None,
            p_qm_ratio: None,
            p_r_obj_m: None,
        }
    }

    #[test]
    fn deterministic_solver_accepts_future_subunity_kappa() {
        let mut event = sample_batch_event(1.0, 0.1);
        event.kappa = 0.65;
        let config = SolverConfig {
            xtol: 1.0e-6,
            rtol: 1.0e-5,
            maxiter: 8,
            mass_max: 1_000.0,
        };
        let (mass, status) = solve_single_event_hf_with_status(&event, &config, None);
        // "Accepts" must be observable: the sub-unity kappa event converges to
        // a real deflection mass instead of being refused. Measured 2026-08-20
        // at ~25.67 kg; the window is deliberately coarse so only a status
        // regression or a gross solver change trips it, not libm ulps.
        assert_eq!(status, MassSolveStatusCode::Converged);
        assert!(
            mass.is_finite() && (1.0..=100.0).contains(&mass),
            "sub-unity kappa solve drifted out of its coarse window: {mass}"
        );
    }

    fn sample_mf_j2_batch_event(offset: f64) -> MfJ2MassSolverEvent {
        let r = 7078.0 + offset;
        let v = (MU / r).sqrt();
        let target_pos = [r, 0.0, 0.0];
        let target_vel = [0.0, v, 0.0];
        let tof_s = 120.0 + offset * 0.1;
        let state0 = [
            target_pos[0],
            target_pos[1],
            target_pos[2],
            target_vel[0],
            target_vel[1],
            target_vel[2],
        ];
        let mut equ = [0.0_f64; 6];
        eci2equinoc_impl(&state0, 6, 0.0, 0.0, &mut equ);
        let mut natural = [0.0_f64; 6];
        equinoc_prop_j2_from_impl(&equ, tof_s, &mut natural);
        MfJ2MassSolverEvent::new(
            target_pos,
            target_vel,
            [0.05, v, 0.08],
            500.0 + offset,
            [natural[0], natural[1], natural[2]],
            tof_s,
            0.5,
            1.0,
        )
    }

    #[test]
    fn mf_j2_nonpositive_or_nonfinite_rtol_disables_residual_convergence() {
        let event = sample_mf_j2_batch_event(0.0);

        for rtol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let result = solve_single_event_mf_j2_with_status(
                &event,
                &SolverConfig {
                    xtol: 0.0,
                    rtol,
                    maxiter: 1,
                    mass_max: 1.0e3,
                },
            );
            assert_eq!(
                result.status,
                MfJ2MassSolveStatusCode::MaxIterReached,
                "rtol={rtol:?} must not activate residual convergence"
            );
            assert!(
                result.root_mass_kg.is_nan(),
                "rtol={rtol:?} nonconvergence must remain infeasible"
            );
        }
    }

    #[test]
    fn test_zero_mass_returns_zero() {
        // Circular orbit at 7000 km with velocity ~7.55 km/s
        let r = 7000.0_f64;
        let v = (MU / r).sqrt(); // ~7.546 km/s for circular orbit

        let p_momentum = [0.0, 100.0 * v, 0.0]; // 100 kg satellite, circular orbit
        let p_mass = 100.0;
        let dv_vec = [0.0, v, 0.0]; // Dust moving with satellite (no delta-v)
                                    // Precompute velocity and relative velocity
        let p_velocity = [
            p_momentum[0] / p_mass,
            p_momentum[1] / p_mass,
            p_momentum[2] / p_mass,
        ];
        let v_rel = [
            dv_vec[0] - p_velocity[0],
            dv_vec[1] - p_velocity[1],
            dv_vec[2] - p_velocity[2],
        ];

        // Bug #8 note: identical baselines test edge case behavior, not realistic scenarios
        let event = MassSolverEvent {
            p_momentum,
            dv_vec,
            p_mass,
            p_pos_intercept: [r, 0.0, 0.0],
            tof_s: 3600.0,
            secondary_conj_pos: [r + 10.0, 0.0, 0.0], // Secondary 10 km away
            min_miss_distance_km: 5.0,
            kappa: 2.0,
            p_pos_conj_truth: [r, 0.0, 0.0],
            p_pos_conj_equ_0: [r, 0.0, 0.0],
            p_velocity,
            v_rel,
            p_equ_intercept: [0.0; 6], // Not used in LF mode
            p_am_ratio: None,
            p_cd: None,
            p_cr: None,
            p_qm_ratio: None,
            p_r_obj_m: None,
        };

        let config = SolverConfig::default();
        let mass = solve_single_event_hf(&event, &config, None);

        // The name of this test is the claim: dust already moving with the
        // satellite needs no mass, so the answer is EXACTLY +0.0. Bits, not
        // `== 0.0`, because `-0.0 == 0.0` and the sign of a returned zero is
        // observable downstream.
        //
        // The predicate this replaced was `mass >= 0.0 || mass.is_nan()`
        // guarding a `mass <= mass_max` check that only ran `if
        // mass.is_finite()`. Measured here, `mass` is +0.0 and the `is_nan()`
        // arm never fires, so the pair admitted every non-negative answer up
        // to 1e6 and admitted NaN by skipping the bound entirely. Returning
        // `mass + 1.0` from `solve_single_event_hf` left it green.
        assert_eq!(
            mass.to_bits(),
            0.0_f64.to_bits(),
            "zero-relative-velocity event must need exactly +0.0 kg, got {mass:?}"
        );
        // Unconditional: a NaN must fail this, not skip it.
        assert!(
            mass <= config.mass_max,
            "mass {mass:?} above cap {}",
            config.mass_max
        );
    }

    #[test]
    fn validate_only_hf_converged_status_matches_full_hf_minimum() {
        let config = SolverConfig {
            xtol: 1e-6,
            rtol: 1e-6,
            maxiter: 80,
            mass_max: 1_000.0,
        };
        let mut full_context = zero_mass_test_context();
        full_context.hf_validate_only = false;
        let mut validate_context = full_context.clone();
        validate_context.hf_validate_only = true;

        for offset_km in [0.25, 0.5, 1.0, 2.0, 4.0] {
            let mut event = valid_zero_mass_test_event();
            event.secondary_conj_pos[0] = event.p_pos_conj_truth[0] + offset_km;
            event.secondary_conj_pos[1] = event.p_pos_conj_truth[1];
            event.secondary_conj_pos[2] = event.p_pos_conj_truth[2];
            event.min_miss_distance_km = 5.0;

            let (full_mass, full_status) =
                solve_single_event_hf_with_status(&event, &config, Some(&full_context));
            let (validate_mass, validate_status) =
                solve_single_event_hf_with_status(&event, &config, Some(&validate_context));
            let validate_profile = last_hf_mass_solve_profile();
            assert_eq!(
                full_status,
                MassSolveStatusCode::Converged,
                "full-HF fixture offset={offset_km} must have a finite root"
            );
            assert_eq!(
                validate_status,
                MassSolveStatusCode::Converged,
                "validate-only fixture offset={offset_km} must retain a finite root"
            );
            assert!(
                (validate_mass - full_mass).abs() <= config.xtol,
                "validate-only returned nonminimal HF mass at offset={offset_km}: \\
                 validate={validate_mass:.12e}, full={full_mass:.12e}"
            );
            assert_eq!(validate_profile.hf_validate_initial_calls, 1);
            // The zero arm of the fallback counter. Paired with the nonzero
            // assertion in
            // `test_validate_only_hf_runs_full_hf_solve_when_lf_seed_is_invalid`:
            // either alone would still hold if the counter were stuck, and
            // together they pin that it MOVES with the thing it names. That
            // pairing is what the field lacked while it was structurally zero.
            // Poison-proved by returning `run_full_hf` from the `Converged` arm
            // of the LF-seed match, which reds this line.
            assert_eq!(
                validate_profile.hf_lf_fallback_count, 0,
                "a converging validate-only solve must record no full-HF fallback"
            );
            assert_eq!(
                validate_profile.detmass_anchor_contract_version,
                DETMASS_ANCHOR_CONTRACT_VERSION
            );
            assert!(validate_profile.detmass_anchor_internal_reference_used);
        }
    }

    #[test]
    fn test_validate_only_hf_checks_zero_mass_when_lf_seed_is_nonpositive() {
        let event = MassSolverEvent {
            p_momentum: [
                2.636_913_030_216_084,
                -0.750_535_028_213_936_9,
                1.839_850_235_022_059_1,
            ],
            dv_vec: [
                -4.095_749_831_000_614,
                0.728_176_993_479_923_8,
                -6.163_159_077_384_986,
            ],
            p_mass: 0.444_741_285_460_396_76,
            p_pos_intercept: [
                4_149.257_322_330_506,
                219.287_005_970_050_02,
                -5_894.655_269_762_477_5,
            ],
            tof_s: 142_119.182_708_859_44,
            secondary_conj_pos: [
                589.271_650_932_486_2,
                -1_200.595_882_801_548_7,
                7_037.562_143_670_847,
            ],
            min_miss_distance_km: 1.0,
            kappa: 1.1,
            p_pos_conj_truth: [
                589.268_623_921_394_7,
                -1_200.595_490_979_283_3,
                7_037.562_456_240_712,
            ],
            p_pos_conj_equ_0: [
                7_191.869_866_505_112,
                0.003_728_474_887_406_881_8,
                0.001_468_648_461_673_225,
            ],
            p_velocity: [
                5.929_094_321_626_445,
                -1.687_576_694_025_566,
                4.136_900_025_185_302_5,
            ],
            v_rel: [
                -10.024_844_152_627_06,
                2.415_753_687_505_49,
                -10.300_059_102_570_29,
            ],
            p_equ_intercept: [
                7_191.869_866_505_112,
                0.003_728_474_887_406_881_8,
                0.001_468_648_461_673_225,
                -0.199_840_649_895_826_67,
                1.152_003_185_734_952_2,
                5.142_792_456_469_9,
            ],
            p_am_ratio: Some(0.006_327_363_053_638_112),
            p_cd: Some(2.2),
            p_cr: Some(1.3),
            p_qm_ratio: Some(0.0),
            p_r_obj_m: Some(0.029_928_859_280_060_473),
        };
        let config = SolverConfig {
            xtol: 1e-6,
            rtol: 1e-6,
            maxiter: 80,
            mass_max: 1.0e6,
        };

        let lf_status = solve_single_event_hf_with_status(&event, &config, None).1;
        assert_eq!(lf_status, MassSolveStatusCode::ConvergedNonPositive);

        let c_coeffs = vec![0.0];
        let s_coeffs = vec![0.0];
        let packed = satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, 1, 0)
            .expect("test gravity coefficients are valid");
        let hf_ctx = HfContext {
            use_high_fidelity: true,
            epoch_jd: 2_460_000.5,
            force_config: Some(Arc::new(ForceConfig {
                sph_order: 0,
                force_flags: 0,
                subtract_first_order: false,
                eps: 1e-9,
                dt_max: 60.0,
                ..ForceConfig::default()
            })),
            packed_coeffs: Some(Arc::new(packed)),
            hf_validate_only: true,
            hf_strict: true,
        };

        let result = solve_single_event_hf_validate_only(
            &event,
            &config,
            &hf_ctx,
            None,
            &mut UnobservedMassSolve,
        );
        let profile = last_hf_mass_solve_profile();
        assert!(
            profile.hf_validate_initial_calls >= 1,
            "strict HF must validate LF nonpositive/safe seeds at mass=0"
        );
        assert_ne!(
            result.1,
            MassSolveStatusCode::HfValidateFallbackLfSeed,
            "strict HF must never return an LF fallback"
        );
    }

    #[test]
    fn test_validate_only_hf_runs_full_hf_solve_when_lf_seed_is_invalid() {
        let mut event = sample_batch_event(1.0, 1.0);
        event.min_miss_distance_km = 1.0e6;
        let eci_state = [
            event.p_pos_intercept[0],
            event.p_pos_intercept[1],
            event.p_pos_intercept[2],
            event.p_velocity[0],
            event.p_velocity[1],
            event.p_velocity[2],
        ];
        eci2equinoc_impl(&eci_state, 6, 0.0, 0.0, &mut event.p_equ_intercept);
        let mut propagated = [0.0; 6];
        equinoc2eci_impl(&event.p_equ_intercept, 6, event.tof_s, 0.0, &mut propagated);
        event.p_pos_conj_equ_0.copy_from_slice(&propagated[..3]);
        event.p_pos_conj_truth = event.p_pos_conj_equ_0;
        let config = SolverConfig {
            xtol: 1e-6,
            rtol: 1e-6,
            maxiter: 8,
            mass_max: 100.0,
        };

        let lf_status = solve_single_event_hf_with_status(&event, &config, None).1;
        assert!(
            !matches!(
                lf_status,
                MassSolveStatusCode::Converged
                    | MassSolveStatusCode::SafeByDefault
                    | MassSolveStatusCode::ConvergedNonPositive
            ),
            "fixture must produce an invalid LF seed, got {lf_status:?}"
        );

        let c_coeffs = vec![0.0];
        let s_coeffs = vec![0.0];
        let packed = satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, 1, 0)
            .expect("test gravity coefficients are valid");
        let hf_ctx = HfContext {
            use_high_fidelity: true,
            epoch_jd: 2_460_000.5,
            force_config: Some(Arc::new(ForceConfig {
                sph_order: 0,
                force_flags: 0,
                subtract_first_order: false,
                eps: 1e-9,
                dt_max: 60.0,
                ..ForceConfig::default()
            })),
            packed_coeffs: Some(Arc::new(packed)),
            hf_validate_only: true,
            hf_strict: true,
        };

        let result = solve_single_event_hf_validate_only(
            &event,
            &config,
            &hf_ctx,
            None,
            &mut UnobservedMassSolve,
        );
        let profile = last_hf_mass_solve_profile();
        assert_ne!(result.1, MassSolveStatusCode::HfValidateLfSeedInvalid);
        assert!(
            profile.hf_full_bracket_calls >= 1,
            "strict HF must run authoritative bracketing when LF cannot seed"
        );
        // The falsifiable half. This fixture forces exactly one fallback, so a
        // count of zero means the counter is dead again -- which is how it
        // shipped twice: first never incremented, then incremented above a
        // nested `hf_profile_reset()` that wiped it. Until this line existed the
        // only assertion on the field was `batch == serial`, which held at
        // `0 == 0` for both defects.
        assert_eq!(
            profile.hf_lf_fallback_count, 1,
            "the LF-to-full-HF fallback counter must survive the nested solve's profile reset"
        );
    }

    #[test]
    fn test_lf_miss_distance_derived_matches_legacy() {
        const MASS_SAMPLES: [f64; 12] = [
            0.0, 0.05, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 125.0, 250.0, 499.95,
        ];

        fn compute_miss_distance_lf_legacy(mass: f64, event: &MassSolverEvent) -> f64 {
            if !mass.is_finite() || mass < 0.0 {
                return f64::INFINITY;
            }
            let new_vel = compute_new_velocity(mass, event);
            if !new_vel[0].is_finite() || !new_vel[1].is_finite() || !new_vel[2].is_finite() {
                return f64::INFINITY;
            }
            let v_sq = vec3_norm_sq(&new_vel);
            let r_sq = vec3_norm_sq(&event.p_pos_intercept);
            let r = r_sq.sqrt();
            if r > 0.0 {
                let specific_energy = 0.5 * v_sq - MU / r;
                if specific_energy > 1e-6 {
                    return f64::INFINITY;
                }
            }
            let eci_state = [
                event.p_pos_intercept[0],
                event.p_pos_intercept[1],
                event.p_pos_intercept[2],
                new_vel[0],
                new_vel[1],
                new_vel[2],
            ];
            let mut equ = [0.0; 6];
            eci2equinoc_impl(&eci_state, 6, 0.0, 0.0, &mut equ);
            if !equ[0].is_finite() || equ[0] <= 0.0 {
                return f64::INFINITY;
            }
            let mut propagated = [0.0; 6];
            equinoc2eci_impl(&equ, 6, event.tof_s, 0.0, &mut propagated);
            if !propagated[0].is_finite() {
                return f64::INFINITY;
            }
            let mut new_pos = [propagated[0], propagated[1], propagated[2]];
            let d0 = event.p_pos_conj_truth[0] - event.p_pos_conj_equ_0[0];
            let d1 = event.p_pos_conj_truth[1] - event.p_pos_conj_equ_0[1];
            let d2 = event.p_pos_conj_truth[2] - event.p_pos_conj_equ_0[2];
            let shift_sq = d0.mul_add(d0, d1.mul_add(d1, d2 * d2));
            if shift_sq > 1e-12 {
                new_pos[0] = event.p_pos_conj_truth[0] + (new_pos[0] - event.p_pos_conj_equ_0[0]);
                new_pos[1] = event.p_pos_conj_truth[1] + (new_pos[1] - event.p_pos_conj_equ_0[1]);
                new_pos[2] = event.p_pos_conj_truth[2] + (new_pos[2] - event.p_pos_conj_equ_0[2]);
            }
            vec3_distance(&new_pos, &event.secondary_conj_pos)
        }

        let r = 7000.0_f64;
        let v = (MU / r).sqrt();
        let p_momentum = [0.0, 100.0 * v, 0.0];
        let p_mass = 100.0;
        let dv_vec = [0.5, v + 0.02, 0.01];
        let p_velocity = [
            p_momentum[0] / p_mass,
            p_momentum[1] / p_mass,
            p_momentum[2] / p_mass,
        ];
        let v_rel = [
            dv_vec[0] - p_velocity[0],
            dv_vec[1] - p_velocity[1],
            dv_vec[2] - p_velocity[2],
        ];
        let event = MassSolverEvent {
            p_momentum,
            dv_vec,
            p_mass,
            p_pos_intercept: [r, 12.0, -8.0],
            tof_s: 3600.0,
            secondary_conj_pos: [r + 10.0, -6.0, 3.0],
            min_miss_distance_km: 5.0,
            kappa: 2.0,
            p_pos_conj_truth: [r + 0.7, 0.4, -0.2],
            p_pos_conj_equ_0: [r - 0.3, -0.1, 0.1],
            p_velocity,
            v_rel,
            p_equ_intercept: [0.0; 6],
            p_am_ratio: None,
            p_cd: None,
            p_cr: None,
            p_qm_ratio: None,
            p_r_obj_m: None,
        };
        let derived = derive_event_invariants(&event, None);
        for mass in MASS_SAMPLES {
            let new = compute_miss_distance_lf(mass, &event, &derived);
            let legacy = compute_miss_distance_lf_legacy(mass, &event);
            if new.is_finite() || legacy.is_finite() {
                let rel = ((new - legacy).abs()) / legacy.abs().max(1.0);
                assert!(
                    rel < 1e-12,
                    "legacy/new drift too large for mass={mass}: rel={rel} new={new} legacy={legacy}"
                );
            } else {
                assert_eq!(new.is_infinite(), legacy.is_infinite());
            }
        }
    }

    #[test]
    fn test_hf_batch_with_profiles_matches_single_solves_exactly() {
        // Parity oracle for the per-row-trace batch path: masses, statuses,
        // and every profile counter must be identical to sequential single
        // solves, including above the parallel-dispatch threshold.
        let hf_ctx_for = |epoch_jd: f64| {
            let c_coeffs = vec![0.0];
            let s_coeffs = vec![0.0];
            let packed = satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, 1, 0)
                .expect("test gravity coefficients are valid");
            HfContext {
                use_high_fidelity: true,
                epoch_jd,
                force_config: Some(Arc::new(ForceConfig {
                    sph_order: 0,
                    force_flags: 0,
                    subtract_first_order: false,
                    eps: 1e-9,
                    dt_max: 60.0,
                    ..ForceConfig::default()
                })),
                packed_coeffs: Some(Arc::new(packed)),
                hf_validate_only: true,
                hf_strict: true,
            }
        };
        // Enough rows to clear HF_BATCH_PAR_THRESHOLD (32) so the rayon
        // branch is what gets verified.
        let n = 48usize;
        let events: Vec<MassSolverEvent> = (0..n)
            .map(|i| {
                let mut event = sample_batch_event(
                    0.5 + fixture_usize_as_f64(i) * 0.05,
                    0.5 + fixture_usize_as_f64(i) * 0.02,
                );
                let eci_state = [
                    event.p_pos_intercept[0],
                    event.p_pos_intercept[1],
                    event.p_pos_intercept[2],
                    event.p_velocity[0],
                    event.p_velocity[1],
                    event.p_velocity[2],
                ];
                eci2equinoc_impl(&eci_state, 6, 0.0, 0.0, &mut event.p_equ_intercept);
                let mut propagated = [0.0; 6];
                equinoc2eci_impl(&event.p_equ_intercept, 6, event.tof_s, 0.0, &mut propagated);
                event.p_pos_conj_equ_0.copy_from_slice(&propagated[..3]);
                event.p_pos_conj_truth = event.p_pos_conj_equ_0;
                event
            })
            .collect();
        let config = SolverConfig {
            xtol: 1e-6,
            rtol: 1e-6,
            maxiter: 8,
            mass_max: 100.0,
        };
        let epochs: Vec<f64> = (0..n)
            .map(|i| 2_460_000.5 + fixture_usize_as_f64(i) * 0.25)
            .collect();

        let mut single_masses = vec![0.0; n];
        let mut single_statuses = vec![MassSolveStatusCode::MissAtZeroNonFinite; n];
        let mut single_profiles = vec![HfMassSolveProfile::default(); n];
        for (((((event, &epoch), mass_out), status_out), profile_out), _row) in events
            .iter()
            .zip(&epochs)
            .zip(&mut single_masses)
            .zip(&mut single_statuses)
            .zip(&mut single_profiles)
            .zip(0..n)
        {
            let ctx = hf_ctx_for(epoch);
            let (mass, status) = solve_single_event_hf_with_status(event, &config, Some(&ctx));
            *mass_out = mass;
            *status_out = status;
            *profile_out = last_hf_mass_solve_profile();
        }

        let mut batch_masses = vec![0.0; n];
        let mut batch_statuses = vec![MassSolveStatusCode::MissAtZeroNonFinite; n];
        let mut batch_profiles = vec![HfMassSolveProfile::default(); n];
        let global_width = rayon::current_num_threads();
        solve_batch_events_hf_with_status_and_profiles_from_builder_global_rayon(
            n,
            &config,
            |i| {
                events
                    .get(i)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("batch event row {i} out of range"))
            },
            |i| {
                epochs
                    .get(i)
                    .copied()
                    .map(hf_ctx_for)
                    .ok_or_else(|| anyhow::anyhow!("batch epoch row {i} out of range"))
            },
            &mut batch_masses,
            &mut batch_statuses,
            &mut batch_profiles,
            global_width,
            global_width,
        )
        .expect("global HF deterministic-mass batch");

        for (i, (((batch_status, single_status), (batch_mass, single_mass)), (b, s))) in
            batch_statuses
                .iter()
                .zip(&single_statuses)
                .zip(batch_masses.iter().zip(&single_masses))
                .zip(batch_profiles.iter().zip(&single_profiles))
                .enumerate()
        {
            assert_eq!(batch_status, single_status, "row {i} status diverged");
            assert!(
                same_or_both_nan(*batch_mass, *single_mass),
                "row {i} mass diverged: batch={batch_mass} single={single_mass}"
            );
            assert_eq!(b.hf_miss_calls_total, s.hf_miss_calls_total, "row {i}");
            assert_eq!(
                b.hf_validate_initial_calls, s.hf_validate_initial_calls,
                "row {i}"
            );
            assert_eq!(
                b.hf_validate_repair_calls, s.hf_validate_repair_calls,
                "row {i}"
            );
            assert_eq!(
                b.hf_validate_refine_calls, s.hf_validate_refine_calls,
                "row {i}"
            );
            assert_eq!(
                b.hf_validate_refine_iterations, s.hf_validate_refine_iterations,
                "row {i}"
            );
            assert_eq!(b.hf_full_bracket_calls, s.hf_full_bracket_calls, "row {i}");
            assert_eq!(b.hf_full_refine_calls, s.hf_full_refine_calls, "row {i}");
            assert_eq!(
                b.hf_full_refine_iterations, s.hf_full_refine_iterations,
                "row {i}"
            );
            assert_eq!(
                b.hf_upper_bracket_shrink_iterations, s.hf_upper_bracket_shrink_iterations,
                "row {i}"
            );
            assert_eq!(b.hf_lf_fallback_count, s.hf_lf_fallback_count, "row {i}");
        }
    }

    #[test]
    fn hf_batch_uses_four_thread_global_pool_without_reordering() {
        if spawn_four_global_pool_child_if_parent() {
            return;
        }

        assert_eq!(
            rayon::current_num_threads(),
            4,
            "child process must configure four global Rayon workers"
        );
        let config = SolverConfig {
            xtol: 1e-6,
            rtol: 1e-6,
            maxiter: 1,
            mass_max: 1.0,
        };
        assert_global_hf_width_mismatch_is_rejected(&config);
        assert_four_global_rayon_workers();

        let hf_ctx_for = |epoch_jd: f64| {
            let c_coeffs = vec![0.0];
            let s_coeffs = vec![0.0];
            let packed = satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, 1, 0)
                .expect("test gravity coefficients are valid");
            HfContext {
                use_high_fidelity: true,
                epoch_jd,
                force_config: Some(Arc::new(ForceConfig {
                    sph_order: 0,
                    force_flags: 0,
                    subtract_first_order: false,
                    eps: 1e-9,
                    dt_max: 60.0,
                    ..ForceConfig::default()
                })),
                packed_coeffs: Some(Arc::new(packed)),
                hf_validate_only: true,
                hf_strict: true,
            }
        };
        let n = 48usize;
        let mut events: Vec<MassSolverEvent> = (0..n)
            .map(|i| {
                let mut event = sample_batch_event(
                    0.5 + fixture_usize_as_f64(i) * 0.05,
                    0.5 + fixture_usize_as_f64(i) * 0.02,
                );
                let eci_state = [
                    event.p_pos_intercept[0],
                    event.p_pos_intercept[1],
                    event.p_pos_intercept[2],
                    event.p_velocity[0],
                    event.p_velocity[1],
                    event.p_velocity[2],
                ];
                eci2equinoc_impl(&eci_state, 6, 0.0, 0.0, &mut event.p_equ_intercept);
                let mut propagated = [0.0; 6];
                equinoc2eci_impl(&event.p_equ_intercept, 6, event.tof_s, 0.0, &mut propagated);
                event.p_pos_conj_equ_0.copy_from_slice(&propagated[..3]);
                event.p_pos_conj_truth = event.p_pos_conj_equ_0;
                event
            })
            .collect();
        let last_event = events.last_mut();
        assert!(
            last_event.is_some(),
            "global-pool fixture must contain a final row"
        );
        let Some(last_event) = last_event else {
            return;
        };
        last_event.min_miss_distance_km = 1.0e9;
        let config = SolverConfig {
            xtol: 1e-6,
            rtol: 1e-6,
            maxiter: 8,
            mass_max: 100.0,
        };
        let epochs: Vec<f64> = (0..n)
            .map(|i| 2_460_000.5 + fixture_usize_as_f64(i) * 0.25)
            .collect();

        let mut scalar_masses = vec![0.0; n];
        let mut scalar_statuses = vec![MassSolveStatusCode::MissAtZeroNonFinite; n];
        let mut scalar_profiles = vec![HfMassSolveProfile::default(); n];
        for ((((event, &epoch), mass_out), status_out), profile_out) in events
            .iter()
            .zip(&epochs)
            .zip(&mut scalar_masses)
            .zip(&mut scalar_statuses)
            .zip(&mut scalar_profiles)
        {
            let context = hf_ctx_for(epoch);
            let (mass, status) = solve_single_event_hf_with_status(event, &config, Some(&context));
            *mass_out = mass;
            *status_out = status;
            *profile_out = last_hf_mass_solve_profile();
        }
        assert!(
            scalar_statuses
                .iter()
                .any(|status| *status != MassSolveStatusCode::Converged),
            "fixture must retain at least one typed non-converged row"
        );

        let observed_pool_width = std::sync::atomic::AtomicU64::new(0);
        let observed_event_worker_mask = std::sync::atomic::AtomicU64::new(0);
        let observed_context_worker_mask = std::sync::atomic::AtomicU64::new(0);
        let parallel_builder_ordinal = std::sync::atomic::AtomicUsize::new(0);
        let parallel_builder_barrier = std::sync::Barrier::new(4);
        let mut batch_masses = vec![0.0; n];
        let mut batch_statuses = vec![MassSolveStatusCode::MissAtZeroNonFinite; n];
        let mut batch_profiles = vec![HfMassSolveProfile::default(); n];
        solve_batch_events_hf_with_status_and_profiles_from_builder_global_rayon(
            n,
            &config,
            |i| {
                observed_pool_width.store(
                    u64::try_from(rayon::current_num_threads()).expect("Rayon width fits u64"),
                    std::sync::atomic::Ordering::Relaxed,
                );
                let worker = rayon::current_thread_index().expect("parallel event builder worker");
                observed_event_worker_mask
                    .fetch_or(1_u64 << worker, std::sync::atomic::Ordering::Relaxed);
                if parallel_builder_ordinal.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                    parallel_builder_barrier.wait();
                }
                events
                    .get(i)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("batch event row {i} out of range"))
            },
            |i| {
                let worker =
                    rayon::current_thread_index().expect("parallel context builder worker");
                observed_context_worker_mask
                    .fetch_or(1_u64 << worker, std::sync::atomic::Ordering::Relaxed);
                epochs
                    .get(i)
                    .copied()
                    .map(hf_ctx_for)
                    .ok_or_else(|| anyhow::anyhow!("batch epoch row {i} out of range"))
            },
            &mut batch_masses,
            &mut batch_statuses,
            &mut batch_profiles,
            4,
            4,
        )
        .expect("global HF deterministic-mass batch");

        assert_eq!(
            observed_pool_width.load(std::sync::atomic::Ordering::Relaxed),
            4
        );
        assert!(
            observed_event_worker_mask
                .load(std::sync::atomic::Ordering::Relaxed)
                .count_ones()
                > 1,
            "HF event builders must use multiple global workers"
        );
        assert!(
            observed_context_worker_mask
                .load(std::sync::atomic::Ordering::Relaxed)
                .count_ones()
                > 1,
            "HF context builders must use multiple global workers"
        );
        assert_eq!(
            batch_statuses, scalar_statuses,
            "typed statuses/order changed"
        );
        for (i, ((batch_mass, scalar_mass), (batch, scalar))) in batch_masses
            .iter()
            .zip(&scalar_masses)
            .zip(batch_profiles.iter().zip(&scalar_profiles))
            .enumerate()
        {
            assert!(
                same_or_both_nan(*batch_mass, *scalar_mass),
                "row {i} mass/order changed: batch={batch_mass} scalar={scalar_mass}"
            );
            assert_hf_profiles_equal(i, batch, scalar);
        }
    }

    #[test]
    fn hf_batch_rejects_invalid_budget_but_nested_invocation_is_serial_and_exact() {
        let invoke = |rayon_threads: usize, rayon_thread_budget: usize| {
            let config = SolverConfig {
                xtol: 1e-6,
                rtol: 1e-6,
                maxiter: 1,
                mass_max: 1.0,
            };
            let mut masses = Vec::new();
            let mut statuses = Vec::new();
            let mut profiles = Vec::new();
            solve_batch_events_hf_with_status_and_profiles_from_builder_global_rayon(
                0,
                &config,
                |_| panic!("invalid budget must fail before event construction"),
                |_| panic!("invalid budget must fail before context construction"),
                &mut masses,
                &mut statuses,
                &mut profiles,
                rayon_threads,
                rayon_thread_budget,
            )
        };

        assert!(invoke(0, 1)
            .expect_err("zero width must fail")
            .to_string()
            .contains("positive"));
        assert!(invoke(1, 0)
            .expect_err("zero budget must fail")
            .to_string()
            .contains("positive"));
        assert!(invoke(2, 1)
            .expect_err("width above budget must fail")
            .to_string()
            .contains("budget"));
        let available = satpy_core::parallel_budget::available_cores();
        assert!(invoke(1, available + 1)
            .expect_err("budget above available cores must fail")
            .to_string()
            .contains("available"));
        if available >= 2 {
            let n = hf_batch_dispatch_parallel_threshold(HF_BATCH_PAR_THRESHOLD);
            let event = valid_zero_mass_test_event();
            let context = zero_mass_test_context();
            let config = SolverConfig {
                xtol: 1e-3,
                rtol: 1e-6,
                maxiter: 4,
                mass_max: 10.0,
            };
            let run = |rayon_threads: usize,
                       worker_masks: Option<(
                &std::sync::atomic::AtomicU64,
                &std::sync::atomic::AtomicU64,
            )>,
                       call_counts: Option<(
                &std::sync::atomic::AtomicUsize,
                &std::sync::atomic::AtomicUsize,
            )>| {
                let mut masses = vec![0.0; n];
                let mut statuses = vec![MassSolveStatusCode::MissAtZeroNonFinite; n];
                let mut profiles = vec![HfMassSolveProfile::default(); n];
                solve_batch_events_hf_with_status_and_profiles_from_builder_global_rayon(
                    n,
                    &config,
                    |_| {
                        if let Some((event_worker_mask, _)) = worker_masks {
                            let worker = rayon::current_thread_index()
                                .expect("nested event builder must remain in outer pool");
                            event_worker_mask
                                .fetch_or(1_u64 << worker, std::sync::atomic::Ordering::Relaxed);
                        }
                        if let Some((event_calls, _)) = call_counts {
                            event_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        Ok(event.clone())
                    },
                    |_| {
                        if let Some((_, context_worker_mask)) = worker_masks {
                            let worker = rayon::current_thread_index()
                                .expect("nested context builder must remain in outer pool");
                            context_worker_mask
                                .fetch_or(1_u64 << worker, std::sync::atomic::Ordering::Relaxed);
                        }
                        if let Some((_, context_calls)) = call_counts {
                            context_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        Ok(context.clone())
                    },
                    &mut masses,
                    &mut statuses,
                    &mut profiles,
                    rayon_threads,
                    2,
                )?;
                anyhow::Ok((masses, statuses, profiles))
            };

            let serial = run(1, None, None).expect("direct serial batch");
            let nested_event_worker_mask = std::sync::atomic::AtomicU64::new(0);
            let nested_context_worker_mask = std::sync::atomic::AtomicU64::new(0);
            let nested_event_calls = std::sync::atomic::AtomicUsize::new(0);
            let nested_context_calls = std::sync::atomic::AtomicUsize::new(0);
            let outer = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("outer test pool");
            let (outer_worker, nested) = outer.install(|| {
                let outer_worker = rayon::current_thread_index().expect("invoking outer worker");
                let result = run(
                    2,
                    Some((&nested_event_worker_mask, &nested_context_worker_mask)),
                    Some((&nested_event_calls, &nested_context_calls)),
                );
                (outer_worker, result)
            });
            let nested = nested.expect("nested batch must fall through serial path");

            let outer_worker_mask = 1_u64 << outer_worker;
            assert_eq!(
                nested_event_calls.load(std::sync::atomic::Ordering::Relaxed),
                n,
                "every nested event builder call must be recorded"
            );
            assert_eq!(
                nested_context_calls.load(std::sync::atomic::Ordering::Relaxed),
                n,
                "every nested context builder call must be recorded"
            );
            assert_eq!(
                nested_event_worker_mask.load(std::sync::atomic::Ordering::Relaxed),
                outer_worker_mask,
                "nested event builders must remain on invoking outer worker"
            );
            assert_eq!(
                nested_context_worker_mask.load(std::sync::atomic::Ordering::Relaxed),
                outer_worker_mask,
                "nested context builders must remain on invoking outer worker"
            );

            assert_eq!(nested.1, serial.1, "nested status/order drift");
            for (row, ((nested_mass, serial_mass), (nested_profile, serial_profile))) in nested
                .0
                .iter()
                .zip(&serial.0)
                .zip(nested.2.iter().zip(&serial.2))
                .enumerate()
            {
                assert!(
                    same_or_both_nan(*nested_mass, *serial_mass),
                    "row {row} nested mass drift: nested={nested_mass} serial={serial_mass}"
                );
                assert_hf_profiles_equal(row, nested_profile, serial_profile);
            }
        }
    }

    /// The PARALLEL arm of the HF batch gate against the serial arm, bit for bit.
    ///
    /// # Why this is not already covered
    ///
    /// The neighbouring width test looks like it covers this and does not. Its
    /// "parallel" call runs inside `outer.install(...)`, which makes
    /// `rayon::current_thread_index()` return `Some`, so `use_parallel` is false
    /// and it takes the SERIAL branch — the test says so itself, in the
    /// `"nested batch must fall through serial path"` expectation. It compares
    /// serial against serial. Nothing in this workspace executes the
    /// `into_par_iter` arm.
    ///
    /// That arm is not a reschedule of the serial one: it builds rows through
    /// `into_par_iter`, then drives `par_iter_mut` over the outputs. Both arms
    /// construct the same `ZeroMassBatchCache` and call the same
    /// `solve_single_event_hf_internal`, so they are expected to agree — this
    /// pins that expectation rather than reporting a suspected bug.
    ///
    /// All three conditions of the gate have to hold at once for the parallel
    /// arm to run: `rayon_threads > 1`, `current_thread_index().is_none()` (so
    /// the call must be at top level, NOT inside a pool), and
    /// `n_events >= hf_batch_dispatch_parallel_threshold(..)`.
    #[test]
    fn hf_batch_parallel_arm_matches_serial_bit_for_bit() {
        let available = satpy_core::parallel_budget::available_cores();
        // The batch entry point requires the requested width to equal the live
        // global pool width, so the parallel arm can only be reached at exactly
        // that width -- passing a smaller "parallel" number is rejected, not
        // silently downgraded.
        let pool_width = rayon::current_num_threads();
        if available < 2 || pool_width < 2 {
            // Single-core host or single-thread pool: the parallel arm is
            // unreachable, and pretending otherwise would be a test that
            // reports success having run nothing.
            return;
        }
        let n = hf_batch_dispatch_parallel_threshold(HF_BATCH_PAR_THRESHOLD);
        let event = valid_zero_mass_test_event();
        let context = zero_mass_test_context();
        let config = SolverConfig {
            xtol: 1e-3,
            rtol: 1e-6,
            maxiter: 4,
            mass_max: 10.0,
        };

        // Builders observe which branch is executing: inside `into_par_iter`
        // they run on a pool worker, so `current_thread_index()` is `Some`. On
        // the serial arm they run on the caller thread and it is `None`. This
        // is a direct observation of the branch taken, not an inference from
        // the gate's inputs.
        let run = |rayon_threads: usize, in_pool: &std::sync::atomic::AtomicUsize| {
            let mut masses = vec![0.0; n];
            let mut statuses = vec![MassSolveStatusCode::MissAtZeroNonFinite; n];
            let mut profiles = vec![HfMassSolveProfile::default(); n];
            solve_batch_events_hf_with_status_and_profiles_from_builder_global_rayon(
                n,
                &config,
                |_| {
                    if rayon::current_thread_index().is_some() {
                        in_pool.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    Ok(event.clone())
                },
                |_| Ok(context.clone()),
                &mut masses,
                &mut statuses,
                &mut profiles,
                rayon_threads,
                rayon_threads.max(1),
            )?;
            anyhow::Ok((masses, statuses, profiles))
        };

        let serial_in_pool = std::sync::atomic::AtomicUsize::new(0);
        let serial = run(1, &serial_in_pool).expect("serial arm");
        let parallel_in_pool = std::sync::atomic::AtomicUsize::new(0);
        let parallel = run(pool_width, &parallel_in_pool).expect("parallel arm");

        // Non-vacuity, both directions. Without these the comparison below can
        // pass while both calls took the same branch.
        assert_eq!(
            serial_in_pool.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the rayon_threads=1 call must take the SERIAL arm, off-pool"
        );
        assert_eq!(
            parallel_in_pool.load(std::sync::atomic::Ordering::Relaxed),
            n,
            "the full-width call must take the PARALLEL arm, building every row \
             on a pool worker; if this is 0 the gate did not flip and this test \
             proves nothing"
        );

        assert_eq!(
            parallel.1, serial.1,
            "status vector must not depend on the branch"
        );
        for (row, ((parallel_mass, serial_mass), (parallel_profile, serial_profile))) in parallel
            .0
            .iter()
            .zip(&serial.0)
            .zip(parallel.2.iter().zip(&serial.2))
            .enumerate()
        {
            assert!(
                same_or_both_nan(*parallel_mass, *serial_mass),
                "row {row} mass differs between the parallel and serial arms: \
                 parallel={parallel_mass} serial={serial_mass}"
            );
            assert_hf_profiles_equal(row, parallel_profile, serial_profile);
        }
    }

    #[test]
    fn hf_batch_rejects_mismatched_output_lengths_without_building_rows() {
        let config = SolverConfig {
            xtol: 1e-6,
            rtol: 1e-6,
            maxiter: 1,
            mass_max: 1.0,
        };
        let mut masses = [];
        let mut statuses = [];
        let mut profiles = [];

        let error = solve_batch_events_hf_with_status_and_profiles_from_builder_global_rayon(
            1,
            &config,
            |_| panic!("length mismatch must reject before event construction"),
            |_| panic!("length mismatch must reject before context construction"),
            &mut masses,
            &mut statuses,
            &mut profiles,
            1,
            1,
        )
        .expect_err("mismatched outputs must return a typed error");

        assert!(error.to_string().contains("masses length"), "{error}");
    }

    fn zero_mass_test_context() -> HfContext {
        let c_coeffs = vec![0.0];
        let s_coeffs = vec![0.0];
        let packed_coeffs = Arc::new(
            satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, 1, 0)
                .expect("test gravity coefficients are valid"),
        );
        HfContext {
            use_high_fidelity: true,
            epoch_jd: 2_460_000.5,
            force_config: Some(Arc::new(ForceConfig {
                sph_order: 0,
                force_flags: 0,
                subtract_first_order: false,
                eps: 1e-9,
                dt_max: 60.0,
                ..ForceConfig::default()
            })),
            packed_coeffs: Some(packed_coeffs),
            hf_validate_only: false,
            hf_strict: true,
        }
    }

    fn valid_zero_mass_test_event() -> MassSolverEvent {
        let mut event = sample_batch_event(0.5, 0.05);
        let state = [
            event.p_pos_intercept[0],
            event.p_pos_intercept[1],
            event.p_pos_intercept[2],
            event.p_velocity[0],
            event.p_velocity[1],
            event.p_velocity[2],
        ];
        eci2equinoc_impl(&state, 6, 0.0, 0.0, &mut event.p_equ_intercept);
        let mut propagated = [0.0; 6];
        equinoc2eci_impl(&event.p_equ_intercept, 6, event.tof_s, 0.0, &mut propagated);
        event.p_pos_conj_equ_0.copy_from_slice(&propagated[..3]);
        event.p_pos_conj_truth = event.p_pos_conj_equ_0;
        event
    }

    fn fixed_impact_test_fixture() -> (MassSolverEvent, HfContext) {
        let mut event = valid_zero_mass_test_event();
        event.p_mass = 500.0;
        event.p_velocity = [0.0, (MU / event.p_pos_intercept[0]).sqrt(), 0.0];
        event.v_rel = [-0.01, -0.02, 0.005];
        event.p_momentum = event.p_velocity.map(|velocity| velocity * event.p_mass);
        event.dv_vec = [
            event.p_velocity[0] + event.v_rel[0],
            event.p_velocity[1] + event.v_rel[1],
            event.p_velocity[2] + event.v_rel[2],
        ];
        event.kappa = 1.0;
        event.p_am_ratio = Some(0.02);
        event.min_miss_distance_km = 1.0;
        let state = zero_mass_eci_state(&event);
        eci2equinoc_impl(&state, 6, 0.0, 0.0, &mut event.p_equ_intercept);
        let mut propagated = [0.0; 6];
        equinoc2eci_impl(&event.p_equ_intercept, 6, event.tof_s, 0.0, &mut propagated);
        event.p_pos_conj_equ_0.copy_from_slice(&propagated[..3]);
        event.p_pos_conj_truth = event.p_pos_conj_equ_0;

        (event, zero_mass_test_context())
    }

    #[derive(Default)]
    struct FixedImpactRecordingObserver {
        legs: Vec<MassLegIdentity>,
        failure: Option<lightyear_odeint_rs::integrator::FinalPropagationFailure>,
    }

    impl MassSolveObserver for FixedImpactRecordingObserver {
        fn integrate_final_leg(
            &mut self,
            _request: ScalarPropagationRequest<'_>,
            leg: &MassLegIdentity,
        ) -> Result<[f64; 6], lightyear_odeint_rs::integrator::FinalPropagationFailure> {
            self.legs.push(*leg);
            self.failure.map_or(Ok([0.0; 6]), Err)
        }
    }

    #[test]
    fn fixed_impact_hf_uses_retained_body_equation() {
        let (event, context) = fixed_impact_test_fixture();

        for retained_mass_kg in [0.0, f64::MIN_POSITIVE, f64::MAX] {
            let request =
                FixedImpactHfRequest::try_new(event.clone(), context.clone(), retained_mass_kg)
                    .expect("finite nonnegative retained mass is accepted");
            let mut observer = FixedImpactRecordingObserver::default();
            let outcome = evaluate_fixed_impact_hf_with_observer(&request, &mut observer)
                .expect("fixture fixed impact propagates");

            let expected_velocity = compute_new_velocity(retained_mass_kg, &event);
            assert_eq!(
                outcome.impact_state_eci().map(f64::to_bits),
                [
                    event.p_pos_intercept[0],
                    event.p_pos_intercept[1],
                    event.p_pos_intercept[2],
                    expected_velocity[0],
                    expected_velocity[1],
                    expected_velocity[2],
                ]
                .map(f64::to_bits),
                "impact changes velocity only",
            );
            let expected_am_ratio = event.p_am_ratio.expect("fixture A/M")
                * (event.p_mass / (event.p_mass + retained_mass_kg));
            assert_eq!(
                outcome.retained_area_to_mass_ratio_m2_per_kg().to_bits(),
                expected_am_ratio.to_bits(),
                "retained mass adjusts A/M exactly once",
            );
            assert_eq!(
                outcome.sampled_retained_mass_kg().to_bits(),
                retained_mass_kg.to_bits(),
                "evaluator binds exact sampled mass, never an expected-mass surrogate",
            );
            assert_eq!(
                observer.legs.len(),
                1,
                "fixed impact executes exactly one retained-mass propagation",
            );

            let mass_leg = observer
                .legs
                .iter()
                .find(|leg| {
                    leg.tag.role == MassLegRole::MassEvaluation
                        && leg.tag.mass_kg_bits == retained_mass_kg.to_bits()
                })
                .expect("fixed impact executes one typed mass leg");
            assert_eq!(
                mass_leg.initial_eci_bits,
                outcome.impact_state_eci().map(f64::to_bits),
                "propagator receives unchanged impact position and authority velocity",
            );
        }

        for invalid_mass in [-1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                FixedImpactHfRequest::try_new(event.clone(), context.clone(), invalid_mass)
                    .expect_err("hostile retained mass rejects"),
                FixedImpactHfFailure::InvalidMass,
            );
        }

        let mut nonfinite_event = event.clone();
        nonfinite_event.p_pos_intercept[0] = f64::NAN;
        assert_eq!(
            FixedImpactHfRequest::try_new(nonfinite_event, context.clone(), 1.0)
                .expect_err("nonfinite fixed-impact input rejects"),
            FixedImpactHfFailure::NonFinite,
        );

        let mut ground_event = event.clone();
        ground_event.p_pos_intercept = [1.0, 0.0, 0.0];
        let ground_request = FixedImpactHfRequest::try_new(ground_event, context.clone(), 1.0)
            .expect("finite ground-crossing state reaches typed evaluator");
        assert_eq!(
            evaluate_fixed_impact_hf_with_observer(
                &ground_request,
                &mut FixedImpactRecordingObserver::default(),
            )
            .expect_err("ground-crossing state rejects"),
            FixedImpactHfFailure::Ground,
        );

        let mut escape_event = event;
        let escape_speed = (2.1 * MU / escape_event.p_pos_intercept[0]).sqrt();
        escape_event.p_velocity = [0.0, escape_speed, 0.0];
        escape_event.p_momentum = escape_event
            .p_velocity
            .map(|velocity| velocity * escape_event.p_mass);
        escape_event.dv_vec = escape_event.p_velocity;
        escape_event.v_rel = [0.0; 3];
        let escape_request = FixedImpactHfRequest::try_new(escape_event, context, 0.0)
            .expect("finite escape state reaches typed evaluator");
        assert_eq!(
            evaluate_fixed_impact_hf_with_observer(
                &escape_request,
                &mut FixedImpactRecordingObserver::default(),
            )
            .expect_err("escaped state rejects"),
            FixedImpactHfFailure::Escape,
        );

        let (event, context) = fixed_impact_test_fixture();
        let integration_request = FixedImpactHfRequest::try_new(event, context, 1.0)
            .expect("integration fixture request");
        let mut failing_observer = FixedImpactRecordingObserver {
            legs: Vec::new(),
            failure: Some(
                lightyear_odeint_rs::integrator::FinalPropagationFailure::IntegrationFailure,
            ),
        };
        assert_eq!(
            evaluate_fixed_impact_hf_with_observer(&integration_request, &mut failing_observer)
                .expect_err("integration failure stays typed"),
            FixedImpactHfFailure::Integration(
                lightyear_odeint_rs::integrator::FinalPropagationFailure::IntegrationFailure,
            ),
        );
    }

    #[test]
    fn fixed_impact_hf_threshold_seam() {
        let threshold_km = 1.0_f64;
        let next_down = f64::from_bits(threshold_km.to_bits() - 1);
        let next_up = f64::from_bits(threshold_km.to_bits() + 1);

        assert_eq!(
            classify_fixed_impact_hf(next_down, threshold_km),
            FixedImpactHfVerdict::Miss,
        );
        assert_eq!(
            classify_fixed_impact_hf(threshold_km, threshold_km),
            FixedImpactHfVerdict::Miss,
            "equality is Miss",
        );
        assert_eq!(
            classify_fixed_impact_hf(next_up, threshold_km),
            FixedImpactHfVerdict::Safe,
        );
    }

    #[test]
    fn fixed_impact_hf_failure_evidence_ids_are_closed() {
        use lightyear_odeint_rs::integrator::FinalPropagationFailure;
        use lightyear_odeint_rs::probe::PropagationCensusError;
        use lightyear_odeint_rs::EclipseError;

        let cases = [
            (
                FixedImpactHfFailure::InvalidMass,
                "fixed-impact:invalid-mass",
            ),
            (
                FixedImpactHfFailure::InvalidInput,
                "fixed-impact:invalid-input",
            ),
            (FixedImpactHfFailure::NonFinite, "fixed-impact:nonfinite"),
            (
                FixedImpactHfFailure::HighFidelityContextRequired,
                "fixed-impact:hf-context-required",
            ),
            (
                FixedImpactHfFailure::HighFidelityPreparationFailed,
                "fixed-impact:hf-preparation-failed",
            ),
            (
                FixedImpactHfFailure::InvalidOrbit,
                "fixed-impact:invalid-orbit",
            ),
            (FixedImpactHfFailure::Ground, "fixed-impact:ground"),
            (FixedImpactHfFailure::Escape, "fixed-impact:escape"),
            (
                FixedImpactHfFailure::Integration(FinalPropagationFailure::Ground),
                "fixed-impact:integration:ground",
            ),
            (
                FixedImpactHfFailure::Integration(FinalPropagationFailure::Gravity(
                    satpy_core::GravityError::UnsupportedOrder,
                )),
                "fixed-impact:integration:gravity:unsupported-order",
            ),
            (
                FixedImpactHfFailure::Integration(FinalPropagationFailure::Census(
                    PropagationCensusError::CounterOverflow,
                )),
                "fixed-impact:integration:census:counter-overflow",
            ),
            (
                FixedImpactHfFailure::Integration(FinalPropagationFailure::Eclipse(
                    EclipseError::Geometry,
                )),
                "fixed-impact:integration:eclipse:geometry",
            ),
            (
                FixedImpactHfFailure::Integration(FinalPropagationFailure::MethodUnsupported),
                "fixed-impact:integration:method-unsupported",
            ),
        ];

        for (failure, expected) in cases {
            assert_eq!(failure.evidence_id(), expected);
            assert!(is_fixed_impact_hf_failure_evidence_id(expected));
        }
        assert!(!is_fixed_impact_hf_failure_evidence_id("fixed-impact"));
        assert!(!is_fixed_impact_hf_failure_evidence_id(
            "fixed-impact:integration:eclipse:authority:missing"
        ));
    }

    #[derive(Default)]
    struct FixedImpactDirectPropagationObserver {
        legs: Vec<MassLegIdentity>,
    }

    impl MassSolveObserver for FixedImpactDirectPropagationObserver {
        fn integrate_final_leg(
            &mut self,
            _request: ScalarPropagationRequest<'_>,
            leg: &MassLegIdentity,
        ) -> Result<[f64; 6], lightyear_odeint_rs::integrator::FinalPropagationFailure> {
            self.legs.push(*leg);
            Ok([0.0; 6])
        }
    }

    #[test]
    fn fixed_impact_hf_direct_drawn_state_preserves_covariance_displacement() {
        let (nominal_event, context) = fixed_impact_test_fixture();
        let mut drawn_event = nominal_event.clone();
        drawn_event.p_pos_intercept[0] += 0.125;
        let drawn_impact_state = zero_mass_eci_state(&drawn_event);
        eci2equinoc_impl(
            &drawn_impact_state,
            6,
            0.0,
            0.0,
            &mut drawn_event.p_equ_intercept,
        );

        let evaluate = |request: FixedImpactHfRequest| {
            let mut observer = FixedImpactDirectPropagationObserver::default();
            let outcome = evaluate_fixed_impact_hf_with_observer(&request, &mut observer)
                .expect("identity-propagation fixture evaluates");
            assert_eq!(observer.legs.len(), 1, "fixed impact must run one leg");
            let leg = observer
                .legs
                .first()
                .expect("fixed impact recorded its sole propagation leg");
            assert_eq!(
                leg.tag.role,
                MassLegRole::MassEvaluation,
                "sole leg must be retained-mass propagation",
            );
            (outcome, observer.legs)
        };
        let position = |outcome: &FixedImpactHfOutcome| {
            <[f64; 3]>::try_from(&outcome.conjunction_state_eci()[..3])
                .expect("conjunction state carries position")
        };
        let canonical_position_bits = |values: [f64; 3]| {
            values.map(|value| {
                if value == 0.0 {
                    0.0_f64.to_bits()
                } else {
                    value.to_bits()
                }
            })
        };

        let (nominal, nominal_legs) = evaluate(
            FixedImpactHfRequest::try_new(nominal_event, context.clone(), 0.0)
                .expect("nominal drawn-state request"),
        );
        let (drawn, drawn_legs) = evaluate(
            FixedImpactHfRequest::try_new(drawn_event, context, 0.0)
                .expect("displaced drawn-state request"),
        );
        let leg_position_bits = |legs: &[MassLegIdentity]| {
            let leg = legs
                .first()
                .expect("fixed impact recorded its sole propagation leg");
            let equinoctial = leg.initial_equinoctial_bits.map(f64::from_bits);
            let mut propagated = [0.0; 6];
            equinoc_prop_from_impl(
                &equinoctial,
                f64::from_bits(leg.t_final_s_bits),
                &mut propagated,
            );
            canonical_position_bits(
                <[f64; 3]>::try_from(&propagated[..3])
                    .expect("mass leg propagation carries position"),
            )
        };
        assert_eq!(
            canonical_position_bits(position(&nominal)),
            leg_position_bits(&nominal_legs),
            "direct propagation must leave nominal state unshifted",
        );
        assert_eq!(
            canonical_position_bits(position(&drawn)),
            leg_position_bits(&drawn_legs),
            "direct propagation must leave displaced state unshifted",
        );
        assert_ne!(
            canonical_position_bits(position(&drawn)),
            canonical_position_bits(position(&nominal)),
            "covariance displacement was erased",
        );
        let outcome_displacement =
            drawn.conjunction_state_eci()[0] - nominal.conjunction_state_eci()[0];
        let leg_displacement = f64::from_bits(leg_position_bits(&drawn_legs)[0])
            - f64::from_bits(leg_position_bits(&nominal_legs)[0]);
        assert_eq!(
            outcome_displacement.to_bits(),
            leg_displacement.to_bits(),
            "draw displacement changed during direct propagation",
        );
    }

    fn valid_eci_with_infeasible_zero_mass_arc() -> MassSolverEvent {
        let mut event = valid_zero_mass_test_event();
        let apogee_radius_km = event.p_pos_intercept[0];
        let perigee_radius_km = 6_477.0;
        let semi_major_km = 0.5 * (apogee_radius_km + perigee_radius_km);
        let apogee_speed_km_s = (MU * (2.0 / apogee_radius_km - 1.0 / semi_major_km)).sqrt();
        event.p_velocity = [0.0, apogee_speed_km_s, 0.0];
        event.p_momentum = event.p_velocity.map(|velocity| velocity * event.p_mass);
        event.dv_vec = [0.0, apogee_speed_km_s + 1.0, 0.0];
        event.v_rel = [0.0, 1.0, 0.0];

        let state = zero_mass_eci_state(&event);
        eci2equinoc_impl(&state, 6, 0.0, 0.0, &mut event.p_equ_intercept);
        let mut propagated = [0.0; 6];
        equinoc2eci_impl(&event.p_equ_intercept, 6, event.tof_s, 0.0, &mut propagated);
        event.p_pos_conj_equ_0.copy_from_slice(&propagated[..3]);
        event.p_pos_conj_truth = event.p_pos_conj_equ_0;
        event.secondary_conj_pos = event.p_pos_conj_equ_0;
        event
    }

    fn direct_zero_mass_hf_reference(
        event: &MassSolverEvent,
        prepared: &PreparedHfConfig,
    ) -> Option<[f64; 3]> {
        let state = zero_mass_eci_state(event);
        let min_radius_km =
            prepared.force_config.earth_radius + lightyear_odeint_rs::types::GROUND_ALTITUDE;
        if !state_clears_min_radius(&state, min_radius_km) {
            return None;
        }
        let propagated = propagate_target_for_mass_authority(
            &state,
            event.tof_s,
            TargetPropagationAuthority::HighFidelity,
            Some(prepared),
            &mut UnobservedMassSolve,
            MassLegTag {
                role: MassLegRole::ZeroMassAnchor,
                mass_kg_bits: 0.0_f64.to_bits(),
            },
        )
        .ok()?;
        Some([propagated[0], propagated[1], propagated[2]])
    }

    fn assert_zero_mass_reference_bits_eq(actual: Option<[f64; 3]>, expected: Option<[f64; 3]>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => {
                assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
            }
            (None, None) => {}
            (actual, expected) => {
                panic!("anchor mismatch: actual={actual:?}, expected={expected:?}")
            }
        }
    }

    #[test]
    fn zero_mass_hf_anchor_matches_checked_direct_eci_authority() {
        let event = valid_zero_mass_test_event();
        let context = zero_mass_test_context();
        let solve_context = context.clone();
        let prepared = prepare_hf_for_event(&event, &context)
            .expect("HF context")
            .expect("HF preparation");
        let rows = vec![(event.clone(), context)];
        let cache = ZeroMassBatchCache::from_rows(&rows);
        let cached = zero_mass_reference_for_event(
            &event,
            Some(&prepared),
            cache.slot_for_row(0),
            &mut UnobservedMassSolve,
        );
        assert!(cached.exact_hf);
        assert_zero_mass_reference_bits_eq(
            cached.position,
            direct_zero_mass_hf_reference(&event, &prepared),
        );

        let (mass, status) = solve_single_event_hf_with_status(
            &event,
            &SolverConfig::default(),
            Some(&solve_context),
        );
        assert!(mass.is_finite());
        assert_eq!(status, MassSolveStatusCode::Converged);
        let profile = last_hf_mass_solve_profile();
        assert_eq!(profile.detmass_anchor_contract_version, 3);
        assert!(profile.detmass_anchor_internal_reference_used);

        let mut nondefault_context = zero_mass_test_context();
        let mut nondefault_force = *nondefault_context
            .force_config
            .as_ref()
            .expect("force")
            .as_ref();
        nondefault_force.integrator_method = lightyear_odeint_rs::types::StepperMethod::Rkv98;
        nondefault_context.force_config = Some(Arc::new(nondefault_force));
        let nondefault_prepared = prepare_hf_for_event(&event, &nondefault_context)
            .expect("HF context")
            .expect("HF preparation");
        let nondefault_rows = vec![(event.clone(), nondefault_context)];
        let nondefault_cache = ZeroMassBatchCache::from_rows(&nondefault_rows);
        let nondefault_cached = zero_mass_reference_for_event(
            &event,
            Some(&nondefault_prepared),
            nondefault_cache.slot_for_row(0),
            &mut UnobservedMassSolve,
        );
        assert!(nondefault_cached.exact_hf);
        assert_zero_mass_reference_bits_eq(
            nondefault_cached.position,
            direct_zero_mass_hf_reference(&event, &nondefault_prepared),
        );
    }

    #[test]
    fn zero_mass_hf_anchor_uses_eci_failure_without_lf_fallback() {
        let mut event = valid_zero_mass_test_event();
        event.p_pos_intercept = [0.0; 3];
        let context = zero_mass_test_context();
        let prepared = prepare_hf_for_event(&event, &context)
            .expect("HF context")
            .expect("HF preparation");

        assert!(direct_zero_mass_hf_reference(&event, &prepared).is_none());
        assert!(zero_mass_reference_hf_from_intercept_eci(
            &event,
            &prepared,
            &mut UnobservedMassSolve,
        )
        .is_none());
        let rows = vec![(event.clone(), context.clone())];
        let cache = ZeroMassBatchCache::from_rows(&rows);
        let reference = zero_mass_reference_for_event(
            &event,
            Some(&prepared),
            cache.slot_for_row(0),
            &mut UnobservedMassSolve,
        );
        assert_eq!(reference.position, None);
        assert!(!reference.exact_hf);

        let (mass, status) =
            solve_single_event_hf_with_status(&event, &SolverConfig::default(), Some(&context));
        assert!(mass.is_nan());
        assert_eq!(
            status,
            MassSolveStatusCode::HfTrajectoryPhysicallyInfeasible
        );
    }

    #[test]
    fn strict_validate_only_rejects_failed_zero_mass_hf_anchor_before_seed_use() {
        let event = valid_eci_with_infeasible_zero_mass_arc();
        let mut context = zero_mass_test_context();
        context.hf_validate_only = true;
        let prepared = prepare_hf_for_event(&event, &context)
            .expect("HF context")
            .expect("HF preparation");
        let min_radius_km =
            prepared.force_config.earth_radius + lightyear_odeint_rs::types::GROUND_ALTITUDE;
        assert!(event
            .p_pos_intercept
            .iter()
            .chain(event.p_velocity.iter())
            .all(|value| value.is_finite()));
        assert!(event.p_pos_intercept[0] > min_radius_km);
        assert!(!state_clears_min_radius(
            &zero_mass_eci_state(&event),
            min_radius_km
        ));
        assert!(zero_mass_reference_hf_from_intercept_eci(
            &event,
            &prepared,
            &mut UnobservedMassSolve,
        )
        .is_none());

        let config = SolverConfig {
            xtol: 1e-6,
            rtol: 1e-6,
            maxiter: 8,
            mass_max: 1.0e6,
        };
        let (lf_seed, lf_status) = solve_single_event_hf_with_status(&event, &config, None);
        assert_eq!(lf_status, MassSolveStatusCode::Converged);
        assert!(lf_seed.is_finite() && lf_seed > 0.0);
        let seed_velocity = compute_new_velocity(lf_seed, &event);
        assert!(state_clears_min_radius(
            &[
                event.p_pos_intercept[0],
                event.p_pos_intercept[1],
                event.p_pos_intercept[2],
                seed_velocity[0],
                seed_velocity[1],
                seed_velocity[2],
            ],
            min_radius_km,
        ));

        hf_profile_reset();
        let (mass, status) = solve_single_event_hf_validate_only(
            &event,
            &config,
            &context,
            None,
            &mut UnobservedMassSolve,
        );
        assert!(mass.is_nan());
        assert_eq!(
            status,
            MassSolveStatusCode::HfTrajectoryPhysicallyInfeasible
        );
        let profile = last_hf_mass_solve_profile();
        assert_eq!(DETMASS_ANCHOR_CONTRACT_VERSION, 3);
        assert_eq!(
            profile.detmass_anchor_contract_version,
            DETMASS_ANCHOR_CONTRACT_VERSION
        );
        assert_eq!(
            profile.detmass_anchor_shift_norm_km.to_bits(),
            0.0f64.to_bits()
        );
        assert!(!profile.detmass_anchor_internal_reference_used);
    }

    #[test]
    fn zero_mass_anchor_cache_splits_eci_state_drift_with_fixed_equinoctial_state() {
        let event = valid_zero_mass_test_event();
        let context = zero_mass_test_context();
        let next = next_fixture_f64;
        let mut position_drift = event.clone();
        let [position_component, ..] = &mut position_drift.p_pos_intercept;
        *position_component = next(*position_component);
        let mut velocity_drift = event.clone();
        let [_, velocity_component, ..] = &mut velocity_drift.p_velocity;
        *velocity_component = next(*velocity_component);

        assert!(position_drift
            .p_equ_intercept
            .iter()
            .zip(event.p_equ_intercept.iter())
            .all(|(actual, expected)| same_or_both_nan(*actual, *expected)));
        assert!(velocity_drift
            .p_equ_intercept
            .iter()
            .zip(event.p_equ_intercept.iter())
            .all(|(actual, expected)| same_or_both_nan(*actual, *expected)));
        assert_ne!(
            AnchorAuthorityKey::new(&event, &context),
            AnchorAuthorityKey::new(&position_drift, &context)
        );
        assert_ne!(
            AnchorAuthorityKey::new(&event, &context),
            AnchorAuthorityKey::new(&velocity_drift, &context)
        );

        let rows = vec![
            (event, context.clone()),
            (position_drift, context.clone()),
            (velocity_drift, context),
        ];
        let cache = ZeroMassBatchCache::from_rows(&rows);
        assert_eq!(cache.anchor_slots.len(), 3);
        assert_eq!(cache.row_to_anchor_slot, vec![0, 1, 2]);
    }

    #[test]
    fn zero_mass_anchor_cache_splits_mixed_authority_rows() {
        let event = valid_zero_mass_test_event();
        let context = zero_mass_test_context();
        let next = next_fixture_f64;

        let mut tof_event = event.clone();
        tof_event.tof_s = next(tof_event.tof_s);
        let tof_context = context.clone();
        assert!(Arc::ptr_eq(
            context.force_config.as_ref().unwrap(),
            tof_context.force_config.as_ref().unwrap()
        ));
        assert!(Arc::ptr_eq(
            context.packed_coeffs.as_ref().unwrap(),
            tof_context.packed_coeffs.as_ref().unwrap()
        ));

        let mut force_context = context.clone();
        let mut force = *force_context.force_config.as_ref().unwrap().as_ref();
        force.dt_max = next(force.dt_max);
        force_context.force_config = Some(Arc::new(force));

        let mut asset_context = context.clone();
        asset_context.packed_coeffs = Some(Arc::new(
            satpy_core::pack_gravity_coeffs(&[next(0.0)], &[0.0], 1, 0)
                .expect("test gravity coefficients are valid"),
        ));

        let mut integrator_context = context.clone();
        let mut integrator_force = *integrator_context.force_config.as_ref().unwrap().as_ref();
        integrator_force.integrator_method = lightyear_odeint_rs::types::StepperMethod::Rkv98;
        integrator_context.force_config = Some(Arc::new(integrator_force));

        let rows = vec![
            (event.clone(), context),
            (tof_event, tof_context),
            (event.clone(), force_context),
            (event.clone(), asset_context),
            (event, integrator_context),
        ];
        let cache = ZeroMassBatchCache::from_rows(&rows);
        assert_eq!(cache.anchor_slots.len(), 5);
        assert_eq!(cache.row_to_anchor_slot, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn exact_zero_mass_miss_requires_hf_anchor_authority() {
        let derived = EventDerived {
            mu_over_r_intercept: 1.0,
            anchor_shift: [0.5, -0.25, 1.5],
            apply_anchor_shift: true,
            anchor_shift_norm_km: 0.0,
            anchor_contract_version: DETMASS_ANCHOR_CONTRACT_VERSION,
            anchor_internal_reference_used: true,
        };
        let position = [10.0, -2.0, 3.0];
        let secondary = [-1.0, 4.0, 8.0];
        let expected = vec3_distance(&apply_anchored_adjustment(position, &derived), &secondary);

        let exact = ZeroMassReference {
            position: Some(position),
            exact_hf: true,
        };
        assert_eq!(
            exact_zero_mass_miss(exact, &derived, &secondary)
                .expect("exact HF anchor must provide zero-mass miss")
                .to_bits(),
            expected.to_bits()
        );

        let fallback = ZeroMassReference {
            position: Some(position),
            exact_hf: false,
        };
        assert!(exact_zero_mass_miss(fallback, &derived, &secondary).is_none());
        assert!(exact_zero_mass_miss(
            ZeroMassReference {
                position: None,
                exact_hf: true,
            },
            &derived,
            &secondary,
        )
        .is_none());
    }

    #[test]
    fn zero_mass_cache_groups_dv_only_rows_and_initializes_once() {
        let first = valid_zero_mass_test_event();
        let mut second = first.clone();
        let [dv_x, ..] = &mut second.dv_vec;
        *dv_x = next_fixture_f64(*dv_x);
        let [velocity_x, ..] = second.p_velocity;
        let [relative_velocity_x, ..] = &mut second.v_rel;
        *relative_velocity_x = *dv_x - velocity_x;
        let context = zero_mass_test_context();
        let rows = vec![(first, context.clone()), (second, context)];

        let cache = ZeroMassBatchCache::from_rows(&rows);
        assert_eq!(cache.anchor_slots.len(), 1);
        assert_eq!(cache.row_to_anchor_slot, vec![0, 0]);
        assert_eq!(cache.miss_slots.len(), 2);

        let calls = std::sync::atomic::AtomicUsize::new(0);
        let slot = cache.anchor_slots.first();
        assert!(slot.is_some(), "cache must contain its shared anchor slot");
        let Some(slot) = slot else {
            return;
        };
        for _ in 0..2 {
            let value = slot.get_or_init(|| {
                calls.fetch_add(1, Ordering::Relaxed);
                ZeroMassReference {
                    position: Some([1.0, 2.0, 3.0]),
                    exact_hf: true,
                }
            });
            assert_eq!(value.position, Some([1.0, 2.0, 3.0]));
            assert!(value.exact_hf);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let miss_calls = std::sync::atomic::AtomicUsize::new(0);
        let miss_slot = cache.miss_slots.first();
        assert!(
            miss_slot.is_some(),
            "cache must contain its first miss slot"
        );
        let Some(miss_slot) = miss_slot else {
            return;
        };
        for _ in 0..2 {
            let value = miss_slot.get_or_init(|| {
                miss_calls.fetch_add(1, Ordering::Relaxed);
                4.5
            });
            assert_eq!(value.to_bits(), 4.5_f64.to_bits());
        }
        assert_eq!(miss_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn preinitialize_hf_anchor_slots_matches_the_row_pass_value_bit_for_bit() {
        let first = valid_zero_mass_test_event();
        let mut second = first.clone();
        let [dv_x, ..] = &mut second.dv_vec;
        *dv_x = next_fixture_f64(*dv_x);
        let [velocity_x, ..] = second.p_velocity;
        let [relative_velocity_x, ..] = &mut second.v_rel;
        *relative_velocity_x = *dv_x - velocity_x;
        let context = zero_mass_test_context();
        let rows = vec![(first.clone(), context.clone()), (second, context.clone())];

        let cache = ZeroMassBatchCache::from_rows(&rows);
        assert_eq!(
            cache.anchor_slots.len(),
            1,
            "fixture rows must share one anchor slot"
        );
        cache.preinitialize_hf_anchor_slots_parallel(&rows);
        let preinitialized = cache
            .anchor_slots
            .first()
            .and_then(|slot| slot.get())
            .copied()
            .expect("pre-initialization must fill the shared HF anchor slot");

        // The row pass would run the same uncached initializer with the same
        // prepared inputs; the pre-initialized value must match it bit for bit.
        let prepared = match prepare_hf_for_event(&first, &context) {
            Some(Ok(prepared)) => prepared,
            other => panic!(
                "HF fixture preparation must succeed, got {:?}",
                other.map(|r| r.is_ok())
            ),
        };
        let reference = zero_mass_reference_for_event_uncached(
            &first,
            Some(&prepared),
            &mut UnobservedMassSolve,
        );
        assert_eq!(preinitialized.exact_hf, reference.exact_hf);
        match (preinitialized.position, reference.position) {
            (Some(pre), Some(row)) => {
                for (pre_axis, row_axis) in pre.iter().zip(row) {
                    assert_eq!(pre_axis.to_bits(), row_axis.to_bits());
                }
            }
            (None, None) => {}
            (pre, row) => panic!("anchor positions diverged: pre={pre:?} row={row:?}"),
        }
    }

    #[test]
    fn preinitialize_declines_validate_only_and_non_hf_rows() {
        let event = valid_zero_mass_test_event();

        let mut validate_only = zero_mass_test_context();
        validate_only.hf_validate_only = true;
        let rows = vec![(event.clone(), validate_only)];
        let cache = ZeroMassBatchCache::from_rows(&rows);
        cache.preinitialize_hf_anchor_slots_parallel(&rows);
        assert!(
            cache.anchor_slots.iter().all(|slot| slot.get().is_none()),
            "validate-only rows must keep their own anchor route"
        );

        let mut low_fidelity = zero_mass_test_context();
        low_fidelity.use_high_fidelity = false;
        let rows = vec![(event, low_fidelity)];
        let cache = ZeroMassBatchCache::from_rows(&rows);
        cache.preinitialize_hf_anchor_slots_parallel(&rows);
        assert!(
            cache.anchor_slots.iter().all(|slot| slot.get().is_none()),
            "non-HF rows keep the cheap LF anchor init in the row pass"
        );
    }

    #[test]
    fn zero_mass_cache_rejects_one_bit_authority_drift() {
        let event = valid_zero_mass_test_event();
        let context = zero_mass_test_context();
        let authority = ZeroMassAuthorityKey::new(&event, &context);
        let next = next_fixture_f64;

        let mut variants = Vec::new();
        push_one_bit_array_variants(&mut variants, &event, |event| &mut event.p_pos_intercept);
        push_one_bit_array_variants(&mut variants, &event, |event| &mut event.p_velocity);
        push_one_bit_array_variants(&mut variants, &event, |event| &mut event.secondary_conj_pos);
        push_one_bit_array_variants(&mut variants, &event, |event| &mut event.p_pos_conj_truth);
        push_one_bit_array_variants(&mut variants, &event, |event| &mut event.p_pos_conj_equ_0);
        push_one_bit_array_variants(&mut variants, &event, |event| &mut event.p_equ_intercept);
        let mut changed = event.clone();
        changed.tof_s = next(changed.tof_s);
        variants.push(changed);
        for field in 0..5 {
            let mut changed = event.clone();
            let value = next(0.0);
            match field {
                0 => changed.p_am_ratio = Some(value),
                1 => changed.p_cd = Some(value),
                2 => changed.p_cr = Some(value),
                3 => changed.p_qm_ratio = Some(value),
                _ => changed.p_r_obj_m = Some(value),
            }
            variants.push(changed);
        }
        for changed in variants {
            assert_ne!(
                ZeroMassAuthorityKey::new(&changed, &context),
                authority,
                "one-bit event authority drift must split cache"
            );
        }

        let mut epoch_changed = context.clone();
        epoch_changed.epoch_jd = next(epoch_changed.epoch_jd);
        assert_ne!(ZeroMassAuthorityKey::new(&event, &epoch_changed), authority);
        let mut force_changed = context.clone();
        let mut force = *force_changed.force_config.as_ref().unwrap().as_ref();
        force.eps = next(force.eps);
        force_changed.force_config = Some(Arc::new(force));
        assert_ne!(ZeroMassAuthorityKey::new(&event, &force_changed), authority);
        let mut coeff_changed = context.clone();
        coeff_changed.packed_coeffs = Some(Arc::new(
            satpy_core::pack_gravity_coeffs(&[next(0.0)], &[0.0], 1, 0)
                .expect("test gravity coefficients are valid"),
        ));
        assert_ne!(ZeroMassAuthorityKey::new(&event, &coeff_changed), authority);

        let mut mass_changed = event.clone();
        mass_changed.p_mass = next(mass_changed.p_mass);
        assert_ne!(
            ZeroMassAuthorityKey::new(&mass_changed, &context),
            authority
        );
        let mut kappa_changed = event.clone();
        kappa_changed.kappa = next(kappa_changed.kappa);
        assert_ne!(
            ZeroMassAuthorityKey::new(&kappa_changed, &context),
            authority
        );
        let [relative_velocity_x, _, relative_velocity_z] = event.v_rel;
        for value in [next(relative_velocity_x), f64::NAN] {
            let mut relative_velocity_changed = event.clone();
            let [component, ..] = &mut relative_velocity_changed.v_rel;
            *component = value;
            assert_ne!(
                ZeroMassAuthorityKey::new(&relative_velocity_changed, &context),
                authority
            );
        }
        assert_eq!(relative_velocity_z.to_bits(), 0.0f64.to_bits());
        let mut signed_zero_changed = event;
        let [_, _, component] = &mut signed_zero_changed.v_rel;
        *component = -0.0;
        assert_ne!(
            ZeroMassAuthorityKey::new(&signed_zero_changed, &context),
            authority
        );
    }

    /// The fixed-width keys pad the absent force-config block with zeros, and
    /// `AuthorityBitsWriter::finish` asserts EXACT fill — so this test is what
    /// makes a forgotten padding arm (or a mis-declared word capacity) a red
    /// test on the None path, which the drift test above never takes.
    #[test]
    fn authority_keys_fill_exactly_on_the_absent_force_config_arm() {
        let event = valid_zero_mass_test_event();
        let mut no_force = zero_mass_test_context();
        no_force.force_config = None;

        let zero_mass_key = ZeroMassAuthorityKey::new(&event, &no_force);
        let anchor_key = AnchorAuthorityKey::new(&event, &no_force);
        assert_eq!(zero_mass_key, ZeroMassAuthorityKey::new(&event, &no_force));
        assert_eq!(anchor_key, AnchorAuthorityKey::new(&event, &no_force));

        // The presence tag, not the padding, is what separates an absent
        // config from a present one: the two keys must differ even though
        // both arms now occupy the same word count.
        let mut aligned = zero_mass_test_context();
        aligned.packed_coeffs = no_force.packed_coeffs;
        assert_ne!(zero_mass_key, ZeroMassAuthorityKey::new(&event, &aligned));
        assert_ne!(anchor_key, AnchorAuthorityKey::new(&event, &aligned));
    }

    #[test]
    fn zero_mass_cache_parallel_batch_matches_scalar_at_mass_threshold() {
        let base = valid_zero_mass_test_event();
        let mut events = vec![base; 33];
        let last_event = events.last_mut();
        assert!(
            last_event.is_some(),
            "mass-threshold fixture must have a final row"
        );
        let Some(last_event) = last_event else {
            return;
        };
        last_event.p_mass = 5.0e-10;
        let last_mass = last_event.p_mass;
        last_event.p_momentum = last_event.p_velocity.map(|velocity| velocity * last_mass);
        let context = zero_mass_test_context();
        let config = SolverConfig {
            xtol: 1e-3,
            rtol: 1e-6,
            maxiter: 4,
            mass_max: 10.0,
        };
        let scalar: Vec<_> = events
            .iter()
            .map(|event| solve_single_event_hf_with_status(event, &config, Some(&context)))
            .collect();
        let last_scalar = scalar.last();
        assert!(
            last_scalar.is_some(),
            "scalar fixture must have a final row"
        );
        let Some((_, last_status)) = last_scalar else {
            return;
        };
        assert_eq!(*last_status, MassSolveStatusCode::MissAtZeroInvalidVelocity);

        let mut masses = vec![0.0; events.len()];
        let mut statuses = vec![MassSolveStatusCode::MissAtZeroNonFinite; events.len()];
        let mut profiles = vec![HfMassSolveProfile::default(); events.len()];
        let global_width = rayon::current_num_threads();
        solve_batch_events_hf_with_status_and_profiles_from_builder_global_rayon(
            events.len(),
            &config,
            |row| {
                events
                    .get(row)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("batch event row {row} out of range"))
            },
            |_| Ok(context.clone()),
            &mut masses,
            &mut statuses,
            &mut profiles,
            global_width,
            global_width,
        )
        .expect("parallel batch");
        assert_eq!(masses.len(), scalar.len(), "batch mass rows changed");
        assert_eq!(statuses.len(), scalar.len(), "batch status rows changed");
        assert_eq!(profiles.len(), scalar.len(), "batch profile rows changed");
        for (row, (((mass, status), profile), (scalar_mass, scalar_status))) in masses
            .iter()
            .zip(&statuses)
            .zip(&profiles)
            .zip(&scalar)
            .enumerate()
        {
            assert_eq!(*status, *scalar_status, "row {row} status drift");
            assert!(
                same_or_both_nan(*mass, *scalar_mass),
                "row {row} mass drift"
            );
            let event = events.get(row);
            assert!(event.is_some(), "row {row} must retain its source event");
            let Some(event) = event else {
                return;
            };
            let scalar_profile = {
                let _result = solve_single_event_hf_with_status(event, &config, Some(&context));
                last_hf_mass_solve_profile()
            };
            assert_eq!(
                (
                    profile.hf_miss_calls_total,
                    profile.hf_full_bracket_calls,
                    profile.hf_full_refine_calls,
                    profile.hf_full_refine_iterations,
                    profile.hf_upper_bracket_shrink_iterations,
                    profile.detmass_anchor_contract_version,
                    profile.detmass_anchor_internal_reference_used,
                ),
                (
                    scalar_profile.hf_miss_calls_total,
                    scalar_profile.hf_full_bracket_calls,
                    scalar_profile.hf_full_refine_calls,
                    scalar_profile.hf_full_refine_iterations,
                    scalar_profile.hf_upper_bracket_shrink_iterations,
                    scalar_profile.detmass_anchor_contract_version,
                    scalar_profile.detmass_anchor_internal_reference_used,
                ),
                "row {row} profile counter drift"
            );
        }
    }

    #[test]
    fn zero_mass_cache_preserves_scalar_rows_profiles_and_typed_failure() {
        let first = valid_zero_mass_test_event();
        let mut second = first.clone();
        let [dv_x, ..] = &mut second.dv_vec;
        *dv_x = 0.75;
        let [velocity_x, ..] = second.p_velocity;
        let [relative_velocity_x, ..] = &mut second.v_rel;
        *relative_velocity_x = *dv_x - velocity_x;
        second.min_miss_distance_km = 1.0e9;
        let events = [first, second];
        let context = zero_mass_test_context();
        let contexts = [context.clone(), context];
        let config = SolverConfig {
            xtol: 1e-3,
            rtol: 1e-6,
            maxiter: 4,
            mass_max: 10.0,
        };

        let mut scalar_masses = [0.0; 2];
        let mut scalar_statuses = [MassSolveStatusCode::MissAtZeroNonFinite; 2];
        let mut scalar_profiles = [HfMassSolveProfile::default(); 2];
        for (((event, context), mass_out), (status_out, profile_out)) in events
            .iter()
            .zip(&contexts)
            .zip(&mut scalar_masses)
            .zip(scalar_statuses.iter_mut().zip(&mut scalar_profiles))
        {
            (*mass_out, *status_out) =
                solve_single_event_hf_with_status(event, &config, Some(context));
            *profile_out = last_hf_mass_solve_profile();
        }
        let hostile_status = scalar_statuses.last();
        assert!(
            hostile_status.is_some(),
            "fixture must contain the hostile row"
        );
        let Some(hostile_status) = hostile_status else {
            return;
        };
        assert_ne!(
            *hostile_status,
            MassSolveStatusCode::Converged,
            "hostile row must retain typed non-convergence"
        );

        let mut batch_masses = [0.0; 2];
        let mut batch_statuses = [MassSolveStatusCode::MissAtZeroNonFinite; 2];
        let mut batch_profiles = [HfMassSolveProfile::default(); 2];
        solve_batch_events_hf_with_status_and_profiles_from_builder_global_rayon(
            2,
            &config,
            |row| {
                events
                    .get(row)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("batch event row {row} out of range"))
            },
            |row| {
                contexts
                    .get(row)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("batch context row {row} out of range"))
            },
            &mut batch_masses,
            &mut batch_statuses,
            &mut batch_profiles,
            1,
            1,
        )
        .expect("batch solve");

        assert_eq!(batch_statuses, scalar_statuses);
        for (row, ((batch_mass, scalar_mass), (batch_profile, scalar_profile))) in batch_masses
            .iter()
            .zip(&scalar_masses)
            .zip(batch_profiles.iter().zip(&scalar_profiles))
            .enumerate()
        {
            assert!(
                same_or_both_nan(*batch_mass, *scalar_mass),
                "row {row} mass drift"
            );
            assert_eq!(
                batch_profile.hf_miss_calls_total, scalar_profile.hf_miss_calls_total,
                "row {row} miss-call drift"
            );
            assert_eq!(
                batch_profile.hf_full_bracket_calls, scalar_profile.hf_full_bracket_calls,
                "row {row} bracket-call drift"
            );
            assert_eq!(
                batch_profile.hf_full_refine_calls, scalar_profile.hf_full_refine_calls,
                "row {row} refine-call drift"
            );
            assert_eq!(
                batch_profile.detmass_anchor_contract_version,
                scalar_profile.detmass_anchor_contract_version,
                "row {row} anchor-contract drift"
            );
            assert!(
                same_or_both_nan(
                    batch_profile.detmass_anchor_shift_norm_km,
                    scalar_profile.detmass_anchor_shift_norm_km,
                ),
                "row {row} anchor-shift drift"
            );
            assert_eq!(
                batch_profile.detmass_anchor_internal_reference_used,
                scalar_profile.detmass_anchor_internal_reference_used,
                "row {row} anchor-source drift"
            );
        }
    }

    #[test]
    fn public_per_context_global_batch_preserves_scalar_order_and_statuses() {
        let first = valid_zero_mass_test_event();
        let mut second = first.clone();
        let [dv_x, ..] = &mut second.dv_vec;
        *dv_x = 0.75;
        let [velocity_x, ..] = second.p_velocity;
        let [relative_velocity_x, ..] = &mut second.v_rel;
        *relative_velocity_x = *dv_x - velocity_x;
        second.min_miss_distance_km = 1.0e9;
        let events = [first, second];
        let first_context = zero_mass_test_context();
        let mut second_context = first_context.clone();
        second_context.epoch_jd += 0.25;
        let contexts = [first_context, second_context];
        let config = SolverConfig {
            xtol: 1e-3,
            rtol: 1e-6,
            maxiter: 4,
            mass_max: 10.0,
        };

        let scalar: Vec<_> = events
            .iter()
            .zip(&contexts)
            .map(|(event, context)| {
                solve_single_event_hf_with_status(event, &config, Some(context))
            })
            .collect();
        let batch =
            solve_batch_events_hf_with_status_per_context_global_rayon(&events, &contexts, &config)
                .expect("per-context global batch");

        assert_eq!(batch.len(), scalar.len());
        for (row, ((batch_mass, batch_status), (scalar_mass, scalar_status))) in
            batch.iter().zip(&scalar).enumerate()
        {
            assert_eq!(*batch_status, *scalar_status, "row {row} status drift");
            assert!(
                same_or_both_nan(*batch_mass, *scalar_mass),
                "row {row} mass drift: batch={batch_mass} scalar={scalar_mass}"
            );
        }

        let error = solve_batch_events_hf_with_status_per_context_global_rayon(
            &events,
            &contexts[..1],
            &config,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("contexts length"), "{error}");
    }

    #[test]
    fn production_strict_hf_batch_issues_operational_mass_from_executed_rows() {
        let first = valid_zero_mass_test_event();
        let mut second = first.clone();
        second.min_miss_distance_km = 1.0e9;
        let events = [first, second];
        let contexts = [zero_mass_test_context(), zero_mass_test_context()];
        let config = SolverConfig::default();
        let issued = solve_batch_events_hf_with_evidence_per_context_global_rayon(
            &events, &contexts, &config,
        )
        .expect("evidence strict-HF batch");

        assert_eq!(issued.len(), 2);
        let converged = issued.first().copied().expect("converged row");
        assert_eq!(converged.status(), MassSolveStatusCode::Converged);
        let operational = converged
            .operational_mass()
            .expect("executed strict-HF evidence is authentic")
            .expect("positive converged strict-HF row must issue operational mass");
        assert_eq!(
            operational.raw_solver_mass_kg().to_bits(),
            converged.mass_kg().to_bits(),
            "operational raw mass differs from the executed batch row"
        );
        let rejected = issued.get(1).copied().expect("rejected row");
        assert_ne!(rejected.status(), MassSolveStatusCode::Converged);
        assert!(
            rejected
                .operational_mass()
                .expect("nonconverged outcome is not malformed")
                .is_none(),
            "nonconverged strict-HF row forged operational deterministic mass"
        );
    }

    #[cfg(feature = "solver-qualification")]
    #[test]
    #[ignore = "requires a fresh process before immutable W1 Rayon authority is initialized"]
    fn qualification_w1_batch_uses_shared_zero_mass_cache_without_fake_hits() {
        assert_eq!(
            nd_sched::init_global_pool_authoritative(1)
                .expect("initialize authoritative qualification W1"),
            1
        );
        let event = valid_zero_mass_test_event();
        let context = zero_mass_test_context();
        let config = SolverConfig {
            xtol: 1e-6,
            rtol: 1e-6,
            maxiter: 12,
            mass_max: 1.0e6,
        };
        let events = [event.clone(), event];
        let contexts = [context.clone(), context];
        let canonical =
            solve_batch_events_hf_with_status_per_context_global_rayon(&events, &contexts, &config)
                .expect("canonical strict-HF batch");
        let mut leg_slots: [Option<QualificationMassLeg>; 128] = std::array::from_fn(|_| None);
        let mut row_slots: [Option<QualificationMassBatchRow>; 2] = std::array::from_fn(|_| None);
        let mut observed = QualificationMassBatchObservation::new(&mut leg_slots, &mut row_slots)
            .expect("empty bounded qualification batch storage");

        solve_qualification_hf_batch_serial(&events, &contexts, &config, 1, &mut observed)
            .expect("observed strict-HF W1 batch");

        assert_eq!(observed.row_count(), canonical.len());
        let mut anchor_count = 0_usize;
        let mut row_anchor_counts = [0_usize; 2];
        for (row_index, (canonical_mass, canonical_status)) in canonical.iter().enumerate() {
            let row = observed.row(row_index).expect("observed row");
            assert_eq!(
                row.status, *canonical_status,
                "row {row_index} status drift"
            );
            assert_eq!(
                row.mass_kg_bits,
                canonical_mass.to_bits(),
                "row {row_index} mass drift"
            );
            let first_leg = usize::try_from(row.first_leg).expect("row first-leg fits usize");
            let leg_count = usize::try_from(row.leg_count).expect("row leg-count fits usize");
            for leg_index in first_leg..first_leg.saturating_add(leg_count) {
                let leg = observed.leg(leg_index).expect("observed scalar leg");
                if leg.role == QualificationMassLegRole::ZeroMassAnchor {
                    anchor_count = anchor_count.saturating_add(1);
                    let row_anchor_count = row_anchor_counts
                        .get_mut(row_index)
                        .expect("qualification test row index is bounded");
                    *row_anchor_count = row_anchor_count.saturating_add(1);
                }
            }
        }
        assert_eq!(anchor_count, 1, "shared cache must emit one actual anchor");
        assert_eq!(
            row_anchor_counts,
            [1, 0],
            "only the cache owner emits the anchor"
        );
    }
}
