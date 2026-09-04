//! Core types for two-phase transfer optimization.
//!
//! This module defines the main data structures used throughout the transfer optimization:
//! - `PlanResult`: Complete result of a transfer plan evaluation
//! - `PlanContext`: Optimization context with constraints and cached data

use crate::oxymoo::local::{LocalOptimizerKind, TuneLevel};
use lightyear_odeint_rs::types::OdeMetrics;
use satpy_core::{cross3, norm3, MU, RE, SEC_PER_DAY};

/// True when every element of `arr` is finite. Shared by the evaluator, the
/// verifier and postprocess, which each used to carry their own copy.
#[inline]
#[must_use]
pub fn all_finite(arr: &[f64]) -> bool {
    arr.iter().all(|x| x.is_finite())
}
#[cfg(test)]
use std::cell::Cell;
use std::sync::Arc;

/// Declares a counter struct's field roster ONCE and stamps the struct plus
/// its field-wise merge (and optionally delta) bodies from that single list,
/// so the add/sub/merge lists can no longer desynchronize from the struct --
/// a counter missed in a hand-kept merge list used to silently merge to 0.
///
/// Field kinds:
/// - `count`: `usize`, merged with `checked_add`, delta with `checked_sub`;
/// - `f64_sum`: `f64`, merged with `+=`, delta with `-`;
/// - `max`: merged with `.max(..)` (no delta form);
/// - `bool_or`: merged with `|=` (no delta form);
/// - `nested`: another roster struct, merged via its `add_delta`, delta via
///   its `delta_since`.
///
/// Emits `fn roster_add(merged, incoming)` mutating `merged` field by field in
/// declaration order -- callers own the copy-then-commit transaction, so an
/// overflow error never publishes partial sums (the overflow error is a single
/// value, so which field trips first is unobservable) -- and, under
/// `sub = test` / `sub = production`, `fn roster_delta_since(current, before)`
/// (`sub = test` gates it `#[cfg(test)]`). The f64 `+=`/`-` are one operation
/// per field, so emission order cannot move a bit.
macro_rules! counter_roster {
    (
        error = $err:ty;
        overflow = $overflow:expr;
        sub = $sub:ident;
        $(#[$smeta:meta])*
        $svis:vis struct $name:ident {
            $(
                $(#[$fmeta:meta])*
                $kind:ident $field:ident: $fty:ty
            ),+ $(,)?
        }
    ) => {
        $(#[$smeta])*
        $svis struct $name {
            $( $(#[$fmeta])* pub $field: $fty, )+
        }

        impl $name {
            fn roster_add(merged: &mut Self, incoming: &Self) -> Result<(), $err> {
                $( counter_roster!(@add $kind, merged, incoming, $field, $overflow); )+
                Ok(())
            }

            counter_roster!(@sub_fns $sub, $err, $overflow; $( $kind $field ),+ );
        }
    };
    (@add count, $m:ident, $i:ident, $f:ident, $overflow:expr) => {
        $m.$f = $m.$f.checked_add($i.$f).ok_or($overflow)?;
    };
    (@add f64_sum, $m:ident, $i:ident, $f:ident, $overflow:expr) => {
        $m.$f += $i.$f;
    };
    (@add max, $m:ident, $i:ident, $f:ident, $overflow:expr) => {
        $m.$f = $m.$f.max($i.$f);
    };
    (@add bool_or, $m:ident, $i:ident, $f:ident, $overflow:expr) => {
        $m.$f |= $i.$f;
    };
    (@add nested, $m:ident, $i:ident, $f:ident, $overflow:expr) => {
        $m.$f.add_delta(&$i.$f)?;
    };
    (@sub_fns none, $err:ty, $overflow:expr; $($rest:tt)*) => {};
    (@sub_fns test, $err:ty, $overflow:expr; $( $kind:ident $field:ident ),+ ) => {
        #[cfg(test)]
        fn roster_delta_since(current: &Self, before: &Self) -> Result<Self, $err> {
            let mut delta = Self::default();
            $( counter_roster!(@sub $kind, delta, current, before, $field, $overflow); )+
            Ok(delta)
        }
    };
    (@sub_fns production, $err:ty, $overflow:expr; $( $kind:ident $field:ident ),+ ) => {
        fn roster_delta_since(current: &Self, before: &Self) -> Result<Self, $err> {
            let mut delta = Self::default();
            $( counter_roster!(@sub $kind, delta, current, before, $field, $overflow); )+
            Ok(delta)
        }
    };
    (@sub count, $d:ident, $c:ident, $b:ident, $f:ident, $overflow:expr) => {
        $d.$f = $c.$f.checked_sub($b.$f).ok_or($overflow)?;
    };
    (@sub f64_sum, $d:ident, $c:ident, $b:ident, $f:ident, $overflow:expr) => {
        $d.$f = $c.$f - $b.$f;
    };
    (@sub nested, $d:ident, $c:ident, $b:ident, $f:ident, $overflow:expr) => {
        $d.$f = $c.$f.delta_since($b.$f)?;
    };
}
pub(crate) use counter_roster;

// ============================================================================
// Constants
// ============================================================================

/// Invalid cost sentinel value
pub const INVALID_COST: f64 = 1e9;
pub(crate) const HIGH_FIDELITY_SPH_ORDER: usize = 5;
pub(crate) const HIGH_FIDELITY_FORCE_FLAGS: i32 = lightyear_odeint_rs::types::ForceFlags::DRAG
    | lightyear_odeint_rs::types::ForceFlags::SRP
    | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
    | lightyear_odeint_rs::types::ForceFlags::MOON_GRAVITY;
/// The exact-JB2008 code the crate's high-fidelity test fixtures fly.
///
/// Test-only since the target-propagation authority validator stopped naming
/// model codes and started asking `atm_model_uses_jb2008_drivers`. It is not a
/// production constant and pins nothing — the shipped code is whatever
/// `part_a_science` compiles in.
#[cfg(test)]
pub(crate) const HIGH_FIDELITY_ATM_MODEL: i32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferLocalOptimizerChoice {
    Auto,
    Fixed(LocalOptimizerKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransferLocalOptimizerConfig {
    pub choice: TransferLocalOptimizerChoice,
    pub tune: TuneLevel,
    pub seed: u64,
}

impl Default for TransferLocalOptimizerConfig {
    fn default() -> Self {
        Self {
            choice: TransferLocalOptimizerChoice::Auto,
            tune: TuneLevel::Default,
            seed: 42,
        }
    }
}

/// Internal transfer sampling policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SamplingMode {
    #[default]
    Fast,
}

/// Internal execution/runtime policy for a planning request.
///
/// Leaf-stage flags share one adaptive rule: a top-level call may fan out on
/// the process-global rayon pool, while a call already running on a rayon worker
/// stays leaf-serial so outer/cross-cell parallelism owns the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each public execution flag independently gates a verified scheduling path"
)]
pub struct ExecutionPolicy {
    pub use_high_fidelity: bool,
    pub require_high_fidelity: bool,
    pub allow_parallel: bool,
    /// Gate for the `OxyMOO` transfer-front NSGA-II batch objective evaluation to
    /// fan out across the global rayon pool (7.4). Distinct from
    /// `allow_parallel`, which gates the *inner* per-decision TOF `par_iter`:
    /// that inner path must stay serial while the batch fans out. Default OFF;
    /// only the single-pair front-solve context opts in.
    pub allow_oxymoo_batch_parallel: bool,
    /// Gate for the verified-superset Lambert branch-expansion stage to fan its
    /// per-source branch evaluations out across the global rayon pool (7.4).
    /// Sibling of `allow_oxymoo_batch_parallel`: both are opted in only by the
    /// single-pair front-solve context, while the inner per-decision TOF
    /// `par_iter` stays serial via `allow_parallel = false`. Default OFF; the
    /// shared runtime gate additionally requires a top-level multi-thread caller.
    pub allow_branch_expansion_parallel: bool,
    /// Gate for the final delta-V polish stage to fan its per-candidate polishes
    /// out across the global rayon pool (7.4). Each polished candidate is a pure
    /// function of its pre-polish `PlanResult` and `ctx`; independent scratch
    /// owners are bit-identical to the serial polish loop. Like
    /// `allow_oxymoo_batch_parallel`, this is default OFF and
    /// only the single-pair front-solve context opts in; the shared runtime gate
    /// additionally requires a top-level multi-thread caller.
    pub allow_polish_parallel: bool,
    /// Gate for the delta-v anchor stage to run its independent Nelder-Mead
    /// anchor optimizations (1 cost anchor + up to N delta-v anchors) across a
    /// global rayon pool (7.4). Each anchor is a self-contained coarse+fine NM
    /// run from a precomputed start, and the stable result collection is
    /// bit-identical to the serial reference. Like
    /// `allow_oxymoo_batch_parallel`, this is distinct from `allow_parallel`
    /// (the inner per-decision TOF `par_iter`, which stays serial). Default OFF;
    /// only the single-pair front-solve context opts in.
    pub allow_anchor_parallel: bool,
    /// Gate for the verified-superset deterministic grid fallback to fan its
    /// per-grid-point Lambert branch evaluations out across the global rayon
    /// pool (7.4). The grid fallback is the last-resort TIME x PHASE x WAIT
    /// enumeration when the front comes back empty; each point evaluates
    /// independently against a per-worker Lambert scratch, and the pushes/dedup
    /// are replayed in serial grid-index order, so the fan-out is bit-identical
    /// to the serial loop.
    /// Sibling of `allow_branch_expansion_parallel`; like it, this is default
    /// OFF and only the single-pair front-solve context opts in; the shared
    /// runtime gate additionally requires a top-level multi-thread caller.
    pub allow_deterministic_grid_parallel: bool,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            use_high_fidelity: false,
            require_high_fidelity: false,
            allow_parallel: true,
            allow_oxymoo_batch_parallel: false,
            allow_branch_expansion_parallel: false,
            allow_polish_parallel: false,
            allow_anchor_parallel: false,
            allow_deterministic_grid_parallel: false,
        }
    }
}

/// Physical object being propagated on a timeline arc.
///
/// This tag prevents a transfer/deployer arc from silently inheriting dust
/// aerodynamic or radiation properties.  It is intentionally small so it can
/// travel with hot-path boundary states without heap allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyRole {
    TransferVehicle,
    Canister,
    Dust,
    DiagnosticTarget,
}

/// Propagation authority used for one stamped segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropagationFidelity {
    J2,
    HighFidelity,
}

/// Authority used when the transfer solver propagates catalogue target states
/// from E0 to candidate intercept epochs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TargetPropagationAuthority {
    HighFidelity,
    /// Mean-fidelity catalogue generated with secular J2 propagation.
    #[default]
    MfJ2,
    /// Analytical catalogue generated with unperturbed Kepler propagation.
    AnalyticalKepler,
}

/// Candidate search deliberately stops at MF propagation.
///
/// Strict-HF propagation belongs to the post-front replay/lowering boundary,
/// where numerical failure remains typed per candidate row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateSearchAuthorityError {
    UnsupportedHighFidelityTransferSearch,
}

impl std::fmt::Display for CandidateSearchAuthorityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedHighFidelityTransferSearch => formatter.write_str(
                "candidate transfer search is MF-only; use strict-HF replay or Hybrid lowering",
            ),
        }
    }
}

impl std::error::Error for CandidateSearchAuthorityError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvalidTargetPropagationAuthorityCode {
    ArithmeticOverflow,
    OptimizerFailure,
    InvalidCode(u8),
    MissingTargetBallistics,
    InvalidTargetBodyForce {
        authority: TargetPropagationAuthority,
    },
    MissingHighFidelityForceConfig,
    InvalidHighFidelitySphericalHarmonicsOrder {
        expected: usize,
        actual: usize,
    },
    InvalidHighFidelityForceFlags {
        expected: i32,
        actual: i32,
    },
    InvalidHighFidelityAtmosphereModel {
        actual: i32,
    },
    Mismatch {
        explicit: TargetPropagationAuthority,
        force_config: TargetPropagationAuthority,
    },
    CandidateSearch(CandidateSearchAuthorityError),
}

impl std::fmt::Display for InvalidTargetPropagationAuthorityCode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArithmeticOverflow => {
                formatter.write_str("transfer authority arithmetic overflow")
            }
            Self::OptimizerFailure => formatter.write_str("transfer optimizer failure"),
            Self::InvalidCode(code) => write!(
                formatter,
                "invalid target propagation authority code {code}; expected 0, 1, or 2"
            ),
            Self::MissingTargetBallistics => write!(
                formatter,
                "high-fidelity target propagation requires explicit target ballistics"
            ),
            Self::InvalidTargetBodyForce { authority } => write!(
                formatter,
                "target body-force config violates {authority:?} propagation authority"
            ),
            Self::MissingHighFidelityForceConfig => write!(
                formatter,
                "high-fidelity target propagation requires explicit 5x5 gravity, drag, SRP, Sun, and Moon force config"
            ),
            Self::InvalidHighFidelitySphericalHarmonicsOrder { expected, actual } => write!(
                formatter,
                "high-fidelity target propagation requires sph_order={expected}, got {actual}"
            ),
            Self::InvalidHighFidelityForceFlags { expected, actual } => write!(
                formatter,
                "high-fidelity target propagation requires force_flags={expected} (drag+SRP+Sun+Moon), got {actual}"
            ),
            Self::InvalidHighFidelityAtmosphereModel { actual } => write!(
                formatter,
                "high-fidelity target propagation requires a JB2008 atm_model, got {actual}"
            ),
            Self::Mismatch {
                explicit,
                force_config,
            } => write!(
                formatter,
                "target propagation authority mismatch: explicit={explicit:?}, force_config={force_config:?}"
            ),
            Self::CandidateSearch(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for InvalidTargetPropagationAuthorityCode {}

impl TryFrom<u8> for TargetPropagationAuthority {
    type Error = InvalidTargetPropagationAuthorityCode;

    #[inline]
    fn try_from(code: u8) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::HighFidelity),
            1 => Ok(Self::MfJ2),
            2 => Ok(Self::AnalyticalKepler),
            _ => Err(InvalidTargetPropagationAuthorityCode::InvalidCode(code)),
        }
    }
}

impl TargetPropagationAuthority {
    #[inline]
    #[must_use]
    pub const fn as_force_config_code(self) -> u8 {
        match self {
            Self::HighFidelity => 0,
            Self::MfJ2 => 1,
            Self::AnalyticalKepler => 2,
        }
    }
}

/// Reject strict-HF selectors before candidate-search work begins.
///
/// Candidate search is MF-only.  A supplied force configuration is rejected
/// before validating its internals, allocating search state, or constructing
/// an RHS; strict-HF uses checked post-front replay/lowering instead.
#[inline]
pub(crate) fn validate_candidate_search_authority(
    authority: TargetPropagationAuthority,
    force_config: Option<&lightyear_odeint_rs::types::ForceConfig>,
    high_fidelity_requested: bool,
) -> Result<(), CandidateSearchAuthorityError> {
    if authority == TargetPropagationAuthority::HighFidelity
        || force_config.is_some()
        || high_fidelity_requested
    {
        return Err(CandidateSearchAuthorityError::UnsupportedHighFidelityTransferSearch);
    }
    Ok(())
}

/// Validate shared force-model authority before any Rust transfer solve starts.
///
/// Hybrid catalogue targets use exactly 5x5 Earth gravity, drag, SRP, Sun,
/// and Moon. Extra force bits are rejected rather than inherited by target propagation.
#[inline]
pub(crate) fn validate_target_propagation_force_config(
    authority: TargetPropagationAuthority,
    force_config: Option<&lightyear_odeint_rs::types::ForceConfig>,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let Some(config) = force_config else {
        return if authority == TargetPropagationAuthority::HighFidelity {
            Err(InvalidTargetPropagationAuthorityCode::MissingHighFidelityForceConfig)
        } else {
            Ok(())
        };
    };
    let config_authority = TargetPropagationAuthority::try_from(config.target_propagation_mode)?;
    if config_authority != authority {
        return Err(InvalidTargetPropagationAuthorityCode::Mismatch {
            explicit: authority,
            force_config: config_authority,
        });
    }
    if authority != TargetPropagationAuthority::HighFidelity {
        return Ok(());
    }

    if config.sph_order != HIGH_FIDELITY_SPH_ORDER {
        return Err(
            InvalidTargetPropagationAuthorityCode::InvalidHighFidelitySphericalHarmonicsOrder {
                expected: HIGH_FIDELITY_SPH_ORDER,
                actual: config.sph_order,
            },
        );
    }
    if config.force_flags != HIGH_FIDELITY_FORCE_FLAGS {
        return Err(
            InvalidTargetPropagationAuthorityCode::InvalidHighFidelityForceFlags {
                expected: HIGH_FIDELITY_FORCE_FLAGS,
                actual: config.force_flags,
            },
        );
    }
    if !lightyear_odeint_rs::rhs::atm_model_uses_jb2008_drivers(config.atm_model) {
        return Err(
            InvalidTargetPropagationAuthorityCode::InvalidHighFidelityAtmosphereModel {
                actual: config.atm_model,
            },
        );
    }
    Ok(())
}

#[inline]
pub(crate) fn validate_target_body_force(
    authority: TargetPropagationAuthority,
    force: BodyForceConfig,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    let valid = match authority {
        TargetPropagationAuthority::HighFidelity => {
            force.role == BodyRole::DiagnosticTarget
                && force.fidelity == PropagationFidelity::HighFidelity
                && force.am_ratio.is_finite()
                && force.am_ratio > 0.0
                && force.cd.is_finite()
                && force.cd > 0.0
                && force.cr.is_finite()
                && force.cr >= 0.0
        }
        TargetPropagationAuthority::MfJ2 | TargetPropagationAuthority::AnalyticalKepler => {
            force.role == BodyRole::DiagnosticTarget
                && force.fidelity == PropagationFidelity::J2
                && force.am_ratio.to_bits() == 0.0_f64.to_bits()
                && force.cd.to_bits() == 0.0_f64.to_bits()
                && force.cr.to_bits() == 0.0_f64.to_bits()
        }
    };
    if valid {
        Ok(())
    } else {
        Err(InvalidTargetPropagationAuthorityCode::InvalidTargetBodyForce { authority })
    }
}

/// Validate scalar target authority before propagation or solve mutation.
#[inline]
pub(crate) fn validate_target_propagation_authority(
    authority: TargetPropagationAuthority,
    target_body_force: BodyForceConfig,
    force_config: Option<&lightyear_odeint_rs::types::ForceConfig>,
) -> Result<(), InvalidTargetPropagationAuthorityCode> {
    validate_target_body_force(authority, target_body_force)?;
    validate_target_propagation_force_config(authority, force_config)
}

/// Per-body perturbation coefficients for a propagation segment.
///
/// `gravity_only` removes body-dependent non-gravitational coefficients for
/// hostile tests and reference comparisons; it is not production transfer or
/// target authority. Production hybrid uses sealed `high_fidelity` tuples,
/// while MF uses analytical `j2` propagation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyForceConfig {
    pub role: BodyRole,
    pub fidelity: PropagationFidelity,
    pub am_ratio: f64,
    pub cd: f64,
    pub cr: f64,
}

impl BodyForceConfig {
    #[inline]
    #[must_use]
    pub fn matches_exact(self, other: Self) -> bool {
        self.role == other.role
            && self.fidelity == other.fidelity
            && self.am_ratio.to_bits() == other.am_ratio.to_bits()
            && self.cd.to_bits() == other.cd.to_bits()
            && self.cr.to_bits() == other.cr.to_bits()
    }

    #[inline]
    #[must_use]
    pub const fn gravity_only(role: BodyRole) -> Self {
        Self {
            role,
            fidelity: PropagationFidelity::HighFidelity,
            am_ratio: 0.0,
            cd: 0.0,
            cr: 0.0,
        }
    }

    #[inline]
    #[must_use]
    pub const fn high_fidelity(role: BodyRole, am_ratio: f64, cd: f64, cr: f64) -> Self {
        Self {
            role,
            fidelity: PropagationFidelity::HighFidelity,
            am_ratio,
            cd,
            cr,
        }
    }

    #[inline]
    #[must_use]
    pub const fn j2(role: BodyRole) -> Self {
        Self {
            role,
            fidelity: PropagationFidelity::J2,
            am_ratio: 0.0,
            cd: 0.0,
            cr: 0.0,
        }
    }
}

/// ECI state with its authoritative source epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StampedEciState {
    pub eci: [f64; 6],
    pub jd: f64,
}

impl StampedEciState {
    #[inline]
    #[must_use]
    pub const fn new(eci: [f64; 6], jd: f64) -> Self {
        Self { eci, jd }
    }
}

/// Pair-ranking dv proxy model used by the pre-verify screening stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PairProxyModel {
    /// Historical model: `hohmann + plane_change` summed (node-aware).
    #[default]
    Sum,
    /// Plane change folded into one burn via the cosine law; tighter
    /// ranking for combined-burn-cheap pairs.
    Combined,
}

/// `OxyMOO` NSGA-II runtime policy for the transfer-front stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OxyMooPolicy {
    /// Historical full policy: population 28, generations 5, all initial decisions.
    #[default]
    Full,
    FastPopulation20,
    FastPopulation16,
    FastGenerations3,
    FastGenerations2,
    FastPopulation20Generations3,
    FastInitialBest1,
    FastPopulation20Generations3InitialBest1,
}

/// Delta-v anchor polishing policy for the transfer-front stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeltaVAnchorPolicy {
    #[default]
    Full,
    NoProbes,
    CostOnlyNoProbes,
    DvOnlyNoProbes,
    SeedLimit2,
    SeedLimit3,
}

/// Final NM delta-v polish scope for the transfer-front stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PolishScopePolicy {
    /// Historical behavior: polish every unique-decision candidate.
    #[default]
    Full,
    /// Polish only candidates NOT epsilon-dominated in (cv, `total_dv`,
    /// `total_time`). A candidate is skipped iff another candidate beats it by
    /// more than the configured margins on BOTH objectives at no-worse cv.
    /// Empirically-gated trade (promoted on multi-seed HV evidence), NOT a
    /// safety proof: measured polish rescues far exceed the margins, so
    /// skipped candidates can lose front-relevant refinement.
    ///
    /// Since the 2026-08-13 `nd-epsilon-membership` reseal, the dv/time
    /// margins are also applied cross-pair at
    /// `finalize_constellation_transfer_superset`, which drops
    /// epsilon-dominated candidates there. The two predicates are NOT
    /// identical: this mask additionally requires the beating candidate to
    /// sit at no-worse cv (within `POLISH_SCOPE_CV_TOL`), while the
    /// constellation drop compares dv/time only — and the superset drop
    /// removes a mask-skipped candidate only if a candidate dominating it by
    /// those margins itself reaches that superset, which the mask does not
    /// itself guarantee. A degenerate-front safety net
    /// still re-polishes the pre-polish snapshot when a front finalizes
    /// empty/single-row with skips. Evidence:
    /// `docs/evidence/front-lane-20260813/divergence-gate.md`.
    NdEpsilon,
    /// Campaign-tunable variant of `NdEpsilon`: identical mask rule with the
    /// delta-v margin overridden to `dv_eps_m_per_s / 1000` km/s (the time
    /// fraction and cv tolerance keep the `NdEpsilon` constants). Stored as
    /// integer m/s so the enum stays `Copy + Eq` and the config token is
    /// exact; `nd_epsilon_dv_mps50` reproduces `NdEpsilon` bit-identically
    /// (50/1000 rounds to the same f64 as the 0.05 literal).
    ///
    /// RE-TUNE CAMPAIGN PLAN (design item d — do NOT flip any default):
    /// 1. Run a production-config campaign with deep transfer telemetry ON
    ///    and collect the distribution of
    ///    `metrics.polish_dv_improvement_max_km_s` (solve.rs, accumulated as
    ///    a *_max columnar transfer-stage stat) across events.
    /// 2. Pick a LOW quantile (p10-p25) of that distribution as the tuned
    ///    margin and express it as this token (e.g. p10=0.075 km/s ->
    ///    `nd_epsilon_dv_mps75`).
    /// 3. Gate promotion on a mandatory HV A/B: nsga2 x >=8 seeds, comparing
    ///    HV + feasible count + front-identity delta vs `nd_epsilon`;
    ///    promote ONLY on HV non-regression + advisor sign-off. Treat the
    ///    tuned constant as code+data (record the campaign artifacts beside
    ///    the config change). Result-changing: a wider margin changes which
    ///    candidates get polished and therefore front content.
    NdEpsilonTuned { dv_eps_m_per_s: u32 },
}

/// Bounds for the tuned `nd_epsilon_dv_mps<N>` token (m/s).
///
/// The lower bound
/// rejects a zero margin (degenerates to strict dominance on dv); the upper
/// bound rejects margins beyond any plausible polish rescue (5 km/s).
pub(crate) const POLISH_SCOPE_TUNED_DV_MPS_MIN: u32 = 1;
pub(crate) const POLISH_SCOPE_TUNED_DV_MPS_MAX: u32 = 5000;

impl PolishScopePolicy {
    #[inline]
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        let normalized = token.trim().to_ascii_lowercase().replace('-', "_");
        match normalized.as_str() {
            "full" | "default" => Some(Self::Full),
            "nd_epsilon" | "nd_eps" => Some(Self::NdEpsilon),
            other => {
                let digits = other.strip_prefix("nd_epsilon_dv_mps")?;
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                let dv_eps_m_per_s: u32 = digits.parse().ok()?;
                if !(POLISH_SCOPE_TUNED_DV_MPS_MIN..=POLISH_SCOPE_TUNED_DV_MPS_MAX)
                    .contains(&dv_eps_m_per_s)
                {
                    return None;
                }
                Some(Self::NdEpsilonTuned { dv_eps_m_per_s })
            }
        }
    }
}

/// Runtime-configurable transfer search-depth knobs.
/// Default values reproduce the historical hard-coded constants exactly,
/// so an unset policy is bit-identical to builds that predate it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchDepthPolicy {
    /// Max TOF samples generated per plan evaluation (clamped to `1..=MAX_TOF_SAMPLES`).
    pub tof_sample_budget: usize,
    /// Allow the coarse seed scan to stop early once a strong candidate is found.
    pub coarse_early_stop: bool,
    /// Cap on fine-stage seed count (also the coarse-to-fine cutoff).
    pub fine_total_limit: usize,
    /// Coarse candidates worse than best+margin stay out of the fine stage (km/s).
    pub coarse_reject_margin_km_s: f64,
    /// Seeds within this cost margin of the fine cutoff are still admitted (km/s).
    pub seed_fine_margin_km_s: f64,
    /// Pair-ranking dv proxy model for the pre-verify screening stage.
    pub pair_proxy_model: PairProxyModel,
    /// Opt-in `OxyMOO` stage policy; default preserves the historical full stage.
    pub oxymoo_policy: OxyMooPolicy,
    /// Opt-in anchor-polish policy; default preserves the historical full stage.
    pub delta_v_anchor_policy: DeltaVAnchorPolicy,
    /// Opt-in final-polish scope policy; default polishes every candidate.
    pub polish_scope_policy: PolishScopePolicy,
}

impl Default for SearchDepthPolicy {
    fn default() -> Self {
        Self {
            tof_sample_budget: 64,
            coarse_early_stop: true,
            fine_total_limit: 10,
            coarse_reject_margin_km_s: 0.05,
            seed_fine_margin_km_s: 0.05,
            pair_proxy_model: PairProxyModel::Sum,
            oxymoo_policy: OxyMooPolicy::Full,
            delta_v_anchor_policy: DeltaVAnchorPolicy::Full,
            polish_scope_policy: PolishScopePolicy::Full,
        }
    }
}

impl SearchDepthPolicy {
    /// TOF budget bounded by the fixed array capacity.
    #[inline]
    #[must_use]
    pub fn clamped_tof_budget(&self) -> usize {
        self.tof_sample_budget.clamp(1, MAX_TOF_SAMPLES)
    }
}

/// Canonical non-derived inputs used to build a `PlanContext`.
#[derive(Clone, Debug)]
pub struct TransferRequest {
    pub dep_eci: [f64; 6],
    pub dep_equ: [f64; 6],
    pub epoch_jd: f64,
    pub tgt_eci: [f64; 6],
    pub tgt_equ: [f64; 6],
    pub max_time_s: f64,
    pub tof_penalty_weight: f64,
    pub revolution_cap: f64,
    pub max_phase_dv: f64,
    pub max_transfer_dv: f64,
    pub min_perigee: f64,
    pub max_apogee: f64,
    pub max_revs: i32,
    pub sampling_mode: SamplingMode,
    pub execution_policy: ExecutionPolicy,
    pub j2_closure_settings: crate::solve::J2ClosureSettings,
    pub search_depth: SearchDepthPolicy,
    pub distance_tol: f64,
    pub deployer_min_distance: f64,
    pub target_propagation_authority: TargetPropagationAuthority,
    pub target_body_force: BodyForceConfig,
    pub force_config: Option<std::sync::Arc<lightyear_odeint_rs::types::ForceConfig>>,
    pub packed_coeffs: Option<std::sync::Arc<satpy_core::PackedGravityCoeffs>>,
    pub polish_metrics: Option<Arc<OdeMetrics>>,
    pub local_optimizer: TransferLocalOptimizerConfig,
}

impl TransferRequest {
    #[must_use]
    pub fn with_j2_closure_settings(j2_closure_settings: crate::solve::J2ClosureSettings) -> Self {
        Self {
            dep_eci: [0.0; 6],
            dep_equ: [0.0; 6],
            epoch_jd: 0.0,
            tgt_eci: [0.0; 6],
            tgt_equ: [0.0; 6],
            max_time_s: 0.0,
            tof_penalty_weight: 0.1,
            revolution_cap: 2.0,
            max_phase_dv: 0.5,
            max_transfer_dv: 2.0,
            min_perigee: DEFAULT_MIN_PERIGEE,
            max_apogee: DEFAULT_MAX_APOGEE,
            max_revs: 1,
            sampling_mode: SamplingMode::Fast,
            execution_policy: ExecutionPolicy::default(),
            j2_closure_settings,
            search_depth: SearchDepthPolicy::default(),
            distance_tol: DISTANCE_TOL,
            deployer_min_distance: DEPLOYER_MIN_DISTANCE,
            target_propagation_authority: TargetPropagationAuthority::MfJ2,
            target_body_force: BodyForceConfig::j2(BodyRole::DiagnosticTarget),
            force_config: None,
            packed_coeffs: None,
            polish_metrics: None,
            local_optimizer: TransferLocalOptimizerConfig::default(),
        }
    }
}

/// Distance tolerance for intercept (km)
pub(crate) const DISTANCE_TOL: f64 = 0.025;

/// Minimum distance from deployer at intercept (km)
pub(crate) const DEPLOYER_MIN_DISTANCE: f64 = 0.12;

/// Minimum time of flight (seconds)
pub const MIN_TOF: f64 = 5.0 * 60.0;

/// Default minimum perigee radius (km)
pub const DEFAULT_MIN_PERIGEE: f64 = RE + 200.0;

/// Default maximum apogee radius (km)
pub const DEFAULT_MAX_APOGEE: f64 = RE + 35000.0;

// Cache configuration

// TOF sampling configuration
//
// There is deliberately no `TOF_SAMPLE_SEPARATION` here. One used to sit at
// this line with the value `0.005` and NO unit in its name or a doc comment,
// while the value the grid search actually applies is the private
// `TOF_SAMPLE_SEPARATION: f64 = 120.0` seconds in `evaluate.rs`. The private
// one shadows this module's within `evaluate.rs`, so the two never collided --
// but a reader who found the `pub` one next to `MAX_TOF_SAMPLES` below (which
// IS imported by `evaluate.rs`) learned a number wrong by a factor of 24000.
// It had no callers anywhere in the workspace. If a shared separation constant
// is ever wanted, give it a `_S` suffix and a single definition.
/// Fixed capacity of the TOF sample arrays.
///
/// The per-solve sample count is
/// bounded by `SearchDepthPolicy::tof_sample_budget` (default 64), so raising
/// this capacity cannot change results for unset policies.
pub const MAX_TOF_SAMPLES: usize = 256;

// ============================================================================
// LambertSolutionEx - Extended solution with target state
// ============================================================================

/// Extended Lambert solution with cost and metadata.
#[derive(Clone, Copy, Debug, Default)]
pub struct LambertSolutionEx {
    /// Cost metric (typically departure dV magnitude)
    pub cost: f64,
    /// Departure delta-V vector (km/s)
    pub dv: [f64; 3],
    /// Arrival delta-V vector (km/s)
    pub arrival_dv: [f64; 3],
    /// Target state at intercept (km, km/s)
    pub tgt_state: [f64; 6],
    /// Best revolution count
    pub best_M: i32,
    /// True if the selected Lambert branch used the low-path solution.
    pub low_path: bool,
    /// True if prograde transfer
    pub prograde: bool,
    /// True if solution is valid
    pub valid: bool,
}

impl LambertSolutionEx {
    /// Create a new invalid solution
    #[inline]
    #[must_use]
    pub fn invalid() -> Self {
        Self {
            cost: INVALID_COST,
            ..Default::default()
        }
    }

    /// Reset to invalid state
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::invalid();
    }
}

/// Optional Lambert branch constraint for branch-fanout evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LambertBranchSelection {
    pub rev: i32,
    pub low_path: bool,
}

// ============================================================================
// EciBasicOrbit - minimal orbit summary
// ============================================================================

/// Basic orbital parameters computed from ECI state.
#[derive(Clone, Copy, Debug, Default)]
pub struct EciBasicOrbit {
    /// Semi-major axis (km)
    pub sma: f64,
    /// Eccentricity
    pub ecc: f64,
    /// Perigee radius (km)
    pub perigee: f64,
    /// Apogee radius (km)
    pub apogee: f64,
    /// Position magnitude (km)
    pub r_mag: f64,
    /// Velocity magnitude (km/s)
    pub v_mag: f64,
}

impl EciBasicOrbit {
    /// Compute from ECI state vector
    #[must_use]
    pub fn from_eci(state: &[f64; 6]) -> Option<Self> {
        let r = [state[0], state[1], state[2]];
        let v = [state[3], state[4], state[5]];

        let r_mag = satpy_core::norm3(&r);
        let v_mag = satpy_core::norm3(&v);

        if r_mag <= 0.0 || !r_mag.is_finite() {
            return None;
        }

        let v2 = v_mag * v_mag;
        let energy = v2 / 2.0 - MU / r_mag;

        if energy >= -1e-10 {
            // Parabolic or hyperbolic - not supported
            return None;
        }

        let sma = -MU / (2.0 * energy);
        if sma <= 0.0 || !sma.is_finite() {
            return None;
        }

        // Angular momentum
        let h = satpy_core::cross3(&r, &v);
        let h_mag = satpy_core::norm3(&h);

        // Eccentricity from p = h^2/mu = a(1-e^2)
        let p = h_mag * h_mag / MU;
        let ecc_sq = 1.0 - p / sma;
        let ecc = if ecc_sq > 0.0 { ecc_sq.sqrt() } else { 0.0 };

        let perigee = sma * (1.0 - ecc);
        let apogee = sma * (1.0 + ecc);

        Some(Self {
            sma,
            ecc,
            perigee,
            apogee,
            r_mag,
            v_mag,
        })
    }
}

// ============================================================================
// WarmStartData - for intra-plane optimization warm-starting
// ============================================================================

/// Warm-start data for optimizer from previous satellite in same plane
#[derive(Clone, Copy, Debug)]
pub struct WarmStartData {
    /// Previous solution: [`time2phase_ratio`, `phase_sma_ratio`, `waittime_ratio`]
    pub x: [f64; 3],
    /// Cost of the previous solution
    pub cost: f64,
    /// Whether warm-start data is valid
    pub valid: bool,
    /// Satellite/deployer index this warm start belongs to, or -1 if unknown.
    pub sat_index: i32,
    /// Target index this warm start belongs to, or -1 if unknown.
    pub target_index: i32,
}

impl Default for WarmStartData {
    fn default() -> Self {
        Self {
            x: [0.0; 3],
            cost: 0.0,
            valid: false,
            sat_index: -1,
            target_index: -1,
        }
    }
}

// ============================================================================
// PlanResult
// ============================================================================

/// Compact transfer-branch status token.
///
/// Hot solver paths create thousands of `PlanResult`s; storing the stable
/// accepted/rejected status as a byte avoids repeated string allocation. `PyO3`
/// and compact payload builders stringify this only at public boundaries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum BranchStatusToken {
    Accepted = 1,
    #[default]
    Rejected = 2,
}

impl BranchStatusToken {
    /// Stable compact code used at numeric telemetry boundaries.
    #[inline]
    #[must_use]
    pub const fn as_code(self) -> u8 {
        match self {
            Self::Accepted => 1,
            Self::Rejected => 2,
        }
    }

    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
        }
    }
}

/// Compact transfer-branch rejection token.
///
/// Rejection details are currently absent from accepted public payloads; keep a
/// code here so future stable reasons can be added without returning to hot-path
/// strings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum BranchRejectionToken {
    #[default]
    None = 0,
    UnsupportedHighFidelityCandidateSearch = 1,
}

impl BranchRejectionToken {
    /// Stable compact code used at numeric telemetry boundaries.
    #[inline]
    #[must_use]
    pub const fn as_code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::UnsupportedHighFidelityCandidateSearch => 1,
        }
    }

    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "",
            Self::UnsupportedHighFidelityCandidateSearch => {
                "unsupported_high_fidelity_candidate_search"
            }
        }
    }
}

/// Compact timing-failure token for invalid timing exits.
///
/// Completes the Backlog-38 token migration (rank-51): `PlanResult` used to
/// carry the stable timing-failure reason as a heap-owning `String`, paying a
/// `.to_string()` per rejected plan on the sampled-decision hot path and a
/// `String` clone at every `PlanResult::clone()` site. The value set is a
/// closed enumeration, so the plan carries this Copy token instead and
/// stringifies only at public boundaries, where `as_str()` renders the
/// byte-identical historical literals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum TimingFailureToken {
    /// No timing failure (rendered as the empty string, as historically).
    #[default]
    None = 0,
    /// Phase+wait pre-sum exceeded the intercept time budget.
    InterceptTransferTimeExceeded = 1,
    /// Remaining transfer headroom was below the minimum time of flight.
    InterceptInsufficientLead = 2,
    /// Phase burn delta-V exceeded the configured bound.
    PhaseDvBoundExceeded = 3,
    /// Transfer time of flight exceeded the deployer revolution cap.
    TransferRevolutionCapExceeded = 4,
}

impl TimingFailureToken {
    /// Stable compact code used at numeric telemetry boundaries.
    #[inline]
    #[must_use]
    pub const fn as_code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::InterceptTransferTimeExceeded => 1,
            Self::InterceptInsufficientLead => 2,
            Self::PhaseDvBoundExceeded => 3,
            Self::TransferRevolutionCapExceeded => 4,
        }
    }

    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "",
            Self::InterceptTransferTimeExceeded => "solver_intercept_transfer_time_exceeded",
            Self::InterceptInsufficientLead => "solver_intercept_insufficient_lead",
            Self::PhaseDvBoundExceeded => "phase_dv_bound_exceeded",
            Self::TransferRevolutionCapExceeded => "solver_transfer_revolution_cap_exceeded",
        }
    }
}

/// Complete result of a transfer plan evaluation.
#[derive(Clone, Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "public result retains independent status flags for authority and replay"
)]
pub struct PlanResult {
    /// Total cost metric (dV + penalties)
    pub cost: f64,
    /// True if solution is valid
    pub valid: bool,

    // High-Fidelity Polish Metrics
    /// Polish optimization steps
    pub polish_steps: u64,
    /// Polish function evaluations
    pub polish_evals: u64,
    /// Polish execution time (microseconds)
    pub polish_time_us: u64,
    /// True if Polish was skipped (cost below threshold or no HF config)
    pub polish_skipped: bool,
    /// True if escape hatch was triggered (retry with full bounds)
    pub escape_triggered: bool,
    // Optimization parameters (input ratios)
    /// Phase time as fraction of `max_time_s`
    pub time2phase_ratio: f64,
    /// Phase orbit SMA as ratio of deployer SMA
    pub phase_sma_ratio: f64,
    /// Wait time as fraction of `max_time_s`
    pub waittime_ratio: f64,

    // Timing (seconds)
    /// Time to phase burn
    pub time2phase: f64,
    /// Coasting time after phase burn
    pub waittime: f64,
    /// Transfer time of flight
    pub tof: f64,

    // Geometry (km)
    /// Intercept distance
    pub distance: f64,
    /// Distance from deployer at intercept
    pub deployer_distance: f64,
    /// Phase orbit semi-major axis
    pub phase_sma: f64,

    // Delta-V vectors (km/s)
    /// Phase burn delta-V [3]
    pub phase_dv: [f64; 3],
    /// Transfer burn delta-V [3]
    pub transfer_dv: [f64; 3],
    /// Arrival burn delta-V [3]
    pub arrival_dv: [f64; 3],

    // Delta-V magnitudes (km/s)
    /// Phase burn magnitude
    pub phase_dv_norm: f64,
    /// Transfer burn magnitude
    pub transfer_dv_norm: f64,
    /// Arrival burn magnitude
    pub arrival_dv_norm: f64,

    // States at key points (km, km/s)
    /// Payload state at intercept [6]
    pub payload_intercept_state: [f64; 6],
    /// Target state at intercept [6]
    pub target_intercept_state: [f64; 6],
    /// Deployer state at intercept [6]
    pub deployer_intercept_state: [f64; 6],
    /// Release state [6]
    pub release_state: [f64; 6],

    // Lambert solution info
    /// Number of complete revolutions
    pub best_M: i32,
    /// True if prograde transfer
    pub prograde: bool,
    /// Lambert branch revolution count for the selected transfer branch.
    pub branch_rev: i32,
    /// Lambert low-path branch flag for the selected transfer branch.
    pub branch_low_path: bool,
    /// Time of flight attached to the selected Lambert branch (seconds).
    pub branch_tof_s: f64,
    /// Departure delta-V magnitude attached to the selected branch (km/s).
    pub branch_departure_dv: f64,
    /// Arrival/rendezvous delta-V magnitude attached to the selected branch (km/s).
    pub branch_arrival_dv: f64,
    /// Total phase + transfer delta-V attached to the selected branch (km/s).
    pub branch_total_dv: f64,
    /// Native branch status token.
    pub branch_status: BranchStatusToken,
    /// Native branch rejection/failure token, empty when accepted.
    pub branch_rejection: BranchRejectionToken,

    // Julian Date timestamps (computed from epoch + time durations)
    /// Julian date of intercept
    pub intercept_jd: f64,
    /// Julian date of phase burn
    pub waittime_jd_start: f64,
    /// Julian date of transfer burn
    pub tof_jd_start: f64,
    /// Stable timing-failure token for invalid timing exits (Copy token in
    /// Rust, rendered as the historical literal at the `PyO3` boundary).
    pub timing_failure_reason: TimingFailureToken,
    /// Total number of function evaluations performed
    pub func_evals: u64,
    /// Optimizer-only function evaluations performed during the selected solve path.
    pub optimizer_func_evals: u64,
    /// Whether the selected optimizer run reported convergence.
    pub optimizer_converged: bool,
    /// Whether a provided warm-start seed was accepted and used.
    pub warm_start_used: bool,
    /// Deployer orbital period (seconds)
    pub dep_period: f64,
    /// Number of dissertation J2 Lambert correction iterations executed.
    pub j2_iteration_count: u32,
    /// Compatibility field: pre-HF dissertation J2 closure residual (meters).
    pub j2_endpoint_residual_m: f64,
    /// Final endpoint residual recomputed after HF transfer propagation (meters).
    pub post_hf_endpoint_residual_m: f64,
    /// Immutable E0 state and constraints required for exact replay.
    pub replay_provenance: ReplayProvenance,
}

/// Candidate-stamped source authority for replay.  Never infer these values
/// from a mutable session at replay time.
#[derive(Clone, Copy, Debug)]
pub struct ReplayProvenance {
    pub launch_pre_impulse_state: [f64; 6],
    pub base_epoch_jd: f64,
    pub max_time_s: f64,
    pub max_phase_dv: f64,
    pub max_transfer_dv: f64,
    pub revolution_cap: f64,
    pub min_perigee: f64,
    pub max_apogee: f64,
    pub distance_tol: f64,
    pub deployer_min_distance: f64,
    pub max_revs: i32,
    /// Catalogue target propagation authority used by the source solve:
    /// 0=legacy HF, 1=MF/J2, 2=analytical/Kepler.
    pub target_propagation_mode: u8,
    /// Exact selected-target area-to-mass ratio used by source solve.
    pub target_am_ratio: f64,
    /// Exact selected-target drag coefficient used by source solve.
    pub target_cd: f64,
    /// Exact selected-target radiation-pressure coefficient used by source solve.
    pub target_cr: f64,
}

impl Default for ReplayProvenance {
    fn default() -> Self {
        Self {
            launch_pre_impulse_state: [f64::NAN; 6],
            base_epoch_jd: f64::NAN,
            max_time_s: f64::NAN,
            max_phase_dv: f64::NAN,
            max_transfer_dv: f64::NAN,
            revolution_cap: f64::NAN,
            min_perigee: f64::NAN,
            max_apogee: f64::NAN,
            distance_tol: f64::NAN,
            deployer_min_distance: f64::NAN,
            max_revs: -1,
            target_propagation_mode: u8::MAX,
            target_am_ratio: f64::NAN,
            target_cd: f64::NAN,
            target_cr: f64::NAN,
        }
    }
}

impl Default for PlanResult {
    fn default() -> Self {
        Self::invalid()
    }
}

impl PlanResult {
    /// Create a new invalid result
    #[must_use]
    pub fn invalid() -> Self {
        Self {
            cost: INVALID_COST,
            valid: false,
            time2phase_ratio: 0.0,
            phase_sma_ratio: 0.0,
            waittime_ratio: 0.0,
            time2phase: 0.0,
            waittime: 0.0,
            tof: 0.0,
            distance: INVALID_COST,
            deployer_distance: INVALID_COST,
            phase_sma: f64::NAN,
            phase_dv: [0.0; 3],
            transfer_dv: [0.0; 3],
            arrival_dv: [0.0; 3],
            phase_dv_norm: f64::NAN,
            transfer_dv_norm: f64::NAN,
            arrival_dv_norm: f64::NAN,
            payload_intercept_state: [0.0; 6],
            target_intercept_state: [0.0; 6],
            deployer_intercept_state: [0.0; 6],
            polish_steps: 0,
            polish_evals: 0,
            polish_time_us: 0,
            polish_skipped: false,
            escape_triggered: false,
            release_state: [0.0; 6],
            best_M: 0,
            prograde: true,
            branch_rev: 0,
            branch_low_path: true,
            branch_tof_s: 0.0,
            branch_departure_dv: f64::NAN,
            branch_arrival_dv: f64::NAN,
            branch_total_dv: f64::NAN,
            branch_status: BranchStatusToken::Rejected,
            branch_rejection: BranchRejectionToken::None,
            intercept_jd: 0.0,
            waittime_jd_start: 0.0,
            tof_jd_start: 0.0,
            timing_failure_reason: TimingFailureToken::None,
            func_evals: 0,
            optimizer_func_evals: 0,
            optimizer_converged: false,
            warm_start_used: false,
            dep_period: 0.0,
            j2_iteration_count: 0,
            j2_endpoint_residual_m: f64::NAN,
            post_hf_endpoint_residual_m: f64::NAN,
            replay_provenance: ReplayProvenance::default(),
        }
    }

    /// Total delta-V (phase + transfer)
    #[inline]
    #[must_use]
    pub fn total_dv(&self) -> f64 {
        self.phase_dv_norm + self.transfer_dv_norm
    }

    /// Total time (time2phase + waittime + tof)
    #[inline]
    #[must_use]
    pub fn total_time(&self) -> f64 {
        self.time2phase + self.waittime + self.tof
    }

    /// Relative payload-target speed at intercept (km/s).
    #[inline]
    #[must_use]
    pub fn relative_velocity(&self) -> f64 {
        norm3(&[
            self.payload_intercept_state[3] - self.target_intercept_state[3],
            self.payload_intercept_state[4] - self.target_intercept_state[4],
            self.payload_intercept_state[5] - self.target_intercept_state[5],
        ])
    }

    /// Combined transfer time / relative velocity objective (s per km/s).
    #[inline]
    #[must_use]
    pub fn time_per_relative_velocity_s_per_km_s(&self) -> f64 {
        let total_time = self.total_time();
        let relative_velocity = self.relative_velocity().abs();
        if total_time.is_finite() && relative_velocity.is_finite() && relative_velocity > 0.0 {
            total_time / relative_velocity
        } else {
            f64::NAN
        }
    }

    /// Two-objective transfer view: min dV, min time per intercept relative speed.
    #[inline]
    #[must_use]
    pub fn transfer_objectives(&self) -> TransferObjectives {
        TransferObjectives::from_plan(self)
    }

    #[inline]
    pub fn set_accepted_branch(
        &mut self,
        rev: i32,
        low_path: bool,
        tof_s: f64,
        departure_dv: f64,
        arrival_dv: f64,
    ) {
        self.branch_rev = rev;
        self.branch_low_path = low_path;
        self.branch_tof_s = tof_s;
        self.branch_departure_dv = departure_dv;
        self.branch_arrival_dv = arrival_dv;
        self.branch_total_dv = self.phase_dv_norm + departure_dv;
        self.branch_status = BranchStatusToken::Accepted;
        self.branch_rejection = BranchRejectionToken::None;
    }
}

/// Multi-objective transfer values.
///
/// The transfer front minimizes total delta-V and the combined time per
/// intercept relative speed objective. Raw total time and relative speed stay
/// exposed as physical diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransferObjectives {
    /// Minimize total phase + transfer delta-V (km/s).
    pub total_dv: f64,
    /// Physical time from t0 to intercept (seconds).
    pub total_time: f64,
    /// Physical payload-target relative speed at intercept (km/s).
    pub relative_velocity: f64,
    /// Minimize `total_time` / `relative_velocity` (s per km/s).
    pub time_per_relative_velocity_s_per_km_s: f64,
}

impl TransferObjectives {
    #[inline]
    #[must_use]
    pub fn from_plan(plan: &PlanResult) -> Self {
        Self {
            total_dv: plan.total_dv(),
            total_time: plan.total_time(),
            relative_velocity: plan.relative_velocity().abs(),
            time_per_relative_velocity_s_per_km_s: plan.time_per_relative_velocity_s_per_km_s(),
        }
    }

    #[inline]
    #[must_use]
    pub const fn as_minimization_array(&self) -> [f64; 2] {
        [self.total_dv, self.time_per_relative_velocity_s_per_km_s]
    }

    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.total_dv.is_finite()
            && self.total_time.is_finite()
            && self.relative_velocity.is_finite()
            && self.relative_velocity.abs() > 0.0
            && self.time_per_relative_velocity_s_per_km_s.is_finite()
    }
}

counter_roster! {
    error = InvalidTargetPropagationAuthorityCode;
    overflow = InvalidTargetPropagationAuthorityCode::ArithmeticOverflow;
    sub = none;
    /// Verified-superset native stage timings and row counts.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct VerifiedSupersetStageMetrics {
        f64_sum batch_event_total_s: f64,
        f64_sum batch_event_prep_s: f64,
        f64_sum batch_satellite_propagation_s: f64,
        f64_sum batch_constellation_solve_s: f64,
        f64_sum pair_screen_s: f64,
        f64_sum selected_pair_solve_s: f64,
        f64_sum selected_pair_context_apply_s: f64,
        f64_sum selected_pair_front_solve_s: f64,
        f64_sum selected_pair_result_append_s: f64,
        f64_sum selected_pair_residual_s: f64,
        f64_sum constellation_finalize_s: f64,
        f64_sum prepare_single_pair_context_s: f64,
        f64_sum seed_rank_s: f64,
        f64_sum seed_build_s: f64,
        f64_sum seed_coarse_eval_s: f64,
        f64_sum seed_fine_eval_s: f64,
        f64_sum seed_sort_select_s: f64,
        f64_sum delta_v_anchor_s: f64,
        f64_sum oxymoo_s: f64,
        f64_sum nsga_run_s: f64,
        f64_sum nsga_materialize_s: f64,
        f64_sum polish_s: f64,
        count polish_candidate_count: usize,
        count polish_scope_skipped_count: usize,
        max polish_dv_improvement_max_km_s: f64,
        count polish_scope_fallback_count: usize,
        f64_sum polish_scope_fallback_s: f64,
        f64_sum branch_expand_s: f64,
        f64_sum branch_eval_s: f64,
        f64_sum finalize_s: f64,
        f64_sum verification_s: f64,
        f64_sum deterministic_fallback_s: f64,
        count pair_proxy_candidate_count: usize,
        count selected_pair_count: usize,
        bool_or pair_proxy_exact_mode: bool,
        count selected_pair_target0_count: usize,
        count selected_pair_target1_count: usize,
        count selected_pair_serial_event_count: usize,
        count selected_pair_parallel_event_count: usize,
        count outer_batch_parallel_event_count: usize,
        count deterministic_fallback_count: usize,
        count pre_oxymoo_candidate_count: usize,
        count post_oxymoo_candidate_count: usize,
        count post_branch_candidate_count: usize,
        count post_finalize_candidate_count: usize,
        count warm_start_received_count: usize,
        count warm_start_pair_match_count: usize,
        count warm_start_seed_consumed_count: usize,
        count warm_start_fine_seed_selected_count: usize,
        count warm_start_oxymoo_initial_count: usize,
        count nsga_materialize_plan_cache_hit_count: usize,
        count nsga_materialize_plan_cache_miss_count: usize,
        count nsga_materialize_all_exact_count: usize,
        count nsga_materialize_recompute_count: usize,
        count lambert_batch_call_count: usize,
        count lambert_batch_row_count: usize,
        count lambert_scalar_tof_count: usize,
        count lambert_branch_attempt_count: usize,
        count lambert_branch_valid_count: usize,
        count lambert_branch_rev0_count: usize,
        count lambert_branch_rev_gt0_count: usize,
        count lambert_branch_low_path_count: usize,
        count lambert_branch_high_path_count: usize,
        count lambert_branch_prograde_count: usize,
        count lambert_branch_retrograde_count: usize,
        count lambert_max_revs_gt0_call_count: usize,
        count near_pi_plane_eval_count: usize,
        count lambert_branch_selection_call_count: usize,
        count target_j2_batch_state_count: usize,
        count target_j2_simd4_chunk_count: usize,
        count target_j2_scalar_state_count: usize,
        count j2_propagate_state_count: usize,
        /// Phase-state sub-cache lookups, split by side; see the same-named fields
        /// on `EvaluationDiagnosticCounters`. Their SUM is the lookup count and is
        /// pool-width invariant even where the split is not.
        count phase_state_cache_hit_count: usize,
        count phase_state_cache_miss_count: usize,
        /// J2 residual-gate evaluations, rejections, and the residual mean's two
        /// components; see the same-named fields on `EvaluationDiagnosticCounters`.
        count j2_correction_gate_eval_count: usize,
        count j2_correction_rejected_count: usize,
        f64_sum j2_correction_residual_m_sum: f64,
        count j2_correction_residual_finite_count: usize,
        f64_sum j2_correction_rejected_residual_m_sum: f64,
        /// Invocations of the J2 endpoint closure; see the same-named field on
        /// `EvaluationDiagnosticCounters`.
        count j2_correction_call_count: usize,
        count j2_correction_iteration_count: usize,
        count j2_correction_lambert_retry_count: usize,
        count branch_source_count: usize,
        count branch_shared_prepare_count: usize,
        count branch_eval_call_count: usize,
        count branch_emitted_count: usize,
        count branch_rejected_count: usize,
        count branch_target_propagation_call_count: usize,
        count branch_lambert_sampling_call_count: usize,
        count branch_brent_call_count: usize,
        count branch_brent_eval_request_count: usize,
        count branch_brent_cache_hit_count: usize,
        count branch_brent_cache_miss_count: usize,
        count branch_j2_correction_call_count: usize,
        f64_sum branch_shared_prepare_s: f64,
        f64_sum branch_phase_release_s: f64,
        f64_sum branch_target_propagation_s: f64,
        f64_sum branch_lambert_sampling_s: f64,
        f64_sum branch_brent_s: f64,
        f64_sum branch_j2_correction_s: f64,
        max branch_rows_per_source_p50: usize,
        max branch_rows_per_source_p95: usize,
        max branch_rows_per_source_max: usize,
        max selected_pair_front_solve_p50_s: f64,
        max selected_pair_front_solve_p95_s: f64,
        max selected_pair_front_solve_pair_max_s: f64,
        max rayon_current_num_threads: usize,
        count selected_pair_parallel_policy_enabled_count: usize,
        /// 7.3 work-count audit counters. Per-stage full plan-evaluation tallies,
        /// `OxyMOO` eval-cache hit/miss accounting, anchor optimizer/probe work, and
        /// serial/parallel batch counts. Parallel counters record top-level leaf
        /// fan-outs on the global rayon pool. They stay 0 when an outer rayon worker
        /// owns the call because adaptive dispatch makes nested leaf stages serial.
        count anchor_full_eval_count: usize,
        count oxymoo_full_eval_count: usize,
        count polish_full_eval_count: usize,
        count branch_full_eval_count: usize,
        count oxymoo_eval_cache_hit_count: usize,
        count oxymoo_eval_cache_miss_count: usize,
        count anchor_nm_run_count: usize,
        count anchor_nm_iteration_count: usize,
        count anchor_probe_eval_count: usize,
        count polish_scope_fallback_full_eval_count: usize,
        count deterministic_fallback_full_eval_count: usize,
        count oxymoo_parallel_batch_count: usize,
        count oxymoo_serial_batch_count: usize,
        count anchor_parallel_count: usize,
        count branch_parallel_count: usize,
        count polish_parallel_count: usize,
    }
}

impl VerifiedSupersetStageMetrics {
    /// Merge independent stage metrics without mutating on count overflow.
    ///
    /// Field-wise in roster declaration order: counts `checked_add`, second
    /// sums `+=`, percentile/max diagnostics `.max`, mode flags `|=`.
    /// Transactional: an overflow error leaves `self` unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidTargetPropagationAuthorityCode::ArithmeticOverflow`]
    /// when any additive count cannot be represented.
    #[inline]
    pub fn add_assign(&mut self, other: Self) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        let mut merged = *self;
        Self::roster_add(&mut merged, &other)?;
        *self = merged;
        Ok(())
    }
}

/// Nondominated transfer candidates for one satellite-debris pair.
#[derive(Clone, Debug, Default)]
pub struct TransferFront {
    /// Verified nondominated candidates, sorted by dV, then time, then relative speed.
    pub candidates: Vec<PlanResult>,
    /// Internal timing/count diagnostics for verified-superset solves.
    pub verified_superset_metrics: VerifiedSupersetStageMetrics,
}

impl TransferFront {
    #[inline]
    #[must_use]
    pub fn new(candidates: Vec<PlanResult>) -> Self {
        Self {
            candidates,
            verified_superset_metrics: VerifiedSupersetStageMetrics::default(),
        }
    }

    #[inline]
    #[must_use]
    pub const fn with_verified_superset_metrics(
        candidates: Vec<PlanResult>,
        verified_superset_metrics: VerifiedSupersetStageMetrics,
    ) -> Self {
        Self {
            candidates,
            verified_superset_metrics,
        }
    }

    #[inline]
    #[must_use]
    pub fn empty() -> Self {
        Self {
            candidates: Vec::new(),
            verified_superset_metrics: VerifiedSupersetStageMetrics::default(),
        }
    }

    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.candidates.len()
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

// ============================================================================
// PlanContext
// ============================================================================

/// Reusable immutable planning parameters shared across many satellite-target pairs.
#[derive(Clone, Debug)]
pub struct PlanContextTemplate {
    pub max_time_s: f64,
    pub tof_penalty_weight: f64,
    pub revolution_cap: f64,
    pub max_phase_dv: f64,
    pub max_transfer_dv: f64,
    pub min_perigee: f64,
    pub max_apogee: f64,
    pub max_revs: i32,
    pub sampling_mode: SamplingMode,
    pub execution_policy: ExecutionPolicy,
    pub j2_closure_settings: crate::solve::J2ClosureSettings,
    pub search_depth: SearchDepthPolicy,
    pub distance_tol: f64,
    pub deployer_min_distance: f64,
    pub target_propagation_authority: TargetPropagationAuthority,
    pub force_config: Option<std::sync::Arc<lightyear_odeint_rs::types::ForceConfig>>,
    pub packed_coeffs: Option<std::sync::Arc<satpy_core::PackedGravityCoeffs>>,
    pub local_optimizer: TransferLocalOptimizerConfig,
}

/// Pair-specific orbital inputs used to refresh a reusable `PlanContext`.
#[derive(Clone, Copy, Debug)]
pub struct PairPlanContextInputs {
    pub dep_eci: [f64; 6],
    pub dep_equ: [f64; 6],
    pub epoch_jd: f64,
    pub tgt_eci: [f64; 6],
    pub tgt_equ: [f64; 6],
    pub dep_sma: f64,
    pub dep_period: f64,
    pub dep_orbit_cached: EciBasicOrbit,
    pub dep_orbit_valid: bool,
    pub tgt_period_cached: f64,
    pub tgt_orbit_valid: bool,
    pub tgt_sma: f64,
    pub tgt_period: f64,
}

/// Optimization context with constraints and cached target orbit info.
///
/// # Future Refactoring: `TransferConstraints` Extraction
///
/// The constraint fields (time, delta-V, orbital, tolerances) could be extracted into
/// a reusable `TransferConstraints` struct to reduce parameter counts in function
/// signatures. Candidate fields for extraction (11 total):
///
/// - **Time constraints:** `max_time_s`, `tof_penalty_weight`, `revolution_cap`
/// - **Delta-V constraints:** `max_phase_dv`, `max_transfer_dv`
/// - **Orbital constraints:** `min_perigee`, `max_apogee`
/// - **Lambert config:** `max_revs`
/// - **Tolerances:** `distance_tol`, `deployer_min_distance`
///
/// This would enable passing constraints by reference and reduce the 20-34 parameter
/// functions that currently exist in solve.rs and lib.rs.
#[derive(Clone, Debug)]
pub struct PlanContext {
    // Deployer state
    /// Deployer ECI state [6]
    pub dep_eci: [f64; 6],
    /// Deployer equinoctial elements [6]
    pub dep_equ: [f64; 6],
    /// Epoch of states (Julian date)
    pub epoch_jd: f64,

    // Target state
    /// Target ECI state [6]
    pub tgt_eci: [f64; 6],
    /// Target equinoctial elements [6]
    pub tgt_equ: [f64; 6],

    // Time constraints
    /// Maximum solver horizon from base epoch to intercept (seconds)
    pub max_time_s: f64,
    /// TOF penalty weight (km/s per hour)
    pub tof_penalty_weight: f64,
    /// Hard cap on TOF in revolutions (reject if `final_tof` > `revolution_cap` * `dep_period`)
    pub revolution_cap: f64,

    // Delta-V constraints
    /// Maximum phase burn delta-V (km/s)
    pub max_phase_dv: f64,
    /// Maximum transfer burn delta-V (km/s)
    pub max_transfer_dv: f64,

    // Orbital constraints
    /// Minimum perigee radius (km)
    pub min_perigee: f64,
    /// Maximum apogee radius (km)
    pub max_apogee: f64,

    // Lambert configuration
    /// Maximum Lambert revolutions to consider
    pub max_revs: i32,
    /// Optional exact branch constraint used only for branch fanout.
    pub lambert_branch_selection: Option<LambertBranchSelection>,

    // Mode / execution policy
    /// Internal transfer sampling policy requested by the caller.
    pub sampling_mode: SamplingMode,
    /// Internal execution/runtime policy for the current solve.
    pub execution_policy: ExecutionPolicy,
    /// Immutable J2 closure authority captured before any Rayon dispatch.
    pub j2_closure_settings: crate::solve::J2ClosureSettings,
    /// Runtime search-depth knobs (defaults reproduce historical constants).
    pub search_depth: SearchDepthPolicy,

    // Tolerances
    /// Distance tolerance for successful rendezvous (km)
    pub distance_tol: f64,
    /// Minimum deployer separation distance (km)
    pub deployer_min_distance: f64,

    // Deployer orbit info
    /// Deployer semi-major axis (km)
    pub dep_sma: f64,
    /// Deployer orbital period (seconds)
    pub dep_period: f64,
    /// Cached deployer orbit parameters
    pub dep_orbit_cached: EciBasicOrbit,
    /// True if deployer orbit cache is valid
    pub dep_orbit_valid: bool,

    // Cached target orbit info
    /// Target orbital period (seconds)
    pub tgt_period_cached: f64,
    /// True if target orbit cache is valid
    pub tgt_orbit_valid: bool,
    /// Target semi-major axis (km) - convenience cache
    pub tgt_sma: f64,
    /// Target orbital period (seconds) - convenience cache
    pub tgt_period: f64,
    /// Cached plane angle between deployer and target (radians)
    pub plane_angle: f64,
    /// True if cached plane angle is valid
    pub plane_angle_valid: bool,

    // High-Fidelity Configuration
    /// Force model configuration (optional)
    pub force_config: Option<std::sync::Arc<lightyear_odeint_rs::types::ForceConfig>>,
    /// Independent catalogue model used for target E0-to-I propagation.
    pub target_propagation_authority: TargetPropagationAuthority,
    /// Explicit ballistic authority for selected catalogue target.
    pub target_body_force: BodyForceConfig,
    /// Immutable packed spherical-harmonic authority (optional for MF/J2).
    pub packed_coeffs: Option<std::sync::Arc<satpy_core::PackedGravityCoeffs>>,
    /// Accumulated ODE metrics from polish/high-fidelity propagation.
    /// Uses Arc for sharing; `OdeMetrics` uses atomic counters internally.
    pub polish_metrics: Option<Arc<OdeMetrics>>,
    pub local_optimizer: TransferLocalOptimizerConfig,
}

impl PlanContext {
    /// Exact transfer-body authority: MF is analytical J2; hybrid is 5x5 +
    /// drag + SRP using the sealed transfer tuple carried by `ForceConfig`.
    #[inline]
    #[must_use]
    pub fn transfer_body_force(&self) -> BodyForceConfig {
        if !self.execution_policy.use_high_fidelity {
            return BodyForceConfig::j2(BodyRole::TransferVehicle);
        }
        self.force_config.as_ref().map_or_else(
            || {
                BodyForceConfig::high_fidelity(
                    BodyRole::TransferVehicle,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                )
            },
            |config| {
                BodyForceConfig::high_fidelity(
                    BodyRole::TransferVehicle,
                    config.am_ratio,
                    config.cd,
                    config.cr,
                )
            },
        )
    }

    #[must_use]
    pub fn with_j2_closure_settings(j2_closure_settings: crate::solve::J2ClosureSettings) -> Self {
        Self {
            dep_eci: [0.0; 6],
            dep_equ: [0.0; 6],
            epoch_jd: 0.0,
            tgt_eci: [0.0; 6],
            tgt_equ: [0.0; 6],
            max_time_s: 0.0,
            tof_penalty_weight: 0.1,
            revolution_cap: 2.0,
            max_phase_dv: 0.5,
            max_transfer_dv: 2.0,
            min_perigee: DEFAULT_MIN_PERIGEE,
            max_apogee: DEFAULT_MAX_APOGEE,
            max_revs: 1, // Default to 1 rev max (reduced from 2 for performance)
            lambert_branch_selection: None,
            sampling_mode: SamplingMode::Fast,
            execution_policy: ExecutionPolicy::default(),
            j2_closure_settings,
            search_depth: SearchDepthPolicy::default(),
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            dep_sma: 0.0,
            dep_period: 0.0,
            dep_orbit_cached: EciBasicOrbit::default(),
            dep_orbit_valid: false,
            tgt_period_cached: 0.0,
            tgt_orbit_valid: false,
            tgt_sma: 0.0,
            tgt_period: 0.0,
            plane_angle: 0.0,
            plane_angle_valid: false,
            force_config: None,
            target_propagation_authority: TargetPropagationAuthority::default(),
            target_body_force: BodyForceConfig::j2(BodyRole::DiagnosticTarget),
            packed_coeffs: None,
            polish_metrics: None,
            local_optimizer: TransferLocalOptimizerConfig::default(),
        }
    }
}

impl PlanContext {
    #[inline]
    #[must_use]
    pub const fn intercept_time_budget_s(&self) -> f64 {
        self.max_time_s
    }

    /// Build a `PlanContext` from explicit non-derived inputs and initialize caches.
    #[must_use]
    pub fn from_request(request: TransferRequest) -> Self {
        let j2_closure_settings = request.j2_closure_settings;
        let mut ctx = Self {
            dep_eci: request.dep_eci,
            dep_equ: request.dep_equ,
            epoch_jd: request.epoch_jd,
            tgt_eci: request.tgt_eci,
            tgt_equ: request.tgt_equ,
            max_time_s: request.max_time_s,
            tof_penalty_weight: request.tof_penalty_weight,
            revolution_cap: request.revolution_cap,
            max_phase_dv: request.max_phase_dv,
            max_transfer_dv: request.max_transfer_dv,
            min_perigee: request.min_perigee,
            max_apogee: request.max_apogee,
            max_revs: request.max_revs,
            sampling_mode: request.sampling_mode,
            execution_policy: request.execution_policy,
            j2_closure_settings: request.j2_closure_settings,
            search_depth: request.search_depth,
            distance_tol: request.distance_tol,
            deployer_min_distance: request.deployer_min_distance,
            target_propagation_authority: request.target_propagation_authority,
            target_body_force: request.target_body_force,
            force_config: request.force_config,
            packed_coeffs: request.packed_coeffs,
            polish_metrics: request.polish_metrics,
            local_optimizer: request.local_optimizer,
            ..Self::with_j2_closure_settings(j2_closure_settings)
        };
        ctx.cache_target_orbit();
        ctx.cache_deployer_orbit();
        ctx.cache_plane_angle();
        ctx
    }

    /// Initialize target orbit cache from ECI state
    pub fn cache_target_orbit(&mut self) {
        if let Some(orbit) = EciBasicOrbit::from_eci(&self.tgt_eci) {
            self.tgt_orbit_valid = true;
            self.tgt_sma = orbit.sma;
            if orbit.sma > 0.0 {
                let sma_cubed = orbit.sma * orbit.sma * orbit.sma;
                self.tgt_period_cached = std::f64::consts::TAU * (sma_cubed / MU).sqrt();
                self.tgt_period = self.tgt_period_cached;
            }
        } else {
            self.tgt_orbit_valid = false;
            self.tgt_sma = 0.0;
            self.tgt_period = 0.0;
        }
    }

    /// Initialize deployer orbit cache from ECI state
    pub fn cache_deployer_orbit(&mut self) {
        if let Some(orbit) = EciBasicOrbit::from_eci(&self.dep_eci) {
            self.dep_orbit_cached = orbit;
            self.dep_orbit_valid = true;
            self.dep_sma = orbit.sma;
            if orbit.sma > 0.0 {
                let sma_cubed = orbit.sma * orbit.sma * orbit.sma;
                self.dep_period = std::f64::consts::TAU * (sma_cubed / MU).sqrt();
            } else {
                self.dep_period = 0.0;
            }
        } else {
            self.dep_orbit_cached = EciBasicOrbit::default();
            self.dep_orbit_valid = false;
            self.dep_sma = 0.0;
            self.dep_period = 0.0;
        }
    }

    /// Cache plane angle between deployer and target orbital planes.
    pub fn cache_plane_angle(&mut self) {
        let h_dep = cross3(
            &[self.dep_eci[0], self.dep_eci[1], self.dep_eci[2]],
            &[self.dep_eci[3], self.dep_eci[4], self.dep_eci[5]],
        );
        let h_tgt = cross3(
            &[self.tgt_eci[0], self.tgt_eci[1], self.tgt_eci[2]],
            &[self.tgt_eci[3], self.tgt_eci[4], self.tgt_eci[5]],
        );
        let h_dep_norm = norm3(&h_dep);
        let h_tgt_norm = norm3(&h_tgt);
        if h_dep_norm > 1e-10 && h_tgt_norm > 1e-10 {
            let cos_angle = (h_dep[0] * h_tgt[0] + h_dep[1] * h_tgt[1] + h_dep[2] * h_tgt[2])
                / (h_dep_norm * h_tgt_norm);
            self.plane_angle = cos_angle.clamp(-1.0, 1.0).acos();
            self.plane_angle_valid = self.plane_angle.is_finite();
        } else {
            self.plane_angle = 0.0;
            self.plane_angle_valid = false;
        }
    }

    /// Reset context for reuse with new satellite-target pair
    ///
    /// Preserves allocated buffers (caches, tables) but invalidates their contents.
    /// This allows context pooling without repeated allocations.
    pub const fn reset(
        &mut self,
        dep_eci: [f64; 6],
        dep_equ: [f64; 6],
        tgt_eci: [f64; 6],
        tgt_equ: [f64; 6],
        epoch_jd: f64,
    ) {
        // Update orbital states
        self.dep_eci = dep_eci;
        self.dep_equ = dep_equ;
        self.tgt_eci = tgt_eci;
        self.tgt_equ = tgt_equ;
        self.epoch_jd = epoch_jd;

        // Invalidate caches - they'll be recomputed on demand
        self.dep_orbit_valid = false;
        self.tgt_orbit_valid = false;
        self.plane_angle_valid = false;

        // Reset cached values
        self.dep_sma = 0.0;
        self.dep_period = 0.0;
        self.tgt_sma = 0.0;
        self.tgt_period = 0.0;
        self.tgt_period_cached = 0.0;
        self.plane_angle = 0.0;

        self.lambert_branch_selection = None;

        // Note: We preserve max_time_s, constraints, and mode flags
        // as these typically don't change between pooled contexts in the same optimization run
    }

    /// Refresh a reusable context from shared template settings plus pair-specific orbit data.
    ///
    /// # Errors
    ///
    /// Returns the authority code when the target-propagation template is invalid.
    pub fn apply_template_pair(
        &mut self,
        template: &PlanContextTemplate,
        inputs: &PairPlanContextInputs,
    ) -> Result<(), InvalidTargetPropagationAuthorityCode> {
        validate_target_propagation_force_config(
            template.target_propagation_authority,
            template.force_config.as_deref(),
        )?;
        let target_propagation_authority = template.target_propagation_authority;

        self.dep_eci = inputs.dep_eci;
        self.dep_equ = inputs.dep_equ;
        self.epoch_jd = inputs.epoch_jd;
        self.tgt_eci = inputs.tgt_eci;
        self.tgt_equ = inputs.tgt_equ;

        self.max_time_s = template.max_time_s;
        self.tof_penalty_weight = template.tof_penalty_weight;
        self.revolution_cap = template.revolution_cap;
        self.max_phase_dv = template.max_phase_dv;
        self.max_transfer_dv = template.max_transfer_dv;
        self.min_perigee = template.min_perigee;
        self.max_apogee = template.max_apogee;
        self.max_revs = template.max_revs;
        self.lambert_branch_selection = None;
        self.sampling_mode = template.sampling_mode;
        self.execution_policy = template.execution_policy;
        self.j2_closure_settings = template.j2_closure_settings;
        self.search_depth = template.search_depth;
        self.distance_tol = template.distance_tol;
        self.deployer_min_distance = template.deployer_min_distance;

        self.dep_sma = inputs.dep_sma;
        self.dep_period = inputs.dep_period;
        self.dep_orbit_cached = inputs.dep_orbit_cached;
        self.dep_orbit_valid = inputs.dep_orbit_valid;

        self.tgt_period_cached = inputs.tgt_period_cached;
        self.tgt_orbit_valid = inputs.tgt_orbit_valid;
        self.tgt_sma = inputs.tgt_sma;
        self.tgt_period = inputs.tgt_period;

        self.force_config.clone_from(&template.force_config);
        self.target_propagation_authority = target_propagation_authority;
        self.packed_coeffs.clone_from(&template.packed_coeffs);
        self.polish_metrics = None;
        self.local_optimizer = template.local_optimizer;

        self.cache_plane_angle();
        Ok(())
    }
}

/// One verified transfer design vector in a constellation-level Pareto front.
#[derive(Clone, Debug)]
pub struct ConstellationTransferCandidate {
    /// True if valid solution found
    pub valid: bool,
    /// Selected satellite index
    pub sat_index: i32,
    /// Selected target index (0 or 1)
    pub target_index: i32,
    /// Estimated objective before full optimization
    pub estimated_objective: f64,
    /// Initial optimization vector
    pub estimated_x: [f64; 3],
    /// Multi-objective values for the optimized plan
    pub objectives: TransferObjectives,
    /// Optimized plan result
    pub optimum: PlanResult,
}

impl Default for ConstellationTransferCandidate {
    fn default() -> Self {
        Self {
            valid: false,
            sat_index: -1,
            target_index: -1,
            estimated_objective: INVALID_COST,
            estimated_x: [0.0; 3],
            objectives: TransferObjectives::default(),
            optimum: PlanResult::invalid(),
        }
    }
}

impl ConstellationTransferCandidate {
    #[must_use]
    pub fn from_plan(
        sat_index: i32,
        target_index: i32,
        estimated_objective: f64,
        estimated_x: [f64; 3],
        plan: PlanResult,
    ) -> Option<Self> {
        if !(plan.valid && plan.cost < INVALID_COST) {
            return None;
        }
        let objectives = plan.transfer_objectives();
        if !objectives.is_finite() {
            return None;
        }
        Some(Self {
            valid: true,
            sat_index,
            target_index,
            estimated_objective,
            estimated_x,
            objectives,
            optimum: plan,
        })
    }
}

/// Nondominated transfer design vectors across all assessed deployer-object pairs.
#[derive(Clone, Debug, Default)]
pub struct ConstellationTransferFront {
    /// Verified nondominated constellation transfer candidates.
    pub candidates: Vec<ConstellationTransferCandidate>,
    /// Aggregated internal timing/count diagnostics for verified-superset solves.
    pub verified_superset_metrics: VerifiedSupersetStageMetrics,
}

impl ConstellationTransferFront {
    #[inline]
    #[must_use]
    pub fn new(candidates: Vec<ConstellationTransferCandidate>) -> Self {
        Self {
            candidates,
            verified_superset_metrics: VerifiedSupersetStageMetrics::default(),
        }
    }

    #[inline]
    #[must_use]
    pub const fn with_verified_superset_metrics(
        candidates: Vec<ConstellationTransferCandidate>,
        verified_superset_metrics: VerifiedSupersetStageMetrics,
    ) -> Self {
        Self {
            candidates,
            verified_superset_metrics,
        }
    }

    #[inline]
    #[must_use]
    pub fn empty() -> Self {
        Self {
            candidates: Vec::new(),
            verified_superset_metrics: VerifiedSupersetStageMetrics::default(),
        }
    }

    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.candidates.len()
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

/// Compact dissertation-oriented transfer payload for fused solve+correct paths.
#[derive(Clone, Debug)]
pub struct CompactTransferCandidate {
    /// True if valid solution found
    pub valid: bool,
    /// Selected satellite/deployer index.
    pub sat_index: i32,
    /// Selected target index (0 or 1)
    pub target_index: i32,
    /// Exact selected-target area-to-mass ratio stamped by source solve.
    pub target_am_ratio: f64,
    /// Exact selected-target drag coefficient stamped by source solve.
    pub target_cd: f64,
    /// Exact selected-target radiation-pressure coefficient stamped by source solve.
    pub target_cr: f64,
    /// Estimated objective before full optimization.
    pub estimated_objective: f64,
    /// Initial optimizer seed vector from front screening.
    pub estimated_x: [f64; 3],
    /// Optimized seed vector suitable for a later warm start.
    pub warm_start_x: [f64; 3],
    /// Optimized plan cost for warm-start ranking.
    pub warm_start_cost: f64,
    /// Whether warm-start data is finite and usable.
    pub warm_start_valid: bool,
    /// Total objective DV before any Python-side composition
    pub total_dv: f64,
    /// Phase burn magnitude
    pub phase_dv_norm: f64,
    /// Exact phase burn applied during the stored candidate trajectory.
    ///
    /// Replay consumes this vector directly; it must never reconstruct a
    /// different phase maneuver through an optimizer call.
    pub phase_dv: [f64; 3],
    /// Transfer burn magnitude
    pub transfer_dv_norm: f64,
    /// Transfer time of flight (seconds)
    pub transfer_tof_s: f64,
    /// Total time from t0 to intercept (seconds)
    pub total_time_s: f64,
    /// Authoritative E0 epoch for the stored candidate controls (JD).
    pub base_epoch_jd: f64,
    /// Time from E0 to the phase burn (seconds).
    pub time_to_phase_s: f64,
    /// Coast duration from phase burn to transfer burn (seconds).
    pub wait_time_s: f64,
    /// Payload-target relative speed at intercept (km/s)
    pub relative_velocity_km_s: f64,
    /// Transfer objective: `total_time_s` / `relative_velocity_km_s` (s per km/s)
    pub time_per_relative_velocity_s_per_km_s: f64,
    /// Solver intercept epoch (JD)
    pub solver_intercept_jd: f64,
    /// Transfer-burn epoch (JD)
    pub tof_jd_start: f64,
    /// Payload state at intercept [6]
    pub payload_intercept_state: [f64; 6],
    /// Target state at intercept [6]
    pub target_intercept_state: [f64; 6],
    /// State immediately before the stored transfer burn at L [6].
    ///
    /// This deliberately does not use "`release_state"`: L is a transfer
    /// maneuver boundary, while canister/dust release R is produced later by
    /// postprocess control.
    pub transfer_burn_pre_state: [f64; 6],
    /// Transfer burn delta-V [3]
    pub transfer_dv: [f64; 3],
    /// Lambert branch revolution count for the selected transfer branch.
    pub branch_rev: i32,
    /// Lambert low-path branch flag for the selected transfer branch.
    pub branch_low_path: bool,
    /// Time of flight attached to the selected Lambert branch (seconds).
    pub branch_tof_s: f64,
    /// Departure delta-V magnitude attached to the selected branch (km/s).
    pub branch_departure_dv: f64,
    /// Arrival/rendezvous delta-V magnitude attached to the selected branch (km/s).
    pub branch_arrival_dv: f64,
    /// Total phase + transfer delta-V attached to the selected branch (km/s).
    pub branch_total_dv: f64,
    /// Native branch status token (perf #15: Copy token in Rust; rendered as
    /// the historical "accepted"/"rejected" string at the `PyO3` boundary).
    pub branch_status: BranchStatusToken,
    /// Native branch rejection/failure token, empty-string when accepted.
    pub branch_rejection: BranchRejectionToken,
    /// Dissertation J2 correction iteration count
    pub j2_iteration_count: u32,
    /// Compatibility field: pre-HF dissertation J2 closure residual (meters)
    pub j2_endpoint_residual_m: f64,
    /// Final endpoint residual recomputed after HF transfer propagation (meters)
    pub post_hf_endpoint_residual_m: f64,
    /// Immutable original E0 launch state; replay rejects missing provenance.
    pub launch_pre_impulse_state: [f64; 6],
    /// Immutable original solver constraints for exact replay.
    pub replay_policy: ReplayProvenance,
}

impl Default for CompactTransferCandidate {
    fn default() -> Self {
        Self {
            valid: false,
            sat_index: -1,
            target_index: -1,
            target_am_ratio: f64::NAN,
            target_cd: f64::NAN,
            target_cr: f64::NAN,
            estimated_objective: INVALID_COST,
            estimated_x: [0.0; 3],
            warm_start_x: [0.0; 3],
            warm_start_cost: INVALID_COST,
            warm_start_valid: false,
            total_dv: INVALID_COST,
            phase_dv_norm: f64::NAN,
            phase_dv: [0.0; 3],
            transfer_dv_norm: f64::NAN,
            transfer_tof_s: 0.0,
            total_time_s: 0.0,
            base_epoch_jd: f64::NAN,
            time_to_phase_s: f64::NAN,
            wait_time_s: f64::NAN,
            relative_velocity_km_s: f64::NAN,
            time_per_relative_velocity_s_per_km_s: f64::NAN,
            solver_intercept_jd: f64::NAN,
            tof_jd_start: f64::NAN,
            payload_intercept_state: [0.0; 6],
            target_intercept_state: [0.0; 6],
            transfer_burn_pre_state: [0.0; 6],
            transfer_dv: [0.0; 3],
            branch_rev: 0,
            branch_low_path: true,
            branch_tof_s: 0.0,
            branch_departure_dv: f64::NAN,
            branch_arrival_dv: f64::NAN,
            branch_total_dv: f64::NAN,
            branch_status: BranchStatusToken::Rejected,
            branch_rejection: BranchRejectionToken::None,
            j2_iteration_count: 0,
            j2_endpoint_residual_m: f64::NAN,
            post_hf_endpoint_residual_m: f64::NAN,
            launch_pre_impulse_state: [f64::NAN; 6],
            replay_policy: ReplayProvenance::default(),
        }
    }
}

impl CompactTransferCandidate {
    #[must_use]
    pub fn from_constellation_candidate(
        candidate: &ConstellationTransferCandidate,
    ) -> Option<Self> {
        let mut out = Self::from_plan_with_sat(
            candidate.sat_index,
            candidate.target_index,
            &candidate.optimum,
        )?;
        out.estimated_objective = candidate.estimated_objective;
        out.estimated_x = candidate.estimated_x;
        Some(out)
    }

    #[must_use]
    pub fn from_plan(target_index: i32, plan: &PlanResult) -> Option<Self> {
        Self::from_plan_with_sat(-1, target_index, plan)
    }

    #[must_use]
    pub fn from_plan_with_sat(
        sat_index: i32,
        target_index: i32,
        plan: &PlanResult,
    ) -> Option<Self> {
        if !plan.valid {
            return None;
        }
        Some(Self {
            valid: true,
            sat_index,
            target_index,
            target_am_ratio: plan.replay_provenance.target_am_ratio,
            target_cd: plan.replay_provenance.target_cd,
            target_cr: plan.replay_provenance.target_cr,
            estimated_objective: plan.cost,
            estimated_x: [
                plan.time2phase_ratio,
                plan.phase_sma_ratio,
                plan.waittime_ratio,
            ],
            warm_start_x: [
                plan.time2phase_ratio,
                plan.phase_sma_ratio,
                plan.waittime_ratio,
            ],
            warm_start_cost: plan.cost,
            warm_start_valid: plan.valid
                && plan.cost.is_finite()
                && plan.cost < INVALID_COST
                && plan.time2phase_ratio.is_finite()
                && plan.phase_sma_ratio.is_finite()
                && plan.waittime_ratio.is_finite(),
            total_dv: plan.total_dv(),
            phase_dv_norm: plan.phase_dv_norm,
            phase_dv: plan.phase_dv,
            transfer_dv_norm: plan.transfer_dv_norm,
            transfer_tof_s: plan.tof,
            total_time_s: plan.total_time(),
            base_epoch_jd: plan.intercept_jd - plan.total_time() / SEC_PER_DAY,
            time_to_phase_s: plan.time2phase,
            wait_time_s: plan.waittime,
            relative_velocity_km_s: plan.relative_velocity().abs(),
            time_per_relative_velocity_s_per_km_s: plan.time_per_relative_velocity_s_per_km_s(),
            solver_intercept_jd: plan.intercept_jd,
            tof_jd_start: plan.tof_jd_start,
            payload_intercept_state: plan.payload_intercept_state,
            target_intercept_state: plan.target_intercept_state,
            transfer_burn_pre_state: plan.release_state,
            transfer_dv: plan.transfer_dv,
            branch_rev: plan.branch_rev,
            branch_low_path: plan.branch_low_path,
            branch_tof_s: plan.branch_tof_s,
            branch_departure_dv: plan.branch_departure_dv,
            branch_arrival_dv: plan.branch_arrival_dv,
            branch_total_dv: plan.branch_total_dv,
            branch_status: plan.branch_status,
            branch_rejection: plan.branch_rejection,
            j2_iteration_count: plan.j2_iteration_count,
            j2_endpoint_residual_m: plan.j2_endpoint_residual_m,
            post_hf_endpoint_residual_m: plan.post_hf_endpoint_residual_m,
            launch_pre_impulse_state: plan.replay_provenance.launch_pre_impulse_state,
            replay_policy: plan.replay_provenance,
        })
    }
}

#[cfg(test)]
thread_local! {
    static TPT_DEEP_TELEMETRY_TEST_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Test-only gate for retired candidate-search HF telemetry.
#[inline]
#[cfg(test)]
pub(crate) fn verified_superset_deep_telemetry_enabled() -> bool {
    TPT_DEEP_TELEMETRY_TEST_OVERRIDE
        .with(Cell::get)
        .unwrap_or(true)
}

#[cfg(test)]
pub(crate) fn with_verified_superset_deep_telemetry_for_test<T>(
    enabled: bool,
    run: impl FnOnce() -> T,
) -> T {
    TPT_DEEP_TELEMETRY_TEST_OVERRIDE.with(|override_value| {
        let previous = override_value.replace(Some(enabled));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run));
        override_value.set(previous);
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
}

// ============================================================================
// TransferComplexity - for adaptive PSO parameters
// ============================================================================

/// Transfer complexity classification for adaptive PSO parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransferComplexity {
    /// Same orbit, tiny delta-V < 0.03 km/s
    Trivial = 0,
    /// Coplanar, similar altitude, low ecc
    Easy = 1,
    /// Small plane change or moderate altitude diff
    Moderate = 2,
    /// Large plane change or altitude
    Hard = 3,
    /// Bi-elliptic territory, extreme ratios
    Extreme = 4,
}

impl TransferComplexity {
    /// Classify transfer geometry
    #[must_use]
    pub fn classify(altitude_diff: f64, plane_angle: f64, sma_ratio: f64, max_ecc: f64) -> Self {
        // Trivial: extremely small maneuvers
        if altitude_diff < 50.0 && plane_angle < 0.02 && max_ecc < 0.02 {
            return Self::Trivial;
        }

        // Easy: LEO-to-LEO, same plane or very small plane change
        if altitude_diff < 400.0 && plane_angle < 0.1 && max_ecc < 0.06 {
            return Self::Easy;
        }

        // Extreme: bi-elliptic territory or very large plane changes
        if !(0.1..=10.0).contains(&sma_ratio) || plane_angle > 1.0 {
            return Self::Extreme;
        }

        // Hard: significant plane change or large altitude diff
        // PHASE 3: Raised plane_angle threshold from 0.5 to 0.6 rad (~34°)
        // Many 25-34° plane changes have clean analytical solutions and don't need Heavy optimization
        if altitude_diff > 1000.0 || plane_angle > 0.6 || max_ecc > 0.3 {
            return Self::Hard;
        }

        // Moderate: everything else
        Self::Moderate
    }

    /// Classify transfer complexity from `PlanContext`.
    /// Computes geometric parameters from deployer and target states.
    #[must_use]
    pub fn classify_from_ctx(ctx: &PlanContext) -> Self {
        // Compute altitude difference
        let altitude_diff = (ctx.tgt_sma - ctx.dep_sma).abs();

        // Compute SMA ratio
        let sma_ratio = if ctx.dep_sma > 0.0 && ctx.tgt_sma > 0.0 {
            (ctx.tgt_sma / ctx.dep_sma).max(ctx.dep_sma / ctx.tgt_sma)
        } else {
            1.0
        };

        // Compute plane angle from angular momentum vectors
        let h_dep = satpy_core::cross3(
            &[ctx.dep_eci[0], ctx.dep_eci[1], ctx.dep_eci[2]],
            &[ctx.dep_eci[3], ctx.dep_eci[4], ctx.dep_eci[5]],
        );
        let h_tgt = satpy_core::cross3(
            &[ctx.tgt_eci[0], ctx.tgt_eci[1], ctx.tgt_eci[2]],
            &[ctx.tgt_eci[3], ctx.tgt_eci[4], ctx.tgt_eci[5]],
        );
        let h_dep_norm = satpy_core::norm3(&h_dep);
        let h_tgt_norm = satpy_core::norm3(&h_tgt);

        let plane_angle = if h_dep_norm > 1e-10 && h_tgt_norm > 1e-10 {
            let cos_angle = satpy_core::dot3(&h_dep, &h_tgt) / (h_dep_norm * h_tgt_norm);
            cos_angle.clamp(-1.0, 1.0).acos()
        } else {
            0.0
        };

        // Get max eccentricity from cached orbit data
        let dep_ecc = EciBasicOrbit::from_eci(&ctx.dep_eci).map_or(0.0, |orbit| orbit.ecc);
        let tgt_ecc = EciBasicOrbit::from_eci(&ctx.tgt_eci).map_or(0.0, |orbit| orbit.ecc);
        let max_ecc = dep_ecc.max(tgt_ecc);

        Self::classify(altitude_diff, plane_angle, sma_ratio, max_ecc)
    }
}

// ============================================================================
// PSO Configuration presets
// ============================================================================

/// PSO configuration preset for `solve_plan`
#[derive(Clone, Copy, Debug)]
pub struct PsoPreset {
    pub swarm_size: usize,
    pub max_iters: usize,
    pub stall_limit: usize,
    pub reinit_frac: f64,
    pub tol: f64,
}

impl PsoPreset {
    /// Get preset for given complexity
    ///
    /// ## Phase 3 Tuning (2026-01-11) - Aligned with C++ for Performance Parity
    /// Matched C++ `two_phase_transfer_native.hpp` PSO budgets:
    /// - Trivial: 500 → 96 evals (81% reduction) - same-orbit transfers
    /// - Easy: 1280 → 288 evals (78% reduction) - coplanar transfers
    /// - Moderate: 2000 → 576 evals (71% reduction) - moderate plane changes
    /// - Hard: 4480 → 1152 evals (74% reduction) - large plane/altitude changes
    /// - Extreme: 8000 → 2592 evals (68% reduction) - bi-elliptic territory
    ///
    /// Rationale: C++ achieves comparable solution quality with significantly fewer evaluations.
    /// This alignment eliminates the 1.5x performance gap caused by excess computational work.
    #[must_use]
    pub const fn for_complexity(complexity: TransferComplexity, thorough: bool) -> Self {
        if thorough {
            // Aggressive preset matching C++ parameters for better performance
            // C++ uses: swarm=36, iters=72, stall=26 (from pso.c)
            // This achieves ~85% C++ parity with 6.0 ms/event vs previous 100/200/50
            return Self {
                swarm_size: 36,
                max_iters: 72,
                stall_limit: 26,
                reinit_frac: 0.22, // Matched to C++ reinit_threshold=0.22
                tol: 1e-7,
            };
        }

        match complexity {
            // Trivial: 8×12 = 96 evals (C++ value)
            // Same-orbit, tiny dV - converges in <10 iters typically
            TransferComplexity::Trivial => Self {
                swarm_size: 8,
                max_iters: 12,
                stall_limit: 3, // Aligned with C++ (iters/4)
                reinit_frac: 0.12,
                tol: 1e-7,
            },
            // Easy: 12×24 = 288 evals (C++ value)
            // Coplanar, similar altitude - analytical solutions guide PSO quickly
            TransferComplexity::Easy => Self {
                swarm_size: 12,
                max_iters: 24,
                stall_limit: 6, // Aligned with C++ (iters/4)
                reinit_frac: 0.14,
                tol: 5e-7,
            },
            // Moderate: 16×36 = 576 evals (C++ value)
            // 15-34° plane change - good analytical starting points reduce search space
            TransferComplexity::Moderate => Self {
                swarm_size: 16,
                max_iters: 36,
                stall_limit: 9, // Aligned with C++ (iters/4)
                reinit_frac: 0.15,
                tol: 1e-6,
            },
            // Hard: 24×48 = 1152 evals (C++ value)
            // Large plane change or altitude - still needs thorough search but not excessive
            TransferComplexity::Hard => Self {
                swarm_size: 24,
                max_iters: 48,
                stall_limit: 12, // Aligned with C++ (iters/4)
                reinit_frac: 0.22,
                tol: 1e-6,
            },
            // Extreme: 36×72 = 2592 evals (C++ value)
            // Bi-elliptic territory - still needs patience but avoid excess plateau iterations
            TransferComplexity::Extreme => Self {
                swarm_size: 36,
                max_iters: 72,
                stall_limit: 26,   // Matched to C++ stall_limit=26 (from pso.c)
                reinit_frac: 0.22, // Matched to C++ reinit_threshold=0.22
                tol: 1e-6,
            },
        }
    }
}

/// L-BFGS configuration preset for adaptive multi-start
#[derive(Clone, Copy, Debug)]
pub struct LbfgsPreset {
    pub num_starts: usize,
    pub base_radius: f64,
    pub max_iters_per_start: usize,
    pub excellent_threshold: f64,
    pub basin_tolerance: f64,
    /// Gradient tolerance for L-BFGS convergence (adaptive per complexity)
    pub gradient_tol: f64,
}

impl LbfgsPreset {
    #[must_use]
    pub const fn for_complexity(complexity: TransferComplexity) -> Self {
        match complexity {
            TransferComplexity::Trivial => Self {
                num_starts: 2,
                base_radius: 0.05,
                max_iters_per_start: 30,
                excellent_threshold: 0.02,
                basin_tolerance: 0.015,
                gradient_tol: 5e-4, // Relaxed tolerance for trivial problems
            },
            TransferComplexity::Easy => Self {
                num_starts: 3,
                base_radius: 0.10,
                max_iters_per_start: 40,
                excellent_threshold: 0.03,
                basin_tolerance: 0.02,
                gradient_tol: 5e-4, // Relaxed tolerance for easy problems
            },
            TransferComplexity::Moderate => Self {
                // Quick Win 1.2: Reduce starts and raise excellent threshold
                num_starts: 4,
                base_radius: 0.15,
                max_iters_per_start: 50,
                excellent_threshold: 0.08,
                basin_tolerance: 0.02,
                gradient_tol: 2e-4, // Moderate tolerance
            },
            TransferComplexity::Hard => Self {
                num_starts: 6,     // Compromise: 6 (was 7, tried 5 but unstable)
                base_radius: 0.20, // Slightly wider than 0.18
                max_iters_per_start: 50,
                excellent_threshold: 0.07, // Between 0.05 (Moderate) and 0.08 (original)
                basin_tolerance: 0.025,    // Original value (tighter than 0.03)
                gradient_tol: 1e-4,        // Conservative tolerance for hard problems
            },
            TransferComplexity::Extreme => Self {
                num_starts: 9,
                base_radius: 0.35,
                max_iters_per_start: 60,
                excellent_threshold: 0.15,
                basin_tolerance: 0.03,
                gradient_tol: 1e-4, // Conservative tolerance for extreme problems
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_failure_authority_code_has_stable_display() {
        assert_eq!(
            InvalidTargetPropagationAuthorityCode::OptimizerFailure.to_string(),
            "transfer optimizer failure"
        );
    }

    #[test]
    fn compact_status_codes_remain_stable() {
        assert_eq!(BranchStatusToken::Accepted.as_code(), 1);
        assert_eq!(BranchStatusToken::Rejected.as_code(), 2);
        assert_eq!(BranchRejectionToken::None.as_code(), 0);
        assert_eq!(
            BranchRejectionToken::UnsupportedHighFidelityCandidateSearch.as_code(),
            1
        );
        assert_eq!(TimingFailureToken::None.as_code(), 0);
        assert_eq!(
            TimingFailureToken::InterceptTransferTimeExceeded.as_code(),
            1
        );
        assert_eq!(TimingFailureToken::InterceptInsufficientLead.as_code(), 2);
        assert_eq!(TimingFailureToken::PhaseDvBoundExceeded.as_code(), 3);
        assert_eq!(
            TimingFailureToken::TransferRevolutionCapExceeded.as_code(),
            4
        );
    }

    #[test]
    fn verified_superset_metrics_add_assign_rejects_overflow_without_mutation() {
        let mut merged = VerifiedSupersetStageMetrics {
            batch_event_total_s: 1.5,
            polish_candidate_count: usize::MAX,
            ..VerifiedSupersetStageMetrics::default()
        };
        let incoming = VerifiedSupersetStageMetrics {
            batch_event_total_s: 2.25,
            polish_candidate_count: 1,
            ..VerifiedSupersetStageMetrics::default()
        };

        assert_eq!(
            merged.add_assign(incoming),
            Err(InvalidTargetPropagationAuthorityCode::ArithmeticOverflow)
        );

        assert_eq!(merged.polish_candidate_count, usize::MAX);
        assert_eq!(merged.batch_event_total_s.to_bits(), 1.5_f64.to_bits());
    }

    #[test]
    fn test_eci_basic_orbit() {
        let r = 6778.0;
        let v = (MU / r).sqrt();
        let state = [r, 0.0, 0.0, 0.0, v, 0.0];
        let orbit = EciBasicOrbit::from_eci(&state);
        assert!(orbit.is_some(), "circular LEO state must form an orbit");
        let Some(orbit) = orbit else {
            return;
        };
        assert!((orbit.sma - r).abs() < 10.0);
        assert!(orbit.ecc < 0.01);
    }

    #[test]
    fn test_transfer_complexity() {
        assert_eq!(
            TransferComplexity::classify(30.0, 0.01, 1.0, 0.01),
            TransferComplexity::Trivial
        );
    }

    #[test]
    fn test_lbfgs_preset() {
        let trivial = LbfgsPreset::for_complexity(TransferComplexity::Trivial);
        assert_eq!(trivial.num_starts, 2);
        assert_eq!(trivial.base_radius.to_bits(), 0.05_f64.to_bits());

        let hard = LbfgsPreset::for_complexity(TransferComplexity::Hard);
        assert_eq!(hard.num_starts, 6);
        assert!(hard.base_radius > trivial.base_radius);

        let extreme = LbfgsPreset::for_complexity(TransferComplexity::Extreme);
        assert_eq!(extreme.num_starts, 9);
        assert!(extreme.base_radius > hard.base_radius);
    }

    #[test]
    fn test_plan_context_from_request_populates_caches() {
        let dep_eci = [6778.0, 0.0, 0.0, 0.0, (MU / 6778.0).sqrt(), 0.0];
        let tgt_eci = [6878.0, 0.0, 0.0, 0.0, (MU / 6878.0).sqrt(), 0.0];

        let mut dep_equ = [0.0; 6];
        let mut tgt_equ = [0.0; 6];
        satpy_core::eci2equinoc_impl(&dep_eci, 6, 0.0, 0.0, &mut dep_equ);
        satpy_core::eci2equinoc_impl(&tgt_eci, 6, 0.0, 0.0, &mut tgt_equ);

        let request = TransferRequest {
            dep_eci,
            dep_equ,
            epoch_jd: 2_460_000.5,
            tgt_eci,
            tgt_equ,
            max_time_s: 86400.0,
            tof_penalty_weight: 0.1,
            revolution_cap: 2.0,
            max_phase_dv: 1.0,
            max_transfer_dv: 2.0,
            min_perigee: 6500.0,
            max_apogee: 50000.0,
            max_revs: 2,
            sampling_mode: SamplingMode::Fast,
            execution_policy: ExecutionPolicy {
                use_high_fidelity: false,
                require_high_fidelity: false,
                allow_parallel: false,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            ..TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };

        let ctx = PlanContext::from_request(request);

        assert_eq!(ctx.sampling_mode, SamplingMode::Fast);
        assert!(!ctx.execution_policy.allow_parallel);
        assert!(ctx.dep_orbit_valid);
        assert!(ctx.tgt_orbit_valid);
        assert!(ctx.dep_period > 0.0);
        assert!(ctx.tgt_period > 0.0);
        assert!(ctx.plane_angle_valid);
    }

    #[test]
    fn test_plan_context_from_request_preserves_explicit_time_fields() {
        let request = TransferRequest {
            max_time_s: 194_400.0,
            ..TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };

        let ctx = PlanContext::from_request(request);

        assert!((ctx.intercept_time_budget_s() - 194_400.0).abs() < 1e-9);
    }

    #[test]
    fn transfer_request_preserves_explicit_j2_closure_authority() {
        let expected = crate::solve::J2ClosureSettings {
            max_iterations: 3,
            endpoint_target_km: 0.000_25,
            correction_step_gain: 0.31,
        };
        let request = TransferRequest {
            j2_closure_settings: expected,
            ..TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };

        let ctx = PlanContext::from_request(request);

        assert_eq!(ctx.j2_closure_settings, expected);
    }

    #[test]
    fn transfer_request_preserves_analytical_target_authority_without_hf_config() {
        let request = TransferRequest {
            target_propagation_authority: TargetPropagationAuthority::AnalyticalKepler,
            force_config: None,
            ..TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };

        let ctx = PlanContext::from_request(request);

        assert_eq!(
            ctx.target_propagation_authority,
            TargetPropagationAuthority::AnalyticalKepler
        );
    }

    #[test]
    fn high_fidelity_scalar_target_rejects_gravity_only_ballistics() {
        let force_config = lightyear_odeint_rs::types::ForceConfig {
            sph_order: 5,
            force_flags: lightyear_odeint_rs::types::ForceFlags::DRAG
                | lightyear_odeint_rs::types::ForceFlags::SRP
                | lightyear_odeint_rs::types::ForceFlags::SUN_GRAVITY
                | lightyear_odeint_rs::types::ForceFlags::MOON_GRAVITY,
            atm_model: HIGH_FIDELITY_ATM_MODEL,
            target_propagation_mode: TargetPropagationAuthority::HighFidelity
                .as_force_config_code(),
            sun_pos: Some([149_600_000.0, 0.0, 0.0]),
            moon_pos: Some([384_400.0, 0.0, 0.0]),
            ..Default::default()
        };

        let outcome = validate_target_propagation_authority(
            TargetPropagationAuthority::HighFidelity,
            BodyForceConfig::gravity_only(BodyRole::DiagnosticTarget),
            Some(&force_config),
        );
        assert!(
            outcome.is_err(),
            "gravity-only target must not enter hybrid propagation"
        );
        let Err(error) = outcome else {
            return;
        };

        assert_eq!(
            error,
            InvalidTargetPropagationAuthorityCode::InvalidTargetBodyForce {
                authority: TargetPropagationAuthority::HighFidelity,
            }
        );
    }

    fn plan_context_template_fixture() -> PlanContextTemplate {
        PlanContextTemplate {
            max_time_s: 86400.0,
            tof_penalty_weight: 0.1,
            revolution_cap: 1.5,
            max_phase_dv: 1.0,
            max_transfer_dv: 2.0,
            min_perigee: 6500.0,
            max_apogee: 50000.0,
            max_revs: 2,
            sampling_mode: SamplingMode::Fast,
            execution_policy: ExecutionPolicy {
                use_high_fidelity: false,
                require_high_fidelity: false,
                allow_parallel: false,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            j2_closure_settings: crate::solve::J2ClosureSettings::default(),
            search_depth: SearchDepthPolicy::default(),
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            target_propagation_authority: TargetPropagationAuthority::MfJ2,
            force_config: None,
            packed_coeffs: None,
            local_optimizer: TransferLocalOptimizerConfig::default(),
        }
    }

    fn pair_plan_context_inputs_fixture() -> Option<PairPlanContextInputs> {
        let dep_eci = [6778.0, 0.0, 0.0, 0.0, (MU / 6778.0_f64).sqrt(), 0.0];
        let tgt_eci = [6878.0, 0.0, 0.0, 0.0, (MU / 6878.0_f64).sqrt(), 0.0];
        let mut dep_equ = [0.0; 6];
        let mut tgt_equ = [0.0; 6];
        satpy_core::eci2equinoc_impl(&dep_eci, 6, 0.0, 0.0, &mut dep_equ);
        satpy_core::eci2equinoc_impl(&tgt_eci, 6, 0.0, 0.0, &mut tgt_equ);
        let dep_orbit = EciBasicOrbit::from_eci(&dep_eci);
        let tgt_orbit = EciBasicOrbit::from_eci(&tgt_eci);
        assert!(
            dep_orbit.is_some(),
            "fixture deployer state must form an orbit"
        );
        assert!(
            tgt_orbit.is_some(),
            "fixture target state must form an orbit"
        );
        let (Some(dep_orbit), Some(tgt_orbit)) = (dep_orbit, tgt_orbit) else {
            return None;
        };
        let dep_period =
            std::f64::consts::TAU * ((dep_orbit.sma * dep_orbit.sma * dep_orbit.sma) / MU).sqrt();
        let tgt_period =
            std::f64::consts::TAU * ((tgt_orbit.sma * tgt_orbit.sma * tgt_orbit.sma) / MU).sqrt();
        Some(PairPlanContextInputs {
            dep_eci,
            dep_equ,
            epoch_jd: 2_460_000.5,
            tgt_eci,
            tgt_equ,
            dep_sma: dep_orbit.sma,
            dep_period,
            dep_orbit_cached: dep_orbit,
            dep_orbit_valid: true,
            tgt_period_cached: tgt_period,
            tgt_orbit_valid: true,
            tgt_sma: tgt_orbit.sma,
            tgt_period,
        })
    }

    #[test]
    fn test_search_depth_policy_reaches_context_via_request_and_template() {
        let custom = SearchDepthPolicy {
            tof_sample_budget: 256,
            coarse_early_stop: false,
            fine_total_limit: 24,
            coarse_reject_margin_km_s: 0.15,
            seed_fine_margin_km_s: 0.15,
            pair_proxy_model: PairProxyModel::Combined,
            oxymoo_policy: OxyMooPolicy::FastPopulation20Generations3InitialBest1,
            delta_v_anchor_policy: DeltaVAnchorPolicy::SeedLimit2,
            polish_scope_policy: PolishScopePolicy::NdEpsilon,
        };

        let ctx = PlanContext::from_request(TransferRequest {
            search_depth: custom,
            ..TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        });
        assert_eq!(ctx.search_depth, custom);

        // Template refresh is the path rayon workers take: a default-policy
        // reusable context must pick up the template's policy.
        let template = PlanContextTemplate {
            search_depth: custom,
            ..plan_context_template_fixture()
        };
        let mut reusable =
            PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        assert_ne!(reusable.search_depth, custom);
        let Some(inputs) = pair_plan_context_inputs_fixture() else {
            return;
        };
        let outcome = reusable.apply_template_pair(&template, &inputs);
        assert!(outcome.is_ok(), "valid target propagation authority");
        assert_eq!(reusable.search_depth, custom);
    }

    #[test]
    fn j2_closure_policy_reaches_reused_pair_context() {
        let custom = crate::solve::J2ClosureSettings {
            max_iterations: 3,
            endpoint_target_km: 0.000_25,
            correction_step_gain: 0.31,
        };
        let template = PlanContextTemplate {
            j2_closure_settings: custom,
            ..plan_context_template_fixture()
        };
        let mut reusable =
            PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        assert_ne!(reusable.j2_closure_settings, custom);
        let Some(inputs) = pair_plan_context_inputs_fixture() else {
            return;
        };
        let outcome = reusable.apply_template_pair(&template, &inputs);
        assert!(outcome.is_ok(), "valid target propagation authority");
        assert_eq!(reusable.j2_closure_settings, custom);
    }

    #[test]
    fn target_catalogue_authority_reaches_reused_pair_context() {
        let force = lightyear_odeint_rs::types::ForceConfig {
            target_propagation_mode: 1,
            ..Default::default()
        };
        let template = PlanContextTemplate {
            force_config: Some(std::sync::Arc::new(force)),
            ..plan_context_template_fixture()
        };
        let mut reusable =
            PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        let Some(inputs) = pair_plan_context_inputs_fixture() else {
            return;
        };
        let outcome = reusable.apply_template_pair(&template, &inputs);
        assert!(outcome.is_ok(), "valid target propagation authority");
        assert_eq!(
            reusable.target_propagation_authority,
            TargetPropagationAuthority::MfJ2
        );
    }

    #[test]
    fn analytical_target_authority_reaches_reused_context_without_hf_config() {
        let template = PlanContextTemplate {
            target_propagation_authority: TargetPropagationAuthority::AnalyticalKepler,
            force_config: None,
            ..plan_context_template_fixture()
        };
        let mut reusable =
            PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());

        let Some(inputs) = pair_plan_context_inputs_fixture() else {
            return;
        };
        let outcome = reusable.apply_template_pair(&template, &inputs);
        assert!(
            outcome.is_ok(),
            "explicit analytical target authority must be valid without HF config"
        );

        assert_eq!(
            reusable.target_propagation_authority,
            TargetPropagationAuthority::AnalyticalKepler
        );
    }

    #[test]
    fn invalid_target_catalogue_authority_fails_before_reused_context_mutation() {
        assert_eq!(
            TargetPropagationAuthority::try_from(0),
            Ok(TargetPropagationAuthority::HighFidelity)
        );
        assert_eq!(
            TargetPropagationAuthority::try_from(1),
            Ok(TargetPropagationAuthority::MfJ2)
        );
        assert_eq!(
            TargetPropagationAuthority::try_from(2),
            Ok(TargetPropagationAuthority::AnalyticalKepler)
        );
        assert_eq!(
            TargetPropagationAuthority::try_from(255),
            Err(InvalidTargetPropagationAuthorityCode::InvalidCode(255))
        );
        let force = lightyear_odeint_rs::types::ForceConfig {
            target_propagation_mode: 255,
            ..Default::default()
        };
        let template = PlanContextTemplate {
            force_config: Some(std::sync::Arc::new(force)),
            ..plan_context_template_fixture()
        };
        let mut reusable =
            PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        let original_epoch = reusable.epoch_jd;
        let original_authority = reusable.target_propagation_authority;

        let Some(inputs) = pair_plan_context_inputs_fixture() else {
            return;
        };
        let outcome = reusable.apply_template_pair(&template, &inputs);
        assert!(
            outcome.is_err(),
            "unknown target propagation code must fail closed"
        );
        let Err(error) = outcome else {
            return;
        };

        assert_eq!(
            error,
            InvalidTargetPropagationAuthorityCode::InvalidCode(255)
        );
        assert_eq!(reusable.epoch_jd.to_bits(), original_epoch.to_bits());
        assert_eq!(reusable.target_propagation_authority, original_authority);
    }

    #[test]
    fn contradictory_target_authority_fails_before_reused_context_mutation() {
        let force = lightyear_odeint_rs::types::ForceConfig {
            target_propagation_mode: TargetPropagationAuthority::MfJ2.as_force_config_code(),
            ..Default::default()
        };
        let template = PlanContextTemplate {
            target_propagation_authority: TargetPropagationAuthority::AnalyticalKepler,
            force_config: Some(std::sync::Arc::new(force)),
            ..plan_context_template_fixture()
        };
        let mut reusable =
            PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        let original_epoch = reusable.epoch_jd;
        let original_authority = reusable.target_propagation_authority;

        let Some(inputs) = pair_plan_context_inputs_fixture() else {
            return;
        };
        let outcome = reusable.apply_template_pair(&template, &inputs);
        assert!(
            outcome.is_err(),
            "contradictory target authorities must fail closed"
        );
        let Err(error) = outcome else {
            return;
        };

        assert_eq!(
            error,
            InvalidTargetPropagationAuthorityCode::Mismatch {
                explicit: TargetPropagationAuthority::AnalyticalKepler,
                force_config: TargetPropagationAuthority::MfJ2,
            }
        );
        assert_eq!(reusable.epoch_jd.to_bits(), original_epoch.to_bits());
        assert_eq!(reusable.target_propagation_authority, original_authority);
    }

    #[test]
    fn compact_candidate_carries_exact_selected_target_ballistics() {
        let mut plan = PlanResult::invalid();
        plan.valid = true;
        plan.replay_provenance.target_am_ratio = 0.02;
        plan.replay_provenance.target_cd = 2.2;
        plan.replay_provenance.target_cr = 1.3;

        let candidate = CompactTransferCandidate::from_plan_with_sat(4, 1, &plan);
        assert!(candidate.is_some(), "valid plan must compact");
        let Some(candidate) = candidate else {
            return;
        };

        assert_eq!(candidate.target_am_ratio.to_bits(), 0.02_f64.to_bits());
        assert_eq!(candidate.target_cd.to_bits(), 2.2_f64.to_bits());
        assert_eq!(candidate.target_cr.to_bits(), 1.3_f64.to_bits());
        assert_eq!(
            candidate.replay_policy.target_am_ratio.to_bits(),
            candidate.target_am_ratio.to_bits()
        );
        assert_eq!(
            candidate.replay_policy.target_cd.to_bits(),
            candidate.target_cd.to_bits()
        );
        assert_eq!(
            candidate.replay_policy.target_cr.to_bits(),
            candidate.target_cr.to_bits()
        );
    }

    // perf-lockin A1 (2026-07-09): size regression ceilings for the hot-scan
    // structs. Dominance/dedup/sort passes drag whole structs through cache;
    // growth past the current footprint should be a conscious decision, not
    // an accident. Raise a ceiling deliberately if a field is truly needed.
    #[test]
    fn test_plan_result_size_ceiling() {
        assert!(
            // Timeline-v2 exact replay carries immutable source authority and
            // constraints.  This is an explicit correctness cost.
            std::mem::size_of::<PlanResult>() <= 704,
            "PlanResult grew past 704B ({}B) - hot scans touch every byte",
            std::mem::size_of::<PlanResult>()
        );
    }

    #[test]
    fn test_compact_transfer_candidate_size_ceiling() {
        assert!(
            // Timeline-v2 retains immutable E0 replay controls and full
            // solver policy; this is deliberate, audited provenance.
            // Exact selected-target ballistics are duplicated deliberately:
            // direct payload fields plus immutable replay provenance.
            std::mem::size_of::<CompactTransferCandidate>() <= 656,
            "CompactTransferCandidate grew past 656B ({}B)",
            std::mem::size_of::<CompactTransferCandidate>()
        );
    }
}
