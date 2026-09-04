#[cfg(feature = "solver-qualification")]
use crate::evaluate::propagate_high_fidelity_state_at_epoch_checked_observed;
use crate::evaluate::{
    propagate_candidate_state_at_epoch, propagate_high_fidelity_state_at_epoch_checked,
    EvaluationArithmeticOverflow, TransferPropagationFailure,
};
use crate::intercept::{
    compute_miss_vector_equinoctial, optimize_intercept_bounded,
    optimize_intercept_bounded_hf_with_model, BoundedInterceptConfig, HfInterceptEvaluation,
};
use crate::lambert_backend::lambert_single_shot;
use crate::py_config::PhysicsConfigError;
use crate::py_config::{PhysicsConfig, PostprocessConfig};
use crate::types::{
    all_finite, BodyForceConfig, BodyRole, ExecutionPolicy, PlanContext, PropagationFidelity,
    StampedEciState, TargetPropagationAuthority, TransferRequest,
};

use dust_estimates_rs::fraction_prepare::{JD_CLOSURE_PHYSICAL_FLOOR_S, JD_CLOSURE_ULP_MULTIPLIER};
use dust_splitting_rs::linalg::dominant_eigenvector6;
use dust_splitting_rs::{split_gaussian_along_axis, split_gaussian_no_axis, SplitConfig};
use num_traits::ToPrimitive;
use satpy_core::{cross3, eci2equinoc_impl, norm3, MU, SEC_PER_DAY};
use smallvec::SmallVec;
#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;

use super::observer::{LegPath, PostprocessLegObserver, UnobservedPostprocessLeg};
use super::session::GlobalCoeffs;
#[cfg(any(test, feature = "bench-internal"))]
use super::session::{ResolvedPostprocessRuntimeSettings, TransferPostprocessScratch};
#[cfg(any(test, feature = "bench-internal"))]
use super::ukf::{propagate_component_means_ukf_batch, propagate_component_ukf_checked};
use super::ukf::{
    propagate_components_ukf_full_batch_observed_by, UkfFullBatchOutput, UkfPropagationFailure,
};
use super::{add_velocity, build_force_config, clamp_dv_guess, MAX_DUST_COMPONENTS};
#[cfg(feature = "solver-qualification")]
use super::{QualificationLegInput, QualificationLegTrace};

#[cfg(test)]
thread_local! {
    static CONJUNCTION_DIAGNOSTIC_CALLS: Cell<usize> = const { Cell::new(0) };
    static STAMPED_PROPAGATION_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
fn reset_conjunction_diagnostic_calls() {
    CONJUNCTION_DIAGNOSTIC_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
fn conjunction_diagnostic_calls() -> usize {
    CONJUNCTION_DIAGNOSTIC_CALLS.with(Cell::get)
}

#[cfg(test)]
fn record_conjunction_diagnostic_call() {
    CONJUNCTION_DIAGNOSTIC_CALLS.with(|calls| {
        calls.set(
            calls
                .get()
                .checked_add(1)
                .expect("test conjunction diagnostic counter overflow"),
        );
    });
}

#[cfg(test)]
fn reset_stamped_propagation_calls() {
    STAMPED_PROPAGATION_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
fn stamped_propagation_calls() -> usize {
    STAMPED_PROPAGATION_CALLS.with(Cell::get)
}

/// Full dust distribution at intercept after postprocess correction.
///
/// Units: km, km/s, km^2, etc (same as ECI state).
#[derive(Clone, Debug)]
pub struct PostprocessDustDistribution {
    pub release_jd: f64,
    pub dust_free_flight_s: f64,
    pub dust_mean: [f64; 6],
    pub weights: SmallVec<[f64; MAX_DUST_COMPONENTS]>,
    pub release_comp_means: SmallVec<[[f64; 6]; MAX_DUST_COMPONENTS]>,
    pub release_comp_covs: SmallVec<[[[f64; 6]; 6]; MAX_DUST_COMPONENTS]>,
    pub comp_means: SmallVec<[[f64; 6]; MAX_DUST_COMPONENTS]>,
    pub comp_covs: SmallVec<[[[f64; 6]; 6]; MAX_DUST_COMPONENTS]>,
    /// Every propagated sigma point, retained for bit-exactness assertions.
    /// Test-only: no production consumer reads this, and retaining it in every
    /// staged distribution held multiple megabytes across a bounded strict-HF
    /// batch. The UKF still returns the points in [`UkfFullBatchOutput`]; only
    /// their retention here is gated.
    #[cfg(test)]
    pub propagated_sigma_points: Vec<[f64; 6]>,
    pub correction_dv_norm: f64,
}

pub struct AuthoritativeReleaseDistribution {
    pub means: Vec<[f64; 6]>,
    pub covariances: Vec<[[f64; 6]; 6]>,
    pub weights: Vec<f64>,
    pub sigma_points: Option<Vec<[f64; 6]>>,
}

#[cfg(any(test, feature = "bench-internal"))]
pub(super) struct PostprocessDustSummary {
    pub(super) dust_mean: [f64; 6],
    pub(super) correction_dv_norm: f64,
}

/// Explicit result of release-epoch control.  This is deliberately separate
/// from the propagated dust distribution: R is the only epoch at which a
/// physical release maneuver may be solved or applied.
#[derive(Clone, Debug, PartialEq)]
pub enum PostprocessControlStatus {
    Applied,
    AppliedZero,
    InvalidTimeline,
    Configuration(PhysicsConfigError),
    CoastFailure,
    SolveFailure,
    /// Payload-specific launch/coast state violates a deterministic protected-
    /// radius constraint before any release-control solve is attempted.
    DeterministicPhysicalInfeasible,
    /// One locally solved release control violates a downstream constraint.
    /// This is not a proof that no other admissible control exists.
    ControlSolutionConstraintViolation,
    PropagationFailure(TransferPropagationFailure),
}

impl fmt::Display for PostprocessControlStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(error) => write!(formatter, "postprocess configuration: {error}"),
            Self::PropagationFailure(error) => {
                write!(formatter, "postprocess propagation: {error}")
            }
            _ => formatter.write_str("postprocess release control failed"),
        }
    }
}

/// Typed reason why a full native postprocess distribution was unavailable.
///
/// Strict HF callers must consume this result directly.  It intentionally has
/// no MF fallback: an unavailable HF asset or failed HF propagation is an
/// objective failure, not permission to silently switch fidelity.
#[derive(Clone, Debug, PartialEq)]
pub enum PostprocessDistributionStatus {
    MissingCandidate,
    InvalidCandidate,
    InvalidTimeline,
    StrictHfAssetsUnavailable,
    InvalidFraction,
    InvalidPlan,
    ArithmeticOverflow,
    Allocation,
    ReleaseControl(PostprocessControlStatus),
    InvalidReleaseCovariance,
    InvalidSplitAxis,
    InvalidSplitAlpha,
    InvalidReleaseDistribution,
    PropagationFailure(TransferPropagationFailure),
    UkfPropagationFailure(UkfPropagationFailure),
}

impl fmt::Display for PostprocessDistributionStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCandidate => formatter.write_str("postprocess candidate is missing"),
            Self::InvalidCandidate => formatter.write_str("postprocess candidate is invalid"),
            Self::InvalidTimeline => formatter.write_str("postprocess timeline is invalid"),
            Self::StrictHfAssetsUnavailable => {
                formatter.write_str("strict HF postprocess assets are unavailable")
            }
            Self::InvalidFraction => formatter.write_str("postprocess fraction is invalid"),
            Self::InvalidPlan => formatter.write_str("postprocess plan is invalid"),
            Self::ArithmeticOverflow => formatter.write_str("postprocess arithmetic overflow"),
            Self::Allocation => formatter.write_str("postprocess allocation failed"),
            Self::ReleaseControl(PostprocessControlStatus::Configuration(error)) => {
                write!(formatter, "postprocess configuration: {error}")
            }
            Self::ReleaseControl(PostprocessControlStatus::PropagationFailure(error))
            | Self::PropagationFailure(error) => {
                write!(formatter, "postprocess propagation: {error}")
            }
            Self::ReleaseControl(_) => formatter.write_str("postprocess release control failed"),
            Self::InvalidReleaseCovariance => {
                formatter.write_str("postprocess release covariance is invalid")
            }
            Self::InvalidSplitAxis => formatter.write_str("postprocess split axis is invalid"),
            Self::InvalidSplitAlpha => formatter.write_str("postprocess split alpha is invalid"),
            Self::InvalidReleaseDistribution => {
                formatter.write_str("postprocess release distribution is invalid")
            }
            Self::UkfPropagationFailure(UkfPropagationFailure::Ephemeris { .. }) => {
                formatter.write_str("postprocess UKF ephemeris is unavailable")
            }
            Self::UkfPropagationFailure(_) => {
                formatter.write_str("postprocess UKF propagation failed")
            }
        }
    }
}

impl std::error::Error for PostprocessControlStatus {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Configuration(error) => Some(error),
            Self::PropagationFailure(error) => Some(error),
            _ => None,
        }
    }
}

impl std::error::Error for PostprocessDistributionStatus {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ReleaseControl(error) => Some(error),
            Self::PropagationFailure(error) => Some(error),
            Self::UkfPropagationFailure(error) => Some(error),
            _ => None,
        }
    }
}

#[inline]
fn try_distribution_vec_with_capacity<T>(
    capacity: usize,
) -> Result<Vec<T>, PostprocessDistributionStatus> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| PostprocessDistributionStatus::Allocation)?;
    Ok(values)
}

#[inline]
fn copy_split_result_slice<T: Copy>(source: &[T]) -> Result<Vec<T>, PostprocessDistributionStatus> {
    let mut copy = try_distribution_vec_with_capacity(source.len())?;
    copy.extend(source.iter().copied());
    Ok(copy)
}

#[inline]
fn distribution_status_from_propagation_failure(
    failure: TransferPropagationFailure,
) -> PostprocessDistributionStatus {
    match failure {
        TransferPropagationFailure::ArithmeticOverflow => {
            PostprocessDistributionStatus::ArithmeticOverflow
        }
        failure => PostprocessDistributionStatus::PropagationFailure(failure),
    }
}

#[inline]
fn distribution_status_from_release_control_status(
    status: PostprocessControlStatus,
) -> PostprocessDistributionStatus {
    match status {
        PostprocessControlStatus::PropagationFailure(failure) => {
            distribution_status_from_propagation_failure(failure)
        }
        status => PostprocessDistributionStatus::ReleaseControl(status),
    }
}

#[cfg(test)]
mod postprocess_distribution_status_tests {
    use super::PostprocessDistributionStatus;

    #[test]
    fn arithmetic_overflow_status_has_stable_display() {
        assert_eq!(
            PostprocessDistributionStatus::ArithmeticOverflow.to_string(),
            "postprocess arithmetic overflow"
        );
    }

    #[test]
    fn direct_distribution_allocation_has_stable_typed_status() {
        assert_eq!(
            super::try_distribution_vec_with_capacity::<u8>(usize::MAX),
            Err(PostprocessDistributionStatus::Allocation)
        );
        assert_eq!(
            PostprocessDistributionStatus::Allocation.to_string(),
            "postprocess allocation failed"
        );
    }

    #[test]
    fn release_control_overflow_projects_to_postprocess_overflow() {
        assert_eq!(
            super::distribution_status_from_release_control_status(
                super::PostprocessControlStatus::PropagationFailure(
                    crate::evaluate::TransferPropagationFailure::ArithmeticOverflow,
                ),
            ),
            PostprocessDistributionStatus::ArithmeticOverflow
        );
    }

    #[test]
    fn ukf_overflow_projects_to_postprocess_overflow() {
        let failure = super::UkfPropagationFailure::Propagation(
            crate::evaluate::TransferPropagationFailure::ArithmeticOverflow,
        );
        assert_eq!(
            super::distribution_status_from_ukf_failure(failure),
            PostprocessDistributionStatus::ArithmeticOverflow
        );
    }

    #[test]
    fn ukf_allocation_projects_to_postprocess_allocation() {
        assert_eq!(
            super::distribution_status_from_ukf_failure(super::UkfPropagationFailure::Allocation,),
            PostprocessDistributionStatus::Allocation
        );
    }

    #[test]
    fn non_overflow_propagation_keeps_its_typed_status() {
        assert_eq!(
            super::distribution_status_from_propagation_failure(
                crate::evaluate::TransferPropagationFailure::Authority,
            ),
            PostprocessDistributionStatus::PropagationFailure(
                crate::evaluate::TransferPropagationFailure::Authority,
            )
        );
    }

    #[test]
    fn census_propagation_keeps_its_exact_typed_status() {
        let failure = crate::evaluate::TransferPropagationFailure::Census(
            lightyear_odeint_rs::probe::PropagationCensusError::MutexPoisoned,
        );

        assert_eq!(
            super::distribution_status_from_propagation_failure(failure.clone()),
            PostprocessDistributionStatus::PropagationFailure(failure)
        );
    }

    #[test]
    fn release_configuration_remains_in_distribution_source_chain() {
        let expected = crate::py_config::PhysicsConfigError::UnsupportedIntegratorMethod;
        let status = PostprocessDistributionStatus::ReleaseControl(
            super::PostprocessControlStatus::Configuration(expected),
        );

        let control_source = std::error::Error::source(&status)
            .expect("distribution must retain release-control failure");
        let config_source = control_source
            .source()
            .expect("release-control failure must retain configuration source");
        assert_eq!(
            config_source.downcast_ref::<crate::py_config::PhysicsConfigError>(),
            Some(&expected)
        );
    }

    #[test]
    fn generic_ukf_failure_remains_in_distribution_source_chain() {
        let status = super::distribution_status_from_ukf_failure(
            super::UkfPropagationFailure::NativeBatch { source: None },
        );

        let source =
            std::error::Error::source(&status).expect("distribution must retain exact UKF failure");
        assert!(
            source
                .downcast_ref::<super::UkfPropagationFailure>()
                .is_some(),
            "unexpected source: {source}"
        );
    }
}

fn distribution_status_from_ukf_failure(
    failure: UkfPropagationFailure,
) -> PostprocessDistributionStatus {
    match failure {
        UkfPropagationFailure::Propagation(error) => {
            distribution_status_from_propagation_failure(error)
        }
        UkfPropagationFailure::Allocation => PostprocessDistributionStatus::Allocation,
        failure => PostprocessDistributionStatus::UkfPropagationFailure(failure),
    }
}

#[derive(Clone, Debug)]
pub struct PostprocessControl {
    /// L, immediately before the stored transfer burn.
    pub transfer_burn_pre_state: StampedEciState,
    /// L, after the stored transfer burn; this is the canister coast input.
    pub canister_launch_state: StampedEciState,
    /// R, before the release-control vector.
    pub release_pre_control_state: StampedEciState,
    /// R, after exactly one release-control vector.
    pub release_post_control_state: StampedEciState,
    /// I predicted from the release-after-control state.
    pub predicted_intercept_state: StampedEciState,
    /// Actual selected catalogue target at I, never a solver-frame proxy.
    pub selected_target_state: StampedEciState,
    pub release_control_dv: [f64; 3],
    pub release_control_dv_norm: f64,
    /// Authoritative partition used to derive C and F for this control.
    pub canister_tof_fraction: f64,
    pub canister_coast_s: f64,
    pub dust_free_flight_s: f64,
    pub fidelity: PropagationFidelity,
    pub status: PostprocessControlStatus,
    /// I->Cj separation only; it never changes a physical state or DV.
    pub conjunction_separation_km: f64,
}

#[derive(Clone, Copy)]
pub(super) struct SummaryPlanInputs {
    pub(super) valid: bool,
    pub(super) release_state: [f64; 6],
    pub(super) transfer_dv: [f64; 3],
    pub(super) tof_jd_start: f64,
    pub(super) min_radius_km: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BoundedDustTiming {
    #[cfg(test)]
    pub(super) transfer_to_intercept: f64,
    #[cfg(test)]
    pub(super) dust_window: f64,
    #[cfg(test)]
    pub(super) pre_window_hold: f64,
    pub(super) canister_coast: f64,
    pub(super) dust_flight: f64,
}

#[derive(Clone, Copy, Debug)]
struct ReleaseTimelineBoundary {
    transfer_burn_jd: f64,
    release_jd: f64,
    intercept_jd: f64,
    canister_coast_s: f64,
    dust_free_flight_s: f64,
}

pub(super) const fn canister_tof_fraction(post: &PostprocessConfig) -> f64 {
    post.canister_tof_fraction
}

pub(super) fn resolve_bounded_dust_timing(
    transfer_to_intercept_s: f64,
    configured_dust_phase_tof_s: f64,
    canister_tof_fraction: f64,
) -> Option<BoundedDustTiming> {
    if !transfer_to_intercept_s.is_finite() || transfer_to_intercept_s <= 0.0 {
        return None;
    }
    if !configured_dust_phase_tof_s.is_finite() || configured_dust_phase_tof_s <= 0.0 {
        return None;
    }
    if !canister_tof_fraction.is_finite() || !(0.0..1.0).contains(&canister_tof_fraction) {
        return None;
    }
    let canister_frac = canister_tof_fraction;
    let dust_window_s = transfer_to_intercept_s.min(configured_dust_phase_tof_s);
    let pre_window_hold_s = (transfer_to_intercept_s - dust_window_s).max(0.0);
    let partitioned_canister_s = canister_frac * dust_window_s;
    let dust_canister_coast_s = pre_window_hold_s + partitioned_canister_s;
    let dust_free_flight_s = (dust_window_s - partitioned_canister_s).max(0.0);
    Some(BoundedDustTiming {
        #[cfg(test)]
        transfer_to_intercept: transfer_to_intercept_s,
        #[cfg(test)]
        dust_window: dust_window_s,
        #[cfg(test)]
        pre_window_hold: pre_window_hold_s,
        canister_coast: dust_canister_coast_s,
        dust_flight: dust_free_flight_s,
    })
}

fn resolve_release_timeline(
    transfer_burn_jd: f64,
    intercept_jd: f64,
    post: &PostprocessConfig,
) -> Option<ReleaseTimelineBoundary> {
    if !transfer_burn_jd.is_finite() || !intercept_jd.is_finite() {
        return None;
    }
    let transfer_to_intercept_s = (intercept_jd - transfer_burn_jd) * SEC_PER_DAY;
    let timing = resolve_bounded_dust_timing(
        transfer_to_intercept_s,
        post.dust_phase_tof_s,
        canister_tof_fraction(post),
    )?;
    if timing.dust_flight <= 0.0 {
        return None;
    }
    let release_jd = transfer_burn_jd + timing.canister_coast / SEC_PER_DAY;
    let recomposed_intercept_jd = release_jd + timing.dust_flight / SEC_PER_DAY;
    if !jd_closure_within_tolerance(recomposed_intercept_jd, intercept_jd) {
        return None;
    }
    Some(ReleaseTimelineBoundary {
        transfer_burn_jd,
        release_jd,
        intercept_jd,
        canister_coast_s: timing.canister_coast,
        dust_free_flight_s: timing.dust_flight,
    })
}

#[inline]
fn jd_ulp_days(value: f64) -> f64 {
    let magnitude = value.abs();
    if !magnitude.is_finite() {
        return f64::INFINITY;
    }
    if magnitude == 0.0 {
        return f64::from_bits(1);
    }
    let bits = magnitude.to_bits();
    let Some(successor_bits) = bits.checked_add(1) else {
        return f64::INFINITY;
    };
    f64::from_bits(successor_bits) - magnitude
}

#[inline]
fn jd_closure_tolerance_s(lhs_jd: f64, rhs_jd: f64) -> f64 {
    let representable_floor_s =
        JD_CLOSURE_ULP_MULTIPLIER * jd_ulp_days(lhs_jd).max(jd_ulp_days(rhs_jd)) * SEC_PER_DAY;
    JD_CLOSURE_PHYSICAL_FLOOR_S.max(representable_floor_s)
}

#[inline]
fn jd_closure_within_tolerance(lhs_jd: f64, rhs_jd: f64) -> bool {
    lhs_jd.is_finite()
        && rhs_jd.is_finite()
        && (lhs_jd - rhs_jd).abs() * SEC_PER_DAY <= jd_closure_tolerance_s(lhs_jd, rhs_jd)
}

pub(super) fn normalize_weights(weights: &mut [f64]) {
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        let inv = 1.0 / sum;
        for w in weights {
            *w *= inv;
        }
    } else if !weights.is_empty() {
        let inv = 1.0 / weights.len().to_f64().unwrap_or(f64::INFINITY);
        for w in weights {
            *w = inv;
        }
    }
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
    let eccentricity = norm3(&e_vec);
    let perigee = (h_sq / MU) / (1.0 + eccentricity);
    (perigee.is_finite() && perigee > 0.0).then_some(perigee)
}

#[inline]
fn state_clears_min_radius(state: &[f64; 6], min_radius_km: f64) -> bool {
    min_radius_km.is_finite()
        && min_radius_km > 0.0
        && norm3(&[state[0], state[1], state[2]]) >= min_radius_km
        && osculating_perigee_km(state).is_some_and(|rp| rp >= min_radius_km)
}

pub(super) fn propagate_stamped_checked(
    state: &StampedEciState,
    dt_s: f64,
    body_force: BodyForceConfig,
    ctx: &PlanContext,
) -> Result<StampedEciState, TransferPropagationFailure> {
    #[cfg(test)]
    STAMPED_PROPAGATION_CALLS.with(|calls| {
        calls.set(
            calls
                .get()
                .checked_add(1)
                .expect("test stamped propagation counter overflow"),
        );
    });
    if !dt_s.is_finite() || !state.jd.is_finite() {
        return Err(TransferPropagationFailure::InvalidInput);
    }
    let mut equ = [0.0; 6];
    eci2equinoc_impl(&state.eci, 6, 0.0, 0.0, &mut equ);
    let out = if ctx.execution_policy.use_high_fidelity {
        propagate_high_fidelity_state_at_epoch_checked(&equ, dt_s, state.jd, body_force, ctx)?
    } else {
        propagate_candidate_state_at_epoch(&equ, dt_s, state.jd, body_force, ctx)
            .map_err(|_: EvaluationArithmeticOverflow| {
                TransferPropagationFailure::ArithmeticOverflow
            })?
            .ok_or(TransferPropagationFailure::InvalidInput)?
    };
    if !out.iter().all(|value| value.is_finite()) {
        return Err(TransferPropagationFailure::NonFiniteOutput);
    }
    let target_jd = state.jd + dt_s / SEC_PER_DAY;
    if !target_jd.is_finite() {
        return Err(TransferPropagationFailure::InvalidInput);
    }
    Ok(StampedEciState::new(out, target_jd))
}

#[cfg(feature = "solver-qualification")]
pub(super) fn propagate_stamped_checked_observed(
    state: &StampedEciState,
    dt_s: f64,
    body_force: BodyForceConfig,
    ctx: &PlanContext,
    path: LegPath,
    trace: &mut QualificationLegTrace,
) -> Result<StampedEciState, TransferPropagationFailure> {
    if !dt_s.is_finite() || !state.jd.is_finite() {
        return Err(TransferPropagationFailure::InvalidInput);
    }
    let mut equ = [0.0; 6];
    eci2equinoc_impl(&state.eci, 6, 0.0, 0.0, &mut equ);
    let out = if ctx.execution_policy.use_high_fidelity {
        let observed = match propagate_high_fidelity_state_at_epoch_checked_observed(
            &equ, dt_s, state.jd, body_force, ctx,
        ) {
            Ok(observed) => observed,
            Err(error) => {
                trace.mark_incomplete(super::QualificationTraceError::IncompleteMetrics);
                return Err(error);
            }
        };
        let outcome = observed.outcome;
        trace.record_observed_transfer(
            QualificationLegInput::new(path, body_force.role, state.jd, 0.0, dt_s, state.eci),
            outcome.clone(),
            observed.scalar_observation,
        );
        outcome?
    } else {
        propagate_candidate_state_at_epoch(&equ, dt_s, state.jd, body_force, ctx)
            .map_err(|_: EvaluationArithmeticOverflow| {
                TransferPropagationFailure::ArithmeticOverflow
            })?
            .ok_or(TransferPropagationFailure::InvalidInput)?
    };
    if !out.iter().all(|value| value.is_finite()) {
        return Err(TransferPropagationFailure::NonFiniteOutput);
    }
    let target_jd = state.jd + dt_s / SEC_PER_DAY;
    if !target_jd.is_finite() {
        return Err(TransferPropagationFailure::InvalidInput);
    }
    Ok(StampedEciState::new(out, target_jd))
}

/// Propagate one stamped state while retaining the exact source failure.
fn propagate_stamped(
    state: &StampedEciState,
    dt_s: f64,
    body_force: BodyForceConfig,
    ctx: &PlanContext,
) -> Result<StampedEciState, TransferPropagationFailure> {
    propagate_stamped_checked(state, dt_s, body_force, ctx)
}

fn resolve_predicted_intercept<O: PostprocessLegObserver>(
    endpoint: Option<[f64; 6]>,
    propagated_source_jd: f64,
    release_epoch_jd: f64,
    release_post_control_state: &StampedEciState,
    dust_free_flight_s: f64,
    dust_body_force: BodyForceConfig,
    dust_ctx: &PlanContext,
    observer: &mut O,
) -> Result<StampedEciState, TransferPropagationFailure> {
    if propagated_source_jd.to_bits() == release_epoch_jd.to_bits() {
        if let Some(eci) = endpoint {
            let target_jd = release_post_control_state.jd + dust_free_flight_s / SEC_PER_DAY;
            if all_finite(&eci) && target_jd.is_finite() {
                return Ok(StampedEciState::new(eci, target_jd));
            }
            return Err(TransferPropagationFailure::NonFiniteOutput);
        }
    }
    observer.propagate_stamped(
        release_post_control_state,
        dust_free_flight_s,
        dust_body_force,
        dust_ctx,
        LegPath::ReleaseEndpointFallback,
    )
}

#[inline]
pub(super) fn finite_positive_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

pub(super) fn release_covariance_from_conf(
    conf: &PhysicsConfig,
    release_state: &[f64; 6],
) -> Option<[[f64; 6]; 6]> {
    let [rx, ry, rz, vx, vy, vz] = *release_state;
    let r = [rx, ry, rz];
    let v = [vx, vy, vz];
    let r_norm = norm3(&r);
    let h = cross3(&r, &v);
    let h_norm = norm3(&h);
    if !(r_norm > 0.0 && h_norm > 0.0) {
        return None;
    }

    let inv_r = 1.0 / r_norm;
    let r_hat = [r[0] * inv_r, r[1] * inv_r, r[2] * inv_r];
    let inv_h = 1.0 / h_norm;
    let n_hat = [h[0] * inv_h, h[1] * inv_h, h[2] * inv_h];
    let t_hat = cross3(&n_hat, &r_hat);
    let basis = [r_hat, t_hat, n_hat];

    let pos_i = finite_positive_or(conf.dust_pos_sigma_m, 0.0) * 1e-3;
    let pos_rc = finite_positive_or(
        conf.dust_pos_sigma_radial_cross_track_m,
        conf.dust_pos_sigma_m,
    ) * 1e-3;
    let vel_i = finite_positive_or(conf.dust_vel_sigma_mps, 0.0) * 1e-3;
    let vel_rc = finite_positive_or(
        conf.dust_vel_sigma_radial_cross_track_mps,
        conf.dust_vel_sigma_mps,
    ) * 1e-3;
    let pos_vars = [pos_rc * pos_rc, pos_i * pos_i, pos_rc * pos_rc];
    let vel_vars = [vel_rc * vel_rc, vel_i * vel_i, vel_rc * vel_rc];

    let mut cov6 = [[0.0; 6]; 6];
    let (position_rows, velocity_rows) = cov6.split_at_mut(3);
    for (i, (position_row, velocity_row)) in position_rows
        .iter_mut()
        .zip(velocity_rows.iter_mut())
        .enumerate()
    {
        for (j, (position_entry, velocity_entry)) in position_row
            .iter_mut()
            .take(3)
            .zip(velocity_row.iter_mut().skip(3))
            .enumerate()
        {
            let mut pos_ij = 0.0;
            let mut vel_ij = 0.0;
            for ((basis_vector, &position_variance), &velocity_variance) in
                basis.iter().zip(pos_vars.iter()).zip(vel_vars.iter())
            {
                let basis_i = basis_vector.get(i).copied()?;
                let basis_j = basis_vector.get(j).copied()?;
                pos_ij += basis_i * position_variance * basis_j;
                vel_ij += basis_i * velocity_variance * basis_j;
            }
            *position_entry = pos_ij;
            *velocity_entry = vel_ij;
        }
    }
    Some(cov6)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SplitAxisError {
    InvalidInput,
}

/// Validate the compiled split criterion and return the explicit split axis it
/// asks for, if any.
///
/// The one compiled producer is `nd_pipeline/src/hybrid.rs`, which emits the
/// literal `"maxvar"` (until 2026-08-06 it reached that token through a
/// one-variant `nd_config::PartAMfSplittingCriterion`).
/// `"maxvar"` names no explicit axis, so this returns `Ok(None)` and the caller
/// splits along the dominant covariance eigenvector. `"linear"` and `"cov"` are
/// legacy aliases with the same meaning; every other token is rejected so a
/// stale config cannot silently rotate the split.
pub(super) fn select_split_axis_strict(
    criterion: &str,
    tof_dust: f64,
) -> Result<Option<[f64; 6]>, SplitAxisError> {
    if !tof_dust.is_finite() {
        return Err(SplitAxisError::InvalidInput);
    }
    if !matches!(criterion, "linear" | "maxvar" | "cov") {
        return Err(SplitAxisError::InvalidInput);
    }

    Ok(None)
}

pub(super) fn compute_lambert_guess(
    r0: &[f64; 3],
    v0: &[f64; 3],
    r_target: &[f64; 3],
    tof_s: f64,
) -> Option<[f64; 3]> {
    let zero_v = [0.0; 3];
    lambert_single_shot(r0, r_target, v0, &zero_v, tof_s, 0, true, true).map(|(dv, _)| dv)
}

#[inline]
pub(super) fn default_intercept_bound_kms(tof_dust_full_s: f64, use_high_fidelity: bool) -> f64 {
    if !use_high_fidelity {
        return 0.1;
    }
    let tof_hours = tof_dust_full_s / 3600.0;
    if tof_hours > 6.0 {
        (1.0 + 0.0833 * (tof_hours - 6.0)).min(2.0)
    } else {
        1.0
    }
}

#[inline]
pub(super) fn build_intercept_cfg(
    post: &PostprocessConfig,
    min_miss_km: f64,
    bound_kms: f64,
    reg_weight: f64,
    max_bound_expansions: usize,
) -> BoundedInterceptConfig {
    BoundedInterceptConfig {
        max_iters: post.fix_ls_max_nfev,
        tol: post.fix_ls_tol,
        skip_tol: post.fix_ls_skip_tol,
        bound: bound_kms,
        max_bound: post.max_physical_dv_kms.max(bound_kms),
        reg_weight,
        min_miss_km,
        max_bound_expansions,
        ..BoundedInterceptConfig::default()
    }
}

struct ReleaseControlSolution {
    dv: [f64; 3],
    endpoint: Option<[f64; 6]>,
}

#[expect(
    clippy::too_many_lines,
    reason = "the MF and strict-HF branches share one ordered physical-control transaction and typed failure path"
)]
fn solve_intercept_delta_dv<O: PostprocessLegObserver>(
    dust_release: &[f64; 6],
    target_pos: [f64; 3],
    tof_dust_full_s: f64,
    conf: &PhysicsConfig,
    post: &PostprocessConfig,
    ctx_dust: &PlanContext,
    mut dv_guess: [f64; 3],
    min_miss_km: f64,
    min_radius_km: f64,
    observer: &mut O,
) -> Result<Option<ReleaseControlSolution>, TransferPropagationFailure> {
    let v0 = [dust_release[3], dust_release[4], dust_release[5]];
    let r0 = [dust_release[0], dust_release[1], dust_release[2]];
    let default_bound = default_intercept_bound_kms(tof_dust_full_s, conf.use_high_fidelity);
    let clears_radius = |dv: [f64; 3]| {
        state_clears_min_radius(
            &[
                r0[0],
                r0[1],
                r0[2],
                v0[0] + dv[0],
                v0[1] + dv[1],
                v0[2] + dv[2],
            ],
            min_radius_km,
        )
    };
    // Finite, violation-sloped pseudo-miss for protected-radius violations.
    // A NaN sentinel walls the LM solver off with no gradient: release
    // orbits whose perigee sits a few km above the floor (common low-LEO
    // geometries) confine it to a thin feasible shell where it exhausts its
    // iteration budget. The penalty dominates any physical miss and slopes
    // back toward feasibility; the hard clears/norm checks downstream still
    // reject any solution that actually violates the constraint.
    let radius_penalty_miss = |dv: [f64; 3]| -> [f64; 3] {
        let state = [
            r0[0],
            r0[1],
            r0[2],
            v0[0] + dv[0],
            v0[1] + dv[1],
            v0[2] + dv[2],
        ];
        let rp = osculating_perigee_km(&state).unwrap_or(0.0);
        let violation_km = (min_radius_km - rp).max(0.0);
        let p = 1.0e6 + 1.0e3 * violation_km;
        [p, p, p]
    };

    // The Jacobian model handed to every HIGH-FIDELITY intercept solve below.
    //
    // The Keplerian form of the same miss map, same target and same time of
    // flight, differing from the objective in one place: it propagates
    // two-body rather than strict HF, so it costs no propagation. The LM
    // differences THIS to build its 3x3, then evaluates every step it proposes
    // against the real objective, so the model steers and never enters the
    // answer. Measured against the true finite-difference Jacobian on 3,402-
    // and 2,550-row strict-HF corpora: `||J_model - J_fd||_F / ||J_fd||_F`
    // median 3.1e-3, 91.5% below 5%.
    //
    // It answers EVERYWHERE, including points that violate the protected
    // radius, and the two alternatives were both built and measured against
    // this one on the 4-design/2-event corpus. Baseline, differencing the
    // strict-HF objective: 3400 rows, 64,137 release-control propagations.
    //
    //   this, smooth everywhere        3402 rows   25,160 props  -60.8%
    //   declines inside the region     3396 rows   34,308 props  -46.3%
    //   penalty mirrored into model    lost rows and saving on both counts
    //
    // Both "safer"-looking variants cost real rows AND saving. The reason is
    // the penalty's magnitude: ~1e6 km against a physical miss of ~1 km, so a
    // single violating probe makes a modelled column ~1e10 and the 3x3
    // effectively rank-1 -- the same conditioning pathology the
    // finite-difference Jacobian has there (measured cond ~1e47 against ~25
    // elsewhere). Whichever way that shape reaches the normal equations, it
    // hurts.
    //
    // Enforcement stays with the OBJECTIVE, which is what actually holds the
    // constraint: the solver may propose a step into the forbidden region, the
    // trial evaluation returns the penalty, the cost rises and the step is
    // rejected. See `smooth_model_never_walks_through_a_penalty_it_cannot_see`.
    //
    // KNOWN LIMIT, named rather than hidden: if the LM ITERATE is already
    // inside the region, the objective returns the flat sentinel while this
    // model returns a finite slope, the two are inconsistent, and the step
    // built from them is too large to be useful -- the solver cannot climb out
    // and the pass exits on its best feasible point instead. Measured on 5,952
    // strict-HF rows across two corpora that costs nothing, because the seed
    // is itself the output of a penalised MF solve and effectively never
    // starts there. It is a real corner, and the test above pins the part that
    // must never break: it cannot end up on the wrong side of the wall.
    let jacobian_model = |dv: [f64; 3]| -> Option<[f64; 3]> {
        Some(compute_miss_vector_equinoctial(
            dv,
            v0,
            r0,
            target_pos,
            tof_dust_full_s,
        ))
    };

    if post.hybrid_mf_seed_hf_refine {
        let original_guess = dv_guess;
        let mf_bound = if post.mf_seed_bound_kms > 0.0 {
            post.mf_seed_bound_kms
        } else {
            0.1
        };
        let max_guess = (0.1_f64).min(mf_bound * 0.95);
        clamp_dv_guess(&mut dv_guess, max_guess);

        let mf_cfg = build_intercept_cfg(
            post,
            min_miss_km,
            mf_bound,
            post.mf_seed_reg_weight,
            post.mf_seed_max_bound_expansions,
        );
        let mf_res = optimize_intercept_bounded(
            |dv| {
                if clears_radius(dv) {
                    compute_miss_vector_equinoctial(dv, v0, r0, target_pos, tof_dust_full_s)
                } else {
                    radius_penalty_miss(dv)
                }
            },
            dv_guess,
            &mf_cfg,
        )?;
        if conf.use_high_fidelity {
            let hf_bound = if post.hf_refine_bound_kms > 0.0 {
                post.hf_refine_bound_kms
            } else {
                default_bound
            };
            let hf_cfg = build_intercept_cfg(
                post,
                min_miss_km,
                hf_bound,
                post.hf_refine_reg_weight,
                post.hf_refine_max_bound_expansions,
            );
            let mut hf_guess = if mf_res.success {
                mf_res.dv
            } else {
                original_guess
            };
            clamp_dv_guess(&mut hf_guess, hf_bound * 0.95);
            let dust_body_force = conf.dust_body_force();
            let mut first_hf_failure = None;
            let hf_res = optimize_intercept_bounded_hf_with_model(
                |dv| {
                    if !clears_radius(dv) {
                        return HfInterceptEvaluation {
                            miss: radius_penalty_miss(dv),
                            endpoint: None,
                        };
                    }
                    let evaluation = observer.miss_vector_hf_with_endpoint(
                        dv,
                        v0,
                        r0,
                        target_pos,
                        tof_dust_full_s,
                        ctx_dust.epoch_jd,
                        dust_body_force,
                        ctx_dust,
                    );
                    match evaluation {
                        Ok(evaluation) => evaluation,
                        Err(failure) => {
                            if first_hf_failure.is_none() {
                                first_hf_failure = Some(failure);
                            }
                            // Optimizer-local poison only. Return preserved typed source below.
                            HfInterceptEvaluation {
                                miss: [f64::NAN; 3],
                                endpoint: None,
                            }
                        }
                    }
                },
                Some(&jacobian_model),
                hf_guess,
                &hf_cfg,
            )?;
            if let Some(failure) = first_hf_failure {
                return Err(failure);
            }
            Ok(hf_res.intercept.success.then_some(ReleaseControlSolution {
                dv: hf_res.intercept.dv,
                endpoint: hf_res.endpoint_for_returned_dv(),
            }))
        } else {
            Ok(mf_res.success.then_some(ReleaseControlSolution {
                dv: mf_res.dv,
                endpoint: None,
            }))
        }
    } else if conf.use_high_fidelity {
        let max_guess = (0.5_f64).min(default_bound * 0.95);
        clamp_dv_guess(&mut dv_guess, max_guess);
        let ls_cfg = build_intercept_cfg(
            post,
            min_miss_km,
            default_bound,
            BoundedInterceptConfig::default().reg_weight,
            BoundedInterceptConfig::default().max_bound_expansions,
        );
        // Same model as the seed-and-refine branch above. This site used to be
        // the one that would silently have paid three propagations per LM
        // iteration: production Part A sets `hybrid_mf_seed_hf_refine`, so it
        // is not the campaign path, and "not the campaign path today" is
        // exactly how a control goes quietly inert.
        let dust_body_force = conf.dust_body_force();
        let mut first_hf_failure = None;
        let res = optimize_intercept_bounded_hf_with_model(
            |dv| {
                if !clears_radius(dv) {
                    return HfInterceptEvaluation {
                        miss: radius_penalty_miss(dv),
                        endpoint: None,
                    };
                }
                let evaluation = observer.miss_vector_hf_with_endpoint(
                    dv,
                    v0,
                    r0,
                    target_pos,
                    tof_dust_full_s,
                    ctx_dust.epoch_jd,
                    dust_body_force,
                    ctx_dust,
                );
                match evaluation {
                    Ok(evaluation) => evaluation,
                    Err(failure) => {
                        if first_hf_failure.is_none() {
                            first_hf_failure = Some(failure);
                        }
                        // Optimizer-local poison only. Return preserved typed source below.
                        HfInterceptEvaluation {
                            miss: [f64::NAN; 3],
                            endpoint: None,
                        }
                    }
                }
            },
            Some(&jacobian_model),
            dv_guess,
            &ls_cfg,
        )?;
        if let Some(failure) = first_hf_failure {
            return Err(failure);
        }
        Ok(res.intercept.success.then_some(ReleaseControlSolution {
            dv: res.intercept.dv,
            endpoint: res.endpoint_for_returned_dv(),
        }))
    } else {
        let max_guess = (0.1_f64).min(default_bound * 0.95);
        clamp_dv_guess(&mut dv_guess, max_guess);
        let ls_cfg = build_intercept_cfg(
            post,
            min_miss_km,
            default_bound,
            BoundedInterceptConfig::default().reg_weight,
            BoundedInterceptConfig::default().max_bound_expansions,
        );
        let res = optimize_intercept_bounded(
            |dv| {
                if clears_radius(dv) {
                    compute_miss_vector_equinoctial(dv, v0, r0, target_pos, tof_dust_full_s)
                } else {
                    radius_penalty_miss(dv)
                }
            },
            dv_guess,
            &ls_cfg,
        )?;
        Ok(res.success.then_some(ReleaseControlSolution {
            dv: res.dv,
            endpoint: None,
        }))
    }
}

pub(super) fn build_ctx(
    epoch_jd: f64,
    conf: &PhysicsConfig,
    coeffs: &GlobalCoeffs,
    am_ratio: f64,
    cd: f64,
    cr: f64,
) -> Result<PlanContext, PhysicsConfigError> {
    let force_config = if conf.use_high_fidelity {
        Some(Arc::new(build_force_config(conf, am_ratio, cd, cr)?))
    } else {
        None
    };
    Ok(PlanContext::from_request(TransferRequest {
        epoch_jd,
        execution_policy: ExecutionPolicy {
            use_high_fidelity: conf.use_high_fidelity,
            require_high_fidelity: conf.require_hf_transfer_correction,
            ..Default::default()
        },
        force_config,
        packed_coeffs: coeffs.packed.clone(),
        ..TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
    }))
}

fn diagnostic_conjunction_separation(
    predicted_intercept: &StampedEciState,
    target_intercept: &StampedEciState,
    conjunction_jd: f64,
    dust_ctx: &PlanContext,
    target_ctx: &PlanContext,
    conf: &PhysicsConfig,
    target_body_force: BodyForceConfig,
    target_propagation_authority: TargetPropagationAuthority,
) -> Result<Option<f64>, TransferPropagationFailure> {
    #[cfg(test)]
    record_conjunction_diagnostic_call();
    if !conjunction_jd.is_finite() || conjunction_jd < predicted_intercept.jd {
        return Ok(None);
    }
    let dt_s = (conjunction_jd - predicted_intercept.jd) * SEC_PER_DAY;
    let dust_force = if conf.use_high_fidelity {
        conf.dust_body_force()
    } else {
        BodyForceConfig::j2(BodyRole::Dust)
    };
    let dust = propagate_stamped(predicted_intercept, dt_s, dust_force, dust_ctx)?;
    let target = match target_propagation_authority {
        TargetPropagationAuthority::HighFidelity => {
            propagate_stamped(target_intercept, dt_s, target_body_force, target_ctx)?
        }
        TargetPropagationAuthority::MfJ2 | TargetPropagationAuthority::AnalyticalKepler => {
            let mut target_equ = [0.0; 6];
            eci2equinoc_impl(&target_intercept.eci, 6, 0.0, 0.0, &mut target_equ);
            let mut target_eci = [0.0; 6];
            if target_propagation_authority == TargetPropagationAuthority::MfJ2 {
                satpy_core::equinoc_prop_j2_from_impl(&target_equ, dt_s, &mut target_eci);
            } else {
                satpy_core::equinoc_prop_from_impl(&target_equ, dt_s, &mut target_eci);
            }
            if target_eci.iter().any(|value| !value.is_finite()) {
                return Ok(None);
            }
            StampedEciState::new(target_eci, target_intercept.jd + dt_s / SEC_PER_DAY)
        }
    };
    Ok(Some(norm3(&[
        dust.eci[0] - target.eci[0],
        dust.eci[1] - target.eci[1],
        dust.eci[2] - target.eci[2],
    ])))
}

/// Solve a release vector against exactly the same R->I propagator used to
/// publish the control.  The historical MF branch used a Keplerian miss model
/// while applying J2 afterward, so even an analytically zero-control fixture
/// acquired a spurious correction.
fn solve_release_control_delta_dv<O: PostprocessLegObserver>(
    release_pre_control_state: &StampedEciState,
    target_pos: [f64; 3],
    dust_free_flight_s: f64,
    conf: &PhysicsConfig,
    post: &PostprocessConfig,
    dust_ctx: &PlanContext,
    dv_guess: [f64; 3],
    min_radius_km: f64,
    observer: &mut O,
) -> Result<Option<ReleaseControlSolution>, TransferPropagationFailure> {
    if conf.use_high_fidelity {
        return solve_intercept_delta_dv(
            &release_pre_control_state.eci,
            target_pos,
            dust_free_flight_s,
            conf,
            post,
            dust_ctx,
            dv_guess,
            post.dust_intercept_tol_km,
            min_radius_km,
            observer,
        );
    }
    // Exact J2 closure is an authoritative applied-zero control.  Test the
    // same stamped propagator used after application before asking a numerical
    // optimizer to rediscover zero; this is a physical residual tolerance,
    // not a ratio or DV relaxation.
    let zero_prediction = propagate_stamped_checked(
        release_pre_control_state,
        dust_free_flight_s,
        BodyForceConfig::j2(BodyRole::Dust),
        dust_ctx,
    )?;
    let zero_miss_km = norm3(&[
        zero_prediction.eci[0] - target_pos[0],
        zero_prediction.eci[1] - target_pos[1],
        zero_prediction.eci[2] - target_pos[2],
    ]);
    if zero_miss_km <= post.dust_intercept_tol_km {
        return Ok(Some(ReleaseControlSolution {
            dv: [0.0; 3],
            // Final propagation historically uses `conf.dust_body_force()`;
            // J2 zero-probe output is therefore not exact reuse authority.
            endpoint: None,
        }));
    }
    let mut guess = dv_guess;
    let bound = default_intercept_bound_kms(dust_free_flight_s, false);
    clamp_dv_guess(&mut guess, (0.1_f64).min(bound * 0.95));
    let cfg = build_intercept_cfg(
        post,
        post.dust_intercept_tol_km,
        bound,
        BoundedInterceptConfig::default().reg_weight,
        BoundedInterceptConfig::default().max_bound_expansions,
    );
    let first_propagation_failure = RefCell::new(None);
    let result = optimize_intercept_bounded(
        |dv| {
            let mut release_after_control = release_pre_control_state.eci;
            add_velocity(&mut release_after_control, &dv);
            let release_after_control =
                StampedEciState::new(release_after_control, release_pre_control_state.jd);
            let predicted = match propagate_stamped_checked(
                &release_after_control,
                dust_free_flight_s,
                BodyForceConfig::j2(BodyRole::Dust),
                dust_ctx,
            ) {
                Ok(predicted) => predicted,
                Err(failure) => {
                    let mut first = first_propagation_failure.borrow_mut();
                    if first.is_none() {
                        *first = Some(failure);
                    }
                    return [f64::NAN; 3];
                }
            };
            [
                predicted.eci[0] - target_pos[0],
                predicted.eci[1] - target_pos[1],
                predicted.eci[2] - target_pos[2],
            ]
        },
        guess,
        &cfg,
    );
    if let Some(failure) = first_propagation_failure.into_inner() {
        return Err(failure);
    }
    let result = result?;
    Ok(result.success.then_some(ReleaseControlSolution {
        dv: result.dv,
        endpoint: None,
    }))
}

/// Construct and apply the one physical release control at R.
///
/// The coast is always performed before the intercept solve when C > 0.
/// `max_physical_dv_kms` is a hard physical constraint: no vector is clipped
/// or silently under-counted.
fn build_physical_release_control<O: PostprocessLegObserver>(
    plan: &SummaryPlanInputs,
    target_intercept_state: &[f64; 6],
    intercept_jd: f64,
    conf: &PhysicsConfig,
    post: &PostprocessConfig,
    coeffs: &GlobalCoeffs,
    observer: &mut O,
) -> Result<(PostprocessControl, PlanContext), PostprocessControlStatus> {
    if !plan.valid {
        return Err(PostprocessControlStatus::InvalidTimeline);
    }
    let timeline = resolve_release_timeline(plan.tof_jd_start, intercept_jd, post)
        .ok_or(PostprocessControlStatus::InvalidTimeline)?;
    let fidelity = if conf.use_high_fidelity {
        PropagationFidelity::HighFidelity
    } else {
        PropagationFidelity::J2
    };
    let transfer_burn_pre_state =
        StampedEciState::new(plan.release_state, timeline.transfer_burn_jd);
    let mut canister_launch = plan.release_state;
    add_velocity(&mut canister_launch, &plan.transfer_dv);
    if !state_clears_min_radius(&canister_launch, plan.min_radius_km) {
        return Err(PostprocessControlStatus::DeterministicPhysicalInfeasible);
    }
    let canister_launch_state = StampedEciState::new(canister_launch, timeline.transfer_burn_jd);

    let canister_ctx = build_ctx(
        timeline.transfer_burn_jd,
        conf,
        coeffs,
        post.canister_am,
        post.canister_cd,
        post.canister_cr,
    )
    .map_err(PostprocessControlStatus::Configuration)?;
    let canister_force = if conf.use_high_fidelity {
        conf.canister_body_force(post.canister_am, post.canister_cd, post.canister_cr)
    } else {
        BodyForceConfig::j2(BodyRole::Canister)
    };
    let release_pre_control_state = if timeline.canister_coast_s > 0.0 {
        let coast = observer.propagate_stamped(
            &canister_launch_state,
            timeline.canister_coast_s,
            canister_force,
            &canister_ctx,
            LegPath::ReleaseCanisterCoast,
        );
        coast.map_err(PostprocessControlStatus::PropagationFailure)?
    } else {
        canister_launch_state
    };
    if !state_clears_min_radius(&release_pre_control_state.eci, plan.min_radius_km) {
        return Err(PostprocessControlStatus::DeterministicPhysicalInfeasible);
    }
    if !jd_closure_within_tolerance(release_pre_control_state.jd, timeline.release_jd) {
        return Err(PostprocessControlStatus::InvalidTimeline);
    }

    let dust_ctx = build_ctx(
        release_pre_control_state.jd,
        conf,
        coeffs,
        conf.am_ratio,
        conf.cd,
        conf.cr,
    )
    .map_err(PostprocessControlStatus::Configuration)?;
    let selected_target_state =
        StampedEciState::new(*target_intercept_state, timeline.intercept_jd);
    let target_pos = [
        target_intercept_state[0],
        target_intercept_state[1],
        target_intercept_state[2],
    ];
    let r0 = [
        release_pre_control_state.eci[0],
        release_pre_control_state.eci[1],
        release_pre_control_state.eci[2],
    ];
    let v0 = [
        release_pre_control_state.eci[3],
        release_pre_control_state.eci[4],
        release_pre_control_state.eci[5],
    ];
    let dv_linear = [
        (target_pos[0] - r0[0]) / timeline.dust_free_flight_s - v0[0],
        (target_pos[1] - r0[1]) / timeline.dust_free_flight_s - v0[1],
        (target_pos[2] - r0[2]) / timeline.dust_free_flight_s - v0[2],
    ];
    let dv_guess = match compute_lambert_guess(&r0, &v0, &target_pos, timeline.dust_free_flight_s) {
        Some(dv_lambert) if norm3(&dv_linear) >= norm3(&dv_lambert) => dv_lambert,
        _ => dv_linear,
    };
    let release_control_solution = solve_release_control_delta_dv(
        &release_pre_control_state,
        target_pos,
        timeline.dust_free_flight_s,
        conf,
        post,
        &dust_ctx,
        dv_guess,
        plan.min_radius_km,
        observer,
    )
    .map_err(PostprocessControlStatus::PropagationFailure)?
    .ok_or(PostprocessControlStatus::SolveFailure)?;
    let release_control_dv = release_control_solution.dv;
    let release_control_dv_norm = norm3(&release_control_dv);
    if !release_control_dv_norm.is_finite() || release_control_dv_norm > post.max_physical_dv_kms {
        return Err(PostprocessControlStatus::ControlSolutionConstraintViolation);
    }
    let mut release_post = release_pre_control_state.eci;
    add_velocity(&mut release_post, &release_control_dv);
    if !state_clears_min_radius(&release_post, plan.min_radius_km) {
        return Err(PostprocessControlStatus::ControlSolutionConstraintViolation);
    }
    let release_post_control_state = StampedEciState::new(release_post, timeline.release_jd);
    let dust_body_force = if conf.use_high_fidelity {
        conf.dust_body_force()
    } else {
        BodyForceConfig::j2(BodyRole::Dust)
    };
    // HF endpoint was produced lexically by the same release state, exact DV,
    // flight time, body force, and `dust_ctx` used here. The source epoch is
    // the only value reconstructed at this boundary, so require bit identity.
    let predicted_intercept_state = resolve_predicted_intercept(
        release_control_solution.endpoint,
        release_pre_control_state.jd,
        timeline.release_jd,
        &release_post_control_state,
        timeline.dust_free_flight_s,
        dust_body_force,
        &dust_ctx,
        observer,
    )
    .map_err(PostprocessControlStatus::PropagationFailure)?;
    if !state_clears_min_radius(&predicted_intercept_state.eci, plan.min_radius_km) {
        return Err(PostprocessControlStatus::ControlSolutionConstraintViolation);
    }
    if !jd_closure_within_tolerance(predicted_intercept_state.jd, timeline.intercept_jd) {
        return Err(PostprocessControlStatus::InvalidTimeline);
    }
    let status = if release_control_dv_norm <= 1e-12 {
        PostprocessControlStatus::AppliedZero
    } else {
        PostprocessControlStatus::Applied
    };
    Ok((
        PostprocessControl {
            transfer_burn_pre_state,
            canister_launch_state,
            release_pre_control_state,
            release_post_control_state,
            predicted_intercept_state,
            selected_target_state,
            release_control_dv,
            release_control_dv_norm,
            canister_tof_fraction: canister_tof_fraction(post),
            canister_coast_s: timeline.canister_coast_s,
            dust_free_flight_s: timeline.dust_free_flight_s,
            fidelity,
            status,
            conjunction_separation_km: f64::NAN,
        },
        dust_ctx,
    ))
}

/// Whether to spend two full intercept->conjunction propagations on
/// `conjunction_separation_km`.
///
/// That field is a pure diagnostic: it never changes a physical state or DV
/// (see its declaration), and no production route consumes its finite value --
/// neither the MF route through `nd_pipeline/src/physics/release_control.rs`
/// nor the strict-HF route through `nd_pipeline/src/native_hybrid.rs`. Both
/// live in `nd_pipeline`, not in this crate. A semantically absent diagnostic becomes
/// `NaN`; an actual propagation failure stays typed and fail-closed.
///
/// It is also expensive out of proportion to its value. The conjunction leg is
/// ~2.25 days against a much shorter dust free-flight leg, and profiling the
/// strict-HF lowering attributed ~21% of ALL Part A campaign work to this one
/// discarded number. Hence an explicit opt-in rather than a default: paths that
/// want the diagnostic ask for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ConjunctionDiagnostic {
    Compute,
    Skip,
}

pub(super) fn build_release_control_with_target_authority(
    plan: &SummaryPlanInputs,
    target_intercept_state: &[f64; 6],
    intercept_jd: f64,
    conjunction_jd: f64,
    conf: &PhysicsConfig,
    post: &PostprocessConfig,
    coeffs: &GlobalCoeffs,
    target_area_to_mass: Option<f64>,
    target_drag_coefficient: Option<f64>,
    target_reflectivity_coefficient: Option<f64>,
    target_propagation_authority: TargetPropagationAuthority,
    conjunction_diagnostic: ConjunctionDiagnostic,
) -> Result<(PostprocessControl, PlanContext, PlanContext), PostprocessControlStatus> {
    if intercept_jd > conjunction_jd + 1e-9 {
        return Err(PostprocessControlStatus::InvalidTimeline);
    }
    let _probe = lightyear_odeint_rs::probe::scope(lightyear_odeint_rs::probe::TAG_RELEASE_CONTROL);
    let (mut control, dust_ctx) = build_physical_release_control(
        plan,
        target_intercept_state,
        intercept_jd,
        conf,
        post,
        coeffs,
        &mut UnobservedPostprocessLeg,
    )?;
    let target_area_to_mass = target_area_to_mass.unwrap_or(0.0);
    let target_drag_coefficient = target_drag_coefficient.unwrap_or(conf.cd);
    let target_reflectivity_coefficient = target_reflectivity_coefficient.unwrap_or(conf.cr);
    let target_ctx = build_ctx(
        intercept_jd,
        conf,
        coeffs,
        target_area_to_mass,
        target_drag_coefficient,
        target_reflectivity_coefficient,
    )
    .map_err(PostprocessControlStatus::Configuration)?;
    let target_body_force = if conf.use_high_fidelity {
        BodyForceConfig::high_fidelity(
            BodyRole::DiagnosticTarget,
            target_area_to_mass,
            target_drag_coefficient,
            target_reflectivity_coefficient,
        )
    } else {
        BodyForceConfig::j2(BodyRole::DiagnosticTarget)
    };
    if conjunction_diagnostic == ConjunctionDiagnostic::Compute {
        control.conjunction_separation_km = diagnostic_conjunction_separation(
            &control.predicted_intercept_state,
            &control.selected_target_state,
            conjunction_jd,
            &dust_ctx,
            &target_ctx,
            conf,
            target_body_force,
            target_propagation_authority,
        )
        .map_err(PostprocessControlStatus::PropagationFailure)?
        .unwrap_or(f64::NAN);
    }
    Ok((control, dust_ctx, target_ctx))
}

#[cfg(any(test, feature = "bench-internal"))]
pub(super) fn build_release_control(
    plan: &SummaryPlanInputs,
    target_intercept_state: &[f64; 6],
    intercept_jd: f64,
    conjunction_jd: f64,
    conf: &PhysicsConfig,
    post: &PostprocessConfig,
    coeffs: &GlobalCoeffs,
    target_area_to_mass: Option<f64>,
    target_drag_coefficient: Option<f64>,
    target_reflectivity_coefficient: Option<f64>,
    conjunction_diagnostic: ConjunctionDiagnostic,
) -> Result<(PostprocessControl, PlanContext, PlanContext), PostprocessControlStatus> {
    let target_propagation_authority = if conf.use_high_fidelity {
        TargetPropagationAuthority::HighFidelity
    } else {
        TargetPropagationAuthority::MfJ2
    };
    build_release_control_with_target_authority(
        plan,
        target_intercept_state,
        intercept_jd,
        conjunction_jd,
        conf,
        post,
        coeffs,
        target_area_to_mass,
        target_drag_coefficient,
        target_reflectivity_coefficient,
        target_propagation_authority,
        conjunction_diagnostic,
    )
}

/// Recompute one physical release control for an explicit canister fraction.
///
/// Unlike the legacy configured path, an explicit v2 fraction is never
/// clamped or relabelled.  Every accepted value produces a fresh L->R coast
/// and R->I solve from the replay-verified transfer state using the fixed
/// configured canister ballistics.
fn postprocess_config_at_fraction(
    post: &PostprocessConfig,
    fraction: f64,
) -> Result<PostprocessConfig, PostprocessControlStatus> {
    if !fraction.is_finite() || !(0.0..1.0).contains(&fraction) {
        return Err(PostprocessControlStatus::InvalidTimeline);
    }
    let mut fraction_post = post.clone();
    fraction_post.canister_tof_fraction = fraction;
    Ok(fraction_post)
}

#[cfg(feature = "solver-qualification")]
pub(super) struct ObservedReleaseControlCoreRequest<'a> {
    pub(super) plan: &'a SummaryPlanInputs,
    pub(super) target_intercept_state: &'a [f64; 6],
    pub(super) intercept_jd: f64,
    pub(super) conf: &'a PhysicsConfig,
    pub(super) post: &'a PostprocessConfig,
    pub(super) coeffs: &'a GlobalCoeffs,
    pub(super) fraction: f64,
}

#[expect(
    clippy::too_many_arguments,
    reason = "one private fraction route keeps physical, target-authority, and diagnostic inputs explicit"
)]
pub(super) fn build_release_control_at_fraction(
    plan: &SummaryPlanInputs,
    target_intercept_state: &[f64; 6],
    intercept_jd: f64,
    conjunction_jd: f64,
    conf: &PhysicsConfig,
    post: &PostprocessConfig,
    coeffs: &GlobalCoeffs,
    target_area_to_mass: Option<f64>,
    target_drag_coefficient: Option<f64>,
    target_reflectivity_coefficient: Option<f64>,
    fraction: f64,
    target_propagation_authority: TargetPropagationAuthority,
    conjunction_diagnostic: ConjunctionDiagnostic,
) -> Result<(PostprocessControl, PlanContext, PlanContext), PostprocessControlStatus> {
    let fraction_post = postprocess_config_at_fraction(post, fraction)?;
    build_release_control_with_target_authority(
        plan,
        target_intercept_state,
        intercept_jd,
        conjunction_jd,
        conf,
        &fraction_post,
        coeffs,
        target_area_to_mass,
        target_drag_coefficient,
        target_reflectivity_coefficient,
        target_propagation_authority,
        conjunction_diagnostic,
    )
}

/// Recompute one physical release control while observing only actual scalar
/// legs. This feature-only diagnostic shares the canonical control core.
#[cfg(feature = "solver-qualification")]
pub(super) fn build_release_control_at_fraction_observed(
    request: &ObservedReleaseControlCoreRequest<'_>,
    trace: &mut QualificationLegTrace,
) -> Result<(PostprocessControl, PlanContext), PostprocessControlStatus> {
    let fraction_post = postprocess_config_at_fraction(request.post, request.fraction)?;
    build_physical_release_control(
        request.plan,
        request.target_intercept_state,
        request.intercept_jd,
        request.conf,
        &fraction_post,
        request.coeffs,
        trace,
    )
}

#[cfg(feature = "solver-qualification")]
pub(super) struct ObservedMaterializationRequest<'a> {
    pub(super) control: &'a PostprocessControl,
    pub(super) ctx_dust: &'a PlanContext,
    pub(super) conf: &'a PhysicsConfig,
    pub(super) post: &'a PostprocessConfig,
    pub(super) split_alpha: Option<f64>,
    pub(super) split_axis: Option<[f64; 6]>,
    pub(super) release_covariance: Option<&'a [[f64; 6]; 6]>,
    pub(super) release_distribution: Option<AuthoritativeReleaseDistribution>,
}

#[cfg(test)]
pub(super) fn compute_corrected_dust_state_summary(
    plan: &SummaryPlanInputs,
    target_intercept_state: &[f64; 6],
    intercept_jd: f64,
    conjunction_jd: f64,
    conf: &PhysicsConfig,
    post: &PostprocessConfig,
    coeffs: &GlobalCoeffs,
    target_area_to_mass: Option<f64>,
    target_drag_coefficient: Option<f64>,
    target_reflectivity_coefficient: Option<f64>,
    scratch: Option<&mut TransferPostprocessScratch>,
) -> Result<Option<PostprocessDustSummary>, PostprocessDistributionStatus> {
    let runtime_settings = ResolvedPostprocessRuntimeSettings::from_postprocess_config(post);
    compute_corrected_dust_state_summary_with_runtime(
        plan,
        target_intercept_state,
        intercept_jd,
        conjunction_jd,
        conf,
        post,
        coeffs,
        runtime_settings,
        target_area_to_mass,
        target_drag_coefficient,
        target_reflectivity_coefficient,
        scratch,
    )
}

#[cfg(any(test, feature = "bench-internal"))]
pub(super) fn compute_corrected_dust_state_summary_with_runtime(
    plan: &SummaryPlanInputs,
    target_intercept_state: &[f64; 6],
    intercept_jd: f64,
    conjunction_jd: f64,
    conf: &PhysicsConfig,
    post: &PostprocessConfig,
    coeffs: &GlobalCoeffs,
    runtime: ResolvedPostprocessRuntimeSettings,
    target_area_to_mass: Option<f64>,
    target_drag_coefficient: Option<f64>,
    target_reflectivity_coefficient: Option<f64>,
    scratch: Option<&mut TransferPostprocessScratch>,
) -> Result<Option<PostprocessDustSummary>, PostprocessDistributionStatus> {
    if !plan.valid {
        return Ok(None);
    }
    if intercept_jd > conjunction_jd + 1e-9 {
        return Ok(None);
    }

    let (control, ctx_dust, _) = build_release_control(
        plan,
        target_intercept_state,
        intercept_jd,
        conjunction_jd,
        conf,
        post,
        coeffs,
        target_area_to_mass,
        target_drag_coefficient,
        target_reflectivity_coefficient,
        ConjunctionDiagnostic::Skip,
    )
    .map_err(distribution_status_from_release_control_status)?;
    let tof_dust = control.dust_free_flight_s;
    let delta_dv = control.release_control_dv;
    // Components start at the physical pre-control release state and receive
    // the returned vector exactly once below.  Starting post-control here
    // would add the same vector a second time during materialization.
    let base_mean = control.release_pre_control_state.eci;
    let Some(cov6) = release_covariance_from_conf(conf, &base_mean) else {
        return Ok(None);
    };
    let split_cfg = SplitConfig::default();
    let num_dists = runtime.num_dists;

    let criterion = conf.splitting_criterion.as_str();
    let Ok(axis) = select_split_axis_strict(criterion, tof_dust) else {
        return Ok(None);
    };

    let split_result = axis.map_or_else(
        || {
            dominant_eigenvector6(&cov6).map_or_else(
                || split_gaussian_no_axis(&base_mean, &cov6, num_dists, Some(-1.0), &split_cfg),
                |eigen_axis| {
                    split_gaussian_along_axis(
                        &base_mean,
                        &cov6,
                        &eigen_axis,
                        num_dists,
                        -1.0,
                        &split_cfg,
                    )
                },
            )
        },
        |explicit_axis| {
            split_gaussian_along_axis(
                &base_mean,
                &cov6,
                &explicit_axis,
                num_dists,
                -1.0,
                &split_cfg,
            )
        },
    );

    let dust_mean = if let Some(scratch) = scratch {
        scratch.last_batch_len = scratch.last_batch_len.max(split_result.means.len());
        scratch.weights.resize(split_result.weights.len(), 0.0);
        scratch.comp_means.clear();
        scratch
            .weights
            .as_mut_slice()
            .copy_from_slice(&split_result.weights);
        let mut corrected_component_means = std::mem::take(&mut scratch.corrected_component_means);
        corrected_component_means.resize(split_result.means.len(), [0.0; 6]);
        normalize_weights(scratch.weights.as_mut_slice());
        for (adjusted, mean) in corrected_component_means
            .iter_mut()
            .zip(split_result.means.iter())
        {
            *adjusted = *mean;
            add_velocity(adjusted, &delta_dv);
        }

        propagate_component_means_ukf_batch(
            corrected_component_means.as_slice(),
            &split_result.covariances,
            tof_dust,
            Some(&ctx_dust),
            scratch,
        )
        .map_err(distribution_status_from_ukf_failure)?;
        scratch.corrected_component_means = corrected_component_means;

        let mut dust_mean = [0.0; 6];
        for (w, mean) in scratch.weights.iter().zip(scratch.comp_means.iter()) {
            for (accumulator, component) in dust_mean.iter_mut().zip(mean.iter()) {
                *accumulator += w * component;
            }
        }

        dust_mean
    } else {
        let mut corrected_component_means: SmallVec<[[f64; 6]; MAX_DUST_COMPONENTS]> =
            SmallVec::with_capacity(split_result.means.len());
        corrected_component_means.resize(split_result.means.len(), [0.0; 6]);
        for (adjusted, mean) in corrected_component_means
            .iter_mut()
            .zip(split_result.means.iter())
        {
            *adjusted = *mean;
            add_velocity(adjusted, &delta_dv);
        }
        let mut local_weights: SmallVec<[f64; MAX_DUST_COMPONENTS]> =
            SmallVec::with_capacity(split_result.weights.len());
        let mut local_comp_means: SmallVec<[[f64; 6]; MAX_DUST_COMPONENTS]> =
            SmallVec::with_capacity(corrected_component_means.len());
        local_weights.extend_from_slice(&split_result.weights);
        normalize_weights(local_weights.as_mut_slice());

        for (mean, cov) in corrected_component_means
            .iter()
            .zip(split_result.covariances.iter())
        {
            let (mean_prop, _cov_prop) =
                propagate_component_ukf_checked(mean, cov, tof_dust, Some(&ctx_dust))
                    .map_err(distribution_status_from_ukf_failure)?;
            local_comp_means.push(mean_prop);
        }

        let mut dust_mean = [0.0; 6];
        for (w, mean) in local_weights.iter().zip(local_comp_means.iter()) {
            for (accumulator, component) in dust_mean.iter_mut().zip(mean.iter()) {
                *accumulator += w * component;
            }
        }

        dust_mean
    };

    let correction_dv_norm = control.release_control_dv_norm;
    Ok(Some(PostprocessDustSummary {
        dust_mean,
        correction_dv_norm,
    }))
}

/// Complete input bundle for materializing the authoritative dust
/// distribution.  Keeping this as one request prevents callers from
/// accidentally decoupling the release, UKF, and split authorities.
pub(super) struct CorrectedDustStateRequest<'a> {
    pub(super) plan: &'a SummaryPlanInputs,
    pub(super) target_intercept_state: &'a [f64; 6],
    pub(super) intercept_jd: f64,
    pub(super) conjunction_jd: f64,
    pub(super) conf: &'a PhysicsConfig,
    pub(super) post: &'a PostprocessConfig,
    pub(super) coeffs: &'a GlobalCoeffs,
    pub(super) split_alpha: Option<f64>,
    pub(super) split_axis: Option<[f64; 6]>,
    pub(super) release_covariance: Option<&'a [[f64; 6]; 6]>,
    pub(super) release_distribution: Option<AuthoritativeReleaseDistribution>,
}

pub(super) fn compute_corrected_dust_state(
    request: CorrectedDustStateRequest<'_>,
) -> Result<PostprocessDustDistribution, PostprocessDistributionStatus> {
    let CorrectedDustStateRequest {
        plan,
        target_intercept_state,
        intercept_jd,
        conjunction_jd,
        conf,
        post,
        coeffs,
        split_alpha,
        split_axis,
        release_covariance,
        release_distribution,
    } = request;
    if !plan.valid {
        return Err(PostprocessDistributionStatus::InvalidPlan);
    }
    if intercept_jd > conjunction_jd + 1e-9 {
        return Err(PostprocessDistributionStatus::InvalidTimeline);
    }

    let (control, ctx_dust) = build_physical_release_control(
        plan,
        target_intercept_state,
        intercept_jd,
        conf,
        post,
        coeffs,
        &mut UnobservedPostprocessLeg,
    )
    .map_err(distribution_status_from_release_control_status)?;
    materialize_corrected_dust_distribution(
        &control,
        &ctx_dust,
        conf,
        post,
        split_alpha,
        split_axis,
        release_covariance,
        release_distribution,
        &mut UnobservedPostprocessLeg,
    )
}

pub(super) fn materialize_corrected_dust_distribution<O: PostprocessLegObserver>(
    control: &PostprocessControl,
    ctx_dust: &PlanContext,
    conf: &PhysicsConfig,
    post: &PostprocessConfig,
    split_alpha: Option<f64>,
    split_axis: Option<[f64; 6]>,
    release_covariance: Option<&[[f64; 6]; 6]>,
    release_distribution: Option<AuthoritativeReleaseDistribution>,
    observer: &mut O,
) -> Result<PostprocessDustDistribution, PostprocessDistributionStatus> {
    let tof_dust = control.dust_free_flight_s;
    // R is post-control authority.  Construct its RTN covariance and split
    // directly at R; rebuilding from pre-control state rotates the covariance
    // frame and silently diverges from Python's authoritative Pc cloud.
    let base_mean = control.release_post_control_state.eci;
    let generated_covariance;
    let cov6 = if let Some(covariance) = release_covariance {
        if !covariance.iter().flatten().all(|value| value.is_finite()) {
            return Err(PostprocessDistributionStatus::InvalidReleaseCovariance);
        }
        covariance
    } else {
        generated_covariance = release_covariance_from_conf(conf, &base_mean)
            .ok_or(PostprocessDistributionStatus::InvalidReleaseCovariance)?;
        &generated_covariance
    };

    let split_cfg = SplitConfig::default();
    let num_dists = post.gmm_components.clamp(1, MAX_DUST_COMPONENTS);

    let criterion = conf.splitting_criterion.as_str();
    let axis = if let Some(axis) = split_axis {
        let norm = axis.iter().map(|value| value * value).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(PostprocessDistributionStatus::InvalidSplitAxis);
        }
        Some(axis)
    } else {
        select_split_axis_strict(criterion, tof_dust)
            .map_err(|_| PostprocessDistributionStatus::InvalidSplitAxis)?
    };
    let alpha = split_alpha.unwrap_or(-1.0);
    if !alpha.is_finite() {
        return Err(PostprocessDistributionStatus::InvalidSplitAlpha);
    }

    let (release_means, release_covariances, release_weights, release_sigma_points) =
        if let Some(release) = release_distribution {
            if release.means.is_empty()
                || release.means.len() > MAX_DUST_COMPONENTS
                || release.means.len() != release.covariances.len()
                || release.means.len() != release.weights.len()
                || !release
                    .means
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite())
                || !release
                    .covariances
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|value| value.is_finite())
                || !release.weights.iter().all(|value| value.is_finite())
            {
                return Err(PostprocessDistributionStatus::InvalidReleaseDistribution);
            }
            let expected_sigma_points = release
                .means
                .len()
                .checked_mul(dust_ukf_rs::NUM_SIGMA)
                .ok_or(PostprocessDistributionStatus::ArithmeticOverflow)?;
            if release
                .sigma_points
                .as_ref()
                .is_some_and(|points| points.len() != expected_sigma_points)
            {
                return Err(PostprocessDistributionStatus::InvalidReleaseDistribution);
            }
            (
                release.means,
                release.covariances,
                release.weights,
                release.sigma_points,
            )
        } else {
            let split_result = axis.map_or_else(
                || {
                    dominant_eigenvector6(cov6).map_or_else(
                        || {
                            split_gaussian_no_axis(
                                &base_mean,
                                cov6,
                                num_dists,
                                Some(alpha),
                                &split_cfg,
                            )
                        },
                        |eigen_axis| {
                            split_gaussian_along_axis(
                                &base_mean,
                                cov6,
                                &eigen_axis,
                                num_dists,
                                alpha,
                                &split_cfg,
                            )
                        },
                    )
                },
                |explicit_axis| {
                    split_gaussian_along_axis(
                        &base_mean,
                        cov6,
                        &explicit_axis,
                        num_dists,
                        alpha,
                        &split_cfg,
                    )
                },
            );
            (
                copy_split_result_slice(&split_result.means)?,
                copy_split_result_slice(&split_result.covariances)?,
                copy_split_result_slice(&split_result.weights)?,
                None,
            )
        };

    let mut weights: SmallVec<[f64; MAX_DUST_COMPONENTS]> = SmallVec::from_slice(&release_weights);
    normalize_weights(&mut weights);

    let mut comp_means: SmallVec<[[f64; 6]; MAX_DUST_COMPONENTS]> =
        SmallVec::with_capacity(release_means.len());
    let mut comp_covs: SmallVec<[[[f64; 6]; 6]; MAX_DUST_COMPONENTS]> =
        SmallVec::with_capacity(release_covariances.len());
    let propagation = propagate_components_ukf_full_batch_observed_by(
        &release_means,
        &release_covariances,
        release_sigma_points.as_deref(),
        tof_dust,
        Some(ctx_dust),
        observer,
    );
    let UkfFullBatchOutput {
        propagated_components: propagated,
        #[cfg(test)]
        propagated_sigma_points,
        // Dropped immediately outside tests: retention is test-only (see the
        // field doc on `PostprocessDustDistribution`).
        #[cfg(not(test))]
            propagated_sigma_points: _,
    } = propagation.map_err(distribution_status_from_ukf_failure)?;
    for (mean_prop, cov_prop) in propagated {
        comp_means.push(mean_prop);
        comp_covs.push(cov_prop);
    }

    let mut dust_mean = [0.0; 6];
    for (w, mean) in weights.iter().zip(comp_means.iter()) {
        for (accumulator, component) in dust_mean.iter_mut().zip(mean.iter()) {
            *accumulator += w * component;
        }
    }

    let correction_dv_norm = control.release_control_dv_norm;
    Ok(PostprocessDustDistribution {
        release_jd: control.release_post_control_state.jd,
        dust_free_flight_s: tof_dust,
        dust_mean,
        weights,
        release_comp_means: SmallVec::from_slice(&release_means),
        release_comp_covs: SmallVec::from_slice(&release_covariances),
        comp_means,
        comp_covs,
        #[cfg(test)]
        propagated_sigma_points,
        correction_dv_norm,
    })
}

#[cfg(feature = "solver-qualification")]
pub(super) fn materialize_corrected_dust_distribution_observed(
    request: ObservedMaterializationRequest<'_>,
    trace: &mut QualificationLegTrace,
) -> Result<PostprocessDustDistribution, PostprocessDistributionStatus> {
    let ObservedMaterializationRequest {
        control,
        ctx_dust,
        conf,
        post,
        split_alpha,
        split_axis,
        release_covariance,
        release_distribution,
    } = request;
    materialize_corrected_dust_distribution(
        control,
        ctx_dust,
        conf,
        post,
        split_alpha,
        split_axis,
        release_covariance,
        release_distribution,
        trace,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ukf_ephemeris_failure_keeps_legacy_display_and_full_typed_chain() {
        use lightyear_odeint_rs::precomputed_ephem::EphemerisCoverageError;
        use lightyear_odeint_rs::session::VariableFinalNativeError;

        let variable_final = std::sync::Arc::new(VariableFinalNativeError::Ephemeris {
            row: 11,
            source: anyhow::Error::new(EphemerisCoverageError::NonFiniteArc {
                jd_a: f64::NAN,
                jd_b: 2_460_000.5,
            }),
        });
        let failure = UkfPropagationFailure::Ephemeris {
            row: 11,
            message: "fixture coverage failure".to_owned(),
            source: variable_final,
        };
        let status = distribution_status_from_ukf_failure(failure.clone());

        assert_eq!(
            status.to_string(),
            "postprocess UKF ephemeris is unavailable"
        );
        assert_eq!(
            status,
            PostprocessDistributionStatus::UkfPropagationFailure(failure)
        );
        let ukf_source =
            std::error::Error::source(&status).expect("distribution must retain UKF source");
        let variable_final_source = ukf_source
            .source()
            .expect("UKF must retain variable-final source");
        assert!(
            matches!(
                variable_final_source.downcast_ref::<VariableFinalNativeError>(),
                Some(VariableFinalNativeError::Ephemeris { row: 11, .. })
            ),
            "unexpected source: {variable_final_source}"
        );
        let ephemeris_source = variable_final_source
            .source()
            .expect("variable-final failure must retain ephemeris cause");
        assert!(
            ephemeris_source
                .downcast_ref::<EphemerisCoverageError>()
                .is_some(),
            "unexpected source: {ephemeris_source}"
        );
    }

    #[test]
    fn release_control_propagation_display_uses_typed_error() {
        let status = PostprocessDistributionStatus::ReleaseControl(
            PostprocessControlStatus::PropagationFailure(TransferPropagationFailure::Authority),
        );

        assert_eq!(
            status.to_string(),
            "postprocess propagation: transfer propagation authority mismatch"
        );
    }

    #[test]
    fn strict_hf_stamped_propagation_retains_missing_asset_failure() {
        let mut request =
            TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        request.epoch_jd = 2_460_000.5;
        request.execution_policy = ExecutionPolicy {
            use_high_fidelity: true,
            require_high_fidelity: true,
            ..ExecutionPolicy::default()
        };
        let ctx = PlanContext::from_request(request);
        let state = StampedEciState::new([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], ctx.epoch_jd);

        assert!(matches!(
            propagate_stamped_checked(
                &state,
                60.0,
                BodyForceConfig::high_fidelity(BodyRole::Dust, 0.01, 2.2, 1.3),
                &ctx,
            ),
            Err(crate::evaluate::TransferPropagationFailure::MissingHighFidelityAssets)
        ));
    }

    #[test]
    fn diagnostic_conjunction_retains_high_fidelity_propagation_failure() {
        let conf = PhysicsConfig {
            use_high_fidelity: true,
            require_hf_transfer_correction: true,
            ..PhysicsConfig::default()
        };
        let mut request =
            TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        request.epoch_jd = 2_460_000.5;
        request.execution_policy = ExecutionPolicy {
            use_high_fidelity: true,
            require_high_fidelity: true,
            ..ExecutionPolicy::default()
        };
        let ctx = PlanContext::from_request(request);
        let state = StampedEciState::new([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], ctx.epoch_jd);
        let target_force =
            BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 0.01, 2.2, 1.3);

        assert!(matches!(
            diagnostic_conjunction_separation(
                &state,
                &state,
                state.jd + 60.0 / SEC_PER_DAY,
                &ctx,
                &ctx,
                &conf,
                target_force,
                TargetPropagationAuthority::HighFidelity,
            ),
            Err(TransferPropagationFailure::MissingHighFidelityAssets)
        ));
    }

    #[test]
    fn unsupported_integrator_is_typed_for_hf_and_inert_for_j2_contexts() {
        let (plan, target_i, intercept_jd, mut conf, post, coeffs) =
            release_control_fixture(0.0, 7200.0).expect("release-control fixture");
        conf.method = "unsupported-integrator".to_owned();

        assert!(matches!(
            build_ctx(
                plan.tof_jd_start,
                &conf,
                &coeffs,
                conf.am_ratio,
                conf.cd,
                conf.cr,
            ),
            Ok(ctx) if ctx.force_config.is_none()
        ));

        conf.use_high_fidelity = true;
        assert!(matches!(
            build_ctx(
                plan.tof_jd_start,
                &conf,
                &coeffs,
                conf.am_ratio,
                conf.cd,
                conf.cr,
            ),
            Err(crate::py_config::PhysicsConfigError::UnsupportedIntegratorMethod)
        ));
        assert!(matches!(
            build_physical_release_control(
                &plan,
                &target_i,
                intercept_jd,
                &conf,
                &post,
                &coeffs,
                &mut UnobservedPostprocessLeg,
            ),
            Err(PostprocessControlStatus::Configuration(
                crate::py_config::PhysicsConfigError::UnsupportedIntegratorMethod
            ))
        ));
    }

    #[test]
    fn strict_hf_intercept_failure_reaches_release_control_solver() {
        let conf = PhysicsConfig {
            use_high_fidelity: true,
            require_hf_transfer_correction: true,
            ..PhysicsConfig::default()
        };
        let mut request =
            TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        request.epoch_jd = 2_460_000.5;
        request.execution_policy = ExecutionPolicy {
            use_high_fidelity: true,
            require_high_fidelity: true,
            ..ExecutionPolicy::default()
        };
        let ctx = PlanContext::from_request(request);

        assert!(matches!(
            solve_intercept_delta_dv(
                &[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
                [7000.0, 1.0, 0.0],
                60.0,
                &conf,
                &super::super::session::default_postprocess_config(),
                &ctx,
                [0.0; 3],
                1e-6,
                1.0,
                &mut UnobservedPostprocessLeg,
            ),
            Err(TransferPropagationFailure::MissingHighFidelityAssets)
        ));
    }

    fn assert_distribution_bits_eq(
        expected: &PostprocessDustDistribution,
        actual: &PostprocessDustDistribution,
    ) {
        assert_eq!(expected.release_jd.to_bits(), actual.release_jd.to_bits());
        assert_eq!(
            expected.dust_free_flight_s.to_bits(),
            actual.dust_free_flight_s.to_bits()
        );
        assert_eq!(
            expected.correction_dv_norm.to_bits(),
            actual.correction_dv_norm.to_bits()
        );
        assert_eq!(expected.weights.len(), actual.weights.len());
        assert_eq!(
            expected.release_comp_means.len(),
            actual.release_comp_means.len()
        );
        assert_eq!(
            expected.release_comp_covs.len(),
            actual.release_comp_covs.len()
        );
        assert_eq!(expected.comp_means.len(), actual.comp_means.len());
        assert_eq!(expected.comp_covs.len(), actual.comp_covs.len());
        assert_eq!(
            expected.propagated_sigma_points.len(),
            actual.propagated_sigma_points.len()
        );

        // The length pins above stop `zip` from silently truncating one side
        // against the other, but they are satisfied by 0 == 0. If both
        // distributions came back degenerate, every `zip` below would yield no
        // pairs and this helper would certify bit-equality having compared no
        // bits. Every caller compares real propagated distributions, so each of
        // these collections must be populated.
        assert!(
            !expected.weights.is_empty(),
            "distribution must carry sigma weights to compare"
        );
        assert!(
            !expected.comp_means.is_empty(),
            "distribution must carry component means to compare"
        );
        assert!(
            !expected.comp_covs.is_empty(),
            "distribution must carry component covariances to compare"
        );
        assert!(
            !expected.propagated_sigma_points.is_empty(),
            "distribution must carry propagated sigma points to compare"
        );

        for (expected, actual) in expected.dust_mean.iter().zip(actual.dust_mean.iter()) {
            assert_eq!(expected.to_bits(), actual.to_bits());
        }
        for (expected, actual) in expected.weights.iter().zip(actual.weights.iter()) {
            assert_eq!(expected.to_bits(), actual.to_bits());
        }
        for (expected, actual) in expected
            .release_comp_means
            .iter()
            .zip(actual.release_comp_means.iter())
            .chain(expected.comp_means.iter().zip(actual.comp_means.iter()))
            .chain(
                expected
                    .propagated_sigma_points
                    .iter()
                    .zip(actual.propagated_sigma_points.iter()),
            )
        {
            for (expected, actual) in expected.iter().zip(actual.iter()) {
                assert_eq!(expected.to_bits(), actual.to_bits());
            }
        }
        for (expected, actual) in expected
            .release_comp_covs
            .iter()
            .zip(actual.release_comp_covs.iter())
            .chain(expected.comp_covs.iter().zip(actual.comp_covs.iter()))
        {
            for (expected, actual) in expected.iter().flatten().zip(actual.iter().flatten()) {
                assert_eq!(expected.to_bits(), actual.to_bits());
            }
        }
    }

    fn assert_physical_control_bits_eq(expected: &PostprocessControl, actual: &PostprocessControl) {
        for (expected, actual) in [
            (
                &expected.transfer_burn_pre_state,
                &actual.transfer_burn_pre_state,
            ),
            (
                &expected.canister_launch_state,
                &actual.canister_launch_state,
            ),
            (
                &expected.release_pre_control_state,
                &actual.release_pre_control_state,
            ),
            (
                &expected.release_post_control_state,
                &actual.release_post_control_state,
            ),
            (
                &expected.predicted_intercept_state,
                &actual.predicted_intercept_state,
            ),
            (
                &expected.selected_target_state,
                &actual.selected_target_state,
            ),
        ] {
            assert_eq!(expected.jd.to_bits(), actual.jd.to_bits());
            for (expected, actual) in expected.eci.iter().zip(actual.eci.iter()) {
                assert_eq!(expected.to_bits(), actual.to_bits());
            }
        }
        for (expected, actual) in expected
            .release_control_dv
            .iter()
            .zip(actual.release_control_dv.iter())
        {
            assert_eq!(expected.to_bits(), actual.to_bits());
        }
        assert_eq!(
            expected.release_control_dv_norm.to_bits(),
            actual.release_control_dv_norm.to_bits()
        );
        assert_eq!(
            expected.canister_tof_fraction.to_bits(),
            actual.canister_tof_fraction.to_bits()
        );
        assert_eq!(
            expected.canister_coast_s.to_bits(),
            actual.canister_coast_s.to_bits()
        );
        assert_eq!(
            expected.dust_free_flight_s.to_bits(),
            actual.dust_free_flight_s.to_bits()
        );
        assert_eq!(expected.fidelity, actual.fidelity);
        assert_eq!(expected.status, actual.status);
    }

    fn release_control_fixture(
        fraction: f64,
        transfer_to_intercept_s: f64,
    ) -> anyhow::Result<(
        SummaryPlanInputs,
        [f64; 6],
        f64,
        PhysicsConfig,
        PostprocessConfig,
        GlobalCoeffs,
    )> {
        let intercept_jd = 2_460_000.5 + transfer_to_intercept_s / SEC_PER_DAY;
        let plan = SummaryPlanInputs {
            valid: true,
            release_state: [7000.0, 0.0, 0.0, 0.0, 7.5, 1.0],
            transfer_dv: [0.0; 3],
            tof_jd_start: 2_460_000.5,
            min_radius_km: satpy_core::RE,
        };
        let conf = PhysicsConfig {
            use_high_fidelity: false,
            require_hf_transfer_correction: false,
            ..PhysicsConfig::default()
        };
        let post = PostprocessConfig {
            dust_phase_tof_s: 7200.0,
            canister_tof_fraction: fraction,
            fix_ls_skip_tol: 1.0,
            dust_intercept_tol_km: 1e-6,
            max_physical_dv_kms: 1.0,
            ..super::super::session::default_postprocess_config()
        };
        let coeffs = GlobalCoeffs {
            packed: None,
            missing: true,
        };
        let timeline = resolve_release_timeline(plan.tof_jd_start, intercept_jd, &post)
            .ok_or_else(|| std::io::Error::other("valid release timeline"))?;
        let canister_ctx = build_ctx(
            plan.tof_jd_start,
            &conf,
            &coeffs,
            post.canister_am,
            post.canister_cd,
            post.canister_cr,
        )?;
        let canister_l = StampedEciState::new(plan.release_state, plan.tof_jd_start);
        let release_r = propagate_stamped(
            &canister_l,
            timeline.canister_coast_s,
            BodyForceConfig::j2(BodyRole::Canister),
            &canister_ctx,
        )?;
        let dust_ctx = build_ctx(
            timeline.release_jd,
            &conf,
            &coeffs,
            conf.am_ratio,
            conf.cd,
            conf.cr,
        )?;
        let target_i = propagate_stamped(
            &release_r,
            timeline.dust_free_flight_s,
            BodyForceConfig::j2(BodyRole::Dust),
            &dust_ctx,
        )?
        .eci;
        Ok((plan, target_i, intercept_jd, conf, post, coeffs))
    }

    /// A strict-HF release-control fixture shaped like the campaign's.
    ///
    /// `release_control_fixture` above is J2 with no gravity assets, so it can
    /// never reach the native sigma batch. This one carries packed
    /// coefficients, `use_high_fidelity`, and `hybrid_mf_seed_hf_refine`, which
    /// together are what the Part A lowering compiles.
    fn strict_hf_release_fixture(
        release_state: [f64; 6],
        transfer_to_intercept_s: f64,
        fraction: f64,
    ) -> anyhow::Result<(
        SummaryPlanInputs,
        [f64; 6],
        f64,
        PhysicsConfig,
        PostprocessConfig,
        GlobalCoeffs,
    )> {
        use lightyear_odeint_rs::types::ForceFlags;

        // The campaign's own coefficient file at the campaign's own order, so
        // the identity is measured against the field production flies.
        const DIR_R6_D15: &[u8] =
            include_bytes!("../../data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

        // The sealed lowering, read rather than restated: a tolerance,
        // `dt_max`, stepper, or force-flag change must reach this fixture, or
        // the identity below is measured on a model production does not run.
        let sealed = nd_config::CompiledPartAScienceV1::part_a_v1();
        let hybrid = sealed.hybrid();
        let mut force_flags = 0;
        if hybrid.force_drag {
            force_flags |= ForceFlags::DRAG;
        }
        if hybrid.force_srp {
            force_flags |= ForceFlags::SRP;
        }
        if hybrid.force_sun {
            force_flags |= ForceFlags::SUN_GRAVITY;
        }
        if hybrid.force_moon {
            force_flags |= ForceFlags::MOON_GRAVITY;
        }
        let order = hybrid.gravity_order;
        // `packed_constants_from_bytes`, NOT `load_constants_from_bytes`: the
        // latter publishes into `GLOBAL_COEFFS`, and two tests in this same lib
        // binary read that global — `session.rs` asserts it is unset after a
        // rejected integrator, and `strict_session_pack_and_identity_ignore_
        // preloaded_global` installs a hostile pack and asserts nothing mutates
        // it. Cargo runs these on parallel threads, so installing a global here
        // is a race that passes until it does not.
        let packed = Arc::new(
            lightyear_odeint_rs::packed_constants_from_bytes(DIR_R6_D15, order)
                .map_err(std::io::Error::other)?
                .as_ref()
                .clone(),
        );
        let coeffs = GlobalCoeffs {
            packed: Some(packed),
            missing: false,
        };
        let conf = PhysicsConfig {
            use_high_fidelity: hybrid.use_high_fidelity,
            require_hf_transfer_correction: hybrid.require_hf_transfer_correction,
            sph_order: order,
            force_flags,
            atm_model: hybrid.atmosphere_model,
            am_ratio: hybrid.dust_am_ratio,
            cd: hybrid.dust_cd,
            cr: hybrid.dust_cr,
            // Left dynamic on purpose: production resolves Sun and Moon from
            // the ephemeris tables, which is the only regime in which the two
            // propagation routes can disagree about an arc anchor.
            sun_pos: None,
            moon_pos: None,
            dt_max: hybrid.dt_max_s,
            tolerance: hybrid.tolerance,
            method: hybrid.integrator_method.to_owned(),
            splitting_criterion: sealed
                .mf_transfer()
                .native_policy
                .splitting_criterion
                .to_owned(),
            ..PhysicsConfig::default()
        };
        // Part A v3 scenario `t0_utc` = 2026-08-17T17:24:29Z, as
        // `JD(2026-08-17T00:00:00Z) + 62_669 s`.
        //
        // This was 2_460_310.5 (2024-01-01) under a comment claiming the JB2008
        // driver tables covered it. They did until the v3 persistence runtime
        // narrowed authorized coverage to
        // 2026-08-15T11:24:29Z..2026-08-31T17:24:29Z
        // (`assets/reference/atmosphere/jb2008/part_a_v3_persistence_v1/manifest.json`).
        // After that every corpus row failed its arc lookup and was swallowed
        // by the `continue` guards below, so the loop compared zero rows.
        let epoch_jd = 2_461_269.5 + 62_669.0 / 86_400.0;
        let intercept_jd = epoch_jd + transfer_to_intercept_s / SEC_PER_DAY;
        let plan = SummaryPlanInputs {
            valid: true,
            release_state,
            transfer_dv: [0.0; 3],
            tof_jd_start: epoch_jd,
            min_radius_km: satpy_core::RE,
        };
        let post = PostprocessConfig {
            dust_phase_tof_s: transfer_to_intercept_s * 0.5,
            canister_tof_fraction: fraction,
            canister_am: hybrid.canister_am,
            canister_cd: hybrid.canister_cd,
            canister_cr: hybrid.canister_cr,
            fix_ls_max_nfev: hybrid.fix_ls_max_nfev,
            fix_ls_tol: hybrid.fix_ls_tol,
            fix_ls_skip_tol: hybrid.fix_ls_skip_tol,
            dust_intercept_tol_km: hybrid.dust_intercept_tol_km,
            max_physical_dv_kms: hybrid.max_physical_dv_kms,
            mf_seed_bound_kms: hybrid.mf_seed_bound_kms,
            hf_refine_bound_kms: hybrid.hf_refine_bound_kms,
            mf_seed_reg_weight: hybrid.mf_seed_reg_weight,
            hf_refine_reg_weight: hybrid.hf_refine_reg_weight,
            mf_seed_max_bound_expansions: hybrid.mf_seed_max_bound_expansions,
            hf_refine_max_bound_expansions: hybrid.hf_refine_max_bound_expansions,
            hybrid_mf_seed_hf_refine: hybrid.hybrid_mf_seed_hf_refine,
            gmm_components: sealed.mf_lowering().gmm_components,
            ..super::super::session::default_postprocess_config()
        };
        let timeline = resolve_release_timeline(plan.tof_jd_start, intercept_jd, &post)
            .ok_or_else(|| std::io::Error::other("valid strict-HF release timeline"))?;
        let canister_ctx = build_ctx(
            plan.tof_jd_start,
            &conf,
            &coeffs,
            post.canister_am,
            post.canister_cd,
            post.canister_cr,
        )?;
        let canister_launch = StampedEciState::new(plan.release_state, plan.tof_jd_start);
        let release_r = propagate_stamped(
            &canister_launch,
            timeline.canister_coast_s,
            conf.canister_body_force(post.canister_am, post.canister_cd, post.canister_cr),
            &canister_ctx,
        )?;
        let dust_ctx = build_ctx(
            release_r.jd,
            &conf,
            &coeffs,
            conf.am_ratio,
            conf.cd,
            conf.cr,
        )?;
        // A target the zero control very nearly reaches, so the LM converges
        // on a small physical vector rather than exhausting its budget.
        let mut target_i = propagate_stamped(
            &release_r,
            timeline.dust_free_flight_s,
            conf.dust_body_force(),
            &dust_ctx,
        )?
        .eci;
        target_i[0] += 0.35;
        target_i[1] -= 0.20;
        target_i[2] += 0.10;
        Ok((plan, target_i, intercept_jd, conf, post, coeffs))
    }

    /// The LM's committed endpoint, and the two HF integrator routes that
    /// disagree about it.
    ///
    /// At the sealed `gmm_components = 1` the splitter returns its input
    /// unsplit, so the single component's mean is `release_post_control_state`
    /// exactly. The two endpoint routes come from different integrator entry
    /// points and have never produced the same bits:
    ///
    /// * `predicted_intercept_state` is the LM's retained endpoint, from
    ///   `compute_miss_vector_hf_with_endpoint` -> `integrate_final_checked`,
    ///   a scalar equinoctial call whose force config comes from
    ///   `stamped_body_force_config`.
    /// * a batch sigma row comes from
    ///   `propagate_sigma_states_with_native_batch` ->
    ///   `integrate_variable_final_into`, whose per-row force config is
    ///   rebuilt by `config_for_jd_mid`.
    ///
    /// They agree only to the integrator's own tolerance. Measured on the 54
    /// strict-HF rows below at the sealed force model: worst position
    /// disagreement 1.55e-5 km (2.3e-9 relative), most rows between 1e-11 and
    /// 1e-7 km.
    ///
    /// A fourth fact stood here until 2026-08-09: that the R18 sigma-row-0
    /// endpoint reuse FIRED, i.e. that materialized sigma row 0 equalled the
    /// committed endpoint bit for bit. It is gone with the reuse. The Julier
    /// simplex has no centre point, so no sigma row is the component mean and
    /// none can duplicate that arc; what replaces the fact is the inverted
    /// assertion in
    /// `julier7_covariance_shift_against_merwe13_is_quantified`, which requires
    /// that NO row equals the committed endpoint. The route disagreement below
    /// survives the change untouched, because it is a property of the two
    /// integrator entry points and not of the sigma set.
    ///
    /// Facts pinned here:
    ///
    /// 1. The unsplit component mean IS the post-control state, bit for bit.
    /// 2. The LM endpoint commitment is exact: the committed endpoint equals
    ///    a fresh scalar propagation on every row.
    /// 3. The two integrator routes still disagree: the committed endpoint
    ///    differs from a one-row native-batch propagation of the same state
    ///    on at least one row. If they are ever unified this fails, which is
    ///    the signal that this pin and the route tripwires should be retired
    ///    together.
    #[test]
    fn committed_endpoint_is_exact_and_the_two_integrator_routes_disagree() {
        // The strict-HF pin's own orbit, then variations in altitude,
        // eccentricity, inclination and phase, each at three transfer windows
        // and three canister fractions. Keplerian rather than hand-written
        // Cartesian so every row is a real LEO that stays above the floor.
        let elements: [[f64; 6]; 6] = [
            [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0],
            [7_178.137, 0.025, 97.4, 125.0, 210.0, 40.0],
            [6_928.137, 0.001, 51.6, 20.0, 90.0, 300.0],
            [7_500.0, 0.05, 63.4, 210.0, 270.0, 120.0],
            [6_878.137, 0.012, 28.5, 300.0, 15.0, 75.0],
            [7_900.0, 0.08, 82.0, 45.0, 130.0, 225.0],
        ];
        let windows: [f64; 3] = [7_200.0, 14_400.0, 21_600.0];
        let fractions: [f64; 3] = [0.0, 0.35, 0.7];

        let mut corpus = Vec::new();
        for element in elements {
            let mut release_state = [0.0; 6];
            satpy_core::kep2eci_impl(&element, true, 0.0, 0.0, true, &mut release_state);
            for window in windows {
                for fraction in fractions {
                    corpus.push((release_state, window, fraction));
                }
            }
        }

        let ulp_delta = |a: &[f64; 6], b: &[f64; 6]| -> [i128; 6] {
            let mut out = [0_i128; 6];
            for (slot, (left, right)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
                *slot = i128::from(left.to_bits()) - i128::from(right.to_bits());
            }
            out
        };

        let mut compared = 0_usize;
        let mut disagreeing = 0_usize;
        let mut worst_km = 0.0_f64;
        for (release_state, transfer_s, fraction) in corpus {
            let Ok((plan, target_i, intercept_jd, conf, post, coeffs)) =
                strict_hf_release_fixture(release_state, transfer_s, fraction)
            else {
                continue;
            };
            let Ok((control, dust_ctx)) = build_physical_release_control(
                &plan,
                &target_i,
                intercept_jd,
                &conf,
                &post,
                &coeffs,
                &mut UnobservedPostprocessLeg,
            ) else {
                continue;
            };

            let distribution = materialize_corrected_dust_distribution(
                &control,
                &dust_ctx,
                &conf,
                &post,
                None,
                None,
                None,
                None,
                &mut UnobservedPostprocessLeg,
            )
            .expect("strict-HF corrected distribution");

            // One component at the sealed K, so row 0 of the flat sigma block
            // is component 0's sigma 0.
            assert_eq!(distribution.release_comp_means.len(), 1);
            assert_eq!(
                distribution.propagated_sigma_points.len(),
                dust_ukf_rs::NUM_SIGMA
            );

            // Fact 1: the duplicate's precondition.
            let released = distribution
                .release_comp_means
                .first()
                .expect("one release component");
            for (axis, (mean, base)) in released
                .iter()
                .zip(control.release_post_control_state.eci.iter())
                .enumerate()
            {
                assert_eq!(
                    mean.to_bits(),
                    base.to_bits(),
                    "unsplit component mean must be the post-control state, axis {axis}"
                );
            }

            // Fact 2: the LM endpoint commitment is exact.
            let fresh_stamped = propagate_stamped_checked(
                &control.release_post_control_state,
                control.dust_free_flight_s,
                conf.dust_body_force(),
                &dust_ctx,
            )
            .expect("fresh stamped propagation of the post-control state");
            assert_eq!(
                ulp_delta(&control.predicted_intercept_state.eci, &fresh_stamped.eci),
                [0; 6],
                "the committed endpoint must equal a fresh scalar propagation"
            );

            // Fact 3: the route disagreement itself, committed endpoint vs a
            // one-row native batch of the same state over the same arc.
            let mut batch_row = [0.0_f64; 6];
            crate::postprocess::ukf::propagate_sigma_states_with_native_batch(
                &control.release_post_control_state.eci,
                &mut batch_row,
                1,
                control.dust_free_flight_s,
                &dust_ctx,
            )
            .expect("one-row native batch");
            let position_km = norm3(&[
                control.predicted_intercept_state.eci[0] - batch_row[0],
                control.predicted_intercept_state.eci[1] - batch_row[1],
                control.predicted_intercept_state.eci[2] - batch_row[2],
            ]);
            if ulp_delta(&control.predicted_intercept_state.eci, &batch_row) != [0; 6] {
                disagreeing += 1;
            }
            worst_km = worst_km.max(position_km);
            compared += 1;
        }

        // Printed, not just asserted: the magnitude is the finding, and it
        // moves with anything that changes the step sequence. Re-read it with
        // `--nocapture` after any integrator change rather than trusting that
        // the bound below still passes.
        println!(
            "sigma-0 route disagreement: {disagreeing}/{compared} rows disagree, \
             worst |dr| {worst_km:.6e} km"
        );

        assert!(
            compared >= 30,
            "corpus must reach at least 30 strict-HF rows, reached {compared}"
        );
        assert!(
            disagreeing > 0,
            "the two routes are expected to disagree; if they no longer do, the \
             scalar and native integrators have converged bit-for-bit and this \
             pin is measuring nothing — confirm that against the two-route \
             disagreement note before retiring it"
        );

        // The two routes must stay inside the campaign's own position
        // corroboration threshold. Read from the seal, not restated: this is a
        // physical anchor, and the measured worst (1.55e-5 km) sits about
        // 1600x below it. A route change that blew through this is a finding
        // about the integrator, not about the UKF.
        let corroboration_km = nd_config::CompiledPartAScienceV1::part_a_v1()
            .hybrid()
            .target_corroboration_position_km;
        assert!(
            worst_km < corroboration_km,
            "route disagreement {worst_km:.6e} km must stay below the sealed \
             corroboration threshold {corroboration_km} km"
        );
    }

    /// Ceiling on the julier7-vs-Merwe13 relative Frobenius covariance shift
    /// over this corpus.
    ///
    /// MEASURED 2026-08-09 in release on macOS/arm64, 54 of 54 rows compared:
    /// worst **4.958608e-4** on both the full 6x6 and the position 3x3, with
    /// the poison arm at 1.106762e1 — 22,300x the shift, so the metric is not
    /// merely responding, it is dominating.
    ///
    /// 1e-3 is 2.02x the measured value. The predecessor ceiling on this
    /// harness sat at ~19x its measurement because it bounded a rounding-scale
    /// effect that had to absorb integrator-truncation wander across hosts;
    /// this quantity is a genuine difference between two quadratures on the
    /// same arcs, so it is stable to the integrator's own reproducibility and
    /// does not need that much room. Tight enough that the shift growing by
    /// half an order trips it.
    ///
    /// This is an UPPER bound on production, not a production number: the
    /// corpus flies 7,200-21,600 s arcs against a census `mean_span_s` of
    /// 2,830, and the simplex error is steeply superlinear in arc length. The
    /// accepted trade was ~1.1e-4 at the mean span rising to ~2-3e-4 on the
    /// 18.2% of arcs in the 7,000-8,000 s band; 4.96e-4 over a corpus whose
    /// SHORTEST arc is 7,200 s is the same finding, measured at the tip.
    const JULIER7_COVARIANCE_SHIFT_CEILING: f64 = 1.0e-3;

    /// The retired Merwe-13 set, rebuilt here as a COMPARISON ARM only.
    ///
    /// This is a test-local reference implementation, not a shadow of a
    /// production constant: `dust_ukf_rs` no longer contains a Merwe generator,
    /// so there is nothing here for this to drift away from. It exists so the
    /// julier7 covariance can be measured against the set it replaced, on the
    /// same arcs, in the same context, with only the quadrature differing.
    ///
    /// Geometry at the retired sealed tuning (`alpha = 1`, `kappa = 0`, so
    /// `lambda = 0`): row 0 is the mean, rows `1..=6` are `mean + L.col(i)` and
    /// rows `7..=12` are `mean - L.col(i)`, for `L` the Cholesky factor of
    /// `(n + lambda) * covar = 6 * covar`. Weights: `wm[0] = 0`, `wc[0] = beta
    /// = 2`, every other weight `1 / (2 * (n + lambda)) = 1/12`.
    const MERWE13_NUM_SIGMA: usize = 13;

    #[expect(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        clippy::needless_range_loop,
        reason = "fixed six-state reference geometry over compile-time-sized arrays, \
                  in the retired generator's own operation order"
    )]
    fn merwe13_reference_points(
        mean: &[f64; 6],
        covar: &[[f64; 6]; 6],
    ) -> Option<[[f64; 6]; MERWE13_NUM_SIGMA]> {
        let mean_vec = nalgebra::SVector::<f64, 6>::from_column_slice(mean);
        let mut covar_mat = nalgebra::SMatrix::<f64, 6, 6>::zeros();
        for (row, values) in covar.iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                *covar_mat.get_mut((row, column))? = *value;
            }
        }
        let factor = nalgebra::Cholesky::new(6.0 * covar_mat)?.unpack();
        let mut points = [[0.0_f64; 6]; MERWE13_NUM_SIGMA];
        points[0] = *mean;
        for (column_index, column) in factor.column_iter().enumerate() {
            let plus = mean_vec + column;
            let minus = mean_vec - column;
            for axis in 0..6 {
                points[column_index + 1][axis] = *plus.get(axis)?;
                points[column_index + 7][axis] = *minus.get(axis)?;
            }
        }
        Some(points)
    }

    fn merwe13_reference_weights() -> ([f64; MERWE13_NUM_SIGMA], [f64; MERWE13_NUM_SIGMA]) {
        let mut wm = [1.0 / 12.0; MERWE13_NUM_SIGMA];
        let mut wc = [1.0 / 12.0; MERWE13_NUM_SIGMA];
        wm[0] = 0.0;
        wc[0] = 2.0;
        (wm, wc)
    }

    /// Thirteen-row reconstruction, in `dust_ukf_rs::rebuild_mean_covar_ukf`'s
    /// own accumulation order.
    ///
    /// The production rebuild is fixed at `NUM_SIGMA = 7` by its array shapes,
    /// so a 13-row arm cannot call it. Order is reproduced deliberately rather
    /// than approximated: the two arms are being compared at 1e-4, and a
    /// different summation order would put its own reassociation noise into
    /// that comparison.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "fixed six-state reduction in `rebuild_mean_covar_ukf`'s own \
                  accumulation order, which this arm exists to reproduce"
    )]
    fn merwe13_rebuild(
        points: &[[f64; 6]; MERWE13_NUM_SIGMA],
        wm: &[f64; MERWE13_NUM_SIGMA],
        wc: &[f64; MERWE13_NUM_SIGMA],
    ) -> nalgebra::SMatrix<f64, 6, 6> {
        let mut mean = nalgebra::SVector::<f64, 6>::zeros();
        for (point, weight) in points.iter().zip(wm.iter()) {
            mean += *weight * nalgebra::SVector::<f64, 6>::from_column_slice(point);
        }
        let mut covar = nalgebra::SMatrix::<f64, 6, 6>::zeros();
        for (point, weight) in points.iter().zip(wc.iter()) {
            let diff = nalgebra::SVector::<f64, 6>::from_column_slice(point) - mean;
            covar += *weight * (diff * diff.transpose());
        }
        for row in 0..6 {
            for column in 0..row {
                let lower = *covar.get((row, column)).expect("index in bounds");
                let upper = *covar.get((column, row)).expect("index in bounds");
                let value = 0.5 * (lower + upper);
                *covar.get_mut((row, column)).expect("index in bounds") = value;
                *covar.get_mut((column, row)).expect("index in bounds") = value;
            }
        }
        covar
    }

    /// Quantifies what the julier7 sigma set does to the published covariance
    /// at the tip, row by row, against the Merwe-13 set it replaced.
    ///
    /// This is the acceptance measurement for the sigma-set change, taken on
    /// the propagated 6x6 rather than on a quadrature identity: both arms start
    /// from the same release mean and covariance, fly the same arc in the same
    /// context, and differ ONLY in which points were drawn and how they are
    /// weighted back together. The audit measured the same quantity against a
    /// degree-5 Gauss-Hermite reference and recorded ~1.1e-4 relative Frobenius
    /// at the census mean span, interpolating to ~2-3e-4 on the 18.2% of arcs
    /// that fly 7,000-8,000 s
    /// (`docs/plans/2026-08-05-hf-hybrid-speedup-audit.md` §15b/§15c). That
    /// shift is the accepted price of going from 12 propagated arcs per row to
    /// 7; this test is what stops it growing quietly.
    ///
    /// This corpus flies LONGER arcs than production does — windows of 7,200 to
    /// 21,600 s against a census `mean_span_s` of 2,830 — and the simplex error
    /// is steeply superlinear in arc length, so the numbers here are an UPPER
    /// bound on what the campaign sees, not an estimate of it.
    ///
    /// It replaces two tests that went with the R18 sigma-row-0 endpoint reuse:
    /// `sealed_merwe_tuning_gives_sigma_row_zero_no_weight_in_the_mean` (which
    /// pinned `wm[0] = 0`, the property that made the reuse cheap) and
    /// `sigma_zero_endpoint_reuse_covariance_displacement_is_quantified` (which
    /// bounded that reuse's displacement at 1e-5). Neither survives the
    /// simplex: it has no centre point, so there is no row 0 to weight out and
    /// no duplicate arc to reuse. The first assertion below is what remains of
    /// them, and it is the load-bearing one — it proves the reuse could not
    /// fire, which is why removing its machinery was safe.
    ///
    /// Non-vacuity, both ways: the production covariance is used directly
    /// rather than rebuilt, and a 1 km poison on one propagated point is proven
    /// to move the metric by at least 100x the measured shift, so a metric that
    /// stopped responding fails instead of reporting a comforting zero.
    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "fixed 6x6 covariance blocks over compile-time-sized arrays; an \
                  out-of-range index here is a test bug and panicking says so"
    )]
    fn julier7_covariance_shift_against_merwe13_is_quantified() {
        let elements: [[f64; 6]; 6] = [
            [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0],
            [7_178.137, 0.025, 97.4, 125.0, 210.0, 40.0],
            [6_928.137, 0.001, 51.6, 20.0, 90.0, 300.0],
            [7_500.0, 0.05, 63.4, 210.0, 270.0, 120.0],
            [6_878.137, 0.012, 28.5, 300.0, 15.0, 75.0],
            [7_900.0, 0.08, 82.0, 45.0, 130.0, 225.0],
        ];
        let windows: [f64; 3] = [7_200.0, 14_400.0, 21_600.0];
        let fractions: [f64; 3] = [0.0, 0.35, 0.7];

        let mut corpus = Vec::new();
        for element in elements {
            let mut release_state = [0.0; 6];
            satpy_core::kep2eci_impl(&element, true, 0.0, 0.0, true, &mut release_state);
            for window in windows {
                for fraction in fractions {
                    corpus.push((release_state, window, fraction));
                }
            }
        }

        let rel_frobenius = |base: &[[f64; 6]; 6],
                             moved: &nalgebra::SMatrix<f64, 6, 6>,
                             dims: std::ops::Range<usize>| {
            let mut delta_sq = 0.0_f64;
            let mut base_sq = 0.0_f64;
            for row in dims.clone() {
                for column in dims.clone() {
                    let base_value = base[row][column];
                    let moved_value = *moved
                        .get((row, column))
                        .expect("covariance index is in bounds");
                    let difference = moved_value - base_value;
                    delta_sq += difference * difference;
                    base_sq += base_value * base_value;
                }
            }
            (delta_sq / base_sq).sqrt()
        };

        let mut compared = 0_usize;
        let mut worst_full = 0.0_f64;
        let mut worst_pos = 0.0_f64;
        let mut worst_poison = 0.0_f64;
        for (release_state, transfer_s, fraction) in corpus {
            let Ok((plan, target_i, intercept_jd, mut conf, post, coeffs)) =
                strict_hf_release_fixture(release_state, transfer_s, fraction)
            else {
                continue;
            };
            // `strict_hf_release_fixture` leaves the four dust release sigmas at
            // their `PhysicsConfig::default()` zero, which makes the release
            // covariance identically zero and sends BOTH arms through
            // `repair_to_psd`'s 1e-12 eigenvalue floor. A covariance shift
            // measured there is a fact about the repair, not about the
            // quadrature. The sealed MF lowering values are installed here so
            // the two sigma sets are compared on the cloud production releases.
            let lowering = nd_config::CompiledPartAScienceV1::part_a_v1().mf_lowering();
            conf.dust_pos_sigma_m = lowering.dust_pos_sigma_m;
            conf.dust_pos_sigma_radial_cross_track_m = lowering.dust_pos_sigma_radial_cross_track_m;
            conf.dust_vel_sigma_mps = lowering.dust_vel_sigma_mps;
            conf.dust_vel_sigma_radial_cross_track_mps =
                lowering.dust_vel_sigma_radial_cross_track_mps;
            let Ok((control, dust_ctx)) = build_physical_release_control(
                &plan,
                &target_i,
                intercept_jd,
                &conf,
                &post,
                &coeffs,
                &mut UnobservedPostprocessLeg,
            ) else {
                continue;
            };

            let distribution = materialize_corrected_dust_distribution(
                &control,
                &dust_ctx,
                &conf,
                &post,
                None,
                None,
                None,
                None,
                &mut UnobservedPostprocessLeg,
            )
            .expect("strict-HF corrected distribution");
            assert_eq!(distribution.release_comp_means.len(), 1);
            let points = &distribution.propagated_sigma_points;
            assert_eq!(points.len(), dust_ukf_rs::NUM_SIGMA);

            // The load-bearing assertion. Every simplex point is off-centre, so
            // NO production row can be the committed endpoint of the release
            // mean's own arc. This is the derivation that retired the R18
            // sigma-row-0 endpoint reuse: its bit guard fired only on a row 0
            // that equalled the release mean, and there is no such row. If this
            // ever fails, the reuse lever is live again and worth re-pricing.
            for (row_index, point) in points.iter().enumerate() {
                let identical = point
                    .iter()
                    .zip(control.predicted_intercept_state.eci.iter())
                    .all(|(produced, committed)| produced.to_bits() == committed.to_bits());
                assert!(
                    !identical,
                    "sigma row {row_index} duplicates the committed endpoint; the simplex \
                     is supposed to have no centre point"
                );
            }

            let release_mean = distribution
                .release_comp_means
                .first()
                .expect("one release component");
            let release_cov = distribution
                .release_comp_covs
                .first()
                .expect("one release component");
            // No PSD-repair fallback here, deliberately: the sealed release
            // covariance is strictly positive definite, so a Cholesky failure
            // means the corpus stopped being the one this measures, and the
            // `compared >= 30` floor below turns that into a red rather than a
            // quietly shorter run.
            let Some(merwe_points) = merwe13_reference_points(release_mean, release_cov) else {
                continue;
            };
            let mut merwe_initial = [0.0_f64; MERWE13_NUM_SIGMA * 6];
            for (row, point) in merwe_points.iter().enumerate() {
                merwe_initial[row * 6..(row + 1) * 6].copy_from_slice(point);
            }
            let mut merwe_propagated = [0.0_f64; MERWE13_NUM_SIGMA * 6];
            crate::postprocess::ukf::propagate_sigma_states_with_native_batch(
                &merwe_initial,
                &mut merwe_propagated,
                MERWE13_NUM_SIGMA,
                control.dust_free_flight_s,
                &dust_ctx,
            )
            .expect("Merwe-13 reference batch");

            let (mean_weights, covariance_weights) = merwe13_reference_weights();
            let mut merwe_rows = [[0.0_f64; 6]; MERWE13_NUM_SIGMA];
            for (row, state) in merwe_rows.iter_mut().enumerate() {
                state.copy_from_slice(&merwe_propagated[row * 6..(row + 1) * 6]);
            }
            let cov_merwe = merwe13_rebuild(&merwe_rows, &mean_weights, &covariance_weights);

            let cov_julier = distribution.comp_covs.first().expect("one component");
            worst_full = worst_full.max(rel_frobenius(cov_julier, &cov_merwe, 0..6));
            worst_pos = worst_pos.max(rel_frobenius(cov_julier, &cov_merwe, 0..3));

            // Poison arm: a 1 km displacement of one propagated Merwe point,
            // which correct arithmetic on either set cannot produce.
            let mut poisoned = merwe_rows;
            poisoned[1][0] += 1.0;
            let cov_poison = merwe13_rebuild(&poisoned, &mean_weights, &covariance_weights);
            worst_poison = worst_poison.max(rel_frobenius(cov_julier, &cov_poison, 0..6));

            compared += 1;
        }

        println!(
            "julier7 vs merwe13 covariance shift over {compared} rows: worst rel-Frobenius \
             full {worst_full:.6e}, position block {worst_pos:.6e}, poison arm \
             {worst_poison:.6e}"
        );

        assert!(
            compared >= 30,
            "corpus must reach at least 30 strict-HF rows, reached {compared}"
        );
        assert!(
            worst_poison > 100.0 * worst_full && worst_poison > 1.0e-6,
            "poison displacement {worst_poison:.3e} must dominate the measured shift \
             {worst_full:.3e}"
        );
        assert!(
            worst_full < JULIER7_COVARIANCE_SHIFT_CEILING,
            "julier7 covariance shift {worst_full:.6e} exceeds the \
             {JULIER7_COVARIANCE_SHIFT_CEILING:.1e} ceiling; the accepted trade was a \
             ~3e-4 relative shift on the production span distribution, and this is a \
             different quantity than the one that was accepted"
        );
    }

    #[test]
    fn malformed_public_sigma_points_are_rejected_before_ukf_materialization() {
        let (plan, target_i, intercept_jd, conf, post, coeffs) =
            release_control_fixture(0.0, 7200.0).expect("release-control fixture");
        let (control, dust_ctx) = build_physical_release_control(
            &plan,
            &target_i,
            intercept_jd,
            &conf,
            &post,
            &coeffs,
            &mut UnobservedPostprocessLeg,
        )
        .expect("physical release control");
        let release_distribution = AuthoritativeReleaseDistribution {
            means: vec![control.release_post_control_state.eci],
            covariances: vec![[[0.0; 6]; 6]],
            weights: vec![1.0],
            sigma_points: Some(Vec::new()),
        };

        assert!(matches!(
            materialize_corrected_dust_distribution(
                &control,
                &dust_ctx,
                &conf,
                &post,
                Some(-1.0),
                Some([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                Some(&[[0.0; 6]; 6]),
                Some(release_distribution),
                &mut UnobservedPostprocessLeg,
            ),
            Err(PostprocessDistributionStatus::InvalidReleaseDistribution)
        ));
    }

    #[test]
    fn mf_release_control_preserves_evaluator_arithmetic_overflow() {
        std::thread::spawn(|| {
            let (plan, mut target_i, _, conf, post, coeffs) =
                release_control_fixture(0.0, 7200.0).expect("release-control fixture");
            target_i[0] += 1.0;
            let release_state = StampedEciState::new(plan.release_state, plan.tof_jd_start);
            let dust_ctx = build_ctx(
                release_state.jd,
                &conf,
                &coeffs,
                conf.am_ratio,
                conf.cd,
                conf.cr,
            )
            .expect("J2 propagation context");

            let mut diagnostics = crate::evaluate::evaluation_diagnostic_snapshot();
            diagnostics.j2_propagate_state_count = usize::MAX - 1;
            crate::evaluate::restore_evaluation_diagnostics(&diagnostics);

            let outcome = solve_release_control_delta_dv(
                &release_state,
                [target_i[0], target_i[1], target_i[2]],
                7200.0,
                &conf,
                &post,
                &dust_ctx,
                [0.0; 3],
                plan.min_radius_km,
                &mut UnobservedPostprocessLeg,
            );
            assert!(matches!(
                outcome,
                Err(TransferPropagationFailure::ArithmeticOverflow)
            ));
        })
        .join()
        .expect("isolated evaluator-overflow test thread must not panic");
    }

    #[test]
    fn carried_endpoint_requires_exact_epoch_or_fresh_propagation() {
        let (plan, _, _, conf, _, coeffs) =
            release_control_fixture(0.0, 7200.0).expect("release-control fixture");
        let source = StampedEciState::new(plan.release_state, plan.tof_jd_start);
        let ctx = build_ctx(source.jd, &conf, &coeffs, conf.am_ratio, conf.cd, conf.cr)
            .expect("J2 propagation context");
        let body_force = BodyForceConfig::j2(BodyRole::Dust);
        let tof_s = 60.0;
        let fresh =
            propagate_stamped(&source, tof_s, body_force, &ctx).expect("fresh propagated endpoint");
        let assert_state_bits_eq = |expected: &StampedEciState, actual: &StampedEciState| {
            assert_eq!(expected.jd.to_bits(), actual.jd.to_bits());
            assert_eq!(expected.eci.map(f64::to_bits), actual.eci.map(f64::to_bits));
        };

        reset_stamped_propagation_calls();
        let exact = resolve_predicted_intercept(
            Some(fresh.eci),
            source.jd,
            source.jd,
            &source,
            tof_s,
            body_force,
            &ctx,
            &mut UnobservedPostprocessLeg,
        )
        .expect("exact carried endpoint");
        assert_state_bits_eq(&fresh, &exact);
        assert_eq!(stamped_propagation_calls(), 0);

        for (endpoint, source_epoch) in [
            (None, source.jd),
            (Some([f64::NAN; 6]), f64::from_bits(source.jd.to_bits() + 1)),
        ] {
            reset_stamped_propagation_calls();
            let fallback = resolve_predicted_intercept(
                endpoint,
                source_epoch,
                source.jd,
                &source,
                tof_s,
                body_force,
                &ctx,
                &mut UnobservedPostprocessLeg,
            )
            .expect("fresh fallback endpoint");
            assert_state_bits_eq(&fresh, &fallback);
            assert_eq!(stamped_propagation_calls(), 1);
        }
    }

    #[test]
    fn non_hf_release_solution_never_carries_an_endpoint() {
        let (plan, target, intercept_jd, conf, post, coeffs) =
            release_control_fixture(0.0, 7200.0).expect("release-control fixture");
        let tof_s = (intercept_jd - plan.tof_jd_start) * SEC_PER_DAY;
        let source = StampedEciState::new(plan.release_state, plan.tof_jd_start);
        let ctx = build_ctx(source.jd, &conf, &coeffs, conf.am_ratio, conf.cd, conf.cr)
            .expect("J2 propagation context");
        let solution = solve_release_control_delta_dv(
            &source,
            [target[0], target[1], target[2]],
            tof_s,
            &conf,
            &post,
            &ctx,
            [0.0; 3],
            plan.min_radius_km,
            &mut UnobservedPostprocessLeg,
        )
        .expect("J2 solver must not fail")
        .expect("J2 zero-control solution");

        assert!(solution.endpoint.is_none());
    }

    #[test]
    fn physical_distribution_matches_old_diagnostic_path_at_fraction_bounds() {
        for fraction in [0.0, 0.95] {
            let (plan, target_i, intercept_jd, mut conf, post, coeffs) =
                release_control_fixture(fraction, 7201.0).expect("release-control fixture");
            conf.splitting_criterion = "maxvar".to_owned();
            let conjunction_jd = intercept_jd + 1800.0 / SEC_PER_DAY;

            reset_conjunction_diagnostic_calls();
            let (diagnostic_control, diagnostic_dust_ctx, _) = build_release_control(
                &plan,
                &target_i,
                intercept_jd,
                conjunction_jd,
                &conf,
                &post,
                &coeffs,
                None,
                None,
                None,
                ConjunctionDiagnostic::Compute,
            )
            .expect("old diagnostic release-control path");
            assert_eq!(conjunction_diagnostic_calls(), 1);
            assert!(diagnostic_control.conjunction_separation_km.is_finite());

            reset_conjunction_diagnostic_calls();
            let (physical_control, physical_dust_ctx) = build_physical_release_control(
                &plan,
                &target_i,
                intercept_jd,
                &conf,
                &post,
                &coeffs,
                &mut UnobservedPostprocessLeg,
            )
            .expect("physical release-control path");
            assert_eq!(conjunction_diagnostic_calls(), 0);
            assert_physical_control_bits_eq(&diagnostic_control, &physical_control);

            let expected = materialize_corrected_dust_distribution(
                &diagnostic_control,
                &diagnostic_dust_ctx,
                &conf,
                &post,
                None,
                None,
                None,
                None,
                &mut UnobservedPostprocessLeg,
            )
            .expect("distribution from old diagnostic control");
            let actual = materialize_corrected_dust_distribution(
                &physical_control,
                &physical_dust_ctx,
                &conf,
                &post,
                None,
                None,
                None,
                None,
                &mut UnobservedPostprocessLeg,
            )
            .expect("distribution from physical control");

            assert_distribution_bits_eq(&expected, &actual);
        }
    }

    #[test]
    fn release_timeline_partitions_l_r_i_for_boundary_fractions_and_windows() {
        let window_s = 7200.0;
        for fraction in [0.0, 0.5, 0.95, 0.9999] {
            for transfer_s in [
                window_s - 1e-3,
                window_s,
                window_s + 1e-3,
                3.0 * SEC_PER_DAY,
            ] {
                let post = PostprocessConfig {
                    dust_phase_tof_s: window_s,
                    canister_tof_fraction: fraction,
                    ..super::super::session::default_postprocess_config()
                };
                let l = 2_460_000.5;
                let i = l + transfer_s / SEC_PER_DAY;
                let boundary = resolve_release_timeline(l, i, &post).expect("valid L/R/I");
                assert!(
                    (boundary.transfer_burn_jd + boundary.canister_coast_s / SEC_PER_DAY
                        - boundary.release_jd)
                        .abs()
                        * SEC_PER_DAY
                        < 1e-4
                );
                assert!(
                    (boundary.release_jd + boundary.dust_free_flight_s / SEC_PER_DAY
                        - boundary.intercept_jd)
                        .abs()
                        * SEC_PER_DAY
                        < 1e-4
                );
                let jd_duration_s =
                    (boundary.intercept_jd - boundary.transfer_burn_jd) * SEC_PER_DAY;
                assert!(
                    (boundary.canister_coast_s + boundary.dust_free_flight_s - jd_duration_s).abs()
                        < 1e-6
                );
                if boundary.canister_coast_s > 0.0 {
                    // JD's ulp at this epoch is about 40 microseconds; a
                    // positive sub-ulp coast still has an authoritative C.
                    assert!(boundary.release_jd >= boundary.transfer_burn_jd);
                }
            }
        }
    }

    #[test]
    fn release_timeline_accepts_representable_jd_recomposition_error() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        for fraction in [0.0, 0.316_666_666_666_666_65, 0.633_333_333_333_333_3, 0.95] {
            for _ in 0..4096 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let numerator = (seed >> 11)
                    .to_f64()
                    .expect("53-bit randomized numerator converts exactly to f64");
                let denominator = (1_u64 << 53)
                    .to_f64()
                    .expect("53-bit randomized denominator converts exactly to f64");
                let epoch_offset_days = numerator / denominator;
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let unit_interval = (seed >> 11)
                    .to_f64()
                    .expect("53-bit randomized numerator converts exactly to f64")
                    / denominator;
                let transfer_s = 1.0 + unit_interval * (2.0 * SEC_PER_DAY - 1.0);
                let transfer_burn_jd = 2_460_000.0 + epoch_offset_days;
                let intercept_jd = transfer_burn_jd + transfer_s / SEC_PER_DAY;
                let post = PostprocessConfig {
                    dust_phase_tof_s: 7200.0,
                    canister_tof_fraction: fraction,
                    ..super::super::session::default_postprocess_config()
                };
                assert!(
                    resolve_release_timeline(transfer_burn_jd, intercept_jd, &post).is_some(),
                    "representable L/R/I partition rejected: fraction={fraction} transfer_s={transfer_s}"
                );
            }
        }
        let jd = 2_460_000.25;
        assert!(jd_closure_within_tolerance(
            jd,
            f64::from_bits(jd.to_bits() + 1)
        ));
        assert!(!jd_closure_within_tolerance(jd, jd + 1.0e-3 / SEC_PER_DAY));
    }

    #[test]
    fn jd_ulp_advances_the_largest_finite_value_to_infinity() {
        assert!(jd_ulp_days(f64::MAX).is_infinite());
    }

    #[test]
    fn osculating_perigee_gate_rejects_endpoint_closed_earth_crossing_arc() {
        let radius = 7000.0;
        let circular_speed = (MU / radius).sqrt();
        let safe = [radius, 0.0, 0.0, 0.0, circular_speed, 0.0];
        assert!(state_clears_min_radius(&safe, 6578.137));

        // One full revolution returns to the same endpoint after 4920 s, but
        // this post-control ellipse has rp=5504.49 km and crosses Earth.
        let crossing = [radius, 0.0, 0.0, 0.0, 7.080_443_746_84, 0.0];
        let perigee = osculating_perigee_km(&crossing).expect("bound orbit");
        assert!((perigee - 5504.49).abs() < 0.1, "perigee={perigee}");
        assert!(!state_clears_min_radius(&crossing, 6578.137));
    }

    #[test]
    fn release_control_uses_coasted_state_once_and_keeps_epoch_arithmetic_consistent() {
        // Epoch checks below are arithmetic self-consistency (jd deltas match
        // coast / free-flight durations within slack); no epoch is compared
        // to the stamped input, so a uniform epoch shift would still pass.
        for (fraction, transfer_s) in [
            (0.0, 7201.0),
            (0.5, 7200.0),
            (0.95, 7201.0),
            (0.9999, 3.0 * SEC_PER_DAY),
        ] {
            let (plan, target_i, intercept_jd, conf, post, coeffs) =
                release_control_fixture(fraction, transfer_s).expect("release-control fixture");
            let (control, _, _) = build_release_control(
                &plan,
                &target_i,
                intercept_jd,
                intercept_jd,
                &conf,
                &post,
                &coeffs,
                None,
                None,
                None,
                ConjunctionDiagnostic::Compute,
            )
            .expect("exact target gives a release-control result");
            assert!(matches!(
                control.status,
                PostprocessControlStatus::Applied | PostprocessControlStatus::AppliedZero
            ));
            // The target was generated from the coasted R state.  Any solver
            // that still consumed L would need a material corrective burn.
            assert!(
                control.release_control_dv_norm <= 1e-8,
                "fraction={fraction} transfer_s={transfer_s} dv={}",
                control.release_control_dv_norm
            );
            assert!(
                (norm3(&control.release_control_dv) - control.release_control_dv_norm).abs()
                    < 1e-12
            );
            assert!(
                (control.release_pre_control_state.jd
                    - control.canister_launch_state.jd
                    - control.canister_coast_s / SEC_PER_DAY)
                    .abs()
                    * SEC_PER_DAY
                    < 1e-4
            );
            assert!(
                (control.predicted_intercept_state.jd
                    - control.release_post_control_state.jd
                    - control.dust_free_flight_s / SEC_PER_DAY)
                    .abs()
                    * SEC_PER_DAY
                    < 1e-4
            );
            if control.canister_coast_s > 0.0 {
                assert_ne!(
                    control.release_pre_control_state.eci.map(f64::to_bits),
                    control.canister_launch_state.eci.map(f64::to_bits),
                    "authoritative coast must change at least one state bit"
                );
            }
            for ((post_control_velocity, pre_control_velocity), control_dv) in control
                .release_post_control_state
                .eci
                .iter()
                .skip(3)
                .zip(control.release_pre_control_state.eci.iter().skip(3))
                .zip(control.release_control_dv.iter())
            {
                assert!(
                    (*post_control_velocity - *pre_control_velocity - *control_dv).abs() < 1e-12
                );
            }
        }
    }

    #[test]
    fn release_control_fraction_override_is_authoritative_and_not_relabelled() {
        let (plan, mut target_i, intercept_jd, conf, mut post, coeffs) =
            release_control_fixture(0.5, 7200.0).expect("release-control fixture");
        target_i[0] += 1.0;
        post.fix_ls_skip_tol = 0.0;
        post.max_physical_dv_kms = 10.0;

        let (early, _, _) = build_release_control_at_fraction(
            &plan,
            &target_i,
            intercept_jd,
            intercept_jd,
            &conf,
            &post,
            &coeffs,
            None,
            None,
            None,
            0.25,
            TargetPropagationAuthority::MfJ2,
            ConjunctionDiagnostic::Compute,
        )
        .expect("quarter-fraction control");
        let (late, _, _) = build_release_control_at_fraction(
            &plan,
            &target_i,
            intercept_jd,
            intercept_jd,
            &conf,
            &post,
            &coeffs,
            None,
            None,
            None,
            0.5,
            TargetPropagationAuthority::MfJ2,
            ConjunctionDiagnostic::Compute,
        )
        .expect("half-fraction control");

        assert_eq!(early.canister_tof_fraction.to_bits(), 0.25_f64.to_bits());
        assert_eq!(late.canister_tof_fraction.to_bits(), 0.5_f64.to_bits());
        assert!((early.canister_coast_s - 1800.0).abs() < 1e-3);
        assert!((late.canister_coast_s - 3600.0).abs() < 1e-3);
        assert!((early.dust_free_flight_s - 5400.0).abs() < 1e-3);
        assert!((late.dust_free_flight_s - 3600.0).abs() < 1e-3);
        assert_ne!(
            early.release_pre_control_state.eci.map(f64::to_bits),
            late.release_pre_control_state.eci.map(f64::to_bits)
        );
        assert_ne!(
            early.release_control_dv.map(f64::to_bits),
            late.release_control_dv.map(f64::to_bits)
        );
    }

    #[test]
    fn release_control_fraction_override_rejects_nonfinite_or_out_of_range_values() {
        let (plan, target_i, intercept_jd, conf, post, coeffs) =
            release_control_fixture(0.5, 7200.0).expect("release-control fixture");
        for fraction in [f64::NAN, -1e-12, 1.0] {
            let result = build_release_control_at_fraction(
                &plan,
                &target_i,
                intercept_jd,
                intercept_jd,
                &conf,
                &post,
                &coeffs,
                None,
                None,
                None,
                fraction,
                TargetPropagationAuthority::MfJ2,
                ConjunctionDiagnostic::Compute,
            );
            assert!(matches!(
                result,
                Err(PostprocessControlStatus::InvalidTimeline)
            ));
        }
    }

    #[test]
    fn release_control_fraction_override_preserves_configured_canister_ballistics() {
        let (_, _, _, _, mut post, _) =
            release_control_fixture(0.5, 7200.0).expect("release-control fixture");
        post.canister_am = 0.004;
        post.canister_cd = 2.7;
        post.canister_cr = 1.8;

        for fraction in [0.0, 0.316_666_666_666_666_65, 0.633_333_333_333_333_3, 0.95] {
            let overridden = postprocess_config_at_fraction(&post, fraction)
                .expect("valid fraction with fixed configured canister ballistics");

            assert_eq!(
                overridden.canister_tof_fraction.to_bits(),
                fraction.to_bits()
            );
            assert_eq!(overridden.canister_am.to_bits(), post.canister_am.to_bits());
            assert_eq!(overridden.canister_cd.to_bits(), post.canister_cd.to_bits());
            assert_eq!(overridden.canister_cr.to_bits(), post.canister_cr.to_bits());
            assert_eq!(post.canister_tof_fraction.to_bits(), 0.5_f64.to_bits());
        }
    }

    #[test]
    fn strict_public_release_control_keeps_diagnostic_and_canister_ballistic_authority() {
        let c = Arc::new(vec![1.0, 0.0, 0.0, 0.0]);
        let s = Arc::new(vec![0.0; 4]);
        let packed = Arc::new(
            satpy_core::pack_gravity_coeffs(&c, &s, 2, 0)
                .expect("test gravity coefficients are valid"),
        );
        let coeffs = GlobalCoeffs {
            packed: Some(packed),
            missing: false,
        };
        let conf = PhysicsConfig {
            use_high_fidelity: true,
            require_hf_transfer_correction: true,
            force_flags: lightyear_odeint_rs::types::ForceFlags::SRP,
            am_ratio: 0.01,
            cd: 2.2,
            cr: 1.3,
            sun_pos: Some([149_597_870.7, 0.0, 0.0]),
            dt_max: 30.0,
            tolerance: 1e-8,
            method: "dopri5".to_string(),
            ..PhysicsConfig::default()
        };
        let transfer_to_intercept_s = 7200.0;
        let intercept_jd = 2_460_000.5 + transfer_to_intercept_s / SEC_PER_DAY;
        let plan = SummaryPlanInputs {
            valid: true,
            release_state: [7000.0, 0.0, 0.0, 0.0, 7.5, 1.0],
            transfer_dv: [0.0; 3],
            tof_jd_start: 2_460_000.5,
            min_radius_km: satpy_core::RE,
        };
        let post_a = PostprocessConfig {
            dust_phase_tof_s: 7200.0,
            canister_tof_fraction: 0.17,
            canister_am: 0.0042,
            canister_cd: 2.2,
            canister_cr: 1.3,
            fix_ls_max_nfev: 100,
            fix_ls_tol: 1e-6,
            fix_ls_skip_tol: 1e-6,
            dust_intercept_tol_km: 1e-5,
            max_physical_dv_kms: 10.0,
            ..super::super::session::default_postprocess_config()
        };
        let post_b = PostprocessConfig {
            canister_am: 0.042,
            canister_cd: 2.7,
            canister_cr: 1.8,
            ..post_a
        };
        let fraction = 0.5;
        let baseline_post =
            postprocess_config_at_fraction(&post_a, fraction).expect("valid HF baseline fraction");
        let timeline = resolve_release_timeline(plan.tof_jd_start, intercept_jd, &baseline_post)
            .expect("valid HF baseline timeline");
        let canister_launch = StampedEciState::new(plan.release_state, plan.tof_jd_start);
        let canister_ctx = build_ctx(
            plan.tof_jd_start,
            &conf,
            &coeffs,
            baseline_post.canister_am,
            baseline_post.canister_cd,
            baseline_post.canister_cr,
        )
        .expect("canister propagation context");
        let baseline_release = propagate_stamped(
            &canister_launch,
            timeline.canister_coast_s,
            conf.canister_body_force(
                baseline_post.canister_am,
                baseline_post.canister_cd,
                baseline_post.canister_cr,
            ),
            &canister_ctx,
        )
        .expect("baseline release propagation");
        let dust_ctx = build_ctx(
            baseline_release.jd,
            &conf,
            &coeffs,
            conf.am_ratio,
            conf.cd,
            conf.cr,
        )
        .expect("dust propagation context");
        let target_i = propagate_stamped(
            &baseline_release,
            timeline.dust_free_flight_s,
            conf.dust_body_force(),
            &dust_ctx,
        )
        .expect("target intercept propagation")
        .eci;

        let conjunction_jd = intercept_jd + 60.0 / SEC_PER_DAY;
        reset_conjunction_diagnostic_calls();
        let (control_a, _, _) = build_release_control_at_fraction(
            &plan,
            &target_i,
            intercept_jd,
            conjunction_jd,
            &conf,
            &post_a,
            &coeffs,
            None,
            None,
            None,
            fraction,
            TargetPropagationAuthority::MfJ2,
            ConjunctionDiagnostic::Compute,
        )
        .expect("HF control with configured tuple A");
        let (control_b, _, _) = build_release_control_at_fraction(
            &plan,
            &target_i,
            intercept_jd,
            conjunction_jd,
            &conf,
            &post_b,
            &coeffs,
            None,
            None,
            None,
            fraction,
            TargetPropagationAuthority::MfJ2,
            ConjunctionDiagnostic::Compute,
        )
        .expect("HF control with configured tuple B");

        assert_eq!(conjunction_diagnostic_calls(), 2);
        assert!(control_a.conjunction_separation_km.is_finite());
        assert!(control_b.conjunction_separation_km.is_finite());
        assert_eq!(control_a.fidelity, PropagationFidelity::HighFidelity);
        assert_eq!(control_b.fidelity, PropagationFidelity::HighFidelity);
        assert_eq!(
            control_a.transfer_burn_pre_state,
            control_b.transfer_burn_pre_state
        );
        assert_eq!(
            control_a.canister_launch_state,
            control_b.canister_launch_state
        );
        assert_eq!(
            control_a.canister_tof_fraction.to_bits(),
            fraction.to_bits()
        );
        assert_eq!(
            control_b.canister_tof_fraction.to_bits(),
            fraction.to_bits()
        );
        assert_eq!(
            control_a.canister_coast_s.to_bits(),
            control_b.canister_coast_s.to_bits()
        );
        assert_eq!(
            control_a.dust_free_flight_s.to_bits(),
            control_b.dust_free_flight_s.to_bits()
        );
        assert_eq!(
            control_a.release_pre_control_state.jd.to_bits(),
            control_b.release_pre_control_state.jd.to_bits()
        );
        assert_eq!(
            control_a.predicted_intercept_state.jd.to_bits(),
            control_b.predicted_intercept_state.jd.to_bits()
        );
        assert_eq!(
            control_a.selected_target_state,
            control_b.selected_target_state
        );

        let release_position_delta_km = norm3(&[
            control_a.release_pre_control_state.eci[0] - control_b.release_pre_control_state.eci[0],
            control_a.release_pre_control_state.eci[1] - control_b.release_pre_control_state.eci[1],
            control_a.release_pre_control_state.eci[2] - control_b.release_pre_control_state.eci[2],
        ]);
        let control_delta_km_s = norm3(&[
            control_a.release_control_dv[0] - control_b.release_control_dv[0],
            control_a.release_control_dv[1] - control_b.release_control_dv[1],
            control_a.release_control_dv[2] - control_b.release_control_dv[2],
        ]);
        assert!(release_position_delta_km > 1e-8);
        assert!(control_delta_km_s > 1e-12);
        for control in [&control_a, &control_b] {
            let miss_km = norm3(&[
                control.predicted_intercept_state.eci[0] - target_i[0],
                control.predicted_intercept_state.eci[1] - target_i[1],
                control.predicted_intercept_state.eci[2] - target_i[2],
            ]);
            assert!(miss_km <= post_a.dust_intercept_tol_km);
            assert!(matches!(
                control.status,
                PostprocessControlStatus::Applied | PostprocessControlStatus::AppliedZero
            ));
        }
        assert_eq!(post_a.canister_tof_fraction.to_bits(), 0.17_f64.to_bits());
        assert_eq!(post_a.canister_am.to_bits(), 0.0042_f64.to_bits());
        assert_eq!(post_b.canister_am.to_bits(), 0.042_f64.to_bits());
    }

    #[test]
    fn release_control_launch_radius_failure_is_deterministic_physical() {
        let (mut plan, target_i, intercept_jd, conf, post, coeffs) =
            release_control_fixture(0.5, 7200.0).expect("release-control fixture");
        plan.min_radius_km = 8000.0;

        let result = build_release_control(
            &plan,
            &target_i,
            intercept_jd,
            intercept_jd,
            &conf,
            &post,
            &coeffs,
            None,
            None,
            None,
            ConjunctionDiagnostic::Compute,
        );

        assert!(matches!(
            result,
            Err(PostprocessControlStatus::DeterministicPhysicalInfeasible)
        ));
    }

    #[test]
    fn release_control_exceeding_physical_limit_fails_instead_of_clipping() {
        let (plan, mut target_i, intercept_jd, conf, mut post, coeffs) =
            release_control_fixture(0.5, 7200.0).expect("release-control fixture");
        target_i[0] += 10.0;
        post.fix_ls_skip_tol = 0.0;
        post.max_physical_dv_kms = 1e-8;
        let result = build_release_control(
            &plan,
            &target_i,
            intercept_jd,
            intercept_jd,
            &conf,
            &post,
            &coeffs,
            None,
            None,
            None,
            ConjunctionDiagnostic::Compute,
        );
        assert!(matches!(
            result,
            Err(PostprocessControlStatus::ControlSolutionConstraintViolation)
        ));
    }

    #[test]
    fn nonzero_release_control_is_counted_once_by_summary_and_full_distribution() {
        let (plan, mut target_i, intercept_jd, mut conf, mut post, coeffs) =
            release_control_fixture(0.5, 7200.0).expect("release-control fixture");
        target_i[0] += 10.0;
        conf.splitting_criterion = "maxvar".to_string();
        post.fix_ls_skip_tol = 0.0;
        post.max_physical_dv_kms = 1.0;
        let (control, _, _) = build_release_control(
            &plan,
            &target_i,
            intercept_jd,
            intercept_jd,
            &conf,
            &post,
            &coeffs,
            None,
            None,
            None,
            ConjunctionDiagnostic::Compute,
        )
        .expect("nonzero control under physical cap");
        assert!(control.release_control_dv_norm > 1e-8);
        let summary = compute_corrected_dust_state_summary(
            &plan,
            &target_i,
            intercept_jd,
            intercept_jd,
            &conf,
            &post,
            &coeffs,
            None,
            None,
            None,
            None,
        )
        .expect("summary release-control computation")
        .expect("summary from one release control");
        let distribution = compute_corrected_dust_state(CorrectedDustStateRequest {
            plan: &plan,
            target_intercept_state: &target_i,
            intercept_jd,
            conjunction_jd: intercept_jd,
            conf: &conf,
            post: &post,
            coeffs: &coeffs,
            split_alpha: None,
            split_axis: None,
            release_covariance: None,
            release_distribution: None,
        })
        .expect("full distribution from one release control");
        assert!((summary.correction_dv_norm - control.release_control_dv_norm).abs() < 1e-12);
        assert!((distribution.correction_dv_norm - control.release_control_dv_norm).abs() < 1e-12);
        for (summary_mean, distribution_mean) in
            summary.dust_mean.iter().zip(distribution.dust_mean)
        {
            assert!((*summary_mean - distribution_mean).abs() < 1e-9);
        }
    }

    #[test]
    fn hf_conjunction_diagnostic_uses_selected_target_ballistics() {
        let c = Arc::new(vec![1.0, 0.0, 0.0, 0.0]);
        let s = Arc::new(vec![0.0; 4]);
        let packed = Arc::new(
            satpy_core::pack_gravity_coeffs(&c, &s, 2, 0)
                .expect("test gravity coefficients are valid"),
        );
        let coeffs = GlobalCoeffs {
            packed: Some(packed),
            missing: false,
        };
        let conf = PhysicsConfig {
            use_high_fidelity: true,
            require_hf_transfer_correction: true,
            force_flags: lightyear_odeint_rs::types::ForceFlags::SRP,
            sun_pos: Some([149_597_870.7, 0.0, 0.0]),
            dt_max: 30.0,
            tolerance: 1e-8,
            ..PhysicsConfig::default()
        };
        let i = 2_460_000.5;
        let predicted = StampedEciState::new([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], i);
        let target = StampedEciState::new([7001.0, 0.0, 0.0, 0.0, 7.49, 0.0], i);
        let dust_ctx =
            build_ctx(i, &conf, &coeffs, 0.0, 0.0, 0.0).expect("dust propagation context");
        let reference_am = 0.01;
        let reference_force =
            BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, reference_am, 2.2, 1.3);
        let mut reference_target_ctx = build_ctx(i, &conf, &coeffs, reference_am, 2.2, 1.3)
            .expect("reference target propagation context");
        reference_target_ctx.target_propagation_authority =
            TargetPropagationAuthority::HighFidelity;
        reference_target_ctx.target_body_force = reference_force;
        let reference = diagnostic_conjunction_separation(
            &predicted,
            &target,
            i + SEC_PER_DAY / SEC_PER_DAY,
            &dust_ctx,
            &reference_target_ctx,
            &conf,
            reference_force,
            TargetPropagationAuthority::HighFidelity,
        )
        .expect("reference HF diagnostic")
        .expect("HF diagnostic with positive reference target AM");
        let selected_force =
            BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 50.0, 2.2, 1.3);
        let mut selected_target_ctx = build_ctx(i, &conf, &coeffs, 50.0, 2.2, 1.3)
            .expect("selected target propagation context");
        selected_target_ctx.target_propagation_authority = TargetPropagationAuthority::HighFidelity;
        selected_target_ctx.target_body_force = selected_force;
        let ballistic = diagnostic_conjunction_separation(
            &predicted,
            &target,
            i + SEC_PER_DAY / SEC_PER_DAY,
            &dust_ctx,
            &selected_target_ctx,
            &conf,
            selected_force,
            TargetPropagationAuthority::HighFidelity,
        )
        .expect("selected HF diagnostic")
        .expect("HF diagnostic with selected target AM");
        assert!(
            (ballistic - reference).abs() > 1e-9,
            "target AM must influence HF conjunction propagation"
        );
    }

    #[test]
    fn conjunction_diagnostic_obeys_explicit_catalogue_target_authority() {
        let c = Arc::new(vec![1.0, 0.0, 0.0, 0.0]);
        let s = Arc::new(vec![0.0; 4]);
        let packed = Arc::new(
            satpy_core::pack_gravity_coeffs(&c, &s, 2, 0)
                .expect("test gravity coefficients are valid"),
        );
        let coeffs = GlobalCoeffs {
            packed: Some(packed),
            missing: false,
        };
        let conf = PhysicsConfig {
            use_high_fidelity: true,
            require_hf_transfer_correction: true,
            dt_max: 30.0,
            tolerance: 1e-8,
            ..PhysicsConfig::default()
        };
        let intercept_jd = 2_460_000.5;
        let predicted = StampedEciState::new([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], intercept_jd);
        let target = StampedEciState::new([7001.0, 0.0, 0.0, 0.0, 7.49, 0.0], intercept_jd);
        let dust_ctx = build_ctx(intercept_jd, &conf, &coeffs, 0.0, 0.0, 0.0)
            .expect("dust propagation context");
        let target_ctx = build_ctx(intercept_jd, &conf, &coeffs, 0.0, 2.2, 1.3)
            .expect("target propagation context");
        let target_force =
            BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 0.0, 2.2, 1.3);
        let conjunction_jd = intercept_jd + 86400.0 / SEC_PER_DAY;
        let analytical = diagnostic_conjunction_separation(
            &predicted,
            &target,
            conjunction_jd,
            &dust_ctx,
            &target_ctx,
            &conf,
            target_force,
            TargetPropagationAuthority::AnalyticalKepler,
        )
        .expect("analytical catalogue diagnostic")
        .expect("hybrid diagnostic with analytical catalogue target");
        let mf = diagnostic_conjunction_separation(
            &predicted,
            &target,
            conjunction_jd,
            &dust_ctx,
            &target_ctx,
            &conf,
            target_force,
            TargetPropagationAuthority::MfJ2,
        )
        .expect("MF catalogue diagnostic")
        .expect("hybrid diagnostic with MF catalogue target");

        assert!(analytical.is_finite());
        assert!(mf.is_finite());
        assert_ne!(
            analytical.to_bits(),
            mf.to_bits(),
            "explicit target authority must govern I-to-Cj propagation"
        );
    }
}
