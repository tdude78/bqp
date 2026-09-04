//! Post-optimization verification for two-phase transfer solutions.
//!
//! This module provides independent verification that forward-propagates
//! the entire maneuver sequence and validates the final intercept distance.

use crate::evaluate::{
    eci_to_equinoctial, propagate_candidate_state_at_epoch,
    propagate_high_fidelity_state_at_epoch_checked,
    propagate_high_fidelity_state_independent_witness,
    propagate_high_fidelity_target_at_authoritative_offset_checked, propagation_epoch_for_segment,
    EvaluationArithmeticOverflow, TransferPropagationFailure,
};
use crate::types::{all_finite, EciBasicOrbit, PlanContext, PlanResult, MIN_TOF};
use satpy_core::norm3;

const DV_TOL_KM_S: f64 = 1e-9;
const STATE_POS_TOL_KM: f64 = 1e-3;
const STATE_VEL_TOL_KM_S: f64 = 1e-6;
const TIMELINE_TOL_S: f64 = 1e-3;

#[derive(Clone, Copy, Debug, PartialEq)]
struct ReplaySegmentEpochs {
    phase: f64,
    wait: f64,
    transfer: f64,
    intercept: f64,
}

#[derive(Clone, Copy, Debug)]
struct VerificationTiming {
    time2phase: f64,
    waittime: f64,
    tof: f64,
    total_time: f64,
}

#[derive(Clone, Copy, Debug)]
struct VerificationReplayStates {
    dep_at_transfer: [f64; 6],
    sat_at_intercept: [f64; 6],
    tgt_at_intercept: [f64; 6],
    deployer_at_intercept: [f64; 6],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayLeg {
    E0ToPhase,
    PhaseCoast,
    TransferArc,
    Target,
    DeployerArc,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReplayFailure {
    InvalidInput,
    Conversion(ReplayLeg),
    Propagation {
        leg: ReplayLeg,
        source: TransferPropagationFailure,
    },
}

impl std::fmt::Display for ReplayFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput => formatter.write_str("transfer replay invalid input"),
            Self::Conversion(leg) => write!(formatter, "transfer replay {leg:?} conversion"),
            Self::Propagation { leg, source } => {
                write!(formatter, "transfer replay {leg:?}: {source}")
            }
        }
    }
}

impl std::error::Error for ReplayFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Propagation { source, .. } => Some(source),
            Self::InvalidInput | Self::Conversion(_) => None,
        }
    }
}

#[inline]
fn replay_segment_epochs(
    base_jd: f64,
    phase_s: f64,
    wait_s: f64,
    transfer_s: f64,
) -> ReplaySegmentEpochs {
    ReplaySegmentEpochs {
        phase: base_jd,
        wait: propagation_epoch_for_segment(base_jd, phase_s),
        transfer: propagation_epoch_for_segment(base_jd, phase_s + wait_s),
        intercept: propagation_epoch_for_segment(base_jd, phase_s + wait_s + transfer_s),
    }
}

fn propagate_replay_target(
    ctx: &PlanContext,
    dt: f64,
) -> Result<[f64; 6], TransferPropagationFailure> {
    if matches!(
        ctx.target_propagation_authority,
        crate::types::TargetPropagationAuthority::HighFidelity
    ) {
        propagate_high_fidelity_target_at_authoritative_offset_checked(ctx, dt)
    } else {
        crate::evaluate::propagate_candidate_target_at_authoritative_offset(ctx, dt)
            .map_err(|_: EvaluationArithmeticOverflow| {
                TransferPropagationFailure::ArithmeticOverflow
            })?
            .ok_or(TransferPropagationFailure::NonFiniteOutput)
    }
}

#[inline]
fn pre_hf_j2_residual_blocks_verification(
    use_high_fidelity: bool,
    iteration_count: u32,
    residual_m: f64,
    target_m: f64,
) -> bool {
    !use_high_fidelity
        && iteration_count > 0
        && (!residual_m.is_finite() || residual_m > target_m + STATE_POS_TOL_KM * 1000.0)
}

/// Result of post-optimization verification
#[derive(Clone, Debug)]
pub struct VerificationResult {
    /// True if verification passed (distance <= tolerance)
    pub verified: bool,
    /// Actual final distance between satellite and target (km)
    pub actual_distance_km: f64,
    /// Tolerance used for verification (km)
    pub tolerance_km: f64,
    /// Margin: tolerance minus actual distance (positive means pass, negative means fail).
    pub margin_km: f64,
    /// Failure message if verification failed. The closed set of fixed
    /// messages is [`VerificationFailureToken`] (zero-alloc `Cow::Borrowed`);
    /// only genuinely dynamic messages allocate (`Cow::Owned`).
    pub propagation_error: Option<std::borrow::Cow<'static, str>>,
}

/// Closed set of fixed verification-failure messages.
///
/// Backlog-38 token pattern: every constant failure text lives here exactly
/// once, rendered by [`Self::as_str`] as the byte-identical historical
/// literal, so the fixed failure set is greppable and constructing one never
/// allocates.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerificationFailureToken {
    ResultMarkedInvalid,
    HfEndpointResidualOutOfTolerance,
    InvalidTimingParameters,
    PhaseTimelineOverflowed,
    RevolutionCapMissingDeployerPeriod,
    DeployerInitialStateNonFinite,
    TargetInitialStateNonFinite,
    PhaseDvNonFinite,
    TransferDvNonFinite,
    ArrivalDvNonFinite,
    DeployerEciToEquinoctialFailed,
    PostPhaseBurnStateNonFinite,
    PostPhaseStateToEquinoctialFailed,
    PostTransferBurnStateNonFinite,
    PostTransferStateToEquinoctialFailed,
    DeployerReleaseStateToEquinoctialFailed,
    ComputedDistanceNonFinite,
    ComputedDeployerDistanceNonFinite,
}

impl VerificationFailureToken {
    /// The byte-identical historical message literal for this token.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ResultMarkedInvalid => "Result marked as invalid",
            Self::HfEndpointResidualOutOfTolerance => {
                "HF verification requires finite post-HF endpoint residual within tolerance"
            }
            Self::InvalidTimingParameters => {
                "Invalid timing parameters (must be non-negative and finite)"
            }
            Self::PhaseTimelineOverflowed => {
                "Invalid timing parameters (phase timeline overflowed)"
            }
            Self::RevolutionCapMissingDeployerPeriod => {
                "Cannot enforce revolution cap without a finite context deployer period"
            }
            Self::DeployerInitialStateNonFinite => "Deployer initial state contains NaN or Inf",
            Self::TargetInitialStateNonFinite => "Target initial state contains NaN or Inf",
            Self::PhaseDvNonFinite => "Phase ΔV contains NaN or Inf",
            Self::TransferDvNonFinite => "Transfer ΔV contains NaN or Inf",
            Self::ArrivalDvNonFinite => "Arrival ΔV contains NaN or Inf",
            Self::DeployerEciToEquinoctialFailed => "Failed to convert deployer ECI to equinoctial",
            Self::PostPhaseBurnStateNonFinite => "Invalid state after phase burn (NaN or Inf)",
            Self::PostPhaseStateToEquinoctialFailed => {
                "Failed to convert post-phase state to equinoctial"
            }
            Self::PostTransferBurnStateNonFinite => {
                "Invalid state after transfer burn (NaN or Inf)"
            }
            Self::PostTransferStateToEquinoctialFailed => {
                "Failed to convert post-transfer state to equinoctial"
            }
            Self::DeployerReleaseStateToEquinoctialFailed => {
                "Failed to convert deployer release state to equinoctial"
            }
            Self::ComputedDistanceNonFinite => "Computed distance is NaN or Inf",
            Self::ComputedDeployerDistanceNonFinite => "Computed deployer distance is NaN or Inf",
        }
    }
}

impl From<VerificationFailureToken> for std::borrow::Cow<'static, str> {
    fn from(token: VerificationFailureToken) -> Self {
        std::borrow::Cow::Borrowed(token.as_str())
    }
}

/// Propagate one replay leg, optionally walking it in bounded sub-segments.
///
/// `max_segment_s == None` is the historical single-call behaviour and forwards
/// verbatim to [`propagate_state_at_epoch`]; the default MF replay path uses it
/// and is therefore bit-identical to the pre-segmentation build.
///
/// `Some(cap)` re-osculates the equinoctial reference every `cap` seconds. Encke
/// integration needs that: one call cannot span an arc longer than the
/// integrator's own rectification window, so a multi-hour HF leg issued as a
/// single call simply returns no state.
///
/// Not [`propagate_independent_witness_leg`]: that leg exists specifically
/// so verification does not replay the code path it is verifying.
/// Substituting this replay leg where the witness leg belongs silently
/// defeats that independence guarantee.
///
/// # Errors
///
/// Returns [`TransferPropagationFailure::InvalidInput`] for a non-finite or
/// negative `dt` (this leg is forward-only; a negative arc must go through
/// [`propagate_replay_leg_backward`], never be silently truncated),
/// [`TransferPropagationFailure::ArithmeticOverflow`] when the
/// low-fidelity evaluator's accounting cannot be represented, the propagator's
/// own [`TransferPropagationFailure`] when a segment fails, and
/// [`TransferPropagationFailure::NonFiniteOutput`] when a segment yields no
/// state, when the state cannot be re-expressed in equinoctial elements, or
/// when the segment walk stops making progress.
pub fn propagate_replay_leg(
    eci: &[f64; 6],
    equ: &[f64; 6],
    dt: f64,
    source_jd: f64,
    body_force: crate::types::BodyForceConfig,
    ctx: &PlanContext,
    max_segment_s: Option<f64>,
) -> Result<[f64; 6], TransferPropagationFailure> {
    // Fail closed on a signed arc: every caller of this leg pre-guards dt >= 0
    // today, and without this rejection a negative dt under `Some(cap)` would
    // hand the propagator a backward arc this walk was never audited for.
    if !dt.is_finite() || dt < 0.0 {
        return Err(TransferPropagationFailure::InvalidInput);
    }
    let Some(cap) = max_segment_s else {
        return if ctx.execution_policy.use_high_fidelity {
            propagate_high_fidelity_state_at_epoch_checked(equ, dt, source_jd, body_force, ctx)
        } else {
            propagate_candidate_state_at_epoch(equ, dt, source_jd, body_force, ctx)
                .map_err(|_: EvaluationArithmeticOverflow| {
                    TransferPropagationFailure::ArithmeticOverflow
                })?
                .ok_or(TransferPropagationFailure::NonFiniteOutput)
        };
    };
    // An unusable cap, or a leg already short enough, is the single-call case verbatim.
    let cap_usable = cap.is_finite() && cap > 0.0 && dt.is_finite();
    if !cap_usable || dt <= cap {
        return if ctx.execution_policy.use_high_fidelity {
            propagate_high_fidelity_state_at_epoch_checked(equ, dt, source_jd, body_force, ctx)
        } else {
            propagate_candidate_state_at_epoch(equ, dt, source_jd, body_force, ctx)
                .map_err(|_: EvaluationArithmeticOverflow| {
                    TransferPropagationFailure::ArithmeticOverflow
                })?
                .ok_or(TransferPropagationFailure::NonFiniteOutput)
        };
    }
    let mut state = *eci;
    let mut state_equ = *equ;
    let mut elapsed = 0.0;
    loop {
        if elapsed >= dt {
            break;
        }
        let step = (dt - elapsed).min(cap);
        let next = if ctx.execution_policy.use_high_fidelity {
            propagate_high_fidelity_state_at_epoch_checked(
                &state_equ,
                step,
                propagation_epoch_for_segment(source_jd, elapsed),
                body_force,
                ctx,
            )?
        } else {
            propagate_candidate_state_at_epoch(
                &state_equ,
                step,
                propagation_epoch_for_segment(source_jd, elapsed),
                body_force,
                ctx,
            )
            .map_err(|_: EvaluationArithmeticOverflow| {
                TransferPropagationFailure::ArithmeticOverflow
            })?
            .ok_or(TransferPropagationFailure::NonFiniteOutput)?
        };
        state = next;
        if !eci_to_equinoctial(&state, &mut state_equ) {
            return Err(TransferPropagationFailure::NonFiniteOutput);
        }
        let next_elapsed = elapsed + step;
        if !next_elapsed.is_finite() || next_elapsed <= elapsed {
            return Err(TransferPropagationFailure::NonFiniteOutput);
        }
        elapsed = next_elapsed;
    }
    all_finite(&state)
        .then_some(state)
        .ok_or(TransferPropagationFailure::NonFiniteOutput)
}

/// Backward twin of [`propagate_replay_leg`] with a mandatory segment cap.
///
/// Exists for the fixed-`Ic` Stage-1 target leg when the candidate intercept
/// precedes the sealed target anchor: production physics backward-extrapolates
/// targets from the common anchor by design, and the production integrator is
/// direction-aware (`integrate_final_checked` accepts `t_final < t0`). The
/// walk mirrors the forward `Some(cap)` segment walk with strictly decreasing
/// `elapsed` and negative per-segment steps; the forward leg itself remains
/// forward-only and rejects a negative `dt`.
///
/// # Errors
///
/// Returns [`TransferPropagationFailure::InvalidInput`] for a non-finite or
/// non-positive `max_segment_s` or a non-finite or non-negative `dt`, the
/// propagator's own [`TransferPropagationFailure`] when a segment fails, and
/// [`TransferPropagationFailure::NonFiniteOutput`] when a segment yields no
/// state, when the state cannot be re-expressed in equinoctial elements, or
/// when the segment walk stops making progress.
pub fn propagate_replay_leg_backward(
    eci: &[f64; 6],
    equ: &[f64; 6],
    dt: f64,
    source_jd: f64,
    body_force: crate::types::BodyForceConfig,
    ctx: &PlanContext,
    max_segment_s: f64,
) -> Result<[f64; 6], TransferPropagationFailure> {
    if !max_segment_s.is_finite() || max_segment_s <= 0.0 || !dt.is_finite() || dt >= 0.0 {
        return Err(TransferPropagationFailure::InvalidInput);
    }
    let mut state = *eci;
    let mut state_equ = *equ;
    let mut elapsed = 0.0;
    loop {
        if elapsed <= dt {
            break;
        }
        let step = (dt - elapsed).max(-max_segment_s);
        let next = if ctx.execution_policy.use_high_fidelity {
            propagate_high_fidelity_state_at_epoch_checked(
                &state_equ,
                step,
                propagation_epoch_for_segment(source_jd, elapsed),
                body_force,
                ctx,
            )?
        } else {
            propagate_candidate_state_at_epoch(
                &state_equ,
                step,
                propagation_epoch_for_segment(source_jd, elapsed),
                body_force,
                ctx,
            )
            .map_err(|_: EvaluationArithmeticOverflow| {
                TransferPropagationFailure::ArithmeticOverflow
            })?
            .ok_or(TransferPropagationFailure::NonFiniteOutput)?
        };
        state = next;
        if !eci_to_equinoctial(&state, &mut state_equ) {
            return Err(TransferPropagationFailure::NonFiniteOutput);
        }
        let next_elapsed = elapsed + step;
        if !next_elapsed.is_finite() || next_elapsed >= elapsed {
            return Err(TransferPropagationFailure::NonFiniteOutput);
        }
        elapsed = next_elapsed;
    }
    all_finite(&state)
        .then_some(state)
        .ok_or(TransferPropagationFailure::NonFiniteOutput)
}

/// Not [`propagate_replay_leg`]: this leg is the independent witness.
///
/// It exists so verification does not reuse the code path under verification.
/// Substitute the replay leg here and the independence guarantee is silently
/// defeated.
///
/// A negative `dt` is a backward arc and is walked by a mirrored backward
/// segment loop (strictly decreasing `elapsed`, per-segment step
/// `max(dt - elapsed, -max_segment_s)`): the sealed B500 bank legitimately
/// contains events whose solver window opens before the common anchor, and
/// production physics backward-extrapolates the target from that anchor by
/// design, so the witness must express what production consumed. The forward
/// loop is untouched.
///
/// # Errors
///
/// Returns [`TransferPropagationFailure::InvalidInput`] for a non-finite or
/// non-positive `max_segment_s` or a non-finite `dt`, the witness
/// propagator's own [`TransferPropagationFailure`] when a segment fails, and
/// [`TransferPropagationFailure::NonFiniteOutput`] when the state cannot be
/// re-expressed in equinoctial elements, when the segment walk stops making
/// progress, or when the final state is not finite.
pub fn propagate_independent_witness_leg(
    eci: &[f64; 6],
    equ: &[f64; 6],
    dt: f64,
    source_jd: f64,
    body_force: crate::types::BodyForceConfig,
    ctx: &PlanContext,
    max_segment_s: f64,
    dt_max_s: f64,
    tolerance: f64,
) -> Result<[f64; 6], TransferPropagationFailure> {
    if !max_segment_s.is_finite() || max_segment_s <= 0.0 || !dt.is_finite() {
        return Err(TransferPropagationFailure::InvalidInput);
    }
    if dt < 0.0 {
        return propagate_independent_witness_leg_backward(
            eci,
            equ,
            dt,
            source_jd,
            body_force,
            ctx,
            max_segment_s,
            dt_max_s,
            tolerance,
        );
    }
    let mut state = *eci;
    let mut state_equ = *equ;
    let mut elapsed = 0.0;
    loop {
        if elapsed >= dt {
            break;
        }
        let step = (dt - elapsed).min(max_segment_s);
        state = propagate_high_fidelity_state_independent_witness(
            &state_equ,
            step,
            propagation_epoch_for_segment(source_jd, elapsed),
            body_force,
            ctx,
            dt_max_s,
            tolerance,
        )?;
        if !eci_to_equinoctial(&state, &mut state_equ) {
            return Err(TransferPropagationFailure::NonFiniteOutput);
        }
        let next_elapsed = elapsed + step;
        if !next_elapsed.is_finite() || next_elapsed <= elapsed {
            return Err(TransferPropagationFailure::NonFiniteOutput);
        }
        elapsed = next_elapsed;
    }
    all_finite(&state)
        .then_some(state)
        .ok_or(TransferPropagationFailure::NonFiniteOutput)
}

/// Mirrored backward segment walk of [`propagate_independent_witness_leg`].
///
/// `elapsed` decreases strictly from `0` toward `dt < 0`; each segment hands
/// the independent witness integrator a negative step, which it walks with its
/// own mirrored backward RK4 loop. Failure taxonomy is identical to the
/// forward walk.
fn propagate_independent_witness_leg_backward(
    eci: &[f64; 6],
    equ: &[f64; 6],
    dt: f64,
    source_jd: f64,
    body_force: crate::types::BodyForceConfig,
    ctx: &PlanContext,
    max_segment_s: f64,
    dt_max_s: f64,
    tolerance: f64,
) -> Result<[f64; 6], TransferPropagationFailure> {
    debug_assert!(dt < 0.0, "backward witness walk requires dt < 0");
    let mut state = *eci;
    let mut state_equ = *equ;
    let mut elapsed = 0.0;
    loop {
        if elapsed <= dt {
            break;
        }
        let step = (dt - elapsed).max(-max_segment_s);
        state = propagate_high_fidelity_state_independent_witness(
            &state_equ,
            step,
            propagation_epoch_for_segment(source_jd, elapsed),
            body_force,
            ctx,
            dt_max_s,
            tolerance,
        )?;
        if !eci_to_equinoctial(&state, &mut state_equ) {
            return Err(TransferPropagationFailure::NonFiniteOutput);
        }
        let next_elapsed = elapsed + step;
        if !next_elapsed.is_finite() || next_elapsed >= elapsed {
            return Err(TransferPropagationFailure::NonFiniteOutput);
        }
        elapsed = next_elapsed;
    }
    all_finite(&state)
        .then_some(state)
        .ok_or(TransferPropagationFailure::NonFiniteOutput)
}

/// Forward replay one immutable maneuver sequence from its original E0 state.
///
/// This deliberately consumes the stored phase and transfer vectors.  It does
/// not call any optimizer or Lambert search; callers use it to rebuild the
/// reported endpoint before [`verify_transfer_result`] checks constraints.
///
/// # Errors
///
/// Returns a typed failure when stored controls are invalid, a state conversion
/// fails, or a replay leg cannot propagate.
pub fn replay_transfer_controls(
    result: &PlanResult,
    ctx: &PlanContext,
) -> Result<PlanResult, ReplayFailure> {
    replay_transfer_controls_segmented(result, ctx, None)
}

/// [`replay_transfer_controls`] with an explicit propagation segment cap.
///
/// `max_segment_s` is `None` for every historical caller. HF acceptance passes
/// `Some(_)` because Encke legs must be rectified periodically; see
/// [`crate::hf_acceptance`].
///
/// # Errors
///
/// Returns a typed failure when stored controls are invalid, a state conversion
/// fails, or a replay leg cannot propagate.
pub fn replay_transfer_controls_segmented(
    result: &PlanResult,
    ctx: &PlanContext,
    max_segment_s: Option<f64>,
) -> Result<PlanResult, ReplayFailure> {
    if !result.valid || !all_finite(&result.phase_dv) || !all_finite(&result.transfer_dv) {
        return Err(ReplayFailure::InvalidInput);
    }
    let time2phase = result.time2phase;
    let waittime = result.waittime;
    let tof = result.tof;
    if !(time2phase.is_finite()
        && time2phase >= 0.0
        && waittime.is_finite()
        && waittime >= 0.0
        && tof.is_finite()
        && tof >= MIN_TOF)
    {
        return Err(ReplayFailure::InvalidInput);
    }
    let epochs = replay_segment_epochs(ctx.epoch_jd, time2phase, waittime, tof);
    let transfer_body = ctx.transfer_body_force();
    let mut dep_equ_start = [0.0; 6];
    if !eci_to_equinoctial(&ctx.dep_eci, &mut dep_equ_start) {
        return Err(ReplayFailure::Conversion(ReplayLeg::E0ToPhase));
    }
    let dep_at_phase = propagate_replay_leg(
        &ctx.dep_eci,
        &dep_equ_start,
        time2phase,
        epochs.phase,
        transfer_body,
        ctx,
        max_segment_s,
    )
    .map_err(|source| ReplayFailure::Propagation {
        leg: ReplayLeg::E0ToPhase,
        source,
    })?;
    let dep_after_phase = apply_delta_v(dep_at_phase, result.phase_dv);
    let mut dep_equ_after_phase = [0.0; 6];
    if !eci_to_equinoctial(&dep_after_phase, &mut dep_equ_after_phase) {
        return Err(ReplayFailure::Conversion(ReplayLeg::PhaseCoast));
    }
    let dep_at_transfer = propagate_replay_leg(
        &dep_after_phase,
        &dep_equ_after_phase,
        waittime,
        epochs.wait,
        transfer_body,
        ctx,
        max_segment_s,
    )
    .map_err(|source| ReplayFailure::Propagation {
        leg: ReplayLeg::PhaseCoast,
        source,
    })?;
    let dep_after_transfer = apply_delta_v(dep_at_transfer, result.transfer_dv);
    let mut dep_equ_after_transfer = [0.0; 6];
    if !eci_to_equinoctial(&dep_after_transfer, &mut dep_equ_after_transfer) {
        return Err(ReplayFailure::Conversion(ReplayLeg::TransferArc));
    }
    let payload_at_intercept = propagate_replay_leg(
        &dep_after_transfer,
        &dep_equ_after_transfer,
        tof,
        epochs.transfer,
        transfer_body,
        ctx,
        max_segment_s,
    )
    .map_err(|source| ReplayFailure::Propagation {
        leg: ReplayLeg::TransferArc,
        source,
    })?;
    let target_at_intercept =
        propagate_replay_target(ctx, time2phase + waittime + tof).map_err(|source| {
            ReplayFailure::Propagation {
                leg: ReplayLeg::Target,
                source,
            }
        })?;
    let mut dep_equ_at_transfer = [0.0; 6];
    if !eci_to_equinoctial(&dep_at_transfer, &mut dep_equ_at_transfer) {
        return Err(ReplayFailure::Conversion(ReplayLeg::DeployerArc));
    }
    let deployer_at_intercept = propagate_replay_leg(
        &dep_at_transfer,
        &dep_equ_at_transfer,
        tof,
        epochs.transfer,
        transfer_body,
        ctx,
        max_segment_s,
    )
    .map_err(|source| ReplayFailure::Propagation {
        leg: ReplayLeg::DeployerArc,
        source,
    })?;
    let mut replayed = result.clone();
    replayed.release_state = dep_at_transfer;
    replayed.payload_intercept_state = payload_at_intercept;
    replayed.target_intercept_state = target_at_intercept;
    replayed.deployer_intercept_state = deployer_at_intercept;
    replayed.arrival_dv = [
        target_at_intercept[3] - payload_at_intercept[3],
        target_at_intercept[4] - payload_at_intercept[4],
        target_at_intercept[5] - payload_at_intercept[5],
    ];
    replayed.arrival_dv_norm = norm3(&replayed.arrival_dv);
    replayed.transfer_dv_norm = norm3(&replayed.transfer_dv);
    replayed.phase_dv_norm = norm3(&replayed.phase_dv);
    replayed.cost = replayed.phase_dv_norm + replayed.transfer_dv_norm;
    replayed.intercept_jd = epochs.intercept;
    replayed.waittime_jd_start = epochs.wait;
    replayed.tof_jd_start = epochs.transfer;
    replayed.distance = distance3(&payload_at_intercept, &target_at_intercept);
    replayed.deployer_distance = distance3(&deployer_at_intercept, &target_at_intercept);
    replayed.post_hf_endpoint_residual_m = replayed.distance * 1000.0;
    replayed.valid = all_finite(&payload_at_intercept)
        && all_finite(&target_at_intercept)
        && replayed.distance.is_finite();
    Ok(replayed)
}

impl Default for VerificationResult {
    fn default() -> Self {
        Self {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km: 0.0,
            margin_km: f64::NEG_INFINITY,
            propagation_error: None,
        }
    }
}

fn validate_verification_timing_and_authority(
    result: &PlanResult,
    ctx: &PlanContext,
    tolerance_km: f64,
) -> Result<VerificationTiming, VerificationResult> {
    if !result.valid {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(VerificationFailureToken::ResultMarkedInvalid.into()),
        });
    }
    if ctx.execution_policy.use_high_fidelity
        && !crate::evaluate::post_hf_residual_accepts(
            result.post_hf_endpoint_residual_m,
            tolerance_km * 1000.0,
        )
    {
        return Err(failed_verification(
            tolerance_km,
            VerificationFailureToken::HfEndpointResidualOutOfTolerance,
        ));
    }
    if !tolerance_km.is_finite() || tolerance_km <= 0.0 {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(
                format!("Invalid tolerance: {tolerance_km:.6} km (must be positive and finite)")
                    .into(),
            ),
        });
    }

    let time2phase = result.time2phase;
    let waittime = result.waittime;
    let tof = result.tof;
    if !time2phase.is_finite()
        || time2phase < 0.0
        || !waittime.is_finite()
        || waittime < 0.0
        || !tof.is_finite()
        || tof < 0.0
    {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(VerificationFailureToken::InvalidTimingParameters.into()),
        });
    }
    if tof < MIN_TOF {
        return Err(failed_verification(
            tolerance_km,
            format!("Transfer time of flight below solver minimum: {tof:.6} < {MIN_TOF:.6} s"),
        ));
    }
    let transfer_start_time = time2phase + waittime;
    let total_time = transfer_start_time + tof;
    if !transfer_start_time.is_finite() || !total_time.is_finite() {
        return Err(failed_verification(
            tolerance_km,
            VerificationFailureToken::PhaseTimelineOverflowed,
        ));
    }
    if limit_exceeded(
        transfer_start_time + MIN_TOF,
        ctx.intercept_time_budget_s(),
        TIMELINE_TOL_S,
    ) {
        return Err(failed_verification(
            tolerance_km,
            format!(
                "Transfer phase has no available overlap with solver horizon: phase/wait end={transfer_start_time:.6} s, minimum TOF={MIN_TOF:.6} s, budget={:.6} s",
                ctx.intercept_time_budget_s()
            ),
        ));
    }
    if limit_exceeded(total_time, ctx.intercept_time_budget_s(), 1e-9) {
        return Err(failed_verification(
            tolerance_km,
            format!(
                "Transfer time exceeds budget: {total_time:.6} > {:.6} s",
                ctx.intercept_time_budget_s()
            ),
        ));
    }
    if let Some(error) = timeline_epoch_error(result, ctx, time2phase, waittime, tof) {
        return Err(failed_verification(tolerance_km, error));
    }
    if result.best_M < 0 {
        return Err(failed_verification(
            tolerance_km,
            format!("Lambert revolution count is negative: {}", result.best_M),
        ));
    }
    let max_revs = ctx.max_revs.max(0);
    if result.best_M > max_revs {
        return Err(failed_verification(
            tolerance_km,
            format!(
                "Lambert revolution count exceeds limit: {} > {}",
                result.best_M, max_revs
            ),
        ));
    }
    let dep_period = context_deployer_period_s(ctx);
    if active_positive_limit(ctx.revolution_cap) {
        let Some(dep_period) = dep_period else {
            return Err(failed_verification(
                tolerance_km,
                VerificationFailureToken::RevolutionCapMissingDeployerPeriod,
            ));
        };
        let revolution_cap_s = ctx.revolution_cap * dep_period;
        if limit_exceeded(tof, revolution_cap_s, TIMELINE_TOL_S) {
            return Err(failed_verification(
                tolerance_km,
                format!(
                    "Transfer time of flight exceeds revolution cap: {tof:.6} > {revolution_cap_s:.6} s (cap {:.6} rev)",
                    ctx.revolution_cap
                ),
            ));
        }
    }
    let target_m = ctx.j2_closure_settings.endpoint_target_km * 1000.0;
    if pre_hf_j2_residual_blocks_verification(
        ctx.execution_policy.use_high_fidelity,
        result.j2_iteration_count,
        result.j2_endpoint_residual_m,
        target_m,
    ) {
        return Err(failed_verification(
            tolerance_km,
            format!(
                "J2 endpoint residual exceeds configured target: {:.6} > {target_m:.6} m",
                result.j2_endpoint_residual_m
            ),
        ));
    }

    Ok(VerificationTiming {
        time2phase,
        waittime,
        tof,
        total_time,
    })
}

fn validate_verification_states_and_controls(
    result: &PlanResult,
    ctx: &PlanContext,
    tolerance_km: f64,
) -> Result<(), VerificationResult> {
    if !all_finite(&ctx.dep_eci) {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(VerificationFailureToken::DeployerInitialStateNonFinite.into()),
        });
    }
    if !all_finite(&ctx.tgt_eci) {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(VerificationFailureToken::TargetInitialStateNonFinite.into()),
        });
    }
    if !all_finite(&result.phase_dv) {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(VerificationFailureToken::PhaseDvNonFinite.into()),
        });
    }
    let phase_dv_norm = norm3(&result.phase_dv);
    if !norm_matches(result.phase_dv_norm, phase_dv_norm) {
        return Err(failed_verification(
            tolerance_km,
            format!(
                "Phase delta-V norm mismatch: stored={:.12} km/s, vector={phase_dv_norm:.12} km/s",
                result.phase_dv_norm
            ),
        ));
    }
    if limit_exceeded(phase_dv_norm, ctx.max_phase_dv, DV_TOL_KM_S) {
        return Err(failed_verification(
            tolerance_km,
            format!(
                "Phase delta-V exceeds limit: {phase_dv_norm:.12} > {:.12} km/s",
                ctx.max_phase_dv
            ),
        ));
    }
    if !all_finite(&result.transfer_dv) {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(VerificationFailureToken::TransferDvNonFinite.into()),
        });
    }
    let transfer_dv_norm = norm3(&result.transfer_dv);
    if !norm_matches(result.transfer_dv_norm, transfer_dv_norm) {
        return Err(failed_verification(
            tolerance_km,
            format!(
                "Transfer delta-V norm mismatch: stored={:.12} km/s, vector={transfer_dv_norm:.12} km/s",
                result.transfer_dv_norm
            ),
        ));
    }
    if limit_exceeded(transfer_dv_norm, ctx.max_transfer_dv, DV_TOL_KM_S) {
        return Err(failed_verification(
            tolerance_km,
            format!(
                "Transfer delta-V exceeds limit: {transfer_dv_norm:.12} > {:.12} km/s",
                ctx.max_transfer_dv
            ),
        ));
    }
    if !all_finite(&result.arrival_dv) {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(VerificationFailureToken::ArrivalDvNonFinite.into()),
        });
    }
    let arrival_dv_norm = norm3(&result.arrival_dv);
    if !norm_matches(result.arrival_dv_norm, arrival_dv_norm) {
        return Err(failed_verification(
            tolerance_km,
            format!(
                "Arrival delta-V norm mismatch: stored={:.12} km/s, vector={arrival_dv_norm:.12} km/s",
                result.arrival_dv_norm
            ),
        ));
    }
    Ok(())
}

fn replay_verification_states(
    result: &PlanResult,
    ctx: &PlanContext,
    timing: VerificationTiming,
    tolerance_km: f64,
) -> Result<VerificationReplayStates, VerificationResult> {
    let VerificationTiming {
        time2phase,
        waittime,
        tof,
        total_time,
    } = timing;
    let mut dep_equ_start = [0.0; 6];
    if !eci_to_equinoctial(&ctx.dep_eci, &mut dep_equ_start) {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(
                VerificationFailureToken::DeployerEciToEquinoctialFailed.into(),
            ),
        });
    }

    let epochs = replay_segment_epochs(ctx.epoch_jd, time2phase, waittime, tof);
    let transfer_body = ctx.transfer_body_force();
    let dep_at_phase = propagate_replay_leg(
        &ctx.dep_eci,
        &dep_equ_start,
        time2phase,
        epochs.phase,
        transfer_body,
        ctx,
        None,
    )
    .map_err(|error| {
        failed_verification(
            tolerance_km,
            format!("Failed to propagate deployer to phase burn (t={time2phase:.1} s): {error}"),
        )
    })?;
    let dep_after_phase = apply_delta_v(dep_at_phase, result.phase_dv);
    if !all_finite(&dep_after_phase) {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(VerificationFailureToken::PostPhaseBurnStateNonFinite.into()),
        });
    }
    if let Some(error) = orbit_bound_error("post-phase", &dep_after_phase, ctx) {
        return Err(failed_verification(tolerance_km, error));
    }

    let mut dep_equ_after_phase = [0.0; 6];
    if !eci_to_equinoctial(&dep_after_phase, &mut dep_equ_after_phase) {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(
                VerificationFailureToken::PostPhaseStateToEquinoctialFailed.into(),
            ),
        });
    }
    let dep_at_transfer = propagate_replay_leg(
        &dep_after_phase,
        &dep_equ_after_phase,
        waittime,
        epochs.wait,
        transfer_body,
        ctx,
        None,
    )
    .map_err(|error| {
        failed_verification(
            tolerance_km,
            format!("Failed to propagate through wait time (t={waittime:.1} s): {error}"),
        )
    })?;
    let dep_after_transfer = apply_delta_v(dep_at_transfer, result.transfer_dv);
    if !all_finite(&dep_after_transfer) {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(
                VerificationFailureToken::PostTransferBurnStateNonFinite.into(),
            ),
        });
    }
    if let Some(error) = orbit_bound_error("post-transfer", &dep_after_transfer, ctx) {
        return Err(failed_verification(tolerance_km, error));
    }

    let mut dep_equ_after_transfer = [0.0; 6];
    if !eci_to_equinoctial(&dep_after_transfer, &mut dep_equ_after_transfer) {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(
                VerificationFailureToken::PostTransferStateToEquinoctialFailed.into(),
            ),
        });
    }
    let sat_at_intercept = propagate_replay_leg(
        &dep_after_transfer,
        &dep_equ_after_transfer,
        tof,
        epochs.transfer,
        transfer_body,
        ctx,
        None,
    )
    .map_err(|error| {
        failed_verification(
            tolerance_km,
            format!("Failed to propagate transfer arc (t={tof:.1} s): {error}"),
        )
    })?;
    let tgt_at_intercept = propagate_replay_target(ctx, total_time).map_err(|error| {
        failed_verification(
            tolerance_km,
            format!("Failed to propagate target to intercept (t={total_time:.1} s): {error}"),
        )
    })?;

    let mut dep_equ_at_transfer = [0.0; 6];
    if !eci_to_equinoctial(&dep_at_transfer, &mut dep_equ_at_transfer) {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(
                VerificationFailureToken::DeployerReleaseStateToEquinoctialFailed.into(),
            ),
        });
    }
    let deployer_at_intercept = propagate_replay_leg(
        &dep_at_transfer,
        &dep_equ_at_transfer,
        tof,
        epochs.transfer,
        transfer_body,
        ctx,
        None,
    )
    .map_err(|error| {
        failed_verification(
            tolerance_km,
            format!("Failed to propagate deployer through transfer arc (t={tof:.1} s): {error}"),
        )
    })?;

    Ok(VerificationReplayStates {
        dep_at_transfer,
        sat_at_intercept,
        tgt_at_intercept,
        deployer_at_intercept,
    })
}

/// Verify a transfer solution by forward-propagating the maneuver sequence.
///
/// This function performs independent verification of an optimization result:
/// 1. Starts from deployer state at epoch
/// 2. Propagates to phase burn time
/// 3. Applies phase ΔV
/// 4. Propagates through wait time
/// 5. Applies transfer ΔV
/// 6. Propagates transfer arc
/// 7. Propagates target to same final epoch
/// 8. Computes final distance
///
/// # Arguments
///
/// * `result` - The optimization result to verify
/// * `ctx` - The planning context (contains initial states and physics config)
/// * `tolerance_km` - Maximum acceptable distance for verification (default: 0.010 km)
///
/// # Returns
///
/// `VerificationResult` with verification status and measured distance
#[must_use]
pub fn verify_transfer_result(
    result: &PlanResult,
    ctx: &PlanContext,
    tolerance_km: f64,
) -> VerificationResult {
    verify_transfer_result_checked(result, ctx, tolerance_km).unwrap_or_else(std::convert::identity)
}

fn verify_transfer_result_checked(
    result: &PlanResult,
    ctx: &PlanContext,
    tolerance_km: f64,
) -> Result<VerificationResult, VerificationResult> {
    let timing = validate_verification_timing_and_authority(result, ctx, tolerance_km)?;
    validate_verification_states_and_controls(result, ctx, tolerance_km)?;

    let replay = replay_verification_states(result, ctx, timing, tolerance_km)?;
    let VerificationReplayStates {
        dep_at_transfer,
        sat_at_intercept,
        tgt_at_intercept,
        deployer_at_intercept,
    } = replay;

    // Step 8: Compute final distance
    let dx = sat_at_intercept[0] - tgt_at_intercept[0];
    let dy = sat_at_intercept[1] - tgt_at_intercept[1];
    let dz = sat_at_intercept[2] - tgt_at_intercept[2];
    let distance = euclidean_norm3(dx, dy, dz);

    // Validate distance
    if !distance.is_finite() {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: f64::NAN,
            tolerance_km,
            margin_km: f64::NEG_INFINITY,
            propagation_error: Some(VerificationFailureToken::ComputedDistanceNonFinite.into()),
        });
    }
    let recomputed_post_hf_residual_m = distance * 1000.0;
    if result.post_hf_endpoint_residual_m.is_finite()
        && !scalar_matches(
            result.post_hf_endpoint_residual_m,
            recomputed_post_hf_residual_m,
            tolerance_km.max(STATE_POS_TOL_KM) * 1000.0,
        )
    {
        return Err(failed_verification_with_distance(
            tolerance_km,
            distance,
            format!(
                "Post-HF endpoint residual mismatch: stored={:.6} m, propagated={:.6} m",
                result.post_hf_endpoint_residual_m, recomputed_post_hf_residual_m
            ),
        ));
    }
    let expected_stored_distance = if distance < tolerance_km {
        0.0
    } else {
        distance
    };
    if !scalar_matches(
        result.distance,
        expected_stored_distance,
        tolerance_km.max(STATE_POS_TOL_KM),
    ) {
        return Err(failed_verification_with_distance(
            tolerance_km,
            distance,
            format!(
                "Intercept distance mismatch: stored={:.9} km, propagated={:.9} km",
                result.distance, distance
            ),
        ));
    }
    let deployer_distance = distance3(&deployer_at_intercept, &tgt_at_intercept);
    if !deployer_distance.is_finite() {
        return Err(failed_verification(
            tolerance_km,
            VerificationFailureToken::ComputedDeployerDistanceNonFinite,
        ));
    }
    if active_positive_limit(ctx.deployer_min_distance)
        && deployer_distance + tolerance_km.max(STATE_POS_TOL_KM) < ctx.deployer_min_distance
    {
        return Err(VerificationResult {
            verified: false,
            actual_distance_km: distance,
            tolerance_km,
            margin_km: tolerance_km - distance,
            propagation_error: Some(
                format!(
                    "Deployer separation below minimum: {:.6} < {:.6} km",
                    deployer_distance, ctx.deployer_min_distance
                )
                .into(),
            ),
        });
    }
    if !scalar_matches(
        result.deployer_distance,
        deployer_distance,
        tolerance_km.max(STATE_POS_TOL_KM),
    ) {
        return Err(failed_verification_with_distance(
            tolerance_km,
            distance,
            format!(
                "Deployer distance mismatch: stored={:.9} km, propagated={:.9} km",
                result.deployer_distance, deployer_distance
            ),
        ));
    }

    let expected_arrival_dv = [
        tgt_at_intercept[3] - sat_at_intercept[3],
        tgt_at_intercept[4] - sat_at_intercept[4],
        tgt_at_intercept[5] - sat_at_intercept[5],
    ];
    if !vector3_matches(&result.arrival_dv, &expected_arrival_dv, STATE_VEL_TOL_KM_S) {
        return Err(failed_verification_with_distance(
            tolerance_km,
            distance,
            format!(
                "Arrival delta-V vector mismatch: stored=[{:.12}, {:.12}, {:.12}], expected=[{:.12}, {:.12}, {:.12}] km/s",
                result.arrival_dv[0],
                result.arrival_dv[1],
                result.arrival_dv[2],
                expected_arrival_dv[0],
                expected_arrival_dv[1],
                expected_arrival_dv[2]
            ),
        ));
    }

    if let Some(error) = stored_state_error(
        "payload intercept",
        &result.payload_intercept_state,
        &sat_at_intercept,
        tolerance_km,
    ) {
        return Err(failed_verification_with_distance(
            tolerance_km,
            distance,
            error,
        ));
    }
    if let Some(error) = stored_state_error(
        "target intercept",
        &result.target_intercept_state,
        &tgt_at_intercept,
        tolerance_km,
    ) {
        return Err(failed_verification_with_distance(
            tolerance_km,
            distance,
            error,
        ));
    }
    if let Some(error) = stored_state_error(
        "deployer intercept",
        &result.deployer_intercept_state,
        &deployer_at_intercept,
        tolerance_km,
    ) {
        return Err(failed_verification_with_distance(
            tolerance_km,
            distance,
            error,
        ));
    }
    if let Some(error) = stored_state_error(
        "release",
        &result.release_state,
        &dep_at_transfer,
        tolerance_km,
    ) {
        return Err(failed_verification_with_distance(
            tolerance_km,
            distance,
            error,
        ));
    }

    // Compute margin and verification status
    let margin = tolerance_km - distance;
    let verified = distance <= tolerance_km;

    Ok(VerificationResult {
        verified,
        actual_distance_km: distance,
        tolerance_km,
        margin_km: margin,
        propagation_error: None,
    })
}

#[inline]
fn distance3(left: &[f64; 6], right: &[f64; 6]) -> f64 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    let dz = left[2] - right[2];
    euclidean_norm3(dx, dy, dz)
}

#[inline]
fn euclidean_norm3(x: f64, y: f64, z: f64) -> f64 {
    (x * x + y * y + z * z).sqrt()
}

#[inline]
fn apply_delta_v([x, y, z, vx, vy, vz]: [f64; 6], [dvx, dvy, dvz]: [f64; 3]) -> [f64; 6] {
    [x, y, z, vx + dvx, vy + dvy, vz + dvz]
}

#[inline]
fn active_positive_limit(limit: f64) -> bool {
    limit.is_finite() && limit > 0.0
}

#[inline]
fn active_orbit_bounds(ctx: &PlanContext) -> bool {
    ctx.min_perigee.is_finite()
        && ctx.max_apogee.is_finite()
        && ctx.min_perigee > 0.0
        && ctx.max_apogee > ctx.min_perigee
}

#[inline]
fn limit_exceeded(value: f64, limit: f64, tolerance: f64) -> bool {
    active_positive_limit(limit) && value.is_finite() && value > limit + tolerance
}

#[inline]
fn context_deployer_period_s(ctx: &PlanContext) -> Option<f64> {
    if active_positive_limit(ctx.dep_period) {
        return Some(ctx.dep_period);
    }
    let orbit = EciBasicOrbit::from_eci(&ctx.dep_eci)?;
    if !active_positive_limit(orbit.sma) {
        return None;
    }
    let period_s =
        std::f64::consts::TAU * ((orbit.sma * orbit.sma * orbit.sma) / satpy_core::MU).sqrt();
    Some(period_s)
}

#[inline]
fn norm_matches(stored: f64, computed: f64) -> bool {
    stored.is_finite()
        && computed.is_finite()
        && (stored - computed).abs() <= DV_TOL_KM_S.max(computed.abs() * 1e-9)
}

#[inline]
fn radius_tolerance(min_perigee: f64, max_apogee: f64) -> f64 {
    const BASE_TOL: f64 = 0.1;
    let range = max_apogee - min_perigee;
    if range > 0.0 && range.is_finite() {
        BASE_TOL + range * 1e-5
    } else {
        BASE_TOL
    }
}

fn orbit_bound_error(label: &str, state: &[f64; 6], ctx: &PlanContext) -> Option<String> {
    let Some(orbit) = EciBasicOrbit::from_eci(state) else {
        return Some(format!("{label} orbit is not bound/elliptic"));
    };

    if !orbit.perigee.is_finite() || !orbit.apogee.is_finite() {
        return Some(format!("{label} orbit has non-finite perigee/apogee"));
    }
    if orbit.perigee < satpy_core::RE {
        return Some(format!(
            "{label} orbit re-enters Earth: perigee={:.6} km < {:.6} km",
            orbit.perigee,
            satpy_core::RE
        ));
    }
    if active_orbit_bounds(ctx) {
        let tol = radius_tolerance(ctx.min_perigee, ctx.max_apogee);
        if orbit.perigee < ctx.min_perigee - tol {
            return Some(format!(
                "{label} orbit perigee below minimum: {:.6} < {:.6} km",
                orbit.perigee, ctx.min_perigee
            ));
        }
        if orbit.apogee > ctx.max_apogee + tol {
            return Some(format!(
                "{label} orbit apogee above maximum: {:.6} > {:.6} km",
                orbit.apogee, ctx.max_apogee
            ));
        }
    }

    None
}

fn stored_state_error(
    label: &str,
    stored: &[f64; 6],
    computed: &[f64; 6],
    tolerance_km: f64,
) -> Option<String> {
    if !all_finite(stored) {
        return Some(format!("{label} stored state contains NaN or Inf"));
    }

    let pos_diff = distance3(stored, computed);
    let dvx = stored[3] - computed[3];
    let dvy = stored[4] - computed[4];
    let dvz = stored[5] - computed[5];
    let vel_diff = euclidean_norm3(dvx, dvy, dvz);
    let pos_tol = tolerance_km.max(STATE_POS_TOL_KM);

    if !pos_diff.is_finite() || !vel_diff.is_finite() {
        return Some(format!("{label} stored state diff is NaN or Inf"));
    }
    if pos_diff > pos_tol || vel_diff > STATE_VEL_TOL_KM_S {
        return Some(format!(
            "{label} stored state inconsistent: position diff={pos_diff:.9} km, velocity diff={vel_diff:.12} km/s"
        ));
    }

    None
}

fn timeline_epoch_error(
    result: &PlanResult,
    ctx: &PlanContext,
    time2phase: f64,
    waittime: f64,
    tof: f64,
) -> Option<String> {
    if !ctx.epoch_jd.is_finite()
        || !result.waittime_jd_start.is_finite()
        || !result.tof_jd_start.is_finite()
        || !result.intercept_jd.is_finite()
    {
        return Some("Timeline timestamps contain NaN or Inf".to_string());
    }

    let tol_days = TIMELINE_TOL_S / satpy_core::SEC_PER_DAY;
    if result.waittime_jd_start + tol_days < ctx.epoch_jd {
        return Some(format!(
            "Phase timeline overlap: phase burn starts before epoch ({:.12} < {:.12})",
            result.waittime_jd_start, ctx.epoch_jd
        ));
    }
    if result.tof_jd_start + tol_days < result.waittime_jd_start {
        return Some(format!(
            "Phase timeline overlap: transfer burn starts before phase/wait phase completes ({:.12} < {:.12})",
            result.tof_jd_start, result.waittime_jd_start
        ));
    }
    if result.intercept_jd + tol_days < result.tof_jd_start {
        return Some(format!(
            "Phase timeline overlap: intercept occurs before transfer phase starts ({:.12} < {:.12})",
            result.intercept_jd, result.tof_jd_start
        ));
    }

    let expected_phase_jd = ctx.epoch_jd + time2phase / satpy_core::SEC_PER_DAY;
    let expected_transfer_jd = expected_phase_jd + waittime / satpy_core::SEC_PER_DAY;
    let expected_intercept_jd = expected_transfer_jd + tof / satpy_core::SEC_PER_DAY;
    if !jd_matches(result.waittime_jd_start, expected_phase_jd) {
        return Some(format!(
            "Phase timeline mismatch: phase burn JD stored={:.12}, expected={:.12}",
            result.waittime_jd_start, expected_phase_jd
        ));
    }
    if !jd_matches(result.tof_jd_start, expected_transfer_jd) {
        return Some(format!(
            "Phase timeline mismatch: transfer burn JD stored={:.12}, expected={:.12}",
            result.tof_jd_start, expected_transfer_jd
        ));
    }
    if !jd_matches(result.intercept_jd, expected_intercept_jd) {
        return Some(format!(
            "Phase timeline mismatch: intercept JD stored={:.12}, expected={:.12}",
            result.intercept_jd, expected_intercept_jd
        ));
    }

    None
}

#[inline]
fn jd_matches(stored: f64, expected: f64) -> bool {
    stored.is_finite()
        && expected.is_finite()
        && ((stored - expected) * satpy_core::SEC_PER_DAY).abs() <= TIMELINE_TOL_S
}

#[inline]
fn scalar_matches(stored: f64, computed: f64, tolerance: f64) -> bool {
    stored.is_finite() && computed.is_finite() && (stored - computed).abs() <= tolerance
}

#[inline]
fn vector3_matches(stored: &[f64; 3], expected: &[f64; 3], tolerance: f64) -> bool {
    stored
        .iter()
        .zip(expected.iter())
        .all(|(stored, expected)| {
            stored.is_finite() && expected.is_finite() && (stored - expected).abs() <= tolerance
        })
}

fn failed_verification(
    tolerance_km: f64,
    error: impl Into<std::borrow::Cow<'static, str>>,
) -> VerificationResult {
    VerificationResult {
        verified: false,
        actual_distance_km: f64::NAN,
        tolerance_km,
        margin_km: f64::NEG_INFINITY,
        propagation_error: Some(error.into()),
    }
}

fn failed_verification_with_distance(
    tolerance_km: f64,
    actual_distance_km: f64,
    error: impl Into<std::borrow::Cow<'static, str>>,
) -> VerificationResult {
    VerificationResult {
        verified: false,
        actual_distance_km,
        tolerance_km,
        margin_km: tolerance_km - actual_distance_km,
        propagation_error: Some(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use satpy_core::{MU, RE};

    #[test]
    fn verifier_replay_stamps_each_segment_from_its_actual_source_epoch() {
        let base_jd = 2_460_000.5;
        let phase_s = 120.0;
        let wait_s = 240.0;
        let transfer_s = 360.0;

        let epochs = replay_segment_epochs(base_jd, phase_s, wait_s, transfer_s);

        assert_eq!(epochs.phase.to_bits(), base_jd.to_bits());
        assert_eq!(
            epochs.wait.to_bits(),
            (base_jd + phase_s / satpy_core::SEC_PER_DAY).to_bits()
        );
        assert_eq!(
            epochs.transfer.to_bits(),
            (base_jd + (phase_s + wait_s) / satpy_core::SEC_PER_DAY).to_bits()
        );
        assert_eq!(
            epochs.intercept.to_bits(),
            (base_jd + (phase_s + wait_s + transfer_s) / satpy_core::SEC_PER_DAY).to_bits()
        );
    }

    #[test]
    fn test_verify_invalid_result() {
        let result = PlanResult::invalid();
        let ctx = PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        let verification = verify_transfer_result(&result, &ctx, 0.010);

        assert!(!verification.verified);
        assert_eq!(
            verification.propagation_error.as_deref(),
            Some("Result marked as invalid")
        );
    }

    #[test]
    fn test_verify_nan_timing() {
        let mut result = PlanResult::invalid();
        result.valid = true;
        result.time2phase = f64::NAN;
        result.waittime = 100.0;
        result.tof = 200.0;

        let ctx = PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        let verification = verify_transfer_result(&result, &ctx, 0.010);

        assert!(!verification.verified);
        assert!(verification
            .propagation_error
            .as_deref()
            .is_some_and(|error| error.contains("Invalid timing parameters")));
    }

    #[test]
    fn replay_controls_recomposes_requested_intercept_from_e0_without_search() {
        let radius = RE + 500.0;
        let velocity = (MU / radius).sqrt();
        let state = [radius, 0.0, 0.0, 0.0, velocity, 0.0];
        let mut ctx = PlanContext {
            dep_eci: state,
            tgt_eci: state,
            epoch_jd: 2_460_000.5,
            max_time_s: 2_000.0,
            max_phase_dv: 0.1,
            max_transfer_dv: 0.1,
            max_revs: 0,
            revolution_cap: 10.0,
            min_perigee: RE,
            max_apogee: 100_000.0,
            distance_tol: 0.025,
            target_propagation_authority: crate::types::TargetPropagationAuthority::MfJ2,
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };
        satpy_core::eci2equinoc_impl(&ctx.dep_eci, 6, 0.0, 0.0, &mut ctx.dep_equ);
        satpy_core::eci2equinoc_impl(&ctx.tgt_eci, 6, 0.0, 0.0, &mut ctx.tgt_equ);
        ctx.cache_deployer_orbit();
        ctx.cache_target_orbit();
        let mut stored = PlanResult::invalid();
        stored.valid = true;
        stored.time2phase = 120.0;
        stored.waittime = 180.0;
        stored.tof = 900.0;
        stored.phase_dv = [0.0; 3];
        stored.transfer_dv = [0.0; 3];
        stored.phase_dv_norm = 0.0;
        stored.transfer_dv_norm = 0.0;
        stored.best_M = 0;

        let replayed = replay_transfer_controls(&stored, &ctx);
        assert!(
            replayed.is_ok(),
            "replay must reproduce valid stored controls"
        );
        let Ok(replayed) = replayed else {
            return;
        };

        assert_eq!(
            replayed.phase_dv.map(f64::to_bits),
            stored.phase_dv.map(f64::to_bits)
        );
        assert_eq!(
            replayed.transfer_dv.map(f64::to_bits),
            stored.transfer_dv.map(f64::to_bits)
        );
        assert!(
            (replayed.waittime_jd_start
                - (ctx.epoch_jd + stored.time2phase / satpy_core::SEC_PER_DAY))
                .abs()
                < 1e-14
        );
        assert!(
            (replayed.tof_jd_start
                - (ctx.epoch_jd + (stored.time2phase + stored.waittime) / satpy_core::SEC_PER_DAY))
                .abs()
                < 1e-14
        );
        assert!(
            (replayed.intercept_jd
                - (ctx.epoch_jd
                    + (stored.time2phase + stored.waittime + stored.tof)
                        / satpy_core::SEC_PER_DAY))
                .abs()
                < 1e-14
        );
        assert!(replayed.distance < 1e-8);
    }

    #[test]
    fn strict_hf_verification_requires_finite_post_hf_residual() {
        let mut result = PlanResult::invalid();
        result.valid = true;
        result.post_hf_endpoint_residual_m = f64::NAN;
        let ctx = PlanContext {
            execution_policy: crate::types::ExecutionPolicy {
                use_high_fidelity: true,
                require_high_fidelity: true,
                allow_parallel: false,
                allow_oxymoo_batch_parallel: false,
                allow_branch_expansion_parallel: false,
                allow_polish_parallel: false,
                allow_anchor_parallel: false,
                allow_deterministic_grid_parallel: false,
            },
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };

        let verification = verify_transfer_result(&result, &ctx, 0.010);

        assert!(!verification.verified);
        assert!(verification
            .propagation_error
            .as_deref()
            .unwrap_or_default()
            .contains("post-HF endpoint residual"));
    }

    #[test]
    fn strict_hf_verification_does_not_reject_pre_hf_j2_residual() {
        assert!(!pre_hf_j2_residual_blocks_verification(
            true, 1, 1.0e9, 10.0,
        ));
        assert!(pre_hf_j2_residual_blocks_verification(
            false, 1, 1.0e9, 10.0,
        ));
    }

    #[test]
    fn replay_leg_rejects_negative_and_nonfinite_dt_fail_closed() {
        let ctx = PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        let state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let mut equ = [0.0; 6];
        assert!(eci_to_equinoctial(&state, &mut equ));
        let body = crate::types::BodyForceConfig::high_fidelity(
            crate::types::BodyRole::DiagnosticTarget,
            0.01,
            2.2,
            1.2,
        );
        for dt in [-1.0, f64::NAN, f64::NEG_INFINITY] {
            for cap in [None, Some(5400.0)] {
                assert_eq!(
                    propagate_replay_leg(&state, &equ, dt, 2_460_000.5, body, &ctx, cap),
                    Err(TransferPropagationFailure::InvalidInput),
                    "forward replay leg must fail closed on dt={dt}, cap={cap:?}"
                );
            }
        }
    }

    #[test]
    fn backward_replay_leg_rejects_forward_and_invalid_controls() {
        let ctx = PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        let state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let mut equ = [0.0; 6];
        assert!(eci_to_equinoctial(&state, &mut equ));
        let body = crate::types::BodyForceConfig::high_fidelity(
            crate::types::BodyRole::DiagnosticTarget,
            0.01,
            2.2,
            1.2,
        );
        // Forward, zero, and non-finite arcs are not this leg's contract.
        for dt in [1.0, 0.0, f64::NAN] {
            assert_eq!(
                propagate_replay_leg_backward(&state, &equ, dt, 2_460_000.5, body, &ctx, 5400.0),
                Err(TransferPropagationFailure::InvalidInput),
            );
        }
        // An unusable segment cap fails closed before any propagation.
        for cap in [0.0, -1.0, f64::NAN] {
            assert_eq!(
                propagate_replay_leg_backward(&state, &equ, -60.0, 2_460_000.5, body, &ctx, cap),
                Err(TransferPropagationFailure::InvalidInput),
            );
        }
    }

    #[test]
    fn witness_leg_negative_dt_passes_the_entry_gate() {
        // A negative dt used to be rejected at the entry guard as
        // InvalidInput. It now walks the mirrored backward branch, so with a
        // context that carries no strict-HF authority the failure is the
        // propagation gate's Authority error -- proof the entry gate no longer
        // refuses the signed arc. Non-finite dt and an unusable cap stay
        // InvalidInput.
        let ctx = PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        let state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let mut equ = [0.0; 6];
        assert!(eci_to_equinoctial(&state, &mut equ));
        let body = crate::types::BodyForceConfig::high_fidelity(
            crate::types::BodyRole::DiagnosticTarget,
            0.01,
            2.2,
            1.2,
        );
        assert_eq!(
            propagate_independent_witness_leg(
                &state,
                &equ,
                -60.0,
                2_460_000.5,
                body,
                &ctx,
                5400.0,
                60.0,
                1.0e-10,
            ),
            Err(TransferPropagationFailure::Authority),
        );
        assert_eq!(
            propagate_independent_witness_leg(
                &state,
                &equ,
                f64::NAN,
                2_460_000.5,
                body,
                &ctx,
                5400.0,
                60.0,
                1.0e-10,
            ),
            Err(TransferPropagationFailure::InvalidInput),
        );
        assert_eq!(
            propagate_independent_witness_leg(
                &state,
                &equ,
                -60.0,
                2_460_000.5,
                body,
                &ctx,
                0.0,
                60.0,
                1.0e-10,
            ),
            Err(TransferPropagationFailure::InvalidInput),
        );
    }

    #[test]
    fn test_all_finite() {
        assert!(all_finite(&[1.0, 2.0, 3.0]));
        assert!(!all_finite(&[1.0, f64::NAN, 3.0]));
        assert!(!all_finite(&[1.0, f64::INFINITY, 3.0]));
        assert!(!all_finite(&[f64::NEG_INFINITY, 2.0, 3.0]));
    }

    #[test]
    fn apply_delta_v_preserves_position_and_adds_velocity() {
        let state = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let updated = apply_delta_v(state, [0.5, -1.0, 2.0]);

        assert_eq!(
            updated.map(f64::to_bits),
            [1.0, 2.0, 3.0, 4.5, 4.0, 8.0].map(f64::to_bits)
        );
    }
}
