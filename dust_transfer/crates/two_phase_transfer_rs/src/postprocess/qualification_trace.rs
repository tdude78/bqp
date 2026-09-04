//! Bounded, caller-owned scalar-leg evidence for solver qualification.
//!
//! This module is deliberately feature-only. Production has no collector,
//! callback, environment switch, or alternate propagation path. One outer
//! qualification replay owns one trace and passes it directly through the
//! same release and UKF numerical cores.

use core::fmt;
use core::mem::{size_of, size_of_val, MaybeUninit};
use core::slice;

use crate::evaluate::TransferPropagationFailure;
use crate::intercept::{compute_miss_vector_hf_with_endpoint_observed, HfInterceptEvaluation};
use crate::types::{BodyForceConfig, BodyRole, PlanContext, StampedEciState};

use super::distribution::propagate_stamped_checked_observed;
/// Actual scalar path, never a conceptual/estimated leg.
pub use super::observer::LegPath as QualificationLegPath;
use super::ukf::{
    propagate_sigma_states_with_context, propagate_sigma_states_with_fresh_observed_context,
    UkfPropagationFailure,
};
use lightyear_odeint_rs::integrator::FinalPropagationFailure;
use lightyear_odeint_rs::{
    ObservedFinalLeg, ObservedFinalMetricError, ObservedFinalMetrics, ObservedSolverTerminalStatus,
};

/// Fixed per-replay evidence ceiling. The Part A controls cap a release solve
/// well below this; exceeding it is evidence loss, never silent truncation.
pub const MAX_QUALIFICATION_LEG_RECORDS: usize = 1_024;

// An address-space overflow means no caller-owned buffer can satisfy the
// requested fixed storage shape. Keep the public byte APIs total for their
// const callers, then make `try_new` fail closed during its exact preflight.
const STORAGE_SIZE_OVERFLOW: usize = usize::MAX;

#[inline]
const fn checked_storage_product(element_size: usize, capacity: usize) -> usize {
    let Some(bytes) = element_size.checked_mul(capacity) else {
        return STORAGE_SIZE_OVERFLOW;
    };
    bytes
}

/// Closed solver tuple stamped into one qualification replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QualificationArmIdentity {
    stamp: [u8; 32],
}

impl QualificationArmIdentity {
    /// Validate sealed nonzero arm provenance before any evidence storage binds.
    ///
    /// # Errors
    ///
    /// Returns [`QualificationTraceError::InvalidIdentity`] when `stamp` is all zeroes.
    pub fn try_new(stamp: [u8; 32]) -> Result<Self, QualificationTraceError> {
        if stamp.iter().all(|&byte| byte == 0) {
            return Err(QualificationTraceError::InvalidIdentity);
        }
        Ok(Self { stamp })
    }
}

/// Stable owner identity for one ordered qualification replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QualificationTraceIdentity {
    pub event_ordinal: u32,
    pub family_ordinal: u8,
    pub candidate_ordinal: u32,
    pub fraction_ordinal: u16,
    pub arm: QualificationArmIdentity,
}

/// Immutable scalar request metadata retained with every outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QualificationLegInput {
    pub path: QualificationLegPath,
    pub body_role: BodyRole,
    pub source_jd_bits: u64,
    pub t0_s_bits: u64,
    pub t_final_s_bits: u64,
    pub initial_eci_bits: [u64; 6],
}

impl QualificationLegInput {
    #[must_use]
    pub fn new(
        path: QualificationLegPath,
        body_role: BodyRole,
        source_jd: f64,
        t0_s: f64,
        t_final_s: f64,
        initial_eci: [f64; 6],
    ) -> Self {
        Self {
            path,
            body_role,
            source_jd_bits: source_jd.to_bits(),
            t0_s_bits: t0_s.to_bits(),
            t_final_s_bits: t_final_s.to_bits(),
            initial_eci_bits: initial_eci.map(f64::to_bits),
        }
    }
}

/// Typed terminal result of an actual scalar leg.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QualificationLegOutcome {
    Endpoint([u64; 6]),
    Failure(QualificationLegFailureCode),
}

/// Heap-free code retained when an actual scalar leg fails.
///
/// The trace intentionally discards nested propagation payloads before it
/// reaches caller-owned storage. Some source failures can carry owned text;
/// qualification evidence keeps only this stable code and never owns that text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum QualificationLegFailureCode {
    Ground,
    LeftEarth,
    Eccentricity,
    TransferArithmeticOverflow,
    TransferCensus,
    TransferInvalidInput,
    TransferAuthority,
    TransferMissingHighFidelityAssets,
    TransferEphemeris,
    FinalNanState,
    FinalEventInvalid,
    FinalGravity,
    FinalCensus,
    FinalEclipseGeometry,
    FinalEclipseUninitializedSide,
    FinalEclipseNonProgress,
    FinalEclipseChatter,
    FinalEclipseBracket,
    FinalEclipseEventOverlap,
    FinalEclipseSplitLimit,
    FinalEclipseEnvelope,
    FinalIntegrationFailure,
    TransferNonFiniteOutput,
    // Appended, like `TransferNonFiniteOutput` before it, so no existing
    // discriminant moves. `EclipseError::Authority` was split out of
    // `Geometry` when an eclipse authority refusal stopped being reported as
    // an infeasible candidate; this arm is the qualification trace catching up.
    FinalEclipseAuthority,
    // Appended for the same reason, so no existing discriminant moves. ESDIRK
    // used to be substituted with Tsit5 on the routes that cannot run it, so a
    // leg that asked for one method and got another was recorded as a SUCCESS.
    // This arm is the qualification trace catching up to the refusal.
    FinalMethodUnsupported,
}

/// One ordered scalar leg record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QualificationLegRecord {
    pub sequence: u32,
    pub input: QualificationLegInput,
    pub outcome: QualificationLegOutcome,
    pub metrics: Result<ObservedFinalMetrics, ObservedFinalMetricError>,
    pub terminal_status: ObservedSolverTerminalStatus,
}

/// Typed evidence failure. It remains separate from physical propagation
/// outcomes so a failed observer cannot masquerade as a science result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QualificationTraceError {
    InvalidIdentity,
    StorageTooSmall,
    StorageTooLarge,
    RecordLimit,
    SequenceOverflow,
    Empty,
    IncompleteMetrics,
    MissingSolverInvocation,
}

impl fmt::Display for QualificationTraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for QualificationTraceError {}

/// Fixed-capacity, single-replay trace over caller-owned replay storage.
///
/// The constructor accepts one exact record capacity. It owns no allocation,
/// callback, or global state; all initialized records stay in the supplied
/// buffer until its caller drops that buffer.
pub struct QualificationLegTrace<'storage> {
    identity: QualificationTraceIdentity,
    records: &'storage mut [MaybeUninit<QualificationLegRecord>],
    len: usize,
    failure: Option<QualificationTraceError>,
}

impl<'storage> QualificationLegTrace<'storage> {
    /// Exact record slots a qualification replay must preflight.
    #[must_use]
    pub const fn required_record_capacity() -> usize {
        MAX_QUALIFICATION_LEG_RECORDS
    }

    /// Exact caller bytes for retained qualification records.
    #[must_use]
    pub const fn required_record_storage_bytes() -> usize {
        checked_storage_product(
            size_of::<QualificationLegRecord>(),
            Self::required_record_capacity(),
        )
    }

    /// Bind one trace to caller-owned fixed replay storage before replay starts.
    ///
    /// # Errors
    ///
    /// Returns [`QualificationTraceError::StorageTooSmall`] or
    /// [`QualificationTraceError::StorageTooLarge`] when supplied storage
    /// shape differs from the fixed replay requirement.
    pub const fn try_new(
        identity: QualificationTraceIdentity,
        records: &'storage mut [MaybeUninit<QualificationLegRecord>],
    ) -> Result<Self, QualificationTraceError> {
        let record_bytes = size_of_val(records);
        if records.len() < Self::required_record_capacity()
            || record_bytes < Self::required_record_storage_bytes()
        {
            return Err(QualificationTraceError::StorageTooSmall);
        }
        if records.len() > Self::required_record_capacity()
            || record_bytes > Self::required_record_storage_bytes()
        {
            return Err(QualificationTraceError::StorageTooLarge);
        }
        Ok(Self {
            identity,
            records,
            len: 0,
            failure: None,
        })
    }

    #[must_use]
    pub const fn identity(&self) -> QualificationTraceIdentity {
        self.identity
    }

    #[must_use]
    pub const fn records(&self) -> &[QualificationLegRecord] {
        // SAFETY: `push` initializes exactly `0..self.len` in order, `len`
        // starts at zero, and it only grows after a successful write. Records
        // are `Copy`, so this trace has no omitted destructor work.
        unsafe { slice::from_raw_parts(self.records.as_ptr().cast(), self.len) }
    }

    /// Reject a UKF observed batch before it allocates its numerical vectors.
    pub(crate) const fn preflight_record_capacity(
        &mut self,
        required: usize,
    ) -> Result<(), QualificationTraceError> {
        let Some(remaining_records) = self.records.len().checked_sub(self.len) else {
            self.mark_incomplete(QualificationTraceError::RecordLimit);
            return Err(QualificationTraceError::RecordLimit);
        };
        if required > remaining_records {
            self.mark_incomplete(QualificationTraceError::RecordLimit);
            return Err(QualificationTraceError::RecordLimit);
        }
        Ok(())
    }

    /// Record an actual direct-release scalar result. Observation failure is
    /// retained and rejected only by `validate_complete`, preserving the
    /// original science outcome and failure ordering.
    pub fn record_transfer(
        &mut self,
        input: QualificationLegInput,
        outcome: Result<[f64; 6], TransferPropagationFailure>,
        metrics: Result<ObservedFinalMetrics, ObservedFinalMetricError>,
        terminal_status: ObservedSolverTerminalStatus,
    ) {
        let outcome = match outcome {
            Ok(endpoint) => QualificationLegOutcome::Endpoint(endpoint.map(f64::to_bits)),
            Err(failure) => QualificationLegOutcome::Failure(transfer_failure_code(&failure)),
        };
        self.push(input, outcome, metrics, terminal_status);
    }

    /// Record one observed strict-HF transfer, or fail the trace closed when a
    /// nonzero numerical leg returns without its required observation.
    pub(crate) fn record_observed_transfer(
        &mut self,
        input: QualificationLegInput,
        outcome: Result<[f64; 6], TransferPropagationFailure>,
        observation: Option<ObservedFinalLeg>,
    ) {
        match observation {
            Some(observation) => self.record_transfer(
                input,
                outcome,
                observation.metrics,
                observation.terminal_status,
            ),
            None if input.t0_s_bits != input.t_final_s_bits => {
                self.mark_incomplete(QualificationTraceError::IncompleteMetrics);
            }
            None => {}
        }
    }

    /// Mark a caller-detected evidence gap without changing its science path.
    pub const fn mark_incomplete(&mut self, error: QualificationTraceError) {
        if self.failure.is_none() {
            self.failure = Some(error);
        }
    }

    fn push(
        &mut self,
        input: QualificationLegInput,
        outcome: QualificationLegOutcome,
        metrics: Result<ObservedFinalMetrics, ObservedFinalMetricError>,
        terminal_status: ObservedSolverTerminalStatus,
    ) {
        if self.failure.is_some() {
            return;
        }
        let Some(next_len) = self.len.checked_add(1) else {
            self.failure = Some(QualificationTraceError::SequenceOverflow);
            return;
        };
        let Some(slot) = self.records.get_mut(self.len) else {
            self.failure = Some(QualificationTraceError::RecordLimit);
            return;
        };
        let Ok(sequence) = u32::try_from(self.len) else {
            self.failure = Some(QualificationTraceError::SequenceOverflow);
            return;
        };
        slot.write(QualificationLegRecord {
            sequence,
            input,
            outcome,
            metrics,
            terminal_status,
        });
        self.len = next_len;
    }

    /// Reject incomplete or empty evidence after the unchanged numerical path
    /// returns. A zero-duration direct leg may legitimately have zero solver
    /// invocations, so only nonzero spans require an invocation.
    ///
    /// # Errors
    ///
    /// Returns the first observer failure, or rejects empty evidence,
    /// incomplete metrics, and nonzero legs without solver invocations.
    pub fn validate_complete(&self) -> Result<(), QualificationTraceError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        if self.len == 0 {
            return Err(QualificationTraceError::Empty);
        }
        for record in self.records() {
            if matches!(record.outcome, QualificationLegOutcome::Failure(_)) {
                continue;
            }
            let metrics = record
                .metrics
                .as_ref()
                .map_err(|_| QualificationTraceError::IncompleteMetrics)?;
            let nonzero_span = record.input.t0_s_bits != record.input.t_final_s_bits;
            if nonzero_span && metrics.solver_invocations == 0 {
                return Err(QualificationTraceError::MissingSolverInvocation);
            }
        }
        Ok(())
    }
}

#[inline]
const fn transfer_failure_code(
    failure: &TransferPropagationFailure,
) -> QualificationLegFailureCode {
    match failure {
        TransferPropagationFailure::ArithmeticOverflow => {
            QualificationLegFailureCode::TransferArithmeticOverflow
        }
        TransferPropagationFailure::Census(_) => QualificationLegFailureCode::TransferCensus,
        TransferPropagationFailure::InvalidInput => {
            QualificationLegFailureCode::TransferInvalidInput
        }
        TransferPropagationFailure::Authority => QualificationLegFailureCode::TransferAuthority,
        TransferPropagationFailure::MissingHighFidelityAssets => {
            QualificationLegFailureCode::TransferMissingHighFidelityAssets
        }
        TransferPropagationFailure::Ephemeris(_) => QualificationLegFailureCode::TransferEphemeris,
        TransferPropagationFailure::Final(failure) => final_failure_code(*failure),
        TransferPropagationFailure::NonFiniteOutput => {
            QualificationLegFailureCode::TransferNonFiniteOutput
        }
    }
}

#[inline]
const fn final_failure_code(failure: FinalPropagationFailure) -> QualificationLegFailureCode {
    match failure {
        FinalPropagationFailure::Ground => QualificationLegFailureCode::Ground,
        FinalPropagationFailure::LeftEarth => QualificationLegFailureCode::LeftEarth,
        FinalPropagationFailure::Eccentricity => QualificationLegFailureCode::Eccentricity,
        FinalPropagationFailure::NanState => QualificationLegFailureCode::FinalNanState,
        FinalPropagationFailure::EventInvalid => QualificationLegFailureCode::FinalEventInvalid,
        FinalPropagationFailure::Gravity(_) => QualificationLegFailureCode::FinalGravity,
        FinalPropagationFailure::Census(_) => QualificationLegFailureCode::FinalCensus,
        FinalPropagationFailure::MethodUnsupported => {
            QualificationLegFailureCode::FinalMethodUnsupported
        }
        FinalPropagationFailure::Eclipse(error) => match error {
            lightyear_odeint_rs::EclipseError::Gravity(_) => {
                QualificationLegFailureCode::FinalGravity
            }
            lightyear_odeint_rs::EclipseError::Geometry => {
                QualificationLegFailureCode::FinalEclipseGeometry
            }
            lightyear_odeint_rs::EclipseError::UninitializedSide => {
                QualificationLegFailureCode::FinalEclipseUninitializedSide
            }
            lightyear_odeint_rs::EclipseError::NonProgress => {
                QualificationLegFailureCode::FinalEclipseNonProgress
            }
            lightyear_odeint_rs::EclipseError::Chatter => {
                QualificationLegFailureCode::FinalEclipseChatter
            }
            lightyear_odeint_rs::EclipseError::Bracket => {
                QualificationLegFailureCode::FinalEclipseBracket
            }
            lightyear_odeint_rs::EclipseError::EventOverlap => {
                QualificationLegFailureCode::FinalEclipseEventOverlap
            }
            lightyear_odeint_rs::EclipseError::SplitLimit => {
                QualificationLegFailureCode::FinalEclipseSplitLimit
            }
            lightyear_odeint_rs::EclipseError::Envelope => {
                QualificationLegFailureCode::FinalEclipseEnvelope
            }
            lightyear_odeint_rs::EclipseError::Authority(_) => {
                QualificationLegFailureCode::FinalEclipseAuthority
            }
        },
        FinalPropagationFailure::IntegrationFailure => {
            QualificationLegFailureCode::FinalIntegrationFailure
        }
    }
}

/// The observing side of the postprocess leg seam.
///
/// Each override runs the fresh-context observed variant of the canonical call
/// the default body would have made. The numbers are the same; the difference
/// is the bounded evidence retained here.
impl super::observer::PostprocessLegObserver for QualificationLegTrace<'_> {
    fn preflight_leg_capacity(&mut self, legs: usize) -> Result<(), UkfPropagationFailure> {
        self.preflight_record_capacity(legs)
            .map_err(UkfPropagationFailure::Qualification)
    }

    fn propagate_stamped(
        &mut self,
        state: &StampedEciState,
        dt_s: f64,
        body_force: BodyForceConfig,
        ctx: &PlanContext,
        path: QualificationLegPath,
    ) -> Result<StampedEciState, TransferPropagationFailure> {
        propagate_stamped_checked_observed(state, dt_s, body_force, ctx, path, self)
    }

    fn miss_vector_hf_with_endpoint(
        &mut self,
        dv: [f64; 3],
        v0: [f64; 3],
        r0: [f64; 3],
        target_pos: [f64; 3],
        tof_s: f64,
        source_jd: f64,
        body_force: BodyForceConfig,
        ctx: &PlanContext,
    ) -> Result<HfInterceptEvaluation, TransferPropagationFailure> {
        compute_miss_vector_hf_with_endpoint_observed(
            dv, v0, r0, target_pos, tof_s, source_jd, body_force, ctx, self,
        )
    }

    fn propagate_ukf_sigma_states(
        &mut self,
        sigma_eci_states: &[f64],
        sigma_propagated: &mut [f64],
        total_sigma: usize,
        tof_s: f64,
        ctx: &PlanContext,
    ) -> Result<(), UkfPropagationFailure> {
        let _ = total_sigma;
        let (Some(force_config), Some(_)) = (ctx.force_config.as_ref(), ctx.packed_coeffs.as_ref())
        else {
            if ctx.execution_policy.require_high_fidelity {
                return Err(UkfPropagationFailure::Propagation(
                    TransferPropagationFailure::MissingHighFidelityAssets,
                ));
            }
            return propagate_sigma_states_with_context(
                sigma_eci_states,
                sigma_propagated,
                tof_s,
                ctx,
            );
        };
        let body_force = BodyForceConfig::high_fidelity(
            BodyRole::Dust,
            force_config.am_ratio,
            force_config.cd,
            force_config.cr,
        );
        propagate_sigma_states_with_fresh_observed_context(
            self,
            sigma_eci_states,
            sigma_propagated,
            tof_s,
            ctx,
            body_force,
        )
    }
}

#[cfg(test)]
mod tests {
    use core::mem::{size_of, MaybeUninit};

    use super::*;

    fn fixed_identity() -> QualificationTraceIdentity {
        QualificationTraceIdentity {
            event_ordinal: 17,
            family_ordinal: 2,
            candidate_ordinal: 41,
            fraction_ordinal: 3,
            arm: QualificationArmIdentity { stamp: [0x17; 32] },
        }
    }

    fn complete_metrics() -> ObservedFinalMetrics {
        ObservedFinalMetrics {
            solver_invocations: 1,
            steps: 7,
            evals: 43,
            ..ObservedFinalMetrics::default()
        }
    }

    fn uninit_records<const N: usize>() -> [MaybeUninit<QualificationLegRecord>; N] {
        std::array::from_fn(|_| MaybeUninit::uninit())
    }

    #[test]
    fn arm_identity_accepts_only_nonzero_sealed_stamp() {
        let stamp = [0xa5; 32];
        assert_eq!(
            QualificationArmIdentity::try_new(stamp),
            Ok(QualificationArmIdentity { stamp })
        );
        assert_eq!(
            QualificationArmIdentity::try_new([0; 32]),
            Err(QualificationTraceError::InvalidIdentity)
        );
    }

    #[test]
    fn final_failure_subtypes_remain_typed_in_the_trace() {
        for (failure, expected) in [
            (
                FinalPropagationFailure::NanState,
                QualificationLegFailureCode::FinalNanState,
            ),
            (
                FinalPropagationFailure::EventInvalid,
                QualificationLegFailureCode::FinalEventInvalid,
            ),
            (
                FinalPropagationFailure::IntegrationFailure,
                QualificationLegFailureCode::FinalIntegrationFailure,
            ),
            (
                FinalPropagationFailure::Eclipse(lightyear_odeint_rs::EclipseError::NonProgress),
                QualificationLegFailureCode::FinalEclipseNonProgress,
            ),
        ] {
            assert_eq!(
                transfer_failure_code(&TransferPropagationFailure::Final(failure)),
                expected
            );
        }
    }

    #[test]
    fn record_preflight_fails_closed_before_over_cap_ukf_work() {
        let mut storage = uninit_records::<MAX_QUALIFICATION_LEG_RECORDS>();
        let mut trace = QualificationLegTrace::try_new(fixed_identity(), &mut storage)
            .expect("fixed record arena is valid");
        let over_cap = MAX_QUALIFICATION_LEG_RECORDS
            .checked_add(1)
            .expect("fixed qualification ceiling has one successor");

        assert_eq!(
            trace.preflight_record_capacity(over_cap),
            Err(QualificationTraceError::RecordLimit)
        );
        assert_eq!(
            trace.validate_complete(),
            Err(QualificationTraceError::RecordLimit)
        );
    }

    #[test]
    fn caller_owned_storage_must_match_exact_trace_requirement() {
        assert_eq!(
            QualificationLegTrace::required_record_capacity(),
            MAX_QUALIFICATION_LEG_RECORDS
        );
        assert_eq!(
            Some(QualificationLegTrace::required_record_storage_bytes()),
            size_of::<QualificationLegRecord>().checked_mul(MAX_QUALIFICATION_LEG_RECORDS)
        );

        let mut too_shallow = uninit_records::<{ MAX_QUALIFICATION_LEG_RECORDS - 1 }>();
        assert!(matches!(
            QualificationLegTrace::try_new(fixed_identity(), &mut too_shallow),
            Err(QualificationTraceError::StorageTooSmall)
        ));

        let mut too_deep = uninit_records::<{ MAX_QUALIFICATION_LEG_RECORDS + 1 }>();
        assert!(matches!(
            QualificationLegTrace::try_new(fixed_identity(), &mut too_deep),
            Err(QualificationTraceError::StorageTooLarge)
        ));

        let mut storage = uninit_records::<MAX_QUALIFICATION_LEG_RECORDS>();
        let trace = QualificationLegTrace::try_new(fixed_identity(), &mut storage)
            .expect("fixed trace storage is valid");
        assert!(trace.records().is_empty());
    }

    #[test]
    fn records_preserve_owner_request_and_insertion_order() {
        let identity = fixed_identity();
        let mut storage = uninit_records::<MAX_QUALIFICATION_LEG_RECORDS>();
        let mut trace = QualificationLegTrace::try_new(identity, &mut storage)
            .expect("fixed trace storage is valid");
        let coast_input = QualificationLegInput::new(
            QualificationLegPath::ReleaseCanisterCoast,
            BodyRole::Canister,
            2_460_000.5,
            0.0,
            60.0,
            [7_000.0, 1.0, 2.0, 3.0, 7.5, 0.1],
        );
        let sigma_input = QualificationLegInput::new(
            QualificationLegPath::UkfSigma {
                component: 2,
                sigma: 12,
            },
            BodyRole::Dust,
            2_460_000.75,
            0.0,
            90.0,
            [7_100.0, 4.0, 5.0, 6.0, 7.4, 0.2],
        );
        let coast_endpoint = [7_001.0, 2.0, 3.0, 4.0, 7.49, 0.11];
        let sigma_endpoint = [7_101.0, 5.0, 6.0, 7.0, 7.39, 0.21];

        trace.record_transfer(
            coast_input,
            Ok(coast_endpoint),
            Ok(complete_metrics()),
            ObservedSolverTerminalStatus::Success,
        );
        trace.record_transfer(
            sigma_input,
            Ok(sigma_endpoint),
            Ok(complete_metrics()),
            ObservedSolverTerminalStatus::EventTriggered,
        );

        assert_eq!(trace.identity(), identity);
        let mut records = trace.records().iter();
        let first = records.next().expect("first recorded leg");
        assert_eq!(first.sequence, 0);
        assert_eq!(first.input, coast_input);
        assert_eq!(
            first.outcome,
            QualificationLegOutcome::Endpoint(coast_endpoint.map(f64::to_bits))
        );
        assert_eq!(first.metrics, Ok(complete_metrics()));
        assert_eq!(first.terminal_status, ObservedSolverTerminalStatus::Success);

        let second = records.next().expect("second recorded leg");
        assert_eq!(second.sequence, 1);
        assert_eq!(second.input, sigma_input);
        assert_eq!(
            second.outcome,
            QualificationLegOutcome::Endpoint(sigma_endpoint.map(f64::to_bits))
        );
        assert_eq!(second.metrics, Ok(complete_metrics()));
        assert_eq!(
            second.terminal_status,
            ObservedSolverTerminalStatus::EventTriggered
        );
        assert!(records.next().is_none());
        assert_eq!(trace.validate_complete(), Ok(()));
    }

    #[test]
    fn empty_trace_fails_closed() {
        let mut storage = uninit_records::<MAX_QUALIFICATION_LEG_RECORDS>();
        let trace = QualificationLegTrace::try_new(fixed_identity(), &mut storage)
            .expect("fixed trace storage is valid");
        assert_eq!(
            trace.validate_complete(),
            Err(QualificationTraceError::Empty)
        );
    }

    #[test]
    fn missing_metrics_fail_closed() {
        let mut storage = uninit_records::<MAX_QUALIFICATION_LEG_RECORDS>();
        let mut trace = QualificationLegTrace::try_new(fixed_identity(), &mut storage)
            .expect("fixed trace storage is valid");
        trace.record_transfer(
            QualificationLegInput::new(
                QualificationLegPath::ReleaseInterceptTrial,
                BodyRole::Dust,
                2_460_000.5,
                0.0,
                60.0,
                [7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            ),
            Ok([7_001.0, 0.0, 0.0, 0.0, 7.49, 0.0]),
            Err(ObservedFinalMetricError::CounterOverflow),
            ObservedSolverTerminalStatus::Unavailable,
        );
        assert_eq!(
            trace.validate_complete(),
            Err(QualificationTraceError::IncompleteMetrics)
        );
    }

    #[test]
    fn missing_nonzero_observation_fails_closed_but_zero_duration_stays_empty() {
        let mut storage = uninit_records::<MAX_QUALIFICATION_LEG_RECORDS>();
        let mut trace = QualificationLegTrace::try_new(fixed_identity(), &mut storage)
            .expect("fixed trace storage is valid");
        trace.record_observed_transfer(
            QualificationLegInput::new(
                QualificationLegPath::ReleaseInterceptTrial,
                BodyRole::Dust,
                2_460_000.5,
                0.0,
                60.0,
                [7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            ),
            Ok([7_001.0, 0.0, 0.0, 0.0, 7.49, 0.0]),
            None,
        );
        assert_eq!(
            trace.validate_complete(),
            Err(QualificationTraceError::IncompleteMetrics)
        );

        let mut storage = uninit_records::<MAX_QUALIFICATION_LEG_RECORDS>();
        let mut trace = QualificationLegTrace::try_new(fixed_identity(), &mut storage)
            .expect("fixed trace storage is valid");
        trace.record_observed_transfer(
            QualificationLegInput::new(
                QualificationLegPath::ReleaseEndpointFallback,
                BodyRole::Dust,
                2_460_000.5,
                0.0,
                0.0,
                [7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            ),
            Ok([7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0]),
            None,
        );
        assert_eq!(
            trace.validate_complete(),
            Err(QualificationTraceError::Empty)
        );
    }

    #[test]
    fn record_ceiling_never_truncates_silently() {
        let mut storage = uninit_records::<MAX_QUALIFICATION_LEG_RECORDS>();
        let mut trace = QualificationLegTrace::try_new(fixed_identity(), &mut storage)
            .expect("fixed trace storage is valid");
        let input = QualificationLegInput::new(
            QualificationLegPath::ReleaseInterceptTrial,
            BodyRole::Dust,
            2_460_000.5,
            0.0,
            0.0,
            [7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
        );
        for _ in 0..MAX_QUALIFICATION_LEG_RECORDS {
            trace.record_transfer(
                input,
                Ok([7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0]),
                Ok(ObservedFinalMetrics::default()),
                ObservedSolverTerminalStatus::Success,
            );
        }
        assert_eq!(trace.records().len(), MAX_QUALIFICATION_LEG_RECORDS);
        trace.record_transfer(
            input,
            Ok([7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0]),
            Ok(ObservedFinalMetrics::default()),
            ObservedSolverTerminalStatus::Success,
        );
        assert_eq!(trace.records().len(), MAX_QUALIFICATION_LEG_RECORDS);
        assert_eq!(
            trace.validate_complete(),
            Err(QualificationTraceError::RecordLimit)
        );
    }
}
