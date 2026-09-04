//! The one observer seam for transfer postprocess scalar legs.
//!
//! Postprocess used to thread `Option<&mut QualificationLegTrace>` through the
//! release control, intercept solve, and UKF sigma paths, and every call site
//! carried a `map_or_else(canonical, observed)` fork under
//! `cfg(feature = "solver-qualification")` plus a duplicate `cfg(not(..))`
//! body. Production now names an observer unconditionally and every trait
//! method's default body *is* the canonical call, so the shipped build
//! monomorphizes on the zero-sized [`UnobservedPostprocessLeg`] and reduces to
//! the same canonical calls it always made.
//!
//! `solver-qualification` supplies the observing implementation in
//! `super::qualification_trace`. Nothing in this module depends on the feature.

use crate::evaluate::TransferPropagationFailure;
use crate::intercept::{compute_miss_vector_hf_with_endpoint, HfInterceptEvaluation};
use crate::types::{BodyForceConfig, PlanContext, StampedEciState};

use super::distribution::propagate_stamped_checked;
use super::ukf::{propagate_sigma_states_with_native_batch, UkfPropagationFailure};

/// Which production leg one observed scalar propagation belongs to.
///
/// This is seam vocabulary rather than evidence: production names the leg at
/// each call site whether or not anyone is observing, so the call sites need
/// no conditional compilation. `solver-qualification` re-exports it under its
/// established public name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(
    dead_code,
    reason = "production names the leg it is on; only the feature-gated observing implementation constructs the per-trial and per-sigma variants"
)]
pub enum LegPath {
    /// The canister coast from launch to the release epoch.
    ReleaseCanisterCoast,
    /// One intercept-solver trial evaluation.
    ReleaseInterceptTrial,
    /// The dust propagation taken when no solver endpoint can be reused.
    ReleaseEndpointFallback,
    /// One UKF sigma point of one mixture component.
    UkfSigma {
        /// Mixture component ordinal.
        component: u8,
        /// Sigma point ordinal within the component.
        sigma: u8,
    },
}

/// The seam every observable postprocess scalar propagation passes through.
///
/// Each method's default body is the canonical production call. An observing
/// implementation overrides it with the fresh-context observed variant, which
/// computes the same numbers and additionally retains bounded evidence.
pub(super) trait PostprocessLegObserver {
    /// Reject a batch this observer cannot retain, before any numerical work.
    ///
    /// # Errors
    ///
    /// Returns when the observer's bounded storage cannot retain `legs` legs.
    fn preflight_leg_capacity(&mut self, legs: usize) -> Result<(), UkfPropagationFailure> {
        let _ = legs;
        Ok(())
    }

    /// Propagate one stamped state, retaining the exact source failure.
    ///
    /// # Errors
    ///
    /// Returns the typed propagation failure of the underlying call.
    fn propagate_stamped(
        &mut self,
        state: &StampedEciState,
        dt_s: f64,
        body_force: BodyForceConfig,
        ctx: &PlanContext,
        path: LegPath,
    ) -> Result<StampedEciState, TransferPropagationFailure> {
        let _ = path;
        propagate_stamped_checked(state, dt_s, body_force, ctx)
    }

    /// Evaluate one intercept-solver trial and its endpoint.
    ///
    /// # Errors
    ///
    /// Returns the typed propagation failure of the underlying call.
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
        compute_miss_vector_hf_with_endpoint(
            dv, v0, r0, target_pos, tof_s, source_jd, body_force, ctx,
        )
    }

    /// Propagate every UKF sigma state of one batch.
    ///
    /// The batch always starts at sigma row 0. It carried a
    /// `first_sigma_ordinal` until the julier7 simplex landed, because the R18
    /// sigma-row-0 endpoint reuse could hand this observer a batch beginning at
    /// row 1; the simplex has no centre point, so that reuse was removed and
    /// every batch is whole again.
    ///
    /// # Errors
    ///
    /// Returns a typed UKF failure when propagation authority is unavailable
    /// or the batch cannot complete.
    fn propagate_ukf_sigma_states(
        &mut self,
        sigma_eci_states: &[f64],
        sigma_propagated: &mut [f64],
        total_sigma: usize,
        tof_s: f64,
        ctx: &PlanContext,
    ) -> Result<(), UkfPropagationFailure> {
        propagate_sigma_states_with_native_batch(
            sigma_eci_states,
            sigma_propagated,
            total_sigma,
            tof_s,
            ctx,
        )
    }
}

/// The production observer: canonical calls, no evidence, zero size.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct UnobservedPostprocessLeg;

impl PostprocessLegObserver for UnobservedPostprocessLeg {}
