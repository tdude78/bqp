//! The one observer seam for the strict-HF mass solver.
//!
//! Production threads an observer through the solve unconditionally, so the
//! solver source carries no `solver-qualification` branch. Every trait method
//! has a default body that *is* the canonical production call, and the shipped
//! build monomorphizes on the zero-sized [`UnobservedMassSolve`]: the observer
//! argument is a ZST, `integrate_final_leg` inlines to the exact
//! `integrate_final_checked` call the solver always made, and the batch hooks
//! compile to nothing.
//!
//! The observed side lives entirely in `super::qualification`, which is the
//! only module that names the observed integrator API. Nothing here depends on
//! the `solver-qualification` feature.

use lightyear_odeint_rs::integrator::FinalPropagationFailure;
use lightyear_odeint_rs::{integrate_final_checked, ScalarPropagationRequest};

use super::MassSolveStatusCode;

/// Which production mass-solver call site executed one scalar final leg.
///
/// This is seam vocabulary rather than evidence: production names the role at
/// each call site whether or not anyone is observing, so the call sites need
/// no conditional compilation. `solver-qualification` re-exports it under its
/// established public name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MassLegRole {
    /// A normal repeated root-solver mass evaluation.
    MassEvaluation,
    /// The exact HF zero-mass anchor used for differential correction.
    ZeroMassAnchor,
    /// The direct zero-mass diagnostic after a non-finite normal evaluation.
    ZeroMassDiagnostic,
}

/// What the caller of a scalar final propagation knows before it runs.
///
/// The propagation helper is shared by three roles and by authorities that
/// never integrate at all, so the caller names its role here rather than the
/// helper inferring one.
#[derive(Clone, Copy, Debug)]
#[allow(
    dead_code,
    reason = "production carries the tag to a default body that ignores it; only the feature-gated observing implementation reads it"
)]
pub(super) struct MassLegTag {
    /// Production call-site role.
    pub(super) role: MassLegRole,
    /// Exact mass candidate used for this leg.
    pub(super) mass_kg_bits: u64,
}

/// Plain-data identity of the one scalar final leg about to be integrated.
///
/// Every field is an exact bit pattern taken from the production inputs at the
/// call site. The seam carries these unconditionally; only an observing
/// implementation reads them.
#[derive(Clone, Copy, Debug)]
#[allow(
    dead_code,
    reason = "production carries the identity to a default body that ignores it; only the feature-gated observing implementation reads it"
)]
pub(super) struct MassLegIdentity {
    /// Role and mass candidate supplied by the call site.
    pub(super) tag: MassLegTag,
    /// UTC-JD source epoch bound into the strict-HF scalar context.
    pub(super) source_jd_bits: u64,
    /// Exact inertial Cartesian state handed to the scalar propagator.
    pub(super) initial_eci_bits: [u64; 6],
    /// Exact initial equinoctial state passed to the final propagator.
    pub(super) initial_equinoctial_bits: [u64; 6],
    /// Exact propagation start time in seconds.
    pub(super) t0_s_bits: u64,
    /// Exact propagation end time in seconds.
    pub(super) t_final_s_bits: u64,
}

/// The seam every strict-HF scalar final propagation passes through.
pub(super) trait MassSolveObserver {
    /// Run the one authoritative scalar final propagation for `leg`.
    ///
    /// The default body is the canonical production call and retains nothing.
    fn integrate_final_leg(
        &mut self,
        request: ScalarPropagationRequest<'_>,
        leg: &MassLegIdentity,
    ) -> Result<[f64; 6], FinalPropagationFailure> {
        let _ = leg;
        integrate_final_checked(request)
    }
}

/// Per-row evidence opened by a [`MassBatchObserver`].
pub(super) trait MassRowObservation: MassSolveObserver {
    /// Number of actual scalar legs this row retained.
    fn retained_legs(&self) -> usize;

    /// Seal this row's evidence before the batch commits it.
    ///
    /// # Errors
    ///
    /// Returns when the row retained a defective or out-of-order record.
    fn seal(self) -> anyhow::Result<()>;
}

/// The batch seam around the one serial strict-HF row loop.
///
/// Observed and unobserved batches run the identical loop: the same cache
/// construction, row timer, solver call, and profile capture.
pub(super) trait MassBatchObserver {
    /// Per-row observer handed to the one production solve.
    type Row<'row>: MassRowObservation
    where
        Self: 'row;

    /// Reject a batch this observer cannot retain, before any row work.
    ///
    /// # Errors
    ///
    /// Returns when the observer's caller-owned storage cannot represent
    /// `expected_rows` rows.
    fn preflight_batch(&mut self, expected_rows: usize) -> anyhow::Result<()>;

    /// Open evidence for one row in deterministic input order.
    ///
    /// # Errors
    ///
    /// Returns when the row is out of order or storage is exhausted.
    fn open_row(&mut self, row: usize) -> anyhow::Result<Self::Row<'_>>;

    /// Seal one completed row after the production solve returned.
    ///
    /// # Errors
    ///
    /// Returns when the row is out of order or cannot be retained.
    fn seal_row(
        &mut self,
        row: usize,
        mass_kg: f64,
        status: MassSolveStatusCode,
        leg_count: usize,
    ) -> anyhow::Result<()>;
}

/// The production observer: canonical integrator, no evidence, zero size.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct UnobservedMassSolve;

impl MassSolveObserver for UnobservedMassSolve {}

impl MassRowObservation for UnobservedMassSolve {
    fn retained_legs(&self) -> usize {
        0
    }

    fn seal(self) -> anyhow::Result<()> {
        Ok(())
    }
}

impl MassBatchObserver for UnobservedMassSolve {
    type Row<'row> = Self;

    fn preflight_batch(&mut self, expected_rows: usize) -> anyhow::Result<()> {
        let _ = expected_rows;
        Ok(())
    }

    fn open_row(&mut self, row: usize) -> anyhow::Result<Self::Row<'_>> {
        let _ = row;
        Ok(Self)
    }

    fn seal_row(
        &mut self,
        row: usize,
        mass_kg: f64,
        status: MassSolveStatusCode,
        leg_count: usize,
    ) -> anyhow::Result<()> {
        let _ = (row, mass_kg, status, leg_count);
        Ok(())
    }
}
