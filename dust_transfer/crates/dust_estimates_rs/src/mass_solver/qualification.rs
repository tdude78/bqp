//! Bounded, caller-owned scalar-leg records for solver qualification.
//!
//! This module is feature-only. It deliberately owns no corpus, wire framing,
//! global storage, or runtime controls; `nd_pipeline` owns those higher-level
//! concerns. Records are retained only in a caller-provided fixed slice.

use lightyear_odeint_rs::integrator::{
    integrate_final_checked_observed, FinalPropagationFailure, ObservedFinalLeg,
};
use lightyear_odeint_rs::ScalarPropagationRequest;

pub use super::observer::MassLegRole as QualificationMassLegRole;
use super::observer::{MassBatchObserver, MassLegIdentity, MassRowObservation, MassSolveObserver};
use super::MassSolveStatusCode;

/// Maximum scalar final legs retained for one qualified mass solve.
///
/// This is the independent bound for one W1 mass batch; downstream event
/// evidence also includes release and UKF legs and therefore has a larger
/// aggregate ceiling.
pub const MAX_QUALIFICATION_MASS_LEGS: usize = 32_768;

/// Maximum completed rows retained by one qualification W1 mass batch.
///
/// This is independent from the scalar-leg limit: a cache-hit row can retain
/// no scalar leg but still consumes one sealed row record.
pub const MAX_QUALIFICATION_MASS_BATCH_ROWS: usize = 4_096;

/// One ordered, observed scalar final propagation from the mass solver.
#[derive(Clone, Debug)]
pub struct QualificationMassLeg {
    /// Zero-based execution order within this one mass solve.
    pub leg_ordinal: u32,
    /// Production call-site role.
    pub role: QualificationMassLegRole,
    /// Exact mass candidate used for this leg.
    pub mass_kg_bits: u64,
    /// UTC-JD source epoch bound into the strict-HF scalar context.
    pub source_jd_bits: u64,
    /// Exact inertial Cartesian state handed to the scalar propagator.
    pub initial_eci_bits: [u64; 6],
    /// Exact initial equinoctial state passed to the final propagator.
    ///
    /// This remains alongside `initial_eci_bits` because the scalar solver
    /// consumes equinoctial coordinates after the one authoritative frame
    /// conversion. Qualification evidence seals both the external ECI request
    /// semantics and the exact solver input.
    pub initial_equinoctial_bits: [u64; 6],
    /// Exact propagation start time in seconds.
    pub t0_s_bits: u64,
    /// Exact propagation end time in seconds.
    pub t_final_s_bits: u64,
    /// Fresh-RHS observed final-leg outcome and local counters.
    pub observed: ObservedFinalLeg,
}

/// Typed failure of caller-owned observation storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QualificationMassObservationError {
    /// Caller offered more slots than the compiled qualification ceiling.
    CapacityExceedsMaximum,
    /// A caller slot was not empty when recording began.
    SlotNotEmpty,
    /// Another observed leg did not fit in the caller-provided fixed slice.
    CapacityExceeded,
    /// A record ordinal cannot be represented by the sealed wire integer type.
    OrdinalOverflow,
    /// A retained ordinal has no record.
    MissingLeg,
    /// A retained record's ordinal no longer matches its fixed position.
    LegOrdinalMismatch,
    /// The bounded batch-row storage cannot represent another completed row.
    BatchRowCapacityExceeded,
    /// A batch-row slot was already occupied before recording began.
    BatchRowSlotNotEmpty,
    /// A batch row was committed out of its deterministic input order.
    BatchRowOrderMismatch,
    /// Caller offered more batch-row slots than qualification permits.
    BatchRowsExceedMaximum,
    /// Caller-owned batch arenas exceed their fixed total byte ceiling.
    BatchStorageBytesExceeded,
}

impl core::fmt::Display for QualificationMassObservationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::CapacityExceedsMaximum => {
                "qualification mass observation capacity exceeds maximum"
            }
            Self::SlotNotEmpty => "qualification mass observation slot is not empty",
            Self::CapacityExceeded => "qualification mass observation capacity exceeded",
            Self::OrdinalOverflow => "qualification mass observation ordinal overflow",
            Self::MissingLeg => "qualification mass observation record is missing",
            Self::LegOrdinalMismatch => "qualification mass observation ordinal mismatch",
            Self::BatchRowCapacityExceeded => "qualification mass batch row capacity exceeded",
            Self::BatchRowSlotNotEmpty => "qualification mass batch row slot is not empty",
            Self::BatchRowOrderMismatch => "qualification mass batch row order mismatch",
            Self::BatchRowsExceedMaximum => "qualification mass batch row capacity exceeds maximum",
            Self::BatchStorageBytesExceeded => {
                "qualification mass batch storage exceeds fixed byte ceiling"
            }
        })
    }
}

impl std::error::Error for QualificationMassObservationError {}

/// Caller-owned, fixed-capacity observed-leg storage.
///
/// A capacity failure latches and records no more legs, but never changes the
/// numerical solve. The enclosing batch validates before committing each row.
pub(super) struct QualificationMassObservation<'a> {
    slots: &'a mut [Option<QualificationMassLeg>],
    recorded: usize,
    failure: Option<QualificationMassObservationError>,
}

impl<'a> QualificationMassObservation<'a> {
    /// Start an empty fixed-capacity observation.
    ///
    /// # Errors
    ///
    /// Returns a typed error if capacity exceeds the compiled ceiling or any
    /// caller-owned slot is already occupied.
    fn new(
        slots: &'a mut [Option<QualificationMassLeg>],
    ) -> Result<Self, QualificationMassObservationError> {
        if slots.len() > MAX_QUALIFICATION_MASS_LEGS {
            return Err(QualificationMassObservationError::CapacityExceedsMaximum);
        }
        if slots.iter().any(Option::is_some) {
            return Err(QualificationMassObservationError::SlotNotEmpty);
        }
        Ok(Self {
            slots,
            recorded: 0,
            failure: None,
        })
    }

    /// Number of retained prefix records.
    #[must_use]
    pub(super) const fn recorded_len(&self) -> usize {
        self.recorded
    }

    /// Return one ordered retained record.
    ///
    /// # Errors
    ///
    /// Returns `MissingLeg` when `ordinal` has no retained record.
    fn leg(
        &self,
        ordinal: usize,
    ) -> Result<&QualificationMassLeg, QualificationMassObservationError> {
        self.slots
            .get(ordinal)
            .and_then(Option::as_ref)
            .ok_or(QualificationMassObservationError::MissingLeg)
    }

    /// Validate one batch row before the enclosing batch seals it.
    ///
    /// A cache-hit row may correctly retain zero scalar legs. Any storage or
    /// ordering defect still fails closed before row commit.
    pub(super) fn finish(&self) -> Result<(), QualificationMassObservationError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        for ordinal in 0..self.recorded {
            let expected = u32::try_from(ordinal)
                .map_err(|_| QualificationMassObservationError::OrdinalOverflow)?;
            let record = self.leg(ordinal)?;
            if record.leg_ordinal != expected {
                return Err(QualificationMassObservationError::LegOrdinalMismatch);
            }
        }
        Ok(())
    }

    pub(super) fn record(
        &mut self,
        role: QualificationMassLegRole,
        mass_kg_bits: u64,
        source_jd_bits: u64,
        initial_eci_bits: [u64; 6],
        initial_equinoctial_bits: [u64; 6],
        t0_s_bits: u64,
        t_final_s_bits: u64,
        observed: ObservedFinalLeg,
    ) {
        if self.failure.is_some() {
            return;
        }
        let Ok(ordinal) = u32::try_from(self.recorded) else {
            self.failure = Some(QualificationMassObservationError::OrdinalOverflow);
            return;
        };
        let Some(next_recorded) = self.recorded.checked_add(1) else {
            self.failure = Some(QualificationMassObservationError::OrdinalOverflow);
            return;
        };
        let Some(slot) = self.slots.get_mut(self.recorded) else {
            self.failure = Some(QualificationMassObservationError::CapacityExceeded);
            return;
        };
        if slot.is_some() {
            self.failure = Some(QualificationMassObservationError::SlotNotEmpty);
            return;
        }
        *slot = Some(QualificationMassLeg {
            leg_ordinal: ordinal,
            role,
            mass_kg_bits,
            source_jd_bits,
            initial_eci_bits,
            initial_equinoctial_bits,
            t0_s_bits,
            t_final_s_bits,
            observed,
        });
        self.recorded = next_recorded;
    }
}

/// One completed strict-HF mass row in actual batch execution order.
///
/// `first_leg..first_leg + leg_count` is a bounded contiguous range in the
/// caller-owned batch record arena. A zero count is valid when the row used a
/// preflight terminal or an already-initialized exact cache entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QualificationMassBatchRow {
    pub mass_kg_bits: u64,
    pub status: MassSolveStatusCode,
    pub first_leg: u32,
    pub leg_count: u32,
}

/// Bounded, caller-owned records for one serial qualification mass batch.
///
/// The fixed W1 qualification path owns one flat leg arena and one ordered row
/// arena. It does not allocate, dispatch callbacks, or create a second cache.
/// Cache hits retain no synthetic leg; their completed row records a zero
/// length range instead.
pub struct QualificationMassBatchObservation<'legs, 'rows> {
    leg_slots: &'legs mut [Option<QualificationMassLeg>],
    row_slots: &'rows mut [Option<QualificationMassBatchRow>],
    used_legs: usize,
    committed_rows: usize,
}

impl<'legs, 'rows> QualificationMassBatchObservation<'legs, 'rows> {
    /// Bind empty, bounded caller storage before a qualification mass batch.
    ///
    /// # Errors
    ///
    /// Returns a typed error when storage exceeds the compiled leg ceiling or
    /// when either caller-owned arena already contains a record.
    pub fn new(
        leg_slots: &'legs mut [Option<QualificationMassLeg>],
        row_slots: &'rows mut [Option<QualificationMassBatchRow>],
    ) -> Result<Self, QualificationMassObservationError> {
        if leg_slots.len() > MAX_QUALIFICATION_MASS_LEGS {
            return Err(QualificationMassObservationError::CapacityExceedsMaximum);
        }
        if row_slots.len() > MAX_QUALIFICATION_MASS_BATCH_ROWS {
            return Err(QualificationMassObservationError::BatchRowsExceedMaximum);
        }
        validate_qualification_batch_storage(leg_slots.len(), row_slots.len())?;
        if leg_slots.iter().any(Option::is_some) {
            return Err(QualificationMassObservationError::SlotNotEmpty);
        }
        if row_slots.iter().any(Option::is_some) {
            return Err(QualificationMassObservationError::BatchRowSlotNotEmpty);
        }
        Ok(Self {
            leg_slots,
            row_slots,
            used_legs: 0,
            committed_rows: 0,
        })
    }

    /// Number of actual scalar legs retained across all committed rows.
    #[must_use]
    pub const fn recorded_len(&self) -> usize {
        self.used_legs
    }

    /// Number of completed rows retained in deterministic input order.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.committed_rows
    }

    /// Read one actual scalar leg.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the requested retained leg is absent.
    pub fn leg(
        &self,
        ordinal: usize,
    ) -> Result<&QualificationMassLeg, QualificationMassObservationError> {
        self.leg_slots
            .get(ordinal)
            .and_then(Option::as_ref)
            .ok_or(QualificationMassObservationError::MissingLeg)
    }

    /// Read one completed batch row.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the requested row is absent.
    pub fn row(
        &self,
        ordinal: usize,
    ) -> Result<&QualificationMassBatchRow, QualificationMassObservationError> {
        self.row_slots
            .get(ordinal)
            .and_then(Option::as_ref)
            .ok_or(QualificationMassObservationError::BatchRowCapacityExceeded)
    }

    pub(super) fn begin_row(
        &mut self,
        row: usize,
    ) -> Result<QualificationMassObservation<'_>, QualificationMassObservationError> {
        if row != self.committed_rows {
            return Err(QualificationMassObservationError::BatchRowOrderMismatch);
        }
        if self.row_slots.get(row).is_none() {
            return Err(QualificationMassObservationError::BatchRowCapacityExceeded);
        }
        let leg_slots = self
            .leg_slots
            .get_mut(self.used_legs..)
            .ok_or(QualificationMassObservationError::CapacityExceeded)?;
        QualificationMassObservation::new(leg_slots)
    }

    pub(super) fn preflight_rows(
        &self,
        expected_rows: usize,
    ) -> Result<(), QualificationMassObservationError> {
        if expected_rows > MAX_QUALIFICATION_MASS_BATCH_ROWS {
            return Err(QualificationMassObservationError::BatchRowsExceedMaximum);
        }
        if self.committed_rows != 0 || self.row_slots.len() != expected_rows {
            return Err(QualificationMassObservationError::BatchRowCapacityExceeded);
        }
        if self.row_slots.iter().any(Option::is_some) {
            return Err(QualificationMassObservationError::BatchRowSlotNotEmpty);
        }
        Ok(())
    }

    pub(super) fn commit_row(
        &mut self,
        row: usize,
        mass_kg: f64,
        status: MassSolveStatusCode,
        leg_count: usize,
    ) -> Result<(), QualificationMassObservationError> {
        if row != self.committed_rows {
            return Err(QualificationMassObservationError::BatchRowOrderMismatch);
        }
        let first_leg = u32::try_from(self.used_legs)
            .map_err(|_| QualificationMassObservationError::OrdinalOverflow)?;
        let leg_count_u32 = u32::try_from(leg_count)
            .map_err(|_| QualificationMassObservationError::OrdinalOverflow)?;
        let next_used_legs = self
            .used_legs
            .checked_add(leg_count)
            .ok_or(QualificationMassObservationError::OrdinalOverflow)?;
        if next_used_legs > self.leg_slots.len() {
            return Err(QualificationMassObservationError::CapacityExceeded);
        }
        let row_slot = self
            .row_slots
            .get_mut(row)
            .ok_or(QualificationMassObservationError::BatchRowCapacityExceeded)?;
        if row_slot.is_some() {
            return Err(QualificationMassObservationError::BatchRowSlotNotEmpty);
        }
        *row_slot = Some(QualificationMassBatchRow {
            mass_kg_bits: mass_kg.to_bits(),
            status,
            first_leg,
            leg_count: leg_count_u32,
        });
        self.used_legs = next_used_legs;
        self.committed_rows = self
            .committed_rows
            .checked_add(1)
            .ok_or(QualificationMassObservationError::OrdinalOverflow)?;
        Ok(())
    }
}

/// The observing side of the mass-solver seam.
///
/// This is the only place the observed integrator API is named. It runs the
/// fresh-RHS observed final call and retains the leg; the numerical outcome it
/// returns is the same value the canonical call would have produced.
impl MassSolveObserver for QualificationMassObservation<'_> {
    fn integrate_final_leg(
        &mut self,
        request: ScalarPropagationRequest<'_>,
        leg: &MassLegIdentity,
    ) -> Result<[f64; 6], FinalPropagationFailure> {
        // The observed API constructs a fresh RHS/context internally; never
        // reuse a canonical or prior-arm Encke history here.
        let observed = integrate_final_checked_observed(request);
        let outcome = observed.outcome;
        self.record(
            leg.tag.role,
            leg.tag.mass_kg_bits,
            leg.source_jd_bits,
            leg.initial_eci_bits,
            leg.initial_equinoctial_bits,
            leg.t0_s_bits,
            leg.t_final_s_bits,
            observed,
        );
        outcome
    }
}

impl MassRowObservation for QualificationMassObservation<'_> {
    fn retained_legs(&self) -> usize {
        self.recorded_len()
    }

    fn seal(self) -> anyhow::Result<()> {
        Self::finish(&self)?;
        Ok(())
    }
}

impl MassBatchObserver for QualificationMassBatchObservation<'_, '_> {
    type Row<'row>
        = QualificationMassObservation<'row>
    where
        Self: 'row;

    fn preflight_batch(&mut self, expected_rows: usize) -> anyhow::Result<()> {
        Self::preflight_rows(self, expected_rows)?;
        Ok(())
    }

    fn open_row(&mut self, row: usize) -> anyhow::Result<Self::Row<'_>> {
        Ok(Self::begin_row(self, row)?)
    }

    fn seal_row(
        &mut self,
        row: usize,
        mass_kg: f64,
        status: MassSolveStatusCode,
        leg_count: usize,
    ) -> anyhow::Result<()> {
        Self::commit_row(self, row, mass_kg, status, leg_count)?;
        Ok(())
    }
}

/// Return exact storage bytes for bounded batch arenas before allocation.
fn qualification_batch_storage_bytes(
    leg_slots: usize,
    row_slots: usize,
) -> Result<usize, QualificationMassObservationError> {
    let leg_bytes = leg_slots
        .checked_mul(core::mem::size_of::<Option<QualificationMassLeg>>())
        .ok_or(QualificationMassObservationError::BatchStorageBytesExceeded)?;
    let row_bytes = row_slots
        .checked_mul(core::mem::size_of::<Option<QualificationMassBatchRow>>())
        .ok_or(QualificationMassObservationError::BatchStorageBytesExceeded)?;
    let total_bytes = leg_bytes
        .checked_add(row_bytes)
        .ok_or(QualificationMassObservationError::BatchStorageBytesExceeded)?;
    Ok(total_bytes)
}

/// Return fixed byte ceiling implied by the compiled record ceilings.
fn max_qualification_batch_storage_bytes() -> Result<usize, QualificationMassObservationError> {
    qualification_batch_storage_bytes(
        MAX_QUALIFICATION_MASS_LEGS,
        MAX_QUALIFICATION_MASS_BATCH_ROWS,
    )
}

fn validate_qualification_batch_storage(
    leg_slots: usize,
    row_slots: usize,
) -> Result<(), QualificationMassObservationError> {
    let total_bytes = qualification_batch_storage_bytes(leg_slots, row_slots)?;
    let max_bytes = max_qualification_batch_storage_bytes()?;
    if total_bytes > max_bytes {
        return Err(QualificationMassObservationError::BatchStorageBytesExceeded);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_observation_rejects_row_and_byte_capacity_before_recording() {
        let mut leg_slots: [Option<QualificationMassLeg>; 0] = [];
        let mut too_many_rows = (0..MAX_QUALIFICATION_MASS_BATCH_ROWS.saturating_add(1))
            .map(|_| None)
            .collect::<Vec<Option<QualificationMassBatchRow>>>();
        let row_error =
            QualificationMassBatchObservation::new(&mut leg_slots, &mut too_many_rows).err();
        assert_eq!(
            row_error,
            Some(QualificationMassObservationError::BatchRowsExceedMaximum)
        );
        let byte_ceiling = max_qualification_batch_storage_bytes()
            .expect("compiled qualification batch ceiling fits usize");
        assert_eq!(
            qualification_batch_storage_bytes(
                MAX_QUALIFICATION_MASS_LEGS,
                MAX_QUALIFICATION_MASS_BATCH_ROWS,
            ),
            Ok(byte_ceiling)
        );
        assert_eq!(
            validate_qualification_batch_storage(
                MAX_QUALIFICATION_MASS_LEGS,
                MAX_QUALIFICATION_MASS_BATCH_ROWS.saturating_add(1),
            ),
            Err(QualificationMassObservationError::BatchStorageBytesExceeded)
        );
        assert_eq!(
            qualification_batch_storage_bytes(usize::MAX, 0),
            Err(QualificationMassObservationError::BatchStorageBytesExceeded)
        );
    }
}
