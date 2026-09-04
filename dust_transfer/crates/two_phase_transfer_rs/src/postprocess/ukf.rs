use nalgebra::{SMatrix, SVector};
#[cfg(feature = "solver-qualification")]
use satpy_core::eci2equinoc_impl_f64;
#[cfg(any(test, feature = "bench-internal"))]
use satpy_core::equinoc_prop_from_impl;
use satpy_core::{eci2equinoc_impl, equinoc_prop_j2_batch_impl};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

#[cfg(feature = "solver-qualification")]
use crate::evaluate::propagate_high_fidelity_state_at_epoch_checked_observed;
use crate::evaluate::TransferPropagationFailure;
use crate::types::PlanContext;

#[cfg(feature = "solver-qualification")]
use super::observer::LegPath;
use super::observer::{PostprocessLegObserver, UnobservedPostprocessLeg};
#[cfg(any(test, feature = "bench-internal"))]
use super::propagate_equinoctial;
use super::propagate_with_ctx_checked;
#[cfg(all(test, feature = "solver-qualification"))]
use super::qualification_trace::MAX_QUALIFICATION_LEG_RECORDS;
#[cfg(feature = "solver-qualification")]
use super::qualification_trace::{
    QualificationLegInput, QualificationLegTrace, QualificationTraceError,
};
#[cfg(any(test, feature = "bench-internal"))]
use super::session::TransferPostprocessScratch;

// Julier minimal-skew simplex at W0 = 0: every point carries the same weight
// `1 / (n + 1)` in both moments, so there is no tuning triple to condition and
// no sign hazard to guard. See `dust_ukf_rs::julier_simplex_weights`.
//
// The one weights object every consumer reads: both production sigma paths and
// the tests bind this const, so they provably share the same compile-time
// evaluation of the const fn (bit-identical to a runtime call by IEEE const
// eval).
pub(super) const UKF_SIGMA_WEIGHTS: dust_ukf_rs::SigmaWeights =
    dust_ukf_rs::julier_simplex_weights();

/// Failure from a postprocess UKF lane.
///
/// Strict-HF callers retain [`TransferPropagationFailure`] until their explicit
/// outer row boundary. Generic callers receive this typed error directly.
#[derive(Clone, Debug)]
pub enum UkfPropagationFailure {
    InvalidInput,
    SigmaConstruction,
    Allocation,
    #[cfg(feature = "solver-qualification")]
    Qualification(QualificationTraceError),
    Propagation(TransferPropagationFailure),
    Ephemeris {
        row: usize,
        message: String,
        source: Arc<lightyear_odeint_rs::session::VariableFinalNativeError>,
    },
    NativeBatch {
        source: Option<Arc<lightyear_odeint_rs::session::VariableFinalNativeError>>,
    },
    NonFiniteOutput,
}

impl PartialEq for UkfPropagationFailure {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::InvalidInput, Self::InvalidInput)
            | (Self::SigmaConstruction, Self::SigmaConstruction)
            | (Self::Allocation, Self::Allocation)
            | (Self::NonFiniteOutput, Self::NonFiniteOutput) => true,
            #[cfg(feature = "solver-qualification")]
            (Self::Qualification(left), Self::Qualification(right)) => left == right,
            (Self::Propagation(left), Self::Propagation(right)) => left == right,
            (
                Self::Ephemeris {
                    row: left_row,
                    message: left_message,
                    ..
                },
                Self::Ephemeris {
                    row: right_row,
                    message: right_message,
                    ..
                },
            ) => left_row == right_row && left_message == right_message,
            (Self::NativeBatch { .. }, Self::NativeBatch { .. }) => true,
            _ => false,
        }
    }
}

impl fmt::Display for UkfPropagationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput => formatter.write_str("UKF input is invalid"),
            Self::SigmaConstruction => formatter.write_str("UKF sigma construction failed"),
            Self::Allocation => formatter.write_str("UKF allocation failed"),
            #[cfg(feature = "solver-qualification")]
            Self::Qualification(_) => formatter.write_str("UKF qualification trace is incomplete"),
            Self::Propagation(error) => write!(formatter, "UKF propagation: {error}"),
            Self::Ephemeris { row, message, .. } => {
                write!(formatter, "UKF ephemeris row {row}: {message}")
            }
            Self::NativeBatch { .. } => formatter.write_str("UKF native batch failed"),
            Self::NonFiniteOutput => formatter.write_str("UKF output is non-finite"),
        }
    }
}

impl std::error::Error for UkfPropagationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Propagation(error) => Some(error),
            Self::Ephemeris { source, .. } => Some(source.as_ref()),
            Self::NativeBatch {
                source: Some(source),
            } => Some(source.as_ref()),
            Self::NativeBatch { source: None } => None,
            #[cfg(feature = "solver-qualification")]
            Self::Qualification(_) => None,
            Self::InvalidInput
            | Self::SigmaConstruction
            | Self::Allocation
            | Self::NonFiniteOutput => None,
        }
    }
}

const UKF_STATE_WIDTH: usize = 6;
const PRODUCTION_RETAIN_PROPAGATED_SIGMA_POINTS: bool = false;

#[cfg(test)]
std::thread_local! {
    static SIGMA_MATERIALIZATION_COUNT: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

#[cfg(test)]
fn reset_sigma_materialization_count() {
    SIGMA_MATERIALIZATION_COUNT.set(0);
}

#[cfg(test)]
fn sigma_materialization_count() -> usize {
    SIGMA_MATERIALIZATION_COUNT.get()
}

/// Named output from the diagnostic full-UKF batch lane.
#[derive(Debug, PartialEq)]
pub struct UkfFullBatchOutput {
    /// Propagated mean and covariance for every release component.
    pub propagated_components: Vec<([f64; 6], [[f64; 6]; 6])>,
    /// Propagated state of every sigma point in component-major order.
    pub propagated_sigma_points: Vec<[f64; 6]>,
}

#[inline]
fn try_reserve_ukf<T>(values: &mut Vec<T>, additional: usize) -> Result<(), UkfPropagationFailure> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| UkfPropagationFailure::Allocation)
}

#[inline]
fn try_reserve_ukf_to_len<T>(
    values: &mut Vec<T>,
    target_len: usize,
) -> Result<(), UkfPropagationFailure> {
    if target_len > values.len() {
        let additional = target_len
            .checked_sub(values.len())
            .ok_or(UkfPropagationFailure::InvalidInput)?;
        try_reserve_ukf(values, additional)?;
    }
    Ok(())
}

#[inline]
#[cfg(any(test, feature = "bench-internal"))]
fn try_reserve_smallvec_ukf<A>(
    values: &mut smallvec::SmallVec<A>,
    additional: usize,
) -> Result<(), UkfPropagationFailure>
where
    A: smallvec::Array,
{
    values
        .try_reserve_exact(additional)
        .map_err(|_| UkfPropagationFailure::Allocation)
}

#[inline]
#[cfg(any(test, feature = "bench-internal"))]
fn try_reserve_smallvec_ukf_to_len<A>(
    values: &mut smallvec::SmallVec<A>,
    target_len: usize,
) -> Result<(), UkfPropagationFailure>
where
    A: smallvec::Array,
{
    if target_len > values.len() {
        let additional = target_len
            .checked_sub(values.len())
            .ok_or(UkfPropagationFailure::InvalidInput)?;
        try_reserve_smallvec_ukf(values, additional)?;
    }
    Ok(())
}

#[inline]
fn try_resize_ukf<T: Clone>(
    values: &mut Vec<T>,
    target_len: usize,
    value: T,
) -> Result<(), UkfPropagationFailure> {
    try_reserve_ukf_to_len(values, target_len)?;
    values.resize(target_len, value);
    Ok(())
}

#[inline]
#[cfg(any(test, feature = "bench-internal"))]
fn try_resize_smallvec_ukf<A>(
    values: &mut smallvec::SmallVec<A>,
    target_len: usize,
    value: A::Item,
) -> Result<(), UkfPropagationFailure>
where
    A: smallvec::Array,
    A::Item: Clone,
{
    try_reserve_smallvec_ukf_to_len(values, target_len)?;
    values.resize(target_len, value);
    Ok(())
}

#[inline]
fn try_ukf_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, UkfPropagationFailure> {
    let mut values = Vec::new();
    try_reserve_ukf(&mut values, capacity)?;
    Ok(values)
}

#[inline]
fn try_ukf_filled_vec<T: Clone>(length: usize, value: T) -> Result<Vec<T>, UkfPropagationFailure> {
    let mut values = Vec::new();
    try_resize_ukf(&mut values, length, value)?;
    Ok(values)
}

#[cfg(any(test, feature = "bench-internal"))]
fn preflight_component_mean_scratch(
    scratch: &mut TransferPostprocessScratch,
    component_count: usize,
    sigma_storage_len: usize,
    total_sigma: usize,
    needs_sigma_tofs: bool,
) -> Result<(), UkfPropagationFailure> {
    try_reserve_smallvec_ukf_to_len(&mut scratch.comp_means, component_count)?;
    try_reserve_ukf_to_len(&mut scratch.sigma_states, sigma_storage_len)?;
    try_reserve_smallvec_ukf_to_len(&mut scratch.component_sigma_offsets, component_count)?;
    try_reserve_ukf_to_len(&mut scratch.sigma_equinoc, sigma_storage_len)?;
    try_reserve_ukf_to_len(&mut scratch.sigma_propagated, sigma_storage_len)?;
    if needs_sigma_tofs {
        try_reserve_ukf_to_len(&mut scratch.sigma_tofs, total_sigma)?;
    }
    Ok(())
}

#[inline]
fn checked_ukf_product(left: usize, right: usize) -> Result<usize, UkfPropagationFailure> {
    left.checked_mul(right)
        .ok_or(UkfPropagationFailure::InvalidInput)
}

#[inline]
fn checked_ukf_row_range(
    row_index: usize,
    storage_len: usize,
) -> Result<Range<usize>, UkfPropagationFailure> {
    let start = checked_ukf_product(row_index, UKF_STATE_WIDTH)?;
    let end = start
        .checked_add(UKF_STATE_WIDTH)
        .ok_or(UkfPropagationFailure::InvalidInput)?;
    (end <= storage_len)
        .then_some(start..end)
        .ok_or(UkfPropagationFailure::InvalidInput)
}

#[cfg(any(test, feature = "bench-internal"))]
fn classify_native_batch_error(error: &anyhow::Error) -> UkfPropagationFailure {
    if let Some(failure) =
        error.downcast_ref::<lightyear_odeint_rs::integrator::FinalPropagationFailure>()
    {
        return UkfPropagationFailure::Propagation(TransferPropagationFailure::from(*failure));
    }
    if let Some(error) =
        error.downcast_ref::<lightyear_odeint_rs::precomputed_ephem::EphemerisCoverageError>()
    {
        return UkfPropagationFailure::Propagation(TransferPropagationFailure::Ephemeris(
            error.clone(),
        ));
    }
    UkfPropagationFailure::NativeBatch { source: None }
}

/// Propagate component means through the UKF batch lane.
///
/// # Errors
///
/// Returns a typed UKF failure rather than erasing an invalid input,
/// unavailable propagation authority, or non-finite output.
#[cfg(any(test, feature = "bench-internal"))]
pub(super) fn propagate_component_means_ukf_batch(
    means: &[[f64; 6]],
    covs: &[[[f64; 6]; 6]],
    tof_s: f64,
    ctx: Option<&PlanContext>,
    scratch: &mut TransferPostprocessScratch,
) -> Result<usize, UkfPropagationFailure> {
    if means.len() != covs.len() || !tof_s.is_finite() {
        return Err(UkfPropagationFailure::InvalidInput);
    }
    let component_count = means.len();
    if component_count == 0 {
        scratch.comp_means.clear();
        return Ok(0);
    }

    let total_sigma = checked_ukf_product(component_count, dust_ukf_rs::NUM_SIGMA)?;
    let sigma_storage_len = checked_ukf_product(total_sigma, UKF_STATE_WIDTH)?;
    let use_hf_batch = ctx.is_some_and(|ctx_ref| ctx_ref.execution_policy.use_high_fidelity);
    preflight_component_mean_scratch(
        scratch,
        component_count,
        sigma_storage_len,
        total_sigma,
        !use_hf_batch,
    )?;

    try_resize_smallvec_ukf(&mut scratch.comp_means, component_count, [0.0; 6])?;
    if scratch.sigma_states.len() != sigma_storage_len {
        try_resize_ukf(&mut scratch.sigma_states, sigma_storage_len, 0.0)?;
    }
    if scratch.component_sigma_offsets.len() != component_count {
        try_resize_smallvec_ukf(&mut scratch.component_sigma_offsets, component_count, 0)?;
    }

    for (component_idx, (mean, cov)) in means.iter().zip(covs.iter()).enumerate() {
        let m_vec = SVector::<f64, 6>::from_column_slice(mean);
        let mut c_mat = SMatrix::<f64, 6, 6>::zeros();
        for (row_index, covariance_row) in cov.iter().enumerate() {
            for (column_index, &covariance) in covariance_row.iter().enumerate() {
                *c_mat
                    .get_mut((row_index, column_index))
                    .ok_or(UkfPropagationFailure::InvalidInput)? = covariance;
            }
        }
        let sigmas = dust_ukf_rs::get_sigmas_ukf(&m_vec, &c_mat)
            .ok_or(UkfPropagationFailure::SigmaConstruction)?;
        let sigma_start = checked_ukf_product(component_idx, dust_ukf_rs::NUM_SIGMA)?;
        *scratch
            .component_sigma_offsets
            .get_mut(component_idx)
            .ok_or(UkfPropagationFailure::InvalidInput)? = sigma_start;
        for r in 0..dust_ukf_rs::NUM_SIGMA {
            let sigma_row = sigma_start
                .checked_add(r)
                .ok_or(UkfPropagationFailure::InvalidInput)?;
            let sigma_state_range = checked_ukf_row_range(sigma_row, scratch.sigma_states.len())?;
            let sigma_state = scratch
                .sigma_states
                .get_mut(sigma_state_range)
                .ok_or(UkfPropagationFailure::InvalidInput)?;
            for (axis, state_value) in sigma_state.iter_mut().enumerate() {
                *state_value = *sigmas
                    .get((r, axis))
                    .ok_or(UkfPropagationFailure::InvalidInput)?;
            }
        }
    }

    if scratch.sigma_equinoc.len() != sigma_storage_len {
        try_resize_ukf(&mut scratch.sigma_equinoc, sigma_storage_len, 0.0)?;
    }
    for (state_chunk, out_chunk) in scratch
        .sigma_states
        .chunks_exact(UKF_STATE_WIDTH)
        .zip(scratch.sigma_equinoc.chunks_exact_mut(UKF_STATE_WIDTH))
    {
        eci2equinoc_impl(state_chunk, 6, 0.0, 0.0, out_chunk);
    }

    if scratch.sigma_propagated.len() != sigma_storage_len {
        try_resize_ukf(&mut scratch.sigma_propagated, sigma_storage_len, 0.0)?;
    }
    if use_hf_batch {
        let ctx_ref = ctx.ok_or(UkfPropagationFailure::InvalidInput)?;
        let force_config = *ctx_ref
            .force_config
            .as_ref()
            .ok_or(UkfPropagationFailure::Propagation(
                TransferPropagationFailure::MissingHighFidelityAssets,
            ))?
            .as_ref();
        let _probe = lightyear_odeint_rs::probe::scope(lightyear_odeint_rs::probe::TAG_UKF_SIGMA);
        let t_eval = [tof_s];
        let request = lightyear_odeint_rs::BatchPropagationRequest {
            initial_equinoc_states: &scratch.sigma_equinoc,
            t_eval: &t_eval,
            t0_s: 0.0,
            t_final_s: tof_s,
            epoch_jd: ctx_ref.epoch_jd,
            force_config,
            ballistics: lightyear_odeint_rs::BatchBallistics::default(),
        };
        lightyear_odeint_rs::integrate_batch_native_into(request, &mut scratch.sigma_propagated)
            .map_err(|error| classify_native_batch_error(&error))?;
        for (equinoc_state, propagated_state) in scratch
            .sigma_equinoc
            .chunks_exact(UKF_STATE_WIDTH)
            .zip(scratch.sigma_propagated.chunks_exact_mut(UKF_STATE_WIDTH))
        {
            let mut baseline = [0.0; 6];
            equinoc_prop_from_impl(equinoc_state, tof_s, &mut baseline);
            for (propagated_value, baseline_value) in propagated_state.iter_mut().zip(baseline) {
                *propagated_value += baseline_value;
            }
        }
    } else {
        if scratch.sigma_tofs.len() == total_sigma {
            scratch.sigma_tofs.fill(tof_s);
        } else {
            try_resize_ukf(&mut scratch.sigma_tofs, total_sigma, tof_s)?;
        }
        let _ = ctx;
        equinoc_prop_j2_batch_impl(
            &scratch.sigma_equinoc,
            &scratch.sigma_tofs,
            &mut scratch.sigma_propagated,
        );
    }

    if scratch.sigma_propagated.len() != sigma_storage_len {
        return Err(UkfPropagationFailure::InvalidInput);
    }

    for (component_idx, sigma_start) in scratch.component_sigma_offsets.iter().copied().enumerate()
    {
        let mut mean_out: [f64; 6] = [0.0; 6];
        for sigma_idx in 0..dust_ukf_rs::NUM_SIGMA {
            let weight = *UKF_SIGMA_WEIGHTS
                .wm
                .get(sigma_idx)
                .ok_or(UkfPropagationFailure::InvalidInput)?;
            let sigma_row = sigma_start
                .checked_add(sigma_idx)
                .ok_or(UkfPropagationFailure::InvalidInput)?;
            let propagated_range =
                checked_ukf_row_range(sigma_row, scratch.sigma_propagated.len())?;
            let propagated_state = scratch
                .sigma_propagated
                .get(propagated_range)
                .ok_or(UkfPropagationFailure::InvalidInput)?;
            for (mean_value, propagated_value) in mean_out.iter_mut().zip(propagated_state) {
                if !propagated_value.is_finite() {
                    return Err(UkfPropagationFailure::NonFiniteOutput);
                }
                {
                    *mean_value += weight * *propagated_value;
                }
            }
        }
        if !mean_out.iter().all(|value| value.is_finite()) {
            return Err(UkfPropagationFailure::NonFiniteOutput);
        }
        *scratch
            .comp_means
            .get_mut(component_idx)
            .ok_or(UkfPropagationFailure::InvalidInput)? = mean_out;
    }

    Ok(total_sigma)
}

#[cfg(any(test, feature = "bench-internal"))]
pub(super) fn propagate_component_ukf_checked(
    mean: &[f64; 6],
    cov: &[[f64; 6]; 6],
    tof_s: f64,
    ctx: Option<&PlanContext>,
) -> Result<([f64; 6], [[f64; 6]; 6]), UkfPropagationFailure> {
    let m_vec = SVector::<f64, 6>::from_column_slice(mean);
    let mut c_mat = SMatrix::<f64, 6, 6>::zeros();
    for (row_index, covariance_row) in cov.iter().enumerate() {
        for (column_index, &covariance) in covariance_row.iter().enumerate() {
            *c_mat
                .get_mut((row_index, column_index))
                .ok_or(UkfPropagationFailure::InvalidInput)? = covariance;
        }
    }
    let sigmas = dust_ukf_rs::get_sigmas_ukf(&m_vec, &c_mat)
        .ok_or(UkfPropagationFailure::SigmaConstruction)?;
    let weights = UKF_SIGMA_WEIGHTS;

    let mut sigmas_prop = SMatrix::<f64, { dust_ukf_rs::NUM_SIGMA }, 6>::zeros();
    for r in 0..dust_ukf_rs::NUM_SIGMA {
        let mut s_eci = [0.0; 6];
        for (axis, state_value) in s_eci.iter_mut().enumerate() {
            *state_value = *sigmas
                .get((r, axis))
                .ok_or(UkfPropagationFailure::InvalidInput)?;
        }
        let out_state = if let Some(ctx_ref) = ctx {
            propagate_with_ctx_checked(&s_eci, tof_s, ctx_ref)
                .map_err(UkfPropagationFailure::Propagation)?
        } else {
            propagate_equinoctial(&s_eci, tof_s).ok_or(UkfPropagationFailure::InvalidInput)?
        };
        for (axis, state_value) in out_state.iter().enumerate() {
            *sigmas_prop
                .get_mut((r, axis))
                .ok_or(UkfPropagationFailure::InvalidInput)? = *state_value;
        }
    }

    let (mean_new, cov_new) =
        dust_ukf_rs::rebuild_mean_covar_ukf(&sigmas_prop, &weights.wm, &weights.wc);

    let mut mean_arr = [0.0; 6];
    mean_arr.copy_from_slice(mean_new.as_slice());
    let mut cov_arr = [[0.0; 6]; 6];
    for (row_index, covariance_row) in cov_arr.iter_mut().enumerate() {
        for (column_index, covariance) in covariance_row.iter_mut().enumerate() {
            *covariance = *cov_new
                .get((row_index, column_index))
                .ok_or(UkfPropagationFailure::InvalidInput)?;
        }
    }
    Ok((mean_arr, cov_arr))
}

fn classify_variable_final_error(
    error: lightyear_odeint_rs::session::VariableFinalNativeError,
) -> UkfPropagationFailure {
    use lightyear_odeint_rs::session::{VariableFinalNativeError, VariableFinalRowFailure};

    match &error {
        VariableFinalNativeError::Row {
            failure: VariableFinalRowFailure::Propagation(failure),
            ..
        } => UkfPropagationFailure::Propagation(TransferPropagationFailure::from(*failure)),
        VariableFinalNativeError::Row {
            failure: VariableFinalRowFailure::NonFiniteOutput,
            ..
        } => UkfPropagationFailure::Propagation(TransferPropagationFailure::NonFiniteOutput),
        VariableFinalNativeError::ArithmeticOverflow => {
            UkfPropagationFailure::Propagation(TransferPropagationFailure::ArithmeticOverflow)
        }
        VariableFinalNativeError::Ephemeris { row, source } => UkfPropagationFailure::Ephemeris {
            row: *row,
            message: source.to_string(),
            source: Arc::new(error),
        },
        VariableFinalNativeError::Row { .. }
        | VariableFinalNativeError::InputContract
        | VariableFinalNativeError::UnsupportedStepper(_)
        | VariableFinalNativeError::RayonConfig(_) => UkfPropagationFailure::NativeBatch {
            source: Some(Arc::new(error)),
        },
    }
}

pub(super) fn propagate_sigma_states_with_context(
    sigma_eci_states: &[f64],
    sigma_propagated: &mut [f64],
    tof_s: f64,
    ctx: &PlanContext,
) -> Result<(), UkfPropagationFailure> {
    if sigma_eci_states.len() != sigma_propagated.len() {
        return Err(UkfPropagationFailure::InvalidInput);
    }
    for (source, destination) in sigma_eci_states
        .chunks_exact(UKF_STATE_WIDTH)
        .zip(sigma_propagated.chunks_exact_mut(UKF_STATE_WIDTH))
    {
        let source: &[f64; UKF_STATE_WIDTH] = source
            .try_into()
            .map_err(|_| UkfPropagationFailure::InvalidInput)?;
        let propagated = propagate_with_ctx_checked(source, tof_s, ctx)
            .map_err(UkfPropagationFailure::Propagation)?;
        destination.copy_from_slice(&propagated);
    }
    Ok(())
}

#[cfg(feature = "solver-qualification")]
pub(super) fn propagate_sigma_states_with_fresh_observed_context(
    trace: &mut QualificationLegTrace,
    sigma_eci_states: &[f64],
    sigma_propagated: &mut [f64],
    tof_s: f64,
    ctx: &PlanContext,
    body_force: crate::types::BodyForceConfig,
) -> Result<(), UkfPropagationFailure> {
    if sigma_eci_states.len() != sigma_propagated.len() {
        return Err(UkfPropagationFailure::InvalidInput);
    }
    for (row, (initial_eci, destination)) in sigma_eci_states
        .chunks_exact(UKF_STATE_WIDTH)
        .zip(sigma_propagated.chunks_exact_mut(UKF_STATE_WIDTH))
        .enumerate()
    {
        // The batch is whole, so the enumeration index IS the global sigma-row
        // ordinal. It was offset by a `first_sigma_ordinal` while the R18
        // sigma-row-0 endpoint reuse could start a batch at row 1.
        let Ok(component) = u8::try_from(row / dust_ukf_rs::NUM_SIGMA) else {
            trace.mark_incomplete(QualificationTraceError::RecordLimit);
            return Err(UkfPropagationFailure::Qualification(
                QualificationTraceError::RecordLimit,
            ));
        };
        let Ok(sigma) = u8::try_from(row % dust_ukf_rs::NUM_SIGMA) else {
            trace.mark_incomplete(QualificationTraceError::RecordLimit);
            return Err(UkfPropagationFailure::Qualification(
                QualificationTraceError::RecordLimit,
            ));
        };
        let mut initial_state = [0.0; UKF_STATE_WIDTH];
        initial_state.copy_from_slice(initial_eci);
        let mut initial_equinoctial = [0.0; UKF_STATE_WIDTH];
        eci2equinoc_impl_f64(
            &initial_state,
            UKF_STATE_WIDTH,
            0.0,
            0.0,
            &mut initial_equinoctial,
        );
        let observed = match propagate_high_fidelity_state_at_epoch_checked_observed(
            &initial_equinoctial,
            tof_s,
            ctx.epoch_jd,
            body_force,
            ctx,
        ) {
            Ok(observed) => observed,
            Err(error) => {
                trace.mark_incomplete(QualificationTraceError::IncompleteMetrics);
                return Err(UkfPropagationFailure::Propagation(error));
            }
        };
        let outcome = observed.outcome;
        trace.record_observed_transfer(
            QualificationLegInput::new(
                LegPath::UkfSigma { component, sigma },
                crate::types::BodyRole::Dust,
                ctx.epoch_jd,
                0.0,
                tof_s,
                initial_state,
            ),
            outcome.clone(),
            observed.scalar_observation,
        );
        let propagated = outcome.map_err(UkfPropagationFailure::Propagation)?;
        destination.copy_from_slice(&propagated);
    }
    Ok(())
}

pub(super) fn propagate_sigma_states_with_native_batch(
    sigma_eci_states: &[f64],
    sigma_propagated: &mut [f64],
    total_sigma: usize,
    tof_s: f64,
    ctx: &PlanContext,
) -> Result<(), UkfPropagationFailure> {
    let (Some(force_config), Some(packed_coeffs)) =
        (ctx.force_config.as_ref(), ctx.packed_coeffs.clone())
    else {
        if ctx.execution_policy.require_high_fidelity {
            return Err(UkfPropagationFailure::Propagation(
                TransferPropagationFailure::MissingHighFidelityAssets,
            ));
        }
        return propagate_sigma_states_with_context(sigma_eci_states, sigma_propagated, tof_s, ctx);
    };
    let force_config = **force_config;
    let arc_end_jd = ctx.epoch_jd + tof_s / satpy_core::SEC_PER_DAY;
    // Bind the builder result: `with_ephemeris_for_arc` consumes `self` and
    // returns the arc-bound config. The error path always fired (`?`), but the
    // Ok value used to be discarded, so the session below was constructed from
    // the unbound copy and only stayed correct because the per-row kernel
    // config re-derives its dynamic ephemeris flags itself
    // (`config_for_jd_mid`). Binding here makes the invariant explicit instead
    // of implicit, and is bit-exact either way: dynamic bodies keep their
    // dynamic flags (positions are read from tables at propagation time), and
    // static caller positions are never touched.
    let force_config = force_config
        .with_ephemeris_for_arc(ctx.epoch_jd, arc_end_jd)
        .map_err(|error| {
            UkfPropagationFailure::Propagation(TransferPropagationFailure::Ephemeris(error))
        })?;
    let force_config = Arc::new(force_config);
    let gravity = lightyear_odeint_rs::ScalarGravityAssets::new(packed_coeffs);
    let propagation_context =
        lightyear_odeint_rs::ScalarPropagationContext::new(ctx.epoch_jd, force_config, gravity);
    let session = lightyear_odeint_rs::LightyearSession::from_context(propagation_context);
    let jd0_arr = try_ukf_filled_vec(total_sigma, ctx.epoch_jd)?;
    let tf_s_arr = try_ukf_filled_vec(total_sigma, tof_s)?;
    let request = lightyear_odeint_rs::session::VariableFinalBatchRequest {
        initial_eci_states: sigma_eci_states,
        epoch_jd: &jd0_arr,
        final_time_s: &tf_s_arr,
        t0_s: 0.0,
        ballistics: lightyear_odeint_rs::session::VariableFinalBallistics::default(),
    };
    if let Err(error) = session.integrate_variable_final_into(request, sigma_propagated) {
        let error = classify_variable_final_error(error);
        if ctx.execution_policy.require_high_fidelity {
            return Err(error);
        }
        propagate_sigma_states_with_context(sigma_eci_states, sigma_propagated, tof_s, ctx)?;
    }
    Ok(())
}

/// Propagate every component through one native batch, matching Python's
/// authoritative many-cloud sigma lane for diagnostic parity.
///
/// # Errors
///
/// Returns a typed failure for invalid UKF inputs, unavailable propagation
/// authority, or an allocation that cannot retain the complete result.
pub fn propagate_components_ukf_full_batch(
    means: &[[f64; 6]],
    covs: &[[[f64; 6]; 6]],
    sigma_points_override: Option<&[[f64; 6]]>,
    tof_s: f64,
    ctx: Option<&PlanContext>,
) -> Result<UkfFullBatchOutput, UkfPropagationFailure> {
    propagate_components_ukf_batch_core::<_, true>(
        means,
        covs,
        sigma_points_override,
        tof_s,
        ctx,
        &mut UnobservedPostprocessLeg,
    )
}

/// Internal observed entry into the one full-batch UKF arithmetic core.
///
/// Unit tests retain propagated points because the test-only distribution
/// exposes them. Production callers discard them without materializing a
/// second vector; diagnostics request the full output through the public entry.
pub(super) fn propagate_components_ukf_full_batch_observed_by<O: PostprocessLegObserver>(
    means: &[[f64; 6]],
    covs: &[[[f64; 6]; 6]],
    sigma_points_override: Option<&[[f64; 6]]>,
    tof_s: f64,
    ctx: Option<&PlanContext>,
    observer: &mut O,
) -> Result<UkfFullBatchOutput, UkfPropagationFailure> {
    propagate_components_ukf_batch_core::<
        O,
        { PRODUCTION_RETAIN_PROPAGATED_SIGMA_POINTS || cfg!(test) },
    >(means, covs, sigma_points_override, tof_s, ctx, observer)
}

fn propagate_components_ukf_batch_core<
    O: PostprocessLegObserver,
    const RETAIN_PROPAGATED_SIGMA_POINTS: bool,
>(
    means: &[[f64; 6]],
    covs: &[[[f64; 6]; 6]],
    sigma_points_override: Option<&[[f64; 6]]>,
    tof_s: f64,
    ctx: Option<&PlanContext>,
    observer: &mut O,
) -> Result<UkfFullBatchOutput, UkfPropagationFailure> {
    if means.len() != covs.len() || !tof_s.is_finite() {
        return Err(UkfPropagationFailure::InvalidInput);
    }
    if means.is_empty() {
        return Ok(UkfFullBatchOutput {
            propagated_components: Vec::new(),
            propagated_sigma_points: Vec::new(),
        });
    }

    let component_count = means.len();
    let total_sigma = checked_ukf_product(component_count, dust_ukf_rs::NUM_SIGMA)?;
    observer.preflight_leg_capacity(total_sigma)?;
    let sigma_storage_len = checked_ukf_product(total_sigma, UKF_STATE_WIDTH)?;
    let mut sigma_eci_states = try_ukf_filled_vec(sigma_storage_len, 0.0)?;
    if let Some(sigma_points) = sigma_points_override {
        if sigma_points.len() != total_sigma {
            return Err(UkfPropagationFailure::InvalidInput);
        }
        for (sigma, destination) in sigma_points
            .iter()
            .zip(sigma_eci_states.chunks_exact_mut(UKF_STATE_WIDTH))
        {
            destination.copy_from_slice(sigma);
        }
    } else {
        for (component_idx, (mean, cov)) in means.iter().zip(covs.iter()).enumerate() {
            let mean_vec = SVector::<f64, 6>::from_column_slice(mean);
            let mut cov_mat = SMatrix::<f64, 6, 6>::zeros();
            for (row_index, covariance_row) in cov.iter().enumerate() {
                for (column_index, &covariance) in covariance_row.iter().enumerate() {
                    *cov_mat
                        .get_mut((row_index, column_index))
                        .ok_or(UkfPropagationFailure::InvalidInput)? = covariance;
                }
            }
            let sigmas = dust_ukf_rs::get_sigmas_ukf(&mean_vec, &cov_mat)
                .ok_or(UkfPropagationFailure::SigmaConstruction)?;
            for sigma_idx in 0..dust_ukf_rs::NUM_SIGMA {
                let sigma_row = checked_ukf_product(component_idx, dust_ukf_rs::NUM_SIGMA)?
                    .checked_add(sigma_idx)
                    .ok_or(UkfPropagationFailure::InvalidInput)?;
                let sigma_state_range = checked_ukf_row_range(sigma_row, sigma_eci_states.len())?;
                let sigma_eci = sigma_eci_states
                    .get_mut(sigma_state_range)
                    .ok_or(UkfPropagationFailure::InvalidInput)?;
                for (axis, state_value) in sigma_eci.iter_mut().enumerate() {
                    *state_value = *sigmas
                        .get((sigma_idx, axis))
                        .ok_or(UkfPropagationFailure::InvalidInput)?;
                }
            }
        }
    }

    let mut sigma_propagated = try_ukf_filled_vec(sigma_storage_len, 0.0)?;
    let _probe = lightyear_odeint_rs::probe::scope(lightyear_odeint_rs::probe::TAG_UKF_SIGMA_PC);
    if let Some(ctx_ref) = ctx {
        // Every sigma row is propagated. The R18 bit-guarded sigma-row-0
        // endpoint reuse lived here and was REMOVED when the julier7 simplex
        // landed: it fired only when row 0 equalled the release mean bit for
        // bit, and the simplex has no centre point — all NUM_SIGMA rows sit at
        // whitened radius sqrt(6), so no row can ever equal the mean and the
        // guard could not fire again. Its 0.755% is subsumed by the point count
        // it was a workaround for. See `dust_ukf_rs::NUM_SIGMA`.
        observer.propagate_ukf_sigma_states(
            &sigma_eci_states,
            &mut sigma_propagated,
            total_sigma,
            tof_s,
            ctx_ref,
        )?;
    } else {
        // Equinoctial sigma states are built HERE, next to their only consumer.
        // They used to be built unconditionally alongside `sigma_eci_states`,
        // so the strict-HF branch above paid `NUM_SIGMA` per component
        // conversions on every call and then propagated from the ECI states
        // instead, discarding every one of them.
        let mut sigma_equinoc = try_ukf_filled_vec(sigma_storage_len, 0.0)?;
        for (sigma_eci, sigma_equinoctial) in sigma_eci_states
            .chunks_exact(UKF_STATE_WIDTH)
            .zip(sigma_equinoc.chunks_exact_mut(UKF_STATE_WIDTH))
        {
            eci2equinoc_impl(sigma_eci, 6, 0.0, 0.0, sigma_equinoctial);
        }
        let sigma_tofs = try_ukf_filled_vec(total_sigma, tof_s)?;
        equinoc_prop_j2_batch_impl(&sigma_equinoc, &sigma_tofs, &mut sigma_propagated);
    }

    rebuild_ukf_output::<RETAIN_PROPAGATED_SIGMA_POINTS>(
        component_count,
        total_sigma,
        &sigma_propagated,
    )
}

fn rebuild_ukf_output<const RETAIN_PROPAGATED_SIGMA_POINTS: bool>(
    component_count: usize,
    total_sigma: usize,
    sigma_propagated: &[f64],
) -> Result<UkfFullBatchOutput, UkfPropagationFailure> {
    let weights = UKF_SIGMA_WEIGHTS;
    let mut out = try_ukf_vec_with_capacity(component_count)?;
    for component_idx in 0..component_count {
        let mut propagated = SMatrix::<f64, { dust_ukf_rs::NUM_SIGMA }, 6>::zeros();
        for sigma_idx in 0..dust_ukf_rs::NUM_SIGMA {
            let sigma_row = checked_ukf_product(component_idx, dust_ukf_rs::NUM_SIGMA)?
                .checked_add(sigma_idx)
                .ok_or(UkfPropagationFailure::InvalidInput)?;
            let propagated_range = checked_ukf_row_range(sigma_row, sigma_propagated.len())?;
            let propagated_state = sigma_propagated
                .get(propagated_range)
                .ok_or(UkfPropagationFailure::InvalidInput)?;
            for (axis, &value) in propagated_state.iter().enumerate() {
                if !value.is_finite() {
                    return Err(UkfPropagationFailure::NonFiniteOutput);
                }
                *propagated
                    .get_mut((sigma_idx, axis))
                    .ok_or(UkfPropagationFailure::InvalidInput)? = value;
            }
        }
        let (mean_new, cov_new) =
            dust_ukf_rs::rebuild_mean_covar_ukf(&propagated, &weights.wm, &weights.wc);
        let mut mean_out = [0.0; 6];
        mean_out.copy_from_slice(mean_new.as_slice());
        let mut cov_out = [[0.0; 6]; 6];
        for (row_index, covariance_row) in cov_out.iter_mut().enumerate() {
            for (column_index, covariance) in covariance_row.iter_mut().enumerate() {
                *covariance = *cov_new
                    .get((row_index, column_index))
                    .ok_or(UkfPropagationFailure::InvalidInput)?;
            }
        }
        out.push((mean_out, cov_out));
    }
    let propagated_sigma_points = if RETAIN_PROPAGATED_SIGMA_POINTS {
        #[cfg(test)]
        SIGMA_MATERIALIZATION_COUNT.set(SIGMA_MATERIALIZATION_COUNT.get().saturating_add(1));

        let mut points = try_ukf_vec_with_capacity(total_sigma)?;
        for row in sigma_propagated.chunks_exact(UKF_STATE_WIDTH) {
            let point = row
                .try_into()
                .map_err(|_| UkfPropagationFailure::InvalidInput)?;
            points.push(point);
        }
        points
    } else {
        Vec::new()
    };
    Ok(UkfFullBatchOutput {
        propagated_components: out,
        propagated_sigma_points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ExecutionPolicy, TransferRequest};

    #[cfg(feature = "solver-qualification")]
    use crate::postprocess::{
        QualificationArmIdentity, QualificationLegRecord, QualificationTraceIdentity,
    };
    #[cfg(feature = "solver-qualification")]
    use core::mem::MaybeUninit;

    fn context(use_high_fidelity: bool, require_high_fidelity: bool) -> PlanContext {
        let mut request =
            TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        request.execution_policy = ExecutionPolicy {
            use_high_fidelity,
            require_high_fidelity,
            ..ExecutionPolicy::default()
        };
        PlanContext::from_request(request)
    }

    #[cfg(feature = "solver-qualification")]
    fn qualification_trace(
        records: &mut [MaybeUninit<QualificationLegRecord>],
    ) -> Result<QualificationLegTrace<'_>, QualificationTraceError> {
        let identity = QualificationTraceIdentity {
            event_ordinal: 0,
            family_ordinal: 0,
            candidate_ordinal: 0,
            fraction_ordinal: 0,
            arm: QualificationArmIdentity::try_new([0x01; 32])?,
        };
        QualificationLegTrace::try_new(identity, records)
    }

    #[cfg(feature = "solver-qualification")]
    #[test]
    fn observed_full_batch_rejects_over_cap_before_numerical_work() {
        let component_count = MAX_QUALIFICATION_LEG_RECORDS
            .checked_div(dust_ukf_rs::NUM_SIGMA)
            .and_then(|count| count.checked_add(1))
            .expect("fixed qualification cap supports one over-cap UKF batch");
        let means = vec![[7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0]; component_count];
        let mut covariance = [[0.0; 6]; 6];
        for (axis, row) in covariance.iter_mut().enumerate() {
            let diagonal = row
                .get_mut(axis)
                .expect("fixed covariance diagonal is in bounds");
            *diagonal = 1.0e-6;
        }
        let covariances = vec![covariance; component_count];
        let mut records: [MaybeUninit<QualificationLegRecord>; MAX_QUALIFICATION_LEG_RECORDS] =
            std::array::from_fn(|_| MaybeUninit::uninit());
        let mut trace =
            qualification_trace(&mut records).expect("fixed qualification trace storage is valid");

        // With no context, the canonical non-observed route would enter its
        // analytical propagation branch. The typed capacity result proves the
        // trace-present route exits before sigma construction or that fallback.
        let outcome = propagate_components_ukf_full_batch_observed_by(
            &means,
            &covariances,
            None,
            60.0,
            None,
            &mut trace,
        );
        assert_eq!(
            outcome,
            Err(UkfPropagationFailure::Qualification(
                QualificationTraceError::RecordLimit,
            ))
        );
        assert!(trace.records().is_empty());
        assert_eq!(
            trace.validate_complete(),
            Err(QualificationTraceError::RecordLimit)
        );
    }

    #[cfg(feature = "solver-qualification")]
    #[test]
    fn qualification_observed_ukf_avoids_reusable_variable_final_batch() {
        let source = include_str!("ukf.rs");
        let production = source
            .split_once("\n#[cfg(test)]")
            .map(|(production, _)| production)
            .expect("UKF source must retain its test boundary");
        let reusable_observer = concat!("integrate_variable_final_", "observed_into");
        let fresh_scalar = concat!(
            "propagate_high_fidelity_state_at_epoch_",
            "checked_observed"
        );

        assert!(
            !production.contains(reusable_observer),
            "qualification observed UKF must not enter the reusable variable-final batch"
        );
        assert!(
            production.contains(fresh_scalar),
            "qualification observed UKF must use the fresh scalar observed path"
        );
    }

    #[cfg(feature = "solver-qualification")]
    #[test]
    fn fresh_observed_sigma_matches_reusable_one_shot_after_identical_reset() {
        let epoch_jd = 2_460_000.5;
        let tof_s = 60.0;
        // Exercise the exact production f64 Cartesian-to-equinoctial kernel.
        // A circular, axis-aligned state hides operation-order divergence from
        // the generic conversion path.
        let initial_eci = [7_000.0, 137.0, -83.0, -0.217, 7.431, 0.913];
        let c = std::sync::Arc::new(vec![1.0, 0.0, 0.0, 0.0]);
        let s = std::sync::Arc::new(vec![0.0; 4]);
        let packed = std::sync::Arc::new(
            satpy_core::pack_gravity_coeffs(&c, &s, 2, 0)
                .expect("zero-order test gravity coefficients are valid"),
        );
        let force_config = std::sync::Arc::new(lightyear_odeint_rs::types::ForceConfig {
            sph_order: 0,
            ..lightyear_odeint_rs::types::ForceConfig::default()
        });
        let mut request =
            TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        request.epoch_jd = epoch_jd;
        request.execution_policy = ExecutionPolicy {
            use_high_fidelity: true,
            require_high_fidelity: true,
            ..ExecutionPolicy::default()
        };
        request.force_config = Some(std::sync::Arc::clone(&force_config));
        request.packed_coeffs = Some(std::sync::Arc::clone(&packed));
        let ctx = PlanContext::from_request(request);

        let reusable_context = lightyear_odeint_rs::ScalarPropagationContext::new(
            epoch_jd,
            std::sync::Arc::clone(&force_config),
            lightyear_odeint_rs::ScalarGravityAssets::new(std::sync::Arc::clone(&packed)),
        );
        let reusable_session =
            lightyear_odeint_rs::LightyearSession::from_context(reusable_context);
        let epochs = [epoch_jd];
        let final_times = [tof_s];
        let mut reusable_output = [f64::NAN; UKF_STATE_WIDTH];
        let mut reusable_observations = [None];
        reusable_session
            .integrate_variable_final_observed_into(
                lightyear_odeint_rs::session::VariableFinalBatchRequest {
                    initial_eci_states: &initial_eci,
                    epoch_jd: &epochs,
                    final_time_s: &final_times,
                    t0_s: 0.0,
                    ballistics: lightyear_odeint_rs::session::VariableFinalBallistics::default(),
                },
                &mut reusable_output,
                &mut reusable_observations,
            )
            .expect("one reusable scalar row must propagate");
        let reusable = reusable_observations[0]
            .take()
            .expect("reusable scalar row must write its observation");
        let reusable_endpoint = reusable
            .outcome
            .expect("reusable scalar row must return an endpoint");

        let mut records: [MaybeUninit<QualificationLegRecord>; MAX_QUALIFICATION_LEG_RECORDS] =
            std::array::from_fn(|_| MaybeUninit::uninit());
        let mut trace =
            qualification_trace(&mut records).expect("fixed qualification trace storage is valid");
        let mut fresh_output = [f64::NAN; UKF_STATE_WIDTH];
        let body_force = crate::types::BodyForceConfig::high_fidelity(
            crate::types::BodyRole::Dust,
            force_config.am_ratio,
            force_config.cd,
            force_config.cr,
        );
        propagate_sigma_states_with_fresh_observed_context(
            &mut trace,
            &initial_eci,
            &mut fresh_output,
            tof_s,
            &ctx,
            body_force,
        )
        .expect("fresh scalar row must propagate");

        assert_eq!(
            fresh_output.map(f64::to_bits),
            reusable_endpoint.map(f64::to_bits)
        );
        assert_eq!(
            reusable_output.map(f64::to_bits),
            reusable_endpoint.map(f64::to_bits)
        );
        let record = trace.records().first().expect("fresh row must be recorded");
        assert_eq!(record.metrics, reusable.metrics);
        assert_eq!(record.terminal_status, reusable.terminal_status);
    }

    #[test]
    fn component_mean_batch_rejects_mismatched_component_counts() {
        let means = [[7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0]];
        let covariances: [[[f64; 6]; 6]; 0] = [];
        let mut scratch = TransferPostprocessScratch::default();

        assert_eq!(
            propagate_component_means_ukf_batch(&means, &covariances, 60.0, None, &mut scratch,),
            Err(UkfPropagationFailure::InvalidInput)
        );
        assert!(scratch.comp_means.is_empty());
        assert!(scratch.sigma_states.is_empty());
    }

    #[test]
    fn checked_ukf_storage_size_rejects_overflow() {
        assert_eq!(
            checked_ukf_product(usize::MAX, UKF_STATE_WIDTH),
            Err(UkfPropagationFailure::InvalidInput)
        );
    }

    #[test]
    fn ukf_allocation_failure_is_typed_without_mutating_the_destination() {
        let mut values = vec![7_u8];
        let capacity_before = values.capacity();

        assert_eq!(
            try_reserve_ukf(&mut values, usize::MAX),
            Err(UkfPropagationFailure::Allocation)
        );
        assert_eq!(values, vec![7_u8]);
        assert_eq!(values.capacity(), capacity_before);
    }

    #[test]
    fn ukf_vec_resize_allows_shrinking_reused_scratch() {
        let mut values = vec![7_u8, 9];
        let capacity_before = values.capacity();

        assert_eq!(try_resize_ukf(&mut values, 1, 0), Ok(()));
        assert_eq!(values, vec![7_u8]);
        assert_eq!(values.capacity(), capacity_before);
    }

    #[test]
    fn production_reconstruction_skips_sigma_materialization_bit_exactly() {
        let total_sigma = dust_ukf_rs::NUM_SIGMA;
        let sigma_propagated: Vec<f64> = (0_u32..)
            .take(total_sigma)
            .flat_map(|sigma| {
                let sigma = f64::from(sigma);
                [
                    7_000.0 + sigma,
                    100.0 - sigma,
                    -25.0 + sigma * 0.5,
                    -0.1 + sigma * 0.01,
                    7.4 - sigma * 0.001,
                    0.8 + sigma * 0.002,
                ]
            })
            .collect();

        reset_sigma_materialization_count();
        let production = rebuild_ukf_output::<PRODUCTION_RETAIN_PROPAGATED_SIGMA_POINTS>(
            1,
            total_sigma,
            &sigma_propagated,
        )
        .expect("finite production reconstruction must succeed");
        assert_eq!(sigma_materialization_count(), 0);
        assert!(production.propagated_sigma_points.is_empty());

        let diagnostic = rebuild_ukf_output::<true>(1, total_sigma, &sigma_propagated)
            .expect("finite diagnostic reconstruction must succeed");
        assert_eq!(sigma_materialization_count(), 1);
        assert_eq!(
            production
                .propagated_components
                .iter()
                .map(|(mean, covariance)| {
                    (
                        mean.map(f64::to_bits),
                        covariance.map(|row| row.map(f64::to_bits)),
                    )
                })
                .collect::<Vec<_>>(),
            diagnostic
                .propagated_components
                .iter()
                .map(|(mean, covariance)| {
                    (
                        mean.map(f64::to_bits),
                        covariance.map(|row| row.map(f64::to_bits)),
                    )
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(diagnostic.propagated_sigma_points.len(), total_sigma);

        let mut nonfinite = sigma_propagated;
        assert!(!nonfinite.is_empty());
        if let Some(first) = nonfinite.first_mut() {
            *first = f64::NAN;
        }
        assert_eq!(
            rebuild_ukf_output::<PRODUCTION_RETAIN_PROPAGATED_SIGMA_POINTS>(
                1,
                total_sigma,
                &nonfinite,
            ),
            rebuild_ukf_output::<true>(1, total_sigma, &nonfinite)
        );
    }

    #[test]
    fn ukf_smallvec_resize_allows_shrinking_reused_scratch() {
        let mut values: smallvec::SmallVec<[u8; 2]> = smallvec::SmallVec::new();
        values.extend_from_slice(&[7, 9]);
        let capacity_before = values.capacity();

        assert_eq!(try_resize_smallvec_ukf(&mut values, 1, 0), Ok(()));
        assert_eq!(values.as_slice(), &[7_u8]);
        assert_eq!(values.capacity(), capacity_before);
    }

    #[test]
    fn component_mean_batch_reuses_scratch_for_fewer_components() {
        let means = [
            [7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            [7_010.0, 0.0, 0.0, 0.0, 7.5, 0.0],
        ];
        let mut covariance = [[0.0; 6]; 6];
        for (axis, row) in covariance.iter_mut().enumerate() {
            *row.get_mut(axis)
                .expect("fixed covariance diagonal is addressable") = 1.0e-4;
        }
        let covariances = [covariance; 2];
        let mut scratch = TransferPostprocessScratch::default();

        assert_eq!(
            propagate_component_means_ukf_batch(&means, &covariances, 60.0, None, &mut scratch),
            Ok(2 * dust_ukf_rs::NUM_SIGMA)
        );
        let capacities_before = (
            scratch.comp_means.capacity(),
            scratch.sigma_states.capacity(),
            scratch.component_sigma_offsets.capacity(),
            scratch.sigma_equinoc.capacity(),
            scratch.sigma_propagated.capacity(),
            scratch.sigma_tofs.capacity(),
        );

        assert_eq!(
            propagate_component_means_ukf_batch(
                &means[..1],
                &covariances[..1],
                60.0,
                None,
                &mut scratch,
            ),
            Ok(dust_ukf_rs::NUM_SIGMA)
        );
        assert_eq!(scratch.comp_means.len(), 1);
        assert_eq!(
            scratch.sigma_states.len(),
            dust_ukf_rs::NUM_SIGMA * UKF_STATE_WIDTH
        );
        assert_eq!(scratch.component_sigma_offsets.len(), 1);
        assert_eq!(
            scratch.sigma_equinoc.len(),
            dust_ukf_rs::NUM_SIGMA * UKF_STATE_WIDTH
        );
        assert_eq!(
            scratch.sigma_propagated.len(),
            dust_ukf_rs::NUM_SIGMA * UKF_STATE_WIDTH
        );
        assert_eq!(scratch.sigma_tofs.len(), dust_ukf_rs::NUM_SIGMA);
        assert_eq!(
            (
                scratch.comp_means.capacity(),
                scratch.sigma_states.capacity(),
                scratch.component_sigma_offsets.capacity(),
                scratch.sigma_equinoc.capacity(),
                scratch.sigma_propagated.capacity(),
                scratch.sigma_tofs.capacity(),
            ),
            capacities_before
        );
    }

    #[test]
    fn ukf_scratch_preflight_failure_leaves_all_lengths_unchanged() {
        let mut scratch = TransferPostprocessScratch::default();
        scratch.comp_means.push([1.0; UKF_STATE_WIDTH]);
        scratch.sigma_states.push(2.0);
        scratch.sigma_equinoc.push(3.0);
        scratch.sigma_propagated.push(4.0);
        scratch.sigma_tofs.push(5.0);
        scratch.component_sigma_offsets.push(6);
        let lengths_before = (
            scratch.comp_means.len(),
            scratch.sigma_states.len(),
            scratch.sigma_equinoc.len(),
            scratch.sigma_propagated.len(),
            scratch.sigma_tofs.len(),
            scratch.component_sigma_offsets.len(),
        );

        assert_eq!(
            preflight_component_mean_scratch(&mut scratch, usize::MAX, UKF_STATE_WIDTH, 1, true,),
            Err(UkfPropagationFailure::Allocation)
        );
        assert_eq!(
            (
                scratch.comp_means.len(),
                scratch.sigma_states.len(),
                scratch.sigma_equinoc.len(),
                scratch.sigma_propagated.len(),
                scratch.sigma_tofs.len(),
                scratch.component_sigma_offsets.len(),
            ),
            lengths_before
        );
        assert_eq!(scratch.comp_means.as_slice(), &[[1.0; UKF_STATE_WIDTH]]);
        assert_eq!(scratch.sigma_states, [2.0]);
        assert_eq!(scratch.sigma_equinoc, [3.0]);
        assert_eq!(scratch.sigma_propagated, [4.0]);
        assert_eq!(scratch.sigma_tofs, [5.0]);
        assert_eq!(scratch.component_sigma_offsets.as_slice(), &[6]);
    }

    #[test]
    fn variable_final_ephemeris_error_retains_typed_source() {
        use lightyear_odeint_rs::precomputed_ephem::EphemerisCoverageError;
        use lightyear_odeint_rs::session::VariableFinalNativeError;

        let ephemeris_error = EphemerisCoverageError::NonFiniteArc {
            jd_a: f64::NAN,
            jd_b: 2_460_000.5,
        };
        let failure = classify_variable_final_error(VariableFinalNativeError::Ephemeris {
            row: 7,
            source: anyhow::Error::new(ephemeris_error.clone()),
        });

        assert_eq!(
            failure.to_string(),
            "UKF ephemeris row 7: dynamic ephemeris arc endpoints must be finite (jd_a=NaN, jd_b=2460000.5)"
        );
        let equivalent = classify_variable_final_error(VariableFinalNativeError::Ephemeris {
            row: 7,
            source: anyhow::Error::new(ephemeris_error),
        });
        assert_eq!(failure, equivalent);
        assert!(matches!(
            failure,
            UkfPropagationFailure::Ephemeris { row: 7, .. }
        ));
        let variable_final_source = std::error::Error::source(&failure)
            .expect("UKF failure must retain variable-final source");
        let variable_final_source = variable_final_source
            .downcast_ref::<VariableFinalNativeError>()
            .expect("UKF source must remain VariableFinalNativeError");
        assert!(matches!(
            variable_final_source,
            VariableFinalNativeError::Ephemeris { row: 7, .. }
        ));
        let source = std::error::Error::source(variable_final_source)
            .expect("variable-final ephemeris failure must retain typed cause");
        assert!(matches!(
            source.downcast_ref::<EphemerisCoverageError>(),
            Some(EphemerisCoverageError::NonFiniteArc { jd_a, jd_b })
                if jd_a.is_nan() && jd_b.to_bits() == 2_460_000.5_f64.to_bits()
        ));
    }

    #[test]
    fn variable_final_native_batch_retains_source_without_changing_classification() {
        use lightyear_odeint_rs::session::VariableFinalNativeError;

        let failure = classify_variable_final_error(VariableFinalNativeError::UnsupportedStepper(
            anyhow::anyhow!("fixture unsupported stepper"),
        ));
        let equivalent =
            classify_variable_final_error(VariableFinalNativeError::UnsupportedStepper(
                anyhow::anyhow!("different diagnostic, same classification"),
            ));

        assert_eq!(failure.to_string(), "UKF native batch failed");
        assert_eq!(failure, equivalent);
        assert!(matches!(
            failure,
            UkfPropagationFailure::NativeBatch { source: Some(_) }
        ));
        let source = std::error::Error::source(&failure)
            .expect("native-batch failure must retain variable-final source");
        assert!(matches!(
            source.downcast_ref::<VariableFinalNativeError>(),
            Some(VariableFinalNativeError::UnsupportedStepper(_))
        ));
    }

    #[test]
    fn variable_final_arithmetic_overflow_stays_as_transfer_overflow() {
        use lightyear_odeint_rs::session::VariableFinalNativeError;

        assert_eq!(
            classify_variable_final_error(VariableFinalNativeError::ArithmeticOverflow),
            UkfPropagationFailure::Propagation(TransferPropagationFailure::ArithmeticOverflow)
        );
    }

    #[test]
    fn final_census_errors_stay_as_transfer_census_errors() {
        use lightyear_odeint_rs::integrator::FinalPropagationFailure;
        use lightyear_odeint_rs::probe::PropagationCensusError;
        use lightyear_odeint_rs::session::{VariableFinalNativeError, VariableFinalRowFailure};

        let expected = UkfPropagationFailure::Propagation(TransferPropagationFailure::Census(
            PropagationCensusError::Allocation,
        ));
        let native_error = anyhow::Error::new(FinalPropagationFailure::Census(
            PropagationCensusError::Allocation,
        ));
        assert_eq!(classify_native_batch_error(&native_error), expected);

        let row_error = VariableFinalNativeError::Row {
            row: 3,
            failure: VariableFinalRowFailure::Propagation(FinalPropagationFailure::Census(
                PropagationCensusError::Allocation,
            )),
        };
        assert_eq!(classify_variable_final_error(row_error), expected);
    }

    #[test]
    fn full_batch_exposes_strict_failure_and_generic_recovery() {
        let means = [[7_000.0, 0.0, 0.0, 0.0, 7.5, 1.0]];
        let mut covariance = [[0.0; 6]; 6];
        for (axis, row) in covariance.iter_mut().enumerate() {
            *row.get_mut(axis)
                .expect("fixed covariance diagonal is addressable") = 1e-6;
        }
        let covariances = [covariance];

        assert_eq!(
            propagate_components_ukf_full_batch(
                &means,
                &covariances,
                None,
                10.0,
                Some(&context(false, true)),
            ),
            Err(UkfPropagationFailure::Propagation(
                TransferPropagationFailure::MissingHighFidelityAssets,
            ))
        );
        let output = propagate_components_ukf_full_batch(
            &means,
            &covariances,
            None,
            10.0,
            Some(&context(false, false)),
        )
        .expect("non-strict diagnostic UKF batch propagates");
        assert_eq!(output.propagated_components.len(), means.len());
        assert_eq!(
            output.propagated_sigma_points.len(),
            means.len() * dust_ukf_rs::NUM_SIGMA
        );
    }

    /// The compiled sigma set must be the one the sealed authority names.
    ///
    /// The predecessor of this test pinned the module's own Merwe tuning
    /// constants against the same literals written a second time, so moving the
    /// sealed Part A value left it green while every read site in this crate
    /// silently measured 1.0/2.0/0.0. The tuning triple is gone -- the simplex
    /// has no tuning -- and what replaces it is a token: `dust_ukf_rs` names the
    /// set it compiles, the sealed authority names the set the campaign flew,
    /// and they must agree. `nd_pipeline` binds the two at COMPILE time with a
    /// `const` assertion; this test is the runtime half, so a `cargo test` of
    /// this crate alone still says which set it is measuring.
    #[test]
    fn compiled_sigma_set_is_the_sealed_one() {
        let sealed = nd_config::CompiledPartAScienceV1::part_a_v1().native_hybrid();
        assert_eq!(
            sealed.ukf_sigma_set,
            dust_ukf_rs::SIGMA_SET_TOKEN,
            "the sealed sigma set and the compiled generator must name the same set"
        );
        assert_eq!(dust_ukf_rs::NUM_SIGMA, 7);
    }

    /// The simplex weights are a partition of unity with no negative entry.
    ///
    /// `rebuild_mean_covar_ukf` forms the covariance as
    /// `sum_i wc[i] * (diff_i diff_i^T)` and applies no PSD repair to the
    /// RESULT, so a negative `wc[i]` would make that sum a difference of outer
    /// products and need not be positive semi-definite. The retired Merwe set
    /// could produce one -- its `w0_c` turned negative outside
    /// `0.518 <= alpha <= 1.93`, which is why this module carried a tuning
    /// validator. The simplex cannot: every weight is the same positive
    /// `1 / (n + 1)` by construction. The validator is gone; this is the
    /// property that made it unnecessary, asserted rather than assumed.
    #[test]
    fn simplex_weights_are_a_nonnegative_partition_of_unity() {
        assert!(UKF_SIGMA_WEIGHTS.wm.iter().all(|value| *value > 0.0));
        assert!(UKF_SIGMA_WEIGHTS.wc.iter().all(|value| *value > 0.0));
        assert_eq!(
            UKF_SIGMA_WEIGHTS.wm.map(f64::to_bits),
            UKF_SIGMA_WEIGHTS.wc.map(f64::to_bits),
            "the simplex weights both moments identically"
        );
        let weight_sum_error = (UKF_SIGMA_WEIGHTS.wm.iter().sum::<f64>() - 1.0).abs();
        assert!(weight_sum_error <= 1e-15);
    }

    /// Regression: an over-span UKF arc (tof past the pre-bound candidate
    /// arc's catalogue coverage) must produce a typed ephemeris coverage
    /// failure at the UKF boundary before any propagation runs, now that the
    /// consuming `with_ephemeris_for_arc` builder's result is bound rather
    /// than discarded.
    #[test]
    fn native_batch_validates_ukf_arc_span_even_when_context_is_prebound() {
        use lightyear_odeint_rs::types::{ForceConfig, ForceFlags};

        lightyear_odeint_rs::precomputed_ephem::load_precomputed_ephemeris(ForceFlags::SUN_GRAVITY)
            .expect("embedded Sun catalogue must load");
        let (start, end) = lightyear_odeint_rs::precomputed_ephem::published_ephemeris()
            .expect("published ephemeris store must exist after load")
            .common_jd_range()
            .expect("published catalogue range must exist");
        let epoch_jd = 0.5 * (start + end);

        // Pre-bind the config for a short candidate arc, exactly as production
        // contexts arrive at the UKF.
        let prebound = ForceConfig {
            force_flags: ForceFlags::SUN_GRAVITY,
            sph_order: 0,
            ..ForceConfig::default()
        }
        .with_ephemeris_for_arc(epoch_jd, epoch_jd + 0.01)
        .expect("short candidate arc inside catalogue coverage must bind");

        let c = std::sync::Arc::new(vec![1.0, 0.0, 0.0, 0.0]);
        let s = std::sync::Arc::new(vec![0.0; 4]);
        let packed = std::sync::Arc::new(
            satpy_core::pack_gravity_coeffs(&c, &s, 2, 0)
                .expect("zero-order test gravity coefficients are valid"),
        );
        let mut request =
            TransferRequest::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        request.epoch_jd = epoch_jd;
        request.execution_policy = ExecutionPolicy {
            use_high_fidelity: true,
            require_high_fidelity: true,
            ..ExecutionPolicy::default()
        };
        request.force_config = Some(std::sync::Arc::new(prebound));
        request.packed_coeffs = Some(packed);
        let ctx = PlanContext::from_request(request);

        // A UKF time of flight that pushes the arc past catalogue coverage.
        let tof_s = (end - epoch_jd + 10.0) * satpy_core::SEC_PER_DAY;
        let means = [[7_000.0, 0.0, 0.0, 0.0, 7.5, 0.0]];
        let mut covariance = [[0.0; 6]; 6];
        for (axis, row) in covariance.iter_mut().enumerate() {
            *row.get_mut(axis)
                .expect("fixed covariance diagonal is addressable") = 1.0e-6;
        }

        let error =
            propagate_components_ukf_full_batch(&means, &[covariance], None, tof_s, Some(&ctx))
                .expect_err("over-span UKF arc must fail ephemeris validation");
        assert!(
            matches!(
                error,
                UkfPropagationFailure::Propagation(TransferPropagationFailure::Ephemeris(
                    lightyear_odeint_rs::precomputed_ephem::EphemerisCoverageError::OutsideRange { .. }
                ))
            ),
            "expected an OutsideRange ephemeris failure, got {error:?}"
        );
    }

    /// The bound config returned by `with_ephemeris_for_arc` now feeds the
    /// native batch session. This pins the invariant that made the previous
    /// discard behavior-neutral: a pre-bound and an unbound context must
    /// produce bit-exact identical sigma propagations, because the per-row
    /// kernel config re-derives dynamic ephemeris flags itself. If that
    /// self-binding ever regresses, the explicit binding at the UKF boundary
    /// keeps the unbound path correct, and this test keeps both honest.
    #[test]
    fn native_batch_binds_unbound_context_bit_exactly_to_prebound() {
        use lightyear_odeint_rs::types::{ForceConfig, ForceFlags};

        lightyear_odeint_rs::precomputed_ephem::load_precomputed_ephemeris(ForceFlags::SUN_GRAVITY)
            .expect("embedded Sun catalogue must load");
        let (start, end) = lightyear_odeint_rs::precomputed_ephem::published_ephemeris()
            .expect("published ephemeris store must exist after load")
            .common_jd_range()
            .expect("published catalogue range must exist");
        let epoch_jd = 0.5 * (start + end);
        let tof_s = 600.0;

        let unbound = ForceConfig {
            force_flags: ForceFlags::SUN_GRAVITY,
            sph_order: 0,
            ..ForceConfig::default()
        };
        let prebound = unbound
            .with_ephemeris_for_arc(epoch_jd, epoch_jd + tof_s / satpy_core::SEC_PER_DAY)
            .expect("candidate arc inside catalogue coverage must bind");

        let c = std::sync::Arc::new(vec![1.0, 0.0, 0.0, 0.0]);
        let s = std::sync::Arc::new(vec![0.0; 4]);
        let packed = std::sync::Arc::new(
            satpy_core::pack_gravity_coeffs(&c, &s, 2, 0)
                .expect("zero-order test gravity coefficients are valid"),
        );
        let context_for = |force_config: ForceConfig| {
            let mut request = TransferRequest::with_j2_closure_settings(
                crate::solve::J2ClosureSettings::default(),
            );
            request.epoch_jd = epoch_jd;
            request.execution_policy = ExecutionPolicy {
                use_high_fidelity: true,
                require_high_fidelity: true,
                ..ExecutionPolicy::default()
            };
            request.force_config = Some(std::sync::Arc::new(force_config));
            request.packed_coeffs = Some(std::sync::Arc::clone(&packed));
            PlanContext::from_request(request)
        };

        let means = [[7_000.0, 137.0, -83.0, -0.217, 7.431, 0.913]];
        let mut covariance = [[0.0; 6]; 6];
        for (axis, row) in covariance.iter_mut().enumerate() {
            *row.get_mut(axis)
                .expect("fixed covariance diagonal is addressable") = 1.0e-6;
        }

        let propagate = |ctx: &PlanContext| {
            propagate_components_ukf_full_batch(&means, &[covariance], None, tof_s, Some(ctx))
                .expect("in-coverage UKF batch must propagate")
        };
        let from_prebound = propagate(&context_for(prebound));
        let from_unbound = propagate(&context_for(unbound));

        assert_eq!(
            from_prebound.propagated_sigma_points.len(),
            from_unbound.propagated_sigma_points.len()
        );
        for (bound_point, unbound_point) in from_prebound
            .propagated_sigma_points
            .iter()
            .zip(from_unbound.propagated_sigma_points.iter())
        {
            assert_eq!(
                bound_point.map(f64::to_bits),
                unbound_point.map(f64::to_bits)
            );
        }
    }
}
