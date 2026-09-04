//! Spherical harmonic gravity field computations.
//!
//! This module provides functions for computing gravitational acceleration using
//! spherical harmonic expansion of the Earth's gravity field. It supports both
//! generic (unpacked) and packed coefficient formats, with optional SIMD acceleration.
//!
//! ## SIMD Optimizations (P2 - DUST-50, DUST-51)
//!
//! When the `simd` feature is enabled, this module uses SIMD for:
//! - Gravity coefficient summation (existing)
//! - ECI/ECEF coordinate transforms via SIMD 2x2 rotation (DUST-50)
//! - V/W Legendre recursion across multiple l-values (DUST-51)

use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use sha2::{Digest, Sha256};
use std::fmt;
use std::sync::OnceLock;
use wide::f64x4;

/// Finite failure modes for gravity coefficient construction and evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GravityError {
    /// The requested harmonic order exceeds the fixed gravity kernel limit.
    UnsupportedOrder,
    /// The C/S storage does not form the required finite square matrix.
    InvalidCoefficientStorage,
    /// Private packed metadata or fixed recurrence storage violated its proof.
    InvariantViolation,
    /// The state vector is too short or has a non-finite component.
    InvalidState,
    /// Julian date input is not finite.
    InvalidTime,
    /// A supplied sine/cosine rotation pair has a non-finite component.
    InvalidRotation,
    /// A gravity position is non-finite or has zero radius.
    InvalidRadius,
}

impl fmt::Display for GravityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedOrder => formatter.write_str("unsupported gravity harmonic order"),
            Self::InvalidCoefficientStorage => {
                formatter.write_str("invalid gravity coefficient storage")
            }
            Self::InvariantViolation => formatter.write_str("gravity invariant violation"),
            Self::InvalidState => formatter.write_str("invalid gravity state"),
            Self::InvalidTime => formatter.write_str("invalid gravity time"),
            Self::InvalidRotation => formatter.write_str("invalid gravity rotation"),
            Self::InvalidRadius => formatter.write_str("invalid gravity radius"),
        }
    }
}

impl std::error::Error for GravityError {}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "checked generic-radius validation preserves the input contract's arithmetic order"
)]
fn validated_state<T: Float>(state: &[T; 6]) -> Result<[T; 6], GravityError> {
    if state.iter().any(|value| !value.is_finite()) {
        return Err(GravityError::InvalidState);
    }
    let &[x, y, z, _, _, _] = state;
    let radius_squared = x * x + y * y + z * z;
    if !radius_squared.is_finite() || radius_squared <= T::zero() {
        return Err(GravityError::InvalidRadius);
    }
    Ok(*state)
}

#[inline]
fn validate_position(position: [f64; 3]) -> Result<(), GravityError> {
    let [x, y, z] = position;
    if !crate::safe_isfinite(x) || !crate::safe_isfinite(y) || !crate::safe_isfinite(z) {
        return Err(GravityError::InvalidRadius);
    }
    let radius_squared = x * x + y * y + z * z;
    if !crate::safe_isfinite(radius_squared) || radius_squared <= 0.0 {
        return Err(GravityError::InvalidRadius);
    }
    Ok(())
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "checked generic-radius validation preserves the input contract's arithmetic order"
)]
fn validate_position_generic<T: Float>(position: [T; 3]) -> Result<(), GravityError> {
    let [x, y, z] = position;
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return Err(GravityError::InvalidRadius);
    }
    let radius_squared = x * x + y * y + z * z;
    if !radius_squared.is_finite() || radius_squared <= T::zero() {
        return Err(GravityError::InvalidRadius);
    }
    Ok(())
}

#[inline]
const fn validate_jd(jd: f64) -> Result<(), GravityError> {
    if crate::safe_isfinite(jd) {
        Ok(())
    } else {
        Err(GravityError::InvalidTime)
    }
}

#[inline]
const fn validate_rotation(sine: f64, cosine: f64) -> Result<(), GravityError> {
    if crate::safe_isfinite(sine) && crate::safe_isfinite(cosine) {
        Ok(())
    } else {
        Err(GravityError::InvalidRotation)
    }
}

#[inline]
fn matrix_value<T: Copy>(
    matrix: &[[T; MAX_RECURSIVE_ORDER]],
    row: usize,
    column: usize,
) -> Result<T, GravityError> {
    matrix
        .get(row)
        .and_then(|values| values.get(column))
        .copied()
        .ok_or(GravityError::InvariantViolation)
}

#[inline]
fn matrix_set<T: Copy>(
    matrix: &mut [[T; MAX_RECURSIVE_ORDER]],
    row: usize,
    column: usize,
    value: T,
) -> Result<(), GravityError> {
    let slot = matrix
        .get_mut(row)
        .and_then(|values| values.get_mut(column))
        .ok_or(GravityError::InvariantViolation)?;
    *slot = value;
    Ok(())
}

/// Load `row[start..start + 4]` as one vector, bounded by a single check.
///
/// The four lanes are proven in bounds together, so the backend is free to
/// issue one contiguous load instead of four checked scalar reads feeding four
/// inserts.
#[inline]
fn row_quad(row: &[f64], start: usize) -> Result<f64x4, GravityError> {
    let end = start
        .checked_add(4)
        .ok_or(GravityError::InvariantViolation)?;
    let window = row
        .get(start..end)
        .ok_or(GravityError::InvariantViolation)?;
    Ok(f64x4::new(four_values(window)?))
}

/// Store one vector into `row[start..start + 4]`, bounded by a single check.
#[inline]
fn store_row_quad(row: &mut [f64], start: usize, values: f64x4) -> Result<(), GravityError> {
    let end = start
        .checked_add(4)
        .ok_or(GravityError::InvariantViolation)?;
    let destination = row
        .get_mut(start..end)
        .ok_or(GravityError::InvariantViolation)?;
    destination.copy_from_slice(&values.to_array());
    Ok(())
}

#[inline]
fn nested_value(
    matrix: &[[f64; MAX_RECURSIVE_ORDER]],
    row: usize,
    column: usize,
) -> Result<f64, GravityError> {
    matrix
        .get(row)
        .and_then(|values| values.get(column))
        .copied()
        .ok_or(GravityError::InvariantViolation)
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "preserve the raw generic Legendre recurrence's floating-point operation order"
)]
fn fill_legendre_generic<T: Float + FromPrimitive>(
    v: &mut [[T; MAX_RECURSIVE_ORDER]],
    w: &mut [[T; MAX_RECURSIVE_ORDER]],
    n: usize,
    x_c2: T,
    y_c2: T,
    z_c2: T,
    c2_re: T,
) -> Result<(), GravityError> {
    let v00 = c2_re.sqrt();
    matrix_set(v, 0, 0, v00)?;
    matrix_set(w, 0, 0, T::zero())?;
    matrix_set(w, 1, 0, T::zero())?;
    matrix_set(v, 1, 0, z_c2 * v00)?;

    for l in 2..n {
        matrix_set(w, l, 0, T::zero())?;
        let l_value = l.to_f64().ok_or(GravityError::InvariantViolation)?;
        let pt1 =
            T::from_f64((l_value * 2.0 - 1.0) / l_value).ok_or(GravityError::InvariantViolation)?;
        let recurrence_correction =
            T::from_f64((l_value - 1.0) / l_value).ok_or(GravityError::InvariantViolation)?;
        let prior = l.checked_sub(1).ok_or(GravityError::InvariantViolation)?;
        let prior_prior = l.checked_sub(2).ok_or(GravityError::InvariantViolation)?;
        let next = pt1 * z_c2 * matrix_value(v, prior, 0)?
            - recurrence_correction * c2_re * matrix_value(v, prior_prior, 0)?;
        matrix_set(v, l, 0, next)?;
    }
    for m in 1..n {
        let m_value = m.to_f64().ok_or(GravityError::InvariantViolation)?;
        let c1 = T::from_f64(m_value * 2.0 - 1.0).ok_or(GravityError::InvariantViolation)?;
        let previous = m.checked_sub(1).ok_or(GravityError::InvariantViolation)?;
        let v_prev = matrix_value(v, previous, previous)?;
        let w_prev = matrix_value(w, previous, previous)?;
        let v_diagonal = c1 * (x_c2 * v_prev - y_c2 * w_prev);
        let w_diagonal = c1 * (x_c2 * w_prev + y_c2 * v_prev);
        matrix_set(v, m, m, v_diagonal)?;
        matrix_set(w, m, m, w_diagonal)?;
        if m.checked_add(1).is_some_and(|next_row| next_row < n) {
            let pt1_mp1 =
                T::from_f64((m_value + 1.0) * 2.0 - 1.0).ok_or(GravityError::InvariantViolation)?;
            let next_row = m.checked_add(1).ok_or(GravityError::InvariantViolation)?;
            matrix_set(v, next_row, m, pt1_mp1 * z_c2 * v_diagonal)?;
            matrix_set(w, next_row, m, pt1_mp1 * z_c2 * w_diagonal)?;
        }
        let first_recurrence_row = m.checked_add(2).ok_or(GravityError::InvariantViolation)?;
        for l in first_recurrence_row..n {
            let l_value = l.to_f64().ok_or(GravityError::InvariantViolation)?;
            let denominator = l
                .checked_sub(m)
                .ok_or(GravityError::InvariantViolation)?
                .to_f64()
                .ok_or(GravityError::InvariantViolation)?;
            let pt1 = T::from_f64((l_value * 2.0 - 1.0) / denominator)
                .ok_or(GravityError::InvariantViolation)?;
            let recurrence_correction = T::from_f64(
                l.checked_add(m)
                    .and_then(|value| value.checked_sub(1))
                    .ok_or(GravityError::InvariantViolation)?
                    .to_f64()
                    .ok_or(GravityError::InvariantViolation)?
                    / denominator,
            )
            .ok_or(GravityError::InvariantViolation)?
                * c2_re;
            let prior = l.checked_sub(1).ok_or(GravityError::InvariantViolation)?;
            let prior_prior = l.checked_sub(2).ok_or(GravityError::InvariantViolation)?;
            let v_next = pt1 * z_c2 * matrix_value(v, prior, m)?
                - recurrence_correction * matrix_value(v, prior_prior, m)?;
            let w_next = pt1 * z_c2 * matrix_value(w, prior, m)?
                - recurrence_correction * matrix_value(w, prior_prior, m)?;
            matrix_set(v, l, m, v_next)?;
            matrix_set(w, l, m, w_next)?;
        }
    }
    Ok(())
}

#[inline]
fn fill_legendre_packed_f64(
    v: &mut [[f64; MAX_RECURSIVE_ORDER]],
    w: &mut [[f64; MAX_RECURSIVE_ORDER]],
    n: usize,
    x_c2: f64,
    y_c2: f64,
    z_c2: f64,
    c2_re: f64,
) -> Result<(), GravityError> {
    let v00 = c2_re.sqrt();
    matrix_set(v, 0, 0, v00)?;
    matrix_set(w, 0, 0, 0.0)?;
    matrix_set(w, 1, 0, 0.0)?;
    matrix_set(v, 1, 0, z_c2 * v00)?;

    for l in 2..n {
        matrix_set(w, l, 0, 0.0)?;
        let lf = l.to_f64().ok_or(GravityError::InvariantViolation)?;
        let pt1 = (2.0 * lf - 1.0) / lf;
        let recurrence_correction = ((lf - 1.0) / lf) * c2_re;
        let next = pt1.mul_add(
            z_c2 * matrix_value(v, l.saturating_sub(1), 0)?,
            -recurrence_correction * matrix_value(v, l.saturating_sub(2), 0)?,
        );
        matrix_set(v, l, 0, next)?;
    }

    for m in 1..n {
        let m_value = m.to_f64().ok_or(GravityError::InvariantViolation)?;
        let c1 = 2.0 * m_value - 1.0;
        let previous = m.saturating_sub(1);
        let v_prev = matrix_value(v, previous, previous)?;
        let w_prev = matrix_value(w, previous, previous)?;
        let v_diagonal = c1 * (x_c2 * v_prev - y_c2 * w_prev);
        let w_diagonal = c1 * (x_c2 * w_prev + y_c2 * v_prev);
        matrix_set(v, m, m, v_diagonal)?;
        matrix_set(w, m, m, w_diagonal)?;
        if m.saturating_add(1) < n {
            let pt1_mp1 = 2.0 * (m_value + 1.0) - 1.0;
            let next_row = m.saturating_add(1);
            matrix_set(v, next_row, m, pt1_mp1 * z_c2 * v_diagonal)?;
            matrix_set(w, next_row, m, pt1_mp1 * z_c2 * w_diagonal)?;
        }
        let mut l = m.saturating_add(2);
        let mut denom = 2.0f64;
        let mut pt1_num = l
            .saturating_mul(2)
            .saturating_sub(1)
            .to_f64()
            .ok_or(GravityError::InvariantViolation)?;
        let mut correction_numerator = l
            .saturating_add(m)
            .saturating_sub(1)
            .to_f64()
            .ok_or(GravityError::InvariantViolation)?;
        while l < n {
            let inv_denom = 1.0 / denom;
            let pt1 = pt1_num * inv_denom;
            let recurrence_correction = (correction_numerator * inv_denom) * c2_re;
            let v_next = pt1.mul_add(
                z_c2 * matrix_value(v, l.saturating_sub(1), m)?,
                -recurrence_correction * matrix_value(v, l.saturating_sub(2), m)?,
            );
            let w_next = pt1.mul_add(
                z_c2 * matrix_value(w, l.saturating_sub(1), m)?,
                -recurrence_correction * matrix_value(w, l.saturating_sub(2), m)?,
            );
            matrix_set(v, l, m, v_next)?;
            matrix_set(w, l, m, w_next)?;
            l = l.saturating_add(1);
            denom += 1.0;
            pt1_num += 2.0;
            correction_numerator += 1.0;
        }
    }
    Ok(())
}

#[inline]
fn fill_legendre_precomputed_f64(
    v: &mut [[f64; MAX_RECURSIVE_ORDER]],
    w: &mut [[f64; MAX_RECURSIVE_ORDER]],
    n: usize,
    x_c2: f64,
    y_c2: f64,
    z_c2: f64,
    c2_re: f64,
    leg: &LegendreCoeffsSimd,
) -> Result<(), GravityError> {
    let v00 = c2_re.sqrt();
    matrix_set(v, 0, 0, v00)?;
    matrix_set(w, 0, 0, 0.0)?;
    matrix_set(w, 1, 0, 0.0)?;
    matrix_set(v, 1, 0, z_c2 * v00)?;

    for l in 2..n {
        matrix_set(w, l, 0, 0.0)?;
        let pt1 = nested_value(&leg.pt1, l, 0)?;
        let recurrence_correction = nested_value(&leg.pt21_factor, l, 0)? * c2_re;
        let next = pt1.mul_add(
            z_c2 * matrix_value(v, l.saturating_sub(1), 0)?,
            -recurrence_correction * matrix_value(v, l.saturating_sub(2), 0)?,
        );
        matrix_set(v, l, 0, next)?;
    }

    for m in 1..n {
        let m_value = m.to_f64().ok_or(GravityError::InvariantViolation)?;
        let c1 = 2.0 * m_value - 1.0;
        let previous = m.saturating_sub(1);
        let v_prev = matrix_value(v, previous, previous)?;
        let w_prev = matrix_value(w, previous, previous)?;
        let v_diagonal = c1 * (x_c2 * v_prev - y_c2 * w_prev);
        let w_diagonal = c1 * (x_c2 * w_prev + y_c2 * v_prev);
        matrix_set(v, m, m, v_diagonal)?;
        matrix_set(w, m, m, w_diagonal)?;
        if m.saturating_add(1) < n {
            let pt1_mp1 = 2.0 * (m_value + 1.0) - 1.0;
            let next_row = m.saturating_add(1);
            matrix_set(v, next_row, m, pt1_mp1 * z_c2 * v_diagonal)?;
            matrix_set(w, next_row, m, pt1_mp1 * z_c2 * w_diagonal)?;
        }
        for l in m.saturating_add(2)..n {
            let pt1 = nested_value(&leg.pt1, l, m)?;
            let recurrence_correction = nested_value(&leg.pt21_factor, l, m)? * c2_re;
            let v_next = pt1.mul_add(
                z_c2 * matrix_value(v, l.saturating_sub(1), m)?,
                -recurrence_correction * matrix_value(v, l.saturating_sub(2), m)?,
            );
            let w_next = pt1.mul_add(
                z_c2 * matrix_value(w, l.saturating_sub(1), m)?,
                -recurrence_correction * matrix_value(w, l.saturating_sub(2), m)?,
            );
            matrix_set(v, l, m, v_next)?;
            matrix_set(w, l, m, w_next)?;
        }
    }
    Ok(())
}

#[inline]
fn two_values<T: Copy>(values: &[T]) -> Result<(T, T), GravityError> {
    let [first, second, ..] = values else {
        return Err(GravityError::InvariantViolation);
    };
    Ok((*first, *second))
}

#[inline]
fn three_values<T: Copy>(values: &[T]) -> Result<(T, T, T), GravityError> {
    let [first, second, third, ..] = values else {
        return Err(GravityError::InvariantViolation);
    };
    Ok((*first, *second, *third))
}

#[inline]
fn four_values<T: Copy>(values: &[T]) -> Result<[T; 4], GravityError> {
    let [first, second, third, fourth, ..] = values else {
        return Err(GravityError::InvariantViolation);
    };
    Ok([*first, *second, *third, *fourth])
}

/// Read the `[m - 1, m, m + 1]` neighbourhood of one recurrence row.
///
/// This is the direct-subslice replacement for a `windows(3).enumerate()` cursor
/// that every caller advanced with a linear `find` to offset `m - 1`. The cursor
/// form cost an iterator step plus a closure call per harmonic order per term,
/// was rebuilt for every degree row, and — because a `&mut` cursor is shared
/// state — serialized the four lane gathers of the packed SIMD quad into a
/// dependency chain. A subslice bounded once by a single range check has none of
/// those properties, and the four lane reads become independent.
///
/// The cursor form silently relied on `m` strictly increasing within a row: a
/// repeated or decreasing order would have found nothing, because the matching
/// window was already consumed. That ordering is separately enforced for every
/// pack by [`PackedGravityCoeffs::validate_metadata`], which rejects
/// `term.m <= previous_m`, so this form returns the same coordinates on every
/// valid input and is defined rather than order-dependent on any other.
#[inline]
fn packed_row_window<T: Copy>(row: &[T], harmonic_order: usize) -> Result<(T, T, T), GravityError> {
    let start = harmonic_order
        .checked_sub(1)
        .ok_or(GravityError::InvariantViolation)?;
    let end = harmonic_order
        .checked_add(2)
        .ok_or(GravityError::InvariantViolation)?;
    let window = row
        .get(start..end)
        .ok_or(GravityError::InvariantViolation)?;
    three_values(window)
}

#[derive(Clone, Copy)]
struct GravityTermCoordinates {
    v_below: f64,
    v_same: f64,
    v_above: f64,
    w_below: f64,
    w_same: f64,
    w_above: f64,
}

/// Gather one packed term's V/W neighbourhood from the two recurrence rows.
///
/// Takes the rows by shared reference rather than a shared `&mut` window
/// cursor, so four calls for the four orders of a SIMD quad are independent and
/// can issue in parallel.
#[inline]
fn packed_term_coordinates(
    v_row: &[f64],
    w_row: &[f64],
    harmonic_order: usize,
) -> Result<GravityTermCoordinates, GravityError> {
    let (v_below, v_same, v_above) = packed_row_window(v_row, harmonic_order)?;
    let (w_below, w_same, w_above) = packed_row_window(w_row, harmonic_order)?;
    Ok(GravityTermCoordinates {
        v_below,
        v_same,
        v_above,
        w_below,
        w_same,
        w_above,
    })
}

use crate::{
    ecef2eci_impl, ecef2eci_impl_sincos, eci2ecef_impl, eci2ecef_impl_sincos, greenwichsrt_impl,
    GRAVITY_REFERENCE_RADIUS_KM, MU,
};

// Wide-literal vector for the quad summation body. A `const` item by policy:
// an inline splat of a literal lowers to a per-call `memset_pattern16` libc
// call on aarch64-macos (see `wide_consts!` in lib.rs), and this one sat in
// the per-(l, m) hot body.
crate::wide_consts! {
    TWO_X4 = 2.0,
}

// =============================================================================
// SIMD V/W Legendre Recursion (DUST-51)
// =============================================================================
//
// The Pines algorithm computes V[l][m] and W[l][m] arrays using two recursions:
// 1. m=0 diagonal: V[l][0] = pt1 * z_c2 * V[l-1][0] - recurrence_correction * V[l-2][0]
// 2. m>0 diagonal: V[l][m] = pt1 * z_c2 * V[l-1][m] - recurrence_correction * V[l-2][m]
//
// The key insight for SIMD optimization:
// - For a fixed l, we can process multiple m values in parallel (within bounds)
// - The recursion coefficients pt1 and recurrence_correction depend on both l and m
// - We process 4 consecutive m values at once using f64x4
//
// This is most beneficial for higher-order gravity models (order >= 8).

/// Precomputed Legendre recursion coefficients.
/// Stores `pt1` and `recurrence_correction` coefficients for each `(l, m)` pair,
/// eliminating per-step divisions in the V/W recursion hot loops.
struct LegendreCoeffsSimd {
    /// pt1[l][m] = (2l - 1) / (l - m) for l >= m+2
    pt1: [[f64; MAX_RECURSIVE_ORDER]; MAX_RECURSIVE_ORDER],
    /// `pt21_factor[l][m] = (l + m - 1) / (l - m)` for `l >= m+2`
    /// (multiply by `c2_re` at runtime)
    pt21_factor: [[f64; MAX_RECURSIVE_ORDER]; MAX_RECURSIVE_ORDER],
}

impl LegendreCoeffsSimd {
    /// Build the fixed coefficient table used by every gravity cache.
    ///
    /// `const fn` so [`SHARED_LEGENDRE_COEFFS`] const-evaluates into rodata.
    /// The builder is pure `+,-,*,/` on small exact integers-as-f64, and
    /// const-eval of those operations is IEEE-identical to running the same
    /// loop at runtime, so every coefficient bit matches a runtime build —
    /// held by `shared_legendre_table_is_bit_identical_to_a_fresh_one`, which
    /// compares this function's runtime evaluation against the static.
    #[expect(
        clippy::indexing_slicing,
        reason = "const builder over fixed-size arrays: while-loops replace the runtime iterator chain, every index is bounded by MAX_RECURSIVE_ORDER, and an out-of-bounds slip fails the static's const evaluation at compile time"
    )]
    #[expect(
        clippy::large_stack_arrays,
        clippy::large_stack_frames,
        reason = "the production path is the const-evaluated SHARED_LEGENDRE_COEFFS static (rodata, no stack frame); the only runtime callers are the bit-identity tests, whose ~800 KB frame fits the 2 MiB test-thread stack"
    )]
    const fn fixed() -> Self {
        let mut pt1 = [[0.0; MAX_RECURSIVE_ORDER]; MAX_RECURSIVE_ORDER];
        let mut pt21_factor = [[0.0; MAX_RECURSIVE_ORDER]; MAX_RECURSIVE_ORDER];

        let mut degree = 2;
        let mut l_value = 2.0;
        while degree < MAX_RECURSIVE_ORDER {
            let mut order = 0;
            let mut order_value = 0.0;
            let mut denominator = l_value;
            while order < degree.saturating_sub(1) {
                pt1[degree][order] = (2.0 * l_value - 1.0) / denominator;
                pt21_factor[degree][order] = (l_value + order_value - 1.0) / denominator;
                order_value += 1.0;
                denominator -= 1.0;
                order = order.saturating_add(1);
            }
            l_value += 1.0;
            degree = degree.saturating_add(1);
        }

        Self { pt1, pt21_factor }
    }

    /// Borrow the process-wide coefficient table.
    ///
    /// [`Self::fixed`] takes no arguments: the table is a pure function of
    /// [`MAX_RECURSIVE_ORDER`], so every cache shares the one const-evaluated
    /// copy in rodata instead of rebuilding it. Nothing can write to the
    /// `static`, so sharing cannot change any evaluated value.
    const fn shared() -> &'static Self {
        &SHARED_LEGENDRE_COEFFS
    }
}

/// The one coefficient table in the process image, const-evaluated into
/// shared file-backed rodata (~274 KB). This used to be an `OnceLock`
/// holding two heap `Box<[[f64; MAX_RECURSIVE_ORDER]]>` built at first use,
/// which cost every gravity-cache construction a `get_or_init` acquire and
/// every process a private heap copy of the same fixed bytes.
static SHARED_LEGENDRE_COEFFS: LegendreCoeffsSimd = LegendreCoeffsSimd::fixed();

/// SIMD-accelerated V/W recursion across multiple m-columns (DUST-51).
///
/// For a fixed l, processes 4 consecutive m-columns simultaneously:
/// V[l][m], V[l][m+1], V[l][m+2], V[l][m+3]
///
/// This is the core SIMD optimization for the Legendre recursion. The key insight
/// is that while we can't parallelize across l (due to dependencies), we CAN
/// parallelize across m because the recursion for different m-columns is independent
/// after the diagonal elements are computed.
///
/// # Requirements
/// - V[l-1][m..m+4] and V[l-2][m..m+4] must already be computed
/// - m + 3 < l to ensure valid array indices
#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "preserve the authority Legendre recurrence's floating-point operation order"
)]
fn legendre_l_row_simd(
    v: &mut [[f64; MAX_RECURSIVE_ORDER]],
    w: &mut [[f64; MAX_RECURSIVE_ORDER]],
    l: usize,
    m_start: usize,
    m_end: usize,
    z_c2: f64,
    c2_re: f64,
    leg: &LegendreCoeffsSimd,
) -> Result<(), GravityError> {
    let z_c2_v = f64x4::splat(z_c2);
    let c2_re_v = f64x4::splat(c2_re);

    let mut m = m_start;

    // SIMD path: process 4 m-columns at once using precomputed coefficients.
    //
    // Every bound this arm needs is resolved once, before the quad loop. The
    // arm touches exactly five rows -- `leg.pt1[l]`, `leg.pt21_factor[l]`,
    // `v/w[l-2]` and `v/w[l-1]` read-only, and `v/w[l]` written -- so the row
    // half of each access is loop-invariant. `split_at_mut(l)` is what makes
    // that expressible: it hands out the read-only rows below `l` and the
    // written row `l` as disjoint borrows, which per-cell reads and writes
    // through one `&mut [[f64; _]]` cannot do while both alias the same slice.
    //
    // What is left inside the loop is one range check per row per quad rather
    // than one per lane: the 26 checked accesses that produced 8 results become
    // 8, and each surviving check bounds a contiguous four-lane slice the
    // backend can fold into a single vector load or store.
    if m + 3 < m_end && m + 3 < l {
        let previous2_row = l.checked_sub(2).ok_or(GravityError::InvariantViolation)?;
        let pt1_row = leg.pt1.get(l).ok_or(GravityError::InvariantViolation)?;
        let correction_row = leg
            .pt21_factor
            .get(l)
            .ok_or(GravityError::InvariantViolation)?;
        if l >= v.len() || l >= w.len() {
            return Err(GravityError::InvariantViolation);
        }
        let (v_below_l, v_from_l) = v.split_at_mut(l);
        let (w_below_l, w_from_l) = w.split_at_mut(l);
        // `v_below_l` is exactly `l` rows long, so the tail from `l - 2` is
        // exactly the two rows the recurrence reads.
        let [v_older_row, v_recent_row] = v_below_l
            .get(previous2_row..)
            .ok_or(GravityError::InvariantViolation)?
        else {
            return Err(GravityError::InvariantViolation);
        };
        let [w_older_row, w_recent_row] = w_below_l
            .get(previous2_row..)
            .ok_or(GravityError::InvariantViolation)?
        else {
            return Err(GravityError::InvariantViolation);
        };
        let v_row = v_from_l
            .first_mut()
            .ok_or(GravityError::InvariantViolation)?;
        let w_row = w_from_l
            .first_mut()
            .ok_or(GravityError::InvariantViolation)?;

        while m + 3 < m_end && m + 3 < l {
            // Load precomputed coefficients (eliminates 8 divisions per SIMD step)
            let pt1 = row_quad(pt1_row, m)?;
            let recurrence_correction = row_quad(correction_row, m)? * c2_re_v;

            // Load V[l-1][m..m+4] and V[l-2][m..m+4]
            let v_prev = row_quad(v_recent_row, m)?;
            let v_prev2 = row_quad(v_older_row, m)?;
            let w_prev = row_quad(w_recent_row, m)?;
            let w_prev2 = row_quad(w_older_row, m)?;

            // V[l][m] = pt1 * z_c2 * V[l-1][m] - recurrence_correction * V[l-2][m]
            let pt1_z = pt1 * z_c2_v;
            let v_new = pt1_z.mul_add(v_prev, -(recurrence_correction * v_prev2));
            let w_new = pt1_z.mul_add(w_prev, -(recurrence_correction * w_prev2));

            // Store results
            store_row_quad(v_row, m, v_new)?;
            store_row_quad(w_row, m, w_new)?;

            m += 4;
        }
    }

    // Scalar cleanup for remainder using precomputed coefficients
    while m < m_end && m < l {
        let pt1 = nested_value(&leg.pt1, l, m)?;
        let recurrence_correction = nested_value(&leg.pt21_factor, l, m)? * c2_re;
        let v_new = pt1.mul_add(
            z_c2 * matrix_value(v, l.saturating_sub(1), m)?,
            -recurrence_correction * matrix_value(v, l.saturating_sub(2), m)?,
        );
        let w_new = pt1.mul_add(
            z_c2 * matrix_value(w, l.saturating_sub(1), m)?,
            -recurrence_correction * matrix_value(w, l.saturating_sub(2), m)?,
        );
        matrix_set(v, l, m, v_new)?;
        matrix_set(w, l, m, w_new)?;
        m += 1;
    }
    Ok(())
}

/// Trip count the monomorphised Legendre arm is specialised for.
///
/// Production runs `gravity_order = 5` (`nd_config::part_a_science`) and the
/// recurrence needs two rows beyond the requested order, so the fill's trip
/// count is `5 + 2`. Every other order keeps the runtime body.
const PROD_LEGENDRE_N: usize = 7;

/// Which arm of [`legendre_vw_dispatch`] ran.
///
/// Returned rather than counted so the selection is observable from a test
/// without a probe living in the production body: the dispatch test reads the
/// arm the real `match` took, not a copy of its predicate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LegendreArm {
    /// The `const N` monomorphisation for the production trip count.
    ConstProd,
    /// The runtime-trip-count body, which serves every other order.
    Runtime,
}

/// Route a Legendre V/W fill to the monomorphised arm when the trip count is
/// the production one.
///
/// The two arms are required to be bit-identical, and are held to that by
/// `legendre_const_arm_is_bit_identical_to_runtime_arm`; the const arm exists
/// only because a compile-time trip count unrolls where a runtime one cannot.
/// Exactly one monomorphisation is instantiated on purpose — each additional
/// arm is I-cache weight paid on every RHS evaluation.
///
/// # Errors
///
/// Propagates the selected arm's checked-bounds failures unchanged.
#[inline]
fn legendre_vw_dispatch(
    v: &mut [[f64; MAX_RECURSIVE_ORDER]],
    w: &mut [[f64; MAX_RECURSIVE_ORDER]],
    n: usize,
    x_c2: f64,
    y_c2: f64,
    z_c2: f64,
    c2_re: f64,
    v00: f64,
    leg: &LegendreCoeffsSimd,
) -> Result<LegendreArm, GravityError> {
    let arm = if n == PROD_LEGENDRE_N {
        LegendreArm::ConstProd
    } else {
        LegendreArm::Runtime
    };
    match arm {
        LegendreArm::ConstProd => legendre_vw_row_major_const::<PROD_LEGENDRE_N>(
            v,
            w,
            x_c2,
            y_c2,
            z_c2,
            c2_re,
            v00,
            leg,
            LEGENDRE_SIMD_L_THRESHOLD,
        )?,
        LegendreArm::Runtime => legendre_vw_row_major_with_threshold(
            v,
            w,
            n,
            x_c2,
            y_c2,
            z_c2,
            c2_re,
            v00,
            leg,
            LEGENDRE_SIMD_L_THRESHOLD,
        )?,
    }
    Ok(arm)
}

/// Stamps THE Legendre V/W fill body for both trip-count spellings:
/// [`legendre_vw_row_major_with_threshold`] passes its runtime `n` and
/// [`legendre_vw_row_major_const`] its `const N`, so the two arms are the same
/// token stream by construction and cannot drift apart. That identity used to
/// be held by hand ("edit both or neither") plus an `include_str!` source-scan
/// test (`legendre_arms_are_the_same_body`), both retired by this macro; the
/// numeric oracle `legendre_const_arm_is_bit_identical_to_runtime_arm` remains
/// the gate that the const monomorphisation moves no bits.
///
/// Every binding the body reads is passed in because `macro_rules!` hygiene
/// keeps definition-site identifiers from resolving to caller locals.
macro_rules! legendre_fill_body {
    ($v:ident, $w:ident, $n:expr, $x_c2:ident, $y_c2:ident, $z_c2:ident,
     $c2_re:ident, $v00:ident, $leg:ident, $simd_threshold:ident) => {{
        // Initialize base cases
        matrix_set($v, 0, 0, $v00)?;
        matrix_set($w, 0, 0, 0.0)?;
        matrix_set($w, 1, 0, 0.0)?;
        matrix_set($v, 1, 0, $z_c2 * $v00)?;

        // m=0 column using precomputed coefficients
        for l in 2..$n {
            matrix_set($w, l, 0, 0.0)?;
            let pt1 = nested_value(&$leg.pt1, l, 0)?;
            let recurrence_correction = nested_value(&$leg.pt21_factor, l, 0)? * $c2_re;
            let next = pt1.mul_add(
                $z_c2 * matrix_value($v, l.saturating_sub(1), 0)?,
                -recurrence_correction * matrix_value($v, l.saturating_sub(2), 0)?,
            );
            matrix_set($v, l, 0, next)?;
        }

        // Fill diagonal and sub-diagonal for all m first
        for m in 1..$n {
            let m_value = m.to_f64().ok_or(GravityError::InvariantViolation)?;
            let c1 = 2.0 * m_value - 1.0;
            let previous = m.saturating_sub(1);
            let v_prev = matrix_value($v, previous, previous)?;
            let w_prev = matrix_value($w, previous, previous)?;
            let v_diagonal = c1 * ($x_c2 * v_prev - $y_c2 * w_prev);
            let w_diagonal = c1 * ($x_c2 * w_prev + $y_c2 * v_prev);
            matrix_set($v, m, m, v_diagonal)?;
            matrix_set($w, m, m, w_diagonal)?;

            if m.saturating_add(1) < $n {
                let pt1_mp1 = 2.0 * (m_value + 1.0) - 1.0;
                let next_row = m.saturating_add(1);
                matrix_set($v, next_row, m, pt1_mp1 * $z_c2 * v_diagonal)?;
                matrix_set($w, next_row, m, pt1_mp1 * $z_c2 * w_diagonal)?;
            }
        }

        // Now fill the remaining elements row by row (for each l)
        // This allows SIMD across m-columns
        // BUG FIX: was `4..`, must be `3..` — v[3][1] is an inner recursion
        // element (l=3, m=1 satisfies l >= m+2) that was previously skipped.
        for l in 3..$n {
            let row_end = l.checked_sub(1).ok_or(GravityError::InvariantViolation)?;
            if l >= $simd_threshold {
                // SIMD path: process 4 m-columns at once using precomputed coefficients
                // Start from m=1 (m=0 already done), end at m < l-1
                legendre_l_row_simd($v, $w, l, 1, row_end, $z_c2, $c2_re, $leg)?;
            } else {
                // Scalar path for small l using precomputed coefficients
                for m in 1..row_end {
                    let pt1 = nested_value(&$leg.pt1, l, m)?;
                    let recurrence_correction = nested_value(&$leg.pt21_factor, l, m)? * $c2_re;
                    let v_next = pt1.mul_add(
                        $z_c2 * matrix_value($v, l.saturating_sub(1), m)?,
                        -recurrence_correction * matrix_value($v, l.saturating_sub(2), m)?,
                    );
                    let w_next = pt1.mul_add(
                        $z_c2 * matrix_value($w, l.saturating_sub(1), m)?,
                        -recurrence_correction * matrix_value($w, l.saturating_sub(2), m)?,
                    );
                    matrix_set($v, l, m, v_next)?;
                    matrix_set($w, l, m, w_next)?;
                }
            }
        }
    }};
}

/// Threshold-parameterised body. The parameter exists so the two settings can
/// be measured against each other; production always enters through
/// [`legendre_vw_dispatch`] with [`LEGENDRE_SIMD_L_THRESHOLD`].
#[inline]
fn legendre_vw_row_major_with_threshold(
    v: &mut [[f64; MAX_RECURSIVE_ORDER]],
    w: &mut [[f64; MAX_RECURSIVE_ORDER]],
    n: usize,
    x_c2: f64,
    y_c2: f64,
    z_c2: f64,
    c2_re: f64,
    v00: f64,
    leg: &LegendreCoeffsSimd,
    simd_threshold: usize,
) -> Result<(), GravityError> {
    legendre_fill_body!(v, w, n, x_c2, y_c2, z_c2, c2_re, v00, leg, simd_threshold);
    Ok(())
}

/// Monomorphised twin of [`legendre_vw_row_major_with_threshold`]: the same
/// body with the trip count as `const N` rather than an argument. Both arms
/// are stamped from [`legendre_fill_body!`], so their identity holds by
/// construction.
///
/// Why it exists: `n` is `max_order + 2`, a runtime value, so the loops cannot
/// unroll. At the production order that trip count is 7 — small enough that the
/// loop overhead is a real fraction of the fill, which runs essentially every
/// RHS evaluation (the V/W cache misses 99.4% of the time).
///
/// # Errors
///
/// Returns [`GravityError::InvariantViolation`] if the fixed workspace cannot
/// satisfy its checked matrix bounds.
#[inline]
fn legendre_vw_row_major_const<const N: usize>(
    v: &mut [[f64; MAX_RECURSIVE_ORDER]],
    w: &mut [[f64; MAX_RECURSIVE_ORDER]],
    x_c2: f64,
    y_c2: f64,
    z_c2: f64,
    c2_re: f64,
    v00: f64,
    leg: &LegendreCoeffsSimd,
    simd_threshold: usize,
) -> Result<(), GravityError> {
    legendre_fill_body!(v, w, N, x_c2, y_c2, z_c2, c2_re, v00, leg, simd_threshold);
    Ok(())
}

// These thresholds were `Lazy<T>` wrapping literals behind accessor functions.
// The former environment knobs are gone; direct constants preserve their
// compile-time values without an acquire load or branch on the hot path.

/// Harmonic orders >= this use the SIMD summation kernels.
///
/// Gates the vector lanes in [`gravity_summation_f64`]. It does NOT gate the
/// Legendre row fill -- that has its own threshold, because the reachable width
/// differs between the two and one number cannot be right for both. See
/// [`LEGENDRE_SIMD_L_THRESHOLD`].
const SIMD_L_THRESHOLD: usize = 8;

/// Legendre rows `l >= this` use the SIMD column fill.
///
/// Production calls the row fill with `n = sph_order + 2 = 7`, so its row loop
/// (`for l in 3..n`) tops out at `l = 6`. The quad loop inside
/// `legendre_l_row_simd` needs `m + 3 < l - 1`, i.e. `l >= 6`, so 6 is the
/// lowest threshold that changes anything and 7 already vectorises nothing.
/// At 8 -- the value this shared with the summation gate -- the SIMD branch was
/// unreachable at every order this project runs, and the function was scalar
/// despite its name.
///
/// Measured at `n = 7`: 4 of the 10 inner cells vectorise, in a single quad on
/// row 6.
const LEGENDRE_SIMD_L_THRESHOLD: usize = 6;

const GRAVITY_FAST_PATH_SPECIALIZATION_MAX: usize = 7;

/// Highest order served by the specialized low-order fast path.
///
/// Clamped to [`GRAVITY_FAST_PATH_SPECIALIZATION_MAX`], which is the highest
/// order that specialization actually exists for -- the clamp is the reason
/// this is not simply written as `5`.
///
/// Public because the specialization accumulates into four lanes and reduces,
/// so it does not sum in the same sequence as the unpacked kernel. A caller
/// comparing the two kernels has to know which orders take it; see
/// `lightyear_odeint_rs`'s `packed_and_unpacked_gravity_kernels_agree`.
pub const GRAVITY_FAST_PATH_ORDER_CAP: usize = if 5 < GRAVITY_FAST_PATH_SPECIALIZATION_MAX {
    5
} else {
    GRAVITY_FAST_PATH_SPECIALIZATION_MAX
};

pub const MAX_ORDER: usize = 128;
pub const MAX_RECURSIVE_ORDER: usize = MAX_ORDER + 3;

#[derive(Clone, Copy, Debug, PartialEq)]
struct PackedGravityTerm {
    /// Harmonic order `m`, strictly increasing within its degree row.
    m: usize,
    /// Normalized C(l,m) coefficient.
    c: f64,
    /// Normalized S(l,m) coefficient.
    s: f64,
    /// `(d + 2)(d + 1)`, where `d = l - m`.
    cf_2: f64,
    /// `d + 1`, where `d = l - m`.
    dm1: f64,
}

/// Four consecutive emitted terms of one degree row, held column-major.
///
/// Structure-of-arrays rather than `[PackedGravityTerm; 4]`. The packed binary64
/// summation consumes a quad as four `f64x4` vectors, one per coefficient field;
/// in the array-of-structs form each vector's lanes sat 40 bytes apart, so every
/// vector cost four scalar loads and three lane inserts. Each field is now
/// contiguous and each vector is a single load. The harmonic orders stay
/// together in `orders` because they index the V/W rows rather than feeding a
/// lane.
///
/// This layout is safe to change: quads are derived storage, a literal copy of
/// `terms.chunks_exact(4)` rebuilt by [`pack_gravity_coeffs`] and re-proven
/// against `terms` by [`PackedGravityCoeffs::validate_metadata`], and
/// [`PackedGravityCoeffs::authority_sha256`] hashes only `terms` -- it excludes
/// derived SIMD quads by documented intent.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PackedGravityQuad {
    /// Harmonic orders of the four lanes, strictly increasing.
    orders: [usize; 4],
    /// Normalized C(l,m) per lane.
    c: [f64; 4],
    /// Normalized S(l,m) per lane.
    s: [f64; 4],
    /// `(d + 2)(d + 1)` per lane, where `d = l - m`.
    cf_2: [f64; 4],
    /// `d + 1` per lane, where `d = l - m`.
    dm1: [f64; 4],
}

impl PackedGravityQuad {
    /// Transpose four emitted terms into the lane-major form.
    ///
    /// This is the only way a quad is ever built, so `pack_gravity_coeffs` and
    /// the validation that re-proves its output cannot drift apart.
    #[inline]
    const fn from_terms(terms: [PackedGravityTerm; 4]) -> Self {
        let [first, second, third, fourth] = terms;
        Self {
            orders: [first.m, second.m, third.m, fourth.m],
            c: [first.c, second.c, third.c, fourth.c],
            s: [first.s, second.s, third.s, fourth.s],
            cf_2: [first.cf_2, second.cf_2, third.cf_2, fourth.cf_2],
            dm1: [first.dm1, second.dm1, third.dm1, fourth.dm1],
        }
    }
}

#[derive(Clone, Debug)]
struct PackedGravityRow {
    /// C(l,0), retained with the terms it shares a recurrence row with.
    central_c: f64,
    /// Nonzero `m=1..=l` terms, in their original harmonic order.
    terms: Box<[PackedGravityTerm]>,
    /// Stable groups of four emitted terms, preserving the legacy SIMD
    /// reduction grouping without exposing raw coefficient storage.
    quads: Box<[PackedGravityQuad]>,
}

#[derive(Clone, Copy)]
enum VwCacheState {
    Empty,
    Ready {
        position_ecef: [f64; 3],
        covered_order: usize,
    },
}

#[derive(Clone)]
pub struct GravityCacheGeneric<T: Copy + Zero> {
    /// Highest harmonic order written into `v`/`w` since the last clear.
    ///
    /// `reset` clears only the square prefix a fill can reach, so this must
    /// cover every fill since the previous clear — not only the most recent
    /// one, and not only the ones that publish a `Ready` state. It therefore
    /// only ever rises WITHIN a fill cycle; letting it shrink there would
    /// strand a live cell from an earlier, wider fill in the same cycle.
    ///
    /// `reset` and `prime_storage` end the cycle and drop it back to zero. That
    /// is the same fact stated after the clear rather than before it: both leave
    /// the entire workspace zero, so "nothing has been written since" is true,
    /// and the next `begin_vw_fill` raises the mark before any write lands.
    /// Carrying a stale wide mark across a clear would be sound but wasteful —
    /// a single cold-path order-20 fill would make every later order-5 reset
    /// clear 22x22 instead of 7x7, for the life of the cache.
    ///
    /// Starts at zero, which is correct rather than merely cheap: a cache that
    /// has never been filled is entirely zero from construction, so there is
    /// nothing outside the initial span for `reset` to clear.
    vw_high_water: usize,
    v: Box<[[T; MAX_RECURSIVE_ORDER]]>,
    w: Box<[[T; MAX_RECURSIVE_ORDER]]>,
    /// All recurrence validity fields move together, so no caller can expose
    /// a workspace as ready after changing only its position or order.
    vw_state: VwCacheState,
    /// Fixed precomputed Legendre recurrence coefficients (eliminates per-step divisions).
    ///
    /// Borrowed from the process-wide table: read-only, identical for every
    /// cache, and independent of `T`.
    legendre_coeffs: &'static LegendreCoeffsSimd,
}

pub type GravityCache = GravityCacheGeneric<f64>;

#[derive(Clone, Debug)]
pub struct PackedGravityCoeffs {
    rows: Box<[PackedGravityRow]>,
    /// Highest harmonic row eligible for dense positional dispatch.
    dense_prefix: usize,
    contains_noncentral: bool,
    max_order: usize,
    /// Exact authority digest of this immutable pack. The private coefficient
    /// storage cannot change after construction, so one successful validation
    /// and hash serves every batch thread without weakening identity.
    authority_sha256_cache: OnceLock<Result<[u8; 32], GravityError>>,
}

impl PackedGravityCoeffs {
    /// Hash the validated mathematical coefficient authority.
    ///
    /// The canonical byte stream is domain-separated and uses big-endian
    /// fixed-width integers: maximum order, then each row's degree, C(l,0)
    /// bits, term count, and ordered `(m, C(l,m), S(l,m))` bits. Derived SIMD
    /// quads and dispatch metadata are deliberately excluded.
    ///
    /// # Errors
    ///
    /// Returns [`GravityError::InvariantViolation`] if private packed metadata
    /// fails validation or cannot be represented by the canonical format.
    pub fn authority_sha256(&self) -> Result<[u8; 32], GravityError> {
        *self
            .authority_sha256_cache
            .get_or_init(|| self.compute_authority_sha256())
    }

    fn compute_authority_sha256(&self) -> Result<[u8; 32], GravityError> {
        self.validate_metadata()?;

        let mut hasher = Sha256::new();
        hasher.update(b"nasa-dust/satpy-core/packed-gravity-authority/v1\0");
        let max_order =
            u64::try_from(self.max_order).map_err(|_| GravityError::InvariantViolation)?;
        hasher.update(max_order.to_be_bytes());

        for (degree, row) in self.rows.iter().enumerate() {
            let degree = u64::try_from(degree).map_err(|_| GravityError::InvariantViolation)?;
            let term_count =
                u64::try_from(row.terms.len()).map_err(|_| GravityError::InvariantViolation)?;
            hasher.update(degree.to_be_bytes());
            hasher.update(row.central_c.to_bits().to_be_bytes());
            hasher.update(term_count.to_be_bytes());
            for term in &row.terms {
                let order = u64::try_from(term.m).map_err(|_| GravityError::InvariantViolation)?;
                hasher.update(order.to_be_bytes());
                hasher.update(term.c.to_bits().to_be_bytes());
                hasher.update(term.s.to_bits().to_be_bytes());
            }
        }

        Ok(hasher.finalize().into())
    }

    /// Whether the degree-one gravity contribution is present.
    ///
    /// This checks C10, C11, and S11 with the authority's strict `abs() >
    /// 1e-18` threshold so callers can select an analytic first-order path
    /// without retaining raw arrays.
    #[must_use]
    pub fn has_nonzero_degree1_terms(&self) -> bool {
        self.rows.get(1).is_some_and(|row| {
            row.central_c.abs() > 1.0e-18
                || row
                    .terms
                    .iter()
                    .any(|term| term.c.abs() > 1.0e-18 || term.s.abs() > 1.0e-18)
        })
    }

    /// Highest harmonic degree encoded by this immutable pack.
    #[must_use]
    pub const fn max_order(&self) -> usize {
        self.max_order
    }

    /// Return a proven packed prefix through `order`.
    ///
    /// # Errors
    ///
    /// Returns [`GravityError::UnsupportedOrder`] when `order` would widen this
    /// immutable coefficient pack, or [`GravityError::InvariantViolation`] if
    /// its private metadata cannot be revalidated.
    pub fn truncated_to(&self, order: usize) -> Result<Self, GravityError> {
        self.validate_metadata()?;
        if order > self.max_order {
            return Err(GravityError::UnsupportedOrder);
        }
        let row_count = order
            .checked_add(1)
            .ok_or(GravityError::InvariantViolation)?;
        let rows = self
            .rows
            .get(..row_count)
            .ok_or(GravityError::InvariantViolation)?;
        let mut dense_prefix = 0usize;
        let mut dense_prefix_open = true;
        let mut contains_noncentral = false;
        for (degree, row) in rows.iter().enumerate() {
            contains_noncentral |= row.terms.iter().any(|term| term.c != 0.0 || term.s != 0.0);
            let dense = row.terms.len() == degree;
            if dense_prefix_open && (dense || row.terms.is_empty()) {
                dense_prefix = degree;
            } else {
                dense_prefix_open = false;
            }
        }
        let truncated = Self {
            rows: rows.to_vec().into_boxed_slice(),
            dense_prefix,
            contains_noncentral,
            max_order: order,
            authority_sha256_cache: OnceLock::new(),
        };
        truncated.validate_metadata()?;
        Ok(truncated)
    }

    fn validate_metadata(&self) -> Result<(), GravityError> {
        let row_count = self
            .max_order
            .checked_add(1)
            .ok_or(GravityError::InvariantViolation)?;
        if self.rows.len() != row_count
            || self.dense_prefix > self.max_order
            || self.max_order > MAX_ORDER
        {
            return Err(GravityError::InvariantViolation);
        }
        let mut expected_dense_prefix = 0usize;
        let mut dense_prefix_open = true;
        let mut expected_contains_noncentral = false;
        for (degree, row) in self.rows.iter().enumerate() {
            if !crate::safe_isfinite(row.central_c)
                || row.terms.iter().any(|term| {
                    !crate::safe_isfinite(term.c)
                        || !crate::safe_isfinite(term.s)
                        || !crate::safe_isfinite(term.cf_2)
                        || !crate::safe_isfinite(term.dm1)
                })
            {
                return Err(GravityError::InvariantViolation);
            }
            let mut previous_m = 0usize;
            for term in &row.terms {
                if term.m == 0 || term.m > degree || term.m <= previous_m {
                    return Err(GravityError::InvariantViolation);
                }
                if term.c == 0.0 && term.s == 0.0 {
                    return Err(GravityError::InvariantViolation);
                }
                let degree_minus_order = degree
                    .checked_sub(term.m)
                    .ok_or(GravityError::InvariantViolation)?
                    .to_f64()
                    .ok_or(GravityError::InvariantViolation)?;
                let expected_cf_2 = (degree_minus_order + 2.0) * (degree_minus_order + 1.0);
                let expected_dm1 = degree_minus_order + 1.0;
                if term.cf_2.to_bits() != expected_cf_2.to_bits()
                    || term.dm1.to_bits() != expected_dm1.to_bits()
                {
                    return Err(GravityError::InvariantViolation);
                }
                previous_m = term.m;
                expected_contains_noncentral |= term.c != 0.0 || term.s != 0.0;
            }
            if row.quads.len() != row.terms.len() / 4 {
                return Err(GravityError::InvariantViolation);
            }
            for (quad, terms) in row.quads.iter().zip(row.terms.chunks_exact(4)) {
                let [first, second, third, fourth] = terms else {
                    return Err(GravityError::InvariantViolation);
                };
                if *quad != PackedGravityQuad::from_terms([*first, *second, *third, *fourth]) {
                    return Err(GravityError::InvariantViolation);
                }
            }
            let dense = row.terms.len() == degree;
            if dense_prefix_open && (dense || row.terms.is_empty()) {
                expected_dense_prefix = degree;
            } else {
                dense_prefix_open = false;
            }
        }
        if self.dense_prefix != expected_dense_prefix
            || self.contains_noncentral != expected_contains_noncentral
        {
            return Err(GravityError::InvariantViolation);
        }
        Ok(())
    }
}

/// Rows every recurrence fill writes regardless of the order it was asked for.
///
/// All four fill kernels open with unconditional `matrix_set`s at rows 0 and 1
/// (`v[0][0]`, `w[0][0]`, `w[1][0]`, `v[1][0]`) before any loop bounded by `n`
/// runs, so a workspace shorter than this errors on EVERY fill rather than only
/// on wide ones. It is the floor [`GravityCacheGeneric::with_rows`] clamps up
/// to; it is not a claim that two rows suffice for any particular order.
const VW_MIN_ROWS: usize = 2;

impl<T: Copy + Zero> GravityCacheGeneric<T> {
    /// Full-width workspace: every order up to [`MAX_ORDER`] fits.
    ///
    /// Benches, oracles and the autodiff paths construct through here and are
    /// not told an order in advance, so this deliberately keeps the
    /// `MAX_RECURSIVE_ORDER`-row allocation. Callers that DO know their order
    /// should use [`Self::with_rows`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_rows(MAX_RECURSIVE_ORDER)
    }

    /// Workspace sized to `rows` recurrence rows instead of the full width.
    ///
    /// A fill for `order` writes only inside the `n x n` square with
    /// `n = order + 2`, so a caller with a fixed order needs `order + 2` rows
    /// and nothing more. `rows` is clamped into
    /// `[VW_MIN_ROWS, MAX_RECURSIVE_ORDER]`.
    ///
    /// The row count is a slice length while the row LENGTH is fixed by the
    /// array type, so only the row count can shrink; a short cache still holds
    /// full-width rows.
    ///
    /// Undersizing is a typed error, never unsoundness: every workspace access
    /// goes through the checked `matrix_*`/`nested_value` helpers, so a fill
    /// wider than the allocation returns [`GravityError::InvariantViolation`].
    ///
    /// Zero-initialising `vec![]` is load-bearing and must stay: [`Self::reset`]
    /// clears only the live prefix and depends on every cell outside it still
    /// holding its constructed zero.
    #[must_use]
    pub fn with_rows(rows: usize) -> Self {
        // `clamp` panics when max < min; both bounds are compile-time constants
        // with `VW_MIN_ROWS = 2 <= 131 = MAX_RECURSIVE_ORDER`, so it cannot.
        let rows = rows.clamp(VW_MIN_ROWS, MAX_RECURSIVE_ORDER);
        Self {
            vw_high_water: 0,
            v: vec![[T::zero(); MAX_RECURSIVE_ORDER]; rows].into_boxed_slice(),
            w: vec![[T::zero(); MAX_RECURSIVE_ORDER]; rows].into_boxed_slice(),
            vw_state: VwCacheState::Empty,
            legendre_coeffs: LegendreCoeffsSimd::shared(),
        }
    }

    #[inline]
    const fn invalidate_vw(&mut self) {
        self.vw_state = VwCacheState::Empty;
    }

    /// Raise the clear bound to cover a fill of `covered_order`.
    ///
    /// Never lowers it within a fill cycle. `reset` and `prime_storage` end the
    /// cycle and reset the bound themselves.
    #[inline]
    fn raise_vw_high_water(&mut self, covered_order: usize) {
        self.vw_high_water = self.vw_high_water.max(covered_order);
    }

    /// Drop any validity claim and record the width the fill about to run needs.
    ///
    /// Every kernel that writes `v`/`w` enters through here, including the ones
    /// that deliberately never publish a `Ready` state, so the clear bound
    /// cannot miss a fill.
    #[inline]
    fn begin_vw_fill(&mut self, covered_order: usize) {
        self.invalidate_vw();
        self.raise_vw_high_water(covered_order);
    }

    /// Rows and columns a fill at the recorded high-water order can reach.
    ///
    /// A fill for `order` runs the recurrence over `n = order + 2` rows and
    /// writes only within that `n x n` square, so clearing the square clears
    /// every cell any fill since the last clear has written.
    ///
    /// Clamped to the REAL row count, not to `MAX_RECURSIVE_ORDER`: a cache
    /// built by [`Self::with_rows`] is shorter than the constant, and a fill
    /// that overran it errored out rather than writing, so there is nothing
    /// past `self.v.len()` to clear.
    #[inline]
    fn vw_live_span(&self) -> usize {
        self.vw_high_water.saturating_add(2).min(self.v.len())
    }

    #[inline]
    const fn reuses_vw(&self, position_ecef: [f64; 3], order: usize) -> bool {
        matches!(
            self.vw_state,
            VwCacheState::Ready {
                position_ecef: cached_position,
                covered_order,
            } if same_position_bits(position_ecef, cached_position) && covered_order >= order
        )
    }

    #[inline]
    fn mark_vw_ready(&mut self, position_ecef: [f64; 3], covered_order: usize) {
        self.raise_vw_high_water(covered_order);
        self.vw_state = VwCacheState::Ready {
            position_ecef,
            covered_order,
        };
    }

    /// Clear this cache's recurrence workspace and validity state.
    ///
    /// Gravity evaluators manage their own exact-position reuse internally.
    ///
    /// Only the square prefix a fill can reach is written. That is not an
    /// approximation of the old full clear, it is equal to it: every cell
    /// outside the prefix still holds the zero it was constructed with, because
    /// the high-water mark covers every fill since the previous clear. It is
    /// bounded because this is called once per Encke rectification segment and
    /// once per eclipse root transaction, while the sealed order-5 prefix is
    /// 7x7 out of the allocation (131x131 from [`Self::new`], 7x131 from
    /// [`Self::with_rows`] at the sealed order).
    ///
    /// A fill wider than the allocation cannot defeat this. It errors inside the
    /// `m = 0` column loop the moment it reaches the first missing row, before
    /// the diagonal and inner loops that are the only writers of columns past 0,
    /// so its residue is confined to column 0 of the rows that DO exist — inside
    /// the square this clears.
    ///
    /// The bound is dropped afterwards, so a cache that ran one wide fill does
    /// not keep paying for it: the whole workspace is zero once the loops below
    /// finish, which is exactly what a zero mark asserts.
    ///
    /// Use [`Self::prime_storage`] when the point is to touch the pages rather
    /// than to clear the live values.
    pub fn reset(&mut self) {
        self.invalidate_vw();
        let live = self.vw_live_span();
        for row in self.v.iter_mut().take(live) {
            if let Some(prefix) = row.get_mut(..live) {
                prefix.fill(T::zero());
            }
        }
        for row in self.w.iter_mut().take(live) {
            if let Some(prefix) = row.get_mut(..live) {
                prefix.fill(T::zero());
            }
        }
        self.vw_high_water = 0;
    }

    /// Write every recurrence page, then clear validity state.
    ///
    /// [`Self::reset`] is the hot-path clear and is deliberately bounded to the
    /// live prefix. This is its page-touching sibling: worker priming and the
    /// resident-set plateau probe exist to fault in and write the whole bounded
    /// allocation, which a prefix clear no longer does.
    ///
    /// Ends the fill cycle and drops the clear bound, same as [`Self::reset`].
    pub fn prime_storage(&mut self) {
        self.invalidate_vw();
        for row in &mut self.v {
            row.fill(T::zero());
        }
        for row in &mut self.w {
            row.fill(T::zero());
        }
        self.vw_high_water = 0;
    }
}

#[inline]
const fn same_position_bits(position: [f64; 3], cached: [f64; 3]) -> bool {
    let [position_x, position_y, position_z] = position;
    let [cached_x, cached_y, cached_z] = cached;
    position_x.to_bits() == cached_x.to_bits()
        && position_y.to_bits() == cached_y.to_bits()
        && position_z.to_bits() == cached_z.to_bits()
}

impl<T: Copy + Zero> Default for GravityCacheGeneric<T> {
    fn default() -> Self {
        Self::new()
    }
}

use std::cell::RefCell;
thread_local! { static THREAD_GRAVITY_CACHE: RefCell<GravityCache> = RefCell::new(GravityCache::new()); }

#[inline]
fn with_gravity_cache<F, R>(f: F) -> R
where
    F: FnOnce(&mut GravityCache) -> R,
{
    THREAD_GRAVITY_CACHE.with(|cache| f(&mut cache.borrow_mut()))
}

/// Touch bounded gravity storage on this thread, then clear semantic state.
///
/// Deliberately NOT `reset`: that clear is bounded to the live recurrence
/// prefix, so routing priming through it would leave most of the allocation
/// untouched and quietly turn this into a no-op.
pub fn prime_thread_gravity_cache() {
    with_gravity_cache(GravityCache::prime_storage);
}

/// Validate flat square harmonic coefficient matrices before packing or evaluation.
///
/// # Errors
///
/// Returns an error when coefficient dimensions, finiteness, or requested order
/// do not satisfy the gravity kernel's bounds.
pub fn validate_flat_gravity_coeffs(
    c_coeffs: &[f64],
    s_coeffs: &[f64],
    order: usize,
) -> Result<usize, GravityError> {
    if c_coeffs.is_empty() {
        return Err(GravityError::InvalidCoefficientStorage);
    }
    let length_as_f64 = c_coeffs
        .len()
        .to_f64()
        .ok_or(GravityError::InvalidCoefficientStorage)?;
    let stride = length_as_f64
        .sqrt()
        .to_usize()
        .ok_or(GravityError::InvalidCoefficientStorage)?;
    if stride.checked_mul(stride) != Some(c_coeffs.len()) {
        return Err(GravityError::InvalidCoefficientStorage);
    }
    validate_gravity_coeffs_with_stride(c_coeffs, s_coeffs, stride, order)?;
    Ok(stride)
}

fn validate_gravity_coeffs_with_stride(
    c_coeffs: &[f64],
    s_coeffs: &[f64],
    stride: usize,
    order: usize,
) -> Result<(), GravityError> {
    if order > MAX_ORDER {
        return Err(GravityError::UnsupportedOrder);
    }
    if stride == 0 {
        return Err(GravityError::InvalidCoefficientStorage);
    }
    let expected_len = stride
        .checked_mul(stride)
        .ok_or(GravityError::InvalidCoefficientStorage)?;
    if c_coeffs.len() != expected_len || s_coeffs.len() != expected_len {
        return Err(GravityError::InvalidCoefficientStorage);
    }
    if order >= stride {
        return Err(GravityError::InvalidCoefficientStorage);
    }
    if c_coeffs.iter().any(|value| !crate::safe_isfinite(*value)) {
        return Err(GravityError::InvalidCoefficientStorage);
    }
    if s_coeffs.iter().any(|value| !crate::safe_isfinite(*value)) {
        return Err(GravityError::InvalidCoefficientStorage);
    }
    Ok(())
}

/// Validate and pack harmonic coefficients for the gravity kernels.
///
/// # Errors
///
/// Returns an error when the supplied coefficient storage cannot prove the
/// shape, finiteness, or harmonic-order invariants required by packed gravity.
pub fn pack_gravity_coeffs(
    c_coeffs: &[f64],
    s_coeffs: &[f64],
    stride: usize,
    order: usize,
) -> Result<PackedGravityCoeffs, GravityError> {
    validate_gravity_coeffs_with_stride(c_coeffs, s_coeffs, stride, order)?;
    let row_count = order
        .checked_add(1)
        .ok_or(GravityError::InvariantViolation)?;
    let mut rows = Vec::with_capacity(row_count);
    let mut contains_noncentral = false;
    let mut dense_prefix = 0usize;
    let mut dense_prefix_open = true;
    for degree in 0..=order {
        let base = degree
            .checked_mul(stride)
            .ok_or(GravityError::InvariantViolation)?;
        let row_end = base
            .checked_add(stride)
            .ok_or(GravityError::InvariantViolation)?;
        let c_row = c_coeffs
            .get(base..row_end)
            .ok_or(GravityError::InvariantViolation)?;
        let s_row = s_coeffs
            .get(base..row_end)
            .ok_or(GravityError::InvariantViolation)?;
        let central_c = c_row
            .first()
            .copied()
            .ok_or(GravityError::InvariantViolation)?;
        let mut terms = Vec::with_capacity(degree);
        for (offset, (&c, &s)) in c_row
            .iter()
            .skip(1)
            .zip(s_row.iter().skip(1))
            .take(degree)
            .enumerate()
        {
            let m = offset
                .checked_add(1)
                .ok_or(GravityError::InvariantViolation)?;
            let degree_minus_order = degree
                .checked_sub(m)
                .ok_or(GravityError::InvariantViolation)?
                .to_f64()
                .ok_or(GravityError::InvariantViolation)?;
            if c == 0.0 && s == 0.0 {
                continue;
            }
            contains_noncentral |= c != 0.0 || s != 0.0;
            terms.push(PackedGravityTerm {
                m,
                c,
                s,
                cf_2: (degree_minus_order + 2.0) * (degree_minus_order + 1.0),
                dm1: degree_minus_order + 1.0,
            });
        }
        let mut quads = Vec::with_capacity(terms.len() / 4);
        for chunk in terms.chunks_exact(4) {
            let [first, second, third, fourth] = chunk else {
                return Err(GravityError::InvariantViolation);
            };
            quads.push(PackedGravityQuad::from_terms([
                *first, *second, *third, *fourth,
            ]));
        }
        let dense = terms.len() == degree;
        if dense_prefix_open && (dense || terms.is_empty()) {
            dense_prefix = degree;
        } else {
            dense_prefix_open = false;
        }
        rows.push(PackedGravityRow {
            central_c,
            terms: terms.into_boxed_slice(),
            quads: quads.into_boxed_slice(),
        });
    }
    let packed = PackedGravityCoeffs {
        rows: rows.into_boxed_slice(),
        dense_prefix,
        contains_noncentral,
        max_order: order,
        authority_sha256_cache: OnceLock::new(),
    };
    packed.validate_metadata()?;
    Ok(packed)
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "preserve the packed generic gravity summation's floating-point operation order"
)]
fn gravity_summation_generic_packed<T: Float + FromPrimitive>(
    v: &[[T; MAX_RECURSIVE_ORDER]],
    w: &[[T; MAX_RECURSIVE_ORDER]],
    coef_sph: T,
    packed: &PackedGravityCoeffs,
) -> Result<(T, T, T), GravityError> {
    let half = T::from_f64(0.5).ok_or(GravityError::InvariantViolation)?;
    let mut first_axis = T::zero();
    let mut second_axis = T::zero();
    let mut third_axis = T::zero();
    let mut degree_plus_one = T::one();

    for ((row, v_row), w_row) in packed
        .rows
        .iter()
        .zip(v.iter().skip(1))
        .zip(w.iter().skip(1))
    {
        let (v0, v1) = two_values(v_row)?;
        let (_, w1) = two_values(w_row)?;
        let central_c = T::from_f64(row.central_c).ok_or(GravityError::InvariantViolation)?;
        if !central_c.is_zero() {
            first_axis = first_axis + coef_sph * (-central_c * v1);
            second_axis = second_axis + coef_sph * (-central_c * w1);
            third_axis = third_axis + coef_sph * (degree_plus_one * (-central_c * v0));
        }

        for term in &row.terms {
            let (v_below, v_same, v_above) = packed_row_window(v_row, term.m)?;
            let (w_below, w_same, w_above) = packed_row_window(w_row, term.m)?;
            let c = T::from_f64(term.c).ok_or(GravityError::InvariantViolation)?;
            let s = T::from_f64(term.s).ok_or(GravityError::InvariantViolation)?;
            let cf_2 = T::from_f64(term.cf_2).ok_or(GravityError::InvariantViolation)?;
            let dm1 = T::from_f64(term.dm1).ok_or(GravityError::InvariantViolation)?;
            let x1 = (-c).mul_add(
                v_above,
                (-s).mul_add(w_above, cf_2 * c.mul_add(v_below, s * w_below)),
            );
            let y1 = (-c).mul_add(
                w_above,
                s.mul_add(v_above, cf_2 * (-c).mul_add(w_below, s * v_below)),
            );
            let z1 = dm1 * (-c).mul_add(v_same, -s * w_same);
            first_axis = first_axis + coef_sph * half * x1;
            second_axis = second_axis + coef_sph * half * y1;
            third_axis = third_axis + coef_sph * z1;
        }
        degree_plus_one = degree_plus_one + T::one();
    }
    Ok((first_axis, second_axis, third_axis))
}

#[inline]
fn gravity_summation_f64_packed(
    v: &[[f64; MAX_RECURSIVE_ORDER]],
    w: &[[f64; MAX_RECURSIVE_ORDER]],
    coef_sph: f64,
    packed: &PackedGravityCoeffs,
) -> Result<(f64, f64, f64), GravityError> {
    let coef_sph_half = coef_sph * 0.5;
    let coef_sph_half_v = f64x4::splat(coef_sph_half);
    let coef_sph_v = f64x4::splat(coef_sph);
    let mut first_axis_vector = f64x4::ZERO;
    let mut second_axis_vector = f64x4::ZERO;
    let mut third_axis_vector = f64x4::ZERO;
    let mut first_axis_scalar = 0.0f64;
    let mut second_axis_scalar = 0.0f64;
    let mut third_axis_scalar = 0.0f64;
    let low_order_dense =
        packed.max_order <= GRAVITY_FAST_PATH_ORDER_CAP && packed.dense_prefix >= packed.max_order;

    for (degree, ((row, v_row), w_row)) in packed
        .rows
        .iter()
        .zip(v.iter().skip(1))
        .zip(w.iter().skip(1))
        .enumerate()
    {
        let (v0, v1) = two_values(v_row)?;
        let (_, w1) = two_values(w_row)?;
        if row.central_c != 0.0 {
            first_axis_scalar += coef_sph * (-row.central_c * v1);
            second_axis_scalar += coef_sph * (-row.central_c * w1);
            let degree_plus_one = degree
                .checked_add(1)
                .ok_or(GravityError::InvariantViolation)?
                .to_f64()
                .ok_or(GravityError::InvariantViolation)?;
            third_axis_scalar += coef_sph * (degree_plus_one * (-row.central_c * v0));
        }

        let use_quads = low_order_dense || degree >= SIMD_L_THRESHOLD;
        let quad_count = if use_quads { row.quads.len() } else { 0 };
        for quad in row.quads.iter().take(quad_count) {
            let [first, second, third, fourth] = quad.orders;
            let first_coordinates = packed_term_coordinates(v_row, w_row, first)?;
            let second_coordinates = packed_term_coordinates(v_row, w_row, second)?;
            let third_coordinates = packed_term_coordinates(v_row, w_row, third)?;
            let fourth_coordinates = packed_term_coordinates(v_row, w_row, fourth)?;
            let coefficients_c = f64x4::new(quad.c);
            let coefficients_s = f64x4::new(quad.s);
            let coefficient_factor = f64x4::new(quad.cf_2);
            let degree_minus_one = f64x4::new(quad.dm1);
            let v_below = f64x4::new([
                first_coordinates.v_below,
                second_coordinates.v_below,
                third_coordinates.v_below,
                fourth_coordinates.v_below,
            ]);
            let v_same = f64x4::new([
                first_coordinates.v_same,
                second_coordinates.v_same,
                third_coordinates.v_same,
                fourth_coordinates.v_same,
            ]);
            let v_above = f64x4::new([
                first_coordinates.v_above,
                second_coordinates.v_above,
                third_coordinates.v_above,
                fourth_coordinates.v_above,
            ]);
            let w_below = f64x4::new([
                first_coordinates.w_below,
                second_coordinates.w_below,
                third_coordinates.w_below,
                fourth_coordinates.w_below,
            ]);
            let w_same = f64x4::new([
                first_coordinates.w_same,
                second_coordinates.w_same,
                third_coordinates.w_same,
                fourth_coordinates.w_same,
            ]);
            let w_above = f64x4::new([
                first_coordinates.w_above,
                second_coordinates.w_above,
                third_coordinates.w_above,
                fourth_coordinates.w_above,
            ]);
            let x1 = (-coefficients_c).mul_add(
                v_above,
                (-coefficients_s).mul_add(
                    w_above,
                    coefficient_factor * coefficients_c.mul_add(v_below, coefficients_s * w_below),
                ),
            );
            let y1 = (-coefficients_c).mul_add(
                w_above,
                coefficients_s.mul_add(
                    v_above,
                    coefficient_factor
                        * (-coefficients_c).mul_add(w_below, coefficients_s * v_below),
                ),
            );
            let z1 = degree_minus_one * (-coefficients_c).mul_add(v_same, -coefficients_s * w_same);
            first_axis_vector += coef_sph_half_v * x1;
            second_axis_vector += coef_sph_half_v * y1;
            third_axis_vector += coef_sph_v * z1;
        }
        let scalar_start = quad_count
            .checked_mul(4)
            .ok_or(GravityError::InvariantViolation)?;
        for term in row.terms.iter().skip(scalar_start) {
            let (v_below, v_same, v_above) = packed_row_window(v_row, term.m)?;
            let (w_below, w_same, w_above) = packed_row_window(w_row, term.m)?;
            let x1 = (-term.c).mul_add(
                v_above,
                (-term.s).mul_add(
                    w_above,
                    term.cf_2 * term.c.mul_add(v_below, term.s * w_below),
                ),
            );
            let y1 = (-term.c).mul_add(
                w_above,
                term.s.mul_add(
                    v_above,
                    term.cf_2 * (-term.c).mul_add(w_below, term.s * v_below),
                ),
            );
            let z1 = term.dm1 * (-term.c).mul_add(v_same, -term.s * w_same);
            first_axis_scalar += coef_sph_half * x1;
            second_axis_scalar += coef_sph_half * y1;
            third_axis_scalar += coef_sph * z1;
        }
    }
    Ok((
        first_axis_vector.reduce_add() + first_axis_scalar,
        second_axis_vector.reduce_add() + second_axis_scalar,
        third_axis_vector.reduce_add() + third_axis_scalar,
    ))
}

/// Validated raw-coefficient oracle summation.
///
/// This path deliberately retains every emitted raw term and scalar reduction
/// sequence. It is the accuracy reference for packed dispatch, whose vector
/// lane reductions are not bit-identical at every supported order.
#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "preserve the raw generic gravity summation's floating-point operation order"
)]
fn gravity_summation_generic_raw<T: Float + FromPrimitive>(
    v_workspace: &[[T; MAX_RECURSIVE_ORDER]],
    w_workspace: &[[T; MAX_RECURSIVE_ORDER]],
    c_coeffs: &[f64],
    s_coeffs: &[f64],
    stride: usize,
    order: usize,
    coef_sph: T,
) -> Result<(T, T, T), GravityError> {
    let half = T::from_f64(0.5).ok_or(GravityError::InvariantViolation)?;
    let two = T::from_f64(2.0).ok_or(GravityError::InvariantViolation)?;
    let mut first_axis = T::zero();
    let mut second_axis = T::zero();
    let mut third_axis = T::zero();

    for degree in 0..=order {
        let base = degree
            .checked_mul(stride)
            .ok_or(GravityError::InvariantViolation)?;
        let row_end = base
            .checked_add(stride)
            .ok_or(GravityError::InvariantViolation)?;
        let c_row = c_coeffs
            .get(base..row_end)
            .ok_or(GravityError::InvariantViolation)?;
        let s_row = s_coeffs
            .get(base..row_end)
            .ok_or(GravityError::InvariantViolation)?;
        let row_index = degree
            .checked_add(1)
            .ok_or(GravityError::InvariantViolation)?;
        let v_row = v_workspace
            .get(row_index)
            .ok_or(GravityError::InvariantViolation)?;
        let w_row = w_workspace
            .get(row_index)
            .ok_or(GravityError::InvariantViolation)?;
        let (v0, v1) = two_values(v_row)?;
        let (_, w1) = two_values(w_row)?;
        let central_c = T::from_f64(
            c_row
                .first()
                .copied()
                .ok_or(GravityError::InvariantViolation)?,
        )
        .ok_or(GravityError::InvariantViolation)?;
        let degree_plus_one = degree
            .checked_add(1)
            .ok_or(GravityError::InvariantViolation)?
            .to_f64()
            .ok_or(GravityError::InvariantViolation)?;
        let degree_plus_one =
            T::from_f64(degree_plus_one).ok_or(GravityError::InvariantViolation)?;
        first_axis = first_axis + coef_sph * (-central_c * v1);
        second_axis = second_axis + coef_sph * (-central_c * w1);
        third_axis = third_axis + coef_sph * (degree_plus_one * (-central_c * v0));

        for (offset, (&coefficient_c, &coefficient_s)) in c_row
            .iter()
            .skip(1)
            .zip(s_row.iter().skip(1))
            .take(degree)
            .enumerate()
        {
            let harmonic_order = offset
                .checked_add(1)
                .ok_or(GravityError::InvariantViolation)?;
            let start = harmonic_order
                .checked_sub(1)
                .ok_or(GravityError::InvariantViolation)?;
            let (v_below, v_same, v_above) =
                three_values(v_row.get(start..).ok_or(GravityError::InvariantViolation)?)?;
            let (w_below, w_same, w_above) =
                three_values(w_row.get(start..).ok_or(GravityError::InvariantViolation)?)?;
            let coefficient_c =
                T::from_f64(coefficient_c).ok_or(GravityError::InvariantViolation)?;
            let coefficient_s =
                T::from_f64(coefficient_s).ok_or(GravityError::InvariantViolation)?;
            let degree_minus_order = degree
                .checked_sub(harmonic_order)
                .ok_or(GravityError::InvariantViolation)?
                .to_f64()
                .ok_or(GravityError::InvariantViolation)?;
            let degree_minus_order =
                T::from_f64(degree_minus_order).ok_or(GravityError::InvariantViolation)?;
            let cf_2 = (degree_minus_order + two) * (degree_minus_order + T::one());
            let x1 = (-coefficient_c).mul_add(
                v_above,
                (-coefficient_s).mul_add(
                    w_above,
                    cf_2 * coefficient_c.mul_add(v_below, coefficient_s * w_below),
                ),
            );
            let y1 = (-coefficient_c).mul_add(
                w_above,
                coefficient_s.mul_add(
                    v_above,
                    cf_2 * (-coefficient_c).mul_add(w_below, coefficient_s * v_below),
                ),
            );
            let dm1 = degree_minus_order + T::one();
            let z1 = dm1 * (-coefficient_c).mul_add(v_same, -coefficient_s * w_same);
            first_axis = first_axis + coef_sph * half * x1;
            second_axis = second_axis + coef_sph * half * y1;
            third_axis = third_axis + coef_sph * z1;
        }
    }
    Ok((first_axis, second_axis, third_axis))
}

/// Validated raw-coefficient binary64 oracle summation.
///
/// The vector and scalar accumulators intentionally follow the established
/// raw recurrence order. Do not route this through [`PackedGravityCoeffs`]:
/// different lane reductions are observable from order five onward.
#[inline]
fn gravity_summation_f64_raw_quad(
    c_row: &[f64],
    s_row: &[f64],
    v_row: &[f64; MAX_RECURSIVE_ORDER],
    w_row: &[f64; MAX_RECURSIVE_ORDER],
    degree: usize,
    harmonic_order: usize,
) -> Result<(f64x4, f64x4, f64x4), GravityError> {
    let coefficients_c = f64x4::new(four_values(
        c_row
            .get(harmonic_order..)
            .ok_or(GravityError::InvariantViolation)?,
    )?);
    let coefficients_s = f64x4::new(four_values(
        s_row
            .get(harmonic_order..)
            .ok_or(GravityError::InvariantViolation)?,
    )?);
    let above_start = harmonic_order
        .checked_add(1)
        .ok_or(GravityError::InvariantViolation)?;
    let v_above = f64x4::new(four_values(
        v_row
            .get(above_start..)
            .ok_or(GravityError::InvariantViolation)?,
    )?);
    let w_above = f64x4::new(four_values(
        w_row
            .get(above_start..)
            .ok_or(GravityError::InvariantViolation)?,
    )?);
    let v_same = f64x4::new(four_values(
        v_row
            .get(harmonic_order..)
            .ok_or(GravityError::InvariantViolation)?,
    )?);
    let w_same = f64x4::new(four_values(
        w_row
            .get(harmonic_order..)
            .ok_or(GravityError::InvariantViolation)?,
    )?);
    let below_start = harmonic_order
        .checked_sub(1)
        .ok_or(GravityError::InvariantViolation)?;
    let v_below = f64x4::new(four_values(
        v_row
            .get(below_start..)
            .ok_or(GravityError::InvariantViolation)?,
    )?);
    let w_below = f64x4::new(four_values(
        w_row
            .get(below_start..)
            .ok_or(GravityError::InvariantViolation)?,
    )?);
    let degree_offset_zero = degree
        .checked_sub(harmonic_order)
        .ok_or(GravityError::InvariantViolation)?
        .to_f64()
        .ok_or(GravityError::InvariantViolation)?;
    let degree_offsets = f64x4::new([
        degree_offset_zero,
        degree_offset_zero - 1.0,
        degree_offset_zero - 2.0,
        degree_offset_zero - 3.0,
    ]);
    let coefficient_factor = (degree_offsets + TWO_X4) * (degree_offsets + f64x4::ONE);
    let x1 = (-coefficients_c).mul_add(
        v_above,
        (-coefficients_s).mul_add(
            w_above,
            coefficient_factor * coefficients_c.mul_add(v_below, coefficients_s * w_below),
        ),
    );
    let y1 = (-coefficients_c).mul_add(
        w_above,
        coefficients_s.mul_add(
            v_above,
            coefficient_factor * (-coefficients_c).mul_add(w_below, coefficients_s * v_below),
        ),
    );
    let z1 =
        (degree_offsets + f64x4::ONE) * (-coefficients_c).mul_add(v_same, -coefficients_s * w_same);
    Ok((x1, y1, z1))
}

/// Validated raw-coefficient binary64 oracle summation.
///
/// The vector and scalar accumulators intentionally follow the established
/// raw recurrence order. Do not route this through [`PackedGravityCoeffs`]:
/// different lane reductions are observable from order five onward.
#[inline]
fn gravity_summation_f64_raw(
    v_workspace: &[[f64; MAX_RECURSIVE_ORDER]],
    w_workspace: &[[f64; MAX_RECURSIVE_ORDER]],
    c_coeffs: &[f64],
    s_coeffs: &[f64],
    stride: usize,
    order: usize,
    coef_sph: f64,
) -> Result<(f64, f64, f64), GravityError> {
    let coef_sph_half = coef_sph * 0.5;
    let coef_sph_half_v = f64x4::splat(coef_sph_half);
    let coef_sph_v = f64x4::splat(coef_sph);
    let mut first_axis_vector = f64x4::ZERO;
    let mut second_axis_vector = f64x4::ZERO;
    let mut third_axis_vector = f64x4::ZERO;
    let mut first_axis_scalar = 0.0f64;
    let mut second_axis_scalar = 0.0f64;
    let mut third_axis_scalar = 0.0f64;

    for degree in 0..=order {
        let base = degree
            .checked_mul(stride)
            .ok_or(GravityError::InvariantViolation)?;
        let row_end = base
            .checked_add(stride)
            .ok_or(GravityError::InvariantViolation)?;
        let c_row = c_coeffs
            .get(base..row_end)
            .ok_or(GravityError::InvariantViolation)?;
        let s_row = s_coeffs
            .get(base..row_end)
            .ok_or(GravityError::InvariantViolation)?;
        let row_index = degree
            .checked_add(1)
            .ok_or(GravityError::InvariantViolation)?;
        let v_row = v_workspace
            .get(row_index)
            .ok_or(GravityError::InvariantViolation)?;
        let w_row = w_workspace
            .get(row_index)
            .ok_or(GravityError::InvariantViolation)?;
        let (v0, v1) = two_values(v_row)?;
        let (_, w1) = two_values(w_row)?;
        let central_c = c_row
            .first()
            .copied()
            .ok_or(GravityError::InvariantViolation)?;
        let degree_plus_one = row_index.to_f64().ok_or(GravityError::InvariantViolation)?;
        first_axis_scalar += coef_sph * (-central_c * v1);
        second_axis_scalar += coef_sph * (-central_c * w1);
        third_axis_scalar += coef_sph * (degree_plus_one * (-central_c * v0));

        let mut harmonic_order = 1usize;
        if degree >= SIMD_L_THRESHOLD {
            while harmonic_order
                .checked_add(3)
                .is_some_and(|last| last <= degree)
            {
                let (x1, y1, z1) = gravity_summation_f64_raw_quad(
                    c_row,
                    s_row,
                    v_row,
                    w_row,
                    degree,
                    harmonic_order,
                )?;
                first_axis_vector += coef_sph_half_v * x1;
                second_axis_vector += coef_sph_half_v * y1;
                third_axis_vector += coef_sph_v * z1;
                harmonic_order = harmonic_order
                    .checked_add(4)
                    .ok_or(GravityError::InvariantViolation)?;
            }
        }
        while harmonic_order <= degree {
            let coefficient_c = c_row
                .get(harmonic_order)
                .copied()
                .ok_or(GravityError::InvariantViolation)?;
            let coefficient_s = s_row
                .get(harmonic_order)
                .copied()
                .ok_or(GravityError::InvariantViolation)?;
            let below = harmonic_order
                .checked_sub(1)
                .ok_or(GravityError::InvariantViolation)?;
            let above = harmonic_order
                .checked_add(1)
                .ok_or(GravityError::InvariantViolation)?;
            let v_below = v_row
                .get(below)
                .copied()
                .ok_or(GravityError::InvariantViolation)?;
            let v_same = v_row
                .get(harmonic_order)
                .copied()
                .ok_or(GravityError::InvariantViolation)?;
            let v_above = v_row
                .get(above)
                .copied()
                .ok_or(GravityError::InvariantViolation)?;
            let w_below = w_row
                .get(below)
                .copied()
                .ok_or(GravityError::InvariantViolation)?;
            let w_same = w_row
                .get(harmonic_order)
                .copied()
                .ok_or(GravityError::InvariantViolation)?;
            let w_above = w_row
                .get(above)
                .copied()
                .ok_or(GravityError::InvariantViolation)?;
            let degree_offset = degree
                .checked_sub(harmonic_order)
                .ok_or(GravityError::InvariantViolation)?
                .to_f64()
                .ok_or(GravityError::InvariantViolation)?;
            let coefficient_factor = (degree_offset + 2.0) * (degree_offset + 1.0);
            let x1 = (-coefficient_c).mul_add(
                v_above,
                (-coefficient_s).mul_add(
                    w_above,
                    coefficient_factor * coefficient_c.mul_add(v_below, coefficient_s * w_below),
                ),
            );
            let y1 = (-coefficient_c).mul_add(
                w_above,
                coefficient_s.mul_add(
                    v_above,
                    coefficient_factor * (-coefficient_c).mul_add(w_below, coefficient_s * v_below),
                ),
            );
            let degree_minus_one = degree_offset + 1.0;
            let z1 = degree_minus_one * (-coefficient_c).mul_add(v_same, -coefficient_s * w_same);
            first_axis_scalar += coef_sph_half * x1;
            second_axis_scalar += coef_sph_half * y1;
            third_axis_scalar += coef_sph * z1;
            harmonic_order = harmonic_order
                .checked_add(1)
                .ok_or(GravityError::InvariantViolation)?;
        }
    }
    Ok((
        first_axis_vector.reduce_add() + first_axis_scalar,
        second_axis_vector.reduce_add() + second_axis_scalar,
        third_axis_vector.reduce_add() + third_axis_scalar,
    ))
}

/// Evaluate validated raw generic gravity as the scalar oracle.
///
/// # Errors
///
/// Returns a typed error for malformed coefficients or invalid state, time, or
/// radius inputs.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "preserve the raw generic gravity evaluation's floating-point operation order"
)]
pub fn spherical_gravity_impl_generic<T: Float + FromPrimitive>(
    state_eci: &[T; 6],
    jd: f64,
    order: usize,
    c_coeffs: &[f64],
    s_coeffs: &[f64],
    stride: usize,
    cache: &mut GravityCacheGeneric<T>,
) -> Result<[T; 3], GravityError> {
    validate_gravity_coeffs_with_stride(c_coeffs, s_coeffs, stride, order)?;
    validate_jd(jd)?;
    let state_eci = validated_state(state_eci)?;
    let mut state_ecef = [T::zero(); 6];
    let gmst = T::from_f64(greenwichsrt_impl(jd)).ok_or(GravityError::InvariantViolation)?;
    let (sin_gmst, cos_gmst) = gmst.sin_cos();
    eci2ecef_impl_sincos(&state_eci, sin_gmst, cos_gmst, &mut state_ecef);

    let [pos_x, pos_y, pos_z, _, _, _] = state_ecef;
    validate_position_generic([pos_x, pos_y, pos_z])?;
    let radius = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z).sqrt();
    let re_t = T::from_f64(GRAVITY_REFERENCE_RADIUS_KM).ok_or(GravityError::InvariantViolation)?;
    let mu_t = T::from_f64(MU).ok_or(GravityError::InvariantViolation)?;
    let c2 = re_t / (radius * radius);
    let coef_sph = mu_t / (re_t * re_t);
    let c2_re = c2 * re_t;
    let x_c2 = pos_x * c2;
    let y_c2 = pos_y * c2;
    let z_c2 = pos_z * c2;
    let n = order
        .checked_add(2)
        .ok_or(GravityError::InvariantViolation)?;
    cache.begin_vw_fill(order);
    let v = &mut cache.v;
    let w = &mut cache.w;
    fill_legendre_generic(v, w, n, x_c2, y_c2, z_c2, c2_re)?;
    let (ax, ay, az) =
        gravity_summation_generic_raw(v, w, c_coeffs, s_coeffs, stride, order, coef_sph)?;
    let acc_ecef = [ax, ay, az, T::zero(), T::zero(), T::zero()];
    let mut acc_eci = [T::zero(); 6];
    ecef2eci_impl_sincos(&acc_ecef, sin_gmst, cos_gmst, &mut acc_eci);
    let [acceleration_x, acceleration_y, acceleration_z, _, _, _] = acc_eci;
    Ok([acceleration_x, acceleration_y, acceleration_z])
}

/// Evaluate validated raw binary64 gravity.
///
/// # Errors
///
/// Returns a typed error for malformed coefficients or invalid state, time, or
/// radius inputs.
#[inline]
pub fn spherical_gravity_impl(
    state_eci: &[f64; 6],
    jd: f64,
    order: usize,
    c_coeffs: &[f64],
    s_coeffs: &[f64],
    stride: usize,
    cache: &mut GravityCache,
) -> Result<[f64; 3], GravityError> {
    validate_jd(jd)?;
    let gmst = greenwichsrt_impl(jd);
    let (sin_gmst, cos_gmst) = gmst.sin_cos();
    spherical_gravity_impl_sincos(
        state_eci, sin_gmst, cos_gmst, order, c_coeffs, s_coeffs, stride, cache,
    )
}

/// Validate raw coefficient storage, then evaluate with a supplied GMST pair.
///
/// # Errors
///
/// Returns a typed error for malformed coefficients or invalid state, rotation,
/// or radius inputs.
#[inline]
pub fn spherical_gravity_impl_sincos(
    state_eci: &[f64; 6],
    sin_gmst: f64,
    cos_gmst: f64,
    order: usize,
    c_coeffs: &[f64],
    s_coeffs: &[f64],
    stride: usize,
    cache: &mut GravityCache,
) -> Result<[f64; 3], GravityError> {
    validate_gravity_coeffs_with_stride(c_coeffs, s_coeffs, stride, order)?;
    validate_rotation(sin_gmst, cos_gmst)?;
    let state_eci = validated_state(state_eci)?;
    let re = GRAVITY_REFERENCE_RADIUS_KM;
    let coef_sph = MU / (re * re);
    let mut state_ecef = [0.0; 6];
    eci2ecef_impl_sincos(&state_eci, sin_gmst, cos_gmst, &mut state_ecef);
    let [pos_x, pos_y, pos_z, _, _, _] = state_ecef;
    validate_position([pos_x, pos_y, pos_z])?;
    if cache.reuses_vw([pos_x, pos_y, pos_z], order) {
        let (ax, ay, az) = gravity_summation_f64_raw(
            &cache.v, &cache.w, c_coeffs, s_coeffs, stride, order, coef_sph,
        )?;
        let acc_ecef = [ax, ay, az, 0.0, 0.0, 0.0];
        let mut acc_eci = [0.0; 6];
        ecef2eci_impl_sincos(&acc_ecef, sin_gmst, cos_gmst, &mut acc_eci);
        let [acceleration_x, acceleration_y, acceleration_z, _, _, _] = acc_eci;
        return Ok([acceleration_x, acceleration_y, acceleration_z]);
    }
    cache.begin_vw_fill(order);
    let radius = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z).sqrt();
    let c2 = re / (radius * radius);
    let c2_re = c2 * re;
    let x_c2 = pos_x * c2;
    let y_c2 = pos_y * c2;
    let z_c2 = pos_z * c2;
    let n = order
        .checked_add(2)
        .ok_or(GravityError::InvariantViolation)?;
    let leg = cache.legendre_coeffs;
    let v = &mut cache.v;
    let w = &mut cache.w;
    let v00 = c2_re.sqrt();
    legendre_vw_dispatch(v, w, n, x_c2, y_c2, z_c2, c2_re, v00, leg)?;
    let (ax, ay, az) =
        gravity_summation_f64_raw(v, w, c_coeffs, s_coeffs, stride, order, coef_sph)?;
    let acc_ecef = [ax, ay, az, 0.0, 0.0, 0.0];
    let mut acc_eci = [0.0; 6];
    ecef2eci_impl_sincos(&acc_ecef, sin_gmst, cos_gmst, &mut acc_eci);
    cache.mark_vw_ready([pos_x, pos_y, pos_z], order);
    let [acceleration_x, acceleration_y, acceleration_z, _, _, _] = acc_eci;
    Ok([acceleration_x, acceleration_y, acceleration_z])
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "preserve the packed authority gravity recurrence's floating-point operation order"
)]
/// Evaluate a validated immutable packed generic gravity model.
///
/// # Errors
///
/// Returns a typed error for invalid state, time, radius, or checked workspace
/// invariants.
pub fn spherical_gravity_impl_generic_packed<T: Float + FromPrimitive>(
    state_eci: &[T; 6],
    jd: f64,
    cache: &mut GravityCacheGeneric<T>,
    packed: &PackedGravityCoeffs,
) -> Result<[T; 3], GravityError> {
    let order = packed.max_order;
    validate_jd(jd)?;
    let state_eci = validated_state(state_eci)?;

    let mut state_ecef = [T::zero(); 6];
    let gmst = T::from_f64(greenwichsrt_impl(jd)).ok_or(GravityError::InvariantViolation)?;
    let (s_gmst, c_gmst) = gmst.sin_cos();
    eci2ecef_impl_sincos(&state_eci, s_gmst, c_gmst, &mut state_ecef);

    let [pos_x, pos_y, pos_z, _, _, _] = state_ecef;
    validate_position_generic([pos_x, pos_y, pos_z])?;
    let r = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z).sqrt();

    let re_t = T::from_f64(GRAVITY_REFERENCE_RADIUS_KM).ok_or(GravityError::InvariantViolation)?;
    let mu_t = T::from_f64(MU).ok_or(GravityError::InvariantViolation)?;

    let c2 = re_t / (r * r);
    let coef_sph = mu_t / (re_t * re_t);
    let c2_re = c2 * re_t;
    let x_c2 = pos_x * c2;
    let y_c2 = pos_y * c2;
    let z_c2 = pos_z * c2;
    let n = order
        .checked_add(2)
        .ok_or(GravityError::InvariantViolation)?;
    cache.begin_vw_fill(order);
    let v = &mut cache.v;
    let w = &mut cache.w;

    fill_legendre_generic(v, w, n, x_c2, y_c2, z_c2, c2_re)?;

    let (ax, ay, az) = gravity_summation_generic_packed(v, w, coef_sph, packed)?;

    let acc_ecef = [ax, ay, az, T::zero(), T::zero(), T::zero()];
    let mut acc_eci = [T::zero(); 6];
    ecef2eci_impl_sincos(&acc_ecef, s_gmst, c_gmst, &mut acc_eci);

    let [acceleration_x, acceleration_y, acceleration_z, _, _, _] = acc_eci;
    Ok([acceleration_x, acceleration_y, acceleration_z])
}

/// SIMD-optimized spherical gravity with packed coefficients (DUST-51).
///
/// Uses FMA optimization for the Legendre recursion.
///
/// # Errors
///
/// Returns a typed error for invalid state, time, radius, or checked workspace
/// invariants.
#[inline]
pub fn spherical_gravity_impl_packed(
    state_eci: &[f64; 6],
    jd: f64,
    cache: &mut GravityCache,
    packed: &PackedGravityCoeffs,
) -> Result<[f64; 3], GravityError> {
    let order = packed.max_order;
    validate_jd(jd)?;
    let state_eci = validated_state(state_eci)?;
    let mut state_ecef = [0.0; 6];
    eci2ecef_impl(&state_eci, jd, &mut state_ecef);
    let [pos_x, pos_y, pos_z, _, _, _] = state_ecef;
    validate_position([pos_x, pos_y, pos_z])?;
    let r = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z).sqrt();
    let re = GRAVITY_REFERENCE_RADIUS_KM;
    let c2 = re / (r * r);
    let coef_sph = MU / (re * re);
    let c2_re = c2 * re;
    let x_c2 = pos_x * c2;
    let y_c2 = pos_y * c2;
    let z_c2 = pos_z * c2;
    let n = order
        .checked_add(2)
        .ok_or(GravityError::InvariantViolation)?;
    cache.begin_vw_fill(order);
    let v = &mut cache.v;
    let w = &mut cache.w;
    fill_legendre_packed_f64(v, w, n, x_c2, y_c2, z_c2, c2_re)?;
    let (ax, ay, az) = gravity_summation_f64_packed(v, w, coef_sph, packed)?;
    let acc_ecef = [ax, ay, az, 0.0, 0.0, 0.0];
    let mut acc_eci = [0.0; 6];
    ecef2eci_impl(&acc_ecef, jd, &mut acc_eci);
    let [acceleration_x, acceleration_y, acceleration_z, _, _, _] = acc_eci;
    Ok([acceleration_x, acceleration_y, acceleration_z])
}

/// SIMD-optimized spherical gravity with packed coefficients and precomputed GMST (DUST-51).
///
/// Uses FMA optimization for the Legendre recursion.
///
/// # Errors
///
/// Returns a typed error for invalid state, rotation, radius, or checked
/// workspace invariants.
#[inline]
pub fn spherical_gravity_impl_sincos_packed(
    state_eci: &[f64; 6],
    sin_gmst: f64,
    cos_gmst: f64,
    cache: &mut GravityCache,
    packed: &PackedGravityCoeffs,
) -> Result<[f64; 3], GravityError> {
    let order = packed.max_order;
    validate_rotation(sin_gmst, cos_gmst)?;
    let state_eci = validated_state(state_eci)?;
    let re = GRAVITY_REFERENCE_RADIUS_KM;
    let coef_sph = MU / (re * re);
    let mut state_ecef = [0.0; 6];
    eci2ecef_impl_sincos(&state_eci, sin_gmst, cos_gmst, &mut state_ecef);
    let [pos_x, pos_y, pos_z, _, _, _] = state_ecef;
    validate_position([pos_x, pos_y, pos_z])?;

    // This frozen sibling builds V/W with the legacy scalar column-major
    // recurrence. The other f64 siblings use the SIMD row-major recurrence;
    // their workspaces can differ by a few ULP from production order 5 onward.
    // Do not consume or publish V/W across those representations. It still has
    // to declare its fill width, or the bounded `reset` would not cover what it
    // writes.
    cache.begin_vw_fill(order);

    let r2 = pos_x * pos_x + pos_y * pos_y + pos_z * pos_z;
    let c2 = re / r2;
    let c2_re = c2 * re;
    let x_c2 = pos_x * c2;
    let y_c2 = pos_y * c2;
    let z_c2 = pos_z * c2;
    let n = order
        .checked_add(2)
        .ok_or(GravityError::InvariantViolation)?;

    let leg = cache.legendre_coeffs;

    let v = &mut cache.v;
    let w = &mut cache.w;
    fill_legendre_precomputed_f64(v, w, n, x_c2, y_c2, z_c2, c2_re, leg)?;
    let (ax, ay, az) = gravity_summation_f64_packed(v, w, coef_sph, packed)?;
    let acc_ecef = [ax, ay, az, 0.0, 0.0, 0.0];
    let mut acc_eci = [0.0; 6];
    ecef2eci_impl_sincos(&acc_ecef, sin_gmst, cos_gmst, &mut acc_eci);
    let [acceleration_x, acceleration_y, acceleration_z, _, _, _] = acc_eci;
    Ok([acceleration_x, acceleration_y, acceleration_z])
}

/// Compute spherical harmonic gravity using thread-local cache.
/// This is the recommended API - no need to manage cache externally.
///
/// # Errors
///
/// Returns a typed error for malformed coefficients or invalid state, time, or
/// radius inputs.
#[inline]
pub fn spherical_gravity(
    state_eci: &[f64; 6],
    jd: f64,
    order: usize,
    c_coeffs: &[f64],
    s_coeffs: &[f64],
    stride: usize,
) -> Result<[f64; 3], GravityError> {
    with_gravity_cache(|cache| {
        spherical_gravity_impl(state_eci, jd, order, c_coeffs, s_coeffs, stride, cache)
    })
}

/// Compute spherical harmonic gravity with packed coefficients using thread-local cache.
/// This is the recommended API for best performance.
///
/// # Errors
///
/// Returns a typed error for invalid state, time, radius, or checked workspace
/// invariants.
#[inline]
pub fn spherical_gravity_packed(
    state_eci: &[f64; 6],
    jd: f64,
    packed: &PackedGravityCoeffs,
) -> Result<[f64; 3], GravityError> {
    with_gravity_cache(|cache| spherical_gravity_impl_packed(state_eci, jd, cache, packed))
}

// ---------------------------------------------------------------------------
// Task 5B-2 `_frame` siblings.
//
// These live BELOW the last `function_markers` entry in `lib.rs`'s DIR-R6
// fragment pin on purpose. That harness walks `windows(2)` over the marker list
// and asserts each span binds EXACTLY ONE `GRAVITY_REFERENCE_RADIUS_KM`; a new
// function placed inside any span would add a second binding and fail the pin.
// Keep them here, after the final marker, or extend the marker list to cover
// them deliberately.
//
// The `_sincos` entry points above are FROZEN and bit-unchanged: the 4BG
// body-fixed oracle and the pinned bench call them. These siblings duplicate
// only the frame-agnostic middle of that computation.
// ---------------------------------------------------------------------------

/// Spherical-harmonic gravity evaluated entirely in the Earth-fixed frame.
///
/// Takes an ITRS position 3-vector and returns an ITRS acceleration 3-vector.
/// The caller applies the GCRS<->ITRS rotation, which under Task 5B-2 is the
/// full IAU 2006/2000A chain rather than a z-rotation by GMST — that is exactly
/// why the frame can no longer be passed as a `(sin, cos)` pair.
///
/// Only the Earth-fixed V/W recurrence workspace is cached. It needs the exact
/// ITRS position and covered order, but no output-frame key; every call reruns
/// coefficient-dependent summation, and `_sincos` always applies its requested
/// output rotation afterward.
///
/// # Errors
///
/// Returns a typed error for malformed coefficients or invalid position/radius
/// inputs.
pub fn spherical_gravity_impl_frame(
    pos_itrs: &[f64; 3],
    order: usize,
    c_coeffs: &[f64],
    s_coeffs: &[f64],
    stride: usize,
    cache: &mut GravityCache,
) -> Result<[f64; 3], GravityError> {
    validate_position(*pos_itrs)?;
    validate_gravity_coeffs_with_stride(c_coeffs, s_coeffs, stride, order)?;
    let radius_km = GRAVITY_REFERENCE_RADIUS_KM;
    let coef_sph = MU / (radius_km * radius_km);
    let [pos_x, pos_y, pos_z] = *pos_itrs;
    if cache.reuses_vw([pos_x, pos_y, pos_z], order) {
        let (ax, ay, az) = gravity_summation_f64_raw(
            &cache.v, &cache.w, c_coeffs, s_coeffs, stride, order, coef_sph,
        )?;
        return Ok([ax, ay, az]);
    }
    cache.begin_vw_fill(order);
    let r = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z).sqrt();
    let c2 = radius_km / (r * r);
    let c2_re = c2 * radius_km;
    let x_c2 = pos_x * c2;
    let y_c2 = pos_y * c2;
    let z_c2 = pos_z * c2;
    let n = order
        .checked_add(2)
        .ok_or(GravityError::InvariantViolation)?;
    let leg = cache.legendre_coeffs;
    let v = &mut cache.v;
    let w = &mut cache.w;
    let v00 = c2_re.sqrt();
    legendre_vw_dispatch(v, w, n, x_c2, y_c2, z_c2, c2_re, v00, leg)?;
    let (ax, ay, az) =
        gravity_summation_f64_raw(v, w, c_coeffs, s_coeffs, stride, order, coef_sph)?;
    cache.mark_vw_ready([pos_x, pos_y, pos_z], order);
    Ok([ax, ay, az])
}

/// Packed-coefficient sibling of [`spherical_gravity_impl_frame`].
///
/// # Errors
///
/// Returns a typed error for invalid position/radius or checked workspace
/// invariants.
pub fn spherical_gravity_impl_frame_packed(
    pos_itrs: &[f64; 3],
    cache: &mut GravityCache,
    packed: &PackedGravityCoeffs,
) -> Result<[f64; 3], GravityError> {
    let order = packed.max_order;
    validate_position(*pos_itrs)?;
    let radius_km = GRAVITY_REFERENCE_RADIUS_KM;
    let coef_sph = MU / (radius_km * radius_km);
    let [pos_x, pos_y, pos_z] = *pos_itrs;

    if cache.reuses_vw([pos_x, pos_y, pos_z], order) {
        let (ax, ay, az) = gravity_summation_f64_packed(&cache.v, &cache.w, coef_sph, packed)?;
        return Ok([ax, ay, az]);
    }
    cache.begin_vw_fill(order);

    let r = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z).sqrt();
    let c2 = radius_km / (r * r);
    let c2_re = c2 * radius_km;
    let x_c2 = pos_x * c2;
    let y_c2 = pos_y * c2;
    let z_c2 = pos_z * c2;
    let n = order
        .checked_add(2)
        .ok_or(GravityError::InvariantViolation)?;

    let leg = cache.legendre_coeffs;
    let v = &mut cache.v;
    let w = &mut cache.w;
    let v00 = c2_re.sqrt();
    legendre_vw_dispatch(v, w, n, x_c2, y_c2, z_c2, c2_re, v00, leg)?;
    let (ax, ay, az) = gravity_summation_f64_packed(v, w, coef_sph, packed)?;
    let result = [ax, ay, az];

    cache.mark_vw_ready([pos_x, pos_y, pos_z], order);
    Ok(result)
}

#[cfg(test)]
mod test_support {
    /// Bounds-checked slice write shared by this file's test modules. Three
    /// copies existed (two generic twins and one f64-monomorphic variant with
    /// an `if let` tail); the generic body is value-identical for every use.
    pub(super) fn test_set<T>(values: &mut [T], index: usize, value: T) {
        let value_slot = values.get_mut(index);
        assert!(
            value_slot.is_some(),
            "test index {index} outside length {}",
            values.len()
        );
        let Some(value_slot) = value_slot else {
            return;
        };
        *value_slot = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use common_rs::test_ok;

    use super::test_support::test_set;

    fn test_f64(value: usize) -> Result<f64, GravityError> {
        u32::try_from(value)
            .map(f64::from)
            .map_err(|_| GravityError::InvariantViolation)
    }

    fn test_get<T>(values: &[T], index: usize) -> Result<&T, GravityError> {
        values.get(index).ok_or(GravityError::InvariantViolation)
    }

    fn assert_f64_array_bits_eq<const N: usize>(left: [f64; N], right: [f64; N]) {
        for (left_value, right_value) in left.into_iter().zip(right) {
            assert_eq!(left_value.to_bits(), right_value.to_bits());
        }
    }

    fn test_zero_matrix() -> Box<[[f64; super::MAX_RECURSIVE_ORDER]]> {
        vec![[0.0; super::MAX_RECURSIVE_ORDER]; super::MAX_RECURSIVE_ORDER].into_boxed_slice()
    }

    #[test]
    fn packed_gravity_authority_digest_is_cached_in_immutable_pack() {
        let packed =
            pack_gravity_coeffs(&[1.0], &[0.0], 1, 0).expect("minimal gravity authority must pack");
        assert!(packed.authority_sha256_cache.get().is_none());
        let first = packed
            .authority_sha256()
            .expect("first gravity authority hash");
        let cached = packed
            .authority_sha256_cache
            .get()
            .expect("gravity authority digest was not cached")
            .expect("cached gravity authority error");
        assert_eq!(first, cached);
        assert_eq!(
            packed.authority_sha256().expect("cached gravity hash"),
            first
        );
    }

    #[test]
    fn thread_gravity_cache_prime_touches_max_storage_and_clears_state() {
        // Poison the far corner first. A freshly constructed cache is already
        // all zero, so without this the zero assertions below would hold even
        // if priming touched nothing at all.
        let last = super::MAX_RECURSIVE_ORDER.saturating_sub(1);
        super::with_gravity_cache(|cache| {
            test_ok!(super::matrix_set(&mut cache.v, last, last, 1.0));
            test_ok!(super::matrix_set(&mut cache.w, last, last, -1.0));
        });

        super::prime_thread_gravity_cache();

        super::with_gravity_cache(|cache| {
            assert_eq!(cache.v.len(), super::MAX_RECURSIVE_ORDER);
            assert_eq!(cache.w.len(), super::MAX_RECURSIVE_ORDER);
            assert_eq!(
                test_ok!(super::matrix_value(&cache.v, 0, 0)).to_bits(),
                0.0f64.to_bits()
            );
            assert_eq!(
                test_ok!(super::matrix_value(&cache.v, last, last)).to_bits(),
                0.0f64.to_bits()
            );
            assert_eq!(
                test_ok!(super::matrix_value(&cache.w, 0, 0)).to_bits(),
                0.0f64.to_bits()
            );
            assert_eq!(
                test_ok!(super::matrix_value(&cache.w, last, last)).to_bits(),
                0.0f64.to_bits()
            );
            assert_eq!(cache.legendre_coeffs.pt1.len(), super::MAX_RECURSIVE_ORDER);
            assert!(matches!(cache.vw_state, VwCacheState::Empty));
        });
    }

    #[test]
    fn gravity_cache_layout_is_bounded_for_worker_stacks() {
        use super::GravityCacheGeneric;

        const MAX_STACK_RESIDENT_CACHE_BYTES: usize = 256;
        assert!(
            std::mem::size_of::<GravityCacheGeneric<f64>>() <= MAX_STACK_RESIDENT_CACHE_BYTES,
            "gravity cache must own large recurrence matrices on bounded heap storage"
        );
        #[cfg(feature = "autodiff")]
        assert!(
            std::mem::size_of::<GravityCacheGeneric<crate::DualVec>>()
                <= MAX_STACK_RESIDENT_CACHE_BYTES
        );

        let cache = GravityCacheGeneric::<f64>::new();
        assert_eq!(cache.v.len(), super::MAX_RECURSIVE_ORDER);
        assert_eq!(cache.w.len(), super::MAX_RECURSIVE_ORDER);
        let clone = cache.clone();
        assert_ne!(cache.v.as_ptr(), clone.v.as_ptr());
        assert_ne!(cache.w.as_ptr(), clone.w.as_ptr());
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn dual_gravity_cache_constructs_and_drops_on_bounded_worker_stack() {
        use super::GravityCacheGeneric;

        let worker = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                for _ in 0..32 {
                    std::hint::black_box(GravityCacheGeneric::<crate::DualVec>::new());
                }
            });
        assert!(
            worker.is_ok(),
            "bounded-stack gravity cache worker must spawn"
        );
        if let Ok(worker) = worker {
            assert!(
                worker.join().is_ok(),
                "bounded-stack gravity cache construction must not overflow"
            );
        }
    }

    #[test]
    fn flat_gravity_coeff_validation_rejects_abort_inputs() {
        let valid_stride = MAX_ORDER + 1;
        let valid = vec![0.0; valid_stride * valid_stride];
        assert_eq!(
            test_ok!(validate_flat_gravity_coeffs(&valid, &valid, MAX_ORDER)),
            valid_stride
        );

        assert_eq!(
            validate_flat_gravity_coeffs(&valid, &valid, MAX_ORDER + 1),
            Err(GravityError::UnsupportedOrder)
        );
        assert_eq!(
            validate_flat_gravity_coeffs(&[0.0; 36], &[0.0; 35], 5),
            Err(GravityError::InvalidCoefficientStorage)
        );
        assert_eq!(
            validate_flat_gravity_coeffs(&[0.0; 35], &[0.0; 35], 5),
            Err(GravityError::InvalidCoefficientStorage)
        );
        let mut nonfinite = [0.0; 36];
        test_set(&mut nonfinite, 1, f64::NAN);
        assert_eq!(
            validate_flat_gravity_coeffs(&nonfinite, &[0.0; 36], 5),
            Err(GravityError::InvalidCoefficientStorage)
        );
    }

    #[test]
    fn pack_gravity_coeffs_rejects_unvalidated_geometry() {
        assert!(pack_gravity_coeffs(&[], &[], 0, 0).is_err());
        assert!(pack_gravity_coeffs(&[1.0], &[], 1, 0).is_err());
        assert!(pack_gravity_coeffs(&[1.0, 0.0], &[0.0, 0.0], 2, 0).is_err());
        assert!(pack_gravity_coeffs(&[f64::NAN], &[0.0], 1, 0).is_err());
    }

    #[test]
    fn packed_coeffs_reject_internal_metadata_tampering() {
        let stride = 6usize;
        let order = 5usize;
        let mut c = vec![0.0; stride * stride];
        let mut s = vec![0.0; stride * stride];
        test_set(&mut c, 0, 1.0);
        let sparse_index = 3usize.saturating_mul(stride).saturating_add(1);
        test_set(&mut c, sparse_index, 2.0e-6);
        test_set(&mut s, sparse_index, -1.0e-6);
        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));

        let mut invalid_derived_term = packed.clone();
        let row = invalid_derived_term.rows.get_mut(3);
        assert!(row.is_some(), "test fixture row must exist");
        let Some(row) = row else {
            return;
        };
        let term = row.terms.get_mut(0);
        assert!(term.is_some(), "test fixture term must exist");
        let Some(term) = term else {
            return;
        };
        term.cf_2 = 0.0;
        assert_eq!(
            invalid_derived_term.validate_metadata(),
            Err(GravityError::InvariantViolation)
        );
        assert_eq!(
            invalid_derived_term.authority_sha256(),
            Err(GravityError::InvariantViolation)
        );

        let mut widened_prefix = packed;
        widened_prefix.dense_prefix = 3;
        assert_eq!(
            widened_prefix.validate_metadata(),
            Err(GravityError::InvariantViolation)
        );
        assert_eq!(
            widened_prefix.authority_sha256(),
            Err(GravityError::InvariantViolation)
        );
    }

    #[test]
    fn packed_authority_hash_tracks_mathematical_coefficients_and_order() {
        let stride = 4usize;
        let order = 3usize;
        let mut c = vec![0.0; stride.saturating_mul(stride)];
        let mut s = vec![0.0; stride.saturating_mul(stride)];
        test_set(&mut c, 0, 1.0);
        let degree_two_order_zero = 2usize.saturating_mul(stride);
        let degree_two_order_two = degree_two_order_zero.saturating_add(2);
        test_set(&mut c, degree_two_order_zero, -1.082_63e-3);
        test_set(&mut c, degree_two_order_two, 1.574_46e-6);
        test_set(&mut s, degree_two_order_two, -9.038_04e-7);

        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        let repeated = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        let authority_hash = test_ok!(packed.authority_sha256());
        assert_eq!(authority_hash, test_ok!(repeated.authority_sha256()));

        let mut changed_c = c.clone();
        test_set(&mut changed_c, degree_two_order_two, 1.574_47e-6);
        let changed = test_ok!(pack_gravity_coeffs(&changed_c, &s, stride, order));
        assert_ne!(authority_hash, test_ok!(changed.authority_sha256()));

        let lower_order = order.saturating_sub(1);
        let direct_lower = test_ok!(pack_gravity_coeffs(&c, &s, stride, lower_order));
        let truncated = test_ok!(packed.truncated_to(lower_order));
        let truncated_hash = test_ok!(truncated.authority_sha256());
        assert_ne!(authority_hash, truncated_hash);
        assert_eq!(truncated_hash, test_ok!(direct_lower.authority_sha256()));
    }

    #[test]
    fn packed_coeffs_preserve_valid_coefficient_bits() {
        let stride = 4usize;
        let order = 3usize;
        let mut c = vec![0.0; stride * stride];
        let mut s = vec![0.0; stride * stride];
        test_set(&mut c, 0, 1.0);
        let first = 3usize.saturating_mul(stride).saturating_add(1);
        let second = 3usize.saturating_mul(stride).saturating_add(3);
        test_set(&mut c, first, -1.234_567_890_123_45e-6);
        test_set(&mut s, first, 9.876_543_210_987_65e-7);
        test_set(&mut c, second, 4.567_890_123_456_78e-8);
        test_set(&mut s, second, -6.789_012_345_678_9e-9);

        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        let row = packed.rows.get(3);
        assert!(row.is_some(), "test fixture row must exist");
        let Some(row) = row else {
            return;
        };
        assert_eq!(row.terms.len(), 2);
        let first_term = row.terms.first();
        assert!(first_term.is_some(), "test fixture first term must exist");
        let Some(first_term) = first_term else {
            return;
        };
        let second_term = row.terms.get(1);
        assert!(second_term.is_some(), "test fixture second term must exist");
        let Some(second_term) = second_term else {
            return;
        };
        assert_eq!(first_term.m, 1);
        let first_c = c.get(first);
        assert!(first_c.is_some(), "test fixture C(3,1) must exist");
        let Some(first_c) = first_c else {
            return;
        };
        let first_s = s.get(first);
        assert!(first_s.is_some(), "test fixture S(3,1) must exist");
        let Some(first_s) = first_s else {
            return;
        };
        assert_eq!(first_term.c.to_bits(), first_c.to_bits());
        assert_eq!(first_term.s.to_bits(), first_s.to_bits());
        assert_eq!(second_term.m, 3);
        let second_c = c.get(second);
        assert!(second_c.is_some(), "test fixture C(3,3) must exist");
        let Some(second_c) = second_c else {
            return;
        };
        let second_s = s.get(second);
        assert!(second_s.is_some(), "test fixture S(3,3) must exist");
        let Some(second_s) = second_s else {
            return;
        };
        assert_eq!(second_term.c.to_bits(), second_c.to_bits());
        assert_eq!(second_term.s.to_bits(), second_s.to_bits());
        test_ok!(packed.validate_metadata());
    }

    #[test]
    fn packed_degree_one_presence_matches_authority_threshold() {
        let stride = 2usize;
        let order = 1usize;
        let mut c = vec![0.0; stride * stride];
        let mut s = vec![0.0; stride * stride];
        test_set(&mut c, 0, 1.0);

        test_set(&mut c, stride, 0.5e-18);
        assert!(!test_ok!(pack_gravity_coeffs(&c, &s, stride, order)).has_nonzero_degree1_terms());

        test_set(&mut c, stride, 1.0e-18);
        assert!(!test_ok!(pack_gravity_coeffs(&c, &s, stride, order)).has_nonzero_degree1_terms());

        test_set(&mut c, stride, (-1.0e-18_f64).next_down());
        assert!(test_ok!(pack_gravity_coeffs(&c, &s, stride, order)).has_nonzero_degree1_terms());

        test_set(&mut c, stride, 0.0);
        test_set(&mut s, stride + 1, (-1.0e-18_f64).next_down());
        assert!(test_ok!(pack_gravity_coeffs(&c, &s, stride, order)).has_nonzero_degree1_terms());
    }

    fn hostile_gravity_fixture() -> Result<(Vec<f64>, Vec<f64>, PackedGravityCoeffs), GravityError>
    {
        let stride = 3usize;
        let order = 2usize;
        let mut c = vec![0.0; stride * stride];
        let s = vec![0.0; stride * stride];
        test_set(&mut c, 0, 1.0);
        let packed = pack_gravity_coeffs(&c, &s, stride, order)?;
        Ok((c, s, packed))
    }

    #[test]
    fn gravity_entrypoints_reject_nonfinite_and_degenerate_inputs() {
        let fixture = hostile_gravity_fixture();
        assert!(fixture.is_ok(), "fixed hostile-test fixture must pack");
        let Ok((c, s, packed)) = fixture else {
            return;
        };
        let order = packed.max_order();
        let stride = order + 1;
        let jd = 2_460_000.25;
        let valid_state = [6_778.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let nonfinite_state = [f64::NAN, 0.0, 0.0, 0.0, 7.5, 0.0];
        let zero_state = [0.0; 6];

        assert_eq!(
            spherical_gravity_impl(
                &nonfinite_state,
                jd,
                order,
                &c,
                &s,
                stride,
                &mut GravityCache::new(),
            ),
            Err(GravityError::InvalidState)
        );
        assert_eq!(
            spherical_gravity_impl(
                &zero_state,
                jd,
                order,
                &c,
                &s,
                stride,
                &mut GravityCache::new(),
            ),
            Err(GravityError::InvalidRadius)
        );
        assert_eq!(
            spherical_gravity_impl(
                &valid_state,
                f64::NAN,
                order,
                &c,
                &s,
                stride,
                &mut GravityCache::new(),
            ),
            Err(GravityError::InvalidTime)
        );
        assert_eq!(
            spherical_gravity_impl_sincos(
                &valid_state,
                f64::INFINITY,
                1.0,
                order,
                &c,
                &s,
                stride,
                &mut GravityCache::new(),
            ),
            Err(GravityError::InvalidRotation)
        );
        assert_eq!(
            spherical_gravity_impl_frame(
                &[0.0, 0.0, 0.0],
                order,
                &c,
                &s,
                stride,
                &mut GravityCache::new(),
            ),
            Err(GravityError::InvalidRadius)
        );

        assert_eq!(
            spherical_gravity_impl_packed(
                &valid_state,
                f64::NAN,
                &mut GravityCache::new(),
                &packed,
            ),
            Err(GravityError::InvalidTime)
        );
        assert_eq!(
            spherical_gravity_impl_sincos_packed(
                &valid_state,
                f64::INFINITY,
                1.0,
                &mut GravityCache::new(),
                &packed,
            ),
            Err(GravityError::InvalidRotation)
        );
        assert_eq!(
            spherical_gravity_impl_frame_packed(
                &[f64::INFINITY, 0.0, 0.0],
                &mut GravityCache::new(),
                &packed,
            ),
            Err(GravityError::InvalidRadius)
        );
    }

    #[test]
    fn gravity_cache_never_reuses_nearby_distinct_positions() {
        let order = 5usize;
        let stride = order + 1;
        let mut c = vec![0.0; stride * stride];
        let mut s = vec![0.0; stride * stride];
        test_set(&mut c, 0, 1.0);
        test_set(&mut c, 2usize.saturating_mul(stride), -1.082_626_68e-3);
        test_set(
            &mut c,
            2usize.saturating_mul(stride).saturating_add(1),
            2.0e-6,
        );
        test_set(
            &mut s,
            2usize.saturating_mul(stride).saturating_add(1),
            -1.0e-6,
        );
        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        let jd = 2_460_000.25;
        let first = [6_778.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let nearby = [6_778.2, 0.1, -0.05, 0.0, 7.5, 0.0];

        let mut warmed = GravityCache::new();
        test_ok!(spherical_gravity_impl(
            &first,
            jd,
            order,
            &c,
            &s,
            stride,
            &mut warmed,
        ));
        let warmed_result = test_ok!(spherical_gravity_impl(
            &nearby,
            jd,
            order,
            &c,
            &s,
            stride,
            &mut warmed,
        ));
        let fresh_result = test_ok!(spherical_gravity_impl(
            &nearby,
            jd,
            order,
            &c,
            &s,
            stride,
            &mut GravityCache::new(),
        ));
        assert_f64_array_bits_eq(warmed_result, fresh_result);

        let mut warmed_packed = GravityCache::new();
        test_ok!(spherical_gravity_impl_packed(
            &first,
            jd,
            &mut warmed_packed,
            &packed,
        ));
        let warmed_packed_result = test_ok!(spherical_gravity_impl_packed(
            &nearby,
            jd,
            &mut warmed_packed,
            &packed,
        ));
        let fresh_packed_result = test_ok!(spherical_gravity_impl_packed(
            &nearby,
            jd,
            &mut GravityCache::new(),
            &packed,
        ));
        assert_f64_array_bits_eq(warmed_packed_result, fresh_packed_result);
    }

    #[test]
    fn gravity_vw_cache_reuse_preserves_changed_earth_rotation_angle() {
        let order = 5usize;
        let stride = order + 1;
        let mut c = vec![0.0; stride * stride];
        let mut s = vec![0.0; stride * stride];
        test_set(&mut c, 0, 1.0);
        test_set(&mut c, 2usize.saturating_mul(stride), -1.082_626_68e-3);
        test_set(
            &mut c,
            2usize.saturating_mul(stride).saturating_add(1),
            2.0e-6,
        );
        test_set(
            &mut s,
            2usize.saturating_mul(stride).saturating_add(1),
            -1.0e-6,
        );
        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        // An ECI point on Earth's spin axis has identical ECEF coordinates at
        // both epochs, while its tesseral acceleration still rotates in ECI.
        let state = [0.0, 0.0, 6_778.0, 0.0, 0.0, 0.0];
        let jd_first = 2_460_000.25;
        let jd_later = jd_first + 0.125;

        let mut warmed = GravityCache::new();
        test_ok!(spherical_gravity_impl(
            &state,
            jd_first,
            order,
            &c,
            &s,
            stride,
            &mut warmed,
        ));
        let warmed_result = test_ok!(spherical_gravity_impl(
            &state,
            jd_later,
            order,
            &c,
            &s,
            stride,
            &mut warmed,
        ));
        let fresh_result = test_ok!(spherical_gravity_impl(
            &state,
            jd_later,
            order,
            &c,
            &s,
            stride,
            &mut GravityCache::new(),
        ));
        assert_f64_array_bits_eq(warmed_result, fresh_result);

        let mut warmed_packed = GravityCache::new();
        test_ok!(spherical_gravity_impl_packed(
            &state,
            jd_first,
            &mut warmed_packed,
            &packed,
        ));
        let warmed_packed_result = test_ok!(spherical_gravity_impl_packed(
            &state,
            jd_later,
            &mut warmed_packed,
            &packed,
        ));
        let fresh_packed_result = test_ok!(spherical_gravity_impl_packed(
            &state,
            jd_later,
            &mut GravityCache::new(),
            &packed,
        ));
        assert_f64_array_bits_eq(warmed_packed_result, fresh_packed_result);
    }

    /// The SIMD branch in the Legendre row fill has to be reachable at the
    /// production shape, which is what its threshold is for. It was not: the
    /// threshold was shared with the summation gate at 8, while the row loop
    /// at `n = 7` never reaches `l = 7`.
    ///
    /// This walks the two guards rather than trusting them -- outer
    /// `for l in 3..n`, then `legendre_l_row_simd(.., 1, l - 1, ..)` whose quad
    /// loop runs while `m + 3 < m_end && m + 3 < l`.
    #[test]
    fn legendre_simd_branch_is_reachable_at_production_shape() {
        use super::{GRAVITY_FAST_PATH_ORDER_CAP, LEGENDRE_SIMD_L_THRESHOLD, SIMD_L_THRESHOLD};

        fn reach(n: usize, threshold: usize) -> (usize, usize) {
            let (mut total, mut vectorised) = (0usize, 0usize);
            for l in 3..n {
                let m_end = l - 1;
                total += m_end.saturating_sub(1);
                if l >= threshold {
                    let mut m = 1;
                    while m + 3 < m_end && m + 3 < l {
                        vectorised += 4;
                        m += 4;
                    }
                }
            }
            (total, vectorised)
        }

        // `spherical_gravity_impl_*` all call the row fill with `order + 2`.
        let n = GRAVITY_FAST_PATH_ORDER_CAP + 2;
        assert_eq!(
            n, 7,
            "production Legendre width changed; re-derive the reach"
        );

        let (total, vectorised) = reach(n, LEGENDRE_SIMD_L_THRESHOLD);
        assert_eq!(total, 10);
        assert!(
            vectorised > 0,
            "the Legendre SIMD branch is unreachable at n={n}: 0 of {total} inner \
             cells vectorise at threshold {LEGENDRE_SIMD_L_THRESHOLD}. The function \
             is named ..._simd_... but is running scalar in production."
        );
        assert_eq!(vectorised, 4, "expected one quad, on row 6");

        // The value this used to share with the summation gate reached nothing,
        // and neither does 7. Both are recorded so a future edit that raises the
        // threshold back trips the assertion above with the reason visible here.
        assert_eq!(reach(n, 7).1, 0);
        assert_eq!(reach(n, SIMD_L_THRESHOLD).1, 0);
        // 6 is the floor: below it nothing further becomes reachable, because
        // rows 3..=5 are too narrow for a quad regardless of threshold.
        assert_eq!(reach(n, 4).1, 4);
    }

    #[test]
    fn packed_coeffs_dense_prefix_tracks_contiguous_dense_rows() {
        let stride = 6usize;
        let order = 5usize;
        let mut c = vec![0.0; stride * stride];
        let mut s = vec![0.0; stride * stride];
        test_set(&mut c, 0, 1.0);
        for l in 1..=order {
            let base = l.saturating_mul(stride);
            for m in 1..=l {
                let degree_order = test_ok!(test_f64(l.saturating_add(m)));
                test_set(&mut c, base.saturating_add(m), 1.0e-6 * degree_order);
                test_set(&mut s, base.saturating_add(m), 1.0e-7 * degree_order);
            }
        }

        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        assert_eq!(packed.dense_prefix, order);
        assert!(packed
            .rows
            .iter()
            .enumerate()
            .all(|(degree, row)| row.terms.len() == degree));
    }

    #[test]
    fn packed_coeffs_dense_prefix_stops_at_first_sparse_row() {
        let stride = 6usize;
        let order = 5usize;
        let mut c = vec![0.0; stride * stride];
        let mut s = vec![0.0; stride * stride];
        test_set(&mut c, 0, 1.0);
        for l in 1..=order {
            let base = l.saturating_mul(stride);
            for m in 1..=l {
                let degree_order = test_ok!(test_f64(l.saturating_add(m)));
                test_set(&mut c, base.saturating_add(m), 1.0e-6 * degree_order);
                test_set(&mut s, base.saturating_add(m), 1.0e-7 * degree_order);
            }
        }
        // Introduce the first sparse row at l=3 by zeroing m=2 term.
        let sparse_index = 3usize.saturating_mul(stride).saturating_add(2);
        test_set(&mut c, sparse_index, 0.0);
        test_set(&mut s, sparse_index, 0.0);

        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        assert_eq!(packed.dense_prefix, 2);
        assert!(packed
            .rows
            .iter()
            .take(3)
            .enumerate()
            .all(|(degree, row)| row.terms.len() == degree));
        let sparse_row = test_ok!(test_get(&packed.rows, 3));
        assert_eq!(sparse_row.terms.len(), 2);
    }

    #[test]
    fn packed_coeffs_sparse_quad_preserves_stable_emitted_order() {
        let stride = 8usize;
        let order = 7usize;
        let mut c = vec![0.0; stride * stride];
        let mut s = vec![0.0; stride * stride];
        test_set(&mut c, 0, 1.0);

        // Build a sparse-gather row at l=6:
        // keep m = {1,2,3,4,6}, drop m=5.
        // This creates a contiguous run of length 4 plus a trailing singleton.
        let base = 6usize.saturating_mul(stride);
        for m in [1usize, 2, 3, 4, 6] {
            let coefficient = test_ok!(test_f64(m.saturating_add(1)));
            test_set(&mut c, base.saturating_add(m), 1.0e-6 * coefficient);
            test_set(&mut s, base.saturating_add(m), 1.0e-7 * coefficient);
        }

        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        let row = test_ok!(test_get(&packed.rows, 6));
        assert_eq!(row.terms.len(), 5);
        assert_eq!(row.quads.len(), 1);
        let quad = test_ok!(test_get(&row.quads, 0));
        assert_eq!(quad.orders, [1, 2, 3, 4]);
        let tail = test_ok!(test_get(&row.terms, 4));
        assert_eq!(tail.m, 6);
    }

    #[test]
    fn packed_coeffs_sparse_contiguous_rows_preserve_first_quad() {
        let stride = 8usize;
        let order = 7usize;
        let mut c = vec![0.0; stride * stride];
        let mut s = vec![0.0; stride * stride];
        test_set(&mut c, 0, 1.0);

        // Sparse contiguous row at l=6:
        // keep m = {2,3,4,5,6}; contiguous but not dense.
        let base = 6usize.saturating_mul(stride);
        for m in [2usize, 3, 4, 5, 6] {
            let coefficient = test_ok!(test_f64(m.saturating_add(1)));
            test_set(&mut c, base.saturating_add(m), 1.0e-6 * coefficient);
            test_set(&mut s, base.saturating_add(m), 1.0e-7 * coefficient);
        }

        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        let row = test_ok!(test_get(&packed.rows, 6));
        assert_eq!(row.terms.len(), 5);
        assert_eq!(row.quads.len(), 1);
        let quad = test_ok!(test_get(&row.quads, 0));
        assert_eq!(quad.orders, [2, 3, 4, 5]);
        let tail = test_ok!(test_get(&row.terms, 4));
        assert_eq!(tail.m, 6);
    }

    #[test]
    fn packed_coeffs_sparse_quads_preserve_noncontiguous_coordinates() {
        let stride = 8usize;
        let order = 7usize;
        let mut c = vec![0.0; stride * stride];
        let mut s = vec![0.0; stride * stride];
        test_set(&mut c, 0, 1.0);

        // Sparse gather row at l=6 with a single non-contiguous 4-lane quad.
        // m = {1,3,4,6}
        let base = 6usize.saturating_mul(stride);
        for m in [1usize, 3, 4, 6] {
            let coefficient = test_ok!(test_f64(m.saturating_add(1)));
            test_set(&mut c, base.saturating_add(m), 1.0e-6 * coefficient);
            test_set(&mut s, base.saturating_add(m), 1.0e-7 * coefficient);
        }

        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        let row = test_ok!(test_get(&packed.rows, 6));
        assert_eq!(row.terms.len(), 4);
        assert_eq!(row.quads.len(), 1);
        let quad = test_ok!(test_get(&row.quads, 0));
        assert_eq!(quad.orders, [1, 3, 4, 6]);
        let c_m1 = test_ok!(test_get(&c, base.saturating_add(1))).to_bits();
        let c_m3 = test_ok!(test_get(&c, base.saturating_add(3))).to_bits();
        let c_m4 = test_ok!(test_get(&c, base.saturating_add(4))).to_bits();
        let c_m6 = test_ok!(test_get(&c, base.saturating_add(6))).to_bits();
        assert_eq!(quad.c.map(f64::to_bits), [c_m1, c_m3, c_m4, c_m6]);
    }

    #[test]
    fn packed_coeffs_precompute_dm1_and_cf2_from_lm() {
        let stride = 7usize;
        let order = 6usize;
        let mut cosine_coefficients = vec![0.0; stride * stride];
        let mut sine_coefficients = vec![0.0; stride * stride];
        test_set(&mut cosine_coefficients, 0, 1.0);

        // Keep a sparse row with known non-zero terms at l=5 and m={1,3,4}.
        let degree = 5usize;
        let base = degree.saturating_mul(stride);
        for order_index in [1usize, 3, 4] {
            let coefficient = test_ok!(test_f64(order_index.saturating_add(1)));
            test_set(
                &mut cosine_coefficients,
                base.saturating_add(order_index),
                1.0e-6 * coefficient,
            );
            test_set(
                &mut sine_coefficients,
                base.saturating_add(order_index),
                1.0e-7 * coefficient,
            );
        }

        let packed = test_ok!(pack_gravity_coeffs(
            &cosine_coefficients,
            &sine_coefficients,
            stride,
            order,
        ));
        let row = test_ok!(test_get(&packed.rows, degree));
        assert_eq!(row.terms.len(), 3);

        for term in &row.terms {
            let degree_delta = degree.checked_sub(term.m);
            assert!(degree_delta.is_some(), "term order must not exceed degree");
            let Some(degree_delta) = degree_delta else {
                return;
            };
            let degree_delta = test_ok!(test_f64(degree_delta));
            assert!((term.dm1 - (degree_delta + 1.0)).abs() <= f64::EPSILON);
            let expected_cf2 = (degree_delta + 2.0) * (degree_delta + 1.0);
            assert!((term.cf_2 - expected_cf2).abs() <= f64::EPSILON);
        }
    }

    #[test]
    fn packed_gravity_matches_unpacked_for_sparse_gather_rows() {
        let order = 21usize;
        let stride = order + 1;
        let total_size = stride * stride;

        let mut c = vec![0.0; total_size];
        let mut s = vec![0.0; total_size];
        test_set(&mut c, 0, 1.0);

        for l in 2..=order {
            let base = l.saturating_mul(stride);
            let degree = test_ok!(test_f64(l));
            test_set(&mut c, base, 1e-3 / degree.powi(2));
            for m in 1..=l {
                if ((m + 2 * l) % 5 == 0) || ((m + l) % 7 == 0) {
                    continue;
                }
                let degree_order = test_ok!(test_f64(l.saturating_mul(m)));
                let magnitude = 1e-6 / degree_order;
                test_set(&mut c, base.saturating_add(m), magnitude);
                test_set(&mut s, base.saturating_add(m), magnitude * 0.5);
            }
        }

        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        let states = [
            [6778.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            [6778.2, 1.1, -0.7, 0.002, 7.499, -0.001],
            [6777.4, -2.3, 0.9, -0.003, 7.501, 0.002],
        ];
        let jd = 2_460_000.5;

        for state in states {
            let mut cache_ref = GravityCache::new();
            let mut cache_packed = GravityCache::new();
            let acc_ref = test_ok!(spherical_gravity_impl(
                &state,
                jd,
                order,
                &c,
                &s,
                stride,
                &mut cache_ref,
            ));
            let acc_packed = test_ok!(spherical_gravity_impl_packed(
                &state,
                jd,
                &mut cache_packed,
                &packed,
            ));
            for (i, (reference, packed_value)) in acc_ref.into_iter().zip(acc_packed).enumerate() {
                let diff = (reference - packed_value).abs();
                let scale = reference.abs().max(packed_value.abs()).max(1.0);
                assert!(
                    diff <= 1e-6 || diff <= 1e-12 * scale,
                    "axis={i} ref={reference} packed={packed_value} diff={diff} scale={scale}",
                );
            }
        }
    }

    #[test]
    fn packed_gravity_matches_unpacked_for_dense_low_orders() {
        let order = 7usize;
        let stride = order + 1;
        let total_size = stride * stride;
        let mut c = vec![0.0; total_size];
        let mut s = vec![0.0; total_size];
        test_set(&mut c, 0, 1.0);

        for l in 1..=order {
            let base = l.saturating_mul(stride);
            let degree = test_ok!(test_f64(l));
            test_set(&mut c, base, 1e-3 / degree.powi(2));
            for m in 1..=l {
                let degree_order = test_ok!(test_f64(l.saturating_add(m)));
                let magnitude = 1e-6 / degree_order;
                test_set(&mut c, base.saturating_add(m), magnitude);
                test_set(&mut s, base.saturating_add(m), magnitude * 0.5);
            }
        }

        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        assert_eq!(packed.dense_prefix, order);

        let states = [
            [6778.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            [6778.2, 1.1, -0.7, 0.002, 7.499, -0.001],
            [6777.4, -2.3, 0.9, -0.003, 7.501, 0.002],
        ];
        let jd = 2_460_000.5;

        for state in states {
            for eval_order in 2usize..=order {
                let mut cache_ref = GravityCache::new();
                let mut cache_packed = GravityCache::new();
                let acc_ref = test_ok!(spherical_gravity_impl(
                    &state,
                    jd,
                    eval_order,
                    &c,
                    &s,
                    stride,
                    &mut cache_ref,
                ));
                let truncated = test_ok!(packed.truncated_to(eval_order));
                let acc_packed = test_ok!(spherical_gravity_impl_packed(
                    &state,
                    jd,
                    &mut cache_packed,
                    &truncated,
                ));
                for (i, (reference, packed_value)) in
                    acc_ref.into_iter().zip(acc_packed).enumerate()
                {
                    let diff = (reference - packed_value).abs();
                    let scale = reference.abs().max(packed_value.abs()).max(1.0);
                    assert!(
                        diff <= 1e-6 || diff <= 1e-12 * scale,
                        "order={eval_order} axis={i} ref={reference} packed={packed_value} diff={diff} scale={scale}",
                    );
                }
            }
        }
    }

    /// The shared process-wide Legendre table must equal a freshly built one
    /// bit for bit, and every cache must borrow that shared table.
    ///
    /// Hoisting the table out of `GravityCacheGeneric::new` is only sound if
    /// the shared copy carries the exact bits `LegendreCoeffsSimd::fixed`
    /// produces, so every entry of both `pt1` and `pt21_factor` is compared by
    /// `to_bits`, never by `==` and never by tolerance.
    #[test]
    fn shared_legendre_table_is_bit_identical_to_a_fresh_one() {
        use super::{GravityCacheGeneric, LegendreCoeffsSimd, MAX_RECURSIVE_ORDER};

        let fresh = LegendreCoeffsSimd::fixed();
        let shared = LegendreCoeffsSimd::shared();

        assert_eq!(fresh.pt1.len(), MAX_RECURSIVE_ORDER);
        assert_eq!(fresh.pt21_factor.len(), MAX_RECURSIVE_ORDER);
        assert_eq!(shared.pt1.len(), MAX_RECURSIVE_ORDER);
        assert_eq!(shared.pt21_factor.len(), MAX_RECURSIVE_ORDER);

        for (degree, (fresh_row, shared_row)) in fresh.pt1.iter().zip(shared.pt1.iter()).enumerate()
        {
            for (order, (fresh_value, shared_value)) in
                fresh_row.iter().zip(shared_row.iter()).enumerate()
            {
                assert_eq!(
                    fresh_value.to_bits(),
                    shared_value.to_bits(),
                    "pt1[{degree}][{order}] fresh={fresh_value} shared={shared_value}"
                );
            }
        }
        for (degree, (fresh_row, shared_row)) in fresh
            .pt21_factor
            .iter()
            .zip(shared.pt21_factor.iter())
            .enumerate()
        {
            for (order, (fresh_value, shared_value)) in
                fresh_row.iter().zip(shared_row.iter()).enumerate()
            {
                assert_eq!(
                    fresh_value.to_bits(),
                    shared_value.to_bits(),
                    "pt21_factor[{degree}][{order}] fresh={fresh_value} shared={shared_value}"
                );
            }
        }

        // Two all-zero tables would also compare equal, so pin how many
        // entries the recurrence actually fills: rows 2..=130 hold degree-1
        // coefficients each.
        let expected_populated: usize = (2..MAX_RECURSIVE_ORDER)
            .map(|degree| degree.saturating_sub(1))
            .sum();
        let populated = |table: &[[f64; MAX_RECURSIVE_ORDER]]| -> usize {
            table
                .iter()
                .flatten()
                .filter(|value| value.to_bits() != 0.0f64.to_bits())
                .count()
        };
        assert_eq!(populated(&shared.pt1), expected_populated);
        assert_eq!(populated(&shared.pt21_factor), expected_populated);

        // The hoist is only real if caches point at the shared table.
        let scalar_cache = GravityCacheGeneric::<f64>::new();
        let another_cache = GravityCacheGeneric::<f64>::new();
        assert!(std::ptr::eq(scalar_cache.legendre_coeffs, shared));
        assert!(std::ptr::eq(
            scalar_cache.legendre_coeffs,
            another_cache.legendre_coeffs
        ));
        assert!(std::ptr::eq(
            scalar_cache.legendre_coeffs.pt1.as_ptr(),
            shared.pt1.as_ptr()
        ));
        assert!(std::ptr::eq(
            scalar_cache.legendre_coeffs.pt21_factor.as_ptr(),
            shared.pt21_factor.as_ptr()
        ));
    }

    /// Verify SIMD row-major Legendre recursion produces bit-identical V/W arrays
    /// to the scalar column-major recursion for various gravity orders.
    #[test]
    fn test_legendre_simd_vs_scalar() {
        use super::LegendreCoeffsSimd;

        // Test representative ECI state for LEO orbit
        let x_c2 = 0.001_234;
        let y_c2 = -0.000_567;
        let z_c2 = 0.000_890;
        let c2_re = 0.000_942;
        let v00 = c2_re.sqrt();

        // Both `continue` guards below skip a coefficient when BOTH arms are
        // exactly zero. If either recursion ever early-returned without writing,
        // every pair would compare 0.0 to 0.0, every element would be skipped,
        // and this test would pass having compared nothing. `compared` is the
        // backstop; see the floor assertion after the loop.
        let mut compared = 0_usize;

        for order in [4, 8, 15, 21, 32] {
            let n = order + 2;
            let leg = LegendreCoeffsSimd::fixed();

            // Scalar column-major (reference implementation)
            let mut v_scalar = test_zero_matrix();
            let mut w_scalar = test_zero_matrix();
            test_ok!(fill_legendre_packed_f64(
                &mut v_scalar,
                &mut w_scalar,
                n,
                x_c2,
                y_c2,
                z_c2,
                c2_re,
            ));

            // SIMD row-major
            let mut v_simd = test_zero_matrix();
            let mut w_simd = test_zero_matrix();

            test_ok!(super::legendre_vw_dispatch(
                &mut v_simd,
                &mut w_simd,
                n,
                x_c2,
                y_c2,
                z_c2,
                c2_re,
                v00,
                &leg,
            ));

            // Compare all elements up to n.
            // SIMD/native codegen can reassociate tiny diagonal terms a few ULP
            // from the scalar column-major path; near-zero values can show larger
            // ULP counts while remaining far below physical precision.
            for l in 0..n {
                for m in 0..=l {
                    let vs = test_ok!(super::matrix_value(&v_scalar, l, m));
                    let vi = test_ok!(super::matrix_value(&v_simd, l, m));
                    if vs == 0.0 && vi == 0.0 {
                        continue;
                    }
                    let ulp_diff = vs.to_bits().abs_diff(vi.to_bits());
                    let abs_diff = (vs - vi).abs();
                    assert!(
                        ulp_diff <= 8 || abs_diff <= 1.0e-28,
                        "V[{l}][{m}] mismatch at order={order}: scalar={vs} simd={vi} ulp_diff={ulp_diff} abs_diff={abs_diff}"
                    );
                    compared += 1;

                    let ws = test_ok!(super::matrix_value(&w_scalar, l, m));
                    let wi = test_ok!(super::matrix_value(&w_simd, l, m));
                    if ws == 0.0 && wi == 0.0 {
                        continue;
                    }
                    let ulp_diff = ws.to_bits().abs_diff(wi.to_bits());
                    let abs_diff = (ws - wi).abs();
                    assert!(
                        ulp_diff <= 8 || abs_diff <= 1.0e-28,
                        "W[{l}][{m}] mismatch at order={order}: scalar={ws} simd={wi} ulp_diff={ulp_diff} abs_diff={abs_diff}"
                    );
                    compared += 1;
                }
            }
        }

        // Corpus is fixed: orders [4, 8, 15, 21, 32] give n = order + 2, and each
        // contributes n(n + 1)/2 (l, m) pairs -- 21 + 55 + 153 + 276 + 595 = 1100
        // pairs, so 2200 V/W comparisons before the zero-skip guards. The floor is
        // set from that corpus size, not from what currently survives.
        assert!(
            compared >= 2000,
            "Legendre SIMD/scalar corpus must compare at least 2000 V/W coefficients, compared {compared}"
        );
    }
}

#[cfg(test)]
mod frame_sibling_equivalence {
    use super::*;

    use common_rs::test_ok;

    use super::test_support::test_set;

    fn assert_f64_array_bits_eq<const N: usize>(left: [f64; N], right: [f64; N]) {
        for (left_value, right_value) in left.into_iter().zip(right) {
            assert_eq!(left_value.to_bits(), right_value.to_bits());
        }
    }

    fn assert_f64_array_bits_ne<const N: usize>(left: [f64; N], right: [f64; N], message: &str) {
        assert!(
            left.into_iter()
                .zip(right)
                .any(|(left_value, right_value)| left_value.to_bits() != right_value.to_bits()),
            "{message}"
        );
    }

    #[test]
    fn frame_packed_shared_cache_observes_changed_coefficients() {
        let stride = 6;
        let order = 4usize;
        let mut c_first = vec![0.0f64; stride * stride];
        let mut s_first = vec![0.0f64; stride * stride];
        test_set(&mut c_first, 0, 1.0);
        test_set(&mut c_first, 2usize.saturating_mul(stride), -1.082_63e-3);
        let degree_two_order_two = 2usize.saturating_mul(stride).saturating_add(2);
        test_set(&mut c_first, degree_two_order_two, 1.574_46e-6);
        test_set(&mut s_first, degree_two_order_two, -9.038_04e-7);
        let packed_first = test_ok!(pack_gravity_coeffs(&c_first, &s_first, stride, order));

        let mut c_second = c_first.clone();
        let mut s_second = s_first.clone();
        test_set(&mut c_second, 2usize.saturating_mul(stride), -2.165_26e-3);
        test_set(
            &mut c_second,
            3usize.saturating_mul(stride).saturating_add(1),
            7.25e-6,
        );
        test_set(&mut s_second, degree_two_order_two, 1.807_608e-6);
        let packed_second = test_ok!(pack_gravity_coeffs(&c_second, &s_second, stride, order));

        let pos_itrs = [-1234.5f64, 5678.9, 3456.7];
        let want_second = test_ok!(spherical_gravity_impl_frame_packed(
            &pos_itrs,
            &mut GravityCache::new(),
            &packed_second,
        ));

        let mut shared = GravityCache::new();
        let first = test_ok!(spherical_gravity_impl_frame_packed(
            &pos_itrs,
            &mut shared,
            &packed_first,
        ));
        assert_f64_array_bits_ne(
            first,
            want_second,
            "test precondition: changed coefficients must change acceleration",
        );
        assert!(matches!(
            shared.vw_state,
            VwCacheState::Ready { covered_order, .. } if covered_order >= order
        ));

        let got_second = test_ok!(spherical_gravity_impl_frame_packed(
            &pos_itrs,
            &mut shared,
            &packed_second,
        ));
        assert_f64_array_bits_eq(got_second, want_second);
        assert!(matches!(
            shared.vw_state,
            VwCacheState::Ready { covered_order, .. } if covered_order >= order
        ));
    }

    #[test]
    fn sincos_shared_cache_observes_coefficients_across_unpacked_and_packed_kernels() {
        let stride = 6;
        let order = 4usize;
        let mut c_first = vec![0.0f64; stride * stride];
        let mut s_first = vec![0.0f64; stride * stride];
        test_set(&mut c_first, 0, 1.0);
        test_set(&mut c_first, 2usize.saturating_mul(stride), -1.082_63e-3);
        let degree_two_order_two = 2usize.saturating_mul(stride).saturating_add(2);
        test_set(&mut c_first, degree_two_order_two, 1.574_46e-6);
        test_set(&mut s_first, degree_two_order_two, -9.038_04e-7);

        let mut c_second = c_first.clone();
        let mut s_second = s_first.clone();
        test_set(&mut c_second, 2usize.saturating_mul(stride), -2.165_26e-3);
        test_set(
            &mut c_second,
            3usize.saturating_mul(stride).saturating_add(1),
            7.25e-6,
        );
        test_set(&mut s_second, degree_two_order_two, 1.807_608e-6);
        let packed_second = test_ok!(pack_gravity_coeffs(&c_second, &s_second, stride, order));

        let state = [-1234.5f64, 5678.9, 3456.7, 1.0, 2.0, 3.0];
        let (sin_gmst, cos_gmst) = 1.234f64.sin_cos();
        let want_second = test_ok!(spherical_gravity_impl_sincos_packed(
            &state,
            sin_gmst,
            cos_gmst,
            &mut GravityCache::new(),
            &packed_second,
        ));

        let mut shared = GravityCache::new();
        let first = test_ok!(spherical_gravity_impl_sincos(
            &state,
            sin_gmst,
            cos_gmst,
            order,
            &c_first,
            &s_first,
            stride,
            &mut shared,
        ));
        let second_raw = test_ok!(spherical_gravity_impl_sincos(
            &state,
            sin_gmst,
            cos_gmst,
            order,
            &c_second,
            &s_second,
            stride,
            &mut GravityCache::new(),
        ));
        assert_f64_array_bits_ne(
            first,
            second_raw,
            "test precondition: changed coefficients must change acceleration",
        );
        assert!(matches!(
            shared.vw_state,
            VwCacheState::Ready { covered_order, .. } if covered_order >= order
        ));

        let got_second = test_ok!(spherical_gravity_impl_sincos_packed(
            &state,
            sin_gmst,
            cos_gmst,
            &mut shared,
            &packed_second,
        ));
        assert_f64_array_bits_eq(got_second, want_second);
        assert!(
            matches!(shared.vw_state, VwCacheState::Empty),
            "the legacy manual recurrence must not publish cross-kernel V/W"
        );
    }

    #[test]
    fn legacy_manual_packed_vw_never_crosses_row_major_kernels() {
        let stride = 6usize;
        let order = 5usize;
        let mut c = vec![0.0f64; stride * stride];
        let s = vec![0.0f64; stride * stride];
        test_set(&mut c, 5usize.saturating_mul(stride), 1.0);
        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));

        let state = [-1234.5f64, 5678.9, 3456.7, 1.0, 2.0, 3.0];
        let (sin_gmst, cos_gmst) = 1.234f64.sin_cos();
        let want_row_major = test_ok!(spherical_gravity_impl_sincos(
            &state,
            sin_gmst,
            cos_gmst,
            order,
            &c,
            &s,
            stride,
            &mut GravityCache::new(),
        ));
        let want_manual = test_ok!(spherical_gravity_impl_sincos_packed(
            &state,
            sin_gmst,
            cos_gmst,
            &mut GravityCache::new(),
            &packed,
        ));
        assert_f64_array_bits_ne(
            want_manual,
            want_row_major,
            "test precondition: order-5 C50 must expose recurrence-bit drift",
        );

        let mut row_then_manual = GravityCache::new();
        test_ok!(spherical_gravity_impl_sincos(
            &state,
            sin_gmst,
            cos_gmst,
            order,
            &c,
            &s,
            stride,
            &mut row_then_manual,
        ));
        assert!(matches!(
            row_then_manual.vw_state,
            VwCacheState::Ready { .. }
        ));
        let got_manual = test_ok!(spherical_gravity_impl_sincos_packed(
            &state,
            sin_gmst,
            cos_gmst,
            &mut row_then_manual,
            &packed,
        ));
        assert_f64_array_bits_eq(got_manual, want_manual);
        assert!(
            matches!(row_then_manual.vw_state, VwCacheState::Empty),
            "manual packed must invalidate overwritten row-major V/W"
        );

        let mut manual_then_row = GravityCache::new();
        test_ok!(spherical_gravity_impl_sincos_packed(
            &state,
            sin_gmst,
            cos_gmst,
            &mut manual_then_row,
            &packed,
        ));
        assert!(
            matches!(manual_then_row.vw_state, VwCacheState::Empty),
            "manual packed must not publish its V/W representation"
        );
        let got_row_major = test_ok!(spherical_gravity_impl_sincos(
            &state,
            sin_gmst,
            cos_gmst,
            order,
            &c,
            &s,
            stride,
            &mut manual_then_row,
        ));
        assert_f64_array_bits_eq(got_row_major, want_row_major);
        assert!(matches!(
            manual_then_row.vw_state,
            VwCacheState::Ready { .. }
        ));
    }

    /// The `_frame` siblings must reproduce the FROZEN `_sincos` entry points
    /// exactly when the caller's rotation is the same z-rotation by GMST that
    /// `_sincos` applies internally. This is what licenses the extraction: any
    /// difference here is a transcription error in the frame-agnostic middle,
    /// not a physics change.
    ///
    /// WHAT THIS WOULD FAIL TO CATCH: it only exercises a z-rotation, because
    /// that is the only frame `_sincos` can express. It therefore validates the
    /// extraction, NOT the full IAU 2006/2000A rotation the RHS will supply —
    /// RED 1 covers that. It also uses fresh caches per call, so it says nothing
    /// about V/W recurrence reuse.
    #[test]
    fn frame_siblings_reproduce_the_frozen_sincos_path() {
        let stride = 6;
        let mut c = vec![0.0f64; stride * stride];
        let mut s = vec![0.0f64; stride * stride];
        test_set(&mut c, 0, 1.0);
        // A few non-trivial harmonics so the summation is actually exercised.
        test_set(&mut c, 2usize.saturating_mul(stride), -1.082_63e-3);
        test_set(&mut c, 3usize.saturating_mul(stride), 2.532_44e-6);
        let degree_two_order_two = 2usize.saturating_mul(stride).saturating_add(2);
        test_set(&mut c, degree_two_order_two, 1.574_46e-6);
        test_set(&mut s, degree_two_order_two, -9.038_04e-7);
        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, 4));

        for state in [
            [6878.0f64, 0.0, 0.0, 0.0, 7.61, 0.0],
            [-1234.5, 5678.9, 3456.7, 1.0, 2.0, 3.0],
            [0.0, 0.0, 7100.0, 0.0, 0.0, 0.0],
        ] {
            for gmst in [0.0f64, 1.234, 3.9, 6.1] {
                let (sn, cs) = gmst.sin_cos();
                for order in [0usize, 2, 4] {
                    // Frozen path.
                    let mut cache_a = GravityCache::new();
                    let want = test_ok!(spherical_gravity_impl_sincos(
                        &state,
                        sn,
                        cs,
                        order,
                        &c,
                        &s,
                        stride,
                        &mut cache_a,
                    ));

                    // Frame path: caller rotates in, calls the sibling, rotates out.
                    let [state_x, state_y, state_z, ..] = state;
                    let pos_itrs = [
                        cs.mul_add(state_x, sn * state_y),
                        (-sn).mul_add(state_x, cs * state_y),
                        state_z,
                    ];
                    let mut cache_b = GravityCache::new();
                    let acc_itrs = test_ok!(spherical_gravity_impl_frame(
                        &pos_itrs,
                        order,
                        &c,
                        &s,
                        stride,
                        &mut cache_b,
                    ));
                    let [acc_x, acc_y, acc_z] = acc_itrs;
                    let got = [
                        cs.mul_add(acc_x, -sn * acc_y),
                        sn.mul_add(acc_x, cs * acc_y),
                        acc_z,
                    ];

                    // Normalise on the VECTOR magnitude, not the component. A
                    // near-zero component is pure cancellation noise, and
                    // per-component normalisation would demand more precision of
                    // it than the arithmetic can carry.
                    let [want_x, want_y, want_z] = want;
                    let want_norm = want_x
                        .mul_add(want_x, want_y.mul_add(want_y, want_z * want_z))
                        .sqrt()
                        .max(f64::MIN_POSITIVE);
                    for (axis, (got_value, want_value)) in got.into_iter().zip(want).enumerate() {
                        assert!(
                            (got_value - want_value).abs() <= 1.0e-13 * want_norm,
                            "order {order} gmst {gmst} axis {axis}: frame {} vs sincos {}, \
                             |delta|/|acc| = {:e}",
                            got_value,
                            want_value,
                            (got_value - want_value).abs() / want_norm
                        );
                    }

                    // Packed sibling against the frozen packed path.
                    let packed_order = test_ok!(packed.truncated_to(order));
                    let mut cache_c = GravityCache::new();
                    let want_p = test_ok!(spherical_gravity_impl_sincos_packed(
                        &state,
                        sn,
                        cs,
                        &mut cache_c,
                        &packed_order,
                    ));
                    let mut cache_d = GravityCache::new();
                    let acc_p = test_ok!(spherical_gravity_impl_frame_packed(
                        &pos_itrs,
                        &mut cache_d,
                        &packed_order,
                    ));
                    let [packed_acc_x, packed_acc_y, packed_acc_z] = acc_p;
                    let got_p = [
                        cs.mul_add(packed_acc_x, -sn * packed_acc_y),
                        sn.mul_add(packed_acc_x, cs * packed_acc_y),
                        packed_acc_z,
                    ];
                    let [want_p_x, want_p_y, want_p_z] = want_p;
                    let want_p_norm = want_p_x
                        .mul_add(want_p_x, want_p_y.mul_add(want_p_y, want_p_z * want_p_z))
                        .sqrt()
                        .max(f64::MIN_POSITIVE);
                    for (axis, (got_value, want_value)) in got_p.into_iter().zip(want_p).enumerate()
                    {
                        assert!(
                            (got_value - want_value).abs() <= 1.0e-13 * want_p_norm,
                            "PACKED order {order} gmst {gmst} axis {axis}: frame {} vs sincos {}, \
                             |delta|/|acc| = {:e}",
                            got_value,
                            want_value,
                            (got_value - want_value).abs() / want_p_norm
                        );
                    }
                }
            }
        }
    }
}

/// Bounded `reset` clear span.
///
/// `reset` clears only the square prefix the recorded high-water order can
/// reach. These tests hold it to the property the unbounded clear had: after
/// `reset` the whole workspace is zero, whatever order or kernel filled it.
///
/// Its own module so it can own its fixtures without threading them past the
/// long-standing test modules above.
#[cfg(test)]
mod bounded_reset_span {
    use super::*;

    use common_rs::test_ok;

    use super::test_support::test_set;

    fn assert_f64_array_bits_eq<const N: usize>(left: [f64; N], right: [f64; N]) {
        for (left_value, right_value) in left.into_iter().zip(right) {
            assert_eq!(left_value.to_bits(), right_value.to_bits());
        }
    }

    /// Harmonic coefficients wide enough to fill `order`.
    fn reset_probe_coeffs(order: usize) -> (Vec<f64>, Vec<f64>, usize) {
        let stride = order.saturating_add(1);
        let cells = stride.saturating_mul(stride);
        let mut c = vec![0.0f64; cells];
        let mut s = vec![0.0f64; cells];
        test_set(&mut c, 0, 1.0);
        for degree in 2..=order {
            let base = degree.saturating_mul(stride);
            test_set(&mut c, base, -1.082_63e-3);
            let diagonal = base.saturating_add(degree);
            test_set(&mut c, diagonal, 1.574_46e-6);
            test_set(&mut s, diagonal, -9.038_04e-7);
        }
        (c, s, stride)
    }

    /// What the packed summation actually walks at the flown order, and the
    /// evidence the K8 close of that lane rests on (`docs/ARC_COST_MAP.md`).
    ///
    /// `gravity_summation_f64_packed` is 6.68% of the production arc, and the
    /// obvious reading of it — a wide SIMD kernel worth optimising as one — is
    /// wrong: at order 5 the quad path covers **8 of 15 terms in two quads**
    /// and the other seven fall to the scalar tail. Deleting the quad path
    /// entirely measured as a null on the arc, which is only interpretable
    /// alongside these counts.
    ///
    /// Asserted rather than merely printed, because the close is only valid
    /// while the counts hold. If a packing change moves them, this goes red and
    /// the lane needs re-pricing rather than silently inheriting a stale
    /// verdict.
    ///
    /// The field is dense on purpose: a real gravity model populates every
    /// `C(l,m)`/`S(l,m)`, and `reset_probe_coeffs` is deliberately sparse, so
    /// packing that instead would count a shape production never flies.
    #[test]
    fn packed_summation_work_at_production_order_is_mostly_scalar() {
        let order = GRAVITY_FAST_PATH_ORDER_CAP;
        let stride = order + 1;
        let mut c = vec![0.0f64; stride * stride];
        let mut s = vec![0.0f64; stride * stride];
        // Magnitudes are irrelevant to a census -- packing keys on which terms
        // are NONZERO -- so these are constants rather than an index-derived
        // value that would need a `usize`-to-`f64` cast the lint refuses.
        for degree in 0..=order {
            for m in 0..=degree {
                test_set(&mut c, degree * stride + m, 1e-6);
                if m > 0 {
                    test_set(&mut s, degree * stride + m, 2e-6);
                }
            }
        }
        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        let quads: usize = packed.rows.iter().map(|row| row.quads.len()).sum();
        let terms: usize = packed.rows.iter().map(|row| row.terms.len()).sum();

        assert!(
            packed.max_order <= GRAVITY_FAST_PATH_ORDER_CAP
                && packed.dense_prefix >= packed.max_order,
            "production order must take the low-order dense path, or the census below \
             describes a branch it does not run"
        );
        assert_eq!(terms, 15, "term count at the flown order");
        assert_eq!(quads, 2, "quad count at the flown order");
        assert!(
            terms - quads * 4 > quads * 4 / 2,
            "the scalar tail ({} terms) is no longer a large fraction of the {terms} \
             flown terms; the K8 verdict that this lane has no vector-vs-scalar lever \
             was measured against 8 lanes and 7 tail terms and must be re-taken",
            terms - quads * 4,
        );
    }

    /// Fill `cache` through the row-major kernel, which publishes a `Ready` state.
    fn reset_probe_fill(
        cache: &mut GravityCache,
        order: usize,
        state: &[f64; 6],
    ) -> Result<[f64; 3], GravityError> {
        let (c, s, stride) = reset_probe_coeffs(order);
        let (sin_gmst, cos_gmst) = 1.234f64.sin_cos();
        spherical_gravity_impl_sincos(state, sin_gmst, cos_gmst, order, &c, &s, stride, cache)
    }

    fn assert_workspace_all_zero(cache: &GravityCache, label: &str) {
        for (name, matrix) in [("v", &cache.v), ("w", &cache.w)] {
            for (row_index, row) in matrix.iter().enumerate() {
                for (column, value) in row.iter().enumerate() {
                    assert_eq!(
                        value.to_bits(),
                        0.0f64.to_bits(),
                        "{label}: {name}[{row_index}][{column}] survived reset"
                    );
                }
            }
        }
    }

    fn assert_workspaces_bits_eq(left: &GravityCache, right: &GravityCache, label: &str) {
        for (name, (left_matrix, right_matrix)) in
            [("v", (&left.v, &right.v)), ("w", (&left.w, &right.w))]
        {
            for (row_index, (left_row, right_row)) in
                left_matrix.iter().zip(right_matrix.iter()).enumerate()
            {
                for (column, (got, want)) in left_row.iter().zip(right_row.iter()).enumerate() {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{label}: {name}[{row_index}][{column}] differs from a fresh cache"
                    );
                }
            }
        }
    }

    /// The bounded clear must cover everything a wide fill wrote.
    ///
    /// Order 20 reaches rows and columns 0..=21, far past the sealed order-5
    /// span of 0..=6, so a clear bound that ignored the high-water mark — or
    /// bounded the row count without bounding the row length — leaves nonzero
    /// cells behind.
    #[test]
    fn reset_clears_every_cell_a_wide_fill_reached() {
        let state = [-1234.5f64, 5678.9, 3456.7, 1.0, 2.0, 3.0];
        let mut cache = GravityCache::new();
        let acceleration = test_ok!(reset_probe_fill(&mut cache, 20, &state));
        assert!(
            acceleration.iter().any(|value| *value != 0.0),
            "test precondition: the wide fill must produce a real acceleration"
        );
        assert!(
            cache
                .v
                .iter()
                .flat_map(|row| row.iter())
                .any(|value| *value != 0.0),
            "test precondition: the wide fill must leave nonzero workspace cells"
        );

        cache.reset();

        assert_workspace_all_zero(&cache, "order 20 fill");
        assert!(matches!(cache.vw_state, VwCacheState::Empty));
    }

    /// The clear bound must not shrink when a narrower fill follows a wide one
    /// WITHIN one fill cycle — that is, with no clear between them.
    ///
    /// A shared cache — the thread-local one, or an RHS reused across segments —
    /// legitimately serves several orders. If `mark_vw_ready` overwrote the bound
    /// instead of raising it, the order-5 fill here would shrink it to 7 and the
    /// order-20 residue in rows 8..=21 would survive.
    ///
    /// This says nothing about the bound ACROSS a clear, and must not be read as
    /// licensing a mark that is monotone for the life of the cache: `reset` ends
    /// the cycle and drops the bound. See
    /// `reset_drops_the_bound_so_a_later_narrow_fill_clears_narrowly`.
    #[test]
    fn reset_bound_never_shrinks_after_a_narrower_fill() {
        let wide_state = [-1234.5f64, 5678.9, 3456.7, 1.0, 2.0, 3.0];
        let narrow_state = [4321.0f64, -2468.0, 5100.5, 1.0, 2.0, 3.0];
        let mut cache = GravityCache::new();
        let _wide = test_ok!(reset_probe_fill(&mut cache, 20, &wide_state));
        let _narrow = test_ok!(reset_probe_fill(&mut cache, 5, &narrow_state));

        cache.reset();

        assert_workspace_all_zero(&cache, "order 20 then order 5 fill");
    }

    /// A reset cache must be indistinguishable from a fresh one for later fills.
    ///
    /// This is the arm that catches a clear bound that is merely too small: the
    /// order-5 refill rewrites only rows and columns 0..=6, so any order-20
    /// residue the reset failed to clear stays readable and shows up here.
    #[test]
    fn narrow_fill_after_reset_matches_a_fresh_cache() {
        let wide_state = [-1234.5f64, 5678.9, 3456.7, 1.0, 2.0, 3.0];
        let narrow_state = [4321.0f64, -2468.0, 5100.5, 1.0, 2.0, 3.0];

        let mut reused = GravityCache::new();
        let _wide = test_ok!(reset_probe_fill(&mut reused, 20, &wide_state));
        reused.reset();
        let got = test_ok!(reset_probe_fill(&mut reused, 5, &narrow_state));

        let mut fresh = GravityCache::new();
        let want = test_ok!(reset_probe_fill(&mut fresh, 5, &narrow_state));

        assert_f64_array_bits_eq(got, want);
        assert_workspaces_bits_eq(&reused, &fresh, "order 5 refill after reset");
    }

    /// Kernels that never publish a `Ready` state must still bound the clear.
    ///
    /// `spherical_gravity_impl_sincos_packed` deliberately leaves `vw_state`
    /// `Empty` — its legacy recurrence must not be consumed across kernels — so
    /// a clear bound fed only by `mark_vw_ready` would never learn how wide this
    /// fill was, and would strand its workspace. The dual-number RHS reaches the
    /// same shape through `spherical_gravity_impl_generic_packed`.
    #[test]
    fn reset_covers_a_fill_that_never_publishes_ready_state() {
        let state = [-1234.5f64, 5678.9, 3456.7, 1.0, 2.0, 3.0];
        let order = 20usize;
        let (c, s, stride) = reset_probe_coeffs(order);
        let packed = test_ok!(pack_gravity_coeffs(&c, &s, stride, order));
        let (sin_gmst, cos_gmst) = 1.234f64.sin_cos();
        let mut cache = GravityCache::new();
        let _acceleration = test_ok!(spherical_gravity_impl_sincos_packed(
            &state, sin_gmst, cos_gmst, &mut cache, &packed,
        ));
        assert!(
            matches!(cache.vw_state, VwCacheState::Empty),
            "test precondition: this kernel must not publish a Ready state"
        );

        cache.reset();

        assert_workspace_all_zero(&cache, "unpublished order 20 fill");
    }

    /// A cache that was never filled is already zero, so its clear is trivial.
    #[test]
    fn reset_on_an_unfilled_cache_leaves_it_zero() {
        let mut cache = GravityCache::new();
        assert_eq!(cache.vw_live_span(), 2);
        cache.reset();
        assert_workspace_all_zero(&cache, "never filled");
        assert_eq!(cache.vw_live_span(), 2);
    }

    /// `reset` must end the fill cycle, not just clear within it.
    ///
    /// A mark that stayed monotone for the life of the cache would be sound but
    /// wasteful: one cold-path order-20 fill would make every later order-5
    /// reset clear 22x22 instead of 7x7, permanently. `THREAD_GRAVITY_CACHE` is
    /// one cache per thread shared across callers that need not all run at the
    /// sealed order, so that is reachable rather than hypothetical.
    #[test]
    fn reset_drops_the_bound_so_a_later_narrow_fill_clears_narrowly() {
        let wide_state = [-1234.5f64, 5678.9, 3456.7, 1.0, 2.0, 3.0];
        let mut cache = GravityCache::new();
        let _wide = test_ok!(reset_probe_fill(&mut cache, 20, &wide_state));
        assert_eq!(
            cache.vw_live_span(),
            22,
            "test precondition: the wide fill must widen the bound"
        );

        cache.reset();
        assert_eq!(
            cache.vw_live_span(),
            2,
            "reset leaves the whole workspace zero, so it must drop the bound"
        );

        cache.begin_vw_fill(5);
        assert_eq!(
            cache.vw_live_span(),
            7,
            "the next fill must raise the bound again, to its own width only"
        );
    }

    /// A workspace sized to its order must behave exactly like a full one, and
    /// must DEGRADE rather than break when asked for more than it can hold.
    ///
    /// Three claims in one test, because they are one property:
    ///
    /// 1. A `with_rows(7)` cache produces bit-identical results to `new()` at
    ///    order 5 — the short allocation changes storage, not arithmetic.
    /// 2. `reset` still leaves the WHOLE short allocation zero, so the "every
    ///    cell outside the live prefix still holds its constructed zero"
    ///    invariant survives the smaller `vw_live_span` clamp.
    /// 3. An order-20 fill on that cache returns
    ///    `GravityError::InvariantViolation` — a typed error, not a panic and
    ///    not out-of-bounds writes — and the failed fill's partial residue is
    ///    still inside what `reset` clears. The fill dies in the `m = 0` column
    ///    loop the moment it reaches the first missing row, before the loops
    ///    that write any column past 0.
    #[test]
    fn a_short_cache_matches_a_full_one_and_errors_instead_of_overrunning() {
        let narrow_state = [4321.0f64, -2468.0, 5100.5, 1.0, 2.0, 3.0];
        let wide_state = [-1234.5f64, 5678.9, 3456.7, 1.0, 2.0, 3.0];

        let mut short = GravityCache::with_rows(7);
        assert_eq!(short.v.len(), 7, "with_rows must size the row count");
        assert_eq!(short.w.len(), 7);
        assert_eq!(
            short.v.first().map_or(0, |row| row.len()),
            super::MAX_RECURSIVE_ORDER,
            "the row LENGTH is fixed by the array type and must not shrink"
        );
        assert_workspace_all_zero(&short, "freshly built short cache");

        // 1. Same bits as the full-width cache at the order it was sized for.
        let got = test_ok!(reset_probe_fill(&mut short, 5, &narrow_state));
        let mut full = GravityCache::new();
        let want = test_ok!(reset_probe_fill(&mut full, 5, &narrow_state));
        assert_f64_array_bits_eq(got, want);

        // 2. The bounded clear still zeroes the entire short allocation.
        short.reset();
        assert_workspace_all_zero(&short, "order 5 fill in a short cache");

        // 3. Overrunning it is a typed error, and its residue is still covered.
        let overrun = reset_probe_fill(&mut short, 20, &wide_state);
        assert_eq!(
            overrun,
            Err(GravityError::InvariantViolation),
            "a fill wider than the allocation must return a typed error"
        );
        short.reset();
        assert_workspace_all_zero(&short, "failed order 20 fill in a short cache");
    }

    /// The clamps on `with_rows` are the constructor's whole safety argument.
    ///
    /// Below `VW_MIN_ROWS` every fill would fail on its unconditional row-1
    /// base cases, so the floor is what keeps a short cache usable at all;
    /// above `MAX_RECURSIVE_ORDER` the extra rows are unreachable, because no
    /// fill can exceed `MAX_ORDER + 2`.
    #[test]
    fn with_rows_clamps_both_ends_and_stays_zeroed() {
        for (requested, expected) in [
            (0usize, super::VW_MIN_ROWS),
            (1, super::VW_MIN_ROWS),
            (2, 2),
            (7, 7),
            (super::MAX_RECURSIVE_ORDER, super::MAX_RECURSIVE_ORDER),
            (usize::MAX, super::MAX_RECURSIVE_ORDER),
        ] {
            let cache = GravityCache::with_rows(requested);
            assert_eq!(cache.v.len(), expected, "with_rows({requested}) row count");
            assert_eq!(cache.w.len(), expected, "with_rows({requested}) row count");
            assert_workspace_all_zero(&cache, "fresh with_rows cache");
            assert_eq!(
                cache.vw_live_span(),
                2,
                "with_rows({requested}) must start with an empty fill cycle"
            );
        }
        assert_eq!(
            GravityCache::new().v.len(),
            super::MAX_RECURSIVE_ORDER,
            "new() must stay at the full width benches and oracles depend on"
        );

        // The point of the exercise, measured off the real slice lengths rather
        // than restated from a comment: `v` and `w` are the only OWNED tables,
        // each row is a `[f64; MAX_RECURSIVE_ORDER]` whose length the row count
        // cannot change.
        let owned_bytes = |cache: &GravityCache| {
            cache
                .v
                .len()
                .saturating_add(cache.w.len())
                .saturating_mul(size_of::<[f64; super::MAX_RECURSIVE_ORDER]>())
        };
        assert_eq!(owned_bytes(&GravityCache::new()), 274_576);
        assert_eq!(owned_bytes(&GravityCache::with_rows(7)), 14_672);
    }

    /// `prime_storage` clears everything, so it ends the fill cycle too.
    #[test]
    fn prime_storage_clears_everything_and_drops_the_bound() {
        let wide_state = [-1234.5f64, 5678.9, 3456.7, 1.0, 2.0, 3.0];
        let mut cache = GravityCache::new();
        let _wide = test_ok!(reset_probe_fill(&mut cache, 20, &wide_state));

        cache.prime_storage();

        assert_workspace_all_zero(&cache, "after prime_storage");
        assert_eq!(cache.vw_live_span(), 2);
        assert!(matches!(cache.vw_state, VwCacheState::Empty));
    }
}

/// R54 — the const-generic Legendre dispatch.
///
/// Three obligations, all in-tree because prose cannot hold a bit claim:
/// the two arms agree to the bit, the dispatch routes only the production trip
/// count to the const arm, and the two bodies are the same body.
#[cfg(test)]
mod const_generic_legendre_dispatch {
    use super::*;

    use common_rs::test_ok;

    /// Derive the fill's five inputs from an ECEF position exactly as every
    /// production evaluator does, so the sweep exercises representative
    /// magnitudes rather than invented ones.
    fn fill_inputs(pos: [f64; 3]) -> (f64, f64, f64, f64, f64) {
        let [pos_x, pos_y, pos_z] = pos;
        let re = GRAVITY_REFERENCE_RADIUS_KM;
        let r = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z).sqrt();
        let c2 = re / (r * r);
        let c2_re = c2 * re;
        (pos_x * c2, pos_y * c2, pos_z * c2, c2_re, c2_re.sqrt())
    }

    /// LEO, a high-inclination pass, a near-polar state, GEO, and a near-equatorial
    /// state — enough spread that a bit difference confined to one regime shows.
    const SWEEP_POSITIONS: [[f64; 3]; 5] = [
        [6800.0, 0.0, 0.0],
        [3500.0, 4200.0, 4900.0],
        [120.0, -260.0, 6980.0],
        [42164.0, 0.0, 0.0],
        [-5100.0, 4600.0, 15.0],
    ];

    /// The production threshold plus shapes either side of it: 99 keeps every
    /// `l` row scalar, 0 and 3 push rows into `legendre_l_row_simd` that
    /// production never sends there.
    const SWEEP_THRESHOLDS: [usize; 5] = [LEGENDRE_SIMD_L_THRESHOLD, 0, 3, 6, 99];

    fn workspace() -> Vec<[f64; MAX_RECURSIVE_ORDER]> {
        vec![[0.0; MAX_RECURSIVE_ORDER]; MAX_RECURSIVE_ORDER]
    }

    fn differing_cells(
        left: &[[f64; MAX_RECURSIVE_ORDER]],
        right: &[[f64; MAX_RECURSIVE_ORDER]],
    ) -> usize {
        left.iter()
            .flatten()
            .zip(right.iter().flatten())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    }

    fn nonzero_cells(values: &[[f64; MAX_RECURSIVE_ORDER]]) -> usize {
        values.iter().flatten().filter(|cell| **cell != 0.0).count()
    }

    /// Run both arms at trip count `N` and report `(differing cells, cells the
    /// runtime arm actually wrote)`. The second number is the vacuity guard: a
    /// pair of no-ops would agree to the bit and prove nothing.
    fn arm_pair<const N: usize>(
        leg: &LegendreCoeffsSimd,
        pos: [f64; 3],
        threshold: usize,
    ) -> Result<(usize, usize), GravityError> {
        let (x_c2, y_c2, z_c2, c2_re, v00) = fill_inputs(pos);
        let (mut v_runtime, mut w_runtime) = (workspace(), workspace());
        let (mut v_const, mut w_const) = (workspace(), workspace());
        legendre_vw_row_major_with_threshold(
            &mut v_runtime,
            &mut w_runtime,
            N,
            x_c2,
            y_c2,
            z_c2,
            c2_re,
            v00,
            leg,
            threshold,
        )?;
        legendre_vw_row_major_const::<N>(
            &mut v_const,
            &mut w_const,
            x_c2,
            y_c2,
            z_c2,
            c2_re,
            v00,
            leg,
            threshold,
        )?;
        let differing = differing_cells(&v_runtime, &v_const)
            .saturating_add(differing_cells(&w_runtime, &w_const));
        let written = nonzero_cells(&v_runtime).saturating_add(nonzero_cells(&w_runtime));
        Ok((differing, written))
    }

    /// GATE (a). The whole lever rests on this: a compile-time trip count may
    /// unroll and reschedule, but it may not move a single bit. If this reds,
    /// the lever is re-pin class and per its parked conditions it is dead.
    ///
    /// Swept over five positions, five SIMD thresholds, and five trip counts
    /// either side of production's, so the equality is not an `N = 7`
    /// coincidence.
    #[test]
    fn legendre_const_arm_is_bit_identical_to_runtime_arm() {
        let leg = LegendreCoeffsSimd::fixed();
        for pos in SWEEP_POSITIONS {
            for threshold in SWEEP_THRESHOLDS {
                let cases = [
                    (5_usize, test_ok!(arm_pair::<5>(&leg, pos, threshold))),
                    (6, test_ok!(arm_pair::<6>(&leg, pos, threshold))),
                    (
                        PROD_LEGENDRE_N,
                        test_ok!(arm_pair::<7>(&leg, pos, threshold)),
                    ),
                    (8, test_ok!(arm_pair::<8>(&leg, pos, threshold))),
                    (9, test_ok!(arm_pair::<9>(&leg, pos, threshold))),
                ];
                for (n, (differing, written)) in cases {
                    assert!(
                        written >= n,
                        "vacuous case: runtime arm wrote {written} cells at \
                         n={n} threshold={threshold} pos={pos:?}"
                    );
                    assert_eq!(
                        differing, 0,
                        "const arm MOVED BITS at n={n} threshold={threshold} \
                         pos={pos:?}: {differing} differing f64 cells"
                    );
                }
            }
        }
    }

    /// GATE (b), part one — a tripwire, and worth being exact about what it can
    /// and cannot prove.
    ///
    /// The fact that matters is that `PROD_LEGENDRE_N` equals production's
    /// `gravity_order` plus the recurrence's two extra rows. Production's order
    /// lives in `nd_config::part_a_science` (`gravity_order: 5`), which is
    /// downstream of this crate and so unreachable from here — asserting `7 == 7`
    /// against our own constant would prove nothing.
    ///
    /// What this crate does hold is [`GRAVITY_FAST_PATH_ORDER_CAP`], whose value
    /// is that same 5 for the same reason: it is the order the campaign flies.
    /// The two are independent definitions that happen to track one number, so
    /// this catches the drift case that matters — an order change that lands on
    /// one and not the other, which would silently route production back to the
    /// runtime arm and lose the lever with every test still green.
    ///
    /// It is a coincidence detector, not a derivation. If you changed the fast
    /// path's cap rather than the flown order, this red is the false alarm of
    /// the pair: confirm `nd_config`'s `gravity_order` and update the constant
    /// here to match it, not to match the cap.
    #[test]
    fn const_arm_is_built_for_the_production_trip_count() {
        assert_eq!(
            PROD_LEGENDRE_N,
            GRAVITY_FAST_PATH_ORDER_CAP.saturating_add(2),
            "the monomorphised trip count has drifted from the flown order + 2"
        );
    }

    /// GATE (b), part two: order 5 takes the const arm and every other order
    /// takes the runtime one — read off the arm the real `match` returned, not
    /// off a second copy of its predicate. The dispatch's output is then held
    /// to the runtime body's bits, so the wired route is checked and not just
    /// the two functions in isolation.
    #[test]
    fn legendre_dispatch_routes_only_the_production_trip_count_to_the_const_arm() {
        let leg = LegendreCoeffsSimd::fixed();
        for pos in SWEEP_POSITIONS {
            let (x_c2, y_c2, z_c2, c2_re, v00) = fill_inputs(pos);
            for n in 0_usize..=12 {
                let (mut v_dispatch, mut w_dispatch) = (workspace(), workspace());
                let arm = test_ok!(legendre_vw_dispatch(
                    &mut v_dispatch,
                    &mut w_dispatch,
                    n,
                    x_c2,
                    y_c2,
                    z_c2,
                    c2_re,
                    v00,
                    &leg,
                ));
                let expected = if n == PROD_LEGENDRE_N {
                    LegendreArm::ConstProd
                } else {
                    LegendreArm::Runtime
                };
                assert_eq!(arm, expected, "dispatch took the wrong arm at n={n}");

                let (mut v_runtime, mut w_runtime) = (workspace(), workspace());
                test_ok!(legendre_vw_row_major_with_threshold(
                    &mut v_runtime,
                    &mut w_runtime,
                    n,
                    x_c2,
                    y_c2,
                    z_c2,
                    c2_re,
                    v00,
                    &leg,
                    LEGENDRE_SIMD_L_THRESHOLD,
                ));
                let differing = differing_cells(&v_dispatch, &v_runtime)
                    .saturating_add(differing_cells(&w_dispatch, &w_runtime));
                assert_eq!(
                    differing, 0,
                    "dispatched fill differs from the runtime body at n={n} pos={pos:?}"
                );
            }
        }
    }
}

/// R56 in-situ price of the const-generic Legendre dispatch.
///
/// R51 measured the two arms behind `#[inline(never)]`, which is *not* how
/// production runs them: `docs/PMU_PROFILE.md` §2a found the fill inlined into
/// `spherical_gravity_impl_frame_packed` with no symbol of its own. This module
/// measures them the way they are actually built — both `#[inline]`, entered
/// through the real dispatch — and charges the dispatch branch to the const arm,
/// which is the arm that has to pay for it.
///
/// It reports ns/call only. The arc share is `calls_per_arc * delta_ns /
/// arc_wall_ns`, and the first factor came from the `legendre-probe` fill
/// counter (retired 2026-08-20 with its feature key), not from here.
#[cfg(test)]
#[expect(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::float_arithmetic,
    reason = "a cfg(test) timing harness: raw ns arithmetic and usize-to-f64 casts"
)]
mod const_generic_legendre_price {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    /// Positions the fill actually sees on a LEO arc: a 12 h propagation from a
    /// ~6,780 km state walks the whole sphere, so the sweep is a ring of unit
    /// directions at LEO radius rather than five hand-picked regimes.
    fn ring_inputs(count: usize) -> Vec<(f64, f64, f64, f64, f64)> {
        (0..count)
            .map(|index| {
                let step = index as f64 * 0.618_033_988_75;
                let (sin_lat, cos_lat) = (step * 2.4).sin_cos();
                let (sin_lon, cos_lon) = (step * 5.1).sin_cos();
                let radius = 6_780.0 + 40.0 * (step * 1.7).sin();
                let pos = [
                    radius * cos_lat * cos_lon,
                    radius * cos_lat * sin_lon,
                    radius * sin_lat,
                ];
                let [pos_x, pos_y, pos_z] = pos;
                let re = GRAVITY_REFERENCE_RADIUS_KM;
                let r = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z).sqrt();
                let c2 = re / (r * r);
                let c2_re = c2 * re;
                (pos_x * c2, pos_y * c2, pos_z * c2, c2_re, c2_re.sqrt())
            })
            .collect()
    }

    /// One timed block. `dispatched` selects the const arm *through the real
    /// `match`*, so its cost includes the branch; the other arm calls the
    /// runtime body directly, which is exactly what production ran before the
    /// dispatch landed.
    fn block(
        dispatched: bool,
        inputs: &[(f64, f64, f64, f64, f64)],
        leg: &LegendreCoeffsSimd,
        v: &mut [[f64; MAX_RECURSIVE_ORDER]],
        w: &mut [[f64; MAX_RECURSIVE_ORDER]],
        calls: usize,
    ) -> f64 {
        let start = Instant::now();
        for index in 0..calls {
            let Some(&(x_c2, y_c2, z_c2, c2_re, v00)) = inputs.get(index % inputs.len()) else {
                continue;
            };
            if dispatched {
                let _ = black_box(legendre_vw_dispatch(
                    black_box(v),
                    black_box(w),
                    black_box(PROD_LEGENDRE_N),
                    x_c2,
                    y_c2,
                    z_c2,
                    c2_re,
                    v00,
                    leg,
                ));
            } else {
                let _ = black_box(legendre_vw_row_major_with_threshold(
                    black_box(v),
                    black_box(w),
                    black_box(PROD_LEGENDRE_N),
                    x_c2,
                    y_c2,
                    z_c2,
                    c2_re,
                    v00,
                    leg,
                    LEGENDRE_SIMD_L_THRESHOLD,
                ));
            }
        }
        black_box(&v);
        black_box(&w);
        start.elapsed().as_secs_f64() / calls as f64 * 1e9
    }

    /// Minimum of many short blocks, arm order alternated block by block so a
    /// thermal or scheduling drift cannot land on one arm.
    #[test]
    #[ignore = "measurement harness; prints ns/call for both Legendre arms"]
    fn legendre_arm_price_ns_per_call() {
        const BLOCKS: usize = 24;
        const CALLS: usize = 50_000;

        let leg = LegendreCoeffsSimd::fixed();
        let inputs = ring_inputs(512);
        let mut v = vec![[0.0; MAX_RECURSIVE_ORDER]; MAX_RECURSIVE_ORDER];
        let mut w = vec![[0.0; MAX_RECURSIVE_ORDER]; MAX_RECURSIVE_ORDER];

        // Warm the caches and let the CPU reach a steady clock before any
        // block counts.
        for _ in 0..4 {
            let _ = block(true, &inputs, &leg, &mut v, &mut w, CALLS);
            let _ = block(false, &inputs, &leg, &mut v, &mut w, CALLS);
        }

        let mut best_dispatch = f64::INFINITY;
        let mut best_runtime = f64::INFINITY;
        for round in 0..BLOCKS {
            // Alternate which arm goes first.
            let order = [round % 2 == 0, round % 2 != 0];
            for dispatched in order {
                let ns = block(dispatched, &inputs, &leg, &mut v, &mut w, CALLS);
                if dispatched {
                    best_dispatch = best_dispatch.min(ns);
                } else {
                    best_runtime = best_runtime.min(ns);
                }
            }
        }

        let delta = best_runtime - best_dispatch;
        let pct = delta / best_runtime * 100.0;
        println!(
            "LEGENDRE_PRICE blocks={BLOCKS} calls_per_block={CALLS} \
             runtime={best_runtime:.3} ns/call dispatch_const={best_dispatch:.3} ns/call \
             delta={delta:.3} ns pct={pct:+.2}%"
        );
        assert!(
            best_runtime.is_finite() && best_dispatch.is_finite(),
            "both arms must produce a finite time"
        );
    }
}
