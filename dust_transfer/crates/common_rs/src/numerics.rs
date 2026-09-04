//! Numerical stability utilities.
//!
//! Provides functions for numerically stable computation of exponentials.

/// Log of smallest positive f64 (approximately -708.4).
/// Used to clamp underflow in scalar exponential computations.
pub const LOG_DBL_MIN: f64 = -708.396_418_532_26;

/// Log of largest finite f64 (approximately 709.8).
/// Used to clamp overflow in scalar exponential computations.
pub const LOG_DBL_MAX: f64 = 709.782_712_893_384;

// There used to be two SIMD siblings here, `safe_exp_f64x2` and
// `safe_exp_f64x4`, and a `tests/exp_lane_homogeneity.rs` guarding all three
// against each other. Their only caller was `dust_estimates_rs::pdf`, which
// summed one GMM mixture through all three widths at once (f64x4 for the first
// 4k components, f64x2 for a 2-remainder, scalar for a final odd one); the
// probabilistic mass search that owned it was deleted in 0f7e079, and the
// wrappers went with it. Only the scalar path below remains, so lane-width
// homogeneity is no longer a property this module can violate.
//
// The history is worth keeping because it cost a real defect. The SIMD pair once
// clamped to a tighter ±700 pair, justified as "staying within the polynomial's
// accurate range", while the scalar clamped to LOG_DBL_MIN/LOG_DBL_MAX. That
// made a mixture sum depend on `n % 4` rather than on the physics: for a
// log-value in (700, LOG_DBL_MAX] the paths differed by a factor of 1.772467e4,
// and in [LOG_DBL_MIN, -700) by 2.256741e-4. The "accurate range" premise was
// then measured over 4e6 points spanning the full clamp and found false --
// wide's polynomial is within 1 ULP of `f64::exp` everywhere in it -- so the
// bounds were unified rather than the SIMD path narrowed. If a SIMD exp is ever
// reintroduced here, it must clamp to these same two constants, and it needs the
// homogeneity test back.

/// Safe exponential that handles underflow/overflow.
///
/// Clamps the input to avoid returning 0.0 (underflow) or infinity (overflow).
///
/// # Arguments
/// * `log_value` - The exponent to compute `exp()` of
///
/// # Returns
/// * `exp(log_value)` clamped to `[MIN_POSITIVE, MAX]`
///
/// # Examples
/// ```
/// use common_rs::safe_exp;
///
/// assert!(safe_exp(-1000.0) > 0.0);  // Would underflow to 0 without clamping
/// assert!(safe_exp(1000.0).is_finite());  // Would overflow to inf without clamping
/// assert!((safe_exp(0.0) - 1.0).abs() < 1e-10);
/// ```
#[inline]
#[must_use]
pub fn safe_exp(log_value: f64) -> f64 {
    // Branchless clamping enables LLVM auto-vectorization when called in loops.
    // f64::clamp compiles to FMAXNM/FMINNM on aarch64, VMAXPD/VMINPD on x86.
    log_value.clamp(LOG_DBL_MIN, LOG_DBL_MAX).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_exp_normal() {
        assert!((safe_exp(0.0) - 1.0).abs() < 1e-10);
        assert!((safe_exp(1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_safe_exp_underflow() {
        // Would be 0.0 without clamping
        assert!(safe_exp(-1000.0) > 0.0);
        // Branchless clamp: exp(LOG_DBL_MIN) ≈ MIN_POSITIVE (within ULP)
        let v = safe_exp(LOG_DBL_MIN - 1.0);
        assert!(v > 0.0 && v <= f64::MIN_POSITIVE * 2.0);
    }

    #[test]
    fn test_safe_exp_overflow() {
        // Would be inf without clamping
        assert!(safe_exp(1000.0).is_finite());
        // Branchless clamp: exp(LOG_DBL_MAX) ≈ MAX (within ULP)
        let v = safe_exp(LOG_DBL_MAX + 1.0);
        assert!(v.is_finite() && v >= f64::MAX / 2.0);
    }
}
