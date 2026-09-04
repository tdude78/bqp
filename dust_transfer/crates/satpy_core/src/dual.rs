use num_traits::{Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use wide::f64x4;

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct DualVec(pub f64x4);

impl DualVec {
    #[inline]
    #[must_use]
    pub fn new(v: f64, d: nalgebra::Vector3<f64>) -> Self {
        Self(f64x4::from([v, d.x, d.y, d.z]))
    }

    #[inline]
    #[must_use]
    pub fn constant(v: f64) -> Self {
        Self(f64x4::from([v, 0.0, 0.0, 0.0]))
    }

    #[inline]
    #[must_use]
    pub const fn v(&self) -> f64 {
        let &[value, _, _, _] = self.0.as_array();
        value
    }

    #[inline]
    #[must_use]
    pub const fn d(&self) -> [f64; 3] {
        let &[_, dx, dy, dz] = self.0.as_array();
        [dx, dy, dz]
    }

    #[inline]
    fn with_value(value: f64, lanes: f64x4) -> Self {
        let [_, dx, dy, dz] = lanes.to_array();
        Self(f64x4::from([value, dx, dy, dz]))
    }

    #[inline]
    fn scaled_with_value(self, value: f64, derivative_scale: f64) -> Self {
        Self::with_value(value, self.0 * f64x4::splat(derivative_scale))
    }
}

impl fmt::Display for DualVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &[value, dx, dy, dz] = self.0.as_array();
        write!(f, "DualVec(v={value}, d=[{dx},{dy},{dz}])")
    }
}

impl PartialEq for DualVec {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.v() == other.v()
    }
}

impl PartialOrd for DualVec {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.v().partial_cmp(&other.v())
    }
}

impl Add for DualVec {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for DualVec {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for DualVec {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;

        let &[a_value, _, _, _] = a.as_array();
        let &[b_value, _, _, _] = b.as_array();
        let a_value_lanes = f64x4::splat(a_value);
        let b_value_lanes = f64x4::splat(b_value);

        let derivative_lanes = (a_value_lanes * b) + (b_value_lanes * a);
        Self::with_value(a_value * b_value, derivative_lanes)
    }
}

impl Div for DualVec {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;

        let &[a_value, _, _, _] = a.as_array();
        let &[b_value, _, _, _] = b.as_array();
        let value = a_value / b_value;

        let a_value_lanes = f64x4::splat(a_value);
        let b_value_lanes = f64x4::splat(b_value);

        let numerator = (a * b_value_lanes) - (b * a_value_lanes);
        let denominator = b_value_lanes * b_value_lanes;
        Self::with_value(value, numerator / denominator)
    }
}

impl Neg for DualVec {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Rem for DualVec {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        let value = self.v();
        let rhs_value = rhs.v();
        let remainder = value % rhs_value;
        let truncated_quotient = (value / rhs_value).trunc();

        let derivative_lanes = self.0 - (rhs.0 * f64x4::splat(truncated_quotient));
        Self::with_value(remainder, derivative_lanes)
    }
}

impl Zero for DualVec {
    #[inline]
    fn zero() -> Self {
        Self::constant(0.0)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.v() == 0.0
    }
}

impl One for DualVec {
    #[inline]
    fn one() -> Self {
        Self::constant(1.0)
    }
}

impl Num for DualVec {
    type FromStrRadixErr = num_traits::ParseFloatError;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(str, radix).map(Self::constant)
    }
}

impl ToPrimitive for DualVec {
    fn to_i64(&self) -> Option<i64> {
        self.v().to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.v().to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.v())
    }
}

impl NumCast for DualVec {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(Self::constant)
    }
}

impl FromPrimitive for DualVec {
    fn from_i64(n: i64) -> Option<Self> {
        n.to_f64().map(Self::constant)
    }
    fn from_u64(n: u64) -> Option<Self> {
        n.to_f64().map(Self::constant)
    }
    fn from_f64(n: f64) -> Option<Self> {
        Some(Self::constant(n))
    }
}

impl Float for DualVec {
    #[inline]
    fn nan() -> Self {
        Self::constant(f64::NAN)
    }
    #[inline]
    fn infinity() -> Self {
        Self::constant(f64::INFINITY)
    }
    #[inline]
    fn neg_infinity() -> Self {
        Self::constant(f64::NEG_INFINITY)
    }
    #[inline]
    fn neg_zero() -> Self {
        Self::constant(-0.0)
    }
    #[inline]
    fn min_positive_value() -> Self {
        Self::constant(f64::MIN_POSITIVE)
    }
    #[inline]
    fn epsilon() -> Self {
        Self::constant(f64::EPSILON)
    }
    #[inline]
    fn min_value() -> Self {
        Self::constant(f64::MIN)
    }
    #[inline]
    fn max_value() -> Self {
        Self::constant(f64::MAX)
    }
    #[inline]
    fn is_nan(self) -> bool {
        self.v().is_nan()
    }
    #[inline]
    fn is_infinite(self) -> bool {
        self.v().is_infinite()
    }
    #[inline]
    fn is_finite(self) -> bool {
        self.v().is_finite()
    }
    #[inline]
    fn is_normal(self) -> bool {
        self.v().is_normal()
    }
    #[inline]
    fn classify(self) -> std::num::FpCategory {
        self.v().classify()
    }
    #[inline]
    fn floor(self) -> Self {
        Self::constant(self.v().floor())
    }
    #[inline]
    fn ceil(self) -> Self {
        Self::constant(self.v().ceil())
    }
    #[inline]
    fn round(self) -> Self {
        Self::constant(self.v().round())
    }
    #[inline]
    fn trunc(self) -> Self {
        Self::constant(self.v().trunc())
    }
    #[inline]
    fn fract(self) -> Self {
        let mut arr = self.0.to_array();
        arr[0] = arr[0].fract();
        Self(f64x4::from(arr))
    }
    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec absolute value delegates to its f64x4 negation implementation"
    )]
    fn abs(self) -> Self {
        let v = self.0.as_array()[0];
        if v >= 0.0 {
            self
        } else {
            -self
        }
    }
    #[inline]
    fn signum(self) -> Self {
        Self::constant(self.0.as_array()[0].signum())
    }
    #[inline]
    fn is_sign_positive(self) -> bool {
        self.0.as_array()[0].is_sign_positive()
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        self.0.as_array()[0].is_sign_negative()
    }
    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec fused multiply-add preserves the existing product-then-sum derivative order"
    )]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec reciprocal delegates to its f64x4 quotient implementation"
    )]
    fn recip(self) -> Self {
        Self::one() / self
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        let value_input = self.v();
        let value = value_input.powi(n);
        let derivative_scale = <f64 as From<i32>>::from(n) * value_input.powi(n.saturating_sub(1));
        self.scaled_with_value(value, derivative_scale)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        let value_input = self.v();
        let exponent = n.v();
        let value = value_input.powf(exponent);
        let log_input = value_input.ln();

        let exponent_term = n.0 * f64x4::splat(log_input);
        let input_term = self.0 * f64x4::splat(exponent / value_input);

        Self::with_value(value, (exponent_term + input_term) * f64x4::splat(value))
    }

    #[inline]
    fn sqrt(self) -> Self {
        let value = self.v().sqrt();
        self.scaled_with_value(value, 0.5 / value)
    }

    #[inline]
    fn exp(self) -> Self {
        // Use scalar safe_exp on lane 0 only — DualVec only needs exp(value),
        // not exp of the derivative lanes (chain rule handles those below).
        use common_rs::safe_exp;

        let value = safe_exp(self.v());

        self.scaled_with_value(value, value)
    }

    #[inline]
    fn exp2(self) -> Self {
        let value = self.v().exp2();
        self.scaled_with_value(value, value * 2.0f64.ln())
    }

    #[inline]
    fn ln(self) -> Self {
        let value_input = self.v();
        self.scaled_with_value(value_input.ln(), 1.0 / value_input)
    }

    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec log must compose its derivative-aware natural-log implementation rather than recursively call this trait method"
    )]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec log2 composes derivative-aware natural log and scalar base conversion"
    )]
    fn log2(self) -> Self {
        self.ln() / Self::constant(2.0f64.ln())
    }

    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec log10 composes derivative-aware natural log and scalar base conversion"
    )]
    fn log10(self) -> Self {
        self.ln() / Self::constant(10.0f64.ln())
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        if self.0.as_array()[0] >= other.0.as_array()[0] {
            self
        } else {
            other
        }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        if self.0.as_array()[0] <= other.0.as_array()[0] {
            self
        } else {
            other
        }
    }

    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec absolute difference delegates to lane-wise f64x4 subtraction"
    )]
    fn abs_sub(self, other: Self) -> Self {
        if self.0.as_array()[0] <= other.0.as_array()[0] {
            Self::zero()
        } else {
            self - other
        }
    }

    #[inline]
    fn cbrt(self) -> Self {
        let value = self.v().cbrt();
        self.scaled_with_value(value, 1.0 / (3.0 * value * value))
    }

    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec hypot composes its derivative-aware multiply and add implementations"
    )]
    fn hypot(self, other: Self) -> Self {
        (self * self + other * other).sqrt()
    }

    #[inline]
    fn sin(self) -> Self {
        let value_input = self.v();
        self.scaled_with_value(value_input.sin(), value_input.cos())
    }

    #[inline]
    fn cos(self) -> Self {
        let value_input = self.v();
        self.scaled_with_value(value_input.cos(), -value_input.sin())
    }

    #[inline]
    fn tan(self) -> Self {
        let value = self.v().tan();
        self.scaled_with_value(value, 1.0 + value * value)
    }

    /// # Domain clamp
    ///
    /// The argument is clamped to [-1, 1] exactly as the f64 twins do
    /// (`lib.rs:893`, `rhs.rs:3678`). A direction cosine that roundoff pushes
    /// to 1 + 1e-16 is a rounding artifact, not a domain error, but `asin`
    /// answers NaN for it AND the derivative `1/sqrt(1 - x*x)` answers NaN --
    /// so the Jacobian silently fills with NaN and the LM/Newton step that
    /// consumes it produces no diagnosis, only a step that goes nowhere.
    ///
    /// The clamp is the identity on every in-domain input, so this moves no
    /// bits on any argument the solver can legitimately produce. At exactly
    /// |x| = 1 the derivative is +/-inf, which is the true derivative there
    /// and, unlike NaN, compares meaningfully against `is_finite`.
    #[inline]
    fn asin(self) -> Self {
        let value_input = self.v().clamp(-1.0, 1.0);
        self.scaled_with_value(
            value_input.asin(),
            1.0 / (1.0 - value_input * value_input).sqrt(),
        )
    }

    /// Clamped for the same reason as [`asin`](Self::asin); see its note.
    #[inline]
    fn acos(self) -> Self {
        let value_input = self.v().clamp(-1.0, 1.0);
        self.scaled_with_value(
            value_input.acos(),
            -1.0 / (1.0 - value_input * value_input).sqrt(),
        )
    }

    #[inline]
    fn atan(self) -> Self {
        let value_input = self.v();
        self.scaled_with_value(value_input.atan(), 1.0 / (1.0 + value_input * value_input))
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let value_input = self.v();
        let other_value = other.v();
        let value = value_input.atan2(other_value);
        let denominator = value_input * value_input + other_value * other_value;
        let derivative_lanes =
            (self.0 * f64x4::splat(other_value)) - (other.0 * f64x4::splat(value_input));
        Self::with_value(value, derivative_lanes / f64x4::splat(denominator))
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec exp_m1 composes its derivative-aware exponential and lane-wise subtraction"
    )]
    fn exp_m1(self) -> Self {
        self.exp() - Self::one()
    }

    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec ln_1p composes derivative-aware lane-wise addition and logarithm"
    )]
    fn ln_1p(self) -> Self {
        (self + Self::one()).ln()
    }

    #[inline]
    fn sinh(self) -> Self {
        let value_input = self.v();
        self.scaled_with_value(value_input.sinh(), value_input.cosh())
    }

    #[inline]
    fn cosh(self) -> Self {
        let value_input = self.v();
        self.scaled_with_value(value_input.cosh(), value_input.sinh())
    }

    #[inline]
    fn tanh(self) -> Self {
        let value = self.v().tanh();
        self.scaled_with_value(value, 1.0 - value * value)
    }

    #[inline]
    fn asinh(self) -> Self {
        let value_input = self.v();
        self.scaled_with_value(
            value_input.asinh(),
            1.0 / (value_input * value_input + 1.0).sqrt(),
        )
    }

    #[inline]
    fn acosh(self) -> Self {
        let value_input = self.v();
        self.scaled_with_value(
            value_input.acosh(),
            1.0 / (value_input * value_input - 1.0).sqrt(),
        )
    }

    #[inline]
    fn atanh(self) -> Self {
        let value_input = self.v();
        self.scaled_with_value(value_input.atanh(), 1.0 / (1.0 - value_input * value_input))
    }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.v().integer_decode()
    }
}

impl AddAssign for DualVec {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for DualVec {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for DualVec {
    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec mul-assign delegates to its checked-by-construction product-rule implementation"
    )]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for DualVec {
    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec div-assign delegates to its checked-by-construction quotient-rule implementation"
    )]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl RemAssign for DualVec {
    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "DualVec rem-assign delegates to its checked-by-construction remainder implementation"
    )]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}
