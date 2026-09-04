#![cfg(feature = "autodiff")]

use std::fmt;

use num_traits::ToPrimitive;
use satpy_core::{
    equinoc_prop_from_impl, pack_gravity_coeffs, spherical_gravity_impl_generic_packed,
    spherical_gravity_impl_packed, DualVec, GravityCache, GravityCacheGeneric, GravityError,
    PackedGravityCoeffs,
};

#[derive(Debug)]
enum GradientCheckError {
    Gravity(GravityError),
    RelativeError { component: usize, value: f64 },
}

impl From<GravityError> for GradientCheckError {
    fn from(error: GravityError) -> Self {
        Self::Gravity(error)
    }
}

impl fmt::Display for GradientCheckError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gravity(error) => write!(formatter, "gravity evaluation failed: {error}"),
            Self::RelativeError { component, value } => {
                write!(
                    formatter,
                    "gradient mismatch at component {component}: {value:e}"
                )
            }
        }
    }
}

fn create_test_coefficients(order: usize) -> Result<PackedGravityCoeffs, GravityError> {
    let stride = order
        .checked_add(1)
        .ok_or(GravityError::InvariantViolation)?;
    let total_size = stride
        .checked_mul(stride)
        .ok_or(GravityError::InvariantViolation)?;
    let mut c_coeffs = vec![0.0; total_size];
    let mut s_coeffs = vec![0.0; total_size];

    let c00 = c_coeffs
        .first_mut()
        .ok_or(GravityError::InvariantViolation)?;
    *c00 = 1.0;

    for degree in 2..=order {
        let row_start = degree
            .checked_mul(stride)
            .ok_or(GravityError::InvariantViolation)?;
        let row_end = row_start
            .checked_add(stride)
            .ok_or(GravityError::InvariantViolation)?;
        let c_row = c_coeffs
            .get_mut(row_start..row_end)
            .ok_or(GravityError::InvariantViolation)?;
        let s_row = s_coeffs
            .get_mut(row_start..row_end)
            .ok_or(GravityError::InvariantViolation)?;
        let degree_value = degree.to_f64().ok_or(GravityError::InvariantViolation)?;
        let zonal = c_row.first_mut().ok_or(GravityError::InvariantViolation)?;
        *zonal = 1e-3 / degree_value.powi(2);

        for (order_index, (cosine, sine)) in c_row
            .iter_mut()
            .zip(s_row.iter_mut())
            .enumerate()
            .skip(1)
            .take(degree)
        {
            let order_value = order_index
                .to_f64()
                .ok_or(GravityError::InvariantViolation)?;
            let magnitude = 1e-6 / (degree_value * order_value);
            *cosine = magnitude;
            *sine = magnitude * 0.5;
        }
    }

    pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order)
}

fn state_with_x_seed(state: &[f64; 6]) -> [DualVec; 6] {
    let mut result = [DualVec::constant(0.0); 6];
    for (index, (output, &value)) in result.iter_mut().zip(state).enumerate() {
        *output = if index == 0 {
            DualVec::new(value, nalgebra::Vector3::new(1.0, 0.0, 0.0))
        } else {
            DualVec::constant(value)
        };
    }
    result
}

fn finite_difference_x_gravity_gradient(
    state: &[f64; 6],
    jd: f64,
    packed: &PackedGravityCoeffs,
    h: f64,
) -> Result<[f64; 3], GravityError> {
    let [x, y, z, vx, vy, vz] = *state;
    let state_plus = [x + h, y, z, vx, vy, vz];
    let state_minus = [x - h, y, z, vx, vy, vz];
    let mut plus_cache = GravityCache::new();
    let plus = spherical_gravity_impl_packed(&state_plus, jd, &mut plus_cache, packed)?;
    let mut minus_cache = GravityCache::new();
    let minus = spherical_gravity_impl_packed(&state_minus, jd, &mut minus_cache, packed)?;

    let [plus_x, plus_y, plus_z] = plus;
    let [minus_x, minus_y, minus_z] = minus;
    Ok([
        (plus_x - minus_x) / (2.0 * h),
        (plus_y - minus_y) / (2.0 * h),
        (plus_z - minus_z) / (2.0 * h),
    ])
}

fn verify_gravity_gradient_matches_finite_difference() -> Result<(), GradientCheckError> {
    let packed = create_test_coefficients(21)?;
    let state = [6778.0, 100.0, 50.0, 0.0, 7.5, 0.1];
    let jd = 2_460_000.5;

    let state_dual = state_with_x_seed(&state);
    let mut cache_dual: GravityCacheGeneric<DualVec> = GravityCacheGeneric::new();
    let acceleration_dual =
        spherical_gravity_impl_generic_packed(&state_dual, jd, &mut cache_dual, &packed)?;
    let [dual_x, dual_y, dual_z] = acceleration_dual;
    let [gradient_x, _, _] = dual_x.d();
    let [gradient_y, _, _] = dual_y.d();
    let [gradient_z, _, _] = dual_z.d();
    let dual_gradient = [gradient_x, gradient_y, gradient_z];
    let finite_difference = finite_difference_x_gravity_gradient(&state, jd, &packed, 1e-6)?;

    for (index, (dual, finite)) in dual_gradient.iter().zip(finite_difference).enumerate() {
        let relative_error = if dual.abs() > 1e-15 {
            ((*dual - finite) / *dual).abs()
        } else {
            (*dual - finite).abs()
        };
        if relative_error >= 1e-4 {
            return Err(GradientCheckError::RelativeError {
                component: index,
                value: relative_error,
            });
        }
    }
    Ok(())
}

#[test]
fn gravity_gradient_matches_finite_difference() {
    let result = verify_gravity_gradient_matches_finite_difference();
    assert!(result.is_ok(), "gravity gradient check failed: {result:?}");
}

#[test]
fn propagation_gradient_matches_finite_difference() {
    let equ = [6777.9934, 0.001, 0.0, 0.4663, 0.0, 0.5];
    let tof = 3600.0;
    let h = 1e-8;

    let [semi_major_axis, h_component, k_component, p_component, q_component, lambda] = equ;
    let equ_dual: [DualVec; 6] = [
        DualVec::new(semi_major_axis, nalgebra::Vector3::new(1.0, 0.0, 0.0)),
        DualVec::constant(h_component),
        DualVec::constant(k_component),
        DualVec::constant(p_component),
        DualVec::constant(q_component),
        DualVec::constant(lambda),
    ];
    let mut out_dual = [DualVec::constant(0.0); 6];
    equinoc_prop_from_impl(&equ_dual, DualVec::constant(tof), &mut out_dual);

    let equ_plus = [
        semi_major_axis + h,
        h_component,
        k_component,
        p_component,
        q_component,
        lambda,
    ];
    let equ_minus = [
        semi_major_axis - h,
        h_component,
        k_component,
        p_component,
        q_component,
        lambda,
    ];
    let mut out_plus = [0.0; 6];
    let mut out_minus = [0.0; 6];
    equinoc_prop_from_impl(&equ_plus, tof, &mut out_plus);
    equinoc_prop_from_impl(&equ_minus, tof, &mut out_minus);

    for (index, ((dual_output, plus_output), minus_output)) in
        out_dual.iter().zip(out_plus).zip(out_minus).enumerate()
    {
        let [dual_gradient, _, _] = dual_output.d();
        let finite_difference = (plus_output - minus_output) / (2.0 * h);
        let relative_error = if dual_gradient.abs() > 1e-15 {
            ((dual_gradient - finite_difference) / dual_gradient).abs()
        } else {
            (dual_gradient - finite_difference).abs()
        };
        assert!(relative_error < 1e-3, "gradient mismatch at output {index}");
    }
}

#[test]
fn dualvec_primal_matches_f64_propagation() {
    let equ = [6777.9934, 0.001, 0.0, 0.4663, 0.0, 0.5];
    let tof = 3600.0;

    let mut out_f64 = [0.0; 6];
    equinoc_prop_from_impl(&equ, tof, &mut out_f64);

    let equ_dual: [DualVec; 6] = equ.map(DualVec::constant);
    let mut out_dual = [DualVec::constant(0.0); 6];
    equinoc_prop_from_impl(&equ_dual, DualVec::constant(tof), &mut out_dual);

    for (index, (scalar, dual)) in out_f64.iter().zip(out_dual).enumerate() {
        let difference = (*scalar - dual.v()).abs();
        assert!(difference < 1e-14, "value mismatch at output {index}");
    }
}

/// A direction cosine that roundoff pushes just outside [-1, 1] must not poison
/// the Jacobian.
///
/// `asin`/`acos` answer NaN outside their domain, and so does the derivative
/// `1/sqrt(1 - x*x)`. Before the clamp, a unit-vector component that came back
/// as `1.0 + 1e-16` -- an everyday rounding artifact, not a modelling error --
/// filled the whole dual number with NaN, and every LM/Newton step downstream
/// inherited it with no diagnosis. The f64 twins (`lib.rs:893`, `rhs.rs:3678`)
/// have always clamped; the autodiff siblings did not.
///
/// This also pins the half that matters more: the clamp is the IDENTITY on
/// in-domain inputs, so it moves no bits on anything the solver can
/// legitimately produce.
#[test]
fn autodiff_asin_acos_survive_roundoff_past_the_unit_domain() {
    use num_traits::Float;

    let just_over = 1.0_f64 + f64::EPSILON;
    assert!(
        just_over > 1.0,
        "the fixture must actually sit outside the domain"
    );

    for (name, overshoot) in [("+1+eps", just_over), ("-1-eps", -just_over)] {
        let probe = DualVec::new(overshoot, nalgebra::Vector3::new(1.0, 0.0, 0.0));

        let arcsine = Float::asin(probe);
        assert!(
            arcsine.v().is_finite(),
            "asin({name}) returned {} -- a NaN here propagates through the \
             entire Jacobian with no diagnosis",
            arcsine.v()
        );
        let arccosine = Float::acos(probe);
        assert!(
            arccosine.v().is_finite(),
            "acos({name}) returned {}",
            arccosine.v()
        );
    }

    // The identity half: in-domain arguments are untouched, so no bits move.
    //
    // Exact equality is the assertion, not an oversight: the claim is that the
    // clamp is the IDENTITY here, so any difference at all -- one ULP included
    // -- is the regression this guards against. A tolerance would let exactly
    // the drift it exists to catch pass.
    #[expect(
        clippy::float_cmp,
        reason = "bit-identity with the unclamped f64 result IS the property"
    )]
    for sample in [-1.0_f64, -0.5, 0.0, 0.25, 0.5, 1.0] {
        let probe = DualVec::new(sample, nalgebra::Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(
            Float::asin(probe).v(),
            sample.asin(),
            "asin({sample}) must equal the unclamped f64 result exactly"
        );
        assert_eq!(
            Float::acos(probe).v(),
            sample.acos(),
            "acos({sample}) must equal the unclamped f64 result exactly"
        );
    }
}
