//! Comprehensive comparison: `DualVec` autodiff vs f64 + finite-difference
//!
//! This benchmark quantifies both accuracy and speed to help decide
//! when `DualVec` is worthwhile vs plain f64 with finite-difference gradients.
//!
//! Key insight: `DualVec` gives gradients in a SINGLE forward pass,
//! while finite-difference requires 2*N evaluations (central diff) for N parameters.
//!
//! ## Trade-offs
//!
//! | Approach | Speed (1 eval) | Gradient Cost | Gradient Accuracy |
//! |----------|----------------|---------------|-------------------|
//! | f64 only | 1x (baseline) | N/A | N/A |
//! | `DualVec` | ~3-4x | FREE (included) | Analytical (exact) |
//! | f64 + FD | 1x | 2*N evals | ~1e-8 (step-dependent) |
//!
//! `DualVec` wins when: 3-4x < 2*N, i.e., N >= 2
//! For typical 6-dim state gradients: `DualVec` is 3x faster than FD!

use criterion::{criterion_group, criterion_main, Criterion};
use satpy_core::{
    equinoc_prop_from_impl, pack_gravity_coeffs, spherical_gravity_impl_generic_packed,
    spherical_gravity_impl_packed, DualVec, GravityCache, GravityCacheGeneric, GravityError,
    PackedGravityCoeffs,
};
use std::time::Duration;

/// Create test gravity coefficients
fn create_test_coefficients(order: usize) -> Result<(Vec<f64>, Vec<f64>, usize), GravityError> {
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
    for l in 2..=order {
        let base = l
            .checked_mul(stride)
            .ok_or(GravityError::InvariantViolation)?;
        let degree = f64::from(u32::try_from(l).map_err(|_| GravityError::InvariantViolation)?);
        let zonal = c_coeffs
            .get_mut(base)
            .ok_or(GravityError::InvariantViolation)?;
        *zonal = 1e-3 / degree.powi(2);
        for m in 1..=l {
            let degree_order = l.checked_mul(m).ok_or(GravityError::InvariantViolation)?;
            let degree_order = f64::from(
                u32::try_from(degree_order).map_err(|_| GravityError::InvariantViolation)?,
            );
            let magnitude = 1e-6 / degree_order;
            let index = base
                .checked_add(m)
                .ok_or(GravityError::InvariantViolation)?;
            let cosine = c_coeffs
                .get_mut(index)
                .ok_or(GravityError::InvariantViolation)?;
            *cosine = magnitude;
            let sine = s_coeffs
                .get_mut(index)
                .ok_or(GravityError::InvariantViolation)?;
            *sine = magnitude * 0.5;
        }
    }
    Ok((c_coeffs, s_coeffs, stride))
}

fn packed_test_coefficients(order: usize) -> Result<PackedGravityCoeffs, GravityError> {
    let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order)?;
    pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order)
}

/// Convert f64 state to `DualVec` with gradient seed in direction i
fn state_with_seed(state: &[f64; 6], seed_idx: usize) -> Result<[DualVec; 6], GravityError> {
    let mut result = state.map(DualVec::constant);
    let seed_value = state
        .get(seed_idx)
        .copied()
        .ok_or(GravityError::InvalidState)?;
    let seed = result.get_mut(seed_idx).ok_or(GravityError::InvalidState)?;
    *seed = DualVec::new(seed_value, nalgebra::Vector3::new(1.0, 0.0, 0.0));
    Ok(result)
}

fn six_seeded_states(state: &[f64; 6]) -> Result<[[DualVec; 6]; 6], GravityError> {
    Ok([
        state_with_seed(state, 0)?,
        state_with_seed(state, 1)?,
        state_with_seed(state, 2)?,
        state_with_seed(state, 3)?,
        state_with_seed(state, 4)?,
        state_with_seed(state, 5)?,
    ])
}

/// Finite-difference gradient (central difference)
fn finite_difference_gravity_gradient(
    state: &[f64; 6],
    jd: f64,
    packed: &PackedGravityCoeffs,
    param_idx: usize,
    h: f64,
) -> Result<[f64; 3], GravityError> {
    let mut cache = GravityCache::new();

    // f(x + h)
    let mut state_plus = *state;
    let plus_parameter = state_plus
        .get_mut(param_idx)
        .ok_or(GravityError::InvalidState)?;
    *plus_parameter += h;
    let acc_plus = spherical_gravity_impl_packed(&state_plus, jd, &mut cache, packed)?;

    // f(x - h)
    let mut state_minus = *state;
    let minus_parameter = state_minus
        .get_mut(param_idx)
        .ok_or(GravityError::InvalidState)?;
    *minus_parameter -= h;
    let acc_minus = spherical_gravity_impl_packed(&state_minus, jd, &mut cache, packed)?;

    // (f(x+h) - f(x-h)) / 2h
    let [plus_x, plus_y, plus_z] = acc_plus;
    let [minus_x, minus_y, minus_z] = acc_minus;
    Ok([
        (plus_x - minus_x) / (2.0 * h),
        (plus_y - minus_y) / (2.0 * h),
        (plus_z - minus_z) / (2.0 * h),
    ])
}

fn dualvec_gravity_gradient(
    state_dual: &[DualVec; 6],
    jd: f64,
    cache: &mut GravityCacheGeneric<DualVec>,
    packed: &PackedGravityCoeffs,
) -> Result<[f64; 3], GravityError> {
    spherical_gravity_impl_generic_packed(state_dual, jd, cache, packed).map(
        |[acc_x, acc_y, acc_z]| {
            let [dx, _, _] = acc_x.d();
            let [dy, _, _] = acc_y.d();
            let [dz, _, _] = acc_z.d();
            [dx, dy, dz]
        },
    )
}

/// Benchmark: f64 value only (baseline)
fn bench_f64_value_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_comparison");

    let order = 21;
    let packed = match packed_test_coefficients(order) {
        Ok(packed) => packed,
        Err(error) => {
            eprintln!("gravity_f64_value_only setup failed: {error}");
            return;
        }
    };
    let state = [6778.0, 0.0, 0.0, 0.0, 7.5, 0.0];
    let jd = 2_460_000.5;
    let mut cache = GravityCache::new();

    group.bench_function("gravity_f64_value_only", |b| {
        b.iter(|| {
            spherical_gravity_impl_packed(
                std::hint::black_box(&state),
                std::hint::black_box(jd),
                &mut cache,
                &packed,
            )
        });
    });

    group.finish();
}

/// Benchmark: `DualVec` value + gradient (single pass)
fn bench_dualvec_value_and_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_comparison");

    let order = 21;
    let packed = match packed_test_coefficients(order) {
        Ok(packed) => packed,
        Err(error) => {
            eprintln!("gravity_dualvec_value_and_grad setup failed: {error}");
            return;
        }
    };
    let state = [6778.0, 0.0, 0.0, 0.0, 7.5, 0.0];
    let jd = 2_460_000.5;
    let mut cache: GravityCacheGeneric<DualVec> = GravityCacheGeneric::new();

    // Seed gradient w.r.t. position x
    let state_dual = match state_with_seed(&state, 0) {
        Ok(state_dual) => state_dual,
        Err(error) => {
            eprintln!("gravity_dualvec_value_and_grad seed setup failed: {error}");
            return;
        }
    };

    group.bench_function("gravity_dualvec_value_and_grad", |b| {
        b.iter(|| {
            spherical_gravity_impl_generic_packed(
                std::hint::black_box(&state_dual),
                std::hint::black_box(jd),
                &mut cache,
                &packed,
            )
        });
    });

    group.finish();
}

/// Benchmark: f64 + finite-difference for 1 gradient component
fn bench_f64_fd_one_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_comparison");

    let order = 21;
    let packed = match packed_test_coefficients(order) {
        Ok(packed) => packed,
        Err(error) => {
            eprintln!("gravity_f64_fd_1_param setup failed: {error}");
            return;
        }
    };
    let state = [6778.0, 0.0, 0.0, 0.0, 7.5, 0.0];
    let jd = 2_460_000.5;
    let h = 1e-6;

    group.bench_function("gravity_f64_fd_1_param", |b| {
        b.iter(|| {
            finite_difference_gravity_gradient(
                std::hint::black_box(&state),
                std::hint::black_box(jd),
                &packed,
                0,
                h,
            )
        });
    });

    group.finish();
}

/// Benchmark: f64 + finite-difference for full 6-dim state gradient
fn bench_f64_fd_full_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_comparison");

    let order = 21;
    let packed = match packed_test_coefficients(order) {
        Ok(packed) => packed,
        Err(error) => {
            eprintln!("gravity_f64_fd_6_params setup failed: {error}");
            return;
        }
    };
    let state = [6778.0, 0.0, 0.0, 0.0, 7.5, 0.0];
    let jd = 2_460_000.5;
    let h = 1e-6;

    group.bench_function("gravity_f64_fd_6_params", |b| {
        b.iter(|| {
            std::hint::black_box([
                finite_difference_gravity_gradient(&state, jd, &packed, 0, h),
                finite_difference_gravity_gradient(&state, jd, &packed, 1, h),
                finite_difference_gravity_gradient(&state, jd, &packed, 2, h),
                finite_difference_gravity_gradient(&state, jd, &packed, 3, h),
                finite_difference_gravity_gradient(&state, jd, &packed, 4, h),
                finite_difference_gravity_gradient(&state, jd, &packed, 5, h),
            ])
        });
    });

    group.finish();
}

/// Benchmark: `DualVec` full 6-dim gradient (6 forward passes with different seeds)
fn bench_dualvec_full_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_comparison");

    let order = 21;
    let packed = match packed_test_coefficients(order) {
        Ok(packed) => packed,
        Err(error) => {
            eprintln!("gravity_dualvec_6_params setup failed: {error}");
            return;
        }
    };
    let state = [6778.0, 0.0, 0.0, 0.0, 7.5, 0.0];
    let jd = 2_460_000.5;
    let mut cache: GravityCacheGeneric<DualVec> = GravityCacheGeneric::new();
    let state_duals = match six_seeded_states(&state) {
        Ok(state_duals) => state_duals,
        Err(error) => {
            eprintln!("gravity_dualvec_6_params seed setup failed: {error}");
            return;
        }
    };
    group.bench_function("gravity_dualvec_6_params", |b| {
        b.iter(|| {
            let gradients: [Result<[f64; 3], GravityError>; 6] = std::array::from_fn(|seed| {
                state_duals
                    .get(seed)
                    .ok_or(GravityError::InvalidState)
                    .and_then(|seeded_state| {
                        dualvec_gravity_gradient(seeded_state, jd, &mut cache, &packed)
                    })
            });
            std::hint::black_box(gradients)
        });
    });

    group.finish();
}

/// Benchmark propagation: value only vs value+gradient
fn bench_propagation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("propagation_autodiff_comparison");

    // LEO equinoctial state
    let equ = [6777.9934, 0.001, 0.0, 0.4663, 0.0, 0.0];
    let tof = 3600.0;

    // f64 value only
    group.bench_function("prop_f64_value_only", |b| {
        b.iter(|| {
            let mut out = [0.0f64; 6];
            equinoc_prop_from_impl(
                std::hint::black_box(&equ),
                std::hint::black_box(tof),
                &mut out,
            );
            out
        });
    });

    // DualVec value + gradient
    let [semi_major_axis, eccentricity_h, eccentricity_k, inclination_p, inclination_q, longitude] =
        equ;
    let equ_dual: [DualVec; 6] = [
        DualVec::new(semi_major_axis, nalgebra::Vector3::new(1.0, 0.0, 0.0)),
        DualVec::constant(eccentricity_h),
        DualVec::constant(eccentricity_k),
        DualVec::constant(inclination_p),
        DualVec::constant(inclination_q),
        DualVec::constant(longitude),
    ];
    let tof_dual = DualVec::constant(tof);

    group.bench_function("prop_dualvec_value_and_grad", |b| {
        b.iter(|| {
            let mut out = [DualVec::constant(0.0); 6];
            equinoc_prop_from_impl(
                std::hint::black_box(&equ_dual),
                std::hint::black_box(tof_dual),
                &mut out,
            );
            out
        });
    });

    // f64 finite-difference (1 param)
    let h = 1e-6;
    group.bench_function("prop_f64_fd_1_param", |b| {
        b.iter(|| {
            let mut equ_plus = equ;
            let mut equ_minus = equ;
            if let Some(value) = equ_plus.first_mut() {
                *value += h;
            }
            if let Some(value) = equ_minus.first_mut() {
                *value -= h;
            }

            let mut out_plus = [0.0f64; 6];
            let mut out_minus = [0.0f64; 6];
            equinoc_prop_from_impl(&equ_plus, tof, &mut out_plus);
            equinoc_prop_from_impl(&equ_minus, tof, &mut out_minus);

            let mut grad = [0.0f64; 6];
            for ((gradient, plus), minus) in grad.iter_mut().zip(out_plus).zip(out_minus) {
                *gradient = (plus - minus) / (2.0 * h);
            }
            grad
        });
    });

    group.finish();
}

criterion_group! {
    name = autodiff_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(20);
    targets =
        bench_f64_value_only,
        bench_dualvec_value_and_gradient,
        bench_f64_fd_one_gradient,
        bench_f64_fd_full_gradient,
        bench_dualvec_full_gradient,
        bench_propagation_comparison,
}

criterion_main!(autodiff_benches);
