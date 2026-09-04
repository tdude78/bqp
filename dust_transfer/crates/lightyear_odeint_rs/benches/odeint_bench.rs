//! Benchmarks for Lightyear ODE integrator hot paths.
//!
//! These benchmarks exercise the core numerical integration code.
//! Run with: `cargo bench --package lightyear_odeint_rs`
//! The `gravity_d5_packed` group (and unfiltered runs) additionally require
//! `PART_A_4BG_BENCH_COEFF_PATH` plus `PART_A_4BG_BENCH_COEFF_SHA256`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;

// Import from satpy_core for the gravity computation (the hot path)
use lightyear_odeint_rs::{
    batch::{integrate_batch_native, BatchBallistics, BatchPropagationRequest},
    config::{GlobalCoeffs, GLOBAL_COEFFS},
    integrator,
    rhs::LightyearRHS,
    types::{BodyInvariants, ForceConfig, ForceFlags},
    StepperMethod,
};
use satpy_core::{
    ecef2eci_impl, eci2ecef_impl, eci2equinoc_impl, equinoc2eci_impl, pack_gravity_coeffs,
    spherical_gravity_impl, spherical_gravity_impl_sincos_packed, GravityCache, GravityError,
    PackedGravityCoeffs,
};

#[cold]
fn benchmark_failure(context: &str) -> ! {
    // Bench-only assertion: setup failures must invalidate a run rather than
    // be hidden by an exit path. This function is never on a timed success
    // path.
    loop {
        assert!(
            std::hint::black_box(false),
            "odeint benchmark setup failed: {context}"
        );
    }
}

#[inline]
fn benchmark_result<T, E: std::fmt::Display>(result: Result<T, E>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => benchmark_failure(&format!("{context}: {error}")),
    }
}

#[inline]
fn benchmark_option<T>(option: Option<T>, context: &str) -> T {
    option.unwrap_or_else(|| benchmark_failure(context))
}

#[inline]
fn benchmark_slot<'a>(values: &'a mut [f64], index: usize, context: &str) -> &'a mut f64 {
    benchmark_option(values.get_mut(index), context)
}

#[inline]
fn usize_to_f64(value: usize, context: &str) -> f64 {
    f64::from(benchmark_result(u32::try_from(value), context))
}

#[inline]
fn cyclic_time(times: &[f64], time_count: u64, iteration: u64, context: &str) -> f64 {
    let offset = benchmark_option(iteration.checked_rem(time_count), context);
    let index = benchmark_result(usize::try_from(offset), context);
    *benchmark_option(times.get(index), context)
}

/// Create test spherical harmonics coefficients (normalized, similar to EGM96)
fn create_test_coefficients(order: usize) -> (Vec<f64>, Vec<f64>, usize) {
    let stride = benchmark_option(order.checked_add(2), "coefficient stride overflow");
    let total_size = benchmark_option(stride.checked_mul(stride), "coefficient matrix overflow");

    let mut c_coeffs = vec![0.0; total_size];
    let mut s_coeffs = vec![0.0; total_size];

    // C[0,0] = 1.0 is the central body term
    *benchmark_slot(&mut c_coeffs, 0, "central coefficient missing") = 1.0;

    // Add realistic-ish higher order terms (magnitude decreasing with degree)
    for l in 2..=order {
        let base = benchmark_option(l.checked_mul(stride), "coefficient row offset overflow");
        // J_l term (zonal)
        let degree = usize_to_f64(l, "coefficient degree exceeds f64-safe u32 range");
        *benchmark_slot(&mut c_coeffs, base, "zonal coefficient slot missing") =
            1e-3 / degree.powi(2);

        // Tesseral/sectoral terms
        for m in 1..=l {
            let degree_order =
                benchmark_option(l.checked_mul(m), "coefficient degree/order overflow");
            let magnitude = 1e-6
                / usize_to_f64(
                    degree_order,
                    "coefficient degree/order exceeds f64-safe u32 range",
                );
            let index = benchmark_option(base.checked_add(m), "coefficient slot offset overflow");
            *benchmark_slot(&mut c_coeffs, index, "cosine coefficient slot missing") = magnitude;
            *benchmark_slot(&mut s_coeffs, index, "sine coefficient slot missing") =
                magnitude * 0.5;
        }
    }

    (c_coeffs, s_coeffs, stride)
}

fn install_global_coeffs(c_coeffs: &[f64], s_coeffs: &[f64], stride: usize, order: usize) {
    let coefficients = Arc::new(benchmark_result(
        pack_gravity_coeffs(c_coeffs, s_coeffs, stride, order),
        "benchmark global gravity coefficient packing",
    ));
    GLOBAL_COEFFS.store(Arc::new(GlobalCoeffs::Loaded(coefficients)));
}

fn create_prod_force_config() -> ForceConfig {
    let sun_pos = [1.495_978_707e8, 1.0e4, -2.0e4];
    let moon_pos = [384_400.0, 2.0e3, -5.0e3];
    ForceConfig {
        sph_order: 5,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        subtract_first_order: true,
        atm_model: 3,
        am_ratio: 0.01,
        cd: 2.2,
        cr: 1.3,
        target_propagation_mode: 0,
        qm_ratio: 0.0,
        r_obj_m: 0.0,
        omega_earth: 7.292_115_0e-5,
        p_sun: 4.56e-6,
        mu_sun: 1.327_124_400_18e11,
        mu_moon: 4_902.800_066,
        mu_jupiter: 1.266_865_34e8,
        mu_venus: 3.248_585_92e5,
        mu_mars: 4.282_837_5e4,
        mu_saturn: 3.793_120_6e7,
        earth_radius: 6378.137,
        sun_pos: Some(sun_pos),
        moon_pos: Some(moon_pos),
        jupiter_pos: None,
        venus_pos: None,
        mars_pos: None,
        saturn_pos: None,
        dynamic_ephemeris_flags: 0,
        sun_invariants: BodyInvariants::precompute(&sun_pos, 1.327_124_400_18e11),
        moon_invariants: BodyInvariants::precompute(&moon_pos, 4_902.800_066),
        jupiter_invariants: None,
        venus_invariants: None,
        mars_invariants: None,
        saturn_invariants: None,
        dt_max: 60.0,
        eps: 1e-8,
        integrator_method: lightyear_odeint_rs::types::StepperMethod::Dopri5Compat,
    }
}

/// Create a LEO state in ECI coordinates
const fn create_leo_state() -> [f64; 6] {
    // LEO circular orbit: r=6778 km (400 km altitude), v=7.67 km/s
    [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0]
}

/// Convert ECI state to equinoctial elements
fn eci_to_equinoc(eci: [f64; 6]) -> [f64; 6] {
    let mut equinoc = [0.0; 6];
    // eci2equinoc_impl(eci, len, t, t0, out)
    eci2equinoc_impl(&eci, 6, 0.0, 0.0, &mut equinoc);
    equinoc
}

/// Benchmark spherical gravity computation (the hot path in RHS evaluation)
fn bench_spherical_gravity(c: &mut Criterion) {
    let mut group = c.benchmark_group("spherical_gravity");

    let state = create_leo_state();
    let jd = 2_460_000.5;

    for &order in &[4, 8, 16, 21] {
        let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order);
        let mut cache = GravityCache::new();

        group.bench_with_input(BenchmarkId::new("order", order), &order, |b, &order| {
            b.iter(|| {
                black_box(benchmark_result(
                    spherical_gravity_impl(
                        std::hint::black_box(&state),
                        std::hint::black_box(jd),
                        std::hint::black_box(order),
                        &c_coeffs,
                        &s_coeffs,
                        stride,
                        &mut cache,
                    ),
                    "spherical gravity evaluation",
                ))
            });
        });
    }

    group.finish();
}

/// An eccentric LEO state at perigee, `a = 7000 km`, `e = 0.01`.
///
/// `create_leo_state` is circular to 3.6e-4, which converges the Kepler loop in
/// about two passes. Production orbits do not, so a conversion benchmarked on a
/// circular state is not measuring the production cost.
fn create_eccentric_leo_state() -> [f64; 6] {
    // Read, not restated. The local copy this replaces read `398_600.441_8`,
    // which is 3e-4 km^3/s^2 away from the `398600.4415` every other site in
    // the tree uses -- a transcribed constant that had already drifted.
    use lightyear_odeint_rs::types::MU;
    let (a, e) = (7000.0_f64, 0.01_f64);
    let r_p = a * (1.0 - e);
    let v_p = (MU * (1.0 + e) / r_p).sqrt();
    [r_p, 0.0, 0.0, 0.0, v_p, 0.0]
}

/// Benchmark equinoctial to ECI conversion (hot path in delta-state computation)
///
/// TWO ARMS ON PURPOSE, because the first one used to be the only one and it
/// was quoted as the cost of the RHS's call. It is not, and the gap was large
/// enough (4.6x against the in-RHS figure) to start an investigation.
///
///   `equinoc2eci_zero_tof_circular`  the historical arm, renamed for what it
///       actually measures: `t = t0 = 0.0`, so `lam + n*(t - t0)` advances by
///       NOTHING, on a state that is circular to 3.6e-4. It is the cheapest
///       call the function admits and it is not a regime the RHS ever enters.
///       Kept, rather than deleted, so the historical number stays reproducible
///       and legible instead of merely disappearing.
///
///   `equinoc2eci_rhs_regime`  what `rhs.rs` actually calls: a nonzero `tof`
///       sweeping a full orbit, on an eccentric state. `tof` varies per call,
///       so the Kepler iteration count varies too and the branch predictor is
///       not handed a constant — which is also the RHS's situation.
fn bench_equinoc_to_eci(c: &mut Criterion) {
    let equinoc = eci_to_equinoc(create_leo_state());

    c.bench_function("equinoc2eci_zero_tof_circular", |b| {
        b.iter(|| {
            let mut result = [0.0; 6];
            // equinoc2eci_impl(elems, len, t, t0, out)
            equinoc2eci_impl(std::hint::black_box(&equinoc), 6, 0.0, 0.0, &mut result);
            result
        });
    });

    let ecc_equinoc = eci_to_equinoc(create_eccentric_leo_state());
    // One orbit of `a = 7000 km` is ~5829 s; 256 samples across it, so the
    // reported figure is a mean over the anomaly rather than one lucky point.
    let tofs: Vec<f64> = (0..256).map(|i| f64::from(i) * (5829.0 / 256.0)).collect();

    c.bench_function("equinoc2eci_rhs_regime", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &tof in &tofs {
                let mut result = [0.0; 6];
                equinoc2eci_impl(
                    std::hint::black_box(&ecc_equinoc),
                    6,
                    std::hint::black_box(tof),
                    0.0,
                    &mut result,
                );
                let [first, ..] = result;
                acc += first;
            }
            acc
        });
    });
}

/// Benchmark ECI to equinoctial conversion
fn bench_eci_to_equinoc(c: &mut Criterion) {
    let eci = create_leo_state();

    c.bench_function("eci2equinoc_single", |b| {
        b.iter(|| {
            let mut result = [0.0; 6];
            eci2equinoc_impl(
                std::hint::black_box(&eci),
                std::hint::black_box(6),
                std::hint::black_box(0.0),
                std::hint::black_box(0.0),
                &mut result,
            );
            result
        });
    });
}

/// Benchmark batch gravity computation (simulating sigma point propagation)
fn bench_gravity_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("gravity_batch");

    let order = 21;
    let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order);
    let jd = 2_460_000.5;

    // Generate varied states (like sigma points) around LEO
    let states: Vec<[f64; 6]> = (0..100)
        .map(|i| {
            let offset = f64::from(i) * 0.001;
            [
                6778.0 + offset,
                offset,
                offset,
                offset * 0.001,
                7.67 + offset * 0.0001,
                offset * 0.0001,
            ]
        })
        .collect();

    // Benchmark with cache reuse (target behavior)
    group.bench_function("100_points_reuse_cache", |b| {
        let mut cache = GravityCache::new();
        b.iter(|| {
            for state in &states {
                std::hint::black_box(benchmark_result(
                    spherical_gravity_impl(
                        state, jd, order, &c_coeffs, &s_coeffs, stride, &mut cache,
                    ),
                    "batch spherical gravity evaluation",
                ));
            }
        });
    });

    group.finish();
}

/// Benchmark conversion chain (kep -> eci -> equinoc) as used in initialization
fn bench_conversion_chain(c: &mut Criterion) {
    let eci = create_leo_state();

    c.bench_function("eci_equinoc_roundtrip", |b| {
        b.iter(|| {
            // ECI -> Equinoc
            let mut equinoc = [0.0; 6];
            eci2equinoc_impl(std::hint::black_box(&eci), 6, 0.0, 0.0, &mut equinoc);

            // Equinoc -> ECI (elems, len, t, t0, out)
            let mut eci_back = [0.0; 6];
            equinoc2eci_impl(std::hint::black_box(&equinoc), 6, 0.0, 0.0, &mut eci_back);

            eci_back
        });
    });
}

/// Benchmark batch conversions (simulating UKF sigma point transformations)
fn bench_batch_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_conversions");

    // Generate batch of ECI states
    let states: Vec<[f64; 6]> = (0..100)
        .map(|i| {
            let offset = f64::from(i) * 0.01;
            [
                6778.0 + offset,
                offset * 10.0,
                offset * 5.0,
                offset * 0.001,
                7.67 + offset * 0.001,
                offset * 0.0005,
            ]
        })
        .collect();

    group.bench_function("100_eci2equinoc", |b| {
        b.iter(|| {
            let mut results = vec![[0.0; 6]; states.len()];
            for (state, result) in states.iter().zip(&mut results) {
                eci2equinoc_impl(state, 6, 0.0, 0.0, result);
            }
            results
        });
    });

    group.finish();
}

/// Benchmark ECI to ECEF transform (to measure Step 6 optimization potential)
fn bench_eci_to_ecef(c: &mut Criterion) {
    let eci = create_leo_state();
    let jd = 2_460_000.5;

    c.bench_function("eci2ecef_single", |b| {
        b.iter(|| {
            let mut result = [0.0; 6];
            eci2ecef_impl(
                std::hint::black_box(&eci),
                std::hint::black_box(jd),
                &mut result,
            );
            result
        });
    });
}

/// Benchmark ECEF to ECI transform
fn bench_ecef_to_eci(c: &mut Criterion) {
    let eci = create_leo_state();
    let jd = 2_460_000.5;
    let mut ecef = [0.0; 6];
    eci2ecef_impl(&eci, jd, &mut ecef);

    c.bench_function("ecef2eci_single", |b| {
        b.iter(|| {
            let mut result = [0.0; 6];
            ecef2eci_impl(
                std::hint::black_box(&ecef),
                std::hint::black_box(jd),
                &mut result,
            );
            result
        });
    });
}

/// Benchmark ECI↔ECEF roundtrip (measures full transform cost)
fn bench_eci_ecef_roundtrip(c: &mut Criterion) {
    let eci = create_leo_state();
    let jd = 2_460_000.5;

    c.bench_function("eci_ecef_roundtrip", |b| {
        b.iter(|| {
            let mut ecef = [0.0; 6];
            eci2ecef_impl(std::hint::black_box(&eci), jd, &mut ecef);
            let mut eci_back = [0.0; 6];
            ecef2eci_impl(&ecef, jd, &mut eci_back);
            eci_back
        });
    });
}

/// Benchmark full Lightyear HF integration over a short window (dominant cost profile).
fn bench_integrator_hf(c: &mut Criterion) {
    let mut group = c.benchmark_group("integrator_hf");

    let order = 21;
    let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order);
    let packed = benchmark_result(
        pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order),
        "integrator HF coefficient packing",
    );

    let config = Arc::new(ForceConfig {
        sph_order: order,
        force_flags: ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY,
        subtract_first_order: false,
        atm_model: 1, // exponential model (fast)
        am_ratio: 0.01,
        cd: 2.2,
        cr: 1.3,
        target_propagation_mode: 0,
        qm_ratio: 0.0,
        r_obj_m: 0.0,
        omega_earth: 7.292_115_0e-5,
        p_sun: 4.56e-6,
        mu_sun: 1.327_124_400_18e11,
        mu_moon: 4_902.800_066,
        mu_jupiter: 1.266_865_34e8,
        mu_venus: 3.248_585_92e5,
        mu_mars: 4.282_837_5e4,
        mu_saturn: 3.793_120_6e7,
        earth_radius: 6378.137,
        sun_pos: Some([1.495_978_707e8, 0.0, 0.0]),
        moon_pos: None,
        jupiter_pos: None,
        venus_pos: None,
        mars_pos: None,
        saturn_pos: None,
        dynamic_ephemeris_flags: 0,
        sun_invariants: BodyInvariants::precompute(
            &[1.495_978_707e8, 0.0, 0.0],
            1.327_124_400_18e11,
        ),
        moon_invariants: None,
        jupiter_invariants: None,
        venus_invariants: None,
        mars_invariants: None,
        saturn_invariants: None,
        dt_max: 60.0,
        eps: 1e-10,
        integrator_method: lightyear_odeint_rs::types::StepperMethod::Dopri5Compat,
    });

    let mut init_equ = [0.0; 6];
    let init_eci = create_leo_state();
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let jd0 = 2_460_000.5;
    let tf_s = 600.0;
    let t_eval = [tf_s];

    let packed = Arc::new(packed);
    let gravity = integrator::ScalarGravityAssets::new(Arc::clone(&packed));
    let context = integrator::ScalarPropagationContext::new(jd0, Arc::clone(&config), gravity);

    group.bench_function("integrate_600s", |b| {
        b.iter(|| {
            black_box(benchmark_result(
                integrator::integrate_final_checked(
                    integrator::ScalarPropagationRequest::new(
                        &context, init_equ, &t_eval, 0.0, tf_s,
                    )
                    .with_events(false),
                ),
                "integrate_600s propagation",
            ));
        });
    });

    group.finish();
}

/// Benchmark solver variants for HF propagation (same config, different stepper).
fn bench_solver_compare_hf(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_compare_hf");

    let order = 21;
    let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order);
    let packed = benchmark_result(
        pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order),
        "solver comparison coefficient packing",
    );

    let config = Arc::new(ForceConfig {
        sph_order: order,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        atm_model: 1,
        am_ratio: 0.01,
        cd: 2.2,
        cr: 1.3,
        sun_pos: Some([1.495_978_707e8, 0.0, 0.0]),
        moon_pos: Some([384_400.0, 0.0, 0.0]),
        sun_invariants: BodyInvariants::precompute(
            &[1.495_978_707e8, 0.0, 0.0],
            1.327_124_400_18e11,
        ),
        moon_invariants: BodyInvariants::precompute(&[384_400.0, 0.0, 0.0], 4_902.800_066),
        dt_max: 60.0,
        eps: 1e-10,
        integrator_method: lightyear_odeint_rs::types::StepperMethod::Dopri5Compat,
        ..ForceConfig::default()
    });

    let mut init_equ = [0.0; 6];
    let init_eci = create_leo_state();
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let jd0 = 2_460_000.5;
    let tf_s = 600.0;
    let t_eval = [tf_s];

    let packed = Arc::new(packed);
    let gravity = integrator::ScalarGravityAssets::new(Arc::clone(&packed));

    let solvers = [
        StepperMethod::Dopri5Compat,
        StepperMethod::Tsit5,
        StepperMethod::Dop853,
        StepperMethod::Rkv98,
    ];

    for solver in solvers {
        let mut solver_config = *config;
        solver_config.integrator_method = solver;
        let solver_config = Arc::new(solver_config);
        let context = integrator::ScalarPropagationContext::new(
            jd0,
            Arc::clone(&solver_config),
            gravity.clone(),
        );
        group.bench_with_input(
            BenchmarkId::new("solver", format!("{solver:?}")),
            &solver,
            |b, &_solver| {
                b.iter(|| {
                    black_box(benchmark_result(
                        integrator::integrate_final_checked(
                            integrator::ScalarPropagationRequest::new(
                                &context, init_equ, &t_eval, 0.0, tf_s,
                            )
                            .with_events(false),
                        ),
                        "solver propagation",
                    ));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RHS `compute_internal` variants using production-like HF settings.
fn bench_rhs_compute_internal_prod(c: &mut Criterion) {
    let mut group = c.benchmark_group("rhs_compute_internal_prod");
    let order = 5;
    let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order);
    let packed = Arc::new(benchmark_result(
        pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order),
        "production RHS coefficient packing",
    ));
    let config = Arc::new(create_prod_force_config());

    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let rhs = benchmark_result(
        LightyearRHS::try_new(init_equ, 0.0, 2_460_000.5, config, packed),
        "production RHS construction",
    );

    let delta = [1.0e-3, -2.0e-3, 7.0e-4, 2.0e-6, -4.0e-6, 1.0e-6];
    let t = 123.456_f64;

    group.bench_function("generic", |b| {
        b.iter(|| {
            black_box(benchmark_result(
                rhs.compute_internal_generic(std::hint::black_box(&delta), std::hint::black_box(t)),
                "production generic RHS evaluation",
            ))
        });
    });
    group.bench_function("dispatch", |b| {
        b.iter(|| {
            black_box(benchmark_result(
                rhs.compute_internal(std::hint::black_box(&delta), std::hint::black_box(t)),
                "production dispatch RHS evaluation",
            ))
        });
    });
    group.finish();
}

/// Benchmark RHS dispatch over a changing sequence, preserving cache/subcycle state.
///
/// The single-call RHS bench above is intentionally cache-hot and useful for
/// dispatch overhead, but the HF sigma solver calls RHS with evolving states
/// and times. This sequence bench keeps construction out of the timing while
/// avoiding a misleading same-argument cache hit on every iteration.
fn bench_rhs_compute_internal_prod_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("rhs_compute_internal_prod_sequence");
    let order = 5;
    let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order);
    let packed = Arc::new(benchmark_result(
        pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order),
        "production RHS sequence coefficient packing",
    ));
    let config = Arc::new(create_prod_force_config());

    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let make_rhs = || {
        benchmark_result(
            LightyearRHS::try_new(
                init_equ,
                0.0,
                2_460_000.5,
                Arc::clone(&config),
                Arc::clone(&packed),
            ),
            "production RHS sequence construction",
        )
    };

    let samples: Vec<([f64; 6], f64)> = (0..4096)
        .map(|i| {
            let x = f64::from(i);
            (
                [
                    (x * 0.031).sin() * 2e-3,
                    (x * 0.017).cos() * 2e-3,
                    (x * 0.013).sin() * 3e-4,
                    (x * 0.041).cos() * 2e-6,
                    (x * 0.053).sin() * 2e-6,
                    (x * 0.067).cos() * 2e-6,
                ],
                x * 0.37,
            )
        })
        .collect();

    group.bench_function("generic_sequence_4096", |b| {
        let rhs = make_rhs();
        b.iter(|| {
            let mut acc = [0.0; 6];
            for (delta, t) in &samples {
                let out = benchmark_result(
                    rhs.compute_internal_generic(
                        std::hint::black_box(delta),
                        std::hint::black_box(*t),
                    ),
                    "production generic RHS sequence evaluation",
                );
                for (acc_axis, out_axis) in acc.iter_mut().zip(out) {
                    *acc_axis += out_axis;
                }
            }
            std::hint::black_box(acc);
        });
    });
    group.bench_function("dispatch_sequence_4096", |b| {
        let rhs = make_rhs();
        b.iter(|| {
            let mut acc = [0.0; 6];
            for (delta, t) in &samples {
                let out = benchmark_result(
                    rhs.compute_internal(std::hint::black_box(delta), std::hint::black_box(*t)),
                    "production dispatch RHS sequence evaluation",
                );
                for (acc_axis, out_axis) in acc.iter_mut().zip(out) {
                    *acc_axis += out_axis;
                }
            }
            std::hint::black_box(acc);
        });
    });
    group.finish();
}

/// Benchmark native sigma batch final-only propagation for production-like HF settings.
fn bench_sigma_batch_final_only_prod(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigma_batch_final_only_prod");
    let order = 5;
    let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order);
    install_global_coeffs(&c_coeffs, &s_coeffs, stride, order);

    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let n_sigma: usize = 15;
    let state_capacity = benchmark_option(n_sigma.checked_mul(6), "sigma state capacity overflow");
    let mut init_states_flat = Vec::with_capacity(state_capacity);
    let n_sigma_f64 = usize_to_f64(n_sigma, "sigma count exceeds f64-safe u32 range");
    let half_sigma = n_sigma_f64 * 0.5;
    for i in 0..n_sigma {
        let centered = usize_to_f64(i, "sigma index exceeds f64-safe u32 range") - half_sigma;
        let scale = 1e-6 * centered;
        for (j, &value) in init_equ.iter().enumerate() {
            let axis = usize_to_f64(j, "sigma axis exceeds f64-safe u32 range");
            init_states_flat.push(value + scale * (axis + 1.0));
        }
    }

    for tf_s in [300.0_f64, 600.0_f64, 1800.0_f64] {
        group.bench_with_input(BenchmarkId::new("tf_s", tf_s), &tf_s, |b, &tf| {
            let t_eval = [tf];
            b.iter(|| {
                let mut config = create_prod_force_config();
                config.eps = 1e-8;
                let out = benchmark_result(
                    integrate_batch_native(BatchPropagationRequest {
                        initial_equinoc_states: std::hint::black_box(&init_states_flat),
                        t_eval: std::hint::black_box(&t_eval),
                        t0_s: 0.0,
                        t_final_s: tf,
                        epoch_jd: 2_460_000.5,
                        force_config: config,
                        ballistics: BatchBallistics {
                            am_ratio: None,
                            cd: None,
                            cr: None,
                        },
                    }),
                    "native batch propagation",
                );
                std::hint::black_box(out);
            });
        });
    }

    group.finish();
}

/// Benchmark the construction slice that shows up in HF variable-final sigma timers.
///
/// The production HF macro spends most sigma time integrating rows, but current
/// stage timers still attribute a few percent of optimize wall to constructing
/// `ReusableFinalNoEventIntegrator`/`LightyearRHS` per row. This bench keeps the
/// surface isolated so candidate scratch changes can be rejected before macro A/B
/// if they do not move the constructor or constructor+propagate path.
fn bench_reusable_final_integrator_prod(c: &mut Criterion) {
    let mut group = c.benchmark_group("reusable_final_integrator_prod");
    let order = 5;
    let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order);
    let packed = Arc::new(benchmark_result(
        pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order),
        "reusable final integrator coefficient packing",
    ));
    let config = Arc::new(create_prod_force_config());

    let init_eci = create_leo_state();
    let init_equ = eci_to_equinoc(init_eci);
    let jd0 = 2_460_000.5;
    let tf_s = 300.0;
    group.bench_function("construct_only", |b| {
        b.iter(|| {
            let gravity = integrator::ScalarGravityAssets::new(Arc::clone(&packed));
            let context = integrator::ScalarPropagationContext::new(
                std::hint::black_box(jd0),
                Arc::clone(&config),
                gravity,
            );
            std::hint::black_box(benchmark_result(
                integrator::ReusableFinalNoEventIntegrator::new(context),
                "reusable integrator construction",
            ));
        });
    });

    group.bench_function("construct_and_propagate_300s", |b| {
        b.iter(|| {
            let gravity = integrator::ScalarGravityAssets::new(Arc::clone(&packed));
            let context = integrator::ScalarPropagationContext::new(
                std::hint::black_box(jd0),
                Arc::clone(&config),
                gravity,
            );
            let mut reusable = benchmark_result(
                integrator::ReusableFinalNoEventIntegrator::new(context),
                "reusable integrator construction",
            );
            std::hint::black_box(benchmark_result(
                reusable.propagate(
                    std::hint::black_box(init_equ),
                    0.0,
                    std::hint::black_box(tf_s),
                ),
                "reusable integrator propagation",
            ));
        });
    });

    group.bench_function("reuse_single_propagate_300s", |b| {
        let gravity = integrator::ScalarGravityAssets::new(Arc::clone(&packed));
        let context = integrator::ScalarPropagationContext::new(jd0, Arc::clone(&config), gravity);
        let mut reusable = benchmark_result(
            integrator::ReusableFinalNoEventIntegrator::new(context),
            "reusable integrator construction",
        );
        b.iter(|| {
            std::hint::black_box(benchmark_result(
                reusable.propagate(
                    std::hint::black_box(init_equ),
                    0.0,
                    std::hint::black_box(tf_s),
                ),
                "reusable integrator propagation",
            ));
        });
    });

    group.finish();
}

const D5_STRIDE: usize = 7;
const D5_ORDER: usize = 5;
const D5_SIN_GMST: f64 = 0.613_116_851_973_433_8;
const D5_COS_GMST: f64 = 0.789_992_227_325_198_5;
const D5_STATES: [[f64; 6]; 4] = [
    [6_778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
    [6_812.0, 27.0, -18.0, -0.03, 7.64, 0.02],
    [6_745.0, -31.0, 22.0, 0.04, 7.71, -0.01],
    [6_790.0, 15.0, 35.0, -0.02, 7.66, 0.03],
];

// Fixed d/o5 source coefficients are packed once before any timed evaluation.
const FIXED_D5_C: [f64; D5_STRIDE * D5_STRIDE] = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0e-10,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -4.841_653_717_36e-4,
    1.2e-6,
    -2.1e-6,
    0.0,
    0.0,
    0.0,
    0.0,
    9.6e-7,
    -7.4e-7,
    5.3e-7,
    -3.2e-7,
    0.0,
    0.0,
    0.0,
    5.8e-7,
    4.9e-7,
    -3.8e-7,
    2.7e-7,
    -1.6e-7,
    0.0,
    0.0,
    -2.6e-7,
    2.1e-7,
    -1.8e-7,
    1.4e-7,
    -1.1e-7,
    8.0e-8,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
];

const FIXED_D5_S: [f64; D5_STRIDE * D5_STRIDE] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.0e-7,
    1.4e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 6.1e-7, -4.2e-7, 2.8e-7, 0.0, 0.0, 0.0, 0.0, -3.7e-7, 3.1e-7,
    -2.4e-7, 1.5e-7, 0.0, 0.0, 0.0, 1.9e-7, -1.5e-7, 1.2e-7, -9.0e-8, 6.0e-8, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
];

struct GlobalCoeffsRestore(Arc<GlobalCoeffs>);

impl Drop for GlobalCoeffsRestore {
    fn drop(&mut self) {
        GLOBAL_COEFFS.store(Arc::clone(&self.0));
    }
}

struct D5Coeffs {
    packed: Arc<PackedGravityCoeffs>,
}

fn load_required_production_d5_coeffs() -> Arc<D5Coeffs> {
    let path = benchmark_result(
        std::env::var("PART_A_4BG_BENCH_COEFF_PATH"),
        "PART_A_4BG_BENCH_COEFF_PATH is required",
    );
    assert!(
        Path::new(&path).is_absolute(),
        "PART_A_4BG_BENCH_COEFF_PATH must be absolute"
    );
    let expected_sha256 = benchmark_result(
        std::env::var("PART_A_4BG_BENCH_COEFF_SHA256"),
        "PART_A_4BG_BENCH_COEFF_SHA256 is required",
    );
    assert!(
        expected_sha256.len() == 64 && expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "PART_A_4BG_BENCH_COEFF_SHA256 must be 64 hexadecimal characters"
    );

    let bytes = benchmark_result(std::fs::read(&path), "read PART_A_4BG_BENCH_COEFF_PATH");
    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(
        actual_sha256,
        expected_sha256.to_ascii_lowercase(),
        "PART_A_4BG_BENCH_COEFF_SHA256 mismatch"
    );
    benchmark_result(
        lightyear_odeint_rs::load_constants_from_bytes(&bytes, D5_ORDER),
        "production d/o5 coefficient loader",
    );
    let packed = benchmark_option(
        lightyear_odeint_rs::get_global_coeffs_packed(),
        "production coefficient loader did not install packed authority",
    );
    assert_eq!(
        packed.max_order(),
        D5_ORDER,
        "production packed authority must support d/o5"
    );
    let coeffs = Arc::new(D5Coeffs { packed });
    validate_d5_packed_evaluation(&coeffs, "production d/o5 packed authority");
    coeffs
}

fn fixed_dense_d5_coeffs() -> Arc<D5Coeffs> {
    let c_coeffs = FIXED_D5_C.to_vec();
    let s_coeffs = FIXED_D5_S.to_vec();
    let packed = Arc::new(benchmark_result(
        pack_gravity_coeffs(&c_coeffs, &s_coeffs, D5_STRIDE, D5_ORDER),
        "fixed d/o5 coefficient packing",
    ));
    assert_eq!(
        packed.max_order(),
        D5_ORDER,
        "fixed packed authority must support d/o5"
    );
    let coeffs = Arc::new(D5Coeffs { packed });
    validate_d5_packed_evaluation(&coeffs, "fixed d/o5 packed authority");
    coeffs
}

fn d5_benchmark_selected() -> bool {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--baseline" | "--save-baseline" | "--output-format" | "--sample-size"
            | "--measurement-time" | "--warm-up-time" | "--nresamples" | "--noise-threshold"
            | "--confidence-level" => {
                let _ = args.next();
            }
            _ if arg.starts_with('-') => {}
            filter => return filter.contains("gravity_d5_packed"),
        }
    }
    true
}

fn packed_d5_acceleration(
    coeffs: &D5Coeffs,
    state: &[f64; 6],
    cache: &mut GravityCache,
) -> Result<[f64; 3], GravityError> {
    spherical_gravity_impl_sincos_packed(state, D5_SIN_GMST, D5_COS_GMST, cache, &coeffs.packed)
}

fn validate_d5_packed_evaluation(coeffs: &D5Coeffs, context: &str) {
    let state = benchmark_option(D5_STATES.first(), "fixed d/o5 state missing");
    let mut cache = GravityCache::new();
    let acceleration = benchmark_result(packed_d5_acceleration(coeffs, state, &mut cache), context);
    assert!(
        acceleration.iter().all(|value| value.is_finite()),
        "{context} produced a non-finite acceleration"
    );
}

fn bench_packed_d5_pair(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    exact_cache_hit_id: &'static str,
    varying_states_id: &'static str,
    coeffs: &Arc<D5Coeffs>,
) {
    let first_state = benchmark_option(D5_STATES.first(), "fixed d/o5 state missing");
    group.bench_function(exact_cache_hit_id, |b| {
        let mut cache = GravityCache::new();
        black_box(benchmark_result(
            packed_d5_acceleration(coeffs, first_state, &mut cache),
            "d/o5 packed gravity cache warm-up",
        ));
        b.iter(|| {
            black_box(benchmark_result(
                packed_d5_acceleration(coeffs, black_box(first_state), &mut cache),
                "d/o5 packed gravity cache-hit evaluation",
            ))
        });
    });
    group.bench_function(varying_states_id, |b| {
        let mut cache = GravityCache::new();
        b.iter(|| {
            let mut sum = [0.0; 3];
            for state in &D5_STATES {
                let acceleration = benchmark_result(
                    packed_d5_acceleration(coeffs, black_box(state), &mut cache),
                    "d/o5 packed gravity varying-state evaluation",
                );
                for (sum_axis, acceleration_axis) in sum.iter_mut().zip(acceleration) {
                    *sum_axis += acceleration_axis;
                }
            }
            black_box(sum)
        });
    });
}

/// Benchmark exact-cache and changing-state d/o5 packed gravity paths.
///
/// Production bytes are mandatory and hash-checked before Criterion starts.
/// The fixed dense pair is code/radius-shape control with same state schedule.
fn bench_gravity_d5_packed(c: &mut Criterion) {
    if !d5_benchmark_selected() {
        return;
    }
    let _restore = GlobalCoeffsRestore(GLOBAL_COEFFS.load_full());
    let production = load_required_production_d5_coeffs();
    let fixed = fixed_dense_d5_coeffs();

    let mut group = c.benchmark_group("gravity_d5_packed");
    bench_packed_d5_pair(
        &mut group,
        "production_exact_cache_hit",
        "production_varying_states_full_recompute",
        &production,
    );
    bench_packed_d5_pair(
        &mut group,
        "fixed_coeffs_exact_cache_hit",
        "fixed_coeffs_varying_states_full_recompute",
        &fixed,
    );
    group.finish();
}

/// Thread scaling of the RHS: how much slower is one derivative when W workers
/// are each computing their own, versus when one worker computes alone?
///
/// Perfect scaling is a FLAT line. Two measurement choices are what make that
/// line mean anything, and both are easy to get wrong:
///
/// **Threads are created once per width, not once per timed region.** The
/// obvious shape -- `Instant::now()`, `thread::scope`, spawn W, join, `elapsed()`
/// -- charges O(W) serial thread creation to the measurement and then divides it
/// by whatever round length criterion happened to choose. On tc107 that alone
/// moved the reported per-iteration cost of an unchanged 199 ns workload to
/// 387 ns at W=16 with short rounds and 279 ns with long ones. The workers were
/// running at 199 ns the whole time. Here the pool is built before any timing
/// and parked on a channel, so a round measures only the loop.
///
/// **Each worker times its own loop, and the MEDIAN is reported.** Timing the
/// span from first start to last join reports the maximum over W samples, which
/// rises with W even when every thread runs at exactly the 1-thread rate. That
/// turns one descheduled thread on a shared node into an apparent scaling
/// collapse. Throughput is what the campaign buys, so throughput is what this
/// reports; the tail is real but it is a separate question and wants its own
/// number rather than being folded into this one.
///
/// `t` is swept across 512 distinct values. It matters more than it looks:
/// the baseline-state cache and frame-rotation memo key on `t`, while the
/// gravity V/W recurrence cache keys on the position `t` produces. A fixed `t`
/// reuses all three, though gravity still reruns coefficient-dependent
/// summation. Historical fixed-`t` fractions predate acceleration-result cache
/// removal and require remeasurement. A real integration lands every RK stage
/// on a distinct time, so this does too.
fn bench_rhs_thread_scaling(c: &mut Criterion) {
    use std::sync::mpsc;
    use std::sync::Barrier;
    use std::time::Duration;

    let mut group = c.benchmark_group("rhs_thread_scaling");
    let order = 5;
    let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order);
    let packed = Arc::new(benchmark_result(
        pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order),
        "RHS thread scaling coefficient packing",
    ));
    let config = Arc::new(create_prod_force_config());

    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let delta = [1.0e-3, -2.0e-3, 7.0e-4, 2.0e-6, -4.0e-6, 1.0e-6];

    for width in [1usize, 8, 16, 32, 64] {
        // One pool per width, alive for the whole of this benchmark id.
        let barrier = Arc::new(Barrier::new(width));
        let (result_tx, result_rx) = mpsc::channel::<u128>();
        let mut job_tx = Vec::with_capacity(width);
        let mut handles = Vec::with_capacity(width);

        for _ in 0..width {
            let (tx, rx) = mpsc::channel::<u64>();
            job_tx.push(tx);
            let worker = benchmark_result(
                LightyearRHS::try_new(
                    init_equ,
                    0.0,
                    2_460_000.5,
                    Arc::clone(&config),
                    Arc::clone(&packed),
                ),
                "thread-scaling worker RHS construction",
            );
            let barrier = Arc::clone(&barrier);
            let result_tx = result_tx.clone();
            handles.push(std::thread::spawn(move || {
                let times: Vec<f64> = (0..512).map(|i| f64::from(i) * 0.37).collect();
                let time_count = benchmark_result(
                    u64::try_from(times.len()),
                    "thread-scaling time sample count exceeds u64",
                );
                // Warm this worker's caches and fault in its gravity cache
                // before any round is measured.
                for iteration in 0_u64..2_000 {
                    let time = cyclic_time(
                        &times,
                        time_count,
                        iteration,
                        "thread-scaling warm-up time sample missing",
                    );
                    black_box(benchmark_result(
                        worker.compute_internal(black_box(&delta), black_box(time)),
                        "thread-scaling warm-up RHS evaluation",
                    ));
                }
                // `recv` fails once the sender is dropped below, which is how
                // the pool is torn down.
                while let Ok(iters) = rx.recv() {
                    barrier.wait();
                    let start = std::time::Instant::now();
                    let mut acc = 0.0f64;
                    for iteration in 0..iters {
                        let t = cyclic_time(
                            &times,
                            time_count,
                            iteration,
                            "thread-scaling timed time sample missing",
                        );
                        let out = benchmark_result(
                            worker.compute_internal(black_box(&delta), black_box(t)),
                            "thread-scaling timed RHS evaluation",
                        );
                        let [_, _, _, acceleration_x, _, _] = out;
                        acc += acceleration_x;
                    }
                    let elapsed = start.elapsed();
                    black_box(acc);
                    if result_tx.send(elapsed.as_nanos()).is_err() {
                        break;
                    }
                }
            }));
        }
        drop(result_tx);

        group.bench_with_input(BenchmarkId::new("threads", width), &width, |b, &w| {
            b.iter_custom(|iters| {
                for tx in &job_tx {
                    benchmark_result(
                        tx.send(iters),
                        "thread-scaling worker exited before dispatch",
                    );
                }
                let mut per_thread = Vec::with_capacity(w);
                for _ in 0..w {
                    per_thread.push(benchmark_result(
                        result_rx.recv(),
                        "thread-scaling worker exited before result",
                    ));
                }
                per_thread.sort_unstable();
                let median_index =
                    benchmark_option(w.checked_div(2), "thread count cannot be zero");
                let median_nanos = benchmark_option(
                    per_thread.get(median_index),
                    "thread-scaling median sample missing",
                );
                Duration::from_nanos(benchmark_result(
                    u64::try_from(*median_nanos),
                    "thread-scaling duration exceeds u64 nanoseconds",
                ))
            });
        });

        drop(job_tx);
        for handle in handles {
            if handle.join().is_err() {
                benchmark_failure("thread-scaling worker panicked");
            }
        }
    }
    group.finish();
}

/// Which force family the production RHS actually spends its time on.
///
/// Cumulative ablation: each arm adds one family to the previous one, so
/// consecutive differences price that family. Measurement only -- no arm is a
/// proposal to change the physics. Exists because an instruction profile is a
/// poor proxy for time here, and this group answers the same question in
/// wall-clock.
fn bench_rhs_force_ablation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rhs_force_ablation");
    let order = 5;
    let (c_coeffs, s_coeffs, stride) = create_test_coefficients(order);
    let packed = Arc::new(benchmark_result(
        pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order),
        "RHS force ablation coefficient packing",
    ));

    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let delta = [1.0e-3, -2.0e-3, 7.0e-4, 2.0e-6, -4.0e-6, 1.0e-6];

    let arms: [(&str, i32, i32); 5] = [
        ("harmonics_only", 0, 0),
        (
            "plus_thirdbody",
            ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY,
            0,
        ),
        (
            "plus_srp",
            ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY | ForceFlags::SRP,
            0,
        ),
        (
            "plus_drag_jb2008",
            ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY | ForceFlags::SRP | ForceFlags::DRAG,
            3,
        ),
        (
            "prod_full",
            ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY,
            3,
        ),
    ];

    for (label, flags, atm) in arms {
        let mut config = create_prod_force_config();
        config.force_flags = flags;
        config.atm_model = atm;
        let rhs = benchmark_result(
            LightyearRHS::try_new(
                init_equ,
                0.0,
                2_460_000.5,
                Arc::new(config),
                Arc::clone(&packed),
            ),
            "force-ablation RHS construction",
        );

        // Sweep `t` so the frame-rotation and time-scale caches behave as they
        // do in a real integration, where every RK stage lands on a distinct
        // time. Holding `t` fixed lets those caches hit twice per call and
        // flatters the result by roughly a factor of two.
        let samples: Vec<f64> = (0..512).map(|i| f64::from(i) * 0.37).collect();
        group.bench_function(label, |b| {
            b.iter(|| {
                let mut acc = 0.0f64;
                for t in &samples {
                    let out = benchmark_result(
                        rhs.compute_internal(black_box(&delta), black_box(*t)),
                        "force-ablation RHS evaluation",
                    );
                    let [_, _, _, acceleration_x, _, _] = out;
                    acc += acceleration_x;
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

/// Decompose the force-free part of the RHS by spherical-harmonic order.
///
/// All arms run with `force_flags = 0`. `sph_order = 0` skips the harmonic block
/// entirely -- and with it the frame rotation, which is resolved INSIDE that
/// block -- so arm 0 is the irreducible floor: baseline `equinoc2eci`, the
/// Battin/Encke correction, the TAI->UTC conversion, and call overhead.
/// `order_1 - order_0` is therefore rotation plus the cheapest harmonics, and
/// the slope across 1..8 is the harmonics term alone.
///
/// Measured on AMD EPYC 7702 at 512 stages per iteration: the floor is about
/// 62% of the order-5 cost, rotation about 15%, harmonics about 24%. Gravity is
/// NOT the dominant term, which is the point of having this group.
fn bench_rhs_order_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("rhs_order_sweep");

    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let delta = [1.0e-3, -2.0e-3, 7.0e-4, 2.0e-6, -4.0e-6, 1.0e-6];
    let samples: Vec<f64> = (0..512).map(|i| f64::from(i) * 0.37).collect();

    for order in [0usize, 1, 2, 3, 5, 8] {
        // Order 0 still needs a valid coefficient block to construct with.
        let coeff_order = order.max(1);
        let (c_coeffs, s_coeffs, stride) = create_test_coefficients(coeff_order);
        let packed = Arc::new(benchmark_result(
            pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, coeff_order),
            "unforced RHS coefficient packing",
        ));

        let mut config = create_prod_force_config();
        config.force_flags = 0;
        config.atm_model = 0;
        config.sph_order = order;

        let rhs = benchmark_result(
            LightyearRHS::try_new(init_equ, 0.0, 2_460_000.5, Arc::new(config), packed),
            "order-sweep RHS construction",
        );

        group.bench_with_input(BenchmarkId::new("sph_order", order), &order, |b, _| {
            b.iter(|| {
                let mut acc = 0.0f64;
                for t in &samples {
                    let out = benchmark_result(
                        rhs.compute_internal(black_box(&delta), black_box(*t)),
                        "order-sweep RHS evaluation",
                    );
                    let [_, _, _, acceleration_x, _, _] = out;
                    acc += acceleration_x;
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rhs_thread_scaling,
    bench_rhs_force_ablation,
    bench_rhs_order_sweep,
    bench_spherical_gravity,
    bench_equinoc_to_eci,
    bench_eci_to_equinoc,
    bench_gravity_batch,
    bench_conversion_chain,
    bench_batch_conversions,
    bench_eci_to_ecef,
    bench_ecef_to_eci,
    bench_eci_ecef_roundtrip,
    bench_integrator_hf,
    bench_solver_compare_hf,
    bench_rhs_compute_internal_prod,
    bench_rhs_compute_internal_prod_sequence,
    bench_gravity_d5_packed,
    bench_sigma_batch_final_only_prod,
    bench_reusable_final_integrator_prod,
);
criterion_main!(benches);
