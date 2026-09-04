//! Criterion benchmarks for the HF deterministic mass solver (Brent's method).
//!
//! These benchmarks exercise the root-finding hot path used in Phase 5 of
//! constellation optimization. Parametrized by miss-distance geometry and
//! initial bracket width to capture the full range of solver behavior.
//!
//! Run with:
//!   `cargo bench -p dust_estimates_rs --profile criterion-fast --bench hf_mass_solver_bench`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dust_estimates_rs::mass_solver::{
    solve_batch_events_mf_j2_with_status_into, solve_single_event_hf,
    solve_single_event_hf_with_status, solve_single_event_mf_j2_with_status, MassSolverEvent,
    MfJ2MassSolveStatusCode, MfJ2MassSolverEvent, SolverConfig,
};

// Earth gravitational parameter (km^3/s^2) and equatorial radius (km), read
// from `satpy_core` rather than restated. A bench that transcribes a physical
// constant is a second source of truth for it, and it silently stops matching
// the code it is timing the moment the authority moves.
use satpy_core::{MU, RE};

/// Create a `MassSolverEvent` for a given miss-distance geometry.
///
/// `miss_factor` controls how far the secondary is from the primary's propagated
/// position (larger = more mass needed). `tof_hours` controls time-of-flight.
fn create_event(miss_factor: f64, tof_hours: f64) -> MassSolverEvent {
    let r = RE + 500.0; // LEO altitude
    let v = (MU / r).sqrt(); // Circular velocity

    let p_mass = 500.0; // 500 kg primary
    let p_momentum = [0.0, p_mass * v, 0.0];
    let p_velocity = [0.0, v, 0.0];

    // Dust approaching at relative velocity (representative of ASAT/debris scenario)
    let v_rel_mag = 0.5; // 500 m/s relative velocity
    let dv_vec = [v_rel_mag, v, 0.0];
    let v_rel = [
        dv_vec[0] - p_velocity[0],
        dv_vec[1] - p_velocity[1],
        dv_vec[2] - p_velocity[2],
    ];

    let tof_s = tof_hours * 3600.0;

    // Place secondary at a distance that requires varying amounts of mass to deflect
    let base_miss_km = 1.0; // 1 km baseline miss
    let secondary_offset = base_miss_km * miss_factor;

    // Approximate propagated position for LEO circular orbit after tof_s
    let omega = v / r; // Mean motion (rad/s)
    let angle = omega * tof_s;
    let prop_x = r * angle.cos();
    let prop_y = r * angle.sin();

    MassSolverEvent {
        p_momentum,
        dv_vec,
        p_mass,
        p_pos_intercept: [r, 0.0, 0.0],
        tof_s,
        secondary_conj_pos: [prop_x + secondary_offset, prop_y, 0.0],
        min_miss_distance_km: 5.0, // Target 5 km miss distance
        kappa: 2.00,               // Production value
        p_pos_conj_truth: [prop_x, prop_y, 0.0],
        p_pos_conj_equ_0: [prop_x, prop_y, 0.0],
        p_velocity,
        v_rel,
        p_equ_intercept: [0.0; 6], // Not used in LF mode
        p_am_ratio: None,
        p_cd: None,
        p_cr: None,
        p_qm_ratio: None,
        p_r_obj_m: None,
    }
}

fn create_mf_j2_event(miss_factor: f64, tof_hours: f64) -> MfJ2MassSolverEvent {
    let r = RE + 500.0;
    let v = (MU / r).sqrt();
    let tof_s = tof_hours * 3600.0;
    let omega = v / r;
    let angle = omega * tof_s;
    let prop_x = r * angle.cos();
    let prop_y = r * angle.sin();
    let secondary_offset = miss_factor;

    MfJ2MassSolverEvent::new(
        [r, 0.0, 0.0],
        [0.0, v, 0.0],
        [0.5, v, 0.0],
        500.0,
        [prop_x + secondary_offset, prop_y, 0.0],
        tof_s,
        5.0,
        2.0,
    )
}

/// Benchmark single-event LF solve across different miss-distance geometries.
///
/// This exercises the Brent's method root-finding with varying difficulty:
/// - "close" (factor=0.5): secondary close → less mass needed → fewer iterations
/// - "moderate" (factor=2.0): moderate distance → typical case
/// - "far" (factor=10.0): far secondary → more mass, more iterations
/// - "extreme" (factor=50.0): extreme deflection → stress test bracket search
fn bench_lf_solve_geometries(c: &mut Criterion) {
    let mut group = c.benchmark_group("hf_mass_solver_lf_geometry");
    let config = SolverConfig {
        xtol: 1e-6,
        rtol: 1e-6,
        maxiter: 80,
        mass_max: 1e6,
    };

    for (label, miss_factor) in [
        ("close", 0.5),
        ("moderate", 2.0),
        ("far", 10.0),
        ("extreme", 50.0),
    ] {
        let event = create_event(miss_factor, 1.0);
        group.bench_with_input(BenchmarkId::new("geometry", label), &event, |b, event| {
            b.iter(|| {
                solve_single_event_hf(
                    std::hint::black_box(event),
                    std::hint::black_box(&config),
                    None,
                )
            });
        });
    }

    group.finish();
}

/// Benchmark solve with varying solver tolerances (bracket width impact).
///
/// Tighter tolerances require more Brent iterations, giving insight into the
/// marginal cost of each iteration.
fn bench_lf_solve_tolerances(c: &mut Criterion) {
    let mut group = c.benchmark_group("hf_mass_solver_lf_tolerance");
    let event = create_event(5.0, 1.0); // Moderate difficulty

    for (label, xtol) in [
        ("coarse_1e-2", 1e-2),
        ("moderate_1e-4", 1e-4),
        ("production_1e-6", 1e-6),
        ("tight_1e-8", 1e-8),
    ] {
        let config = SolverConfig {
            xtol,
            rtol: 1e-6,
            maxiter: 120,
            mass_max: 1e6,
        };
        group.bench_with_input(BenchmarkId::new("xtol", label), &config, |b, config| {
            b.iter(|| {
                solve_single_event_hf(
                    std::hint::black_box(&event),
                    std::hint::black_box(config),
                    None,
                )
            });
        });
    }

    group.finish();
}

/// Benchmark solve with varying time-of-flight (propagation cost dominance).
///
/// Longer `ToF` means more orbital periods which tests equinoctial propagation.
fn bench_lf_solve_tof(c: &mut Criterion) {
    let mut group = c.benchmark_group("hf_mass_solver_lf_tof");
    let config = SolverConfig {
        xtol: 1e-6,
        rtol: 1e-6,
        maxiter: 80,
        mass_max: 1e6,
    };

    for tof_hours in [0.5, 1.0, 2.0, 4.0] {
        let event = create_event(5.0, tof_hours);
        group.bench_with_input(
            BenchmarkId::new("tof_h", format!("{tof_hours:.1}")),
            &event,
            |b, event| {
                b.iter(|| {
                    solve_single_event_hf(
                        std::hint::black_box(event),
                        std::hint::black_box(&config),
                        None,
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch LF solve (18 events, simulating one individual evaluation).
fn bench_lf_batch_18_events(c: &mut Criterion) {
    let config = SolverConfig {
        xtol: 1e-6,
        rtol: 1e-6,
        maxiter: 80,
        mass_max: 1e6,
    };

    // Create 18 events with varying geometry (representative of production)
    let events: Vec<MassSolverEvent> = (0_u8..18)
        .map(|i| {
            #[expect(
                clippy::suboptimal_flops,
                reason = "benchmark fixture retains its established unfused IEEE-754 input values"
            )]
            let miss_factor = 1.0 + f64::from(i) * 3.0;
            #[expect(
                clippy::suboptimal_flops,
                reason = "benchmark fixture retains its established unfused IEEE-754 input values"
            )]
            let tof_hours = 0.5 + f64::from(i) * 0.2;
            create_event(miss_factor, tof_hours)
        })
        .collect();

    c.bench_function("hf_mass_solver_lf_batch_18", |b| {
        b.iter(|| {
            let mut total = 0.0_f64;
            for event in &events {
                total += solve_single_event_hf(
                    std::hint::black_box(event),
                    std::hint::black_box(&config),
                    None,
                );
            }
            std::hint::black_box(total)
        });
    });
}

/// Benchmark status-code path (`with_status` variant, measures overhead of status tracking).
fn bench_lf_solve_with_status(c: &mut Criterion) {
    let config = SolverConfig {
        xtol: 1e-6,
        rtol: 1e-6,
        maxiter: 80,
        mass_max: 1e6,
    };
    let event = create_event(5.0, 1.0);

    let mut group = c.benchmark_group("hf_mass_solver_status_overhead");

    group.bench_function("without_status", |b| {
        b.iter(|| {
            solve_single_event_hf(
                std::hint::black_box(&event),
                std::hint::black_box(&config),
                None,
            )
        });
    });

    group.bench_function("with_status", |b| {
        b.iter(|| {
            solve_single_event_hf_with_status(
                std::hint::black_box(&event),
                std::hint::black_box(&config),
                None,
            )
        });
    });

    group.finish();
}

fn bench_mf_j2_status_batch_direct_fill(c: &mut Criterion) {
    let config = SolverConfig {
        xtol: 1e-6,
        rtol: 1e-6,
        maxiter: 80,
        mass_max: 1e6,
    };
    let mut group = c.benchmark_group("mf_j2_status_batch_direct_fill");

    for n_events in [8_u8, 32, 128] {
        let events: Vec<MfJ2MassSolverEvent> = (0..n_events)
            .map(|i| {
                let miss_factor = 1.0 + f64::from(i % 17) * 0.5;
                let tof_hours = 0.5 + f64::from(i % 9) * 0.25;
                create_mf_j2_event(miss_factor, tof_hours)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("direct_fill", n_events),
            &events,
            |b, events| {
                let n = events.len();
                let mut masses = vec![0.0; n];
                let mut statuses = vec![MfJ2MassSolveStatusCode::MissAtZeroNonFinite; n];
                let mut miss_zero = vec![0.0; n];
                let mut miss_root = vec![0.0; n];
                let mut miss_upper = vec![0.0; n];
                let mut iterations = vec![0usize; n];
                b.iter(|| {
                    solve_batch_events_mf_j2_with_status_into(
                        std::hint::black_box(events),
                        std::hint::black_box(&config),
                        &mut masses,
                        &mut statuses,
                        &mut miss_zero,
                        &mut miss_root,
                        &mut miss_upper,
                        &mut iterations,
                    );
                    let checksum = masses
                        .first()
                        .zip(miss_zero.first())
                        .zip(miss_root.first())
                        .zip(miss_upper.first())
                        .zip(iterations.first())
                        .zip(statuses.first())
                        .and_then(
                            |(((((mass, zero), root), upper), iterations), status)| {
                                u32::try_from(*iterations).ok().map(|iterations| {
                                    #[expect(
                                        clippy::as_conversions,
                                        reason = "status enum has a pinned i32 representation; preserve checksum semantics"
                                    )]
                                    let status_code = *status as i32;
                                    *mass
                                        + *zero
                                        + *root
                                        + *upper
                                        + f64::from(iterations)
                                        + f64::from(status_code)
                                })
                            },
                        );
                    std::hint::black_box(checksum)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_loop", n_events),
            &events,
            |b, events| {
                b.iter(|| {
                    let mut total = 0.0_f64;
                    for event in events {
                        total += solve_single_event_mf_j2_with_status(
                            std::hint::black_box(event),
                            std::hint::black_box(&config),
                        )
                        .root_mass_kg;
                    }
                    std::hint::black_box(total)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_lf_solve_geometries,
    bench_lf_solve_tolerances,
    bench_lf_solve_tof,
    bench_lf_batch_18_events,
    bench_lf_solve_with_status,
    bench_mf_j2_status_batch_direct_fill,
);
criterion_main!(benches);
