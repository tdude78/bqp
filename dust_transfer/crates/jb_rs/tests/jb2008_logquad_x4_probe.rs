use jb_rs::jb2008::{
    jb2008_density, jb2008_density_fitted_v7, jb2008_density_logquad_x4_approx_v1,
    jb2008_density_logquad_x4_approx_v2, Jb2008Input, JB2008_LOGQUAD_X4_APPROX_V1_MODEL_NAME,
    JB2008_LOGQUAD_X4_APPROX_V1_TRANSFORM,
};

fn orekit_local_input(altitude_km: f64) -> Jb2008Input {
    Jb2008Input {
        mjd_utc: 52_951.003_805_740_744,
        sun_declination_rad: -0.285_987_757_544_287,
        // The sealed Orekit pair, differenced: sat_ra 1.282_118_868_515_03
        // minus sun_ra 3.046_653_643_566_772. Kept as the subtraction so the
        // provenance of both halves stays legible; one rounding, exactly as
        // the kernel used to perform it.
        hour_angle_rad: 1.282_118_868_515_03 - 3.046_653_643_566_772,
        sat_geocentric_lat_rad: -1.487_718_654_399_9,
        sat_altitude_m: altitude_km * 1000.0,
        f10: 91.00,
        f10b: 137.10,
        s10: 108.80,
        s10b: 123.80,
        m10: 116.70,
        m10b: 128.50,
        y10: 168.00,
        y10b: 138.60,
        dst_temperature_correction_k: 43.0,
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "the broad-grid receipt intentionally preserves its established binary64 input lattice"
)]
fn broad_grid_input(
    mjd_utc: f64,
    altitude_km: f64,
    latitude_deg: f64,
    local_solar_time_hour: f64,
) -> Jb2008Input {
    let solar_phase = std::f64::consts::TAU * (mjd_utc - 51_999.75) / 365.2422;
    // The Sun's right ascension used to be synthesized here only so that the
    // satellite's could be built as `sun_ra + hour_angle` and the kernel could
    // subtract it back off. The kernel takes the hour angle, which this fixture
    // already had in hand.
    let hour_angle = local_solar_time_hour * std::f64::consts::TAU / 24.0 - std::f64::consts::PI;
    let f10b = 120.0 + 25.0 * solar_phase.cos();
    let s10b = 112.0 + 18.0 * (solar_phase + 0.2).sin();
    let m10b = 128.0 + 16.0 * (solar_phase - 0.5).cos();
    let y10b = 138.0 + 20.0 * (solar_phase + 0.7).sin();
    Jb2008Input {
        mjd_utc,
        sun_declination_rad: 23.44_f64.to_radians() * solar_phase.sin(),
        hour_angle_rad: hour_angle,
        sat_geocentric_lat_rad: latitude_deg.to_radians(),
        sat_altitude_m: altitude_km * 1000.0,
        f10: f10b + 6.0 * (solar_phase + 0.1).sin(),
        f10b,
        s10: s10b + 5.0 * (solar_phase + 0.4).cos(),
        s10b,
        m10: m10b + 4.0 * (solar_phase - 0.3).sin(),
        m10b,
        y10: y10b + 7.0 * (solar_phase + 0.8).cos(),
        y10b,
        dst_temperature_correction_k: 25.0 + 18.0 * (solar_phase - 0.2).sin(),
    }
}

#[test]
fn shared_core_checksum_tracks_corrected_species_reduction_bits() {
    // Shared-core regression checksum, not independent validation. Exact rows
    // at 91/120/200/1000/3000/35000 km also match direct Orekit vectors; the
    // remaining exact rows and every x4 value are internal receipts only.
    //
    // RE-PINNED for the species round-trip removal in `jb_density`, which
    // stopped carrying the five number densities as `ln(x)` and now carries the
    // linear factor and the log offset separately: `exp(ln(x) + y)` became
    // `x * exp(y)`, retiring five `ln` calls per density evaluation.
    //
    // ONLY THE PROBE COLUMN MOVES, and that asymmetry is the point. The change
    // is gated on `QuadratureProfile::RETIRE_SPECIES_ROUND_TRIP`, which is true
    // for the x4 approximation and FALSE for the exact profile, because
    // `orekit_synthetic_mapping_matches_rust_primitive_kernel` requires the
    // exact profile to reproduce the sealed Orekit 13.1.2 fixture bit for bit.
    // Orekit computes the logarithms, so the restructuring loses that by
    // construction; ungated it turned 11 fixture cases red. Every exact value
    // below is therefore unchanged, including all six that match direct Orekit
    // vectors.
    //
    // `jb_tsub_l`'s `powf(2.5)` -> `x * x * sqrt(x)` retirement landed in the
    // same commit and applies to BOTH profiles, but is invisible here: it
    // perturbs the exospheric temperature, `orekit_local_input` holds all eight
    // solar indices and both angles fixed across these ten rows, and at those
    // fixed values the substitution happens to be bit-exact. It moves 4 of the
    // 278 varied inputs in `tests/jb2008_libm_probe.rs` and none of these, and
    // it leaves the sealed Orekit fixture green.
    //
    // probe column, old -> new:
    //
    // |     km | ULP | relative  |
    // |-------:|----:|----------:|
    // |     90 |  -5 | 6.120e-16 |
    // |     91 |  -2 | 3.031e-16 |
    // |    105 | -23 | 3.498e-15 |
    // |    120 |  -9 | 1.620e-15 |
    // |    200 | -31 | 5.707e-15 |
    // |    400 | +18 | 2.711e-15 |
    // |    500 |  +8 | 1.765e-15 |
    // |  1,000 |  -6 | 8.617e-16 |
    // |  3,000 |  -4 | 6.057e-16 |
    // | 35,000 |   0 |         0 |
    //
    // 35,000 km does not move because at that altitude the sum is dominated by
    // hydrogen, whose factor is the literal `1.0` and which never had a `ln` to
    // retire. Worst move 5.707e-15, which is 88x inside the 5e-13 the direct
    // Orekit vectors are asserted at.
    //
    // The direction is not a wash. Adjudicated at 60 decimal digits over all
    // 1,601 `(factor, offset)` pairs the corpus produces, the retired form's
    // worst relative error against the true value was 1.528e-14 (68.80 ULP) and
    // its mean 1.458e-15; the landed form's worst is 1.782e-16 (0.80 ULP) and
    // its mean 4.980e-17. The landed form is nearer on 1,298 pairs and the
    // retired form on 8. The new probe bits are the more accurate ones -- which
    // is exactly why they cannot be used on the exact profile, whose contract is
    // fidelity to Orekit rather than to the true value.
    let cases = [
        (90.0, 0x3ecd_063c_d0f5_3dc0, 0x3ecd_063c_d0f5_3dbb),
        (91.0, 0x3ec7_7149_31e9_f622, 0x3ec7_7149_31e9_ef7d),
        (105.0, 0x3e87_5cc8_e386_51aa, 0x3e87_5cc8_e20b_9f6d),
        (120.0, 0x3e53_be2b_a665_fe0f, 0x3e53_be2b_7cdc_5b89),
        (200.0, 0x3df3_4c7a_5183_4365, 0x3df3_4c7a_3fa2_eeba),
        (400.0, 0x3d87_970e_cba2_69b5, 0x3d87_970f_f778_5fd3),
        (500.0, 0x3d60_1a98_9681_01cf, 0x3d60_1a99_b85e_e5fd),
        (1000.0, 0x3ce8_bd1c_1c45_728b, 0x3ce8_bd1c_9b7e_4d45),
        (3000.0, 0x3c87_75ff_fb88_6d4a, 0x3c87_7600_226b_fded),
        (35_000.0, 0x3c52_7c8f_ee50_4f59, 0x3c52_7c8f_5a69_643e),
    ];
    for (altitude_km, expected_exact, expected_probe) in cases {
        let input = orekit_local_input(altitude_km);
        assert_eq!(
            jb2008_density(input).expect("exact density").to_bits(),
            expected_exact,
            "exact altitude_km={altitude_km}",
        );
        assert_eq!(
            jb2008_density_logquad_x4_approx_v1(input)
                .expect("approximation density")
                .to_bits(),
            expected_probe,
            "probe altitude_km={altitude_km}",
        );
    }
}

#[test]
fn x4_approximation_has_explicit_nonexact_identity() {
    assert_eq!(
        JB2008_LOGQUAD_X4_APPROX_V1_MODEL_NAME,
        "orekit_13_1_2_jb2008_logquad_x4_approx_v1"
    );
    assert_eq!(
        JB2008_LOGQUAD_X4_APPROX_V1_TRANSFORM,
        "exact_jb2008_log_intervals_0.010_0.025_0.075_times_4"
    );
    assert!(
        jb2008_density_logquad_x4_approx_v1(orekit_local_input(400.0))
            .expect("approximation density")
            .is_finite()
    );
}

#[test]
fn x4_broad_grid_density_error_stays_within_candidate_threshold() {
    let mjd_utc = [51_999.75, 54_000.0, 57_000.25, 60_000.0, 60_648.5];
    let altitude_km = [
        90.0, 91.0, 100.0, 105.0, 120.0, 200.0, 240.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0,
        1500.0, 2000.0, 3000.0, 5000.0, 35_000.0,
    ];
    let latitude_deg = [-75.0, -45.0, 0.0, 45.0, 75.0];
    let local_solar_time_hour = [0.0, 6.0, 12.0, 18.0];
    let mut relative_errors = Vec::with_capacity(1800);

    for mjd_utc in mjd_utc {
        for altitude_km in altitude_km {
            for latitude_deg in latitude_deg {
                for local_solar_time_hour in local_solar_time_hour {
                    let input =
                        broad_grid_input(mjd_utc, altitude_km, latitude_deg, local_solar_time_hour);
                    let exact = jb2008_density(input).expect("exact density");
                    let probe =
                        jb2008_density_logquad_x4_approx_v1(input).expect("approximation density");
                    relative_errors.push((probe - exact).abs() / exact);
                }
            }
        }
    }

    relative_errors.sort_by(f64::total_cmp);
    let p99_index = relative_errors.len().saturating_mul(99) / 100;
    let p99 = relative_errors.get(p99_index).copied().unwrap_or(f64::NAN);
    let max = relative_errors.last().copied().unwrap_or(f64::NAN);
    println!(
        "x4_density_grid samples={} p99_relative_error={p99:e} max_relative_error={max:e}",
        relative_errors.len()
    );
    assert_eq!(relative_errors.len(), 1800);
    assert!(
        max <= 3.0e-6,
        "x4 density max relative error {max:e} exceeds candidate threshold"
    );
}

/// The bound that authorizes `atm_model` 6, and the ONLY gate that bounds it.
///
/// # Read this before citing an accuracy gate for the abscissa count
///
/// `strict_hf_production_arc_accuracy` and
/// `strict_hf_v3_production_arc_accuracy` do NOT bound this profile. Both
/// difference an arc at production `eps` against the SAME arc at a tighter
/// `eps`, so a quadrature bias is common-mode and cancels exactly; what they
/// measure is integrator truncation. Measured on 2026-08-09 they stay green at
/// `middle 0.400`, whose density error is 2.829e-3 — 28x over the bound below —
/// and model 6 reads 14x "better" than model 5 on the V3 gate while being 36x
/// worse here. This test is what stands between the abscissa count and a
/// silently coarser atmosphere.
///
/// # Where 1.0e-4 comes from
///
/// The 3.0e-6 bound this replaces asserted that model 5 tracks model 4 over a
/// lattice running to 35,000 km — a reproducibility claim written far outside
/// the 626--986 km band the campaign flies. The user re-scoped it to 1.0e-4 on
/// 2026-08-09 (R16 decision doc, R22 addendum). The bound is still asserted
/// over the FULL lattice rather than a production-band subset, because model 6
/// passes it there anyway: keeping the wider domain costs nothing and keeps the
/// far field from drifting unobserved.
#[test]
fn v2_broad_grid_density_error_stays_within_rescoped_bound() {
    // The crate's constant, not a local literal. The unit test advertised as
    // this gate's poison proof used to carry its own copy of `1.0e-4`, so
    // relaxing the value here left that proof green and vacuous.
    const RESCOPED_BOUND: f64 = jb_rs::jb2008::V2_RESCOPED_DENSITY_BOUND;

    let mjd_utc = [51_999.75, 54_000.0, 57_000.25, 60_000.0, 60_648.5];
    let altitude_km = [
        90.0, 91.0, 100.0, 105.0, 120.0, 200.0, 240.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0,
        1500.0, 2000.0, 3000.0, 5000.0, 35_000.0,
    ];
    let latitude_deg = [-75.0, -45.0, 0.0, 45.0, 75.0];
    let local_solar_time_hour = [0.0, 6.0, 12.0, 18.0];

    let mut relative_errors = Vec::with_capacity(1800);
    let mut v1_relative_errors = Vec::with_capacity(1800);
    // The band production actually flies is 626--986 km; these are the lattice
    // rows that bracket it. Reported, not asserted, so the gate keeps its wider
    // domain while the campaign-relevant number stays visible in the receipt.
    let mut production_band_errors = Vec::new();
    let mut worst_altitude_km = f64::NAN;
    let mut worst_error = 0.0_f64;

    for mjd_utc in mjd_utc {
        for altitude_km in altitude_km {
            for latitude_deg in latitude_deg {
                for local_solar_time_hour in local_solar_time_hour {
                    let input =
                        broad_grid_input(mjd_utc, altitude_km, latitude_deg, local_solar_time_hour);
                    let exact = jb2008_density(input).expect("exact density");
                    let v2 = jb2008_density_logquad_x4_approx_v2(input)
                        .expect("model 6 density must be defined everywhere model 4 is");
                    let v1 = jb2008_density_logquad_x4_approx_v1(input).expect("model 5 density");
                    let error = (v2 - exact).abs() / exact;
                    if error > worst_error {
                        worst_error = error;
                        worst_altitude_km = altitude_km;
                    }
                    if (600.0..=1000.0).contains(&altitude_km) {
                        production_band_errors.push(error);
                    }
                    relative_errors.push(error);
                    v1_relative_errors.push((v1 - exact).abs() / exact);
                }
            }
        }
    }

    relative_errors.sort_by(f64::total_cmp);
    production_band_errors.sort_by(f64::total_cmp);
    v1_relative_errors.sort_by(f64::total_cmp);
    let max = relative_errors.last().copied().unwrap_or(f64::NAN);
    let band_max = production_band_errors.last().copied().unwrap_or(f64::NAN);
    let v1_max = v1_relative_errors.last().copied().unwrap_or(f64::NAN);
    println!(
        "v2_density_grid samples={} max_relative_error={max:e} worst_at_km={worst_altitude_km} \
         production_band_max={band_max:e} model5_max_for_reference={v1_max:e}",
        relative_errors.len()
    );

    assert_eq!(relative_errors.len(), 1800);
    assert!(
        max <= RESCOPED_BOUND,
        "model 6 density max relative error {max:e} exceeds the re-scoped bound {RESCOPED_BOUND:e}"
    );
    // Non-vacuity: model 6 must actually BE the coarse profile. If someone
    // re-points `jb2008_density_logquad_x4_approx_v2` at v1's log steps this
    // test would otherwise keep passing while measuring model 5 -- the exact
    // failure mode `tests-reading-sealed-constant-degrade-silently` describes.
    // Model 5's own maximum over this lattice is 1.605e-6.
    assert!(
        max > 1.0e-5,
        "model 6 max relative error {max:e} is at model-5 scale; this test is no longer \
         measuring the coarse profile"
    );
    assert!(
        v1_max < 3.0e-6,
        "model 5 must be UNCHANGED by the model 6 landing: {v1_max:e} is not its sealed scale"
    );
}

/// The authority for `atm_model` 7, and the only thing that bounds it.
///
/// Same lattice, same bound and same reporting as model 6's gate, so the two
/// receipts are read side by side. The bound is the 1.0e-4 the user re-scoped
/// on 2026-08-09 (R16 decision doc, R22 addendum); it is asserted over the FULL
/// lattice rather than a production-band subset because model 7 passes it there
/// anyway.
///
/// Model 7's error is model 6's error to four significant digits, because the
/// fit residual (worst scalar 7.434e-6) sits an order of magnitude below the
/// quadrature bias model 7 inherits. That is a fact about the ladder, and it is
/// also a hazard for this test: NO assertion on the magnitude of model 7's
/// error can tell model 7 apart from model 6. The non-vacuity floor below is
/// therefore structural -- it counts rows where the two profiles DISAGREE.
#[test]
fn v7_broad_grid_density_error_stays_within_rescoped_bound() {
    const RESCOPED_BOUND: f64 = 1.0e-4;

    let mjd_utc = [51_999.75, 54_000.0, 57_000.25, 60_000.0, 60_648.5];
    let altitude_km = [
        90.0, 91.0, 100.0, 105.0, 120.0, 200.0, 240.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0,
        1500.0, 2000.0, 3000.0, 5000.0, 35_000.0,
    ];
    let latitude_deg = [-75.0, -45.0, 0.0, 45.0, 75.0];
    let local_solar_time_hour = [0.0, 6.0, 12.0, 18.0];

    let mut relative_errors = Vec::with_capacity(1800);
    let mut v6_relative_errors = Vec::with_capacity(1800);
    // The band production actually flies is 626--986 km; these are the lattice
    // rows that bracket it. Reported, not asserted, matching model 6's gate.
    let mut production_band_errors = Vec::new();
    let mut worst_altitude_km = f64::NAN;
    let mut worst_error = 0.0_f64;
    // Structural non-vacuity: rows at or above 500 km run BOTH fitted plans, so
    // the fit must move the density there. Rows below 105 km run neither.
    let mut fitted_rows_that_moved = 0_usize;
    let mut fitted_rows = 0_usize;
    let mut sub_105_rows_identical_to_v6 = 0_usize;
    let mut sub_105_rows = 0_usize;

    for mjd_utc in mjd_utc {
        for altitude_km in altitude_km {
            for latitude_deg in latitude_deg {
                for local_solar_time_hour in local_solar_time_hour {
                    let input =
                        broad_grid_input(mjd_utc, altitude_km, latitude_deg, local_solar_time_hour);
                    let exact = jb2008_density(input).expect("exact density");
                    let v7 = jb2008_density_fitted_v7(input)
                        .expect("model 7 density must be defined everywhere model 4 is");
                    let v6 = jb2008_density_logquad_x4_approx_v2(input).expect("model 6 density");
                    let error = (v7 - exact).abs() / exact;
                    if error > worst_error {
                        worst_error = error;
                        worst_altitude_km = altitude_km;
                    }
                    if (600.0..=1000.0).contains(&altitude_km) {
                        production_band_errors.push(error);
                    }
                    if altitude_km >= 500.0 {
                        fitted_rows += 1;
                        if v7.to_bits() != v6.to_bits() {
                            fitted_rows_that_moved += 1;
                        }
                    }
                    if altitude_km < 105.0 {
                        sub_105_rows += 1;
                        if v7.to_bits() == v6.to_bits() {
                            sub_105_rows_identical_to_v6 += 1;
                        }
                    }
                    relative_errors.push(error);
                    v6_relative_errors.push((v6 - exact).abs() / exact);
                }
            }
        }
    }

    relative_errors.sort_by(f64::total_cmp);
    production_band_errors.sort_by(f64::total_cmp);
    v6_relative_errors.sort_by(f64::total_cmp);
    let max = relative_errors.last().copied().unwrap_or(f64::NAN);
    let band_max = production_band_errors.last().copied().unwrap_or(f64::NAN);
    let v6_max = v6_relative_errors.last().copied().unwrap_or(f64::NAN);
    println!(
        "v7_density_grid samples={} max_relative_error={max:e} worst_at_km={worst_altitude_km} \
         production_band_max={band_max:e} model6_max_for_reference={v6_max:e} \
         fitted_rows_that_moved={fitted_rows_that_moved}/{fitted_rows} \
         sub_105_identical_to_model6={sub_105_rows_identical_to_v6}/{sub_105_rows}",
        relative_errors.len()
    );

    assert_eq!(relative_errors.len(), 1800);
    assert!(
        max <= RESCOPED_BOUND,
        "model 7 density max relative error {max:e} exceeds the re-scoped bound {RESCOPED_BOUND:e}"
    );
    // Non-vacuity, structural. Re-pointing `jb2008_density_fitted_v7` at model
    // 6 would leave every assertion on the ERROR above satisfied, because the
    // two errors agree to four digits by construction. Only disagreement
    // between the profiles proves the fit is the thing being measured.
    assert!(
        fitted_rows_that_moved * 2 > fitted_rows,
        "model 7 moved the density on only {fitted_rows_that_moved} of {fitted_rows} rows at or \
         above 500 km; the fitted plans are not engaged and this test is measuring model 6"
    );
    // The other half of the same claim: below 105 km neither fixed plan runs,
    // so model 7 has nothing to replace and MUST be model 6 bit for bit. If
    // this fails the fit has leaked into a segment it was never fitted on.
    assert_eq!(
        sub_105_rows_identical_to_v6, sub_105_rows,
        "model 7 differs from model 6 below 105 km, where no fixed plan is walked"
    );
    // Model 6 must be UNCHANGED by the model 7 landing.
    assert!(
        (5.0e-5..1.0e-4).contains(&v6_max),
        "model 6 must be UNCHANGED by the model 7 landing: {v6_max:e} is not its 5.747e-5 scale"
    );
}

/// The temperature domain guard, proved on both sides and in the middle.
///
/// `dst_temperature_correction_k` enters the exospheric temperature additively
/// and is the only driver that does, so sweeping it walks Texo across the fit
/// domain without perturbing anything else. Inside `[500, 2600]` K model 7 runs
/// the fit and must differ from model 6; outside it both fitted accessors
/// return the walked plan, so model 7 must equal model 6 BIT FOR BIT.
///
/// Asserting the ordered pattern -- identical, then differing, then identical
/// -- is what makes this more than a smoke test: a guard stuck open produces no
/// identical tail, a guard stuck closed produces no differing middle, and an
/// inverted comparison swaps the two.
#[test]
fn model_7_falls_back_to_the_walked_plan_outside_its_temperature_domain() {
    // 700 km: inside the band production flies, and above the 500 km threshold
    // where BOTH fixed plans are used, so the guard is exercised twice a call.
    let altitude_km = 700.0;
    let mut below_domain_identical = 0_usize;
    let mut in_domain_differing = 0_usize;
    let mut above_domain_identical = 0_usize;
    let mut transitions = Vec::new();
    let mut previous_identical: Option<bool> = None;

    // Texo runs low-to-high with the offset, so the sweep crosses the lower
    // bound first and the upper bound last.
    // Stepped over integers rather than accumulated in f64: the sweep bound is
    // then exact, and the printed transition offsets are the round numbers they
    // claim to be rather than a drifting partial sum.
    for step in 0..=400_i32 {
        let offset = f64::from(step).mul_add(10.0, -1500.0);
        let mut input = broad_grid_input(57_000.25, altitude_km, 0.0, 12.0);
        input.dst_temperature_correction_k = offset;
        if let (Ok(v7), Ok(v6)) = (
            jb2008_density_fitted_v7(input),
            jb2008_density_logquad_x4_approx_v2(input),
        ) {
            let identical = v7.to_bits() == v6.to_bits();
            if previous_identical != Some(identical) {
                transitions.push((offset, identical));
                previous_identical = Some(identical);
            }
            if identical {
                if in_domain_differing == 0 {
                    below_domain_identical += 1;
                } else {
                    above_domain_identical += 1;
                }
            } else {
                in_domain_differing += 1;
            }
        }
    }

    println!(
        "v7_domain_sweep below_identical={below_domain_identical} in_domain_differing=\
         {in_domain_differing} above_identical={above_domain_identical} transitions={transitions:?}"
    );

    assert!(
        below_domain_identical > 0,
        "no sampled temperature fell below the fit domain; the sweep never tested the low guard"
    );
    assert!(
        in_domain_differing > 0,
        "model 7 never differed from model 6 anywhere in the sweep; the fit is not engaged and \
         the fallback assertions below are vacuous"
    );
    assert!(
        above_domain_identical > 0,
        "no sampled temperature rose above the fit domain; the sweep never tested the high guard"
    );
    // Exactly two crossings: into the domain and out of it. More would mean the
    // guard is chattering; fewer means one side was never reached.
    assert_eq!(
        transitions.len(),
        3,
        "expected an identical/differing/identical sweep with two crossings, got {transitions:?}"
    );
}
