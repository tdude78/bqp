//! Round-13 probe: dump whole-density bits over the logquad corpus and the
//! Orekit vector altitudes, so a libm-retiring edit can be priced against a
//! captured baseline rather than against a re-derivation of itself.
//!
//! Prints one `IDX <index> <hexbits> <decimal>` line per input. Diff two runs.
//!
//! # Every test here is `#[ignore]`d, because none of them can fail
//!
//! This is a MEASUREMENT HARNESS, not a gate. Each test prints; an `Err` from
//! the density is printed too, not raised. Measured 2026-08-14 at `d88fd76`:
//! with `jb2008_density` returning `rho * 2.0`, all five tests here stayed
//! GREEN, while the same poison reddened 9 real tests elsewhere in this crate
//! (5 in `src/lib.rs` — including `matches_exact_orekit_jar_35000km_vector`
//! and `fixed_lower_plan_preserves_exact_density_bits` — and 4 in
//! `tests/jb2008_logquad_x4_probe.rs`, which despite its name DOES assert
//! error bounds). So the coverage is elsewhere, and until this commit the
//! only thing these five contributed to a sweep was five rows of green that
//! no change could ever move.
//!
//! They stay because the captured-baseline diff is the point. Run them
//! deliberately:
//!
//! ```sh
//! cargo test -p jb_rs --test jb2008_libm_probe -- --ignored --nocapture
//! ```
//!
//! If you want one of these to GATE something, it needs an assertion against a
//! captured baseline — see `tests/fitted_v7_density_pin.rs` for that shape.

use jb_rs::jb2008::{
    jb2008_density, jb2008_density_fitted_v7, jb2008_density_logquad_x4_approx_v1,
    jb2008_density_logquad_x4_approx_v2, Jb2008Input,
};

/// Byte-identical to the private `orekit_local_input` in `jb2008.rs`.
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

/// Byte-identical to the private `logquad_inputs` in `jb2008.rs`.
fn logquad_inputs() -> Vec<Jb2008Input> {
    (0_i32..257)
        .map(|index| {
            let mut input = orekit_local_input(200.0 + f64::from(index.saturating_mul(37) % 1_300));
            input.mjd_utc += f64::from(index) * 0.125;
            input.sun_declination_rad += f64::from(index % 17) * 0.003;
            // The two right ascensions used to be swept at 0.019 and 0.031
            // per index; only their difference ever reached the kernel, so
            // sweeping the hour angle at 0.012 covers the same ground.
            input.hour_angle_rad =
                (input.hour_angle_rad + f64::from(index) * 0.012).rem_euclid(std::f64::consts::TAU);
            input.sat_geocentric_lat_rad += f64::from(index % 19) * 0.002;
            input.f10 += f64::from(index % 23);
            input.f10b += f64::from(index % 29);
            input
        })
        .collect()
}

fn corpus() -> Vec<Jb2008Input> {
    let mut inputs = logquad_inputs();
    inputs.extend(
        [
            90.0,
            95.0,
            100.0,
            104.999_999,
            105.0,
            105.000_001,
            106.0,
            120.0,
            200.0,
            240.0,
            300.0,
            400.0,
            500.0,
            600.0,
            626.2,
            800.0,
            985.7,
            1000.0,
            1500.0,
            2300.0,
            3000.0,
            35_000.0,
        ]
        .map(orekit_local_input)
        .iter()
        .copied(),
    );
    inputs
}

/// The exact ten altitudes `shared_core_checksum_tracks_corrected_species_
/// reduction_bits` pins, in its order, so a re-pin is a transcription and not a
/// re-derivation.
#[test]
#[ignore = "measurement harness: prints shared-core checksum rows for the Orekit vector altitudes and asserts no value"]
fn dump_shared_core_checksum_rows() {
    for altitude_km in [
        90.0,
        91.0,
        105.0,
        120.0,
        200.0,
        400.0,
        500.0,
        1000.0,
        3000.0,
        35_000.0_f64,
    ] {
        let input = orekit_local_input(altitude_km);
        let exact = jb2008_density(input).expect("exact density");
        let probe = jb2008_density_logquad_x4_approx_v1(input).expect("probe density");
        println!(
            "ROW {altitude_km} {:#018x} {:#018x} {exact:.17e} {probe:.17e}",
            exact.to_bits(),
            probe.to_bits()
        );
    }
}

#[test]
#[ignore = "measurement harness: prints exact-profile density bits and asserts no value"]
fn dump_exact_profile_density_bits() {
    for (index, input) in corpus().into_iter().enumerate() {
        match jb2008_density(input) {
            Ok(rho) => println!("EXACT {index} {:#018x} {rho:.17e}", rho.to_bits()),
            Err(error) => println!("EXACT {index} ERR {error:?}"),
        }
    }
}

#[test]
#[ignore = "measurement harness: prints logquad-v1 density bits and asserts no value"]
fn dump_logquad_profile_density_bits() {
    for (index, input) in corpus().into_iter().enumerate() {
        match jb2008_density_logquad_x4_approx_v1(input) {
            Ok(rho) => println!("LOGQ {index} {:#018x} {rho:.17e}", rho.to_bits()),
            Err(error) => println!("LOGQ {index} ERR {error:?}"),
        }
    }
}

/// The coarse profile behind `atm_model` 6.
///
/// Added because the two dumps above predate models 6 and 7, so a diff of two
/// runs could show green while the profile the campaign actually flies had
/// moved. A bit probe that cannot see the flown model is the same trap as a
/// gate that cannot see the flown model.
#[test]
#[ignore = "measurement harness: prints coarse-v2 (atm_model 6) density bits and asserts no value"]
fn dump_coarse_profile_density_bits() {
    for (index, input) in corpus().into_iter().enumerate() {
        match jb2008_density_logquad_x4_approx_v2(input) {
            Ok(rho) => println!("COARSE {index} {:#018x} {rho:.17e}", rho.to_bits()),
            Err(error) => println!("COARSE {index} ERR {error:?}"),
        }
    }
}

/// The fitted profile behind `atm_model` 7 — the one the campaign flies.
#[test]
#[ignore = "measurement harness: prints fitted-v7 (atm_model 7, the flown one) density bits and asserts no value"]
fn dump_fitted_profile_density_bits() {
    for (index, input) in corpus().into_iter().enumerate() {
        match jb2008_density_fitted_v7(input) {
            Ok(rho) => println!("FIT {index} {:#018x} {rho:.17e}", rho.to_bits()),
            Err(error) => println!("FIT {index} ERR {error:?}"),
        }
    }
}
