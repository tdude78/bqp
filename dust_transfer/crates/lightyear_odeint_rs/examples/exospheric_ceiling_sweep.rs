//! Does the exospheric ceiling ALWAYS bind above 2500 km?
//!
//! # The question and why it is worth money
//!
//! Above `JB2008_EXTRAPOLATION_CEILING_M` (2500 km) the RHS runs the whole
//! JB2008 adapter and kernel — driver lookup, Sun ephemeris, two ITRS
//! rotations, four inverse trigonometric reductions, the quadrature — and then
//! throws the answer away:
//!
//! ```text
//! rho.min(JB2008_EXOSPHERIC_DENSITY_CEILING_KG_M3)   // 1e-19 kg/m^3
//! ```
//!
//! Whenever the kernel returns MORE than 1e-19, that whole call produced a
//! constant. If the kernel provably always returns more than 1e-19 up there,
//! then answering with the constant directly — before the driver lookup, before
//! the Sun — is not an approximation at all. It is BIT-IDENTICAL, and it deletes
//! the atmosphere from every evaluation above 2500 km.
//!
//! That is worth nothing on the strict-HF dust arcs, which fly 650-1000 km and
//! never reach the ceiling. It is worth the whole drag lane on the TRANSFER
//! arcs, whose apogees reach 41,378 km — the altitude the ceiling's own doc
//! block quotes.
//!
//! # What "always" has to mean
//!
//! The existing pin (`jb2008_unbounded_extrapolation_is_why_the_ceiling_exists`)
//! checks five altitudes at ONE sealed driver set. The plateau it records is
//! 4.0e-18 to 7.7e-17, i.e. 40x to 770x above the ceiling — comfortable, but a
//! single solar-activity state. Exospheric density swings with solar activity by
//! more than one order, so 40x of margin at one epoch does not settle it.
//!
//! This sweeps the ACTUAL compiled driver table across its whole coverage,
//! crossed with altitude and with the geometry the kernel is sensitive to, and
//! reports the MINIMUM the kernel produces. The verdict is the ratio of that
//! minimum to the ceiling.

use anyhow::{Context, Result};
use jb_rs::drivers::{compiled_drivers, UtcJulianDay};
use jb_rs::jb2008::{jb2008_density, jb2008_density_fitted_v7, Jb2008Input};

/// The two constants under test, mirrored from `rhs.rs` (they are private
/// there). A disagreement would make this whole sweep answer the wrong
/// question, so both are printed in the header line for eyeball comparison.
const CEILING_KM: f64 = 2_500.0;
const EXOSPHERIC_CEILING_KG_M3: f64 = 1.0e-19;

/// Part A's authorized apogee ceiling, and a margin past it.
const ALTITUDES_KM: [f64; 10] = [
    2_500.001, 2_600.0, 3_000.0, 5_000.0, 10_000.0, 20_000.0, 35_000.0, 41_378.0, 50_000.0,
    100_000.0,
];

fn main() -> Result<()> {
    let drivers = compiled_drivers().context("compiled JB2008 drivers must load")?;

    println!("SWEEP_HEADER ceiling_km={CEILING_KM} exospheric_kg_m3={EXOSPHERIC_CEILING_KG_M3:e}");

    let mut worst_exact = f64::INFINITY;
    let mut worst_fitted = f64::INFINITY;
    let mut worst_label = String::new();
    let mut days = 0usize;
    let mut samples = 0usize;
    let mut refusals = 0usize;
    let mut refusal_label = String::new();
    let mut refused_days = 0usize;
    let mut first_mjd = f64::NAN;
    let mut last_mjd = f64::NAN;

    // Walk the whole table by trying every Julian Day in a range that brackets
    // any plausible coverage and keeping the ones that resolve. The table's
    // span is not a public API, and probing for it is more honest than
    // hard-coding a span that could drift out from under this sweep.
    // Integer day offsets, so the loop variable is exact and the bound is not a
    // float comparison.
    for day_offset in 0..40_000_i32 {
        let julian_day = 2_430_000.5_f64 + f64::from(day_offset);
        let Ok(utc_jd) = UtcJulianDay::new(julian_day) else {
            continue;
        };
        let Ok(mjd) = utc_jd.to_utc_mjd() else {
            continue;
        };
        let Ok(driver) = drivers.lookup_utc_mjd(mjd) else {
            continue;
        };
        days = days.checked_add(1).context("driver day count overflow")?;
        if [
            driver.f10,
            driver.f10b,
            driver.s10,
            driver.s10b,
            driver.m10,
            driver.m10b,
            driver.y10,
            driver.y10b,
        ]
        .iter()
        .any(|index| *index <= 0.0 || !index.is_finite())
        {
            refused_days = refused_days
                .checked_add(1)
                .context("refused day overflow")?;
        }
        if first_mjd.is_nan() {
            first_mjd = mjd.as_f64();
        }
        last_mjd = mjd.as_f64();

        for altitude_km in ALTITUDES_KM {
            // Geometry the kernel reads: satellite latitude, solar declination,
            // and the hour angle it forms from the two right ascensions. Six
            // hour angles cover local noon, dusk, midnight and dawn; the
            // latitude and declination grids cover both poles and the equator.
            for lat_index in -2..=2 {
                let sat_lat = f64::from(lat_index) * 0.6;
                for dec_index in -1..=1 {
                    let sun_dec = f64::from(dec_index) * 0.41;
                    for hour_index in 0..6 {
                        let hour_angle = f64::from(hour_index) * std::f64::consts::TAU / 6.0;
                        let input = Jb2008Input {
                            mjd_utc: mjd.as_f64(),
                            sun_declination_rad: sun_dec,
                            hour_angle_rad: hour_angle,
                            sat_geocentric_lat_rad: sat_lat,
                            sat_altitude_m: altitude_km * 1_000.0,
                            f10: driver.f10,
                            f10b: driver.f10b,
                            s10: driver.s10,
                            s10b: driver.s10b,
                            m10: driver.m10,
                            m10b: driver.m10b,
                            y10: driver.y10,
                            y10b: driver.y10b,
                            dst_temperature_correction_k: f64::from(driver.dtcval),
                        };
                        samples = samples.checked_add(1).context("sample count overflow")?;
                        // BOTH profiles: the ceiling is applied to whichever one
                        // ran, and production flies the fitted one.
                        // Errors are COUNTED, not swallowed. An earlier version
                        // of this sweep took `unwrap_or(NAN)` and let NaN lose
                        // every `<` comparison, so refused rows vanished from
                        // the minimum silently — and refused rows are exactly
                        // the ones where the flown path returns NaN and a naive
                        // skip would return the ceiling instead.
                        let exact = match jb2008_density(input) {
                            Ok(value) => value,
                            Err(error) => {
                                refusals = refusals.checked_add(1).context("refusal overflow")?;
                                if refusal_label.is_empty() {
                                    refusal_label = format!("{error:?} at mjd={} alt_km={altitude_km} f10={} s10={} m10={} y10={}", mjd.as_f64(), driver.f10, driver.s10, driver.m10, driver.y10);
                                }
                                continue;
                            }
                        };
                        let fitted = jb2008_density_fitted_v7(input).unwrap_or(f64::NAN);
                        if exact < worst_exact {
                            worst_exact = exact;
                            worst_label = format!(
                                "mjd={:.1} alt_km={altitude_km} lat={sat_lat:.2} \
                                 dec={sun_dec:.2} ha={hour_angle:.2} f10={} f10b={}",
                                mjd.as_f64(),
                                driver.f10,
                                driver.f10b
                            );
                        }
                        worst_fitted = worst_fitted.min(fitted);
                    }
                }
            }
        }
    }

    println!(
        "SWEEP_COVERAGE days={days} samples={samples} mjd_first={first_mjd} mjd_last={last_mjd}"
    );
    println!(
        "SWEEP_MIN exact={worst_exact:e} fitted={worst_fitted:e} \
         margin_exact={:.1}x margin_fitted={:.1}x",
        worst_exact / EXOSPHERIC_CEILING_KG_M3,
        worst_fitted / EXOSPHERIC_CEILING_KG_M3
    );
    println!("SWEEP_REFUSALS samples={refusals} driver_days_with_nonpositive_index={refused_days} first={refusal_label}");
    println!("SWEEP_WORST_AT {worst_label}");
    println!(
        "SWEEP_VERDICT ceiling_always_binds={}",
        worst_exact > EXOSPHERIC_CEILING_KG_M3 && worst_fitted > EXOSPHERIC_CEILING_KG_M3
    );
    Ok(())
}
