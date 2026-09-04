use jb_rs::drivers::{Jb2008Drivers, UtcJulianDay, UtcModifiedJulianDay};
use num_traits::ToPrimitive;
use std::path::Path;

const LICENSE_FIXTURE: &[u8] = b"synthetic SET license acknowledgement";

fn append_format(text: &mut String, arguments: std::fmt::Arguments<'_>) {
    use std::fmt::Write as _;

    assert!(
        text.write_fmt(arguments).is_ok(),
        "formatting a String must work"
    );
}

fn solfsmy_fixture() -> String {
    let mut text = String::from(
        "# F10, S10, M10, Y10 data release synthetic-fixture-v1\n\
# YYYY DDD JulianDay F10 F81c S10 S81c M10 M81c Y10 Y81c Ssrc\n",
    );
    for day in 0..8 {
        let day_of_year = 310_i32.saturating_add(day);
        let julian_day = 2_459_890_i32.saturating_add(day);
        append_format(
            &mut text,
            format_args!(
                "2022 {} {} {} {} {} {} {} {} {} {} TEST\n",
                day_of_year,
                julian_day,
                100_i32.saturating_add(day),
                200_i32.saturating_add(day),
                300_i32.saturating_add(day),
                400_i32.saturating_add(day),
                500_i32.saturating_add(day),
                600_i32.saturating_add(day),
                700_i32.saturating_add(day),
                800_i32.saturating_add(day),
            ),
        );
    }
    text
}

fn dtcfile_fixture() -> String {
    let mut text = String::new();
    for day in 0..8 {
        append_format(
            &mut text,
            format_args!("DTC 2022 {}", 310_i32.saturating_add(day)),
        );
        for hour in 0..24 {
            let dtc_value = day.saturating_mul(100).saturating_add(hour);
            append_format(&mut text, format_args!(" {dtc_value}"));
        }
        text.push('\n');
    }
    text
}

fn dtcfile_with_extreme_adjacent_values() -> String {
    let mut text = String::new();
    for day in 0..8 {
        append_format(
            &mut text,
            format_args!("DTC 2022 {}", 310_i32.saturating_add(day)),
        );
        for hour in 0..24 {
            let value = match (day, hour) {
                (6, 18) => i32::MIN,
                (6, 19) => i32::MAX,
                _ => 0,
            };
            append_format(&mut text, format_args!(" {value}"));
        }
        text.push('\n');
    }
    text
}

fn manifest_fixture(solfsmy: &str, dtcfile: &str) -> String {
    format!(
        r#"{{
  "schema": "jb2008_offline_data_authority_v1",
  "source": {{"immutability_policy": "verbatim_download_sha256_and_size_locked"}},
  "required_catalogue_coverage": {{
    "start_utc_date": "2021-11-09",
    "end_utc_date": "2022-11-11",
    "max_input_lag_days": 5,
    "effective_driver_start_utc_date": "2021-11-04"
  }},
  "files": {{
    "SOLFSMY.TXT": {{
      "size_bytes": {},
      "sha256": "8f24d9e2e56265f54807abb809f0300e5181ba84192f34aff34369957119dd72",
      "release_header": "F10, S10, M10, Y10 data release synthetic-fixture-v1",
      "source_declared_record_count": 11,
      "parsed_coverage": {{"first_utc_date": "2022-11-06", "last_utc_date": "2022-11-13", "record_count": 8}}
    }},
    "DTCFILE.TXT": {{
      "size_bytes": {},
      "sha256": "30611d72c249fda1333505bc4cfa77dd81502de8aa90b4a57524dec3b7d1f3c3",
      "parsed_coverage": {{"first_utc_date": "2022-11-06", "last_utc_date": "2022-11-13", "record_count": 8}}
    }}
  }},
  "license": {{
    "acknowledged": true,
    "local_file": "License.html",
    "source_url": "https://sol.spacenvironment.net/JB2008/License.html",
    "size_bytes": 37,
    "sha256": "7c8132aca5fcd139530b1a7373efa41b86b7fe7ba7e615a23e12f6b5096941b3"
  }}
}}"#,
        solfsmy.len(),
        dtcfile.len()
    )
}

#[test]
fn approved_manifest_binds_bytes_and_uses_set_dtcval_rounding() {
    let solfsmy = solfsmy_fixture();
    let dtcfile = dtcfile_fixture();
    let raw = Jb2008Drivers::from_set_bytes(solfsmy.as_bytes(), dtcfile.as_bytes()).unwrap();
    assert_eq!(
        raw.identity().solfsmy_sha256_hex(),
        "8f24d9e2e56265f54807abb809f0300e5181ba84192f34aff34369957119dd72"
    );
    assert_eq!(
        raw.identity().dtcfile_sha256_hex(),
        "30611d72c249fda1333505bc4cfa77dd81502de8aa90b4a57524dec3b7d1f3c3"
    );
    let manifest = manifest_fixture(&solfsmy, &dtcfile);
    let drivers = raw;

    let utc_jd = UtcJulianDay::new(2_459_896.270_833_333_5).unwrap();
    let input = drivers.lookup_utc_jd(utc_jd).unwrap();
    assert_eq!(input.dtcval, 619);
    assert_eq!(drivers.identity().source_declared_record_count, 0);
    assert_eq!(drivers.identity().solfsmy_parsed_record_count, 8);
    assert!(!drivers.identity().license_acknowledged);
    assert!(Jb2008Drivers::from_approved_set_bytes(
        solfsmy.as_bytes(),
        dtcfile.as_bytes(),
        manifest.as_bytes(),
        LICENSE_FIXTURE,
    )
    .is_err());

    let mjd = UtcModifiedJulianDay::new(59_895.770_833_333_5).unwrap();
    assert_eq!(
        mjd.to_utc_jd().unwrap().as_f64().to_bits(),
        utc_jd.as_f64().to_bits()
    );
}

#[test]
fn final_dtc_date_needs_next_row_only_for_hour_23_interpolation() {
    let solfsmy = solfsmy_fixture();
    let dtcfile = dtcfile_fixture();
    let drivers = Jb2008Drivers::from_set_bytes(solfsmy.as_bytes(), dtcfile.as_bytes()).unwrap();

    assert!(drivers
        .lookup_utc_jd(UtcJulianDay::new(2_459_897.437_5).unwrap())
        .is_ok());
    assert!(drivers
        .lookup_utc_jd(UtcJulianDay::new(2_459_897.479_166_666_5).unwrap())
        .is_err());
}

#[test]
fn dtc_interpolation_promotes_extremes_before_subtraction() {
    let solfsmy = solfsmy_fixture();
    let dtcfile = dtcfile_with_extreme_adjacent_values();
    let drivers = Jb2008Drivers::from_set_bytes(solfsmy.as_bytes(), dtcfile.as_bytes()).unwrap();
    let utc_day_fraction = 18.5 / 24.0 - 0.000_000_1;
    let utc_jd = UtcJulianDay::new(2_459_896.0 - 0.5 + utc_day_fraction).unwrap();
    let day_jd = (utc_jd.as_f64() + 0.5).floor();
    let hour =
        ((utc_jd.as_f64() - (day_jd - 0.5) + 0.000_000_1) * 24.0).min(23.999_999_999_999_996);
    let fraction = hour - hour.floor();
    let expected =
        (f64::from(i32::MIN) + fraction * (f64::from(i32::MAX) - f64::from(i32::MIN)) + 0.5)
            .trunc();

    assert!((f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&expected));
    assert_eq!(
        Some(drivers.lookup_utc_jd(utc_jd).unwrap().dtcval),
        expected.to_i32()
    );
}

#[test]
fn validates_gregorian_jd_across_leap_year_rollover() {
    let dates = [
        (2020, 363),
        (2020, 364),
        (2020, 365),
        (2020, 366),
        (2021, 1),
        (2021, 2),
        (2021, 3),
        (2021, 4),
    ];
    let mut solfsmy = String::from("# F10, S10, M10, Y10 data release rollover-test\n");
    let mut dtcfile = String::new();
    for (offset, (year, doy)) in dates.into_iter().enumerate() {
        append_format(
            &mut solfsmy,
            format_args!(
                "{year} {doy} {} 100 200 300 400 500 600 700 800 TEST\n",
                2_459_212_usize.saturating_add(offset)
            ),
        );
        append_format(&mut dtcfile, format_args!("DTC {year} {doy}"));
        for hour in 0..24 {
            append_format(&mut dtcfile, format_args!(" {hour}"));
        }
        dtcfile.push('\n');
    }
    assert!(Jb2008Drivers::from_set_bytes(solfsmy.as_bytes(), dtcfile.as_bytes()).is_ok());
}

#[test]
fn approved_catalogue_oracle_over_the_compiled_data() {
    // Until 2026-08-08 this test looked for the catalogue five directories
    // ABOVE the crate (a pre-integration layout that no checkout has), and a
    // bare `return` skip guard made it report green while running nothing --
    // in every tree, since at least the worktree era. The same four files have
    // lived in-repo at `crates/jb_rs/data/jb2008/` (hashed into `jb_sha256`)
    // the whole time; the expected manifest/license SHA-256 values below match
    // them byte-for-byte, so pointing here revived every assertion unchanged.
    // No skip guard: if the data is missing the test must FAIL, not vanish.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/jb2008");
    let manifest_path = root.join("manifest.json");
    let solfsmy = std::fs::read(root.join("SOLFSMY.TXT")).unwrap();
    let dtcfile = std::fs::read(root.join("DTCFILE.TXT")).unwrap();
    let manifest = std::fs::read(manifest_path).unwrap();
    let license = std::fs::read(root.join("License.html")).unwrap();
    let drivers =
        Jb2008Drivers::from_approved_set_bytes(&solfsmy, &dtcfile, &manifest, &license).unwrap();
    let input = drivers
        .lookup_utc_jd(UtcJulianDay::new(2_459_528.0).unwrap())
        .unwrap();
    assert_eq!(
        (input.f10, input.f10b, input.s10, input.s10b),
        (88.0, 88.3, 80.2, 79.2)
    );
    assert_eq!(
        (input.m10, input.m10b, input.y10, input.y10b),
        (102.3, 98.0, 112.3, 99.5)
    );
    assert_eq!(input.dtcval, 50);
    assert_eq!(drivers.identity().source_declared_record_count, 10_926);
    assert_eq!(drivers.identity().solfsmy_parsed_record_count, 10_746);
    assert_eq!(
        drivers.identity().manifest_sha256_hex(),
        "09d2cb0cdb5cb805f6b10e9e2141e02d9320c91071491e2b32da87857deac01c"
    );
    assert_eq!(
        drivers.identity().license_sha256_hex(),
        "9d2ec826044266557880c7863443ac2609312db39780b1c8239f5578dd75d387"
    );

    for mut corrupted in [
        solfsmy.clone(),
        dtcfile.clone(),
        manifest.clone(),
        license.clone(),
    ] {
        if let Some(first_byte) = corrupted.first_mut() {
            *first_byte ^= 1;
        }
        let supplied = if corrupted.len() == solfsmy.len() {
            (&corrupted[..], &dtcfile[..], &manifest[..], &license[..])
        } else if corrupted.len() == dtcfile.len() {
            (&solfsmy[..], &corrupted[..], &manifest[..], &license[..])
        } else if corrupted.len() == manifest.len() {
            (&solfsmy[..], &dtcfile[..], &corrupted[..], &license[..])
        } else {
            (&solfsmy[..], &dtcfile[..], &manifest[..], &corrupted[..])
        };
        assert!(Jb2008Drivers::from_approved_set_bytes(
            supplied.0, supplied.1, supplied.2, supplied.3
        )
        .is_err());
    }
}

#[test]
fn set_dtcval_bias_fixes_half_hour_rounding_without_day_spill() {
    let mut solfsmy = String::from("# F10, S10, M10, Y10 data release dtc-bias-test\n");
    let mut dtcfile = String::new();
    for offset in 0..9 {
        append_format(
            &mut solfsmy,
            format_args!(
                "2022 {} {} 100 100 100 100 100 100 100 100 TEST\n",
                306_i32.saturating_add(offset),
                2_459_886_i32.saturating_add(offset)
            ),
        );
        append_format(
            &mut dtcfile,
            format_args!("DTC 2022 {}", 306_i32.saturating_add(offset)),
        );
        for hour in 0..24 {
            let value = i32::from(offset == 7 && hour == 3);
            append_format(&mut dtcfile, format_args!(" {value}"));
        }
        dtcfile.push('\n');
    }
    let drivers = Jb2008Drivers::from_set_bytes(solfsmy.as_bytes(), dtcfile.as_bytes()).unwrap();

    let half_hour = drivers
        .lookup_utc_jd(UtcJulianDay::new(2_459_892.604_166_666_5).unwrap())
        .unwrap();
    assert_eq!(half_hour.dtcval, 1);

    assert!(drivers
        .lookup_utc_jd(UtcJulianDay::new(2_459_893.499_999_999_5).unwrap())
        .is_ok());
    assert!(drivers
        .lookup_utc_jd(UtcJulianDay::new(2_459_893.5).unwrap())
        .is_ok());
}

#[test]
fn lookup_applies_documented_lags_and_hourly_dtc_interpolation() {
    let solfsmy = solfsmy_fixture();
    let dtcfile = dtcfile_fixture();
    let drivers = Jb2008Drivers::from_set_bytes(solfsmy.as_bytes(), dtcfile.as_bytes()).unwrap();

    // JD 2459896.270833... is calendar day 2459896, 18:30 UTC.
    let input = drivers
        .lookup_utc_jd(UtcJulianDay::new(2_459_896.270_833_333_5).unwrap())
        .unwrap();
    assert!(drivers
        .lookup_utc_jd(UtcJulianDay::new(2_459_894.5).unwrap())
        .is_ok());
    assert!(drivers
        .lookup_utc_jd(UtcJulianDay::new(2_459_893.5).unwrap())
        .is_err());

    // F10/S10: 1 d lag; M10: 2 d; Y10: 5 d.
    assert_eq!(input.f10.to_bits(), 105.0_f64.to_bits());
    assert_eq!(input.f10b.to_bits(), 205.0_f64.to_bits());
    assert_eq!(input.s10.to_bits(), 305.0_f64.to_bits());
    assert_eq!(input.s10b.to_bits(), 405.0_f64.to_bits());
    assert_eq!(input.m10.to_bits(), 504.0_f64.to_bits());
    assert_eq!(input.m10b.to_bits(), 604.0_f64.to_bits());
    assert_eq!(input.y10.to_bits(), 701.0_f64.to_bits());
    assert_eq!(input.y10b.to_bits(), 801.0_f64.to_bits());
    assert_eq!(input.dtcval, 619);

    let identity = drivers.identity();
    assert_eq!(
        identity.solfsmy_release_header,
        "# F10, S10, M10, Y10 data release synthetic-fixture-v1"
    );
    assert_eq!(
        identity.solfsmy_coverage_start_jd.to_bits(),
        2_459_890.0_f64.to_bits()
    );
    assert_eq!(
        identity.solfsmy_coverage_end_jd.to_bits(),
        2_459_897.0_f64.to_bits()
    );
    assert_eq!(
        identity.dtc_coverage_start_jd.to_bits(),
        2_459_890.0_f64.to_bits()
    );
    assert_eq!(
        identity.dtc_coverage_end_jd.to_bits(),
        2_459_897.0_f64.to_bits()
    );
    assert_eq!(
        identity.solfsmy_sha256_hex(),
        "8f24d9e2e56265f54807abb809f0300e5181ba84192f34aff34369957119dd72"
    );
    assert_eq!(
        identity.dtcfile_sha256_hex(),
        "30611d72c249fda1333505bc4cfa77dd81502de8aa90b4a57524dec3b7d1f3c3"
    );
}

#[test]
fn rejects_missing_nonfinite_duplicate_and_out_of_range_drivers() {
    let solfsmy = solfsmy_fixture();
    let dtcfile = dtcfile_fixture();

    assert!(Jb2008Drivers::from_set_bytes(b"", dtcfile.as_bytes()).is_err());
    assert!(Jb2008Drivers::from_set_bytes(
        solfsmy.replacen("100 200", "NaN 200", 1).as_bytes(),
        dtcfile.as_bytes()
    )
    .is_err());

    let duplicate_dtcfile = format!("{dtcfile}{}\n", dtcfile.lines().next().unwrap());
    assert!(
        Jb2008Drivers::from_set_bytes(solfsmy.as_bytes(), duplicate_dtcfile.as_bytes()).is_err()
    );

    let missing_dtcfile = dtcfile
        .lines()
        .filter(|line| !line.starts_with("DTC 2022 314 "))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(Jb2008Drivers::from_set_bytes(solfsmy.as_bytes(), missing_dtcfile.as_bytes()).is_err());

    let drivers = Jb2008Drivers::from_set_bytes(solfsmy.as_bytes(), dtcfile.as_bytes()).unwrap();
    assert!(UtcJulianDay::new(f64::NAN).is_err());
    assert!(drivers
        .lookup_utc_jd(UtcJulianDay::new(2_459_999.0).unwrap())
        .is_err());
}
