//! Bakes the tracked SET driver catalogues into static row tables.
//!
//! `drivers.rs` used to text-parse `SOLFSMY.TXT` + `DTCFILE.TXT` (~21.5k
//! lines: split, f64-parse, per-line calendar validation, contiguity checks)
//! at first `compiled_drivers()` call in every process. This script parses the
//! same tracked bytes once per build with a MIRROR of the runtime parser and
//! emits the validated rows as static tables; parse or validation failures
//! become BUILD failures for the compiled set. The runtime SHA-256 trust root
//! (manifest + license digests against the approved constants, and the
//! catalogue digests bound into the identity) is KEPT verbatim in
//! `drivers.rs` — this script moves parsing, not trust.
//!
//! # Bit contract
//!
//! Numeric fields are decimal text parsed by `str::parse::<f64>` — correctly
//! rounded, so the same text yields the same bits on any host — and emitted as
//! `from_bits` literals. Calendar/ordinal math is integer. The
//! `baked_rows_are_bit_identical_to_the_parser` oracle test in `drivers.rs`
//! re-runs the kept runtime parser (`from_set_bytes`) over the same bytes and
//! bit-compares every baked row, so drift between the mirrored parsers is a
//! red test, not a silent skew.

use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

type BuildResult<T> = Result<T, Box<dyn Error>>;

fn fail(message: String) -> Box<dyn Error> {
    message.into()
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct DateKey {
    year: i32,
    doy: i32,
    ordinal: i64,
}

struct SolarRow {
    date: DateKey,
    jd: f64,
    fields: [f64; 8],
    source: String,
}

struct DtcRow {
    date: DateKey,
    values: [i32; 24],
}

const fn days_in_year(year: i32) -> i32 {
    if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
        366
    } else {
        365
    }
}

fn parse_date(year: &str, doy: &str, source: &str, line_number: usize) -> BuildResult<DateKey> {
    let year = year
        .parse::<i32>()
        .map_err(|_| fail(format!("{source} line {line_number} has invalid year")))?;
    let doy = doy.parse::<i32>().map_err(|_| {
        fail(format!(
            "{source} line {line_number} has invalid day of year"
        ))
    })?;
    if year < 1 || !(1..=days_in_year(year)).contains(&doy) {
        return Err(fail(format!(
            "{source} line {line_number} has invalid calendar date"
        )));
    }
    let completed_years = i64::from(year)
        .checked_sub(1)
        .ok_or_else(|| fail("calendar year underflows".to_owned()))?;
    let completed_days = completed_years
        .checked_mul(365)
        .and_then(|value| value.checked_add(completed_years / 4))
        .and_then(|value| value.checked_sub(completed_years / 100))
        .and_then(|value| value.checked_add(completed_years / 400))
        .and_then(|value| value.checked_add(i64::from(doy).checked_sub(1)?))
        .ok_or_else(|| fail("calendar ordinal overflows".to_owned()))?;
    Ok(DateKey {
        year,
        doy,
        ordinal: completed_days,
    })
}

fn month_day_from_doy(year: i32, doy: i32) -> BuildResult<(i32, i32)> {
    if year < 1 || !(1..=days_in_year(year)).contains(&doy) {
        return Err(fail("calendar date is outside Gregorian range".to_owned()));
    }
    let month_lengths = [
        31,
        if days_in_year(year) == 366 { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut remaining = doy;
    for (index, length) in month_lengths.into_iter().enumerate() {
        if remaining <= length {
            let month = i32::try_from(index)
                .ok()
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| fail("Gregorian month index overflows".to_owned()))?;
            return Ok((month, remaining));
        }
        remaining = remaining
            .checked_sub(length)
            .ok_or_else(|| fail("Gregorian day-of-year underflows".to_owned()))?;
    }
    Err(fail(
        "validated Gregorian day-of-year has no month".to_owned(),
    ))
}

fn exact_i64_as_f64(value: i64) -> BuildResult<f64> {
    const MAX_EXACT_I64: i64 = 9_007_199_254_740_992;
    if !(-MAX_EXACT_I64..=MAX_EXACT_I64).contains(&value) {
        return Err(fail("Julian day exceeds exact f64 range".to_owned()));
    }
    let truncated = i32::try_from(value)
        .map_err(|_| fail("Julian day exceeds exact i32-backed range".to_owned()))?;
    Ok(f64::from(truncated))
}

/// MIRROR of `drivers::gregorian_utc_noon_jd`; the oracle bit-compares.
fn gregorian_utc_noon_jd(date: DateKey) -> BuildResult<f64> {
    let (month, day) = month_day_from_doy(date.year, date.doy)?;
    let month = i64::from(month);
    let day = i64::from(day);
    let adjustment = 14_i64
        .checked_sub(month)
        .ok_or_else(|| fail("Gregorian month adjustment underflows".to_owned()))?
        / 12;
    let year = i64::from(date.year)
        .checked_add(4_800)
        .and_then(|value| value.checked_sub(adjustment))
        .ok_or_else(|| fail("Gregorian year overflows".to_owned()))?;
    let shifted_month = month
        .checked_add(
            12_i64
                .checked_mul(adjustment)
                .ok_or_else(|| fail("Gregorian month adjustment overflows".to_owned()))?,
        )
        .and_then(|value| value.checked_sub(3))
        .ok_or_else(|| fail("Gregorian shifted month overflows".to_owned()))?;
    let month_term = 153_i64
        .checked_mul(shifted_month)
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| fail("Gregorian month term overflows".to_owned()))?
        / 5;
    let year_days = 365_i64
        .checked_mul(year)
        .ok_or_else(|| fail("Gregorian year-day term overflows".to_owned()))?;
    let jdn = day
        .checked_add(month_term)
        .and_then(|value| value.checked_add(year_days))
        .and_then(|value| value.checked_add(year / 4))
        .and_then(|value| value.checked_sub(year / 100))
        .and_then(|value| value.checked_add(year / 400))
        .and_then(|value| value.checked_sub(32_045))
        .ok_or_else(|| fail("Gregorian Julian day overflows".to_owned()))?;
    exact_i64_as_f64(jdn)
}

fn parse_finite_f64(value: &str, field: &str, line_number: usize) -> BuildResult<f64> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| fail(format!("{field} line {line_number} is not numeric")))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(fail(format!("{field} line {line_number} is non-finite")))
    }
}

/// MIRROR of `drivers::parse_solfsmy` including its validation.
fn parse_solfsmy(input: &str) -> BuildResult<(Vec<SolarRow>, String)> {
    let mut release_header: Option<String> = None;
    let mut rows: Vec<SolarRow> = Vec::new();
    for (line_index, raw_line) in input.lines().enumerate() {
        let line_number = line_index
            .checked_add(1)
            .ok_or_else(|| fail("catalogue line count overflows".to_owned()))?;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            if release_header.is_none() {
                release_header = Some(line.to_owned());
            }
            continue;
        }
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        let [year, doy, jd_text, f10, f10b, s10, s10b, m10, m10b, y10, y10b, source] =
            fields.as_slice()
        else {
            return Err(fail(format!(
                "SOLFSMY line {line_number} has {} fields; expected 12",
                fields.len()
            )));
        };
        let date = parse_date(year, doy, "SOLFSMY", line_number)?;
        let jd = parse_finite_f64(jd_text, "SOLFSMY Julian day", line_number)?;
        if !matches!(jd.fract().to_bits(), 0 | 0x8000_0000_0000_0000) {
            return Err(fail(format!(
                "SOLFSMY line {line_number} Julian day is not an integer noon key"
            )));
        }
        let numeric = [
            parse_finite_f64(f10, "SOLFSMY F10", line_number)?,
            parse_finite_f64(f10b, "SOLFSMY F81c", line_number)?,
            parse_finite_f64(s10, "SOLFSMY S10", line_number)?,
            parse_finite_f64(s10b, "SOLFSMY S81c", line_number)?,
            parse_finite_f64(m10, "SOLFSMY M10", line_number)?,
            parse_finite_f64(m10b, "SOLFSMY M81c", line_number)?,
            parse_finite_f64(y10, "SOLFSMY Y10", line_number)?,
            parse_finite_f64(y10b, "SOLFSMY Y81c", line_number)?,
        ];
        if source.is_empty() {
            return Err(fail(format!(
                "SOLFSMY line {line_number} has empty source field"
            )));
        }
        rows.push(SolarRow {
            date,
            jd,
            fields: numeric,
            source: (*source).to_owned(),
        });
    }
    let release_header =
        release_header.ok_or_else(|| fail("SOLFSMY release header missing".to_owned()))?;
    if !release_header.starts_with("# F10, S10, M10, Y10 data release") {
        return Err(fail("SOLFSMY release header is unrecognized".to_owned()));
    }
    // Validation mirror of `drivers::validate_solar_rows`.
    let first = rows
        .first()
        .ok_or_else(|| fail("SOLFSMY has no data rows".to_owned()))?;
    if !first.jd.is_finite() {
        return Err(fail("SOLFSMY first Julian day is non-finite".to_owned()));
    }
    for row in &rows {
        let expected_jd = gregorian_utc_noon_jd(row.date)?;
        if row.jd.to_bits() != expected_jd.to_bits() {
            return Err(fail(
                "SOLFSMY Gregorian UTC date and Julian day disagree".to_owned(),
            ));
        }
    }
    for pair in rows.windows(2) {
        let [previous, next] = pair else {
            continue;
        };
        let expected_jd = previous.jd + 1.0;
        let expected_ordinal = previous
            .date
            .ordinal
            .checked_add(1)
            .ok_or_else(|| fail("SOLFSMY calendar ordinal overflows".to_owned()))?;
        if next.jd.to_bits() != expected_jd.to_bits() || next.date.ordinal != expected_ordinal {
            return Err(fail(
                "SOLFSMY rows are missing, duplicated, or out of order".to_owned(),
            ));
        }
    }
    Ok((rows, release_header))
}

/// MIRROR of `drivers::parse_dtcfile` including its validation.
fn parse_dtcfile(input: &str) -> BuildResult<Vec<DtcRow>> {
    let mut rows: Vec<DtcRow> = Vec::new();
    for (line_index, raw_line) in input.lines().enumerate() {
        let line_number = line_index
            .checked_add(1)
            .ok_or_else(|| fail("catalogue line count overflows".to_owned()))?;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        let Some((marker, calendar_and_values)) = fields.split_first() else {
            return Err(fail(format!(
                "DTCFILE line {line_number} must contain DTC plus 26 fields"
            )));
        };
        let Some((year, day_and_values)) = calendar_and_values.split_first() else {
            return Err(fail(format!(
                "DTCFILE line {line_number} must contain DTC plus 26 fields"
            )));
        };
        let Some((doy, value_fields)) = day_and_values.split_first() else {
            return Err(fail(format!(
                "DTCFILE line {line_number} must contain DTC plus 26 fields"
            )));
        };
        if *marker != "DTC" || value_fields.len() != 24 {
            return Err(fail(format!(
                "DTCFILE line {line_number} must contain DTC plus 26 fields"
            )));
        }
        let date = parse_date(year, doy, "DTCFILE", line_number)?;
        let values: Vec<i32> = value_fields
            .iter()
            .map(|value| {
                value
                    .parse::<i32>()
                    .map_err(|_| fail(format!("DTCFILE dTc line {line_number} is not an integer")))
            })
            .collect::<BuildResult<_>>()?;
        let values: [i32; 24] = values
            .try_into()
            .map_err(|_| fail("DTCFILE has an invalid hourly-value count".to_owned()))?;
        rows.push(DtcRow { date, values });
    }
    // Validation mirror of `drivers::validate_dtc_rows`.
    rows.first()
        .ok_or_else(|| fail("DTCFILE has no data rows".to_owned()))?;
    for pair in rows.windows(2) {
        let [previous, next] = pair else {
            continue;
        };
        let expected_ordinal = previous
            .date
            .ordinal
            .checked_add(1)
            .ok_or_else(|| fail("DTCFILE calendar ordinal overflows".to_owned()))?;
        if next.date.ordinal != expected_ordinal {
            return Err(fail(
                "DTCFILE rows are missing, duplicated, or out of order".to_owned(),
            ));
        }
    }
    Ok(rows)
}

fn grouped_decimal_i64(value: i64) -> String {
    let raw = format!("{}", value.abs());
    let mut reversed = String::new();
    for (index, digit) in raw.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            reversed.push('_');
        }
        reversed.push(digit);
    }
    let grouped: String = reversed.chars().rev().collect();
    if value < 0 {
        format!("-{grouped}")
    } else {
        grouped
    }
}

fn grouped_hex_u64(bits: u64) -> String {
    let raw = format!("{bits:016x}");
    let mut grouped = String::new();
    for (index, digit) in raw.chars().enumerate() {
        if index > 0 && index % 4 == 0 {
            grouped.push('_');
        }
        grouped.push(digit);
    }
    grouped
}

fn f64_from_bits_literal(value: f64) -> String {
    format!("f64::from_bits(0x{})", grouped_hex_u64(value.to_bits()))
}

fn main() -> BuildResult<()> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let data_dir = manifest_dir.join("data/jb2008");
    let solfsmy_path = data_dir.join("SOLFSMY.TXT");
    let dtcfile_path = data_dir.join("DTCFILE.TXT");
    println!("cargo:rerun-if-changed={}", solfsmy_path.display());
    println!("cargo:rerun-if-changed={}", dtcfile_path.display());

    let solfsmy_bytes = fs::read(&solfsmy_path)?;
    let dtcfile_bytes = fs::read(&dtcfile_path)?;
    let solfsmy = std::str::from_utf8(&solfsmy_bytes)
        .map_err(|_| fail("SOLFSMY is not valid UTF-8".to_owned()))?;
    let dtcfile = std::str::from_utf8(&dtcfile_bytes)
        .map_err(|_| fail("DTCFILE is not valid UTF-8".to_owned()))?;

    let (solar_rows, release_header) = parse_solfsmy(solfsmy)?;
    let dtc_rows = parse_dtcfile(dtcfile)?;

    let mut out = String::new();
    out.push_str("// Generated by build.rs from data/jb2008/*.TXT. Do not edit.\n");
    writeln!(
        out,
        "pub(super) const BAKED_SOLFSMY_RELEASE_HEADER: &str = {release_header:?};"
    )?;
    writeln!(
        out,
        "pub(super) static BAKED_SOLAR_ROWS: [BakedSolarRow; {}] = [",
        grouped_decimal_i64(i64::try_from(solar_rows.len())?)
    )?;
    for row in &solar_rows {
        let [f10, f10b, s10, s10b, m10, m10b, y10, y10b] = row.fields;
        writeln!(
            out,
            "    BakedSolarRow {{ year: {}, doy: {}, ordinal: {}, jd: {}, f10: {}, f10b: {}, \
             s10: {}, s10b: {}, m10: {}, m10b: {}, y10: {}, y10b: {}, source: {:?} }},",
            row.date.year,
            row.date.doy,
            grouped_decimal_i64(row.date.ordinal),
            f64_from_bits_literal(row.jd),
            f64_from_bits_literal(f10),
            f64_from_bits_literal(f10b),
            f64_from_bits_literal(s10),
            f64_from_bits_literal(s10b),
            f64_from_bits_literal(m10),
            f64_from_bits_literal(m10b),
            f64_from_bits_literal(y10),
            f64_from_bits_literal(y10b),
            row.source
        )?;
    }
    out.push_str("];\n");
    writeln!(
        out,
        "pub(super) static BAKED_DTC_ROWS: [BakedDtcRow; {}] = [",
        grouped_decimal_i64(i64::try_from(dtc_rows.len())?)
    )?;
    for row in &dtc_rows {
        let values: Vec<String> = row.values.iter().map(|value| format!("{value}")).collect();
        writeln!(
            out,
            "    BakedDtcRow {{ year: {}, doy: {}, ordinal: {}, values: [{}] }},",
            row.date.year,
            row.date.doy,
            grouped_decimal_i64(row.date.ordinal),
            values.join(", ")
        )?;
    }
    out.push_str("];\n");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    fs::write(out_dir.join("jb2008_driver_tables.rs"), out)?;
    Ok(())
}
