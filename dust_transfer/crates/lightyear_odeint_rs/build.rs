//! Bakes the four tracked ephemeris catalogues into static `from_bits` tables.
//!
//! `precomputed_ephem.rs` used to `include_bytes!` the catalogues and, at first
//! load, byte-decode ~1.06 MB into a heap `Vec<f64>`, scan it twice (finiteness
//! and the direction-rate supremum), and SHA-256 the same fixed bytes in two
//! `LazyLock`s — all of it compile-time-constant work re-done in every process
//! and every test binary, plus a ~1 MB heap duplicate of bytes already in
//! rodata. This script does that work once per build: it validates the exact
//! invariants the runtime loader enforces (magic, version, grid/`jd_end` bit
//! agreement, body id, epoch tags, reserved zeros, exact payload size,
//! finiteness) as BUILD failures, and emits per-body static position tables as
//! `f64::from_bits` literals plus header consts, the precomputed rate
//! supremum, and the SHA-256 identities.
//!
//! # Bit contract
//!
//! `f64::from_le_bytes` -> `f64::from_bits` is an exact bit reinterpretation,
//! so the tables are bit-identical to the runtime decode by construction. The
//! rate supremum is FP arithmetic, so its two helpers below are TOKEN COPIES
//! of `PrecomputedEphemeris::{max_normalized_direction_rate_per_day,
//! interval_direction_rate_supremum}` (sqrt and fused `mul_add` are exactly
//! rounded IEEE operations, deterministic across hosts). The
//! `generated_embedded_tables_are_bit_identical_to_the_parser` oracle test in
//! `precomputed_ephem.rs` re-runs the kept runtime parser over the same bytes
//! and bit-compares every emitted value, so a drift in either copy is a red
//! test, not a silent skew.

use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

const MAGIC: &[u8; 4] = b"DUST";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 48;
const F64_BYTES: usize = 8;

type BuildResult<T> = Result<T, Box<dyn Error>>;

fn fail(message: String) -> Box<dyn Error> {
    message.into()
}

struct ParsedCatalogue {
    prefix: &'static str,
    body_id: u8,
    size_bytes: usize,
    n_samples: usize,
    jd_start: f64,
    jd_end: f64,
    dt_days: f64,
    epoch_scale_tag: u8,
    epoch_representation_tag: u8,
    positions: Vec<f64>,
    max_direction_rate_per_day: f64,
    sha256: [u8; 32],
}

fn header_array<const LENGTH: usize>(
    header: &[u8],
    range: std::ops::Range<usize>,
    field: &str,
) -> BuildResult<[u8; LENGTH]> {
    let bytes = header
        .get(range)
        .ok_or_else(|| fail(format!("catalogue header is missing {field}")))?;
    bytes
        .try_into()
        .map_err(|_| fail(format!("catalogue header has malformed {field}")))
}

fn header_byte(header: &[u8], index: usize, field: &str) -> BuildResult<u8> {
    header
        .get(index)
        .copied()
        .ok_or_else(|| fail(format!("catalogue header is missing {field}")))
}

/// TOKEN COPY of `PrecomputedEphemeris::interval_direction_rate_supremum`.
/// Any edit must land in both, and the oracle test bit-compares the results.
fn interval_direction_rate_supremum(before: [f64; 3], after: [f64; 3], dt_days: f64) -> f64 {
    let [before_x, before_y, before_z] = before;
    let [after_x, after_y, after_z] = after;
    let delta = [after_x - before_x, after_y - before_y, after_z - before_z];
    let [delta_x, delta_y, delta_z] = delta;
    let cross = [
        before_y * delta_z - before_z * delta_y,
        before_z * delta_x - before_x * delta_z,
        before_x * delta_y - before_y * delta_x,
    ];
    let cross_norm = cross
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    let delta_sq = delta
        .iter()
        .map(|component| component * component)
        .sum::<f64>();
    let projection = before_x.mul_add(delta_x, before_y.mul_add(delta_y, before_z * delta_z));
    let closest = if delta_sq > 0.0 {
        (-projection / delta_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let nearest = [
        closest.mul_add(delta_x, before_x),
        closest.mul_add(delta_y, before_y),
        closest.mul_add(delta_z, before_z),
    ];
    let nearest_sq = nearest
        .iter()
        .map(|component| component * component)
        .sum::<f64>();
    if !(nearest_sq.is_finite() && nearest_sq > 0.0) {
        return f64::NAN;
    }
    cross_norm / (dt_days * nearest_sq)
}

/// TOKEN COPY of `PrecomputedEphemeris::max_normalized_direction_rate_per_day`.
/// Any edit must land in both, and the oracle test bit-compares the results.
fn max_normalized_direction_rate_per_day(positions: &[f64], n_samples: usize, dt_days: f64) -> f64 {
    let Some(interval_count) = n_samples.checked_sub(1) else {
        return 0.0;
    };
    if interval_count == 0 || !(dt_days.is_finite() && dt_days > 0.0) {
        return 0.0;
    }
    let mut supremum = 0.0_f64;
    for interval in 0..interval_count {
        let (Some(before), Some(after)) = (
            interval
                .checked_mul(3)
                .and_then(|start| positions.get(start..start.checked_add(3)?)),
            interval
                .checked_add(1)
                .and_then(|next| next.checked_mul(3))
                .and_then(|start| positions.get(start..start.checked_add(3)?)),
        ) else {
            return f64::NAN;
        };
        let (Ok(before), Ok(after)): (Result<&[f64; 3], _>, Result<&[f64; 3], _>) =
            (before.try_into(), after.try_into())
        else {
            return f64::NAN;
        };
        supremum = supremum.max(interval_direction_rate_supremum(*before, *after, dt_days));
    }
    supremum
}

fn usize_to_exact_f64(value: usize) -> BuildResult<f64> {
    let converted = u32::try_from(value)
        .ok()
        .map(f64::from)
        .ok_or_else(|| fail(format!("sample count {value} is not exactly representable")))?;
    Ok(converted)
}

fn parse_catalogue(
    prefix: &'static str,
    expected_body_id: u8,
    bytes: &[u8],
) -> BuildResult<ParsedCatalogue> {
    let sha256: [u8; 32] = Sha256::digest(bytes).into();
    let header = bytes
        .get(..HEADER_SIZE)
        .ok_or_else(|| fail(format!("{prefix}: file is smaller than header")))?;
    let data = bytes
        .get(HEADER_SIZE..)
        .ok_or_else(|| fail(format!("{prefix}: file is smaller than header")))?;

    let magic: [u8; 4] = header_array(header, 0..4, "magic")?;
    if magic != *MAGIC {
        return Err(fail(format!("{prefix}: invalid magic {magic:?}")));
    }
    let version = u32::from_le_bytes(header_array(header, 4..8, "version")?);
    if version != VERSION {
        return Err(fail(format!("{prefix}: unsupported version {version}")));
    }
    let n_samples_u64 = u64::from_le_bytes(header_array(header, 8..16, "sample count")?);
    let n_samples = usize::try_from(n_samples_u64)
        .map_err(|_| fail(format!("{prefix}: sample count exceeds usize")))?;
    let jd_start = f64::from_le_bytes(header_array(header, 16..24, "JD start")?);
    let jd_end_header = f64::from_le_bytes(header_array(header, 24..32, "JD end")?);
    let dt_days = f64::from_le_bytes(header_array(header, 32..40, "step")?);
    let body_id = header_byte(header, 40, "body ID")?;

    if n_samples == 0 {
        return Err(fail(format!("{prefix}: catalogue has no samples")));
    }
    if !jd_start.is_finite() || !jd_end_header.is_finite() || !dt_days.is_finite() || dt_days <= 0.0
    {
        return Err(fail(format!(
            "{prefix}: requires finite JD start and positive finite step"
        )));
    }
    let interval_count = n_samples
        .checked_sub(1)
        .ok_or_else(|| fail(format!("{prefix}: catalogue has no samples")))?;
    let interval_count = usize_to_exact_f64(interval_count)?;
    // Keep producer multiply-then-add rounding: bit-compared against the
    // sealed catalogue header, exactly as the runtime loader does.
    let computed_jd_end = jd_start + interval_count * dt_days;
    if jd_end_header.to_bits() != computed_jd_end.to_bits() {
        return Err(fail(format!(
            "{prefix}: declared JD end disagrees with sample grid"
        )));
    }
    if body_id != expected_body_id {
        return Err(fail(format!(
            "{prefix}: body_id {body_id} disagrees with expected {expected_body_id}"
        )));
    }
    let epoch_scale_tag = header_byte(header, 41, "epoch scale tag")?;
    if epoch_scale_tag > 0x04 {
        return Err(fail(format!(
            "{prefix}: unknown epoch scale tag {epoch_scale_tag:#04x}"
        )));
    }
    let epoch_representation_tag = header_byte(header, 42, "epoch representation tag")?;
    if epoch_representation_tag > 0x02 {
        return Err(fail(format!(
            "{prefix}: unknown epoch representation tag {epoch_representation_tag:#04x}"
        )));
    }
    let reserved = header
        .get(43..HEADER_SIZE)
        .ok_or_else(|| fail(format!("{prefix}: header is missing reserved bytes")))?;
    if reserved.iter().any(|byte| *byte != 0) {
        return Err(fail(format!(
            "{prefix}: reserved header bytes must be zero"
        )));
    }
    let expected_len = n_samples
        .checked_mul(3)
        .and_then(|count| count.checked_mul(F64_BYTES))
        .ok_or_else(|| fail(format!("{prefix}: sample count overflows byte length")))?;
    if data.len() != expected_len {
        return Err(fail(format!(
            "{prefix}: file size does not match declared payload"
        )));
    }

    let positions: Vec<f64> = data
        .chunks_exact(F64_BYTES)
        .map(|chunk| {
            let bytes: [u8; F64_BYTES] = chunk
                .try_into()
                .map_err(|_| fail(format!("{prefix}: incomplete position value")))?;
            Ok(f64::from_le_bytes(bytes))
        })
        .collect::<BuildResult<_>>()?;
    if positions.iter().any(|value| !value.is_finite()) {
        return Err(fail(format!("{prefix}: positions must be finite")));
    }

    let max_direction_rate_per_day =
        max_normalized_direction_rate_per_day(&positions, n_samples, dt_days);

    Ok(ParsedCatalogue {
        prefix,
        body_id,
        size_bytes: bytes.len(),
        n_samples,
        jd_start,
        jd_end: jd_end_header,
        dt_days,
        epoch_scale_tag,
        epoch_representation_tag,
        positions,
        max_direction_rate_per_day,
        sha256,
    })
}

fn grouped_decimal(value: usize) -> String {
    let raw = format!("{value}");
    let mut reversed = String::new();
    for (index, digit) in raw.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            reversed.push('_');
        }
        reversed.push(digit);
    }
    reversed.chars().rev().collect()
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

fn lowercase_hex(bytes: &[u8]) -> String {
    let mut encoded = String::new();
    for byte in bytes {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn write_catalogue_items(out: &mut String, parsed: &ParsedCatalogue) -> BuildResult<()> {
    let prefix = parsed.prefix;
    writeln!(
        out,
        "pub(super) static {prefix}_POSITIONS: [f64; {}] = [",
        grouped_decimal(parsed.positions.len())
    )?;
    for value in &parsed.positions {
        writeln!(out, "    {},", f64_from_bits_literal(*value))?;
    }
    out.push_str("];\n");
    writeln!(
        out,
        "pub(super) const {prefix}_N_SAMPLES: usize = {};",
        grouped_decimal(parsed.n_samples)
    )?;
    for (name, value) in [
        ("JD_START", parsed.jd_start),
        ("JD_END", parsed.jd_end),
        ("DT_DAYS", parsed.dt_days),
        (
            "MAX_DIRECTION_RATE_PER_DAY",
            parsed.max_direction_rate_per_day,
        ),
    ] {
        writeln!(
            out,
            "pub(super) const {prefix}_{name}: f64 = {}; // {value:e}",
            f64_from_bits_literal(value)
        )?;
    }
    writeln!(
        out,
        "pub(super) const {prefix}_EPOCH_SCALE_TAG: u8 = {:#04x};",
        parsed.epoch_scale_tag
    )?;
    writeln!(
        out,
        "pub(super) const {prefix}_EPOCH_REPRESENTATION_TAG: u8 = {:#04x};",
        parsed.epoch_representation_tag
    )?;
    writeln!(
        out,
        "pub(super) const {prefix}_BODY_ID: u8 = {};",
        parsed.body_id
    )?;
    writeln!(
        out,
        "pub(super) const {prefix}_SIZE_BYTES: usize = {};",
        grouped_decimal(parsed.size_bytes)
    )?;
    let sha_bytes: Vec<String> = parsed
        .sha256
        .iter()
        .map(|byte| format!("{byte:#04x}"))
        .collect();
    writeln!(
        out,
        "pub(super) const {prefix}_CONTENT_SHA256: [u8; 32] = [{}];",
        sha_bytes.join(", ")
    )?;
    writeln!(
        out,
        "pub(super) const {prefix}_CONTENT_SHA256_HEX: &str = \"{}\";",
        lowercase_hex(&parsed.sha256)
    )?;
    Ok(())
}

/// Mirrors the record format of `EMBEDDED_EPHEMERIS_BUNDLE_SHA256_HEX`:
/// `name=<catalogue sha256 hex>\n` in `Body::DEFAULT` order.
fn bundle_sha256_hex(catalogues: &[(&str, &ParsedCatalogue)]) -> String {
    let mut hasher = Sha256::new();
    for (body_name, parsed) in catalogues {
        hasher.update(body_name.as_bytes());
        hasher.update(b"=");
        hasher.update(lowercase_hex(&parsed.sha256).as_bytes());
        hasher.update(b"\n");
    }
    lowercase_hex(&hasher.finalize())
}

fn main() -> BuildResult<()> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let ephemeris_dir = manifest_dir.join("data/ephemeris");

    let mut out = String::new();
    out.push_str("// Generated by build.rs from data/ephemeris/*.bin. Do not edit.\n");

    let mut parsed_catalogues = Vec::new();
    for (file, prefix, body_id) in [
        ("sun.bin", "SUN", 0_u8),
        ("moon.bin", "MOON", 1_u8),
        ("jupiter.bin", "JUPITER", 2_u8),
        ("venus.bin", "VENUS", 3_u8),
    ] {
        let path = ephemeris_dir.join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        let bytes = fs::read(&path)?;
        parsed_catalogues.push(parse_catalogue(prefix, body_id, &bytes)?);
    }

    for parsed in &parsed_catalogues {
        write_catalogue_items(&mut out, parsed)?;
    }

    let bundle_records: Vec<(&str, &ParsedCatalogue)> = ["sun", "moon", "jupiter", "venus"]
        .iter()
        .zip(parsed_catalogues.iter())
        .map(|(name, parsed)| (*name, parsed))
        .collect();
    writeln!(
        out,
        "pub(super) const EMBEDDED_BUNDLE_SHA256_HEX: &str = \"{}\";",
        bundle_sha256_hex(&bundle_records)
    )?;

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    fs::write(out_dir.join("ephemeris_tables.rs"), out)?;
    Ok(())
}
