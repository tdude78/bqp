//! Immutable, offline JB2008 driver ingestion.
//!
//! This module parses SET's unmodified `SOLFSMY.TXT` and `DTCFILE.TXT` bytes
//! and exposes lagged solar inputs plus linearly interpolated hourly dTc. Part
//! A v3 additionally derives one sealed, synthetic last-value-persistence
//! scenario in memory from the final validated SET rows. It does not download
//! data, select an atmosphere model, or evaluate atmospheric density.

use crate::jb2008::{JB2008_KERNEL_NAME, JB2008_KERNEL_VERSION};
use anyhow::{anyhow, Context as _};
use num_traits::ToPrimitive;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

const F10_S10_LAG_DAYS: usize = 1;
const M10_LAG_DAYS: usize = 2;
const Y10_LAG_DAYS: usize = 5;
const HOURS_PER_DAY: f64 = 24.0;
const SET_DTC_DAY_BIAS: f64 = 0.000_000_1;
const LAST_HOUR_BEFORE_DAY_SPILL: f64 = 23.999_999_999_999_996;
const APPROVED_MANIFEST_SIZE_BYTES: usize = 1_820;
const APPROVED_MANIFEST_SHA256: &str =
    "09d2cb0cdb5cb805f6b10e9e2141e02d9320c91071491e2b32da87857deac01c";
const APPROVED_LICENSE_SIZE_BYTES: usize = 9_094;
const APPROVED_LICENSE_SHA256: &str =
    "9d2ec826044266557880c7863443ac2609312db39780b1c8239f5578dd75d387";
const APPROVED_LICENSE_SOURCE_URL: &str = "https://sol.spacenvironment.net/JB2008/License.html";
const PART_A_V3_MANIFEST_SIZE_BYTES: usize = 1_614;
const PART_A_V3_MANIFEST_SHA256: &str =
    "92c841895482614b3c05bd9dde704fcd82bcdab2a8df47a22130ea658f321fb7";
const PART_A_V3_AUTHORITY_ID: &str = "part-a-v3-jb2008-last-value-persistence-v1";
const PART_A_V3_CLAIM: &str =
    "synthetic last-value-persistence atmosphere scenario; not observed after source cutoff; not a space-weather forecast";
const PART_A_V3_POLICY: &str = "last-validated-parent-row-persistence-v1";
const PART_A_V3_OBSERVED_CUTOFF_UTC_DATE: &str = "2026-06-03";
const PART_A_V3_T0_UTC: &str = "2026-08-17T17:24:29Z";
const PART_A_V3_AUTHORIZED_START_UTC: &str = "2026-08-15T11:24:29Z";
const PART_A_V3_AUTHORIZED_END_UTC: &str = "2026-08-31T17:24:29Z";
const PART_A_V3_AUTHORIZED_START_JD: f64 = 2_461_267.975_335_648_3;
const PART_A_V3_T0_JD: f64 = 2_461_270.225_335_648_3;
const PART_A_V3_AUTHORIZED_END_JD: f64 = 2_461_284.225_335_648_3;
const PART_A_V3_SOLAR_SUPPORT_FIRST_UTC_DATE: &str = "2026-08-10";
const PART_A_V3_SOLAR_SUPPORT_LAST_UTC_DATE: &str = "2026-09-01";
const PART_A_V3_SOLAR_FIRST_DOY: i32 = 222;
const PART_A_V3_SOLAR_LAST_DOY: i32 = 244;
const PART_A_V3_DTC_SUPPORT_FIRST_UTC_DATE: &str = "2026-08-15";
const PART_A_V3_DTC_SUPPORT_LAST_UTC_DATE: &str = "2026-09-01";
const PART_A_V3_DTC_FIRST_DOY: i32 = 227;
const PART_A_V3_DTC_LAST_DOY: i32 = 244;

// The ordering these constants claim, checked at BUILD time rather than when a
// test happens to run. They are all `const`, so there is no reason to defer it.
const _: () = assert!(
    PART_A_V3_AUTHORIZED_START_JD <= PART_A_V3_T0_JD
        && PART_A_V3_T0_JD <= PART_A_V3_AUTHORIZED_END_JD,
    "t0 is outside its own authorized window"
);
const _: () = assert!(
    PART_A_V3_SOLAR_FIRST_DOY <= PART_A_V3_DTC_FIRST_DOY
        && PART_A_V3_DTC_LAST_DOY <= PART_A_V3_SOLAR_LAST_DOY,
    "the DTC support span is not contained in the solar support span"
);

const COMPILED_SOLFSMY: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/jb2008/SOLFSMY.TXT"
));
const COMPILED_DTCFILE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/jb2008/DTCFILE.TXT"
));
const COMPILED_MANIFEST: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/jb2008/manifest.json"
));
const COMPILED_LICENSE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/jb2008/License.html"
));
const COMPILED_PART_A_V3_MANIFEST: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/reference/atmosphere/jb2008/part_a_v3_persistence_v1/manifest.json"
));

/// Build-time-baked SOLFSMY row emitted by `build.rs`. The
/// `baked_rows_are_bit_identical_to_the_parser` oracle pins every field
/// against the kept runtime parser.
pub(crate) struct BakedSolarRow {
    pub year: i32,
    pub doy: i32,
    pub ordinal: i64,
    pub jd: f64,
    pub f10: f64,
    pub f10b: f64,
    pub s10: f64,
    pub s10b: f64,
    pub m10: f64,
    pub m10b: f64,
    pub y10: f64,
    pub y10b: f64,
    pub source: &'static str,
}

/// Build-time-baked DTCFILE row emitted by `build.rs`.
pub(crate) struct BakedDtcRow {
    pub year: i32,
    pub doy: i32,
    pub ordinal: i64,
    pub values: [i32; 24],
}

/// Baked row tables for the compiled SET catalogue (see `build.rs`).
mod baked {
    use super::{BakedDtcRow, BakedSolarRow};
    include!(concat!(env!("OUT_DIR"), "/jb2008_driver_tables.rs"));
}

/// Immutable SET authority parsed and hash-validated once per linked native image.
static COMPILED_DRIVERS: OnceLock<Arc<Jb2008Drivers>> = OnceLock::new();
static COMPILED_DRIVERS_LOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());
/// Immutable Part A v3 persistence authority derived once per linked image.
static COMPILED_PART_A_V3_DRIVERS: OnceLock<Arc<Jb2008Drivers>> = OnceLock::new();
/// Its own lock, NOT shared with `COMPILED_DRIVERS_LOAD`: building the v3
/// drivers calls `compiled_drivers()`, so one lock for both would deadlock a
/// cold Part A load on a single thread.
static COMPILED_PART_A_V3_DRIVERS_LOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Double-checked behind a lock, and the lock is PER CACHE, passed in by the
// caller. That is not a style choice: a single module-wide lock DEADLOCKS the
// moment one cached initializer loads another, which `jb_rs` does --
// `part_a_v3_drivers` -> `build_part_a_v3_drivers` -> `compiled_drivers`, two
// caches deep on one thread. `std::sync::Mutex` is not reentrant, so a shared
// lock would hang the cold Part A JB2008 load outright.
//
// The lock exists because `OnceLock` + `get_or_init` guarantees one STORED
// value, not one call to `initialize`. Every thread missing the `get()` runs the
// full load -- here always an expensive authority -- and `get_or_init` then keeps
// one result and discards the rest. Measured on the equivalent path in
// `nd_pipeline::event_bank_v3`: 14 of 16 racing threads did the work twice over.
//
// `OnceLock<Result<..>>` would also serialise this and is rejected: it caches a
// FAILURE, so one bad load is read back as an answer by every later caller. A
// failure here drops the guard with the cache still empty and the next caller
// retries. That is what `success_only` names.
fn success_only_cached<T>(
    cache: &OnceLock<Arc<T>>,
    cold_load: &std::sync::Mutex<()>,
    initialize: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<Arc<T>> {
    if let Some(value) = cache.get() {
        return Ok(Arc::clone(value));
    }
    let _guard = cold_load
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(value) = cache.get() {
        return Ok(Arc::clone(value));
    }
    let candidate = Arc::new(initialize()?);
    Ok(Arc::clone(cache.get_or_init(|| candidate)))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jb2008DriverInput {
    pub f10: f64,
    pub f10b: f64,
    pub s10: f64,
    pub s10b: f64,
    pub m10: f64,
    pub m10b: f64,
    pub y10: f64,
    pub y10b: f64,
    pub dtcval: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtcJulianDay(f64);

impl UtcJulianDay {
    /// Construct a finite UTC Julian day.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` is non-finite.
    pub fn new(value: f64) -> anyhow::Result<Self> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(anyhow!("UTC Julian day is non-finite"))
        }
    }
    #[must_use]
    pub const fn as_f64(self) -> f64 {
        self.0
    }

    /// Convert this UTC Julian day to UTC Modified Julian Day.
    ///
    /// # Errors
    ///
    /// Returns an error if conversion produces a non-finite value.
    pub fn to_utc_mjd(self) -> anyhow::Result<UtcModifiedJulianDay> {
        UtcModifiedJulianDay::new(self.0 - 2_400_000.5)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtcModifiedJulianDay(f64);

impl UtcModifiedJulianDay {
    /// Construct a finite UTC Modified Julian Day.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` is non-finite.
    pub fn new(value: f64) -> anyhow::Result<Self> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(anyhow!("UTC modified Julian day is non-finite"))
        }
    }
    /// Convert this UTC Modified Julian Day to UTC Julian Day.
    ///
    /// # Errors
    ///
    /// Returns an error if conversion produces a non-finite value.
    pub fn to_utc_jd(self) -> anyhow::Result<UtcJulianDay> {
        UtcJulianDay::new(self.0 + 2_400_000.5)
    }

    #[must_use]
    pub const fn as_f64(self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Jb2008DriverIdentity {
    pub solfsmy_release_header: String,
    pub solfsmy_coverage_start_jd: f64,
    pub solfsmy_coverage_end_jd: f64,
    pub dtc_coverage_start_jd: f64,
    pub dtc_coverage_end_jd: f64,
    pub solfsmy_source_size_bytes: usize,
    pub dtcfile_source_size_bytes: usize,
    pub source_declared_record_count: usize,
    pub solfsmy_parsed_record_count: usize,
    pub dtcfile_parsed_record_count: usize,
    pub license_acknowledged: bool,
    pub license_local_file: String,
    manifest_sha256: [u8; 32],
    license_sha256: [u8; 32],
    solfsmy_sha256: [u8; 32],
    dtcfile_sha256: [u8; 32],
}

/// Compact provenance for `COMPILED_DRIVERS`.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledJb2008Identity {
    pub kernel_name: &'static str,
    pub kernel_version: &'static str,
    pub manifest_sha256: String,
    pub solfsmy_sha256: String,
    pub dtcfile_sha256: String,
    pub license_sha256: String,
    pub set_release: String,
    pub solfsmy_coverage_start_jd: f64,
    pub solfsmy_coverage_end_jd: f64,
    pub dtc_coverage_start_jd: f64,
    pub dtc_coverage_end_jd: f64,
}

/// Compact provenance for the compiled Part A v3 persistence authority.
#[derive(Debug, Clone, PartialEq)]
pub struct PartAV3Jb2008Identity {
    pub authority_id: &'static str,
    pub claim: &'static str,
    pub policy: &'static str,
    pub manifest_sha256: String,
    pub parent_manifest_sha256: String,
    pub parent_solfsmy_sha256: String,
    pub parent_dtcfile_sha256: String,
    pub parent_license_sha256: String,
    pub observed_cutoff_utc_date: &'static str,
    pub t0_utc: &'static str,
    pub t0_utc_jd: f64,
    pub authorized_start_utc: &'static str,
    pub authorized_start_utc_jd: f64,
    pub authorized_end_utc: &'static str,
    pub authorized_end_utc_jd: f64,
    pub solar_support_first_utc_date: &'static str,
    pub solar_support_last_utc_date: &'static str,
    pub dtc_support_first_utc_date: &'static str,
    pub dtc_support_last_utc_date: &'static str,
    pub source_solar_fields_bits: [u64; 8],
    pub source_dtc_value: i32,
}

impl Jb2008DriverIdentity {
    #[must_use]
    pub fn solfsmy_sha256_hex(&self) -> String {
        sha256_hex(&self.solfsmy_sha256)
    }

    #[must_use]
    pub fn dtcfile_sha256_hex(&self) -> String {
        sha256_hex(&self.dtcfile_sha256)
    }

    #[must_use]
    pub fn manifest_sha256_hex(&self) -> String {
        sha256_hex(&self.manifest_sha256)
    }

    #[must_use]
    pub fn license_sha256_hex(&self) -> String {
        sha256_hex(&self.license_sha256)
    }
}

fn exact_usize_as_f64(value: usize, context: &str) -> anyhow::Result<f64> {
    u32::try_from(value)
        .map(f64::from)
        .with_context(|| format!("{context} exceeds exact f64 index range"))
}

fn nonnegative_integral_f64_as_usize(value: f64, context: &str) -> anyhow::Result<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(anyhow!("{context} is not a nonnegative integer"));
    }
    value
        .to_usize()
        .ok_or_else(|| anyhow!("{context} is outside usize range"))
}

fn exact_i64_as_f64(value: i64, context: &str) -> anyhow::Result<f64> {
    const MAX_EXACT_I64: i64 = 9_007_199_254_740_992;
    if !(-MAX_EXACT_I64..=MAX_EXACT_I64).contains(&value) {
        return Err(anyhow!("{context} exceeds exact f64 integer range"));
    }
    value
        .to_f64()
        .ok_or_else(|| anyhow!("{context} cannot convert to f64"))
}

#[derive(Debug, Clone)]
pub struct Jb2008Drivers {
    solar_rows: Vec<SolarRow>,
    dtc_rows: Vec<DtcRow>,
    identity: Jb2008DriverIdentity,
    authorized_utc_arc: Option<AuthorizedUtcArc>,
    part_a_v3_identity: Option<PartAV3Jb2008Identity>,
}

#[derive(Debug, Clone, Copy)]
struct AuthorizedUtcArc {
    start_jd: f64,
    end_jd: f64,
}

impl Jb2008Drivers {
    /// Parse raw SET-shaped bytes without granting approved-source authority.
    ///
    /// Intended for parser tests and tooling; production uses `compiled_drivers`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid UTF-8, malformed rows, or invalid
    /// catalogue coverage.
    pub fn from_set_bytes(solfsmy_bytes: &[u8], dtcfile_bytes: &[u8]) -> anyhow::Result<Self> {
        let solfsmy = std::str::from_utf8(solfsmy_bytes).context("SOLFSMY is not valid UTF-8")?;
        let dtcfile = std::str::from_utf8(dtcfile_bytes).context("DTCFILE is not valid UTF-8")?;
        let (solar_rows, release_header) = parse_solfsmy(solfsmy)?;
        let dtc_rows = parse_dtcfile(dtcfile)?;
        Self::from_rows(
            solar_rows,
            dtc_rows,
            release_header,
            solfsmy_bytes,
            dtcfile_bytes,
        )
    }

    /// Assemble drivers from validated rows plus the raw bytes they came from.
    ///
    /// The bytes are hashed into the identity — the SHA-256 trust root stays a
    /// runtime computation on both the parsed and the baked path.
    fn from_rows(
        solar_rows: Vec<SolarRow>,
        dtc_rows: Vec<DtcRow>,
        release_header: String,
        solfsmy_bytes: &[u8],
        dtcfile_bytes: &[u8],
    ) -> anyhow::Result<Self> {
        let (Some(first_solar), Some(last_solar), Some(first_dtc), Some(last_dtc)) = (
            solar_rows.first(),
            solar_rows.last(),
            dtc_rows.first(),
            dtc_rows.last(),
        ) else {
            return Err(anyhow!("parsed JB2008 catalogue has no data rows"));
        };
        let dtc_coverage_start_jd = solar_jd_for_date(&solar_rows, first_dtc.date)?;
        let dtc_coverage_end_jd = solar_jd_for_date(&solar_rows, last_dtc.date)?;
        let identity = Jb2008DriverIdentity {
            solfsmy_release_header: release_header,
            solfsmy_coverage_start_jd: first_solar.jd,
            solfsmy_coverage_end_jd: last_solar.jd,
            dtc_coverage_start_jd,
            dtc_coverage_end_jd,
            solfsmy_source_size_bytes: solfsmy_bytes.len(),
            dtcfile_source_size_bytes: dtcfile_bytes.len(),
            source_declared_record_count: 0,
            solfsmy_parsed_record_count: solar_rows.len(),
            dtcfile_parsed_record_count: dtc_rows.len(),
            license_acknowledged: false,
            license_local_file: String::new(),
            manifest_sha256: [0; 32],
            license_sha256: [0; 32],
            solfsmy_sha256: Sha256::digest(solfsmy_bytes).into(),
            dtcfile_sha256: Sha256::digest(dtcfile_bytes).into(),
        };

        Ok(Self {
            solar_rows,
            dtc_rows,
            identity,
            authorized_utc_arc: None,
            part_a_v3_identity: None,
        })
    }

    /// Assemble the compiled catalogue from the build-time-baked row tables.
    ///
    /// Zero text parsing: `build.rs` parsed and validated the same tracked
    /// bytes with a mirrored parser at build time, and the
    /// `baked_rows_are_bit_identical_to_the_parser` oracle pins the mirror
    /// against `from_set_bytes`. The bytes are still hashed here (trust root).
    fn from_baked_compiled_rows(
        solfsmy_bytes: &[u8],
        dtcfile_bytes: &[u8],
    ) -> anyhow::Result<Self> {
        let solar_rows = baked::BAKED_SOLAR_ROWS
            .iter()
            .map(|row| SolarRow {
                date: DateKey {
                    year: row.year,
                    doy: row.doy,
                    ordinal: row.ordinal,
                },
                jd: row.jd,
                f10: row.f10,
                f10b: row.f10b,
                s10: row.s10,
                s10b: row.s10b,
                m10: row.m10,
                m10b: row.m10b,
                y10: row.y10,
                y10b: row.y10b,
                source: Cow::Borrowed(row.source),
            })
            .collect();
        let dtc_rows = baked::BAKED_DTC_ROWS
            .iter()
            .map(|row| DtcRow {
                date: DateKey {
                    year: row.year,
                    doy: row.doy,
                    ordinal: row.ordinal,
                },
                values: row.values,
            })
            .collect();
        Self::from_rows(
            solar_rows,
            dtc_rows,
            baked::BAKED_SOLFSMY_RELEASE_HEADER.to_owned(),
            solfsmy_bytes,
            dtcfile_bytes,
        )
    }

    /// Parse and validate an approved immutable SET authority bundle.
    ///
    /// # Errors
    ///
    /// Returns an error for any trust-root, hash, manifest, parser, or
    /// coverage failure.
    pub fn from_approved_set_bytes(
        solfsmy_bytes: &[u8],
        dtcfile_bytes: &[u8],
        manifest_bytes: &[u8],
        license_bytes: &[u8],
    ) -> anyhow::Result<Self> {
        let (manifest_digest, license_digest) = approved_trust_root(manifest_bytes, license_bytes)?;
        let drivers = Self::from_set_bytes(solfsmy_bytes, dtcfile_bytes)?;
        Self::bind_manifest(
            drivers,
            manifest_bytes,
            license_bytes,
            manifest_digest,
            license_digest,
        )
    }

    /// Compiled-authority constructor: verbatim trust root, baked rows.
    fn from_approved_compiled_baked() -> anyhow::Result<Self> {
        let (manifest_digest, license_digest) =
            approved_trust_root(COMPILED_MANIFEST, COMPILED_LICENSE)?;
        let drivers = Self::from_baked_compiled_rows(COMPILED_SOLFSMY, COMPILED_DTCFILE)?;
        Self::bind_manifest(
            drivers,
            COMPILED_MANIFEST,
            COMPILED_LICENSE,
            manifest_digest,
            license_digest,
        )
    }

    fn bind_manifest(
        mut drivers: Self,
        manifest_bytes: &[u8],
        license_bytes: &[u8],
        manifest_digest: [u8; 32],
        license_digest: [u8; 32],
    ) -> anyhow::Result<Self> {
        let manifest: AuthorityManifest = serde_json::from_slice(manifest_bytes)
            .context("JB2008 authority manifest is invalid JSON")?;
        validate_manifest(&manifest, &drivers, license_bytes, &license_digest)?;
        let solfsmy = manifest
            .files
            .get("SOLFSMY.TXT")
            .ok_or_else(|| anyhow!("JB2008 authority manifest misses SOLFSMY.TXT"))?;
        drivers.identity.source_declared_record_count = solfsmy
            .source_declared_record_count
            .ok_or_else(|| anyhow!("SOLFSMY declared record count missing"))?;
        drivers.identity.license_acknowledged = manifest.license.acknowledged;
        drivers.identity.license_local_file = manifest.license.local_file;
        drivers.identity.manifest_sha256 = manifest_digest;
        drivers.identity.license_sha256 = license_digest;
        Ok(drivers)
    }

    #[must_use]
    pub const fn identity(&self) -> &Jb2008DriverIdentity {
        &self.identity
    }

    /// Look up lagged drivers for a finite UTC Julian day.
    ///
    /// # Errors
    ///
    /// Returns an error when required catalogue rows or interpolation
    /// inputs are unavailable.
    pub fn lookup_utc_jd(&self, utc_jd: UtcJulianDay) -> anyhow::Result<Jb2008DriverInput> {
        self.validate_authorized_utc_instant(utc_jd)?;
        self.lookup(utc_jd.as_f64())
    }

    /// Look up lagged drivers for a finite UTC Modified Julian Day.
    ///
    /// # Errors
    ///
    /// Returns an error when conversion, required catalogue rows, or
    /// interpolation inputs are unavailable.
    pub fn lookup_utc_mjd(
        &self,
        utc_mjd: UtcModifiedJulianDay,
    ) -> anyhow::Result<Jb2008DriverInput> {
        self.lookup_utc_jd(utc_mjd.to_utc_jd()?)
    }

    /// Validate complete UTC arc before propagation enters an RHS loop.
    ///
    /// Contiguous parsed catalogues make start's five-day Y10 lag and end's
    /// next DTC day sufficient for every day between endpoints.
    ///
    /// # Errors
    ///
    /// Returns an error when arc order, lag coverage, or next-day DTC
    /// coverage is invalid.
    pub fn validate_utc_arc(
        &self,
        start_utc_jd: UtcJulianDay,
        end_utc_jd: UtcJulianDay,
    ) -> anyhow::Result<()> {
        if end_utc_jd.as_f64() < start_utc_jd.as_f64() {
            return Err(anyhow!("UTC arc end precedes start"));
        }
        self.validate_authorized_utc_instant(start_utc_jd)?;
        self.validate_authorized_utc_instant(end_utc_jd)?;
        let (start_index, start_solar) = self.solar_row_at_utc(start_utc_jd)?;
        let (_, end_solar) = self.solar_row_at_utc(end_utc_jd)?;
        start_index
            .checked_sub(Y10_LAG_DAYS)
            .ok_or_else(|| anyhow!("missing Y10 5-day lagged driver"))?;

        self.dtc_row_index(start_solar.date)?;
        let end_dtc_index = self.dtc_row_index(end_solar.date)?;
        let next_dtc_index = end_dtc_index
            .checked_add(1)
            .ok_or_else(|| anyhow!("next DTCFILE index overflows"))?;
        let next_ordinal = end_solar
            .date
            .ordinal
            .checked_add(1)
            .ok_or_else(|| anyhow!("next DTCFILE date overflows"))?;
        self.dtc_rows
            .get(next_dtc_index)
            .filter(|next| next.date.ordinal == next_ordinal)
            .ok_or_else(|| anyhow!("missing next DTCFILE day for UTC arc"))?;
        Ok(())
    }

    fn validate_authorized_utc_instant(&self, utc_jd: UtcJulianDay) -> anyhow::Result<()> {
        let Some(authorized) = self.authorized_utc_arc else {
            return Ok(());
        };
        if utc_jd.as_f64() < authorized.start_jd || utc_jd.as_f64() > authorized.end_jd {
            return Err(anyhow!(
                "lookup outside Part A v3 authorized persistence arc",
            ));
        }
        Ok(())
    }

    fn solar_row_at_utc(&self, utc_jd: UtcJulianDay) -> anyhow::Result<(usize, &SolarRow)> {
        let day_jd = (utc_jd.as_f64() + 0.5).floor();
        if !day_jd.is_finite() {
            return Err(anyhow!("lookup day is non-finite"));
        }
        let first_solar = self
            .solar_rows
            .first()
            .ok_or_else(|| anyhow!("SOLFSMY has no data rows"))?;
        let solar_offset = day_jd - first_solar.jd;
        let solar_len = exact_usize_as_f64(self.solar_rows.len(), "SOLFSMY row count")?;
        if solar_offset < 0.0 || solar_offset >= solar_len {
            return Err(anyhow!("lookup outside SOLFSMY coverage"));
        }
        let solar_index = nonnegative_integral_f64_as_usize(solar_offset, "SOLFSMY row offset")?;
        let solar = self
            .solar_rows
            .get(solar_index)
            .filter(|row| row.jd.to_bits() == day_jd.to_bits())
            .ok_or_else(|| anyhow!("lookup outside SOLFSMY coverage"))?;
        Ok((solar_index, solar))
    }

    fn dtc_row_index(&self, date: DateKey) -> anyhow::Result<usize> {
        let first_dtc = self
            .dtc_rows
            .first()
            .ok_or_else(|| anyhow!("DTCFILE has no data rows"))?;
        let dtc_offset = date
            .ordinal
            .checked_sub(first_dtc.date.ordinal)
            .ok_or_else(|| anyhow!("lookup outside DTCFILE coverage"))?;
        let dtc_index =
            usize::try_from(dtc_offset).map_err(|_| anyhow!("lookup outside DTCFILE coverage"))?;
        self.dtc_rows
            .get(dtc_index)
            .filter(|row| row.date == date)
            .ok_or_else(|| anyhow!("lookup outside DTCFILE coverage"))?;
        Ok(dtc_index)
    }

    fn lookup(&self, jd: f64) -> anyhow::Result<Jb2008DriverInput> {
        let utc_jd = UtcJulianDay::new(jd)?;
        let day_jd = (jd + 0.5).floor();
        if !day_jd.is_finite() {
            return Err(anyhow!("lookup day is non-finite"));
        }
        let (solar_index, solar) = self.solar_row_at_utc(utc_jd)?;
        let solar_date = solar.date;
        let f10_s10_index = solar_index
            .checked_sub(F10_S10_LAG_DAYS)
            .ok_or_else(|| anyhow!("missing F10/S10 lagged driver"))?;
        let m10_index = solar_index
            .checked_sub(M10_LAG_DAYS)
            .ok_or_else(|| anyhow!("missing M10 lagged driver"))?;
        let y10_index = solar_index
            .checked_sub(Y10_LAG_DAYS)
            .ok_or_else(|| anyhow!("missing Y10 lagged driver"))?;
        let dtc = self.interpolate_dtc(solar_date, jd - (day_jd - 0.5))?;

        let f10_s10 = self
            .solar_rows
            .get(f10_s10_index)
            .ok_or_else(|| anyhow!("missing F10/S10 lagged driver"))?;
        let m10 = self
            .solar_rows
            .get(m10_index)
            .ok_or_else(|| anyhow!("missing M10 lagged driver"))?;
        let y10 = self
            .solar_rows
            .get(y10_index)
            .ok_or_else(|| anyhow!("missing Y10 lagged driver"))?;
        Ok(Jb2008DriverInput {
            f10: f10_s10.f10,
            f10b: f10_s10.f10b,
            s10: f10_s10.s10,
            s10b: f10_s10.s10b,
            m10: m10.m10,
            m10b: m10.m10b,
            y10: y10.y10,
            y10b: y10.y10b,
            dtcval: dtc,
        })
    }

    fn interpolate_dtc(&self, date: DateKey, utc_day_fraction: f64) -> anyhow::Result<i32> {
        if !utc_day_fraction.is_finite() || !(0.0..1.0).contains(&utc_day_fraction) {
            return Err(anyhow!("lookup UTC fraction is outside calendar day"));
        }
        let first_dtc = self
            .dtc_rows
            .first()
            .ok_or_else(|| anyhow!("DTCFILE has no data rows"))?;
        let dtc_offset = date
            .ordinal
            .checked_sub(first_dtc.date.ordinal)
            .ok_or_else(|| anyhow!("lookup outside DTCFILE coverage"))?;
        let row_index =
            usize::try_from(dtc_offset).map_err(|_| anyhow!("lookup outside DTCFILE coverage"))?;
        if self.dtc_rows.get(row_index).map(|row| row.date) != Some(date) {
            return Err(anyhow!("lookup outside DTCFILE coverage"));
        }
        let hour =
            ((utc_day_fraction + SET_DTC_DAY_BIAS) * HOURS_PER_DAY).min(LAST_HOUR_BEFORE_DAY_SPILL);
        let lower_hour = nonnegative_integral_f64_as_usize(hour.floor(), "DTCFILE hour")?;
        if lower_hour >= 24 {
            return Err(anyhow!("lookup hour is outside DTCFILE range"));
        }
        let fraction = hour - exact_usize_as_f64(lower_hour, "DTCFILE hour")?;
        let current_row = self
            .dtc_rows
            .get(row_index)
            .ok_or_else(|| anyhow!("lookup outside DTCFILE coverage"))?;
        let upper = if lower_hour == 23 && fraction > 0.0 {
            let next_row_index = row_index
                .checked_add(1)
                .ok_or_else(|| anyhow!("next DTCFILE index overflows"))?;
            let next_ordinal = date
                .ordinal
                .checked_add(1)
                .ok_or_else(|| anyhow!("next DTCFILE date overflows"))?;
            self.dtc_rows
                .get(next_row_index)
                .filter(|row| row.date.ordinal == next_ordinal)
                .ok_or_else(|| anyhow!("missing next DTCFILE day for interpolation"))?
                .values
                .first()
                .copied()
                .ok_or_else(|| anyhow!("DTCFILE row has no hourly values"))?
        } else {
            let upper_hour = lower_hour.checked_add(1).map_or(23, |hour| hour.min(23));
            current_row
                .values
                .get(upper_hour)
                .copied()
                .ok_or_else(|| anyhow!("DTCFILE hour is outside row"))?
        };
        let lower = current_row
            .values
            .get(lower_hour)
            .copied()
            .ok_or_else(|| anyhow!("DTCFILE hour is outside row"))?;
        let interpolated = f64::from(lower) + fraction * (f64::from(upper) - f64::from(lower));
        let rounded = (interpolated + 0.5).trunc();
        rounded
            .to_i32()
            .ok_or_else(|| anyhow!("interpolated DTCVAL is outside i32 range"))
    }
}

/// Clones linked-image immutable authority. No parsing, hashing, or lookup.
///
/// # Errors
///
/// Returns an error if linked approved authority failed validation.
pub fn compiled_drivers() -> anyhow::Result<Arc<Jb2008Drivers>> {
    success_only_cached(&COMPILED_DRIVERS, &COMPILED_DRIVERS_LOAD, || {
        Jb2008Drivers::from_approved_compiled_baked()
    })
}

/// Clones linked-image Part A v3 synthetic persistence authority.
///
/// Source rows are generated in memory from the final hash-validated parent
/// SET row. No generated driver file, runtime override, or fallback exists.
///
/// # Errors
///
/// Returns an error for any parent trust-root, scenario-manifest, source-row,
/// support-coverage, or authorized-arc mismatch.
pub fn compiled_part_a_v3_drivers() -> anyhow::Result<Arc<Jb2008Drivers>> {
    if let Some(drivers) = COMPILED_PART_A_V3_DRIVERS.get() {
        return Ok(Arc::clone(drivers));
    }
    // Resolve the PARENT before taking our own cold-load lock.
    //
    // `build_part_a_v3_drivers` calls `compiled_drivers()`, so running it under
    // `COMPILED_PART_A_V3_DRIVERS_LOAD` means holding one cold-load lock while
    // taking another. With a single shared lock that was an outright
    // self-deadlock (fixed in `ad483a6e`); with per-cache locks it is merely an
    // ordering that nothing enforces, and an ordering nothing enforces is a
    // deadlock waiting for the second edge.
    //
    // Warming the parent here removes the nesting rather than reasoning about
    // it. After this returns Ok, `COMPILED_DRIVERS` is populated, so the call
    // inside `build_part_a_v3_drivers` takes the `get()` fast path and acquires
    // no lock at all -- so no thread ever holds two of these at once.
    drop(compiled_drivers()?);
    success_only_cached(
        &COMPILED_PART_A_V3_DRIVERS,
        &COMPILED_PART_A_V3_DRIVERS_LOAD,
        build_part_a_v3_drivers,
    )
}

/// Returns compact identity for compiled Part A v3 persistence authority.
///
/// # Errors
///
/// Returns an error if linked parent or scenario authority fails validation.
pub fn compiled_part_a_v3_identity() -> anyhow::Result<PartAV3Jb2008Identity> {
    compiled_part_a_v3_drivers()?
        .part_a_v3_identity
        .clone()
        .ok_or_else(|| anyhow!("compiled Part A v3 driver identity is missing"))
}

fn build_part_a_v3_drivers() -> anyhow::Result<Jb2008Drivers> {
    let scenario_digest: [u8; 32] = Sha256::digest(COMPILED_PART_A_V3_MANIFEST).into();
    if COMPILED_PART_A_V3_MANIFEST.len() != PART_A_V3_MANIFEST_SIZE_BYTES
        || sha256_hex(&scenario_digest) != PART_A_V3_MANIFEST_SHA256
    {
        return Err(anyhow!(
            "Part A v3 persistence manifest does not match compiled trust root",
        ));
    }
    let scenario: PartAV3PersistenceManifest = serde_json::from_slice(COMPILED_PART_A_V3_MANIFEST)
        .context("Part A v3 persistence manifest is invalid JSON")?;
    let parent = compiled_drivers()?;
    let source_solar_fields_bits = validate_part_a_v3_manifest(&scenario, &parent)?;
    let source_solar = parent
        .solar_rows
        .last()
        .ok_or_else(|| anyhow!("validated parent SOLFSMY has no final row"))?;

    let solar_rows = (PART_A_V3_SOLAR_FIRST_DOY..=PART_A_V3_SOLAR_LAST_DOY)
        .map(|doy| {
            let date = part_a_v3_date(doy)?;
            Ok(SolarRow {
                date,
                jd: gregorian_utc_noon_jd(date)?,
                f10: source_solar.f10,
                f10b: source_solar.f10b,
                s10: source_solar.s10,
                s10b: source_solar.s10b,
                m10: source_solar.m10,
                m10b: source_solar.m10b,
                y10: source_solar.y10,
                y10b: source_solar.y10b,
                source: Cow::Borrowed(PART_A_V3_AUTHORITY_ID),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let dtc_values = [scenario.parent_set_authority.final_dtc_value; 24];
    let dtc_rows = (PART_A_V3_DTC_FIRST_DOY..=PART_A_V3_DTC_LAST_DOY)
        .map(|doy| {
            Ok(DtcRow {
                date: part_a_v3_date(doy)?,
                values: dtc_values,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    validate_solar_rows(&solar_rows)?;
    validate_dtc_rows(&dtc_rows)?;

    let first_solar = solar_rows
        .first()
        .ok_or_else(|| anyhow!("Part A v3 solar support is empty"))?;
    let last_solar = solar_rows
        .last()
        .ok_or_else(|| anyhow!("Part A v3 solar support is empty"))?;
    let first_dtc = dtc_rows
        .first()
        .ok_or_else(|| anyhow!("Part A v3 DTC support is empty"))?;
    let last_dtc = dtc_rows
        .last()
        .ok_or_else(|| anyhow!("Part A v3 DTC support is empty"))?;
    let mut identity = parent.identity.clone();
    PART_A_V3_AUTHORITY_ID.clone_into(&mut identity.solfsmy_release_header);
    identity.solfsmy_coverage_start_jd = first_solar.jd;
    identity.solfsmy_coverage_end_jd = last_solar.jd;
    identity.dtc_coverage_start_jd = gregorian_utc_noon_jd(first_dtc.date)?;
    identity.dtc_coverage_end_jd = gregorian_utc_noon_jd(last_dtc.date)?;
    identity.solfsmy_source_size_bytes = 0;
    identity.dtcfile_source_size_bytes = 0;
    identity.source_declared_record_count = solar_rows.len();
    identity.solfsmy_parsed_record_count = solar_rows.len();
    identity.dtcfile_parsed_record_count = dtc_rows.len();
    identity.manifest_sha256 = scenario_digest;

    let part_a_v3_identity = PartAV3Jb2008Identity {
        authority_id: PART_A_V3_AUTHORITY_ID,
        claim: PART_A_V3_CLAIM,
        policy: PART_A_V3_POLICY,
        manifest_sha256: PART_A_V3_MANIFEST_SHA256.to_owned(),
        parent_manifest_sha256: parent.identity.manifest_sha256_hex(),
        parent_solfsmy_sha256: parent.identity.solfsmy_sha256_hex(),
        parent_dtcfile_sha256: parent.identity.dtcfile_sha256_hex(),
        parent_license_sha256: parent.identity.license_sha256_hex(),
        observed_cutoff_utc_date: PART_A_V3_OBSERVED_CUTOFF_UTC_DATE,
        t0_utc: PART_A_V3_T0_UTC,
        t0_utc_jd: PART_A_V3_T0_JD,
        authorized_start_utc: PART_A_V3_AUTHORIZED_START_UTC,
        authorized_start_utc_jd: PART_A_V3_AUTHORIZED_START_JD,
        authorized_end_utc: PART_A_V3_AUTHORIZED_END_UTC,
        authorized_end_utc_jd: PART_A_V3_AUTHORIZED_END_JD,
        solar_support_first_utc_date: PART_A_V3_SOLAR_SUPPORT_FIRST_UTC_DATE,
        solar_support_last_utc_date: PART_A_V3_SOLAR_SUPPORT_LAST_UTC_DATE,
        dtc_support_first_utc_date: PART_A_V3_DTC_SUPPORT_FIRST_UTC_DATE,
        dtc_support_last_utc_date: PART_A_V3_DTC_SUPPORT_LAST_UTC_DATE,
        source_solar_fields_bits,
        source_dtc_value: scenario.parent_set_authority.final_dtc_value,
    };
    let drivers = Jb2008Drivers {
        solar_rows,
        dtc_rows,
        identity,
        authorized_utc_arc: Some(AuthorizedUtcArc {
            start_jd: PART_A_V3_AUTHORIZED_START_JD,
            end_jd: PART_A_V3_AUTHORIZED_END_JD,
        }),
        part_a_v3_identity: Some(part_a_v3_identity),
    };
    drivers.validate_utc_arc(
        UtcJulianDay::new(PART_A_V3_AUTHORIZED_START_JD)?,
        UtcJulianDay::new(PART_A_V3_AUTHORIZED_END_JD)?,
    )?;
    Ok(drivers)
}

/// Returns compact identity for compiled, hash-validated SET authority.
///
/// # Errors
///
/// Returns an error if linked approved authority failed validation.
pub fn compiled_identity() -> anyhow::Result<CompiledJb2008Identity> {
    let drivers = compiled_drivers()?;
    let identity = drivers.identity();
    Ok(CompiledJb2008Identity {
        kernel_name: JB2008_KERNEL_NAME,
        kernel_version: JB2008_KERNEL_VERSION,
        manifest_sha256: identity.manifest_sha256_hex(),
        solfsmy_sha256: identity.solfsmy_sha256_hex(),
        dtcfile_sha256: identity.dtcfile_sha256_hex(),
        license_sha256: identity.license_sha256_hex(),
        set_release: identity
            .solfsmy_release_header
            .trim_start_matches("# ")
            .to_owned(),
        solfsmy_coverage_start_jd: identity.solfsmy_coverage_start_jd,
        solfsmy_coverage_end_jd: identity.solfsmy_coverage_end_jd,
        dtc_coverage_start_jd: identity.dtc_coverage_start_jd,
        dtc_coverage_end_jd: identity.dtc_coverage_end_jd,
    })
}

/// Validate compiled SET drivers before entering a UTC propagation arc.
///
/// # Errors
///
/// Returns an error if linked approved authority or requested arc
/// coverage is invalid.
pub fn validate_utc_arc(
    start_utc_jd: UtcJulianDay,
    end_utc_jd: UtcJulianDay,
) -> anyhow::Result<()> {
    compiled_drivers()?.validate_utc_arc(start_utc_jd, end_utc_jd)
}

#[derive(Debug, Clone)]
struct SolarRow {
    date: DateKey,
    jd: f64,
    f10: f64,
    f10b: f64,
    s10: f64,
    s10b: f64,
    m10: f64,
    m10b: f64,
    y10: f64,
    y10b: f64,
    /// `Borrowed` on the baked and Part A v3 paths, `Owned` from the parser.
    source: Cow<'static, str>,
}

#[derive(Debug, Clone)]
struct DtcRow {
    date: DateKey,
    values: [i32; 24],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DateKey {
    year: i32,
    doy: i32,
    ordinal: i64,
}

fn one_based_line_number(line_index: usize) -> anyhow::Result<usize> {
    line_index
        .checked_add(1)
        .ok_or_else(|| anyhow!("catalogue line count overflows"))
}

fn parse_solfsmy(input: &str) -> anyhow::Result<(Vec<SolarRow>, String)> {
    let mut release_header = None;
    let mut rows = Vec::new();
    for (line_index, raw_line) in input.lines().enumerate() {
        let line_number = one_based_line_number(line_index)?;
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
            return Err(anyhow!(
                "SOLFSMY line {line_number} has {} fields; expected 12",
                fields.len()
            ));
        };
        let date = parse_date(year, doy, "SOLFSMY", line_number)?;
        let jd = parse_finite_f64(jd_text, "SOLFSMY Julian day", line_number)?;
        if !matches!(jd.fract().to_bits(), 0 | 0x8000_0000_0000_0000) {
            return Err(anyhow!(
                "SOLFSMY line {line_number} Julian day is not an integer noon key"
            ));
        }
        let f10 = parse_finite_f64(f10, "SOLFSMY F10", line_number)?;
        let f10b = parse_finite_f64(f10b, "SOLFSMY F81c", line_number)?;
        let s10 = parse_finite_f64(s10, "SOLFSMY S10", line_number)?;
        let s10b = parse_finite_f64(s10b, "SOLFSMY S81c", line_number)?;
        let m10 = parse_finite_f64(m10, "SOLFSMY M10", line_number)?;
        let m10b = parse_finite_f64(m10b, "SOLFSMY M81c", line_number)?;
        let y10 = parse_finite_f64(y10, "SOLFSMY Y10", line_number)?;
        let y10b = parse_finite_f64(y10b, "SOLFSMY Y81c", line_number)?;
        if source.is_empty() {
            return Err(anyhow!("SOLFSMY line {line_number} has empty source field"));
        }
        rows.push(SolarRow {
            date,
            jd,
            f10,
            f10b,
            s10,
            s10b,
            m10,
            m10b,
            y10,
            y10b,
            source: Cow::Owned((*source).to_owned()),
        });
    }
    let release_header = release_header.ok_or_else(|| anyhow!("SOLFSMY release header missing"))?;
    if !release_header.starts_with("# F10, S10, M10, Y10 data release") {
        return Err(anyhow!("SOLFSMY release header is unrecognized"));
    }
    validate_solar_rows(&rows)?;
    Ok((rows, release_header))
}

fn parse_dtc_values(value_fields: &[&str], line_number: usize) -> anyhow::Result<[i32; 24]> {
    let values = value_fields
        .iter()
        .map(|value| {
            value
                .parse::<i32>()
                .with_context(|| format!("DTCFILE dTc line {line_number} is not an integer"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    values
        .try_into()
        .map_err(|_| anyhow!("DTCFILE has an invalid hourly-value count"))
}

fn parse_dtcfile(input: &str) -> anyhow::Result<Vec<DtcRow>> {
    let mut rows = Vec::new();
    for (line_index, raw_line) in input.lines().enumerate() {
        let line_number = one_based_line_number(line_index)?;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        let Some((marker, calendar_and_values)) = fields.split_first() else {
            return Err(anyhow!(
                "DTCFILE line {line_number} must contain DTC plus 26 fields"
            ));
        };
        let Some((year, day_and_values)) = calendar_and_values.split_first() else {
            return Err(anyhow!(
                "DTCFILE line {line_number} must contain DTC plus 26 fields"
            ));
        };
        let Some((doy, value_fields)) = day_and_values.split_first() else {
            return Err(anyhow!(
                "DTCFILE line {line_number} must contain DTC plus 26 fields"
            ));
        };
        if *marker != "DTC" || value_fields.len() != 24 {
            return Err(anyhow!(
                "DTCFILE line {line_number} must contain DTC plus 26 fields"
            ));
        }
        let date = parse_date(year, doy, "DTCFILE", line_number)?;
        let values = parse_dtc_values(value_fields, line_number)?;
        rows.push(DtcRow { date, values });
    }
    validate_dtc_rows(&rows)?;
    Ok(rows)
}

fn parse_date(year: &str, doy: &str, source: &str, line_number: usize) -> anyhow::Result<DateKey> {
    let year = year
        .parse::<i32>()
        .with_context(|| format!("{source} line {line_number} has invalid year"))?;
    let doy = doy
        .parse::<i32>()
        .with_context(|| format!("{source} line {line_number} has invalid day of year"))?;
    if year < 1 || !(1..=days_in_year(year)).contains(&doy) {
        return Err(anyhow!(
            "{source} line {line_number} has invalid calendar date"
        ));
    }
    let completed_years = i64::from(year)
        .checked_sub(1)
        .ok_or_else(|| anyhow!("calendar year underflows"))?;
    let completed_days = completed_years
        .checked_mul(365)
        .and_then(|value| value.checked_add(completed_years / 4))
        .and_then(|value| value.checked_sub(completed_years / 100))
        .and_then(|value| value.checked_add(completed_years / 400))
        .and_then(|value| value.checked_add(i64::from(doy).checked_sub(1)?))
        .ok_or_else(|| anyhow!("calendar ordinal overflows"))?;
    Ok(DateKey {
        year,
        doy,
        ordinal: completed_days,
    })
}

const fn days_in_year(year: i32) -> i32 {
    if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
        366
    } else {
        365
    }
}

fn parse_finite_f64(value: &str, field: &str, line_number: usize) -> anyhow::Result<f64> {
    let parsed = value
        .parse::<f64>()
        .with_context(|| format!("{field} line {line_number} is not numeric"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(anyhow!("{field} line {line_number} is non-finite"))
    }
}

fn validate_solar_rows(rows: &[SolarRow]) -> anyhow::Result<()> {
    let first = rows
        .first()
        .ok_or_else(|| anyhow!("SOLFSMY has no data rows"))?;
    if !first.jd.is_finite() {
        return Err(anyhow!("SOLFSMY first Julian day is non-finite"));
    }
    for row in rows {
        if row.source.is_empty() {
            return Err(anyhow!("SOLFSMY row has empty source field"));
        }
        let expected_jd = gregorian_utc_noon_jd(row.date)?;
        if row.jd.to_bits() != expected_jd.to_bits() {
            return Err(anyhow!(
                "SOLFSMY Gregorian UTC date and Julian day disagree",
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
            .ok_or_else(|| anyhow!("SOLFSMY calendar ordinal overflows"))?;
        if next.jd.to_bits() != expected_jd.to_bits() || next.date.ordinal != expected_ordinal {
            return Err(anyhow!(
                "SOLFSMY rows are missing, duplicated, or out of order",
            ));
        }
    }
    Ok(())
}

fn validate_dtc_rows(rows: &[DtcRow]) -> anyhow::Result<()> {
    rows.first()
        .ok_or_else(|| anyhow!("DTCFILE has no data rows"))?;
    for pair in rows.windows(2) {
        let [previous, next] = pair else {
            continue;
        };
        let expected_ordinal = previous
            .date
            .ordinal
            .checked_add(1)
            .ok_or_else(|| anyhow!("DTCFILE calendar ordinal overflows"))?;
        if next.date.ordinal != expected_ordinal {
            return Err(anyhow!(
                "DTCFILE rows are missing, duplicated, or out of order",
            ));
        }
    }
    Ok(())
}

fn solar_jd_for_date(rows: &[SolarRow], date: DateKey) -> anyhow::Result<f64> {
    rows.binary_search_by(|row| row.date.cmp(&date))
        .map_err(|_| anyhow!("DTCFILE coverage has no matching SOLFSMY calendar date"))
        .and_then(|index| {
            rows.get(index)
                .map(|row| row.jd)
                .ok_or_else(|| anyhow!("SOLFSMY binary search returned invalid row"))
        })
}

/// Verbatim SHA-256 trust root for the approved SET authority bundle.
fn approved_trust_root(
    manifest_bytes: &[u8],
    license_bytes: &[u8],
) -> anyhow::Result<([u8; 32], [u8; 32])> {
    let manifest_digest: [u8; 32] = Sha256::digest(manifest_bytes).into();
    let license_digest: [u8; 32] = Sha256::digest(license_bytes).into();
    if manifest_bytes.len() != APPROVED_MANIFEST_SIZE_BYTES
        || sha256_hex(&manifest_digest) != APPROVED_MANIFEST_SHA256
        || license_bytes.len() != APPROVED_LICENSE_SIZE_BYTES
        || sha256_hex(&license_digest) != APPROVED_LICENSE_SHA256
    {
        return Err(anyhow!(
            "JB2008 manifest or license does not match approved trust root",
        ));
    }
    Ok((manifest_digest, license_digest))
}

fn sha256_hex(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        let high_index = usize::from(*byte >> 4);
        let low_index = usize::from(*byte & 0x0f);
        let Some(&high) = HEX.get(high_index) else {
            return String::new();
        };
        let Some(&low) = HEX.get(low_index) else {
            return String::new();
        };
        output.push(char::from(high));
        output.push(char::from(low));
    }
    output
}

fn gregorian_utc_noon_jd(date: DateKey) -> anyhow::Result<f64> {
    let (month, day) = month_day_from_doy(date.year, date.doy)?;
    let month = i64::from(month);
    let day = i64::from(day);
    let adjustment = 14_i64
        .checked_sub(month)
        .ok_or_else(|| anyhow!("Gregorian month adjustment underflows"))?
        / 12;
    let year = i64::from(date.year)
        .checked_add(4_800)
        .and_then(|value| value.checked_sub(adjustment))
        .ok_or_else(|| anyhow!("Gregorian year overflows"))?;
    let shifted_month = month
        .checked_add(
            12_i64
                .checked_mul(adjustment)
                .ok_or_else(|| anyhow!("Gregorian month adjustment overflows"))?,
        )
        .and_then(|value| value.checked_sub(3))
        .ok_or_else(|| anyhow!("Gregorian shifted month overflows"))?;
    let month_term = 153_i64
        .checked_mul(shifted_month)
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| anyhow!("Gregorian month term overflows"))?
        / 5;
    let year_days = 365_i64
        .checked_mul(year)
        .ok_or_else(|| anyhow!("Gregorian year-day term overflows"))?;
    let jdn = day
        .checked_add(month_term)
        .and_then(|value| value.checked_add(year_days))
        .and_then(|value| value.checked_add(year / 4))
        .and_then(|value| value.checked_sub(year / 100))
        .and_then(|value| value.checked_add(year / 400))
        .and_then(|value| value.checked_sub(32_045))
        .ok_or_else(|| anyhow!("Gregorian Julian day overflows"))?;
    exact_i64_as_f64(jdn, "Gregorian Julian day")
}

fn month_day_from_doy(year: i32, doy: i32) -> anyhow::Result<(i32, i32)> {
    if year < 1 || !(1..=days_in_year(year)).contains(&doy) {
        return Err(anyhow!("calendar date is outside Gregorian range"));
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
                .ok_or_else(|| anyhow!("Gregorian month index overflows"))?;
            return Ok((month, remaining));
        }
        remaining = remaining
            .checked_sub(length)
            .ok_or_else(|| anyhow!("Gregorian day-of-year underflows"))?;
    }
    Err(anyhow!("validated Gregorian day-of-year has no month"))
}

fn iso_date(date: DateKey) -> anyhow::Result<String> {
    let (month, day) = month_day_from_doy(date.year, date.doy)?;
    Ok(format!("{:04}-{month:02}-{day:02}", date.year))
}

fn part_a_v3_date(doy: i32) -> anyhow::Result<DateKey> {
    parse_date("2026", &doy.to_string(), "Part A v3 persistence", 1)
}

fn parse_manifest_f64_bits(value: &str) -> anyhow::Result<u64> {
    let digits = value
        .strip_prefix("0x")
        .filter(|digits| digits.len() == 16)
        .ok_or_else(|| anyhow!("Part A v3 solar source bit string is not canonical"))?;
    let bits = u64::from_str_radix(digits, 16)
        .context("Part A v3 solar source bit string is invalid hexadecimal")?;
    if format!("0x{bits:016x}") != value {
        return Err(anyhow!(
            "Part A v3 solar source bit string is not canonical",
        ));
    }
    Ok(bits)
}

fn validate_part_a_v3_manifest(
    manifest: &PartAV3PersistenceManifest,
    parent: &Jb2008Drivers,
) -> anyhow::Result<[u64; 8]> {
    let parent_identity = parent.identity();
    let solar_count = usize::try_from(PART_A_V3_SOLAR_LAST_DOY - PART_A_V3_SOLAR_FIRST_DOY + 1)
        .context("Part A v3 solar support count is invalid")?;
    let dtc_count = usize::try_from(PART_A_V3_DTC_LAST_DOY - PART_A_V3_DTC_FIRST_DOY + 1)
        .context("Part A v3 DTC support count is invalid")?;
    if manifest.schema != "nasa-dust-part-a-v3-jb2008-persistence-authority-v1"
        || manifest.authority_id != PART_A_V3_AUTHORITY_ID
        || manifest.claim != PART_A_V3_CLAIM
        || manifest.policy != PART_A_V3_POLICY
        || manifest.parent_set_authority.manifest_sha256 != APPROVED_MANIFEST_SHA256
        || manifest.parent_set_authority.manifest_sha256 != parent_identity.manifest_sha256_hex()
        || manifest.parent_set_authority.solfsmy_sha256 != parent_identity.solfsmy_sha256_hex()
        || manifest.parent_set_authority.dtcfile_sha256 != parent_identity.dtcfile_sha256_hex()
        || manifest.parent_set_authority.license_sha256 != APPROVED_LICENSE_SHA256
        || manifest.parent_set_authority.license_sha256 != parent_identity.license_sha256_hex()
        || manifest.parent_set_authority.observed_cutoff_utc_date
            != PART_A_V3_OBSERVED_CUTOFF_UTC_DATE
        || manifest.scenario.t0_utc != PART_A_V3_T0_UTC
        || manifest.scenario.authorized_start_utc != PART_A_V3_AUTHORIZED_START_UTC
        || manifest.scenario.authorized_end_utc != PART_A_V3_AUTHORIZED_END_UTC
        || manifest.scenario.solar_support_first_utc_date != PART_A_V3_SOLAR_SUPPORT_FIRST_UTC_DATE
        || manifest.scenario.solar_support_last_utc_date != PART_A_V3_SOLAR_SUPPORT_LAST_UTC_DATE
        || manifest.scenario.solar_support_record_count != solar_count
        || manifest.scenario.dtc_support_first_utc_date != PART_A_V3_DTC_SUPPORT_FIRST_UTC_DATE
        || manifest.scenario.dtc_support_last_utc_date != PART_A_V3_DTC_SUPPORT_LAST_UTC_DATE
        || manifest.scenario.dtc_support_record_count != dtc_count
        || manifest.scenario.max_input_lag_days != Y10_LAG_DAYS
        || !manifest.scenario.next_day_dtc_required
    {
        return Err(anyhow!(
            "Part A v3 persistence manifest policy or parent authority is invalid",
        ));
    }

    let source_solar = parent
        .solar_rows
        .last()
        .ok_or_else(|| anyhow!("validated parent SOLFSMY has no final row"))?;
    let source_dtc = parent
        .dtc_rows
        .last()
        .ok_or_else(|| anyhow!("validated parent DTCFILE has no final row"))?;
    if iso_date(source_solar.date)? != PART_A_V3_OBSERVED_CUTOFF_UTC_DATE
        || source_solar.date != source_dtc.date
        || source_solar.source != manifest.parent_set_authority.final_solfsmy_source_token
        || manifest.parent_set_authority.final_dtc_hour_utc != 23
        || source_dtc.values.get(23).copied() != Some(manifest.parent_set_authority.final_dtc_value)
    {
        return Err(anyhow!(
            "Part A v3 persistence source row or source claim mismatch",
        ));
    }
    let actual_bits = [
        source_solar.f10.to_bits(),
        source_solar.f10b.to_bits(),
        source_solar.s10.to_bits(),
        source_solar.s10b.to_bits(),
        source_solar.m10.to_bits(),
        source_solar.m10b.to_bits(),
        source_solar.y10.to_bits(),
        source_solar.y10b.to_bits(),
    ];
    let declared_bits: [u64; 8] = manifest
        .parent_set_authority
        .final_solar_fields_bits
        .iter()
        .map(|value| parse_manifest_f64_bits(value))
        .collect::<anyhow::Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow!("Part A v3 solar source bit count is invalid"))?;
    if actual_bits != declared_bits {
        return Err(anyhow!("Part A v3 persistence solar source bits mismatch"));
    }
    Ok(actual_bits)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartAV3PersistenceManifest {
    schema: String,
    authority_id: String,
    claim: String,
    policy: String,
    parent_set_authority: PartAV3ParentSetAuthority,
    scenario: PartAV3PersistenceScenario,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartAV3ParentSetAuthority {
    manifest_sha256: String,
    solfsmy_sha256: String,
    dtcfile_sha256: String,
    license_sha256: String,
    observed_cutoff_utc_date: String,
    final_solfsmy_source_token: String,
    final_solar_fields_bits: [String; 8],
    final_dtc_hour_utc: usize,
    final_dtc_value: i32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartAV3PersistenceScenario {
    t0_utc: String,
    authorized_start_utc: String,
    authorized_end_utc: String,
    solar_support_first_utc_date: String,
    solar_support_last_utc_date: String,
    solar_support_record_count: usize,
    dtc_support_first_utc_date: String,
    dtc_support_last_utc_date: String,
    dtc_support_record_count: usize,
    max_input_lag_days: usize,
    next_day_dtc_required: bool,
}

#[derive(Deserialize)]
struct AuthorityManifest {
    schema: String,
    source: ManifestSource,
    required_catalogue_coverage: RequiredCatalogueCoverage,
    files: BTreeMap<String, ManifestFile>,
    license: ManifestLicense,
}

#[derive(Deserialize)]
struct ManifestSource {
    immutability_policy: String,
}

#[derive(Deserialize)]
struct ManifestFile {
    size_bytes: usize,
    sha256: String,
    #[serde(default)]
    release_header: Option<String>,
    #[serde(default)]
    source_declared_record_count: Option<usize>,
    parsed_coverage: ManifestCoverage,
}

#[derive(Deserialize)]
struct ManifestCoverage {
    first_utc_date: String,
    last_utc_date: String,
    record_count: usize,
}

#[derive(Deserialize)]
struct ManifestLicense {
    acknowledged: bool,
    local_file: String,
    source_url: String,
    size_bytes: usize,
    sha256: String,
}

#[derive(Deserialize)]
struct RequiredCatalogueCoverage {
    start_utc_date: String,
    end_utc_date: String,
    max_input_lag_days: usize,
    effective_driver_start_utc_date: String,
}

fn validate_manifest(
    manifest: &AuthorityManifest,
    drivers: &Jb2008Drivers,
    license_bytes: &[u8],
    license_digest: &[u8; 32],
) -> anyhow::Result<()> {
    if manifest.schema != "jb2008_offline_data_authority_v1"
        || manifest.source.immutability_policy != "verbatim_download_sha256_and_size_locked"
        || !manifest.license.acknowledged
        || manifest.license.local_file != "License.html"
        || manifest.license.source_url != APPROVED_LICENSE_SOURCE_URL
        || manifest.license.size_bytes != license_bytes.len()
        || manifest.license.sha256 != sha256_hex(license_digest)
        || manifest.required_catalogue_coverage.start_utc_date != "2021-11-09"
        || manifest.required_catalogue_coverage.end_utc_date != "2022-11-11"
        || manifest.required_catalogue_coverage.max_input_lag_days != Y10_LAG_DAYS
        || manifest
            .required_catalogue_coverage
            .effective_driver_start_utc_date
            != "2021-11-04"
    {
        return Err(anyhow!(
            "JB2008 authority manifest policy or license acknowledgement is invalid",
        ));
    }
    let sol = manifest
        .files
        .get("SOLFSMY.TXT")
        .ok_or_else(|| anyhow!("JB2008 authority manifest misses SOLFSMY.TXT"))?;
    let dtc = manifest
        .files
        .get("DTCFILE.TXT")
        .ok_or_else(|| anyhow!("JB2008 authority manifest misses DTCFILE.TXT"))?;
    validate_manifest_file(
        sol,
        &drivers.identity.solfsmy_sha256_hex(),
        drivers
            .identity
            .solfsmy_release_header
            .trim_start_matches("# "),
        drivers.identity.solfsmy_source_size_bytes,
        &drivers.solar_rows,
        "SOLFSMY",
    )?;
    validate_manifest_file(
        dtc,
        &drivers.identity.dtcfile_sha256_hex(),
        "",
        drivers.identity.dtcfile_source_size_bytes,
        &drivers.dtc_rows,
        "DTCFILE",
    )?;
    Ok(())
}

fn validate_manifest_file<T>(
    file: &ManifestFile,
    actual_hash: &str,
    actual_header: &str,
    actual_size_bytes: usize,
    rows: &[T],
    name: &str,
) -> anyhow::Result<()>
where
    T: HasDate,
{
    let first = rows.first().ok_or_else(|| anyhow!("parsed rows missing"))?;
    let last = rows.last().ok_or_else(|| anyhow!("parsed rows missing"))?;
    if file.size_bytes != actual_size_bytes
        || file.sha256 != actual_hash
        || file.parsed_coverage.first_utc_date != iso_date(first.date())?
        || file.parsed_coverage.last_utc_date != iso_date(last.date())?
        || file.parsed_coverage.record_count != rows.len()
    {
        return Err(anyhow!("{name} manifest hash or coverage mismatch"));
    }
    if let Some(header) = &file.release_header {
        if header != actual_header {
            return Err(anyhow!("SOLFSMY manifest release header mismatch"));
        }
    }
    Ok(())
}

trait HasDate {
    fn date(&self) -> DateKey;
}

impl HasDate for SolarRow {
    fn date(&self) -> DateKey {
        self.date
    }
}

impl HasDate for DtcRow {
    fn date(&self) -> DateKey {
        self.date
    }
}

#[cfg(test)]
mod tests {

    /// The ISO, JD and DOY forms of the authority window must be one fact.
    ///
    /// They are three parallel constants. Manifest validation compares strings
    /// and counts, support rows derive from DOYs, and runtime authorization
    /// uses JDs, so nothing required them to agree -- three independent
    /// projections of a window, any one of which could be edited alone.
    ///
    /// Derived here from the ISO strings by ordinary calendar arithmetic, which
    /// is an oracle independent of the sealed time chain rather than a
    /// restatement of it.
    #[test]
    fn iso_jd_and_doy_projections_of_the_authority_window_agree() {
        /// Julian Day Number for a UTC calendar date, by Fliegel-Van Flandern.
        ///
        /// Pure integer arithmetic, so the day count is exact and the only
        /// rounding anywhere in this test is the time-of-day fraction added
        /// below.
        fn julian_day_number(y: i64, m: i64, d: i64) -> i64 {
            let a = (14 - m) / 12;
            let yy = y + 4800 - a;
            let mm = m + 12 * a - 3;
            d + (153 * mm + 2) / 5 + 365 * yy + yy / 4 - yy / 100 + yy / 400 - 32_045
        }

        /// Julian Date at a UTC calendar instant.
        ///
        /// `i32` then `f64::from`, never `as`: both conversions are exact and
        /// the workspace denies silent `as`. A JDN near 2.46e6 and a
        /// seconds-of-day below 86401 both fit `i32` with room to spare, and
        /// `try_from` turns "with room to spare" into something checked.
        fn julian_date(y: i64, m: i64, d: i64, hh: i64, mi: i64, ss: i64) -> f64 {
            let day = f64::from(i32::try_from(julian_day_number(y, m, d)).expect("JDN fits i32"));
            let seconds = f64::from(
                i32::try_from(hh * 3600 + mi * 60 + ss).expect("seconds-of-day fits i32"),
            );
            day - 0.5 + seconds / 86_400.0
        }

        /// Day of year for a UTC calendar date, as a difference of day numbers.
        fn day_of_year(y: i64, m: i64, d: i64) -> i32 {
            let elapsed = julian_day_number(y, m, d) - julian_day_number(y, 1, 1);
            i32::try_from(elapsed + 1).expect("a DOY fits i32")
        }

        // ISO -> JD, exactly. These are the values runtime authorizes against.
        assert_eq!(
            julian_date(2026, 8, 15, 11, 24, 29).to_bits(),
            PART_A_V3_AUTHORIZED_START_JD.to_bits(),
            "authorized start JD does not match {PART_A_V3_AUTHORIZED_START_UTC}"
        );
        assert_eq!(
            julian_date(2026, 8, 17, 17, 24, 29).to_bits(),
            PART_A_V3_T0_JD.to_bits(),
            "t0 JD does not match {PART_A_V3_T0_UTC}"
        );
        assert_eq!(
            julian_date(2026, 8, 31, 17, 24, 29).to_bits(),
            PART_A_V3_AUTHORIZED_END_JD.to_bits(),
            "authorized end JD does not match {PART_A_V3_AUTHORIZED_END_UTC}"
        );

        // ISO -> DOY, exactly. These are the values the support rows index by.
        assert_eq!(day_of_year(2026, 8, 10), PART_A_V3_SOLAR_FIRST_DOY);
        assert_eq!(day_of_year(2026, 9, 1), PART_A_V3_SOLAR_LAST_DOY);
        assert_eq!(day_of_year(2026, 8, 15), PART_A_V3_DTC_FIRST_DOY);
        assert_eq!(day_of_year(2026, 9, 1), PART_A_V3_DTC_LAST_DOY);
    }
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;

    /// The build.rs-baked row tables against the kept runtime parser, bit for
    /// bit. build.rs carries a MIRROR of `parse_solfsmy`/`parse_dtcfile`; this
    /// oracle re-runs the runtime parser over the same tracked bytes so drift
    /// between the two parsers is a red test, not a silent skew in the
    /// compiled authority.
    #[test]
    fn baked_rows_are_bit_identical_to_the_parser() -> anyhow::Result<()> {
        let parsed = Jb2008Drivers::from_set_bytes(COMPILED_SOLFSMY, COMPILED_DTCFILE)?;
        let baked = Jb2008Drivers::from_baked_compiled_rows(COMPILED_SOLFSMY, COMPILED_DTCFILE)?;

        anyhow::ensure!(
            parsed.identity.solfsmy_release_header == baked.identity.solfsmy_release_header,
            "release header"
        );
        anyhow::ensure!(parsed.identity == baked.identity, "identity");
        anyhow::ensure!(
            parsed.solar_rows.len() == baked.solar_rows.len(),
            "solar row count"
        );
        for (index, (left, right)) in parsed
            .solar_rows
            .iter()
            .zip(baked.solar_rows.iter())
            .enumerate()
        {
            anyhow::ensure!(left.date == right.date, "solar row {index} date");
            let pairs = [
                (left.jd, right.jd),
                (left.f10, right.f10),
                (left.f10b, right.f10b),
                (left.s10, right.s10),
                (left.s10b, right.s10b),
                (left.m10, right.m10),
                (left.m10b, right.m10b),
                (left.y10, right.y10),
                (left.y10b, right.y10b),
            ];
            anyhow::ensure!(
                pairs
                    .iter()
                    .all(|(runtime, baked)| runtime.to_bits() == baked.to_bits()),
                "solar row {index} field bits"
            );
            anyhow::ensure!(left.source == right.source, "solar row {index} source");
        }
        anyhow::ensure!(
            parsed.dtc_rows.len() == baked.dtc_rows.len(),
            "dtc row count"
        );
        for (index, (left, right)) in parsed
            .dtc_rows
            .iter()
            .zip(baked.dtc_rows.iter())
            .enumerate()
        {
            anyhow::ensure!(left.date == right.date, "dtc row {index} date");
            anyhow::ensure!(left.values == right.values, "dtc row {index} values");
        }
        // Non-vacuity: the compiled catalogue is tens of thousands of rows.
        anyhow::ensure!(parsed.solar_rows.len() > 1_000, "solar corpus size");
        anyhow::ensure!(parsed.dtc_rows.len() > 1_000, "dtc corpus size");
        Ok(())
    }

    #[test]
    fn owned_driver_validation_uses_anyhow_result() {
        fn assert_anyhow_result(_: anyhow::Result<UtcJulianDay>) {}

        assert_anyhow_result(UtcJulianDay::new(2_459_600.5));
    }

    #[test]
    fn invalid_calendar_year_preserves_parse_source() {
        let error =
            parse_date("not-a-year", "1", "SOLFSMY", 17).expect_err("malformed year must fail");

        assert_eq!(error.to_string(), "SOLFSMY line 17 has invalid year");
        assert!(error.downcast_ref::<std::num::ParseIntError>().is_some());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn oversized_exact_index_preserves_conversion_source() {
        let error = exact_usize_as_f64(usize::MAX, "solar row")
            .expect_err("usize::MAX exceeds exact u32-backed index range");

        assert_eq!(error.to_string(), "solar row exceeds exact f64 index range");
        assert!(error.downcast_ref::<std::num::TryFromIntError>().is_some());
    }

    #[test]
    fn success_only_cache_retries_failure_and_reuses_success() -> anyhow::Result<()> {
        let cache: OnceLock<Arc<u8>> = OnceLock::new();
        let lock = std::sync::Mutex::new(());
        let attempts = AtomicUsize::new(0);

        let first = success_only_cached(&cache, &lock, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!("first validation fails")
        });
        anyhow::ensure!(first.is_err());
        anyhow::ensure!(cache.get().is_none());

        let second = success_only_cached(&cache, &lock, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok(7)
        })?;
        let third = success_only_cached(&cache, &lock, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok(9)
        })?;

        anyhow::ensure!(attempts.load(Ordering::Relaxed) == 2);
        anyhow::ensure!(Arc::ptr_eq(&second, &third));
        anyhow::ensure!(*third == 7);
        Ok(())
    }

    #[test]
    fn part_a_v3_provider_is_persistent_cached_and_arc_bounded() -> anyhow::Result<()> {
        let first = compiled_part_a_v3_drivers()?;
        let second = compiled_part_a_v3_drivers()?;
        anyhow::ensure!(Arc::ptr_eq(&first, &second));

        let start = UtcJulianDay::new(PART_A_V3_AUTHORIZED_START_JD)?;
        let middle = UtcJulianDay::new(PART_A_V3_T0_JD)?;
        let end = UtcJulianDay::new(PART_A_V3_AUTHORIZED_END_JD)?;
        let start_input = first.lookup_utc_jd(start)?;
        anyhow::ensure!(start_input == first.lookup_utc_jd(middle)?);
        anyhow::ensure!(start_input == first.lookup_utc_jd(end)?);
        anyhow::ensure!(start_input.dtcval == 50);
        first.validate_utc_arc(start, end)?;

        anyhow::ensure!(first
            .lookup_utc_jd(UtcJulianDay::new(PART_A_V3_AUTHORIZED_START_JD - 1.0e-6)?)
            .is_err());
        anyhow::ensure!(first
            .lookup_utc_jd(UtcJulianDay::new(PART_A_V3_AUTHORIZED_END_JD + 1.0e-6)?)
            .is_err());
        anyhow::ensure!(first
            .validate_utc_arc(
                start,
                UtcJulianDay::new(PART_A_V3_AUTHORIZED_END_JD + 1.0e-6)?,
            )
            .is_err());
        Ok(())
    }

    #[test]
    fn part_a_v3_manifest_binds_claim_parent_and_source_bits() -> anyhow::Result<()> {
        let parent = compiled_drivers()?;
        let manifest: PartAV3PersistenceManifest =
            serde_json::from_slice(COMPILED_PART_A_V3_MANIFEST)?;
        let source_bits = validate_part_a_v3_manifest(&manifest, &parent)?;
        anyhow::ensure!(source_bits == compiled_part_a_v3_identity()?.source_solar_fields_bits);

        let mut wrong_claim: PartAV3PersistenceManifest =
            serde_json::from_slice(COMPILED_PART_A_V3_MANIFEST)?;
        wrong_claim.claim.push('x');
        anyhow::ensure!(validate_part_a_v3_manifest(&wrong_claim, &parent).is_err());

        let mut wrong_source: PartAV3PersistenceManifest =
            serde_json::from_slice(COMPILED_PART_A_V3_MANIFEST)?;
        wrong_source.parent_set_authority.final_solar_fields_bits[0] =
            "0x0000000000000000".to_owned();
        anyhow::ensure!(validate_part_a_v3_manifest(&wrong_source, &parent).is_err());
        Ok(())
    }
}
