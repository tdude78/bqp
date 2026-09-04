//! Precomputed planetary ephemeris for fast position lookups.
//!
//! This module provides ~700x faster ephemeris queries compared to ANISE by using
//! precomputed binary catalogues with linear interpolation.
//!
//! # Binary Format
//!
//! Each catalogue file has:
//! - Header (48 bytes): magic, version, `n_samples`, `jd_start`, `jd_end`,
//!   `dt_days`, `body_id`
//! - Data: `n_samples × 3 × f64` Earth-centered geometric positions in km,
//!   aligned to ICRS axes.
//!
//! # Usage
//!
//! ```ignore
//! use lightyear_odeint_rs::precomputed_ephem::{
//!     get_precomputed_ephemeris, load_precomputed_ephemeris, Body,
//! };
//!
//! // Load catalogues (automatically finds them in search paths)
//! load_precomputed_ephemeris(FORCE_SUN | FORCE_MOON)?;
//!
//! // Get position (~10ns). `utc` is a `jb_rs::drivers::UtcJulianDay`; the
//! // manifest-bound Part A authority is the production query API.
//! if let Some(table) = get_precomputed_ephemeris().as_deref().and_then(|e| e.get(Body::Sun)) {
//!     let pos = table.position_at_part_a_utc_jd(utc)?;
//!     println!("Sun: {:?}", pos);
//! }
//! ```

use anyhow::Context;
use num_traits::ToPrimitive;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fmt;
// Nothing in this module opens a file by handle: every production path goes
// through `std::fs::read` or `include_bytes!`. A `#[cfg(test)]`-only
// `load_mmap` used to sit alongside `load_buffered`, carrying the module's only
// I/O `unsafe` (`memmap2::Mmap::map` is unsafe because the mapping aliases a
// file another process may truncate or rewrite under it, which is UB the type
// system cannot see). Its one caller was a test asserting it agreed with the
// loader production actually uses, so it bought nothing outside tests and is
// deleted rather than justified.
use std::borrow::Cow;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, RwLock};

/// Binary format constants
const MAGIC: &[u8; 4] = b"DUST";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 48;
static LEGACY_UNTAGGED_CATALOGUE_NOTICE: AtomicBool = AtomicBool::new(false);

/// Consume the process-level legacy-catalogue notice at a quiescent owner seam.
/// Loaders only latch this bit; they never perform worker-reachable I/O.
#[must_use]
pub fn take_legacy_untagged_catalogue_notice() -> bool {
    LEGACY_UNTAGGED_CATALOGUE_NOTICE.swap(false, Ordering::AcqRel)
}

fn read_header_array<const LENGTH: usize>(
    header: &[u8],
    range: std::ops::Range<usize>,
    field: &str,
) -> io::Result<[u8; LENGTH]> {
    let bytes = header.get(range).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("precomputed ephemeris header is missing {field}"),
        )
    })?;
    bytes.try_into().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("precomputed ephemeris header has malformed {field}"),
        )
    })
}

fn read_header_byte(header: &[u8], index: usize, field: &str) -> io::Result<u8> {
    header.get(index).copied().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("precomputed ephemeris header is missing {field}"),
        )
    })
}

fn usize_to_exact_f64(value: usize) -> io::Result<f64> {
    value
        .to_f64()
        .filter(|converted| converted.to_usize() == Some(value))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "precomputed ephemeris sample count is not exactly representable as f64",
            )
        })
}

fn lowercase_hex(bytes: &[u8]) -> String {
    let mut encoded = String::new();
    for &byte in bytes {
        for nibble in [byte >> 4, byte & 0x0f] {
            let digit = match nibble {
                0 => '0',
                1 => '1',
                2 => '2',
                3 => '3',
                4 => '4',
                5 => '5',
                6 => '6',
                7 => '7',
                8 => '8',
                9 => '9',
                10 => 'a',
                11 => 'b',
                12 => 'c',
                13 => 'd',
                14 => 'e',
                15 => 'f',
                _ => '\0',
            };
            encoded.push(digit);
        }
    }
    encoded
}

/// Tracked default catalogues. The `.bin` payloads are no longer embedded raw:
/// `build.rs` parses, validates, and bakes them into the static tables in
/// [`embedded_tables`] (positions as `from_bits` literals, header consts, the
/// direction-rate supremum, and the SHA-256 identities), so the per-process
/// decode, double scan, double hash, and ~1 MB heap duplicate are gone. The
/// raw bytes remain compiled into TEST binaries only, as the oracle input for
/// `generated_embedded_tables_are_bit_identical_to_the_parser`.
static EMBEDDED_MANIFEST: &[u8] = include_bytes!("../data/ephemeris/manifest.json");
#[cfg(test)]
static EMBEDDED_SUN: &[u8] = include_bytes!("../data/ephemeris/sun.bin");
#[cfg(test)]
static EMBEDDED_MOON: &[u8] = include_bytes!("../data/ephemeris/moon.bin");
#[cfg(test)]
static EMBEDDED_JUPITER: &[u8] = include_bytes!("../data/ephemeris/jupiter.bin");
#[cfg(test)]
static EMBEDDED_VENUS: &[u8] = include_bytes!("../data/ephemeris/venus.bin");

/// Build-time-baked catalogue tables emitted by `build.rs`.
mod embedded_tables {
    include!(concat!(env!("OUT_DIR"), "/ephemeris_tables.rs"));
}

/// One baked catalogue's fields, in the shape [`PrecomputedEphemeris`] loads.
struct EmbeddedTable {
    positions: &'static [f64],
    n_samples: usize,
    jd_start: f64,
    jd_end: f64,
    dt_days: f64,
    max_direction_rate_per_day: f64,
    epoch_scale_tag: u8,
    epoch_representation_tag: u8,
    body_id: u8,
    size_bytes: usize,
    content_sha256: [u8; 32],
    content_sha256_hex: &'static str,
}

const fn embedded_table(body: Body) -> Option<EmbeddedTable> {
    use embedded_tables as t;
    match body {
        Body::Sun => Some(EmbeddedTable {
            positions: &t::SUN_POSITIONS,
            n_samples: t::SUN_N_SAMPLES,
            jd_start: t::SUN_JD_START,
            jd_end: t::SUN_JD_END,
            dt_days: t::SUN_DT_DAYS,
            max_direction_rate_per_day: t::SUN_MAX_DIRECTION_RATE_PER_DAY,
            epoch_scale_tag: t::SUN_EPOCH_SCALE_TAG,
            epoch_representation_tag: t::SUN_EPOCH_REPRESENTATION_TAG,
            body_id: t::SUN_BODY_ID,
            size_bytes: t::SUN_SIZE_BYTES,
            content_sha256: t::SUN_CONTENT_SHA256,
            content_sha256_hex: t::SUN_CONTENT_SHA256_HEX,
        }),
        Body::Moon => Some(EmbeddedTable {
            positions: &t::MOON_POSITIONS,
            n_samples: t::MOON_N_SAMPLES,
            jd_start: t::MOON_JD_START,
            jd_end: t::MOON_JD_END,
            dt_days: t::MOON_DT_DAYS,
            max_direction_rate_per_day: t::MOON_MAX_DIRECTION_RATE_PER_DAY,
            epoch_scale_tag: t::MOON_EPOCH_SCALE_TAG,
            epoch_representation_tag: t::MOON_EPOCH_REPRESENTATION_TAG,
            body_id: t::MOON_BODY_ID,
            size_bytes: t::MOON_SIZE_BYTES,
            content_sha256: t::MOON_CONTENT_SHA256,
            content_sha256_hex: t::MOON_CONTENT_SHA256_HEX,
        }),
        Body::Jupiter => Some(EmbeddedTable {
            positions: &t::JUPITER_POSITIONS,
            n_samples: t::JUPITER_N_SAMPLES,
            jd_start: t::JUPITER_JD_START,
            jd_end: t::JUPITER_JD_END,
            dt_days: t::JUPITER_DT_DAYS,
            max_direction_rate_per_day: t::JUPITER_MAX_DIRECTION_RATE_PER_DAY,
            epoch_scale_tag: t::JUPITER_EPOCH_SCALE_TAG,
            epoch_representation_tag: t::JUPITER_EPOCH_REPRESENTATION_TAG,
            body_id: t::JUPITER_BODY_ID,
            size_bytes: t::JUPITER_SIZE_BYTES,
            content_sha256: t::JUPITER_CONTENT_SHA256,
            content_sha256_hex: t::JUPITER_CONTENT_SHA256_HEX,
        }),
        Body::Venus => Some(EmbeddedTable {
            positions: &t::VENUS_POSITIONS,
            n_samples: t::VENUS_N_SAMPLES,
            jd_start: t::VENUS_JD_START,
            jd_end: t::VENUS_JD_END,
            dt_days: t::VENUS_DT_DAYS,
            max_direction_rate_per_day: t::VENUS_MAX_DIRECTION_RATE_PER_DAY,
            epoch_scale_tag: t::VENUS_EPOCH_SCALE_TAG,
            epoch_representation_tag: t::VENUS_EPOCH_REPRESENTATION_TAG,
            body_id: t::VENUS_BODY_ID,
            size_bytes: t::VENUS_SIZE_BYTES,
            content_sha256: t::VENUS_CONTENT_SHA256,
            content_sha256_hex: t::VENUS_CONTENT_SHA256_HEX,
        }),
        Body::Mars | Body::Saturn => None,
    }
}

/// Body IDs (must match the binary catalogue's body IDs)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Body {
    Sun = 0,
    Moon = 1,
    Jupiter = 2,
    Venus = 3,
    Mars = 4,
    Saturn = 5,
}

impl Body {
    /// Get filename for this body's catalogue
    #[must_use]
    pub const fn filename(self) -> &'static str {
        match self {
            Self::Sun => "sun.bin",
            Self::Moon => "moon.bin",
            Self::Jupiter => "jupiter.bin",
            Self::Venus => "venus.bin",
            Self::Mars => "mars.bin",
            Self::Saturn => "saturn.bin",
        }
    }

    /// Stable lowercase body name for diagnostics and error messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sun => "sun",
            Self::Moon => "moon",
            Self::Jupiter => "jupiter",
            Self::Venus => "venus",
            Self::Mars => "mars",
            Self::Saturn => "saturn",
        }
    }

    /// Get force flag bit for this body
    #[must_use]
    pub(crate) const fn force_flag(self) -> i32 {
        match self {
            Self::Sun => 4,      // DUST_FORCE_SUN
            Self::Moon => 8,     // DUST_FORCE_MOON
            Self::Jupiter => 16, // DUST_FORCE_JUPITER
            Self::Venus => 32,   // DUST_FORCE_VENUS
            Self::Mars => 64,    // DUST_FORCE_MARS
            Self::Saturn => 128, // DUST_FORCE_SATURN
        }
    }

    /// All supported bodies
    pub const ALL: [Self; 6] = [
        Self::Sun,
        Self::Moon,
        Self::Jupiter,
        Self::Venus,
        Self::Mars,
        Self::Saturn,
    ];

    /// Bodies we generate catalogues for by default
    pub const DEFAULT: [Self; 4] = [Self::Sun, Self::Moon, Self::Jupiter, Self::Venus];

    #[must_use]
    const fn id(self) -> u8 {
        match self {
            Self::Sun => 0,
            Self::Moon => 1,
            Self::Jupiter => 2,
            Self::Venus => 3,
            Self::Mars => 4,
            Self::Saturn => 5,
        }
    }
}

/// Time scale of a catalogue's independent variable, header byte 41.
///
/// The scale was previously recorded ONLY in `data/ephemeris/manifest.json`
/// (`"epoch_scale": "utc"`), which no code path opens: the `.bin` files are
/// `include_bytes!`'d and only the 48-byte header is parsed. A caller could
/// therefore hand a TT Julian Day to a UTC-indexed grid and nothing complained.
/// That is not a rounding concern — at the Part A `TT - UTC` of 69.184 s it
/// displaces the Moon by 70.8 km, above the 39.7 km the grid's own linear
/// interpolation already costs mid-interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EpochScale {
    /// Written by a producer predating the tag. NEVER treated as any scale.
    Unspecified = 0x00,
    Utc = 0x01,
    Tai = 0x02,
    Tt = 0x03,
    Tdb = 0x04,
}

impl EpochScale {
    const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0x00 => Some(Self::Unspecified),
            0x01 => Some(Self::Utc),
            0x02 => Some(Self::Tai),
            0x03 => Some(Self::Tt),
            0x04 => Some(Self::Tdb),
            _ => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::Utc => "UTC",
            Self::Tai => "TAI",
            Self::Tt => "TT",
            Self::Tdb => "TDB",
        }
    }
}

/// How a catalogue's independent variable is expressed, header byte 42.
///
/// A scale tag alone is not sufficient. The analytic packs are indexed by
/// astronomical Julian Date; a DE440s-derived pack is indexed by seconds past
/// J2000. Both can legitimately be TDB, so a one-byte scale tag would let a
/// TDB-seconds catalogue satisfy a TDB-Julian-Date query — the same class of
/// defect as the one the scale tag exists to close, one level down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EpochRepresentation {
    /// Written by a producer predating the tag. NEVER treated as any encoding.
    Unspecified = 0x00,
    /// Astronomical Julian Date, e.g. 2458849.5.
    JulianDate = 0x01,
    /// Seconds past the J2000 epoch.
    SecondsPastJ2000 = 0x02,
}

impl EpochRepresentation {
    const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0x00 => Some(Self::Unspecified),
            0x01 => Some(Self::JulianDate),
            0x02 => Some(Self::SecondsPastJ2000),
            _ => None,
        }
    }
}

/// Why a Part A ephemeris query was refused.
///
/// Every variant is a refusal, never a fallback: there is deliberately no
/// "assume the catalogue is fine" path.
// No `Eq`: `OutOfRange` carries `f64`. The comparison callers need is
// `PartialEq` for assertions and matching, which f64 provides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpochScaleError {
    /// Catalogue bytes were not admitted by the compiled Part A manifest.
    PartAManifestAuthorityRequired,
    /// The catalogue was admitted; the value falls outside the sampled grid.
    OutOfRange { value: f64, start: f64, end: f64 },
}

impl fmt::Display for EpochScaleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PartAManifestAuthorityRequired => write!(
                formatter,
                "catalogue is not bound to the compiled Part A UTC-JD ephemeris manifest"
            ),
            Self::OutOfRange { value, start, end } => write!(
                formatter,
                "epoch {value} outside catalogue coverage [{start}, {end}]"
            ),
        }
    }
}

impl std::error::Error for EpochScaleError {}

/// Typed failure returned before constructing an RHS with dynamic ephemeris.
#[derive(Debug, Clone, PartialEq)]
pub enum EphemerisCoverageError {
    NonFiniteArc {
        jd_a: f64,
        jd_b: f64,
    },
    CatalogueLoad {
        requested_flags: i32,
        message: String,
    },
    CatalogueLoadSource {
        requested_flags: i32,
        message: String,
        cause: EphemerisCause,
    },
    MissingBody {
        body: Body,
    },
    InvalidCatalogue {
        body: Body,
    },
    OutsideRange {
        body: Body,
        required_start: f64,
        required_end: f64,
        available_start: f64,
        available_end: f64,
    },
}

/// Cloneable owner for a catalogue failure's concrete error chain.
#[derive(Debug, Clone)]
pub struct EphemerisCause(Arc<anyhow::Error>);

impl PartialEq for EphemerisCause {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || format!("{:#}", self.0) == format!("{:#}", other.0)
    }
}

impl EphemerisCoverageError {
    pub(crate) fn catalogue_source(
        requested_flags: i32,
        message: String,
        cause: anyhow::Error,
    ) -> Self {
        Self::CatalogueLoadSource {
            requested_flags,
            message,
            cause: EphemerisCause(Arc::new(cause)),
        }
    }

    #[cfg(test)]
    fn catalogue_io(requested_flags: i32, context: &str, cause: io::Error) -> Self {
        Self::catalogue_source(requested_flags, format!("{context}: {cause}"), cause.into())
    }
}

impl fmt::Display for EphemerisCoverageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteArc { jd_a, jd_b } => write!(
                formatter,
                "dynamic ephemeris arc endpoints must be finite (jd_a={jd_a:?}, jd_b={jd_b:?})"
            ),
            Self::CatalogueLoad {
                requested_flags,
                message,
            }
            | Self::CatalogueLoadSource {
                requested_flags,
                message,
                ..
            } => write!(
                formatter,
                "failed loading dynamic ephemeris catalogues for flags {requested_flags}: {message}"
            ),
            Self::MissingBody { body } => write!(
                formatter,
                "dynamic ephemeris catalogue missing for requested body {}",
                body.name()
            ),
            Self::InvalidCatalogue { body } => write!(
                formatter,
                "dynamic ephemeris catalogue for {} has no usable samples",
                body.name()
            ),
            Self::OutsideRange {
                body,
                required_start,
                required_end,
                available_start,
                available_end,
            } => write!(
                formatter,
                "dynamic ephemeris coverage failure for {}: required [{required_start:.12}, {required_end:.12}], available [{available_start:.12}, {available_end:.12}]",
                body.name()
            ),
        }
    }
}

impl std::error::Error for EphemerisCoverageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CatalogueLoadSource { cause, .. } => Some(cause.0.as_ref().as_ref()),
            _ => None,
        }
    }
}

/// Precomputed ephemeris for a single body.
#[derive(Debug, Clone)]
pub struct PrecomputedEphemeris {
    /// Start Julian date
    jd_start: f64,
    /// Time step in days
    dt_days: f64,
    /// Precomputed 1/dt for fast interpolation
    inv_dt: f64,
    /// Number of samples
    n_samples: usize,
    /// Earth-centered geometric positions in km, ICRS axes. Borrowed from the
    /// build-time-baked static tables on the embedded path (zero-copy,
    /// zero-decode); owned on the directory-override loader path.
    positions: Cow<'static, [f64]>,
    /// Immutable SHA-256 of exact validated binary catalogue bytes.
    content_sha256: [u8; 32],
    /// Time scale of `jd_start`/`dt_days`, from header byte 41.
    epoch_scale: EpochScale,
    /// Encoding of `jd_start`/`dt_days`, from header byte 42.
    epoch_representation: EpochRepresentation,
    /// Exact embedded bytes admitted by the compiled UTC-JD manifest.
    part_a_utc_manifest_authorized: bool,
    /// Supremum over the whole grid of the angular rate of the NORMALIZED
    /// interpolated direction, in radians per day. See
    /// [`Self::max_normalized_direction_rate_per_day`] for the derivation.
    max_direction_rate_per_day: f64,
}

impl PrecomputedEphemeris {
    /// Load a catalogue from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or does not satisfy the
    /// binary catalogue format.
    pub fn load(path: &Path) -> io::Result<Self> {
        Self::load_buffered(path)
    }

    fn load_buffered(path: &Path) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::load_bytes(&bytes, Self::body_for_filename(path))
    }

    fn load_embedded(body: Body) -> io::Result<Self> {
        part_a_ephemeris_authority()?;
        let table = embedded_table(body).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "no embedded precomputed ephemeris catalogue for {}",
                    body.name()
                ),
            )
        })?;
        // The baked fields were validated and derived by build.rs running the
        // same checks and the same arithmetic as `from_header_and_position_bytes`
        // over the same tracked bytes; the
        // `generated_embedded_tables_are_bit_identical_to_the_parser` oracle
        // bit-compares the two. Tag decoding and the legacy-notice latch are
        // kept here because they are runtime behavior, not derived data.
        let epoch_scale = EpochScale::from_tag(table.epoch_scale_tag).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "baked {} catalogue declares unknown epoch scale tag {:#04x}",
                    body.name(),
                    table.epoch_scale_tag
                ),
            )
        })?;
        let epoch_representation = EpochRepresentation::from_tag(table.epoch_representation_tag)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "baked {} catalogue declares unknown epoch representation tag {:#04x}",
                        body.name(),
                        table.epoch_representation_tag
                    ),
                )
            })?;
        if epoch_scale == EpochScale::Unspecified
            || epoch_representation == EpochRepresentation::Unspecified
        {
            LEGACY_UNTAGGED_CATALOGUE_NOTICE.store(true, Ordering::Release);
        }
        let dt_days = table.dt_days;
        // Same expression as the parser path so the stored value is
        // bit-identical: dt_days > 0 was validated at build time.
        let inv_dt = if dt_days > 0.0 { 1.0 / dt_days } else { 0.0 };
        Ok(Self {
            jd_start: table.jd_start,
            dt_days,
            inv_dt,
            n_samples: table.n_samples,
            positions: Cow::Borrowed(table.positions),
            content_sha256: table.content_sha256,
            epoch_scale,
            epoch_representation,
            part_a_utc_manifest_authorized: true,
            max_direction_rate_per_day: table.max_direction_rate_per_day,
        })
    }

    fn load_bytes(bytes: &[u8], expected_body: Option<Body>) -> io::Result<Self> {
        let content_sha256: [u8; 32] = Sha256::digest(bytes).into();
        let header = bytes.get(..HEADER_SIZE).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "precomputed ephemeris file is smaller than header",
            )
        })?;
        let data = bytes.get(HEADER_SIZE..).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "precomputed ephemeris file is smaller than header",
            )
        })?;
        Self::from_header_and_position_bytes(header, data, expected_body, content_sha256)
    }

    fn body_for_filename(path: &Path) -> Option<Body> {
        Body::ALL
            .into_iter()
            .find(|body| path.file_name().and_then(|name| name.to_str()) == Some(body.filename()))
    }

    fn from_header_and_position_bytes(
        header: &[u8],
        data: &[u8],
        expected_body: Option<Body>,
        content_sha256: [u8; 32],
    ) -> io::Result<Self> {
        if header.len() != HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "precomputed ephemeris header has an invalid length",
            ));
        }

        let magic = read_header_array::<4>(header, 0..4, "magic")?;
        if magic != *MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid magic: {magic:?}"),
            ));
        }

        let version = u32::from_le_bytes(read_header_array::<4>(header, 4..8, "version")?);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported version: {version}"),
            ));
        }

        let n_samples_u64 =
            u64::from_le_bytes(read_header_array::<8>(header, 8..16, "sample count")?);
        let n_samples = usize::try_from(n_samples_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "precomputed ephemeris sample count exceeds platform usize",
            )
        })?;
        let jd_start = f64::from_le_bytes(read_header_array::<8>(header, 16..24, "JD start")?);
        let jd_end_header = f64::from_le_bytes(read_header_array::<8>(header, 24..32, "JD end")?);
        let dt_days = f64::from_le_bytes(read_header_array::<8>(header, 32..40, "step")?);
        let body_id = read_header_byte(header, 40, "body ID")?;

        if n_samples == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "precomputed ephemeris catalogue must contain at least one sample",
            ));
        }
        if !jd_start.is_finite()
            || !jd_end_header.is_finite()
            || !dt_days.is_finite()
            || dt_days <= 0.0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "precomputed ephemeris catalogue requires finite JD start and positive finite step",
            ));
        }
        let interval_count = n_samples.checked_sub(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "precomputed ephemeris catalogue must contain at least one sample",
            )
        })?;
        let interval_count = usize_to_exact_f64(interval_count)?;
        // Keep producer multiply-then-add rounding: this value is bit-compared
        // against the sealed catalogue header.
        let computed_jd_end = jd_start + interval_count * dt_days;
        if jd_end_header.to_bits() != computed_jd_end.to_bits() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "precomputed ephemeris declared JD end disagrees with sample grid",
            ));
        }
        if let Some(expected) = expected_body {
            if body_id != expected.id() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "precomputed ephemeris body_id disagrees with filename",
                ));
            }
        }
        // Byte 41 is the epoch scale, byte 42 the epoch representation. An
        // UNKNOWN tag is rejected rather than ignored: it means the file was
        // written by a producer this loader cannot interpret, and guessing is
        // exactly the failure being closed. Tag 0x00 on either byte is accepted
        // at LOAD — rejecting it here would brick the sealed catalogues, whose
        // reserved bytes are zero, on the commit that introduces this parsing.
        // It is refused at the QUERY boundary instead; see `position_at`.
        let epoch_scale_tag = read_header_byte(header, 41, "epoch scale tag")?;
        let epoch_scale = EpochScale::from_tag(epoch_scale_tag).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "precomputed ephemeris declares unknown epoch scale tag {epoch_scale_tag:#04x}"
                ),
            )
        })?;
        let epoch_representation_tag = read_header_byte(header, 42, "epoch representation tag")?;
        let epoch_representation = EpochRepresentation::from_tag(epoch_representation_tag)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "precomputed ephemeris declares unknown epoch representation tag \
                         {epoch_representation_tag:#04x}"
                    ),
                )
            })?;
        // Bytes 43..48 stay reserved and stay enforced-zero. Narrowed from
        // 41..48; the two bytes removed from this range are now meaningful.
        let reserved = header.get(43..HEADER_SIZE).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "precomputed ephemeris header is missing reserved bytes",
            )
        })?;
        if reserved.iter().any(|byte| *byte != 0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "precomputed ephemeris reserved header bytes must be zero",
            ));
        }
        let expected = n_samples
            .checked_mul(3)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "precomputed ephemeris sample count overflows byte length",
                )
            })?;
        if data.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "precomputed ephemeris file size does not match declared payload",
            ));
        }

        let positions: Vec<f64> = data
            .chunks_exact(8)
            .map(|chunk| {
                let bytes: [u8; 8] = chunk.try_into().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "precomputed ephemeris contains an incomplete position value",
                    )
                })?;
                Ok(f64::from_le_bytes(bytes))
            })
            .collect::<io::Result<_>>()?;
        if positions.iter().any(|value| !value.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "precomputed ephemeris positions must be finite",
            ));
        }

        let inv_dt = if dt_days > 0.0 { 1.0 / dt_days } else { 0.0 };
        let max_direction_rate_per_day =
            Self::max_normalized_direction_rate_per_day(&positions, n_samples, dt_days);

        if epoch_scale == EpochScale::Unspecified
            || epoch_representation == EpochRepresentation::Unspecified
        {
            LEGACY_UNTAGGED_CATALOGUE_NOTICE.store(true, Ordering::Release);
        }

        Ok(Self {
            jd_start,
            dt_days,
            inv_dt,
            n_samples,
            positions: Cow::Owned(positions),
            content_sha256,
            epoch_scale,
            epoch_representation,
            part_a_utc_manifest_authorized: false,
            max_direction_rate_per_day,
        })
    }

    /// Time scale this catalogue's grid is indexed by.
    #[must_use]
    #[inline]
    pub const fn epoch_scale(&self) -> EpochScale {
        self.epoch_scale
    }

    /// Encoding this catalogue's grid is indexed by.
    ///
    /// Read only by `baked_static_tables_match_the_byte_parser`, which is the
    /// point: the baked table and the byte parser must agree on the tag.
    #[must_use]
    #[inline]
    pub const fn epoch_representation(&self) -> EpochRepresentation {
        self.epoch_representation
    }

    /// Position from the compiled Part A UTC-JD authority.
    ///
    /// This IS the production query API. The embedded catalogues are
    /// intentionally untagged; this path accepts them only after their
    /// manifest semantics and exact table bytes have been validated.
    ///
    /// # Errors
    ///
    /// Returns an error when this catalogue was not admitted by the compiled
    /// Part A manifest, or when the requested UTC Julian Date is out of range.
    #[inline]
    pub fn position_at_part_a_utc_jd(
        &self,
        utc: jb_rs::drivers::UtcJulianDay,
    ) -> Result<[f64; 3], EpochScaleError> {
        if !self.part_a_utc_manifest_authorized {
            return Err(EpochScaleError::PartAManifestAuthorityRequired);
        }
        let value = utc.as_f64();
        self.get_position(value).ok_or_else(|| {
            let (start, end) = self.jd_range();
            EpochScaleError::OutOfRange { value, start, end }
        })
    }

    /// Clamped position from the compiled Part A UTC-JD authority.
    ///
    /// # Errors
    ///
    /// Returns an error when this catalogue was not admitted by the compiled
    /// Part A manifest.
    #[inline]
    pub(crate) fn position_at_part_a_utc_jd_clamped(
        &self,
        utc: jb_rs::drivers::UtcJulianDay,
    ) -> Result<[f64; 3], EpochScaleError> {
        if !self.part_a_utc_manifest_authorized {
            return Err(EpochScaleError::PartAManifestAuthorityRequired);
        }
        Ok(self.get_position_clamped(utc.as_f64()))
    }

    /// Get the JD range covered by this ephemeris.
    #[must_use]
    #[inline]
    pub fn jd_range(&self) -> (f64, f64) {
        let interval_count = self.n_samples.saturating_sub(1);
        let interval_count = interval_count.to_f64().unwrap_or(f64::INFINITY);
        // Preserve catalogue multiply-then-add rounding.
        let jd_end = self.jd_start + interval_count * self.dt_days;
        (self.jd_start, jd_end)
    }

    /// Supremum over the entire grid of `|d(u)/dt|`, where `u` is the
    /// NORMALIZED interpolated direction and `t` is measured in days.
    ///
    /// # Derivation, exact and closed form
    ///
    /// Inside grid interval `i` the catalogue is the straight Cartesian line
    /// `r(s) = P_i + s * D`, `D = P_{i+1} - P_i`, `s in [0, 1]`, traversed in
    /// `dt_days`. For `u = r / |r|`,
    ///
    /// ```text
    ///   du/dt = r_dot / |r| - r (r . r_dot) / |r|^3
    ///   |du/dt|^2 = (|r|^2 |r_dot|^2 - (r . r_dot)^2) / |r|^4
    ///             = |r x r_dot|^2 / |r|^4
    /// ```
    ///
    /// so the angular rate is exactly `|r x r_dot| / |r|^2`. The cross product
    /// is CONSTANT along the segment, because
    /// `r x r_dot = (P_i + s D) x (D / dt) = (P_i x D) / dt` — the `s D x D`
    /// term vanishes. The rate is therefore maximized precisely where `|r|` is
    /// minimized, and the minimizer of a straight segment's distance to the
    /// origin is the clamped projection `s* = clamp(-(P_i . D)/|D|^2, 0, 1)`.
    /// Each interval's supremum is thus available in closed form with no search
    /// and no sampling, and the grid maximum of those is a true supremum over
    /// the whole catalogue — not an estimate from probed points.
    ///
    /// # Why this is the quantity the eclipse bound needs
    ///
    /// Angular path length over any subinterval is the integral of the
    /// instantaneous rate, so it is at most this supremum times the elapsed
    /// days. That holds for EVERY subinterval, including ones straddling grid
    /// nodes, which is why the caller needs no node-crossing special case.
    ///
    /// A non-finite result means some interpolated position passes through the
    /// origin, i.e. the direction is undefined somewhere on the grid. Callers
    /// must fail closed on that. Establishing it once here is strictly stronger
    /// than the per-lookup `norm_sq > 0` test it replaces, which could only ever
    /// speak for the points actually probed.
    fn max_normalized_direction_rate_per_day(
        positions: &[f64],
        n_samples: usize,
        dt_days: f64,
    ) -> f64 {
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
            supremum = supremum.max(Self::interval_direction_rate_supremum(
                *before, *after, dt_days,
            ));
        }
        supremum
    }

    /// One grid interval's exact angular-rate supremum, in radians per day.
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

    /// Grid-wide supremum of the normalized direction's angular rate, rad/day.
    #[inline]
    pub(crate) const fn max_direction_rate_per_day(&self) -> f64 {
        self.max_direction_rate_per_day
    }

    /// Whether [`Self::position_at_part_a_utc_jd`] would answer for this JD.
    ///
    /// Exactly the admission test of [`Self::get_position`] plus the Part A
    /// manifest authority, and nothing else — it exists so a caller that needs
    /// only the RANGE verdict can skip the interpolation while keeping the
    /// erroring (never clamping) edge contract byte for byte.
    /// `admits_jd_matches_position_at_part_a_utc_jd` pins the equivalence.
    #[inline]
    pub(crate) fn admits_part_a_utc_jd(&self, jd: f64) -> bool {
        if !self.part_a_utc_manifest_authorized || !jd.is_finite() {
            return false;
        }
        if self.n_samples == 0 {
            return false;
        }
        if self.n_samples == 1 {
            return (jd - self.jd_start).abs() <= f64::EPSILON;
        }
        let fractional_index = (jd - self.jd_start) * self.inv_dt;
        let Some(final_sample) = self.n_samples.checked_sub(1) else {
            return false;
        };
        let Some(final_fractional_index) = final_sample.to_f64() else {
            return false;
        };
        fractional_index >= 0.0 && fractional_index <= final_fractional_index
    }

    fn sample_at(&self, sample_index: usize) -> Option<[f64; 3]> {
        let start = sample_index.checked_mul(3)?;
        let end = start.checked_add(3)?;
        let values = self.positions.get(start..end)?;
        let position: &[f64; 3] = values.try_into().ok()?;
        Some(*position)
    }

    fn interpolate_linear(
        before: [f64; 3],
        after: [f64; 3],
        interpolation_fraction: f64,
    ) -> [f64; 3] {
        let [before_x, before_y, before_z] = before;
        let [after_x, after_y, after_z] = after;
        [
            before_x * (1.0 - interpolation_fraction) + after_x * interpolation_fraction,
            before_y * (1.0 - interpolation_fraction) + after_y * interpolation_fraction,
            before_z * (1.0 - interpolation_fraction) + after_z * interpolation_fraction,
        ]
    }

    /// Get position at the given UTC Julian Date using linear interpolation.
    ///
    /// This is the interpolator behind [`Self::position_at_part_a_utc_jd`],
    /// which is the entry point that also enforces the Part A manifest
    /// authority. It is production-permanent, not a migration leftover.
    ///
    /// Returns None if JD is outside the covered range.
    ///
    /// # Performance
    /// ~10ns per call (no branching in hot path)
    #[inline]
    pub(crate) fn get_position(&self, jd: f64) -> Option<[f64; 3]> {
        if self.n_samples == 0 {
            return None;
        }
        if self.n_samples == 1 {
            if (jd - self.jd_start).abs() <= f64::EPSILON {
                return self.sample_at(0);
            }
            return None;
        }

        // Compute fractional index
        let fractional_index = (jd - self.jd_start) * self.inv_dt;

        // Check bounds
        let final_sample = self.n_samples.checked_sub(1)?;
        let final_fractional_index = final_sample.to_f64()?;
        if fractional_index < 0.0 || fractional_index > final_fractional_index {
            return None;
        }
        if fractional_index >= final_fractional_index {
            return self.sample_at(final_sample);
        }

        // Integer and fractional parts
        let sample_index = fractional_index.to_usize()?;
        let sample_index_as_f64 = sample_index.to_f64()?;
        let interpolation_fraction = fractional_index - sample_index_as_f64;
        let next_sample = sample_index.checked_add(1)?;

        Some(Self::interpolate_linear(
            self.sample_at(sample_index)?,
            self.sample_at(next_sample)?,
            interpolation_fraction,
        ))
    }

    /// Get position, clamping to bounds if outside range.
    ///
    /// Production-permanent interpolator behind
    /// [`Self::position_at_part_a_utc_jd_clamped`], same UTC-Julian-Date
    /// contract as [`Self::get_position`].
    #[inline]
    pub(crate) fn get_position_clamped(&self, jd: f64) -> [f64; 3] {
        if self.n_samples < 2 {
            return self.sample_at(0).unwrap_or([0.0; 3]);
        }
        let (jd_start, jd_end) = self.jd_range();
        if jd <= jd_start {
            return self.sample_at(0).unwrap_or([0.0; 3]);
        }
        if jd >= jd_end {
            let Some(last_sample) = self.n_samples.checked_sub(1) else {
                return [0.0; 3];
            };
            return self.sample_at(last_sample).unwrap_or([0.0; 3]);
        }
        let fractional_index = (jd - self.jd_start) * self.inv_dt;
        let Some(sample_index) = fractional_index.to_usize() else {
            return [0.0; 3];
        };
        let Some(sample_index_as_f64) = sample_index.to_f64() else {
            return [0.0; 3];
        };
        let interpolation_fraction = fractional_index - sample_index_as_f64;
        let Some(next_sample) = sample_index.checked_add(1) else {
            return [0.0; 3];
        };
        let Some(before) = self.sample_at(sample_index) else {
            return [0.0; 3];
        };
        let Some(after) = self.sample_at(next_sample) else {
            return [0.0; 3];
        };

        Self::interpolate_linear(before, after, interpolation_fraction)
    }

    /// Get number of samples.
    #[must_use]
    pub const fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get time step in days.
    #[must_use]
    pub const fn dt_days(&self) -> f64 {
        self.dt_days
    }

    /// SHA-256 of exact validated bytes, computed once during catalogue load.
    #[must_use]
    pub(crate) fn content_sha256_hex(&self) -> String {
        lowercase_hex(&self.content_sha256)
    }
}

/// Collection of precomputed ephemeris for all bodies.
#[derive(Debug, Default, Clone)]
pub struct AllPrecomputedEphemeris {
    pub sun: Option<PrecomputedEphemeris>,
    pub moon: Option<PrecomputedEphemeris>,
    pub jupiter: Option<PrecomputedEphemeris>,
    pub venus: Option<PrecomputedEphemeris>,
    pub mars: Option<PrecomputedEphemeris>,
    pub saturn: Option<PrecomputedEphemeris>,
}

impl AllPrecomputedEphemeris {
    /// Get ephemeris for a body.
    #[must_use]
    #[inline]
    pub const fn get(&self, body: Body) -> Option<&PrecomputedEphemeris> {
        match body {
            Body::Sun => self.sun.as_ref(),
            Body::Moon => self.moon.as_ref(),
            Body::Jupiter => self.jupiter.as_ref(),
            Body::Venus => self.venus.as_ref(),
            Body::Mars => self.mars.as_ref(),
            Body::Saturn => self.saturn.as_ref(),
        }
    }

    /// Validate complete inclusive coverage for every requested dynamic body.
    ///
    /// `dynamic_body_flags` uses each body's force bit. Arc direction is
    /// irrelevant: both absolute endpoints are normalized before validation.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite arc endpoints, absent or invalid required
    /// catalogues, or a required interval outside available coverage.
    pub(crate) fn validate_dynamic_arc(
        &self,
        dynamic_body_flags: i32,
        jd_a: f64,
        jd_b: f64,
    ) -> Result<(), EphemerisCoverageError> {
        if !jd_a.is_finite() || !jd_b.is_finite() {
            return Err(EphemerisCoverageError::NonFiniteArc { jd_a, jd_b });
        }
        let required_start = jd_a.min(jd_b);
        let required_end = jd_a.max(jd_b);
        for body in Body::ALL {
            if (dynamic_body_flags & body.force_flag()) == 0 {
                continue;
            }
            let ephem = self
                .get(body)
                .ok_or(EphemerisCoverageError::MissingBody { body })?;
            if ephem.n_samples == 0 || !ephem.dt_days.is_finite() || ephem.dt_days <= 0.0 {
                return Err(EphemerisCoverageError::InvalidCatalogue { body });
            }
            let (available_start, available_end) = ephem.jd_range();
            if required_start < available_start || required_end > available_end {
                return Err(EphemerisCoverageError::OutsideRange {
                    body,
                    required_start,
                    required_end,
                    available_start,
                    available_end,
                });
            }
        }
        Ok(())
    }

    /// Set ephemeris for a body.
    pub fn set(&mut self, body: Body, data: PrecomputedEphemeris) {
        match body {
            Body::Sun => self.sun = Some(data),
            Body::Moon => self.moon = Some(data),
            Body::Jupiter => self.jupiter = Some(data),
            Body::Venus => self.venus = Some(data),
            Body::Mars => self.mars = Some(data),
            Body::Saturn => self.saturn = Some(data),
        }
    }

    /// Check which bodies are loaded.
    #[must_use]
    pub(crate) fn loaded_bodies(&self) -> Vec<Body> {
        let mut bodies = Vec::new();
        if self.sun.is_some() {
            bodies.push(Body::Sun);
        }
        if self.moon.is_some() {
            bodies.push(Body::Moon);
        }
        if self.jupiter.is_some() {
            bodies.push(Body::Jupiter);
        }
        if self.venus.is_some() {
            bodies.push(Body::Venus);
        }
        if self.mars.is_some() {
            bodies.push(Body::Mars);
        }
        if self.saturn.is_some() {
            bodies.push(Body::Saturn);
        }
        bodies
    }

    /// Get the JD range covered by all loaded bodies (intersection).
    #[must_use]
    pub fn common_jd_range(&self) -> Option<(f64, f64)> {
        let mut jd_start = f64::NEG_INFINITY;
        let mut jd_end = f64::INFINITY;

        for body in Body::ALL {
            if let Some(ephem) = self.get(body) {
                let (start, end) = ephem.jd_range();
                jd_start = jd_start.max(start);
                jd_end = jd_end.min(end);
            }
        }

        if jd_start < jd_end {
            Some((jd_start, jd_end))
        } else {
            None
        }
    }
}

// ============================================================================
// Global Singleton and Loading
// ============================================================================

/// Global precomputed ephemeris singleton.
///
/// This is merge-safe across repeated load calls: later requests can add bodies.
static GLOBAL_EPHEMERIS: std::sync::LazyLock<RwLock<Option<Arc<AllPrecomputedEphemeris>>>> =
    std::sync::LazyLock::new(|| RwLock::new(None));

/// Raw tracked catalogue bytes, test-only oracle input. Production identity
/// comes from the build-time-baked consts.
#[cfg(test)]
fn embedded_catalogue(body: Body) -> Option<&'static [u8]> {
    match body {
        Body::Sun => Some(EMBEDDED_SUN),
        Body::Moon => Some(EMBEDDED_MOON),
        Body::Jupiter => Some(EMBEDDED_JUPITER),
        Body::Venus => Some(EMBEDDED_VENUS),
        Body::Mars | Body::Saturn => None,
    }
}

const fn embedded_catalogue_index(body: Body) -> Option<usize> {
    match body {
        Body::Sun => Some(0),
        Body::Moon => Some(1),
        Body::Jupiter => Some(2),
        Body::Venus => Some(3),
        Body::Mars | Body::Saturn => None,
    }
}

/// Per-catalogue SHA-256, baked by `build.rs` from the same tracked bytes the
/// retired `LazyLock` hashed at first use (oracle-pinned bit-identical).
const EMBEDDED_CATALOGUE_SHA256: [[u8; 32]; 4] = [
    embedded_tables::SUN_CONTENT_SHA256,
    embedded_tables::MOON_CONTENT_SHA256,
    embedded_tables::JUPITER_CONTENT_SHA256,
    embedded_tables::VENUS_CONTENT_SHA256,
];

const EMBEDDED_CATALOGUE_SHA256_HEX: [&str; 4] = [
    embedded_tables::SUN_CONTENT_SHA256_HEX,
    embedded_tables::MOON_CONTENT_SHA256_HEX,
    embedded_tables::JUPITER_CONTENT_SHA256_HEX,
    embedded_tables::VENUS_CONTENT_SHA256_HEX,
];

/// SHA-256 of exact bytes compiled for one default ephemeris catalogue.
#[must_use]
pub fn embedded_catalogue_sha256_hex(body: Body) -> Option<&'static str> {
    let index = embedded_catalogue_index(body)?;
    EMBEDDED_CATALOGUE_SHA256_HEX.get(index).copied()
}

/// SHA-256 of `name=<catalogue sha256 hex>\n` records in [`Body::DEFAULT`] order.
#[must_use]
pub const fn embedded_ephemeris_bundle_sha256_hex() -> &'static str {
    embedded_tables::EMBEDDED_BUNDLE_SHA256_HEX
}

/// Validated identity of the compiled Part A UTC-JD ephemeris assets.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartAEphemerisAuthority {
    manifest_sha256: String,
    table_bundle_sha256: String,
}

impl PartAEphemerisAuthority {
    #[must_use]
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    #[must_use]
    pub fn table_bundle_sha256(&self) -> &str {
        &self.table_bundle_sha256
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartAManifestBody {
    body_id: u8,
    dt_days: f64,
    jd_end: f64,
    jd_start: f64,
    n_samples: usize,
    sha256: String,
    size_bytes: usize,
    source_target: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartAManifestBodies {
    sun: PartAManifestBody,
    moon: PartAManifestBody,
    jupiter: PartAManifestBody,
    venus: PartAManifestBody,
}

impl PartAManifestBodies {
    const fn get(&self, body: Body) -> Option<&PartAManifestBody> {
        match body {
            Body::Sun => Some(&self.sun),
            Body::Moon => Some(&self.moon),
            Body::Jupiter => Some(&self.jupiter),
            Body::Venus => Some(&self.venus),
            Body::Mars | Body::Saturn => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartAManifestHashes {
    sun: String,
    moon: String,
    jupiter: String,
    venus: String,
}

impl PartAManifestHashes {
    fn get(&self, body: Body) -> Option<&str> {
        match body {
            Body::Sun => Some(&self.sun),
            Body::Moon => Some(&self.moon),
            Body::Jupiter => Some(&self.jupiter),
            Body::Venus => Some(&self.venus),
            Body::Mars | Body::Saturn => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartAEphemerisManifest {
    #[serde(rename = "astropy_version")]
    _astropy_version: String,
    axes: String,
    binary_format_version: u32,
    bodies: PartAManifestBodies,
    coordinate_semantics: String,
    dynamics_independent_time_scale: String,
    epoch_representation: String,
    epoch_scale: String,
    #[serde(rename = "erfa_version")]
    _erfa_version: String,
    frame: String,
    future_utc_policy: String,
    header_format: String,
    header_size_bytes: usize,
    #[serde(rename = "independent_oracle")]
    _independent_oracle: serde_json::Value,
    interpolation: String,
    light_time_correction: String,
    origin: String,
    payload_format: String,
    payload_size_policy: String,
    reserved_header_bytes: String,
    sha256: PartAManifestHashes,
    source: String,
    source_api: String,
    source_api_time_construction: String,
    stellar_aberration_correction: String,
}

static PART_A_EPHEMERIS_AUTHORITY: std::sync::OnceLock<PartAEphemerisAuthority> =
    std::sync::OnceLock::new();
static PART_A_EPHEMERIS_AUTHORITY_LOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
fn success_only_cached<'cache, T, E>(
    cache: &'cache std::sync::OnceLock<T>,
    cold_load: &std::sync::Mutex<()>,
    initialize: impl FnOnce() -> Result<T, E>,
) -> Result<&'cache T, E> {
    if let Some(value) = cache.get() {
        return Ok(value);
    }
    let _guard = cold_load
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(value) = cache.get() {
        return Ok(value);
    }
    let candidate = initialize()?;
    Ok(cache.get_or_init(|| candidate))
}

/// Validate compiled manifest semantics and exact table bytes once.
///
/// # Errors
///
/// Returns an error when the compiled manifest or its embedded table bytes do
/// not satisfy the sealed Part A authority contract.
pub fn part_a_ephemeris_authority() -> io::Result<&'static PartAEphemerisAuthority> {
    success_only_cached(
        &PART_A_EPHEMERIS_AUTHORITY,
        &PART_A_EPHEMERIS_AUTHORITY_LOAD,
        || validate_part_a_ephemeris_authority(EMBEDDED_MANIFEST),
    )
}

fn validate_part_a_ephemeris_authority(bytes: &[u8]) -> io::Result<PartAEphemerisAuthority> {
    validate_part_a_ephemeris_manifest(bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn validate_part_a_ephemeris_manifest(bytes: &[u8]) -> anyhow::Result<PartAEphemerisAuthority> {
    let manifest: PartAEphemerisManifest =
        serde_json::from_slice(bytes).context("parsing compiled Part A ephemeris manifest JSON")?;
    for (actual, expected, field) in [
        (manifest.axes.as_str(), "icrs", "axes"),
        (
            manifest.coordinate_semantics.as_str(),
            "geometric",
            "coordinate_semantics",
        ),
        (
            manifest.dynamics_independent_time_scale.as_str(),
            "tdb_internal_via_astropy",
            "dynamics_independent_time_scale",
        ),
        (
            manifest.epoch_representation.as_str(),
            "astronomical_julian_date",
            "epoch_representation",
        ),
        (manifest.epoch_scale.as_str(), "utc", "epoch_scale"),
        (
            manifest.frame.as_str(),
            "earth_centered_icrs_cartesian_km",
            "frame",
        ),
        (
            manifest.future_utc_policy.as_str(),
            "frozen_last_known_leap_second",
            "future_utc_policy",
        ),
        (
            manifest.header_format.as_str(),
            "<4sIQdddB7s",
            "header_format",
        ),
        (
            manifest.interpolation.as_str(),
            "linear_on_utc_jd_grid",
            "interpolation",
        ),
        (
            manifest.light_time_correction.as_str(),
            "none",
            "light_time_correction",
        ),
        (manifest.origin.as_str(), "earth_center", "origin"),
        (
            manifest.payload_format.as_str(),
            "n_samples_x_3_little_endian_float64_km",
            "payload_format",
        ),
        (
            manifest.payload_size_policy.as_str(),
            "exact_no_trailing_bytes",
            "payload_size_policy",
        ),
        (
            manifest.reserved_header_bytes.as_str(),
            "zero",
            "reserved_header_bytes",
        ),
        (
            manifest.source.as_str(),
            "astropy_builtin_same_epoch_barycentric_difference",
            "source",
        ),
        (
            manifest.source_api.as_str(),
            "get_body_barycentric(body,t)-get_body_barycentric('earth',t)",
            "source_api",
        ),
        (
            manifest.source_api_time_construction.as_str(),
            "astropy.time.Time(jd, format='jd', scale='utc')",
            "source_api_time_construction",
        ),
        (
            manifest.stellar_aberration_correction.as_str(),
            "none",
            "stellar_aberration_correction",
        ),
    ] {
        if actual != expected {
            return Err(anyhow::anyhow!(
                "compiled Part A ephemeris manifest {field} is {actual:?}, expected {expected:?}"
            ));
        }
    }
    if manifest.binary_format_version != VERSION || manifest.header_size_bytes != HEADER_SIZE {
        return Err(anyhow::anyhow!(
            "compiled Part A ephemeris manifest binary format mismatch"
        ));
    }

    for body in Body::DEFAULT {
        let declared = manifest
            .bodies
            .get(body)
            .ok_or_else(|| anyhow::anyhow!("manifest missing {}", body.name()))?;
        // The compiled table's own format invariants (magic, version, grid
        // bit-agreement, tags, reserved zeros, exact payload) are enforced by
        // build.rs as BUILD failures; what remains at runtime is the manifest
        // cross-check against the baked identity, byte for byte the same
        // comparisons the retired `load_bytes` re-parse fed.
        let table = embedded_table(body)
            .ok_or_else(|| anyhow::anyhow!("compiled table missing {}", body.name()))?;
        let expected_target = match body {
            Body::Sun => "sun_center",
            Body::Moon => "moon_center",
            Body::Jupiter => "jupiter_barycenter",
            Body::Venus => "venus_center",
            Body::Mars | Body::Saturn => {
                return Err(anyhow::anyhow!(
                    "compiled Part A manifest includes unsupported default body {}",
                    body.name()
                ));
            }
        };
        let table_hash = table.content_sha256_hex;
        if declared.body_id != body.id()
            || table.body_id != body.id()
            || declared.dt_days.to_bits() != table.dt_days.to_bits()
            || declared.jd_start.to_bits() != table.jd_start.to_bits()
            || declared.jd_end.to_bits() != table.jd_end.to_bits()
            || declared.n_samples != table.n_samples
            || declared.size_bytes != table.size_bytes
            || declared.source_target != expected_target
            || declared.sha256 != table_hash
            || manifest.sha256.get(body) != Some(table_hash)
        {
            return Err(anyhow::anyhow!(
                "compiled Part A ephemeris manifest disagrees with {} table",
                body.name()
            ));
        }
    }

    Ok(PartAEphemerisAuthority {
        manifest_sha256: lowercase_hex(&Sha256::digest(bytes)),
        table_bundle_sha256: embedded_ephemeris_bundle_sha256_hex().to_owned(),
    })
}

// Nesting depth for the guard above, per thread.
#[cfg(test)]
thread_local! {
    static EPHEMERIS_GUARD_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// RAII half of [`ephemeris_test_guard`]. Dropping it unwinds one level of
/// depth, and releases the mutex only at the outermost level.
#[cfg(test)]
pub(crate) struct EphemerisTestGuard {
    /// `None` for a nested acquisition: the outer guard still owns the lock.
    _inner: Option<std::sync::MutexGuard<'static, ()>>,
}

#[cfg(test)]
impl Drop for EphemerisTestGuard {
    fn drop(&mut self) {
        EPHEMERIS_GUARD_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Serializes every test that touches the process-global ephemeris catalogue.
///
/// The mutex protects `()`: it orders tests, it does not own data. So a
/// `PoisonError` here carries exactly one fact -- some *earlier* test panicked
/// while holding the guard -- and nothing about whether *this* test can run.
/// Propagating it turned a single real failure into 22 phantom ones on
/// `x86_64` Linux, where `integrator::tests::production_wrapper_converges_with_tolerance`
/// panics under the guard and every later ephemeris test then aborted with
/// `PoisonError` instead of reporting its own verdict. Recover instead, so the
/// failure count stays equal to the number of defects.
///
/// What the panicking test *may* have left behind is a half-loaded catalogue,
/// so clear it on the recovery path. That path runs only after a test has
/// already failed; a green run never reaches it.
///
/// REENTRANT, and it has to be. The first sentence above used to be false: this
/// guard only ever ordered the tests that *called* it, which is the handful
/// that publish a temp catalogue. Every other test reaches the same global
/// through `ForceConfig::with_ephemeris_for_arc` and took no lock at all, so a
/// `load_precomputed_ephemeris_from_dirs` test -- the one path that publishes
/// without the embedded check -- could have its deliberately-conflicting
/// catalogue observed by any test running beside it, which then failed with
/// `cached sun ephemeris SHA-256 ... conflicts with compiled SHA-256 ...`.
/// The victim was whoever happened to be scheduled there, which is why the same
/// defect reported as three different tests. `with_ephemeris_for_arc` now takes
/// this guard under `cfg(test)`, making the claim true; since eleven tests hold
/// it across a fixture build that itself calls that function, a non-reentrant
/// mutex would deadlock instead.
///
/// Depth is thread-local and the real `MutexGuard` is held only by the
/// outermost acquisition on a thread, so a nested one neither relocks nor
/// releases early.
#[cfg(test)]
pub(crate) fn ephemeris_test_guard() -> EphemerisTestGuard {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

    let depth = EPHEMERIS_GUARD_DEPTH.with(std::cell::Cell::get);
    EPHEMERIS_GUARD_DEPTH.with(|current| current.set(depth.saturating_add(1)));
    if depth > 0 {
        // Already held further up this thread's stack; re-locking would hang.
        return EphemerisTestGuard { _inner: None };
    }
    let lock = LOCK.get_or_init(|| std::sync::Mutex::new(()));
    let inner = match lock.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let guard = poisoned.into_inner();
            clear_global_ephemeris_state();
            guard
        }
    };
    EphemerisTestGuard {
        _inner: Some(inner),
    }
}

/// Returns the process-global catalogue to its unloaded state, *all* of it.
///
/// Clearing `GLOBAL_EPHEMERIS` alone is not enough and stopped being enough the
/// moment the lock-free publish landed: `PUBLISHED_EPHEMERIS` and
/// `LOADED_BODY_FLAGS` are read on a fast path that never touches the `RwLock`,
/// so a half-reset leaves the next loader announcing catalogues that are no
/// longer reachable. It then fails to load, and every ephemeris test after it
/// fails with it -- which is the 22-phantom-failure shape from `x86_64`,
/// reappearing through a new pair of statics rather than through poison.
///
/// The leaked allocation stays alive, so any `&'static` a test already took
/// remains valid; it is simply no longer reachable.
#[cfg(test)]
#[expect(
    clippy::significant_drop_tightening,
    reason = "the write guard intentionally spans the publication flags so readers cannot observe a half-reset catalogue"
)]
pub(crate) fn clear_global_ephemeris_state() {
    // Take the write lock through poison recovery, since it can be poisoned for
    // the same reason and by the same test. Overwriting outright cannot observe
    // torn state.
    let mut global = match GLOBAL_EPHEMERIS.write() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    *global = None;
    LOADED_BODY_FLAGS.store(0, Ordering::Release);
    PUBLISHED_EPHEMERIS.store(std::ptr::null_mut(), Ordering::Release);
    PUBLISHED_EMBEDDED_CONFLICT.store(false, Ordering::Release);
}

/// Search paths for ephemeris catalogues.
const fn ephemeris_search_paths() -> Vec<PathBuf> {
    // Runtime defaults are compile-time embedded. Unsupported bodies fail closed;
    // tests and explicit loaders may still supply controlled search directories.
    Vec::new()
}

/// Find a catalogue file in the given search directories.
fn find_catalogue_in(search_dirs: &[PathBuf], body: Body) -> Option<PathBuf> {
    let filename = body.filename();
    for dir in search_dirs {
        let path = dir.join(filename);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

/// Load precomputed ephemeris for the specified bodies.
///
/// This function is idempotent - calling it multiple times with the same flags
/// will reuse the existing loaded data.
///
/// # Arguments
/// * `force_flags` - Bitfield of force flags (`FORCE_SUN | FORCE_MOON | ...`)
///
/// # Returns
/// Ok(()) if at least one body was loaded, Err otherwise.
///
/// # Errors
///
/// Returns an error when requested catalogues cannot be loaded or validated.
pub fn load_precomputed_ephemeris(force_flags: i32) -> io::Result<()> {
    load_precomputed_ephemeris_from_embedded_and_dirs(force_flags, &ephemeris_search_paths())
}

/// Load precomputed ephemeris from explicit search directories. Internal
/// seam so unit tests can inject a temp catalogue directory; production
/// callers go through the pathless wrapper above.
/// Directory-scoped loader. Every caller is in this module's test suite —
/// production loads the embedded catalogues — so it is `#[cfg(test)]` rather
/// than carrying a module-wide `allow(unused)` to keep it quiet.
#[cfg(test)]
fn load_precomputed_ephemeris_from_dirs(
    force_flags: i32,
    search_dirs: &[PathBuf],
) -> io::Result<()> {
    load_precomputed_ephemeris_from_sources(force_flags, search_dirs, false)
}

fn load_precomputed_ephemeris_from_embedded_and_dirs(
    force_flags: i32,
    search_dirs: &[PathBuf],
) -> io::Result<()> {
    load_precomputed_ephemeris_from_sources(force_flags, search_dirs, true)
}

/// Bodies currently published in `GLOBAL_EPHEMERIS`, as a `ForceFlags` mask.
///
/// Written only while the `GLOBAL_EPHEMERIS` write lock is held, so it never
/// disagrees with the store. Read without any lock, which is the entire point:
/// `load_precomputed_ephemeris_from_sources` is called on every HF row and
/// every HF propagation segment, and all it usually needs to know is whether
/// there is anything left to do.
static LOADED_BODY_FLAGS: AtomicI32 = AtomicI32::new(0);

/// Whether the published set disagrees with the compiled catalogues.
///
/// Recomputed under the write lock on every publish, so the lock-free path can
/// answer "is there a conflict to report?" with one load instead of walking
/// four catalogue headers. Only `load_precomputed_ephemeris_from_dirs` can
/// create a conflicting state -- it publishes without ever running the embedded
/// check -- but that is reachable, so the flag is maintained unconditionally.
static PUBLISHED_EMBEDDED_CONFLICT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The published catalogue set, reachable without taking a lock.
///
/// Holds a raw pointer to the same allocation the `Arc` in `GLOBAL_EPHEMERIS`
/// points at, with one strong reference deliberately leaked so the target
/// outlives every borrow. No data is duplicated: this is `Arc::into_raw` of a
/// clone of the published handle, not a second copy of the catalogues.
///
/// `get_precomputed_ephemeris` cannot serve the hot path because its signature
/// hands back an owned `Arc`, and taking the read lock plus bumping a refcount
/// is exactly the per-call shared-line traffic that has to go. Callers here
/// only ever borrow, so give them a borrow.
static PUBLISHED_EPHEMERIS: std::sync::atomic::AtomicPtr<AllPrecomputedEphemeris> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Borrow the published catalogue set without locking or touching a refcount.
///
/// Returns `None` before the first successful load. Pairs with the `Release`
/// store made under the write lock, so a non-null pointer is fully initialized.
#[must_use]
#[inline]
pub fn published_ephemeris() -> Option<&'static AllPrecomputedEphemeris> {
    let ptr = PUBLISHED_EPHEMERIS.load(Ordering::Acquire);
    if ptr.is_null() {
        None
    } else {
        // SAFETY: the pointer came from `Arc::into_raw` on a handle whose
        // strong count was never released, so the allocation is live for the
        // rest of the process. `AllPrecomputedEphemeris` is immutable once
        // published -- every mutation builds a fresh value and publishes a new
        // pointer -- so handing out `&'static` aliases is sound.
        Some(unsafe { &*ptr })
    }
}

/// Union of every body's force flag.
#[inline]
fn all_body_flags() -> i32 {
    Body::ALL
        .into_iter()
        .fold(0, |mask, body| mask | body.force_flag())
}

/// The mask to publish for a resolved catalogue set.
fn loaded_body_flags_of(ephem: &AllPrecomputedEphemeris) -> i32 {
    Body::ALL.into_iter().fold(0, |mask, body| {
        if ephem.get(body).is_some() {
            mask | body.force_flag()
        } else {
            mask
        }
    })
}

/// Reject a cached catalogue whose bytes disagree with the compiled one.
///
/// Lifted out of `load_precomputed_ephemeris_from_sources` so the read-only
/// fast path and the write path run the identical check rather than two copies
/// that could drift apart.
fn reject_embedded_sha_conflict(existing: &AllPrecomputedEphemeris) -> io::Result<()> {
    for body in Body::DEFAULT {
        let Some(cached) = existing.get(body) else {
            continue;
        };
        let index = embedded_catalogue_index(body).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "default body {} has no embedded identity index",
                    body.name()
                ),
            )
        })?;
        let compiled_sha256 = EMBEDDED_CATALOGUE_SHA256.get(index).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "default body {} identity index is out of range",
                    body.name()
                ),
            )
        })?;
        let compiled_hash = embedded_catalogue_sha256_hex(body).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("default body {} has no compiled identity", body.name()),
            )
        })?;
        if cached.content_sha256 != *compiled_sha256 {
            let body_name = body.name();
            let cached_hash = cached.content_sha256_hex();
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("cached {body_name} ephemeris SHA-256 {cached_hash} conflicts with compiled SHA-256 {compiled_hash}"),
            ));
        }
    }
    Ok(())
}

/// True when every body `force_flags` asks for is already resolved.
fn all_requested_bodies_present(existing: &AllPrecomputedEphemeris, force_flags: i32) -> bool {
    Body::ALL
        .into_iter()
        .all(|body| (force_flags & body.force_flag()) == 0 || existing.get(body).is_some())
}

fn load_precomputed_ephemeris_from_sources(
    force_flags: i32,
    search_dirs: &[PathBuf],
    use_embedded: bool,
) -> io::Result<()> {
    // FAST PATH, taken by every call after the first.
    //
    // This function is not, despite the note below, only reached during
    // initialization. `ForceConfig::with_ephemeris_for_arc` calls it on every
    // HF row (`dust_estimates_rs::mass_solver::prepare_hf_for_event`) and on
    // every HF propagation segment (`two_phase_transfer_rs::evaluate::
    // stamped_body_force_config`), and production always reaches it: with
    // `atm_model` 4 and drag on, `required_dynamic_ephemeris_flags` sets
    // SUN_GRAVITY unconditionally, and `with_ephemeris_for_arc` writes that
    // result back into `dynamic_ephemeris_flags`, so derived configs stay on
    // this path too.
    //
    // Taking the EXCLUSIVE lock to discover there was nothing to do serialized
    // the whole worker pool on one word. Measured with a probe that calls this
    // from W threads and self-times each one (arm64, per-call ns, p50):
    //   W=1 12.8 | W=2 71.7 | W=4 150.3 | W=8 322.9
    // That is worse than serial -- aggregate throughput FELL from 78 M/s at
    // W=1 to 25 M/s at W=8 -- because contending writers convoy on the lock
    // word on top of being serialized by it. Through the real
    // `with_ephemeris_for_arc` entry point the same sweep read 53.2 / 187.3 /
    // 619.5 / 1047.6 ns.
    //
    // A shared READ lock is not the fix. It is still an atomic
    // read-modify-write on one word, so every reader CASes the same cache line
    // and they convoy on it just as writers do. Measured, same probe, p50 ns:
    //   read-lock fast path: W=1 14.3 | W=2 114.0 | W=4 167.8 | W=8 571.5
    // against 322.9 at W=8 for the write lock it replaced. It made this worse,
    // so it is not what landed.
    //
    // What works is not taking a lock at all. The question the fast path asks
    // is only "which bodies are published?", which is one integer, so keep it
    // in an atomic beside the store: written exclusively under the write lock,
    // read with a plain load. After startup that line is never written again,
    // so it sits SHARED-clean in every core's cache and costs no coherence
    // traffic at any worker count -- the same reason `frame_authority()`
    // returning a borrow instead of cloning an `Arc` was a win.
    //
    // `Acquire` here pairs with the `Release` store made after the new value is
    // published, so seeing a body's bit guarantees seeing its data.
    //
    // The SHA-conflict guard still runs here, and has to. "Everything published
    // was already validated" is FALSE: `load_precomputed_ephemeris_from_dirs`
    // publishes with `use_embedded == false` and so never runs the check, which
    // is precisely the state
    // `embedded_loader_rejects_conflicting_cached_default_body` builds -- seed a
    // Sun catalogue from a directory, then demand that the next embedded load
    // reject it. Skipping the guard here turned that test's `expect_err` into an
    // `Ok`. It costs four 32-byte compares against immutable, shared-clean
    // lines, which is not what was expensive; the exclusive lock was.
    // Running the guard itself on every call still cost measurable scaling --
    // 44.3 ns at W=1 against 117.9 at W=8 (arm64) -- because it walks four
    // catalogue headers per call. But its verdict is a pure function of what is
    // published, and publishing happens under the write lock, so the verdict
    // can be computed once there and read here as a single flag. The
    // message-building path below is entered only when a conflict actually
    // exists, i.e. never in the steady state, so the error text and kind are
    // unchanged.
    let requested_bodies = force_flags & all_body_flags();
    if requested_bodies & !LOADED_BODY_FLAGS.load(Ordering::Acquire) == 0 {
        if use_embedded && PUBLISHED_EMBEDDED_CONFLICT.load(Ordering::Acquire) {
            if let Some(existing) = published_ephemeris() {
                reject_embedded_sha_conflict(existing)?;
            }
        }
        return Ok(());
    }

    // Initialization is rare and catalogue I/O is bounded.  Hold the write
    // lock across read/merge/publish so concurrent partial requests cannot
    // overwrite each other's newly loaded bodies.
    //
    // Everything above is re-checked here rather than trusted: the read lock
    // was released before this one was taken, so another thread may have
    // published in the gap.
    let mut global = GLOBAL_EPHEMERIS
        .write()
        .map_err(|_| io::Error::other("precomputed ephemeris lock poisoned"))?;
    let existing = global.clone();

    if use_embedded {
        if let Some(existing_ephem) = &existing {
            reject_embedded_sha_conflict(existing_ephem)?;
        }
    }

    // Check if already loaded with sufficient bodies.
    if let Some(ref existing_ephem) = existing {
        if all_requested_bodies_present(existing_ephem, force_flags) {
            return Ok(());
        }
    }

    let mut ephem = existing
        .as_ref()
        .map(|arc| arc.as_ref().clone())
        .unwrap_or_default();
    let mut any_loaded = !ephem.loaded_bodies().is_empty();

    for body in Body::ALL {
        if (force_flags & body.force_flag()) == 0 || ephem.get(body).is_some() {
            continue;
        }
        if use_embedded && embedded_table(body).is_some() {
            let data = PrecomputedEphemeris::load_embedded(body)?;
            ephem.set(body, data);
            any_loaded = true;
        } else if let Some(path) = find_catalogue_in(search_dirs, body) {
            let data = PrecomputedEphemeris::load(&path)?;
            ephem.set(body, data);
            any_loaded = true;
        }
    }

    if !any_loaded {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "No precomputed ephemeris catalogues found",
        ));
    }

    // Store merged set in global.
    let missing_requested: Vec<&'static str> = Body::ALL
        .into_iter()
        .filter(|body| (force_flags & body.force_flag()) != 0)
        .filter(|body| ephem.get(*body).is_none())
        .map(Body::filename)
        .collect();
    if !missing_requested.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "Missing requested precomputed ephemeris catalogues: {}",
                missing_requested.join(", ")
            ),
        ));
    }
    // Publish the mask with `Release` BEFORE dropping the write lock, so an
    // `Acquire` load on the fast path that sees a body's bit also sees its
    // data. The store happens after `ephem` is final and while the lock is
    // still held, so no reader can observe a mask wider than the store.
    let published = loaded_body_flags_of(&ephem);
    let conflicts = reject_embedded_sha_conflict(&ephem).is_err();
    let handle = Arc::new(ephem);
    // Leak one strong reference so `published_ephemeris()` can hand out
    // `&'static`. Only ever runs while the write lock is held, and only when a
    // body was actually loaded, so this is bounded by the catalogue count
    // rather than by call volume. Any previously published pointer leaks with
    // it, which is what keeps borrows taken against it valid.
    PUBLISHED_EPHEMERIS.store(
        Arc::into_raw(Arc::clone(&handle)).cast_mut(),
        Ordering::Release,
    );
    *global = Some(handle);
    PUBLISHED_EMBEDDED_CONFLICT.store(conflicts, Ordering::Release);
    // Published LAST, because it is the flag the lock-free path gates on: an
    // Acquire load that sees these bits also sees the pointer and the conflict
    // verdict stored above.
    LOADED_BODY_FLAGS.store(published, Ordering::Release);
    drop(global);
    Ok(())
}

/// Try to load precomputed ephemeris, silently returning None on failure.
#[must_use]
pub fn try_load_precomputed_ephemeris(force_flags: i32) -> Option<Arc<AllPrecomputedEphemeris>> {
    load_precomputed_ephemeris(force_flags).ok()?;
    GLOBAL_EPHEMERIS.read().ok().and_then(|g| g.clone())
}

/// Get the global precomputed ephemeris if loaded.
#[must_use]
#[inline]
pub fn get_precomputed_ephemeris() -> Option<Arc<AllPrecomputedEphemeris>> {
    GLOBAL_EPHEMERIS.read().ok().and_then(|g| g.clone())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn part_a_authority_cache_retries_failures_and_reuses_success() {
        let cache = std::sync::OnceLock::new();
        let lock = std::sync::Mutex::new(());
        let attempts = AtomicUsize::new(0);
        let first = success_only_cached(&cache, &lock, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Err::<u8, _>("hostile first validation".to_owned())
        });
        assert!(first.is_err());
        assert!(cache.get().is_none());

        let second = success_only_cached(&cache, &lock, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok::<u8, String>(7)
        })
        .expect("retry succeeds");
        let third = success_only_cached(&cache, &lock, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok::<u8, String>(9)
        })
        .expect("success is cached");
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
        assert!(std::ptr::eq(second, third));
        assert_eq!(*third, 7);
    }

    #[test]
    fn part_a_authority_io_boundary_keeps_manifest_parse_source() {
        let error = validate_part_a_ephemeris_authority(b"{")
            .expect_err("malformed manifest JSON must fail validation");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            error.to_string(),
            "parsing compiled Part A ephemeris manifest JSON"
        );
        let mut source = std::error::Error::source(&error);
        let mut found_parse_source = false;
        while let Some(cause) = source {
            if cause.downcast_ref::<serde_json::Error>().is_some() {
                found_parse_source = true;
                break;
            }
            source = cause.source();
        }
        assert!(
            found_parse_source,
            "io boundary must retain the serde_json parse source"
        );
    }

    // One reset, shared with the guard's poison-recovery path. Two copies of
    // this drifted apart once already: the copy in the guard cleared only
    // `GLOBAL_EPHEMERIS` and went stale when the lock-free publish landed.
    use super::clear_global_ephemeris_state as reset_global;

    #[test]
    fn malformed_short_header_returns_typed_error() {
        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&1_u64.to_le_bytes());
        header.extend_from_slice(&2_458_849.5_f64.to_le_bytes());
        header.extend_from_slice(&2_458_849.5_f64.to_le_bytes());
        header.extend_from_slice(&1.0_f64.to_le_bytes());
        header.extend_from_slice(&[0_u8; 7]);
        assert_eq!(header.len(), HEADER_SIZE - 1);

        let result = PrecomputedEphemeris::from_header_and_position_bytes(
            &header,
            &[0_u8; 24],
            None,
            [0_u8; 32],
        );

        assert!(matches!(
            result,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof
        ));
    }

    /// Legacy untagged catalogue: epoch bytes 0x00/0x00, exactly like the four
    /// sealed `.bin` files in `data/ephemeris/` before the stage-2 re-stamp.
    fn write_catalogue(path: &Path, body_id: u8, positions: &[[f64; 3]]) {
        write_catalogue_tagged(path, body_id, positions, 0x00, 0x00);
    }

    /// Catalogue with explicit epoch scale and representation tags.
    fn write_catalogue_tagged(
        path: &Path,
        body_id: u8,
        positions: &[[f64; 3]],
        scale_tag: u8,
        representation_tag: u8,
    ) {
        let n_samples = u64::try_from(positions.len()).expect("test catalogue count fits u64");
        let jd_start = 2_458_849.5_f64;
        let interval_count = n_samples
            .saturating_sub(1)
            .to_f64()
            .expect("test catalogue count fits f64");
        let jd_end = 2_458_849.5_f64 + interval_count;
        let dt_days = 1.0_f64;

        let mut header = Vec::with_capacity(HEADER_SIZE);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&n_samples.to_le_bytes());
        header.extend_from_slice(&jd_start.to_le_bytes());
        header.extend_from_slice(&jd_end.to_le_bytes());
        header.extend_from_slice(&dt_days.to_le_bytes());
        header.push(body_id);
        header.push(scale_tag);
        header.push(representation_tag);
        header.extend_from_slice(&[0_u8; 5]);

        let mut file = File::create(path).expect("create catalogue file");
        file.write_all(&header).expect("write header");
        for position in positions {
            for component in position {
                file.write_all(&component.to_le_bytes())
                    .expect("write position component");
            }
        }
        file.flush().expect("flush catalogue");
    }

    fn mutate_catalogue_byte(path: &Path, offset: usize, value: u8) {
        let mut bytes = fs::read(path).expect("read catalogue");
        *bytes
            .get_mut(offset)
            .expect("catalogue mutation offset lies in fixture") = value;
        fs::write(path, bytes).expect("rewrite catalogue");
    }

    #[test]
    fn loader_rejects_header_body_filename_mismatch() {
        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_body_mismatch_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let path = temp_dir.join("sun.bin");
        write_catalogue(&path, Body::Moon.id(), &[[1.0, 2.0, 3.0]]);

        let error = PrecomputedEphemeris::load(&path)
            .expect_err("sun filename with Moon body_id must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn loader_rejects_declared_jd_end_tamper() {
        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_jd_end_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let path = temp_dir.join("sun.bin");
        write_catalogue(&path, Body::Sun.id(), &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let wrong_end = 2_459_999.5_f64.to_le_bytes();
        let mut bytes = fs::read(&path).expect("read catalogue");
        bytes
            .get_mut(24..32)
            .expect("JD-end bytes lie in fixture header")
            .copy_from_slice(&wrong_end);
        fs::write(&path, bytes).expect("rewrite catalogue");

        let error = PrecomputedEphemeris::load(&path)
            .expect_err("declared/computed JD end mismatch must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn loader_rejects_nonzero_reserved_header_bytes() {
        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_reserved_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let path = temp_dir.join("sun.bin");
        write_catalogue(&path, Body::Sun.id(), &[[1.0, 2.0, 3.0]]);
        // Byte 43, not 41. Bytes 41 and 42 are now the epoch scale and epoch
        // representation tags; the still-reserved range narrowed to 43..48.
        // Probing 41 here would now assert that a VALID UTC tag is rejected.
        mutate_catalogue_byte(&path, 43, 1);

        let error =
            PrecomputedEphemeris::load(&path).expect_err("nonzero reserved header bytes must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn loader_rejects_trailing_payload_bytes() {
        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_trailing_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let path = temp_dir.join("sun.bin");
        write_catalogue(&path, Body::Sun.id(), &[[1.0, 2.0, 3.0]]);
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open catalogue");
        file.write_all(&[0_u8]).expect("append trailing byte");

        let error =
            PrecomputedEphemeris::load(&path).expect_err("trailing payload bytes must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_body_filenames() {
        assert_eq!(Body::Sun.filename(), "sun.bin");
        assert_eq!(Body::Moon.filename(), "moon.bin");
        assert_eq!(Body::Jupiter.filename(), "jupiter.bin");
        assert_eq!(Body::Venus.filename(), "venus.bin");
    }

    #[test]
    fn test_body_force_flags() {
        assert_eq!(Body::Sun.force_flag(), 4);
        assert_eq!(Body::Moon.force_flag(), 8);
        assert_eq!(Body::Jupiter.force_flag(), 16);
        assert_eq!(Body::Venus.force_flag(), 32);
    }

    #[expect(
        clippy::float_cmp,
        reason = "exact grid samples are part of the interpolation regression contract"
    )]
    #[test]
    fn test_interpolation_accuracy() {
        // Create a simple test ephemeris with known values
        let positions = vec![
            0.0, 0.0, 0.0, // t=0
            10.0, 20.0, 30.0, // t=1
            20.0, 40.0, 60.0, // t=2
        ];

        let ephem = PrecomputedEphemeris {
            jd_start: 0.0,
            dt_days: 1.0,
            inv_dt: 1.0,
            n_samples: 3,
            positions: positions.into(),
            content_sha256: [0_u8; 32],
            epoch_scale: EpochScale::Utc,
            epoch_representation: EpochRepresentation::JulianDate,
            part_a_utc_manifest_authorized: false,
            max_direction_rate_per_day: 0.0,
        };

        // Test exact samples
        let p0 = ephem.get_position(0.0).unwrap();
        assert_eq!(p0, [0.0, 0.0, 0.0]);

        let p1 = ephem.get_position(1.0).unwrap();
        assert_eq!(p1, [10.0, 20.0, 30.0]);

        // Test interpolation at midpoint
        let p_mid = ephem.get_position(0.5).expect("midpoint is in range");
        let [mid_x, mid_y, mid_z] = p_mid;
        assert!((mid_x - 5.0).abs() < 1e-10);
        assert!((mid_y - 10.0).abs() < 1e-10);
        assert!((mid_z - 15.0).abs() < 1e-10);

        // Test bounds
        assert!(ephem.get_position(-0.1).is_none());
        assert!(ephem.get_position(2.1).is_none());
    }

    #[test]
    fn test_jd_range() {
        let positions = vec![0.0; 9]; // 3 samples
        let dt = 1.0 / 24.0; // 1 hour
        let ephem = PrecomputedEphemeris {
            jd_start: 2_460_000.0,
            dt_days: dt,
            inv_dt: 24.0,
            n_samples: 3,
            positions: positions.into(),
            content_sha256: [0_u8; 32],
            epoch_scale: EpochScale::Utc,
            epoch_representation: EpochRepresentation::JulianDate,
            part_a_utc_manifest_authorized: false,
            max_direction_rate_per_day: 0.0,
        };

        let (start, end) = ephem.jd_range();
        assert_eq!(start.to_bits(), 2_460_000.0_f64.to_bits());
        // 3 samples spanning 2 intervals: end = start + 2*dt
        let expected_end = 2_460_000.0 + 2.0 * dt;
        assert!(
            (end - expected_end).abs() < 1e-10,
            "end {end} != expected {expected_end}"
        );
    }

    #[expect(
        clippy::float_cmp,
        reason = "endpoint lookup must remain bit-exact for sealed catalogue samples"
    )]
    #[test]
    fn test_interpolation_endpoint_inclusive() {
        let ephem = PrecomputedEphemeris {
            jd_start: 10.0,
            dt_days: 1.0,
            inv_dt: 1.0,
            n_samples: 2,
            positions: vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0].into(),
            content_sha256: [0_u8; 32],
            epoch_scale: EpochScale::Utc,
            epoch_representation: EpochRepresentation::JulianDate,
            part_a_utc_manifest_authorized: false,
            max_direction_rate_per_day: 0.0,
        };

        assert_eq!(ephem.get_position(11.0), Some([1.0, 2.0, 3.0]));
        assert!(ephem.get_position(11.000_000_1).is_none());
        assert_eq!(ephem.get_position_clamped(11.0), [1.0, 2.0, 3.0]);
        assert_eq!(ephem.get_position_clamped(12.0), [1.0, 2.0, 3.0]);
    }

    fn test_ephemeris(jd_start: f64, n_samples: usize) -> PrecomputedEphemeris {
        PrecomputedEphemeris {
            jd_start,
            dt_days: 1.0,
            inv_dt: 1.0,
            n_samples,
            positions: vec![1.0; n_samples.saturating_mul(3)].into(),
            content_sha256: [0_u8; 32],
            epoch_scale: EpochScale::Utc,
            epoch_representation: EpochRepresentation::JulianDate,
            part_a_utc_manifest_authorized: false,
            max_direction_rate_per_day: 0.0,
        }
    }

    /// Same shape as [`test_ephemeris`] but untagged, i.e. a legacy catalogue.
    fn legacy_untagged_ephemeris(jd_start: f64, n_samples: usize) -> PrecomputedEphemeris {
        PrecomputedEphemeris {
            epoch_scale: EpochScale::Unspecified,
            epoch_representation: EpochRepresentation::Unspecified,
            ..test_ephemeris(jd_start, n_samples)
        }
    }

    // ========================================================================
    // Epoch scale/representation tags
    // ========================================================================

    #[test]
    fn loader_reads_epoch_tags_and_rejects_unknown_ones() {
        let _guard = ephemeris_test_guard();
        let _ = take_legacy_untagged_catalogue_notice();
        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_epoch_tag_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let path = temp_dir.join("sun.bin");

        // Tagged UTC / Julian Date round-trips through the header.
        write_catalogue_tagged(&path, Body::Sun.id(), &[[1.0, 2.0, 3.0]], 0x01, 0x01);
        let tagged = PrecomputedEphemeris::load(&path).expect("tagged catalogue must load");
        assert_eq!(tagged.epoch_scale(), EpochScale::Utc);
        assert!(!take_legacy_untagged_catalogue_notice());
        assert_eq!(
            tagged.epoch_representation(),
            EpochRepresentation::JulianDate
        );

        // Untagged still LOADS. Rejecting at load would brick the four sealed
        // catalogues, whose tag bytes are zero.
        write_catalogue_tagged(&path, Body::Sun.id(), &[[1.0, 2.0, 3.0]], 0x00, 0x00);
        let untagged = PrecomputedEphemeris::load(&path).expect("legacy catalogue must still load");
        assert_eq!(untagged.epoch_scale(), EpochScale::Unspecified);
        assert!(take_legacy_untagged_catalogue_notice());
        assert!(!take_legacy_untagged_catalogue_notice());

        // An unknown tag is a file this loader cannot interpret. Reject, never
        // guess — guessing is the defect being closed.
        write_catalogue_tagged(&path, Body::Sun.id(), &[[1.0, 2.0, 3.0]], 0x05, 0x01);
        let error = PrecomputedEphemeris::load(&path).expect_err("unknown scale tag must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        write_catalogue_tagged(&path, Body::Sun.id(), &[[1.0, 2.0, 3.0]], 0x01, 0x03);
        let error =
            PrecomputedEphemeris::load(&path).expect_err("unknown representation tag must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        let _ = fs::remove_dir_all(temp_dir);
    }

    /// The four sealed catalogues compiled into the binary are untagged.
    /// This pins that fact so a re-stamp has to update it deliberately,
    /// together with the manifest sha256 block.
    #[test]
    fn embedded_sealed_catalogues_are_untagged() {
        for body in Body::DEFAULT {
            let ephem =
                PrecomputedEphemeris::load_embedded(body).expect("embedded catalogue must load");
            assert_eq!(
                ephem.epoch_scale(),
                EpochScale::Unspecified,
                "{} is tagged; update this test and the manifest sha256 block together",
                body.name()
            );
        }
    }

    #[test]
    fn part_a_utc_authority_accepts_only_manifest_bound_embedded_tables() {
        let authority = part_a_ephemeris_authority().expect("compiled Part A authority");
        assert_eq!(
            authority.table_bundle_sha256(),
            embedded_ephemeris_bundle_sha256_hex()
        );
        assert_eq!(authority.manifest_sha256().len(), 64);

        let utc = jb_rs::drivers::UtcJulianDay::new(2_460_000.5).expect("finite UTC JD");
        let embedded = PrecomputedEphemeris::load_embedded(Body::Sun).expect("embedded Sun");
        let typed = embedded
            .position_at_part_a_utc_jd(utc)
            .expect("manifest-bound UTC lookup");
        let raw = embedded
            .get_position(utc.as_f64())
            .expect("same in-range lookup");
        assert_eq!(typed.map(f64::to_bits), raw.map(f64::to_bits));

        assert!(legacy_untagged_ephemeris(10.0, 3)
            .position_at_part_a_utc_jd(jb_rs::drivers::UtcJulianDay::new(11.0).unwrap())
            .is_err());
    }

    #[test]
    fn part_a_utc_authority_rejects_manifest_semantic_or_table_swap() {
        let mut semantic: serde_json::Value =
            serde_json::from_slice(EMBEDDED_MANIFEST).expect("tracked manifest JSON");
        *semantic
            .get_mut("epoch_scale")
            .expect("tracked manifest contains epoch scale") =
            serde_json::Value::String("tt".to_owned());
        let semantic_bytes = serde_json::to_vec(&semantic).unwrap();
        assert!(validate_part_a_ephemeris_manifest(&semantic_bytes).is_err());

        let mut table: serde_json::Value =
            serde_json::from_slice(EMBEDDED_MANIFEST).expect("tracked manifest JSON");
        *table
            .get_mut("bodies")
            .and_then(|bodies| bodies.get_mut("sun"))
            .and_then(|sun| sun.get_mut("sha256"))
            .expect("tracked manifest contains sun SHA-256") =
            serde_json::Value::String("00".repeat(32));
        let table_bytes = serde_json::to_vec(&table).unwrap();
        assert!(validate_part_a_ephemeris_manifest(&table_bytes).is_err());

        let mut unknown_top: serde_json::Value =
            serde_json::from_slice(EMBEDDED_MANIFEST).expect("tracked manifest JSON");
        let replaced = unknown_top
            .as_object_mut()
            .expect("tracked manifest must be a top-level object")
            .insert(
                "unexpected_authority".to_owned(),
                serde_json::Value::Bool(true),
            );
        assert!(
            replaced.is_none(),
            "fixture must not already carry unknown fields"
        );
        assert!(
            validate_part_a_ephemeris_manifest(&serde_json::to_vec(&unknown_top).unwrap()).is_err()
        );

        let mut unknown_body: serde_json::Value =
            serde_json::from_slice(EMBEDDED_MANIFEST).expect("tracked manifest JSON");
        let replaced = unknown_body
            .get_mut("bodies")
            .and_then(serde_json::Value::as_object_mut)
            .and_then(|bodies| bodies.get_mut("sun"))
            .and_then(serde_json::Value::as_object_mut)
            .expect("tracked manifest must contain a Sun object")
            .insert(
                "unexpected_authority".to_owned(),
                serde_json::Value::Bool(true),
            );
        assert!(
            replaced.is_none(),
            "fixture must not already carry unknown fields"
        );
        assert!(
            validate_part_a_ephemeris_manifest(&serde_json::to_vec(&unknown_body).unwrap())
                .is_err()
        );
    }

    #[test]
    fn part_a_utc_clamp_is_typed_and_manifest_bound() {
        let embedded = PrecomputedEphemeris::load_embedded(Body::Sun).expect("embedded Sun");
        let (start, end) = embedded.jd_range();
        let before = jb_rs::drivers::UtcJulianDay::new(start - 1.0).unwrap();
        let after = jb_rs::drivers::UtcJulianDay::new(end + 1.0).unwrap();
        assert_eq!(
            embedded
                .position_at_part_a_utc_jd_clamped(before)
                .unwrap()
                .map(f64::to_bits),
            embedded.get_position_clamped(start).map(f64::to_bits)
        );
        assert_eq!(
            embedded
                .position_at_part_a_utc_jd_clamped(after)
                .unwrap()
                .map(f64::to_bits),
            embedded.get_position_clamped(end).map(f64::to_bits)
        );
        assert_eq!(
            legacy_untagged_ephemeris(10.0, 3).position_at_part_a_utc_jd_clamped(
                jb_rs::drivers::UtcJulianDay::new(11.0).unwrap()
            ),
            Err(EpochScaleError::PartAManifestAuthorityRequired)
        );
    }

    #[test]
    fn dynamic_arc_validation_includes_both_endpoints_and_backward_arcs() {
        let all = AllPrecomputedEphemeris {
            sun: Some(test_ephemeris(10.0, 3)),
            ..Default::default()
        };

        all.validate_dynamic_arc(Body::Sun.force_flag(), 10.0, 12.0)
            .expect("catalogue endpoints are inclusive");
        all.validate_dynamic_arc(Body::Sun.force_flag(), 12.0, 10.0)
            .expect("backward arc must normalize endpoints");
    }

    #[test]
    fn dynamic_arc_validation_rejects_one_ulp_outside_catalogue() {
        let all = AllPrecomputedEphemeris {
            sun: Some(test_ephemeris(10.0, 3)),
            ..Default::default()
        };
        let one_ulp_after_end = f64::from_bits(12.0_f64.to_bits() + 1);

        let error = all
            .validate_dynamic_arc(Body::Sun.force_flag(), 10.0, one_ulp_after_end)
            .expect_err("one ULP outside must fail before RHS construction");
        assert!(matches!(
            error,
            EphemerisCoverageError::OutsideRange {
                body: Body::Sun,
                ..
            }
        ));
    }

    #[test]
    fn dynamic_arc_validation_reports_missing_body_as_typed_error() {
        let all = AllPrecomputedEphemeris::default();

        let error = all
            .validate_dynamic_arc(Body::Moon.force_flag(), 10.0, 11.0)
            .expect_err("missing requested body must fail closed");
        assert_eq!(
            error,
            EphemerisCoverageError::MissingBody { body: Body::Moon }
        );
    }

    #[test]
    fn selected_catalogue_validation_failure_keeps_io_kind() {
        let _guard = ephemeris_test_guard();
        reset_global();
        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_selected_error_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp catalogue directory");
        fs::write(temp_dir.join(Body::Sun.filename()), b"bad")
            .expect("write malformed selected catalogue");

        let error = load_precomputed_ephemeris_from_dirs(
            Body::Sun.force_flag(),
            std::slice::from_ref(&temp_dir),
        )
        .expect_err("selected malformed catalogue must report its validation failure");

        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        let _ = fs::remove_dir_all(temp_dir);
        reset_global();
    }

    #[test]
    fn catalogue_failure_exposes_owned_typed_source() {
        let error = EphemerisCoverageError::catalogue_io(
            Body::Sun.force_flag(),
            "loading selected catalogue",
            io::Error::new(io::ErrorKind::PermissionDenied, "hostile permissions"),
        );

        let source = std::error::Error::source(&error).expect("typed catalogue source");
        let io = source
            .downcast_ref::<io::Error>()
            .expect("catalogue I/O source stays downcastable");
        assert_eq!(io.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn empty_catalogue_is_rejected_before_range_metadata_can_underflow() {
        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_empty_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let path = temp_dir.join("sun.bin");
        write_catalogue(&path, Body::Sun.id(), &[]);

        let error = PrecomputedEphemeris::load(&path)
            .expect_err("zero-sample catalogue must fail at load time");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_merges_when_new_bodies_requested() {
        let _guard = ephemeris_test_guard();
        reset_global();

        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_merge_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");

        write_catalogue(
            &temp_dir.join("sun.bin"),
            Body::Sun.id(),
            &[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        );
        write_catalogue(
            &temp_dir.join("moon.bin"),
            Body::Moon.id(),
            &[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
        );

        let search_dirs = [temp_dir.clone()];

        load_precomputed_ephemeris_from_dirs(Body::Sun.force_flag(), &search_dirs)
            .expect("load sun");
        let loaded = get_precomputed_ephemeris().expect("ephem loaded");
        assert!(loaded.sun.is_some());
        assert!(loaded.moon.is_none());

        load_precomputed_ephemeris_from_dirs(
            Body::Sun.force_flag() | Body::Moon.force_flag(),
            &search_dirs,
        )
        .expect("load sun+moon");
        let loaded = get_precomputed_ephemeris().expect("ephem loaded");
        assert!(loaded.sun.is_some());
        assert!(loaded.moon.is_some());

        let _ = fs::remove_dir_all(&temp_dir);
        reset_global();
    }

    #[test]
    fn catalogue_content_identity_matches_source_and_stays_immutable() {
        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_identity_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let path = temp_dir.join("sun.bin");
        write_catalogue(&path, Body::Sun.id(), &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let expected = lowercase_hex(&Sha256::digest(fs::read(&path).expect("read catalogue")));
        let loaded = PrecomputedEphemeris::load(&path).expect("load catalogue");

        assert_eq!(loaded.content_sha256_hex(), expected);
        mutate_catalogue_byte(&path, HEADER_SIZE, 0xff);
        assert_eq!(
            loaded.content_sha256_hex(),
            expected,
            "loaded content identity must not follow later file mutation"
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_cold_partial_sun_then_moon_then_flags_15_loads_both_bodies() {
        // Asserts presence (`is_some`) of both tables after the sequence
        // only; it does not compare loaded positions against the written
        // catalogues, so a flags=15 load substituting different table data
        // would still pass.
        let _guard = ephemeris_test_guard();
        reset_global();

        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_partial_sequence_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        write_catalogue(
            &temp_dir.join("sun.bin"),
            Body::Sun.id(),
            &[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        );
        write_catalogue(
            &temp_dir.join("moon.bin"),
            Body::Moon.id(),
            &[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
        );
        let search_dirs = [temp_dir.clone()];

        load_precomputed_ephemeris_from_dirs(Body::Sun.force_flag(), &search_dirs)
            .expect("load sun");
        load_precomputed_ephemeris_from_dirs(Body::Moon.force_flag(), &search_dirs)
            .expect("merge moon");
        load_precomputed_ephemeris_from_dirs(15, &search_dirs)
            .expect("flags=15 must retain sun and moon");

        let loaded = get_precomputed_ephemeris().expect("ephem loaded");
        assert!(loaded.sun.is_some());
        assert!(loaded.moon.is_some());

        let _ = fs::remove_dir_all(&temp_dir);
        reset_global();
    }

    #[test]
    fn extending_partial_global_never_repins_mutated_loaded_catalogue() {
        let _guard = ephemeris_test_guard();
        reset_global();

        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_pinned_identity_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let sun_path = temp_dir.join("sun.bin");
        let moon_path = temp_dir.join("moon.bin");
        write_catalogue(
            &sun_path,
            Body::Sun.id(),
            &[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        );
        let search_dirs = [temp_dir.clone()];

        load_precomputed_ephemeris_from_dirs(Body::Sun.force_flag(), &search_dirs)
            .expect("load sun");
        let initially_loaded = get_precomputed_ephemeris().expect("sun loaded");
        let pinned_sha = initially_loaded
            .get(Body::Sun)
            .expect("sun loaded")
            .content_sha256_hex();
        let pinned_position = initially_loaded
            .get(Body::Sun)
            .and_then(|table| table.get_position(2_458_849.5))
            .expect("sun position");

        write_catalogue(
            &sun_path,
            Body::Sun.id(),
            &[[101.0, 0.0, 0.0], [102.0, 0.0, 0.0]],
        );
        let error = load_precomputed_ephemeris_from_dirs(
            Body::Sun.force_flag() | Body::Moon.force_flag(),
            &search_dirs,
        )
        .expect_err("missing moon must fail closed");
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        let after_failed_extension = get_precomputed_ephemeris().expect("sun stays loaded");
        assert_eq!(
            after_failed_extension
                .get(Body::Sun)
                .expect("sun stays loaded")
                .content_sha256_hex(),
            pinned_sha
        );
        assert_eq!(
            after_failed_extension
                .get(Body::Sun)
                .and_then(|table| table.get_position(2_458_849.5))
                .expect("sun position stays loaded")
                .map(f64::to_bits),
            pinned_position.map(f64::to_bits)
        );

        write_catalogue(
            &moon_path,
            Body::Moon.id(),
            &[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
        );
        load_precomputed_ephemeris_from_dirs(
            Body::Sun.force_flag() | Body::Moon.force_flag(),
            &search_dirs,
        )
        .expect("merge missing moon");
        let extended = get_precomputed_ephemeris().expect("sun and moon loaded");
        assert_eq!(
            extended
                .get(Body::Sun)
                .expect("sun stays loaded")
                .content_sha256_hex(),
            pinned_sha
        );
        assert_eq!(
            extended
                .get(Body::Sun)
                .and_then(|table| table.get_position(2_458_849.5))
                .expect("sun position stays loaded")
                .map(f64::to_bits),
            pinned_position.map(f64::to_bits)
        );
        assert!(extended.get(Body::Moon).is_some());

        let _ = fs::remove_dir_all(&temp_dir);
        reset_global();
    }

    #[test]
    fn test_partial_global_does_not_report_missing_requested_body_as_loaded() {
        let _guard = ephemeris_test_guard();
        reset_global();

        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_partial_missing_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        write_catalogue(
            &temp_dir.join("sun.bin"),
            Body::Sun.id(),
            &[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        );
        let search_dirs = [temp_dir.clone()];

        load_precomputed_ephemeris_from_dirs(Body::Sun.force_flag(), &search_dirs)
            .expect("load sun");
        let error = load_precomputed_ephemeris_from_dirs(Body::Moon.force_flag(), &search_dirs)
            .expect_err("missing requested moon catalogue must fail closed");
        assert_eq!(error.kind(), io::ErrorKind::NotFound);

        let _ = fs::remove_dir_all(&temp_dir);
        reset_global();
    }

    #[test]
    fn default_search_paths_are_empty_and_ignore_retired_env_dir() {
        let retired = concat!("NASA_DUST_", "EPHEMERIS_DIR");
        std::env::set_var(retired, "/definitely/retired/ephemeris");
        let paths = ephemeris_search_paths();
        std::env::remove_var(retired);
        assert!(paths.is_empty());
    }

    #[test]
    fn search_paths_do_not_depend_on_compile_time_absolute_source_paths() {
        assert!(
            ephemeris_search_paths()
                .iter()
                .all(|path| !path.is_absolute()),
            "release ephemeris lookup must not depend on a compile-time absolute source path"
        );
    }

    #[test]
    fn embedded_catalogues_load_without_filesystem_search_paths() {
        let _guard = ephemeris_test_guard();
        reset_global();

        let flags = Body::DEFAULT
            .iter()
            .fold(0, |flags, body| flags | body.force_flag());
        load_precomputed_ephemeris_from_embedded_and_dirs(flags, &[])
            .expect("tracked default catalogues must load from binary bytes");
        let loaded = get_precomputed_ephemeris().expect("embedded catalogues loaded");

        for body in Body::DEFAULT {
            let expected = lowercase_hex(&Sha256::digest(
                embedded_catalogue(body).expect("embedded body"),
            ));
            assert_eq!(
                loaded
                    .get(body)
                    .expect("requested embedded body loaded")
                    .content_sha256_hex(),
                expected,
                "embedded {} bytes must preserve tracked catalogue identity",
                body.name()
            );
        }

        reset_global();
    }

    #[test]
    fn embedded_catalogue_hash_api_binds_tracked_bytes() {
        for (body, expected) in [
            (
                Body::Sun,
                "883e095b8a6e8df59cd528828707f6c578f6d1ffbbb05bee2df42d32d716c5b4",
            ),
            (
                Body::Moon,
                "ca64a1a8b0e043002388f43244dd169b1439ef323f883ec6cca9b7df5f731e5c",
            ),
            (
                Body::Jupiter,
                "d9d506bd9227870e594ba7894a9652e8bd58de333b75cd08b3bf8cbf93163437",
            ),
            (
                Body::Venus,
                "69b82d3502f10d4d2b8db337373d2a858c4ac235bb7ce2856f507d8d0f30930f",
            ),
        ] {
            assert_eq!(embedded_catalogue_sha256_hex(body), Some(expected));
        }
        assert_eq!(embedded_catalogue_sha256_hex(Body::Mars), None);
        assert_eq!(embedded_catalogue_sha256_hex(Body::Saturn), None);
    }

    /// The build.rs-baked tables against the kept runtime parser, bit for bit.
    ///
    /// build.rs re-implements the header validation, the byte decode, and —
    /// the only FP arithmetic — a TOKEN COPY of the direction-rate supremum.
    /// This oracle runs the runtime parser over the same tracked bytes and
    /// bit-compares every baked value, so drift in either copy is a red test
    /// here rather than a silent skew in the flown tables.
    #[test]
    fn generated_embedded_tables_are_bit_identical_to_the_parser() {
        for body in Body::DEFAULT {
            let bytes = embedded_catalogue(body).expect("tracked catalogue bytes");
            let parsed = PrecomputedEphemeris::load_bytes(bytes, Some(body))
                .expect("tracked catalogue parses");
            let table = embedded_table(body).expect("baked catalogue table");
            let name = body.name();
            assert_eq!(parsed.n_samples, table.n_samples, "{name}: n_samples");
            assert_eq!(
                parsed.jd_start.to_bits(),
                table.jd_start.to_bits(),
                "{name}: jd_start"
            );
            assert_eq!(
                parsed.jd_range().1.to_bits(),
                table.jd_end.to_bits(),
                "{name}: jd_end"
            );
            assert_eq!(
                parsed.dt_days.to_bits(),
                table.dt_days.to_bits(),
                "{name}: dt_days"
            );
            assert_eq!(
                parsed.max_direction_rate_per_day.to_bits(),
                table.max_direction_rate_per_day.to_bits(),
                "{name}: max_direction_rate_per_day"
            );
            assert_eq!(
                EpochScale::from_tag(table.epoch_scale_tag),
                Some(parsed.epoch_scale),
                "{name}: epoch scale tag"
            );
            assert_eq!(
                EpochRepresentation::from_tag(table.epoch_representation_tag),
                Some(parsed.epoch_representation),
                "{name}: epoch representation tag"
            );
            assert_eq!(parsed.content_sha256, table.content_sha256, "{name}: sha");
            assert_eq!(
                parsed.content_sha256_hex(),
                table.content_sha256_hex,
                "{name}: sha hex"
            );
            assert_eq!(bytes.len(), table.size_bytes, "{name}: size");
            assert_eq!(body.id(), table.body_id, "{name}: body id");
            assert_eq!(
                parsed.positions.len(),
                table.positions.len(),
                "{name}: position count"
            );
            let mismatched = parsed
                .positions
                .iter()
                .zip(table.positions.iter())
                .filter(|(runtime, baked)| runtime.to_bits() != baked.to_bits())
                .count();
            assert_eq!(mismatched, 0, "{name}: {mismatched} position bit diffs");
        }

        // Bundle digest oracle: recompute the record format over the tracked
        // bytes and compare against the baked const.
        let mut hasher = Sha256::new();
        for body in Body::DEFAULT {
            let bytes = embedded_catalogue(body).expect("tracked catalogue bytes");
            hasher.update(body.name().as_bytes());
            hasher.update(b"=");
            hasher.update(lowercase_hex(&Sha256::digest(bytes)).as_bytes());
            hasher.update(b"\n");
        }
        assert_eq!(
            lowercase_hex(&hasher.finalize()),
            embedded_ephemeris_bundle_sha256_hex(),
            "bundle digest"
        );
    }

    #[test]
    fn embedded_ephemeris_bundle_hash_binds_body_order_and_hashes() {
        assert_eq!(
            embedded_ephemeris_bundle_sha256_hex(),
            "c89a49993c6f4f3284be9262e8f2804b94a497c4bcf1e05853e59b739294eb6f"
        );
    }

    #[test]
    fn embedded_loader_rejects_conflicting_cached_default_body() {
        let _guard = ephemeris_test_guard();
        reset_global();

        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_conflicting_cache_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        write_catalogue(
            &temp_dir.join(Body::Sun.filename()),
            Body::Sun.id(),
            &[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        );
        let search_dirs = [temp_dir.clone()];
        load_precomputed_ephemeris_from_dirs(Body::Sun.force_flag(), &search_dirs)
            .expect("seed explicit Sun catalogue");
        let cached_sha = get_precomputed_ephemeris()
            .expect("explicit Sun loaded")
            .get(Body::Sun)
            .expect("Sun loaded")
            .content_sha256_hex();
        assert_ne!(
            cached_sha,
            embedded_catalogue_sha256_hex(Body::Sun).expect("embedded Sun identity")
        );

        let error = load_precomputed_ephemeris_from_embedded_and_dirs(Body::Sun.force_flag(), &[])
            .expect_err("embedded loader must reject conflicting cached Sun");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            get_precomputed_ephemeris()
                .expect("conflicting cache remains inspectable")
                .get(Body::Sun)
                .expect("cached Sun remains present")
                .content_sha256_hex(),
            cached_sha
        );

        let _ = fs::remove_dir_all(&temp_dir);
        reset_global();
    }

    #[test]
    fn try_loader_returns_none_for_conflicting_cached_default_body() {
        let _guard = ephemeris_test_guard();
        reset_global();

        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_conflicting_try_cache_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        write_catalogue(
            &temp_dir.join(Body::Sun.filename()),
            Body::Sun.id(),
            &[[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        );
        let search_dirs = [temp_dir.clone()];
        load_precomputed_ephemeris_from_dirs(Body::Sun.force_flag(), &search_dirs)
            .expect("seed conflicting explicit Sun catalogue");

        assert!(
            try_load_precomputed_ephemeris(Body::Sun.force_flag()).is_none(),
            "try loader must not expose cache after embedded identity rejection"
        );

        let _ = fs::remove_dir_all(&temp_dir);
        reset_global();
    }

    #[test]
    fn embedded_loader_rejects_unrequested_conflicting_cached_default_body() {
        let _guard = ephemeris_test_guard();
        reset_global();

        let temp_dir = std::env::temp_dir().join(format!(
            "dust_ephem_cross_body_conflicting_cache_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        write_catalogue(
            &temp_dir.join(Body::Sun.filename()),
            Body::Sun.id(),
            &[[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        );
        let search_dirs = [temp_dir.clone()];
        load_precomputed_ephemeris_from_dirs(Body::Sun.force_flag(), &search_dirs)
            .expect("seed conflicting explicit Sun catalogue");

        let error = load_precomputed_ephemeris_from_embedded_and_dirs(Body::Moon.force_flag(), &[])
            .expect_err("Moon request must reject conflicting cached Sun");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(
            get_precomputed_ephemeris()
                .expect("conflicting Sun cache remains inspectable")
                .get(Body::Moon)
                .is_none(),
            "failed load must not publish embedded Moon beside conflicting Sun"
        );
        assert!(
            try_load_precomputed_ephemeris(Body::Moon.force_flag()).is_none(),
            "try loader must fail closed on unrequested conflicting cached Sun"
        );

        let _ = fs::remove_dir_all(&temp_dir);
        reset_global();
    }
}
