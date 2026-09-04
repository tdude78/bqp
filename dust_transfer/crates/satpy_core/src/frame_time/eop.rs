//! Earth-orientation input handling, transliterated from the sealed 4AF
//! generator's `load_eop`, `dat_at_mjd`, and double-double `lagrange`.
//!
//! Real-EOP cases interpolate the four `finals2000A.all` nodes bracketing the
//! anchor (`center_mjd-1..center_mjd+2`) with a double-double four-node
//! Lagrange polynomial in continuous-TAI seconds. Zero-EOP cases bypass this.

use super::dd::{from, Dd};
use super::timescale::{dat, jd2cal, DJM0};
use num_traits::ToPrimitive;
use std::fmt;

/// Failure while decoding or interpolating Bulletin-A EOP input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EopError {
    /// A fixed-width field falls outside the supplied record.
    FieldRange {
        /// Zero-based field start byte.
        start: usize,
        /// Fixed field width in bytes.
        width: usize,
        /// Record byte length.
        input_len: usize,
    },
    /// A field is not ASCII, numeric, or finite.
    InvalidField {
        /// Zero-based field start byte.
        start: usize,
        /// Fixed field width in bytes.
        width: usize,
    },
    /// The MJD field cannot be represented as an integer MJD.
    InvalidMjd,
    /// A selected record has an invalid Bulletin-A quality flag.
    InvalidFlag {
        /// Requested MJD.
        wanted_mjd: i32,
        /// Zero-based flag byte.
        column: usize,
    },
    /// More than one record matched an MJD.
    DuplicateRecord {
        /// Requested MJD.
        wanted_mjd: i32,
    },
    /// No complete record matched an MJD.
    MissingRecord {
        /// Requested MJD.
        wanted_mjd: i32,
    },
    /// The embedded UTC/TAI conversion rejected an otherwise parsed date.
    TimeScale(&'static str),
}

impl fmt::Display for EopError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldRange {
                start,
                width,
                input_len,
            } => write!(
                formatter,
                "EOP field {start}..{} exceeds record length {input_len}",
                start.saturating_add(*width)
            ),
            Self::InvalidField { start, width } => write!(
                formatter,
                "invalid EOP field {start}..{}",
                start.saturating_add(*width)
            ),
            Self::InvalidMjd => formatter.write_str("invalid EOP MJD field"),
            Self::InvalidFlag { wanted_mjd, column } => {
                write!(
                    formatter,
                    "invalid EOP flag at byte {column} for MJD {wanted_mjd}"
                )
            }
            Self::DuplicateRecord { wanted_mjd } => {
                write!(formatter, "duplicate EOP record for MJD {wanted_mjd}")
            }
            Self::MissingRecord { wanted_mjd } => {
                write!(formatter, "missing EOP record for MJD {wanted_mjd}")
            }
            Self::TimeScale(reason) => {
                write!(formatter, "EOP time-scale conversion failed: {reason}")
            }
        }
    }
}

impl std::error::Error for EopError {}

/// Raw Bulletin-A row values (arcsec / seconds, before radian scaling).
#[derive(Clone, Copy, Debug)]
pub struct EopRow {
    pub xp: f64,
    pub yp: f64,
    pub dut1: f64,
    pub dx: f64,
    pub dy: f64,
}

fn fixed_field(bytes: &[u8], start: usize, width: usize) -> Result<f64, EopError> {
    let end = start.checked_add(width).ok_or(EopError::FieldRange {
        start,
        width,
        input_len: bytes.len(),
    })?;
    let slice = bytes.get(start..end).ok_or(EopError::FieldRange {
        start,
        width,
        input_len: bytes.len(),
    })?;
    let text = std::str::from_utf8(slice).map_err(|_| EopError::InvalidField { start, width })?;
    let value = text
        .trim()
        .parse::<f64>()
        .map_err(|_| EopError::InvalidField { start, width })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EopError::InvalidField { start, width })
    }
}

/// Load the single Bulletin-A record for `wanted_mjd`, matching the generator's
/// fixed-width field offsets and I/P flag validation.
///
/// # Errors
///
/// Returns [`EopError`] for malformed, missing, duplicate, or invalid-flag data.
pub fn load_eop(finals: &str, wanted_mjd: i32) -> Result<EopRow, EopError> {
    let mut result = None;
    for line in finals.lines() {
        let bytes = line.as_bytes();
        if bytes.len() < 125 {
            continue;
        }
        let mjd = fixed_field(bytes, 7, 8)?
            .to_i32()
            .ok_or(EopError::InvalidMjd)?;
        if mjd != wanted_mjd {
            continue;
        }
        for column in [16, 57, 95] {
            let flag_ok = bytes
                .get(column)
                .is_some_and(|flag| *flag == b'I' || *flag == b'P');
            if !flag_ok {
                return Err(EopError::InvalidFlag { wanted_mjd, column });
            }
        }
        if result.is_some() {
            return Err(EopError::DuplicateRecord { wanted_mjd });
        }
        result = Some(EopRow {
            xp: fixed_field(bytes, 18, 9)?,
            yp: fixed_field(bytes, 37, 9)?,
            dut1: fixed_field(bytes, 58, 10)?,
            dx: fixed_field(bytes, 97, 9)?,
            dy: fixed_field(bytes, 116, 9)?,
        });
    }
    result.ok_or(EopError::MissingRecord { wanted_mjd })
}

/// `TAI - UTC` at the start of the given integer MJD (seconds).
///
/// # Errors
///
/// Returns [`EopError`] when ERFA-compatible UTC/TAI conversion rejects `mjd`.
pub fn dat_at_mjd(mjd: i32) -> Result<f64, EopError> {
    let (status, year, month, day, _fraction) = jd2cal(DJM0, f64::from(mjd));
    if status != 0 {
        return Err(EopError::TimeScale("node MJD out of range"));
    }
    let (status, delta_at) = dat(year, month, day, 0.0);
    if status < 0 || !delta_at.is_finite() {
        return Err(EopError::TimeScale("node TAI-UTC unavailable"));
    }
    Ok(delta_at)
}

/// Double-double four-node Lagrange interpolation.
#[must_use]
pub fn lagrange(x: Dd, xs: &[Dd; 4], ys: &[f64; 4]) -> Dd {
    let &[x0, x1, x2, x3] = xs;
    let &[y0, y1, y2, y3] = ys;
    let abscissae = [x0, x1, x2, x3];
    let values = [y0, y1, y2, y3];
    let mut sum = from(0.0);
    for (node_index, (node_x, node_value)) in abscissae.into_iter().zip(values).enumerate() {
        let mut term = from(node_value);
        for (other_index, other_x) in abscissae.into_iter().enumerate() {
            if other_index != node_index {
                term = term.mul_dd(x.sub_dd(other_x).div_dd(node_x.sub_dd(other_x)));
            }
        }
        sum = sum.add_dd(term);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_field_does_not_panic() {
        assert!(matches!(
            fixed_field(b"not-a-number", 0, 12),
            Err(EopError::InvalidField {
                start: 0,
                width: 12
            })
        ));
    }

    #[test]
    fn missing_record_does_not_panic() {
        assert!(matches!(
            load_eop("", 58_000),
            Err(EopError::MissingRecord { wanted_mjd: 58_000 })
        ));
    }
}
