//! Segment-cached GCRS->ITRS frame authority for production dynamics.
//!
//! The exact chain in [`super::chain`] costs 23.7 us per evaluation with zero
//! EOP and 2.36 ms with real EOP, against a 290.9 ns production RHS evaluation.
//! It cannot be called per integrator stage. This module caches the
//! slowly-varying factors on a canonical segment grid and leaves only the
//! Earth-rotation angle to be evaluated per stage.
//!
//! # Exactness of the fast factor
//!
//! `R3(theta) = cos(theta) E1 + sin(theta) E2 + E3` with `E1 = diag(1,1,0)`,
//! `E2 = [[0,1,0],[-1,0,0],[0,0,0]]` and `E3 = diag(0,0,1)`, so
//! `R = RPOM R3(theta) RC2I` becomes exactly
//! `R = cos(theta) M1 + sin(theta) M2 + M3` with `M_i = RPOM E_i RC2I`.
//! With `w_i` the columns of `RPOM` and `q_i^T` the rows of `RC2I`, the three
//! matrices are `w1 q1^T + w2 q2^T`, `w1 q2^T - w2 q1^T` and `w3 q3^T`, of rank
//! 2, 2 and 1. The decomposition is an identity, not an approximation: all of
//! the modelling error lives in holding `RPOM` and `RC2I` on a segment.
//!
//! # Segmentation
//!
//! The index is two-level, `(j, k)`. A uniform grid cannot also align to leap
//! instants: successive leaps are separated in TAI by an integer number of days
//! plus one second, so no single epoch yields an hour grid aligned to every
//! leap. Here `j` selects the leap interval and `k = floor((tai_s - t_j) / W)`
//! with `W = 1800 s`. Because leaps fall at UTC midnight and 86400 is divisible
//! by 1800, the remainder of every interval is exactly one second, which the
//! interval's final segment absorbs. `TAI - UTC` is therefore constant across a
//! segment and the UTC needed by the JB drivers is one fused multiply-add.
//!
//! `(j, k)` is a pure function of absolute TAI given the frozen leap table, so
//! the segmentation does not depend on how work was cut into arcs, how rows
//! were batched, or how many workers ran. Each segment is built from `(j, k)`
//! alone with no cross-segment carry, recurrence, or accumulation.

use std::cell::RefCell;
use std::sync::OnceLock;

use num_traits::ToPrimitive;
use sha2::{Digest, Sha256};

use super::cio::{c2ixys, pom00, sp00, Mat3};
use super::dd::from;
use super::era::era;
use super::iau2006::xys06a;
use super::timescale::{dat, jd2cal, taitt, utctai, DAS2R, DAYSEC, DJM0};

/// Canonical segment width in TAI seconds; the half-width is 900 s.
///
/// The in-segment residual against the exact chain IS quadratic in the
/// half-width. Measured, segment centred 2022-08-12T04:30, worst over a 33-point
/// sweep at 7000 km (odd step count, so samples land on fractional seconds and
/// exercise the `tai_s` rounding rather than sitting on representable instants):
///
/// | half-width s | worst vs chain | as rad |
/// |---|---|---|
/// | 1800 | 0.231948 mm | 3.3135e-11 |
/// |  900 | 0.067382 mm | 9.6260e-12 |
/// |  450 | 0.055760 mm | 7.9657e-12 |
/// |  225 | 0.055409 mm | 7.9156e-12 |
///
/// On exactly-representable instants the halving ratios are 4.0026, 4.0012 and
/// 4.0009 — clean quadratic convergence. The flattening below a 900 s half-width
/// is NOT the model failing to converge; it is the 0.121700 mm `tai_s` staircase
/// (ULP `2^-22 s` at 1.5655e9), which no choice of `W` can move.
///
/// `W = 1800` is chosen so the 1e-10 element bound of the Task 5B routing REDs
/// clears its OWN stated rationale — "at least five times the element-equivalent
/// of the in-segment residual" — at 10.39x, rather than the 3.02x that `W = 3600`
/// would leave. A half-width below 900 s buys almost nothing because the `tai_s`
/// staircase dominates from there down.
///
/// An earlier revision of this comment claimed 0.13 mm at a 1800 s half-width
/// "by quadratic scaling". That figure was never measured and is superseded by
/// the table; the scaling law it invoked, however, is correct.
pub const SEGMENT_WIDTH_S: f64 = 1800.0;

/// Declared fast-path accuracy bound, metres of position error at 7000 km.
pub const FAST_PATH_BOUND_M: f64 = 1.0e-3;

/// Interpolation degree of the in-segment correction (linear).
pub const SEGMENT_DEGREE: u32 = 1;

/// Version string of the underlying exact chain, mixed into the authority id.
pub const CHAIN_VERSION: &str = "erfa-2.0.1-sofa-20231011-frame-time-v3";

/// Sealed EOP table: 19,645 contiguous complete records over MJD 41684..61328,
/// five little-endian binary64 each, emitted by
/// `scripts/regenerate-frame-time-eop.sh`. Never parsed from text at runtime.
static EOP_TABLE: &[u8] = include_bytes!("eop_table.bin");

const EOP_FIRST_MJD: i32 = 41684;
const EOP_LAST_MJD: i32 = 61328;
const EOP_RECORD_BYTES: usize = 40;
const EOP_RECORDS: usize = 19_645;

/// One Bulletin-A record in source units: arcseconds, seconds, milliarcseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EopRecord {
    pub xp_arcsec: f64,
    pub yp_arcsec: f64,
    pub dut1_s: f64,
    pub dx_mas: f64,
    pub dy_mas: f64,
}

/// Why a frame resolution failed. All variants are fail-closed: the caller must
/// not fall back to an extrapolated or reused row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameAuthorityError {
    /// The epoch lies outside the sealed EOP table's complete-record span.
    EpochOutsideSealedSpan { mjd: i32 },
    /// The sealed table did not have the expected shape.
    SealedTableCorrupt(&'static str),
    /// A time-scale conversion in the sealed chain rejected the epoch.
    TimeScale(&'static str),
}

impl std::fmt::Display for FrameAuthorityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EpochOutsideSealedSpan { mjd } => write!(
                f,
                "epoch MJD {mjd} is outside the sealed EOP span {EOP_FIRST_MJD}..{EOP_LAST_MJD}"
            ),
            Self::SealedTableCorrupt(why) => write!(f, "sealed EOP table is corrupt: {why}"),
            Self::TimeScale(why) => write!(f, "time-scale conversion failed: {why}"),
        }
    }
}

impl std::error::Error for FrameAuthorityError {}

/// Read one sealed EOP record. The MJD is implied by the record index, so the
/// table's contiguity is load-bearing and is asserted by the generator.
///
/// # Errors
///
/// Returns an error when `mjd` is outside the sealed span or a sealed record is
/// structurally unavailable.
pub fn eop_record(mjd: i32) -> Result<EopRecord, FrameAuthorityError> {
    if !(EOP_FIRST_MJD..=EOP_LAST_MJD).contains(&mjd) {
        return Err(FrameAuthorityError::EpochOutsideSealedSpan { mjd });
    }
    let relative_mjd = mjd
        .checked_sub(EOP_FIRST_MJD)
        .ok_or(FrameAuthorityError::SealedTableCorrupt("record index"))?;
    let index = usize::try_from(relative_mjd)
        .map_err(|_| FrameAuthorityError::SealedTableCorrupt("record index"))?;
    let offset = index
        .checked_mul(EOP_RECORD_BYTES)
        .ok_or(FrameAuthorityError::SealedTableCorrupt("record offset"))?;
    let record_end = offset
        .checked_add(EOP_RECORD_BYTES)
        .ok_or(FrameAuthorityError::SealedTableCorrupt("record end"))?;
    if record_end > EOP_TABLE.len() {
        return Err(FrameAuthorityError::SealedTableCorrupt("record past end"));
    }
    let value = |slot: usize| -> Result<f64, FrameAuthorityError> {
        let field_offset = slot
            .checked_mul(8)
            .ok_or(FrameAuthorityError::SealedTableCorrupt("field offset"))?;
        let base = offset
            .checked_add(field_offset)
            .ok_or(FrameAuthorityError::SealedTableCorrupt("field base"))?;
        let field_end = base
            .checked_add(8)
            .ok_or(FrameAuthorityError::SealedTableCorrupt("field end"))?;
        let bytes = <&[u8; 8]>::try_from(
            EOP_TABLE
                .get(base..field_end)
                .ok_or(FrameAuthorityError::SealedTableCorrupt("field past end"))?,
        )
        .map_err(|_| FrameAuthorityError::SealedTableCorrupt("field width"))?;
        Ok(f64::from_le_bytes(*bytes))
    };
    Ok(EopRecord {
        xp_arcsec: value(0)?,
        yp_arcsec: value(1)?,
        dut1_s: value(2)?,
        dx_mas: value(3)?,
        dy_mas: value(4)?,
    })
}

/// Leap instants as continuous-TAI seconds since the sealed epoch, paired with
/// the `TAI - UTC` that holds from that instant onward.
///
/// Built from the same `dat` table the exact chain uses, restricted to the
/// integer-leap era. The table starts at MJD 41684 (1973-01-01), so pre-1972
/// rubber seconds and negative leaps cannot occur.
fn leap_intervals() -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    // Walk the sealed span day by day at UTC midnight; a change in `dat` marks
    // a leap instant. This reads the same authority the chain does rather than
    // duplicating the change list.
    let mut previous: Option<f64> = None;
    for mjd in EOP_FIRST_MJD..=EOP_LAST_MJD {
        let (status, y, m, d, _f) = jd2cal(DJM0, f64::from(mjd));
        if status != 0 {
            continue;
        }
        let (s, delta_at) = dat(y, m, d, 0.0);
        if s < 0 {
            continue;
        }
        let changed = previous
            .is_none_or(|previous_delta_at| previous_delta_at.to_bits() != delta_at.to_bits());
        if changed {
            // TAI seconds of this UTC midnight, measured from the span start.
            let Some(days_since_start) = mjd.checked_sub(EOP_FIRST_MJD) else {
                continue;
            };
            let tai_s = f64::from(days_since_start) * DAYSEC + delta_at;
            out.push((tai_s, delta_at));
            previous = Some(delta_at);
        }
    }
    out
}

/// One canonical segment's frozen state. Six matrices plus the theta cubic.
#[derive(Clone, Copy, Debug)]
pub struct FrameSegment {
    /// `R = cos(theta)(M1 + dt K1) + sin(theta)(M2 + dt K2) + (M3 + dt K3)`.
    pub m: [Mat3; 3],
    pub k: [Mat3; 3],
    /// Theta cubic in `dt` seconds from the segment centre.
    pub b: [f64; 4],
    /// `TAI - UTC`, constant across the segment by construction.
    pub delta_at_s: f64,
    /// Segment centre, continuous TAI seconds from the sealed span start.
    pub centre_tai_s: f64,
}

/// A resolved per-stage rotation. Position uses `r`, acceleration uses `r^T`.
#[derive(Clone, Copy, Debug)]
pub struct FrameRotation {
    /// GCRS-to-ITRS passive rotation at this stage.
    pub r: Mat3,
    /// `TAI - UTC` for the enclosing sealed interval.
    pub delta_at_s: f64,
    /// Angular velocity of ITRS relative to GCRS, expressed in GCRS.
    pub itrs_angular_velocity_gcrs: [f64; 3],
}

impl FrameRotation {
    /// GCRS -> ITRS on a 3-vector.
    #[inline]
    #[must_use]
    pub fn to_itrs(&self, v: &[f64; 3]) -> [f64; 3] {
        let [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]] = self.r;
        let &[vx, vy, vz] = v;
        [
            r00.mul_add(vx, r01.mul_add(vy, r02 * vz)),
            r10.mul_add(vx, r11.mul_add(vy, r12 * vz)),
            r20.mul_add(vx, r21.mul_add(vy, r22 * vz)),
        ]
    }

    /// ITRS -> GCRS on a 3-vector (the transpose apply).
    #[inline]
    #[must_use]
    pub fn to_gcrs(&self, v: &[f64; 3]) -> [f64; 3] {
        let [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]] = self.r;
        let &[vx, vy, vz] = v;
        [
            r00.mul_add(vx, r10.mul_add(vy, r20 * vz)),
            r01.mul_add(vx, r11.mul_add(vy, r21 * vz)),
            r02.mul_add(vx, r12.mul_add(vy, r22 * vz)),
        ]
    }
}

fn matmul(a: &Mat3, b: &Mat3) -> Mat3 {
    let &[[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]] = a;
    let &[[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]] = b;
    [
        [
            a00 * b00 + a01 * b10 + a02 * b20,
            a00 * b01 + a01 * b11 + a02 * b21,
            a00 * b02 + a01 * b12 + a02 * b22,
        ],
        [
            a10 * b00 + a11 * b10 + a12 * b20,
            a10 * b01 + a11 * b11 + a12 * b21,
            a10 * b02 + a11 * b12 + a12 * b22,
        ],
        [
            a20 * b00 + a21 * b10 + a22 * b20,
            a20 * b01 + a21 * b11 + a22 * b21,
            a20 * b02 + a21 * b12 + a22 * b22,
        ],
    ]
}

fn centred_matrix_slope(plus: &Mat3, minus: &Mat3, half: f64) -> Mat3 {
    let &[[p00, p01, p02], [p10, p11, p12], [p20, p21, p22]] = plus;
    let &[[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] = minus;
    [
        [
            (p00 - m00) / (2.0 * half),
            (p01 - m01) / (2.0 * half),
            (p02 - m02) / (2.0 * half),
        ],
        [
            (p10 - m10) / (2.0 * half),
            (p11 - m11) / (2.0 * half),
            (p12 - m12) / (2.0 * half),
        ],
        [
            (p20 - m20) / (2.0 * half),
            (p21 - m21) / (2.0 * half),
            (p22 - m22) / (2.0 * half),
        ],
    ]
}

fn transpose_apply(matrix: &Mat3, vector: &[f64; 3]) -> [f64; 3] {
    let &[[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] = matrix;
    let &[v0, v1, v2] = vector;
    [
        m00 * v0 + m10 * v1 + m20 * v2,
        m01 * v0 + m11 * v1 + m21 * v2,
        m02 * v0 + m12 * v1 + m22 * v2,
    ]
}

fn theta_coefficients(theta_centre: f64, theta_minus: f64, theta_plus: f64, half: f64) -> [f64; 4] {
    [
        theta_centre,
        (theta_plus - theta_minus) / (2.0 * half),
        (theta_plus - 2.0 * theta_centre + theta_minus) / (2.0 * half * half),
        0.0,
    ]
}

fn segment_centre(start: f64, segment_index: f64) -> f64 {
    start + (segment_index + 0.5) * SEGMENT_WIDTH_S
}

/// `RPOM E_i RC2I` for the three fixed selectors, without materialising `E_i`.
fn split_matrices(rpom: &Mat3, rc2i: &Mat3) -> [Mat3; 3] {
    const E1: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
    const E2: Mat3 = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    const E3: Mat3 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    [
        matmul(rpom, &matmul(&E1, rc2i)),
        matmul(rpom, &matmul(&E2, rc2i)),
        matmul(rpom, &matmul(&E3, rc2i)),
    ]
}

/// The sealed frame authority: the segment grid plus its provenance digest.
pub struct FrameAuthority {
    intervals: Vec<(f64, f64)>,
    authority_sha256: [u8; 32],
    authority_id: u64,
}

impl FrameAuthority {
    /// Build from the compiled sealed bytes. A pure function of those bytes, so
    /// first touch from any thread yields an identical authority.
    fn from_sealed() -> Self {
        assert_eq!(
            EOP_TABLE.len(),
            EOP_RECORDS * EOP_RECORD_BYTES,
            "sealed EOP table must be {} bytes",
            EOP_RECORDS * EOP_RECORD_BYTES
        );
        let intervals = leap_intervals();

        let mut hasher = Sha256::new();
        hasher.update(EOP_TABLE);
        for (tai_s, delta_at) in &intervals {
            hasher.update(tai_s.to_le_bytes());
            hasher.update(delta_at.to_le_bytes());
        }
        hasher.update(EOP_FIRST_MJD.to_le_bytes());
        hasher.update(SEGMENT_WIDTH_S.to_le_bytes());
        hasher.update(SEGMENT_DEGREE.to_le_bytes());
        hasher.update(CHAIN_VERSION.as_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        let [head0, head1, head2, head3, head4, head5, head6, head7, ..] = digest;
        let head = [head0, head1, head2, head3, head4, head5, head6, head7];

        Self {
            intervals,
            authority_sha256: digest,
            authority_id: u64::from_le_bytes(head),
        }
    }

    /// Diagnostic/cache key: low 64 bits of [`Self::authority_sha256`].
    #[must_use]
    pub const fn authority_id(&self) -> u64 {
        self.authority_id
    }

    /// Full provenance digest over the sealed table, leap table, epoch,
    /// segment width, interpolation degree and chain version.
    #[must_use]
    pub const fn authority_sha256(&self) -> [u8; 32] {
        self.authority_sha256
    }

    /// Two-level `(j, k)` segment index for a continuous-TAI instant.
    ///
    /// `j` is the leap interval containing `tai_s`; `k` counts whole `W` from
    /// that interval's start. The interval's trailing one-second remainder is
    /// absorbed by its final segment rather than emitted as a one-second stub.
    ///
    /// # Errors
    ///
    /// Returns an error when `tai_s` is non-finite or outside the sealed span.
    pub fn segment_index(&self, tai_s: f64) -> Result<(usize, usize), FrameAuthorityError> {
        if !tai_s.is_finite() {
            return Err(FrameAuthorityError::TimeScale("non-finite TAI seconds"));
        }
        let (after_last_mjd, sealed_end_tai_s) = sealed_span_end()?;
        if tai_s >= sealed_end_tai_s {
            return Err(FrameAuthorityError::EpochOutsideSealedSpan {
                mjd: after_last_mjd,
            });
        }
        let Some(j) = self
            .intervals
            .iter()
            .rposition(|(start, _)| *start <= tai_s)
        else {
            return Err(FrameAuthorityError::EpochOutsideSealedSpan {
                mjd: EOP_FIRST_MJD
                    .checked_sub(1)
                    .ok_or(FrameAuthorityError::SealedTableCorrupt("sealed start MJD"))?,
            });
        };
        let (start, _) = self
            .intervals
            .get(j)
            .copied()
            .ok_or(FrameAuthorityError::SealedTableCorrupt("interval index"))?;
        let mut k = ((tai_s - start) / SEGMENT_WIDTH_S)
            .floor()
            .to_i64()
            .ok_or(FrameAuthorityError::TimeScale("segment index overflow"))?;
        if k < 0 {
            k = 0;
        }
        // Absorb the interval remainder into the final segment.
        if let Some((next_start, _)) = j
            .checked_add(1)
            .and_then(|next| self.intervals.get(next))
            .copied()
        {
            let whole = ((next_start - start) / SEGMENT_WIDTH_S)
                .floor()
                .to_i64()
                .ok_or(FrameAuthorityError::TimeScale("segment width overflow"))?;
            if whole > 0 && k >= whole {
                k = whole
                    .checked_sub(1)
                    .ok_or(FrameAuthorityError::SealedTableCorrupt("segment width"))?;
            }
        }
        let k = usize::try_from(k)
            .map_err(|_| FrameAuthorityError::TimeScale("negative segment index"))?;
        Ok((j, k))
    }

    /// `TAI - UTC` for the interval containing `tai_s`.
    ///
    /// # Errors
    ///
    /// Returns an error when `tai_s` cannot resolve to a sealed interval.
    pub fn delta_at_s(&self, tai_s: f64) -> Result<f64, FrameAuthorityError> {
        let (j, _) = self.segment_index(tai_s)?;
        self.intervals
            .get(j)
            .map(|(_, delta_at_s)| *delta_at_s)
            .ok_or(FrameAuthorityError::SealedTableCorrupt("interval index"))
    }

    /// Build the segment identified by `(j, k)`. Depends on the index alone.
    ///
    /// # Errors
    ///
    /// Returns an error when the index does not resolve to sealed frame data.
    pub fn build_segment(
        &self,
        interval_index: usize,
        segment_index: usize,
    ) -> Result<FrameSegment, FrameAuthorityError> {
        let (start, delta_at_s) = *self
            .intervals
            .get(interval_index)
            .ok_or(FrameAuthorityError::SealedTableCorrupt("interval index"))?;
        let segment_index = segment_index
            .to_f64()
            .ok_or(FrameAuthorityError::TimeScale("segment index precision"))?;
        let centre = segment_centre(start, segment_index);
        let half = SEGMENT_WIDTH_S / 2.0;

        // Frozen factors and their centred slopes come from the same builders at
        // three node times, so the segment depends on (j, k) alone.
        // Unwrap the flanking angles onto the centre's branch before they are
        // ever differenced: `era` normalises to [0, 2*pi) and one segment per
        // sidereal day straddles that wrap. See `unwrap_near`.
        let theta_centre = theta_at(centre, delta_at_s)?;
        let theta_minus = unwrap_near(theta_at(centre - half, delta_at_s)?, theta_centre);
        let theta_plus = unwrap_near(theta_at(centre + half, delta_at_s)?, theta_centre);

        let m_centre = split_at(centre, delta_at_s)?;
        let m_minus = split_at(centre - half, delta_at_s)?;
        let m_plus = split_at(centre + half, delta_at_s)?;

        let matrices = m_centre;
        let [m_plus_0, m_plus_1, m_plus_2] = &m_plus;
        let [m_minus_0, m_minus_1, m_minus_2] = &m_minus;
        let k_slope = [
            centred_matrix_slope(m_plus_0, m_minus_0, half),
            centred_matrix_slope(m_plus_1, m_minus_1, half),
            centred_matrix_slope(m_plus_2, m_minus_2, half),
        ];

        // Theta cubic about the centre from the three sampled angles plus the
        // closed-form linear rate; the quadratic and cubic terms carry the
        // UT1-TAI curvature, which is what the daily second difference needs.
        let theta_curve = theta_coefficients(theta_centre, theta_minus, theta_plus, half);

        Ok(FrameSegment {
            m: matrices,
            k: k_slope,
            b: theta_curve,
            delta_at_s,
            centre_tai_s: centre,
        })
    }

    /// `build_segment`, memoized per thread.
    ///
    /// The rebuild is the expensive half of the frame path: three `split_at`
    /// calls, each a full IAU 2006/2000A chain including `xys06a`, measured at
    /// roughly 30x one force evaluation. `SEGMENT_WIDTH_S` is 1800 s, so a
    /// 14-hour arc crosses ~28 segments and a 197 s arc crosses one -- which is
    /// exactly why this cost is 19-34% of self time on the long campaign arcs
    /// and 0.00% on the short one.
    ///
    /// A cache already existed for this, in `RHSCache`, but it holds exactly ONE
    /// segment and lives no longer than its RHS instance. (It is not scoped to a
    /// single propagation — `cached_segment` is deliberately exempt from
    /// `reset_cache`, so on a reused integrator it does survive across
    /// propagations. It dies with the RHS, and the leg path builds a fresh RHS
    /// per leg.) Either way one entry cannot hold an arc that crosses ~28
    /// segments, and the mass solver walks the same arc thousands of times, so
    /// that scope could never hit where the reuse actually is. This is the same
    /// cache at the scope the invariance is at, not a new one.
    ///
    /// Sound because `build_segment` is a pure function of `(j, k)` and the
    /// sealed tables: `intervals` is immutable, `EOP_TABLE` is `include_bytes!`,
    /// and the module holds no thread-local, atomic, interior-mutable or
    /// environment-derived state. Keyed on `authority_id` as well as `(j, k)`
    /// so a separately-constructed authority can never read these entries.
    ///
    /// Thread-local rather than shared: the value is pure, so per-thread copies
    /// are identical and duplication costs only memory, while a shared map
    /// would put a lock on the hot path. Direct-mapped; see
    /// [`SEGMENT_MEMO_SLOTS`] for the measurement that sizes it.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested segment cannot resolve from sealed
    /// frame data or the per-thread memo shape is unexpectedly invalid.
    pub fn segment_cached(&self, j: usize, k: usize) -> Result<FrameSegment, FrameAuthorityError> {
        let slot = (j.wrapping_mul(0x9E37) ^ k) & (SEGMENT_MEMO_SLOTS - 1);
        if let Some(hit) = SEGMENT_MEMO.with(|memo| {
            let memo = memo.borrow();
            match memo.get(slot).copied().flatten() {
                Some(entry)
                    if entry.authority_id == self.authority_id && entry.j == j && entry.k == k =>
                {
                    Some(entry.segment)
                }
                _ => None,
            }
        }) {
            return Ok(hit);
        }
        let segment = self.build_segment(j, k)?;
        SEGMENT_MEMO.with(|memo| -> Result<(), FrameAuthorityError> {
            let mut memo = memo.borrow_mut();
            let cell = memo
                .get_mut(slot)
                .ok_or(FrameAuthorityError::SealedTableCorrupt("segment memo slot"))?;
            *cell = Some(SegmentMemoEntry {
                authority_id: self.authority_id,
                j,
                k,
                segment,
            });
            Ok(())
        })?;
        Ok(segment)
    }

    /// Resolve the rotation at a continuous-TAI instant.
    ///
    /// # Errors
    ///
    /// Returns an error when `tai_s` cannot resolve to a sealed segment.
    pub fn rotation_at(&self, tai_s: f64) -> Result<FrameRotation, FrameAuthorityError> {
        let (j, k) = self.segment_index(tai_s)?;
        let segment = self.segment_cached(j, k)?;
        Ok(segment.rotation_at(tai_s))
    }
}

/// Direct-mapped slots in the per-thread segment memo. Power of two so the
/// index is a mask.
///
/// CHOSEN BY SWEEP, NOT BY ARGUMENT. The previous value, 64, was argued from
/// "the longest campaign arc crosses ~28 segments, so a 28-segment working set
/// does not collide." The measurement below contradicts that conclusion. It does
/// NOT identify what is wrong with it: neither what a worker actually traverses
/// nor the failure mode was investigated, and with a direct-mapped cache the
/// cause could equally be the `(j * 0x9E37) ^ k` key-to-slot mapping colliding.
///
/// Counters in `segment_cached` over one campaign-faithful hybrid batch, since
/// removed:
///
/// ```text
/// slots     calls    rebuilds   distinct (thread, j, k)
///    64   1600000      72,650                      1,073
///   256   1600000       1,074                      1,074
///  1024   1600000       1,078                      1,078
/// ```
///
/// The third column counts `(thread, j, k)` triples — the set behind it was a
/// `thread_local!`, so a segment touched by N workers contributes N. It bounds
/// the number of distinct SEGMENTS from above and is not a count of them, and
/// dividing it by a worker count that was never measured would not make it one.
///
/// **What the rebuild column establishes, and nothing more:** at 64 slots each
/// `(thread, j, k)` was built 67.7 times on average — 72,650 builds against
/// 1,073 first-touches — and at 256 `rebuilds == distinct`, so no pair was ever
/// built twice. 1024 changes neither number materially.
///
/// That is sufficient to pick 256 and is NOT a working-set size. It says this
/// workload rebuilds at 64 and does not at 256; it does not say how many entries
/// are live, because a direct-mapped cache can thrash on the key-to-slot mapping
/// with far fewer live entries than it has slots. 67.7 is also a mean, not a
/// per-segment fact — the distribution was not recorded.
const SEGMENT_MEMO_SLOTS: usize = 256;

/// Per-thread footprint of [`SEGMENT_MEMO`], asserted rather than described.
///
/// The prose that used to sit on `SEGMENT_MEMO_SLOTS` claimed "~32 KB per
/// thread" and nothing checked it. This does.
const SEGMENT_MEMO_BYTES: usize =
    SEGMENT_MEMO_SLOTS * std::mem::size_of::<Option<SegmentMemoEntry>>();
const _: () = assert!(SEGMENT_MEMO_BYTES == 131_072);

#[derive(Clone, Copy)]
struct SegmentMemoEntry {
    authority_id: u64,
    j: usize,
    k: usize,
    segment: FrameSegment,
}

thread_local! {
    /// Boxed so [`SEGMENT_MEMO_BYTES`] never sits on the stack at
    /// initialisation.
    static SEGMENT_MEMO: RefCell<Box<[Option<SegmentMemoEntry>]>> =
        RefCell::new(vec![None; SEGMENT_MEMO_SLOTS].into_boxed_slice());
}

impl FrameSegment {
    /// Per-stage evaluation: three FMA for theta, one `sin_cos`, 45 FMA for `R`.
    /// No `mod2pi`, because theta is segment-anchored and stays O(1) rad.
    #[inline]
    #[must_use]
    pub fn rotation_at(&self, tai_s: f64) -> FrameRotation {
        let dt = tai_s - self.centre_tai_s;
        let theta = theta_from_cubic(&self.b, dt);
        // M1's `[-π, π)` A/B arm. `theta` is segment-anchored, so it is O(1) rad
        // and needs no `mod2pi` — but "O(1)" is not "small", and R55's census
        // put 79% of the `sin`/`cos` calls issued here above glibc's 2.426265
        // reduction threshold. One compare and one subtract move them below it.
        // See `crate::WRAP_TO_SIGNED_PI`; never committed `true`.
        let theta = if crate::WRAP_TO_SIGNED_PI {
            signed_pi_theta(theta)
        } else {
            theta
        };
        let (sn, cs) = theta.sin_cos();
        let &[[[k000, k001, k002], [k010, k011, k012], [k020, k021, k022]], [[k100, k101, k102], [k110, k111, k112], [k120, k121, k122]], [[k200, k201, k202], [k210, k211, k212], [k220, k221, k222]]] =
            &self.k;
        let &[[[m000, m001, m002], [m010, m011, m012], [m020, m021, m022]], [[m100, m101, m102], [m110, m111, m112], [m120, m121, m122]], [[m200, m201, m202], [m210, m211, m212], [m220, m221, m222]]] =
            &self.m;
        let r = [
            [
                cs.mul_add(
                    dt.mul_add(k000, m000),
                    sn.mul_add(dt.mul_add(k100, m100), dt.mul_add(k200, m200)),
                ),
                cs.mul_add(
                    dt.mul_add(k001, m001),
                    sn.mul_add(dt.mul_add(k101, m101), dt.mul_add(k201, m201)),
                ),
                cs.mul_add(
                    dt.mul_add(k002, m002),
                    sn.mul_add(dt.mul_add(k102, m102), dt.mul_add(k202, m202)),
                ),
            ],
            [
                cs.mul_add(
                    dt.mul_add(k010, m010),
                    sn.mul_add(dt.mul_add(k110, m110), dt.mul_add(k210, m210)),
                ),
                cs.mul_add(
                    dt.mul_add(k011, m011),
                    sn.mul_add(dt.mul_add(k111, m111), dt.mul_add(k211, m211)),
                ),
                cs.mul_add(
                    dt.mul_add(k012, m012),
                    sn.mul_add(dt.mul_add(k112, m112), dt.mul_add(k212, m212)),
                ),
            ],
            [
                cs.mul_add(
                    dt.mul_add(k020, m020),
                    sn.mul_add(dt.mul_add(k120, m120), dt.mul_add(k220, m220)),
                ),
                cs.mul_add(
                    dt.mul_add(k021, m021),
                    sn.mul_add(dt.mul_add(k121, m121), dt.mul_add(k221, m221)),
                ),
                cs.mul_add(
                    dt.mul_add(k022, m022),
                    sn.mul_add(dt.mul_add(k122, m122), dt.mul_add(k222, m222)),
                ),
            ],
        ];
        // Exact analytic derivative of this segment's centred interpolant. A
        // segment-end secant attenuates even constant Earth rotation by
        // sin(omega*h)/(omega*h); atmosphere-relative velocity needs the rate
        // at this stage instead.
        let theta_dot = self.b[1] + dt * (2.0 * self.b[2] + dt * 3.0 * self.b[3]);
        let derivative = |k0: f64, m0: f64, k1: f64, m1: f64, k2: f64| {
            let a = dt.mul_add(k0, m0);
            let b = dt.mul_add(k1, m1);
            let frozen_slope = cs.mul_add(k0, sn.mul_add(k1, k2));
            theta_dot.mul_add((-sn).mul_add(a, cs * b), frozen_slope)
        };
        let rdot = [
            [
                derivative(k000, m000, k100, m100, k200),
                derivative(k001, m001, k101, m101, k201),
                derivative(k002, m002, k102, m102, k202),
            ],
            [
                derivative(k010, m010, k110, m110, k210),
                derivative(k011, m011, k111, m111, k211),
                derivative(k012, m012, k112, m112, k212),
            ],
            [
                derivative(k020, m020, k120, m120, k220),
                derivative(k021, m021, k121, m121, k221),
                derivative(k022, m022, k122, m122, k222),
            ],
        ];

        // Passive GCRS->ITRS convention: Rdot R^T = -[omega_itrs x].
        // Only its six off-diagonal entries are needed; avoid forming the
        // full matrix on every RHS evaluation.
        let dot = |a: &[f64; 3], b: &[f64; 3]| {
            let &[a0, a1, a2] = a;
            let &[b0, b1, b2] = b;
            a0 * b0 + a1 * b1 + a2 * b2
        };
        let [rdot0, rdot1, rdot2] = rdot;
        let [r0, r1, r2] = r;
        let omega_itrs = [
            0.5 * (dot(&rdot1, &r2) - dot(&rdot2, &r1)),
            0.5 * (dot(&rdot2, &r0) - dot(&rdot0, &r2)),
            0.5 * (dot(&rdot0, &r1) - dot(&rdot1, &r0)),
        ];
        let itrs_angular_velocity_gcrs = transpose_apply(&r, &omega_itrs);
        FrameRotation {
            r,
            delta_at_s: self.delta_at_s,
            itrs_angular_velocity_gcrs,
        }
    }
}

fn theta_from_cubic(coefficients: &[f64; 4], dt: f64) -> f64 {
    let &[b0, b1, b2, b3] = coefficients;
    b0 + dt * (b1 + dt * (b2 + dt * b3))
}

/// One `τ` shift into `[-π, π)`, for the M1 A/B arm only.
///
/// A single conditional subtract covers `[-π, 3π)`, which is the whole range a
/// segment-anchored `theta` built from a `[0, 2π)` anchor can occupy; anything
/// outside falls through unchanged rather than looping, because the arm exists
/// to price a libm branch and must not add one of its own.
#[inline]
fn signed_pi_theta(theta: f64) -> f64 {
    if theta >= std::f64::consts::PI {
        theta - std::f64::consts::TAU
    } else {
        theta
    }
}

/// Daily-linear Bulletin-A quantities at one continuous-TAI instant, in
/// radians (and seconds for `dut1`). The in-segment residual of the daily
/// linear form is micrometre-level over one hour.
struct EopAt {
    xp: f64,
    yp: f64,
    dx: f64,
    dy: f64,
    /// `UT1 - TAI` in seconds, interpolated with each node's own `TAI - UTC`
    /// already removed. Never `dut1` alone: see the note in `eop_at`.
    ut1_tai_s: f64,
    /// TAI as an exact big part plus a day fraction in `[0, 1)`, mirroring the
    /// exact chain's `(anchor1, anchor2)` split (`chain.rs:111`, where `utctai`
    /// returns the full JD and a sub-day remainder).
    ///
    /// Collapsing these into one TAI MJD costs real accuracy and it is measured,
    /// not assumed: a binary64 at MJD 59803 has ULP `2^-37 d = 628.6427 ns`,
    /// which the ERA rate turns into a 0.320889 mm staircase at 7000 km. That
    /// staircase, not the segment model, was the floor the in-segment residual
    /// plateaued on when the half-width was shrunk 32x.
    tai_jd1: f64,
    tai_jd2: f64,
}

/// `(EOP_LAST_MJD + 1, its node TAI seconds)` — the sealed span's exclusive end.
///
/// Both components are functions of sealed compile-time constants alone, so the
/// `jd2cal` plus `dat` walk inside `node_tai_seconds` returns the same pair on
/// every call. `segment_index` is on the per-RHS-evaluation path, so the walk is
/// hoisted here. Validation failures are not cached: a later call retries the
/// sealed computation, while a success is immutable for the process lifetime.
static SEALED_SPAN_END: OnceLock<(i32, f64)> = OnceLock::new();
static SEALED_SPAN_END_LOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    cache: &'cache OnceLock<T>,
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

fn sealed_span_end() -> Result<(i32, f64), FrameAuthorityError> {
    success_only_cached(&SEALED_SPAN_END, &SEALED_SPAN_END_LOAD, || {
        let after_last_mjd = EOP_LAST_MJD
            .checked_add(1)
            .ok_or(FrameAuthorityError::SealedTableCorrupt("sealed end MJD"))?;
        let (sealed_end_tai_s, _) = node_tai_seconds(after_last_mjd)?;
        Ok((after_last_mjd, sealed_end_tai_s))
    })
    .copied()
}

/// TAI seconds, from the sealed span start, of a node's UTC midnight. Each node
/// carries its own `TAI - UTC`, exactly as the exact chain's per-node `utctai`
/// does, so a node set spanning a leap stays correctly spaced.
fn node_tai_seconds(mjd: i32) -> Result<(f64, f64), FrameAuthorityError> {
    let (status, y, m, d, _f) = jd2cal(DJM0, f64::from(mjd));
    if status != 0 {
        return Err(FrameAuthorityError::TimeScale("node MJD out of range"));
    }
    let (s, delta_at) = dat(y, m, d, 0.0);
    if s < 0 {
        return Err(FrameAuthorityError::TimeScale("node TAI-UTC unavailable"));
    }
    let days_since_start = mjd
        .checked_sub(EOP_FIRST_MJD)
        .ok_or(FrameAuthorityError::TimeScale(
            "node MJD before sealed epoch",
        ))?;
    Ok((f64::from(days_since_start) * DAYSEC + delta_at, delta_at))
}

/// Four-node Lagrange over the bracketing daily records, in continuous-TAI
/// seconds — the same interpolation the exact chain applies.
///
/// A two-node linear form is NOT sufficient here and this is measured, not
/// assumed: over one day the linear-versus-Lagrange difference in `dut1`
/// reaches 2.487e-5 s, which is 1.813e-9 rad of Earth rotation, 12.69 mm at
/// 7000 km. That alone breaks the 1 mm fast-path bound, and it was the entire
/// discrepancy when this function interpolated linearly.
fn eop_at(tai_s: f64, delta_at_s: f64) -> Result<EopAt, FrameAuthorityError> {
    let utc_mjd_real = (tai_s - delta_at_s) / DAYSEC + f64::from(EOP_FIRST_MJD);
    let centre = utc_mjd_real
        .floor()
        .to_i32()
        .ok_or(FrameAuthorityError::TimeScale("UTC MJD out of range"))?;
    let (tai_jd1, tai_jd2) = tai_jd_from_seconds(tai_s);

    // `ut1_tai` is interpolated as ONE quantity with each node's own
    // `TAI - UTC` already removed, exactly as the exact chain does at
    // `chain.rs:137` (`raw_values[q] = row.dut1 - dat_at_mjd(mjd)`).
    //
    // Interpolating `dut1` alone and subtracting the segment's constant
    // `delta_at_s` afterwards is NOT equivalent, and the difference is measured,
    // not assumed: it agrees only while all four nodes share one `dat`, because
    // a Lagrange basis reproduces a constant exactly. Across a leap the node
    // dats are `[36, 37, 37, 37]` and the two forms diverge by up to 6.204e-2 s
    // of UT1 at 2017-01-01T12:15, which is 31667 mm at 7000 km. It vanishes
    // exactly at 2017-01-01T00:00 because the evaluation point sits ON a node,
    // where that node's basis weight is 1 and the rest are 0 -- which is why the
    // declared "post-leap" gate epoch never caught it.
    let node_mjd_0 = centre
        .checked_sub(1)
        .ok_or(FrameAuthorityError::TimeScale("UTC MJD underflow"))?;
    let node_mjd_1 = node_mjd_0
        .checked_add(1)
        .ok_or(FrameAuthorityError::TimeScale("UTC MJD overflow"))?;
    let node_mjd_2 = node_mjd_1
        .checked_add(1)
        .ok_or(FrameAuthorityError::TimeScale("UTC MJD overflow"))?;
    let node_mjd_3 = node_mjd_2
        .checked_add(1)
        .ok_or(FrameAuthorityError::TimeScale("UTC MJD overflow"))?;
    let record_0 = eop_record(node_mjd_0)?;
    let (node_tai_s_0, node_delta_at_0) = node_tai_seconds(node_mjd_0)?;
    let record_1 = eop_record(node_mjd_1)?;
    let (node_tai_s_1, node_delta_at_1) = node_tai_seconds(node_mjd_1)?;
    let record_2 = eop_record(node_mjd_2)?;
    let (node_tai_s_2, node_delta_at_2) = node_tai_seconds(node_mjd_2)?;
    let record_3 = eop_record(node_mjd_3)?;
    let (node_tai_s_3, node_delta_at_3) = node_tai_seconds(node_mjd_3)?;
    let abscissa = [
        node_tai_s_0 - tai_s,
        node_tai_s_1 - tai_s,
        node_tai_s_2 - tai_s,
        node_tai_s_3 - tai_s,
    ];
    let ut1_tai = [
        record_0.dut1_s - node_delta_at_0,
        record_1.dut1_s - node_delta_at_1,
        record_2.dut1_s - node_delta_at_2,
        record_3.dut1_s - node_delta_at_3,
    ];

    Ok(EopAt {
        xp: lagrange_four(
            &[
                record_0.xp_arcsec,
                record_1.xp_arcsec,
                record_2.xp_arcsec,
                record_3.xp_arcsec,
            ],
            &abscissa,
        ) * DAS2R,
        yp: lagrange_four(
            &[
                record_0.yp_arcsec,
                record_1.yp_arcsec,
                record_2.yp_arcsec,
                record_3.yp_arcsec,
            ],
            &abscissa,
        ) * DAS2R,
        dx: lagrange_four(
            &[
                record_0.dx_mas,
                record_1.dx_mas,
                record_2.dx_mas,
                record_3.dx_mas,
            ],
            &abscissa,
        ) * 1e-3
            * DAS2R,
        dy: lagrange_four(
            &[
                record_0.dy_mas,
                record_1.dy_mas,
                record_2.dy_mas,
                record_3.dy_mas,
            ],
            &abscissa,
        ) * 1e-3
            * DAS2R,
        ut1_tai_s: lagrange_four(&ut1_tai, &abscissa),
        tai_jd1,
        tai_jd2,
    })
}

fn lagrange_four(values: &[f64; 4], abscissa: &[f64; 4]) -> f64 {
    let &[value_0, value_1, value_2, value_3] = values;
    let &[abscissa_0, abscissa_1, abscissa_2, abscissa_3] = abscissa;
    let mut term_0 = value_0;
    term_0 *= (0.0 - abscissa_1) / (abscissa_0 - abscissa_1);
    term_0 *= (0.0 - abscissa_2) / (abscissa_0 - abscissa_2);
    term_0 *= (0.0 - abscissa_3) / (abscissa_0 - abscissa_3);
    let mut term_1 = value_1;
    term_1 *= (0.0 - abscissa_0) / (abscissa_1 - abscissa_0);
    term_1 *= (0.0 - abscissa_2) / (abscissa_1 - abscissa_2);
    term_1 *= (0.0 - abscissa_3) / (abscissa_1 - abscissa_3);
    let mut term_2 = value_2;
    term_2 *= (0.0 - abscissa_0) / (abscissa_2 - abscissa_0);
    term_2 *= (0.0 - abscissa_1) / (abscissa_2 - abscissa_1);
    term_2 *= (0.0 - abscissa_3) / (abscissa_2 - abscissa_3);
    let mut term_3 = value_3;
    term_3 *= (0.0 - abscissa_0) / (abscissa_3 - abscissa_0);
    term_3 *= (0.0 - abscissa_1) / (abscissa_3 - abscissa_1);
    term_3 *= (0.0 - abscissa_2) / (abscissa_3 - abscissa_2);
    let mut sum = 0.0;
    sum += term_0;
    sum += term_1;
    sum += term_2;
    sum += term_3;
    sum
}

/// Two-part TAI Julian Day for a continuous-TAI instant.
///
/// Splits continuous-TAI seconds into an exact JD big part and a day fraction in
/// `[0, 1)`, so no intermediate ever holds a TAI MJD in a single binary64. This
/// is the same split the exact chain uses, so callers can hand the pair straight
/// to `taitt` or `taiutc` without collapsing to a single binary64 first.
///
/// `day * DAYSEC` is exact (both integers, product below `2^53`) and the
/// subtraction is exact by Sterbenz, so the fraction carries every bit `tai_s`
/// has. `DJM0 + EOP_FIRST_MJD + day` is a multiple of 0.5 below `2^22` and is
/// therefore exact too.
#[must_use]
pub fn tai_jd_from_seconds(tai_s: f64) -> (f64, f64) {
    let mut day = (tai_s / DAYSEC).floor();
    let mut frac_s = tai_s - day * DAYSEC;
    // The division rounds, so pull the fraction back into range if it stepped out.
    if frac_s < 0.0 {
        day -= 1.0;
        frac_s += DAYSEC;
    } else if frac_s >= DAYSEC {
        day += 1.0;
        frac_s -= DAYSEC;
    }
    (DJM0 + f64::from(EOP_FIRST_MJD) + day, frac_s / DAYSEC)
}

/// `RPOM E_i RC2I` at one instant, shared by the centre and the slope nodes.
fn split_at(tai_s: f64, delta_at_s: f64) -> Result<[Mat3; 3], FrameAuthorityError> {
    let e = eop_at(tai_s, delta_at_s)?;
    let (tt1, tt2) = taitt(e.tai_jd1, e.tai_jd2);
    let (x, y, s) = xys06a(tt1, tt2);
    let rc2i = c2ixys(x + e.dx, y + e.dy, s);
    let rpom = pom00(e.xp, e.yp, sp00(tt1, tt2));
    Ok(split_matrices(&rpom, &rc2i))
}

/// Earth Rotation Angle at one instant, from the double-double `era` on
/// `UT1 = TAI + (UT1 - UTC) - (TAI - UTC)`. Called only at segment build.
fn theta_at(tai_s: f64, delta_at_s: f64) -> Result<f64, FrameAuthorityError> {
    let e = eop_at(tai_s, delta_at_s)?;
    Ok(era(
        from(e.tai_jd1),
        from(e.tai_jd2).add_dd(from(e.ut1_tai_s / DAYSEC)),
    )
    .to_f64())
}

/// Put `theta` on the same 2*pi branch as `reference`.
///
/// `era` normalises to `[0, 2*pi)`, so a segment whose flanking nodes straddle
/// the wrap would difference two angles a full turn apart. Measured, not
/// assumed: at 2017-01-01T17:30 the raw triple is
/// `(6.218973183, 0.067045754, 0.198303623)`, giving `b1 = -1.672408e-3` against
/// a true ERA rate of `7.292105e-5` — a 1.85 rad error, 1.29e10 mm at 7000 km.
/// One segment per sidereal day straddles the wrap, so this is not an edge case.
///
/// The f64 `TAU` differs from 2*pi by about 2.4e-16 rad, which reaches `b1` as
/// 6.8e-20 rad/s and the segment edge as 1.2e-16 rad — 8.6e-7 mm, negligible.
fn unwrap_near(theta: f64, reference: f64) -> f64 {
    theta + std::f64::consts::TAU * ((reference - theta) / std::f64::consts::TAU).round()
}

/// Process-wide sealed default. Unlike `GlobalCoeffs`, whose default is empty
/// and requires an explicit caller, this self-constructs from the compiled
/// sealed bytes on first load, so no caller wiring exists and none is required.
/// `Lazy` guarantees exactly-once construction, and the construction is a pure
/// function of those bytes.
///
/// It is deliberately NOT swappable. A replaceable global frame authority would
/// defeat the argument that results cannot depend on worker count: the W1/W8
/// guarantee rests on the authority being unreplaceable mid-run, so a swap cell
/// would weaken the exact property this module exists to provide.
static FRAME_AUTHORITY: std::sync::LazyLock<FrameAuthority> =
    std::sync::LazyLock::new(FrameAuthority::from_sealed);

/// The sealed frame authority.
///
/// Borrowed, never reference-counted. The authority is a `Lazy` static that is
/// constructed once and — per the note above — deliberately never replaced, so
/// it outlives every caller and a `&'static` is exactly as valid as an owned
/// handle.
///
/// This used to be `Lazy<Arc<FrameAuthority>>` behind an `Arc::clone`, and
/// nothing anywhere ever stored the resulting `Arc`: every caller immediately
/// called a `&self` method on it and dropped it. The clone/drop pair was
/// therefore two atomic read-modify-writes on ONE process-wide cache line, paid
/// on every call. `LightyearRHS::frame_rotation_at` is the only hot-path caller
/// and it runs twice per RHS evaluation, so a 64-thread campaign had all 64
/// workers contending on that single line four times per derivative — a cost
/// that grows with worker count, which is the worst possible shape for a
/// throughput knob we are trying to scale.
///
/// Returning a borrow removes the refcount traffic entirely. No data, no
/// arithmetic, and no ordering changes, so this is bit-exact by construction.
#[must_use]
pub fn frame_authority() -> &'static FrameAuthority {
    &FRAME_AUTHORITY
}

/// Continuous-TAI seconds from the sealed span start for a two-part UTC JD.
///
/// The date MUST arrive in two parts, as every other entry point in this module
/// takes it (`jd2cal`, `utctai`, `taitt`) and as the exact chain supplies it
/// (`chain.rs:109`, `dtf2d_utc` returning a full JD plus a sub-day fraction).
/// A single-binary64 JD near 2.46e6 has ULP `2^-31 d = 40.233 us`, which the
/// ERA rate turns into a 20.537 mm sawtooth at 7000 km — 20x the fast-path
/// bound, and measured at 10.117769 mm worst before this signature was split.
///
/// # Errors
///
/// Returns an error when the UTC epoch cannot convert under sealed time data.
pub fn tai_seconds_from_utc_jd(utc_jd1: f64, utc_jd2: f64) -> Result<f64, FrameAuthorityError> {
    // Converted through the sealed `utctai`, NOT by scaling the UTC day
    // fraction by 86400 and adding DAT.
    //
    // UTC quasi-JD encodes a leap day's sub-day position as `seconds / 86401`,
    // because that day genuinely has 86401 seconds. Decoding it as
    // `fraction * 86400 + DAT` reads that fraction against the wrong day
    // length, so the recovered instant runs early inside the leap day: about
    // half a second at noon and approaching a full second near 23:59:60, which
    // the ERA rate turns into hundreds of metres of Earth-rotation displacement
    // at orbital radius.
    //
    // `utctai` is the transform that knows the day length, so the fraction is
    // never scaled here at all.
    //
    // The existing coverage compared midnight to midnight, where the DAT step
    // and the compression cancel at the endpoints and hide the interior error.
    let (status, tai_jd1, tai_jd2) = utctai(utc_jd1, utc_jd2);
    if status < 0 {
        return Err(FrameAuthorityError::TimeScale("UTC JD out of range"));
    }
    // Two-part throughout, for the reason the previous implementation gave and
    // which still applies: collapsing to one MJD near 59803 reintroduces a
    // 628.6 ns quantum, which the ERA rate turns into a 20.537 mm sawtooth at
    // 7000 km. `utctai` returns the large part first, so subtracting the sealed
    // span origin from THAT part keeps the small part untouched.
    let day_part = tai_jd1 - DJM0 - f64::from(EOP_FIRST_MJD);
    Ok(day_part * DAYSEC + tai_jd2 * DAYSEC)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A leap day's sub-day fraction must be read against an 86401-second day.
    ///
    /// UTC quasi-JD encodes the position within a positive leap day as
    /// `seconds / 86401`. Decoding it as `fraction * 86400 + DAT` reads that
    /// fraction against the wrong day length, so the recovered instant runs
    /// early INSIDE the day: about half a second at noon, approaching a full
    /// second near 23:59:60.
    ///
    /// The oracle is elapsed TAI, which is unaffected by how UTC labels the
    /// day: from 00:00 to 12:00 on any day, leap or not, exactly 43200 seconds
    /// of TAI pass. The old decode returned 43199.5 for a leap day.
    ///
    /// Midnight-to-midnight was already covered and cannot see this: the DAT
    /// step at the boundary and the interior compression cancel at the
    /// endpoints, which is why the defect survived.
    #[test]
    fn leap_day_noon_is_half_a_day_of_tai_after_its_midnight() {
        // 2016-12-31 carried a positive leap second at 23:59:60 UTC and sits
        // inside the sealed EOP span.
        let (status_midnight, jd1_midnight, jd2_midnight) =
            super::super::timescale::dtf2d_utc(2016, 12, 31, 0, 0, 0.0);
        let (status_noon, jd1_noon, jd2_noon) =
            super::super::timescale::dtf2d_utc(2016, 12, 31, 12, 0, 0.0);
        assert_eq!(status_midnight, 0, "leap-day midnight must encode");
        assert_eq!(status_noon, 0, "leap-day noon must encode");

        let midnight = tai_seconds_from_utc_jd(jd1_midnight, jd2_midnight)
            .expect("leap-day midnight must convert");
        let noon = tai_seconds_from_utc_jd(jd1_noon, jd2_noon).expect("leap-day noon must convert");
        let elapsed = noon - midnight;
        assert!(
            (elapsed - 43_200.0).abs() < 1.0e-6,
            "leap-day noon is {elapsed} TAI seconds after its midnight, not 43200; \
             the sub-day fraction is being read against an 86400-second day"
        );

        // The whole leap day is 86401 seconds of TAI, which is the property the
        // midnight-to-midnight coverage already had. Asserted here too so this
        // test cannot pass by making every interval wrong in the same way.
        let (status_next, jd1_next, jd2_next) =
            super::super::timescale::dtf2d_utc(2017, 1, 1, 0, 0, 0.0);
        assert_eq!(status_next, 0, "the day after a leap day must encode");
        let next = tai_seconds_from_utc_jd(jd1_next, jd2_next)
            .expect("the day after a leap day must convert");
        let day = next - midnight;
        assert!(
            (day - 86_401.0).abs() < 1.0e-6,
            "the leap day spans {day} TAI seconds, not 86401"
        );
    }

    #[test]
    fn centred_interpolant_positive_z_rate_has_positive_gcrs_omega() {
        let segment = FrameSegment {
            m: [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            k: [[[0.0; 3]; 3]; 3],
            b: [0.0, 7.292_115e-5, 0.0, 0.0],
            delta_at_s: 37.0,
            centre_tai_s: 1000.0,
        };
        let rotation = segment.rotation_at(1000.0);
        assert_eq!(
            rotation.itrs_angular_velocity_gcrs.map(f64::to_bits),
            [
                0.0_f64.to_bits(),
                0.0_f64.to_bits(),
                7.292_115e-5_f64.to_bits()
            ]
        );
    }

    #[test]
    fn sealed_cache_retries_failures_and_reuses_success() {
        let cache = OnceLock::new();
        let lock = std::sync::Mutex::new(());
        let attempts = AtomicUsize::new(0);

        let first = success_only_cached(&cache, &lock, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Err::<u8, _>(FrameAuthorityError::TimeScale("hostile first attempt"))
        });
        assert!(first.is_err());
        assert!(cache.get().is_none());

        let second = success_only_cached(&cache, &lock, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok::<u8, FrameAuthorityError>(7)
        })
        .expect("second attempt succeeds");
        let third = success_only_cached(&cache, &lock, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok::<u8, FrameAuthorityError>(9)
        })
        .expect("cached success");

        assert_eq!(attempts.load(Ordering::Relaxed), 2);
        assert!(std::ptr::eq(second, third));
        assert_eq!(*third, 7);
    }

    #[test]
    fn sealed_table_has_the_declared_shape() {
        assert_eq!(EOP_TABLE.len(), 785_800);
        assert_eq!(EOP_TABLE.len(), EOP_RECORDS * EOP_RECORD_BYTES);
        assert_eq!(
            usize::try_from(EOP_LAST_MJD - EOP_FIRST_MJD + 1),
            Ok(EOP_RECORDS)
        );
    }

    #[test]
    fn sealed_records_match_the_known_bulletin_a_row() {
        // Class (a): recomputed from the sealed table and verified bit-exact
        // against the scanning `load_eop` parser across all 19,645 records.
        let row = eop_record(60310).expect("row inside the sealed span");
        assert_eq!(row.xp_arcsec.to_bits(), 0.136_912_f64.to_bits());
        assert_eq!(row.yp_arcsec.to_bits(), 0.202_19_f64.to_bits());
        assert_eq!(row.dut1_s.to_bits(), 0.008_783_7_f64.to_bits());
        assert_eq!(row.dx_mas.to_bits(), 0.295_f64.to_bits());
        assert_eq!(row.dy_mas.to_bits(), (-0.095_f64).to_bits());
    }

    /// Fail-closed past the sealed span rather than extrapolating or reusing
    /// the last complete row.
    #[test]
    fn epochs_beyond_the_sealed_span_are_rejected() {
        assert_eq!(
            eop_record(EOP_LAST_MJD + 1),
            Err(FrameAuthorityError::EpochOutsideSealedSpan {
                mjd: EOP_LAST_MJD + 1
            })
        );
        assert_eq!(
            eop_record(EOP_FIRST_MJD - 1),
            Err(FrameAuthorityError::EpochOutsideSealedSpan {
                mjd: EOP_FIRST_MJD - 1
            })
        );
        assert!(eop_record(EOP_FIRST_MJD).is_ok());
        assert!(eop_record(EOP_LAST_MJD).is_ok());
    }

    /// The interval remainder is exactly one second at W = 3600, because leaps
    /// fall at UTC midnight and 86400 is divisible by 3600.
    #[test]
    fn leap_interval_remainder_is_exactly_one_second() {
        let authority = frame_authority();
        let intervals = &authority.intervals;
        assert!(intervals.len() > 20, "expected the integer-leap era");
        for pair in intervals.windows(2) {
            let [first, second] = pair else {
                continue;
            };
            let span = second.0 - first.0;
            let remainder = span - (span / SEGMENT_WIDTH_S).floor() * SEGMENT_WIDTH_S;
            assert!(
                (remainder - 1.0).abs() < 1e-6 || remainder.abs() < 1e-6,
                "interval span {span} left remainder {remainder}, expected 0 or exactly 1 s"
            );
        }
    }

    /// The index is a pure function of absolute TAI, so it cannot depend on how
    /// work was cut into arcs.
    #[test]
    fn segment_index_is_arc_independent() {
        let authority = frame_authority();
        // Sourced: assets/part_a/search_b500_v2.json, events[0].conjunction_jd.
        let instant = tai_seconds_from_utc_jd(2_459_794.5, -0.315_579_753).expect("Part A epoch");
        let direct = authority.segment_index(instant).expect("index");
        // The same instant reached by adding an offset to an earlier anchor.
        let anchor = instant - 12_345.678;
        let viaanchor = authority
            .segment_index(anchor + 12_345.678)
            .expect("index via anchor");
        assert_eq!(direct, viaanchor);
    }

    #[test]
    fn segment_index_rejects_finite_epoch_past_sealed_span() {
        assert_eq!(
            frame_authority().segment_index(f64::MAX),
            Err(FrameAuthorityError::EpochOutsideSealedSpan {
                mjd: EOP_LAST_MJD + 1,
            })
        );
    }

    #[test]
    fn tai_seconds_rejects_malformed_cancelling_jd_parts() {
        assert_eq!(
            tai_seconds_from_utc_jd(f64::MAX, -f64::MAX),
            Err(FrameAuthorityError::TimeScale("UTC JD out of range"))
        );
    }

    /// THE load-bearing test: the segment cache must reproduce the exact chain
    /// within the declared fast-path bound. Everything else about this module
    /// is bookkeeping; this is the accuracy claim.
    ///
    /// The bound is position error at 7000 km, not a matrix-element bound, and
    /// it is the Task 5B segment budget — NOT the Task 5A cold-chain
    /// `max|dR| <= 5e-13`, which binds the module against its sealed fixture.
    #[test]
    fn segment_cache_matches_the_exact_chain_within_the_declared_bound() {
        use crate::frame_time::chain::{self, EopPolicy, Epoch};

        let finals = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/reference/frame_time/finals2000A.all"),
        )
        .expect("sealed finals2000A.all");

        // Epochs spanning the Part A window and the fixture corpus, deliberately
        // sampled away from segment centres so the in-segment drift correction
        // is exercised rather than bypassed.
        let epochs = [
            Epoch {
                y: 2022,
                m: 8,
                d: 12,
                hh: 4,
                mm: 25,
                ss: 0.0,
                name: "part-a",
            },
            Epoch {
                y: 2022,
                m: 8,
                d: 12,
                hh: 4,
                mm: 55,
                ss: 30.0,
                name: "part-a+30m",
            },
            Epoch {
                y: 2024,
                m: 1,
                d: 1,
                hh: 0,
                mm: 0,
                ss: 0.0,
                name: "2024",
            },
            Epoch {
                y: 2017,
                m: 1,
                d: 1,
                hh: 0,
                mm: 0,
                ss: 0.0,
                name: "post-leap",
            },
        ];

        let authority = frame_authority();
        let mut worst_m = 0.0f64;
        for epoch in &epochs {
            let exact_dd = chain::frame_matrix(epoch, EopPolicy::Real, 0.0, &finals)
                .expect("sealed frame input resolves");
            let mut exact = [[0.0f64; 3]; 3];
            for (exact_row, exact_dd_row) in exact.iter_mut().zip(exact_dd.iter()) {
                for (exact_value, exact_dd_value) in exact_row.iter_mut().zip(exact_dd_row.iter()) {
                    *exact_value = exact_dd_value.to_f64();
                }
            }

            let (status, d1, d2) = crate::frame_time::timescale::dtf2d_utc(
                epoch.y, epoch.m, epoch.d, epoch.hh, epoch.mm, epoch.ss,
            );
            assert_eq!(status, 0, "{} must be a valid UTC instant", epoch.name);
            // Two parts, never `d1 + d2`: collapsing them costs 40.233 us and
            // was worth 10.117769 mm at 7000 km.
            let tai_s = tai_seconds_from_utc_jd(d1, d2).expect("inside the sealed span");
            let cached = authority.rotation_at(tai_s).expect("rotation resolves");

            // Position error at 7000 km over a spanning set of directions.
            for point in [
                [7000.0, 0.0, 0.0],
                [0.0, 7000.0, 0.0],
                [0.0, 0.0, 7000.0],
                [4041.45, 4041.45, 4041.45],
            ] {
                let got = cached.to_itrs(&point);
                let [point_x, point_y, point_z] = point;
                let mut want = [0.0f64; 3];
                for (want_value, &[exact_x, exact_y, exact_z]) in want.iter_mut().zip(&exact) {
                    *want_value = exact_x * point_x + exact_y * point_y + exact_z * point_z;
                }
                let [got_x, got_y, got_z] = got;
                let [want_x, want_y, want_z] = want;
                let error_m = ((got_x - want_x).powi(2)
                    + (got_y - want_y).powi(2)
                    + (got_z - want_z).powi(2))
                .sqrt()
                    * 1000.0;
                worst_m = worst_m.max(error_m);
            }
        }
        assert!(
            worst_m <= FAST_PATH_BOUND_M,
            "segment cache must track the exact chain within {FAST_PATH_BOUND_M} m at 7000 km; \
             worst was {worst_m:.6} m"
        );
    }

    /// Cache-key diagnostics must not depend on thread count or first-touch
    /// order. Full provenance uses `authority_sha256`.
    #[test]
    fn authority_id_is_identical_across_thread_counts() {
        let single = frame_authority().authority_id();
        let mut handles = Vec::new();
        for _ in 0..8 {
            handles.push(std::thread::spawn(|| frame_authority().authority_id()));
        }
        for handle in handles {
            assert_eq!(handle.join().expect("worker"), single);
        }
        // A freshly built authority must agree with the installed one.
        assert_eq!(FrameAuthority::from_sealed().authority_id(), single);
    }
}
