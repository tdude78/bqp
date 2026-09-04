//! GCRS->ITRS rotation, its five-point-stencil time derivatives, and state
//! transforms, transliterated from the sealed 4AF generator
//! (`frame_matrix`, `derivatives`, `transform_state`).
//!
//! RC2I and RPOM come from the ordinary binary64 chain (`iau2006`, `cio`); ERA,
//! the outer `RPOM * R3(ERA) * RC2I` composition, EOP interpolation, and the
//! conditioned centered five-point stencil are double-double.

use super::cio::{c2ixys, pom00, sp00};
use super::dd::{from, sincos, Dd};
use super::eop::{dat_at_mjd, lagrange, load_eop, EopError};
use super::era::era;
use super::iau2006::xys06a;
use super::timescale::{cal2jd, dat, dtf2d_utc, taitt, utctai, DAS2R, DAYSEC, DJM0};
use num_traits::ToPrimitive;
use std::fmt;

pub type DdMat = [[Dd; 3]; 3];

/// Failure while constructing a fixture-frame transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameChainError {
    /// EOP input could not be decoded or validated.
    Eop(EopError),
    /// ERFA-compatible time conversion rejected an input.
    TimeScale(&'static str),
}

impl From<EopError> for FrameChainError {
    fn from(error: EopError) -> Self {
        Self::Eop(error)
    }
}

impl fmt::Display for FrameChainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eop(error) => write!(formatter, "EOP frame input failed: {error}"),
            Self::TimeScale(reason) => {
                write!(formatter, "frame time-scale conversion failed: {reason}")
            }
        }
    }
}

impl std::error::Error for FrameChainError {}

/// Fixture step for `Rdot`/`Rddot` (seconds).
pub const H_FIXTURE_S: f64 = 0.25;

#[derive(Clone, Copy)]
pub struct Epoch {
    pub y: i32,
    pub m: i32,
    pub d: i32,
    pub hh: i32,
    pub mm: i32,
    pub ss: f64,
    pub name: &'static str,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EopPolicy {
    Zero,
    Real,
}

/// The five declared corpus epochs, in fixture order.
pub const EPOCHS: [Epoch; 5] = [
    Epoch {
        y: 2000,
        m: 1,
        d: 1,
        hh: 12,
        mm: 0,
        ss: 0.0,
        name: "2000-01-01T12:00:00",
    },
    Epoch {
        y: 2016,
        m: 12,
        d: 31,
        hh: 23,
        mm: 59,
        ss: 59.0,
        name: "2016-12-31T23:59:59",
    },
    Epoch {
        y: 2016,
        m: 12,
        d: 31,
        hh: 23,
        mm: 59,
        ss: 60.0,
        name: "2016-12-31T23:59:60",
    },
    Epoch {
        y: 2017,
        m: 1,
        d: 1,
        hh: 0,
        mm: 0,
        ss: 0.0,
        name: "2017-01-01T00:00:00",
    },
    Epoch {
        y: 2024,
        m: 1,
        d: 1,
        hh: 0,
        mm: 0,
        ss: 0.0,
        name: "2024-01-01T00:00:00",
    },
];

/// The two declared GCRS test states (km, km/s, km/s^2), in fixture order.
pub const POSITIONS: [[f64; 3]; 2] = [[7000.0, 0.0, 0.0], [0.0, -7000.0, 1000.0]];
pub const VELOCITIES: [[f64; 3]; 2] = [[0.0, 7.5, 1.0], [-7.4, 0.2, 0.0]];
pub const ACCELERATIONS: [[f64; 3]; 2] = [[-0.008, 0.0, 0.0], [0.0, 0.0075, -0.001]];

fn matrix_multiply(a: &DdMat, b: &DdMat) -> DdMat {
    let &[[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]] = b;
    let columns = [[b00, b10, b20], [b01, b11, b21], [b02, b12, b22]];
    let mut out = [[from(0.0); 3]; 3];
    for (out_row, left_row) in out.iter_mut().zip(a) {
        for (out_value, column) in out_row.iter_mut().zip(columns) {
            let mut value = from(0.0);
            for (left, right) in left_row.iter().zip(column) {
                value = value.add_dd(left.mul_dd(right));
            }
            *out_value = value;
        }
    }
    out
}

/// GCRS->ITRS rotation matrix (double-double) at a stencil offset (seconds).
///
/// # Errors
///
/// Returns [`FrameChainError`] for invalid time-scale or EOP input.
pub fn frame_matrix(
    epoch: &Epoch,
    policy: EopPolicy,
    offset: f64,
    finals: &str,
) -> Result<DdMat, FrameChainError> {
    // Anchor UTC -> continuous TAI.
    let (utc_status, utc1, utc2) =
        dtf2d_utc(epoch.y, epoch.m, epoch.d, epoch.hh, epoch.mm, epoch.ss);
    if utc_status < 0 {
        return Err(FrameChainError::TimeScale("anchor UTC conversion failed"));
    }
    let (tai_status, anchor1, anchor2) = utctai(utc1, utc2);
    if tai_status != 0 {
        return Err(FrameChainError::TimeScale("anchor TAI conversion failed"));
    }

    let sample1 = from(anchor1);
    let sample2 = from(anchor2).add_dd(from(offset / DAYSEC));

    let (
        raw_ut1_tai,
        polar_motion_x,
        polar_motion_y,
        celestial_longitude_correction,
        celestial_obliquity_correction,
    );
    match policy {
        EopPolicy::Real => {
            let (calendar_status, _djm0, djm) = cal2jd(epoch.y, epoch.m, epoch.d);
            if calendar_status < 0 {
                return Err(FrameChainError::TimeScale(
                    "EOP centre calendar conversion failed",
                ));
            }
            let center = djm
                .to_i32()
                .ok_or(FrameChainError::TimeScale("EOP centre MJD is not integral"))?;
            let previous = center
                .checked_sub(1)
                .ok_or(FrameChainError::TimeScale("EOP centre MJD underflow"))?;
            let next = center
                .checked_add(1)
                .ok_or(FrameChainError::TimeScale("EOP centre MJD overflow"))?;
            let following = center
                .checked_add(2)
                .ok_or(FrameChainError::TimeScale("EOP centre MJD overflow"))?;
            let node_mjds = [previous, center, next, following];
            let mut abscissae = [from(0.0); 4];
            let mut raw_values = [0.0_f64; 4];
            let mut xp_values = [0.0_f64; 4];
            let mut yp_values = [0.0_f64; 4];
            let mut longitude_correction_samples = [0.0_f64; 4];
            let mut obliquity_correction_samples = [0.0_f64; 4];
            for (
                (((((abscissa, raw_value), xp_value), yp_value), longitude_value), obliquity_value),
                mjd,
            ) in abscissae
                .iter_mut()
                .zip(&mut raw_values)
                .zip(&mut xp_values)
                .zip(&mut yp_values)
                .zip(&mut longitude_correction_samples)
                .zip(&mut obliquity_correction_samples)
                .zip(node_mjds)
            {
                let row = load_eop(finals, mjd)?;
                let (node_status, node1, node2) = utctai(DJM0, f64::from(mjd));
                if node_status != 0 {
                    return Err(FrameChainError::TimeScale(
                        "EOP node UTC/TAI conversion failed",
                    ));
                }
                *abscissa = from(node1)
                    .sub_dd(from(anchor1))
                    .add_dd(from(node2).sub_dd(from(anchor2)))
                    .scale(DAYSEC);
                *raw_value = row.dut1 - dat_at_mjd(mjd)?;
                *xp_value = row.xp;
                *yp_value = row.yp;
                *longitude_value = row.dx;
                *obliquity_value = row.dy;
            }
            raw_ut1_tai = lagrange(from(offset), &abscissae, &raw_values);
            polar_motion_x = lagrange(from(offset), &abscissae, &xp_values).to_f64() * DAS2R;
            polar_motion_y = lagrange(from(offset), &abscissae, &yp_values).to_f64() * DAS2R;
            celestial_longitude_correction =
                lagrange(from(offset), &abscissae, &longitude_correction_samples).to_f64()
                    * 1e-3
                    * DAS2R;
            celestial_obliquity_correction =
                lagrange(from(offset), &abscissae, &obliquity_correction_samples).to_f64()
                    * 1e-3
                    * DAS2R;
        }
        EopPolicy::Zero => {
            let day_fraction = (f64::from(epoch.hh) * 3600.0
                + f64::from(epoch.mm) * 60.0
                + epoch.ss.min(59.999_999))
                / DAYSEC;
            let (status, delta_at) = dat(epoch.y, epoch.m, epoch.d, day_fraction);
            if status < 0 || !delta_at.is_finite() {
                return Err(FrameChainError::TimeScale(
                    "zero-EOP anchor TAI-UTC conversion failed",
                ));
            }
            raw_ut1_tai = from(-delta_at);
            polar_motion_x = 0.0;
            polar_motion_y = 0.0;
            celestial_longitude_correction = 0.0;
            celestial_obliquity_correction = 0.0;
        }
    }

    let (tt1, tt2) = taitt(sample1.to_f64(), sample2.to_f64());
    let (x, y, s) = xys06a(tt1, tt2);
    let rc2i = c2ixys(
        x + celestial_longitude_correction,
        y + celestial_obliquity_correction,
        s,
    );
    let rpom = pom00(polar_motion_x, polar_motion_y, sp00(tt1, tt2));

    let earth_rotation_day = sample1;
    let earth_rotation_fraction = sample2.add_dd(raw_ut1_tai.div_dd(from(DAYSEC)));
    let (sine, cosine) = sincos(era(earth_rotation_day, earth_rotation_fraction));

    let zero = from(0.0);
    let one = from(1.0);
    let r3: DdMat = [
        [cosine, sine, zero],
        [sine.neg_dd(), cosine, zero],
        [zero, zero, one],
    ];
    let [[rc2i_00, rc2i_01, rc2i_02], [rc2i_10, rc2i_11, rc2i_12], [rc2i_20, rc2i_21, rc2i_22]] =
        rc2i;
    let [[rpom_00, rpom_01, rpom_02], [rpom_10, rpom_11, rpom_12], [rpom_20, rpom_21, rpom_22]] =
        rpom;
    let rc2i_dd = [
        [from(rc2i_00), from(rc2i_01), from(rc2i_02)],
        [from(rc2i_10), from(rc2i_11), from(rc2i_12)],
        [from(rc2i_20), from(rc2i_21), from(rc2i_22)],
    ];
    let rpom_dd = [
        [from(rpom_00), from(rpom_01), from(rpom_02)],
        [from(rpom_10), from(rpom_11), from(rpom_12)],
        [from(rpom_20), from(rpom_21), from(rpom_22)],
    ];
    let intermediate = matrix_multiply(&r3, &rc2i_dd);
    Ok(matrix_multiply(&rpom_dd, &intermediate))
}

/// `R`, `Rdot`, `Rddot` from the conditioned centered five-point stencil.
///
/// # Errors
///
/// Returns [`FrameChainError`] when any stencil sample cannot be constructed.
pub fn derivatives(
    epoch: &Epoch,
    policy: EopPolicy,
    h: f64,
    finals: &str,
) -> Result<(DdMat, DdMat, DdMat), FrameChainError> {
    let mut samples = [[[from(0.0); 3]; 3]; 5];
    for (sample, offset) in samples.iter_mut().zip([-2.0_f64, -1.0, 0.0, 1.0, 2.0]) {
        *sample = frame_matrix(epoch, policy, offset * h, finals)?;
    }
    let [minus_two, minus_one, value, plus_one, plus_two] = samples;
    let mut derivative = [[from(0.0); 3]; 3];
    let mut second = [[from(0.0); 3]; 3];
    for (
        (((((derivative_row, second_row), minus_two_row), minus_one_row), value_row), plus_one_row),
        plus_two_row,
    ) in derivative
        .iter_mut()
        .zip(&mut second)
        .zip(&minus_two)
        .zip(&minus_one)
        .zip(&value)
        .zip(&plus_one)
        .zip(&plus_two)
    {
        for (
            (
                (
                    (((derivative_value, second_value), minus_two_value), minus_one_value),
                    value_at_zero,
                ),
                plus_one_value,
            ),
            plus_two_value,
        ) in derivative_row
            .iter_mut()
            .zip(second_row)
            .zip(minus_two_row)
            .zip(minus_one_row)
            .zip(value_row)
            .zip(plus_one_row)
            .zip(plus_two_row)
        {
            let near_difference = plus_one_value.sub_dd(*minus_one_value);
            let far_difference = plus_two_value.sub_dd(*minus_two_value);
            let near_second = plus_one_value
                .sub_dd(*value_at_zero)
                .add_dd(minus_one_value.sub_dd(*value_at_zero));
            let far_second = plus_two_value
                .sub_dd(*value_at_zero)
                .add_dd(minus_two_value.sub_dd(*value_at_zero));
            *derivative_value = near_difference
                .scale(8.0)
                .sub_dd(far_difference)
                .div_dd(from(12.0 * h));
            *second_value = near_second
                .scale(16.0)
                .sub_dd(far_second)
                .div_dd(from(12.0 * h * h));
        }
    }
    Ok((value, derivative, second))
}

/// Transform a GCRS state to ITRS using `R`, `Rdot`, `Rddot` (double-double).
#[must_use]
pub fn transform_state(
    rotation: &DdMat,
    rotation_rate: &DdMat,
    rotation_acceleration: &DdMat,
    r_gcrs: &[f64; 3],
    v_gcrs: &[f64; 3],
    a_gcrs: &[f64; 3],
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let mut r_itrs = [0.0; 3];
    let mut v_itrs = [0.0; 3];
    let mut a_itrs = [0.0; 3];
    for (
        ((((rotation_row, rate_row), acceleration_row), position_output), velocity_output),
        acceleration_output,
    ) in rotation
        .iter()
        .zip(rotation_rate)
        .zip(rotation_acceleration)
        .zip(r_itrs.iter_mut())
        .zip(v_itrs.iter_mut())
        .zip(a_itrs.iter_mut())
    {
        let mut position = from(0.0);
        let mut velocity = from(0.0);
        let mut acceleration = from(0.0);
        for (
            ((((matrix_value, matrix_rate), matrix_acceleration), position_input), velocity_input),
            acceleration_input,
        ) in rotation_row
            .iter()
            .zip(rate_row)
            .zip(acceleration_row)
            .zip(r_gcrs)
            .zip(v_gcrs)
            .zip(a_gcrs)
        {
            position = position.add_dd(matrix_value.scale(*position_input));
            velocity = velocity.add_dd(
                matrix_value
                    .scale(*velocity_input)
                    .add_dd(matrix_rate.scale(*position_input)),
            );
            acceleration = acceleration.add_dd(
                matrix_value
                    .scale(*acceleration_input)
                    .add_dd(matrix_rate.scale(2.0 * *velocity_input))
                    .add_dd(matrix_acceleration.scale(*position_input)),
            );
        }
        *position_output = position.to_f64();
        *velocity_output = velocity.to_f64();
        *acceleration_output = acceleration.to_f64();
    }
    (r_itrs, v_itrs, a_itrs)
}
