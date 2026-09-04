//! Fundamental arguments (IERS 2003 / MHB2000), transliterated from sealed ERFA 2.0.1.
//!
//! Includes `fal03`, `falp03`, `faf03`, `fad03`, `faom03`, planetary
//! `fame03`..`faur03`, and `fapa03`. `t` is Julian centuries since J2000 TT.
//!
//! ERFA's Neptune term `fane03` is deliberately absent: this module is private
//! and the IAU 2006 series in `iau2006.rs` takes no Neptune argument, so it
//! would have had no caller and no fixture comparing it.
//!
//! Rust `%` on `f64` matches C `fmod` (remainder with the sign of the dividend),
//! so the reductions reproduce ERFA bit-for-bit.

use super::timescale::{D2PI, DAS2R, TURNAS};

/// Mean anomaly of the Moon.
#[must_use]
pub fn fal03(t: f64) -> f64 {
    (485_868.249_036
        + t * (1_717_915_923.217_8 + t * (31.879_2 + t * (0.051_635 + t * (-0.000_244_70)))))
        % TURNAS
        * DAS2R
}

/// Mean anomaly of the Sun.
#[must_use]
pub fn falp03(t: f64) -> f64 {
    (1_287_104.793_048
        + t * (129_596_581.048_1 + t * (-0.553_2 + t * (0.000_136 + t * (-0.000_011_49)))))
        % TURNAS
        * DAS2R
}

/// Mean longitude of the Moon minus that of the ascending node.
#[must_use]
pub fn faf03(t: f64) -> f64 {
    (335_779.526_232
        + t * (1_739_527_262.847_8 + t * (-12.751_2 + t * (-0.001_037 + t * (0.000_004_17)))))
        % TURNAS
        * DAS2R
}

/// Mean elongation of the Moon from the Sun.
#[must_use]
pub fn fad03(t: f64) -> f64 {
    (1_072_260.703_692
        + t * (1_602_961_601.209_0 + t * (-6.370_6 + t * (0.006_593 + t * (-0.000_031_69)))))
        % TURNAS
        * DAS2R
}

/// Mean longitude of the ascending node of the Moon.
#[must_use]
pub fn faom03(t: f64) -> f64 {
    (450_160.398_036
        + t * (-6_962_890.543_1 + t * (7.472_2 + t * (0.007_702 + t * (-0.000_059_39)))))
        % TURNAS
        * DAS2R
}

/// Mean longitude of Mercury.
#[must_use]
pub fn fame03(t: f64) -> f64 {
    (4.402_608_842 + 2_608.790_314_157_4 * t) % D2PI
}

/// Mean longitude of Venus.
#[must_use]
pub fn fave03(t: f64) -> f64 {
    (3.176_146_697 + 1_021.328_554_621_1 * t) % D2PI
}

/// Mean longitude of Earth.
#[must_use]
pub fn fae03(t: f64) -> f64 {
    (1.753_470_314 + 628.307_584_999_1 * t) % D2PI
}

/// Mean longitude of Mars.
#[must_use]
pub fn fama03(t: f64) -> f64 {
    (6.203_480_913 + 334.061_242_670_0 * t) % D2PI
}

/// Mean longitude of Jupiter.
#[must_use]
pub fn faju03(t: f64) -> f64 {
    (0.599_546_497 + 52.969_096_264_1 * t) % D2PI
}

/// Mean longitude of Saturn.
#[must_use]
pub fn fasa03(t: f64) -> f64 {
    (0.874_016_757 + 21.329_910_496_0 * t) % D2PI
}

/// Mean longitude of Uranus.
#[must_use]
pub fn faur03(t: f64) -> f64 {
    (5.481_293_872 + 7.478_159_856_7 * t) % D2PI
}

/// General accumulated precession in longitude.
#[must_use]
pub fn fapa03(t: f64) -> f64 {
    (0.024_381_750 + 0.000_005_386_91 * t) * t
}
