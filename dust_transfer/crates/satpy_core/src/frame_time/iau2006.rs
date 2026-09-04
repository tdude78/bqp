//! IAU 2006/2000A CIO-based precession-nutation, transliterated from sealed ERFA 2.0.1.
//!
//! Includes `nut00a`, `nut06a`, `pfw06`, `obl06`, `pnm06a`, `bpn2xy`, `s06`,
//! and `xys06a`. Series tables live in generated `tables` module.
//!
//! Note: `nut00a` deliberately uses MHB2000 forms for l', D and the planetary
//! arguments that differ slightly from the IERS-2003 `fa*03` helpers; those are
//! inlined here exactly as ERFA does, not delegated.

use super::cio::{bpn2xy_pair, fw2m, Mat3};
use super::fund_args::{
    fad03, fae03, faf03, faju03, fal03, falp03, fama03, fame03, faom03, fapa03, fasa03, faur03,
    fave03,
};
use super::tables::{NUT_LS, NUT_PL, S06_S0, S06_S1, S06_S2, S06_S3, S06_S4};
use super::timescale::{D2PI, DAS2R, DJ00, DJC, TURNAS, U2R};

#[inline]
#[must_use]
fn t_jc(date1: f64, date2: f64) -> f64 {
    ((date1 - DJ00) + date2) / DJC
}

/// IAU 2000A nutation in longitude and obliquity (radians). Returns `(dpsi, deps)`.
#[must_use]
pub fn nut00a(date1: f64, date2: f64) -> (f64, f64) {
    let t = t_jc(date1, date2);

    // Luni-solar fundamental (Delaunay) arguments.
    let el = fal03(t);
    let elp = (1_287_104.793_05
        + t * (129_596_581.048_1 + t * (-0.553_2 + t * (0.000_136 + t * (-0.000_011_49)))))
        % TURNAS
        * DAS2R;
    let f = faf03(t);
    let d = (1_072_260.703_69
        + t * (1_602_961_601.209_0 + t * (-6.370_6 + t * (0.006_593 + t * (-0.000_031_69)))))
        % TURNAS
        * DAS2R;
    let om = faom03(t);

    let mut dp = 0.0_f64;
    let mut de = 0.0_f64;
    for term in NUT_LS.iter().rev() {
        let [moon_anomaly, sun_anomaly, moon_latitude, moon_elongation, moon_node] = term.n;
        let arg = (f64::from(moon_anomaly) * el
            + f64::from(sun_anomaly) * elp
            + f64::from(moon_latitude) * f
            + f64::from(moon_elongation) * d
            + f64::from(moon_node) * om)
            % D2PI;
        let sarg = arg.sin();
        let carg = arg.cos();
        dp += (term.sp + term.spt * t) * sarg + term.cp * carg;
        de += (term.ce + term.cet * t) * carg + term.se * sarg;
    }
    let dpsils = dp * U2R;
    let depsls = de * U2R;

    // Planetary arguments (MHB2000 forms for the lunar quantities).
    let moon_mean_anomaly = (2.355_555_98 + 8_328.691_426_955_4 * t) % D2PI;
    let moon_latitude = (1.627_905_234 + 8_433.466_158_131 * t) % D2PI;
    let moon_elongation = (5.198_466_741 + 7_771.377_146_812_1 * t) % D2PI;
    let moon_node = (2.182_439_20 - 33.757_045 * t) % D2PI;
    let general_precession = fapa03(t);
    let mercury_longitude = fame03(t);
    let venus_longitude = fave03(t);
    let earth_longitude = fae03(t);
    let mars_longitude = fama03(t);
    let jupiter_longitude = faju03(t);
    let saturn_longitude = fasa03(t);
    let uranus_longitude = faur03(t);
    let neptune_longitude = (5.321_159_000 + 3.812_777_400_0 * t) % D2PI;

    let mut dp = 0.0_f64;
    let mut de = 0.0_f64;
    for term in NUT_PL.iter().rev() {
        let [moon_anomaly_multiplier, moon_latitude_multiplier, moon_elongation_multiplier, moon_node_multiplier, mercury_multiplier, venus_multiplier, earth_multiplier, mars_multiplier, jupiter_multiplier, saturn_multiplier, uranus_multiplier, neptune_multiplier, precession_multiplier] =
            term.n;
        let arg = (f64::from(moon_anomaly_multiplier) * moon_mean_anomaly
            + f64::from(moon_latitude_multiplier) * moon_latitude
            + f64::from(moon_elongation_multiplier) * moon_elongation
            + f64::from(moon_node_multiplier) * moon_node
            + f64::from(mercury_multiplier) * mercury_longitude
            + f64::from(venus_multiplier) * venus_longitude
            + f64::from(earth_multiplier) * earth_longitude
            + f64::from(mars_multiplier) * mars_longitude
            + f64::from(jupiter_multiplier) * jupiter_longitude
            + f64::from(saturn_multiplier) * saturn_longitude
            + f64::from(uranus_multiplier) * uranus_longitude
            + f64::from(neptune_multiplier) * neptune_longitude
            + f64::from(precession_multiplier) * general_precession)
            % D2PI;
        let sarg = arg.sin();
        let carg = arg.cos();
        dp += f64::from(term.sp) * sarg + f64::from(term.cp) * carg;
        de += f64::from(term.se) * sarg + f64::from(term.ce) * carg;
    }
    let dpsipl = dp * U2R;
    let depspl = de * U2R;

    (dpsils + dpsipl, depsls + depspl)
}

/// IAU 2006/2000A nutation (radians). Returns `(dpsi, deps)`.
#[must_use]
pub fn nut06a(date1: f64, date2: f64) -> (f64, f64) {
    let t = t_jc(date1, date2);
    let fj2 = -2.7774e-6 * t;
    let (dp, de) = nut00a(date1, date2);
    (dp + dp * (0.4697e-6 + fj2), de + de * fj2)
}

/// Mean obliquity of the ecliptic, IAU 2006.
#[must_use]
pub fn obl06(date1: f64, date2: f64) -> f64 {
    let t = t_jc(date1, date2);
    (84_381.406
        + (-46.836_769
            + (-0.000_183_1 + (0.002_003_40 + (-0.000_000_576 + (-0.000_000_043_4) * t) * t) * t)
                * t)
            * t)
        * DAS2R
}

/// Precession angles, IAU 2006 (Fukushima-Williams). Returns `(gamb, phib, psib, epsa)`.
#[must_use]
pub fn pfw06(date1: f64, date2: f64) -> (f64, f64, f64, f64) {
    let t = t_jc(date1, date2);
    let frame_bias_longitude = (-0.052_928
        + (10.556_378
            + (0.493_204_4 + (-0.000_312_38 + (-0.000_002_788 + (0.000_000_026_0) * t) * t) * t)
                * t)
            * t)
        * DAS2R;
    let precession_inclination = (84_381.412_819
        + (-46.811_016
            + (0.051_126_8 + (0.000_532_89 + (-0.000_000_440 + (-0.000_000_017_6) * t) * t) * t)
                * t)
            * t)
        * DAS2R;
    let precession_longitude = (-0.041_775
        + (5_038.481_484
            + (1.558_417_5 + (-0.000_185_22 + (-0.000_026_452 + (-0.000_000_014_8) * t) * t) * t)
                * t)
            * t)
        * DAS2R;
    let mean_obliquity = obl06(date1, date2);
    (
        frame_bias_longitude,
        precession_inclination,
        precession_longitude,
        mean_obliquity,
    )
}

/// Bias-precession-nutation matrix, IAU 2006/2000A.
#[must_use]
pub fn pnm06a(date1: f64, date2: f64) -> Mat3 {
    let (frame_bias_longitude, precession_inclination, precession_longitude, mean_obliquity) =
        pfw06(date1, date2);
    let (nutation_longitude, nutation_obliquity) = nut06a(date1, date2);
    fw2m(
        frame_bias_longitude,
        precession_inclination,
        precession_longitude + nutation_longitude,
        mean_obliquity + nutation_obliquity,
    )
}

/// CIO locator s, given CIP (x, y). IAU 2006/2000A.
#[must_use]
pub fn s06(date1: f64, date2: f64, x: f64, y: f64) -> f64 {
    let t = t_jc(date1, date2);

    let fundamental_arguments = [
        fal03(t),
        falp03(t),
        faf03(t),
        fad03(t),
        faom03(t),
        fave03(t),
        fae03(t),
        fapa03(t),
    ];

    // Polynomial coefficients (microarcsec), w0..w5.
    let mut coefficient_0 = 94.00e-6;
    let mut coefficient_1 = 3808.65e-6;
    let mut coefficient_2 = -122.68e-6;
    let mut coefficient_3 = -72574.11e-6;
    let mut coefficient_4 = 27.98e-6;
    let coefficient_5 = 15.62e-6;

    let accumulate = |w: &mut f64, table: &[super::tables::CioTerm]| {
        for term in table.iter().rev() {
            let mut argument = 0.0_f64;
            for (multiplier, fundamental_argument) in term.n.iter().zip(&fundamental_arguments) {
                argument += f64::from(*multiplier) * fundamental_argument;
            }
            *w += term.s * argument.sin() + term.c * argument.cos();
        }
    };
    accumulate(&mut coefficient_0, &S06_S0);
    accumulate(&mut coefficient_1, &S06_S1);
    accumulate(&mut coefficient_2, &S06_S2);
    accumulate(&mut coefficient_3, &S06_S3);
    accumulate(&mut coefficient_4, &S06_S4);

    (coefficient_0
        + (coefficient_1
            + (coefficient_2 + (coefficient_3 + (coefficient_4 + coefficient_5 * t) * t) * t) * t)
            * t)
        * DAS2R
        - x * y / 2.0
}

/// CIP (x, y) and CIO locator s, IAU 2006/2000A. Returns `(x, y, s)`.
#[must_use]
pub fn xys06a(date1: f64, date2: f64) -> (f64, f64, f64) {
    let rbpn = pnm06a(date1, date2);
    let (x, y) = bpn2xy_pair(&rbpn);
    let s = s06(date1, date2, x, y);
    (x, y, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Gate 3: precession-nutation X/Y/s against ERFA canonical unit-test values.
    //
    // PROVENANCE OF EVERY BOUND BELOW: ERFA v2.0.1, `src/t_erfa_c.c` — the same
    // release this module transliterates (see the file header). Reference values
    // and tolerances are both ERFA's, copied verbatim:
    //
    //   t_xys06a :9868-:9870   x 1e-14   y 1e-15   s 1e-18
    //   t_nut00a :5585,:5587   dpsi 1e-13   deps 1e-13
    //   t_nut06a :5641,:5643   dpsi 1e-13   deps 1e-13
    //   t_obl06  :5736         1e-14
    //   t_pfw06  :6045-:6051   gamb 1e-16   phib 1e-12   psib 1e-14   epsa 1e-12
    //
    // These bounds were previously tightened 10x-1000x below ERFA's published
    // values with no recorded justification. That is what broke on x86_64 Linux:
    // `y` measured |dy| = 1.35e-16 there, which is 7.4x INSIDE ERFA's 1e-15 and
    // 1.35x outside the tightened 1e-16. Nothing about the transliteration had
    // moved. `xys06a` sums ~1400 sin/cos terms through the platform libm, and
    // IEEE-754 requires correct rounding for `sqrt` but for none of the
    // transcendentals, so Apple libm and glibc differ by 1-2 ULP per call.
    // Accumulating to 1.35e-16 is the expected outcome of that, not a port error.
    //
    // SCALE: 1.35e-16 rad is 2.8e-11 arcsec — about 5.7 nm of CIP displacement at
    // GEO, 0.9 nm at LEO.
    //
    // WHAT THIS GATE PROVES: that the transliteration is faithful to ERFA, at
    // ERFA's own standard of faithful. It does NOT prove cross-libm bit
    // agreement and cannot, for the reason above. Do not read it as that.
    //
    // IF YOU ARE ABOUT TO TIGHTEN ONE OF THESE: any bound here must be an
    // external citation, like these are. A value chosen because it passes on the
    // host in front of you is not a bound — it is a description of that host, and
    // it will fail the next time the libm, compiler, or platform changes. That is
    // precisely how the previous set of numbers got here.
    #[test]
    fn gate3_xys06a_and_nutation() {
        let (x, y, s) = xys06a(2_400_000.5, 53736.0);
        assert!(
            (x - f64::from_bits(0x3f42_fa1a_06d8_83b1)).abs() < 1e-14,
            "x {x}"
        );
        assert!(
            (y - f64::from_bits(0x3f05_1454_cd94_9261)).abs() < 1e-15,
            "y {y}"
        );
        assert!(
            (s - f64::from_bits(0xbe4a_3332_ced4_92dc)).abs() < 1e-18,
            "s {s}"
        );

        let (nutation_longitude_2000a, nutation_obliquity_2000a) = nut00a(2_400_000.5, 53736.0);
        assert!(
            (nutation_longitude_2000a - f64::from_bits(0xbee4_328e_1194_16d1)).abs() < 1e-13,
            "dpsi {nutation_longitude_2000a}"
        );
        assert!(
            (nutation_obliquity_2000a - f64::from_bits(0x3f05_4d96_5975_c17d)).abs() < 1e-13,
            "deps {nutation_obliquity_2000a}"
        );

        let (nutation_longitude_2006, nutation_obliquity_2006) = nut06a(2_400_000.5, 53736.0);
        assert!(
            (nutation_longitude_2006 - f64::from_bits(0xbee4_328e_7845_71db)).abs() < 1e-13,
            "dpsi6 {nutation_longitude_2006}"
        );
        assert!(
            (nutation_obliquity_2006 - f64::from_bits(0x3f05_4d96_1de6_7e8f)).abs() < 1e-13,
            "deps6 {nutation_obliquity_2006}"
        );

        assert!(
            (obl06(2_400_000.5, 54388.0) - f64::from_bits(0x3fda_2e48_95e8_acd0)).abs() < 1e-14
        );

        let (bias_gamma, ecliptic_inclination, precession_longitude, mean_obliquity) =
            pfw06(2_400_000.5, 50_123.999_9);
        assert!(
            (bias_gamma - f64::from_bits(0xbec2_d1a3_6a39_3bad)).abs() < 1e-16,
            "gamb {bias_gamma}"
        );
        assert!(
            (ecliptic_inclination - f64::from_bits(0x3fda_2eb7_e41e_4436)).abs() < 1e-12,
            "phib {ecliptic_inclination}"
        );
        assert!(
            (precession_longitude - f64::from_bits(0xbf4f_22d1_1f44_18d0)).abs() < 1e-14,
            "psib {precession_longitude}"
        );
        assert!(
            (mean_obliquity - f64::from_bits(0x3fda_2eb7_c56e_25bd)).abs() < 1e-12,
            "epsa {mean_obliquity}"
        );
    }
}
