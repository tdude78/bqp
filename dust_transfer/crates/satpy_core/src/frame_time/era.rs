//! Earth Rotation Angle in double-double, transliterated from sealed 4AF
//! generator (`dd_era`).
//!
//! Split constants are generator's exact
//! `dd(hi, lo)` pairs; Rust has no hex-float literal, so each is reconstructed
//! from its IEEE-754 bit pattern (bit-identical to the C `0x...p...` forms).
//!
//! The generator raises ERA to double-double specifically to remove the
//! ~1.97e-14 rad binary64 argument-reduction noise that the five-point stencil
//! would otherwise amplify past the 1e-13 s^-1 Rdot bound.

use super::dd::{dd, from, Dd};

#[inline]
const fn c0() -> Dd {
    dd(
        f64::from_bits(0x3fe8_ee09_84cc_2772),
        f64::from_bits(0x3c73_7791_8272_71b7),
    )
}

#[inline]
const fn c1() -> Dd {
    dd(
        f64::from_bits(0x3f66_6d9b_93e6_5515),
        f64::from_bits(0x3c01_a9fd_4c72_9390),
    )
}

/// `2π` as a double-double (generator's `tau`).
#[inline]
#[must_use]
pub const fn tau() -> Dd {
    dd(
        f64::from_bits(0x4019_21fb_5444_2d18),
        f64::from_bits(0x3cb1_a626_3314_5c07),
    )
}

/// Earth Rotation Angle (radians) from a two-part UT1 date, in double-double.
#[must_use]
pub fn era(ut11: Dd, ut12: Dd) -> Dd {
    let (d1, d2) = if ut11.to_f64() < ut12.to_f64() {
        (ut11, ut12)
    } else {
        (ut12, ut11)
    };
    let t = d1.add_dd(d2.sub_dd(from(2_451_545.0)));
    let f = d1
        .sub_dd(from(d1.hi.floor()))
        .add_dd(d2.sub_dd(from(d2.hi.floor())));

    let tau = tau();
    let mut theta = tau.mul_dd(f.add_dd(c0()).add_dd(c1().mul_dd(t)));
    let turns = theta.div_dd(tau).to_f64().floor();
    theta = theta.sub_dd(tau.scale(turns));
    if theta.to_f64() < 0.0 {
        theta = theta.add_dd(tau);
    }
    theta
}

#[cfg(test)]
mod tests {
    use super::super::dd::from;
    use super::era;

    // Gate 2: double-double ERA against the ERFA eraEra00 canonical value.
    #[test]
    fn gate2_era00_canonical() {
        let theta = era(from(2_400_000.5), from(54388.0)).to_f64();
        let reference = f64::from_bits(0x3fd9_bf04_3b93_8823);
        assert!((theta - reference).abs() < 1e-12, "ERA {theta}");
    }
}
