//! Rotation-matrix primitives and CIO-based frame matrices.
//!
//! Transliterated from sealed ERFA 2.0.1 (`ir`, `rx`, `ry`, `rz`, `sp00`, `pom00`, `c2ixys`,
//! `fw2m`). Ordinary binary64: these feed the double-double composition as
//! `dd::from` operands, so f64 fidelity is within the 5e-13 budget.

use super::timescale::{DAS2R, DJ00, DJC};

pub type Mat3 = [[f64; 3]; 3];

#[must_use]
pub const fn ir() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Rotate an `r` matrix about the x-axis (pre-multiply by `R_x(phi)`).
pub fn rx(phi: f64, r: &mut Mat3) {
    let s = phi.sin();
    let c = phi.cos();
    let a10 = c * r[1][0] + s * r[2][0];
    let a11 = c * r[1][1] + s * r[2][1];
    let a12 = c * r[1][2] + s * r[2][2];
    let a20 = -s * r[1][0] + c * r[2][0];
    let a21 = -s * r[1][1] + c * r[2][1];
    let a22 = -s * r[1][2] + c * r[2][2];
    r[1] = [a10, a11, a12];
    r[2] = [a20, a21, a22];
}

/// Rotate an r-matrix about the y-axis.
pub fn ry(theta: f64, r: &mut Mat3) {
    let s = theta.sin();
    let c = theta.cos();
    let a00 = c * r[0][0] - s * r[2][0];
    let a01 = c * r[0][1] - s * r[2][1];
    let a02 = c * r[0][2] - s * r[2][2];
    let a20 = s * r[0][0] + c * r[2][0];
    let a21 = s * r[0][1] + c * r[2][1];
    let a22 = s * r[0][2] + c * r[2][2];
    r[0] = [a00, a01, a02];
    r[2] = [a20, a21, a22];
}

/// Rotate an r-matrix about the z-axis.
pub fn rz(psi: f64, r: &mut Mat3) {
    let s = psi.sin();
    let c = psi.cos();
    let a00 = c * r[0][0] + s * r[1][0];
    let a01 = c * r[0][1] + s * r[1][1];
    let a02 = c * r[0][2] + s * r[1][2];
    let a10 = -s * r[0][0] + c * r[1][0];
    let a11 = -s * r[0][1] + c * r[1][1];
    let a12 = -s * r[0][2] + c * r[1][2];
    r[0] = [a00, a01, a02];
    r[1] = [a10, a11, a12];
}

/// TIO locator s', positioning the Terrestrial Intermediate Origin.
#[must_use]
pub fn sp00(date1: f64, date2: f64) -> f64 {
    let t = ((date1 - DJ00) + date2) / DJC;
    -47e-6 * t * DAS2R
}

/// Polar-motion matrix from polar coordinates and s'.
#[must_use]
pub fn pom00(xp: f64, yp: f64, sp: f64) -> Mat3 {
    let mut r = ir();
    rz(sp, &mut r);
    ry(-xp, &mut r);
    rx(-yp, &mut r);
    r
}

/// Celestial-to-intermediate matrix from CIP (x, y) and CIO locator s.
#[must_use]
pub fn c2ixys(cip_x: f64, cip_y: f64, cio_locator: f64) -> Mat3 {
    let radius_squared = cip_x * cip_x + cip_y * cip_y;
    let longitude = if radius_squared > 0.0 {
        cip_y.atan2(cip_x)
    } else {
        0.0
    };
    let tilt = (radius_squared / (1.0 - radius_squared)).sqrt().atan();
    let mut matrix = ir();
    rz(longitude, &mut matrix);
    ry(tilt, &mut matrix);
    rz(-(longitude + cio_locator), &mut matrix);
    matrix
}

/// Extract CIP (x, y) from a bias-precession-nutation matrix.
#[must_use]
pub const fn bpn2xy_pair(rbpn: &Mat3) -> (f64, f64) {
    (rbpn[2][0], rbpn[2][1])
}

/// Form a rotation matrix from Fukushima-Williams angles.
#[must_use]
pub fn fw2m(gamb: f64, phib: f64, psi: f64, eps: f64) -> Mat3 {
    let mut r = ir();
    rz(gamb, &mut r);
    rx(phib, &mut r);
    rz(-psi, &mut r);
    rx(-eps, &mut r);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    // Gate 4: c2ixys / sp00 / pom00 against ERFA canonical unit-test values.
    #[test]
    fn gate4_cio_and_polar_motion() {
        let x = f64::from_bits(0x3f42_fa1a_070e_fd6d);
        let y = f64::from_bits(0x3f05_1454_b4b6_426f);
        let s = f64::from_bits(0xbe4a_333e_d8d8_3281);
        let rc2i = c2ixys(x, y, s);
        assert!((rc2i[0][0] - f64::from_bits(0x3fef_ffff_a5f7_ff89)).abs() < 1e-13);
        assert!((rc2i[0][2] - f64::from_bits(0xbf42_fa1a_0754_0687)).abs() < 1e-13);
        assert!((rc2i[1][1] - f64::from_bits(0x3fef_ffff_ff90_ea1f)).abs() < 1e-13);
        assert!((rc2i[2][0] - x).abs() < 1e-13);
        assert!((rc2i[2][2] - f64::from_bits(0x3fef_ffff_a588_e9aa)).abs() < 1e-13);

        assert!((sp00(2_400_000.5, 52541.0) - f64::from_bits(0xbd9b_5761_56a2_fb42)).abs() < 1e-24);

        let rpom = pom00(
            2.550_602_38e-7,
            1.860_359_247e-6,
            -1.367_174_580_728_891_5e-11,
        );
        assert!((rpom[0][0] - f64::from_bits(0x3fef_ffff_ffff_fedb)).abs() < 1e-13);
        assert!((rpom[0][2] - f64::from_bits(0x3e91_1de6_ca34_1596)).abs() < 1e-16);
        assert!((rpom[1][1] - f64::from_bits(0x3fef_ffff_ffff_c31d)).abs() < 1e-13);
    }
}
