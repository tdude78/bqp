//! Physical constants shared by the scalar RHS and its dual-number twin.
//!
//! # Why this module exists
//!
//! `rhs.rs` and `rhs_dual.rs` compute the same dynamics — one in `f64`, one in
//! `DualVec` so the Jacobian and STM paths can differentiate through it. They
//! carried **eleven independently declared copies** of the constants below.
//! Eleven copies that agree today is not eleven copies that agree tomorrow, and
//! the failure mode is silent: `rhs_dual` supplies the DERIVATIVES, so a drift
//! between the two files does not produce a wrong value with a wrong answer next
//! to it. It produces a correct value with a derivative that no longer belongs
//! to it, which nothing in the test suite is positioned to notice.
//!
//! That was not a hypothetical. The Lorentz dipole direction had **already
//! diverged** when this module was written: `rhs.rs` used precomputed literals
//! `LORENTZ_THETA_SIN`/`LORENTZ_THETA_COS`, while `rhs_dual.rs` called
//! `LORENTZ_DIPOLE_THETA_RAD.sin_cos()` at runtime. The guard test
//! `lorentz_dipole_sin_cos_match_their_stated_angle` asserts those agree to
//! `1e-15`, which is a tolerance, not bit equality — so the scalar and dual paths
//! were free to use different bits for the same physical direction. Both now read
//! the same two literals, so the question cannot arise.
//!
//! # Scope
//!
//! Only constants genuinely shared by both files belong here. Values specific to
//! one path stay where they are used. Nothing here is `pub` beyond the crate:
//! these are implementation details of the force model, not an API.
//!
//! Constants that live elsewhere on purpose and must NOT be duplicated into this
//! module: `satpy_core::{MU, RE, J2}` and `GRAVITY_REFERENCE_RADIUS_KM` (the
//! DIR-R6 gravity reference radius, deliberately 54 cm SMALLER than WGS84 `RE`
//! — 6378.13646 km against 6378.137 km — and used only inside the
//! spherical-harmonic kernels). This said "larger" until 2026-08-09, which
//! inverted the sign of the only quantity it states. The same sentence next to
//! `WGS84_FLATTENING` in `rhs.rs` was corrected on 2026-08-04; this copy was
//! missed then, which is the argument for not having two copies.

pub const KM_TO_M: f64 = 1000.0;
pub const M_TO_KM: f64 = 0.001;
pub const INV_LIGHT_SPEED_SQ: f64 = 1.0 / (299_792.458 * 299_792.458);

/// Earth dipole moment (Tesla * m^3).
pub const EARTH_DIPOLE_STRENGTH: f64 = 7.94e15;

/// Geomagnetic dipole colatitude. Nothing evaluates this at runtime: the sine
/// and cosine below are precomputed and both RHS paths read those. It is kept
/// because it is the only record of where those two literals come from, and
/// `lorentz_dipole_sin_cos_match_their_stated_angle` fails if they drift from
/// it -- a bare pair of magic numbers could be edited to anything. `cfg(test)`
/// because being checked is its entire job.
#[cfg(test)]
pub const LORENTZ_DIPOLE_THETA_RAD: f64 = 169.74_f64.to_radians();
pub const LORENTZ_THETA_SIN: f64 = 0.178_115_290_264_210_22;
pub const LORENTZ_THETA_COS: f64 = -0.984_009_625_651_139_7;

pub const MEAN_ION_MASS_KG: f64 = 2.656_696_2e-26;
pub const MIN_NUMBER_DENSITY: f64 = 1e6;
pub const MIN_COULOMB_LOG: f64 = 1.0;
pub const BOLTZMANN_K: f64 = 1.380_649e-23;
pub const ELEMENTARY_CHARGE: f64 = 1.6e-19;

/// Astronomical unit in km (IAU 2012 definition).
pub const AU_KM: f64 = 149_597_870.7;

/// Vacuum permittivity ε₀ (F/m). Required for SI-form Debye length.
/// Audit Phase 3.1 fix: prior code used the Gaussian/CGS Debye formula
/// `kT / (4π n e²)` evaluated with SI constants — off by sqrt(4π ε₀)⁻¹
/// ≈ 95,000×. No production config currently sets
/// `ForceFlags::COULOMB_DRAG` (the flag itself stays parseable and would
/// fly if set — nothing gates it off); the formula is now SI-correct for
/// any future enablement.
pub const VACUUM_PERMITTIVITY: f64 = 8.854_187_812_8e-12;
