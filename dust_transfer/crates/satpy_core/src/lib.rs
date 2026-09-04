// Core math and conversion logic shared between conversions_rs and propagators_rs
// OPTIMIZED VERSION: Focus on FMA, reduced branching, SIMD, and auto-vectorization

extern "C" {
    pub fn fmod(x: f64, y: f64) -> f64;
}

// blas_src extern removed: ndarray "blas" feature dropped; Accelerate was linked
// but never called. Re-enable with the blas-src dep in satpy_core/Cargo.toml if needed.
// #[cfg(any(target_os = "macos", target_os = "linux"))]
// extern crate blas_src;

use num_traits::{Float, FromPrimitive};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use wide::f64x4;

#[cfg(feature = "autodiff")]
pub mod dual;
#[cfg(feature = "autodiff")]
pub use dual::DualVec;
pub mod optim;

// Extracted modules
pub mod frame_time;
pub mod gravity;
pub mod mean_elements;
pub mod parallel_budget;
pub mod parallel_utils;

// Re-export gravity module types and functions
pub use gravity::{
    pack_gravity_coeffs, spherical_gravity, spherical_gravity_impl, spherical_gravity_impl_frame,
    spherical_gravity_impl_frame_packed, spherical_gravity_impl_generic,
    spherical_gravity_impl_generic_packed, spherical_gravity_impl_packed,
    spherical_gravity_impl_sincos, spherical_gravity_impl_sincos_packed, spherical_gravity_packed,
    validate_flat_gravity_coeffs, GravityCache, GravityCacheGeneric, GravityError,
    PackedGravityCoeffs, MAX_ORDER, MAX_RECURSIVE_ORDER,
};

/// Bit-level finite check (safe under fast-math)
#[inline]
#[must_use]
pub const fn safe_isfinite(x: f64) -> bool {
    x.to_bits() & 0x7FF0_0000_0000_0000 != 0x7FF0_0000_0000_0000
}

// The former crate-local `wide_consts!` (const f64x4 items instead of inline
// splats — the memset_pattern16 rationale lives on the macro's doc) now comes
// from common_rs; the body names `wide::f64x4`, resolved here at the call
// site, so this import is the only wiring.
pub(crate) use common_rs::wide_consts;

// Match C++ constants and evaluation order as closely as possible.
pub const SEC_PER_DAY: f64 = 86400.0; // seconds per day
pub const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
/// Earth rotation rate in rad/s, as the IERS sidereal value.
///
/// A SECOND rate exists on the production path and is not this one:
/// `nd_pipeline::constellation_design::EARTH_ROTATION_RAD_S` is
/// `2*PI / 86164.0905`, i.e. 7.292_115_858e-5, which differs from this constant
/// by 1.176e-7 relative. Neither is wrong — they are the same quantity written
/// to different precision — and they are NOT to be unified: that constant's file
/// is byte-hashed into `part_a_family_source_sha256`, so editing it at all
/// invalidates every sealed receipt.
///
/// The disagreement is bounded and was measured: over a repeat-ground-track
/// design it moves the ground trace by about 0.55 m, against a 50 km snap
/// tolerance. It is five orders under the thing it feeds.
pub const EARTH_OMEGA: f64 = 7.292_115_0e-5;
pub const GMST_SEC2RAD: f64 = (2.0 * std::f64::consts::PI) / 86400.0;
pub const GMST_LINEAR: f64 = 8_640_184.812_866 + 876_600.0 * 3600.0;
pub const GMST_CONST: f64 = 67310.54841;
pub const MU: f64 = 398_600.441_5; // km^3/s^2
pub const INV_MU: f64 = 1.0 / 398_600.441_5;
/// WGS84 equatorial radius, in km.
///
/// Geometric: this is the figure of the Earth
/// used for altitude, ground distance, and shadow geometry, and for the secular
/// J2 model below, whose `J2` is quoted against the same WGS84 figure.
///
/// This is NOT the gravity model's reference radius. See
/// [`GRAVITY_REFERENCE_RADIUS_KM`].
pub const RE: f64 = 6378.137; // km (WGS84)
/// DIR-R6 gravity model reference radius, in km. A model parameter, not a
/// figure of the Earth: the spherical-harmonic coefficients `C_lm`/`S_lm` are
/// normalised against exactly this radius, so it must be used verbatim wherever
/// the harmonic series is evaluated or the coefficients no longer mean what the
/// model publisher fitted.
///
/// It sits 0.00054 km (54 cm) from [`RE`] and the two are deliberately NOT
/// unified — they answer different questions. Swapping either for the other is
/// silently wrong rather than loudly wrong. The bit pattern is pinned by
/// `gravity_reference_radius_contract_tests`, which also asserts `gravity.rs`
/// never reaches for `RE`.
pub(crate) const GRAVITY_REFERENCE_RADIUS_KM: f64 = 6378.13646;
/// WGS84 flattening, dimensionless.
///
/// Pairs with [`RE`] as the semi-major axis,
/// NOT with [`GRAVITY_REFERENCE_RADIUS_KM`] — the ellipsoid is a figure of the
/// Earth and the gravity reference radius is a model parameter.
///
/// The polar radius it implies, `RE * (1 - WGS84_FLATTENING)` = 6356.752 km,
/// sits 21.385 km below `RE`. That is not a rounding difference: it is exactly
/// the error the ground-impact guard carried at the poles while it compared
/// against a spherical `RE`, which could declare impact for a vehicle still
/// 21 km up.
pub const WGS84_FLATTENING: f64 = 1.0 / 298.257_223_563;
pub const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;
pub const J2: f64 = 1.082_626_68e-3; // dimensionless
pub const TOL: f64 = 1e-12;
pub const MAXITER: usize = 500;
/// Floor on `|h|`, the specific angular momentum, in km^2/s.
///
/// Guards the divisor of `acos(h_z / |h|)` when the state is degenerate.
///
/// # The six degeneracy floors, and why they are six
///
/// This constant and the five below it are the degeneracy floors of
/// `eci2kep_impl` and of the equinoctial plane reconstruction. All six hold
/// `1e-9`, and that shared magnitude is a HISTORY, not a physical quantity:
/// one constant used to floor six quantities carrying four different units --
/// specific angular momentum (km^2/s), the node vector (km^2/s), a cross
/// product of the two (km^4/s^2), a position (km), an eccentricity, and a
/// `tan(i/2)` (the last two dimensionless but unrelated). They are written out
/// separately, each with its own literal, so that moving one does not move the
/// other five.
///
/// ## The hazard the shared magnitude hides
///
/// `|n| = |h| sin i`, so testing `|n| > 1e-9` is testing
/// `sin i > 1e-9 / |h|` -- an INCLINATION threshold that moves with the orbit's
/// SIZE. For a LEO with `|h| ~ 5.2e4 km^2/s` the equatorial band is
/// `sin i < 1.9e-14`; for an arc with twice the angular momentum it is half
/// that. So whether a fixture is "equatorial enough" to take the degenerate
/// branch depends on its semi-major axis, not only on its inclination, and a
/// fixture that exercises the branch at one altitude may not at another.
/// [`TAN_HALF_INCLINATION_FLOOR`] does not share that scaling, so the two
/// "equatorial" branches in this workspace do not agree about what equatorial
/// means.
///
/// ## Why they still all hold the same number
///
/// Each of these floors is a DISCONTINUITY SWITCH, not a rounding tolerance:
/// crossing one changes which formula runs. Giving any of them a
/// physically-derived value of its own would flip branches for orbits near the
/// threshold, move flown bits, and re-baseline the sealed pins -- a science
/// decision with a campaign re-run attached, not a cleanup. Separating them
/// makes that decision statable per quantity, which it was not while one
/// constant stood for all of them. Note what that requires: a floor may name
/// exactly ONE quantity. A name like "the argument-of-latitude floor" describes
/// a ROLE, and a role can quietly acquire a second quantity in a second unit --
/// which is how `|h x n|` in km^4/s^2 and `|r|` in km came to share one test.
pub const ANGULAR_MOMENTUM_FLOOR_KM2_PER_S: f64 = 1e-9;

/// Floor on `|n|`, the node vector norm, in km^2/s.
///
/// This is the EQUATORIAL SWITCH: below it the node line is treated as
/// undefined and RAAN is pinned to zero. As an inclination test it is scaled by
/// `1/|h|` -- see [`ANGULAR_MOMENTUM_FLOOR_KM2_PER_S`].
pub const NODE_VECTOR_FLOOR_KM2_PER_S: f64 = 1e-9;

/// Floor on `|h x n|`, the node-cross-momentum vector norm, in km^4/s^2.
///
/// One of the two divisors of the argument-of-latitude quotient
/// `(w . r) / (|w| |r|)`. The other is [`POSITION_NORM_FLOOR_KM`], which is a
/// LENGTH -- the two are floored separately because they are not the same
/// quantity and share no unit.
pub const NODE_CROSS_MOMENTUM_FLOOR_KM4_PER_S2: f64 = 1e-9;

/// Floor on `|r|`, the position vector norm, in km.
///
/// The second divisor of the argument-of-latitude quotient; see
/// [`NODE_CROSS_MOMENTUM_FLOOR_KM4_PER_S2`]. It is a THRESHOLD, not a zero
/// test -- `|r| <= 1e-9` km is enough to take the degenerate arm -- so code
/// that reads it as "only when `|r|` is exactly zero" would be wrong to
/// replace it with a comparison against zero.
///
/// # It does not decide the branch on its own
///
/// This floor is consulted only inside the near-circular, non-equatorial arm:
/// `e < CIRCULAR_E_TOL` and `|n| >= NODE_VECTOR_FLOOR_KM2_PER_S` both have to
/// hold before control reaches it at all. There it is one conjunct of two --
/// `|w| > w_floor && |r| > r_floor` -- so EITHER quantity falling to its floor
/// takes the degenerate arm, and this constant alone determines nothing.
///
/// # Why neither conjunct trips on a physical state
///
/// As a length the magnitude is hard to defend, and the unit conversion is
/// why: `1e-9` km is `1e-6` m, one MICROMETRE, against orbital radii of order
/// 7e3 km -- a factor of 7e12. The other conjunct is bounded for a less
/// obvious reason. `node_vector` is built as `[-h_y, h_x, 0]`, which is
/// `z_hat x h`, so `n` is exactly perpendicular to `h` and
/// `|w| = |h x n| = |h| |n|`. The enclosing arm already guarantees
/// `|n| >= 1e-9`, so at a LEO `|h|` of 5.28e4 km^2/s the smallest `|w|` that
/// can reach the test is 5.28e-5 km^4/s^2 -- a factor of 5.28e4 above its own
/// floor. So on physical input the computed arm is the one taken, and these
/// floors select between formulas only for synthetic states.
///
/// It is also not what makes a near-zero position safe, and should not be
/// relied on for that. `position_norm` is already an unguarded divisor twice
/// above, in `radial_velocity` and in the `-mu / position_norm` term of the
/// specific energy. At `|r|` exactly zero both are non-finite -- energy is
/// -inf and radial velocity is NaN -- before this floor is ever consulted.
/// Through the rest of the band the two stay finite and are merely absurd:
/// at `|r| = 1e-9` km the specific energy is about -3.99e14 km^2/s^2 against
/// -28.8 for a 7000 km orbit. Either way the state is already meaningless by
/// the time the argument of latitude is reconstructed, so this floor changes
/// which formula runs, not whether the answer is usable.
///
/// The identical number against a km^4/s^2 quantity is a completely different
/// test, which is why the two no longer share a constant.
pub const POSITION_NORM_FLOOR_KM: f64 = 1e-9;

/// Floor on the eccentricity magnitude, dimensionless.
///
/// Used only to keep `e` out of a denominator; it is not the circular-orbit
/// test, which is [`CIRCULAR_E_TOL`].
pub const ECCENTRICITY_DIVISOR_FLOOR: f64 = 1e-9;

/// Floor on `tan(i/2)` from the equinoctial pair `(p, q)`, dimensionless.
///
/// Below it the orbit is treated as equatorial and both inclination and RAAN
/// are pinned to zero. Unlike [`NODE_VECTOR_FLOOR_KM2_PER_S`] this one really
/// IS an inclination test with no `|h|` scaling in it -- `tan(i/2) <= 1e-9`
/// means `i <= 2e-9 rad` for any orbit size.
pub const TAN_HALF_INCLINATION_FLOOR: f64 = 1e-9;

#[cfg(test)]
mod gravity_reference_radius_contract_tests {
    use super::GRAVITY_REFERENCE_RADIUS_KM;

    const EXPECTED_DIR_R6_RADIUS_BITS: u64 = 0x40b8_ea22_ef0a_e536;

    macro_rules! ensure {
        ($condition:expr, $message:literal $(, $argument:expr)* $(,)?) => {{
            if !$condition {
                return Err(anyhow::anyhow!($message $(, $argument)*));
            }
        }};
    }

    fn count(source: &str, needle: &str) -> usize {
        source.match_indices(needle).count()
    }

    fn compact_code(source: &str) -> String {
        source
            .lines()
            .filter_map(|line| line.split("//").next())
            .flat_map(str::chars)
            .filter(|character| !character.is_whitespace())
            .collect()
    }

    fn source_between<'a>(
        source: &'a str,
        start_marker: &str,
        end_marker: &str,
    ) -> anyhow::Result<&'a str> {
        let start = source
            .find(start_marker)
            .ok_or_else(|| anyhow::anyhow!("missing gravity function marker {start_marker}"))?;
        let remaining_source = source
            .get(start..)
            .ok_or_else(|| anyhow::anyhow!("invalid gravity marker offset for {start_marker}"))?;
        let offset = remaining_source
            .find(end_marker)
            .ok_or_else(|| anyhow::anyhow!("missing gravity function marker {end_marker}"))?;
        let end = start
            .checked_add(offset)
            .ok_or_else(|| anyhow::anyhow!("gravity marker offset overflow"))?;
        source
            .get(start..end)
            .ok_or_else(|| anyhow::anyhow!("invalid gravity function range for {start_marker}"))
    }

    fn function_header<'a>(
        function_source: &'a str,
        start_marker: &str,
    ) -> anyhow::Result<&'a str> {
        let header_end = function_source
            .find(") ->")
            .ok_or_else(|| anyhow::anyhow!("missing gravity header terminator {start_marker}"))?;
        function_source
            .get(..header_end)
            .ok_or_else(|| anyhow::anyhow!("invalid gravity header {start_marker}"))
    }

    fn validate_rust_source(rust_source: &str) -> anyhow::Result<()> {
        ensure!(
            rust_source.contains("GRAVITY_REFERENCE_RADIUS_KM"),
            "Rust gravity kernels must name the DIR-R6 radius"
        );
        let rust_code_uses_wgs84_radius = rust_source
            .lines()
            .filter_map(|line| line.split("//").next())
            .flat_map(|line| {
                line.split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            })
            .any(|token| token == "RE");
        ensure!(
            !rust_code_uses_wgs84_radius,
            "Rust gravity kernels must not consume the rounded WGS84 radius"
        );
        ensure!(
            !rust_source.contains("6378.137"),
            "Rust gravity kernels must not hard-code the rounded WGS84 radius"
        );
        validate_raw_oracles(rust_source)?;
        validate_packed_evaluators(rust_source)?;
        validate_fixed_state_evaluator_contracts(rust_source)
    }

    fn validate_raw_oracles(rust_source: &str) -> anyhow::Result<()> {
        let raw_oracles = [
            (
                "pub fn spherical_gravity_impl_generic<T",
                "pub fn spherical_gravity_impl(",
                1,
                "gravity_summation_generic_raw(",
            ),
            (
                "pub fn spherical_gravity_impl(",
                "pub fn spherical_gravity_impl_sincos(",
                0,
                "spherical_gravity_impl_sincos(",
            ),
            (
                "pub fn spherical_gravity_impl_sincos(",
                "pub fn spherical_gravity_impl_generic_packed<T",
                1,
                "gravity_summation_f64_raw(",
            ),
            (
                "pub fn spherical_gravity_impl_frame(",
                "pub fn spherical_gravity_impl_frame_packed(",
                1,
                "gravity_summation_f64_raw(",
            ),
        ];
        for (start_marker, end_marker, radius_bindings, required_route) in raw_oracles {
            let function_source = source_between(rust_source, start_marker, end_marker)?;
            let header = function_header(function_source, start_marker)?;
            for raw_parameter in [
                "order: usize",
                "c_coeffs: &[f64]",
                "s_coeffs: &[f64]",
                "stride: usize",
            ] {
                ensure!(
                    header.contains(raw_parameter),
                    "{start_marker} must retain raw oracle parameter {raw_parameter}"
                );
            }
            ensure!(
                !function_source.contains("PackedGravityCoeffs"),
                "{start_marker} must not consume packed coefficients"
            );
            ensure!(
                !function_source.contains("pack_gravity_coeffs("),
                "{start_marker} must not construct packed coefficients"
            );
            ensure!(
                count(function_source, "GRAVITY_REFERENCE_RADIUS_KM") == radius_bindings,
                "{start_marker} must have the expected raw-radius bindings"
            );
            ensure!(
                function_source.contains(required_route),
                "{start_marker} must retain raw oracle route {required_route}"
            );
        }
        Ok(())
    }

    fn packed_formula_fragments(index: usize) -> anyhow::Result<&'static [&'static str]> {
        match index {
            0 => Ok(&[
                "letre_t=T::from_f64(GRAVITY_REFERENCE_RADIUS_KM).ok_or(GravityError::InvariantViolation)?;",
                "letmu_t=T::from_f64(MU).ok_or(GravityError::InvariantViolation)?;",
                "letc2=re_t/(r*r);",
                "letcoef_sph=mu_t/(re_t*re_t);",
                "letc2_re=c2*re_t;",
            ]),
            1 => Ok(&[
                "letre=GRAVITY_REFERENCE_RADIUS_KM;",
                "letc2=re/(r*r);",
                "letcoef_sph=MU/(re*re);",
                "letc2_re=c2*re;",
            ]),
            2 => Ok(&[
                "letre=GRAVITY_REFERENCE_RADIUS_KM;",
                "letr2=pos_x*pos_x+pos_y*pos_y+pos_z*pos_z;",
                "letc2=re/r2;",
                "letcoef_sph=MU/(re*re);",
                "letc2_re=c2*re;",
            ]),
            3 => Ok(&[
                "letradius_km=GRAVITY_REFERENCE_RADIUS_KM;",
                "letc2=radius_km/(r*r);",
                "letcoef_sph=MU/(radius_km*radius_km);",
                "letc2_re=c2*radius_km;",
            ]),
            _ => Err(anyhow::anyhow!("unexpected packed gravity evaluator")),
        }
    }

    fn validate_packed_evaluators(rust_source: &str) -> anyhow::Result<()> {
        let packed_evaluators = [
            (
                "pub fn spherical_gravity_impl_generic_packed<T",
                "pub fn spherical_gravity_impl_packed(",
                "gravity_summation_generic_packed(",
            ),
            (
                "pub fn spherical_gravity_impl_packed(",
                "pub fn spherical_gravity_impl_sincos_packed(",
                "gravity_summation_f64_packed(",
            ),
            (
                "pub fn spherical_gravity_impl_sincos_packed(",
                "/// Compute spherical harmonic gravity using thread-local cache.",
                "gravity_summation_f64_packed(",
            ),
            (
                "pub fn spherical_gravity_impl_frame_packed(",
                "#[cfg(test)]",
                "gravity_summation_f64_packed(",
            ),
        ];
        for (index, (start_marker, end_marker, required_route)) in
            packed_evaluators.into_iter().enumerate()
        {
            let function_source = source_between(rust_source, start_marker, end_marker)?;
            let header = function_header(function_source, start_marker)?;
            ensure!(
                header.contains("packed: &PackedGravityCoeffs"),
                "{start_marker} must require validated packed coefficients"
            );
            for raw_parameter in [
                "order: usize",
                "c_coeffs: &[f64]",
                "s_coeffs: &[f64]",
                "stride: usize",
            ] {
                ensure!(
                    !header.contains(raw_parameter),
                    "{start_marker} must not expose raw parameter {raw_parameter}"
                );
            }
            ensure!(
                !function_source.contains("gravity_summation_generic_raw("),
                "{start_marker} must not route through raw generic summation"
            );
            ensure!(
                !function_source.contains("gravity_summation_f64_raw("),
                "{start_marker} must not route through raw f64 summation"
            );
            ensure!(
                !function_source.contains("pack_gravity_coeffs("),
                "{start_marker} must not repack coefficient storage"
            );
            ensure!(
                count(function_source, "GRAVITY_REFERENCE_RADIUS_KM") == 1,
                "{start_marker} must bind exactly one local DIR-R6 radius"
            );
            ensure!(
                function_source.contains(required_route),
                "{start_marker} must retain packed route {required_route}"
            );
            let compact = compact_code(function_source);
            let required_fragments = packed_formula_fragments(index)?;
            for fragment in required_fragments {
                ensure!(
                    compact.contains(fragment),
                    "{start_marker} must contain formula fragment {fragment}"
                );
            }
        }
        Ok(())
    }

    fn validate_fixed_state_evaluator_contracts(rust_source: &str) -> anyhow::Result<()> {
        let state_evaluators = [
            (
                "pub fn spherical_gravity_impl_generic<T",
                "pub fn spherical_gravity_impl(",
                "state_eci: &[T; 6]",
            ),
            (
                "pub fn spherical_gravity_impl(",
                "pub fn spherical_gravity_impl_sincos(",
                "state_eci: &[f64; 6]",
            ),
            (
                "pub fn spherical_gravity_impl_sincos(",
                "pub fn spherical_gravity_impl_generic_packed<T",
                "state_eci: &[f64; 6]",
            ),
            (
                "pub fn spherical_gravity_impl_generic_packed<T",
                "pub fn spherical_gravity_impl_packed(",
                "state_eci: &[T; 6]",
            ),
            (
                "pub fn spherical_gravity_impl_packed(",
                "pub fn spherical_gravity_impl_sincos_packed(",
                "state_eci: &[f64; 6]",
            ),
            (
                "pub fn spherical_gravity_impl_sincos_packed(",
                "/// Compute spherical harmonic gravity using thread-local cache.",
                "state_eci: &[f64; 6]",
            ),
            (
                "pub fn spherical_gravity(\n",
                "/// Compute spherical harmonic gravity with packed coefficients using thread-local cache.",
                "state_eci: &[f64; 6]",
            ),
            (
                "pub fn spherical_gravity_packed(\n",
                "// ---------------------------------------------------------------------------\n// Task 5B-2",
                "state_eci: &[f64; 6]",
            ),
        ];
        for (start_marker, end_marker, required_state) in state_evaluators {
            let function_source = source_between(rust_source, start_marker, end_marker)?;
            let header = function_header(function_source, start_marker)?;
            ensure!(
                header.contains(required_state),
                "{start_marker} must require exactly six state elements"
            );
        }

        for public_test_metadata in [
            "pub const fn is_empty",
            "pub const fn dense_prefix",
            "pub fn row_len",
            "pub fn rows_are_dense_inclusive",
            "pub fn with_gravity_cache",
        ] {
            ensure!(
                !rust_source.contains(public_test_metadata),
                "gravity must not expose test-only helper {public_test_metadata}"
            );
        }

        let production_source = rust_source
            .split("#[cfg(test)]")
            .next()
            .ok_or_else(|| anyhow::anyhow!("gravity production source must exist"))?;
        for fixed_array_index in ["state_ecef[", "acc_eci[", "position[", "cached["] {
            ensure!(
                !production_source.contains(fixed_array_index),
                "gravity production must destructure fixed array {fixed_array_index}"
            );
        }
        ensure!(
            !production_source.contains("GRAVITY_REUSE_VW"),
            "gravity must not carry a stale always-true reuse gate"
        );
        Ok(())
    }

    fn validate_cuda_source(cuda_source: &str) -> anyhow::Result<()> {
        ensure!(
            !cuda_source.contains("GPU_GRAVITY_REFERENCE_RADIUS_BITS"),
            "CUDA must not carry an unused radius-bits sentinel"
        );
        let cuda_literal_prefix = "GPU_GRAVITY_REFERENCE_RADIUS_KM = ";
        let cuda_literal = cuda_source
            .split_once(cuda_literal_prefix)
            .and_then(|(_, suffix)| suffix.split_once(';'))
            .map(|(literal, _)| literal.trim())
            .ok_or_else(|| anyhow::anyhow!("CUDA gravity radius literal must exist"))?
            .parse::<f64>()
            .map_err(|error| {
                anyhow::anyhow!("CUDA gravity radius literal must parse as f64: {error}")
            })?;
        ensure!(
            cuda_literal.to_bits() == EXPECTED_DIR_R6_RADIUS_BITS,
            "CUDA gravity radius literal must equal exact DIR-R6 binary64 bits"
        );
        ensure!(
            !cuda_source.contains("GPU_RE"),
            "CUDA gravity must not consume the rounded WGS84 radius"
        );
        ensure!(
            !cuda_source.contains("6378.137"),
            "CUDA gravity must not hard-code the rounded WGS84 radius"
        );
        let cuda_kernel = cuda_source
            .split_once("__global__ void\nspherical_gravity_batch_dense")
            .map(|(_, body)| body)
            .ok_or_else(|| anyhow::anyhow!("CUDA gravity kernel must exist"))?;
        ensure!(
            count(cuda_kernel, "GPU_GRAVITY_REFERENCE_RADIUS_KM") == 1,
            "CUDA gravity kernel must bind exactly one local DIR-R6 radius"
        );
        // Task 5B-2: the kernel must not reconstruct a rotation from an
        // Earth-rotation angle. It receives the assembled 9-double GCRS->ITRS
        // matrix from the host, resolved by the sealed frame authority. A scalar
        // angle cannot express bias-precession-nutation, the equation of the
        // origins or polar motion, so any reappearance of `gmst` here means the
        // GPU path has silently diverged from the CPU one by ~31.6 km at 7000 km.
        ensure!(
            !cuda_source.to_lowercase().contains("gmst"),
            "CUDA gravity must not name GMST: the rotation is shipped assembled"
        );
        ensure!(
            cuda_kernel.contains("rot_gcrs_to_itrs"),
            "CUDA gravity kernel must take the assembled GCRS->ITRS rotation"
        );
        let compact_cuda_kernel = compact_code(cuda_kernel);
        for fragment in [
            "constdoublere=GPU_GRAVITY_REFERENCE_RADIUS_KM;",
            "constdoublec2=re/r2;",
            "constdoublec2_re=c2*re;",
            "constdoublecoef_sph=GPU_MU/(re*re);",
        ] {
            ensure!(
                compact_cuda_kernel.contains(fragment),
                "CUDA gravity kernel must contain formula fragment {fragment}"
            );
        }
        Ok(())
    }

    #[test]
    fn gravity_kernels_bind_dir_r6_reference_radius_bits() -> anyhow::Result<()> {
        ensure!(
            GRAVITY_REFERENCE_RADIUS_KM.to_bits() == EXPECTED_DIR_R6_RADIUS_BITS,
            "Rust gravity radius must equal exact DIR-R6 binary64 bits"
        );
        validate_rust_source(include_str!("gravity.rs"))?;
        validate_cuda_source(include_str!("../kernels/cuda/gravity.cu"))
    }
}

// Higher threshold for near-circular orbit handling to avoid omega pi-ambiguity
// Orbits with e < 5e-2 have numerically unstable omega; treat as circular
// Treat only *extremely* small eccentricities as circular when computing angles.
//
// Why: for near-circular orbits the classical elements (argp, nu) become ill-conditioned,
// but using too-large a "circular" threshold (e.g. 0.05) causes discontinuities and breaks
// parity with the C++ reference implementation (and downstream propagation parity tests).
pub const CIRCULAR_E_TOL: f64 = 1e-9;

/// `tan(i/2)` at or below which an equinoctial state is treated as equatorial,
/// so that the ascending node — and with it the split of the longitude of
/// periapsis into (raan, argp) — is undefined.
///
/// Why a threshold at all: the retrograde-free equinoctial elements carry
/// `p = tan(i/2) sin(raan)` and `q = tan(i/2) cos(raan)`, so at `i = 0` both are
/// *exactly* zero and `equinoc2kep_impl`'s recovery `atan2(h q - k p, k q + h p)`
/// degenerates to `atan2(+-0.0, +-0.0)` — a value decided entirely by the sign
/// bits of the zeros, not by the state. `raan = atan2(p, q)` degenerates the
/// same way. The information is not lost, only mislabelled: `(h, k)` still carry
/// the longitude of periapsis `atan2(h, k) = raan + argp`, which is the only
/// combination an equatorial orbit defines.
///
/// Why this value: `1e-12` is `i <= 2.3e-12` rad, so relabelling
/// `(raan, argp) -> (0, raan + argp)` inside the band displaces no position by
/// more than `a * i` — about 14 nm at `a = 7000` km, four orders below the 9e-8
/// km the `i = 1e-9` deg round trip already achieves. That case has
/// `tan(i/2) = 8.7e-12` and stays on the general branch, so the band is entered
/// only where the general branch was already returning sign-bit noise.
pub const EQUATORIAL_PQ_TOL: f64 = 1e-12;

const KEPLER_SMALL_E: f64 = 1e-3;
const KEPLER_GUESS_E: f64 = 0.8;
#[cfg(feature = "parallel")]
const PROP_BATCH_MIN_LEN: usize = 64;

/// Batch length at or above which propagation goes parallel.
#[cfg(feature = "parallel")]
const PROP_BATCH_THRESHOLD: usize = 512;

/// Batch length at or above which orbital-parameter extraction goes parallel.
#[cfg(feature = "parallel")]
const ORBITAL_PARAMS_PAR_THRESHOLD: usize = 256;

// Wide-literal vectors for the SIMD kernels below. Each mirrors the scalar
// spelling it replaced (the `1.0 - 1e-12` and `10.0 * TOL` expressions
// const-evaluate to the identical binary64 the runtime expression produced).
wide_consts! {
    HALF_X4 = 0.5,
    SIXTH_X4 = 1.0 / 6.0,
    TWO_X4 = 2.0,
    THREE_X4 = 3.0,
    FIVE_X4 = 5.0,
    THREE_HALVES_X4 = 1.5,
    THREE_QUARTERS_X4 = 0.75,
    MU_X4 = MU,
    RE_X4 = RE,
    J2_X4 = J2,
    TOL_X4 = TOL,
    TEN_TOL_X4 = 10.0 * TOL,
    NAN_X4 = f64::NAN,
    INF_X4 = f64::INFINITY,
    TAU_X4 = std::f64::consts::TAU,
    TWO_PI_X4 = TWO_PI,
    KEPLER_STEP_TOL_X4 = 1e-12,
    KEPLER_GUESS_E_X4 = KEPLER_GUESS_E,
    E_CLAMP_MAX_X4 = 1.0 - 1e-12,
}

/// Convert a finite scalar for a generic orbital kernel.
///
/// `Float` implementations used by the kernels represent finite binary64
/// constants. A non-representable constant produces `NaN`, so callers keep the
/// existing numerical invalid-state signal instead of panicking.
#[inline]
#[must_use]
fn scalar<T: Float + FromPrimitive>(value: f64) -> T {
    T::from_f64(value).unwrap_or_else(T::nan)
}

#[inline]
#[must_use]
fn six_values<T: Copy>(values: &[T]) -> Option<[T; 6]> {
    <&[T; 6]>::try_from(values.get(..6)?).ok().copied()
}

#[inline]
fn write_six<T>(values: &mut [T], replacement: [T; 6]) {
    if let Some(target) = values.get_mut(..6) {
        for (destination, source) in target.iter_mut().zip(replacement) {
            *destination = source;
        }
    }
}

#[inline]
fn write_nan_six<T: Float>(values: &mut [T]) {
    for value in values.iter_mut().take(6) {
        *value = T::nan();
    }
}

#[inline]
fn write_nan_first<T: Float>(values: &mut [T]) {
    if let Some(value) = values.first_mut() {
        *value = T::nan();
    }
}

// `rayon_min_len` used to live here: `RAYON_MIN_LEN_OVERRIDE.unwrap_or(default)`
// where the override was `Lazy::new(|| None)`. It was an identity function
// wrapping a knob that no longer exists, so its two call sites now pass
// PROP_BATCH_MIN_LEN straight to `with_min_len`.

/// A/B arm selector for the glibc argument-reduction lever (M1, `[-π, π)` wrap).
///
/// glibc's `sin` takes a slower reduction branch once `|x| > 2.426265` and
/// costs +34% there (`docs/PMU_PROFILE.md` §7). Every angle this crate wraps
/// lands in `[0, 2π)`, which puts a large fraction of the arc's `sin`/`cos`
/// calls past that threshold for no reason: `[-π, π)` is the same angle and a
/// cheaper libm call. Set by `sed` between two builds of the arc driver, never
/// committed `true` — it is a BIT-MOVER (`x − 2π` is a different representable
/// `f64`, so `sin` differs at ULP scale), so landing it needs a re-pin ledger.
///
/// Companion const: `jb_rs::jb2008`'s `WRAP_TO_SIGNED_PI`. Flip both together.
///
/// # MEASURED AND CLOSED, 2026-08-11, on production silicon
///
/// **It pays about 0.35% of arc and that is not enough to land it.** Measured
/// in situ on the `TinkerCliffs` nodes `tc082`/`tc294`/`tc197`/`tc183` (EPYC 7702, glibc),
/// sealed release flags, `ND_BLOCK_PROPS=1 ND_BLOCKS=120`, against a
/// byte-identical control in the same rounds: **−0.31%, −0.32% and −0.38%**
/// across three independently-controlled schedules. R55's interposer-derived
/// −0.442% prediction is therefore about right in magnitude and was never the
/// reason to stop.
///
/// The reason to stop is the bill. The arm moves bits (the V3 arc's position
/// moves ~4e-9 km and every pin with it), it moves the eval count 6742 → 6752,
/// and it costs 1.5% of the truncation-error budget — four reseals for an
/// effect that no single run separated from the instrument's own arm-to-arm
/// bias. `docs/PMU_PROFILE.md` §10.6 carries the full ledger.
///
/// Left compiled-dead rather than deleted, exactly as `jb2008::CEILING_PROBE`
/// is, so that anyone re-opening the lever measures it instead of rebuilding it.
const WRAP_TO_SIGNED_PI: bool = false;

/// [`mod2pi`], re-targeted to `[-π, π)`.
///
/// Deliberately built ON TOP of `mod2pi` rather than replacing its `%`: the
/// arm must change the wrap TARGET and nothing else. Spelling this as a
/// call-free `x − τ·⌊x/τ + ½⌋` would also delete a `compiler_builtins` `fmod`
/// and confound the two effects.
#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "angle wrapping is generic floating-point arithmetic with explicit finite-state handling"
)]
fn wrap_to_signed_pi<T: Float + FromPrimitive>(angle: T) -> T {
    let x = mod2pi(angle);
    if x >= scalar::<T>(std::f64::consts::PI) {
        x - scalar::<T>(TWO_PI)
    } else {
        x
    }
}

/// The equinoctial longitude wrap, under whichever arm is built.
///
/// Shift-invariant at both consumers: the Halley residual is
/// `true_anomaly + a_term − longitude` and the tail reads only
/// `longitude − true_anomaly`, so subtracting `τ` from `longitude` subtracts it
/// from `true_anomaly` too and every published quantity is unchanged
/// mathematically.
#[inline]
fn wrap_equinoctial_longitude<T: Float + FromPrimitive>(angle: T) -> T {
    if WRAP_TO_SIGNED_PI {
        wrap_to_signed_pi(angle)
    } else {
        mod2pi(angle)
    }
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "angle wrapping is generic floating-point arithmetic with explicit finite-state handling"
)]
pub fn mod2pi<T: Float + FromPrimitive>(angle: T) -> T {
    if !angle.is_finite() {
        return T::nan();
    }
    let two_pi = scalar::<T>(TWO_PI);
    let mut x = angle % two_pi;
    if x < T::zero() {
        x = x + two_pi;
    }
    if x >= two_pi {
        x = x - two_pi;
    }
    if x.is_zero() {
        T::zero()
    } else {
        x
    }
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "eccentricity clamp retains its established generic floating-point calculation"
)]
pub fn clamp_eccentricity<T: Float + FromPrimitive>(e: T) -> T {
    if !e.is_finite() || e <= T::zero() {
        return T::zero();
    }
    let eps = scalar::<T>(10.0 * TOL);
    if e >= T::one() {
        let capped = T::one() - eps;
        if capped > T::zero() {
            capped
        } else {
            T::zero()
        }
    } else {
        e
    }
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "three-vector norm preserves its FMA and final multiplication order"
)]
pub fn norm3<T: Float>(v: &[T; 3]) -> T {
    let &[x, y, z] = v;
    x.mul_add(x, y.mul_add(y, z * z)).sqrt()
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "three-vector dot product uses generic Float IEEE semantics and preserves FMA operation order; no integer arithmetic occurs"
)]
pub fn dot3<T: Float>(a: &[T; 3], b: &[T; 3]) -> T {
    let &[ax, ay, az] = a;
    let &[bx, by, bz] = b;
    ax.mul_add(bx, ay.mul_add(by, az * bz))
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "cross product preserves its FMA and multiplication order"
)]
pub fn cross3<T: Float>(a: &[T; 3], b: &[T; 3]) -> [T; 3] {
    let &[ax, ay, az] = a;
    let &[bx, by, bz] = b;
    [
        ay.mul_add(bz, -(az * by)),
        az.mul_add(bx, -(ax * bz)),
        ax.mul_add(by, -(ay * bx)),
    ]
}

#[inline]
#[must_use]
pub fn greenwichsrt_impl(jd: f64) -> f64 {
    let t_ut1 = (jd - 2_451_545.0) / 36_525.0;
    let gmst_sec = (-6.2e-6_f64)
        .mul_add(t_ut1, 0.093_104)
        .mul_add(t_ut1, GMST_LINEAR)
        .mul_add(t_ut1, GMST_CONST);
    mod2pi(gmst_sec * GMST_SEC2RAD)
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "ECI-to-ECEF transform preserves its generic floating-point operation order"
)]
pub fn eci2ecef_impl_sincos<T: Float + FromPrimitive>(state: &[T], s: T, c: T, out: &mut [T]) {
    let Some([rx, ry, rz, vx, vy, vz]) = six_values(state) else {
        write_nan_six(out);
        return;
    };

    let omega = scalar::<T>(EARTH_OMEGA);

    let rxe = c.mul_add(rx, s * ry);
    let rye = (-s).mul_add(rx, c * ry);
    let rze = rz;

    let vxe = c.mul_add(vx, s.mul_add(vy, omega * rye));
    let vye = (-s).mul_add(vx, c.mul_add(vy, -omega * rxe));
    let vze = vz;

    write_six(out, [rxe, rye, rze, vxe, vye, vze]);
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic Float transform uses IEEE finite/NaN semantics, has no integer-overflow path, and preserves frame-transform operation order; checked integer math is irrelevant"
)]
pub fn ecef2eci_impl_sincos<T: Float + FromPrimitive>(state: &[T], s: T, c: T, out: &mut [T]) {
    let Some([rx, ry, rz, vx, vy, vz]) = six_values(state) else {
        write_nan_six(out);
        return;
    };

    let omega = scalar::<T>(EARTH_OMEGA);

    let vcx = vx - omega * ry;
    let vcy = vy + omega * rx;
    let vcz = vz;

    let rxi = c.mul_add(rx, -s * ry);
    let ryi = s.mul_add(rx, c * ry);
    let rzi = rz;

    let vxi = c.mul_add(vcx, -s * vcy);
    let vyi = s.mul_add(vcx, c * vcy);
    let vzi = vcz;

    write_six(out, [rxi, ryi, rzi, vxi, vyi, vzi]);
}

#[inline]
pub fn eci2ecef_impl(state: &[f64], jd: f64, out: &mut [f64]) {
    let gmst = greenwichsrt_impl(jd);
    let (s, c) = gmst.sin_cos();
    eci2ecef_impl_sincos(state, s, c, out);
}

#[inline]
pub fn ecef2eci_impl(state: &[f64], jd: f64, out: &mut [f64]) {
    let gmst = greenwichsrt_impl(jd);
    let (s, c) = gmst.sin_cos();
    ecef2eci_impl_sincos(state, s, c, out);
}

/// Geocentric spherical (lat, lon, alt) from an ITRS position 3-vector.
///
/// Frame sibling of [`eci_to_geocentric_spherical`]: the caller has already
/// applied the GCRS->ITRS rotation, so no `(sin, cos)` frame pair is needed and
/// the rotation can be the full IAU 2006/2000A chain rather than a z-rotation.
///
/// The longitude this returns is what makes the thermosphere proxy's local time
/// meaningful — under the legacy GMST rotation it was wrong by the omitted
/// precession, nutation and polar-motion terms.
#[inline]
#[must_use]
pub fn geocentric_spherical_from_itrs(pos_itrs: &[f64; 3], earth_radius: f64) -> (f64, f64, f64) {
    let [x, y, z] = *pos_itrs;
    let r = (x * x + y * y + z * z).sqrt();
    let alt_km = r - earth_radius;
    let lat_rad = if r > 1e-10 {
        (z / r).clamp(-1.0, 1.0).asin()
    } else {
        0.0
    };
    let lon_rad = y.atan2(x);
    (lat_rad, lon_rad, alt_km)
}

/// Convert ECI to geocentric spherical (lat, lon, alt) using cached GMST
/// Optimized for spherical Earth - only 2 trig calls (vs 5-7 for ellipsoid)
#[inline]
#[must_use]
pub fn eci_to_geocentric_spherical(
    state: &[f64],
    sin_gmst: f64,
    cos_gmst: f64,
    earth_radius: f64,
) -> (f64, f64, f64) {
    let Some([x_eci, y_eci, z_eci]) = state
        .get(..3)
        .and_then(|values| <&[f64; 3]>::try_from(values).ok())
        .copied()
    else {
        return (f64::NAN, f64::NAN, f64::NAN);
    };

    // ECI to ECEF rotation (using cached GMST)
    let x_ecef = cos_gmst * x_eci + sin_gmst * y_eci;
    let y_ecef = -sin_gmst * x_eci + cos_gmst * y_eci;
    let z_ecef = z_eci;

    // Geocentric spherical (for sphere, geodetic = geocentric)
    let r = (x_ecef * x_ecef + y_ecef * y_ecef + z_ecef * z_ecef).sqrt();
    let alt_km = r - earth_radius;

    let lat_rad = if r > 1e-10 {
        (z_ecef / r).clamp(-1.0, 1.0).asin()
    } else {
        0.0
    };
    let lon_rad = y_ecef.atan2(x_ecef);

    (lat_rad * RAD_TO_DEG, lon_rad * RAD_TO_DEG, alt_km)
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic Float Kepler iteration uses IEEE finite/NaN semantics and preserves established Halley operation order; no integer arithmetic occurs"
)]
fn solve_kepler_e_core_wrapped<T: Float + FromPrimitive>(mm: T, e_safe: T) -> (T, T, T) {
    let (sin_m, cos_m) = mm.sin_cos();
    let tol = scalar::<T>(TOL);
    if e_safe < tol {
        return (mm, sin_m, cos_m);
    }

    let sin_double_mean_anomaly = scalar::<T>(2.0) * sin_m * cos_m;
    let small_e = scalar::<T>(KEPLER_SMALL_E);

    if e_safe < small_e {
        let e2 = e_safe * e_safe;
        let mut e_anom = mm + e_safe * sin_m + scalar::<T>(0.5) * e2 * sin_double_mean_anomaly;
        let (mut s, mut c) = e_anom.sin_cos();

        for _ in 0..2 {
            let f = (-e_safe).mul_add(s, e_anom - mm);
            let fp = (-e_safe).mul_add(c, T::one());
            let fpp = e_safe * s;
            let fppp = e_safe * c;

            let delta1 = -f / fp;
            let delta2 = -f / (fp + scalar::<T>(0.5) * delta1 * fpp);
            let delta3 = -f
                / (fp
                    + scalar::<T>(0.5) * delta2 * fpp
                    + scalar::<T>(1.0 / 6.0) * delta2 * delta2 * fppp);

            e_anom = e_anom + delta3;
            if delta3.abs() < tol {
                break;
            }

            let (s_d, c_d) = delta3.sin_cos();
            let s_new = s.mul_add(c_d, c * s_d);
            c = c.mul_add(c_d, -s * s_d);
            s = s_new;
        }

        let (s, c) = e_anom.sin_cos();
        return (e_anom, s, c);
    }

    let mut e_anom = if e_safe < scalar::<T>(KEPLER_GUESS_E) {
        mm + e_safe * sin_m + scalar::<T>(0.5) * e_safe * e_safe * sin_double_mean_anomaly
    } else if mm >= T::zero() {
        scalar::<T>(std::f64::consts::PI)
    } else {
        scalar::<T>(-std::f64::consts::PI)
    };
    let (mut s, mut c) = e_anom.sin_cos();

    // Whether the loop LEFT by converging, rather than by running out of
    // iterations. Without it the two exits are indistinguishable and the
    // exhausted one returns its last iterate as though it were a root.
    let mut converged = false;
    for _ in 0..MAXITER {
        let f = (-e_safe).mul_add(s, e_anom - mm);
        let fp = (-e_safe).mul_add(c, T::one());
        let fpp = e_safe * s;
        let fppp = e_safe * c;

        let delta1 = -f / fp;
        let delta2 = -f / (fp + scalar::<T>(0.5) * delta1 * fpp);
        let delta3 = -f
            / (fp
                + scalar::<T>(0.5) * delta2 * fpp
                + scalar::<T>(1.0 / 6.0) * delta2 * delta2 * fppp);

        e_anom = e_anom + delta3;
        if delta3.abs() < tol {
            converged = true;
            break;
        }

        let (s_d, c_d) = delta3.sin_cos();
        let s_new = s.mul_add(c_d, c * s_d);
        c = c.mul_add(c_d, -s * s_d);
        s = s_new;
    }

    if !converged {
        // Exhaustion is a NON-ANSWER, so say so rather than return the last
        // iterate as though it were a root. This loop gates on |delta3| alone,
        // and in a runaway the cubic denominator grows like f0^2, so the step
        // falls under tolerance while the iterate is still radians from any
        // root -- the failure this function's own comment records at 747 of
        // 145,440 samples, up to 3.06 rad.
        //
        // Only THIS loop gets the treatment. The small-e branch above runs a
        // fixed `for _ in 0..2` refinement where using both iterations is the
        // design, not a failure, so a non-answer there would be a false one.
        //
        // Costs nothing on the converged path, which is every path measured:
        // the grid test sweeps e to 1 - 1e-12 over 161,280 samples, worst
        // residual 1.8e-15, no exhaustion anywhere. NaN keeps the signature and
        // the hot path unchanged and is the shape lambert_rs:839 already uses.
        let nan = T::nan();
        return (nan, nan, nan);
    }

    (e_anom, s, c)
}

/// SIMD Kepler equation solver: solve for 4 mean anomalies simultaneously.
///
/// Uses Halley's method (3rd order Newton) with masked convergence tracking.
///
/// # Arguments
/// * `mm` - Mean anomalies (4 values packed in f64x4)
/// * `e` - Eccentricities (4 values packed in f64x4)
///
/// # Returns
/// (`sin_E`, `cos_E`, `E`) - Sine, cosine, and eccentric anomaly for all 4 lanes
#[inline]
#[must_use]
pub fn solve_kepler_e_simd(mm: f64x4, e: f64x4) -> (f64x4, f64x4, f64x4) {
    let zero = f64x4::ZERO;
    let one = f64x4::ONE;
    let half = HALF_X4;
    let sixth = SIXTH_X4;
    let tol = KEPLER_STEP_TOL_X4;
    let e_thresh = KEPLER_GUESS_E_X4;
    let pi = f64x4::PI;

    // Clamp eccentricity to [0, 1-eps]
    let e_safe = e.max(zero).min(E_CLAMP_MAX_X4);

    // Wrap mean anomaly to [-π, π] for faster convergence
    // Uses same logic as scalar: M' = M mod 2π, then shift if > π
    let two_pi = TAU_X4;
    let mm_wrapped = mm - (mm / two_pi).floor() * two_pi; // mm mod 2π, range [0, 2π)
    let mm_wrapped = (mm_wrapped.simd_gt(pi)).select(mm_wrapped - two_pi, mm_wrapped); // shift to [-π, π]

    // Initial guess: E0 = M for low e, E0 = sign(M) * pi for high e.
    //
    // The sign matters. `solve_kepler_e_core_wrapped` seeds `+pi` only for
    // `M >= 0` and `-pi` otherwise, and this path must match it: seeding `+pi`
    // against a wrapped `M < 0` starts the iterate nearly 2*pi from the root,
    // and Halley then runs away instead of converging. The runaway is silent,
    // because the cubic denominator grows like f0^2, so `delta3 ~ -6/(f''' f0)`
    // drops below the step tolerance while the iterate is still rad away from
    // any root -- and the loop below gates convergence on the STEP only, with
    // no residual test and no non-convergence channel. That returned E for
    // 747 of 145,440 grid samples with |E - e sin E - M| up to 3.06 rad, all
    // of them at wrapped M < 0 (equivalently M in (pi, 2*pi)) and e >= 0.8019.
    //
    // `simd_ge` rather than `simd_gt` for the same reason: the scalar switches
    // to the pi seed at `!(e < KEPLER_GUESS_E)`, so e == 0.8 exactly belongs to
    // the high-e branch on both paths.
    let high_e_mask = e_safe.simd_ge(e_thresh);
    let seed_pi = mm_wrapped.simd_ge(zero).select(pi, -pi);
    let e0 = high_e_mask.select(seed_pi, mm_wrapped);

    let mut ea = e0;
    let (mut sin_e, mut cos_e) = ea.sin_cos();

    // Convergence mask: starts all-false (0.0), becomes all-true (-1.0 bit pattern) per lane
    let mut converged = zero;

    // Halley's method iteration
    for _ in 0..MAXITER {
        // f(E) = E - e*sin(E) - M
        let f0 = ea - e_safe * sin_e - mm_wrapped;
        // f'(E) = 1 - e*cos(E)
        let f1 = one - e_safe * cos_e;
        // f''(E) = e*sin(E)
        let f2 = e_safe * sin_e;
        // f'''(E) = e*cos(E)
        let f3 = e_safe * cos_e;

        // Halley's 3rd-order correction
        let delta1 = -f0 / f1;
        let delta2 = -f0 / (f1 + half * delta1 * f2);
        let delta3 = -f0 / (f1 + half * delta2 * f2 + sixth * delta2 * delta2 * f3);

        // Update E (only in unconverged lanes)
        let not_converged = converged.simd_eq(zero);
        ea = not_converged.select(ea + delta3, ea);

        // Check convergence
        let conv_mask = delta3.abs().simd_lt(tol);
        converged |= conv_mask;

        // Early exit if all lanes converged
        if converged.all() {
            break;
        }

        // Update sin/cos using angle addition (cheaper than full sin/cos)
        let (s_d, c_d) = delta3.sin_cos();
        let sin_e_new = sin_e * c_d + cos_e * s_d;
        let cos_e_new = cos_e * c_d - sin_e * s_d;
        cos_e = not_converged.select(cos_e_new, cos_e);
        sin_e = not_converged.select(sin_e_new, sin_e);
    }

    // Final sin/cos (ensure accuracy after iterations)
    let (sin_e_final, cos_e_final) = ea.sin_cos();

    // NaN-on-exhaustion, per lane, for the reason given in the scalar twin:
    // a lane that never converged holds its last iterate, which the caller
    // cannot distinguish from a root. `converged` is already tracked per lane
    // for the early exit, so this is one select on a path that never runs.
    let exhausted = converged.simd_eq(zero);
    let nan = f64x4::splat(f64::NAN);
    (
        exhausted.select(nan, sin_e_final),
        exhausted.select(nan, cos_e_final),
        exhausted.select(nan, ea),
    )
}

/// SIMD mean-to-true anomaly conversion.
///
/// Lane-for-lane identical in FORM to [`mean_to_true_anomaly_impl`], which is
/// the point: until 2026-08-20 the two twins were different formulas with
/// different domain policies, so the vectorized path answered where the scalar
/// refused, and disagreed with it near E = pi.
///
/// The old SIMD form was the full-angle half-tangent
/// `nu = 2*atan(sqrt((1+e)/(1-e)) * sin E / (1 + cos E))`. Its denominator
/// `1 + cos E` vanishes at E = pi -- the apoapsis of every orbit -- where it
/// gives 0/0, and it carries no quadrant information, so it cannot distinguish
/// nu from nu +/- 2*pi without the wrap doing the work. The scalar twin has
/// always used the quadrant-safe half-angle form
/// `nu = 2*atan2(sqrt(1+e) * sin(E/2), sqrt(1-e) * cos(E/2))`, which is finite
/// and correct at E = pi because the two arguments never vanish together.
///
/// Domain policy now matches too. The scalar takes |e| and returns NaN for
/// |e| >= 1; this clamped instead, so a hyperbolic eccentricity got a silently
/// fabricated elliptical answer. A parabolic or hyperbolic orbit has no true
/// anomaly on this branch, and saying so is the honest answer.
///
/// # Arguments
/// * `mm` - Mean anomalies (4 values)
/// * `e` - Eccentricities (4 values)
///
/// # Returns
/// True anomalies (4 values), or NaN in any lane with |e| >= 1.
#[inline]
#[must_use]
pub fn mean_to_true_anomaly_simd(mm: f64x4, e: f64x4) -> f64x4 {
    let one = f64x4::ONE;

    // |e|, then reject the non-elliptical lanes -- both as the scalar does.
    let e_abs = e.abs();
    let non_elliptical = e_abs.simd_ge(one);
    let e_safe = e_abs.min(E_CLAMP_MAX_X4);

    let (_sin_e, _cos_e, ea) = solve_kepler_e_simd(mm, e_safe);

    // nu = 2 * atan2(sqrt(1+e) * sin(E/2), sqrt(1-e) * cos(E/2))
    let sqrt_one_plus_eccentricity = (one + e_safe).sqrt();
    let sqrt_one_minus_eccentricity = (one - e_safe).max(f64x4::ZERO).sqrt();
    let (sh_sin, sh_cos) = (ea * f64x4::splat(0.5)).sin_cos();
    let half_nu = (sqrt_one_plus_eccentricity * sh_sin).atan2(sqrt_one_minus_eccentricity * sh_cos);

    let nu = mod2pi_simd(half_nu + half_nu);
    non_elliptical.select(f64x4::splat(f64::NAN), nu)
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic Float Kepler normalization uses IEEE finite/NaN semantics and preserves established operation order; no integer arithmetic occurs"
)]
fn solve_kepler_e_core<T: Float + FromPrimitive>(m: T, e: T) -> (T, T, T) {
    let e_safe = clamp_eccentricity(e.abs());
    let two_pi = scalar::<T>(TWO_PI);
    let pi = scalar::<T>(std::f64::consts::PI);

    let mut mm = m % two_pi;
    if mm > pi {
        mm = mm - two_pi;
    }
    if mm < -pi {
        mm = mm + two_pi;
    }

    solve_kepler_e_core_wrapped(mm, e_safe)
}

#[inline]
pub fn solve_kepler_e<T: Float + FromPrimitive>(m: T, e: T) -> T {
    let (e_anom, _, _) = solve_kepler_e_core(m, e);
    mod2pi(e_anom)
}

#[inline]
pub fn solve_kepler_e_sincos<T: Float + FromPrimitive>(m: T, e: T) -> (T, T) {
    let (_, s, c) = solve_kepler_e_core(m, e);
    (s, c)
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic Float anomaly conversion uses IEEE finite/NaN semantics and preserves established operation order; no integer arithmetic occurs"
)]
pub fn mean_to_true_anomaly_impl<T: Float + FromPrimitive>(m: T, e: T) -> T {
    if !m.is_finite() {
        return T::nan();
    }
    let e_val = e.abs();
    if e_val >= T::one() {
        return T::nan();
    }
    let e_safe = clamp_eccentricity(e_val);
    let e_anom = solve_kepler_e(m, e_safe);
    let sqrt_one_plus_eccentricity = (T::one() + e_safe).sqrt();
    let sqrt_one_minus_eccentricity = (T::one() - e_safe).max(T::zero()).sqrt();
    let sh = scalar::<T>(0.5) * e_anom;
    let (sh_sin, sh_cos) = sh.sin_cos();
    let nu = scalar::<T>(2.0)
        * (sqrt_one_plus_eccentricity * sh_sin).atan2(sqrt_one_minus_eccentricity * sh_cos);
    mod2pi(nu)
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic Float anomaly conversion uses IEEE finite/NaN semantics and preserves established operation order; no integer arithmetic occurs"
)]
pub fn true_to_mean_anomaly_impl<T: Float + FromPrimitive>(nu: T, e: T) -> T {
    if !nu.is_finite() {
        return T::nan();
    }
    let e_val = e.abs();
    if e_val >= T::one() {
        return T::nan();
    }
    let e_safe = clamp_eccentricity(e_val);
    let (sn, cn) = nu.sin_cos();
    let mut denom = e_safe.mul_add(cn, T::one());
    let tol = scalar::<T>(TOL);
    if denom.abs() < tol {
        denom = if denom >= T::zero() { tol } else { -tol };
    }
    let sqrt_one_minus_e2 = e_safe.mul_add(-e_safe, T::one()).max(T::zero()).sqrt();
    let sin_e = (sqrt_one_minus_e2 * sn) / denom;
    let cos_e = (e_safe + cn) / denom;
    let e_anom = sin_e.atan2(cos_e);
    let e_anom_mod = mod2pi(e_anom);
    e_anom_mod - e_safe * e_anom_mod.sin()
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "ECI-to-Kepler conversion retains its established generic floating-point operation order"
)]
pub fn eci2kep_impl<T: Float + FromPrimitive>(
    state: &[T],
    deg: bool,
    true_anom: bool,
    out: &mut [T],
) {
    let Some([position_x, position_y, position_z, velocity_x, velocity_y, velocity_z]) =
        six_values(state)
    else {
        write_nan_six(out);
        return;
    };

    let position_norm_squared = position_x.mul_add(
        position_x,
        position_y.mul_add(position_y, position_z * position_z),
    );
    let position_norm = position_norm_squared.sqrt();
    let velocity_norm_squared = velocity_x.mul_add(
        velocity_x,
        velocity_y.mul_add(velocity_y, velocity_z * velocity_z),
    );
    let radial_velocity = (position_x.mul_add(
        velocity_x,
        position_y.mul_add(velocity_y, position_z * velocity_z),
    )) / position_norm;
    let mu = scalar::<T>(MU);
    let energy = velocity_norm_squared.mul_add(scalar::<T>(0.5), -mu / position_norm);

    let tol = scalar::<T>(TOL);
    let angular_momentum_floor = scalar::<T>(ANGULAR_MOMENTUM_FLOOR_KM2_PER_S);
    let node_floor = scalar::<T>(NODE_VECTOR_FLOOR_KM2_PER_S);
    let node_cross_momentum_floor = scalar::<T>(NODE_CROSS_MOMENTUM_FLOOR_KM4_PER_S2);
    let position_norm_floor = scalar::<T>(POSITION_NORM_FLOOR_KM);
    let eccentricity_divisor_floor = scalar::<T>(ECCENTRICITY_DIVISOR_FLOOR);
    let circular_e_tol = scalar::<T>(CIRCULAR_E_TOL);

    if !matches!(energy.partial_cmp(&-tol), Some(std::cmp::Ordering::Less)) {
        write_nan_six(out);
        return;
    }

    let angular_momentum = [
        position_y.mul_add(velocity_z, -position_z * velocity_y),
        position_z.mul_add(velocity_x, -position_x * velocity_z),
        position_x.mul_add(velocity_y, -position_y * velocity_x),
    ];
    let [angular_x, angular_y, angular_z] = angular_momentum;
    let angular_norm = norm3(&angular_momentum);
    let node_vector = [-angular_y, angular_x, T::zero()];
    let [node_x, node_y, node_z] = node_vector;
    let node_norm = norm3(&node_vector);

    let inclination = (angular_z
        / if angular_norm < angular_momentum_floor {
            angular_momentum_floor
        } else {
            angular_norm
        })
    .clamp(-T::one(), T::one())
    .acos();
    let mut raan = if node_norm > node_floor {
        node_y.atan2(node_x)
    } else {
        T::zero()
    };
    raan = mod2pi(raan);

    let e_vec = {
        let inv_position_norm = T::one() / position_norm;
        let inv_mu = T::one() / mu;
        let factor = velocity_norm_squared - mu * inv_position_norm;
        let radial_velocity_scale = position_norm * radial_velocity;
        [
            (factor.mul_add(position_x, -radial_velocity_scale * velocity_x)) * inv_mu,
            (factor.mul_add(position_y, -radial_velocity_scale * velocity_y)) * inv_mu,
            (factor.mul_add(position_z, -radial_velocity_scale * velocity_z)) * inv_mu,
        ]
    };
    let [eccentricity_x, eccentricity_y, eccentricity_z] = e_vec;
    let e_safe = clamp_eccentricity(norm3(&e_vec));

    let mut argp = T::zero();
    if e_safe > circular_e_tol && node_norm > node_floor {
        let xv = (node_x.mul_add(eccentricity_x, node_y * eccentricity_y)) / (node_norm * e_safe);
        let xv_clip = xv.clamp(-T::one(), T::one());
        argp = xv_clip.acos();
        // Quadrant: sign(sin argp) == sign(e_z) because e_z = |e| sin(argp) sin(i)
        // and sin(i) > 0 for 0 < i < pi. Testing (n x e)_z instead would pick up
        // an extra sign(cos i) through h_z and flip argp for retrograde orbits.
        if eccentricity_z < T::zero() {
            argp = scalar::<T>(TWO_PI) - argp;
        }
    } else if e_safe > circular_e_tol {
        // Equatorial and elliptical. The node vector (-h_y, h_x, 0) has
        // vanished, so raan was pinned to zero above and the branch that needs
        // it cannot run; leaving argp at zero as well used to discard the
        // orientation of the ellipse entirely, rotating the reconstructed
        // position by the true argp (4,828 km at argp = 45 deg on a 7000 km
        // orbit). The orientation survives in the eccentricity vector, which
        // lies in the equatorial plane here, so recover the longitude of
        // periapsis from it directly and let argp carry the whole angle.
        //
        // The y sign is the retrograde correction. Reconstruction applies
        // R3(-raan) R1(-i) R3(-argp), so at i = pi the R1 flips y and the
        // periapsis direction is (cos argp, -sin argp, 0); at i = 0 it is
        // (cos argp, sin argp, 0). Inclination is acos(h_z / |h|), so the
        // branch is exactly sign(h_z). i = pi reaches this branch too: sin(pi)
        // does not round to zero, so node_norm lands near 6e-12 — below
        // NODE_VECTOR_FLOOR_KM2_PER_S, but not at it.
        let periapsis_y = if angular_z < T::zero() {
            -eccentricity_y
        } else {
            eccentricity_y
        };
        argp = periapsis_y.atan2(eccentricity_x);
    }
    argp = mod2pi(argp);

    let nu = if e_safe < circular_e_tol {
        if node_norm < node_floor {
            // Circular equatorial: both the node line and periapsis are
            // undefined, raan and argp are both zero, so nu carries the true
            // longitude measured in the equatorial plane. The y sign is the
            // same retrograde correction as in the argp branch above —
            // reconstruction applies R1(-i), which at i = pi flips y, so
            // without it a retrograde circular equatorial orbit came back
            // mirrored: nu = 10 deg recovered as 350 deg, 2,431 km away.
            let longitude_y = if angular_z < T::zero() {
                -position_y
            } else {
                position_y
            };
            longitude_y.atan2(position_x)
        } else {
            let cos_u_raw = (node_x
                .mul_add(position_x, node_y.mul_add(position_y, node_z * position_z)))
                / (node_norm * position_norm);
            let cos_u = cos_u_raw.clamp(-T::one(), T::one());
            let sin_u = {
                let w_vec = [
                    angular_y.mul_add(node_z, -angular_z * node_y),
                    angular_z.mul_add(node_x, -angular_x * node_z),
                    angular_x.mul_add(node_y, -angular_y * node_x),
                ];
                let [w_x, w_y, w_z] = w_vec;
                let w_norm = norm3(&w_vec);
                if w_norm > node_cross_momentum_floor && position_norm > position_norm_floor {
                    let sin_u_raw = (w_x
                        .mul_add(position_x, w_y.mul_add(position_y, w_z * position_z)))
                        / (w_norm * position_norm);
                    sin_u_raw.clamp(-T::one(), T::one())
                } else {
                    T::zero()
                }
            };
            sin_u.atan2(cos_u)
        }
    } else {
        let e_denom = e_safe.max(eccentricity_divisor_floor);
        let dot_er = (eccentricity_x.mul_add(
            position_x,
            eccentricity_y.mul_add(position_y, eccentricity_z * position_z),
        )) / (e_denom * position_norm);
        let xv = dot_er.clamp(-T::one(), T::one());
        let mut nu_tmp = xv.acos();
        let rv = position_x.mul_add(
            velocity_x,
            position_y.mul_add(velocity_y, position_z * velocity_z),
        );
        if rv < T::zero() {
            nu_tmp = scalar::<T>(TWO_PI) - nu_tmp;
        }
        nu_tmp
    };
    let nu = mod2pi(nu);

    let a = -mu / (scalar::<T>(2.0) * energy);
    let e_anom = if e_safe < tol {
        nu
    } else {
        let sqrt_one_minus_e2 = e_safe.mul_add(-e_safe, T::one()).max(T::zero()).sqrt();
        let (sin_nu, cos_nu) = nu.sin_cos();
        let mut denom = e_safe.mul_add(cos_nu, T::one());
        if denom.abs() < tol {
            denom = if denom >= T::zero() { tol } else { -tol };
        }
        let sin_e_tmp = (sqrt_one_minus_e2 * sin_nu) / denom;
        let cos_e_tmp = (e_safe + cos_nu) / denom;
        sin_e_tmp.atan2(cos_e_tmp)
    };
    let e_anom_mod = mod2pi(e_anom);
    let m_anom = mod2pi(e_anom_mod - e_safe * e_anom_mod.sin());
    let anomaly = if true_anom { nu } else { m_anom };

    if deg {
        let r2d = scalar::<T>(180.0 / std::f64::consts::PI);
        write_six(
            out,
            [
                a,
                e_safe,
                inclination * r2d,
                raan * r2d,
                argp * r2d,
                anomaly * r2d,
            ],
        );
    } else {
        write_six(out, [a, e_safe, inclination, raan, argp, anomaly]);
    }
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic Float formula uses IEEE finite/NaN semantics, has no integer-overflow path, and preserves parity-pinned operation order; checked integer math is irrelevant"
)]
pub fn kep2eci_impl<T: Float + FromPrimitive>(
    state: &[T],
    deg: bool,
    t0: T,
    t: T,
    true_anom: bool,
    out: &mut [T],
) {
    let Some(
        [semi_major_axis, eccentricity_input, inclination_input, raan_input, argument_periapsis_input, anomaly_input],
    ) = six_values(state)
    else {
        write_nan_six(out);
        return;
    };
    let mut eccentricity = eccentricity_input.abs();
    let mut inclination = inclination_input;
    let mut raan = raan_input;
    let mut argument_periapsis = argument_periapsis_input;
    let mut anomaly = anomaly_input;

    if !matches!(
        semi_major_axis.partial_cmp(&T::zero()),
        Some(std::cmp::Ordering::Greater)
    ) || eccentricity >= T::one()
    {
        write_nan_six(out);
        return;
    }

    if deg {
        let degrees_to_radians = scalar::<T>(std::f64::consts::PI / 180.0);
        inclination = inclination * degrees_to_radians;
        raan = raan * degrees_to_radians;
        argument_periapsis = argument_periapsis * degrees_to_radians;
        anomaly = anomaly * degrees_to_radians;
    }

    eccentricity = clamp_eccentricity(eccentricity);
    inclination = mod2pi(inclination);
    raan = mod2pi(raan);
    argument_periapsis = mod2pi(argument_periapsis);
    anomaly = mod2pi(anomaly);

    let mu = scalar::<T>(MU);
    let mean_motion = (mu / (semi_major_axis * semi_major_axis * semi_major_axis)).sqrt();
    let initial_mean_anomaly = if true_anom {
        true_to_mean_anomaly_impl(anomaly, eccentricity)
    } else {
        anomaly
    };
    let mean_anomaly = initial_mean_anomaly + mean_motion * (t - t0);
    let (sin_eccentric_anomaly, cos_eccentric_anomaly) =
        solve_kepler_e_sincos(mean_anomaly, eccentricity);

    let radius_magnitude = semi_major_axis * eccentricity.mul_add(-cos_eccentric_anomaly, T::one());
    let sqrt_one_minus_eccentricity_squared = eccentricity
        .mul_add(-eccentricity, T::one())
        .max(T::zero())
        .sqrt();
    let sqrt_mu_a = (mu * semi_major_axis).sqrt();

    let (sin_inclination, cos_inclination) = inclination.sin_cos();
    let (sin_raan, cos_raan) = raan.sin_cos();
    let (sin_argument_periapsis, cos_argument_periapsis) = argument_periapsis.sin_cos();

    let row0 = [
        cos_raan.mul_add(
            cos_argument_periapsis,
            -sin_raan * sin_argument_periapsis * cos_inclination,
        ),
        (-cos_raan).mul_add(
            sin_argument_periapsis,
            -sin_raan * cos_argument_periapsis * cos_inclination,
        ),
        sin_raan * sin_inclination,
    ];
    let row1 = [
        sin_raan.mul_add(
            cos_argument_periapsis,
            cos_raan * sin_argument_periapsis * cos_inclination,
        ),
        (-sin_raan).mul_add(
            sin_argument_periapsis,
            cos_raan * cos_argument_periapsis * cos_inclination,
        ),
        -cos_raan * sin_inclination,
    ];
    let row2 = [
        sin_argument_periapsis * sin_inclination,
        cos_argument_periapsis * sin_inclination,
        cos_inclination,
    ];
    let perifocal_position = [
        semi_major_axis * (cos_eccentric_anomaly - eccentricity),
        semi_major_axis * sqrt_one_minus_eccentricity_squared * sin_eccentric_anomaly,
        T::zero(),
    ];
    let perifocal_velocity = [
        -sqrt_mu_a * sin_eccentric_anomaly / radius_magnitude,
        sqrt_mu_a * sqrt_one_minus_eccentricity_squared * cos_eccentric_anomaly / radius_magnitude,
        T::zero(),
    ];

    write_six(
        out,
        [
            dot3(&row0, &perifocal_position),
            dot3(&row1, &perifocal_position),
            dot3(&row2, &perifocal_position),
            dot3(&row0, &perifocal_velocity),
            dot3(&row1, &perifocal_velocity),
            dot3(&row2, &perifocal_velocity),
        ],
    );
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic Float formula uses IEEE finite/NaN semantics, has no integer-overflow path, and preserves parity-pinned operation order; checked integer math is irrelevant"
)]
pub fn eci2equinoc_impl<T: Float + FromPrimitive>(
    state: &[T],
    len: usize,
    t: T,
    t0: T,
    out: &mut [T],
) {
    if len != 6 && len != 7 {
        write_nan_six(out);
        return;
    }
    let Some([position_x, position_y, position_z, velocity_x, velocity_y, velocity_z]) =
        six_values(state)
    else {
        write_nan_six(out);
        return;
    };

    let position = [position_x, position_y, position_z];
    let velocity = [velocity_x, velocity_y, velocity_z];
    let position_norm = norm3(&position);
    let velocity_norm = norm3(&velocity);
    let inverse_position_norm = T::one() / position_norm;
    let mu = scalar::<T>(MU);
    let energy = velocity_norm.mul_add(
        velocity_norm * scalar::<T>(0.5),
        -mu * inverse_position_norm,
    );
    let tol = scalar::<T>(TOL);
    if energy >= -tol {
        write_nan_six(out);
        return;
    }
    let angular_momentum = cross3(&position, &velocity);
    let angular_momentum_norm = norm3(&angular_momentum);
    let inverse_angular_momentum_norm = T::one() / angular_momentum_norm;
    let inverse_mu = T::one() / mu;
    let [cross_x, cross_y, cross_z] = cross3(&velocity, &angular_momentum);
    let eccentricity_vector = [
        cross_x.mul_add(inverse_mu, -position_x * inverse_position_norm),
        cross_y.mul_add(inverse_mu, -position_y * inverse_position_norm),
        cross_z.mul_add(inverse_mu, -position_z * inverse_position_norm),
    ];
    let eccentricity = clamp_eccentricity(norm3(&eccentricity_vector));
    let semi_major_axis = -mu / (scalar::<T>(2.0) * energy);
    if !matches!(
        semi_major_axis.partial_cmp(&T::zero()),
        Some(std::cmp::Ordering::Greater)
    ) {
        write_nan_six(out);
        return;
    }
    let [angular_x, angular_y, angular_z] = angular_momentum;
    let angular_z_normalized = angular_z * inverse_angular_momentum_norm;
    let angular_denominator = T::one() + angular_z_normalized;
    let angular_coefficient = if angular_denominator.abs() < tol {
        T::zero()
    } else {
        (T::one() - angular_z_normalized) / angular_denominator
    };
    let equinoctial_p = (T::one() + angular_coefficient)
        * angular_x
        * inverse_angular_momentum_norm
        * scalar::<T>(0.5);
    let equinoctial_q = -(T::one() + angular_coefficient)
        * angular_y
        * inverse_angular_momentum_norm
        * scalar::<T>(0.5);
    let p_squared = equinoctial_p * equinoctial_p;
    let q_squared = equinoctial_q * equinoctial_q;
    let two_pq = scalar::<T>(2.0) * equinoctial_p * equinoctial_q;
    let basis_denominator = p_squared.mul_add(T::one(), q_squared.mul_add(T::one(), T::one()));
    let basis_coefficient = T::one() / basis_denominator;
    let x_hat = [
        basis_coefficient * (q_squared.mul_add(T::one(), T::one() - p_squared)),
        basis_coefficient * two_pq,
        basis_coefficient * (scalar::<T>(-2.0) * equinoctial_p),
    ];
    let y_hat = [
        basis_coefficient * two_pq,
        basis_coefficient * (p_squared.mul_add(T::one(), T::one() - q_squared)),
        basis_coefficient * (scalar::<T>(2.0) * equinoctial_q),
    ];
    let h_value = dot3(&y_hat, &eccentricity_vector);
    let k_value = dot3(&x_hat, &eccentricity_vector);
    let x_deq = dot3(&x_hat, &velocity);
    let y_deq = dot3(&y_hat, &velocity);
    let sqrt_mu_a = (mu * semi_major_axis).sqrt();
    let inverse_semi_major_axis = T::one() / semi_major_axis;
    let coefficient_2 = position_norm / sqrt_mu_a;
    let sqrt_term = eccentricity
        .mul_add(-eccentricity, T::one())
        .max(T::zero())
        .sqrt();
    let longitude_denominator = T::one() + sqrt_term;
    let safe_longitude_denominator = if longitude_denominator.abs() < tol {
        scalar::<T>(
            if longitude_denominator.to_f64().unwrap_or(1.0) >= 0.0 {
                1.0
            } else {
                -1.0
            } * TOL,
        )
    } else {
        longitude_denominator
    };
    let coefficient_3 =
        (position_norm * inverse_semi_major_axis - T::one()) / safe_longitude_denominator;
    let sin_true_anomaly = (-x_deq).mul_add(coefficient_2, -h_value * coefficient_3);
    let cos_true_anomaly = y_deq.mul_add(coefficient_2, -k_value * coefficient_3);
    let true_anomaly = sin_true_anomaly.atan2(cos_true_anomaly);
    let (sin_true_anomaly, cos_true_anomaly) = true_anomaly.sin_cos();
    let mut longitude =
        true_anomaly + h_value.mul_add(cos_true_anomaly, -k_value * sin_true_anomaly);
    let delta_t = t - t0;
    if delta_t != T::zero() {
        let inverse_semi_major_axis_squared = inverse_semi_major_axis * inverse_semi_major_axis;
        let mean_motion = sqrt_mu_a * inverse_semi_major_axis_squared;
        longitude = longitude + mean_motion * delta_t;
    }
    longitude = mod2pi(longitude);
    write_six(
        out,
        [
            semi_major_axis,
            h_value,
            k_value,
            equinoctial_p,
            equinoctial_q,
            longitude,
        ],
    );
}

/// Specialized `f64` version of `eci2equinoc_impl` without generic trait-dispatch overhead.
///
/// This version eliminates generic `from_f64` conversion overhead (about 15--20 calls per
/// invocation) by using inline `f64` math.
///
/// Performance: 15-25% faster than the generic version for f64 inputs.
#[inline]
pub fn eci2equinoc_impl_f64(state: &[f64], len: usize, t: f64, t0: f64, out: &mut [f64]) {
    // Fast path check
    if len != 6 && len != 7 {
        write_nan_six(out);
        return;
    }

    // Extract position and velocity
    let Some([rx, ry, rz, vx, vy, vz]) = six_values(state) else {
        write_nan_six(out);
        return;
    };

    // Position and velocity magnitudes (inline norm3)
    let r_norm = (rx * rx + ry * ry + rz * rz).sqrt();
    let v_norm = (vx * vx + vy * vy + vz * vz).sqrt();
    let inv_r = 1.0 / r_norm;

    // Energy check: E = 0.5*v^2 - mu/r (must be negative for bound orbit)
    // Use the shared gravity parameter, preserving the specialized f64 operation order.
    let energy = 0.5_f64.mul_add(v_norm * v_norm, -MU * inv_r);
    if energy >= -1e-12 {
        write_nan_six(out);
        return;
    }

    // Specific angular momentum: h = r × v (inline cross3)
    let hx = ry * vz - rz * vy;
    let hy = rz * vx - rx * vz;
    let hz = rx * vy - ry * vx;
    let h_norm = (hx * hx + hy * hy + hz * hz).sqrt();
    let inv_h = 1.0 / h_norm;

    // Eccentricity vector: e = (v × h)/mu - r/|r|
    let tmp_x = vy * hz - vz * hy;
    let tmp_y = vz * hx - vx * hz;
    let tmp_z = vx * hy - vy * hx;
    let ex = tmp_x.mul_add(INV_MU, -rx * inv_r);
    let ey = tmp_y.mul_add(INV_MU, -ry * inv_r);
    let ez = tmp_z.mul_add(INV_MU, -rz * inv_r);
    let e_mag = (ex * ex + ey * ey + ez * ez).sqrt();

    // Clamp eccentricity (inline clamp_eccentricity)
    let e_safe = if !e_mag.is_finite() || e_mag <= 0.0 {
        0.0
    } else if e_mag >= 1.0 {
        1.0 - 1e-11 // 1 - 10*TOL
    } else {
        e_mag
    };

    // Semi-major axis: a = -mu / (2*E)
    let a = -MU / (2.0 * energy);
    if !matches!(a.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        write_nan_six(out);
        return;
    }

    // p, q coefficients from angular momentum
    let normalized_angular_momentum_z = hz * inv_h;
    let denom_h = 1.0 + normalized_angular_momentum_z;
    let hcoef = if denom_h.abs() < 1e-12 {
        0.0
    } else {
        (1.0 - normalized_angular_momentum_z) / denom_h
    };
    let p = (1.0 + hcoef) * hx * inv_h * 0.5;
    let q = -(1.0 + hcoef) * hy * inv_h * 0.5;

    // Build equatorial frame vectors
    let p2 = p * p;
    let q2 = q * q;
    let two_pq = 2.0 * p * q;
    let denom = p2.mul_add(1.0, q2.mul_add(1.0, 1.0));
    let coef = 1.0 / denom;

    let x_hat_0 = coef * (q2.mul_add(1.0, 1.0 - p2));
    let x_hat_1 = coef * two_pq;
    let x_hat_2 = coef * (-2.0 * p);

    let y_hat_0 = coef * two_pq;
    let y_hat_1 = coef * (p2.mul_add(1.0, 1.0 - q2));
    let y_hat_2 = coef * (2.0 * q);

    // h, k from eccentricity vector projections (inline dot3)
    let h_val = y_hat_0 * ex + y_hat_1 * ey + y_hat_2 * ez;
    let k_val = x_hat_0 * ex + x_hat_1 * ey + x_hat_2 * ez;

    // Velocity projections
    let xdeq = x_hat_0 * vx + x_hat_1 * vy + x_hat_2 * vz;
    let ydeq = y_hat_0 * vx + y_hat_1 * vy + y_hat_2 * vz;

    // True longitude computation
    let sqrt_ma = (MU * a).sqrt();
    let inv_a = 1.0 / a;
    let coef2 = r_norm / sqrt_ma;

    let sqrt_term = (1.0 - e_safe * e_safe).max(0.0).sqrt();
    let denom_s = 1.0 + sqrt_term;
    let safe_denom_s = if denom_s.abs() < 1e-12 {
        if denom_s >= 0.0 {
            1e-12
        } else {
            -1e-12
        }
    } else {
        denom_s
    };
    let coef3 = (r_norm * inv_a - 1.0) / safe_denom_s;

    let s_f_anom = (-xdeq).mul_add(coef2, -h_val * coef3);
    let c_f_anom = ydeq.mul_add(coef2, -k_val * coef3);
    let f_anom = s_f_anom.atan2(c_f_anom);

    let (sin_f_anom, cos_f_anom) = f_anom.sin_cos();
    let mut lam = f_anom + h_val.mul_add(cos_f_anom, -k_val * sin_f_anom);

    // Mean longitude adjustment
    let delta_t = t - t0;
    if delta_t != 0.0 {
        let inv_a2 = inv_a * inv_a;
        let mean_motion = sqrt_ma * inv_a2;
        lam += mean_motion * delta_t;
    }

    // Wrap to [0, 2π) (inline mod2pi for f64)
    if lam.is_finite() {
        lam %= TWO_PI;
        if lam < 0.0 {
            lam += TWO_PI;
        }
        if lam >= TWO_PI {
            lam -= TWO_PI;
        }
    } else {
        lam = f64::NAN;
    }

    write_six(out, [a, h_val, k_val, p, q, lam]);
}

/// SIMD ECI to Equinoctial conversion: process 4 states simultaneously.
///
/// # Arguments
/// * `rx, ry, rz` - Position components (4 satellites packed)
/// * `vx, vy, vz` - Velocity components (4 satellites packed)
/// * `t` - Current time (4 values)
/// * `t0` - Reference epoch (4 values)
///
/// # Returns
/// [a, h, k, p, q, lam] - Equinoctial elements (4 satellites packed in each)
#[inline]
#[must_use]
pub fn eci2equinoc_simd(
    rx: f64x4,
    ry: f64x4,
    rz: f64x4,
    vx: f64x4,
    vy: f64x4,
    vz: f64x4,
    t: f64x4,
    t0: f64x4,
) -> [f64x4; 6] {
    let zero = f64x4::ZERO;
    let one = f64x4::ONE;
    let half = HALF_X4;
    let two = TWO_X4;
    let mu = MU_X4;
    let tol = TOL_X4;
    let nan = NAN_X4;

    // Position and velocity magnitudes
    let r_sq = rx * rx + ry * ry + rz * rz;
    let r_norm = r_sq.sqrt();
    let v_sq = vx * vx + vy * vy + vz * vz;

    // Energy: E = 0.5*v^2 - mu/r  (must be negative for bound orbit)
    let inv_r = one / r_norm;
    let energy = v_sq * half - mu * inv_r;

    // Validity check: energy < -TOL (bound orbit)
    let valid = energy.simd_lt(-tol);

    // Specific angular momentum: h = r × v
    let hx = ry * vz - rz * vy;
    let hy = rz * vx - rx * vz;
    let hz = rx * vy - ry * vx;
    let h_sq = hx * hx + hy * hy + hz * hz;
    let h_norm = h_sq.sqrt();
    let inv_h = one / h_norm;

    // Eccentricity vector: e = (v × h)/mu - r/|r|
    let inv_mu = one / mu;
    let temp_x = vy * hz - vz * hy;
    let temp_y = vz * hx - vx * hz;
    let temp_z = vx * hy - vy * hx;
    let ex = temp_x * inv_mu - rx * inv_r;
    let ey = temp_y * inv_mu - ry * inv_r;
    let ez = temp_z * inv_mu - rz * inv_r;
    let e_mag_sq = ex * ex + ey * ey + ez * ez;
    let e_mag = e_mag_sq.sqrt();
    let e_safe = clamp_eccentricity_simd(e_mag);

    // Semi-major axis: a = -mu / (2*E)
    let a = -mu / (two * energy);

    // Check a > 0 (additional validity)
    let valid = valid & a.simd_gt(zero);

    // Compute equinoctial p, q from angular momentum
    // These relate to inclination and RAAN
    let normalized_angular_momentum_z = hz * inv_h;
    let denom_h = one + normalized_angular_momentum_z;

    // Handle near-polar case (hz ≈ -1)
    let near_polar = denom_h.abs().simd_lt(tol);
    let hcoef = near_polar.select(zero, (one - normalized_angular_momentum_z) / denom_h);

    let p = (one + hcoef) * hx * inv_h * half;
    let q = -(one + hcoef) * hy * inv_h * half;

    // Compute x_hat and y_hat basis vectors from p, q
    let p2 = p * p;
    let q2 = q * q;
    let two_pq = two * p * q;
    let denom = one + p2 + q2;
    let coef = one / denom;

    let x_hat_x = coef * (one + q2 - p2);
    let x_hat_y = coef * two_pq;
    let x_hat_z = coef * (-two * p);

    let y_hat_x = coef * two_pq;
    let y_hat_y = coef * (one + p2 - q2);
    let y_hat_z = coef * (two * q);

    // Compute h, k from eccentricity vector projections
    let h_val = y_hat_x * ex + y_hat_y * ey + y_hat_z * ez;
    let k_val = x_hat_x * ex + x_hat_y * ey + x_hat_z * ez;

    // Compute velocity projections for true longitude
    let xdeq = x_hat_x * vx + x_hat_y * vy + x_hat_z * vz;
    let ydeq = y_hat_x * vx + y_hat_y * vy + y_hat_z * vz;

    // Compute true anomaly f
    let sqrt_ma = (mu * a).sqrt();
    let inv_a = one / a;
    let coef2 = r_norm / sqrt_ma;
    let sqrt_term = (one - e_safe * e_safe).max(zero).sqrt();
    let denom_s = one + sqrt_term;

    // Handle circular orbit case (e ≈ 0, denom_s ≈ 2)
    let near_circular = denom_s.abs().simd_lt(tol);
    let safe_denom_s = near_circular.select(tol.copysign(denom_s), denom_s);

    let coef3 = (r_norm * inv_a - one) / safe_denom_s;
    let s_f_anom = -xdeq * coef2 - h_val * coef3;
    let c_f_anom = ydeq * coef2 - k_val * coef3;
    let f_anom = s_f_anom.atan2(c_f_anom);
    let (sin_f_anom, cos_f_anom) = f_anom.sin_cos();

    // Mean longitude: lam = f + h*cos(f) - k*sin(f) + n*(t - t0)
    let mut lam = f_anom + h_val * cos_f_anom - k_val * sin_f_anom;

    // Add mean motion correction
    let delta_t = t - t0;
    let inv_a2 = inv_a * inv_a;
    let mean_motion = sqrt_ma * inv_a2;
    lam += mean_motion * delta_t;

    // Wrap to [0, 2π)
    lam = mod2pi_simd(lam);

    // Mask invalid lanes with NaN
    let a_out = valid.select(a, nan);
    let h_out = valid.select(h_val, nan);
    let k_out = valid.select(k_val, nan);
    let p_out = valid.select(p, nan);
    let q_out = valid.select(q, nan);
    let lam_out = valid.select(lam, nan);

    [a_out, h_out, k_out, p_out, q_out, lam_out]
}

/// The part of an equinoctial-to-ECI conversion that does not depend on time.
///
/// # Why this exists
///
/// [`equinoc2eci_impl`] runs once per RHS evaluation on the baseline cache's
/// miss path, and on the pinned strict-HF arc it misses most evaluations --
/// 0.85 calls per evaluation, measured, against roughly 5.6% of the arc. Every
/// one of those calls receives the SAME six elements; only `t` differs.
///
/// So everything derived from the elements alone was being recomputed and
/// discarded on each call: three `sqrt` (eccentricity, mean motion, and the one
/// inside the series coefficient), a reciprocal, and the nine products of the
/// `x_hat`/`y_hat` rotation. None of it depends on time. Holding a baseline
/// across a propagation performs that work once.
///
/// # Bit-identity is structural, not asserted
///
/// [`equinoc2eci_impl`] is now exactly [`Self::new`] followed by
/// [`Self::state_at`], so no second copy of this arithmetic exists to drift
/// from the first. Every expression below keeps the form it had -- including
/// each `mul_add` and each association, which are NOT interchangeable with
/// their algebraic equals here. `q_squared.mul_add(one, one - p_squared)` is
/// `q2 + (1 - p2)` and rounds differently from `(1 + q2) - p2`; the series
/// coefficient's `mul_add(-e, one)` is a single rounding where `1 - e*e` is
/// two. That is also why this type does not serve `equinoc2eci_impl_f64`, whose
/// body spells several of these the other way and is a different function
/// producing different bits.
#[derive(Clone, Copy, Debug)]
pub struct EquinoctialBaseline<T> {
    semi_major_axis: T,
    h_value: T,
    k_value: T,
    mean_motion: T,
    /// `1 + sqrt(1 - e^2)`, the equinoctial series denominator.
    eccentricity_coefficient: T,
    x_hat: [T; 3],
    y_hat: [T; 3],
    /// The element set's own mean longitude, before any time advance.
    longitude_0: T,
}

impl<T: Float + FromPrimitive> EquinoctialBaseline<T> {
    /// Build the time-invariant part, or `None` in exactly the cases where
    /// [`equinoc2eci_impl`] writes NaN without iterating: a bad length, a
    /// semi-major axis that is not strictly positive (NaN included, which is
    /// why this is a `partial_cmp` and not a `>`), or a non-sub-unity
    /// eccentricity.
    #[inline]
    #[must_use]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "generic Float formula uses IEEE finite/NaN semantics, has no integer-overflow path, and preserves parity-pinned operation order; checked integer math is irrelevant"
    )]
    pub fn new(elems: &[T], len: usize) -> Option<Self> {
        if len != 6 && len != 7 {
            return None;
        }
        let [semi_major_axis, h_value, k_value, equinoctial_p, equinoctial_q, longitude_input] =
            six_values(elems)?;
        if !matches!(
            semi_major_axis.partial_cmp(&T::zero()),
            Some(std::cmp::Ordering::Greater)
        ) {
            return None;
        }
        let eccentricity_raw = h_value.mul_add(h_value, k_value * k_value).sqrt();
        if eccentricity_raw >= T::one() {
            return None;
        }
        let mu = scalar::<T>(MU);
        let eccentricity = clamp_eccentricity(eccentricity_raw);
        let p_squared = equinoctial_p * equinoctial_p;
        let q_squared = equinoctial_q * equinoctial_q;
        let basis_denominator = p_squared.mul_add(T::one(), q_squared.mul_add(T::one(), T::one()));
        let basis_coefficient = T::one() / basis_denominator;
        Some(Self {
            semi_major_axis,
            h_value,
            k_value,
            mean_motion: (mu / (semi_major_axis * semi_major_axis * semi_major_axis)).sqrt(),
            eccentricity_coefficient: T::one()
                + eccentricity
                    .mul_add(-eccentricity, T::one())
                    .max(T::zero())
                    .sqrt(),
            x_hat: [
                basis_coefficient * (q_squared.mul_add(T::one(), T::one() - p_squared)),
                basis_coefficient * (scalar::<T>(2.0) * equinoctial_p * equinoctial_q),
                basis_coefficient * (scalar::<T>(-2.0) * equinoctial_p),
            ],
            y_hat: [
                basis_coefficient * (scalar::<T>(2.0) * equinoctial_p * equinoctial_q),
                basis_coefficient * (p_squared.mul_add(T::one(), T::one() - q_squared)),
                basis_coefficient * (scalar::<T>(2.0) * equinoctial_q),
            ],
            longitude_0: longitude_input,
        })
    }

    /// The ECI state at `t`, or NaN in `out[0]` if the Kepler iteration does not
    /// converge -- the one failure mode that depends on time and therefore
    /// cannot be decided in [`Self::new`].
    #[inline]
    pub fn state_at(&self, t: T, t0: T, out: &mut [T]) {
        self.state_at_seeded(t, t0, None, out);
    }

    /// [`Self::state_at`], seeding the longitude solve from a previous root.
    ///
    /// Returns the converged eccentric-minus-mean longitude offset `F - L`, to
    /// be handed back as `seed_offset` on the next call for the SAME element
    /// set, or `None` where the iteration did not converge and `out[0]` is NaN.
    ///
    /// # Why an offset and not the root
    ///
    /// `L` is `mod2pi`'d, so a root carried across the seam is a full
    /// revolution away from the one being sought and costs iterations instead
    /// of saving them. The offset does not wrap: `|F - L| <= e` for every root
    /// of `F + h cos F - k sin F = L`, so `L + offset` lands within the
    /// previous call's own convergence error of the new root no matter where in
    /// the orbit either call sits, and needs no unwrapping and no division.
    ///
    /// # `None` is bit-identical to the unseeded solve
    ///
    /// The unseeded seed is `L` itself and `None` reproduces it exactly, so
    /// [`Self::state_at`] is unchanged to the last ULP. A `Some` seed is a
    /// DIFFERENT starting point and the loop's exit test is on the step, not on
    /// a residual: the root it returns is the same to well within the 1e-12
    /// step tolerance but is NOT bit-identical, so every digest downstream of a
    /// seeded call moves.
    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "generic Float formula uses IEEE finite/NaN semantics, has no integer-overflow path, and preserves parity-pinned operation order; checked integer math is irrelevant"
    )]
    pub fn state_at_seeded(&self, t: T, t0: T, seed_offset: Option<T>, out: &mut [T]) -> Option<T> {
        let semi_major_axis = self.semi_major_axis;
        let h_value = self.h_value;
        let k_value = self.k_value;
        let mean_motion = self.mean_motion;
        let eccentricity_coefficient = self.eccentricity_coefficient;
        let longitude = wrap_equinoctial_longitude(self.longitude_0 + mean_motion * (t - t0));

        // `None` must reproduce the unseeded starting point EXACTLY, which is
        // why the default is `longitude` itself and not `longitude + zero`.
        // The two agree here -- `mod2pi` returns `+0.0` for a zero result and a
        // strictly positive value otherwise, and `x + 0.0` is `x` for every
        // such `x` -- but the identity that has to hold is on the seed, not on
        // an addition, and writing it this way does not depend on that fact.
        let mut true_anomaly = seed_offset.map_or(longitude, |offset| longitude + offset);
        let mut iteration_count = 0;
        let (mut sin_true_anomaly, mut cos_true_anomaly) = true_anomaly.sin_cos();
        let tolerance = scalar::<T>(TOL);
        while iteration_count < MAXITER {
            let a_term = h_value * cos_true_anomaly - k_value * sin_true_anomaly;
            let b_term = h_value * sin_true_anomaly + k_value * cos_true_anomaly;
            let f0 = true_anomaly + a_term - longitude;
            let f1 = T::one() - b_term;
            let f2 = -a_term;
            let f3 = b_term;
            let delta1 = -f0 / f1;
            let delta2 = -f0 / (f1 + scalar::<T>(0.5) * delta1 * f2);
            let delta3 = -f0
                / (f1
                    + scalar::<T>(0.5) * delta2 * f2
                    + scalar::<T>(1.0 / 6.0) * delta2 * delta2 * f3);
            true_anomaly = true_anomaly + delta3;
            if delta3.abs() < tolerance {
                break;
            }
            let (sin_delta, cos_delta) = delta3.sin_cos();
            let sin_true_anomaly_new =
                sin_true_anomaly.mul_add(cos_delta, cos_true_anomaly * sin_delta);
            cos_true_anomaly = cos_true_anomaly.mul_add(cos_delta, -sin_true_anomaly * sin_delta);
            sin_true_anomaly = sin_true_anomaly_new;
            iteration_count += 1;
        }
        if iteration_count == MAXITER {
            write_nan_first(out);
            return None;
        }
        let coefficient = (longitude - true_anomaly) / eccentricity_coefficient;
        let x_equinoctial = semi_major_axis * (cos_true_anomaly - k_value - h_value * coefficient);
        let y_equinoctial = semi_major_axis * (sin_true_anomaly - h_value + k_value * coefficient);
        let [x_hat_x, x_hat_y, x_hat_z] = self.x_hat;
        let [y_hat_x, y_hat_y, y_hat_z] = self.y_hat;
        let position_x = x_hat_x * x_equinoctial + y_hat_x * y_equinoctial;
        let position_y = x_hat_y * x_equinoctial + y_hat_y * y_equinoctial;
        let position_z = x_hat_z * x_equinoctial + y_hat_z * y_equinoctial;
        let position_norm = norm3(&[position_x, position_y, position_z]);
        let true_anomaly_rate = semi_major_axis * mean_motion / position_norm;
        let coefficient_3 = (mean_motion - true_anomaly_rate) / eccentricity_coefficient;
        let x_deq = semi_major_axis
            * ((-true_anomaly_rate).mul_add(sin_true_anomaly, -h_value * coefficient_3));
        let y_deq = semi_major_axis
            * (true_anomaly_rate.mul_add(cos_true_anomaly, k_value * coefficient_3));
        write_six(
            out,
            [
                position_x,
                position_y,
                position_z,
                x_hat_x * x_deq + y_hat_x * y_deq,
                x_hat_y * x_deq + y_hat_y * y_deq,
                x_hat_z * x_deq + y_hat_z * y_deq,
            ],
        );
        // `-coefficient * eccentricity_coefficient` would round differently.
        // The offset is a seed, not a result, but it is the quantity the next
        // call adds to its own `longitude`, so it stays the plain difference.
        Some(true_anomaly - longitude)
    }
}

/// One lane of [`EquinoctialBaseline::state_at_seeded_x4`]'s four-at-a-time
/// longitude solve.
///
/// A struct per lane rather than four parallel arrays so every loop in that
/// function is an `iter_mut`: `clippy::indexing_slicing` is denied in this
/// crate, and the `.get()?` form it pushes you to would put a loop-exit edge in
/// the middle of the arithmetic the function exists to keep straight-line.
#[derive(Clone, Copy)]
struct Lane {
    longitude: f64,
    true_anomaly: f64,
    sin_true_anomaly: f64,
    cos_true_anomaly: f64,
    /// The last Halley step this lane computed, kept across the convergence
    /// test so the rotation below can reuse it.
    step: f64,
    done: bool,
}

impl EquinoctialBaseline<f64> {
    /// Four times solved together, every lane seeded from the SAME incoming
    /// offset.
    ///
    /// # Why this exists, and why it is not a SIMD kernel
    ///
    /// [`Self::state_at_seeded`] is LATENCY-bound, not throughput-bound. One
    /// call is three dependent divisions per Halley pass at ~1.9 passes, a
    /// division and a square root in the tail, and a `sin_cos` — a single
    /// dependency chain with nothing to overlap. Chaining the seed makes the
    /// calls dependent on each other too, so a run of them is one long serial
    /// chain and the machine's issue width sits idle.
    ///
    /// Giving it four INDEPENDENT chains is the whole lever. Measured on the
    /// M1 Max against a faithful replica of the serial path, at the pinned
    /// arc's shape: 82.3 ns per solve serial against 28.7 ns packed, i.e.
    /// 2.86x, while doing strictly MORE arithmetic (2.000 passes per solve
    /// against 1.875, 1.938 `sin_cos` against 1.875) because a converged lane
    /// keeps iterating until its pack-mates catch up. Nothing here is a wider
    /// instruction; the same measurement with the `sin_cos` removed entirely
    /// moves the serial number by 6-8%, which is what says the transcendental
    /// is not the cost.
    ///
    /// # The shared seed is the price, and it moves bits
    ///
    /// Lane `i` cannot seed from lane `i - 1` without rebuilding the serial
    /// chain this exists to break, so all four start from `seed_offset`. That
    /// is a DIFFERENT starting point from the one the serial order would give
    /// lanes 1..3, and the loop exits on the step rather than on a residual, so
    /// the roots differ from the serial ones in the last ULP. Every digest
    /// downstream moves. It is bounded, not arbitrary: `|F - L| <= e` for every
    /// root, so a shared seed is wrong by at most the offset's drift across the
    /// four times, and at the pinned arc that is ~2e-3 against Halley's cubic
    /// convergence.
    ///
    /// # Convergence and failure are per lane
    ///
    /// The loop runs until EVERY lane has converged; a lane that converges
    /// early is frozen by `done[i]` and its arithmetic from then on is
    /// discarded, so its result is exactly what it would have been had the loop
    /// stopped for it alone. A lane that exhausts `MAXITER` gets NaN in its
    /// own `out[0]` and contributes no seed, matching
    /// [`Self::state_at_seeded`]'s contract one lane at a time.
    ///
    /// The returned offset is the LAST lane's, so the caller's next pack
    /// continues from a root of this one. `None` if any lane failed: a pack
    /// with a failure has no trustworthy root to carry, and the next call must
    /// start where an unseeded loop starts.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "same IEEE finite/NaN formula as state_at_seeded, whose operation order it reproduces per lane; no integer-overflow path"
    )]
    pub fn state_at_seeded_x4(
        &self,
        times: [f64; 4],
        t0: f64,
        seed_offset: Option<f64>,
        out: &mut [[f64; 6]; 4],
    ) -> Option<f64> {
        let semi_major_axis = self.semi_major_axis;
        let h_value = self.h_value;
        let k_value = self.k_value;
        let mean_motion = self.mean_motion;
        let eccentricity_coefficient = self.eccentricity_coefficient;

        let mut lanes = [Lane {
            longitude: 0.0,
            true_anomaly: 0.0,
            sin_true_anomaly: 0.0,
            cos_true_anomaly: 0.0,
            step: 0.0,
            done: false,
        }; 4];
        for (lane, &time) in lanes.iter_mut().zip(times.iter()) {
            let longitude =
                wrap_equinoctial_longitude(self.longitude_0 + mean_motion * (time - t0));
            // Spelled exactly as the scalar path spells it, `None` included:
            // the unseeded start is `longitude` itself, not `longitude + 0.0`.
            let true_anomaly = seed_offset.map_or(longitude, |offset| longitude + offset);
            let (sin_true_anomaly, cos_true_anomaly) = true_anomaly.sin_cos();
            lane.longitude = longitude;
            lane.true_anomaly = true_anomaly;
            lane.sin_true_anomaly = sin_true_anomaly;
            lane.cos_true_anomaly = cos_true_anomaly;
        }

        let tolerance = TOL;
        let mut iteration_count = 0;
        while iteration_count < MAXITER {
            for lane in &mut lanes {
                let sin_f = lane.sin_true_anomaly;
                let cos_f = lane.cos_true_anomaly;
                let a_term = h_value * cos_f - k_value * sin_f;
                let b_term = h_value * sin_f + k_value * cos_f;
                let f0 = lane.true_anomaly + a_term - lane.longitude;
                let f1 = 1.0 - b_term;
                let f2 = -a_term;
                let f3 = b_term;
                let delta1 = -f0 / f1;
                let delta2 = -f0 / (f1 + 0.5 * delta1 * f2);
                let delta3 = -f0 / (f1 + 0.5 * delta2 * f2 + (1.0 / 6.0) * delta2 * delta2 * f3);
                lane.step = delta3;
                if !lane.done {
                    lane.true_anomaly += delta3;
                }
            }
            let mut all_done = true;
            for lane in &mut lanes {
                if !lane.done && lane.step.abs() < tolerance {
                    lane.done = true;
                }
                all_done &= lane.done;
            }
            if all_done {
                break;
            }
            for lane in &mut lanes {
                // A converged lane's (sin, cos) must NOT advance: the scalar
                // path breaks before this rotation, so its tail reads the pair
                // from before the final step. Freezing here reproduces that.
                if lane.done {
                    continue;
                }
                let (sin_delta, cos_delta) = lane.step.sin_cos();
                let sin_new = lane
                    .sin_true_anomaly
                    .mul_add(cos_delta, lane.cos_true_anomaly * sin_delta);
                lane.cos_true_anomaly = lane
                    .cos_true_anomaly
                    .mul_add(cos_delta, -lane.sin_true_anomaly * sin_delta);
                lane.sin_true_anomaly = sin_new;
            }
            iteration_count += 1;
        }

        let mut any_failed = false;
        for (slot, lane) in out.iter_mut().zip(lanes.iter()) {
            if !lane.done {
                any_failed = true;
                write_nan_first(slot);
                continue;
            }
            let true_anomaly = lane.true_anomaly;
            let sin_true_anomaly = lane.sin_true_anomaly;
            let cos_true_anomaly = lane.cos_true_anomaly;
            let coefficient = (lane.longitude - true_anomaly) / eccentricity_coefficient;
            let x_equinoctial =
                semi_major_axis * (cos_true_anomaly - k_value - h_value * coefficient);
            let y_equinoctial =
                semi_major_axis * (sin_true_anomaly - h_value + k_value * coefficient);
            let [x_hat_x, x_hat_y, x_hat_z] = self.x_hat;
            let [y_hat_x, y_hat_y, y_hat_z] = self.y_hat;
            let position_x = x_hat_x * x_equinoctial + y_hat_x * y_equinoctial;
            let position_y = x_hat_y * x_equinoctial + y_hat_y * y_equinoctial;
            let position_z = x_hat_z * x_equinoctial + y_hat_z * y_equinoctial;
            let position_norm = norm3(&[position_x, position_y, position_z]);
            let true_anomaly_rate = semi_major_axis * mean_motion / position_norm;
            let coefficient_3 = (mean_motion - true_anomaly_rate) / eccentricity_coefficient;
            let x_deq = semi_major_axis
                * ((-true_anomaly_rate).mul_add(sin_true_anomaly, -h_value * coefficient_3));
            let y_deq = semi_major_axis
                * (true_anomaly_rate.mul_add(cos_true_anomaly, k_value * coefficient_3));
            *slot = [
                position_x,
                position_y,
                position_z,
                x_hat_x * x_deq + y_hat_x * y_deq,
                x_hat_y * x_deq + y_hat_y * y_deq,
                x_hat_z * x_deq + y_hat_z * y_deq,
            ];
        }
        if any_failed {
            return None;
        }
        lanes.last().map(|lane| lane.true_anomaly - lane.longitude)
    }
}

/// Equinoctial elements to an ECI state at one time.
///
/// Callers converting the SAME elements at many times should hold an
/// [`EquinoctialBaseline`] instead: this function is `new` and `state_at`
/// called back to back, and so pays the time-invariant half on every call.
#[inline]
pub fn equinoc2eci_impl<T: Float + FromPrimitive>(
    elems: &[T],
    len: usize,
    t: T,
    t0: T,
    out: &mut [T],
) {
    match EquinoctialBaseline::new(elems, len) {
        Some(baseline) => baseline.state_at(t, t0, out),
        None => write_nan_first(out),
    }
}

/// Specialized `f64` version of `equinoc2eci_impl` without generic trait-dispatch overhead.
///
/// This keeps the same numerical algorithm as the generic implementation but
/// avoids repeated `FromPrimitive` conversions in the hot f64 propagation path.
#[inline]
pub fn equinoc2eci_impl_f64(elems: &[f64], len: usize, t: f64, t0: f64, out: &mut [f64]) {
    if len != 6 && len != 7 {
        write_nan_first(out);
        return;
    }

    let Some([a, h_val, k_val, p, q, mut lam]) = six_values(elems) else {
        write_nan_first(out);
        return;
    };
    if a <= 0.0 {
        write_nan_first(out);
        return;
    }

    let e_raw = h_val.mul_add(h_val, k_val * k_val).sqrt();
    if e_raw >= 1.0 {
        write_nan_first(out);
        return;
    }

    let e_safe = clamp_eccentricity(e_raw);
    let n_val = (MU / (a * a * a)).sqrt();
    lam = mod2pi(lam + n_val * (t - t0));
    let mut f_anom = lam;
    let mut count = 0usize;
    let (mut sin_f, mut cos_f) = f_anom.sin_cos();
    while count < MAXITER {
        let a_term = h_val * cos_f - k_val * sin_f;
        let b_term = h_val * sin_f + k_val * cos_f;
        let f0 = f_anom + a_term - lam;
        let f1 = 1.0 - b_term;
        let f2 = -a_term;
        let f3 = b_term;
        let delta1 = -f0 / f1;
        let delta2 = -f0 / (f1 + 0.5 * delta1 * f2);
        let delta3 = -f0 / (f1 + 0.5 * delta2 * f2 + (1.0 / 6.0) * delta2 * delta2 * f3);
        f_anom += delta3;
        if delta3.abs() < TOL {
            break;
        }
        let (s_d, c_d) = delta3.sin_cos();
        let sin_f_new = sin_f.mul_add(c_d, cos_f * s_d);
        cos_f = cos_f.mul_add(c_d, -sin_f * s_d);
        sin_f = sin_f_new;
        count = count.saturating_add(1);
    }
    if count == MAXITER {
        write_nan_first(out);
        return;
    }

    let ecoef = 1.0 + (1.0 - e_safe * e_safe).max(0.0).sqrt();
    let coef = (lam - f_anom) / ecoef;
    let equinoctial_x = a * (cos_f - k_val - h_val * coef);
    let equinoctial_y = a * (sin_f - h_val + k_val * coef);
    let p2 = p * p;
    let q2 = q * q;
    let denom = 1.0 + p2 + q2;
    let coef2 = 1.0 / denom;
    let x_hat = [
        coef2 * (1.0 + q2 - p2),
        coef2 * (2.0 * p * q),
        coef2 * (-2.0 * p),
    ];
    let y_hat = [
        coef2 * (2.0 * p * q),
        coef2 * (1.0 + p2 - q2),
        coef2 * (2.0 * q),
    ];
    let [x_hat_x, x_hat_y, x_hat_z] = x_hat;
    let [y_hat_x, y_hat_y, y_hat_z] = y_hat;
    let position_x = x_hat_x * equinoctial_x + y_hat_x * equinoctial_y;
    let position_y = x_hat_y * equinoctial_x + y_hat_y * equinoctial_y;
    let position_z = x_hat_z * equinoctial_x + y_hat_z * equinoctial_y;
    let r_norm = norm3(&[position_x, position_y, position_z]);
    let f_dot = a * n_val / r_norm;
    let coef3 = (n_val - f_dot) / ecoef;
    let equinoctial_velocity_x = a * ((-f_dot).mul_add(sin_f, -h_val * coef3));
    let equinoctial_velocity_y = a * (f_dot.mul_add(cos_f, k_val * coef3));
    write_six(
        out,
        [
            position_x,
            position_y,
            position_z,
            x_hat_x * equinoctial_velocity_x + y_hat_x * equinoctial_velocity_y,
            x_hat_y * equinoctial_velocity_x + y_hat_y * equinoctial_velocity_y,
            x_hat_z * equinoctial_velocity_x + y_hat_z * equinoctial_velocity_y,
        ],
    );
}

// ========== SIMD Equinoctial Helpers ==========

#[inline]
fn mod2pi_simd(x: f64x4) -> f64x4 {
    let two_pi = TWO_PI_X4;
    let zero = f64x4::ZERO;

    // Compute x % TWO_PI using: x - floor(x/TWO_PI) * TWO_PI
    let quot = (x / two_pi).floor();
    let mut result = x - quot * two_pi;

    // Wrap negative values: if result < 0, add TWO_PI
    let neg_mask = result.simd_lt(zero);
    result = neg_mask.select(result + two_pi, result);

    // Wrap values >= TWO_PI: if result >= TWO_PI, subtract TWO_PI
    let high_mask = result.simd_ge(two_pi);
    result = high_mask.select(result - two_pi, result);

    // If exactly zero, ensure it's positive zero
    let zero_mask = result.simd_eq(zero);
    zero_mask.select(zero, result)
}

#[inline]
fn clamp_eccentricity_simd(e: f64x4) -> f64x4 {
    let zero = f64x4::ZERO;
    let one = f64x4::ONE;
    let eps = TEN_TOL_X4;
    let one_minus_eps = one - eps;

    // Clamp to [0.0, 1.0 - eps]
    e.max(zero).min(one_minus_eps)
}

// ========== SIMD Equinoctial to ECI Kernel ==========

/// SIMD kernel for converting equinoctial elements to ECI state vectors.
/// Processes 4 states in parallel using f64x4 SIMD vectors.
///
/// # Arguments
/// * `elems` - Array of 6 SIMD vectors [a, h, k, p, q, lam], each containing 4 values
/// * `t` - Propagation time (4 values)
/// * `t0` - Reference epoch (4 values)
///
/// # Returns
/// Array of 6 SIMD vectors [rx, ry, rz, vx, vy, vz] containing the ECI state
#[inline]
#[must_use]
pub fn equinoc2eci_simd(elems: &[f64x4; 6], t: f64x4, t0: f64x4) -> [f64x4; 6] {
    let [semi_major_axis, h_value, k_value, equinoctial_p, equinoctial_q, longitude_input] = *elems;

    let zero = f64x4::ZERO;
    let one = f64x4::ONE;
    let half = HALF_X4;
    let two = TWO_X4;
    let sixth = SIXTH_X4;
    let mu = MU_X4;
    let tol = TOL_X4;
    let nan = NAN_X4;

    // Check validity: a > 0
    let valid_a = semi_major_axis.simd_gt(zero);

    // Compute eccentricity and check e < 1
    let eccentricity_raw = (h_value * h_value + k_value * k_value).sqrt();
    let valid_eccentricity = eccentricity_raw.simd_lt(one);
    let valid = valid_a & valid_eccentricity;

    // Early exit if all lanes are invalid (rare, but check anyway)
    // For SIMD, we'll compute anyway and mask at the end

    let eccentricity = clamp_eccentricity_simd(eccentricity_raw);
    let mean_motion = (mu / (semi_major_axis * semi_major_axis * semi_major_axis)).sqrt();

    // Propagate mean longitude: lam = lam_in + n*(t - t0)
    let longitude = mod2pi_simd(longitude_input + mean_motion * (t - t0));

    // Solve for eccentric longitude f_anom using Newton-Raphson iteration
    let mut true_anomaly = longitude;
    let mut count = 0;
    let mut converged = f64x4::ZERO; // Mask: 0.0 = not converged, -1.0 (all bits set) = converged

    // Precompute sin/cos of f_anom via the canonical wide path.
    let (mut sin_true_anomaly, mut cos_true_anomaly) = true_anomaly.sin_cos();

    while count < MAXITER {
        let a_term = h_value * cos_true_anomaly - k_value * sin_true_anomaly;
        let b_term = h_value * sin_true_anomaly + k_value * cos_true_anomaly;

        // f0 = f_anom + a_term - lam
        let f0 = true_anomaly + a_term - longitude;
        let f1 = one - b_term;
        let f2 = -a_term;
        let f3 = b_term;

        // Halley's method (3rd order)
        let delta1 = -f0 / f1;
        let delta2 = -f0 / (f1 + half * delta1 * f2);
        let delta3 = -f0 / (f1 + half * delta2 * f2 + sixth * delta2 * delta2 * f3);

        // Update f_anom
        true_anomaly += delta3;

        // Check convergence: |delta3| < tol
        let conv_mask = delta3.abs().simd_lt(tol);
        converged |= conv_mask;

        // If all lanes converged, break
        if converged.all() {
            break;
        }

        // Update sin/cos incrementally using angle addition formulas
        let (s_d, c_d) = delta3.sin_cos();
        let sin_true_anomaly_new = sin_true_anomaly * c_d + cos_true_anomaly * s_d;
        cos_true_anomaly = cos_true_anomaly * c_d - sin_true_anomaly * s_d;
        sin_true_anomaly = sin_true_anomaly_new;

        count = count.saturating_add(1);
    }

    // If not converged after MAXITER, mark as invalid
    let valid = valid & converged;

    // Compute position and velocity in equinoctial frame
    let eccentricity_coefficient = one + (one - eccentricity * eccentricity).max(zero).sqrt();
    let coefficient = (longitude - true_anomaly) / eccentricity_coefficient;
    let x_equinoctial = semi_major_axis * (cos_true_anomaly - k_value - h_value * coefficient);
    let y_equinoctial = semi_major_axis * (sin_true_anomaly - h_value + k_value * coefficient);

    // Compute transformation from equinoctial to ECI frame
    let p_squared = equinoctial_p * equinoctial_p;
    let q_squared = equinoctial_q * equinoctial_q;
    let basis_denominator = one + p_squared + q_squared;
    let basis_coefficient = one / basis_denominator;

    let x_hat = [
        basis_coefficient * (one - p_squared + q_squared),
        basis_coefficient * (two * equinoctial_p * equinoctial_q),
        basis_coefficient * (-two * equinoctial_p),
    ];
    let y_hat = [
        basis_coefficient * (two * equinoctial_p * equinoctial_q),
        basis_coefficient * (one + p_squared - q_squared),
        basis_coefficient * (two * equinoctial_q),
    ];
    let [x_hat_x, x_hat_y, x_hat_z] = x_hat;
    let [y_hat_x, y_hat_y, y_hat_z] = y_hat;

    // Position in ECI
    let rx = x_hat_x * x_equinoctial + y_hat_x * y_equinoctial;
    let ry = x_hat_y * x_equinoctial + y_hat_y * y_equinoctial;
    let rz = x_hat_z * x_equinoctial + y_hat_z * y_equinoctial;

    // Compute velocity
    let r_norm = (rx * rx + ry * ry + rz * rz).sqrt();
    let true_anomaly_rate = semi_major_axis * mean_motion / r_norm;
    let coefficient_3 = (mean_motion - true_anomaly_rate) / eccentricity_coefficient;
    let x_deq = semi_major_axis * (-true_anomaly_rate * sin_true_anomaly - h_value * coefficient_3);
    let y_deq = semi_major_axis * (true_anomaly_rate * cos_true_anomaly + k_value * coefficient_3);

    // Velocity in ECI
    let vx = x_hat_x * x_deq + y_hat_x * y_deq;
    let vy = x_hat_y * x_deq + y_hat_y * y_deq;
    let vz = x_hat_z * x_deq + y_hat_z * y_deq;

    // Mask invalid lanes with NaN
    [
        valid.select(rx, nan),
        valid.select(ry, nan),
        valid.select(rz, nan),
        valid.select(vx, nan),
        valid.select(vy, nan),
        valid.select(vz, nan),
    ]
}

// ========== SIMD Transposition Helpers for AoS ↔ SoA Conversion ==========

/// Transpose 4 ECI states from `AoS` to `SoA` for SIMD processing.
#[inline]
const fn transpose_eci_aos_to_soa(chunk: &[f64; 24]) -> (f64x4, f64x4, f64x4, f64x4, f64x4, f64x4) {
    let [rx0, ry0, rz0, vx0, vy0, vz0, rx1, ry1, rz1, vx1, vy1, vz1, rx2, ry2, rz2, vx2, vy2, vz2, rx3, ry3, rz3, vx3, vy3, vz3] =
        *chunk;
    (
        f64x4::new([rx0, rx1, rx2, rx3]),
        f64x4::new([ry0, ry1, ry2, ry3]),
        f64x4::new([rz0, rz1, rz2, rz3]),
        f64x4::new([vx0, vx1, vx2, vx3]),
        f64x4::new([vy0, vy1, vy2, vy3]),
        f64x4::new([vz0, vz1, vz2, vz3]),
    )
}

/// Transpose 4 equinoctial states from `SoA` to `AoS`.
#[inline]
fn transpose_equ_soa_to_aos(equ: &[f64x4; 6], out: &mut [f64; 24]) {
    let [semi_major_axis, h_value, k_value, equinoctial_p, equinoctial_q, longitude] = *equ;
    let [a0, a1, a2, a3] = *semi_major_axis.as_array();
    let [h0, h1, h2, h3] = *h_value.as_array();
    let [k0, k1, k2, k3] = *k_value.as_array();
    let [p0, p1, p2, p3] = *equinoctial_p.as_array();
    let [q0, q1, q2, q3] = *equinoctial_q.as_array();
    let [lambda0, lambda1, lambda2, lambda3] = *longitude.as_array();

    for (out_state, values) in out.chunks_exact_mut(6).zip([
        [a0, h0, k0, p0, q0, lambda0],
        [a1, h1, k1, p1, q1, lambda1],
        [a2, h2, k2, p2, q2, lambda2],
        [a3, h3, k3, p3, q3, lambda3],
    ]) {
        out_state.copy_from_slice(&values);
    }
}

/// Transpose 4 equinoctial states from `AoS` to `SoA`.
#[inline]
const fn transpose_equ_aos_to_soa(chunk: &[f64; 24]) -> [f64x4; 6] {
    let [a0, h0, k0, p0, q0, lambda0, a1, h1, k1, p1, q1, lambda1, a2, h2, k2, p2, q2, lambda2, a3, h3, k3, p3, q3, lambda3] =
        *chunk;
    [
        f64x4::new([a0, a1, a2, a3]),
        f64x4::new([h0, h1, h2, h3]),
        f64x4::new([k0, k1, k2, k3]),
        f64x4::new([p0, p1, p2, p3]),
        f64x4::new([q0, q1, q2, q3]),
        f64x4::new([lambda0, lambda1, lambda2, lambda3]),
    ]
}

/// Transpose 4 ECI states from `SoA` to `AoS`.
#[inline]
fn transpose_eci_soa_to_aos(eci: &[f64x4; 6], out: &mut [f64; 24]) {
    let [position_x, position_y, position_z, velocity_x, velocity_y, velocity_z] = *eci;
    let [rx0, rx1, rx2, rx3] = *position_x.as_array();
    let [ry0, ry1, ry2, ry3] = *position_y.as_array();
    let [rz0, rz1, rz2, rz3] = *position_z.as_array();
    let [vx0, vx1, vx2, vx3] = *velocity_x.as_array();
    let [vy0, vy1, vy2, vy3] = *velocity_y.as_array();
    let [vz0, vz1, vz2, vz3] = *velocity_z.as_array();

    for (out_state, values) in out.chunks_exact_mut(6).zip([
        [rx0, ry0, rz0, vx0, vy0, vz0],
        [rx1, ry1, rz1, vx1, vy1, vz1],
        [rx2, ry2, rz2, vx2, vy2, vz2],
        [rx3, ry3, rz3, vx3, vy3, vz3],
    ]) {
        out_state.copy_from_slice(&values);
    }
}

/// Propagate a single equinoctial state to ECI at multiple time steps.
///
/// `equinoc`: six-element equinoctial state.
/// `t_vals`: propagation times in seconds.
/// `t0`: reference epoch in seconds.
/// `out`: output slice sized `t_vals.len() * 6`.
#[inline]
pub fn equinoc_prop_step_impl(equinoc: &[f64], t_vals: &[f64], t0: f64, out: &mut [f64]) {
    let n = t_vals.len();
    let Some(output_len) = n.checked_mul(6) else {
        return;
    };
    if n == 0 || out.len() < output_len {
        return;
    }
    let Some(equinoc_state) = six_values(equinoc) else {
        return;
    };
    let Some(out_prefix) = out.get_mut(..output_len) else {
        return;
    };

    let mut time_blocks = t_vals.chunks_exact(4);
    let mut out_blocks = out_prefix.chunks_exact_mut(24);
    for (time_block, out_block) in time_blocks.by_ref().zip(out_blocks.by_ref()) {
        let (Ok(time_array), Ok(out_array)) = (
            <&[f64; 4]>::try_from(time_block),
            <&mut [f64; 24]>::try_from(out_block),
        ) else {
            return;
        };
        equinoc_prop_step_simd4(&equinoc_state, time_array, t0, out_array);
    }

    for (&time, out_state) in time_blocks
        .remainder()
        .iter()
        .zip(out_blocks.into_remainder().chunks_exact_mut(6))
    {
        equinoc2eci_impl_f64(&equinoc_state, 6, time, t0, out_state);
    }
}

/// Propagate a single equinoctial state to ECI at four time steps (SIMD-assisted).
///
/// This is a fixed-size SIMD helper for batch candidate evaluation.
pub fn equinoc_prop_step_simd4(
    equinoc: &[f64; 6],
    t_vals: &[f64; 4],
    t0: f64,
    out: &mut [f64; 24],
) {
    let [semi_major_axis, h_value, k_value, equinoctial_p, equinoctial_q, longitude] = *equinoc;
    let [time0, time1, time2, time3] = *t_vals;
    let elems = [
        f64x4::splat(semi_major_axis),
        f64x4::splat(h_value),
        f64x4::splat(k_value),
        f64x4::splat(equinoctial_p),
        f64x4::splat(equinoctial_q),
        f64x4::splat(longitude),
    ];
    let time = f64x4::new([time0, time1, time2, time3]);
    let t0_vec = f64x4::splat(t0);
    let eci = equinoc2eci_simd(&elems, time, t0_vec);
    transpose_eci_soa_to_aos(&eci, out);
}

#[inline]
pub fn equinoc_prop_step_add_to_impl(equinoc: &[f64], t_vals: &[f64], t0: f64, out: &mut [f64]) {
    let n = t_vals.len();
    let Some(output_len) = n.checked_mul(6) else {
        return;
    };
    if n == 0 || out.len() < output_len {
        return;
    }
    let Some(equinoc_state) = six_values(equinoc) else {
        return;
    };
    let Some(out_prefix) = out.get_mut(..output_len) else {
        return;
    };

    let mut time_blocks = t_vals.chunks_exact(4);
    let mut out_blocks = out_prefix.chunks_exact_mut(24);
    let mut block = [0.0_f64; 24];
    for (time_block, out_block) in time_blocks.by_ref().zip(out_blocks.by_ref()) {
        let Ok(time_array) = <&[f64; 4]>::try_from(time_block) else {
            return;
        };
        equinoc_prop_step_simd4(&equinoc_state, time_array, t0, &mut block);
        for (destination, increment) in out_block.iter_mut().zip(block) {
            *destination += increment;
        }
    }

    let mut scalar = [0.0_f64; 6];
    for (&time, out_state) in time_blocks
        .remainder()
        .iter()
        .zip(out_blocks.into_remainder().chunks_exact_mut(6))
    {
        equinoc2eci_impl_f64(&equinoc_state, 6, time, t0, &mut scalar);
        for (destination, increment) in out_state.iter_mut().zip(scalar) {
            *destination += increment;
        }
    }
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic Float conversion uses IEEE finite/NaN semantics, has no integer-overflow path, and preserves orbital-element operation order; checked integer math is irrelevant"
)]
pub fn equinoc2kep_impl<T: Float + FromPrimitive>(
    equinoc: &[T],
    deg: bool,
    true_anom: bool,
    out: &mut [T],
) {
    if equinoc.len() != 6 {
        write_nan_first(out);
        return;
    }
    let Some(
        [semi_major_axis, h_equinoctial, k_equinoctial, p_equinoctial, q_equinoctial, longitude],
    ) = six_values(equinoc)
    else {
        write_nan_first(out);
        return;
    };
    if semi_major_axis <= T::zero() {
        write_nan_first(out);
        return;
    }
    let eccentricity = clamp_eccentricity(
        h_equinoctial
            .mul_add(h_equinoctial, k_equinoctial * k_equinoctial)
            .sqrt(),
    );
    let tan_half_inclination = p_equinoctial
        .mul_add(p_equinoctial, q_equinoctial * q_equinoctial)
        .sqrt();
    let inclination = mod2pi(scalar::<T>(2.0) * tan_half_inclination.atan());
    // Equatorial: the node line, and with it the (raan, argp) split, is
    // undefined — see `EQUATORIAL_PQ_TOL`. Both angles below would be
    // `atan2(+-0.0, +-0.0)`, i.e. decided by sign bits. Convention: put the
    // whole longitude of periapsis `atan2(h, k) = raan + argp` into argp and
    // report `raan = 0`. That keeps their sum — the only physically defined
    // combination — exact, so `anomaly = longitude - raan - argp` still
    // recovers the mean anomaly and the state round-trips.
    let (raan, argument_periapsis) = if tan_half_inclination <= scalar::<T>(EQUATORIAL_PQ_TOL) {
        // A circular equatorial orbit has h = k = 0 exactly as well, and
        // `atan2(h, k)` is then sign-bit noise for the same reason. Periapsis
        // is genuinely undefined there, so pin it to zero and let the anomaly
        // carry the whole angle — the reconstruction adds raan + argp back
        // either way, so the state is unchanged.
        //
        // The test is `> 0`, not `> CIRCULAR_E_TOL`. Any nonzero eccentricity
        // leaves (h, k) nonzero and `atan2(h, k)` exact, and discarding it
        // instead costs a position error of order `a * e` — 0.1 um at
        // e = 1e-11, but real, and there is no reason to pay it.
        let longitude_periapsis = if eccentricity > T::zero() {
            mod2pi(h_equinoctial.atan2(k_equinoctial))
        } else {
            T::zero()
        };
        (T::zero(), longitude_periapsis)
    } else {
        (
            mod2pi(p_equinoctial.atan2(q_equinoctial)),
            mod2pi(
                (h_equinoctial.mul_add(q_equinoctial, -k_equinoctial * p_equinoctial))
                    .atan2(k_equinoctial.mul_add(q_equinoctial, h_equinoctial * p_equinoctial)),
            ),
        )
    };
    let mut anomaly = mod2pi(longitude - raan - argument_periapsis);
    if true_anom {
        anomaly = mean_to_true_anomaly_impl(anomaly, eccentricity);
        anomaly = mod2pi(anomaly);
    }
    if deg {
        let radians_to_degrees = scalar::<T>(180.0 / std::f64::consts::PI);
        write_six(
            out,
            [
                semi_major_axis,
                eccentricity,
                inclination * radians_to_degrees,
                raan * radians_to_degrees,
                argument_periapsis * radians_to_degrees,
                anomaly * radians_to_degrees,
            ],
        );
    } else {
        write_six(
            out,
            [
                semi_major_axis,
                eccentricity,
                inclination,
                raan,
                argument_periapsis,
                anomaly,
            ],
        );
    }
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic Float conversion uses IEEE finite/NaN semantics, has no integer-overflow path, and preserves orbital-element operation order; checked integer math is irrelevant"
)]
pub fn kep2equinoc_impl<T: Float + FromPrimitive>(
    kep: &[T],
    deg: bool,
    true_anom: bool,
    out: &mut [T],
) {
    if kep.len() != 6 {
        write_nan_six(out);
        return;
    }
    let Some(
        [semi_major_axis, eccentricity_input, inclination_input, raan_input, argument_periapsis_input, anomaly_input],
    ) = six_values(kep)
    else {
        write_nan_six(out);
        return;
    };
    if semi_major_axis <= T::zero() || !semi_major_axis.is_finite() {
        write_nan_six(out);
        return;
    }
    let mut eccentricity = eccentricity_input;
    let mut inclination = inclination_input;
    let mut raan = raan_input;
    let mut argument_periapsis = argument_periapsis_input;
    let mut anomaly = anomaly_input;

    if deg {
        let degrees_to_radians = scalar::<T>(std::f64::consts::PI / 180.0);
        inclination = inclination * degrees_to_radians;
        raan = raan * degrees_to_radians;
        argument_periapsis = argument_periapsis * degrees_to_radians;
        anomaly = anomaly * degrees_to_radians;
    }

    let eccentricity_absolute = eccentricity.abs();
    if eccentricity_absolute >= T::one() {
        write_nan_six(out);
        return;
    }
    eccentricity = clamp_eccentricity(eccentricity_absolute);

    let (sin_longitude_periapsis, cos_longitude_periapsis) = (argument_periapsis + raan).sin_cos();
    let h_value = eccentricity * sin_longitude_periapsis;
    let k_value = eccentricity * cos_longitude_periapsis;

    let half_inclination = inclination * scalar::<T>(0.5);
    let tan_half_inclination = half_inclination.tan();
    let (sin_raan, cos_raan) = raan.sin_cos();
    let p_value = tan_half_inclination * sin_raan;
    let q_value = tan_half_inclination * cos_raan;

    let mut mean_anomaly = anomaly;
    if true_anom {
        mean_anomaly = true_to_mean_anomaly_impl(anomaly, eccentricity);
    }

    let longitude = mod2pi(mean_anomaly + argument_periapsis + raan);

    write_six(
        out,
        [
            semi_major_axis,
            h_value,
            k_value,
            p_value,
            q_value,
            longitude,
        ],
    );
}

// === ORBITAL PARAMS BATCH ===

/// Compute derived orbital parameters for a single orbit.
#[inline]
fn orbital_params_one(a: f64, e: f64, mu: f64) -> (f64, f64, f64, f64) {
    if !matches!(a.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let e_safe = clamp_eccentricity(e.abs());
    let one_minus_e2 = 1.0 - e_safe * e_safe;
    let b = a * one_minus_e2.max(0.0).sqrt();
    let apogee = a * (1.0 + e_safe);
    let perigee = a * (1.0 - e_safe);
    let period = if e_safe < 1.0 {
        std::f64::consts::TAU * (a * a * a / mu).sqrt()
    } else {
        f64::NAN
    };
    (b, apogee, perigee, period)
}

/// Batch compute orbital parameters.
///
/// # Arguments
/// * `a` - Semimajor axes [km]
/// * `e` - Eccentricities [dimensionless]  
/// * `mu` - Gravitational parameter [km³/s²]
/// * `b_out`, `apogee_out`, `perigee_out`, `period_out` - Output slices
pub fn orbital_params_batch_impl(
    a: &[f64],
    e: &[f64],
    mu: f64,
    b_out: &mut [f64],
    apogee_out: &mut [f64],
    perigee_out: &mut [f64],
    period_out: &mut [f64],
) {
    let n = a.len();
    let (
        Some(e_values),
        Some(b_values),
        Some(apogee_values),
        Some(perigee_values),
        Some(period_values),
    ) = (
        e.get(..n),
        b_out.get_mut(..n),
        apogee_out.get_mut(..n),
        perigee_out.get_mut(..n),
        period_out.get_mut(..n),
    )
    else {
        return;
    };

    #[cfg(feature = "parallel")]
    if n >= ORBITAL_PARAMS_PAR_THRESHOLD {
        let _ = nd_sched::init_global_pool(None);
        if rayon::current_thread_index().is_none() {
            use rayon::prelude::*;
            // Zip all output slices with input slices for parallel processing
            let iter = b_values
                .par_iter_mut()
                .zip(apogee_values.par_iter_mut())
                .zip(perigee_values.par_iter_mut())
                .zip(period_values.par_iter_mut())
                .zip(a.par_iter())
                .zip(e_values.par_iter());

            iter.for_each(
                |(((((b_ref, ap_ref), pe_ref), period_ref), &a_val), &e_val)| {
                    let (b, ap, pe, period) = orbital_params_one(a_val, e_val, mu);
                    *b_ref = b;
                    *ap_ref = ap;
                    *pe_ref = pe;
                    *period_ref = period;
                },
            );
            return;
        }
    }

    // Serial fallback
    for (((((b_ref, ap_ref), pe_ref), period_ref), &a_value), &e_value) in b_values
        .iter_mut()
        .zip(apogee_values.iter_mut())
        .zip(perigee_values.iter_mut())
        .zip(period_values.iter_mut())
        .zip(a.iter())
        .zip(e_values.iter())
    {
        let (b, ap, pe, period) = orbital_params_one(a_value, e_value, mu);
        *b_ref = b;
        *ap_ref = ap;
        *pe_ref = pe;
        *period_ref = period;
    }
}

#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic Float J2 secular propagation uses IEEE finite/NaN semantics, has no integer-overflow path, and preserves authority-pinned operation order; checked integer math is irrelevant"
)]
pub fn advance_equinoc_j2_impl<T: Float + FromPrimitive>(equinoc: &[T], delta_t: T, out: &mut [T]) {
    // First-order secular J2 model on equinoctial mean elements (no short-period terms).
    //
    // Element definitions:
    //   h = e * sin(omega + Omega)
    //   k = e * cos(omega + Omega)
    //   p = tan(i/2) * sin(Omega)
    //   q = tan(i/2) * cos(Omega)
    //   lambda = M + omega + Omega
    //
    // Keplerian secular rates:
    //   Omega_dot = -1.5 * n * J2 * (Re/p_orbit)^2 * cos(i)
    //   omega_dot =  0.75 * n * J2 * (Re/p_orbit)^2 * (5*cos(i)^2 - 1)
    //   M_dot     =  n + 0.75 * n * J2 * (Re/p_orbit)^2 * sqrt(1-e^2) * (3*cos(i)^2 - 1)
    //
    // Equinoctial mapping:
    //   (p, q) rotates by Delta Omega
    //   (h, k) rotates by Delta (Omega + omega)
    //   lambda advances by Delta (M + omega + Omega)
    //   a is invariant
    if out.len() < 6 {
        return;
    }
    let Some([semi_major_axis, h_value, k_value, equinoctial_p, equinoctial_q, longitude]) =
        six_values(equinoc)
    else {
        write_nan_first(out);
        return;
    };

    if !matches!(
        semi_major_axis.partial_cmp(&T::zero()),
        Some(std::cmp::Ordering::Greater)
    ) || !semi_major_axis.is_finite()
        || !h_value.is_finite()
        || !k_value.is_finite()
        || !equinoctial_p.is_finite()
        || !equinoctial_q.is_finite()
        || !longitude.is_finite()
        || !delta_t.is_finite()
    {
        write_nan_first(out);
        return;
    }

    let eccentricity_squared = h_value.mul_add(h_value, k_value * k_value);
    if eccentricity_squared < T::zero() || eccentricity_squared >= T::one() {
        write_nan_first(out);
        return;
    }
    let one_minus_eccentricity_squared = T::one() - eccentricity_squared;
    if !matches!(
        one_minus_eccentricity_squared.partial_cmp(&T::zero()),
        Some(std::cmp::Ordering::Greater)
    ) {
        write_nan_first(out);
        return;
    }

    let tan_half_inclination_squared =
        equinoctial_p.mul_add(equinoctial_p, equinoctial_q * equinoctial_q);
    let cos_inclination =
        (T::one() - tan_half_inclination_squared) / (T::one() + tan_half_inclination_squared);
    let orbit_parameter = semi_major_axis * one_minus_eccentricity_squared;
    if !matches!(
        orbit_parameter.partial_cmp(&T::zero()),
        Some(std::cmp::Ordering::Greater)
    ) {
        write_nan_first(out);
        return;
    }

    let mu = scalar::<T>(MU);
    let re = scalar::<T>(RE);
    let j2 = scalar::<T>(J2);
    let one = T::one();
    let three_halves = scalar::<T>(1.5);
    let three_quarters = scalar::<T>(0.75);
    let five = scalar::<T>(5.0);
    let three = scalar::<T>(3.0);

    let mean_motion = (mu / (semi_major_axis * semi_major_axis * semi_major_axis)).sqrt();
    let j2_factor = j2 * (re / orbit_parameter) * (re / orbit_parameter);
    let cos_inclination_squared = cos_inclination * cos_inclination;
    let raan_rate = -three_halves * mean_motion * j2_factor * cos_inclination;
    let periapsis_rate =
        three_quarters * mean_motion * j2_factor * (five * cos_inclination_squared - one);
    let mean_anomaly_rate = mean_motion
        + three_quarters
            * mean_motion
            * j2_factor
            * one_minus_eccentricity_squared.sqrt()
            * (three * cos_inclination_squared - one);
    let longitude_rate = raan_rate + periapsis_rate + mean_anomaly_rate;

    let d_raan = raan_rate * delta_t;
    let d_peri_long = (raan_rate + periapsis_rate) * delta_t;
    let (sin_raan, cos_raan) = d_raan.sin_cos();
    let (sin_peri, cos_peri) = d_peri_long.sin_cos();

    write_six(
        out,
        [
            semi_major_axis,
            h_value * cos_peri + k_value * sin_peri,
            k_value * cos_peri - h_value * sin_peri,
            equinoctial_p * cos_raan + equinoctial_q * sin_raan,
            equinoctial_q * cos_raan - equinoctial_p * sin_raan,
            mod2pi(longitude + longitude_rate * delta_t),
        ],
    );
}

pub fn equinoc_prop_from_impl<T: Float + FromPrimitive>(equi: &[T], tof: T, out_state: &mut [T]) {
    equinoc2eci_impl(equi, 6, tof, T::zero(), out_state);
}

#[inline]
pub fn equinoc_prop_j2_from_impl(equi: &[f64], tof: f64, out_state: &mut [f64]) {
    if out_state.len() < 6 {
        return;
    }
    let mut equ_adv = [0.0_f64; 6];
    advance_equinoc_j2_impl(equi, tof, &mut equ_adv);
    let [semi_major_axis, ..] = equ_adv;
    if !semi_major_axis.is_finite() {
        write_nan_first(out_state);
        return;
    }
    equinoc2eci_impl(&equ_adv, 6, 0.0, 0.0, out_state);
}

#[inline]
fn advance_equinoc_j2_simd(equinoc: &[f64x4; 6], delta_t: f64x4) -> [f64x4; 6] {
    let zero = f64x4::ZERO;
    let one = f64x4::ONE;
    let inf = INF_X4;
    let nan = NAN_X4;

    let [semi_major_axis, h_value, k_value, equinoctial_p, equinoctial_q, longitude] = *equinoc;

    let mut valid = semi_major_axis.simd_gt(zero)
        & semi_major_axis.abs().simd_lt(inf)
        & h_value.abs().simd_lt(inf)
        & k_value.abs().simd_lt(inf)
        & equinoctial_p.abs().simd_lt(inf)
        & equinoctial_q.abs().simd_lt(inf)
        & longitude.abs().simd_lt(inf)
        & delta_t.abs().simd_lt(inf);

    let eccentricity_squared = h_value.mul_add(h_value, k_value * k_value);
    valid &= eccentricity_squared.simd_ge(zero) & eccentricity_squared.simd_lt(one);
    let one_minus_eccentricity_squared = one - eccentricity_squared;
    valid &= one_minus_eccentricity_squared.simd_gt(zero);

    let tan_half_inclination_squared =
        equinoctial_p.mul_add(equinoctial_p, equinoctial_q * equinoctial_q);
    let cos_inclination =
        (one - tan_half_inclination_squared) / (one + tan_half_inclination_squared);
    let orbit_parameter = semi_major_axis * one_minus_eccentricity_squared;
    valid &= orbit_parameter.simd_gt(zero);

    let mu = MU_X4;
    let re = RE_X4;
    let j2 = J2_X4;
    let three_halves = THREE_HALVES_X4;
    let three_quarters = THREE_QUARTERS_X4;
    let five = FIVE_X4;
    let three = THREE_X4;

    let mean_motion = (mu / (semi_major_axis * semi_major_axis * semi_major_axis)).sqrt();
    let re_over_orbit_parameter = re / orbit_parameter;
    let j2_factor = j2 * re_over_orbit_parameter * re_over_orbit_parameter;
    let cos_inclination_squared = cos_inclination * cos_inclination;
    let raan_rate = -three_halves * mean_motion * j2_factor * cos_inclination;
    let periapsis_rate =
        three_quarters * mean_motion * j2_factor * (five * cos_inclination_squared - one);
    let mean_anomaly_rate = mean_motion
        + three_quarters
            * mean_motion
            * j2_factor
            * one_minus_eccentricity_squared.sqrt()
            * (three * cos_inclination_squared - one);
    let longitude_rate = raan_rate + periapsis_rate + mean_anomaly_rate;

    let d_raan = raan_rate * delta_t;
    let d_peri_long = (raan_rate + periapsis_rate) * delta_t;
    let (sin_raan, cos_raan) = d_raan.sin_cos();
    let (sin_peri, cos_peri) = d_peri_long.sin_cos();

    [
        valid.select(semi_major_axis, nan),
        valid.select(h_value * cos_peri + k_value * sin_peri, nan),
        valid.select(k_value * cos_peri - h_value * sin_peri, nan),
        valid.select(equinoctial_p * cos_raan + equinoctial_q * sin_raan, nan),
        valid.select(equinoctial_q * cos_raan - equinoctial_p * sin_raan, nan),
        valid.select(mod2pi_simd(longitude + longitude_rate * delta_t), nan),
    ]
}

#[inline]
pub fn advance_equinoc_j2_batch_block4(
    equinoc_block: &[f64; 24],
    tofs: &[f64; 4],
    out_equ: &mut [f64; 24],
) {
    {
        let equ = transpose_equ_aos_to_soa(equinoc_block);
        let adv = advance_equinoc_j2_simd(&equ, f64x4::new(*tofs));
        let [semi_major_axis, ..] = adv;
        if semi_major_axis
            .to_array()
            .iter()
            .all(|value| value.is_finite())
        {
            transpose_equ_soa_to_aos(&adv, out_equ);
            return;
        }
    }

    for ((equinoc_state, out_state), &tof) in equinoc_block
        .chunks_exact(6)
        .zip(out_equ.chunks_exact_mut(6))
        .zip(tofs)
    {
        advance_equinoc_j2_impl(equinoc_state, tof, out_state);
    }
}

#[inline]
pub fn equinoc_prop_j2_batch_block4(
    equinoc_block: &[f64; 24],
    tofs: &[f64; 4],
    out: &mut [f64; 24],
) {
    {
        let equ = transpose_equ_aos_to_soa(equinoc_block);
        let adv = advance_equinoc_j2_simd(&equ, f64x4::new(*tofs));
        let [semi_major_axis, ..] = adv;
        if semi_major_axis
            .to_array()
            .iter()
            .all(|value| value.is_finite())
        {
            let eci = equinoc2eci_simd(&adv, f64x4::ZERO, f64x4::ZERO);
            transpose_eci_soa_to_aos(&eci, out);
            return;
        }
    }

    for ((equinoc_state, out_state), &tof) in equinoc_block
        .chunks_exact(6)
        .zip(out.chunks_exact_mut(6))
        .zip(tofs)
    {
        equinoc_prop_j2_from_impl(equinoc_state, tof, out_state);
    }
}

#[inline]
pub fn equinoc_prop_j2_step_simd4(
    equinoc: &[f64; 6],
    t_vals: &[f64; 4],
    t0: f64,
    out: &mut [f64; 24],
) {
    let [semi_major_axis, h_value, k_value, equinoctial_p, equinoctial_q, longitude] = *equinoc;
    let [time0, time1, time2, time3] = *t_vals;
    let equinoc_block = [
        semi_major_axis,
        h_value,
        k_value,
        equinoctial_p,
        equinoctial_q,
        longitude,
        semi_major_axis,
        h_value,
        k_value,
        equinoctial_p,
        equinoctial_q,
        longitude,
        semi_major_axis,
        h_value,
        k_value,
        equinoctial_p,
        equinoctial_q,
        longitude,
        semi_major_axis,
        h_value,
        k_value,
        equinoctial_p,
        equinoctial_q,
        longitude,
    ];
    let tofs = [time0 - t0, time1 - t0, time2 - t0, time3 - t0];
    equinoc_prop_j2_batch_block4(&equinoc_block, &tofs, out);
}

#[inline]
pub fn equinoc_prop_j2_step_impl(equinoc: &[f64], t_vals: &[f64], t0: f64, out: &mut [f64]) {
    let n = t_vals.len();
    let Some(output_len) = n.checked_mul(6) else {
        return;
    };
    if n == 0 || out.len() < output_len {
        return;
    }
    let Some(equinoc_state) = six_values(equinoc) else {
        return;
    };
    let Some(out_prefix) = out.get_mut(..output_len) else {
        return;
    };

    let mut time_blocks = t_vals.chunks_exact(4);
    let mut out_blocks = out_prefix.chunks_exact_mut(24);
    for (time_block, out_block) in time_blocks.by_ref().zip(out_blocks.by_ref()) {
        let (Ok(time_array), Ok(out_array)) = (
            <&[f64; 4]>::try_from(time_block),
            <&mut [f64; 24]>::try_from(out_block),
        ) else {
            return;
        };
        equinoc_prop_j2_step_simd4(&equinoc_state, time_array, t0, out_array);
    }

    for (&time, out_state) in time_blocks
        .remainder()
        .iter()
        .zip(out_blocks.into_remainder().chunks_exact_mut(6))
    {
        equinoc_prop_j2_from_impl(&equinoc_state, time - t0, out_state);
    }
}

#[inline]
pub fn equinoc_prop_j2_batch_impl(equinoc_matrix: &[f64], tofs: &[f64], out: &mut [f64]) {
    let _ = equinoc_prop_j2_batch_profiled_impl(equinoc_matrix, tofs, out);
}

/// Propagate each equinoctial row at one common offset through the scalar J2
/// authority. Unlike the SIMD batch path, every output is bit-identical to a
/// direct [`equinoc_prop_j2_from_impl`] call.
#[must_use]
pub fn equinoc_prop_j2_batch_exact_at_impl(
    equinoctial: &[[f64; 6]],
    tof: f64,
    out: &mut [[f64; 6]],
) -> bool {
    if equinoctial.len() != out.len() || !tof.is_finite() {
        return false;
    }

    #[cfg(feature = "parallel")]
    equinoctial
        .par_iter()
        .zip(out.par_iter_mut())
        .for_each(|(input, output)| equinoc_prop_j2_from_impl(input, tof, output));

    #[cfg(not(feature = "parallel"))]
    equinoctial
        .iter()
        .zip(out.iter_mut())
        .for_each(|(input, output)| equinoc_prop_j2_from_impl(input, tof, output));

    true
}

/// Same computation as [`equinoc_prop_j2_batch_impl`], returning whether its
/// top-level Rayon dispatch branch ran. Diagnostic only; output order/math are unchanged.
#[inline]
pub fn equinoc_prop_j2_batch_profiled_impl(
    equinoc_matrix: &[f64],
    tofs: &[f64],
    out: &mut [f64],
) -> bool {
    let n = tofs.len();
    let Some(needed) = n.checked_mul(6) else {
        return false;
    };
    if n == 0 {
        return false;
    }
    if equinoc_matrix.len() < needed || out.len() < needed {
        return false;
    }
    let Some(equinoc_values) = equinoc_matrix.get(..needed) else {
        return false;
    };
    let Some(out_values) = out.get_mut(..needed) else {
        return false;
    };

    let Some(simd_state_count) = n.checked_sub(n % 4) else {
        return false;
    };
    let Some(simd_value_count) = simd_state_count.checked_mul(6) else {
        return false;
    };

    #[cfg(feature = "parallel")]
    if n >= PROP_BATCH_THRESHOLD && simd_state_count > 0 && rayon::current_thread_index().is_none()
    {
        let (Some(equinoc_simd), Some(tofs_simd), Some(out_simd)) = (
            equinoc_values.get(..simd_value_count),
            tofs.get(..simd_state_count),
            out_values.get_mut(..simd_value_count),
        ) else {
            return false;
        };
        let _ = nd_sched::init_global_pool(None);
        equinoc_simd
            .par_chunks_exact(24)
            .zip(tofs_simd.par_chunks_exact(4))
            .zip(out_simd.par_chunks_exact_mut(24))
            .with_min_len(PROP_BATCH_MIN_LEN / 4)
            .for_each(|((equinoc_chunk, tof_chunk), out_chunk)| {
                let (Ok(equinoc_block), Ok(tof_block), Ok(out_block)) = (
                    <&[f64; 24]>::try_from(equinoc_chunk),
                    <&[f64; 4]>::try_from(tof_chunk),
                    <&mut [f64; 24]>::try_from(out_chunk),
                ) else {
                    return;
                };
                equinoc_prop_j2_batch_block4(equinoc_block, tof_block, out_block);
            });

        let (Some(equinoc_tail), Some(tofs_tail), Some(out_tail)) = (
            equinoc_values.get(simd_value_count..),
            tofs.get(simd_state_count..),
            out_values.get_mut(simd_value_count..),
        ) else {
            return false;
        };
        for ((equinoc_state, out_state), &tof) in equinoc_tail
            .chunks_exact(6)
            .zip(out_tail.chunks_exact_mut(6))
            .zip(tofs_tail)
        {
            equinoc_prop_j2_from_impl(equinoc_state, tof, out_state);
        }
        return true;
    }

    let mut equinoc_blocks = equinoc_values.chunks_exact(24);
    let mut tof_blocks = tofs.chunks_exact(4);
    let mut out_blocks = out_values.chunks_exact_mut(24);
    for ((equinoc_chunk, tof_chunk), out_chunk) in equinoc_blocks
        .by_ref()
        .zip(tof_blocks.by_ref())
        .zip(out_blocks.by_ref())
    {
        let (Ok(equinoc_block), Ok(tof_block), Ok(out_block)) = (
            <&[f64; 24]>::try_from(equinoc_chunk),
            <&[f64; 4]>::try_from(tof_chunk),
            <&mut [f64; 24]>::try_from(out_chunk),
        ) else {
            return false;
        };
        equinoc_prop_j2_batch_block4(equinoc_block, tof_block, out_block);
    }

    for ((equinoc_state, out_state), &tof) in equinoc_blocks
        .remainder()
        .chunks_exact(6)
        .zip(out_blocks.into_remainder().chunks_exact_mut(6))
        .zip(tof_blocks.remainder())
    {
        equinoc_prop_j2_from_impl(equinoc_state, tof, out_state);
    }
    false
}

#[inline]
pub fn eci2equinoc_simd4(eci_block: &[f64; 24], t: f64, t0: f64, equ_block: &mut [f64; 24]) {
    {
        let (rx, ry, rz, vx, vy, vz) = transpose_eci_aos_to_soa(eci_block);
        let time_vector = f64x4::splat(t);
        let reference_time_vector = f64x4::splat(t0);
        let result = eci2equinoc_simd(rx, ry, rz, vx, vy, vz, time_vector, reference_time_vector);
        transpose_equ_soa_to_aos(&result, equ_block);
    }
}

#[inline]
pub fn equinoc2eci_simd4(equ_block: &[f64; 24], t: f64, t0: f64, eci_block: &mut [f64; 24]) {
    {
        let elems = transpose_equ_aos_to_soa(equ_block);
        let time_vector = f64x4::splat(t);
        let reference_time_vector = f64x4::splat(t0);
        let result = equinoc2eci_simd(&elems, time_vector, reference_time_vector);
        transpose_eci_soa_to_aos(&result, eci_block);
    }
}

#[inline]
pub fn kep2eci_simd4(kep_block: &[f64; 24], t0: f64, t: f64, eci_block: &mut [f64; 24]) {
    // Keep as scalar loop - kep2eci has complex logic and parameter handling
    // that doesn't benefit significantly from SIMD
    for (kep_state, eci_state) in kep_block.chunks_exact(6).zip(eci_block.chunks_exact_mut(6)) {
        kep2eci_impl(kep_state, false, t0, t, false, eci_state);
    }
}

/// SIMD16 variants for processing 16 states (96 f64s) at once for better vectorization
#[inline]
pub fn eci2equinoc_simd16(eci_block: &[f64; 96], t: f64, t0: f64, equ_block: &mut [f64; 96]) {
    for (eci_slice, equ_slice) in eci_block
        .chunks_exact(24)
        .zip(equ_block.chunks_exact_mut(24))
    {
        let Some(eci_chunk) = <&[f64; 24]>::try_from(eci_slice).ok() else {
            return;
        };
        let Some(equ_chunk) = <&mut [f64; 24]>::try_from(equ_slice).ok() else {
            return;
        };
        eci2equinoc_simd4(eci_chunk, t, t0, equ_chunk);
    }
}

/// Tests for core functionality (f64-only, always run)
#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ToPrimitive;

    const EPS: f64 = 1e-12;

    fn test_value<T: Copy>(values: &[T], index: usize) -> T {
        let Some(value) = values.get(index) else {
            panic!("test index {index} outside length {}", values.len());
        };
        *value
    }

    fn test_range<T>(values: &[T], start: usize, len: usize) -> &[T] {
        let end = start.saturating_add(len);
        assert!(end <= values.len(), "test range outside storage");
        values.get(start..end).unwrap_or(&[])
    }

    fn test_range_mut<T>(values: &mut [T], start: usize, len: usize) -> &mut [T] {
        let end = start.saturating_add(len);
        let length = values.len();
        assert!(end <= length, "test range outside storage");
        values.get_mut(start..end).unwrap_or(&mut [])
    }

    // ==================== f64 Function Tests ====================

    #[test]
    fn test_norm3() {
        let v = [3.0, 4.0, 0.0];
        let n = norm3(&v);
        assert!((n - 5.0).abs() < EPS);
    }

    #[test]
    fn test_dot3() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let d = dot3(&a, &b);
        assert!((d - 32.0).abs() < EPS); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_cross3() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = cross3(&a, &b);
        assert!((c[0] - 0.0).abs() < EPS);
        assert!((c[1] - 0.0).abs() < EPS);
        assert!((c[2] - 1.0).abs() < EPS); // i x j = k
    }

    #[test]
    fn test_mod2pi_f64() {
        let angle = 3.0 * std::f64::consts::PI;
        let m = mod2pi(angle);
        assert!((m - std::f64::consts::PI).abs() < EPS);
    }

    #[test]
    fn test_eci2equinoc_roundtrip_f64() {
        // Create a simple circular orbit in ECI
        let r = 7000.0; // km
        let v = (MU / r).sqrt(); // circular velocity
        let eci = [r, 0.0, 0.0, 0.0, v, 0.0];
        let mut equ = [0.0; 6];
        let mut eci_back = [0.0; 6];

        eci2equinoc_impl(&eci, 6, 0.0, 0.0, &mut equ);
        equinoc2eci_impl(&equ, 6, 0.0, 0.0, &mut eci_back);

        for (i, (original, roundtrip)) in eci.into_iter().zip(eci_back).enumerate() {
            assert!(
                (original - roundtrip).abs() < 1e-9,
                "ECI roundtrip failed at index {i}: {original} vs {roundtrip}"
            );
        }
    }

    #[test]
    fn test_kep2equinoc_roundtrip_f64() {
        let kep = [7000.0, 0.1, 0.4, 1.0, 0.3, 0.2];
        let mut equ = [0.0; 6];
        let mut kep_back = [0.0; 6];

        kep2equinoc_impl(&kep, false, false, &mut equ);
        equinoc2kep_impl(&equ, false, false, &mut kep_back);

        // Compare SMA and eccentricity directly
        assert!((kep[0] - kep_back[0]).abs() < 1e-9);
        assert!((kep[1] - kep_back[1]).abs() < 1e-9);

        // Compare angular elements modulo 2π
        let angle_diff = |a: f64, b: f64| {
            let mut d = mod2pi(a - b);
            if d > std::f64::consts::PI {
                d = std::f64::consts::TAU - d;
            }
            d
        };
        assert!(angle_diff(kep[2], kep_back[2]) < 1e-9);
        assert!(angle_diff(kep[3], kep_back[3]) < 1e-9);
        assert!(angle_diff(kep[4], kep_back[4]) < 1e-9);
        assert!(angle_diff(kep[5], kep_back[5]) < 1e-9);
    }

    #[test]
    fn kep2eci_nontrivial_rotation_uses_sine_in_second_row() {
        let semi_major_axis = 7_000.0;
        let inclination = std::f64::consts::FRAC_PI_3;
        let raan = 0.7;
        let argument_periapsis = 0.4;
        let true_anomaly = std::f64::consts::FRAC_PI_2;
        let state = [
            semi_major_axis,
            0.0,
            inclination,
            raan,
            argument_periapsis,
            true_anomaly,
        ];
        let mut eci = [f64::NAN; 6];

        kep2eci_impl(&state, false, 0.0, 0.0, true, &mut eci);

        let (sin_raan, cos_raan) = raan.sin_cos();
        let (sin_argument_periapsis, cos_argument_periapsis) = argument_periapsis.sin_cos();
        let expected_y = semi_major_axis
            * ((-sin_raan).mul_add(
                sin_argument_periapsis,
                cos_raan * cos_argument_periapsis * inclination.cos(),
            ));
        let [_, actual_y, _, _, _, _] = eci;

        assert!(
            (actual_y - expected_y).abs() < EPS,
            "nontrivial Kepler-to-ECI y rotation: {actual_y} vs {expected_y}"
        );
    }

    // ========== SIMD Equinoctial Tests ==========

    #[test]
    fn test_mod2pi_simd() {
        use wide::f64x4;

        // Test various angles
        let angles = f64x4::new([
            0.0,
            3.0 * std::f64::consts::PI,
            -std::f64::consts::PI,
            7.0 * std::f64::consts::PI,
        ]);

        let result = mod2pi_simd(angles);
        let result_arr = result.to_array();

        // Compare with scalar version
        let scalar_results = [
            mod2pi(0.0),
            mod2pi(3.0 * std::f64::consts::PI),
            mod2pi(-std::f64::consts::PI),
            mod2pi(7.0 * std::f64::consts::PI),
        ];

        for (i, (simd_value, scalar_value)) in
            result_arr.into_iter().zip(scalar_results).enumerate()
        {
            assert!(
                (simd_value - scalar_value).abs() < 1e-12,
                "mod2pi_simd failed at lane {i}: {simd_value} vs {scalar_value}"
            );
        }
    }

    #[test]
    fn test_clamp_eccentricity_simd() {
        use wide::f64x4;

        let e_vals = f64x4::new([-0.5, 0.3, 0.999_999, 1.5]);
        let result = clamp_eccentricity_simd(e_vals);
        let result_arr = result.to_array();

        // Compare with scalar version
        let scalar_results = [
            clamp_eccentricity(-0.5),
            clamp_eccentricity(0.3),
            clamp_eccentricity(0.999_999),
            clamp_eccentricity(1.5),
        ];

        for (i, (simd_value, scalar_value)) in
            result_arr.into_iter().zip(scalar_results).enumerate()
        {
            assert!(
                (simd_value - scalar_value).abs() < 1e-12,
                "clamp_eccentricity_simd failed at lane {i}: {simd_value} vs {scalar_value}"
            );
        }
    }

    #[test]
    fn test_equinoc2eci_simd() {
        use wide::f64x4;

        // Create 4 test states with different orbital characteristics
        // State 1: Circular equatorial orbit
        let a1 = 7000.0;
        let e1 = 0.0;
        let i1 = 0.0;
        let raan1 = 0.0;
        let omega1 = 0.0;
        let m1 = 0.0;

        // State 2: Elliptical inclined orbit
        let a2 = 8000.0;
        let e2 = 0.1;
        let i2 = 0.5;
        let raan2 = 0.3;
        let omega2 = 0.7;
        let m2 = std::f64::consts::PI / 4.0;

        // State 3: Higher eccentricity
        let a3 = 9000.0;
        let e3 = 0.3;
        let i3 = 1.0;
        let raan3 = 1.5;
        let omega3 = 2.0;
        let m3 = std::f64::consts::PI;

        // State 4: Near-circular polar orbit
        let a4 = 7500.0;
        let e4 = 0.001;
        let i4 = std::f64::consts::PI / 2.0;
        let raan4 = 0.0;
        let omega4 = 0.0;
        let m4 = std::f64::consts::PI / 2.0;

        // Convert Keplerian to equinoctial for each state
        let mut equ1 = [0.0; 6];
        let mut equ2 = [0.0; 6];
        let mut equ3 = [0.0; 6];
        let mut equ4 = [0.0; 6];

        let kep1 = [a1, e1, i1, raan1, omega1, m1];
        let kep2 = [a2, e2, i2, raan2, omega2, m2];
        let kep3 = [a3, e3, i3, raan3, omega3, m3];
        let kep4 = [a4, e4, i4, raan4, omega4, m4];

        // First convert to ECI, then to equinoctial
        let mut eci1 = [0.0; 6];
        let mut eci2 = [0.0; 6];
        let mut eci3 = [0.0; 6];
        let mut eci4 = [0.0; 6];

        kep2eci_impl(&kep1, false, 0.0, 0.0, false, &mut eci1);
        kep2eci_impl(&kep2, false, 0.0, 0.0, false, &mut eci2);
        kep2eci_impl(&kep3, false, 0.0, 0.0, false, &mut eci3);
        kep2eci_impl(&kep4, false, 0.0, 0.0, false, &mut eci4);

        eci2equinoc_impl(&eci1, 6, 0.0, 0.0, &mut equ1);
        eci2equinoc_impl(&eci2, 6, 0.0, 0.0, &mut equ2);
        eci2equinoc_impl(&eci3, 6, 0.0, 0.0, &mut equ3);
        eci2equinoc_impl(&eci4, 6, 0.0, 0.0, &mut equ4);

        // Pack into SIMD vectors
        let elems = [
            f64x4::new([equ1[0], equ2[0], equ3[0], equ4[0]]), // a
            f64x4::new([equ1[1], equ2[1], equ3[1], equ4[1]]), // h
            f64x4::new([equ1[2], equ2[2], equ3[2], equ4[2]]), // k
            f64x4::new([equ1[3], equ2[3], equ3[3], equ4[3]]), // p
            f64x4::new([equ1[4], equ2[4], equ3[4], equ4[4]]), // q
            f64x4::new([equ1[5], equ2[5], equ3[5], equ4[5]]), // lam
        ];

        let t = f64x4::splat(0.0);
        let t0 = f64x4::splat(0.0);

        // Call SIMD kernel
        let result = equinoc2eci_simd(&elems, t, t0);

        // Compare with scalar results
        let mut eci1_check = [0.0; 6];
        let mut eci2_check = [0.0; 6];
        let mut eci3_check = [0.0; 6];
        let mut eci4_check = [0.0; 6];

        equinoc2eci_impl(&equ1, 6, 0.0, 0.0, &mut eci1_check);
        equinoc2eci_impl(&equ2, 6, 0.0, 0.0, &mut eci2_check);
        equinoc2eci_impl(&equ3, 6, 0.0, 0.0, &mut eci3_check);
        equinoc2eci_impl(&equ4, 6, 0.0, 0.0, &mut eci4_check);

        let scalar_results = [
            [eci1_check[0], eci2_check[0], eci3_check[0], eci4_check[0]],
            [eci1_check[1], eci2_check[1], eci3_check[1], eci4_check[1]],
            [eci1_check[2], eci2_check[2], eci3_check[2], eci4_check[2]],
            [eci1_check[3], eci2_check[3], eci3_check[3], eci4_check[3]],
            [eci1_check[4], eci2_check[4], eci3_check[4], eci4_check[4]],
            [eci1_check[5], eci2_check[5], eci3_check[5], eci4_check[5]],
        ];

        // Compare results
        for (component, (simd_component, scalar_component)) in
            result.into_iter().zip(scalar_results).enumerate()
        {
            let simd_values = simd_component.to_array();
            for (lane, (simd_value, scalar_value)) in
                simd_values.into_iter().zip(scalar_component).enumerate()
            {
                let diff = (simd_value - scalar_value).abs();
                assert!(
                    diff < 1e-9,
                    "equinoc2eci_simd failed at component {component} lane {lane}: SIMD={simd_value} scalar={scalar_value} diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_eci2equinoc_simd_matches_scalar() {
        use wide::f64x4;

        // Create 4 test ECI states with different orbital characteristics
        // State 1: ~400km circular, 45 deg inclination
        let r1 = 6778.0;
        let v1 = (MU / r1).sqrt();
        let eci1 = [r1, 0.0, 0.0, 0.0, v1 * 0.707, v1 * 0.707];

        // State 2: Elliptical, different RAAN
        let r2 = 7000.0;
        let v2 = (MU / r2).sqrt() * 0.95; // slightly elliptical
        let eci2 = [r2 * 0.707, r2 * 0.707, 0.0, -v2 * 0.5, v2 * 0.5, v2 * 0.707];

        // State 3: Polar orbit
        let r3 = 7200.0;
        let v3 = (MU / r3).sqrt();
        let eci3 = [0.0, r3, 0.0, -v3, 0.0, 0.0];

        // State 4: Equatorial orbit
        let r4 = 6900.0;
        let v4 = (MU / r4).sqrt();
        let eci4 = [r4, 0.0, 0.0, 0.0, v4, 0.0];

        // Pack into SIMD vectors
        let rx = f64x4::new([eci1[0], eci2[0], eci3[0], eci4[0]]);
        let ry = f64x4::new([eci1[1], eci2[1], eci3[1], eci4[1]]);
        let rz = f64x4::new([eci1[2], eci2[2], eci3[2], eci4[2]]);
        let vx = f64x4::new([eci1[3], eci2[3], eci3[3], eci4[3]]);
        let vy = f64x4::new([eci1[4], eci2[4], eci3[4], eci4[4]]);
        let vz = f64x4::new([eci1[5], eci2[5], eci3[5], eci4[5]]);
        let t = f64x4::ZERO;
        let t0 = f64x4::ZERO;

        // Call SIMD kernel
        let simd_result = eci2equinoc_simd(rx, ry, rz, vx, vy, vz, t, t0);

        // Compare with scalar results
        let eci_states = [eci1, eci2, eci3, eci4];
        for (i, state) in eci_states.iter().enumerate() {
            let mut scalar_out = [0.0f64; 6];
            eci2equinoc_impl(state, 6, 0.0, 0.0, &mut scalar_out);

            for (j, (simd_vector, scalar_val)) in simd_result.iter().zip(scalar_out).enumerate() {
                let simd_val = test_value(&simd_vector.to_array(), i);

                // Allow for numerical differences in angle wrapping
                let err = (simd_val - scalar_val).abs();
                let err_wrapped = (err - TWO_PI).abs().min(err);

                assert!(
                    err_wrapped < 1e-6,
                    "State {i} element {j} mismatch: scalar={scalar_val}, simd={simd_val}, err={err_wrapped}"
                );
            }
        }
    }

    #[test]
    fn test_equinoc_prop_step_impl_simd_tail_matches_scalar() {
        let equinoc = [7000.0, 0.001, 0.002, 0.10, 0.20, 0.30];
        let t_vals = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];
        let mut out_simd = vec![0.0; t_vals.len() * 6];
        let mut out_scalar = vec![0.0; t_vals.len() * 6];

        equinoc_prop_step_impl(&equinoc, &t_vals, 0.0, &mut out_simd);
        for (idx, &t) in t_vals.iter().enumerate() {
            let base = idx.saturating_mul(6);
            equinoc2eci_impl_f64(
                &equinoc,
                6,
                t,
                0.0,
                test_range_mut(&mut out_scalar, base, 6),
            );
        }

        for (idx, (actual, expected)) in out_simd.iter().zip(out_scalar.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= 1.0e-8,
                "step component {idx} mismatch: simd={actual} scalar={expected} diff={diff}"
            );
        }

        let mut add_out = vec![1.0; t_vals.len() * 6];
        equinoc_prop_step_add_to_impl(&equinoc, &t_vals, 0.0, &mut add_out);
        for (idx, (actual, expected)) in add_out.iter().zip(out_scalar.iter()).enumerate() {
            let diff = (actual - (expected + 1.0)).abs();
            assert!(
                diff <= 1.0e-8,
                "add-to component {idx} mismatch: actual={actual} expected={} diff={diff}",
                expected + 1.0
            );
        }
    }

    #[test]
    fn test_advance_equinoc_j2_batch_block4_matches_scalar() {
        let equ_states = [
            [7000.0, 0.001, 0.002, 0.10, 0.20, 0.30],
            [7020.0, 0.002, 0.003, 0.40, 0.50, 0.60],
            [7040.0, 0.003, 0.004, 0.70, 0.80, 0.90],
            [7060.0, 0.004, 0.005, 1.00, 1.10, 1.20],
        ];
        let tofs = [120.0, 600.0, 1800.0, 3600.0];
        let mut equ_block = [0.0_f64; 24];
        for (target, equ) in equ_block.chunks_exact_mut(6).zip(&equ_states) {
            target.copy_from_slice(equ);
        }

        let mut block = [0.0_f64; 24];
        advance_equinoc_j2_batch_block4(&equ_block, &tofs, &mut block);

        for (lane, (equ, tof)) in equ_states.iter().zip(tofs).enumerate() {
            let mut scalar = [0.0_f64; 6];
            advance_equinoc_j2_impl(equ, tof, &mut scalar);
            let base = lane.saturating_mul(6);
            for (component, (block_value, scalar_value)) in test_range(&block, base, 6)
                .iter()
                .copied()
                .zip(scalar)
                .enumerate()
            {
                let diff = (block_value - scalar_value).abs();
                assert!(
                    diff <= 1e-12,
                    "lane={lane} component={component} block={block_value} scalar={scalar_value} diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_equinoc_prop_j2_step_simd4_matches_scalar() {
        let equinoc = [7000.0, 0.001, 0.002, 0.10, 0.20, 0.30];
        let t_vals = [120.0, 600.0, 1800.0, 3600.0];
        let t0 = 45.0;
        let mut block = [0.0_f64; 24];
        equinoc_prop_j2_step_simd4(&equinoc, &t_vals, t0, &mut block);

        for (lane, &t) in t_vals.iter().enumerate() {
            let mut scalar = [0.0_f64; 6];
            equinoc_prop_j2_from_impl(&equinoc, t - t0, &mut scalar);
            let base = lane.saturating_mul(6);
            for (component, (block_value, scalar_value)) in test_range(&block, base, 6)
                .iter()
                .copied()
                .zip(scalar)
                .enumerate()
            {
                let diff = (block_value - scalar_value).abs();
                assert!(
                    diff <= 1e-8,
                    "lane={lane} component={component} block={block_value} scalar={scalar_value} diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_equinoc_prop_j2_batch_block4_matches_scalar() {
        let equ_states = [
            [7000.0, 0.001, 0.002, 0.10, 0.20, 0.30],
            [7020.0, 0.002, 0.003, 0.40, 0.50, 0.60],
            [7040.0, 0.003, 0.004, 0.70, 0.80, 0.90],
            [7060.0, 0.004, 0.005, 1.00, 1.10, 1.20],
        ];
        let tofs = [120.0, 600.0, 1800.0, 3600.0];
        let mut equ_block = [0.0_f64; 24];
        for (target, equ) in equ_block.chunks_exact_mut(6).zip(&equ_states) {
            target.copy_from_slice(equ);
        }

        let mut block = [0.0_f64; 24];
        equinoc_prop_j2_batch_block4(&equ_block, &tofs, &mut block);

        for (lane, (equ, tof)) in equ_states.iter().zip(tofs).enumerate() {
            let mut scalar = [0.0_f64; 6];
            equinoc_prop_j2_from_impl(equ, tof, &mut scalar);
            let base = lane.saturating_mul(6);
            for (component, (block_value, scalar_value)) in test_range(&block, base, 6)
                .iter()
                .copied()
                .zip(scalar)
                .enumerate()
            {
                let diff = (block_value - scalar_value).abs();
                assert!(
                    diff <= 1e-8,
                    "lane={lane} component={component} block={block_value} scalar={scalar_value} diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_equinoc_prop_j2_batch_impl_matches_scalar_with_tails() {
        for n in [1_usize, 3, 4, 5, 9, 17] {
            assert_equinoc_prop_j2_batch_impl_matches_scalar(n);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_equinoc_prop_j2_batch_impl_parallel_tail_matches_scalar() {
        assert_equinoc_prop_j2_batch_impl_matches_scalar(PROP_BATCH_THRESHOLD + 1);
    }

    #[test]
    fn test_equinoc_prop_j2_batch_exact_at_impl_is_bit_exact_and_fail_closed() {
        let equinoctial = [
            [7000.0, 0.001, 0.002, 0.10, 0.20, 0.30],
            [7200.0, -0.002, 0.003, 0.12, 0.18, 1.30],
            [7400.0, 0.004, -0.001, 0.14, 0.16, 2.30],
        ];
        let tof = 86_400.0;
        let mut batch = [[0.0; 6]; 3];
        assert!(equinoc_prop_j2_batch_exact_at_impl(
            &equinoctial,
            tof,
            &mut batch
        ));
        for (input, actual) in equinoctial.iter().zip(batch) {
            let mut expected = [0.0; 6];
            equinoc_prop_j2_from_impl(input, tof, &mut expected);
            assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
        }

        let mut wrong_length = [[0.0; 6]; 2];
        assert!(!equinoc_prop_j2_batch_exact_at_impl(
            &equinoctial,
            tof,
            &mut wrong_length
        ));
        assert!(!equinoc_prop_j2_batch_exact_at_impl(
            &equinoctial,
            f64::NAN,
            &mut batch
        ));
    }

    fn assert_equinoc_prop_j2_batch_impl_matches_scalar(n: usize) {
        let storage_length = n.saturating_mul(6);
        let mut equinoc_matrix = vec![0.0_f64; storage_length];
        let mut tofs = vec![0.0_f64; n];
        for (idx, tof) in tofs.iter_mut().enumerate().take(n) {
            let base = idx.saturating_mul(6);
            let index_value = idx.to_f64().unwrap_or_default();
            test_range_mut(&mut equinoc_matrix, base, 6).copy_from_slice(&[
                index_value.mul_add(15.0, 7000.0),
                index_value.mul_add(1.0e-4, 0.001),
                index_value.mul_add(1.0e-4, 0.002),
                index_value.mul_add(0.02, 0.10),
                index_value.mul_add(0.02, 0.20),
                index_value.mul_add(0.03, 0.30),
            ]);
            *tof = index_value.mul_add(300.0, 60.0);
        }

        let mut batch = vec![0.0_f64; storage_length];
        equinoc_prop_j2_batch_impl(&equinoc_matrix, &tofs, &mut batch);

        for (idx, &tof) in tofs.iter().enumerate().take(n) {
            let base = idx.saturating_mul(6);
            let mut scalar = [0.0_f64; 6];
            equinoc_prop_j2_from_impl(test_range(&equinoc_matrix, base, 6), tof, &mut scalar);
            for (component, (batch_value, scalar_value)) in test_range(&batch, base, 6)
                .iter()
                .copied()
                .zip(scalar)
                .enumerate()
            {
                let diff = (batch_value - scalar_value).abs();
                assert!(
                    diff <= 1e-8,
                    "idx={idx} component={component} batch={batch_value} scalar={scalar_value} diff={diff}"
                );
            }
        }
    }

    /// `equinoc2eci_impl` (generic, line 2368) and `equinoc2eci_impl_f64`
    /// (line 2386) agree to a bounded ULP distance and are NOT bit-identical.
    ///
    /// This test documents that divergence; it must never be used to argue the
    /// two are interchangeable. They are deliberately different associations of
    /// the same algebra:
    ///
    /// * generic `q_squared.mul_add(one, one - p_squared)` is `q2 + (1 - p2)`;
    ///   `_f64` writes `1.0 + q2 - p2`, i.e. `(1 + q2) - p2`.
    /// * generic `eccentricity.mul_add(-eccentricity, one)` is a true FMA, one
    ///   rounding; `_f64` writes `1.0 - e*e`, two roundings.
    ///
    /// `rhs.rs` calls the GENERIC one, so routing that caller through the `_f64`
    /// body -- the obvious "de-duplication" -- moves every strict-HF arc's bits
    /// and every digest downstream of them.
    ///
    /// # Why the closeness assertion alone was not a guard
    ///
    /// Until 2026-08-10 this test asserted only `diff < 1e-10`. That is
    /// satisfied by two divergent bodies AND by one body called twice, so the
    /// unification it exists to prevent would have landed green. The
    /// divergence assertion below is the half that can actually fail: it reds
    /// the moment the two bodies start agreeing bit-for-bit, which is the
    /// signal that someone collapsed them.
    ///
    /// Measured on this corpus (3 element sets x 3 TOFs x 6 components = 54
    /// draws), macOS/Apple libm, 2026-08-10: worst distance 1 ULP, 5 of 54
    /// components differing. The bound below is 4 ULP rather than the measured
    /// 1 because the absolute values run through `sin`/`cos` and this repo has
    /// four standing Mac-vs-glibc bit axes; it is a guard against an algebraic
    /// change, not a libm pin.
    #[test]
    fn equinoc2eci_f64_and_generic_agree_closely_and_are_not_identical() {
        const MAX_ULPS: i64 = 4;
        let samples = [
            [7000.0, 0.001, 0.002, 0.05, -0.03, 0.1],
            [42164.0, 0.02, -0.01, 0.2, 0.15, 2.5],
            [12000.0, -0.05, 0.08, -0.1, 0.25, 5.8],
        ];
        let tofs = [0.0, 60.0, 1800.0];
        let mut differing = 0usize;
        let mut compared = 0usize;

        for elems in samples {
            for tof in tofs {
                let mut generic: [f64; 6] = [0.0; 6];
                let mut specialized: [f64; 6] = [0.0; 6];
                equinoc2eci_impl(&elems, 6, tof, 0.0, &mut generic);
                equinoc2eci_impl_f64(&elems, 6, tof, 0.0, &mut specialized);
                for (i, (generic_value, specialized_value)) in
                    generic.into_iter().zip(specialized).enumerate()
                {
                    compared = compared.saturating_add(1);
                    let diff = (generic_value - specialized_value).abs();
                    assert!(
                        diff < 1e-10,
                        "mismatch at idx {i} for tof {tof}: generic={generic_value} specialized={specialized_value} diff={diff}"
                    );
                    let ulps = i64::from_ne_bytes(generic_value.to_bits().to_ne_bytes())
                        .wrapping_sub(i64::from_ne_bytes(
                            specialized_value.to_bits().to_ne_bytes(),
                        ));
                    assert!(
                        ulps.abs() <= MAX_ULPS,
                        "idx {i} at tof {tof} is {ulps} ULP apart, past the {MAX_ULPS} ULP bound; \
                         one of the two bodies changed its algebra"
                    );
                    if ulps != 0 {
                        differing = differing.saturating_add(1);
                    }
                }
            }
        }

        assert!(
            differing > 0,
            "the generic and _f64 bodies agreed on all {compared} components. They are \
             DELIBERATELY different associations and this test exists to keep that visible: \
             if they were just unified, the strict-HF arc bits and every digest below them \
             moved with it. Re-pin deliberately or revert -- do not delete this assertion."
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn orbital_params_first_touch_uses_scheduler_pool() {
        const CHILD_ENV: &str = "NASA_DUST_ORBITAL_PARAMS_SCHED_POOL_CHILD";
        const CHILD_MARKER: &str = "NASA_DUST_ORBITAL_PARAMS_SCHED_POOL_CHILD_EXECUTED";
        const TEST_NAME: &str = "tests::orbital_params_first_touch_uses_scheduler_pool";

        if std::env::var_os(CHILD_ENV).is_none() {
            let output = std::process::Command::new(
                std::env::current_exe().expect("current Rust test executable"),
            )
            .args([TEST_NAME, "--exact", "--nocapture"])
            .env(CHILD_ENV, "1")
            .env("RUST_TEST_THREADS", "1")
            .output()
            .expect("spawn isolated orbital-params first-touch child test");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "child failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            assert!(
                stdout.contains(CHILD_MARKER),
                "child reported success without executing test\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            return;
        }

        println!("{CHILD_MARKER}");
        assert_eq!(nd_sched::init_global_pool(Some(2)), 2);
        let n = ORBITAL_PARAMS_PAR_THRESHOLD;
        let a = vec![7_000.0; n];
        let e = vec![0.01; n];
        let mut b = vec![0.0; n];
        let mut apogee = vec![0.0; n];
        let mut perigee = vec![0.0; n];
        let mut period = vec![0.0; n];
        orbital_params_batch_impl(&a, &e, MU, &mut b, &mut apogee, &mut perigee, &mut period);
        assert!(b.iter().all(|value| value.is_finite()));

        let mut worker_names = rayon::broadcast(|_| {
            std::thread::current()
                .name()
                .unwrap_or("<unnamed>")
                .to_owned()
        });
        worker_names.sort();
        assert_eq!(worker_names.len(), 2, "configured scheduler width must win");
        assert!(
            worker_names
                .iter()
                .all(|name| name.starts_with("nd-sched-")),
            "raw satpy batch first touch must use scheduler pool, got {worker_names:?}"
        );
    }
}

/// `DualVec` autodiff tests (only compiled with `autodiff` feature)
#[cfg(all(test, feature = "autodiff"))]
mod autodiff_tests {
    use super::*;
    use num_traits::Float;

    const EPS: f64 = 1e-12;
    const GRAD_EPS: f64 = 1e-6;

    fn test_range<T>(values: &[T], start: usize, len: usize) -> &[T] {
        let end = start.saturating_add(len);
        assert!(end <= values.len(), "test range outside storage");
        values.get(start..end).unwrap_or(&[])
    }

    fn test_range_mut<T>(values: &mut [T], start: usize, len: usize) -> &mut [T] {
        let end = start.saturating_add(len);
        let length = values.len();
        assert!(end <= length, "test range outside storage");
        values.get_mut(start..end).unwrap_or(&mut [])
    }

    // Helper to create DualVec with gradient in direction i
    fn dual_var(v: f64, i: usize) -> DualVec {
        let mut d = [0.0, 0.0, 0.0];
        assert!(i < d.len(), "dual direction outside gradient storage");
        if let Some(direction) = d.get_mut(i) {
            *direction = 1.0;
        }
        DualVec::new(v, nalgebra::Vector3::new(d[0], d[1], d[2]))
    }

    // Helper to compute finite difference gradient
    fn fd_grad<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    // ==================== DualVec Arithmetic Tests ====================

    #[test]
    fn test_dualvec_add() {
        let a = DualVec::new(2.0, nalgebra::Vector3::new(1.0, 0.0, 0.0));
        let b = DualVec::new(3.0, nalgebra::Vector3::new(0.0, 1.0, 0.0));
        let c = a + b;
        assert!((c.v() - 5.0).abs() < EPS);
        let d = c.d();
        assert!((d[0] - 1.0).abs() < EPS);
        assert!((d[1] - 1.0).abs() < EPS);
        assert!((d[2] - 0.0).abs() < EPS);
    }

    #[test]
    fn test_dualvec_sub() {
        let a = DualVec::new(5.0, nalgebra::Vector3::new(2.0, 1.0, 0.0));
        let b = DualVec::new(3.0, nalgebra::Vector3::new(1.0, 0.0, 0.0));
        let c = a - b;
        assert!((c.v() - 2.0).abs() < EPS);
        let d = c.d();
        assert!((d[0] - 1.0).abs() < EPS);
        assert!((d[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_dualvec_mul() {
        let a = DualVec::new(2.0, nalgebra::Vector3::new(1.0, 0.0, 0.0));
        let b = DualVec::new(3.0, nalgebra::Vector3::new(0.0, 1.0, 0.0));
        let c = a * b;
        assert!((c.v() - 6.0).abs() < EPS);
        let d = c.d();
        assert!((d[0] - 3.0).abs() < EPS);
        assert!((d[1] - 2.0).abs() < EPS);
    }

    #[test]
    fn test_dualvec_div() {
        let a = DualVec::new(6.0, nalgebra::Vector3::new(1.0, 0.0, 0.0));
        let b = DualVec::new(2.0, nalgebra::Vector3::new(0.0, 1.0, 0.0));
        let c = a / b;
        assert!((c.v() - 3.0).abs() < EPS);
        let d = c.d();
        assert!((d[0] - 0.5).abs() < EPS);
        assert!((d[1] - (-1.5)).abs() < EPS);
    }

    #[test]
    fn test_dualvec_neg() {
        let a = DualVec::new(3.0, nalgebra::Vector3::new(1.0, 2.0, 3.0));
        let b = -a;
        assert!((b.v() - (-3.0)).abs() < EPS);
        let d = b.d();
        assert!((d[0] - (-1.0)).abs() < EPS);
        assert!((d[1] - (-2.0)).abs() < EPS);
        assert!((d[2] - (-3.0)).abs() < EPS);
    }

    // ==================== DualVec Float Trait Tests ====================

    #[test]
    fn test_dualvec_sqrt() {
        let x = 4.0;
        let a = dual_var(x, 0);
        let b = a.sqrt();
        assert!((b.v() - 2.0).abs() < EPS);
        let expected_grad = fd_grad(f64::sqrt, x, 1e-8);
        assert!((b.d()[0] - expected_grad).abs() < GRAD_EPS);
    }

    #[test]
    fn test_dualvec_exp() {
        let x = 1.0;
        let a = dual_var(x, 0);
        let b = a.exp();
        assert!((b.v() - x.exp()).abs() < EPS);
        let expected_grad = fd_grad(f64::exp, x, 1e-8);
        assert!((b.d()[0] - expected_grad).abs() < GRAD_EPS);
    }

    #[test]
    fn test_dualvec_ln() {
        let x = 2.0;
        let a = dual_var(x, 0);
        let b = a.ln();
        assert!((b.v() - x.ln()).abs() < EPS);
        let expected_grad = fd_grad(f64::ln, x, 1e-8);
        assert!((b.d()[0] - expected_grad).abs() < GRAD_EPS);
    }

    #[test]
    fn test_dualvec_sin() {
        let x = 0.5;
        let a = dual_var(x, 0);
        let b = a.sin();
        assert!((b.v() - x.sin()).abs() < EPS);
        let expected_grad = fd_grad(f64::sin, x, 1e-8);
        assert!((b.d()[0] - expected_grad).abs() < GRAD_EPS);
    }

    #[test]
    fn test_dualvec_cos() {
        let x = 0.5;
        let a = dual_var(x, 0);
        let b = a.cos();
        assert!((b.v() - x.cos()).abs() < EPS);
        let expected_grad = fd_grad(f64::cos, x, 1e-8);
        assert!((b.d()[0] - expected_grad).abs() < GRAD_EPS);
    }

    #[test]
    fn test_dualvec_tan() {
        let x = 0.3;
        let a = dual_var(x, 0);
        let b = a.tan();
        assert!((b.v() - x.tan()).abs() < EPS);
        let expected_grad = fd_grad(f64::tan, x, 1e-8);
        assert!((b.d()[0] - expected_grad).abs() < GRAD_EPS);
    }

    #[test]
    fn test_dualvec_powi() {
        let x = 2.0;
        let a = dual_var(x, 0);
        let b = a.powi(3);
        assert!((b.v() - 8.0).abs() < EPS);
        let expected_grad = fd_grad(|t| t.powi(3), x, 1e-8);
        assert!((b.d()[0] - expected_grad).abs() < GRAD_EPS);
    }

    #[test]
    fn test_dualvec_atan2() {
        let y_value = 1.0;
        let x_value = 2.0;
        let y_dual = dual_var(y_value, 0);
        let x_dual = dual_var(x_value, 1);
        let result = y_dual.atan2(x_dual);
        assert!((result.v() - y_value.atan2(x_value)).abs() < EPS);
        let y_gradient = fd_grad(|value| value.atan2(x_value), y_value, 1e-8);
        let x_gradient = fd_grad(|value| y_value.atan2(value), x_value, 1e-8);
        assert!((result.d()[0] - y_gradient).abs() < GRAD_EPS);
        assert!((result.d()[1] - x_gradient).abs() < GRAD_EPS);
    }

    // ==================== Generic Function Tests with DualVec ====================

    #[test]
    fn test_norm3_dualvec() {
        let x_component = dual_var(3.0, 0);
        let y_component = dual_var(4.0, 1);
        let z_component = dual_var(0.0, 2);
        let vector = [x_component, y_component, z_component];
        let norm = norm3(&vector);
        assert!((norm.v() - 5.0).abs() < EPS);
        assert!((norm.d()[0] - 0.6).abs() < GRAD_EPS);
        assert!((norm.d()[1] - 0.8).abs() < GRAD_EPS);
        assert!((norm.d()[2] - 0.0).abs() < GRAD_EPS);
    }

    #[test]
    fn test_dot3_dualvec() {
        let a0 = dual_var(1.0, 0);
        let a1 = dual_var(2.0, 1);
        let a2 = dual_var(3.0, 2);
        let b0 = DualVec::constant(4.0);
        let b1 = DualVec::constant(5.0);
        let b2 = DualVec::constant(6.0);
        let d = dot3(&[a0, a1, a2], &[b0, b1, b2]);
        assert!((d.v() - 32.0).abs() < EPS);
        assert!((d.d()[0] - 4.0).abs() < GRAD_EPS);
        assert!((d.d()[1] - 5.0).abs() < GRAD_EPS);
        assert!((d.d()[2] - 6.0).abs() < GRAD_EPS);
    }

    #[test]
    fn test_cross3_dualvec() {
        let a0 = dual_var(1.0, 0);
        let a1 = dual_var(2.0, 1);
        let a2 = dual_var(3.0, 2);
        let b0 = DualVec::constant(4.0);
        let b1 = DualVec::constant(5.0);
        let b2 = DualVec::constant(6.0);
        let c = cross3(&[a0, a1, a2], &[b0, b1, b2]);
        assert!((c[0].v() - (-3.0)).abs() < EPS);
        assert!((c[1].v() - 6.0).abs() < EPS);
        assert!((c[2].v() - (-3.0)).abs() < EPS);
    }

    #[test]
    fn test_mod2pi_dualvec() {
        let angle = dual_var(std::f64::consts::PI / 4.0, 0);
        let m = mod2pi(angle);
        assert!((m.v() - std::f64::consts::PI / 4.0).abs() < EPS);
        assert!((m.d()[0] - 1.0).abs() < GRAD_EPS);
    }

    // ==================== Gradient Validation Tests ====================

    #[test]
    fn test_equinoc_prop_gradient_vs_fd() {
        let equ = [7000.0, 0.01, 0.0, 0.0, 0.0, 0.5];
        let tof = 1000.0;

        let tof_dual = DualVec::new(tof, nalgebra::Vector3::new(1.0, 0.0, 0.0));
        let equ_dual: [DualVec; 6] = [
            DualVec::constant(equ[0]),
            DualVec::constant(equ[1]),
            DualVec::constant(equ[2]),
            DualVec::constant(equ[3]),
            DualVec::constant(equ[4]),
            DualVec::constant(equ[5]),
        ];
        let mut out_dual = [DualVec::constant(0.0); 6];
        equinoc_prop_from_impl(&equ_dual, tof_dual, &mut out_dual);

        let h = 1e-6;
        let mut out_plus = [0.0; 6];
        let mut out_minus = [0.0; 6];
        equinoc_prop_from_impl(&equ, tof + h, &mut out_plus);
        equinoc_prop_from_impl(&equ, tof - h, &mut out_minus);

        for (i, ((dual_value, plus_value), minus_value)) in out_dual
            .iter()
            .zip(out_plus)
            .zip(out_minus)
            .take(3)
            .enumerate()
        {
            let dual_grad = dual_value.d()[0];
            let fd_grad = (plus_value - minus_value) / (2.0 * h);
            let err = (dual_grad - fd_grad).abs();
            let rel_err = err / fd_grad.abs().max(1e-10);
            assert!(
                rel_err < 0.01 || err < 1e-6,
                "Gradient mismatch at index {i}: DualVec={dual_grad:.6}, FD={fd_grad:.6}"
            );
        }
    }

    #[test]
    fn test_eci2equinoc_gradient_vs_fd() {
        let r = 7000.0;
        let v = (MU / r).sqrt();
        let eci = [r + 100.0, 200.0, 50.0, 0.1, v, 0.05];
        let t = 1000.0;
        let t0 = 0.0;
        let h = 1e-6;

        let mut eci_plus = eci;
        let mut eci_minus = eci;
        eci_plus[0] += h;
        eci_minus[0] -= h;
        let mut equ_plus = [0.0; 6];
        let mut equ_minus = [0.0; 6];
        eci2equinoc_impl(&eci_plus, 6, t, t0, &mut equ_plus);
        eci2equinoc_impl(&eci_minus, 6, t, t0, &mut equ_minus);

        let eci_dual: [DualVec; 6] = [
            DualVec::new(eci[0], nalgebra::Vector3::new(1.0, 0.0, 0.0)),
            DualVec::constant(eci[1]),
            DualVec::constant(eci[2]),
            DualVec::constant(eci[3]),
            DualVec::constant(eci[4]),
            DualVec::constant(eci[5]),
        ];
        let mut equ_dual = [DualVec::constant(0.0); 6];
        eci2equinoc_impl(
            &eci_dual,
            6,
            DualVec::constant(t),
            DualVec::constant(t0),
            &mut equ_dual,
        );

        let dual_grad = equ_dual[0].d()[0];
        let fd_grad = (equ_plus[0] - equ_minus[0]) / (2.0 * h);
        let err = (dual_grad - fd_grad).abs();
        let rel_err = err / fd_grad.abs().max(1e-10);
        assert!(rel_err < 0.01 || err < 1e-6);
    }

    // ========== SIMD Kepler Tests ==========

    #[test]
    fn test_kepler_simd_matches_scalar() {
        use wide::f64x4;

        // Test various eccentricities and mean anomalies.
        //
        // The M values are all in (pi, 2pi) DELIBERATELY. The shipped version
        // of this test used [0.5, 1.0, 2.0, 3.0] against e = [0.1, 0.5, 0.8,
        // 0.99], and every one of those lanes stepped around the region where
        // `solve_kepler_e_simd` was broken: the +pi seed only diverges once the
        // wrapped mean anomaly is NEGATIVE, which needs M > pi, and the one
        // high-e lane that could have reached it sat at M = 3.0 < pi. The
        // e = 0.8 lane dodged separately, on `simd_gt(0.8)` being false at
        // exactly 0.8 where the scalar takes the pi seed.
        let mm_arr = [3.267_256_359_733_385, 4.0, 5.5, 6.2];
        let e_arr = [0.802, 0.8, 0.5, 0.99];

        let mm_simd = f64x4::new(mm_arr);
        let e_simd = f64x4::new(e_arr);

        let (sine_simd, cosine_simd, eccentric_anomaly_simd) = solve_kepler_e_simd(mm_simd, e_simd);

        let sine_values = sine_simd.to_array();
        let cosine_values = cosine_simd.to_array();
        let eccentric_anomaly_values = eccentric_anomaly_simd.to_array();

        for (i, ((((mean_anomaly, eccentricity), eccentric_anomaly), sine_value), cosine_value)) in
            mm_arr
                .into_iter()
                .zip(e_arr)
                .zip(eccentric_anomaly_values)
                .zip(sine_values)
                .zip(cosine_values)
                .enumerate()
        {
            // `solve_kepler_e_core`, not `_core_wrapped`: the SIMD path wraps M
            // into [-pi, pi] internally, so the comparator has to wrap too or
            // the two answers differ by a legitimate 2*pi for every M > pi.
            let (ea_scalar, sin_scalar, cos_scalar) =
                solve_kepler_e_core(mean_anomaly, eccentricity);

            // Allow small relative error due to different iteration paths
            let ea_err = (eccentric_anomaly - ea_scalar).abs();
            assert!(
                ea_err < 1e-10,
                "Lane {i} E mismatch: scalar={ea_scalar}, simd={eccentric_anomaly}"
            );

            let sin_err = (sine_value - sin_scalar).abs();
            assert!(
                sin_err < 1e-10,
                "Lane {i} sin(E) mismatch: scalar={sin_scalar}, simd={sine_value}"
            );

            let cos_err = (cosine_value - cos_scalar).abs();
            assert!(
                cos_err < 1e-10,
                "Lane {i} cos(E) mismatch: scalar={cos_scalar}, simd={cosine_value}"
            );
        }
    }

    #[test]
    fn test_kepler_simd_convergence() {
        use wide::f64x4;

        // Edge case: high eccentricity near parabolic. Three of the four lanes
        // sit past pi, where the wrapped mean anomaly is negative — the shipped
        // version of this test was `splat(1.5)`, so all four lanes were the
        // same point and none of them was in the half of the revolution where
        // the seed was wrong.
        let mm_arr = [1.5, 3.5, 4.8, 6.0];
        let mm = f64x4::new(mm_arr);
        let e = f64x4::splat(0.9999);

        let (sin_e, cos_e, ea) = solve_kepler_e_simd(mm, e);

        // Verify Kepler's equation: E - e*sin(E) = M, against the same wrapped
        // M the solver used.
        let two_pi = f64x4::splat(std::f64::consts::TAU);
        let mm_wrapped = mm.simd_gt(f64x4::PI).select(mm - two_pi, mm);
        let residual = ea - e * sin_e - mm_wrapped;
        let residual_arr = residual.to_array();

        for (i, &residual_value) in residual_arr.iter().enumerate() {
            assert!(
                residual_value.abs() < 1e-10,
                "Lane {i} residual too large: {residual_value}"
            );
        }

        // The residual above never reads `cos_e`, so a wrong third return would
        // pass unnoticed. Check it against the eccentric anomaly the same call
        // reported, and check the pair lands on the unit circle.
        let sin_arr = sin_e.to_array();
        let cos_arr = cos_e.to_array();
        let ea_arr = ea.to_array();
        for (i, ((sin_value, cos_value), ea_value)) in
            sin_arr.into_iter().zip(cos_arr).zip(ea_arr).enumerate()
        {
            assert!(
                (cos_value - ea_value.cos()).abs() < 1e-12,
                "Lane {i} cos(E) = {cos_value} disagrees with cos({ea_value})"
            );
            let unit = sin_value.mul_add(sin_value, cos_value * cos_value);
            assert!(
                (unit - 1.0).abs() < 1e-12,
                "Lane {i} sin/cos pair is off the unit circle by {}",
                unit - 1.0
            );
        }
    }

    #[test]
    fn test_kepler_simd_large_mean_anomaly() {
        use wide::f64x4;

        // Test large mean anomaly wrapping (M = 10π should wrap correctly)
        let mm_large = f64x4::new([
            10.0 * std::f64::consts::PI,
            -10.0 * std::f64::consts::PI,
            10.0f64.mul_add(std::f64::consts::TAU, 3.267_256_359_733_385),
            -15.3 * std::f64::consts::PI,
        ]);
        // Every e here used to be below the 0.8 seed switch, so this test
        // exercised large-M wrapping only on the low-e branch. Lane 2 is now
        // high-e AND lands where the wrong seed actually diverged.
        //
        // Getting there took more than raising e. The bad seed only runs away
        // when the wrapped M sits just below -pi, i.e. when M mod 2pi is just
        // ABOVE pi — about 10% of the high-e half-revolution, not all of it.
        // 10pi and -10pi wrap to 0 and 20.5pi to +0.5pi; a high-e lane parked
        // on any of those converges anyway and proves nothing. Lane 2 is the
        // audit's own sample, pushed out by ten revolutions so that it tests
        // the wrapping and the seed together.
        let e_vals = f64x4::new([0.1, 0.85, 0.802, 0.5]);

        let (sine_simd, cosine_simd, eccentric_anomaly_simd) =
            solve_kepler_e_simd(mm_large, e_vals);

        let sine_values = sine_simd.to_array();
        let cosine_values = cosine_simd.to_array();
        let eccentric_anomaly_values = eccentric_anomaly_simd.to_array();
        let mean_anomalies = mm_large.to_array();
        let eccentricities = e_vals.to_array();

        // Compare with scalar version (which wraps internally)
        for (i, ((((mean_anomaly, eccentricity), eccentric_anomaly), sine_value), cosine_value)) in
            mean_anomalies
                .into_iter()
                .zip(eccentricities)
                .zip(eccentric_anomaly_values)
                .zip(sine_values)
                .zip(cosine_values)
                .enumerate()
        {
            let (ea_scalar, sin_scalar, cos_scalar) =
                solve_kepler_e_core(mean_anomaly, eccentricity);

            // Allow small error due to numerical differences
            let ea_err = (eccentric_anomaly - ea_scalar).abs();
            assert!(
                ea_err < 1e-9,
                "Lane {i} E mismatch for M={:.2}π: scalar={ea_scalar}, simd={eccentric_anomaly}",
                mean_anomaly / std::f64::consts::PI
            );

            let sin_err = (sine_value - sin_scalar).abs();
            assert!(
                sin_err < 1e-9,
                "Lane {i} sin(E) mismatch for M={:.2}π: scalar={sin_scalar}, simd={sine_value}",
                mean_anomaly / std::f64::consts::PI
            );

            let cos_err = (cosine_value - cos_scalar).abs();
            assert!(
                cos_err < 1e-9,
                "Lane {i} cos(E) mismatch for M={:.2}π: scalar={cos_scalar}, simd={cosine_value}",
                mean_anomaly / std::f64::consts::PI
            );
        }
    }

    #[test]
    fn test_mean_to_true_anomaly_simd() {
        use wide::f64x4;

        // Test mean-to-true anomaly conversion including wrapping.
        //
        // Lanes 2 and 3 are past pi at e >= 0.8, which is the region the broken
        // SIMD seed returned non-solutions for. The shipped version had its
        // only M > pi lane at e = 0.8 exactly, where `simd_gt(0.8)` was false
        // and the high-e seed never engaged — so it read as coverage of the
        // failing region while testing the low-e branch.
        let mm_vals = f64x4::new([0.5, 2.0, 5.5, 3.267_256_359_733_385]);
        let e_vals = f64x4::new([0.0, 0.5, 0.95, 0.802]);

        let true_anomaly_simd = mean_to_true_anomaly_simd(mm_vals, e_vals);
        let true_anomalies = true_anomaly_simd.to_array();
        let mean_anomalies = mm_vals.to_array();
        let eccentricities = e_vals.to_array();

        // Compare with scalar version
        for (i, ((mean_anomaly, eccentricity), true_anomaly)) in mean_anomalies
            .into_iter()
            .zip(eccentricities)
            .zip(true_anomalies)
            .enumerate()
        {
            let nu_scalar = mean_to_true_anomaly_impl(mean_anomaly, eccentricity);

            let nu_err = (true_anomaly - nu_scalar).abs();
            assert!(
                nu_err < 1e-9,
                "Lane {i} true anomaly mismatch: scalar={nu_scalar}, simd={true_anomaly}"
            );

            // Verify result is in [0, 2π]
            assert!(
                (0.0..std::f64::consts::TAU).contains(&true_anomaly),
                "Lane {i} true anomaly out of range: {true_anomaly}"
            );
        }
    }

    #[test]
    fn test_eci2equinoc_simd4_matches_scalar() {
        // LEO orbit test states - diverse positions around Earth
        let eci_states = [
            // State 0: Equatorial, circular
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            // State 1: Inclined
            [6778.0, 0.0, 100.0, 0.0, 7.5, 0.5],
            // State 2: Different position
            [0.0, 6778.0, 0.0, -7.67, 0.0, 0.0],
            // State 3: Polar-ish
            [4800.0, 0.0, 4800.0, 0.0, 5.4, 5.4],
        ];

        let mut eci_block = [0.0; 24];
        for (target, state) in eci_block.chunks_exact_mut(6).zip(&eci_states) {
            target.copy_from_slice(state);
        }

        // SIMD path
        let mut simd_out = [0.0; 24];
        eci2equinoc_simd4(&eci_block, 0.0, 0.0, &mut simd_out);

        // Scalar path
        let mut scalar_out = [0.0; 24];
        for state_index in 0..4 {
            let base = state_index * 6;
            eci2equinoc_impl(
                test_range(&eci_block, base, 6),
                6,
                0.0,
                0.0,
                test_range_mut(&mut scalar_out, base, 6),
            );
        }

        // Compare with tolerance
        for (i, (simd_value, scalar_value)) in simd_out.into_iter().zip(scalar_out).enumerate() {
            let diff = (simd_value - scalar_value).abs();
            let rel_err = if scalar_value.abs() > 1e-10 {
                diff / scalar_value.abs()
            } else {
                diff
            };
            assert!(
                rel_err < 1e-9,
                "Mismatch at index {i}: SIMD={simd_value:.10e} scalar={scalar_value:.10e} diff={diff:.2e}"
            );
        }
    }

    #[test]
    fn test_equinoc2eci_simd4_roundtrip() {
        // Start with ECI, convert to equinoctial, convert back
        let eci_orig = [
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            [6778.0, 0.0, 100.0, 0.0, 7.5, 0.5],
            [0.0, 6778.0, 0.0, -7.67, 0.0, 0.0],
            [4800.0, 0.0, 4800.0, 0.0, 5.4, 5.4],
        ];

        let mut eci_block = [0.0; 24];
        for (target, state) in eci_block.chunks_exact_mut(6).zip(&eci_orig) {
            target.copy_from_slice(state);
        }

        // ECI → Equinoctial
        let mut equ_block = [0.0; 24];
        eci2equinoc_simd4(&eci_block, 0.0, 0.0, &mut equ_block);

        // Equinoctial → ECI
        let mut eci_roundtrip = [0.0; 24];
        equinoc2eci_simd4(&equ_block, 0.0, 0.0, &mut eci_roundtrip);

        // Compare with original
        for (i, (roundtrip, original)) in eci_roundtrip.into_iter().zip(eci_block).enumerate() {
            let diff = (roundtrip - original).abs();
            assert!(
                diff < 1e-6,
                "Roundtrip mismatch at {i}: orig={original:.6} roundtrip={roundtrip:.6} diff={diff:.2e}"
            );
        }
    }
}
