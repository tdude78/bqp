//! Dust mass estimation cores for the in-process pipeline.
//!
//! This crate provides native Rust implementations of the dust mass
//! computation algorithms, exposed as buffer-free `pub fn` cores for the
//! in-process pipeline (no `Python`, `PyO3`, or `NumPy`).
//!
//! # What is here
//!
//! - [`mass_solver`] — the deterministic HF/J2 mass solver the pipeline runs.
//! - [`shared_target_contact_mass_requirement`] — the Part A production
//!   shared-target contact-mass authority.
//! - [`fraction_prepare`], [`fraction_finalize`] — the historical
//!   qualification and parity stages. `fraction_event`, the batched centroid
//!   stage that sat beside them, was deleted on 2026-08-21: it existed only to
//!   collapse the per-row `validate_raw_cloud_centroid` crossings of a Python
//!   boundary this crate no longer has, and had no Rust caller at all.
//! - `pc_inversion_bplane_from_states_batch_core` and the B-plane projection
//!   cores below, retained by qualification and unrelated parity tests.
//!
//! The legacy independent-convolution entry point is not part of the normal
//! production API:
//!
//! ```compile_fail
//! use dust_estimates_rs::pc_inversion_bplane_from_states_batch_core;
//! ```
//!
//! # What used to be here
//!
//! The **probabilistic (GMM) dust-mass search** — a polar grid search over the
//! chi-square confidence region, golden-section refinement, projected gradient
//! descent and dense-grid verification, all over a Gaussian-mixture dust cloud
//! — was removed on 2026-08-06. Its entry point was
//! `compute::compute_required_dust_mass_impl`; see `docs/REFACTOR_BLOCKLIST.md`
//! entry B4 for the removal record, including the final golden-constant run.

// mimalloc global allocator moved to lightyear_odeint_rs (single cdylib avoids TLS conflict).

pub mod fraction_finalize;
pub mod fraction_prepare;
pub mod mass_solver;
mod shared_target;

pub use mass_solver::DeterministicMassRoute;
pub use shared_target::{
    prepare_shared_target_conditional_capture_source, project_shared_target_bplane_components,
    replay_shared_target_contact_mass, shared_target_contact_mass_requirement,
    shared_target_contact_mass_requirement_under_limit, Binary64PacketCountUnrepresentable,
    Binary64PacketCountUnrepresentableReason, C12Binary64LogMinimumWitnessV1,
    ConditionalCaptureEstimate, DustMassClaim, DustScenarioIdentity,
    PreparedConditionalCaptureSource, ScenarioBoundDeterministicMass, SharedTargetBplaneProjection,
    SharedTargetBplaneProjectionInputs, SharedTargetConditionalCaptureSourceInputs,
    SharedTargetMassEstimate, SharedTargetMassInputs, SharedTargetPacketCountGovernor,
    SharedTargetPositionTreatment, SharedTargetQuadrature, SharedTargetReplay,
    SharedTargetReplayInputs, SharedTargetScenario, SharedTargetScenarioContentIdentity,
    MAX_EXACT_BINARY64_PACKET_COUNT, SHARED_TARGET_CLAIM_ID, SHARED_TARGET_COUNT_CERTIFICATE_ID,
    SHARED_TARGET_DRAW_INTEGRATION_ID, SHARED_TARGET_METHOD_ID,
};

#[expect(
    clippy::suspicious_operation_groupings,
    reason = "preserve established non-fused IEEE-754 covariance eigendecomposition order"
)]
pub(crate) fn sanitize_covariance_2d_values(
    a_raw: f64,
    b01_raw: f64,
    b10_raw: f64,
    d_raw: f64,
    min_eigenvalue: f64,
    max_eigenvalue: f64,
) -> anyhow::Result<([f64; 4], bool)> {
    if ![
        a_raw,
        b01_raw,
        b10_raw,
        d_raw,
        min_eigenvalue,
        max_eigenvalue,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        return Err(anyhow::anyhow!(
            "projected covariance contains non-finite values"
        ));
    }
    let a = a_raw;
    let b = 0.5 * (b01_raw + b10_raw);
    let d = d_raw;
    // Scale-safe symmetric 2x2 eigenvalues.
    //
    // The direct form computed `disc = tr^2 - 4*det`, took its square root and
    // returned `0.5 * (tr - sqrt_disc)` as the small eigenvalue. For a spectrum
    // the compiled bounds permit -- diag(1e12, 1e-6) sits inside 1e-6..1e6 once
    // scaled -- `tr^2` is 1e24 while `4*det` is 4e6, so the subtraction loses
    // every bit of the small eigenvalue and `tr - sqrt_disc` cancels to zero.
    // The minor variance was then clamped to `min_eigenvalue`, shrinking it by
    // orders of magnitude, and nothing reported that anything had happened.
    //
    // Two changes remove the cancellation. The discriminant is formed as
    // `hypot(half_diff, b)`, which never squares the trace, and the eigenvalue
    // NEARER zero is recovered from the exact product identity
    // `lam_near = det / lam_far` rather than from a difference of two nearly
    // equal numbers. `lam_far` is the root further from zero and is computed by
    // ADDING quantities of the same sign, so it carries no cancellation to pass
    // on.
    let tr = a + d;
    let det = a * d - b * b;
    let half_tr = 0.5 * tr;
    let half_diff = 0.5 * (a - d);
    let radius = half_diff.hypot(b);
    let lam_far = if half_tr >= 0.0 {
        half_tr + radius
    } else {
        half_tr - radius
    };
    let lam_near = if lam_far == 0.0 {
        // Both roots are zero: `lam_far` is the larger-magnitude one.
        0.0
    } else {
        det / lam_far
    };
    let (raw_lam1, raw_lam2) = if lam_near <= lam_far {
        (lam_near, lam_far)
    } else {
        (lam_far, lam_near)
    };
    let lam1 = raw_lam1.max(min_eigenvalue).min(max_eigenvalue);
    let lam2 = raw_lam2.max(min_eigenvalue).min(max_eigenvalue);
    // Auditability: report when the eigenvalue clamp actually altered the
    // input instead of silently sanitizing (audit finding, 2026-07-11).
    let clamped = lam1.partial_cmp(&raw_lam1) != Some(std::cmp::Ordering::Equal)
        || lam2.partial_cmp(&raw_lam2) != Some(std::cmp::Ordering::Equal);
    if (lam1 - lam2).abs() < 1e-15 * lam1.abs().max(1.0) {
        return Ok(([lam1, 0.0, 0.0, lam1], clamped));
    }

    // Eigenvector for lambda solves b*v_x + (d - lambda)*v_y = 0, i.e.
    // v = (lambda - d, b). For (near-)diagonal input the pairing must follow
    // the eigenvalue ORDER: lam1 = min(a, d), so its axis is x only when
    // a <= d. The old unconditional (1,0)/(0,1) pairing silently SWAPPED the
    // x/y variances of any diagonal input with a > d.
    let (mut v1x, mut v1y, mut v2x, mut v2y) = if b.abs() > 1e-15 {
        (raw_lam1 - d, b, raw_lam2 - d, b)
    } else if a <= d {
        (1.0, 0.0, 0.0, 1.0)
    } else {
        (0.0, 1.0, 1.0, 0.0)
    };
    let first_axis_term = v1x * v1x;
    let first_orthogonal_term = v1y * v1y;
    let n1 = (first_axis_term + first_orthogonal_term).sqrt();
    if n1 > 1e-15 {
        let inv = 1.0 / n1;
        v1x *= inv;
        v1y *= inv;
    } else {
        v1x = 1.0;
        v1y = 0.0;
    }
    let second_axis_term = v2x * v2x;
    let second_orthogonal_term = v2y * v2y;
    let n2 = (second_axis_term + second_orthogonal_term).sqrt();
    if n2 > 1e-15 {
        let inv = 1.0 / n2;
        v2x *= inv;
        v2y *= inv;
    } else {
        v2x = 0.0;
        v2y = 1.0;
    }
    Ok((
        [
            lam1 * v1x * v1x + lam2 * v2x * v2x,
            lam1 * v1x * v1y + lam2 * v2x * v2y,
            lam1 * v1y * v1x + lam2 * v2y * v2x,
            lam1 * v1y * v1y + lam2 * v2y * v2y,
        ],
        clamped,
    ))
}

#[cfg(any(test, feature = "solver-qualification"))]
pub(crate) type PcInversionCoreResult = (
    f64,
    &'static str,
    f64,
    f64,
    usize,
    usize,
    usize,
    f64,
    usize,
    f64,
);

#[cfg(any(test, feature = "solver-qualification"))]
#[derive(Debug, Clone, PartialEq)]
pub struct PcInversionBplaneBatchRow {
    pub projection_clamped: usize,
    pub debris_cov: [f64; 4],
    pub pc: PcInversionCoreResult,
}

#[cfg(any(test, feature = "solver-qualification"))]
struct BplaneProjectionInputs<'a> {
    aligned_means_3d: &'a [f64],
    aligned_covs_3d: &'a [f64],
    target_pos_eci: &'a [f64],
    plane: &'a [f64],
    target_cov_3d: &'a [f64],
    cov_min_eig: f64,
    cov_max_eig: f64,
}

#[cfg(any(test, feature = "solver-qualification"))]
struct BplaneProjectionOutputs<'a> {
    projected_means: &'a mut [f64],
    projected_covariances: &'a mut [f64],
    debris_covariance: &'a mut [f64],
}

pub(crate) fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    let [l0, l1, l2] = left;
    let [r0, r1, r2] = right;
    l0 * r0 + l1 * r1 + l2 * r2
}

fn bilinear3(left: [f64; 3], covariance: [f64; 9], right: [f64; 3]) -> f64 {
    let [l0, l1, l2] = left;
    let [r0, r1, r2] = right;
    let [c00, c01, c02, c10, c11, c12, c20, c21, c22] = covariance;
    let mut value = 0.0;
    value += l0 * c00 * r0;
    value += l0 * c01 * r1;
    value += l0 * c02 * r2;
    value += l1 * c10 * r0;
    value += l1 * c11 * r1;
    value += l1 * c12 * r2;
    value += l2 * c20 * r0;
    value += l2 * c21 * r1;
    value += l2 * c22 * r2;
    value
}

pub(crate) fn project_covariance_to_bplane(covariance: [f64; 9], plane: [f64; 6]) -> [f64; 4] {
    let [p00, p01, p02, p10, p11, p12] = plane;
    let first = [p00, p01, p02];
    let second = [p10, p11, p12];
    [
        bilinear3(first, covariance, first),
        bilinear3(first, covariance, second),
        bilinear3(second, covariance, first),
        bilinear3(second, covariance, second),
    ]
}

/// Projects component and debris covariances into one B-plane.
///
/// # Errors
///
/// Returns an error when packed dimensions are inconsistent, fixed-size inputs
/// are malformed, or projected covariance sanitization fails.
#[cfg(any(test, feature = "solver-qualification"))]
fn project_bplane_components_core(
    inputs: &BplaneProjectionInputs<'_>,
    outputs: BplaneProjectionOutputs<'_>,
) -> anyhow::Result<usize> {
    let aligned_means_3d = inputs.aligned_means_3d;
    let aligned_covs_3d = inputs.aligned_covs_3d;
    let target_pos_eci = inputs.target_pos_eci;
    let plane = inputs.plane;
    let target_cov_3d = inputs.target_cov_3d;
    let cov_min_eig = inputs.cov_min_eig;
    let cov_max_eig = inputs.cov_max_eig;
    let BplaneProjectionOutputs {
        projected_means,
        projected_covariances,
        debris_covariance,
    } = outputs;
    if !aligned_means_3d.len().is_multiple_of(3) {
        return Err(anyhow::anyhow!(
            "aligned_means_3d length must be divisible by 3"
        ));
    }
    let n = aligned_means_3d.len() / 3;
    let Some(aligned_covariance_len) = n.checked_mul(9) else {
        return Err(anyhow::anyhow!("aligned covariance length overflow"));
    };
    if aligned_covs_3d.len() != aligned_covariance_len {
        return Err(anyhow::anyhow!(
            "aligned_covs_3d length must equal component_count * 9"
        ));
    }
    let Ok(target_position) = <[f64; 3]>::try_from(target_pos_eci) else {
        return Err(anyhow::anyhow!("target_pos_eci length must equal 3"));
    };
    let Ok(plane) = <[f64; 6]>::try_from(plane) else {
        return Err(anyhow::anyhow!("p_bp length must equal 6"));
    };
    let Ok(target_covariance) = <[f64; 9]>::try_from(target_cov_3d) else {
        return Err(anyhow::anyhow!("target_cov_3d length must equal 9"));
    };
    let Some(projected_mean_len) = n.checked_mul(2) else {
        return Err(anyhow::anyhow!("projected mean length overflow"));
    };
    if projected_means.len() != projected_mean_len {
        return Err(anyhow::anyhow!(
            "proj_means_out length must equal component_count * 2"
        ));
    }
    let Some(projected_covariance_len) = n.checked_mul(4) else {
        return Err(anyhow::anyhow!("projected covariance length overflow"));
    };
    if projected_covariances.len() != projected_covariance_len {
        return Err(anyhow::anyhow!(
            "proj_covs_out length must equal component_count * 4"
        ));
    }
    let Ok(debris_covariance) = <&mut [f64; 4]>::try_from(debris_covariance) else {
        return Err(anyhow::anyhow!("debris_cov_out length must equal 4"));
    };

    let mut clamped_count = 0usize;
    for (((mean, covariance), projected_mean), projected_covariance) in aligned_means_3d
        .chunks_exact(3)
        .zip(aligned_covs_3d.chunks_exact(9))
        .zip(projected_means.chunks_exact_mut(2))
        .zip(projected_covariances.chunks_exact_mut(4))
    {
        let Ok(mean) = <[f64; 3]>::try_from(mean) else {
            return Err(anyhow::anyhow!("aligned mean row must contain 3 values"));
        };
        let Ok(covariance) = <[f64; 9]>::try_from(covariance) else {
            return Err(anyhow::anyhow!(
                "aligned covariance row must contain 9 values"
            ));
        };
        let [mean_x, mean_y, mean_z] = mean;
        let [target_x, target_y, target_z] = target_position;
        let delta = [mean_x - target_x, mean_y - target_y, mean_z - target_z];
        let [p00, p01, p02, p10, p11, p12] = plane;
        let [projected_x, projected_y] = projected_mean else {
            return Err(anyhow::anyhow!("projected mean row must contain 2 values"));
        };
        *projected_x = dot3([p00, p01, p02], delta);
        *projected_y = dot3([p10, p11, p12], delta);

        let raw = project_covariance_to_bplane(covariance, plane);
        let [raw00, raw01, raw10, raw11] = raw;
        let (sanitized, clamped) =
            sanitize_covariance_2d_values(raw00, raw01, raw10, raw11, cov_min_eig, cov_max_eig)?;
        if clamped {
            clamped_count = clamped_count
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("projection clamp count overflow"))?;
        }
        projected_covariance.copy_from_slice(&sanitized);
    }

    let raw_debris = project_covariance_to_bplane(target_covariance, plane);
    let [raw00, raw01, raw10, raw11] = raw_debris;
    let (debris, debris_clamped) =
        sanitize_covariance_2d_values(raw00, raw01, raw10, raw11, cov_min_eig, cov_max_eig)?;
    if debris_clamped {
        clamped_count = clamped_count
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("projection clamp count overflow"))?;
    }
    *debris_covariance = debris;
    Ok(clamped_count)
}

#[inline]
#[expect(
    clippy::suspicious_operation_groupings,
    reason = "preserve established non-fused symmetric determinant operation order"
)]
fn det_inv_symmetric_2x2(cov: &[f64]) -> (f64, f64, f64, f64) {
    let Ok([a, b01, b10, c]) = <[f64; 4]>::try_from(cov) else {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    };
    let b = 0.5 * (b01 + b10);
    let det = a * c - b * b;
    if !det.is_finite() || det <= 0.0 {
        return (det, f64::NAN, f64::NAN, f64::NAN);
    }
    let inv_det = 1.0 / det;
    (det, c * inv_det, -b * inv_det, a * inv_det)
}

#[inline]
fn mahalanobis_2x2(dx: f64, dy: f64, inv00: f64, inv01: f64, inv11: f64) -> f64 {
    inv00 * dx * dx + 2.0 * inv01 * dx * dy + inv11 * dy * dy
}

#[cfg(any(test, feature = "solver-qualification"))]
fn integrate_gaussian_disk_capture_probability_core(
    mean_x: f64,
    mean_y: f64,
    cov: &[f64; 4],
    radius_km: f64,
    radial_samples: usize,
    angular_samples: usize,
) -> f64 {
    if !radius_km.is_finite() || radius_km <= 0.0 {
        return 0.0;
    }
    let (det_cov, inv00, inv01, inv11) = det_inv_symmetric_2x2(cov);
    if !det_cov.is_finite() || det_cov <= 0.0 {
        return 0.0;
    }
    let nr = radial_samples.max(1);
    let nt = angular_samples.max(1);
    let (Ok(radial_count), Ok(angular_count)) = (u32::try_from(nr), u32::try_from(nt)) else {
        return 0.0;
    };
    let dr = radius_km / f64::from(radial_count);
    let dtheta = std::f64::consts::TAU / f64::from(angular_count);
    let norm_const = 1.0 / (std::f64::consts::TAU * det_cov.sqrt());
    // Hoist the loop-invariant midpoint trig: theta = (it + 0.5) * dtheta depends
    // only on `it`, yet cos/sin were recomputed once per radial index ir. Build a
    // length-nt midpoint table once per call and index it in the inner loop.
    // `((it + 0.5) * dtheta).cos()` is bit-identical whether computed inline or
    // from the table (same input, same libm call); the ir-outer/it-inner loop
    // nesting and the `integral +=` accumulation order are untouched.
    let mut cos_mid = Vec::with_capacity(nt);
    let mut sin_mid = Vec::with_capacity(nt);
    for it in 0..angular_count {
        let theta = (f64::from(it) + 0.5) * dtheta;
        cos_mid.push(theta.cos());
        sin_mid.push(theta.sin());
    }
    let mut integral = 0.0;
    for ir in 0..radial_count {
        let r_mid = (f64::from(ir) + 0.5) * dr;
        let area_weight = r_mid * dr * dtheta;
        for (&cosine, &sine) in cos_mid.iter().zip(&sin_mid) {
            let dx = r_mid * cosine - mean_x;
            let dy = r_mid * sine - mean_y;
            let quad = mahalanobis_2x2(dx, dy, inv00, inv01, inv11);
            integral += norm_const * (-0.5 * quad).exp() * area_weight;
        }
    }
    if integral.is_finite() {
        integral.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinitePacketMassBound {
    pub release_mass_kg: f64,
    pub released_packet_count: u64,
    pub required_captured_packets: u64,
    pub packet_mass_kg: f64,
    pub expected_captured_packets: f64,
}

#[inline]
fn next_up_positive(value: f64) -> f64 {
    debug_assert!(value.is_finite() && value > 0.0);
    let Some(bits) = value.to_bits().checked_add(1) else {
        return f64::INFINITY;
    };
    f64::from_bits(bits)
}

fn u64_to_f64(value: u64) -> f64 {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = value.to_le_bytes();
    let low = u32::from_le_bytes([b0, b1, b2, b3]);
    let high = u32::from_le_bytes([b4, b5, b6, b7]);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}

fn positive_integral_f64_to_u64(value: f64) -> Option<u64> {
    if !value.is_finite() || !(1.0..18_446_744_073_709_551_616.0).contains(&value) {
        return None;
    }
    let bits = value.to_bits();
    let exponent_bits = bits.checked_shr(52)? & 0x7ff;
    let exponent = i32::try_from(exponent_bits).ok()?.checked_sub(1023)?;
    let significand = (bits & 0x000f_ffff_ffff_ffff) | (1_u64.checked_shl(52)?);
    if exponent >= 52 {
        significand.checked_shl(u32::try_from(exponent.checked_sub(52)?).ok()?)
    } else {
        significand.checked_shr(u32::try_from(52_i32.checked_sub(exponent)?).ok()?)
    }
}

#[inline]
fn checked_ceil_packet_count(value: f64, label: &str) -> anyhow::Result<u64> {
    let count = value.ceil();
    // The first exactly represented value outside `u64` is 2^64.
    // Reject that exclusive upper boundary before Rust's saturating float cast.
    let Some(count) = positive_integral_f64_to_u64(count) else {
        return Err(anyhow::anyhow!("{label} packet count exceeds u64"));
    };
    Ok(count)
}

/// Computes conservative finite-packet release mass.
///
/// CLAIM CLASS (2026-08-18): like the shared-target lane, this is a CONTACT
/// bound — released mass for >=1-grain contact at `target_probability` — and
/// NOT a deflection/momentum-delivery bound.
///
/// MODEL LIMIT (audit 2026-08-16): "conservative" holds only under the
/// independence assumption — every packet is an independent Bernoulli trial.
/// With the sealed `grains_per_independent_packet = 1`, production masses
/// (~0.3 kg / ~5e8 grains) make the Chernoff confidence term load-free: the
/// `target_probability` knob moves the released mass by ~1e-4 relative
/// (pinned by `tests/confidence_term_is_load_free.rs`), so the result is
/// effectively `deterministic mass / capture_probability`. The dominant
/// correlated error — the shared target-ephemeris draw folded into every
/// grain's capture covariance — is NOT expressible in this bound; the
/// achieved hit confidence against a real target is bounded by that shared
/// realization, not by `target_probability`. Quote outputs as
/// model-conditional masses, never as `P(hit) >= target_probability`.
///
/// # Errors
///
/// Returns an error for invalid probabilities, masses, packet size, or counts
/// outside the finite `u64`/`f64` authority domain.
pub fn finite_packet_release_mass_bound_core(
    capture_probability: f64,
    target_probability: f64,
    deterministic_required_mass_kg: f64,
    grain_mass_kg: f64,
    grains_per_independent_packet: u64,
) -> anyhow::Result<FinitePacketMassBound> {
    if !capture_probability.is_finite() || !(0.0..=1.0).contains(&capture_probability) {
        return Err(anyhow::anyhow!("capture probability must lie in (0, 1]"));
    }
    if capture_probability == 0.0 {
        return Err(anyhow::anyhow!("capture probability must lie in (0, 1]"));
    }
    if !target_probability.is_finite() || !(0.0..1.0).contains(&target_probability) {
        return Err(anyhow::anyhow!("target probability must lie in (0, 1)"));
    }
    if !deterministic_required_mass_kg.is_finite() || deterministic_required_mass_kg <= 0.0 {
        return Err(anyhow::anyhow!(
            "deterministic required mass must be finite and > 0"
        ));
    }
    if !grain_mass_kg.is_finite() || grain_mass_kg <= 0.0 {
        return Err(anyhow::anyhow!("grain mass must be finite and > 0"));
    }
    if grains_per_independent_packet == 0 {
        return Err(anyhow::anyhow!(
            "grains per independent packet must be positive"
        ));
    }
    let packet_mass_kg = grain_mass_kg * u64_to_f64(grains_per_independent_packet);
    if !packet_mass_kg.is_finite() || packet_mass_kg <= 0.0 {
        return Err(anyhow::anyhow!("independent packet mass is invalid"));
    }
    let required_ratio = deterministic_required_mass_kg / packet_mass_kg;
    let required_captured_packets =
        checked_ceil_packet_count(required_ratio.max(1.0), "required captured")?;
    let log_failure_inverse = -(-target_probability).ln_1p();
    let root_term = (2.0 * log_failure_inverse).sqrt();
    // Contraction immunity: on x86 with `-Cllvm-args=-fp-contract=on` LLVM is
    // free to fuse the two products straddling the `+` in the discriminant
    // (`root_term*root_term + 4.0*required`) into a single FMA. FMA keeps the
    // intermediate product at infinite precision, so its result can differ in
    // the last bit from the Python oracle's strict IEEE-754 mul-then-add. That
    // one-ulp shift survives `/ capture_probability`, `next_up_positive`, and
    // `ceil`, flipping `released_packet_count` by a whole packet. `black_box`
    // is a semantic identity (zero-cost, no change to the operation or its
    // order) but an optimization barrier: it forces each product to be rounded
    // to f64 and materialized before the add/square, so no FMA can form. This
    // mirrors the Python evaluation `root_term*root_term + 4.0*required_packets`
    // and `(...) ** 2` exactly.
    let root_term_squared = std::hint::black_box(root_term * root_term);
    let required_linear_term = std::hint::black_box(4.0 * u64_to_f64(required_captured_packets));
    let discriminant_root = (root_term_squared + required_linear_term).sqrt();
    let inner_sum = std::hint::black_box(root_term + discriminant_root);
    let inner_square = std::hint::black_box(inner_sum.powi(2));
    let expected_required = 0.25 * inner_square;
    let released_expectation = expected_required / capture_probability;
    if !released_expectation.is_finite() || released_expectation <= 0.0 {
        return Err(anyhow::anyhow!("released packet count exceeds u64"));
    }
    let released_packet_count = checked_ceil_packet_count(
        next_up_positive(released_expectation).max(u64_to_f64(required_captured_packets)),
        "released",
    )?;
    let release_mass_kg = u64_to_f64(released_packet_count) * packet_mass_kg;
    if !release_mass_kg.is_finite() {
        return Err(anyhow::anyhow!("released mass exceeds finite f64"));
    }
    Ok(FinitePacketMassBound {
        release_mass_kg,
        released_packet_count,
        required_captured_packets,
        packet_mass_kg,
        expected_captured_packets: u64_to_f64(released_packet_count) * capture_probability,
    })
}

#[cfg(any(test, feature = "solver-qualification"))]
struct PcInversionInputs<'a> {
    debris_covariance: &'a [f64],
    projected_means: &'a [f64],
    projected_covariances: &'a [f64],
    weights: &'a [f64],
    component_count: usize,
    area_km2: f64,
    hit_probability: f64,
    deterministic_mass: f64,
    grain_mass_kg: f64,
    grains_per_packet: u64,
    covariance_minimum: f64,
    covariance_maximum: f64,
    radial_samples: usize,
    angular_samples: usize,
    small_area_eta_max: f64,
}

#[cfg(any(test, feature = "solver-qualification"))]
enum ValidatedPcInputs<'a> {
    Infeasible(PcInversionCoreResult),
    Valid {
        debris_covariance: [f64; 4],
        projected_means: &'a [f64],
        projected_covariances: &'a [f64],
        weights: &'a [f64],
    },
}

#[cfg(any(test, feature = "solver-qualification"))]
const fn invalid_pc_result(mode: &'static str) -> PcInversionCoreResult {
    (
        f64::INFINITY,
        mode,
        0.0,
        f64::NAN,
        0,
        0,
        0,
        f64::NAN,
        0,
        f64::NAN,
    )
}

#[cfg(any(test, feature = "solver-qualification"))]
fn validate_pc_inputs<'a>(inputs: &PcInversionInputs<'a>) -> anyhow::Result<ValidatedPcInputs<'a>> {
    if inputs.area_km2 <= 0.0 || !inputs.area_km2.is_finite() {
        return Ok(ValidatedPcInputs::Infeasible(invalid_pc_result(
            "invalid_area",
        )));
    }
    if !inputs.hit_probability.is_finite() || !(0.0..1.0).contains(&inputs.hit_probability) {
        return Ok(ValidatedPcInputs::Infeasible(invalid_pc_result(
            "invalid_probability",
        )));
    }
    if inputs.component_count == 0 {
        return Err(anyhow::anyhow!(
            "n_components must be positive for Pc inversion"
        ));
    }
    let Ok(debris_covariance) = <[f64; 4]>::try_from(inputs.debris_covariance) else {
        return Err(anyhow::anyhow!("invalid Pc inversion debris covariance"));
    };
    if !debris_covariance.iter().all(|v| v.is_finite()) {
        return Err(anyhow::anyhow!("invalid Pc inversion debris covariance"));
    }
    let required_means = inputs
        .component_count
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("invalid Pc inversion component arrays"))?;
    let required_covariances = inputs
        .component_count
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("invalid Pc inversion component arrays"))?;
    if inputs.projected_means.len() < required_means
        || inputs.projected_covariances.len() < required_covariances
        || inputs.weights.len() < inputs.component_count
    {
        return Err(anyhow::anyhow!("invalid Pc inversion component arrays"));
    }
    let weights = inputs
        .weights
        .get(..inputs.component_count)
        .ok_or_else(|| anyhow::anyhow!("invalid Pc inversion mixture weights"))?;
    if !weights.iter().all(|v| v.is_finite() && *v >= 0.0) || weights.iter().sum::<f64>() <= 0.0 {
        return Err(anyhow::anyhow!("invalid Pc inversion mixture weights"));
    }
    Ok(ValidatedPcInputs::Valid {
        debris_covariance,
        projected_means: inputs.projected_means,
        projected_covariances: inputs.projected_covariances,
        weights,
    })
}

/// Solves one projected Pc inversion row.
///
/// # Errors
///
/// Returns an error for malformed packed components, invalid covariance, failed
/// quadrature, or invalid finite-packet mass inputs.
#[expect(
    clippy::suspicious_operation_groupings,
    reason = "preserve established non-fused IEEE-754 Pc inversion formulas and accumulation"
)]
#[cfg(any(test, feature = "solver-qualification"))]
/// Inverts the capture probability for the required release mass.
///
/// # Two defects fixed 2026-08-20, both biasing Pc DOWNWARD
///
/// **The small-area gate was inverted on non-finite eta.** The branch read
/// `if eta.is_finite() && !small_area_valid { quadrature } else { linear }`, so
/// a NON-FINITE `eta` fell to the `else` and took the linear small-area
/// expansion. That is backwards. `eta` is the disk area over the covariance
/// scale, so a non-finite `eta` means the disk is enormous against the
/// covariance -- exactly the regime where the small-area expansion is least
/// valid. The component that most needed quadrature was the one component
/// guaranteed not to get it. Since `small_area_valid` already requires both
/// `eta` and the curvature term to be finite and in bounds, testing it alone is
/// correct AND simpler: anything not demonstrably valid goes to quadrature,
/// which fails loudly under the strict probabilistic mass policy rather than
/// approximating in a regime it does not cover.
///
/// **A NaN component was dropped in silence.** `if log_term.is_finite()` also
/// swallowed NaN, and a dropped component lowers the capture sum, therefore
/// Pc, therefore the required release mass -- toward UNDER-protection, with no
/// counter recording it. The two defects compounded: non-finite `eta` took the
/// linear branch, `curvature_rel.ln_1p()` on a non-finite curvature produced
/// NaN, and the NaN was then deleted. NaN is now a hard error.
///
/// `-inf` still drops, and that is not the same case: it is what `weight.ln()`
/// returns for a zero-weight component, whose contribution `exp(-inf)` is
/// exactly zero. Dropping it is exact, not lossy. `+inf` is grouped with NaN:
/// an infinite contribution is as broken as an undefined one.
#[expect(
    clippy::too_many_lines,
    reason = "205/200 after the 2026-08-20 correctness fixes above; the body was already at the limit. Splitting the per-component branch out needs a parameter struct (it closes over dx, dy, cov_rel, radius, both sample counts, the validity flag, two log terms, the curvature and two counters), which is a refactor with its own review, not something to attach to a physics fix"
)]
fn pc_inversion_mass_core(inputs: &PcInversionInputs<'_>) -> anyhow::Result<PcInversionCoreResult> {
    let n_components = inputs.component_count;
    let area_km2 = inputs.area_km2;
    let hit_probability = inputs.hit_probability;
    let det_mass = inputs.deterministic_mass;
    let grain_mass_kg = inputs.grain_mass_kg;
    let grains_per_independent_packet = inputs.grains_per_packet;
    let cov_min_eig = inputs.covariance_minimum;
    let cov_max_eig = inputs.covariance_maximum;
    let radial_samples = inputs.radial_samples;
    let angular_samples = inputs.angular_samples;
    let small_area_eta_max = inputs.small_area_eta_max;
    let (debris_covariance, proj_means_2d, proj_covs_2d, weights) =
        match validate_pc_inputs(inputs)? {
            ValidatedPcInputs::Infeasible(result) => return Ok(result),
            ValidatedPcInputs::Valid {
                debris_covariance,
                projected_means,
                projected_covariances,
                weights,
            } => (
                debris_covariance,
                projected_means,
                projected_covariances,
                weights,
            ),
        };

    let radius_km = (area_km2 / std::f64::consts::PI).sqrt();
    let log_area = area_km2.max(1e-300).ln();
    let norm_const_log = -std::f64::consts::TAU.ln();
    let mut log_terms: Vec<f64> = Vec::with_capacity(n_components);
    let mut eta_max_seen = 0.0_f64;
    let mut small_area_components = 0usize;
    let mut quadrature_components = 0usize;
    let mut clamped_components = 0usize;
    let mut curvature_rel_max = 0.0_f64;
    for ((&weight, projected_mean), projected_covariance) in weights
        .iter()
        .zip(proj_means_2d.chunks_exact(2))
        .zip(proj_covs_2d.chunks_exact(4))
    {
        if weight == 0.0 {
            continue;
        }
        let Ok([cov00, cov01, cov10, cov11]) = <[f64; 4]>::try_from(projected_covariance) else {
            return Err(anyhow::anyhow!(
                "invalid Pc inversion component covariance row"
            ));
        };
        let [debris00, debris01, debris10, debris11] = debris_covariance;
        let raw = [
            cov00 + debris00,
            cov01 + debris01,
            cov10 + debris10,
            cov11 + debris11,
        ];
        let [raw00, raw01, raw10, raw11] = raw;
        let (cov_rel, cov_clamped) =
            sanitize_covariance_2d_values(raw00, raw01, raw10, raw11, cov_min_eig, cov_max_eig)?;
        if cov_clamped {
            clamped_components = clamped_components
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("clamped component count overflow"))?;
        }
        let (det_rel, inv00, inv01, inv11) = det_inv_symmetric_2x2(&cov_rel);
        if !det_rel.is_finite() || det_rel <= 0.0 {
            return Err(anyhow::anyhow!("invalid Pc inversion component covariance"));
        }
        let Ok([dx, dy]) = <[f64; 2]>::try_from(projected_mean) else {
            return Err(anyhow::anyhow!("invalid Pc inversion component mean row"));
        };
        let mahal = mahalanobis_2x2(dx, dy, inv00, inv01, inv11);
        if !mahal.is_finite() {
            return Err(anyhow::anyhow!(
                "invalid Pc inversion component Mahalanobis distance"
            ));
        }
        let sqrt_det_rel = det_rel.sqrt();
        let log_pdf_at_origin = norm_const_log - 0.5 * det_rel.ln() - 0.5 * mahal;
        let eta_component = area_km2 / (std::f64::consts::TAU * sqrt_det_rel).max(1e-300);
        if eta_component.is_finite() {
            eta_max_seen = eta_max_seen.max(eta_component);
        }
        // Second-order relative correction of the linear area*pdf
        // approximation over a disk of radius r:
        //   integral ~ A*pdf(0) * (1 + (r^2/8)*(||inv(S)*mu||^2 - tr(inv(S))))
        // (mu = component mean offset from the origin). The eta ratio alone
        // never sees the Mahalanobis geometry, so deep-tail high-aspect
        // components passed the old gate while the linear term underestimated
        // Pc by up to ~70% (audit finding, 2026-07-11). Gate on BOTH eta and
        // the curvature term, and apply the correction on the fast path.
        let grad_x = inv00 * dx + inv01 * dy;
        let grad_y = inv01 * dx + inv11 * dy;
        let curvature_rel =
            0.125 * radius_km * radius_km * ((grad_x * grad_x + grad_y * grad_y) - (inv00 + inv11));
        if curvature_rel.is_finite() {
            curvature_rel_max = curvature_rel_max.max(curvature_rel.abs());
        }
        let small_area_valid = eta_component.is_finite()
            && eta_component <= small_area_eta_max
            && curvature_rel.is_finite()
            && curvature_rel.abs() <= small_area_eta_max;
        // Routes on the VALIDITY of the small-area expansion and nothing else;
        // see this function's own docs for why the old `eta.is_finite() &&`
        // conjunct inverted the decision.
        let log_p_capture_component = if small_area_valid {
            small_area_components = small_area_components
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("small-area component count overflow"))?;
            log_area + log_pdf_at_origin + curvature_rel.ln_1p()
        } else {
            let p_capture_component = integrate_gaussian_disk_capture_probability_core(
                dx,
                dy,
                &cov_rel,
                radius_km,
                radial_samples,
                angular_samples,
            );
            if p_capture_component.is_finite() && p_capture_component > 0.0 {
                quadrature_components = quadrature_components
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("quadrature component count overflow"))?;
                p_capture_component.max(1e-300).ln()
            } else {
                return Err(anyhow::anyhow!(
                    "Pc quadrature failed under strict probabilistic mass policy"
                ));
            }
        };
        let log_term = weight.ln() + log_p_capture_component;
        // NaN and +inf are broken components; -inf is an exact zero-weight one.
        anyhow::ensure!(
            log_term.is_finite() || log_term == f64::NEG_INFINITY,
            "Pc component produced a non-finite log-weight ({log_term}) under \
             strict probabilistic mass policy; dropping it would bias the \
             capture sum, and therefore the required release mass, DOWNWARD"
        );
        if log_term.is_finite() {
            log_terms.push(log_term);
        }
    }
    if log_terms.is_empty() {
        return Ok((
            f64::INFINITY,
            "unresolved",
            0.0,
            eta_max_seen,
            small_area_components,
            quadrature_components,
            0,
            f64::NAN,
            clamped_components,
            curvature_rel_max,
        ));
    }
    let max_log_term = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp = log_terms
        .iter()
        .map(|value| (*value - max_log_term).exp())
        .sum::<f64>();
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return Ok((
            f64::INFINITY,
            "unresolved",
            0.0,
            eta_max_seen,
            small_area_components,
            quadrature_components,
            log_terms.len(),
            f64::NAN,
            clamped_components,
            curvature_rel_max,
        ));
    }
    // Preserve arbitrarily small finite capture probabilities in log space.
    // Flooring at 1e-12 silently understated required mass for weak overlap.
    let log_p_capture = (max_log_term + sum_exp.ln()).min(0.0);
    let probability_inflation = -(-hit_probability).ln_1p();
    if probability_inflation <= 0.0 || !probability_inflation.is_finite() {
        return Ok((
            f64::INFINITY,
            "invalid_probability",
            log_p_capture.exp(),
            eta_max_seen,
            small_area_components,
            quadrature_components,
            log_terms.len(),
            probability_inflation,
            clamped_components,
            curvature_rel_max,
        ));
    }
    let capture_probability = log_p_capture.exp();
    let mass = finite_packet_release_mass_bound_core(
        capture_probability,
        hit_probability,
        det_mass,
        grain_mass_kg,
        grains_per_independent_packet,
    )?
    .release_mass_kg;
    let mode = if quadrature_components > 0 && small_area_components > 0 {
        "mixed_quadrature_small_area"
    } else if quadrature_components > 0 {
        "numerical_quadrature"
    } else {
        "small_area_density"
    };
    Ok((
        mass,
        mode,
        log_p_capture.exp(),
        eta_max_seen,
        small_area_components,
        quadrature_components,
        log_terms.len(),
        probability_inflation,
        clamped_components,
        curvature_rel_max,
    ))
}

#[cfg(any(test, feature = "solver-qualification"))]
fn packed_range(start: usize, end: usize, width: usize) -> Option<std::ops::Range<usize>> {
    Some(start.checked_mul(width)?..end.checked_mul(width)?)
}

/// Solves packed B-plane rows after projecting each component.
///
/// # Errors
///
/// Returns an error when row offsets, packed dimensions, projection inputs, or
/// per-row Pc inputs are invalid.
#[cfg(any(test, feature = "solver-qualification"))]
struct PcInversionBatchInputs<'a> {
    aligned_means_3d: &'a [f64],
    aligned_covariances_3d: &'a [f64],
    weights: &'a [f64],
    row_offsets: &'a [usize],
    target_positions_eci: &'a [f64],
    planes: &'a [f64],
    target_covariances_3d: &'a [f64],
    areas_km2: &'a [f64],
    deterministic_masses: &'a [f64],
    hit_probability: f64,
    grain_mass_kg: f64,
    grains_per_packet: u64,
    covariance_minimum: f64,
    covariance_maximum: f64,
    radial_samples: usize,
    angular_samples: usize,
    small_area_eta_max: f64,
}

#[cfg(any(test, feature = "solver-qualification"))]
fn pc_inversion_bplane_batch_core(
    inputs: &PcInversionBatchInputs<'_>,
) -> anyhow::Result<Vec<PcInversionBplaneBatchRow>> {
    let aligned_means_3d = inputs.aligned_means_3d;
    let aligned_covs_3d = inputs.aligned_covariances_3d;
    let gmm_weights = inputs.weights;
    let row_offsets = inputs.row_offsets;
    let target_positions_eci = inputs.target_positions_eci;
    let p_bps = inputs.planes;
    let target_covs_3d = inputs.target_covariances_3d;
    let area_km2 = inputs.areas_km2;
    let det_masses = inputs.deterministic_masses;
    let hit_probability = inputs.hit_probability;
    let grain_mass_kg = inputs.grain_mass_kg;
    let grains_per_independent_packet = inputs.grains_per_packet;
    let cov_min_eig = inputs.covariance_minimum;
    let cov_max_eig = inputs.covariance_maximum;
    let radial_samples = inputs.radial_samples;
    let angular_samples = inputs.angular_samples;
    let small_area_eta_max = inputs.small_area_eta_max;
    if row_offsets.len() < 2 || row_offsets.first().copied() != Some(0) {
        return Err(anyhow::anyhow!(
            "row_offsets must start at zero and contain at least two entries"
        ));
    }
    let Some(n_rows) = row_offsets.len().checked_sub(1) else {
        return Err(anyhow::anyhow!("row offset count underflow"));
    };
    let n_components = row_offsets.last().copied().unwrap_or(0);
    let Some(component_mean_len) = n_components.checked_mul(3) else {
        return Err(anyhow::anyhow!("component mean length overflow"));
    };
    let Some(component_covariance_len) = n_components.checked_mul(9) else {
        return Err(anyhow::anyhow!("component covariance length overflow"));
    };
    if aligned_means_3d.len() != component_mean_len
        || aligned_covs_3d.len() != component_covariance_len
        || gmm_weights.len() != n_components
    {
        return Err(anyhow::anyhow!(
            "component arrays must match final row offset"
        ));
    }
    let Some(target_position_len) = n_rows.checked_mul(3) else {
        return Err(anyhow::anyhow!("target position length overflow"));
    };
    let Some(plane_len) = n_rows.checked_mul(6) else {
        return Err(anyhow::anyhow!("B-plane length overflow"));
    };
    let Some(target_covariance_len) = n_rows.checked_mul(9) else {
        return Err(anyhow::anyhow!("target covariance length overflow"));
    };
    if target_positions_eci.len() != target_position_len
        || p_bps.len() != plane_len
        || target_covs_3d.len() != target_covariance_len
        || area_km2.len() != n_rows
        || det_masses.len() != n_rows
    {
        return Err(anyhow::anyhow!("row arrays must match row_offsets length"));
    }
    for offsets in row_offsets.windows(2) {
        let [start, end] = offsets else {
            return Err(anyhow::anyhow!("row offset pair must contain two entries"));
        };
        if end <= start {
            return Err(anyhow::anyhow!(
                "every Pc batch row must contain at least one component"
            ));
        }
    }

    let projected_mean_len = n_components
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("projected mean length overflow"))?;
    let projected_covariance_len = n_components
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("projected covariance length overflow"))?;
    let debris_covariance_len = n_rows
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("debris covariance length overflow"))?;
    let mut projected_means = vec![0.0; projected_mean_len];
    let mut projected_covs = vec![0.0; projected_covariance_len];
    let mut debris_covs = vec![0.0; debris_covariance_len];
    let mut projection_clamped = vec![0_usize; n_rows];
    for (
        ((((offsets, target_position), plane), target_covariance), debris_covariance),
        clamp_count,
    ) in row_offsets
        .windows(2)
        .zip(target_positions_eci.chunks_exact(3))
        .zip(p_bps.chunks_exact(6))
        .zip(target_covs_3d.chunks_exact(9))
        .zip(debris_covs.chunks_exact_mut(4))
        .zip(&mut projection_clamped)
    {
        let [start, end] = offsets else {
            return Err(anyhow::anyhow!("row offset pair must contain two entries"));
        };
        let mean_range = packed_range(*start, *end, 3)
            .ok_or_else(|| anyhow::anyhow!("aligned mean range overflow"))?;
        let covariance_range = packed_range(*start, *end, 9)
            .ok_or_else(|| anyhow::anyhow!("aligned covariance range overflow"))?;
        let projected_mean_range = packed_range(*start, *end, 2)
            .ok_or_else(|| anyhow::anyhow!("projected mean range overflow"))?;
        let projected_covariance_range = packed_range(*start, *end, 4)
            .ok_or_else(|| anyhow::anyhow!("projected covariance range overflow"))?;
        let aligned_means = aligned_means_3d
            .get(mean_range)
            .ok_or_else(|| anyhow::anyhow!("aligned mean range out of bounds"))?;
        let aligned_covariances = aligned_covs_3d
            .get(covariance_range)
            .ok_or_else(|| anyhow::anyhow!("aligned covariance range out of bounds"))?;
        let projected_mean_output = projected_means
            .get_mut(projected_mean_range)
            .ok_or_else(|| anyhow::anyhow!("projected mean range out of bounds"))?;
        let projected_covariance_output = projected_covs
            .get_mut(projected_covariance_range)
            .ok_or_else(|| anyhow::anyhow!("projected covariance range out of bounds"))?;
        *clamp_count = project_bplane_components_core(
            &BplaneProjectionInputs {
                aligned_means_3d: aligned_means,
                aligned_covs_3d: aligned_covariances,
                target_pos_eci: target_position,
                plane,
                target_cov_3d: target_covariance,
                cov_min_eig,
                cov_max_eig,
            },
            BplaneProjectionOutputs {
                projected_means: projected_mean_output,
                projected_covariances: projected_covariance_output,
                debris_covariance,
            },
        )?;
    }

    let mut rows = Vec::with_capacity(n_rows);
    for ((((offsets, &projection_clamped), debris_covariance), &area), &det_mass) in row_offsets
        .windows(2)
        .zip(&projection_clamped)
        .zip(debris_covs.chunks_exact(4))
        .zip(area_km2)
        .zip(det_masses)
    {
        let [start, end] = offsets else {
            return Err(anyhow::anyhow!("row offset pair must contain two entries"));
        };
        let projected_mean_range = packed_range(*start, *end, 2)
            .ok_or_else(|| anyhow::anyhow!("projected mean range overflow"))?;
        let projected_covariance_range = packed_range(*start, *end, 4)
            .ok_or_else(|| anyhow::anyhow!("projected covariance range overflow"))?;
        let projected_mean = projected_means
            .get(projected_mean_range)
            .ok_or_else(|| anyhow::anyhow!("projected mean range out of bounds"))?;
        let projected_covariance = projected_covs
            .get(projected_covariance_range)
            .ok_or_else(|| anyhow::anyhow!("projected covariance range out of bounds"))?;
        let weights = gmm_weights
            .get(*start..*end)
            .ok_or_else(|| anyhow::anyhow!("weight range out of bounds"))?;
        let Ok(debris_covariance_array) = <[f64; 4]>::try_from(debris_covariance) else {
            return Err(anyhow::anyhow!(
                "debris covariance row must contain four values"
            ));
        };
        rows.push(PcInversionBplaneBatchRow {
            projection_clamped,
            debris_cov: debris_covariance_array,
            pc: pc_inversion_mass_core(&PcInversionInputs {
                debris_covariance,
                projected_means: projected_mean,
                projected_covariances: projected_covariance,
                weights,
                component_count: end
                    .checked_sub(*start)
                    .ok_or_else(|| anyhow::anyhow!("component count underflow"))?,
                area_km2: area,
                hit_probability,
                deterministic_mass: det_mass,
                grain_mass_kg,
                grains_per_packet: grains_per_independent_packet,
                covariance_minimum: cov_min_eig,
                covariance_maximum: cov_max_eig,
                radial_samples,
                angular_samples,
                small_area_eta_max,
            })?,
        });
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Boundary A′ slice A1 — fused Pc B-plane geometry + solve driver.
//
// Ports the per-row Python geometry loop of
// `release_mass.py::precompute_release_mass_pc_batch` (B-plane basis, weight
// normalization, 3-D mean/cov block extraction) into the crate so the caller
// crosses the boundary ONCE per event with the raw cloud SoA. The RIC
// target covariance stays a caller-memoized input column (design A′.3: BLAS
// dgemm 3x3 is not bit-replicable by sequential loops).
//
// Replication discipline (design A′.5 A-R1/A-R2):
// * `np.linalg.norm(3-vec)` == sqrt(x*x + y*y + z*z) sequential (measured
//   0/200k mismatches on arm64; TC x86 L3 gate required before default-on).
// * `np.cross` == mul/sub component formula (measured 0/200k mismatches).
// * No `powi(2)`; squares are explicit multiplication.
// * Weight normalization replicates the <=8-component sequential-Python branch
//   of `mass_math.normalize_mixture_weights` exactly (zero-clamp, sequential
//   sum, multiply by reciprocal); >8 components is a fail-loud error.
//   Unreachable while `dust_splitting_rs::MAX_COMPONENTS` stays at 7, but
//   the literal 8 below is NOT derived from that constant — raising the
//   native GMM cap silently makes this branch reachable.
// ---------------------------------------------------------------------------

/// Exact replication of `mass_math.normalize_mixture_weights`'s `size <= 8`
/// branch. Errors mirror the Python `ValueError` messages verbatim.
#[cfg(any(test, feature = "solver-qualification"))]
fn normalize_mixture_weights_small(weights: &[f64], out: &mut Vec<f64>) -> anyhow::Result<()> {
    if weights.is_empty() {
        return Err(anyhow::anyhow!(
            "Dust GMM weights are invalid: no positive finite mass"
        ));
    }
    if weights.len() > 8 {
        return Err(anyhow::anyhow!(
            "pc_inversion_bplane_from_states: component count exceeds the sequential \
             normalization capacity (8); native GMM capacity is 1..=7"
        ));
    }
    let start = out.len();
    let mut total = 0.0_f64;
    for &raw in weights {
        let value = if raw.is_finite() && raw > 0.0 {
            raw
        } else {
            0.0
        };
        out.push(value);
        total += value;
    }
    if !total.is_finite() || total <= 0.0 {
        out.truncate(start);
        return Err(anyhow::anyhow!(
            "Dust GMM weights are invalid: no positive finite mass"
        ));
    }
    let inv_total = 1.0 / total;
    let Some(appended_weights) = out.get_mut(start..) else {
        return Err(anyhow::anyhow!("normalized weight range is out of bounds"));
    };
    for value in appended_weights {
        *value *= inv_total;
    }
    Ok(())
}

/// Component cross product replicating `np.cross` for 3-vectors bit-for-bit
/// (mul then sub; no FMA contraction — expressions kept in a dedicated fn so
/// the x86 `-fp-contract=on` scope stays auditable, design A-R1).
#[inline]
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    let [a0, a1, a2] = *a;
    let [b0, b1, b2] = *b;
    [a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0]
}

/// Sequential Euclidean norm replicating `np.linalg.norm` for 3-vectors.
#[inline]
fn norm3(v: &[f64; 3]) -> f64 {
    let [x, y, z] = *v;
    let x_square = x * x;
    let y_square = y * y;
    let z_square = z * z;
    (x_square + y_square + z_square).sqrt()
}

/// Exact replication of the per-row B-plane basis construction in
/// `release_mass.py::precompute_release_mass_pc_batch` (`v_rel` -> `v_hat` ->
/// `arb` pick -> `b_hat` -> `h_hat`). Returns `[b_hat, h_hat]` flattened (6,).
/// Error strings mirror the Python `ValueError`s verbatim.
pub(crate) fn pc_bplane_basis_from_states(
    v_hf_mean: &[f64],
    target_state: &[f64],
) -> anyhow::Result<[f64; 6]> {
    let Ok([velocity_x, velocity_y, velocity_z]) = <[f64; 3]>::try_from(v_hf_mean) else {
        return Err(anyhow::anyhow!(
            "HF cloud velocity authority must be finite with shape (3,)"
        ));
    };
    if ![velocity_x, velocity_y, velocity_z]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(anyhow::anyhow!(
            "HF cloud velocity authority must be finite with shape (3,)"
        ));
    }
    let Ok([_, _, _, reference_x, reference_y, reference_z]) = <[f64; 6]>::try_from(target_state)
    else {
        return Err(anyhow::anyhow!(
            "target_state must be finite with shape (6,)"
        ));
    };
    let v_rel = [
        velocity_x - reference_x,
        velocity_y - reference_y,
        velocity_z - reference_z,
    ];
    let v_rel_norm = norm3(&v_rel);
    if v_rel_norm < 1e-12 {
        return Err(anyhow::anyhow!(
            "Degenerate encounter geometry: |v_rel| too small for B-plane"
        ));
    }
    let [relative_x, relative_y, relative_z] = v_rel;
    let v_hat = [
        relative_x / v_rel_norm,
        relative_y / v_rel_norm,
        relative_z / v_rel_norm,
    ];
    // `np.dot(v_hat, arb)` with a unit basis vector is the bare component.
    let [v_hat_x, _, _] = v_hat;
    let arb: [f64; 3] = if v_hat_x.abs() > 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let b_raw = cross3(&v_hat, &arb);
    let b_norm = norm3(&b_raw);
    let [b_raw_x, b_raw_y, b_raw_z] = b_raw;
    let b_hat = [b_raw_x / b_norm, b_raw_y / b_norm, b_raw_z / b_norm];
    let h_hat = cross3(&v_hat, &b_hat);
    let [b_hat_x, b_hat_y, b_hat_z] = b_hat;
    let [h_hat_x, h_hat_y, h_hat_z] = h_hat;
    Ok([b_hat_x, b_hat_y, b_hat_z, h_hat_x, h_hat_y, h_hat_z])
}

/// Inputs for fused from-states Pc inversion.
#[cfg(any(test, feature = "solver-qualification"))]
#[derive(Clone, Copy)]
pub struct PcInversionFromStatesInputs<'a> {
    pub gmm_means6: &'a [f64],
    pub gmm_covariances6: &'a [f64],
    pub raw_weights: &'a [f64],
    pub row_offsets: &'a [usize],
    pub target_states: &'a [f64],
    pub hf_velocity_means: &'a [f64],
    pub target_covariances_3d: &'a [f64],
    pub areas_km2: &'a [f64],
    pub deterministic_masses: &'a [f64],
    pub hit_probability: f64,
    pub grain_mass_kg: f64,
    pub grains_per_packet: u64,
    pub covariance_minimum: f64,
    pub covariance_maximum: f64,
    pub radial_samples: usize,
    pub angular_samples: usize,
    pub small_area_eta_max: f64,
}

/// # Errors
///
/// Returns an error for malformed row packing, nonfinite state/science inputs,
/// degenerate encounter geometry, or downstream projection/Pc failure.
#[cfg(any(test, feature = "solver-qualification"))]
pub fn pc_inversion_bplane_from_states_batch_core(
    inputs: &PcInversionFromStatesInputs<'_>,
) -> anyhow::Result<(Vec<PcInversionBplaneBatchRow>, Vec<f64>)> {
    let gmm_means6 = inputs.gmm_means6;
    let gmm_covs6 = inputs.gmm_covariances6;
    let gmm_weights_raw = inputs.raw_weights;
    let row_offsets = inputs.row_offsets;
    let target_states = inputs.target_states;
    let v_hf_means = inputs.hf_velocity_means;
    let target_covs_3d = inputs.target_covariances_3d;
    let area_km2 = inputs.areas_km2;
    let det_masses = inputs.deterministic_masses;
    let hit_probability = inputs.hit_probability;
    let grain_mass_kg = inputs.grain_mass_kg;
    let grains_per_independent_packet = inputs.grains_per_packet;
    let cov_min_eig = inputs.covariance_minimum;
    let cov_max_eig = inputs.covariance_maximum;
    let radial_samples = inputs.radial_samples;
    let angular_samples = inputs.angular_samples;
    let small_area_eta_max = inputs.small_area_eta_max;
    if row_offsets.len() < 2 || row_offsets.first().copied() != Some(0) {
        return Err(anyhow::anyhow!(
            "row_offsets must start at zero and contain at least two entries"
        ));
    }
    let n_rows = row_offsets
        .len()
        .checked_sub(1)
        .ok_or_else(|| anyhow::anyhow!("row count underflow"))?;
    let n_components = row_offsets.last().copied().unwrap_or(0);
    let mean6_len = n_components
        .checked_mul(6)
        .ok_or_else(|| anyhow::anyhow!("cloud mean length overflow"))?;
    let covariance6_len = n_components
        .checked_mul(36)
        .ok_or_else(|| anyhow::anyhow!("cloud covariance length overflow"))?;
    if gmm_means6.len() != mean6_len
        || gmm_covs6.len() != covariance6_len
        || gmm_weights_raw.len() != n_components
    {
        return Err(anyhow::anyhow!(
            "cloud SoA arrays must match final row offset"
        ));
    }
    let target_state_len = n_rows
        .checked_mul(6)
        .ok_or_else(|| anyhow::anyhow!("target state length overflow"))?;
    let velocity_mean_len = n_rows
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("velocity mean length overflow"))?;
    let target_covariance_len = n_rows
        .checked_mul(9)
        .ok_or_else(|| anyhow::anyhow!("target covariance length overflow"))?;
    if target_states.len() != target_state_len
        || v_hf_means.len() != velocity_mean_len
        || target_covs_3d.len() != target_covariance_len
        || area_km2.len() != n_rows
        || det_masses.len() != n_rows
    {
        return Err(anyhow::anyhow!("row arrays must match row_offsets length"));
    }
    if !(hit_probability.is_finite() && 0.0 < hit_probability && hit_probability < 1.0) {
        return Err(anyhow::anyhow!(
            "hit_probability must lie in (0, 1), got {hit_probability}"
        ));
    }
    for offsets in row_offsets.windows(2) {
        let [start, end] = offsets else {
            return Err(anyhow::anyhow!("row offset pair must contain two entries"));
        };
        if end <= start {
            return Err(anyhow::anyhow!(
                "every Pc batch row must contain at least one component"
            ));
        }
    }
    // Ingress validation consumed ONCE per event (design A′.1): these replace
    // the per-row `as_contract_array` / `validate_mass_pipeline_inputs` value
    // checks of the Python loop with batched scans.
    for ((target_state, &det_mass), &area) in
        target_states.chunks_exact(6).zip(det_masses).zip(area_km2)
    {
        if !det_mass.is_finite() || det_mass <= 0.0 {
            return Err(anyhow::anyhow!(
                "det_mass must be positive and finite, got {det_mass}"
            ));
        }
        if !area.is_finite() || area <= 0.0 {
            return Err(anyhow::anyhow!(
                "area_km2 must be positive and finite, got {area}"
            ));
        }
        if !target_state.iter().all(|value| value.is_finite()) {
            return Err(anyhow::anyhow!(
                "target_state must be finite with shape (6,)"
            ));
        }
    }

    let aligned_mean_len = n_components
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("aligned mean length overflow"))?;
    let aligned_covariance_len = n_components
        .checked_mul(9)
        .ok_or_else(|| anyhow::anyhow!("aligned covariance length overflow"))?;
    let plane_len = n_rows
        .checked_mul(6)
        .ok_or_else(|| anyhow::anyhow!("plane length overflow"))?;
    let mut aligned_means_3d = vec![0.0_f64; aligned_mean_len];
    let mut aligned_covs_3d = vec![0.0_f64; aligned_covariance_len];
    let mut weights_norm: Vec<f64> = Vec::with_capacity(n_components);
    let mut planes = vec![0.0_f64; plane_len];

    for (mean6, aligned_mean) in gmm_means6
        .chunks_exact(6)
        .zip(aligned_means_3d.chunks_exact_mut(3))
    {
        let source = mean6
            .get(..3)
            .ok_or_else(|| anyhow::anyhow!("cloud mean row missing position"))?;
        aligned_mean.copy_from_slice(source);
    }
    for (covariance6, aligned_covariance) in gmm_covs6
        .chunks_exact(36)
        .zip(aligned_covs_3d.chunks_exact_mut(9))
    {
        for (source_row, output_row) in covariance6
            .chunks_exact(6)
            .take(3)
            .zip(aligned_covariance.chunks_exact_mut(3))
        {
            let source = source_row
                .get(..3)
                .ok_or_else(|| anyhow::anyhow!("cloud covariance row missing 3D block"))?;
            output_row.copy_from_slice(source);
        }
    }
    for offsets in row_offsets.windows(2) {
        let [start, end] = offsets else {
            return Err(anyhow::anyhow!("row offset pair must contain two entries"));
        };
        let weights = gmm_weights_raw
            .get(*start..*end)
            .ok_or_else(|| anyhow::anyhow!("weight row range out of bounds"))?;
        normalize_mixture_weights_small(weights, &mut weights_norm)?;
    }
    for ((velocity_mean, target_state), plane) in v_hf_means
        .chunks_exact(3)
        .zip(target_states.chunks_exact(6))
        .zip(planes.chunks_exact_mut(6))
    {
        let basis = pc_bplane_basis_from_states(velocity_mean, target_state)?;
        plane.copy_from_slice(&basis);
    }
    let target_positions: Vec<f64> = target_states
        .chunks_exact(6)
        .flat_map(|state| state.iter().take(3).copied())
        .collect();
    let rows = pc_inversion_bplane_batch_core(&PcInversionBatchInputs {
        aligned_means_3d: &aligned_means_3d,
        aligned_covariances_3d: &aligned_covs_3d,
        weights: &weights_norm,
        row_offsets,
        target_positions_eci: &target_positions,
        planes: &planes,
        target_covariances_3d: target_covs_3d,
        areas_km2: area_km2,
        deterministic_masses: det_masses,
        hit_probability,
        grain_mass_kg,
        grains_per_packet: grains_per_independent_packet,
        covariance_minimum: cov_min_eig,
        covariance_maximum: cov_max_eig,
        radial_samples,
        angular_samples,
        small_area_eta_max,
    })?;
    Ok((rows, planes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_validation_uses_anyhow_result() {
        fn assert_anyhow_result(_: anyhow::Result<FinitePacketMassBound>) {}

        assert_anyhow_result(finite_packet_release_mass_bound_core(0.0, 0.9, 1.0, 1.0, 1));
    }

    fn test_index_f64(index: usize) -> f64 {
        u32::try_from(index).map_or(f64::NAN, f64::from)
    }

    fn same_float_bits(left: &[f64], right: &[f64]) -> bool {
        left.iter()
            .map(|value| value.to_bits())
            .eq(right.iter().map(|value| value.to_bits()))
    }

    // -----------------------------------------------------------------
    // Boundary A′ slice A1 — from-states Pc driver replication tests
    // -----------------------------------------------------------------

    #[test]
    fn a1_normalize_weights_matches_python_small_branch() {
        // Python: zero-clamp non-finite / non-positive, sequential sum,
        // multiply by reciprocal.
        let mut out = Vec::new();
        normalize_mixture_weights_small(&[0.2, f64::NAN, -1.0, 0.6], &mut out)
            .expect("valid weights");
        let total = 0.2 + 0.0 + 0.0 + 0.6;
        let inv = 1.0 / total;
        assert!(same_float_bits(&out, &[0.2 * inv, 0.0, 0.0, 0.6 * inv]));
    }

    #[test]
    fn a1_normalize_weights_fail_loud() {
        let mut out = Vec::new();
        let err = normalize_mixture_weights_small(&[0.0, -2.0, f64::NAN], &mut out)
            .expect_err("no positive mass");
        assert!(err.to_string().contains("no positive finite mass"), "{err}");
        assert!(out.is_empty(), "failed rows must not leave partial output");
        let err = normalize_mixture_weights_small(&[1.0; 9], &mut out)
            .expect_err(">8 components must fail loud (A-R5)");
        assert!(err.to_string().contains("capacity"), "{err}");
    }

    #[test]
    fn a1_bplane_basis_matches_release_mass_formula() {
        let v_hf = [1.25, -3.5, 0.75];
        let target = [7000.0, -12.0, 340.0, 0.5, 6.5, -1.5];
        let basis = pc_bplane_basis_from_states(&v_hf, &target).expect("well-posed");
        // Manual replication of release_mass.py:439-449.
        let [v_hf_x, v_hf_y, v_hf_z] = v_hf;
        let [_, _, _, target_u, target_v, target_w] = target;
        let v_rel = [v_hf_x - target_u, v_hf_y - target_v, v_hf_z - target_w];
        let norm = norm3(&v_rel);
        let [rel_x, rel_y, rel_z] = v_rel;
        let v_hat = [rel_x / norm, rel_y / norm, rel_z / norm];
        let [v_hat_x, _, _] = v_hat;
        assert!(
            v_hat_x.abs() <= 0.9,
            "test vector must take the arb=x branch"
        );
        let b_raw = cross3(&v_hat, &[1.0, 0.0, 0.0]);
        let b_norm = norm3(&b_raw);
        let [b_raw_x, b_raw_y, b_raw_z] = b_raw;
        let b_hat = [b_raw_x / b_norm, b_raw_y / b_norm, b_raw_z / b_norm];
        let h_hat = cross3(&v_hat, &b_hat);
        let [b0, b1, b2] = b_hat;
        let [h0, h1, h2] = h_hat;
        assert!(same_float_bits(&basis, &[b0, b1, b2, h0, h1, h2]));
    }

    #[test]
    fn a1_bplane_basis_arb_switch_branch() {
        // v_rel dominated by x => |v_hat[0]| > 0.9 => arb flips to y.
        let v_hf = [10.0, 0.1, 0.05];
        let target = [7000.0, -12.0, 340.0, 0.0, 0.0, 0.0];
        let basis = pc_bplane_basis_from_states(&v_hf, &target).expect("well-posed");
        let norm = norm3(&[10.0, 0.1, 0.05]);
        let v_hat = [10.0 / norm, 0.1 / norm, 0.05 / norm];
        let [v_hat_x, _, _] = v_hat;
        assert!(v_hat_x.abs() > 0.9);
        let b_raw = cross3(&v_hat, &[0.0, 1.0, 0.0]);
        let b_norm = norm3(&b_raw);
        let [basis_x, ..] = basis;
        let [b_raw_x, ..] = b_raw;
        assert_eq!(basis_x.to_bits(), (b_raw_x / b_norm).to_bits());
    }

    #[test]
    fn a1_bplane_basis_fail_loud() {
        let err = pc_bplane_basis_from_states(&[f64::NAN, 0.0, 0.0], &[0.0; 6])
            .expect_err("non-finite v_hf");
        assert!(err.to_string().contains("finite with shape (3,)"), "{err}");
        let err = pc_bplane_basis_from_states(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
            .expect_err("degenerate v_rel");
        assert!(
            err.to_string().contains("Degenerate encounter geometry"),
            "{err}"
        );
    }

    #[test]
    fn a1_from_states_core_equals_prenormalized_core() {
        // The fused driver must reproduce pc_inversion_bplane_batch_core fed
        // with the Python-equivalent pre-normalized inputs, byte-for-byte.
        let n_rows = 2usize;
        let nc = 3usize;
        let row_offsets = [0usize, nc, 2 * nc];
        let mut gmm_means6 = vec![0.0; 2 * nc * 6];
        let mut gmm_covs6 = vec![0.0; 2 * nc * 36];
        let mut weights_raw = vec![0.0; 2 * nc];
        for (component, ((mean6, covariance6), weight)) in gmm_means6
            .chunks_exact_mut(6)
            .zip(gmm_covs6.chunks_exact_mut(36))
            .zip(&mut weights_raw)
            .enumerate()
        {
            let component_f64 = test_index_f64(component);
            for (axis, mean) in mean6.iter_mut().enumerate() {
                let axis_f64 = test_index_f64(axis);
                *mean = 10.0 + component_f64 * 0.37 + axis_f64 * 0.11;
            }
            for (axis, row) in covariance6.chunks_exact_mut(6).enumerate() {
                // Diagonal-dominant SPD 6x6; only the 3x3 block is consumed.
                if let Some(diagonal) = row.get_mut(axis) {
                    *diagonal = 2.0 + 0.1 * test_index_f64(axis);
                }
            }
            if let Some(covariance_xy) = covariance6.get_mut(1) {
                *covariance_xy = 0.05;
            }
            if let Some(covariance_yx) = covariance6.get_mut(6) {
                *covariance_yx = 0.05;
            }
            *weight = 0.3 + 0.2 * test_index_f64(component % nc);
        }
        let target_states = [
            10.2, 11.4, 9.9, 0.4, -0.2, 0.1, //
            10.5, 11.0, 10.3, -0.3, 0.25, -0.15,
        ];
        let v_hf_means = [1.2, 0.4, -0.6, -0.8, 1.1, 0.5];
        let mut target_covs_3d = vec![0.0; n_rows * 9];
        for covariance in target_covs_3d.chunks_exact_mut(9) {
            for (axis, row) in covariance.chunks_exact_mut(3).enumerate() {
                if let Some(diagonal) = row.get_mut(axis) {
                    *diagonal = 1e-4;
                }
            }
        }
        let area_km2 = [1.0e-3, 2.0e-3];
        let det_masses = [5.0, 7.5];
        let (rows_fused, planes) =
            pc_inversion_bplane_from_states_batch_core(&PcInversionFromStatesInputs {
                gmm_means6: &gmm_means6,
                gmm_covariances6: &gmm_covs6,
                raw_weights: &weights_raw,
                row_offsets: &row_offsets,
                target_states: &target_states,
                hf_velocity_means: &v_hf_means,
                target_covariances_3d: &target_covs_3d,
                areas_km2: &area_km2,
                deterministic_masses: &det_masses,
                hit_probability: 0.9,
                grain_mass_kg: 1.0e-6,
                grains_per_packet: 1000,
                covariance_minimum: 1e-12,
                covariance_maximum: 1e12,
                radial_samples: 8,
                angular_samples: 8,
                small_area_eta_max: 5.0,
            })
            .expect("fused driver");
        // Pre-normalized replication (what the Python loop fed the old entry).
        let mut aligned_means = vec![0.0; 2 * nc * 3];
        let mut aligned_covs = vec![0.0; 2 * nc * 9];
        let mut weights_norm = Vec::new();
        for offset_pair in row_offsets.windows(2) {
            let &[start, end] = offset_pair else {
                continue;
            };
            if let Some(row_weights) = weights_raw.get(start..end) {
                normalize_mixture_weights_small(row_weights, &mut weights_norm).expect("weights");
            }
        }
        for (aligned_mean, mean6) in aligned_means
            .chunks_exact_mut(3)
            .zip(gmm_means6.chunks_exact(6))
        {
            if let Some(position) = mean6.get(..3) {
                aligned_mean.copy_from_slice(position);
            }
        }
        for (aligned_covariance, covariance6) in aligned_covs
            .chunks_exact_mut(9)
            .zip(gmm_covs6.chunks_exact(36))
        {
            for (aligned_row, source_row) in aligned_covariance
                .chunks_exact_mut(3)
                .zip(covariance6.chunks_exact(6))
                .take(3)
            {
                if let Some(source_prefix) = source_row.get(..3) {
                    aligned_row.copy_from_slice(source_prefix);
                }
            }
        }
        let target_positions = target_states
            .chunks_exact(6)
            .flat_map(|state| state.iter().take(3).copied())
            .collect::<Vec<_>>();
        let rows_direct = pc_inversion_bplane_batch_core(&PcInversionBatchInputs {
            aligned_means_3d: &aligned_means,
            aligned_covariances_3d: &aligned_covs,
            weights: &weights_norm,
            row_offsets: &row_offsets,
            target_positions_eci: &target_positions,
            planes: &planes,
            target_covariances_3d: &target_covs_3d,
            areas_km2: &area_km2,
            deterministic_masses: &det_masses,
            hit_probability: 0.9,
            grain_mass_kg: 1.0e-6,
            grains_per_packet: 1000,
            covariance_minimum: 1e-12,
            covariance_maximum: 1e12,
            radial_samples: 8,
            angular_samples: 8,
            small_area_eta_max: 5.0,
        })
        .expect("direct core");
        assert_eq!(rows_fused.len(), rows_direct.len());
        for (fused, direct) in rows_fused.iter().zip(rows_direct.iter()) {
            assert_eq!(fused.pc.0.to_bits(), direct.pc.0.to_bits(), "mass bits");
            assert!(
                same_float_bits(&fused.debris_cov, &direct.debris_cov),
                "debris cov"
            );
            assert_eq!(fused.projection_clamped, direct.projection_clamped);
        }
    }

    #[test]
    fn project_bplane_components_core_reachable_from_native_slices() {
        // Buffer-free native reachability check for the internal projection
        // core: a single component with an identity 3x3 covariance projected
        // onto the x/y axes must yield the plain (x, y) mean and a 2x2 identity
        // covariance, with no eigenvalue clamping. Proves the b-plane
        // projection math is callable from a Rust caller with plain `&[f64]`
        // in / `&mut [f64]` out (no numpy, no pyo3).
        let aligned_means_3d = [1.0_f64, 2.0, 3.0];
        let aligned_covs_3d = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let target_pos_eci = [0.0_f64, 0.0, 0.0];
        // 2x3 projection: row 0 = x-axis, row 1 = y-axis.
        let p_bp = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0];
        let target_cov_3d = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut proj_means = [0.0_f64; 2];
        let mut proj_covs = [0.0_f64; 4];
        let mut debris_cov = [0.0_f64; 4];
        let clamped = project_bplane_components_core(
            &BplaneProjectionInputs {
                aligned_means_3d: &aligned_means_3d,
                aligned_covs_3d: &aligned_covs_3d,
                target_pos_eci: &target_pos_eci,
                plane: &p_bp,
                target_cov_3d: &target_cov_3d,
                cov_min_eig: 1e-12,
                cov_max_eig: 1e12,
            },
            BplaneProjectionOutputs {
                projected_means: &mut proj_means,
                projected_covariances: &mut proj_covs,
                debris_covariance: &mut debris_cov,
            },
        )
        .expect("trivial projection must succeed");
        assert_eq!(clamped, 0);
        assert!(same_float_bits(&proj_means, &[1.0, 2.0]));
        assert!(same_float_bits(&proj_covs, &[1.0, 0.0, 0.0, 1.0]));
        assert!(same_float_bits(&debris_cov, &[1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn pc_inversion_bplane_batch_matches_scalar_rows() {
        let means = [
            7000.4, -0.002, 0.001, 7000.501, 0.003, -0.002, 7000.499, -0.001, 0.001,
        ];
        let covs = [
            1.0e-4, 0.0, 0.0, 0.0, 2.0e-4, 0.0, 0.0, 0.0, 3.0e-4, 2.0e-4, 1.0e-5, 0.0, 1.0e-5,
            3.0e-4, 0.0, 0.0, 0.0, 4.0e-4, 3.0e-4, 0.0, 2.0e-5, 0.0, 4.0e-4, 0.0, 2.0e-5, 0.0,
            5.0e-4,
        ];
        let weights = [1.0, 0.35, 0.65];
        let offsets = [0_usize, 1, 3];
        let target_positions = [7000.0, 0.0, 0.0, 7000.5, 0.0, 0.0];
        let p_bps = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let target_covs = [
            8.0e-5, 0.0, 0.0, 0.0, 9.0e-5, 0.0, 0.0, 0.0, 1.0e-4, 1.1e-4, 0.0, 0.0, 0.0, 1.2e-4,
            0.0, 0.0, 0.0, 1.3e-4,
        ];
        let areas = [1.0e-6, 2.0e-6];
        let det_masses = [0.25, 0.5];
        let common = (0.95, 6.45e-10, 3_u64, 1.0e-12, 1.0e12, 24, 100, 0.01);

        let batch = pc_inversion_bplane_batch_core(&PcInversionBatchInputs {
            aligned_means_3d: &means,
            aligned_covariances_3d: &covs,
            weights: &weights,
            row_offsets: &offsets,
            target_positions_eci: &target_positions,
            planes: &p_bps,
            target_covariances_3d: &target_covs,
            areas_km2: &areas,
            deterministic_masses: &det_masses,
            hit_probability: common.0,
            grain_mass_kg: common.1,
            grains_per_packet: common.2,
            covariance_minimum: common.3,
            covariance_maximum: common.4,
            radial_samples: common.5,
            angular_samples: common.6,
            small_area_eta_max: common.7,
        })
        .expect("batch must resolve");
        assert_eq!(batch.len(), 2);

        let row_inputs = offsets
            .windows(2)
            .zip(target_positions.chunks_exact(3))
            .zip(p_bps.chunks_exact(6))
            .zip(target_covs.chunks_exact(9))
            .zip(&areas)
            .zip(&det_masses)
            .zip(&batch);
        for (
            (((((offset_pair, target_position), plane), target_covariance), area), det_mass),
            batch_row,
        ) in row_inputs
        {
            let &[start, end] = offset_pair else {
                continue;
            };
            let n_components = end - start;
            let mut projected_means = vec![0.0; n_components * 2];
            let mut projected_covs = vec![0.0; n_components * 4];
            let mut debris_cov = [0.0; 4];
            let Some(mean_range) = packed_range(start, end, 3) else {
                continue;
            };
            let Some(covariance_range) = packed_range(start, end, 9) else {
                continue;
            };
            let Some(row_means) = means.get(mean_range) else {
                continue;
            };
            let Some(row_covariances) = covs.get(covariance_range) else {
                continue;
            };
            let Some(row_weights) = weights.get(start..end) else {
                continue;
            };
            let projection_clamped = project_bplane_components_core(
                &BplaneProjectionInputs {
                    aligned_means_3d: row_means,
                    aligned_covs_3d: row_covariances,
                    target_pos_eci: target_position,
                    plane,
                    target_cov_3d: target_covariance,
                    cov_min_eig: common.3,
                    cov_max_eig: common.4,
                },
                BplaneProjectionOutputs {
                    projected_means: &mut projected_means,
                    projected_covariances: &mut projected_covs,
                    debris_covariance: &mut debris_cov,
                },
            )
            .expect("scalar projection must resolve");
            let scalar = pc_inversion_mass_core(&PcInversionInputs {
                debris_covariance: &debris_cov,
                projected_means: &projected_means,
                projected_covariances: &projected_covs,
                weights: row_weights,
                component_count: n_components,
                area_km2: *area,
                hit_probability: common.0,
                deterministic_mass: *det_mass,
                grain_mass_kg: common.1,
                grains_per_packet: common.2,
                covariance_minimum: common.3,
                covariance_maximum: common.4,
                radial_samples: common.5,
                angular_samples: common.6,
                small_area_eta_max: common.7,
            })
            .expect("scalar Pc must resolve");

            assert_eq!(batch_row.projection_clamped, projection_clamped);
            assert!(same_float_bits(&batch_row.debris_cov, &debris_cov));
            assert_eq!(batch_row.pc.0.to_bits(), scalar.0.to_bits());
            assert_eq!(batch_row.pc.1, scalar.1);
        }
    }

    #[test]
    fn pc_inversion_rejects_unrepresentable_finite_packet_count() {
        let error = pc_inversion_mass_core(&PcInversionInputs {
            debris_covariance: &[0.0, 0.0, 0.0, 0.0],
            projected_means: &[0.0, 0.0],
            projected_covariances: &[1.0, 0.0, 0.0, 1.0],
            weights: &[1.0],
            component_count: 1,
            area_km2: 1.0e-20,
            hit_probability: 0.99,
            deterministic_mass: 1.0,
            grain_mass_kg: 6.45e-10,
            grains_per_packet: 1,
            covariance_minimum: 1.0e-12,
            covariance_maximum: 1.0e12,
            radial_samples: 24,
            angular_samples: 100,
            small_area_eta_max: 0.01,
        })
        .expect_err("packet count beyond u64 must fail closed");

        assert!(error
            .to_string()
            .contains("released packet count exceeds u64"));
    }

    /// A widely separated spectrum must keep its small eigenvalue.
    ///
    /// The direct form computed `0.5 * (tr - sqrt(tr^2 - 4*det))`. Here
    /// `tr^2` is 1e32 and `4*det` is 4e10, which is far below one ULP of 1e32,
    /// so the discriminant rounded to `tr^2` exactly, its root to `tr`, and the
    /// small eigenvalue cancelled to zero. It was then clamped to the floor and
    /// nothing reported that the minor variance had been destroyed.
    ///
    /// The oracle is the exact product identity: for a diagonal input the two
    /// eigenvalues ARE the diagonal, so the small one must come back as 1e-6
    /// bit-for-bit rather than as the 1e-30 floor.
    #[test]
    fn sanitize_keeps_the_small_eigenvalue_of_a_separated_spectrum() {
        let (out, clamped) =
            sanitize_covariance_2d_values(1.0e16, 0.0, 0.0, 1.0e-6, 1.0e-30, 1.0e30)
                .expect("well-formed diagonal covariance");
        assert!(
            !clamped,
            "a spectrum inside the bounds must not report a clamp"
        );
        let [out00, out01, out10, out11] = out;
        assert_eq!(
            out11.to_bits(),
            1.0e-6_f64.to_bits(),
            "small eigenvalue was destroyed by cancellation: {out11}"
        );
        assert_eq!(out00.to_bits(), 1.0e16_f64.to_bits(), "out00 = {out00}");
        assert_eq!(out01.to_bits(), 0.0_f64.to_bits());
        assert_eq!(out10.to_bits(), 0.0_f64.to_bits());

        // The same separation with the axes swapped, so the result is a
        // property of the arithmetic and not of which diagonal entry is large.
        let (swapped, _) = sanitize_covariance_2d_values(1.0e-6, 0.0, 0.0, 1.0e16, 1.0e-30, 1.0e30)
            .expect("well-formed diagonal covariance");
        let [swapped00, _, _, swapped11] = swapped;
        assert_eq!(swapped00.to_bits(), 1.0e-6_f64.to_bits());
        assert_eq!(swapped11.to_bits(), 1.0e16_f64.to_bits());
    }

    #[test]
    fn sanitize_preserves_diagonal_covariance_axis_order() {
        // Diagonal input with a > d must come back unchanged, not
        // axis-swapped (pre-existing eigenvector mispairing bug).
        let (out, clamped) = sanitize_covariance_2d_values(1.0, 0.0, 0.0, 1.0e-4, 1.0e-12, 1.0e12)
            .expect("well-formed diagonal covariance");
        assert!(!clamped);
        let [out00, out01, _, out11] = out;
        assert!((out00 - 1.0).abs() < 1e-15, "out00 = {out00}");
        assert!((out11 - 1.0e-4).abs() < 1e-4 * 1e-12, "out11 = {out11}");
        assert_eq!(out01.to_bits(), 0.0_f64.to_bits());
        // And the a < d ordering stays correct too.
        let (out2, _) = sanitize_covariance_2d_values(1.0e-4, 0.0, 0.0, 1.0, 1.0e-12, 1.0e12)
            .expect("well-formed diagonal covariance");
        let [out2_00, _, _, out2_11] = out2;
        assert!((out2_00 - 1.0e-4).abs() < 1e-4 * 1e-12);
        assert!((out2_11 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn pc_inversion_deep_tail_high_aspect_routes_to_quadrature() {
        // eta is tiny (5e-5) but the mean sits 4-sigma out along the short
        // axis of a 100:1 aspect covariance, so the linear area*pdf estimate
        // is curvature-dominated; the old eta-only gate kept it on the
        // small-area path and underestimated Pc by tens of percent.
        let radius_km = 1.0e-3_f64;
        let area = std::f64::consts::PI * radius_km * radius_km;
        let result = pc_inversion_mass_core(&PcInversionInputs {
            debris_covariance: &[0.0, 0.0, 0.0, 0.0],
            projected_means: &[0.0, 0.04],
            projected_covariances: &[1.0, 0.0, 0.0, 1.0e-4],
            weights: &[1.0],
            component_count: 1,
            area_km2: area,
            hit_probability: 0.99,
            deterministic_mass: 1.0,
            grain_mass_kg: 6.45e-10,
            grains_per_packet: 1,
            covariance_minimum: 1.0e-12,
            covariance_maximum: 1.0e12,
            radial_samples: 48,
            angular_samples: 180,
            small_area_eta_max: 0.01,
        })
        .expect("deep-tail geometry must resolve");
        assert_eq!(result.1, "numerical_quadrature");
        assert_eq!(result.5, 1);
        assert!(result.9 > 0.01, "curvature_rel_max = {}", result.9);
    }

    /// A component whose `eta` overflows must not take the SMALL-AREA branch.
    ///
    /// `eta` is the disk area over the covariance scale, so `eta = inf` is the
    /// most extreme "disk enormous against the covariance" case there is -- the
    /// regime where the small-area expansion is least valid. The old gate read
    /// `if eta.is_finite() && !small_area_valid { quadrature } else { linear }`,
    /// which sent exactly that component to the linear approximation, produced
    /// a non-finite log-weight from it, and then dropped the component silently
    /// -- lowering the capture sum, Pc, and the required release mass, toward
    /// UNDER-protection.
    ///
    /// Reachability, so nobody over- or under-states this: with validated
    /// inputs the overflow needs an absurd disk (the smallest positive
    /// subnormal determinant still leaves the denominator near 1.4e-161, so the
    /// area must exceed ~2.5e147 km^2). It is a latent defect, not an observed
    /// one -- which is why the fix had to be free of bit movement, and is.
    #[test]
    fn pc_inversion_refuses_a_component_whose_eta_overflows() {
        let inputs = PcInversionInputs {
            debris_covariance: &[0.0, 0.0, 0.0, 0.0],
            projected_means: &[0.0, 0.0],
            projected_covariances: &[1.0e-6, 0.0, 0.0, 1.0e-6],
            weights: &[1.0],
            component_count: 1,
            area_km2: f64::MAX,
            hit_probability: 0.99,
            deterministic_mass: 1.0,
            grain_mass_kg: 6.45e-10,
            grains_per_packet: 1,
            covariance_minimum: 1.0e-30,
            covariance_maximum: 1.0e30,
            radial_samples: 48,
            angular_samples: 180,
            small_area_eta_max: 0.01,
        };
        // The OLD code answered Ok here, and that is the whole defect: it took
        // the linear branch, produced a non-finite log-weight, dropped the only
        // component, and reported "unresolved" as though the geometry had
        // simply not resolved -- indistinguishable, to every caller, from an
        // honest non-answer. Under the strict probabilistic mass policy a
        // component that cannot be evaluated is an ERROR, not a silent
        // subtraction from the capture sum.
        let outcome = pc_inversion_mass_core(&inputs);
        assert!(
            outcome.is_err(),
            "expected a refusal; got Ok({:?}), which is the silent-drop path \
             that biases the required release mass downward",
            outcome.map(|result| result.1)
        );
    }

    #[test]
    fn pc_inversion_small_area_second_order_matches_quadrature() {
        // In-gate case: small eta AND small curvature. The corrected
        // small-area value must agree with full quadrature to ~1e-4 rel.
        let radius_km = 1.0e-3_f64;
        let area = std::f64::consts::PI * radius_km * radius_km;
        let small = pc_inversion_mass_core(&PcInversionInputs {
            debris_covariance: &[0.0, 0.0, 0.0, 0.0],
            projected_means: &[0.5, 0.2],
            projected_covariances: &[1.0, 0.0, 0.0, 0.25],
            weights: &[1.0],
            component_count: 1,
            area_km2: area,
            hit_probability: 0.99,
            deterministic_mass: 1.0,
            grain_mass_kg: 6.45e-10,
            grains_per_packet: 1,
            covariance_minimum: 1.0e-12,
            covariance_maximum: 1.0e12,
            radial_samples: 48,
            angular_samples: 180,
            small_area_eta_max: 0.01,
        })
        .expect("small-area geometry must resolve");
        assert_eq!(small.1, "small_area_density");
        // Force quadrature on the identical geometry via a zero eta gate.
        let quad = pc_inversion_mass_core(&PcInversionInputs {
            debris_covariance: &[0.0, 0.0, 0.0, 0.0],
            projected_means: &[0.5, 0.2],
            projected_covariances: &[1.0, 0.0, 0.0, 0.25],
            weights: &[1.0],
            component_count: 1,
            area_km2: area,
            hit_probability: 0.99,
            deterministic_mass: 1.0,
            grain_mass_kg: 6.45e-10,
            grains_per_packet: 1,
            covariance_minimum: 1.0e-12,
            covariance_maximum: 1.0e12,
            radial_samples: 96,
            angular_samples: 360,
            small_area_eta_max: 0.0,
        })
        .expect("quadrature reference must resolve");
        assert_eq!(quad.1, "numerical_quadrature");
        let rel = ((small.2 - quad.2) / quad.2).abs();
        assert!(
            rel < 1.0e-4,
            "rel dev {} (small {}, quad {})",
            rel,
            small.2,
            quad.2
        );
    }

    #[test]
    fn pc_inversion_reports_covariance_clamp_activation() {
        let radius_km = 1.0e-3_f64;
        let area = std::f64::consts::PI * radius_km * radius_km;
        // Component eigenvalue below cov_min_eig forces the clamp.
        let result = pc_inversion_mass_core(&PcInversionInputs {
            debris_covariance: &[0.0, 0.0, 0.0, 0.0],
            projected_means: &[0.1, 0.0],
            projected_covariances: &[1.0, 0.0, 0.0, 1.0e-9],
            weights: &[1.0],
            component_count: 1,
            area_km2: area,
            hit_probability: 0.99,
            deterministic_mass: 1.0,
            grain_mass_kg: 6.45e-10,
            grains_per_packet: 1,
            covariance_minimum: 1.0e-6,
            covariance_maximum: 1.0e6,
            radial_samples: 48,
            angular_samples: 180,
            small_area_eta_max: 0.01,
        })
        .expect("clamped geometry must resolve");
        assert_eq!(result.8, 1, "clamp activation must be reported");
        // Well-conditioned input: no clamp reported.
        let clean = pc_inversion_mass_core(&PcInversionInputs {
            debris_covariance: &[0.0, 0.0, 0.0, 0.0],
            projected_means: &[0.1, 0.0],
            projected_covariances: &[1.0, 0.0, 0.0, 0.25],
            weights: &[1.0],
            component_count: 1,
            area_km2: area,
            hit_probability: 0.99,
            deterministic_mass: 1.0,
            grain_mass_kg: 6.45e-10,
            grains_per_packet: 1,
            covariance_minimum: 1.0e-6,
            covariance_maximum: 1.0e6,
            radial_samples: 48,
            angular_samples: 180,
            small_area_eta_max: 0.01,
        })
        .expect("clean geometry must resolve");
        assert_eq!(clean.8, 0);
    }

    #[test]
    fn pc_inversion_rejects_unit_hit_probability_without_saturation() {
        let result = pc_inversion_mass_core(&PcInversionInputs {
            debris_covariance: &[0.0, 0.0, 0.0, 0.0],
            projected_means: &[0.0, 0.0],
            projected_covariances: &[1.0, 0.0, 0.0, 1.0],
            weights: &[1.0],
            component_count: 1,
            area_km2: 1.0,
            hit_probability: 1.0,
            deterministic_mass: 1.0,
            grain_mass_kg: 6.45e-10,
            grains_per_packet: 1,
            covariance_minimum: 1.0e-12,
            covariance_maximum: 1.0e12,
            radial_samples: 24,
            angular_samples: 100,
            small_area_eta_max: 0.01,
        })
        .expect("invalid probability is a typed infeasible result");

        assert!(result.0.is_infinite());
        assert_eq!(result.1, "invalid_probability");
    }
}

#[test]
fn finite_packet_mass_bound_debug_tc_operands() {
    // Pinned operands from the 2026-07-16 TC pack_w24_006 oracle mismatch
    // (native released 13478532104040 vs CPython oracle 13478532104039).
    // Prints every intermediate's bit pattern so a cross-platform diff can
    // identify the diverging operation; asserts nothing beyond success.
    let p = 3.850_362_835_540_304e-05_f64;
    let target = 0.99_f64;
    let det = 0.334_730_837_494_134_9_f64;
    let gm = 6.450_736_915_371_043e-10_f64;
    let packet_mass = gm * 1.0;
    let ratio = det / packet_mass;
    let required_count = checked_ceil_packet_count(ratio.max(1.0), "required captured");
    assert!(
        required_count.is_ok(),
        "debug operands must yield a finite required count"
    );
    let req = required_count.unwrap_or(0);
    let lfi = -(-target).ln_1p();
    let root = (2.0 * lfi).sqrt();
    let rr = std::hint::black_box(root * root);
    let fr = std::hint::black_box(4.0 * u64_to_f64(req));
    let inner_sqrt = (rr + fr).sqrt();
    let inner = std::hint::black_box(root + inner_sqrt);
    let expected = 0.25 * std::hint::black_box(inner.powi(2));
    let rel = expected / p;
    let na = next_up_positive(rel);
    let released_count = checked_ceil_packet_count(na.max(u64_to_f64(req)), "released");
    assert!(
        released_count.is_ok(),
        "debug operands must yield a finite released count"
    );
    let released = released_count.unwrap_or(0);
    println!("ratio_bits={:016x}", ratio.to_bits());
    println!("req={req}");
    println!("lfi_bits={:016x}", lfi.to_bits());
    println!("root_bits={:016x}", root.to_bits());
    println!("rr_bits={:016x}", rr.to_bits());
    println!("fr_bits={:016x}", fr.to_bits());
    println!("inner_sqrt_bits={:016x}", inner_sqrt.to_bits());
    println!("inner_bits={:016x}", inner.to_bits());
    println!("expected_bits={:016x}", expected.to_bits());
    println!("rel_bits={:016x}", rel.to_bits());
    println!("na_bits={:016x}", na.to_bits());
    println!("released_manual={released}");
    let bound = finite_packet_release_mass_bound_core(p, target, det, gm, 1).expect("bound");
    println!("released_core={}", bound.released_packet_count);
    println!("mass_core_bits={:016x}", bound.release_mass_kg.to_bits());
}

#[test]
fn finite_packet_mass_bound_matches_dimensional_contract() {
    let deterministic_mass = 5.0e-6;
    let grain_mass = deterministic_mass / 5.0;
    let bound = finite_packet_release_mass_bound_core(0.2, 0.95, deterministic_mass, grain_mass, 1)
        .expect("valid finite packet model");
    assert_eq!(bound.required_captured_packets, 5);
    assert!(bound.released_packet_count > 5);
    assert_eq!(
        bound.release_mass_kg.to_bits(),
        (u64_to_f64(bound.released_packet_count) * grain_mass).to_bits()
    );
    assert_eq!(
        bound.expected_captured_packets.to_bits(),
        (u64_to_f64(bound.released_packet_count) * 0.2).to_bits()
    );
}

#[test]
fn finite_packet_mass_bound_does_not_round_down_true_mass_excess() {
    let next_bits = 1.0_f64.to_bits().checked_add(1);
    assert!(
        next_bits.is_some(),
        "one-point-zero bits must have a successor"
    );
    let required_mass = next_bits.map_or(f64::NAN, f64::from_bits);
    let bound = finite_packet_release_mass_bound_core(1.0, 0.95, required_mass, 1.0, 1)
        .expect("valid finite packet model");
    assert_eq!(bound.required_captured_packets, 2);
}

#[test]
fn finite_packet_mass_bound_rejects_nonfinite_release_expectation_without_panicking() {
    let outcome = std::panic::catch_unwind(|| {
        finite_packet_release_mass_bound_core(f64::from_bits(1), 0.95, 1.0, 1.0, 1)
    });
    let result = outcome.expect("finite inputs must not panic at packet-count boundary");
    let error = result.expect_err("unrepresentable released packet count must fail closed");
    assert!(error
        .to_string()
        .contains("released packet count exceeds u64"));
}

#[test]
fn finite_packet_mass_bound_rejects_exact_f64_u64_upper_boundary() {
    let error = finite_packet_release_mass_bound_core(
        1.0,
        f64::MIN_POSITIVE,
        18_446_744_073_709_551_616.0,
        1.0,
        1,
    )
    .expect_err("2^64 cannot be represented as a u64 packet count");
    assert!(error
        .to_string()
        .contains("required captured packet count exceeds u64"));
}

/// Strict-order reference for the finite-packet mass bound.
///
/// Every floating-point operation is wrapped in `std::hint::black_box`, which
/// is a semantic identity but an unconditional optimization barrier: each
/// intermediate is rounded to f64 and materialized before the next operation,
/// so `LLVM` cannot form an FMA across any `mul`/`add` pair regardless of
/// `-fp-contract` settings or `target-cpu`. This is therefore the exact
/// strict IEEE-754 evaluation the Python oracle
/// `finite_packet_release_mass_bound` performs. The production
/// `finite_packet_release_mass_bound_core` must match it bit-for-bit; on x86
/// (where contraction is enabled) that equality is the contraction-immunity
/// proof, and on arm64 it pins value-neutrality of the `black_box` insertion.
#[cfg(test)]
fn finite_packet_release_mass_bound_strict_reference(
    capture_probability: f64,
    target_probability: f64,
    deterministic_required_mass_kg: f64,
    grain_mass_kg: f64,
    grains_per_independent_packet: u64,
) -> anyhow::Result<FinitePacketMassBound> {
    use std::hint::black_box;
    if !capture_probability.is_finite() || !(0.0..=1.0).contains(&capture_probability) {
        return Err(anyhow::anyhow!("capture probability must lie in (0, 1]"));
    }
    if capture_probability == 0.0 {
        return Err(anyhow::anyhow!("capture probability must lie in (0, 1]"));
    }
    if !target_probability.is_finite() || !(0.0..1.0).contains(&target_probability) {
        return Err(anyhow::anyhow!("target probability must lie in (0, 1)"));
    }
    if !deterministic_required_mass_kg.is_finite() || deterministic_required_mass_kg <= 0.0 {
        return Err(anyhow::anyhow!(
            "deterministic required mass must be finite and > 0"
        ));
    }
    if !grain_mass_kg.is_finite() || grain_mass_kg <= 0.0 {
        return Err(anyhow::anyhow!("grain mass must be finite and > 0"));
    }
    if grains_per_independent_packet == 0 {
        return Err(anyhow::anyhow!(
            "grains per independent packet must be positive"
        ));
    }
    let packet_mass_kg = black_box(grain_mass_kg * u64_to_f64(grains_per_independent_packet));
    if !packet_mass_kg.is_finite() || packet_mass_kg <= 0.0 {
        return Err(anyhow::anyhow!("independent packet mass is invalid"));
    }
    let required_ratio = black_box(deterministic_required_mass_kg / packet_mass_kg);
    let required_captured_packets =
        checked_ceil_packet_count(required_ratio.max(1.0), "required captured")?;
    let log_failure_inverse = black_box(-(-target_probability).ln_1p());
    let root_term = black_box((2.0 * log_failure_inverse).sqrt());
    let root_term_squared = black_box(root_term * root_term);
    let required_linear_term = black_box(4.0 * u64_to_f64(required_captured_packets));
    let discriminant_sum = black_box(root_term_squared + required_linear_term);
    let discriminant_root = black_box(discriminant_sum.sqrt());
    let inner_sum = black_box(root_term + discriminant_root);
    let inner_square = black_box(inner_sum.powi(2));
    let expected_required = black_box(0.25 * inner_square);
    let released_expectation = black_box(expected_required / capture_probability);
    if !released_expectation.is_finite() || released_expectation <= 0.0 {
        return Err(anyhow::anyhow!("released packet count exceeds u64"));
    }
    let released_packet_count = checked_ceil_packet_count(
        next_up_positive(released_expectation).max(u64_to_f64(required_captured_packets)),
        "released",
    )?;
    let release_mass_kg = black_box(u64_to_f64(released_packet_count) * packet_mass_kg);
    if !release_mass_kg.is_finite() {
        return Err(anyhow::anyhow!("released mass exceeds finite f64"));
    }
    Ok(FinitePacketMassBound {
        release_mass_kg,
        released_packet_count,
        required_captured_packets,
        packet_mass_kg,
        expected_captured_packets: black_box(
            u64_to_f64(released_packet_count) * capture_probability,
        ),
    })
}

#[test]
fn finite_packet_mass_bound_is_contraction_immune_on_hostile_grid() {
    // Grid chosen to stress every last-bit-sensitive path: near-degenerate and
    // unit capture probabilities, targets crowding 1.0, ratios straddling the
    // ceil boundary (0.5, exactly 1.0, just above 1.0), and packet multiplicities
    // that scale the discriminant terms across many orders of magnitude.
    let capture_probabilities = [1e-9, 1e-6, 0.1, 0.25, 0.5, 0.9, 0.999_999, 1.0];
    let targets = [0.5, 0.9, 0.95, 0.99, 1.0 - 1e-12];
    let ratios = [0.5, 1.0, 1.000_000_000_1, 7.3, 1e6, 1e12];
    let grains_grid = [1u64, 10, 1000];
    let grain_mass_kg = 1.0;

    for &capture in &capture_probabilities {
        for &target in &targets {
            for &ratio in &ratios {
                for &grains in &grains_grid {
                    let packet_mass = grain_mass_kg * u64_to_f64(grains);
                    let required_mass = ratio * packet_mass;
                    let ctx =
                        format!("capture={capture} target={target} ratio={ratio} grains={grains}");
                    let patched = finite_packet_release_mass_bound_core(
                        capture,
                        target,
                        required_mass,
                        grain_mass_kg,
                        grains,
                    );
                    let reference = finite_packet_release_mass_bound_strict_reference(
                        capture,
                        target,
                        required_mass,
                        grain_mass_kg,
                        grains,
                    );
                    match (patched, reference) {
                        (Ok(p), Ok(r)) => {
                            assert_eq!(
                                p.release_mass_kg.to_bits(),
                                r.release_mass_kg.to_bits(),
                                "release_mass_kg bits differ ({ctx})"
                            );
                            assert_eq!(
                                p.released_packet_count, r.released_packet_count,
                                "released_packet_count differs ({ctx})"
                            );
                            assert_eq!(
                                p.required_captured_packets, r.required_captured_packets,
                                "required_captured_packets differs ({ctx})"
                            );
                            assert_eq!(
                                p.packet_mass_kg.to_bits(),
                                r.packet_mass_kg.to_bits(),
                                "packet_mass_kg bits differ ({ctx})"
                            );
                            assert_eq!(
                                p.expected_captured_packets.to_bits(),
                                r.expected_captured_packets.to_bits(),
                                "expected_captured_packets bits differ ({ctx})"
                            );
                        }
                        (Err(p), Err(r)) => {
                            assert_eq!(
                                p.to_string(),
                                r.to_string(),
                                "error messages differ ({ctx})"
                            );
                        }
                        (patched, reference) => {
                            let variants_match = std::mem::discriminant(&patched)
                                == std::mem::discriminant(&reference);
                            assert!(
                                variants_match,
                                "patched/reference Ok-vs-Err mismatch: \
                                 patched={patched:?} reference={reference:?} ({ctx})"
                            );
                        }
                    }
                }
            }
        }
    }
}
