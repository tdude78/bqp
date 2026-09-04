use anyhow::{anyhow, bail};
use nalgebra::{Cholesky, SMatrix, SVector, SymmetricEigen};

pub const DIM: usize = 6;

/// The sigma set this crate compiles, as a token.
///
/// The sealed Part A science authority carries the same string and each
/// consumer binds the two with a `const` assertion, so an edit on either side
/// is a BUILD failure rather than a silent reseal. See
/// `nd_config::part_a_science::PartANativeHybridControls`.
pub const SIGMA_SET_TOKEN: &str = "julier7-w0-zero";

/// [`DIM`] as an `f64`, for the simplex weight and offset algebra.
///
/// Written rather than cast: `usize as f64` is a silent lossy conversion this
/// workspace denies, and the assertion below makes a changed [`DIM`] a build
/// failure instead of a quietly wrong quadrature.
const DIM_F: f64 = 6.0;
const _: () = assert!(DIM == 6);

/// Julier's minimal-skew simplex at `W0 = 0`: `n + 1 = 7` points.
///
/// The `n + 2` recursion emits a centre point at index 0 carrying weight `W0`.
/// At `W0 = 0` that point reaches NEITHER moment, so it is not a sigma point at
/// all and is dropped — propagating it would be paying for an arc that
/// multiplies by literal `0.0`. What remains is `n + 1` off-centre points which
/// all land at whitened radius `sqrt(6)`, exactly the radius the retired Merwe
/// set used at the sealed `lambda = 0`.
///
/// Was `2 * DIM + 1 = 13` (Van der Merwe, scaled) until 2026-08-09. The trade
/// is recorded in `docs/plans/2026-08-05-hf-hybrid-speedup-audit.md` §15b/§15c:
/// a simplex is asymmetric and so is NOT third-degree exact (third-moment error
/// 2.041 against Merwe's exact 0), which costs a relative-Frobenius covariance
/// shift of ~1.1e-4 at the census mean span and ~2-3e-4 on the 18.2% of arcs
/// that fly 7,000-8,000 s. That shift is the accepted price of the point count.
pub const NUM_SIGMA: usize = DIM + 1;

const SIGMA_BLOCK_LEN: usize = NUM_SIGMA * DIM;
const COVARIANCE_LEN: usize = DIM * DIM;
const MIN_EIGENVALUE_FLOOR: f64 = 1e-12;

/// Reconstruction weights over a propagated sigma stack.
///
/// The simplex carries one weight per point in BOTH moments (`wm == wc`), so
/// unlike the retired Merwe set there is no centre point whose mean weight is
/// zero while its covariance weight is `beta`-inflated. The two arrays are kept
/// distinct because every consumer's reduction is written against a
/// mean/covariance weight pair.
/// Reconstruction produced a nonfinite or negative-variance row.
///
/// Distinct from the generic `1` the outputs are pre-filled with, so a row that
/// reached the finisher and was rejected there is distinguishable from one that
/// never reached it at all.
pub const FAILURE_CODE_NONFINITE_OUTPUT: i32 = 2;

pub struct SigmaWeights {
    pub wm: [f64; NUM_SIGMA],
    pub wc: [f64; NUM_SIGMA],
}

/// The simplex weight `w1 = (1 - W0) / (n + 1)` at the sealed `W0 = 0`.
///
/// Every retained point carries it, in both moments. `alpha`/`beta`/`kappa` do
/// not appear: they are the Merwe scaling triple, and the simplex has no
/// centre point for `beta` to inflate and no `lambda` to scale by. The sealed
/// authority dropped all three when this set landed.
#[must_use]
pub const fn julier_simplex_weights() -> SigmaWeights {
    let w1 = 1.0 / (DIM_F + 1.0);
    SigmaWeights {
        wm: [w1; NUM_SIGMA],
        wc: [w1; NUM_SIGMA],
    }
}

/// The simplex's whitened offsets `xi`, one row per retained point, so that
/// `x_i = mean + S * xi_i` for `S = chol(covar)`.
///
/// Hand-spelled `from_bits` literals rather than a computed table: the values
/// come from the `julier_whitened_offsets` sqrt recursion (test-only, kept as
/// the oracle), and `f64::sqrt` is not `const`, so the recursion cannot run at
/// compile time. The
/// `julier_offsets_table_matches_generator` pin test asserts every entry
/// bit-equal to the recursion's output, so an edit to either side that moves a
/// single bit is a test failure, not a silent reseal.
static JULIER_WHITENED_OFFSETS: [[f64; DIM]; NUM_SIGMA] = [
    [
        f64::from_bits(0xbffd_eeea_1168_3f49),
        f64::from_bits(0xbff1_482f_86c4_0c43),
        f64::from_bits(0xbfe8_70be_4c1c_28b2),
        f64::from_bits(0xbfe2_ee73_dadc_9b57),
        f64::from_bits(0xbfde_ea39_50a8_511e),
        f64::from_bits(0xbfda_20bd_700c_2c3f),
    ],
    [
        f64::from_bits(0x3ffd_eeea_1168_3f49),
        f64::from_bits(0xbff1_482f_86c4_0c43),
        f64::from_bits(0xbfe8_70be_4c1c_28b2),
        f64::from_bits(0xbfe2_ee73_dadc_9b57),
        f64::from_bits(0xbfde_ea39_50a8_511e),
        f64::from_bits(0xbfda_20bd_700c_2c3f),
    ],
    [
        0.0,
        f64::from_bits(0x4001_482f_86c4_0c43),
        f64::from_bits(0xbfe8_70be_4c1c_28b2),
        f64::from_bits(0xbfe2_ee73_dadc_9b57),
        f64::from_bits(0xbfde_ea39_50a8_511e),
        f64::from_bits(0xbfda_20bd_700c_2c3f),
    ],
    [
        0.0,
        0.0,
        f64::from_bits(0x4002_548e_b915_1e86),
        f64::from_bits(0xbfe2_ee73_dadc_9b57),
        f64::from_bits(0xbfde_ea39_50a8_511e),
        f64::from_bits(0xbfda_20bd_700c_2c3f),
    ],
    [
        0.0,
        0.0,
        0.0,
        f64::from_bits(0x4002_ee73_dadc_9b57),
        f64::from_bits(0xbfde_ea39_50a8_511e),
        f64::from_bits(0xbfda_20bd_700c_2c3f),
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        f64::from_bits(0x4003_5263_d269_32b3),
        f64::from_bits(0xbfda_20bd_700c_2c3f),
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        f64::from_bits(0x4003_988e_1409_212f),
    ],
];

/// The generator behind [`JULIER_WHITENED_OFFSETS`], kept as the pin oracle.
///
/// This is Julier's `n + 2` recursion (Julier 2003, *The spherical simplex
/// unscented transformation*) run at `W0 = 0` with the zero-weight centre point
/// dropped, in the recursion's own operation order — the same arithmetic the
/// decision instrument measured the accuracy trade with, so its numbers
/// transfer to this generator rather than merely describing it.
///
/// Allocation-free: the recursion's working set is `(DIM + 2) x DIM` and never
/// grows, so it is carried on the stack. Cheaper than the Cholesky it feeds.
#[cfg(test)]
#[must_use]
#[expect(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "fixed six-state simplex recursion over compile-time-sized arrays, in \
              the recursion's own operation order"
)]
fn julier_whitened_offsets() -> [[f64; DIM]; NUM_SIGMA] {
    let w1 = 1.0 / (DIM_F + 1.0);

    // j = 1: three scalars. Row 0 is the centre and stays all-zero throughout.
    let mut seq = [[0.0_f64; DIM]; DIM + 2];
    seq[1][0] = -1.0 / (2.0 * w1).sqrt();
    seq[2][0] = 1.0 / (2.0 * w1).sqrt();

    let mut jf = 1.0_f64;
    for j in 2..=DIM {
        jf += 1.0;
        let step = 1.0 / (jf * (jf + 1.0) * w1).sqrt();
        // Points 1..=j keep their prefix and extend by -step; the new apex at
        // index j + 1 is zero everywhere but its own axis. The centre at index 0
        // extends by a zero, which it already holds.
        for point in seq.iter_mut().take(j + 1).skip(1) {
            point[j - 1] = -step;
        }
        seq[j + 1] = [0.0; DIM];
        seq[j + 1][j - 1] = jf * step;
    }

    // Drop the centre: at W0 = 0 it carries no weight in either moment.
    let mut offsets = [[0.0_f64; DIM]; NUM_SIGMA];
    offsets.copy_from_slice(&seq[1..]);
    offsets
}

/// Repair a matrix toward PSD by flooring every eigenvalue below
/// `MIN_EIGENVALUE_FLOOR` (1e-12) — this clamps small POSITIVE eigenvalues
/// too, not just negative ones, so the result is strictly
/// positive-definite rather than merely PSD.
#[cold]
#[inline(never)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "nalgebra PSD reconstruction preserves established operation order"
)]
fn repair_to_psd(mat: &SMatrix<f64, DIM, DIM>) -> Option<SMatrix<f64, DIM, DIM>> {
    // Symmetric eigendecomposition
    let eigen = SymmetricEigen::new(*mat);
    let mut eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    // Floor every eigenvalue below MIN_EIGENVALUE_FLOOR (incl. small
    // positives, which also break the downstream Cholesky).
    let mut any_floored = false;
    for eigenvalue in eigenvalues.iter_mut() {
        if *eigenvalue < MIN_EIGENVALUE_FLOOR {
            *eigenvalue = MIN_EIGENVALUE_FLOOR;
            any_floored = true;
        }
    }

    if !any_floored {
        return None; // No eigenvalue below floor; Cholesky failed for another reason
    }

    // Reconstruct: V * diag(eigenvalues) * V^T
    let diag = SMatrix::<f64, DIM, DIM>::from_diagonal(&eigenvalues);
    Some(eigenvectors * diag * eigenvectors.transpose())
}

/// Generate the Julier minimal-skew simplex sigma points at `W0 = 0`.
///
/// `x_i = mean + S * xi_i`, for `S` the lower Cholesky factor of `covar` and
/// `xi_i` the whitened offsets of [`JULIER_WHITENED_OFFSETS`]. Every one of the
/// [`NUM_SIGMA`] rows is OFF-CENTRE: unlike the retired Merwe set there is no
/// row that reproduces the mean, so no row's arc duplicates a propagation the
/// caller already holds.
///
/// The retired Merwe generator scaled the covariance by `(n + lambda)` BEFORE
/// the Cholesky and used unit multiples of the factor's columns; the simplex
/// carries its `sqrt(6)` radius in the offsets instead and factors `covar`
/// as-is. One consequence is worth naming: `repair_to_psd`'s
/// `MIN_EIGENVALUE_FLOOR` now floors eigenvalues of the raw covariance rather
/// than of `6 * covar`, so the cold repair path is six times tighter than it
/// was. It fires only when the direct Cholesky fails.
#[must_use]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "fixed six-state sigma geometry, in nalgebra's established operation order"
)]
pub fn get_sigmas_ukf(
    mean: &SVector<f64, DIM>,
    covar: &SMatrix<f64, DIM, DIM>,
) -> Option<SMatrix<f64, NUM_SIGMA, DIM>> {
    // Compute Cholesky decomposition: covar = L * L^T. First try direct.
    let chol = if let Some(cholesky) = Cholesky::new(*covar) {
        cholesky
    } else {
        // Fallback: repair to nearest PSD and retry
        let repaired = repair_to_psd(covar)?;
        Cholesky::new(repaired)?
    };
    let l = chol.unpack();

    let mut sigmas = SMatrix::<f64, NUM_SIGMA, DIM>::zeros();
    for (row_index, offset) in JULIER_WHITENED_OFFSETS.iter().enumerate() {
        let xi = SVector::<f64, DIM>::from_column_slice(offset);
        sigmas
            .row_mut(row_index)
            .copy_from(&(mean + l * xi).transpose());
    }

    Some(sigmas)
}

#[must_use]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "preserve sequential UKF mean and covariance reduction order"
)]
pub fn rebuild_mean_covar_ukf(
    sigmas: &SMatrix<f64, NUM_SIGMA, DIM>,
    wm: &[f64; NUM_SIGMA],
    wc: &[f64; NUM_SIGMA],
) -> (SVector<f64, DIM>, SMatrix<f64, DIM, DIM>) {
    let mut mean = SVector::<f64, DIM>::zeros();
    for (i, weight) in wm.iter().enumerate().take(NUM_SIGMA) {
        mean += *weight * sigmas.row(i).transpose();
    }

    let mut covar = SMatrix::<f64, DIM, DIM>::zeros();
    for (i, weight) in wc.iter().enumerate().take(NUM_SIGMA) {
        let diff = sigmas.row(i).transpose() - mean;
        covar += *weight * (diff * diff.transpose());
    }

    // Ensure symmetry
    for row in 0..DIM {
        for column in 0..row {
            let (Some(&lower), Some(&upper)) = (covar.get((row, column)), covar.get((column, row)))
            else {
                continue;
            };
            let value = 0.5 * (lower + upper);
            if let Some(lower) = covar.get_mut((row, column)) {
                *lower = value;
            }
            if let Some(upper) = covar.get_mut((column, row)) {
                *upper = value;
            }
        }
    }

    (mean, covar)
}

#[inline]
fn sigmas_from_row_slice(row: &[f64]) -> SMatrix<f64, NUM_SIGMA, DIM> {
    debug_assert_eq!(row.len(), SIGMA_BLOCK_LEN);
    SMatrix::<f64, NUM_SIGMA, DIM>::from_row_slice(row)
}

/// Apply a linear position correction (`pos += vel * linear_dt`) to every
/// sigma point before reconstructing mean/covar. Used by the packed finisher
/// to align an MF-propagated sigma stack onto the row's HF reference time.
fn sigmas_from_row_slice_with_linear_dt(
    row: &[f64],
    linear_dt: f64,
) -> SMatrix<f64, NUM_SIGMA, DIM> {
    debug_assert_eq!(row.len(), SIGMA_BLOCK_LEN);
    if linear_dt == 0.0 {
        return sigmas_from_row_slice(row);
    }
    let mut adjusted = [0.0f64; SIGMA_BLOCK_LEN];
    adjusted.copy_from_slice(row);
    for sigma in adjusted.chunks_exact_mut(DIM) {
        let [x, y, z, vx, vy, vz] = sigma else {
            continue;
        };
        *x += *vx * linear_dt;
        *y += *vy * linear_dt;
        *z += *vz * linear_dt;
    }
    sigmas_from_row_slice(&adjusted)
}

fn fill_covar_row_major_slice(out: &mut [f64], covar: &SMatrix<f64, DIM, DIM>) {
    debug_assert_eq!(out.len(), COVARIANCE_LEN);
    for (output_row, covariance_row) in out.chunks_exact_mut(DIM).zip(covar.row_iter()) {
        for (output, value) in output_row.iter_mut().zip(covariance_row.iter()) {
            *output = *value;
        }
    }
}

/// Shared rebuild step for one component's `NUM_SIGMA * DIM` propagated
/// sigma block: optionally linear-dt-correct, then UKF-rebuild mean/covar.
/// `linear_dt == 0.0` is a no-op, so callers that never correct (the `_many`
/// finisher) and callers that do (the packed finisher) share one code path.
fn rebuild_component_mean_covar(
    sigma_block: &[f64],
    linear_dt: f64,
    mean_weights: &[f64; NUM_SIGMA],
    covariance_weights: &[f64; NUM_SIGMA],
) -> (SVector<f64, DIM>, SMatrix<f64, DIM, DIM>) {
    let sigmas_mat = sigmas_from_row_slice_with_linear_dt(sigma_block, linear_dt);
    rebuild_mean_covar_ukf(&sigmas_mat, mean_weights, covariance_weights)
}

/// Rebuild propagated MF sigma points into per-component means/covariances.
///
/// This path performs no row aggregation or linear-dt correction.
/// Byte-exact port of the oracle's `finish_dust_sigma_stack_many_into`
/// (`src/ukf/native/rust/dust_ukf_rs/src/lib.rs`), stripped of the pyo3
/// numpy boundary.
///
/// `propagated_sigmas` is `n * NUM_SIGMA * DIM` row-major (component-major,
/// then sigma-point-major, then state axis). Returns the component count `n`
/// on success.
///
/// # Errors
///
/// Returns an error when any input/output buffer shape or weight shape is invalid.
#[cfg(test)]
fn finish_dust_sigma_stack_many_into(
    propagated_sigmas: &[f64],
    mean_weights: &[f64; NUM_SIGMA],
    covariance_weights: &[f64; NUM_SIGMA],
    means_out: &mut [f64],
    covs_out: &mut [f64],
) -> anyhow::Result<usize> {
    if propagated_sigmas.len().checked_rem(SIGMA_BLOCK_LEN) != Some(0) {
        bail!(
            "propagated_sigmas length must be divisible by {SIGMA_BLOCK_LEN} (= {NUM_SIGMA} * {DIM})"
        );
    }
    let n = propagated_sigmas.chunks_exact(SIGMA_BLOCK_LEN).len();
    let expected_means = n
        .checked_mul(DIM)
        .ok_or_else(|| anyhow!("means_out expected length overflow"))?;
    if means_out.len() != expected_means {
        bail!("means_out must have length {expected_means}");
    }
    let expected_covariances = n
        .checked_mul(COVARIANCE_LEN)
        .ok_or_else(|| anyhow!("covs_out expected length overflow"))?;
    if covs_out.len() != expected_covariances {
        bail!("covs_out must have length {expected_covariances}");
    }

    for ((sigma_block, mean_output), covariance_output) in propagated_sigmas
        .chunks_exact(SIGMA_BLOCK_LEN)
        .zip(means_out.chunks_exact_mut(DIM))
        .zip(covs_out.chunks_exact_mut(COVARIANCE_LEN))
    {
        let (mean, covariance) =
            rebuild_component_mean_covar(sigma_block, 0.0, mean_weights, covariance_weights);
        mean_output.copy_from_slice(mean.as_slice());
        fill_covar_row_major_slice(covariance_output, &covariance);
    }

    Ok(n)
}

/// Rebuild and aggregate propagated MF sigma points.
///
/// This linear-dt-corrects each component onto its row's HF reference time,
/// then aggregates components into a weighted HF velocity mean and scalar
/// position variance. Byte-exact port
/// of the oracle's `finish_dust_sigma_stack_packed_into`
/// (`src/ukf/native/rust/dust_ukf_rs/src/lib.rs`), stripped of the pyo3
/// numpy boundary.
///
/// Note on partial-write-on-error: unlike the oracle (which stages results
/// in owned buffers and only copies into the caller's numpy arrays once the
/// whole computation succeeds), this slice-based port writes directly into
/// `*_out` as it goes. On `Err`, the `*_out` slices may hold a partial
/// result rather than being left untouched — callers must treat any `Err`
/// as "discard this cell's outputs", not "outputs are unchanged".
///
/// # Errors
///
/// Returns an error for malformed shapes, invalid offsets or weights, non-finite
/// correction times, or count overflow.
#[derive(Clone, Copy)]
pub struct PackedSigmaStackInput<'a> {
    pub propagated_sigmas: &'a [f64],
    pub mean_weights: &'a [f64; NUM_SIGMA],
    pub covariance_weights: &'a [f64; NUM_SIGMA],
    pub row_offsets: &'a [i64],
    pub component_weights: &'a [f64],
    pub component_linear_dt_s: &'a [f64],
}

pub struct PackedSigmaStackOutput<'a> {
    pub means: &'a mut [f64],
    pub covariances: &'a mut [f64],
    pub component_weights: &'a mut [f64],
    pub hf_velocity_means: &'a mut [f64],
    pub position_variances: &'a mut [f64],
    pub valid: &'a mut [bool],
    pub failure_codes: &'a mut [i32],
}

/// Rebuild and aggregate propagated MF sigma points into caller-owned outputs.
///
/// # Errors
///
/// Returns an error for malformed shapes, invalid offsets or weights, non-finite
/// correction times, or count overflow.
pub fn finish_dust_sigma_stack_packed_into(
    input: PackedSigmaStackInput<'_>,
    output: PackedSigmaStackOutput<'_>,
) -> anyhow::Result<usize> {
    let PackedSigmaStackInput {
        propagated_sigmas,
        mean_weights,
        covariance_weights,
        row_offsets,
        component_weights: weights,
        component_linear_dt_s,
    } = input;
    let PackedSigmaStackOutput {
        means: means_out,
        covariances: covs_out,
        component_weights: weights_out,
        hf_velocity_means: v_hf_mean_out,
        position_variances: pos_var_out,
        valid: valid_out,
        failure_codes: failure_code_out,
    } = output;
    if propagated_sigmas.len().checked_rem(SIGMA_BLOCK_LEN) != Some(0) {
        bail!(
            "propagated_sigmas length must be divisible by {SIGMA_BLOCK_LEN} (= {NUM_SIGMA} * {DIM})"
        );
    }
    let n_components = propagated_sigmas.chunks_exact(SIGMA_BLOCK_LEN).len();
    if weights.len() != n_components {
        bail!("weights must have length {n_components}");
    }
    if component_linear_dt_s.len() != n_components {
        bail!("component_linear_dt_s must have length {n_components}");
    }
    let expected_means = n_components
        .checked_mul(DIM)
        .ok_or_else(|| anyhow!("means_out expected length overflow"))?;
    if means_out.len() != expected_means {
        bail!("means_out must have length {expected_means}");
    }
    let expected_covariances = n_components
        .checked_mul(COVARIANCE_LEN)
        .ok_or_else(|| anyhow!("covs_out expected length overflow"))?;
    if covs_out.len() != expected_covariances {
        bail!("covs_out must have length {expected_covariances}");
    }
    if weights_out.len() != n_components {
        bail!("weights_out must have length {n_components}");
    }
    if row_offsets.is_empty() {
        bail!("row_offsets must have at least one element");
    }
    let n_rows = row_offsets
        .len()
        .checked_sub(1)
        .ok_or_else(|| anyhow!("row_offsets must have at least one element"))?;
    let expected_velocities = n_rows
        .checked_mul(3)
        .ok_or_else(|| anyhow!("v_hf_mean_out expected length overflow"))?;
    if v_hf_mean_out.len() != expected_velocities {
        bail!("v_hf_mean_out must have length {expected_velocities}");
    }
    for (name, len) in [
        ("pos_var_out", pos_var_out.len()),
        ("valid_out", valid_out.len()),
        ("failure_code_out", failure_code_out.len()),
    ] {
        if len != n_rows {
            bail!("{name} must have length {n_rows}");
        }
    }

    means_out.fill(0.0);
    covs_out.fill(0.0);
    weights_out.copy_from_slice(weights);
    v_hf_mean_out.fill(f64::NAN);
    pos_var_out.fill(f64::NAN);
    valid_out.fill(false);
    failure_code_out.fill(1);

    for (((sigma_block, &linear_dt), mean_output), covariance_output) in propagated_sigmas
        .chunks_exact(SIGMA_BLOCK_LEN)
        .zip(component_linear_dt_s)
        .zip(means_out.chunks_exact_mut(DIM))
        .zip(covs_out.chunks_exact_mut(COVARIANCE_LEN))
    {
        if !linear_dt.is_finite() {
            bail!("component_linear_dt_s contains non-finite values");
        }
        let (mean, covariance) =
            rebuild_component_mean_covar(sigma_block, linear_dt, mean_weights, covariance_weights);
        mean_output.copy_from_slice(mean.as_slice());
        fill_covar_row_major_slice(covariance_output, &covariance);
    }

    let mut valid_count = 0usize;
    let row_outputs = v_hf_mean_out
        .chunks_exact_mut(3)
        .zip(pos_var_out.iter_mut())
        .zip(valid_out.iter_mut())
        .zip(failure_code_out.iter_mut());
    for (offset_pair, (((velocity_output, position_variance_output), valid), failure_code)) in
        row_offsets.windows(2).zip(row_outputs)
    {
        let &[lo_raw, hi_raw] = offset_pair else {
            bail!("row offset window has invalid width");
        };
        if lo_raw < 0 || hi_raw < 0 {
            bail!("row_offsets must be non-negative");
        }
        if hi_raw < lo_raw {
            bail!("row_offsets must be monotonic");
        }
        let lo = usize::try_from(lo_raw).map_err(|_| anyhow!("row offset is unrepresentable"))?;
        let hi = usize::try_from(hi_raw).map_err(|_| anyhow!("row offset is unrepresentable"))?;
        if hi > n_components {
            bail!(
                "row_offsets references component {hi}, but only {n_components} components exist"
            );
        }
        if hi == lo {
            continue;
        }

        let row_weights = weights
            .get(lo..hi)
            .ok_or_else(|| anyhow!("row offsets lie outside component weights"))?;
        let total_weight: f64 = row_weights.iter().copied().sum();
        if !total_weight.is_finite() || total_weight <= 0.0 {
            bail!("row weights must sum to a positive finite value");
        }
        let inv_weight = 1.0 / total_weight;
        let mut velocity_mean = [0.0f64; 3];
        let mut position_centroid = [0.0f64; 3];
        let component_count = hi
            .checked_sub(lo)
            .ok_or_else(|| anyhow!("row component count underflow"))?;
        for (&component_weight, component_mean) in row_weights
            .iter()
            .zip(means_out.chunks_exact(DIM).skip(lo).take(component_count))
        {
            let w = component_weight * inv_weight;
            if !w.is_finite() || w < 0.0 {
                bail!("row weights must be finite and non-negative");
            }
            let &[px, py, pz, vx, vy, vz] = component_mean else {
                bail!("component mean row has invalid width");
            };
            let [centroid_x, centroid_y, centroid_z] = &mut position_centroid;
            *centroid_x += w * px;
            *centroid_y += w * py;
            *centroid_z += w * pz;
            let [velocity_x, velocity_y, velocity_z] = &mut velocity_mean;
            *velocity_x += w * vx;
            *velocity_y += w * vy;
            *velocity_z += w * vz;
        }

        let mut pos_var = 0.0f64;
        let means = means_out.chunks_exact(DIM).skip(lo).take(component_count);
        let covariances = covs_out
            .chunks_exact(COVARIANCE_LEN)
            .skip(lo)
            .take(component_count);
        for ((&component_weight, component_mean), component_covariance) in
            row_weights.iter().zip(means).zip(covariances)
        {
            let w = component_weight * inv_weight;
            let &[px, py, pz, ..] = component_mean else {
                bail!("component mean row has invalid width");
            };
            let [centroid_x, centroid_y, centroid_z] = position_centroid;
            let mut diff_norm_sq = 0.0f64;
            for (position, centroid) in [(px, centroid_x), (py, centroid_y), (pz, centroid_z)] {
                let diff = position - centroid;
                diff_norm_sq += diff * diff;
            }
            let &[c00, _, _, _, _, _, _, c11, _, _, _, _, _, _, c22, ..] = component_covariance
            else {
                bail!("component covariance row has invalid width");
            };
            let trace = c00 + c11 + c22;
            pos_var += w * ((trace + diff_norm_sq) / 3.0);
        }

        // FINITE SCAN BEFORE PUBLISHING SUCCESS. The finisher used to write
        // `valid = true` and `failure_code = 0` unconditionally, having scanned
        // nothing: finite inputs can still overflow in the squared deviations
        // above, and a NaN or Inf propagated sigma flows through reconstruction
        // untouched. A row that is not finite is not a valid row, whatever the
        // arithmetic that produced it.
        //
        // `pos_var` must also be non-negative: it is a variance, and a negative
        // value means the weighted trace/deviation aggregation lost its meaning
        // rather than merely rounding.
        if !velocity_mean.iter().all(|value| value.is_finite())
            || !pos_var.is_finite()
            || pos_var < 0.0
        {
            *valid = false;
            *failure_code = FAILURE_CODE_NONFINITE_OUTPUT;
            continue;
        }
        velocity_output.copy_from_slice(&velocity_mean);
        *position_variance_output = pos_var;
        *valid = true;
        *failure_code = 0;
        valid_count = valid_count
            .checked_add(1)
            .ok_or_else(|| anyhow!("valid row count overflow"))?;
    }

    Ok(valid_count)
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    let &[ax, ay, az] = a;
    let &[bx, by, bz] = b;
    [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]
}

fn norm3(a: &[f64; 3]) -> f64 {
    let &[x, y, z] = a;
    (x * x + y * y + z * z).sqrt()
}

fn normalize3(a: [f64; 3], norm: f64) -> [f64; 3] {
    let [x, y, z] = a;
    [x / norm, y / norm, z / norm]
}

fn fill_axis_covariance(
    out: &mut [f64],
    axis_offset: usize,
    axis_r: [f64; 3],
    axis_i: [f64; 3],
    axis_c: [f64; 3],
    radial_variance: f64,
    in_track_variance: f64,
    cross_track_variance: f64,
) {
    for (((output_row, radial_row), in_track_row), cross_track_row) in out
        .chunks_exact_mut(DIM)
        .skip(axis_offset)
        .take(3)
        .zip(axis_r)
        .zip(axis_i)
        .zip(axis_c)
    {
        for (((output, radial_column), in_track_column), cross_track_column) in output_row
            .iter_mut()
            .skip(axis_offset)
            .take(3)
            .zip(axis_r)
            .zip(axis_i)
            .zip(axis_c)
        {
            *output = radial_variance * radial_row * radial_column
                + in_track_variance * in_track_row * in_track_column
                + cross_track_variance * cross_track_row * cross_track_column;
        }
    }
}

/// Build release covariance matrices from RIC-frame scalar sigmas.
///
/// # Errors
///
/// Returns an error for malformed input/output buffer shapes or count overflow.
pub fn dust_release_covariances_from_ric_sigmas(
    release_states_eci: &[f64],
    n_rows: usize,
    pos_sigma_in_track_m: &[f64],
    pos_sigma_radial_cross_track_m: &[f64],
    vel_sigma_in_track_mps: &[f64],
    vel_sigma_radial_cross_track_mps: &[f64],
    covs_out: &mut [f64],
    valid_out: &mut [bool],
) -> anyhow::Result<usize> {
    let expected_states = n_rows
        .checked_mul(DIM)
        .ok_or_else(|| anyhow!("release state expected length overflow"))?;
    if release_states_eci.len() != expected_states {
        bail!(
            "release_states_eci length mismatch: expected {expected_states}, got {}",
            release_states_eci.len()
        );
    }
    for (name, actual) in [
        ("pos_sigma_in_track_m", pos_sigma_in_track_m.len()),
        (
            "pos_sigma_radial_cross_track_m",
            pos_sigma_radial_cross_track_m.len(),
        ),
        ("vel_sigma_in_track_mps", vel_sigma_in_track_mps.len()),
        (
            "vel_sigma_radial_cross_track_mps",
            vel_sigma_radial_cross_track_mps.len(),
        ),
        ("valid_out", valid_out.len()),
    ] {
        if actual != n_rows {
            bail!("{name} length mismatch: expected {n_rows}, got {actual}");
        }
    }
    let expected_covariances = n_rows
        .checked_mul(COVARIANCE_LEN)
        .ok_or_else(|| anyhow!("covariance expected length overflow"))?;
    if covs_out.len() != expected_covariances {
        bail!(
            "covs_out length mismatch: expected {expected_covariances}, got {}",
            covs_out.len()
        );
    }

    let mut valid_count = 0usize;
    let rows = release_states_eci
        .chunks_exact(DIM)
        .zip(covs_out.chunks_exact_mut(COVARIANCE_LEN))
        .zip(valid_out.iter_mut())
        .zip(pos_sigma_in_track_m)
        .zip(pos_sigma_radial_cross_track_m)
        .zip(vel_sigma_in_track_mps)
        .zip(vel_sigma_radial_cross_track_mps);
    for ((((((state, covariance), valid), &pos_i_m), &pos_rc_m), &vel_i_mps), &vel_rc_mps) in rows {
        covariance.fill(0.0);
        *valid = false;

        if !state.iter().all(|value| value.is_finite()) {
            continue;
        }
        let &[rx, ry, rz, vx, vy, vz] = state else {
            continue;
        };
        let radius = [rx, ry, rz];
        let velocity = [vx, vy, vz];
        let r_norm = norm3(&radius);
        let h_vec = cross3(&radius, &velocity);
        let h_norm = norm3(&h_vec);
        if r_norm < 1e-12 || h_norm < 1e-12 {
            continue;
        }
        let r_hat = normalize3(radius, r_norm);
        let h_hat = normalize3(h_vec, h_norm);
        let i_vec = cross3(&h_hat, &r_hat);
        let i_norm = norm3(&i_vec);
        if i_norm < 1e-12 {
            continue;
        }
        let i_hat = normalize3(i_vec, i_norm);

        let pos_i = pos_i_m / 1000.0;
        let pos_rc = pos_rc_m / 1000.0;
        let vel_i = vel_i_mps / 1000.0;
        let vel_rc = vel_rc_mps / 1000.0;
        if ![pos_i, pos_rc, vel_i, vel_rc]
            .iter()
            .all(|value| value.is_finite())
        {
            continue;
        }

        fill_axis_covariance(
            covariance,
            0,
            r_hat,
            i_hat,
            h_hat,
            pos_rc * pos_rc,
            pos_i * pos_i,
            pos_rc * pos_rc,
        );
        fill_axis_covariance(
            covariance,
            3,
            r_hat,
            i_hat,
            h_hat,
            vel_rc * vel_rc,
            vel_i * vel_i,
            vel_rc * vel_rc,
        );
        *valid = true;
        valid_count = valid_count
            .checked_add(1)
            .ok_or_else(|| anyhow!("valid release covariance count overflow"))?;
    }
    Ok(valid_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mean(offset: f64) -> SVector<f64, DIM> {
        SVector::<f64, DIM>::from_row_slice(&[
            7000.0 + offset,
            0.1 * offset,
            -0.2 * offset,
            0.01 * offset,
            7.5 + 0.001 * offset,
            -0.003 * offset,
        ])
    }

    fn sample_covar(scale: f64) -> SMatrix<f64, DIM, DIM> {
        let diag = SVector::<f64, DIM>::from_row_slice(&[
            1.0 + scale,
            1.2 + scale,
            1.4 + scale,
            1.0e-3 + scale * 1.0e-4,
            1.2e-3 + scale * 1.0e-4,
            1.4e-3 + scale * 1.0e-4,
        ]);
        SMatrix::<f64, DIM, DIM>::from_diagonal(&diag)
    }

    /// Exact-output policy: signed zero and every NaN bit (sign plus payload)
    /// are significant. Do not weaken these checks to floating-point equality.
    fn assert_f64_bits_eq(actual: f64, expected: f64, context: &str) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{context}: expected {:016x}, got {:016x}",
            expected.to_bits(),
            actual.to_bits()
        );
    }

    fn assert_f64_slice_bits_eq(actual: &[f64], expected: &[f64], context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context}: slice length");
        for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
            assert_f64_bits_eq(actual_value, expected_value, &format!("{context}[{index}]"));
        }
    }

    fn first_three_diagonal_trace(covariance: &SMatrix<f64, DIM, DIM>) -> f64 {
        let mut rows = covariance.as_slice().chunks_exact(DIM);
        let first_row = rows.next().expect("first covariance row");
        let second_row = rows.next().expect("second covariance row");
        let third_row = rows.next().expect("third covariance row");
        let first_diagonal = *first_row.first().expect("first diagonal");
        let second_diagonal = *second_row.get(1).expect("second diagonal");
        let third_diagonal = *third_row.get(2).expect("third diagonal");
        first_diagonal + second_diagonal + third_diagonal
    }

    #[test]
    fn exact_float_assertion_policy_distinguishes_zero_sign_and_nan_payload() {
        let positive_zero = 0.0f64;
        let negative_zero = -0.0f64;
        let nan_a = f64::from_bits(0x7ff8_0000_0000_0042);
        let nan_b = f64::from_bits(0x7ff8_0000_0000_0043);

        assert_ne!(positive_zero.to_bits(), negative_zero.to_bits());
        assert!(nan_a.is_nan());
        assert!(nan_b.is_nan());
        assert_ne!(nan_a.to_bits(), nan_b.to_bits());
        assert_f64_slice_bits_eq(
            &[negative_zero, nan_a],
            &[-0.0, f64::from_bits(0x7ff8_0000_0000_0042)],
            "exact float policy",
        );
    }

    #[test]
    fn packed_finisher_keeps_legacy_reduction_evaluation_order() {
        let source = include_str!("lib.rs");
        let production_source = source
            .rsplit_once("\n#[cfg(test)]\nmod tests")
            .expect("production source precedes test module")
            .0;
        let compact_production_source: String = production_source.split_whitespace().collect();

        assert!(
            compact_production_source.contains("letmutdiff_norm_sq=0.0f64;for(position,centroid)in[(px,centroid_x),(py,centroid_y),(pz,centroid_z)]{letdiff=position-centroid;diff_norm_sq+=diff*diff;}"),
            "diff norm must retain legacy zero-seeded axis reduction"
        );
        assert!(
            compact_production_source.contains("lettrace=c00+c11+c22;"),
            "covariance trace must retain legacy left-associated a + b + c evaluation"
        );
        assert!(
            compact_production_source.contains("pubmean_weights:&'a[f64;NUM_SIGMA],"),
            "packed mean weights must be fixed to the six-state sigma count"
        );
        assert!(
            compact_production_source.contains("pubcovariance_weights:&'a[f64;NUM_SIGMA],"),
            "packed covariance weights must be fixed to the six-state sigma count"
        );
        assert!(
            !production_source.contains("fn weights_arrays("),
            "packed finisher must not retain a prefix-copy weight shim"
        );
        assert!(
            compact_production_source
                .contains("pubstructSigmaWeights{pubwm:[f64;NUM_SIGMA],pubwc:[f64;NUM_SIGMA],}"),
            "sigma weights must be fixed to the six-state sigma count"
        );
    }

    /// The simplex's quadrature degree, checked where it is free to check.
    ///
    /// A sigma set's moment error against the standard six-variate Gaussian is
    /// pure quadrature over the whitened offsets — no propagation, no force
    /// model, no tolerance. That makes this the cheapest possible statement of
    /// what the julier7 set does and does not buy, and it is the statement the
    /// landing decision rests on:
    ///
    /// * `m1 = 0` and `m2 = I` EXACTLY — the set reproduces the mean and the
    ///   full covariance, so it is second-degree exact and the covariance it
    ///   propagates is the caller's own, not an approximation of it.
    /// * `m3 != 0` — a simplex is asymmetric, so the `+L.col(i)`/`-L.col(i)`
    ///   cancellation that made the retired Merwe set third-degree exact is
    ///   gone and CANNOT be recovered by any choice of `W0`. The measured worst
    ///   axis error is 2.041. This is the accepted cost, asserted rather than
    ///   described so that a future edit claiming to restore exactness has to
    ///   come here and say so.
    /// * every point at whitened radius `sqrt(6)` — the retired set's radius at
    ///   the sealed `lambda = 0`, so the set did not move further into the tails
    ///   to buy its point count.
    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "fixed six-state moment audit over compile-time-sized arrays; an \
                  out-of-range axis here is a test bug and panicking says so"
    )]
    fn julier_simplex_is_second_degree_exact_and_not_third() {
        let offsets = julier_whitened_offsets();
        let w1 = julier_simplex_weights();

        // Literals, not `DIM + 1`. `NUM_SIGMA` is DEFINED as `DIM + 1` and
        // `offsets` is `[[f64; DIM]; NUM_SIGMA]`, so both of the obvious
        // spellings — `NUM_SIGMA == DIM + 1` and `offsets.len() == NUM_SIGMA` —
        // are true by construction at every dimension and pin nothing. The
        // julier7 lane is sealed at seven points over six states; that is the
        // property, and it must go red if either moves.
        assert_eq!(DIM, 6, "the sealed sigma lane carries six states");
        assert_eq!(NUM_SIGMA, 7, "the julier simplex ships seven sigma points");
        assert_eq!(offsets.len(), 7);

        // No point is the centre. This is what deletes the sigma-0 duplicate
        // arc: there is no row whose propagation the caller already holds.
        for (index, offset) in offsets.iter().enumerate() {
            let radius = offset.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (radius - 6.0_f64.sqrt()).abs() < 1e-15,
                "point {index} sits at whitened radius {radius}, expected sqrt(6)"
            );
        }

        let moment = |order: i32| -> [f64; DIM] {
            let mut out = [0.0; DIM];
            for (axis, value) in out.iter_mut().enumerate() {
                *value = offsets
                    .iter()
                    .zip(w1.wm.iter())
                    .map(|(offset, weight)| weight * offset[axis].powi(order))
                    .sum();
            }
            out
        };

        for (axis, value) in moment(1).iter().enumerate() {
            assert!(value.abs() < 1e-15, "m1 axis {axis} is {value}, expected 0");
        }
        for (axis, value) in moment(2).iter().enumerate() {
            assert!(
                (value - 1.0).abs() < 1e-15,
                "m2 axis {axis} is {value}, expected 1"
            );
        }
        // Off-diagonal second moments must vanish too, or the set would not
        // reproduce a general covariance under the Cholesky map.
        for a in 0..DIM {
            for b in (a + 1)..DIM {
                let cross: f64 = offsets
                    .iter()
                    .zip(w1.wm.iter())
                    .map(|(offset, weight)| weight * offset[a] * offset[b])
                    .sum();
                assert!(cross.abs() < 1e-15, "m2 cross ({a},{b}) is {cross}");
            }
        }

        let worst_m3 = moment(3).iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(
            (worst_m3 - 2.041_241_452_319_315).abs() < 1e-12,
            "third-moment error is {worst_m3}; the ledgered value is 2.041241452319315. \
             A change here is a change in the set's quadrature degree, not a rounding \
             difference, and it invalidates the accuracy trade recorded on NUM_SIGMA"
        );
    }

    /// The generator maps the whitened offsets through the caller's own
    /// covariance, and the round trip is exact.
    ///
    /// Non-vacuity is the point: a generator that ignored `covar` entirely, or
    /// one whose offsets did not satisfy the moment identities above, would
    /// still produce seven finite rows. Rebuilding the mean and covariance from
    /// the UNPROPAGATED points must return the inputs, and the poison arm proves
    /// the comparison can fail — a 1e-3 relative perturbation of one input
    /// variance has to show up in the rebuild.
    #[test]
    fn unpropagated_simplex_rebuilds_its_own_mean_and_covariance() {
        let weights = julier_simplex_weights();
        let mean = sample_mean(3.0);
        let covar = sample_covar(0.75);

        let sigmas = get_sigmas_ukf(&mean, &covar).expect("sigma generation should succeed");
        let (rebuilt_mean, rebuilt_covar) =
            rebuild_mean_covar_ukf(&sigmas, &weights.wm, &weights.wc);

        let mean_scale = mean.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        for (axis, (got, want)) in rebuilt_mean.iter().zip(mean.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-12 * mean_scale,
                "mean axis {axis} rebuilt as {got}, input {want}"
            );
        }
        let covar_scale = covar.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let worst = rebuilt_covar
            .iter()
            .zip(covar.iter())
            .fold(0.0_f64, |acc, (got, want)| acc.max((got - want).abs()));
        assert!(
            worst <= 1e-12 * covar_scale,
            "covariance round trip is off by {worst}, scale {covar_scale}"
        );

        let mut poisoned = covar;
        let [first, ..] = poisoned.as_mut_slice() else {
            panic!("covariance has a first element");
        };
        *first *= 1.001;
        let poisoned_sigmas =
            get_sigmas_ukf(&mean, &poisoned).expect("poisoned sigma generation should succeed");
        let (_, poisoned_covar) =
            rebuild_mean_covar_ukf(&poisoned_sigmas, &weights.wm, &weights.wc);
        let poison_move = poisoned_covar
            .iter()
            .zip(covar.iter())
            .fold(0.0_f64, |acc, (got, want)| acc.max((got - want).abs()));
        assert!(
            poison_move > 100.0 * worst.max(f64::MIN_POSITIVE),
            "a 1e-3 covariance poison moved the rebuild by {poison_move}, which does not \
             dominate the {worst} round-trip residual; this comparison proves nothing"
        );
    }

    #[test]
    fn release_covariance_ric_axes_write_direct_6x6_mapping() {
        let release_state_eci = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let mut covariances = vec![f64::NAN; COVARIANCE_LEN];
        let mut valid = [false];

        let valid_count = dust_release_covariances_from_ric_sigmas(
            &release_state_eci,
            1,
            &[2000.0],
            &[3000.0],
            &[4000.0],
            &[5000.0],
            &mut covariances,
            &mut valid,
        )
        .expect("axis-aligned RIC covariance should succeed");

        let expected = [
            9.0, 0.0, 0.0, 0.0, 0.0, 0.0, // position radial
            0.0, 4.0, 0.0, 0.0, 0.0, 0.0, // position in-track
            0.0, 0.0, 9.0, 0.0, 0.0, 0.0, // position cross-track
            0.0, 0.0, 0.0, 25.0, 0.0, 0.0, // velocity radial
            0.0, 0.0, 0.0, 0.0, 16.0, 0.0, // velocity in-track
            0.0, 0.0, 0.0, 0.0, 0.0, 25.0, // velocity cross-track
        ];

        assert_eq!(valid_count, 1);
        assert_eq!(valid, [true]);
        assert_f64_slice_bits_eq(&covariances, &expected, "RIC 6x6 mapping");
    }

    fn flatten_sigmas(sigmas: &SMatrix<f64, NUM_SIGMA, DIM>) -> Vec<f64> {
        let mut out = Vec::with_capacity(SIGMA_BLOCK_LEN);
        for row in sigmas.row_iter() {
            out.extend(row.iter().copied());
        }
        out
    }

    fn flatten_covar(covar: &SMatrix<f64, DIM, DIM>) -> Vec<f64> {
        let mut out = Vec::with_capacity(COVARIANCE_LEN);
        for row in covar.row_iter() {
            out.extend(row.iter().copied());
        }
        out
    }

    /// Two-component, one-row case with hand-checked row aggregation.
    /// Per-component mean/covar reconstruction is cross-checked bit-exact
    /// against a direct `get_sigmas_ukf` + `rebuild_mean_covar_ukf` call
    /// (the same primitives the ported finisher reuses internally); the
    /// row-aggregation math (`v_hf_mean`/`pos_var`/weights passthrough) is
    /// hand-derived independently with power-of-two weights so every
    /// intermediate division/multiplication is exact in f64, making the
    /// bit-exact comparison meaningful rather than a tolerance fudge.
    /// A nonfinite propagated sigma must not be published as a valid row.
    ///
    /// The finisher used to write `valid = true` and `failure_code = 0` having
    /// scanned nothing, so a NaN that entered through reconstruction left as a
    /// successful result. The consumer treats `valid` as authority, which is
    /// what made this publishable rather than merely wrong.
    #[test]
    fn packed_finisher_refuses_to_publish_a_nonfinite_row() {
        let sigma_weights = julier_simplex_weights();
        let mean = sample_mean(0.0);
        let covar = sample_covar(0.0);
        let sigmas =
            get_sigmas_ukf(&mean, &covar).expect("component sigma generation should succeed");

        let mut propagated_sigmas = flatten_sigmas(&sigmas);
        // Poison one propagated sigma component, exactly as a diverged
        // propagation would.
        *propagated_sigmas
            .get_mut(3)
            .expect("sigma block must have a velocity component") = f64::NAN;

        let weights = [1.0f64];
        let component_linear_dt_s = [0.0f64];
        let row_offsets = [0i64, 1i64];
        let mut means_out = vec![0.0f64; DIM];
        let mut covs_out = vec![0.0f64; DIM * DIM];
        let mut weights_out = vec![0.0f64; 1];
        let mut v_hf_mean_out = vec![0.0f64; 3];
        let mut pos_var_out = vec![0.0f64; 1];
        let mut valid_out = vec![true; 1];
        let mut failure_code_out = vec![0i32; 1];

        let valid_count = finish_dust_sigma_stack_packed_into(
            PackedSigmaStackInput {
                propagated_sigmas: &propagated_sigmas,
                mean_weights: &sigma_weights.wm,
                covariance_weights: &sigma_weights.wc,
                row_offsets: &row_offsets,
                component_weights: &weights,
                component_linear_dt_s: &component_linear_dt_s,
            },
            PackedSigmaStackOutput {
                means: &mut means_out,
                covariances: &mut covs_out,
                component_weights: &mut weights_out,
                hf_velocity_means: &mut v_hf_mean_out,
                position_variances: &mut pos_var_out,
                valid: &mut valid_out,
                failure_codes: &mut failure_code_out,
            },
        )
        .expect("the finisher must reject the row, not error out");

        assert_eq!(valid_count, 0, "a nonfinite row was counted as valid");
        assert_eq!(valid_out, [false], "a nonfinite row was published as valid");
        assert_eq!(
            failure_code_out,
            [FAILURE_CODE_NONFINITE_OUTPUT],
            "the rejection must be distinguishable from a row that never reached \
             the finisher"
        );
    }

    #[test]
    fn packed_finisher_matches_hand_checked_row_aggregation() {
        let sigma_weights = julier_simplex_weights();

        let mean_a = sample_mean(0.0);
        let covar_a = sample_covar(0.0);
        let mean_b = sample_mean(5.0);
        let covar_b = sample_covar(1.0);

        let sigmas_a =
            get_sigmas_ukf(&mean_a, &covar_a).expect("component A sigma generation should succeed");
        let sigmas_b =
            get_sigmas_ukf(&mean_b, &covar_b).expect("component B sigma generation should succeed");

        let (expected_mean_a, expected_covar_a) =
            rebuild_mean_covar_ukf(&sigmas_a, &sigma_weights.wm, &sigma_weights.wc);
        let (expected_mean_b, expected_covar_b) =
            rebuild_mean_covar_ukf(&sigmas_b, &sigma_weights.wm, &sigma_weights.wc);

        let mut propagated_sigmas = flatten_sigmas(&sigmas_a);
        propagated_sigmas.extend(flatten_sigmas(&sigmas_b));

        // Power-of-two weights: 3.0 / 4.0 and 1.0 / 4.0 are exact in f64,
        // so the aggregation arithmetic below is bit-reproducible.
        let weights = [3.0f64, 1.0f64];
        let component_linear_dt_s = [0.0f64, 0.0f64];
        let row_offsets = [0i64, 2i64];

        let mut means_out = vec![0.0f64; 2 * DIM];
        let mut covs_out = vec![0.0f64; 2 * DIM * DIM];
        let mut weights_out = vec![0.0f64; 2];
        let mut v_hf_mean_out = vec![0.0f64; 3];
        let mut pos_var_out = vec![0.0f64; 1];
        let mut valid_out = vec![false; 1];
        let mut failure_code_out = vec![0i32; 1];

        let valid_count = finish_dust_sigma_stack_packed_into(
            PackedSigmaStackInput {
                propagated_sigmas: &propagated_sigmas,
                mean_weights: &sigma_weights.wm,
                covariance_weights: &sigma_weights.wc,
                row_offsets: &row_offsets,
                component_weights: &weights,
                component_linear_dt_s: &component_linear_dt_s,
            },
            PackedSigmaStackOutput {
                means: &mut means_out,
                covariances: &mut covs_out,
                component_weights: &mut weights_out,
                hf_velocity_means: &mut v_hf_mean_out,
                position_variances: &mut pos_var_out,
                valid: &mut valid_out,
                failure_codes: &mut failure_code_out,
            },
        )
        .expect("packed finisher should succeed");

        assert_eq!(valid_count, 1);
        assert_eq!(valid_out, [true]);
        assert_eq!(failure_code_out, [0]);
        assert_f64_slice_bits_eq(&weights_out, &weights, "component weights");

        let mut mean_rows = means_out.chunks_exact(DIM);
        let mean_output_a = mean_rows.next().expect("first mean row");
        let mean_output_b = mean_rows.next().expect("second mean row");
        assert!(mean_rows.next().is_none());
        assert_f64_slice_bits_eq(
            mean_output_a,
            expected_mean_a.as_slice(),
            "component mean A",
        );
        assert_f64_slice_bits_eq(
            mean_output_b,
            expected_mean_b.as_slice(),
            "component mean B",
        );
        let mut covariance_rows = covs_out.chunks_exact(COVARIANCE_LEN);
        let covariance_output_a = covariance_rows.next().expect("first covariance row");
        let covariance_output_b = covariance_rows.next().expect("second covariance row");
        assert!(covariance_rows.next().is_none());
        let flat_expected_covar_a = flatten_covar(&expected_covar_a);
        let flat_expected_covar_b = flatten_covar(&expected_covar_b);
        assert_f64_slice_bits_eq(
            covariance_output_a,
            &flat_expected_covar_a,
            "component covariance A",
        );
        assert_f64_slice_bits_eq(
            covariance_output_b,
            &flat_expected_covar_b,
            "component covariance B",
        );

        // Hand-derived row aggregation over the two (already bit-verified)
        // component means/covars above, in the same lo..hi accumulation
        // order the finisher uses.
        let wa = 3.0f64 / 4.0f64;
        let wb = 1.0f64 / 4.0f64;
        let (first_positions, first_velocities) = expected_mean_a
            .as_slice()
            .split_at_checked(3)
            .expect("first component mean has position and velocity axes");
        let (second_positions, second_velocities) = expected_mean_b
            .as_slice()
            .split_at_checked(3)
            .expect("second component mean has position and velocity axes");
        let mut expected_centroid = [0.0f64; 3];
        let mut expected_v_hf_mean = [0.0f64; 3];
        for ((centroid, first_position), second_position) in expected_centroid
            .iter_mut()
            .zip(first_positions)
            .zip(second_positions)
        {
            *centroid = wa * *first_position + wb * *second_position;
        }
        for ((velocity, first_velocity), second_velocity) in expected_v_hf_mean
            .iter_mut()
            .zip(first_velocities)
            .zip(second_velocities)
        {
            *velocity = wa * *first_velocity + wb * *second_velocity;
        }
        let mut first_diff_sq = 0.0f64;
        for (&position, &centroid) in first_positions.iter().zip(expected_centroid.iter()) {
            let delta = position - centroid;
            first_diff_sq += delta * delta;
        }
        let mut second_diff_sq = 0.0f64;
        for (&position, &centroid) in second_positions.iter().zip(expected_centroid.iter()) {
            let delta = position - centroid;
            second_diff_sq += delta * delta;
        }
        let trace_a = first_three_diagonal_trace(&expected_covar_a);
        let trace_b = first_three_diagonal_trace(&expected_covar_b);
        let expected_pos_var =
            wa * ((trace_a + first_diff_sq) / 3.0) + wb * ((trace_b + second_diff_sq) / 3.0);

        assert_f64_slice_bits_eq(&v_hf_mean_out, &expected_v_hf_mean, "HF velocity mean");
        assert_f64_slice_bits_eq(&pos_var_out, &[expected_pos_var], "position variance");
    }

    #[test]
    fn many_finisher_matches_direct_rebuild_and_packed_dt_zero_path() {
        let weights = julier_simplex_weights();

        let means = [sample_mean(0.0), sample_mean(2.0)];
        let covars = [sample_covar(0.0), sample_covar(0.5)];
        let sigmas: Vec<_> = means
            .iter()
            .zip(covars.iter())
            .map(|(m, c)| get_sigmas_ukf(m, c).expect("sigma gen"))
            .collect();

        let mut propagated_sigmas = Vec::new();
        for s in &sigmas {
            propagated_sigmas.extend(flatten_sigmas(s));
        }

        let mut means_out = vec![0.0f64; 2 * DIM];
        let mut covs_out = vec![0.0f64; 2 * DIM * DIM];
        let n = finish_dust_sigma_stack_many_into(
            &propagated_sigmas,
            &weights.wm,
            &weights.wc,
            &mut means_out,
            &mut covs_out,
        )
        .expect("many finisher should succeed");
        assert_eq!(n, 2);

        for ((mean_output, covariance_output), sigma_points) in means_out
            .chunks_exact(DIM)
            .zip(covs_out.chunks_exact(COVARIANCE_LEN))
            .zip(sigmas.iter())
        {
            let (expected_mean, expected_covar) =
                rebuild_mean_covar_ukf(sigma_points, &weights.wm, &weights.wc);
            assert_f64_slice_bits_eq(mean_output, expected_mean.as_slice(), "many component mean");
            assert_f64_slice_bits_eq(
                covariance_output,
                &flatten_covar(&expected_covar),
                "many component covariance",
            );
        }
    }

    /// Pin: the hand-spelled static table IS the recursion's output, bit for
    /// bit, all 42 entries. The static exists only because `f64::sqrt` is not
    /// `const`; the recursion stays as the oracle so an edit to either side —
    /// a mistyped hex literal or a changed recursion — is a red test here, not
    /// a silently different sigma set.
    #[test]
    fn julier_offsets_table_matches_generator() {
        let generated = julier_whitened_offsets();
        assert_eq!(JULIER_WHITENED_OFFSETS.len(), NUM_SIGMA);
        assert_eq!(generated.len(), NUM_SIGMA);
        for (row_index, (table_row, generated_row)) in JULIER_WHITENED_OFFSETS
            .iter()
            .zip(generated.iter())
            .enumerate()
        {
            assert_eq!(table_row.len(), DIM);
            assert_f64_slice_bits_eq(
                table_row,
                generated_row,
                &format!("julier whitened offsets row {row_index}"),
            );
        }
    }

    #[test]
    fn psd_repair_clamps_negative_eigenvalue_for_sigma_generation() {
        let mean = sample_mean(0.0);
        let mut covar = sample_covar(0.0);
        let [first, ..] = covar.as_mut_slice() else {
            panic!("covariance has a first element");
        };
        *first = -1.0e-13;

        let sigmas =
            get_sigmas_ukf(&mean, &covar).expect("PSD repair should allow near-PSD covariance");

        for value in sigmas.iter() {
            assert!(value.is_finite());
        }
    }
}
