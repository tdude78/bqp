//! Permanent instrument for the `#41` cause claim.
//!
//! `f587ae7` corrected the Cunningham y-partial in `satpy_core::gravity`: the
//! `m >= 1` group had V and W transposed, so the y component of the
//! non-central acceleration was wrong. That correction — not the
//! `EGM_GOC_2` -> `GO_CONS_GCF_2_DIR_R6_d15` coefficient swap — is what moved
//! the strict-HF `dv`. The measurement behind that claim originally lived in a
//! scratch worktree; this test is the reproducible replacement.
//!
//! Two things are pinned:
//!   1. the production packed kernel evaluates the CORRECTED coupling, matching
//!      an independent Cunningham/Montenbruck-Gill reference written here, and
//!      NOT the transposed form, so reintroducing the bug fails this test;
//!   2. the magnitude separating the two forms, 2.9206e-8 km/s^2 at the fixture
//!      state, which is the number the `#41` attribution rests on — measured
//!      BETWEEN the two reference forms and then anchored to production, so the
//!      magnitude and the kernel it is attributed to are pinned together.
//!
//! Both tests carry their own sensitivity proof: each re-runs its predicate
//! against the production result with the transposition's delta added back and
//! asserts the predicate REJECTS it. A gate that has stopped gating therefore
//! fails here rather than passing quietly. This is not decoration — the second
//! test previously compared this file's reference against itself and survived a
//! full seven-site transposition of the production y coupling (`#73`).
//!
//! Deliberately NOT covered: the coefficient-swap arm of the comparison. The
//! superseded `EGM_GOC_2` bytes were deleted in `f587ae7` and are not in the
//! tree, so its measured 8.227e-11 km/s^2 remains reproducible only from that
//! commit. Recorded as a limit rather than silently dropped.
//!
//! Uses only `lightyear_odeint_rs::config` and `satpy_core`. It never touches
//! `nd_config` and never reaches `require_production_hybrid_authority`, so it
//! needs no authority bypass to run.

use lightyear_odeint_rs::config::{get_global_coeffs_packed, load_constants_from_bytes};
use satpy_core::{spherical_gravity_impl_sincos_packed, GravityCache, MU};

const ORDER: usize = 5;
const TABLE_PADDING: usize = 3;
const TABLE_SPAN: usize = ORDER + TABLE_PADDING;
/// DIR-R6's normalisation radius, in km, spelled out ON PURPOSE.
///
/// This is the one constant in this file that must NOT be imported, and the
/// reason is the file's whole job. Everything below is an independent
/// re-derivation of the Cunningham recurrence, compared against
/// `spherical_gravity_impl_sincos_packed`. An oracle that took its reference
/// radius from `satpy_core` would share a failure mode with the kernel it
/// checks: re-figure the radius there and BOTH sides move together, the
/// comparison stays green, and the instrument stops instrumenting. The radius
/// is a property of the coefficient file this test loads
/// (`GO_CONS_GCF_2_DIR_R6_d15.txt`), not of `satpy_core`, so restating it here
/// is the correct dependency direction.
///
/// It equals `satpy_core`'s `pub(crate) GRAVITY_REFERENCE_RADIUS_KM`, which no
/// test binary can reach anyway. Do not "repair" this into an import, and note
/// the near-miss waiting for anyone who tries: `satpy_core::RE` IS `pub`, sits
/// 8.5e-8 away at 6378.137, and is the WRONG constant — a figure of the Earth
/// rather than a model parameter, as its own doc comment says.
///
/// If `GRAVITY_REFERENCE_RADIUS_KM` is ever promoted to `pub`,
/// `scripts/test_no_shadow_production_constants.sh` will start flagging this
/// line. The correct response is a `SHADOW_ALLOWED` entry pointing at this
/// comment, not a binding.
const REFERENCE_RADIUS_KM: f64 = 6378.13646;
const RECORDED_ATTRIBUTION_MAGNITUDE: f64 = 2.920_579e-8;
const DIR_R6: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

type Table = Vec<Vec<f64>>;

/// Which y-partial form the reference evaluates.
#[derive(Clone, Copy, PartialEq, Eq)]
enum YCoupling {
    /// Montenbruck & Gill eq. 3.33, the form the tree carries.
    Corrected,
    /// The pre-`f587ae7` form, with V and W transposed in the `m - 1` group.
    Transposed,
}

fn usize_to_f64(value: usize, label: &str) -> anyhow::Result<f64> {
    let value = u32::try_from(value)
        .map_err(|_| anyhow::anyhow!("{label} must fit the reference's u32 domain: {value}"))?;
    Ok(f64::from(value))
}

fn checked_add(left: usize, right: usize, label: &str) -> anyhow::Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| anyhow::anyhow!("{label} addition overflowed"))
}

fn checked_sub(left: usize, right: usize, label: &str) -> anyhow::Result<usize> {
    left.checked_sub(right)
        .ok_or_else(|| anyhow::anyhow!("{label} subtraction underflowed"))
}

fn checked_mul(left: usize, right: usize, label: &str) -> anyhow::Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow::anyhow!("{label} multiplication overflowed"))
}

fn table_value(table: &[Vec<f64>], row: usize, column: usize) -> anyhow::Result<f64> {
    table
        .get(row)
        .and_then(|values| values.get(column))
        .copied()
        .ok_or_else(|| anyhow::anyhow!("reference table index [{row}][{column}] is out of range"))
}

fn table_cell_mut(table: &mut [Vec<f64>], row: usize, column: usize) -> anyhow::Result<&mut f64> {
    table
        .get_mut(row)
        .and_then(|values| values.get_mut(column))
        .ok_or_else(|| anyhow::anyhow!("reference table index [{row}][{column}] is out of range"))
}

fn factorial(value: usize) -> anyhow::Result<f64> {
    (1..=value)
        .try_fold(1.0_f64, |product, term| {
            Ok(product * usize_to_f64(term, "factorial term")?)
        })
        .map(|product| product.max(1.0))
}

/// Parses the sealed coefficient table and denormalizes it.
fn denormalized_coefficients() -> anyhow::Result<(Table, Table)> {
    let mut c = vec![vec![0.0; TABLE_SPAN]; TABLE_SPAN];
    let mut s = vec![vec![0.0; TABLE_SPAN]; TABLE_SPAN];
    *table_cell_mut(&mut c, 0, 0)? = 1.0;
    let table = std::str::from_utf8(DIR_R6)
        .map_err(|error| anyhow::anyhow!("sealed table must be UTF-8: {error}"))?;
    for line in table.lines() {
        let mut fields = line.split_whitespace();
        let (Some(degree_raw), Some(order_raw), Some(c_raw), Some(s_raw)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let degree = degree_raw.parse::<usize>().map_err(|error| {
            anyhow::anyhow!("invalid coefficient degree `{degree_raw}`: {error}")
        })?;
        let order = order_raw
            .parse::<usize>()
            .map_err(|error| anyhow::anyhow!("invalid coefficient order `{order_raw}`: {error}"))?;
        if degree > ORDER || order > degree {
            continue;
        }
        let kind = if order == 0 { 1.0 } else { 2.0 };
        let degree_minus_order = checked_sub(degree, order, "coefficient degree/order")?;
        let twice_degree = checked_mul(2, degree, "coefficient normalization")?;
        let degree_factor = checked_add(twice_degree, 1, "coefficient normalization")?;
        let degree_plus_order = checked_add(degree, order, "coefficient degree/order")?;
        let norm = (factorial(degree_minus_order)?
            * usize_to_f64(degree_factor, "coefficient normalization")?
            * kind
            / factorial(degree_plus_order)?)
        .sqrt();
        let c_value = c_raw
            .parse::<f64>()
            .map_err(|error| anyhow::anyhow!("invalid C coefficient `{c_raw}`: {error}"))?;
        let s_value = s_raw
            .parse::<f64>()
            .map_err(|error| anyhow::anyhow!("invalid S coefficient `{s_raw}`: {error}"))?;
        *table_cell_mut(&mut c, degree, order)? = c_value * norm;
        *table_cell_mut(&mut s, degree, order)? = s_value * norm;
    }
    Ok((c, s))
}

/// Cunningham V/W recursion.
#[expect(
    clippy::suboptimal_flops,
    reason = "independent Cunningham oracle preserves established binary64 expression order"
)]
fn vw_tables(position: [f64; 3]) -> anyhow::Result<(Table, Table)> {
    let [position_x, position_y, position_z] = position;
    let radius_squared =
        position_x * position_x + position_y * position_y + position_z * position_z;
    let radius = radius_squared.sqrt();
    let mut v = vec![vec![0.0; TABLE_SPAN]; TABLE_SPAN];
    let mut w = vec![vec![0.0; TABLE_SPAN]; TABLE_SPAN];
    *table_cell_mut(&mut v, 0, 0)? = REFERENCE_RADIUS_KM / radius;
    let (x_ratio, y_ratio, z_ratio) = (
        position_x * REFERENCE_RADIUS_KM / radius_squared,
        position_y * REFERENCE_RADIUS_KM / radius_squared,
        position_z * REFERENCE_RADIUS_KM / radius_squared,
    );
    let radius_ratio_squared = REFERENCE_RADIUS_KM * REFERENCE_RADIUS_KM / radius_squared;
    let final_row = checked_sub(TABLE_SPAN, 1, "Cunningham table span")?;
    for m in 0..final_row {
        if m > 0 {
            let previous_m = checked_sub(m, 1, "Cunningham diagonal")?;
            let twice_m = checked_mul(2, m, "Cunningham diagonal")?;
            let diagonal_factor = checked_sub(twice_m, 1, "Cunningham diagonal")?;
            let diagonal_factor = usize_to_f64(diagonal_factor, "Cunningham diagonal")?;
            let previous_v = table_value(&v, previous_m, previous_m)?;
            let previous_w = table_value(&w, previous_m, previous_m)?;
            *table_cell_mut(&mut v, m, m)? =
                diagonal_factor * (x_ratio * previous_v - y_ratio * previous_w);
            *table_cell_mut(&mut w, m, m)? =
                diagonal_factor * (x_ratio * previous_w + y_ratio * previous_v);
        }
        let next_m = checked_add(m, 1, "Cunningham row")?;
        let twice_m = checked_mul(2, m, "Cunningham row")?;
        let first_off_diagonal_factor = checked_add(twice_m, 1, "Cunningham row")?;
        let first_off_diagonal_factor = usize_to_f64(first_off_diagonal_factor, "Cunningham row")?;
        let diagonal_v = table_value(&v, m, m)?;
        let diagonal_w = table_value(&w, m, m)?;
        *table_cell_mut(&mut v, next_m, m)? = first_off_diagonal_factor * z_ratio * diagonal_v;
        *table_cell_mut(&mut w, next_m, m)? = first_off_diagonal_factor * z_ratio * diagonal_w;
        let first_l = checked_add(m, 2, "Cunningham recursion")?;
        for l in first_l..TABLE_SPAN {
            let denominator = usize_to_f64(
                checked_sub(l, m, "Cunningham denominator")?,
                "Cunningham denominator",
            )?;
            let twice_l = checked_mul(2, l, "Cunningham recursion")?;
            let leading_factor = checked_sub(twice_l, 1, "Cunningham recursion")?;
            let leading_factor = usize_to_f64(leading_factor, "Cunningham recursion")?;
            let l_plus_m = checked_add(l, m, "Cunningham recursion")?;
            let trailing_factor = checked_sub(l_plus_m, 1, "Cunningham recursion")?;
            let trailing_factor = usize_to_f64(trailing_factor, "Cunningham recursion")?;
            let previous_l = checked_sub(l, 1, "Cunningham recursion")?;
            let previous_previous_l = checked_sub(l, 2, "Cunningham recursion")?;
            let v_previous = table_value(&v, previous_l, m)?;
            let w_previous = table_value(&w, previous_l, m)?;
            let v_previous_previous = table_value(&v, previous_previous_l, m)?;
            let w_previous_previous = table_value(&w, previous_previous_l, m)?;
            *table_cell_mut(&mut v, l, m)? = (leading_factor * z_ratio * v_previous
                - trailing_factor * radius_ratio_squared * v_previous_previous)
                / denominator;
            *table_cell_mut(&mut w, l, m)? = (leading_factor * z_ratio * w_previous
                - trailing_factor * radius_ratio_squared * w_previous_previous)
                / denominator;
        }
    }
    Ok((v, w))
}

/// Independent reference acceleration, in km/s^2.
#[expect(
    clippy::suboptimal_flops,
    reason = "independent Cunningham/Montenbruck-Gill oracle preserves formula operation order"
)]
fn reference_acceleration(position: [f64; 3], coupling: YCoupling) -> anyhow::Result<[f64; 3]> {
    let (c, s) = denormalized_coefficients()?;
    let (v, w) = vw_tables(position)?;
    let scale = MU / (REFERENCE_RADIUS_KM * REFERENCE_RADIUS_KM);
    let mut acceleration_x = 0.0;
    let mut acceleration_y = 0.0;
    let mut acceleration_z = 0.0;
    for l in 0..=ORDER {
        for m in 0..=l {
            let clm = table_value(&c, l, m)?;
            let slm = table_value(&s, l, m)?;
            if clm == 0.0 && slm == 0.0 {
                continue;
            }
            let next_l = checked_add(l, 1, "Cunningham acceleration")?;
            let (x1, y1, z1, weight) = if m == 0 {
                let degree_factor = usize_to_f64(next_l, "Cunningham acceleration")?;
                (
                    -clm * table_value(&v, next_l, 1)?,
                    -clm * table_value(&w, next_l, 1)?,
                    degree_factor * (-clm * table_value(&v, next_l, 0)?),
                    1.0,
                )
            } else {
                let previous_m = checked_sub(m, 1, "Cunningham acceleration")?;
                let next_m = checked_add(m, 1, "Cunningham acceleration")?;
                let l_minus_m = checked_sub(l, m, "Cunningham acceleration")?;
                let first_cf2 = checked_add(l_minus_m, 2, "Cunningham acceleration")?;
                let second_cf2 = checked_add(l_minus_m, 1, "Cunningham acceleration")?;
                let cf2 = usize_to_f64(
                    checked_mul(first_cf2, second_cf2, "Cunningham acceleration")?,
                    "Cunningham acceleration",
                )?;
                let tail = match coupling {
                    YCoupling::Corrected => {
                        -clm * table_value(&w, next_l, previous_m)?
                            + slm * table_value(&v, next_l, previous_m)?
                    }
                    YCoupling::Transposed => {
                        -clm * table_value(&v, next_l, previous_m)?
                            + slm * table_value(&w, next_l, previous_m)?
                    }
                };
                (
                    (-clm * table_value(&v, next_l, next_m)?
                        - slm * table_value(&w, next_l, next_m)?)
                        + cf2
                            * (clm * table_value(&v, next_l, previous_m)?
                                + slm * table_value(&w, next_l, previous_m)?),
                    (-clm * table_value(&w, next_l, next_m)?
                        + slm * table_value(&v, next_l, next_m)?)
                        + cf2 * tail,
                    usize_to_f64(
                        checked_add(l_minus_m, 1, "Cunningham acceleration")?,
                        "Cunningham acceleration",
                    )? * (-clm * table_value(&v, next_l, m)? - slm * table_value(&w, next_l, m)?),
                    0.5,
                )
            };
            acceleration_x += scale * weight * x1;
            acceleration_y += scale * weight * y1;
            acceleration_z += scale * weight * z1;
        }
    }
    Ok([acceleration_x, acceleration_y, acceleration_z])
}

fn production_acceleration(position: [f64; 3]) -> anyhow::Result<[f64; 3]> {
    load_constants_from_bytes(DIR_R6, ORDER)
        .map_err(|error| anyhow::anyhow!("sealed coefficients must load: {error}"))?;
    let packed = get_global_coeffs_packed()
        .ok_or_else(|| anyhow::anyhow!("coefficients must be global after loading"))?;
    let [position_x, position_y, position_z] = position;
    let state = [position_x, position_y, position_z, 0.0, 0.0, 0.0];
    let mut cache = GravityCache::default();
    spherical_gravity_impl_sincos_packed(&state, 0.0, 1.0, &mut cache, packed.as_ref())
        .map_err(|error| anyhow::anyhow!("production packed gravity evaluation failed: {error}"))
}

#[expect(
    clippy::suboptimal_flops,
    reason = "oracle norm preserves the recorded scalar operation order"
)]
fn norm(vector: [f64; 3]) -> f64 {
    let [x_component, y_component, z_component] = vector;
    (x_component * x_component + y_component * y_component + z_component * z_component).sqrt()
}

fn difference(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    let [left_x, left_y, left_z] = lhs;
    let [right_x, right_y, right_z] = rhs;
    [left_x - right_x, left_y - right_y, left_z - right_z]
}

fn sum(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    let [left_x, left_y, left_z] = lhs;
    let [right_x, right_y, right_z] = rhs;
    [left_x + right_x, left_y + right_y, left_z + right_z]
}

const FIXTURE_STATE: [f64; 3] = [6778.0, 0.0, 0.0];

/// What reintroducing the transposition ADDS to an evaluated acceleration.
///
/// The two reference forms share their coefficient table and their whole V/W
/// recursion; only the `m >= 1` y term differs, so their difference isolates
/// the defect exactly. Production and the reference agree on x and y to machine
/// precision (see [`carries_corrected_y_coupling`]), which makes adding this
/// vector to a production result a faithful stand-in for a production kernel
/// that carries the defect — and lets the sensitivity proofs run without
/// touching `satpy_core::gravity`, which is guard-tested on its literal source
/// text and fails at runtime if edited.
fn transposition_defect(position: [f64; 3]) -> anyhow::Result<[f64; 3]> {
    Ok(difference(
        reference_acceleration(position, YCoupling::Transposed)?,
        reference_acceleration(position, YCoupling::Corrected)?,
    ))
}

/// The corrected-coupling predicate, returned as a `Result` rather than
/// asserted so each caller can check that it ACCEPTS the production kernel and
/// REJECTS the defective form.
///
/// It reads the y component alone, deliberately. The independently written
/// reference sits ~3e-8 km/s^2 from production overall, but that offset lives
/// entirely in z: on x and y the two agree to machine precision at every
/// position used here (measured y gaps 2.0e-23, 0.0 and 8.7e-19 km/s^2, against
/// defects of 2.9e-8 .. 7.8e-8). Judging the y coupling on the y component
/// therefore separates signal from that unrelated z discrepancy by fifteen or
/// more orders of magnitude, instead of letting the two sit at comparable size
/// inside a 3-vector norm.
fn carries_corrected_y_coupling(evaluated: [f64; 3], position: [f64; 3]) -> anyhow::Result<()> {
    let corrected = reference_acceleration(position, YCoupling::Corrected)?;
    let defect = transposition_defect(position)?;
    let [_, evaluated_y, _] = evaluated;
    let [_, corrected_y, _] = corrected;
    let [_, defect_y, _] = defect;
    let gap = (evaluated_y - corrected_y).abs();
    // Machine-precision agreement, scaled to the larger of |a_y| and the defect
    // size so the bound stays meaningful where a_y itself is near zero (it is
    // 2.05e-8 at the fixture state, on the equator with y = 0).
    let tolerance = 1.0e-12 * corrected_y.abs().max(norm(defect));
    if gap <= tolerance {
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "y coupling is not the corrected form at {position:?}: evaluated_y={evaluated_y:.12e} \
         corrected_y={corrected_y:.12e} gap={gap:.6e} tolerance={tolerance:.6e} \
         defect_y={defect_y:.6e}",
    ))
}

const fn is_exact_zero(value: f64) -> bool {
    matches!(value.classify(), std::num::FpCategory::Zero)
}

#[test]
fn production_kernel_uses_the_corrected_cunningham_y_coupling() -> anyhow::Result<()> {
    for position in [
        FIXTURE_STATE,
        [4500.0, 4200.0, 2600.0],
        [3000.0, -2000.0, 5900.0],
    ] {
        let actual = production_acceleration(position)?;
        let corrected = reference_acceleration(position, YCoupling::Corrected)?;
        let transposed = reference_acceleration(position, YCoupling::Transposed)?;

        let to_corrected = norm(difference(actual, corrected));
        let to_transposed = norm(difference(actual, transposed));
        let separation = norm(difference(corrected, transposed));

        // The reference is an independently written implementation, so it
        // carries a small residual offset against the production kernel
        // (~2.6e-8 km/s^2, 3e-6 relative) from modelling detail rather than
        // from the coupling. Absolute agreement is therefore NOT asserted;
        // what is asserted is that production sits nearer the corrected form
        // than the transposed one. Reintroducing the transposition moves
        // production by `separation` and flips this inequality.
        if !matches!(
            to_transposed.partial_cmp(&(to_corrected * 1.2)),
            Some(std::cmp::Ordering::Greater)
        ) {
            return Err(anyhow::anyhow!(
                "production kernel does not favour the corrected Cunningham y \
                 coupling at {position:?}: to_corrected={to_corrected:.6e} \
                 to_transposed={to_transposed:.6e} separation={separation:.6e}"
            ));
        }

        // Sensitivity, proven rather than assumed. The residual above (2.6e-8
        // at the fixture state) is the same order as `separation` (2.9e-8), so
        // the 1.2 margin being sufficient is a fact about these numbers, not
        // something the inequality's shape guarantees. Re-run it against the
        // production result with the transposition's own delta added back —
        // the number the pre-`f587ae7` kernel would have produced — and require
        // that it FAILS.
        let defective = sum(actual, transposition_defect(position)?);
        let defective_to_corrected = norm(difference(defective, corrected));
        let defective_to_transposed = norm(difference(defective, transposed));
        if !matches!(
            defective_to_transposed.partial_cmp(&(defective_to_corrected * 1.2)),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        ) {
            return Err(anyhow::anyhow!(
                "the assertion above cannot detect the transposition at \
                 {position:?}: the defective form still satisfies it with \
                 to_corrected={defective_to_corrected:.6e} \
                 to_transposed={defective_to_transposed:.6e}"
            ));
        }

        // The same defective form must also fail the sharper y-only predicate,
        // while production passes it.
        carries_corrected_y_coupling(actual, position)?;
        if carries_corrected_y_coupling(defective, position).is_ok() {
            return Err(anyhow::anyhow!(
                "the y-only predicate accepted transposed coupling at {position:?}"
            ));
        }
    }
    Ok(())
}

/// The `#41` attribution magnitude — and, since `f587ae7` is a claim about the
/// PRODUCTION kernel, that production is the end of the interval that magnitude
/// is measured across.
///
/// REPAIRED. The previous body compared `reference_acceleration(Corrected)`
/// against `reference_acceleration(Transposed)` and nothing else. Both sides
/// were computed in this file from this file's own `YCoupling` enum, so no
/// change to `satpy_core::gravity` could move either one — the comparison was
/// closed over the test's own code. It measured 2.9206e-8 correctly and gated
/// nothing: `#73` transposed the y coupling at all seven production sites and
/// this test stayed GREEN while its neighbour went RED.
///
/// The magnitude assertion is kept verbatim, because it is the recorded number
/// and it was never the broken part. What is added is the production anchor:
/// production must sit at the corrected end, and `RECORDED` away from the
/// transposed end, of the very interval the magnitude measures.
#[test]
fn corrected_coupling_reproduces_the_recorded_attribution_magnitude() -> anyhow::Result<()> {
    let corrected = reference_acceleration(FIXTURE_STATE, YCoupling::Corrected)?;
    let transposed = reference_acceleration(FIXTURE_STATE, YCoupling::Transposed)?;
    let delta = difference(corrected, transposed);
    let [delta_x, _, delta_z] = delta;

    // The correction is a y-partial fix: x and z must be untouched.
    if !is_exact_zero(delta_x) {
        return Err(anyhow::anyhow!("x component must not move"));
    }
    if !is_exact_zero(delta_z) {
        return Err(anyhow::anyhow!("z component must not move"));
    }

    // Recorded in the closeout ledger and the submission runbook.
    let measured = norm(delta);
    let relative =
        (measured - RECORDED_ATTRIBUTION_MAGNITUDE).abs() / RECORDED_ATTRIBUTION_MAGNITUDE;
    if !matches!(
        relative.partial_cmp(&1.0e-4),
        Some(std::cmp::Ordering::Less)
    ) {
        return Err(anyhow::anyhow!(
            "recorded #41 magnitude not reproduced: measured={measured:.6e} \
             recorded={RECORDED_ATTRIBUTION_MAGNITUDE:.6e} relative={relative:.3e}"
        ));
    }

    // --- the production anchor: this is what makes the test able to fail ---

    let actual = production_acceleration(FIXTURE_STATE)?;
    carries_corrected_y_coupling(actual, FIXTURE_STATE)?;

    // Production must be RECORDED away from the transposed form, not merely
    // nearer to one than the other. Reintroducing the transposition drives this
    // gap to zero.
    let [_, actual_y, _] = actual;
    let [_, transposed_y, _] = transposed;
    let gap_to_transposed = (actual_y - transposed_y).abs();
    let gap_relative =
        (gap_to_transposed - RECORDED_ATTRIBUTION_MAGNITUDE).abs() / RECORDED_ATTRIBUTION_MAGNITUDE;
    if !matches!(
        gap_relative.partial_cmp(&1.0e-4),
        Some(std::cmp::Ordering::Less)
    ) {
        return Err(anyhow::anyhow!(
            "production is not `RECORDED` away from the transposed form: \
             gap={gap_to_transposed:.6e} recorded={RECORDED_ATTRIBUTION_MAGNITUDE:.6e} \
             relative={gap_relative:.3e}"
        ));
    }

    // --- sensitivity proof ---
    //
    // Both new assertions must REJECT a production kernel carrying the defect,
    // simulated by adding the transposition's own delta to the production
    // result. Without this the repair would itself be unproven.
    let defective = sum(actual, transposition_defect(FIXTURE_STATE)?);
    let rejection = carries_corrected_y_coupling(defective, FIXTURE_STATE)
        .err()
        .ok_or_else(|| {
            anyhow::anyhow!("the corrected-coupling predicate accepted the defective form")
        })?;
    if !rejection
        .to_string()
        .contains("y coupling is not the corrected form")
    {
        return Err(rejection);
    }
    let [_, defective_y, _] = defective;
    let defective_gap = (defective_y - transposed_y).abs();
    let defective_relative =
        (defective_gap - RECORDED_ATTRIBUTION_MAGNITUDE).abs() / RECORDED_ATTRIBUTION_MAGNITUDE;
    if !matches!(
        defective_relative.partial_cmp(&1.0e-4),
        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
    ) {
        return Err(anyhow::anyhow!(
            "the magnitude anchor cannot detect the transposition: the defective \
             form still sits RECORDED from the transposed one \
             (gap={defective_gap:.6e} recorded={RECORDED_ATTRIBUTION_MAGNITUDE:.6e})"
        ));
    }
    Ok(())
}
