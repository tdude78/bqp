//! Order verification for every Butcher tableau in the crate.
//!
//! # Why this exists
//!
//! The HF lane integrates with these tableaus, and until now five of the six had
//! no test at all while the sixth had a spot check on three `c` entries. A
//! transcription error in a coefficient does not crash and does not obviously
//! misbehave: it silently drops the method's order, so an integrator advertised
//! as 9th order quietly delivers 5th, and every downstream accuracy claim built
//! on it is wrong by an unknown factor. `docs/PART_A_RESULTS_MATRIX.md` recorded
//! this gap as "No in-tree order-condition test for any RK tableau."
//!
//! # What is checked, and what each check can and cannot catch
//!
//! 1. **Row-sum consistency**, `c_i == sum_j a[i][j]`. Exact, and the single most
//!    effective typo detector: almost any mistyped `a` entry breaks it.
//!
//! 2. **Quadrature conditions**, `sum_i b_i c_i^(k-1) == 1/k` for `k = 1..=p`.
//!    These follow from requiring the method to integrate `y' = t^(k-1)` exactly,
//!    and they are the "bushy tree" subset of the full order conditions. They are
//!    NECESSARY for order `p` but not sufficient — a tableau can satisfy all of
//!    them and still fail a coupled condition. Cheap and exact, so they run as a
//!    first filter.
//!
//! 3. **Observed convergence order** on a nonlinear problem with a closed-form
//!    solution. This IS sufficient in practice: it measures the property the
//!    other two only bound. If any order condition is violated, the observed
//!    order collapses to the highest order actually satisfied.
//!
//! Checks 1 and 2 are exact algebra and pin the coefficients. Check 3 is the one
//! that would survive a reviewer asking "how do you know Vern9 is ninth order?"
//!
//! The embedded error weights `b_hat` are checked to their own advertised order,
//! because an error estimator of the wrong order silently mis-sizes every
//! adaptive step.

use super::Tableau;

/// Rate parameter. Larger values sharpen the solution's knee near `t = 1/OMEGA`,
/// which keeps truncation error well above round-off at the step sizes a 9th
/// order method needs. At OMEGA = 1 the problem is so easy that dop853 reaches
/// 1e-10 in five steps and the convergence ratio measures rounding, not order.
const OMEGA: f64 = 3.0;

/// Test problem: `y' = -2 w^2 t y^2`, `y(0) = 1`, exact `y = 1/(1 + w^2 t^2)`.
///
/// Nonlinear in `y` and explicitly `t`-dependent, so it exercises coupled order
/// conditions that a linear problem such as `y' = -y` cannot see. Smooth and
/// bounded on the interval, with no singularity to confound the measurement.
#[inline]
fn rhs(t: f64, y: f64) -> f64 {
    -2.0 * OMEGA * OMEGA * t * y * y
}

#[inline]
fn exact(t: f64) -> f64 {
    1.0 / (1.0 + OMEGA * OMEGA * t * t)
}

/// One fixed explicit Runge-Kutta step from raw coefficients.
#[expect(
    clippy::indexing_slicing,
    reason = "the order oracle intentionally direct-indexes static tableau data and preserves raw RK operation order"
)]
fn rk_step_raw(
    coefficients: &[&[f64]],
    weights: &[f64],
    nodes: &[f64],
    stage_count: usize,
    start_time: f64,
    initial_state: f64,
    step_size: f64,
) -> f64 {
    let mut stage_derivatives = vec![0.0_f64; stage_count];
    for stage_index in 0..stage_count {
        let mut accumulator = initial_state;
        // Rows may be stored ragged (only the populated prefix), so bound by the
        // row's own length as well as by the stage index.
        let row = coefficients[stage_index];
        for coefficient_index in 0..stage_index.min(row.len()) {
            let coefficient = row[coefficient_index];
            if coefficient != 0.0 {
                accumulator += step_size * coefficient * stage_derivatives[coefficient_index];
            }
        }
        stage_derivatives[stage_index] =
            rhs(start_time + nodes[stage_index] * step_size, accumulator);
    }
    let mut output_state = initial_state;
    for stage_index in 0..stage_count {
        if weights[stage_index] != 0.0 {
            output_state += step_size * weights[stage_index] * stage_derivatives[stage_index];
        }
    }
    output_state
}

/// Integrate to `t_end` with `steps` fixed steps and return the absolute error.
fn fixed_step_error_raw(
    coefficients: &[&[f64]],
    weights: &[f64],
    nodes: &[f64],
    stage_count: usize,
    end_time: f64,
    step_count: u16,
) -> f64 {
    let step_size = end_time / f64::from(step_count);
    let mut state = exact(0.0);
    let mut time = 0.0;
    for _ in 0..step_count {
        state = rk_step_raw(
            coefficients,
            weights,
            nodes,
            stage_count,
            time,
            state,
            step_size,
        );
        time += step_size;
    }
    (state - exact(end_time)).abs()
}

fn fixed_step_error(tableau: &Tableau, end_time: f64, step_count: u16) -> f64 {
    fixed_step_error_raw(
        tableau.a,
        tableau.b,
        tableau.c,
        tableau.stages,
        end_time,
        step_count,
    )
}

/// `sum_i b_i c_i^(k-1) - 1/k`, the quadrature-condition residual.
fn quadrature_residual(weights: &[f64], nodes: &[f64], order_condition: i32) -> f64 {
    let lhs: f64 = weights
        .iter()
        .zip(nodes.iter())
        .map(|(&weight, &node)| weight * node.powi(order_condition.saturating_sub(1)))
        .sum();
    lhs - 1.0 / f64::from(order_condition)
}

/// The corpus name every explicit `Method` variant must appear under.
///
/// Exhaustive by construction: adding a `Method` variant fails to compile
/// here, forcing [`all_tableaus`] (and everything sharing its corpus, like
/// `err3_pairing`) to grow in the same edit instead of silently not covering
/// the new tableau.
fn corpus_name(method: crate::odesolve::solver::Method) -> &'static str {
    use crate::odesolve::solver::Method;
    match method {
        Method::Dopri5 => "dopri5",
        Method::Dop853 => "dop853",
        Method::Tsit5 => "tsit5",
        Method::Vern7 => "vern7",
        Method::Vern9 => "vern9",
        Method::Rkv98 => "rkv98",
    }
}

/// One entry per arm of [`corpus_name`]. A variant missing from this list is
/// caught by the membership assertion in [`all_tableaus`], not by the
/// compiler, so keep the two adjacent.
const ALL_METHODS: [crate::odesolve::solver::Method; 6] = [
    crate::odesolve::solver::Method::Dopri5,
    crate::odesolve::solver::Method::Dop853,
    crate::odesolve::solver::Method::Tsit5,
    crate::odesolve::solver::Method::Vern7,
    crate::odesolve::solver::Method::Vern9,
    crate::odesolve::solver::Method::Rkv98,
];

/// Every explicit tableau in the crate, with the order each advertises.
///
/// Every test in this module loops over this corpus and asserts per tableau, so
/// an empty or truncated corpus would make all of them pass while measuring
/// nothing. The count is pinned EXACTLY to the `Method` enum (via
/// [`corpus_name`]'s exhaustive match plus the membership check below), so a
/// row dropped from the list or a variant added to the enum both fail here.
/// The pin lives in the corpus rather than in each caller so no future test
/// can consume it unpinned.
pub(super) fn all_tableaus() -> Vec<(&'static str, &'static Tableau)> {
    let tableaus = vec![
        ("dopri5", super::dopri5::tableau()),
        ("dop853", super::dop853::tableau()),
        ("tsit5", super::tsit5::tableau()),
        ("vern7", super::vern7::tableau()),
        ("vern9", super::vern9::tableau()),
        ("rkv98", super::rkv98::tableau()),
    ];
    assert_eq!(
        tableaus.len(),
        ALL_METHODS.len(),
        "tableau corpus has {} rows; the Method enum names {}",
        tableaus.len(),
        ALL_METHODS.len()
    );
    for method in ALL_METHODS {
        let name = corpus_name(method);
        assert!(
            tableaus.iter().any(|(entry, _)| *entry == name),
            "tableau corpus is missing {name}, which Method::{method:?} advertises"
        );
    }
    tableaus
}

#[expect(
    clippy::float_cmp,
    clippy::indexing_slicing,
    reason = "the static-tableau invariant test intentionally direct-indexes entries and checks exact zero coefficients"
)]
#[test]
fn every_tableau_is_row_sum_consistent() {
    for (name, tableau) in all_tableaus() {
        assert_eq!(
            tableau.a.len(),
            tableau.stages,
            "{name}: a has wrong row count"
        );
        assert_eq!(
            tableau.b.len(),
            tableau.stages,
            "{name}: b has wrong length"
        );
        assert_eq!(
            tableau.c.len(),
            tableau.stages,
            "{name}: c has wrong length"
        );
        for stage_index in 0..tableau.stages {
            let row = tableau.a[stage_index];
            let row_sum: f64 = row.iter().take(stage_index).sum();
            let residual = (row_sum - tableau.c[stage_index]).abs();
            assert!(
                residual < 1.0e-13,
                "{name}: row {stage_index} violates c_i = sum_j a_ij by {residual:.3e} \
                 (c = {}, row sum = {row_sum})",
                tableau.c[stage_index]
            );
            // Explicit method: nothing on or above the diagonal.
            for (coefficient_index, &coefficient) in row.iter().enumerate().skip(stage_index) {
                assert_eq!(
                    coefficient,
                    0.0,
                    "{name}: a[{stage_index}][{coefficient_index}] is non-zero, so the tableau is not explicit"
                );
            }
        }
    }
}

#[test]
fn every_tableau_satisfies_its_quadrature_conditions() {
    for (name, tableau) in all_tableaus() {
        // sum b_i = 1 is the k = 1 case and must hold to full precision.
        let sum_b: f64 = tableau.b.iter().sum();
        assert!(
            (sum_b - 1.0).abs() < 1.0e-13,
            "{name}: sum(b) = {sum_b}, expected 1"
        );

        for order_condition in 2..=i32::from(tableau.order) {
            let lhs: f64 = tableau
                .b
                .iter()
                .zip(tableau.c.iter())
                .map(|(&weight, &node)| weight * node.powi(order_condition.saturating_sub(1)))
                .sum();
            let rhs_val = 1.0 / f64::from(order_condition);
            let residual = (lhs - rhs_val).abs();
            // Tolerance loosens with k because c_i^(k-1) amplifies representation
            // error in the stored coefficients; 1e-12 at k=9 is still far tighter
            // than any real transcription mistake.
            assert!(
                residual < 1.0e-12,
                "{name}: quadrature condition k = {order_condition} fails by {residual:.3e} \
                 (got {lhs}, expected {rhs_val}); the tableau cannot be order {}",
                tableau.order
            );
        }
    }
}

#[test]
fn embedded_error_weights_satisfy_their_own_quadrature_conditions() {
    // A tableau stores its embedded pair in ONE of two equivalent forms: the
    // secondary weights directly in `b_hat`, or the difference
    // `err = btilde = b - b_hat`. Only rkv98 uses `b_hat`; the other five use
    // `err`. Keying solely off `b_hat` -- as this test did until the corpus
    // floor below exposed it -- skipped five of six tableaus and validated a
    // single error estimator while claiming to gate them all.
    //
    // Reconstruct b_hat = b - err for the `err` form so every tableau is
    // checked, and keep `checked` as the backstop against silently sliding
    // back to a partial corpus.
    let mut checked = 0_usize;
    for (name, tableau) in all_tableaus() {
        let b_hat: Vec<f64> = match (tableau.b_hat, tableau.err) {
            (Some(b_hat), _) => b_hat.to_vec(),
            (None, Some(err)) => {
                assert_eq!(
                    err.len(),
                    tableau.b.len(),
                    "{name}: err and b must have the same length"
                );
                tableau
                    .b
                    .iter()
                    .zip(err.iter())
                    .map(|(&weight, &difference)| weight - difference)
                    .collect()
            }
            (None, None) => continue,
        };
        let b_hat = b_hat.as_slice();
        let sum: f64 = b_hat.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1.0e-13,
            "{name}: sum(b_hat) = {sum}, expected 1"
        );
        for order_condition in 2..=i32::from(tableau.order_err) {
            let lhs: f64 = b_hat
                .iter()
                .zip(tableau.c.iter())
                .map(|(&weight, &node)| weight * node.powi(order_condition.saturating_sub(1)))
                .sum();
            let residual = (lhs - 1.0 / f64::from(order_condition)).abs();
            assert!(
                residual < 1.0e-12,
                "{name}: embedded weights fail quadrature condition k = {order_condition} by \
                 {residual:.3e}; the error estimator is not order {}",
                tableau.order_err
            );
        }
        checked += 1;
    }

    // All six explicit tableaus in the crate are embedded pairs, so the floor is
    // the full corpus size rather than whatever currently survives the guard.
    assert!(
        checked >= 6,
        "embedded-weight corpus must check at least 6 tableaus, checked {checked}"
    );
}

/// The sufficient check: measure the order actually delivered.
///
/// Halving the step must reduce the error by `2^p`. The usable window is bounded
/// below by round-off and above by the pre-asymptotic regime, and for a 9th order
/// method it is narrow — so rather than hand-tuning a step count per method, this
/// scans for the coarsest pair whose FINE error still sits safely above round-off.
/// That keeps the instrument honest if someone later changes the test problem.
#[test]
fn every_tableau_delivers_its_advertised_convergence_order() {
    const T_END: f64 = 1.5;
    /// Fine error must stay above this or the ratio measures rounding.
    const ROUNDOFF_FLOOR: f64 = 1.0e-12;
    /// Coarse error must be below this or the method is not yet asymptotic.
    const ASYMPTOTIC_CEILING: f64 = 1.0e-5;

    for (name, tableau) in all_tableaus() {
        // Keep the LAST qualifying pair, not the first. The first pair to dip
        // under the ceiling sits at the pre-asymptotic edge, where the observed
        // rate is still climbing toward p; the finest pair that stays above
        // round-off is the one actually in the asymptotic regime.
        let mut chosen: Option<(u16, f64, f64)> = None;
        for steps in 2_u16..=512 {
            let e_coarse = fixed_step_error(tableau, T_END, steps);
            let e_fine = fixed_step_error(tableau, T_END, steps * 2);
            if e_coarse.is_finite()
                && e_fine.is_finite()
                && e_coarse < ASYMPTOTIC_CEILING
                && e_fine > ROUNDOFF_FLOOR
            {
                chosen = Some((steps, e_coarse, e_fine));
            }
        }

        let (steps, e_coarse, e_fine) = chosen.unwrap_or_else(|| {
            panic!(
                "{name}: found no step count where the coarse error is below \
                 {ASYMPTOTIC_CEILING:.0e} and the fine error above \
                 {ROUNDOFF_FLOOR:.0e}; the test problem no longer suits this order"
            )
        });

        let observed = (e_coarse / e_fine).log2();
        let advertised = f64::from(tableau.order);
        println!(
            "{name:>8}: advertised {advertised:>4.1}, observed {observed:>5.2} \
             (n = {steps}, err {e_coarse:.3e} -> {e_fine:.3e})"
        );

        // One order of slack: the asymptotic rate is approached, not attained, and
        // the fine error is nearer round-off than the coarse one. A transcription
        // error costs several orders, not one.
        assert!(
            observed > advertised - 1.0,
            "{name}: observed convergence order {observed:.2} is far below the \
             advertised {advertised}; a coefficient is likely wrong"
        );
    }
}

/// PROVES the instrument can fail.
///
/// A gate that cannot detect the defect it targets is worse than no gate, because
/// it reports confidence it has not earned. This perturbs a single weight of the
/// vern9 tableau and shows both checks react: the exact quadrature residual
/// exceeds its tolerance, and the measured convergence order collapses from 9th
/// to 1st. Without this, the tests above would be unfalsifiable.
#[expect(
    clippy::indexing_slicing,
    reason = "the mutation-sensitivity test deliberately changes fixed static-table weight index 3"
)]
#[test]
fn the_order_checks_are_sensitive_to_a_single_perturbed_weight() {
    let tab = super::vern9::tableau();

    // A perturbation far smaller than any plausible transcription slip.
    let mut b_small = tab.b.to_vec();
    b_small[3] += 1.0e-10;
    let residual = quadrature_residual(&b_small, tab.c, 2).abs();
    assert!(
        residual > 1.0e-12,
        "a 1e-10 weight perturbation left the quadrature residual at {residual:.3e}, \
         inside the 1e-12 tolerance: the exact check is value-blind"
    );

    // A perturbation the size of a real typo must destroy the convergence order.
    let mut b_bad = tab.b.to_vec();
    b_bad[3] += 1.0e-3;
    let e_coarse = fixed_step_error_raw(tab.a, &b_bad, tab.c, tab.stages, 1.5, 7);
    let e_fine = fixed_step_error_raw(tab.a, &b_bad, tab.c, tab.stages, 1.5, 14);
    let observed = (e_coarse / e_fine).log2();
    println!("vern9 with b[3] += 1e-3: observed order {observed:.2} (was 10.84)");
    assert!(
        observed < 3.0,
        "a 1e-3 weight error still measured order {observed:.2}; the convergence \
         check cannot detect a broken tableau"
    );
}
