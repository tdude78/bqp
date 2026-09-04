//! ESDIRK implicit step with Newton iteration.
//!
//! Performs a single step of an ESDIRK method by solving the implicit stage
//! equations via simplified Newton iteration with a single LU factorization
//! of the iteration matrix W = I − h·γ·J.

use crate::odesolve::lu6::Lu6;
use crate::odesolve::solver::{IntegrationStats, OdeSystem};
use crate::odesolve::tableau_esdirk::EsdirkTableau;

/// Trait for providing the Jacobian df/dy evaluated at (t, y).
///
/// The Jacobian is written into a 6×6 array `jac`, stored row-major:
/// `jac[i][j] = ∂f_i/∂y_j`.
///
/// Implementations live at the `lightyear_odeint_rs` crate level (e.g.
/// via automatic differentiation or finite differences). This trait is
/// intentionally generic — it knows nothing about `LightyearDualRHS`.
pub trait JacobianProvider {
    fn jacobian(&self, t: f64, y: &[f64], jac: &mut [[f64; 6]; 6]);
}

/// Outcome of a single ESDIRK step attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplicitStepResult {
    /// Step succeeded; `y_next` / err are valid.
    Accepted,
    /// Inputs cannot describe a valid ESDIRK step.
    InvalidInput,
    /// Newton iteration failed to converge within the allowed iterations.
    NewtonFailed,
}

const ESDIRK_STAGE_COUNT: usize = 6;

/// Perform one ESDIRK step.
///
/// # Arguments
///
/// * `system`         — ODE right-hand-side f(t, y)
/// * `jac_provider`   — Jacobian evaluator df/dy
/// * `tableau`        — ESDIRK tableau (6 stages, order 4/3)
/// * `t`              — Current time
/// * `y`              — Current state (6-element)
/// * `h`              — Step size (signed; negative for backward integration)
/// * `k`              — Stage derivatives (6 stages × 6 state components); filled on return
/// * `y_next`         — Solution at t+h (filled on return if Accepted)
/// * `err`            — Error estimate `y_next` − `y_hat` (filled on return if Accepted)
/// * `newton_tol`     — Convergence tolerance for Newton iteration (max-norm)
/// * `max_newton_iter` — Maximum Newton iterations per implicit stage
/// * `stats`          — Accumulator for step/eval counters
/// * `reuse_k0`       — If `Some`, reuse as k[0] (FSAL from previous step)
///
/// # Algorithm
///
/// Stage 0 is explicit: k[0] = f(t, y).
///
/// For stages i = 1..5:
///   1. Compute the explicit predictor `z_i` = y + h·Σ_{m<i} a[i][m]·k[m].
///   2. On stage 1, compute the Jacobian and factor W = I − h·γ·J.
///   3. Run simplified Newton iteration to solve k[i] − f(t + c[i]·h, `z_i` + h·γ·k[i]) = 0.
///
/// After all 6 stages, form:
///   `y_next` = y + h·Σ b[i]·k[i]
///   err    = h·Σ (b[i] − `b_hat`[i])·k[i]
///
/// FSAL property: since the method is stiffly accurate, k[5] = f(t+h, `y_next`),
/// and the caller can pass it as `reuse_k0` on the next step.
#[expect(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    reason = "the fixed-size numerical kernel follows standard RK notation"
)]
pub fn esdirk_step<S: OdeSystem, J: JacobianProvider>(
    system: &S,
    jac_provider: &J,
    tableau: &EsdirkTableau,
    t: f64,
    y: &[f64; 6],
    h: f64,
    k: &mut [[f64; 6]; 6],
    y_next: &mut [f64; 6],
    err: &mut [f64; 6],
    newton_tol: f64,
    max_newton_iter: usize,
    stats: &mut IntegrationStats,
    reuse_k0: Option<&[f64; 6]>,
) -> ImplicitStepResult {
    if tableau.stages != ESDIRK_STAGE_COUNT
        || !newton_tol.is_finite()
        || newton_tol <= 0.0
        || max_newton_iter == 0
    {
        return ImplicitStepResult::InvalidInput;
    }
    let Some(implicit_stage_evals) = (ESDIRK_STAGE_COUNT - 1).checked_mul(max_newton_iter) else {
        return ImplicitStepResult::InvalidInput;
    };
    let initial_stage_evals = usize::from(reuse_k0.is_none());
    let Some(required_evals) = implicit_stage_evals.checked_add(initial_stage_evals) else {
        return ImplicitStepResult::InvalidInput;
    };
    if stats.evals.checked_add(required_evals).is_none() {
        return ImplicitStepResult::InvalidInput;
    }
    let gamma = tableau.gamma;

    // --- Stage 0: explicit ---
    let Some(first_stage) = k.first_mut() else {
        return ImplicitStepResult::NewtonFailed;
    };
    if let Some(k0) = reuse_k0 {
        *first_stage = *k0;
    } else {
        let mut dy = [0.0f64; 6];
        let Some(evals) = stats.evals.checked_add(1) else {
            return ImplicitStepResult::InvalidInput;
        };
        system.rhs(t, y, &mut dy);
        *first_stage = dy;
        stats.evals = evals;
    }

    // Scratch arrays for Newton iteration
    let mut jac = [[0.0f64; 6]; 6];
    let mut w_lu: Option<Lu6> = None;
    let mut z: [f64; 6];
    let mut y_trial = [0.0f64; 6];
    let mut f_trial = [0.0f64; 6];
    let mut residual = [0.0f64; 6];

    // --- Implicit stages 1..5 ---
    for stage in 1..tableau.stages {
        let Some(tableau_row) = tableau.a.get(stage) else {
            return ImplicitStepResult::NewtonFailed;
        };
        // 1. Explicit predictor: z = y + h * sum_{m=0}^{stage-1} a[stage][m] * k[m]
        z = *y;
        for (k_row, &a_sm) in k.iter().zip(tableau_row).take(stage) {
            if a_sm != 0.0 {
                let ha = h * a_sm;
                for (z_j, &k_mj) in z.iter_mut().zip(k_row.iter()) {
                    *z_j += ha * k_mj;
                }
            }
        }

        // 2. On stage 1: compute Jacobian and factor W = I - h*gamma*J
        if stage == 1 {
            jac_provider.jacobian(t, y, &mut jac);

            // Form W = I - h*gamma*J
            let hg = h * gamma;
            let mut w_mat = [[0.0f64; 6]; 6];
            for (row_index, (w_row, jac_row)) in w_mat.iter_mut().zip(jac.iter()).enumerate() {
                for (value, &jacobian) in w_row.iter_mut().zip(jac_row) {
                    *value = -hg * jacobian;
                }
                let Some(diagonal) = w_row.get_mut(row_index) else {
                    return ImplicitStepResult::NewtonFailed;
                };
                *diagonal += 1.0;
            }
            let lu = Lu6::factor(&w_mat);
            if lu.singular {
                return ImplicitStepResult::NewtonFailed;
            }
            w_lu = Some(lu);
        }

        let Some(lu) = w_lu.as_ref() else {
            return ImplicitStepResult::NewtonFailed;
        };

        // 3. Initial guess: k[stage] = k[stage-1]
        let (prior_stages, current_and_later) = k.split_at_mut(stage);
        let Some(previous_stage) = prior_stages.last().copied() else {
            return ImplicitStepResult::NewtonFailed;
        };
        let Some(current_stage) = current_and_later.first_mut() else {
            return ImplicitStepResult::NewtonFailed;
        };
        *current_stage = previous_stage;

        // 4. Newton iteration
        let Some(&stage_node) = tableau.c.get(stage) else {
            return ImplicitStepResult::NewtonFailed;
        };
        let t_stage = t + stage_node * h;
        let mut converged = false;

        for _iter in 0..max_newton_iter {
            // Trial point: y_trial = z + h * gamma * k[stage]
            let hg = h * gamma;
            for ((trial, &predictor), &derivative) in
                y_trial.iter_mut().zip(z.iter()).zip(current_stage.iter())
            {
                *trial = predictor + hg * derivative;
            }

            // Evaluate RHS at trial point
            let Some(evals) = stats.evals.checked_add(1) else {
                return ImplicitStepResult::InvalidInput;
            };
            system.rhs(t_stage, &y_trial, &mut f_trial);
            stats.evals = evals;

            // Residual: G = k[stage] - f_trial
            for ((value, &derivative), &force) in residual
                .iter_mut()
                .zip(current_stage.iter())
                .zip(f_trial.iter())
            {
                *value = derivative - force;
            }

            // Solve W * dk = -G  =>  dk stored in residual (negated first)
            for value in &mut residual {
                *value = -*value;
            }
            lu.solve(&mut residual);

            // Update: k[stage] += dk
            let mut dk_norm = 0.0f64;
            for (derivative, &correction) in current_stage.iter_mut().zip(residual.iter()) {
                *derivative += correction;
                let abs_dk = correction.abs();
                if abs_dk > dk_norm {
                    dk_norm = abs_dk;
                }
            }

            // Check convergence
            if dk_norm < newton_tol {
                converged = true;
                break;
            }
        }

        if !converged {
            return ImplicitStepResult::NewtonFailed;
        }
    }

    // --- Form solution and error estimate ---
    // y_next = y + h * sum(b[i] * k[i])
    *y_next = *y;
    err.fill(0.0);

    // Kahan compensation for error accumulation to avoid catastrophic
    // cancellation at tight eps.
    let mut err_comp = [0.0f64; 6];
    for ((&b_i, &b_hat_i), k_row) in tableau
        .b
        .iter()
        .zip(tableau.b_hat.iter())
        .zip(k.iter())
        .take(tableau.stages)
    {
        let hb = h * b_i;
        let h_err = h * (b_i - b_hat_i);
        for ((y_next_j, err_j), (err_comp_j, &k_ij)) in y_next
            .iter_mut()
            .zip(err.iter_mut())
            .zip(err_comp.iter_mut().zip(k_row.iter()))
        {
            *y_next_j += hb * k_ij;
            let term = h_err * k_ij;
            let y_k = term - *err_comp_j;
            let t = *err_j + y_k;
            *err_comp_j = (t - *err_j) - y_k;
            *err_j = t;
        }
    }

    ImplicitStepResult::Accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odesolve::tableau_esdirk::{esdirk43_tableau, EsdirkTableau};
    use std::cell::Cell;

    // --- Test systems ---

    /// Harmonic oscillator: y'' = -y
    /// State: [y, v] but extended to 6D as [y, 0, 0, v, 0, 0]
    struct HarmonicOscillator;

    impl OdeSystem for HarmonicOscillator {
        fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) {
            // dy[0]/dt = y[3] (velocity)
            // dy[3]/dt = -y[0] (acceleration)
            // Others are zero
            let (Some(&position), Some(&velocity)) = (y.first(), y.get(3)) else {
                return;
            };
            let rates = [velocity, 0.0, 0.0, -position, 0.0, 0.0];
            for (derivative, value) in dy.iter_mut().zip(rates) {
                *derivative = value;
            }
        }
    }

    struct HarmonicOscillatorJac;

    impl JacobianProvider for HarmonicOscillatorJac {
        fn jacobian(&self, _t: f64, _y: &[f64], jac: &mut [[f64; 6]; 6]) {
            for row in jac.iter_mut() {
                *row = [0.0; 6];
            }
            jac[0][3] = 1.0;
            jac[3][0] = -1.0;
        }
    }

    /// Stiff exponential decay: y' = -lambda * y, lambda = 1000
    /// Extended to 6D: [y, 0, 0, 0, 0, 0]
    struct StiffDecay {
        lambda: f64,
    }

    impl OdeSystem for StiffDecay {
        fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) {
            let Some(&state) = y.first() else {
                return;
            };
            let Some(state_derivative) = dy.first_mut() else {
                return;
            };
            *state_derivative = -self.lambda * state;
            for value in dy.iter_mut().skip(1) {
                *value = 0.0;
            }
        }
    }

    struct StiffDecayJac {
        lambda: f64,
    }

    impl JacobianProvider for StiffDecayJac {
        fn jacobian(&self, _t: f64, _y: &[f64], jac: &mut [[f64; 6]; 6]) {
            for row in jac.iter_mut() {
                *row = [0.0; 6];
            }
            jac[0][0] = -self.lambda;
        }
    }

    #[derive(Default)]
    struct CallbackCountingSystem {
        calls: Cell<usize>,
    }

    impl OdeSystem for CallbackCountingSystem {
        fn rhs(&self, _t: f64, _y: &[f64], dy: &mut [f64]) {
            self.calls.set(self.calls.get().saturating_add(1));
            dy.fill(0.0);
        }
    }

    #[derive(Default)]
    struct CallbackCountingJacobian {
        calls: Cell<usize>,
    }

    impl JacobianProvider for CallbackCountingJacobian {
        fn jacobian(&self, _t: f64, _y: &[f64], jac: &mut [[f64; 6]; 6]) {
            self.calls.set(self.calls.get().saturating_add(1));
            jac.fill([0.0; 6]);
        }
    }

    fn populated_stats(evals: usize) -> IntegrationStats {
        IntegrationStats {
            steps: 17,
            evals,
            saturated_steps: 19,
            underflow_accepts: 23,
            rejected_steps: 29,
            min_accepted_h: 31.25,
            // Distinct sentinels, like every field above: this helper exists to
            // prove the implicit step passes stats through UNCHANGED, and a
            // field left at its default would be indistinguishable from one the
            // callee zeroed. Deliberately NOT `..Default::default()`.
            first_accepted_h: [43.5, 47.25, 53.125, 59.0625, 61.5],
            tail_h_sum: 67.75,
            tail_h_count: 71,
            segment_span_s: 73.5,
            cache_cluster_steps: 37,
            cache_cluster_steps_untruncated: 41,
            final_controller_h: 79.25,
        }
    }

    fn assert_stats_unchanged(actual: IntegrationStats, expected: IntegrationStats) {
        assert_eq!(actual.steps, expected.steps);
        assert_eq!(actual.evals, expected.evals);
        assert_eq!(actual.saturated_steps, expected.saturated_steps);
        assert_eq!(actual.underflow_accepts, expected.underflow_accepts);
        assert_eq!(actual.rejected_steps, expected.rejected_steps);
        assert_eq!(
            actual.min_accepted_h.to_bits(),
            expected.min_accepted_h.to_bits()
        );
        assert_eq!(
            actual.first_accepted_h.map(f64::to_bits),
            expected.first_accepted_h.map(f64::to_bits)
        );
        assert_eq!(actual.tail_h_sum.to_bits(), expected.tail_h_sum.to_bits());
        assert_eq!(actual.tail_h_count, expected.tail_h_count);
        assert_eq!(
            actual.segment_span_s.to_bits(),
            expected.segment_span_s.to_bits()
        );
        assert_eq!(actual.cache_cluster_steps, expected.cache_cluster_steps);
        assert_eq!(
            actual.cache_cluster_steps_untruncated,
            expected.cache_cluster_steps_untruncated
        );
        assert_eq!(
            actual.final_controller_h.to_bits(),
            expected.final_controller_h.to_bits()
        );
    }

    fn assert_invalid_preflight(
        tableau: &EsdirkTableau,
        newton_tol: f64,
        max_newton_iter: usize,
        evals: usize,
    ) {
        let system = CallbackCountingSystem::default();
        let jacobian = CallbackCountingJacobian::default();
        let y = [1.0; 6];
        let mut k = [[0.0; 6]; 6];
        let mut y_next = [0.0; 6];
        let mut err = [0.0; 6];
        let mut stats = populated_stats(evals);
        let before = stats;

        let result = esdirk_step(
            &system,
            &jacobian,
            tableau,
            0.0,
            &y,
            0.1,
            &mut k,
            &mut y_next,
            &mut err,
            newton_tol,
            max_newton_iter,
            &mut stats,
            None,
        );

        assert_eq!(result, ImplicitStepResult::InvalidInput);
        assert_eq!(system.calls.get(), 0);
        assert_eq!(jacobian.calls.get(), 0);
        assert_stats_unchanged(stats, before);
    }

    fn tableau_with_stages(stages: usize) -> EsdirkTableau {
        let source = esdirk43_tableau();
        EsdirkTableau {
            stages,
            a: source.a,
            c: source.c,
            b: source.b,
            b_hat: source.b_hat,
            gamma: source.gamma,
            order_err: source.order_err,
        }
    }

    #[test]
    fn esdirk_step_rejects_non_six_stage_tableau_before_callbacks() {
        let malformed = tableau_with_stages(1);
        assert_invalid_preflight(&malformed, 1.0e-12, 1, 43);
    }

    #[test]
    fn esdirk_step_rejects_zero_newton_iterations_before_callbacks() {
        assert_invalid_preflight(esdirk43_tableau(), 1.0e-12, 0, 43);
    }

    #[test]
    fn esdirk_step_rejects_nonpositive_or_nonfinite_newton_tolerance_before_callbacks() {
        for newton_tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_invalid_preflight(esdirk43_tableau(), newton_tol, 1, 43);
        }
    }

    #[test]
    fn esdirk_step_rejects_insufficient_full_eval_budget_before_callbacks() {
        // One explicit stage plus five implicit stages, each with one Newton RHS.
        assert_invalid_preflight(esdirk43_tableau(), 1.0e-12, 1, usize::MAX - 5);
    }

    #[test]
    fn test_single_step_harmonic() {
        let sys = HarmonicOscillator;
        let jac = HarmonicOscillatorJac;
        let tab = esdirk43_tableau();

        let y0: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // y=1, v=0
        let h = 0.1;
        let mut k = [[0.0f64; 6]; 6];
        let mut y_next = [0.0f64; 6];
        let mut err = [0.0f64; 6];
        let mut stats = IntegrationStats::default();

        let result = esdirk_step(
            &sys,
            &jac,
            tab,
            0.0,
            &y0,
            h,
            &mut k,
            &mut y_next,
            &mut err,
            1e-12,
            20,
            &mut stats,
            None,
        );

        assert_eq!(result, ImplicitStepResult::Accepted);

        // Exact: y(0.1) = cos(0.1) ≈ 0.99500416527803
        // v(0.1) = -sin(0.1) ≈ -0.09983341664683
        let y_exact = 0.1_f64.cos();
        let v_exact = -(0.1_f64.sin());

        assert!(
            (y_next[0] - y_exact).abs() < 1e-8,
            "y(0.1): expected {}, got {} (err {:e})",
            y_exact,
            y_next[0],
            (y_next[0] - y_exact).abs()
        );
        assert!(
            (y_next[3] - v_exact).abs() < 1e-8,
            "v(0.1): expected {}, got {} (err {:e})",
            v_exact,
            y_next[3],
            (y_next[3] - v_exact).abs()
        );
    }

    #[test]
    fn test_single_step_stiff_decay() {
        let lambda = 1000.0;
        let sys = StiffDecay { lambda };
        let jac = StiffDecayJac { lambda };
        let tab = esdirk43_tableau();

        let y0: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // Test stability: multiple smaller steps so Newton converges well
        let h = 0.001; // h*lambda = 1, moderate stiffness per step
        let n_steps = 10; // total: 10 * 0.001 = 0.01 time units
        let mut y = y0;
        let mut t = 0.0;

        for _ in 0..n_steps {
            let mut k = [[0.0f64; 6]; 6];
            let mut y_next = [0.0f64; 6];
            let mut err = [0.0f64; 6];
            let mut stats = IntegrationStats::default();

            let result = esdirk_step(
                &sys,
                &jac,
                tab,
                t,
                &y,
                h,
                &mut k,
                &mut y_next,
                &mut err,
                1e-10,
                30,
                &mut stats,
                None,
            );
            assert_eq!(result, ImplicitStepResult::Accepted);
            y = y_next;
            t += h;
        }

        // Exact: y(0.01) = exp(-1000 * 0.01) = exp(-10) ≈ 4.54e-5
        // L-stable method should damp; verify the result is stable and small
        assert!(y[0].is_finite(), "Solution not finite: {}", y[0]);
        assert!(
            y[0].abs() < 0.01,
            "L-stable method should strongly damp over 10 steps: got {}",
            y[0]
        );
    }

    #[test]
    fn test_harmonic_convergence_order() {
        // Verify 4th-order convergence by halving step size
        let sys = HarmonicOscillator;
        let jac = HarmonicOscillatorJac;
        let tab = esdirk43_tableau();

        let y0: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let t_final: f64 = 0.5;
        let y_exact = t_final.cos();

        let mut errors = Vec::new();
        let steps_list = [10, 20, 40, 80];

        for &n_steps in &steps_list {
            let h = t_final / f64::from(n_steps);
            let mut y = y0;
            let mut t = 0.0;
            let mut k = [[0.0f64; 6]; 6];
            let mut y_next = [0.0f64; 6];
            let mut err_est = [0.0f64; 6];
            let mut stats = IntegrationStats::default();
            let mut reuse: Option<[f64; 6]> = None;

            for _ in 0..n_steps {
                let result = esdirk_step(
                    &sys,
                    &jac,
                    tab,
                    t,
                    &y,
                    h,
                    &mut k,
                    &mut y_next,
                    &mut err_est,
                    1e-14,
                    50,
                    &mut stats,
                    reuse.as_ref(),
                );
                assert_eq!(result, ImplicitStepResult::Accepted);

                // FSAL: k[5] can be reused as k[0] for next step
                reuse = Some(k[5]);
                y = y_next;
                t += h;
            }

            let error = (y[0] - y_exact).abs();
            errors.push(error);
        }

        // Check convergence rates between successive refinements
        // For order-4 method with step halving, error should decrease by factor ~16
        for (error_pair, step_pair) in errors.windows(2).zip(steps_list.windows(2)) {
            let [previous_error, current_error] = error_pair else {
                continue;
            };
            let [previous_steps, current_steps] = step_pair else {
                continue;
            };
            let ratio = previous_error / current_error;
            // Allow some tolerance: expect ratio > 12 for order 4 (theoretical is 16)
            assert!(
                ratio > 12.0,
                "Convergence ratio {previous_steps}/{current_steps} = {ratio:.2} (expected ~16 for order 4)"
            );
        }
    }

    #[test]
    fn test_fsal_property() {
        // Verify that k[5] from one step equals k[0] computed fresh at the next step
        let sys = HarmonicOscillator;
        let jac = HarmonicOscillatorJac;
        let tab = esdirk43_tableau();

        let y0: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let h = 0.1;
        let mut k = [[0.0f64; 6]; 6];
        let mut y_next = [0.0f64; 6];
        let mut err = [0.0f64; 6];
        let mut stats = IntegrationStats::default();

        // Step 1: no FSAL reuse
        let result = esdirk_step(
            &sys,
            &jac,
            tab,
            0.0,
            &y0,
            h,
            &mut k,
            &mut y_next,
            &mut err,
            1e-14,
            50,
            &mut stats,
            None,
        );
        assert_eq!(result, ImplicitStepResult::Accepted);

        let k5_from_step1 = k[5];

        // Compute f(t+h, y_next) directly
        let mut f_direct = [0.0f64; 6];
        sys.rhs(h, &y_next, &mut f_direct);

        // k[5] should equal f(t+h, y_next) because the method is stiffly accurate
        for (index, (&stage_derivative, &direct_derivative)) in
            k5_from_step1.iter().zip(f_direct.iter()).enumerate()
        {
            let difference = (stage_derivative - direct_derivative).abs();
            assert!(
                difference < 1e-10,
                "FSAL check: k[5][{index}] = {stage_derivative}, f_direct[{index}] = {direct_derivative} (diff {difference:e})"
            );
        }
    }

    #[test]
    fn test_backward_integration() {
        let sys = HarmonicOscillator;
        let jac = HarmonicOscillatorJac;
        let tab = esdirk43_tableau();

        // Start at t=0.5, integrate backward to t=0
        let t_start: f64 = 0.5;
        let y0: [f64; 6] = [t_start.cos(), 0.0, 0.0, -t_start.sin(), 0.0, 0.0];
        let h = -0.05; // negative step for backward integration
        let n_steps = 10;

        let mut y = y0;
        let mut t = t_start;
        let mut k = [[0.0f64; 6]; 6];
        let mut y_next = [0.0f64; 6];
        let mut err = [0.0f64; 6];
        let mut stats = IntegrationStats::default();

        for _ in 0..n_steps {
            let result = esdirk_step(
                &sys,
                &jac,
                tab,
                t,
                &y,
                h,
                &mut k,
                &mut y_next,
                &mut err,
                1e-14,
                50,
                &mut stats,
                None,
            );
            assert_eq!(result, ImplicitStepResult::Accepted);
            y = y_next;
            t += h;
        }

        // Should arrive back near y(0) = [1, 0, 0, 0, 0, 0]
        assert!(
            (y[0] - 1.0).abs() < 1e-6,
            "Backward integration: y[0] = {} (expected 1.0)",
            y[0]
        );
        assert!(
            y[3].abs() < 1e-6,
            "Backward integration: y[3] = {} (expected 0.0)",
            y[3]
        );
    }
}
