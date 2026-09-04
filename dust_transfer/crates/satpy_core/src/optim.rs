use num_traits::{Float, FromPrimitive};

/// Result from Newton's method for 1D root/optimization.
#[derive(Debug, Clone, Copy)]
pub struct Newton1dResult<T: Float> {
    pub x: T,
    pub fx: T,
    pub iterations: usize,
    pub converged: bool,
}

/// Newton's method for 1D optimization: find x where f'(x) = 0.
///
/// **Differential oracle, not dead code.** Nothing in the workspace calls this
/// outside `test_brent_matches_newton` below, which is the point: production
/// minimizes through `minimize_scalar_bounded` (Brent), and this is the
/// independent second method that check compares it against. Deleting it for
/// having no production caller converts that cross-method agreement test into
/// nothing. See `docs/REFACTOR_BLOCKLIST.md`, "Only an inline test calls it is
/// not a deadness finding".
#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic floating optimizer preserves established Newton step ordering"
)]
pub fn newton_1d_minimize<T, F>(
    f_df: F,
    x0: T,
    bounds: Option<(T, T)>,
    tol: T,
    max_iter: usize,
) -> Newton1dResult<T>
where
    T: Float + FromPrimitive,
    F: Fn(T) -> (T, T), // returns (f(x), f'(x))
{
    let (
        Some(default_max_step),
        Some(finite_difference_scale),
        Some(second_derivative_floor),
        Some(gradient_scale),
        Some(half),
    ) = (
        T::from_f64(1e10),
        T::from_f64(1e-8),
        T::from_f64(1e-14),
        T::from_f64(100.0),
        T::from_f64(0.5),
    )
    else {
        return Newton1dResult {
            x: x0,
            fx: T::nan(),
            iterations: 0,
            converged: false,
        };
    };
    let mut x = x0;
    let mut fx;
    let mut dfx;

    // Get initial values
    let (f0, df0) = f_df(x);
    fx = f0;
    dfx = df0;

    // Compute max step based on bounds or use large default
    let max_step = bounds.map_or(default_max_step, |(a, b)| (b - a).abs());

    for iter in 0..max_iter {
        // Check convergence (derivative near zero means at extremum)
        if dfx.abs() < tol {
            return Newton1dResult {
                x,
                fx,
                iterations: iter,
                converged: true,
            };
        }

        // Approximate second derivative via finite difference of first derivative
        let h = finite_difference_scale * x.abs().max(finite_difference_scale);
        let (_, df_plus) = f_df(x + h);
        let second_derivative = (df_plus - dfx) / h;

        // Avoid division by zero or tiny second derivatives
        if second_derivative.abs() < second_derivative_floor {
            // Fall back to gradient descent step
            let step = -dfx.signum() * h * gradient_scale;
            x = x + step;
        } else {
            // Newton step
            let step = -dfx / second_derivative;
            // Limit step size to avoid overshooting (use half the range if bounded)
            let step_limit = max_step * half;
            let clamped_step = step.clamp(-step_limit, step_limit);
            x = x + clamped_step;
        }

        // Apply bounds
        if let Some((lo, hi)) = bounds {
            x = x.clamp(lo, hi);
        }

        // Re-evaluate
        let (f_new, df_new) = f_df(x);
        fx = f_new;
        dfx = df_new;
    }

    Newton1dResult {
        x,
        fx,
        iterations: max_iter,
        converged: false,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MinimizeResult<T: Float> {
    pub x: T,
    pub fx: T,
    pub iterations: usize,
    pub func_evals: usize,
    pub status: i32,
    pub converged: bool,
}

const STATUS_EVALUATION_COUNTER_OVERFLOW: i32 = -2;

#[inline]
#[must_use]
const fn reserve_function_evaluation(evaluations: &mut usize) -> bool {
    let Some(next_evaluations) = evaluations.checked_add(1) else {
        return false;
    };
    *evaluations = next_evaluations;
    true
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "generic Float formula uses IEEE finite/NaN semantics and preserves Brent operation order; integer evaluation accounting uses checked_add"
)]
/// Minimize a fallible scalar objective over a bounded interval.
///
/// # Errors
///
/// Returns the first error produced by `objective` without evaluating it again.
pub fn minimize_scalar_bounded<T, F, E>(
    mut objective: F,
    mut lower_bound: T,
    mut upper_bound: T,
    xatol: T,
    maxiter: usize,
) -> Result<MinimizeResult<T>, E>
where
    T: Float + FromPrimitive,
    F: FnMut(T) -> Result<T, E>,
{
    let mut result = MinimizeResult {
        x: T::nan(),
        fx: T::nan(),
        iterations: 0,
        func_evals: 0,
        status: -1,
        converged: false,
    };

    if !lower_bound.is_finite()
        || !upper_bound.is_finite()
        || lower_bound == upper_bound
        || !xatol.is_finite()
        || xatol <= T::zero()
        || maxiter < 1
    {
        return Ok(result);
    }

    if lower_bound > upper_bound {
        std::mem::swap(&mut lower_bound, &mut upper_bound);
    }

    let (Some(cgold), Some(sqrt_eps), Some(half), Some(two)) = (
        T::from_f64(0.381_966_011_250_105_1),
        T::from_f64(1.490_116_119_384_765_6e-08),
        T::from_f64(0.5),
        T::from_f64(2.0),
    ) else {
        return Ok(result);
    };

    let mut current = lower_bound + cgold * (upper_bound - lower_bound);
    let mut previous = current;
    let mut older = current;

    if !reserve_function_evaluation(&mut result.func_evals) {
        result.status = STATUS_EVALUATION_COUNTER_OVERFLOW;
        return Ok(result);
    }
    let mut current_value = objective(current)?;
    let mut previous_value = current_value;
    let mut older_value = current_value;

    let mut last_step = T::zero();
    let mut trial_step = T::zero();

    for iter in 1..=maxiter {
        result.iterations = iter;
        let midpoint = half * (lower_bound + upper_bound);
        let tol1 = xatol + sqrt_eps * current.abs();
        let tol2 = two * tol1;

        if (current - midpoint).abs() <= (tol2 - half * (upper_bound - lower_bound)) {
            result.x = current;
            result.fx = current_value;
            result.status = 0;
            result.converged = true;
            return Ok(result);
        }

        let mut numerator;
        let mut denominator;
        let mut accept_parabolic = false;

        if current != previous
            && current != older
            && previous != older
            && current_value.is_finite()
            && previous_value.is_finite()
            && older_value.is_finite()
        {
            let r_fit = (current - previous) * (current_value - older_value);
            let q_fit = (current - older) * (current_value - previous_value);
            numerator = (current - older) * q_fit - (current - previous) * r_fit;
            denominator = two * (q_fit - r_fit);
            if denominator > T::zero() {
                numerator = -numerator;
            } else {
                denominator = -denominator;
            }

            if denominator > T::zero() {
                let candidate = current + numerator / denominator;
                if candidate > (lower_bound + tol1)
                    && candidate < (upper_bound - tol1)
                    && numerator.abs() < half * denominator * last_step.abs()
                {
                    trial_step = numerator / denominator;
                    accept_parabolic = true;
                }
            }
        }

        if !accept_parabolic {
            last_step = if current >= midpoint {
                lower_bound - current
            } else {
                upper_bound - current
            };
            trial_step = cgold * last_step;
        }

        let step = if trial_step.abs() >= tol1 {
            trial_step
        } else if trial_step >= T::zero() {
            tol1
        } else {
            -tol1
        };
        let mut candidate = current + step;
        if candidate <= lower_bound {
            candidate = lower_bound + tol1;
        }
        if candidate >= upper_bound {
            candidate = upper_bound - tol1;
        }

        if !reserve_function_evaluation(&mut result.func_evals) {
            result.x = current;
            result.fx = current_value;
            result.status = STATUS_EVALUATION_COUNTER_OVERFLOW;
            return Ok(result);
        }
        let candidate_value = objective(candidate)?;

        if candidate_value <= current_value {
            if candidate >= current {
                lower_bound = current;
            } else {
                upper_bound = current;
            }
            older = previous;
            older_value = previous_value;
            previous = current;
            previous_value = current_value;
            current = candidate;
            current_value = candidate_value;
        } else {
            if candidate < current {
                lower_bound = candidate;
            } else {
                upper_bound = candidate;
            }
            if candidate_value <= previous_value || previous == current {
                older = previous;
                older_value = previous_value;
                previous = candidate;
                previous_value = candidate_value;
            } else if candidate_value <= older_value || older == current || older == previous {
                older = candidate;
                older_value = candidate_value;
            }
        }
        last_step = trial_step;
    }

    result.x = current;
    result.fx = current_value;
    result.status = 1;
    result.converged = false;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    fn unwrap_infallible<T>(result: Result<T, Infallible>) -> T {
        match result {
            Ok(value) => value,
            Err(never) => match never {},
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum ObjectiveError {
        Rejected,
    }

    #[test]
    fn brent_propagates_objective_error() {
        let mut calls = 0usize;
        let result = minimize_scalar_bounded(
            |_x: f64| {
                calls = calls.saturating_add(1);
                Err(ObjectiveError::Rejected)
            },
            0.0,
            2.0,
            1e-8,
            50,
        );

        match result {
            Err(error) => assert_eq!(error, ObjectiveError::Rejected),
            Ok(_) => panic!("objective error must propagate"),
        }
        assert_eq!(calls, 1);
    }

    #[test]
    fn test_newton_1d_quadratic() {
        // Minimize f(x) = (x - 3)^2
        // f'(x) = 2(x - 3)
        // Minimum at x = 3
        let result = newton_1d_minimize(
            |x: f64| {
                let f = (x - 3.0).powi(2);
                let df = 2.0 * (x - 3.0);
                (f, df)
            },
            0.0, // Start far from minimum
            None,
            1e-10,
            50,
        );

        assert!(
            result.converged,
            "Did not converge after {} iterations",
            result.iterations
        );
        assert!(
            (result.x - 3.0).abs() < 1e-6,
            "x = {}, expected 3.0",
            result.x
        );
        assert!(result.fx < 1e-10, "fx = {}", result.fx);
        // Newton should converge quickly for quadratics
        assert!(
            result.iterations < 20,
            "Took {} iterations",
            result.iterations
        );
    }

    #[test]
    fn test_newton_1d_with_bounds() {
        // Minimize f(x) = x^2, but with bounds [1, 5]
        // True minimum at x = 0, but bounded minimum at x = 1
        let result =
            newton_1d_minimize(|x: f64| (x * x, 2.0 * x), 3.0, Some((1.0, 5.0)), 1e-10, 20);

        assert!((result.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_newton_1d_sin() {
        // Minimize f(x) = sin(x) on [0, 2*PI]
        // Minimum at x = 3*PI/2
        let result = newton_1d_minimize(
            |x: f64| (x.sin(), x.cos()),
            4.0, // Start near minimum
            Some((0.0, std::f64::consts::TAU)),
            1e-10,
            20,
        );

        let expected = 1.5 * std::f64::consts::PI;
        assert!(
            (result.x - expected).abs() < 1e-4,
            "x = {}, expected {}",
            result.x,
            expected
        );
    }

    #[test]
    fn test_brent_matches_newton() {
        // Both methods should find same minimum for smooth function
        let f = |x: f64| Ok::<f64, Infallible>((x - 2.5).powi(2) + 0.1 * x.sin());
        let f_df = |x: f64| {
            let fx = (x - 2.5).powi(2) + 0.1 * x.sin();
            let dfx = 2.0 * (x - 2.5) + 0.1 * x.cos();
            (fx, dfx)
        };

        let brent = unwrap_infallible(minimize_scalar_bounded(f, 0.0, 5.0, 1e-8, 50));
        let newton = newton_1d_minimize(f_df, 2.0, Some((0.0, 5.0)), 1e-8, 20);

        assert!(
            (brent.x - newton.x).abs() < 1e-4,
            "Brent x = {}, Newton x = {}",
            brent.x,
            newton.x
        );
    }

    #[test]
    fn test_brent_accepts_mutable_objective_state() {
        let mut evals = 0usize;
        let result = unwrap_infallible(minimize_scalar_bounded(
            |x: f64| {
                evals += 1;
                Ok::<f64, Infallible>((x - 1.25).powi(2))
            },
            0.0,
            2.0,
            1e-8,
            50,
        ));

        assert!(result.converged);
        assert!(evals > 0);
        assert_eq!(evals, result.func_evals);
    }

    #[test]
    fn function_evaluation_counter_rejects_usize_max() {
        let mut evaluations = usize::MAX;

        assert!(!reserve_function_evaluation(&mut evaluations));
        assert_eq!(evaluations, usize::MAX);
    }

    #[test]
    fn brent_accepts_usize_max_iterations_without_counter_wrap() {
        let result = unwrap_infallible(minimize_scalar_bounded(
            |x: f64| Ok::<f64, Infallible>((x - 1.0).powi(2)),
            0.0,
            2.0,
            1e-8,
            usize::MAX,
        ));

        assert!(result.converged);
        assert_ne!(result.status, STATUS_EVALUATION_COUNTER_OVERFLOW);
        assert!(result.func_evals > 0);
    }
}
