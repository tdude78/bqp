//! Behaviour of the local scalar optimizers, moved inline when the `oxymoo`
//! crate was absorbed into this one: it reaches `run_nelder_mead3`, which is
//! not re-exported even from the module root, so it cannot run from `tests/`.

use crate::oxymoo::local::{
    run_local_optimizer, run_nelder_mead3, LocalOptimizerConfig, LocalOptimizerKind,
    LocalScalarProblem3, TuneLevel,
};
use anyhow::Result;

struct Quadratic;

fn quadratic_cost(x: &[f64; 3]) -> f64 {
    let first_delta = x[0] - 0.25;
    let second_delta = x[1] - 0.5;
    let third_delta = x[2] - 0.75;
    let first_term = first_delta.powi(2);
    let second_term = second_delta.powi(2);
    let third_term = third_delta.powi(2);
    first_term + second_term + third_term
}

impl LocalScalarProblem3 for Quadratic {
    fn value(&self, x: &[f64; 3]) -> Result<f64> {
        Ok(quadratic_cost(x))
    }

    fn value_gradient(&self, x: &[f64; 3]) -> Option<(f64, [f64; 3])> {
        Some((
            quadratic_cost(x),
            [2.0 * (x[0] - 0.25), 2.0 * (x[1] - 0.5), 2.0 * (x[2] - 0.75)],
        ))
    }
}

struct Rosenbrock;

fn rosenbrock_cost(x: &[f64; 3]) -> f64 {
    let first_delta = 1.0 - x[0];
    let first_term = first_delta.powi(2);
    let first_square = x[0] * x[0];
    let second_delta = x[1] - first_square;
    let second_term = 100.0 * second_delta.powi(2);
    let third_delta = 1.0 - x[1];
    let third_term = third_delta.powi(2);
    let second_square = x[1] * x[1];
    let fourth_delta = x[2] - second_square;
    let fourth_term = 100.0 * fourth_delta.powi(2);
    first_term + second_term + third_term + fourth_term
}

impl LocalScalarProblem3 for Rosenbrock {
    fn value(&self, x: &[f64; 3]) -> Result<f64> {
        Ok(rosenbrock_cost(x))
    }
}

struct NonFiniteProblem;

impl LocalScalarProblem3 for NonFiniteProblem {
    fn value(&self, _x: &[f64; 3]) -> Result<f64> {
        Ok(f64::NAN)
    }

    fn value_gradient(&self, _x: &[f64; 3]) -> Option<(f64, [f64; 3])> {
        Some((f64::NAN, [0.0; 3]))
    }
}

const fn config(kind: LocalOptimizerKind) -> LocalOptimizerConfig {
    LocalOptimizerConfig {
        kind,
        max_iters: 600,
        tolerance: 1e-7,
        seed: 42,
        tune: TuneLevel::Conservative,
        min_iters: crate::oxymoo::DEFAULT_NM_MIN_ITERS,
    }
}

#[test]
fn local_optimizer_configs_have_stable_names() {
    assert_eq!(LocalOptimizerKind::NelderMead.name(), "nelder_mead");
    assert_eq!(LocalOptimizerKind::Pso.name(), "pso");
    assert_eq!(LocalOptimizerKind::Lbfgs.name(), "lbfgs");
    assert_eq!(TuneLevel::Default.name(), "default");
    assert_eq!(TuneLevel::Conservative.name(), "conservative");
    assert_eq!(TuneLevel::Aggressive.name(), "aggressive");
}

#[test]
fn nelder_mead3_solves_bounded_quadratic() {
    let result = run_local_optimizer(
        &Quadratic,
        [-1.0; 3],
        [2.0; 3],
        [0.0, 0.0, 0.0],
        config(LocalOptimizerKind::NelderMead),
    )
    .unwrap();

    assert!(result.cost < 1e-5, "{result:?}");
    assert!(result.x.iter().all(|value| (-1.0..=2.0).contains(value)));
}

#[test]
fn direct_nelder_mead3_clamps_out_of_bounds_initial() {
    let result = run_nelder_mead3(
        &Quadratic,
        [0.0; 3],
        [1.0; 3],
        [-10.0, 0.5, 12.0],
        config(LocalOptimizerKind::NelderMead),
    )
    .unwrap();

    assert!(result.x.iter().all(|value| (0.0..=1.0).contains(value)));
    assert!(result.cost < 1e-4, "{result:?}");
}

#[test]
fn pso3_is_seed_deterministic_and_respects_bounds() {
    let cfg = LocalOptimizerConfig {
        max_iters: 80,
        ..config(LocalOptimizerKind::Pso)
    };
    let first = run_local_optimizer(&Quadratic, [0.0; 3], [1.0; 3], [0.1; 3], cfg).unwrap();
    let second = run_local_optimizer(&Quadratic, [0.0; 3], [1.0; 3], [0.1; 3], cfg).unwrap();

    assert_eq!(first.x.map(f64::to_bits), second.x.map(f64::to_bits));
    assert_eq!(first.cost.to_bits(), second.cost.to_bits());
    assert!(first.cost < 1e-3, "{first:?}");
    assert!(first.x.iter().all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn lbfgs3_requires_gradient_and_uses_it_when_available() {
    let result = run_local_optimizer(
        &Quadratic,
        [0.0; 3],
        [1.0; 3],
        [0.9, 0.1, 0.2],
        config(LocalOptimizerKind::Lbfgs),
    )
    .unwrap();

    assert!(result.cost < 1e-8, "{result:?}");

    let err = run_local_optimizer(
        &Rosenbrock,
        [-5.0; 3],
        [5.0; 3],
        [0.5; 3],
        config(LocalOptimizerKind::Lbfgs),
    )
    .unwrap_err();

    assert_eq!(err.to_string(), "local optimizer gradient is unavailable");
}

#[test]
fn local_optimizers_fail_loud_on_non_finite_objectives() {
    for kind in [
        LocalOptimizerKind::NelderMead,
        LocalOptimizerKind::Pso,
        LocalOptimizerKind::Lbfgs,
    ] {
        let err = run_local_optimizer(
            &NonFiniteProblem,
            [0.0; 3],
            [1.0; 3],
            [0.5; 3],
            config(kind),
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "local optimizer objective returned a non-finite value"
        );
    }
}

/// The Nelder-Mead iteration budget is `max_iters * iters_factor` raised to
/// `min_iters`, and at `TuneLevel::Default` that floor is what runs for every
/// request below 34. Callers have read `max_iters` as the iteration count and
/// been wrong by up to 10x, so pin the arithmetic.
///
/// Counted through the objective: with the simplex tolerance unreachable the
/// loop never exits early, so evaluations are a strictly increasing function of
/// the effective iteration count and equal across configs that produce the same
/// one.
#[test]
fn nelder_mead_iteration_budget_is_the_floor_until_the_request_beats_it() {
    struct Counting {
        calls: std::cell::Cell<usize>,
    }

    impl LocalScalarProblem3 for Counting {
        fn value(&self, x: &[f64; 3]) -> Result<f64> {
            self.calls.set(self.calls.get() + 1);
            // Steep and asymmetric: the simplex keeps moving, so the sd
            // tolerance is not what ends the loop.
            let first_delta = x[0] - 0.9;
            let first_term = first_delta.powi(2) * 100.0;
            let second_delta = x[1] + 3.0;
            let second_term = second_delta.powi(2);
            let third_term = (x[2] - 0.9).abs();
            Ok(first_term + second_term + third_term)
        }
    }

    let evals = |max_iters: usize, min_iters: usize| {
        let problem = Counting {
            calls: std::cell::Cell::new(0),
        };
        run_nelder_mead3(
            &problem,
            [0.0; 3],
            [1.0; 3],
            [0.1; 3],
            LocalOptimizerConfig {
                kind: LocalOptimizerKind::NelderMead,
                max_iters,
                tolerance: 1e-12,
                seed: 0,
                tune: TuneLevel::Default,
                min_iters,
            },
        )
        .unwrap();
        problem.calls.get()
    };

    // 0.3 * 33 = 9.9 -> floored to 10; 0.3 * 34 = 10.2 -> 10 as well. The whole
    // range 1..=34 is one budget at the default floor.
    let floored = evals(12, crate::oxymoo::DEFAULT_NM_MIN_ITERS);
    for request in [1_usize, 4, 8, 12, 33, 34] {
        assert_eq!(
            evals(request, crate::oxymoo::DEFAULT_NM_MIN_ITERS),
            floored,
            "max_iters {request} did not collapse onto the {} floor",
            crate::oxymoo::DEFAULT_NM_MIN_ITERS
        );
    }
    // 0.3 * 40 = 12 clears the floor and costs more.
    assert!(
        evals(40, crate::oxymoo::DEFAULT_NM_MIN_ITERS) > floored,
        "max_iters 40 should reach 12 iterations and outspend the floor"
    );

    // Pinning both bounds to the same value makes the count exact, and fewer
    // iterations then really do cost less.
    let exact = |n: usize| evals(n, n);
    assert_eq!(
        exact(10),
        floored,
        "min_iters = max_iters = 10 is the floor"
    );
    for smaller in [1_usize, 2, 5, 9] {
        assert!(
            exact(smaller) < floored,
            "an exact budget of {smaller} should be cheaper than the floor of 10"
        );
    }
}
