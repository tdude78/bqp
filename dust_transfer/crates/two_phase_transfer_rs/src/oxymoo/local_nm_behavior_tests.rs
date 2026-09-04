//! Behavior checks for the `oxymoo` LIBRARY Nelder-Mead local optimizer as
//! consumed by this crate (formerly misnamed `test_handrolled_nm.rs`;
//! nothing here is hand-rolled).
//!
//! It lived in `tests/` while `oxymoo` was a separate crate. `oxymoo` is a
//! private module here now and `LocalOptimizerConfig`, `LocalScalarProblem3`,
//! `run_local_optimizer` and `DEFAULT_NM_MIN_ITERS` are not re-exported, so the
//! checks moved inline rather than the surface being republished for them.

use crate::oxymoo::local::{
    run_local_optimizer, LocalOptimizerConfig, LocalOptimizerKind, LocalScalarProblem3, TuneLevel,
};

struct Rosenbrock;

impl LocalScalarProblem3 for Rosenbrock {
    fn value(&self, x: &[f64; 3]) -> anyhow::Result<f64> {
        let [x0, x1, x2] = *x;
        let a = 1.0;
        let b = 100.0;
        let first = (a - x0).powi(2);
        let x0_squared = x0.powi(2);
        let x1_minus_x0_squared = x1 - x0_squared;
        let second = b * x1_minus_x0_squared.powi(2);
        let third = (a - x1).powi(2);
        let x1_squared = x1.powi(2);
        let x2_minus_x1_squared = x2 - x1_squared;
        let fourth = b * x2_minus_x1_squared.powi(2);
        let first_two = first + second;
        let first_three = first_two + third;
        Ok(first_three + fourth)
    }
}

struct NoisyQuadratic;

impl LocalScalarProblem3 for NoisyQuadratic {
    fn value(&self, x: &[f64; 3]) -> anyhow::Result<f64> {
        let [x0, x1, x2] = *x;
        let first = (x0 - 1.5).powi(2);
        let second = 2.0 * (x1 + 0.5).powi(2);
        let third = 0.5 * (x2 - 0.3).powi(2);
        let base = (first + second) + third;
        let first_angle_term = x0 * 31.7;
        let second_angle_term = x1 * 17.3;
        let third_angle_term = x2 * 7.1;
        let first_two_angle_terms = first_angle_term + second_angle_term;
        let noise = 1e-4 * (first_two_angle_terms + third_angle_term).sin();
        Ok(base + noise)
    }
}

struct BoundaryQuadratic;

impl LocalScalarProblem3 for BoundaryQuadratic {
    fn value(&self, x: &[f64; 3]) -> anyhow::Result<f64> {
        let [x0, x1, x2] = *x;
        let first = (x0 - 3.0).powi(2);
        let second = (x1 - 3.0).powi(2);
        let third = (x2 - 3.0).powi(2);
        Ok((first + second) + third)
    }
}

const fn nm_config(max_iters: usize) -> LocalOptimizerConfig {
    LocalOptimizerConfig {
        kind: LocalOptimizerKind::NelderMead,
        max_iters,
        tolerance: 1e-8,
        seed: 7,
        tune: TuneLevel::Conservative,
        min_iters: crate::oxymoo::DEFAULT_NM_MIN_ITERS,
    }
}

#[test]
fn oxymoo_nelder_mead_rosenbrock_converges() -> anyhow::Result<()> {
    let result = run_local_optimizer(
        &Rosenbrock,
        [-5.0; 3],
        [5.0; 3],
        [0.0, 0.0, 0.0],
        nm_config(2000),
    )?;
    anyhow::ensure!(result.cost < 1e-4, "{result:?}");
    for value in result.x {
        anyhow::ensure!((value - 1.0).abs() < 0.05, "{result:?}");
    }
    Ok(())
}

#[test]
fn oxymoo_nelder_mead_noisy_quadratic_finds_minimum() -> anyhow::Result<()> {
    let result = run_local_optimizer(
        &NoisyQuadratic,
        [-5.0; 3],
        [5.0; 3],
        [0.0, 0.0, 0.0],
        nm_config(1000),
    )?;
    anyhow::ensure!(result.cost < 0.01, "{result:?}");
    let [x0, x1, x2] = result.x;
    anyhow::ensure!((x0 - 1.5).abs() < 0.1, "{result:?}");
    anyhow::ensure!((x1 + 0.5).abs() < 0.1, "{result:?}");
    anyhow::ensure!((x2 - 0.3).abs() < 0.1, "{result:?}");
    Ok(())
}

#[test]
fn oxymoo_nelder_mead_bounds_respected() -> anyhow::Result<()> {
    let result = run_local_optimizer(
        &BoundaryQuadratic,
        [-1.0; 3],
        [1.0; 3],
        [0.0; 3],
        nm_config(500),
    )?;
    for value in result.x {
        anyhow::ensure!((-1.0 - 1e-9..=1.0 + 1e-9).contains(&value), "{result:?}");
        anyhow::ensure!((value - 1.0).abs() < 0.05, "{result:?}");
    }
    Ok(())
}

#[test]
fn oxymoo_nelder_mead_public_api_improves_initial_cost() -> anyhow::Result<()> {
    let initial = [0.5, 0.5, 0.5];
    let result = run_local_optimizer(&Rosenbrock, [-5.0; 3], [5.0; 3], initial, nm_config(500))?;
    anyhow::ensure!(result.cost < Rosenbrock.value(&initial)?, "{result:?}");
    Ok(())
}
