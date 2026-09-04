//! NSGA-II kernel behaviour, moved inline when the `oxymoo` crate was absorbed
//! into this one: everything it names is private to `crate::oxymoo` now, so a
//! `tests/` binary cannot reach it without republishing the whole surface.

use crate::oxymoo::{
    crowding_distance, fast_nondominated_sort, ArithmeticOverflow, Nsga2, Nsga2Config,
    PopulationSnapshot, Problem, SortConfig, VariableKind, VariableSpec,
};
use anyhow::anyhow;
use approx::relative_eq;

#[test]
fn constrained_sort_places_feasible_fronts_before_infeasible_rows() -> anyhow::Result<()> {
    let objectives = vec![
        0.0, 0.0, //
        1.0, 1.0, //
        0.5, 0.5, //
        0.0, 2.0, //
    ];
    let cv = vec![0.0, 0.0, 0.25, 0.0];

    let fronts = fast_nondominated_sort(&objectives, 4, 2, &cv, SortConfig::default())?;

    ensure_equal(
        &fronts,
        &vec![vec![0], vec![1, 3], vec![2]],
        "feasibility-first fronts differ",
    )?;
    Ok(())
}

#[test]
fn constrained_sort_is_feasibility_first_for_single_objective_rows() -> anyhow::Result<()> {
    let objectives = vec![
        3.0,    //
        5.0,    //
        -100.0, //
        4.0,    //
        -200.0, //
    ];
    let cv = vec![0.0, 0.0, 0.10, 0.0, 0.01];

    let fronts = fast_nondominated_sort(&objectives, 5, 1, &cv, SortConfig::default())?;

    ensure_equal(
        &fronts,
        &vec![vec![0], vec![3], vec![1], vec![4], vec![2]],
        "single-objective feasibility-first fronts differ",
    )?;
    Ok(())
}

#[test]
fn crowding_distance_marks_boundary_points_and_normalizes_interior_distance() -> anyhow::Result<()>
{
    let objectives = vec![
        0.0, 1.0, //
        0.5, 0.5, //
        1.0, 0.0, //
    ];
    let front = vec![0, 1, 2];

    let distances = crowding_distance(&objectives, 3, 2, &front)?;

    ensure_true(
        value_at(&distances, 0, "first crowding distance")?.is_infinite(),
        "first crowding distance is not infinite",
    )?;
    ensure_relative_equal(
        value_at(&distances, 1, "middle crowding distance")?,
        2.0,
        1e-12,
        "middle crowding distance differs",
    )?;
    ensure_true(
        value_at(&distances, 2, "last crowding distance")?.is_infinite(),
        "last crowding distance is not infinite",
    )?;
    Ok(())
}

#[derive(Clone)]
struct MixedTradeoffProblem {
    variables: Vec<VariableSpec>,
}

impl MixedTradeoffProblem {
    fn new() -> Self {
        Self {
            variables: vec![
                VariableSpec::new(-2.0, 2.0, VariableKind::Continuous),
                VariableSpec::new(0.0, 4.0, VariableKind::Integer),
            ],
        }
    }
}

impl Problem for MixedTradeoffProblem {
    fn variable_specs(&self) -> &[VariableSpec] {
        &self.variables
    }

    fn objective_count(&self) -> usize {
        2
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "the deterministic NSGA-II fixture preserves its established floating-point expression order"
    )]
    fn evaluate(&self, decision: &[f64], objectives: &mut [f64]) -> anyhow::Result<f64> {
        let decision_pair: &[f64; 2] = decision
            .try_into()
            .map_err(|_| anyhow!("mixed-tradeoff decision must have two coordinates"))?;
        let [continuous, integer] = *decision_pair;
        let objective_pair: &mut [f64; 2] = objectives
            .try_into()
            .map_err(|_| anyhow!("mixed-tradeoff objective row must have two coordinates"))?;
        let [first_objective, second_objective] = objective_pair;
        let integer = integer.round();
        *first_objective = (continuous - 0.2).powi(2) + 0.05 * integer;
        *second_objective = (continuous + 0.8).powi(2) + 0.05 * (4.0 - integer);
        Ok(0.0)
    }
}

#[test]
fn crowding_distance_supports_single_objective_fronts() -> anyhow::Result<()> {
    let objectives = vec![0.0, 0.5, 1.0];
    let front = vec![0, 1, 2];

    let distances = crowding_distance(&objectives, 3, 1, &front)?;

    ensure_true(
        value_at(&distances, 0, "first crowding distance")?.is_infinite(),
        "first crowding distance is not infinite",
    )?;
    ensure_relative_equal(
        value_at(&distances, 1, "middle crowding distance")?,
        1.0,
        1e-12,
        "middle crowding distance differs",
    )?;
    ensure_true(
        value_at(&distances, 2, "last crowding distance")?.is_infinite(),
        "last crowding distance is not infinite",
    )?;
    Ok(())
}

#[test]
fn nsga2_is_seed_deterministic_and_repairs_integer_variables() -> anyhow::Result<()> {
    let config = Nsga2Config {
        population_size: 32,
        generations: 40,
        seed: 7,
        crossover_probability: 0.9,
        mutation_probability: None,
        eta_c: 15.0,
        eta_m: 20.0,
        tournament_size: 2,
        ..Nsga2Config::default()
    };

    let mut first = Nsga2::new(MixedTradeoffProblem::new(), config.clone())?;
    let first_result = first.run()?;
    let mut second = Nsga2::new(MixedTradeoffProblem::new(), config)?;
    let second_result = second.run()?;

    let first_front = first_result
        .fronts
        .first()
        .ok_or_else(|| anyhow!("first NSGA-II result has no front"))?;
    let second_front = second_result
        .fronts
        .first()
        .ok_or_else(|| anyhow!("second NSGA-II result has no front"))?;
    ensure_equal(
        &first_front.len(),
        &second_front.len(),
        "seeded front lengths differ",
    )?;
    ensure_population_trajectory_bits_equal(
        &first_result.population,
        &second_result.population,
        "seeded population differs",
    )?;

    for row in 0..first_result.population.len() {
        let decision = first_result.population.decision(row)?;
        let continuous = decision
            .first()
            .copied()
            .ok_or_else(|| anyhow!("decision row {row} has no continuous coordinate"))?;
        let integer = decision
            .get(1)
            .copied()
            .ok_or_else(|| anyhow!("decision row {row} has no integer coordinate"))?;
        ensure_true(
            (-2.0..=2.0).contains(&continuous),
            "continuous decision is outside its domain",
        )?;
        ensure_true(
            (0.0..=4.0).contains(&integer),
            "integer decision is outside its domain",
        )?;
        ensure_relative_equal(
            integer,
            integer.round(),
            1e-12,
            "integer decision was not repaired to the lattice",
        )?;
    }

    let front0 = first_result
        .fronts
        .first()
        .ok_or_else(|| anyhow!("first NSGA-II result has no front"))?;
    ensure_true(!front0.is_empty(), "first front is empty")?;
    let recomputed_fronts = fast_nondominated_sort(
        &first_result.population.objectives,
        first_result.population.len(),
        first_result.population.objective_count(),
        &first_result.population.constraint_violations,
        SortConfig::default(),
    )?;
    let recomputed_front = recomputed_fronts
        .first()
        .ok_or_else(|| anyhow!("recomputed sort has no front"))?;
    ensure_equal(recomputed_front, front0, "recomputed front differs")?;
    ensure_equal(&first_result.generations, &40, "generation count differs")?;
    ensure_equal(&first_result.evaluations, &1312, "evaluation count differs")?;
    for &row in front0 {
        let rank = first_result
            .population
            .ranks
            .get(row)
            .ok_or_else(|| anyhow!("front row {row} has no rank"))?;
        ensure_equal(rank, &0, "first-front rank differs")?;
    }
    assert_front_is_nondominated(&first_result.population.objectives, front0, 2)?;
    Ok(())
}

#[test]
fn nsga2_stops_before_exceeding_max_evaluations() -> anyhow::Result<()> {
    let config = Nsga2Config {
        population_size: 16,
        generations: 100,
        max_evaluations: Some(64),
        seed: 19,
        ..Nsga2Config::default()
    };

    let mut optimizer = Nsga2::new(MixedTradeoffProblem::new(), config)?;
    let result = optimizer.run()?;

    ensure_equal(&result.generations, &3, "budgeted generation count differs")?;
    ensure_equal(
        &result.evaluations,
        &64,
        "budgeted evaluation count differs",
    )?;
    Ok(())
}

#[test]
fn nsga2_run_owned_with_problem_preserves_result_and_returns_problem() -> anyhow::Result<()> {
    let config = Nsga2Config {
        population_size: 16,
        generations: 8,
        seed: 29,
        ..Nsga2Config::default()
    };

    let reference = Nsga2::new(MixedTradeoffProblem::new(), config.clone())?.run()?;
    let (problem, actual) =
        Nsga2::new(MixedTradeoffProblem::new(), config)?.run_owned_with_problem()?;

    ensure_equal(
        &problem.variable_specs().len(),
        &2,
        "returned problem variable count differs",
    )?;
    ensure_result_trajectory_bits_equal(&actual, &reference, "owned result differs")?;
    Ok(())
}

#[test]
fn invalid_integer_variable_and_too_small_budget_fail_fast() -> anyhow::Result<()> {
    let bad_problem = MixedTradeoffProblem {
        variables: vec![VariableSpec::new(0.2, 0.8, VariableKind::Integer)],
    };
    let err = Nsga2::new(bad_problem, Nsga2Config::default())
        .err()
        .ok_or_else(|| anyhow!("invalid integer variable unexpectedly succeeded"))?;
    ensure_true(
        err.to_string().contains("invalid variable 0"),
        "invalid integer failure lacks its diagnostic",
    )?;

    let err = Nsga2::new(
        MixedTradeoffProblem::new(),
        Nsga2Config {
            population_size: 16,
            max_evaluations: Some(15),
            ..Nsga2Config::default()
        },
    )
    .err()
    .ok_or_else(|| anyhow!("too-small budget unexpectedly succeeded"))?;
    ensure_true(
        err.to_string()
            .contains("max_evaluations 15 is smaller than population_size 16"),
        "too-small budget failure lacks its diagnostic",
    )?;
    Ok(())
}

#[test]
fn nsga2_capacity_overflow_has_typed_cause() -> anyhow::Result<()> {
    let error = Nsga2::new(
        MixedTradeoffProblem::new(),
        Nsga2Config {
            population_size: usize::MAX,
            ..Nsga2Config::default()
        },
    )
    .err()
    .ok_or_else(|| anyhow!("overflowing NSGA-II capacity unexpectedly succeeded"))?;

    ensure_true(
        error.downcast_ref::<ArithmeticOverflow>().is_some(),
        "NSGA-II capacity overflow lacks an ArithmeticOverflow cause",
    )
}

fn assert_front_is_nondominated(
    objectives: &[f64],
    front: &[usize],
    n_objectives: usize,
) -> anyhow::Result<()> {
    for (pos, &left) in front.iter().enumerate() {
        let left_objectives = objective_row(objectives, left, n_objectives)?;
        for &right in front.iter().skip(pos.saturating_add(1)) {
            let right_objectives = objective_row(objectives, right, n_objectives)?;
            let left_dominates = numerical_dominates(left_objectives, right_objectives)?;
            let right_dominates = numerical_dominates(right_objectives, left_objectives)?;
            ensure_true(
                !left_dominates,
                "left front row dominates another front row",
            )?;
            ensure_true(
                !right_dominates,
                "right front row dominates another front row",
            )?;
        }
    }
    Ok(())
}

#[test]
fn nondomination_oracle_treats_signed_zero_as_equal() -> anyhow::Result<()> {
    assert_front_is_nondominated(&[-0.0, 0.0], &[0, 1], 1)
}

#[test]
fn nondomination_oracle_rejects_nan() -> anyhow::Result<()> {
    let error = assert_front_is_nondominated(&[f64::NAN, 0.0], &[0, 1], 1)
        .err()
        .ok_or_else(|| anyhow!("nondomination oracle accepted NaN"))?;
    ensure_true(
        error.to_string().contains("NaN"),
        "nondomination oracle NaN failure lacks its diagnostic",
    )
}

#[test]
fn trajectory_oracle_distinguishes_signed_zero() -> anyhow::Result<()> {
    let error = ensure_f64_bits_equal(&[-0.0], &[0.0], "signed-zero trajectory mismatch")
        .err()
        .ok_or_else(|| anyhow!("trajectory oracle accepted distinct signed-zero bits"))?;
    ensure_true(
        error
            .to_string()
            .contains("signed-zero trajectory mismatch"),
        "trajectory oracle signed-zero failure lacks its diagnostic",
    )
}

fn numerical_dominates(left: &[f64], right: &[f64]) -> anyhow::Result<bool> {
    if left.len() != right.len() {
        return Err(anyhow!("test nondomination rows have inconsistent widths"));
    }

    let mut strictly_better = false;
    for (&left_value, &right_value) in left.iter().zip(right) {
        if left_value.is_nan() || right_value.is_nan() {
            return Err(anyhow!("test nondomination oracle rejects NaN objectives"));
        }
        if !left_value.is_finite() || !right_value.is_finite() {
            return Err(anyhow!(
                "test nondomination oracle requires finite objectives"
            ));
        }
        if left_value > right_value {
            return Ok(false);
        }
        if left_value < right_value {
            strictly_better = true;
        }
    }
    Ok(strictly_better)
}

fn objective_row(objectives: &[f64], row: usize, n_objectives: usize) -> anyhow::Result<&[f64]> {
    if n_objectives == 0 {
        return Err(anyhow!("test objective matrix has zero width"));
    }
    let rows = objectives.chunks_exact(n_objectives);
    if !rows.remainder().is_empty() {
        return Err(anyhow!("test objective matrix has a partial row"));
    }
    objectives
        .chunks_exact(n_objectives)
        .nth(row)
        .ok_or_else(|| anyhow!("test objective row {row} is missing"))
}

fn value_at(values: &[f64], index: usize, label: &str) -> anyhow::Result<f64> {
    values
        .get(index)
        .copied()
        .ok_or_else(|| anyhow!("{label} is missing at index {index}"))
}

fn ensure_true(condition: bool, message: &str) -> anyhow::Result<()> {
    if condition {
        Ok(())
    } else {
        Err(anyhow!("{message}"))
    }
}

fn ensure_equal<T: PartialEq>(actual: &T, expected: &T, message: &str) -> anyhow::Result<()> {
    ensure_true(actual == expected, message)
}

fn ensure_relative_equal(
    actual: f64,
    expected: f64,
    epsilon: f64,
    message: &str,
) -> anyhow::Result<()> {
    ensure_true(relative_eq!(actual, expected, epsilon = epsilon), message)
}

fn ensure_f64_bits_equal(actual: &[f64], expected: &[f64], message: &str) -> anyhow::Result<()> {
    let actual_bits: Vec<u64> = actual.iter().map(|value| value.to_bits()).collect();
    let expected_bits: Vec<u64> = expected.iter().map(|value| value.to_bits()).collect();
    ensure_equal(&actual_bits, &expected_bits, message)
}

fn ensure_population_trajectory_bits_equal(
    actual: &PopulationSnapshot,
    expected: &PopulationSnapshot,
    message: &str,
) -> anyhow::Result<()> {
    ensure_equal(&actual.len(), &expected.len(), message)?;
    ensure_equal(
        &actual.variable_count(),
        &expected.variable_count(),
        message,
    )?;
    ensure_equal(
        &actual.objective_count(),
        &expected.objective_count(),
        message,
    )?;
    ensure_f64_bits_equal(&actual.decisions, &expected.decisions, message)?;
    ensure_f64_bits_equal(&actual.objectives, &expected.objectives, message)?;
    ensure_f64_bits_equal(
        &actual.constraint_violations,
        &expected.constraint_violations,
        message,
    )?;
    ensure_equal(&actual.ranks, &expected.ranks, message)?;
    ensure_f64_bits_equal(&actual.crowding, &expected.crowding, message)
}

fn ensure_result_trajectory_bits_equal(
    actual: &crate::oxymoo::Nsga2Result,
    expected: &crate::oxymoo::Nsga2Result,
    message: &str,
) -> anyhow::Result<()> {
    ensure_population_trajectory_bits_equal(&actual.population, &expected.population, message)?;
    ensure_equal(&actual.fronts, &expected.fronts, message)?;
    ensure_equal(&actual.generations, &expected.generations, message)?;
    ensure_equal(&actual.evaluations, &expected.evaluations, message)
}

struct PreflightProbeProblem {
    variables: Vec<VariableSpec>,
    evaluation_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Problem for PreflightProbeProblem {
    fn variable_specs(&self) -> &[VariableSpec] {
        &self.variables
    }

    fn objective_count(&self) -> usize {
        1
    }

    fn evaluate(&self, _: &[f64], objectives: &mut [f64]) -> anyhow::Result<f64> {
        self.evaluation_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let objective = objectives
            .first_mut()
            .ok_or_else(|| anyhow!("preflight probe objective row is empty"))?;
        *objective = 0.0;
        Ok(0.0)
    }
}

/// Construct with a deliberately invalid input and require the NAMED guard.
///
/// `is_err()` alone would be satisfied by any earlier arm of `Nsga2::new`,
/// which runs `validate_config`, then `validate_initial_decisions`, then
/// `validate_variables` in order. Every caller here poisons one knob and
/// leaves `population_size = 2` -- one above the `< 2` bail -- so a change
/// that moved that floor, or that made the shared `PreflightProbeProblem`
/// itself unconstructible, would turn all six cases green for a reason none
/// of them names. `diagnostic` pins which guard actually spoke.
fn ensure_constructor_fails_before_evaluation(
    config: Nsga2Config,
    variables: Vec<VariableSpec>,
    diagnostic: &str,
    label: &str,
) -> anyhow::Result<()> {
    let evaluation_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let result = Nsga2::new(
        PreflightProbeProblem {
            variables,
            evaluation_calls: std::sync::Arc::clone(&evaluation_calls),
        },
        config,
    );
    let error = result.err().ok_or_else(|| anyhow!("{label}"))?.to_string();
    ensure_true(
        error.contains(diagnostic),
        &format!("{label}: rejected for the wrong reason -- wanted {diagnostic:?}, got {error:?}"),
    )?;
    ensure_equal(
        &evaluation_calls.load(std::sync::atomic::Ordering::Relaxed),
        &0,
        "invalid NSGA-II input evaluated the problem before rejection",
    )
}

/// Positive control for the six rejections above, on both of their claims.
///
/// Without it, a constructor that refused EVERY input -- or a
/// `PreflightProbeProblem` that stopped being valid -- would satisfy all of
/// them. The config here is the same `population_size = 2` shape each poisoned
/// case starts from, with nothing poisoned.
///
/// It also gives the `evaluation_calls == 0` half of those cases something to
/// mean. `Nsga2::new` evaluates the seeded population, so a valid construction
/// moves the counter; if it ever stopped doing so, "rejected before
/// evaluation" would hold for every input including the good ones, and the
/// six rejections would be measuring a counter that no longer counts.
#[test]
fn nsga2_accepts_the_unpoisoned_preflight_shape() -> anyhow::Result<()> {
    let evaluation_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    Nsga2::new(
        PreflightProbeProblem {
            variables: vec![VariableSpec::new(0.0, 1.0, VariableKind::Continuous)],
            evaluation_calls: std::sync::Arc::clone(&evaluation_calls),
        },
        Nsga2Config {
            population_size: 2,
            ..Nsga2Config::default()
        },
    )?;
    ensure_true(
        evaluation_calls.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "a valid NSGA-II construction evaluated nothing, so the rejection \
         cases' evaluation counter cannot discriminate",
    )
}

#[test]
fn nsga2_rejects_nonfinite_or_negative_tolerances_before_evaluation() -> anyhow::Result<()> {
    let variables = vec![VariableSpec::new(0.0, 1.0, VariableKind::Continuous)];
    let cases = [
        (
            "constraint_tolerance must be finite and non-negative: NaN",
            "NaN constraint tolerance unexpectedly succeeded",
            Nsga2Config {
                population_size: 2,
                constraint_tolerance: f64::NAN,
                ..Nsga2Config::default()
            },
        ),
        (
            "constraint_tolerance must be finite and non-negative: -1",
            "negative constraint tolerance unexpectedly succeeded",
            Nsga2Config {
                population_size: 2,
                constraint_tolerance: -1.0,
                ..Nsga2Config::default()
            },
        ),
        (
            "objective_tolerance must be finite and non-negative: inf",
            "infinite objective tolerance unexpectedly succeeded",
            Nsga2Config {
                population_size: 2,
                objective_tolerance: f64::INFINITY,
                ..Nsga2Config::default()
            },
        ),
        (
            "objective_tolerance must be finite and non-negative: -1",
            "negative objective tolerance unexpectedly succeeded",
            Nsga2Config {
                population_size: 2,
                objective_tolerance: -1.0,
                ..Nsga2Config::default()
            },
        ),
    ];

    for (diagnostic, label, config) in cases {
        ensure_constructor_fails_before_evaluation(config, variables.clone(), diagnostic, label)?;
    }
    Ok(())
}

#[test]
fn nsga2_rejects_nonfinite_requested_seed_before_evaluation() -> anyhow::Result<()> {
    let variables = vec![VariableSpec::new(0.0, 1.0, VariableKind::Continuous)];
    let config = Nsga2Config {
        population_size: 2,
        initial_decisions: vec![f64::NAN],
        ..Nsga2Config::default()
    };

    ensure_constructor_fails_before_evaluation(
        config,
        variables,
        "initial decision value at index 0 is non-finite",
        "non-finite requested seed unexpectedly succeeded",
    )
}

#[test]
fn nsga2_rejects_nonfinite_continuous_span_before_evaluation() -> anyhow::Result<()> {
    let variables = vec![VariableSpec::new(
        -f64::MAX,
        f64::MAX,
        VariableKind::Continuous,
    )];
    let config = Nsga2Config {
        population_size: 2,
        ..Nsga2Config::default()
    };

    ensure_constructor_fails_before_evaluation(
        config,
        variables,
        "invalid variable 0",
        "non-finite continuous span unexpectedly succeeded",
    )
}

/// Fixture that counts how many times [`Problem::evaluate`] is invoked, and
/// exposes an evaluation whose raw constraint value can be negative so the
/// default `evaluate_batch` non-negative clamp is exercised.
struct BatchCountingProblem {
    variables: Vec<VariableSpec>,
    evaluate_calls: std::cell::Cell<usize>,
}

impl BatchCountingProblem {
    fn new() -> Self {
        Self {
            variables: vec![
                VariableSpec::new(0.0, 1.0, VariableKind::Continuous),
                VariableSpec::new(0.0, 1.0, VariableKind::Continuous),
            ],
            evaluate_calls: std::cell::Cell::new(0),
        }
    }
}

impl Problem for BatchCountingProblem {
    fn variable_specs(&self) -> &[VariableSpec] {
        &self.variables
    }

    fn objective_count(&self) -> usize {
        2
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "the serial-reference fixture deliberately preserves the evaluator's established floating-point expression order"
    )]
    fn evaluate(&self, decision: &[f64], objectives: &mut [f64]) -> anyhow::Result<f64> {
        let next_calls = self
            .evaluate_calls
            .get()
            .checked_add(1)
            .ok_or_else(|| anyhow!("batch fixture evaluation count overflow"))?;
        self.evaluate_calls.set(next_calls);
        let decision_pair: &[f64; 2] = decision
            .try_into()
            .map_err(|_| anyhow!("batch fixture decision must have two coordinates"))?;
        let [first_decision, second_decision] = *decision_pair;
        let objective_pair: &mut [f64; 2] = objectives
            .try_into()
            .map_err(|_| anyhow!("batch fixture objective row must have two coordinates"))?;
        let [first_objective, second_objective] = objective_pair;
        *first_objective = first_decision * first_decision;
        *second_objective = (1.0 - first_decision).powi(2) + second_decision;
        // Raw constraint value straddles zero so `.max(0.0)` clamping matters.
        Ok(first_decision - 0.5)
    }
}

#[test]
fn default_evaluate_batch_matches_hand_rolled_serial_loop() -> anyhow::Result<()> {
    let n_variables = 2;
    let n_objectives = 2;
    let decisions = vec![
        0.10, 0.20, //
        0.70, 0.90, //
        0.50, 0.00, //
        0.90, 0.40, //
        0.30, 0.65, //
    ];
    let decision_rows = decisions.chunks_exact(n_variables);
    if !decision_rows.remainder().is_empty() {
        return Err(anyhow!("serial-reference decisions have a partial row"));
    }
    let n_individuals = decision_rows.len();

    // Hand-rolled serial reference: same iteration order, same clamp.
    let reference = BatchCountingProblem::new();
    let mut expected_objectives = vec![0.0; decisions.len()];
    let mut expected_cv = vec![0.0; n_individuals];
    for ((decision, objective_row), cv_slot) in decisions
        .chunks_exact(n_variables)
        .zip(expected_objectives.chunks_exact_mut(n_objectives))
        .zip(expected_cv.iter_mut())
    {
        let cv = reference.evaluate(decision, objective_row)?;
        *cv_slot = cv.max(0.0);
    }
    let reference_calls = reference.evaluate_calls.get();

    // Default `evaluate_batch` over the same decisions.
    let batched = BatchCountingProblem::new();
    let mut batch_objectives = vec![0.0; decisions.len()];
    let mut batch_cv = vec![0.0; n_individuals];
    batched.evaluate_batch(
        &decisions,
        n_variables,
        n_objectives,
        &mut batch_objectives,
        &mut batch_cv,
    )?;
    let batch_calls = batched.evaluate_calls.get();

    ensure_equal(
        &batch_objectives,
        &expected_objectives,
        "default batch objectives differ from serial evaluation",
    )?;
    ensure_equal(
        &batch_cv,
        &expected_cv,
        "default batch constraints differ from serial evaluation",
    )?;
    ensure_equal(
        &batch_calls,
        &n_individuals,
        "default batch call count differs from row count",
    )?;
    ensure_equal(
        &batch_calls,
        &reference_calls,
        "default batch call count differs from serial evaluation",
    )?;
    Ok(())
}

#[test]
fn default_evaluate_batch_rejects_malformed_matrix_shapes() {
    let problem = BatchCountingProblem::new();

    let mut objectives = [0.0; 2];
    let mut violations = [0.0];
    assert!(problem
        .evaluate_batch(&[], 0, 2, &mut objectives, &mut violations)
        .is_err());

    let mut objectives = [0.0; 2];
    let mut violations = [0.0];
    assert!(problem
        .evaluate_batch(&[0.1, 0.2, 0.3], 2, 2, &mut objectives, &mut violations)
        .is_err());

    let mut objectives = [0.0; 1];
    let mut violations = [0.0];
    assert!(problem
        .evaluate_batch(&[0.1, 0.2], 2, 2, &mut objectives, &mut violations)
        .is_err());

    let mut objectives = [0.0; 2];
    assert!(problem
        .evaluate_batch(&[0.1, 0.2], 2, 2, &mut objectives, &mut [])
        .is_err());
}

/// Problem whose `evaluate_batch` override records how many times the engine
/// routed a batch through it, while preserving default per-row semantics.
struct OverrideBatchProblem {
    variables: Vec<VariableSpec>,
    batch_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Problem for OverrideBatchProblem {
    fn variable_specs(&self) -> &[VariableSpec] {
        &self.variables
    }

    fn objective_count(&self) -> usize {
        2
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "the custom-batch fixture deliberately preserves the evaluator's established floating-point expression order"
    )]
    fn evaluate(&self, decision: &[f64], objectives: &mut [f64]) -> anyhow::Result<f64> {
        let decision_pair: &[f64; 2] = decision
            .try_into()
            .map_err(|_| anyhow!("override fixture decision must have two coordinates"))?;
        let [first_decision, second_decision] = *decision_pair;
        let objective_pair: &mut [f64; 2] = objectives
            .try_into()
            .map_err(|_| anyhow!("override fixture objective row must have two coordinates"))?;
        let [first_objective, second_objective] = objective_pair;
        *first_objective = first_decision * first_decision;
        *second_objective = (1.0 - first_decision).powi(2) + second_decision;
        Ok(0.0)
    }

    fn evaluate_batch(
        &self,
        decisions: &[f64],
        n_variables: usize,
        n_objectives: usize,
        objectives: &mut [f64],
        constraint_violations: &mut [f64],
    ) -> anyhow::Result<()> {
        if n_variables == 0 || n_objectives == 0 {
            return Err(anyhow!("override fixture requires nonzero row widths"));
        }
        let row_count = constraint_violations.len();
        let decision_rows = decisions.chunks_exact(n_variables);
        if !decision_rows.remainder().is_empty() || decision_rows.len() != row_count {
            return Err(anyhow!("override fixture decision matrix shape is invalid"));
        }
        let objective_rows = objectives.chunks_exact_mut(n_objectives);
        if !objective_rows.into_remainder().is_empty() {
            return Err(anyhow!(
                "override fixture objective matrix has a partial row"
            ));
        }
        if objectives.chunks_exact(n_objectives).len() != row_count {
            return Err(anyhow!(
                "override fixture objective matrix row count is invalid"
            ));
        }
        self.batch_calls
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |count| count.checked_add(1),
            )
            .map_err(|_| anyhow!("override fixture batch count overflow"))?;
        for ((decision, objective_row), cv_slot) in decisions
            .chunks_exact(n_variables)
            .zip(objectives.chunks_exact_mut(n_objectives))
            .zip(constraint_violations.iter_mut())
        {
            let cv = self.evaluate(decision, objective_row)?;
            *cv_slot = cv.max(0.0);
        }
        Ok(())
    }
}

#[test]
fn engine_routes_evaluation_through_custom_evaluate_batch() -> anyhow::Result<()> {
    let batch_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let problem = OverrideBatchProblem {
        variables: vec![
            VariableSpec::new(0.0, 1.0, VariableKind::Continuous),
            VariableSpec::new(0.0, 1.0, VariableKind::Continuous),
        ],
        batch_calls: std::sync::Arc::clone(&batch_calls),
    };

    let config = Nsga2Config {
        population_size: 8,
        generations: 3,
        seed: 11,
        ..Nsga2Config::default()
    };

    let mut optimizer = Nsga2::new(problem, config)?;
    let result = optimizer.run()?;

    // One batch for the initial population plus one per generation step.
    let observed = batch_calls.load(std::sync::atomic::Ordering::Relaxed);
    let expected_calls = result
        .generations
        .checked_add(1)
        .ok_or_else(|| anyhow!("expected batch call count overflow"))?;
    ensure_equal(
        &observed,
        &expected_calls,
        "custom batch call count differs",
    )?;
    ensure_true(
        observed >= 1,
        "engine never called the custom evaluate_batch",
    )?;
    ensure_true(result.evaluations > 0, "engine did not evaluate any rows")?;
    Ok(())
}
