use anyhow::{bail, Context, Result};

use crate::oxymoo::{ArithmeticOverflow, Nsga2Config, VariableKind, VariableSpec};

const MAX_EXACT_F64_INTEGER: f64 = 9_007_199_254_740_992.0;
const MIN_EXACT_F64_INTEGER: f64 = -MAX_EXACT_F64_INTEGER;

/// Multiply two collection dimensions without wrapping.
///
/// # Errors
///
/// Returns an error when the requested shape does not fit in `usize`.
pub fn checked_product(left: usize, right: usize, description: &'static str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(ArithmeticOverflow)
        .with_context(|| format!("{description} overflows usize: {left} * {right}"))
}

/// Add two collection counters without wrapping.
///
/// # Errors
///
/// Returns an error when the requested count does not fit in `usize`.
pub fn checked_sum(left: usize, right: usize, description: &'static str) -> Result<usize> {
    left.checked_add(right)
        .ok_or(ArithmeticOverflow)
        .with_context(|| format!("{description} overflows usize: {left} + {right}"))
}

/// Subtract one validated collection offset from another.
///
/// # Errors
///
/// Returns an error when `right` exceeds `left`.
pub fn checked_difference(left: usize, right: usize, description: &'static str) -> Result<usize> {
    left.checked_sub(right)
        .ok_or(ArithmeticOverflow)
        .with_context(|| format!("{description} underflows usize: {left} - {right}"))
}

/// Convert a practical collection count into an exactly representable `f64`.
///
/// # Errors
///
/// Returns an error when the count exceeds `u32`, which this optimizer treats
/// as an invalid control because it cannot allocate or evaluate that many rows
/// safely in one call.
pub fn count_as_f64(value: usize, description: &'static str) -> Result<f64> {
    let value = u32::try_from(value)
        .map_err(|_| ArithmeticOverflow)
        .with_context(|| format!("{description} exceeds u32: {value}"))?;
    Ok(f64::from(value))
}

/// Validate NSGA-II configuration values before allocation or mutation.
///
/// # Errors
///
/// Returns an error for invalid population, probability, evaluation-limit, or
/// distribution-index controls.
pub fn validate_config(config: &Nsga2Config) -> Result<()> {
    validate_nonnegative_finite("constraint_tolerance", config.constraint_tolerance)?;
    validate_nonnegative_finite("objective_tolerance", config.objective_tolerance)?;
    validate_common_evolution_config(
        config.population_size,
        config.max_evaluations,
        config.crossover_probability,
        config.mutation_probability,
        config.eta_c,
        config.eta_m,
    )?;
    if config.tournament_size == 0 {
        bail!("tournament size must be at least 1");
    }
    Ok(())
}

/// Validate a finite probability in the inclusive unit interval.
///
/// # Errors
///
/// Returns an error when `value` is non-finite or outside `[0, 1]`.
pub fn validate_probability(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        bail!("invalid probability {name}: {value}");
    }
}

fn validate_common_evolution_config(
    population_size: usize,
    max_evaluations: Option<usize>,
    crossover_probability: f64,
    mutation_probability: Option<f64>,
    eta_c: f64,
    eta_m: f64,
) -> Result<()> {
    if population_size < 2 {
        bail!("population size must be at least 2");
    }
    if let Some(max_evaluations) = max_evaluations {
        if max_evaluations < population_size {
            bail!(
                "max_evaluations {max_evaluations} is smaller than population_size {population_size}"
            );
        }
    }
    validate_probability("crossover_probability", crossover_probability)?;
    if let Some(value) = mutation_probability {
        validate_probability("mutation_probability", value)?;
    }
    if !eta_c.is_finite() || !eta_m.is_finite() || eta_c <= 0.0 || eta_m <= 0.0 {
        bail!("eta values must be finite and positive");
    }
    Ok(())
}

/// Validate variable bounds and integer lattices.
///
/// # Errors
///
/// Returns an error for an empty list, a non-finite bound, an inverted domain,
/// or an invalid integer lattice.
pub fn validate_variables(variables: &[VariableSpec]) -> Result<()> {
    if variables.is_empty() {
        bail!("variable list must not be empty");
    }
    for (index, spec) in variables.iter().enumerate() {
        let valid = spec.lower.is_finite()
            && spec.upper.is_finite()
            && spec.lower <= spec.upper
            && match spec.kind {
                VariableKind::Continuous => {
                    if spec.lower < spec.upper {
                        (spec.upper - spec.lower).is_finite()
                    } else {
                        true
                    }
                }
                VariableKind::Integer => valid_integer_lattice_bounds(spec),
            };
        if !valid {
            let lower = spec.lower;
            let upper = spec.upper;
            let kind = spec.kind;
            bail!("invalid variable {index}: lower={lower}, upper={upper}, kind={kind:?}");
        }
    }
    Ok(())
}

fn validate_nonnegative_finite(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        bail!("{name} must be finite and non-negative: {value}");
    }
}

fn valid_integer_lattice_bounds(spec: &VariableSpec) -> bool {
    let lower = spec.lower.ceil();
    let upper = spec.upper.floor();
    lower <= upper && lower >= MIN_EXACT_F64_INTEGER && upper <= MAX_EXACT_F64_INTEGER
}

/// Validate a finite, row-major objective matrix.
///
/// # Errors
///
/// Returns an error for an invalid width, an overflowing shape, a length
/// mismatch, or a non-finite objective.
pub fn validate_objective_matrix(
    objectives: &[f64],
    n_individuals: usize,
    n_objectives: usize,
) -> Result<()> {
    if n_objectives == 0 {
        bail!("objective count must be at least 1");
    }
    let expected = checked_product(n_individuals, n_objectives, "objective matrix length")?;
    if objectives.len() != expected {
        let len = objectives.len();
        bail!("objective matrix length {len} does not match n_individuals * n_objectives = {expected}");
    }
    for (row, objective_row) in objectives.chunks_exact(n_objectives).enumerate() {
        for (col, objective) in objective_row.iter().enumerate() {
            if !objective.is_finite() {
                bail!("non-finite value at row {row}, column {col}");
            }
        }
    }
    Ok(())
}

/// Validate a finite per-row constraint-violation vector.
///
/// # Errors
///
/// Returns an error for a length mismatch or a non-finite value.
pub fn validate_constraint_vector(
    constraint_violations: &[f64],
    n_individuals: usize,
) -> Result<()> {
    if constraint_violations.len() != n_individuals {
        let len = constraint_violations.len();
        bail!("constraint violation length {len} does not match n_individuals = {n_individuals}");
    }
    for (row, &cv) in constraint_violations.iter().enumerate() {
        if !cv.is_finite() {
            bail!("non-finite constraint violation at row {row}");
        }
    }
    Ok(())
}
