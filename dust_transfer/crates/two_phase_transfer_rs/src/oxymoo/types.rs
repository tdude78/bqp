use crate::oxymoo::validation::{checked_difference, checked_product, checked_sum};
use anyhow::{bail, Context, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariableKind {
    Continuous,
    Integer,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VariableSpec {
    pub lower: f64,
    pub upper: f64,
    pub kind: VariableKind,
}

impl VariableSpec {
    /// The solver builds its specs field-wise; this is the fixture constructor
    /// the tests and the Criterion harness use, and compiles only for them.
    #[cfg(any(test, feature = "bench-internal"))]
    #[must_use]
    pub const fn new(lower: f64, upper: f64, kind: VariableKind) -> Self {
        Self { lower, upper, kind }
    }
}

pub trait Problem {
    fn variable_specs(&self) -> &[VariableSpec];
    fn objective_count(&self) -> usize;

    /// Evaluate one decision row and write its objective row.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined evaluation error.
    fn evaluate(&self, decision: &[f64], objectives: &mut [f64]) -> Result<f64>;

    /// Evaluate a batch of decision rows into row-addressed objective and
    /// constraint-violation storage.
    ///
    /// `decisions` is a flat, row-major slice of `n_individuals * n_variables`
    /// values, `objectives` is a flat, row-major slice of
    /// `n_individuals * n_objectives` values, and `constraint_violations` holds
    /// one value per row. The number of rows is `decisions.len() / n_variables`;
    /// callers size `objectives` and `constraint_violations` to match.
    ///
    /// The default implementation performs the serial, per-row evaluation used
    /// by the engine: for each row it calls [`Problem::evaluate`], clamps the
    /// returned constraint violation to be non-negative, and rejects non-finite
    /// constraint violations or objectives. Implementations may override this to
    /// evaluate rows in a different manner (for example, in parallel) but must
    /// preserve the same per-row writes and error semantics.
    ///
    /// # Errors
    ///
    /// Returns an error for zero-width row shapes, shape mismatches, or
    /// non-finite values returned by the problem.
    fn evaluate_batch(
        &self,
        decisions: &[f64],
        n_variables: usize,
        n_objectives: usize,
        objectives: &mut [f64],
        constraint_violations: &mut [f64],
    ) -> Result<()> {
        if n_variables == 0 {
            bail!("problem evaluation variable count must be at least 1");
        }
        if n_objectives == 0 {
            bail!("problem evaluation objective count must be at least 1");
        }

        let row_count = constraint_violations.len();
        let expected_decisions = checked_product(row_count, n_variables, "decision matrix length")?;
        if decisions.len() != expected_decisions {
            bail!(
                "decision matrix length {} does not match row_count * n_variables = {expected_decisions}",
                decisions.len()
            );
        }
        let expected_objectives =
            checked_product(row_count, n_objectives, "objective matrix length")?;
        if objectives.len() != expected_objectives {
            bail!(
                "objective matrix length {} does not match row_count * n_objectives = {expected_objectives}",
                objectives.len()
            );
        }

        for (row, ((decision, objective_row), cv_slot)) in decisions
            .chunks_exact(n_variables)
            .zip(objectives.chunks_exact_mut(n_objectives))
            .zip(constraint_violations.iter_mut())
            .enumerate()
        {
            let cv = self.evaluate(decision, objective_row)?;
            if !cv.is_finite() {
                anyhow::bail!(
                    "problem evaluation returned non-finite constraint violation at row {row}"
                );
            }
            *cv_slot = cv.max(0.0);
            for (col, objective) in objective_row.iter().enumerate() {
                if !objective.is_finite() {
                    anyhow::bail!(
                        "problem evaluation wrote non-finite objective at row {row}, column {col}"
                    );
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SortConfig {
    pub constraint_tolerance: f64,
    pub objective_tolerance: f64,
}

impl Default for SortConfig {
    fn default() -> Self {
        Self {
            constraint_tolerance: 0.0,
            objective_tolerance: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Nsga2Config {
    pub population_size: usize,
    pub generations: usize,
    pub max_evaluations: Option<usize>,
    pub seed: u64,
    pub initial_decisions: Vec<f64>,
    pub crossover_probability: f64,
    pub mutation_probability: Option<f64>,
    pub eta_c: f64,
    pub eta_m: f64,
    pub tournament_size: usize,
    pub constraint_tolerance: f64,
    pub objective_tolerance: f64,
}

impl Default for Nsga2Config {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 100,
            max_evaluations: None,
            seed: 0,
            initial_decisions: Vec::new(),
            crossover_probability: 0.9,
            mutation_probability: None,
            eta_c: 20.0,
            eta_m: 20.0,
            tournament_size: 2,
            constraint_tolerance: 0.0,
            objective_tolerance: 0.0,
        }
    }
}

impl Nsga2Config {
    pub(crate) const fn sort_config(&self) -> SortConfig {
        SortConfig {
            constraint_tolerance: self.constraint_tolerance,
            objective_tolerance: self.objective_tolerance,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PopulationSnapshot {
    pub decisions: Vec<f64>,
    pub objectives: Vec<f64>,
    pub constraint_violations: Vec<f64>,
    pub ranks: Vec<usize>,
    pub crowding: Vec<f64>,
    n_individuals: usize,
    n_variables: usize,
    n_objectives: usize,
}

impl PopulationSnapshot {
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n_individuals
    }

    #[must_use]
    pub const fn variable_count(&self) -> usize {
        self.n_variables
    }

    #[must_use]
    pub const fn objective_count(&self) -> usize {
        self.n_objectives
    }

    /// Return one decision row after validating its backing shape.
    ///
    /// # Errors
    ///
    /// Returns an error when `row` is out of range or the public backing
    /// vector no longer matches this snapshot's declared shape.
    pub fn decision(&self, row: usize) -> Result<&[f64]> {
        self.validate_shape()?;
        Self::row(
            &self.decisions,
            row,
            self.n_individuals,
            self.n_variables,
            "decision",
        )
    }

    /// Return one objective row after validating its backing shape.
    ///
    /// # Errors
    ///
    /// Returns an error when `row` is out of range or the public backing
    /// vector no longer matches this snapshot's declared shape.
    pub fn objectives(&self, row: usize) -> Result<&[f64]> {
        self.validate_shape()?;
        Self::row(
            &self.objectives,
            row,
            self.n_individuals,
            self.n_objectives,
            "objective",
        )
    }

    pub(crate) fn empty(
        n_individuals: usize,
        n_variables: usize,
        n_objectives: usize,
    ) -> Result<Self> {
        let snapshot = Self {
            decisions: matrix_zeros(
                n_individuals,
                n_variables,
                "snapshot decision matrix allocation",
            )?,
            objectives: matrix_zeros(
                n_individuals,
                n_objectives,
                "snapshot objective matrix allocation",
            )?,
            constraint_violations: filled_vector(
                n_individuals,
                0.0,
                "snapshot constraint vector allocation",
            )?,
            ranks: filled_vector(n_individuals, usize::MAX, "snapshot rank vector allocation")?,
            crowding: filled_vector(n_individuals, 0.0, "snapshot crowding vector allocation")?,
            n_individuals,
            n_variables,
            n_objectives,
        };
        snapshot.validate_shape()?;
        Ok(snapshot)
    }

    pub(crate) fn resize_reuse(
        &mut self,
        n_individuals: usize,
        n_variables: usize,
        n_objectives: usize,
    ) -> Result<()> {
        let decision_len = checked_product(
            n_individuals,
            n_variables,
            "snapshot decision matrix allocation",
        )?;
        let objective_len = checked_product(
            n_individuals,
            n_objectives,
            "snapshot objective matrix allocation",
        )?;
        reserve_length(
            &mut self.decisions,
            decision_len,
            "snapshot decision matrix allocation",
        )?;
        reserve_length(
            &mut self.objectives,
            objective_len,
            "snapshot objective matrix allocation",
        )?;
        reserve_length(
            &mut self.constraint_violations,
            n_individuals,
            "snapshot constraint vector allocation",
        )?;
        reserve_length(
            &mut self.ranks,
            n_individuals,
            "snapshot rank vector allocation",
        )?;
        reserve_length(
            &mut self.crowding,
            n_individuals,
            "snapshot crowding vector allocation",
        )?;
        self.n_individuals = n_individuals;
        self.n_variables = n_variables;
        self.n_objectives = n_objectives;
        self.decisions.resize(decision_len, 0.0);
        self.decisions.fill(0.0);
        self.objectives.resize(objective_len, 0.0);
        self.objectives.fill(0.0);
        self.constraint_violations.resize(n_individuals, 0.0);
        self.constraint_violations.fill(0.0);
        self.ranks.resize(n_individuals, usize::MAX);
        self.ranks.fill(usize::MAX);
        self.crowding.resize(n_individuals, 0.0);
        self.crowding.fill(0.0);
        self.validate_shape()
    }

    /// Copy one fully validated source row into this snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error when either snapshot's public storage shape is invalid
    ///, their row widths are incompatible, or either row is out of range.
    pub(crate) fn copy_row_from(&mut self, dst: usize, source: &Self, src: usize) -> Result<()> {
        self.validate_shape()?;
        source.validate_shape()?;
        if self.n_variables != source.n_variables || self.n_objectives != source.n_objectives {
            bail!(
                "snapshot row copy has incompatible widths: destination variables={}, objectives={}; source variables={}, objectives={}",
                self.n_variables,
                self.n_objectives,
                source.n_variables,
                source.n_objectives,
            );
        }
        self.decision_mut(dst)?
            .copy_from_slice(source.decision(src)?);
        self.objectives_mut(dst)?
            .copy_from_slice(source.objectives(src)?);
        let source_cv = *source
            .constraint_violations
            .get(src)
            .ok_or_else(|| anyhow::anyhow!("source constraint row {src} is out of range"))?;
        let destination_cv = self
            .constraint_violations
            .get_mut(dst)
            .ok_or_else(|| anyhow::anyhow!("destination constraint row {dst} is out of range"))?;
        *destination_cv = source_cv;
        Ok(())
    }

    /// Validate all public storage vectors against the declared snapshot shape.
    ///
    /// # Errors
    ///
    /// Returns an error when a checked matrix dimension overflows or any public
    /// vector has a length inconsistent with this snapshot.
    pub(crate) fn validate_shape(&self) -> Result<()> {
        let decision_len = checked_product(
            self.n_individuals,
            self.n_variables,
            "snapshot decision matrix length",
        )?;
        if self.decisions.len() != decision_len {
            bail!(
                "snapshot decision matrix length {} does not match {decision_len}",
                self.decisions.len()
            );
        }
        let objective_len = checked_product(
            self.n_individuals,
            self.n_objectives,
            "snapshot objective matrix length",
        )?;
        if self.objectives.len() != objective_len {
            bail!(
                "snapshot objective matrix length {} does not match {objective_len}",
                self.objectives.len()
            );
        }
        for (label, length) in [
            ("constraint violation", self.constraint_violations.len()),
            ("rank", self.ranks.len()),
            ("crowding", self.crowding.len()),
        ] {
            if length != self.n_individuals {
                bail!(
                    "snapshot {label} length {length} does not match {} rows",
                    self.n_individuals
                );
            }
        }
        Ok(())
    }

    fn row<'a>(
        values: &'a [f64],
        row: usize,
        row_count: usize,
        width: usize,
        label: &'static str,
    ) -> Result<&'a [f64]> {
        if row >= row_count {
            bail!("{label} row {row} is out of range for {row_count} rows");
        }
        let expected = checked_product(row_count, width, "snapshot matrix length")?;
        if values.len() != expected {
            bail!(
                "{label} matrix length {} does not match snapshot shape {}/{}",
                values.len(),
                row_count,
                width
            );
        }
        let start = checked_product(row, width, "snapshot row offset")?;
        let end = checked_sum(start, width, "snapshot row end")?;
        values
            .get(start..end)
            .ok_or_else(|| anyhow::anyhow!("{label} row {row} is out of bounds"))
    }

    fn decision_mut(&mut self, row: usize) -> Result<&mut [f64]> {
        self.validate_shape()?;
        Self::row_mut(
            &mut self.decisions,
            row,
            self.n_individuals,
            self.n_variables,
            "decision",
        )
    }

    fn objectives_mut(&mut self, row: usize) -> Result<&mut [f64]> {
        self.validate_shape()?;
        Self::row_mut(
            &mut self.objectives,
            row,
            self.n_individuals,
            self.n_objectives,
            "objective",
        )
    }

    fn row_mut<'a>(
        values: &'a mut [f64],
        row: usize,
        row_count: usize,
        width: usize,
        label: &'static str,
    ) -> Result<&'a mut [f64]> {
        if row >= row_count {
            bail!("{label} row {row} is out of range for {row_count} rows");
        }
        let expected = checked_product(row_count, width, "snapshot matrix length")?;
        if values.len() != expected {
            bail!(
                "{label} matrix length {} does not match snapshot shape {row_count}/{width}",
                values.len()
            );
        }
        let start = checked_product(row, width, "snapshot row offset")?;
        let end = checked_sum(start, width, "snapshot row end")?;
        values
            .get_mut(start..end)
            .ok_or_else(|| anyhow::anyhow!("{label} row {row} is out of bounds"))
    }
}

fn matrix_zeros(rows: usize, columns: usize, description: &'static str) -> Result<Vec<f64>> {
    let length = checked_product(rows, columns, description)?;
    let mut values = Vec::new();
    reserve_length(&mut values, length, description)?;
    values.resize(length, 0.0);
    Ok(values)
}

fn filled_vector<T: Clone>(length: usize, value: T, description: &'static str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    reserve_length(&mut values, length, description)?;
    values.resize(length, value);
    Ok(values)
}

fn reserve_length<T>(values: &mut Vec<T>, length: usize, description: &'static str) -> Result<()> {
    let additional = if length > values.len() {
        checked_difference(length, values.len(), description)?
    } else {
        0
    };
    values
        .try_reserve_exact(additional)
        .with_context(|| format!("{description} allocation for {length} values failed"))
}

#[derive(Clone, Debug, PartialEq)]
pub struct Nsga2Result {
    pub population: PopulationSnapshot,
    pub fronts: Vec<Vec<usize>>,
    pub generations: usize,
    pub evaluations: usize,
}

#[cfg(test)]
mod tests {
    use super::{reserve_length, PopulationSnapshot};
    use anyhow::bail;
    use anyhow::Result;
    use std::collections::TryReserveError;

    #[test]
    fn snapshot_allocation_retains_try_reserve_cause() -> Result<()> {
        let mut values = Vec::<u8>::new();
        let error = reserve_length(&mut values, usize::MAX, "test snapshot")
            .err()
            .ok_or_else(|| {
                anyhow::anyhow!("impossible snapshot allocation unexpectedly succeeded")
            })?;
        if error.downcast_ref::<TryReserveError>().is_none() {
            bail!("snapshot allocation error lost its TryReserveError cause");
        }
        if error.to_string() != format!("test snapshot allocation for {} values failed", usize::MAX)
        {
            bail!("snapshot allocation error lost its outer context");
        }
        Ok(())
    }

    #[test]
    fn public_snapshot_storage_corruption_fails_closed() -> Result<()> {
        let mut snapshot = PopulationSnapshot::empty(1, 2, 1)?;
        snapshot.decisions.pop();

        if snapshot.decision(0).is_ok() {
            bail!("corrupt public decision storage was accepted");
        }
        if snapshot.objectives(0).is_ok() {
            bail!("corrupt public decision storage was accepted for objective access");
        }
        Ok(())
    }

    #[test]
    fn resize_reuse_resets_every_population_vector() -> Result<()> {
        let mut snapshot = PopulationSnapshot::empty(1, 2, 1)?;
        snapshot.decisions.fill(1.0);
        snapshot.objectives.fill(2.0);
        snapshot.constraint_violations.fill(3.0);
        snapshot.ranks.fill(4);
        snapshot.crowding.fill(5.0);

        snapshot.resize_reuse(1, 2, 1)?;

        if snapshot.decisions != vec![0.0, 0.0] {
            bail!("resize reuse did not clear decisions");
        }
        if snapshot.objectives != vec![0.0] {
            bail!("resize reuse did not clear objectives");
        }
        if snapshot.constraint_violations != vec![0.0] {
            bail!("resize reuse did not clear constraint violations");
        }
        if snapshot.ranks != vec![usize::MAX] {
            bail!("resize reuse did not reset ranks");
        }
        if snapshot.crowding != vec![0.0] {
            bail!("resize reuse did not clear crowding");
        }
        Ok(())
    }

    #[test]
    fn copy_row_from_rejects_incompatible_widths_without_mutation() -> Result<()> {
        let mut destination = PopulationSnapshot::empty(1, 2, 1)?;
        destination.decisions.copy_from_slice(&[1.0, 2.0]);
        destination.objectives.copy_from_slice(&[3.0]);
        destination.constraint_violations.copy_from_slice(&[4.0]);
        let source = PopulationSnapshot::empty(1, 1, 2)?;
        let before = destination.clone();

        let call = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            destination.copy_row_from(0, &source, 0)
        }));
        let Ok(result) = call else {
            bail!("incompatible snapshot row copy panicked");
        };
        if result.is_ok() {
            bail!("incompatible snapshot row copy unexpectedly succeeded");
        }
        if destination != before {
            bail!("incompatible snapshot row copy mutated its destination");
        }
        Ok(())
    }
}
