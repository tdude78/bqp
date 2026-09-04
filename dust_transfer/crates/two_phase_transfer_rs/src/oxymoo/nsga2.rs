use crate::oxymoo::operators::{
    crowded_better, initial_decisions_with_random_fill, polynomial_mutation, sbx_crossover_into,
    validate_initial_decisions,
};
use crate::oxymoo::sort::{
    assign_capped_survivor_selection_into, assign_rank_and_crowding_into,
    rebuild_fronts_from_ranks, recompute_crowding_for_front, SortWorkspace,
};
use crate::oxymoo::validation::{
    checked_difference, checked_product, checked_sum, count_as_f64, validate_config,
};
use crate::oxymoo::{Nsga2Config, Nsga2Result, PopulationSnapshot, Problem, VariableSpec};
use anyhow::{bail, Context, Result};
use rand::RngExt;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use smallvec::SmallVec;

pub struct Nsga2<P> {
    problem: P,
    config: Nsga2Config,
    variables: Vec<VariableSpec>,
    n_objectives: usize,
    rng: Xoshiro256PlusPlus,
    population: PopulationSnapshot,
    fronts: Vec<Vec<usize>>,
    sort_workspace: SortWorkspace,
    combined_fronts: Vec<Vec<usize>>,
    offspring_decisions: Vec<f64>,
    offspring: PopulationSnapshot,
    combined: PopulationSnapshot,
    next_population: PopulationSnapshot,
    selected_rows: Vec<usize>,
    split_order: Vec<usize>,
    split_front: Vec<usize>,
    child1: Vec<f64>,
    child2: Vec<f64>,
    generation: usize,
    evaluations: usize,
}

fn reserved_vec<T>(capacity: usize, description: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    reserve_values(&mut values, capacity, description)?;
    Ok(values)
}

fn reserve_values<T>(values: &mut Vec<T>, additional: usize, description: &str) -> Result<()> {
    values
        .try_reserve(additional)
        .with_context(|| format!("{description} allocation for {additional} values failed"))
}

impl<P: Problem> Nsga2<P> {
    /// Construct a validated NSGA-II optimizer.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid config, variable, objective, allocation, or
    /// initial-decision shapes.
    pub fn new(problem: P, config: Nsga2Config) -> Result<Self> {
        validate_config(&config)?;
        let variables = problem.variable_specs().to_vec();
        validate_initial_decisions(&config.initial_decisions, &variables)?;
        let n_objectives = problem.objective_count();
        if n_objectives == 0 {
            bail!("objective count must be at least 1");
        }
        let n_variables = variables.len();
        let decision_capacity = checked_product(
            config.population_size,
            n_variables,
            "NSGA-II decision capacity",
        )?;
        checked_product(
            config.population_size,
            n_objectives,
            "NSGA-II objective capacity",
        )?;
        let combined_population_size = checked_sum(
            config.population_size,
            config.population_size,
            "NSGA-II combined population size",
        )?;
        checked_product(
            combined_population_size,
            n_variables,
            "NSGA-II combined decision capacity",
        )?;
        checked_product(
            combined_population_size,
            n_objectives,
            "NSGA-II combined objective capacity",
        )?;

        let mut optimizer = Self {
            problem,
            rng: Xoshiro256PlusPlus::seed_from_u64(config.seed),
            population: PopulationSnapshot::empty(
                config.population_size,
                n_variables,
                n_objectives,
            )?,
            fronts: Vec::new(),
            sort_workspace: SortWorkspace::default(),
            combined_fronts: Vec::new(),
            offspring_decisions: reserved_vec(decision_capacity, "NSGA-II offspring decisions")?,
            offspring: PopulationSnapshot::empty(
                config.population_size,
                n_variables,
                n_objectives,
            )?,
            combined: PopulationSnapshot::empty(
                combined_population_size,
                n_variables,
                n_objectives,
            )?,
            next_population: PopulationSnapshot::empty(
                config.population_size,
                n_variables,
                n_objectives,
            )?,
            selected_rows: reserved_vec(config.population_size, "NSGA-II selected rows")?,
            split_order: reserved_vec(combined_population_size, "NSGA-II split order")?,
            split_front: reserved_vec(config.population_size, "NSGA-II split front")?,
            child1: reserved_vec(n_variables, "NSGA-II first child")?,
            child2: reserved_vec(n_variables, "NSGA-II second child")?,
            generation: 0,
            evaluations: 0,
            variables,
            n_objectives,
            config,
        };

        optimizer.population.validate_shape()?;
        optimizer.offspring.validate_shape()?;
        optimizer.combined.validate_shape()?;
        optimizer.next_population.validate_shape()?;

        let initial_decisions = initial_decisions_with_random_fill(
            &optimizer.config.initial_decisions,
            optimizer.config.population_size,
            &optimizer.variables,
            &mut optimizer.rng,
        )?;
        optimizer.population = optimizer.evaluate_decisions(initial_decisions)?;
        optimizer.refresh_rank_and_crowding()?;
        Ok(optimizer)
    }

    /// Advance one fully budgeted generation.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid snapshot shape, failed problem evaluation,
    /// or a checked control-counter overflow.
    pub fn step(&mut self) -> Result<()> {
        self.population.validate_shape()?;
        self.offspring.validate_shape()?;
        self.combined.validate_shape()?;
        self.next_population.validate_shape()?;
        if !self.can_evaluate_full_generation()? {
            return Ok(());
        }
        make_offspring_into(
            &self.population,
            &self.variables,
            &self.config,
            &mut self.rng,
            &mut self.offspring_decisions,
            &mut self.child1,
            &mut self.child2,
        )?;
        let evaluated = evaluate_decisions_into(
            &self.problem,
            &self.offspring_decisions,
            &mut self.offspring,
            self.variables.len(),
            self.n_objectives,
        )?;
        self.evaluations = checked_sum(self.evaluations, evaluated, "NSGA-II evaluation count")?;
        combine_population_sort_view_into(&self.population, &self.offspring, &mut self.combined)?;
        let split_rank = assign_capped_survivor_selection_into(
            &mut self.combined,
            self.config.sort_config(),
            self.config.population_size,
            &mut self.sort_workspace,
            &mut self.combined_fronts,
            &mut self.selected_rows,
            &mut self.split_order,
        )?;
        select_parent_offspring_rows_with_metadata_into(
            &self.population,
            &self.offspring,
            &self.combined,
            &self.selected_rows,
            &mut self.next_population,
        )?;
        self.next_population.validate_shape()?;
        rebuild_fronts_from_ranks(
            &self.next_population.ranks,
            self.next_population.len(),
            &mut self.fronts,
            &mut self.sort_workspace,
        )?;
        if let Some(rank) = split_rank {
            if let Some(front) = self.fronts.get(rank) {
                self.split_front.clear();
                self.split_front.extend_from_slice(front);
                recompute_crowding_for_front(
                    &mut self.next_population,
                    &self.split_front,
                    &mut self.sort_workspace,
                )?;
            }
        }
        std::mem::swap(&mut self.population, &mut self.next_population);
        self.generation = checked_sum(self.generation, 1, "NSGA-II generation count")?;
        Ok(())
    }

    /// Run until the configured generation or evaluation budget is exhausted.
    ///
    /// # Errors
    ///
    /// Returns an error propagated from [`Self::step`].
    ///
    /// Production drives the solver through `run_owned_with_problem`; this is
    /// the plain generation loop the tests and the Criterion harness measure.
    #[cfg(any(test, feature = "bench-internal"))]
    pub fn run(&mut self) -> Result<Nsga2Result> {
        while self.generation < self.config.generations {
            if !self.can_evaluate_full_generation()? {
                break;
            }
            self.step()?;
        }
        Ok(Nsga2Result {
            population: self.population.clone(),
            fronts: self.fronts.clone(),
            generations: self.generation,
            evaluations: self.evaluations,
        })
    }

    /// Run and consume the optimizer, returning both its problem and result.
    ///
    /// # Errors
    ///
    /// Returns an error propagated from [`Self::step`].
    pub fn run_owned_with_problem(mut self) -> Result<(P, Nsga2Result)> {
        while self.generation < self.config.generations {
            if !self.can_evaluate_full_generation()? {
                break;
            }
            self.step()?;
        }
        let result = Nsga2Result {
            population: self.population,
            fronts: self.fronts,
            generations: self.generation,
            evaluations: self.evaluations,
        };
        Ok((self.problem, result))
    }

    fn can_evaluate_full_generation(&self) -> Result<bool> {
        let Some(limit) = self.config.max_evaluations else {
            return Ok(true);
        };
        Ok(checked_sum(
            self.evaluations,
            self.config.population_size,
            "NSGA-II full-generation evaluation budget",
        )? <= limit)
    }

    /// Read-only views for `first_front_objective_signature`, the stable-objective
    /// stop the policy bench compares against. Gated with it.
    #[cfg(feature = "bench-internal")]
    #[must_use]
    pub const fn population(&self) -> &PopulationSnapshot {
        &self.population
    }

    #[cfg(feature = "bench-internal")]
    #[must_use]
    pub fn fronts(&self) -> &[Vec<usize>] {
        &self.fronts
    }

    #[cfg(feature = "bench-internal")]
    #[must_use]
    pub fn into_problem_and_result(self) -> (P, Nsga2Result) {
        let result = Nsga2Result {
            population: self.population,
            fronts: self.fronts,
            generations: self.generation,
            evaluations: self.evaluations,
        };
        (self.problem, result)
    }

    fn evaluate_decisions(&mut self, decisions: Vec<f64>) -> Result<PopulationSnapshot> {
        let n_variables = self.variables.len();
        if n_variables == 0 {
            bail!("NSGA-II variable count must be at least 1");
        }
        let decision_rows = decisions.chunks_exact(n_variables);
        if !decision_rows.remainder().is_empty() {
            bail!(
                "NSGA-II decision length {} is not a whole number of rows with width {n_variables}",
                decisions.len()
            );
        }
        let n_individuals = decision_rows.len();
        let expected = checked_product(n_individuals, n_variables, "NSGA-II decision length")?;
        if decisions.len() != expected {
            bail!(
                "NSGA-II decision length {} does not match expected {expected}",
                decisions.len()
            );
        }
        let mut population =
            PopulationSnapshot::empty(n_individuals, n_variables, self.n_objectives)?;
        population.validate_shape()?;
        population.decisions = decisions;
        population.validate_shape()?;

        self.problem.evaluate_batch(
            &population.decisions,
            n_variables,
            self.n_objectives,
            &mut population.objectives,
            &mut population.constraint_violations,
        )?;
        population.validate_shape()?;
        self.evaluations =
            checked_sum(self.evaluations, n_individuals, "NSGA-II evaluation count")?;
        Ok(population)
    }

    fn refresh_rank_and_crowding(&mut self) -> Result<()> {
        self.population.validate_shape()?;
        assign_rank_and_crowding_into(
            &mut self.population,
            self.config.sort_config(),
            &mut self.sort_workspace,
            &mut self.fronts,
        )
    }
}

fn make_offspring_into(
    population: &PopulationSnapshot,
    variables: &[VariableSpec],
    config: &Nsga2Config,
    rng: &mut Xoshiro256PlusPlus,
    decisions: &mut Vec<f64>,
    child1: &mut Vec<f64>,
    child2: &mut Vec<f64>,
) -> Result<()> {
    population.validate_shape()?;
    let n_variables = variables.len();
    if n_variables == 0 {
        bail!("NSGA-II offspring generation requires at least one variable");
    }
    if population.variable_count() != n_variables {
        bail!(
            "NSGA-II population variable width {} does not match variable list width {n_variables}",
            population.variable_count()
        );
    }
    if population.len() != config.population_size {
        bail!(
            "NSGA-II population row count {} does not match configured population size {}",
            population.len(),
            config.population_size
        );
    }
    let mutation_probability = config.mutation_probability.map_or_else(
        || {
            count_as_f64(n_variables, "NSGA-II variable count")
                .map(|variable_count| 1.0 / variable_count)
        },
        Ok,
    )?;
    let target_decision_len = checked_product(
        config.population_size,
        n_variables,
        "NSGA-II offspring decision length",
    )?;
    decisions.clear();
    reserve_values(decisions, target_decision_len, "NSGA-II offspring")?;
    while decisions.len() < target_decision_len {
        let p1 = binary_tournament_select(population, config.tournament_size, rng)?;
        let p2 = binary_tournament_select(population, config.tournament_size, rng)?;
        sbx_crossover_into(
            population.decision(p1)?,
            population.decision(p2)?,
            variables,
            config.crossover_probability,
            config.eta_c,
            rng,
            child1,
            child2,
        )?;
        polynomial_mutation(child1, variables, mutation_probability, config.eta_m, rng)?;
        polynomial_mutation(child2, variables, mutation_probability, config.eta_m, rng)?;
        if child1.len() != n_variables || child2.len() != n_variables {
            bail!(
                "NSGA-II offspring operator produced rows of widths {} and {} instead of {n_variables}",
                child1.len(),
                child2.len()
            );
        }
        decisions.extend_from_slice(child1);
        if decisions.len() < target_decision_len {
            decisions.extend_from_slice(child2);
        }
    }
    Ok(())
}

fn binary_tournament_select(
    population: &PopulationSnapshot,
    tournament_size: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<usize> {
    population.validate_shape()?;
    let row_count = population.len();
    if row_count == 0 {
        bail!("NSGA-II tournament cannot select from an empty population");
    }
    if tournament_size == 0 {
        bail!("NSGA-II tournament size must be at least 1");
    }
    if row_count == 1 {
        return Ok(0);
    }
    let k = tournament_size.min(row_count);
    if k == 1 {
        return Ok(rng.random_range(0..row_count));
    }
    if k == 2 {
        let first = rng.random_range(0..row_count);
        let second_upper = checked_difference(row_count, 1, "NSGA-II tournament range")?;
        let second_without_first = rng.random_range(0..second_upper);
        let second = if second_without_first >= first {
            checked_sum(second_without_first, 1, "NSGA-II tournament row")?
        } else {
            second_without_first
        };
        return Ok(
            if crowded_better(first, second, &population.ranks, &population.crowding, rng)? {
                first
            } else {
                second
            },
        );
    }
    let mut contenders: SmallVec<[usize; 8]> = SmallVec::new();
    while contenders.len() < k {
        let idx = rng.random_range(0..row_count);
        if !contenders.contains(&idx) {
            contenders.push(idx);
        }
    }
    let mut best = contenders
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("NSGA-II tournament produced no contenders"))?;
    for &candidate in contenders.iter().skip(1) {
        if crowded_better(
            candidate,
            best,
            &population.ranks,
            &population.crowding,
            rng,
        )? {
            best = candidate;
        }
    }
    Ok(best)
}

fn evaluate_decisions_into<P: Problem>(
    problem: &P,
    decisions: &[f64],
    population: &mut PopulationSnapshot,
    n_variables: usize,
    n_objectives: usize,
) -> Result<usize> {
    if n_variables == 0 {
        bail!("NSGA-II evaluated decision rows require at least one variable");
    }
    if n_objectives == 0 {
        bail!("NSGA-II evaluated decision rows require at least one objective");
    }
    population.validate_shape()?;
    let decision_rows = decisions.chunks_exact(n_variables);
    if !decision_rows.remainder().is_empty() {
        bail!(
            "NSGA-II offspring decision length {} is not a whole number of rows with width {n_variables}",
            decisions.len()
        );
    }
    let n_individuals = decision_rows.len();
    let expected_len = checked_product(
        n_individuals,
        n_variables,
        "NSGA-II offspring decision length",
    )?;
    if decisions.len() != expected_len {
        bail!(
            "NSGA-II offspring decision length {} does not match expected {expected_len}",
            decisions.len()
        );
    }
    population.resize_reuse(n_individuals, n_variables, n_objectives)?;
    population.validate_shape()?;
    population.decisions.copy_from_slice(decisions);

    problem.evaluate_batch(
        &population.decisions,
        n_variables,
        n_objectives,
        &mut population.objectives,
        &mut population.constraint_violations,
    )?;
    population.validate_shape()?;
    Ok(n_individuals)
}

#[cfg(test)]
fn select_survivor_rows_into(
    fronts: &[Vec<usize>],
    crowding: &[f64],
    population_size: usize,
    selected: &mut Vec<usize>,
    split_order: &mut Vec<usize>,
) -> Result<Option<usize>> {
    for front in fronts {
        for &row in front {
            if crowding.get(row).is_none() {
                bail!("test survivor selection crowding row {row} is missing");
            }
        }
    }
    selected.clear();
    for (rank, front) in fronts.iter().enumerate() {
        let selected_with_front = checked_sum(
            selected.len(),
            front.len(),
            "test survivor selection row count",
        )?;
        if selected_with_front <= population_size {
            selected.extend_from_slice(front);
            continue;
        }
        let remaining = checked_difference(
            population_size,
            selected.len(),
            "test survivor selection remaining row count",
        )?;
        split_order.clear();
        split_order.extend_from_slice(front);
        let by_crowding_desc = |&left: &usize, &right: &usize| {
            let left_crowding = crowding.get(left).copied().map_or(f64::NAN, |value| value);
            let right_crowding = crowding.get(right).copied().map_or(f64::NAN, |value| value);
            right_crowding
                .total_cmp(&left_crowding)
                .then_with(|| left.cmp(&right))
        };
        if remaining < split_order.len() {
            let (top, _, _) = split_order.select_nth_unstable_by(remaining, by_crowding_desc);
            top.sort_by(by_crowding_desc);
            selected.extend(top.iter().copied());
        } else {
            split_order.sort_by(by_crowding_desc);
            selected.extend(split_order.iter().copied());
        }
        return Ok(Some(rank));
    }
    Ok(None)
}

fn combine_population_sort_view_into(
    left: &PopulationSnapshot,
    right: &PopulationSnapshot,
    out: &mut PopulationSnapshot,
) -> Result<()> {
    left.validate_shape()?;
    right.validate_shape()?;
    out.validate_shape()?;
    if left.objective_count() != right.objective_count() {
        bail!(
            "NSGA-II sort-view objective widths differ: left={}, right={}",
            left.objective_count(),
            right.objective_count()
        );
    }
    let total_rows = checked_sum(left.len(), right.len(), "NSGA-II sort-view row count")?;
    out.resize_reuse(total_rows, 0, left.objective_count())?;
    out.validate_shape()?;

    let objective_split = left.objectives.len();
    out.objectives
        .get_mut(..objective_split)
        .ok_or_else(|| anyhow::anyhow!("NSGA-II sort-view left objective range is invalid"))?
        .copy_from_slice(&left.objectives);
    out.objectives
        .get_mut(objective_split..)
        .ok_or_else(|| anyhow::anyhow!("NSGA-II sort-view right objective range is invalid"))?
        .copy_from_slice(&right.objectives);

    let constraint_split = left.constraint_violations.len();
    out.constraint_violations
        .get_mut(..constraint_split)
        .ok_or_else(|| anyhow::anyhow!("NSGA-II sort-view left constraint range is invalid"))?
        .copy_from_slice(&left.constraint_violations);
    out.constraint_violations
        .get_mut(constraint_split..)
        .ok_or_else(|| anyhow::anyhow!("NSGA-II sort-view right constraint range is invalid"))?
        .copy_from_slice(&right.constraint_violations);
    out.validate_shape()?;
    Ok(())
}

fn select_parent_offspring_rows_with_metadata_into(
    parent: &PopulationSnapshot,
    offspring: &PopulationSnapshot,
    metadata: &PopulationSnapshot,
    rows: &[usize],
    out: &mut PopulationSnapshot,
) -> Result<()> {
    parent.validate_shape()?;
    offspring.validate_shape()?;
    metadata.validate_shape()?;
    out.validate_shape()?;
    if parent.variable_count() != offspring.variable_count() {
        bail!(
            "NSGA-II parent and offspring variable widths differ: parent={}, offspring={}",
            parent.variable_count(),
            offspring.variable_count()
        );
    }
    if parent.objective_count() != offspring.objective_count() {
        bail!(
            "NSGA-II parent and offspring objective widths differ: parent={}, offspring={}",
            parent.objective_count(),
            offspring.objective_count()
        );
    }
    let combined_len = checked_sum(parent.len(), offspring.len(), "NSGA-II combined row count")?;
    if metadata.len() != combined_len {
        bail!(
            "NSGA-II metadata row count {} does not match combined row count {combined_len}",
            metadata.len()
        );
    }
    for &combined_row in rows {
        if combined_row >= combined_len {
            bail!(
                "NSGA-II selected combined row {combined_row} is out of range for {combined_len} rows"
            );
        }
        if combined_row >= parent.len() {
            let offspring_row =
                checked_difference(combined_row, parent.len(), "NSGA-II offspring row offset")?;
            if offspring_row >= offspring.len() {
                bail!(
                    "NSGA-II selected offspring row {offspring_row} is out of range for {} rows",
                    offspring.len()
                );
            }
        }
    }
    out.resize_reuse(
        rows.len(),
        parent.variable_count(),
        parent.objective_count(),
    )?;
    let parent_len = parent.len();
    for (dst, &combined_row) in rows.iter().enumerate() {
        if combined_row < parent_len {
            out.copy_row_from(dst, parent, combined_row)?;
        } else {
            let offspring_row =
                checked_difference(combined_row, parent_len, "NSGA-II offspring row offset")?;
            out.copy_row_from(dst, offspring, offspring_row)?;
        }
        let rank = *metadata.ranks.get(combined_row).ok_or_else(|| {
            anyhow::anyhow!("NSGA-II metadata rank row {combined_row} is missing")
        })?;
        let crowding = *metadata.crowding.get(combined_row).ok_or_else(|| {
            anyhow::anyhow!("NSGA-II metadata crowding row {combined_row} is missing")
        })?;
        *out.ranks
            .get_mut(dst)
            .ok_or_else(|| anyhow::anyhow!("NSGA-II output rank row {dst} is missing"))? = rank;
        *out.crowding
            .get_mut(dst)
            .ok_or_else(|| anyhow::anyhow!("NSGA-II output crowding row {dst} is missing"))? =
            crowding;
    }
    out.validate_shape()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oxymoo::VariableKind;
    use std::collections::TryReserveError;

    struct UnitProblem;

    #[test]
    fn reserved_vector_allocation_retains_try_reserve_cause() -> Result<()> {
        let error = reserved_vec::<u8>(usize::MAX, "test vector")
            .err()
            .ok_or_else(|| {
                anyhow::anyhow!("impossible vector allocation unexpectedly succeeded")
            })?;
        if error.downcast_ref::<TryReserveError>().is_none() {
            bail!("vector allocation error lost its TryReserveError cause");
        }
        if error.to_string() != format!("test vector allocation for {} values failed", usize::MAX) {
            bail!("vector allocation error lost its outer context");
        }
        Ok(())
    }

    impl Problem for UnitProblem {
        fn variable_specs(&self) -> &[VariableSpec] {
            const VARIABLES: [VariableSpec; 1] =
                [VariableSpec::new(0.0, 1.0, VariableKind::Continuous)];
            &VARIABLES
        }

        fn objective_count(&self) -> usize {
            1
        }

        fn evaluate(&self, decision: &[f64], objectives: &mut [f64]) -> Result<f64> {
            let value = *decision
                .first()
                .ok_or_else(|| anyhow::anyhow!("unit problem decision is missing"))?;
            let objective = objectives
                .first_mut()
                .ok_or_else(|| anyhow::anyhow!("unit problem objective is missing"))?;
            *objective = value;
            Ok(0.0)
        }
    }

    #[test]
    fn full_generation_budget_overflow_has_typed_cause() -> Result<()> {
        let mut optimizer = Nsga2::new(
            UnitProblem,
            Nsga2Config {
                population_size: 2,
                max_evaluations: Some(usize::MAX),
                ..Nsga2Config::default()
            },
        )?;
        optimizer.evaluations = usize::MAX;

        let error = optimizer
            .step()
            .err()
            .ok_or_else(|| anyhow::anyhow!("overflowing generation budget unexpectedly stopped"))?;
        if error
            .downcast_ref::<crate::oxymoo::ArithmeticOverflow>()
            .is_none()
        {
            bail!("generation-budget overflow lacks ArithmeticOverflow cause");
        }
        Ok(())
    }

    #[test]
    fn split_front_selection_matches_full_sort_order() -> Result<()> {
        let fronts = vec![vec![0, 1], vec![2, 3, 4, 5, 6]];
        let crowding = vec![0.0, 0.0, 0.5, 2.0, 2.0, 1.0, 0.5];
        let mut selected = Vec::new();
        let mut split_order = Vec::new();

        let split_rank =
            select_survivor_rows_into(&fronts, &crowding, 5, &mut selected, &mut split_order)?;

        if split_rank != Some(1) {
            bail!("test survivor split rank differs from the full-sort reference");
        }
        if selected != vec![0, 1, 3, 4, 5] {
            bail!("test survivor rows differ from the full-sort reference");
        }
        Ok(())
    }

    #[test]
    fn two_source_survivor_gather_preserves_combined_order_and_metadata() -> Result<()> {
        let mut parent = PopulationSnapshot::empty(2, 1, 2)?;
        parent.decisions = vec![0.1, 0.9];
        parent.objectives = vec![0.1, 0.9, 0.9, 0.1];
        parent.constraint_violations = vec![0.0, 0.0];
        let mut offspring = PopulationSnapshot::empty(2, 1, 2)?;
        offspring.decisions = vec![0.4, 0.6];
        offspring.objectives = vec![0.4, 0.6, 0.6, 0.4];
        offspring.constraint_violations = vec![0.0, 0.0];
        let mut metadata = PopulationSnapshot::empty(4, 0, 2)?;
        metadata.ranks = vec![0, 1, 0, 2];
        metadata.crowding = vec![1.0, 2.0, 3.0, 4.0];
        let rows = vec![2, 0, 3];
        let mut out = PopulationSnapshot::empty(0, 0, 0)?;

        select_parent_offspring_rows_with_metadata_into(
            &parent, &offspring, &metadata, &rows, &mut out,
        )?;

        if out.decisions != vec![0.4, 0.1, 0.6] {
            bail!("test survivor decision gather differs from the reference");
        }
        if out.objectives != vec![0.4, 0.6, 0.1, 0.9, 0.6, 0.4] {
            bail!("test survivor objective gather differs from the reference");
        }
        if out.constraint_violations != vec![0.0, 0.0, 0.0] {
            bail!("test survivor constraint gather differs from the reference");
        }
        if out.ranks != vec![0, 0, 2] {
            bail!("test survivor rank gather differs from the reference");
        }
        if out.crowding != vec![3.0, 1.0, 4.0] {
            bail!("test survivor crowding gather differs from the reference");
        }
        Ok(())
    }
}
