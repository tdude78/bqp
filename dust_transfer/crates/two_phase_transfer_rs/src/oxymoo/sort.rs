use crate::oxymoo::validation::{
    checked_difference, checked_product, checked_sum, validate_constraint_vector,
    validate_objective_matrix,
};
use crate::oxymoo::{PopulationSnapshot, SortConfig};
use anyhow::{bail, Result};
use std::cmp::Ordering;

const NO_EDGE: usize = usize::MAX;

#[derive(Default)]
pub struct SortWorkspace {
    first_dominated: Vec<usize>,
    dominated_targets: Vec<usize>,
    next_dominated: Vec<usize>,
    dominated_by_count: Vec<usize>,
    current_front: Vec<usize>,
    next_front: Vec<usize>,
    front_pool: Vec<Vec<usize>>,
    crowding_order: Vec<CrowdingPoint>,
    crowding_distances: Vec<f64>,
    two_objective_rank_tails: Vec<f64>,
    two_objective_feasible: Vec<TwoObjectivePoint>,
    two_objective_infeasible: Vec<TwoObjectivePoint>,
    rank_counts: Vec<usize>,
    rank_offsets: Vec<usize>,
    selection_order: Vec<CrowdingPoint>,
}

#[derive(Clone, Copy)]
struct CrowdingPoint {
    index: usize,
    value: f64,
}

#[derive(Clone, Copy)]
struct TwoObjectivePoint {
    row: usize,
    first: f64,
    second: f64,
    constraint_violation: f64,
}

impl SortWorkspace {
    fn prepare_sort(&mut self, n_individuals: usize) {
        self.first_dominated.resize(n_individuals, NO_EDGE);
        self.first_dominated.fill(NO_EDGE);
        self.dominated_targets.clear();
        self.next_dominated.clear();
        self.dominated_by_count.resize(n_individuals, 0);
        self.dominated_by_count.fill(0);
        self.current_front.clear();
        self.next_front.clear();
    }

    fn recycle_fronts(&mut self, fronts: &mut Vec<Vec<usize>>) {
        for mut front in fronts.drain(..) {
            front.clear();
            self.front_pool.push(front);
        }
    }

    fn push_front_from(&mut self, fronts: &mut Vec<Vec<usize>>, rows: &[usize]) {
        let mut front = self.front_pool.pop().unwrap_or_default();
        front.clear();
        front.extend_from_slice(rows);
        fronts.push(front);
    }
}

/// Sort a validated objective matrix into deterministic constrained Pareto fronts.
///
/// # Errors
///
/// Returns an error when the objective or constraint shapes are invalid, values
/// are non-finite, or either configured tolerance is invalid.
///
/// The allocating wrapper. The solver calls `fast_nondominated_sort_limited_into`
/// with a reused workspace; this exists as the reference the tests and the
/// Criterion harness compare that against, and compiles only for them.
#[cfg(any(test, feature = "bench-internal"))]
pub fn fast_nondominated_sort(
    objectives: &[f64],
    n_individuals: usize,
    n_objectives: usize,
    constraint_violations: &[f64],
    config: SortConfig,
) -> Result<Vec<Vec<usize>>> {
    let mut workspace = SortWorkspace::default();
    let mut fronts = Vec::new();
    fast_nondominated_sort_limited_into(
        objectives,
        n_individuals,
        n_objectives,
        constraint_violations,
        config,
        &mut workspace,
        &mut fronts,
        None,
    )?;
    Ok(fronts)
}

fn fast_nondominated_sort_limited_into(
    objectives: &[f64],
    n_individuals: usize,
    n_objectives: usize,
    constraint_violations: &[f64],
    config: SortConfig,
    workspace: &mut SortWorkspace,
    fronts: &mut Vec<Vec<usize>>,
    capacity: Option<usize>,
) -> Result<()> {
    validate_objective_matrix(objectives, n_individuals, n_objectives)?;
    validate_constraint_vector(constraint_violations, n_individuals)?;
    validate_sort_config(config)?;

    let two_objective_fast_capacity = two_objective_fast_path_capacity(n_individuals, capacity);
    if n_objectives == 2
        && exactly_equal(config.objective_tolerance, 0.0)
        && two_objective_fast_path_is_finite(objectives, n_individuals, constraint_violations)
    {
        fast_two_objective_capped_sort_into(
            objectives,
            n_individuals,
            constraint_violations,
            config,
            workspace,
            fronts,
            two_objective_fast_capacity,
        )?;
        return Ok(());
    }

    generic_nondominated_sort_limited_into(
        objectives,
        n_individuals,
        n_objectives,
        constraint_violations,
        config,
        workspace,
        fronts,
        capacity,
    )
}

fn validate_sort_config(config: SortConfig) -> Result<()> {
    if !config.constraint_tolerance.is_finite() || config.constraint_tolerance < 0.0 {
        bail!(
            "constraint_tolerance must be finite and non-negative: {}",
            config.constraint_tolerance
        );
    }
    if !config.objective_tolerance.is_finite() || config.objective_tolerance < 0.0 {
        bail!(
            "objective_tolerance must be finite and non-negative: {}",
            config.objective_tolerance
        );
    }
    Ok(())
}

fn exactly_equal(left: f64, right: f64) -> bool {
    matches!(left.partial_cmp(&right), Some(Ordering::Equal))
}

fn generic_nondominated_sort_limited_into(
    objectives: &[f64],
    n_individuals: usize,
    n_objectives: usize,
    constraint_violations: &[f64],
    config: SortConfig,
    workspace: &mut SortWorkspace,
    fronts: &mut Vec<Vec<usize>>,
    capacity: Option<usize>,
) -> Result<()> {
    workspace.prepare_sort(n_individuals);
    workspace.recycle_fronts(fronts);
    let mut covered = 0usize;

    let objective_rows = objectives.chunks_exact(n_objectives);
    for (left, (left_objectives, &left_constraint_violation)) in objective_rows
        .clone()
        .zip(constraint_violations.iter())
        .enumerate()
    {
        let first_right = checked_sum(left, 1, "nondominated sort row offset")?;
        for (right, (right_objectives, &right_constraint_violation)) in objective_rows
            .clone()
            .zip(constraint_violations.iter())
            .enumerate()
            .skip(first_right)
        {
            match constrained_dominance(
                left_objectives,
                right_objectives,
                n_objectives,
                left_constraint_violation,
                right_constraint_violation,
                config,
            )? {
                Dominance::Left => {
                    push_dominated_edge(
                        left,
                        right,
                        &mut workspace.first_dominated,
                        &mut workspace.dominated_targets,
                        &mut workspace.next_dominated,
                    )?;
                    let dominated_by_count = workspace
                        .dominated_by_count
                        .get_mut(right)
                        .ok_or_else(|| anyhow::anyhow!("missing dominated row {right}"))?;
                    *dominated_by_count =
                        checked_sum(*dominated_by_count, 1, "nondominated domination count")?;
                }
                Dominance::Right => {
                    push_dominated_edge(
                        right,
                        left,
                        &mut workspace.first_dominated,
                        &mut workspace.dominated_targets,
                        &mut workspace.next_dominated,
                    )?;
                    let dominated_by_count = workspace
                        .dominated_by_count
                        .get_mut(left)
                        .ok_or_else(|| anyhow::anyhow!("missing dominated row {left}"))?;
                    *dominated_by_count =
                        checked_sum(*dominated_by_count, 1, "nondominated domination count")?;
                }
                Dominance::Neither => {}
            }
        }
    }

    workspace.current_front.extend(
        workspace
            .dominated_by_count
            .iter()
            .enumerate()
            .filter_map(|(row, &count)| (count == 0).then_some(row)),
    );

    while !workspace.current_front.is_empty() {
        workspace.current_front.sort_unstable();
        workspace.next_front.clear();

        for &parent in &workspace.current_front {
            let mut edge = *workspace
                .first_dominated
                .get(parent)
                .ok_or_else(|| anyhow::anyhow!("missing dominated edge source {parent}"))?;
            while edge != NO_EDGE {
                let child = *workspace
                    .dominated_targets
                    .get(edge)
                    .ok_or_else(|| anyhow::anyhow!("missing dominated edge {edge}"))?;
                let dominated_by_count = workspace
                    .dominated_by_count
                    .get_mut(child)
                    .ok_or_else(|| anyhow::anyhow!("missing dominated row {child}"))?;
                *dominated_by_count =
                    checked_difference(*dominated_by_count, 1, "nondominated domination count")?;
                if *dominated_by_count == 0 {
                    workspace.next_front.push(child);
                }
                edge = *workspace
                    .next_dominated
                    .get(edge)
                    .ok_or_else(|| anyhow::anyhow!("missing next dominated edge {edge}"))?;
            }
        }
        let current = std::mem::take(&mut workspace.current_front);
        covered = checked_sum(covered, current.len(), "nondominated front coverage")?;
        workspace.push_front_from(fronts, &current);
        if capacity.is_some_and(|limit| covered >= limit) {
            workspace.current_front = current;
            break;
        }
        workspace.current_front = current;
        std::mem::swap(&mut workspace.current_front, &mut workspace.next_front);
    }

    Ok(())
}

fn two_objective_fast_path_is_finite(
    objectives: &[f64],
    n_individuals: usize,
    constraint_violations: &[f64],
) -> bool {
    let Some(objective_len) = n_individuals.checked_mul(2) else {
        return false;
    };
    objectives.len() == objective_len
        && constraint_violations.len() == n_individuals
        && objectives
            .chunks_exact(2)
            .zip(constraint_violations.iter())
            .all(|(objective_row, constraint_violation)| {
                constraint_violation.is_finite()
                    && objective_row.iter().all(|objective| objective.is_finite())
            })
}

fn two_objective_fast_path_capacity(n_individuals: usize, capacity: Option<usize>) -> usize {
    capacity.unwrap_or(n_individuals)
}

fn fast_two_objective_capped_sort_into(
    objectives: &[f64],
    n_individuals: usize,
    constraint_violations: &[f64],
    config: SortConfig,
    workspace: &mut SortWorkspace,
    fronts: &mut Vec<Vec<usize>>,
    capacity: usize,
) -> Result<()> {
    workspace.prepare_sort(n_individuals);
    workspace.recycle_fronts(fronts);
    if capacity == 0 || n_individuals == 0 {
        return Ok(());
    }

    workspace.two_objective_feasible.clear();
    workspace.two_objective_infeasible.clear();
    for (row, (objective_row, &constraint_violation)) in objectives
        .chunks_exact(2)
        .zip(constraint_violations.iter())
        .enumerate()
    {
        let (first, second) = two_objective_values(objective_row)?;
        let point = TwoObjectivePoint {
            row,
            first,
            second,
            constraint_violation,
        };
        if constraint_violation <= config.constraint_tolerance {
            workspace.two_objective_feasible.push(point);
        } else {
            workspace.two_objective_infeasible.push(point);
        }
    }

    {
        let (feasible, infeasible, rank_tails, ranks) = (
            &mut workspace.two_objective_feasible,
            &mut workspace.two_objective_infeasible,
            &mut workspace.two_objective_rank_tails,
            &mut workspace.dominated_by_count,
        );
        ranks.fill(usize::MAX);
        feasible.sort_unstable_by(|left, right| {
            left.first
                .total_cmp(&right.first)
                .then_with(|| left.second.total_cmp(&right.second))
                .then_with(|| left.row.cmp(&right.row))
        });
        rank_tails.clear();

        let mut remaining = feasible.as_slice();
        while let Some((point, tail)) = remaining.split_first() {
            let duplicate_count = tail
                .iter()
                .take_while(|other| {
                    exactly_equal(other.first, point.first)
                        && exactly_equal(other.second, point.second)
                })
                .count();
            let group_len = checked_sum(duplicate_count, 1, "two-objective duplicate group")?;
            let group = remaining
                .get(..group_len)
                .ok_or_else(|| anyhow::anyhow!("two-objective duplicate group is out of bounds"))?;
            let next = remaining.get(group_len..).ok_or_else(|| {
                anyhow::anyhow!("two-objective duplicate group tail is out of bounds")
            })?;

            let rank = rank_tails.partition_point(|tail| *tail <= point.second);
            if rank == rank_tails.len() {
                rank_tails.push(point.second);
            } else {
                let rank_tail = rank_tails
                    .get_mut(rank)
                    .ok_or_else(|| anyhow::anyhow!("missing two-objective rank tail {rank}"))?;
                *rank_tail = (*rank_tail).min(point.second);
            }
            for member in group {
                let rank_slot = ranks.get_mut(member.row).ok_or_else(|| {
                    anyhow::anyhow!("missing two-objective rank row {}", member.row)
                })?;
                *rank_slot = rank;
            }
            remaining = next;
        }

        let mut current_infeasible_rank = rank_tails.len();
        infeasible.sort_unstable_by(|left, right| {
            left.constraint_violation
                .total_cmp(&right.constraint_violation)
                .then_with(|| left.row.cmp(&right.row))
        });
        let mut previous_constraint_violation = None;
        for point in infeasible.iter() {
            if previous_constraint_violation
                .is_some_and(|previous| !exactly_equal(point.constraint_violation, previous))
            {
                current_infeasible_rank =
                    checked_sum(current_infeasible_rank, 1, "two-objective infeasible rank")?;
            }
            let rank_slot = ranks
                .get_mut(point.row)
                .ok_or_else(|| anyhow::anyhow!("missing two-objective rank row {}", point.row))?;
            *rank_slot = current_infeasible_rank;
            previous_constraint_violation = Some(point.constraint_violation);
        }
    }

    let Some(max_rank) = workspace
        .dominated_by_count
        .iter()
        .copied()
        .filter(|rank| *rank != usize::MAX)
        .max()
    else {
        return Ok(());
    };

    let rank_count_len = checked_sum(max_rank, 1, "two-objective rank count length")?;
    workspace.rank_counts.clear();
    workspace.rank_counts.resize(rank_count_len, 0);
    for &rank in &workspace.dominated_by_count {
        if rank != usize::MAX {
            let rank_count = workspace
                .rank_counts
                .get_mut(rank)
                .ok_or_else(|| anyhow::anyhow!("missing two-objective rank count {rank}"))?;
            *rank_count = checked_sum(*rank_count, 1, "two-objective rank count")?;
        }
    }

    let rank_offset_len = checked_sum(max_rank, 2, "two-objective rank offset length")?;
    workspace.rank_offsets.clear();
    workspace.rank_offsets.resize(rank_offset_len, 0);
    for (rank, &rank_count) in workspace.rank_counts.iter().enumerate() {
        let offset = *workspace
            .rank_offsets
            .get(rank)
            .ok_or_else(|| anyhow::anyhow!("missing two-objective rank offset {rank}"))?;
        let next_offset = checked_sum(offset, rank_count, "two-objective rank offset")?;
        let next_rank = checked_sum(rank, 1, "two-objective rank offset index")?;
        let next_offset_slot = workspace
            .rank_offsets
            .get_mut(next_rank)
            .ok_or_else(|| anyhow::anyhow!("missing two-objective rank offset {next_rank}"))?;
        *next_offset_slot = next_offset;
    }

    let total = *workspace
        .rank_offsets
        .last()
        .ok_or_else(|| anyhow::anyhow!("missing two-objective final rank offset"))?;
    workspace.current_front.clear();
    workspace.current_front.resize(total, 0);
    workspace.next_front.clear();
    workspace
        .next_front
        .extend(workspace.rank_offsets.iter().take(rank_count_len).copied());
    {
        let (ranks, current_front, next_front) = (
            &workspace.dominated_by_count,
            &mut workspace.current_front,
            &mut workspace.next_front,
        );
        for (row, &rank) in ranks.iter().enumerate() {
            if rank == usize::MAX {
                continue;
            }
            let slot = *next_front
                .get(rank)
                .ok_or_else(|| anyhow::anyhow!("missing two-objective front offset {rank}"))?;
            let row_slot = current_front
                .get_mut(slot)
                .ok_or_else(|| anyhow::anyhow!("missing two-objective front slot {slot}"))?;
            *row_slot = row;
            let next_slot = checked_sum(slot, 1, "two-objective front slot")?;
            let next_front_slot = next_front
                .get_mut(rank)
                .ok_or_else(|| anyhow::anyhow!("missing two-objective front offset {rank}"))?;
            *next_front_slot = next_slot;
        }
    }

    let mut covered = 0usize;
    let (rank_offsets, current_front, front_pool) = (
        &workspace.rank_offsets,
        &workspace.current_front,
        &mut workspace.front_pool,
    );
    for (&start, &stop) in rank_offsets.iter().zip(rank_offsets.iter().skip(1)) {
        if start == stop {
            continue;
        }
        let rows = current_front
            .get(start..stop)
            .ok_or_else(|| anyhow::anyhow!("two-objective front range is out of bounds"))?;
        let mut front = front_pool.pop().unwrap_or_default();
        front.clear();
        front.extend_from_slice(rows);
        covered = checked_sum(covered, front.len(), "two-objective front coverage")?;
        fronts.push(front);
        if covered >= capacity {
            break;
        }
    }
    Ok(())
}

fn two_objective_values(objective_row: &[f64]) -> Result<(f64, f64)> {
    let mut objectives = objective_row.iter().copied();
    let first = objectives
        .next()
        .ok_or_else(|| anyhow::anyhow!("two-objective row is missing objective 0"))?;
    let second = objectives
        .next()
        .ok_or_else(|| anyhow::anyhow!("two-objective row is missing objective 1"))?;
    if objectives.next().is_some() {
        bail!("two-objective row has more than two objectives");
    }
    Ok((first, second))
}

fn push_dominated_edge(
    source: usize,
    target: usize,
    first_dominated: &mut [usize],
    dominated_targets: &mut Vec<usize>,
    next_dominated: &mut Vec<usize>,
) -> Result<()> {
    let previous_edge = *first_dominated
        .get(source)
        .ok_or_else(|| anyhow::anyhow!("missing dominated edge source {source}"))?;
    let edge = dominated_targets.len();
    if edge == NO_EDGE {
        bail!("nondominated edge count exceeds usize");
    }
    dominated_targets.push(target);
    next_dominated.push(previous_edge);
    let source_edge = first_dominated
        .get_mut(source)
        .ok_or_else(|| anyhow::anyhow!("missing dominated edge source {source}"))?;
    *source_edge = edge;
    Ok(())
}

/// Compute NSGA-II crowding distances for one valid front.
///
/// # Errors
///
/// Returns an error when the objective matrix is malformed, contains a
/// non-finite value, or a front member is outside the population.
///
/// Allocating wrapper over `crowding_distance_into`, kept for the same reason
/// as `fast_nondominated_sort` above and compiled in the same configurations.
#[cfg(any(test, feature = "bench-internal"))]
pub fn crowding_distance(
    objectives: &[f64],
    n_individuals: usize,
    n_objectives: usize,
    front: &[usize],
) -> Result<Vec<f64>> {
    validate_objective_matrix(objectives, n_individuals, n_objectives)?;
    validate_front_indices(front, n_individuals)?;
    let mut ordered = Vec::new();
    let mut distances = Vec::new();
    crowding_distance_into(
        objectives,
        n_individuals,
        n_objectives,
        front,
        &mut ordered,
        &mut distances,
    )?;
    Ok(distances)
}

fn crowding_distance_into(
    objectives: &[f64],
    n_individuals: usize,
    n_objectives: usize,
    front: &[usize],
    ordered: &mut Vec<CrowdingPoint>,
    distances: &mut Vec<f64>,
) -> Result<()> {
    let n_points = front.len();
    distances.clear();
    distances.resize(n_points, 0.0);
    if n_points == 0 {
        return Ok(());
    }
    if n_points <= 2 {
        distances.fill(f64::INFINITY);
        return Ok(());
    }

    ordered.clear();
    ordered.extend(
        front
            .iter()
            .enumerate()
            .map(|(index, _)| CrowdingPoint { index, value: 0.0 }),
    );
    for obj in 0..n_objectives {
        for point in ordered.iter_mut() {
            let global = *front
                .get(point.index)
                .ok_or_else(|| anyhow::anyhow!("missing crowding front point {}", point.index))?;
            let objective_row = matrix_row(
                objectives,
                n_individuals,
                n_objectives,
                global,
                "crowding objective",
            )?;
            point.value = *objective_row
                .get(obj)
                .ok_or_else(|| anyhow::anyhow!("missing crowding objective {obj}"))?;
        }
        ordered.sort_by(|left, right| left.value.total_cmp(&right.value));

        let first = *ordered
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing first crowding point"))?;
        let last = *ordered
            .last()
            .ok_or_else(|| anyhow::anyhow!("missing last crowding point"))?;
        let first_distance = distances
            .get_mut(first.index)
            .ok_or_else(|| anyhow::anyhow!("missing first crowding distance"))?;
        *first_distance = f64::INFINITY;
        let last_distance = distances
            .get_mut(last.index)
            .ok_or_else(|| anyhow::anyhow!("missing last crowding distance"))?;
        *last_distance = f64::INFINITY;

        let denominator = last.value - first.value;
        if denominator <= 1e-14 * first.value.abs().max(last.value.abs()).max(1.0) {
            continue;
        }

        for points in ordered.windows(3) {
            let [previous, current, next] = points else {
                continue;
            };
            let distance = distances
                .get_mut(current.index)
                .ok_or_else(|| anyhow::anyhow!("missing crowding distance"))?;
            if distance.is_infinite() {
                continue;
            }
            *distance += (next.value - previous.value) / denominator;
        }
    }

    Ok(())
}

fn validate_front_indices(front: &[usize], n_individuals: usize) -> Result<()> {
    for &index in front {
        if index >= n_individuals {
            bail!("front index {index} is out of bounds for population size {n_individuals}");
        }
    }
    Ok(())
}

fn matrix_row<'a>(
    values: &'a [f64],
    n_individuals: usize,
    width: usize,
    row: usize,
    label: &'static str,
) -> Result<&'a [f64]> {
    if row >= n_individuals {
        bail!("{label} row {row} is out of bounds for population size {n_individuals}");
    }
    let start = checked_product(row, width, "matrix row offset")?;
    let end = checked_sum(start, width, "matrix row end")?;
    values
        .get(start..end)
        .ok_or_else(|| anyhow::anyhow!("{label} row {row} is out of bounds"))
}

pub fn assign_rank_and_crowding_into(
    population: &mut PopulationSnapshot,
    config: SortConfig,
    workspace: &mut SortWorkspace,
    fronts: &mut Vec<Vec<usize>>,
) -> Result<()> {
    population.validate_shape()?;
    fast_nondominated_sort_limited_into(
        &population.objectives,
        population.len(),
        population.objective_count(),
        &population.constraint_violations,
        config,
        workspace,
        fronts,
        None,
    )?;
    population.ranks.fill(usize::MAX);
    population.crowding.fill(0.0);

    for (rank, front) in fronts.iter().enumerate() {
        validate_front_indices(front, population.len())?;
        for &index in front {
            let rank_slot = population
                .ranks
                .get_mut(index)
                .ok_or_else(|| anyhow::anyhow!("missing rank row {index}"))?;
            *rank_slot = rank;
        }
        assign_crowding_to_front(
            population,
            front,
            &mut workspace.crowding_order,
            &mut workspace.crowding_distances,
        )?;
    }
    Ok(())
}

pub fn assign_capped_survivor_selection_into(
    population: &mut PopulationSnapshot,
    config: SortConfig,
    population_size: usize,
    workspace: &mut SortWorkspace,
    fronts: &mut Vec<Vec<usize>>,
    selected: &mut Vec<usize>,
    split_order: &mut Vec<usize>,
) -> Result<Option<usize>> {
    population.validate_shape()?;
    fast_nondominated_sort_limited_into(
        &population.objectives,
        population.len(),
        population.objective_count(),
        &population.constraint_violations,
        config,
        workspace,
        fronts,
        Some(population_size),
    )?;
    population.ranks.fill(usize::MAX);
    population.crowding.fill(0.0);
    selected.clear();
    let mut split_rank = None;

    for (rank, front) in fronts.iter().enumerate() {
        validate_front_indices(front, population.len())?;
        for &index in front {
            let rank_slot = population
                .ranks
                .get_mut(index)
                .ok_or_else(|| anyhow::anyhow!("missing rank row {index}"))?;
            *rank_slot = rank;
        }
        assign_crowding_to_front(
            population,
            front,
            &mut workspace.crowding_order,
            &mut workspace.crowding_distances,
        )?;
        let selected_with_front =
            checked_sum(selected.len(), front.len(), "survivor selection coverage")?;
        if selected_with_front <= population_size {
            selected.extend_from_slice(front);
            continue;
        }
        let remaining = checked_difference(
            population_size,
            selected.len(),
            "survivor selection remaining capacity",
        )?;
        split_order.clear();
        split_order.extend_from_slice(front);
        workspace.selection_order.clear();
        for &index in split_order.iter() {
            let crowding = *population
                .crowding
                .get(index)
                .ok_or_else(|| anyhow::anyhow!("missing crowding row {index}"))?;
            workspace.selection_order.push(CrowdingPoint {
                index,
                value: crowding,
            });
        }
        let by_crowding_desc = |left: &CrowdingPoint, right: &CrowdingPoint| {
            right
                .value
                .total_cmp(&left.value)
                .then_with(|| left.index.cmp(&right.index))
        };
        if remaining < workspace.selection_order.len() {
            let (top, _, _) = workspace
                .selection_order
                .select_nth_unstable_by(remaining, by_crowding_desc);
            // perf-hunt-r2 #5b: comparator is a strict total order (index
            // tie-break), so unstable sort is mathematically identical.
            top.sort_unstable_by(by_crowding_desc);
            selected.extend(top.iter().map(|point| point.index));
        } else {
            workspace.selection_order.sort_unstable_by(by_crowding_desc);
            selected.extend(workspace.selection_order.iter().map(|point| point.index));
        }
        split_rank = Some(rank);
        break;
    }
    Ok(split_rank)
}

pub fn rebuild_fronts_from_ranks(
    ranks: &[usize],
    n_individuals: usize,
    fronts: &mut Vec<Vec<usize>>,
    workspace: &mut SortWorkspace,
) -> Result<()> {
    validate_rank_vector(ranks, n_individuals)?;
    workspace.recycle_fronts(fronts);
    let max_rank = ranks
        .iter()
        .take(n_individuals)
        .copied()
        .filter(|rank| *rank != usize::MAX)
        .max();
    let Some(max_rank) = max_rank else {
        return Ok(());
    };
    let front_count = checked_sum(max_rank, 1, "front count")?;
    for _ in 0..front_count {
        let front = workspace.front_pool.pop().unwrap_or_default();
        fronts.push(front);
    }
    for front in fronts.iter_mut() {
        front.clear();
    }
    for (row, &rank) in ranks.iter().enumerate() {
        if rank != usize::MAX {
            let front = fronts
                .get_mut(rank)
                .ok_or_else(|| anyhow::anyhow!("missing front rank {rank}"))?;
            front.push(row);
        }
    }
    Ok(())
}

pub fn recompute_crowding_for_front(
    population: &mut PopulationSnapshot,
    front: &[usize],
    workspace: &mut SortWorkspace,
) -> Result<()> {
    population.validate_shape()?;
    validate_front_indices(front, population.len())?;
    validate_objective_matrix(
        &population.objectives,
        population.len(),
        population.objective_count(),
    )?;
    for &row in front {
        let crowding = population
            .crowding
            .get_mut(row)
            .ok_or_else(|| anyhow::anyhow!("missing crowding row {row}"))?;
        *crowding = 0.0;
    }
    assign_crowding_to_front(
        population,
        front,
        &mut workspace.crowding_order,
        &mut workspace.crowding_distances,
    )
}

fn assign_crowding_to_front(
    population: &mut PopulationSnapshot,
    front: &[usize],
    ordered: &mut Vec<CrowdingPoint>,
    distances: &mut Vec<f64>,
) -> Result<()> {
    validate_front_indices(front, population.len())?;
    let n_objectives = population.objective_count();
    crowding_distance_into(
        &population.objectives,
        population.len(),
        n_objectives,
        front,
        ordered,
        distances,
    )?;
    for (&row, &distance) in front.iter().zip(distances.iter()) {
        let crowding = population
            .crowding
            .get_mut(row)
            .ok_or_else(|| anyhow::anyhow!("missing crowding row {row}"))?;
        *crowding = distance;
    }
    Ok(())
}

fn validate_rank_vector(ranks: &[usize], n_individuals: usize) -> Result<()> {
    if ranks.len() != n_individuals {
        bail!(
            "rank length {} does not match population size {n_individuals}",
            ranks.len()
        );
    }
    for (row, &rank) in ranks.iter().enumerate() {
        if rank != usize::MAX && rank >= n_individuals {
            bail!("rank {rank} at row {row} is out of bounds for population size {n_individuals}");
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Dominance {
    Left,
    Right,
    Neither,
}

// perf-hunt-r2 #6 (2026-07-08): inline hint — called O(n^2)/generation;
// thin-LTO cgu=16 can otherwise split it from the caller unit.
#[inline]
fn constrained_dominance(
    left_objectives: &[f64],
    right_objectives: &[f64],
    n_objectives: usize,
    left_constraint_violation: f64,
    right_constraint_violation: f64,
    config: SortConfig,
) -> Result<Dominance> {
    let left_feasible = left_constraint_violation <= config.constraint_tolerance;
    let right_feasible = right_constraint_violation <= config.constraint_tolerance;
    match (left_feasible, right_feasible) {
        (true, false) => return Ok(Dominance::Left),
        (false, true) => return Ok(Dominance::Right),
        (false, false) => {
            return Ok(
                match left_constraint_violation.total_cmp(&right_constraint_violation) {
                    Ordering::Less => Dominance::Left,
                    Ordering::Greater => Dominance::Right,
                    Ordering::Equal => Dominance::Neither,
                },
            );
        }
        (true, true) => {}
    }

    let (left_dominates, right_dominates) = if n_objectives == 2 {
        pareto_flags_2(
            left_objectives,
            right_objectives,
            config.objective_tolerance,
        )?
    } else {
        pareto_flags(
            left_objectives,
            right_objectives,
            config.objective_tolerance,
        )
    };
    Ok(match (left_dominates, right_dominates) {
        (true, false) => Dominance::Left,
        (false, true) => Dominance::Right,
        _ => Dominance::Neither,
    })
}

#[inline]
fn pareto_flags_2(left: &[f64], right: &[f64], tolerance: f64) -> Result<(bool, bool)> {
    let mut left_values = left.iter().copied();
    let mut right_values = right.iter().copied();
    let left0 = left_values
        .next()
        .ok_or_else(|| anyhow::anyhow!("two-objective dominance row is missing objective 0"))?;
    let right0 = right_values
        .next()
        .ok_or_else(|| anyhow::anyhow!("two-objective dominance row is missing objective 0"))?;
    let left1 = left_values
        .next()
        .ok_or_else(|| anyhow::anyhow!("two-objective dominance row is missing objective 1"))?;
    let right1 = right_values
        .next()
        .ok_or_else(|| anyhow::anyhow!("two-objective dominance row is missing objective 1"))?;
    if left_values.next().is_some() || right_values.next().is_some() {
        bail!("two-objective dominance received a row with the wrong width");
    }
    let d0 = left0 - right0;
    let d1 = left1 - right1;
    let left_dominates = d0 <= tolerance && d1 <= tolerance && (d0 < -tolerance || d1 < -tolerance);
    let right_dominates =
        d0 >= -tolerance && d1 >= -tolerance && (d0 > tolerance || d1 > tolerance);
    Ok((left_dominates, right_dominates))
}

#[inline]
fn pareto_flags(left: &[f64], right: &[f64], tolerance: f64) -> (bool, bool) {
    let mut left_all_leq = true;
    let mut left_any_lt = false;
    let mut right_all_leq = true;
    let mut right_any_lt = false;
    for (&a, &b) in left.iter().zip(right.iter()) {
        let diff = a - b;
        if diff > tolerance {
            left_all_leq = false;
        }
        if diff < -tolerance {
            left_any_lt = true;
        }
        if -diff > tolerance {
            right_all_leq = false;
        }
        if -diff < -tolerance {
            right_any_lt = true;
        }
    }
    (left_all_leq && left_any_lt, right_all_leq && right_any_lt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_assignment_rejects_malformed_rank_storage_without_panicking() -> Result<()> {
        let mut population = PopulationSnapshot::empty(2, 1, 2)?;
        population.objectives = vec![1.0, 2.0, 2.0, 1.0];
        population.ranks.clear();
        let mut workspace = SortWorkspace::default();
        let mut fronts = Vec::new();

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assign_rank_and_crowding_into(
                &mut population,
                SortConfig::default(),
                &mut workspace,
                &mut fronts,
            )
        }));

        anyhow::ensure!(
            matches!(outcome, Ok(Err(_))),
            "malformed rank storage did not fail closed without panicking"
        );
        Ok(())
    }

    #[test]
    fn workspace_sort_matches_allocating_sort_and_crowding_assignment() -> Result<()> {
        let objectives = vec![
            1.0, 4.0, //
            2.0, 2.0, //
            4.0, 1.0, //
            3.0, 5.0, //
        ];
        let cvs = vec![0.0; 4];
        let config = SortConfig::default();

        let allocating = fast_nondominated_sort(&objectives, 4, 2, &cvs, config)?;
        let mut workspace = SortWorkspace::default();
        let mut fronts = Vec::new();
        fast_nondominated_sort_limited_into(
            &objectives,
            4,
            2,
            &cvs,
            config,
            &mut workspace,
            &mut fronts,
            None,
        )?;

        anyhow::ensure!(
            fronts == allocating,
            "workspace fronts differ from allocating sort: {fronts:?} != {allocating:?}"
        );

        let mut population = PopulationSnapshot::empty(4, 1, 2)?;
        population.objectives = objectives;
        let mut assigned_fronts = Vec::new();
        assign_rank_and_crowding_into(
            &mut population,
            config,
            &mut workspace,
            &mut assigned_fronts,
        )?;

        anyhow::ensure!(
            assigned_fronts == allocating,
            "assigned fronts differ from allocating sort: {assigned_fronts:?} != {allocating:?}"
        );
        anyhow::ensure!(
            population.ranks.first() == Some(&0),
            "first population rank was not zero"
        );
        anyhow::ensure!(
            population
                .crowding
                .first()
                .is_some_and(|value| value.is_infinite()),
            "first population crowding distance was not infinite"
        );
        Ok(())
    }

    #[test]
    fn crowding_distance_preserves_stable_tie_order_between_objectives() -> Result<()> {
        let objectives = vec![
            2.0, 0.0, //
            1.0, 0.0, //
            3.0, 1.0, //
            4.0, 2.0, //
        ];
        let distances = crowding_distance(&objectives, 4, 2, &[0, 1, 2, 3])?;
        let first = *distances
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing first crowding distance"))?;
        let second = *distances
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("missing second crowding distance"))?;
        let third = *distances
            .get(2)
            .ok_or_else(|| anyhow::anyhow!("missing third crowding distance"))?;
        let fourth = *distances
            .get(3)
            .ok_or_else(|| anyhow::anyhow!("missing fourth crowding distance"))?;

        anyhow::ensure!(first.is_finite(), "first tied point became a boundary");
        anyhow::ensure!(second.is_infinite(), "second tied point lost its boundary");
        anyhow::ensure!(
            third.is_finite(),
            "third point unexpectedly became a boundary"
        );
        anyhow::ensure!(fourth.is_infinite(), "fourth point lost its boundary");
        anyhow::ensure!(
            first < third,
            "stable tied-point ordering was not reflected in crowding: {first} >= {third}"
        );
        Ok(())
    }

    #[test]
    fn capped_two_objective_sort_matches_full_generic_front_prefix() -> Result<()> {
        let objectives = vec![
            2.0, 5.0, //
            1.0, 6.0, //
            3.0, 3.0, //
            4.0, 2.0, //
            5.0, 1.0, //
            2.0, 5.0, //
            6.0, 6.0, //
            0.5, 9.0, //
            7.0, 0.5, //
        ];
        let cv = vec![0.0; 9];
        let config = SortConfig::default();
        let full = fast_nondominated_sort(&objectives, 9, 2, &cv, config)?;
        let mut expected = Vec::new();
        let mut covered = 0usize;
        for front in full {
            covered += front.len();
            expected.push(front);
            if covered >= 5 {
                break;
            }
        }

        let mut workspace = SortWorkspace::default();
        let mut capped = Vec::new();
        fast_nondominated_sort_limited_into(
            &objectives,
            9,
            2,
            &cv,
            config,
            &mut workspace,
            &mut capped,
            Some(5),
        )?;

        anyhow::ensure!(
            capped == expected,
            "capped fronts differ from generic prefix: {capped:?} != {expected:?}"
        );
        Ok(())
    }

    #[test]
    fn capped_two_objective_sort_matches_constraints_and_ties() -> Result<()> {
        let objectives = vec![
            1.0, 4.0, //
            1.0, 4.0, //
            2.0, 3.0, //
            3.0, 2.0, //
            4.0, 1.0, //
            0.0, 0.0, //
            0.0, 0.0, //
        ];
        let cv = vec![0.0, 0.0, 0.0, 0.25, 0.10, 0.25, 0.10];
        let config = SortConfig::default();
        let full = fast_nondominated_sort(&objectives, 7, 2, &cv, config)?;
        let mut expected = Vec::new();
        let mut covered = 0usize;
        for front in full {
            covered += front.len();
            expected.push(front);
            if covered >= 6 {
                break;
            }
        }

        let mut workspace = SortWorkspace::default();
        let mut capped = Vec::new();
        fast_nondominated_sort_limited_into(
            &objectives,
            7,
            2,
            &cv,
            config,
            &mut workspace,
            &mut capped,
            Some(6),
        )?;

        anyhow::ensure!(
            capped == expected,
            "capped fronts differ from generic prefix: {capped:?} != {expected:?}"
        );
        Ok(())
    }

    #[test]
    fn full_two_objective_sort_matches_generic_constraints_and_ties() -> Result<()> {
        let objectives = vec![
            1.0, 4.0, //
            1.0, 4.0, //
            2.0, 3.0, //
            3.0, 2.0, //
            4.0, 1.0, //
            0.0, 0.0, //
            0.0, 0.0, //
            2.0, 2.0, //
            5.0, 5.0, //
        ];
        let cv = vec![0.0, 0.0, 0.0, 0.25, 0.10, 0.25, 0.10, 0.0, 0.0];
        let config = SortConfig::default();

        let mut generic_workspace = SortWorkspace::default();
        let mut generic = Vec::new();
        generic_nondominated_sort_limited_into(
            &objectives,
            9,
            2,
            &cv,
            config,
            &mut generic_workspace,
            &mut generic,
            None,
        )?;

        let mut fast_workspace = SortWorkspace::default();
        let mut fast = Vec::new();
        fast_nondominated_sort_limited_into(
            &objectives,
            9,
            2,
            &cv,
            config,
            &mut fast_workspace,
            &mut fast,
            None,
        )?;

        anyhow::ensure!(
            fast == generic,
            "fast fronts differ from generic fronts: {fast:?} != {generic:?}"
        );
        Ok(())
    }

    #[test]
    fn full_two_objective_sort_matches_generic_rank_deep() -> Result<()> {
        let n = 64usize;
        let mut objectives = Vec::with_capacity(n * 2);
        for idx in 0..n {
            let value = f64::from(u32::try_from(idx)?);
            objectives.push(value);
            objectives.push(value);
        }
        let cv = vec![0.0; n];
        let config = SortConfig::default();

        let mut generic_workspace = SortWorkspace::default();
        let mut generic = Vec::new();
        generic_nondominated_sort_limited_into(
            &objectives,
            n,
            2,
            &cv,
            config,
            &mut generic_workspace,
            &mut generic,
            None,
        )?;

        let mut fast_workspace = SortWorkspace::default();
        let mut fast = Vec::new();
        fast_nondominated_sort_limited_into(
            &objectives,
            n,
            2,
            &cv,
            config,
            &mut fast_workspace,
            &mut fast,
            None,
        )?;

        anyhow::ensure!(
            fast == generic,
            "fast fronts differ from generic fronts: {fast:?} != {generic:?}"
        );
        Ok(())
    }

    #[test]
    fn two_objective_fast_path_rejects_nonfinite_inputs() {
        assert!(!two_objective_fast_path_is_finite(
            &[1.0, f64::NAN, 2.0, 3.0],
            2,
            &[0.0, 0.0],
        ));
        assert!(!two_objective_fast_path_is_finite(
            &[1.0, 2.0, 2.0, 3.0],
            2,
            &[0.0, f64::INFINITY],
        ));
        assert!(two_objective_fast_path_is_finite(
            &[1.0, 2.0, 2.0, 3.0],
            2,
            &[0.0, 0.0],
        ));
    }

    #[test]
    fn two_objective_fast_path_capacity_treats_full_sort_as_capped_at_population() {
        assert_eq!(two_objective_fast_path_capacity(7, None), 7);
        assert_eq!(two_objective_fast_path_capacity(7, Some(3)), 3);
    }

    #[test]
    fn capped_two_objective_sort_matches_rank_deep_generic_prefix() -> Result<()> {
        let n = 64usize;
        let mut objectives = Vec::with_capacity(n * 2);
        for idx in 0..n {
            let value = f64::from(u32::try_from(idx)?);
            objectives.push(value);
            objectives.push(value);
        }
        let cv = vec![0.0; n];
        let config = SortConfig::default();
        let full = fast_nondominated_sort(&objectives, n, 2, &cv, config)?;
        let mut expected = Vec::new();
        let mut covered = 0usize;
        for front in full {
            covered += front.len();
            expected.push(front);
            if covered >= 17 {
                break;
            }
        }

        let mut workspace = SortWorkspace::default();
        let mut capped = Vec::new();
        fast_nondominated_sort_limited_into(
            &objectives,
            n,
            2,
            &cv,
            config,
            &mut workspace,
            &mut capped,
            Some(17),
        )?;

        anyhow::ensure!(
            capped == expected,
            "capped fronts differ from generic prefix: {capped:?} != {expected:?}"
        );
        Ok(())
    }
}
