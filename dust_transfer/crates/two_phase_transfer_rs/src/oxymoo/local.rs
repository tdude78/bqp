use crate::oxymoo::{
    validation::{checked_difference, checked_sum, count_as_f64},
    ArithmeticOverflow,
};
use anyhow::{bail, Context, Result};
use num_traits::ToPrimitive;
use rand::RngExt;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::cmp::Ordering;

const N: usize = 3;
const SIMPLEX: usize = N + 1;
const N_AS_F64: f64 = 3.0;
const SIMPLEX_AS_F64: f64 = 4.0;
const LBFGS_HISTORY: usize = 10;
const PSO_UNSET_PERSONAL_COST: f64 = 1.0e30;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalOptimizerKind {
    NelderMead,
    Pso,
    Lbfgs,
}

impl LocalOptimizerKind {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NelderMead => "nelder_mead",
            Self::Pso => "pso",
            Self::Lbfgs => "lbfgs",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum TuneLevel {
    #[default]
    Default,
    Conservative,
    Aggressive,
}

impl TuneLevel {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Conservative => "conservative",
            Self::Aggressive => "aggressive",
        }
    }
}

/// Historic Nelder-Mead iteration floor.
///
/// `nelder_mead_impl` scales `max_iters` by the tune level's `iters_factor`
/// and then raises the result to at least this many iterations. With
/// `TuneLevel::Default`'s factor of 0.3 that floor swallows every requested
/// `max_iters` below 34, so callers asking for fewer get 10 anyway. Kept as
/// the default so existing call sites are bit-for-bit unchanged; a caller that
/// wants its request honoured sets `min_iters` explicitly.
pub const DEFAULT_NM_MIN_ITERS: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalOptimizerConfig {
    pub kind: LocalOptimizerKind,
    pub max_iters: usize,
    pub tolerance: f64,
    pub seed: u64,
    pub tune: TuneLevel,
    /// Floor applied to the tune-scaled iteration budget. See
    /// [`DEFAULT_NM_MIN_ITERS`]. Nelder-Mead only.
    pub min_iters: usize,
}

impl Default for LocalOptimizerConfig {
    fn default() -> Self {
        Self {
            kind: LocalOptimizerKind::NelderMead,
            max_iters: 128,
            tolerance: 1e-6,
            seed: 0,
            tune: TuneLevel::Default,
            min_iters: DEFAULT_NM_MIN_ITERS,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalOptimizeResult {
    pub x: [f64; N],
    pub cost: f64,
    pub evaluations: u64,
    pub converged: bool,
}

pub trait LocalScalarProblem3 {
    /// Evaluate one point.
    ///
    /// # Errors
    ///
    /// Returns the concrete error reported by the objective.
    fn value(&self, x: &[f64; N]) -> Result<f64>;

    fn value_gradient(&self, x: &[f64; N]) -> Option<(f64, [f64; N])> {
        let _ = x;
        None
    }
}

/// Run the configured local optimizer.
///
/// # Errors
///
/// Returns an error for invalid controls, bounds, or initial point; a
/// non-finite objective; or an unavailable/non-finite L-BFGS gradient.
pub fn run_local_optimizer<P: LocalScalarProblem3>(
    problem: &P,
    lower: [f64; N],
    upper: [f64; N],
    initial: [f64; N],
    config: LocalOptimizerConfig,
) -> Result<LocalOptimizeResult> {
    validate_inputs(lower, upper, initial, config)?;
    let mut initial = initial;
    clamp_to_bounds(&mut initial, &lower, &upper);
    match config.kind {
        LocalOptimizerKind::NelderMead => run_nelder_mead3(problem, lower, upper, initial, config),
        LocalOptimizerKind::Pso => run_pso3(problem, lower, upper, initial, config),
        LocalOptimizerKind::Lbfgs => run_lbfgs3(problem, lower, upper, initial, config),
    }
}

/// Run bounded three-variable Nelder-Mead.
///
/// # Errors
///
/// Returns an error for invalid controls, bounds, or initial point, or a
/// non-finite objective value.
pub fn run_nelder_mead3<P: LocalScalarProblem3>(
    problem: &P,
    lower: [f64; N],
    upper: [f64; N],
    initial: [f64; N],
    config: LocalOptimizerConfig,
) -> Result<LocalOptimizeResult> {
    validate_inputs(lower, upper, initial, config)?;
    count_as_f64(config.max_iters, "local optimizer max_iters")?;
    count_as_f64(config.min_iters, "local optimizer min_iters")?;
    let mut initial = initial;
    clamp_to_bounds(&mut initial, &lower, &upper);
    let tune = NmTuneParams::for_level(config.tune, config.tolerance);
    nelder_mead_impl(
        problem,
        lower,
        upper,
        initial,
        config.max_iters,
        config.min_iters,
        tune,
    )
}

/// Run bounded three-variable particle-swarm optimization.
///
/// # Errors
///
/// Returns an error for invalid controls, bounds, or initial point, or a
/// non-finite objective value.
pub fn run_pso3<P: LocalScalarProblem3>(
    problem: &P,
    lower: [f64; N],
    upper: [f64; N],
    initial: [f64; N],
    config: LocalOptimizerConfig,
) -> Result<LocalOptimizeResult> {
    validate_inputs(lower, upper, initial, config)?;
    let mut initial = initial;
    clamp_to_bounds(&mut initial, &lower, &upper);
    let swarm_size = pso_swarm_size(config.tune);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
    let mut positions = vec![[0.0; N]; swarm_size];
    let mut velocities = vec![[0.0; N]; swarm_size];
    let mut personal_best = vec![[0.0; N]; swarm_size];
    let mut personal_cost = vec![PSO_UNSET_PERSONAL_COST; swarm_size];
    let mut global_best = initial;
    let mut global_cost = eval(problem, &initial)?;
    let mut evaluations = 1_u64;

    let (initial_position, remaining_positions) = positions
        .split_first_mut()
        .ok_or_else(|| anyhow::anyhow!("local optimizer PSO swarm is empty"))?;
    let (_, remaining_velocities) = velocities
        .split_first_mut()
        .ok_or_else(|| anyhow::anyhow!("local optimizer PSO velocity swarm is empty"))?;
    let (initial_personal_best, remaining_personal_best) = personal_best
        .split_first_mut()
        .ok_or_else(|| anyhow::anyhow!("local optimizer PSO personal-best swarm is empty"))?;
    let (initial_personal_cost, remaining_personal_cost) = personal_cost
        .split_first_mut()
        .ok_or_else(|| anyhow::anyhow!("local optimizer PSO personal-cost swarm is empty"))?;
    if remaining_positions.len() != remaining_velocities.len()
        || remaining_positions.len() != remaining_personal_best.len()
        || remaining_positions.len() != remaining_personal_cost.len()
    {
        bail!("local optimizer PSO swarm state has inconsistent lengths");
    }

    *initial_position = initial;
    *initial_personal_best = initial;
    *initial_personal_cost = global_cost;

    for (((position, velocity), best_position), best_cost) in remaining_positions
        .iter_mut()
        .zip(remaining_velocities.iter_mut())
        .zip(remaining_personal_best.iter_mut())
        .zip(remaining_personal_cost.iter_mut())
    {
        for ((position_value, velocity_value), (lower_bound, upper_bound)) in position
            .iter_mut()
            .zip(velocity.iter_mut())
            .zip(lower.iter().zip(upper.iter()))
        {
            let width = *upper_bound - *lower_bound;
            let position_fraction = rng.random_range(0.0..1.0);
            let position_offset = position_fraction * width;
            *position_value = *lower_bound + position_offset;
            let velocity_fraction = rng.random_range(-0.5..0.5);
            let velocity_scale = velocity_fraction * 0.2;
            *velocity_value = velocity_scale * width;
        }
        *best_position = *position;
        let cost = eval(problem, position)?;
        *best_cost = cost;
        evaluations = increment_evaluations(evaluations)?;
        if *best_cost < global_cost {
            global_cost = *best_cost;
            global_best = *best_position;
        }
    }

    let mut stall = 0_usize;
    let stall_limit = ((config.max_iters / 4).max(8)).min(config.max_iters.max(1));
    let inertia = 0.72;
    let cognitive = 1.49;
    let social = 1.49;

    for _ in 0..config.max_iters {
        let before = global_cost;
        for (((position, velocity), best_position), best_cost) in positions
            .iter_mut()
            .zip(velocities.iter_mut())
            .zip(personal_best.iter_mut())
            .zip(personal_cost.iter_mut())
        {
            for (
                (((position_value, velocity_value), best_value), global_best_value),
                (lower_bound, upper_bound),
            ) in position
                .iter_mut()
                .zip(velocity.iter_mut())
                .zip(best_position.iter())
                .zip(global_best.iter())
                .zip(lower.iter().zip(upper.iter()))
            {
                let random_cognitive = rng.random_range(0.0..1.0);
                let random_social = rng.random_range(0.0..1.0);
                let inertia_term = inertia * *velocity_value;
                let cognitive_delta = *best_value - *position_value;
                let cognitive_term = cognitive * random_cognitive * cognitive_delta;
                let social_delta = *global_best_value - *position_value;
                let social_term = social * random_social * social_delta;
                *velocity_value = inertia_term + cognitive_term + social_term;
                let width = *upper_bound - *lower_bound;
                let vmax = 0.5 * width;
                *velocity_value = (*velocity_value).clamp(-vmax, vmax);
                *position_value += *velocity_value;
                if *position_value < *lower_bound {
                    *position_value = *lower_bound + (*lower_bound - *position_value).min(width);
                    *velocity_value *= -0.5;
                } else if *position_value > *upper_bound {
                    *position_value = *upper_bound - (*position_value - *upper_bound).min(width);
                    *velocity_value *= -0.5;
                }
                *position_value = (*position_value).clamp(*lower_bound, *upper_bound);
            }

            let cost = eval(problem, position)?;
            evaluations = increment_evaluations(evaluations)?;
            if cost < *best_cost {
                *best_cost = cost;
                *best_position = *position;
                if cost < global_cost {
                    global_cost = cost;
                    global_best = *position;
                }
            }
        }

        if (before - global_cost).abs() <= config.tolerance.max(0.0) {
            stall = checked_sum(stall, 1, "local optimizer PSO stall count")?;
        } else {
            stall = 0;
        }
        if stall >= stall_limit {
            break;
        }
    }

    Ok(LocalOptimizeResult {
        x: global_best,
        cost: global_cost,
        evaluations,
        converged: global_cost.is_finite() && global_cost < crate::types::INVALID_COST,
    })
}

/// Run bounded three-variable L-BFGS.
///
/// # Errors
///
/// Returns an error for invalid controls, bounds, or initial point; an
/// unavailable gradient; or a non-finite objective or gradient value.
pub fn run_lbfgs3<P: LocalScalarProblem3>(
    problem: &P,
    lower: [f64; N],
    upper: [f64; N],
    initial: [f64; N],
    config: LocalOptimizerConfig,
) -> Result<LocalOptimizeResult> {
    validate_inputs(lower, upper, initial, config)?;
    let mut initial = initial;
    clamp_to_bounds(&mut initial, &lower, &upper);
    lbfgs_impl(
        problem,
        lower,
        upper,
        initial,
        config.max_iters,
        config.tolerance,
    )
}

fn validate_inputs(
    lower: [f64; N],
    upper: [f64; N],
    initial: [f64; N],
    config: LocalOptimizerConfig,
) -> Result<()> {
    if config.max_iters == 0 || !config.tolerance.is_finite() || config.tolerance < 0.0 {
        bail!("local optimizer config is invalid");
    }
    if initial.iter().any(|value| !value.is_finite()) {
        bail!("local optimizer initial point is invalid");
    }
    for (lower_bound, upper_bound) in lower.iter().zip(upper.iter()) {
        let nondegenerate_span_is_finite =
            lower_bound >= upper_bound || (*upper_bound - *lower_bound).is_finite();
        if !lower_bound.is_finite()
            || !upper_bound.is_finite()
            || lower_bound > upper_bound
            || !nondegenerate_span_is_finite
        {
            bail!("local optimizer bounds are invalid");
        }
    }
    Ok(())
}

fn increment_evaluations(evaluations: u64) -> Result<u64> {
    evaluations
        .checked_add(1)
        .ok_or(ArithmeticOverflow)
        .with_context(|| format!("local optimizer evaluation count overflows u64: {evaluations}"))
}

fn eval<P: LocalScalarProblem3>(problem: &P, x: &[f64; N]) -> Result<f64> {
    let cost = problem.value(x)?;
    if cost.is_finite() {
        Ok(cost)
    } else {
        bail!("local optimizer objective returned a non-finite value")
    }
}

fn value_gradient<P: LocalScalarProblem3>(problem: &P, x: &[f64; N]) -> Result<(f64, [f64; N])> {
    let Some((cost, gradient)) = problem.value_gradient(x) else {
        bail!("local optimizer gradient is unavailable");
    };
    if !cost.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
        bail!("local optimizer objective returned a non-finite value");
    }
    Ok((cost, gradient))
}

fn clamp_to_bounds(x: &mut [f64; N], lower: &[f64; N], upper: &[f64; N]) {
    for (value, (lower_bound, upper_bound)) in x.iter_mut().zip(lower.iter().zip(upper.iter())) {
        *value = (*value).clamp(*lower_bound, *upper_bound);
    }
}

const fn pso_swarm_size(tune: TuneLevel) -> usize {
    match tune {
        TuneLevel::Default => 24,
        TuneLevel::Conservative => 36,
        TuneLevel::Aggressive => 16,
    }
}

#[derive(Clone, Copy, Debug)]
struct NmTuneParams {
    perturbation: f64,
    sd_tolerance: f64,
    iters_factor: f64,
}

impl NmTuneParams {
    const fn for_level(level: TuneLevel, tolerance: f64) -> Self {
        match level {
            TuneLevel::Default => Self {
                perturbation: 0.02,
                sd_tolerance: tolerance.max(1e-3),
                iters_factor: 0.3,
            },
            TuneLevel::Conservative => Self {
                perturbation: 0.05,
                sd_tolerance: tolerance.max(1e-8),
                iters_factor: 1.0,
            },
            TuneLevel::Aggressive => Self {
                perturbation: 0.015,
                sd_tolerance: tolerance.max(5e-3),
                iters_factor: 0.2,
            },
        }
    }
}

fn nelder_mead_impl<P: LocalScalarProblem3>(
    problem: &P,
    lower: [f64; N],
    upper: [f64; N],
    initial: [f64; N],
    max_iters: usize,
    min_iters: usize,
    tune: NmTuneParams,
) -> Result<LocalOptimizeResult> {
    let mut verts = [initial; SIMPLEX];
    let (_, remaining_vertices) = verts
        .split_first_mut()
        .ok_or_else(|| anyhow::anyhow!("local optimizer simplex is empty"))?;
    for (dimension, (((vertex, lower_bound), upper_bound), initial_value)) in remaining_vertices
        .iter_mut()
        .zip(lower.iter())
        .zip(upper.iter())
        .zip(initial.iter())
        .enumerate()
    {
        *vertex = initial;
        let width = *upper_bound - *lower_bound;
        let step = (width * tune.perturbation).max(1e-6);
        let coordinate = vertex
            .get_mut(dimension)
            .ok_or_else(|| anyhow::anyhow!("local optimizer simplex coordinate is missing"))?;
        *coordinate = (*coordinate + step).min(*upper_bound);
        if matches!(
            (*coordinate).partial_cmp(initial_value),
            Some(Ordering::Equal)
        ) {
            *coordinate = (*coordinate - step).max(*lower_bound);
        }
    }

    let mut costs = [0.0; SIMPLEX];
    let mut evaluations = 0_u64;
    for (vertex, cost) in verts.iter_mut().zip(costs.iter_mut()) {
        clamp_to_bounds(vertex, &lower, &upper);
        *cost = eval(problem, vertex)?;
        evaluations = increment_evaluations(evaluations)?;
    }

    let max_iters_as_f64 = count_as_f64(max_iters, "local optimizer max_iters")?;
    let min_iters_as_f64 = count_as_f64(min_iters, "local optimizer min_iters")?;
    let scaled_max_iters = max_iters_as_f64 * tune.iters_factor;
    let effective_iters_as_f64 = scaled_max_iters.max(min_iters_as_f64);
    let effective_iters = effective_iters_as_f64
        .to_usize()
        .ok_or(ArithmeticOverflow)
        .context("local optimizer effective iteration count is not representable")?;
    for _ in 0..effective_iters {
        sort_simplex(&mut verts, &mut costs)?;
        if simplex_sd(&costs) <= tune.sd_tolerance {
            let best_vertex = *verts
                .first()
                .ok_or_else(|| anyhow::anyhow!("local optimizer simplex is empty"))?;
            let best_cost = *costs
                .first()
                .ok_or_else(|| anyhow::anyhow!("local optimizer simplex costs are empty"))?;
            return Ok(LocalOptimizeResult {
                x: best_vertex,
                cost: best_cost,
                evaluations,
                converged: true,
            });
        }

        let centroid = simplex_centroid(&verts);
        let worst_vertex = *verts
            .last()
            .ok_or_else(|| anyhow::anyhow!("local optimizer simplex is empty"))?;
        let worst_cost = *costs
            .last()
            .ok_or_else(|| anyhow::anyhow!("local optimizer simplex costs are empty"))?;
        let best_cost = *costs
            .first()
            .ok_or_else(|| anyhow::anyhow!("local optimizer simplex costs are empty"))?;
        let second_worst_cost = costs
            .iter()
            .rev()
            .nth(1)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("local optimizer simplex needs two costs"))?;
        let reflected = project_from_centroid(&centroid, &worst_vertex, 1.0, &lower, &upper);
        let reflected_cost = eval(problem, &reflected)?;
        evaluations = increment_evaluations(evaluations)?;

        if reflected_cost < best_cost {
            let expanded = project_from_centroid(&centroid, &worst_vertex, 2.0, &lower, &upper);
            let expanded_cost = eval(problem, &expanded)?;
            evaluations = increment_evaluations(evaluations)?;
            let worst_vertex_slot = verts
                .last_mut()
                .ok_or_else(|| anyhow::anyhow!("local optimizer simplex is empty"))?;
            let worst_cost_slot = costs
                .last_mut()
                .ok_or_else(|| anyhow::anyhow!("local optimizer simplex costs are empty"))?;
            if expanded_cost < reflected_cost {
                *worst_vertex_slot = expanded;
                *worst_cost_slot = expanded_cost;
            } else {
                *worst_vertex_slot = reflected;
                *worst_cost_slot = reflected_cost;
            }
        } else if reflected_cost < second_worst_cost {
            *verts
                .last_mut()
                .ok_or_else(|| anyhow::anyhow!("local optimizer simplex is empty"))? = reflected;
            *costs
                .last_mut()
                .ok_or_else(|| anyhow::anyhow!("local optimizer simplex costs are empty"))? =
                reflected_cost;
        } else {
            let contracted = if reflected_cost < worst_cost {
                midpoint(&centroid, &reflected, 0.5, &lower, &upper)
            } else {
                midpoint(&centroid, &worst_vertex, 0.5, &lower, &upper)
            };
            let contracted_cost = eval(problem, &contracted)?;
            evaluations = increment_evaluations(evaluations)?;
            if contracted_cost < worst_cost {
                *verts
                    .last_mut()
                    .ok_or_else(|| anyhow::anyhow!("local optimizer simplex is empty"))? =
                    contracted;
                *costs
                    .last_mut()
                    .ok_or_else(|| anyhow::anyhow!("local optimizer simplex costs are empty"))? =
                    contracted_cost;
            } else {
                let best_vertex = *verts
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("local optimizer simplex is empty"))?;
                let (_, remaining_vertices) = verts
                    .split_first_mut()
                    .ok_or_else(|| anyhow::anyhow!("local optimizer simplex is empty"))?;
                let (_, remaining_costs) = costs
                    .split_first_mut()
                    .ok_or_else(|| anyhow::anyhow!("local optimizer simplex costs are empty"))?;
                if remaining_vertices.len() != remaining_costs.len() {
                    bail!("local optimizer simplex state has inconsistent lengths");
                }
                for (vertex, cost) in remaining_vertices
                    .iter_mut()
                    .zip(remaining_costs.iter_mut())
                {
                    *vertex = midpoint(&best_vertex, vertex, 0.5, &lower, &upper);
                    *cost = eval(problem, vertex)?;
                    evaluations = increment_evaluations(evaluations)?;
                }
            }
        }
    }

    sort_simplex(&mut verts, &mut costs)?;
    let best_vertex = *verts
        .first()
        .ok_or_else(|| anyhow::anyhow!("local optimizer simplex is empty"))?;
    let best_cost = *costs
        .first()
        .ok_or_else(|| anyhow::anyhow!("local optimizer simplex costs are empty"))?;
    Ok(LocalOptimizeResult {
        x: best_vertex,
        cost: best_cost,
        evaluations,
        converged: false,
    })
}

fn sort_simplex(verts: &mut [[f64; N]; SIMPLEX], costs: &mut [f64; SIMPLEX]) -> Result<()> {
    for left in 0..SIMPLEX {
        let right_start = checked_sum(left, 1, "local optimizer simplex sort offset")?;
        let (left_costs, right_costs) = costs.split_at_mut(right_start);
        let (left_vertices, right_vertices) = verts.split_at_mut(right_start);
        let left_cost = left_costs
            .last_mut()
            .ok_or_else(|| anyhow::anyhow!("local optimizer simplex costs are empty"))?;
        let left_vertex = left_vertices
            .last_mut()
            .ok_or_else(|| anyhow::anyhow!("local optimizer simplex is empty"))?;
        if right_costs.len() != right_vertices.len() {
            bail!("local optimizer simplex state has inconsistent lengths");
        }
        for (right_cost, right_vertex) in right_costs.iter_mut().zip(right_vertices.iter_mut()) {
            if *right_cost < *left_cost {
                std::mem::swap(left_cost, right_cost);
                std::mem::swap(left_vertex, right_vertex);
            }
        }
    }
    Ok(())
}

fn simplex_sd(costs: &[f64; SIMPLEX]) -> f64 {
    let mean = costs.iter().sum::<f64>() / SIMPLEX_AS_F64;
    let sum_of_squared_deviations = costs.iter().map(|cost| (cost - mean).powi(2)).sum::<f64>();
    (sum_of_squared_deviations / SIMPLEX_AS_F64).sqrt()
}

fn simplex_centroid(verts: &[[f64; N]; SIMPLEX]) -> [f64; N] {
    let mut centroid = [0.0; N];
    for vertex in verts.iter().take(N) {
        for (total, coordinate) in centroid.iter_mut().zip(vertex.iter()) {
            *total += *coordinate;
        }
    }
    for value in &mut centroid {
        *value /= N_AS_F64;
    }
    centroid
}

fn project_from_centroid(
    centroid: &[f64; N],
    worst: &[f64; N],
    factor: f64,
    lower: &[f64; N],
    upper: &[f64; N],
) -> [f64; N] {
    let mut out = [0.0; N];
    for ((output, (centroid_value, worst_value)), (lower_bound, upper_bound)) in out
        .iter_mut()
        .zip(centroid.iter().zip(worst.iter()))
        .zip(lower.iter().zip(upper.iter()))
    {
        let difference = *centroid_value - *worst_value;
        let scaled_difference = factor * difference;
        let projected = *centroid_value + scaled_difference;
        *output = projected.clamp(*lower_bound, *upper_bound);
    }
    out
}

fn midpoint(
    a: &[f64; N],
    b: &[f64; N],
    factor: f64,
    lower: &[f64; N],
    upper: &[f64; N],
) -> [f64; N] {
    let mut out = [0.0; N];
    for ((output, (left, right)), (lower_bound, upper_bound)) in out
        .iter_mut()
        .zip(a.iter().zip(b.iter()))
        .zip(lower.iter().zip(upper.iter()))
    {
        let difference = *right - *left;
        let scaled_difference = factor * difference;
        let middle = *left + scaled_difference;
        *output = middle.clamp(*lower_bound, *upper_bound);
    }
    out
}

fn lbfgs_impl<P: LocalScalarProblem3>(
    problem: &P,
    lower: [f64; N],
    upper: [f64; N],
    initial: [f64; N],
    max_iters: usize,
    tolerance: f64,
) -> Result<LocalOptimizeResult> {
    let mut current_point = initial;
    let (mut current_cost, mut current_gradient) = value_gradient(problem, &current_point)?;
    let mut evaluations = 1_u64;

    let mut position_history = [[0.0; N]; LBFGS_HISTORY];
    let mut gradient_history = [[0.0; N]; LBFGS_HISTORY];
    let mut inverse_curvature_history = [0.0; LBFGS_HISTORY];
    let mut history_count = 0_usize;
    let mut history_head = 0_usize;

    for _ in 0..max_iters {
        if norm(&current_gradient) <= tolerance {
            return Ok(LocalOptimizeResult {
                x: current_point,
                cost: current_cost,
                evaluations,
                converged: true,
            });
        }

        let mut recursion_input = current_gradient;
        let mut alpha = [0.0; LBFGS_HISTORY];
        for reverse_offset in (0..history_count).rev() {
            let history_index = lbfgs_history_index(history_head, reverse_offset)?;
            let position_change = *position_history.get(history_index).ok_or_else(|| {
                anyhow::anyhow!("local optimizer L-BFGS position history is invalid")
            })?;
            let gradient_change = *gradient_history.get(history_index).ok_or_else(|| {
                anyhow::anyhow!("local optimizer L-BFGS gradient history is invalid")
            })?;
            let inverse_curvature =
                *inverse_curvature_history
                    .get(history_index)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "local optimizer L-BFGS inverse-curvature history is invalid"
                        )
                    })?;
            let alpha_value = inverse_curvature * dot(&position_change, &recursion_input);
            *alpha.get_mut(reverse_offset).ok_or_else(|| {
                anyhow::anyhow!("local optimizer L-BFGS alpha history is invalid")
            })? = alpha_value;
            axpy(&mut recursion_input, &gradient_change, -alpha_value);
        }
        let mut inverse_hessian_product = recursion_input;
        if history_count > 0 {
            let last_history_index = lbfgs_history_index(history_head, 0)?;
            let last_gradient_change =
                *gradient_history.get(last_history_index).ok_or_else(|| {
                    anyhow::anyhow!("local optimizer L-BFGS gradient history is invalid")
                })?;
            let last_position_change =
                *position_history.get(last_history_index).ok_or_else(|| {
                    anyhow::anyhow!("local optimizer L-BFGS position history is invalid")
                })?;
            let gradient_norm_squared = dot(&last_gradient_change, &last_gradient_change);
            let position_gradient_dot = dot(&last_position_change, &last_gradient_change);
            if gradient_norm_squared > 1e-12 {
                scale(
                    &mut inverse_hessian_product,
                    position_gradient_dot / gradient_norm_squared,
                );
            }
        }
        for reverse_offset in (0..history_count).rev() {
            let history_index = lbfgs_history_index(history_head, reverse_offset)?;
            let position_change = *position_history.get(history_index).ok_or_else(|| {
                anyhow::anyhow!("local optimizer L-BFGS position history is invalid")
            })?;
            let gradient_change = *gradient_history.get(history_index).ok_or_else(|| {
                anyhow::anyhow!("local optimizer L-BFGS gradient history is invalid")
            })?;
            let inverse_curvature =
                *inverse_curvature_history
                    .get(history_index)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "local optimizer L-BFGS inverse-curvature history is invalid"
                        )
                    })?;
            let beta = inverse_curvature * dot(&gradient_change, &inverse_hessian_product);
            let alpha_value = *alpha.get(reverse_offset).ok_or_else(|| {
                anyhow::anyhow!("local optimizer L-BFGS alpha history is invalid")
            })?;
            let alpha_minus_beta = alpha_value - beta;
            axpy(
                &mut inverse_hessian_product,
                &position_change,
                alpha_minus_beta,
            );
        }
        let mut descent_direction = inverse_hessian_product;
        scale(&mut descent_direction, -1.0);

        let mut step = 1.0;
        let directional = dot(&current_gradient, &descent_direction);
        let previous_point = current_point;
        let previous_gradient = current_gradient;
        let previous_cost = current_cost;
        let mut accepted = false;
        for _ in 0..24 {
            let mut trial_point = previous_point;
            axpy(&mut trial_point, &descent_direction, step);
            clamp_to_bounds(&mut trial_point, &lower, &upper);
            let (trial_cost, trial_gradient) = value_gradient(problem, &trial_point)?;
            evaluations = increment_evaluations(evaluations)?;
            let armijo_step = 1e-4 * step;
            let armijo_decrease = armijo_step * directional;
            let armijo_bound = previous_cost + armijo_decrease;
            if trial_cost <= armijo_bound {
                current_point = trial_point;
                current_cost = trial_cost;
                current_gradient = trial_gradient;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }

        let mut position_change = current_point;
        axpy(&mut position_change, &previous_point, -1.0);
        let mut gradient_change = current_gradient;
        axpy(&mut gradient_change, &previous_gradient, -1.0);
        let curvature = dot(&gradient_change, &position_change);
        if curvature > 1e-12 {
            *position_history.get_mut(history_head).ok_or_else(|| {
                anyhow::anyhow!("local optimizer L-BFGS position history is invalid")
            })? = position_change;
            *gradient_history.get_mut(history_head).ok_or_else(|| {
                anyhow::anyhow!("local optimizer L-BFGS gradient history is invalid")
            })? = gradient_change;
            *inverse_curvature_history
                .get_mut(history_head)
                .ok_or_else(|| {
                    anyhow::anyhow!("local optimizer L-BFGS inverse-curvature history is invalid")
                })? = 1.0 / curvature;
            history_head = advance_lbfgs_history_head(history_head)?;
            history_count = increment_lbfgs_history_count(history_count)?;
        }
    }

    Ok(LocalOptimizeResult {
        x: current_point,
        cost: current_cost,
        evaluations,
        converged: norm(&current_gradient) <= tolerance,
    })
}

fn lbfgs_history_index(history_head: usize, reverse_offset: usize) -> Result<usize> {
    if history_head >= LBFGS_HISTORY || reverse_offset >= LBFGS_HISTORY {
        bail!("local optimizer L-BFGS history index is invalid");
    }
    let distance = checked_sum(reverse_offset, 1, "local optimizer L-BFGS history distance")?;
    if history_head >= distance {
        checked_difference(
            history_head,
            distance,
            "local optimizer L-BFGS history index",
        )
    } else {
        let wrapped_distance = checked_difference(
            distance,
            history_head,
            "local optimizer L-BFGS history wrap distance",
        )?;
        checked_difference(
            LBFGS_HISTORY,
            wrapped_distance,
            "local optimizer L-BFGS history wrapped index",
        )
    }
}

fn advance_lbfgs_history_head(history_head: usize) -> Result<usize> {
    if history_head >= LBFGS_HISTORY {
        bail!("local optimizer L-BFGS history head is invalid");
    }
    let next_head = checked_sum(history_head, 1, "local optimizer L-BFGS history head")?;
    if next_head == LBFGS_HISTORY {
        Ok(0)
    } else {
        Ok(next_head)
    }
}

fn increment_lbfgs_history_count(history_count: usize) -> Result<usize> {
    if history_count > LBFGS_HISTORY {
        bail!("local optimizer L-BFGS history count is invalid");
    }
    if history_count == LBFGS_HISTORY {
        Ok(LBFGS_HISTORY)
    } else {
        checked_sum(history_count, 1, "local optimizer L-BFGS history count")
    }
}

fn dot(left: &[f64; N], right: &[f64; N]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

fn norm(vector: &[f64; N]) -> f64 {
    dot(vector, vector).sqrt()
}

fn scale(values: &mut [f64; N], factor: f64) {
    for value in values {
        *value *= factor;
    }
}

fn axpy(output: &mut [f64; N], input: &[f64; N], factor: f64) {
    for (output_value, input_value) in output.iter_mut().zip(input.iter()) {
        let scaled_input = factor * *input_value;
        *output_value += scaled_input;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        run_lbfgs3, run_local_optimizer, run_nelder_mead3, run_pso3, LocalOptimizeResult,
        LocalOptimizerConfig, LocalOptimizerKind, LocalScalarProblem3, Result, TuneLevel, N,
    };
    use std::cell::Cell;

    struct Constant;

    impl LocalScalarProblem3 for Constant {
        fn value(&self, _: &[f64; N]) -> Result<f64> {
            Ok(0.0)
        }
    }

    struct InvalidTransferCost;

    impl LocalScalarProblem3 for InvalidTransferCost {
        fn value(&self, _: &[f64; N]) -> Result<f64> {
            Ok(crate::types::INVALID_COST)
        }
    }

    struct CallProbe {
        called: Cell<bool>,
    }

    impl LocalScalarProblem3 for CallProbe {
        fn value(&self, _: &[f64; N]) -> Result<f64> {
            self.called.set(true);
            Ok(0.0)
        }
    }

    #[derive(Debug)]
    struct ObjectiveFailure;

    impl std::fmt::Display for ObjectiveFailure {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("test objective failure")
        }
    }

    impl std::error::Error for ObjectiveFailure {}

    struct FailingObjective {
        calls: Cell<u8>,
    }

    impl LocalScalarProblem3 for FailingObjective {
        fn value(&self, _: &[f64; N]) -> Result<f64> {
            let Some(next_calls) = self.calls.get().checked_add(1) else {
                return Err(ObjectiveFailure.into());
            };
            self.calls.set(next_calls);
            Err(ObjectiveFailure.into())
        }
    }

    struct FlatGradient;

    impl LocalScalarProblem3 for FlatGradient {
        fn value(&self, _: &[f64; N]) -> Result<f64> {
            Ok(0.0)
        }

        fn value_gradient(&self, _: &[f64; N]) -> Option<(f64, [f64; N])> {
            Some((0.0, [0.0; N]))
        }
    }

    struct Quadratic;

    impl LocalScalarProblem3 for Quadratic {
        fn value(&self, x: &[f64; N]) -> Result<f64> {
            let first_term = (x[0] - 0.25).powi(2);
            let second_term = (x[1] - 0.5).powi(2);
            let first_two_terms = first_term + second_term;
            let third_term = (x[2] - 0.75).powi(2);
            Ok(first_two_terms + third_term)
        }

        fn value_gradient(&self, x: &[f64; N]) -> Option<(f64, [f64; N])> {
            Some((
                self.value(x).ok()?,
                [2.0 * (x[0] - 0.25), 2.0 * (x[1] - 0.5), 2.0 * (x[2] - 0.75)],
            ))
        }
    }

    type ResultBits = ([u64; N], u64, u64, bool);

    fn result_bits(result: Result<LocalOptimizeResult>) -> anyhow::Result<ResultBits> {
        result.map(|value| {
            (
                value.x.map(f64::to_bits),
                value.cost.to_bits(),
                value.evaluations,
                value.converged,
            )
        })
    }

    fn too_many_iterations() -> Option<usize> {
        usize::try_from(u64::from(u32::MAX) + 1).ok()
    }

    #[test]
    fn pso_rejects_non_finite_initial_point_before_evaluation() {
        let problem = CallProbe {
            called: Cell::new(false),
        };
        let result = run_pso3(
            &problem,
            [0.0; N],
            [1.0; N],
            [f64::NAN, 0.5, 0.5],
            LocalOptimizerConfig {
                kind: LocalOptimizerKind::Pso,
                max_iters: 1,
                ..LocalOptimizerConfig::default()
            },
        );

        assert!(result.is_err(), "non-finite initial point must be rejected");
        assert!(!problem.called.get(), "objective must not be evaluated");
    }

    #[test]
    fn local_optimizers_reject_nonfinite_non_degenerate_span_before_evaluation() {
        for kind in [
            LocalOptimizerKind::NelderMead,
            LocalOptimizerKind::Pso,
            LocalOptimizerKind::Lbfgs,
        ] {
            let problem = CallProbe {
                called: Cell::new(false),
            };
            let result = run_local_optimizer(
                &problem,
                [-f64::MAX, 0.0, 0.0],
                [f64::MAX, 1.0, 1.0],
                [0.0, 0.5, 0.5],
                LocalOptimizerConfig {
                    kind,
                    max_iters: 1,
                    ..LocalOptimizerConfig::default()
                },
            );

            assert!(result.is_err(), "{kind:?} accepted an infinite span");
            assert!(
                result
                    .as_ref()
                    .err()
                    .is_some_and(|error| error.to_string().contains("bounds")),
                "{kind:?} infinite-span error lacks its bounds diagnostic"
            );
            assert!(
                !problem.called.get(),
                "{kind:?} evaluated before rejecting the infinite span"
            );
        }
    }

    #[test]
    fn pso_propagates_typed_objective_error_before_followup_evaluation() {
        let problem = FailingObjective {
            calls: Cell::new(0),
        };
        let result = run_pso3(
            &problem,
            [0.0; N],
            [1.0; N],
            [0.5; N],
            LocalOptimizerConfig {
                kind: LocalOptimizerKind::Pso,
                max_iters: 1,
                ..LocalOptimizerConfig::default()
            },
        );

        assert!(
            result
                .as_ref()
                .err()
                .and_then(|error| error.downcast_ref::<ObjectiveFailure>())
                .is_some(),
            "typed objective failure must cross eval unchanged"
        );
        assert_eq!(
            problem.calls.get(),
            1,
            "failure must stop further evaluation"
        );
    }

    #[test]
    fn evaluation_counter_overflow_has_typed_source() {
        let result = super::increment_evaluations(u64::MAX);

        assert!(
            result
                .as_ref()
                .err()
                .and_then(|error| error.downcast_ref::<crate::oxymoo::ArithmeticOverflow>())
                .is_some(),
            "evaluation counter overflow must retain its typed source"
        );
    }

    #[test]
    fn nelder_mead_rejects_unrepresentable_iteration_control() {
        let Some(too_many_iterations) = too_many_iterations() else {
            return;
        };
        let result = run_nelder_mead3(
            &Constant,
            [0.0; N],
            [1.0; N],
            [0.5; N],
            LocalOptimizerConfig {
                kind: LocalOptimizerKind::NelderMead,
                max_iters: too_many_iterations,
                min_iters: 1,
                ..LocalOptimizerConfig::default()
            },
        );

        assert!(
            result.is_err(),
            "unrepresentable iteration control must be rejected"
        );
        assert!(
            result
                .as_ref()
                .err()
                .and_then(|error| error.downcast_ref::<crate::oxymoo::ArithmeticOverflow>())
                .is_some(),
            "unrepresentable iteration control must retain its typed source"
        );
    }

    #[test]
    fn pso_ignores_nelder_mead_only_iteration_floor() {
        let Some(too_many_iterations) = too_many_iterations() else {
            return;
        };
        let result = run_pso3(
            &Constant,
            [0.0; N],
            [1.0; N],
            [0.5; N],
            LocalOptimizerConfig {
                kind: LocalOptimizerKind::Pso,
                max_iters: 1,
                min_iters: too_many_iterations,
                ..LocalOptimizerConfig::default()
            },
        );

        assert!(result.is_ok(), "PSO must ignore Nelder-Mead-only min_iters");
    }

    #[test]
    fn direct_nelder_mead_checks_its_floor_regardless_of_config_kind() {
        let Some(too_many_iterations) = too_many_iterations() else {
            return;
        };
        let result = run_nelder_mead3(
            &Constant,
            [0.0; N],
            [1.0; N],
            [0.5; N],
            LocalOptimizerConfig {
                kind: LocalOptimizerKind::Pso,
                max_iters: 1,
                min_iters: too_many_iterations,
                ..LocalOptimizerConfig::default()
            },
        );

        assert!(
            result.is_err(),
            "direct Nelder-Mead must validate its own min_iters"
        );
    }

    #[test]
    fn lbfgs_ignores_nelder_mead_only_iteration_floor() {
        let Some(too_many_iterations) = too_many_iterations() else {
            return;
        };
        let result = run_lbfgs3(
            &FlatGradient,
            [0.0; N],
            [1.0; N],
            [0.5; N],
            LocalOptimizerConfig {
                kind: LocalOptimizerKind::NelderMead,
                max_iters: 1,
                min_iters: too_many_iterations,
                ..LocalOptimizerConfig::default()
            },
        );

        assert!(
            result.is_ok(),
            "L-BFGS must ignore Nelder-Mead-only min_iters"
        );
    }

    #[test]
    fn pso_does_not_cap_large_iteration_limit_before_objective_evaluation() {
        let Some(too_many_iterations) = too_many_iterations() else {
            return;
        };
        let problem = FailingObjective {
            calls: Cell::new(0),
        };
        let result = run_pso3(
            &problem,
            [0.0; N],
            [1.0; N],
            [0.5; N],
            LocalOptimizerConfig {
                kind: LocalOptimizerKind::Pso,
                max_iters: too_many_iterations,
                ..LocalOptimizerConfig::default()
            },
        );

        assert!(
            result
                .as_ref()
                .err()
                .and_then(|error| error.downcast_ref::<ObjectiveFailure>())
                .is_some(),
            "PSO must not reject integer controls it never converts to f64"
        );
        assert_eq!(
            problem.calls.get(),
            1,
            "objective must receive the first call"
        );
    }

    #[test]
    fn lbfgs_accepts_large_iteration_limit_when_initial_gradient_converges() {
        let Some(too_many_iterations) = too_many_iterations() else {
            return;
        };
        let result = run_lbfgs3(
            &FlatGradient,
            [0.0; N],
            [1.0; N],
            [0.5; N],
            LocalOptimizerConfig {
                kind: LocalOptimizerKind::Lbfgs,
                max_iters: too_many_iterations,
                ..LocalOptimizerConfig::default()
            },
        );

        assert!(
            result.is_ok(),
            "L-BFGS must retain integer controls it never converts to f64"
        );
    }

    #[test]
    fn nelder_mead_normal_input_bits_are_pinned() {
        assert_eq!(
            result_bits(run_nelder_mead3(
                &Quadratic,
                [0.0; N],
                [1.0; N],
                [0.1, 0.8, 0.2],
                LocalOptimizerConfig {
                    kind: LocalOptimizerKind::NelderMead,
                    max_iters: 48,
                    tolerance: 1e-12,
                    seed: 0,
                    tune: TuneLevel::Conservative,
                    min_iters: 32,
                },
            ))
            .expect("pinned Nelder-Mead input must optimize"),
            (
                [
                    4_598_164_799_301_564_950,
                    4_602_677_363_864_128_971,
                    4_604_929_438_637_084_230
                ],
                4_502_703_360_661_103_539,
                89,
                false,
            )
        );
    }

    #[test]
    fn pso_normal_input_bits_are_pinned() {
        assert_eq!(
            result_bits(run_pso3(
                &Quadratic,
                [0.0; N],
                [1.0; N],
                [0.1, 0.8, 0.2],
                LocalOptimizerConfig {
                    kind: LocalOptimizerKind::Pso,
                    max_iters: 32,
                    tolerance: 0.0,
                    seed: 0x5eed,
                    tune: TuneLevel::Default,
                    min_iters: 7,
                },
            ))
            .expect("pinned PSO input must optimize"),
            (
                [
                    4_598_159_545_539_513_471,
                    4_602_691_206_536_463_327,
                    4_604_932_661_068_775_048
                ],
                4_522_144_560_526_631_047,
                792,
                true,
            )
        );
    }

    #[test]
    fn pso_does_not_accept_transfer_invalid_cost_as_converged() {
        let result = run_pso3(
            &InvalidTransferCost,
            [0.0; N],
            [1.0; N],
            [0.5; N],
            LocalOptimizerConfig {
                kind: LocalOptimizerKind::Pso,
                max_iters: 1,
                tolerance: 0.0,
                seed: 0,
                tune: TuneLevel::Default,
                min_iters: 0,
            },
        )
        .expect("the transfer invalid-cost sentinel is finite");

        assert_eq!(result.cost.to_bits(), crate::types::INVALID_COST.to_bits());
        assert!(!result.converged);
    }

    #[test]
    fn lbfgs_normal_input_bits_are_pinned() {
        assert_eq!(
            result_bits(run_lbfgs3(
                &Quadratic,
                [0.0; N],
                [1.0; N],
                [0.1, 0.8, 0.2],
                LocalOptimizerConfig {
                    kind: LocalOptimizerKind::Lbfgs,
                    max_iters: 8,
                    tolerance: 1e-12,
                    seed: 0,
                    tune: TuneLevel::Default,
                    min_iters: 1,
                },
            ))
            .expect("pinned L-BFGS input must optimize"),
            (
                [
                    4_598_175_219_545_276_416,
                    4_602_678_819_172_646_912,
                    4_604_930_618_986_332_160
                ],
                0,
                3,
                true,
            )
        );
    }
}
