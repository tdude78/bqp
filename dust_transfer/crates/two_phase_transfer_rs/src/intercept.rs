//! High-performance intercept optimization in Rust.
//!
//! This module implements a Newton-Raphson solver with Armijo line search
//! and Levenberg-Marquardt fallback for optimizing the dust intercept Δv.
//!
//! The bounded variant (`optimize_intercept_bounded`) constrains the delta-V
//! vector to an L2 ball: ||dv|| <= bound.

use nalgebra::{Matrix3, Matrix6x3, Vector3, Vector6};
use std::f64;

/// One successful high-fidelity miss evaluation plus its propagated endpoint.
///
/// Propagation errors stay in the checked result until the strict postprocess
/// boundary deliberately records them; this value only carries valid trials.
#[derive(Debug, Clone, Copy)]
pub struct HfInterceptEvaluation {
    pub miss: [f64; 3],
    pub endpoint: Option<[f64; 6]>,
}

/// Bounded-HF result with the selected propagated endpoint kept outside the
/// public optimizer result.
#[derive(Debug, Clone, Copy)]
pub struct BoundedHfInterceptResult {
    pub intercept: BoundedInterceptResult,
    endpoint: Option<[f64; 6]>,
}

impl BoundedHfInterceptResult {
    #[inline]
    #[must_use]
    pub const fn endpoint_for_returned_dv(self) -> Option<[f64; 6]> {
        self.endpoint
    }
}

// ============================================================================
// Bounded Levenberg-Marquardt Optimizer (for scipy.least_squares replacement)
// ============================================================================

/// Configuration for the L2-ball-bounded intercept optimizer.
///
/// Replaces `scipy.optimize.least_squares` with Trust Region Reflective algorithm.
#[derive(Debug, Clone, Copy)]
pub struct BoundedInterceptConfig {
    /// Maximum iterations for LM solver
    pub max_iters: usize,
    /// Residual tolerance (km) - converge when miss < tol
    pub tol: f64,
    /// Step tolerance (km/s) - converge when ||dx|| < `step_tol`
    pub step_tol: f64,
    /// Jacobian finite difference epsilon
    pub jac_eps: f64,
    /// Initial LM damping factor
    pub damping_initial: f64,
    /// Damping adjustment factor
    pub damping_factor: f64,
    /// L2-ball radius for the delta-v vector [km/s]
    pub bound: f64,
    /// Maximum adaptive L2-ball radius [km/s]
    pub max_bound: f64,
    /// Regularization weight on delta-V magnitude (L2 penalty)
    pub reg_weight: f64,
    /// Skip tolerance: accept early if miss < `skip_tol` [km]
    pub skip_tol: f64,
    /// Minimum miss distance target [km]
    pub min_miss_km: f64,
    /// Maximum bound expansion iterations
    pub max_bound_expansions: usize,
}

impl Default for BoundedInterceptConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tol: 1e-5,
            step_tol: 1e-9,
            jac_eps: 1e-4,
            damping_initial: 1e-4,
            damping_factor: 10.0,
            bound: 0.5, // 0.5 km/s default bound
            max_bound: 2.0,
            reg_weight: 1e-3, // Small regularization
            skip_tol: 1.0,    // Skip if miss < 1 km
            min_miss_km: 2.5, // Target miss distance
            max_bound_expansions: 7,
        }
    }
}

/// Result of bounded intercept optimization
#[derive(Debug, Clone, Copy)]
pub struct BoundedInterceptResult {
    /// Optimized delta-V [km/s]
    pub dv: [f64; 3],
    /// Final miss distance [km]
    pub miss_km: f64,
    /// Number of iterations used
    pub iters: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// True if converged to tolerance
    pub converged: bool,
    /// True if solution satisfies miss distance constraint
    pub success: bool,
    /// Final bound used (may be expanded)
    pub final_bound: f64,
}

#[cfg(feature = "solver-qualification")]
use crate::evaluate::propagate_high_fidelity_state_at_epoch_checked_observed;
use crate::evaluate::{propagate_high_fidelity_state_at_epoch_checked, TransferPropagationFailure};
#[cfg(feature = "solver-qualification")]
use crate::postprocess::{
    QualificationLegInput, QualificationLegPath, QualificationLegTrace, QualificationTraceError,
};
use crate::types::{BodyForceConfig, PlanContext};

/// Propagate and compute miss distance vector (Equinoctial)
#[must_use]
pub fn compute_miss_vector_equinoctial(
    dv: [f64; 3],
    v0: [f64; 3],
    r0: [f64; 3],
    target_pos: [f64; 3],
    tof_s: f64,
) -> [f64; 3] {
    let v_new = [v0[0] + dv[0], v0[1] + dv[1], v0[2] + dv[2]];
    let state0 = [r0[0], r0[1], r0[2], v_new[0], v_new[1], v_new[2]];
    let mut equ = [0.0; 6];
    satpy_core::eci2equinoc_impl(&state0, 6, 0.0, 0.0, &mut equ);
    let mut state_intercept = [0.0; 6];
    satpy_core::equinoc_prop_from_impl(&equ, tof_s, &mut state_intercept);

    [
        state_intercept[0] - target_pos[0],
        state_intercept[1] - target_pos[1],
        state_intercept[2] - target_pos[2],
    ]
}

/// Propagate an HF interceptor trial and retain its endpoint for exact reuse.
pub(crate) fn compute_miss_vector_hf_with_endpoint(
    dv: [f64; 3],
    v0: [f64; 3],
    r0: [f64; 3],
    target_pos: [f64; 3],
    tof_s: f64,
    source_jd: f64,
    body_force: BodyForceConfig,
    ctx: &PlanContext,
) -> Result<HfInterceptEvaluation, TransferPropagationFailure> {
    let v_new = [v0[0] + dv[0], v0[1] + dv[1], v0[2] + dv[2]];
    let state0 = [r0[0], r0[1], r0[2], v_new[0], v_new[1], v_new[2]];

    let mut equ = [0.0; 6];
    satpy_core::eci2equinoc_impl(&state0, 6, 0.0, 0.0, &mut equ);

    let state_intercept =
        propagate_high_fidelity_state_at_epoch_checked(&equ, tof_s, source_jd, body_force, ctx)?;

    Ok(HfInterceptEvaluation {
        miss: [
            state_intercept[0] - target_pos[0],
            state_intercept[1] - target_pos[1],
            state_intercept[2] - target_pos[2],
        ],
        endpoint: Some(state_intercept),
    })
}

/// Qualification-only strict-HF intercept trial using the same scalar core.
/// Penalty-only optimizer probes never reach this function and therefore never
/// manufacture a propagation record.
#[cfg(feature = "solver-qualification")]
pub(crate) fn compute_miss_vector_hf_with_endpoint_observed(
    dv: [f64; 3],
    v0: [f64; 3],
    r0: [f64; 3],
    target_pos: [f64; 3],
    tof_s: f64,
    source_jd: f64,
    body_force: BodyForceConfig,
    ctx: &PlanContext,
    trace: &mut QualificationLegTrace,
) -> Result<HfInterceptEvaluation, TransferPropagationFailure> {
    let v_new = [v0[0] + dv[0], v0[1] + dv[1], v0[2] + dv[2]];
    let state0 = [r0[0], r0[1], r0[2], v_new[0], v_new[1], v_new[2]];
    let mut equ = [0.0; 6];
    satpy_core::eci2equinoc_impl(&state0, 6, 0.0, 0.0, &mut equ);
    let observed = match propagate_high_fidelity_state_at_epoch_checked_observed(
        &equ, tof_s, source_jd, body_force, ctx,
    ) {
        Ok(observed) => observed,
        Err(error) => {
            trace.mark_incomplete(QualificationTraceError::IncompleteMetrics);
            return Err(error);
        }
    };
    let outcome = observed.outcome;
    trace.record_observed_transfer(
        QualificationLegInput::new(
            QualificationLegPath::ReleaseInterceptTrial,
            body_force.role,
            source_jd,
            0.0,
            tof_s,
            state0,
        ),
        outcome.clone(),
        observed.scalar_observation,
    );
    let state_intercept = outcome?;
    Ok(HfInterceptEvaluation {
        miss: [
            state_intercept[0] - target_pos[0],
            state_intercept[1] - target_pos[1],
            state_intercept[2] - target_pos[2],
        ],
        endpoint: Some(state_intercept),
    })
}

// ============================================================================
// L2-Ball-Constrained Levenberg-Marquardt with Regularization
// ============================================================================

/// Check whether delta-v is on the L2-ball boundary within `atol`.
#[inline]
fn at_boundary(dv: &Vector3<f64>, bound: f64, atol: f64) -> bool {
    // Boundary of the L2 ball (matches project_to_ball):
    // expansion triggers when the solution presses against the norm ceiling.
    (dv.norm() - bound).abs() < atol
}

/// Project vector onto the L2 ball of radius `bound`.
///
/// The bound is a PHYSICAL delta-v ceiling (`max_physical_dv_kms` at the
/// release-control call sites). A per-axis box admits vectors up to
/// sqrt(3) * bound in norm — 73% over the ceiling at full saturation —
/// so the projection must be onto the norm ball, not the box.
#[inline]
fn project_to_ball(v: &mut Vector3<f64>, bound: f64) {
    let norm = v.norm();
    if norm > bound && norm > 0.0 {
        let scale = bound / norm;
        for i in 0..3 {
            v[i] *= scale;
        }
    }
}

/// Build 6-element regularized residual: [`miss_xyz`, sqrt(reg)*`dv_xyz`]
#[inline]
fn build_residual(miss: &Vector3<f64>, dv: &Vector3<f64>, sqrt_reg: f64) -> Vector6<f64> {
    Vector6::new(
        miss[0],
        miss[1],
        miss[2],
        sqrt_reg * dv[0],
        sqrt_reg * dv[1],
        sqrt_reg * dv[2],
    )
}

#[inline]
fn solve_lm_normal_step(lhs: Matrix3<f64>, b: Vector3<f64>) -> Option<Vector3<f64>> {
    let step = lhs
        .cholesky()
        .map(|chol| chol.solve(&b))
        .or_else(|| lhs.lu().solve(&b))?;
    step.iter().all(|value| value.is_finite()).then_some(step)
}

/// Single LM solve pass with an L2-ball constraint.
///
/// What one bounded-LM pass converged to, including the miss VECTOR at the
/// best point rather than only its norm.
///
/// The vector is what makes a bound-expansion re-entry seedable: the next pass
/// restarts at `x` and would otherwise re-evaluate the objective there, which
/// under the strict-HF release control is a whole propagation.
struct LmSolveOutcome {
    x: Vector3<f64>,
    miss_km: f64,
    miss_vec: [f64; 3],
    endpoint: Option<[f64; 6]>,
    iters: usize,
    nfev: usize,
    converged: bool,
}

#[inline]
fn checked_lm_count_add(
    current: usize,
    incoming: usize,
) -> Result<usize, TransferPropagationFailure> {
    current
        .checked_add(incoming)
        .ok_or(TransferPropagationFailure::ArithmeticOverflow)
}

#[inline]
fn checked_lm_census_prop_delta(
    current: u64,
    before: u64,
) -> Result<u64, TransferPropagationFailure> {
    current
        .checked_sub(before)
        .ok_or(TransferPropagationFailure::ArithmeticOverflow)
}

#[inline]
fn checked_lm_census_u32(value: usize) -> Result<u32, TransferPropagationFailure> {
    u32::try_from(value).map_err(|_| TransferPropagationFailure::ArithmeticOverflow)
}

#[inline]
fn checked_lm_trial_evals_add(current: u32) -> Result<u32, TransferPropagationFailure> {
    current
        .checked_add(1)
        .ok_or(TransferPropagationFailure::ArithmeticOverflow)
}

#[inline]
fn checked_lm_totals(
    total_iters: usize,
    total_nfev: usize,
    incoming_iters: usize,
    incoming_nfev: usize,
) -> Result<(usize, usize), TransferPropagationFailure> {
    let next_iters = checked_lm_count_add(total_iters, incoming_iters)?;
    let next_nfev = checked_lm_count_add(total_nfev, incoming_nfev)?;
    Ok((next_iters, next_nfev))
}

/// A cheap stand-in for the objective, used ONLY to build the LM Jacobian.
///
/// The Jacobian never substitutes for the real objective: every trial step is
/// evaluated with `f` and rejected unless it lowers the true cost. It does
/// steer the search path, however, and can therefore change the selected
/// result. The 2026-07-28 ambient-probe gate demonstrated that sensitivity on
/// the production-shaped strict-HF corpus.
///
/// For the strict-HF release control the model is the KEPLERIAN form of the
/// smooth miss map, `miss(dv) = r(T; r0, v0 + dv) - r_target`; protected-radius
/// enforcement remains in the real objective. Historical projected-probe
/// comparisons over 3,402- and 2,550-row corpora reported:
/// `||J_model - J_fd||_F / ||J_fd||_F` median 3.1e-3, 91.5% below 5%.
/// Those accuracy statistics do not describe the current ambient-coordinate
/// difference and must be remeasured before reuse.
///
/// `None` means "difference the real objective", which under strict HF is three
/// propagations per LM iteration and can choose a different path than the
/// model. See `solve_intercept_delta_dv`, where both high-fidelity branches
/// currently supply a model. This is an explicit parameter rather than an
/// ambient thread-local precisely so that a new call site cannot quietly choose
/// either science-sensitive path: there is no implicit default.
/// Returning `None` for a point means "I do not describe the objective here",
/// and the LM falls back to differencing the real objective for that iteration.
/// A model that answers `Some` everywhere is differenced everywhere.
pub type JacobianModel<'a> = Option<&'a (dyn Fn([f64; 3]) -> Option<[f64; 3]> + Sync)>;

/// One-sided ambient-coordinate difference of `model` at `x`.
///
/// The iterate is bounded, but its Jacobian describes the surrounding miss
/// map, so each probe is exactly `x + jac_eps * e_j` without projection.
/// `None` means the requested increment is numerically zero or the model
/// declines/returns a non-finite value; the caller then differences the real
/// objective.
///
/// The user explicitly accepted this ambient behavior after its failed
/// production-shaped identity gate. Accepted-displacement convergence was
/// accepted later under a separate, independently scoped authorization.
/// Neither acceptance authorizes future science movement or re-pinning.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "ambient finite differences retain their established IEEE-754 subtraction and division order"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "the only dynamic coordinate accesses are over the fixed three-axis Cartesian basis"
)]
fn model_jacobian(
    model: &(dyn Fn([f64; 3]) -> Option<[f64; 3]> + Sync),
    x: &Vector3<f64>,
    jac_eps: f64,
) -> Option<[f64; 9]> {
    // The ITERATE has to be a point the model describes. When it is not -- the
    // solver is sitting inside the protected-radius region, where the objective
    // returns a flat sentinel a Keplerian miss map knows nothing about -- the
    // model's slope and the objective's residual are inconsistent by orders of
    // magnitude, and the normal equations built from the two produce a step
    // that is not merely wrong but unboundedly large. Differencing the real
    // objective is what gets the solver out of that region, so pay for it.
    let base = Vector3::from_column_slice(&model([x[0], x[1], x[2]])?);
    let mut out = [0.0f64; 9];
    for j in 0..3 {
        let mut probe_point = *x;
        probe_point[j] += jac_eps;
        let step = probe_point[j] - x[j];
        if step.abs() <= 1e-12 {
            return None;
        }
        let shifted =
            Vector3::from_column_slice(&model([probe_point[0], probe_point[1], probe_point[2]])?);
        let column = (shifted - base) / step;
        for r in 0..3 {
            out[r * 3 + j] = column[r];
        }
    }
    out.iter().all(|value| value.is_finite()).then_some(out)
}

/// Bit-level equality, so a reused objective value is only ever substituted
/// for an input the objective would have received identically. `==` on f64
/// would also match `-0.0` against `0.0`, which are distinct inputs to a
/// propagator.
#[inline]
fn bits_equal3(a: &[f64; 3], b: &[f64; 3]) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

/// Returns (`optimized_dv`, `miss_km`, iters, nfev, converged)
///
/// `dv_start` must already be inside the active L2 ball and is consumed
/// unchanged; re-projecting a boundary point is not bit-idempotent.
///
/// `seed` carries a `(point, miss)` pair the caller has ALREADY evaluated.
/// It is consumed only when the start point is bit-identical to `point`, in
/// which case `f` would return `miss` again and the call is a
/// duplicate high-fidelity propagation. Measured on the one-event strict-HF
/// harness: 550 of 8,604 release-control propagations were bit-identical
/// repeats, and the entry evaluation here is where they came from — the
/// skip-tolerance probe in `optimize_intercept_bounded` evaluates the seed,
/// and each bound expansion re-enters at a `best_x` already evaluated inside
/// the previous pass.
///
/// `nfev` still counts objective REQUESTS, not propagations, so every
/// diagnostic and receipt that reads it is unchanged.
#[cfg(test)]
fn lm_solve_bounded<F>(
    f: &F,
    model: JacobianModel<'_>,
    dv_start: Vector3<f64>,
    bound: f64,
    config: &BoundedInterceptConfig,
    seed: Option<([f64; 3], [f64; 3])>,
) -> Result<LmSolveOutcome, TransferPropagationFailure>
where
    F: Fn([f64; 3]) -> [f64; 3],
{
    let mut evaluated = |dv| HfInterceptEvaluation {
        miss: f(dv),
        endpoint: None,
    };
    let evaluated_seed = seed.map(|(point, miss)| {
        (
            point,
            HfInterceptEvaluation {
                miss,
                endpoint: None,
            },
        )
    });
    lm_solve_bounded_evaluated(
        &mut evaluated,
        model,
        dv_start,
        bound,
        config,
        evaluated_seed,
    )
}

#[expect(
    clippy::too_many_lines,
    reason = "the bounded LM is one ordered numerical state machine; splitting it risks changing evaluation and census transaction order"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "the pinned LM method deliberately uses IEEE-754 matrix/vector arithmetic and finite differences"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "all dynamic subscripts are constrained by fixed 3- or 6-dimensional LM basis loops"
)]
fn lm_solve_bounded_evaluated<F>(
    f: &mut F,
    model: JacobianModel<'_>,
    dv_start: Vector3<f64>,
    bound: f64,
    config: &BoundedInterceptConfig,
    seed: Option<([f64; 3], HfInterceptEvaluation)>,
) -> Result<LmSolveOutcome, TransferPropagationFailure>
where
    F: FnMut([f64; 3]) -> HfInterceptEvaluation,
{
    use lightyear_odeint_rs::probe;

    let census = probe::LM_CENSUS;
    let pass_id = if census {
        probe::lm_next_pass_id().map_err(TransferPropagationFailure::Census)?
    } else {
        0
    };
    let pass_props0 = if census {
        probe::tl_all_props().map_err(TransferPropagationFailure::Census)?
    } else {
        0
    };

    let sqrt_reg = config.reg_weight.sqrt();
    let mut x = dv_start;

    let start_point = [x[0], x[1], x[2]];
    let start_evaluation = match seed {
        Some((seed_point, seed_evaluation)) if bits_equal3(&seed_point, &start_point) => {
            seed_evaluation
        }
        _ => {
            let _rc = lightyear_odeint_rs::probe::census_scope(
                lightyear_odeint_rs::probe::TAG_RC_LM_ENTRY,
            );
            f(start_point)
        }
    };
    let mut miss_vec = start_evaluation.miss;
    let mut current_endpoint = start_evaluation.endpoint;
    let mut miss = Vector3::from_column_slice(&miss_vec);
    let mut residual = build_residual(&miss, &x, sqrt_reg);
    let mut cost = residual.norm_squared();
    let mut nfev = 1; // Initial evaluation

    // A non-finite objective at the seed (protected-radius sentinel or HF
    // propagation failure at an aggressive clamped guess) is unrecoverable:
    // NaN cost rejects every LM step. Zero delta-v — the timeline coast —
    // is always an admissible seed; restart there when it is finite.
    if !cost.is_finite() && x.norm() > 0.0 {
        let zero = Vector3::zeros();
        nfev = checked_lm_count_add(nfev, 1)?;
        let zero_evaluation = {
            let _rc = lightyear_odeint_rs::probe::census_scope(
                lightyear_odeint_rs::probe::TAG_RC_ZERO_DV,
            );
            f([0.0; 3])
        };
        let restart_miss_data = zero_evaluation.miss;
        let restart_miss = Vector3::from_column_slice(&restart_miss_data);
        let restart_residual = build_residual(&restart_miss, &zero, sqrt_reg);
        let restart_cost = restart_residual.norm_squared();
        if restart_cost.is_finite() {
            x = zero;
            miss_vec = restart_miss_data;
            miss = restart_miss;
            residual = restart_residual;
            cost = restart_cost;
            current_endpoint = zero_evaluation.endpoint;
        }
    }

    let mut lambda = config.damping_initial;
    let mut converged = false;
    let mut iters = 0;

    // Best solution tracking
    let mut best_x = x;
    let mut best_miss_km = miss.norm();
    // Carried so the caller can seed a bound-expansion re-entry at this exact
    // point. Only the NORM used to be returned, and a norm cannot seed the
    // objective, so every expansion pass re-propagated a point this pass had
    // already evaluated.
    let mut best_miss_vec = miss_vec;
    let mut best_endpoint = current_endpoint;

    let pass_start_miss_km = miss.norm();

    for (i, iters_this_pass) in (1..=config.max_iters).enumerate() {
        iters = iters_this_pass;

        let iter_props0 = if census {
            probe::tl_all_props().map_err(TransferPropagationFailure::Census)?
        } else {
            0
        };
        let mut fd_dmiss_km = [0.0f64; 3];
        let mut fd_step_kms = [0.0f64; 3];

        let miss_norm = miss.norm();
        if miss_norm < config.tol {
            converged = true;
            break;
        }

        // The iterate the Jacobian is built AT. `x` moves when the damping
        // loop below accepts a step, so it cannot be read after the fact.
        let jac_x = x;
        let iter_cost = cost;
        let mut fd_jac = [0.0f64; 9];
        let mut fd_svals = [0.0f64; 3];
        let mut solved = false;
        let mut accepted_step_norm = f64::INFINITY;
        let mut trial_evals = 0u32;

        // Build 6x3 Jacobian
        // Rows 0-2: the miss vector's derivative, from `model` when the caller
        //           supplied one and from a finite difference of the objective
        //           otherwise
        // Rows 3-5: sqrt(reg) * I_3 (analytical)
        let mut jac = Matrix6x3::zeros();

        // Computed only on the modelled path; with no model the real-objective
        // finite-difference loop runs directly. Bit-identity and propagation
        // counts recorded before the model Jacobian landed described projected
        // probes and do not apply here — the feature that once forced that
        // older path (`lm-fd-jacobian`) was removed 2026-08-05, having never
        // been enabled by any build. Current ambient behavior was explicitly
        // accepted under its own scoped authorization.
        let modelled = model.and_then(|model| {
            for j in 0..3 {
                fd_step_kms[j] = (x[j] + config.jac_eps) - x[j];
            }
            model_jacobian(model, &x, config.jac_eps)
        });

        let jacobian_route;
        if let Some(modelled) = modelled {
            jacobian_route = probe::LmJacobianRoute::Model;
            for r in 0..3 {
                for c in 0..3 {
                    jac[(r, c)] = modelled[r * 3 + c];
                }
            }
        } else {
            jacobian_route = if model.is_some() {
                probe::LmJacobianRoute::RealFdFallback
            } else {
                probe::LmJacobianRoute::RealFdNoModel
            };
            for j in 0..3 {
                let mut x_eps = x;
                x_eps[j] += config.jac_eps;

                nfev = checked_lm_count_add(nfev, 1)?;
                let miss_eps_vec = {
                    let _rc = lightyear_odeint_rs::probe::census_scope(
                        lightyear_odeint_rs::probe::TAG_RC_FD_JACOBIAN,
                    );
                    f([x_eps[0], x_eps[1], x_eps[2]]).miss
                };
                let miss_eps = Vector3::from_column_slice(&miss_eps_vec);

                let actual_eps = x_eps[j] - x[j];
                fd_step_kms[j] = actual_eps;
                if census {
                    fd_dmiss_km[j] = (miss_eps - miss).norm();
                }
                if actual_eps.abs() > 1e-12 {
                    let dmiss = (miss_eps - miss) / actual_eps;
                    jac[(0, j)] = dmiss[0];
                    jac[(1, j)] = dmiss[1];
                    jac[(2, j)] = dmiss[2];
                }
            }
        }

        // Analytical regularization Jacobian: sqrt(reg) * I_3
        jac[(3, 0)] = sqrt_reg;
        jac[(4, 1)] = sqrt_reg;
        jac[(5, 2)] = sqrt_reg;

        // Solve (J^T J + λI) dx = -J^T r
        let jt = jac.transpose();
        let jtj = jt * jac;
        let b = -jt * residual;

        if census {
            let block = Matrix3::new(
                jac[(0, 0)],
                jac[(0, 1)],
                jac[(0, 2)],
                jac[(1, 0)],
                jac[(1, 1)],
                jac[(1, 2)],
                jac[(2, 0)],
                jac[(2, 1)],
                jac[(2, 2)],
            );
            for r in 0..3 {
                for c in 0..3 {
                    fd_jac[r * 3 + c] = block[(r, c)];
                }
            }
            let s = block.singular_values();
            fd_svals = [s[0], s[1], s[2]];
        }

        // Try increasing damping until we find a descending step
        for _ in 0..10 {
            let mut lhs = jtj;
            for k in 0..3 {
                lhs[(k, k)] += lambda;
            }

            if let Some(dx) = solve_lm_normal_step(lhs, b) {
                let mut x_new = x + dx;
                project_to_ball(&mut x_new, bound);

                trial_evals = checked_lm_trial_evals_add(trial_evals)?;
                nfev = checked_lm_count_add(nfev, 1)?;
                let new_evaluation = {
                    let _rc = lightyear_odeint_rs::probe::census_scope(
                        lightyear_odeint_rs::probe::TAG_RC_TRIAL_STEP,
                    );
                    f([x_new[0], x_new[1], x_new[2]])
                };
                let miss_new_vec = new_evaluation.miss;
                let miss_new = Vector3::from_column_slice(&miss_new_vec);
                let residual_new = build_residual(&miss_new, &x_new, sqrt_reg);
                let cost_new = residual_new.norm_squared();

                if cost_new < cost {
                    // Accept step
                    accepted_step_norm = (x_new - x).norm();
                    x = x_new;
                    miss = miss_new;
                    residual = residual_new;
                    cost = cost_new;
                    current_endpoint = new_evaluation.endpoint;
                    lambda /= config.damping_factor;
                    solved = true;

                    // Update best
                    let miss_km = miss.norm();
                    if miss_km < best_miss_km {
                        best_x = x;
                        best_miss_km = miss_km;
                        best_miss_vec = miss_new_vec;
                        best_endpoint = current_endpoint;
                    }
                    break;
                }
            }
            lambda *= config.damping_factor;
        }

        if census {
            let iter = checked_lm_census_u32(i)?;
            let props = checked_lm_census_prop_delta(
                probe::tl_all_props().map_err(TransferPropagationFailure::Census)?,
                iter_props0,
            )?;
            let row = probe::LmIterRow {
                pass: pass_id,
                iter,
                props,
                miss_km: miss_norm,
                cost: iter_cost,
                fd_dmiss_km,
                fd_step_kms,
                fd_svals,
                fd_jac,
                x_kms: [jac_x[0], jac_x[1], jac_x[2]],
                jacobian_route,
                trial_evals,
                accepted: solved,
            };
            probe::record_lm_iter(row);
        }

        if !solved {
            break;
        }

        // Check tolerance on the displacement that was actually accepted.
        if accepted_step_norm < config.step_tol {
            converged = true;
            break;
        }
    }

    if census {
        let props = checked_lm_census_prop_delta(
            probe::tl_all_props().map_err(TransferPropagationFailure::Census)?,
            pass_props0,
        )?;
        let iters = checked_lm_census_u32(iters)?;
        let nfev = checked_lm_census_u32(nfev)?;
        let row = probe::LmPassRow {
            pass: pass_id,
            props,
            iters,
            nfev,
            converged,
            bound_kms: bound,
            start_miss_km: pass_start_miss_km,
            best_miss_km,
        };
        probe::record_lm_pass(row);
    }

    Ok(LmSolveOutcome {
        x: best_x,
        miss_km: best_miss_km,
        miss_vec: best_miss_vec,
        endpoint: best_endpoint,
        iters,
        nfev,
        converged,
    })
}

/// L2-ball-constrained Levenberg-Marquardt optimizer with regularization.
///
/// Replaces `scipy.optimize.least_squares` for dust intercept optimization.
///
/// # Algorithm
/// - Minimizes 6-element residual: [`miss_xyz`, sqrt(reg)*`dv_xyz`]
/// - L2-ball constraint: ||dv|| <= bound
/// - Adaptive bound expansion if solution at boundary
///
/// # Arguments
/// - `f`: Function computing miss vector given delta-V
/// - `dv_guess`: Initial delta-V guess [km/s]
/// - `config`: Optimizer configuration
///
/// # Returns
/// Optimization result with final delta-V, miss distance, and diagnostics
///
/// Differences the objective itself to build the Jacobian. Call sites whose
/// objective is a strict-HF propagation should use
/// [`optimize_intercept_bounded_with_model`] instead and hand it a cheap
/// [`JacobianModel`]; this form costs three propagations per LM iteration.
///
/// # Errors
///
/// Returns [`TransferPropagationFailure::ArithmeticOverflow`] if the bounded
/// optimizer's evaluation accounting cannot be represented.
pub fn optimize_intercept_bounded<F>(
    f: F,
    dv_guess: [f64; 3],
    config: &BoundedInterceptConfig,
) -> Result<BoundedInterceptResult, TransferPropagationFailure>
where
    F: FnMut([f64; 3]) -> [f64; 3],
{
    optimize_intercept_bounded_with_model(f, None, dv_guess, config)
}

/// [`optimize_intercept_bounded`], with a cheap stand-in for the objective used
/// only to build the LM Jacobian.
///
/// See [`JacobianModel`] for the search-path and science implications. Every
/// proposed step is accepted or rejected on the real objective, but changing
/// the Jacobian can still change the final selected result.
///
/// # Errors
///
/// Returns [`TransferPropagationFailure::ArithmeticOverflow`] if the bounded
/// optimizer's evaluation accounting cannot be represented.
pub fn optimize_intercept_bounded_with_model<F>(
    mut f: F,
    model: JacobianModel<'_>,
    dv_guess: [f64; 3],
    config: &BoundedInterceptConfig,
) -> Result<BoundedInterceptResult, TransferPropagationFailure>
where
    F: FnMut([f64; 3]) -> [f64; 3],
{
    Ok(optimize_intercept_bounded_evaluated(
        |dv| HfInterceptEvaluation {
            miss: f(dv),
            endpoint: None,
        },
        model,
        dv_guess,
        config,
    )?
    .intercept)
}

/// Bounded optimizer entry that retains the propagated endpoint.
///
/// The endpoint kept is the one belonging to the exact selected delta-V.
/// Solver decisions and the public result remain identical to
/// [`optimize_intercept_bounded_with_model`].
///
/// # Errors
///
/// Returns [`TransferPropagationFailure::ArithmeticOverflow`] if the bounded
/// optimizer's evaluation accounting cannot be represented.
pub fn optimize_intercept_bounded_hf_with_model<F>(
    f: F,
    model: JacobianModel<'_>,
    dv_guess: [f64; 3],
    config: &BoundedInterceptConfig,
) -> Result<BoundedHfInterceptResult, TransferPropagationFailure>
where
    F: FnMut([f64; 3]) -> HfInterceptEvaluation,
{
    optimize_intercept_bounded_evaluated(f, model, dv_guess, config)
}

fn optimize_intercept_bounded_evaluated<F>(
    mut f: F,
    model: JacobianModel<'_>,
    dv_guess: [f64; 3],
    config: &BoundedInterceptConfig,
) -> Result<BoundedHfInterceptResult, TransferPropagationFailure>
where
    F: FnMut([f64; 3]) -> HfInterceptEvaluation,
{
    let max_bound = config.max_bound.max(config.bound);
    let mut current_bound = config.bound.min(max_bound);
    let mut x0 = Vector3::from_column_slice(&dv_guess);
    project_to_ball(&mut x0, current_bound);
    let start = [x0.x, x0.y, x0.z];

    // Early skip check
    let initial_evaluation = {
        let _rc =
            lightyear_odeint_rs::probe::census_scope(lightyear_odeint_rs::probe::TAG_RC_SKIP_PROBE);
        f(start)
    };
    let miss0_vec = initial_evaluation.miss;
    let miss0 = Vector3::from_column_slice(&miss0_vec).norm();
    if miss0 < config.skip_tol.min(config.min_miss_km) {
        return Ok(BoundedHfInterceptResult {
            intercept: BoundedInterceptResult {
                dv: start,
                miss_km: miss0,
                iters: 0,
                nfev: 1,
                converged: true,
                success: miss0 < config.min_miss_km,
                final_bound: current_bound,
            },
            endpoint: initial_evaluation.endpoint,
        });
    }

    // First optimization pass
    // The skip probe above already evaluated the projected start. Hand that value down
    // rather than paying a second identical propagation for it.
    let first = lm_solve_bounded_evaluated(
        &mut f,
        model,
        x0,
        current_bound,
        config,
        Some((start, initial_evaluation)),
    )?;
    let (mut best_x, mut best_miss, mut total_iters, mut total_nfev, mut converged) = (
        first.x,
        first.miss_km,
        first.iters,
        first.nfev,
        first.converged,
    );
    let mut best_miss_vec = first.miss_vec;
    let mut best_endpoint = first.endpoint;

    // Adaptive bound expansion if at boundary and not converged
    for _ in 0..config.max_bound_expansions {
        // Check success condition
        if best_miss < config.min_miss_km {
            break;
        }

        // Check if at boundary
        if !at_boundary(&best_x, current_bound, 1e-9) {
            break;
        }

        // Expand bounds
        let new_bound = (current_bound * 2.0).min(max_bound);
        if new_bound <= current_bound + 1e-12 {
            break; // Already at cap
        }
        current_bound = new_bound;

        // Re-solve with expanded bounds
        // The prior pass returned `best_x` inside the old ball, so it is also
        // inside this larger ball and must remain bit-identical for seed reuse.
        let next = lm_solve_bounded_evaluated(
            &mut f,
            model,
            best_x,
            current_bound,
            config,
            Some((
                [best_x.x, best_x.y, best_x.z],
                HfInterceptEvaluation {
                    miss: best_miss_vec,
                    endpoint: best_endpoint,
                },
            )),
        )?;
        let (x_new, miss_new, iters_new, nfev_new, conv_new) =
            (next.x, next.miss_km, next.iters, next.nfev, next.converged);

        (total_iters, total_nfev) =
            checked_lm_totals(total_iters, total_nfev, iters_new, nfev_new)?;

        if miss_new < best_miss {
            best_x = x_new;
            best_miss = miss_new;
            best_miss_vec = next.miss_vec;
            best_endpoint = next.endpoint;
            converged = conv_new;
        }
    }

    let dv = [best_x.x, best_x.y, best_x.z];
    Ok(BoundedHfInterceptResult {
        intercept: BoundedInterceptResult {
            dv,
            miss_km: best_miss,
            iters: total_iters,
            nfev: total_nfev,
            converged,
            success: best_miss < config.min_miss_km,
            final_bound: current_bound,
        },
        endpoint: best_endpoint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_lm_counts_reject_overflow_before_result_construction() {
        assert_eq!(
            checked_lm_count_add(usize::MAX, 1),
            Err(TransferPropagationFailure::ArithmeticOverflow)
        );
        assert_eq!(
            checked_lm_totals(3, usize::MAX, 2, 1),
            Err(TransferPropagationFailure::ArithmeticOverflow)
        );
    }

    #[test]
    fn lm_trial_evaluation_count_rejects_overflow() {
        assert_eq!(
            checked_lm_trial_evals_add(u32::MAX),
            Err(TransferPropagationFailure::ArithmeticOverflow)
        );
    }

    #[test]
    fn lm_census_values_reject_nonrepresentable_or_reversed_observations() {
        assert_eq!(
            checked_lm_census_prop_delta(7, 8),
            Err(TransferPropagationFailure::ArithmeticOverflow)
        );
        let Ok(max_u32) = usize::try_from(u32::MAX) else {
            return;
        };
        assert_eq!(checked_lm_census_u32(max_u32), Ok(u32::MAX));
        if let Some(beyond_u32) = max_u32.checked_add(1) {
            assert_eq!(
                checked_lm_census_u32(beyond_u32),
                Err(TransferPropagationFailure::ArithmeticOverflow)
            );
        }
    }

    #[test]
    fn hf_miss_evaluation_retains_missing_assets() {
        let mut request = crate::types::TransferRequest::with_j2_closure_settings(
            crate::solve::J2ClosureSettings::default(),
        );
        request.epoch_jd = 2_460_000.5;
        request.execution_policy = crate::types::ExecutionPolicy {
            use_high_fidelity: true,
            require_high_fidelity: true,
            ..crate::types::ExecutionPolicy::default()
        };
        let ctx = PlanContext::from_request(request);

        assert!(matches!(
            compute_miss_vector_hf_with_endpoint(
                [0.0; 3],
                [0.0, 7.5, 0.0],
                [7000.0, 0.0, 0.0],
                [7000.0, 1.0, 0.0],
                60.0,
                ctx.epoch_jd,
                BodyForceConfig::high_fidelity(crate::types::BodyRole::Dust, 0.01, 2.2, 1.3),
                &ctx,
            ),
            Err(crate::evaluate::TransferPropagationFailure::MissingHighFidelityAssets)
        ));
    }

    #[test]
    fn lm_normal_step_matches_direct_solution_without_inverse() {
        let lhs = Matrix3::new(4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0);
        let b = Vector3::new(1.0, -2.0, 0.5);

        let step = solve_lm_normal_step(lhs, b).expect("positive definite solve");

        assert!((lhs * step - b).norm() < 1e-12);
    }

    #[test]
    fn lm_normal_step_rejects_singular_normal_system() {
        let lhs = Matrix3::zeros();
        let b = Vector3::new(1.0, 0.0, 0.0);

        assert!(solve_lm_normal_step(lhs, b).is_none());
    }

    #[test]
    fn model_jacobian_uses_ambient_axis_probes_at_ball_boundary() {
        let config = BoundedInterceptConfig {
            max_iters: 1,
            tol: 0.0,
            bound: 0.487_654_321,
            max_bound: 0.487_654_321,
            reg_weight: 0.0,
            ..BoundedInterceptConfig::default()
        };
        let start = [config.bound, 0.0, 0.0];
        let objective = |dv: [f64; 3]| [dv[0] - 0.25, dv[1], dv[2]];
        let model_inputs = std::sync::Mutex::new(Vec::new());
        let model = |dv: [f64; 3]| {
            model_inputs
                .lock()
                .expect("model input log mutex poisoned")
                .push(dv.map(f64::to_bits));
            Some(objective(dv))
        };
        let [start_x, start_y, start_z] = start;
        let mut expected = vec![start.map(f64::to_bits)];
        for probe in [
            [start_x + config.jac_eps, start_y, start_z],
            [start_x, start_y + config.jac_eps, start_z],
            [start_x, start_y, start_z + config.jac_eps],
        ] {
            expected.push(probe.map(f64::to_bits));
        }

        let jacobian = model_jacobian(&model, &Vector3::from_column_slice(&start), config.jac_eps)
            .expect("ambient linear model must produce a Jacobian");

        assert_eq!(
            model_inputs
                .into_inner()
                .expect("model input log mutex poisoned"),
            expected,
            "the analytic Jacobian must difference the ambient miss map"
        );
        assert_eq!(
            jacobian.map(f64::to_bits),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0].map(f64::to_bits),
            "the ambient derivative of the identity-offset model must be I"
        );
    }

    #[test]
    fn real_fd_jacobian_uses_ambient_axis_probes_at_ball_boundary() {
        let config = BoundedInterceptConfig {
            max_iters: 1,
            tol: 0.0,
            bound: 0.476_543_219,
            max_bound: 0.476_543_219,
            reg_weight: 0.0,
            ..BoundedInterceptConfig::default()
        };
        let start = [config.bound, 0.0, 0.0];
        let objective_inputs = std::cell::RefCell::new(Vec::new());
        let objective = |dv: [f64; 3]| {
            objective_inputs.borrow_mut().push(dv.map(f64::to_bits));
            [dv[0] - 0.25, dv[1] + 0.1, dv[2] - 0.2]
        };
        let [start_x, start_y, start_z] = start;
        let mut expected = vec![start.map(f64::to_bits)];
        for probe in [
            [start_x + config.jac_eps, start_y, start_z],
            [start_x, start_y + config.jac_eps, start_z],
            [start_x, start_y, start_z + config.jac_eps],
        ] {
            expected.push(probe.map(f64::to_bits));
        }

        let _ = lm_solve_bounded(
            &objective,
            None,
            Vector3::from_column_slice(&start),
            config.bound,
            &config,
            None,
        )
        .expect("real finite-difference Jacobian fixture must solve");

        let inputs = objective_inputs.into_inner();
        assert!(
            inputs.len() >= expected.len(),
            "missing real finite-difference probes: {inputs:?}"
        );
        assert_eq!(
            inputs.get(..expected.len()),
            Some(expected.as_slice()),
            "the real finite-difference Jacobian must use ambient axis probes"
        );
    }

    #[test]
    fn lm_census_reports_the_jacobian_route_and_actual_real_fd_steps() {
        use lightyear_odeint_rs::probe;

        // In a default build this test asserts NOTHING: `LM_CENSUS` is a
        // `const false` without `lightyear_odeint_rs/prop-census`, and the
        // green you see in a plain workspace sweep is this early return, not
        // coverage. The armed lane is real and documented -- the closeout plan
        // runs the intercept suite BOTH ways (`--features
        // lightyear_odeint_rs/prop-census`), and `cfg`-gating this fn on a
        // forwarded two_phase feature would silently drop it from that
        // documented invocation, which selects features on lightyear only.
        if !probe::LM_CENSUS {
            return;
        }

        let config = BoundedInterceptConfig {
            max_iters: 1,
            tol: 0.0,
            bound: 0.5,
            max_bound: 0.5,
            reg_weight: 0.0,
            ..BoundedInterceptConfig::default()
        };
        let objective = |dv: [f64; 3]| [dv[0] - 0.25, dv[1], dv[2]];
        let row_at = |start: [f64; 3]| {
            probe::lm_iter_rows()
                .expect("LM census rows must remain readable during this test")
                .into_iter()
                .find(|row| bits_equal3(&row.x_kms, &start))
                .unwrap_or_else(|| panic!("no census row at unique test start {start:?}"))
        };
        let assert_route_token = |route: probe::LmJacobianRoute, expected: &str| {
            assert_eq!(
                route.as_str(),
                expected,
                "diagnostic route token must be stable"
            );
        };
        let assert_interior_steps = |row: probe::LmIterRow, start: [f64; 3]| {
            let recorded_steps = row.fd_step_kms;
            for (axis, (recorded_step, start_component)) in
                recorded_steps.into_iter().zip(start).enumerate()
            {
                let actual = (start_component + config.jac_eps) - start_component;
                assert_eq!(
                    recorded_step.to_bits(),
                    actual.to_bits(),
                    "axis {axis} did not record the exact real-FD displacement: {row:?}"
                );
            }
        };

        let no_model_start = [0.012_345_678_9, -0.023_456_789, 0.034_567_89];
        let _ = lm_solve_bounded(
            &objective,
            None,
            Vector3::from_column_slice(&no_model_start),
            config.bound,
            &config,
            None,
        )
        .expect("no-model LM census fixture must solve");
        let no_model = row_at(no_model_start);

        let declining_model = |_dv: [f64; 3]| None;
        let declined_start = [-0.043_210_987_6, 0.032_109_876_5, -0.021_098_765_4];
        let _ = lm_solve_bounded(
            &objective,
            Some(&declining_model),
            Vector3::from_column_slice(&declined_start),
            config.bound,
            &config,
            None,
        )
        .expect("declined-model LM census fixture must solve");
        let declined = row_at(declined_start);

        let nonfinite_model = |_dv: [f64; 3]| Some([f64::NAN, 0.0, 0.0]);
        let nonfinite_start = [0.052_109_876_5, -0.041_098_765_4, 0.030_987_654_3];
        let _ = lm_solve_bounded(
            &objective,
            Some(&nonfinite_model),
            Vector3::from_column_slice(&nonfinite_start),
            config.bound,
            &config,
            None,
        )
        .expect("nonfinite-model LM census fixture must solve");
        let nonfinite = row_at(nonfinite_start);

        let linear_model = |dv: [f64; 3]| Some(objective(dv));
        let active_bound = 0.487_654_321;
        let active_config = BoundedInterceptConfig {
            bound: active_bound,
            max_bound: active_bound,
            ..config
        };
        let active_start = [active_bound, 0.0, 0.0];
        let _ = lm_solve_bounded(
            &objective,
            Some(&linear_model),
            Vector3::from_column_slice(&active_start),
            active_config.bound,
            &active_config,
            None,
        )
        .expect("active-model LM census fixture must solve");
        let active_model = row_at(active_start);

        let active_fd_bound = 0.465_432_109;
        let active_fd_config = BoundedInterceptConfig {
            bound: active_fd_bound,
            max_bound: active_fd_bound,
            ..config
        };
        let active_fd_start = [active_fd_bound, 0.0, 0.0];
        let _ = lm_solve_bounded(
            &objective,
            None,
            Vector3::from_column_slice(&active_fd_start),
            active_fd_config.bound,
            &active_fd_config,
            None,
        )
        .expect("active real-FD LM census fixture must solve");
        let active_fd = row_at(active_fd_start);

        let model_start = [0.061_728_394_5, -0.017_283_945_6, 0.028_394_561_7];
        let _ = lm_solve_bounded(
            &objective,
            Some(&linear_model),
            Vector3::from_column_slice(&model_start),
            config.bound,
            &config,
            None,
        )
        .expect("model LM census fixture must solve");
        let model = row_at(model_start);

        assert_route_token(model.jacobian_route, "model");
        assert_route_token(no_model.jacobian_route, "real_fd_no_model");
        assert_route_token(declined.jacobian_route, "real_fd_fallback");
        assert_route_token(nonfinite.jacobian_route, "real_fd_fallback");
        assert_route_token(active_model.jacobian_route, "model");
        assert_route_token(active_fd.jacobian_route, "real_fd_no_model");
        assert_interior_steps(no_model, no_model_start);
        assert_interior_steps(declined, declined_start);
        assert_interior_steps(nonfinite, nonfinite_start);

        let active_model_steps = active_model.fd_step_kms;
        let active_fd_steps = active_fd.fd_step_kms;
        for (axis, (((model_step, active_component), fd_step), fd_component)) in active_model_steps
            .into_iter()
            .zip(active_start)
            .zip(active_fd_steps)
            .zip(active_fd_start)
            .enumerate()
        {
            let actual = (active_component + active_config.jac_eps) - active_component;
            assert_eq!(
                model_step.to_bits(),
                actual.to_bits(),
                "active-bound axis {axis} did not record the ambient displacement"
            );
            let actual_fd = (fd_component + active_fd_config.jac_eps) - fd_component;
            assert_eq!(
                fd_step.to_bits(),
                actual_fd.to_bits(),
                "active-bound real-FD axis {axis} did not record the ambient displacement"
            );
        }
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0].map(f64::to_bits);
        assert_eq!(
            active_model.fd_jac.map(f64::to_bits),
            identity,
            "the analytic boundary Jacobian must equal I"
        );
        assert_eq!(
            active_fd.fd_jac.map(f64::to_bits),
            identity,
            "the real-FD boundary Jacobian must equal I"
        );
    }

    #[test]
    fn bounded_hf_intercept_carries_the_endpoint_for_its_exact_returned_dv() {
        let config = BoundedInterceptConfig {
            max_iters: 12,
            tol: 1e-12,
            step_tol: 1e-12,
            jac_eps: 1e-5,
            damping_initial: 1e-4,
            damping_factor: 10.0,
            bound: 0.5,
            max_bound: 0.5,
            reg_weight: 0.0,
            skip_tol: 0.0,
            min_miss_km: 1e-8,
            max_bound_expansions: 0,
        };
        let result = optimize_intercept_bounded_hf_with_model(
            |dv| HfInterceptEvaluation {
                miss: [dv[0] - 0.125, dv[1] + 0.25, dv[2] - 0.0625],
                endpoint: Some([dv[0], dv[1], dv[2], 4.0, 5.0, 6.0]),
            },
            None,
            [0.2, -0.2, 0.1],
            &config,
        )
        .expect("bounded HF endpoint fixture must solve");

        assert!(result.intercept.success);
        let endpoint = result
            .endpoint_for_returned_dv()
            .expect("bounded HF winner must retain its exact endpoint");
        let [endpoint_x, endpoint_y, endpoint_z, _, _, _] = endpoint;
        assert_eq!(
            [endpoint_x, endpoint_y, endpoint_z].map(f64::to_bits),
            result.intercept.dv.map(f64::to_bits)
        );
    }

    #[test]
    fn bounded_intercept_does_not_skip_above_success_tolerance() {
        let miss_fn = |dv: [f64; 3]| {
            let correction = 1_000.0 * dv[0];
            [0.5 - correction, 0.0, 0.0]
        };
        let config = BoundedInterceptConfig {
            skip_tol: 1.0,
            min_miss_km: 0.01,
            bound: 0.1,
            reg_weight: 0.0,
            ..BoundedInterceptConfig::default()
        };

        let result = optimize_intercept_bounded(miss_fn, [0.0; 3], &config)
            .expect("bounded intercept tolerance fixture must solve");

        assert!(result.success, "result: {result:?}");
        assert!(result.miss_km < config.min_miss_km);
        assert!(result.nfev > 1, "unsafe seed must be refined");
    }

    #[test]
    fn bounded_intercept_projects_start_before_early_success() {
        let inputs = std::cell::RefCell::new(Vec::new());
        let config = BoundedInterceptConfig {
            bound: 1.0,
            max_bound: 1.0,
            skip_tol: 1.0,
            min_miss_km: 1.0,
            ..BoundedInterceptConfig::default()
        };
        let expected = [2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0];

        let result = optimize_intercept_bounded(
            |dv| {
                inputs.borrow_mut().push(dv.map(f64::to_bits));
                [0.0; 3]
            },
            [2.0, 2.0, 1.0],
            &config,
        )
        .expect("projected-start fixture must solve");

        assert_eq!(
            inputs.into_inner(),
            vec![expected.map(f64::to_bits)],
            "the early-success probe must evaluate only the projected start"
        );
        assert_eq!(
            result.dv.map(f64::to_bits),
            expected.map(f64::to_bits),
            "early success must return the projected start"
        );
        assert_eq!(result.nfev, 1);
        assert_eq!(result.iters, 0);
        assert!(result.converged);
        assert!(result.success);
    }

    #[test]
    fn bounded_intercept_reuses_non_idempotent_projected_start() {
        let raw_start = Vector3::new(
            -8.303_602_981_421_536,
            -2.021_223_470_826_829,
            -2.867_483_738_616_087_4,
        );
        let mut projected_once = raw_start;
        project_to_ball(&mut projected_once, 1.0);
        let mut projected_twice = projected_once;
        project_to_ball(&mut projected_twice, 1.0);
        let projected_once_bits = [
            projected_once.x.to_bits(),
            projected_once.y.to_bits(),
            projected_once.z.to_bits(),
        ];
        let projected_twice_bits = [
            projected_twice.x.to_bits(),
            projected_twice.y.to_bits(),
            projected_twice.z.to_bits(),
        ];
        assert_ne!(
            projected_once_bits, projected_twice_bits,
            "the hostile vector must distinguish one projection from two"
        );

        let inputs = std::cell::RefCell::new(Vec::new());
        let config = BoundedInterceptConfig {
            max_iters: 0,
            bound: 1.0,
            max_bound: 1.0,
            skip_tol: 0.0,
            min_miss_km: 0.0,
            ..BoundedInterceptConfig::default()
        };
        let result = optimize_intercept_bounded(
            |dv| {
                inputs.borrow_mut().push(dv.map(f64::to_bits));
                [1.0, 0.0, 0.0]
            },
            [raw_start.x, raw_start.y, raw_start.z],
            &config,
        )
        .expect("non-idempotent projected-start fixture must solve");

        assert_eq!(
            inputs.into_inner(),
            vec![projected_once_bits],
            "the first LM pass must reuse the public projected-start evaluation"
        );
        assert_eq!(
            result.dv.map(f64::to_bits),
            projected_once_bits,
            "the first LM pass must preserve the first projection bits"
        );
        assert_eq!(result.nfev, 1);
        assert_eq!(result.iters, 0);
    }

    #[test]
    fn bounded_intercept_converges_on_small_accepted_projected_displacement() {
        let config = BoundedInterceptConfig {
            max_iters: 5,
            tol: 0.0,
            step_tol: 2.0e-4,
            jac_eps: 1.0e-6,
            damping_initial: 1.0e-12,
            bound: 1.0,
            max_bound: 1.0,
            reg_weight: 0.0,
            skip_tol: 0.0,
            min_miss_km: 0.0,
            max_bound_expansions: 0,
            ..BoundedInterceptConfig::default()
        };
        let objective = |dv: [f64; 3]| [dv[0] - 2.0, dv[1], dv[2]];
        let model = |dv: [f64; 3]| Some(objective(dv));

        let result = optimize_intercept_bounded_with_model(
            objective,
            Some(&model),
            [0.999_9, 0.0, 0.0],
            &config,
        )
        .expect("small accepted-displacement fixture must solve");

        assert_eq!(
            result.dv.map(f64::to_bits),
            [1.0, 0.0, 0.0].map(f64::to_bits),
            "the accepted outward step must return the projected boundary point"
        );
        assert_eq!(result.miss_km.to_bits(), 1.0f64.to_bits());
        assert_eq!(
            result.iters, 1,
            "convergence must occur on first acceptance"
        );
        assert!(result.converged, "result: {result:?}");
        assert!(
            !result.success,
            "step convergence must not imply miss success"
        );
    }

    #[test]
    fn bounded_intercept_expands_to_configured_physical_ceiling() {
        let cfg = BoundedInterceptConfig {
            bound: 1.0,
            max_bound: 7.5,
            max_bound_expansions: 7,
            min_miss_km: 1.0e-6,
            skip_tol: 1.0e-6,
            reg_weight: 0.0,
            ..BoundedInterceptConfig::default()
        };
        let result = optimize_intercept_bounded(|dv| [dv[0] - 3.0, dv[1], dv[2]], [0.0; 3], &cfg)
            .expect("configured-ceiling reachable fixture must solve");
        assert!(result.success, "{result:?}");
        assert!((result.dv[0] - 3.0).abs() < 1.0e-6, "{result:?}");
        assert!(result.final_bound >= 3.0);

        let unreachable =
            optimize_intercept_bounded(|dv| [dv[0] - 8.0, dv[1], dv[2]], [0.0; 3], &cfg)
                .expect("configured-ceiling unreachable fixture must solve");
        assert!(!unreachable.success, "{unreachable:?}");
        assert!(unreachable.final_bound <= 7.5);
    }

    #[test]
    fn bounded_intercept_recovers_from_nan_seed_via_zero_dv() {
        // Objective: NaN outside a 0.2-radius ball around zero (e.g. a
        // protected-radius sentinel), quadratic-in-dv miss inside it with the
        // solution at dv = (0.05, 0, 0). A clamped far guess (norm 0.95)
        // lands in NaN-land; the solver must recover via the zero-dv seed.
        let f = |dv: [f64; 3]| {
            let x_squared = dv[0] * dv[0];
            let y_squared = dv[1] * dv[1];
            let z_squared = dv[2] * dv[2];
            let first_two_square_sum = x_squared + y_squared;
            let n = (first_two_square_sum + z_squared).sqrt();
            if n > 0.2 {
                return [f64::NAN; 3];
            }
            [10.0 * (dv[0] - 0.05), 10.0 * dv[1], 10.0 * dv[2]]
        };
        let cfg = BoundedInterceptConfig {
            bound: 1.0,
            max_bound: 1.0,
            min_miss_km: 1e-3,
            ..BoundedInterceptConfig::default()
        };
        let res = optimize_intercept_bounded(f, [0.55, 0.55, 0.55], &cfg)
            .expect("NaN-seed recovery fixture must solve");
        assert!(
            res.success,
            "expected zero-seed recovery, got miss={} dv={:?}",
            res.miss_km, res.dv
        );
        assert!((res.dv[0] - 0.05).abs() < 1e-3);
    }

    #[test]
    fn zero_recovery_carries_its_actual_evaluation_through_internal_seed_reuse() {
        fn scope_entries(report: &str, tag: &str) -> Option<u64> {
            let prefix = format!("PROP_SCOPE {tag},entries,");
            report
                .lines()
                .find_map(|line| line.strip_prefix(&prefix))
                .and_then(|entries| entries.parse().ok())
        }

        let scope_before_report = lightyear_odeint_rs::probe::report()
            .expect("census report must remain available during this test");
        let census_enabled = lightyear_odeint_rs::probe::LM_CENSUS;
        let zero_scope_before = if census_enabled {
            for tag in [
                "rc_skip_probe",
                "rc_lm_entry",
                "rc_zero_dv",
                "rc_fd_jacobian",
                "rc_trial_step",
            ] {
                assert!(
                    scope_entries(&scope_before_report, tag).is_some(),
                    "missing stable PROP_SCOPE {tag} row"
                );
            }
            scope_entries(&scope_before_report, "rc_zero_dv").unwrap()
        } else {
            0
        };

        let inputs = std::cell::RefCell::new(Vec::new());
        let start = [0.1, 0.0, 0.0];
        let zero_miss = [3.0, 4.0, 0.0];
        let zero_endpoint = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut objective = |dv: [f64; 3]| {
            inputs.borrow_mut().push(dv.map(f64::to_bits));
            if bits_equal3(&dv, &start) {
                HfInterceptEvaluation {
                    miss: [f64::NAN; 3],
                    endpoint: None,
                }
            } else if bits_equal3(&dv, &[0.0; 3]) {
                HfInterceptEvaluation {
                    miss: zero_miss,
                    endpoint: Some(zero_endpoint),
                }
            } else {
                panic!("unexpected objective input: {dv:?}");
            }
        };
        let config = BoundedInterceptConfig {
            max_iters: 0,
            ..BoundedInterceptConfig::default()
        };

        let recovered = lm_solve_bounded_evaluated(
            &mut objective,
            None,
            Vector3::from_column_slice(&start),
            config.bound,
            &config,
            None,
        )
        .expect("zero-recovery fixture must solve");

        assert_eq!(recovered.x, Vector3::zeros());
        assert_eq!(recovered.miss_km.to_bits(), 5.0f64.to_bits());
        assert_eq!(
            recovered.miss_vec.map(f64::to_bits),
            zero_miss.map(f64::to_bits)
        );
        assert_eq!(recovered.endpoint, Some(zero_endpoint));
        assert_eq!(recovered.nfev, 2);
        assert_eq!(
            *inputs.borrow(),
            vec![start.map(f64::to_bits), [0.0; 3].map(f64::to_bits)],
            "recovery route must evaluate only the failed start then zero"
        );

        // The outer optimizer stops here because zero is interior to every
        // positive bound. Exercise the inner seed seam directly so future or
        // internal reuse cannot attach the failed start's miss to zero.
        let seeded = lm_solve_bounded_evaluated(
            &mut objective,
            None,
            recovered.x,
            config.max_bound,
            &config,
            Some((
                [recovered.x.x, recovered.x.y, recovered.x.z],
                HfInterceptEvaluation {
                    miss: recovered.miss_vec,
                    endpoint: recovered.endpoint,
                },
            )),
        )
        .expect("zero-recovery seed-reuse fixture must solve");

        assert_eq!(seeded.x, Vector3::zeros());
        assert_eq!(seeded.miss_km.to_bits(), 5.0f64.to_bits());
        assert_eq!(
            seeded.miss_vec.map(f64::to_bits),
            zero_miss.map(f64::to_bits)
        );
        assert_eq!(seeded.endpoint, Some(zero_endpoint));
        assert_eq!(
            seeded.nfev, 1,
            "internal seed reuse keeps logical objective-request accounting"
        );
        assert_eq!(
            inputs.borrow().len(),
            2,
            "internal seed reuse must keep the exact recovered zero evaluation"
        );
        if census_enabled {
            let scope_after_report = lightyear_odeint_rs::probe::report()
                .expect("census report must remain available during this test");
            let zero_scope_after = scope_entries(&scope_after_report, "rc_zero_dv")
                .expect("stable PROP_SCOPE rc_zero_dv row");
            assert!(
                zero_scope_after > zero_scope_before,
                "zero-recovery branch entry must increment its direct scope census"
            );
        }
    }

    #[test]
    fn bounded_intercept_enforces_l2_norm_not_per_axis_box() {
        // A diagonal target reachable at dv = [0.75, 0.75, 0.75] sits inside
        // the per-axis box for bound=1.0 but at ||dv|| = 1.299 breaches the
        // physical ceiling; the solver must never return an accepted result
        // whose norm exceeds the (non-expandable) bound.
        let cfg = BoundedInterceptConfig {
            bound: 1.0,
            max_bound: 1.0,
            max_bound_expansions: 0,
            min_miss_km: 1.0e-6,
            skip_tol: 1.0e-9,
            reg_weight: 0.0,
            ..BoundedInterceptConfig::default()
        };
        let result = optimize_intercept_bounded(
            |dv| [dv[0] - 0.75, dv[1] - 0.75, dv[2] - 0.75],
            [0.0; 3],
            &cfg,
        )
        .expect("L2-bound fixture must solve");
        let x_squared = result.dv[0] * result.dv[0];
        let y_squared = result.dv[1] * result.dv[1];
        let z_squared = result.dv[2] * result.dv[2];
        let first_two_square_sum = x_squared + y_squared;
        let norm = (first_two_square_sum + z_squared).sqrt();
        assert!(
            norm <= 1.0 + 1.0e-9,
            "returned dv norm {norm} exceeds the physical bound: {result:?}"
        );
        // The exact solution is outside the ball, so success must be false.
        assert!(!result.success, "{result:?}");
    }

    /// The seeded entry evaluation must remove exactly one objective call and
    /// change nothing else, bit for bit.
    ///
    /// This is the gate on the `seed` argument of [`lm_solve_bounded`]. The
    /// optimizer's iterates are a deterministic function of the objective
    /// values it sees, so substituting an ALREADY COMPUTED value for an input
    /// that is bit-identical cannot move the trajectory -- but only if the
    /// bit-identity check is real. A `==` comparison, a tolerance, or a stale
    /// seed would all silently feed the solver a value the objective would not
    /// have returned, and every downstream float would drift.
    #[test]
    fn seeded_entry_evaluation_is_bit_identical_and_saves_one_call() {
        let calls = std::cell::Cell::new(0usize);
        let objective = |dv: [f64; 3]| {
            calls.set(calls.get() + 1);
            // Nonlinear and asymmetric, so a wrong first residual diverges
            // rather than coincidentally agreeing.
            [
                dv[0].mul_add(3.0, -0.021) + dv[1] * dv[1],
                dv[1].mul_add(2.5, 0.013) + dv[2] * dv[0],
                dv[2].mul_add(4.0, -0.007) + dv[0] * dv[1],
            ]
        };
        let config = BoundedInterceptConfig {
            min_miss_km: 1.0e-9,
            skip_tol: 1.0e-9,
            bound: 0.5,
            max_bound: 0.5,
            max_bound_expansions: 0,
            ..BoundedInterceptConfig::default()
        };
        let start = Vector3::new(0.004, -0.002, 0.001);
        let point = [start.x, start.y, start.z];

        calls.set(0);
        let unseeded = lm_solve_bounded(&objective, None, start, config.bound, &config, None)
            .expect("unseeded entry-evaluation fixture must solve");
        let unseeded_calls = calls.get();

        let seed_miss = objective(point);
        calls.set(0);
        let seeded = lm_solve_bounded(
            &objective,
            None,
            start,
            config.bound,
            &config,
            Some((point, seed_miss)),
        )
        .expect("seeded entry-evaluation fixture must solve");
        let seeded_calls = calls.get();

        assert_eq!(
            seeded_calls + 1,
            unseeded_calls,
            "the seed must remove exactly one objective evaluation"
        );
        assert_eq!(
            seeded.x.map(f64::to_bits),
            unseeded.x.map(f64::to_bits),
            "seeded solve moved a dv component"
        );
        assert_eq!(
            seeded.miss_km.to_bits(),
            unseeded.miss_km.to_bits(),
            "miss moved"
        );
        assert_eq!(
            (seeded.iters, seeded.nfev, seeded.converged),
            (unseeded.iters, unseeded.nfev, unseeded.converged)
        );
        assert_eq!(
            seeded.miss_vec.map(f64::to_bits),
            unseeded.miss_vec.map(f64::to_bits),
            "seeded solve moved the returned miss vector"
        );
    }

    /// A seed for a DIFFERENT point must be ignored, not trusted.
    ///
    /// If a mismatched seed were accepted the solver would begin from a
    /// fabricated residual instead of evaluating its exact start point.
    #[test]
    fn mismatched_seed_is_ignored() {
        let calls = std::cell::Cell::new(0usize);
        let objective = |dv: [f64; 3]| {
            calls.set(calls.get() + 1);
            [dv[0] - 0.01, dv[1] + 0.02, dv[2] - 0.03]
        };
        let config = BoundedInterceptConfig {
            min_miss_km: 1.0e-9,
            skip_tol: 1.0e-9,
            max_bound_expansions: 0,
            ..BoundedInterceptConfig::default()
        };
        let start = Vector3::new(0.004, -0.002, 0.001);

        calls.set(0);
        let unseeded = lm_solve_bounded(&objective, None, start, config.bound, &config, None)
            .expect("unseeded stale-seed fixture must solve");
        let unseeded_calls = calls.get();

        calls.set(0);
        let poisoned = lm_solve_bounded(
            &objective,
            None,
            start,
            config.bound,
            &config,
            // Same value in every slot but one ulp off in the first.
            Some((
                [f64::from_bits(start.x.to_bits() + 1), start.y, start.z],
                [1.0e6, 1.0e6, 1.0e6],
            )),
        )
        .expect("mismatched-seed fixture must solve");
        assert_eq!(calls.get(), unseeded_calls, "a stale seed must not be used");
        assert_eq!(poisoned.x.map(f64::to_bits), unseeded.x.map(f64::to_bits));
    }

    /// A row sitting ON the protected-radius floor still solves when the
    /// Jacobian comes from a model that knows nothing about that floor.
    ///
    /// This is the corner the production `jacobian_model` deliberately leaves
    /// out. `solve_intercept_delta_dv` hands the LM a SMOOTH Keplerian model
    /// while the objective keeps the 1e6-km violation penalty, because
    /// mirroring the penalty into the model reproduces its rank-1 conditioning
    /// pathology and measurably cost rows. The argument for doing that is that
    /// enforcement belongs to the objective: the solver may propose a step into
    /// the forbidden region, but the trial evaluation returns the penalty, the
    /// cost rises, and the step is rejected.
    ///
    /// That argument is what this test pins, on the geometry the penalty was
    /// written for: the solver STARTS inside the forbidden region, where the
    /// objective returns the flat 1e6 sentinel and its own finite difference
    /// therefore measures the penalty slope rather than the physics.
    #[test]
    fn smooth_model_cannot_climb_out_of_a_penalty_region_it_starts_in() {
        // Feasible iff dv[0] >= floor. The unconstrained optimum is FEASIBLE,
        // at 0.25, and the start point at 0.05 is not.
        let floor = 0.10_f64;
        let optimum = 0.25_f64;
        let start = [0.05, 0.02, 0.02];
        let miss_of = |dv: [f64; 3]| [(dv[0] - optimum) * 1.0e3, dv[1] * 1.0e3, dv[2] * 1.0e3];
        // Answers everywhere, exactly like the production `jacobian_model`.
        let model = |dv: [f64; 3]| Some(miss_of(dv));
        // The OBJECTIVE walls off the violation exactly as release control does.
        let objective = |dv: [f64; 3]| -> [f64; 3] {
            if dv[0] < floor {
                let violation_penalty = 1.0e3 * (floor - dv[0]);
                let p = 1.0e6 + violation_penalty;
                return [p, p, p];
            }
            miss_of(dv)
        };
        let config = BoundedInterceptConfig {
            max_iters: 50,
            tol: 1.0e-5,
            bound: 0.5,
            max_bound: 0.5,
            min_miss_km: 1.0e4,
            skip_tol: 0.0,
            max_bound_expansions: 0,
            ..BoundedInterceptConfig::default()
        };

        let modelled =
            optimize_intercept_bounded_with_model(objective, Some(&model), start, &config)
                .expect("penalty-region model fixture must solve");
        let differenced = optimize_intercept_bounded(objective, start, &config)
            .expect("penalty-region real-FD fixture must solve");

        // KNOWN LIMIT, pinned. Starting INSIDE the region, the objective's
        // flat sentinel and the model's finite slope are inconsistent, the
        // step built from them is unusable, and the pass exits on its entry
        // point rather than climbing out. Differencing the real objective DOES
        // escape, because its Jacobian there is the penalty slope itself.
        //
        // Measured cost of this limit on 5,952 strict-HF rows across two
        // corpora: zero rows, because the seed is the output of a penalised MF
        // solve and effectively never starts inside. Written down so that the
        // day it does matter, it is already known.
        assert_eq!(
            modelled.dv.map(f64::to_bits),
            start.map(f64::to_bits),
            "known limit changed: the model arm now moves off a violating seed"
        );
        assert!(
            differenced.dv[0] >= floor && (differenced.dv[0] - optimum).abs() < 1.0e-3,
            "finite differences should escape the region and converge: {differenced:?}"
        );
    }

    /// From a FEASIBLE start, a Jacobian blind to the protected radius must not
    /// be able to walk the solver through it.
    ///
    /// This is the property the production `jacobian_model` depends on, and the
    /// reason it is allowed to be smooth: enforcement belongs to the objective,
    /// which returns the 1e6-km sentinel for any trial step that violates, so
    /// the cost rises and the step is rejected. The geometry is the hostile
    /// one -- the unconstrained optimum is INSIDE the wall, so the model points
    /// straight at it on every iteration.
    #[test]
    fn smooth_model_never_walks_through_a_penalty_it_cannot_see() {
        let floor = 0.10_f64;
        let optimum = floor - 0.05; // infeasible: the model always points here
        let start = [0.30, 0.20, 0.20]; // feasible
        let miss_of = |dv: [f64; 3]| [(dv[0] - optimum) * 1.0e3, dv[1] * 1.0e3, dv[2] * 1.0e3];
        let model = |dv: [f64; 3]| Some(miss_of(dv));
        let objective = |dv: [f64; 3]| -> [f64; 3] {
            if dv[0] < floor {
                let violation_penalty = 1.0e3 * (floor - dv[0]);
                let p = 1.0e6 + violation_penalty;
                return [p, p, p];
            }
            miss_of(dv)
        };
        let config = BoundedInterceptConfig {
            max_iters: 50,
            tol: 1.0e-5,
            bound: 0.5,
            max_bound: 0.5,
            min_miss_km: 1.0e4,
            skip_tol: 0.0,
            max_bound_expansions: 0,
            ..BoundedInterceptConfig::default()
        };

        let modelled =
            optimize_intercept_bounded_with_model(objective, Some(&model), start, &config)
                .expect("protected-radius model fixture must solve");

        assert!(
            modelled.dv[0] >= floor,
            "model-Jacobian solve walked through the protected radius: {modelled:?}"
        );
        assert!(
            modelled.miss_km.is_finite() && modelled.miss_km < 1.0e5,
            "model-Jacobian solve ended in the penalty region: {modelled:?}"
        );
    }

    /// `bits_equal3` must separate the two zeroes. They are equal under `==`
    /// and are distinct inputs to a propagator.
    #[test]
    fn bits_equal3_separates_signed_zero() {
        assert!(bits_equal3(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]));
        assert!(!bits_equal3(&[0.0, 1.0, 2.0], &[-0.0, 1.0, 2.0]));
        assert!(!bits_equal3(&[f64::NAN, 1.0, 2.0], &[1.0, 1.0, 2.0]));
    }
}
