//! Core ODE integration algorithms.
//!
//! This module delegates adaptive integration to the `odesolve` module's
//! Lightyear-compatible DOPRI5 loop, with Encke event handling preserved.

use num_traits::ToPrimitive;
use satpy_core::{eci2equinoc_impl_f64, equinoc2eci_impl, GravityError, PackedGravityCoeffs};
use std::borrow::Cow;
use std::sync::Arc;

use crate::odesolve::{
    integrate_final, integrate_final_esdirk, integrate_final_with_events,
    integrate_final_with_events_and_scratch, integrate_final_with_scratch,
    integrate_lightyear_dopri5, integrate_lightyear_dopri5_final,
    integrate_lightyear_dopri5_unforced, integrate_sampled, integrate_sampled_esdirk,
    integrate_sampled_esdirk_into, integrate_sampled_into, integrate_sampled_unforced,
    integrate_sampled_with_events, integrate_sampled_with_events_esdirk, ErrorControl,
    EventDecision as OdeEventDecision, EventHandler as OdeEventHandler,
    IntegrationResult as OdeIntegrationResult, IntegrationResultSampled,
    IntegrationStatus as OdeIntegrationStatus, IntegratorConfig, LightyearConfig,
    Method as OdeMethod, OdeSystem as OdeSystemTrait, SolverScratch,
};

use crate::eclipse::EclipseError;
#[cfg(any(feature = "scalar-leg-observer", test))]
use crate::eclipse::EclipseSide;
use crate::eclipse_coordinator::{
    eclipse_error_name, integrate_binary_eclipse_scalar, integrate_binary_eclipse_scalar_with_rhs,
    BinaryEclipseContext, BinaryEclipseRun,
};
#[cfg(test)]
use crate::eclipse_coordinator::{
    eclipse_test_state_guard, TEST_ECLIPSE_ROOTS, TEST_ECLIPSE_SPLITS,
    TEST_HIDDEN_DOUBLE_ACCEPTED_STEPS, TEST_ROOT_TRANSACTION_CONTINUATIONS,
    TEST_ROOT_TRANSACTION_RESETS,
};
#[cfg(feature = "scalar-leg-observer")]
use crate::eclipse_coordinator::{
    integrate_binary_eclipse_scalar_observed, integrate_binary_eclipse_scalar_with_rhs_observed,
};
use crate::events::{check_event_crossing, evaluate_all_events, EventState};
use crate::probe::PropagationCensusError;
use crate::rhs::{effective_scalar_srp, BaselineCalculator, LightyearRHS};
use crate::types::StepperMethod;

use crate::types::{ForceConfig, IntegrationResult, OdeMetrics, PERTURB_DEVIATION_THRESHOLD_KM};

#[cfg(feature = "scalar-leg-observer")]
use crate::odesolve::IntegrationStats;

/// Immutable spherical-gravity inputs shared by scalar propagation requests.
#[derive(Clone)]
pub struct ScalarGravityAssets {
    pub(crate) packed: Arc<PackedGravityCoeffs>,
}

impl ScalarGravityAssets {
    #[must_use]
    pub const fn new(packed: Arc<PackedGravityCoeffs>) -> Self {
        Self { packed }
    }
}

/// Immutable authority and assets for one scalar propagation family.
///
/// Construct once per force configuration, then borrow it for all sampled or
/// final requests. It owns only immutable `Arc` assets; no RHS cache or Encke
/// state lives here.
#[derive(Clone)]
pub struct ScalarPropagationContext {
    pub(crate) jd0: f64,
    pub(crate) config: Arc<ForceConfig>,
    pub(crate) gravity: ScalarGravityAssets,
}

impl ScalarPropagationContext {
    #[must_use]
    pub const fn new(jd0: f64, config: Arc<ForceConfig>, gravity: ScalarGravityAssets) -> Self {
        Self {
            jd0,
            config,
            gravity,
        }
    }

    pub(crate) fn new_rhs(
        &self,
        init_equinoc_state: [f64; 6],
        t0_s: f64,
    ) -> anyhow::Result<LightyearRHS> {
        LightyearRHS::try_new(
            init_equinoc_state,
            t0_s,
            self.jd0,
            Arc::clone(&self.config),
            Arc::clone(&self.gravity.packed),
        )
    }

    fn binary_eclipse_context(&self) -> BinaryEclipseContext {
        BinaryEclipseContext {
            eps: self.config.eps,
            jd0: self.jd0,
            config: Arc::clone(&self.config),
            packed: Arc::clone(&self.gravity.packed),
            stepper: self.config.integrator_method,
        }
    }
}

/// Sampling contract for a scalar propagation request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampledOutputMode {
    /// Reconstruct requested samples from accepted solver steps.
    Interpolated,
    /// Force solver steps at every requested sample time.
    ForceEvaluationTimes,
}

/// One scalar propagation request bound to immutable force authority.
///
/// It carries no caches. Each execution creates or resets an RHS explicitly,
/// preserving the history-dependent Encke baseline semantics.
#[derive(Clone, Copy)]
pub struct ScalarPropagationRequest<'a> {
    context: &'a ScalarPropagationContext,
    init_equinoc_state: [f64; 6],
    t_eval: &'a [f64],
    t0_s: f64,
    t_final_s: f64,
    enable_events: bool,
    stepper: StepperMethod,
    output_mode: SampledOutputMode,
}

impl<'a> ScalarPropagationRequest<'a> {
    #[must_use]
    pub fn new(
        context: &'a ScalarPropagationContext,
        init_equinoc_state: [f64; 6],
        t_eval: &'a [f64],
        t0_s: f64,
        t_final_s: f64,
    ) -> Self {
        Self {
            context,
            init_equinoc_state,
            t_eval,
            t0_s,
            t_final_s,
            enable_events: false,
            stepper: context.config.integrator_method,
            output_mode: SampledOutputMode::Interpolated,
        }
    }

    #[must_use]
    pub const fn with_events(mut self, enable_events: bool) -> Self {
        self.enable_events = enable_events;
        self
    }

    #[must_use]
    const fn with_resolved_stepper(mut self, stepper: StepperMethod) -> Self {
        self.stepper = stepper;
        self
    }

    #[must_use]
    pub const fn with_output_mode(mut self, output_mode: SampledOutputMode) -> Self {
        self.output_mode = output_mode;
        self
    }

    #[must_use]
    const fn force_eval(self) -> bool {
        matches!(self.output_mode, SampledOutputMode::ForceEvaluationTimes)
    }
}

/// Maximum number of integration steps allowed per satellite to prevent runaway memory growth.
// Sampled HF catalogue/state-table paths can exceed 50k accepted steps over
// multi-day spans at strict tolerances; use a higher cap to avoid false
// max-step termination on otherwise valid trajectories.
pub(crate) const MAX_STEPS: usize = 500_000;

/// The rectification segment cap: one LEO orbital period.
///
/// **This is the single definition.** It was previously restated in
/// `eclipse_coordinator.rs`, in `two_phase_transfer_rs::hf_acceptance`, and
/// again as a local `const` inside the HF replay test -- four private copies
/// of one number, all bit-identical but none binding the others. The replay
/// test's own comment asserted it was "the same 5400 s cap the integrator
/// applies internally", which nothing enforced: had the integrator's cap
/// moved, the replay would have walked different chunks and compared two
/// different computations while staying green.
///
/// # Walked upward and REFUTED (R46, 2026-08-10), and the null prices a whole
/// literature axis
///
/// Every regularized or element-space formulation in the astrodynamics
/// literature -- equinoctial VOP, Dromo/EDromo, Kustaanheimo-Stiefel,
/// Sperling-Burdet, Stiefel-Scheifele, stabilized Cowell -- has exactly one
/// structural advantage over the Encke this tree already flies: it carries no
/// reference to rectify, so it never pays a rectification restart. Sending this
/// constant to infinity IS that arm, at zero implementation cost. On the
/// 12-draw census (`r43_leg_census`, vern7, eps 1e-8, `dt_max` 300, atm 7, era
/// a4b3791):
///
/// ```text
/// MAX_RECT_SEGMENT   RHS evals        d      segments   steps   rejected
///        1_350 s        93,625   +23.74%          886   9,273          1
///        5_400 s        75,661     ----           796   7,482          5
///       21_600 s        74,821    -1.11%          803   7,148        282
///          1e9 s        74,819    -1.11%          799   7,150        280
/// ```
///
/// **The ceiling on the entire formulation axis is 1.11% of RHS evaluations,**
/// before any such formulation pays its own per-evaluation conversion cost. The
/// `1_350` s rung is the non-vacuity control and it is emphatic, so the null is
/// not the signature of an unwired knob.
///
/// Two mechanisms explain it, and both are visible in the table. First, the
/// segment count does not fall (796 -> 799 at infinity): this cap is already
/// nearly inert in production, because eclipse root transactions end segments
/// long before `5_400` s elapses, and no formulation removes those -- the SRP
/// force is discontinuous across a root, so every method must restart there.
/// Second, what the longer segments do buy is bought back: rejections go
/// 5 -> 280, a 56x rise, as the Encke deviation grows and the controller starts
/// fighting. That is the same rejection-straggler mechanism that refuted h0
/// widening in R43.
///
/// The deviation-threshold half of rectification was walked separately and
/// reads under 3% on this tree, non-monotone -- see the ladder recorded at
/// `PERTURB_DEVIATION_THRESHOLD_KM` in `types.rs`.
pub const MAX_RECTIFICATION_SEGMENT_S: f64 = 5_400.0;

/// Validate the scalar solver encoded in immutable force authority.
///
/// Part A must never accept a second method through an adapter argument.
pub(crate) fn validate_scalar_stepper_authority(
    config: &ForceConfig,
    context: &str,
) -> anyhow::Result<()> {
    let stepper = config.integrator_method;
    let guarded_high_fidelity = crate::rhs::atm_model_uses_jb2008_drivers(config.atm_model);
    let binary_eclipse = effective_scalar_srp(config);
    if (guarded_high_fidelity || binary_eclipse)
        && matches!(stepper, StepperMethod::Esdirk43 | StepperMethod::Auto)
    {
        let model = match config.atm_model {
            4 => "JB2008 model4",
            5 => "JB2008 model5 candidate approximation",
            6 => "JB2008 model6 production approximation",
            _ => "binary-eclipse SRP",
        };
        return Err(anyhow::anyhow!(
            "{model} requires explicit {context} method; ESDIRK/dual/STM and auto are unsupported"
        ));
    }
    Ok(())
}

#[inline]
/// Resolve Auto to a concrete method based on eps.
/// vern9 for eps >= 2e-8 (best accuracy), dopri5 below (avoids vern9 cliff).
fn resolve_auto_stepper(stepper: StepperMethod, eps: f64) -> StepperMethod {
    match stepper {
        StepperMethod::Auto => {
            if eps >= 2e-8 {
                StepperMethod::Vern9
            } else {
                StepperMethod::Dopri5Compat
            }
        }
        other => other,
    }
}

const fn stepper_ode_method(stepper: StepperMethod) -> OdeMethod {
    match stepper {
        // ESDIRK doesn't map to explicit OdeMethod; callers must handle it.
        StepperMethod::Tsit5 | StepperMethod::Esdirk43 => OdeMethod::Tsit5,
        StepperMethod::Dop853 => OdeMethod::Dop853,
        StepperMethod::Rkv98 => OdeMethod::Rkv98,
        StepperMethod::Dopri5Compat => OdeMethod::Dopri5,
        StepperMethod::Vern7 => OdeMethod::Vern7,
        // Auto should be resolved by `resolve_auto_stepper` before reaching
        // here; this mirrors its eps >= 2e-8 branch as a fallback.
        StepperMethod::Vern9 | StepperMethod::Auto => OdeMethod::Vern9,
    }
}

/// What ended the PREVIOUS solver entry, from this one's point of view.
///
/// The distinction is about the Encke baseline, not about the clock. A solver
/// entry restarts the step-size controller from a cold guess, and whether that
/// restart is avoidable depends on whether the error history it discarded is
/// still valid — which is a question about the baseline the deviation is
/// measured against, not about how long the entry ran.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SegmentBoundary {
    /// First entry of a propagation. Nothing precedes it.
    ArcStart,
    /// The baseline moved: `reset_for_propagation` ran between the previous
    /// entry and this one, so the deviation this entry integrates is measured
    /// against different elements and the previous entry's error history says
    /// nothing about it. The controller's exit `h` DOES still describe the
    /// trajectory, though, and is carried across this boundary as the next
    /// entry's opening guess (`hcarry_take`) — the class-keyed carry measured
    /// −3.6/−4.0% of cell evaluations. What was refuted at `c546130` was
    /// raising the cold `h0` GUESS uniformly (10.77 m endpoint breach, two
    /// non-finite masses), and carrying `err_prev` with the step was measured
    /// nearly worthless (−0.53% vs −3.75%, R46): the error HISTORY is invalid
    /// here, the exit step SIZE is not.
    Rebased,
    /// Same baseline, same deviation, same elements: the previous entry stopped
    /// only because it had to land on an intermediate time. The trajectory is
    /// continuous across this boundary in every quantity the controller reads.
    EventContinuation,
}

/// Controls for one scalar solver segment.
///
/// These values travel together because changing any of them changes accepted
/// steps, event roots, or the Encke baseline reached by the next segment.
#[derive(Clone, Copy)]
pub(crate) struct SegmentControls {
    pub(crate) t0_s: f64,
    pub(crate) t_final_s: f64,
    pub(crate) eps: f64,
    pub(crate) dt_max: f64,
    pub(crate) force_eval: bool,
    pub(crate) fast_single: bool,
    pub(crate) max_steps: usize,
    pub(crate) max_rejects: usize,
    pub(crate) stepper: StepperMethod,
    pub(crate) boundary: SegmentBoundary,
}

/// Step-size carry: open a `Rebased` unclamped leg at the controller `h` its
/// predecessor on the same arc exited with, instead of the cold
/// `default_h0 = span/100` guess.
///
/// A `Rebased` boundary is an Encke baseline refresh, not a dynamics
/// discontinuity — the trajectory is continuous there in every quantity the
/// controller reads, so the predecessor's exit `h` is a better opening guess
/// than a span heuristic. The carry is a GUESS, never an answer: the solver
/// clamps it to `[h_min, h_max]` and the error controller re-adapts from the
/// first step.
///
/// Three rules, each measured (`docs/evidence/hcarry-20260813/hcarry-rebase.md`):
///
/// - **Unclamped legs only** (`dt_max > HCARRY_UNCLAMPED_DT_MAX_S`). Clamped
///   eclipse root-refinement legs neither store nor consume: they open at
///   their 10 s cap already, and leaking a 10 s replay `h` into a 300 s Encke
///   segment was measured at +4.33% (era `fc222c0`).
/// - **`Rebased` boundaries only.** `EventContinuation` window legs take one
///   span-bounded step by construction, and `ArcStart` has no predecessor.
/// - **Reset at every propagation entry** (`hcarry_reset`), so the carry can
///   never cross from one arc into another. This is load-bearing for
///   determinism, not just quality: propagations are scheduled onto threads by
///   rayon, and a cross-arc carry would make output bits depend on which arc
///   last ran on the thread — the leak that kept the R46 probe unlandable.
///
/// Measured at tip (p4g1x8 / p24g1x8 hybrid census cells, deterministic
/// counts): −3.62% / −4.03% of cell RHS evaluations; objective movement 2–3
/// orders below the `c546130` mass-reproducibility floor. Bit-moving by
/// design — bit pins are re-sealed with the change.
const HCARRY_UNCLAMPED_DT_MAX_S: f64 = 60.0;

thread_local! {
    static HCARRY_H: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// Clear the carried step size. MUST be called at every propagation entry
/// whose legs reach [`integrate_segment_with_method`] with a `Rebased`
/// boundary; see the carry rules above for why this is load-bearing.
pub(crate) fn hcarry_reset() {
    HCARRY_H.with(|slot| slot.set(0.0));
}

/// The carried opening step for this leg: an unclamped `Rebased` continuation
/// with a stored predecessor, `None` otherwise.
fn hcarry_take(controls: &SegmentControls) -> Option<f64> {
    if controls.boundary != SegmentBoundary::Rebased || controls.dt_max <= HCARRY_UNCLAMPED_DT_MAX_S
    {
        return None;
    }
    let h = HCARRY_H.with(std::cell::Cell::get);
    (h.is_finite() && h > 0.0).then_some(h)
}

/// Store an unclamped leg's exit `h` for the next `Rebased` leg on this arc.
fn hcarry_store(controls: &SegmentControls, final_controller_h: f64) {
    if controls.dt_max > HCARRY_UNCLAMPED_DT_MAX_S
        && final_controller_h.is_finite()
        && final_controller_h > 0.0
    {
        HCARRY_H.with(|slot| slot.set(final_controller_h));
    }
}

pub(crate) fn integrate_segment_with_method(
    system: &LightyearSystem<'_>,
    y0: &[f64],
    t_eval: &[f64],
    controls: SegmentControls,
    handler: Option<&mut dyn OdeEventHandler>,
) -> Result<IntegrationResultSampled, GravityError> {
    debug_assert!(
        !matches!(
            controls.stepper,
            StepperMethod::Esdirk43 | StepperMethod::Auto
        ),
        "ESDIRK/Auto must be intercepted before integrate_segment_with_method"
    );
    system.rhs.clear_gravity_error();
    crate::probe::bump_segment();
    let eps_eff = controls.eps.max(1e-12);
    let probe_out = if controls.stepper == StepperMethod::Dopri5Compat {
        // DOPRI5 absolute-error mode becomes pathologically slow/unstable when eps
        // is set below machine-meaningful scales for this problem family.
        // Clamp to a conservative floor to prevent max-step exhaustion with
        // near-zero progress (e.g. eps=1e-14).
        // Unforced sampling reconstructs requested output only. Hermite
        // samples never become a committed eclipse root state.
        let ly_config = LightyearConfig {
            eps: eps_eff,
            dt_max: controls.dt_max,
            max_steps: controls.max_steps,
            max_rejects: controls.max_rejects,
            force_eval: controls.force_eval,
            fast_single: controls.fast_single,
        };
        Ok(if controls.force_eval {
            handler.map_or_else(
                || {
                    integrate_lightyear_dopri5(
                        system,
                        y0,
                        t_eval,
                        controls.t0_s,
                        controls.t_final_s,
                        ly_config,
                        None,
                    )
                },
                |handler| {
                    integrate_lightyear_dopri5(
                        system,
                        y0,
                        t_eval,
                        controls.t0_s,
                        controls.t_final_s,
                        ly_config,
                        Some(handler),
                    )
                },
            )
        } else {
            handler.map_or_else(
                || {
                    integrate_lightyear_dopri5_unforced(
                        system,
                        y0,
                        t_eval,
                        controls.t0_s,
                        controls.t_final_s,
                        ly_config,
                        None,
                    )
                },
                |handler| {
                    integrate_lightyear_dopri5_unforced(
                        system,
                        y0,
                        t_eval,
                        controls.t0_s,
                        controls.t_final_s,
                        ly_config,
                        Some(handler),
                    )
                },
            )
        })
    } else {
        let cfg = IntegratorConfig {
            error_control: ErrorControl::Absolute { eps: eps_eff },
            h0: hcarry_take(&controls),
            h_min: controls.dt_max.abs().clamp(f64::MIN_POSITIVE, 1e-12),
            h_max: controls.dt_max,
            max_steps: controls.max_steps,
            max_rejects: controls.max_rejects,
            force_eval: controls.force_eval,
        };
        let method = stepper_ode_method(controls.stepper);
        if controls.force_eval {
            handler.map_or_else(
                || integrate_sampled_into_or_allocating(system, method, y0, t_eval, cfg),
                |handler| {
                    Ok(integrate_sampled_with_events(
                        system, method, y0, t_eval, cfg, handler,
                    ))
                },
            )
        } else {
            handler.map_or_else(
                || {
                    Ok(integrate_sampled_unforced(
                        system, method, y0, t_eval, cfg, None,
                    ))
                },
                |handler| {
                    Ok(integrate_sampled_unforced(
                        system,
                        method,
                        y0,
                        t_eval,
                        cfg,
                        Some(handler),
                    ))
                },
            )
        }
    };
    let probe_out = probe_out?;
    hcarry_store(&controls, probe_out.stats.final_controller_h);
    let gravity_error = system.rhs.take_gravity_error();
    crate::probe::bump_steps(probe_out.stats.steps);
    crate::probe::bump_saturated(probe_out.stats.saturated_steps);
    crate::probe::bump_rejected(probe_out.stats.rejected_steps);
    crate::probe::observe_min_h(probe_out.stats.min_accepted_h);
    crate::probe::observe_ramp(
        controls.boundary,
        probe_out.stats.segment_span_s,
        &probe_out.stats.first_accepted_h,
        probe_out.stats.tail_h_sum,
        probe_out.stats.tail_h_count,
    );
    // This is the only one of the three solver entry points the eclipse
    // coordinator's `MAX_ROOT_REFINEMENT_STEP_S` clamp can reach, so it is the
    // only one that can file a clamped leg. The other two pass production
    // `dt_max` and are unclamped by construction.
    crate::probe::observe_leg(
        controls.dt_max,
        probe_out.stats.segment_span_s,
        probe_out.stats.steps,
        probe_out.stats.evals,
        probe_out.stats.rejected_steps,
    );
    crate::probe::observe_cache_cluster(
        probe_out.stats.cache_cluster_steps,
        probe_out.stats.cache_cluster_steps_untruncated,
    );
    crate::probe::observe_underflow(probe_out.stats.underflow_accepts);
    gravity_error.map_or(Ok(probe_out), Err)
}

fn sampled_from_into_result(
    t_eval: &[f64],
    n_state: usize,
    states: Vec<f64>,
    result: OdeIntegrationResult,
) -> IntegrationResultSampled {
    IntegrationResultSampled {
        times: t_eval.to_vec(),
        states,
        n_state,
        status: result.status,
        stats: result.stats,
        event: result.event,
    }
}

fn integrate_sampled_into_or_allocating(
    system: &LightyearSystem<'_>,
    method: OdeMethod,
    y0: &[f64],
    t_eval: &[f64],
    cfg: IntegratorConfig,
) -> Result<IntegrationResultSampled, GravityError> {
    let n_state = y0.len();
    let mut states = vec![0.0; t_eval.len().saturating_mul(n_state)];
    let first_result = integrate_sampled_into(system, method, y0, t_eval, cfg, &mut states);
    if let Some(error) = system.rhs.take_gravity_error() {
        return Err(error);
    }
    if matches!(first_result.status, OdeIntegrationStatus::Success) {
        Ok(sampled_from_into_result(
            t_eval,
            n_state,
            states,
            first_result,
        ))
    } else {
        system.rhs.clear_gravity_error();
        let fallback = integrate_sampled(system, method, y0, t_eval, cfg);
        system.rhs.take_gravity_error().map_or(Ok(fallback), Err)
    }
}

// ============================================================================
// Standardized ODE System + Event Handler (Encke)
// ============================================================================

pub(crate) struct LightyearSystem<'a> {
    pub(crate) rhs: &'a LightyearRHS,
}

#[cfg(feature = "autodiff")]
#[inline]
fn take_esdirk_gravity_error(
    scalar_rhs: &LightyearRHS,
    dual_rhs: &crate::rhs_dual::LightyearDualRHS,
) -> Option<GravityError> {
    let scalar_error = scalar_rhs.take_gravity_error();
    let dual_error = dual_rhs.take_gravity_error();
    scalar_error.or(dual_error)
}

impl OdeSystemTrait for LightyearSystem<'_> {
    fn rhs(&self, t: f64, y: &[f64], dy: &mut [f64]) {
        let Some(delta) = y
            .get(..6)
            .and_then(|values| <&[f64; 6]>::try_from(values).ok())
        else {
            dy.fill(f64::NAN);
            return;
        };
        let Some(dy_out) = dy
            .get_mut(..6)
            .and_then(|values| <&mut [f64; 6]>::try_from(values).ok())
        else {
            dy.fill(f64::NAN);
            return;
        };
        crate::probe::tag_add(&crate::probe::RHS_EVALS, crate::probe::current_tag());
        if self.rhs.eclipse_envelope_is_active() {
            if let Err(error) = self.rhs.validate_eclipse_envelope_at_delta(delta, t) {
                self.rhs.record_eclipse_error(error);
                *dy_out = [f64::NAN; 6];
                return;
            }
        }
        // The `RHS_EVALS` bump above is unconditional; `strict_hf_pin` asserts
        // on the count.
        *dy_out = self
            .rhs
            .compute_internal(delta, t)
            .map_or([f64::NAN; 6], |derivative| derivative);
    }

    /// Hands the prefill only the nodes whose baselines can actually be read.
    ///
    /// `prefill_stage_baselines` packs its input four at a time, so the pack
    /// count is `ceil(n / 4)` and only the node COUNT decides it. Vern7's ten
    /// nodes cost three packs — 4, 4, and a two-lane tail that pays a full x4
    /// solve for two useful values. Two of those ten earn nothing:
    ///
    /// * `c[0] == 0.0`. Its baseline is read only when `k[0]` is computed
    ///   rather than recycled from the previous step's endpoint derivative,
    ///   which mid-segment is never. Dropping it costs one scalar
    ///   `state_at_seeded` on the steps that do read it — segment starts — and
    ///   `baseline_exact_memo` absorbs any repeat.
    /// * `c[9] == c[8] == 1.0`. The table is keyed on the exact bits of
    ///   `(t + node * h) - t0_s`, so both nodes produce the SAME key and the
    ///   second entry is a duplicate the lookup can never reach. Dropping it
    ///   costs nothing at all: `c[9]`'s lookup lands on `c[8]`'s entry.
    ///
    /// Eight nodes pack into two, so the step spends 2 x4 solves where it spent
    /// 3. Filtered HERE rather than in `prefill_stage_baselines` because the
    /// premise is about this integrator's `k[0]` recycling and this tableau's
    /// repeated endpoint, not about the prefill, which is shared.
    ///
    /// The filter is on RAW BITS, deliberately. What it is really deduplicating
    /// is the table key, and the key is a deterministic function of the node,
    /// so equal bits in means equal key out — whereas `==` would also fold
    /// `-0.0` into `0.0` and drop a node whose key is genuinely distinct.
    fn prefill_stage_times(&self, t: f64, h: f64, nodes: &[f64]) {
        const ZERO_BITS: u64 = 0.0_f64.to_bits();
        let mut live = [0.0_f64; MAX_PREFILL_NODES];
        let mut count = 0_usize;
        let mut previous_bits: Option<u64> = None;
        for &node in nodes {
            let bits = node.to_bits();
            if bits == ZERO_BITS || previous_bits == Some(bits) {
                continue;
            }
            let Some(slot) = live.get_mut(count) else {
                // More live nodes than this buffer holds: hand the prefill the
                // original list rather than a truncated one, which would leave
                // real stage times unfilled and silently slower.
                self.rhs.prefill_stage_baselines(t, h, nodes);
                return;
            };
            *slot = node;
            previous_bits = Some(bits);
            count = count.saturating_add(1);
        }
        self.rhs
            .prefill_stage_baselines(t, h, live.get(..count).unwrap_or(nodes));
    }
}

/// Capacity of the filtered stage-node buffer in `prefill_stage_times`.
///
/// The prefill's own table holds 16, and no compiled tableau has more stages
/// than that; a tableau that did would take the unfiltered fallback rather than
/// a truncated list.
const MAX_PREFILL_NODES: usize = 16;

pub(crate) struct EnckeEventHandler<'a> {
    baseline_calc: BaselineCalculator<'a>,
    event_state: EventState,
    t0_s: f64,
    detection: Option<crate::types::EventDetection>,
    rhs: &'a LightyearRHS,
    event_interp_tol: f64,
    eps: f64,
    max_rejects: usize,
    event_invalid: bool,
}

impl<'a> EnckeEventHandler<'a> {
    pub(crate) fn new(
        baseline_calc: BaselineCalculator<'a>,
        t0_s: f64,
        init_state: [f64; 6],
        rhs: &'a LightyearRHS,
        event_interp_tol: f64,
        eps: f64,
        max_rejects: usize,
    ) -> Self {
        let mut es = EventState::default();
        let r_base = baseline_calc.get_baseline_state(t0_s);
        es.prev_values = evaluate_all_events(&init_state, &r_base);
        es.prev_time = t0_s;
        es.initialized = true;
        Self {
            baseline_calc,
            event_state: es,
            t0_s,
            detection: None,
            rhs,
            event_interp_tol,
            eps,
            max_rejects,
            event_invalid: false,
        }
    }

    pub(crate) const fn detection(&self) -> Option<crate::types::EventDetection> {
        self.detection
    }

    pub(crate) const fn take_detection(&mut self) -> Option<crate::types::EventDetection> {
        self.detection.take()
    }

    pub(crate) const fn take_event_invalid(&mut self) -> bool {
        let v = self.event_invalid;
        self.event_invalid = false;
        v
    }
}

impl OdeEventHandler for EnckeEventHandler<'_> {
    fn on_step(
        &mut self,
        prev_t: f64,
        prev_y: &[f64],
        previous_derivative: &[f64],
        next_t: f64,
        next_y: &[f64],
        next_derivative: &[f64],
    ) -> OdeEventDecision {
        let prev_state = slice_to_state(prev_y);
        let next_state = slice_to_state(next_y);
        let prev_dy_arr = slice_to_state(previous_derivative);
        let next_dy_arr = slice_to_state(next_derivative);

        let r_base_next = self.baseline_calc.get_baseline_state(next_t);
        let curr_values = evaluate_all_events(&next_state, &r_base_next);

        let mut baseline_fn = |time: f64| self.baseline_calc.get_baseline_state(time);
        let detection = check_event_crossing(
            &prev_state,
            &next_state,
            &prev_dy_arr,
            &next_dy_arr,
            prev_t,
            next_t,
            &self.event_state.prev_values,
            &curr_values,
            self.t0_s,
            &mut baseline_fn,
        );

        if detection.detected {
            let mut detection = detection;
            let (t_event, clamped) = clamp_event_time(prev_t, next_t, detection.refined_time);
            let h = next_t - prev_t;
            let mut tau = if h.abs() > 0.0 {
                (t_event - prev_t) / h
            } else {
                0.0
            };
            tau = tau.clamp(0.0, 1.0);

            let hermite = crate::types::hermite_interp(
                &prev_state,
                &next_state,
                &prev_dy_arr,
                &next_dy_arr,
                h,
                tau,
            );
            let linear = linear_interp(&prev_state, &next_state, tau);
            let hermite_ok = all_finite_state(&hermite);
            let linear_ok = all_finite_state(&linear);
            let mut interp_error = if hermite_ok && linear_ok {
                interp_error(&hermite, &linear)
            } else {
                f64::NAN
            };
            let method: &str;
            let state_at_event;
            let mut event_invalid = false;
            let dt_event = t_event - prev_t;

            if hermite_ok && linear_ok && interp_error <= self.event_interp_tol {
                method = if clamped { "hermite_clamp" } else { "hermite" };
                state_at_event = hermite;
            } else {
                let mut micro_state: Option<[f64; 6]> = None;
                if dt_event.is_finite() && dt_event.abs() > 0.0 {
                    let micro_eps = (self.eps * 0.1).max(1e-12);
                    let micro_cfg = LightyearConfig {
                        eps: micro_eps,
                        dt_max: dt_event.abs().max(1e-12),
                        max_steps: 20,
                        max_rejects: self.max_rejects,
                        force_eval: false,
                        fast_single: true,
                    };
                    let system = LightyearSystem { rhs: self.rhs };
                    let micro = integrate_lightyear_dopri5_final(
                        &system,
                        &prev_state,
                        prev_t,
                        t_event,
                        micro_cfg,
                        None,
                    );
                    if let (OdeIntegrationStatus::Success, Some(values)) = (
                        micro.status,
                        micro
                            .y
                            .get(..6)
                            .and_then(|slice| <&[f64; 6]>::try_from(slice).ok()),
                    ) {
                        let mut arr = [0.0; 6];
                        arr.copy_from_slice(values);
                        if all_finite_state(&arr) {
                            micro_state = Some(arr);
                        }
                    }
                }

                if let Some(ms) = micro_state {
                    method = "micro_step";
                    interp_error = 0.0;
                    state_at_event = ms;
                } else if linear_ok {
                    method = if clamped { "linear_clamp" } else { "linear" };
                    state_at_event = linear;
                } else {
                    method = "endpoint";
                    state_at_event = next_state;
                    interp_error = f64::NAN;
                    event_invalid = true;
                }
            }

            if detection.event_type == crate::types::EventType::PerturbDeviation
                && dt_event.abs() < 1e-9
            {
                event_invalid = true;
            }

            detection.refined_time = t_event;
            detection.state_at_event = state_at_event;
            detection.interp_method = crate::types::InterpMethod::from_str(method);
            detection.interp_error = interp_error;

            if event_invalid {
                self.event_invalid = true;
            }

            let t_event = detection.refined_time;
            let y_event = detection.state_at_event;
            self.detection = Some(detection);
            return OdeEventDecision::Stop {
                t_event,
                y_event: y_event.to_vec(),
            };
        }

        self.event_state.prev_values = curr_values;
        self.event_state.prev_time = next_t;
        OdeEventDecision::Continue
    }
}

#[inline]
pub(crate) fn slice_to_state(slice: &[f64]) -> [f64; 6] {
    slice
        .get(..6)
        .and_then(|values| <&[f64; 6]>::try_from(values).ok())
        .copied()
        .unwrap_or([f64::NAN; 6])
}

#[inline]
fn all_finite_state(state: &[f64; 6]) -> bool {
    state.iter().all(|v| v.is_finite())
}

#[inline]
fn position_norm_sq(state: &[f64; 6]) -> f64 {
    let [x, y, z, ..] = *state;
    x.mul_add(x, y.mul_add(y, z * z))
}

#[inline]
#[expect(
    clippy::float_cmp,
    reason = "Exact segment endpoint equality controls a discrete rebase boundary; tolerance changes trajectories."
)]
fn needs_rectification(
    delta_sq: f64,
    threshold_sq: f64,
    segment_t_final_s: f64,
    arc_t_final_s: f64,
) -> bool {
    delta_sq > threshold_sq || segment_t_final_s != arc_t_final_s
}

#[inline]
fn bounded_segment_end(current_t_s: f64, arc_t_final_s: f64, max_segment_s: f64) -> f64 {
    let remaining = arc_t_final_s - current_t_s;
    if remaining.abs() > max_segment_s {
        current_t_s + max_segment_s * remaining.signum()
    } else {
        arc_t_final_s
    }
}

#[inline]
fn rebase_equinoc_from_delta(
    init_equinoc_state: &[f64; 6],
    init_t_s: f64,
    segment_t_final_s: f64,
    delta: &[f64; 6],
) -> [f64; 6] {
    let mut baseline_eci = [0.0; 6];
    equinoc2eci_impl(
        init_equinoc_state,
        6,
        segment_t_final_s - init_t_s,
        0.0,
        &mut baseline_eci,
    );
    let eci = add_state_vectors(&baseline_eci, delta);
    let mut new_equinoc = [0.0; 6];
    eci2equinoc_impl_f64(&eci, 6, 0.0, 0.0, &mut new_equinoc);
    new_equinoc
}

fn append_visible_sampled_states(
    all_times: &mut Vec<f64>,
    all_states: &mut Vec<[f64; 6]>,
    segment_times: &[f64],
    segment_states: &[[f64; 6]],
    visible_start: usize,
    visible_end: usize,
    curr_init_equinoc: &[f64; 6],
    curr_t0_s: f64,
    orig_init_equinoc: &[f64; 6],
    orig_t0_s: f64,
    had_rebase: bool,
) {
    for (&time, state) in segment_times
        .iter()
        .zip(segment_states)
        .skip(visible_start)
        .take(visible_end.saturating_sub(visible_start))
    {
        all_times.push(time);
        all_states.push(if had_rebase {
            correct_delta_to_original_baseline(
                state,
                time,
                curr_init_equinoc,
                curr_t0_s,
                orig_init_equinoc,
                orig_t0_s,
            )
        } else {
            *state
        });
    }
}

fn elapsed_micros_u64(start_time: std::time::Instant) -> u64 {
    u64::try_from(start_time.elapsed().as_micros()).unwrap_or(u64::MAX)
}

#[inline]
fn add_state_vectors(left: &[f64; 6], right: &[f64; 6]) -> [f64; 6] {
    let mut sum = [0.0; 6];
    for ((output, &left_value), &right_value) in sum.iter_mut().zip(left).zip(right) {
        *output = left_value + right_value;
    }
    sum
}

#[inline]
fn subtract_state_vectors(left: &[f64; 6], right: &[f64; 6]) -> [f64; 6] {
    let mut difference = [0.0; 6];
    for ((output, &left_value), &right_value) in difference.iter_mut().zip(left).zip(right) {
        *output = left_value - right_value;
    }
    difference
}

#[inline]
fn linear_interp(prev_state: &[f64; 6], next_state: &[f64; 6], tau: f64) -> [f64; 6] {
    let mut out = [0.0; 6];
    for ((out, &prev), &next) in out.iter_mut().zip(prev_state).zip(next_state) {
        *out = prev + tau * (next - prev);
    }
    out
}

#[inline]
fn interp_error(hermite: &[f64; 6], linear: &[f64; 6]) -> f64 {
    let mut deltas = hermite
        .iter()
        .zip(linear)
        .map(|(&left, &right)| left - right);
    let err_pos = deltas
        .by_ref()
        .take(3)
        .fold(0.0, |acc, delta| acc + delta * delta);
    let err_vel = deltas.fold(0.0, |acc, delta| acc + delta * delta);
    // Same monotone-`sqrt` fold as the eclipse scan's motion bound: the larger
    // root is the root of the larger sum, so one `sqrt` returns the bits two did.
    err_pos.max(err_vel).sqrt()
}

#[inline]
fn clamp_event_time(prev_t: f64, next_t: f64, t_event: f64) -> (f64, bool) {
    let (t_min, t_max) = if prev_t <= next_t {
        (prev_t, next_t)
    } else {
        (next_t, prev_t)
    };
    let mut t = if t_event.is_finite() {
        t_event
    } else {
        0.5 * (prev_t + next_t)
    };
    let mut clamped = !t_event.is_finite();
    if t < t_min {
        t = t_min;
        clamped = true;
    } else if t > t_max {
        t = t_max;
        clamped = true;
    }
    (t, clamped)
}

#[inline]
pub(crate) fn flatten_states(states: Vec<f64>, n_state: usize) -> Vec<[f64; 6]> {
    if states.is_empty() {
        return Vec::new();
    }

    if n_state == 6 && states.len().is_multiple_of(6) {
        return f64_vec_into_state6_vec(states);
    }

    states
        .chunks_exact(6)
        .map(|chunk| {
            let mut out = [0.0; 6];
            out.copy_from_slice(chunk);
            out
        })
        .collect()
}

#[inline]
fn f64_vec_into_state6_vec(states: Vec<f64>) -> Vec<[f64; 6]> {
    let len = states.len();
    // HARD assert, not `debug_assert`. This is the divisibility precondition of
    // the `Box::from_raw` below, and the campaign runs release, where a
    // `debug_assert` is compiled out. A release caller with `len % 6 != 0`
    // would round `n_state` down and hand the deallocator a size smaller than
    // the allocation: heap corruption, with no diagnostic. One predictable
    // branch per call is the price of that not being possible.
    assert_eq!(
        len % 6,
        0,
        "flattened state buffer must be a whole number of 6-element states"
    );
    let n_state = len / 6;

    // Reinterpret the exact-length boxed allocation as [[f64; 6]] to avoid
    // per-state copy churn on sampled integrations.
    //
    // NOTE: `bytemuck::allocation::try_cast_vec` would express this with no
    // `unsafe` at all, but `bytemuck` is not a dependency of this crate and
    // adding one edits the workspace `Cargo.toml`, which moves
    // `build_policy_sha256`. If `bytemuck` ever arrives here for another
    // reason, replace this whole body with it.
    let raw = Box::into_raw(states.into_boxed_slice());
    let raw_states = std::ptr::slice_from_raw_parts_mut(raw.cast::<[f64; 6]>(), n_state);
    // SAFETY: `raw` is the unique, non-null, correctly-aligned pointer to a
    // `[f64]` of `len` elements that `Box::into_raw` just yielded, and nothing
    // else holds it.
    //
    // Layout: `[f64; 6]` is a primitive array, so it has the size of six `f64`
    // (48 bytes, no padding -- arrays never pad) and the alignment of `f64`
    // (8). The reconstituted slice therefore covers `48 * n_state` bytes at
    // alignment 8; the original covers `8 * len` bytes at alignment 8. The
    // `assert_eq!` above makes `len == 6 * n_state`, so `48 * n_state ==
    // 8 * len`: SAME base pointer, SAME byte extent, SAME alignment. The
    // `Box` deallocates with exactly the layout it was allocated with, which
    // is the condition that would otherwise be violated.
    //
    // Validity: every bit pattern of `f64` is a valid `f64`, so reinterpreting
    // initialized `f64` storage as `[f64; 6]` leaves no uninitialized or
    // invalid values. This is a pure regrouping, not a bit reinterpretation --
    // the payload is bit-identical.
    let boxed_states = unsafe { Box::from_raw(raw_states) };
    boxed_states.into_vec()
}

/// Convert a segment-relative Encke delta to the original baseline's delta.
///
/// When Encke rectification rebases to a new equinoctial reference, output deltas
/// from subsequent segments are relative to that new reference. This function
/// recovers the delta relative to the original baseline:
///
/// ```text
/// delta_orig(t) = delta_seg(t) + equinoc2eci(seg, t-seg_t0) - equinoc2eci(orig, t-orig_t0)
/// ```
#[inline]
pub(crate) fn correct_delta_to_original_baseline(
    delta_seg: &[f64; 6],
    t: f64,
    seg_equinoc: &[f64; 6],
    seg_t0: f64,
    orig_equinoc: &[f64; 6],
    orig_t0: f64,
) -> [f64; 6] {
    let mut base_seg = [0.0; 6];
    equinoc2eci_impl(seg_equinoc, 6, t - seg_t0, 0.0, &mut base_seg);

    let mut base_orig = [0.0; 6];
    equinoc2eci_impl(orig_equinoc, 6, t - orig_t0, 0.0, &mut base_orig);

    subtract_state_vectors(&add_state_vectors(delta_seg, &base_seg), &base_orig)
}

/// Reusable no-event final-state integrator for repeated propagations in one solve loop.
///
/// The embedded `LightyearRHS` is owned and reused sequentially; before each propagation
/// we reset initial state/time and invalidate the entire RHS cache.
pub struct ReusableFinalNoEventIntegrator {
    rhs: LightyearRHS,
    eclipse_root_rhs: Option<LightyearRHS>,
}

impl ReusableFinalNoEventIntegrator {
    /// Construct a reusable final-state integrator.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured scalar stepper or RHS authority is invalid.
    pub fn new(context: ScalarPropagationContext) -> anyhow::Result<Self> {
        let ScalarPropagationContext {
            jd0,
            config,
            gravity: ScalarGravityAssets { packed },
        } = context;
        validate_scalar_stepper_authority(config.as_ref(), "reusable final")?;
        let rhs =
            LightyearRHS::try_new([0.0; 6], 0.0, jd0, Arc::clone(&config), Arc::clone(&packed))?;
        let eclipse_root_rhs = if effective_scalar_srp(config.as_ref()) {
            Some(LightyearRHS::try_new(
                [0.0; 6],
                0.0,
                jd0,
                Arc::clone(&config),
                Arc::clone(&packed),
            )?)
        } else {
            None
        };
        Ok(Self {
            rhs,
            eclipse_root_rhs,
        })
    }

    /// Integrate from `t0_s` to `t_final_s` with no event handling and return final delta state.
    ///
    /// # Errors
    ///
    /// Returns a typed propagation failure when the solver rejects the trajectory.
    pub fn propagate(
        &mut self,
        init_equinoc_state: [f64; 6],
        t0_s: f64,
        t_final_s: f64,
    ) -> Result<[f64; 6], FinalPropagationFailure> {
        let eps = self.rhs.config.eps;
        self.rhs.adapt_cache_policy_for_eps(eps);
        self.rhs.reset_for_propagation(init_equinoc_state, t0_s);
        let stepper = resolve_auto_stepper(self.rhs.config.integrator_method, eps);
        // Counted HERE and not in `integrate_final_no_events_with_rhs`, because
        // that core is also the per-segment worker for Encke rectification
        // (see the segmented call in the checked path) -- bumping inside it
        // would count segments and call them propagations. This is the public
        // boundary and one call is one logical propagation by definition.
        //
        // Without this, `PROPAGATIONS` and `SPAN_MS` read 0 on the whole
        // reusable-no-event path while `SEGMENTS` counted correctly, so the
        // census printed `propagations 0  span 0.0 s` next to real data.
        crate::probe::bump_propagation(t0_s, t_final_s).map_err(FinalPropagationFailure::Census)?;
        if effective_scalar_srp(&self.rhs.config) {
            let root_rhs = self
                .eclipse_root_rhs
                .as_mut()
                .ok_or(FinalPropagationFailure::IntegrationFailure)?;
            let result = integrate_binary_eclipse_scalar_with_rhs(
                &BinaryEclipseRun {
                    init_equinoc_state,
                    t_eval: &[t_final_s],
                    t0_s,
                    tf_s: t_final_s,
                    enable_events: false,
                    eps,
                    stepper,
                },
                &mut self.rhs,
                root_rhs,
            )
            .map_err(final_failure_from_eclipse)?;
            return final_state_from_result(&result);
        }
        #[cfg(feature = "scalar-leg-observer")]
        return integrate_final_no_events_with_rhs(&self.rhs, t0_s, t_final_s, eps, stepper, None);
        #[cfg(not(feature = "scalar-leg-observer"))]
        integrate_final_no_events_with_rhs(&self.rhs, t0_s, t_final_s, eps, stepper)
    }
}

/// Reusable terminal-event-aware final-state integrator.
///
/// Force authority, epoch, tolerance, and stepper stay fixed. Each propagation
/// replaces baseline state/time and invalidates every state-dependent RHS cache.
/// Instances are sequential and must never be shared across worker threads.
pub(crate) struct ReusableFinalCheckedIntegrator {
    context: ScalarPropagationContext,
    rhs: LightyearRHS,
    eclipse_root_rhs: Option<LightyearRHS>,
    stats: FinalCheckedReuseStats,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct FinalCheckedReuseStats {
    pub propagations: usize,
    pub rhs_construct_count: usize,
    pub rhs_reuse_hits: usize,
}

impl ReusableFinalCheckedIntegrator {
    /// Construct a reusable terminal-event-aware integrator.
    ///
    /// # Errors
    ///
    /// Returns a typed failure when stepper, force, or RHS authority is invalid.
    pub fn new(context: ScalarPropagationContext) -> Result<Self, FinalPropagationFailure> {
        let eps = context.config.eps;
        let stepper = context.config.integrator_method;
        if validate_scalar_stepper_authority(&context.config, "reusable checked final").is_err()
            || crate::rhs::validate_atmosphere_model_code(context.config.atm_model).is_err()
            || (matches!(stepper, StepperMethod::Esdirk43)
                && crate::rhs_dual::validate_dual_newton_force_config(&context.config).is_err())
        {
            return Err(FinalPropagationFailure::IntegrationFailure);
        }
        let mut rhs = context
            .new_rhs([0.0; 6], 0.0)
            .map_err(|_| FinalPropagationFailure::IntegrationFailure)?;
        rhs.adapt_cache_policy_for_eps(eps);
        let mut eclipse_root_rhs = if effective_scalar_srp(&context.config) {
            Some(
                context
                    .new_rhs([0.0; 6], 0.0)
                    .map_err(|_| FinalPropagationFailure::IntegrationFailure)?,
            )
        } else {
            None
        };
        if let Some(root_rhs) = eclipse_root_rhs.as_mut() {
            root_rhs.adapt_cache_policy_for_eps(eps);
        }
        let rhs_construct_count = usize::from(eclipse_root_rhs.is_some())
            .checked_add(1)
            .ok_or(FinalPropagationFailure::IntegrationFailure)?;
        Ok(Self {
            context,
            rhs,
            eclipse_root_rhs,
            stats: FinalCheckedReuseStats {
                rhs_construct_count,
                ..FinalCheckedReuseStats::default()
            },
        })
    }

    /// Propagate with terminal-event checks.
    ///
    /// # Errors
    ///
    /// Returns a typed failure when force validation or propagation fails.
    pub(crate) fn propagate_checked(
        &mut self,
        init_equinoc_state: [f64; 6],
        t0_s: f64,
        t_final_s: f64,
    ) -> Result<[f64; 6], FinalPropagationFailure> {
        self.record_propagation()?;
        let final_times = [t_final_s];
        let request = ScalarPropagationRequest::new(
            &self.context,
            init_equinoc_state,
            &final_times,
            t0_s,
            t_final_s,
        )
        .with_events(true);
        if let Some(root_rhs) = self.eclipse_root_rhs.as_mut() {
            integrate_final_checked_core(
                request,
                FinalExecution::ReusedEclipse {
                    lane_rhs: &mut self.rhs,
                    root_rhs,
                },
            )
        } else {
            integrate_final_checked_core(request, FinalExecution::ReusedNoEclipse(&mut self.rhs))
        }
    }

    fn record_propagation(&mut self) -> Result<(), FinalPropagationFailure> {
        let propagations = self
            .stats
            .propagations
            .checked_add(1)
            .ok_or(FinalPropagationFailure::IntegrationFailure)?;
        if self.stats.propagations > 0 {
            let reuse_increment = usize::from(self.eclipse_root_rhs.is_some())
                .checked_add(1)
                .ok_or(FinalPropagationFailure::IntegrationFailure)?;
            self.stats.rhs_reuse_hits = self
                .stats
                .rhs_reuse_hits
                .checked_add(reuse_increment)
                .ok_or(FinalPropagationFailure::IntegrationFailure)?;
        }
        self.stats.propagations = propagations;
        Ok(())
    }

    /// Propagate through the identical reusable checked core with local
    /// feature-only diagnostic metrics.
    ///
    /// The reused RHS still resets its history-dependent Encke state on every
    /// row. Canonical builds do not compile this method and no override enters
    /// the numerical path.
    #[cfg(feature = "scalar-leg-observer")]
    #[must_use]
    pub(crate) fn propagate_checked_observed(
        &mut self,
        init_equinoc_state: [f64; 6],
        t0_s: f64,
        t_final_s: f64,
    ) -> ObservedFinalLeg {
        let mut observation = FinalObservation::new();
        let outcome = if self.record_propagation().is_err() {
            observation.mark_incomplete(ObservedFinalMetricError::CounterOverflow);
            Err(FinalPropagationFailure::IntegrationFailure)
        } else {
            let final_times = [t_final_s];
            let request = ScalarPropagationRequest::new(
                &self.context,
                init_equinoc_state,
                &final_times,
                t0_s,
                t_final_s,
            )
            .with_events(true);
            if let Some(root_rhs) = self.eclipse_root_rhs.as_mut() {
                integrate_final_checked_core_observed(
                    request,
                    FinalExecution::ReusedEclipse {
                        lane_rhs: &mut self.rhs,
                        root_rhs,
                    },
                    &mut observation,
                )
            } else {
                integrate_final_checked_core_observed(
                    request,
                    FinalExecution::ReusedNoEclipse(&mut self.rhs),
                    &mut observation,
                )
            }
        };
        let (metrics, terminal_status) = observation.into_parts();
        ObservedFinalLeg {
            outcome,
            metrics,
            terminal_status,
        }
    }

    /// Return counters for the one sequential reusable RHS instance.
    ///
    /// No production caller: the counters exist so the two RHS-reuse tests can
    /// assert that repeated propagations construct one RHS and hit the reuse
    /// path thereafter. That assertion IS the guard on the reuse contract, so
    /// the accounting is not removable with them.
    #[must_use]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the RHS-reuse tests are the only readers")
    )]
    pub const fn stats(&self) -> FinalCheckedReuseStats {
        self.stats
    }
}

fn integrate_final_no_events_with_rhs(
    rhs: &LightyearRHS,
    t0_s: f64,
    t_final_s: f64,
    eps: f64,
    stepper: StepperMethod,
    #[cfg(feature = "scalar-leg-observer")] observation: Option<&mut FinalObservation>,
) -> Result<[f64; 6], FinalPropagationFailure> {
    let dt_max = rhs.config.dt_max;
    let max_rejects = 50;
    let system = LightyearSystem { rhs };
    let y0 = [0.0; 6];

    // Release-visible, not a debug_assert. This helper is explicit-only, and
    // `stepper_ode_method` maps ESDIRK to Tsit5, so a debug build panicked here
    // while a release build quietly integrated with a different method.
    if matches!(stepper, StepperMethod::Esdirk43 | StepperMethod::Auto) {
        return Err(FinalPropagationFailure::MethodUnsupported);
    }
    rhs.clear_gravity_error();
    crate::probe::bump_segment();
    let eps_eff = eps.max(1e-12);
    let result = if stepper == StepperMethod::Dopri5Compat {
        let ly_config = LightyearConfig {
            eps: eps_eff,
            dt_max,
            max_steps: MAX_STEPS,
            max_rejects,
            force_eval: false,
            fast_single: true,
        };
        integrate_lightyear_dopri5_final(&system, &y0, t0_s, t_final_s, ly_config, None)
    } else {
        let cfg = IntegratorConfig {
            error_control: ErrorControl::Absolute { eps: eps_eff },
            h0: None,
            h_min: 1e-12,
            h_max: dt_max,
            max_steps: MAX_STEPS,
            max_rejects,
            force_eval: false,
        };
        integrate_final(
            &system,
            stepper_ode_method(stepper),
            &y0,
            t0_s,
            t_final_s,
            cfg,
        )
    };

    let gravity_error = rhs.take_gravity_error();
    crate::probe::bump_steps(result.stats.steps);
    crate::probe::bump_saturated(result.stats.saturated_steps);
    crate::probe::bump_rejected(result.stats.rejected_steps);
    crate::probe::observe_min_h(result.stats.min_accepted_h);
    crate::probe::observe_ramp(
        SegmentBoundary::ArcStart,
        result.stats.segment_span_s,
        &result.stats.first_accepted_h,
        result.stats.tail_h_sum,
        result.stats.tail_h_count,
    );
    crate::probe::observe_leg(
        dt_max,
        result.stats.segment_span_s,
        result.stats.steps,
        result.stats.evals,
        result.stats.rejected_steps,
    );
    crate::probe::observe_cache_cluster(
        result.stats.cache_cluster_steps,
        result.stats.cache_cluster_steps_untruncated,
    );
    crate::probe::observe_underflow(result.stats.underflow_accepts);
    #[cfg(feature = "scalar-leg-observer")]
    if let Some(observation) = observation {
        observation.record_solver(&result.stats, result.status);
        observation.record_encke_segment();
    }
    if let Some(error) = gravity_error {
        crate::probe::observe_prop_return(false);
        return Err(FinalPropagationFailure::Gravity(error));
    }
    if !matches!(result.status, OdeIntegrationStatus::Success) || result.y.len() < 6 {
        crate::probe::observe_prop_return(false);
        return Err(FinalPropagationFailure::IntegrationFailure);
    }
    let Some(values) = result
        .y
        .get(..6)
        .and_then(|slice| <&[f64; 6]>::try_from(slice).ok())
    else {
        crate::probe::observe_prop_return(false);
        return Err(FinalPropagationFailure::IntegrationFailure);
    };
    let mut delta_last = [0.0; 6];
    delta_last.copy_from_slice(values);
    // A `Success` status is not the same as a usable answer: record whether the
    // state actually came back finite, not merely that the solver said so.
    crate::probe::observe_prop_return(delta_last.iter().all(|v| v.is_finite()));
    Ok(delta_last)
}

/// Final-state integration using ESDIRK4(3) or Auto-switching meta-solver.
///
/// Constructs `DualVecJacobian` from the provided `LightyearDualRHS` and delegates
/// to the appropriate solver. No event handling.
#[cfg(feature = "autodiff")]
fn integrate_final_no_events_esdirk(
    rhs: &LightyearRHS,
    dual_rhs: &crate::rhs_dual::LightyearDualRHS,
    t0_s: f64,
    t_final_s: f64,
    eps: f64,
    stepper: StepperMethod,
) -> Result<[f64; 6], FinalPropagationFailure> {
    let dt_max = rhs.config.dt_max;
    let max_rejects = 50;
    let system = LightyearSystem { rhs };
    let y0 = [0.0; 6];

    rhs.clear_gravity_error();
    dual_rhs.reset_gravity_error();
    let result = match stepper {
        StepperMethod::Esdirk43 => {
            let jac = crate::adaptive_solver::DualVecJacobian::new(dual_rhs);
            let eps_eff = eps.max(1e-12);
            let cfg = IntegratorConfig {
                error_control: ErrorControl::Absolute { eps: eps_eff },
                h0: None,
                h_min: 1e-12,
                h_max: dt_max,
                max_steps: MAX_STEPS,
                max_rejects,
                force_eval: false,
            };
            integrate_final_esdirk(&system, &jac, &y0, t0_s, t_final_s, cfg)
        }
        StepperMethod::Auto => match crate::adaptive_solver::integrate_auto_final(
            rhs,
            dual_rhs,
            &y0,
            t0_s,
            t_final_s,
            eps,
            dt_max,
            MAX_STEPS,
            max_rejects,
        ) {
            Ok(result) => result,
            Err(error) => {
                let scalar_error = rhs.take_gravity_error();
                let _ = dual_rhs.take_gravity_error();
                return Err(FinalPropagationFailure::Gravity(
                    scalar_error.map_or(error, |scalar_error| scalar_error),
                ));
            }
        },
        // Unsupported explicit stepper in ESDIRK path: fail fast.
        _ => return Err(FinalPropagationFailure::IntegrationFailure),
    };

    if let Some(error) = take_esdirk_gravity_error(rhs, dual_rhs) {
        crate::probe::observe_prop_return(false);
        return Err(FinalPropagationFailure::Gravity(error));
    }
    if !matches!(result.status, OdeIntegrationStatus::Success) || result.y.len() < 6 {
        crate::probe::observe_prop_return(false);
        return Err(FinalPropagationFailure::IntegrationFailure);
    }
    let Some(values) = result
        .y
        .get(..6)
        .and_then(|slice| <&[f64; 6]>::try_from(slice).ok())
    else {
        crate::probe::observe_prop_return(false);
        return Err(FinalPropagationFailure::IntegrationFailure);
    };
    let mut delta_last = [0.0; 6];
    delta_last.copy_from_slice(values);
    // A `Success` status is not the same as a usable answer: record whether the
    // state actually came back finite, not merely that the solver said so.
    crate::probe::observe_prop_return(delta_last.iter().all(|v| v.is_finite()));
    Ok(delta_last)
}

// ============================================================================
// Core Implementation
// ============================================================================

fn finish_sampled_event_segment(
    mut result: IntegrationResult,
    segment: IntegrationResultSampled,
    mut handler: Option<EnckeEventHandler<'_>>,
    start_time: std::time::Instant,
) -> IntegrationResult {
    let IntegrationResultSampled {
        times,
        states,
        n_state,
        status,
        stats,
        event,
    } = segment;
    result.times = times;
    result.states = flatten_states(states, n_state);

    let event_invalid = handler.as_mut().is_some_and(|handler| {
        if let Some(detection) = handler.take_detection() {
            result.event_time = detection.refined_time;
            result.state_at_event = detection.state_at_event;
            result.event_interp_method = detection.interp_method;
            result.event_interp_error = detection.interp_error;
            if detection.event_type == crate::types::EventType::PerturbDeviation {
                result.perturb_deviation_fired = true;
            } else {
                result.terminal_event_fired = true;
                result.terminal_event_name = Cow::Borrowed(detection.event_type.name());
            }
        }
        handler.take_event_invalid()
    });
    if event_invalid {
        result.terminal_event_fired = true;
        result.perturb_deviation_fired = false;
        result.terminal_event_name = Cow::Borrowed("event_invalid");
    }
    if matches!(status, OdeIntegrationStatus::EventTriggered)
        && !result.terminal_event_fired
        && !result.perturb_deviation_fired
    {
        if let Some(event) = event {
            if event.y.len() == 6 {
                result.event_time = event.t;
                result.state_at_event.copy_from_slice(&event.y);
            }
            // Preserves the retired string mapping exactly: "handler"/"clamp"
            // fell through InterpMethod::from_str's default arm to None;
            // "linear"/"linear_clamp" both parsed to Linear.
            result.event_interp_method = match event.interp_method {
                crate::odesolve::SanitizedInterp::Handler
                | crate::odesolve::SanitizedInterp::Clamp => crate::types::InterpMethod::None,
                crate::odesolve::SanitizedInterp::Linear
                | crate::odesolve::SanitizedInterp::LinearClamp => {
                    crate::types::InterpMethod::Linear
                }
            };
            result.event_interp_error = event.interp_error;
        }
        result.terminal_event_fired = true;
        result.terminal_event_name = Cow::Borrowed("event_triggered");
    }
    match status {
        OdeIntegrationStatus::Success | OdeIntegrationStatus::EventTriggered => {}
        OdeIntegrationStatus::MaxStepsExceeded => {
            result.max_steps_exceeded = true;
            result.terminal_event_fired = true;
            result.terminal_event_name = Cow::Borrowed("max_steps_exceeded");
        }
        OdeIntegrationStatus::StepUnderflow => {
            result.terminal_event_fired = true;
            result.terminal_event_name = Cow::Borrowed("step_underflow");
        }
        OdeIntegrationStatus::InvalidInput => {
            result.terminal_event_fired = true;
            result.terminal_event_name = Cow::Borrowed("invalid_input");
        }
        OdeIntegrationStatus::NanEncountered => {
            result.terminal_event_fired = true;
            result.terminal_event_name = Cow::Borrowed("nan_encountered");
        }
        OdeIntegrationStatus::NonFiniteState => {
            result.terminal_event_fired = true;
            result.terminal_event_name = Cow::Borrowed("non_finite_state");
        }
        OdeIntegrationStatus::MaxRejectsExceeded => {
            result.terminal_event_fired = true;
            result.terminal_event_name = Cow::Borrowed("max_rejects_exceeded");
        }
        OdeIntegrationStatus::EventInvalid => {
            result.terminal_event_fired = true;
            result.terminal_event_name = Cow::Borrowed("event_invalid");
        }
    }
    result.metrics =
        OdeMetrics::from_values(stats.steps, stats.evals, elapsed_micros_u64(start_time));
    result
}

fn sampled_gravity_failure(
    mut result: IntegrationResult,
    error: GravityError,
    start_time: std::time::Instant,
) -> IntegrationResult {
    result.terminal_event_fired = true;
    result.terminal_event_name = Cow::Borrowed("");
    result.terminal_gravity_error = Some(error);
    result.metrics = OdeMetrics::from_values(0, 0, elapsed_micros_u64(start_time));
    result
}

#[derive(Clone, Copy)]
struct SampledRun<'context, 'eval> {
    request: ScalarPropagationRequest<'context>,
    eval_slice: &'eval [f64],
    fast_single: bool,
    start_time: std::time::Instant,
}

fn integrate_sampled_event_path(run: SampledRun<'_, '_>) -> IntegrationResult {
    let request = run.request;
    let context = request.context;
    let init_equinoc_state = request.init_equinoc_state;
    let t0_s = request.t0_s;
    let t_final_s = request.t_final_s;
    let eps = context.config.eps;
    let stepper = request.stepper;
    let force_eval = request.force_eval();
    let eval_slice = run.eval_slice;
    let fast_single = run.fast_single;
    let start_time = run.start_time;
    let config = &context.config;
    let packed = &context.gravity.packed;
    let jd0 = context.jd0;
    let mut result = IntegrationResult::default();
    let dt_max = config.dt_max;
    let event_interp_tol = 1e-6;
    let max_rejects = 50;
    #[cfg(feature = "autodiff")]
    let dual_inputs = matches!(stepper, StepperMethod::Esdirk43)
        .then(|| (Arc::clone(config), Arc::clone(packed)));
    let Ok(mut rhs) = context.new_rhs(init_equinoc_state, t0_s) else {
        result.terminal_event_fired = true;
        result.terminal_event_name = Cow::Borrowed("invalid_force_config");
        return result;
    };
    rhs.init_equinoc_state = init_equinoc_state;
    rhs.t0_s = t0_s;
    rhs.adapt_cache_policy_for_eps(eps);
    rhs.reset_cache();
    let system = LightyearSystem { rhs: &rhs };
    let y0 = [0.0; 6];
    let mut handler = Some(EnckeEventHandler::new(
        rhs.baseline_calculator(),
        t0_s,
        y0,
        &rhs,
        event_interp_tol,
        eps,
        max_rejects,
    ));
    let segment = match stepper {
        #[cfg(feature = "autodiff")]
        StepperMethod::Esdirk43 => {
            let Some((config, packed)) = dual_inputs else {
                result.terminal_event_fired = true;
                result.terminal_event_name = Cow::Borrowed("invalid_force_config");
                return result;
            };
            let Ok(dual_rhs) = crate::rhs_dual::LightyearDualRHS::new(
                init_equinoc_state,
                t0_s,
                jd0,
                config,
                packed,
            ) else {
                result.terminal_event_fired = true;
                result.terminal_event_name = Cow::Borrowed("invalid_force_config");
                return result;
            };
            let dual_rhs = Box::new(dual_rhs);
            let jac = crate::adaptive_solver::DualVecJacobian::new(&dual_rhs);
            let cfg = IntegratorConfig {
                error_control: ErrorControl::Absolute {
                    eps: eps.max(1e-12),
                },
                h0: None,
                h_min: 1e-12,
                h_max: dt_max,
                max_steps: MAX_STEPS,
                max_rejects,
                force_eval,
            };
            if let Some(handler) = handler.as_mut() {
                rhs.clear_gravity_error();
                dual_rhs.reset_gravity_error();
                let segment = integrate_sampled_with_events_esdirk(
                    &system, &jac, &y0, eval_slice, cfg, handler,
                );
                if let Some(error) = take_esdirk_gravity_error(&rhs, &dual_rhs) {
                    return sampled_gravity_failure(result, error, start_time);
                }
                segment
            } else {
                let mut states = vec![0.0; eval_slice.len().saturating_mul(y0.len())];
                rhs.clear_gravity_error();
                dual_rhs.reset_gravity_error();
                let first_result =
                    integrate_sampled_esdirk_into(&system, &jac, &y0, eval_slice, cfg, &mut states);
                if let Some(error) = take_esdirk_gravity_error(&rhs, &dual_rhs) {
                    return sampled_gravity_failure(result, error, start_time);
                }
                if matches!(first_result.status, OdeIntegrationStatus::Success) {
                    sampled_from_into_result(eval_slice, y0.len(), states, first_result)
                } else {
                    rhs.clear_gravity_error();
                    dual_rhs.reset_gravity_error();
                    let fallback = integrate_sampled_esdirk(&system, &jac, &y0, eval_slice, cfg);
                    if let Some(error) = take_esdirk_gravity_error(&rhs, &dual_rhs) {
                        return sampled_gravity_failure(result, error, start_time);
                    }
                    fallback
                }
            }
        }
        #[cfg(not(feature = "autodiff"))]
        StepperMethod::Esdirk43 => {
            result.terminal_event_fired = true;
            result.terminal_event_name = Cow::Borrowed("invalid_force_config");
            return result;
        }
        _ => match integrate_segment_with_method(
            &system,
            &y0,
            eval_slice,
            SegmentControls {
                t0_s,
                t_final_s,
                eps,
                dt_max,
                force_eval,
                fast_single,
                max_steps: MAX_STEPS,
                max_rejects,
                stepper,
                boundary: SegmentBoundary::ArcStart,
            },
            handler
                .as_mut()
                .map(|value| -> &mut dyn OdeEventHandler { value }),
        ) {
            Ok(segment) => segment,
            Err(error) => return sampled_gravity_failure(result, error, start_time),
        },
    };
    finish_sampled_event_segment(result, segment, handler, start_time)
}

struct SampledRectState {
    original_equinoc: [f64; 6],
    original_t0_s: f64,
    current_equinoc: [f64; 6],
    current_t0_s: f64,
    had_rebase: bool,
    times: Vec<f64>,
    states: Vec<[f64; 6]>,
    total_steps: usize,
    total_evals: usize,
    status: OdeIntegrationStatus,
    gravity_error: Option<GravityError>,
    eval_consumed: usize,
    rhs: Option<LightyearRHS>,
}

struct SampledRectSettings<'a> {
    t_final_s: f64,
    eps: f64,
    context: &'a ScalarPropagationContext,
    force_eval: bool,
    stepper: StepperMethod,
    threshold_sq: f64,
}

enum RectAdvance {
    Continue,
    Stop(OdeIntegrationStatus),
    Gravity(GravityError),
    Return(Box<IntegrationResult>),
}

fn advance_unsampled_rect_segment(
    state: &mut SampledRectState,
    settings: &SampledRectSettings<'_>,
    segment_tf: f64,
) -> RectAdvance {
    #[cfg(feature = "autodiff")]
    if matches!(settings.stepper, StepperMethod::Esdirk43) {
        let Ok(mut rhs) = settings
            .context
            .new_rhs(state.current_equinoc, state.current_t0_s)
        else {
            return RectAdvance::Stop(OdeIntegrationStatus::InvalidInput);
        };
        rhs.adapt_cache_policy_for_eps(settings.eps);
        rhs.reset_for_propagation(state.current_equinoc, state.current_t0_s);
        let Ok(dual_rhs) = crate::rhs_dual::LightyearDualRHS::new(
            state.current_equinoc,
            state.current_t0_s,
            settings.context.jd0,
            Arc::clone(&settings.context.config),
            Arc::clone(&settings.context.gravity.packed),
        ) else {
            return RectAdvance::Stop(OdeIntegrationStatus::InvalidInput);
        };
        let dual_rhs = Box::new(dual_rhs);
        let delta = match integrate_final_no_events_esdirk(
            &rhs,
            &dual_rhs,
            state.current_t0_s,
            segment_tf,
            settings.eps,
            settings.stepper,
        ) {
            Ok(delta) => delta,
            Err(FinalPropagationFailure::Gravity(error)) => return RectAdvance::Gravity(error),
            Err(_) => return RectAdvance::Stop(OdeIntegrationStatus::NonFiniteState),
        };
        if needs_rectification(
            position_norm_sq(&delta),
            settings.threshold_sq,
            segment_tf,
            settings.t_final_s,
        ) {
            state.current_equinoc = rebase_equinoc_from_delta(
                &state.current_equinoc,
                state.current_t0_s,
                segment_tf,
                &delta,
            );
            state.current_t0_s = segment_tf;
            state.had_rebase = true;
        }
        return RectAdvance::Continue;
    }

    if state.rhs.is_none() {
        let Ok(rhs) = settings
            .context
            .new_rhs(state.current_equinoc, state.current_t0_s)
        else {
            return RectAdvance::Stop(OdeIntegrationStatus::InvalidInput);
        };
        state.rhs = Some(rhs);
    }
    let Some(rhs) = state.rhs.as_mut() else {
        return RectAdvance::Stop(OdeIntegrationStatus::InvalidInput);
    };
    rhs.adapt_cache_policy_for_eps(settings.eps);
    rhs.reset_for_propagation(state.current_equinoc, state.current_t0_s);
    #[cfg(not(feature = "scalar-leg-observer"))]
    let delta = match integrate_final_no_events_with_rhs(
        &*rhs,
        state.current_t0_s,
        segment_tf,
        settings.eps,
        settings.stepper,
    ) {
        Ok(delta) => delta,
        Err(FinalPropagationFailure::Gravity(error)) => return RectAdvance::Gravity(error),
        Err(_) => return RectAdvance::Stop(OdeIntegrationStatus::NonFiniteState),
    };
    #[cfg(feature = "scalar-leg-observer")]
    let delta = match integrate_final_no_events_with_rhs(
        &*rhs,
        state.current_t0_s,
        segment_tf,
        settings.eps,
        settings.stepper,
        None,
    ) {
        Ok(delta) => delta,
        Err(FinalPropagationFailure::Gravity(error)) => return RectAdvance::Gravity(error),
        Err(_) => return RectAdvance::Stop(OdeIntegrationStatus::NonFiniteState),
    };
    if needs_rectification(
        position_norm_sq(&delta),
        settings.threshold_sq,
        segment_tf,
        settings.t_final_s,
    ) {
        state.current_equinoc = rebase_equinoc_from_delta(
            &state.current_equinoc,
            state.current_t0_s,
            segment_tf,
            &delta,
        );
        state.current_t0_s = segment_tf;
        state.had_rebase = true;
    }
    RectAdvance::Continue
}

fn sampled_rect_solver(
    state: &mut SampledRectState,
    settings: &SampledRectSettings<'_>,
    solver_eval: &[f64],
    segment_tf: f64,
) -> Result<IntegrationResultSampled, RectAdvance> {
    if state.rhs.is_none() {
        let Ok(rhs) = settings
            .context
            .new_rhs(state.current_equinoc, state.current_t0_s)
        else {
            return Err(RectAdvance::Stop(OdeIntegrationStatus::InvalidInput));
        };
        state.rhs = Some(rhs);
    }
    let Some(rhs) = state.rhs.as_mut() else {
        return Err(RectAdvance::Stop(OdeIntegrationStatus::InvalidInput));
    };
    rhs.adapt_cache_policy_for_eps(settings.eps);
    rhs.reset_for_propagation(state.current_equinoc, state.current_t0_s);
    let system = LightyearSystem { rhs: &*rhs };
    let y0 = [0.0; 6];
    match settings.stepper {
        #[cfg(feature = "autodiff")]
        StepperMethod::Esdirk43 => {
            let Ok(dual_rhs) = crate::rhs_dual::LightyearDualRHS::new(
                state.current_equinoc,
                state.current_t0_s,
                settings.context.jd0,
                Arc::clone(&settings.context.config),
                Arc::clone(&settings.context.gravity.packed),
            ) else {
                return Err(RectAdvance::Return(Box::new(IntegrationResult {
                    terminal_event_fired: true,
                    terminal_event_name: Cow::Borrowed("invalid_force_config"),
                    ..IntegrationResult::default()
                })));
            };
            let dual_rhs = Box::new(dual_rhs);
            let jac = crate::adaptive_solver::DualVecJacobian::new(&dual_rhs);
            let cfg = IntegratorConfig {
                error_control: ErrorControl::Absolute {
                    eps: settings.eps.max(1e-12),
                },
                h0: None,
                h_min: 1e-12,
                h_max: settings.context.config.dt_max,
                max_steps: MAX_STEPS,
                max_rejects: 50,
                force_eval: settings.force_eval,
            };
            let mut states = vec![0.0; solver_eval.len().saturating_mul(y0.len())];
            rhs.clear_gravity_error();
            dual_rhs.reset_gravity_error();
            let first_result =
                integrate_sampled_esdirk_into(&system, &jac, &y0, solver_eval, cfg, &mut states);
            if let Some(error) = take_esdirk_gravity_error(rhs, &dual_rhs) {
                return Err(RectAdvance::Gravity(error));
            }
            Ok(
                if matches!(first_result.status, OdeIntegrationStatus::Success) {
                    sampled_from_into_result(solver_eval, y0.len(), states, first_result)
                } else {
                    rhs.clear_gravity_error();
                    dual_rhs.reset_gravity_error();
                    let fallback = integrate_sampled_esdirk(&system, &jac, &y0, solver_eval, cfg);
                    if let Some(error) = take_esdirk_gravity_error(rhs, &dual_rhs) {
                        return Err(RectAdvance::Gravity(error));
                    }
                    fallback
                },
            )
        }
        #[cfg(not(feature = "autodiff"))]
        StepperMethod::Esdirk43 => Err(RectAdvance::Return(Box::new(IntegrationResult {
            terminal_event_fired: true,
            terminal_event_name: Cow::Borrowed("invalid_force_config"),
            ..IntegrationResult::default()
        }))),
        _ => integrate_segment_with_method(
            &system,
            &y0,
            solver_eval,
            SegmentControls {
                t0_s: state.current_t0_s,
                t_final_s: segment_tf,
                eps: settings.eps,
                dt_max: settings.context.config.dt_max,
                force_eval: settings.force_eval,
                fast_single: solver_eval.len() == 1
                    && solver_eval
                        .first()
                        .is_some_and(|time| (*time - segment_tf).abs() < 1e-9),
                max_steps: MAX_STEPS,
                max_rejects: 50,
                stepper: settings.stepper,
                boundary: SegmentBoundary::Rebased,
            },
            None,
        )
        .map_err(RectAdvance::Gravity),
    }
}

fn advance_sampled_rect_segment(
    state: &mut SampledRectState,
    settings: &SampledRectSettings<'_>,
    seg_eval: &[f64],
    seg_count: usize,
    segment_tf: f64,
) -> RectAdvance {
    let needs_hidden_start = !matches!(settings.stepper, StepperMethod::Dopri5Compat)
        && seg_eval
            .first()
            .is_none_or(|time| (time - state.current_t0_s).abs() > 1e-9);
    let needs_hidden_endpoint = seg_eval
        .last()
        .is_none_or(|time| (time - segment_tf).abs() > 1e-9);
    let mut augmented_eval = Vec::new();
    let solver_eval = if needs_hidden_start || needs_hidden_endpoint {
        augmented_eval.reserve(
            seg_eval
                .len()
                .saturating_add(usize::from(needs_hidden_start))
                .saturating_add(usize::from(needs_hidden_endpoint)),
        );
        if needs_hidden_start {
            augmented_eval.push(state.current_t0_s);
        }
        augmented_eval.extend_from_slice(seg_eval);
        if needs_hidden_endpoint {
            augmented_eval.push(segment_tf);
        }
        augmented_eval.as_slice()
    } else {
        seg_eval
    };
    let segment = match sampled_rect_solver(state, settings, solver_eval, segment_tf) {
        Ok(segment) => segment,
        Err(outcome) => return outcome,
    };
    let times = segment.times;
    let states = flatten_states(segment.states, segment.n_state);
    state.total_steps = state.total_steps.saturating_add(segment.stats.steps);
    state.total_evals = state.total_evals.saturating_add(segment.stats.evals);

    let returned_len = times.len().min(states.len());
    let hidden_start_returned = needs_hidden_start
        && returned_len > 0
        && times
            .iter()
            .zip(&states)
            .next()
            .is_some_and(|(&time, _)| (time - state.current_t0_s).abs() <= 1e-9);
    let endpoint_returned = times
        .iter()
        .zip(&states)
        .next_back()
        .is_some_and(|(&time, _)| (time - segment_tf).abs() <= 1e-9);
    let visible_start = usize::from(hidden_start_returned);
    let visible_end =
        returned_len.saturating_sub(usize::from(needs_hidden_endpoint && endpoint_returned));
    let append_visible = |state: &mut SampledRectState| {
        append_visible_sampled_states(
            &mut state.times,
            &mut state.states,
            &times,
            &states,
            visible_start,
            visible_end,
            &state.current_equinoc,
            state.current_t0_s,
            &state.original_equinoc,
            state.original_t0_s,
            state.had_rebase,
        );
    };
    if !matches!(segment.status, OdeIntegrationStatus::Success) {
        append_visible(state);
        return RectAdvance::Stop(segment.status);
    }
    if !endpoint_returned || (needs_hidden_start && !hidden_start_returned) {
        append_visible(state);
        return RectAdvance::Stop(OdeIntegrationStatus::InvalidInput);
    }
    let Some((_, last_delta)) = times.iter().zip(&states).next_back() else {
        return RectAdvance::Stop(OdeIntegrationStatus::InvalidInput);
    };
    let needs_rebase = needs_rectification(
        position_norm_sq(last_delta),
        settings.threshold_sq,
        segment_tf,
        settings.t_final_s,
    );
    append_visible(state);
    state.eval_consumed = state.eval_consumed.saturating_add(seg_count);
    if needs_rebase {
        state.current_equinoc = rebase_equinoc_from_delta(
            &state.current_equinoc,
            state.current_t0_s,
            segment_tf,
            last_delta,
        );
        state.current_t0_s = segment_tf;
        state.had_rebase = true;
    }
    RectAdvance::Continue
}

fn finish_sampled_rectification(
    mut state: SampledRectState,
    start_time: std::time::Instant,
) -> IntegrationResult {
    let mut result = IntegrationResult::default();
    if let Some(error) = state.gravity_error {
        result.terminal_event_fired = true;
        result.terminal_gravity_error = Some(error);
    } else {
        match state.status {
            OdeIntegrationStatus::Success => {}
            OdeIntegrationStatus::MaxStepsExceeded => {
                result.max_steps_exceeded = true;
                result.terminal_event_fired = true;
                result.terminal_event_name = Cow::Borrowed("max_steps_exceeded");
            }
            OdeIntegrationStatus::StepUnderflow => {
                result.terminal_event_fired = true;
                result.terminal_event_name = Cow::Borrowed("step_underflow");
            }
            OdeIntegrationStatus::NanEncountered => {
                result.terminal_event_fired = true;
                result.terminal_event_name = Cow::Borrowed("nan_encountered");
            }
            OdeIntegrationStatus::NonFiniteState => {
                result.terminal_event_fired = true;
                result.terminal_event_name = Cow::Borrowed("non_finite_state");
            }
            OdeIntegrationStatus::MaxRejectsExceeded => {
                result.terminal_event_fired = true;
                result.terminal_event_name = Cow::Borrowed("max_rejects_exceeded");
            }
            status => {
                result.terminal_event_fired = true;
                result.terminal_event_name = Cow::Owned(format!("{status:?}"));
            }
        }
    }
    result.times = std::mem::take(&mut state.times);
    result.states = std::mem::take(&mut state.states);
    result.metrics = OdeMetrics::from_values(
        state.total_steps,
        state.total_evals,
        elapsed_micros_u64(start_time),
    );
    result
}

fn integrate_sampled_rectified_path(run: SampledRun<'_, '_>) -> IntegrationResult {
    let request = run.request;
    let eval_slice = run.eval_slice;
    let init_equinoc_state = request.init_equinoc_state;
    let t0_s = request.t0_s;
    let t_final_s = request.t_final_s;
    let eps = request.context.config.eps;
    let stepper = request.stepper;
    let force_eval = request.force_eval();
    let start_time = run.start_time;
    let mut state = SampledRectState {
        original_equinoc: init_equinoc_state,
        original_t0_s: t0_s,
        current_equinoc: init_equinoc_state,
        current_t0_s: t0_s,
        had_rebase: false,
        times: Vec::new(),
        states: Vec::new(),
        total_steps: 0,
        total_evals: 0,
        status: OdeIntegrationStatus::Success,
        gravity_error: None,
        eval_consumed: 0,
        rhs: None,
    };
    let settings = SampledRectSettings {
        t_final_s,
        eps,
        context: request.context,
        force_eval,
        stepper,
        threshold_sq: PERTURB_DEVIATION_THRESHOLD_KM * PERTURB_DEVIATION_THRESHOLD_KM,
    };
    let forward = t_final_s >= t0_s;
    // Step-size carry is scoped to THIS propagation. The segments below pass
    // `SegmentBoundary::Rebased`, so without this reset the first one would
    // open at whatever `h` the previous arc on this rayon thread exited with —
    // nondeterministic across schedules. Guarded by
    // `hcarry_reset_scopes_carry_to_one_propagation` in `tests/rect_loop_pin.rs`.
    hcarry_reset();
    // `reached_endpoint` tracks ARC PROGRESS, which is not the same thing as
    // the Encke reference epoch.
    //
    // `current_t0_s` is the epoch of the reference orbit and advances only when
    // a segment rebases. `needs_rectification` forces a rebase on every segment
    // whose end is not the arc end, so intermediate segments advance it -- but
    // the FINAL segment rebases only if its deviation exceeds the threshold.
    // For an ordinary short arc it does not, so `current_t0_s` stopped at the
    // start, the loop condition below never became true, and the terminal leg
    // was recomputed until the iteration cap. That work produced no output --
    // `eval_consumed` was already exhausted -- and was absent from the reported
    // statistics.
    let mut reached_endpoint = (t_final_s - state.current_t0_s).abs() < 1e-12;
    for _ in 0..100 {
        if reached_endpoint {
            break;
        }
        let segment_tf =
            bounded_segment_end(state.current_t0_s, t_final_s, MAX_RECTIFICATION_SEGMENT_S);
        let Some(remaining_eval) = eval_slice.get(state.eval_consumed..) else {
            state.status = OdeIntegrationStatus::InvalidInput;
            break;
        };
        let count = if forward {
            remaining_eval.partition_point(|&time| time <= segment_tf + 1e-9)
        } else {
            remaining_eval.partition_point(|&time| time >= segment_tf - 1e-9)
        };
        let Some(segment_eval) = remaining_eval.get(..count) else {
            state.status = OdeIntegrationStatus::InvalidInput;
            break;
        };
        let outcome = if segment_eval.is_empty() {
            advance_unsampled_rect_segment(&mut state, &settings, segment_tf)
        } else {
            advance_sampled_rect_segment(&mut state, &settings, segment_eval, count, segment_tf)
        };
        match outcome {
            RectAdvance::Continue => {
                // The segment that ends at the arc end IS the arc, completed.
                if (t_final_s - segment_tf).abs() < 1e-12 {
                    reached_endpoint = true;
                }
            }
            RectAdvance::Stop(status) => {
                state.status = status;
                break;
            }
            RectAdvance::Gravity(error) => {
                state.gravity_error = Some(error);
                break;
            }
            RectAdvance::Return(result) => return *result,
        }
    }
    // CAP EXHAUSTION FAILS CLOSED. The loop used to fall out of a spent budget
    // with `status` still `Success`, so an arc longer than 100 segments
    // published its accumulated rows and reported success for a propagation
    // that never reached `tf`.
    if !reached_endpoint && matches!(state.status, OdeIntegrationStatus::Success) {
        state.status = OdeIntegrationStatus::MaxStepsExceeded;
    }
    finish_sampled_rectification(state, start_time)
}

/// Integrate scalar Encke dynamics with one typed force-and-asset authority.
///
/// This is the SAMPLED path: the rect loop `rect_loop_pin` and `prop_timing`
/// refer to by its former name `integrate_sampled_inner` lives here.
///
/// # Errors
///
/// Returns a census failure rather than emitting partial propagation evidence.
pub fn integrate_adaptive(
    request: ScalarPropagationRequest<'_>,
) -> Result<IntegrationResult, PropagationCensusError> {
    let context = request.context;
    let init_equinoc_state = request.init_equinoc_state;
    let t_eval = request.t_eval;
    let t0_s = request.t0_s;
    let t_final_s = request.t_final_s;
    let eps = context.config.eps;
    let enable_events = request.enable_events;
    let stepper = request.stepper;
    let config = &context.config;
    let mut result = IntegrationResult::default();
    let start_time = std::time::Instant::now();
    if validate_scalar_stepper_authority(config, "sampled integration").is_err()
        || crate::rhs::validate_atmosphere_model_code(config.atm_model).is_err()
        || (matches!(stepper, StepperMethod::Esdirk43)
            && crate::rhs_dual::validate_dual_newton_force_config(config).is_err())
    {
        result.terminal_event_fired = true;
        result.terminal_event_name = Cow::Borrowed("invalid_force_config");
        return Ok(result);
    }

    if effective_scalar_srp(config) {
        // Binary eclipse is science-authoritative only when requested output
        // times cannot change the accepted RK trajectory. Therefore the generic
        // `force_eval` request is deliberately overridden here: coordinated
        // sampling always reconstructs from accepted steps, then discards and
        // replays any step containing a boundary.
        crate::probe::bump_propagation(t0_s, t_final_s)?;
        return Ok(
            match integrate_binary_eclipse_scalar(
                init_equinoc_state,
                t_eval,
                t0_s,
                t_final_s,
                enable_events,
                context.binary_eclipse_context(),
            ) {
                Ok(result) => result,
                Err(error) => IntegrationResult {
                    terminal_event_fired: true,
                    terminal_event_name: Cow::Borrowed(eclipse_error_name(error)),
                    terminal_eclipse_error: Some(error),
                    ..IntegrationResult::default()
                },
            },
        );
    }

    let fast_single = !enable_events
        && t_eval.len() == 1
        && t_eval
            .first()
            .is_some_and(|time| (*time - t_final_s).abs() < 1e-9);

    let eval_idx = if fast_single {
        0
    } else {
        let dt_total_initial = t_final_s - t0_s;
        let tol = 1e-9;
        if dt_total_initial >= 0.0 {
            // Forward integration: skip evaluation times strictly before t0.
            t_eval.partition_point(|&te| te < t0_s - tol)
        } else {
            // Backward integration: skip evaluation times strictly after t0.
            t_eval.partition_point(|&te| te > t0_s + tol)
        }
    };

    // Compute eval_slice before potential Arc consumption (needed by both paths).
    let fallback_eval = [t_final_s];
    let eval_slice = t_eval
        .get(eval_idx..)
        .filter(|slice| !slice.is_empty())
        .unwrap_or(&fallback_eval);

    // Resolve Auto to concrete method before any dispatch.
    let stepper = resolve_auto_stepper(stepper, eps);

    // ========================================================================
    // !enable_events: segmented Encke rectification for the sampled path.
    //
    // Mirrors the always-on rectification in
    // `integrate_final_checked`:
    // integrate at most MAX_RECTIFICATION_SEGMENT_S seconds at a time, check
    // |δr|² after each segment, rebase the Encke reference orbit if
    // PERTURB_DEVIATION_THRESHOLD_KM is exceeded. Output deltas from rebased
    // segments are corrected back to the original baseline.
    //
    // That threshold is 10 km. This comment said "the 2 km threshold" until
    // 2026-08-04, which is the value the constant carried before the authorized
    // re-pin of 2026-07-27 moved it 2.0 -> 10.0 (see `tests/strict_hf_pin.rs`
    // and the constant's own doc in `types.rs`); read literally it overstated
    // how often the Encke reference is rebased by a factor of five in deviation.
    // Take the number from PERTURB_DEVIATION_THRESHOLD_KM, not from here.
    //
    // Short arcs (≤ MAX_RECTIFICATION_SEGMENT_S, i.e. 5400 s) take a single loop
    // iteration with zero overhead: no rebase, no delta correction.
    // ========================================================================
    if !enable_events {
        return Ok(integrate_sampled_rectified_path(SampledRun {
            request: request.with_resolved_stepper(stepper),
            eval_slice,
            fast_single,
            start_time,
        }));
    }
    // ENDPOINT POSTCONDITION for the event routes that infer their interval.
    //
    // DOPRI receives `t0_s`/`t_final_s` explicitly. Every other sampled event
    // wrapper takes only `t_eval` and reads the integration interval off its
    // first and last elements. So if the first sample is after `t0`, the
    // initial state is silently relabelled as belonging to that later time and
    // the dynamics and events before it never happen; if the last sample is
    // before `tf`, the tail is never propagated. Both directions are exposed.
    //
    // Refused rather than inferred. Quietly integrating a different interval
    // than the caller asked for and reporting success is the failure mode worth
    // preventing; a caller that wants the sparse behaviour can ask for the
    // interval it actually wants.
    if !matches!(stepper, StepperMethod::Dopri5Compat)
        && !sampled_eval_spans_request(eval_slice, t0_s, t_final_s)
    {
        return Ok(IntegrationResult {
            terminal_event_fired: true,
            terminal_event_name: Cow::Borrowed("sampled_eval_does_not_span_request"),
            ..IntegrationResult::default()
        });
    }
    Ok(integrate_sampled_event_path(SampledRun {
        request: request.with_resolved_stepper(stepper),
        eval_slice,
        fast_single,
        start_time,
    }))
}

/// Whether a sampled evaluation grid covers the requested physical interval.
///
/// Direction-aware: a backward arc has `t_final_s < t0_s` and a descending
/// grid. The tolerance matches the one the dispatcher already uses to drop
/// samples before `t0`, so a grid that survives that trim is judged on the same
/// scale that produced it.
fn sampled_eval_spans_request(eval: &[f64], t0_s: f64, t_final_s: f64) -> bool {
    const SPAN_TOL_S: f64 = 1e-9;
    let (Some(&first), Some(&last)) = (eval.first(), eval.last()) else {
        return false;
    };
    if !first.is_finite() || !last.is_finite() {
        return false;
    }
    if t_final_s >= t0_s {
        first <= t0_s + SPAN_TOL_S && last >= t_final_s - SPAN_TOL_S
    } else {
        first >= t0_s - SPAN_TOL_S && last <= t_final_s + SPAN_TOL_S
    }
}

/// Typed failure from final-only HF propagation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinalPropagationFailure {
    Ground,
    LeftEarth,
    Eccentricity,
    NanState,
    EventInvalid,
    Gravity(GravityError),
    Census(crate::probe::PropagationCensusError),
    Eclipse(EclipseError),
    IntegrationFailure,
    /// The requested stepper cannot be executed on the selected route.
    ///
    /// Distinct from `IntegrationFailure`: nothing was integrated. Previously
    /// these routes substituted Tsit5 and returned a successful result computed
    /// by a method the caller did not ask for, which is worse than failing.
    MethodUnsupported,
}

/// Terminal status observed for a diagnostic scalar final leg.
///
/// `Unavailable` means canonical path cannot provide complete local solver
/// accounting. A completed finite leg reports `Success` even when its final
/// internal solve stopped at a handled, nonterminal event before composition
/// reached the requested endpoint.
#[cfg(feature = "scalar-leg-observer")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservedSolverTerminalStatus {
    NotStarted,
    Unavailable,
    Success,
    MaxStepsExceeded,
    StepUnderflow,
    InvalidInput,
    NanEncountered,
    EventTriggered,
    NonFiniteState,
    MaxRejectsExceeded,
    EventInvalid,
}

#[cfg(feature = "scalar-leg-observer")]
impl From<OdeIntegrationStatus> for ObservedSolverTerminalStatus {
    fn from(status: OdeIntegrationStatus) -> Self {
        match status {
            OdeIntegrationStatus::Success => Self::Success,
            OdeIntegrationStatus::MaxStepsExceeded => Self::MaxStepsExceeded,
            OdeIntegrationStatus::StepUnderflow => Self::StepUnderflow,
            OdeIntegrationStatus::InvalidInput => Self::InvalidInput,
            OdeIntegrationStatus::NanEncountered => Self::NanEncountered,
            OdeIntegrationStatus::EventTriggered => Self::EventTriggered,
            OdeIntegrationStatus::NonFiniteState => Self::NonFiniteState,
            OdeIntegrationStatus::MaxRejectsExceeded => Self::MaxRejectsExceeded,
            OdeIntegrationStatus::EventInvalid => Self::EventInvalid,
        }
    }
}

/// Local diagnostic metrics cannot be complete.
///
/// `CounterOverflow` is local accounting failure. The remaining variants mark
/// canonical paths whose complete metric source is not available. Physics
/// outcome stays independently valid or invalid.
#[cfg(feature = "scalar-leg-observer")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservedFinalMetricError {
    CounterOverflow,
    EclipseMetricsUnavailable,
    EsdirkMetricsUnavailable,
    EventSegmentMetricsUnavailable,
}

/// Fixed scalar-leg counters for local, non-production diagnostics.
///
/// `min_accepted_h_bits` is absent when no solver accepted a step.  It stores
/// the exact binary64 representation instead of a formatted tolerance value.
#[cfg(feature = "scalar-leg-observer")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ObservedFinalMetrics {
    pub solver_invocations: usize,
    pub encke_segments: usize,
    pub encke_rebases: usize,
    pub steps: usize,
    pub evals: usize,
    pub rejected_steps: usize,
    pub saturated_steps: usize,
    pub underflow_accepts: usize,
    pub cache_cluster_steps: usize,
    pub cache_cluster_steps_untruncated: usize,
    pub min_accepted_h_bits: Option<u64>,
    pub eclipse_ingress: usize,
    pub eclipse_egress: usize,
    /// `Some(0)` records a nonzero binary-eclipse leg in this direction with
    /// no committed crossing. `None` means direction did not apply.
    pub eclipse_forward_splits: Option<usize>,
    /// `Some(0)` records a nonzero binary-eclipse leg in this direction with
    /// no committed crossing. `None` means direction did not apply.
    pub eclipse_backward_splits: Option<usize>,
    pub eclipse_collapsed_pairs: usize,
}

/// One observed final-leg outcome. This type is only compiled for the
/// feature-gated diagnostic; canonical propagation keeps its existing return
/// type and makes no observer allocation or callback.
#[cfg(feature = "scalar-leg-observer")]
#[derive(Clone, Debug)]
pub struct ObservedFinalLeg {
    pub outcome: Result<[f64; 6], FinalPropagationFailure>,
    pub metrics: Result<ObservedFinalMetrics, ObservedFinalMetricError>,
    pub terminal_status: ObservedSolverTerminalStatus,
}

#[cfg(feature = "scalar-leg-observer")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FinalObservation {
    metrics: ObservedFinalMetrics,
    error: Option<ObservedFinalMetricError>,
    terminal_status: ObservedSolverTerminalStatus,
}

#[cfg(feature = "scalar-leg-observer")]
impl FinalObservation {
    pub(crate) const fn new() -> Self {
        Self {
            metrics: ObservedFinalMetrics {
                solver_invocations: 0,
                encke_segments: 0,
                encke_rebases: 0,
                steps: 0,
                evals: 0,
                rejected_steps: 0,
                saturated_steps: 0,
                underflow_accepts: 0,
                cache_cluster_steps: 0,
                cache_cluster_steps_untruncated: 0,
                min_accepted_h_bits: None,
                eclipse_ingress: 0,
                eclipse_egress: 0,
                eclipse_forward_splits: None,
                eclipse_backward_splits: None,
                eclipse_collapsed_pairs: 0,
            },
            error: None,
            terminal_status: ObservedSolverTerminalStatus::NotStarted,
        }
    }

    fn increment(counter: &mut usize) -> Result<(), ObservedFinalMetricError> {
        *counter = counter
            .checked_add(1)
            .ok_or(ObservedFinalMetricError::CounterOverflow)?;
        Ok(())
    }

    fn add(counter: &mut usize, value: usize) -> Result<(), ObservedFinalMetricError> {
        *counter = counter
            .checked_add(value)
            .ok_or(ObservedFinalMetricError::CounterOverflow)?;
        Ok(())
    }

    fn record(&mut self, result: Result<(), ObservedFinalMetricError>) {
        if self.error.is_none() {
            self.error = result.err();
        }
    }

    pub(crate) fn mark_incomplete(&mut self, error: ObservedFinalMetricError) {
        if self.terminal_status == ObservedSolverTerminalStatus::NotStarted {
            self.terminal_status = ObservedSolverTerminalStatus::Unavailable;
        }
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    pub(crate) fn record_completed_success(&mut self) {
        if self.terminal_status != ObservedSolverTerminalStatus::Unavailable {
            self.terminal_status = ObservedSolverTerminalStatus::Success;
        }
    }

    pub(crate) fn record_solver(
        &mut self,
        solver_stats: &IntegrationStats,
        terminal_status: OdeIntegrationStatus,
    ) {
        if self.terminal_status != ObservedSolverTerminalStatus::Unavailable {
            self.terminal_status = terminal_status.into();
        }
        let result = {
            let metrics = &mut self.metrics;
            (|| {
                Self::increment(&mut metrics.solver_invocations)?;
                Self::add(&mut metrics.steps, solver_stats.steps)?;
                Self::add(&mut metrics.evals, solver_stats.evals)?;
                Self::add(&mut metrics.rejected_steps, solver_stats.rejected_steps)?;
                Self::add(&mut metrics.saturated_steps, solver_stats.saturated_steps)?;
                Self::add(
                    &mut metrics.underflow_accepts,
                    solver_stats.underflow_accepts,
                )?;
                Self::add(
                    &mut metrics.cache_cluster_steps,
                    solver_stats.cache_cluster_steps,
                )?;
                Self::add(
                    &mut metrics.cache_cluster_steps_untruncated,
                    solver_stats.cache_cluster_steps_untruncated,
                )?;
                if solver_stats.min_accepted_h.is_finite() && solver_stats.min_accepted_h > 0.0 {
                    let candidate = solver_stats.min_accepted_h;
                    let prior = metrics.min_accepted_h_bits.map(f64::from_bits);
                    if prior.is_none_or(|value| candidate < value) {
                        metrics.min_accepted_h_bits = Some(candidate.to_bits());
                    }
                }
                Ok(())
            })()
        };
        self.record(result);
    }

    pub(crate) fn record_encke_segment(&mut self) {
        let result = Self::increment(&mut self.metrics.encke_segments);
        self.record(result);
    }

    pub(crate) fn record_encke_rebase(&mut self) {
        let result = Self::increment(&mut self.metrics.encke_rebases);
        self.record(result);
    }

    pub(crate) const fn record_eclipse_direction(&mut self, forward: bool) {
        let direction = if forward {
            &mut self.metrics.eclipse_forward_splits
        } else {
            &mut self.metrics.eclipse_backward_splits
        };
        *direction = Some(0);
    }

    pub(crate) fn record_eclipse_crossing(
        &mut self,
        forward: bool,
        old_side: EclipseSide,
        new_side: EclipseSide,
    ) {
        let (chronological_before, chronological_after) = if forward {
            (old_side, new_side)
        } else {
            (new_side, old_side)
        };
        let ingress = match (chronological_before, chronological_after) {
            (EclipseSide::Lit, EclipseSide::Shadow) => true,
            (EclipseSide::Shadow, EclipseSide::Lit) => false,
            _ => {
                self.mark_incomplete(ObservedFinalMetricError::EclipseMetricsUnavailable);
                return;
            }
        };
        let direction = if forward {
            &mut self.metrics.eclipse_forward_splits
        } else {
            &mut self.metrics.eclipse_backward_splits
        };
        let Some(direction) = direction.as_mut() else {
            self.mark_incomplete(ObservedFinalMetricError::EclipseMetricsUnavailable);
            return;
        };
        let result = if ingress {
            Self::increment(&mut self.metrics.eclipse_ingress)
        } else {
            Self::increment(&mut self.metrics.eclipse_egress)
        }
        .and_then(|()| Self::increment(direction));
        self.record(result);
    }

    pub(crate) fn record_eclipse_collapsed_pairs(&mut self, collapsed_pairs: usize) {
        let result = Self::add(&mut self.metrics.eclipse_collapsed_pairs, collapsed_pairs);
        self.record(result);
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        Result<ObservedFinalMetrics, ObservedFinalMetricError>,
        ObservedSolverTerminalStatus,
    ) {
        (
            self.error.map_or(Ok(self.metrics), Err),
            self.terminal_status,
        )
    }
}

impl std::fmt::Display for FinalPropagationFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gravity(error) => write!(formatter, "{error}"),
            Self::Eclipse(error) => write!(formatter, "{error}"),
            Self::Census(error) => write!(formatter, "{error}"),
            other => write!(formatter, "final propagation {other:?}"),
        }
    }
}

impl std::error::Error for FinalPropagationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gravity(error) => Some(error),
            Self::Eclipse(error) => Some(error),
            Self::Census(error) => Some(error),
            _ => None,
        }
    }
}

#[inline]
const fn final_failure_from_eclipse(error: EclipseError) -> FinalPropagationFailure {
    match error {
        EclipseError::Gravity(error) => FinalPropagationFailure::Gravity(error),
        error => FinalPropagationFailure::Eclipse(error),
    }
}

impl FinalPropagationFailure {
    /// Stable closed identifier for fixed-impact evidence serialization.
    ///
    /// Every nested typed failure remains distinct. This avoids parsing
    /// `Display`, allocating a formatted string, or exposing private modules to
    /// downstream evidence owners.
    #[must_use]
    pub const fn evidence_id(self) -> &'static str {
        match self {
            Self::Ground => "fixed-impact:integration:ground",
            Self::LeftEarth => "fixed-impact:integration:left-earth",
            Self::Eccentricity => "fixed-impact:integration:eccentricity",
            Self::NanState => "fixed-impact:integration:nan-state",
            Self::EventInvalid => "fixed-impact:integration:event-invalid",
            Self::Gravity(error) => final_gravity_evidence_id(error, false),
            Self::Census(error) => match error {
                PropagationCensusError::CounterOverflow => {
                    "fixed-impact:integration:census:counter-overflow"
                }
                PropagationCensusError::MutexPoisoned => {
                    "fixed-impact:integration:census:mutex-poisoned"
                }
                PropagationCensusError::Allocation => "fixed-impact:integration:census:allocation",
                PropagationCensusError::CollectionActive => {
                    "fixed-impact:integration:census:collection-active"
                }
            },
            Self::Eclipse(error) => match error {
                EclipseError::Gravity(error) => final_gravity_evidence_id(error, true),
                EclipseError::Authority(error) => final_authority_evidence_id(error),
                EclipseError::Geometry => "fixed-impact:integration:eclipse:geometry",
                EclipseError::UninitializedSide => {
                    "fixed-impact:integration:eclipse:uninitialized-side"
                }
                EclipseError::NonProgress => "fixed-impact:integration:eclipse:non-progress",
                EclipseError::Chatter => "fixed-impact:integration:eclipse:chatter",
                EclipseError::Bracket => "fixed-impact:integration:eclipse:bracket",
                EclipseError::EventOverlap => "fixed-impact:integration:eclipse:event-overlap",
                EclipseError::SplitLimit => "fixed-impact:integration:eclipse:split-limit",
                EclipseError::Envelope => "fixed-impact:integration:eclipse:envelope",
            },
            Self::IntegrationFailure => "fixed-impact:integration:failure",
            Self::MethodUnsupported => "fixed-impact:integration:method-unsupported",
        }
    }

    #[inline]
    const fn from_terminal_event(event_type: crate::types::EventType) -> Self {
        match event_type {
            crate::types::EventType::Ground => Self::Ground,
            crate::types::EventType::LeftEarth => Self::LeftEarth,
            crate::types::EventType::Eccentricity => Self::Eccentricity,
            crate::types::EventType::NanState => Self::NanState,
            crate::types::EventType::PerturbDeviation => Self::IntegrationFailure,
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_physical_infeasible(self) -> bool {
        matches!(self, Self::Ground | Self::LeftEarth | Self::Eccentricity)
    }

    /// True when the strict-HF enclosure REFUSED the configuration.
    ///
    /// Distinct from [`Self::is_physical_infeasible`] and from everything else:
    /// an authority refusal says the INPUTS are wrong, not that this trajectory
    /// is. Callers that collapse it into a generic integration failure lose the
    /// only fact worth reporting -- which is what happened until 2026-08-19,
    /// when a missing `r_obj_m` in the strict-HF carve-out surfaced as
    /// `MassSolveStatusCode::MissAtZeroHfIntegrateFailure` and cost a
    /// three-layer instrumentation pass to trace back.
    #[must_use]
    pub const fn is_authority_refusal(self) -> bool {
        matches!(self, Self::Eclipse(EclipseError::Authority(_)))
    }
}

const fn final_gravity_evidence_id(error: GravityError, eclipse: bool) -> &'static str {
    match (eclipse, error) {
        (false, GravityError::UnsupportedOrder) => {
            "fixed-impact:integration:gravity:unsupported-order"
        }
        (false, GravityError::InvalidCoefficientStorage) => {
            "fixed-impact:integration:gravity:invalid-coefficient-storage"
        }
        (false, GravityError::InvariantViolation) => {
            "fixed-impact:integration:gravity:invariant-violation"
        }
        (false, GravityError::InvalidState) => "fixed-impact:integration:gravity:invalid-state",
        (false, GravityError::InvalidTime) => "fixed-impact:integration:gravity:invalid-time",
        (false, GravityError::InvalidRotation) => {
            "fixed-impact:integration:gravity:invalid-rotation"
        }
        (false, GravityError::InvalidRadius) => "fixed-impact:integration:gravity:invalid-radius",
        (true, GravityError::UnsupportedOrder) => {
            "fixed-impact:integration:eclipse:gravity:unsupported-order"
        }
        (true, GravityError::InvalidCoefficientStorage) => {
            "fixed-impact:integration:eclipse:gravity:invalid-coefficient-storage"
        }
        (true, GravityError::InvariantViolation) => {
            "fixed-impact:integration:eclipse:gravity:invariant-violation"
        }
        (true, GravityError::InvalidState) => {
            "fixed-impact:integration:eclipse:gravity:invalid-state"
        }
        (true, GravityError::InvalidTime) => {
            "fixed-impact:integration:eclipse:gravity:invalid-time"
        }
        (true, GravityError::InvalidRotation) => {
            "fixed-impact:integration:eclipse:gravity:invalid-rotation"
        }
        (true, GravityError::InvalidRadius) => {
            "fixed-impact:integration:eclipse:gravity:invalid-radius"
        }
    }
}

const fn final_authority_evidence_id(
    error: crate::strict_hf_enclosure::StrictHfAuthorityError,
) -> &'static str {
    use crate::strict_hf_enclosure::{IdentityKind, StrictHfAuthorityError};

    match error {
        StrictHfAuthorityError::MissingAsset(kind) => match kind {
            IdentityKind::Epoch => "fixed-impact:integration:eclipse:authority:missing:epoch",
            IdentityKind::Force => "fixed-impact:integration:eclipse:authority:missing:force",
            IdentityKind::Science => "fixed-impact:integration:eclipse:authority:missing:science",
            IdentityKind::Gravity => "fixed-impact:integration:eclipse:authority:missing:gravity",
            IdentityKind::Ephemeris => {
                "fixed-impact:integration:eclipse:authority:missing:ephemeris"
            }
            IdentityKind::Atmosphere => {
                "fixed-impact:integration:eclipse:authority:missing:atmosphere"
            }
            IdentityKind::Frame => "fixed-impact:integration:eclipse:authority:missing:frame",
        },
        StrictHfAuthorityError::InvalidAsset(kind) => match kind {
            IdentityKind::Epoch => "fixed-impact:integration:eclipse:authority:invalid:epoch",
            IdentityKind::Force => "fixed-impact:integration:eclipse:authority:invalid:force",
            IdentityKind::Science => "fixed-impact:integration:eclipse:authority:invalid:science",
            IdentityKind::Gravity => "fixed-impact:integration:eclipse:authority:invalid:gravity",
            IdentityKind::Ephemeris => {
                "fixed-impact:integration:eclipse:authority:invalid:ephemeris"
            }
            IdentityKind::Atmosphere => {
                "fixed-impact:integration:eclipse:authority:invalid:atmosphere"
            }
            IdentityKind::Frame => "fixed-impact:integration:eclipse:authority:invalid:frame",
        },
        StrictHfAuthorityError::IdentityMismatch(kind) => match kind {
            IdentityKind::Epoch => "fixed-impact:integration:eclipse:authority:mismatch:epoch",
            IdentityKind::Force => "fixed-impact:integration:eclipse:authority:mismatch:force",
            IdentityKind::Science => "fixed-impact:integration:eclipse:authority:mismatch:science",
            IdentityKind::Gravity => "fixed-impact:integration:eclipse:authority:mismatch:gravity",
            IdentityKind::Ephemeris => {
                "fixed-impact:integration:eclipse:authority:mismatch:ephemeris"
            }
            IdentityKind::Atmosphere => {
                "fixed-impact:integration:eclipse:authority:mismatch:atmosphere"
            }
            IdentityKind::Frame => "fixed-impact:integration:eclipse:authority:mismatch:frame",
        },
    }
}

/// Whether `value` is one exact identifier minted by
/// [`FinalPropagationFailure::evidence_id`].
///
/// Validation is structural over closed typed segments. It allocates nothing
/// and accepts no formatted `Display` text or unknown suffix.
#[must_use]
pub fn is_final_propagation_failure_evidence_id(value: &str) -> bool {
    const PREFIX: &str = "fixed-impact:integration:";
    let Some(suffix) = value.strip_prefix(PREFIX) else {
        return false;
    };
    if matches!(
        suffix,
        "ground"
            | "left-earth"
            | "eccentricity"
            | "nan-state"
            | "event-invalid"
            | "failure"
            | "method-unsupported"
    ) {
        return true;
    }
    if let Some(gravity) = suffix.strip_prefix("gravity:") {
        return is_gravity_evidence_suffix(gravity);
    }
    if let Some(census) = suffix.strip_prefix("census:") {
        return matches!(
            census,
            "counter-overflow" | "mutex-poisoned" | "allocation" | "collection-active"
        );
    }
    if let Some(gravity) = suffix.strip_prefix("eclipse:gravity:") {
        return is_gravity_evidence_suffix(gravity);
    }
    if let Some(authority) = suffix.strip_prefix("eclipse:authority:") {
        let Some((failure, identity)) = authority.split_once(':') else {
            return false;
        };
        return matches!(failure, "missing" | "invalid" | "mismatch")
            && matches!(
                identity,
                "epoch" | "force" | "science" | "gravity" | "ephemeris" | "atmosphere" | "frame"
            );
    }
    matches!(
        suffix,
        "eclipse:geometry"
            | "eclipse:uninitialized-side"
            | "eclipse:non-progress"
            | "eclipse:chatter"
            | "eclipse:bracket"
            | "eclipse:event-overlap"
            | "eclipse:split-limit"
            | "eclipse:envelope"
    )
}

fn is_gravity_evidence_suffix(value: &str) -> bool {
    matches!(
        value,
        "unsupported-order"
            | "invalid-coefficient-storage"
            | "invariant-violation"
            | "invalid-state"
            | "invalid-time"
            | "invalid-rotation"
            | "invalid-radius"
    )
}

/// Return the typed terminal failure recorded by a sampled integration.
///
/// Sampled callers need this before inspecting partial rows, so transfer and
/// batch layers do not reconstruct authority failures from display strings.
#[must_use]
pub fn final_propagation_failure(result: &IntegrationResult) -> Option<FinalPropagationFailure> {
    if let Some(error) = result.terminal_gravity_error {
        return Some(FinalPropagationFailure::Gravity(error));
    }
    if let Some(error) = result.terminal_eclipse_error {
        return Some(final_failure_from_eclipse(error));
    }
    if result.terminal_event_fired {
        return Some(match result.terminal_event_name.as_ref() {
            "ground" => FinalPropagationFailure::Ground,
            "left_earth" => FinalPropagationFailure::LeftEarth,
            "eccentricity" => FinalPropagationFailure::Eccentricity,
            "nan_state" | "nan_encountered" | "non_finite_state" => {
                FinalPropagationFailure::NanState
            }
            "event_invalid" => FinalPropagationFailure::EventInvalid,
            _ => FinalPropagationFailure::IntegrationFailure,
        });
    }
    None
}

fn final_state_from_result(
    result: &IntegrationResult,
) -> Result<[f64; 6], FinalPropagationFailure> {
    if let Some(failure) = final_propagation_failure(result) {
        return Err(failure);
    }
    result
        .states
        .last()
        .copied()
        .ok_or(FinalPropagationFailure::IntegrationFailure)
}

/// Integrate only the final state and return a typed failure on invalid or terminal paths.
///
/// # Errors
///
/// Returns a typed propagation failure when validation, event handling, or integration fails.
pub fn integrate_final_checked(
    request: ScalarPropagationRequest<'_>,
) -> Result<[f64; 6], FinalPropagationFailure> {
    integrate_final_checked_core(request, FinalExecution::Fresh)
}

/// Run checked final path with fresh RHS built from caller request and fixed counters.
///
/// This feature-gated diagnostic accepts no extra RHS, solver, force model, or
/// asset override.
#[cfg(feature = "scalar-leg-observer")]
#[must_use]
pub fn integrate_final_checked_observed(request: ScalarPropagationRequest<'_>) -> ObservedFinalLeg {
    let mut observation = FinalObservation::new();
    let outcome =
        integrate_final_checked_core_observed(request, FinalExecution::Fresh, &mut observation);
    let (metrics, terminal_status) = observation.into_parts();
    ObservedFinalLeg {
        outcome,
        metrics,
        terminal_status,
    }
}

/// Which RHS instances a checked final propagation runs on: freshly built, or
/// reused from the caller (with a separate root RHS when eclipse is active).
enum FinalExecution<'a> {
    Fresh,
    ReusedNoEclipse(&'a mut LightyearRHS),
    ReusedEclipse {
        lane_rhs: &'a mut LightyearRHS,
        root_rhs: &'a mut LightyearRHS,
    },
}

#[derive(Clone, Copy)]
struct FinalRun<'a> {
    request: ScalarPropagationRequest<'a>,
    max_restarts: usize,
}

struct FinalSegment<'a> {
    context: &'a ScalarPropagationContext,
    current_equinoc: [f64; 6],
    current_t0_s: f64,
    t_final_s: f64,
    eps: f64,
    stepper: StepperMethod,
    max_rejects: usize,
}

/// Records the outcome of every checked propagation, then delegates.
///
/// A wrapper rather than a counter at each of the six exits, so a new failure
/// mode added later cannot escape the census by forgetting to increment.
fn integrate_final_checked_core(
    request: ScalarPropagationRequest<'_>,
    execution: FinalExecution<'_>,
) -> Result<[f64; 6], FinalPropagationFailure> {
    #[cfg(feature = "scalar-leg-observer")]
    let out = integrate_final_checked_inner(request, execution, None);
    #[cfg(not(feature = "scalar-leg-observer"))]
    let out = integrate_final_checked_inner(request, execution);
    crate::probe::observe_prop_return(
        out.as_ref()
            .is_ok_and(|state| state.iter().all(|v| v.is_finite())),
    );
    out
}

#[cfg(feature = "scalar-leg-observer")]
fn integrate_final_checked_core_observed(
    request: ScalarPropagationRequest<'_>,
    execution: FinalExecution<'_>,
    observation: &mut FinalObservation,
) -> Result<[f64; 6], FinalPropagationFailure> {
    let out = integrate_final_checked_inner(request, execution, Some(observation));
    let completed_finite = out
        .as_ref()
        .is_ok_and(|state| state.iter().all(|value| value.is_finite()));
    if completed_finite {
        observation.record_completed_success();
    }
    crate::probe::observe_prop_return(completed_finite);
    out
}

fn run_checked_final_segment(
    rhs: &LightyearRHS,
    segment: &FinalSegment<'_>,
    handler: Option<&mut EnckeEventHandler<'_>>,
    scratch: &mut SolverScratch,
) -> Result<OdeIntegrationResult, FinalPropagationFailure> {
    let context = segment.context;
    let current_equinoc = segment.current_equinoc;
    let current_t0_s = segment.current_t0_s;
    let t_final_s = segment.t_final_s;
    let eps = segment.eps.max(1e-12);
    let stepper = segment.stepper;
    let max_rejects = segment.max_rejects;
    let dt_max = context.config.dt_max;
    let system = LightyearSystem { rhs };
    let y0 = [0.0; 6];
    match stepper {
        StepperMethod::Dopri5Compat => {
            fn coerce_handler<'handler>(
                handler: &'handler mut EnckeEventHandler<'_>,
            ) -> &'handler mut dyn OdeEventHandler {
                handler
            }
            let event_handler = handler.map(coerce_handler);
            rhs.clear_gravity_error();
            let result = integrate_lightyear_dopri5_final(
                &system,
                &y0,
                current_t0_s,
                t_final_s,
                LightyearConfig {
                    eps,
                    dt_max,
                    max_steps: MAX_STEPS,
                    max_rejects,
                    force_eval: false,
                    fast_single: true,
                },
                event_handler,
            );
            rhs.take_gravity_error().map_or(Ok(result), |error| {
                Err(FinalPropagationFailure::Gravity(error))
            })
        }
        #[cfg(feature = "autodiff")]
        StepperMethod::Esdirk43 => {
            // Refused here, before the dual RHS and Jacobian are built: there is
            // no ESDIRK event driver, and the previous code fell through to
            // `integrate_final_with_events` with a hard-coded `OdeMethod::Tsit5`
            // after paying for dual state it then never used.
            if handler.is_some() {
                return Err(FinalPropagationFailure::MethodUnsupported);
            }
            let dual_rhs = crate::rhs_dual::LightyearDualRHS::new(
                current_equinoc,
                current_t0_s,
                context.jd0,
                Arc::clone(&context.config),
                Arc::clone(&context.gravity.packed),
            )
            .map_err(|_| FinalPropagationFailure::IntegrationFailure)?;
            let dual_rhs = Box::new(dual_rhs);
            let jac = crate::adaptive_solver::DualVecJacobian::new(&dual_rhs);
            let cfg = IntegratorConfig {
                error_control: ErrorControl::Absolute { eps },
                h0: None,
                h_min: 1e-12,
                h_max: dt_max,
                max_steps: MAX_STEPS,
                max_rejects,
                force_eval: false,
            };
            rhs.clear_gravity_error();
            dual_rhs.reset_gravity_error();
            let result = handler.map_or_else(
                || integrate_final_esdirk(&system, &jac, &y0, current_t0_s, t_final_s, cfg),
                |handler| {
                    integrate_final_with_events(
                        &system,
                        OdeMethod::Tsit5,
                        &y0,
                        current_t0_s,
                        t_final_s,
                        cfg,
                        handler,
                    )
                },
            );
            take_esdirk_gravity_error(rhs, &dual_rhs).map_or(Ok(result), |error| {
                Err(FinalPropagationFailure::Gravity(error))
            })
        }
        _ => {
            let cfg = IntegratorConfig {
                error_control: ErrorControl::Absolute { eps },
                h0: None,
                h_min: 1e-12,
                h_max: dt_max,
                max_steps: MAX_STEPS,
                max_rejects,
                force_eval: false,
            };
            let method = stepper_ode_method(stepper);
            rhs.clear_gravity_error();
            // `match`, not `map_or_else`: both arms need `&mut scratch`, and two
            // closures cannot hold unique access to it at once.
            let result = match handler {
                None => integrate_final_with_scratch(
                    &system,
                    method,
                    &y0,
                    current_t0_s,
                    t_final_s,
                    cfg,
                    scratch,
                ),
                Some(handler) => integrate_final_with_events_and_scratch(
                    &system,
                    method,
                    &y0,
                    current_t0_s,
                    t_final_s,
                    cfg,
                    handler,
                    scratch,
                ),
            };
            rhs.take_gravity_error().map_or(Ok(result), |error| {
                Err(FinalPropagationFailure::Gravity(error))
            })
        }
    }
}

fn integrate_checked_final_without_events(
    run: FinalRun<'_>,
    #[cfg(feature = "scalar-leg-observer")] mut observation: Option<&mut FinalObservation>,
) -> Result<[f64; 6], FinalPropagationFailure> {
    let request = run.request;
    let context = request.context;
    let init_equinoc_state = request.init_equinoc_state;
    let t0_s = request.t0_s;
    let t_final_s = request.t_final_s;
    let eps = context.config.eps;
    let stepper = request.stepper;
    let max_restarts = run.max_restarts;
    let threshold_sq = PERTURB_DEVIATION_THRESHOLD_KM * PERTURB_DEVIATION_THRESHOLD_KM;
    let original_equinoc = init_equinoc_state;
    let original_t0_s = t0_s;
    let mut current_equinoc = init_equinoc_state;
    let mut current_t0_s = t0_s;
    let mut had_restart = false;
    let mut rhs: Option<LightyearRHS> = None;
    for _ in 0..max_restarts {
        if had_restart && (t_final_s - current_t0_s).abs() < 1e-12 {
            let mut original_baseline = [0.0; 6];
            equinoc2eci_impl(
                &original_equinoc,
                6,
                t_final_s - original_t0_s,
                0.0,
                &mut original_baseline,
            );
            let mut current_eci = [0.0; 6];
            equinoc2eci_impl(&current_equinoc, 6, 0.0, 0.0, &mut current_eci);
            return Ok(subtract_state_vectors(&current_eci, &original_baseline));
        }
        let segment_tf = bounded_segment_end(current_t0_s, t_final_s, MAX_RECTIFICATION_SEGMENT_S);
        #[cfg(feature = "autodiff")]
        if matches!(stepper, StepperMethod::Esdirk43) {
            #[cfg(feature = "scalar-leg-observer")]
            if let Some(observation) = observation.as_deref_mut() {
                observation.mark_incomplete(ObservedFinalMetricError::EsdirkMetricsUnavailable);
            }
            let mut scalar_rhs = context
                .new_rhs(current_equinoc, current_t0_s)
                .map_err(|_| FinalPropagationFailure::IntegrationFailure)?;
            scalar_rhs.adapt_cache_policy_for_eps(eps);
            scalar_rhs.reset_for_propagation(current_equinoc, current_t0_s);
            let dual_rhs = crate::rhs_dual::LightyearDualRHS::new(
                current_equinoc,
                current_t0_s,
                context.jd0,
                Arc::clone(&context.config),
                Arc::clone(&context.gravity.packed),
            )
            .map_err(|_| FinalPropagationFailure::IntegrationFailure)?;
            let delta = integrate_final_no_events_esdirk(
                &scalar_rhs,
                &Box::new(dual_rhs),
                current_t0_s,
                segment_tf,
                eps,
                stepper,
            )?;
            if needs_rectification(
                position_norm_sq(&delta),
                threshold_sq,
                segment_tf,
                t_final_s,
            ) {
                current_equinoc =
                    rebase_equinoc_from_delta(&current_equinoc, current_t0_s, segment_tf, &delta);
                current_t0_s = segment_tf;
                had_restart = true;
                continue;
            }
            return if had_restart {
                Ok(correct_delta_to_original_baseline(
                    &delta,
                    t_final_s,
                    &current_equinoc,
                    current_t0_s,
                    &original_equinoc,
                    original_t0_s,
                ))
            } else {
                Ok(delta)
            };
        }
        if rhs.is_none() {
            rhs = Some(
                context
                    .new_rhs(current_equinoc, current_t0_s)
                    .map_err(|_| FinalPropagationFailure::IntegrationFailure)?,
            );
        }
        let scalar_rhs = rhs
            .as_mut()
            .ok_or(FinalPropagationFailure::IntegrationFailure)?;
        scalar_rhs.adapt_cache_policy_for_eps(eps);
        scalar_rhs.reset_for_propagation(current_equinoc, current_t0_s);
        #[cfg(feature = "scalar-leg-observer")]
        let delta = integrate_final_no_events_with_rhs(
            &*scalar_rhs,
            current_t0_s,
            segment_tf,
            eps,
            stepper,
            observation.as_deref_mut(),
        )?;
        #[cfg(not(feature = "scalar-leg-observer"))]
        let delta = integrate_final_no_events_with_rhs(
            &*scalar_rhs,
            current_t0_s,
            segment_tf,
            eps,
            stepper,
        )?;
        if needs_rectification(
            position_norm_sq(&delta),
            threshold_sq,
            segment_tf,
            t_final_s,
        ) {
            #[cfg(feature = "scalar-leg-observer")]
            if let Some(observation) = observation.as_deref_mut() {
                observation.record_encke_rebase();
            }
            current_equinoc =
                rebase_equinoc_from_delta(&current_equinoc, current_t0_s, segment_tf, &delta);
            current_t0_s = segment_tf;
            had_restart = true;
            continue;
        }
        return if had_restart {
            Ok(correct_delta_to_original_baseline(
                &delta,
                t_final_s,
                &current_equinoc,
                current_t0_s,
                &original_equinoc,
                original_t0_s,
            ))
        } else {
            Ok(delta)
        };
    }
    Err(FinalPropagationFailure::IntegrationFailure)
}

fn integrate_checked_final_with_events(
    run: FinalRun<'_>,
    mut reusable_rhs: Option<&mut LightyearRHS>,
    #[cfg(feature = "scalar-leg-observer")] mut observation: Option<&mut FinalObservation>,
) -> Result<[f64; 6], FinalPropagationFailure> {
    let request = run.request;
    let context = request.context;
    let init_equinoc_state = request.init_equinoc_state;
    let t0_s = request.t0_s;
    let t_final_s = request.t_final_s;
    let eps = context.config.eps;
    let stepper = request.stepper;
    let max_restarts = run.max_restarts;
    let original_equinoc = init_equinoc_state;
    let original_t0_s = t0_s;
    let mut current_equinoc = init_equinoc_state;
    let mut current_t0_s = t0_s;
    let mut owned_rhs = None;
    // One workspace for the whole rectification loop. Each pass through the
    // loop is one solver entry, and a long arc takes ~170 of them per
    // propagation; the solver used to allocate nine buffers on every one of
    // those entries and free them again on return. `SolverScratch::prepare`
    // zeroes what it reuses, so a segment cannot read a previous segment's
    // bytes.
    let mut scratch = SolverScratch::new();
    // Every pass below resets the baseline, so no pass can carry the previous
    // one's controller state; the only thing this distinguishes is the first
    // pass, which had no predecessor to carry from in the first place.
    let mut boundary = SegmentBoundary::ArcStart;
    for _ in 0..max_restarts {
        let rhs = if let Some(rhs) = reusable_rhs.as_deref_mut() {
            rhs
        } else {
            if owned_rhs.is_none() {
                owned_rhs = Some(
                    context
                        .new_rhs(current_equinoc, current_t0_s)
                        .map_err(|_| FinalPropagationFailure::IntegrationFailure)?,
                );
            }
            owned_rhs
                .as_mut()
                .ok_or(FinalPropagationFailure::IntegrationFailure)?
        };
        rhs.adapt_cache_policy_for_eps(eps);
        rhs.reset_for_propagation(current_equinoc, current_t0_s);
        let mut handler = Some(EnckeEventHandler::new(
            rhs.baseline_calculator(),
            current_t0_s,
            [0.0; 6],
            &*rhs,
            1e-6,
            eps,
            50,
        ));
        let event_handler = handler.as_mut();
        #[cfg(not(feature = "scalar-leg-observer"))]
        let result = run_checked_final_segment(
            &*rhs,
            &FinalSegment {
                context,
                current_equinoc,
                current_t0_s,
                t_final_s,
                eps,
                stepper,
                max_rejects: 50,
            },
            event_handler,
            &mut scratch,
        )?;
        #[cfg(feature = "scalar-leg-observer")]
        let result = match run_checked_final_segment(
            &*rhs,
            &FinalSegment {
                context,
                current_equinoc,
                current_t0_s,
                t_final_s,
                eps,
                stepper,
                max_rejects: 50,
            },
            event_handler,
            &mut scratch,
        ) {
            Ok(result) => result,
            Err(error) => {
                if let Some(observation) = observation.as_deref_mut() {
                    observation
                        .mark_incomplete(ObservedFinalMetricError::EventSegmentMetricsUnavailable);
                }
                return Err(error);
            }
        };
        crate::probe::bump_segment();
        crate::probe::bump_steps(result.stats.steps);
        crate::probe::bump_saturated(result.stats.saturated_steps);
        crate::probe::bump_rejected(result.stats.rejected_steps);
        crate::probe::observe_min_h(result.stats.min_accepted_h);
        crate::probe::observe_ramp(
            boundary,
            result.stats.segment_span_s,
            &result.stats.first_accepted_h,
            result.stats.tail_h_sum,
            result.stats.tail_h_count,
        );
        // `run_checked_final_segment` reads `context.config.dt_max` verbatim,
        // so the Encke rectification loop is unclamped by construction.
        crate::probe::observe_leg(
            context.config.dt_max,
            result.stats.segment_span_s,
            result.stats.steps,
            result.stats.evals,
            result.stats.rejected_steps,
        );
        crate::probe::observe_cache_cluster(
            result.stats.cache_cluster_steps,
            result.stats.cache_cluster_steps_untruncated,
        );
        crate::probe::observe_underflow(result.stats.underflow_accepts);
        #[cfg(feature = "scalar-leg-observer")]
        if let Some(observation) = observation.as_deref_mut() {
            observation.record_solver(&result.stats, result.status);
            observation.record_encke_segment();
        }

        let mut event_invalid = false;
        let detection = handler.as_mut().and_then(|handler| {
            if handler.take_event_invalid() {
                event_invalid = true;
            }
            handler.take_detection()
        });
        if event_invalid {
            return Err(FinalPropagationFailure::EventInvalid);
        }
        if let Some(detection) = detection {
            if detection.event_type == crate::types::EventType::PerturbDeviation {
                let event_time = detection.refined_time;
                if !event_time.is_finite() || (event_time - current_t0_s).abs() < 1e-9 {
                    return Err(FinalPropagationFailure::IntegrationFailure);
                }
                let mut baseline = [0.0; 6];
                satpy_core::equinoc2eci_impl(
                    &current_equinoc,
                    6,
                    event_time - current_t0_s,
                    0.0,
                    &mut baseline,
                );
                let eci = add_state_vectors(&baseline, &detection.state_at_event);
                eci2equinoc_impl_f64(&eci, 6, 0.0, 0.0, &mut current_equinoc);
                #[cfg(feature = "scalar-leg-observer")]
                if let Some(observation) = observation.as_deref_mut() {
                    observation.record_encke_rebase();
                }
                current_t0_s = event_time;
                boundary = SegmentBoundary::Rebased;
                continue;
            }
            return Err(FinalPropagationFailure::from_terminal_event(
                detection.event_type,
            ));
        }
        if !matches!(result.status, OdeIntegrationStatus::Success) {
            return Err(match result.status {
                OdeIntegrationStatus::NanEncountered | OdeIntegrationStatus::NonFiniteState => {
                    FinalPropagationFailure::NanState
                }
                OdeIntegrationStatus::EventInvalid => FinalPropagationFailure::EventInvalid,
                _ => FinalPropagationFailure::IntegrationFailure,
            });
        }
        let Some(delta) = result
            .y
            .get(..6)
            .and_then(|slice| <&[f64; 6]>::try_from(slice).ok())
        else {
            return Err(FinalPropagationFailure::IntegrationFailure);
        };
        return Ok(correct_delta_to_original_baseline(
            delta,
            t_final_s,
            &current_equinoc,
            current_t0_s,
            &original_equinoc,
            original_t0_s,
        ));
    }
    Err(FinalPropagationFailure::IntegrationFailure)
}

fn integrate_final_checked_inner(
    request: ScalarPropagationRequest<'_>,
    execution: FinalExecution<'_>,
    #[cfg(feature = "scalar-leg-observer")] mut observation: Option<&mut FinalObservation>,
) -> Result<[f64; 6], FinalPropagationFailure> {
    let context = request.context;
    let init_equinoc_state = request.init_equinoc_state;
    let t0_s = request.t0_s;
    let t_final_s = request.t_final_s;
    let eps = context.config.eps;
    let enable_events = request.enable_events;
    let stepper = request.stepper;
    if validate_scalar_stepper_authority(&context.config, "checked final integration").is_err()
        || crate::rhs::validate_atmosphere_model_code(context.config.atm_model).is_err()
        || (matches!(stepper, StepperMethod::Esdirk43)
            && crate::rhs_dual::validate_dual_newton_force_config(&context.config).is_err())
    {
        return Err(FinalPropagationFailure::IntegrationFailure);
    }
    // Resolve Auto to concrete method before any dispatch.
    let stepper = resolve_auto_stepper(stepper, eps);
    crate::probe::bump_propagation(t0_s, t_final_s).map_err(FinalPropagationFailure::Census)?;
    #[cfg(feature = "prop-census")]
    {
        // Both writers are infallible: a census store that fills latches the
        // census invalid and this propagation runs anyway. Instrumentation may
        // stop observing; it may not turn an arc into an infeasible design.
        crate::probe::record_state(
            &init_equinoc_state,
            t0_s,
            t_final_s,
            &crate::probe::ScienceKey {
                jd0: context.jd0,
                am_ratio: context.config.am_ratio,
                cd: context.config.cd,
                cr: context.config.cr,
                eps,
                dt_max: context.config.dt_max,
                sph_order: context.config.sph_order,
                force_flags: context.config.force_flags,
                atm_model: context.config.atm_model,
            },
        );
        crate::probe::capture_arc(crate::probe::CensusArc {
            tag: crate::probe::current_tag(),
            init_equinoc: &init_equinoc_state,
            jd0: context.jd0,
            start_time_s: t0_s,
            final_time_s: t_final_s,
            eps,
            sph_order: context.config.sph_order,
            force_flags: context.config.force_flags,
            atm_model: context.config.atm_model,
            am_ratio: context.config.am_ratio,
            cd: context.config.cd,
            cr: context.config.cr,
            dt_max: context.config.dt_max,
        });
    }
    if effective_scalar_srp(&context.config) {
        let coordinated = match execution {
            FinalExecution::ReusedEclipse { lane_rhs, root_rhs } => {
                #[cfg(feature = "scalar-leg-observer")]
                {
                    if let Some(observation) = observation.as_deref_mut() {
                        integrate_binary_eclipse_scalar_with_rhs_observed(
                            &BinaryEclipseRun {
                                init_equinoc_state,
                                t_eval: &[t_final_s],
                                t0_s,
                                tf_s: t_final_s,
                                enable_events,
                                eps,
                                stepper,
                            },
                            lane_rhs,
                            root_rhs,
                            observation,
                        )
                    } else {
                        integrate_binary_eclipse_scalar_with_rhs(
                            &BinaryEclipseRun {
                                init_equinoc_state,
                                t_eval: &[t_final_s],
                                t0_s,
                                tf_s: t_final_s,
                                enable_events,
                                eps,
                                stepper,
                            },
                            lane_rhs,
                            root_rhs,
                        )
                    }
                }
                #[cfg(not(feature = "scalar-leg-observer"))]
                integrate_binary_eclipse_scalar_with_rhs(
                    &BinaryEclipseRun {
                        init_equinoc_state,
                        t_eval: &[t_final_s],
                        t0_s,
                        tf_s: t_final_s,
                        enable_events,
                        eps,
                        stepper,
                    },
                    lane_rhs,
                    root_rhs,
                )
            }
            FinalExecution::Fresh => {
                #[cfg(feature = "scalar-leg-observer")]
                {
                    observation.as_deref_mut().map_or_else(
                        || {
                            integrate_binary_eclipse_scalar(
                                init_equinoc_state,
                                &[t_final_s],
                                t0_s,
                                t_final_s,
                                enable_events,
                                context.binary_eclipse_context(),
                            )
                        },
                        |observation| {
                            integrate_binary_eclipse_scalar_observed(
                                init_equinoc_state,
                                &[t_final_s],
                                t0_s,
                                t_final_s,
                                enable_events,
                                context.binary_eclipse_context(),
                                observation,
                            )
                        },
                    )
                }
                #[cfg(not(feature = "scalar-leg-observer"))]
                {
                    integrate_binary_eclipse_scalar(
                        init_equinoc_state,
                        &[t_final_s],
                        t0_s,
                        t_final_s,
                        enable_events,
                        context.binary_eclipse_context(),
                    )
                }
            }
            FinalExecution::ReusedNoEclipse(_) => Err(EclipseError::Geometry),
        }
        .map_err(final_failure_from_eclipse)?;
        return final_state_from_result(&coordinated);
    }
    let restart_step_s = context.config.dt_max.abs().max(1.0);
    let max_restarts = ((t_final_s - t0_s).abs() / restart_step_s)
        .ceil()
        .to_usize()
        .unwrap_or(usize::MAX)
        .saturating_add(128)
        .clamp(100, MAX_STEPS);
    if !enable_events {
        #[cfg(feature = "scalar-leg-observer")]
        {
            let run = FinalRun {
                request: request.with_resolved_stepper(stepper),
                max_restarts,
            };
            return integrate_checked_final_without_events(run, observation.as_deref_mut());
        }
        #[cfg(not(feature = "scalar-leg-observer"))]
        return integrate_checked_final_without_events(FinalRun {
            request: request.with_resolved_stepper(stepper),
            max_restarts,
        });
    }
    let reusable_rhs = match execution {
        FinalExecution::Fresh => None,
        FinalExecution::ReusedNoEclipse(rhs) => Some(rhs),
        FinalExecution::ReusedEclipse { .. } => {
            return Err(FinalPropagationFailure::IntegrationFailure)
        }
    };
    #[cfg(feature = "scalar-leg-observer")]
    {
        let run = FinalRun {
            request: request.with_resolved_stepper(stepper),
            max_restarts,
        };
        integrate_checked_final_with_events(run, reusable_rhs, observation)
    }
    #[cfg(not(feature = "scalar-leg-observer"))]
    integrate_checked_final_with_events(
        FinalRun {
            request: request.with_resolved_stepper(stepper),
            max_restarts,
        },
        reusable_rhs,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BodyInvariants, ForceFlags, MU};
    use satpy_core::{pack_gravity_coeffs, SEC_PER_DAY};

    #[test]
    fn final_propagation_failure_evidence_ids_bind_nested_failures() {
        use crate::strict_hf_enclosure::{IdentityKind, StrictHfAuthorityError};

        let cases = [
            (
                FinalPropagationFailure::Ground,
                "fixed-impact:integration:ground",
            ),
            (
                FinalPropagationFailure::Gravity(GravityError::InvalidRotation),
                "fixed-impact:integration:gravity:invalid-rotation",
            ),
            (
                FinalPropagationFailure::Census(PropagationCensusError::Allocation),
                "fixed-impact:integration:census:allocation",
            ),
            (
                FinalPropagationFailure::Eclipse(EclipseError::Gravity(GravityError::InvalidTime)),
                "fixed-impact:integration:eclipse:gravity:invalid-time",
            ),
            (
                FinalPropagationFailure::Eclipse(EclipseError::Authority(
                    StrictHfAuthorityError::IdentityMismatch(IdentityKind::Science),
                )),
                "fixed-impact:integration:eclipse:authority:mismatch:science",
            ),
            (
                FinalPropagationFailure::Eclipse(EclipseError::SplitLimit),
                "fixed-impact:integration:eclipse:split-limit",
            ),
            (
                FinalPropagationFailure::MethodUnsupported,
                "fixed-impact:integration:method-unsupported",
            ),
        ];

        for (failure, expected) in cases {
            let id: &'static str = failure.evidence_id();
            assert_eq!(id, expected);
            assert!(is_final_propagation_failure_evidence_id(id));
        }
        for hostile in [
            "integration:ground",
            "fixed-impact:integration",
            "fixed-impact:integration:gravity",
            "fixed-impact:integration:gravity:unknown",
            "fixed-impact:integration:eclipse:authority:mismatch",
            "fixed-impact:integration:eclipse:authority:mismatch:science:extra",
            "fixed-impact:integration:eclipse:unknown",
        ] {
            assert!(!is_final_propagation_failure_evidence_id(hostile));
        }
    }

    fn test_usize_as_f64(value: usize, context: &str) -> f64 {
        f64::from(u32::try_from(value).expect(context))
    }

    fn test_coefficients(order: usize) -> Arc<PackedGravityCoeffs> {
        let stride = order
            .checked_add(2)
            .expect("test coefficient stride must not overflow");
        let total = stride
            .checked_mul(stride)
            .expect("test coefficient array length must not overflow");
        let mut c = vec![0.0; total];
        let mut s = vec![0.0; total];
        *c.get_mut(0)
            .expect("test coefficient array must contain C[0,0]") = 1.0;
        for l in 2..=order {
            let base = l
                .checked_mul(stride)
                .expect("test coefficient row offset must not overflow");
            *c.get_mut(base)
                .expect("test coefficient array must contain degree term") =
                1e-3 / test_usize_as_f64(l, "test degree must fit u32").powi(2);
            for m in 1..=l {
                let degree_product = l
                    .checked_mul(m)
                    .expect("test degree product must not overflow");
                let mag =
                    1e-6 / test_usize_as_f64(degree_product, "test degree product must fit u32");
                let coefficient_index = base
                    .checked_add(m)
                    .expect("test coefficient index must not overflow");
                *c.get_mut(coefficient_index)
                    .expect("test coefficient array must contain cosine term") = mag;
                *s.get_mut(coefficient_index)
                    .expect("test coefficient array must contain sine term") = -0.5 * mag;
            }
        }
        let packed = pack_gravity_coeffs(&c, &s, stride, order)
            .expect("test gravity coefficients must pack");
        Arc::new(packed)
    }

    fn j2_only_coefficients(order: usize) -> Arc<PackedGravityCoeffs> {
        let stride = order
            .checked_add(2)
            .expect("test coefficient stride must not overflow");
        let total = stride
            .checked_mul(stride)
            .expect("test coefficient array length must not overflow");
        let mut c = vec![0.0; total];
        let s = vec![0.0; total];
        *c.get_mut(0)
            .expect("test coefficient array must contain C[0,0]") = 1.0;
        let j2_index = 2_usize
            .checked_mul(stride)
            .expect("test J2 coefficient index must not overflow");
        *c.get_mut(j2_index)
            .expect("test coefficient array must contain C[2,0]") = -1.082_63e-3;
        let packed = pack_gravity_coeffs(&c, &s, stride, order)
            .expect("test J2 gravity coefficients must pack");
        Arc::new(packed)
    }

    fn event_enabled_j2_final_checked(
        init_eci: [f64; 6],
        tf_s: f64,
    ) -> Result<[f64; 6], FinalPropagationFailure> {
        let packed = j2_only_coefficients(5);
        let config = Arc::new(ForceConfig {
            sph_order: 5,
            force_flags: 0,
            subtract_first_order: true,
            dt_max: 60.0,
            eps: 1.0e-8,
            integrator_method: StepperMethod::Vern9,
            ..ForceConfig::default()
        });
        let mut init = [0.0; 6];
        eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init);
        let gravity = ScalarGravityAssets::new(packed);
        let context = ScalarPropagationContext::new(2_460_310.5, config, gravity);
        integrate_final_checked(
            ScalarPropagationRequest::new(&context, init, &[tf_s], 0.0, tf_s).with_events(true),
        )
    }

    fn event_enabled_j2_final(init_eci: [f64; 6], tf_s: f64) -> Option<[f64; 6]> {
        event_enabled_j2_final_checked(init_eci, tf_s).ok()
    }

    fn assert_state_close(a: [f64; 6], b: [f64; 6], tol: f64) {
        for (index, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "component {index} differs: lhs={lhs} rhs={rhs} diff={diff} tol={tol}"
            );
        }
    }

    #[test]
    fn slice_to_state_short_input_returns_nan_state() {
        let state = slice_to_state(&[1.0; 5]);

        assert!(state.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn event_enabled_order5_circular_orbit_survives_7200_seconds() {
        let radius_km = 7000.0;
        let speed_km_s = (MU / radius_km).sqrt();
        let result = event_enabled_j2_final([radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0], 7200.0)
            .expect("safe circular orbit must not fire terminal event");

        assert!(result.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn event_enabled_endpoint_closed_orbit_rejects_hidden_ground_crossing() {
        let period_s = 4920.0;
        let semi_major_km = (MU * (period_s / (2.0 * std::f64::consts::PI)).powi(2)).cbrt();
        let apogee_km = 7000.0;
        let twice_semi_major_km = 2.0 * semi_major_km;
        let perigee_km = twice_semi_major_km - apogee_km;
        let apogee_speed_km_s = (MU * (2.0 / apogee_km - 1.0 / semi_major_km)).sqrt();
        assert!((perigee_km - 5504.49).abs() < 0.1);

        let result = event_enabled_j2_final_checked(
            [apogee_km, 0.0, 0.0, 0.0, apogee_speed_km_s, 0.0],
            period_s,
        );

        assert_eq!(
            result,
            Err(FinalPropagationFailure::Ground),
            "continuous ground event must type endpoint-closed impact orbit"
        );
    }

    #[test]
    fn sampled_result_reports_exact_eclipse_failure() {
        let result = IntegrationResult {
            terminal_event_fired: true,
            terminal_event_name: Cow::Borrowed("eclipse_envelope"),
            terminal_eclipse_error: Some(EclipseError::Envelope),
            ..IntegrationResult::default()
        };
        assert_eq!(
            final_propagation_failure(&result),
            Some(FinalPropagationFailure::Eclipse(EclipseError::Envelope))
        );
    }

    #[cfg(feature = "scalar-leg-observer")]
    #[test]
    fn observed_final_without_srp_matches_canonical_bits_and_reports_local_stats() {
        let config = Arc::new(ForceConfig {
            sph_order: 0,
            force_flags: 0,
            subtract_first_order: true,
            dt_max: 60.0,
            eps: 1.0e-8,
            integrator_method: StepperMethod::Vern9,
            ..ForceConfig::default()
        });
        let context = ScalarPropagationContext::new(
            2_460_310.5,
            config,
            ScalarGravityAssets::new(test_coefficients(0)),
        );
        let state = [7_050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let times = [600.0];
        let request = ScalarPropagationRequest::new(&context, state, &times, 0.0, 600.0);
        let canonical = integrate_final_checked(request).expect("canonical final propagation");

        let observed = integrate_final_checked_observed(request);

        let observed_state = observed.outcome.expect("observed final propagation");
        assert_eq!(
            observed_state.map(f64::to_bits),
            canonical.map(f64::to_bits)
        );
        let metrics = observed.metrics.expect("metric accounting");
        assert_eq!(metrics.solver_invocations, 1);
        assert_eq!(metrics.encke_segments, 1);
        assert_eq!(metrics.encke_rebases, 0);
        assert!(metrics.steps > 0);
        assert!(metrics.evals > 0);
        assert!(metrics.min_accepted_h_bits.is_some());
        assert_eq!(
            observed.terminal_status,
            ObservedSolverTerminalStatus::Success
        );
    }

    #[cfg(feature = "scalar-leg-observer")]
    #[test]
    fn observed_counter_overflow_preserves_last_solver_status() {
        let mut observation = FinalObservation::new();
        observation.metrics.solver_invocations = usize::MAX;

        observation.record_solver(
            &crate::odesolve::IntegrationStats::default(),
            OdeIntegrationStatus::MaxRejectsExceeded,
        );

        let (metrics, terminal_status) = observation.into_parts();
        assert_eq!(metrics, Err(ObservedFinalMetricError::CounterOverflow));
        assert_eq!(
            terminal_status,
            ObservedSolverTerminalStatus::MaxRejectsExceeded
        );
    }

    #[cfg(feature = "scalar-leg-observer")]
    #[test]
    fn observed_incomplete_mark_preserves_last_solver_status() {
        let mut observation = FinalObservation::new();
        observation.record_solver(
            &crate::odesolve::IntegrationStats::default(),
            OdeIntegrationStatus::Success,
        );

        observation.mark_incomplete(ObservedFinalMetricError::EventSegmentMetricsUnavailable);

        let (metrics, terminal_status) = observation.into_parts();
        assert_eq!(
            metrics,
            Err(ObservedFinalMetricError::EventSegmentMetricsUnavailable)
        );
        assert_eq!(terminal_status, ObservedSolverTerminalStatus::Success);
    }

    #[cfg(feature = "scalar-leg-observer")]
    #[test]
    fn observed_incomplete_terminal_status_latches_after_later_solver() {
        let mut observation = FinalObservation::new();
        observation.mark_incomplete(ObservedFinalMetricError::EclipseMetricsUnavailable);

        observation.record_solver(
            &crate::odesolve::IntegrationStats::default(),
            OdeIntegrationStatus::Success,
        );

        let (metrics, terminal_status) = observation.into_parts();
        assert_eq!(
            metrics,
            Err(ObservedFinalMetricError::EclipseMetricsUnavailable)
        );
        assert_eq!(terminal_status, ObservedSolverTerminalStatus::Unavailable);
    }

    #[cfg(feature = "scalar-leg-observer")]
    #[test]
    fn observed_completed_leg_normalizes_internal_event_terminal_status() {
        let mut observation = FinalObservation::new();
        observation.record_solver(
            &crate::odesolve::IntegrationStats::default(),
            OdeIntegrationStatus::EventTriggered,
        );

        observation.record_completed_success();

        let (metrics, terminal_status) = observation.into_parts();
        assert_eq!(metrics.unwrap_or_default().solver_invocations, 1);
        assert_eq!(terminal_status, ObservedSolverTerminalStatus::Success);
    }

    #[cfg(feature = "scalar-leg-observer")]
    #[test]
    fn observed_completed_leg_does_not_hide_unavailable_metrics() {
        let mut observation = FinalObservation::new();
        observation.mark_incomplete(ObservedFinalMetricError::EclipseMetricsUnavailable);

        observation.record_completed_success();

        let (metrics, terminal_status) = observation.into_parts();
        assert_eq!(
            metrics,
            Err(ObservedFinalMetricError::EclipseMetricsUnavailable)
        );
        assert_eq!(terminal_status, ObservedSolverTerminalStatus::Unavailable);
    }

    #[cfg(feature = "scalar-leg-observer")]
    #[test]
    fn observed_event_solver_boundary_error_never_reports_complete_metrics() {
        let context = invalid_time_gravity_context(StepperMethod::Vern9);
        let observed = integrate_final_checked_observed(
            ScalarPropagationRequest::new(
                &context,
                [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
                &[60.0],
                0.0,
                60.0,
            )
            .with_events(true),
        );

        assert_eq!(
            observed.outcome,
            Err(FinalPropagationFailure::Gravity(
                satpy_core::GravityError::InvalidTime
            ))
        );
        assert_eq!(
            observed.metrics,
            Err(ObservedFinalMetricError::EventSegmentMetricsUnavailable)
        );
        assert_eq!(
            observed.terminal_status,
            ObservedSolverTerminalStatus::Unavailable
        );
    }

    #[cfg(feature = "scalar-leg-observer")]
    #[test]
    fn observed_binary_eclipse_forward_matches_canonical_bits_and_reports_complete_metrics() {
        let (state, jd0, tf_s, config, packed) = convergence_fixture();
        let context = ScalarPropagationContext::new(
            jd0,
            config,
            ScalarGravityAssets::new(Arc::clone(&packed)),
        );
        let times = [tf_s];
        let request = ScalarPropagationRequest::new(&context, state, &times, 0.0, tf_s);
        let canonical = integrate_final_checked(request).expect("canonical binary eclipse");
        let coordinated = integrate_binary_eclipse_scalar(
            state,
            &times,
            0.0,
            tf_s,
            false,
            context.binary_eclipse_context(),
        )
        .expect("coordinated binary eclipse");

        let observed = integrate_final_checked_observed(request);

        let observed_state = observed.outcome.expect("observed binary eclipse");
        assert_eq!(
            observed_state.map(f64::to_bits),
            canonical.map(f64::to_bits)
        );
        let metrics = observed.metrics.expect("complete eclipse accounting");
        assert!(metrics.solver_invocations > 1);
        assert!(metrics.encke_segments > 0);
        assert!(metrics.solver_invocations > metrics.encke_segments);
        assert!(metrics.encke_rebases >= 2);
        assert_eq!(metrics.steps, coordinated.metrics.total_steps);
        assert_eq!(metrics.evals, coordinated.metrics.total_evals);
        assert!(metrics.eclipse_ingress > 0);
        assert!(metrics.eclipse_egress > 0);
        assert_eq!(
            metrics.eclipse_forward_splits,
            Some(metrics.eclipse_ingress + metrics.eclipse_egress)
        );
        assert!(metrics
            .eclipse_forward_splits
            .is_some_and(|splits| splits >= 2));
        let committed = metrics
            .eclipse_forward_splits
            .expect("forward eclipse leg must report committed roots");
        let root_solver_invocations = committed
            .checked_mul(3)
            .expect("root solver invocation count overflow");
        let expected_solver_invocations = metrics
            .encke_segments
            .checked_add(root_solver_invocations)
            .expect("total solver invocation count overflow");
        assert_eq!(
            metrics.solver_invocations, expected_solver_invocations,
            "each committed root must use one refine, proof, and window solve"
        );
        assert_eq!(metrics.eclipse_backward_splits, None);
        assert_eq!(
            metrics.eclipse_collapsed_pairs,
            coordinated.metrics.eclipse_collapsed_pairs
        );
        assert_eq!(
            observed.terminal_status,
            ObservedSolverTerminalStatus::Success
        );
    }

    #[cfg(feature = "scalar-leg-observer")]
    #[test]
    fn observed_binary_eclipse_backward_matches_canonical_bits_and_counts_chronology() {
        let (state, jd0, tf_s, config, packed) = convergence_fixture();
        let context = ScalarPropagationContext::new(
            jd0,
            config,
            ScalarGravityAssets::new(Arc::clone(&packed)),
        );
        let times = [tf_s];
        let forward = integrate_final_checked(ScalarPropagationRequest::new(
            &context, state, &times, 0.0, tf_s,
        ))
        .expect("forward binary eclipse");
        let mut baseline = [0.0; 6];
        equinoc2eci_impl(&state, 6, tf_s, 0.0, &mut baseline);
        let final_eci = add_state_vectors(&baseline, &forward);
        let mut backward_state = [0.0; 6];
        eci2equinoc_impl_f64(&final_eci, 6, 0.0, 0.0, &mut backward_state);
        let backward_times = [0.0];
        let request =
            ScalarPropagationRequest::new(&context, backward_state, &backward_times, tf_s, 0.0);
        let canonical = integrate_final_checked(request).expect("canonical backward eclipse");
        let coordinated = integrate_binary_eclipse_scalar(
            backward_state,
            &backward_times,
            tf_s,
            0.0,
            false,
            context.binary_eclipse_context(),
        )
        .expect("coordinated backward eclipse");

        let observed = integrate_final_checked_observed(request);
        let observed_state = observed.outcome.expect("observed backward eclipse");
        assert_eq!(
            observed_state.map(f64::to_bits),
            canonical.map(f64::to_bits)
        );
        let metrics = observed
            .metrics
            .expect("complete backward eclipse accounting");
        assert!(metrics.solver_invocations > 1);
        assert!(metrics.encke_segments > 0);
        assert!(metrics.solver_invocations > metrics.encke_segments);
        assert!(metrics.encke_rebases >= 2);
        assert_eq!(metrics.steps, coordinated.metrics.total_steps);
        assert_eq!(metrics.evals, coordinated.metrics.total_evals);
        assert!(metrics.eclipse_ingress > 0);
        assert!(metrics.eclipse_egress > 0);
        assert_eq!(metrics.eclipse_forward_splits, None);
        assert_eq!(
            metrics.eclipse_backward_splits,
            Some(metrics.eclipse_ingress + metrics.eclipse_egress)
        );
        assert!(metrics
            .eclipse_backward_splits
            .is_some_and(|splits| splits >= 2));
        assert_eq!(
            metrics.eclipse_collapsed_pairs,
            coordinated.metrics.eclipse_collapsed_pairs
        );
        assert_eq!(
            observed.terminal_status,
            ObservedSolverTerminalStatus::Success
        );
    }

    #[cfg(feature = "scalar-leg-observer")]
    #[test]
    fn observed_binary_eclipse_hidden_double_fixture_keeps_full_local_accounting() {
        let radius_km = 7_000.0;
        let speed_km_s = (MU / radius_km).sqrt();
        let period_s = 2.0 * std::f64::consts::PI * (radius_km.powi(3) / MU).sqrt();
        let normal_x = (6_378.137 - 100.0) / radius_km;
        let normal_z = (1.0 - normal_x * normal_x).sqrt();
        let mut state = [0.0; 6];
        eci2equinoc_impl_f64(
            &[
                radius_km * normal_z,
                0.0,
                -radius_km * normal_x,
                0.0,
                speed_km_s,
                0.0,
            ],
            6,
            0.0,
            0.0,
            &mut state,
        );
        let context = ScalarPropagationContext::new(
            2_460_310.5,
            Arc::new(ForceConfig {
                sph_order: 0,
                force_flags: ForceFlags::SRP,
                am_ratio: 1.0e-12,
                cr: 1.0,
                sun_pos: Some([149_597_870.7, 0.0, 0.0]),
                dt_max: 900.0,
                eps: 1.0e-6,
                integrator_method: StepperMethod::Vern9,
                ..ForceConfig::default()
            }),
            ScalarGravityAssets::new(test_coefficients(0)),
        );
        let times = [period_s];
        let request = ScalarPropagationRequest::new(&context, state, &times, 0.0, period_s);
        let canonical = integrate_final_checked(request).expect("canonical hidden-double eclipse");
        let coordinated = integrate_binary_eclipse_scalar(
            state,
            &times,
            0.0,
            period_s,
            false,
            context.binary_eclipse_context(),
        )
        .expect("coordinated hidden-double eclipse");

        let observed = integrate_final_checked_observed(request);

        assert_eq!(
            observed
                .outcome
                .expect("observed hidden-double eclipse")
                .map(f64::to_bits),
            canonical.map(f64::to_bits)
        );
        let metrics = observed.metrics.expect("complete hidden-double accounting");
        assert!(metrics.solver_invocations > metrics.encke_segments);
        assert_eq!(metrics.steps, coordinated.metrics.total_steps);
        assert_eq!(metrics.evals, coordinated.metrics.total_evals);
        assert_eq!(
            metrics.eclipse_forward_splits,
            Some(metrics.eclipse_ingress + metrics.eclipse_egress)
        );
        assert!(metrics
            .eclipse_forward_splits
            .is_some_and(|splits| splits >= 2));
        assert_eq!(metrics.eclipse_backward_splits, None);
        assert_eq!(
            metrics.eclipse_collapsed_pairs,
            coordinated.metrics.eclipse_collapsed_pairs
        );
        assert_eq!(
            observed.terminal_status,
            ObservedSolverTerminalStatus::Success
        );
    }

    #[cfg(feature = "scalar-leg-observer")]
    #[test]
    fn observed_binary_eclipse_failure_preserves_physics_and_fails_metrics_closed() {
        let (_, jd0, _, config, packed) = convergence_fixture();
        let outside_eci = [60_000.0, 0.0, 0.0, 0.0, 2.5, 0.0];
        let mut outside = [0.0; 6];
        eci2equinoc_impl_f64(&outside_eci, 6, 0.0, 0.0, &mut outside);
        let context = ScalarPropagationContext::new(jd0, config, ScalarGravityAssets::new(packed));
        let times = [600.0];
        let request = ScalarPropagationRequest::new(&context, outside, &times, 0.0, 600.0);
        let canonical = integrate_final_checked(request);

        let observed = integrate_final_checked_observed(request);

        assert_eq!(
            canonical,
            Err(FinalPropagationFailure::Eclipse(EclipseError::Envelope))
        );
        assert_eq!(observed.outcome, canonical);
        assert_eq!(
            observed.metrics,
            Err(ObservedFinalMetricError::EclipseMetricsUnavailable)
        );
        assert_eq!(
            observed.terminal_status,
            ObservedSolverTerminalStatus::Unavailable
        );
    }

    #[cfg(all(feature = "scalar-leg-observer", feature = "autodiff"))]
    #[test]
    fn observed_esdirk_never_reports_complete_metrics_before_solver_support() {
        let context = invalid_time_gravity_context(StepperMethod::Esdirk43);
        let observed = integrate_final_checked_observed(ScalarPropagationRequest::new(
            &context,
            [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
            &[60.0],
            0.0,
            60.0,
        ));

        assert!(matches!(
            observed.outcome,
            Err(FinalPropagationFailure::Gravity(
                satpy_core::GravityError::InvalidTime
            ))
        ));
        assert_eq!(
            observed.metrics,
            Err(ObservedFinalMetricError::EsdirkMetricsUnavailable)
        );
        assert_eq!(
            observed.terminal_status,
            ObservedSolverTerminalStatus::Unavailable
        );
    }

    #[test]
    fn event_enabled_order5_circular_orbit_survives_48_hours() {
        let radius_km = 7000.0;
        let speed_km_s = (MU / radius_km).sqrt();
        let result =
            event_enabled_j2_final([radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0], 48.0 * 3600.0)
                .expect("duration-scaled restart cap must support 48-hour safe orbit");

        assert!(result.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn reusable_final_checked_sequence_matches_fresh_bits_and_resets() {
        let packed = test_coefficients(5);
        let config = Arc::new(ForceConfig {
            sph_order: 5,
            force_flags: 0,
            subtract_first_order: true,
            dt_max: 60.0,
            eps: 1.0e-8,
            integrator_method: StepperMethod::Vern9,
            ..ForceConfig::default()
        });
        let jd0 = 2_460_310.5;
        let start_time_s = 0.0;
        let final_time_s = 1_800.0;
        let states = [
            [7_050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
            [7_130.0, -0.0015, 0.0012, -0.0007, 0.0006, 1.1],
            [7_050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
        ];
        let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
        let fresh_context = ScalarPropagationContext::new(jd0, Arc::clone(&config), gravity);
        let fresh = states.map(|state| {
            integrate_final_checked(
                ScalarPropagationRequest::new(
                    &fresh_context,
                    state,
                    &[final_time_s],
                    start_time_s,
                    final_time_s,
                )
                .with_events(true),
            )
            .expect("fresh checked propagation")
        });

        let gravity = ScalarGravityAssets::new(packed);
        let reusable_context = ScalarPropagationContext::new(jd0, config, gravity);
        let mut reusable = ReusableFinalCheckedIntegrator::new(reusable_context)
            .expect("reusable checked integrator");
        let reused = states.map(|state| {
            reusable
                .propagate_checked(state, start_time_s, final_time_s)
                .expect("reused checked propagation")
        });

        assert_eq!(
            reused.map(|state| state.map(f64::to_bits)),
            fresh.map(|state| state.map(f64::to_bits))
        );
        assert_eq!(reused[0].map(f64::to_bits), reused[2].map(f64::to_bits));
        let reuse_stats = reusable.stats();
        assert_eq!(reuse_stats.propagations, 3);
        assert_eq!(reuse_stats.rhs_construct_count, 1);
        assert_eq!(reuse_stats.rhs_reuse_hits, 2);
    }

    #[test]
    fn reusable_final_checked_constructor_rejects_jb2008_coulomb_without_panic() {
        let packed = test_coefficients(5);
        let config = Arc::new(ForceConfig {
            sph_order: 5,
            force_flags: crate::types::ForceFlags::DRAG | crate::types::ForceFlags::COULOMB_DRAG,
            atm_model: 4,
            eps: 1.0e-8,
            integrator_method: StepperMethod::Vern9,
            ..ForceConfig::default()
        });

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let gravity = ScalarGravityAssets::new(packed);
            let context = ScalarPropagationContext::new(2_460_310.5, config, gravity);
            ReusableFinalCheckedIntegrator::new(context)
        }));

        assert!(result.is_ok(), "fallible constructor must not panic");
        assert!(matches!(
            result.expect("panic checked"),
            Err(FinalPropagationFailure::IntegrationFailure)
        ));
    }

    #[test]
    fn scalar_gravity_assets_reject_requested_order_above_packed_authority() {
        let packed = test_coefficients(0);
        let config = Arc::new(ForceConfig {
            sph_order: 1,
            integrator_method: StepperMethod::Vern9,
            ..ForceConfig::default()
        });
        let gravity = ScalarGravityAssets::new(packed);
        let context = ScalarPropagationContext::new(2_460_310.5, config, gravity);

        assert!(
            ReusableFinalNoEventIntegrator::new(context).is_err(),
            "requested spherical-harmonic order above packed authority must fail closed"
        );
    }

    fn invalid_time_gravity_context(stepper: StepperMethod) -> ScalarPropagationContext {
        let config = Arc::new(ForceConfig {
            sph_order: 1,
            force_flags: 0,
            dt_max: 60.0,
            eps: 1.0e-8,
            integrator_method: stepper,
            ..ForceConfig::default()
        });
        ScalarPropagationContext::new(
            f64::NAN,
            config,
            ScalarGravityAssets::new(test_coefficients(1)),
        )
    }

    fn assert_invalid_time_gravity_result(result: &IntegrationResult) {
        assert_eq!(
            result.terminal_gravity_error,
            Some(satpy_core::GravityError::InvalidTime),
            "sampled propagation must preserve the evaluator's exact error: {result:?}"
        );
        assert_eq!(
            final_propagation_failure(result),
            Some(FinalPropagationFailure::Gravity(
                satpy_core::GravityError::InvalidTime
            )),
            "typed gravity failure must take priority over solver terminal status"
        );
    }

    #[test]
    fn sampled_no_event_gravity_error_precedes_solver_nonfinite_status() {
        assert_eq!(IntegrationResult::default().terminal_gravity_error, None);
        let context = invalid_time_gravity_context(StepperMethod::Vern9);
        let result = integrate_adaptive(
            ScalarPropagationRequest::new(
                &context,
                [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
                &[0.0, 60.0],
                0.0,
                60.0,
            )
            .with_events(false),
        )
        .expect("gravity failure must not exhaust propagation census");
        assert_invalid_time_gravity_result(&result);
    }

    #[test]
    fn sampled_event_gravity_error_precedes_solver_nonfinite_status() {
        let context = invalid_time_gravity_context(StepperMethod::Vern9);
        let result = integrate_adaptive(
            ScalarPropagationRequest::new(
                &context,
                [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
                &[0.0, 60.0],
                0.0,
                60.0,
            )
            .with_events(true),
        )
        .expect("gravity failure must not exhaust propagation census");
        assert_invalid_time_gravity_result(&result);
    }

    #[test]
    fn final_no_event_gravity_error_precedes_solver_nonfinite_status() {
        let context = invalid_time_gravity_context(StepperMethod::Vern9);
        let result = integrate_final_checked(
            ScalarPropagationRequest::new(
                &context,
                [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
                &[60.0],
                0.0,
                60.0,
            )
            .with_events(false),
        );
        assert_eq!(
            result,
            Err(FinalPropagationFailure::Gravity(
                satpy_core::GravityError::InvalidTime
            ))
        );
    }

    #[test]
    fn final_event_gravity_error_precedes_solver_nonfinite_status() {
        let context = invalid_time_gravity_context(StepperMethod::Vern9);
        let result = integrate_final_checked(
            ScalarPropagationRequest::new(
                &context,
                [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
                &[60.0],
                0.0,
                60.0,
            )
            .with_events(true),
        );
        assert_eq!(
            result,
            Err(FinalPropagationFailure::Gravity(
                satpy_core::GravityError::InvalidTime
            ))
        );
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn sampled_esdirk_gravity_error_precedes_solver_nonfinite_status() {
        let context = invalid_time_gravity_context(StepperMethod::Esdirk43);
        let result = integrate_adaptive(
            ScalarPropagationRequest::new(
                &context,
                [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
                &[60.0],
                0.0,
                60.0,
            )
            .with_events(false),
        )
        .expect("gravity failure must not exhaust propagation census");
        assert_invalid_time_gravity_result(&result);
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn sampled_event_esdirk_gravity_error_precedes_solver_nonfinite_status() {
        let context = invalid_time_gravity_context(StepperMethod::Esdirk43);
        let result = integrate_adaptive(
            ScalarPropagationRequest::new(
                &context,
                [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
                &[0.0, 60.0],
                0.0,
                60.0,
            )
            .with_events(true),
        )
        .expect("gravity failure must not exhaust propagation census");
        assert_invalid_time_gravity_result(&result);
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn final_esdirk_gravity_error_precedes_solver_nonfinite_status() {
        let context = invalid_time_gravity_context(StepperMethod::Esdirk43);
        let result = integrate_final_checked(
            ScalarPropagationRequest::new(
                &context,
                [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
                &[60.0],
                0.0,
                60.0,
            )
            .with_events(false),
        );
        assert_eq!(
            result,
            Err(FinalPropagationFailure::Gravity(
                satpy_core::GravityError::InvalidTime
            ))
        );
    }

    /// An unrunnable method is reported BEFORE a broken gravity model.
    ///
    /// This slot used to assert gravity-error precedence for ESDIRK on the
    /// event route. ESDIRK never had an event driver: that route substituted
    /// Tsit5, so the test asserted Tsit5 precedence under an ESDIRK label, and
    /// the real event-route invariant is already covered on Vern9 by
    /// `final_event_gravity_error_precedes_solver_nonfinite_status` above.
    ///
    /// What is genuinely new is the ordering: the refusal comes first even
    /// though this context's gravity model is also broken, because a run that
    /// is refused performs no gravity evaluation to report on.
    #[cfg(feature = "autodiff")]
    #[test]
    fn esdirk_event_refusal_precedes_a_broken_gravity_model() {
        let context = invalid_time_gravity_context(StepperMethod::Esdirk43);
        let result = integrate_final_checked(
            ScalarPropagationRequest::new(
                &context,
                [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
                &[60.0],
                0.0,
                60.0,
            )
            .with_events(true),
        );
        assert_eq!(result, Err(FinalPropagationFailure::MethodUnsupported));
    }

    #[test]
    fn guarded_hf_direct_and_reusable_entrypoints_reject_invalid_config_before_work() {
        for model in [4, 5] {
            let packed = test_coefficients(0);
            let config = Arc::new(ForceConfig {
                atm_model: model,
                eps: 1.0e-8,
                integrator_method: StepperMethod::Auto,
                ..ForceConfig::default()
            });

            let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
            let context = ScalarPropagationContext::new(2_460_310.5, Arc::clone(&config), gravity);
            let sampled = integrate_adaptive(
                ScalarPropagationRequest::new(&context, [0.0; 6], &[], 0.0, 0.0).with_events(false),
            )
            .expect("sampled invalid authority check must not exhaust census");
            assert!(sampled.terminal_event_fired);
            assert_eq!(sampled.terminal_event_name, "invalid_force_config");

            let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
            let context = ScalarPropagationContext::new(2_460_310.5, Arc::clone(&config), gravity);
            let checked = ReusableFinalCheckedIntegrator::new(context);
            assert_eq!(
                checked.err(),
                Some(FinalPropagationFailure::IntegrationFailure),
                "model={model} checked reusable accepted invalid config"
            );

            let gravity = ScalarGravityAssets::new(packed);
            let context = ScalarPropagationContext::new(2_460_310.5, config, gravity);
            let reusable = ReusableFinalNoEventIntegrator::new(context);
            assert!(
                reusable.is_err(),
                "model={model} no-event reusable accepted invalid config"
            );
        }
    }

    #[test]
    fn active_binary_srp_direct_and_reusable_entrypoints_reject_invalid_config() {
        let packed = test_coefficients(0);
        let config = Arc::new(ForceConfig {
            atm_model: 3,
            force_flags: ForceFlags::SRP,
            am_ratio: 0.02,
            cr: 1.3,
            sun_pos: Some([149_597_870.7, 0.0, 0.0]),
            eps: 1.0e-8,
            integrator_method: StepperMethod::Auto,
            ..ForceConfig::default()
        });

        let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
        let context = ScalarPropagationContext::new(2_460_310.5, Arc::clone(&config), gravity);
        let sampled = integrate_adaptive(
            ScalarPropagationRequest::new(&context, [0.0; 6], &[], 0.0, 0.0).with_events(false),
        )
        .expect("sampled invalid authority check must not exhaust census");
        assert!(sampled.terminal_event_fired);
        assert_eq!(sampled.terminal_event_name, "invalid_force_config");

        let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
        let context = ScalarPropagationContext::new(2_460_310.5, Arc::clone(&config), gravity);
        let checked = ReusableFinalCheckedIntegrator::new(context);
        assert_eq!(
            checked.err(),
            Some(FinalPropagationFailure::IntegrationFailure)
        );

        let gravity = ScalarGravityAssets::new(packed);
        let context = ScalarPropagationContext::new(2_460_310.5, config, gravity);
        let reusable = ReusableFinalNoEventIntegrator::new(context);
        assert!(reusable.is_err());
    }

    #[test]
    fn binary_eclipse_rejects_auto_and_esdirk_before_integration() {
        for stepper in [StepperMethod::Auto, StepperMethod::Esdirk43] {
            let config = ForceConfig {
                force_flags: ForceFlags::SRP,
                am_ratio: 1.0e-8,
                cr: 1.2,
                sun_pos: Some([149_597_870.7, 0.0, 0.0]),
                integrator_method: stepper,
                ..ForceConfig::default()
            };
            let error = validate_scalar_stepper_authority(&config, "binary eclipse")
                .expect_err("binary eclipse must be explicit scalar RK");
            assert!(error.to_string().contains("binary-eclipse SRP"), "{error}");
        }
    }

    #[test]
    fn sampled_binary_srp_preserves_typed_invalid_sun_geometry() {
        let packed = test_coefficients(5);
        let config = Arc::new(ForceConfig {
            sph_order: 5,
            force_flags: ForceFlags::SRP,
            am_ratio: 0.02,
            cr: 1.3,
            sun_pos: Some([f64::NAN, 0.0, 0.0]),
            eps: 1.0e-8,
            integrator_method: StepperMethod::Vern9,
            ..ForceConfig::default()
        });
        let gravity = ScalarGravityAssets::new(packed);
        let context = ScalarPropagationContext::new(2_460_310.5, config, gravity);
        let result = integrate_adaptive(
            ScalarPropagationRequest::new(
                &context,
                [7_000.0, 0.001, 0.0, 0.0, 0.0, 0.0],
                &[60.0],
                0.0,
                60.0,
            )
            .with_events(false),
        )
        .expect("sampled binary SRP geometry check must not exhaust census");
        assert!(result.terminal_event_fired);
        assert_eq!(result.terminal_eclipse_error, Some(EclipseError::Geometry));
        assert_eq!(result.terminal_event_name, "eclipse_geometry");
        assert!(result.times.is_empty());
        assert!(result.states.is_empty());
    }

    #[test]
    fn test_reusable_final_no_events_matches_one_shot_and_resets_state() {
        let packed = test_coefficients(5);
        let config = Arc::new(ForceConfig {
            sph_order: 5,
            force_flags: 0,
            dt_max: 60.0,
            eps: 1.0e-8,
            ..ForceConfig::default()
        });

        let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
        let reusable_context =
            ScalarPropagationContext::new(2_460_310.5, Arc::clone(&config), gravity);
        let mut reusable = ReusableFinalNoEventIntegrator::new(reusable_context)
            .expect("matching reusable authority");

        let tf_s = 1800.0;
        // Equinoctial elements: [a, h, k, p, q, L]
        let init_a = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let init_b = [7130.0, -0.0015, 0.0012, -0.0007, 0.0006, 1.1];

        let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
        let fresh_context =
            ScalarPropagationContext::new(2_460_310.5, Arc::clone(&config), gravity);
        let expected_a = integrate_final_checked(
            ScalarPropagationRequest::new(&fresh_context, init_a, &[tf_s], 0.0, tf_s)
                .with_events(false),
        )
        .expect("one-shot propagation A failed");
        let got_a = reusable
            .propagate(init_a, 0.0, tf_s)
            .expect("reusable propagation A failed");
        // Tolerance relaxed from 1e-12 to 1e-7: the one-shot path goes through
        // the rectification restart loop wrapper, and the ReusableFinalNoEvent
        // integrator initializes with [0;6] then resets, producing ~1e-9 f64
        // jitter in baseline caches. 1e-7 km = 0.1 mm, still very tight.
        assert_state_close(got_a, expected_a, 1e-7);

        let expected_b = integrate_final_checked(
            ScalarPropagationRequest::new(&fresh_context, init_b, &[tf_s], 0.0, tf_s)
                .with_events(false),
        )
        .expect("one-shot propagation B failed");
        let got_b = reusable
            .propagate(init_b, 0.0, tf_s)
            .expect("reusable propagation B failed");
        assert_state_close(got_b, expected_b, 1e-7);

        // Repeat A after B to ensure cache/state from prior run is not retained.
        let got_a_again = reusable
            .propagate(init_a, 0.0, tf_s)
            .expect("reusable propagation A2 failed");
        assert_state_close(got_a_again, expected_a, 1e-7);
    }

    #[test]
    fn reusable_final_constructs_with_explicit_stepper() {
        // Constructor smoke test only: nothing here observes which stepper
        // the reusable path actually resolved or propagates with it.
        let packed = test_coefficients(5);
        let config = Arc::new(ForceConfig {
            sph_order: 5,
            force_flags: 0,
            dt_max: 60.0,
            integrator_method: StepperMethod::Dopri5Compat,
            ..ForceConfig::default()
        });

        let gravity = ScalarGravityAssets::new(packed);
        let context = ScalarPropagationContext::new(2_460_310.5, config, gravity);
        let reusable = ReusableFinalNoEventIntegrator::new(context);

        assert!(
            reusable.is_ok(),
            "context stepper must construct reusable path"
        );
    }

    /// Short-arc sampled path (< `MAX_RECT_SEGMENT`) must produce the same
    /// final state as the final-only path — verifying zero regression from
    /// the segmented wrapper (single iteration, no rebase).
    ///
    /// Uses the same 1800s/eps=1e-8 parameters as `test_reusable_final_no_events`
    /// which is known to produce valid final-only output.
    #[test]
    fn test_sampled_vs_final_parity_short_arc() {
        let packed = test_coefficients(5);
        let config = Arc::new(ForceConfig {
            sph_order: 5,
            force_flags: 0,
            dt_max: 60.0,
            eps: 1.0e-8,
            integrator_method: StepperMethod::Dopri5Compat,
            ..ForceConfig::default()
        });

        let tf_s = 1800.0; // 30 min — well under MAX_RECT_SEGMENT (5400s)
        let init = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let jd0 = 2_460_310.5;
        let gravity = ScalarGravityAssets::new(packed);
        let context = ScalarPropagationContext::new(jd0, config, gravity);

        // Final-only path (known good at 1800s — see test_reusable_final_no_events).
        let delta_final = integrate_final_checked(
            ScalarPropagationRequest::new(&context, init, &[tf_s], 0.0, tf_s).with_events(false),
        )
        .expect("final-only propagation failed");

        // Sampled path with eval only at tf (matches final-only's eval grid).
        let sampled_result = integrate_adaptive(
            ScalarPropagationRequest::new(&context, init, &[tf_s], 0.0, tf_s).with_events(false),
        )
        .expect("sampled final-parity propagation census");

        assert!(
            !sampled_result.states.is_empty(),
            "sampled path returned no states"
        );
        assert!(
            !sampled_result.terminal_event_fired,
            "sampled path fired terminal event: {}",
            sampled_result.terminal_event_name
        );

        // Final state parity: same eval grid → same Dopri5 step sequence.
        // Allow 1e-7 km = 0.1 mm (same as reusable test) for baseline cache jitter.
        let last_sampled = sampled_result.states.last().unwrap();
        assert_state_close(*last_sampled, delta_final, 1e-7);
    }

    /// Multi-segment sampled integration (arc > `MAX_RECT_SEGMENT`) must
    /// produce finite output states with monotonic times.
    #[test]
    fn test_sampled_long_arc_multi_segment_completes() {
        let packed = test_coefficients(5);
        let config = Arc::new(ForceConfig {
            sph_order: 5,
            force_flags: 0,
            dt_max: 60.0,
            eps: 1.0e-6,
            integrator_method: StepperMethod::Dopri5Compat,
            ..ForceConfig::default()
        });

        // 10800s = 3 hours ≈ 2 LEO orbits — triggers ≥2 rectification segments.
        let tf_s = 10800.0;
        let init = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let jd0 = 2_460_310.5;

        let t_eval: Vec<f64> = (1..=36)
            .map(|index| test_usize_as_f64(index, "test sample index must fit u32") * 300.0)
            .collect();
        let gravity = ScalarGravityAssets::new(packed);
        let context = ScalarPropagationContext::new(jd0, config, gravity);
        let sampled_result = integrate_adaptive(
            ScalarPropagationRequest::new(&context, init, &t_eval, 0.0, tf_s).with_events(false),
        )
        .expect("sampled long-arc propagation census");

        // Verify it completed (or at least produced partial output).
        assert!(
            !sampled_result.states.is_empty(),
            "sampled long-arc produced no states"
        );

        // All output states must be finite.
        for (i, s) in sampled_result.states.iter().enumerate() {
            for (j, value) in s.iter().enumerate().take(6) {
                assert!(value.is_finite(), "non-finite state[{i}][{j}] = {value}");
            }
        }

        // Times must be monotonically increasing.
        for window in sampled_result.times.windows(2) {
            let [previous, next] = window else {
                panic!("windows(2) must produce two timestamps");
            };
            assert!(
                next >= previous,
                "non-monotonic times: {previous} followed by {next}"
            );
        }
    }

    fn assert_sparse_sampled_rectification_matches_final(direction: f64, method: StepperMethod) {
        let packed = test_coefficients(5);
        let config = Arc::new(ForceConfig {
            sph_order: 5,
            force_flags: 0,
            subtract_first_order: true,
            dt_max: 60.0,
            eps: 1.0e-8,
            integrator_method: method,
            ..ForceConfig::default()
        });
        let init = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let tf_s = direction * 10_800.0;
        let t_eval = [direction * 3_000.0, direction * 9_000.0, tf_s];
        let jd0 = 2_460_310.5;
        let gravity = ScalarGravityAssets::new(packed);
        let context = ScalarPropagationContext::new(jd0, config, gravity);

        let expected = integrate_final_checked(
            ScalarPropagationRequest::new(&context, init, &[tf_s], 0.0, tf_s).with_events(false),
        )
        .expect("final-only rectification oracle failed");

        let sampled = integrate_adaptive(
            ScalarPropagationRequest::new(&context, init, &t_eval, 0.0, tf_s).with_events(false),
        )
        .expect("sampled rectification propagation census");

        assert!(
            !sampled.terminal_event_fired,
            "{method:?}: {}",
            sampled.terminal_event_name
        );
        assert_eq!(sampled.times, t_eval, "{method:?}");
        assert_eq!(sampled.states.len(), t_eval.len(), "{method:?}");
        assert_state_close(*sampled.states.last().unwrap(), expected, 1e-5);
    }

    #[test]
    fn sampled_sparse_forward_rectification_uses_hidden_segment_endpoint() {
        for method in [StepperMethod::Dopri5Compat, StepperMethod::Vern9] {
            assert_sparse_sampled_rectification_matches_final(1.0, method);
        }
    }

    #[test]
    fn sampled_sparse_backward_rectification_uses_hidden_segment_endpoint() {
        for method in [StepperMethod::Dopri5Compat, StepperMethod::Vern9] {
            assert_sparse_sampled_rectification_matches_final(-1.0, method);
        }
    }

    /// A sampled event grid that does not span the request must be refused.
    ///
    /// Only DOPRI receives `t0_s`/`t_final_s`; every other sampled event
    /// wrapper reads its interval off `t_eval`. A grid starting after `t0`
    /// silently relabels the initial state and skips the dynamics before it,
    /// and a grid ending before `tf` never propagates the tail.
    ///
    /// Both arms are asserted, so this cannot pass by refusing everything.
    #[test]
    fn sampled_event_grid_must_span_the_requested_interval() {
        // Forward.
        assert!(
            sampled_eval_spans_request(&[0.0, 1800.0, 3600.0], 0.0, 3600.0),
            "a grid that spans the request must be accepted"
        );
        assert!(
            !sampled_eval_spans_request(&[1800.0, 3600.0], 0.0, 3600.0),
            "a grid starting after t0 skips the dynamics before its first sample"
        );
        assert!(
            !sampled_eval_spans_request(&[0.0, 1800.0], 0.0, 3600.0),
            "a grid ending before tf never propagates the tail"
        );

        // Backward, where the grid descends and the inequalities invert.
        assert!(
            sampled_eval_spans_request(&[3600.0, 1800.0, 0.0], 3600.0, 0.0),
            "a spanning backward grid must be accepted"
        );
        assert!(
            !sampled_eval_spans_request(&[1800.0, 0.0], 3600.0, 0.0),
            "a backward grid starting before t0 skips its leading dynamics"
        );
        assert!(
            !sampled_eval_spans_request(&[3600.0, 1800.0], 3600.0, 0.0),
            "a backward grid ending after tf never propagates the tail"
        );

        // Degenerate and nonfinite grids are refused rather than inferred.
        assert!(!sampled_eval_spans_request(&[], 0.0, 3600.0));
        assert!(!sampled_eval_spans_request(
            &[f64::NAN, 3600.0],
            0.0,
            3600.0
        ));
    }

    /// ESDIRK must refuse the routes that cannot run it, in RELEASE.
    ///
    /// Both shapes previously substituted Tsit5 and returned a successful
    /// result computed by a method the caller never asked for. The final-only
    /// helper was guarded by a `debug_assert!`, so the substitution was
    /// invisible in exactly the profile the campaign runs; the event route
    /// hard-coded `OdeMethod::Tsit5` after building dual state it then discarded.
    ///
    /// This test carries no `cfg(debug_assertions)`: it must hold in both
    /// profiles, which is the property that was missing.
    #[test]
    #[cfg(feature = "autodiff")]
    fn esdirk_refuses_routes_it_cannot_run_instead_of_substituting_tsit5() {
        // Spherical gravity on purpose. Nonspherical gravity is refused for a
        // DIFFERENT reason -- the dual Newton matrix would differentiate
        // another field -- and that refusal lands first, which would make this
        // test pass without ever exercising the method routing it is about.
        let packed = test_coefficients(1);
        let config = Arc::new(ForceConfig {
            sph_order: 1,
            force_flags: 0,
            subtract_first_order: true,
            dt_max: 60.0,
            eps: 1.0e-8,
            integrator_method: StepperMethod::Esdirk43,
            ..ForceConfig::default()
        });
        let init = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let tf_s = 10_800.0;
        let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
        let context = ScalarPropagationContext::new(2_460_310.5, Arc::clone(&config), gravity);

        let with_events = integrate_final_checked(
            ScalarPropagationRequest::new(&context, init, &[tf_s], 0.0, tf_s).with_events(true),
        );
        assert_eq!(
            with_events.unwrap_err(),
            FinalPropagationFailure::MethodUnsupported,
            "ESDIRK with an event handler silently ran another method"
        );

        // The no-event route is the one ESDIRK genuinely supports, so it must
        // NOT be refused -- otherwise this test would pass by rejecting
        // everything and would say nothing about method substitution.
        let without_events = integrate_final_checked(
            ScalarPropagationRequest::new(&context, init, &[tf_s], 0.0, tf_s).with_events(false),
        );
        assert!(
            !matches!(
                without_events,
                Err(FinalPropagationFailure::MethodUnsupported)
            ),
            "ESDIRK was refused on the route that can actually run it"
        );
    }

    /// Rectified high-order final-only propagation must restart from the
    /// segment epoch without corrupting the Encke reference state.
    #[test]
    fn test_high_order_final_long_arc_survives_rectification() -> anyhow::Result<()> {
        let packed = test_coefficients(5);
        let config = Arc::new(ForceConfig {
            sph_order: 5,
            force_flags: 0,
            subtract_first_order: true,
            dt_max: 60.0,
            eps: 1.0e-8,
            ..ForceConfig::default()
        });

        let init = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let tf_s = 10_800.0;
        for method in [StepperMethod::Vern9, StepperMethod::Rkv98] {
            let mut config_for_method = *config;
            config_for_method.integrator_method = method;
            let config = Arc::new(config_for_method);
            let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
            let context = ScalarPropagationContext::new(2_460_310.5, Arc::clone(&config), gravity);
            let segmented = integrate_final_checked(
                ScalarPropagationRequest::new(&context, init, &[tf_s], 0.0, tf_s)
                    .with_events(false),
            )?;

            if !segmented.iter().all(|value| value.is_finite()) {
                return Err(anyhow::anyhow!(
                    "{method:?} returned non-finite delta: {segmented:?}"
                ));
            }

            let mut rhs = LightyearRHS::new(
                init,
                0.0,
                2_460_310.5,
                Arc::clone(&config),
                Arc::clone(&packed),
            );
            rhs.adapt_cache_policy_for_eps(1e-8);
            rhs.reset_for_propagation(init, 0.0);
            #[cfg(feature = "scalar-leg-observer")]
            let unsegmented =
                integrate_final_no_events_with_rhs(&rhs, 0.0, tf_s, 1e-8, method, None)?;
            #[cfg(not(feature = "scalar-leg-observer"))]
            let unsegmented = integrate_final_no_events_with_rhs(&rhs, 0.0, tf_s, 1e-8, method)?;

            for (index, (lhs, rhs)) in segmented.iter().zip(unsegmented.iter()).enumerate() {
                let diff = (lhs - rhs).abs();
                if diff > 1e-5 {
                    return Err(anyhow::anyhow!(
                        "component {index} differs: lhs={lhs} rhs={rhs} diff={diff} tol=0.00001"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Shared dynamic-Sun/SRP fixture for binary-eclipse scalar paths.
    #[cfg(test)]
    fn convergence_fixture() -> (
        [f64; 6],
        f64,
        f64,
        Arc<ForceConfig>,
        Arc<PackedGravityCoeffs>,
    ) {
        let flags =
            ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
        crate::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("test ephemeris catalogues must load");
        let packed = test_coefficients(5);
        let jd0 = 2_460_310.5;
        let tf_s = 7200.0;
        let config = Arc::new(
            ForceConfig {
                sph_order: 5,
                force_flags: flags,
                subtract_first_order: true,
                atm_model: 3,
                am_ratio: 0.02,
                cd: 2.2,
                cr: 1.3,
                dt_max: 60.0,
                eps: 1e-8,
                integrator_method: StepperMethod::Vern9,
                ..ForceConfig::default()
            }
            .with_ephemeris_for_arc(jd0, jd0 + tf_s / SEC_PER_DAY)
            .expect("test arc must have dynamic ephemeris coverage"),
        );
        (
            [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
            jd0,
            tf_s,
            config,
            packed,
        )
    }

    #[cfg(test)]
    fn position_gap(a: &[f64; 6], b: &[f64; 6]) -> f64 {
        a[..3]
            .iter()
            .zip(&b[..3])
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn root_discrepancy_within_budget(discrepancy_m: f64) -> bool {
        discrepancy_m.is_finite() && discrepancy_m + 0.001 <= 0.10
    }

    #[test]
    fn root_discrepancy_gate_uses_metres_and_rejects_0_1001_m() {
        assert!(root_discrepancy_within_budget(0.099));
        assert!(!root_discrepancy_within_budget(0.1001));
    }

    /// One-shot generic scalar driver, retained only for SRP-off coverage.
    #[cfg(test)]
    fn one_shot_endpoint(
        jd0: f64,
        config: Arc<ForceConfig>,
        packed: Arc<PackedGravityCoeffs>,
        init: [f64; 6],
        tf_s: f64,
        eps: f64,
        h_max: f64,
    ) -> [f64; 6] {
        let mut rhs = LightyearRHS::new(init, 0.0, jd0, config, packed);
        rhs.adapt_cache_policy_for_eps(eps);
        rhs.reset_for_propagation(init, 0.0);
        let system = LightyearSystem { rhs: &rhs };
        let result = integrate_final(
            &system,
            stepper_ode_method(StepperMethod::Vern9),
            &[0.0; 6],
            0.0,
            tf_s,
            IntegratorConfig {
                error_control: ErrorControl::Absolute { eps },
                h0: None,
                h_min: 1e-12,
                h_max,
                max_steps: MAX_STEPS,
                max_rejects: 50,
                force_eval: false,
            },
        );
        result.y.try_into().expect("six-state endpoint")
    }

    /// Synthetic binary-eclipse coordinator stays within 0.10 m of its tight reference.
    ///
    /// This is a unit fixture (`atm_model = 3`, `dt_max = 60 s`), not a
    /// Part-A model-5/current-controls accuracy claim.
    #[test]
    fn synthetic_binary_eclipse_coordinator_converges_with_tight_reference() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let (init, jd0, tf_s, config, packed) = convergence_fixture();
        let run = |eps: f64| {
            let mut config_for_eps = *config;
            config_for_eps.eps = eps;
            let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
            let context = ScalarPropagationContext::new(jd0, Arc::new(config_for_eps), gravity);
            integrate_final_checked(
                ScalarPropagationRequest::new(&context, init, &[tf_s], 0.0, tf_s)
                    .with_events(false),
            )
            .expect("binary-eclipse propagation")
        };
        let reference = run(1e-12);
        for eps in [1e-8, 1e-9, 1e-10, 1e-11] {
            let candidate = run(eps);
            assert!(candidate.iter().all(|value| value.is_finite()));
            assert!(
                position_gap(&candidate, &reference) <= 1.0e-4,
                "eps={eps:e} exceeds the 0.10 m fixed eclipse/reference bound"
            );
            assert_eq!(candidate.map(f64::to_bits), run(eps).map(f64::to_bits));
        }
    }

    /// Production tolerance band carries no tolerance-independent RHS error floor.
    #[test]
    fn production_band_carries_no_stale_cache_error_floor() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let (init, jd0, tf_s, config, packed) = convergence_fixture();
        let run = |eps: f64| -> [f64; 6] {
            let mut config_for_eps = *config;
            config_for_eps.eps = eps;
            let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
            let context = ScalarPropagationContext::new(jd0, Arc::new(config_for_eps), gravity);
            integrate_final_checked(
                ScalarPropagationRequest::new(&context, init, &[tf_s], 0.0, tf_s)
                    .with_events(false),
            )
            .expect("production-wrapper propagation must succeed")
        };

        let reference = run(1e-13);
        for exponent in 6..=9 {
            let eps = 10f64.powi(-exponent);
            let error_km = position_gap(&run(eps), &reference);
            assert!(
                error_km < 5e-5,
                "eps=1e-{exponent} endpoint is {error_km:e} km from a 1e-13 \
                 reference, over the 5e-5 km bound. An error that does not fall \
                 when eps falls is a stale cache in the RHS, not a step-size \
                 problem -- tightening the tolerance will not fix it."
            );
        }
    }

    /// Raw SRP integration fails closed until eclipse side is coordinated.
    #[test]
    fn raw_srp_driver_fails_closed_before_collecting_step_metrics() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let (init, jd0, tf_s, config, packed) = convergence_fixture();
        let mut rhs = LightyearRHS::new(init, 0.0, jd0, config, packed);
        rhs.adapt_cache_policy_for_eps(1e-8);
        rhs.reset_for_propagation(init, 0.0);
        let system = LightyearSystem { rhs: &rhs };
        let raw = integrate_final(
            &system,
            stepper_ode_method(StepperMethod::Vern9),
            &[0.0; 6],
            0.0,
            tf_s,
            IntegratorConfig {
                error_control: ErrorControl::Absolute { eps: 1e-8 },
                h0: None,
                h_min: 1e-12,
                h_max: 60.0,
                max_steps: MAX_STEPS,
                max_rejects: 50,
                force_eval: false,
            },
        );
        assert!(matches!(
            raw.status,
            OdeIntegrationStatus::NanEncountered | OdeIntegrationStatus::NonFiniteState
        ));
        assert_eq!(
            rhs.take_eclipse_error(),
            Some(EclipseError::UninitializedSide)
        );
    }

    #[test]
    fn every_srp_replay_stage_fails_closed_outside_part_a_envelope() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let (init, jd0, _, config, packed) = convergence_fixture();
        let mut rhs = LightyearRHS::new(init, 0.0, jd0, config, packed);
        rhs.reset_for_propagation(init, 0.0);
        rhs.set_eclipse_side(EclipseSide::Lit);
        let system = LightyearSystem { rhs: &rhs };
        let mut derivative = [0.0; 6];
        OdeSystemTrait::rhs(
            &system,
            0.0,
            &[50_000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &mut derivative,
        );
        assert!(derivative.iter().all(|value| value.is_nan()));
        assert_eq!(rhs.take_eclipse_error(), Some(EclipseError::Envelope));
    }

    /// Generic one-shot scalar integration remains available when SRP is disabled.
    #[test]
    fn one_shot_generic_non_srp_path_remains_available() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let (init, jd0, tf_s, config, packed) = convergence_fixture();
        let mut generic = *config;
        generic.force_flags &= !ForceFlags::SRP;
        let endpoint = one_shot_endpoint(jd0, Arc::new(generic), packed, init, tf_s, 1e-8, 60.0);
        assert!(endpoint.iter().all(|value| value.is_finite()));
    }

    /// Sampled and final APIs share one binary-eclipse trajectory.
    #[test]
    fn sampled_and_final_binary_eclipse_paths_agree_at_matched_tolerance() {
        let _eclipse_state_guard = eclipse_test_state_guard();
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let (init, jd0, _, config, packed) = convergence_fixture();
        // This interval contains both ingress and egress for the fixture.
        let tf_s = 3600.0;
        let eps = 1e-9;
        TEST_ECLIPSE_SPLITS.store(0, std::sync::atomic::Ordering::Relaxed);
        TEST_ROOT_TRANSACTION_RESETS.store(0, std::sync::atomic::Ordering::Relaxed);
        TEST_ROOT_TRANSACTION_CONTINUATIONS.store(0, std::sync::atomic::Ordering::Relaxed);
        TEST_ECLIPSE_ROOTS.lock().expect("root capture").clear();
        let final_result = integrate_binary_eclipse_scalar(
            init,
            &[tf_s],
            0.0,
            tf_s,
            false,
            BinaryEclipseContext {
                eps,
                jd0,
                config: Arc::clone(&config),
                packed: Arc::clone(&packed),
                stepper: StepperMethod::Vern9,
            },
        )
        .expect("final coordinated propagation");
        assert!(
            !final_result.terminal_event_fired && !final_result.max_steps_exceeded,
            "coordinated final path must reach its requested terminal time: {}",
            final_result.terminal_event_name
        );
        assert_eq!(
            TEST_ECLIPSE_SPLITS.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "fixture must execute ingress and egress"
        );
        assert_eq!(
            TEST_ROOT_TRANSACTION_RESETS.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "each committed crossing must start exactly one root transaction"
        );
        assert_eq!(
            TEST_ROOT_TRANSACTION_CONTINUATIONS.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "each root transaction must continue without resetting its RHS"
        );
        let production_roots = TEST_ECLIPSE_ROOTS.lock().expect("root capture").clone();
        assert_eq!(production_roots.len(), 2);
        let mut tight_config = *config;
        tight_config.dt_max = 10.0;
        TEST_ECLIPSE_ROOTS.lock().expect("root capture").clear();
        integrate_binary_eclipse_scalar(
            init,
            &[tf_s],
            0.0,
            tf_s,
            false,
            BinaryEclipseContext {
                eps: 1e-12,
                jd0,
                config: Arc::new(tight_config),
                packed: Arc::clone(&packed),
                stepper: StepperMethod::Vern9,
            },
        )
        .expect("tight full-arc eclipse reference");
        let reference_roots = TEST_ECLIPSE_ROOTS.lock().expect("root capture").clone();
        assert_eq!(reference_roots.len(), 2);
        for (production, reference) in production_roots.iter().zip(&reference_roots) {
            let conservative_discrepancy_m = (production - reference).abs() * 20.0 * 1000.0;
            assert!(
                root_discrepancy_within_budget(conservative_discrepancy_m),
                "root discrepancy plus 1 mm bracket exceeds 0.10 m: {conservative_discrepancy_m} m"
            );
        }
        let final_state = *final_result.states.last().expect("final endpoint");
        TEST_ECLIPSE_SPLITS.store(0, std::sync::atomic::Ordering::Relaxed);
        TEST_ECLIPSE_ROOTS.lock().expect("root capture").clear();
        let sampled = integrate_binary_eclipse_scalar(
            init,
            &[600.0, 1200.0, 1800.0, 2400.0, 3000.0, tf_s],
            0.0,
            tf_s,
            false,
            BinaryEclipseContext {
                eps,
                jd0,
                config: Arc::clone(&config),
                packed: Arc::clone(&packed),
                stepper: StepperMethod::Vern9,
            },
        )
        .expect("sampled coordinated propagation");
        assert_eq!(
            TEST_ECLIPSE_SPLITS.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "sampled fixture must execute ingress and egress"
        );
        assert_eq!(
            final_state.map(f64::to_bits),
            sampled
                .states
                .last()
                .expect("sampled endpoint")
                .map(f64::to_bits)
        );

        for (&sample_time, sample_state) in sampled.times.iter().zip(&sampled.states) {
            let reference = integrate_binary_eclipse_scalar(
                init,
                &[sample_time],
                0.0,
                sample_time,
                false,
                BinaryEclipseContext {
                    eps: 1e-12,
                    jd0,
                    config: Arc::new(tight_config),
                    packed: Arc::clone(&packed),
                    stepper: StepperMethod::Vern9,
                },
            )
            .expect("tight sampled reference");
            let reference_state = reference.states.last().expect("reference endpoint");
            assert!(
                position_gap(sample_state, reference_state) <= 1.0e-4,
                "sample at {sample_time} s exceeds 0.10 m dense-output bound"
            );
        }

        let mut forward_baseline = [0.0; 6];
        equinoc2eci_impl(&init, 6, tf_s, 0.0, &mut forward_baseline);
        let mut final_eci = [0.0; 6];
        for ((final_component, baseline_component), delta_component) in
            final_eci.iter_mut().zip(forward_baseline).zip(final_state)
        {
            *final_component = baseline_component + delta_component;
        }
        let mut backward_init = [0.0; 6];
        eci2equinoc_impl_f64(&final_eci, 6, 0.0, 0.0, &mut backward_init);
        TEST_ECLIPSE_SPLITS.store(0, std::sync::atomic::Ordering::Relaxed);
        TEST_ROOT_TRANSACTION_RESETS.store(0, std::sync::atomic::Ordering::Relaxed);
        TEST_ROOT_TRANSACTION_CONTINUATIONS.store(0, std::sync::atomic::Ordering::Relaxed);
        integrate_binary_eclipse_scalar(
            backward_init,
            &[0.0],
            tf_s,
            0.0,
            false,
            BinaryEclipseContext {
                eps,
                jd0,
                config: Arc::new(tight_config),
                packed,
                stepper: StepperMethod::Vern9,
            },
        )
        .expect("backward coordinated propagation");
        assert_eq!(
            TEST_ECLIPSE_SPLITS.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "backward fixture must execute egress and ingress"
        );
        assert_eq!(
            TEST_ROOT_TRANSACTION_RESETS.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "each backward crossing must start exactly one root transaction"
        );
        assert_eq!(
            TEST_ROOT_TRANSACTION_CONTINUATIONS.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "each backward transaction must continue without resetting its RHS"
        );
    }

    #[test]
    fn binary_eclipse_reusable_integrators_own_and_reuse_both_rhs_instances() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let (init, jd0, _, config, packed) = convergence_fixture();
        let tf_s = 3_600.0;
        let eps = 1.0e-9;
        let mut config_for_eps = *config;
        config_for_eps.eps = eps;
        let config = Arc::new(config_for_eps);
        let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
        let fresh_context = ScalarPropagationContext::new(jd0, Arc::clone(&config), gravity);
        let expected = integrate_final_checked(
            ScalarPropagationRequest::new(&fresh_context, init, &[tf_s], 0.0, tf_s)
                .with_events(true),
        )
        .expect("fresh binary eclipse");
        let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
        let checked_context = ScalarPropagationContext::new(jd0, Arc::clone(&config), gravity);
        let mut checked = ReusableFinalCheckedIntegrator::new(checked_context)
            .expect("checked reusable binary eclipse");
        assert!(checked.eclipse_root_rhs.is_some());
        for _ in 0..2 {
            let state = checked
                .propagate_checked(init, 0.0, tf_s)
                .expect("checked reused binary eclipse");
            assert_eq!(state.map(f64::to_bits), expected.map(f64::to_bits));
        }
        assert_eq!(
            checked.stats(),
            FinalCheckedReuseStats {
                propagations: 2,
                rhs_construct_count: 2,
                rhs_reuse_hits: 2,
            }
        );

        let gravity = ScalarGravityAssets::new(packed);
        let no_event_context = ScalarPropagationContext::new(jd0, config, gravity);
        let mut no_event = ReusableFinalNoEventIntegrator::new(no_event_context)
            .expect("no-event reusable binary eclipse");
        assert!(no_event.eclipse_root_rhs.is_some());
        let first = no_event
            .propagate(init, 0.0, tf_s)
            .expect("first no-event reuse");
        let second = no_event
            .propagate(init, 0.0, tf_s)
            .expect("second no-event reuse");
        assert_eq!(first.map(f64::to_bits), second.map(f64::to_bits));
    }

    /// Fresh-vs-reused bit identity in the cell production actually flies:
    /// DISTINCT initial states, eclipse/SRP coordinator active, JB2008
    /// `atm_model: 5`.
    ///
    /// The two older reuse tests each cover one axis and miss the intersection.
    /// `reusable_final_checked_sequence_matches_fresh_bits_and_resets` varies
    /// the state but runs `force_flags: 0` — no drag, no SRP, no third body,
    /// hence no eclipse coordinator and no atmosphere.
    /// `binary_eclipse_reusable_integrators_own_and_reuse_both_rhs_instances`
    /// runs the production flags but propagates the SAME state twice, at
    /// `atm_model: 3`. Production flies
    /// `DRAG | SRP | SUN_GRAVITY | MOON_GRAVITY` at `atm_model: 5`.
    ///
    /// This matters because `reset_for_propagation` deliberately does NOT clear
    /// `cached_segment`, `cached_rotation` or `cached_driver_utc_jd`. That
    /// exemption is justified by those keys being absolute TAI, so a change of
    /// initial state cannot alias them — an argument this test converts into a
    /// measurement. The JB2008 driver cache is the reason `atm_model` is 5 here
    /// rather than 3: model 3 never consults `cached_driver_utc_jd`.
    #[test]
    fn binary_eclipse_reuse_matches_fresh_across_distinct_states_under_jb2008() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let flags =
            ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
        crate::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("test ephemeris catalogues must load");
        let packed = test_coefficients(5);
        let jd0 = 2_460_310.5;
        let tf_s = 3_600.0;
        let config = Arc::new(
            ForceConfig {
                sph_order: 5,
                force_flags: flags,
                subtract_first_order: true,
                atm_model: 5,
                am_ratio: 0.02,
                cd: 2.2,
                cr: 1.3,
                dt_max: 60.0,
                eps: 1.0e-9,
                integrator_method: StepperMethod::Vern9,
                ..ForceConfig::default()
            }
            .with_ephemeris_for_arc(jd0, jd0 + tf_s / SEC_PER_DAY)
            .expect("test arc must have dynamic ephemeris and JB2008 driver coverage"),
        );
        // Three states, the first repeated last: the integrator must return to
        // its own earlier answer after intervening work at a different state.
        // A stale baseline can survive a single reuse; it cannot survive this.
        let states = [
            [7_050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
            [7_130.0, -0.0015, 0.0012, -0.0007, 0.0006, 1.1],
            [7_050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2],
        ];

        let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
        let fresh_context = ScalarPropagationContext::new(jd0, Arc::clone(&config), gravity);
        let fresh = states.map(|state| {
            integrate_final_checked(
                ScalarPropagationRequest::new(&fresh_context, state, &[tf_s], 0.0, tf_s)
                    .with_events(true),
            )
            .expect("fresh checked binary-eclipse propagation")
        });

        let gravity = ScalarGravityAssets::new(packed);
        let reusable_context = ScalarPropagationContext::new(jd0, config, gravity);
        let mut reusable = ReusableFinalCheckedIntegrator::new(reusable_context)
            .expect("reusable checked binary-eclipse integrator");
        // Non-vacuity: without the eclipse root RHS this fixture would be
        // testing the plain scalar path and would prove nothing about the cell
        // it names.
        assert!(
            reusable.eclipse_root_rhs.is_some(),
            "SRP-on fixture must take the binary-eclipse path"
        );
        let reused = states.map(|state| {
            reusable
                .propagate_checked(state, 0.0, tf_s)
                .expect("reused checked binary-eclipse propagation")
        });

        assert_eq!(
            reused.map(|state| state.map(f64::to_bits)),
            fresh.map(|state| state.map(f64::to_bits)),
            "reused integrator must reproduce fresh bits at every distinct state"
        );
        assert_eq!(
            reused[0].map(f64::to_bits),
            reused[2].map(f64::to_bits),
            "returning to an earlier state must reproduce its earlier answer"
        );
        // Non-vacuity: if the two states converged to the same bits, the
        // identity above would hold for a broken cache too.
        assert_ne!(
            reused[0].map(f64::to_bits),
            reused[1].map(f64::to_bits),
            "the distinct states must produce distinct answers"
        );
        let jb2008_reuse = reusable.stats();
        assert_eq!(jb2008_reuse.propagations, 3);
        // Non-vacuity: the reuse path must actually have been taken.
        assert!(
            jb2008_reuse.rhs_reuse_hits > 0,
            "reuse path must be exercised, not rebuilt every call"
        );
    }

    #[test]
    fn binary_eclipse_reusable_checked_recovers_after_envelope_failure() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let (init, jd0, _, config, packed) = convergence_fixture();
        let eps = 1.0e-9;
        let tf_s = 3_600.0;
        let mut config_for_eps = *config;
        config_for_eps.eps = eps;
        let config = Arc::new(config_for_eps);
        let gravity = ScalarGravityAssets::new(Arc::clone(&packed));
        let context = ScalarPropagationContext::new(jd0, Arc::clone(&config), gravity);
        let constructed = ReusableFinalCheckedIntegrator::new(context);
        assert!(constructed.is_ok());
        let Some(mut reusable) = constructed.ok() else {
            return;
        };

        let first = reusable.propagate_checked(init, 0.0, tf_s);
        assert!(first.is_ok());
        let Some(first) = first.ok() else {
            return;
        };

        let outside_eci = [60_000.0, 0.0, 0.0, 0.0, 2.5, 0.0];
        let mut outside = [0.0; 6];
        eci2equinoc_impl_f64(&outside_eci, 6, 0.0, 0.0, &mut outside);
        assert!(matches!(
            reusable.propagate_checked(outside, 0.0, 600.0),
            Err(FinalPropagationFailure::Eclipse(EclipseError::Envelope))
        ));

        let third = reusable.propagate_checked(init, 0.0, tf_s);
        assert!(third.is_ok());
        let Some(third) = third.ok() else {
            return;
        };
        assert_eq!(first.map(f64::to_bits), third.map(f64::to_bits));
        assert_eq!(
            reusable.stats(),
            FinalCheckedReuseStats {
                propagations: 3,
                rhs_construct_count: 2,
                rhs_reuse_hits: 4,
            }
        );
    }

    #[test]
    fn binary_eclipse_sampling_overrides_force_eval_without_trajectory_drift() {
        let _eclipse_state_guard = eclipse_test_state_guard();
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let (init, jd0, _, config, packed) = convergence_fixture();
        let mut config_for_eps = *config;
        config_for_eps.eps = 1.0e-9;
        let config = Arc::new(config_for_eps);
        TEST_ECLIPSE_ROOTS.lock().expect("root capture").clear();
        integrate_binary_eclipse_scalar(
            init,
            &[3_600.0],
            0.0,
            3_600.0,
            false,
            BinaryEclipseContext {
                eps: 1.0e-9,
                jd0,
                config: Arc::clone(&config),
                packed: Arc::clone(&packed),
                stepper: StepperMethod::Vern9,
            },
        )
        .expect("root discovery");
        let roots = TEST_ECLIPSE_ROOTS.lock().expect("root capture").clone();
        assert_eq!(roots.len(), 2);
        let mut t_eval = vec![0.0];
        for root in roots {
            t_eval.extend([root - 5.0e-10, root, root + 5.0e-10]);
        }
        t_eval.push(3_600.0);
        let gravity = ScalarGravityAssets::new(packed);
        let context = ScalarPropagationContext::new(jd0, config, gravity);
        let run = |force_eval| {
            let output_mode = if force_eval {
                SampledOutputMode::ForceEvaluationTimes
            } else {
                SampledOutputMode::Interpolated
            };
            integrate_adaptive(
                ScalarPropagationRequest::new(&context, init, &t_eval, 0.0, 3_600.0)
                    .with_events(false)
                    .with_output_mode(output_mode),
            )
        };
        let unforced = run(false).expect("unforced sampled propagation census");
        let hostile = run(true).expect("force-eval sampled propagation census");
        assert!(!unforced.terminal_event_fired, "{unforced:?}");
        assert_eq!(
            unforced
                .times
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>(),
            t_eval.iter().map(|time| time.to_bits()).collect::<Vec<_>>(),
            "near-root requested times must not collapse onto the committed root"
        );
        assert_eq!(hostile.terminal_event_fired, unforced.terminal_event_fired);
        assert_eq!(hostile.times, unforced.times);
        assert_eq!(
            hostile
                .states
                .iter()
                .flatten()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            unforced
                .states
                .iter()
                .flatten()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(hostile.metrics.total_steps, unforced.metrics.total_steps);
        assert_eq!(hostile.metrics.total_evals, unforced.metrics.total_evals);
    }

    #[test]
    fn coordinator_commits_hidden_double_crossing_forward_and_backward_in_order() {
        #[cfg(feature = "prop-census")]
        let _census = crate::probe::test_census_guard();
        let _eclipse_state_guard = eclipse_test_state_guard();
        let radius_km = 7_000.0;
        let speed_km_s = (MU / radius_km).sqrt();
        let period_s = 2.0 * std::f64::consts::PI * (radius_km.powi(3) / MU).sqrt();
        let normal_x = (6_378.137 - 100.0) / radius_km;
        let normal_z = (1.0 - normal_x * normal_x).sqrt();
        let mut init = [0.0; 6];
        eci2equinoc_impl_f64(
            &[
                radius_km * normal_z,
                0.0,
                -radius_km * normal_x,
                0.0,
                speed_km_s,
                0.0,
            ],
            6,
            0.0,
            0.0,
            &mut init,
        );
        let packed = test_coefficients(0);
        let config = Arc::new(ForceConfig {
            sph_order: 0,
            force_flags: ForceFlags::SRP,
            am_ratio: 1.0e-12,
            cr: 1.0,
            sun_pos: Some([149_597_870.7, 0.0, 0.0]),
            dt_max: 900.0,
            eps: 1.0e-6,
            integrator_method: StepperMethod::Vern9,
            ..ForceConfig::default()
        });
        let run = |state, start, end| {
            #[cfg(feature = "prop-census")]
            crate::probe::reset().expect("site census reset");
            TEST_ECLIPSE_ROOTS.lock().expect("root capture").clear();
            TEST_ECLIPSE_SPLITS.store(0, std::sync::atomic::Ordering::Relaxed);
            TEST_HIDDEN_DOUBLE_ACCEPTED_STEPS.store(0, std::sync::atomic::Ordering::Relaxed);
            let result = integrate_binary_eclipse_scalar(
                state,
                &[end],
                start,
                end,
                false,
                BinaryEclipseContext {
                    eps: 1.0e-6,
                    jd0: 2_460_310.5,
                    config: Arc::clone(&config),
                    packed: Arc::clone(&packed),
                    stepper: StepperMethod::Vern9,
                },
            )
            .expect("hidden-double coordinator propagation");
            assert_eq!(
                TEST_ECLIPSE_SPLITS.load(std::sync::atomic::Ordering::Relaxed),
                2
            );
            assert!(
                TEST_HIDDEN_DOUBLE_ACCEPTED_STEPS.load(std::sync::atomic::Ordering::Relaxed) > 0,
                "fixture never presented same-side accepted endpoints containing two crossings"
            );
            let roots = TEST_ECLIPSE_ROOTS.lock().expect("root capture").clone();
            assert_eq!(roots.len(), 2);
            let [first_root, second_root] = roots.as_slice() else {
                panic!("hidden-double fixture must capture exactly two roots");
            };
            if end > start {
                assert!(first_root < second_root);
            } else {
                assert!(first_root > second_root);
            }
            #[cfg(feature = "prop-census")]
            {
                use crate::probe::EclipseTransactionSite::{Main, Proof, Refine, Window};

                let sites = crate::probe::eclipse_transaction_site_snapshot();
                let site_row = |site: crate::probe::EclipseTransactionSite| {
                    sites.get(site.index()).copied().expect("known site")
                };
                let sum = sites.iter().copied().fold(
                    crate::probe::EclipseTransactionSiteCensus::default(),
                    |mut total, row| {
                        total.legs += row.legs;
                        total.steps += row.steps;
                        total.evals += row.evals;
                        total.rejected += row.rejected;
                        total
                    },
                );
                assert_eq!(
                    sum.steps,
                    u64::try_from(result.metrics.total_steps).expect("step total fits u64")
                );
                assert_eq!(
                    sum.evals,
                    u64::try_from(result.metrics.total_evals).expect("eval total fits u64")
                );
                assert_eq!(site_row(Refine).legs, 2);
                assert_eq!(site_row(Proof).legs, 2);
                assert_eq!(site_row(Window).legs, 2);
                assert!(site_row(Main).legs > 0);
                assert!(site_row(Refine).steps > 0);
                assert!(site_row(Proof).steps > 0);
                assert!(site_row(Window).steps > 0);
                assert!(site_row(Refine).evals > 0);
                assert!(site_row(Proof).evals > 0);
                assert!(site_row(Window).evals > 0);

                let rendered = crate::probe::report().expect("site census report");
                for name in ["main", "refine", "proof", "window"] {
                    assert!(
                        rendered.contains(&format!("PROP_ECLIPSE_SITE {name},")),
                        "missing {name} site row:\n{rendered}"
                    );
                }
            }
            (result, roots)
        };
        let (forward, _) = run(init, 0.0, period_s);
        let final_delta = *forward.states.last().expect("forward endpoint");
        let mut baseline = [0.0; 6];
        equinoc2eci_impl(&init, 6, period_s, 0.0, &mut baseline);
        for (baseline_component, delta_component) in baseline.iter_mut().zip(final_delta) {
            *baseline_component += delta_component;
        }
        let mut backward_init = [0.0; 6];
        eci2equinoc_impl_f64(&baseline, 6, 0.0, 0.0, &mut backward_init);
        run(backward_init, period_s, 0.0);
    }

    #[test]
    fn all_force_high_order_propagation_uses_continuous_ephemeris() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let flags =
            ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
        crate::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("test ephemeris catalogues must load");
        let packed = test_coefficients(5);
        let init = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let jd0 = 2_460_310.5;
        let tf_s = 7200.0;
        let dynamic_config = ForceConfig {
            sph_order: 5,
            force_flags: flags,
            subtract_first_order: true,
            atm_model: 3,
            am_ratio: 0.02,
            cd: 2.2,
            cr: 1.3,
            dt_max: 60.0,
            eps: 1e-8,
            integrator_method: StepperMethod::Vern9,
            ..ForceConfig::default()
        }
        .with_ephemeris_for_arc(jd0, jd0 + tf_s / SEC_PER_DAY)
        .expect("test arc must have dynamic ephemeris coverage");
        assert_ne!(dynamic_config.dynamic_ephemeris_flags & flags, 0);
        let dynamic_config = Arc::new(dynamic_config);

        let expected_cache_policy = {
            let mut rhs = LightyearRHS::new(
                init,
                0.0,
                jd0,
                Arc::clone(&dynamic_config),
                Arc::clone(&packed),
            );
            rhs.adapt_cache_policy_for_eps(1e-11);
            rhs.cache_policy_for_test()
        };

        let propagate =
            |config: &Arc<ForceConfig>, expected_policy: f64, method: StepperMethod, eps: f64| {
                let mut rhs =
                    LightyearRHS::new(init, 0.0, jd0, Arc::clone(config), Arc::clone(&packed));
                rhs.adapt_cache_policy_for_eps(1e-11);
                assert_eq!(
                    rhs.cache_policy_for_test().to_bits(),
                    expected_policy.to_bits(),
                    "{method:?} eps={eps:e} changed fixed RHS/cache policy"
                );
                let mut method_config = **config;
                method_config.integrator_method = method;
                let result = integrate_binary_eclipse_scalar(
                    init,
                    &[tf_s],
                    0.0,
                    tf_s,
                    false,
                    BinaryEclipseContext {
                        eps,
                        jd0,
                        config: Arc::new(method_config),
                        packed: Arc::clone(&packed),
                        stepper: method,
                    },
                )
                .unwrap_or_else(|error| panic!("{method:?} eps={eps:e} failed: {error:?}"));
                assert!(
                    !result.terminal_event_fired,
                    "{method:?} eps={eps:e} failed: {}",
                    result.terminal_event_name
                );
                result.states.last().copied().expect("six-state endpoint")
            };

        let sweep = [
            (StepperMethod::Vern9, 1e-7),
            (StepperMethod::Vern9, 1e-8),
            (StepperMethod::Vern9, 1e-9),
            (StepperMethod::Rkv98, 1e-11),
            (StepperMethod::Rkv98, 1e-12),
        ];
        let forward: Vec<[f64; 6]> = sweep
            .iter()
            .map(|&(method, eps)| propagate(&dynamic_config, expected_cache_policy, method, eps))
            .collect();
        let reverse: Vec<[f64; 6]> = sweep
            .iter()
            .rev()
            .map(|&(method, eps)| propagate(&dynamic_config, expected_cache_policy, method, eps))
            .collect();
        for ((forward_endpoint, reverse_endpoint), &(method, eps)) in
            forward.iter().zip(reverse.iter().rev()).zip(sweep.iter())
        {
            for (component, (forward_component, reverse_component)) in forward_endpoint
                .iter()
                .zip(reverse_endpoint.iter())
                .enumerate()
            {
                assert_eq!(
                    forward_component.to_bits(),
                    reverse_component.to_bits(),
                    "{method:?} eps={eps:e} endpoint component {component} depends on sweep order"
                );
            }
        }

        let error_norms = |actual: &[f64; 6], reference: &[f64; 6]| {
            let position = actual[..3]
                .iter()
                .zip(&reference[..3])
                .map(|(actual, reference)| (actual - reference).powi(2))
                .sum::<f64>()
                .sqrt();
            let velocity = actual[3..]
                .iter()
                .zip(&reference[3..])
                .map(|(actual, reference)| (actual - reference).powi(2))
                .sum::<f64>()
                .sqrt();
            (position, velocity)
        };
        let rkv98_reference = forward
            .get(3)
            .expect("sweep must contain the RKV98 reference endpoint");
        let vern9_errors: Vec<(f64, f64)> = forward
            .get(..3)
            .expect("sweep must contain three Vern9 endpoints")
            .iter()
            .map(|endpoint| error_norms(endpoint, rkv98_reference))
            .collect();
        let tight_vern9_error = *vern9_errors
            .get(2)
            .expect("sweep must contain the tight Vern9 endpoint");
        assert!(
            tight_vern9_error.0 < 1e-5,
            "Vern9 eps=1e-9 position error {} km exceeds 1e-5 km",
            tight_vern9_error.0
        );
        assert!(
            tight_vern9_error.1 < 1e-7,
            "Vern9 eps=1e-9 velocity error {} km/s exceeds 1e-7 km/s",
            tight_vern9_error.1
        );

        let rkv98_self_error = error_norms(
            forward
                .get(3)
                .expect("sweep must contain the first RKV98 endpoint"),
            forward
                .get(4)
                .expect("sweep must contain the second RKV98 endpoint"),
        );
        assert!(
            rkv98_self_error.0 < 1e-7,
            "RKV98 eps=1e-11/1e-12 position delta {} km exceeds 1e-7 km",
            rkv98_self_error.0
        );
        assert!(
            rkv98_self_error.1 < 1e-9,
            "RKV98 eps=1e-11/1e-12 velocity delta {} km/s exceeds 1e-9 km/s",
            rkv98_self_error.1
        );
        let mut frozen_config = *dynamic_config;
        frozen_config.dynamic_ephemeris_flags = 0;
        frozen_config.sun_invariants = frozen_config
            .sun_pos
            .and_then(|position| BodyInvariants::precompute(&position, frozen_config.mu_sun));
        frozen_config.moon_invariants = frozen_config
            .moon_pos
            .and_then(|position| BodyInvariants::precompute(&position, frozen_config.mu_moon));
        let frozen_config = Arc::new(frozen_config);
        let frozen_cache_policy = {
            let mut rhs = LightyearRHS::new(
                init,
                0.0,
                jd0,
                Arc::clone(&frozen_config),
                Arc::clone(&packed),
            );
            rhs.adapt_cache_policy_for_eps(1e-11);
            rhs.cache_policy_for_test()
        };
        let frozen_vern9 = propagate(
            &frozen_config,
            frozen_cache_policy,
            StepperMethod::Vern9,
            1e-8,
        );
        let frozen_position_delta_km = error_norms(
            forward
                .get(1)
                .expect("sweep must contain the eps=1e-8 Vern9 endpoint"),
            &frozen_vern9,
        )
        .0;
        assert!(
            frozen_position_delta_km > 1e-8,
            "fixture did not distinguish continuous and frozen ephemeris"
        );
    }

    /// `SegmentControls` for the step-size-carry rule tests below. Only
    /// `boundary` and `dt_max` participate in the carry decision.
    fn hcarry_controls(boundary: SegmentBoundary, dt_max: f64) -> SegmentControls {
        SegmentControls {
            t0_s: 0.0,
            t_final_s: 300.0,
            eps: 1e-8,
            dt_max,
            force_eval: false,
            fast_single: false,
            max_steps: MAX_STEPS,
            max_rejects: 50,
            stepper: StepperMethod::Vern7,
            boundary,
        }
    }

    // The carry state is thread-local and every #[test] runs on its own
    // thread, so these tests cannot see each other's state — but each one
    // still opens with `hcarry_reset()` so it also cannot see a same-thread
    // predecessor if the harness ever reuses threads.

    #[test]
    fn hcarry_roundtrips_on_unclamped_rebased_legs_only() {
        hcarry_reset();
        hcarry_store(&hcarry_controls(SegmentBoundary::Rebased, 300.0), 77.5);
        assert_eq!(
            hcarry_take(&hcarry_controls(SegmentBoundary::Rebased, 300.0)),
            Some(77.5),
            "an unclamped Rebased leg must consume the stored exit h"
        );
        assert_eq!(
            hcarry_take(&hcarry_controls(SegmentBoundary::EventContinuation, 300.0)),
            None,
            "an event continuation must not consume the carry"
        );
        assert_eq!(
            hcarry_take(&hcarry_controls(SegmentBoundary::ArcStart, 300.0)),
            None,
            "an arc start has no predecessor and must not consume the carry"
        );
        assert_eq!(
            hcarry_take(&hcarry_controls(SegmentBoundary::Rebased, 10.0)),
            None,
            "a clamped root-refinement leg must not consume an unclamped h"
        );
    }

    #[test]
    fn hcarry_rejects_clamped_and_degenerate_stores() {
        hcarry_reset();
        hcarry_store(&hcarry_controls(SegmentBoundary::Rebased, 10.0), 9.0);
        assert_eq!(
            hcarry_take(&hcarry_controls(SegmentBoundary::Rebased, 300.0)),
            None,
            "a clamped leg's exit h must never seed an unclamped leg"
        );
        hcarry_store(&hcarry_controls(SegmentBoundary::Rebased, 300.0), 77.5);
        hcarry_store(&hcarry_controls(SegmentBoundary::Rebased, 300.0), 0.0);
        hcarry_store(&hcarry_controls(SegmentBoundary::Rebased, 300.0), f64::NAN);
        assert_eq!(
            hcarry_take(&hcarry_controls(SegmentBoundary::Rebased, 300.0)),
            Some(77.5),
            "zero and non-finite exit h must not clobber a valid carry"
        );
    }

    #[test]
    fn hcarry_reset_clears_the_carry() {
        hcarry_reset();
        hcarry_store(&hcarry_controls(SegmentBoundary::Rebased, 300.0), 77.5);
        hcarry_reset();
        assert_eq!(
            hcarry_take(&hcarry_controls(SegmentBoundary::Rebased, 300.0)),
            None,
            "a propagation-entry reset must clear the previous arc's carry"
        );
    }
}
