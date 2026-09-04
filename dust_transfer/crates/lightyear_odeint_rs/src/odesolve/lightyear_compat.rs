//! Lightyear-compatible integrator wrappers (DOPRI5 + absolute eps + force-eval alignment).
//!
//! THIS FILE HOLDS A SECOND PI STEP-SIZE CONTROLLER. `solver.rs` has the
//! other two (`integrate_internal` for explicit RK, `integrate_internal_esdirk`
//! for the implicit loop); the reject branch below calls itself "this third
//! copy" and means it. The Dopri5 tableau makes the duplication look benign --
//! `order_err: 4` and `err3: None` put the generic controller's
//! `alpha`/`beta`/`inv_order`/`max_growth` on exactly the literals here, and
//! safety, the accept and reject clamps and the `eps * 0.1` floor all match --
//! so a diff of the arithmetic reads clean and is not. The two loops disagree
//! on the first step, on `h_min`, on `h0` carry, on `k[0]` reuse after a
//! rejection, on the output-grid match tolerance, on where `eps` is floored,
//! on the PI memory after a rejection, on what `IntegrationStats::steps`
//! COUNTS, and on how much of `IntegrationStats` is written at all.
//!
//! `crates/lightyear_odeint_rs/src/odesolve/dopri5_compat_divergence_tests.rs`
//! runs both on one problem and pins
//! those differences, with the full list. It is the only thing in the
//! workspace that does: nothing else calls this module and `solver.rs` in the
//! same test. If you unify the loops, that file is meant to go red.
//!
//! DO NOT UNIFY AS A DRIVE-BY. `StepperMethod::Dopri5Compat` is
//! production-reachable -- the B500 screen arms select it by compiled
//! authority, `resolve_auto_stepper` returns it below `eps = 2e-8`, and
//! `py_config`'s empty method token lands on it -- though it is NOT the flown
//! stepper (the sealed campaign authority is Vern7). Changing this loop moves
//! its step sequence and therefore anything pinned on it, so it needs its own
//! re-baseline rather than a cleanup commit.

use crate::odesolve::solver::{
    all_finite, rk_step, sanitize_event, EventDecision, EventHandler, IntegrationEvent,
    IntegrationResult, IntegrationResultSampled, IntegrationStats, IntegrationStatus, Method,
    OdeSystem,
};

#[derive(Debug, Clone, Copy)]
pub struct LightyearConfig {
    pub eps: f64,
    pub dt_max: f64,
    pub max_steps: usize,
    pub max_rejects: usize,
    pub force_eval: bool,
    pub fast_single: bool,
}

impl Default for LightyearConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            dt_max: 60.0,
            max_steps: 50_000,
            max_rejects: 50,
            force_eval: false,
            fast_single: true,
        }
    }
}

/// Lightyear-compatible DOPRI5 integration with absolute error control and force-eval alignment.
pub fn integrate_lightyear_dopri5<S: OdeSystem>(
    system: &S,
    y0: &[f64],
    t_eval: &[f64],
    t0: f64,
    tf: f64,
    config: LightyearConfig,
    event_handler: Option<&mut dyn EventHandler>,
) -> IntegrationResultSampled {
    integrate_lightyear_dopri5_impl(system, y0, t_eval, t0, tf, config, event_handler, false)
}

/// DOPRI5 sampled output from accepted-step interpolation without forcing
/// steps onto the requested output grid.
pub fn integrate_lightyear_dopri5_unforced<S: OdeSystem>(
    system: &S,
    y0: &[f64],
    t_eval: &[f64],
    t0: f64,
    tf: f64,
    mut config: LightyearConfig,
    event_handler: Option<&mut dyn EventHandler>,
) -> IntegrationResultSampled {
    config.force_eval = false;
    integrate_lightyear_dopri5_impl(system, y0, t_eval, t0, tf, config, event_handler, true)
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "preserve established IEEE operation order in the compatibility integrator"
)]
#[expect(
    clippy::float_cmp,
    reason = "exact endpoint and direction comparisons define integration control flow"
)]
#[expect(
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "the compatibility kernel intentionally mirrors compact RK notation"
)]
fn integrate_lightyear_dopri5_impl<S: OdeSystem>(
    system: &S,
    y0: &[f64],
    t_eval: &[f64],
    t0: f64,
    tf: f64,
    config: LightyearConfig,
    mut event_handler: Option<&mut dyn EventHandler>,
    dense_unforced: bool,
) -> IntegrationResultSampled {
    let n = y0.len();
    if n == 0
        || t_eval.is_empty()
        || !t0.is_finite()
        || !tf.is_finite()
        || !config.eps.is_finite()
        || config.eps <= 0.0
    {
        return IntegrationResultSampled {
            times: Vec::new(),
            states: Vec::new(),
            n_state: n,
            status: IntegrationStatus::InvalidInput,
            stats: IntegrationStats::default(),
            event: None,
        };
    }

    // Floor eps to prevent rounding-dominated error estimates from destabilising
    // the step-size controller.  The generic solver paths already enforce this;
    // the Dopri5-compat path was missing it.
    let eps = config.eps.max(1e-12);

    let direction = if tf >= t0 { 1.0 } else { -1.0 };
    if !crate::odesolve::solver::is_sorted_dir(t_eval, direction) {
        return IntegrationResultSampled {
            times: Vec::new(),
            states: Vec::new(),
            n_state: n,
            status: IntegrationStatus::InvalidInput,
            stats: IntegrationStats::default(),
            event: None,
        };
    }

    let eval_tol = 1e-9;
    let (min_t, max_t) = if direction >= 0.0 { (t0, tf) } else { (tf, t0) };
    if t_eval.iter().any(|&time| time < min_t || time > max_t) {
        return IntegrationResultSampled {
            times: Vec::new(),
            states: Vec::new(),
            n_state: n,
            status: IntegrationStatus::InvalidInput,
            stats: IntegrationStats::default(),
            event: None,
        };
    }

    let mut stats = IntegrationStats::default();
    let Some(output_capacity) = t_eval.len().checked_mul(n) else {
        return IntegrationResultSampled {
            times: Vec::new(),
            states: Vec::new(),
            n_state: n,
            status: IntegrationStatus::InvalidInput,
            stats,
            event: None,
        };
    };
    let mut outputs: Vec<f64> = Vec::with_capacity(output_capacity);
    let mut times: Vec<f64> = Vec::with_capacity(t_eval.len().saturating_add(1));

    let fast_single = config.fast_single
        && event_handler.is_none()
        && t_eval.len() == 1
        && t_eval
            .first()
            .is_some_and(|time| time.to_bits() == tf.to_bits());

    let mut eval_idx = 0usize;

    if !fast_single {
        let dt_total_initial = tf - t0;
        let direction_initial = if dt_total_initial >= 0.0 { 1.0 } else { -1.0 };
        while t_eval
            .get(eval_idx)
            .is_some_and(|time| direction_initial * (*time - t0) < 0.0)
        {
            eval_idx += 1;
        }
    }

    let tableau = Method::Dopri5.tableau();
    let stages = tableau.stages;

    let mut y = y0.to_vec();
    let mut t = t0;
    let mut dy = vec![0.0; n];
    system.rhs(t, &y, &mut dy);
    stats.evals += 1;

    let dt_total = tf - t0;
    let dt_max_abs = config.dt_max.abs();
    let compat_h_min = 1e-10;
    let compat_h_max = if dt_max_abs.is_finite() {
        dt_max_abs.max(compat_h_min)
    } else {
        compat_h_min
    };
    // Respect configured max step from the very first step.
    // Without this, small-dt configurations (e.g. dt_max=0.1) could start with
    // an oversized first step (e.g. ~6s for a 600s span), destabilizing the
    // strict DOPRI5 compat path before adaptive control can shrink h.
    let mut h = (dt_total / 100.0).abs().clamp(1e-3, compat_h_max.max(1e-3)) * direction;
    let h_min = compat_h_min;

    let Some(stage_values) = stages.checked_mul(n) else {
        return IntegrationResultSampled {
            times,
            states: outputs,
            n_state: n,
            status: IntegrationStatus::InvalidInput,
            stats,
            event: None,
        };
    };
    let mut k: Vec<f64> = vec![0.0; stage_values];
    let mut y_tmp = vec![0.0; n];
    let mut y_next = vec![0.0; n];
    let mut err = vec![0.0; n];
    let mut err3 = vec![0.0; n];
    let mut primary_error_compensation = vec![0.0; n];
    let mut secondary_error_compensation = vec![0.0; n];
    let mut dense_sample = vec![0.0; n];

    let mut last_t = t;
    let mut last_y = y.clone();
    let mut rejects = 0usize;
    // PI step-size controller state (Gustafsson)
    let mut err_prev: f64 = 0.0;
    let mut have_err_prev = false;
    let mut just_rejected = false;

    // Collect any eval points that coincide with t0 before the main loop.
    // Without this, eval points at t0 are never matched because the accept
    // block only runs after a step has been taken.
    if !fast_single {
        while let Some(&eval_time) = t_eval
            .get(eval_idx)
            .filter(|time| time.to_bits() == t.to_bits())
        {
            times.push(eval_time);
            outputs.extend_from_slice(&y);
            eval_idx += 1;
        }
    }

    let mut t_comp: f64 = 0.0;

    while tf != t && (tf - t).signum() == direction {
        if stats.steps >= config.max_steps {
            return IntegrationResultSampled {
                times,
                states: outputs,
                n_state: n,
                status: IntegrationStatus::MaxStepsExceeded,
                stats,
                event: None,
            };
        }

        let h_remaining = tf - t;
        let mut h_step = if h.abs() < h_remaining.abs() {
            h
        } else {
            h_remaining
        };

        if config.force_eval && !fast_single && eval_idx < t_eval.len() {
            let Some(&next_eval) = t_eval.get(eval_idx) else {
                return IntegrationResultSampled {
                    times,
                    states: outputs,
                    n_state: n,
                    status: IntegrationStatus::InvalidInput,
                    stats,
                    event: None,
                };
            };
            let dt_to_eval = next_eval - t;
            if dt_to_eval.signum() == direction
                && dt_to_eval.abs() > eval_tol
                && dt_to_eval.abs() < h_step.abs()
            {
                h_step = dt_to_eval;
            }
        }

        let lands_on_tf = h_step.to_bits() == h_remaining.to_bits();

        let evals = match rk_step(
            system,
            tableau,
            t,
            &y,
            h_step,
            &mut k,
            &mut y_tmp,
            &mut y_next,
            &mut err,
            &mut err3,
            &mut primary_error_compensation,
            &mut secondary_error_compensation,
            Some(&dy),
            None,
            None,
        ) {
            Ok(evaluations) => evaluations,
            Err(step_status) => {
                return IntegrationResultSampled {
                    times,
                    states: outputs,
                    n_state: n,
                    status: step_status,
                    stats,
                    event: None,
                };
            }
        };
        stats.evals += evals;
        stats.steps += 1;

        if !all_finite(&y_next) || !all_finite(&err) {
            rejects += 1;
            system.on_step_reject();
            if rejects > config.max_rejects {
                return IntegrationResultSampled {
                    times,
                    states: outputs,
                    n_state: n,
                    status: IntegrationStatus::NonFiniteState,
                    stats,
                    event: None,
                };
            }
            h = (h_step.abs() * 0.1).clamp(h_min, compat_h_max) * direction;
            continue;
        }

        let mut max_err = 0.0;
        for &value in err.iter().take(n) {
            let e = value.abs();
            if e > max_err {
                max_err = e;
            }
        }

        if !max_err.is_finite() {
            rejects += 1;
            system.on_step_reject();
            if rejects > config.max_rejects {
                return IntegrationResultSampled {
                    times,
                    states: outputs,
                    n_state: n,
                    status: IntegrationStatus::NonFiniteState,
                    stats,
                    event: None,
                };
            }
            h = (h_step.abs() * 0.1).clamp(h_min, compat_h_max) * direction;
            continue;
        }

        if max_err <= eps || h_step.abs() <= h_min {
            let t_next = if lands_on_tf {
                tf
            } else {
                let kahan_y = h_step - t_comp;
                let next = t + kahan_y;
                t_comp = (next - t) - kahan_y;
                next
            };
            if !t_next.is_finite() {
                return IntegrationResultSampled {
                    times,
                    states: outputs,
                    n_state: n,
                    status: IntegrationStatus::NonFiniteState,
                    stats,
                    event: None,
                };
            }
            let Some(last_stage) = stages.checked_sub(1) else {
                return IntegrationResultSampled {
                    times,
                    states: outputs,
                    n_state: n,
                    status: IntegrationStatus::InvalidInput,
                    stats,
                    event: None,
                };
            };
            let Some(stage_start) = last_stage.checked_mul(n) else {
                return IntegrationResultSampled {
                    times,
                    states: outputs,
                    n_state: n,
                    status: IntegrationStatus::InvalidInput,
                    stats,
                    event: None,
                };
            };
            let Some(dy_next) = k.get(stage_start..stage_values) else {
                return IntegrationResultSampled {
                    times,
                    states: outputs,
                    n_state: n,
                    status: IntegrationStatus::InvalidInput,
                    stats,
                    event: None,
                };
            };
            rejects = 0;

            if let Some(handler) = event_handler.as_deref_mut() {
                match handler.on_step(t, &y, &dy, t_next, &y_next, dy_next) {
                    EventDecision::Continue => {}
                    EventDecision::Stop { t_event, y_event } => {
                        let sanitized =
                            sanitize_event(t, t_next, &y, &y_next, t_event, y_event, direction);
                        let (t_event, y_event, method, error) = match sanitized {
                            Ok(v) => v,
                            Err(event_status) => {
                                return IntegrationResultSampled {
                                    times,
                                    states: outputs,
                                    n_state: n,
                                    status: event_status,
                                    stats,
                                    event: None,
                                };
                            }
                        };
                        let event = IntegrationEvent {
                            t: t_event,
                            y: y_event.clone(),
                            interp_method: method,
                            interp_error: error,
                        };
                        if times
                            .last()
                            .is_none_or(|last| (t_event - *last).abs() > eval_tol)
                        {
                            times.push(t_event);
                            outputs.extend_from_slice(&y_event);
                        }
                        return IntegrationResultSampled {
                            times,
                            states: outputs,
                            n_state: n,
                            status: IntegrationStatus::EventTriggered,
                            stats,
                            event: Some(event),
                        };
                    }
                }
            }

            if dense_unforced && !fast_single {
                while let Some(&sample_t) = t_eval
                    .get(eval_idx)
                    .filter(|time| direction * (**time - t_next) <= 0.0)
                {
                    times.push(sample_t);
                    if sample_t.to_bits() == t_next.to_bits() {
                        outputs.extend_from_slice(&y_next);
                    } else {
                        let h_dense = t_next - t;
                        let tau = ((sample_t - t) / h_dense).clamp(0.0, 1.0);
                        let tau2 = tau * tau;
                        let tau3 = tau2 * tau;
                        let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
                        let h10 = tau3 - 2.0 * tau2 + tau;
                        let h01 = -2.0 * tau3 + 3.0 * tau2;
                        let h11 = tau3 - tau2;
                        for (
                            (((sample, &previous), &previous_derivative), &next),
                            &next_derivative,
                        ) in dense_sample
                            .iter_mut()
                            .zip(&y)
                            .zip(&dy)
                            .zip(&y_next)
                            .zip(dy_next)
                        {
                            *sample = h00 * previous
                                + h10 * h_dense * previous_derivative
                                + h01 * next
                                + h11 * h_dense * next_derivative;
                        }
                        outputs.extend_from_slice(&dense_sample);
                    }
                    eval_idx += 1;
                }
            } else if !fast_single {
                while let Some(&eval_time) = t_eval
                    .get(eval_idx)
                    .filter(|time| (**time - t_next).abs() < eval_tol)
                {
                    times.push(eval_time);
                    outputs.extend_from_slice(&y_next);
                    eval_idx += 1;
                }
            }

            y.copy_from_slice(&y_next);
            t = t_next;
            last_t = t;
            if !fast_single {
                last_y.clone_from(&y);
            }
            dy.copy_from_slice(dy_next);

            // PI step-size controller (Gustafsson) on accept.
            // Guard against zero/underflow error: when max_err is exactly zero
            // the PI ratio degenerates.  Use eps*0.1 as a floor so growth
            // stays bounded instead of unconditional.
            let eff_err = if max_err > 0.0 { max_err } else { eps * 0.1 };
            {
                let factor = if have_err_prev && err_prev > 0.0 {
                    let alpha = 0.7 / 5.0; // order ~5 for DOPRI5
                    let beta = 0.4 / 5.0;
                    0.9 * (eps / eff_err).powf(alpha) * (err_prev / eps).powf(beta)
                } else {
                    0.9 * (eps / eff_err).powf(0.2)
                };
                let max_growth = if just_rejected { 2.0 } else { 5.0 };
                h = (h_step.abs() * factor.clamp(0.2, max_growth)).clamp(h_min, compat_h_max)
                    * direction;
            }
            err_prev = max_err;
            have_err_prev = true;
            just_rejected = false;
        } else {
            rejects += 1;
            system.on_step_reject();
            if rejects > config.max_rejects {
                return IntegrationResultSampled {
                    times,
                    states: outputs,
                    n_state: n,
                    status: IntegrationStatus::MaxRejectsExceeded,
                    stats,
                    event: None,
                };
            }
            let factor = 0.9 * (eps / max_err).powf(0.2);
            h = (h_step.abs() * factor.clamp(0.1, 0.5)).clamp(h_min, compat_h_max) * direction;
            // KNOWN DEFECT, DELIBERATELY NOT FIXED HERE.
            //
            // Seeding the PI memory from a REJECTED step is wrong — the I-term
            // then differences errors measured at two different step sizes, and
            // Hairer's DOPRI5/DOP853 leave `facold` untouched on rejection for
            // that reason. `solver.rs` was corrected on both its loops (see the
            // long note on the explicit-RK reject branch there); this third copy
            // was left alone on purpose.
            //
            // Why: this is the bit-compatibility path. Changing the step
            // sequence here would move any DOPRI5-compat pinned result, and the
            // point of this loop is to reproduce the legacy stepper, not to be
            // the best controller available. Fixing it is a deliberate
            // behaviour change that needs its own re-baseline, not a drive-by.
            //
            // `saturated_steps` on `IntegrationStats` is likewise not
            // instrumented here and stays 0 on this path.
            err_prev = max_err;
            have_err_prev = true;
            just_rejected = true;
        }
    }

    if fast_single && (tf - last_t).abs() <= 1e-11 {
        if let Some(&eval_time) = t_eval.first() {
            times.push(eval_time);
        }
        outputs.extend_from_slice(&y);
    } else if times.is_empty() {
        times.push(tf);
        outputs.extend_from_slice(&last_y);
    }

    IntegrationResultSampled {
        times,
        states: outputs,
        n_state: n,
        status: IntegrationStatus::Success,
        stats,
        event: None,
    }
}

/// Lightyear-compatible DOPRI5 integration that returns only the final state.
pub fn integrate_lightyear_dopri5_final<S: OdeSystem>(
    system: &S,
    y0: &[f64],
    t0: f64,
    tf: f64,
    config: LightyearConfig,
    event_handler: Option<&mut dyn EventHandler>,
) -> IntegrationResult {
    let t_eval = [tf];
    let result = integrate_lightyear_dopri5(system, y0, &t_eval, t0, tf, config, event_handler);
    let n = y0.len();
    let t = *result.times.last().unwrap_or(&tf);
    let final_state = if result.states.len() == n {
        result.states
    } else if result.states.len() >= n {
        let offset = result.states.len().saturating_sub(n);
        result.states.get(offset..).unwrap_or_default().to_vec()
    } else {
        vec![0.0; n]
    };

    IntegrationResult {
        t,
        y: final_state,
        status: result.status,
        stats: result.stats,
        event: result.event,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct Linear;

    impl OdeSystem for Linear {
        fn rhs(&self, _t: f64, _y: &[f64], dy: &mut [f64]) {
            if let Some(derivative) = dy.first_mut() {
                *derivative = 1.0;
            }
        }
    }

    struct Exponential;

    impl OdeSystem for Exponential {
        fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) {
            let (Some(&state), Some(derivative)) = (y.first(), dy.first_mut()) else {
                return;
            };
            *derivative = state;
        }
    }

    struct Continue;

    impl EventHandler for Continue {
        fn on_step(
            &mut self,
            _prev_t: f64,
            _prev_y: &[f64],
            _prev_dy: &[f64],
            _next_t: f64,
            _next_y: &[f64],
            _next_dy: &[f64],
        ) -> EventDecision {
            EventDecision::Continue
        }
    }

    #[derive(Default)]
    struct EndpointRecorder {
        first_next_t: Cell<Option<f64>>,
        last_next_t: Cell<Option<f64>>,
    }

    impl EventHandler for EndpointRecorder {
        fn on_step(
            &mut self,
            _prev_t: f64,
            _prev_y: &[f64],
            _prev_dy: &[f64],
            next_t: f64,
            _next_y: &[f64],
            _next_dy: &[f64],
        ) -> EventDecision {
            if self.first_next_t.get().is_none() {
                self.first_next_t.set(Some(next_t));
            }
            self.last_next_t.set(Some(next_t));
            EventDecision::Continue
        }
    }

    #[test]
    fn compat_terminal_callback_reports_exact_requested_endpoint() {
        let start = 27.0;
        let end = 54.451_202_197_643_7;
        let mut handler = EndpointRecorder::default();
        let result = integrate_lightyear_dopri5_final(
            &Linear,
            &[2.0],
            start,
            end,
            LightyearConfig {
                eps: 1.0,
                dt_max: 0.013_4,
                ..LightyearConfig::default()
            },
            Some(&mut handler),
        );

        assert_eq!(result.status, IntegrationStatus::Success);
        assert_eq!(
            handler.last_next_t.get().map(f64::to_bits),
            Some(end.to_bits())
        );
    }

    #[test]
    fn compat_forced_interior_sample_is_not_labeled_as_terminal() {
        let interior = 0.005_f64;
        let end = 1.0_f64;
        let samples = [0.0, interior, end];
        let mut handler = EndpointRecorder::default();
        let result = integrate_lightyear_dopri5(
            &Linear,
            &[2.0],
            &samples,
            0.0,
            end,
            LightyearConfig {
                eps: 1.0,
                dt_max: 2.0,
                force_eval: true,
                ..LightyearConfig::default()
            },
            Some(&mut handler),
        );

        assert_eq!(result.status, IntegrationStatus::Success);
        assert_eq!(
            handler.first_next_t.get().map(f64::to_bits),
            Some(interior.to_bits())
        );
        assert_eq!(
            handler.last_next_t.get().map(f64::to_bits),
            Some(end.to_bits())
        );
    }

    #[test]
    fn unforced_compat_ignores_hostile_force_eval_flag() {
        let samples = [0.0, 0.13, 0.71, 1.37, 2.0];
        let config = LightyearConfig {
            dt_max: 0.2,
            fast_single: false,
            ..LightyearConfig::default()
        };
        let expected =
            integrate_lightyear_dopri5_unforced(&Linear, &[2.0], &samples, 0.0, 2.0, config, None);
        let hostile = integrate_lightyear_dopri5_unforced(
            &Linear,
            &[2.0],
            &samples,
            0.0,
            2.0,
            LightyearConfig {
                force_eval: true,
                ..config
            },
            None,
        );
        assert_eq!(hostile.stats.steps, expected.stats.steps);
        assert_eq!(hostile.stats.evals, expected.stats.evals);
        assert_eq!(hostile.times, expected.times);
        assert_eq!(hostile.states, expected.states);
    }

    #[test]
    fn unforced_compat_preserves_near_endpoint_state_and_ignores_force_eval() {
        // DOPRI5 compat formerly treated this public time as equal to the
        // hidden segment endpoint because its sample tolerance was 1 ns.
        let near_start = 5.0e-10;
        let near_final = 1.0 - 5.0e-10;
        let samples = [0.0, near_start, 0.4, near_final, 1.0];
        let config = LightyearConfig {
            dt_max: 0.2,
            fast_single: false,
            ..LightyearConfig::default()
        };
        let run = |force_eval| {
            integrate_lightyear_dopri5_unforced(
                &Linear,
                &[2.0],
                &samples,
                0.0,
                1.0,
                LightyearConfig {
                    force_eval,
                    ..config
                },
                None,
            )
        };

        let unforced = run(false);
        let hostile = run(true);
        assert_eq!(unforced.status, IntegrationStatus::Success);
        assert_eq!(
            unforced
                .times
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>(),
            samples
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>()
        );
        let start_state = unforced.states.first().copied().unwrap_or(f64::NAN);
        let near_start_state = unforced.states.get(1).copied().unwrap_or(f64::NAN);
        let near_final_state = unforced.states.get(3).copied().unwrap_or(f64::NAN);
        let final_state = unforced.states.get(4).copied().unwrap_or(f64::NAN);
        assert!((near_start_state - (2.0 + near_start)).abs() < 1.0e-10);
        assert!((near_final_state - (2.0 + near_final)).abs() < 1.0e-10);
        assert_ne!(start_state.to_bits(), near_start_state.to_bits());
        assert_ne!(near_final_state.to_bits(), final_state.to_bits());
        assert_eq!(hostile.stats.steps, unforced.stats.steps);
        assert_eq!(hostile.stats.evals, unforced.stats.evals);
        assert_eq!(hostile.times, unforced.times);
        assert_eq!(hostile.states, unforced.states);
    }

    #[test]
    fn unforced_compat_returns_exact_endpoint_after_hidden_start() {
        let samples = [0.0, 1.0];
        let mut handler = Continue;
        let result = integrate_lightyear_dopri5_unforced(
            &Linear,
            &[2.0],
            &samples,
            0.0,
            1.0,
            LightyearConfig {
                eps: 1.0e-8,
                dt_max: 60.0,
                fast_single: false,
                ..LightyearConfig::default()
            },
            Some(&mut handler),
        );

        assert_eq!(result.status, IntegrationStatus::Success);
        assert_eq!(
            result
                .times
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>(),
            samples
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(result.states.len(), samples.len());
        assert!((result.states.last().copied().unwrap_or(f64::NAN) - 3.0).abs() < 1.0e-10);
    }

    #[test]
    fn unforced_compat_returns_endpoint_after_adaptive_tail() {
        let samples = [0.0, 1.0];
        let result = integrate_lightyear_dopri5_unforced(
            &Exponential,
            &[1.0],
            &samples,
            0.0,
            1.0,
            LightyearConfig {
                eps: 1.0e-8,
                dt_max: 60.0,
                fast_single: false,
                ..LightyearConfig::default()
            },
            None,
        );

        assert_eq!(result.status, IntegrationStatus::Success);
        assert_eq!(
            result
                .times
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>(),
            samples
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(result.states.len(), samples.len());
        assert!(
            (result.states.last().copied().unwrap_or(f64::NAN) - std::f64::consts::E).abs()
                < 1.0e-7
        );
    }
}
