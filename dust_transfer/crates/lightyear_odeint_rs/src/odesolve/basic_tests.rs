//! Behaviour of the standalone ODE solvers.
//!
//! Inline since `odesolve_lightyear` was absorbed into this crate: everything
//! it names is private to `crate::odesolve` now, so a `tests/` binary cannot
//! reach it without republishing the whole solver surface.

use crate::odesolve::{
    integrate_final, integrate_final_with_events, ErrorControl, EventDecision, EventHandler,
    IntegrationStatus, IntegratorConfig, Method, OdeSystem,
};
use approx::assert_relative_eq;
use std::cell::{Cell, RefCell};

struct ExpSystem;

impl OdeSystem for ExpSystem {
    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) {
        let (Some(&state), Some(derivative)) = (y.first(), dy.first_mut()) else {
            return;
        };
        *derivative = state;
    }
}

#[test]
fn vern9_observed_order_matches_method() {
    let step_sizes = [1.0, 0.5, 0.25, 0.125];
    let expected_steps = [4, 8, 16, 32];
    let exact = 4.0_f64.exp();
    let roundoff_guard = 128.0 * f64::EPSILON * exact;
    let mut errors = [0.0; 4];

    for ((&step_size, &step_count), error_slot) in step_sizes
        .iter()
        .zip(expected_steps.iter())
        .zip(errors.iter_mut())
    {
        let cfg = IntegratorConfig {
            error_control: ErrorControl::Absolute { eps: 1.0 },
            h0: Some(step_size),
            h_min: step_size,
            h_max: step_size,
            ..Default::default()
        };
        let result = integrate_final(&ExpSystem, Method::Vern9, &[1.0], 0.0, 4.0, cfg);

        assert_eq!(result.status, IntegrationStatus::Success, "h={step_size}");
        assert_eq!(result.t.to_bits(), 4.0_f64.to_bits(), "h={step_size}");
        assert_eq!(result.stats.steps, step_count, "h={step_size}");
        let endpoint = result.y.first().copied().unwrap_or(f64::NAN);
        *error_slot = (endpoint - exact).abs();
        assert!(error_slot.is_finite(), "h={step_size}");
    }

    let mut qualifying_pairs = 0;
    for (error_pair, step_pair) in errors.windows(2).zip(step_sizes.windows(2)) {
        let [coarse_error, fine_error] = error_pair else {
            continue;
        };
        let [coarse_step_size, _] = step_pair else {
            continue;
        };
        assert!(
            coarse_error > fine_error,
            "endpoint error did not decrease: h={coarse_step_size} error={coarse_error}, h/2 error={fine_error}"
        );
        if coarse_error > &roundoff_guard && fine_error > &roundoff_guard {
            let observed_order = (*coarse_error / *fine_error).log2();
            assert!(
                (8.0..=10.5).contains(&observed_order),
                "h={coarse_step_size} observed order={observed_order}"
            );
            qualifying_pairs += 1;
        }
    }
    assert!(
        qualifying_pairs >= 2,
        "only {qualifying_pairs} pairs exceeded roundoff guard {roundoff_guard}"
    );
}

#[test]
fn test_exp_growth_tsit5() {
    let sys = ExpSystem;
    let cfg = IntegratorConfig {
        error_control: ErrorControl::Scaled {
            rtol: 1e-10,
            atol: 1e-12,
        },
        ..Default::default()
    };
    let result = integrate_final(&sys, Method::Tsit5, &[1.0], 0.0, 1.0, cfg);
    assert_relative_eq!(
        result.y.first().copied().unwrap_or(f64::NAN),
        std::f64::consts::E,
        epsilon = 1e-6
    );
}

#[test]
fn test_exp_growth_dop853() {
    let sys = ExpSystem;
    let cfg = IntegratorConfig {
        error_control: ErrorControl::Scaled {
            rtol: 1e-12,
            atol: 1e-12,
        },
        ..Default::default()
    };
    let result = integrate_final(&sys, Method::Dop853, &[1.0], 0.0, 1.0, cfg);
    assert_relative_eq!(
        result.y.first().copied().unwrap_or(f64::NAN),
        std::f64::consts::E,
        epsilon = 1e-8
    );
}

#[test]
fn test_exp_growth_rkv98() {
    let sys = ExpSystem;
    let cfg = IntegratorConfig {
        error_control: ErrorControl::Scaled {
            rtol: 1e-12,
            atol: 1e-12,
        },
        ..Default::default()
    };
    let result = integrate_final(&sys, Method::Rkv98, &[1.0], 0.0, 1.0, cfg);
    assert_relative_eq!(
        result.y.first().copied().unwrap_or(f64::NAN),
        std::f64::consts::E,
        epsilon = 1e-6
    );
}

#[test]
fn test_exp_growth_dopri5_absolute() {
    let sys = ExpSystem;
    let cfg = IntegratorConfig {
        error_control: ErrorControl::Absolute { eps: 1e-8 },
        ..Default::default()
    };
    let result = integrate_final(&sys, Method::Dopri5, &[1.0], 0.0, 1.0, cfg);
    assert_relative_eq!(
        result.y.first().copied().unwrap_or(f64::NAN),
        std::f64::consts::E,
        epsilon = 1e-5
    );
}

struct LinearSystem;

impl OdeSystem for LinearSystem {
    fn rhs(&self, _t: f64, _y: &[f64], dy: &mut [f64]) {
        if let Some(derivative) = dy.first_mut() {
            *derivative = 1.0;
        }
    }
}

struct StopAtHalf;

impl EventHandler for StopAtHalf {
    fn on_step(
        &mut self,
        prev_t: f64,
        prev_y: &[f64],
        _prev_dy: &[f64],
        next_t: f64,
        next_y: &[f64],
        _next_dy: &[f64],
    ) -> EventDecision {
        let (Some(&previous_state), Some(&next_state)) = (prev_y.first(), next_y.first()) else {
            return EventDecision::Continue;
        };
        if previous_state < 0.5 && next_state >= 0.5 {
            let denom = next_state - previous_state;
            let frac = if denom == 0.0 {
                0.0
            } else {
                (0.5 - previous_state) / denom
            };
            let t_event = prev_t + frac * (next_t - prev_t);
            return EventDecision::Stop {
                t_event,
                y_event: vec![0.5],
            };
        }
        EventDecision::Continue
    }
}

#[test]
fn test_event_stop() {
    let sys = LinearSystem;
    let cfg = IntegratorConfig {
        error_control: ErrorControl::Scaled {
            rtol: 1e-9,
            atol: 1e-12,
        },
        h0: Some(1.0),
        h_max: 1.0,
        ..Default::default()
    };
    let mut handler = StopAtHalf;
    let result =
        integrate_final_with_events(&sys, Method::Tsit5, &[0.0], 0.0, 2.0, cfg, &mut handler);
    assert!(matches!(result.status, IntegrationStatus::EventTriggered));
    assert_relative_eq!(result.t, 0.5, epsilon = 1e-6);
    assert_relative_eq!(
        result.y.first().copied().unwrap_or(f64::NAN),
        0.5,
        epsilon = 1e-6
    );
}

struct NonFiniteSystem;

impl OdeSystem for NonFiniteSystem {
    fn rhs(&self, _t: f64, _y: &[f64], dy: &mut [f64]) {
        if let Some(derivative) = dy.first_mut() {
            *derivative = f64::NAN;
        }
    }
}

#[test]
fn test_nonfinite_state() {
    let sys = NonFiniteSystem;
    let cfg = IntegratorConfig {
        max_rejects: 2,
        ..Default::default()
    };
    let result = integrate_final(&sys, Method::Tsit5, &[1.0], 0.0, 1.0, cfg);
    assert!(matches!(result.status, IntegrationStatus::NonFiniteState));
}

#[test]
fn test_max_rejects_exceeded() {
    let sys = ExpSystem;
    let cfg = IntegratorConfig {
        error_control: ErrorControl::Absolute { eps: 1e-20 },
        h0: Some(1.0),
        h_min: 1e-12,
        h_max: 1.0,
        max_steps: 10,
        max_rejects: 0,
        ..Default::default()
    };
    let result = integrate_final(&sys, Method::Tsit5, &[1.0], 0.0, 1.0, cfg);
    assert!(matches!(
        result.status,
        IntegrationStatus::MaxRejectsExceeded
    ));
}

#[derive(Default)]
struct CallHistorySystem {
    calls: Cell<u32>,
    base_derivatives: RefCell<Vec<f64>>,
}

impl OdeSystem for CallHistorySystem {
    fn rhs(&self, t: f64, y: &[f64], dy: &mut [f64]) {
        let call = self.calls.get().saturating_add(1);
        self.calls.set(call);
        let derivative = y.first().copied().unwrap_or(0.0) + f64::from(call);
        if t.to_bits() == 0.0_f64.to_bits() {
            self.base_derivatives.borrow_mut().push(derivative);
        }
        if let Some(slot) = dy.first_mut() {
            *slot = derivative;
        }
    }
}

#[test]
fn rejected_vern7_step_recomputes_stage_zero_after_later_rhs_history() {
    let system = CallHistorySystem::default();
    let result = integrate_final(
        &system,
        Method::Vern7,
        &[1.0],
        0.0,
        1.0,
        IntegratorConfig {
            error_control: ErrorControl::Absolute { eps: 1e-20 },
            h0: Some(1.0),
            h_min: 1e-12,
            h_max: 1.0,
            max_steps: 10,
            max_rejects: 1,
            force_eval: false,
        },
    );

    assert!(
        result.stats.rejected_steps >= 1,
        "fixture must reject at least one Vern7 attempt"
    );
    let derivatives = system.base_derivatives.borrow();
    let [first, second, ..] = derivatives.as_slice() else {
        panic!(
            "retry must recompute stage zero at identical (t,y) after later stages changed RHS history; observed {derivatives:?}"
        );
    };
    assert_ne!(
        first.to_bits(),
        second.to_bits(),
        "fixture must expose call-history-dependent derivatives at identical (t,y)"
    );
}

struct BadEvent;

impl EventHandler for BadEvent {
    fn on_step(
        &mut self,
        _prev_t: f64,
        _prev_y: &[f64],
        _prev_dy: &[f64],
        next_t: f64,
        _next_y: &[f64],
        _next_dy: &[f64],
    ) -> EventDecision {
        EventDecision::Stop {
            t_event: next_t + 1.0,
            y_event: vec![1.0, 2.0],
        }
    }
}

#[test]
fn test_event_sanitize() {
    let sys = LinearSystem;
    let cfg = IntegratorConfig {
        h0: Some(0.5),
        h_max: 0.5,
        ..Default::default()
    };
    let mut handler = BadEvent;
    let result =
        integrate_final_with_events(&sys, Method::Tsit5, &[0.0], 0.0, 1.0, cfg, &mut handler);
    assert!(matches!(result.status, IntegrationStatus::EventTriggered));
    let event = result.event.expect("event");
    assert_eq!(
        event.interp_method,
        crate::odesolve::SanitizedInterp::LinearClamp
    );
}

// ---------------------------------------------------------------------------
// Adaptive step-size controller.
// ---------------------------------------------------------------------------

/// A REJECTED step must not seed the PI controller's error memory.
///
/// # The defect this pins
///
/// `err_prev` feeds the I-term `(err_prev/eps)^beta` of the Gustafsson PI law,
/// which is meant to estimate the trend of the error sequence along the
/// ACCEPTED trajectory. The reject branch used to write `err_prev = err_norm;
/// have_err_prev = true`, so the next accepted step differenced an error
/// measured at `h` against one measured at `0.1h..0.5h`. That is not a trend —
/// it is mostly the `h^(p+1)` scaling of the step reduction itself. Hairer's
/// reference DOPRI5 and DOP853 both leave `facold` untouched on rejection for
/// exactly this reason.
///
/// The bias ran in the wrong direction: it inflated `h` immediately after a
/// rejection, which is the one moment the controller should be conservative.
///
/// # Why a step COUNT is a valid detector here
///
/// `h0 = h_max = 2.0` against `eps = 1e-14` guarantees the first step is
/// rejected, so the reject branch is exercised and the difference is
/// observable in the step sequence. Measured on this fixture:
///
/// | | steps | evals | endpoint error |
/// |---|---|---|---|
/// | seeding from rejects (defect) | 45 | 752 | 4.263e-14 |
/// | not seeding (fixed) | **47** | **784** | **3.553e-14** |
///
/// The fix costs 2 steps (+4.4%) and returns 17% less endpoint error, which is
/// the expected shape of the trade: the old controller was taking larger steps
/// after a rejection than the error justified.
///
/// Both numbers are IDENTICAL under debug and `--release`. That is not
/// automatic for a step count — an ulp-level change can flip an accept/reject
/// decision and cascade — so it was checked rather than assumed. If this test
/// starts failing under one profile only, suspect the fixture, not the fix.
///
/// WHAT THIS DOES NOT ASSERT: that 47 is optimal. It pins the controller's
/// behaviour so the seeding cannot silently return.
#[test]
fn rejected_steps_do_not_seed_the_pi_controller_memory() {
    let cfg = IntegratorConfig {
        error_control: ErrorControl::Absolute { eps: 1e-14 },
        h0: Some(2.0),
        h_min: 1e-12,
        h_max: 2.0,
        ..Default::default()
    };
    let result = integrate_final(&ExpSystem, Method::Vern9, &[1.0], 0.0, 4.0, cfg);

    assert_eq!(result.status, IntegrationStatus::Success);
    assert_eq!(
        result.stats.steps, 47,
        "step sequence moved. 45 is the signature of the PI memory being seeded \
         from rejected steps again; anything else means the controller or this \
         fixture changed and the table in this test's doc must be re-measured."
    );
    assert_eq!(
        result.stats.underflow_accepts, 0,
        "no step should have been force-accepted at h_min here"
    );

    let exact = 4.0_f64.exp();
    let error = (result.y.first().copied().unwrap_or(f64::NAN) - exact).abs();
    assert!(
        error < 1e-13,
        "endpoint error {error:e} exceeds 1e-13 (measured 3.553e-14)"
    );
}

/// Steps force-accepted at `h_min` are COUNTED, not silently swallowed.
///
/// # The condition
///
/// The accept test is `err_norm <= accept_threshold || h_step.abs() <= h_min`.
/// The second disjunct takes a step whose error test FAILED, because `h` cannot
/// be reduced any further. The run then returns `IntegrationStatus::Success`
/// with a state that does not meet the requested tolerance.
///
/// It is worse than a single bad step: the accept branch resets `rejects = 0`,
/// so `max_rejects` — the guard that would otherwise surface this — can never
/// fire once the integration is pinned at `h_min`. Before
/// `underflow_accepts` existed there was nothing anywhere in the returned
/// `IntegrationResult` to distinguish this from a clean solve.
///
/// # The fixture
///
/// `h_min = h_max = 1.0` pins the step, and `eps = 1e-30` is unsatisfiable at
/// that step size, so every one of the 10 steps takes the force-accept path.
/// The assertion `underflow_accepts == steps` is what makes this a test of the
/// COUNTER rather than of the condition.
///
/// # This is instrumentation, not a live defect
///
/// Production sets `h_min = 1e-12` s everywhere. For Vern9 to fail the error
/// test at `h = 1e-12` the ninth derivative would have to be around `1e100`, so
/// the path is unreachable for any non-pathological RHS — which is why it is
/// counted rather than made terminal. Making it terminal would risk turning a
/// survivable hiccup into a failed propagation for a case that does not occur.
/// A nonzero count in the field means the RHS has gone badly non-smooth and the
/// diagnosis belongs there.
#[test]
fn steps_force_accepted_at_h_min_are_counted() {
    let cfg = IntegratorConfig {
        error_control: ErrorControl::Absolute { eps: 1e-30 },
        h0: Some(1.0),
        h_min: 1.0,
        h_max: 1.0,
        ..Default::default()
    };
    let result = integrate_final(&ExpSystem, Method::Vern9, &[1.0], 0.0, 10.0, cfg);

    // The point of the test: it "succeeds" while violating the tolerance.
    assert_eq!(result.status, IntegrationStatus::Success);
    assert_eq!(result.stats.steps, 10);
    assert_eq!(
        result.stats.underflow_accepts, result.stats.steps,
        "every step here fails the error test and is accepted only because \
         h == h_min; if this is not counted the violation is invisible to the \
         caller"
    );
}

/// The `saturated_steps` counter distinguishes the cap from the controller.
///
/// Companion to `production_band_saturation_is_measured_not_assumed` in
/// `lightyear_odeint_rs`, which asserts the same property on the real force
/// model. This one is the unit-level proof that the counter means what it says,
/// on a problem whose answer is known by construction.
///
/// `ExpSystem` at `eps = 1.0` is trivially accurate, so the controller always
/// demands growth and every accepted step is clipped by `h_max`. Tightening to
/// `eps = 1e-14` with a large `h_max` puts the controller in charge instead.
/// Asserting BOTH directions is what rules out a counter stuck at full scale or
/// stuck at zero.
#[test]
fn saturated_steps_separates_the_cap_from_the_controller() {
    // Cap-bound: loose tolerance, tight cap.
    let capped = integrate_final(
        &ExpSystem,
        Method::Vern9,
        &[1.0],
        0.0,
        4.0,
        IntegratorConfig {
            error_control: ErrorControl::Absolute { eps: 1.0 },
            h0: Some(0.25),
            h_min: 1e-12,
            h_max: 0.25,
            ..Default::default()
        },
    );
    assert_eq!(capped.status, IntegrationStatus::Success);
    assert_eq!(
        capped.stats.saturated_steps, capped.stats.steps,
        "at eps=1.0 with h_max=0.25 the cap must set every step"
    );

    // Controller-bound: tight tolerance, slack cap.
    let controlled = integrate_final(
        &ExpSystem,
        Method::Vern9,
        &[1.0],
        0.0,
        4.0,
        IntegratorConfig {
            error_control: ErrorControl::Absolute { eps: 1e-14 },
            h0: Some(0.25),
            h_min: 1e-12,
            h_max: 4.0,
            ..Default::default()
        },
    );
    assert_eq!(controlled.status, IntegrationStatus::Success);
    assert_eq!(
        controlled.stats.saturated_steps, 0,
        "at eps=1e-14 with h_max=4.0 the error controller, not the cap, must \
         set every step"
    );
}

/// A narrow pulse partway through the arc, so rejections occur AFTER several
/// steps have already been accepted.
///
/// `ExpSystem` cannot exercise that: with `h0 = h_max` its very first step is
/// rejected, so `have_err_prev` is false at every rejection and the reject
/// branch's treatment of the PI memory is unobservable. This system is smooth
/// until `t = 3`, by which point a real accepted-error history exists.
struct BumpSystem;

impl OdeSystem for BumpSystem {
    fn rhs(&self, t: f64, _y: &[f64], dy: &mut [f64]) {
        if let Some(derivative) = dy.first_mut() {
            *derivative = (-4000.0 * (t - 3.0) * (t - 3.0)).exp();
        }
    }
}

/// The reject branch must leave the PI memory ALONE, not clear it.
///
/// # Two different fixes, and this pins which one is in force
///
/// Seeding `err_prev` from a rejected step is wrong (see
/// `rejected_steps_do_not_seed_the_pi_controller_memory`). But there are two
/// ways to stop doing it, and they are not equivalent:
///
/// - **Leave both `err_prev` and `have_err_prev` untouched.** The next accepted
///   step keeps using the PI form with the last ACCEPTED error as its trend
///   anchor. This is what Hairer's `dopri5.f` and `dop853.f` do — `FACOLD` is
///   assigned only in the step-accepted branch — and it is what this solver
///   does.
/// - **Set `have_err_prev = false`.** The next accepted step falls back to the
///   I-free form `0.9*(eps/err)^(1/order)`. This also removes the defect, but it
///   throws away a still-valid anchor: it is the REJECTED error that is
///   meaningless, not the last accepted one.
///
/// On this fixture they diverge — **54 steps for the Hairer form against 55 for
/// the flag-clearing form**, with different endpoint bits. So the distinction is
/// load-bearing rather than stylistic, which is the whole reason this test
/// exists alongside the other one.
///
/// The non-finite and Newton-failure branches DO clear `have_err_prev`, and
/// that remains correct: those abandon a step whose error estimate is
/// meaningless (NaN, or no converged Newton solution), so there is no
/// trustworthy history to carry forward. Here the history is fine and only the
/// current step failed.
#[test]
fn the_reject_branch_preserves_the_last_accepted_error_as_the_pi_anchor() {
    let cfg = IntegratorConfig {
        error_control: ErrorControl::Absolute { eps: 1e-13 },
        h0: Some(1.0),
        h_min: 1e-12,
        h_max: 1.0,
        ..Default::default()
    };
    let result = integrate_final(&BumpSystem, Method::Vern9, &[1.0], 0.0, 5.0, cfg);

    assert_eq!(result.status, IntegrationStatus::Success);
    assert_eq!(
        result.stats.steps, 54,
        "step sequence moved. 55 is the signature of the reject branch clearing \
         `have_err_prev` instead of leaving the PI memory untouched; Hairer's \
         reference keeps the last ACCEPTED error as the trend anchor."
    );

    // The exact solution is 1 + integral of the Gaussian pulse over [0, 5],
    // which is sqrt(pi/4000) to well under the tolerance since the pulse is
    // fully contained in the interval.
    let exact = 1.0 + (std::f64::consts::PI / 4000.0).sqrt();
    let error = (result.y.first().copied().unwrap_or(f64::NAN) - exact).abs();
    assert!(
        error < 1e-11,
        "endpoint error {error:e} exceeds 1e-11; the pulse integral should be \
         resolved to near the requested tolerance"
    );
}
