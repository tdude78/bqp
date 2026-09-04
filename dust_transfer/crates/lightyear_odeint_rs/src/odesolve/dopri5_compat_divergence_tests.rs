//! The two DOPRI5 step-size controllers, side by side.
//!
//! WHY THIS FILE EXISTS. `lightyear_compat::integrate_lightyear_dopri5_impl`
//! carries its own PI (Gustafsson) controller, about 150 lines that restate
//! what `solver::integrate_internal` already does. Until this file there was
//! NO test anywhere in the workspace that ran both on one problem, so the two
//! could drift apart -- and have -- with nothing to say so. Every existing
//! test of the compat path exercises the compat path alone.
//!
//! WHAT IT PINS: THE DIVERGENCE, NOT THE AGREEMENT. These are not two
//! implementations of one contract that happen to differ; they are a legacy
//! stepper and its successor, and the compat loop is deliberately frozen
//! (see the "KNOWN DEFECT, DELIBERATELY NOT FIXED HERE" note on its reject
//! branch). So the assertions below record what is true TODAY. If you unify
//! them, these tests are supposed to go red -- that is the review this file
//! is buying, and the right response is to update it in the same commit,
//! not to loosen it.
//!
//! WHAT IS ACTUALLY DIFFERENT. Measured against `solver.rs` at this commit,
//! with the Dopri5 tableau on both sides (`order_err: 4`, `err3: None`, so
//! the generic controller's `alpha`/`beta`/`inv_order`/`max_growth` land on
//! the literals the compat loop hardcodes -- the arithmetic looks identical
//! and is not):
//!
//! * PI memory on reject: compat writes `err_prev`/`have_err_prev` from a
//!   REJECTED step; `solver.rs` deliberately does not, on either of its
//!   loops. This is the one difference the compat source already documents.
//! * First step: compat `dt_total / 100`, clamped to at least 1e-3;
//!   `solver.rs` `span / 2` when the span is 60 s or less, `span / 100`
//!   above, with no low clamp.
//! * `h_min`: compat 1e-10, `solver.rs` 1e-12 (and 1e-12 is what production
//!   passes). This is not only a floor -- it feeds the force-accept
//!   condition `h.abs() <= h_min` on both sides, so it decides which steps
//!   are accepted DESPITE failing the error test.
//! * `h0` carry: `solver.rs` honours `IntegratorConfig::h0`; `LightyearConfig`
//!   has no such field, so the h-carry lever is inert on the compat path.
//! * `h_max` fallback on a non-finite cap: compat 1e-10, `solver.rs` the span.
//! * Retry derivative: both paths recompute `k[0]` after a rejection because
//!   `OdeSystem` does not promise call-history-independent derivatives.
//! * Output-grid match tolerance: compat 1e-9, `solver.rs` 1e-12.
//! * `eps` floor: compat floors at 1e-12 internally, `solver.rs` leaves it to
//!   the caller.
//! * `steps` MEANS SOMETHING DIFFERENT. Compat does `stats.steps += 1`
//!   immediately after the RK step, BEFORE the accept test, so it counts
//!   ATTEMPTS. `solver.rs` does it inside the accept branch, so it counts
//!   ACCEPTED steps. A reader comparing step counts across the two is
//!   comparing different units; both identities are pinned below.
//! * Telemetry: compat writes 2 of the 13 `IntegrationStats` fields. The
//!   other eleven are structural zeros on this path, not measurements --
//!   `rejected_steps` is pinned as such below, and it is a REAL hole, not a
//!   quiet fixture: instrumenting all three of compat's reject arms on the
//!   fixture below turns its reported 0 into 18.
//!
//! NOT A UNIFICATION, AND NOT AN ARGUMENT FOR ONE. `Dopri5Compat` is
//! production-reachable (the B500 screen arms, `resolve_auto_stepper` below
//! 2e-8, and the empty-token default in `py_config`) but is NOT the flown
//! stepper -- the sealed campaign authority is Vern7. Merging the two loops
//! would move the compat path's step sequence and therefore any result pinned
//! on it; it needs its own re-baseline.
//!
//! THAT REACHABILITY CLAIM IS NOT TESTED HERE, AND CANNOT BE. Every case in
//! this file builds an `OdeSystem` by hand and calls the two integrators
//! DIRECTLY, so it pins the divergence between two functions and says nothing
//! about which of them production selects. It builds them itself, which is the
//! point: the divergence has to be pinned on inputs nobody's selection logic
//! chose. (Until the `odesolve_lightyear` crate was absorbed into this one, the
//! reason was stronger — it was a leaf crate that could not see
//! `resolve_auto_stepper` at all. It can now; the file still does not, on
//! purpose.) The live selection is pinned from the entry point that owns it,
//! in `lightyear_odeint_rs/tests/dopri5_compat_live_selection.rs`: `Auto`
//! below `eps = 2e-8` returns the same bits through the production
//! `integrate_final_checked` entry as an explicitly configured `Dopri5Compat`,
//! and Vern9's bits at or above it. Change the resolution and that file reds;
//! change these loops and this one does. Both are the review a unification
//! owes.

use crate::odesolve::{
    integrate_final, integrate_lightyear_dopri5_final, ErrorControl, IntegrationStatus,
    IntegratorConfig, LightyearConfig, Method, OdeSystem,
};

/// A two-body-flavoured stiff-ish oscillator: `y'' = -k y` written as a first
/// order pair, with `k` large enough that the controller has to work and the
/// closed-form answer is exact. Nothing about the comparison depends on the
/// system being orbital; what it needs is a problem whose error estimate
/// varies enough over the span to exercise the PI term.
struct Oscillator {
    k: f64,
}

impl OdeSystem for Oscillator {
    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) {
        let (Some(&position), Some(&velocity)) = (y.first(), y.get(1)) else {
            return;
        };
        if let Some(slot) = dy.first_mut() {
            *slot = velocity;
        }
        if let Some(slot) = dy.get_mut(1) {
            *slot = -self.k * position;
        }
    }
}

/// A system whose derivative jumps hard partway through the span, so the
/// controller is forced into rejections rather than a smooth ramp.
struct Kicked;

impl OdeSystem for Kicked {
    fn rhs(&self, t: f64, y: &[f64], dy: &mut [f64]) {
        let Some(&state) = y.first() else {
            return;
        };
        let stiffness = if t > 5.0 { 5.0e3 } else { 1.0 };
        if let Some(slot) = dy.first_mut() {
            *slot = -stiffness * state;
        }
    }
}

const T0: f64 = 0.0;
const TF: f64 = 100.0;
const EPS: f64 = 1e-9;
const DT_MAX: f64 = 60.0;

const fn compat_config() -> LightyearConfig {
    LightyearConfig {
        eps: EPS,
        dt_max: DT_MAX,
        max_steps: 200_000,
        max_rejects: 50,
        force_eval: false,
        // The single-sample fast path, which is what the production final
        // entry takes.
        fast_single: true,
    }
}

/// The generic solver configured as close to the compat loop as its own
/// options allow: same absolute error control, same cap, same budgets. What
/// remains different after this is the controller itself, which is the point.
const fn generic_config() -> IntegratorConfig {
    IntegratorConfig {
        error_control: ErrorControl::Absolute { eps: EPS },
        h0: None,
        h_min: 1e-12,
        h_max: DT_MAX,
        max_steps: 200_000,
        max_rejects: 50,
        force_eval: false,
    }
}

/// Both controllers solve the problem, and both solve it to the tolerance
/// they were given -- but not with the same step sequence.
///
/// The first half is the safety property: a divergence pin that did not also
/// check both answers would keep passing if one controller broke outright.
/// The second half is the divergence: identical tableau, identical
/// alpha/beta/safety literals, and still a different number of steps,
/// because the reject-branch PI memory, the first step and `h_min` all
/// differ.
#[test]
fn compat_and_generic_dopri5_agree_on_the_answer_but_not_on_the_step_sequence() {
    let system = Oscillator { k: 4.0 };
    let y0 = [1.0, 0.0];

    let compat = integrate_lightyear_dopri5_final(&system, &y0, T0, TF, compat_config(), None);
    let generic = integrate_final(&system, Method::Dopri5, &y0, T0, TF, generic_config());

    assert_eq!(
        compat.status,
        IntegrationStatus::Success,
        "the compat controller must solve this problem"
    );
    assert_eq!(
        generic.status,
        IntegrationStatus::Success,
        "the generic controller must solve this problem"
    );

    // Closed form: y = cos(sqrt(k) t), y' = -sqrt(k) sin(sqrt(k) t).
    let omega = 2.0_f64;
    let exact = [(omega * TF).cos(), -omega * (omega * TF).sin()];
    // Loose against the exact solution and tight between the two: the
    // absolute-eps controllers are asked for 1e-9 per step, and 100 s of
    // accumulation over hundreds of steps is not 1e-9 of global error.
    let global_tol = 1e-5;
    for (index, &reference) in exact.iter().enumerate() {
        let compat_value = compat.y.get(index).copied().unwrap_or(f64::NAN);
        let generic_value = generic.y.get(index).copied().unwrap_or(f64::NAN);
        assert!(
            (compat_value - reference).abs() < global_tol,
            "compat component {index} is {compat_value}, exact {reference}"
        );
        assert!(
            (generic_value - reference).abs() < global_tol,
            "generic component {index} is {generic_value}, exact {reference}"
        );
    }

    println!(
        "dopri5 controllers: compat steps={} evals={} rejected_steps={} | \
         generic steps={} evals={} rejected_steps={}",
        compat.stats.steps,
        compat.stats.evals,
        compat.stats.rejected_steps,
        generic.stats.steps,
        generic.stats.evals,
        generic.stats.rejected_steps,
    );

    assert!(
        compat.stats.steps > 0 && generic.stats.steps > 0,
        "neither arm may take zero steps; a zero-step arm compares nothing"
    );
    assert_ne!(
        (compat.stats.steps, compat.stats.evals),
        (generic.stats.steps, generic.stats.evals),
        "the two DOPRI5 controllers took the SAME step and evaluation counts. \
         Either they were unified -- in which case delete this assertion in \
         the same commit as the unification and say so -- or this fixture \
         stopped exercising the difference, in which case it is no longer a \
         differential test"
    );
}

/// `steps` counts attempts on one path and accepted steps on the other, and
/// only one path reports its rejections.
///
/// BOTH CLAIMS ARE PROVED BY ARITHMETIC, NOT BY A RECORDED COUNT, so this
/// test does not move with the host's libm. DOPRI5 spends exactly six RHS
/// evaluations per attempted step (FSAL reuses the seventh), plus one
/// evaluation before the loop:
///
/// * `solver.rs` counts ACCEPTED steps. Every attempt costs six evaluations
///   through FSAL, plus one for the initial derivative and one fresh `k[0]`
///   after each rejection, so
///   `evals == 6 * (steps + rejected_steps) + 1 + rejected_steps`.
/// * compat counts ATTEMPTS -- `stats.steps += 1` sits above its accept test
///   -- so every rejection is already inside `steps` and
///   `evals == 6 * steps + 1` holds no matter how many rejections it took.
///
/// The second identity is what makes compat's `rejected_steps: 0` readable
/// as "never instrumented" rather than "never rejected": the field carries no
/// information either way, and the eval identity is the only rejection
/// instrument on that path. It IS a hole and not a quiet fixture -- adding
/// `stats.rejected_steps += 1` to all three of compat's reject arms turns
/// this fixture's reported 0 into 18, which is the poison for the assertion
/// below.
///
/// Eleven of the thirteen `IntegrationStats` fields are structural zeros on
/// the compat path, and `lightyear_odeint_rs::integrator` feeds all of them
/// to the propagation probe regardless.
#[test]
fn compat_counts_attempted_steps_and_reports_none_of_its_rejections() {
    let y0 = [1.0];
    let t0 = 0.0;
    let tf = 10.0;

    let mut compat_cfg = compat_config();
    compat_cfg.dt_max = 5.0;
    let compat = integrate_lightyear_dopri5_final(&Kicked, &y0, t0, tf, compat_cfg, None);

    let mut generic_cfg = generic_config();
    generic_cfg.h_max = 5.0;
    let generic = integrate_final(&Kicked, Method::Dopri5, &y0, t0, tf, generic_cfg);

    assert_eq!(
        compat.status,
        IntegrationStatus::Success,
        "the compat arm must reach tf, or its counters describe a failure"
    );
    assert_eq!(
        generic.status,
        IntegrationStatus::Success,
        "the generic arm must reach tf"
    );

    println!(
        "kicked system: compat steps={} evals={} rejected_steps={} | \
         generic steps={} evals={} rejected_steps={}",
        compat.stats.steps,
        compat.stats.evals,
        compat.stats.rejected_steps,
        generic.stats.steps,
        generic.stats.evals,
        generic.stats.rejected_steps,
    );

    assert!(
        generic.stats.rejected_steps > 0,
        "the fixture must actually force rejections, or neither identity below \
         separates the two step-counting conventions. The generic arm reported \
         {} rejections",
        generic.stats.rejected_steps
    );
    assert_eq!(
        generic.stats.evals,
        6 * (generic.stats.steps + generic.stats.rejected_steps) + 1 + generic.stats.rejected_steps,
        "solver.rs no longer counts ACCEPTED steps with six evaluations per \
         attempt plus a fresh k[0] after each rejection: steps={} rejected={} evals={}",
        generic.stats.steps,
        generic.stats.rejected_steps,
        generic.stats.evals,
    );
    assert_eq!(
        compat.stats.evals,
        6 * compat.stats.steps + 1,
        "the compat loop no longer counts ATTEMPTED steps: steps={} evals={}. \
         If its counter moved below the accept test it now means what \
         solver.rs means, and every reader that compared the two got a unit \
         change with no signal",
        compat.stats.steps,
        compat.stats.evals,
    );
    assert_eq!(
        compat.stats.rejected_steps, 0,
        "the compat loop reported {} rejected steps. It has never incremented \
         that counter on any of its three reject arms, so either it was \
         instrumented -- good, and then delete this pin and the note it cites \
         in the same commit -- or something else is writing the field",
        compat.stats.rejected_steps
    );
    assert_eq!(
        compat.stats.saturated_steps, 0,
        "same structural zero: the compat loop does not instrument saturation \
         either"
    );
}
