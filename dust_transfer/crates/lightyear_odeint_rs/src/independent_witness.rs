//! Independent fixed-epoch numerical witness integrator.
//!
//! This deliberately does not call the production ODE engine. It uses classical
//! RK4 with step doubling and Richardson error estimation while evaluating the
//! same full [`crate::rhs::LightyearRHS`] force model.

use crate::eclipse::classify_binary_cylinder;
use crate::integrator::{FinalPropagationFailure, ScalarPropagationContext, MAX_STEPS};
use crate::rhs::{effective_scalar_srp, LightyearRHS};

/// Integrate one Encke-delta segment through an implementation-independent RK4 route.
///
/// `dt_max_s` and `tolerance` are explicit witness authority. The returned six
/// components are the final Encke delta, not the reconstructed Cartesian state.
///
/// A backward arc (`t_final_s < t0_s`) is supported through a mirrored
/// backward walk in [`integrate_fixed_ic_witness_backward`]; production
/// physics backward-extrapolates targets from the sealed common anchor by
/// design, so the witness must be able to express what production consumed.
/// The forward walk below is untouched by that addition.
///
/// # Errors
///
/// Fails closed on invalid controls, force/eclipsing errors, non-finite state,
/// step underflow, or the bounded step count.
pub fn integrate_fixed_ic_witness(
    context: &ScalarPropagationContext,
    init_equinoc_state: [f64; 6],
    t0_s: f64,
    t_final_s: f64,
    dt_max_s: f64,
    tolerance: f64,
) -> Result<[f64; 6], FinalPropagationFailure> {
    if !t0_s.is_finite()
        || !t_final_s.is_finite()
        || !dt_max_s.is_finite()
        || dt_max_s <= 0.0
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || init_equinoc_state.iter().any(|value| !value.is_finite())
    {
        return Err(FinalPropagationFailure::IntegrationFailure);
    }
    // Backward arcs take their own mirrored walk; the forward loop below is
    // byte-for-byte the pre-signed-dt code.
    if t_final_s < t0_s {
        return integrate_fixed_ic_witness_backward(
            context,
            init_equinoc_state,
            t0_s,
            t_final_s,
            dt_max_s,
            tolerance,
        );
    }
    // Backward arcs were diverted above, so `<=` is exactly equality here.
    // Written as a comparison rather than `==` to keep it lint-clean.
    if t_final_s <= t0_s {
        return Ok([0.0; 6]);
    }

    context
        .new_rhs(init_equinoc_state, t0_s)
        .and_then(|rhs| rhs.validate_strict_hf_arc(t0_s, t_final_s))
        .map_err(|_| FinalPropagationFailure::IntegrationFailure)?;

    let mut t = t0_s;
    let mut state = [0.0; 6];
    let mut h = dt_max_s.min(t_final_s - t0_s);
    let h_min = 1.0e-9_f64;
    for _ in 0..MAX_STEPS {
        let remaining = t_final_s - t;
        if remaining <= 0.0 {
            return finite(state);
        }
        h = h.min(remaining);

        // The RHS baseline cache is intentionally history-dependent. Build
        // separate force evaluators so the full-step and two-half-step error
        // branches cannot seed one another.
        let mut full_rhs = context
            .new_rhs(init_equinoc_state, t0_s)
            .map_err(|_| FinalPropagationFailure::IntegrationFailure)?;
        full_rhs.adapt_cache_policy_for_eps(tolerance);
        let full = rk4_step(&full_rhs, state, t, h)?;

        let mut half_rhs = context
            .new_rhs(init_equinoc_state, t0_s)
            .map_err(|_| FinalPropagationFailure::IntegrationFailure)?;
        half_rhs.adapt_cache_policy_for_eps(tolerance);
        let half = rk4_step(&half_rhs, state, t, 0.5 * h)?;
        let two_half = rk4_step(&half_rhs, half, t + 0.5 * h, 0.5 * h)?;

        let mut corrected = [0.0; 6];
        let mut error = 0.0_f64;
        for ((slot, &two_half), &full) in corrected.iter_mut().zip(&two_half).zip(&full) {
            let estimate = (two_half - full) / 15.0;
            *slot = two_half + estimate;
            error = error.max(estimate.abs());
        }
        finite(corrected)?;
        if error <= tolerance {
            state = corrected;
            t += h;
            let scale = if error == 0.0 {
                2.0
            } else {
                (0.9 * (tolerance / error).powf(0.2)).clamp(0.2, 2.0)
            };
            h = (h * scale).min(dt_max_s);
        } else {
            let scale = (0.9 * (tolerance / error).powf(0.2)).clamp(0.1, 0.5);
            h *= scale;
            if h < h_min {
                return Err(FinalPropagationFailure::IntegrationFailure);
            }
        }
    }
    Err(FinalPropagationFailure::IntegrationFailure)
}

/// Mirrored backward walk of the forward loop in
/// [`integrate_fixed_ic_witness`]: the epoch cursor `t` decreases from `t0_s`
/// to `t_final_s` and every RK4 stage is evaluated with a negative step
/// (`rk4_step` already takes signed `h`). Step control, Richardson error
/// estimation, and the failure taxonomy are identical to the forward walk.
///
/// The strict-HF arc is validated over the chronological span
/// `[t_final_s, t0_s]` because `validate_strict_hf_arc` requires
/// `elapsed_start <= elapsed_end`.
fn integrate_fixed_ic_witness_backward(
    context: &ScalarPropagationContext,
    init_equinoc_state: [f64; 6],
    t0_s: f64,
    t_final_s: f64,
    dt_max_s: f64,
    tolerance: f64,
) -> Result<[f64; 6], FinalPropagationFailure> {
    debug_assert!(t_final_s < t0_s, "backward walk requires t_final_s < t0_s");
    context
        .new_rhs(init_equinoc_state, t0_s)
        .and_then(|rhs| rhs.validate_strict_hf_arc(t_final_s, t0_s))
        .map_err(|_| FinalPropagationFailure::IntegrationFailure)?;

    let mut t = t0_s;
    let mut state = [0.0; 6];
    // `h` is the positive step magnitude; each accepted step moves `t` by `-h`.
    let mut h = dt_max_s.min(t0_s - t_final_s);
    let h_min = 1.0e-9_f64;
    for _ in 0..MAX_STEPS {
        let remaining = t - t_final_s;
        if remaining <= 0.0 {
            return finite(state);
        }
        h = h.min(remaining);

        // Same separate-evaluator discipline as the forward walk: the RHS
        // baseline cache is history-dependent, so the full-step and
        // two-half-step error branches must not seed one another.
        let mut full_rhs = context
            .new_rhs(init_equinoc_state, t0_s)
            .map_err(|_| FinalPropagationFailure::IntegrationFailure)?;
        full_rhs.adapt_cache_policy_for_eps(tolerance);
        let full = rk4_step(&full_rhs, state, t, -h)?;

        let mut half_rhs = context
            .new_rhs(init_equinoc_state, t0_s)
            .map_err(|_| FinalPropagationFailure::IntegrationFailure)?;
        half_rhs.adapt_cache_policy_for_eps(tolerance);
        let half = rk4_step(&half_rhs, state, t, -0.5 * h)?;
        let two_half = rk4_step(&half_rhs, half, t - 0.5 * h, -0.5 * h)?;

        let mut corrected = [0.0; 6];
        let mut error = 0.0_f64;
        for ((slot, &two_half), &full) in corrected.iter_mut().zip(&two_half).zip(&full) {
            let estimate = (two_half - full) / 15.0;
            *slot = two_half + estimate;
            error = error.max(estimate.abs());
        }
        finite(corrected)?;
        if error <= tolerance {
            state = corrected;
            t -= h;
            let scale = if error == 0.0 {
                2.0
            } else {
                (0.9 * (tolerance / error).powf(0.2)).clamp(0.2, 2.0)
            };
            h = (h * scale).min(dt_max_s);
        } else {
            let scale = (0.9 * (tolerance / error).powf(0.2)).clamp(0.1, 0.5);
            h *= scale;
            if h < h_min {
                return Err(FinalPropagationFailure::IntegrationFailure);
            }
        }
    }
    Err(FinalPropagationFailure::IntegrationFailure)
}

fn rk4_step(
    rhs: &LightyearRHS,
    state: [f64; 6],
    t: f64,
    h: f64,
) -> Result<[f64; 6], FinalPropagationFailure> {
    let k1 = evaluate(rhs, &state, t)?;
    let y2 = add_scaled(state, k1, 0.5 * h);
    let k2 = evaluate(rhs, &y2, t + 0.5 * h)?;
    let y3 = add_scaled(state, k2, 0.5 * h);
    let k3 = evaluate(rhs, &y3, t + 0.5 * h)?;
    let y4 = add_scaled(state, k3, h);
    let k4 = evaluate(rhs, &y4, t + h)?;
    let mut out = [0.0; 6];
    for (slot, (&state, (&k1, (&k2, (&k3, &k4))))) in out.iter_mut().zip(
        state
            .iter()
            .zip(k1.iter().zip(k2.iter().zip(k3.iter().zip(&k4)))),
    ) {
        *slot = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    finite(out)
}

fn evaluate(
    rhs: &LightyearRHS,
    state: &[f64; 6],
    t: f64,
) -> Result<[f64; 6], FinalPropagationFailure> {
    if effective_scalar_srp(&rhs.config) {
        let (position, sun) = rhs
            .eclipse_geometry_at_delta(state, t)
            .map_err(FinalPropagationFailure::Eclipse)?;
        let side = classify_binary_cylinder(position, sun, rhs.config.earth_radius)
            .map_err(FinalPropagationFailure::Eclipse)?;
        rhs.set_eclipse_side(side);
        rhs.validate_eclipse_envelope_at_delta(state, t)
            .map_err(FinalPropagationFailure::Eclipse)?;
    }
    let derivative = rhs
        .compute_internal(state, t)
        .map_err(FinalPropagationFailure::Gravity)?;
    finite(derivative)
}

fn add_scaled(mut state: [f64; 6], derivative: [f64; 6], scale: f64) -> [f64; 6] {
    for (slot, &derivative) in state.iter_mut().zip(&derivative) {
        *slot += scale * derivative;
    }
    state
}

fn finite(state: [f64; 6]) -> Result<[f64; 6], FinalPropagationFailure> {
    state
        .iter()
        .all(|value| value.is_finite())
        .then_some(state)
        .ok_or(FinalPropagationFailure::NanState)
}
