#![allow(non_snake_case)]
//! Lightyear ODE Integrator - Rust Port
//!
//! High-performance delta-state propagator using a Lightyear-compatible DOPRI5 integrator
//! (via the `odesolve` module) with full event handling support.
//!
//! # Why there is no crate-level `allow(unused)`
//!
//! There used to be one, and it is why several hundred lines of unreachable
//! force-model and batching code accumulated here without anyone noticing: the
//! compiler knew, and the attribute stopped it saying so. Do not restore it.
//! If a specific item must outlive its callers, annotate THAT item and say why,
//! so the reason is greppable and dies with the item.
//!
//! ## `strict_hf_enclosure_authority_is_private`
//!
//! Strict-HF enclosure authority is an internal capability. External crates
//! cannot name or construct its token, supply identities, or inject bounds.
//!
//! ```compile_fail
//! use lightyear_odeint_rs::strict_hf_enclosure::StrictHfEnclosureAuthority;
//! ```

// Internal modules
pub(crate) mod eclipse;
mod eclipse_coordinator;
pub use eclipse::EclipseError;
mod events;
pub mod independent_witness;
// The standalone ODE solvers (Tsit5, Dop853, RKV98, Dopri5, Vern7, Vern9,
// ESDIRK4(3)). Absorbed from the `odesolve_lightyear` crate, whose only
// workspace consumer was this one. Private: `integrator`, `eclipse_coordinator`
// and `adaptive_solver` are the callers, and nothing outside this crate ever
// named it.
mod odesolve;
mod physical_constants;
pub mod rhs;
mod strict_hf_enclosure;
mod utils;

// Public modules (re-exported for use by other crates)
pub mod batch;
pub mod config;
pub mod integrator;
pub mod precomputed_ephem;
pub mod probe;
pub mod session;
pub mod types;

#[cfg(feature = "autodiff")]
mod rhs_dual;

#[cfg(feature = "autodiff")]
mod jacobian; // Jacobian extraction via DualVec AD (ESDIRK prerequisite)

#[cfg(feature = "autodiff")]
mod adaptive_solver; // Stiffness-adaptive meta-solver (Tsit5 <-> ESDIRK switching)

// Re-exports for convenient access
pub use batch::{
    integrate_batch_native, integrate_batch_native_into, BatchBallistics, BatchPropagationRequest,
};
pub use config::get_global_coeffs_packed;
pub use config::load_constants_from_bytes;
pub use config::packed_constants_from_bytes;
pub use integrator::{
    integrate_adaptive, integrate_final_checked, SampledOutputMode, ScalarGravityAssets,
    ScalarPropagationContext, ScalarPropagationRequest,
};
#[cfg(feature = "scalar-leg-observer")]
pub use integrator::{
    integrate_final_checked_observed, ObservedFinalLeg, ObservedFinalMetricError,
    ObservedFinalMetrics, ObservedSolverTerminalStatus,
};
pub use session::{LightyearSession, VariableFinalBallistics, VariableFinalBatchRequest};
pub use types::StepperMethod;

pub(crate) const CONSTANTS_NOT_LOADED_MESSAGE: &str =
    "spherical harmonics constants not loaded; call load_constants() first";

pub(crate) fn init_states_shape_message(shape: &[usize]) -> String {
    format!("init_equinoc_states must have shape (n_sigma, 6), got {shape:?}")
}

pub(crate) fn output_shape_mismatch_message(
    n_sigma: usize,
    n_times: usize,
    shape: &[usize],
) -> String {
    format!("states_out must have shape ({n_sigma}, {n_times}, 6), got {shape:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ForceFlags;
    use satpy_core::{eci2equinoc_impl, equinoc_prop_from_impl, PackedGravityCoeffs};
    use std::sync::Arc;

    /// Create minimal mock spherical harmonic coefficients.
    ///
    /// For unit tests without file I/O, we create minimal coefficient arrays.
    /// With `sph_order=2`, only J2 matters; with `sph_order=0`, no harmonics are used.
    fn mock_coefficients(order: usize) -> Arc<PackedGravityCoeffs> {
        let stride = order.saturating_add(2);
        let total_size = stride.saturating_mul(stride);
        let mut c_coeffs = vec![0.0; total_size];
        let s_coeffs = vec![0.0; total_size];

        // C[0,0] = 1.0 is the point-mass gravity term
        *c_coeffs
            .get_mut(0)
            .expect("mock coefficient array must contain point-mass term") = 1.0;

        // J2 term (if order >= 2)
        if order >= 2 {
            let j2_index = stride.saturating_mul(2);
            *c_coeffs
                .get_mut(j2_index)
                .expect("mock coefficient array must contain J2 term") = -1.08263e-3;
        }

        let packed = satpy_core::pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, order)
            .expect("mock test gravity coefficients must pack");

        Arc::new(packed)
    }

    #[test]
    fn test_integrate_adaptive_basic() {
        // Use minimal force config (sph_order=0 for point-mass only)
        let config = Arc::new(types::ForceConfig {
            sph_order: 0,
            eps: 1e-10,
            integrator_method: types::StepperMethod::Dopri5Compat,
            ..types::ForceConfig::default()
        });

        let packed = mock_coefficients(0);

        // Initial state (delta = 0)
        let init_state = [0.0; 6];
        let t_eval = vec![0.0, 60.0, 120.0];
        let jd0 = 2_460_000.0;

        let gravity = integrator::ScalarGravityAssets::new(packed);
        let context = integrator::ScalarPropagationContext::new(jd0, config, gravity);
        let result = integrator::integrate_adaptive(
            integrator::ScalarPropagationRequest::new(&context, init_state, &t_eval, 0.0, 120.0)
                .with_events(false),
        )
        .expect("basic sampled propagation census");

        // Check that integration completed successfully
        // Note: Without interpolation, we may not hit exact t_eval points
        // but we should have some output states
        assert!(!result.max_steps_exceeded);
        assert!(!result.terminal_event_fired);
        assert!(
            result.metrics.total_steps > 0,
            "Should have taken at least one step"
        );
        assert!(
            result.metrics.total_evals > 0,
            "Should have performed function evaluations"
        );
    }

    fn build_force_config(
        force_flags: i32,
        atm_model: i32,
        am_ratio: f64,
        cd: f64,
        cr: f64,
        sun_pos: Option<[f64; 3]>,
    ) -> Arc<types::ForceConfig> {
        Arc::new(types::ForceConfig {
            sph_order: 0,
            force_flags,
            atm_model,
            am_ratio,
            cd,
            cr,
            sun_pos,
            eps: 1e-10,
            integrator_method: types::StepperMethod::Dopri5Compat,
            ..types::ForceConfig::default()
        })
    }

    fn propagate_eci(
        init_equ: [f64; 6],
        dt_s: f64,
        jd0: f64,
        config: &Arc<types::ForceConfig>,
        packed: &Arc<PackedGravityCoeffs>,
    ) -> [f64; 6] {
        let gravity = integrator::ScalarGravityAssets::new(Arc::clone(packed));
        let context = integrator::ScalarPropagationContext::new(jd0, Arc::clone(config), gravity);
        let final_delta = integrator::integrate_final_checked(
            integrator::ScalarPropagationRequest::new(&context, init_equ, &[dt_s], 0.0, dt_s)
                .with_events(false),
        )
        .expect("integration failed");

        let mut baseline = [0.0; 6];
        equinoc_prop_from_impl(&init_equ, dt_s, &mut baseline);

        let mut out = [0.0; 6];
        for ((out_value, baseline_value), final_delta_value) in
            out.iter_mut().zip(&baseline).zip(&final_delta)
        {
            *out_value = baseline_value + final_delta_value;
        }
        out
    }

    fn round_trip_error(
        init_equ: [f64; 6],
        dt_s: f64,
        config: &Arc<types::ForceConfig>,
        packed: &Arc<PackedGravityCoeffs>,
    ) -> (f64, f64) {
        let jd0 = 2_460_000.0;

        let mut init_eci = [0.0; 6];
        equinoc_prop_from_impl(&init_equ, 0.0, &mut init_eci);

        let fwd = propagate_eci(init_equ, dt_s, jd0, config, packed);

        let mut equ_fwd = [0.0; 6];
        eci2equinoc_impl(&fwd, 6, 0.0, 0.0, &mut equ_fwd);

        let back = propagate_eci(equ_fwd, -dt_s, jd0, config, packed);

        let squared_error = |(back_value, initial_value): (&f64, &f64)| {
            let difference = back_value - initial_value;
            difference * difference
        };
        let pos_err: f64 = back.iter().zip(&init_eci).take(3).map(squared_error).sum();
        let vel_err: f64 = back.iter().zip(&init_eci).skip(3).map(squared_error).sum();

        (pos_err.sqrt(), vel_err.sqrt())
    }

    #[test]
    fn test_round_trip_gravity_only_600s() {
        let packed = mock_coefficients(0);
        let config = build_force_config(0, 0, 0.0, 0.0, 0.0, None);
        let init_equ = [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (pos_err, vel_err) = round_trip_error(init_equ, 600.0, &config, &packed);
        println!("gravity_600s pos_err_km={pos_err:.6e} vel_err_km_s={vel_err:.6e}");
        assert!(pos_err <= 1e-2, "pos_err too large: {pos_err}");
        assert!(vel_err <= 1e-5, "vel_err too large: {vel_err}");
    }

    #[test]
    fn test_round_trip_drag_only_600s() {
        let packed = mock_coefficients(0);
        let config = build_force_config(ForceFlags::DRAG, 1, 0.01, 2.2, 0.0, None);
        let init_equ = [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (pos_err, vel_err) = round_trip_error(init_equ, 600.0, &config, &packed);
        println!("drag_600s pos_err_km={pos_err:.6e} vel_err_km_s={vel_err:.6e}");
        assert!(pos_err <= 1e-2, "pos_err too large: {pos_err}");
        assert!(vel_err <= 1e-5, "vel_err too large: {vel_err}");
    }

    #[test]
    fn test_round_trip_drag_srp_600s() {
        let packed = mock_coefficients(0);
        let sun_pos = [1.496e8, 0.0, 0.0];
        let flags = ForceFlags::DRAG | ForceFlags::SRP;
        let config = build_force_config(flags, 1, 0.01, 2.2, 1.3, Some(sun_pos));
        let init_equ = [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (pos_err, vel_err) = round_trip_error(init_equ, 600.0, &config, &packed);
        println!("drag_srp_600s pos_err_km={pos_err:.6e} vel_err_km_s={vel_err:.6e}");
        assert!(pos_err <= 1e-2, "pos_err too large: {pos_err}");
        assert!(vel_err <= 1e-5, "vel_err too large: {vel_err}");
    }

    #[test]
    fn test_round_trip_drag_only_3600s() {
        let packed = mock_coefficients(0);
        let config = build_force_config(ForceFlags::DRAG, 1, 0.01, 2.2, 0.0, None);
        let init_equ = [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (pos_err, vel_err) = round_trip_error(init_equ, 3600.0, &config, &packed);
        println!("drag_3600s pos_err_km={pos_err:.6e} vel_err_km_s={vel_err:.6e}");
        assert!(pos_err <= 1e-2, "pos_err too large: {pos_err}");
        assert!(vel_err <= 1e-5, "vel_err too large: {vel_err}");
    }
}
