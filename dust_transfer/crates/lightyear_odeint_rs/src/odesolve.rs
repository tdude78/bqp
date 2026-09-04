//! Standalone ODE solvers for Lightyear (Tsit5, Dop853, RKV98, Dopri5, Vern7,
//! Vern9, ESDIRK4(3)).
//!
//! This was the `odesolve_lightyear` crate until it was absorbed here. It had
//! exactly one consumer in the workspace — this crate — so its 139 public items
//! were API for nobody, and `pub` in a library keeps an item out of rustc's
//! dead-code analysis entirely.
//!
//! It carried no cargo features and still contains no `cfg(feature = ...)`, so
//! every build compiles exactly one arm of it. `rejected_a2_and_rhs_context_api_are_absent`
//! in `crates/lightyear_odeint_rs/src/rhs.rs` asserts that, where it used to
//! assert the absent `[features]` table in the crate's own manifest.

mod implicit_step;
mod lightyear_compat;
mod lu6;
pub mod solver;
mod tableau;
mod tableau_esdirk;

#[cfg(test)]
mod basic_tests;
#[cfg(test)]
mod dopri5_compat_divergence_tests;

pub use solver::{
    integrate_final, integrate_final_esdirk, integrate_final_with_events,
    integrate_final_with_events_and_scratch, integrate_final_with_scratch, integrate_sampled,
    integrate_sampled_esdirk, integrate_sampled_esdirk_into, integrate_sampled_into,
    integrate_sampled_unforced, integrate_sampled_with_events,
    integrate_sampled_with_events_esdirk, ErrorControl, EventDecision, EventHandler,
    IntegrationResult, IntegrationResultSampled, IntegrationStats, IntegrationStatus,
    IntegratorConfig, Method, OdeSystem, SanitizedInterp, SolverScratch,
};

pub use implicit_step::JacobianProvider;

pub use lightyear_compat::{
    integrate_lightyear_dopri5, integrate_lightyear_dopri5_final,
    integrate_lightyear_dopri5_unforced, LightyearConfig,
};
