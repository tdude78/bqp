//! Shared strict-HF session setup for the natural-conjunction test binaries.
//!
//! Every field here is read from `StrictHfForceAuthority::PART_A`, so this
//! helper cannot stop representing production: when the compiled science moves,
//! it moves with it.
//!
//! **There is a fourth `strict_physics` in `src/postprocess/session.rs`, it is
//! deliberately NOT this one, and it must not be folded in here.** That copy
//! hardcodes `sph_order`, `am_ratio`, `cd` and `cr` as literals precisely so
//! that `strict_hf_context_binds_the_sealed_part_a_force_authority` can assert
//! the literals against the authority and go red when the two disagree. It is
//! a differential oracle; this is plumbing. Merging them would delete the
//! comparison and leave a self-comparison behind.
//!
//! Each integration test binary links its own copy of this module, so items
//! only one binary needs read as dead there.
#![allow(dead_code, reason = "shared fixture surface differs per test binary")]

use two_phase_transfer_rs::{
    PhysicsConfig, StrictHfForceAuthority, TransferPostprocessSessionCore,
};

/// 2026-08-17T17:24:29Z, the sealed Part A v3 T0. The JB2008 v3 persistence
/// manifest authorizes 2026-08-15T11:24:29Z through 2026-08-31T17:24:29Z, so a
/// fixture epoch outside that window cannot propagate at all.
pub const T0_JD_UTC: f64 = 2_461_270.225_335_648_3;

pub fn strict_physics() -> PhysicsConfig {
    let authority = StrictHfForceAuthority::PART_A;
    PhysicsConfig {
        use_high_fidelity: true,
        require_hf_transfer_correction: true,
        sph_order: authority.gravity_order,
        force_flags: authority.force_flags,
        atm_model: authority.atmosphere_model,
        method: authority.integrator_method.to_owned(),
        dt_max: authority.dt_max_s,
        tolerance: authority.tolerance,
        transfer_am_ratio: authority.transfer_body_force.am_ratio,
        transfer_cd: authority.transfer_body_force.cd,
        transfer_cr: authority.transfer_body_force.cr,
        ..PhysicsConfig::default()
    }
}

#[expect(
    clippy::expect_used,
    reason = "test-only setup helper: a refused Part A authority must abort \
              the test loudly; clippy's allow-expect-in-tests covers \
              `#[test]` fns, not free helpers"
)]
pub fn strict_session() -> TransferPostprocessSessionCore {
    TransferPostprocessSessionCore::try_new(Some(strict_physics()), None)
        .expect("compiled Part A authority must prepare a strict session")
}
