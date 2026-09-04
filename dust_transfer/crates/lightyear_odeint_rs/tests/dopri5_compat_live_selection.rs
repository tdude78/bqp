//! The LIVE path onto the frozen DOPRI5 compat loop, pinned end to end.
//!
//! WHY A SECOND FILE. `src/odesolve/dopri5_compat_divergence_tests.rs` pins the
//! two DOPRI5 controllers against each other, and its header rests the whole
//! "do not unify as a drive-by" argument on `Dopri5Compat` being
//! production-reachable -- "the B500 screen arms, `resolve_auto_stepper` below
//! 2e-8, and the empty-token default in `py_config`". It does not test any of
//! that: every case there constructs an `OdeSystem` by hand and calls the two
//! integrators DIRECTLY, so it pins the divergence between two functions and
//! says nothing about whether production can reach either of them. (That used
//! to be forced -- it lived in `odesolve_lightyear`, a leaf crate that could
//! not see `resolve_auto_stepper`. Absorbing that crate here removed the
//! obstacle and not the reason.)
//!
//! That is the hole this file closes, and it closes it from the crate that owns
//! the selection. `resolve_auto_stepper` is private, so this drives it the way
//! production does: through the public `integrate_final_checked` entry, with
//! `StepperMethod::Auto` in the compiled force authority.
//!
//! WHAT IT PINS. The selection is bit-exact on both sides of its own threshold:
//! `Auto` below `eps = 2e-8` returns the SAME BITS as an explicitly configured
//! `Dopri5Compat`, and at or above it the same bits as `Vern9`. It is not
//! enough to check one side -- an `Auto` that had silently collapsed to a
//! single method would still match one arm -- so the third assertion is that
//! the two arms differ from each other, which is what makes the first two mean
//! something.
//!
//! THE THRESHOLD IS `eps >= 2e-8` -> Vern9, BELOW -> `Dopri5Compat`, and the
//! cases below sit one representable step apart across it rather than at round
//! numbers, so moving the constant in either direction reds this file.
//!
use lightyear_odeint_rs::types::ForceConfig;
use lightyear_odeint_rs::{
    integrate_final_checked, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest, StepperMethod,
};
use satpy_core::{pack_gravity_coeffs, PackedGravityCoeffs};
use std::sync::Arc;

/// `resolve_auto_stepper`'s threshold, restated here so this file goes red when
/// the constant moves rather than silently re-pinning to the new one.
///
/// A test-local copy of a production constant is normally a trap -- it degrades
/// to agreeing with whatever the source says. It is safe here BECAUSE nothing
/// below reads it as an oracle: the assertions compare two real propagations
/// against each other, and this literal only places the two probe points.
const AUTO_THRESHOLD_EPS: f64 = 2e-8;

/// A J2-only order-5 table. Small on purpose: the pin is about which
/// controller ran, not about the force model.
#[expect(
    clippy::expect_used,
    reason = "fixture construction: a pack this file cannot build must stop \
    the run, not silently pin a different force model"
)]
fn j2_pack(order: usize) -> Arc<PackedGravityCoeffs> {
    let stride = order
        .checked_add(2)
        .expect("test coefficient stride must not overflow");
    let total = stride
        .checked_mul(stride)
        .expect("test coefficient array length must not overflow");
    let mut c = vec![0.0; total];
    let s = vec![0.0; total];
    *c.get_mut(0).expect("C[0,0] slot must exist") = 1.0;
    let j2_index = 2_usize
        .checked_mul(stride)
        .expect("J2 index must not overflow");
    *c.get_mut(j2_index).expect("C[2,0] slot must exist") = -1.082_63e-3;
    Arc::new(pack_gravity_coeffs(&c, &s, stride, order).expect("J2 coefficients must pack"))
}

/// One 600 s arc through the PRODUCTION entry, with `method` compiled into the
/// force authority exactly as a campaign configuration would carry it.
///
/// `packed` is passed in rather than built here so the comparison uses the same
/// immutable gravity authority shape as production.
#[expect(
    clippy::expect_used,
    reason = "an arc that fails to integrate must stop the run: a swallowed \
    failure would leave the bit comparisons below reading a fallback"
)]
fn final_state(packed: &Arc<PackedGravityCoeffs>, method: StepperMethod, eps: f64) -> [f64; 6] {
    let packed = Arc::clone(packed);
    let config = Arc::new(ForceConfig {
        sph_order: 5,
        force_flags: 0,
        subtract_first_order: true,
        dt_max: 60.0,
        eps,
        integrator_method: method,
        ..ForceConfig::default()
    });
    let context =
        ScalarPropagationContext::new(2_460_310.5, config, ScalarGravityAssets::new(packed));
    let state = [7_050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
    let times = [600.0];
    integrate_final_checked(ScalarPropagationRequest::new(
        &context, state, &times, 0.0, 600.0,
    ))
    .expect("the fixture arc must integrate under every stepper")
}

/// The live entry resolves `Auto` onto the compat loop below the threshold and
/// onto Vern9 at it, bit for bit.
///
/// POISON. Three independent ones, and each reds a different assertion:
///
/// * Delete the `resolve_auto_stepper` call in `integrate_final_checked` --
///   `Auto` then falls through `stepper_ode_method`'s `Vern9 | Auto` arm and
///   the sub-threshold case matches Vern9 instead of the compat loop.
/// * Flip the comparison to `eps > 2e-8` or move the constant -- the arm that
///   straddles it swaps and both equality assertions red.
/// * Point `StepperMethod::Dopri5Compat` at the generic controller in
///   `integrate_final_no_events_with_rhs` -- the compat arm's bits move and the
///   sub-threshold equality reds, which is exactly the re-baseline the
///   divergence pin's header says a unification owes.
#[test]
fn auto_resolves_onto_the_compat_dopri5_loop_below_the_threshold() {
    let packed = j2_pack(5);

    // One representable step either side of the threshold, so the two probe
    // points cannot both sit in the same branch however the comparison is
    // written.
    let below = AUTO_THRESHOLD_EPS - f64::EPSILON * AUTO_THRESHOLD_EPS;
    let at_or_above = AUTO_THRESHOLD_EPS;
    assert!(
        below < AUTO_THRESHOLD_EPS && at_or_above >= AUTO_THRESHOLD_EPS,
        "the two probe points must straddle the threshold, or this file pins \
         one branch twice"
    );

    let auto_below = final_state(&packed, StepperMethod::Auto, below);
    let compat_below = final_state(&packed, StepperMethod::Dopri5Compat, below);
    let vern9_below = final_state(&packed, StepperMethod::Vern9, below);

    assert_ne!(
        compat_below.map(f64::to_bits),
        vern9_below.map(f64::to_bits),
        "the two steppers returned identical bits on this fixture, so matching \
         one of them proves nothing. Either they were unified -- update the \
         divergence pin in the same commit -- or this arc is too easy to \
         separate them and needs a longer span"
    );
    assert_eq!(
        auto_below.map(f64::to_bits),
        compat_below.map(f64::to_bits),
        "Auto below eps={AUTO_THRESHOLD_EPS:e} did NOT land on the frozen \
         Dopri5Compat loop. That loop is only production-reachable through this \
         resolution, the B500 screen arms and py_config's empty token; if the \
         resolution moved, the compat loop's whole reachability argument moved \
         with it"
    );

    let auto_at = final_state(&packed, StepperMethod::Auto, at_or_above);
    let vern9_at = final_state(&packed, StepperMethod::Vern9, at_or_above);
    assert_eq!(
        auto_at.map(f64::to_bits),
        vern9_at.map(f64::to_bits),
        "Auto at eps={AUTO_THRESHOLD_EPS:e} did not land on Vern9; the \
         threshold's upper branch moved"
    );
}
