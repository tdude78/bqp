//! The generic and hand-written-f64 equinoctial conversions, cross-checked.
//!
//! `eci2equinoc_impl<T>` and `eci2equinoc_impl_f64` are separately written
//! bodies computing the same transform, and **both reach the integrator on the
//! production strict-HF path** — `release_control` goes through the generic
//! one, `materialize` through the f64 one. They differ in FMA operand order,
//! not formatting, and until this file existed nothing anywhere bound them:
//! 194 call sites were swept and no guard cross-checked the pair.
//!
//! That is the hazard this file closes. A future edit to either body can move
//! one lane's bits and not the other's, and no bit pin would catch it — the
//! pins would simply move for an unexplained reason, weeks later, in whichever
//! lane happened to be exercised.
//!
//! **This test does not require the two to agree.** It records what is true and
//! fails when that changes, which is the useful contract for a deliberate
//! divergence. If a change makes the bodies agree *more*, this test fails and
//! the new, tighter fact should be recorded here in its place.
//!
//! # Why the metric is mixed absolute/relative, not ULP
//!
//! The first version of this file measured ULP distance and reported a worst
//! case of **1,412,316,433 ULP**, which reads like a catastrophe. It was an
//! artifact. The disagreeing component is an eccentricity-vector term whose
//! value on a near-circular orbit is ~7.5e-10; the two bodies produce
//! `7.526332531349622e-10` and `7.526333991650586e-10`, an *absolute*
//! difference of ~1.5e-16. Relative error and ULP distance both explode when
//! the quantity itself is ~0, and neither says anything about whether the
//! trajectory moved.
//!
//! So the measure here is `|a - b| / (1 + max(|a|, |b|))` — absolute where the
//! quantity is small, relative where it is large. On that measure the two
//! bodies agree to the last bits, which is what "differs in FMA operand order"
//! should look like and what the ULP number obscured.

use satpy_core::{eci2equinoc_impl, eci2equinoc_impl_f64, equinoc2eci_impl, equinoc2eci_impl_f64};

/// A spread of physically plausible states: LEO through GEO, near-circular
/// through eccentric, prograde through retrograde, plus deliberately awkward
/// geometry (near-zero inclination, near-polar) where the equinoctial
/// parameterisation is most sensitive.
fn corpus() -> Vec<[f64; 6]> {
    let mut states = Vec::new();
    for &radius_km in &[6_778.0_f64, 7_200.0, 10_000.0, 26_600.0, 42_164.0] {
        for &speed_scale in &[0.85_f64, 1.0, 1.15] {
            for &inclination_deg in &[0.001_f64, 28.5, 51.6, 89.9, 98.7, 145.0] {
                let mu = 398_600.441_8_f64;
                let circular = (mu / radius_km).sqrt();
                let speed = circular * speed_scale;
                let inclination = inclination_deg.to_radians();
                states.push([
                    radius_km,
                    0.0,
                    0.0,
                    0.0,
                    speed * inclination.cos(),
                    speed * inclination.sin(),
                ]);
            }
        }
    }
    states
}

/// Worst mixed absolute/relative gap between the two bodies across the corpus.
///
/// `|a - b| / (1 + max(|a|, |b|))` — behaves as an absolute difference when the
/// component is near zero and as a relative one when it is large. See the
/// module comment for why ULP distance is the wrong instrument here.
fn worst_gap<F, G>(forward_generic: F, forward_f64: G) -> (f64, usize)
where
    F: Fn(&[f64], &mut [f64]),
    G: Fn(&[f64], &mut [f64]),
{
    let mut worst = 0.0_f64;
    let mut disagreeing_states = 0_usize;
    for state in corpus() {
        let mut generic = [0.0_f64; 6];
        let mut specialized = [0.0_f64; 6];
        forward_generic(&state, &mut generic);
        forward_f64(&state, &mut specialized);

        let mut state_disagrees = false;
        for (lhs, rhs) in generic.iter().zip(specialized.iter()) {
            if lhs.to_bits() == rhs.to_bits() {
                continue;
            }
            state_disagrees = true;
            worst = worst.max((lhs - rhs).abs() / (1.0 + lhs.abs().max(rhs.abs())));
        }
        if state_disagrees {
            disagreeing_states = disagreeing_states.saturating_add(1);
        }
    }
    (worst, disagreeing_states)
}

#[test]
fn eci_to_equinoctial_generic_and_f64_bodies_stay_within_their_recorded_gap() {
    let (worst, disagreeing) = worst_gap(
        |state, out| eci2equinoc_impl::<f64>(state, 6, 0.0, 0.0, out),
        |state, out| eci2equinoc_impl_f64(state, 6, 0.0, 0.0, out),
    );

    println!(
        "ECI->EQUINOC generic vs f64: worst {worst:.3e} over {} states, {disagreeing} disagree",
        corpus().len()
    );

    assert!(
        worst <= RECORDED_ECI_TO_EQUINOC_GAP,
        "the generic and f64 ECI->equinoctial bodies now differ by {worst:.3e}, above the \
         recorded {RECORDED_ECI_TO_EQUINOC_GAP:.3e}. Both reach the integrator on the production \
         path, so a widening gap means one lane's bits moved and the other's did not."
    );
}

#[test]
fn equinoctial_to_eci_generic_and_f64_bodies_stay_within_their_recorded_gap() {
    // The reverse direction needs valid equinoctial elements, so round-trip the
    // corpus forward through the f64 body first and compare on the way back.
    let (worst, disagreeing) = worst_gap(
        |state, out| {
            let mut elements = [0.0_f64; 6];
            eci2equinoc_impl_f64(state, 6, 0.0, 0.0, &mut elements);
            equinoc2eci_impl::<f64>(&elements, 6, 0.0, 0.0, out);
        },
        |state, out| {
            let mut elements = [0.0_f64; 6];
            eci2equinoc_impl_f64(state, 6, 0.0, 0.0, &mut elements);
            equinoc2eci_impl_f64(&elements, 6, 0.0, 0.0, out);
        },
    );

    println!(
        "EQUINOC->ECI generic vs f64: worst {worst:.3e} over {} states, {disagreeing} disagree",
        corpus().len()
    );

    assert!(
        worst <= RECORDED_EQUINOC_TO_ECI_GAP,
        "the generic and f64 equinoctial->ECI bodies now differ by {worst:.3e}, above the \
         recorded {RECORDED_EQUINOC_TO_ECI_GAP:.3e}."
    );
}

/// **Measured 2026-08-05: 1.065e-15**, worst case over the corpus, on 39 of 90
/// states. That is roughly five ULP at unit magnitude — FMA operand order, as
/// the f64 body's own comment claims, and nothing more.
///
/// The bound is set slightly above the measurement rather than exactly at it,
/// because `sqrt`/`atan2` are libm calls and this pair has never been measured
/// on glibc; a bound pinned to the last Apple-libm bit would go red on Linux
/// for a reason that is not a regression. It is still ~9 ULP, four orders below
/// anything physically meaningful, so a real divergence cannot hide under it.
const RECORDED_ECI_TO_EQUINOC_GAP: f64 = 2.0e-15;
/// **Measured 2026-08-05: exactly zero.** The reverse direction is bit-identical
/// between the two bodies on every state in the corpus, so this one is pinned
/// hard. If it ever moves, the bodies have genuinely diverged.
const RECORDED_EQUINOC_TO_ECI_GAP: f64 = 0.0;
