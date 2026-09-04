//! Osculating <-> mean equinoctial element conversion for the secular-J2 lane.
//!
//! # Why this exists
//!
//! [`crate::advance_equinoc_j2_impl`] is a first-order SECULAR J2 propagator on
//! MEAN equinoctial elements: it advances the node, perigee and mean longitude at
//! the classical rates and holds `a`, `e`, `i` fixed. It is only valid when seeded
//! with mean elements, and it returns mean elements.
//!
//! Catalogue targets do not arrive that way. Per the compiled event-anchor
//! authority they are SGP4 outputs transformed TEME -> ITRS -> CIRS -> GCRS, i.e.
//! instantaneous state vectors, whose algebraic conversion to equinoctial yields
//! OSCULATING elements. Feeding those to a mean propagator biases the semi-major
//! axis by the J2 short-period offset, which biases mean motion, which drifts
//! along-track WITHOUT BOUND: measured 39.5 km after one revolution and 735.6 km
//! after 18.6 for a 7097 km, i = 74 deg target — linear to four significant
//! figures, the signature of a mean-motion error rather than a periodic term.
//!
//! # Method, and why it is not Brouwer-Lyddane
//!
//! The classical route is the first-order Brouwer (1959) short-period generating
//! function, singular at small `e` and small `i`, repaired by Lyddane (1963).
//! Transcribing those coefficients by hand is the single highest-risk way to
//! "fix" this: a wrong coefficient produces a result that looks converged and is
//! wrong, and nothing downstream would catch it.
//!
//! This module instead uses the defining property directly. Under J2 the
//! osculating semi-major axis has NO secular drift — J2 does no net work over a
//! revolution — so its average over exactly one orbital period IS the mean
//! semi-major axis. The same averaging removes the short-period content of every
//! other element. That is exact to first order, has no singularity at `e -> 0` or
//! at any inclination, and requires no transcribed coefficients. It costs one
//! numerical revolution per conversion, which is an ingestion-time operation
//! performed once per target per event, not a hot loop.
//!
//! The inverse (mean -> osculating) is obtained by fixed-point iteration on the
//! forward map, which converges in a handful of steps because the correction is
//! O(J2) ~ 1e-3.
//!
//! # Measured effect and the operating point
//!
//! Along-track position error against numerical J2 truth, for the 7097 km
//! i = 74 deg fixture target (`tests::conversion_removes_the_secular_drift` and
//! `tests::forward_only_conversion_is_the_affordable_operating_point` print these):
//!
//! | seeding | 1 rev | 5 revs | 18.62 revs | character |
//! |---|---|---|---|---|
//! | osculating, raw | 39.7 km | 198.4 | 738.5 | UNBOUNDED, linear |
//! | mean, forward only | 4.41 km | 4.46 | 3.12 | BOUNDED |
//! | mean, forward + inverse | 0.069 km | 0.35 | 1.37 | best |
//!
//! The forward conversion alone converts an unbounded drift into a bounded
//! short-period residual. That is the qualitative fix; the inverse then buys a
//! further ~15x on the residual.
//!
//! **The forward-only row was 24.4 / 24.7 / 24.3 km until 2026-07-25, and about
//! 83% of that was an epoch-registration artifact rather than physics.** The
//! averaging accumulated `(h, k, p, q)` raw, so it returned elements dated to the
//! centroid of the averaging window instead of the input epoch. Fixed by
//! referring each sample back through [`crate::advance_equinoc_j2_impl`] before
//! accumulating it. A second, smaller registration error remained after that —
//! the de-rotation took its rates from each SAMPLE, whose `a` carries the
//! short-period offset that feeds `lambda_dot`'s Keplerian term — and removing
//! it took the row from 6.25 / 6.06 / 4.46 to the values above. Anyone comparing
//! against numbers from before 2026-07-25 should know they were measuring a
//! timestamp error, not physics.
//!
//! **Note the forward+inverse column at 18.62 revs went 0.30 -> 1.37 km in that
//! second fix, and the larger number is the more trustworthy one.** 1.37 km over
//! 18.62 revs is 0.074 km/rev, which matches the O(J2^2) secular truncation this
//! method cannot remove. The 0.30 km it replaced is 0.016 km/rev — BELOW that
//! floor, i.e. a fortuitous cancellation between two registration errors rather
//! than a better answer. Forward-only, the row production would actually use,
//! improved on every horizon.
//!
//! Cost is two numerical revolutions of RK4 for the forward map (the first
//! establishes the mean `a` that sets the second's de-rotation rate), and
//! `MAX_INVERSE_ITERS` of those for the inverse. This is an ingestion-time
//! operation performed once per target per event, not a hot loop. The inverse
//! remains intended for ingestion-side and diagnostic use rather than
//! per-propagation output.
//!
//! No committed benchmark measures either figure; earlier revisions of this file
//! quoted 60.6 us / 436.8 us, which appear in no bench in the tree and should not
//! be cited.
//!
//! # Scope
//!
//! J2 only, matching the lane this feeds. It deliberately does not model drag,
//! SRP or higher-order gravity: converting against a richer force model than the
//! propagator uses would introduce a new inconsistency in place of the one it
//! removes.

use crate::{eci2equinoc_impl_f64, equinoc2eci_impl, J2, MU, RE};

/// Samples per revolution used to average out short-period content. The J2
/// short-period terms are dominated by the 2u harmonic, so a few dozen samples
/// already resolve them; 64 leaves generous margin and keeps the cost trivial.
const SAMPLES_PER_REV: u8 = 64;

/// RK4 substeps between consecutive samples.
const SUBSTEPS: u8 = 8;

/// Maximum fixed-point iterations for the mean -> osculating inverse.
const MAX_INVERSE_ITERS: usize = 12;

/// Convergence tolerance on the inverse. Applied RELATIVE to `a` (which is in km)
/// and absolute to the five dimensionless components. It must be relative for
/// `a`: as an absolute km bound, 1e-11 is 1.4e-15 relative at LEO, which is under
/// the floating-point noise floor of the RK4 revolution the forward map runs, so
/// the iteration reaches that floor and then random-walks below the tolerance
/// without ever satisfying it.
const INVERSE_TOL: f64 = 1.0e-11;

#[inline]
fn j2_acceleration(position_km: [f64; 3]) -> [f64; 3] {
    let [position_x, position_y, position_z] = position_km;
    let radius_squared =
        position_x * position_x + position_y * position_y + position_z * position_z;
    let radius = radius_squared.sqrt();
    let two_body = -MU / (radius_squared * radius);
    let j2_scale = 1.5 * J2 * MU * RE * RE / (radius_squared * radius_squared * radius);
    let z_ratio = position_z * position_z / radius_squared;
    [
        two_body * position_x + j2_scale * position_x * (5.0 * z_ratio - 1.0),
        two_body * position_y + j2_scale * position_y * (5.0 * z_ratio - 1.0),
        two_body * position_z + j2_scale * position_z * (5.0 * z_ratio - 3.0),
    ]
}

#[inline]
fn derivative(state: &[f64; 6]) -> [f64; 6] {
    let &[position_x, position_y, position_z, velocity_x, velocity_y, velocity_z] = state;
    let [acceleration_x, acceleration_y, acceleration_z] =
        j2_acceleration([position_x, position_y, position_z]);
    [
        velocity_x,
        velocity_y,
        velocity_z,
        acceleration_x,
        acceleration_y,
        acceleration_z,
    ]
}

fn rk4_step(state: &[f64; 6], dt: f64) -> [f64; 6] {
    let k1 = derivative(state);
    let mut tmp = [0.0; 6];
    for ((next, current), slope) in tmp.iter_mut().zip(state).zip(&k1) {
        *next = *current + 0.5 * dt * *slope;
    }
    let k2 = derivative(&tmp);
    for ((next, current), slope) in tmp.iter_mut().zip(state).zip(&k2) {
        *next = *current + 0.5 * dt * *slope;
    }
    let k3 = derivative(&tmp);
    for ((next, current), slope) in tmp.iter_mut().zip(state).zip(&k3) {
        *next = *current + dt * *slope;
    }
    let k4 = derivative(&tmp);
    let mut out = [0.0; 6];
    for ((((next, current), first), second), third_and_fourth) in out
        .iter_mut()
        .zip(state)
        .zip(&k1)
        .zip(&k2)
        .zip(k3.iter().zip(&k4))
    {
        let (third, fourth) = third_and_fourth;
        *next = *current + dt / 6.0 * (*first + 2.0 * *second + 2.0 * *third + *fourth);
    }
    out
}

/// Osculating semi-major axis from a Cartesian state via the vis-viva relation.
fn osculating_sma(state: &[f64; 6]) -> Option<f64> {
    let &[position_x, position_y, position_z, velocity_x, velocity_y, velocity_z] = state;
    let radius =
        (position_x * position_x + position_y * position_y + position_z * position_z).sqrt();
    let speed_squared = velocity_x * velocity_x + velocity_y * velocity_y + velocity_z * velocity_z;
    if !matches!(radius.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        return None;
    }
    let energy_term = 2.0 / radius - speed_squared / MU;
    if !matches!(
        energy_term.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    ) {
        return None; // unbound
    }
    Some(1.0 / energy_term)
}

fn equinoctial_of(state: &[f64; 6]) -> Option<[f64; 6]> {
    let mut equ = [0.0; 6];
    eci2equinoc_impl_f64(state, 6, 0.0, 0.0, &mut equ);
    equ.iter().all(|v| v.is_finite()).then_some(equ)
}

/// Unwrap `value` to the branch nearest `reference`.
#[inline]
fn unwrap_near(value: f64, reference: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    value + two_pi * ((reference - value) / two_pi).round()
}

/// One averaging pass over a revolution, with every sample referred back to the
/// input epoch before it is accumulated. Returns the summed equinoctial
/// components and the summed mean longitude, or `None` if the arc leaves the
/// finite domain.
///
/// # Why the samples must be de-rotated first
///
/// Averaging the equinoctial components RAW does not return the elements at the
/// input epoch. It returns them at the centroid of the averaging window, roughly
/// half a period late, because four of the six components carry secular motion:
/// `(p, q)` rotate at `Omega_dot` and `(h, k)` at `Omega_dot + omega_dot`. Only
/// `a` is secularly invariant under first-order J2 and can be averaged directly.
///
/// The registration error that produces is NOT small and it is NOT the
/// short-period residual it resembles. Measured on the fixture target, the raw
/// average sits 24.4 km from J2 truth while the epoch-referred average sits
/// 4.4 km away — so about 83% of what this module previously reported as
/// "bounded short-period residual" was a timestamp error, not physics. It stayed
/// hidden because the forward+inverse round trip cancels it identically: the
/// centroid offset is a displacement along the secular flow, so the inverse
/// fixed point undoes exactly what the forward map introduced. Only the
/// forward-only path — the one production uses — was exposed.
///
/// The de-rotation runs through [`crate::advance_equinoc_j2_impl`] at
/// `delta_t = -elapsed` rather than through locally rewritten rate formulas. That
/// is deliberate: it is the same kernel the propagator this feeds will use, so
/// the conversion cannot drift from the model it exists to seed, and there is no
/// second copy of the `Omega_dot` / `omega_dot` / `M_dot` expressions to keep in
/// agreement. It also fixes the mean longitude for free — the kernel de-trends
/// `lambda` at the full secular rate `Omega_dot + omega_dot + M_dot`, where this
/// function previously subtracted only the Keplerian `n`.
fn average_over_revolution(
    state: &[f64; 6],
    sample_dt: f64,
    rate_sma_km: f64,
) -> Option<([f64; 5], f64)> {
    let step = sample_dt / f64::from(SUBSTEPS);
    let mut acc = [0.0_f64; 5];
    let mut lambda_acc = 0.0_f64;
    let mut lambda_prev = 0.0_f64;
    let mut current = *state;
    let mut elapsed = 0.0_f64;

    for sample in 0..SAMPLES_PER_REV {
        let equ = equinoctial_of(&current)?;

        // Refer this sample back to the input epoch along the secular flow.
        //
        // The rate-determining semi-major axis is substituted for `rate_sma_km`
        // before the kernel sees it. The kernel derives every secular rate from
        // the element set it is handed, and `lambda_dot`'s leading term is the
        // Keplerian `n = sqrt(mu/a^3)`. Feeding it the SAMPLE's `a` feeds it the
        // short-period `delta a`, which does NOT average out of the de-rotation
        // because the correction is weighted by `elapsed` and
        // `<delta_a(t) * t> != 0` over the window. That leaves a first-order
        // registration error in `lambda`, measured at 7.27e-4 rad -- about
        // 5.2 km along-track, i.e. roughly 1.8 km of the residual this module
        // reports. Substituting the mean `a` removes it.
        //
        // Only `a` is substituted. The rates also depend on `e` and `i` through
        // `h,k,p,q`, but those enter the rate expressions multiplied by J2 and
        // their short-period content is therefore second order in the
        // correction; the angles themselves must stay the sample's, since they
        // are what is being rotated.
        let [sample_sma, sample_h, sample_k, sample_p, sample_q, sample_lambda] = equ;
        let rate_ref = [
            rate_sma_km,
            sample_h,
            sample_k,
            sample_p,
            sample_q,
            sample_lambda,
        ];
        let mut at_epoch = [0.0_f64; 6];
        crate::advance_equinoc_j2_impl(&rate_ref, -elapsed, &mut at_epoch);
        // Restore the sample's own semi-major axis; `a` is secularly invariant
        // under first-order J2, so the kernel passes it through unchanged and the
        // substitution above must not leak into what is averaged.
        if !at_epoch.iter().all(|v| v.is_finite()) {
            return None;
        }

        let [_, epoch_h, epoch_k, epoch_p, epoch_q, epoch_lambda] = at_epoch;
        let epoch_components = [sample_sma, epoch_h, epoch_k, epoch_p, epoch_q];
        for (sum, component) in acc.iter_mut().zip(epoch_components) {
            *sum += component;
        }
        // The kernel wraps lambda into [0, 2pi); unwrap to the nearest branch so
        // the sum does not straddle a discontinuity.
        let lambda = if sample == 0 {
            epoch_lambda
        } else {
            unwrap_near(epoch_lambda, lambda_prev)
        };
        lambda_prev = lambda;
        lambda_acc += lambda;

        if sample < SAMPLES_PER_REV.saturating_sub(1) {
            for _ in 0..SUBSTEPS {
                current = rk4_step(&current, step);
            }
            elapsed += sample_dt;
        }
    }
    Some((acc, lambda_acc))
}

/// Mean equinoctial elements from an osculating Cartesian state.
///
/// Averages the osculating elements over exactly one orbital period of J2 motion,
/// with every sample first referred back to the input epoch along the secular
/// flow. The result is therefore the mean element set AT THE INPUT EPOCH for all
/// six components, not at the window centroid — see
/// [`average_over_revolution`] for why that distinction is worth 20 km.
///
/// Two passes. The first establishes the mean semi-major axis; the second
/// re-runs the average with the de-rotation rates evaluated at that mean `a`
/// rather than at each sample's osculating one. See [`average_over_revolution`]
/// for why the difference is a first-order effect and not a refinement.
///
/// Returns `None` for non-finite input, a non-elliptical state, or a conversion
/// that fails to stay finite across the revolution.
#[must_use]
pub fn mean_equinoctial_from_osculating_state(state: &[f64; 6]) -> Option<[f64; 6]> {
    if !state.iter().all(|v| v.is_finite()) {
        return None;
    }
    let a0 = osculating_sma(state)?;
    let period = std::f64::consts::TAU * (a0 * a0 * a0 / MU).sqrt();
    if !period.is_finite() || period <= 0.0 {
        return None;
    }
    let sample_count = f64::from(SAMPLES_PER_REV);
    let sample_dt = period / sample_count;

    // Pass 1 establishes the mean semi-major axis, using the osculating `a` as
    // the de-rotation rate reference. Pass 2 re-runs with the mean one. `a` is
    // the only quantity that needs refining: it is secularly invariant, so its
    // average is already correct after one pass, and it is what sets the rate
    // that pass 1 got slightly wrong.
    let ([first_sma, _, _, _, _], _) = average_over_revolution(state, sample_dt, a0)?;
    let a_mean = first_sma / sample_count;
    if !matches!(a_mean.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        return None;
    }
    let ([mean_sma, mean_h, mean_k, mean_p, mean_q], lambda_acc) =
        average_over_revolution(state, sample_dt, a_mean)?;
    if !matches!(
        (mean_sma / sample_count).partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    ) {
        return None;
    }

    let mean = [
        mean_sma / sample_count,
        mean_h / sample_count,
        mean_k / sample_count,
        mean_p / sample_count,
        mean_q / sample_count,
        (lambda_acc / sample_count).rem_euclid(std::f64::consts::TAU),
    ];
    mean.iter().all(|v| v.is_finite()).then_some(mean)
}

/// Mean equinoctial elements from an osculating equinoctial set.
#[must_use]
pub fn mean_equinoctial_from_osculating(osculating: &[f64; 6]) -> Option<[f64; 6]> {
    let mut state = [0.0; 6];
    equinoc2eci_impl(osculating, 6, 0.0, 0.0, &mut state);
    if !state.iter().all(|v| v.is_finite()) {
        return None;
    }
    mean_equinoctial_from_osculating_state(&state)
}

/// Osculating equinoctial elements corresponding to a mean set.
///
/// Fixed-point inversion of [`mean_equinoctial_from_osculating`]: seed the
/// osculating guess with the mean elements and add the residual until the forward
/// map reproduces the requested mean set. The correction is O(J2), so this
/// converges geometrically.
#[must_use]
pub fn osculating_equinoctial_from_mean(mean: &[f64; 6]) -> Option<[f64; 6]> {
    let mut guess = *mean;
    for _ in 0..MAX_INVERSE_ITERS {
        let produced = mean_equinoctial_from_osculating(&guess)?;
        let [mean_sma, mean_h, mean_k, mean_p, mean_q, mean_lambda] = *mean;
        let [produced_sma, produced_h, produced_k, produced_p, produced_q, produced_lambda] =
            produced;
        // Angular residual on the shortest branch.
        let two_pi = std::f64::consts::TAU;
        let mut d_lambda = (mean_lambda - produced_lambda).rem_euclid(two_pi);
        if d_lambda > std::f64::consts::PI {
            d_lambda -= two_pi;
        }
        let residual = [
            mean_sma - produced_sma,
            mean_h - produced_h,
            mean_k - produced_k,
            mean_p - produced_p,
            mean_q - produced_q,
            d_lambda,
        ];

        for (guess_component, residual_component) in guess.iter_mut().zip(&residual) {
            *guess_component += *residual_component;
        }
        // `a` is in km and the other five are dimensionless, so one absolute
        // tolerance cannot serve both. Applying `INVERSE_TOL` to `a` directly
        // demanded 1.4e-15 RELATIVE at LEO -- below the noise floor of the RK4
        // revolution the forward map runs, so the `a` residual would reach the
        // floor and then random-walk there forever, and this function returned
        // `None` after `MAX_INVERSE_ITERS` for a converged answer. Scale it.
        let [residual_sma, _, _, _, _, _] = residual;
        let [guess_sma, _, _, _, _, _] = guess;
        let converged = residual_sma.abs() < INVERSE_TOL * guess_sma.abs().max(1.0)
            && residual
                .iter()
                .skip(1)
                .all(|component| component.abs() < INVERSE_TOL);
        if converged {
            return guess.iter().all(|v| v.is_finite()).then_some(guess);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ToPrimitive;

    /// A 7097 km, i = 74 deg, e = 6.7e-3 target — the fixture case whose raw
    /// osculating seeding drifts 39.5 km per revolution.
    const TARGET1: [f64; 6] = [
        -4_152.844_262_529_589,
        4_683.464_893_827_048,
        -3_443.780_513_960_792_6,
        0.507_038_319_140_948_9,
        -4.096_209_093_836_054_5,
        -6.195_722_320_889_406,
    ];

    fn propagate_truth(state: &[f64; 6], dt: f64, steps: usize) -> [f64; 6] {
        let h = dt / steps.to_f64().unwrap_or(1.0);
        let mut s = *state;
        for _ in 0..steps {
            s = rk4_step(&s, h);
        }
        s
    }

    fn position_error_km(a: &[f64; 6], b: &[f64; 6]) -> f64 {
        let [ax, ay, az, ..] = *a;
        let [bx, by, bz, ..] = *b;
        let dx = ax - bx;
        let dy = ay - by;
        let dz = az - bz;
        dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt()
    }

    #[test]
    fn mean_semi_major_axis_differs_from_osculating_by_the_short_period_offset() {
        let osc_a = osculating_sma(&TARGET1).expect("bound orbit");
        let mean = mean_equinoctial_from_osculating_state(&TARGET1).expect("conversion");
        let delta = mean[0] - osc_a;
        // First-order J2 short-period band for this inclination is about +-8.6 km.
        assert!(
            delta.abs() > 0.1 && delta.abs() < 12.0,
            "mean-minus-osculating a = {delta} km is outside the plausible \
             short-period band; conversion is suspect"
        );
    }

    #[test]
    fn round_trip_mean_to_osculating_reproduces_the_input() {
        let mean = mean_equinoctial_from_osculating_state(&TARGET1).expect("forward");
        let osc = osculating_equinoctial_from_mean(&mean).expect("inverse");
        let back = mean_equinoctial_from_osculating(&osc).expect("forward again");
        assert!(
            (back[0] - mean[0]).abs() < 1.0e-6,
            "round-trip a moved by {} km",
            back[0] - mean[0]
        );
        for (i, (back_value, mean_value)) in back.iter().zip(&mean).enumerate().skip(1).take(4) {
            assert!(
                (back_value - mean_value).abs() < 1.0e-9,
                "round-trip component {i} moved by {}",
                back_value - mean_value
            );
        }
    }

    /// THE POINT OF THE MODULE. Seeding the secular propagator with converted mean
    /// elements must remove the SECULAR along-track drift. The residual is then the
    /// bounded short-period amplitude, which does not grow with arc length.
    #[test]
    fn conversion_removes_the_secular_drift() {
        let period = {
            let a = osculating_sma(&TARGET1).unwrap();
            std::f64::consts::TAU * (a * a * a / MU).sqrt()
        };
        let osc_equ = equinoctial_of(&TARGET1).expect("equinoctial");
        let mean_equ = mean_equinoctial_from_osculating_state(&TARGET1).expect("mean");

        let mut raw = Vec::new();
        let mut converted = Vec::new();
        for revs in [1.0_f64, 5.0, 18.62] {
            let dt = revs * period;
            let truth =
                propagate_truth(&TARGET1, dt, (revs * 4000.0).to_usize().unwrap_or_default());

            // Raw: osculating elements fed straight to the secular propagator.
            let mut raw_state = [0.0; 6];
            crate::equinoc_prop_j2_from_impl(&osc_equ, dt, &mut raw_state);
            raw.push(position_error_km(&raw_state, &truth));

            // Converted: mean elements in, and the mean result mapped back out.
            let mut advanced = [0.0; 6];
            crate::advance_equinoc_j2_impl(&mean_equ, dt, &mut advanced);
            let osc_out = osculating_equinoctial_from_mean(&advanced).expect("inverse");
            let mut conv_state = [0.0; 6];
            equinoc2eci_impl(&osc_out, 6, 0.0, 0.0, &mut conv_state);
            converted.push(position_error_km(&conv_state, &truth));
        }

        println!("raw            (osculating seeded)     : {raw:?} km");
        println!("mean+inverse   (both directions)       : {converted:?} km");

        // Raw error grows linearly: the 18.62-rev error is ~18x the 1-rev error.
        let [raw_one, _, raw_long] = raw.as_slice() else {
            panic!("raw error vector must contain three arcs");
        };
        let [converted_one, _, converted_long] = converted.as_slice() else {
            panic!("converted error vector must contain three arcs");
        };
        assert!(
            raw_long / raw_one > 10.0,
            "expected the raw path to drift roughly linearly, got ratio {}",
            raw_long / raw_one
        );
        // The drift is not ELIMINATED - averaging is first order, so an O(J2^2)
        // mean-motion residual still accumulates - but the RATE collapses.
        let raw_rate = raw_one;
        let converted_rate = converted_one;
        assert!(
            *converted_rate < raw_rate / 100.0,
            "drift rate per revolution barely improved: raw {raw_rate} km/rev vs \
             converted {converted_rate} km/rev"
        );
        assert!(
            *converted_long < raw_long / 100.0,
            "long-arc error barely improved: raw {raw_long} km vs converted {converted_long} km"
        );
    }

    /// COST/BENEFIT for the production wiring decision.
    ///
    /// The forward map costs one numerical revolution. The fixed-point inverse
    /// costs several. Ingestion runs the forward map once per target per event —
    /// negligible. Applying the inverse on every propagation output would put a
    /// multi-revolution solve on the hot path, which is not affordable.
    ///
    /// This measures what is actually lost by shipping the FORWARD conversion
    /// only, and treating the propagated mean state as the position. The residual
    /// is then the bounded short-period amplitude, which does not accumulate.
    #[test]
    fn forward_only_conversion_is_the_affordable_operating_point() {
        let period = {
            let a = osculating_sma(&TARGET1).unwrap();
            std::f64::consts::TAU * (a * a * a / MU).sqrt()
        };
        let osc_equ = equinoctial_of(&TARGET1).expect("equinoctial");
        let mean_equ = mean_equinoctial_from_osculating_state(&TARGET1).expect("mean");

        let mut raw = Vec::new();
        let mut forward_only = Vec::new();
        for revs in [1.0_f64, 5.0, 18.62] {
            let dt = revs * period;
            let truth =
                propagate_truth(&TARGET1, dt, (revs * 4000.0).to_usize().unwrap_or_default());

            let mut raw_state = [0.0; 6];
            crate::equinoc_prop_j2_from_impl(&osc_equ, dt, &mut raw_state);
            raw.push(position_error_km(&raw_state, &truth));

            // Forward conversion only: no inverse, mean state used as position.
            let mut fwd_state = [0.0; 6];
            crate::equinoc_prop_j2_from_impl(&mean_equ, dt, &mut fwd_state);
            forward_only.push(position_error_km(&fwd_state, &truth));
        }

        println!("raw          (osculating seeded): {raw:?} km");
        println!("forward-only (mean seeded)      : {forward_only:?} km");

        // Forward-only leaves a BOUNDED short-period residual rather than a
        // growing one: the long arc must not be much worse than one revolution.
        let [_, _, raw_long] = raw.as_slice() else {
            panic!("raw error vector must contain three arcs");
        };
        let [forward_one, _, forward_long] = forward_only.as_slice() else {
            panic!("forward-only error vector must contain three arcs");
        };
        assert!(
            *forward_long < 4.0 * forward_one.max(1.0e-6),
            "forward-only residual is still growing secularly: {forward_only:?}"
        );
        // And it must still beat the raw path decisively on the long arc.
        assert!(
            *forward_long < raw_long / 20.0,
            "forward-only did not materially help: raw {raw_long} km vs forward-only {forward_long} km"
        );
    }

    /// Guards the sampling parameters. If someone lowers `SAMPLES_PER_REV` or
    /// SUBSTEPS for speed, this fails before the accuracy claim silently rots.
    #[test]
    fn sampling_resolution_is_sufficient_for_the_accuracy_claim() {
        let mean = mean_equinoctial_from_osculating_state(&TARGET1).expect("mean");
        // Halving the step must not move the mean semi-major axis materially.
        let a_coarse = mean[0];
        let refined = {
            // Re-average with 4x the substeps by propagating a denser truth arc.
            let a0 = osculating_sma(&TARGET1).unwrap();
            let period = std::f64::consts::TAU * (a0 * a0 * a0 / MU).sqrt();
            let samples = SAMPLES_PER_REV * 2;
            let dt = period / f64::from(samples);
            let step = dt / f64::from(SUBSTEPS * 2);
            let mut s = TARGET1;
            let mut acc = 0.0;
            for i in 0..samples {
                acc += osculating_sma(&s).unwrap();
                if i + 1 < samples {
                    for _ in 0..(SUBSTEPS * 2) {
                        s = rk4_step(&s, step);
                    }
                }
            }
            acc / f64::from(samples)
        };
        assert!(
            (a_coarse - refined).abs() < 1.0e-4,
            "mean a is resolution-dependent: {a_coarse} km at the shipped \
             resolution vs {refined} km refined; sampling is too coarse"
        );
    }
}
