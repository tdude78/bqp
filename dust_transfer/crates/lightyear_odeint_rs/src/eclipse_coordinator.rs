use std::borrow::Cow;
use std::sync::Arc;

use crate::odesolve::{
    EventDecision as OdeEventDecision, EventHandler as OdeEventHandler,
    IntegrationStatus as OdeIntegrationStatus,
};
use satpy_core::{eci2equinoc_impl_f64, equinoc2eci_impl, GravityError, PackedGravityCoeffs};

use crate::eclipse::{
    binary_cylinder_geometry, first_crossing_in_step, EclipseBracket, EclipseError,
    EclipseScanResult, EclipseSide, ScanEndpointState, MAX_BOUNDARY_SEPARATION_KM,
    PART_A_ECLIPSE_RADIUS_CAP_KM, PART_A_ECLIPSE_SPEED_CAP_KM_S,
};
use crate::integrator::{
    correct_delta_to_original_baseline, flatten_states, integrate_segment_with_method,
    slice_to_state, EnckeEventHandler, LightyearSystem, SegmentBoundary, SegmentControls,
    MAX_STEPS,
};
#[cfg(feature = "scalar-leg-observer")]
use crate::integrator::{FinalObservation, ObservedFinalMetricError};
use crate::rhs::{BaselineCalculator, LightyearRHS};
use crate::types::{ForceConfig, IntegrationResult, OdeMetrics, StepperMethod};

#[cfg(test)]
pub static TEST_ECLIPSE_SPLITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub static TEST_ECLIPSE_ROOTS: std::sync::Mutex<Vec<f64>> = std::sync::Mutex::new(Vec::new());
#[cfg(test)]
pub static TEST_HIDDEN_DOUBLE_ACCEPTED_STEPS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub static TEST_ROOT_TRANSACTION_RESETS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub static TEST_ROOT_TRANSACTION_CONTINUATIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_ECLIPSE_STATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
thread_local! {
    static TEST_ECLIPSE_CAPTURE_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Serialize tests that inspect the process-global eclipse trace counters.
///
/// Coordinator calls outside this guard intentionally do not mutate the
/// counters, so unrelated SRP tests cannot perturb an asserted trace between
/// two propagation calls.
#[cfg(test)]
#[must_use]
pub struct EclipseTestStateGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl Drop for EclipseTestStateGuard {
    fn drop(&mut self) {
        TEST_ECLIPSE_CAPTURE_ENABLED.with(|enabled| enabled.set(false));
    }
}

#[cfg(test)]
pub fn eclipse_test_state_guard() -> EclipseTestStateGuard {
    let lock = TEST_ECLIPSE_STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    TEST_ECLIPSE_CAPTURE_ENABLED.with(|enabled| enabled.set(true));
    EclipseTestStateGuard { _lock: lock }
}

#[cfg(test)]
fn eclipse_test_capture_enabled() -> bool {
    TEST_ECLIPSE_CAPTURE_ENABLED.with(std::cell::Cell::get)
}

#[cfg(test)]
fn record_root_transaction_reset() {
    if eclipse_test_capture_enabled() {
        TEST_ROOT_TRANSACTION_RESETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
fn record_root_transaction_continuation() {
    if eclipse_test_capture_enabled() {
        TEST_ROOT_TRANSACTION_CONTINUATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

const MAX_ROOT_TOTAL_UNCERTAINTY_KM: f64 = 1.0e-4;
const ROOT_OLD_SIDE_MARGIN_KM: f64 = 2.5e-5;

// How many times a root transaction may pull its old-side proof anchor back to
// a crossing its own proof leg found, before failing closed.
//
// `ROOT_OLD_SIDE_MARGIN_KM` is a TRANSPORT bound: it budgets carrying the
// detector's root estimate a few microseconds backward, not the error of the
// estimate itself. At a grazing (near-tangential) crossing the coarse
// detector's Hermite root and the fine-step re-scan's root can disagree by
// more than that transport window, so the proof point lands past the true
// root and the proof leg re-detects the crossing the transaction is trying to
// commit. That crossing is a strictly earlier, better-resolved anchor, so the
// recovery is a bounded restart from it — never a wider margin, which would
// move every transaction's arithmetic. Two restarts cover the observed
// grazing geometry (one disagreement plus one ratchet); anything deeper is a
// detector fault and keeps the fail-closed answer.
const MAX_OLD_SIDE_PROOF_RESTARTS: usize = 2;

// How much of the measured post-root transport error the scan-start margin must
// clear before its re-derived side counts as information rather than roundoff.
//
// 16x covers carrying that error across the skip window and the binary64 slop in
// re-deriving geometry from it; both are far below the error itself.
const POST_ROOT_TRANSPORT_SAFETY: f64 = 16.0;

// Absolute floor under the scan-start margin gate, independent of how small the
// measured transport error happens to be.
//
// One micrometre. Binary64 geometry roundoff at Part A radii is the scale to
// beat: at r ~ 7,000 km the axial dot product rounds at ~1e-12 km, and the
// radial margin, which `binary_cylinder_geometry` deliberately evaluates as
// `(radial^2 - R^2) / (radial + R)` to avoid catastrophic cancellation, rounds
// at ~6e-13 km. Call the total 1e-11 km. This floor is 100x that, and it is
// three orders above the largest transport error observed on a production root
// (3.8e-12 km). It exists so a rebase that happens to round-trip exactly
// (transport error 0.0) still cannot certify a side out of pure roundoff.
const MIN_POST_ROOT_DECISIVE_MARGIN_KM: f64 = 1.0e-9;

/// A committed root that a continuation may treat as certified, carrying the
/// error that transporting it onto the continuing lane's baseline actually cost.
///
/// The transport error is MEASURED, not assumed: `rebase_after_eclipse_root`
/// computes it exactly as the distance between the committed root position and
/// the position the equinoctial round trip carries it to, and refuses to publish
/// the root at all when it exceeds `MAX_BOUNDARY_SEPARATION_KM`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CertifiedRoot {
    t: f64,
    transport_error_km: f64,
}

/// The boundary margin the post-root scan start must strictly exceed before its
/// re-derived side is authoritative.
///
/// This replaced a flat comparison against `MAX_BOUNDARY_SEPARATION_KM`, which
/// conflated two different quantities. `MAX_BOUNDARY_SEPARATION_KM` is the
/// CROSSING DETECTOR's resolution: the geometry-motion bound `first_crossing_in_
/// step` brackets a root to. The quantity that actually contaminates geometry at
/// the scan start is something else entirely - the equinoctial round-trip error
/// that carried the committed root onto this lane's baseline - and that is six
/// orders of magnitude smaller than the detector resolution in practice.
///
/// Using the detector resolution as the transport bound silently imposed a
/// minimum crossing transversality that nothing declared or checked. The scan
/// start sits at most `MAX_ROOT_TOTAL_UNCERTAINTY_KM / PART_A_ECLIPSE_SPEED_CAP_
/// KM_S = 5 us` past the root, so clearing a 1 mm margin there required the
/// boundary-normal approach speed to exceed `1e-6 / 5e-6 = 0.2 km/s`. Any
/// crossing shallower than that - below about 1.6 degrees to the shadow cylinder
/// at LEO speed - was structurally unpropagable by every path in this system,
/// including production exact refinement, with geometry that was in fact clean.
///
/// NORAD 40054 at t0 + 62,804.6376 s is one: measured transport error
/// 1.07e-12 km, scan-start margin 4.60e-7 km growing monotonically and linearly
/// at 1.24e-2 of along-track displacement, i.e. a boundary-normal approach of
/// 0.0906 km/s. Decisive by five orders of magnitude against every error term
/// present, and rejected as roundoff.
///
/// The returned bound is CLAMPED AT `MAX_BOUNDARY_SEPARATION_KM` so it can never
/// exceed the threshold it replaced. That makes bit-neutrality a theorem rather
/// than a measurement: `margin > MAX_BOUNDARY_SEPARATION_KM` implies
/// `margin > post_root_decisive_margin_km(..)`, so every step this gate used to
/// admit it still admits, from the identical `edge` with the identical
/// `scan_from`. Only steps that used to error can change.
#[expect(
    clippy::manual_clamp,
    reason = "f64::clamp panics when max < min; this gate is on the flown path and must \
              degrade rather than panic if the two constants are ever reordered"
)]
fn post_root_decisive_margin_km(transport_error_km: f64) -> Result<f64, EclipseError> {
    if !(transport_error_km.is_finite() && transport_error_km >= 0.0) {
        return Err(EclipseError::Geometry);
    }
    let scaled = outward_nonnegative(POST_ROOT_TRANSPORT_SAFETY * transport_error_km)?;
    Ok(scaled
        .max(MIN_POST_ROOT_DECISIVE_MARGIN_KM)
        .min(MAX_BOUNDARY_SEPARATION_KM))
}
// The accepted-step Hermite path is only a crossing detector. Re-scan the
// discarded step with short, actual solver steps before committing physics.
// The detector/replay scan brackets geometry to one millimetre
// (`MAX_BOUNDARY_SEPARATION_KM`); the committed replay root is independently
// bounded to 0.10 m by `MAX_ROOT_TOTAL_UNCERTAINTY_KM` above.
//
// 10.0 s, raised from 2.0 s on 2026-08-03 (perf recovery, B500-gated). The
// clamp is a DETECTOR obligation on every root-transaction leg, so it must
// stay global: a structural attempt to run only the straddle leg fine broke
// the Hermite detector on coarse approach legs and failed the release B500
// event-0 gate with a deterministic EclipseBracket.
//
// THE SOUNDNESS EDGE IS BETWEEN 12 AND 15 s, NOT 60. This comment used to
// carry `2/4/5/10/60 s -> 861/752/672/570/Err(Bracket)`. That sweep was taken
// on the V1 model-4 arc, which is NOT the arc the campaign flies. Re-measured
// on the model-5 arc, in
// docs/plans/2026-07-31-part-a-200-generation-fast-hybrid.md, section
// "2026-08-03 eclipse refinement clamp sweep":
//
//   clamp s |  2  |  5  | 10  | 12  | 15  | 25+
//   steps   | 892 | 632 | 570 | 593 | 570 | Err(Bracket)
//
// and the first failure, at 15 s, is SILENT: the arc still propagates and
// posts plausible counts (570 steps, indistinguishable from the incumbent)
// while one lib test fails `Bracket` --
// `integrator::tests::coordinator_commits_hidden_double_crossing_forward_and_backward_in_order`.
// Failure is also NON-MONOTONE in this constant -- 12 s is
// sound but COSTLIER than 10 s, 593 steps against 570, and 28 s propagates
// the model-5 arc while the model-4 arc misbrackets -- so a surviving step
// count above the edge is luck wearing a misbracket, not speed.
//
// DO NOT READ THAT SWEEP AS AN ACCURACY ORDERING. Every accuracy reading in
// it (0.012-0.115 m) sits BELOW that metric's own measured noise floor, so the
// ordering across clamp values is step-sequence chaos. The floor was 1.035 m
// when this was written and re-measured to 0.20 m at ba6a249 (see
// `strict_hf_pin::ACCURACY_METRIC_FLOOR_M`); the margin narrows from 9x-86x to
// 1.7x-17x and the conclusion is unchanged.
// The previous claim here that accuracy "improves through 10 s" is withdrawn:
// it was reading noise.
//
// What actually justifies 10.0 s is that it is the cheapest value that keeps
// the detector sound, and that the guard is the lib detector arm -- the
// `lightyear_odeint_rs` lib suite, which is the only instrument that sees the
// silent 15 s failure (237 tests when the sweep was taken; do not re-derive a
// pass criterion from that count, it drifts) -- plus the fail-closed 0.10 m
// root budget. Do not raise it without re-running BOTH the lib suite and the
// B500 event-0 gate; the lib suite is the one that matters, because at 15 s
// the arc alone reports success.
//
// AND THE WHOLE JUSTIFICATION IS EMPIRICAL. There is no analytic bound behind
// 10.0 s anywhere in this crate: what exists is the sweep above, and that sweep
// was taken on TWO arcs (the V1 model-4 arc and the model-5 arc the campaign
// flies). Because the failure is non-monotone in this constant, two arcs cannot
// establish where the edge is for a third — they can only show that 10 s
// survived on both while 15 s did not. A value chosen this way is a value with
// margin nobody has measured, not a value with a proof.
//
// Kept at 10.0 s for the campaign on that basis: it is the incumbent, it is the
// cheapest surviving value on both arcs, and the campaign is not the place to
// find out where the edge really is. Deriving an actual bound on the clamp from
// the Hermite detector's bracketing condition — rather than sampling arcs until
// one breaks — is a POST-CAMPAIGN study, and it is the only thing that would
// let this constant move on evidence rather than on luck.
const MAX_ROOT_REFINEMENT_STEP_S: f64 = 10.0;
use crate::integrator::MAX_RECTIFICATION_SEGMENT_S as MAX_RECT_SEGMENT;
const MAX_ECLIPSE_SPLITS: usize = 4096;

#[inline]
fn same_time_value(left: f64, right: f64) -> bool {
    matches!(left.partial_cmp(&right), Some(std::cmp::Ordering::Equal))
}

#[inline]
fn micros_to_u64(micros: u128) -> u64 {
    u64::try_from(micros).map_or(u64::MAX, |value| value)
}

#[inline]
fn elapsed_micros(start: std::time::Instant) -> u64 {
    micros_to_u64(start.elapsed().as_micros())
}

const fn coordinator_metrics(
    total_steps: usize,
    total_evals: usize,
    total_time_us: u64,
    eclipse_collapsed_pairs: usize,
) -> OdeMetrics {
    let mut metrics = OdeMetrics::from_values(total_steps, total_evals, total_time_us);
    metrics.eclipse_collapsed_pairs = eclipse_collapsed_pairs;
    metrics
}

fn checked_count_add(count: &mut usize, amount: usize) -> Result<(), EclipseError> {
    *count = count.checked_add(amount).ok_or(EclipseError::SplitLimit)?;
    Ok(())
}

/// Resolve one direct scalar-solver result after draining its eclipse latch.
/// The solver has already consumed its gravity latch and returns that exact
/// error before status; gravity still wins while the eclipse latch is drained.
#[inline]
fn resolve_solver_boundary<T>(
    rhs: &LightyearRHS,
    result: Result<T, GravityError>,
) -> Result<T, EclipseError> {
    let eclipse = rhs.take_eclipse_error();
    match result {
        Err(error) => Err(EclipseError::Gravity(error)),
        Ok(value) => eclipse.map_or_else(|| Ok(value), Err),
    }
}

fn segmented_endpoint(current_t: f64, remaining: f64, final_t: f64) -> f64 {
    if remaining.abs() > MAX_RECT_SEGMENT {
        current_t + remaining.signum() * MAX_RECT_SEGMENT
    } else {
        final_t
    }
}

#[inline]
fn hermite_position_velocity(
    p0: [f64; 3],
    v0: [f64; 3],
    p1: [f64; 3],
    v1: [f64; 3],
    t0: f64,
    t1: f64,
    t: f64,
) -> Result<([f64; 3], [f64; 3]), EclipseError> {
    let h = t1 - t0;
    if !h.is_finite() || same_time_value(h, 0.0) || !t.is_finite() {
        return Err(EclipseError::NonProgress);
    }
    let tau = ((t - t0) / h).clamp(0.0, 1.0);
    let tau2 = tau * tau;
    let tau3 = tau2 * tau;
    let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
    let h10 = tau3 - 2.0 * tau2 + tau;
    let h01 = -2.0 * tau3 + 3.0 * tau2;
    let h11 = tau3 - tau2;
    let dh00 = (6.0 * tau2 - 6.0 * tau) / h;
    let dh10 = 3.0 * tau2 - 4.0 * tau + 1.0;
    let dh01 = (-6.0 * tau2 + 6.0 * tau) / h;
    let dh11 = 3.0 * tau2 - 2.0 * tau;
    let [p0_x, p0_y, p0_z] = p0;
    let [v0_x, v0_y, v0_z] = v0;
    let [p1_x, p1_y, p1_z] = p1;
    let [v1_x, v1_y, v1_z] = v1;
    let position = [
        h00 * p0_x + h10 * h * v0_x + h01 * p1_x + h11 * h * v1_x,
        h00 * p0_y + h10 * h * v0_y + h01 * p1_y + h11 * h * v1_y,
        h00 * p0_z + h10 * h * v0_z + h01 * p1_z + h11 * h * v1_z,
    ];
    let velocity = [
        dh00 * p0_x + dh10 * v0_x + dh01 * p1_x + dh11 * v1_x,
        dh00 * p0_y + dh10 * v0_y + dh01 * p1_y + dh11 * v1_y,
        dh00 * p0_z + dh10 * v0_z + dh01 * p1_z + dh11 * v1_z,
    ];
    if position
        .iter()
        .chain(velocity.iter())
        .all(|value| value.is_finite())
    {
        Ok((position, velocity))
    } else {
        Err(EclipseError::Geometry)
    }
}

#[inline]
fn distance3_squared(left: [f64; 3], right: [f64; 3]) -> f64 {
    let [left_x, left_y, left_z] = left;
    let [right_x, right_y, right_z] = right;
    let dx = right_x - left_x;
    let dy = right_y - left_y;
    let dz = right_z - left_z;
    dx.mul_add(dx, dy.mul_add(dy, dz * dz))
}

#[inline]
fn distance3(left: [f64; 3], right: [f64; 3]) -> f64 {
    distance3_squared(left, right).sqrt()
}

struct AcceptedStepReconstruction {
    p0: [f64; 3],
    v0: [f64; 3],
    p1: [f64; 3],
    v1: [f64; 3],
    t0: f64,
    t1: f64,
}

#[derive(Clone, Copy)]
struct AcceptedDeltaStep {
    state0: [f64; 6],
    derivative0: [f64; 6],
    state1: [f64; 6],
    derivative1: [f64; 6],
    t0: f64,
    t1: f64,
}

impl AcceptedDeltaStep {
    fn state_at(self, t: f64) -> Result<[f64; 6], EclipseError> {
        let inside = if self.t1 >= self.t0 {
            t >= self.t0 && t <= self.t1
        } else {
            t <= self.t0 && t >= self.t1
        };
        let h = self.t1 - self.t0;
        if !inside || !t.is_finite() || !h.is_finite() || same_time_value(h, 0.0) {
            return Err(EclipseError::Bracket);
        }
        let tau = ((t - self.t0) / h).clamp(0.0, 1.0);
        let tau2 = tau * tau;
        let tau3 = tau2 * tau;
        let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
        let h10 = tau3 - 2.0 * tau2 + tau;
        let h01 = -2.0 * tau3 + 3.0 * tau2;
        let h11 = tau3 - tau2;
        let mut state = [0.0; 6];
        for ((((value, state0), derivative0), state1), derivative1) in state
            .iter_mut()
            .zip(self.state0.iter())
            .zip(self.derivative0.iter())
            .zip(self.state1.iter())
            .zip(self.derivative1.iter())
        {
            *value =
                h00 * *state0 + h10 * h * *derivative0 + h01 * *state1 + h11 * h * *derivative1;
        }
        Ok(state)
    }
}

fn reconstruct_position_velocity(
    baseline_state: [f64; 6],
    delta: &[f64; 6],
    derivative: &[f64; 6],
) -> ([f64; 3], [f64; 3]) {
    let mut position = [0.0; 3];
    let mut velocity = [0.0; 3];
    for ((component, baseline_component), delta_component) in position
        .iter_mut()
        .zip(baseline_state.iter())
        .zip(delta.iter())
    {
        *component = *baseline_component + *delta_component;
    }
    for ((component, baseline_component), derivative_component) in velocity
        .iter_mut()
        .zip(baseline_state.iter().skip(3))
        .zip(derivative.iter())
    {
        *component = *baseline_component + *derivative_component;
    }
    (position, velocity)
}

impl AcceptedStepReconstruction {
    fn new(
        baseline: &BaselineCalculator<'_>,
        prev_t: f64,
        prev_delta: &[f64; 6],
        prev_derivative: &[f64; 6],
        next_t: f64,
        next_delta: &[f64; 6],
        next_derivative: &[f64; 6],
    ) -> Self {
        let (p0, v0) = reconstruct_position_velocity(
            baseline.get_baseline_state(prev_t),
            prev_delta,
            prev_derivative,
        );
        let (p1, v1) = reconstruct_position_velocity(
            baseline.get_baseline_state(next_t),
            next_delta,
            next_derivative,
        );
        Self {
            p0,
            v0,
            p1,
            v1,
            t0: prev_t,
            t1: next_t,
        }
    }

    fn position_velocity(&self, t: f64) -> Result<([f64; 3], [f64; 3]), EclipseError> {
        hermite_position_velocity(self.p0, self.v0, self.p1, self.v1, self.t0, self.t1, t)
    }

    /// This cubic Hermite state plus the dynamic Sun axis is the production
    /// numerical continuous path for eclipse event detection. The Bezier
    /// control-polygon bound lets the recursive scan prove every crossing on
    /// that path. It is not an exact ODE enclosure; tight-reference endpoint
    /// and root gates bound its approximation error. No reconstructed state is
    /// ever committed.
    ///
    /// The endpoint states arrive from the scan rather than being re-derived
    /// here. The scan evaluated this same interpolant at both instants to
    /// classify the endpoints, so recomputing them returned identical bits by
    /// construction; on a production arc that duplication was 71% of all Hermite
    /// evaluations.
    fn motion_bound_between(
        rhs: &LightyearRHS,
        old: ScanEndpointState,
        new: ScanEndpointState,
    ) -> Result<f64, EclipseError> {
        let (t0, p0, v0) = (old.t, old.position, old.velocity);
        let (t1, p1, v1) = (new.t, new.position, new.velocity);
        let h = t1 - t0;
        let c1 = [
            p0[0] + h * v0[0] / 3.0,
            p0[1] + h * v0[1] / 3.0,
            p0[2] + h * v0[2] / 3.0,
        ];
        let c2 = [
            p1[0] - h * v1[0] / 3.0,
            p1[1] - h * v1[1] / 3.0,
            p1[2] - h * v1[2] / 3.0,
        ];
        let satellite_path = distance3(p0, c1) + distance3(c1, c2) + distance3(c2, p1);
        // IEEE-754 `sqrt` is exactly rounded and monotone non-decreasing, so the
        // largest root is the root of the largest radicand. Reducing before the
        // root returns the same bits from one `sqrt` instead of four. The `0.0`
        // seed survives the move because `sqrt(0.0) == 0.0`, and a NaN radicand
        // is discarded by `f64::max` on either side of the root.
        let max_radius = [p0, c1, c2, p1]
            .iter()
            .map(|position| distance3_squared(*position, [0.0; 3]))
            .fold(0.0_f64, f64::max)
            .sqrt();
        let sun_angle = rhs.eclipse_sun_direction_path_bound(t0, t1)?;
        let bound = satellite_path + 2.0 * max_radius * sun_angle;
        if !bound.is_finite() {
            return Err(EclipseError::Geometry);
        }
        if same_time_value(bound, 0.0) {
            return Ok(0.0);
        }
        let rounded = f64::from_bits(
            bound
                .to_bits()
                .checked_add(8)
                .ok_or(EclipseError::Geometry)?,
        );
        if rounded.is_finite() {
            Ok(rounded)
        } else {
            Err(EclipseError::Geometry)
        }
    }

    fn scan_crossings(
        &self,
        rhs: &LightyearRHS,
        t0: f64,
        t1: f64,
    ) -> Result<EclipseScanResult, EclipseError> {
        first_crossing_in_step(
            t0,
            t1,
            rhs.config.earth_radius,
            MAX_BOUNDARY_SEPARATION_KM,
            MAX_ECLIPSE_SPLITS,
            |t| {
                let (position, velocity) = self.position_velocity(t)?;
                Ok((position, velocity, rhs.eclipse_sun_at(t)?))
            },
            |left, right| Self::motion_bound_between(rhs, left, right),
        )
    }
}

fn outward_nonnegative(value: f64) -> Result<f64, EclipseError> {
    if !(value.is_finite() && value >= 0.0) {
        return Err(EclipseError::Geometry);
    }
    if same_time_value(value, 0.0) {
        return Ok(0.0);
    }
    let rounded = f64::from_bits(
        value
            .to_bits()
            .checked_add(1)
            .ok_or(EclipseError::Geometry)?,
    );
    if rounded.is_finite() {
        Ok(rounded)
    } else {
        Err(EclipseError::Geometry)
    }
}

fn replay_root_uncertainty_km(rhs: &LightyearRHS, t0: f64, t1: f64) -> Result<f64, EclipseError> {
    let dt_bound = outward_nonnegative(PART_A_ECLIPSE_SPEED_CAP_KM_S * (t1 - t0).abs())?;
    let diameter = outward_nonnegative(2.0 * PART_A_ECLIPSE_RADIUS_CAP_KM)?;
    let sun_angle = rhs.eclipse_sun_direction_path_bound(t0, t1)?;
    let axis_bound = outward_nonnegative(diameter * sun_angle)?;
    outward_nonnegative(dt_bound + axis_bound)
}

fn deepest_directed_time_within_root_bound(
    rhs: &LightyearRHS,
    old_t: f64,
    directed_new_limit: f64,
    max_bound_km: f64,
) -> Result<f64, EclipseError> {
    if same_time_value(old_t, directed_new_limit)
        || !directed_new_limit.is_finite()
        || !(max_bound_km.is_finite() && max_bound_km > 0.0)
    {
        return Err(EclipseError::NonProgress);
    }
    if replay_root_uncertainty_km(rhs, old_t, directed_new_limit)? <= max_bound_km {
        return Ok(directed_new_limit);
    }
    let mut inside = old_t;
    let mut outside = directed_new_limit;
    for _ in 0..64 {
        let midpoint = inside + 0.5 * (outside - inside);
        if same_time_value(midpoint, inside) || same_time_value(midpoint, outside) {
            break;
        }
        if replay_root_uncertainty_km(rhs, old_t, midpoint)? <= max_bound_km {
            inside = midpoint;
        } else {
            outside = midpoint;
        }
    }
    if same_time_value(inside, old_t) || !inside.is_finite() {
        Err(EclipseError::NonProgress)
    } else {
        Ok(inside)
    }
}

struct CoordinatedEventHandler<'a> {
    rhs: &'a LightyearRHS,
    baseline: BaselineCalculator<'a>,
    fixed_side: EclipseSide,
    normal: Option<EnckeEventHandler<'a>>,
    eclipse_outcome: Option<EclipseBracket>,
    eclipse_step: Option<AcceptedDeltaStep>,
    eclipse_error: Option<EclipseError>,
    collapsed_pairs: usize,
    /// Committed root, if any, whose side a root transaction already proved, and
    /// the measured cost of transporting it onto this segment's baseline.
    /// Geometry re-derived at that instant is a sign query inside the root's own
    /// uncertainty ball, so it carries no information.
    certified_root: Option<CertifiedRoot>,
}

impl<'a> CoordinatedEventHandler<'a> {
    fn new(
        rhs: &'a LightyearRHS,
        fixed_side: EclipseSide,
        t0_s: f64,
        init_state: [f64; 6],
        event_interp_tol: f64,
        eps: f64,
        max_rejects: usize,
        enable_events: bool,
        certified_root: Option<CertifiedRoot>,
    ) -> Self {
        let normal = enable_events.then(|| {
            EnckeEventHandler::new(
                rhs.baseline_calculator(),
                t0_s,
                init_state,
                rhs,
                event_interp_tol,
                eps,
                max_rejects,
            )
        });
        Self {
            rhs,
            baseline: rhs.baseline_calculator(),
            fixed_side,
            normal,
            eclipse_outcome: None,
            eclipse_step: None,
            eclipse_error: None,
            collapsed_pairs: 0,
            certified_root,
        }
    }

    /// The committed root at `t`, if `t` is the single instant one proved.
    fn certified_root_at(&self, t: f64) -> Option<CertifiedRoot> {
        self.certified_root
            .filter(|certified| same_time_value(t, certified.t))
    }

    const fn take_eclipse_outcome(&mut self) -> Result<Option<EclipseBracket>, EclipseError> {
        match self.eclipse_error.take() {
            Some(error) => Err(error),
            None => Ok(self.eclipse_outcome.take()),
        }
    }

    fn take_collapsed_pairs(&mut self) -> usize {
        std::mem::take(&mut self.collapsed_pairs)
    }

    const fn take_eclipse_step(&mut self) -> Option<AcceptedDeltaStep> {
        self.eclipse_step.take()
    }

    fn take_detection(&mut self) -> Option<crate::types::EventDetection> {
        self.normal
            .as_mut()
            .and_then(EnckeEventHandler::take_detection)
    }

    fn take_event_invalid(&mut self) -> bool {
        self.normal
            .as_mut()
            .is_some_and(EnckeEventHandler::take_event_invalid)
    }
}

impl OdeEventHandler for CoordinatedEventHandler<'_> {
    fn on_step(
        &mut self,
        prev_t: f64,
        previous_state: &[f64],
        previous_derivative: &[f64],
        next_t: f64,
        next_state_values: &[f64],
        next_derivative: &[f64],
    ) -> OdeEventDecision {
        let prev_state = slice_to_state(previous_state);
        let next_state = slice_to_state(next_state_values);
        let outcome = (|| {
            self.rhs
                .validate_eclipse_envelope_at_delta(&prev_state, prev_t)?;
            self.rhs
                .validate_eclipse_envelope_at_delta(&next_state, next_t)?;
            let reconstruction = AcceptedStepReconstruction::new(
                &self.baseline,
                prev_t,
                &prev_state,
                &slice_to_state(previous_derivative),
                next_t,
                &next_state,
                &slice_to_state(next_derivative),
            );
            let old_position = reconstruction.position_velocity(prev_t)?.0;
            let old_sun = self.rhs.eclipse_sun_at(prev_t)?;
            let old_geometry =
                binary_cylinder_geometry(old_position, old_sun, self.rhs.config.earth_radius)?;
            // A committed root proves the side at exactly one instant, on the
            // root transaction's own baseline. Transporting that state onto
            // this segment's Encke baseline perturbs it by an equinoctial round
            // trip that can exceed the root's own distance to the boundary, so
            // geometry re-derived at that instant is roundoff. Only geometry
            // away from it is authoritative.
            let certified_root = self.certified_root_at(prev_t);
            if old_geometry.side != self.fixed_side && certified_root.is_none() {
                return Err(EclipseError::Chatter);
            }
            // Inside the committed root's uncertainty ball the sign of the
            // boundary margin is not resolvable. Begin the scan at the deepest
            // instant the root bound still covers, where geometry is decisive
            // again, and require it to agree with the committed side.
            //
            // This SKIPS `[prev_t, edge]`, and nothing scans it — line 679 is
            // the only production crossing scan, and every leg of the root
            // transaction stops at or before the root (`run_root_transaction_leg`
            // ends at the event: `validate_eclipse_root_commit` requires
            // `endpoint_t == event.t`). An earlier version of this comment said
            // the root transaction "has already certified that the only crossing
            // there is the one it committed"; it does not, and cannot —
            // `first_crossing_in_step` returns only the EARLIEST crossing, so a
            // scan that stopped at this root certifies nothing after it.
            //
            // The check below is also blind to what would hide there: it reads
            // geometry at the single instant `edge`, so an EVEN number of
            // crossings inside (a grazing out-and-back) leaves `edge` on the
            // committed side with a healthy margin and passes by construction.
            //
            // What actually makes the skip sound is the WINDOW'S DURATION, not a
            // certificate. `edge` is chosen so the whole span carries at most
            // `MAX_ROOT_TOTAL_UNCERTAINTY_KM` of certified geometry motion, and
            // that bound is `PART_A_ECLIPSE_SPEED_CAP_KM_S * dt` plus an axis
            // term, so `dt <= 1.0e-4 / 20 = 5 us`. Eclipse side gates SRP alone
            // (`compute_srp_with_precomputed` returns zero in shadow), so
            // dropping an excursion costs at most one SRP acceleration applied
            // with the wrong sign for 5 us: at the compiled dust parameters
            // 4.56e-6 N/m^2 * 1.3 * 1.948 m^2/kg = 1.15e-5 m/s^2, that is
            // 5.8e-11 m/s of velocity error, or 2.5 um over a 12 h arc. Four
            // hundred times under the one-millimetre bracket certificate that
            // the detector itself only resolves to.
            //
            // That argument is a function of `MAX_ROOT_TOTAL_UNCERTAINTY_KM`,
            // which nothing here re-derives, so `post_root_skip_window_stays_far_
            // under_the_bracket_certificate` pins it. Raise that constant and
            // the skipped window widens with no compensating check.
            let scan_from = if let Some(certified) = certified_root {
                let edge = deepest_directed_time_within_root_bound(
                    self.rhs,
                    prev_t,
                    next_t,
                    MAX_ROOT_TOTAL_UNCERTAINTY_KM,
                )?;
                let edge_position = reconstruction.position_velocity(edge)?.0;
                let edge_sun = self.rhs.eclipse_sun_at(edge)?;
                let edge_geometry = binary_cylinder_geometry(
                    edge_position,
                    edge_sun,
                    self.rhs.config.earth_radius,
                )?;
                // The margin here must beat the error that actually contaminates
                // it - the measured cost of transporting the committed root onto
                // this baseline - and NOT the crossing detector's one-millimetre
                // resolution, which is a bound on a different quantity and is six
                // orders of magnitude larger. See `post_root_decisive_margin_km`.
                let decisive_margin_km =
                    post_root_decisive_margin_km(certified.transport_error_km)?;
                if edge_geometry.side != self.fixed_side
                    || edge_geometry.boundary_margin_km <= decisive_margin_km
                {
                    return Err(EclipseError::Chatter);
                }
                edge
            } else {
                prev_t
            };
            let scan = reconstruction.scan_crossings(self.rhs, scan_from, next_t)?;
            self.collapsed_pairs = self
                .collapsed_pairs
                .checked_add(scan.collapsed_pairs)
                .ok_or(EclipseError::SplitLimit)?;
            let Some(mut bracket) = scan.crossing else {
                return Ok(None);
            };
            #[cfg(test)]
            {
                let next_position = reconstruction.position_velocity(next_t)?.0;
                let next_sun = self.rhs.eclipse_sun_at(next_t)?;
                let next_side = binary_cylinder_geometry(
                    next_position,
                    next_sun,
                    self.rhs.config.earth_radius,
                )?
                .side;
                if eclipse_test_capture_enabled()
                    && next_side == old_geometry.side
                    && reconstruction
                        .scan_crossings(self.rhs, bracket.t_new, next_t)?
                        .crossing
                        .is_some()
                {
                    TEST_HIDDEN_DOUBLE_ACCEPTED_STEPS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            let accepted_baseline = self.baseline.get_baseline_state(prev_t);
            let mut accepted_eci_old = [0.0; 6];
            for ((eci_component, baseline_component), delta_component) in accepted_eci_old
                .iter_mut()
                .zip(accepted_baseline.iter())
                .zip(prev_state.iter())
            {
                *eci_component = *baseline_component + *delta_component;
            }
            bracket.accepted_t_old = prev_t;
            bracket.accepted_t_new = next_t;
            bracket.accepted_eci_old = accepted_eci_old;
            if bracket.old_side != self.fixed_side {
                return Err(EclipseError::Chatter);
            }
            Ok(Some(bracket))
        })();

        let stop_at_accepted_endpoint = || OdeEventDecision::Stop {
            t_event: next_t,
            y_event: next_state.to_vec(),
        };
        let eclipse_outcome = match outcome {
            Err(error) => {
                self.eclipse_error = Some(error);
                return stop_at_accepted_endpoint();
            }
            Ok(outcome) => outcome,
        };
        let normal_decision = self.normal.as_mut().map(|handler| {
            handler.on_step(
                prev_t,
                previous_state,
                previous_derivative,
                next_t,
                next_state_values,
                next_derivative,
            )
        });
        let Some(bracket) = eclipse_outcome else {
            return normal_decision.unwrap_or(OdeEventDecision::Continue);
        };
        let detection = self.normal.as_ref().and_then(EnckeEventHandler::detection);
        let terminal_is_earlier = detection.is_some_and(|event| {
            if next_t >= prev_t {
                event.refined_time <= bracket.t_old
            } else {
                event.refined_time >= bracket.t_old
            }
        });
        let eclipse_is_earlier = detection.is_none_or(|event| {
            if next_t >= prev_t {
                event.refined_time >= bracket.t_new
            } else {
                event.refined_time <= bracket.t_new
            }
        });
        if terminal_is_earlier {
            if let Some(decision) = normal_decision {
                return decision;
            }
            self.eclipse_error = Some(EclipseError::Bracket);
            return stop_at_accepted_endpoint();
        }
        if !eclipse_is_earlier {
            self.eclipse_error = Some(EclipseError::EventOverlap);
            return stop_at_accepted_endpoint();
        }
        self.eclipse_outcome = Some(bracket);
        self.eclipse_step = Some(AcceptedDeltaStep {
            state0: prev_state,
            derivative0: slice_to_state(previous_derivative),
            state1: next_state,
            derivative1: slice_to_state(next_derivative),
            t0: prev_t,
            t1: next_t,
        });
        stop_at_accepted_endpoint()
    }
}

const fn integration_status_name(status: OdeIntegrationStatus) -> &'static str {
    match status {
        OdeIntegrationStatus::Success => "success",
        OdeIntegrationStatus::MaxStepsExceeded => "max_steps_exceeded",
        OdeIntegrationStatus::StepUnderflow => "step_underflow",
        OdeIntegrationStatus::InvalidInput => "invalid_input",
        OdeIntegrationStatus::NanEncountered => "nan_encountered",
        OdeIntegrationStatus::EventTriggered => "event_triggered",
        OdeIntegrationStatus::NonFiniteState => "non_finite_state",
        OdeIntegrationStatus::MaxRejectsExceeded => "max_rejects_exceeded",
        OdeIntegrationStatus::EventInvalid => "event_invalid",
    }
}

pub const fn eclipse_error_name(error: EclipseError) -> &'static str {
    match error {
        EclipseError::Gravity(_) => "eclipse_gravity",
        // Distinct from `eclipse_geometry` on purpose: a name that says
        // "geometry" for a refused configuration is what made this class of
        // failure unreadable in the first place.
        EclipseError::Authority(_) => "eclipse_strict_hf_authority",
        EclipseError::Geometry => "eclipse_geometry",
        EclipseError::UninitializedSide => "eclipse_uninitialized_side",
        EclipseError::NonProgress => "eclipse_non_progress",
        EclipseError::Chatter => "eclipse_chatter",
        EclipseError::Bracket => "eclipse_bracket",
        EclipseError::EventOverlap => "eclipse_event_overlap",
        EclipseError::SplitLimit => "eclipse_split_limit",
        EclipseError::Envelope => "eclipse_envelope",
    }
}

pub struct BinaryEclipseContext {
    pub eps: f64,
    pub jd0: f64,
    pub config: Arc<ForceConfig>,
    pub packed: Arc<PackedGravityCoeffs>,
    pub stepper: StepperMethod,
}

/// Immutable scalar-run request.  Keeping propagation controls together makes
/// the two-RHS coordinator boundary explicit without an argument-list shim.
pub struct BinaryEclipseRun<'a> {
    pub init_equinoc_state: [f64; 6],
    pub t_eval: &'a [f64],
    pub t0_s: f64,
    pub tf_s: f64,
    pub enable_events: bool,
    pub eps: f64,
    pub stepper: StepperMethod,
}

struct CoordinatorSettings<'a> {
    init_equinoc_state: [f64; 6],
    t0_s: f64,
    tf_s: f64,
    eps: f64,
    stepper: StepperMethod,
    forward: bool,
    enable_events: bool,
    config: &'a ForceConfig,
    start_time: std::time::Instant,
}

struct CoordinatorProgress<'a> {
    output: &'a mut IntegrationResult,
    total_steps: &'a mut usize,
    total_evals: &'a mut usize,
    curr_equinoc: &'a mut [f64; 6],
    curr_t: &'a mut f64,
    side: &'a mut EclipseSide,
    eval_index: &'a mut usize,
    split_count: &'a mut usize,
    last_root: &'a mut Option<CertifiedRoot>,
    collapsed_pairs: &'a mut usize,
    #[cfg(feature = "scalar-leg-observer")]
    observation: Option<&'a mut FinalObservation>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SegmentOutcome {
    Continue,
    Finished,
}

fn zero_span_output(
    requested_times: &[f64],
    start_s: f64,
    end_s: f64,
) -> Result<IntegrationResult, EclipseError> {
    if !same_time_value(start_s, end_s)
        || !start_s.is_finite()
        || requested_times
            .iter()
            .any(|time| !time.is_finite() || !same_time_value(*time, start_s))
    {
        return Err(EclipseError::NonProgress);
    }
    Ok(IntegrationResult {
        times: requested_times.to_vec(),
        states: vec![[0.0; 6]; requested_times.len()],
        ..IntegrationResult::default()
    })
}

fn solver_sample_times(
    public_times: &[f64],
    curr_t: f64,
    segment_tf: f64,
) -> Result<Vec<f64>, EclipseError> {
    let needs_start = public_times
        .first()
        .is_none_or(|time| !same_time_value(*time, curr_t));
    let needs_endpoint = public_times
        .last()
        .is_none_or(|time| !same_time_value(*time, segment_tf));
    let solver_capacity = public_times
        .len()
        .checked_add(usize::from(needs_start))
        .and_then(|length| length.checked_add(usize::from(needs_endpoint)))
        .ok_or(EclipseError::SplitLimit)?;
    let mut solver_times = Vec::new();
    solver_times
        .try_reserve_exact(solver_capacity)
        .map_err(|_| EclipseError::SplitLimit)?;
    if needs_start {
        solver_times.push(curr_t);
    }
    solver_times.extend_from_slice(public_times);
    if needs_endpoint {
        solver_times.push(segment_tf);
    }
    Ok(solver_times)
}

pub fn integrate_binary_eclipse_scalar(
    init_equinoc_state: [f64; 6],
    t_eval: &[f64],
    start_s: f64,
    end_s: f64,
    enable_events: bool,
    context: BinaryEclipseContext,
) -> Result<IntegrationResult, EclipseError> {
    #[cfg(feature = "scalar-leg-observer")]
    {
        integrate_binary_eclipse_scalar_inner(
            init_equinoc_state,
            t_eval,
            start_s,
            end_s,
            enable_events,
            context,
            None,
        )
    }
    #[cfg(not(feature = "scalar-leg-observer"))]
    integrate_binary_eclipse_scalar_inner(
        init_equinoc_state,
        t_eval,
        start_s,
        end_s,
        enable_events,
        context,
    )
}

/// Run fresh binary-eclipse scalar propagation with invocation-local metrics.
///
/// This is feature-only in this private module: canonical callers keep their
/// existing result-only path and never construct an observer.
#[cfg(feature = "scalar-leg-observer")]
pub fn integrate_binary_eclipse_scalar_observed(
    init_equinoc_state: [f64; 6],
    t_eval: &[f64],
    start_s: f64,
    end_s: f64,
    enable_events: bool,
    context: BinaryEclipseContext,
    observation: &mut FinalObservation,
) -> Result<IntegrationResult, EclipseError> {
    let result = integrate_binary_eclipse_scalar_inner(
        init_equinoc_state,
        t_eval,
        start_s,
        end_s,
        enable_events,
        context,
        Some(&mut *observation),
    );
    if result.is_err() {
        observation.mark_incomplete(ObservedFinalMetricError::EclipseMetricsUnavailable);
    }
    result
}

/// Keep the REASON an RHS refused to build.
///
/// Both call sites used `map_err(|_| EclipseError::Geometry)`, which threw away
/// the distinction between "the strict-HF enclosure refused this configuration"
/// and "the geometry is bad". The former is an authority decision about inputs
/// and the latter a numerical one, and collapsing them meant an
/// `IdentityMismatch(Science)` reached the mass solver as
/// `MissAtZeroHfIntegrateFailure` -- a propagation failure that never happened.
fn rhs_construction_error(error: anyhow::Error) -> EclipseError {
    // Consuming `downcast`, not `downcast_ref`: the error is owned here and
    // nothing downstream reads it, so borrowing it only to copy the payload out
    // left the original to be dropped unread.
    error
        .downcast::<crate::strict_hf_enclosure::StrictHfAuthorityError>()
        .map_or(EclipseError::Geometry, EclipseError::Authority)
}

fn integrate_binary_eclipse_scalar_inner(
    init_equinoc_state: [f64; 6],
    t_eval: &[f64],
    start_s: f64,
    end_s: f64,
    enable_events: bool,
    context: BinaryEclipseContext,
    #[cfg(feature = "scalar-leg-observer")] observation: Option<&mut FinalObservation>,
) -> Result<IntegrationResult, EclipseError> {
    let BinaryEclipseContext {
        eps,
        jd0,
        config,
        packed,
        stepper,
    } = context;
    let mut lane_rhs = LightyearRHS::try_new(
        init_equinoc_state,
        start_s,
        jd0,
        Arc::clone(&config),
        Arc::clone(&packed),
    )
    .map_err(rhs_construction_error)?;
    let mut root_rhs = LightyearRHS::try_new(init_equinoc_state, start_s, jd0, config, packed)
        .map_err(rhs_construction_error)?;
    integrate_binary_eclipse_scalar_with_rhs_inner(
        &BinaryEclipseRun {
            init_equinoc_state,
            t_eval,
            t0_s: start_s,
            tf_s: end_s,
            enable_events,
            eps,
            stepper,
        },
        &mut lane_rhs,
        &mut root_rhs,
        #[cfg(feature = "scalar-leg-observer")]
        observation,
    )
}

pub fn integrate_binary_eclipse_scalar_with_rhs(
    run: &BinaryEclipseRun<'_>,
    lane_rhs: &mut LightyearRHS,
    root_rhs: &mut LightyearRHS,
) -> Result<IntegrationResult, EclipseError> {
    #[cfg(feature = "scalar-leg-observer")]
    {
        integrate_binary_eclipse_scalar_with_rhs_inner(run, lane_rhs, root_rhs, None)
    }
    #[cfg(not(feature = "scalar-leg-observer"))]
    integrate_binary_eclipse_scalar_with_rhs_inner(run, lane_rhs, root_rhs)
}

/// Run the existing reusable-RHS binary-eclipse core with local metrics.
///
/// This crate-private feature seam does not expose a second propagator or an
/// observer callback to canonical callers. It only lets the bounded
/// qualification diagnostic observe the same sequential reusable-RHS path.
#[cfg(feature = "scalar-leg-observer")]
pub fn integrate_binary_eclipse_scalar_with_rhs_observed(
    run: &BinaryEclipseRun<'_>,
    lane_rhs: &mut LightyearRHS,
    root_rhs: &mut LightyearRHS,
    observation: &mut FinalObservation,
) -> Result<IntegrationResult, EclipseError> {
    let result = integrate_binary_eclipse_scalar_with_rhs_inner(
        run,
        lane_rhs,
        root_rhs,
        Some(&mut *observation),
    );
    if result.is_err() {
        observation.mark_incomplete(ObservedFinalMetricError::EclipseMetricsUnavailable);
    }
    result
}

fn integrate_binary_eclipse_scalar_with_rhs_inner(
    run: &BinaryEclipseRun<'_>,
    lane_rhs: &mut LightyearRHS,
    root_rhs: &mut LightyearRHS,
    #[cfg(feature = "scalar-leg-observer")] mut observation: Option<&mut FinalObservation>,
) -> Result<IntegrationResult, EclipseError> {
    let BinaryEclipseRun {
        init_equinoc_state,
        t_eval,
        t0_s,
        tf_s,
        enable_events,
        eps,
        stepper,
    } = *run;
    if same_time_value(tf_s, t0_s) {
        return zero_span_output(t_eval, t0_s, tf_s);
    }
    let config = Arc::clone(&lane_rhs.config);
    let start_time = std::time::Instant::now();
    let forward = tf_s >= t0_s;
    let mut output = IntegrationResult::default();
    let mut total_steps = 0;
    let mut total_evals = 0;
    let mut curr_equinoc = init_equinoc_state;
    let mut curr_t = t0_s;
    let mut eval_index = 0;
    let mut split_count = 0;
    let mut last_root: Option<CertifiedRoot> = None;
    let mut collapsed_pairs = 0;

    #[cfg(feature = "scalar-leg-observer")]
    if let Some(observation) = observation.as_deref_mut() {
        observation.record_eclipse_direction(forward);
    }

    while let Some(time) = t_eval.get(eval_index) {
        let precedes_start = if forward { *time < t0_s } else { *time > t0_s };
        if !precedes_start {
            break;
        }
        checked_count_add(&mut eval_index, 1)?;
    }

    lane_rhs.adapt_cache_policy_for_eps(eps);
    lane_rhs.reset_for_propagation(curr_equinoc, curr_t);
    // The carried controller `h` (step-size carry across Rebased boundaries)
    // is scoped to ONE propagation; see `hcarry_reset`'s doc for why the reset
    // is load-bearing for determinism under rayon.
    crate::integrator::hcarry_reset();
    root_rhs.adapt_cache_policy_for_eps(eps);
    lane_rhs.validate_eclipse_envelope_at_delta(&[0.0; 6], curr_t)?;
    let (initial_position, initial_sun) = lane_rhs.eclipse_geometry_at_delta(&[0.0; 6], curr_t)?;
    let mut side = crate::eclipse::classify_binary_cylinder(
        initial_position,
        initial_sun,
        config.earth_radius,
    )?;
    let settings = CoordinatorSettings {
        init_equinoc_state,
        t0_s,
        tf_s,
        eps,
        stepper,
        forward,
        enable_events,
        config: &config,
        start_time,
    };

    for _ in 0..MAX_STEPS {
        if same_time_value(tf_s, curr_t) {
            output.metrics = coordinator_metrics(
                total_steps,
                total_evals,
                elapsed_micros(start_time),
                collapsed_pairs,
            );
            return Ok(output);
        }
        let remaining = tf_s - curr_t;
        let segment_tf = segmented_endpoint(curr_t, remaining, tf_s);

        let public_start = eval_index;
        while let Some(time) = t_eval.get(eval_index) {
            let falls_in_segment = if forward {
                *time <= segment_tf
            } else {
                *time >= segment_tf
            };
            if !falls_in_segment {
                break;
            }
            checked_count_add(&mut eval_index, 1)?;
        }
        let public_times = t_eval
            .get(public_start..eval_index)
            .ok_or(EclipseError::Bracket)?;
        let solver_times = solver_sample_times(public_times, curr_t, segment_tf)?;

        lane_rhs.reset_for_propagation(curr_equinoc, curr_t);
        lane_rhs.set_eclipse_side(side);
        let system = LightyearSystem { rhs: lane_rhs };
        let mut handler = CoordinatedEventHandler::new(
            lane_rhs,
            side,
            curr_t,
            [0.0; 6],
            1e-6,
            eps,
            50,
            enable_events,
            last_root,
        );
        let trial = resolve_solver_boundary(lane_rhs, {
            #[cfg(feature = "prop-census")]
            let _site =
                crate::probe::eclipse_transaction_scope(crate::probe::EclipseTransactionSite::Main);
            integrate_segment_with_method(
                &system,
                &[0.0; 6],
                &solver_times,
                SegmentControls {
                    t0_s: curr_t,
                    t_final_s: segment_tf,
                    eps,
                    dt_max: config.dt_max,
                    force_eval: false,
                    fast_single: false,
                    max_steps: MAX_STEPS,
                    max_rejects: 50,
                    stepper,
                    boundary: SegmentBoundary::Rebased,
                },
                Some(&mut handler),
            )
        })?;
        #[cfg(feature = "scalar-leg-observer")]
        if let Some(observation) = observation.as_deref_mut() {
            observation.record_solver(&trial.stats, trial.status);
            observation.record_encke_segment();
        }
        checked_count_add(&mut total_steps, trial.stats.steps)?;
        checked_count_add(&mut total_evals, trial.stats.evals)?;
        let eclipse = handler.take_eclipse_outcome()?;
        let collapsed_delta = handler.take_collapsed_pairs();
        checked_count_add(&mut collapsed_pairs, collapsed_delta)?;
        #[cfg(feature = "scalar-leg-observer")]
        if let Some(observation) = observation.as_deref_mut() {
            observation.record_eclipse_collapsed_pairs(collapsed_delta);
        }
        let event_invalid = handler.take_event_invalid();
        let detection = handler.take_detection();

        if let Some(bracket) = eclipse {
            let outcome = resolve_eclipse_split(
                CoordinatorProgress {
                    output: &mut output,
                    total_steps: &mut total_steps,
                    total_evals: &mut total_evals,
                    curr_equinoc: &mut curr_equinoc,
                    curr_t: &mut curr_t,
                    side: &mut side,
                    eval_index: &mut eval_index,
                    split_count: &mut split_count,
                    last_root: &mut last_root,
                    collapsed_pairs: &mut collapsed_pairs,
                    #[cfg(feature = "scalar-leg-observer")]
                    observation: observation.as_deref_mut(),
                },
                &settings,
                lane_rhs,
                root_rhs,
                &trial,
                public_times,
                public_start,
                bracket,
            )?;
            if outcome == SegmentOutcome::Finished {
                return Ok(output);
            }
            continue;
        }

        let outcome = {
            let mut progress = CoordinatorProgress {
                output: &mut output,
                total_steps: &mut total_steps,
                total_evals: &mut total_evals,
                curr_equinoc: &mut curr_equinoc,
                curr_t: &mut curr_t,
                side: &mut side,
                eval_index: &mut eval_index,
                split_count: &mut split_count,
                last_root: &mut last_root,
                collapsed_pairs: &mut collapsed_pairs,
                #[cfg(feature = "scalar-leg-observer")]
                observation: observation.as_deref_mut(),
            };
            complete_non_eclipse_segment(
                &mut progress,
                &settings,
                trial,
                public_times,
                public_start,
                segment_tf,
                event_invalid,
                detection,
            )?
        };
        if outcome == SegmentOutcome::Finished {
            return Ok(output);
        }
    }
    Err(EclipseError::SplitLimit)
}

struct RefinedEclipseTrial {
    result: crate::odesolve::IntegrationResultSampled,
    scan: EclipseScanResult,
    crossing_step: Option<AcceptedDeltaStep>,
    event_invalid: bool,
    detection: Option<crate::types::EventDetection>,
}

fn resolve_eclipse_split(
    mut progress: CoordinatorProgress<'_>,
    settings: &CoordinatorSettings<'_>,
    lane_rhs: &mut LightyearRHS,
    root_rhs: &mut LightyearRHS,
    trial: &crate::odesolve::IntegrationResultSampled,
    public_times: &[f64],
    public_start: usize,
    bracket: EclipseBracket,
) -> Result<SegmentOutcome, EclipseError> {
    let mut replay_equinoc = [0.0; 6];
    eci2equinoc_impl_f64(&bracket.accepted_eci_old, 6, 0.0, 0.0, &mut replay_equinoc);
    let source_pre_count = public_times.partition_point(|time| {
        if settings.forward {
            *time <= bracket.accepted_t_old
        } else {
            *time >= bracket.accepted_t_old
        }
    });
    let source_step_count = public_times.partition_point(|time| {
        if settings.forward {
            *time <= bracket.accepted_t_new
        } else {
            *time >= bracket.accepted_t_new
        }
    });
    let refinement_public_times = public_times
        .get(source_pre_count..source_step_count)
        .ok_or(EclipseError::Bracket)?;
    let refinement_solver_times = solver_sample_times(
        refinement_public_times,
        bracket.accepted_t_old,
        bracket.accepted_t_new,
    )?;
    let refined_trial = refine_eclipse_bracket(
        root_rhs,
        settings,
        &bracket,
        replay_equinoc,
        &refinement_solver_times,
        progress.total_steps,
        progress.total_evals,
        *progress.last_root,
        #[cfg(feature = "scalar-leg-observer")]
        &mut progress.observation,
    )?;
    checked_count_add(progress.collapsed_pairs, refined_trial.scan.collapsed_pairs)?;
    #[cfg(feature = "scalar-leg-observer")]
    if let Some(observation) = progress.observation.as_deref_mut() {
        observation.record_eclipse_collapsed_pairs(refined_trial.scan.collapsed_pairs);
    }
    let source_pre_times = public_times
        .get(..source_pre_count)
        .ok_or(EclipseError::Bracket)?;
    for &public_time in source_pre_times {
        let returned = trial
            .times
            .iter()
            .position(|time| time.to_bits() == public_time.to_bits())
            .ok_or(EclipseError::Bracket)?;
        let offset = returned
            .checked_mul(trial.n_state)
            .ok_or(EclipseError::Bracket)?;
        let state_end = offset.checked_add(6).ok_or(EclipseError::Bracket)?;
        let state = trial
            .states
            .get(offset..state_end)
            .map(slice_to_state)
            .ok_or(EclipseError::Bracket)?;
        progress.output.times.push(public_time);
        progress
            .output
            .states
            .push(correct_delta_to_original_baseline(
                &state,
                public_time,
                &*progress.curr_equinoc,
                *progress.curr_t,
                &settings.init_equinoc_state,
                settings.t0_s,
            ));
    }
    let Some(refined) = refined_trial.scan.crossing else {
        *progress.curr_equinoc = replay_equinoc;
        *progress.curr_t = bracket.accepted_t_old;
        *progress.eval_index = public_start
            .checked_add(source_step_count)
            .ok_or(EclipseError::SplitLimit)?;
        #[cfg(feature = "scalar-leg-observer")]
        if let Some(observation) = progress.observation.as_deref_mut() {
            observation.record_encke_rebase();
        }
        return complete_non_eclipse_segment(
            &mut progress,
            settings,
            refined_trial.result,
            refinement_public_times,
            public_start
                .checked_add(source_pre_count)
                .ok_or(EclipseError::SplitLimit)?,
            bracket.accepted_t_new,
            refined_trial.event_invalid,
            refined_trial.detection,
        );
    };
    let refined_pre_count = refinement_public_times.partition_point(|time| {
        if settings.forward {
            *time <= refined.accepted_t_old
        } else {
            *time >= refined.accepted_t_old
        }
    });
    let refined_pre_times = refinement_public_times
        .get(..refined_pre_count)
        .ok_or(EclipseError::Bracket)?;
    let refined_states = flatten_states(refined_trial.result.states, refined_trial.result.n_state);
    if refined_states.len() != refined_trial.result.times.len() || refined_states.is_empty() {
        return Err(EclipseError::Bracket);
    }
    for &public_time in refined_pre_times {
        let returned = refined_trial
            .result
            .times
            .iter()
            .position(|time| same_time_value(*time, public_time))
            .ok_or(EclipseError::Bracket)?;
        let state = *refined_states.get(returned).ok_or(EclipseError::Bracket)?;
        progress.output.times.push(public_time);
        progress
            .output
            .states
            .push(correct_delta_to_original_baseline(
                &state,
                public_time,
                &replay_equinoc,
                bracket.accepted_t_old,
                &settings.init_equinoc_state,
                settings.t0_s,
            ));
    }
    let pre_step_public_count = source_pre_count
        .checked_add(refined_pre_count)
        .ok_or(EclipseError::SplitLimit)?;

    let mut root_equinoc = [0.0; 6];
    eci2equinoc_impl_f64(&refined.accepted_eci_old, 6, 0.0, 0.0, &mut root_equinoc);
    *progress.curr_equinoc = root_equinoc;
    *progress.curr_t = refined.accepted_t_old;
    *progress.eval_index = public_start
        .checked_add(pre_step_public_count)
        .ok_or(EclipseError::SplitLimit)?;
    #[cfg(feature = "scalar-leg-observer")]
    if let Some(observation) = progress.observation.as_deref_mut() {
        observation.record_encke_rebase();
    }

    commit_eclipse_root_transaction(
        &mut progress,
        settings,
        lane_rhs,
        root_rhs,
        &refined,
        public_times,
        public_start,
        pre_step_public_count,
    )
}

fn refine_eclipse_bracket(
    root_rhs: &mut LightyearRHS,
    settings: &CoordinatorSettings<'_>,
    source: &EclipseBracket,
    replay_equinoc: [f64; 6],
    solver_times: &[f64],
    total_steps: &mut usize,
    total_evals: &mut usize,
    certified_root: Option<CertifiedRoot>,
    #[cfg(feature = "scalar-leg-observer")] observation: &mut Option<&mut FinalObservation>,
) -> Result<RefinedEclipseTrial, EclipseError> {
    let replay_start = source.accepted_t_old;
    let replay_end = source.accepted_t_new;
    if !replay_start.is_finite()
        || !replay_end.is_finite()
        || same_time_value(replay_start, replay_end)
        || source.old_side == source.new_side
    {
        return Err(EclipseError::Bracket);
    }
    root_rhs.reset_for_propagation(replay_equinoc, replay_start);
    root_rhs.set_eclipse_side(source.old_side);
    root_rhs.validate_eclipse_envelope_at_delta(&[0.0; 6], replay_start)?;
    let system = LightyearSystem { rhs: root_rhs };
    let mut handler = CoordinatedEventHandler::new(
        root_rhs,
        source.old_side,
        replay_start,
        [0.0; 6],
        1e-6,
        settings.eps,
        50,
        settings.enable_events,
        certified_root,
    );
    let result = resolve_solver_boundary(root_rhs, {
        #[cfg(feature = "prop-census")]
        let _site =
            crate::probe::eclipse_transaction_scope(crate::probe::EclipseTransactionSite::Refine);
        integrate_segment_with_method(
            &system,
            &[0.0; 6],
            solver_times,
            SegmentControls {
                t0_s: replay_start,
                t_final_s: replay_end,
                eps: settings.eps,
                dt_max: settings.config.dt_max.min(MAX_ROOT_REFINEMENT_STEP_S),
                force_eval: false,
                fast_single: false,
                max_steps: MAX_STEPS,
                max_rejects: 50,
                stepper: settings.stepper,
                boundary: SegmentBoundary::Rebased,
            },
            Some(&mut handler),
        )
    })?;
    #[cfg(feature = "scalar-leg-observer")]
    if let Some(observation) = observation.as_deref_mut() {
        observation.record_solver(&result.stats, result.status);
    }
    checked_count_add(total_steps, result.stats.steps)?;
    checked_count_add(total_evals, result.stats.evals)?;
    let outcome = handler.take_eclipse_outcome()?;
    let crossing_step = handler.take_eclipse_step();
    let collapsed_pairs = handler.take_collapsed_pairs();
    let event_invalid = handler.take_event_invalid();
    let detection = handler.take_detection();
    let Some(bracket) = outcome else {
        let valid_non_eclipse_status = matches!(result.status, OdeIntegrationStatus::Success)
            || (matches!(result.status, OdeIntegrationStatus::EventTriggered)
                && (event_invalid || detection.is_some()));
        if valid_non_eclipse_status {
            return Ok(RefinedEclipseTrial {
                result,
                scan: EclipseScanResult {
                    crossing: None,
                    collapsed_pairs,
                },
                crossing_step,
                event_invalid,
                detection,
            });
        }
        return Err(EclipseError::Bracket);
    };
    if !matches!(result.status, OdeIntegrationStatus::EventTriggered)
        || bracket.old_side != source.old_side
        || bracket.new_side == source.old_side
        || !bracket.geometry_motion_bound_km.is_finite()
        || !(0.0..=MAX_BOUNDARY_SEPARATION_KM).contains(&bracket.geometry_motion_bound_km)
    {
        return Err(EclipseError::Bracket);
    }
    let ordered_inside_source = if settings.forward {
        source.accepted_t_old <= bracket.accepted_t_old
            && bracket.accepted_t_old < bracket.accepted_t_new
            && bracket.accepted_t_new <= source.accepted_t_new
    } else {
        source.accepted_t_old >= bracket.accepted_t_old
            && bracket.accepted_t_old > bracket.accepted_t_new
            && bracket.accepted_t_new >= source.accepted_t_new
    };
    if !ordered_inside_source {
        return Err(EclipseError::NonProgress);
    }
    Ok(RefinedEclipseTrial {
        result,
        scan: EclipseScanResult {
            crossing: Some(bracket),
            collapsed_pairs,
        },
        crossing_step,
        event_invalid,
        detection,
    })
}

fn run_root_transaction_leg(
    #[cfg(feature = "prop-census")] census_site: crate::probe::EclipseTransactionSite,
    root_rhs: &LightyearRHS,
    progress: &mut CoordinatorProgress<'_>,
    settings: &CoordinatorSettings<'_>,
    side: EclipseSide,
    initial_t: f64,
    final_t: f64,
    initial_delta: [f64; 6],
    solver_times: &[f64],
    boundary: SegmentBoundary,
) -> Result<RefinedEclipseTrial, EclipseError> {
    if !initial_t.is_finite() || !final_t.is_finite() || same_time_value(initial_t, final_t) {
        return Err(EclipseError::NonProgress);
    }
    #[cfg(feature = "prop-census")]
    let _site = crate::probe::eclipse_transaction_scope(census_site);
    let system = LightyearSystem { rhs: root_rhs };
    let mut handler = CoordinatedEventHandler::new(
        root_rhs,
        side,
        initial_t,
        initial_delta,
        1e-6,
        settings.eps,
        50,
        settings.enable_events,
        *progress.last_root,
    );
    let result = resolve_solver_boundary(
        root_rhs,
        integrate_segment_with_method(
            &system,
            &initial_delta,
            solver_times,
            SegmentControls {
                t0_s: initial_t,
                t_final_s: final_t,
                eps: settings.eps,
                dt_max: settings
                    .config
                    .dt_max
                    .min(MAX_ROOT_REFINEMENT_STEP_S)
                    .min((final_t - initial_t).abs()),
                force_eval: false,
                fast_single: false,
                max_steps: MAX_STEPS,
                max_rejects: 50,
                stepper: settings.stepper,
                boundary,
            },
            Some(&mut handler),
        ),
    )?;
    #[cfg(feature = "scalar-leg-observer")]
    if let Some(observation) = progress.observation.as_deref_mut() {
        observation.record_solver(&result.stats, result.status);
    }
    checked_count_add(progress.total_steps, result.stats.steps)?;
    checked_count_add(progress.total_evals, result.stats.evals)?;
    let crossing = handler.take_eclipse_outcome()?;
    let crossing_step = handler.take_eclipse_step();
    let collapsed_pairs = handler.take_collapsed_pairs();
    checked_count_add(progress.collapsed_pairs, collapsed_pairs)?;
    #[cfg(feature = "scalar-leg-observer")]
    if let Some(observation) = progress.observation.as_deref_mut() {
        observation.record_eclipse_collapsed_pairs(collapsed_pairs);
    }
    let event_invalid = handler.take_event_invalid();
    let detection = handler.take_detection();
    Ok(RefinedEclipseTrial {
        result,
        scan: EclipseScanResult {
            crossing,
            collapsed_pairs,
        },
        crossing_step,
        event_invalid,
        detection,
    })
}

fn sampled_state_at_time(
    result: &crate::odesolve::IntegrationResultSampled,
    crossing_step: Option<&AcceptedDeltaStep>,
    time: f64,
) -> Result<[f64; 6], EclipseError> {
    let returned = result
        .times
        .iter()
        .position(|returned_time| same_time_value(*returned_time, time));
    let Some(returned) = returned else {
        return crossing_step
            .copied()
            .ok_or(EclipseError::Bracket)?
            .state_at(time);
    };
    let offset = returned
        .checked_mul(result.n_state)
        .ok_or(EclipseError::Bracket)?;
    let state_end = offset.checked_add(6).ok_or(EclipseError::Bracket)?;
    result
        .states
        .get(offset..state_end)
        .map(slice_to_state)
        .ok_or(EclipseError::Bracket)
}

fn append_transaction_samples(
    output: &mut IntegrationResult,
    result: &crate::odesolve::IntegrationResultSampled,
    crossing_step: Option<&AcceptedDeltaStep>,
    public_times: &[f64],
    base_equinoc: &[f64; 6],
    base_t: f64,
    settings: &CoordinatorSettings<'_>,
) -> Result<(), EclipseError> {
    for &time in public_times {
        let state = sampled_state_at_time(result, crossing_step, time)?;
        output.times.push(time);
        output.states.push(correct_delta_to_original_baseline(
            &state,
            time,
            base_equinoc,
            base_t,
            &settings.init_equinoc_state,
            settings.t0_s,
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ValidatedEclipseRoot {
    bracket: EclipseBracket,
    time: f64,
    delta: [f64; 6],
}

fn validate_eclipse_root_commit(
    root_rhs: &LightyearRHS,
    expected: &EclipseBracket,
    proof_old_t: f64,
    result: &crate::odesolve::IntegrationResultSampled,
    actual: EclipseBracket,
    last_root: Option<CertifiedRoot>,
    forward: bool,
) -> Result<ValidatedEclipseRoot, EclipseError> {
    let event = result.event.as_ref().ok_or(EclipseError::Bracket)?;
    let endpoint_t = *result.times.last().ok_or(EclipseError::Bracket)?;
    if !matches!(result.status, OdeIntegrationStatus::EventTriggered)
        || actual.old_side != expected.old_side
        || actual.new_side != expected.new_side
        || !same_time_value(event.t, actual.accepted_t_new)
        || !same_time_value(endpoint_t, event.t)
    {
        return Err(EclipseError::Bracket);
    }
    let delta = sampled_state_at_time(result, None, event.t)?;
    root_rhs.validate_eclipse_envelope_at_delta(&delta, event.t)?;
    let (position, sun) = root_rhs.eclipse_geometry_at_delta(&delta, event.t)?;
    let committed_side =
        binary_cylinder_geometry(position, sun, root_rhs.config.earth_radius)?.side;
    if committed_side != actual.new_side
        || replay_root_uncertainty_km(root_rhs, proof_old_t, event.t)?
            > MAX_ROOT_TOTAL_UNCERTAINTY_KM
    {
        return Err(EclipseError::Bracket);
    }
    if !event.t.is_finite()
        || same_time_value(event.t, actual.accepted_t_old)
        || last_root.is_some_and(|last| {
            if forward {
                event.t <= last.t
            } else {
                event.t >= last.t
            }
        })
    {
        return Err(EclipseError::Chatter);
    }
    Ok(ValidatedEclipseRoot {
        bracket: actual,
        time: event.t,
        delta,
    })
}

fn rebase_after_eclipse_root(
    progress: &mut CoordinatorProgress<'_>,
    lane_rhs: &mut LightyearRHS,
    root_t: f64,
    root_delta: [f64; 6],
    new_side: EclipseSide,
) -> Result<(), EclipseError> {
    #[cfg(test)]
    if eclipse_test_capture_enabled() {
        TEST_ECLIPSE_ROOTS
            .lock()
            .map_err(|_| EclipseError::Bracket)?
            .push(root_t);
        TEST_ECLIPSE_SPLITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    let base_t = *progress.curr_t;
    let base_equinoc = *progress.curr_equinoc;
    let mut root_baseline = [0.0; 6];
    equinoc2eci_impl(&base_equinoc, 6, root_t - base_t, 0.0, &mut root_baseline);
    let mut root_eci = [0.0; 6];
    for ((eci_component, baseline_component), delta_component) in root_eci
        .iter_mut()
        .zip(root_baseline.iter())
        .zip(root_delta.iter())
    {
        *eci_component = *baseline_component + *delta_component;
    }
    eci2equinoc_impl_f64(&root_eci, 6, 0.0, 0.0, progress.curr_equinoc);
    *progress.curr_t = root_t;
    *progress.side = new_side;
    lane_rhs.reset_for_propagation(*progress.curr_equinoc, root_t);
    let (rebased_position, _rebased_sun) = lane_rhs.eclipse_geometry_at_delta(&[0.0; 6], root_t)?;
    // The committed side is authority at `root_t`; the root transaction proved
    // it on its own baseline, and the continuation window already spends the
    // root's uncertainty budget. What remains to bound here is only the
    // equinoctial round trip that carries the root onto this lane's baseline.
    let [root_x, root_y, root_z, _, _, _] = root_eci;
    let rebase_error_km =
        outward_nonnegative(distance3([root_x, root_y, root_z], rebased_position))?;
    if rebase_error_km > MAX_BOUNDARY_SEPARATION_KM {
        return Err(EclipseError::Bracket);
    }
    // Publish the measured transport error with the root, not just the instant.
    // The continuation's scan-start gate needs the error this rebase ACTUALLY
    // cost; charging it the `MAX_BOUNDARY_SEPARATION_KM` ceiling instead made
    // shallow crossings unpropagable. See `post_root_decisive_margin_km`.
    *progress.last_root = Some(CertifiedRoot {
        t: root_t,
        transport_error_km: rebase_error_km,
    });
    Ok(())
}

/// Where the old-side proof leg must reach, and how many of this window's
/// public times it carries.
///
/// A restart can leave the segment origin sitting bit-exactly on the crossing
/// it is about to commit. There is then no old-side interval left to prove, and
/// the returned proof point is `base_t` itself with a zero sample count: the
/// leg would otherwise be asked to integrate from `base_t` to `base_t`,
/// degenerate at both ends, since the bound search rejects equal arguments and
/// the leg rejects a zero span. That origin needs no proof, because
/// `commit_eclipse_root_transaction` has already established that
/// `*progress.side` equals `bracket.old_side`, and re-deriving geometry at the
/// origin would read exactly the boundary roundoff that the post-root scan
/// certification removed.
fn root_old_side_proof_point(
    root_rhs: &LightyearRHS,
    settings: &CoordinatorSettings<'_>,
    root_old_t: f64,
    base_t: f64,
    root_public_candidates: &[f64],
) -> Result<(f64, usize), EclipseError> {
    if same_time_value(root_old_t, base_t) {
        return Ok((base_t, 0));
    }
    let proof_old_t = deepest_directed_time_within_root_bound(
        root_rhs,
        root_old_t,
        base_t,
        ROOT_OLD_SIDE_MARGIN_KM,
    )?;
    let proof_public_count = root_public_candidates.partition_point(|time| {
        if settings.forward {
            *time <= proof_old_t
        } else {
            *time >= proof_old_t
        }
    });
    Ok((proof_old_t, proof_public_count))
}

/// The state the window leg opens from, once the proof point's side is
/// confirmed to still be the old one.
///
/// `proof` is `None` when the proof point is the segment origin. The origin
/// carries the transaction's entry invariant, `*progress.side ==
/// bracket.old_side`, so it owes no geometric check; deriving one there would
/// only re-read the boundary roundoff.
fn proven_old_side_delta(
    root_rhs: &LightyearRHS,
    proof: Option<&RefinedEclipseTrial>,
    proof_old_t: f64,
    old_side: EclipseSide,
) -> Result<[f64; 6], EclipseError> {
    let Some(proof) = proof else {
        return Ok([0.0; 6]);
    };
    let proof_delta = sampled_state_at_time(&proof.result, None, proof_old_t)?;
    root_rhs.validate_eclipse_envelope_at_delta(&proof_delta, proof_old_t)?;
    let (proof_position, proof_sun) =
        root_rhs.eclipse_geometry_at_delta(&proof_delta, proof_old_t)?;
    if binary_cylinder_geometry(proof_position, proof_sun, root_rhs.config.earth_radius)?.side
        != old_side
    {
        return Err(EclipseError::Bracket);
    }
    Ok(proof_delta)
}

/// How the old-side proof leg of a root transaction resolved.
enum OldSideProofOutcome {
    /// The old side is proven up to `proof_old_t`; the window leg may run.
    /// `proof` is `None` when the segment origin was itself the crossing.
    Proven {
        proof: Option<RefinedEclipseTrial>,
        proof_old_t: f64,
        proof_public_count: usize,
    },
    /// The proof leg hit a terminal or invalid normal event first; the caller
    /// must complete a non-eclipse segment from this trial.
    NonEclipse {
        trial: RefinedEclipseTrial,
        proof_old_t: f64,
        proof_public_count: usize,
    },
}

// The proof anchor opens at the detector's root estimate. When the proof
// leg's own fine scan finds a crossing anyway — a grazing crossing whose
// true root sits earlier than the detector estimate by more than the
// transport margin — that crossing is a strictly earlier, better-resolved
// anchor, and the acquisition restarts from it (bounded by
// `MAX_OLD_SIDE_PROOF_RESTARTS`) instead of failing closed. The committed
// root always comes from the caller's window leg, never from any anchor, so
// a restart changes where the proof point sits and nothing about what gets
// committed.
fn acquire_old_side_proof(
    progress: &mut CoordinatorProgress<'_>,
    settings: &CoordinatorSettings<'_>,
    root_rhs: &mut LightyearRHS,
    bracket: &EclipseBracket,
    base_t: f64,
    base_equinoc: [f64; 6],
    root_public_candidates: &[f64],
) -> Result<OldSideProofOutcome, EclipseError> {
    let mut proof_anchor_t = bracket.t_old;
    let mut proof_restarts_left = MAX_OLD_SIDE_PROOF_RESTARTS;
    loop {
        let (proof_old_t, proof_public_count) = root_old_side_proof_point(
            root_rhs,
            settings,
            proof_anchor_t,
            base_t,
            root_public_candidates,
        )?;
        // Only the degenerate origin lands the proof point back on the origin;
        // the bound search never returns its own starting instant.
        let origin_is_crossing = same_time_value(proof_old_t, base_t);
        let proof_public = root_public_candidates
            .get(..proof_public_count)
            .ok_or(EclipseError::Bracket)?;
        root_rhs.reset_for_propagation(base_equinoc, base_t);
        root_rhs.set_eclipse_side(bracket.old_side);
        root_rhs.validate_eclipse_envelope_at_delta(&[0.0; 6], base_t)?;
        #[cfg(test)]
        record_root_transaction_reset();
        if origin_is_crossing {
            return Ok(OldSideProofOutcome::Proven {
                proof: None,
                proof_old_t,
                proof_public_count,
            });
        }
        let proof_solver_times = solver_sample_times(proof_public, base_t, proof_old_t)?;
        let proof = run_root_transaction_leg(
            #[cfg(feature = "prop-census")]
            crate::probe::EclipseTransactionSite::Proof,
            root_rhs,
            progress,
            settings,
            bracket.old_side,
            base_t,
            proof_old_t,
            [0.0; 6],
            &proof_solver_times,
            SegmentBoundary::Rebased,
        )?;
        if let Some(found) = proof.scan.crossing.as_ref() {
            let ratchets_earlier = if settings.forward {
                found.t_old < proof_anchor_t
            } else {
                found.t_old > proof_anchor_t
            };
            if proof_restarts_left == 0
                || !found.t_old.is_finite()
                || !ratchets_earlier
                || found.old_side != bracket.old_side
            {
                return Err(EclipseError::Bracket);
            }
            // The zero guard above makes this exact; saturating only to keep
            // the operation side-effect-free for the lint.
            proof_restarts_left = proof_restarts_left.saturating_sub(1);
            proof_anchor_t = found.t_old;
            continue;
        }
        if proof.event_invalid || proof.detection.is_some() {
            return Ok(OldSideProofOutcome::NonEclipse {
                trial: proof,
                proof_old_t,
                proof_public_count,
            });
        }
        if !matches!(proof.result.status, OdeIntegrationStatus::Success) {
            return Err(EclipseError::Bracket);
        }
        return Ok(OldSideProofOutcome::Proven {
            proof: Some(proof),
            proof_old_t,
            proof_public_count,
        });
    }
}

fn commit_eclipse_root_transaction(
    progress: &mut CoordinatorProgress<'_>,
    settings: &CoordinatorSettings<'_>,
    lane_rhs: &mut LightyearRHS,
    root_rhs: &mut LightyearRHS,
    bracket: &EclipseBracket,
    public_times: &[f64],
    public_start: usize,
    pre_step_public_count: usize,
) -> Result<SegmentOutcome, EclipseError> {
    if *progress.side != bracket.old_side
        || !same_time_value(*progress.curr_t, bracket.accepted_t_old)
    {
        return Err(EclipseError::Chatter);
    }
    let base_t = *progress.curr_t;
    let base_equinoc = *progress.curr_equinoc;
    let root_public_candidates = public_times
        .get(pre_step_public_count..)
        .ok_or(EclipseError::Bracket)?;
    let (proof, proof_old_t, proof_public_count) = match acquire_old_side_proof(
        progress,
        settings,
        root_rhs,
        bracket,
        base_t,
        base_equinoc,
        root_public_candidates,
    )? {
        OldSideProofOutcome::Proven {
            proof,
            proof_old_t,
            proof_public_count,
        } => (proof, proof_old_t, proof_public_count),
        OldSideProofOutcome::NonEclipse {
            trial,
            proof_old_t,
            proof_public_count,
        } => {
            let proof_public = root_public_candidates
                .get(..proof_public_count)
                .ok_or(EclipseError::Bracket)?;
            return complete_non_eclipse_segment(
                progress,
                settings,
                trial.result,
                proof_public,
                public_start
                    .checked_add(pre_step_public_count)
                    .ok_or(EclipseError::SplitLimit)?,
                proof_old_t,
                trial.event_invalid,
                trial.detection,
            );
        }
    };
    let proof_public = root_public_candidates
        .get(..proof_public_count)
        .ok_or(EclipseError::Bracket)?;
    let proof_delta =
        proven_old_side_delta(root_rhs, proof.as_ref(), proof_old_t, bracket.old_side)?;

    let window_limit = deepest_directed_time_within_root_bound(
        root_rhs,
        proof_old_t,
        bracket.accepted_t_new,
        MAX_ROOT_TOTAL_UNCERTAINTY_KM,
    )?;
    let window_stops_before_refined_root = (settings.forward && window_limit < bracket.t_new)
        || (!settings.forward && window_limit > bracket.t_new);
    if window_stops_before_refined_root {
        return Err(EclipseError::NonProgress);
    }
    let post_proof_candidates = root_public_candidates
        .get(proof_public_count..)
        .ok_or(EclipseError::Bracket)?;
    let window_public_count = post_proof_candidates.partition_point(|time| {
        if settings.forward {
            *time <= window_limit
        } else {
            *time >= window_limit
        }
    });
    let window_public = post_proof_candidates
        .get(..window_public_count)
        .ok_or(EclipseError::Bracket)?;
    let window_solver_times = solver_sample_times(window_public, proof_old_t, window_limit)?;
    #[cfg(test)]
    record_root_transaction_continuation();
    // The proof leg, where there was one, left `root_rhs` on exactly the
    // baseline this leg opens from and handed over its own final delta: the
    // trajectory is continuous here and the split exists only to place
    // `proof_old_t`. Where the origin was already the crossing there is no
    // proof leg, and the reset that opened the transaction is the boundary.
    let window_boundary = if proof.is_some() {
        SegmentBoundary::EventContinuation
    } else {
        SegmentBoundary::Rebased
    };
    let window = run_root_transaction_leg(
        #[cfg(feature = "prop-census")]
        crate::probe::EclipseTransactionSite::Window,
        root_rhs,
        progress,
        settings,
        bracket.old_side,
        proof_old_t,
        window_limit,
        proof_delta,
        &window_solver_times,
        window_boundary,
    )?;
    let Some(committed_bracket) = window.scan.crossing else {
        if let Some(proof) = proof.as_ref() {
            append_transaction_samples(
                progress.output,
                &proof.result,
                None,
                proof_public,
                &base_equinoc,
                base_t,
                settings,
            )?;
        }
        if window.event_invalid || window.detection.is_some() {
            return complete_non_eclipse_segment(
                progress,
                settings,
                window.result,
                window_public,
                public_start
                    .checked_add(pre_step_public_count)
                    .and_then(|index| index.checked_add(proof_public_count))
                    .ok_or(EclipseError::SplitLimit)?,
                window_limit,
                window.event_invalid,
                window.detection,
            );
        }
        return Err(EclipseError::Bracket);
    };
    let committed = validate_eclipse_root_commit(
        root_rhs,
        bracket,
        proof_old_t,
        &window.result,
        committed_bracket,
        *progress.last_root,
        settings.forward,
    )?;
    checked_count_add(progress.split_count, 1)?;
    if *progress.split_count > MAX_ECLIPSE_SPLITS {
        return Err(EclipseError::SplitLimit);
    }

    let committed_public_count = root_public_candidates.partition_point(|time| {
        if settings.forward {
            *time <= committed.time
        } else {
            *time >= committed.time
        }
    });
    let committed_window_public = root_public_candidates
        .get(proof_public_count..committed_public_count)
        .ok_or(EclipseError::Bracket)?;
    if let Some(proof) = proof.as_ref() {
        append_transaction_samples(
            progress.output,
            &proof.result,
            None,
            proof_public,
            &base_equinoc,
            base_t,
            settings,
        )?;
    }
    append_transaction_samples(
        progress.output,
        &window.result,
        window.crossing_step.as_ref(),
        committed_window_public,
        &base_equinoc,
        base_t,
        settings,
    )?;
    rebase_after_eclipse_root(
        progress,
        lane_rhs,
        committed.time,
        committed.delta,
        committed.bracket.new_side,
    )?;
    *progress.eval_index = public_start
        .checked_add(pre_step_public_count)
        .and_then(|index| index.checked_add(committed_public_count))
        .ok_or(EclipseError::SplitLimit)?;
    #[cfg(feature = "scalar-leg-observer")]
    if let Some(observation) = progress.observation.as_deref_mut() {
        observation.record_encke_rebase();
        observation.record_eclipse_crossing(
            settings.forward,
            committed.bracket.old_side,
            committed.bracket.new_side,
        );
    }
    Ok(SegmentOutcome::Continue)
}

fn complete_non_eclipse_segment(
    progress: &mut CoordinatorProgress<'_>,
    settings: &CoordinatorSettings<'_>,
    trial: crate::odesolve::IntegrationResultSampled,
    public_times: &[f64],
    public_start: usize,
    segment_tf: f64,
    event_invalid: bool,
    detection: Option<crate::types::EventDetection>,
) -> Result<SegmentOutcome, EclipseError> {
    if event_invalid {
        progress.output.terminal_event_fired = true;
        progress.output.terminal_event_name = Cow::Borrowed("event_invalid");
        progress.output.metrics = coordinator_metrics(
            *progress.total_steps,
            *progress.total_evals,
            elapsed_micros(settings.start_time),
            *progress.collapsed_pairs,
        );
        return Ok(SegmentOutcome::Finished);
    }
    if let Some(detection) = detection {
        if detection.event_type == crate::types::EventType::PerturbDeviation {
            let event_t = detection.refined_time;
            if !event_t.is_finite() || same_time_value(event_t, *progress.curr_t) {
                return Err(EclipseError::NonProgress);
            }
            let pre_event_public_count = public_times.partition_point(|time| {
                if settings.forward {
                    *time <= event_t
                } else {
                    *time >= event_t
                }
            });
            let trial_states = flatten_states(trial.states, trial.n_state);
            let pre_event_public_times = public_times
                .get(..pre_event_public_count)
                .ok_or(EclipseError::Bracket)?;
            // KNOWN HOLE, and the reason the sampled route rejects arcs the
            // checked route propagates. `pre_event_public_count` admits every
            // requested time up to and INCLUDING `event_t`, but the solver
            // stopped its trial at the event root, so the samples it emitted end
            // at the last accepted step before it. A requested time strictly
            // inside the terminating step was therefore never produced, the
            // `position` below misses, and the whole propagation fails as
            // `Bracket` -- a name that says the eclipse geometry failed when the
            // eclipse geometry is not involved.
            //
            // Removing this one `?` (returning any state instead) removes the
            // divergence at every output density tried, which is how the
            // mechanism was attributed. Closing it properly needs dense output
            // WITHIN the event step, which the trial does not carry.
            // Pinned, with the instrumented numbers, in
            // `tests/sampled_route_divergence_pin.rs`; do not "fix" this by
            // shrinking `pre_event_public_count`, which would silently drop a
            // requested output time instead of failing on it.
            for &public_time in pre_event_public_times {
                let state = if same_time_value(public_time, event_t) {
                    detection.state_at_event
                } else {
                    let returned = trial
                        .times
                        .iter()
                        .position(|time| same_time_value(*time, public_time))
                        .ok_or(EclipseError::Bracket)?;
                    *trial_states.get(returned).ok_or(EclipseError::Bracket)?
                };
                progress.output.times.push(public_time);
                progress
                    .output
                    .states
                    .push(correct_delta_to_original_baseline(
                        &state,
                        public_time,
                        &*progress.curr_equinoc,
                        *progress.curr_t,
                        &settings.init_equinoc_state,
                        settings.t0_s,
                    ));
            }
            progress.output.perturb_deviation_fired = true;
            progress.output.event_time = event_t;
            progress.output.state_at_event = detection.state_at_event;
            progress.output.event_interp_method = detection.interp_method;
            progress.output.event_interp_error = detection.interp_error;
            let mut event_baseline = [0.0; 6];
            equinoc2eci_impl(
                &*progress.curr_equinoc,
                6,
                event_t - *progress.curr_t,
                0.0,
                &mut event_baseline,
            );
            let mut event_eci = [0.0; 6];
            for ((eci_component, baseline_component), delta_component) in event_eci
                .iter_mut()
                .zip(event_baseline.iter())
                .zip(detection.state_at_event.iter())
            {
                *eci_component = *baseline_component + *delta_component;
            }
            eci2equinoc_impl_f64(&event_eci, 6, 0.0, 0.0, progress.curr_equinoc);
            let next_eval_index = public_start
                .checked_add(pre_event_public_count)
                .ok_or(EclipseError::SplitLimit)?;
            *progress.curr_t = event_t;
            *progress.eval_index = next_eval_index;
            *progress.last_root = None;
            #[cfg(feature = "scalar-leg-observer")]
            if let Some(observation) = progress.observation.as_deref_mut() {
                observation.record_encke_rebase();
            }
            return Ok(SegmentOutcome::Continue);
        }
        progress.output.terminal_event_fired = true;
        progress.output.terminal_event_name = Cow::Borrowed(detection.event_type.name());
        progress.output.event_time = detection.refined_time;
        progress.output.state_at_event = detection.state_at_event;
        progress.output.metrics = coordinator_metrics(
            *progress.total_steps,
            *progress.total_evals,
            elapsed_micros(settings.start_time),
            *progress.collapsed_pairs,
        );
        return Ok(SegmentOutcome::Finished);
    }
    if !matches!(trial.status, OdeIntegrationStatus::Success) {
        progress.output.terminal_event_fired = true;
        progress.output.terminal_event_name = Cow::Borrowed(integration_status_name(trial.status));
        progress.output.max_steps_exceeded =
            matches!(trial.status, OdeIntegrationStatus::MaxStepsExceeded);
        progress.output.metrics = coordinator_metrics(
            *progress.total_steps,
            *progress.total_evals,
            elapsed_micros(settings.start_time),
            *progress.collapsed_pairs,
        );
        return Ok(SegmentOutcome::Finished);
    }
    let trial_states = flatten_states(trial.states, trial.n_state);
    if trial_states.len() != trial.times.len() || trial_states.is_empty() {
        return Err(EclipseError::Bracket);
    }
    let endpoint_time = *trial.times.last().ok_or(EclipseError::Bracket)?;
    if !same_time_value(endpoint_time, segment_tf) {
        // `endpoint_delta` anchors the next Encke baseline at `segment_tf`.
        // A solver result ending at a public sample instead would silently
        // pair that stale state with the later epoch.
        return Err(EclipseError::Bracket);
    }
    for &public_time in public_times {
        let returned = trial
            .times
            .iter()
            .position(|time| time.to_bits() == public_time.to_bits())
            .ok_or(EclipseError::Bracket)?;
        let state = *trial_states.get(returned).ok_or(EclipseError::Bracket)?;
        progress.output.times.push(public_time);
        progress
            .output
            .states
            .push(correct_delta_to_original_baseline(
                &state,
                public_time,
                &*progress.curr_equinoc,
                *progress.curr_t,
                &settings.init_equinoc_state,
                settings.t0_s,
            ));
    }
    let endpoint_delta = *trial_states.last().ok_or(EclipseError::Bracket)?;
    if same_time_value(segment_tf, settings.tf_s) {
        progress.output.metrics = coordinator_metrics(
            *progress.total_steps,
            *progress.total_evals,
            elapsed_micros(settings.start_time),
            *progress.collapsed_pairs,
        );
        return Ok(SegmentOutcome::Finished);
    }
    let mut endpoint_baseline = [0.0; 6];
    equinoc2eci_impl(
        &*progress.curr_equinoc,
        6,
        segment_tf - *progress.curr_t,
        0.0,
        &mut endpoint_baseline,
    );
    let mut endpoint_eci = [0.0; 6];
    for ((eci_component, baseline_component), delta_component) in endpoint_eci
        .iter_mut()
        .zip(endpoint_baseline.iter())
        .zip(endpoint_delta.iter())
    {
        *eci_component = *baseline_component + *delta_component;
    }
    eci2equinoc_impl_f64(&endpoint_eci, 6, 0.0, 0.0, progress.curr_equinoc);
    *progress.curr_t = segment_tf;
    *progress.last_root = None;
    #[cfg(feature = "scalar-leg-observer")]
    if let Some(observation) = progress.observation.as_deref_mut() {
        observation.record_encke_rebase();
    }
    Ok(SegmentOutcome::Continue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odesolve::{IntegrationResultSampled, IntegrationStats};
    use crate::types::ForceFlags;
    use satpy_core::{pack_gravity_coeffs, GravityError, MU, SEC_PER_DAY};

    fn gravity_latch_rhs() -> anyhow::Result<LightyearRHS> {
        let c_coeffs = [1.0, 0.0, 0.0, 0.0];
        let s_coeffs = [0.0; 4];
        let packed = Arc::new(pack_gravity_coeffs(&c_coeffs, &s_coeffs, 2, 1)?);
        LightyearRHS::try_new(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            2_451_545.0,
            Arc::new(ForceConfig {
                sph_order: 1,
                ..ForceConfig::default()
            }),
            packed,
        )
    }

    #[test]
    fn solver_boundary_returns_exact_gravity_error_and_drains_both_latches() -> anyhow::Result<()> {
        let rhs = gravity_latch_rhs()?;
        rhs.record_eclipse_error(EclipseError::Geometry);
        if rhs.compute_internal(&[f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0)
            != Err(GravityError::InvalidRadius)
        {
            return Err(anyhow::anyhow!(
                "scalar direct path must return exact gravity error"
            ));
        }

        let Some(gravity) = rhs.take_gravity_error() else {
            return Err(anyhow::anyhow!(
                "scalar solver boundary must consume its gravity latch"
            ));
        };
        if gravity != GravityError::InvalidRadius {
            return Err(anyhow::anyhow!(
                "scalar gravity latch must retain exact error"
            ));
        }
        if resolve_solver_boundary(&rhs, Err::<(), _>(gravity))
            != Err(EclipseError::Gravity(GravityError::InvalidRadius))
        {
            return Err(anyhow::anyhow!(
                "gravity must win solver boundary while eclipse latch drains"
            ));
        }
        if rhs.take_eclipse_error().is_some() || rhs.take_gravity_error().is_some() {
            return Err(anyhow::anyhow!(
                "solver boundary must drain both error latches"
            ));
        }
        Ok(())
    }

    #[test]
    fn exact_zero_span_preserves_requested_grid_and_root_state() {
        let root_t = 42.0_f64;
        let requested = [root_t, root_t];
        let result = zero_span_output(&requested, root_t, root_t).unwrap_or_default();
        assert_eq!(
            result
                .times
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>(),
            requested
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(result.states, vec![[0.0; 6]; requested.len()]);
        assert_eq!(result.metrics.eclipse_collapsed_pairs, 0);
        assert!(zero_span_output(&requested, root_t, root_t + 5.0e-13).is_err());
        assert!(zero_span_output(&[root_t + 5.0e-13], root_t, root_t).is_err());
    }

    #[test]
    fn accepted_step_collapse_reaches_coordinated_result_metrics() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let flags = ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
        crate::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("test ephemeris catalogues must load");

        let jd0 = 2_460_310.5;
        let t0_s = 1_000_000.0_f64;
        let accepted_next_s = f64::from_bits(t0_s.to_bits() + 1);
        let stride = 2;
        let c_coeffs = vec![1.0, 0.0, 0.0, 0.0];
        let s_coeffs = vec![0.0; 4];
        let packed = Arc::new(
            pack_gravity_coeffs(&c_coeffs, &s_coeffs, stride, 0)
                .expect("eclipse-coordinator test gravity coefficients must pack"),
        );
        let config = Arc::new(
            ForceConfig {
                sph_order: 0,
                force_flags: flags,
                atm_model: 3,
                am_ratio: 0.02,
                cr: 1.3,
                dt_max: 60.0,
                integrator_method: StepperMethod::Vern9,
                ..ForceConfig::default()
            }
            .with_ephemeris_for_arc(jd0, jd0 + accepted_next_s / SEC_PER_DAY)
            .expect("test arc must have dynamic ephemeris coverage"),
        );
        let init_equinoc = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let mut rhs = LightyearRHS::try_new(init_equinoc, 0.0, jd0, Arc::clone(&config), packed)
            .expect("scalar eclipse RHS");
        rhs.reset_for_propagation(init_equinoc, 0.0);

        let baseline = rhs.baseline_calculator();
        let baseline_state = baseline.get_baseline_state(t0_s);
        let sun = rhs.eclipse_sun_at(t0_s).expect("dynamic Sun");
        let sun_norm = sun.iter().map(|value| value * value).sum::<f64>().sqrt();
        let sun_unit = sun.map(|value| value / sun_norm);
        let perpendicular_raw = [-sun_unit[1], sun_unit[0], 0.0];
        let perpendicular_norm = perpendicular_raw
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let perpendicular = perpendicular_raw.map(|value| value / perpendicular_norm);
        let radial_km = config.earth_radius + 1.0e-11;
        let target_position = [
            (-7000.0_f64).mul_add(sun_unit[0], radial_km * perpendicular[0]),
            (-7000.0_f64).mul_add(sun_unit[1], radial_km * perpendicular[1]),
            (-7000.0_f64).mul_add(sun_unit[2], radial_km * perpendicular[2]),
        ];
        let mut delta = [0.0; 6];
        for ((delta_component, target_component), baseline_component) in delta
            .iter_mut()
            .zip(target_position)
            .zip(baseline_state)
            .take(3)
        {
            *delta_component = target_component - baseline_component;
        }
        let (position, dynamic_sun) = rhs
            .eclipse_geometry_at_delta(&delta, t0_s)
            .expect("near-tangent geometry");
        let geometry = binary_cylinder_geometry(position, dynamic_sun, config.earth_radius)
            .expect("binary geometry");
        assert_eq!(geometry.side, EclipseSide::Lit);
        assert!(geometry.boundary_margin_km < MAX_BOUNDARY_SEPARATION_KM);
        rhs.set_eclipse_side(geometry.side);

        let derivative = [0.0; 6];
        let mut handler = CoordinatedEventHandler::new(
            &rhs,
            geometry.side,
            t0_s,
            delta,
            1.0e-6,
            1.0e-9,
            50,
            false,
            None,
        );
        assert!(matches!(
            handler.on_step(
                t0_s,
                &delta,
                &derivative,
                accepted_next_s,
                &delta,
                &derivative,
            ),
            OdeEventDecision::Continue
        ));
        let mut collapsed_pairs = handler.take_collapsed_pairs();
        assert_eq!(
            collapsed_pairs, 1,
            "production step scanner must count collapse"
        );

        let mut output = IntegrationResult::default();
        let mut total_steps = 1;
        let mut total_evals = 2;
        let mut curr_equinoc = init_equinoc;
        let mut curr_t = t0_s;
        let mut side = geometry.side;
        let mut eval_index = 1;
        let mut split_count = 0;
        let mut last_root = None;
        let settings = CoordinatorSettings {
            init_equinoc_state: init_equinoc,
            t0_s,
            tf_s: accepted_next_s,
            eps: 1.0e-9,
            stepper: StepperMethod::Vern9,
            forward: true,
            enable_events: false,
            config: &config,
            start_time: std::time::Instant::now(),
        };
        let trial = IntegrationResultSampled {
            times: vec![accepted_next_s],
            states: delta.to_vec(),
            n_state: 6,
            status: OdeIntegrationStatus::Success,
            stats: IntegrationStats::default(),
            event: None,
        };
        #[cfg(feature = "scalar-leg-observer")]
        let mut observation = FinalObservation::new();
        #[cfg(feature = "scalar-leg-observer")]
        {
            observation.record_solver(&trial.stats, trial.status);
            observation.record_encke_segment();
            observation.record_eclipse_collapsed_pairs(collapsed_pairs);
        }
        let outcome = complete_non_eclipse_segment(
            &mut CoordinatorProgress {
                output: &mut output,
                total_steps: &mut total_steps,
                total_evals: &mut total_evals,
                curr_equinoc: &mut curr_equinoc,
                curr_t: &mut curr_t,
                side: &mut side,
                eval_index: &mut eval_index,
                split_count: &mut split_count,
                last_root: &mut last_root,
                collapsed_pairs: &mut collapsed_pairs,
                #[cfg(feature = "scalar-leg-observer")]
                observation: Some(&mut observation),
            },
            &settings,
            trial,
            &[accepted_next_s],
            0,
            accepted_next_s,
            false,
            None,
        )
        .expect("coordinator finalization");
        assert!(matches!(outcome, SegmentOutcome::Finished));
        assert_eq!(output.metrics.eclipse_collapsed_pairs, 1);
        #[cfg(feature = "scalar-leg-observer")]
        assert_eq!(
            observation.into_parts().0,
            Ok(crate::integrator::ObservedFinalMetrics {
                solver_invocations: 1,
                encke_segments: 1,
                eclipse_collapsed_pairs: 1,
                ..crate::integrator::ObservedFinalMetrics::default()
            })
        );
    }

    #[test]
    fn non_eclipse_segment_rejects_missing_solver_endpoint_before_rebase() {
        let mut output = IntegrationResult::default();
        let mut total_steps = 1;
        let mut total_evals = 2;
        let initial_equinoc = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let mut curr_equinoc = initial_equinoc;
        let mut curr_t = 0.0;
        let mut side = EclipseSide::Lit;
        let mut eval_index = 0;
        let mut split_count = 0;
        let mut last_root = None;
        let mut collapsed_pairs = 0;
        let config = ForceConfig::default();
        let settings = CoordinatorSettings {
            init_equinoc_state: initial_equinoc,
            t0_s: 0.0,
            tf_s: 2.0,
            eps: 1.0e-9,
            stepper: StepperMethod::Vern9,
            forward: true,
            enable_events: false,
            config: &config,
            start_time: std::time::Instant::now(),
        };
        let trial = IntegrationResultSampled {
            // A public sample is not an Encke rebase state for the 1 s segment.
            // Accepting it would advance `curr_t` to 1 s while preserving the
            // baseline reconstructed at 0.5 s.
            times: vec![0.5],
            states: vec![0.0; 6],
            n_state: 6,
            status: OdeIntegrationStatus::Success,
            stats: IntegrationStats::default(),
            event: None,
        };

        let result = complete_non_eclipse_segment(
            &mut CoordinatorProgress {
                output: &mut output,
                total_steps: &mut total_steps,
                total_evals: &mut total_evals,
                curr_equinoc: &mut curr_equinoc,
                curr_t: &mut curr_t,
                side: &mut side,
                eval_index: &mut eval_index,
                split_count: &mut split_count,
                last_root: &mut last_root,
                collapsed_pairs: &mut collapsed_pairs,
                #[cfg(feature = "scalar-leg-observer")]
                observation: None,
            },
            &settings,
            trial,
            &[0.5],
            0,
            1.0,
            false,
            None,
        );

        assert!(matches!(result, Err(EclipseError::Bracket)));
        assert_eq!(curr_t.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            curr_equinoc.map(f64::to_bits),
            initial_equinoc.map(f64::to_bits)
        );
        assert!(output.times.is_empty());
        assert!(output.states.is_empty());
    }

    #[test]
    fn origin_crossing_commits_without_an_old_side_proof_leg() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let flags = ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
        crate::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("test ephemeris catalogues must load");

        let jd0 = 2_460_310.5;
        let base_t = 1_000_000.0_f64;
        let accepted_t_new = base_t + 60.0;
        let packed = Arc::new(
            pack_gravity_coeffs(&[1.0, 0.0, 0.0, 0.0], &[0.0; 4], 2, 0)
                .expect("origin-crossing test gravity coefficients must pack"),
        );
        let config = Arc::new(
            ForceConfig {
                sph_order: 0,
                force_flags: flags,
                atm_model: 3,
                am_ratio: 0.02,
                cr: 1.3,
                dt_max: 60.0,
                integrator_method: StepperMethod::Vern9,
                ..ForceConfig::default()
            }
            .with_ephemeris_for_arc(jd0, jd0 + accepted_t_new / SEC_PER_DAY)
            .expect("test arc must have dynamic ephemeris coverage"),
        );
        let init_equinoc = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let mut rhs = LightyearRHS::try_new(init_equinoc, 0.0, jd0, Arc::clone(&config), packed)
            .expect("scalar eclipse RHS");
        rhs.reset_for_propagation(init_equinoc, base_t);

        let bracket_with_old_t = |bracket_t_old: f64| EclipseBracket {
            accepted_t_old: base_t,
            accepted_t_new,
            accepted_eci_old: [0.0; 6],
            t_old: bracket_t_old,
            // A restart-produced origin crossing is vanishingly narrow.
            t_new: bracket_t_old + 1.19e-10,
            old_side: EclipseSide::Shadow,
            new_side: EclipseSide::Lit,
            geometry_motion_bound_km: 0.0,
        };
        let settings = CoordinatorSettings {
            init_equinoc_state: init_equinoc,
            t0_s: base_t,
            tf_s: accepted_t_new,
            eps: 1.0e-9,
            stepper: StepperMethod::Vern9,
            forward: true,
            enable_events: false,
            config: &config,
            start_time: std::time::Instant::now(),
        };
        let public_candidates = [base_t + 10.0, base_t + 20.0];

        // The mechanism: a proof point search across a zero-width old-side
        // interval is the degenerate call that raised NonProgress.
        assert_eq!(
            deepest_directed_time_within_root_bound(&rhs, base_t, base_t, ROOT_OLD_SIDE_MARGIN_KM),
            Err(EclipseError::NonProgress)
        );

        // An origin crossing must not make that call at all. It returns the
        // origin as its own proof point and claims none of the public times,
        // which hands every one of them to the window leg.
        let origin = root_old_side_proof_point(
            &rhs,
            &settings,
            bracket_with_old_t(base_t).t_old,
            base_t,
            &public_candidates,
        )
        .expect("an origin crossing must not need an old-side proof leg");
        assert_eq!(origin.0.to_bits(), base_t.to_bits());
        assert_eq!(origin.1, 0);

        // A bracket with real old-side room still proves it on an interior
        // point, strictly inside the approach.
        let interior_t_old = base_t - 30.0;
        let (proof_old_t, proof_public_count) = root_old_side_proof_point(
            &rhs,
            &settings,
            bracket_with_old_t(interior_t_old).t_old,
            base_t,
            &public_candidates,
        )
        .expect("a bracket with old-side room must find a proof point");
        assert!(proof_old_t > interior_t_old && proof_old_t <= base_t);
        assert_ne!(proof_old_t.to_bits(), interior_t_old.to_bits());
        assert_eq!(
            proof_public_count, 0,
            "no public time precedes the proof point"
        );

        // With no proof leg the window opens on the origin's own state, and no
        // geometry is re-derived to get there.
        assert_eq!(
            proven_old_side_delta(&rhs, None, base_t, EclipseSide::Shadow),
            Ok([0.0; 6])
        );
    }

    /// The post-root scan skips `[prev_t, edge]` and nothing else scans it, so
    /// a grazing out-and-back crossing pair inside that window is dropped. What
    /// keeps that sound is not a certificate but the window's DURATION, and the
    /// duration is set by `MAX_ROOT_TOTAL_UNCERTAINTY_KM` — a constant the skip
    /// site never re-derives. Raising it silently widens the unscanned window.
    ///
    /// So pin the headroom rather than the constant. Eclipse side gates SRP
    /// alone, so a dropped excursion costs one SRP acceleration applied with the
    /// wrong sign for the window's duration; the resulting arc position error
    /// must stay far under the millimetre the detector itself resolves a bracket
    /// to. This fails if that headroom is spent.
    #[test]
    fn post_root_skip_window_stays_far_under_the_bracket_certificate() {
        // The window is sized by the motion bound, whose dominant term is
        // `PART_A_ECLIPSE_SPEED_CAP_KM_S * dt`, so the skipped span is at most:
        let window_s = MAX_ROOT_TOTAL_UNCERTAINTY_KM / PART_A_ECLIPSE_SPEED_CAP_KM_S;

        // Compiled dust SRP acceleration, m/s^2: p_sun * cr * (A/m).
        let hybrid = nd_config::CompiledPartAScienceV1::part_a_v1().hybrid();
        let srp_accel_m_s2 = 4.56e-6 * hybrid.dust_cr * hybrid.dust_am_ratio;

        // Getting the side wrong for the whole window, propagated over a 12 h arc.
        let arc_s = 43_200.0_f64;
        let position_error_m = srp_accel_m_s2 * window_s * arc_s;

        let certificate_m = MAX_BOUNDARY_SEPARATION_KM * 1_000.0;
        assert!(
            position_error_m * 100.0 < certificate_m,
            "the post-root skip has eaten its headroom: a dropped grazing pair now costs \
             {position_error_m:e} m over a 12 h arc against a {certificate_m:e} m bracket \
             certificate (window {window_s:e} s). Nothing scans that window, so restoring \
             margin means lowering MAX_ROOT_TOTAL_UNCERTAINTY_KM or scanning [prev_t, edge]."
        );
    }

    /// Bit-neutrality, as a theorem rather than a measurement.
    ///
    /// `post_root_decisive_margin_km` may never return more than the flat
    /// `MAX_BOUNDARY_SEPARATION_KM` it replaced. That single fact is what makes
    /// the tightened gate safe to land: `margin > MAX_BOUNDARY_SEPARATION_KM`
    /// implies `margin > post_root_decisive_margin_km(..)`, so every accepted
    /// step stays accepted, from the same `edge`, with the same `scan_from`, and
    /// only steps that used to raise `Chatter` can change.
    #[test]
    fn post_root_decisive_margin_never_exceeds_the_detector_resolution() {
        // Every transport error a published root can carry: the rebase rejects
        // anything above `MAX_BOUNDARY_SEPARATION_KM` before publishing.
        let mut transport_km = 0.0_f64;
        for _ in 0..2_000 {
            let bound = post_root_decisive_margin_km(transport_km)
                .expect("an in-budget transport error must yield a bound");
            assert!(
                bound <= MAX_BOUNDARY_SEPARATION_KM,
                "the tightened gate must never out-demand the flat bound it replaced: \
                 transport {transport_km:.6e} km produced {bound:.6e} km against \
                 {MAX_BOUNDARY_SEPARATION_KM:.6e} km"
            );
            assert!(
                bound >= MIN_POST_ROOT_DECISIVE_MARGIN_KM.min(MAX_BOUNDARY_SEPARATION_KM),
                "the gate must never fall below its own floor"
            );
            transport_km += MAX_BOUNDARY_SEPARATION_KM / 1_000.0;
        }
        // Even a nonsensically large transport error clamps rather than
        // out-demanding the old constant.
        assert_eq!(
            post_root_decisive_margin_km(1.0),
            Ok(MAX_BOUNDARY_SEPARATION_KM)
        );
        assert_eq!(
            post_root_decisive_margin_km(-1.0),
            Err(EclipseError::Geometry)
        );
        assert_eq!(
            post_root_decisive_margin_km(f64::NAN),
            Err(EclipseError::Geometry)
        );
    }

    /// The defect this gate was changed to fix, stated numerically.
    ///
    /// The scan start sits at most `MAX_ROOT_TOTAL_UNCERTAINTY_KM /
    /// PART_A_ECLIPSE_SPEED_CAP_KM_S` past the root. Demanding a full
    /// `MAX_BOUNDARY_SEPARATION_KM` of margin there is therefore a hidden
    /// demand on the crossing's boundary-normal speed. NORAD 40054 at
    /// t0 + 62,804.6376 s crosses at 0.0906 km/s and was rejected as roundoff
    /// with clean, monotone geometry.
    #[test]
    fn flat_gate_imposed_an_undeclared_transversality_floor() {
        let window_s = MAX_ROOT_TOTAL_UNCERTAINTY_KM / PART_A_ECLIPSE_SPEED_CAP_KM_S;
        let required_normal_speed_km_s = MAX_BOUNDARY_SEPARATION_KM / window_s;

        // The measured crossing that the flat gate rejected.
        let observed_normal_speed_km_s = 0.0906_f64;
        assert!(
            observed_normal_speed_km_s < required_normal_speed_km_s,
            "fixture must reproduce the rejection the flat gate performed"
        );

        // Against the error that actually contaminates the scan start - the
        // measured rebase transport, 1.07e-12 km on that root - the same margin
        // is decisive by five orders of magnitude.
        let observed_margin_km = observed_normal_speed_km_s * window_s;
        let bound = post_root_decisive_margin_km(1.07e-12).expect("measured transport error");
        assert!(
            observed_margin_km > bound * 100.0,
            "the tightened gate must resolve this crossing with room to spare: \
             margin {observed_margin_km:.6e} km against bound {bound:.6e} km"
        );
    }

    #[test]
    fn post_root_step_scans_from_the_certified_edge() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let flags = ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
        crate::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("test ephemeris catalogues must load");

        let jd0 = 2_460_310.5;
        let root_t = 1_000_000.0_f64;
        let step_end_s = root_t + 1.0;
        let packed = Arc::new(
            pack_gravity_coeffs(&[1.0, 0.0, 0.0, 0.0], &[0.0; 4], 2, 0)
                .expect("edge-scan test gravity coefficients must pack"),
        );
        let config = Arc::new(
            ForceConfig {
                sph_order: 0,
                force_flags: flags,
                atm_model: 3,
                am_ratio: 0.02,
                cr: 1.3,
                dt_max: 60.0,
                integrator_method: StepperMethod::Vern9,
                ..ForceConfig::default()
            }
            .with_ephemeris_for_arc(jd0, jd0 + step_end_s / SEC_PER_DAY)
            .expect("test arc must have dynamic ephemeris coverage"),
        );
        let init_equinoc = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let mut rhs = LightyearRHS::try_new(init_equinoc, 0.0, jd0, Arc::clone(&config), packed)
            .expect("scalar eclipse RHS");
        rhs.reset_for_propagation(init_equinoc, 0.0);

        let baseline = rhs.baseline_calculator();
        let sun = rhs.eclipse_sun_at(root_t).expect("dynamic Sun");
        let sun_norm = sun.iter().map(|value| value * value).sum::<f64>().sqrt();
        let sun_unit = sun.map(|value| value / sun_norm);
        let perpendicular_raw = [-sun_unit[1], sun_unit[0], 0.0];
        let perpendicular_norm = perpendicular_raw
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let perpendicular = perpendicular_raw.map(|value| value / perpendicular_norm);

        // A path behind the Earth that grazes the shadow cylinder from the lit
        // side at the root instant and then sinks into shadow at 1 km/s. The
        // root instant's own margin is the `1.0e-11` km offset below, i.e.
        // 10 nm -- not the "10 pm" this comment claimed until 2026-08-04, which
        // was a 1000x understatement of the very quantity the fixture is built
        // around. 10 nm is still five orders of magnitude below the one
        // millimetre the detector scan brackets to, so geometry re-derived at
        // the root instant is roundoff and the fixture's premise holds.
        let inward_rate_km_s = -1.0_f64;
        let radial_at =
            |t: f64| inward_rate_km_s.mul_add(t - root_t, config.earth_radius + 1.0e-11);
        let target_state_at = |t: f64| {
            let radial_km = radial_at(t);
            [
                (-7000.0_f64).mul_add(sun_unit[0], radial_km * perpendicular[0]),
                (-7000.0_f64).mul_add(sun_unit[1], radial_km * perpendicular[1]),
                (-7000.0_f64).mul_add(sun_unit[2], radial_km * perpendicular[2]),
                inward_rate_km_s * perpendicular[0],
                inward_rate_km_s * perpendicular[1],
                inward_rate_km_s * perpendicular[2],
            ]
        };
        let state_and_derivative_at = |t: f64| {
            let baseline_state = baseline.get_baseline_state(t);
            let target = target_state_at(t);
            let mut delta = [0.0; 6];
            for ((component, target_component), baseline_component) in delta
                .iter_mut()
                .zip(target.iter())
                .zip(baseline_state.iter())
            {
                *component = *target_component - *baseline_component;
            }
            // The reconstruction adds the delta derivative's first three
            // components to the baseline velocity, so the delta derivative that
            // yields the target velocity is the target velocity itself minus
            // the baseline's.
            let mut derivative = [0.0; 6];
            for (component, delta_velocity) in
                derivative.iter_mut().zip(delta.iter().skip(3)).take(3)
            {
                *component = *delta_velocity;
            }
            (delta, derivative)
        };
        let (root_delta, root_derivative) = state_and_derivative_at(root_t);
        let (end_delta, end_derivative) = state_and_derivative_at(step_end_s);

        let (root_position, root_sun) = rhs
            .eclipse_geometry_at_delta(&root_delta, root_t)
            .expect("root geometry");
        let root_geometry = binary_cylinder_geometry(root_position, root_sun, config.earth_radius)
            .expect("binary geometry");
        assert_eq!(
            root_geometry.side,
            EclipseSide::Lit,
            "the construction must contradict the committed side at the root"
        );
        assert!(
            root_geometry.boundary_margin_km < MAX_BOUNDARY_SEPARATION_KM,
            "the contradiction must sit inside the transport error, not outside it"
        );

        let committed_side = EclipseSide::Shadow;
        rhs.set_eclipse_side(committed_side);

        // Without the root's certification this step is indistinguishable from
        // genuine chatter, and the handler must still reject it.
        let mut uncertified = CoordinatedEventHandler::new(
            &rhs,
            committed_side,
            root_t,
            root_delta,
            1.0e-6,
            1.0e-9,
            50,
            false,
            None,
        );
        let _ = uncertified.on_step(
            root_t,
            &root_delta,
            &root_derivative,
            step_end_s,
            &end_delta,
            &end_derivative,
        );
        assert_eq!(
            uncertified.take_eclipse_outcome(),
            Err(EclipseError::Chatter)
        );

        // With it, the scan starts at the certified edge, where geometry agrees
        // with the committed side by a resolvable margin, and the step completes.
        let mut certified = CoordinatedEventHandler::new(
            &rhs,
            committed_side,
            root_t,
            root_delta,
            1.0e-6,
            1.0e-9,
            50,
            false,
            // The largest transport error observed on a production root, so the
            // fixture is charged a realistic rebase cost rather than zero.
            Some(CertifiedRoot {
                t: root_t,
                transport_error_km: 3.8e-12,
            }),
        );
        let decision = certified.on_step(
            root_t,
            &root_delta,
            &root_derivative,
            step_end_s,
            &end_delta,
            &end_derivative,
        );
        assert!(matches!(decision, OdeEventDecision::Continue));
        assert_eq!(certified.take_eclipse_outcome(), Ok(None));

        // The edge the handler scans from must be strictly inside the step and
        // decisive: the root bound reaches it in well under a millisecond.
        let edge = deepest_directed_time_within_root_bound(
            &rhs,
            root_t,
            step_end_s,
            MAX_ROOT_TOTAL_UNCERTAINTY_KM,
        )
        .expect("certified edge");
        assert!(edge > root_t && edge < step_end_s);
        assert!(
            radial_at(edge) < config.earth_radius,
            "the certified edge must land on the committed side"
        );
    }

    fn rebase_root(
        lane_rhs: &mut LightyearRHS,
        init_equinoc: [f64; 6],
        t0_s: f64,
        root_t: f64,
        root_delta: [f64; 6],
        new_side: EclipseSide,
    ) -> (Result<(), EclipseError>, Option<CertifiedRoot>) {
        let mut output = IntegrationResult::default();
        let mut total_steps = 0;
        let mut total_evals = 0;
        let mut curr_equinoc = init_equinoc;
        let mut curr_t = t0_s;
        let mut side = EclipseSide::Lit;
        let mut eval_index = 0;
        let mut split_count = 0;
        let mut last_root = None;
        let mut collapsed_pairs = 0;
        let result = rebase_after_eclipse_root(
            &mut CoordinatorProgress {
                output: &mut output,
                total_steps: &mut total_steps,
                total_evals: &mut total_evals,
                curr_equinoc: &mut curr_equinoc,
                curr_t: &mut curr_t,
                side: &mut side,
                eval_index: &mut eval_index,
                split_count: &mut split_count,
                last_root: &mut last_root,
                collapsed_pairs: &mut collapsed_pairs,
                #[cfg(feature = "scalar-leg-observer")]
                observation: None,
            },
            lane_rhs,
            root_t,
            root_delta,
            new_side,
        );
        (result, last_root)
    }

    /// ECI position a rebase would carry `root_eci` to, through the same
    /// equinoctial round trip `rebase_after_eclipse_root` performs.
    fn equinoctial_transport_error_km(root_eci: [f64; 6]) -> f64 {
        let mut equinoc = [0.0; 6];
        eci2equinoc_impl_f64(&root_eci, 6, 0.0, 0.0, &mut equinoc);
        let mut transported = [0.0; 6];
        equinoc2eci_impl(&equinoc, 6, 0.0, 0.0, &mut transported);
        distance3(
            [root_eci[0], root_eci[1], root_eci[2]],
            [transported[0], transported[1], transported[2]],
        )
    }

    #[test]
    fn rebase_bounds_transport_error_and_trusts_the_committed_side() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let flags = ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
        crate::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("test ephemeris catalogues must load");

        let jd0 = 2_460_310.5;
        let t0_s = 0.0_f64;
        let root_t = 60.0_f64;
        let packed = Arc::new(
            pack_gravity_coeffs(&[1.0, 0.0, 0.0, 0.0], &[0.0; 4], 2, 0)
                .expect("eclipse-rebase test gravity coefficients must pack"),
        );
        let config = Arc::new(
            ForceConfig {
                sph_order: 0,
                force_flags: flags,
                atm_model: 3,
                am_ratio: 0.02,
                cr: 1.3,
                dt_max: 60.0,
                integrator_method: StepperMethod::Vern9,
                ..ForceConfig::default()
            }
            .with_ephemeris_for_arc(jd0, jd0 + root_t / SEC_PER_DAY)
            .expect("test arc must have dynamic ephemeris coverage"),
        );
        let init_equinoc = [7050.0, 0.001, -0.0008, 0.0005, -0.0004, 0.2];
        let mut lane_rhs =
            LightyearRHS::try_new(init_equinoc, t0_s, jd0, Arc::clone(&config), packed)
                .expect("scalar eclipse RHS");
        lane_rhs.reset_for_propagation(init_equinoc, t0_s);

        // The side a committed root wrote is authority at the root instant. A
        // rebase must accept it even where re-derived geometry disagrees, since
        // there the boundary margin is smaller than the transport error that
        // carrying the root onto this lane's baseline introduces.
        let mut root_baseline = [0.0; 6];
        equinoc2eci_impl(&init_equinoc, 6, root_t - t0_s, 0.0, &mut root_baseline);
        let contradicted_side = {
            let (position, sun) = lane_rhs
                .eclipse_geometry_at_delta(&[0.0; 6], root_t)
                .expect("root geometry");
            let derived = binary_cylinder_geometry(position, sun, config.earth_radius)
                .expect("binary geometry")
                .side;
            match derived {
                EclipseSide::Lit => EclipseSide::Shadow,
                EclipseSide::Shadow => EclipseSide::Lit,
            }
        };
        let (accepted, accepted_last_root) = rebase_root(
            &mut lane_rhs,
            init_equinoc,
            t0_s,
            root_t,
            [0.0; 6],
            contradicted_side,
        );
        assert_eq!(
            accepted,
            Ok(()),
            "a rebase must not re-derive the side the root transaction committed"
        );
        assert_eq!(
            accepted_last_root.map(|root| root.t.to_bits()),
            Some(root_t.to_bits()),
            "an in-transport-budget rebase must publish the root"
        );
        let published_transport_km = accepted_last_root
            .expect("an in-transport-budget rebase publishes a root")
            .transport_error_km;
        assert!(
            published_transport_km.is_finite()
                && (0.0..=MAX_BOUNDARY_SEPARATION_KM).contains(&published_transport_km),
            "the published transport error must be the measured one, inside the budget \
             the rebase just enforced (got {published_transport_km:.6e} km)"
        );

        // What the rebase does still owe is its own transport error. Displace
        // the root far enough that the equinoctial round trip alone exceeds the
        // boundary separation, and the rebase must refuse to publish.
        // The round trip is exact at some phases, so walk a fixed sweep and
        // take the first circular state whose transport error clears the bound.
        let far_radius_km = 1.0e12_f64;
        let far_speed_km_s = (MU / far_radius_km).sqrt();
        let far_root_eci = (0..64)
            .map(|index| {
                let phase = f64::from(index) * 0.1;
                [
                    far_radius_km * phase.cos(),
                    far_radius_km * phase.sin() * 0.8,
                    far_radius_km * phase.sin() * 0.6,
                    -far_speed_km_s * phase.sin(),
                    far_speed_km_s * phase.cos() * 0.8,
                    far_speed_km_s * phase.cos() * 0.6,
                ]
            })
            .find(|candidate| {
                equinoctial_transport_error_km(*candidate) > MAX_BOUNDARY_SEPARATION_KM
            })
            .expect("a far circular orbit whose equinoctial round trip exceeds the bound");
        let mut far_delta = [0.0; 6];
        for ((delta, eci), baseline) in far_delta
            .iter_mut()
            .zip(far_root_eci.iter())
            .zip(root_baseline.iter())
        {
            *delta = *eci - *baseline;
        }
        let transport_error_km = equinoctial_transport_error_km(far_root_eci);
        assert!(
            transport_error_km > MAX_BOUNDARY_SEPARATION_KM,
            "test construction must actually exceed the transport budget \
             (got {transport_error_km:.6e} km)"
        );

        let (rejected, rejected_last_root) = rebase_root(
            &mut lane_rhs,
            init_equinoc,
            t0_s,
            root_t,
            far_delta,
            EclipseSide::Shadow,
        );
        assert_eq!(rejected, Err(EclipseError::Bracket));
        assert!(
            rejected_last_root.is_none(),
            "an over-budget rebase must not publish a root"
        );
    }

    #[test]
    fn checked_count_add_rejects_overflow_without_mutating_counter() {
        let mut count = usize::MAX;
        assert_eq!(
            checked_count_add(&mut count, 1),
            Err(EclipseError::SplitLimit)
        );
        assert_eq!(count, usize::MAX);
    }

    #[test]
    fn same_time_value_preserves_ieee_equality_semantics() {
        assert!(same_time_value(0.0, -0.0));
        assert!(same_time_value(f64::INFINITY, f64::INFINITY));
        assert!(!same_time_value(f64::NAN, f64::NAN));
        assert!(!same_time_value(1.0, 2.0));
    }

    #[test]
    fn outward_rounding_rejects_nonfinite_result() {
        assert_eq!(outward_nonnegative(f64::MAX), Err(EclipseError::Geometry));
    }

    #[test]
    fn micros_to_u64_saturates_instead_of_wrapping() {
        assert_eq!(micros_to_u64(u128::from(u64::MAX)), u64::MAX);
        assert_eq!(
            micros_to_u64(u128::from(u64::MAX).saturating_add(1)),
            u64::MAX
        );
    }
}
