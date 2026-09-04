use satpy_core::GravityError;

// One millimetre, which sits two orders of magnitude beneath the 0.10 m
// endpoint budget, leaving room for replay/rebase rounding and force-side
// switching.
//
// That ratio is a statement about error budget, NOT about how far this constant
// can be raised, and the difference has been measured. On the V3 production arc
// at `286dad1` (Vern7, atmosphere model 7), spending the first order is real but
// modest and spending the second does not work at all:
//
//   1 mm     4,495 scan entries, 2,133 splits, 42 crossings   (what ships)
//   10 mm    4,016 scan entries,               42 crossings   -10.7% entries
//   100 mm   the arc does not propagate: `EclipseError::NonProgress`
//
// So the usable slack is ONE order of magnitude, and even that is not free: the
// scan bisects to a binary64 fixed point, so any move here relocates every
// committed root and re-pins the strict-HF and rect-loop digests, with a sign
// nothing controls.
//
// The entry counts above are smaller than the ones this comment used to carry
// (10,273 and 9,200), because the split-margin no-crossing certificate removed
// more than half of all scan entries without moving a root. That shrank what
// this lever is worth as well as what it costs: it now buys 479 entries rather
// than 1,073, on a population less than half the size. The "about 1.2% of arc
// wall" once quoted here was measured against the old population and no longer
// applies; the 10 mm arm has NOT been re-timed on the wall since. Treat 10 mm as
// a declined lever whose remaining upside is unmeasured and whose re-pin bill is
// unchanged, not as headroom lying around.
//
// The instrument that produced every entry count above is NOT in this tree. The
// exit-class census and the crossing vs no-crossing attribution live on the
// unmerged `r35-speed` branch (`3b29464`), as
// `examples/r35_scan_census.rs` plus scan counters in this file. It is a probe,
// NOT FOR MERGE: it prints from inside the scan and would cost an entry-rate
// branch on the flown path. The superseded 10,273-entry population is its
// output, so re-deriving or re-splitting any of these counts means reinstating
// it from that branch rather than rebuilding it from scratch.
pub const MAX_BOUNDARY_SEPARATION_KM: f64 = 1.0e-6;

/// Numerical envelope for Part A Earth-orbit eclipse splitting. Valid campaign
/// orbits stay above the ground guard, below the authorized 41,378 km default
/// apogee ceiling, and below 11 km/s even after the 2 km/s transfer bound.
/// These wider limits are numerical fail-closed guards, not feasibility tests.
pub const PART_A_ECLIPSE_RADIUS_MIN_KM: f64 = 6_000.0;
pub const PART_A_ECLIPSE_RADIUS_CAP_KM: f64 = 50_000.0;
// At the 6,578.137 km authorized minimum perigee, two-body escape speed is
// below 11.1 km/s. Adding compiled Hybrid's 7.5 km/s physical delta-v ceiling
// remains below 18.6 km/s; 20 km/s preserves explicit headroom.
pub const PART_A_ECLIPSE_SPEED_CAP_KM_S: f64 = 20.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EclipseSide {
    Lit,
    Shadow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EclipseError {
    /// Exact packed-gravity evaluation failure observed by the scalar
    /// eclipse-coordinator solver boundary.
    Gravity(GravityError),
    /// The strict-HF enclosure REFUSED the configuration -- an authority
    /// decision, not a numerical one.
    ///
    /// This existed as `Geometry` until 2026-08-19, because both `try_new`
    /// calls below flattened their error with `map_err(|_| Geometry)`. That one
    /// character cost a full diagnosis: an `IdentityMismatch(Science)` from
    /// `strict_hf_enclosure::issue_for_rhs` surfaced as
    /// `MissAtZeroHfIntegrateFailure` with `det_mass` NaN, so a refused
    /// configuration was indistinguishable from a failed integration, and the
    /// Hybrid lane degraded every row to a frozen terminal instead of failing
    /// loudly about its own inputs.
    Authority(crate::strict_hf_enclosure::StrictHfAuthorityError),
    Geometry,
    UninitializedSide,
    NonProgress,
    Chatter,
    Bracket,
    EventOverlap,
    SplitLimit,
    Envelope,
}

impl std::fmt::Display for EclipseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gravity(error) => write!(formatter, "binary eclipse gravity: {error}"),
            Self::Authority(error) => {
                write!(formatter, "binary eclipse strict-HF authority: {error:?}")
            }
            error => write!(formatter, "binary eclipse {error:?}"),
        }
    }
}

impl std::error::Error for EclipseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gravity(error) => Some(error),
            _ => None,
        }
    }
}

/// Preserve the established split-point rounding used by the binary eclipse
/// root scanner. Fusing this expression changes the representable midpoint
/// near a boundary and can alter the first bracket chosen.
#[must_use]
fn binary64_midpoint(old: f64, new: f64) -> f64 {
    old + 0.5 * (new - old)
}

/// `+0.0` and `-0.0` both represent a zero-duration integration interval.
#[must_use]
#[expect(
    clippy::float_cmp,
    reason = "zero-span detection must retain IEEE signed-zero equality and exact endpoint identity"
)]
fn is_zero_span(t_old: f64, t_new: f64) -> bool {
    t_old == t_new
}

pub fn validate_part_a_eclipse_envelope(
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
) -> Result<(), EclipseError> {
    let norm = |vector: [f64; 3]| {
        vector[0]
            .mul_add(
                vector[0],
                vector[1].mul_add(vector[1], vector[2] * vector[2]),
            )
            .sqrt()
    };
    let radius = norm(position_km);
    let speed = norm(velocity_km_s);
    if position_km
        .iter()
        .chain(velocity_km_s.iter())
        .all(|value| value.is_finite())
        && radius.is_finite()
        && speed.is_finite()
        && (PART_A_ECLIPSE_RADIUS_MIN_KM..=PART_A_ECLIPSE_RADIUS_CAP_KM).contains(&radius)
        && speed <= PART_A_ECLIPSE_SPEED_CAP_KM_S
    {
        Ok(())
    } else {
        Err(EclipseError::Envelope)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EclipseBracket {
    pub accepted_t_old: f64,
    pub accepted_t_new: f64,
    pub accepted_eci_old: [f64; 6],
    pub t_old: f64,
    pub t_new: f64,
    pub old_side: EclipseSide,
    pub new_side: EclipseSide,
    /// Certified upper bound on satellite motion plus cylinder-axis motion
    /// across this bracket. This, not endpoint chord length, is the spatial
    /// uncertainty gate.
    pub geometry_motion_bound_km: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EclipseScanResult {
    pub crossing: Option<EclipseBracket>,
    pub collapsed_pairs: usize,
}

#[inline]
pub fn classify_binary_cylinder(
    satellite_position_km: [f64; 3],
    sun_position_km: [f64; 3],
    earth_radius_km: f64,
) -> Result<EclipseSide, EclipseError> {
    Ok(binary_cylinder_geometry(satellite_position_km, sun_position_km, earth_radius_km)?.side)
}

#[derive(Clone, Copy)]
pub struct BinaryCylinderGeometry {
    pub side: EclipseSide,
    pub boundary_margin_km: f64,
}

pub fn binary_cylinder_geometry(
    satellite_position_km: [f64; 3],
    sun_position_km: [f64; 3],
    earth_radius_km: f64,
) -> Result<BinaryCylinderGeometry, EclipseError> {
    if !(satellite_position_km.iter().all(|value| value.is_finite())
        && sun_position_km.iter().all(|value| value.is_finite())
        && earth_radius_km.is_finite()
        && earth_radius_km > 0.0)
    {
        return Err(EclipseError::Geometry);
    }
    let sun_norm_sq = sun_position_km
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    if !(sun_norm_sq.is_finite() && sun_norm_sq > 0.0) {
        return Err(EclipseError::Geometry);
    }
    let inverse_sun_norm = 1.0 / sun_norm_sq.sqrt();
    let sun_direction = [
        sun_position_km[0] * inverse_sun_norm,
        sun_position_km[1] * inverse_sun_norm,
        sun_position_km[2] * inverse_sun_norm,
    ];
    let axial = satellite_position_km[0].mul_add(
        sun_direction[0],
        satellite_position_km[1].mul_add(
            sun_direction[1],
            satellite_position_km[2] * sun_direction[2],
        ),
    );
    let radial = [
        satellite_position_km[0] - axial * sun_direction[0],
        satellite_position_km[1] - axial * sun_direction[1],
        satellite_position_km[2] - axial * sun_direction[2],
    ];
    let radial_sq = radial[0].mul_add(
        radial[0],
        radial[1].mul_add(radial[1], radial[2] * radial[2]),
    );
    let radius_sq = earth_radius_km * earth_radius_km;
    if !(axial.is_finite() && radial_sq.is_finite() && radius_sq.is_finite()) {
        return Err(EclipseError::Geometry);
    }
    let radial_distance = radial_sq.sqrt();
    let signed_radial_margin_km = (radial_sq - radius_sq) / (radial_distance + earth_radius_km);
    let (side, boundary_margin_km) = if axial < 0.0 && radial_sq < radius_sq {
        (EclipseSide::Shadow, (-axial).min(-signed_radial_margin_km))
    } else {
        let margin = match (axial >= 0.0, radial_sq >= radius_sq) {
            (true, true) => axial.max(signed_radial_margin_km),
            (true, false) => axial,
            (false, true) => signed_radial_margin_km,
            (false, false) => 0.0,
        };
        (EclipseSide::Lit, margin)
    };
    if !(boundary_margin_km.is_finite() && boundary_margin_km >= 0.0) {
        return Err(EclipseError::Geometry);
    }
    Ok(BinaryCylinderGeometry {
        side,
        boundary_margin_km,
    })
}

/// One end of a scan interval as the scan already knows it: the instant and the
/// continuous-path state evaluated there.
///
/// The motion bound is a function of the two endpoint states, and the scan has
/// those states in hand from `state_at`. Handing the bound the states instead of
/// the times is what keeps it from re-deriving them: on the production path the
/// bound and the scan share one Hermite interpolant, so re-derivation returns
/// the same bits it was already given.
#[derive(Clone, Copy)]
pub struct ScanEndpointState {
    pub t: f64,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
}

#[derive(Clone, Copy)]
struct ScanEndpoint {
    t: f64,
    position: [f64; 3],
    velocity: [f64; 3],
    side: EclipseSide,
    boundary_margin_km: f64,
}

impl ScanEndpoint {
    const fn state(self) -> ScanEndpointState {
        ScanEndpointState {
            t: self.t,
            position: self.position,
            velocity: self.velocity,
        }
    }
}

/// Sufficient condition for "this interval contains no side change", given
/// `motion_bound`: a certified upper bound on the TOTAL geometry motion —
/// satellite path plus cylinder-axis sweep — across the interval.
///
/// A boundary reached at an interior instant `t` has to be reached from both
/// ends. Getting there from the old endpoint costs at least `margin_old` of
/// motion, and getting from there to the new endpoint costs at least
/// `margin_new`. True motion is additive across `t`, so those two costs are
/// paid out of one `motion_bound`:
///
/// ```text
///   margin_old + margin_new <= motion([old,t]) + motion([t,new]) <= motion_bound
/// ```
///
/// Contrapositive: `margin_old + margin_new > motion_bound` proves no boundary
/// is reachable anywhere inside, and the shared endpoint side therefore holds
/// throughout.
///
/// This replaced `min(margin_old, margin_new) > motion_bound`, which spends the
/// whole budget twice over — once against each endpoint — and so demands the
/// nearer endpoint alone out-run the entire interval's motion. That is the
/// binding constraint exactly where it is most pessimistic: the sub-intervals
/// abutting a bracket, whose inner endpoint sits arbitrarily close to the root
/// and whose margin therefore tends to zero. Those intervals could only be
/// certified by bisecting them to nothing, and proving them empty is what the
/// earliest-crossing guarantee costs.
///
/// Both forms certify the same predicate on the same premise, and for
/// non-negative margins `min(x, y) > b` implies `x + y > b`, so this one fires
/// wherever the old one did and additionally where the margins are lopsided. It
/// prunes only intervals that provably hold no crossing, so it cannot move a
/// root: on the V3 production arc and a 17-case geometry sweep every committed
/// bracket, every collapsed-pair count and every final state stayed
/// bit-identical while scan entries fell 52.7%.
#[inline]
fn no_crossing_certified(margin_old: f64, margin_new: f64, motion_bound: f64) -> bool {
    margin_old + margin_new > motion_bound
}

fn separation_km(left: [f64; 3], right: [f64; 3]) -> Result<f64, EclipseError> {
    let dx = right[0] - left[0];
    let dy = right[1] - left[1];
    let dz = right[2] - left[2];
    let separation = dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt();
    if separation.is_finite() {
        Ok(separation)
    } else {
        Err(EclipseError::Geometry)
    }
}

fn endpoint<F>(t: f64, earth_radius_km: f64, state_at: &mut F) -> Result<ScanEndpoint, EclipseError>
where
    F: FnMut(f64) -> Result<([f64; 3], [f64; 3], [f64; 3]), EclipseError>,
{
    if !t.is_finite() {
        return Err(EclipseError::Geometry);
    }
    let (position, velocity, sun) = state_at(t)?;
    let geometry = binary_cylinder_geometry(position, sun, earth_radius_km)?;
    Ok(ScanEndpoint {
        t,
        position,
        velocity,
        side: geometry.side,
        boundary_margin_km: geometry.boundary_margin_km,
    })
}

fn scan_interval<F, B>(
    old: ScanEndpoint,
    new: ScanEndpoint,
    inherited_motion_bound: Option<f64>,
    earth_radius_km: f64,
    max_separation_km: f64,
    splits_left: &mut usize,
    state_at: &mut F,
    relative_motion_bound: &mut B,
) -> Result<EclipseScanResult, EclipseError>
where
    F: FnMut(f64) -> Result<([f64; 3], [f64; 3], [f64; 3]), EclipseError>,
    B: FnMut(ScanEndpointState, ScanEndpointState) -> Result<f64, EclipseError>,
{
    let separation = separation_km(old.position, new.position)?;
    if old.side == new.side
        && inherited_motion_bound.is_some_and(|bound| {
            no_crossing_certified(old.boundary_margin_km, new.boundary_margin_km, bound)
        })
    {
        return Ok(EclipseScanResult::default());
    }
    let motion_bound = relative_motion_bound(old.state(), new.state())?;
    if !(motion_bound.is_finite() && motion_bound >= separation && motion_bound >= 0.0) {
        return Err(EclipseError::Bracket);
    }
    if old.side == new.side
        && no_crossing_certified(old.boundary_margin_km, new.boundary_margin_km, motion_bound)
    {
        return Ok(EclipseScanResult::default());
    }
    if motion_bound <= max_separation_km && old.side != new.side {
        return {
            Ok(EclipseScanResult {
                crossing: Some(EclipseBracket {
                    accepted_t_old: old.t,
                    accepted_t_new: new.t,
                    accepted_eci_old: [0.0; 6],
                    t_old: old.t,
                    t_new: new.t,
                    old_side: old.side,
                    new_side: new.side,
                    geometry_motion_bound_km: motion_bound,
                }),
                collapsed_pairs: 0,
            })
        };
    }
    let midpoint_t = binary64_midpoint(old.t, new.t);
    if is_zero_span(midpoint_t, old.t) || is_zero_span(midpoint_t, new.t) || !midpoint_t.is_finite()
    {
        // Adjacent binary64 instants contain no representable evaluation time.
        // A certified same-side pair contributes at most one millimetre of
        // unresolved geometry motion, so no boundary can be localized inside
        // it. Record that numerical collapse and continue on the unchanged
        // force side. Larger unresolved intervals remain fatal.
        if old.side == new.side && motion_bound <= MAX_BOUNDARY_SEPARATION_KM {
            return Ok(EclipseScanResult {
                crossing: None,
                collapsed_pairs: 1,
            });
        }
        return Err(EclipseError::NonProgress);
    }
    if *splits_left == 0 {
        return Err(EclipseError::SplitLimit);
    }
    *splits_left = (*splits_left)
        .checked_sub(1)
        .ok_or(EclipseError::SplitLimit)?;
    let midpoint = endpoint(midpoint_t, earth_radius_km, state_at)?;
    let mut first = scan_interval(
        old,
        midpoint,
        Some(motion_bound),
        earth_radius_km,
        max_separation_km,
        splits_left,
        state_at,
        relative_motion_bound,
    )?;
    if first.crossing.is_some() {
        return Ok(first);
    }
    let second = scan_interval(
        midpoint,
        new,
        Some(motion_bound),
        earth_radius_km,
        max_separation_km,
        splits_left,
        state_at,
        relative_motion_bound,
    )?;
    first.collapsed_pairs = first
        .collapsed_pairs
        .checked_add(second.collapsed_pairs)
        .ok_or(EclipseError::SplitLimit)?;
    first.crossing = second.crossing;
    Ok(first)
}

pub fn first_crossing_in_step<F, B>(
    t_old: f64,
    t_new: f64,
    earth_radius_km: f64,
    max_separation_km: f64,
    max_splits: usize,
    mut state_at: F,
    mut relative_motion_bound: B,
) -> Result<EclipseScanResult, EclipseError>
where
    F: FnMut(f64) -> Result<([f64; 3], [f64; 3], [f64; 3]), EclipseError>,
    B: FnMut(ScanEndpointState, ScanEndpointState) -> Result<f64, EclipseError>,
{
    if !(max_separation_km.is_finite() && max_separation_km > 0.0) || is_zero_span(t_old, t_new) {
        return Err(EclipseError::NonProgress);
    }
    let old = endpoint(t_old, earth_radius_km, &mut state_at)?;
    let new = endpoint(t_new, earth_radius_km, &mut state_at)?;
    let mut splits_left = max_splits;
    scan_interval(
        old,
        new,
        None,
        earth_radius_km,
        max_separation_km,
        &mut splits_left,
        &mut state_at,
        &mut relative_motion_bound,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        binary64_midpoint, binary_cylinder_geometry, classify_binary_cylinder,
        first_crossing_in_step, no_crossing_certified, validate_part_a_eclipse_envelope,
        EclipseError, EclipseSide, ScanEndpointState, MAX_BOUNDARY_SEPARATION_KM,
    };

    const EARTH_RADIUS_KM: f64 = 6378.137;
    const SUN: [f64; 3] = [149_597_870.7, 0.0, 0.0];

    #[test]
    fn compiled_shadow_authority_is_binary_cylinder_only() {
        let retired_module = ["pub mod shadow", "_probe;"].concat();
        assert!(
            !include_str!("lib.rs").contains(&retired_module),
            "retired continuous-shadow probe remains in compiled crate surface"
        );
        assert_eq!(
            classify_binary_cylinder([-7000.0, 0.0, 0.0], SUN, EARTH_RADIUS_KM),
            Ok(EclipseSide::Shadow)
        );
        assert_eq!(
            classify_binary_cylinder([7000.0, 0.0, 0.0], SUN, EARTH_RADIUS_KM),
            Ok(EclipseSide::Lit)
        );
        assert_eq!(
            classify_binary_cylinder([f64::NAN, 0.0, 0.0], SUN, EARTH_RADIUS_KM),
            Err(EclipseError::Geometry)
        );
    }

    /// The no-crossing certificate spends one motion budget across the whole
    /// interval, not one budget per endpoint.
    ///
    /// The lopsided case is the one that matters and the one a `min`-based test
    /// gets wrong: an interval whose inner endpoint sits close to the boundary
    /// is exactly the interval abutting a bracket, and refusing to certify it
    /// is what forces the scan to bisect toward a root it has already found.
    /// Pinning the asymmetric case keeps a future `min` from reading as an
    /// equivalent tidy-up.
    ///
    /// Soundness is not asserted here — it is carried by the scan-level tests,
    /// above all `moving_sun_axis_reveals_hidden_crossings_for_stationary_satellite`,
    /// which fails if this certificate ever prunes an interval that holds a
    /// crossing.
    #[test]
    fn no_crossing_certificate_spends_one_budget_across_the_interval() {
        // Lopsided: neither endpoint alone out-runs the motion bound, but
        // together they cover it, so no interior instant can reach the boundary.
        assert!(no_crossing_certified(3.0, 2.0, 4.0));
        // The symmetric predicate this replaced would refuse the same interval.
        assert!(3.0_f64.min(2.0) <= 4.0);
        // Genuinely unresolved: the budget covers both margins at once.
        assert!(!no_crossing_certified(1.5, 2.0, 4.0));
        // A zero margin means an endpoint is on the boundary; the far endpoint
        // must then cover the whole budget by itself.
        assert!(no_crossing_certified(5.0, 0.0, 4.0));
        assert!(!no_crossing_certified(4.0, 0.0, 4.0));
    }

    // The bound closures below are analytic in time, as the reference bounds
    // these tests certify against are. They take the endpoint states the scan
    // supplies and read only the instants out of them.
    fn double_crossing_state(t: f64) -> Result<([f64; 3], [f64; 3], [f64; 3]), EclipseError> {
        if !t.is_finite() {
            return Err(EclipseError::Geometry);
        }
        let offset = 8.0 * (t - 0.5) * (t - 0.5) - 1.0;
        Ok((
            [-7000.0, EARTH_RADIUS_KM + offset, 0.0],
            [0.0, 16.0 * (t - 0.5), 0.0],
            SUN,
        ))
    }

    fn padded_upper_bound(value: f64) -> Result<f64, EclipseError> {
        let padded_bits = value
            .to_bits()
            .checked_add(8)
            .ok_or(EclipseError::Geometry)?;
        Ok(f64::from_bits(padded_bits) + 1.0e-9)
    }

    fn moving_axis_path_bound(radius: f64, t0: f64, t1: f64) -> Result<f64, EclipseError> {
        if !t0.is_finite() || !t1.is_finite() {
            return Err(EclipseError::Geometry);
        }
        let angular_speed = (0.004 * (1.0 - 2.0 * t0))
            .abs()
            .max((0.004 * (1.0 - 2.0 * t1)).abs());
        let value = 2.0 * radius * angular_speed * (t1 - t0).abs();
        padded_upper_bound(value)
    }

    #[test]
    fn cylinder_boundaries_and_sunward_states_are_lit() {
        assert_eq!(
            classify_binary_cylinder([-7000.0, 0.0, 0.0], SUN, EARTH_RADIUS_KM),
            Ok(EclipseSide::Shadow)
        );
        for position in [
            [-7000.0, EARTH_RADIUS_KM, 0.0],
            [0.0, 0.0, 0.0],
            [-0.0, 0.0, 0.0],
            [7000.0, 0.0, 0.0],
        ] {
            assert_eq!(
                classify_binary_cylinder(position, SUN, EARTH_RADIUS_KM),
                Ok(EclipseSide::Lit)
            );
        }
    }

    #[test]
    fn sunward_outside_cylinder_margin_ignores_harmless_terminator_crossing() {
        let outside = EARTH_RADIUS_KM + 100.0;
        let behind = binary_cylinder_geometry([-1.0e-12, outside, 0.0], SUN, EARTH_RADIUS_KM)
            .expect("behind outside geometry");
        let sunward = binary_cylinder_geometry([1.0e-12, outside, 0.0], SUN, EARTH_RADIUS_KM)
            .expect("sunward outside geometry");
        assert!(behind.boundary_margin_km > 99.0);
        assert!(sunward.boundary_margin_km > 99.0);

        let inside = binary_cylinder_geometry([1.0e-12, 0.0, 0.0], SUN, EARTH_RADIUS_KM)
            .expect("sunward inside geometry");
        assert_eq!(inside.side, EclipseSide::Lit);
        assert_eq!(inside.boundary_margin_km.to_bits(), 1.0e-12_f64.to_bits());
    }

    #[test]
    fn invalid_geometry_fails_closed() {
        for sun in [
            [0.0, 0.0, 0.0],
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 0.0, 0.0],
        ] {
            assert_eq!(
                classify_binary_cylinder([-7000.0, 0.0, 0.0], sun, EARTH_RADIUS_KM),
                Err(EclipseError::Geometry)
            );
        }
        assert_eq!(
            classify_binary_cylinder([f64::NAN, 0.0, 0.0], SUN, EARTH_RADIUS_KM),
            Err(EclipseError::Geometry)
        );
        assert_eq!(
            classify_binary_cylinder([-7000.0, 0.0, 0.0], SUN, 0.0),
            Err(EclipseError::Geometry)
        );
    }

    #[test]
    fn part_a_eclipse_envelope_rejects_states_outside_campaign_headroom() {
        assert!(validate_part_a_eclipse_envelope([7_000.0, 0.0, 0.0], [0.0, 19.9, 0.0]).is_ok());
        assert_eq!(
            validate_part_a_eclipse_envelope([7_000.0, 0.0, 0.0], [0.0, 20.1, 0.0]),
            Err(EclipseError::Envelope)
        );
        assert_eq!(
            validate_part_a_eclipse_envelope([5_999.0, 0.0, 0.0], [0.0, 7.5, 0.0]),
            Err(EclipseError::Envelope)
        );
        assert_eq!(
            validate_part_a_eclipse_envelope([50_001.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            Err(EclipseError::Envelope)
        );
    }

    fn double_crossing_path_bound(
        old: ScanEndpointState,
        new: ScanEndpointState,
    ) -> Result<f64, EclipseError> {
        let (t0, t1) = (old.t, new.t);
        if !t0.is_finite() || !t1.is_finite() {
            return Err(EclipseError::Geometry);
        }
        let max_speed = 16.0 * (t0 - 0.5).abs().max((t1 - 0.5).abs());
        let bound = max_speed * (t1 - t0).abs();
        padded_upper_bound(bound)
    }

    #[test]
    fn same_side_step_finds_earliest_forward_crossing() {
        let hit = first_crossing_in_step(
            0.0,
            1.0,
            EARTH_RADIUS_KM,
            MAX_BOUNDARY_SEPARATION_KM,
            1_000_000,
            double_crossing_state,
            double_crossing_path_bound,
        )
        .expect("finite scan")
        .crossing
        .expect("two hidden crossings");
        assert_eq!(hit.old_side, EclipseSide::Lit);
        assert_eq!(hit.new_side, EclipseSide::Shadow);
        assert!(hit.t_old < hit.t_new);
        assert!(hit.geometry_motion_bound_km <= MAX_BOUNDARY_SEPARATION_KM);
        assert!((binary64_midpoint(hit.t_old, hit.t_new) - 0.146_446_609_4).abs() < 1e-5);
        let second = first_crossing_in_step(
            hit.t_new,
            1.0,
            EARTH_RADIUS_KM,
            MAX_BOUNDARY_SEPARATION_KM,
            1_000_000,
            double_crossing_state,
            double_crossing_path_bound,
        )
        .expect("finite second scan")
        .crossing
        .expect("second hidden crossing");
        assert_eq!(
            (second.old_side, second.new_side),
            (EclipseSide::Shadow, EclipseSide::Lit)
        );
        assert!(hit.t_new < second.t_old);
    }

    #[test]
    fn same_side_step_finds_earliest_backward_crossing() {
        let hit = first_crossing_in_step(
            1.0,
            0.0,
            EARTH_RADIUS_KM,
            MAX_BOUNDARY_SEPARATION_KM,
            1_000_000,
            double_crossing_state,
            double_crossing_path_bound,
        )
        .expect("finite scan")
        .crossing
        .expect("two hidden crossings");
        assert_eq!(hit.old_side, EclipseSide::Lit);
        assert_eq!(hit.new_side, EclipseSide::Shadow);
        assert!(hit.t_old > hit.t_new);
        assert!(hit.geometry_motion_bound_km <= MAX_BOUNDARY_SEPARATION_KM);
        assert!((binary64_midpoint(hit.t_old, hit.t_new) - 0.853_553_390_6).abs() < 1e-5);
        let second = first_crossing_in_step(
            hit.t_new,
            0.0,
            EARTH_RADIUS_KM,
            MAX_BOUNDARY_SEPARATION_KM,
            1_000_000,
            double_crossing_state,
            double_crossing_path_bound,
        )
        .expect("finite second backward scan")
        .crossing
        .expect("second hidden backward crossing");
        assert_eq!(
            (second.old_side, second.new_side),
            (EclipseSide::Shadow, EclipseSide::Lit)
        );
        assert!(hit.t_new > second.t_old);
    }

    #[test]
    fn scan_limit_is_typed_failure() {
        assert_eq!(
            first_crossing_in_step(
                0.0,
                1.0,
                EARTH_RADIUS_KM,
                MAX_BOUNDARY_SEPARATION_KM,
                1,
                double_crossing_state,
                double_crossing_path_bound,
            ),
            Err(EclipseError::SplitLimit)
        );
    }

    #[test]
    fn adjacent_certified_same_side_pair_is_counted_forward_and_backward() {
        let t0 = 1.0_f64;
        let t1 = f64::from_bits(t0.to_bits() + 1);
        let state = |_| Ok(([-7000.0, EARTH_RADIUS_KM, 0.0], [0.0; 3], SUN));
        let small_bound =
            |_: ScanEndpointState, _: ScanEndpointState| Ok(0.5 * MAX_BOUNDARY_SEPARATION_KM);
        for (old, new) in [(t0, t1), (t1, t0)] {
            let outcome = first_crossing_in_step(
                old,
                new,
                EARTH_RADIUS_KM,
                MAX_BOUNDARY_SEPARATION_KM,
                0,
                state,
                small_bound,
            )
            .expect("certified adjacent same-side interval is resolved");
            assert!(outcome.crossing.is_none());
            assert_eq!(outcome.collapsed_pairs, 1);
        }

        assert_eq!(
            first_crossing_in_step(
                t0,
                t1,
                EARTH_RADIUS_KM,
                MAX_BOUNDARY_SEPARATION_KM,
                8,
                state,
                |_, _| Ok(2.0 * MAX_BOUNDARY_SEPARATION_KM),
            ),
            Err(EclipseError::NonProgress)
        );
    }

    #[test]
    fn nonadjacent_same_side_interval_still_requires_split_budget() {
        let t0 = 1.0_f64;
        let t2 = f64::from_bits(
            t0.to_bits()
                .checked_add(2)
                .expect("two successor instants exist"),
        );
        assert_eq!(
            first_crossing_in_step(
                t0,
                t2,
                EARTH_RADIUS_KM,
                MAX_BOUNDARY_SEPARATION_KM,
                0,
                |_| Ok(([-7000.0, EARTH_RADIUS_KM, 0.0], [0.0; 3], SUN)),
                |_, _| Ok(0.5 * MAX_BOUNDARY_SEPARATION_KM),
            ),
            Err(EclipseError::SplitLimit)
        );
    }

    #[test]
    fn moving_sun_axis_reveals_hidden_crossings_for_stationary_satellite() {
        let satellite = [-7000.0, EARTH_RADIUS_KM + 1.0, 0.0];
        let radius = satellite
            .iter()
            .map(|component| component * component)
            .sum::<f64>()
            .sqrt();
        let state = |t: f64| {
            let theta = -0.004 * t * (1.0 - t);
            Ok((
                satellite,
                [0.0; 3],
                [SUN[0] * theta.cos(), SUN[0] * theta.sin(), 0.0],
            ))
        };
        let bound = |old: ScanEndpointState, new: ScanEndpointState| {
            moving_axis_path_bound(radius, old.t, new.t)
        };
        let hit = first_crossing_in_step(
            0.0,
            1.0,
            EARTH_RADIUS_KM,
            MAX_BOUNDARY_SEPARATION_KM,
            1_000_000,
            state,
            bound,
        )
        .expect("certified moving-axis scan")
        .crossing
        .expect("Sun-axis motion creates two hidden crossings");
        assert_eq!(
            (hit.old_side, hit.new_side),
            (EclipseSide::Lit, EclipseSide::Shadow)
        );
        assert!(binary64_midpoint(hit.t_old, hit.t_new) < 0.5);
        assert!(hit.geometry_motion_bound_km <= MAX_BOUNDARY_SEPARATION_KM);
    }
}
