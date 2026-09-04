//! Types for the Lightyear ODE integrator
//!
//! Mirrors C++ `lightyear_odeint.hpp` types.

/// State type for the delta-state integration [dx, dy, dz, dvx, dvy, dvz]
pub(crate) type StateType = [f64; 6];

/// Earth gravitational parameter, in km^3/s^2, and the WGS84 equatorial
/// radius, in km. Re-exported from `satpy_core`, not restated: both were
/// declared here a second time with the same literals, which is two sources of
/// truth for two of the most load-bearing numbers in the tree and a divergence
/// waiting to happen. This crate already sources `WGS84_FLATTENING` and
/// `SEC_PER_DAY` from `satpy_core` (`rhs.rs:38`, `rhs.rs:8`); these two were
/// the inconsistency.
///
/// `RE` is used here only for geometry: the ground-strike event in `events.rs`
/// and the perigee guard in `session.rs`. It is NOT the gravity reference
/// radius; see [`ForceConfig::earth_radius`] for the full statement of the
/// distinction, and `satpy_core::GRAVITY_REFERENCE_RADIUS_KM` for the value
/// that answers the other question.
pub use satpy_core::{MU, RE};
/// Encke rectification threshold, in km: when the deviation `|dr|` from the
/// osculating reference exceeds this, the reference is re-anchored to the
/// current state.
///
/// Encke integrates the DEVIATION from a Keplerian reference and neglects terms
/// quadratic in `d/r`. Rectifying more often costs a reference re-anchor; less
/// often grows that neglected term as `(d/r)^2`.
///
/// This was 2.0 km, justified analytically: at LEO (`r ~ 7000 km`) a 2 km
/// deviation is `d/r ~ 2.9e-4`, so the neglected quadratic term is ~1e-7
/// relative. Raising it to 10 km makes `d/r ~ 1.4e-3` and grows that analytic
/// term by 25x.
///
/// The basis for 10.0 is measurement, not that bound: the bound is loose by
/// orders of magnitude in both directions and decides nothing.
///
/// Measured on `tests/tolerance_cost_accuracy.rs`, which exists for exactly this
/// diff (`TOL_PIN` endpoints, rebuilt at each threshold), 2.0 km -> 10.0 km:
///
/// ```text
/// evaluations @ eps 1e-8            85,051 -> 80,503   (-5.4%)
/// Encke segments @ eps 1e-8            765 ->    308   (-59.7%)
/// endpoint move, 7200 s arcs      0.00005 -- 0.006 m
/// endpoint move, 43200 s arcs     0.005   -- 0.587 m
/// endpoint move, 111874 s arcs    0.110   -- 0.191 m
/// ```
///
/// **The endpoint move grows with arc length and its worst case is 0.587 m, not
/// the ~0.0035 m that was circulated when this was approved.** Stated here
/// because the discrepancy is 168x and the next person to read this comment
/// should not re-derive it from a number nobody measured.
///
/// One honest limit on the above: those endpoints are taken at `eps = 1e-9`,
/// which is NOT converged enough to resolve 0.587 m — that same run reports a
/// worst-arc error of 0.20 m (at 2.0 km) and 0.77 m (at 10.0 km) against its own
/// reference. So 0.587 m is an upper bound contaminated by tolerance error, not
/// a clean Encke-only figure. The one arc with a proper converged reference is
/// `tests/strict_hf_pin.rs` (`alt800`, 43200 s, eps 1e-8 vs 1e-12): its
/// truncation error moves 0.323 m -> 0.328 m, i.e. essentially not at all.
///
/// Cost side, on the same rebuild: the saving SHRINKS with arc duration, because
/// `MAX_RECT_SEGMENT = 5400 s` in `integrator.rs` forces restarts that no
/// threshold can remove.
///
/// # 2026-08-10: the ladder above 10 km was walked under Vern7, and 10.0 STAYS
///
/// Measurement basis: commit 68ce8b8, the VERN7 tree — the one this constant
/// would land on. An earlier version of this table was walked on Vern9
/// (bce8eaf) and is preserved below only as the superseded era, because a
/// ladder priced under one stepper does not transfer across a method change:
/// the step sequence sets the segment and rebase counts the whole measurement
/// rests on. Any future reopen must re-walk on the tree it would land on.
///
/// Each rung was a full rebuild, measured on the retired 4x2 production census
/// (`prop-census` + `ND_MASS_ROW_DUMP=1`, 3,388 mass rows; historical record in
/// `docs/PART_A_RESULTS_MATRIX.md`) and on the V3 arc in
/// `tests/strict_hf_pin.rs`. The census reproduced all 3,388 rows and the eval
/// count to the unit on that dated tree, so every displacement below was signal,
/// not run-to-run noise. The retained one-event zero-bad-return owner is
/// `nd_pipeline/tests/native_hybrid_evaluator.rs`.
///
/// ```text
/// km   census RHS evals    d      rows  moved  worst row   bad props  endpoint
/// 10    210,106,750      ----     3,388   --       --          14        --
/// 20    208,520,164     -0.76%    3,386  60.5%  1.000x Brent    16     0.234 m
/// 40    209,165,165     -0.45%    3,387  60.3%  1.000x Brent    18     0.076 m
/// 160   205,441,567     -2.22%    3,388  60.1%  0.038x Brent    21     0.096 m
/// ```
///
/// **The prize is gone, because Vern7 already took it.** This lever and the
/// stepper swap were competing for the same cost — the restart ramp — and the
/// swap landed first: at an unchanged 10 km threshold the same census fell
/// 248,194,336 -> 210,106,750 evaluations, i.e. -15.35%. What is left for the
/// threshold is 0.45% to 2.22% and it is NOT MONOTONE (40 km is worse than
/// 20 km), which is the signature of step-sequence dice rather than of a
/// saving. Under Vern9 this same ladder read -3.51% / -5.79% / -7.45% / -8.46%;
/// quoting those numbers now would be quoting a superseded tree.
///
/// **The analytic `(d/r)^2` budget predicts none of it either.** Across the
/// ladder that term grows 4x / 16x / 256x while the measured accuracy channels
/// stay flat and non-monotone: both 1 m gates read 0.133 / 0.176 / 0.008 m
/// (model 4) and 0.134 / 0.174 / 0.006 m (model 6), best at the LOOSEST rung. A
/// quantity that does not move over a 256x range of the bound is not governed by
/// that bound; the mover is step-sequence chaos, and it saturates at rung one.
///
/// **The damage, unlike the saving, WAS monotone on that tree.** Bad propagation
/// returns climbed 14 -> 16 -> 18 -> 21, breaching the then-local 14-return
/// ratchet at every rung. That host-conditioned 4x2 ratchet was retired with its
/// measurement harness. One or two of the 3,388 mass rows stopped being solved
/// at every rung — solvability-boundary rows a last-ULP reshuffle flips out — and
/// at 20 and 40 km the worst surviving row moves by exactly `0.5 * xtol` =
/// 5.0e-7 kg, a FULL Brent indifference interval, against precedents in this lane
/// that moved 0.336x and 0.37x of it while touching ~1.6% of rows rather than 60%.
///
/// So on the Vern7 tree the lever fails four of the five land bars: it buys under
/// 3%, it loses rows, it puts two rows outside Brent's interval, and at 20 km the
/// V3 endpoint moves 0.234 m, above that metric's own 0.2 m noise floor. Only the
/// two 1 m gates still pass.
///
/// **The bit-pin battery is a poor detector here, and that survived the swap.**
/// On the Vern9 tree all six release pins ran GREEN at 20 km with no re-pin: the
/// `rect_loop_pin` arc is floored at 8 segments by `MAX_RECT_SEGMENT` so no
/// deviation restart occurs on it at any threshold, and the V3 endpoint moved
/// only 0.0088 m. Under Vern7 the V3 tripwire DOES trip (0.234 m at 20 km), so
/// half the blindness is gone; the three rect-loop digests still cannot see this
/// constant at all, because the mechanism that blinds them — their arc never
/// exercising a deviation restart — is a property of the arc and not of the
/// stepper. The safe reading is one-directional either way: a green pin run is
/// not evidence that this constant may be raised; only the mass corpus can
/// answer that.
///
/// Superseded Vern9 walk (bce8eaf), kept because it is what the refusal was
/// first argued on and the two trees disagree about the size of the prize:
/// -3.51% / -5.79% / -7.45% / -8.46% at 20 / 40 / 80 / 160 km, three of 3,390
/// rows lost at every rung, 64% of rows moved, and a bad-propagation breach of
/// 29 against the then-current ratchet of 23.
pub const PERTURB_DEVIATION_THRESHOLD_KM: f64 = 10.0;

pub const GROUND_ALTITUDE: f64 = 100.0; // km above surface

/// Explicit adaptive integrator authority carried with every force context.
///
/// `Auto` remains available to legacy callers, but production transfer replay
/// supplies a concrete method so tolerance changes cannot silently switch
/// algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepperMethod {
    Dopri5Compat,
    Tsit5,
    Dop853,
    Rkv98,
    Vern7,
    Vern9,
    Esdirk43,
    Auto,
}

/// Default maximum ODE step size (seconds).
///
/// This is the LIBRARY default and is not what the campaign flies: production
/// reads `dt_max_s` off the sealed Part A authority
/// (`nd_config::CompiledPartAScienceV1::part_a_v1().hybrid()`), which is 300 s.
///
/// # Raising the production cap above 300 s is REFUTED (2026-08-09, reconfirmed
/// # under Vern7 2026-08-10)
///
/// Measurement basis: commit bce8eaf, the Vern9-era tree (the 7,875-eval V3
/// arc below is the tell; the Vern7 tree pins a different count). The null
/// has NOT been re-measured under Vern7. The observed mechanism — leg
/// boundaries from eclipse roots and `MAX_RECTIFICATION_SEGMENT_S` binding
/// before the cap — does not obviously involve the integrator, but that is an
/// argument, not a measurement; re-run the ladder before relying on the null
/// on any other tree.
///
/// Measured on the 4x2 production census, one full rebuild per rung, by moving
/// the sealed `dt_max_s` and restoring it. The first three rows are the Vern9
/// tree the ladder was originally walked on; the last is the Vern7 recheck:
///
/// ```text
/// dt_max   census RHS evals      d       sat_frac   mean step   V3 arc
///   60 s     415,650,408     +67.47%      0.887      51.6 s     13,567 evals
///  300 s     248,194,336      ----        0.264      88.4 s      7,875 evals
///  600 s     248,315,798      +0.05%      0.264      88.4 s     BIT IDENTICAL
/// 1200 s     248,286,702      +0.04%      0.264      88.4 s     BIT IDENTICAL
///  600 s     210,129,590      +0.01%      0.110      -----      BIT IDENTICAL  (Vern7)
/// ```
///
/// The 60 s rung is the non-vacuity control and it proves the knob is wired:
/// lowering the cap costs 67% more evaluations and moves the V3 arc. Raising it
/// buys NOTHING — 4x the cap is +0.04%, i.e. the wrong sign inside noise, and the
/// V3 arc comes back bit-for-bit identical at both 600 and 1200 s.
///
/// Vern7 makes the null STRONGER, not weaker: it takes smaller steps (667 against
/// 474 on the V3 arc), so saturation against the cap falls from 0.264 to 0.110 and
/// there is even less for a bigger cap to release.
///
/// The mechanism is in `sat_frac` and `mean step`, neither of which moves: above
/// 300 s the binding constraint is not the cap but the LEG, since the
/// event-enabled path re-enters the solver at every eclipse root transaction and
/// at every `MAX_RECTIFICATION_SEGMENT_S` boundary. `saturated_steps` counts
/// controller DEMAND against the cap, not the step actually taken, so a
/// saturation fraction of 0.264 that is unchanged by doubling the cap means those
/// steps were being truncated by a leg boundary all along.
pub(crate) const DEFAULT_DT_MAX_S: f64 = 60.0;

/// Event types that can terminate or trigger restart of integration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EventType {
    PerturbDeviation = 0, // Triggers restart, not termination
    Ground = 1,           // Terminal - hit ground
    LeftEarth = 2,        // Terminal - escaped
    NanState = 3,         // Terminal - numerical failure
    Eccentricity = 4,     // Terminal - hyperbolic orbit
}

impl EventType {
    pub(crate) const NUM_EVENTS: usize = 5;

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::PerturbDeviation => "perturb_deviation",
            Self::Ground => "ground",
            Self::LeftEarth => "left_earth",
            Self::NanState => "nan_state",
            Self::Eccentricity => "eccentricity",
        }
    }
}

/// Interpolation method used for event refinement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpMethod {
    #[default]
    None,
    Hermite,
    Linear,
}

impl InterpMethod {
    /// Get the string representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Hermite => "hermite",
            Self::Linear => "linear",
        }
    }

    /// Parse from a string.
    #[expect(clippy::should_implement_trait)]
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "hermite" => Self::Hermite,
            "linear" | "linear_clamp" => Self::Linear,
            _ => Self::None,
        }
    }
}

/// Force flags for enabling dust forces
pub struct ForceFlags;

impl ForceFlags {
    pub const DRAG: i32 = 1;
    pub const SRP: i32 = 2;
    pub const SUN_GRAVITY: i32 = 4;
    pub const MOON_GRAVITY: i32 = 8;
    // Extended third-body flags (precomputed ephemeris integration)
    pub(crate) const JUPITER_GRAVITY: i32 = 16;
    pub(crate) const VENUS_GRAVITY: i32 = 32;
    pub(crate) const MARS_GRAVITY: i32 = 64;
    pub(crate) const SATURN_GRAVITY: i32 = 128;
    // Electromagnetic / relativistic terms (opt-in only).
    pub(crate) const LORENTZ: i32 = 256;
    pub(crate) const COULOMB_DRAG: i32 = 512;
    pub(crate) const RELATIVITY: i32 = 1024;
    // Combined flags
    pub(crate) const THIRDBODY_ALL: i32 = 4 + 8 + 16 + 32 + 64 + 128;
    // Keep the "all" force set stable (drag + srp + third-body gravity).
    pub const ALL: i32 = 255;
}

/// Precomputed invariants for third-body gravity calculations.
/// These values depend only on body position and mu, not satellite position,
/// so they can be computed once per integration segment.
#[derive(Clone, Copy, Debug, Default)]
pub struct BodyInvariants {
    /// Unit vector from Earth to body: `body_pos / |body_pos|`.
    pub body_norm: [f64; 3],
    /// Inverse body distance: `1 / |body_pos|`.
    pub inv_body_dist: f64,
    /// Precomputed coefficient: `mu_body / |body_pos|^2`.
    pub mu_coef: f64,
}

impl BodyInvariants {
    /// Precompute invariants for a celestial body.
    /// Returns `None` if `body_pos` is zero (invalid).
    #[inline]
    #[must_use]
    pub fn precompute(body_pos: &[f64; 3], mu_body: f64) -> Option<Self> {
        let &[body_x, body_y, body_z] = body_pos;
        let body_dist_sq = body_x * body_x + body_y * body_y + body_z * body_z;

        if body_dist_sq == 0.0 {
            return None;
        }

        let body_dist = body_dist_sq.sqrt();
        let inv_body_dist = 1.0 / body_dist;
        let inv_body_dist_sq = inv_body_dist * inv_body_dist;

        Some(Self {
            body_norm: [
                body_x * inv_body_dist,
                body_y * inv_body_dist,
                body_z * inv_body_dist,
            ],
            inv_body_dist,
            mu_coef: mu_body * inv_body_dist_sq,
        })
    }
}

/// Configuration for perturbation forces during integration
#[derive(Clone, Copy, Debug)]
pub struct ForceConfig {
    pub sph_order: usize,

    pub force_flags: i32,
    pub subtract_first_order: bool,
    pub atm_model: i32,
    pub am_ratio: f64,
    pub cd: f64,
    pub cr: f64,
    /// Transfer-solver catalogue target propagation authority:
    /// 0=HF legacy, 1=MF/J2, 2=analytical/Kepler.
    pub target_propagation_mode: u8,
    pub qm_ratio: f64,
    pub r_obj_m: f64,
    pub omega_earth: f64,
    pub p_sun: f64,

    // Gravitational parameters (km^3/s^2) - DE431 values
    pub mu_sun: f64,
    pub mu_moon: f64,
    pub mu_jupiter: f64,
    pub mu_venus: f64,
    pub mu_mars: f64,
    pub mu_saturn: f64,

    /// Earth radius used ONLY for geometry that reduces a position to an
    /// altitude or an occultation: drag altitude (`r - earth_radius`), the
    /// geocentric lat/lon/alt reduction, cylindrical binary eclipse, and the
    /// ground-impact guard (`earth_radius + GROUND_ALTITUDE`). Defaults to the
    /// WGS84 equatorial radius 6378.137 km.
    ///
    /// This is NOT the gravity reference radius. The spherical-harmonic
    /// potential is referenced to `satpy_core`'s DIR-R6 constant
    /// `GRAVITY_REFERENCE_RADIUS_KM` = 6378.13646 km, which the gravity kernels
    /// bind internally and which no `ForceConfig` field can override — the two
    /// differ by 0.00054 km (54 cm) and are deliberately separate. Setting
    /// `earth_radius` does not move the gravity field.
    pub earth_radius: f64,

    // Body positions (supplied by the caller or the precomputed binary catalogues)
    pub sun_pos: Option<[f64; 3]>,
    pub moon_pos: Option<[f64; 3]>,
    pub jupiter_pos: Option<[f64; 3]>,
    pub venus_pos: Option<[f64; 3]>,
    pub mars_pos: Option<[f64; 3]>,
    pub saturn_pos: Option<[f64; 3]>,

    /// Body-force bits whose positions came from the native ephemeris rather
    /// than an explicit caller override. The RHS must refresh these bodies at
    /// its current absolute JD instead of freezing the segment midpoint.
    pub dynamic_ephemeris_flags: i32,

    // Precomputed third-body invariants (computed once per segment, not per RHS evaluation)
    pub sun_invariants: Option<BodyInvariants>,
    pub moon_invariants: Option<BodyInvariants>,
    pub jupiter_invariants: Option<BodyInvariants>,
    pub venus_invariants: Option<BodyInvariants>,
    pub mars_invariants: Option<BodyInvariants>,
    pub saturn_invariants: Option<BodyInvariants>,

    /// Maximum integration step size (seconds). Default is
    /// [`DEFAULT_DT_MAX_S`] = 60.0.
    pub dt_max: f64,
    /// Integration tolerance. Default is 1e-8 (see `ForceConfig::default`);
    /// this doc previously said 1e-5, which was never the default here.
    pub eps: f64,
    /// Authoritative adaptive integration method.
    pub integrator_method: StepperMethod,
}

/// Per-integration performance counters.
///
/// Plain fields, deliberately. These used to be `AtomicUsize`/`AtomicU64`
/// under a comment claiming "lock-free concurrent updates" -- but nothing ever
/// updated them concurrently, or indeed at all. The integrator accumulates its
/// totals in ordinary `usize` locals and publishes them once through
/// [`OdeMetrics::from_values`]; the three `add_*` mutators that justified the
/// atomics had no callers anywhere in the workspace. What the atomics actually
/// bought was a hand-written `Clone`, `Relaxed` loads at every read site, and
/// a false claim about the concurrency model.
#[derive(Debug, Default, Clone, Copy)]
pub struct OdeMetrics {
    pub total_steps: usize,
    pub total_evals: usize,
    pub total_time_us: u64,
    /// Adjacent binary64, same-side eclipse pairs accepted only after their
    /// certified total geometry motion was bounded to one millimetre.
    pub eclipse_collapsed_pairs: usize,
}

impl OdeMetrics {
    /// Create new metrics with all counters at zero
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create new metrics with specific values
    #[must_use]
    pub(crate) const fn from_values(
        total_steps: usize,
        total_evals: usize,
        total_time_us: u64,
    ) -> Self {
        Self {
            total_steps,
            total_evals,
            total_time_us,
            eclipse_collapsed_pairs: 0,
        }
    }
}

impl Default for ForceConfig {
    fn default() -> Self {
        Self {
            sph_order: 7,
            force_flags: 0,
            subtract_first_order: false,
            atm_model: 0,
            am_ratio: 0.0,
            cd: 0.0,
            cr: 0.0,
            target_propagation_mode: 0,
            qm_ratio: 0.0,
            r_obj_m: 0.0,
            omega_earth: 7.292_115_0e-5,
            p_sun: 4.56e-6,
            // Gravitational parameters (km^3/s^2) - DE431 values
            mu_sun: 1.327_124_400_18e11,
            mu_moon: 4_902.800_066,
            mu_jupiter: 1.266_865_34e8,
            mu_venus: 3.248_585_92e5,
            mu_mars: 4.282_837_5e4,
            mu_saturn: 3.793_120_6e7,
            earth_radius: 6378.137,
            sun_pos: None,
            moon_pos: None,
            jupiter_pos: None,
            venus_pos: None,
            mars_pos: None,
            saturn_pos: None,
            dynamic_ephemeris_flags: 0,
            sun_invariants: None,
            moon_invariants: None,
            jupiter_invariants: None,
            venus_invariants: None,
            mars_invariants: None,
            saturn_invariants: None,
            dt_max: DEFAULT_DT_MAX_S,
            // Aligned with `hf.eps` in
            // crates/nd_config/tests/fixtures/dissertation_production.yaml.
            eps: 1e-8,
            // NOT what the campaign flies. Compiled science moved
            // `integrator_method` to "vern7" on 2026-08-09 (R26) and this
            // default deliberately did not follow.
            //
            // Production never reads it: `nd_pipeline::hybrid`'s
            // `part_a_physics_from_controls` sets the method from the sealed
            // authority on every strict-HF config. Changing this to Vern7 would
            // therefore buy no production fidelity and would silently
            // re-baseline `end_to_end_routed_propagation_matches_its_pinned_state`,
            // whose two profile digests are measured on exactly this default.
            //
            // The claim this comment used to make -- that every
            // `..ForceConfig::default()` in `src/` sits inside a `#[cfg(test)]`
            // module -- is true of THIS crate and false of the workspace.
            // Compiler-checked 2026-08-11 by gating this `impl Default` behind
            // `#[cfg(test)]` and running `cargo check --workspace --lib`: this
            // crate's library compiles, and exactly two non-test sites break,
            // both in `two_phase_transfer_rs/src/`.
            // `postprocess.rs::build_force_config` is safe because it assigns
            // `integrator_method` from `PhysicsConfig` and never sees this
            // value. `hf_acceptance.rs::gravity_only_transfer_force_config`
            // does NOT assign it and so inherits Vern9; it is a `pub fn` whose
            // callers are today all tests, which is what keeps it harmless, not
            // anything about where it is declared.
            //
            // Read it as "the library's own stepper", not as "the production
            // stepper".
            integrator_method: StepperMethod::Vern9,
        }
    }
}

/// Exact structural identity for secondary `ForceConfig` inputs.
///
/// Compared by full equality, never summarized. Callers that require complete
/// force identity pair these words with primary force controls such as order,
/// model, flags, coefficients, tolerances, and packed-gravity authority.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForceConfigIdentity {
    subtract_first_order: u64,
    target_propagation_mode: u64,
    dynamic_ephemeris_flags: u64,
    /// `qm_ratio`, `r_obj_m`, `omega_earth`, `p_sun`, the six `mu_*`, and
    /// `earth_radius`, as bits.
    scalars: [u64; 11],
    /// Per body: presence tag then three position components, as bits.
    bodies: [[u64; 4]; 6],
}

impl ForceConfigIdentity {
    /// The exact words, for callers that need a content key of their own.
    ///
    /// Exposed so evidence counters can key on semantic identity instead of
    /// `Arc::as_ptr`, which counts allocations rather than physics.
    #[must_use]
    pub fn words(&self) -> [u64; 38] {
        let mut words = [0_u64; 38];
        // Split rather than index: no offset arithmetic to get wrong, and the
        // shape of the layout is visible in the code instead of in three
        // separate range expressions that have to agree.
        let (flags, rest) = words.split_at_mut(3);
        flags.copy_from_slice(&[
            self.subtract_first_order,
            self.target_propagation_mode,
            self.dynamic_ephemeris_flags,
        ]);
        let (scalars, bodies) = rest.split_at_mut(self.scalars.len());
        scalars.copy_from_slice(&self.scalars);
        for (slot, body) in bodies.chunks_exact_mut(4).zip(&self.bodies) {
            slot.copy_from_slice(body);
        }
        words
    }
}

impl ForceConfig {
    /// Exact bits of secondary RHS-input fields in this config.
    ///
    /// Primary controls (`sph_order`, `atm_model`, `force_flags`, `am_ratio`,
    /// `cd`, `cr`, `eps`, and `dt_max`) remain separate at callers that require
    /// complete force identity. These words cover `subtract_first_order`,
    /// `target_propagation_mode`, `dynamic_ephemeris_flags`, object properties,
    /// physical constants, and six active caller-supplied body positions. A
    /// matching dynamic-ephemeris flag makes the stored position inactive, so
    /// all such placeholders use one sentinel exactly as strict-HF science
    /// authority does. Static presence remains distinct from a zero vector.
    ///
    /// The `*_invariants` are deliberately NOT folded: they are per-segment
    /// precomputations derived from the positions and `mu_*` values that ARE
    /// folded, and a caller that hand-sets invariants inconsistent with their
    /// sources is outside every contract this crate states.
    ///
    /// This is not a hash. A former 64-bit fold admitted constructive collisions
    /// between physically distinct configs. Structural words keep evidence
    /// identity exact.
    #[must_use]
    pub fn force_config_identity(&self) -> ForceConfigIdentity {
        let mut scalars = [0_u64; 11];
        for (slot, value) in scalars.iter_mut().zip([
            self.qm_ratio,
            self.r_obj_m,
            self.omega_earth,
            self.p_sun,
            self.mu_sun,
            self.mu_moon,
            self.mu_jupiter,
            self.mu_venus,
            self.mu_mars,
            self.mu_saturn,
            self.earth_radius,
        ]) {
            *slot = value.to_bits();
        }
        let mut bodies = [[0_u64; 4]; 6];
        for ((slot, body), flag) in bodies
            .iter_mut()
            .zip([
                self.sun_pos,
                self.moon_pos,
                self.jupiter_pos,
                self.venus_pos,
                self.mars_pos,
                self.saturn_pos,
            ])
            .zip([
                ForceFlags::SUN_GRAVITY,
                ForceFlags::MOON_GRAVITY,
                ForceFlags::JUPITER_GRAVITY,
                ForceFlags::VENUS_GRAVITY,
                ForceFlags::MARS_GRAVITY,
                ForceFlags::SATURN_GRAVITY,
            ])
        {
            if (self.dynamic_ephemeris_flags & flag) != 0 {
                *slot = [3, 0, 0, 0];
                continue;
            }
            // Presence is carried in word 0 so `None` cannot equal a zero
            // vector, exactly as the replaced fold did.
            match body {
                None => *slot = [1, 0, 0, 0],
                Some(position) => {
                    *slot = [
                        2,
                        position[0].to_bits(),
                        position[1].to_bits(),
                        position[2].to_bits(),
                    ];
                }
            }
        }
        ForceConfigIdentity {
            subtract_first_order: u64::from(self.subtract_first_order),
            target_propagation_mode: u64::from(self.target_propagation_mode),
            dynamic_ephemeris_flags: u64::from(u32::from_ne_bytes(
                self.dynamic_ephemeris_flags.to_ne_bytes(),
            )),
            scalars,
            bodies,
        }
    }

    /// Body-force bits that require catalogue-backed time-varying positions.
    #[must_use]
    pub fn required_dynamic_ephemeris_flags(&self) -> i32 {
        let mut dynamic_flags = self.dynamic_ephemeris_flags & ForceFlags::THIRDBODY_ALL;
        // JB2008 consumes the Earth-centred Sun vector at every RK stage for
        // right ascension/declination, so a caller-provided static Sun cannot
        // satisfy its geometry contract.
        if crate::rhs::atm_model_uses_jb2008_drivers(self.atm_model)
            && (self.force_flags & ForceFlags::DRAG) != 0
        {
            dynamic_flags |= ForceFlags::SUN_GRAVITY;
        }
        if self.sun_pos.is_none()
            && (self.force_flags & (ForceFlags::SUN_GRAVITY | ForceFlags::SRP)) != 0
        {
            dynamic_flags |= ForceFlags::SUN_GRAVITY;
        }
        for (flag, missing) in [
            (ForceFlags::MOON_GRAVITY, self.moon_pos.is_none()),
            (ForceFlags::JUPITER_GRAVITY, self.jupiter_pos.is_none()),
            (ForceFlags::VENUS_GRAVITY, self.venus_pos.is_none()),
            (ForceFlags::MARS_GRAVITY, self.mars_pos.is_none()),
            (ForceFlags::SATURN_GRAVITY, self.saturn_pos.is_none()),
        ] {
            if missing && (self.force_flags & flag) != 0 {
                dynamic_flags |= flag;
            }
        }
        dynamic_flags
    }

    /// Resolve catalogue-backed bodies and validate the complete absolute arc.
    ///
    /// Explicit caller positions remain fixed overrides. Missing positions for
    /// requested bodies become dynamic and are checked at both inclusive arc
    /// endpoints before any RHS can be constructed. Requested forces are never
    /// disabled: missing or insufficient catalogues return a typed error.
    ///
    /// # Errors
    ///
    /// Returns an error when either endpoint is non-finite, requested force
    /// data is unavailable, or Part A ephemeris/JB2008 authority validation
    /// fails.
    pub fn with_ephemeris_for_arc(
        mut self,
        jd_a: f64,
        jd_b: f64,
    ) -> Result<Self, crate::precomputed_ephem::EphemerisCoverageError> {
        use crate::precomputed_ephem::{
            load_precomputed_ephemeris, published_ephemeris, Body, EphemerisCoverageError,
        };

        if !jd_a.is_finite() || !jd_b.is_finite() {
            return Err(EphemerisCoverageError::NonFiniteArc { jd_a, jd_b });
        }

        if crate::rhs::atm_model_uses_jb2008_drivers(self.atm_model)
            && (self.force_flags & ForceFlags::COULOMB_DRAG) != 0
        {
            return Err(EphemerisCoverageError::CatalogueLoad {
                requested_flags: ForceFlags::SUN_GRAVITY,
                message: "JB2008 exact/approximation modes cannot be combined with Coulomb drag"
                    .to_string(),
            });
        }

        if crate::rhs::atm_model_uses_jb2008_drivers(self.atm_model)
            && (self.force_flags & ForceFlags::DRAG) != 0
        {
            let utc_start = jb_rs::drivers::UtcJulianDay::new(jd_a.min(jd_b)).map_err(|error| {
                EphemerisCoverageError::catalogue_source(
                    ForceFlags::SUN_GRAVITY,
                    format!("JB2008 UTC arc endpoint is invalid: {error}"),
                    error,
                )
            })?;
            let utc_end = jb_rs::drivers::UtcJulianDay::new(jd_a.max(jd_b)).map_err(|error| {
                EphemerisCoverageError::catalogue_source(
                    ForceFlags::SUN_GRAVITY,
                    format!("JB2008 UTC arc endpoint is invalid: {error}"),
                    error,
                )
            })?;
            let authority =
                crate::rhs::jb2008_driver_authority(self.atm_model).ok_or_else(|| {
                    EphemerisCoverageError::CatalogueLoad {
                        requested_flags: ForceFlags::SUN_GRAVITY,
                        message: "JB2008 atmosphere model has no compiled driver authority"
                            .to_string(),
                    }
                })?;
            let drivers = authority.load().map_err(|error| {
                EphemerisCoverageError::catalogue_source(
                    ForceFlags::SUN_GRAVITY,
                    format!("JB2008 driver authority failure: {error}"),
                    error,
                )
            })?;
            drivers
                .validate_utc_arc(utc_start, utc_end)
                .map_err(|error| {
                    EphemerisCoverageError::catalogue_source(
                        ForceFlags::SUN_GRAVITY,
                        format!("JB2008 UTC driver coverage failure: {error}"),
                        error,
                    )
                })?;
        }

        let dynamic_flags = self.required_dynamic_ephemeris_flags();

        if dynamic_flags != 0 {
            load_precomputed_ephemeris(dynamic_flags).map_err(|error| {
                EphemerisCoverageError::catalogue_source(
                    dynamic_flags,
                    error.to_string(),
                    error.into(),
                )
            })?;
            // Borrowed, not an owned `Arc`. This runs on every HF row and every
            // HF propagation segment; `get_precomputed_ephemeris` would take
            // the store's read lock and bump a refcount, which is two
            // read-modify-writes on process-wide cache lines per call and
            // serializes the worker pool. Nothing here outlives the borrow.
            let ephem =
                published_ephemeris().ok_or_else(|| EphemerisCoverageError::CatalogueLoad {
                    requested_flags: dynamic_flags,
                    message: "global ephemeris store unavailable after successful load".to_string(),
                })?;
            ephem.validate_dynamic_arc(dynamic_flags, jd_a, jd_b)?;

            let jd_anchor = 0.5 * jd_a + 0.5 * jd_b;
            let utc_anchor = jb_rs::drivers::UtcJulianDay::new(jd_anchor).map_err(|error| {
                EphemerisCoverageError::catalogue_source(
                    dynamic_flags,
                    format!("dynamic ephemeris UTC anchor is invalid: {error}"),
                    error,
                )
            })?;
            let position = |body: Body| {
                ephem
                    .get(body)
                    .map(|table| table.position_at_part_a_utc_jd(utc_anchor))
                    .transpose()
                    .map_err(|error| {
                        EphemerisCoverageError::catalogue_source(
                            dynamic_flags,
                            format!(
                                "{} ephemeris is outside Part A UTC-JD authority: {error}",
                                body.name()
                            ),
                            error.into(),
                        )
                    })?
                    .ok_or(EphemerisCoverageError::MissingBody { body })
                    .map(Some)
            };
            if (dynamic_flags & ForceFlags::SUN_GRAVITY) != 0 {
                self.sun_pos = position(Body::Sun)?;
            }
            if (dynamic_flags & ForceFlags::MOON_GRAVITY) != 0 {
                self.moon_pos = position(Body::Moon)?;
            }
            if (dynamic_flags & ForceFlags::JUPITER_GRAVITY) != 0 {
                self.jupiter_pos = position(Body::Jupiter)?;
            }
            if (dynamic_flags & ForceFlags::VENUS_GRAVITY) != 0 {
                self.venus_pos = position(Body::Venus)?;
            }
            if (dynamic_flags & ForceFlags::MARS_GRAVITY) != 0 {
                self.mars_pos = position(Body::Mars)?;
            }
            if (dynamic_flags & ForceFlags::SATURN_GRAVITY) != 0 {
                self.saturn_pos = position(Body::Saturn)?;
            }
        }

        self.dynamic_ephemeris_flags = dynamic_flags;
        let dynamic_or_static = |flag: i32, position: Option<[f64; 3]>, mu: f64| {
            if (dynamic_flags & flag) != 0 {
                None
            } else {
                position.and_then(|value| BodyInvariants::precompute(&value, mu))
            }
        };
        self.sun_invariants = dynamic_or_static(ForceFlags::SUN_GRAVITY, self.sun_pos, self.mu_sun);
        self.moon_invariants =
            dynamic_or_static(ForceFlags::MOON_GRAVITY, self.moon_pos, self.mu_moon);
        self.jupiter_invariants = dynamic_or_static(
            ForceFlags::JUPITER_GRAVITY,
            self.jupiter_pos,
            self.mu_jupiter,
        );
        self.venus_invariants =
            dynamic_or_static(ForceFlags::VENUS_GRAVITY, self.venus_pos, self.mu_venus);
        self.mars_invariants =
            dynamic_or_static(ForceFlags::MARS_GRAVITY, self.mars_pos, self.mu_mars);
        self.saturn_invariants =
            dynamic_or_static(ForceFlags::SATURN_GRAVITY, self.saturn_pos, self.mu_saturn);
        Ok(self)
    }
}

/// Event detection result
#[derive(Debug, Clone, Copy)]
pub(crate) struct EventDetection {
    pub detected: bool,
    pub event_type: EventType,
    pub refined_time: f64,
    pub state_at_event: [f64; 6],
    pub interp_method: InterpMethod,
    pub interp_error: f64,
}

impl Default for EventDetection {
    fn default() -> Self {
        Self {
            detected: false,
            event_type: EventType::Ground, // placeholder
            refined_time: 0.0,
            state_at_event: [0.0; 6],
            interp_method: InterpMethod::None,
            interp_error: 0.0,
        }
    }
}

/// Result of an integration with event handling
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    pub times: Vec<f64>,       // Output times (n_out,)
    pub states: Vec<[f64; 6]>, // Output states (n_out, 6)

    pub terminal_event_fired: bool, // True if a terminal event occurred
    /// Name of the terminal event. Every assignment site is a fixed literal
    /// or a `&'static str` name table except one catch-all
    /// `format!("{status:?}")` arm, hence `Cow` rather than `&'static str`.
    pub terminal_event_name: std::borrow::Cow<'static, str>,
    /// Typed binary-eclipse numerical failure, when that coordinator stopped
    /// the sampled propagation. The name remains only for display/legacy logs.
    pub terminal_eclipse_error: Option<crate::eclipse::EclipseError>,
    /// Typed packed-gravity evaluation failure, preserved through sampled
    /// propagation instead of reconstructing it from a terminal display name.
    pub terminal_gravity_error: Option<satpy_core::GravityError>,

    pub perturb_deviation_fired: bool, // True if perturb_deviation fired (non-terminal restart)
    pub max_steps_exceeded: bool,      // True if max steps reached (safety)
    pub event_time: f64,               // Time at which event occurred
    pub state_at_event: [f64; 6],      // State at event time
    pub event_interp_method: InterpMethod, // Event interpolation method used
    pub event_interp_error: f64,       // Event interpolation error estimate
    pub metrics: OdeMetrics,           // Performance metrics
}

impl Default for IntegrationResult {
    fn default() -> Self {
        Self {
            times: Vec::new(),
            states: Vec::new(),
            terminal_event_fired: false,
            terminal_event_name: std::borrow::Cow::Borrowed(""),
            terminal_eclipse_error: None,
            terminal_gravity_error: None,
            perturb_deviation_fired: false,
            max_steps_exceeded: false,
            event_time: 0.0,
            state_at_event: [0.0; 6],
            event_interp_method: InterpMethod::None,
            event_interp_error: 0.0,
            metrics: OdeMetrics::default(),
        }
    }
}

/// Cubic Hermite interpolation.
///
/// `H(tau) = (1-tau)^2(1+2tau)y0 + tau^2(3-2tau)y1 + tau(1-tau)^2 h*dy0 - tau^2(1-tau) h*dy1`.
///
/// Here `tau = (t - t0) / h`.
#[inline]
#[must_use]
pub(crate) fn hermite_interp(
    y0: &[f64; 6],
    y1: &[f64; 6],
    dy0: &[f64; 6],
    dy1: &[f64; 6],
    h: f64,
    tau: f64,
) -> [f64; 6] {
    let tau2 = tau * tau;
    let tau3 = tau2 * tau;

    let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
    let h10 = tau3 - 2.0 * tau2 + tau;
    let h01 = -2.0 * tau3 + 3.0 * tau2;
    let h11 = tau3 - tau2;

    let mut out = [0.0; 6];
    for ((((output, &state_start), &state_end), &derivative_start), &derivative_end) in out
        .iter_mut()
        .zip(y0.iter())
        .zip(y1.iter())
        .zip(dy0.iter())
        .zip(dy1.iter())
    {
        *output = h00 * state_start
            + h10 * h * derivative_start
            + h01 * state_end
            + h11 * h * derivative_end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_body_position(config: &mut ForceConfig, body_index: usize, position: [f64; 3]) {
        match body_index {
            0 => config.sun_pos = Some(position),
            1 => config.moon_pos = Some(position),
            2 => config.jupiter_pos = Some(position),
            3 => config.venus_pos = Some(position),
            4 => config.mars_pos = Some(position),
            5 => config.saturn_pos = Some(position),
            _ => panic!("test body index must be in 0..6"),
        }
    }

    #[test]
    fn force_config_identity_ignores_only_dynamic_body_positions() {
        let flags = [
            ForceFlags::SUN_GRAVITY,
            ForceFlags::MOON_GRAVITY,
            ForceFlags::JUPITER_GRAVITY,
            ForceFlags::VENUS_GRAVITY,
            ForceFlags::MARS_GRAVITY,
            ForceFlags::SATURN_GRAVITY,
        ];

        for (body_index, flag) in flags.into_iter().enumerate() {
            let mut base = ForceConfig {
                dynamic_ephemeris_flags: flag,
                ..ForceConfig::default()
            };
            set_body_position(&mut base, body_index, [1.0, 2.0, 3.0]);
            let mut moved = base;
            set_body_position(&mut moved, body_index, [4.0, 5.0, 6.0]);

            assert!(
                base.force_config_identity() == moved.force_config_identity(),
                "dynamic body {body_index} retained an inactive stored position"
            );

            let other_body_index = body_index.wrapping_add(1) % flags.len();
            let mut unflagged_moved = base;
            set_body_position(&mut unflagged_moved, other_body_index, [7.0, 8.0, 9.0]);
            assert!(
                base.force_config_identity() != unflagged_moved.force_config_identity(),
                "dynamic body {body_index} hid unflagged body {other_body_index}"
            );

            base.dynamic_ephemeris_flags = 0;
            moved.dynamic_ephemeris_flags = 0;
            assert!(
                base.force_config_identity() != moved.force_config_identity(),
                "static body {body_index} lost its active stored position"
            );
        }
    }

    #[test]
    fn checked_ephemeris_marks_catalogue_positions_dynamic_but_keeps_overrides_fixed() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let flags = ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY | ForceFlags::SRP;
        let ephem = crate::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("test ephemeris catalogues must load");
        let jd = ephem
            .common_jd_range()
            .map(|(start, end)| 0.5 * (start + end))
            .expect("test ephemeris range must exist");
        let explicit_sun = [149_597_870.7, 1234.0, -567.0];
        let config = ForceConfig {
            force_flags: flags,
            sun_pos: Some(explicit_sun),
            ..ForceConfig::default()
        };

        let resolved = config
            .with_ephemeris_for_arc(jd, jd)
            .expect("catalogue-backed point arc must resolve");

        assert_eq!(resolved.sun_pos, Some(explicit_sun));
        assert_eq!(
            resolved.dynamic_ephemeris_flags & ForceFlags::SUN_GRAVITY,
            0
        );
        assert_ne!(
            resolved.dynamic_ephemeris_flags & ForceFlags::MOON_GRAVITY,
            0
        );
        assert!(resolved.sun_invariants.is_some());
        assert!(resolved.moon_invariants.is_none());
    }

    #[test]
    fn checked_ephemeris_resolution_validates_full_arc_before_rhs() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let flags = ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY | ForceFlags::SRP;
        let ephem = crate::precomputed_ephem::try_load_precomputed_ephemeris(flags)
            .expect("test ephemeris catalogues must load");
        let (start, end) = ephem
            .common_jd_range()
            .expect("test ephemeris range must exist");
        let config = ForceConfig {
            force_flags: flags,
            ..ForceConfig::default()
        };

        let resolved = config
            .with_ephemeris_for_arc(start, end)
            .expect("inclusive full arc must resolve");
        assert_eq!(resolved.force_flags, flags);
        assert_eq!(
            resolved.dynamic_ephemeris_flags & flags,
            flags & !ForceFlags::SRP
        );

        let one_ulp_after_end = f64::from_bits(end.to_bits() + 1);
        let error = config
            .with_ephemeris_for_arc(start, one_ulp_after_end)
            .expect_err("out-of-range endpoint must fail before RHS");
        assert!(matches!(
            error,
            crate::precomputed_ephem::EphemerisCoverageError::OutsideRange { .. }
        ));
    }

    #[test]
    fn checked_ephemeris_resolution_preserves_explicit_override_semantics() {
        let explicit_sun = [149_597_870.7, 1234.0, -567.0];
        let flags = ForceFlags::SUN_GRAVITY | ForceFlags::SRP;
        let config = ForceConfig {
            force_flags: flags,
            sun_pos: Some(explicit_sun),
            ..ForceConfig::default()
        };

        let resolved = config
            .with_ephemeris_for_arc(1.0, 2.0)
            .expect("fixed explicit override requires no catalogue coverage");
        assert_eq!(resolved.force_flags, flags);
        assert_eq!(resolved.sun_pos, Some(explicit_sun));
        assert_eq!(resolved.dynamic_ephemeris_flags, 0);
        assert!(resolved.sun_invariants.is_some());
    }

    #[test]
    fn jb2008_drag_forces_dynamic_sun_despite_static_override() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let jd = 2_459_600.5;
        let explicit_sun = [149_597_870.7, 1234.0, -567.0];
        let config = ForceConfig {
            force_flags: ForceFlags::DRAG,
            atm_model: 4,
            sun_pos: Some(explicit_sun),
            ..ForceConfig::default()
        };

        assert_ne!(
            config.required_dynamic_ephemeris_flags() & ForceFlags::SUN_GRAVITY,
            0,
            "JB2008 drag must use stage-resolved Sun geometry"
        );
        let resolved = config
            .with_ephemeris_for_arc(jd, jd + 1.0)
            .expect("JB2008 test arc must resolve");
        assert_ne!(
            resolved.dynamic_ephemeris_flags & ForceFlags::SUN_GRAVITY,
            0
        );
        assert_ne!(resolved.sun_pos, Some(explicit_sun));
        assert!(resolved.sun_invariants.is_none());
    }

    #[test]
    fn jb2008_drag_rejects_arc_outside_driver_coverage() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        // Native Sun ephemeris reaches 2028; compiled JB2008 drivers end in
        // 2026, so this must fail before an RHS can enter its hot loop.
        let outside_jb_driver_jd = 2_462_000.5;
        let error = ForceConfig {
            force_flags: ForceFlags::DRAG,
            atm_model: 4,
            sun_pos: Some([149_597_870.7, 0.0, 0.0]),
            ..ForceConfig::default()
        }
        .with_ephemeris_for_arc(outside_jb_driver_jd, outside_jb_driver_jd + 1.0)
        .expect_err("JB2008 driver arc coverage must fail closed");

        assert!(error.to_string().contains("JB2008"));
        assert!(
            std::error::Error::source(&error).is_some(),
            "JB2008 driver error must remain an owned source"
        );
    }

    #[test]
    fn part_a_v3_jb2008_rejects_outside_authorized_persistence_arc() {
        let _guard = crate::precomputed_ephem::ephemeris_test_guard();
        let identity = jb_rs::drivers::compiled_part_a_v3_identity()
            .expect("compiled Part A v3 driver identity");
        let outside = identity.authorized_end_utc_jd + 1.0;
        let error = ForceConfig {
            force_flags: ForceFlags::DRAG,
            atm_model: 8,
            ..ForceConfig::default()
        }
        .with_ephemeris_for_arc(outside, outside + 0.01)
        .expect_err("model 8 must reject arcs outside its sealed persistence window");

        assert!(error
            .to_string()
            .contains("Part A v3 authorized persistence arc"));
    }

    #[test]
    fn force_config_default_preserves_legacy_hf_target_mode() {
        assert_eq!(ForceConfig::default().target_propagation_mode, 0);
    }

    #[test]
    fn sampled_result_preserves_exact_gravity_failure_at_final_boundary() {
        let result = IntegrationResult {
            terminal_gravity_error: Some(satpy_core::GravityError::InvalidRadius),
            ..IntegrationResult::default()
        };

        assert_eq!(
            crate::integrator::final_propagation_failure(&result),
            Some(crate::integrator::FinalPropagationFailure::Gravity(
                satpy_core::GravityError::InvalidRadius
            ))
        );
    }
}
