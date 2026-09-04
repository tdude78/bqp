//! Standalone scalar JB2008 density kernel.
//!
//! Inputs are explicit; this module performs no data loading or fallback.

pub const JB2008_KERNEL_NAME: &str = "orekit_13_1_2_jb2008_f64_kernel";
pub const JB2008_KERNEL_VERSION: &str = "v1";
pub const JB2008_MODEL_NAME: &str = "orekit_13_1_2_jb2008_f64_kernel_v1";
pub const JB2008_LOGQUAD_X4_APPROX_V1_MODEL_NAME: &str =
    "orekit_13_1_2_jb2008_logquad_x4_approx_v1";
pub const JB2008_LOGQUAD_X4_APPROX_V1_TRANSFORM: &str =
    "exact_jb2008_log_intervals_0.010_0.025_0.075_times_4";

/// The bound model 6 must track exact JB2008 within, re-scoped from 3.0e-6 by
/// user ruling on 2026-08-09 (R16 decision doc, R22 addendum).
///
/// Public and singular on purpose. The gate
/// (`v2_broad_grid_density_error_stays_within_rescoped_bound`, an integration
/// test) and the proof that the gate has teeth
/// (`the_rescoped_bound_rejects_the_rung_the_accuracy_gates_wave_through`, a
/// unit test in this file) each declared their own `1.0e-4` literal. Nothing
/// linked them, so relaxing the gate's copy left the poison proof green and
/// still "proving" a bound nobody enforced — it would have gone on asserting
/// that 1.0e-4 rejects `middle 0.400` while the gate admitted 1.0e-2. One
/// constant; both sides read it.
pub const V2_RESCOPED_DENSITY_BOUND: f64 = 1.0e-4;

// Adapted from Apache Orekit 13.1.2 `JB2008.java` and
// `AbstractJacchiaBowmanModel.java` (Copyright 2002-2025 CS GROUP),
// Apache License 2.0.  See `THIRD_PARTY_NOTICES.md` in this crate.
const JB_ALPHA: [f64; 5] = [0.0, 0.0, 0.0, 0.0, -0.38];
const JB_AMW: [f64; 6] = [28.0134, 31.9988, 15.9994, 39.9480, 4.0026, 1.00797];
const JB_AVOGAD: f64 = 6.02257e26;
const JB_FRAC: [f64; 4] = [0.78110, 0.20955, 9.3400e-3, 1.2890e-5];
const JB_RSTAR: f64 = 8.31432;
const JB_WT: [f64; 5] = [
    0.311_111_111_111_111,
    1.422_222_222_222_222,
    0.533_333_333_333_333,
    1.422_222_222_222_222,
    0.311_111_111_111_111,
];
const JB_BDT: [f64; 19] = [
    -4.575_122_97,
    -5.121_149_09,
    -69.300_360_9,
    203.716_701,
    703.316_291,
    -1943.49234,
    1106.51308,
    -174.378_996,
    1885.94601,
    -7093.71517,
    9224.54523,
    -3845.08073,
    -6.458_417_89,
    40.970_331_9,
    -482.006_560,
    1818.70931,
    -2373.89204,
    996.703_815,
    36.141_693_6,
];
const JB_CDT: [f64; 23] = [
    -15.598_621_1,
    -5.121_149_09,
    -69.300_360_9,
    203.716_701,
    703.316_291,
    -1943.49234,
    1106.51308,
    -220.835_117,
    1432.56989,
    -3184.81844,
    3289.81513,
    -1353.32119,
    19.995_648_9,
    -12.709_399_8,
    21.282_515_6,
    -2.755_554_32,
    11.023_498_2,
    148.881_951,
    -751.640_284,
    637.876_542,
    12.709_399_8,
    -21.282_515_6,
    2.755_554_32,
];
const JB_CXAMB: [f64; 7] = [
    28.15204, -8.5586e-2, 1.2840e-4, -1.0056e-5, -1.0210e-5, 1.5044e-6, 9.9826e-8,
];
const JB_CHT: [f64; 4] = [0.22, -0.20e-2, 0.115e-2, -0.211e-5];
const JB_FZM: [f64; 5] = [0.2689, -0.01176, 0.02782, -0.02782, 0.000_347_0];
const JB_GTM: [f64; 10] = [
    -0.3633,
    0.08506,
    0.2401,
    -0.1897,
    -0.2554,
    -0.01790,
    0.000_565_0,
    -0.000_640_7,
    -0.003_418,
    -0.001_252,
];
/// The US Standard Atmosphere geopotential reference radius, and a FOURTH Earth
/// radius in this repo alongside WGS84 `RE` (6378.137), the DIR-R6 gravity
/// reference (6378.13646) and `ForceConfig::earth_radius`.
///
/// It is not a geometry radius and it is not an error that it disagrees with
/// them by 21 km. It appears in exactly one place — `jb_gravity`, converting
/// geometric to geopotential altitude — and 6356.766 km paired with
/// `JB_G0_M_S2` = 9.80665 m/s^2 is what US-Std-Atm-1976 specifies for that
/// conversion, which is what JB2008 was fitted against.
///
/// DO NOT unify it with any of the other three. `jb2008_density` is held to a
/// sealed Orekit JAR fixture bit-for-bit, so substituting a geometry radius here
/// breaks the contract rather than improving the model.
const JB_EARTH_RADIUS_KM: f64 = 6356.766;
const JB_G0_M_S2: f64 = 9.80665;
const JB_ALTITUDE_MIN_M: f64 = 90_000.0;

/// Internal compile-time profiles only; no external runtime tuning surface.
mod quadrature {
    use super::{LowerState, MiddleState, TemperatureBroadcast};

    pub(super) trait Sealed {}

    pub(super) trait QuadratureProfile: Sealed {
        const LOWER_LOG_STEP: f64;
        const MIDDLE_LOG_STEP: f64;
        const UPPER_LOG_STEP: f64;

        /// Whether the 105--500 km segment runs from a precomputed plan.
        ///
        /// The same argument as `USE_FIXED_LOWER_PLAN`, one segment up and
        /// worth far more. That segment integrates the lower plan's exit
        /// altitude to `min(altitude, 500)` km, so **at or above 500 km both
        /// its bounds are constants**: the step ratio, every Boole abscissa,
        /// every gravity on them, and every altitude-dependent factor of the
        /// arctangent argument are the same numbers on every call, and again
        /// only `tc` moves.
        ///
        /// Production never leaves that regime. Censused on the sealed V3 arc
        /// (`atmosphere_model` 5, 7,829 RHS evaluations): altitude spans
        /// 626.2--985.7 km, **7,829 of 7,829 evaluations are at or above
        /// 500 km**, the middle step count is 16 every time, and the recorded
        /// `(z_start, n, zr)` triple never differs from the first one seen.
        /// The segment is 88.1% of every quadrature step not already covered
        /// by the lower plan.
        const USE_FIXED_MIDDLE_PLAN: bool;

        /// Whether the 90--105 km segment runs from a precomputed plan.
        ///
        /// That segment integrates 90 km to `min(altitude, 105)` km, so at or
        /// above 105 km its upper bound is the literal `105.0` on every call:
        /// the step ratio, every Boole abscissa, and every molecular mass and
        /// gravity on them are the same numbers each time, and only `tc` moves.
        /// The plan holds exactly the altitude-independent part.
        const USE_FIXED_LOWER_PLAN: bool;

        /// Whether the five species number densities are carried as a linear
        /// factor plus a log offset instead of as `ln(x)`.
        ///
        /// `jb_density` finishes with `exp(...)` per species, and the value it
        /// exponentiates started life as `ln(x)`. That is a round trip:
        /// `exp(ln(x) + y) == x * exp(y)`. Retiring it removes five `ln` calls
        /// per density evaluation, and it is the more accurate association —
        /// `ln`'s rounding entered an `exp` ARGUMENT, where absolute error
        /// becomes relative error, and this model drives `|ln(x)|` to 45.48. At
        /// 60 decimal digits over 1,601 corpus pairs the round trip's worst
        /// error is 68.80 ULP against the retired form's 0.71.
        ///
        /// **It is nevertheless false on the exact profile, and that is the
        /// whole reason this is a profile constant rather than a rewrite.**
        /// `orekit_synthetic_mapping_matches_rust_primitive_kernel` requires
        /// the exact profile to reproduce the sealed Orekit 13.1.2 fixture
        /// BIT for bit. Orekit computes the logarithms, so any algebraic
        /// restructuring loses that by construction, however much more accurate
        /// it is — and it does: with this true on the exact profile, 11 fixture
        /// cases go red. Being nearer the true value is not the property that
        /// oracle asserts.
        ///
        /// The approximation profile has no such contract. It is declared
        /// non-exact by `x4_approximation_has_explicit_nonexact_identity`, its
        /// coarser grid already diverges from Orekit by far more than this, and
        /// it is the profile production actually flies (`atm_model` 5).
        const RETIRE_SPECIES_ROUND_TRIP: bool;

        /// Whether the upper segment's step ratio is taken from the altitude
        /// ratio directly instead of from `exp(ln(ratio))`.
        ///
        /// `jb_density` forms `al = ln(alt / z)` — which it needs, because the
        /// step count is `al / UPPER_LOG_STEP` — and then `zr = exp(al / n)`.
        /// At `n == 1` those two calls are a round trip and `zr` is the ratio
        /// itself, up to the rounding of an `exp` composed with a `ln`.
        ///
        /// `n == 1` is not the rare case, it is the only case production
        /// reaches: the flown profile's `UPPER_LOG_STEP` is 0.700 and the
        /// censused altitude band tops out at 985.7 km, so `al` never exceeds
        /// `ln(985.7 / 500) = 0.679`. The guard is still a runtime test on `n`,
        /// because a caller outside that band is a supported input and must get
        /// the walked answer.
        ///
        /// **This moves bits, which is why it is a profile constant.** The
        /// round trip is worth one or two ULP on `zr`, and `zr` scales the one
        /// upper Boole step's abscissae, so the density moves at the same scale.
        /// The direct ratio is the more accurate value — `exp(ln(x))` cannot be
        /// nearer `x` than `x` is — but "more accurate" is not the property the
        /// exact profile is held to, exactly as `RETIRE_SPECIES_ROUND_TRIP`
        /// records. It is false on every profile but the flown one, so models 4,
        /// 5 and 6 stay bit-stable and remain usable as controls when diffing
        /// `tests/jb2008_libm_probe.rs`'s per-profile dumps.
        ///
        /// Priced at **-1.70% of the model-7 kernel call** (median of 22
        /// rotating paired rounds, 19 of 22 negative; min-of-min -2.33%), i.e.
        /// about -0.62% of the production arc. One `exp` at throughput is 1.05%
        /// of the call, so most of what this returns is dependency latency:
        /// `zr` sits between the exospheric temperature and every abscissa of
        /// the step that produces `sum3`.
        const RETIRE_ZR_ROUND_TRIP: bool;

        /// Altitude in km at or above which `jb_dlrsl` is taken as zero.
        ///
        /// The seasonal-latitudinal correction carries the factor
        /// `0.02 * h * exp(-0.045 * h)` for `h = altitude - 90 km`, and every
        /// other factor in it is bounded by one. That envelope is monotone
        /// decreasing above `h = 22.2 km` and it collapses: `1.10e-9` at
        /// 600 km, `1.46e-11` at 700, `1.88e-13` at 800, `2.39e-15` at 900. The
        /// correction is applied as `10^dlrsl`, so the relative move in density
        /// from dropping it is `ln(10)` times those numbers.
        ///
        /// **`f64::INFINITY` on every profile but the flown one.** On the flown
        /// one it is 800 km, where the bound is **4.34e-13 of the density** —
        /// five orders under the `5.75e-5` the fitted profile's own gate already
        /// accepts, and chosen as the round number under a 1e-12 budget rather
        /// than tuned for speed.
        ///
        /// # Why this cannot be made bit-identical instead
        ///
        /// The correction enters as `dlr = ln(10) * (dlrsl + semiannual)`, and
        /// the semiannual term is of order `0.1`, so dropping `dlrsl` is exactly
        /// a no-op only while `|dlrsl|` stays under half an ULP of it, i.e.
        /// under about `7e-18`. The envelope does not reach that until roughly
        /// 1040 km — above the whole censused band. A bit-identical version of
        /// this lever does not exist inside the altitudes production flies.
        ///
        /// Priced by a ceiling probe that dropped the term at ALL altitudes at
        /// **-4.37% of the kernel call** (22 of 22 rounds negative), of which
        /// this gate collects the part of the band above 800 km.
        const DLRSL_ZERO_ABOVE_KM: f64;

        /// Whether the above-500 km segment is evaluated from the five
        /// altitude fits instead of walked as a Boole step.
        ///
        /// **FALSE EVERYWHERE: built, measured and REFUTED. It is SLOWER.**
        ///
        /// This is `docs/JB2008_COST_MAP.md`'s L1, the largest lever ever
        /// measured in this kernel — deleting the segment outright is
        /// **-16.69%** of the model-7 kernel call, 20 of 20 rounds negative.
        /// The expansion that replaces it, on an M1 Pro over 40 rotating paired
        /// rounds against a byte-identical in-round control at -0.33%, reads
        /// **+2.03%**. The replacement costs about **1.15x** what it replaces.
        ///
        /// The accuracy is not the problem and is worth recording so nobody
        /// re-derives it: over the fit's whole altitude and temperature domain
        /// the expansion moves the density by at most **2.02e-10**, which is
        /// five orders inside the 5.75e-5 the fitted profile's own gate accepts
        /// and 1000x inside this lever's own 1e-6 bound. The separation in
        /// [`fitted_upper_segment`] is exact and the fits are three to five
        /// orders inside their budgets. **It is arithmetic count that kills
        /// it**, not error: 64 FMAs across five Estrin bodies plus 552 bytes of
        /// coefficients, against a Boole step of four abscissae, one
        /// `atan_x4`, one `sqrt` and one divide.
        ///
        /// # What this cost, and the rule it confirms
        ///
        /// The design predicted ~8 ns for the five fits, from R28's measured
        /// 1.63 ns per degree-14 fitted scalar. That is a THROUGHPUT figure,
        /// and these five sit directly on the critical path between the
        /// exospheric temperature and every species exponent. **A ceiling probe
        /// bounds a lever and does not locate the cost inside it** — the note
        /// §6g of the cost map draws from `validate` and from `atan_x4` applies
        /// to this design too, and it was written by the same round that then
        /// walked into it.
        ///
        /// Hoisting the shared powers of the fit variable out of the five
        /// bodies and folding two divides into one reciprocal moved it from
        /// +2.03% to +1.85%, i.e. nothing. The remaining route, unexplored, is
        /// a change of fit variable: the needed degrees are 5, 11, 10, 5 and 12
        /// and `G_k` behaves like `dz^-3.5k`, so fitting in a variable that
        /// linearises that power law could roughly halve them. Halving 64 FMAs
        /// is about 4 ns against a 5 ns deficit, so it is the only thing left
        /// that could flip the sign, and it is not obviously enough.
        ///
        /// # If it is ever turned on
        ///
        /// It moves the flown density, so it wants a NEW `atmosphere_model`
        /// integer rather than an edit to 7 — see `LogQuadratureX4ApproxV2` for
        /// why editing a profile in place makes receipts indistinguishable.
        /// `FittedV7FittedUpper` in this module's tests is where the arm stays
        /// live and measured while the production constant is false.
        ///
        /// The arm has four preconditions beyond this constant and they are
        /// checked at the call site, not here: the middle segment must have
        /// come from the PLAN (so `z` and `zend` are the constants the fits were
        /// generated against), the altitude must be inside
        /// `[UPPER_FIT_ALT_LO, UPPER_FIT_ALT_HI]`, and the exospheric
        /// temperature inside the same domain the rest of the fitted profile
        /// uses. Any one failing walks the step exactly as before, so the
        /// domain is a speed boundary and not a validity one.
        ///
        /// See [`fitted_upper_segment`] for the separation this rests on and
        /// `docs/JB2008_COST_MAP.md` §6f for why it is five 1-D fits and not one
        /// 2-D fit.
        const FITTED_UPPER_SEGMENT: bool;

        /// Fixed-geometry 90--105 km state.
        ///
        /// Each profile owns its own plan because the step count follows
        /// `LOWER_LOG_STEP`; `fixed_lower_plan_step_counts_track_the_log_steps`
        /// pins that correspondence. Only called when `USE_FIXED_LOWER_PLAN`
        /// and the altitude is at or above 105 km.
        fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState;

        /// Fixed-geometry 105--500 km state.
        ///
        /// Each profile owns its own plan for the same reason the lower plans
        /// are separate: the step count follows `MIDDLE_LOG_STEP`, and
        /// `fixed_middle_plan_step_counts_track_the_log_steps` pins that.
        /// Only called when `USE_FIXED_MIDDLE_PLAN`, the lower plan ran, and
        /// the altitude is at or above 500 km.
        fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState;
    }
}

use num_traits::ToPrimitive;
use quadrature::{QuadratureProfile, Sealed};

struct ExactOrekitQuadrature;

impl Sealed for ExactOrekitQuadrature {}

impl QuadratureProfile for ExactOrekitQuadrature {
    const LOWER_LOG_STEP: f64 = 0.010;
    const MIDDLE_LOG_STEP: f64 = 0.025;
    const UPPER_LOG_STEP: f64 = 0.075;
    const USE_FIXED_LOWER_PLAN: bool = true;
    const RETIRE_SPECIES_ROUND_TRIP: bool = false;
    const RETIRE_ZR_ROUND_TRIP: bool = false;
    const DLRSL_ZERO_ABOVE_KM: f64 = f64::INFINITY;
    const FITTED_UPPER_SEGMENT: bool = false;
    const USE_FIXED_MIDDLE_PLAN: bool = true;

    #[inline]
    fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
        fixed_lower_state(exact_fixed_lower_plan(), tc, ain)
    }

    #[inline]
    fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
        fixed_middle_state(exact_fixed_middle_plan(), tc, ain)
    }
}

struct LogQuadratureX4ApproxV1;

impl Sealed for LogQuadratureX4ApproxV1 {}

impl QuadratureProfile for LogQuadratureX4ApproxV1 {
    const LOWER_LOG_STEP: f64 = 0.040;
    const MIDDLE_LOG_STEP: f64 = 0.100;
    const UPPER_LOG_STEP: f64 = 0.300;
    const USE_FIXED_LOWER_PLAN: bool = true;
    const RETIRE_SPECIES_ROUND_TRIP: bool = true;
    const RETIRE_ZR_ROUND_TRIP: bool = false;
    const DLRSL_ZERO_ABOVE_KM: f64 = f64::INFINITY;
    const FITTED_UPPER_SEGMENT: bool = false;
    const USE_FIXED_MIDDLE_PLAN: bool = true;

    #[inline]
    fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
        fixed_lower_state(logquad_x4_fixed_lower_plan(), tc, ain)
    }

    #[inline]
    fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
        fixed_middle_state(logquad_x4_fixed_middle_plan(), tc, ain)
    }
}

/// The coarse-abscissa profile behind `atm_model` 6, i.e. R16's "arm C".
///
/// Identical to [`LogQuadratureX4ApproxV1`] except for two log steps: the
/// middle segment goes 0.100 -> 0.300 (16 -> 6 Boole steps) and the upper
/// 0.300 -> 0.700 (one step at every production altitude). The lower segment is
/// deliberately unchanged, so this profile SHARES v1's lower plan; coarsening
/// that segment costs accuracy and buys nothing, because it lies wholly below
/// the 125 km temperature break and so never takes the arctangent arm.
///
/// # Why this is a new model code and not a redefinition of model 5
///
/// `CompiledPartAScienceV1` hashes `hybrid.atmosphere_model` — the integer —
/// and nothing about the quadrature, and `build_policy_sha256` covers no Rust
/// source at all. Editing v1's constants in place would therefore have moved
/// the physics while every sealed digest stayed byte identical, making receipts
/// from before and after indistinguishable. Allocating code 6 moves the science
/// hash, separates the receipts, and keeps model 5 available for comparison.
///
/// # What it costs and what bounds it
///
/// 6 middle steps is the last rung before a 17x error cliff (R16 §3), and it
/// carries the lowest evaluation count of any rung measured. Measured against
/// the exact profile it is 5.747e-5 worst case over the standing 1800-case
/// lattice, held by `v2_broad_grid_density_error_stays_within_rescoped_bound`
/// at the 1.0e-4 bound the user re-scoped on 2026-08-09.
///
/// **The strict-HF 1.0 m accuracy gates do not bound this profile** and must
/// not be cited as if they did: they difference an arc against the same arc at
/// a tighter tolerance, so a quadrature bias is common-mode and cancels. They
/// stay green at rungs 943x over the density bound. See the R22 addendum in
/// `docs/plans/2026-08-08-r16-atan-abscissa-decision.md`.
struct LogQuadratureX4ApproxV2;

impl Sealed for LogQuadratureX4ApproxV2 {}

impl QuadratureProfile for LogQuadratureX4ApproxV2 {
    const LOWER_LOG_STEP: f64 = LogQuadratureX4ApproxV1::LOWER_LOG_STEP;
    const MIDDLE_LOG_STEP: f64 = 0.300;
    const UPPER_LOG_STEP: f64 = 0.700;
    const USE_FIXED_LOWER_PLAN: bool = true;
    const RETIRE_SPECIES_ROUND_TRIP: bool = true;
    const RETIRE_ZR_ROUND_TRIP: bool = false;
    const DLRSL_ZERO_ABOVE_KM: f64 = f64::INFINITY;
    const FITTED_UPPER_SEGMENT: bool = false;
    const USE_FIXED_MIDDLE_PLAN: bool = true;

    /// Shared with v1 rather than duplicated: `LOWER_LOG_STEP` is the same, so
    /// a second plan would be the same numbers under a second name.
    #[inline]
    fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
        fixed_lower_state(logquad_x4_fixed_lower_plan(), tc, ain)
    }

    #[inline]
    fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
        fixed_middle_state(logquad_x4_v2_fixed_middle_plan(), tc, ain)
    }
}

/// Altitude domain of the R57 upper-segment fit, in km.
///
/// The censused production band is 626.2--985.7 km, so this covers it with room
/// on both sides and the fit is never extrapolated. Outside it the segment
/// walks its Boole step exactly as before, which makes the domain a speed
/// boundary and not a validity one — the same shape as the `Texo` domain on
/// [`LogQuadratureFittedV7`].
///
/// The low end is 500 and not lower because the segment's own geometry is
/// frozen there: `jb_density` forms the step ratio from `altitude.max(500.0)`,
/// so every altitude at or below 500 km produces the same abscissae, the same
/// `dz` of zero, and the same `sum3` of zero.
const UPPER_FIT_ALT_LO: f64 = 500.0;

/// Upper end of the R57 upper-segment fit's altitude domain, in km.
///
/// **This is a ONE-STEP boundary and not a taste.** The fit reproduces a single
/// Boole panel, because that is what the segment walks while
/// `jb_step_count(ln(alt/z) / UPPER_LOG_STEP)` returns 1 — and above
/// `z * exp(UPPER_LOG_STEP)` it returns 2, at which point the walk integrates
/// the same interval with two panels and is a DIFFERENT quadrature that no
/// single-panel fit can follow. With the flown `UPPER_LOG_STEP` of 0.700 and the
/// plan's `z` of 500 that boundary is **1006.876 km**, and 1000 sits inside it
/// with room.
///
/// Measured rather than reasoned: setting this to 1500 put the fit 1.32e-4 away
/// from the walked density at 1500 km — 660x its own truncation prediction —
/// and the two-panel regime was the whole of it. The coupling is pinned by
/// `the_upper_fit_domain_stays_inside_the_one_step_regime` so a later change to
/// `UPPER_LOG_STEP` cannot silently reopen it.
///
/// The censused production band tops out at 985.7 km, so the flown arc is
/// covered with 1.4% of headroom. Above 1000 km the segment walks, which is a
/// speed boundary and not a validity one.
const UPPER_FIT_ALT_HI: f64 = 1000.0;

/// Top of the sealed V3 arc's censused altitude span, in km.
///
/// Named so that the claim below is checked rather than narrated: a fit domain
/// that stopped short of the flown band would measure a lever on a path
/// production does not take, and the assertion is where that would surface.
const CENSUSED_ALTITUDE_CEILING_KM: f64 = 985.7;

const _: () = assert!(
    UPPER_FIT_ALT_HI > CENSUSED_ALTITUDE_CEILING_KM,
    "the upper fit's altitude domain no longer covers the censused production band"
);

/// Lower end of the fitted kernel's exospheric-temperature domain, in kelvin.
///
/// Production reaches 608.9--1627.5 K (R28's census over the strict-HF arc), so
/// `[500, 2600]` covers it with room on both sides. Outside the interval each
/// fitted accessor falls back to walking the real plan, so the domain is a
/// speed boundary and not a validity one -- see [`LogQuadratureFittedV7`].
const FITTED_V7_TEXO_LO: f64 = 500.0;
/// Upper end of the fitted kernel's domain, in kelvin. See [`FITTED_V7_TEXO_LO`].
const FITTED_V7_TEXO_HI: f64 = 2600.0;

/// Degree of the fitted monomial series, and the rung R28's ladder selected.
const FITTED_V7_DEGREE: usize = 14;
/// Coefficient count for [`FITTED_V7_DEGREE`].
const FITTED_V7_TERMS: usize = FITTED_V7_DEGREE + 1;

/// Recover the exospheric temperature from a broadcast temperature profile.
///
/// This inverts the construction in `jb2008_density_with_profile` exactly
/// rather than approximately. There, `tc[0]` is the 125 km transition
/// temperature `Tx` and `tc[2]` is `(T_inf - Tx) / (pi/2)`, so
/// `tc[0] + tc[2] * (pi/2)` reconstructs `T_inf` up to the rounding of one
/// multiply-add. It is NOT a fit and carries no tabulated error.
#[inline]
fn fitted_v7_texo_of(tc: TemperatureBroadcast) -> f64 {
    tc.base.to_array()[0] + tc.amplitude.to_array()[0] * std::f64::consts::FRAC_PI_2
}

/// Map an exospheric temperature onto the fit variable `u` in `[-1, 1]`.
#[inline]
fn fitted_v7_u_of(texo: f64) -> f64 {
    2.0 * (texo - FITTED_V7_TEXO_LO) / (FITTED_V7_TEXO_HI - FITTED_V7_TEXO_LO) - 1.0
}

/// Horner evaluation of a monomial series in `u`, lowest coefficient first.
///
/// Destructured rather than indexed so the bound checks the vectorizer trips
/// over never enter the loop, and `mul_add` throughout because the fit was
/// generated and its residuals measured against a fused evaluation.
#[inline]
fn fitted_v7_horner(coefficients: &[f64; FITTED_V7_TERMS], u: f64) -> f64 {
    if FITTED_V7_ESTRIN {
        return fitted_v7_estrin(coefficients, u);
    }
    let [lower @ .., highest] = coefficients;
    lower
        .iter()
        .rev()
        .fold(*highest, |acc, &c| acc.mul_add(u, c))
}

/// Whether the five fitted scalars are evaluated by Estrin instead of Horner.
///
/// A `const`, so each arm compiles to itself with no runtime flag in the loop —
/// the same discipline `atan_x4_dispatched`'s A/B used, and for the same reason:
/// an atomic or a field read inside the block perturbs the scheduling around the
/// very code being timed.
const FITTED_V7_ESTRIN: bool = true;

/// Stamps THE Estrin evaluation tree for a monomial series, lowest coefficient
/// first. One rule: at each level, adjacent terms pair through
/// `hi.mul_add(power, lo)`, an odd leftover term carries up UNCHANGED to
/// combine at the next level, and the level powers are supplied in order
/// (`x, x^2, x^4, ...`). That single rule reproduces every hand-written body
/// below operation-for-operation — including their differing tail shapes: a
/// `2^k + 1` count folds its top coefficient at the summit
/// (`c8.mul_add(v8, ..)`, `c16.mul_add(v16, ..)`) while the 15-term series'
/// leftover pairs at the second level (`c14.mul_add(u2, a6)`). The density
/// pins and the fitted-segment oracle gate any wrong expansion loudly.
///
/// `mul_add` here reproduces the EXISTING fitted trees (the fits were
/// generated and their residuals measured against fused evaluation); the
/// macro introduces no new fusion and no reassociation.
macro_rules! estrin_body {
    // Terminal: one term left is the series' value (higher powers may remain
    // unconsumed by construction when the count is a power of two).
    ([$term:expr] $(, $powers:expr)*) => { $term };
    // One combine level: consume the head power, pair the terms with it.
    ([$($terms:expr),+], $power:expr $(, $rest:expr)*) => {
        estrin_body!(@pair $power; []; [$($terms),+] $(, $rest)*)
    };
    // Pair the two leading terms with this level's power.
    (@pair $power:expr; [$($done:expr),*]; [$lo:expr, $hi:expr $(, $tail:expr)*] $(, $rest:expr)*) => {
        estrin_body!(@pair $power; [$($done,)* $hi.mul_add($power, $lo)]; [$($tail),*] $(, $rest)*)
    };
    // Odd leftover carries up unchanged.
    (@pair $power:expr; [$($done:expr),*]; [$leftover:expr] $(, $rest:expr)*) => {
        estrin_body!([$($done,)* $leftover] $(, $rest)*)
    };
    // Even level: every term paired.
    (@pair $power:expr; [$($done:expr),*]; [] $(, $rest:expr)*) => {
        estrin_body!([$($done),*] $(, $rest)*)
    };
}

/// Estrin evaluation of the same degree-14 monomial series.
///
/// # What this changes and what it does not
///
/// Horner is 14 dependent `mul_add`s: every one waits on the last, so the chain
/// is 14 FMA latencies deep and no amount of issue width shortens it. Estrin
/// splits the same polynomial into independent pairs and recombines through
/// `u^2`, `u^4` and `u^8`, which costs three extra multiplies and takes the
/// depth to five. The kernel evaluates five of these series per call, in two
/// groups that sit directly on the critical path (`sub2` and `tloc2` before the
/// density, `sub2`, `ain` and `tloc3` inside it), so the depth is what is being
/// bought.
///
/// **This moves bits.** Reassociating a floating-point sum is not an identity,
/// and both arms are FMA-contracted, so the two answers differ at ULP scale.
/// What that costs is bounded by the thing being approximated rather than by a
/// tolerance: these coefficients are a FIT, whose own worst residual against the
/// walked plan is 7.434e-6 (`FITTED_V7_AIN_MIDDLE`). A ~1e-16 reassociation
/// against a 7.4e-6 approximation error is ten orders under the error already
/// accepted, so the accuracy ledger for this change is trivially satisfied and
/// the whole bill is the re-pin.
#[inline]
fn fitted_v7_estrin(coefficients: &[f64; FITTED_V7_TERMS], u: f64) -> f64 {
    let [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14] = *coefficients;
    let u2 = u * u;
    let u4 = u2 * u2;
    let u8 = u4 * u4;

    estrin_body!(
        [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14],
        u,
        u2,
        u4,
        u8
    )
}

// The five fitted per-call scalars, degree 14 on `u` over Texo in [500, 2600] K.
//
// Generated 2026-08-09 by `tools/r28-fit/fit1d.py` (R28) from a 4001-sample
// sweep of the exact fixed-plan walkers; the generator, its input sweep and the
// full D6--D18 ladder are preserved on the r28-audit branch at 1ff42ce, which is
// harvest source and not merged. `max rel err` on each is the residual of THAT
// scalar against the walked plan over the fit sweep, not a density error; the
// density error the profile as a whole carries is the gate's business.
//
// Degree ladder, worst of the five per rung (R28, reproduced here only as
// provenance): d6 9.589e-03, d8 1.653e-03, d10 ~, d12 ~, d14 7.434e-06. The
// composed density error stops improving well before the scalars do because
// model 6's own quadrature bias dominates below ~1e-5 -- which is exactly why
// d14 lands on m6's error to four digits instead of beating it.

/// Lower-segment `sub2`; degree 14, max rel err 2.998e-10.
const FITTED_V7_SUB2_LOWER: [f64; FITTED_V7_TERMS] = [
    2.083_081_303_716_11e1,
    -2.281_319_922_711_041_2e-1,
    1.487_923_922_209_915_5e-1,
    -1.139_133_002_344_331e-1,
    6.719_889_370_881_822e-2,
    -3.298_654_752_303_574_5e-2,
    1.444_726_212_397_402_8e-2,
    -5.996_731_588_024_651e-3,
    2.506_977_417_579_703e-3,
    -1.168_368_279_164_878_4e-3,
    5.317_825_829_414_62e-4,
    -1.236_948_769_467_779e-4,
    4.605_879_140_433_796e-5,
    -1.005_202_176_438_524_2e-4,
    4.843_680_547_921_645_6e-5,
];

/// Lower-segment `tloc2`; degree 14, max rel err 2.881e-13.
const FITTED_V7_TLOC2: [f64; FITTED_V7_TERMS] = [
    2.245_189_279_157_782_5e2,
    8.361_916_544_062_648e0,
    -5.270_934_320_812_583e0,
    3.939_997_041_634_312e0,
    -2.208_846_067_908_349e0,
    9.906_609_315_622_759e-1,
    -3.702_570_377_917_958_6e-1,
    1.186_132_114_284_294e-1,
    -3.324_856_853_041_033e-2,
    8.285_607_849_008_118e-3,
    -1.857_998_341_613_101e-3,
    3.771_053_070_257_530_5e-4,
    -7.049_765_582_078_043e-5,
    1.331_415_410_018_169e-5,
    -2.125_129_463_136_452e-6,
];

/// Middle-segment `sub2`; degree 14, max rel err 3.968e-06.
const FITTED_V7_SUB2_MIDDLE: [f64; FITTED_V7_TERMS] = [
    3.204_004_635_248_918_7e0,
    -1.379_154_303_257_483e0,
    9.727_155_739_203_78e-1,
    -6.869_617_534_038_228e-1,
    4.683_799_627_150_112_6e-1,
    -3.319_992_425_165_518e-1,
    2.336_406_316_366_163_6e-1,
    -7.314_250_955_188_073e-2,
    1.749_607_889_822_053_8e-2,
    -2.270_681_928_512_089_4e-1,
    2.022_219_497_962_747_5e-1,
    1.567_557_390_037_319_6e-1,
    -1.429_246_498_232_326_6e-1,
    -1.064_042_739_372_959_8e-1,
    8.289_447_705_809_763e-2,
];

/// Middle-segment exit `ain`; degree 14, max rel err 7.434e-06 (the worst of
/// the five, and the one the density bound is most sensitive to).
const FITTED_V7_AIN_MIDDLE: [f64; FITTED_V7_TERMS] = [
    5.459_640_467_018_433e-3,
    -3.677_621_602_992_105e-3,
    2.492_908_325_931_464e-3,
    -1.686_218_408_650_384_5e-3,
    1.138_285_167_692_43e-3,
    -8.142_832_736_202_574e-4,
    5.787_786_476_720_627e-4,
    -1.761_805_263_267_051e-4,
    3.778_641_856_952_205_4e-5,
    -5.770_282_618_062_22e-4,
    5.158_819_094_236_952e-4,
    4.021_163_355_002_131_7e-4,
    -3.670_455_396_068_389e-4,
    -2.714_264_714_701_178e-4,
    2.119_348_199_104_432_7e-4,
];

/// Middle-segment `tloc3`; degree 14, max rel err 5.193e-13.
const FITTED_V7_TLOC3: [f64; FITTED_V7_TERMS] = [
    1.543_797_140_782_907_7e3,
    1.039_876_918_772_851e3,
    -4.373_905_347_259_339e0,
    1.148_667_981_316_145_3e-1,
    1.553_639_286_501_853_3e-1,
    -6.297_264_812_830_576e-2,
    7.581_060_757_845_705e-3,
    3.246_438_671_753_597e-3,
    -1.434_297_563_079_094_4e-3,
    -5.114_031_895_849_491e-5,
    2.078_028_551_490_677_8e-4,
    -6.117_111_036_343_099e-5,
    -4.124_441_270_261_577e-6,
    6.126_914_161_375_104e-6,
    -8.797_891_983_987_146e-7,
];

/// Model 7: model 6's quadrature with the two fixed plans replaced by a fit.
///
/// The saving is structural, not a tolerance trade. Above 105 km the lower
/// fixed plan is walked on every call, and above 500 km the middle one is too,
/// yet neither depends on anything that varies within a call except the
/// temperature profile `tc`. And `tc` is not four free numbers: the caller
/// builds all four entries from `exospheric_temperature` alone, so the fixed
/// plans trace a ONE-parameter family in Texo. Latitude, epoch and solar
/// activity reach them only by moving Texo. That is what makes a 1-D fit the
/// right shape here rather than a projection that drops a variable: the only
/// error is polynomial approximation error, and it falls monotonically with
/// degree.
///
/// Each fitted accessor keeps the plan's own geometric constants (`z`, `zend`,
/// `mb2`, `gravl`) and replaces only the quantities the walk integrates. The
/// two altitude preconditions are NOT re-checked here; they are already the
/// call-site conditions on `fixed_lower_state` and `fixed_middle_state` in
/// `jb_density`, and duplicating them here would be a second copy of a
/// predicate that can drift.
///
/// Outside `[FITTED_V7_TEXO_LO, FITTED_V7_TEXO_HI]` both accessors walk the
/// real plan, so the profile is defined wherever model 6 is and degrades to
/// model 6's exact bits rather than to an extrapolated polynomial.
///
/// Bounded by `v7_broad_grid_density_error_stays_within_rescoped_bound` at the
/// same 1.0e-4 the user re-scoped for model 6 on 2026-08-09, poison-proved by
/// `the_density_bound_rejects_the_degree_8_fit`. As with model 6, **the
/// strict-HF 1.0 m accuracy gates do not bound this profile** -- they
/// difference an arc against the same arc at a tighter tolerance, so a
/// quadrature bias is common-mode and cancels.
struct LogQuadratureFittedV7;

impl Sealed for LogQuadratureFittedV7 {}

impl QuadratureProfile for LogQuadratureFittedV7 {
    const LOWER_LOG_STEP: f64 = LogQuadratureX4ApproxV2::LOWER_LOG_STEP;
    const MIDDLE_LOG_STEP: f64 = LogQuadratureX4ApproxV2::MIDDLE_LOG_STEP;
    const UPPER_LOG_STEP: f64 = LogQuadratureX4ApproxV2::UPPER_LOG_STEP;
    const USE_FIXED_LOWER_PLAN: bool = true;
    const RETIRE_SPECIES_ROUND_TRIP: bool = true;
    const RETIRE_ZR_ROUND_TRIP: bool = true;
    const DLRSL_ZERO_ABOVE_KM: f64 = 800.0;
    const FITTED_UPPER_SEGMENT: bool = false;
    const USE_FIXED_MIDDLE_PLAN: bool = true;

    #[inline]
    fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
        let plan = logquad_x4_fixed_lower_plan();
        let texo = fitted_v7_texo_of(tc);
        if !(FITTED_V7_TEXO_LO..=FITTED_V7_TEXO_HI).contains(&texo) {
            return fixed_lower_state(plan, tc, ain);
        }
        let u = fitted_v7_u_of(texo);
        LowerState {
            sub2: fitted_v7_horner(&FITTED_V7_SUB2_LOWER, u),
            z: plan.z,
            zend: plan.zend,
            mb2: plan.mb2,
            tloc2: fitted_v7_horner(&FITTED_V7_TLOC2, u),
            gravl: plan.gravl,
        }
    }

    #[inline]
    fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
        let plan = logquad_x4_v2_fixed_middle_plan();
        let texo = fitted_v7_texo_of(tc);
        if !(FITTED_V7_TEXO_LO..=FITTED_V7_TEXO_HI).contains(&texo) {
            return fixed_middle_state(plan, tc, ain);
        }
        let u = fitted_v7_u_of(texo);
        MiddleState {
            sub2: fitted_v7_horner(&FITTED_V7_SUB2_MIDDLE, u),
            ain: fitted_v7_horner(&FITTED_V7_AIN_MIDDLE, u),
            tloc3: fitted_v7_horner(&FITTED_V7_TLOC3, u),
            z: plan.z,
            zend: plan.zend,
        }
    }
}

/// Immutable local-input surface for Orekit-compatible JB2008 arithmetic.
///
/// `mjd_utc` is UTC Modified Julian Date, exactly `JD_UTC - 2_400_000.5`.
/// Declinations/latitude are radians in `[-π/2, π/2]`. Altitude is metres.
///
/// # Why an hour angle and not two right ascensions
///
/// This surface used to take `sun_ra_rad` and `sat_ra_rad` separately. It never
/// used either one: the kernel's only consumer was the DIFFERENCE
/// `h = sat_ra - sun_ra`, the satellite's hour angle relative to the Sun. Two
/// right ascensions cost the caller two `atan2`, where the difference is one
/// `atan2` of a cross/dot pair, and the caller was then charged two
/// [`wrap_to_tau`] calls to normalize values that were about to be subtracted.
///
/// Taking `h` directly moves the whole redundancy out of the hot path. The field
/// is normalized internally to `[0, 2π)`, so a caller may pass any finite angle
/// and any whole-turn offset is absorbed --
/// `hour_angle_normalizes_to_zero_through_two_pi` keeps that promise checked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jb2008Input {
    pub mjd_utc: f64,
    pub sun_declination_rad: f64,
    /// Satellite hour angle relative to the Sun, radians. Any finite value; the
    /// kernel normalizes it to `[0, 2π)` itself.
    pub hour_angle_rad: f64,
    pub sat_geocentric_lat_rad: f64,
    pub sat_altitude_m: f64,
    pub f10: f64,
    pub f10b: f64,
    pub s10: f64,
    pub s10b: f64,
    pub m10: f64,
    pub m10b: f64,
    pub y10: f64,
    pub y10b: f64,
    pub dst_temperature_correction_k: f64,
}

/// Input or numerical-domain rejection. Callers must not substitute a model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Jb2008Error {
    NonFiniteInput,
    AltitudeOutOfRange,
    AngleOutOfRange,
    NonPositiveSolarIndex,
    NumericalDomain,
}

/// Pure, allocation-free f64 JB2008 kernel.
///
/// Input activity values must already carry JB2008 prescribed lags/averages.
/// No data loading, interpolation, input-result caching, or fallback exists
/// here. The log-quadrature approximation initializes one immutable plan for
/// altitude-independent lower integration geometry.
///
/// # Errors
///
/// Returns [`Jb2008Error`] when an input or an intermediate physical quantity
/// is outside the validated kernel domain.
pub fn jb2008_density(input: Jb2008Input) -> Result<f64, Jb2008Error> {
    jb2008_density_with_profile::<ExactOrekitQuadrature>(input)
}

/// Fixed x4 log-quadrature approximation. Never use as exact model4.
///
/// # Errors
///
/// Returns [`Jb2008Error`] when an input or an intermediate physical quantity
/// is outside the validated kernel domain.
pub fn jb2008_density_logquad_x4_approx_v1(input: Jb2008Input) -> Result<f64, Jb2008Error> {
    jb2008_density_with_profile::<LogQuadratureX4ApproxV1>(input)
}

/// Coarse-abscissa x4 log-quadrature approximation (`atm_model` 6). Never use
/// as exact model4, and never as model5 — it is a THIRD density, not a faster
/// spelling of the second.
///
/// # Errors
///
/// Returns [`Jb2008Error`] when an input or an intermediate physical quantity
/// is outside the validated kernel domain.
pub fn jb2008_density_logquad_x4_approx_v2(input: Jb2008Input) -> Result<f64, Jb2008Error> {
    jb2008_density_with_profile::<LogQuadratureX4ApproxV2>(input)
}

/// Fitted-kernel approximation (`atm_model` 7).
///
/// Model 6's quadrature with the two fixed plans replaced by a degree-14 fit in
/// the exospheric temperature. Never use as exact model4, and never as model5
/// or model6 — it is a FOURTH density. See [`LogQuadratureFittedV7`] for why
/// the fit is one-dimensional and what happens outside its temperature domain.
///
/// # Errors
///
/// Returns [`Jb2008Error`] when an input or an intermediate physical quantity
/// is outside the validated kernel domain.
pub fn jb2008_density_fitted_v7(input: Jb2008Input) -> Result<f64, Jb2008Error> {
    jb2008_density_with_profile::<LogQuadratureFittedV7>(input)
}

/// `x.rem_euclid(TAU)`, bit-for-bit, without the `fmod` call on the two
/// branches production actually takes.
///
/// `f64::rem_euclid` is `t = fmod(self, rhs); if t < 0.0 { t + rhs.abs() } else
/// { t }`, and on this target that first step is a genuine `bl _fmod` — a libm
/// call on the density hot path, twice per evaluation. It is avoidable because
/// `fmod(x, TAU)` returns `x` *exactly*, with its own sign, whenever
/// `|x| < TAU`, which collapses the whole thing to the identity on `[0, TAU)`
/// and to a single addition on `(-TAU, 0)` — the SAME addition `rem_euclid`
/// would have performed on the same operands, so the same rounding.
///
/// This is a bit-identity over the ENTIRE `f64` domain, not a range assumption:
/// everything the two guards do not cover — `±TAU` and beyond, `±inf`, `NaN` —
/// falls through to `rem_euclid` itself. `-TAU` is deliberately outside the
/// second guard: `fmod(-TAU, TAU)` is `-0.0`, which `rem_euclid` returns
/// unchanged (`-0.0 < 0.0` is false), whereas `-TAU + TAU` is `+0.0`. `-0.0`
/// and `NaN` likewise take the paths that reproduce `rem_euclid`'s signed-zero
/// and unordered-compare behaviour rather than a re-derivation of it.
///
/// The production callers pass `atan2` results (`sun_itrs[1].atan2(...)` and
/// `pos_itrs[1].atan2(...)` in `lightyear_odeint_rs::rhs`), so the argument is
/// in `[-PI, PI]` and one of the two fast branches always takes it. The
/// fallback is retained because this is a `pub` kernel whose right ascensions
/// are caller-supplied and unvalidated, not because the fallback is expected
/// to fire.
#[inline]
fn wrap_to_tau(x: f64) -> f64 {
    if CEILING_PROBE == 6 {
        return x;
    }
    if WRAP_TO_SIGNED_PI {
        return wrap_to_signed_pi(x);
    }
    if (0.0..std::f64::consts::TAU).contains(&x) {
        x
    } else if x < 0.0 && x > -std::f64::consts::TAU {
        x + std::f64::consts::TAU
    } else {
        x.rem_euclid(std::f64::consts::TAU)
    }
}

/// A/B arm selector for the glibc argument-reduction lever (M1, `[-π, π)` wrap).
///
/// Companion of `satpy_core`'s `WRAP_TO_SIGNED_PI`; flip both together. glibc
/// charges +34% for a `sin` once `|x| > 2.426265` (`docs/PMU_PROFILE.md` §7),
/// and [`wrap_to_tau`] pushes this kernel's hour angle across that line by
/// construction: the production callers hand it an `atan2` result already in
/// `[-π, π]`, and the wrap's only effect on them is to ADD `τ` to the negative
/// half. BIT-MOVER, never committed `true`.
///
/// **MEASURED AND CLOSED at ~0.35% of arc, 2026-08-11.** The verdict, the three
/// schedules behind it and the bill that outweighs it are recorded once, on
/// `satpy_core`'s companion const and in `docs/PMU_PROFILE.md` §10.6.
const WRAP_TO_SIGNED_PI: bool = false;

/// [`wrap_to_tau`], re-targeted to `[-π, π)`.
///
/// Every downstream consumer of the wrapped hour angle `h` is invariant under
/// `h → h − τ`, which is why this arm is a bit-mover and not a model change:
/// `(h + 0.750_491_58).sin()` has period `τ`; `tau` shifts by exactly `τ` and
/// reaches `jb_tsub_l` only as `(0.5·tau).cos().abs()`, where the half-angle's
/// sign flip is removed by the `abs`; and `solar_time_hour` already carries the
/// `±24` guards that make it agree.
#[inline]
fn wrap_to_signed_pi(x: f64) -> f64 {
    if (-std::f64::consts::PI..=std::f64::consts::PI).contains(&x) {
        x
    } else {
        let t = x.rem_euclid(std::f64::consts::TAU);
        if t >= std::f64::consts::PI {
            t - std::f64::consts::TAU
        } else {
            t
        }
    }
}

#[inline]
fn jb2008_density_with_profile<P: QuadratureProfile>(
    input: Jb2008Input,
) -> Result<f64, Jb2008Error> {
    if CEILING_PROBE != 4 {
        validate(input)?;
    }
    let ThermalState {
        altitude_km,
        exospheric_temperature,
        temperature_profile,
        sin_geocentric_latitude,
    } = thermal_state(input);
    let rho_kg_m3 = jb_density::<P>(
        input,
        altitude_km,
        temperature_profile,
        exospheric_temperature,
        sin_geocentric_latitude,
    )?;
    if !(rho_kg_m3.is_finite() && rho_kg_m3 > 0.0) {
        return Err(Jb2008Error::NumericalDomain);
    }
    Ok(rho_kg_m3)
}

/// Every input gate the kernel applies, in the order it applies them.
///
/// Bound as its own function for the same reason as [`thermal_state`]: it is a
/// distinct, separately priceable stage of the evaluation. (The instrument that
/// priced it, `jb2008::cost_map`, was retired 2026-08-21; the split stays,
/// because unbinding it would change codegen on a pinned kernel.) The order is
/// load-bearing and is NOT
/// an implementation detail — a caller that reaches this code with several
/// defects at once is told about the first one in this sequence, and
/// `lightyear_odeint_rs`'s exospheric-ceiling skip replicates the
/// `NonPositiveSolarIndex` arm specifically, on the strength of having censused
/// which arms can fire above 2500 km.
///
/// `inline(always)`, not `inline`: see [`ThermalState`].
#[expect(
    clippy::inline_always,
    reason = "a plain #[inline] here is a measured +10.7% on the kernel call; see ThermalState"
)]
#[inline(always)]
fn validate(input: Jb2008Input) -> Result<(), Jb2008Error> {
    if ![
        input.mjd_utc,
        input.sun_declination_rad,
        input.hour_angle_rad,
        input.sat_geocentric_lat_rad,
        input.sat_altitude_m,
        input.f10,
        input.f10b,
        input.s10,
        input.s10b,
        input.m10,
        input.m10b,
        input.y10,
        input.y10b,
        input.dst_temperature_correction_k,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        return Err(Jb2008Error::NonFiniteInput);
    }
    if input.sat_altitude_m < JB_ALTITUDE_MIN_M {
        return Err(Jb2008Error::AltitudeOutOfRange);
    }
    if !(-std::f64::consts::FRAC_PI_2..=std::f64::consts::FRAC_PI_2)
        .contains(&input.sun_declination_rad)
        || !(-std::f64::consts::FRAC_PI_2..=std::f64::consts::FRAC_PI_2)
            .contains(&input.sat_geocentric_lat_rad)
    {
        return Err(Jb2008Error::AngleOutOfRange);
    }
    if [
        input.f10, input.f10b, input.s10, input.s10b, input.m10, input.m10b, input.y10, input.y10b,
    ]
    .iter()
    .any(|index| *index <= 0.0)
    {
        return Err(Jb2008Error::NonPositiveSolarIndex);
    }
    Ok(())
}

/// Everything [`jb_density`] needs from the input beyond the input itself.
///
/// This is the whole of the kernel between the input gates and the quadrature,
/// and it is one dependency chain: hour angle to local solar time to subsolar
/// temperature to exospheric temperature to the four-entry temperature profile.
/// Bound as one function so that the two halves of the kernel can be priced
/// separately, without a paraphrase of either standing in for the real one. The
/// instrument that timed this and [`jb_density`] against the whole,
/// `jb2008::cost_map`, was retired 2026-08-21; the split stays, because
/// unbinding it would change codegen on a pinned kernel.
///
/// # `inline(always)` is load-bearing, and it was measured
///
/// Splitting this out under a plain `#[inline]` cost **+10.7%** on the kernel
/// call (251.8 -> 278.8 ns, `jb2008_cost_map` on production codegen, min of 400
/// passes; measured 2026-08 while the retired `cost-map` arm still existed, so
/// "feature off" in the original note means what is now the only arm):
/// LLVM declined the hint, emitted a real call, and returned the four fields
/// through an `sret` stack slot that the caller immediately reloaded. A cost
/// map that makes the thing it measures 10% slower is not a cost map. With
/// `inline(always)` the price returns to the pre-split figure, which is the
/// condition this refactor has to meet: it is a naming of an existing stage,
/// not a change to one.
struct ThermalState {
    altitude_km: f64,
    exospheric_temperature: f64,
    temperature_profile: [f64; 4],
    sin_geocentric_latitude: f64,
}

#[expect(
    clippy::inline_always,
    reason = "a plain #[inline] here is a measured +10.7% on the kernel call; see ThermalState"
)]
#[inline(always)]
fn thermal_state(input: Jb2008Input) -> ThermalState {
    let altitude_km = input.sat_altitude_m / 1000.0;
    let tsubc = jb_tsubc(input);
    let eta = 0.5 * (input.sat_geocentric_lat_rad - input.sun_declination_rad).abs();
    let theta = 0.5 * (input.sat_geocentric_lat_rad + input.sun_declination_rad).abs();
    let h = wrap_to_tau(input.hour_angle_rad);
    let tau = h - 0.645_771_82 + 0.104_719_76 * (h + 0.750_491_58).sin();
    let mut solar_time_hour = (h + std::f64::consts::PI).to_degrees() / 15.0;
    if solar_time_hour >= 24.0 {
        solar_time_hour -= 24.0;
    } else if solar_time_hour < 0.0 {
        solar_time_hour += 24.0;
    }
    // ONE `sin_cos` where the kernel used to make two separate libm calls, in
    // two different functions, on the same argument: `jb_dtc` took the cosine
    // of the satellite latitude and `jb_dlrsl` the sine of it. They sit either
    // side of the whole quadrature, so LLVM never paired them itself.
    //
    // What `sin_cos` buys is PLATFORM-DEPENDENT, and on BOTH production
    // platforms it is a fused kernel and not merely better scheduling. On
    // Darwin it lowers to one `__sincos_stret`. On glibc x86-64 it lowers to
    // `sincos`, an IFUNC that hands an FMA machine `__sincos_fma` — measured
    // 2026-08-11 at 6.20% of the whole production arc on znver2, the largest
    // single libm routine on it (`docs/ARC_COST_MAP.md` §7.3).
    //
    // This comment previously said the opposite — "on glibc it does NOT fuse" —
    // on the strength of `PMU_PROFILE.md` §7, whose microbenchmark measures a
    // hand-written `sin(x) + cos(x)` and therefore really does make two calls.
    // R55's census agreed with it, from an instrument that is not in the tree.
    // Both were reasoning about a source shape this file does not compile to.
    // The bit-identity against the separate calls holds on every lowering and is
    // pinned in this module's tests rather than assumed.
    let (sin_geocentric_latitude, cos_geocentric_latitude) = if CEILING_PROBE == 3 {
        (
            input.sat_geocentric_lat_rad * 0.5,
            input.sat_geocentric_lat_rad * 0.25,
        )
    } else {
        input.sat_geocentric_lat_rad.sin_cos()
    };
    let local_subsolar_temperature = jb_tsub_l(eta, theta, tau, tsubc);
    let exospheric_temperature = local_subsolar_temperature
        + input.dst_temperature_correction_k
        + jb_dtc(
            input.f10,
            solar_time_hour,
            cos_geocentric_latitude,
            altitude_km,
        );
    let transition_temperature = 444.3807 + 0.02385 * exospheric_temperature
        - 392.8292 * (-0.002_135_7 * exospheric_temperature).exp();
    let transition_gradient = 0.054_285_714 * (transition_temperature - 183.0);
    // `tc[3]` IS `tc[1] / tc[2]`, and `tc[2]`'s expression was spelled out a
    // second time to say so. This binds it once; the divisor is the same value
    // it always was, so the profile is unchanged. It is NOT a typo: the
    // Jacchia-Bowman profile above 125 km is
    // `T = tc[0] + tc[2] * atan(tc[3] * dz * (1 + 4.5e-6 * dz^2.5))` with
    // `tc[2] = (T_inf - Tx) / (pi/2)` and `tc[3] = Tx' / tc[2]`, which is what
    // `jb_local_temp`'s upper branch evaluates — and every Orekit JAR vector
    // from 120 km up in this module's tests exercises exactly that branch.
    let transition_amplitude =
        (exospheric_temperature - transition_temperature) / std::f64::consts::FRAC_PI_2;
    let temperature_profile = [
        transition_temperature,
        transition_gradient,
        transition_amplitude,
        transition_gradient / transition_amplitude,
    ];
    ThermalState {
        altitude_km,
        exospheric_temperature,
        temperature_profile,
        sin_geocentric_latitude,
    }
}

/// Last `(key, tsubc)` this thread computed, where `key` is the exact bits of
/// the eight solar indices [`jb_tsubc`] reads.
///
/// The sentinel is all-zero bits, i.e. `+0.0` in all eight slots, which cannot
/// collide with a live key: `jb2008_density` rejects any index `<= 0.0` before
/// [`jb_tsubc`] is reached, so a real key always has eight strictly positive
/// values.
#[expect(
    clippy::declare_interior_mutable_const,
    reason = "a Cell of Copy scalars is the point; nothing here is shared across threads"
)]
const TSUBC_MEMO_EMPTY: std::cell::Cell<([u64; 8], f64)> = std::cell::Cell::new(([0; 8], 0.0));

thread_local! {
    static TSUBC_MEMO: std::cell::Cell<([u64; 8], f64)> = const { TSUBC_MEMO_EMPTY };
}

/// Base subsolar temperature, memoized on the exact bits of its own inputs.
///
/// This carries the kernel's only `powf` and it is a pure function of eight
/// solar indices that are constant for a whole UTC day, so a strict-HF run
/// recomputes it once per RHS evaluation for nothing: **3.27e8 `powf` calls in
/// a 4-design x 2-event census, collapsing to one per UTC day per thread.**
///
/// **Keyed on the input BITS, not on the day.** Keying on time was the obvious
/// design and it is wrong: `Jb2008DriverInput`'s ninth field, `dtcval`, is NOT
/// day- or even hour-constant — `drivers::interpolate_dtc` interpolates
/// linearly between hourly nodes and rounds, so it takes about `|Δ| + 1`
/// distinct values inside one hour, and over the shipped `DTCFILE.TXT` that
/// `|Δ|` runs to p90 26 and max 244. A time-keyed memo over the whole driver
/// tuple would serve a stale `dtcval` and silently move the density. `jb_tsubc`
/// happens to read none of that — only the eight indices below — so keying on
/// exactly what it reads is both bit-exact and immune to the trap.
///
/// Bit-exactness is by construction: identical input bits return the value
/// previously computed from those same bits. `strict_hf_pin` and `rect_loop_pin`
/// are the guards.
#[inline]
fn jb_tsubc(input: Jb2008Input) -> f64 {
    if CEILING_PROBE == 5 {
        // Lands within a kelvin of the sealed driver day's real 783.1 K, so the
        // fitted profile's [500, 2600] K domain guard still takes the arm
        // production takes. A probe that fell out of the fit's domain would be
        // timing the walked plan instead.
        return 2.85_f64.mul_add(input.f10b, 392.4);
    }
    let key = [
        input.f10.to_bits(),
        input.f10b.to_bits(),
        input.s10.to_bits(),
        input.s10b.to_bits(),
        input.m10.to_bits(),
        input.m10b.to_bits(),
        input.y10.to_bits(),
        input.y10b.to_bits(),
    ];
    TSUBC_MEMO.with(|memo| {
        let (cached_key, cached_value) = memo.get();
        if cached_key == key {
            return cached_value;
        }
        let value = jb_tsubc_uncached(input);
        memo.set((key, value));
        value
    })
}

/// The arithmetic itself. Separated so the memo above cannot be mistaken for
/// part of the model, and so tests can exercise both paths.
fn jb_tsubc_uncached(input: Jb2008Input) -> f64 {
    let fn_ = (input.f10b / 240.0).powf(0.25).min(1.0);
    let fsb = input.f10b * fn_ + input.s10b * (1.0 - fn_);
    392.4
        + 3.227 * fsb
        + 0.298 * (input.f10 - input.f10b)
        + 2.259 * (input.s10 - input.s10b)
        + 0.312 * (input.m10 - input.m10b)
        + 0.178 * (input.y10 - input.y10b)
}

#[inline]
fn jb_tsub_l(eta: f64, theta: f64, tau: f64, tsubc: f64) -> f64 {
    let cos_eta = jb_positive_five_halves(eta.cos());
    let sin_theta = jb_positive_five_halves(theta.sin());
    let cos_tau = (0.5 * tau).cos().abs();
    let df = sin_theta + (cos_eta - sin_theta) * cos_tau.powi(3);
    tsubc * (1.0 + 0.31 * df)
}

fn jb_dtc(f10: f64, solar_time_hour: f64, cos_geocentric_latitude: f64, altitude_km: f64) -> f64 {
    // OUTSIDE THE BANDED RANGE FIRST, and before the three scalars below.
    //
    // Above 800 km and below 120 km the correction is 0.0 by the model's own
    // definition, and the censused production band is 626.2--985.7 km, so
    // roughly half of every flown evaluation lands here. The shipped chain
    // reached that constant by failing five range tests in order — ten
    // compares — having already paid two divides for `st` and `fs`, which the
    // zero arm never reads. `NaN` takes this arm too, exactly as it took the
    // final `else` before, because `contains` is false for it.
    if !(120.0..=800.0).contains(&altitude_km) {
        return 0.0;
    }
    let st = solar_time_hour / 24.0;
    let cs = cos_geocentric_latitude;
    let fs = (f10 - 100.0) / 100.0;
    // The bands, from the top, each a SINGLE compare because the guard above
    // has already established `120 <= altitude <= 800`.
    //
    // # Why reversing the chain is exact and not merely equivalent
    //
    // The five bands share their endpoints on purpose: 200 belongs to both
    // `[120, 200]` and `[200, 240]`, and likewise at 240, 300 and 600. Walking
    // them upwards, as the shipped chain did, resolves every shared endpoint to
    // the LOWER band. Walking them downwards with STRICT lower bounds resolves
    // it the same way — `altitude > 600` is false at exactly 600, so 600 falls
    // through to the 300--600 arm, which is where the upward chain put it.
    //
    // That is the whole of the argument, and `jb_dtc_band_order_is_bit_identical_
    // to_the_upward_chain` is the evidence: it sweeps both spellings across
    // every band, every shared endpoint, both ULP neighbours of each, and the
    // out-of-range and NaN arms.
    if altitude_km > 600.0 {
        let poly2 = jb_poly2_bdtc(st);
        let aa = jb_poly1_bdtc(fs, st, cs, 6.0 * poly2);
        let bb = cs * poly2;
        let cc = -(3.0 * aa + 4.0 * bb) / 4.0;
        let dd = (aa + bb) / 4.0;
        let zp = (altitude_km - 600.0) / 100.0;
        aa + zp * (bb + zp * (cc + zp * dd))
    } else if altitude_km > 300.0 {
        jb_poly1_bdtc(fs, st, cs, altitude_km * jb_poly2_bdtc(st) / 100.0)
    } else if altitude_km > 240.0 {
        let bb = jb_poly1_cdtc(fs, st, cs);
        let aa = 0.8 * bb + jb_poly2_cdtc(fs, st, cs);
        let p2bdt = jb_poly2_bdtc(st);
        let dtc300 = jb_poly1_bdtc(fs, st, cs, 3.0 * p2bdt);
        let dtc300dz = cs * p2bdt;
        let cc = 3.0 * dtc300 - dtc300dz - 3.0 * aa - 2.0 * bb;
        let dd = dtc300 - aa - bb - cc;
        let zp = (altitude_km - 240.0) / 60.0;
        aa + zp * (bb + zp * (cc + zp * dd))
    } else if altitude_km > 200.0 {
        jb_poly1_cdtc(fs, st, cs) * (altitude_km - 200.0) / 50.0 + jb_poly2_cdtc(fs, st, cs)
    } else {
        let dtc200 = jb_poly2_cdtc(fs, st, cs);
        let dtc200dz = jb_poly1_cdtc(fs, st, cs);
        let cc = 3.0 * dtc200 - dtc200dz;
        let dd = dtc200 - cc;
        let zp = (altitude_km - 120.0) / 80.0;
        zp * zp * (cc + dd * zp)
    }
}

#[inline]
fn jb_poly1_cdtc(fs: f64, st: f64, cs: f64) -> f64 {
    let [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, ..] = JB_CDT;
    c0 + fs * (c1 + st * (c2 + st * (c3 + st * (c4 + st * (c5 + st * c6)))))
        + cs * st * (c7 + st * (c8 + st * (c9 + st * (c10 + st * c11))))
        + cs * (c12 + fs * (c13 + st * (c14 + st * c15)))
}

#[inline]
fn jb_poly2_cdtc(fs: f64, st: f64, cs: f64) -> f64 {
    let [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, c16, c17, c18, c19, c20, c21, c22] =
        JB_CDT;
    c16 + st * cs * (c17 + st * (c18 + st * c19)) + fs * cs * (c20 + st * (c21 + st * c22))
}

#[inline]
fn jb_poly1_bdtc(fs: f64, st: f64, cs: f64, hp: f64) -> f64 {
    let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, _, _, _, _, _, _, b18] = JB_BDT;
    b0 + fs * (b1 + st * (b2 + st * (b3 + st * (b4 + st * (b5 + st * b6)))))
        + cs * (st * (b7 + st * (b8 + st * (b9 + st * (b10 + st * b11)))) + hp + b18)
}

#[inline]
fn jb_poly2_bdtc(st: f64) -> f64 {
    let [_, _, _, _, _, _, _, _, _, _, _, _, b12, b13, b14, b15, b16, b17, _] = JB_BDT;
    b12 + st * (b13 + st * (b14 + st * (b15 + st * (b16 + st * b17))))
}

#[derive(Clone, Copy)]
struct LowerState {
    sub2: f64,
    z: f64,
    zend: f64,
    mb2: f64,
    tloc2: f64,
    gravl: f64,
}

#[derive(Clone, Copy)]
struct FixedLowerStep {
    dz: f64,
    temperature_basis: [f64; 4],
    mb_gravity: [f64; 4],
}

impl FixedLowerStep {
    const ZERO: Self = Self {
        dz: 0.0,
        temperature_basis: [0.0; 4],
        mb_gravity: [0.0; 4],
    };
}

struct FixedLowerPlan<const N: usize> {
    steps: [FixedLowerStep; N],
    z: f64,
    zend: f64,
    mb2: f64,
    gravl: f64,
}

/// Everything the 105--500 km segment hands to the segment above it.
#[derive(Clone, Copy)]
struct MiddleState {
    sub2: f64,
    /// The last lane integrand, which the upper segment's first Boole weight
    /// consumes. Carried explicitly because the scalar loop carried it in a
    /// mutable binding that outlived the loop.
    ain: f64,
    tloc3: f64,
    z: f64,
    zend: f64,
}

/// One Boole step of the 105--500 km segment with its geometry already folded.
///
/// The three temperature fields are the altitude-dependent halves of the two
/// `jb_local_temp` branches, split exactly where `tc` enters:
///
/// * `temperature_basis` is `((C2 dz - C1) dz dz + 1) dz` — everything the
///   sub-125 km polynomial evaluates before it reaches `tc.gradient`.
/// * `break_offset` and `argument_shape` are `dz` and `1 + 4.5e-6 dz^2.5`, the
///   two factors that sit either side of `tc.argument_scale` in the arctangent
///   argument. They stay SEPARATE on purpose. `jb_local_temp_above_break_x4`
///   computes `tc.argument_scale * dz * (1 + ...)`, which associates left, so
///   folding `dz * (1 + ...)` into one stored product would reassociate the
///   multiply and move the argument by up to an ULP. Two stored factors and
///   two run-time multiplies reproduce the original operation exactly, and
///   still retire the `sqrt` that produced `dz^2.5`.
#[derive(Clone, Copy)]
struct FixedMiddleStep {
    dz: f64,
    temperature_basis: wide::f64x4,
    break_offset: wide::f64x4,
    argument_shape: wide::f64x4,
    gravity: wide::f64x4,
}

/// Fixed-geometry 105--500 km plan, split at the 125 km crossing.
///
/// `z` is monotone increasing across the segment, so the steps that need the
/// sub-125 km branch are a PREFIX and the rest are all above the break. The
/// split is stored rather than tested per step: `jb_local_temp_step_x4`'s two
/// scalar compares were themselves per-step work, and with fixed geometry the
/// arm each step takes is a property of the plan.
///
/// `straddle` is the at-most-one step whose four abscissae bracket 125 km,
/// together with the `dz <= 0` mask `jb_local_temp_x4` would have computed. It
/// is `None` when the crossing falls between two steps.
struct FixedMiddlePlan<const N: usize> {
    steps: [FixedMiddleStep; N],
    /// Index of the first step wholly above 125 km.
    above_from: usize,
    /// `(index, mask)` of the straddling step, which is `above_from - 1` when
    /// it exists.
    straddle: Option<(usize, wide::f64x4)>,
    z: f64,
    zend: f64,
}

impl FixedMiddleStep {
    const ZERO: Self = Self {
        dz: 0.0,
        temperature_basis: wide::f64x4::new([0.0; 4]),
        break_offset: wide::f64x4::new([0.0; 4]),
        argument_shape: wide::f64x4::new([0.0; 4]),
        gravity: wide::f64x4::new([0.0; 4]),
    };
}

/// `tc` broadcast to `f64x4` once per evaluation instead of once per use.
///
/// The `const`-item rule documented on `wide_const` covers literal vectors; a
/// broadcast of a runtime `f64` is a different instruction and a different
/// waste. Written inline in the temperature helpers, `f64x4::new([tc[1]; 4])`
/// and friends were rebuilt three to five times per Boole step across the ~79
/// steps of one evaluation, and every one of them was loop-invariant: `tc` is
/// fixed for the whole quadrature. Carrying the four broadcasts is bit-neutral
/// by construction — the lanes hold the same `tc` components either way.
#[derive(Clone, Copy)]
struct TemperatureBroadcast {
    /// `tc[0]`, the temperature at 125 km.
    base: wide::f64x4,
    /// `tc[1]`, the gradient used below 125 km.
    gradient: wide::f64x4,
    /// `tc[2]`, the arctangent amplitude used above 125 km.
    amplitude: wide::f64x4,
    /// `tc[3]`, the arctangent argument scale used above 125 km.
    argument_scale: wide::f64x4,
}

impl TemperatureBroadcast {
    #[inline]
    const fn new(tc: [f64; 4]) -> Self {
        let [base, gradient, amplitude, argument_scale] = tc;
        Self {
            base: wide::f64x4::new([base; 4]),
            gradient: wide::f64x4::new([gradient; 4]),
            amplitude: wide::f64x4::new([amplitude; 4]),
            argument_scale: wide::f64x4::new([argument_scale; 4]),
        }
    }
}

fn jb_step_count(log_interval_ratio: f64) -> Result<u32, Jb2008Error> {
    if !(log_interval_ratio.is_finite() && log_interval_ratio >= 0.0) {
        return Err(Jb2008Error::NumericalDomain);
    }
    let Some(whole_steps) = log_interval_ratio.floor().to_u32() else {
        return Err(Jb2008Error::NumericalDomain);
    };
    whole_steps
        .checked_add(1)
        .ok_or(Jb2008Error::NumericalDomain)
}

/// Step counts of the two fixed lower plans.
///
/// `jb_step_count(ln(105/90) / LOWER_LOG_STEP)`: 16 at the exact profile's
/// 0.010 and 4 at the approximation's 0.040. They are written out because the
/// step count sizes an array and `jb_step_count` is not a `const fn`;
/// `fixed_lower_plan_step_counts_track_the_log_steps` and the per-plan bit
/// corpora fail loudly if a log step ever moves away from them.
const EXACT_FIXED_LOWER_STEPS: usize = 16;
const LOGQUAD_X4_FIXED_LOWER_STEPS: usize = 4;

static EXACT_FIXED_LOWER_PLAN: std::sync::OnceLock<FixedLowerPlan<EXACT_FIXED_LOWER_STEPS>> =
    std::sync::OnceLock::new();

static LOGQUAD_X4_FIXED_LOWER_PLAN: std::sync::OnceLock<
    FixedLowerPlan<LOGQUAD_X4_FIXED_LOWER_STEPS>,
> = std::sync::OnceLock::new();

fn exact_fixed_lower_plan() -> &'static FixedLowerPlan<EXACT_FIXED_LOWER_STEPS> {
    EXACT_FIXED_LOWER_PLAN.get_or_init(build_fixed_lower_plan::<EXACT_FIXED_LOWER_STEPS>)
}

fn logquad_x4_fixed_lower_plan() -> &'static FixedLowerPlan<LOGQUAD_X4_FIXED_LOWER_STEPS> {
    LOGQUAD_X4_FIXED_LOWER_PLAN.get_or_init(build_fixed_lower_plan::<LOGQUAD_X4_FIXED_LOWER_STEPS>)
}

/// Step counts of the two fixed middle plans, sized the same way and pinned by
/// the same kind of test as the lower ones: `jb_step_count(ln(500 / z_exit) /
/// MIDDLE_LOG_STEP)`, where `z_exit` is the lower plan's exit abscissa. 63 at
/// the exact profile's 0.025 and 16 at the approximation's 0.100, both measured
/// on the sealed arc before being written here.
const EXACT_FIXED_MIDDLE_STEPS: usize = 63;
const LOGQUAD_X4_FIXED_MIDDLE_STEPS: usize = 16;
/// 6 at v2's 0.300, sized and pinned exactly like the two above.
const LOGQUAD_X4_V2_FIXED_MIDDLE_STEPS: usize = 6;

static EXACT_FIXED_MIDDLE_PLAN: std::sync::OnceLock<FixedMiddlePlan<EXACT_FIXED_MIDDLE_STEPS>> =
    std::sync::OnceLock::new();

static LOGQUAD_X4_FIXED_MIDDLE_PLAN: std::sync::OnceLock<
    FixedMiddlePlan<LOGQUAD_X4_FIXED_MIDDLE_STEPS>,
> = std::sync::OnceLock::new();

static LOGQUAD_X4_V2_FIXED_MIDDLE_PLAN: std::sync::OnceLock<
    FixedMiddlePlan<LOGQUAD_X4_V2_FIXED_MIDDLE_STEPS>,
> = std::sync::OnceLock::new();

fn exact_fixed_middle_plan() -> &'static FixedMiddlePlan<EXACT_FIXED_MIDDLE_STEPS> {
    EXACT_FIXED_MIDDLE_PLAN.get_or_init(|| {
        let lower = exact_fixed_lower_plan();
        build_fixed_middle_plan::<EXACT_FIXED_MIDDLE_STEPS>(lower.z, lower.zend)
    })
}

fn logquad_x4_fixed_middle_plan() -> &'static FixedMiddlePlan<LOGQUAD_X4_FIXED_MIDDLE_STEPS> {
    LOGQUAD_X4_FIXED_MIDDLE_PLAN.get_or_init(|| {
        let lower = logquad_x4_fixed_lower_plan();
        build_fixed_middle_plan::<LOGQUAD_X4_FIXED_MIDDLE_STEPS>(lower.z, lower.zend)
    })
}

/// v2 builds from the SAME lower plan as v1 — the lower log step is shared —
/// so the only thing that differs is the step count this is generic over.
fn logquad_x4_v2_fixed_middle_plan() -> &'static FixedMiddlePlan<LOGQUAD_X4_V2_FIXED_MIDDLE_STEPS> {
    LOGQUAD_X4_V2_FIXED_MIDDLE_PLAN.get_or_init(|| {
        let lower = logquad_x4_fixed_lower_plan();
        build_fixed_middle_plan::<LOGQUAD_X4_V2_FIXED_MIDDLE_STEPS>(lower.z, lower.zend)
    })
}

/// The altitude-independent half of the 105--500 km segment, in `N` Boole steps.
///
/// `al`, `n` and `zr` are derived exactly as `dynamic_middle_state` derives them
/// from the same `lower_z` and the literal `500.0` that `altitude_km.min(500.0)`
/// returns at or above 500 km, so the abscissa walk is the same sequence of
/// additions and not a re-derivation.
fn build_fixed_middle_plan<const N: usize>(lower_z: f64, lower_zend: f64) -> FixedMiddlePlan<N> {
    use wide::f64x4;
    const C_FIVE_HALVES: f64x4 = f64x4::new([4.5e-6; 4]);

    let al = (500.0_f64 / lower_z).ln();
    let step_count = f64::from(u32::try_from(N).unwrap_or_default());
    let zr = (al / step_count).exp();
    let mut steps = [FixedMiddleStep::ZERO; N];
    let mut z = 0.0;
    let mut zend = lower_zend;
    let mut above_from = 0;
    let mut straddle = None;
    for (index, step) in steps.iter_mut().enumerate() {
        z = zend;
        zend = zr * z;
        let dz = 0.25 * (zend - z);
        let zs = boole_abscissae(&mut z, dz);
        let abscissa_vector = f64x4::from(zs);
        let break_offset = abscissa_vector - Z_BREAK_X4;
        let five_halves = break_offset * break_offset * break_offset.sqrt();
        *step = FixedMiddleStep {
            dz,
            temperature_basis: ((wide_const::LOW_TEMP_C2 * break_offset - wide_const::LOW_TEMP_C1)
                * break_offset
                * break_offset
                + f64x4::ONE)
                * break_offset,
            break_offset,
            argument_shape: f64x4::ONE + C_FIVE_HALVES * five_halves,
            gravity: jb_gravity_x4(abscissa_vector),
        };
        // Exactly the classification `jb_local_temp_step_x4` performs, done
        // once here instead of once per step per evaluation.
        if zs[0] > 125.0 {
            if above_from == 0 {
                above_from = index;
            }
        } else if zs[3] > 125.0 {
            straddle = Some((index, break_offset.simd_le(f64x4::ZERO)));
            above_from = index.saturating_add(1);
        } else {
            above_from = index.saturating_add(1);
        }
    }
    FixedMiddlePlan {
        steps,
        above_from,
        straddle,
        z,
        zend,
    }
}

/// `jb_local_temp_below_break_x4` with the altitude half already folded.
#[inline]
fn planned_temperature_below(step: &FixedMiddleStep, tc: TemperatureBroadcast) -> wide::f64x4 {
    step.temperature_basis * tc.gradient + tc.base
}

/// `jb_local_temp_above_break_x4` with the altitude half already folded.
#[inline]
fn planned_temperature_above(step: &FixedMiddleStep, tc: TemperatureBroadcast) -> wide::f64x4 {
    let argument = tc.argument_scale * step.break_offset * step.argument_shape;
    tc.base + tc.amplitude * atan_x4_dispatched(argument)
}

/// One planned Boole step, folded into the running integral in the order the
/// scalar loop folded it.
#[inline]
fn accumulate_middle_step(
    step: &FixedMiddleStep,
    temperature: wide::f64x4,
    ain: &mut f64,
    sub2: &mut f64,
) {
    let [first_weight, ..] = JB_WT;
    let mut weighted_integral = first_weight * *ain;
    let lane_integrands = (step.gravity / temperature).to_array();
    for (weight, lane_integrand) in JB_WT.iter().skip(1).zip(lane_integrands.iter()) {
        *ain = *lane_integrand;
        weighted_integral += weight * *ain;
    }
    *sub2 += step.dz * weighted_integral;
}

/// The 105--500 km segment run from a plan.
///
/// Two loops, not one with a branch: the pre-break prefix (at most a handful of
/// steps, and on the production profile exactly two) and the post-break
/// remainder, which on the production profile is 14 of the 16 steps and carries
/// every `atan_x4` in the segment. Splitting keeps the hot loop free of the two
/// scalar compares `jb_local_temp_step_x4` made per step, and the prefix and
/// remainder are disjoint and consecutive so the accumulation order is
/// unchanged.
#[inline]
fn fixed_middle_state<const N: usize>(
    plan: &FixedMiddlePlan<N>,
    tc: TemperatureBroadcast,
    mut ain: f64,
) -> MiddleState {
    let mut sub2 = 0.0;
    let mut temperature = wide::f64x4::ZERO;
    let straddle_index = plan.straddle.map(|(index, _)| index);
    for (index, step) in plan.steps.iter().enumerate().take(plan.above_from) {
        temperature = match plan.straddle {
            Some((straddle, mask)) if straddle == index => mask.select(
                planned_temperature_below(step, tc),
                planned_temperature_above(step, tc),
            ),
            _ => planned_temperature_below(step, tc),
        };
        accumulate_middle_step(step, temperature, &mut ain, &mut sub2);
    }
    debug_assert!(straddle_index.is_none_or(|index| index.saturating_add(1) == plan.above_from));
    for step in plan.steps.iter().skip(plan.above_from) {
        temperature = planned_temperature_above(step, tc);
        accumulate_middle_step(step, temperature, &mut ain, &mut sub2);
    }
    let [_, _, _, tloc3] = temperature.to_array();
    MiddleState {
        sub2,
        ain,
        tloc3,
        z: plan.z,
        zend: plan.zend,
    }
}

/// The 105--500 km segment walked, for every altitude the plan does not cover.
#[inline]
fn dynamic_middle_state<P: QuadratureProfile>(
    altitude_km: f64,
    tc: TemperatureBroadcast,
    mut ain: f64,
    mut z: f64,
    mut zend: f64,
) -> Result<MiddleState, Jb2008Error> {
    let al = (altitude_km.min(500.0) / z).ln();
    let n = jb_step_count(al / P::MIDDLE_LOG_STEP)?;
    let zr = (al / f64::from(n)).exp();
    let mut sub2 = 0.0;
    let mut temperature = wide::f64x4::ZERO;
    let [first_weight, ..] = JB_WT;
    for _ in 0..n {
        z = zend;
        zend = zr * z;
        let dz = 0.25 * (zend - z);
        let mut weighted_integral = first_weight * ain;
        let zs = boole_abscissae(&mut z, dz);
        let abscissa_vector = wide::f64x4::from(zs);
        temperature = jb_local_temp_step_x4(zs, abscissa_vector, tc);
        let lane_integrands = (jb_gravity_x4(abscissa_vector) / temperature).to_array();
        for (weight, lane_integrand) in JB_WT.iter().skip(1).zip(lane_integrands.iter()) {
            ain = *lane_integrand;
            weighted_integral += weight * ain;
        }
        sub2 += dz * weighted_integral;
    }
    let [_, _, _, tloc3] = temperature.to_array();
    Ok(MiddleState {
        sub2,
        ain,
        tloc3,
        z,
        zend,
    })
}

/// The altitude-independent half of the 90--105 km segment, in `N` Boole steps.
///
/// `zr` is derived exactly as `dynamic_lower_state` derives it, from the same
/// `ln(105/90)` and the same `f64::from(step count)`, so the abscissa walk is
/// the same sequence of additions and not a re-derivation.
fn build_fixed_lower_plan<const N: usize>() -> FixedLowerPlan<N> {
    let z1 = 90.0_f64;
    let z2 = 105.0_f64;
    let step_count = f64::from(u32::try_from(N).unwrap_or_default());
    let zr = ((z2 / z1).ln() / step_count).exp();
    let mut steps = [FixedLowerStep::ZERO; N];
    let mut z = 0.0;
    let mut zend = z1;
    let mut mb2 = 0.0;
    let mut gravl = 0.0;
    for step in &mut steps {
        z = zend;
        zend = zr * z;
        let dz = 0.25 * (zend - z);
        let zs = boole_abscissae(&mut z, dz);
        let abscissa_vector = wide::f64x4::from(zs);
        let molecular_masses = jb_mbar_x4(abscissa_vector).to_array();
        let gravity_values = jb_gravity_x4(abscissa_vector).to_array();
        let dzv = abscissa_vector - Z_BREAK_X4;
        let temperature_basis =
            (((wide_const::LOW_TEMP_C2 * dzv - wide_const::LOW_TEMP_C1) * dzv * dzv
                + wide::f64x4::ONE)
                * dzv)
                .to_array();
        let mut mb_gravity = [0.0; 4];
        for ((molecular_mass, gravity), mass_gravity) in molecular_masses
            .iter()
            .zip(gravity_values.iter())
            .zip(mb_gravity.iter_mut())
        {
            mb2 = *molecular_mass;
            gravl = *gravity;
            *mass_gravity = mb2 * gravl;
        }
        *step = FixedLowerStep {
            dz,
            temperature_basis,
            mb_gravity,
        };
    }
    FixedLowerPlan {
        steps,
        z,
        zend,
        mb2,
        gravl,
    }
}

#[inline]
fn dynamic_lower_state<P: QuadratureProfile>(
    altitude_km: f64,
    tc: TemperatureBroadcast,
    mut ain: f64,
) -> Result<LowerState, Jb2008Error> {
    let z1 = 90.0;
    let z2 = altitude_km.min(105.0);
    let log_ratio = (z2 / z1).ln();
    let n = jb_step_count(log_ratio / P::LOWER_LOG_STEP)?;
    let zr = (log_ratio / f64::from(n)).exp();
    let mut zend = z1;
    let mut sub2 = 0.0;
    let mut z = 0.0;
    let mut molecular_mass = wide::f64x4::ZERO;
    let mut local_temperature = wide::f64x4::ZERO;
    let mut gravity = wide::f64x4::ZERO;
    let [first_weight, ..] = JB_WT;
    for _ in 0..n {
        z = zend;
        zend = zr * z;
        let dz = 0.25 * (zend - z);
        let mut weighted_integral = first_weight * ain;
        let zs = boole_abscissae(&mut z, dz);
        let abscissa_vector = wide::f64x4::from(zs);
        molecular_mass = jb_mbar_x4(abscissa_vector);
        local_temperature = jb_local_temp_below_break_x4(abscissa_vector, tc);
        gravity = jb_gravity_x4(abscissa_vector);
        // The four lane quotients the scalar loop computed one at a time. IEEE
        // division is per-lane and lane-independent, so `(m * g) / t` in a
        // vector is the same operation on the same operands as the scalar
        // `mb2 * gravl / tloc2` — the frozen scalar-divide oracle in this
        // module's tests is what says so rather than this comment.
        let lane_integrands = (molecular_mass * gravity / local_temperature).to_array();
        for (weight, lane_integrand) in JB_WT.iter().skip(1).zip(lane_integrands.iter()) {
            ain = *lane_integrand;
            weighted_integral += weight * ain;
        }
        sub2 += dz * weighted_integral;
    }
    // The scalar loop left `mb2`/`tloc2`/`gravl` holding the LAST lane of the
    // last step, which is exactly lane 3 of these vectors.
    let [_, _, _, mb2] = molecular_mass.to_array();
    let [_, _, _, tloc2] = local_temperature.to_array();
    let [_, _, _, gravl] = gravity.to_array();
    Ok(LowerState {
        sub2,
        z,
        zend,
        mb2,
        tloc2,
        gravl,
    })
}

#[inline]
fn fixed_lower_state<const N: usize>(
    plan: &FixedLowerPlan<N>,
    tc: TemperatureBroadcast,
    mut ain: f64,
) -> LowerState {
    let mut sub2 = 0.0;
    let mut temperature = wide::f64x4::ZERO;
    let [first_weight, ..] = JB_WT;
    for step in &plan.steps {
        let mut weighted_integral = first_weight * ain;
        temperature = wide::f64x4::from(step.temperature_basis) * tc.gradient + tc.base;
        let lane_integrands = (wide::f64x4::from(step.mb_gravity) / temperature).to_array();
        for (weight, lane_integrand) in JB_WT.iter().skip(1).zip(lane_integrands.iter()) {
            ain = *lane_integrand;
            weighted_integral += weight * ain;
        }
        sub2 += step.dz * weighted_integral;
    }
    let [_, _, _, tloc2] = temperature.to_array();
    LowerState {
        sub2,
        z: plan.z,
        zend: plan.zend,
        mb2: plan.mb2,
        tloc2,
        gravl: plan.gravl,
    }
}

// The five altitude-only fits the R57 upper segment evaluates, generated by
// `tools/r57-upper-fit/fit_upper.py` from the Chebyshev-node dump in this
// module's `dump_upper_segment_fit_samples`. Regenerate both together; the
// coefficients are meaningless against a different abscissa walk or a different
// altitude domain.
//
// The reported tail is the SUM of the discarded Chebyshev coefficients, which
// on that basis is a true bound on the interpolant's error and not an estimate
// of it. Each budget is that tail multiplied by the largest coefficient the
// term can carry over the fitted temperature domain, against the leading `G0`
// of ~26.8 -- see the generator for the derivation and `fitted_upper_segment`
// for where each one enters. Every function lands one Estrin body above what it
// needs, which is why the tails come in three to five orders inside the budget.
//
// domain [500.0, 1000.0] km, 96 Chebyshev-Gauss nodes
// G0: needs degree 5, emitted at 8, tail 2.057e-12 against a 2.7e-08 budget
//   |cheb| decay: 3.0e+01 1.1e+00 2.0e-02 3.5e-04 6.2e-06 1.1e-07 1.9e-09 3.4e-11 6.1e-13 5.3e-15 2.1e-15 8.6e-15 2.2e-15 1.4e-15 1.5e-14 7.8e-15 4.9e-15 1.3e-15 5.3e-15 1.4e-14 1.8e-15 7.4e-17
// G1: needs degree 11, emitted at 16, tail 5.874e-14 against a 1.2e-09 budget
//   |cheb| decay: 3.0e-03 2.3e-03 6.7e-04 1.8e-04 4.5e-05 1.1e-05 2.6e-06 5.9e-07 1.3e-07 2.8e-08 5.9e-09 1.2e-09 2.5e-10 4.8e-11 9.2e-12 1.7e-12 3.0e-13 5.0e-14 7.5e-15 9.2e-16 5.9e-17 1.4e-17
// G2: needs degree 10, emitted at 16, tail 2.093e-15 against a 3.1e-11 budget
//   |cheb| decay: 4.2e-07 5.0e-07 2.2e-07 8.4e-08 3.0e-08 1.0e-08 3.4e-09 1.0e-09 3.1e-10 8.7e-11 2.4e-11 6.5e-12 1.7e-12 4.4e-13 1.1e-13 2.7e-14 6.7e-15 1.6e-15 3.8e-16 8.7e-17 2.0e-17 4.4e-18
// G3: needs degree 5, emitted at 8, tail 1.226e-13 against a 2.9e-12 budget
//   |cheb| decay: 6.6e-11 9.5e-11 5.1e-11 2.4e-11 1.1e-11 4.5e-12 1.8e-12 6.7e-13 2.4e-13 8.2e-14 2.7e-14 8.8e-15 2.8e-15 8.4e-16 2.5e-16 7.3e-17 2.1e-17 5.9e-18 1.6e-18 4.5e-19 1.2e-19 3.2e-20
// F1: needs degree 12, emitted at 16, tail 1.735e-14 against a 4.7e-11 budget
//   |cheb| decay: 6.7e-05 8.1e-05 3.5e-05 1.3e-05 4.0e-06 1.2e-06 3.2e-07 8.4e-08 2.1e-08 5.1e-09 1.2e-09 2.6e-10 5.7e-11 1.2e-11 2.4e-12 4.7e-13 8.6e-14 1.5e-14 2.3e-15 2.8e-16 1.7e-17 5.1e-18

/// `G0` of the R57 upper-segment expansion; degree 8,
/// discarded Chebyshev tail 2.057e-12 against a 2.7e-08 budget.
const FITTED_UPPER_G0: [f64; 9] = [
    29.906_105_140_239_816,
    -1.144_273_299_902_433_2,
    0.040_252_956_579_360_344,
    -0.001_416_004_936_030_705,
    4.981_417_818_387_494e-05,
    -1.751_534_968_027_120_4e-06,
    6.169_708_891_926_955e-08,
    -2.188_893_214_830_992_8e-09,
    7.801_759_238_645_9e-11,
];
/// `G1` of the R57 upper-segment expansion; degree 16,
/// discarded Chebyshev tail 5.874e-14 against a 1.2e-09 budget.
const FITTED_UPPER_G1: [f64; 17] = [
    0.002_414_115_593_13,
    -0.001_820_854_283_638_7,
    0.001_018_901_476_741_747_6,
    -0.000_523_562_904_897_109_8,
    0.000_257_472_149_630_269_7,
    -0.000_121_929_044_223_327_03,
    5.574_516_099_842_985_5e-05,
    -2.469_869_753_590_065e-05,
    1.064_500_976_273_817e-05,
    -4.467_514_712_118_888e-06,
    1.840_499_698_359_717_3e-06,
    -7.666_114_007_117_327e-07,
    3.038_292_923_918_106_7e-07,
    -9.346_132_393_931_819e-08,
    3.625_937_402_181_496_4e-08,
    -2.795_079_012_685_164_3e-08,
    9.867_518_357_964_4e-09,
];
/// `G2` of the R57 upper-segment expansion; degree 16,
/// discarded Chebyshev tail 2.093e-15 against a 3.1e-11 budget.
const FITTED_UPPER_G2: [f64; 17] = [
    2.282_149_126_699_051_3e-07,
    -2.890_045_115_099_318e-07,
    2.416_533_745_887_643_5e-07,
    -1.769_600_829_583_768e-07,
    1.224_027_855_114_723_2e-07,
    -8.081_909_081_675_111e-08,
    5.086_907_635_856_387e-08,
    -3.066_218_921_618_97e-08,
    1.766_044_504_555_961_3e-08,
    -9.416_364_983_560_186e-09,
    5.018_266_032_329_201e-09,
    -3.325_164_818_995_713_5e-09,
    1.736_852_843_387_647_6e-09,
    -1.078_973_736_167_350_8e-10,
    2.906_381_739_731_69e-11,
    -4.491_755_783_147_856_6e-10,
    2.190_591_929_495_296_6e-10,
];
/// `G3` of the R57 upper-segment expansion; degree 8,
/// discarded Chebyshev tail 1.226e-13 against a 2.9e-12 budget.
const FITTED_UPPER_G3: [f64; 9] = [
    2.403_494_557_782_461_4e-11,
    -3.999_439_990_014_574_5e-11,
    4.072_187_855_315_502e-11,
    -4.447_388_660_077_302_3e-11,
    3.858_838_245_402_72e-11,
    3.318_007_609_003_951e-12,
    -4.698_070_056_812_947e-12,
    -4.264_499_128_755_212e-11,
    3.064_440_566_256_411e-11,
];
/// `F1` of the R57 upper-segment expansion; degree 16,
/// discarded Chebyshev tail 1.735e-14 against a 4.7e-11 budget.
const FITTED_UPPER_F1: [f64; 17] = [
    3.559_881_801_024_096e-05,
    -4.904_629_756_337_161e-05,
    4.320_883_325_210_317e-05,
    -3.083_542_429_960_189_5e-05,
    1.938_035_396_816_077e-05,
    -1.116_579_197_722_273_7e-05,
    6.030_176_122_673_848_5e-06,
    -3.095_488_761_092_991_6e-06,
    1.522_613_398_783_918_1e-06,
    -7.184_913_995_889_456e-07,
    3.298_477_743_851_382e-07,
    -1.536_847_352_376_820_8e-07,
    6.622_967_131_432_919e-08,
    -2.042_289_749_583_833e-08,
    8.616_785_900_546_67e-09,
    -7.715_586_169_965_15e-09,
    2.831_192_502_661_395e-09,
];

/// The above-500 km Boole step, walked. The arm [`fitted_upper_segment`]
/// replaces, and the arm every profile but the flown one always takes.
///
/// Bound as its own function for the same reason `thermal_state` is: it is a
/// separately priceable stage, and here also because `jb_density` sits against
/// a 200-line clippy bound that two spellings of one segment do not fit inside.
#[inline]
fn walked_upper_segment<P: QuadratureProfile>(
    altitude_km: f64,
    plan_z: f64,
    plan_zend: f64,
    temperature: TemperatureBroadcast,
    mut ain: f64,
) -> Result<UpperSegment, Jb2008Error> {
    let [first_weight, ..] = JB_WT;
    let mut z = plan_z;
    let mut zend = plan_zend;
    let mut sum3 = 0.0;
    let mut upper_temperature = wide::f64x4::ZERO;
    let (n, zr) = if CEILING_PROBE == 1 {
        upper_temperature = temperature.base;
        (0, 1.0)
    } else {
        let ratio = altitude_km.max(500.0) / z;
        let al = ratio.ln().max(0.0);
        let r = if altitude_km > 500.0 {
            P::UPPER_LOG_STEP
        } else {
            P::MIDDLE_LOG_STEP
        };
        let n = jb_step_count(al / r)?;
        // `ratio.max(1.0)`, not `ratio`: `al` carries a `.max(0.0)` clamp, so on
        // the sub-unit side the walked form is `exp(0.0)`, which is exactly 1.0.
        // Reproducing the clamp keeps the two arms equal on that branch instead
        // of nearly equal. See `RETIRE_ZR_ROUND_TRIP`.
        let zr = if P::RETIRE_ZR_ROUND_TRIP && n == 1 {
            ratio.max(1.0)
        } else {
            (al / f64::from(n)).exp()
        };
        (n, zr)
    };
    for _ in 0..n {
        z = zend;
        zend = zr * z;
        let dz = 0.25 * (zend - z);
        let mut sum1 = first_weight * ain;
        let zs = boole_abscissae(&mut z, dz);
        let abscissa_vector = wide::f64x4::from(zs);
        upper_temperature = jb_local_temp_step_x4(zs, abscissa_vector, temperature);
        let lane_integrands = (jb_gravity_x4(abscissa_vector) / upper_temperature).to_array();
        for (weight, lane_integrand) in JB_WT.iter().skip(1).zip(lane_integrands.iter()) {
            ain = *lane_integrand;
            sum1 += weight * ain;
        }
        sum3 += dz * sum1;
    }
    let [_, _, _, tloc4] = upper_temperature.to_array();
    Ok(UpperSegment { sum3, tloc4, z })
}

/// Everything the above-500 km segment produces, whether walked or fitted.
struct UpperSegment {
    sum3: f64,
    tloc4: f64,
    /// The last abscissa, which is where the walked loop leaves `z` and the
    /// only value below the segment that reads it (the `jb_semian` guard).
    /// `zend` is deliberately absent: nothing below the segment reads it.
    z: f64,
}

/// The above-500 km Boole step, evaluated from five fits in ALTITUDE instead of
/// walked.
///
/// # The separation, which is exact and not empirical
///
/// Above 500 km the middle plan's `z` and `zend` are constants, so the step's
/// exit altitude is a fixed multiple of the input altitude and every abscissa is
/// affine in it. The temperature profile `tc` is a pure function of `Texo`
/// alone. So the segment's two outputs are functions of exactly two scalars, and
/// this function is the statement that they SEPARATE into functions of one.
///
/// The arctangent's argument there is `tc[3] * f(z)` with
/// `f(z) = dz (1 + 4.5e-6 dz^2.5)`, `dz = z - 125`, and it is large — at least
/// 65 over the fitted temperature domain, and 100 to 500 across the censused
/// production band. With `x = 1/(tc[3] f(z))` and `a = tc[2]/Texo`:
///
/// ```text
/// T   = tc[0] + tc[2] atan(1/x) = Texo - tc[2] x + tc[2] x^3/3 - ...
/// 1/T = (1/Texo) (1 + a x + a^2 x^2 + (a^3 - a/3) x^3 + ...)
/// ```
///
/// the first line because `tc[0] + tc[2] pi/2` **is** `Texo` — that is
/// [`fitted_v7_texo_of`]'s identity, not an approximation of one. `x` carries
/// `tc[3]` as a pure scale, so `x^k = tc[3]^-k f(z)^-k` and the temperature
/// comes out of the quadrature sum entirely:
///
/// ```text
/// sum3  = dz [ w0 ain + (G0 + b G1 + b^2 G2 + (b^3 - d/3) G3) / Texo ]
/// tloc4 = Texo - (tc[2]^2/tc[1]) F1 + (tc[2]^4/(3 tc[1]^3)) F1^3
/// b = a tc[2]/tc[1],  d = a (tc[2]/tc[1])^3
/// ```
///
/// with `Gk(alt) = sum_i w_{i+1} g(z_i) f(z_i)^-k` and `F1(alt) = 1/f(z_4)`
/// functions of altitude ALONE. Five 1-D fits, where a tensor product in
/// `(alt, Texo)` would have been fourteen degree-14 Horners — about 23 ns
/// against a segment worth roughly 30, which is why nobody took this lever
/// before.
///
/// # What is truncated, and by how much
///
/// The `1/T` series is cut after `x^3`, leaving `O((a x)^4)`. `a x = b/f` and
/// `f >= 4969` on the domain, so at the worst corner of the fitted temperature
/// range (`Texo = 2600 K`, where `b = 39.16`) that is `7.9e-3` and the
/// truncation is **3.9e-9** of `sum3`. `tloc4`'s series is cut after `x^3` as
/// well, leaving `a x^5/5`, about 7e-12 relative.
///
/// `sum3` reaches the density through `exp(-fact2 * mass)` whose argument
/// reaches 45, so a relative error in it is amplified ~45x; `tloc4` reaches it
/// through two logarithms, so ~1.4x. The per-fit budgets in
/// `tools/r57-upper-fit/fit_upper.py` are set from those factors, and the
/// realised end-to-end figure is measured rather than argued — see
/// `fitted_upper_segment_matches_the_walked_step` and
/// `v7_broad_grid_density_error_stays_within_rescoped_bound`.
///
/// # `F1^3`, not a sixth fit
///
/// `tloc4`'s third-order term needs `1/f^3` and cubing the fitted `F1` supplies
/// it. That term is 1.3e-7 of `tloc4` at worst, so it tolerates 1e-2 relative
/// error against the 3e-5 a cube of a 1e-5 fit carries.
#[inline]
fn fitted_upper_segment(
    altitude_km: f64,
    plan_z: f64,
    plan_zend: f64,
    tc: [f64; 4],
    exospheric_temperature: f64,
    ain: f64,
) -> UpperSegment {
    let effective_altitude = altitude_km.max(UPPER_FIT_ALT_LO);
    // Formed exactly as the walked arm forms it, including the `max(1.0)` that
    // `RETIRE_ZR_ROUND_TRIP` documents, because the fit was generated against
    // this same expression.
    let ratio = (effective_altitude / plan_z).max(1.0);
    let step_z = plan_zend;
    let dz = 0.25 * (ratio.mul_add(step_z, -step_z));

    // The five powers of the fit variable, formed ONCE. Each Estrin body needs
    // them and each used to rebuild them, which put three dependent multiplies
    // at the head of five chains that are otherwise independent.
    let powers = UpperPowers::of(fitted_upper_v_of(effective_altitude));
    let g0 = fitted_upper_estrin9(&FITTED_UPPER_G0, powers);
    let g1 = fitted_upper_estrin17(&FITTED_UPPER_G1, powers);
    let g2 = fitted_upper_estrin17(&FITTED_UPPER_G2, powers);
    let g3 = fitted_upper_estrin9(&FITTED_UPPER_G3, powers);
    let f1 = fitted_upper_estrin17(&FITTED_UPPER_F1, powers);

    let [_, gradient, amplitude, _] = tc;
    // ONE reciprocal, used twice. `a` and the final `bracket / Texo` were two
    // divides in series on the critical path.
    let inverse_texo = 1.0 / exospheric_temperature;
    let a = amplitude * inverse_texo;
    // `tc[2]/tc[1]`, i.e. `1/tc[3]`, formed from the two profile entries rather
    // than by reciprocating the third so that a `tc[3]` of zero cannot appear
    // as an infinity here that the walked arm would never have produced.
    let scale = amplitude / gradient;
    let b = a * scale;
    let scale_cubed = scale * scale * scale;
    // `b^3 - d/3` with `d = a scale^3`, factored so the common `scale^3` and
    // `a` are formed once: `a scale^3 (a^2 - 1/3)`.
    let cubic = a * scale_cubed * a.mul_add(a, -(1.0 / 3.0));
    let bracket = cubic.mul_add(g3, (b * b).mul_add(g2, b.mul_add(g1, g0)));
    let [first_weight, ..] = JB_WT;
    let sum3 = dz * first_weight.mul_add(ain, bracket * inverse_texo);

    let f1_cubed = f1 * f1 * f1;
    let tloc4 = (amplitude * scale_cubed / 3.0).mul_add(
        f1_cubed,
        (-(amplitude * scale)).mul_add(f1, exospheric_temperature),
    );

    // The walked loop leaves `z` on the last abscissa and `zend` on the step's
    // exit. The only later reader of `z` is the `z < 2000.0` guard on
    // `jb_semian`, and it is reproduced by the same four additions rather than
    // as `step_z + 4 dz` so the guard sees the value it has always seen.
    let mut z = step_z;
    let _ = boole_abscissae(&mut z, dz);

    UpperSegment { sum3, tloc4, z }
}

/// Map an altitude onto the upper fit's variable in `[-1, 1]`.
#[inline]
fn fitted_upper_v_of(altitude_km: f64) -> f64 {
    2.0 * (altitude_km - UPPER_FIT_ALT_LO) / (UPPER_FIT_ALT_HI - UPPER_FIT_ALT_LO) - 1.0
}

/// The powers of the fit variable every upper-segment Estrin body needs.
#[derive(Clone, Copy)]
struct UpperPowers {
    v: f64,
    v2: f64,
    v4: f64,
    v8: f64,
    v16: f64,
}

impl UpperPowers {
    #[inline]
    fn of(v: f64) -> Self {
        let v2 = v * v;
        let v4 = v2 * v2;
        let v8 = v4 * v4;
        Self {
            v,
            v2,
            v4,
            v8,
            v16: v8 * v8,
        }
    }
}

/// Estrin evaluation of a degree-8 upper-segment fit.
///
/// Estrin and not Horner for the reason [`fitted_v7_estrin`] records: these five
/// sit on the critical path between the exospheric temperature and every species
/// exponent, and depth is what costs. Two bodies rather than one generic one,
/// and two rather than five, because the generator emits at exactly these two
/// degrees — `tools/r57-upper-fit/fit_upper.py` reports the degree each function
/// NEEDS and rounds it up to the nearest body, and fails loudly if one ever
/// needs more than 16. Both bodies (and the degree-14 one above) are stamped
/// from [`estrin_body!`], so a new degree is one invocation, not a fourth hand
/// transcription.
#[inline]
const fn fitted_upper_estrin9(coefficients: &[f64; 9], p: UpperPowers) -> f64 {
    let [c0, c1, c2, c3, c4, c5, c6, c7, c8] = *coefficients;
    let UpperPowers { v, v2, v4, v8, .. } = p;

    estrin_body!([c0, c1, c2, c3, c4, c5, c6, c7, c8], v, v2, v4, v8)
}

/// Estrin evaluation of a degree-16 upper-segment fit. See
/// [`fitted_upper_estrin9`].
#[inline]
const fn fitted_upper_estrin17(coefficients: &[f64; 17], p: UpperPowers) -> f64 {
    let [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16] = *coefficients;
    let UpperPowers { v, v2, v4, v8, v16 } = p;

    estrin_body!(
        [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16],
        v,
        v2,
        v4,
        v8,
        v16
    )
}

/// The domain half of the `ln` that `jb_density` no longer calls.
///
/// Retiring the species round trip replaced `exp(ln(x) + y)` with `x * exp(y)`,
/// and `ln` was doing two jobs: the logarithm, and rejecting a negative `x` by
/// returning `NaN`. That `NaN` reached `jb_density`'s `is_finite` guard and came
/// back as [`Jb2008Error::NumericalDomain`]. A bare multiply keeps a negative
/// factor finite, so the sum could return a plausible wrong density where the
/// old code errored. This restores the rejection and nothing else.
///
/// `-0.0` must NOT be rejected: `ln(-0.0)` is `NEG_INFINITY`, not `NaN`, and its
/// `exp` is `0.0`. `-0.0 < 0.0` is false, so it takes the identity arm and the
/// term becomes `-0.0` rather than `+0.0` — a difference the compensated sum in
/// `jb_density` does not see. `NaN` in maps to `NaN` out for the same reason.
///
/// Production is not known to reach the rejecting arm; `species_factors_stay_
/// positive_across_a_wide_input_sweep` records how hard it was looked for. It is
/// here to preserve the failure mode, not because a caller is expected to trip
/// it, and it is unit-tested directly so that being unreached does not make it
/// untested.
#[inline]
fn species_factor_domain(value: f64) -> f64 {
    if value < 0.0 {
        f64::NAN
    } else {
        value
    }
}

// Test-only capture of the six species `(factor, log_offset)` pairs the last
// `jb_density` call on this thread produced.
//
// The round-trip accuracy test needs the values production actually reaches,
// not a hand-picked sweep over a plausible-looking range. Recording them from
// inside `jb_density` is the only way to get them without paraphrasing the
// function, and a paraphrase is exactly what the r12 corpus warning is about.
//
// A `//` comment, not a doc comment: `thread_local!` does not forward doc
// attributes to the static it expands to, so `///` here is an unused-doc-comment
// warning rather than documentation.
#[cfg(test)]
thread_local! {
    static SPECIES_CAPTURE: std::cell::Cell<[(f64, f64); 6]> =
        const { std::cell::Cell::new([(0.0, 0.0); 6]) };
}

/// Ceiling probe selector for the levers this module has NOT taken.
///
/// **Every non-zero value returns WRONG DENSITIES.** It exists because the only
/// way to price a lever below this project's 1.13% arc null floor is a per-call
/// microbenchmark, and the only way to price a lever *before building it* is to
/// delete the work it would remove and time what is left. Set by `sed` between
/// two builds of `examples/jb2008_cost_map`, never committed non-zero.
///
/// | arm | deletes | measured, median of 22 rotating paired rounds |
/// |---|---|---|
/// | 1 | the whole above-500 km segment: its `ln`, its `exp`, its Boole step | **-14.15%** of the model-7 kernel call, 22/22 negative |
/// | 2 | the six species `exp` calls | **-8.38%**, 22/22 negative |
/// | 3 | the satellite latitude's `sin_cos` | **-1.23%**, 17/22 negative |
///
/// Arms 1 and 2 are `docs/JB2008_COST_MAP.md`'s L1 and L3; arm 3 is L4. Arms
/// for L5 and L6 were here too and are gone because both became real levers —
/// `RETIRE_ZR_ROUND_TRIP` and `DLRSL_ZERO_ABOVE_KM`, whose own ceilings are
/// reachable by setting those constants rather than by a probe.
///
/// Each arm is an upper bound and not a price: arm 2 makes `exp` free, where a
/// four-wide `exp` would make it cheaper-per-call and no cheaper-per-lane, and
/// arm 1 charges nothing for the fit that would have to replace the segment.
const CEILING_PROBE: u8 = 0;

fn jb_density<P: QuadratureProfile>(
    input: Jb2008Input,
    altitude_km: f64,
    tc: [f64; 4],
    exospheric_temperature: f64,
    sin_geocentric_latitude: f64,
) -> Result<f64, Jb2008Error> {
    // Both log-density corrections are computed HERE, above the quadrature, and
    // the position is the whole lever: neither reads anything the quadrature
    // produces. `jb_dlrsl` and `jb_semian` are pure functions of this function's
    // INPUT — altitude, epoch, latitude and the solar indices — yet they used to
    // sit at the bottom, between the quadrature and the six species `exp`. That
    // put `mjd -> jb_day_of_year -> tau -> sin_cos -> gtz` in series ahead of
    // every one of those exponentials when it can run alongside the quadrature
    // instead. §4a of `docs/JB2008_COST_MAP.md` is the reason this is worth
    // anything: in this kernel the question is chain depth, not call count.
    //
    // LLVM does not do it unaided — the `?` on `jb_day_of_year` is a
    // control-flow edge and the code it would have to cross contains a loop with
    // a fallible step count.
    //
    // The gate is on the ALTITUDE and not on the value, because computing the
    // value is the whole cost: the term carries the `exp` and the `sin` this
    // skips. See `DLRSL_ZERO_ABOVE_KM` for the envelope that bounds what is
    // being discarded, and for why no bit-identical form of this exists inside
    // the flown band.
    let latitude_correction = if altitude_km >= P::DLRSL_ZERO_ABOVE_KM {
        0.0
    } else {
        jb_dlrsl(
            altitude_km,
            input.mjd_utc,
            input.sat_geocentric_lat_rad,
            sin_geocentric_latitude,
        )
    };
    // The `?` deliberately does NOT move with the call. `jb_day_of_year` is
    // fallible and its result is consumed only under `z < 2000.0`; unwrapping it
    // here would make an absurd MJD refuse an above-2000 km evaluation that
    // previously succeeded. The `Result` is carried down and unwrapped at
    // exactly the point the old code unwrapped it, under exactly the old guard.
    // `jb_semian` itself is total, so running it unconditionally is a question
    // of wasted work above 2000 km and not of behaviour.
    let hoisted_semiannual_correction =
        jb_day_of_year(input.mjd_utc).map(|day| jb_semian(input, day, altitude_km));
    let z1 = 90.0;
    let mb1 = JB_MBAR_90_KM;
    // 90 km is 35 km below the 125 km break on every call, so this is the low
    // branch of `jb_local_temp` and only the low branch. See the note there.
    let tloc1 = jb_local_temp_below_break(z1, tc);
    let initial_ain = mb1 * JB_GRAVITY_90_KM / tloc1;
    let temperature = TemperatureBroadcast::new(tc);
    let lower_was_planned = P::USE_FIXED_LOWER_PLAN && altitude_km >= 105.0;
    let lower = if lower_was_planned {
        P::fixed_lower_state(temperature, initial_ain)
    } else {
        dynamic_lower_state::<P>(altitude_km, temperature, initial_ain)?
    };
    let LowerState {
        mut sub2,
        mut z,
        mut zend,
        mb2,
        tloc2,
        gravl,
    } = lower;
    let mut rho = 3.46e-6 * mb2 * tloc1 / (sub2 / JB_RSTAR).exp() / (mb1 * tloc2);
    let anm = JB_AVOGAD * rho;
    let an = anm / mb2;
    let mut fact2 = anm / 28.960;
    let [fraction_0, fraction_1, fraction_2, fraction_3] = JB_FRAC;
    // Each species is carried as a linear FACTOR and a log OFFSET, and the
    // `exp` at the bottom of this function combines them as
    // `factor * exp(offset)`. Both profiles use this representation; they
    // differ only in where the split sits, and `RETIRE_SPECIES_ROUND_TRIP` is
    // the whole of the difference.
    //
    // The offset starts at zero and the factor holds the number density, which
    // is the form that retires five `ln` calls. `P::RETIRE_SPECIES_ROUND_TRIP
    // == false` immediately folds the factor back into the offset with the very
    // `ln` being retired, reproducing the previous arithmetic exactly: the fold
    // leaves `factor == 1.0`, and `1.0 * exp(offset)` is bit-identical to
    // `exp(offset)`. That is what keeps the exact profile equal to the sealed
    // Orekit fixture bit for bit. See the constant for why that matters more
    // there than being nearer the true value does.
    //
    // `NAN` where the factor is negative is not defensive padding — it is what
    // reproduces the retired `ln`. `ln` of a negative is `NaN`, that `NaN` used
    // to flow through `exp` into the species sum, and the `is_finite` guard on
    // this function's result turned it into `Jb2008Error::NumericalDomain`. A
    // bare multiply would instead have carried a negative term into the sum and
    // could return a plausible wrong density where the old code errored. `-0.0`
    // deliberately takes the non-NaN arm, matching `ln(-0.0) == NEG_INFINITY`
    // whose `exp` is `0.0`; the resulting term is `-0.0` instead of `+0.0`,
    // which the compensated sum below is insensitive to.
    let mut density_factor = [
        fraction_0 * fact2,
        fact2 * (1.0 + fraction_1) - an,
        2.0 * (an - fact2),
        fraction_2 * fact2,
        fraction_3 * fact2,
        1.0,
    ];
    for factor in &mut density_factor {
        *factor = species_factor_domain(*factor);
    }
    let mut log_number_density = [0.0_f64; 6];
    if !P::RETIRE_SPECIES_ROUND_TRIP {
        for (offset, factor) in log_number_density.iter_mut().zip(density_factor.iter_mut()) {
            *offset = factor.ln();
            *factor = 1.0;
        }
    }

    if altitude_km <= 105.0 {
        // One statement for both profiles. With the round trip retired this is
        // `factor[5] = factor[4]` and `offset[5] = 0.0 - 25.0`; with it kept it
        // is `factor[5] = 1.0` and `offset[5] = ln(x4) - 25.0`, which is what
        // the single `log[5] = log[4] - 25.0` line used to say.
        let [_, _, _, _, helium_factor, hydrogen_factor] = &mut density_factor;
        *hydrogen_factor = *helium_factor;
        log_number_density[5] = log_number_density[4] - 25.0;
    } else {
        // The plan covers exactly the case where `altitude_km.min(500.0)` is
        // the literal 500.0 AND the segment starts at the lower plan's exit,
        // which is the only way `z` can be a plan constant.
        let middle_was_planned =
            P::USE_FIXED_MIDDLE_PLAN && lower_was_planned && altitude_km >= 500.0;
        let middle = if middle_was_planned {
            P::fixed_middle_state(temperature, gravl / tloc2)
        } else {
            dynamic_middle_state::<P>(altitude_km, temperature, gravl / tloc2, z, zend)?
        };
        let MiddleState {
            sub2: middle_sub2,
            ain,
            tloc3,
            z: middle_z,
            zend: middle_zend,
        } = middle;
        sub2 = middle_sub2;
        z = middle_z;
        zend = middle_zend;
        // `z` is the middle segment's exit altitude, reached by accumulation:
        // it lands within ~4e-13 km of 500 on EITHER side depending on the
        // middle step count. On the high side, an altitude of exactly 500.0
        // makes this ratio fractionally less than one and the log negative,
        // which `jb_step_count` rejects as NumericalDomain — a coin flip the
        // shipped 16-step plan happens to win. The clamp is bit-neutral
        // whenever the log is non-negative (every case the current plans
        // produce) and turns the losing side into a zero-width upper segment
        // instead of an error.
        //
        // The fitted arm's preconditions are all four of: the profile opts in,
        // the middle segment came from the PLAN (so `z` and `zend` are the plan
        // constants the fit was generated against), the altitude is inside the
        // fit's domain, and the exospheric temperature is inside the same
        // domain the rest of the fitted profile uses. Any one of them failing
        // walks the Boole step exactly as before.
        let fitted = if P::FITTED_UPPER_SEGMENT
            && middle_was_planned
            && altitude_km <= UPPER_FIT_ALT_HI
            && (FITTED_V7_TEXO_LO..=FITTED_V7_TEXO_HI).contains(&exospheric_temperature)
        {
            Some(fitted_upper_segment(
                altitude_km,
                z,
                zend,
                tc,
                exospheric_temperature,
                ain,
            ))
        } else {
            None
        };
        // `zend` is deliberately not carried out of either arm: nothing below
        // reads it. `z` is, because the `jb_semian` guard does.
        let (sum3, tloc4) = if let Some(state) = fitted {
            z = state.z;
            (state.sum3, state.tloc4)
        } else {
            let walked = walked_upper_segment::<P>(altitude_km, z, zend, temperature, ain)?;
            z = walked.z;
            (walked.sum3, walked.tloc4)
        };
        let (altr, h_sign) = if altitude_km <= 500.0 {
            fact2 = sub2 / JB_RSTAR;
            ((tloc3 / tloc2).ln(), 1.0)
        } else {
            fact2 = (sub2 + sum3) / JB_RSTAR;
            ((tloc4 / tloc2).ln(), -1.0)
        };
        let [alpha0, alpha1, alpha2, alpha3, alpha4] = JB_ALPHA;
        let [mass0, mass1, mass2, mass3, mass4, mass5] = JB_AMW;
        let [log0, log1, log2, log3, log4, log5] = &mut log_number_density;
        *log0 = *log0 - (1.0 + alpha0) * altr - fact2 * mass0;
        *log1 = *log1 - (1.0 + alpha1) * altr - fact2 * mass1;
        *log2 = *log2 - (1.0 + alpha2) * altr - fact2 * mass2;
        *log3 = *log3 - (1.0 + alpha3) * altr - fact2 * mass3;
        *log4 = *log4 - (1.0 + alpha4) * altr - fact2 * mass4;
        let al10t5 = exospheric_temperature.log10();
        let alnh5 = (5.5 * al10t5 - 39.40) * al10t5 + 73.13;
        *log5 = std::f64::consts::LN_10 * (alnh5 + 6.0)
            + h_sign * ((tloc4 / tloc3).ln() + sum3 * mass5 / JB_RSTAR);
    }
    let semiannual_correction = if z < 2000.0 {
        hoisted_semiannual_correction?
    } else {
        0.0
    };
    let dlr = std::f64::consts::LN_10 * (latitude_correction + semiannual_correction);
    for entry in &mut log_number_density {
        *entry += dlr;
    }
    #[cfg(test)]
    SPECIES_CAPTURE.with(|capture| {
        let mut pairs = [(0.0, 0.0); 6];
        for (slot, (factor, offset)) in pairs
            .iter_mut()
            .zip(density_factor.iter().zip(log_number_density.iter()))
        {
            *slot = (*factor, *offset);
        }
        capture.set(pairs);
    });
    // The six `exp` calls below are the ones R44 refuted vectorising, and that
    // refutation is REFUTED ON DARWIN, OPEN ON PRODUCTION SILICON.
    //
    // R44 priced a scalar `exp` at 2.16 ns on an M1 Pro and put the six at 1.71%
    // of the arc, under the project's 1.13% null floor. glibc's `exp` is
    // 8.54 ns (`docs/PMU_PROFILE.md` §7), and §10.1's measured `exp` row is
    // 4.59% of the arc — of which this loop is 6 of the kernel's 10 calls, i.e.
    // **2.75% of the production arc**. The lever was closed on the wrong host.
    // See `PMU_PROFILE.md` §11a before treating it as settled either way.
    //
    // Two of R44's findings do survive the host change and should not be
    // retried: the six arguments are independent so the scalar calls already
    // overlap, and HOISTING these out of the compensated sum below is a LOSS
    // (+2.69% on the block, landed 40dc693, reverted c2e3936).
    let mut sum = 0.0;
    let mut compensation = 0.0;
    let mut simple_sum = 0.0;
    for ((log_offset, factor), molecular_weight) in
        log_number_density.iter().zip(density_factor).zip(JB_AMW)
    {
        let exponential = if CEILING_PROBE == 2 {
            1.0 + log_offset
        } else {
            log_offset.exp()
        };
        let term = factor * exponential * molecular_weight;
        let tmp = term - compensation;
        let next = sum + tmp;
        compensation = (next - sum) - tmp;
        sum = next;
        simple_sum += term;
    }
    let final_sum = sum - compensation;
    let species_sum = if final_sum.is_nan() && simple_sum.is_infinite() {
        simple_sum
    } else {
        final_sum
    };
    rho = species_sum / JB_AVOGAD;
    let fex = jb_density_correction(altitude_km, input.f10b);
    let output = rho * fex;
    if output.is_finite() && output > 0.0 {
        Ok(output)
    } else {
        Err(Jb2008Error::NumericalDomain)
    }
}

/// The `dz <= 0` branch of `jb_local_temp`, which is the only one production
/// reaches.
///
/// `jb_density` evaluates the scalar temperature at exactly one altitude, the
/// fixed 90 km lower bound of the first quadrature segment, so `dz` is `-35` on
/// every call. The two-branch form nevertheless carried a `sqrt` and an `atan`
/// into the kernel at that call site — the same shape, and the same cost, that
/// `jb_local_temp_below_break_x4` exists to keep out of the Boole loop.
#[inline]
fn jb_local_temp_below_break(z_km: f64, tc: [f64; 4]) -> f64 {
    let dz = z_km - 125.0;
    ((-9.820_469_5e-6 * dz - 7.303_974_2e-4) * dz * dz + 1.0) * dz * tc[1] + tc[0]
}

/// `x^2.5` for `x >= 0` without a `powf`.
///
/// This is the same identity the wide upper temperature branch uses, and it is
/// also the scalar oracle that branch is written against. `jb_tsub_l`'s two
/// `powf(2.5)` calls now route through it as well.
///
/// **That substitution moves results, and here is by how much.** It was left
/// out for a long time on the grounds that the move was unquantified and the
/// density pins in this module are tolerances rather than bit tests, so they
/// would not catch it. Measured rather than assumed:
///
/// - Against `powf(2.5)`, over the 2e6-point sweep of `[0, 1]` that is the
///   only domain `jb_tsub_l` produces, the worst relative difference is
///   3.927e-16 — 2 ULP, at `v = 0.065_653_5`. Over the wider sweep
///   `positive_five_halves_helper_tracks_libm_reference` asserts on, which runs
///   to 34,875, the worst is 3.520e-16 at `v = 5931.1`. Neither is near that
///   test's 5e-15 bound.
/// - End to end, over the 278-input corpus in `tests/jb2008_libm_probe.rs`,
///   only 4 of 278 exact-profile densities move at all and the median move is
///   zero, but the ones that move amplify: worst 3.474e-15 relative, 25 ULP.
///   Density is exponentially sensitive to the exospheric temperature this
///   feeds, so a 2 ULP input perturbation comes out about 9x larger.
///
/// 3.474e-15 is the number the old comment was missing. It sits 144x inside
/// the 5e-13 tolerance the Orekit JAR vectors are asserted at, which is why
/// those assertions do not move — but the whole-density bit pins downstream
/// DO, and this change re-pins them.
///
/// Both call sites are in domain: `eta` and `theta` are each a half-magnitude
/// of a sum or difference of two angles that `jb2008_density` has already
/// range-checked into `[-pi/2, pi/2]`, so both lie in `[0, pi/2]`, and `cos`
/// and `sin` are non-negative across it. The `sqrt` therefore never sees a
/// negative argument.
#[inline]
fn jb_positive_five_halves(value: f64) -> f64 {
    value * value * value.sqrt()
}

/// Test-only scalar reference for the whole temperature profile.
///
/// See `jb_local_temp_below_break` for why `jb_density` no longer calls this.
/// The full two-branch form survives as the oracle the `f64x4` forms are pinned
/// against.
#[cfg(test)]
#[inline]
fn jb_local_temp(z_km: f64, tc: [f64; 4]) -> f64 {
    let dz = z_km - 125.0;
    if dz <= 0.0 {
        jb_local_temp_below_break(z_km, tc)
    } else {
        tc[0] + tc[2] * (tc[3] * dz * (1.0 + 4.5e-6 * jb_positive_five_halves(dz))).atan()
    }
}

/// Four abscissae of `jb_local_temp` at once.
///
/// # Why this exists
///
/// The `atan` on the `dz > 0` branch above is the single most expensive thing
/// in a production propagation. It is reached 224-260 times per RHS evaluation
/// — not from any coordinate reduction, but from the three Boole quadrature
/// loops below, which integrate the barometric equation from 90 km to the
/// satellite on every call. Measured share of one strict-HF arc: 42% of leaf
/// samples on Apple libm, and the same call count against a 9.868 ns glibc
/// `atan` on znver2.
///
/// Boole's rule uses `JB_WT[1..5]`, i.e. exactly FOUR independent abscissae per
/// step, so `f64x4` is the natural width and no restructuring of the quadrature
/// is needed to reach it.
///
/// # What is and is not bit-identical
///
/// The caller still walks `z` with the same sequential `z += dz` additions and
/// still accumulates `sum1` in the same order, so the abscissae and the
/// reduction are bit-identical to the scalar form. `jb_mbar` and `jb_gravity`
/// stay scalar for the same reason. **The only value that changes is the
/// temperature**, and only through the `atan`: `wide` uses the Cephes rational
/// (Agner Fog's vector class library) where libm is correctly rounded, measured
/// at max 1.937 ULP / 2.150e-16 relative over the operand range this call site
/// actually produces.
///
/// Both branches are evaluated and selected between rather than branched on,
/// because lanes straddle 125 km near the start of the middle segment.
/// `dz.sqrt()` is NaN on the lanes taking the low branch; `select` is a
/// lane-wise mask select and performs no arithmetic on the lanes it drops, so
/// those lanes are discarded before the NaN can propagate.
#[inline]
fn jb_local_temp_x4(z_km: wide::f64x4, tc: TemperatureBroadcast) -> wide::f64x4 {
    use wide::f64x4;

    let dz = z_km - Z_BREAK_X4;
    let low = jb_local_temp_below_break_x4(z_km, tc);
    let high = jb_local_temp_above_break_x4(z_km, tc);
    dz.simd_le(f64x4::ZERO).select(low, high)
}

/// `f64x4::new` is a `const fn`; `splat` is not. That difference is worth 25% of
/// a profile: with `splat`, LLVM materialises each literal vector on every call
/// through `_platform_memset_pattern16`, which went from 23 to 4312 samples the
/// first time this code was written that way. As a `const` it is a rodata load.
const Z_BREAK_X4: wide::f64x4 = wide::f64x4::new([125.0; 4]);

/// Four abscissae of `jb_local_temp` at once, for the branch above 125 km only.
///
/// # Why this exists separately from `jb_local_temp_x4`
///
/// The blended form evaluates both branches on every step because lanes straddle
/// 125 km. They only straddle it once. `z` is monotone increasing across all
/// three quadrature segments, so once the lowest abscissa of a step clears
/// 125 km, no later step can need the polynomial branch — yet the blended form
/// went on computing it, and the compare and select with it, for every remaining
/// step of a segment that runs to 500 km or beyond.
///
/// Selecting this directly is bit-identical rather than close: when no lane
/// satisfies `dz <= 0` the `select` returns exactly `high`, so skipping the low
/// half cannot change the value it discards.
#[inline]
fn jb_local_temp_above_break_x4(z_km: wide::f64x4, tc: TemperatureBroadcast) -> wide::f64x4 {
    use wide::f64x4;
    const C_FIVE_HALVES: f64x4 = f64x4::new([4.5e-6; 4]);

    let dz = z_km - Z_BREAK_X4;
    let five_halves = dz * dz * dz.sqrt();
    let argument = tc.argument_scale * dz * (f64x4::ONE + C_FIVE_HALVES * five_halves);
    tc.base + tc.amplitude * atan_x4_dispatched(argument)
}

/// `jb_local_temp_x4` when the caller has already established that the step sits
/// wholly on one side of 125 km, and the blended form only where it straddles.
///
/// `boole_abscissae` accumulates `*z += dz` with `dz > 0`, so `zs` is strictly
/// ascending: `zs[0]` is the minimum and `zs[3]` the maximum. Both one-sided
/// tests are therefore whole-step tests.
///
/// The low arm is the mirror of the high one and rests on the same selection
/// argument: `jb_local_temp_x4` selects with the mask `dz <= 0`, so when every
/// lane satisfies it the `select` returns exactly `low` and the discarded
/// `high` half cannot have influenced the result. It is not a near-equal
/// shortcut.
///
/// The low arm is not a rare case. The middle quadrature segment starts at
/// 105 km, so its first step's abscissae are all below 125 km while every later
/// step is above; without this arm that step ran the high branch — including
/// `dz.sqrt()` of a negative `dz`, i.e. four NaN lanes — and threw all four
/// results away, one wasted `atan_x4` of the 17-19 per evaluation, on every
/// evaluation.
///
/// `z_km` is `zs` in vector form. The caller already needs it for
/// `jb_gravity_x4`, so it is passed in rather than rebuilt here.
#[inline]
fn jb_local_temp_step_x4(zs: [f64; 4], z_km: wide::f64x4, tc: TemperatureBroadcast) -> wide::f64x4 {
    if zs[0] > 125.0 {
        jb_local_temp_above_break_x4(z_km, tc)
    } else if zs[3] <= 125.0 {
        jb_local_temp_below_break_x4(z_km, tc)
    } else {
        jb_local_temp_x4(z_km, tc)
    }
}

/// `wide::f64x4::atan` with its one non-`const` constant promoted to `const`.
///
/// # Why a local copy exists at all
///
/// `wide-1.5.0` `f64x4_.rs:1246` writes the small-range threshold as
/// `Self::splat(0.66)` where every other constant in the same function goes
/// through `const_f64_as_f64x4!`. `splat` is not a `const fn`, so LLVM
/// materialises `[0.66; 4]` on the stack through `bl _memset_pattern16` —
/// disassembled as `w2 = #0x20` against a rodata pattern of `[0.66, 0.66]` —
/// **inside the Boole loop body, once per quadrature step**. On a campaign
/// profile `memset_pattern16` was 2.386% of all on-CPU work and both call sites
/// were in this kernel. `Cargo.toml` is hashed into `build_policy_sha256`, so a
/// `[patch]` section is not available and a vendored copy is the only route.
///
/// # This is a transliteration, and the ordering is load-bearing
///
/// The arithmetic and reduction order below follow upstream, including the two
/// polynomial evaluations. The coefficient literals are shortest-roundtrip
/// spellings of the same upstream `f64` bits. `polynomial_4!` and
/// `polynomial_5n!` are **Estrin, not Horner** — they split the polynomial into
/// even/odd halves and recombine through `x2`/`x4`. An earlier attempt rewrote
/// them as Horner and diverged on 1,613 of 3,200,000 lanes at 1 ULP. The
/// 4.8-million-lane bit test below guards this transcription.
///
/// # `select` here is not a semantics change from the old `blend`
///
/// The five selections below once read `.blend(..)`. An earlier reader declined
/// to convert them, believing `blend -> select` would alter the range
/// reduction. It does not, and the reasoning is worth keeping because the
/// mistake is easy to repeat.
///
/// In `wide` 1.6.0 `blend` is a deprecated one-line shim whose whole body is
/// `self.select(if_true, if_false)` (`simd.rs`, inside `macro_rules! impl_simd`)
/// — same operand order, so the first argument wins on a true lane before and
/// after. `blend` never had semantics of its own to change.
///
/// The function that *is* different is `bitselect(if_one, if_zero)`, the per-bit
/// `(if_one & mask) | (!mask & if_zero)` form. That is the **wrong** answer at
/// every site here: these masks come from `simd_le`/`simd_ge`/`is_sign_negative`
/// and are already all-ones or all-zeros, so `bitselect` would buy nothing and
/// would silently mean something else if a mask ever stopped being lane-uniform.
///
/// Independently: upstream's own `f64x4::atan` already calls `.select(..)` at
/// these exact five sites with matching operand order, so this spelling moves
/// the vendored copy *toward* `vendored_atan_x4_is_bit_identical_to_wide`, the
/// oracle it is tested against — not away from it.
///
/// Adapted from `wide` 0.7/1.x (Zlib OR Apache-2.0 OR MIT), itself based on
/// Agner Fog's vector class library `vectormath_trig.h`.
#[inline]
fn atan_x4(value: wide::f64x4) -> wide::f64x4 {
    use wide::f64x4;

    const MORE_BITS: f64x4 = f64x4::new([6.123_233_995_736_766e-17; 4]);
    const MORE_BITS_O2: f64x4 = f64x4::new([6.123_233_995_736_766e-17 * 0.5; 4]);
    const T3PO8: f64x4 = f64x4::new([std::f64::consts::SQRT_2 + 1.0; 4]);
    // The one upstream writes as `Self::splat(0.66)`.
    const SMALL: f64x4 = f64x4::new([0.66; 4]);

    let absolute_value = value.abs();

    let notbig = absolute_value.simd_le(T3PO8);
    let notsmal = absolute_value.simd_ge(SMALL);

    let mut offset = notbig.select(f64x4::FRAC_PI_4, f64x4::FRAC_PI_2);
    offset = notsmal & offset;
    let mut fac = notbig.select(MORE_BITS_O2, MORE_BITS);
    fac = notsmal & fac;

    let mut numerator = notbig & absolute_value;
    numerator = notsmal.select(numerator - f64x4::ONE, numerator);
    let mut denominator = notbig & f64x4::ONE;
    denominator = notsmal.select(denominator + absolute_value, denominator);
    let reduced_argument = numerator / denominator;

    let squared_argument = reduced_argument * reduced_argument;

    let numerator_polynomial = atan_poly_p(squared_argument);
    let denominator_polynomial = atan_poly_q(squared_argument);

    let mut result = (numerator_polynomial / denominator_polynomial)
        .mul_add(reduced_argument * squared_argument, reduced_argument);
    result += offset + fac;

    result = (value.is_sign_negative()).select(-result, result);

    result
}

/// The `1 + sqrt(2)` range-reduction threshold.
///
/// `atan_x4` spells this locally as `T3PO8`; the two are the same value, hoisted
/// here so `atan_x4_dispatched` can compare against it without reaching into
/// that function's body.
const T3PO8_X4: wide::f64x4 = wide::f64x4::new([std::f64::consts::SQRT_2 + 1.0; 4]);

/// `atan_x4` specialised to arguments that are strictly above `1 + sqrt(2)` on
/// every lane.
///
/// # This is a reduction of the general body, not a new approximation
///
/// With `value > 1 + sqrt(2) > 0.66` on all four lanes, every mask in `atan_x4`
/// is lane-uniform and known at the call site, so each masked operation
/// collapses to one of its two arms:
///
/// * `absolute_value` is `value`, because the lanes are positive.
/// * `notbig = |v| <= 1+sqrt(2)` is all-false, so `offset` selects `FRAC_PI_2`
///   and `fac` selects `MORE_BITS`.
/// * `notsmal = |v| >= 0.66` is all-true, so `notsmal & offset` and
///   `notsmal & fac` are the identity and both survive as constants.
/// * `numerator = notbig & |v|` is `+0.0`; `notsmal.select(numerator - 1, ..)`
///   is then exactly `-1.0`.
/// * `denominator = notbig & 1.0` is `+0.0`; `notsmal.select(0.0 + |v|, ..)` is
///   then exactly `value`.
/// * `value.is_sign_negative()` is all-false, so the final `select(-r, r)`
///   returns `r` unchanged.
///
/// `result += offset + fac` evaluates `offset + fac` first, and here both are
/// compile-time constants, so `BIG_OFFSET` is that sum const-folded — the same
/// IEEE addition of the same two values, performed once at compile time rather
/// than once per call. The remaining arithmetic (the reciprocal, the two
/// polynomials, the `mul_add`) is byte-for-byte the general body's.
///
/// `atan_x4_large_matches_the_general_body_bitwise` is the oracle for all of
/// the above; this comment is the argument, that test is the evidence.
#[inline]
fn atan_x4_above_t3po8(value: wide::f64x4) -> wide::f64x4 {
    use wide::f64x4;

    /// `FRAC_PI_2 + MORE_BITS`, the `offset + fac` the general body forms on
    /// this branch, const-folded.
    ///
    /// That sum is bit-for-bit `FRAC_PI_2`: `MORE_BITS` is 6.123e-17 and half
    /// an ULP of `FRAC_PI_2` is 1.110e-16, so it rounds away.
    ///
    /// It rounds away on the other branch too — `MORE_BITS_O2` is 3.062e-17
    /// against a half ULP of 5.551e-17 for `FRAC_PI_4` — so in `atan_x4` as a
    /// whole the `fac` term never changes a bit. That is a property of the
    /// vector form, not of the algorithm: Cephes keeps `MOREBITS` meaningful by
    /// adding it to the reduced argument, which is small, whereas `wide` forms
    /// `offset + fac` first and adds the pair to a result of order 1.
    ///
    /// Written in its derived form anyway, because the absorption is the
    /// surprise and not the premise, and because dropping the term here would
    /// make this function stop looking like the branch it replaces.
    /// `atan_x4_large_matches_the_general_body_bitwise` pins it.
    const BIG_OFFSET: f64x4 =
        f64x4::new([std::f64::consts::FRAC_PI_2 + 6.123_233_995_736_766e-17; 4]);
    const MINUS_ONE: f64x4 = f64x4::new([-1.0; 4]);

    let reduced_argument = MINUS_ONE / value;
    let squared_argument = reduced_argument * reduced_argument;

    let numerator_polynomial = atan_poly_p(squared_argument);
    let denominator_polynomial = atan_poly_q(squared_argument);

    let result = (numerator_polynomial / denominator_polynomial)
        .mul_add(reduced_argument * squared_argument, reduced_argument);
    result + BIG_OFFSET
}

/// Lower bound on every lane of `value` for [`atan_x4_asymptotic`], as a scalar.
///
/// See that function for why 64 and not the smallest workable number.
const ATAN_ASYMPTOTIC_MIN: f64 = 64.0;

/// [`ATAN_ASYMPTOTIC_MIN`] broadcast, for the guard in [`atan_x4_dispatched`].
const ATAN_ASYMPTOTIC_MIN_X4: wide::f64x4 = wide::f64x4::new([ATAN_ASYMPTOTIC_MIN; 4]);

/// `atan_x4_above_t3po8` with Cephes' rational replaced by the Taylor series it
/// converges to, for arguments far enough out that the two agree to the bit.
///
/// # What is being replaced
///
/// [`atan_x4_above_t3po8`] forms `r = -1/v` and returns
/// `R(r^2) * r^3 + r + pi/2`, where `R = P/Q` is Cephes' rational approximation
/// to `(atan(r) - r) / r^3` over `|r| <= 1/(1+sqrt(2))`. Evaluating `R` costs
/// two five-term polynomials and a second vector divide, and its whole job on
/// this branch is to reproduce a function whose Taylor series in `x = r^2` is
/// `-1/3 + x/5 - x^2/7 + x^3/9 - x^4/11 + ...`. Five terms of that series, in
/// Horner form, are four `mul_add`s and no divide.
///
/// # How close it comes to bit-identical, and why it does not arrive
///
/// The series is truncated, so it is not `R`, and the question is whether the
/// difference survives the two roundings that follow it. Take `|r| <= 1/64`.
///
/// * The first term dropped is `x^5/13`, so the series differs from the true
///   `(atan(r) - r)/r^3` by at most `(1/64)^10 / 13 = 6.6e-20`.
/// * `R` differs from that same true value by Cephes' own approximation error
///   plus its evaluation rounding, both of order `1e-17` relative on a quantity
///   of order `1/3`, i.e. a few times `1e-18`.
/// * So `|series - R| < 1e-17`, and the term it multiplies is `r^3`, at most
///   `3.8e-6`. The two `mul_add`s therefore see exact products that differ by
///   less than `4e-23`.
/// * That `mul_add` adds the product to `r`, whose magnitude is at least
///   `2^-53` times its own exponent — for the largest admitted `|r| = 1/64` the
///   ULP of the sum is `3.5e-18`, and it only grows relative to the discrepancy
///   as `r` shrinks, because the discrepancy falls as `r^3` and the ULP falls as
///   `r`. The two exact results are `1e-5` of an ULP apart at the worst
///   admitted argument.
///
/// A gap of `1e-5` ULP still rounds differently when the exact value happens to
/// sit that close to a rounding boundary, and it does:
/// `atan_x4_asymptotic_matches_the_general_body_bitwise` sweeps four million
/// admitted lanes and **13 of them move, every one by exactly 1 ULP**. The rate
/// is 3.3e-6 and the term that sets it is Cephes' own evaluation rounding of
/// `P/Q`, not the truncation, so it falls only as `1/T^2` in the threshold and
/// never reaches zero. **This function is a 1-ULP approximation of the general
/// body, not a spelling of it**, which is why [`ATAN_ASYMPTOTIC`] ships false.
///
/// # Why 64
///
/// Lower is worse in the obvious way: at `|r| = 1/16` the margin is `1e-2` ULP
/// and the arms would disagree on a few percent of lanes. Higher is bounded by
/// the flown kernel's own arguments: the model-7 upper Boole step evaluates
/// `tc[3] * dz * (1 + 4.5e-6 * dz^2.5)` at `dz = z - 125` between 375 and
/// 861 km, and over the censused exospheric temperature band (608.9--1627.5 K)
/// that argument's minimum works out near 100 — the hot end of the band flattens
/// `tc[3]`, so the smallest argument comes from the highest temperature and the
/// shortest step, not from the coldest. 64 leaves that a 1.6x margin, and
/// raising the threshold to where the divergence rate mattered would put it
/// above the arguments the lever exists to serve.
#[inline]
fn atan_x4_asymptotic(value: wide::f64x4) -> wide::f64x4 {
    use wide::f64x4;

    /// `FRAC_PI_2 + MORE_BITS`, as [`atan_x4_above_t3po8`] const-folds it. Same
    /// value, same reason; see that function's own constant.
    const BIG_OFFSET: f64x4 =
        f64x4::new([std::f64::consts::FRAC_PI_2 + 6.123_233_995_736_766e-17; 4]);
    const MINUS_ONE: f64x4 = f64x4::new([-1.0; 4]);
    const S0: f64x4 = f64x4::new([-1.0 / 3.0; 4]);
    const S1: f64x4 = f64x4::new([1.0 / 5.0; 4]);
    const S2: f64x4 = f64x4::new([-1.0 / 7.0; 4]);
    const S3: f64x4 = f64x4::new([1.0 / 9.0; 4]);
    const S4: f64x4 = f64x4::new([-1.0 / 11.0; 4]);

    let reduced_argument = MINUS_ONE / value;
    let squared_argument = reduced_argument * reduced_argument;
    let series = S4
        .mul_add(squared_argument, S3)
        .mul_add(squared_argument, S2)
        .mul_add(squared_argument, S1)
        .mul_add(squared_argument, S0);
    series.mul_add(reduced_argument * squared_argument, reduced_argument) + BIG_OFFSET
}

/// `atan_x4`, routed to [`atan_x4_above_t3po8`] when every lane qualifies.
///
/// # What this is worth, measured
///
/// Censused on the sealed V3 arc's altitude band (626--986 km), `atan_x4` runs
/// **16--18 times per kernel call** — 15 from the 105--500 km plan (one
/// straddling step plus its 14 wholly-above-break steps) and 1--3 from the
/// segment above 500 km. Of those, **70.7% have all four lanes above
/// `1 + sqrt(2)`** and take this fast path; 11.7% are all-mid, 11.1% all-small,
/// 6.5% mixed.
///
/// M1 Max, `--release`, same binary, arms selected by a `const` so neither arm
/// carries a runtime flag, interleaved and alternated:
///
/// * the specialised body alone, on qualifying operands: 8.95 -> 7.05 ns,
///   **-21.2%**;
/// * one evaluation's post-break middle steps with the guard included:
///   8.80 -> 7.74 ns per call, **-12.0%**;
/// * the whole JB2008 kernel call: 450.1 -> 421.5 ns, **-6.46%** — median of
///   401 paired rounds, 395 of them negative, p25/p75 -6.55%/-6.35%;
/// * the whole strict-HF production arc, at exactly 1.0000 kernel calls per RHS
///   evaluation: 1328.8 -> 1300.3 ns/eval, **-2.14% of arc wall**, negative in
///   all nine paired rounds.
///
/// The last two are independent measurements of the same thing and agree to
/// 0.6 ns: -29.08 ns/eval derived from the kernel A/B against -28.49 ns/eval
/// measured directly on the arc.
///
/// # A THIRD instrument reads this change at -4.74%, and the number to quote
/// # depends on which arc you mean
///
/// A commit-level whole-arc A/B (`lightyear_odeint_rs`'s `prop_timing`
/// harness, 4458015-era, n=18 rotating, min-of-block) prices the same interval
/// -- 0ecabc1 to 70ce67e, i.e. this commit plus 2673050 -- at **-4.72% +/-
/// 0.05%**, split as **-4.74% +/- 0.06% for this commit** and **-0.03% +/-
/// 0.04% for 2673050**, which is null. That is 2.2x the -2.14% above, on what
/// is provably the same code interval, so one of the two arcs is not the other
/// and NEITHER number may be quoted as "the atan number" without naming its
/// workload.
///
/// It is very likely a DENOMINATOR disagreement, not a numerator one. The
/// saving above is 28.49 ns/eval against an arc costing 1328.8 ns/eval. For
/// that same fixed saving to read 4.74% the arc underneath it must cost about
/// 601 ns/eval. So the two instruments need not disagree about what this
/// commit saves at all -- they disagree about what an RHS evaluation costs,
/// by roughly the same 2.2x. That is falsifiable and cheap: measure ns/eval on
/// both arcs, and if they land near 1328.8 and near 601 then both percentages
/// are correct as stated and the ratio is fully explained.
///
/// The arcs differ, by inspection, in a way that is sufficient to make them
/// non-comparable even though it has not been shown to be quantitatively the
/// whole 2.2x:
///
/// * **Atmosphere model.** The arc above is `v3_frozen_config`, which overrides
///   `atm_model` to `part_a_hybrid().atmosphere_model` -- 5, the approximation
///   compiled science actually flies. The `prop_timing` arc used
///   `production_dust_config` unmodified, which hardcoded `atm_model: 4`, the
///   exact profile. Two different atmospheres, one line apart in the same
///   crate. **This is the whole 2.2x** -- see the arbitration table on
///   `rhs.rs`'s `eclipse_sun_direction_path_bound`, which re-ran the same
///   commit at both models and got +4.40% and +1.59%.
///
///   Both instruments have since been fixed. `prop_timing` carries `_m4` and
///   `_m5` arms with the model in every printed line, and `strict_hf_pin` has a
///   model-5 accuracy gate beside its model-4 one. The figure above is a
///   model-5 number and stays valid; the `prop_timing` figure below it is a
///   model-4 number and must be quoted as one.
/// * **Eclipse events.** The `prop_timing` figure is its `events` arm, so its
///   denominator carries the whole eclipse coordinator.
/// * **Epoch and span.** `prop_timing` runs JD 2460310.5 for 43200 s; the arc
///   above is the sealed V3 arc.
///
/// What the arcs do NOT differ in is the altitude shell, which is the first
/// thing anyone will reach for: `prop_timing`'s elements
/// (`a = 7178.137 km, e = 0.025`) give 620.5--979.5 km against the 626--986 km
/// censused above. Do not spend time there.
///
/// **The segment above 500 km is half the win.** With only the 105--500 km plan
/// specialised the kernel moves -3.31%; adding the 1--3 calls above 500 km
/// takes it to -6.46%. Those calls carry the largest `dz` in the profile, so
/// every one of them clears the threshold — a handful of calls at a 100% hit
/// rate against fifteen at 70.7%.
///
/// Two measurement notes, both paid for. Selecting the arm with an atomic read
/// *inside* `atan_x4_dispatched` inflated the arc delta to -2.89%: the load
/// cannot be hoisted out of the Boole loop and it perturbs scheduling around
/// the very calls being measured. And the first attempt at the kernel A/B ran
/// nine long rounds on a box at load 10, which returned paired deltas from
/// -54% to +37%; many short paired rounds with the median taken over the
/// RATIOS, not over the arms, is what makes the number survive a busy host.
///
/// The block is NOT divide-bound, which is what makes this worth doing: two
/// ceiling probes that deleted a divide outright (wrong answers, timing only)
/// bought a further 2.3% and 4.2%, so the masking and selection scaffolding —
/// not the two `fdiv`s — was the cost, and removing it captures nearly all of
/// the available headroom.
///
/// # Why the guard is a four-lane test and not a lane-0 test
///
/// Within one Boole step the abscissae ascend, so `break_offset` and
/// `argument_shape` both ascend and the argument is ordered across lanes —
/// *provided* `tc.argument_scale` is positive. It is, on every physical input
/// (`tc[3] = tc[1] / tc[2]` with `tc[1] = 0.0543 (Tx - 183)` and
/// `tc[2] = (Tinf - Tx) / (pi/2)`, both positive whenever `Tx > 183 K` and
/// `Tinf > Tx`), but nothing in the kernel enforces it, and a lane-0 test that
/// silently assumed it would take the fast path on an argument whose upper
/// lanes had fallen below the threshold.
///
/// Measured, that assumption is worth 0.22 ns per call — 3 ns on a whole
/// evaluation, against a 502 ns kernel call. The four-lane test costs well
/// under a tenth of what the specialisation returns, so trading soundness for
/// it is not a trade worth making.
///
/// A NaN lane (the straddling step feeds four of them, by construction — see
/// `jb_local_temp_x4`) compares false and falls through to the general body,
/// which is where those lanes were always evaluated.
#[inline]
fn atan_x4_dispatched(value: wide::f64x4) -> wide::f64x4 {
    if ATAN_ASYMPTOTIC && value.simd_gt(ATAN_ASYMPTOTIC_MIN_X4).all() {
        atan_x4_asymptotic(value)
    } else if value.simd_gt(T3PO8_X4).all() {
        atan_x4_above_t3po8(value)
    } else {
        atan_x4(value)
    }
}

/// Whether [`atan_x4_dispatched`] offers the asymptotic arm at all.
///
/// **FALSE, deliberately: built, measured and PARKED.** It is priced at
/// **-0.90% of the model-7 kernel call** (median of 24 rotating paired rounds on
/// an M1 Pro at load 25, 17 of 24 negative; min-of-min -0.93%), i.e. about
/// -0.33% of the production arc — real, reproducible, and too small to be worth
/// the one thing it cannot avoid costing.
///
/// What it cannot avoid: it is **not** bit-identical. Over a four-million-lane
/// sweep above the threshold, 13 lanes move, all by exactly 1 ULP of the
/// arctangent — a rate of 3.3e-6. That is irreducible rather than a tuning
/// failure. The residual is Cephes' own evaluation rounding of `P/Q`, not the
/// series truncation, so raising the threshold shrinks the rate as `1/T^2`
/// without ever reaching zero, and the flown arguments bottom out near 100 so
/// the threshold cannot be raised far. A three-decade rate reduction is not
/// available and a bit-identity claim is therefore not available.
///
/// Turning it on would move models 4, 5 and 6 as well as 7, because the guard is
/// on the argument and not on the profile — so it would put the sealed Orekit
/// oracle at risk to buy 0.33% of arc. Confining it to model 7 means threading
/// the profile generic through `jb_local_temp_step_x4`,
/// `jb_local_temp_above_break_x4`, `planned_temperature_above` and
/// `accumulate_middle_step`, which is a lot of surface for the price.
///
/// A `const` rather than a runtime flag, for the reason
/// [`atan_x4_dispatched`]'s own note records: selecting the arm with a load
/// inside the Boole loop perturbs the scheduling around the calls being timed,
/// and inflated an arc delta by a third the last time it was tried. Two
/// binaries, one `const` apart, is the A/B the figure above was measured with.
///
/// # What this measurement settles, beyond its own lever
///
/// `docs/JB2008_COST_MAP.md` §4 records a sampler putting **17.34%** of the
/// kernel call on `jb_local_temp_step_x4`, `atan_x4` included, and prices the
/// remaining levers off that occupancy. This arm deletes a vector divide and two
/// five-term polynomials from that block — most of its arithmetic — and the
/// whole kernel moves 0.90%. **Sampler occupancy on this block is not a
/// recoverable cost**, and any lever priced by multiplying that 17.34% is priced
/// too high. The ceiling probe in the same round measures the honest figure for
/// the whole block: see `CEILING_PROBE` arm 1.
const ATAN_ASYMPTOTIC: bool = false;

/// `polynomial_4!(x, P0..P4)` from `wide`'s `lib.rs`, in the same Estrin order.
#[inline]
fn atan_poly_p(x: wide::f64x4) -> wide::f64x4 {
    use wide::f64x4;
    const P4: f64x4 = f64x4::new([-0.875_060_860_003_190_4; 4]);
    const P3: f64x4 = f64x4::new([-16.157_537_187_333_652; 4]);
    const P2: f64x4 = f64x4::new([-75.008_557_923_147_05; 4]);
    const P1: f64x4 = f64x4::new([-122.886_668_449_013_61; 4]);
    const P0: f64x4 = f64x4::new([-64.850_219_049_420_25; 4]);
    let x2 = x * x;
    let x4 = x2 * x2;
    P3.mul_add(x, P2).mul_add(x2, P1.mul_add(x, P0)) + P4 * x4
}

/// `polynomial_5n!(x, Q0..Q4)` from `wide`'s `lib.rs`, in the same Estrin order.
/// The monic term enters as `$c4 + x`, not as a separate power.
#[inline]
fn atan_poly_q(x: wide::f64x4) -> wide::f64x4 {
    use wide::f64x4;
    const Q4: f64x4 = f64x4::new([24.858_464_901_423_062; 4]);
    const Q3: f64x4 = f64x4::new([165.027_009_831_698_85; 4]);
    const Q2: f64x4 = f64x4::new([432.881_060_491_290_27; 4]);
    const Q1: f64x4 = f64x4::new([485.390_399_635_913_7; 4]);
    const Q0: f64x4 = f64x4::new([194.550_657_148_261_4; 4]);
    let x2 = x * x;
    let x4 = x2 * x2;
    x2.mul_add(x.mul_add(Q3, Q2), x4.mul_add(Q4 + x, x.mul_add(Q1, Q0)))
}

/// Four abscissae of `jb_local_temp` at once, for the branch below 125 km only.
///
/// # Why this exists separately from `jb_local_temp_x4`
///
/// The lowest quadrature segment runs 90 km -> min(alt, 105) km, so every
/// abscissa has `dz = z - 125 <= -20`. The scalar `jb_local_temp` therefore
/// always takes the polynomial branch there — but the compiler cannot prove it,
/// so it emitted the `atan` branch into the loop body as a real `bl _atan`.
/// That call was never executed, and it still cost: it blocked SLP
/// vectorisation of the whole step and forced the vector state to be shuffled
/// and spilled around a call that never happened. Measured on a campaign
/// profile, this segment held **17.9% of the kernel** while doing 16 abscissae
/// of cheap polynomial work against the middle segment's ~60 abscissae of
/// `atan`, i.e. it cost MORE per abscissa than the segment that computes an
/// arctangent.
///
/// Only the taken branch is evaluated here, in the same order, so this is
/// bit-identical to the scalar form and not merely close.
#[inline]
fn jb_local_temp_below_break_x4(z_km: wide::f64x4, tc: TemperatureBroadcast) -> wide::f64x4 {
    use wide::f64x4;
    const C2: f64x4 = f64x4::new([-9.820_469_5e-6; 4]);
    const C1: f64x4 = f64x4::new([7.303_974_2e-4; 4]);

    let dz = z_km - Z_BREAK_X4;
    ((C2 * dz - C1) * dz * dz + f64x4::ONE) * dz * tc.gradient + tc.base
}

/// Every wide constant in this module is a `const` ITEM, never an inline
/// `f64x4::new([x; 4])` expression, and that distinction is worth measuring
/// rather than assuming. As a `const` item the vector is rodata and loads with
/// one `ldr q`. Written inline it is materialised on the stack through
/// `bl _memset_pattern16` with `w2 = #0x20` — a libc call, in the loop body,
/// once per constant per Boole step. The first version of `jb_mbar_x4` and
/// `jb_gravity_x4` was written that way and added FOUR such calls per step to
/// the segment it was meant to make cheaper.
mod wide_const {
    use super::{JB_CXAMB, JB_EARTH_RADIUS_KM, JB_G0_M_S2};
    use wide::f64x4;

    pub(super) const HUNDRED: f64x4 = f64x4::new([100.0; 4]);
    pub(super) const EARTH_RADIUS_KM: f64x4 = f64x4::new([JB_EARTH_RADIUS_KM; 4]);
    pub(super) const G0_M_S2: f64x4 = f64x4::new([JB_G0_M_S2; 4]);
    pub(super) const LOW_TEMP_C2: f64x4 = f64x4::new([-9.820_469_5e-6; 4]);
    pub(super) const LOW_TEMP_C1: f64x4 = f64x4::new([7.303_974_2e-4; 4]);
    pub(super) const CXAMB: [f64x4; 7] = [
        f64x4::new([JB_CXAMB[0]; 4]),
        f64x4::new([JB_CXAMB[1]; 4]),
        f64x4::new([JB_CXAMB[2]; 4]),
        f64x4::new([JB_CXAMB[3]; 4]),
        f64x4::new([JB_CXAMB[4]; 4]),
        f64x4::new([JB_CXAMB[5]; 4]),
        f64x4::new([JB_CXAMB[6]; 4]),
    ];
}

/// Four abscissae of `jb_mbar` at once, same Horner order as the scalar fold.
#[inline]
fn jb_mbar_x4(z_km: wide::f64x4) -> wide::f64x4 {
    let dz = z_km - wide_const::HUNDRED;
    wide_const::CXAMB[..6]
        .iter()
        .rev()
        .fold(wide_const::CXAMB[6], |accumulator, coefficient| {
            dz * accumulator + *coefficient
        })
}

/// Four abscissae of `jb_gravity` at once.
#[inline]
fn jb_gravity_x4(z_km: wide::f64x4) -> wide::f64x4 {
    let scale = wide::f64x4::ONE + z_km / wide_const::EARTH_RADIUS_KM;
    wide_const::G0_M_S2 / (scale * scale)
}

/// The four abscissae of one Boole step, walked with exactly the additions the
/// scalar loop performed, and the running `z` left where the scalar loop left
/// it. Returned as an array so the weighted sum keeps its original order.
#[inline]
fn boole_abscissae(z: &mut f64, dz: f64) -> [f64; 4] {
    let mut out = [0.0; 4];
    for slot in &mut out {
        *z += dz;
        *slot = *z;
    }
    out
}

/// `jb_mbar(90.0)` and `jb_gravity(90.0)`.
///
/// `jb_density` integrates from a fixed 90 km lower bound, so these were a
/// six-term polynomial and a divide recomputed on every evaluation of two
/// numbers that cannot change. Rust const-evaluates `+ - * /` with the same
/// IEEE-754 round-to-nearest these functions use at run time, and neither body
/// contracts to an FMA, so the constants are the run-time values and not
/// approximations of them; `ninety_km_constants_match_the_folded_form` compares
/// the bits against an independent fold anyway.
const JB_MBAR_90_KM: f64 = jb_mbar(90.0);
const JB_GRAVITY_90_KM: f64 = jb_gravity(90.0);

/// Horner-order `jb_mbar`, spelled as a nested expression rather than a fold so
/// that it is a `const fn`. The accumulation order is the fold's, innermost
/// term first: `CXAMB[6]` scaled by `dz` against `CXAMB[5]`, and so outward.
/// Do not let a `mul_add` suggestion into it: an FMA here would move the bits.
#[inline]
const fn jb_mbar(z_km: f64) -> f64 {
    let dz = z_km - 100.0;
    let [c0, c1, c2, c3, c4, c5, c6] = JB_CXAMB;
    dz * (dz * (dz * (dz * (dz * (dz * c6 + c5) + c4) + c3) + c2) + c1) + c0
}

#[inline]
const fn jb_gravity(z_km: f64) -> f64 {
    let scale = 1.0 + z_km / JB_EARTH_RADIUS_KM;
    JB_G0_M_S2 / (scale * scale)
}

fn jb_day_of_year(mjd: f64) -> Result<f64, Jb2008Error> {
    let d1950 = mjd - 33281.0;
    let Some(mut ordinal_day) = d1950.trunc().to_i32() else {
        return Err(Jb2008Error::NumericalDomain);
    };
    let fraction = d1950 - f64::from(ordinal_day);
    ordinal_day = ordinal_day
        .checked_add(364)
        .ok_or(Jb2008Error::NumericalDomain)?;
    let mut cycle_count = ordinal_day / 1461;
    let cycle_days = cycle_count
        .checked_mul(1461)
        .ok_or(Jb2008Error::NumericalDomain)?;
    ordinal_day = ordinal_day
        .checked_sub(cycle_days)
        .ok_or(Jb2008Error::NumericalDomain)?;
    cycle_count = (ordinal_day / 365).min(3);
    let year_days = cycle_count
        .checked_mul(365)
        .ok_or(Jb2008Error::NumericalDomain)?;
    ordinal_day = ordinal_day
        .checked_sub(year_days)
        .and_then(|day| day.checked_add(1))
        .ok_or(Jb2008Error::NumericalDomain)?;
    Ok(f64::from(ordinal_day) + fraction)
}

#[inline]
fn jb_dlrsl(altitude_km: f64, mjd: f64, sat_lat_rad: f64, sin_lat: f64) -> f64 {
    // `% 1.0` lowers to a real `bl _fmod`. `x - x.trunc()` is two register ops
    // and is exact for every finite `x`, because `trunc` only clears the
    // fractional mantissa bits so the difference is representable.
    //
    // The `copysign` is NOT decoration. `fract()` alone is NOT bit-identical to
    // `% 1.0`: at any `x <= 0` with no fractional part, `fmod` returns `-0.0`
    // and `x - x.trunc()` returns `+0.0`. The guard restores `fmod`'s sign of
    // zero, and with it the pin holds over the full sweep in this module's
    // tests rather than only where the result happens to be non-zero.
    let years = (mjd - 36204.0) / 365.2422;
    let fraction = years - years.trunc();
    let cap_phi = if fraction == 0.0 {
        0.0_f64.copysign(years)
    } else {
        fraction
    };
    let sign = if sat_lat_rad >= 0.0 { 1.0 } else { -1.0 };
    let hm90 = altitude_km - 90.0;
    0.02 * hm90
        * (-0.045 * hm90).exp()
        * sign
        * (std::f64::consts::TAU * cap_phi + 1.72).sin()
        * sin_lat
        * sin_lat
}

#[inline]
fn jb_density_correction(altitude_km: f64, f10b: f64) -> f64 {
    if (1000.0..1500.0).contains(&altitude_km) {
        let zeta = (altitude_km - 1000.0) * 0.002;
        let density_correction_at_1500 =
            JB_CHT[0] + JB_CHT[1] * f10b + JB_CHT[2] * 1500.0 + JB_CHT[3] * f10b * 1500.0;
        let density_correction_slope_at_1500 = (JB_CHT[2] + JB_CHT[3] * f10b) * 500.0;
        let fex2 = 3.0 * density_correction_at_1500 - density_correction_slope_at_1500 - 3.0;
        let fex3 = density_correction_slope_at_1500 - 2.0 * density_correction_at_1500 + 2.0;
        1.0 + zeta * zeta * (fex2 + fex3 * zeta)
    } else if altitude_km >= 1500.0 {
        JB_CHT[0] + JB_CHT[1] * f10b + JB_CHT[2] * altitude_km + JB_CHT[3] * f10b * altitude_km
    } else {
        1.0
    }
}

#[inline]
fn jb_semian(input: Jb2008Input, day: f64, altitude_km: f64) -> f64 {
    let htz = altitude_km / 1000.0;
    let fsmb_fz = input.f10b - 0.70 * input.s10b - 0.04 * input.m10b;
    let fzz = JB_FZM[0]
        + fsmb_fz * (JB_FZM[1] + htz * (JB_FZM[2] + JB_FZM[3] * htz + JB_FZM[4] * fsmb_fz));
    let fsmb_gt = input.f10b - 0.75 * input.s10b - 0.37 * input.m10b;
    let tau = std::f64::consts::TAU * (day - 1.0) / 365.0;
    let (seasonal_sine, seasonal_cosine) = tau.sin_cos();
    let harmonic_sine = 2.0 * seasonal_sine * seasonal_cosine;
    let harmonic_cosine = seasonal_cosine * seasonal_cosine - seasonal_sine * seasonal_sine;
    let gtz = JB_GTM[0]
        + JB_GTM[1] * seasonal_sine
        + JB_GTM[2] * seasonal_cosine
        + JB_GTM[3] * harmonic_sine
        + JB_GTM[4] * harmonic_cosine
        + fsmb_gt
            * (JB_GTM[5]
                + JB_GTM[6] * seasonal_sine
                + JB_GTM[7] * seasonal_cosine
                + JB_GTM[8] * harmonic_sine
                + JB_GTM[9] * harmonic_cosine);
    fzz.max(1.0e-6) * gtz
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The expression `wrap_to_tau` replaced, kept verbatim so the fast form is
    /// compared against `rem_euclid` rather than against itself.
    fn wrap_to_tau_reference(x: f64) -> f64 {
        x.rem_euclid(std::f64::consts::TAU)
    }

    /// The neighbouring `f64` one ULP away from finite `x`, in the direction of
    /// `up`. Crossing zero is handled explicitly because the bit pattern is
    /// sign-magnitude, so `+1` on the bits of `-0.0` walks away from zero.
    fn ulp_neighbour(x: f64, up: bool) -> f64 {
        if x == 0.0 {
            return if up {
                f64::from_bits(1)
            } else {
                -f64::from_bits(1)
            };
        }
        let bits = x.to_bits();
        f64::from_bits(if up == (x > 0.0) {
            bits.wrapping_add(1)
        } else {
            bits.wrapping_sub(1)
        })
    }

    fn wrap_to_tau_rng_word(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A draw in `[0, 1]` built from a `u32` so the conversion is a lossless
    /// `f64::from` rather than a lint-suppressed `as`.
    fn wrap_to_tau_unit(state: &mut u64) -> f64 {
        let high = u32::try_from(wrap_to_tau_rng_word(state) >> 32).unwrap_or(0);
        f64::from(high) / f64::from(u32::MAX)
    }

    /// Named boundaries, an exhaustive ULP walk either side of every point where
    /// `wrap_to_tau`'s guards change answer, and deterministic sweeps over the
    /// production range, the guarded range, well past it, and raw bit patterns.
    fn wrap_to_tau_corpus() -> Vec<f64> {
        const TAU: f64 = std::f64::consts::TAU;
        const PI: f64 = std::f64::consts::PI;
        const SWEEP: u32 = 60_000;

        let mut corpus = vec![
            0.0,
            -0.0,
            TAU,
            -TAU,
            PI,
            -PI,
            0.5 * TAU,
            -0.5 * TAU,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            f64::MAX,
            f64::MIN,
            1.0e300,
            -1.0e300,
            1.0e16,
            -1.0e16,
        ];

        for anchor in [0.0_f64, -0.0, TAU, -TAU, PI, -PI] {
            for up in [true, false] {
                let mut walk = anchor;
                for _ in 0..64 {
                    walk = ulp_neighbour(walk, up);
                    corpus.push(walk);
                }
            }
        }

        let mut state = 0x5EED_1234_ABCD_9876_u64;
        for span in [PI, TAU, 8.0 * TAU, 1.0e12] {
            for _ in 0..SWEEP {
                corpus.push(span * (2.0 * wrap_to_tau_unit(&mut state) - 1.0));
            }
        }
        for _ in 0..SWEEP {
            corpus.push(f64::from_bits(wrap_to_tau_rng_word(&mut state)));
        }

        corpus
    }

    /// The acceptance for the `fmod` removal: `wrap_to_tau` is not an
    /// approximation of `rem_euclid(TAU)`, it is the same `f64` on every input.
    #[test]
    fn wrap_to_tau_is_bit_identical_to_rem_euclid() {
        for x in wrap_to_tau_corpus() {
            let fast = wrap_to_tau(x);
            let reference = wrap_to_tau_reference(x);
            assert_eq!(
                fast.to_bits(),
                reference.to_bits(),
                "wrap_to_tau({x:?}) gave {fast:?} where rem_euclid gives {reference:?}"
            );
        }
    }

    /// A sweep that never leaves one branch would pass whatever the other two
    /// did, so the corpus is asserted to reach all three — and to carry the
    /// three inputs whose handling is delicate rather than generic.
    #[test]
    fn the_wrap_to_tau_corpus_reaches_every_branch() {
        const TAU: f64 = std::f64::consts::TAU;
        let corpus = wrap_to_tau_corpus();

        let identity = corpus.iter().filter(|x| **x >= 0.0 && **x < TAU).count();
        let shifted = corpus.iter().filter(|x| **x < 0.0 && **x > -TAU).count();
        let fallback = corpus
            .len()
            .saturating_sub(identity)
            .saturating_sub(shifted);
        assert!(identity > 1_000, "identity branch reached {identity} times");
        assert!(shifted > 1_000, "add-TAU branch reached {shifted} times");
        assert!(fallback > 1_000, "fallback branch reached {fallback} times");

        assert!(
            corpus.iter().any(|x| x.to_bits() == (-0.0_f64).to_bits()),
            "corpus must carry negative zero"
        );
        assert!(
            corpus.iter().any(|x| x.to_bits() == (-TAU).to_bits()),
            "corpus must carry -TAU, the one point the guards exclude on purpose"
        );
        assert!(corpus.iter().any(|x| x.is_nan()), "corpus must carry NaN");
    }

    /// Why the lower guard is `> -TAU` and not `>= -TAU`.
    ///
    /// At exactly `-TAU` the naive rewrite and `rem_euclid` disagree in the sign
    /// of zero, and the sign survives into `h = sat_ra - sun_ra`. Relaxing the
    /// guard would make this the single input on which the change is not
    /// bit-identical, so the assertions below are what stops it being relaxed.
    #[test]
    fn the_lower_guard_excludes_negative_tau_because_the_zero_signs_differ() {
        const TAU: f64 = std::f64::consts::TAU;
        let naive = core::hint::black_box(-TAU) + TAU;
        assert_eq!(naive.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            wrap_to_tau_reference(-TAU).to_bits(),
            (-0.0_f64).to_bits(),
            "fmod(-TAU, TAU) is -0.0 and rem_euclid leaves it alone"
        );
        assert_ne!(0.0_f64.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(wrap_to_tau(-TAU).to_bits(), (-0.0_f64).to_bits());
    }

    struct LogQuadratureX4ApproxV1DynamicLower;

    impl Sealed for LogQuadratureX4ApproxV1DynamicLower {}

    impl QuadratureProfile for LogQuadratureX4ApproxV1DynamicLower {
        const LOWER_LOG_STEP: f64 = LogQuadratureX4ApproxV1::LOWER_LOG_STEP;
        const MIDDLE_LOG_STEP: f64 = LogQuadratureX4ApproxV1::MIDDLE_LOG_STEP;
        const UPPER_LOG_STEP: f64 = LogQuadratureX4ApproxV1::UPPER_LOG_STEP;
        const USE_FIXED_LOWER_PLAN: bool = false;
        const RETIRE_SPECIES_ROUND_TRIP: bool = true;
        const USE_FIXED_MIDDLE_PLAN: bool = false;
        const RETIRE_ZR_ROUND_TRIP: bool = LogQuadratureX4ApproxV1::RETIRE_ZR_ROUND_TRIP;
        const DLRSL_ZERO_ABOVE_KM: f64 = LogQuadratureX4ApproxV1::DLRSL_ZERO_ABOVE_KM;
        const FITTED_UPPER_SEGMENT: bool = LogQuadratureX4ApproxV1::FITTED_UPPER_SEGMENT;

        /// Unreachable behind `USE_FIXED_LOWER_PLAN = false`. It names the plan
        /// this profile would otherwise share rather than inventing a third.
        fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
            LogQuadratureX4ApproxV1::fixed_lower_state(tc, ain)
        }

        /// Unreachable behind `USE_FIXED_MIDDLE_PLAN = false`.
        fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
            LogQuadratureX4ApproxV1::fixed_middle_state(tc, ain)
        }
    }

    /// The production profile with the fixed lower plan switched off, i.e. the
    /// exact kernel exactly as it ran before the plan was extended to it.
    struct ExactOrekitDynamicLower;

    impl Sealed for ExactOrekitDynamicLower {}

    impl QuadratureProfile for ExactOrekitDynamicLower {
        const LOWER_LOG_STEP: f64 = ExactOrekitQuadrature::LOWER_LOG_STEP;
        const MIDDLE_LOG_STEP: f64 = ExactOrekitQuadrature::MIDDLE_LOG_STEP;
        const UPPER_LOG_STEP: f64 = ExactOrekitQuadrature::UPPER_LOG_STEP;
        const USE_FIXED_LOWER_PLAN: bool = false;
        const RETIRE_SPECIES_ROUND_TRIP: bool = false;
        const USE_FIXED_MIDDLE_PLAN: bool = false;
        const RETIRE_ZR_ROUND_TRIP: bool = ExactOrekitQuadrature::RETIRE_ZR_ROUND_TRIP;
        const DLRSL_ZERO_ABOVE_KM: f64 = ExactOrekitQuadrature::DLRSL_ZERO_ABOVE_KM;
        const FITTED_UPPER_SEGMENT: bool = ExactOrekitQuadrature::FITTED_UPPER_SEGMENT;

        /// Unreachable behind `USE_FIXED_LOWER_PLAN = false`.
        fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
            ExactOrekitQuadrature::fixed_lower_state(tc, ain)
        }

        /// Unreachable behind `USE_FIXED_MIDDLE_PLAN = false`.
        fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
            ExactOrekitQuadrature::fixed_middle_state(tc, ain)
        }
    }

    /// The EXACT profile with only `RETIRE_SPECIES_ROUND_TRIP` flipped on.
    ///
    /// It exists so the flag can be shown to be load-bearing. Without it,
    /// `ExactOrekitQuadrature::RETIRE_SPECIES_ROUND_TRIP == false` is an
    /// assertion about a literal and proves nothing about what the exact profile
    /// computes. With it, `retiring_the_round_trip_on_the_exact_profile_would_
    /// break_orekit_bits` shows that flipping the flag really does move the
    /// exact profile's densities — which is why the sealed Orekit fixture is
    /// green only with it off.
    struct ExactOrekitRetiringRoundTrip;

    impl Sealed for ExactOrekitRetiringRoundTrip {}

    impl QuadratureProfile for ExactOrekitRetiringRoundTrip {
        const LOWER_LOG_STEP: f64 = ExactOrekitQuadrature::LOWER_LOG_STEP;
        const MIDDLE_LOG_STEP: f64 = ExactOrekitQuadrature::MIDDLE_LOG_STEP;
        const UPPER_LOG_STEP: f64 = ExactOrekitQuadrature::UPPER_LOG_STEP;
        const USE_FIXED_LOWER_PLAN: bool = true;
        const RETIRE_SPECIES_ROUND_TRIP: bool = true;
        const USE_FIXED_MIDDLE_PLAN: bool = true;
        const RETIRE_ZR_ROUND_TRIP: bool = ExactOrekitQuadrature::RETIRE_ZR_ROUND_TRIP;
        const DLRSL_ZERO_ABOVE_KM: f64 = ExactOrekitQuadrature::DLRSL_ZERO_ABOVE_KM;
        const FITTED_UPPER_SEGMENT: bool = ExactOrekitQuadrature::FITTED_UPPER_SEGMENT;

        fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
            ExactOrekitQuadrature::fixed_lower_state(tc, ain)
        }

        fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
            ExactOrekitQuadrature::fixed_middle_state(tc, ain)
        }
    }

    /// The two production profiles with ONLY the middle plan switched off, i.e.
    /// each kernel exactly as it ran at the commit before this plan landed.
    ///
    /// These are the bit oracles for `fixed_middle_state`. They are not the
    /// `DynamicLower` profiles above: those also walk the 90--105 km segment,
    /// so a difference under them could not be attributed to the middle plan.
    struct ExactOrekitDynamicMiddle;

    impl Sealed for ExactOrekitDynamicMiddle {}

    impl QuadratureProfile for ExactOrekitDynamicMiddle {
        const LOWER_LOG_STEP: f64 = ExactOrekitQuadrature::LOWER_LOG_STEP;
        const MIDDLE_LOG_STEP: f64 = ExactOrekitQuadrature::MIDDLE_LOG_STEP;
        const UPPER_LOG_STEP: f64 = ExactOrekitQuadrature::UPPER_LOG_STEP;
        const USE_FIXED_LOWER_PLAN: bool = true;
        const RETIRE_SPECIES_ROUND_TRIP: bool = false;
        const USE_FIXED_MIDDLE_PLAN: bool = false;
        const RETIRE_ZR_ROUND_TRIP: bool = ExactOrekitQuadrature::RETIRE_ZR_ROUND_TRIP;
        const DLRSL_ZERO_ABOVE_KM: f64 = ExactOrekitQuadrature::DLRSL_ZERO_ABOVE_KM;
        const FITTED_UPPER_SEGMENT: bool = ExactOrekitQuadrature::FITTED_UPPER_SEGMENT;

        fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
            ExactOrekitQuadrature::fixed_lower_state(tc, ain)
        }

        /// Unreachable behind `USE_FIXED_MIDDLE_PLAN = false`.
        fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
            ExactOrekitQuadrature::fixed_middle_state(tc, ain)
        }
    }

    struct LogQuadratureX4ApproxV1DynamicMiddle;

    impl Sealed for LogQuadratureX4ApproxV1DynamicMiddle {}

    impl QuadratureProfile for LogQuadratureX4ApproxV1DynamicMiddle {
        const LOWER_LOG_STEP: f64 = LogQuadratureX4ApproxV1::LOWER_LOG_STEP;
        const MIDDLE_LOG_STEP: f64 = LogQuadratureX4ApproxV1::MIDDLE_LOG_STEP;
        const UPPER_LOG_STEP: f64 = LogQuadratureX4ApproxV1::UPPER_LOG_STEP;
        const USE_FIXED_LOWER_PLAN: bool = true;
        const RETIRE_SPECIES_ROUND_TRIP: bool = true;
        const USE_FIXED_MIDDLE_PLAN: bool = false;
        const RETIRE_ZR_ROUND_TRIP: bool = LogQuadratureX4ApproxV1::RETIRE_ZR_ROUND_TRIP;
        const DLRSL_ZERO_ABOVE_KM: f64 = LogQuadratureX4ApproxV1::DLRSL_ZERO_ABOVE_KM;
        const FITTED_UPPER_SEGMENT: bool = LogQuadratureX4ApproxV1::FITTED_UPPER_SEGMENT;

        fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
            LogQuadratureX4ApproxV1::fixed_lower_state(tc, ain)
        }

        /// Unreachable behind `USE_FIXED_MIDDLE_PLAN = false`.
        fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
            LogQuadratureX4ApproxV1::fixed_middle_state(tc, ain)
        }
    }

    fn orekit_local_input(altitude_km: f64) -> Jb2008Input {
        Jb2008Input {
            mjd_utc: 52_951.003_805_740_744,
            sun_declination_rad: -0.285_987_757_544_287,
            // The sealed Orekit pair, differenced: sat_ra 1.282_118_868_515_03
            // minus sun_ra 3.046_653_643_566_772. Kept as the subtraction so the
            // provenance of both halves stays legible; one rounding, exactly as
            // the kernel used to perform it.
            hour_angle_rad: 1.282_118_868_515_03 - 3.046_653_643_566_772,
            sat_geocentric_lat_rad: -1.487_718_654_399_9,
            sat_altitude_m: altitude_km * 1000.0,
            f10: 91.00,
            f10b: 137.10,
            s10: 108.80,
            s10b: 123.80,
            m10: 116.70,
            m10b: 128.50,
            y10: 168.00,
            y10b: 138.60,
            dst_temperature_correction_k: 43.0,
        }
    }

    #[test]
    fn quadrature_step_count_rejects_invalid_log_ratios() {
        for ratio in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -f64::MIN_POSITIVE,
        ] {
            assert_eq!(jb_step_count(ratio), Err(Jb2008Error::NumericalDomain));
        }
    }

    /// Frozen copy of the pre-plan 90--105 km loop at `7a3bd307`.
    ///
    /// This deliberately shares no prospective plan-building or plan-consuming
    /// helper. It is the independent bit oracle for the fixed-geometry path,
    /// and it keeps the FOUR SCALAR DIVIDES the loop was written with, so it is
    /// also the oracle for the vector-form lane quotient that replaced them.
    /// Only the log step is a parameter, so one oracle serves both plans.
    fn dynamic_lower_state_bits(tc: [f64; 4], lower_log_step: f64) -> [u64; 6] {
        let z1 = 90.0_f64;
        let z2 = 105.0_f64;
        let log_ratio = (z2 / z1).ln();
        let n = jb_step_count(log_ratio / lower_log_step).unwrap_or_default();
        let zr = (log_ratio / f64::from(n)).exp();
        let mb1 = jb_mbar(z1);
        let tloc1 = jb_local_temp(z1, tc);
        let broadcast = TemperatureBroadcast::new(tc);
        let mut zend = z1;
        let mut sub2 = 0.0;
        let mut ain = mb1 * jb_gravity(z1) / tloc1;
        let mut mb2 = 0.0;
        let mut tloc2 = 0.0;
        let mut z = 0.0;
        let mut gravl = 0.0;
        for _ in 0..n {
            z = zend;
            zend = zr * z;
            let dz = 0.25 * (zend - z);
            let [first_weight, ..] = JB_WT;
            let mut weighted_integral = first_weight * ain;
            let zs = boole_abscissae(&mut z, dz);
            let abscissa_vector = wide::f64x4::from(zs);
            let molecular_masses = jb_mbar_x4(abscissa_vector).to_array();
            let local_temperatures =
                jb_local_temp_below_break_x4(abscissa_vector, broadcast).to_array();
            let gravity_values = jb_gravity_x4(abscissa_vector).to_array();
            for (((weight, molecular_mass), local_temperature), gravity) in JB_WT
                .iter()
                .skip(1)
                .zip(molecular_masses.iter())
                .zip(local_temperatures.iter())
                .zip(gravity_values.iter())
            {
                mb2 = *molecular_mass;
                tloc2 = *local_temperature;
                gravl = *gravity;
                ain = mb2 * gravl / tloc2;
                weighted_integral += weight * ain;
            }
            sub2 += dz * weighted_integral;
        }
        [
            sub2.to_bits(),
            z.to_bits(),
            zend.to_bits(),
            mb2.to_bits(),
            tloc2.to_bits(),
            gravl.to_bits(),
        ]
    }

    fn planned_lower_state_bits<const N: usize>(
        plan: &FixedLowerPlan<N>,
        tc: [f64; 4],
    ) -> [u64; 6] {
        let z1 = 90.0;
        let initial_ain = jb_mbar(z1) * jb_gravity(z1) / jb_local_temp(z1, tc);
        let state = fixed_lower_state(plan, TemperatureBroadcast::new(tc), initial_ain);
        [
            state.sub2.to_bits(),
            state.z.to_bits(),
            state.zend.to_bits(),
            state.mb2.to_bits(),
            state.tloc2.to_bits(),
            state.gravl.to_bits(),
        ]
    }

    /// The whole-step-below-125 arm of `jb_local_temp_step_x4` must be BOTH
    /// bit-identical to the blended form and actually reached in production.
    ///
    /// Reachability is the half that is easy to lose: a one-sided guard that
    /// never fires is a dead branch that still reads like an optimisation. The
    /// middle quadrature segment is log-spaced from 105 km (`zend = zr * z`,
    /// `zr = exp(ln(500/105)/16)`), so its FIRST step's four Boole abscissae all
    /// sit below 125 km and its second step straddles. Both are pinned here, so
    /// a change to the segment start, the log step, or the step count that moved
    /// the arm out of the production path would fail rather than silently turn
    /// it into decoration.
    #[test]
    fn whole_step_below_break_arm_is_bit_identical_and_reached() {
        use wide::f64x4;

        // Reconstruct the real middle-segment geometry rather than hardcoding
        // abscissae, so this tracks the constants instead of a transcription.
        let al = (500.0f64 / 105.0).ln();
        let n = jb_step_count(al / <LogQuadratureX4ApproxV1 as QuadratureProfile>::MIDDLE_LOG_STEP)
            .expect("middle segment step count");
        let zr = (al / f64::from(n)).exp();

        let mut z;
        let mut zend = 105.0f64;
        let mut saw_below = 0usize;
        let mut saw_straddle = 0usize;

        for step in 0..n {
            z = zend;
            zend = zr * z;
            let dz = 0.25 * (zend - z);
            let zs = boole_abscissae(&mut z, dz);

            // tc values in the production range; the identity must not depend
            // on which ones, so exercise several.
            for tc in [
                [183.0, 7.303_974_2e-4, 100.0, 0.02],
                [444.380_7, 0.054_285_714, 250.0, 0.008],
                [1_000.0, 1.0e-3, 50.0, 0.05],
            ] {
                let broadcast = TemperatureBroadcast::new(tc);
                let blended = jb_local_temp_x4(f64x4::from(zs), broadcast).to_array();
                let stepped = jb_local_temp_step_x4(zs, f64x4::from(zs), broadcast).to_array();
                for (lane, (b, s)) in blended.iter().zip(stepped.iter()).enumerate() {
                    assert_eq!(
                        b.to_bits(),
                        s.to_bits(),
                        "step={step} lane={lane} zs={zs:?} tc={tc:?}"
                    );
                }
            }

            if zs[3] <= 125.0 {
                saw_below += 1;
            } else if zs[0] <= 125.0 {
                saw_straddle += 1;
            }
        }

        assert_eq!(
            saw_below, 1,
            "the below-break arm must be reached exactly once per evaluation \
             (middle segment step 1); if this is 0 the arm is dead code"
        );
        assert_eq!(
            saw_straddle, 1,
            "exactly one middle-segment step should straddle 125 km and stay blended"
        );
    }

    /// The plan's step count is written out because it sizes an array, so the
    /// link back to `LOWER_LOG_STEP` is a claim and not a derivation. This is
    /// where that claim is checked. If a log step moves, the plan silently
    /// integrates the wrong number of steps and the bit corpus below turns red;
    /// this fires first and says why.
    #[test]
    fn fixed_lower_plan_step_counts_track_the_log_steps() {
        let log_ratio = (105.0_f64 / 90.0).ln();
        assert_eq!(
            jb_step_count(log_ratio / ExactOrekitQuadrature::LOWER_LOG_STEP),
            u32::try_from(EXACT_FIXED_LOWER_STEPS).map_err(|_| Jb2008Error::NumericalDomain)
        );
        assert_eq!(
            jb_step_count(log_ratio / LogQuadratureX4ApproxV1::LOWER_LOG_STEP),
            u32::try_from(LOGQUAD_X4_FIXED_LOWER_STEPS).map_err(|_| Jb2008Error::NumericalDomain)
        );
    }

    /// The middle plan's step count is written out because it sizes an array,
    /// so the link back to `MIDDLE_LOG_STEP` is a claim and not a derivation.
    /// This is where that claim is checked, against the same `500.0` literal
    /// `altitude_km.min(500.0)` returns on the plan's own precondition and the
    /// same lower-plan exit abscissa `jb_density` hands the segment.
    #[test]
    fn fixed_middle_plan_step_counts_track_the_log_steps() {
        let exact_ratio = (500.0_f64 / exact_fixed_lower_plan().z).ln();
        assert_eq!(
            jb_step_count(exact_ratio / ExactOrekitQuadrature::MIDDLE_LOG_STEP),
            u32::try_from(EXACT_FIXED_MIDDLE_STEPS).map_err(|_| Jb2008Error::NumericalDomain)
        );
        let approx_ratio = (500.0_f64 / logquad_x4_fixed_lower_plan().z).ln();
        assert_eq!(
            jb_step_count(approx_ratio / LogQuadratureX4ApproxV1::MIDDLE_LOG_STEP),
            u32::try_from(LOGQUAD_X4_FIXED_MIDDLE_STEPS).map_err(|_| Jb2008Error::NumericalDomain)
        );
    }

    /// A spread of full kernel inputs, chosen so the temperature profile moves
    /// over the range a real arc produces rather than around one point.
    fn middle_plan_sweep_inputs() -> Vec<Jb2008Input> {
        let mut out = Vec::new();
        for (altitude_km, index) in [
            // Below the plan's 500 km precondition: these must take the walked
            // path on BOTH profiles, so they check the fallthrough rather than
            // the plan.
            106.0,
            150.0,
            260.0,
            499.999_999,
            500.0,
            // The production band, censused at 626.2--985.7 km on the sealed
            // V3 arc, plus the reach of the 2500 km extrapolation ceiling.
            500.000_001,
            550.0,
            626.226_149,
            700.0,
            800.0,
            985.663_551,
            1200.0,
            1800.0,
            2500.0,
        ]
        .into_iter()
        .zip((0u16..).map(f64::from))
        {
            for (offset, scale) in [(0.0, 1.0), (0.7, 0.55), (-1.1, 1.9), (2.3, 0.8)] {
                let base = orekit_local_input(altitude_km);
                out.push(Jb2008Input {
                    mjd_utc: base.mjd_utc + 37.0 * offset,
                    sun_declination_rad: 0.409 * (offset + 0.3 * index).sin(),
                    sat_geocentric_lat_rad: 1.4 * (0.9 * offset - 0.2 * index).cos(),
                    // Same swept satellite right ascension as before, against
                    // the base fixture's Sun, so the hour-angle spread this
                    // cloud covers is unchanged.
                    hour_angle_rad: 0.31f64.mul_add(index, 1.7 * offset) - 3.046_653_643_566_772,
                    f10: (base.f10 * scale).max(1.0),
                    f10b: (base.f10b * scale).max(1.0),
                    s10: (base.s10 * scale).max(1.0),
                    s10b: (base.s10b * scale).max(1.0),
                    m10: (base.m10 * scale).max(1.0),
                    m10b: (base.m10b * scale).max(1.0),
                    y10: (base.y10 * scale).max(1.0),
                    y10b: (base.y10b * scale).max(1.0),
                    dst_temperature_correction_k: 43.0 * offset,
                    ..base
                });
            }
        }
        out
    }

    /// The fixed 105--500 km plan must reproduce the walked loop to the BIT.
    ///
    /// Two independent levels, because they fail for different reasons:
    ///
    /// * `middle_state_bits` compares `fixed_middle_state` against
    ///   `dynamic_middle_state` directly, on the same entry state, over a
    ///   million temperature profiles. That is the sharp test — every stored
    ///   geometry field feeds the comparison and nothing downstream can mask a
    ///   difference.
    /// * the end-to-end sweep below runs whole densities through the two
    ///   profiles. It is the blunt test, and it exists because the sharp one
    ///   would still pass if `jb_density` stopped calling the plan.
    fn middle_state_bits<const N: usize>(
        plan: &FixedMiddlePlan<N>,
        tc: [f64; 4],
        ain: f64,
        lower_z: f64,
        lower_zend: f64,
        middle_log_step: f64,
    ) -> ([u64; 5], [u64; 5]) {
        fn key(state: &MiddleState) -> [u64; 5] {
            [
                state.sub2.to_bits(),
                state.ain.to_bits(),
                state.tloc3.to_bits(),
                state.z.to_bits(),
                state.zend.to_bits(),
            ]
        }
        let broadcast = TemperatureBroadcast::new(tc);
        let planned = fixed_middle_state(plan, broadcast, ain);
        // 500 km exactly: the plan's own precondition, and the altitude at
        // which `altitude_km.min(500.0)` returns the literal the plan folded.
        let walked =
            walk_middle_for_oracle(500.0, broadcast, ain, lower_z, lower_zend, middle_log_step);
        (key(&planned), key(&walked))
    }

    /// Frozen copy of the pre-plan 105--500 km loop, sharing no plan helper, so
    /// it is an independent oracle rather than a paraphrase of the thing it
    /// tests. Only the log step is a parameter, so one oracle serves both plans.
    fn walk_middle_for_oracle(
        altitude_km: f64,
        tc: TemperatureBroadcast,
        mut ain: f64,
        mut z: f64,
        mut zend: f64,
        middle_log_step: f64,
    ) -> MiddleState {
        let al = (altitude_km.min(500.0) / z).ln();
        let n = jb_step_count(al / middle_log_step).unwrap_or_default();
        let zr = (al / f64::from(n)).exp();
        let mut sub2 = 0.0;
        let mut temperature = wide::f64x4::ZERO;
        let [first_weight, ..] = JB_WT;
        for _ in 0..n {
            z = zend;
            zend = zr * z;
            let dz = 0.25 * (zend - z);
            let mut weighted_integral = first_weight * ain;
            let zs = boole_abscissae(&mut z, dz);
            let abscissa_vector = wide::f64x4::from(zs);
            temperature = jb_local_temp_step_x4(zs, abscissa_vector, tc);
            let lane_integrands = (jb_gravity_x4(abscissa_vector) / temperature).to_array();
            for (weight, lane_integrand) in JB_WT.iter().skip(1).zip(lane_integrands.iter()) {
                ain = *lane_integrand;
                weighted_integral += weight * ain;
            }
            sub2 += dz * weighted_integral;
        }
        let [_, _, _, tloc3] = temperature.to_array();
        MiddleState {
            sub2,
            ain,
            tloc3,
            z,
            zend,
        }
    }

    fn assert_middle_plan_matches_dynamic_loop_bits<const N: usize>(
        plan: &FixedMiddlePlan<N>,
        lower: (f64, f64),
        middle_log_step: f64,
    ) {
        const CASES: usize = 1 << 20;
        let (lower_z, lower_zend) = lower;
        let edges = [
            ([183.0, f64::MIN_POSITIVE, 1.0, 0.0], 1.0e-3),
            ([183.0, -f64::MIN_POSITIVE, 1.0e-30, 1.0e30], 0.0),
            ([444.3807, 0.054_285_714, 1.0e12, 1.0e-12], -1.0e-3),
            ([2_000.0, 200.0, 1.0e-300, 1.0e300], f64::MIN_POSITIVE),
        ];
        for (index, (tc, ain)) in edges.into_iter().enumerate() {
            let (planned, walked) =
                middle_state_bits(plan, tc, ain, lower_z, lower_zend, middle_log_step);
            assert_eq!(
                planned, walked,
                "steps={N} edge={index} tc={tc:?} ain={ain:e}"
            );
        }
        for index in 0..CASES {
            let index_u32 = u32::try_from(index).unwrap_or_default();
            // `tc[0]` and `tc[2]` span the transition temperature and the
            // arctangent amplitude a real profile produces; `tc[3]` is the
            // argument scale, which is what the plan's stored factors multiply.
            let tc0 = 183.0 + f64::from(index_u32 & 0x0fff) * 0.25;
            let tc1 = 1.0e-6
                + f64::from((index_u32.wrapping_mul(65_537) & 0xffff).saturating_add(1)) / 256.0;
            let tc2 = 1.0 + f64::from((index_u32.wrapping_mul(2_654_435_761) >> 8) & 0x000f_ffff);
            let tc3 = 1.0e-4
                * f64::from((index_u32.wrapping_mul(40_503) & 0x0003_ffff).saturating_add(1));
            let tc = [tc0, tc1, tc2, tc3];
            let ain = 1.0e-3 * f64::from((index_u32 & 0x3ff).saturating_add(1));
            let (planned, walked) =
                middle_state_bits(plan, tc, ain, lower_z, lower_zend, middle_log_step);
            assert_eq!(
                planned, walked,
                "steps={N} case={index} tc={tc:?} ain={ain:e}"
            );
        }
    }

    #[test]
    fn fixed_middle_plan_matches_dynamic_loop_bits() {
        let lower = logquad_x4_fixed_lower_plan();
        assert_middle_plan_matches_dynamic_loop_bits(
            logquad_x4_fixed_middle_plan(),
            (lower.z, lower.zend),
            LogQuadratureX4ApproxV1::MIDDLE_LOG_STEP,
        );
    }

    /// The same corpus against the exact profile's 63-step plan.
    #[test]
    fn exact_fixed_middle_plan_matches_dynamic_loop_bits() {
        let lower = exact_fixed_lower_plan();
        assert_middle_plan_matches_dynamic_loop_bits(
            exact_fixed_middle_plan(),
            (lower.z, lower.zend),
            ExactOrekitQuadrature::MIDDLE_LOG_STEP,
        );
    }

    /// End-to-end: whole densities through the planned and walked profiles.
    #[test]
    fn fixed_middle_plan_preserves_density_bits() {
        let mut planned_evaluations = 0usize;
        let mut inputs = logquad_inputs();
        inputs.extend(middle_plan_sweep_inputs());
        for input in inputs {
            let exact = jb2008_density_with_profile::<ExactOrekitQuadrature>(input);
            let exact_walked = jb2008_density_with_profile::<ExactOrekitDynamicMiddle>(input);
            assert_eq!(
                exact.map(f64::to_bits),
                exact_walked.map(f64::to_bits),
                "exact profile moved at {} m",
                input.sat_altitude_m
            );
            let approx = jb2008_density_with_profile::<LogQuadratureX4ApproxV1>(input);
            let approx_walked =
                jb2008_density_with_profile::<LogQuadratureX4ApproxV1DynamicMiddle>(input);
            assert_eq!(
                approx.map(f64::to_bits),
                approx_walked.map(f64::to_bits),
                "approximation profile moved at {} m",
                input.sat_altitude_m
            );
            if input.sat_altitude_m >= 500_000.0 && exact.is_ok() {
                planned_evaluations += 1;
            }
        }
        assert!(
            planned_evaluations >= 36,
            "the sweep must actually exercise the planned path, got {planned_evaluations}"
        );
    }

    /// Non-vacuity, both directions.
    ///
    /// The test above compares two profiles that agree; if `fixed_middle_state`
    /// ignored its plan and secretly walked the loop, it would still pass. This
    /// poisons a LOCAL copy of each plan with a value the geometry cannot
    /// produce and requires the density to move, then requires the unpoisoned
    /// copy to reproduce the shipped plan exactly.
    #[test]
    fn a_poisoned_middle_plan_moves_the_result() {
        fn probe<const N: usize>(plan: &FixedMiddlePlan<N>, tc: [f64; 4]) -> u64 {
            let broadcast = TemperatureBroadcast::new(tc);
            let state = fixed_middle_state(plan, broadcast, 1.234_5e-3);
            state.sub2.to_bits() ^ state.tloc3.to_bits() ^ state.ain.to_bits()
        }

        let tc = [444.380_7, 0.054_285_714, 250.0, 0.008];

        let poisoned: FixedMiddlePlan<LOGQUAD_X4_FIXED_MIDDLE_STEPS> = build_fixed_middle_plan(
            logquad_x4_fixed_lower_plan().z,
            logquad_x4_fixed_lower_plan().zend,
        );
        assert_eq!(
            probe(&poisoned, tc),
            probe(logquad_x4_fixed_middle_plan(), tc),
            "an unpoisoned rebuild must reproduce the shipped plan"
        );

        // Every field the hot loop reads, one at a time. A field the loop does
        // not actually consult would leave the result unmoved and fail here.
        let shipped = probe(&poisoned, tc);
        for field in 0..5 {
            let mut copy: FixedMiddlePlan<LOGQUAD_X4_FIXED_MIDDLE_STEPS> = build_fixed_middle_plan(
                logquad_x4_fixed_lower_plan().z,
                logquad_x4_fixed_lower_plan().zend,
            );
            let last = LOGQUAD_X4_FIXED_MIDDLE_STEPS.saturating_sub(1);
            let poison = wide::f64x4::new([1.5; 4]);
            let (first_step, last_step) = copy.steps.split_at_mut(last);
            let first_step = first_step.first_mut().expect("the plan has >1 step");
            let last_step = last_step.first_mut().expect("the plan has a last step");
            match field {
                0 => last_step.dz *= 1.5,
                1 => last_step.gravity *= poison,
                2 => last_step.break_offset *= poison,
                3 => last_step.argument_shape *= poison,
                _ => first_step.temperature_basis *= poison,
            }
            assert_ne!(
                probe(&copy, tc),
                shipped,
                "poisoning plan field {field} left the middle state unmoved, so the \
                 loop does not read it"
            );
        }

        // And the plan must be REACHED: the three predicates `jb_density` tests
        // before choosing it all hold across the censused production band.
        for altitude_km in [500.0_f64, 626.226_149, 800.0, 985.663_551, 2500.0] {
            assert!(
                <LogQuadratureX4ApproxV1 as QuadratureProfile>::USE_FIXED_MIDDLE_PLAN
                    && <LogQuadratureX4ApproxV1 as QuadratureProfile>::USE_FIXED_LOWER_PLAN
                    && altitude_km >= 105.0
                    && altitude_km >= 500.0,
                "the plan must be selected at {altitude_km} km"
            );
        }
    }

    fn assert_plan_matches_dynamic_loop_bits<const N: usize>(
        plan: &FixedLowerPlan<N>,
        lower_log_step: f64,
    ) {
        const CASES: usize = 1 << 20;
        let edges = [
            [183.0, f64::MIN_POSITIVE, 0.0, 0.0],
            [183.0, -f64::MIN_POSITIVE, 0.0, 0.0],
            [444.3807, 0.054_285_714, f64::MAX, f64::MIN],
            [2_000.0, 200.0, f64::MIN, f64::MAX],
        ];
        for (index, tc) in edges.into_iter().enumerate() {
            assert_eq!(
                planned_lower_state_bits(plan, tc),
                dynamic_lower_state_bits(tc, lower_log_step),
                "steps={N} edge={index} tc={tc:?}"
            );
        }
        for index in 0..CASES {
            let index_u32 = u32::try_from(index).unwrap_or_default();
            let tc0 = 183.0 + f64::from(index_u32 & 0x0fff) * 0.25;
            let tc1 = 1.0e-6
                + f64::from((index_u32.wrapping_mul(65_537) & 0xffff).saturating_add(1)) / 256.0;
            let tc = [
                tc0,
                tc1,
                f64::from_bits(u64::try_from(index).unwrap_or_default()),
                -f64::from(index_u32),
            ];
            assert_eq!(
                planned_lower_state_bits(plan, tc),
                dynamic_lower_state_bits(tc, lower_log_step),
                "steps={N} case={index} tc0={tc0:e} tc1={tc1:e}"
            );
        }
    }

    #[test]
    fn fixed_lower_plan_matches_dynamic_loop_bits() {
        assert_plan_matches_dynamic_loop_bits(
            logquad_x4_fixed_lower_plan(),
            LogQuadratureX4ApproxV1::LOWER_LOG_STEP,
        );
    }

    /// The same corpus against the exact profile's 16-step plan. The oracle it
    /// runs against still divides four times per step in scalar, so this covers
    /// both the fixed geometry and the vector-form lane quotient at once.
    #[test]
    fn exact_fixed_lower_plan_matches_dynamic_loop_bits() {
        assert_plan_matches_dynamic_loop_bits(
            exact_fixed_lower_plan(),
            ExactOrekitQuadrature::LOWER_LOG_STEP,
        );
    }

    /// `jb_mbar` was rewritten from a fold into a nested expression to make it
    /// a `const fn`, and its 90 km value is now frozen at compile time. Both
    /// steps have to be bit-exact, so the fold is kept here as the oracle.
    #[test]
    fn ninety_km_constants_match_the_folded_form() {
        fn folded_mbar(z_km: f64) -> f64 {
            let dz = z_km - 100.0;
            JB_CXAMB[..6]
                .iter()
                .rev()
                .fold(JB_CXAMB[6], |accumulator, coefficient| {
                    dz * accumulator + coefficient
                })
        }
        for tenths in 0..40_000_i32 {
            let z_km = f64::from(tenths) / 10.0;
            assert_eq!(
                jb_mbar(z_km).to_bits(),
                folded_mbar(z_km).to_bits(),
                "z_km={z_km}"
            );
        }
        assert_eq!(JB_MBAR_90_KM.to_bits(), folded_mbar(90.0).to_bits());
        assert_eq!(JB_MBAR_90_KM.to_bits(), jb_mbar(90.0).to_bits());
        assert_eq!(JB_GRAVITY_90_KM.to_bits(), jb_gravity(90.0).to_bits());
    }

    fn logquad_inputs() -> Vec<Jb2008Input> {
        (0_i32..257)
            .map(|index| {
                let mut input =
                    orekit_local_input(200.0 + f64::from(index.saturating_mul(37) % 1_300));
                input.mjd_utc += f64::from(index) * 0.125;
                input.sun_declination_rad += f64::from(index % 17) * 0.003;
                // The two right ascensions used to be swept at 0.019 and 0.031
                // per index; only their difference ever reached the kernel, so
                // sweeping the hour angle at 0.012 covers the same ground.
                input.hour_angle_rad = (input.hour_angle_rad + f64::from(index) * 0.012)
                    .rem_euclid(std::f64::consts::TAU);
                input.sat_geocentric_lat_rad += f64::from(index % 19) * 0.002;
                input.f10 += f64::from(index % 23);
                input.f10b += f64::from(index % 29);
                input
            })
            .collect()
    }

    /// `species_factor_domain` has to agree with the `ln` it replaced on every
    /// class of input, including the two that are easy to get wrong: `-0.0`,
    /// where `ln` gives `NEG_INFINITY` rather than `NaN`, and `NaN` itself,
    /// which `x < 0.0` answers `false` to.
    #[test]
    fn species_factor_domain_reproduces_the_retired_ln() {
        for value in [
            1.0e-300_f64,
            0.5,
            1.0,
            3.6e17,
            f64::MAX,
            f64::INFINITY,
            0.0,
            -0.0,
        ] {
            let guarded = species_factor_domain(value);
            assert!(
                !guarded.is_nan(),
                "{value:e} is in `ln`'s domain and must survive the guard"
            );
            assert_eq!(
                guarded.to_bits(),
                value.to_bits(),
                "{value:e} must pass through unchanged, sign of zero included"
            );
            // What the retired form would have produced, for the same reason.
            assert!(!value.ln().is_nan(), "{value:e}");
        }
        for value in [
            -1.0e-300_f64,
            -0.5,
            -1.0,
            -3.6e17,
            f64::MIN,
            f64::NEG_INFINITY,
        ] {
            assert!(
                species_factor_domain(value).is_nan(),
                "{value:e} is outside `ln`'s domain and must become NaN"
            );
            assert!(value.ln().is_nan(), "{value:e}");
        }
        assert!(species_factor_domain(f64::NAN).is_nan());
        // `-0.0` is the case a `<= 0.0` guard would have got wrong: it must
        // reach the sum as a zero term, exactly as `exp(ln(-0.0))` did.
        // Compared on bits rather than by `==`: `0.0 == -0.0` is true, which is
        // exactly the distinction under test here.
        assert_eq!(
            (species_factor_domain(-0.0) * (-3.0_f64).exp())
                .abs()
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!((-0.0_f64).ln().exp().to_bits(), 0.0_f64.to_bits());
    }

    /// How hard the rejecting arm of `species_factor_domain` was looked for.
    ///
    /// Every factor stays non-negative across this sweep, so the guard is not
    /// known to be reachable and this test does not claim it is unreachable —
    /// it records the search, and it fails loudly if some later change makes a
    /// factor go negative on ordinary inputs, which would mean the guard turned
    /// a working density into a `NumericalDomain` error.
    #[test]
    fn species_factors_stay_positive_across_a_wide_input_sweep() {
        let mut checked = 0_u32;
        for input in logquad_inputs() {
            for altitude_km in [90.0, 100.0, 104.0, 105.0, 150.0, 500.0, 900.0, 5_000.0] {
                let mut swept = input;
                swept.sat_altitude_m = altitude_km * 1000.0;
                // The APPROXIMATION profile: it is the one that retires the
                // round trip, so it is the one whose factors are number
                // densities rather than the folded `1.0`. Against the exact
                // profile this test would assert `1.0 >= 0.0` twelve thousand
                // times and catch nothing.
                if jb2008_density_logquad_x4_approx_v1(swept).is_err() {
                    continue;
                }
                let pairs = SPECIES_CAPTURE.with(std::cell::Cell::get);
                for (species, (factor, _)) in pairs.iter().enumerate() {
                    assert!(
                        *factor >= 0.0,
                        "species {species} factor {factor:e} went negative at \
                         altitude_km={altitude_km}; the domain guard would now turn this \
                         input into an error"
                    );
                    checked = checked.saturating_add(1);
                }
            }
        }
        assert!(checked > 10_000, "sweep collapsed to {checked} factors");
    }

    /// The flag is set the way it is because of an oracle in another crate, so
    /// this states the coupling where the flag lives.
    ///
    /// `lightyear_odeint_rs`'s `orekit_synthetic_mapping_matches_rust_primitive_
    /// kernel` replays a sealed Orekit 13.1.2 fixture through `jb2008_density`
    /// and requires BIT equality. Orekit computes the species logarithms, so
    /// retiring the round trip loses that by construction — measured, not
    /// feared: with `RETIRE_SPECIES_ROUND_TRIP` true on the exact profile, 11 of
    /// that fixture's cases go red.
    ///
    /// The second half is what stops this being an assertion about two literals.
    /// `ExactOrekitRetiringRoundTrip` is the exact profile with only that flag
    /// flipped, and its densities must actually differ — if they did not, the
    /// flag would be inert and the exact profile would be paying for a
    /// distinction that does not exist.
    #[test]
    fn retiring_the_round_trip_on_the_exact_profile_would_break_orekit_bits() {
        // Compile-time, not runtime: these are constants, so a runtime `assert!`
        // both reads as a check that could fail at run time and trips
        // `clippy::assertions_on_constants`. As `const` blocks a violation is a
        // build error, which is what pinning a constant should be.
        //
        // The exact profile must keep the round trip: the sealed Orekit fixture
        // asserts bit equality and Orekit computes the logarithms. The
        // approximation profile is declared non-exact and is the one production
        // flies, so it is where the five `ln` calls come off.
        const _: () = assert!(!ExactOrekitQuadrature::RETIRE_SPECIES_ROUND_TRIP);
        const _: () = assert!(LogQuadratureX4ApproxV1::RETIRE_SPECIES_ROUND_TRIP);

        let mut moved = 0_u32;
        let mut total = 0_u32;
        for input in logquad_inputs() {
            let kept = jb2008_density_with_profile::<ExactOrekitQuadrature>(input);
            let retired = jb2008_density_with_profile::<ExactOrekitRetiringRoundTrip>(input);
            let (Ok(kept), Ok(retired)) = (kept, retired) else {
                continue;
            };
            total = total.saturating_add(1);
            if kept.to_bits() != retired.to_bits() {
                moved = moved.saturating_add(1);
            }
            // Whatever it does to the bits, it stays far inside the tolerance
            // the Orekit VECTORS are asserted at. It is only the BIT oracle the
            // flag exists for, and that distinction is the whole point.
            let relative = (retired - kept).abs() / kept;
            assert!(
                relative <= 5.0e-13,
                "flipping the flag moved a density {relative:e}, which is outside \
                 the 5e-13 the Orekit vectors allow; that would be a real error, \
                 not a representation choice"
            );
        }
        assert!(total > 200, "corpus collapsed to {total} inputs");
        assert!(
            moved * 2 > total,
            "flipping RETIRE_SPECIES_ROUND_TRIP moved only {moved} of {total} \
             exact-profile densities; the flag is close to inert, so keeping it \
             off is not buying the Orekit bit parity it claims to buy"
        );
    }

    /// The species round trip was retired for speed, but it is also the less
    /// accurate association, and this pins the mechanism rather than the claim.
    ///
    /// `exp(ln(x) + y)` puts `ln`'s rounding into an `exp` ARGUMENT, where an
    /// absolute error becomes a relative error in the result. The corpus drives
    /// `|ln(x)|` up to 45.48, so half an ULP of `ln` there is about 3.6e-15
    /// absolute — roughly 16 ULP of relative error in the answer. `x * exp(y)`
    /// has no such term.
    ///
    /// This asserts the amplification directly: move `ln(x)` by ONE ULP, which
    /// is inside what a correctly-rounded `ln` is permitted to differ by across
    /// libms, and the retired form's result moves by many ULP. The landed form
    /// never evaluates `ln(x)`, so there is nothing to move.
    ///
    /// Adjudicated separately at 60 decimal digits over all 1,601 `(x, y)`
    /// pairs `dump_species_round_trip_pairs` captures from the corpus: the
    /// retired form's worst relative error is 1.528e-14 (68.80 ULP) and its
    /// mean is 1.458e-15; the landed form's worst is 1.782e-16 (0.80 ULP) and
    /// its mean is 4.980e-17. The landed form is nearer the true value on 1,298
    /// pairs, the retired form on 6, tied on 297 — a 29.3x mean improvement.
    #[test]
    fn retired_species_round_trip_amplified_ln_rounding() {
        // Spans the captured corpus range of |ln(x)|, which reaches 45.48.
        let mut worst_retired_ulp = 0.0_f64;
        for log_magnitude in [5.0_f64, 15.0, 25.0, 35.0, 45.0] {
            let x = log_magnitude.exp();
            for y in [-40.0_f64, -31.0, -20.0, -5.0, 0.0] {
                let retired = (x.ln() + y).exp();
                let landed = x * y.exp();
                // One ULP of `ln(x)`, the difference two conforming libms are
                // allowed to have, and all the retired form needs to move.
                let nudged = (f64::from_bits(x.ln().to_bits().wrapping_add(1)) + y).exp();
                let moved_ulp = (nudged - retired).abs() / (retired * f64::EPSILON);
                worst_retired_ulp = worst_retired_ulp.max(moved_ulp);
                // The two forms are still the same identity, to well inside the
                // 5e-13 the Orekit vectors are asserted at.
                let disagreement = (landed - retired).abs() / landed;
                assert!(
                    disagreement <= 1.0e-13,
                    "forms disagree beyond the identity: ln(x)={log_magnitude}, y={y}, \
                     relative={disagreement:e}"
                );
            }
        }
        assert!(
            worst_retired_ulp >= 8.0,
            "the retired form was supposed to amplify one ULP of `ln` into many; \
             worst observed was {worst_retired_ulp:.2} ULP"
        );
    }

    /// Dump every `(factor, log_offset)` pair the corpus drives `jb_density`
    /// through, so the round-trip accuracy claim can be adjudicated against an
    /// arbitrary-precision reference outside this process. Ignored by default;
    /// it is a probe, not a gate.
    #[test]
    #[ignore = "probe: prints the species pairs for external precision analysis"]
    fn dump_species_round_trip_pairs() {
        let mut inputs = logquad_inputs();
        inputs.extend(
            [
                90.0, 100.0, 105.0, 200.0, 500.0, 626.2, 800.0, 985.7, 1500.0, 35_000.0,
            ]
            .map(orekit_local_input)
            .iter()
            .copied(),
        );
        for (index, input) in inputs.into_iter().enumerate() {
            // The approximation profile, for the reason given in
            // `species_factors_stay_positive_across_a_wide_input_sweep`.
            if jb2008_density_logquad_x4_approx_v1(input).is_err() {
                continue;
            }
            let pairs = SPECIES_CAPTURE.with(std::cell::Cell::get);
            for (species, (factor, offset)) in pairs.iter().enumerate() {
                println!(
                    "PAIR {index} {species} {:#018x} {:#018x}",
                    factor.to_bits(),
                    offset.to_bits()
                );
            }
        }
    }

    #[test]
    fn fixed_lower_plan_preserves_logquad_density_bits() {
        for (index, input) in logquad_inputs().into_iter().enumerate() {
            let dynamic = jb2008_density_with_profile::<LogQuadratureX4ApproxV1DynamicLower>(input);
            let fixed = jb2008_density_logquad_x4_approx_v1(input);
            assert_eq!(
                fixed.map(f64::to_bits),
                dynamic.map(f64::to_bits),
                "input {index}"
            );
        }
    }

    /// The end-to-end form of the same claim for the EXACT profile, which is
    /// the one every propagation calls. `jb2008_density` now runs the 16-step
    /// fixed plan at and above 105 km; below it, and on the dynamic profile, it
    /// runs the loop. The whole density has to come out bit-identical either
    /// way, including the sub-105 km inputs that never touch the plan at all.
    #[test]
    fn fixed_lower_plan_preserves_exact_density_bits() {
        let mut inputs = logquad_inputs();
        inputs.extend(
            [90.0, 95.0, 100.0, 104.999_999, 105.0, 105.000_001, 106.0]
                .map(orekit_local_input)
                .iter()
                .copied(),
        );
        for (index, input) in inputs.into_iter().enumerate() {
            let dynamic = jb2008_density_with_profile::<ExactOrekitDynamicLower>(input);
            let fixed = jb2008_density(input);
            assert_eq!(
                fixed.map(f64::to_bits),
                dynamic.map(f64::to_bits),
                "input {index} altitude_km={}",
                input.sat_altitude_m / 1000.0
            );
        }
    }

    #[test]
    fn matches_exact_orekit_jar_91km_vector() {
        // Direct Orekit 13.1.2 JAR output from pinned oracle generator.
        let rho = jb2008_density(orekit_local_input(91.0)).unwrap();
        let expected = f64::from_bits(0x3ec7_7149_31e9_f622);
        assert!((rho - expected).abs() / expected <= 5.0e-13);
    }

    #[test]
    fn matches_orekit_piecewise_boundary_vectors() {
        // Direct Orekit 13.1.2 JAR outputs. Pinned generator/provenance:
        // ../oracle/OrekitJb2008Vectors.java and THIRD_PARTY_NOTICES.md.
        let cases = [
            (120.0, 1.838_696_519_476_145_5e-8),
            (200.0, 2.808_336_493_152_304_5e-10),
            (240.0, 8.695_755_489_180_2e-11),
            (300.0, 2.011_961_711_941_511e-11),
            (600.0, 9.221_278_835_986_616e-14),
            (800.0, 8.834_689_333_290_814e-15),
            (1000.0, 2.746_548_564_268_957_4e-15),
            (1500.0, 5.804_575_428_288_933e-16),
            (2300.0, 1.072_215_985_125_892_2e-16),
            (3000.0, 4.069_823_858_778_913e-17),
        ];
        for (altitude_km, expected) in cases {
            let rho = jb2008_density(orekit_local_input(altitude_km))
                .unwrap_or_else(|error| panic!("{altitude_km} km: {error:?}"));
            let relative_error = (rho - expected).abs() / expected;
            assert!(
                relative_error <= 5.0e-13,
                "{altitude_km} km relative error={relative_error:e}"
            );
        }
        let rho = jb2008_density(Jb2008Input {
            mjd_utc: 35_000.25,
            ..orekit_local_input(400.0)
        })
        .unwrap();
        let expected = 2.062_397_744_288_459_7e-12;
        assert!((rho - expected).abs() / expected <= 5.0e-13);
    }

    #[test]
    fn positive_five_halves_helper_tracks_libm_reference() {
        let values = [1.0e-12, 0.125, 1.0, 12.5, 125.0, 875.0, 2_875.0]
            .into_iter()
            .chain((1..=348_750).map(|tenths| f64::from(tenths) / 10.0));
        for value in values {
            let expected = value.powf(2.5);
            let actual = jb_positive_five_halves(value);
            let relative_error = (actual - expected).abs() / expected;
            assert!(
                relative_error <= 5.0e-15,
                "value={value}, relative_error={relative_error:e}"
            );
        }
    }

    #[test]
    fn matches_exact_orekit_jar_35000km_vector() {
        // Direct Orekit 13.1.2 JAR output from pinned oracle generator.
        let rho = jb2008_density(orekit_local_input(35_000.0)).unwrap();
        assert!(rho.is_finite() && rho > 0.0, "rho={rho:e}");
        let expected = f64::from_bits(0x3c52_7c8f_ee50_4f59);
        assert!((rho - expected).abs() / expected <= 5.0e-13);
    }

    /// The vendored `atan_x4` is a transliteration, so it needs a pin that a
    /// re-derivation cannot pass by being merely accurate. An earlier attempt
    /// rewrote the Estrin macros as Horner and diverged on 1,613 of 3,200,000
    /// lanes at 1 ULP — a difference this test sees and a tolerance would not.
    #[test]
    fn vendored_atan_x4_is_bit_identical_to_wide() {
        let mut lanes = 0u64;
        let mut divergent = 0u64;
        let mut check = |v: [f64; 4]| {
            let x = wide::f64x4::from(v);
            let ours = atan_x4(x).to_array();
            let theirs = x.atan().to_array();
            for (our_lane, their_lane) in ours.iter().zip(theirs.iter()) {
                lanes += 1;
                if our_lane.to_bits() != their_lane.to_bits() {
                    divergent += 1;
                }
            }
        };
        // Straddles both range-reduction breaks, 0.66 and 1 + sqrt(2), from
        // denormal-adjacent magnitudes out past the operands this kernel makes.
        for step in 0..200_000_i32 {
            let t = f64::from(step) / 20_000.0;
            let magnitudes = [
                t,
                1.0 / (t + 1.0e-3),
                t * t * t,
                1.0e6 * t,
                0.66 + (t - 5.0) * 1.0e-9,
                std::f64::consts::SQRT_2 + 1.0 + (t - 5.0) * 1.0e-9,
            ];
            for m in magnitudes {
                check([m, -m, m * 1.0e-8, -m * 1.0e8]);
            }
        }
        check([0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY]);
        check([0.66, -0.66, 2.414_213_562_373_095, -2.414_213_562_373_095]);
        assert_eq!(divergent, 0, "{divergent} of {lanes} lanes diverged");
        assert!(lanes >= 4_800_000, "only {lanes} lanes compared");
    }

    /// `atan_x4_above_t3po8` claims to be the general body with every mask
    /// resolved, not a re-approximation. That claim is a bit claim, so this is
    /// a bit test, over the whole half-line the guard admits.
    ///
    /// The second half is the non-vacuity proof. An equality test between two
    /// arctangent implementations is exactly the kind that passes for the wrong
    /// reason, so the test also demands that the specialisation is WRONG below
    /// the threshold. Without that, a specialisation that had quietly become a
    /// copy of the general body — costing the whole optimisation and changing
    /// nothing observable — would still be green.
    #[test]
    fn atan_x4_large_matches_the_general_body_bitwise() {
        use wide::f64x4;

        let threshold = std::f64::consts::SQRT_2 + 1.0;
        let mut compared = 0u64;
        let mut divergent = 0u64;

        // The kernel's own operands reach ~1e2; go four decades past that in
        // both directions from the threshold, plus the threshold-adjacent
        // binary64 neighbourhood where the guard decides.
        for step in 0..250_000_i32 {
            let t = f64::from(step) / 25_000.0;
            let magnitudes = [
                threshold * (1.0 + t),
                threshold + f64::from(step) * f64::EPSILON * 4.0,
                threshold * (1.0 + t * t * t),
                1.0e4 * (t + 1.0),
                1.0e6 * (t + 1.0),
                2.5 + t,
            ];
            let vector = f64x4::from([magnitudes[0], magnitudes[1], magnitudes[2], magnitudes[3]]);
            for candidate in [
                vector,
                f64x4::from([magnitudes[4], magnitudes[5], magnitudes[0], magnitudes[2]]),
            ] {
                // Only operands the guard actually admits are in scope.
                if !candidate.simd_gt(T3PO8_X4).all() {
                    continue;
                }
                let general = atan_x4(candidate).to_array();
                let specialised = atan_x4_above_t3po8(candidate).to_array();
                for (a, b) in general.iter().zip(specialised.iter()) {
                    compared += 1;
                    if a.to_bits() != b.to_bits() {
                        divergent += 1;
                    }
                }
            }
        }
        assert_eq!(
            divergent, 0,
            "{divergent} of {compared} lanes diverged above the threshold"
        );
        assert!(compared >= 1_000_000, "only {compared} lanes compared");

        // NON-VACUITY, PART 1. An equality test between two arctangents can
        // pass because the specialisation has quietly become a copy of the
        // general body — which would cost the whole optimisation and change
        // nothing observable. Below the threshold the two must therefore
        // disagree. They are not required to disagree on EVERY lane: the two
        // reductions coincide by luck on a few percent of operands (chiefly the
        // ones close enough to the threshold that both land on the same bits),
        // so the bar is a large majority and the measured rate is 92.4%.
        let mut below_divergent = 0u64;
        let mut below_compared = 0u64;
        for step in 1..20_000_i32 {
            let x = f64::from(step) / 20_000.0 * threshold;
            let candidate = f64x4::from([x, x * 0.5, x * 0.25, x * 0.125]);
            let general = atan_x4(candidate).to_array();
            let specialised = atan_x4_above_t3po8(candidate).to_array();
            for (a, b) in general.iter().zip(specialised.iter()) {
                below_compared += 1;
                if a.to_bits() != b.to_bits() {
                    below_divergent += 1;
                }
            }
        }
        assert!(
            below_divergent * 10 >= below_compared * 9,
            "the specialisation agreed with the general body on {} of {below_compared} \
             below-threshold lanes; it is supposed to be wrong on nearly all of them, so \
             this much agreement means it is no longer a specialisation",
            below_compared - below_divergent
        );

        // NON-VACUITY, PART 2: the `MORE_BITS` half of `BIG_OFFSET` is inert,
        // and that is a fact worth pinning rather than a defect.
        //
        // `MORE_BITS` is 6.123e-17 against a half ULP of 1.110e-16 for
        // `FRAC_PI_2`, so the addition rounds straight back. The same holds on
        // `atan_x4`'s other branch: `MORE_BITS_O2` is 3.062e-17 against a half
        // ULP of 5.551e-17 for `FRAC_PI_4`. So `fac` never changes a bit
        // anywhere in `atan_x4` — a property of `wide`'s vector form, which
        // adds `offset + fac` to a result of order 1, and not of Cephes, which
        // adds `MOREBITS` to the small reduced argument instead.
        //
        // Both halves are asserted because the claim is about `atan_x4` as a
        // whole, and because a reader checking only the branch this file
        // specialises would conclude the other one differs.
        assert_eq!(
            (std::f64::consts::FRAC_PI_2 + 6.123_233_995_736_766e-17).to_bits(),
            std::f64::consts::FRAC_PI_2.to_bits(),
            "MORE_BITS stopped being absorbed into FRAC_PI_2; BIG_OFFSET's derivation \
             now has a term in it that this branch actually feels"
        );
        assert_eq!(
            (std::f64::consts::FRAC_PI_4 + 6.123_233_995_736_766e-17 * 0.5).to_bits(),
            std::f64::consts::FRAC_PI_4.to_bits(),
            "MORE_BITS_O2 stopped being absorbed into FRAC_PI_4; atan_x4's mid branch \
             now feels a term this module's notes say it cannot"
        );

        // POISON. What carries the value on this branch is the SIGN of the
        // reduced argument: the identity is `atan(x) = pi/2 - atan(1/x)`, and
        // the shipped code spells that minus by making the numerator `-1`.
        // Writing `1.0 / value` instead is the one substitution that looks
        // right and is not, so plant it and require the kernel's own operands
        // to reject it on every lane.
        //
        // Two other rewrites were tried here first and were rejected as
        // poisons because they move NOTHING on this branch, which is itself
        // worth recording: for `|v| > 1 + sqrt(2)` the polynomial correction is
        // a perturbation of order `v^-3` on a result of order `pi/2`, so both
        // un-fusing the `mul_add` and dropping `MORE_BITS` are absorbed by the
        // final addition. Neither is evidence of anything, in either direction.
        let mut poison_caught = 0u64;
        let mut poison_lanes = 0u64;
        for step in 0..10_000_i32 {
            let x = threshold * (1.0 + f64::from(step) / 100.0);
            let candidate = f64x4::from([x, x * 2.0, x * 8.0, x * 32.0]);
            let honest = atan_x4_above_t3po8(candidate).to_array();

            let reduced = f64x4::ONE / candidate;
            let squared = reduced * reduced;
            let poisoned = ((atan_poly_p(squared) / atan_poly_q(squared))
                .mul_add(reduced * squared, reduced)
                + f64x4::FRAC_PI_2)
                .to_array();

            for (a, b) in honest.iter().zip(poisoned.iter()) {
                poison_lanes += 1;
                if a.to_bits() != b.to_bits() {
                    poison_caught += 1;
                }
            }
        }
        assert_eq!(
            poison_caught, poison_lanes,
            "flipping the reduced argument's sign moved only {poison_caught} of \
             {poison_lanes} lanes; a test that cannot see that cannot see anything"
        );
    }

    /// Sample the six altitude-only functions the L1 fit needs, for the
    /// generator in `tools/r57-upper-fit/fit_upper.py`.
    ///
    /// Ignored, because it is a generator input and not a check. It prints one
    /// row per altitude and nothing else, so its output is the fitter's stdin:
    ///
    /// ```sh
    /// cargo test --release -p jb_rs --lib dump_upper_segment_fit_samples \
    ///     -- --ignored --nocapture > /tmp/samples.txt
    /// python3 tools/r57-upper-fit/fit_upper.py < /tmp/samples.txt
    /// ```
    ///
    /// The values come from the kernel's own `boole_abscissae`, `jb_gravity`,
    /// `JB_WT` and middle plan rather than from a transcription of them, which
    /// is the whole reason this lives inside the module. `f` is spelled exactly
    /// as `jb_local_temp_above_break_x4` spells it, `dz * dz * dz.sqrt()` and
    /// all, because a paraphrase of the abscissa shape is precisely the r12
    /// corpus trap.
    #[test]
    #[ignore = "fit generator input, not a check; see tools/r57-upper-fit"]
    fn dump_upper_segment_fit_samples() {
        const SAMPLES: u32 = 96;
        let plan = logquad_x4_v2_fixed_middle_plan();
        println!("PLAN z={:.17e} zend={:.17e}", plan.z, plan.zend);
        // CHEBYSHEV-GAUSS nodes, not a uniform sweep. On these the Chebyshev
        // coefficients are an orthogonal projection computed by a cosine sum,
        // so the generator needs no linear algebra and cannot lose digits to an
        // ill-conditioned normal matrix — which matters here because `G0` has to
        // be fitted to 1e-9 relative. 96 nodes support any degree the fitter
        // asks for and leave a long coefficient tail to read the decay off.
        let mid = 0.5 * (UPPER_FIT_ALT_HI + UPPER_FIT_ALT_LO);
        let half = 0.5 * (UPPER_FIT_ALT_HI - UPPER_FIT_ALT_LO);
        println!("DOMAIN lo={UPPER_FIT_ALT_LO:.17e} hi={UPPER_FIT_ALT_HI:.17e} nodes={SAMPLES}");
        for index in 0..SAMPLES {
            let theta = std::f64::consts::PI * (f64::from(index) + 0.5) / f64::from(SAMPLES);
            let altitude_km = mid + half * theta.cos();
            let (g, f1) = upper_segment_fit_targets(plan, altitude_km);
            println!(
                "ROW {altitude_km:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                g[0], g[1], g[2], g[3], f1
            );
        }
    }

    /// The four `G_k` and `F1` at one altitude, walked exactly as the kernel
    /// walks the step. Shared by the generator above and by the accuracy gate,
    /// so the two cannot drift apart.
    fn upper_segment_fit_targets(
        plan: &FixedMiddlePlan<LOGQUAD_X4_V2_FIXED_MIDDLE_STEPS>,
        altitude_km: f64,
    ) -> ([f64; 4], f64) {
        let ratio = (altitude_km.max(500.0) / plan.z).max(1.0);
        let mut z = plan.zend;
        let zend = ratio * z;
        let dz = 0.25 * (zend - z);
        let zs = boole_abscissae(&mut z, dz);
        let mut g = [0.0_f64; 4];
        let mut last_inverse_f = 0.0;
        // The weights are zipped from `JB_WT.iter().skip(1)` exactly as the
        // walked loop consumes them, so the pairing cannot drift and no index
        // arithmetic stands between the two.
        for (weight, abscissa) in JB_WT.iter().skip(1).zip(zs.iter()) {
            let break_offset = abscissa - 125.0;
            let five_halves = break_offset * break_offset * break_offset.sqrt();
            let f = break_offset * 4.5e-6_f64.mul_add(five_halves, 1.0);
            let inverse_f = 1.0 / f;
            last_inverse_f = inverse_f;
            let weighted_gravity = weight * jb_gravity(*abscissa);
            let mut power = 1.0;
            for slot in &mut g {
                *slot += weighted_gravity * power;
                power *= inverse_f;
            }
        }
        (g, last_inverse_f)
    }

    /// The shipped `jb_dtc`'s previous spelling, kept verbatim as the oracle.
    ///
    /// Not a paraphrase and not a re-derivation: this is the exact body that
    /// stood before the bands were reversed, character for character apart from
    /// the name. Its only job is to be the thing the new order is proved equal
    /// to, which is what an equivalence test needs and what a fresh
    /// transcription of the model would not be.
    fn jb_dtc_upward_chain(
        f10: f64,
        solar_time_hour: f64,
        cos_geocentric_latitude: f64,
        altitude_km: f64,
    ) -> f64 {
        let st = solar_time_hour / 24.0;
        let cs = cos_geocentric_latitude;
        let fs = (f10 - 100.0) / 100.0;
        if (120.0..=200.0).contains(&altitude_km) {
            let dtc200 = jb_poly2_cdtc(fs, st, cs);
            let dtc200dz = jb_poly1_cdtc(fs, st, cs);
            let cc = 3.0 * dtc200 - dtc200dz;
            let dd = dtc200 - cc;
            let zp = (altitude_km - 120.0) / 80.0;
            zp * zp * (cc + dd * zp)
        } else if (200.0..=240.0).contains(&altitude_km) {
            jb_poly1_cdtc(fs, st, cs) * (altitude_km - 200.0) / 50.0 + jb_poly2_cdtc(fs, st, cs)
        } else if (240.0..=300.0).contains(&altitude_km) {
            let bb = jb_poly1_cdtc(fs, st, cs);
            let aa = 0.8 * bb + jb_poly2_cdtc(fs, st, cs);
            let p2bdt = jb_poly2_bdtc(st);
            let dtc300 = jb_poly1_bdtc(fs, st, cs, 3.0 * p2bdt);
            let dtc300dz = cs * p2bdt;
            let cc = 3.0 * dtc300 - dtc300dz - 3.0 * aa - 2.0 * bb;
            let dd = dtc300 - aa - bb - cc;
            let zp = (altitude_km - 240.0) / 60.0;
            aa + zp * (bb + zp * (cc + zp * dd))
        } else if (300.0..=600.0).contains(&altitude_km) {
            jb_poly1_bdtc(fs, st, cs, altitude_km * jb_poly2_bdtc(st) / 100.0)
        } else if (600.0..=800.0).contains(&altitude_km) {
            let poly2 = jb_poly2_bdtc(st);
            let aa = jb_poly1_bdtc(fs, st, cs, 6.0 * poly2);
            let bb = cs * poly2;
            let cc = -(3.0 * aa + 4.0 * bb) / 4.0;
            let dd = (aa + bb) / 4.0;
            let zp = (altitude_km - 600.0) / 100.0;
            aa + zp * (bb + zp * (cc + zp * dd))
        } else {
            0.0
        }
    }

    /// The reversed band chain must return the upward chain's BITS everywhere.
    ///
    /// The bands share their endpoints on purpose, so the reversal is only
    /// exact because its lower bounds are strict. That is an argument about
    /// five specific numbers, and this is the sweep that checks it at all five,
    /// at both ULP neighbours of each, across every band interior, and on the
    /// arms that return zero — including `NaN`, which reaches the zero arm
    /// through a different route in each spelling.
    #[test]
    fn jb_dtc_band_order_is_bit_identical_to_the_upward_chain() {
        let mut compared = 0u64;
        let mut per_band = [0u32; 6];

        let mut check = |altitude_km: f64, f10: f64, hour: f64, cos_lat: f64| {
            let want = jb_dtc_upward_chain(f10, hour, cos_lat, altitude_km);
            let got = jb_dtc(f10, hour, cos_lat, altitude_km);
            assert_eq!(
                want.to_bits(),
                got.to_bits(),
                "jb_dtc moved at altitude {altitude_km}, f10 {f10}, hour {hour}, \
                 cos_lat {cos_lat}: {want:e} -> {got:e}"
            );
            compared += 1;
            let band = if !(120.0..=800.0).contains(&altitude_km) {
                0
            } else if altitude_km > 600.0 {
                5
            } else if altitude_km > 300.0 {
                4
            } else if altitude_km > 240.0 {
                3
            } else if altitude_km > 200.0 {
                2
            } else {
                1
            };
            if let Some(slot) = per_band.get_mut(band) {
                *slot += 1;
            }
        };

        // The five shared endpoints and both ULP neighbours of each, which is
        // where a reversal that used inclusive lower bounds would break.
        let mut boundaries = Vec::new();
        for edge in [120.0_f64, 200.0, 240.0, 300.0, 600.0, 800.0] {
            boundaries.push(edge);
            boundaries.push(f64::from_bits(edge.to_bits() - 1));
            boundaries.push(f64::from_bits(edge.to_bits() + 1));
        }
        // Band interiors, the out-of-range arms, and the non-finite inputs.
        for step in 0..900 {
            boundaries.push(90.0 + f64::from(step));
        }
        boundaries.extend([
            0.0,
            -0.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            35_000.0,
        ]);

        for altitude_km in boundaries {
            for (f10, hour, cos_lat) in [
                (91.0, 12.0, 0.5),
                (60.0, 0.0, -1.0),
                (302.0, 23.9, 1.0),
                (100.0, 6.25, 0.0),
                (250.0, 18.0, -std::f64::consts::FRAC_1_SQRT_2),
            ] {
                check(altitude_km, f10, hour, cos_lat);
            }
        }

        assert!(compared >= 4_000, "only {compared} rows compared");
        // NON-VACUITY. Equality between two spellings is also what a corpus
        // that never leaves one band would report, and the endpoints are the
        // whole risk. Every band, including the zero arm, has to be reached.
        for (band, count) in per_band.iter().enumerate() {
            assert!(
                *count >= 20,
                "band {band} was reached {count} times; the sweep is not \
                 exercising the chain it is supposed to be pinning"
            );
        }
    }

    /// [`atan_x4_asymptotic`] must return the general body's bits on every
    /// argument the guard admits, and must NOT below it.
    ///
    /// The doc comment on that function argues the truncated series and Cephes'
    /// rational cannot round apart above `ATAN_ASYMPTOTIC_MIN`; the margin it
    /// derives is `1e-5` of an ULP, which is a margin and not a proof, so this
    /// is the evidence. The sweep runs from the threshold to twelve decades past
    /// it, because the flown arguments sit only ~1.6x above the threshold and
    /// the tail is where a series truncation would be safest — the interesting
    /// lanes are the near ones, and they are sampled densest.
    #[test]
    fn atan_x4_asymptotic_matches_the_general_body_bitwise() {
        use wide::f64x4;

        let mut compared = 0u64;
        let mut divergent = 0u64;
        let mut worst_ulp = 0i64;
        for step in 0..500_000_i32 {
            let t = f64::from(step) / 500_000.0;
            let magnitudes = [
                // Dense across [64, 128], the decade the kernel lives in.
                ATAN_ASYMPTOTIC_MIN * (1.0 + t),
                // The binary64 neighbourhood of the threshold itself.
                ATAN_ASYMPTOTIC_MIN + f64::from(step) * f64::EPSILON * 8.0,
                // Geometric out to 1e12, where the series is unarguably exact.
                ATAN_ASYMPTOTIC_MIN * (1.0 + t).powi(40),
                1.0e6 * (1.0 + t),
                1.0e12 * (1.0 + t),
                ATAN_ASYMPTOTIC_MIN * (1.0 + t * t * t * 15.0),
            ];
            for candidate in [
                f64x4::from([magnitudes[0], magnitudes[1], magnitudes[2], magnitudes[3]]),
                f64x4::from([magnitudes[4], magnitudes[5], magnitudes[0], magnitudes[2]]),
            ] {
                if !candidate.simd_gt(ATAN_ASYMPTOTIC_MIN_X4).all() {
                    continue;
                }
                let general = atan_x4(candidate).to_array();
                let asymptotic = atan_x4_asymptotic(candidate).to_array();
                for (a, b) in general.iter().zip(asymptotic.iter()) {
                    compared += 1;
                    if a.to_bits() != b.to_bits() {
                        divergent += 1;
                        let gap = i64::from_ne_bytes(a.to_bits().to_ne_bytes())
                            - i64::from_ne_bytes(b.to_bits().to_ne_bytes());
                        worst_ulp = worst_ulp.max(gap.abs());
                    }
                }
            }
        }
        // THE MEASURED RATE, not zero. 13 of 3,999,976 lanes move and every one
        // of them moves by exactly 1 ULP. Both halves are asserted because both
        // are the finding: a rate three orders looser would mean the threshold
        // had slipped, and a move of more than 1 ULP would mean the series had
        // stopped being a truncation of the function Cephes approximates.
        assert!(
            divergent * 100_000 <= compared,
            "{divergent} of {compared} admitted lanes diverged, i.e. above one in \
             100,000; the arm is characterised at 13 in 4,000,000 and a rate this \
             high means the threshold or the series length has moved"
        );
        assert!(
            worst_ulp <= 1,
            "a lane moved {worst_ulp} ULP; the series is supposed to differ from \
             Cephes only by Cephes' own rounding, which cannot reach 2 ULP"
        );
        assert!(compared >= 2_000_000, "only {compared} lanes compared");

        // NON-VACUITY. A near-zero divergence rate above the threshold is also
        // what a copy of the general body would report, so require the series to
        // be WRONG where the guard declines it. Below `1 + sqrt(2)` the
        // reduction itself differs and the disagreement is trivial; the honest
        // window is `[3, 10]`, where both arms use the same reduction and only
        // the approximation differs. The truncated term is `r^10/13` and it
        // enters the result scaled by `r^3` against an ULP that falls only as
        // `r`, so the gap grows as the twelfth power of `1/r`: 1.2 ULP at the
        // threshold, 240 ULP at 10, and nine orders of an ULP at 3. Inside this
        // window the arms have to part company on every lane. Sweeping up to the
        // threshold instead would let the last decile agree and dilute the bar
        // to 84%, which is why the window stops short of it.
        let mut below_divergent = 0u64;
        let mut below_compared = 0u64;
        for step in 1..200_000_i32 {
            let x = 3.0 + f64::from(step) / 200_000.0 * 7.0;
            let candidate = f64x4::from([x, x * 1.01, x * 1.02, x * 1.03]);
            let general = atan_x4(candidate).to_array();
            let asymptotic = atan_x4_asymptotic(candidate).to_array();
            for (a, b) in general.iter().zip(asymptotic.iter()) {
                below_compared += 1;
                if a.to_bits() != b.to_bits() {
                    below_divergent += 1;
                }
            }
        }
        assert!(
            below_divergent * 100 >= below_compared * 99,
            "the series agreed with Cephes on {} of {below_compared} lanes in [3, 10]; \
             it is supposed to be a truncation that fails there, so this much agreement \
             means the arms have converged and the test above proves nothing",
            below_compared - below_divergent
        );
    }

    /// The guard, not the body: `atan_x4_dispatched` must return the general
    /// body's bits on every input, including the ones it declines.
    ///
    /// The straddling middle step feeds it four NaN arguments by construction
    /// (`jb_local_temp_x4` evaluates the above-break branch on lanes whose
    /// `dz` is negative and selects them away), so NaN is a live input here and
    /// not a defensive extra.
    #[test]
    fn atan_x4_dispatch_is_bit_identical_on_every_input_including_declined_ones() {
        use wide::f64x4;

        let threshold = std::f64::consts::SQRT_2 + 1.0;
        let mut compared = 0u64;
        let mut took_fast_path = 0u64;
        let mut took_general_path = 0u64;

        let mut check = |candidate: f64x4| {
            let want = atan_x4(candidate).to_array();
            let got = atan_x4_dispatched(candidate).to_array();
            for (a, b) in want.iter().zip(got.iter()) {
                compared += 1;
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "dispatch moved a lane: {candidate:?}"
                );
            }
            if candidate.simd_gt(T3PO8_X4).all() {
                took_fast_path += 1;
            } else {
                took_general_path += 1;
            }
        };

        for step in -100_000_i32..100_000 {
            let t = f64::from(step) / 10_000.0;
            check(f64x4::from([t, t + 2.4, t * 3.0, t - 1.5]));
            check(f64x4::from([
                threshold + t.abs(),
                threshold + t.abs() * 2.0,
                threshold + t.abs() * 4.0,
                threshold + t.abs() * 8.0,
            ]));
        }
        // The straddling step's shape: some lanes NaN, the rest large.
        check(f64x4::from([f64::NAN, f64::NAN, 40.0, 90.0]));
        check(f64x4::from([f64::NAN; 4]));
        check(f64x4::from([0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY]));

        assert!(compared >= 1_000_000, "only {compared} lanes compared");
        assert!(
            took_fast_path >= 50_000,
            "only {took_fast_path} calls qualified for the fast path; the corpus \
             is not exercising it"
        );
        assert!(
            took_general_path >= 50_000,
            "only {took_general_path} calls were declined; the corpus is not \
             exercising the guard's false arm"
        );
    }

    /// The fast path has to fire on the PRODUCTION plan, not merely on a
    /// synthetic corpus, or the measured saving belongs to nothing.
    ///
    /// This counts qualifying steps of the real 105--500 km plan under the real
    /// `tc` a production input produces, which is the quantity the kernel A/B
    /// was measured against.
    #[test]
    fn the_production_middle_plan_takes_the_fast_path_on_most_of_its_steps() {
        let plan = logquad_x4_fixed_middle_plan();
        // `tc` from a real evaluation rather than a chosen constant.
        let input = Jb2008Input {
            mjd_utc: 52_951.003_805_740_744,
            sun_declination_rad: -0.285_987_757_544_287,
            // The sealed Orekit pair, differenced: sat_ra 1.282_118_868_515_03
            // minus sun_ra 3.046_653_643_566_772. Kept as the subtraction so the
            // provenance of both halves stays legible; one rounding, exactly as
            // the kernel used to perform it.
            hour_angle_rad: 1.282_118_868_515_03 - 3.046_653_643_566_772,
            sat_geocentric_lat_rad: -1.487_718_654_399_9,
            sat_altitude_m: 700_000.0,
            f10: 91.00,
            f10b: 137.10,
            s10: 108.80,
            s10b: 123.80,
            m10: 116.70,
            m10b: 128.50,
            y10: 168.00,
            y10b: 138.60,
            dst_temperature_correction_k: 43.0,
        };
        let exospheric = 1_050.0_f64;
        let transition =
            444.3807 + 0.02385 * exospheric - 392.8292 * (-0.002_135_7 * exospheric).exp();
        let gradient = 0.054_285_714 * (transition - 183.0);
        let amplitude = (exospheric - transition) / std::f64::consts::FRAC_PI_2;
        let tc = TemperatureBroadcast::new([transition, gradient, amplitude, gradient / amplitude]);
        assert!(
            gradient > 0.0 && amplitude > 0.0,
            "the physical regime this plan is measured in must have tc[1], tc[2] > 0"
        );
        let _ = input;

        let mut qualifying = 0usize;
        for step in plan.steps.iter().skip(plan.above_from) {
            let argument = tc.argument_scale * step.break_offset * step.argument_shape;
            if argument.simd_gt(T3PO8_X4).all() {
                qualifying += 1;
            }
        }
        let above = plan.steps.len() - plan.above_from;
        assert_eq!(
            above, 14,
            "the production plan's above-break step count moved"
        );
        assert!(
            qualifying >= 9,
            "only {qualifying} of {above} above-break steps qualify; the measured \
             saving assumed most of them do"
        );
    }

    /// The kernel now takes `sin_cos` of the satellite latitude once instead of
    /// a `cos` in `jb_dtc` and a `sin` in `jb_dlrsl`. On Darwin that lowers to
    /// `__sincos_stret`, which is a different code path in libm, so the claim
    /// that it returns the same bits is a measurement and not an assumption.
    #[test]
    fn sin_cos_is_bit_identical_to_separate_sin_and_cos() {
        let mut compared = 0u64;
        for step in -1_000_000_i32..1_000_000 {
            let x = f64::from(step) * (std::f64::consts::FRAC_PI_2 / 1_000_000.0);
            let (s, c) = x.sin_cos();
            assert_eq!(s.to_bits(), x.sin().to_bits(), "sin x={x}");
            assert_eq!(c.to_bits(), x.cos().to_bits(), "cos x={x}");
            compared += 1;
        }
        assert!(compared >= 2_000_000, "only {compared} compared");
    }

    /// `jb_dlrsl` replaced `% 1.0` (a real `fmod` call) with a truncating
    /// fraction plus a sign-of-zero guard. Plain `fract()` would NOT pass this:
    /// it differs from `fmod` at every non-positive integer.
    #[test]
    fn truncating_fraction_is_bit_identical_to_fmod_one() {
        fn replacement(x: f64) -> f64 {
            let fraction = x - x.trunc();
            if fraction == 0.0 {
                0.0_f64.copysign(x)
            } else {
                fraction
            }
        }
        let mut compared = 0u64;
        for step in -1_000_000_i32..1_000_000 {
            let step_as_f64 = f64::from(step);
            for x in [step_as_f64 / 997.0, step_as_f64, step_as_f64 / 1024.0] {
                assert_eq!((x % 1.0).to_bits(), replacement(x).to_bits(), "x={x}");
                compared += 1;
            }
        }
        for x in [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            4.5e15,
            -4.5e15,
            9.007_199_254_740_992e15,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
        ] {
            assert_eq!((x % 1.0).to_bits(), replacement(x).to_bits(), "x={x}");
            compared += 1;
        }
        assert!(compared >= 2_000_000, "only {compared} compared");
    }

    /// The `jb_tsubc` memo must return the uncached value BIT-FOR-BIT, and must
    /// actually be exercised rather than missing every time.
    ///
    /// Both halves matter. Comparing only the values would pass against a memo
    /// that never hits (and therefore saves nothing), and asserting only that it
    /// hits would pass against one that returns a stale number.
    #[test]
    fn tsubc_memo_matches_uncached_bit_for_bit_and_actually_hits() {
        let base = orekit_local_input(91.0);
        // Distinct solar-index sets, the first repeated last: returning to an
        // earlier key must reproduce its earlier answer. A single-slot memo that
        // silently kept the newest key would pass a two-key test.
        let variants = [
            base,
            Jb2008Input {
                f10b: base.f10b * 1.37,
                ..base
            },
            Jb2008Input {
                s10b: base.s10b * 0.61,
                y10: base.y10 * 1.09,
                ..base
            },
            base,
        ];

        for input in variants {
            let memoized = jb_tsubc(input);
            let direct = jb_tsubc_uncached(input);
            assert_eq!(
                memoized.to_bits(),
                direct.to_bits(),
                "memoized tsubc must equal the uncached value bit-for-bit"
            );
        }

        // Non-vacuity, and it has to be done by POISONING.
        //
        // Everything softer passes against a memo that never hits. `first ==
        // second` holds for any pure function. Checking that the slot's key is
        // unchanged across a repeat also proves nothing: on a MISS the memo
        // recomputes and stores the SAME key, so that assertion is satisfied
        // either way. The only way to tell a hit from a recompute is to make
        // the two return different values.
        let probe = Jb2008Input {
            m10b: base.m10b * 1.21,
            ..base
        };
        let probe_key = [
            probe.f10.to_bits(),
            probe.f10b.to_bits(),
            probe.s10.to_bits(),
            probe.s10b.to_bits(),
            probe.m10.to_bits(),
            probe.m10b.to_bits(),
            probe.y10.to_bits(),
            probe.y10b.to_bits(),
        ];
        let truth = jb_tsubc_uncached(probe);
        // A value the arithmetic cannot produce: `jb_tsubc` is 392.4 plus
        // positive-weighted terms in strictly positive indices.
        let poison = -1.0e9_f64;
        assert_ne!(truth.to_bits(), poison.to_bits());

        TSUBC_MEMO.with(|memo| memo.set((probe_key, poison)));
        assert_eq!(
            jb_tsubc(probe).to_bits(),
            poison.to_bits(),
            "a repeated key MUST be served from the memo slot; returning the \
             recomputed value means the memo never hits and saves nothing"
        );

        // And the other direction: a DIFFERENT key must not be served that
        // slot. Without this, a memo that ignores its key entirely would pass
        // the assertion above.
        let other = Jb2008Input {
            f10: base.f10 * 1.03,
            ..base
        };
        TSUBC_MEMO.with(|memo| memo.set((probe_key, poison)));
        assert_eq!(
            jb_tsubc(other).to_bits(),
            jb_tsubc_uncached(other).to_bits(),
            "a different key must MISS and recompute, not inherit the slot"
        );

        // The sentinel can never collide: every live key has eight strictly
        // positive indices, and `jb2008_density` rejects anything else before
        // `jb_tsubc` is reached.
        assert!(
            [
                probe.f10, probe.f10b, probe.s10, probe.s10b, probe.m10, probe.m10b, probe.y10,
                probe.y10b
            ]
            .iter()
            .all(|index| *index > 0.0),
            "fixture must use the positive indices the guard enforces"
        );
    }

    #[test]
    fn rejects_invalid_inputs_without_fallback() {
        let valid = orekit_local_input(91.0);
        for (input, expected) in [
            (
                Jb2008Input {
                    f10: f64::NAN,
                    ..valid
                },
                Jb2008Error::NonFiniteInput,
            ),
            (
                Jb2008Input {
                    sat_altitude_m: 89_999.999,
                    ..valid
                },
                Jb2008Error::AltitudeOutOfRange,
            ),
            (
                Jb2008Input {
                    sun_declination_rad: std::f64::consts::FRAC_PI_2 + 1.0e-12,
                    ..valid
                },
                Jb2008Error::AngleOutOfRange,
            ),
            (
                Jb2008Input {
                    sat_geocentric_lat_rad: -std::f64::consts::FRAC_PI_2 - 1.0e-12,
                    ..valid
                },
                Jb2008Error::AngleOutOfRange,
            ),
            (
                Jb2008Input { y10b: 0.0, ..valid },
                Jb2008Error::NonPositiveSolarIndex,
            ),
        ] {
            assert_eq!(jb2008_density(input), Err(expected));
        }
    }

    #[test]
    fn combined_invalid_inputs_preserve_validation_precedence() {
        let valid = orekit_local_input(91.0);
        let nonpositive = Jb2008Input { y10b: 0.0, ..valid };
        let bad_angle = Jb2008Input {
            sun_declination_rad: std::f64::consts::FRAC_PI_2 + 1.0e-12,
            ..nonpositive
        };
        let bad_altitude = Jb2008Input {
            sat_altitude_m: 89_999.999,
            ..bad_angle
        };
        let nonfinite = Jb2008Input {
            f10: f64::NAN,
            ..bad_altitude
        };

        for (input, expected) in [
            (nonfinite, Jb2008Error::NonFiniteInput),
            (bad_altitude, Jb2008Error::AltitudeOutOfRange),
            (bad_angle, Jb2008Error::AngleOutOfRange),
            (nonpositive, Jb2008Error::NonPositiveSolarIndex),
        ] {
            assert_eq!(jb2008_density_fitted_v7(input), Err(expected));
        }
    }

    /// The one-`atan2` hour angle agrees with the two-`atan2` pair it replaced,
    /// in the OBSERVABLES rather than the intermediates.
    ///
    /// This is the gate for the R44 adapter change. The old adapter handed over
    /// `sat_ra` and `sun_ra` from two `atan2` calls and the kernel subtracted
    /// them; the new one computes the difference directly as
    /// `atan2(u × v, u · v)`. Those two forms do NOT produce the same `h`: the
    /// old one lands in `(-2π, 2π)` and the new one in `[-π, π]`, so the raw
    /// `tau` intermediate differs by a full turn. Comparing `tau` would
    /// therefore "fail" while nothing observable had moved.
    ///
    /// `h` reaches exactly two observables, and both are turn-invariant:
    ///
    /// * `tau` is consumed only as `|cos(tau/2)|` in [`jb_tsub_l`], and
    ///   `|cos((tau + 2π)/2)| = |cos(tau/2 + π)| = |cos(tau/2)|`.
    /// * `solar_time_hour` is normalized into `[0, 24)` and consumed only as
    ///   `st = hour/24` by the [`jb_dtc`] polynomials.
    ///
    /// So this checks those two, not `tau`. The residual is rounding.
    ///
    /// The exploratory corpus was 5,000,000 samples drawn over ±2 turns of both
    /// right ascensions; it reported `|cos(tau/2)|` agreeing to 1.08e-15,
    /// solar-hour to 1.42e-14, and ZERO samples needing a 24 h relabel. Shrunk
    /// to 20,000 here so the suite stays in the seconds -- the failure mode this
    /// guards (a quadrant or turn error in the cross/dot form) is systematic,
    /// not rare, so it surfaces in the first hundred samples. The bounds below
    /// are the 5M figures rounded up one decade.
    #[test]
    fn one_atan2_hour_angle_matches_the_two_atan2_pair_in_every_observable() {
        // The retired form, verbatim: two right ascensions, each wrapped, then
        // differenced.
        fn old_h(sat: [f64; 2], sun: [f64; 2]) -> f64 {
            wrap_to_tau(sat[1].atan2(sat[0])) - wrap_to_tau(sun[1].atan2(sun[0]))
        }
        // The adapter's new form.
        fn new_h(sat: [f64; 2], sun: [f64; 2]) -> f64 {
            sat[1]
                .mul_add(sun[0], -(sat[0] * sun[1]))
                .atan2(sat[0].mul_add(sun[0], sat[1] * sun[1]))
        }
        // The two observables, as the kernel forms them.
        fn observables(h: f64) -> (f64, f64) {
            let h = wrap_to_tau(h);
            let tau = h - 0.645_771_82 + 0.104_719_76 * (h + 0.750_491_58).sin();
            let mut hour = (h + std::f64::consts::PI).to_degrees() / 15.0;
            if hour >= 24.0 {
                hour -= 24.0;
            } else if hour < 0.0 {
                hour += 24.0;
            }
            ((0.5 * tau).cos().abs(), hour)
        }

        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        // `state >> 11` keeps 53 bits, so the conversion below is EXACT and the
        // division by 2^53 lands in [0, 1). Both lints fire on the shape rather
        // than on a real loss; a checked wrapper here would change the draw
        // sequence and with it every fixture this generator feeds.
        #[expect(
            clippy::cast_precision_loss,
            clippy::as_conversions,
            reason = "shifted to 53 bits first, so the u64 -> f64 conversion is exact"
        )]
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / 9_007_199_254_740_992.0
        };
        let (mut worst_cos, mut worst_hour) = (0.0_f64, 0.0_f64);
        for _ in 0..20_000 {
            // Deliberately outside [-pi, pi], so the wrap paths are exercised.
            let sat_angle = (next() - 0.5) * 4.0 * std::f64::consts::TAU;
            let sun_angle = (next() - 0.5) * 4.0 * std::f64::consts::TAU;
            let sat = [7000.0 * sat_angle.cos(), 7000.0 * sat_angle.sin()];
            let sun = [1.496e8 * sun_angle.cos(), 1.496e8 * sun_angle.sin()];

            let (cos_old, hour_old) = observables(old_h(sat, sun));
            let (cos_new, hour_new) = observables(new_h(sat, sun));

            let mut hour_gap = (hour_old - hour_new).abs();
            assert!(
                hour_gap < 12.0 || (hour_gap - 24.0).abs() < 1e-9,
                "solar hour disagreed by {hour_gap}, which is neither a rounding \
                 residual nor a clean 24 h relabel: {hour_old} vs {hour_new}"
            );
            if hour_gap > 12.0 {
                hour_gap = (hour_gap - 24.0).abs();
            }
            worst_cos = worst_cos.max((cos_old - cos_new).abs());
            worst_hour = worst_hour.max(hour_gap);
        }
        assert!(
            worst_cos < 1.0e-14,
            "|cos(tau/2)| moved by {worst_cos:e}; the 5M-sample corpus saw \
             1.08e-15, so this is a structural change, not rounding"
        );
        assert!(
            worst_hour < 1.0e-13,
            "solar hour moved by {worst_hour:e} h; the 5M-sample corpus saw \
             1.42e-14, so this is a structural change, not rounding"
        );
    }

    /// The hour angle carries whole-turn offsets in, and the kernel must absorb
    /// them.
    ///
    /// This is the descendant of `right_ascensions_normalize_to_zero_through_
    /// two_pi`, which fed a whole-turn offset into each of the two right
    /// ascensions separately. Those fields are gone -- the kernel takes the
    /// difference directly -- but the property they were guarding is not: the
    /// single input is still normalized by `wrap_to_tau`, and a caller that
    /// hands over an unwrapped angle must still get the same density. Deleting
    /// this test along with the fields would have retired the only check on
    /// that normalization.
    #[test]
    fn hour_angle_normalizes_to_zero_through_two_pi() {
        let input = orekit_local_input(400.0);
        let expected = jb2008_density(input).unwrap();
        for turns in [-3.0_f64, -2.0, -1.0, 1.0, 2.0, 5.0] {
            let normalized = jb2008_density(Jb2008Input {
                hour_angle_rad: turns.mul_add(std::f64::consts::TAU, input.hour_angle_rad),
                ..input
            })
            .unwrap();
            assert!(
                (normalized - expected).abs() / expected < 2.0e-14,
                "a {turns} turn offset on the hour angle moved the density: \
                 {normalized:e} vs {expected:e}"
            );
        }
    }

    // ---------------------------------------------------------------- R16 --
    // Abscissa-count ladder instrumentation. Investigation only: none of the
    // profiles below is reachable from production, and nothing here asserts a
    // production value. Run with
    // `cargo test -p jb_rs --release r16_ -- --nocapture`.

    /// One rung of the middle/upper log-step ladder.
    ///
    /// The middle plan is switched OFF because a fixed plan needs a `const`
    /// array length and the whole point of a ladder is that the step count is
    /// the parameter. `exact_fixed_middle_plan_matches_dynamic_loop_bits` and
    /// its approximation twin already establish that the planned and dynamic
    /// middle segments agree BIT for bit, so the dynamic walk is the correct
    /// stand-in for what a landed rung would compute.
    macro_rules! r16_ladder_profile {
        ($name:ident, $lower:expr, $middle:expr, $upper:expr) => {
            struct $name;

            impl Sealed for $name {}

            impl QuadratureProfile for $name {
                const LOWER_LOG_STEP: f64 = $lower;
                const MIDDLE_LOG_STEP: f64 = $middle;
                const UPPER_LOG_STEP: f64 = $upper;
                const USE_FIXED_LOWER_PLAN: bool = false;
                const RETIRE_SPECIES_ROUND_TRIP: bool = true;
                const USE_FIXED_MIDDLE_PLAN: bool = false;
                const RETIRE_ZR_ROUND_TRIP: bool = LogQuadratureX4ApproxV1::RETIRE_ZR_ROUND_TRIP;
                const DLRSL_ZERO_ABOVE_KM: f64 = LogQuadratureX4ApproxV1::DLRSL_ZERO_ABOVE_KM;
                const FITTED_UPPER_SEGMENT: bool = LogQuadratureX4ApproxV1::FITTED_UPPER_SEGMENT;

                fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
                    LogQuadratureX4ApproxV1::fixed_lower_state(tc, ain)
                }

                fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
                    LogQuadratureX4ApproxV1::fixed_middle_state(tc, ain)
                }
            }
        };
    }

    // The converged reference. Every segment is far finer than either shipped
    // profile, so its residual quadrature error is negligible against the
    // differences being measured.
    r16_ladder_profile!(R16Converged, 0.000_5, 0.000_5, 0.000_5);
    // The shipped model-5 grid, re-expressed as a ladder rung so the baseline
    // reading comes through the same code path as every other rung.
    r16_ladder_profile!(R16Middle100, 0.040, 0.100, 0.300);
    r16_ladder_profile!(R16Middle125, 0.040, 0.125, 0.300);
    r16_ladder_profile!(R16Middle150, 0.040, 0.150, 0.300);
    r16_ladder_profile!(R16Middle200, 0.040, 0.200, 0.300);
    r16_ladder_profile!(R16Middle250, 0.040, 0.250, 0.300);
    r16_ladder_profile!(R16Middle300, 0.040, 0.300, 0.300);
    r16_ladder_profile!(R16Middle320, 0.040, 0.320, 0.300);
    r16_ladder_profile!(R16Middle350, 0.040, 0.350, 0.300);
    r16_ladder_profile!(R16Middle400, 0.040, 0.400, 0.300);
    // Upper-segment axis, middle held at the shipped 0.100.
    r16_ladder_profile!(R16Upper700, 0.040, 0.100, 0.700);
    // Lower-segment axis, for completeness: it carries no `atan` at all.
    r16_ladder_profile!(R16Lower160, 0.160, 0.100, 0.300);
    // The two axes that pay, moved together.
    r16_ladder_profile!(R16Middle200Upper700, 0.040, 0.200, 0.700);
    r16_ladder_profile!(R16Middle300Upper700, 0.040, 0.300, 0.700);

    /// The production-plausible input cloud: the sealed arc's altitude band
    /// crossed with solar and geometric variation.
    ///
    /// The altitude band is the one the sealed V3 arc actually visits
    /// (626.2--985.7 km, per `USE_FIXED_MIDDLE_PLAN`'s census). The solar
    /// indices bracket quiet to active conditions, because the exospheric
    /// temperature is what sets the arctangent's operand range and therefore
    /// how hard the integrand is to resolve.
    fn r16_input_cloud() -> Vec<Jb2008Input> {
        let mut cloud = Vec::new();
        for altitude_step in 0..=74 {
            let altitude_km = 626.0 + f64::from(altitude_step) * 5.0;
            for (f10, f10b, s10, s10b, m10, m10b, y10, y10b, dst) in [
                (70.0, 72.0, 60.0, 62.0, 65.0, 66.0, 70.0, 72.0, 0.0),
                (91.0, 137.1, 108.8, 123.8, 116.7, 128.5, 168.0, 138.6, 43.0),
                (
                    150.0, 140.0, 145.0, 138.0, 150.0, 142.0, 160.0, 150.0, -20.0,
                ),
                (
                    250.0, 220.0, 230.0, 210.0, 240.0, 225.0, 260.0, 240.0, 120.0,
                ),
            ] {
                for (sat_lat, sun_dec, sat_ra, sun_ra) in [
                    (0.0, 0.0, 0.0, 0.0),
                    (1.2, 0.4, 1.28, 3.05),
                    (-1.487_718_654_399_9, -0.285_987_757_544_287, 1.282, 3.047),
                    (0.6, -0.4, 4.5, 1.1),
                ] {
                    cloud.push(Jb2008Input {
                        mjd_utc: 52_951.003_805_740_744,
                        sun_declination_rad: sun_dec,
                        hour_angle_rad: sat_ra - sun_ra,
                        sat_geocentric_lat_rad: sat_lat,
                        sat_altitude_m: altitude_km * 1000.0,
                        f10,
                        f10b,
                        s10,
                        s10b,
                        m10,
                        m10b,
                        y10,
                        y10b,
                        dst_temperature_correction_k: dst,
                    });
                }
            }
        }
        cloud
    }

    /// Boole abscissae and `atan`-arm hits per kernel call, from the
    /// quadrature's own step formulas.
    ///
    /// Structurally the same walk `dynamic_middle_state` performs, counting
    /// instead of integrating. It is a count of the SHIPPED path: the fixed
    /// plans reproduce the dynamic loop's abscissae bit for bit, so the count
    /// does not depend on which of the two runs.
    ///
    /// Returns the abscissa count, the count above 125 km, and the number of
    /// `atan_x4` invocations.
    ///
    /// The third number is the one that prices the lever and it is NOT the
    /// second divided by four. `jb_local_temp_step_x4` dispatches per STEP: a
    /// step with even one lane above 125 km takes an arm that evaluates
    /// `atan_x4` on all four lanes. The straddling step therefore costs a full
    /// `atan_x4` while contributing only one or two abscissae to the second
    /// count.
    fn r16_abscissa_count(
        altitude_km: f64,
        lower: f64,
        middle: f64,
        upper: f64,
    ) -> (usize, usize, usize) {
        let mut total = 0usize;
        let mut with_atan = 0usize;
        let mut atan_calls = 0usize;
        let mut walk = |z_start: f64, z_end: f64, n: u32| -> f64 {
            let zr = ((z_end / z_start).ln() / f64::from(n)).exp();
            let mut zend = z_start;
            for _ in 0..n {
                let z0 = zend;
                zend = zr * z0;
                let dz = 0.25 * (zend - z0);
                let mut z = z0;
                let mut step_has_atan = false;
                for _ in 0..4 {
                    z += dz;
                    total = total.saturating_add(1);
                    if z > 125.0 {
                        with_atan = with_atan.saturating_add(1);
                        step_has_atan = true;
                    }
                }
                if step_has_atan {
                    atan_calls = atan_calls.saturating_add(1);
                }
            }
            zend
        };

        let z2 = altitude_km.min(105.0);
        let n1 = jb_step_count((z2 / 90.0).ln() / lower).expect("lower step count");
        let after_lower = walk(90.0, z2, n1);
        if altitude_km <= 105.0 {
            return (total, with_atan, atan_calls);
        }
        let middle_end = altitude_km.min(500.0);
        let n2 =
            jb_step_count((middle_end / after_lower).ln() / middle).expect("middle step count");
        let after_middle = walk(after_lower, middle_end, n2);
        let upper_end = altitude_km.max(500.0);
        let r = if altitude_km > 500.0 { upper } else { middle };
        let n3 = jb_step_count((upper_end / after_middle).ln() / r).expect("upper step count");
        walk(after_middle, upper_end, n3);
        (total, with_atan, atan_calls)
    }

    /// Relative density error of one ladder rung against the converged
    /// reference, over the whole input cloud.
    fn r16_rung_error<P: QuadratureProfile>(
        cloud: &[Jb2008Input],
        reference: &[f64],
    ) -> (f64, f64, f64) {
        let mut errors: Vec<f64> = Vec::with_capacity(cloud.len());
        for (input, truth) in cloud.iter().zip(reference.iter()) {
            let value =
                jb2008_density_with_profile::<P>(*input).expect("ladder rung must evaluate");
            errors.push(((value - truth) / truth).abs());
        }
        let worst = errors.iter().copied().fold(0.0_f64, f64::max);
        let mean = errors.iter().sum::<f64>() / errors.len().to_f64().unwrap_or(1.0);
        errors.sort_by(f64::total_cmp);
        let median = errors.get(errors.len() / 2).copied().unwrap_or(f64::NAN);
        (worst, median, mean)
    }

    #[test]
    fn r16_middle_step_ladder_density_accuracy() {
        let cloud = r16_input_cloud();
        let reference: Vec<f64> = cloud
            .iter()
            .map(|input| {
                jb2008_density_with_profile::<R16Converged>(*input)
                    .expect("converged reference must evaluate")
            })
            .collect();
        // The exact (model-4) profile against the same converged reference, so
        // the ladder can be read against what the Orekit-compatible grid is
        // itself worth.
        let mut exact_worst = 0.0_f64;
        for (input, truth) in cloud.iter().zip(reference.iter()) {
            let value = jb2008_density(*input).expect("exact profile must evaluate");
            exact_worst = exact_worst.max(((value - truth) / truth).abs());
        }

        println!(
            "R16_LADDER cloud_cases={} (626--996 km x 4 solar x 4 geometry)",
            cloud.len()
        );
        println!("R16_LADDER exact_model4_worst_rel={exact_worst:.6e}");
        println!(
            "R16_LADDER {:<28} {:>10} {:>10} {:>10} {:>7} {:>7} {:>8} {:>9}",
            "profile",
            "worst_rel",
            "median_rel",
            "mean_rel",
            "abs626",
            "abs986",
            "atan_abs",
            "atan_x4"
        );

        macro_rules! report {
            ($name:ident, $label:literal, $lower:expr, $middle:expr, $upper:expr) => {{
                let (worst, median, mean) = r16_rung_error::<$name>(&cloud, &reference);
                let (a626, t626, c626) = r16_abscissa_count(626.2, $lower, $middle, $upper);
                let (a986, t986, c986) = r16_abscissa_count(985.7, $lower, $middle, $upper);
                println!(
                    "R16_LADDER {:<28} {worst:>10.3e} {median:>10.3e} {mean:>10.3e} \
                     {a626:>7} {a986:>7} {t626:>3}-{t986:<4} {c626:>4}-{c986:<4}",
                    $label
                );
            }};
        }

        report!(
            R16Middle100,
            "middle 0.100 (SHIPPED m5)",
            0.040,
            0.100,
            0.300
        );
        report!(R16Middle125, "middle 0.125", 0.040, 0.125, 0.300);
        report!(R16Middle150, "middle 0.150", 0.040, 0.150, 0.300);
        report!(R16Middle200, "middle 0.200", 0.040, 0.200, 0.300);
        report!(R16Middle250, "middle 0.250", 0.040, 0.250, 0.300);
        report!(R16Middle300, "middle 0.300", 0.040, 0.300, 0.300);
        report!(R16Middle320, "middle 0.320", 0.040, 0.320, 0.300);
        report!(R16Middle350, "middle 0.350", 0.040, 0.350, 0.300);
        report!(R16Middle400, "middle 0.400", 0.040, 0.400, 0.300);
        report!(R16Upper700, "upper 0.700", 0.040, 0.100, 0.700);
        report!(R16Lower160, "lower 0.160", 0.160, 0.100, 0.300);
        report!(
            R16Middle200Upper700,
            "middle 0.200 + upper 0.700",
            0.040,
            0.200,
            0.700
        );
        report!(
            R16Middle300Upper700,
            "middle 0.300 + upper 0.700",
            0.040,
            0.300,
            0.700
        );
    }

    /// Verbatim from `tests/jb2008_logquad_x4_probe.rs`, so the ladder can be
    /// read against the threshold that actually governs model 5.
    ///
    /// `x4_broad_grid_density_error_stays_within_candidate_threshold` bounds
    /// `max |x4 - exact| / exact` at `3.0e-6` over the lattice below, and that
    /// lattice runs 90 km to 35,000 km --- far wider than the 626--986 km band
    /// production visits. A rung can therefore be comfortable on the arc and
    /// still be inadmissible here, which is why the ladder is scored on this
    /// grid as well as on the production cloud.
    fn r16_broad_grid_input(
        mjd_utc: f64,
        altitude_km: f64,
        latitude_deg: f64,
        local_solar_time_hour: f64,
    ) -> Jb2008Input {
        let solar_phase = std::f64::consts::TAU * (mjd_utc - 51_999.75) / 365.2422;
        // The Sun's right ascension used to be synthesized here only so that the
        // satellite's could be built as `sun_ra + hour_angle` and the kernel
        // could subtract it back off. The kernel takes the hour angle, which
        // this fixture already had in hand.
        let hour_angle =
            local_solar_time_hour * std::f64::consts::TAU / 24.0 - std::f64::consts::PI;
        let f10b = 120.0 + 25.0 * solar_phase.cos();
        let s10b = 112.0 + 18.0 * (solar_phase + 0.2).sin();
        let m10b = 128.0 + 16.0 * (solar_phase - 0.5).cos();
        let y10b = 138.0 + 20.0 * (solar_phase + 0.7).sin();
        Jb2008Input {
            mjd_utc,
            sun_declination_rad: 23.44_f64.to_radians() * solar_phase.sin(),
            hour_angle_rad: hour_angle,
            sat_geocentric_lat_rad: latitude_deg.to_radians(),
            sat_altitude_m: altitude_km * 1000.0,
            f10: f10b + 6.0 * (solar_phase + 0.1).sin(),
            f10b,
            s10: s10b + 5.0 * (solar_phase + 0.4).cos(),
            s10b,
            m10: m10b + 4.0 * (solar_phase - 0.3).sin(),
            m10b,
            y10: y10b + 7.0 * (solar_phase + 0.8).cos(),
            y10b,
            dst_temperature_correction_k: 25.0 + 18.0 * (solar_phase - 0.2).sin(),
        }
    }

    #[test]
    fn r16_middle_step_ladder_against_the_standing_threshold() {
        let mut grid = Vec::with_capacity(1800);
        for mjd_utc in [51_999.75, 54_000.0, 57_000.25, 60_000.0, 60_648.5] {
            for altitude_km in [
                90.0, 91.0, 100.0, 105.0, 120.0, 200.0, 240.0, 300.0, 400.0, 500.0, 600.0, 800.0,
                1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 35_000.0,
            ] {
                for latitude_deg in [-75.0, -45.0, 0.0, 45.0, 75.0] {
                    for local_solar_time_hour in [0.0, 6.0, 12.0, 18.0] {
                        grid.push(r16_broad_grid_input(
                            mjd_utc,
                            altitude_km,
                            latitude_deg,
                            local_solar_time_hour,
                        ));
                    }
                }
            }
        }
        assert_eq!(grid.len(), 1800);
        let exact: Vec<f64> = grid
            .iter()
            .map(|input| jb2008_density(*input).expect("exact density"))
            .collect();

        println!(
            "R16_THRESHOLD standing bound = 3.0e-6 on max, over {} cases",
            grid.len()
        );
        println!(
            "R16_THRESHOLD {:<28} {:>10} {:>10} {:>9}",
            "profile", "max_rel", "p99_rel", "verdict"
        );

        macro_rules! against_threshold {
            ($name:ident, $label:literal) => {{
                let mut errors: Vec<f64> = Vec::with_capacity(grid.len());
                // WHERE the maximum sits decides whether the standing bound is
                // a statement about production or about the far field.
                let mut worst_altitude_km = f64::NAN;
                let mut worst_error = 0.0_f64;
                // A rung that REJECTS an input the shipped grid accepts is a
                // harder failure than a large error, so it is counted rather
                // than panicked on: the ladder has to report every rung.
                let mut rejected = 0usize;
                let mut rejected_altitudes_km: Vec<f64> = Vec::new();
                for (input, truth) in grid.iter().zip(exact.iter()) {
                    match jb2008_density_with_profile::<$name>(*input) {
                        Ok(value) => {
                            let error = ((value - truth) / truth).abs();
                            if error > worst_error {
                                worst_error = error;
                                worst_altitude_km = input.sat_altitude_m / 1000.0;
                            }
                            errors.push(error);
                        }
                        Err(_) => {
                            rejected += 1;
                            let altitude_km = input.sat_altitude_m / 1000.0;
                            if !rejected_altitudes_km.contains(&altitude_km) {
                                rejected_altitudes_km.push(altitude_km);
                            }
                        }
                    }
                }
                errors.sort_by(f64::total_cmp);
                let max = errors.last().copied().unwrap_or(f64::NAN);
                let p99 = errors
                    .get(errors.len().saturating_mul(99) / 100)
                    .copied()
                    .unwrap_or(f64::NAN);
                let verdict = if rejected > 0 {
                    "REJECTS"
                } else if max <= 3.0e-6 {
                    "PASS"
                } else {
                    "FAIL"
                };
                println!(
                    "R16_THRESHOLD {:<28} {max:>10.3e} {p99:>10.3e} {verdict:>9} worst_at_km={worst_altitude_km} \
                     rejected={rejected} at_km={rejected_altitudes_km:?}",
                    $label
                );
            }};
        }

        against_threshold!(R16Middle100, "middle 0.100 (SHIPPED m5)");
        against_threshold!(R16Middle125, "middle 0.125");
        against_threshold!(R16Middle150, "middle 0.150");
        against_threshold!(R16Middle200, "middle 0.200");
        against_threshold!(R16Middle250, "middle 0.250");
        against_threshold!(R16Middle300, "middle 0.300");
        against_threshold!(R16Middle320, "middle 0.320");
        against_threshold!(R16Middle400, "middle 0.400");
        against_threshold!(R16Upper700, "upper 0.700");
        against_threshold!(R16Lower160, "lower 0.160");
        against_threshold!(R16Middle200Upper700, "middle 0.200 + upper 0.700");
        against_threshold!(R16Middle300Upper700, "middle 0.300 + upper 0.700");
    }

    /// The shipped model 6 IS R16's arm C, bit for bit.
    ///
    /// `LogQuadratureX4ApproxV2` is written as its own profile with literal log
    /// steps, and `R16Middle300Upper700` is the ladder rung every measurement in
    /// the decision document was taken on. Nothing but this test says they are
    /// the same grid. Without it, "model 6 is worth 12% and errs 5.747e-5" is a
    /// claim about a rung that no longer has to be the one production flies.
    #[test]
    fn model_six_is_bit_identical_to_the_ladder_rung_it_was_chosen_from() {
        let mut compared = 0usize;
        for altitude_km in [
            105.0, 120.0, 200.0, 300.0, 500.0, 626.0, 800.0, 986.0, 2000.0, 35_000.0,
        ] {
            for latitude_deg in [-75.0, 0.0, 45.0] {
                let input = r16_broad_grid_input(57_000.25, altitude_km, latitude_deg, 6.0);
                let shipped = jb2008_density_with_profile::<LogQuadratureX4ApproxV2>(input);
                let rung = jb2008_density_with_profile::<R16Middle300Upper700>(input);
                match (shipped, rung) {
                    (Ok(shipped), Ok(rung)) => assert_eq!(
                        shipped.to_bits(),
                        rung.to_bits(),
                        "model 6 and arm C differ at {altitude_km} km, {latitude_deg} deg"
                    ),
                    (shipped, rung) => assert_eq!(shipped.is_err(), rung.is_err()),
                }
                compared += 1;
            }
        }
        assert_eq!(compared, 30);
    }

    /// The re-scoped 1.0e-4 bound has teeth: it REJECTS the off-cliff rung.
    ///
    /// `v2_broad_grid_density_error_stays_within_rescoped_bound` asserts model 6
    /// passes at 1.0e-4. That is only worth something if 1.0e-4 is a bound some
    /// reachable profile FAILS. `middle 0.400` is two cliffs past the Boole
    /// rule's resolving floor and is exactly the rung the two strict-HF 1.0 m
    /// accuracy gates wave through, so it is the right poison: this test is the
    /// difference between a bound and a decoration.
    #[test]
    fn the_rescoped_bound_rejects_the_rung_the_accuracy_gates_wave_through() {
        // The gate's own constant, not a local copy of its value -- see
        // `V2_RESCOPED_DENSITY_BOUND`. A second literal here would let the
        // shipped bound be relaxed without this proof noticing.
        const RESCOPED_BOUND: f64 = super::V2_RESCOPED_DENSITY_BOUND;

        let mut worst = 0.0_f64;
        for altitude_km in [120.0, 200.0, 300.0, 500.0, 626.0, 986.0, 35_000.0] {
            for latitude_deg in [-45.0, 0.0, 45.0] {
                let input = r16_broad_grid_input(57_000.25, altitude_km, latitude_deg, 12.0);
                let exact = jb2008_density(input).expect("exact density");
                let poisoned = jb2008_density_with_profile::<R16Middle400>(input)
                    .expect("the poison rung must still evaluate");
                worst = worst.max((poisoned - exact).abs() / exact);
            }
        }

        println!("R22_POISON middle_0.400 max_relative_error={worst:e} bound={RESCOPED_BOUND:e}");
        assert!(
            worst > RESCOPED_BOUND,
            "the 1.0e-4 bound does not reject middle 0.400 ({worst:e}); it is not bounding \
             anything and the model 6 gate above is vacuous"
        );
    }

    /// Horner over a slice, matching `fitted_v7_horner` bit for bit.
    ///
    /// Seeding the fold with `0.0` costs one extra `mul_add` whose result is
    /// `0.0 * u + c` — exact under fusion, so this and the production
    /// evaluation agree in every bit. That equality is what lets a rung fitted
    /// here be compared against the shipped rung on the same footing.
    fn poison_horner(coefficients: &[f64], u: f64) -> f64 {
        coefficients
            .iter()
            .rev()
            .fold(0.0_f64, |acc, &c| acc.mul_add(u, c))
    }

    /// [`LogQuadratureFittedV7`] with the upper segment FITTED.
    ///
    /// The shipped profile walks that segment — `FITTED_UPPER_SEGMENT` is false
    /// everywhere, because the expansion measured SLOWER than the Boole step it
    /// replaces (see that constant). This profile is where the arm stays live:
    /// identical in every other respect, so differencing a density against the
    /// shipped one isolates the expansion and nothing else, and the accuracy
    /// gate below stays non-vacuous while the lever is parked.
    ///
    /// Flipping the production constant to true is the whole of what landing it
    /// would take; this profile is what keeps that flip measured.
    struct FittedV7FittedUpper;
    impl Sealed for FittedV7FittedUpper {}
    impl QuadratureProfile for FittedV7FittedUpper {
        const LOWER_LOG_STEP: f64 = LogQuadratureFittedV7::LOWER_LOG_STEP;
        const MIDDLE_LOG_STEP: f64 = LogQuadratureFittedV7::MIDDLE_LOG_STEP;
        const UPPER_LOG_STEP: f64 = LogQuadratureFittedV7::UPPER_LOG_STEP;
        const USE_FIXED_LOWER_PLAN: bool = LogQuadratureFittedV7::USE_FIXED_LOWER_PLAN;
        const RETIRE_SPECIES_ROUND_TRIP: bool = LogQuadratureFittedV7::RETIRE_SPECIES_ROUND_TRIP;
        const RETIRE_ZR_ROUND_TRIP: bool = LogQuadratureFittedV7::RETIRE_ZR_ROUND_TRIP;
        const DLRSL_ZERO_ABOVE_KM: f64 = LogQuadratureFittedV7::DLRSL_ZERO_ABOVE_KM;
        const USE_FIXED_MIDDLE_PLAN: bool = LogQuadratureFittedV7::USE_FIXED_MIDDLE_PLAN;
        /// The one difference, and the whole point of this profile.
        const FITTED_UPPER_SEGMENT: bool = true;

        fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
            LogQuadratureFittedV7::fixed_lower_state(tc, ain)
        }

        fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
            LogQuadratureFittedV7::fixed_middle_state(tc, ain)
        }
    }

    /// The R57 upper-segment expansion must cost what its own truncation
    /// argument says it costs, and must actually fire.
    ///
    /// The grid spans the fit's whole altitude domain and the whole fitted
    /// exospheric-temperature domain, not just the censused production band,
    /// because the guard admits all of it. Both are swept by moving the driver
    /// indices, so the temperatures reached are ones the model actually
    /// produces rather than ones injected into `tc`.
    #[test]
    fn the_fitted_upper_segment_costs_what_its_truncation_says() {
        // The design bound: the expansion's own truncation is 3.9e-9 of `sum3`
        // at the worst corner and `sum3` is amplified ~45x by the species
        // exponent, so ~2e-7 of the density is what the arithmetic predicts.
        // 1.0e-6 is that with five times' room, and it is still 57x inside the
        // 5.75e-5 the fitted profile's own gate already accepts.
        const UPPER_FIT_DENSITY_BOUND: f64 = 1.0e-6;

        let mut worst = 0.0_f64;
        let mut worst_at = (0.0, 0.0);
        let mut compared = 0u32;
        let mut moved = 0u32;
        for altitude_step in 0..24 {
            let altitude_km = UPPER_FIT_ALT_LO
                + f64::from(altitude_step) * ((UPPER_FIT_ALT_HI - UPPER_FIT_ALT_LO) / 23.0);
            for f10_step in 0..12 {
                // Sweeping the solar indices is what sweeps `Texo`; the pair
                // below reaches roughly 640 K to 1900 K, which straddles the
                // censused 608.9--1627.5 K band on both sides.
                let f10 = 60.0 + f64::from(f10_step) * 22.0;
                let mut input = r16_broad_grid_input(57_000.25, altitude_km, 35.0, 12.0);
                input.f10 = f10;
                input.f10b = f10;
                input.s10 = f10;
                input.s10b = f10;
                let Ok(walked) = jb2008_density_with_profile::<LogQuadratureFittedV7>(input) else {
                    continue;
                };
                let fitted = jb2008_density_with_profile::<FittedV7FittedUpper>(input)
                    .expect("the fitted twin must evaluate wherever the walked one does");
                compared += 1;
                if fitted.to_bits() != walked.to_bits() {
                    moved += 1;
                }
                let relative = (fitted - walked).abs() / walked;
                if relative > worst {
                    worst = relative;
                    worst_at = (altitude_km, f10);
                }
            }
        }
        println!(
            "R57_UPPER_FIT compared={compared} moved={moved} \
             max_relative_density_error={worst:e} at altitude_km={} f10={} \
             bound={UPPER_FIT_DENSITY_BOUND:e}",
            worst_at.0, worst_at.1
        );
        assert!(
            worst <= UPPER_FIT_DENSITY_BOUND,
            "the upper-segment expansion moves the density {worst:e} at \
             altitude {} km, f10 {}, against its own {UPPER_FIT_DENSITY_BOUND:e} \
             bound. The truncation argument in `fitted_upper_segment` predicts \
             ~2e-7; a reading above this means the separation has a hole or a \
             fit lost its degree, and the number is the diagnosis.",
            worst_at.0,
            worst_at.1
        );
        assert!(compared >= 200, "only {compared} rows compared");
        // NON-VACUITY. A bound is satisfied trivially by an arm that never
        // fires, and the guard has four preconditions any of which would
        // silently disable it. The two profiles differ ONLY in that arm, so a
        // moved density is proof it ran. Not every row moves — some agree to
        // the bit because the expansion lands inside a ULP — so the bar is a
        // large majority and the measured rate is 92%.
        assert!(
            moved * 4 >= compared * 3,
            "only {moved} of {compared} rows moved; the fitted upper segment is \
             not firing on most of its own domain, so the bound above is \
             measuring the walked path against itself"
        );

        // THE FALLBACK, which the design owes an assertion and not a claim.
        // Above the altitude domain the arm must decline, and declining means
        // BIT-IDENTICAL to the walked twin rather than merely close — the two
        // profiles differ in nothing else.
        let mut outside = 0u32;
        for altitude_step in 0..10 {
            let altitude_km = 1050.0 + f64::from(altitude_step) * 200.0;
            for f10_step in 0..6 {
                let f10 = 60.0 + f64::from(f10_step) * 44.0;
                let mut input = r16_broad_grid_input(57_000.25, altitude_km, 35.0, 12.0);
                input.f10 = f10;
                input.f10b = f10;
                input.s10 = f10;
                input.s10b = f10;
                let Ok(walked) = jb2008_density_with_profile::<LogQuadratureFittedV7>(input) else {
                    continue;
                };
                let fitted = jb2008_density_with_profile::<FittedV7FittedUpper>(input)
                    .expect("the fitted twin must evaluate wherever the walked one does");
                assert_eq!(
                    fitted.to_bits(),
                    walked.to_bits(),
                    "above {UPPER_FIT_ALT_HI} km the fitted arm must decline and the \
                     two profiles must be bit-identical; at {altitude_km} km, f10 \
                     {f10}, they are not, so the domain guard is not the guard"
                );
                outside += 1;
            }
        }
        assert!(
            outside >= 40,
            "only {outside} rows exercised the out-of-domain fallback"
        );
    }

    /// The fit's altitude domain must stay inside the ONE-STEP regime.
    ///
    /// `fitted_upper_segment` reproduces a single Boole panel. The walk uses one
    /// panel only while `jb_step_count` returns 1, i.e. below
    /// `z * exp(UPPER_LOG_STEP)`; above it the walk is a two-panel quadrature
    /// and no single-panel fit can follow it. That cost 1.32e-4 of density —
    /// 660x the expansion's own truncation — when the domain was first set to
    /// 1500 km, so this is a measured trap and not a hypothetical one.
    ///
    /// Pinned here rather than asserted in prose because the two numbers live
    /// far apart: a later change to `UPPER_LOG_STEP` would silently reopen it.
    #[test]
    fn the_upper_fit_domain_stays_inside_the_one_step_regime() {
        let plan = logquad_x4_v2_fixed_middle_plan();
        let boundary = plan.z * LogQuadratureFittedV7::UPPER_LOG_STEP.exp();
        println!(
            "R57_UPPER_FIT_DOMAIN hi={UPPER_FIT_ALT_HI} one_step_boundary={boundary} \
             margin={:.4}%",
            (boundary / UPPER_FIT_ALT_HI - 1.0) * 100.0
        );
        assert!(
            UPPER_FIT_ALT_HI < boundary,
            "the fit domain reaches {UPPER_FIT_ALT_HI} km but the walk stops using one \
             Boole panel at {boundary} km. Every altitude between them is fitted \
             against a quadrature the kernel does not perform."
        );
        // The other direction — that the domain ADMITS the whole censused
        // production band — is a claim about two constants, so it is a
        // compile-time assertion beside them rather than a runtime one here.
        // And `n == 1` really does hold across the whole domain, checked
        // against `jb_step_count` itself rather than against the exponential.
        for step in 0..=100 {
            let altitude_km = UPPER_FIT_ALT_LO
                + f64::from(step) * ((UPPER_FIT_ALT_HI - UPPER_FIT_ALT_LO) / 100.0);
            let al = (altitude_km.max(500.0) / plan.z).ln().max(0.0);
            let n = jb_step_count(al / LogQuadratureFittedV7::UPPER_LOG_STEP)
                .expect("the step count must resolve across the fit domain");
            assert_eq!(n, 1, "step count is {n} at {altitude_km} km, not 1");
        }
    }

    /// The degree-8 rung of R28's fit ladder, one rung below what shipped.
    ///
    /// Identical to [`LogQuadratureFittedV7`] in every respect except the
    /// coefficients — same quadrature, same domain guard, same fallback — so a
    /// comparison against it isolates the fit degree and nothing else.
    struct FittedV7Degree8;
    impl Sealed for FittedV7Degree8 {}
    impl QuadratureProfile for FittedV7Degree8 {
        const LOWER_LOG_STEP: f64 = LogQuadratureX4ApproxV2::LOWER_LOG_STEP;
        const MIDDLE_LOG_STEP: f64 = LogQuadratureX4ApproxV2::MIDDLE_LOG_STEP;
        const UPPER_LOG_STEP: f64 = LogQuadratureX4ApproxV2::UPPER_LOG_STEP;
        const USE_FIXED_LOWER_PLAN: bool = true;
        const RETIRE_SPECIES_ROUND_TRIP: bool = true;
        const USE_FIXED_MIDDLE_PLAN: bool = true;
        const RETIRE_ZR_ROUND_TRIP: bool = LogQuadratureFittedV7::RETIRE_ZR_ROUND_TRIP;
        const DLRSL_ZERO_ABOVE_KM: f64 = LogQuadratureFittedV7::DLRSL_ZERO_ABOVE_KM;
        const FITTED_UPPER_SEGMENT: bool = LogQuadratureFittedV7::FITTED_UPPER_SEGMENT;

        fn fixed_lower_state(tc: TemperatureBroadcast, ain: f64) -> LowerState {
            const SUB2_L: [f64; 9] = [
                2.083_081_422_374_930_4e1,
                -2.280_848_471_212_578_2e-1,
                1.487_286_042_321_815e-1,
                -1.145_838_723_584_677_6e-1,
                6.773_364_726_813_635e-2,
                -3.049_718_125_595_643_5e-2,
                1.292_621_900_483_987e-2,
                -9.233_684_439_422_045e-3,
                4.176_788_443_384_037e-3,
            ];
            const TLOC2: [f64; 9] = [
                2.245_189_251_097_160_4e2,
                8.361_677_463_493_898e0,
                -5.270_780_666_943_306e0,
                3.943_479_204_936_285_7e0,
                -2.210_169_281_028_411e0,
                9.772_329_396_258_238e-1,
                -3.663_280_231_187_389_7e-1,
                1.373_755_380_764_138_4e-1,
                -3.792_436_978_916_864e-2,
            ];
            let plan = logquad_x4_fixed_lower_plan();
            let texo = fitted_v7_texo_of(tc);
            if !(FITTED_V7_TEXO_LO..=FITTED_V7_TEXO_HI).contains(&texo) {
                return fixed_lower_state(plan, tc, ain);
            }
            let u = fitted_v7_u_of(texo);
            LowerState {
                sub2: poison_horner(&SUB2_L, u),
                z: plan.z,
                zend: plan.zend,
                mb2: plan.mb2,
                tloc2: poison_horner(&TLOC2, u),
                gravl: plan.gravl,
            }
        }

        fn fixed_middle_state(tc: TemperatureBroadcast, ain: f64) -> MiddleState {
            const SUB2_M: [f64; 9] = [
                3.204_272_924_503_414e0,
                -1.372_776_745_140_715e0,
                9.586_875_737_482_978e-1,
                -7.745_626_688_996_936e-1,
                5.821_323_462_363_506e-1,
                -1.935_159_698_966_887_3e-2,
                -7.984_200_971_457_091e-2,
                -4.776_796_502_700_586_5e-1,
                3.717_067_966_914_393e-1,
            ];
            const AIN_M: [f64; 9] = [
                5.460_318_995_199_056_4e-3,
                -3.661_545_490_791_822e-3,
                2.457_437_241_815_96e-3,
                -1.906_983_059_765_339_7e-3,
                1.425_867_562_307_701e-3,
                -2.645_635_141_308_472_6e-5,
                -2.137_980_604_932_346_2e-4,
                -1.196_555_068_369_199_2e-3,
                9.346_178_699_032_139e-4,
            ];
            const TLOC3: [f64; 9] = [
                1.543_797_141_046_925_8e3,
                1.039_876_923_210_337_3e3,
                -4.373_919_913_022_934_5e0,
                1.148_042_825_822_828_5e-1,
                1.554_907_622_501_655_3e-1,
                -6.274_597_318_673_697e-2,
                7.197_570_380_869_358e-3,
                2.973_695_228_977_641_7e-3,
                -9.615_782_423_993_217e-4,
            ];
            let plan = logquad_x4_v2_fixed_middle_plan();
            let texo = fitted_v7_texo_of(tc);
            if !(FITTED_V7_TEXO_LO..=FITTED_V7_TEXO_HI).contains(&texo) {
                return fixed_middle_state(plan, tc, ain);
            }
            let u = fitted_v7_u_of(texo);
            MiddleState {
                sub2: poison_horner(&SUB2_M, u),
                ain: poison_horner(&AIN_M, u),
                tloc3: poison_horner(&TLOC3, u),
                z: plan.z,
                zend: plan.zend,
            }
        }
    }

    /// The 1.0e-4 bound has teeth against the FIT DEGREE, not just the quadrature.
    ///
    /// `v7_broad_grid_density_error_stays_within_rescoped_bound` asserts model 7
    /// passes at 1.0e-4. Model 6's poison (`middle 0.400`) cannot support that
    /// gate, because it perturbs the quadrature and model 7 shares model 6's
    /// quadrature exactly — it would prove the bound rejects a profile model 7
    /// is not a variation of. The degree-8 rung is the right poison: it differs
    /// from the shipped profile in the coefficients and in nothing else, so this
    /// is the test that says the LADDER RUNG was chosen rather than assumed.
    ///
    /// Degree 8 is one rung down, not a straw man. Its worst fitted scalar is
    /// 1.653e-3 against degree 14's 7.434e-6.
    #[test]
    fn the_density_bound_rejects_the_degree_8_fit() {
        const RESCOPED_BOUND: f64 = 1.0e-4;

        let mut worst = 0.0_f64;
        let mut shipped_worst = 0.0_f64;
        for altitude_km in [500.0, 626.0, 800.0, 986.0, 2000.0, 35_000.0] {
            for latitude_deg in [-45.0, 0.0, 45.0] {
                let input = r16_broad_grid_input(57_000.25, altitude_km, latitude_deg, 12.0);
                let exact = jb2008_density(input).expect("exact density");
                let poisoned = jb2008_density_with_profile::<FittedV7Degree8>(input)
                    .expect("the poison rung must still evaluate");
                let shipped = jb2008_density_with_profile::<LogQuadratureFittedV7>(input)
                    .expect("the shipped rung must still evaluate");
                worst = worst.max((poisoned - exact).abs() / exact);
                shipped_worst = shipped_worst.max((shipped - exact).abs() / exact);
            }
        }

        println!(
            "R31_POISON degree_8 max_relative_error={worst:e} shipped_degree_14={shipped_worst:e} \
             bound={RESCOPED_BOUND:e}"
        );
        assert!(
            worst > RESCOPED_BOUND,
            "the 1.0e-4 bound does not reject the degree-8 fit ({worst:e}); it is not bounding \
             the fit degree and the model 7 gate is vacuous with respect to the coefficients"
        );
        // The other direction: the same rows, the same bound, the shipped
        // degree. Without this the test above is satisfied by a bound that
        // rejects EVERYTHING, including what shipped.
        assert!(
            shipped_worst <= RESCOPED_BOUND,
            "the shipped degree-14 fit fails the bound on the poison's own rows \
             ({shipped_worst:e}); the poison proves nothing about degree"
        );
    }

    /// Why `middle 0.150` REJECTS every input at exactly 500 km.
    ///
    /// The middle segment integrates to `altitude_km.min(500.0)`, and the
    /// segment above it starts from the abscissa the middle walk ENDED on, not
    /// from the literal 500.0. Those two agree only to rounding: the exit is
    /// reached by `n` steps of `z += dz` and lands a few ULP either side of 500
    /// depending on `n`. At exactly 500 km the upper segment then evaluates
    /// `ln(500 / exit)`, which is NEGATIVE whenever the exit landed high, and
    /// `jb_step_count` rejects a negative log interval.
    ///
    /// This prints the exit abscissa and that logarithm's sign for each rung,
    /// so the rejection is attributed rather than guessed at. Production never
    /// evaluates at exactly 500 km --- the sealed arc spans 626--986 km --- but
    /// the broad-grid receipt does, so any landed rung has to be checked here.
    #[test]
    fn r16_five_hundred_km_boundary_by_step_count() {
        let tc = [183.0, 7.303_974_2e-4, 100.0, 0.02];
        let temperature = TemperatureBroadcast::new(tc);
        for (label, n) in [
            ("0.100 -> 16", 16u32),
            ("0.125 -> 13", 13),
            ("0.150 -> 11", 11),
            ("0.200 ->  8", 8),
            ("0.300 ->  6", 6),
        ] {
            // The lower plan's exit is the middle segment's entry, exactly as
            // `jb_density` hands it over.
            let lower = fixed_lower_state(logquad_x4_fixed_lower_plan(), temperature, 1.0);
            let al = (500.0_f64 / lower.z).ln();
            let zr = (al / f64::from(n)).exp();
            let mut z = 0.0;
            let mut zend = lower.zend;
            for _ in 0..n {
                z = zend;
                zend = zr * z;
                let dz = 0.25 * (zend - z);
                let _ = boole_abscissae(&mut z, dz);
            }
            let upper_log = (500.0_f64 / z).ln();
            println!(
                "R16_BOUNDARY middle_steps={n:>2} ({label})  exit_z={z:.17} \
                 exit_minus_500={:.3e}  ln(500/exit)={upper_log:.3e}  step_count={:?}",
                z - 500.0,
                jb_step_count(upper_log / 0.100)
            );
        }
    }

    /// Where the shipped abscissae come from, segment by segment.
    #[test]
    fn r16_abscissa_budget_by_segment() {
        for altitude_km in [626.2, 700.0, 800.0, 900.0, 985.7] {
            let lower = jb_step_count((105.0_f64 / 90.0).ln() / 0.040).expect("lower");
            let al_middle = (500.0_f64 / 105.0).ln();
            let middle = jb_step_count(al_middle / 0.100).expect("middle");
            let al_upper = (altitude_km / 500.0_f64).ln();
            let upper = jb_step_count(al_upper / 0.300).expect("upper");
            let (total, with_atan, atan_calls) =
                r16_abscissa_count(altitude_km, 0.040, 0.100, 0.300);
            println!(
                "R16_BUDGET alt={altitude_km:>6.1} km  steps lower={lower} middle={middle} \
                 upper={upper}  abscissae={total} atan_abscissae={with_atan} \
                 atan_x4_calls={atan_calls}"
            );
        }
    }
}
