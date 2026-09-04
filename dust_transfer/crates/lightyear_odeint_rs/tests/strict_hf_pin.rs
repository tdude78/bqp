//! The gate's only pin on a strict-HF propagation under the PRODUCTION force
//! model.
//!
//! # IT RUNS IN RELEASE ONLY, AND THAT IS A COVERAGE GAP TO KNOW ABOUT
//!
//! This header used to read "Not `#[ignore]`d: it must fail CI." That is no
//! longer true and the difference matters. `strict_hf_v3_production_arc_is_pinned`
//! carries `#[cfg_attr(not(feature = "bitpin"), ignore)]`, because `fp-contract`
//! makes a debug build integrate a DIFFERENT TRAJECTORY — so in debug the pin
//! fails for a reason that has nothing to do with a regression, and before it
//! was gated it aborted the whole workspace run and hid ~1600 tests behind it.
//! (The gate used to key on `debug_assertions`, but `--profile fast-test` also
//! sets debug-assertions=false with non-release codegen, so the pin ran there
//! under the wrong flags; the explicit `bitpin` feature names the one lane —
//! `--release --features bitpin` — whose codegen the baseline was captured on.)
//!
//! The consequence, stated plainly so nobody has to rediscover it: **a debug
//! `cargo test --workspace` gives ZERO strict-HF pin protection.** The blindness
//! this file was written to end is still present in exactly that configuration.
//! The pin only guards what it claims to guard when run as:
//!
//! ```sh
//! cargo test --release --features bitpin -p lightyear_odeint_rs --test strict_hf_pin
//! ```
//!
//! Anyone changing strict-HF propagation and running only a debug gate will see
//! it pass. That is the same failure mode recorded below — a gate reporting
//! success while seeing nothing — reintroduced by build profile instead of by
//! missing coverage.
//!
//! # Why this file exists
//!
//! Before it, no test in the gate could observe a change to strict-HF
//! propagation output. That was established by measurement, not by reading:
//! carrying the step size across Encke rectification restarts moved the
//! one-event harness mass from `0x4022ae7adfec5f8a` to `0x4022ae47c0f5fa0f`
//! and cut right-hand-side evaluations 22.3%, and the full workspace gate
//! reported 2147 passed / 0 failed. It saw nothing.
//!
//! The three candidates that look like they cover this do not:
//!
//! - `stage3_rust_ric_cov_yields_byte_exact_masses` replays captured
//!   `v_hf_means` and `det_masses` out of a JSON fixture into the Pc batch. It
//!   never integrates anything.
//! - `stage1_transfer_front_matches_oracle_all_candidates_dv_for_byte_exact_event`
//!   is the medium-fidelity lane.
//! - The strict-HF unit tests in `dust_estimates_rs::mass_solver` do integrate,
//!   but at `sph_order: 0` with `force_flags: 0` — two-body only, where the
//!   Encke deviation never reaches the rectification threshold — and they
//!   assert status codes and call counts rather than values.
//! - The parallel-vs-serial `to_bits()` comparisons are self-referential: both
//!   sides move together, so a global change passes them unchanged.
//!
//! # FIVE TESTS, THREE NUMERICAL JOBS, TWO ATMOSPHERE MODELS
//!
//! | test | model | question it answers |
//! |---|---|---|
//! | `strict_hf_v3_production_arc_is_pinned` | 7 | did the derivative move? |
//! | `strict_hf_v3_production_arc_accuracy` | 7 | NOT what its name suggests -- see below |
//! | `strict_hf_production_arc_accuracy` | 4 | NOT what its name suggests -- see below |
//!
//! Two additional tests pin V3 sizing and the production baseline-cache hit
//! rate. Compiled Part A authority moved 4 -> 5 at `2df59d4`, 5 -> 6 at
//! `0dad3d0`, 6 -> 7 at `690b416`, and now selects model 8. Compiled Part A
//! authority alone decides the flown identity. The fitted v7 kernel remains a
//! component beneath model 8, but does not itself pin persistence-driver
//! authority. Model 4 is the exact JB2008 profile, kept because it is bound to
//! a sealed Orekit fixture and because `rect_loop_pin`'s digests and
//! `prop_timing`'s historical numbers are measured on it. Earlier fitted models
//! remain comparison profiles.
//!
//! # NEITHER `_accuracy` TEST BOUNDS THE ATMOSPHERE MODEL
//!
//! Read this before citing either one for a force-model change. Both propagate
//! the arc at production `eps` and again at `REFERENCE_EPS` and difference the
//! two endpoints. **Both sides use the same force model**, so a bias in the
//! physics is common-mode and cancels exactly; what survives is integrator
//! truncation. They answer "is the integration delivering what its tolerance
//! asks for", which is a real question and not this one.
//!
//! Measured on 2026-08-09 against the JB2008 abscissa ladder: both stay GREEN
//! at `middle 0.400`, a quadrature whose density error is 2.829e-3, i.e. 28x
//! over the bound that authorizes model 6 and 943x over model 5's. Worse, the
//! V3 gate read 0.016336 m at model 5 and 0.001128 m at model 6 -- 14x
//! "better" for a profile 36x less accurate, because the coarser grid nudged
//! the trajectory onto a path the controller resolves more tightly. The metric
//! is anti-correlated with density accuracy on that sample, not merely blind.
//!
//! The gate that DOES bound the atmosphere is
//! `v2_broad_grid_density_error_stays_within_rescoped_bound` in `jb_rs`, with
//! `the_rescoped_bound_rejects_the_rung_the_accuracy_gates_wave_through` as its
//! poison proof. See `docs/plans/2026-08-08-r16-atan-abscissa-decision.md`.
//!
//! The third row is a legacy name and NOT a production gate despite reading
//! like one. Until 2026-08-07 it was the only accuracy gate here, its config
//! builder was called `production_dust_config`, and the flying approximation's
//! accuracy was therefore gated by nothing at all. The builder is now
//! `exact_profile_dust_config`; the test name is unchanged for the reason its
//! own doc comment gives.
//!
//! **`strict_hf_v3_production_arc_is_pinned` is a REPRODUCIBILITY TRIPWIRE, not
//! an accuracy gate.** Any change to the derivative trips it. That is correct
//! behaviour and expected, not a defect. It CANNOT tell you whether accuracy
//! degraded and must never be read as if it can. Re-baselining it after an
//! intended change is routine.
//!
//! An earlier version of this file claimed its 1 cm bound sat "far above the
//! micrometre-scale ULP noise a 12 h arc accumulates". **That was false and
//! never measured.** Perturbing each element of the initial state by ONE ULP
//! and re-propagating moves this arc's endpoint by 0.028, 0.539, 0.054, 0.126,
//! 0.470 and 0.972 m -- the fast angle worst. The bound is ~100x BELOW the
//! noise floor of the quantity it bounds, which is exactly why it trips on
//! everything.
//!
//! The worked example: the `geodetic_altitude_km` algebra patch is 2.7
//! NANOMETRES at the function over 200,000 positions, and it moved this
//! endpoint 0.205 m and forced a re-baseline. That is the tripwire behaving
//! correctly.
//!
//! **The ~1 m figure in that paragraph is an ERA number, correct when written
//! and superseded since.** Both floors were re-measured at `ba6a249` by
//! `examples/v3_accuracy_floor.rs`, and the era that recorded them was
//! re-measured directly at `6a856aa` with the same program. See
//! `ACCURACY_METRIC_FLOOR_M` for the numbers and the attribution; the short
//! version is that the arc's own truncation error fell 26.6x and the floor fell
//! 25.7x with it, so the floor did not "stop reproducing", it moved with the
//! accuracy it is a floor on.
//!
//! **The step-count hypothesis this file used to carry is REFUTED.** It held
//! that the floor was chaos amplification over the step sequence and that the
//! arc "fell from 3,844 steps to 471". The 3,844 belongs to a transient tree on
//! 2026-08-02, a week AFTER the 1.035 m was recorded, and is not the 1.035 m
//! era at all. Measured on the tree that recorded it, `6a856aa` runs this arc
//! in 359 steps and 64 segments; the tip runs it in 461 steps and 67 segments.
//! Both counters are essentially unchanged, and steps went UP.
//!
//! Note also that "one ULP on each of the six" is not six equal perturbations.
//! The elements span five orders of magnitude, so one ULP of element 0
//! (7178.137 km) is 9.1e-13 absolute against 3.5e-18 for element 2 (0.0227,
//! dimensionless). At the tip that leaves element 2's production trajectory BIT
//! IDENTICAL, so the axis is dead and the sample is effectively five. The
//! harness now CALIBRATES each element to the smallest ULP count that moves the
//! endpoint's bits, and fails rather than reporting a floor with a dead axis.
//! This was a tip-only defect: at `6a856aa` every element moved the trajectory
//! at one ULP, so it is not part of why the recorded number differs.
//!
//! Consequence for the tripwire, not acted on here: its 1 cm bound sits ~19x
//! below the floor rather than the ~100x this header used to claim. The
//! direction is unchanged and so is the conclusion -- it still trips on every
//! derivative change, which is what it is for.
//!
//! **The two `_accuracy` tests are the accuracy gates.** Each compares the
//! production tolerance against a converged `eps = 1e-12` reference in the same
//! run. A previous run is not more accurate, only previous, so only a reference
//! can answer "did accuracy degrade".
//!
//! The log entries below were all written when there was one accuracy gate, so
//! where they say `strict_hf_production_arc_accuracy` they mean the model-4
//! reading, and none of them constrains the model-5 arc.
//!
//! **Every entry below dated before 2026-08-07 states its margin against the
//! then-recorded 1.035 m floor.** Those margins are left as written, because
//! they record what was believed when the entry was made and editing them would
//! turn a dated record into a fabricated one. To read one against the floor
//! measured since, divide its stated ratio by 5.2 -- "20x below the floor"
//! becomes 3.9x below. Every such claim survives the correction in direction;
//! none of them survives being re-quoted as a ratio.
//!
//! # Why the bits are not pinned here
//!
//! Raw endpoint bits are per-libm, not per-arch: Apple libm and glibc disagree
//! 1-2 ULP on every transcendental, and one ULP is worth up to a metre on this
//! arc. A bit pin here would be a Mac-only assertion failing on the production
//! architecture for reasons unrelated to any change. The exact-bit pin lives on
//! the exactness authority, in the `#[ignore]`d one-event harness.

use anyhow::Context;
use num_traits::ToPrimitive;
use std::sync::Arc;

use lightyear_odeint_rs::integrator::{
    integrate_final_checked, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest,
};
use lightyear_odeint_rs::probe;
use lightyear_odeint_rs::types::StepperMethod;
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags};
use satpy_core::{eci2equinoc_impl_f64, equinoc2eci_impl, kep2eci_impl, SEC_PER_DAY};

const JD0: f64 = 2_460_310.5;

/// Epoch of the V3 arc, inside the JB2008 v3 persistence window.
///
/// `JD0` is 2024-01-13 and the Part A v3 authorized persistence arc runs
/// 2026-08-15 to 2026-08-31, per `scenario.authorized_start_utc` /
/// `authorized_end_utc` in
/// `assets/reference/atmosphere/jb2008/part_a_v3_persistence_v1/manifest.json`.
/// Every V3 test therefore failed with "lookup outside
/// Part A v3 authorized persistence arc" before reaching any physics -- the
/// same stale-fixture-epoch class as `b3d37d79`.
///
/// This value is the manifest's own `scenario.t0_utc`, 2026-08-17T17:24:29Z,
/// which is also `common_anchor_jd_utc` in the sealed `search_b500_v3.json`.
/// It is deliberately NOT `JD0 + something`: the V1 pin flies a different arc
/// and must keep its own epoch, or re-pinning one would silently re-pin both.
const V3_JD0: f64 = 2_461_270.225_335_648_3;
const DIR_R6_D15: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

/// 12 h. Long enough that Encke rectification restarts the solver tens of
/// times -- which is the regime the counters exist to watch -- and short enough
/// that trajectory divergence does not swamp the position tolerance.
const TOF_S: f64 = 43_200.0;

// RE-PINNED from 6_900 / 357 / [-215.980542903540396, 1883.16104124726326,
// 7043.91584043725470]. The endpoint moved 0.189 m, against this arc's ~1 m
// single-ULP noise floor.
//
// Cause: `jb_local_temp_x4`. JB2008's Boole quadrature evaluates the
// temperature at four independent abscissae per step and called libm `atan`
// once per abscissa -- 224-260 times per RHS evaluation, measured at 42% of
// this arc's leaf samples. Those four are now one `f64x4`, whose `atan` is the
// Cephes rational at 1.937 ULP where libm is correctly rounded. The abscissae
// and the weighted sum are UNCHANGED to the bit; only the temperature moves.
//
// `rhs_evals` moved 6_900 -> 7_142 on this arc, which is +3.5%, and that is
// NOT a cost of the change: over the 12-arc `tolerance_cost_accuracy` corpus
// at the same tolerance the count moved 75_113 -> 74_982, i.e. -0.17%. One arc
// cannot resolve a step-controller perturbation this small.
//
// Pinned on the exactness authority (Apple arm64), 2026-07-26, at
// perf/rhs-hot-loop 56e80fe.
//
// RE-PINNED from 7_799 / 359 / [-215.980395778327221, 1883.16084052815518,
// 7043.91587586438618], which were set at 79960bb. The endpoint moved 0.054 m.
//
// Cause: the packed-gravity `dense_prefix` fix, which lets the `dense_low_order`
// SIMD specialization run for the first time. That specialization accumulates
// into four f64 lanes and reduces at the end, where the path it replaced summed
// sequentially -- so the harmonic sum is reassociated, and nothing else changes.
//
// The accuracy cost of the reassociation was bounded at the function level, not
// inferred from this arc: over 200k random LEO positions, old vs new
// acceleration differ by at most 5.82e-18 km/s^2 (5.8e-15 m/s^2), 6.7e-16
// relative, bitwise identical on 25,819 of them. See
// `satpy_core/tests/gravity_dense_prefix_accuracy.rs`.
//
// 0.054 m is not that cost. It is this arc's step-sequence response to a
// few-ULP perturbation of the derivative -- the same sensitivity recorded
// elsewhere on this branch, where a deliberate 1-ULP change moved a campaign
// endpoint 0.26 m. The counters moved with it (rhs_evals +1.6%, steps +3.3%,
// both inside the 5% band), which is what a re-timed step sequence looks like
// and not what a wrong derivative looks like.
//
// Verified attributable: at this same commit with the fix reverted, this test
// reproduced the old constants exactly, so none of the 0.054 m is drift from
// the merges that landed since 79960bb.
// RE-PINNED 2026-07-27, authorized, for PERTURB_DEVIATION_THRESHOLD_KM 2.0 -> 10.0
// km. Copied verbatim from this test's own STRICT_HF_PIN line, not re-derived.
//
//   rhs_evals  7923 -> 7434  (-6.2%)
//   steps       371 ->  362  (-2.4%)
//   rejected    101 ->   80  (-21%)
//   segments     64 ->   20  (-69%)
//   endpoint moved 0.651111 m
//
// Encke-only: this test's `production_dust_config` hardcodes `atm_model: 4` and
// does not read compiled science, so the simultaneous atmosphere_model 4 -> 5
// change cannot reach it. Confirmed by running the pin on both sides of that
// edit and getting a byte-identical line.
//
// Segments fell 69% but evaluations only 6.2%, which is not a contradiction:
// `MAX_RECT_SEGMENT = 5400 s` floors this 43200 s arc at 8 segments, so
// deviation-triggered restarts went 56 -> 12, and a restart is one cheap
// reference re-anchor that does not proportionally reduce RHS work.
//
// Accuracy did NOT degrade: `strict_hf_production_arc_accuracy` stayed green and
// moved 0.323 -> 0.328 m against its 3.0 m gate.
// RE-PINNED 2026-07-27, authorized, for the `k[0]` reuse in
// `lightyear_odeint_rs::odesolve::solver`. Copied verbatim from this test's own
// STRICT_HF_PIN line, not re-derived.
//
//   rhs_evals  7434 -> 6900  (-7.2%)
//   steps       362 ->  357  (-1.4%)
//   rejected     80 ->   78
//   segments     20 ->   18
//   endpoint moved 0.346398 m
//
// `rhs_evals` is OUTSIDE the 5% `COUNT_BAND`, so this constant had to move; the
// -1.4% on `steps` alone would have passed unnoticed.
//
// The saving is redundant work, not a step-size change: 342 of the removed
// evaluations are the event handler's end-of-step derivative, which is the next
// step's `k[0]` at the identical `(t, y)`, and 80 are rejected retries
// re-deriving a `k[0]` that does not depend on `h`.
//
// Why this moves the endpoint AT ALL, given it computes the same quantities:
// `LightyearRHS::compute_internal` is NOT a pure function of `(t, y)`. The Encke
// baseline is memoized on a TIME TOLERANCE (`rhs.rs:2866`), so the derivative
// depends on which nearby `t` last warmed the cache -- 24 element mismatches up
// to ~180 ULP measured at an identical `(t, y)`. Deleting a redundant call
// changes the warming sequence, and this arc turns ULPs into metres.
//
// Accuracy did NOT degrade and did NOT measurably improve: on the 12-arc
// `tolerance_cost_accuracy` corpus at production eps, p50 0.0655 -> 0.0423 m and
// max 1.028 -> 0.611 m, both moves at or below the per-arc `TOL_REFDRIFT` floor
// of 9e-7..0.187 m. `strict_hf_production_arc_accuracy` reads 0.328 -> 0.017 m
// against its 3.0 m gate, also below its own 1.035 m noise floor.
// NOT RE-PINNED. Logged 2026-08-02 for a displacement that was never recorded
// when it happened, across a6b4de4 `fix(physics): commit eclipse roots
// continuously` and its companion 601bf6e `test(physics): bind eclipse
// transaction cost`, which also touches the coordinator and the integrator.
// The counters were not bisected between the two, so the deltas below belong
// to the pair. The constants are unchanged and still read what 56e80fe set, so
// this entry explains a RED tripwire rather than retiring one.
//
//   rhs_evals  62_160 -> 14_082  (-77%)
//   steps       3_844 ->    861  (-78%)
//   rejected       11 ->     16
//   segments      491 ->     66  (-87%)
//   endpoint displacement 0.195213 -> 0.127388 m
//
// Cause, offered as a HYPOTHESIS and not established by a revert: a6b4de4
// replaced the eclipse coordinator's re-detection replay path with a single
// continuous fine-origin root transaction. The old path re-detected and
// re-integrated each crossing, so `segments` counted re-detections rather than
// physics, and the counters above are the shape of that duplicated work
// disappearing.
//
// The counter assertions never saw any of it. They are SECONDARY and sit below
// the endpoint assertion, which is PRIMARY and fires first, so once the
// endpoint moved the 77% and 78% count breaches became unreachable. Nothing
// that size could have passed the 5% `COUNT_BAND` had it been reached.
//
// The 0.195213 m this arc already carried BEFORE a6b4de4 is separately
// unattributed. The pin was stale when that change landed, so neither number
// is a clean single-change delta from 56e80fe, and re-pinning to the current
// line would silently bless both displacements at once.
//
// The post-a6b4de4 signature is stable across the eclipse work that followed.
// Measured byte-identical at 9aef0db (post-root scan certification) and again
// at f485c14 (origin crossings committed without a proof leg): 14_082 / 861 /
// 16 / 66 and 0.127388 m at both, so neither of those fixes moved physics on
// this arc.
//
// Accuracy did NOT degrade: `strict_hf_production_arc_accuracy` reads
// 0.048677 m against its 3.0 m gate, below its own 1.035 m noise floor.
//
// NOT RE-PINNED. Logged 2026-08-03 for the DELIBERATE clamp-recovery
// displacement (perf fix, authorized under the deadline campaign):
// `MAX_ROOT_REFINEMENT_STEP_S` raised 2.0 -> 10.0 s globally. The sweep on
// this arc (2/4/5/10/60 s -> 861/752/672/570/Err(Bracket) steps) showed
// accuracy improving through 10 s with the 0.10 m root budget enforced
// independently and fail-closed by `replay_root_uncertainty_km`.
//
// A structural alternative (fine clamp only on the straddle/window legs,
// production dt_max on proof and approach) posted similar counts but FAILED
// the release B500 event-0 gate with a deterministic EclipseBracket: the
// clamp is a detector obligation on every root-transaction leg, not a
// straddle-interval one. That change was reverted before any freeze.
//
//   rhs_evals  14_082 -> 9_339  (-34%)
//   steps         861 ->   570  (-34%)
//   rejected       16 ->    10
//   segments       66 ->    69
//   endpoint displacement 0.127388 -> 0.163000 m (vs the stale 56e80fe pin;
//   still ~6x below this arc's ~1 m single-ULP noise floor)
//
// Accuracy arm remains green. The V1 tripwire was DELETED (census B): it was a
// permanent red pinning atm_model 4, which nothing flies since production read
// atmosphere_model 5 at 2df59d4, and a gate nobody can run green is a gate
// nobody reads. Its replacement is the sealed `STRICT_HF_V3_PIN`, measured only
// after this clamp recovery and the campaign solver freeze.
/// Secondary net, deliberately loose. Counters exist to catch a change that
/// alters the work WITHOUT moving the answer -- pure-waste removal, or a
/// reordering that cancels. They are not the primary detector, and sizing them
/// as if they were is a trap: on this arc the `h`-carry moved evaluations by
/// -1.99%, which a 2% band would have passed. Position caught it by 118x.
///
/// 5% is wide enough that a libm disagreement flipping accept/reject decisions
/// cannot false-fire it across platforms.
const COUNT_BAND: f64 = 0.05;

/// The compiled Part A science authority, read rather than restated.
///
/// These values used to be literals here. A literal copy of a sealed constant
/// is a READER, not an asserter: it keeps passing while integrating a
/// trajectory the campaign does not fly, and says nothing when the authority
/// moves underneath it. That is exactly how `atm_model` came to be stale.
const fn part_a_hybrid() -> &'static nd_config::PartAHybridControls {
    nd_config::CompiledPartAScienceV1::part_a_v1().hybrid()
}

/// The EXACT JB2008 profile, `atm_model: 4`. **This is not production.**
///
/// It was called `production_dust_config` until 2026-08-07 and the name was the
/// defect: it reads six fields off the sealed authority and then hardcodes the
/// seventh, so it looks like the campaign configuration and is not one. Two
/// instruments were built on it in that belief -- this file's accuracy gate and
/// `prop_timing`'s whole timing harness -- and a round was spent arbitrating a
/// timing disagreement that was only ever this line. Renamed rather than
/// re-documented because the loud warning `prop_timing` already carried did not
/// stop it being read as production.
///
/// Kept, not deleted. `strict_hf_exact_profile_arc_accuracy` is the only
/// accuracy gate on the exact profile, which is the one that has to reproduce a
/// sealed Orekit fixture bit for bit, and `rect_loop_pin`'s three digests are
/// measured on it.
/// The compiled stepper, resolved rather than restated.
///
/// This used to be an `assert_eq!(controls.integrator_method, "vern9")` next to
/// a hardcoded `StepperMethod::Vern9`. The assert was a real tripwire -- it
/// fired the moment compiled science moved to Vern7 -- but it could only ever
/// say "stop", and the literal beside it meant that relaxing the assert without
/// also editing the literal would leave this file integrating a trajectory the
/// campaign no longer flies, silently and green. Resolving the token instead
/// removes the second step, and the `panic!` keeps the fail-closed property for
/// a stepper this file genuinely cannot build.
#[expect(
    clippy::panic,
    reason = "a stepper this file cannot build must stop the run, not silently measure a different one"
)]
fn authority_stepper() -> StepperMethod {
    match part_a_hybrid().integrator_method {
        "vern7" => StepperMethod::Vern7,
        "vern9" => StepperMethod::Vern9,
        other => panic!("compiled science selects a stepper this file does not build: {other}"),
    }
}

fn exact_profile_dust_config() -> ForceConfig {
    let controls = part_a_hybrid();
    ForceConfig {
        sph_order: controls.gravity_order,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        subtract_first_order: true,
        // NOT `controls.atmosphere_model`, and deliberately so. This is the
        // exact profile, whose log entries below are all measured at
        // exact-JB2008. The divergence from the authority is the point -- see
        // `v3_frozen_config`, which is the arc that tracks it.
        //
        // This comment used to name production's atmosphere model as well
        // ("has read `atmosphere_model: 5` since 2df59d4"). That was true when
        // written and rotted at the next reseal, then at the one after it.
        //
        // Deliberately not replaced with the current value: a literal here has
        // no reader that needs it and no gate that checks it, so it decays
        // silently and the next reader believes it. This arm is 4 because the
        // digests below were measured at 4 -- that is the whole reason, and it
        // is a fact about this file rather than about the seal. If you need
        // what production reads, read `nd_config::part_a_science`, which is
        // the only copy that cannot go stale.
        atm_model: 4,
        am_ratio: controls.dust_am_ratio,
        cd: controls.dust_cd,
        cr: controls.dust_cr,
        target_propagation_mode: 0,
        dt_max: controls.dt_max_s,
        eps: controls.tolerance,
        integrator_method: authority_stepper(),
        ..ForceConfig::default()
    }
}

/// The integration tolerance the campaign compiles, not a copy of it.
///
/// `nd_config/src/part_a_science.rs` `tolerance` is the authority. Binding it
/// here is what makes the pin notice a tolerance change: with the literal in
/// place, editing the sealed value moved production and left this gate green.
const fn production_eps() -> f64 {
    part_a_hybrid().tolerance
}

const REFERENCE_EPS: f64 = 1.0e-12;

/// Probe counters are process-global and Cargo runs a binary's tests on
/// parallel threads, so two tests calling `probe::reset()` clobber each
/// other's census. Not hypothetical: it read 113 evaluations against a pinned
/// 7,799 the first time the accuracy gate was added.
static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn propagate_exact_at(eps: f64) -> anyhow::Result<([f64; 6], probe::TagCensus)> {
    propagate_config_at_epoch(exact_profile_dust_config(), eps, JD0)
}

fn propagate_v3_at(eps: f64) -> anyhow::Result<([f64; 6], probe::TagCensus)> {
    propagate_config_at_epoch(v3_frozen_config(), eps, V3_JD0)
}

fn propagate_config_at_epoch(
    mut base_config: ForceConfig,
    eps: f64,
    jd0: f64,
) -> anyhow::Result<([f64; 6], probe::TagCensus)> {
    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("gravity coefficients must load")?;
    let packed =
        lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")?;

    let kep = [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0];
    let mut init_eci = [0.0; 6];
    kep2eci_impl(&kep, true, 0.0, 0.0, true, &mut init_eci);
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    base_config.eps = eps;
    let config = base_config
        .with_ephemeris_for_arc(jd0, jd0 + TOF_S / SEC_PER_DAY)
        .context("production ephemeris and JB2008 assets must cover the pinned arc")?;

    // These pins test direct arc physics. Reset the probe census immediately
    // before integration so RHS/step/segment non-vacuity belongs to this arc.
    probe::reset()?;
    let gravity = ScalarGravityAssets::new(packed);
    let context = ScalarPropagationContext::new(jd0, Arc::new(config), gravity);
    let delta = integrate_final_checked(
        ScalarPropagationRequest::new(&context, init_equ, &[TOF_S], 0.0, TOF_S).with_events(true),
    )
    .context("the pinned strict-HF arc must propagate")?;

    let mut base_at_tf = [0.0; 6];
    equinoc2eci_impl(&init_equ, 6, TOF_S, 0.0, &mut base_at_tf);
    let mut state = base_at_tf;
    for (state_component, delta_component) in state.iter_mut().zip(delta) {
        *state_component += delta_component;
    }

    let census = probe::snapshot();
    let total = census.iter().try_fold(
        probe::TagCensus::default(),
        |mut acc, entry| -> anyhow::Result<_> {
            acc.rhs_evals = acc
                .rhs_evals
                .checked_add(entry.rhs_evals)
                .context("strict-HF RHS-evaluation census overflow")?;
            acc.propagations = acc
                .propagations
                .checked_add(entry.propagations)
                .context("strict-HF propagation census overflow")?;
            acc.segments = acc
                .segments
                .checked_add(entry.segments)
                .context("strict-HF segment census overflow")?;
            acc.steps = acc
                .steps
                .checked_add(entry.steps)
                .context("strict-HF step census overflow")?;
            acc.rejected = acc
                .rejected
                .checked_add(entry.rejected)
                .context("strict-HF rejected-step census overflow")?;
            Ok(acc)
        },
    )?;
    Ok((state, total))
}

/// The V3 sealed authority, which is NOT what `exact_profile_dust_config`
/// builds.
///
/// That function hardcodes `atm_model: 4` and does not read compiled science
/// for that one field. It was accurate when written and is not any more:
/// compiled authority moved beyond `atmosphere_model: 4` at `2df59d4`. So the
/// V1 pin watched a trajectory the campaign stopped flying, which is the
/// coverage hole this pin closes rather than a defect in V1 -- V1's own log
/// entry records the model-4 hardcode as the reason an atmosphere change could
/// not reach it.
///
/// Everything else resolves from the compiled campaign authority or the
/// frozen exact-profile fixture: the authority-selected integrator, `eps
/// 1e-8`, `dt_max 300 s`, binary cylindrical shadow via the event-enabled
/// eclipse coordinator, and `MAX_ROOT_REFINEMENT_STEP_S = 10 s` from source.
fn v3_frozen_config() -> ForceConfig {
    ForceConfig {
        // Read, not restated: this arc's whole job is to watch the trajectory
        // the compiled authority selects. If `atmosphere_model` moves, this
        // pin must go red rather than keep flying the old model.
        atm_model: part_a_hybrid().atmosphere_model,
        ..exact_profile_dust_config()
    }
}

/// # These constants are APPLE-LIBM values and will not reproduce on glibc
///
/// The same caveat this file's header states for the V1 pin applies here and is
/// worth repeating at the constants rather than 400 lines above them: Apple
/// libm and glibc disagree 1-2 ULP on every transcendental, and one ULP is
/// worth up to a metre on this arc. `V3_POS_TOL_KM` is 1 cm. So a Linux CI
/// running this will fail it for reasons unrelated to any change, and needs its
/// own baseline measured on that libm -- not a loosened tolerance, which would
/// destroy the only property the pin has.
///
/// Measured on the exactness authority (Apple arm64), most recently 2026-08-06.
/// Copy replacements from the `STRICT_HF_V3_PIN` line the test prints; do not
/// re-derive them.
///
/// # Re-pinned 2026-08-07 — eclipse Sun-direction supremum
///
/// | | old | new |
/// |---|---|---|
/// | `rhs_evals` | 7,560 | **7,827** |
/// | `steps` | 458 | **471** |
/// | endpoint | `-215.98052835901174, 1883.1610220126431, 7043.91584385317` | see below |
///
/// **Cause:** `LightyearRHS::eclipse_sun_direction_path_bound`. The bound on how
/// far the Sun direction sweeps across an eclipse interval was a sum of exact
/// great-circle steps between crossed ephemeris nodes — two table lookups, two
/// normalizations and an `atan2` per call. It is now a per-grid supremum on the
/// sweep RATE times the elapsed time: one multiply. The replacement is a valid
/// upper bound but a looser one, and the eclipse scan subdivides against it, so
/// the evaluation points move.
///
/// **The eval count went UP, and that is not a regression.** It is a reshuffle:
/// a differently-bracketed scan takes 13 more accepted steps and 4 more
/// rejects. The direction is not even stable — against the PREVIOUS baseline
/// this same change moved evals DOWN 92. Neither number is the saving and
/// neither should be quoted as one.
///
/// **This lever OVERLAPS the JB2008 libm retirement directly below.** That one
/// made transcendentals cheaper; this one deletes an `atan2` outright, so the
/// two compete for part of the same cost. The 5.39% ± 0.24% wall figure
/// measured for this change was taken against 9516707, i.e. BEFORE the libm
/// work landed, and it is NOT valid as an increment on top of it. Treat the
/// two as overlapping until someone re-measures this lever on this base.
/// Corroborating evidence that they overlap: on this base the SPARSE and MIXED
/// rect-loop digests land on exactly the values they had before the libm change
/// existed, i.e. this change ERASES the libm change's effect on those two
/// eclipse-dominated cases.
///
/// **This is a derivative change, not an accuracy change.** The endpoint moved
/// 3.00 cm against the 1 cm tripwire, which sits ~100x below the arc's ~1 m
/// single-ULP noise floor and so trips on any derivative change however
/// accurate. `strict_hf_production_arc_accuracy` stayed green: 0.0663 m before
/// and 0.0101 m after against a 3 m bound. Both readings sit more than 15x
/// BELOW that metric's own 1.035 m noise floor, so the movement is not
/// resolvable by this instrument and must not be reported as an improvement any
/// more than the previous entry's may be reported as a degradation. Note the
/// caveat that entry records: this gate measures `atm_model: 4` while the pin
/// flies the approximation.
///
/// # Re-pinned 2026-08-07 — JB2008 scalar-libm retirement
///
/// | | old | new |
/// |---|---|---|
/// | `rhs_evals` | 7,829 | **7,560** |
/// | `steps` | 473 | **458** |
/// | endpoint | `-215.98053902355073, 1883.1610365394647, 7043.915841650385` | see below |
///
/// **Cause:** two changes in `crates/jb_rs/src/jb2008.rs`, landed together, and
/// THIS pin is the only one in the gate that sees both. `v3_frozen_config`
/// reads `atm_model` from compiled science, which is 5, the x4 approximation.
/// The other JB2008-sensitive pins here and in `rect_loop_pin` hardcode
/// `atm_model: 4`, the exact profile, and see only the second change.
///
/// 1. `jb_density` stopped carrying the five species number densities as
///    `ln(x)` through to the `exp` at the bottom of the function —
///    `exp(ln(x) + y)` became `x * exp(y)`, retiring five `ln` calls per density
///    evaluation. Gated on `QuadratureProfile::RETIRE_SPECIES_ROUND_TRIP`,
///    which is TRUE only on the approximation profile: the exact profile has to
///    reproduce a sealed Orekit fixture bit for bit and Orekit computes the
///    logarithms.
/// 2. `jb_tsub_l`'s two `powf(2.5)` calls became `x * x * sqrt(x)`, on both
///    profiles.
///
/// **The new bits are the more accurate ones, and that is measured, not
/// asserted.** Adjudicated at 60 decimal digits over all 1,601 `(factor,
/// offset)` pairs the JB2008 corpus produces: the retired association's worst
/// relative error against the true value was 1.528e-14 (68.80 ULP) and its mean
/// 1.458e-15; the landed one's worst is 1.782e-16 (0.80 ULP) and its mean
/// 4.980e-17. `ln`'s rounding entered an `exp` ARGUMENT, where absolute error
/// becomes relative error, and this corpus drives `|ln(x)|` to 45.48.
///
/// **The endpoint move is a derivative change, not an accuracy change, and this
/// time that is measured rather than argued.** It moved 1.82 cm against a 1 cm
/// tripwire that sits ~100x below the arc's ~1 m single-ULP noise floor. Run the
/// same old-vs-new comparison at `eps / 1e4` and the endpoints differ by
/// 0.56 MILLIMETRES — 32x less. The 1.82 cm is 97% re-timed step sequence and
/// 3% changed physics.
///
/// `rhs_evals` fell 3.4% and `steps` 3.2% on this arc, and that is NOT a work
/// saving: at `eps / 1e4` the same counter moves 25,270 -> 25,223, i.e. -0.19%.
/// One arc cannot resolve a step-controller perturbation this small, the same
/// caveat the `jb_local_temp_x4` re-pin above records. The real cost change is
/// per evaluation and was measured separately at -4.3% of arc wall time, work
/// matched at the tighter tolerance.
///
/// Both counters stay inside `COUNT_BAND`, so the secondary assertion did not
/// force this re-pin; the constants move anyway because a stale baseline is how
/// the rect-loop pins became a three-day "authorized red".
///
/// **`strict_hf_production_arc_accuracy` does not watch this trajectory.** It
/// calls `propagate_exact_at`, i.e. `exact_profile_dust_config`, which hardcodes
/// `atm_model: 4` — so the only accuracy gate in this file measures the exact
/// profile while the only pin that flies compiled science measures the
/// approximation. Across this change that gate read 0.0470 m before and
/// 0.0663 m after against a 3 m bound, a 45x margin. Both readings sit more
/// than 15x BELOW that metric's own 1.035 m noise floor, so the movement is not
/// resolvable by this instrument and must not be reported as a measured
/// degradation. The gap is recorded here rather than fixed because widening
/// that gate to `atm_model` 5 needs its own bound sized off its own corpus.
///
/// **CLOSED 2026-08-07** by `strict_hf_v3_production_arc_accuracy`, which is
/// that bound sized off that corpus. The gap described in this paragraph stood
/// from 2df59d4 until then; the model-5 arc's accuracy was gated by nothing.
///
/// # Re-pinned 2026-08-06 — short-span `h0`
///
/// | | old | new |
/// |---|---|---|
/// | `rhs_evals` | 9,383 | **7,829** |
/// | `steps` | 570 | **473** |
/// | endpoint | `-215.9805576072203, 1883.1610621527193, 7043.915838831278` | `-215.98053902355073, 1883.1610365394647, 7043.915841650385` |
///
/// **Cause:** `lightyear_odeint_rs::odesolve::solver::SHORT_SPAN_H0_S`. Segments of 60 s or
/// less now open at `span/2` instead of `span/100`, which on this arc's clamped
/// root legs was a ~94x under-guess costing accepted steps. The eval drop on
/// this single arc is 16.6%; the campaign census measured 18.2%.
///
/// **This is a derivative change, not an accuracy change.** The endpoint moved
/// 3.18 cm against a 1 cm tripwire that sits ~100x below the arc's ~1 m
/// single-ULP noise floor, so it trips on any derivative change however
/// accurate. `strict_hf_production_arc_accuracy` stayed green across the same
/// change and is the gate that answers the accuracy question: truncation error
/// read 0.0123 m before and 0.0470 m after against a 3 m bound, and both
/// readings are more than 20x BELOW that metric's own 1.035 m noise floor, so
/// the difference between them is not resolvable by this instrument and must
/// not be reported as a measured degradation.
/// # Re-pinned 2026-08-09 — `atm_model` 5 -> 6 (R22 abscissa)
///
/// | | old | new |
/// |---|---|---|
/// | `rhs_evals` | 7,827 | **7,669** |
/// | `steps` | 471 | **462** |
/// | endpoint | `-215.98054589599604, 1883.1610461752396, 7043.915841185145` | `-215.9807297503743, 1883.1612929757234, 7043.915788273872` |
///
/// **Cause:** the campaign's `atmosphere_model` moved 5 -> 6, i.e. JB2008's
/// middle Boole log step 0.100 -> 0.300 and upper 0.300 -> 0.700 (R16 arm C),
/// landed as a new model code so the science hash would move with the physics.
/// A slightly different density perturbs the trajectory and the adaptive
/// controller takes a different path, which is where the −2.02% evaluation
/// count comes from — the coarser grid costs fewer abscissae AND fewer steps.
///
/// **This is a derivative change, not an accuracy change.** The endpoint moved
/// 0.312 m against the 1 cm tripwire below, which sits ~19x under this arc's
/// 0.2 m noise floor and so trips on any derivative change however accurate.
/// The accuracy question is answered by
/// `v2_broad_grid_density_error_stays_within_rescoped_bound` (`jb_rs`), NOT by
/// the two gates in this file: both difference an arc against the same arc at
/// a tighter `eps`, so a quadrature bias cancels common-mode and they stay
/// green at rungs 28x over the density bound. `strict_hf_v3_production_arc_accuracy`
/// read 0.016336 m before and 0.001128 m after — it got "better", which is
/// exactly why it must not be cited here.
///
/// # 2026-08-09: the equinoctial warm seed
///
/// `LightyearRHS::baseline_warm_offset` seeds the equinoctial longitude solve
/// from the previous call's converged root rather than from the mean
/// longitude, which drops this arc from 25,937 Halley passes to 17,084. The
/// loop exits on the STEP, so a different seed lands on a different last-ULP
/// root and the whole trajectory moves with it.
///
/// | | before | after |
/// |---|---|---|
/// | `rhs_evals` | 7,827 | **7,702** |
/// | `steps` | 471 | **465** |
/// | `rejected` | 15 | 13 |
/// | `segments` | 66 | 67 |
/// | endpoint | see below | moved 0.044784 m |
///
/// **Both counters stayed inside `COUNT_BAND`** -- -1.60% and -1.27% against
/// a 5% band -- so neither constant HAD to move and both are re-pinned anyway,
/// to keep the band centred on what the tree actually does.
///
/// **The evaluation drop is not the lever's doing and must not be quoted as
/// its speedup.** Nothing about seeding a root solve removes an integrator
/// step; this arc simply landed on a slightly luckier step sequence once its
/// ULPs moved, and another arc's dice could go the other way. The structural
/// saving is the passes, and the wall A/B that priced it separated the two:
/// -5.63% total on this arc, of which -1.60% is this eval-count move and
/// -4.10% is per-evaluation.
///
/// **This is a derivative change, not an accuracy change.** The endpoint moved
/// 4.48 cm against the 1 cm tripwire, which trips on any derivative change
/// however accurate. Both 1 m accuracy gates stayed green:
/// `strict_hf_production_arc_accuracy` read 0.010067 m before and 0.042518 m
/// after, `strict_hf_v3_production_arc_accuracy` 0.016336 m before and
/// 0.061242 m after. Both moves are below that metric's own 0.2 m noise floor
/// and so are not resolvable by this instrument; neither may be reported as a
/// measured degradation.
/// # R22 COMPOSED RE-PIN (2026-08-09, integration): both movers above landed
/// in the same round — model 6 AND the warm seed — so the shipped constants
/// below are the COMPOSED measurement, taken fresh on the merged tree and
/// reproduced twice in release. The two single-lever value sets in the
/// ledgers above (7,669/462 for model 6 alone; 7,702/465 for the seed alone)
/// existed only on agent branches and never shipped. Note the composed eval
/// count (7,942) is HIGHER than base (7,827) and than either single lever:
/// the eval-count component is step-sequence dice, exactly as the warm-seed
/// ledger warns — the levers' structural savings are per-pass and per-abscissa
/// and are priced by their own A/Bs, never by this counter.
/// Endpoint bits: `0xc06aff625810d738`, `0x409d6ca532c9f7ed`,
/// `0x40bb83ea7091ab88` (rejected=13, segments=67).
/// # R24 STAGE-BASELINE PREFILL RE-PIN (2026-08-09)
/// `LightyearRHS::prefill_stage_baselines` resolves a step's 16 stage
/// baselines four at a time before the stage loop runs, and the four lanes of
/// a pack share ONE incoming seed instead of chaining. A shared seed is a
/// different starting point for three lanes in four and the longitude loop
/// exits on the step, so the roots move in the last ULP and every digest
/// below it moves with them. Bounded, not arbitrary: `|F - L| <= e` for every
/// root, so the shared seed is wrong by at most the offset's drift across the
/// four stage times.
/// Measured, not asserted: the endpoint moved 0.095850 m, which is ~2x BELOW
/// this arc's own 0.2 m noise floor and so is not a resolvable accuracy
/// change in either direction; both accuracy arms stayed green. The wall is
/// what the change was for -- `events_m6`, two independently built binaries,
/// interleaved, 10 pairs, min-of-block per run, host load <= 3.3 on 10 cores:
/// 7.087 ms base against 6.656 ms after, i.e. 1.065x, and `new` won all 10
/// pairs. Per RHS evaluation, which divides out the eval-count dice below,
/// 892.4 ns against 845.2 ns = 1.056x.
/// The eval count FELL here (7,942 -> 7,875, and steps 480 -> 474). That is
/// the same step-sequence dice the R22 note warns about and is NOT the
/// lever's doing; the structural saving is the per-evaluation figure above.
/// Endpoint bits: `0xc06aff61e1c9a05f`, `0x409d6ca51ed33382`,
/// `0x40bb83ea7182c6b7` (rejected=15, segments=66).
///
/// # Re-baselined 2026-08-09 — R26 Vern7 swap
///
/// `integrator_method` moved "vern9" -> "vern7" in compiled science, and this
/// file now RESOLVES that token rather than hardcoding a stepper, so the arc
/// below is integrated by Vern7. Endpoint moved 0.045280 m against the 1 cm
/// tripwire — expected, and not an accuracy verdict: the bound sits ~19x below
/// this arc's 0.2 m noise floor, so any derivative change trips it. Both
/// accuracy arms stayed GREEN through the swap.
///
/// The counters move a long way and in OPPOSITE directions, which is the
/// signature of an order change rather than of dice: evaluations 7,875 ->
/// 6,752 (-14.3%) while steps 474 -> 667 (+40.7%). Vern7 is a 10-stage
/// 7th-order pair against Vern9's 16-stage 9th-order one, so it takes many
/// more, much cheaper steps. Rejected steps fell 15 -> 2 and segments 66 -> 64.
/// Per-arc wall on the A/B corpus is -9.6%, smaller than the -13.9% evaluation
/// saving because Vern7 costs ~5% more per evaluation in controller and event
/// overhead.
///
/// # Re-baselined 2026-08-09 — R31 model 7 (fitted JB2008 kernel)
///
/// `atmosphere_model` moved 6 -> 7 in compiled science. This arc RESOLVES that
/// integer (`v3_frozen_config`), so it flies the fitted kernel. Endpoint moved
/// 0.025271 m against the 1 cm tripwire — expected, and again not an accuracy
/// verdict for the reason above. Both accuracy arms stayed GREEN, and the V3
/// arm has room to spare: truncation error 0.099689 m against its 1.0 m gate.
///
/// The counters barely move, which is the signature of a QUADRATURE change
/// rather than an order change: evaluations 6,752 -> 6,742 (-0.15%) and steps
/// 667 -> 666 (-0.15%), both in the same direction, with rejected steps (2) and
/// segments (64) unchanged. Model 7 keeps model 6's Boole log steps and only
/// replaces how the two upper fixed plans compute their five per-call scalars,
/// so the density it returns is model 6's to four significant digits and the
/// controller sees very nearly the same trajectory. The win is per-evaluation
/// cost, not step count — see the science seal for the measured arc wall.
///
/// Reproduced twice on the gate host, bit-identical in all five counters and
/// all three endpoint components.
///
/// RE-PINNED 2026-08-13 (h-carry across rebase boundaries, default-on): the V3
/// arc is SRP-effective, so its eclipse-coordinated legs consume the carried
/// controller step and the derivative path changes — evals `6_742 -> 6_396`,
/// steps 666 -> 633, endpoint moved 0.181 m against the 0.010 m tripwire but
/// ~19x below this arc's measured 0.2 m noise floor, and the separate
/// `strict_hf_production_arc_accuracy` gate stayed GREEN (accuracy did not
/// degrade; the tripwire did its job on an intended change). Reproduced 3/3
/// in release on the M1 gate host.
// Re-baselined 2026-08-19 with the epoch move; see V3_PINNED_POS_KM.
//   old epoch rhs_evals 6_396, steps 633
// Re-baselined 2026-08-25 for frame-chain-v2 full-frame atmosphere-relative
// drag. The corrected passive-frame sign and exact centred-interpolant rate
// intentionally move the derivative: rhs_evals 6_431 -> 6_421 and steps
// 636 -> 635. The separate production-arc accuracy gate is the accuracy
// verdict; this test remains the byte/work tripwire.
const V3_PINNED_RHS_EVALS: u64 = 6_421;
const V3_PINNED_STEPS: u64 = 635;
///
/// Unlike `PINNED_POS_KM` these are already shortest, so `excessive_precision`
/// does not fire and is deliberately not suppressed here -- if a future edit
/// pads them back out to 17 digits the lint should say so. Dropping a digit
/// still changes the pinned value and silently re-baselines the gate.
/// RE-BASELINED 2026-08-10 — R44 one-`atan2` JB2008 hour angle.
///
///     [-215.98065698200983, 1883.1611949229605, 7043.915807821153]
///  -> [-215.98065697840764, 1883.1611949180335, 7043.9158078217215]
///
/// CAUSE: the JB2008 adapter now derives the satellite's hour angle from one
/// `atan2` of a cross/dot pair instead of subtracting two right ascensions from
/// two `atan2` calls. Same angle to rounding, different binary64
/// representative, so the density moves in its last digits and drag with it.
///
/// **This pin did NOT trip.** The endpoint moved 6.13 µm, which is 0.061% of
/// the 1 cm tripwire. It is re-baselined anyway, deliberately: a sub-tripwire
/// move left unpinned is budget quietly spent, and a later change that trips
/// this gate would then be carrying an unknown share of someone else's
/// residual. `rhs_evals`/`steps`/`rejected`/`segments` are all UNMOVED at
/// 6742/666/2/64, so there is no eval-count dice roll in this one.
///
/// Accuracy did not degrade: the V3 truncation metric went 0.099689 ->
/// 0.098470 m, inside the established 0.0964--0.1030 band, so the 0.15 m
/// sizing tripwire below is unchanged.
///
/// RE-BASELINED AGAIN 2026-08-10 — R44 unscaled hypot in
/// `geodetic_altitude_km`.
///
///     [-215.98065697840764, 1883.1611949180335, 7043.9158078217215]
///  -> [-215.98065698285842, 1883.1611949240298, 7043.915807820562]
///
/// CAUSE: the three `f64::hypot` calls became `(a*a + b*b).sqrt()`, which is
/// not correctly rounded where `hypot` is, so the geodetic altitude moves in
/// its last digits and carries into JB2008's `sat_altitude_m`.
///
/// This pin did not trip either: 7.56 µm, 0.076% of the 1 cm tripwire, and
/// `rhs_evals`/`steps`/`rejected`/`segments` are again UNMOVED at 6742/666/2/64.
/// Truncation metric 0.098470 -> 0.099075 m, still inside 0.0964--0.1030.
///
/// RE-BASELINED 2026-08-11 — R56 Estrin evaluation of the model-7 fitted
/// Horners.
///
///     [-215.98065698285842, 1883.1611949240298, 7043.915807820562]
///  -> [-215.98065698552062, 1883.1611949276785, 7043.915807820215]
///
/// CAUSE: `jb_rs::jb2008::fitted_v7_horner` evaluates the five degree-14 fitted
/// scalars by Estrin instead of Horner. Reassociating a floating-point sum is
/// not an identity, so the fitted `sub2`/`tloc2`/`ain`/`tloc3` move in their
/// last digits, the model-7 density with them, and drag after that. **Only
/// `atm_model` 7 moves**: models 4, 5 and 6 are bit-identical over all 835 rows
/// of `jb2008_libm_probe`'s corpus, which is why the three `RECT_LOOP_PIN`
/// digests below are unchanged — that file hardcodes `atm_model: 4`.
///
/// This pin did not trip: 4.53 µm, **0.045% of the 1 cm tripwire**, with
/// `rhs_evals`/`steps`/`rejected`/`segments` UNMOVED at 6742/666/2/64 — no
/// eval-count dice roll. Truncation metric 0.099075 -> 0.099355 m, inside the
/// established 0.0964--0.1030 band, so `V3_SIZING_TRIPWIRE_M` is unchanged.
///
/// RE-BASELINED 2026-08-11 — R57 `RETIRE_ZR_ROUND_TRIP` and
/// `DLRSL_ZERO_ABOVE_KM`, both model-7 only.
///
///     [-215.98065698552062, 1883.1611949276785, 7043.915807820215]
///  -> [-215.9806569840459,  1883.161194925717,  7043.915807820689]
///
/// CAUSE: two profile constants in `jb_rs::jb2008`, false on every profile but
/// the flown one. `RETIRE_ZR_ROUND_TRIP` takes the upper segment's step ratio
/// from the altitude ratio instead of from `exp(ln(ratio))` at `n == 1`, which
/// is every production call; `DLRSL_ZERO_ABOVE_KM = 800` drops the
/// seasonal-latitudinal correction above 800 km, where its own envelope
/// `0.02*h*exp(-0.045*h)` bounds it at 1.88e-13 and so bounds the density move
/// at 4.34e-13. **Only `atm_model` 7 moves**: models 4, 5 and 6 are
/// bit-identical over all 843 rows of `jb2008_libm_probe`'s dumps, and the
/// worst model-7 row moves 3.3899e-13 — the derived envelope arriving 1.3x
/// above the measurement.
///
/// This pin did not trip: **2.5 µm, 0.025% of the 1 cm tripwire**, with
/// `rhs_evals`/`steps`/`rejected`/`segments` UNMOVED at 6742/666/2/64. It is
/// re-baselined anyway, on the same argument as the 6.13 µm entry above: a
/// sub-tripwire move left unpinned is budget quietly spent.
///
/// Truncation metric 0.099355 -> **0.095837 m**, which is 0.6 mm BELOW the
/// 0.0964 floor of the band the three entries above cite. That is not a
/// warning, and here is why in both directions. The band is an EMPIRICAL RANGE
/// over 96 draws (`examples/v3_accuracy_floor.rs`, two seeds x 48), not a
/// bound, so its floor is a sample minimum and a 97th draw landing 0.6 mm under
/// it is what a sample minimum does. And the move is the RIGHT SIZE for the
/// mechanism: that constant's own note prices ULP perturbation of this metric
/// at about +-3 mm, and this is -3.5 mm. So read it as a draw, not as an
/// accuracy result in either direction — the density change is 1e-13 relative,
/// which cannot buy 3 mm of truncation accuracy on its own.
///
/// What it does do is move the metric AWAY from `V3_SIZING_TRIPWIRE_M`. The
/// margin to 0.15 m grows from 0.0506 m to 0.0542 m, so the tripwire keeps its
/// sizing argument with room to spare and is unchanged.
///
/// # This gate does NOT detect a model-7 physics change, and three entries in
/// # a row now say so
///
/// The tolerance's own comment says it sits "~100x below this arc's ~1 m
/// single-ULP noise floor, so any derivative change trips it however accurate".
/// That is true of a single-ULP perturbation of the STATE. It is not true of a
/// change to the atmosphere: the last three re-baselines here moved the
/// endpoint 6.13, 7.56 and 4.53 µm against a 10,000 µm tripwire — every one of
/// them between 1,300x and 2,200x under it. **Nothing in this repository bit-
/// detects a model-7 density change.** `rect_loop_pin`, the only real digest
/// detector, hardcodes `atm_model: 4`, and this file's V3 arm is a
/// centimetre-scale tripwire.
///
/// So a green run of the pin suite is NOT evidence that the flown atmosphere is
/// unchanged, and no future reader should take it as such. What did the
/// detecting for R56 was `jb2008_libm_probe`'s per-profile bit dumps, diffed
/// across the two trees; that is the instrument to use for an atmosphere edit,
/// and it is the reason `dump_fitted_profile_density_bits` was added to it.
// Re-baselined 2026-08-11 (stage-prefill node filter, integrator.rs
// `prefill_stage_times`): the prefill stopped packing `c[0]` and the duplicate
// `c[9]`, so node 8 is solved from a different warm-start seed. THIS PIN DID
// NOT TRIP -- the endpoint moved 4 nm, 2.5 million x under `V3_POS_TOL_KM` --
// and is re-baselined anyway, per this file's standing rule that a
// sub-tripwire move left unpinned is budget quietly spent.
//   old [-215.9806569840459, 1883.161194925717, 7043.915807820689]
//
// # Re-baselined 2026-08-19 -- the arc epoch moved into the authorized window
//
// The V3 arc flew at `JD0` = 2460310.5 (2024-01-13) while the Part A v3 JB2008
// persistence arc is 2026-08-15 to 2026-08-31, so every V3 test failed with
// "lookup outside Part A v3 authorized persistence arc" before reaching any
// physics. The arc now flies at `V3_JD0`, the manifest's own `scenario.t0_utc`.
// A 2.5-year epoch change is a different trajectory, so the endpoint moved
// 21703.75 m -- which is the tripwire doing its job on an intended change, not
// an accuracy verdict. `strict_hf_v3_production_arc_accuracy` is green at the
// new epoch, which is the accuracy question.
//
// Copied verbatim from the STRICT_HF_V3_PIN line on Apple arm64. The APPLE-LIBM
// caveat above applies unchanged.
//   old epoch [-215.98076366232678, 1883.161339085555, 7043.91578018637]
// Re-baselined 2026-08-25 for the same frame-chain-v2 correction. The endpoint
// moved 0.969166 m from the frame-chain-v1 scalar-z drag result.
const V3_PINNED_POS_KM: [f64; 3] = [
    -2.285_346_849_457_164e2,
    1.900_294_379_888_474e3,
    7.039_448_907_530_028e3,
];

/// 1 cm, deliberately tight for the same reason the deleted V1 tolerance was:
/// deliberately ~100x below this arc's ~1 m single-ULP noise floor, so any
/// derivative change trips it however accurate. Accuracy remains
/// `strict_hf_production_arc_accuracy`'s job, not this one's.
const V3_POS_TOL_KM: f64 = 1.0e-5;

/// The ARMED derivative watchdog for all post-campaign work.
///
/// Unlike the V1 pin -- which is red by design against a stale 56e80fe
/// baseline, for displacements deliberately logged and never re-pinned -- this
/// one is green and must stay green. It pins the model-7 frozen incumbent the
/// campaign actually flies, so a change that moves strict-HF propagation shows
/// up here immediately instead of hiding behind V1's standing red.
///
/// Release-only for exactly V1's reason: `fp-contract` makes a debug build
/// integrate a different trajectory, so in debug this cannot pass and never
/// could. See the module header.
///
/// The campaign binary is frozen and unaffected by this file.
#[test]
#[cfg_attr(
    not(feature = "bitpin"),
    ignore = "bit tripwire: needs production flags; fp-contract makes debug a different trajectory"
)]
fn strict_hf_v3_production_arc_is_pinned() {
    let _serial = PROBE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (state, census) = propagate_config_at_epoch(v3_frozen_config(), production_eps(), V3_JD0)
        .unwrap_or_else(|error| panic!("the V3 strict-HF arc must propagate: {error:?}"));

    // Printed unconditionally: re-pinning after an authorized science change
    // should be a copy of this line, not a re-derivation.
    println!(
        "STRICT_HF_V3_PIN rhs_evals={} steps={} rejected={} segments={} pos_km=[{:.17e}, {:.17e}, {:.17e}]",
        census.rhs_evals, census.steps, census.rejected, census.segments, state[0], state[1], state[2]
    );

    assert!(
        census.segments > 10,
        "the V3 arc must actually exercise Encke rectification, got {} segments; \
         if the threshold or the arc changed, this pin no longer watches restarts",
        census.segments
    );

    // PRIMARY. Checked first so its message is the one that surfaces.
    let [state_x, state_y, state_z, ..] = state;
    let [pinned_x, pinned_y, pinned_z] = V3_PINNED_POS_KM;
    #[expect(
        clippy::suboptimal_flops,
        reason = "the tripwire intentionally preserves the established non-FMA reduction order"
    )]
    let err_km = ((state_x - pinned_x).powi(2)
        + (state_y - pinned_y).powi(2)
        + (state_z - pinned_z).powi(2))
    .sqrt();
    assert!(
        err_km <= V3_POS_TOL_KM,
        "V3 strict-HF endpoint moved {:.6} m against a {:.6} m TRIPWIRE. The \
         derivative changed. This is NOT an accuracy verdict -- the bound sits \
         ~19x below this arc's measured {ACCURACY_METRIC_FLOOR_M} m noise floor, \
         so any derivative change trips it however accurate. If the change was \
         intended, \
         re-baseline from the STRICT_HF_V3_PIN line above. If you are on glibc, \
         read the caveat on V3_PINNED_RHS_EVALS before assuming a regression. \
         To ask whether accuracy DEGRADED, read strict_hf_production_arc_accuracy.",
        err_km * 1000.0,
        V3_POS_TOL_KM * 1000.0
    );

    // SECONDARY. Catches work changes that leave the answer alone.
    #[expect(
        clippy::cast_precision_loss,
        clippy::as_conversions,
        reason = "the counter predicate deliberately retains the established floating 5-percent band"
    )]
    let within = |measured: u64, pinned: u64| -> bool {
        (measured as f64 - pinned as f64).abs() <= COUNT_BAND * pinned as f64
    };
    assert!(
        within(census.rhs_evals, V3_PINNED_RHS_EVALS),
        "V3 right-hand-side evaluations {} against pinned {} (band {:.0}%), \
         while the endpoint did NOT move. Work changed without changing the answer.",
        census.rhs_evals,
        V3_PINNED_RHS_EVALS,
        COUNT_BAND * 100.0
    );
    assert!(
        within(census.steps, V3_PINNED_STEPS),
        "V3 accepted steps {} against pinned {} (band {:.0}%), while the \
         endpoint did NOT move.",
        census.steps,
        V3_PINNED_STEPS,
        COUNT_BAND * 100.0
    );
}

/// Measured noise floor of the ACCURACY METRIC, which is what this gate bounds.
///
/// The hope was that `|endpoint(eps) - endpoint(eps_ref)|` would be steadier
/// than either endpoint, both terms moving together under a perturbation. It is
/// NOT: the two tolerances take different step sequences, so their trajectories
/// decorrelate and the difference carries that decorrelation on top of the
/// truncation error. Measured, not assumed -- assuming it is what put a 1 cm
/// bound on a 1 m quantity.
///
/// Measured at `ba6a249` on the exactness authority (Apple arm64) by
/// `examples/v3_accuracy_floor.rs`: 48 draws under each of two seeds, every
/// element moved by a random count of ULPs that the harness first CALIBRATED to
/// be the smallest count that changes the production endpoint's bits.
///
/// ```text
///                            p50      p90      max (96 draws)
/// model 4, exact profile    0.043 m  0.121 m  0.147 m
/// model 5, historical      0.025 m  0.095 m  0.192 m
/// ```
///
/// 0.20 m is the max over both models and both seeds, rounded up. It is a
/// RECORD, not a bound: nothing compares against it, it only annotates the two
/// gates' output so a reading can be judged against the noise.
///
/// # This replaces 1.035 m, which was correct for its tree
///
/// The old value came from six single-ULP readings -- 0.092, 0.603, 0.009,
/// 0.190, 0.533, 1.035 m -- taken at `6a856aa` on 2026-07-26. That tree was
/// re-measured directly with today's calibrated harness, which reproduces its
/// recipe (worst single-ULP metric 1.056 m against the recorded 1.035 m) and its
/// arc (unperturbed 0.269 m against the 0.27 m recorded beside `ACCURACY_TOL_M`).
/// So the era measurement was sound.
///
/// ```text
///                              6a856aa    ba6a249    ratio
/// unperturbed error, model 4   0.269 m    0.010 m    26.6x
/// metric floor, 96 draws       3.782 m    0.147 m    25.7x
/// accepted steps                 359        461      0.78x
/// segments                        64         67      0.96x
/// ```
///
/// **The floor tracks the arc's own truncation error, and the arc got 26.6x
/// more accurate.** The two ratios agree to 3%. That is the whole explanation.
///
/// Three things this refutes, all of which were on the record here:
///
/// 1. The step-count hypothesis. The counters barely moved between the two
///    trees, and steps went UP. The "3,844 steps" it was built on belongs to a
///    transient tree on 2026-08-02, a week after 1.035 m was recorded.
/// 2. Segments as the alternative driver: 64 to 67, also unchanged.
/// 3. "The recorded floor did not reproduce." It reproduces exactly, on the
///    tree that recorded it.
///
/// The six-sample estimate did understate its own era: the same tree reads
/// 3.782 m over 96 calibrated draws against the 1.035 m six single-ULP readings
/// gave it. Six draws do not estimate a tail, which is why this constant is now
/// sized off 96 and reported as a distribution.
///
/// # It is a function of eps, not a property of the arc
///
/// The endpoint floor down an eps ladder at the tip, model 4: 0.146 m at 1e-8,
/// 0.026 m at 1e-10, 0.002 m at 1e-12. A `dt_max` sweep says the same thing
/// from the other side -- forcing 4,406 steps instead of 461, at constant
/// segments, LOWERS the floor 17x. So more integration work means a lower floor,
/// which is the opposite of what the step-count hypothesis needed.
const ACCURACY_METRIC_FLOOR_M: f64 = 0.20;

/// 1 m, sized off the WORST measured arc rather than this convenient one.
///
/// ```text
/// worst production-eps error, 11-arc corpus, at 579a1b6  0.313 m  <- NOT binding
/// same corpus, same commit, +-2x eps neighbourhood       0.313 -- 3.279 m
/// worst production-eps error, 11-arc corpus, stale era   2.31  m
/// this arc's own value, at ba6a249                       0.01  m
/// metric noise floor, re-measured                        0.20  m
/// dust_intercept_tol_km science budget                  10.00  m
/// ```
///
/// Sizing a bound off the convenient arc is how three bad tolerances happened;
/// 0.27 m would have given a gate that fires on a healthy arc from the same
/// corpus.
///
/// RE-PINNED 3.0 -> 1.0 (2026-08-08). The previous comment held this bound at
/// 3.0 m explicitly "until `tolerance_cost_accuracy.rs` re-reads its corpus".
/// That re-read happened at 579a1b6: all 11 arcs, worst 0.313 m at production
/// eps -- the corpus improved 7.4x, consistent with this arc's own 26.6x. The
/// new bound carries 3.2x margin over the re-measured corpus worst and 5x over
/// the 0.20 m noise floor, versus the old bound's 1.3x margin over its era's
/// corpus. Re-size again from the corpus, never from one arc.
///
/// # 2026-08-09: the 0.313 m is REAL and the 3.2x margin claim is NOT
///
/// The measurement above reproduces exactly -- `alt800_am1.948_tof111874`,
/// 0.31347 m, unmodified harness at 579a1b6, re-run at `c4ea964`. Nothing about
/// its provenance is in doubt. What is wrong is treating it as a quantity that
/// can carry a margin.
///
/// "Corpus worst at production eps" is one draw from a scatter. Holding the
/// tree FIXED at 579a1b6 and sweeping eps across a +-2x neighbourhood of
/// production (7e-9 .. 2e-8, nine rungs, same physics throughout), the corpus
/// worst reads:
///
/// ```text
/// 579a1b6, Vern9   min 0.313   median 0.969   max 3.279   (10.5x spread)
/// c4ea964, Vern9   min 0.196   median 0.832   max 2.231   (11.4x spread)
/// c4ea964, Vern7   min 0.680   median 1.831   max 2.847   (4.2x spread)
/// ```
///
/// **0.313 m is the MINIMUM of its own neighbourhood** -- the single luckiest
/// rung at the commit it was taken on. Against that neighbourhood's median the
/// 1 m bound has ~1x margin, and against its max it has 0.3x. The stated 3.2x
/// does not exist at any commit.
///
/// The corpus has NOT degraded since. Every statistic at the tip is equal or
/// better than at 579a1b6 (median 0.969 -> 0.832, max 3.279 -> 2.231, p90 over
/// all 99 arc-rung draws 0.480 -> 0.382). A tip reading of 1.134 m on the same
/// arc at exactly 1e-8 is that arc's neighbourhood MAXIMUM, against a 579a1b6
/// neighbourhood MEDIAN of 0.313 m -- the arc's own median barely moved
/// (0.313 -> 0.329). Comparing single rungs across commits compares lottery
/// draws.
///
/// **What this does and does not license.** It does NOT reopen the bound: this
/// gate measures ITS OWN arc, which reads ~0.08 m, so 1 m is a 12x margin on
/// the thing actually asserted, and the 0.20 m floor supports 1 m at 5x. It
/// DOES retire the corpus-margin sentence above. Do not re-size this bound from
/// a corpus rung again -- if the corpus is consulted, read the neighbourhood
/// distribution, and expect a spread of 10x.
const ACCURACY_TOL_M: f64 = 1.0;

/// Does the production tolerance deliver its accuracy on the EXACT profile,
/// against a converged reference rather than against a previous run?
///
/// The question the tripwire cannot answer. It exists because a provably
/// algebra-equivalent change -- `geodetic_altitude_km`, 2.7 nm at the function
/// -- landed 4.2x further from truth at production `eps` than baseline, and
/// nothing in the system could see the direction it moved.
///
/// # This gate measures `atm_model: 4`, which the campaign does not fly
///
/// Stated at the test rather than only in the log entries below, because for
/// months this was the ONLY accuracy gate in the file and was read as covering
/// production. It does not: it propagates `exact_profile_dust_config`.
/// `strict_hf_v3_production_arc_accuracy` is the gate that answers the same
/// question for the model the campaign flies, and it is the one to read for
/// anything describing production accuracy.
///
/// Kept and not widened, for the reason its own bound records: it is the only
/// accuracy coverage the exact profile has, and the exact profile is the one
/// bound to a sealed Orekit fixture.
///
/// The name predates the split and is deliberately unchanged. It appears in 22
/// places across nine files including `PART_A_LAUNCH_RUNBOOK.md`, every one of
/// them a measurement attributed to this identifier; renaming would turn an
/// accurate record into a dangling one, and the sibling test's name already
/// supplies the disambiguation that a rename would have bought.
#[test]
fn strict_hf_production_arc_accuracy() {
    let _serial = PROBE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (production, _) = propagate_exact_at(production_eps())
        .unwrap_or_else(|error| panic!("the production strict-HF arc must propagate: {error:?}"));
    let (reference, _) = propagate_exact_at(REFERENCE_EPS)
        .unwrap_or_else(|error| panic!("the reference strict-HF arc must propagate: {error:?}"));

    let [production_x, production_y, production_z, ..] = production;
    let [reference_x, reference_y, reference_z, ..] = reference;
    #[expect(
        clippy::suboptimal_flops,
        reason = "the accuracy gate intentionally preserves its established non-FMA reduction order"
    )]
    let err_m = ((production_x - reference_x).powi(2)
        + (production_y - reference_y).powi(2)
        + (production_z - reference_z).powi(2))
    .sqrt()
        * 1000.0;

    println!(
        "STRICT_HF_ACCURACY truncation_err_m={err_m:.6} tol_m={ACCURACY_TOL_M} \
         metric_noise_floor_m={ACCURACY_METRIC_FLOOR_M}"
    );

    assert!(
        err_m <= ACCURACY_TOL_M,
        "truncation error at production eps is {err_m:.4} m against a \
         {ACCURACY_TOL_M} m gate and a 10 m dust_intercept_tol_km budget. \
         Unlike the tripwire this IS an accuracy verdict: the integration is no \
         longer delivering what its tolerance asks for. The gate sits 20x above \
         this metric's own {ACCURACY_METRIC_FLOOR_M} m noise floor, so a \
         reading anywhere near it is real -- but the bound is sized off an \
         11-arc corpus worst of 2.31 m that has not been re-measured, so \
         confirm against the multi-arc corpus in tolerance_cost_accuracy.rs."
    );
}

/// 3 m on the model the campaign flies, sized off its OWN corpus.
///
/// Re-measured at `ba6a249` on the exactness authority (Apple arm64) by
/// `examples/v3_accuracy_floor.rs`, which propagates this exact config at both
/// tolerances and re-reads the same difference over 96 draws whose per-element
/// ULP counts the harness calibrated to be trajectory-visible:
///
/// ```text
/// this arc's own reading, unperturbed                   0.0163 m
/// metric noise floor, model 5, p50 / p90 / max   0.025 / 0.095 / 0.192 m
/// worst production-eps error, 11-arc corpus, at 579a1b6  0.313  m   <- NOT BINDING
/// same corpus, same commit, +-2x eps neighbourhood       0.313 -- 3.279 m
/// dust_intercept_tol_km science budget                  10.00   m
/// ```
///
/// 1 m clears the arc's own reading by 61x, sits 5.2x above the measured
/// floor's worst draw, 3.2x above the re-measured corpus worst, and 10x below
/// the science budget. (The 3.2x is withdrawn -- see the 2026-08-09 note
/// below. The 61x and the table above are the ba6a249 model-5/Vern9
/// measurements; the arc's reading is era-dependent and the live number is in
/// the RESOLVED note below.)
///
/// # RE-PINNED 3.0 -> 1.0 (2026-08-08)
///
/// The previous version of this comment held the bound at 3 m because the
/// 11-arc corpus worst (2.31 m) had not been re-measured since the arc
/// improved 26.6x, and said the way to sharpen was "to re-read the corpus in
/// `tolerance_cost_accuracy.rs`, which is cheap and has not been done". Done
/// at 579a1b6: all 11 arcs at production eps, worst 0.313 m (alt800 long
/// arc), a 7.4x corpus improvement consistent with this arc's own. A
/// floor-sized bound alone would still justify only ~2 m (10x the 0.192 m
/// worst draw); it is the corpus re-read that licenses 1 m, exactly as the
/// old comment demanded. Re-size from the corpus, never from one arc.
///
/// # 2026-08-09: the corpus rung cannot license anything, and this bound leans
/// on it harder than its sibling
///
/// See [`ACCURACY_TOL_M`] for the measurement. In short: 0.313 m reproduces
/// exactly and its provenance is sound, but it is the MINIMUM of a nine-rung
/// sweep across a +-2x tolerance neighbourhood whose median is 0.969 m and
/// whose max is 3.279 m -- at the very commit it was taken on. The corpus has
/// not degraded since (every tip statistic is equal or better), so there is no
/// regression here; the defect is that a single rung was read as a bound.
///
/// This block is the exposed one. Its sibling can fall back on its own arc, and
/// so can this one, but the sentence above explicitly says the FLOOR alone
/// justifies only ~2 m and that the corpus is what buys 1 m. With the corpus
/// citation retired, 1 m is tighter than the stated floor rule supports.
///
/// # RESOLVED 2026-08-09: keep 1 m, on the arc-margin argument
///
/// The 10x-floor sizing rule is replaced, not satisfied: this bound is now
/// justified by the arc it actually gates, at the era that actually flies.
/// Measured at the model-7 land (Vern7, atmosphere model 7) by running this
/// very gate: `truncation_err_m = 0.099689`, so 1 m clears the asserted
/// quantity by 10.0x. (The reading is strongly era-dependent — 0.0163 m at
/// model 5/Vern9, 0.00113 m at model 6/Vern9, 0.0984 m at model 6/Vern7,
/// 0.0997 m at model 7/Vern7 — so any margin quoted here must be re-measured
/// by re-running the gate, never carried
/// forward.) The gate therefore trips on an order-of-magnitude truncation
/// regression while staying 10x under the 10 m science budget. Relaxing to
/// 2 m would halve that sensitivity and no evidence asks for it — the gate is
/// green and the corpus has not degraded. Drift in the arc's own reading is
/// the finding, and that trigger is ENFORCED, not prose: see
/// [`V3_SIZING_TRIPWIRE_M`], asserted in the same gate. Re-size from a full
/// eps-neighbourhood sweep of the corpus (see [`ACCURACY_TOL_M`]), never
/// from one rung.
const V3_ACCURACY_TOL_M: f64 = 1.0;

/// The enforced re-size trigger for the arc-margin argument above.
///
/// Grounded in the noise distribution of the configuration this gate flies,
/// not an earlier era's: `examples/v3_accuracy_floor.rs` at the model-7 land
/// (Vern7, atmosphere model 7, two independent seeds x 48 calibrated
/// trajectory-visible draws) puts the metric's full perturbation
/// distribution at 0.0964 -- 0.1030 m around the 0.099689 m unperturbed
/// reading. (Model 6/Vern7 measured 0.0958 -- 0.1008 m around 0.098435 m —
/// the fitted-kernel swap barely moved the metric.) The metric at this
/// configuration is truncation-dominated: ULP perturbation moves it by
/// ~+-3 mm, not the ~0.19 m the model-5/Vern9-era floor showed.
///
/// **That range is 96 DRAWS, not a bound, and a reading has already landed
/// outside it.** R57's model-7 atmosphere change (see `V3_PINNED_POS_KM`) reads
/// 0.095837 m, 0.6 mm below the 0.0964 floor, which is a sample minimum
/// behaving like one rather than a signal — the move is -3.5 mm against the
/// +-3 mm this paragraph already prices ULP perturbation at. Do NOT treat
/// "inside 0.0964 -- 0.1030" as a pass condition: nothing asserts it, the only
/// thing it sizes is the tripwire below, and a value under the floor moves
/// AWAY from that tripwire.
///
/// 0.15 m is ~1.5x the distribution max, so a trip cannot be perturbation
/// noise; the
/// extra headroom above the band exists because the metric's cross-host
/// (libm) behaviour is unmeasured until the first cluster run reports its
/// value.
///
/// Tripping this is NOT an accuracy verdict — the 1 m gate is that, and it
/// is a SEPARATE test so the two cannot be conflated at the cargo level. A
/// trip is an order to re-argue the bound. A deliberate truncation-moving
/// land re-measures the floor at its configuration and updates the RESOLVED
/// baseline and this constant in the same commit with old -> new -> cause,
/// exactly like a digest re-pin.
const V3_SIZING_TRIPWIRE_M: f64 = 0.15;

/// The accuracy gate for the model the campaign actually flies.
///
/// Same question and same method as `strict_hf_production_arc_accuracy`, on
/// `v3_frozen_config` instead of `exact_profile_dust_config`. It exists because
/// until 2026-08-07 the flying approximation's accuracy was gated by nothing at
/// all: `strict_hf_v3_production_arc_is_pinned` watches its bits, and a bit
/// tripwire ~100x below the arc's noise floor trips on every derivative change
/// however accurate, so it can only say "something moved", never "it got
/// worse".
///
/// The gap was visible and recorded rather than closed twice -- the JB2008
/// libm-retirement entry above says so in as many words, and deferred it
/// because widening the model-4 gate "needs its own bound sized off its own
/// corpus". This is that bound; see `V3_ACCURACY_TOL_M`.
///
/// Not release-gated, unlike the tripwire beside it. It compares two
/// propagations of the same build against each other, so `fp-contract` moves
/// both terms and a debug run measures a real, if slower, truncation error.
#[test]
fn strict_hf_v3_production_arc_accuracy() {
    let _serial = PROBE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // The point of the whole test: assert what is being flown, so that an
    // authority move cannot leave this gate silently measuring the
    // exact profile the way its model-4 sibling does.
    let flown = v3_frozen_config();
    assert_eq!(
        flown.atm_model,
        part_a_hybrid().atmosphere_model,
        "the V3 accuracy gate must propagate the compiled atmosphere model"
    );
    assert_ne!(
        flown.atm_model,
        exact_profile_dust_config().atm_model,
        "the V3 accuracy gate has collapsed onto the exact profile; it is now a \
         duplicate of strict_hf_production_arc_accuracy and gates nothing new"
    );

    let err_m = v3_truncation_err_m("STRICT_HF_V3_ACCURACY");

    assert!(
        err_m <= V3_ACCURACY_TOL_M,
        "truncation error at production eps on atm_model {} is {err_m:.4} m \
         against a {V3_ACCURACY_TOL_M} m gate and a 10 m dust_intercept_tol_km \
         budget. This IS an accuracy verdict on the model the campaign flies: \
         the integration is no longer delivering what its tolerance asks for. \
         Read V3_ACCURACY_TOL_M before loosening it -- the bound rests on the \
         arc-margin argument in its RESOLVED note, and a reading anywhere near \
         the gate is far outside the metric's measured perturbation band \
         (0.0964 -- 0.1030 m at this configuration), so it is real; confirming \
         it needs the multi-arc corpus in tolerance_cost_accuracy.rs.",
        flown.atm_model
    );
}

/// The sizing tripwire, deliberately a SEPARATE test from the science gate:
/// a trip here and a 1 m accuracy failure must be distinguishable at the
/// cargo level (launch section 4 drivers judge limbs by test outcome, and a
/// conflated assert would mislabel a sizing finding as a launch-blocking
/// accuracy failure and halt the remaining gates).
#[test]
fn strict_hf_v3_sizing_tripwire() {
    let _serial = PROBE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let err_m = v3_truncation_err_m("STRICT_HF_V3_SIZING");

    assert!(
        err_m <= V3_SIZING_TRIPWIRE_M,
        "V3 truncation error at production eps is {err_m:.4} m: above the \
         {V3_SIZING_TRIPWIRE_M} m sizing tripwire (measured perturbation \
         distribution 0.0964 -- 0.1030 m at Vern7 / atmosphere model 7; a \
         reading above ~1.5x its max cannot be noise). This is NOT an \
         accuracy failure -- the science gate is strict_hf_v3_production_arc_\
         accuracy -- it means the arc-margin argument sizing the 1 m bound no \
         longer describes this tree. Re-run examples/v3_accuracy_floor.rs at \
         this configuration, re-argue the bound per V3_ACCURACY_TOL_M's \
         RESOLVED note, and update the baseline and this tripwire in the same \
         commit with old -> new -> cause. Do not delete this assert to make \
         it pass."
    );
}

/// The differenced truncation metric both V3 tests read: production eps vs
/// the 1e-12 reference on the flown V3 arc, printed under `tag` so each test
/// reports the measured metres even when green.
#[expect(
    clippy::panic,
    reason = "test-only helper shared by two #[test] fns; a failed propagation must fail the caller"
)]
fn v3_truncation_err_m(tag: &str) -> f64 {
    let (production, _) = propagate_v3_at(production_eps())
        .unwrap_or_else(|error| panic!("the V3 strict-HF arc must propagate: {error:?}"));
    let (reference, _) = propagate_v3_at(REFERENCE_EPS)
        .unwrap_or_else(|error| panic!("the V3 reference strict-HF arc must propagate: {error:?}"));

    let [production_x, production_y, production_z, ..] = production;
    let [reference_x, reference_y, reference_z, ..] = reference;
    #[expect(
        clippy::suboptimal_flops,
        reason = "the accuracy gate intentionally preserves its established non-FMA reduction order"
    )]
    let err_m = ((production_x - reference_x).powi(2)
        + (production_y - reference_y).powi(2)
        + (production_z - reference_z).powi(2))
    .sqrt()
        * 1000.0;

    println!(
        "{tag} atm_model={} truncation_err_m={err_m:.6} tol_m={V3_ACCURACY_TOL_M} \
         sizing_tripwire_m={V3_SIZING_TRIPWIRE_M}",
        v3_frozen_config().atm_model
    );
    err_m
}

/// The `BaselineCalculator` hit rate on the flown V3 arc, printed rather than
/// asserted, and the non-vacuity witness for the shared baseline slot.
///
/// `LightyearRHS::baseline_calculator` used to hand every caller a calculator
/// that owned its memo, and the calculator is minted fresh at each site, so
/// the entry was thrown away and every instance opened with a guaranteed miss.
/// Sharing one slot on the RHS is bit-identical -- the producer is still the
/// pure `equinoc2eci_impl`, the key is still the exact bits of `tof`, and
/// `reset_for_propagation` clears the slot in the same breath as it replaces
/// the elements the key is interpreted against -- so the pins above cannot see
/// it, green or red. THIS is what sees it: the miss count is the number of
/// conversions the arc actually pays for, and a change that does not move it
/// has done nothing.
///
/// Recorded on this arc, M1 Max, at the commit that shared the slot:
///
/// | arm | consults | hits | misses | hit rate |
/// |---|---:|---:|---:|---:|
/// | private slot per calculator | see the printed line | | | |
/// | one slot on the RHS | " | | | |
///
/// It prints instead of asserting because the counts are a function of the
/// step schedule, which every tolerance and stepper change moves; an assert
/// here would be a second pin on the schedule wearing a cache's clothes.
#[test]
fn baseline_calculator_hit_rate_on_the_v3_arc() {
    let _serial = PROBE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    propagate_v3_at(production_eps()).expect("the flown V3 arc must propagate");

    let census = probe::snapshot();
    let hits: u64 = census.iter().map(|entry| entry.baseline_calc_hit).sum();
    let misses: u64 = census.iter().map(|entry| entry.baseline_calc_miss).sum();
    let consults = hits + misses;
    let rate = match (hits.to_f64(), consults.to_f64()) {
        (Some(hit_count), Some(consult_count)) if consult_count > 0.0 => {
            100.0 * hit_count / consult_count
        }
        _ => 0.0,
    };
    println!(
        "BASELINE_CALC_CENSUS consults={consults} hits={hits} misses={misses} \
         hit_rate_pct={rate:.2}"
    );
}
