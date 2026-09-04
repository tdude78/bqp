//! Wall-clock harness for one strict-HF lowering propagation.
//!
//! MEASUREMENT HARNESS, `#[ignore]`d. It exists so an A/B can be run
//! INTERLEAVED ACROSS TWO BUILDS: each invocation warms, then times
//! `block_props()` propagations per block for `blocks()` blocks and prints one
//! `PROP_TIMING_BLOCK` line per block. Alternate invocations of the two builds
//! and take min-of-block per arm; the minimum is the statistic that is robust
//! to the host load this box actually has, and `rhs_share_of_prop`-style
//! ratios are NOT to be used here (they are a contention thermometer, not a
//! build measurement).
//!
//! # THE A/B NULL FLOOR IS A PROPERTY OF THE BLOCK SHAPE, NOT OF THE ARC
//!
//! Read this before quoting any floor for this harness, and before parking a
//! lever as locally unresolvable.
//!
//! The floor is measured by entering the SAME binary twice under two labels and
//! asking how far it separates from itself. That number is not a constant of the
//! machine. It is a function of `block_props()`, because min-of-many only
//! recovers an unloaded time when some block lands in a clean slice, and the
//! longer the block the less often that happens:
//!
//! | block shape | blocks/arm | same-binary null | measured |
//! |---|---|---|---|
//! | `ND_BLOCK_PROPS=25`, the sealed default (~125 ms/block) | 60 | **1.13%** | R44, 2026-08-10, load ~8 |
//! | `ND_BLOCK_PROPS=1` (~5 ms/block) | 1500 | **0.10%** | 2026-08-11, load 20-290 |
//!
//! **An 11x difference from the block shape alone.** The 1.13% figure has been
//! cited as a property of the arc and has parked a class of sub-1% levers as
//! locally unresolvable; several of those were parked on an artifact of this
//! harness rather than on a property of the machine. The single-prop instrument
//! is also load-proof where the 25-prop one is not: the 25-prop min read 10.8 ms
//! at load 170 and 5.0 ms at load ~100, while the single-prop min read 4.93 ms
//! at load 244.
//!
//! So: **never quote a floor without its block shape**, exactly as a saving is
//! never quoted without its two totals. If you are re-opening a parked lever,
//! check which shape produced the bound you inherited.
//!
//! Worked example, the const-generic Legendre dispatch (2026-08-11): predicted
//! 2.13% of arc from factors, measured -3.28% directly at
//! `ND_BLOCK_PROPS=1 ND_BLOCKS=250`, against the 0.10% null above. At the sealed
//! 25-prop shape that lever sits at 2.9x its floor; the factor route was needed
//! only because the 1.13% figure was taken to forbid the direct measurement.
//!
//! **Third refinement, 2026-08-11, and it changes what you should REPORT rather
//! than what you should measure.** Every floor above is a SAME-BINARY null: one
//! build entered twice under two labels. Almost no real A/B is that. Real arms
//! are two BUILDS, and they diverge before either one runs. Two builds of
//! identical source into separate `CARGO_TARGET_DIR`s, measured on the MF
//! harness the same day, separate by **2.60%** on a 10-core host at load 13 --
//! against **0.06%** for the same binary against a copy of itself on that same
//! host, and against a 0.12%-class two-build figure on a quiet one. So the
//! floor is a function of the block shape AND of the host AND of whether the
//! arms are one build or two. Measure the control you actually need, in the
//! session you are measuring in, and say which of the three you ran.
//!
//! CAUTION ON THE 2.60% IN THAT PARAGRAPH: it is WITHDRAWN, see the withdrawal
//! below. The two binaries behind it were not identical source. The 0.12% figure
//! in the two-build section further down is a SEPARATE measurement, by a
//! different agent on a different host -- not affected by this withdrawal, but
//! also NOT a threshold to adopt. See the caveat there: measure your own
//! two-build control rather than inheriting anyone's number.
//!
//! **Reduce INSIDE a round, then across rounds** -- never with a global minimum
//! over every block of an arm. That is a max-order statistic: one lucky slice on
//! one arm sets it and the other arm cannot answer. The same data read 1.19%
//! apart on a global minimum and 0.118% apart on a mean of per-round minima.
//!
//! **THE TWO-BUILD FLOOR NUMBERS THAT WERE HERE ARE WITHDRAWN.** A "+2.60% at
//! 5 of 12" control and a "-5.34% at 10 of 12" control were both published in
//! this header, and NEITHER WAS A CONTROL: the two binaries came from the tree
//! before and after a lever landed, so the pair measured that lever twice while
//! wearing a control's label. Do not quote either figure, and do not quote the
//! conclusion drawn from the second one -- that a null "produces 10 of 12" was
//! inferred from a comparison that was not a null.
//!
//! What those two readings DO establish is worse for wall-clock A/B and rests on
//! nothing but their own disagreement: **the same comparison, the same two
//! binaries, read +2.60% and then -5.34%. An eight-point swing on identical
//! inputs.** Whatever that host was doing, no sub-5% wall claim survives it. No
//! clean two-build floor for this host is currently known -- **and one now is,
//! supplied below, because the withdrawal above is mine and so is the repair.**
//!
//! **THE CLEAN NUMBER, provenance stated — and then a THREE-WAY test that
//! changes what it means.** Binaries M1, M2, M3, all built from `f05e38f` into
//! separate `CARGO_TARGET_DIR`s, digests recorded, all three pairwise
//! comparisons run on the MF harness at `mf-p8-e24`, 12 rounds each:
//!
//! | pair | mean of round ratios, `total_s` | spread | rounds |
//! |---|---|---|---|
//! | M1 vs M2 | -0.23% | 0.9786..1.0072 | 7/12 |
//! | M1 vs M3 | +0.11% | 0.9908..1.0119 | 2/12 |
//! | M2 vs M3 | +0.15% | 0.9904..1.0071 | 3/12 |
//!
//! **LAYOUT BIAS IS REFUTED.** If one binary were simply laid out better, the
//! three ratios would COMPOSE: `r(1->2) * r(2->3)` should equal `r(1->3)`. It
//! comes to 0.99920 against a measured 1.0011 — a **transitivity residual of
//! -0.19%, which is 83% of the largest individual deviation.** The composition
//! error is the same size as the effects, so no consistent per-binary speed
//! exists at this resolution. Three builds of one commit agree to ~0.2%.
//!
//! **SO THE "TWO-BUILD FLOOR" ON THIS HOST WAS NEVER ABOUT THE BUILDS.** The
//! SAME pair, M1 vs M2, read **+2.26% in a contended window and -0.23% in a
//! quiet one** — a 2.5-point swing on identical binaries. What was being
//! measured and called build divergence is host contention, which the
//! same-binary null suffers equally. Quote build-to-build divergence at this
//! shape as **~0.2%**, and quote contention separately, because they are
//! different terms and only one of them goes away when the box is idle. This
//! supersedes the +2.26% figure, which was mine, measured in a contended
//! window, and offered here as a floor when it was a weather report.
//!
//! That also puts this host's quiet-window two-build term within a factor of
//! two of sp1-floor's independently measured 0.12% at `ND_BLOCK_PROPS=1`, on a
//! different host with its own provenance. Two agents, two hosts, one story.
//!
//! **And the round count is now thoroughly dead as a discriminator.** Across
//! nulls on this host the round count has read 2, 3, 4, 5, 6, 7 and 10 out of
//! 12. It spans nearly the whole range. Only the maximum means anything, and
//! only as part of separation.
//!
//! **A CONTROL ARM'S PROVENANCE MUST BE PROVEN, NOT REMEMBERED.** "Prove the
//! control arm is inert" is already a rule here, and it was followed for the
//! poison arms and the untouched buckets -- then two binaries were labelled
//! "identical source" from recollection of how they had been built. Binaries
//! accumulate across an afternoon and get labelled by INTENT instead of by
//! PROVENANCE. The check is cheap: build both arms in the same command, from a
//! stated commit, and record that commit beside the numbers.
//!
//! The strongest form needs no statistic at all: **the lever's WORST round beats
//! the control's BEST round, every round, zero overlap.** Worked example the
//! same day: a bit-identical SIMD-constant fix scored 12 of 12 SEPARATED on both
//! timing buckets while the two buckets it cannot reach scored 9 of 12 and 6 of
//! 12 -- null-shaped. **SEPARATION is the criterion that survived**, because it
//! is the one form the control cannot fake: a biased null still overlaps.
//!
//! Two other levers scored 10 of 12, overlapping, at about -3.1%, and both are
//! WITHDRAWN. Note the reason, because the first one given was itself retracted:
//! NOT "because a null also reached 10 of 12" -- that null turned out not to be
//! a null. They are withdrawn because an overlapping round count never
//! established a SIZE, and because the same comparison on that host reproduced
//! eight points apart. Below the separation threshold there, wall clock has
//! nothing to say.
//!
//! A lever that moves what it touches and leaves what it does not is
//! self-controlling, and that is better evidence than a bigger number.
//!
//! **When separation is unreachable, COUNTED WORK REMOVED is the only evidence
//! left.** For a bit-identical change that is sufficient on its own: the counts
//! are exact, contention-immune, and the change cannot alter the answer. Do not
//! reach for a wall number to dress it up -- a null on this host will happily
//! supply one with the sign you were hoping for.
//!
//! Two things the shorter block does NOT buy you. Arm order must still rotate
//! every round, or a monotone warm-up trend lands on whichever arm holds the
//! last slot and the null cannot see it by construction. And the control arm
//! must still be inert: a `black_box(false)` dropped into a hot predicate adds a
//! barrier on every call, so it prices the barrier alongside the lever.
//!
//! # THE SAME-BINARY NULL IS NOT THE FLOOR FOR A TWO-BUILD A/B
//!
//! Every figure above is one binary entered twice. Almost no real A/B is: the
//! two arms are two BUILDS, and that carries a term the same-binary null cannot
//! contain. Measured 2026-08-11 by building the identical source into two
//! `CARGO_TARGET_DIR`s and running them as separate arms in the same rotation:
//!
//! | comparison | separation, single-prop shape |
//! |---|---|
//! | same binary, two labels | **0.018%** best-round, 0.30% mean-of-round-min |
//! | two builds, identical source | **0.113%** and **0.118%**, two runs |
//!
//! PROVENANCE CAVEAT, added after a sibling figure was retracted for exactly
//! this: "identical source" here is the measuring agent's ASSERTION, and no
//! commit or binary hash was recorded beside it in this header. A pair labelled
//! the same way on another host turned out to be one build from before a lever
//! and one from after. So the two numbers in that table are a RECORD of what one
//! session saw, not a bar, not a bound, and not something to compare a lever
//! against.
//!
//! **DO NOT ADOPT 0.12% AS A THRESHOLD.** It is recorded above as the largest
//! two-build separation anyone has measured with this harness, and that is all
//! it is: one agent, one host, one afternoon, provenance asserted rather than
//! proven. Every other finding in this header says a floor is a function of the
//! block shape AND the host AND the session, so a portable numeric bar is the
//! one thing this evidence cannot support -- and a sibling figure gathered the
//! same way was retracted the same day when its two "identical" builds turned
//! out to straddle a lever.
//!
//! What IS operative: **measure your own two-build control, in the session you
//! are measuring in, from a stated commit, with both arms built in one command
//! and their hashes recorded beside the numbers.** The figures above tell you
//! roughly what to expect and which comparison to run. They do not tell you what
//! your lever has to clear.
//!
//! The qualitative results DO carry, because they are about shape rather than
//! magnitude: a two-build null exceeds a same-binary null, both exceed nothing,
//! and arms differ before either one runs. So build BOTH controls when the
//! margin is small -- an independently built twin of the control is the only
//! thing that separates the lever from its own layout.
//!
//! **And take per-ROUND minima, not the global minimum.** The global min over
//! ~1200 blocks is a max-order statistic over 1200 draws, so one lucky clean
//! slice on one arm moves it and no amount of averaging on the other arm answers
//! back. The same two control binaries above read **1.19% apart on the global
//! min and 0.118% apart on the mean of per-round minima, in the same data** --
//! the global min had caught one outlying block on one arm. Reduce within a
//! round first, then across rounds.
//!
//! The strongest form, and the one that needs no choice of statistic at all: if
//! the lever's WORST round beats the control's BEST round across every round of
//! every run, the separation is not a statistic, it is an ordering. That is what
//! settled the shared baseline slot (`rhs.rs::baseline_calc_memo`) at -2.0%,
//! 20 rounds to 20 with no overlap.
//!
//! Deliberately NOT in `alloc_census.rs` (which no longer exists; the surviving allocation instrument is `two_phase_transfer_rs/benches/allocation_bench.rs`): that file installs a counting global
//! allocator, whose disarmed cost is one relaxed atomic load per allocation —
//! which is proportional to the allocation count and would therefore bias an
//! A/B whose whole subject is the allocation count.
//!
//! # TWO ATMOSPHERE MODELS, AND EVERY BLOCK LINE NAMES ITS OWN
//!
//! This harness used to have exactly one arm per code path, all three of them
//! `atm_model: 4`, under a builder called `production_dust_config`. The campaign
//! has flown `atmosphere_model: 5` since 2df59d4, so every arc-wall percentage
//! ever taken through this file is a MODEL-4 share of an arc nobody flies. That
//! is not a rounding difference: `ExactOrekitQuadrature` (model 4) runs a
//! 63-step middle quadrature plan where `LogQuadratureX4ApproxV1` (model 5) runs
//! 16, so model 4 issues roughly four times the `atan_x4` traffic per kernel
//! call, and the arcs are not even the same length (7,976 evaluations against
//! 7,560). Arbitrated at `741b7ff`, re-running one commit's A/B at both models:
//!
//! | corpus | atan lever (`6f785d2`) | eclipse supremum (`4458015` vs `70ce67e`) |
//! |---|---|---|
//! | `atm_model: 4` (`exact_profile_dust_config`) | +4.40%, 81.5 ns/eval | +6.17% |
//! | `atm_model: 5` (historical authority arm) | +1.59%, 20.6 ns/eval | +2.73% |
//!
//! Two lanes reported +4.74% and -2.14% for that one commit and spent a round
//! calling it an instrument disagreement. It was the atmosphere model. Neither
//! number was wrong; neither could be quoted without naming its corpus.
//!
//! So there is no default here and there is no flag. Each code path gets **two
//! named tests**, and each prints its model in the `arm=` field, because a
//! recorded `PROP_TIMING_BLOCK` line has to stay attributable years after the
//! command that produced it is forgotten:
//!
//! | code path | current compiled authority | model 4 (historical exact profile) |
//! |---|---|---|
//! | `integrate_final_checked`, events on | `strict_hf_propagation_events_authority_ns_per_prop` | `..._events_m4_...` |
//! | `integrate_final_checked`, events off | `strict_hf_propagation_noevents_authority_ns_per_prop` | `..._noevents_m4_...` |
//! | `integrate_adaptive`, sampled | `rectloop_authority_sampled_ns_per_prop` | `rectloop_m4_...` |
//!
//! **Every `arm=` label is a substring of its own test name**, so the label off
//! a recorded line is always a working filter. That is deliberate: the first
//! draft named the tests `..._ns_per_prop_m5` while labelling the arms
//! `events_m6`, and `-- --ignored events_m6` then matched nothing and exited 0
//! -- a filter that silently runs zero tests, which this repo has been bitten by
//! before. One overlap survives and is harmless: `events_authority` also
//! matches `noevents_authority`. Filter on `noevents_` or on the full name to
//! separate them.
//!
//! # ACTIVE SELECTORS ARE STABLE; RECORDED MODEL LABELS ARE HISTORICAL
//!
//! These arms take their model from the sealed authority, so when
//! `atmosphere_model` moved 5 -> 6 (R22 abscissa) and again 6 -> 7 (R31 fitted
//! kernel) they started flying the new model WITHOUT any edit here. Leaving the
//! names alone would have kept every future `PROP_TIMING_BLOCK` line labelled
//! with a model it was not measuring, which is the mislabelling this header
//! exists to prevent. There is no model-5 or model-6 arm to preserve: the
//! config reads the authority, so those names denote configurations the file
//! can no longer produce.
//!
//! **Recorded lines labelled `m5` or `m6` are NOT reproducible on this tree.**
//! Model 6 is roughly 12% faster per arc than model 5, and model 7 is faster
//! again than model 6 by the margin recorded on the science seal. Do not
//! compare across labels and read the difference as a regression or as a win
//! from anything else.
//!
//! Active selectors now say `authority`, read model 8 and every other control
//! from compiled science, and read the epoch from the compiled Part A v3
//! persistence identity. `PROP_TIMING_IDENTITY` makes those bytes explicit in
//! every output. The `m4` arm remains only to reproduce historical measurements
//! at its original valid epoch. A current-authority/model-4 ratio is invalid:
//! those lanes use different atmosphere models and epochs.
//!
//! **The split is not cosmetic, and here is the receipt.** First serialised run
//! on the M1 Max at `741b7ff`, min-of-block: `events_m4` 13.589 ms per
//! propagation against the then-current model-6 arm at 10.002 ms. The arc the
//! campaign flies is 26% cheaper than the one every recorded number was taken
//! on -- close to the ~29% the arbitration derived independently from the
//! evaluation counts -- and model 7 widens that gap further. Any share of arc
//! wall carried across from `m4` to the flown arm is wrong by at least that
//! factor before the lever itself is even considered.
//!
//! ```sh
//! # everything
//! cargo test --release -p lightyear_odeint_rs --test prop_timing \
//!     -- --ignored --nocapture
//! # just current compiled authority -- stable substring filter
//! cargo test --release -p lightyear_odeint_rs --test prop_timing \
//!     -- --ignored --nocapture authority
//! ```

use anyhow::Context;
use num_traits::ToPrimitive;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lightyear_odeint_rs::integrator::{
    integrate_adaptive, integrate_final_checked, ScalarGravityAssets, ScalarPropagationContext,
    ScalarPropagationRequest,
};
use lightyear_odeint_rs::types::StepperMethod;
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags, IntegrationResult};
use satpy_core::{eci2equinoc_impl_f64, kep2eci_impl, SEC_PER_DAY};

const HISTORICAL_M4_JD0: f64 = 2_460_310.5;
const TOF_S: f64 = 43_200.0;
const DIR_R6_D15: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

const WARM_PROPS: usize = 20;

/// The sealed block shape. Every recorded `PROP_TIMING_BLOCK` line in this
/// repo's docs was taken at 25 x 6, and both stay the default so no existing
/// number moves. `ND_BLOCK_PROPS` / `ND_BLOCKS` override them for one
/// measurement -- read the block-shape section of the module header before
/// using either, because the A/B null floor is a function of these two values
/// and quoting a floor without its block shape is how 1.13% became a law.
const DEFAULT_BLOCK_PROPS: usize = 25;
const DEFAULT_BLOCKS: usize = 6;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

/// Propagations per timed block; `ND_BLOCK_PROPS` overrides.
fn block_props() -> usize {
    env_usize("ND_BLOCK_PROPS", DEFAULT_BLOCK_PROPS)
}

/// Timed blocks per invocation; `ND_BLOCKS` overrides.
fn blocks() -> usize {
    env_usize("ND_BLOCKS", DEFAULT_BLOCKS)
}

/// Cargo runs a binary's tests on parallel threads, and these six arms are all
/// CPU-bound propagation loops. Left unserialised they measure each other:
/// running all six at once on the M1 Max returned 27.5--57.5 ms per propagation
/// against the ~10 ms a lone arm costs, i.e. a 2--6x inflation that varies block
/// to block. Min-of-block does not save you from it, because every block of
/// every arm is contended.
///
/// This is the same hazard `rhs_share_of_prop` was retired for -- a number that
/// is monotone in host load is a thermometer, not a measurement -- and the arm
/// count doubling from three to six made it worse rather than introducing it.
/// The lock is held for the whole warm-plus-blocks run of one arm, so the
/// others park rather than compete, and a full `--ignored` sweep costs the sum
/// of the arms instead of returning six unusable numbers.
///
/// It does NOT protect against load from outside this process. Read the
/// min-of-block note in the module header for that.
static TIMING_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn require_complete_sampled_result(result: IntegrationResult) -> anyhow::Result<IntegrationResult> {
    anyhow::ensure!(
        !result.max_steps_exceeded,
        "sampled timing arc exceeded its integration-step limit"
    );
    anyhow::ensure!(
        !result.terminal_event_fired,
        "sampled timing arc stopped at terminal event {}",
        result.terminal_event_name
    );
    Ok(result)
}

/// The compiled Part A science authority, read rather than restated.
///
/// The literals that used to sit below -- gravity order 5, `am_ratio` 1.948,
/// `cd` 2.2, `cr` 1.3, `dt_max` 300 s, `eps` 1e-8 -- were a READER of a sealed
/// constant, which is the failure mode that keeps passing while measuring
/// something else. They are unchanged in value by this rewrite; they are now
/// BOUND, so an authority move lands here instead of quietly leaving this
/// harness timing the old science.
const fn part_a_hybrid() -> &'static nd_config::PartAHybridControls {
    nd_config::CompiledPartAScienceV1::part_a_v1().hybrid()
}

/// Everything the campaign compiles, with the atmosphere left to the caller.
///
/// The two public builders below differ in `atm_model` and in NOTHING else, so
/// an A/B across them isolates the quadrature profile.
/// The compiled stepper, resolved rather than restated.
///
/// Replaces a paired `assert_eq!(..., "vern9")` tripwire and hardcoded
/// `StepperMethod::Vern9`. The assert fired correctly on the Vern9 -> Vern7
/// swap, but relaxing it without also editing the literal beside it would have
/// left this harness measuring a stepper the campaign no longer flies, green
/// and silent. Resolving the token removes that second step; the `panic!`
/// keeps the fail-closed property for a stepper this file cannot build.
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

fn dust_config(atm_model: i32) -> ForceConfig {
    let controls = part_a_hybrid();
    ForceConfig {
        sph_order: controls.gravity_order,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        subtract_first_order: true,
        atm_model,
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

/// What the campaign flies: `atm_model` read from the sealed authority.
///
/// Read, not restated, for `strict_hf_pin::v3_frozen_config`'s reason -- if
/// `atmosphere_model` ever moves this harness must follow it rather than keep
/// timing the old model. This is the arm to quote for anything describing what
/// the campaign costs.
///
/// The stable selector deliberately carries no model number. The emitted
/// identity and its exact test bind the current model without renaming callers
/// whenever the compiled authority advances.
fn authority_config() -> ForceConfig {
    dust_config(part_a_hybrid().atmosphere_model)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PropTimingIdentity {
    atmosphere_model: i32,
    epoch_bits: u64,
    stepper: &'static str,
    gravity_order: usize,
    tolerance_bits: u64,
    dt_max_bits: u64,
}

fn prop_timing_identity() -> anyhow::Result<PropTimingIdentity> {
    let controls = part_a_hybrid();
    let scenario = jb_rs::drivers::compiled_part_a_v3_identity()
        .context("compiled Part A v3 persistence identity must load")?;
    Ok(PropTimingIdentity {
        atmosphere_model: controls.atmosphere_model,
        epoch_bits: scenario.t0_utc_jd.to_bits(),
        stepper: controls.integrator_method,
        gravity_order: controls.gravity_order,
        tolerance_bits: controls.tolerance.to_bits(),
        dt_max_bits: controls.dt_max_s.to_bits(),
    })
}

fn emit_prop_timing_identity(identity: PropTimingIdentity, lane: &str) {
    println!(
        "PROP_TIMING_IDENTITY lane={lane} atmosphere_model={} epoch_bits={:#018x} \
         stepper={} gravity_order={} tolerance_bits={:#018x} dt_max_bits={:#018x} \
         target_arch={} debug_assertions={} prop_census={} profile_symbols={} \
         bitpin={} scalar_leg_observer={} avx2={} fma={} neon={}",
        identity.atmosphere_model,
        identity.epoch_bits,
        identity.stepper,
        identity.gravity_order,
        identity.tolerance_bits,
        identity.dt_max_bits,
        std::env::consts::ARCH,
        cfg!(debug_assertions),
        cfg!(feature = "prop-census"),
        cfg!(feature = "profile-symbols"),
        cfg!(feature = "bitpin"),
        cfg!(feature = "scalar-leg-observer"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "fma"),
        cfg!(target_feature = "neon"),
    );
}

fn authority_epoch() -> anyhow::Result<f64> {
    Ok(f64::from_bits(prop_timing_identity()?.epoch_bits))
}

/// The EXACT JB2008 profile, `atm_model: 4`, which the campaign does not fly.
///
/// Deliberately hardcoded and deliberately kept. It is the corpus every
/// historical `PROP_TIMING_BLOCK` numbers were taken on -- see the table in the
/// module header -- so deleting it would make those numbers unreproducible, and
/// it stresses `ExactOrekitQuadrature`'s 63-step middle plan, which is roughly
/// four times the `atan_x4` traffic per kernel call that model 5 issued.
///
/// It is NOT production. Nothing measured here may be quoted as an arc-wall
/// share without naming the model.
fn exact_profile_dust_config() -> ForceConfig {
    dust_config(4)
}

#[test]
fn current_authority_timing_identity_is_exact() -> anyhow::Result<()> {
    let scenario = jb_rs::drivers::compiled_part_a_v3_identity()?;
    let identity = prop_timing_identity()?;
    let controls = part_a_hybrid();

    assert_eq!(identity.atmosphere_model, controls.atmosphere_model);
    assert_eq!(identity.epoch_bits, scenario.t0_utc_jd.to_bits());
    assert_eq!(identity.stepper, controls.integrator_method);
    assert_eq!(identity.gravity_order, controls.gravity_order);
    assert_eq!(identity.tolerance_bits, controls.tolerance.to_bits());
    assert_eq!(identity.dt_max_bits, controls.dt_max_s.to_bits());
    Ok(())
}

#[test]
#[ignore = "measurement harness; prints per-block ns/propagation"]
fn strict_hf_propagation_events_authority_ns_per_prop() -> anyhow::Result<()> {
    timed_arc(
        &authority_config(),
        authority_epoch()?,
        true,
        "events_authority",
    )
}

/// The exact-profile control for the arm above. See the module header before
/// quoting anything from it.
#[test]
#[ignore = "measurement harness; prints per-block ns/propagation"]
fn strict_hf_propagation_events_m4_ns_per_prop() -> anyhow::Result<()> {
    timed_arc(
        &exact_profile_dust_config(),
        HISTORICAL_M4_JD0,
        true,
        "events_m4_historical",
    )
}

/// The `enable_events = false` Encke loop is a SEPARATE code path in
/// `integrate_final_checked_inner`, not a
/// configuration of the one above, and it is what the batch/session callers
/// take. It gets its own arm so its own change can be priced on its own.
#[test]
#[ignore = "measurement harness; prints per-block ns/propagation"]
fn strict_hf_propagation_noevents_authority_ns_per_prop() -> anyhow::Result<()> {
    timed_arc(
        &authority_config(),
        authority_epoch()?,
        false,
        "noevents_authority",
    )
}

/// The exact-profile control for the arm above.
#[test]
#[ignore = "measurement harness; prints per-block ns/propagation"]
fn strict_hf_propagation_noevents_m4_ns_per_prop() -> anyhow::Result<()> {
    timed_arc(
        &exact_profile_dust_config(),
        HISTORICAL_M4_JD0,
        false,
        "noevents_m4_historical",
    )
}

/// The SAMPLED rect loop in `integrate_sampled_inner`, which is a
/// different function from the two arms above and is what `batch.rs` and
/// `two_phase_transfer_rs::evaluate` reach. One eval time at `tf`, so seven of
/// the eight segments take the final-only arm and one takes the sampled arm.
#[test]
#[ignore = "measurement harness; prints per-block ns/propagation"]
fn rectloop_authority_sampled_ns_per_prop() -> anyhow::Result<()> {
    rect_loop_arc(
        &authority_config(),
        authority_epoch()?,
        "rectloop_authority",
    )
}

/// The exact-profile control for the arm above.
#[test]
#[ignore = "measurement harness; prints per-block ns/propagation"]
fn rectloop_m4_sampled_ns_per_prop() -> anyhow::Result<()> {
    rect_loop_arc(
        &exact_profile_dust_config(),
        HISTORICAL_M4_JD0,
        "rectloop_m4_historical",
    )
}

#[test]
#[ignore = "release-only one-block current-authority smoke"]
fn current_authority_one_block_release_smoke() -> anyhow::Result<()> {
    timed_arc_with_shape(
        &authority_config(),
        authority_epoch()?,
        true,
        "events_authority_smoke",
        0,
        1,
        1,
    )
}

fn rect_loop_arc(base_config: &ForceConfig, epoch_jd: f64, label: &str) -> anyhow::Result<()> {
    let _serial = TIMING_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("gravity coefficients must load")?;
    let packed =
        lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")?;

    let kep = [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0];
    let mut init_eci = [0.0; 6];
    kep2eci_impl(&kep, true, 0.0, 0.0, true, &mut init_eci);
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let config = Arc::new(
        (*base_config)
            .with_ephemeris_for_arc(epoch_jd, epoch_jd + TOF_S / SEC_PER_DAY)
            .context("production ephemeris and JB2008 assets must cover the pinned arc")?,
    );
    let gravity = ScalarGravityAssets::new(packed);
    let context = ScalarPropagationContext::new(epoch_jd, config, gravity);

    if label.ends_with("authority") {
        emit_prop_timing_identity(prop_timing_identity()?, label);
    } else {
        println!(
            "PROP_TIMING_IDENTITY lane={label} historical=true atmosphere_model={} \
             epoch_bits={:#018x} stepper={} gravity_order={} tolerance_bits={:#018x} \
             dt_max_bits={:#018x}",
            base_config.atm_model,
            epoch_jd.to_bits(),
            part_a_hybrid().integrator_method,
            base_config.sph_order,
            base_config.eps.to_bits(),
            base_config.dt_max.to_bits(),
        );
    }

    let run = || {
        integrate_adaptive(
            ScalarPropagationRequest::new(&context, init_equ, &[TOF_S], 0.0, TOF_S)
                .with_events(false),
        )
    };

    for _ in 0..WARM_PROPS {
        black_box(require_complete_sampled_result(run()?)?);
    }
    for block in 0..blocks() {
        let start = Instant::now();
        for _ in 0..block_props() {
            black_box(require_complete_sampled_result(run()?)?);
        }
        let elapsed = start.elapsed();
        println!(
            "PROP_TIMING_BLOCK arm={label} block={block} props={} ns_per_prop={:.1}",
            block_props(),
            ns_per_propagation(elapsed, block_props())?
        );
    }
    Ok(())
}

fn timed_arc(
    base_config: &ForceConfig,
    epoch_jd: f64,
    enable_events: bool,
    label: &str,
) -> anyhow::Result<()> {
    timed_arc_with_shape(
        base_config,
        epoch_jd,
        enable_events,
        label,
        WARM_PROPS,
        block_props(),
        blocks(),
    )
}

fn timed_arc_with_shape(
    base_config: &ForceConfig,
    epoch_jd: f64,
    enable_events: bool,
    label: &str,
    warm_props: usize,
    props_per_block: usize,
    block_count: usize,
) -> anyhow::Result<()> {
    let _serial = TIMING_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // This harness times one arc repeatedly through `integrate_final_checked`.
    // Every repetition performs direct propagation; the warm/block split
    // measures the integrator rather than final-result reuse.
    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("gravity coefficients must load")?;
    let packed =
        lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")?;

    let kep = [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0];
    let mut init_eci = [0.0; 6];
    kep2eci_impl(&kep, true, 0.0, 0.0, true, &mut init_eci);
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let config = Arc::new(
        (*base_config)
            .with_ephemeris_for_arc(epoch_jd, epoch_jd + TOF_S / SEC_PER_DAY)
            .context("production ephemeris and JB2008 assets must cover the pinned arc")?,
    );
    let gravity = ScalarGravityAssets::new(packed);
    let context = ScalarPropagationContext::new(epoch_jd, config, gravity);

    if label.contains("authority") {
        emit_prop_timing_identity(prop_timing_identity()?, label);
    } else {
        println!(
            "PROP_TIMING_IDENTITY lane={label} historical=true atmosphere_model={} \
             epoch_bits={:#018x} stepper={} gravity_order={} tolerance_bits={:#018x} \
             dt_max_bits={:#018x}",
            base_config.atm_model,
            epoch_jd.to_bits(),
            part_a_hybrid().integrator_method,
            base_config.sph_order,
            base_config.eps.to_bits(),
            base_config.dt_max.to_bits(),
        );
    }

    let run = || {
        integrate_final_checked(
            ScalarPropagationRequest::new(&context, init_equ, &[TOF_S], 0.0, TOF_S)
                .with_events(enable_events),
        )
    };

    for _ in 0..warm_props {
        let state = match run() {
            Ok(state) => state,
            Err(error) => {
                return Err(
                    anyhow::Error::new(error).context("the pinned strict-HF arc must propagate")
                );
            }
        };
        black_box(state);
    }

    for block in 0..block_count {
        let start = Instant::now();
        for _ in 0..props_per_block {
            let state = match run() {
                Ok(state) => state,
                Err(error) => {
                    return Err(anyhow::Error::new(error)
                        .context("the pinned strict-HF arc must propagate"));
                }
            };
            black_box(state);
        }
        let elapsed = start.elapsed();
        println!(
            "PROP_TIMING_BLOCK arm={label} block={block} props={} ns_per_prop={:.1}",
            props_per_block,
            ns_per_propagation(elapsed, props_per_block)?
        );
    }
    Ok(())
}

fn ns_per_propagation(elapsed: Duration, props_per_block: usize) -> anyhow::Result<f64> {
    let elapsed_ns = elapsed
        .as_nanos()
        .to_f64()
        .context("elapsed timing cannot convert to f64 nanoseconds")?;
    let props = props_per_block
        .to_f64()
        .context("block propagation count cannot convert to f64")?;
    Ok(elapsed_ns / props)
}
