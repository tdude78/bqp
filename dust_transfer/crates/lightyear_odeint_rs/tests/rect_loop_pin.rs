//! Bit pin on the `enable_events == false` Encke rectification loop inside
//! `integrate_sampled_inner` — the SAMPLED path.
//!
//! # Why this file exists
//!
//! `strict_hf_pin` does not reach this code. It calls
//! `integrate_final_checked`, which is a different
//! function with its own restart loops; the sampled `'rect_loop` here is
//! reached only through `integrate_adaptive`, i.e. from
//! `batch.rs`'s `integrate_batch_native_*` (the UKF-sigma route, 39 arcs per
//! row) and from `two_phase_transfer_rs::evaluate`. Before this file, a change
//! to that loop could move every one of those trajectories and the gate would
//! report green.
//!
//! The specific hazard this was written for: the loop builds a fresh
//! `LightyearRHS` per segment, and the sampled arm re-arms it with
//! `reset_cache()` — which clears caches but does NOT move the Encke baseline
//! (`init_equinoc_state` / `t0_s`). A fresh instance gets its baseline from the
//! constructor; a REUSED instance would keep the previous segment's baseline
//! unless the call is `reset_for_propagation`. That failure is silent: the
//! integration still converges, it just converges around the wrong reference,
//! and the only visible symptom is different output bits.
//!
//! # What it pins
//!
//! Raw `f64` bits of every returned delta, for three eval-time layouts that
//! drive the loop through different arms:
//!
//! - `DENSE`  — an eval time in every 5400 s segment, so every segment takes
//!   the sampled arm.
//! - `SPARSE` — one eval time at `tf` only, so seven segments take the
//!   final-only arm and one takes the sampled arm.
//! - `MIXED`  — eval times inside two interior segments only, so the two arms
//!   INTERLEAVE. This is the layout that would catch a baseline carried from a
//!   final-only segment into a sampled one, or the reverse.
//!
//! `DENSE` goes through `integrate_batch_native` with three sigma
//! points, which is the batch entry the UKF-sigma stack calls. The other two go
//! through `integrate_adaptive`, which is `evaluate.rs`'s entry.
//!
//! # RELEASE ONLY, for `strict_hf_pin`'s reason
//!
//! `fp-contract` and release inlining fix the FMA contraction pattern, so a
//! debug build of this same source integrates a different trajectory and no bit
//! pin on it can pass. The assertions carry
//! `#[cfg_attr(not(feature = "bitpin"), ignore)]` — an explicit lane key,
//! because gating on `debug_assertions` let `--profile fast-test`
//! (debug-assertions=false, lto off, cgu=16) run the pins under codegen the
//! baselines were never captured on. Run as:
//!
//! ```sh
//! cargo test --release -p lightyear_odeint_rs --features bitpin --test rect_loop_pin
//! ```
//!
//! Re-baselining after an intended change is a verbatim copy of the
//! `RECT_LOOP_PIN` lines this file prints, not a re-derivation.

use anyhow::Context;
use std::sync::Arc;

use lightyear_odeint_rs::integrator::{
    integrate_adaptive, ScalarGravityAssets, ScalarPropagationContext, ScalarPropagationRequest,
    MAX_RECTIFICATION_SEGMENT_S,
};
use lightyear_odeint_rs::types::StepperMethod;
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags};
use satpy_core::{eci2equinoc_impl_f64, kep2eci_impl, SEC_PER_DAY};

/// The compiled stepper, resolved rather than restated.
///
/// This file's config used to hardcode `StepperMethod::Vern9` in two places
/// with nothing tying it to compiled science. That is a coverage trap rather
/// than a harmless fixture: the sampled rectification loop pinned here is
/// reached from the UKF-sigma batch route, which the campaign DOES fly, so a
/// hardcoded stepper leaves these three digests guarding a trajectory
/// production has stopped integrating -- and they stay GREEN while doing it.
/// Through the Vern9 -> Vern7 swap they did exactly that, which is how this
/// was found.
///
/// `atm_model` below is deliberately NOT read from the authority; see its own
/// comment. The two are separate decisions and only one of them was a defect.
#[expect(
    clippy::panic,
    reason = "a stepper this file cannot build must stop the run, not silently measure a different one"
)]
fn authority_stepper() -> StepperMethod {
    match nd_config::CompiledPartAScienceV1::part_a_v1()
        .hybrid()
        .integrator_method
    {
        "vern7" => StepperMethod::Vern7,
        "vern9" => StepperMethod::Vern9,
        other => panic!("compiled science selects a stepper this file does not build: {other}"),
    }
}

const JD0: f64 = 2_460_310.5;
const DIR_R6_D15: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

/// 12 h, the same arc `strict_hf_pin` uses. At the integrator's
/// `MAX_RECTIFICATION_SEGMENT_S` this floors the loop at 8 segments before any
/// deviation-triggered rebase.
const TOF_S: f64 = 43_200.0;

/// The segment count this file's three eval-time layouts are built around.
///
/// Every layout below is described in terms of "each segment" or "segments 3
/// and 6", which is only true while `TOF_S / MAX_RECTIFICATION_SEGMENT_S` is
/// this number. Derived, not written down, so that moving the production cap
/// moves this instead of silently leaving the layouts pointing at the wrong
/// segments — the exact desync the cap's own doc comment records happening to
/// four earlier private copies of 5400.
const SEGMENTS: u32 = 8;
const EPS: f64 = 1.0e-8;

/// The EXACT JB2008 profile, `atm_model: 4`. **This is not production**, which
/// has flown `atmosphere_model: 5` since 2df59d4.
///
/// Called `production_dust_config` until 2026-08-07, and renamed for the reason
/// `strict_hf_pin::exact_profile_dust_config` records: the name was read as the
/// campaign configuration by two instruments that were not measuring it.
///
/// Model 4 is deliberate HERE and is not the same oversight. This file pins the
/// SAMPLED rect loop's output bits, and the atmosphere is the workload rather
/// than the subject -- the exact profile is the heavier one, so it exercises
/// more of the derivative per segment. All three digests below are measured on
/// it; moving the model would move all three and buy no coverage the model-5
/// arcs in `strict_hf_pin` do not already have.
fn exact_profile_dust_config() -> ForceConfig {
    ForceConfig {
        sph_order: 5,
        force_flags: ForceFlags::DRAG
            | ForceFlags::SRP
            | ForceFlags::SUN_GRAVITY
            | ForceFlags::MOON_GRAVITY,
        subtract_first_order: true,
        atm_model: 4,
        am_ratio: 1.948,
        cd: 2.2,
        cr: 1.3,
        target_propagation_mode: 0,
        dt_max: 300.0,
        eps: EPS,
        integrator_method: authority_stepper(),
        ..ForceConfig::default()
    }
}

struct Fixture {
    init_equ: [f64; 6],
    config: ForceConfig,
    packed: Arc<satpy_core::PackedGravityCoeffs>,
}

fn fixture() -> anyhow::Result<Fixture> {
    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("gravity coefficients must load")?;
    let packed =
        lightyear_odeint_rs::get_global_coeffs_packed().context("packed coefficients must load")?;

    let kep = [7_178.137, 0.025, 97.4, 125.0, 210.0, 180.0];
    let mut init_eci = [0.0; 6];
    kep2eci_impl(&kep, true, 0.0, 0.0, true, &mut init_eci);
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl_f64(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let config = exact_profile_dust_config()
        .with_ephemeris_for_arc(JD0, JD0 + TOF_S / SEC_PER_DAY)
        .context("production ephemeris and JB2008 assets must cover the pinned arc")?;

    Ok(Fixture {
        init_equ,
        config,
        packed,
    })
}

/// FNV-1a over the raw bits of every returned delta component, in order.
///
/// A digest rather than 100+ literals: the DENSE case returns 3 x 8 x 6 = 144
/// f64s. FNV-1a is used because it is three lines and order-sensitive; nothing
/// here needs a cryptographic property, only that a single flipped mantissa bit
/// anywhere in the output changes the printed value.
fn digest(values: impl IntoIterator<Item = f64>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values {
        for byte in v.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// The batch entry `batch.rs` exposes and the UKF-sigma stack calls, with
/// `enable_events = false` hardcoded inside it. Three sigma points, because the
/// defect being guarded is per-arc and a single arc cannot show a per-arc state
/// leak between arcs.
///
/// `am_ratio_arr` IS REQUIRED to reach the loop under test, and this is not
/// incidental. `integrate_batch_native_into` routes to
/// `ReusableFinalNoEventIntegrator` — a different function entirely, with its
/// own restart loop — whenever there is no per-state ballistic override, and it
/// does so regardless of how many eval times were asked for. Only a per-state
/// array sends the batch directly through `integrate_adaptive` with interpolated
/// output, and into `integrate_sampled_inner`'s
/// `'rect_loop`. Verified by
/// measurement, not by reading: without the array this arc allocates 2 blocks
/// over 128 KiB per batch and with it 48, which is 2 per segment across 3 arcs
/// of 8 segments.
fn run_batch_dense(fixture: &Fixture) -> anyhow::Result<(Vec<f64>, usize)> {
    // One eval time at the end of each rectification segment. Bound to the
    // production cap: a test-local 5_400.0 sat here until R21 and pinned
    // nothing, so moving `MAX_RECTIFICATION_SEGMENT_S` would have left this
    // sampling every segment boundary but the integrator's.
    anyhow::ensure!(
        TOF_S / MAX_RECTIFICATION_SEGMENT_S == f64::from(SEGMENTS),
        "this file's layouts assume {SEGMENTS} segments; TOF_S {TOF_S} over \
         MAX_RECTIFICATION_SEGMENT_S {MAX_RECTIFICATION_SEGMENT_S} is not that"
    );
    let t_eval: Vec<f64> = (1..=SEGMENTS)
        .map(|i| f64::from(i) * MAX_RECTIFICATION_SEGMENT_S)
        .collect();

    // Three sigma-like points: the nominal state and two one-ULP neighbours.
    // Distinct enough to be distinct arcs, close enough to stay on one orbit.
    let mut init_states = Vec::with_capacity(18);
    for k in 0..3u64 {
        let mut equ = fixture.init_equ;
        let base_bits = equ[0].to_bits();
        let sigma_bits = base_bits
            .checked_add(k)
            .with_context(|| format!("sigma neighbor overflows: {base_bits:#018x} + {k}"))?;
        equ[0] = f64::from_bits(sigma_bits);
        init_states.extend_from_slice(&equ);
    }

    let am_ratio = [fixture.config.am_ratio; 3];
    let mut batch_config = fixture.config;
    batch_config.eps = EPS;
    batch_config.integrator_method = authority_stepper();
    let out =
        lightyear_odeint_rs::integrate_batch_native(lightyear_odeint_rs::BatchPropagationRequest {
            initial_equinoc_states: &init_states,
            t_eval: &t_eval,
            t0_s: 0.0,
            t_final_s: TOF_S,
            epoch_jd: JD0,
            force_config: batch_config,
            ballistics: lightyear_odeint_rs::BatchBallistics {
                am_ratio: Some(&am_ratio),
                cd: None,
                cr: None,
            },
        })
        .context("the batch sigma stack must propagate")?;
    let n = out.len();
    Ok((out, n))
}

/// `evaluate.rs`'s entry, `enable_events = false`.
fn run_sampled(fixture: &Fixture, t_eval: &[f64]) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let gravity = ScalarGravityAssets::new(Arc::clone(&fixture.packed));
    let context = ScalarPropagationContext::new(JD0, Arc::new(fixture.config), gravity);
    let result = integrate_adaptive(
        ScalarPropagationRequest::new(&context, fixture.init_equ, t_eval, 0.0, TOF_S)
            .with_events(false),
    )
    .context("sampled rect-loop propagation census failed")?;
    anyhow::ensure!(
        !result.terminal_event_fired && !result.max_steps_exceeded,
        "the sampled rect-loop arc must complete: {}",
        result.terminal_event_name
    );
    let flat: Vec<f64> = result
        .states
        .iter()
        .flat_map(|s| s.iter().copied())
        .collect();
    Ok((result.times, flat))
}

// ---------------------------------------------------------------------------
// Pinned digests. Copy replacements VERBATIM from the RECT_LOOP_PIN lines.
//
// Re-pinned 2026-08-06 on the exactness authority (Apple arm64) after a full
// bisect of 756d690..11cf4e4 (524 commits, 34 probes). These pins had been
// red since 2026-08-03 and were being read as an "authorized red"; the bisect
// showed that was a stale baseline, not an authorization. FIVE digest states
// exist across the range, every transition commit-exact and every one an
// intended eclipse-physics change:
//
//   state P (the previous constants)  ..2b9973d   captured 2026-07-28 @ d75d6b8
//   P -> A  074b9cf  feat(hybrid)!: add binary eclipse coordinator
//   A -> B  a6b4de4  fix(physics): commit eclipse roots continuously
//   B -> D  3fd4848  perf(physics): run root-transaction legs at production
//                    step size -- its own message says "deliberate tripwire
//                    displacement logged, NOT RE-PINNED", which is where the
//                    standing red began
//   D -> C  a4365fa  perf(physics): raise root refinement clamp to 10 s
//                    (partially reverts 3fd4848; accuracy 0.006622 m; B500
//                    event-0 gate passed)
//   state C = the constants below, stable across 300+ commits since.
//   C -> E  2026-08-06  perf(solver): open short segments at span/2
//                    (`lightyear_odeint_rs::odesolve::solver::SHORT_SPAN_H0_S`). All three
//                    cases move because all three run segments under 60 s, so
//                    all three take the new opening step. Intended derivative
//                    change; `strict_hf_production_arc_accuracy` stayed green.
//
//     DENSE   0xb770_9c9d_cbf3_4bfa -> 0x35c2_ab9b_b8e3_b79a
//     SPARSE  0x3c9e_b7cc_178d_55c2 -> 0x81f8_01a4_96bf_faa9
//     MIXED   0xf7e0_9dfc_b16b_3571 -> 0x023d_b40e_14a3_a3c1
//
//   E -> F  2026-08-07  perf(jb2008): `jb_tsub_l`'s two `powf(2.5)` calls
//                    became `x * x * sqrt(x)`.
//
//                    NOT the species round-trip retirement that landed in the
//                    same commit. This file hardcodes `atm_model: 4`, the EXACT
//                    profile, and that change is gated off there by
//                    `QuadratureProfile::RETIRE_SPECIES_ROUND_TRIP` so the
//                    sealed Orekit fixture stays bit-green. These three digests
//                    move from the `powf` substitution alone, which applies to
//                    both profiles.
//
//                    Worth knowing while reading this file: `atm_model: 4` is
//                    not what production flies. Compiled science reads 5, the
//                    x4 approximation, and the species change DOES apply there.
//                    `strict_hf_pin`'s V3 pin is the one that watches it.
//
//                    The substitution is 2 ULP against `powf` over `[0, 1]`,
//                    the only domain `jb_tsub_l` produces, and it moves 4 of the
//                    278 densities in `jb_rs`'s corpus. Intended numerical
//                    change; the sealed Orekit fixture and
//                    `strict_hf_production_arc_accuracy` both stayed green.
//
//     DENSE   0x35c2_ab9b_b8e3_b79a -> 0xc827_4107_2422_5d6a
//     SPARSE  0x81f8_01a4_96bf_faa9 -> 0x9bec_0499_b8ce_335b
//     MIXED   0x023d_b40e_14a3_a3c1 -> 0xedb8_4bf5_85b6_2d9d
//
//   F -> G  2026-08-07  perf(physics): bound the eclipse Sun sweep by a
//                    per-grid supremum on its RATE
//                    (`LightyearRHS::eclipse_sun_direction_path_bound`)
//                    instead of summing exact great-circle steps between
//                    crossed ephemeris nodes. All three cases move because
//                    all three are eclipse-heavy and the eclipse scan
//                    subdivides against that bound, so a different bound
//                    puts the evaluation points in different places. The
//                    replacement is looser but still a valid upper bound --
//                    0.23% loose on the strict-HF arc, at most 7.0% anywhere
//                    in the grid -- and a looser upper bound prunes less and
//                    subdivides more, never the reverse. Intended derivative
//                    change; all three cases still complete with their shapes
//                    unchanged (144 / 1 / 4), i.e. no new misbracket, and
//                    `strict_hf_production_arc_accuracy` stayed green.
//
//                    READ THIS BEFORE PRICING EITHER LEVER: G OVERLAPS F.
//                    F made transcendentals cheaper; G deletes an `atan2`
//                    outright, so they compete for part of the same cost.
//                    The tell is in the digests below -- SPARSE and MIXED
//                    land on EXACTLY the values they held at state E, before
//                    F existed. G erases F's effect on those two cases
//                    entirely. Their wall figures are therefore NOT additive,
//                    and G's 5.39% was measured against E, not F.
//
//     DENSE   0xc827_4107_2422_5d6a -> 0x25b2_de36_a416_2ba8
//     SPARSE  0x9bec_0499_b8ce_335b -> 0xd0d1_296b_601d_1f3c   (= state E)
//     MIXED   0xedb8_4bf5_85b6_2d9d -> 0xa6fc_ccec_ef33_7543   (= state E)
//
// None of the seven claims "no numerical change", so no transition is a bug.
// THE RULE, from this incident: a commit that moves these digests re-pins
// them IN THE SAME COMMIT, old value -> new value -> cause. A red left
// standing becomes a stale baseline that hides the next real move -- this
// one stood for three days and cost a session-wide "authorized red" myth.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 2026-08-09, the equinoctial warm seed (`LightyearRHS::baseline_warm_offset`).
// All three digests moved together, re-pinned in the commit that moved them,
// from the RECT_LOOP_PIN lines the tests print:
//
//     DENSE   0x25b2_de36_a416_2ba8 -> 0x92d3_4ce9_7cb7_039f
//     SPARSE  0xd0d1_296b_601d_1f3c -> 0xfb05_7361_c5d4_d4a8
//     MIXED   0xa6fc_ccec_ef33_7543 -> 0xe84b_670f_2917_b3b1
//
// CAUSE, and it is a derivative change rather than a physics change: the
// equinoctial longitude solve is seeded from the previous call's converged
// root instead of from the mean longitude, which drops the pinned arc from
// 25,937 Halley passes to 17,084. The loop's exit test is on the STEP, so a
// different seed reaches a root that differs in the last ULP, and every
// trajectory downstream of it moves by that much. No output SHAPE moved --
// all three `len` assertions were green throughout, and the arm structure
// each case exists to exercise is untouched.
//
// The accuracy question belongs to `strict_hf_pin`'s two 1 m gates, which
// stayed green: truncation error read 0.010067 m -> 0.042518 m and
// 0.016336 m -> 0.061242 m, both moves below that metric's own 0.2 m noise
// floor and both readings ~16x inside the bound.
// ---------------------------------------------------------------------------
const PIN_DENSE_LEN: usize = 144;
// ---------------------------------------------------------------------------
// 2026-08-09, the stage-baseline prefill
// (`LightyearRHS::prefill_stage_baselines`). All three digests moved together
// again, re-pinned in the commit that moved them, from the RECT_LOOP_PIN lines
// the tests print:
//
//     DENSE   0x92d3_4ce9_7cb7_039f -> 0x839b_ec54_cf47_894f
//     SPARSE  0xfb05_7361_c5d4_d4a8 -> 0xb783_7e11_a082_a1d1
//     MIXED   0xe84b_670f_2917_b3b1 -> 0x1a89_3022_9529_8576
//
// CAUSE, and it is the warm seed's successor rather than a new mechanism: an
// RK step's 16 stage baselines are now resolved four at a time before the
// stage loop runs, and the four lanes of a pack share ONE incoming seed
// instead of chaining. Three lanes in four therefore start from a different
// point than the serial order gave them, and the same step-exit test lands on
// a root that differs in the last ULP. No output SHAPE moved -- all three
// `len` assertions stayed green (4 / 1 / 144), and the arm structure is
// untouched. The accuracy question is `strict_hf_pin`'s: both of its 1 m gates
// stayed green across this change.
//
// ---------------------------------------------------------------------------
// RE-BASELINED 2026-08-09 — R26 Vern7 swap
//
//     DENSE   0x839b_ec54_cf47_894f -> 0xff91_33b8_d603_e4a2
//     SPARSE  0xb783_7e11_a082_a1d1 -> 0x7119_2250_b273_a5a1
//     MIXED   0x1a89_3022_9529_8576 -> 0x1fd7_bf92_ab7b_fbef
//
// CAUSE: compiled science moved `integrator_method` "vern9" -> "vern7", and
// this file now RESOLVES that token instead of hardcoding `Vern9`. So the arc
// is integrated by a different tableau and every returned delta changes. This
// is a whole-trajectory change, not a last-ULP one like the entry above.
//
// The stepper hardcode is itself the finding. Before this commit these three
// digests would have stayed GREEN through the swap, still guarding a Vern9
// trajectory the campaign had stopped flying — a bit pin that cannot see a
// change to the very thing it exists to watch. The sampled loop pinned here is
// reached from the UKF-sigma batch route, which production does fly.
//
// No output SHAPE moved: all three `len` assertions stayed green (144 / 1 / 4),
// so the arm structure and segment count are untouched and only the numbers
// inside them differ. Accuracy remains `strict_hf_pin`'s question; both of its
// 1 m gates stayed green across the swap.
//
// ---------------------------------------------------------------------------
// RE-BASELINED 2026-08-10 — R44 one-`atan2` JB2008 hour angle
//
//     DENSE   0xff91_33b8_d603_e4a2 -> 0xfb3b_73f4_ceb0_8044
//     SPARSE  0x7119_2250_b273_a5a1 -> UNMOVED
//     MIXED   0x1fd7_bf92_ab7b_fbef -> UNMOVED
//
// CAUSE: the JB2008 adapter used to hand the kernel two right ascensions from
// two `atan2` calls, and the kernel subtracted them to get the satellite's hour
// angle. It now computes that difference directly, as one `atan2` of a
// cross/dot pair. The result is the same angle to rounding, but it lands on a
// different binary64 representative — the old form produced `h` in `(-2π, 2π)`
// and the new one in `[-π, π]` — so the density moves in its last digits and
// every drag-perturbed delta below it moves with it.
//
// ONLY DENSE MOVED, and that asymmetry is the expected one rather than a
// surprise: DENSE puts an eval time in all eight 5400 s segments, so it is the
// only arm whose every segment integrates through drag. SPARSE (one eval at
// `tf`) and MIXED (two interior segments) both stayed bit-identical, which also
// says the change is confined to the drag path and did not perturb the arm
// structure or the segmentation.
//
// No output SHAPE moved: all three `len` assertions stayed green (144 / 1 / 4).
// Accuracy is `strict_hf_pin`'s question and both of its 1 m gates stayed
// green; the V3 endpoint moved 6.1 µm and the truncation metric went
// 0.099689 -> 0.098470 m, inside the established 0.0964--0.1030 band, so the
// 0.15 m sizing tripwire did not need re-sizing.
//
// The equivalence is gated, not asserted: `jb_rs`'s
// `one_atan2_hour_angle_matches_the_two_atan2_pair_in_every_observable` checks
// both quantities the hour angle actually reaches.
//
// ---------------------------------------------------------------------------
// RE-BASELINED 2026-08-10 — R44 unscaled hypot in `geodetic_altitude_km`
//
//     DENSE   0xfb3b_73f4_ceb0_8044 -> 0x951d_4a75_3d1c_517f
//     SPARSE  0x7119_2250_b273_a5a1 -> 0x7ab2_1fbb_aaa6_f549
//     MIXED   0x1fd7_bf92_ab7b_fbef -> 0xb9c2_b8d9_9b0b_3fb8
//
// CAUSE: `geodetic_altitude_km`'s three `f64::hypot` calls became
// `(a*a + b*b).sqrt()`. `hypot` is correctly rounded and the squared form is
// not, so the geodetic altitude moves in its last digits, and it feeds JB2008's
// `sat_altitude_m` on every drag evaluation.
//
// ALL THREE MOVED, where the entry above moved DENSE alone. That is the tell
// that this change sits in a different place: the hour angle is only consumed
// by the sampled arm's segments, but the altitude is consumed by every drag
// evaluation in every arm, final-only segments included. A change here that had
// left SPARSE or MIXED green would have meant the altitude was NOT reaching
// those arms, which would itself be the bug.
//
// No output SHAPE moved: all three `len` assertions stayed green (144 / 1 / 4).
// The V3 endpoint moved 7.56 µm against its 1 cm tripwire and the truncation
// metric went 0.098470 -> 0.099075 m, still inside the 0.0964--0.1030 band.
//
// The premise that licenses dropping the scaling — that every operand is
// kilometre-scale, so `hypot`'s over/underflow handling is dead code — is
// pinned by `rhs::geodetic_hypot_range_tests::
// geodetic_altitude_km_operands_stay_in_the_squarable_range` rather than
// asserted in prose.
// ---------------------------------------------------------------------------
// NOT RE-BASELINED 2026-08-11 — R57 model-7 atmosphere levers, recorded here
// because the silence is the finding.
//
// `jb_rs::jb2008`'s `RETIRE_ZR_ROUND_TRIP` and `DLRSL_ZERO_ABOVE_KM` moved the
// flown density on 31 of 278 corpus rows, worst 3.3899e-13 relative. All three
// digests below are UNCHANGED, and they had to be: this file hardcodes
// `atm_model: 4` in its fixture, so it is blind to every model-7 change by
// construction. See `strict_hf_pin`'s note on `V3_PINNED_POS_KM` — the
// instrument that detects an atmosphere edit is `jb2008_libm_probe`'s
// per-profile bit dumps, diffed across two trees, and a green run of THIS file
// is not evidence about the atmosphere the campaign flies.
//
// The two entries above both close by citing "the 0.0964--0.1030 band" for the
// V3 truncation metric. That range is a 96-draw EMPIRICAL distribution and not
// a bound, and R57 read 0.095837 m, 0.6 mm below its floor, with the sizing
// tripwire green and its margin larger than before. The band is defined and
// re-stated at `strict_hf_pin`'s `V3_SIZING_TRIPWIRE_M`; read it there rather
// than treating "inside the band" as a pass condition.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// RE-PINNED 2026-08-11 -- stage-prefill node filter. All three moved together.
//
//   DENSE  0x951d4a753d1c517f -> 0x5938da3a86462226
//   SPARSE 0x7ab21fbbaaa6f549 -> 0xbcea604d8881654c
//   MIXED  0xb9c2b8d99b0b3fb8 -> 0x5ceaea5a3c70d7a7
//
// Cause: `integrator.rs::prefill_stage_times` now hands the prefill only the
// stage nodes whose baselines can be read -- dropping `c[0] == 0.0` and the
// duplicate `c[9] == c[8] == 1.0`. Eight nodes pack into two x4 solves instead
// of three, and node 8 lands in a different pack, so it is solved from a
// different warm-start seed and moves in the last ULP. Counted, not estimated:
// prefill packs per propagation 2004 -> 1336 (-33.3%), steps and rhs_evals
// both UNCHANGED at 666 / 6742.
//
// All three moving together is the expected shape -- the change is in the
// derivative every arm integrates. `strict_hf_pin` did NOT trip (4 nm) and was
// re-baselined anyway; `jb_rs`'s `fitted_v7_density_pin` did NOT move and must
// not, since no atmosphere code is touched -- verified green at its existing
// digests in the same run.
// ---------------------------------------------------------------------------
// RE-PINNED 2026-08-26 -- frame-drag correction. All three moved together.
//
//   DENSE  0x97769574_3f6320fd -> 0xc9fadc7315f7a572
//   SPARSE 0x13d80ea4974b534a  -> 0xe060171d2349caaf
//   MIXED  0x7efef8d1f5eceff8  -> 0x2611e32abb7d054e
//
// Cause: `1eed76ce fix(hybrid): bind retained mass and frame drag`. Drag is an
// RHS term, so correcting it moves the derivative every arm integrates, and
// all three arms moving together is the expected shape.
//
// THE RULE ABOVE WAS MISSED, and this entry exists to say so rather than to
// hide it. `1eed76ce` moved these digests and did not re-pin them in the
// commit that moved them, so the file sat red from that commit onward. It was
// found during the C12 closeout by running the pins, not by a gate: nothing
// between that commit and this one ran `--features bitpin`.
//
// Bisected, not assumed. Green at `b2636c09`, `18a0b860` and `53870cdd`; red
// at `1eed76ce` and at every commit after it. The three replacement digests
// below reproduced identically in two independent release runs on this host,
// once before and once after the retired force-config interner experiment --
// which was also evidence that experiment was bit-neutral.
//
// `strict_hf_pin` did NOT move and was verified green 5/5 in the same runs; it
// must not be re-baselined here.
// ---------------------------------------------------------------------------
const PIN_DENSE_DIGEST: u64 = 0xc9fa_dc73_15f7_a572;

const PIN_SPARSE_LEN: usize = 1;
const PIN_SPARSE_DIGEST: u64 = 0xe060_171d_2349_caaf;

const PIN_MIXED_LEN: usize = 4;
const PIN_MIXED_DIGEST: u64 = 0x2611_e32a_bb7d_054e;

/// Every segment carries an eval time: the sampled arm, eight times over.
#[test]
#[cfg_attr(
    not(feature = "bitpin"),
    ignore = "bit pin: needs production flags; fp-contract makes debug a different trajectory"
)]
fn rect_loop_dense_sampled_is_pinned() -> anyhow::Result<()> {
    let fixture = fixture()?;
    let (out, n) = run_batch_dense(&fixture)?;
    let d = digest(out.iter().copied());
    println!("RECT_LOOP_PIN case=DENSE len={n} digest={d:#018x}");
    anyhow::ensure!(n == PIN_DENSE_LEN, "DENSE output shape moved");
    anyhow::ensure!(
        d == PIN_DENSE_DIGEST,
        "DENSE sampled rect-loop output moved. Every 5400 s segment of this arc \
         takes the sampled arm, so a per-segment RHS that carried a stale Encke \
         baseline lands here first. Re-baseline from the RECT_LOOP_PIN line \
         above only if the derivative was intended to change."
    );
    Ok(())
}

/// One eval time, at `tf`. Seven segments take the final-only arm, the eighth
/// takes the sampled arm.
#[test]
#[cfg_attr(
    not(feature = "bitpin"),
    ignore = "bit pin: needs production flags; fp-contract makes debug a different trajectory"
)]
fn rect_loop_sparse_final_only_is_pinned() -> anyhow::Result<()> {
    let fixture = fixture()?;
    let (times, flat) = run_sampled(&fixture, &[TOF_S])?;
    let d = digest(flat.iter().copied());
    println!(
        "RECT_LOOP_PIN case=SPARSE len={} digest={d:#018x} last_t={:.6}",
        times.len(),
        times.last().copied().unwrap_or(f64::NAN)
    );
    anyhow::ensure!(times.len() == PIN_SPARSE_LEN, "SPARSE output shape moved");
    anyhow::ensure!(
        d == PIN_SPARSE_DIGEST,
        "SPARSE rect-loop output moved: seven final-only segments feeding one \
         sampled segment."
    );
    Ok(())
}

/// Eval times inside two interior segments only, so the final-only and sampled
/// arms INTERLEAVE inside one loop. The arms are the two places a hoisted RHS
/// would be shared, so this is the layout that prices sharing them.
#[test]
#[cfg_attr(
    not(feature = "bitpin"),
    ignore = "bit pin: needs production flags; fp-contract makes debug a different trajectory"
)]
fn rect_loop_mixed_arms_are_pinned() -> anyhow::Result<()> {
    let fixture = fixture()?;
    // Segment k spans ((k-1)*5400, k*5400]. Times below sit in segments 3 and
    // 6, leaving 1, 2, 4, 5, 7 and 8 on the final-only arm.
    let t_eval = [13_000.0, 16_000.0, 30_000.0, 32_000.0];
    let (times, flat) = run_sampled(&fixture, &t_eval)?;
    let d = digest(flat.iter().copied());
    println!(
        "RECT_LOOP_PIN case=MIXED len={} digest={d:#018x}",
        times.len()
    );
    anyhow::ensure!(times.len() == PIN_MIXED_LEN, "MIXED output shape moved");
    anyhow::ensure!(
        d == PIN_MIXED_DIGEST,
        "MIXED rect-loop output moved. This case interleaves the final-only and \
         sampled arms, which is exactly where a shared per-propagation RHS would \
         carry a baseline across an arm boundary."
    );
    Ok(())
}

/// The step-size carry (`hcarry_*` in `integrator.rs`) must be scoped to ONE
/// propagation: every segment of this loop passes `SegmentBoundary::Rebased`,
/// so without the reset at `integrate_sampled_rectified_path`'s entry the
/// FIRST segment of an arc would open at whatever `h` the previous arc on the
/// same thread exited with. Under rayon that predecessor is
/// schedule-dependent, which would make output bits nondeterministic — so this
/// guard is about determinism, not accuracy.
///
/// The test is deliberately relative (same process, same flags, no pinned
/// digest): run an arc on a fresh thread, run a different arc to leave a
/// carried `h` behind, then run the first arc again on the same thread. With
/// the reset in place the two runs are bit-identical; delete the reset and the
/// third run's opening segment consumes the polluter's exit `h`.
#[test]
#[cfg_attr(
    not(feature = "bitpin"),
    ignore = "runs three full arcs; debug-profile physics is too slow for the default battery"
)]
fn hcarry_reset_scopes_carry_to_one_propagation() -> anyhow::Result<()> {
    let mut fixture = fixture()?;
    // SRP off, deliberately: an SRP-effective config is routed to the binary
    // eclipse coordinator before the `enable_events` dispatch ever runs
    // (`integrate_adaptive`'s `effective_scalar_srp` branch), and that entry
    // has its own reset. Clearing the flag is what makes these arcs reach
    // `integrate_sampled_rectified_path` — the entry this test guards.
    fixture.config.force_flags &= !ForceFlags::SRP;
    let t_eval = [13_000.0, 16_000.0, 30_000.0, 32_000.0];
    let (_, baseline) = run_sampled(&fixture, &t_eval)?;
    // Full-arc polluter: leaves its exit `h` in this thread's carry slot.
    let (_, polluter) = run_sampled(&fixture, &[TOF_S])?;
    anyhow::ensure!(
        !polluter.is_empty(),
        "the polluter arc must actually propagate"
    );
    let (_, rerun) = run_sampled(&fixture, &t_eval)?;
    anyhow::ensure!(
        digest(baseline.iter().copied()) == digest(rerun.iter().copied()),
        "an arc's bits moved because another arc ran before it on the same \
         thread: the step-size carry leaked across a propagation entry. The \
         `hcarry_reset()` call at `integrate_sampled_rectified_path`'s entry \
         is missing or no longer reachable."
    );
    Ok(())
}
