use std::cell::RefCell;

use lightyear_odeint_rs::probe as lyprobe;

#[derive(Clone, Copy, Debug, Default)]
pub struct HfMassSolveProfile {
    pub hf_miss_calls_total: u64,
    pub hf_miss_elapsed_s_total: f64,
    pub hf_validate_initial_calls: u64,
    pub hf_validate_initial_elapsed_s: f64,
    pub hf_validate_repair_calls: u64,
    pub hf_validate_repair_elapsed_s: f64,
    pub hf_validate_refine_calls: u64,
    pub hf_validate_refine_elapsed_s: f64,
    pub hf_validate_refine_iterations: u64,
    pub hf_full_bracket_calls: u64,
    pub hf_full_bracket_elapsed_s: f64,
    pub hf_full_refine_calls: u64,
    pub hf_full_refine_elapsed_s: f64,
    pub hf_full_refine_iterations: u64,
    pub hf_upper_bracket_shrink_iterations: u64,
    pub hf_lf_fallback_count: u64,
    pub detmass_anchor_contract_version: u32,
    pub detmass_anchor_shift_norm_km: f64,
    pub detmass_anchor_internal_reference_used: bool,
}

#[derive(Clone, Copy)]
pub(super) enum HfProfileStage {
    ValidateInitial,
    ValidateRepair,
    ValidateRefine,
    FullBracket,
    FullRefine,
}

const HF_COUNTERS_ENABLED: bool = true;

thread_local! {
    static LAST_HF_MASS_SOLVE_PROFILE: RefCell<HfMassSolveProfile> =
        RefCell::new(HfMassSolveProfile::default());
}

#[inline]
pub(super) const fn hf_counters_enabled() -> bool {
    HF_COUNTERS_ENABLED
}

#[inline]
pub(super) fn hf_profile_reset() {
    LAST_HF_MASS_SOLVE_PROFILE.with(|slot| {
        *slot.borrow_mut() = HfMassSolveProfile::default();
    });
}

#[inline]
pub(super) fn hf_profile_record(stage: HfProfileStage, elapsed_s: f64) {
    let elapsed = if elapsed_s.is_finite() && elapsed_s >= 0.0 {
        elapsed_s
    } else {
        0.0
    };
    let stage_index = match stage {
        HfProfileStage::ValidateInitial => lyprobe::STAGE_VALIDATE_INITIAL,
        HfProfileStage::ValidateRepair => lyprobe::STAGE_VALIDATE_REPAIR,
        HfProfileStage::ValidateRefine => lyprobe::STAGE_VALIDATE_REFINE,
        HfProfileStage::FullBracket => lyprobe::STAGE_FULL_BRACKET,
        HfProfileStage::FullRefine => lyprobe::STAGE_FULL_REFINE,
    };
    lyprobe::bump_stage(stage_index);
    LAST_HF_MASS_SOLVE_PROFILE.with(|slot| {
        let mut p = slot.borrow_mut();
        p.hf_miss_calls_total = p.hf_miss_calls_total.saturating_add(1);
        p.hf_miss_elapsed_s_total += elapsed;
        match stage {
            HfProfileStage::ValidateInitial => {
                p.hf_validate_initial_calls = p.hf_validate_initial_calls.saturating_add(1);
                p.hf_validate_initial_elapsed_s += elapsed;
            }
            HfProfileStage::ValidateRepair => {
                p.hf_validate_repair_calls = p.hf_validate_repair_calls.saturating_add(1);
                p.hf_validate_repair_elapsed_s += elapsed;
            }
            HfProfileStage::ValidateRefine => {
                p.hf_validate_refine_calls = p.hf_validate_refine_calls.saturating_add(1);
                p.hf_validate_refine_elapsed_s += elapsed;
            }
            HfProfileStage::FullBracket => {
                p.hf_full_bracket_calls = p.hf_full_bracket_calls.saturating_add(1);
                p.hf_full_bracket_elapsed_s += elapsed;
            }
            HfProfileStage::FullRefine => {
                p.hf_full_refine_calls = p.hf_full_refine_calls.saturating_add(1);
                p.hf_full_refine_elapsed_s += elapsed;
            }
        }
    });
}

#[inline]
pub(super) fn hf_profile_inc_validate_refine_iteration() {
    LAST_HF_MASS_SOLVE_PROFILE.with(|slot| {
        let mut p = slot.borrow_mut();
        p.hf_validate_refine_iterations = p.hf_validate_refine_iterations.saturating_add(1);
    });
}

#[inline]
pub(super) fn hf_profile_inc_full_refine_iteration() {
    LAST_HF_MASS_SOLVE_PROFILE.with(|slot| {
        let mut p = slot.borrow_mut();
        p.hf_full_refine_iterations = p.hf_full_refine_iterations.saturating_add(1);
    });
}

/// Record that the validate-only path gave up and ran the authoritative full-HF
/// bracket instead.
///
/// `hf_lf_fallback_count` was declared and compared by tests but never
/// incremented anywhere, so every comparison of it was between two zeroes and
/// proved nothing. It matters because validate-only is the default and is the
/// whole reason a solve costs O(5-10) HF calls instead of O(30+): if it were
/// quietly falling back, the campaign would be paying full price while the
/// code claimed otherwise.
///
/// # THE FIRST WIRE-UP DID NOT WORK EITHER, AND ITS RECEIPT IS VOID
///
/// The call was placed at the TOP of `run_full_hf`, one line above a re-entry
/// into `solve_single_event_hf_internal` whose first statement is
/// [`hf_profile_reset`]. Every increment was erased before any caller could
/// snapshot it, so the field stayed structurally zero and the batch-vs-serial
/// identity sweep kept comparing `0 == 0` -- the same shape as the "never
/// incremented" bug it was meant to close.
///
/// # THERE ARE TWO RESETS ON THIS PATH, NOT ONE
///
/// `solve_single_event_hf_validate_only` also wipes the profile in its normal
/// course: its step 1 LF seed calls `solve_single_event_hf_with_status`, which
/// is `solve_single_event_hf_internal`, which resets. So ANY thread-local
/// profile field incremented in that function before the LF seed returns is
/// lost, whether or not a fallback happens. Only the process-global `lyprobe`
/// counters (`bump_stage(STAGE_ROWS_VALIDATE_ONLY)`) survive there, because they
/// are not part of this struct.
///
/// Measured while poison-proving: an increment at the top of
/// `solve_single_event_hf_validate_only` reads back as `1` at the increment and
/// `0` at the caller, five times out of five. Put new counters for this path
/// AFTER the LF seed, and prove the position with a test rather than by reading
/// the control flow.
///
/// The receipt this comment used to carry -- "239 validate-only entries, 0
/// fallbacks" -- was read off that dead counter and could not have reported
/// anything else. It is withdrawn. The surviving evidence for the same claim is
/// `hf_full_bracket_calls`, which is recorded AFTER the nested reset and which
/// read zero over 9,659 production strict-HF rows at era 286dad1.
///
/// Moved below the nested call 2026-08-10, and
/// `test_validate_only_hf_runs_full_hf_solve_when_lf_seed_is_invalid` now
/// asserts a nonzero count on a fixture that forces the fallback, so the field
/// can fail.
#[inline]
pub(super) fn hf_profile_inc_lf_fallback() {
    LAST_HF_MASS_SOLVE_PROFILE.with(|slot| {
        let mut p = slot.borrow_mut();
        p.hf_lf_fallback_count = p.hf_lf_fallback_count.saturating_add(1);
    });
}

#[inline]
pub(super) fn hf_profile_inc_upper_bracket_shrink() {
    LAST_HF_MASS_SOLVE_PROFILE.with(|slot| {
        let mut p = slot.borrow_mut();
        p.hf_upper_bracket_shrink_iterations =
            p.hf_upper_bracket_shrink_iterations.saturating_add(1);
    });
}

#[inline]
pub(super) fn hf_profile_snapshot() -> HfMassSolveProfile {
    LAST_HF_MASS_SOLVE_PROFILE.with(|slot| *slot.borrow())
}

#[inline]
pub(super) fn hf_profile_set_anchor_diagnostics(
    anchor_contract_version: u32,
    anchor_shift_norm_km: f64,
    anchor_internal_reference_used: bool,
) {
    LAST_HF_MASS_SOLVE_PROFILE.with(|slot| {
        let mut p = slot.borrow_mut();
        p.detmass_anchor_contract_version = anchor_contract_version;
        p.detmass_anchor_shift_norm_km = anchor_shift_norm_km;
        p.detmass_anchor_internal_reference_used = anchor_internal_reference_used;
    });
}
