//! Native finalize glue for the `pc_fraction_grid` stage-driver port.
//!
//! SLICE 1 of the fraction-grid native port (design doc
//! `plans/2026-07-16-fraction-grid-native-port-design.md` IN THE ORACLE REPO;
//! it has never been tracked here).
//! This module owns the closed-enum classifier that mirrors the Python
//! `_required_mass_kind` exactly. Later slices grow it into the full
//! `fraction_grid_finalize_mf_into` verdict driver; slice 1 is the round-trip
//! proof only.
//!
//! The reason/kind code tables are FROZEN. Their live authority is
//! `nd_pipeline::physics::reason::REASON_CODES` (24 entries, index ==
//! `reason_code`), which the private consts below must stay aligned with;
//! `REASON_CODE_COUNT` is consumed cross-crate by `nd_pipeline`'s
//! `solver_qualification`. The classifier never
//! invents a kind: an out-of-table `reason_code` (the design 2.3 code `-1`
//! sentinel, or any value outside `[0, REASON_CODE_COUNT)`) is a fail-loud error
//! (`fraction_grid_native_unknown_reason_code`), never a silent degrade.
//!
//! This module previously sourced the tables from `fraction_grid_parity.py`
//! `REASON_CODES`/`KIND_CODES` and cited `dust_phase.py` line numbers
//! throughout, calling the L1 parity harness the thing that catches drift.
//! Retired 2026-08-07 (see the sweep note on `mf_verdict_label_det_prefix`):
//! neither Python module exists to check against, and the surviving constant
//! is spelled `KIND_TO_CODE`, not `KIND_CODES`. The behavioural contract is
//! unchanged and still gated — by the captured oracle rows in
//! `nd_pipeline/tests/fixtures/oracle_rows_{3,24}event.json`, not by Python.

/// Number of members in the frozen `REASON_CODES` table.
/// Index range is `[0, REASON_CODE_COUNT)`; anything else is unknown → fail-loud.
pub const REASON_CODE_COUNT: i32 = 24;

// `required_mass_kind` closed enum (index == code).
pub const KIND_EXACT: i32 = 0;
pub(crate) const KIND_LOWER_BOUND: i32 = 1;
pub(crate) const KIND_PHYSICAL_INFEASIBLE: i32 = 2;
pub const KIND_UNAVAILABLE: i32 = 3;

// Reason codes consulted by the classifier / MF verdict guards (indices into
// REASON_CODES; the frozen table is mirrored in `nd_pipeline::physics::reason`).
pub const REASON_OK: i32 = 0;
pub const REASON_SAFE_BY_DEFAULT: i32 = 1;
const REASON_DETERMINISTIC_MASS_PHYSICS_LIMITED: i32 = 2;
pub const REASON_DETERMINISTIC_MASS_INVALID: i32 = 3;
const REASON_PROB_MASS_FLOOR_HARD_LIMIT: i32 = 4;
const REASON_PROB_MASS_HARD_LIMIT: i32 = 5;
pub const REASON_PROB_MASS_INVALID: i32 = 6;
const REASON_ATMOSPHERIC_GUARD: i32 = 8;
const REASON_DETERMINISTIC_MASS_HARD_LIMIT: i32 = 22;

// `det_status_label_code` routing (design 2.1). The
// native detmass status label collapses to one of three routing codes; the
// verdict guard consults this BEFORE the fidelity branch, exactly as Python.
pub const LABEL_OTHER: i32 = 0;
pub const LABEL_SAFE_BY_DEFAULT: i32 = 1;
pub const LABEL_PHYSICS_LIMITED: i32 = 2;

// `row_state_code` (design 2.1). Slice 4 grows the driver to own the
// precheck-reject classes as well: `spec_none`/`cloud_none`/`centroid_reject`
// are `result is None` rows whose verdict is a pure passthrough of the
// already-resolved infeasible executed-dv + reason code (design 2.3), while
// `detmass_invalid_skip` re-enters the MF verdict tree (its invalid `det_mass`
// is caught by the deterministic-mass gate exactly as a prepared-ok row).
pub const ROW_STATE_PREPARED_OK: i32 = 0;
pub(crate) const ROW_STATE_SPEC_NONE: i32 = 1;
pub(crate) const ROW_STATE_CLOUD_NONE: i32 = 2;
pub(crate) const ROW_STATE_CENTROID_REJECT: i32 = 3;
pub(crate) const ROW_STATE_DETMASS_INVALID_SKIP: i32 = 4;
const PAR_THRESHOLD: usize = 256;

/// Classify the `required_mass_kind` for a finalize row.
///
/// Byte-for-byte port of `_required_mass_kind(feasible, reason, mass_kg)`
/// operating on the reason *code* rather than the reason string.
/// Returns a `KIND_*` code.
///
/// Fail-loud: `reason_code` outside `[0, REASON_CODE_COUNT)` (including the
/// design 2.3 `-1` unknown sentinel) returns an `Err`, never a fabricated kind.
///
/// # Errors
///
/// Returns an error when `reason_code` is outside the frozen table.
#[inline]
pub fn classify_required_mass_kind_core(
    feasible: bool,
    reason_code: i32,
    mass_kg: f64,
) -> anyhow::Result<i32> {
    if !(0..REASON_CODE_COUNT).contains(&reason_code) {
        return Err(anyhow::anyhow!(
            "fraction_grid_native_unknown_reason_code reason_code={reason_code}"
        ));
    }
    // exact iff feasible with a positive finite mass (matches the Python
    // `np.isfinite(mass_kg) and mass_kg > 0.0`; NaN/inf are non-finite, and
    // `NaN > 0.0` is false, exactly as in Rust).
    if feasible && mass_kg.is_finite() && mass_kg > 0.0 {
        return Ok(KIND_EXACT);
    }
    if reason_code == REASON_ATMOSPHERIC_GUARD {
        return Ok(KIND_PHYSICAL_INFEASIBLE);
    }
    let is_hard_or_physics_limited = reason_code == REASON_DETERMINISTIC_MASS_HARD_LIMIT
        || reason_code == REASON_DETERMINISTIC_MASS_PHYSICS_LIMITED
        || reason_code == REASON_PROB_MASS_FLOOR_HARD_LIMIT
        || reason_code == REASON_PROB_MASS_HARD_LIMIT;
    if is_hard_or_physics_limited && mass_kg.is_finite() && mass_kg > 0.0 {
        return Ok(KIND_LOWER_BOUND);
    }
    Ok(KIND_UNAVAILABLE)
}

/// One finalized MF verdict row (design 2.2): the four-tuple the Python
/// `_evaluate_prepared_candidate` returns plus the derived `required_mass_kind`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MfVerdict {
    pub executed_dv: f64,
    pub mass_kg: f64,
    pub feasible: bool,
    pub reason_code: i32,
    pub mass_kind_code: i32,
}

/// Port of the MF-branch verdict guard tree, the slice-2 target.
/// Reproduces, in order:
///   1. native-status label routing (`safe_by_default` / `physics_limited`),
///      short-circuiting BEFORE the fidelity branch (design R4);
///   2. the MF deterministic-mass validity gate (`< min_practical`, `<= 0`,
///      non-finite → `deterministic_mass_invalid`);
///   3. the deterministic floor hard-limit short-circuit
///      (`floor_mass >= hard_limit` → `prob_mass_floor_hard_limit`, Pc never
///      ran); and
///   4. the precomputed-Pc `total_mass` guards (invalid / hard-limit / ok).
///
/// The floor / Pc *math* stays Python for slice 2: `floor_mass` and
/// `pc_total_mass` cross as inputs (isolating the branch logic; slice 3 ports
/// the math). `is_finite` matches numpy `np.isfinite` (NaN and ±inf are
/// non-finite; comparisons against NaN are false), so the branch decisions are
/// byte-identical to the Python oracle. The `prob_mass_exception` try/except
/// wrapper is a Python-only control-flow reason that cannot be reconstructed
/// from precomputed values; the caller excludes those
/// rows from the parity compare (`pc_valid == 0` marks them).
///
/// `mass_kind_code` is derived through the slice-1 classifier so the emitted
/// row carries the same `required_mass_kind` the Python row-builder computes.
///
/// # Errors
///
/// Returns an error when the derived reason code is outside the frozen table.
#[inline]
pub fn fraction_grid_finalize_mf_row(
    executed_dv: f64,
    det_mass: f64,
    det_status_label_code: i32,
    floor_mass: f64,
    pc_total_mass: f64,
    pc_valid: bool,
    dust_hard_limit_kg: f64,
    min_practical: f64,
) -> anyhow::Result<MfVerdict> {
    let (executed_dv, mass_kg, feasible, reason_code) = mf_verdict_tuple(
        executed_dv,
        det_mass,
        det_status_label_code,
        floor_mass,
        pc_total_mass,
        pc_valid,
        dust_hard_limit_kg,
        min_practical,
    );
    let mass_kind_code = classify_required_mass_kind_core(feasible, reason_code, mass_kg)?;
    Ok(MfVerdict {
        executed_dv,
        mass_kg,
        feasible,
        reason_code,
        mass_kind_code,
    })
}

/// Raw slice-3 MF verdict row.
///
/// Floor plus precomputed-Pc `total_mass` is computed natively from the raw
/// operands via [`mf_verdict_tuple_raw`], then classified through the slice-1
/// classifier. This is the production row builder; the injected
/// [`fraction_grid_finalize_mf_row`] remains only for branch-logic unit tests.
///
/// # Errors
///
/// Returns an error for invalid raw mass inputs or an unknown reason code.
#[inline]
pub(crate) fn fraction_grid_finalize_mf_row_from_raw(
    executed_dv: f64,
    det_mass: f64,
    det_status_label_code: i32,
    pc_raw_mass: f64,
    pc_valid: bool,
    grain_mass_kg: f64,
    grains_per_independent_packet: u64,
    hit_probability: f64,
    dust_hard_limit_kg: f64,
    min_practical: f64,
) -> anyhow::Result<MfVerdict> {
    let (executed_dv, mass_kg, feasible, reason_code) = mf_verdict_tuple_raw(
        executed_dv,
        det_mass,
        det_status_label_code,
        pc_raw_mass,
        pc_valid,
        grain_mass_kg,
        grains_per_independent_packet,
        hit_probability,
        dust_hard_limit_kg,
        min_practical,
    )?;
    let mass_kind_code = classify_required_mass_kind_core(feasible, reason_code, mass_kg)?;
    Ok(MfVerdict {
        executed_dv,
        mass_kg,
        feasible,
        reason_code,
        mass_kind_code,
    })
}

/// Native port of `compute_release_mass_floor` (release_mass.py:303-328): the
/// best-case-capture finite-packet lower bound plus its confidence log term.
///
/// Mirrors Python exactly: a `hit_probability` outside `(0, 1)` or a
/// non-positive/non-finite inflation short-circuits to `(inf, inf)`; otherwise
/// the finite-packet bound runs with `capture_probability = 1.0` and
/// `target_probability = hit_probability`. Slice 3 R5: the bound squares by
/// explicit multiplication (`finite_packet_release_mass_bound_core` uses
/// `black_box(x.powi(2))`, a hardware fmul, NOT `x**2`/libm `pow`), matching the
/// Python side that was fixed to `chernoff_root_sum * chernoff_root_sum`
/// (memory note f6ccc5e1) so the released-packet ceil never flips by one packet
/// on x86 under `-fp-contract=on`. `probability_inflation` feeds a Python
/// diagnostic only (not `rows_hash`); the floor mass is the load-bearing value.
///
/// The bound only errors on inputs Python would also reject (`det_mass <= 0`,
/// invalid grain mass/packet count); the MF verdict path computes the floor only
/// after the deterministic-mass validity gate, so a valid `det_mass` never trips
/// it. An error propagates fail-loud rather than fabricating a mass.
///
/// # Errors
///
/// Returns an error for invalid deterministic mass, grain mass, or packet count.
pub fn release_mass_floor_core(
    det_mass: f64,
    grain_mass_kg: f64,
    grains_per_independent_packet: u64,
    hit_probability: f64,
) -> anyhow::Result<(f64, f64)> {
    let probability = hit_probability;
    // Mirrors Python `not np.isfinite(p) or not 0.0 < p < 1.0`.
    if !(probability.is_finite() && probability > 0.0 && probability < 1.0) {
        return Ok((f64::INFINITY, f64::INFINITY));
    }
    // `-np.log1p(-probability)`: numpy log1p == libm log1p == Rust ln_1p.
    let probability_inflation = -(-probability).ln_1p();
    if !probability_inflation.is_finite() || probability_inflation <= 0.0 {
        return Ok((f64::INFINITY, f64::INFINITY));
    }
    let bound = crate::finite_packet_release_mass_bound_core(
        1.0,
        probability,
        det_mass,
        grain_mass_kg,
        grains_per_independent_packet,
    )?;
    Ok((bound.release_mass_kg, probability_inflation))
}

/// Shared prefix: native-status label routing + the MF deterministic-mass
/// validity gate, evaluated BEFORE any floor/Pc math (design R4).
/// `Some(verdict)` means the row resolves here; `None` means `det_mass` is
/// valid and the caller proceeds to the floor/Pc tail. Source of truth for both
/// the injected (slice-2) and raw (slice-3) row builders so the two can never
/// drift on the label/det gates.
///
/// PARITY-ANCHOR SWEEP, 2026-08-07. This routing was documented across the
/// module as mirroring `dust_phase.py:1677-1710`. That anchor is retired here
/// and at every other site in this crate, as a pointer only:
///
/// * `dust_phase.py` appears in no commit of either Python repo (0 of 1,236 in
///   `nasa_dust_clean`, 0 of 4,258 in `nasa_dust`). The sole surviving copy is
///   an untracked 3,086-line working file dated 2026-07-16 in a dead
///   `nasa_dust_clean` worktree.
/// * The cited revision is strictly later than that copy, so the line numbers
///   do not resolve against it either: drift grows with line number (`:306`
///   → def at 302, `:1426` → 1128, `:2698` → 2334) and `:3483`, cited by the
///   since-deleted `fraction_event.rs` (removed 2026-08-21, see the crate root
///   docs), was past its 3,086-line EOF.
/// * `det_status_label_code` itself is a name this crate coined. It occurs in
///   no Python tree, live or orphaned.
///
/// What is NOT retired is the behaviour. d68e3ec made this routing reachable
/// for the first time (Stage 3 previously killed the run one stage earlier),
/// and it stays gated by the captured oracle rows, which are bytes and do not
/// depend on any Python source still existing.
#[inline]
fn mf_verdict_label_det_prefix(
    executed_dv: f64,
    det_mass: f64,
    det_status_label_code: i32,
    dust_hard_limit_kg: f64,
    min_practical: f64,
) -> Option<(f64, f64, bool, i32)> {
    // (1) native-status label routing, BEFORE the fidelity branch (1677-1697).
    if det_status_label_code == LABEL_SAFE_BY_DEFAULT {
        return Some((executed_dv, 0.0, false, REASON_SAFE_BY_DEFAULT));
    }
    if det_status_label_code == LABEL_PHYSICS_LIMITED {
        if det_mass.is_finite() && det_mass > 0.0 && det_mass >= min_practical {
            return Some((
                executed_dv,
                det_mass,
                false,
                REASON_DETERMINISTIC_MASS_PHYSICS_LIMITED,
            ));
        }
        return Some((
            executed_dv,
            dust_hard_limit_kg,
            false,
            REASON_DETERMINISTIC_MASS_INVALID,
        ));
    }
    // (2) MF deterministic-mass validity gate (1698-1710).
    if !det_mass.is_finite() || det_mass <= 0.0 || det_mass < min_practical {
        return Some((
            executed_dv,
            dust_hard_limit_kg,
            false,
            REASON_DETERMINISTIC_MASS_INVALID,
        ));
    }
    None
}

/// Shared tail: the deterministic floor hard-limit short-circuit plus the
/// precomputed-Pc `total_mass` guards, given an
/// already-resolved `floor_mass` and `total_mass`. Source of truth for the
/// branch decisions; the slice-2 injected path passes Python-computed masses,
/// the slice-3 raw path passes natively-computed masses.
#[inline]
fn mf_verdict_tail(
    executed_dv: f64,
    floor_mass: f64,
    total_mass: f64,
    pc_valid: bool,
    dust_hard_limit_kg: f64,
) -> (f64, f64, bool, i32) {
    // (3) deterministic floor hard-limit short-circuit (1722-1765): Pc never
    // ran, so the reason must not claim Pc authority.
    if dust_hard_limit_kg.is_finite()
        && dust_hard_limit_kg > 0.0
        && floor_mass.is_finite()
        && floor_mass >= dust_hard_limit_kg
    {
        return (
            executed_dv,
            floor_mass,
            false,
            REASON_PROB_MASS_FLOOR_HARD_LIMIT,
        );
    }
    // (4) precomputed-Pc total_mass guards (1823-1837). `pc_valid == false`
    // marks a row the caller must exclude from the compare (Python raised, i.e.
    // prob_mass_exception, which the native path does not own); we still return
    // a well-formed verdict so the buffer stays dense.
    if !pc_valid || !total_mass.is_finite() || total_mass <= 0.0 {
        return (
            executed_dv,
            dust_hard_limit_kg,
            false,
            REASON_PROB_MASS_INVALID,
        );
    }
    if total_mass >= dust_hard_limit_kg {
        return (executed_dv, total_mass, false, REASON_PROB_MASS_HARD_LIMIT);
    }
    (executed_dv, total_mass, true, REASON_OK)
}

/// Injected (slice-2) guard tree: floor / Pc math stays Python, `floor_mass` and
/// `pc_total_mass` cross as inputs. Retained for the branch-logic unit tests; the
/// production driver uses [`mf_verdict_tuple_raw`].
#[inline]
fn mf_verdict_tuple(
    executed_dv: f64,
    det_mass: f64,
    det_status_label_code: i32,
    floor_mass: f64,
    pc_total_mass: f64,
    pc_valid: bool,
    dust_hard_limit_kg: f64,
    min_practical: f64,
) -> (f64, f64, bool, i32) {
    if let Some(verdict) = mf_verdict_label_det_prefix(
        executed_dv,
        det_mass,
        det_status_label_code,
        dust_hard_limit_kg,
        min_practical,
    ) {
        return verdict;
    }
    mf_verdict_tail(
        executed_dv,
        floor_mass,
        pc_total_mass,
        pc_valid,
        dust_hard_limit_kg,
    )
}

/// Raw (slice-3) guard tree: the floor and precomputed-Pc `total_mass` are
/// computed NATIVELY from the raw operands, replacing the slice-2 precomputed
/// `floor_mass`/`pc_total_mass` inputs.
///
/// After the label/det prefix (which never touches the floor, so an invalid
/// `det_mass` never reaches `release_mass_floor_core`), the floor is derived
/// from `det_mass` + the per-event floor scalars, then the precomputed-Pc unpack
/// (`compute_release_mass(..., precomputed_pc_result=)`, release_mass.py:735-762)
/// enforces the floor on the raw Pc quadrature mass:
/// `total_mass = max(pc_raw_mass, floor_mass)`. An invalid raw Pc mass (Python
/// would raise `prob_mass_exception`, an excluded row) or a `pc_valid == false`
/// row is routed to `prob_mass_invalid` by the tail; those rows are excluded from
/// the parity compare.
#[inline]
fn mf_verdict_tuple_raw(
    executed_dv: f64,
    det_mass: f64,
    det_status_label_code: i32,
    pc_raw_mass: f64,
    pc_valid: bool,
    grain_mass_kg: f64,
    grains_per_independent_packet: u64,
    hit_probability: f64,
    dust_hard_limit_kg: f64,
    min_practical: f64,
) -> anyhow::Result<(f64, f64, bool, i32)> {
    if let Some(verdict) = mf_verdict_label_det_prefix(
        executed_dv,
        det_mass,
        det_status_label_code,
        dust_hard_limit_kg,
        min_practical,
    ) {
        return Ok(verdict);
    }
    // det_mass is valid here (prefix returned None): compute the floor natively.
    let (floor_mass, _probability_inflation) = release_mass_floor_core(
        det_mass,
        grain_mass_kg,
        grains_per_independent_packet,
        hit_probability,
    )?;
    // Floor enforcement on the raw Pc quadrature mass (release_mass.py:752-755:
    // `if total_mass < min_release_mass_floor: total_mass = floor`). Written as
    // the exact Python branch, not `.max`, to keep the NaN/`-0.0` semantics.
    //
    // MEASURED UNREACHABLE FROM THE NATIVE CALLER, AND IT STILL MUST STAY.
    // Across all three `physics_3event` events, 1384/1384 rows took the `else`
    // arm: the floor won zero times. That is a property of one caller, not of
    // this branch. Three reasons it is not deletable:
    //
    // 1. It is not `.max()`, and the difference is reachable. For
    //    `pc_raw_mass = NaN`, `NaN < floor_mass` is false, so `total_mass`
    //    stays NaN and `mf_verdict_tail`'s `!total_mass.is_finite()` guard
    //    routes the row to `REASON_PROB_MASS_INVALID`. `f64::max` returns the
    //    non-NaN operand, so it would hand back a finite `floor_mass` and
    //    promote an invalid row to a feasible one carrying a fabricated mass.
    // 2. `pc_raw_mass` is a caller-supplied slice on a public API
    //    (`fraction_grid_finalize_mf_core`), not an internal value. The native
    //    path happens to guarantee `pc_raw_mass >= floor_mass` because its
    //    `capture_probability` is `exp(log_p_capture.min(0.0)) <= 1`, so
    //    dividing by it can only inflate past the `capture_probability = 1.0`
    //    floor. `nd_pipeline/tests/physics_finalize.rs` feeds this same API straight from
    //    captured Python oracle JSON, which carries no such guarantee.
    // 3. Python has this branch, so parity requires it. Deleting it would pass
    //    every current test — no fixture row exercises it — and silently
    //    diverge on the first row where the oracle's Pc mass falls under the
    //    floor.
    let total_mass = if pc_raw_mass < floor_mass {
        floor_mass
    } else {
        pc_raw_mass
    };
    Ok(mf_verdict_tail(
        executed_dv,
        floor_mass,
        total_mass,
        pc_valid,
        dust_hard_limit_kg,
    ))
}

/// Native port of `_validate_raw_cloud_centroid`:
/// decide whether a propagated GMM cloud's weighted physical centroid is stale.
///
/// Mirrors the Python oracle branch-for-branch on the cloud `SoA`
/// (`gmm_weights[:n]`, `gmm_means[:n, :3]`) sliced to the active components:
///   1. no active components (`n == 0`) → reject (Python raises "requires active
///      components");
///   2. `normalize_mixture_weights` (`mass_math.py:8`): clamp each weight to
///      `w if w.is_finite() && w > 0 else 0`, sum; a non-finite / non-positive
///      total → reject (Python `normalize_mixture_weights` raises "no positive
///      finite mass"); otherwise normalise by `1/total` (the `size <= 8`
///      sequential fold — the GMM component count is small, so numpy never takes
///      its pairwise arm here);
///   3. weighted centroid `sum_i w_i * mean_i` per axis; a non-finite centroid →
///      reject (Python raises "must be finite");
///   4. `offset = ||centroid - expected||`; `!offset.is_finite() || offset > tol`
///      → reject (Python raises "exceeds physical tolerance").
///
/// The returned `offset_km` is **not** load-bearing: it never crosses into the
/// row buffer (Python discards it except for an off-by-default diagnostic), so
/// only the `rejected` boolean must agree with Python. The physical offsets are
/// either ~0 (accept) or large (reject), never within ulps of the 0.01 km
/// tolerance, so the sequential fold's 1-ulp divergence from numpy's centroid
/// cannot flip the decision. The difference terms are squared by explicit
/// multiplication (R5 discipline) rather than `powi`/`pow`.
///
/// `means_flat` is the row-major `n × 3` position block; a length mismatch is a
/// fail-loud contract break (the shim always passes `3 * n`).
///
/// # Errors
///
/// Returns an error when `means_flat` is not exactly three coordinates per weight.
///
/// # Not an oracle, and not yet wired
///
/// It has no Rust caller outside the three tests below, and they compare it
/// against nothing else — so by the rule in `docs/REFACTOR_BLOCKLIST.md`
/// ("Only an inline test calls it") this is a relationship, not a comparison.
/// It is kept anyway because `docs/mf_pipeline_blueprint.md` names
/// `validate_raw_cloud_centroid` as a `[NAT] dust_estimates` call of the MF
/// dust-evolve stage, i.e. as a port target that has not been connected yet,
/// and because the doc comment above is the only written statement of the
/// Python accept/reject semantics it has to reproduce. Its batched sibling
/// `fraction_event::validate_raw_cloud_centroid_batch_core` — which was its
/// last non-test caller — was deleted on 2026-08-21 as a Python-FFI batching
/// layer with no boundary left to cross. If the blueprint line goes, this
/// should go with it.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "unwired MF port target named by docs/mf_pipeline_blueprint.md; see the \
                  \"Not an oracle\" note above before deleting"
    )
)]
pub(crate) fn validate_raw_cloud_centroid_core(
    weights_raw: &[f64],
    means_flat: &[f64],
    expected: &[f64; 3],
    tolerance_km: f64,
) -> anyhow::Result<(f64, bool)> {
    let n = weights_raw.len();
    let expected_mean_len = n.saturating_mul(3);
    if means_flat.len() != expected_mean_len {
        return Err(anyhow::anyhow!(
            "validate_raw_cloud_centroid means_flat has {}, expected {} (3 x {n})",
            means_flat.len(),
            expected_mean_len
        ));
    }
    // (1) no active components → reject.
    if n == 0 {
        return Ok((f64::NAN, true));
    }
    // (2) normalize_mixture_weights: clamp then sum, reject on non-positive mass.
    let mut clamped = Vec::with_capacity(n);
    let mut total = 0.0_f64;
    for &w in weights_raw {
        let value = if w.is_finite() && w > 0.0 { w } else { 0.0 };
        clamped.push(value);
        total += value;
    }
    if !total.is_finite() || total <= 0.0 {
        return Ok((f64::NAN, true));
    }
    let inv_total = 1.0 / total;
    // (3) weighted centroid per axis (sequential fold; matches numpy for small n).
    let mut centroid = [0.0_f64; 3];
    let [centroid_x, centroid_y, centroid_z] = &mut centroid;
    for (&w, mean) in clamped.iter().zip(means_flat.chunks_exact(3)) {
        let Ok(&[mean_x, mean_y, mean_z]) = <&[f64; 3]>::try_from(mean) else {
            continue;
        };
        let wn = w * inv_total;
        *centroid_x += wn * mean_x;
        *centroid_y += wn * mean_y;
        *centroid_z += wn * mean_z;
    }
    if !(centroid_x.is_finite() && centroid_y.is_finite() && centroid_z.is_finite()) {
        return Ok((f64::NAN, true));
    }
    // (4) offset = ||centroid - expected||; square by explicit multiplication.
    let [expected_x, expected_y, expected_z] = *expected;
    let dx = *centroid_x - expected_x;
    let dy = *centroid_y - expected_y;
    let dz = *centroid_z - expected_z;
    let offset_km = (dx * dx + dy * dy + dz * dz).sqrt();
    let rejected = !offset_km.is_finite() || offset_km > tolerance_km;
    Ok((offset_km, rejected))
}

/// Passthrough verdict for a precheck-reject row (design 2.3): `result is None`
/// rows (`spec_none`/`cloud_none`/`centroid_reject`) whose executed-dv and reason
/// were already resolved Python-side (candidate-prepare / cloud-evolve / centroid
/// stages stay Python in slices 1-4). The driver emits the row buffer entry
/// verbatim — the infeasible executed-dv, a `NaN` release mass, `feasible=false`,
/// and the passthrough reason — then derives `required_mass_kind` through the
/// slice-1 classifier so the emitted row matches the Python row-builder
/// (`_authoritative_pc_row_from_result`) byte-for-byte.
#[inline]
fn fraction_grid_finalize_passthrough_row(
    executed_dv: f64,
    reason_passthrough: i32,
) -> anyhow::Result<MfVerdict> {
    let mass_kg = f64::NAN;
    let mass_kind_code = classify_required_mass_kind_core(false, reason_passthrough, mass_kg)?;
    Ok(MfVerdict {
        executed_dv,
        mass_kg,
        feasible: false,
        reason_code: reason_passthrough,
        mass_kind_code,
    })
}

/// Batch driver over the MF verdict guard tree (design 3.2 `*_core`).
///
/// Rows are disjoint, so the parallel arm applies the identical per-row closure
/// without a reduction reorder and is byte-identical to the serial fallback.
///
/// Slice 4 routes the full `row_state_code` taxonomy: `prepared_ok` and
/// `detmass_invalid_skip` run the MF verdict tree (the latter's invalid
/// `det_mass` is caught by the deterministic-mass gate), while `spec_none` /
/// `cloud_none` / `centroid_reject` emit the passthrough verdict from the
/// resolved `executed_dv` + `reason_passthrough` inputs. An
/// out-of-range state is fail-loud.
///
/// # Errors
///
/// Returns an error for mismatched input lengths, invalid row states, or invalid
/// raw mass inputs.
pub fn fraction_grid_finalize_mf_core(
    executed_dv: &[f64],
    row_state_code: &[i32],
    det_mass: &[f64],
    det_status_label_code: &[i32],
    pc_raw_mass: &[f64],
    pc_valid: &[u8],
    reason_passthrough: &[i32],
    dust_hard_limit_kg: f64,
    min_practical: f64,
    grain_mass_kg: f64,
    grains_per_independent_packet: u64,
    hit_probability: f64,
) -> anyhow::Result<Vec<MfVerdict>> {
    let n = executed_dv.len();
    for (name, len) in [
        ("row_state_code", row_state_code.len()),
        ("det_mass", det_mass.len()),
        ("det_status_label_code", det_status_label_code.len()),
        ("pc_raw_mass", pc_raw_mass.len()),
        ("pc_valid", pc_valid.len()),
        ("reason_passthrough", reason_passthrough.len()),
    ] {
        if len != n {
            return Err(anyhow::anyhow!(
                "fraction_grid_finalize_mf length mismatch: {name} has {len}, expected {n}"
            ));
        }
    }

    let compute_row = |index: usize| -> anyhow::Result<MfVerdict> {
        let (
            Some(&state),
            Some(&row_executed_dv),
            Some(&row_det_mass),
            Some(&row_det_status),
            Some(&row_pc_mass),
            Some(&row_pc_valid),
            Some(&row_reason),
        ) = (
            row_state_code.get(index),
            executed_dv.get(index),
            det_mass.get(index),
            det_status_label_code.get(index),
            pc_raw_mass.get(index),
            pc_valid.get(index),
            reason_passthrough.get(index),
        )
        else {
            return Err(anyhow::anyhow!(
                "fraction_grid_finalize_mf missing validated row {index}"
            ));
        };
        match state {
            ROW_STATE_PREPARED_OK | ROW_STATE_DETMASS_INVALID_SKIP => {
                fraction_grid_finalize_mf_row_from_raw(
                    row_executed_dv,
                    row_det_mass,
                    row_det_status,
                    row_pc_mass,
                    row_pc_valid != 0,
                    grain_mass_kg,
                    grains_per_independent_packet,
                    hit_probability,
                    dust_hard_limit_kg,
                    min_practical,
                )
            }
            ROW_STATE_SPEC_NONE | ROW_STATE_CLOUD_NONE | ROW_STATE_CENTROID_REJECT => {
                fraction_grid_finalize_passthrough_row(row_executed_dv, row_reason)
            }
            _ => Err(anyhow::anyhow!(
                "fraction_grid_native_row_state_unknown row_index={index} row_state_code={state}"
            )),
        }
    };

    // Row-disjoint verdicts: threshold-gated parallel arm (matches the house
    // `should_parallelize` pattern). Both arms fold the identical closure.
    if satpy_core::parallel_utils::should_parallelize(n, PAR_THRESHOLD) {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .map(compute_row)
            .collect::<anyhow::Result<Vec<_>>>()
    } else {
        (0..n).map(compute_row).collect::<anyhow::Result<Vec<_>>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_requires_feasible_and_positive_finite_mass() {
        assert_eq!(
            classify_required_mass_kind_core(true, 0, 1.0).unwrap(),
            KIND_EXACT
        );
        // feasible but non-positive / non-finite mass is not exact.
        assert_eq!(
            classify_required_mass_kind_core(true, 0, 0.0).unwrap(),
            KIND_UNAVAILABLE
        );
        assert_eq!(
            classify_required_mass_kind_core(true, 0, f64::NAN).unwrap(),
            KIND_UNAVAILABLE
        );
        assert_eq!(
            classify_required_mass_kind_core(true, 0, f64::INFINITY).unwrap(),
            KIND_UNAVAILABLE
        );
    }

    #[test]
    fn atmospheric_guard_is_physical_infeasible() {
        assert_eq!(
            classify_required_mass_kind_core(false, REASON_ATMOSPHERIC_GUARD, f64::NAN).unwrap(),
            KIND_PHYSICAL_INFEASIBLE
        );
        // atmospheric_guard wins over the mass gate even with a positive mass,
        // as long as the row is infeasible.
        assert_eq!(
            classify_required_mass_kind_core(false, REASON_ATMOSPHERIC_GUARD, 5.0).unwrap(),
            KIND_PHYSICAL_INFEASIBLE
        );
    }

    #[test]
    fn hard_and_physics_limited_reasons_are_lower_bound_with_positive_mass() {
        for code in [
            REASON_DETERMINISTIC_MASS_HARD_LIMIT,
            REASON_DETERMINISTIC_MASS_PHYSICS_LIMITED,
            REASON_PROB_MASS_FLOOR_HARD_LIMIT,
            REASON_PROB_MASS_HARD_LIMIT,
        ] {
            assert_eq!(
                classify_required_mass_kind_core(false, code, 3.0).unwrap(),
                KIND_LOWER_BOUND,
                "code {code} with positive mass must be lower_bound"
            );
            // Without a positive finite mass those reasons fall through to
            // unavailable.
            assert_eq!(
                classify_required_mass_kind_core(false, code, f64::NAN).unwrap(),
                KIND_UNAVAILABLE,
                "code {code} with NaN mass must be unavailable"
            );
        }
    }

    #[test]
    fn unknown_reason_code_is_fail_loud() {
        for bad in [-1, REASON_CODE_COUNT, REASON_CODE_COUNT + 100, i32::MIN] {
            let err = classify_required_mass_kind_core(false, bad, 1.0).unwrap_err();
            assert!(
                err.to_string()
                    .contains("fraction_grid_native_unknown_reason_code"),
                "expected fail-loud token, got {err}"
            );
        }
    }

    #[test]
    fn ok_reason_with_infeasible_row_is_unavailable() {
        // reason_code 0 == "ok" but the row is infeasible: not exact (feasible
        // is false), not a special reason → unavailable.
        assert_eq!(
            classify_required_mass_kind_core(false, 0, 1.0).unwrap(),
            KIND_UNAVAILABLE
        );
    }

    // ---- slice 2: MF verdict guard tree ----

    fn row(
        executed_dv: f64,
        det_mass: f64,
        label: i32,
        floor_mass: f64,
        pc_total_mass: f64,
        pc_valid: bool,
        hard_limit: f64,
        min_practical: f64,
    ) -> MfVerdict {
        fraction_grid_finalize_mf_row(
            executed_dv,
            det_mass,
            label,
            floor_mass,
            pc_total_mass,
            pc_valid,
            hard_limit,
            min_practical,
        )
        .unwrap()
    }

    fn assert_float_bits_eq(actual: f64, expected: f64) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn safe_by_default_short_circuits_before_fidelity_branch() {
        // R4: label wins even with an otherwise-feasible mass.
        let v = row(2.5, 5.0, LABEL_SAFE_BY_DEFAULT, 1.0, 3.0, true, 100.0, 1e-9);
        assert_float_bits_eq(v.executed_dv, 2.5);
        assert_float_bits_eq(v.mass_kg, 0.0);
        assert!(!v.feasible);
        assert_eq!(v.reason_code, REASON_SAFE_BY_DEFAULT);
        assert_eq!(v.mass_kind_code, KIND_UNAVAILABLE);
    }

    #[test]
    fn physics_limited_boundary_at_min_practical() {
        // det_mass == min_practical is inclusive (>=) → physics_limited.
        let v = row(2.0, 4.0, LABEL_PHYSICS_LIMITED, 0.0, 0.0, true, 100.0, 4.0);
        assert_float_bits_eq(v.mass_kg, 4.0);
        assert_eq!(v.reason_code, REASON_DETERMINISTIC_MASS_PHYSICS_LIMITED);
        assert_eq!(v.mass_kind_code, KIND_LOWER_BOUND);
        // Just below min_practical → invalid with hard-limit mass.
        let below = row(
            2.0,
            3.999,
            LABEL_PHYSICS_LIMITED,
            0.0,
            0.0,
            true,
            100.0,
            4.0,
        );
        assert_float_bits_eq(below.mass_kg, 100.0);
        assert_eq!(below.reason_code, REASON_DETERMINISTIC_MASS_INVALID);
    }

    #[test]
    fn mf_det_mass_invalid_gate() {
        for bad in [f64::NAN, 0.0, -1.0, 0.5] {
            let v = row(2.0, bad, LABEL_OTHER, 1.0, 3.0, true, 100.0, 1.0);
            assert_eq!(
                v.reason_code, REASON_DETERMINISTIC_MASS_INVALID,
                "det_mass {bad} must be invalid"
            );
            assert_float_bits_eq(v.mass_kg, 100.0);
            assert_float_bits_eq(v.executed_dv, 2.0);
        }
    }

    #[test]
    fn floor_hard_limit_short_circuit_returns_floor_not_pc() {
        // R6: floor_mass >= hard_limit → prob_mass_floor_hard_limit with the
        // floor mass, and Pc (pc_total_mass) is never consulted.
        let v = row(2.0, 5.0, LABEL_OTHER, 150.0, 7.0, true, 100.0, 1.0);
        assert_float_bits_eq(v.mass_kg, 150.0);
        assert_eq!(v.reason_code, REASON_PROB_MASS_FLOOR_HARD_LIMIT);
        assert_eq!(v.mass_kind_code, KIND_LOWER_BOUND);
    }

    #[test]
    fn pc_total_mass_guards() {
        // invalid pc (non-finite / <= 0 / pc_valid false) → hard limit + invalid.
        for (mass, valid) in [(f64::NAN, true), (0.0, true), (5.0, false)] {
            let v = row(2.0, 5.0, LABEL_OTHER, 1.0, mass, valid, 100.0, 1.0);
            assert_eq!(v.reason_code, REASON_PROB_MASS_INVALID);
            assert_float_bits_eq(v.mass_kg, 100.0);
        }
        // total_mass == hard_limit is inclusive (>=) → hard limit reason.
        let hl = row(2.0, 5.0, LABEL_OTHER, 1.0, 100.0, true, 100.0, 1.0);
        assert_eq!(hl.reason_code, REASON_PROB_MASS_HARD_LIMIT);
        assert_float_bits_eq(hl.mass_kg, 100.0);
        assert!(!hl.feasible);
        // feasible ok row.
        let ok = row(2.0, 5.0, LABEL_OTHER, 1.0, 42.0, true, 100.0, 1.0);
        assert_eq!(ok.reason_code, REASON_OK);
        assert_float_bits_eq(ok.mass_kg, 42.0);
        assert!(ok.feasible);
        assert_eq!(ok.mass_kind_code, KIND_EXACT);
    }

    #[test]
    fn core_serial_and_parallel_are_byte_identical() {
        // Build a raw batch that exercises every branch, then assert the
        // threshold-gated core (which may take the parallel arm) agrees
        // bit-for-bit with an explicit serial fold of the raw row builder
        // (design R10). Both fold the identical per-row closure.
        let indices = 0..512_u32;
        let n = usize::try_from(indices.end).unwrap_or(0);
        let executed: Vec<f64> = indices
            .clone()
            .map(|index| 2.0 + f64::from(index) * 1e-3)
            .collect();
        let states = vec![ROW_STATE_PREPARED_OK; n];
        let det: Vec<f64> = indices
            .clone()
            .map(|index| match index % 5 {
                0 => f64::NAN,
                1 => 0.0,
                _ => 5.0 + f64::from(index) * 1e-4,
            })
            .collect();
        let labels: Vec<i32> = indices
            .clone()
            .map(|index| match index % 7 {
                0 => LABEL_SAFE_BY_DEFAULT,
                1 => LABEL_PHYSICS_LIMITED,
                _ => LABEL_OTHER,
            })
            .collect();
        let pc_raw: Vec<f64> = indices
            .clone()
            .map(|index| match index % 4 {
                0 => f64::NAN,
                1 => 150.0,
                _ => 42.0 + f64::from(index) * 1e-4,
            })
            .collect();
        let pc_valid: Vec<u8> = indices.map(|index| u8::from(index % 13 != 0)).collect();
        let reason_pass = vec![REASON_OK; n];
        let hard = 100.0;
        let min_practical = 1.0;
        let grain_mass_kg = 1.0e-6;
        let grains = 1u64;
        let hit_probability = 0.05;

        let via_core = fraction_grid_finalize_mf_core(
            &executed,
            &states,
            &det,
            &labels,
            &pc_raw,
            &pc_valid,
            &reason_pass,
            hard,
            min_practical,
            grain_mass_kg,
            grains,
            hit_probability,
        )
        .unwrap();
        let serial: Vec<MfVerdict> = executed
            .iter()
            .zip(&det)
            .zip(&labels)
            .zip(&pc_raw)
            .zip(&pc_valid)
            .map(
                |((((&executed_dv, &det_mass), &label), &raw_mass), &is_valid)| {
                    fraction_grid_finalize_mf_row_from_raw(
                        executed_dv,
                        det_mass,
                        label,
                        raw_mass,
                        is_valid != 0,
                        grain_mass_kg,
                        grains,
                        hit_probability,
                        hard,
                        min_practical,
                    )
                    .unwrap()
                },
            )
            .collect();
        for (a, b) in via_core.iter().zip(serial.iter()) {
            assert_eq!(a.executed_dv.to_bits(), b.executed_dv.to_bits());
            assert_eq!(a.mass_kg.to_bits(), b.mass_kg.to_bits());
            assert_eq!(a.feasible, b.feasible);
            assert_eq!(a.reason_code, b.reason_code);
            assert_eq!(a.mass_kind_code, b.mass_kind_code);
        }
    }

    // ---- slice 3: native floor + precomputed-Pc unpack ----

    #[test]
    fn release_mass_floor_matches_bound_and_probability_guards() {
        // Out-of-(0,1) hit probability short-circuits to (inf, inf), mirroring
        // compute_release_mass_floor's guard, regardless of det_mass.
        for bad_p in [0.0, 1.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
            let (floor, infl) = release_mass_floor_core(5.0, 1.0e-6, 1, bad_p).unwrap();
            assert_float_bits_eq(floor, f64::INFINITY);
            assert_float_bits_eq(infl, f64::INFINITY);
        }
        // In-range: the floor equals the finite-packet bound with
        // capture_probability=1.0 and target=hit_probability, bit-for-bit.
        let (det, gm, grains, p) = (7.3, 1.0e-6, 1u64, 0.05);
        let (floor, _infl) = release_mass_floor_core(det, gm, grains, p).unwrap();
        let bound = crate::finite_packet_release_mass_bound_core(1.0, p, det, gm, grains).unwrap();
        assert_eq!(floor.to_bits(), bound.release_mass_kg.to_bits());
    }

    #[test]
    fn raw_row_floor_hard_limit_and_floor_enforcement() {
        // Pinned operands: pick a probability/grain so the floor sits between the
        // raw Pc mass and the hard limit, forcing floor enforcement
        // (total_mass = floor), then a hard-limit case (floor >= hard_limit).
        let (gm, grains, p, min_practical) = (1.0e-3, 1u64, 0.5, 1.0e-9);
        // det large enough that the floor exceeds a tiny raw Pc mass but stays
        // below the hard limit → floor-enforced ok/feasible row.
        let (floor, _) = release_mass_floor_core(0.01, gm, grains, p).unwrap();
        assert!(floor.is_finite() && floor > 0.0);
        let hard = floor + 10.0;
        // raw Pc mass below the floor → enforcement lifts total_mass to floor.
        let enforced = fraction_grid_finalize_mf_row_from_raw(
            2.0,
            0.01,
            LABEL_OTHER,
            floor * 0.5,
            true,
            gm,
            grains,
            p,
            hard,
            min_practical,
        )
        .unwrap();
        assert_eq!(enforced.mass_kg.to_bits(), floor.to_bits());
        assert!(enforced.feasible);
        assert_eq!(enforced.reason_code, REASON_OK);
        // floor >= hard_limit → prob_mass_floor_hard_limit with the floor mass,
        // Pc never consulted (R6).
        let hl = fraction_grid_finalize_mf_row_from_raw(
            2.0,
            0.01,
            LABEL_OTHER,
            floor + 1.0,
            true,
            gm,
            grains,
            p,
            floor,
            min_practical,
        )
        .unwrap();
        assert_eq!(hl.mass_kg.to_bits(), floor.to_bits());
        assert_eq!(hl.reason_code, REASON_PROB_MASS_FLOOR_HARD_LIMIT);
        assert_eq!(hl.mass_kind_code, KIND_LOWER_BOUND);
    }

    #[test]
    fn core_rejects_unknown_row_state() {
        let err = fraction_grid_finalize_mf_core(
            &[2.0],
            &[99], // out-of-taxonomy row state
            &[5.0],
            &[LABEL_OTHER],
            &[42.0],
            &[1],
            &[REASON_OK],
            100.0,
            1.0,
            1.0e-6,
            1,
            0.05,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("fraction_grid_native_row_state_unknown"),
            "expected unknown-row-state guard, got {err}"
        );
    }

    // ---- slice 4: precheck-reject routing + centroid validate ----

    #[test]
    fn precheck_states_emit_passthrough_verdict() {
        // spec_none/cloud_none/centroid_reject rows are `result is None` rows:
        // the driver emits (executed_dv, NaN, false, reason) and
        // derives required_mass_kind through the classifier. Assert exact reason
        // + kind for each precheck class (design R3).
        // spec_none carries release_control_missing (11) → unavailable;
        // cloud_none carries atmospheric_guard (8) → physical_infeasible;
        // centroid_reject carries raw_cloud_centroid (9) → unavailable.
        for (state, reason, expected_kind) in [
            (ROW_STATE_SPEC_NONE, 11, KIND_UNAVAILABLE),
            (
                ROW_STATE_CLOUD_NONE,
                REASON_ATMOSPHERIC_GUARD,
                KIND_PHYSICAL_INFEASIBLE,
            ),
            (ROW_STATE_CENTROID_REJECT, 9, KIND_UNAVAILABLE),
        ] {
            let verdicts = fraction_grid_finalize_mf_core(
                &[7.25],
                &[state],
                &[f64::NAN],
                &[LABEL_OTHER],
                &[f64::NAN],
                &[0],
                &[reason],
                100.0,
                1.0,
                1.0e-6,
                1,
                0.05,
            )
            .unwrap();
            let v = verdicts
                .first()
                .copied()
                .expect("single-row verdict must be present");
            assert_eq!(v.executed_dv.to_bits(), 7.25_f64.to_bits());
            assert!(v.mass_kg.is_nan());
            assert!(!v.feasible);
            assert_eq!(v.reason_code, reason);
            assert_eq!(v.mass_kind_code, expected_kind, "state {state}");
        }
    }

    #[test]
    fn detmass_invalid_skip_reenters_verdict_tree() {
        // row_state 4 runs the MF verdict tree: an invalid det_mass is caught by
        // the deterministic-mass gate exactly as a prepared-ok row.
        let verdicts = fraction_grid_finalize_mf_core(
            &[2.0],
            &[ROW_STATE_DETMASS_INVALID_SKIP],
            &[f64::NAN],
            &[LABEL_OTHER],
            &[f64::NAN],
            &[0],
            &[REASON_OK],
            100.0,
            1.0,
            1.0e-6,
            1,
            0.05,
        )
        .unwrap();
        let verdict = verdicts
            .first()
            .copied()
            .expect("single-row verdict must be present");
        assert_eq!(verdict.reason_code, REASON_DETERMINISTIC_MASS_INVALID);
        assert_eq!(verdict.mass_kg.to_bits(), 100.0_f64.to_bits());
    }

    #[test]
    fn centroid_accepts_near_target_and_rejects_far() {
        // Two equal-weight components straddling the expected position: centroid
        // sits at the midpoint. A tiny offset (< tol) accepts; a large one
        // rejects. offset is the euclidean norm of the centroid-minus-expected.
        let weights = [1.0_f64, 1.0];
        // means at expected +/- 0.001 km on x → centroid == expected → offset 0.
        let expected = [100.0_f64, 200.0, 300.0];
        let means_near = [
            100.001, 200.0, 300.0, // comp 0
            99.999, 200.0, 300.0, // comp 1
        ];
        let (offset, rejected) =
            validate_raw_cloud_centroid_core(&weights, &means_near, &expected, 0.01).unwrap();
        assert!(offset < 1e-9, "near offset {offset}");
        assert!(!rejected);
        // shift both components far off → centroid far → reject.
        let means_far = [
            110.0, 200.0, 300.0, //
            110.0, 200.0, 300.0,
        ];
        let (offset_far, rejected_far) =
            validate_raw_cloud_centroid_core(&weights, &means_far, &expected, 0.01).unwrap();
        assert!(offset_far > 9.0, "far offset {offset_far}");
        assert!(rejected_far);
    }

    #[test]
    fn centroid_rejects_degenerate_weights_and_empty() {
        let expected = [0.0_f64; 3];
        // no active components → reject.
        let (_, rejected_empty) =
            validate_raw_cloud_centroid_core(&[], &[], &expected, 0.01).unwrap();
        assert!(rejected_empty);
        // all-zero / non-finite weights → no positive mass → reject.
        let (_, rejected_zero) =
            validate_raw_cloud_centroid_core(&[0.0, f64::NAN, -1.0], &[0.0; 9], &expected, 0.01)
                .unwrap();
        assert!(rejected_zero);
        // non-finite mean → non-finite centroid → reject.
        let (_, rejected_nan) =
            validate_raw_cloud_centroid_core(&[1.0], &[f64::INFINITY, 0.0, 0.0], &expected, 0.01)
                .unwrap();
        assert!(rejected_nan);
    }

    #[test]
    fn centroid_length_mismatch_is_fail_loud() {
        let expected = [0.0_f64; 3];
        let err =
            validate_raw_cloud_centroid_core(&[1.0, 1.0], &[0.0; 3], &expected, 0.01).unwrap_err();
        assert!(err.to_string().contains("means_flat"), "got {err}");
    }
}
