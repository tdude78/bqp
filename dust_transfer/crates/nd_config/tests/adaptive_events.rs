//! The controllable `X + Y*n` event schedule, at the config seam.
//!
//! Events one objective call spends per design are `X + Y*n`: `X` events open
//! the Beta convergence assessment, each extra round buys `Y` more, and `n` is
//! the OBSERVED number of extra rounds the stop asked for. `X` and `Y` are
//! declarable here; `n` never is.
//!
//! These tests exist to answer two questions the seal battery cannot:
//! defaults must be indistinguishable from the sealed constant (or this is a
//! silent science change), and a non-default schedule must be VISIBLE in the
//! science digest (or the plumbing is inert and every other test still passes).

use std::path::PathBuf;

use common_rs::{require_err, require_ok};
use nd_config::{CompiledPartAScienceV1, Config, PartACampaignScope};

/// The sealed schedule, as a triple, read from the compiled authority rather
/// than retyped -- a literal here would pass even after the seal moved.
const fn sealed_schedule() -> (usize, usize, usize) {
    let mf = CompiledPartAScienceV1::part_a_v1().mf();
    (
        mf.adaptive_initial_events,
        mf.adaptive_event_step,
        mf.adaptive_stage_count,
    )
}

fn exact36_yaml() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("configs/part_a_exact36.yaml");
    let result = std::fs::read_to_string(path);
    assert!(result.is_ok(), "exact36 YAML must load: {result:?}");
    result.unwrap_or_default()
}

#[test]
fn a_config_without_the_block_resolves_the_sealed_schedule() {
    let cfg = require_ok!(Config::from_yaml_str("config:\n  version: 1\n"));

    assert_eq!(
        require_ok!(cfg.adaptive_event_schedule()),
        sealed_schedule()
    );
}

#[test]
fn sealed_oa8_schedule_spans_8_through_all_500_events() {
    let science = CompiledPartAScienceV1::part_a_v1();
    let mf = science.mf();
    assert_eq!(
        (
            mf.adaptive_initial_events,
            mf.adaptive_event_step,
            mf.adaptive_stage_count,
        ),
        (8, 4, 124)
    );
    assert_eq!(mf.adaptive_stage_index(8), Some(0));
    assert_eq!(mf.adaptive_stage_index(500), Some(123));
    assert_eq!(mf.adaptive_stage_index(6), None);
}

/// DEFAULTS ARE THE SEAL, byte for byte.
///
/// Declaring the sealed schedule explicitly must be indistinguishable from not
/// declaring it: same resolved triple, same science digest, and specifically a
/// BORROWED authority, so the default path cannot even allocate a second
/// authority that might drift from the constant.
#[test]
fn declaring_the_sealed_schedule_is_the_sealed_authority() {
    let (initial, step, stages) = sealed_schedule();
    let yaml = format!(
        "config:\n  version: 1\nscience:\n  adaptive_events:\n    initial: {initial}\n    step: {step}\n    stages: {stages}\n"
    );
    let cfg = require_ok!(Config::from_yaml_str(&yaml));

    assert_eq!(
        require_ok!(cfg.adaptive_event_schedule()),
        (initial, step, stages)
    );
    let resolved = require_ok!(cfg.resolved_part_a_science());
    assert!(
        matches!(resolved, std::borrow::Cow::Borrowed(_)),
        "the default schedule must resolve to the sealed authority itself"
    );
    assert_eq!(
        resolved.sha256_hex(),
        CompiledPartAScienceV1::part_a_v1().sha256_hex()
    );
}

/// POSITIVE CONTROL. Without this the whole feature could be inert.
///
/// Every other test in this file passes if `science.adaptive_events` is parsed
/// and then dropped on the floor. This one fails unless the declared schedule
/// reaches the compiled authority AND is folded into the science digest, which
/// is what makes an overridden ladder impossible to mistake for the sealed one.
#[test]
fn overriding_x_or_y_moves_the_science_digest() {
    let sealed = CompiledPartAScienceV1::part_a_v1().sha256_hex();
    let (initial, step, stages) = sealed_schedule();

    // Every override must be strictly off the seal on the axis it names, or the
    // assertion below is vacuous.
    for (label, x, y) in [
        ("smaller X", 4, step),
        ("smaller Y", initial, 2),
        ("both", 4, 2),
    ] {
        let yaml = format!(
            "config:\n  version: 1\nscience:\n  adaptive_events:\n    initial: {x}\n    step: {y}\n"
        );
        let cfg = require_ok!(Config::from_yaml_str(&yaml));

        assert_eq!(
            require_ok!(cfg.adaptive_event_schedule()),
            (x, y, stages),
            "{label}: unset keys must keep their sealed values"
        );
        let resolved = require_ok!(cfg.resolved_part_a_science());
        assert_eq!(
            resolved.mf().adaptive_initial_events,
            x,
            "{label}: X did not reach the compiled authority"
        );
        assert_eq!(
            resolved.mf().adaptive_event_step,
            y,
            "{label}: Y did not reach the compiled authority"
        );
        assert_ne!(
            resolved.sha256_hex(),
            sealed,
            "{label}: an overridden ladder must not carry the sealed digest"
        );
    }
}

/// Canonical Part A may DECLARE the schedule but may not CHANGE it.
///
/// The receipt writers (`nd_part_a_evidence::authority`, `receipt`) stamp
/// `part_a_v1().sha256_hex()` directly, so a canonical campaign on an
/// overridden ladder would publish a digest for a ladder it never ran. The
/// override is a non-canonical capability by construction, not by convention.
#[test]
fn canonical_part_a_refuses_an_overridden_schedule() {
    let (initial, step, stages) = sealed_schedule();

    let declared = format!(
        "{}\nscience:\n  adaptive_events:\n    initial: {initial}\n    step: {step}\n    stages: {stages}\n",
        exact36_yaml()
    );
    let cfg = require_ok!(Config::from_yaml_str(&declared));
    require_ok!(cfg.validate_part_a_semantics(PartACampaignScope::Exact36));

    // `initial: 24` and `step: 8` are the schedule R53 retired. Naming them here
    // makes the flip explicit: what was the sealed default through the Julier
    // reseal is, from this commit, an override canonical Part A must refuse.
    for (field, yaml_key, value) in [
        ("initial", "initial", 24),
        ("step", "step", 8),
        ("stages", "stages", 30),
    ] {
        let overridden = format!(
            "{}\nscience:\n  adaptive_events:\n    {yaml_key}: {value}\n",
            exact36_yaml()
        );
        let cfg = require_ok!(Config::from_yaml_str(&overridden));
        let error = require_err!(cfg.validate_part_a_semantics(PartACampaignScope::Exact36));
        assert!(
            error
                .to_string()
                .contains(&format!("science.adaptive_events.{field}")),
            "{field}: {error:#}"
        );
    }
}

#[test]
fn a_ladder_cannot_run_past_the_sealed_event_bank() {
    let bank_events = CompiledPartAScienceV1::part_a_v1().mf().b500_event_count;

    // The sealed ladder's own last stage stays inside the bank.
    let (initial, step, stages) = sealed_schedule();
    assert!(initial + step * (stages - 1) <= bank_events);

    require_ok!(CompiledPartAScienceV1::with_adaptive_events(
        initial, step, stages
    ));

    // Derive the ceiling from the bank, never from a copied stage count. The
    // representative-event ladder is sealed to span through all 500 events.
    let max_stages = (bank_events - initial) / step + 1;
    assert_eq!(max_stages, stages);
    assert_eq!(initial + step * (stages - 1), bank_events);

    // One stage past the bank's own ceiling is refused.
    let error = require_err!(CompiledPartAScienceV1::with_adaptive_events(
        initial,
        step,
        max_stages + 1
    ));
    assert!(
        matches!(
            error,
            nd_config::PartAAdaptiveEventsError::ExceedsEventBank { .. }
        ),
        "{error}"
    );

    for (x, y, s) in [(0, step, stages), (initial, 0, stages), (initial, step, 0)] {
        assert!(
            matches!(
                CompiledPartAScienceV1::with_adaptive_events(x, y, s),
                Err(nd_config::PartAAdaptiveEventsError::NotPositive { .. })
            ),
            "({x}, {y}, {s}) must be refused"
        );
    }
}

/// A single-stage ladder is legal: `stages = 1` means the Beta stop is never
/// offered a second round, so `n` is pinned to 0 by construction rather than
/// by the stop's arithmetic.
#[test]
fn a_one_stage_ladder_is_buildable() {
    let science = require_ok!(CompiledPartAScienceV1::with_adaptive_events(24, 8, 1));

    assert_eq!(science.mf().adaptive_stage_count, 1);
    assert_eq!(science.mf().adaptive_stage_index(24), Some(0));
    assert_eq!(science.mf().adaptive_stage_index(32), None);
}
