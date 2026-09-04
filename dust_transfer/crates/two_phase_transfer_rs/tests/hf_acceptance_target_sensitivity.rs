//! Legacy MF/gravity-only HF replay target-sensitivity diagnostic.
//!
//! `hf_acceptance_replay` keeps whatever catalogue target authority the context declares, so the
//! MF-vs-HF *delta* isolates transfer-arc fidelity. That is a statement about the delta, not about
//! the absolute residual. This test measures both halves of the question directly:
//!
//!   A. Within one context, is the target endpoint bit-identical in the MF and HF arms?
//!      (If yes, the MF/HF delta really is transfer-arc-only.)
//!   B. Does swapping the target's E0 state for its MEAN-EQUIVALENT
//!      (`mean_equinoctial_from_osculating_state` -> `equinoc2eci_impl`) move the reported HF
//!      residual, or does it pass straight through?
//!
//! B is the one that decides whether HF acceptance can serve as an independent validator for the
//! mean-element target correction.
//!
//! Run with `--nocapture` to read the tables.

use anyhow::Context as _;
use satpy_core::mean_elements::mean_equinoctial_from_osculating_state;
use serde_json::Value;
use two_phase_transfer_rs::evaluate::eci_to_equinoctial;
use two_phase_transfer_rs::hf_acceptance::{
    gravity_only_transfer_force_config, hf_acceptance_replay, HfGravityAuthority,
};
use two_phase_transfer_rs::types::{BodyRole, PlanContext, PlanResult, TargetPropagationAuthority};
use two_phase_transfer_rs::{replay_transfer_controls, J2ClosureSettings};

const DIR_R6_D15: &[u8] = include_bytes!("../data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

fn fixture() -> anyhow::Result<Value> {
    let path = format!(
        "{}/../nd_pipeline/tests/fixtures/physics_3event/stage1_transfer.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    serde_json::from_str(&text).context("stage1 fixture is strict JSON")
}

fn field<'a>(value: &'a Value, key: &str) -> anyhow::Result<&'a Value> {
    value
        .get(key)
        .with_context(|| format!("fixture is missing field {key}"))
}

fn num(v: &Value, key: &str) -> anyhow::Result<f64> {
    field(v, key)?
        .as_f64()
        .with_context(|| format!("fixture field {key} is not a number"))
}

fn integer(v: &Value, key: &str) -> anyhow::Result<i64> {
    field(v, key)?
        .as_i64()
        .with_context(|| format!("fixture field {key} is not an integer"))
}

fn vecn<const N: usize>(v: &Value) -> anyhow::Result<[f64; N]> {
    let arr = v.as_array().context("expected an array")?;
    anyhow::ensure!(arr.len() == N, "expected a {N}-vector");
    let mut out = [0.0; N];
    for (slot, item) in out.iter_mut().zip(arr) {
        *slot = item.as_f64().context("array entry is not a number")?;
    }
    Ok(out)
}

fn equ_of(eci: &[f64; 6]) -> anyhow::Result<[f64; 6]> {
    let mut equ = [0.0; 6];
    anyhow::ensure!(
        eci_to_equinoctial(eci, &mut equ),
        "eci -> equinoctial failed for state {eci:?}"
    );
    Ok(equ)
}

fn sep_m(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let &[a_x, a_y, a_z, ..] = a;
    let &[b_x, b_y, b_z, ..] = b;
    let x_sq = (a_x - b_x) * (a_x - b_x);
    let y_sq = (a_y - b_y) * (a_y - b_y);
    let z_sq = (a_z - b_z) * (a_z - b_z);
    (x_sq + y_sq + z_sq).sqrt() * 1000.0
}

/// Mean-equivalent of an osculating ECI state: osculating -> mean equinoctial -> ECI at the same
/// epoch. This is exactly the substitution the mean-element target correction performs.
fn mean_equivalent_state(eci: &[f64; 6]) -> anyhow::Result<[f64; 6]> {
    let mean = mean_equinoctial_from_osculating_state(eci)
        .context("sealed catalogue target must admit mean elements")?;
    let mut out = [0.0; 6];
    satpy_core::equinoc2eci_impl(&mean, 6, 0.0, 0.0, &mut out);
    anyhow::ensure!(
        out.iter().all(|v| v.is_finite()),
        "mean-equivalent state went non-finite"
    );
    Ok(out)
}

fn context_for(event: &Value, target_eci: [f64; 6]) -> anyhow::Result<PlanContext> {
    let sc = field(event, "selected_candidate")?;
    let native = field(event, "native_args")?;
    let resolved = field(native, "resolved_constellation_solve_native")?;
    let dep_eci = vecn(field(sc, "launch_pre_impulse_state")?)?;
    Ok(PlanContext {
        dep_eci,
        dep_equ: equ_of(&dep_eci)?,
        epoch_jd: num(sc, "base_epoch_jd")?,
        tgt_eci: target_eci,
        tgt_equ: equ_of(&target_eci)?,
        max_time_s: num(resolved, "max_time_s")?,
        tof_penalty_weight: num(resolved, "tof_penalty_weight")?,
        revolution_cap: num(resolved, "revolution_cap")?,
        max_phase_dv: num(resolved, "max_phase_dv")?,
        max_transfer_dv: num(resolved, "max_transfer_dv")?,
        min_perigee: num(resolved, "min_perigee")?,
        max_apogee: num(resolved, "max_apogee")?,
        max_revs: i32::try_from(integer(resolved, "max_revs")?)
            .context("fixture max_revs must fit i32")?,
        distance_tol: num(resolved, "distance_tol")?,
        deployer_min_distance: num(resolved, "deployer_min_distance")?,
        target_propagation_authority: TargetPropagationAuthority::MfJ2,
        target_body_force: two_phase_transfer_rs::types::BodyForceConfig::j2(
            BodyRole::DiagnosticTarget,
        ),
        ..PlanContext::with_j2_closure_settings(J2ClosureSettings::default())
    })
}

fn sealed_target_state(event: &Value) -> anyhow::Result<[f64; 6]> {
    let sc = field(event, "selected_candidate")?;
    let native = field(event, "native_args")?;
    let key = match integer(sc, "target_index")? {
        0 => "target1_eci_at_solver_epoch",
        1 => "target2_eci_at_solver_epoch",
        other => anyhow::bail!("unexpected sealed target index {other}"),
    };
    vecn(field(native, key)?)
}

fn candidate_for(event: &Value) -> anyhow::Result<PlanResult> {
    let sc = field(event, "selected_candidate")?;
    let mut result = PlanResult::invalid();
    result.valid = true;
    result.time2phase = num(sc, "time_to_phase_s")?;
    result.waittime = num(sc, "wait_time_s")?;
    result.tof = num(sc, "transfer_tof_s")?;
    result.phase_dv = vecn(field(sc, "phase_dv")?)?;
    result.transfer_dv = vecn(field(sc, "transfer_dv")?)?;
    Ok(result)
}

#[test]
fn hf_acceptance_residual_responds_to_a_mean_equivalent_catalogue_target() -> anyhow::Result<()> {
    let authority = HfGravityAuthority::load(DIR_R6_D15, gravity_only_transfer_force_config())
        .context("embedded production gravity coefficients must load")?;

    let fx = fixture()?;
    let events = field(&fx, "events")?.as_array().context("events array")?;

    println!(
        "\nA. target endpoint, MF arm vs HF arm, same context\n{:<7} {:>22} {:>16}",
        "event", "MF/HF target sep (m)", "bit-identical"
    );
    let mut arm_rows = Vec::new();
    for event in events {
        let index = integer(event, "event_index")?;
        let ctx = context_for(event, sealed_target_state(event)?)?;
        let candidate = candidate_for(event)?;
        let mf = replay_transfer_controls(&candidate, &ctx).context("MF replay")?;
        let hf = hf_acceptance_replay(&candidate, &ctx, &authority).context("HF replay")?;
        let bits_equal = mf
            .target_intercept_state
            .iter()
            .zip(hf.replayed.target_intercept_state.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "{index:<7} {:>22.3e} {bits_equal:>16}",
            sep_m(
                &mf.target_intercept_state,
                &hf.replayed.target_intercept_state
            )
        );
        arm_rows.push((index, bits_equal));
    }

    println!(
        "\nB. osculating vs mean-equivalent catalogue target, HF acceptance\n\
         {:<7} {:>16} {:>16} {:>14} {:>16} {:>16}",
        "event", "HF osc (m)", "HF mean (m)", "shift (m)", "E0 tgt move (m)", "tgt@I move (m)"
    );

    let mut rows = Vec::new();
    for event in events {
        let index = integer(event, "event_index")?;
        let osc_target = sealed_target_state(event)?;
        let mean_target = mean_equivalent_state(&osc_target)?;
        let candidate = candidate_for(event)?;

        let ctx_osc = context_for(event, osc_target)?;
        let ctx_mean = context_for(event, mean_target)?;

        let hf_osc =
            hf_acceptance_replay(&candidate, &ctx_osc, &authority).context("HF osc replay")?;
        let hf_mean =
            hf_acceptance_replay(&candidate, &ctx_mean, &authority).context("HF mean replay")?;
        let mf_osc = replay_transfer_controls(&candidate, &ctx_osc).context("MF osc replay")?;
        let mf_mean = replay_transfer_controls(&candidate, &ctx_mean).context("MF mean replay")?;

        let e0_move_m = sep_m(&osc_target, &mean_target);
        let intercept_move_m = sep_m(
            &hf_osc.replayed.target_intercept_state,
            &hf_mean.replayed.target_intercept_state,
        );
        let shift_m = hf_mean.residual_m - hf_osc.residual_m;

        println!(
            "{index:<7} {:>16.1} {:>16.1} {shift_m:>14.1} {e0_move_m:>16.1} \
             {intercept_move_m:>16.1}",
            hf_osc.residual_m, hf_mean.residual_m
        );

        // The payload arc cannot depend on the catalogue target; confirm it did not move, so the
        // whole residual shift is attributable to the target substitution.
        let payload_move_m = sep_m(
            &hf_osc.replayed.payload_intercept_state,
            &hf_mean.replayed.payload_intercept_state,
        );
        rows.push((
            index,
            hf_osc.residual_m,
            hf_mean.residual_m,
            mf_osc.distance * 1000.0,
            mf_mean.distance * 1000.0,
            e0_move_m,
            intercept_move_m,
            payload_move_m,
        ));
    }

    println!(
        "\n{:<7} {:>16} {:>16} {:>14} {:>18}",
        "event", "MF osc (m)", "MF mean (m)", "MF shift (m)", "payload move (m)"
    );
    for row in &rows {
        println!(
            "{:<7} {:>16.3} {:>16.1} {:>14.1} {:>18.3e}",
            row.0,
            row.3,
            row.4,
            row.4 - row.3,
            row.7
        );
    }
    println!();

    // A. Same context, both arms: the target leg is the same analytical J2 call, so it must be
    // bit-identical. This is what makes the MF/HF delta a transfer-arc measurement.
    for (index, bits_equal) in &arm_rows {
        anyhow::ensure!(
            bits_equal,
            "event {index}: MF and HF arms disagree on the catalogue target endpoint; the \
             MF-vs-HF delta would no longer isolate transfer-arc fidelity"
        );
    }

    for row in &rows {
        let (index, hf_osc, hf_mean, _, _, e0_move_m, intercept_move_m, payload_move_m) = *row;
        // The payload arc is target-independent by construction; verify, do not assume.
        anyhow::ensure!(
            payload_move_m < 1e-6,
            "event {index}: payload endpoint moved {payload_move_m:.3e} m under a target-only \
             substitution; the residual shift would not be attributable to the target"
        );
        // B. The substitution must actually be a substitution.
        anyhow::ensure!(
            e0_move_m > 1.0,
            "event {index}: mean-equivalent target is within {e0_move_m:.3} m of the osculating \
             state; this event does not exercise the question"
        );
        // The reported residual is NOT blind: a target-only change moves it.
        anyhow::ensure!(
            (hf_mean - hf_osc).abs() > 1.0,
            "event {index}: HF acceptance residual did not respond to a {e0_move_m:.1} m target \
             substitution (osc {hf_osc:.1} m, mean {hf_mean:.1} m); it would be blind to the \
             mean-element correction"
        );
        // ...and it responds because the target endpoint moved, not for some other reason.
        anyhow::ensure!(
            intercept_move_m > 1.0,
            "event {index}: target intercept state did not move under the substitution"
        );
    }

    Ok(())
}
