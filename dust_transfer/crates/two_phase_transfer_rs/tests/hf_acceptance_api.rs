//! Legacy MF/gravity-only HF replay diagnostic through the module API.
//!
//! `tests/hf_acceptance_replay.rs` established the finding by driving the integrator by hand. This
//! test drives the same three sealed MF candidates through `HfGravityAuthority` +
//! `hf_acceptance_replay` and must land on the same diagnostic numbers. It does not represent
//! compiled Hybrid acceptance physics.
//!
//! Two arms per candidate:
//!   * MF: `replay_transfer_controls` on the untouched (non-HF) context. This is the self-check —
//!     it must reproduce each sealed `j2_endpoint_residual_m` to sub-metre, otherwise the context
//!     reconstruction is wrong and the HF number would mean nothing.
//!   * HF: `hf_acceptance_replay` under the gravity-only 5x5 Encke model
//!     (`tab:lane_perturb_policy` row 2), legs chunked at `HF_REPLAY_MAX_SEGMENT_S`.
//!
//! The catalogue target is MF/J2 in both arms, so the delta isolates transfer-arc fidelity.
//!
//! Run with `--nocapture` to read the table.

use anyhow::Context as _;
use serde_json::Value;
use two_phase_transfer_rs::evaluate::eci_to_equinoctial;
use two_phase_transfer_rs::hf_acceptance::{
    gravity_only_transfer_force_config, hf_acceptance_replay, HfGravityAuthority,
};
use two_phase_transfer_rs::types::{BodyRole, PlanContext, PlanResult, TargetPropagationAuthority};
use two_phase_transfer_rs::{replay_transfer_controls, J2ClosureSettings};

const DIR_R6_D15: &[u8] = include_bytes!("../data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

/// Sealed MF fixture. Lives in `nd_pipeline` because that is where Stage-1 parity reads it.
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

/// Rebuild the solve context for one sealed event from its recorded native arguments.
fn context_for(event: &Value) -> anyhow::Result<PlanContext> {
    let sc = field(event, "selected_candidate")?;
    let native = field(event, "native_args")?;
    let resolved = field(native, "resolved_constellation_solve_native")?;

    let target_index = integer(sc, "target_index")?;
    let target_key = match target_index {
        0 => "target1_eci_at_solver_epoch",
        1 => "target2_eci_at_solver_epoch",
        other => anyhow::bail!("unexpected sealed target index {other}"),
    };

    let dep_eci: [f64; 6] = vecn(field(sc, "launch_pre_impulse_state")?)?;
    let tgt_eci: [f64; 6] = vecn(field(native, target_key)?)?;

    let ctx = PlanContext {
        dep_eci,
        dep_equ: equ_of(&dep_eci)?,
        epoch_jd: num(sc, "base_epoch_jd")?,
        tgt_eci,
        tgt_equ: equ_of(&tgt_eci)?,
        max_time_s: num(resolved, "max_time_s")?,
        tof_penalty_weight: num(resolved, "tof_penalty_weight")?,
        revolution_cap: num(resolved, "revolution_cap")?,
        max_phase_dv: num(resolved, "max_phase_dv")?,
        max_transfer_dv: num(resolved, "max_transfer_dv")?,
        min_perigee: num(resolved, "min_perigee")?,
        max_apogee: num(resolved, "max_apogee")?,
        max_revs: i32::try_from(integer(resolved, "max_revs")?)?,
        distance_tol: num(resolved, "distance_tol")?,
        deployer_min_distance: num(resolved, "deployer_min_distance")?,
        // The sealed rows report MfJ2 for the catalogue target; both arms keep it.
        target_propagation_authority: TargetPropagationAuthority::MfJ2,
        target_body_force: two_phase_transfer_rs::types::BodyForceConfig::j2(
            BodyRole::DiagnosticTarget,
        ),
        ..PlanContext::with_j2_closure_settings(J2ClosureSettings::default())
    };
    let authority = field(resolved, "target_propagation_authority")?
        .as_str()
        .context("target authority token")?;
    anyhow::ensure!(
        authority == "MfJ2",
        "fixture changed catalogue target authority; this test assumes MF/J2 on both arms"
    );
    Ok(ctx)
}

/// Rebuild the solved candidate from its sealed controls. No optimizer, no Lambert search.
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
fn mf_selected_candidates_report_gravity_only_hf_diagnostic_residuals() -> anyhow::Result<()> {
    let authority = HfGravityAuthority::load(DIR_R6_D15, gravity_only_transfer_force_config())
        .context("embedded production gravity coefficients must load")?;

    let fx = fixture()?;
    let events = field(&fx, "events")?.as_array().context("events array")?;

    println!(
        "\n{:<7} {:>13} {:>14} {:>14} {:>10}",
        "event", "MF miss (m)", "HF miss (m)", "sealed (m)", "tol (m)"
    );

    let mut rows = Vec::new();
    for event in events {
        let index = integer(event, "event_index")?;
        let ctx = context_for(event)?;
        let candidate = candidate_for(event)?;
        let sealed_m = num(
            field(event, "selected_candidate")?,
            "j2_endpoint_residual_m",
        )?;

        let mf = replay_transfer_controls(&candidate, &ctx).context("MF replay")?;
        let mf_m = mf.distance * 1000.0;

        let hf =
            hf_acceptance_replay(&candidate, &ctx, &authority).context("HF acceptance replay")?;

        println!(
            "{index:<7} {mf_m:>13.3} {:>14.1} {sealed_m:>14.3} {:>10.1}",
            hf.residual_m, hf.tolerance_m
        );
        rows.push((index, mf_m, hf, sealed_m));
    }

    println!();
    for (index, mf_m, hf, sealed_m) in &rows {
        println!(
            "event {index}: sealed {sealed_m:.3} m, MF replay {mf_m:.3} m, HF replay {:.1} m \
             ({:.1} km), accepted={}",
            hf.residual_m,
            hf.residual_m / 1000.0,
            hf.accepted
        );
    }

    // Self-check: the reconstructed context must reproduce the sealed MF closure. Without this,
    // the HF number below would be measuring a bad context, not a fidelity gap.
    for (index, mf_m, _, sealed_m) in &rows {
        anyhow::ensure!(
            (mf_m - sealed_m).abs() < 0.5,
            "event {index}: MF replay {mf_m:.6} m does not reproduce sealed {sealed_m:.6} m; \
             the reconstructed PlanContext is wrong"
        );
    }

    for (index, _, hf, _) in &rows {
        anyhow::ensure!(
            hf.residual_m.is_finite(),
            "event {index}: HF acceptance replay must produce a finite residual"
        );
        anyhow::ensure!(
            hf.accepted == (hf.residual_m <= hf.tolerance_m),
            "event {index}: acceptance flag must follow the residual"
        );
    }

    // The finding, pinned, in event order. These are the same three misses the by-hand harness in
    // tests/hf_acceptance_replay.rs reports — the shipped API must not
    // quietly measure something else. Wide bands (10%) because the assertion exists to catch a
    // changed model, not to freeze float noise.
    let expected_km = [146.0_f64, 1713.0, 82.0];
    // `zip` truncates to the shorter side, so without this the loop below
    // silently checks fewer events than it claims. `rows` comes from a fixture
    // owned by ANOTHER crate (`nd_pipeline`'s Stage-1 parity fixture, see the
    // header), so its length is not this test's to assume: regenerate it with
    // two events and the third residual check would vanish, green.
    anyhow::ensure!(
        rows.len() == expected_km.len(),
        "fixture supplies {} events but {} residuals are pinned; zip would drop the difference",
        rows.len(),
        expected_km.len()
    );
    for ((index, _, hf, _), expected) in rows.iter().zip(expected_km) {
        let measured_km = hf.residual_m / 1000.0;
        anyhow::ensure!(
            (measured_km - expected).abs() <= 0.10 * expected,
            "event {index}: HF acceptance miss {measured_km:.1} km is not within 10% of the \
             established {expected:.0} km"
        );
    }

    let worst_km = rows
        .iter()
        .map(|row| row.2.residual_m / 1000.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let tol_m = rows
        .first()
        .context("at least one fixture event")?
        .2
        .tolerance_m;
    println!("\nworst HF acceptance miss: {worst_km:.1} km (declared distance_tol {tol_m:.1} m)\n");

    // Every sealed candidate fails the mandated HF acceptance. Recorded, not silently tolerated.
    anyhow::ensure!(
        rows.iter().all(|row| !row.2.accepted),
        "a sealed candidate now clears HF acceptance; that is a physics change, re-read the table"
    );
    Ok(())
}
