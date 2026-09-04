//! Legacy MF/gravity-only HF replay diagnostic.
//!
//! `docs/reference/dissertation/chapters/3_methodology.tex` (`tab:lane_perturb_policy`) assigns
//! `equinoc_prop_j2` mean-element secular J2 to the **bulk search lane only**, and states that
//! "Final acceptance requires typed HF status and fixed-I authority" — every canonical transfer
//! candidate is supposed to get an exact HF replay at the fixed target epoch.
//!
//! Part A does not do that. `nd_config::part_a_science` ships `require_high_fidelity: false`, the
//! sealed rows report `effective_use_high_fidelity: false` and `analytical_j2_only` on all four
//! body segments, and `verify::verify_transfer_result` gates its post-HF residual check on
//! `ctx.execution_policy.use_high_fidelity` — so with HF off the acceptance check re-propagates
//! with the *same* MF model that produced the candidate. The sealed sub-metre intercept miss is
//! therefore self-consistency of one reduced model, not a physical closure.
//!
//! This diagnostic replays each sealed `selected_candidate` twice — once on analytical J2
//! (reproducing the fixture, which validates the harness) and once on a gravity-only Encke
//! transfer-body context — and reports the intercept miss under each. It is not production Hybrid
//! acceptance physics.
//!
//! The target is propagated on MF/J2 in BOTH arms, so the reported delta isolates transfer-arc
//! fidelity rather than confounding it with target-model differences.
//!
//! The HF arm mirrors the production calling contract exactly: `subtract_first_order: true` (the
//! integrator returns an Encke delta against the analytic two-body baseline, so without it
//! central gravity is counted twice and the arc diverges), `.with_ephemeris_for_arc` stamping
//! before every call as `evaluate::stamped_body_force_config` does, and terminal events enabled.
//! Long legs are walked in `MAX_RECT_SEGMENT` chunks, matching the integrator's own 5400 s
//! rectification cap (`MAX_RECTIFICATION_SEGMENT_S` in `lightyear_odeint_rs::integrator`) and the one-LEO-orbit segment the methodology cites.
//!
//! The MF arm is the harness's self-check: it must reproduce the sealed sub-metre closure, and
//! does (0.9 / 0.8 / 0.8 m against sealed 0.853 / 0.780 / 0.781 m). Without that, the HF number
//! would carry no weight.
//!
//! Run with `--nocapture` to read the table.

use std::sync::Arc;

use anyhow::Context as _;
use lightyear_odeint_rs::types::ForceConfig;
use satpy_core::{equinoc_prop_from_impl, equinoc_prop_j2_from_impl};
use serde_json::Value;
use two_phase_transfer_rs::evaluate::{eci_to_equinoctial, TransferPropagationFailure};

const DIR_R6_D15: &[u8] = include_bytes!("../data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");
const SEC_PER_DAY: f64 = 86_400.0;

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
        "eci -> equinoctial failed for state {eci:?} -> {equ:?}"
    );
    Ok(equ)
}

fn miss_m(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let &[a_x, a_y, a_z, ..] = a;
    let &[b_x, b_y, b_z, ..] = b;
    let d_x = a_x - b_x;
    let d_y = a_y - b_y;
    let d_z = a_z - b_z;
    let x_sq = d_x * d_x;
    let y_sq = d_y * d_y;
    let z_sq = d_z * d_z;
    let planar_sq = x_sq + y_sq;
    (planar_sq + z_sq).sqrt() * 1000.0
}

fn apply_velocity_delta(mut state: [f64; 6], delta: [f64; 3]) -> [f64; 6] {
    let (_, velocity) = state.split_at_mut(3);
    for (component, increment) in velocity.iter_mut().zip(delta) {
        *component += increment;
    }
    state
}

/// Analytical secular-J2 leg — the MF lane's own propagator.
fn mf_leg(eci: &[f64; 6], dt: f64) -> anyhow::Result<[f64; 6]> {
    let equ = equ_of(eci)?;
    let mut out = [0.0; 6];
    equinoc_prop_j2_from_impl(&equ, dt, &mut out);
    anyhow::ensure!(out.iter().all(|x| x.is_finite()), "MF leg went non-finite");
    Ok(out)
}

/// One HF Encke sub-segment: two-body equinoctial baseline plus the integrated perturbation
/// delta, exactly the composition `evaluate::propagate_state_at_epoch` uses at its HF branch.
fn hf_segment(eci: &[f64; 6], dt: f64, jd: f64, cfg: &ForceConfig) -> anyhow::Result<[f64; 6]> {
    let equ = equ_of(eci)?;
    let packed = lightyear_odeint_rs::get_global_coeffs_packed()
        .context("missing high-fidelity gravity assets")?;
    // Third-body/SRP invariants must be stamped for the arc, exactly as
    // evaluate::stamped_body_force_config does before every production HF call.
    let stamped = Arc::new(
        cfg.with_ephemeris_for_arc(jd, jd + dt / SEC_PER_DAY)
            .context("stamp replay ephemeris")?,
    );
    let gravity = lightyear_odeint_rs::ScalarGravityAssets::new(packed);
    let context = lightyear_odeint_rs::ScalarPropagationContext::new(jd, stamped, gravity);
    let t_eval = [dt];
    let delta = lightyear_odeint_rs::integrate_final_checked(
        lightyear_odeint_rs::ScalarPropagationRequest::new(&context, equ, &t_eval, 0.0, dt)
            .with_events(true),
    )
    .map_err(TransferPropagationFailure::from)
    .context("integrate HF segment")?;
    let mut baseline = [0.0; 6];
    equinoc_prop_from_impl(&equ, dt, &mut baseline);
    let [b0, b1, b2, b3, b4, b5] = baseline;
    let [d0, d1, d2, d3, d4, d5] = delta;
    let out = [b0 + d0, b1 + d1, b2 + d2, b3 + d3, b4 + d4, b5 + d5];
    anyhow::ensure!(
        out.iter().all(|x| x.is_finite()),
        "HF segment went non-finite"
    );
    Ok(out)
}

/// HF leg, split into rectification segments. A single Encke call cannot span these arcs — the
/// checked final-state API returns a typed failure once the arc needs more steps than one
/// segment's budget —
/// so the leg is walked in `MAX_RECT_SEGMENT` chunks, re-osculating the equinoctial reference at
/// each boundary. That is the same 5400 s cap the integrator applies internally
/// (`MAX_RECTIFICATION_SEGMENT_S` in `lightyear_odeint_rs::integrator`) and that the methodology documents as one LEO orbital period.
fn hf_leg(eci: &[f64; 6], dt: f64, jd: f64, cfg: &ForceConfig) -> anyhow::Result<[f64; 6]> {
    use lightyear_odeint_rs::integrator::MAX_RECTIFICATION_SEGMENT_S as MAX_RECT_SEGMENT;
    let mut state = *eci;
    let mut done = 0.0;
    loop {
        if done >= dt {
            break;
        }
        let step = (dt - done).min(MAX_RECT_SEGMENT);
        state = hf_segment(&state, step, jd + done / SEC_PER_DAY, cfg)?;
        done += step;
    }
    Ok(state)
}

/// Replay the stored burn sequence E0 -> phase -> coast -> transfer -> intercept.
fn replay(sc: &Value, hf: Option<&ForceConfig>) -> anyhow::Result<[f64; 6]> {
    let launch: [f64; 6] = vecn(field(sc, "launch_pre_impulse_state")?)?;
    let phase_dv: [f64; 3] = vecn(field(sc, "phase_dv")?)?;
    let transfer_dv: [f64; 3] = vecn(field(sc, "transfer_dv")?)?;
    let t2p = num(sc, "time_to_phase_s")?;
    let wait = num(sc, "wait_time_s")?;
    let tof = num(sc, "transfer_tof_s")?;
    let base_jd = num(sc, "base_epoch_jd")?;

    let leg = |eci: &[f64; 6], dt: f64, jd: f64| {
        hf.map_or_else(|| mf_leg(eci, dt), |cfg| hf_leg(eci, dt, jd, cfg))
    };

    let phase_end = leg(&launch, t2p, base_jd)?;
    let at_phase = apply_velocity_delta(phase_end, phase_dv);
    let coast_end = leg(&at_phase, wait, base_jd + t2p / SEC_PER_DAY)?;
    let at_transfer = apply_velocity_delta(coast_end, transfer_dv);
    leg(&at_transfer, tof, base_jd + (t2p + wait) / SEC_PER_DAY)
}

#[test]
fn mf_selected_candidates_under_gravity_only_hf_diagnostic_replay() -> anyhow::Result<()> {
    lightyear_odeint_rs::load_constants_from_bytes(DIR_R6_D15, 5)
        .context("embedded production gravity coefficients must load")?;
    // Gravity-only 5x5 transfer-body context per tab:lane_perturb_policy. am/cd/cr stay finite
    // and positive because the HF transfer-body guard in evaluate.rs rejects a body without them;
    // force_flags = 0 keeps drag/SRP/third-body off, so they are inert here.
    // Gravity-only 5x5 transfer-body context per tab:lane_perturb_policy.
    // subtract_first_order is REQUIRED: the integrator returns an Encke delta against the
    // analytic two-body baseline, so without it central gravity is counted twice and the
    // "perturbation" diverges (14 km at 60 s, NaN by one hour - measured).
    let hf_cfg = ForceConfig {
        sph_order: 5,
        force_flags: 0,
        subtract_first_order: true,
        am_ratio: 0.01,
        cd: 2.2,
        cr: 1.3,
        target_propagation_mode: 1,
        dt_max: 300.0,
        ..ForceConfig::default()
    };

    let fx = fixture()?;
    let events = field(&fx, "events")?.as_array().context("events array")?;

    println!(
        "\n{:<7} {:>13} {:>14} {:>14} {:>12}",
        "event", "MF miss (m)", "HF miss (m)", "sealed (m)", "tol (m)"
    );

    let mut rows = Vec::new();
    for ev in events {
        let sc = field(ev, "selected_candidate")?;
        let index = integer(ev, "event_index")?;
        let total_time = num(sc, "total_time_s")?;

        // Target on MF/J2 in both arms, from the sealed intercept-epoch state the fixture stores.
        let target: [f64; 6] = vecn(field(sc, "target_intercept_state")?)?;
        let sealed_payload: [f64; 6] = vecn(field(sc, "payload_intercept_state")?)?;
        let sealed = miss_m(&sealed_payload, &target);

        let mf = miss_m(&replay(sc, None)?, &target);
        let hf = miss_m(&replay(sc, Some(&hf_cfg))?, &target);

        println!(
            "{index:<7} {mf:>13.1} {hf:>14.1} {sealed:>14.3} {:>12.1}",
            25.0
        );
        rows.push((index, mf, hf, sealed, total_time));
    }

    println!();
    for &(index, mf, hf, sealed, total_time) in &rows {
        println!(
            "event {index}: horizon {:.2} h, sealed closure {sealed:.3} m, MF replay {mf:.1} m, \
             HF replay {hf:.1} m",
            total_time / 3600.0
        );
    }

    let worst_hf = rows.iter().map(|r| r.2).fold(f64::NEG_INFINITY, f64::max);
    println!("\nworst HF acceptance miss: {worst_hf:.1} m (declared distance_tol 25.0 m)\n");

    // Record, do not gate. Whether the HF miss clears 25 m is the finding this test exists to
    // report; asserting it would turn a measurement into a gate that has not been earned yet.
    anyhow::ensure!(
        rows.iter().all(|r| r.2.is_finite()),
        "every HF replay must produce a finite miss distance"
    );
    Ok(())
}
