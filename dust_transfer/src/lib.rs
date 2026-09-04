// Copyright (c) 2026 Truman DeWalch. All rights reserved.
// Licensed under the PolyForm Strict License 1.0.0 with the additional
// evaluation permission stated in LICENSE.md. Use only; no changes, no
// distribution, no commercial use. This is dust_transfer MVP 0.1.0.

//! Python bindings for the two-phase transfer optimizer and the Lightyear
//! high-fidelity propagator.
//!
//! Three things are exposed:
//!
//! * [`propagate`] / [`propagate_final`] — high-fidelity (Encke, spherical-harmonic gravity + JB2008
//!   drag + SRP + Sun/Moon) propagation of one ECI state.
//! * [`TransferProblem`] — the two-phase (phasing + Lambert transfer) intercept
//!   optimizer. `solve()` returns the Pareto front of candidates; `replay()`
//!   and `hf_verify()` re-fly one candidate under the medium-fidelity (J2) and
//!   high-fidelity models respectively.
//! * `kep_to_eci` / `eci_to_kep` — small element conversions so callers do not
//!   need another library to build a state.
//!
//! Units everywhere: km, km/s, seconds, Julian Date (UTC).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use lightyear_odeint_rs::integrator::{
    integrate_adaptive, integrate_final_checked, FinalPropagationFailure, ScalarGravityAssets,
    ScalarPropagationContext, ScalarPropagationRequest,
};
use lightyear_odeint_rs::types::{ForceConfig, ForceFlags, StepperMethod};
use satpy_core::{eci2kep_impl, equinoc2eci_impl, kep2eci_impl, PackedGravityCoeffs, SEC_PER_DAY};
use two_phase_transfer_rs::evaluate::eci_to_equinoctial;
use two_phase_transfer_rs::hf_acceptance::{hf_acceptance_replay, HfGravityAuthority};
use two_phase_transfer_rs::types::{
    BodyForceConfig, BodyRole, ExecutionPolicy, PlanContext, PlanResult, SamplingMode,
    SearchDepthPolicy, TargetPropagationAuthority, TransferLocalOptimizerConfig, TransferRequest,
};
use two_phase_transfer_rs::{replay_transfer_controls, solve_plan, J2ClosureSettings};

/// GOCE DIR-R6 spherical-harmonic gravity field, truncated to degree/order 15.
/// This is the same file the campaign flies; it is embedded so nothing has to
/// be located on disk at run time.
const DIR_R6_D15: &[u8] =
    include_bytes!("../crates/two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

const MAX_GRAVITY_ORDER: usize = 15;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Accept any 6-element sequence (list, tuple, numpy array).
fn six(value: &Bound<'_, PyAny>, what: &str) -> PyResult<[f64; 6]> {
    let v: Vec<f64> = value
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{what} must be a sequence of 6 floats")))?;
    let out: [f64; 6] = v
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{what} must have exactly 6 elements")))?;
    if !out.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err(format!("{what} contains a non-finite value")));
    }
    Ok(out)
}

fn three(list: &Bound<'_, PyAny>, what: &str) -> PyResult<[f64; 3]> {
    let v: Vec<f64> = list.extract()?;
    v.try_into()
        .map_err(|_| PyValueError::new_err(format!("{what} must have exactly 3 elements")))
}

fn equinoctial_of(eci: &[f64; 6], what: &str) -> PyResult<[f64; 6]> {
    let mut equ = [0.0; 6];
    if !eci_to_equinoctial(eci, &mut equ) {
        return Err(PyValueError::new_err(format!(
            "{what} is not a bound, non-degenerate elliptical orbit (equinoctial conversion failed)"
        )));
    }
    Ok(equ)
}

/// Gravity coefficient tables are parsed once per requested order and shared.
fn packed_gravity(order: usize) -> PyResult<Arc<PackedGravityCoeffs>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<PackedGravityCoeffs>>>> = OnceLock::new();
    if order == 0 || order > MAX_GRAVITY_ORDER {
        return Err(PyValueError::new_err(format!(
            "gravity_order must be in 1..={MAX_GRAVITY_ORDER}, got {order}"
        )));
    }
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| PyRuntimeError::new_err("gravity coefficient cache lock poisoned"))?;
    if let Some(packed) = guard.get(&order) {
        return Ok(Arc::clone(packed));
    }
    let packed = lightyear_odeint_rs::packed_constants_from_bytes(DIR_R6_D15, order)
        .map_err(|e| PyRuntimeError::new_err(format!("gravity coefficients failed to load: {e}")))?;
    guard.insert(order, Arc::clone(&packed));
    Ok(packed)
}

fn stepper_from_name(name: &str) -> PyResult<StepperMethod> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "vern7" => StepperMethod::Vern7,
        "vern9" => StepperMethod::Vern9,
        "dop853" => StepperMethod::Dop853,
        "tsit5" => StepperMethod::Tsit5,
        "rkv98" => StepperMethod::Rkv98,
        "dopri5" => StepperMethod::Dopri5Compat,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown integrator method {other:?}; use one of vern7, vern9, dop853, tsit5, rkv98, dopri5"
            )))
        }
    })
}

/// Build a force model from the user-facing switches.
#[allow(clippy::too_many_arguments)]
fn force_config(
    gravity_order: usize,
    drag: bool,
    srp: bool,
    sun: bool,
    moon: bool,
    atm_model: i32,
    am_ratio: f64,
    cd: f64,
    cr: f64,
    tol: f64,
    dt_max_s: f64,
    method: &str,
    target_mode: u8,
) -> PyResult<ForceConfig> {
    if gravity_order == 0 || gravity_order > MAX_GRAVITY_ORDER {
        return Err(PyValueError::new_err(format!(
            "gravity_order must be in 1..={MAX_GRAVITY_ORDER}, got {gravity_order}"
        )));
    }
    if !(am_ratio.is_finite() && am_ratio > 0.0 && cd.is_finite() && cd > 0.0 && cr.is_finite() && cr >= 0.0) {
        return Err(PyValueError::new_err(
            "am_ratio and cd must be finite and > 0, cr finite and >= 0",
        ));
    }
    if !(tol.is_finite() && tol > 0.0 && dt_max_s.is_finite() && dt_max_s > 0.0) {
        return Err(PyValueError::new_err("tol and dt_max_s must be finite and > 0"));
    }
    if drag && !matches!(atm_model, 1 | 3 | 4 | 5 | 6 | 7 | 8) {
        return Err(PyValueError::new_err(format!(
            "unknown atm_model {atm_model}; use 4 (exact JB2008), 7 (fitted JB2008), 8 (campaign persistence scenario), 1 (exponential)"
        )));
    }
    let mut flags = 0;
    if drag {
        flags |= ForceFlags::DRAG;
    }
    if srp {
        flags |= ForceFlags::SRP;
    }
    if sun {
        flags |= ForceFlags::SUN_GRAVITY;
    }
    if moon {
        flags |= ForceFlags::MOON_GRAVITY;
    }
    Ok(ForceConfig {
        sph_order: gravity_order,
        force_flags: flags,
        // The integrator returns an Encke delta against the two-body baseline;
        // this must be on or central gravity is counted twice.
        subtract_first_order: true,
        atm_model: if drag { atm_model } else { 0 },
        am_ratio,
        cd,
        cr,
        target_propagation_mode: target_mode,
        dt_max: dt_max_s,
        eps: tol,
        integrator_method: stepper_from_name(method)?,
        ..ForceConfig::default()
    })
}

fn state_to_py<'py>(py: Python<'py>, state: [f64; 6]) -> Bound<'py, PyArray1<f64>> {
    state.to_vec().into_pyarray(py)
}

// ---------------------------------------------------------------------------
// Element conversions
// ---------------------------------------------------------------------------

/// Keplerian elements `[a_km, e, i_deg, raan_deg, argp_deg, true_anomaly_deg]`
/// to an ECI state `[x, y, z, vx, vy, vz]` (km, km/s).
#[pyfunction]
fn kep_to_eci<'py>(py: Python<'py>, kep: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kep = six(kep, "kep")?;
    if kep[0] <= 0.0 || !(0.0..1.0).contains(&kep[1]) {
        return Err(PyValueError::new_err("need a_km > 0 and 0 <= e < 1"));
    }
    let mut out = [0.0; 6];
    kep2eci_impl(&kep, true, 0.0, 0.0, true, &mut out);
    Ok(state_to_py(py, out))
}

/// ECI state (km, km/s) to Keplerian elements
/// `[a_km, e, i_deg, raan_deg, argp_deg, true_anomaly_deg]`.
#[pyfunction]
fn eci_to_kep<'py>(py: Python<'py>, eci: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let eci = six(eci, "eci")?;
    let mut out = [0.0; 6];
    eci2kep_impl(&eci, true, true, &mut out);
    Ok(state_to_py(py, out))
}

// ---------------------------------------------------------------------------
// High-fidelity propagation
// ---------------------------------------------------------------------------

/// Propagate one ECI state with the high-fidelity force model.
///
/// Returns a dict with `times_s` (n,), `states_eci` (n, 6), `completed`,
/// `terminal_event`, `steps` and `rhs_evals`. The sample grid always starts at
/// t = 0 and ends at `tof_s`; `times_s` (seconds after `epoch_jd`) adds
/// samples in between.
///
/// This is the rectified sampled path (Encke re-baselined every orbit). It
/// has no terminal-event detection: an orbit that hits the ground keeps
/// integrating. Use `propagate_final` when you want impact / escape checks.
#[pyfunction]
#[pyo3(signature = (
    state_eci, epoch_jd, tof_s, times_s=None, *,
    gravity_order=5, drag=true, srp=true, sun=true, moon=true, atm_model=7,
    am_ratio=0.01, cd=2.2, cr=1.3, tol=1e-8, dt_max_s=300.0, method="vern7",
))]
#[allow(clippy::too_many_arguments)]
fn propagate<'py>(
    py: Python<'py>,
    state_eci: &Bound<'py, PyAny>,
    epoch_jd: f64,
    tof_s: f64,
    times_s: Option<&Bound<'py, PyAny>>,
    gravity_order: usize,
    drag: bool,
    srp: bool,
    sun: bool,
    moon: bool,
    atm_model: i32,
    am_ratio: f64,
    cd: f64,
    cr: f64,
    tol: f64,
    dt_max_s: f64,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let init_eci = six(state_eci, "state_eci")?;
    if !(epoch_jd.is_finite() && tof_s.is_finite() && tof_s > 0.0) {
        return Err(PyValueError::new_err("epoch_jd must be finite and tof_s > 0"));
    }
    let user_times: Vec<f64> = match times_s {
        Some(t) => t
            .extract()
            .map_err(|_| PyValueError::new_err("times_s must be a sequence of floats"))?,
        None => vec![tof_s],
    };
    if user_times.windows(2).any(|w| w[1] <= w[0])
        || user_times.iter().any(|t| !t.is_finite() || *t < 0.0 || *t > tof_s)
    {
        return Err(PyValueError::new_err(
            "times_s must be strictly increasing, within [0, tof_s]",
        ));
    }
    // The event-aware sampled path requires the sample grid to span the whole
    // arc, so the initial and final epochs are always part of the grid.
    let mut t_eval = Vec::with_capacity(user_times.len() + 2);
    if user_times.first().is_none_or(|t| *t > 0.0) {
        t_eval.push(0.0);
    }
    t_eval.extend_from_slice(&user_times);
    if user_times.last().is_none_or(|t| *t < tof_s) {
        t_eval.push(tof_s);
    }

    let config = force_config(
        gravity_order, drag, srp, sun, moon, atm_model, am_ratio, cd, cr, tol, dt_max_s, method, 0,
    )?
    .with_ephemeris_for_arc(epoch_jd, epoch_jd + tof_s / SEC_PER_DAY)
    .map_err(|e| PyValueError::new_err(format!("ephemeris / JB2008 driver coverage: {e}")))?;
    let packed = packed_gravity(gravity_order)?;
    let init_equ = equinoctial_of(&init_eci, "state_eci")?;

    let result = py.detach(|| {
        let context = ScalarPropagationContext::new(
            epoch_jd,
            Arc::new(config),
            ScalarGravityAssets::new(packed),
        );
        integrate_adaptive(
            ScalarPropagationRequest::new(&context, init_equ, &t_eval, 0.0, tof_s)
                .with_events(false),
        )
    })
    .map_err(|e| PyRuntimeError::new_err(format!("propagation census failure: {e:?}")))?;

    // The integrator returns Encke deltas against the analytic two-body
    // baseline from the initial equinoctial elements; add the baseline back.
    let n = result.times.len();
    let mut flat = Vec::with_capacity(n * 6);
    for (t, delta) in result.times.iter().zip(&result.states) {
        let mut base = [0.0; 6];
        equinoc2eci_impl(&init_equ, 6, *t, 0.0, &mut base);
        for (b, d) in base.iter().zip(delta) {
            flat.push(b + d);
        }
    }
    let states = PyArray1::from_vec(py, flat)
        .reshape([n, 6])
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let completed = !result.terminal_event_fired
        && !result.max_steps_exceeded
        && result.terminal_gravity_error.is_none()
        && result.terminal_eclipse_error.is_none()
        && n == t_eval.len();
    let terminal_event = if !result.terminal_event_name.is_empty() {
        result.terminal_event_name.to_string()
    } else if let Some(error) = result.terminal_gravity_error {
        format!("gravity_{error:?}").to_ascii_lowercase()
    } else if let Some(error) = result.terminal_eclipse_error {
        format!("eclipse_{error:?}").to_ascii_lowercase()
    } else if !completed {
        "incomplete".to_owned()
    } else {
        String::new()
    };
    let out = PyDict::new(py);
    out.set_item("times_s", result.times.clone().into_pyarray(py))?;
    out.set_item("states_eci", states)?;
    out.set_item("completed", completed)?;
    out.set_item("terminal_event", terminal_event)?;
    out.set_item("max_steps_exceeded", result.max_steps_exceeded)?;
    out.set_item("steps", result.metrics.total_steps)?;
    out.set_item("rhs_evals", result.metrics.total_evals)?;
    out.set_item("wall_us", result.metrics.total_time_us)?;
    Ok(out)
}

/// Propagate one ECI state and return only the final state, with physical
/// checks (ground impact, escape, eccentricity blow-up) enforced.
///
/// Raises `ValueError` when the arc is physically infeasible and
/// `RuntimeError` on an integration failure. Same force-model kwargs as
/// `propagate`.
#[pyfunction]
#[pyo3(signature = (
    state_eci, epoch_jd, tof_s, *,
    gravity_order=5, drag=true, srp=true, sun=true, moon=true, atm_model=7,
    am_ratio=0.01, cd=2.2, cr=1.3, tol=1e-8, dt_max_s=300.0, method="vern7",
))]
#[allow(clippy::too_many_arguments)]
fn propagate_final<'py>(
    py: Python<'py>,
    state_eci: &Bound<'py, PyAny>,
    epoch_jd: f64,
    tof_s: f64,
    gravity_order: usize,
    drag: bool,
    srp: bool,
    sun: bool,
    moon: bool,
    atm_model: i32,
    am_ratio: f64,
    cd: f64,
    cr: f64,
    tol: f64,
    dt_max_s: f64,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let init_eci = six(state_eci, "state_eci")?;
    if !(epoch_jd.is_finite() && tof_s.is_finite() && tof_s > 0.0) {
        return Err(PyValueError::new_err("epoch_jd must be finite and tof_s > 0"));
    }
    let config = force_config(
        gravity_order, drag, srp, sun, moon, atm_model, am_ratio, cd, cr, tol, dt_max_s, method, 0,
    )?
    .with_ephemeris_for_arc(epoch_jd, epoch_jd + tof_s / SEC_PER_DAY)
    .map_err(|e| PyValueError::new_err(format!("ephemeris / JB2008 driver coverage: {e}")))?;
    let packed = packed_gravity(gravity_order)?;
    let init_equ = equinoctial_of(&init_eci, "state_eci")?;
    let t_eval = [tof_s];
    let delta = py.detach(|| {
        let context = ScalarPropagationContext::new(
            epoch_jd,
            Arc::new(config),
            ScalarGravityAssets::new(packed),
        );
        integrate_final_checked(
            ScalarPropagationRequest::new(&context, init_equ, &t_eval, 0.0, tof_s).with_events(true),
        )
    });
    let delta = match delta {
        Ok(delta) => delta,
        Err(failure) if failure.is_physical_infeasible() => {
            let why = match failure {
                FinalPropagationFailure::Ground => "trajectory hits the ground",
                FinalPropagationFailure::LeftEarth => "trajectory escapes Earth",
                _ => "trajectory eccentricity left the valid range",
            };
            return Err(PyValueError::new_err(format!("physically infeasible arc: {why}")));
        }
        Err(FinalPropagationFailure::Gravity(satpy_core::GravityError::InvalidRadius)) => {
            return Err(PyValueError::new_err(
                "physically infeasible arc: radius fell below the Earth's surface",
            ));
        }
        Err(FinalPropagationFailure::Eclipse(lightyear_odeint_rs::EclipseError::Envelope)) => {
            return Err(PyValueError::new_err(
                "state outside the SRP eclipse envelope (radius 6000-50000 km, speed <= 20 km/s); pass srp=False",
            ));
        }
        Err(failure) => {
            return Err(PyRuntimeError::new_err(format!("propagation failed: {failure:?}")));
        }
    };
    let mut state = [0.0; 6];
    equinoc2eci_impl(&init_equ, 6, tof_s, 0.0, &mut state);
    for (s, d) in state.iter_mut().zip(delta) {
        *s += d;
    }
    Ok(state_to_py(py, state))
}

// ---------------------------------------------------------------------------
// Transfer optimizer
// ---------------------------------------------------------------------------

fn candidate_to_py<'py>(py: Python<'py>, c: &PlanResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("valid", c.valid)?;
    d.set_item("cost", c.cost)?;
    // Timeline (seconds from epoch): phase burn at 0, coast `time2phase`,
    // wait `waittime`, transfer burn, Lambert arc of `tof`, intercept.
    d.set_item("time2phase_s", c.time2phase)?;
    d.set_item("waittime_s", c.waittime)?;
    d.set_item("tof_s", c.tof)?;
    d.set_item("total_time_s", c.time2phase + c.waittime + c.tof)?;
    d.set_item("phase_sma_km", c.phase_sma)?;
    d.set_item("phase_dv", state_to_py3(py, c.phase_dv))?;
    d.set_item("transfer_dv", state_to_py3(py, c.transfer_dv))?;
    d.set_item("arrival_dv", state_to_py3(py, c.arrival_dv))?;
    d.set_item("phase_dv_norm", c.phase_dv_norm)?;
    d.set_item("transfer_dv_norm", c.transfer_dv_norm)?;
    d.set_item("arrival_dv_norm", c.arrival_dv_norm)?;
    // Intercept, not rendezvous: the arrival burn is reported but not spent.
    d.set_item("total_dv", c.phase_dv_norm + c.transfer_dv_norm)?;
    d.set_item("miss_distance_km", c.distance)?;
    d.set_item("deployer_distance_km", c.deployer_distance)?;
    d.set_item("release_state", state_to_py(py, c.release_state))?;
    d.set_item("payload_intercept_state", state_to_py(py, c.payload_intercept_state))?;
    d.set_item("target_intercept_state", state_to_py(py, c.target_intercept_state))?;
    d.set_item("deployer_intercept_state", state_to_py(py, c.deployer_intercept_state))?;
    d.set_item("intercept_jd", c.intercept_jd)?;
    d.set_item("lambert_revs", c.best_M)?;
    d.set_item("prograde", c.prograde)?;
    d.set_item("branch_status", format!("{:?}", c.branch_status))?;
    d.set_item("branch_rejection", format!("{:?}", c.branch_rejection))?;
    d.set_item("timing_failure", format!("{:?}", c.timing_failure_reason))?;
    d.set_item("func_evals", c.func_evals)?;
    d.set_item("optimizer_converged", c.optimizer_converged)?;
    d.set_item("post_hf_endpoint_residual_m", c.post_hf_endpoint_residual_m)?;
    // Raw optimizer coordinates, kept so a candidate can be re-created exactly.
    d.set_item("time2phase_ratio", c.time2phase_ratio)?;
    d.set_item("phase_sma_ratio", c.phase_sma_ratio)?;
    d.set_item("waittime_ratio", c.waittime_ratio)?;
    Ok(d)
}

fn state_to_py3<'py>(py: Python<'py>, v: [f64; 3]) -> Bound<'py, PyArray1<f64>> {
    v.to_vec().into_pyarray(py)
}

/// Rebuild the controls of a candidate (as returned by `solve()`) so it can be
/// re-flown. Only the five control fields matter; everything else is recomputed.
fn candidate_from_py(d: &Bound<'_, PyDict>) -> PyResult<PlanResult> {
    fn get_f64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
        d.get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("candidate is missing {key:?}")))?
            .extract()
    }
    fn get_vec3(d: &Bound<'_, PyDict>, key: &str) -> PyResult<[f64; 3]> {
        let item = d
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("candidate is missing {key:?}")))?;
        three(&item, key)
    }
    let mut c = PlanResult::invalid();
    c.valid = true;
    c.time2phase = get_f64(d, "time2phase_s")?;
    c.waittime = get_f64(d, "waittime_s")?;
    c.tof = get_f64(d, "tof_s")?;
    c.phase_dv = get_vec3(d, "phase_dv")?;
    c.transfer_dv = get_vec3(d, "transfer_dv")?;
    Ok(c)
}

/// One deployer→target intercept problem.
///
/// Construct with the two ECI states at a common epoch, then call `solve()`.
/// Defaults are the campaign's sealed MF-transfer controls.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct TransferProblem {
    dep_eci: [f64; 6],
    tgt_eci: [f64; 6],
    epoch_jd: f64,
    max_time_s: f64,
    max_phase_dv: f64,
    max_transfer_dv: f64,
    max_revs: i32,
    min_perigee_km: f64,
    max_apogee_km: f64,
    tof_penalty_weight: f64,
    revolution_cap: f64,
    distance_tol_km: f64,
    deployer_min_distance_km: f64,
    tof_sample_budget: usize,
    coarse_early_stop: bool,
    fine_total_limit: usize,
    coarse_reject_margin_km_s: f64,
    seed_fine_margin_km_s: f64,
    j2_max_iterations: usize,
    j2_endpoint_target_km: f64,
    j2_correction_step_gain: f64,
    seed: u64,
    parallel: bool,
}

impl TransferProblem {
    fn context(&self) -> PyResult<PlanContext> {
        let dep_equ = equinoctial_of(&self.dep_eci, "deployer_eci")?;
        let tgt_equ = equinoctial_of(&self.tgt_eci, "target_eci")?;
        let j2 = J2ClosureSettings {
            max_iterations: self.j2_max_iterations,
            endpoint_target_km: self.j2_endpoint_target_km,
            correction_step_gain: self.j2_correction_step_gain,
        };
        Ok(PlanContext::from_request(TransferRequest {
            dep_eci: self.dep_eci,
            dep_equ,
            epoch_jd: self.epoch_jd,
            tgt_eci: self.tgt_eci,
            tgt_equ,
            max_time_s: self.max_time_s,
            tof_penalty_weight: self.tof_penalty_weight,
            revolution_cap: self.revolution_cap,
            max_phase_dv: self.max_phase_dv,
            max_transfer_dv: self.max_transfer_dv,
            min_perigee: self.min_perigee_km,
            max_apogee: self.max_apogee_km,
            max_revs: self.max_revs,
            sampling_mode: SamplingMode::Fast,
            execution_policy: ExecutionPolicy {
                allow_parallel: self.parallel,
                ..ExecutionPolicy::default()
            },
            j2_closure_settings: j2,
            search_depth: SearchDepthPolicy {
                tof_sample_budget: self.tof_sample_budget,
                coarse_early_stop: self.coarse_early_stop,
                fine_total_limit: self.fine_total_limit,
                coarse_reject_margin_km_s: self.coarse_reject_margin_km_s,
                seed_fine_margin_km_s: self.seed_fine_margin_km_s,
                ..SearchDepthPolicy::default()
            },
            distance_tol: self.distance_tol_km,
            deployer_min_distance: self.deployer_min_distance_km,
            // The candidate search is medium-fidelity by design: the target
            // is propagated under J2 and each candidate is closed with the
            // J2 corrector. High fidelity enters through `hf_verify`.
            target_propagation_authority: TargetPropagationAuthority::MfJ2,
            target_body_force: BodyForceConfig::j2(BodyRole::DiagnosticTarget),
            force_config: None,
            packed_coeffs: None,
            polish_metrics: None,
            local_optimizer: TransferLocalOptimizerConfig {
                seed: self.seed,
                ..TransferLocalOptimizerConfig::default()
            },
        }))
    }
}

#[pymethods]
impl TransferProblem {
    #[new]
    #[pyo3(signature = (
        deployer_eci, target_eci, epoch_jd, *,
        max_time_s=172_800.0, max_phase_dv=1.25, max_transfer_dv=1.25, max_revs=4,
        min_perigee_km=6578.137, max_apogee_km=41378.137,
        tof_penalty_weight=0.1, revolution_cap=2.0,
        distance_tol_km=0.025, deployer_min_distance_km=0.12,
        tof_sample_budget=256, coarse_early_stop=false, fine_total_limit=10,
        coarse_reject_margin_km_s=0.15, seed_fine_margin_km_s=0.15,
        j2_max_iterations=5, j2_endpoint_target_km=0.01, j2_correction_step_gain=1.0,
        seed=42, parallel=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        deployer_eci: &Bound<'_, PyAny>,
        target_eci: &Bound<'_, PyAny>,
        epoch_jd: f64,
        max_time_s: f64,
        max_phase_dv: f64,
        max_transfer_dv: f64,
        max_revs: i32,
        min_perigee_km: f64,
        max_apogee_km: f64,
        tof_penalty_weight: f64,
        revolution_cap: f64,
        distance_tol_km: f64,
        deployer_min_distance_km: f64,
        tof_sample_budget: usize,
        coarse_early_stop: bool,
        fine_total_limit: usize,
        coarse_reject_margin_km_s: f64,
        seed_fine_margin_km_s: f64,
        j2_max_iterations: usize,
        j2_endpoint_target_km: f64,
        j2_correction_step_gain: f64,
        seed: u64,
        parallel: bool,
    ) -> PyResult<Self> {
        let dep_eci = six(deployer_eci, "deployer_eci")?;
        let tgt_eci = six(target_eci, "target_eci")?;
        if !(epoch_jd.is_finite() && max_time_s.is_finite() && max_time_s > 0.0) {
            return Err(PyValueError::new_err("epoch_jd must be finite and max_time_s > 0"));
        }
        if max_revs < 0 {
            return Err(PyValueError::new_err("max_revs must be >= 0"));
        }
        let problem = Self {
            dep_eci,
            tgt_eci,
            epoch_jd,
            max_time_s,
            max_phase_dv,
            max_transfer_dv,
            max_revs,
            min_perigee_km,
            max_apogee_km,
            tof_penalty_weight,
            revolution_cap,
            distance_tol_km,
            deployer_min_distance_km,
            tof_sample_budget,
            coarse_early_stop,
            fine_total_limit,
            coarse_reject_margin_km_s,
            seed_fine_margin_km_s,
            j2_max_iterations,
            j2_endpoint_target_km,
            j2_correction_step_gain,
            seed,
            parallel,
        };
        // Fail early on a degenerate state rather than inside `solve()`.
        problem.context()?;
        Ok(problem)
    }

    /// Run the two-phase transfer search. Returns the Pareto front of valid
    /// candidates (list of dicts), sorted by total delta-v.
    fn solve<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut ctx = self.context()?;
        let front = py
            .detach(|| solve_plan(&mut ctx, None))
            .map_err(|e| PyRuntimeError::new_err(format!("transfer solve failed: {e:?}")))?;
        let mut candidates: Vec<&PlanResult> = front.candidates.iter().filter(|c| c.valid).collect();
        candidates.sort_by(|a, b| {
            let ka = a.phase_dv_norm + a.transfer_dv_norm;
            let kb = b.phase_dv_norm + b.transfer_dv_norm;
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let list = PyList::empty(py);
        for c in candidates {
            list.append(candidate_to_py(py, c)?)?;
        }
        Ok(list)
    }

    /// Re-fly one candidate under the medium-fidelity (J2) model used by the
    /// search and report the endpoint miss. A tiny miss confirms the candidate
    /// is self-consistent.
    fn replay<'py>(&self, py: Python<'py>, candidate: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
        let ctx = self.context()?;
        let c = candidate_from_py(candidate)?;
        let replayed = py
            .detach(|| replay_transfer_controls(&c, &ctx))
            .map_err(|e| PyRuntimeError::new_err(format!("MF replay failed: {e:?}")))?;
        let out = candidate_to_py(py, &replayed)?;
        out.set_item("residual_m", replayed.distance * 1000.0)?;
        Ok(out)
    }

    /// Re-fly one candidate's transfer arc under the high-fidelity propagator
    /// and report the endpoint miss against the tolerance.
    ///
    /// `forces="gravity"` flies spherical-harmonic gravity only (the classic
    /// acceptance diagnostic); `forces="full"` adds JB2008 drag, SRP and
    /// Sun/Moon third-body gravity for the vehicle described by
    /// `am_ratio`/`cd`/`cr`.
    #[pyo3(signature = (candidate, *, forces="gravity", gravity_order=5, am_ratio=0.01, cd=2.2, cr=1.3,
                        atm_model=7, tol=1e-8, dt_max_s=300.0, method="vern7"))]
    #[allow(clippy::too_many_arguments)]
    fn hf_verify<'py>(
        &self,
        py: Python<'py>,
        candidate: &Bound<'py, PyDict>,
        forces: &str,
        gravity_order: usize,
        am_ratio: f64,
        cd: f64,
        cr: f64,
        atm_model: i32,
        tol: f64,
        dt_max_s: f64,
        method: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let full = match forces {
            "gravity" => false,
            "full" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "forces must be \"gravity\" or \"full\", got {other:?}"
                )))
            }
        };
        let ctx = self.context()?;
        let c = candidate_from_py(candidate)?;
        let mut config = force_config(
            gravity_order,
            full,
            full,
            full,
            full,
            atm_model,
            am_ratio,
            cd,
            cr,
            tol,
            dt_max_s,
            method,
            TargetPropagationAuthority::MfJ2.as_force_config_code(),
        )?;
        if full {
            let arc_end = self.epoch_jd + (c.time2phase + c.waittime + c.tof) / SEC_PER_DAY;
            config = config
                .with_ephemeris_for_arc(self.epoch_jd, arc_end)
                .map_err(|e| PyValueError::new_err(format!("ephemeris / JB2008 driver coverage: {e}")))?;
        }
        let authority = HfGravityAuthority::load(DIR_R6_D15, config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let report = py
            .detach(|| hf_acceptance_replay(&c, &ctx, &authority))
            .map_err(|e| PyRuntimeError::new_err(format!("HF replay failed: {e:?}")))?;
        let out = PyDict::new(py);
        out.set_item("residual_m", report.residual_m)?;
        out.set_item("tolerance_m", report.tolerance_m)?;
        out.set_item("accepted", report.accepted)?;
        out.set_item("replayed", candidate_to_py(py, &report.replayed)?)?;
        Ok(out)
    }

    #[getter]
    fn epoch_jd(&self) -> f64 {
        self.epoch_jd
    }

    #[getter]
    fn deployer_eci<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        state_to_py(py, self.dep_eci)
    }

    #[getter]
    fn target_eci<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        state_to_py(py, self.tgt_eci)
    }

    fn __repr__(&self) -> String {
        format!(
            "TransferProblem(epoch_jd={}, max_time_s={}, max_phase_dv={}, max_transfer_dv={}, max_revs={})",
            self.epoch_jd, self.max_time_s, self.max_phase_dv, self.max_transfer_dv, self.max_revs
        )
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(propagate, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_final, m)?)?;
    m.add_function(wrap_pyfunction!(kep_to_eci, m)?)?;
    m.add_function(wrap_pyfunction!(eci_to_kep, m)?)?;
    m.add_class::<TransferProblem>()?;
    m.add("MU_EARTH_KM3_S2", satpy_core::MU)?;
    Ok(())
}
