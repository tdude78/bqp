use std::fs::{remove_file, File};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;

use lightyear_odeint_rs::config::{self, GlobalCoeffs};
use lightyear_odeint_rs::integrator::{
    integrate_adaptive, ScalarGravityAssets, ScalarPropagationContext, ScalarPropagationRequest,
};
use lightyear_odeint_rs::types::StepperMethod;
use lightyear_odeint_rs::types::{BodyInvariants, ForceConfig, ForceFlags, InterpMethod};
use satpy_core::{eci2equinoc_impl, equinoc2eci_impl, pack_gravity_coeffs, PackedGravityCoeffs};

const AU_KM: f64 = 1.495_978_707e8;
const MOON_DIST_KM: f64 = 384_400.0;
const J2: f64 = -1.08263e-3;

const TF_S: f64 = 600.0;
const POS_TOL_KM: f64 = 0.01; // 10 m
const VEL_TOL_KM_S: f64 = 1e-5; // 1 cm/s

const MASS_KG: f64 = 4.0;
const AREA_M2: f64 = 0.01;
const AM_RATIO: f64 = AREA_M2 / MASS_KG; // ~0.0025 m^2/kg
const CD: f64 = 2.2;
const CR: f64 = 1.3;

static GLOBAL_COEFF_LOCK: std::sync::LazyLock<Mutex<()>> =
    std::sync::LazyLock::new(|| Mutex::new(()));

fn create_leo_state() -> [f64; 6] {
    let r = 7000.0; // km
    let mu = 398_600.441_5_f64;
    let v = (mu / r).sqrt();
    [r, 0.0, 0.0, 0.0, v, 0.0]
}

fn build_force_config(
    eps: f64,
    dt_max: f64,
    force_flags: i32,
    sph_order: usize,
    sun_pos: Option<[f64; 3]>,
    moon_pos: Option<[f64; 3]>,
) -> Arc<ForceConfig> {
    let mu_sun = 1.327_124_400_18e11;
    let mu_moon = 4_902.800_066;
    let mu_jupiter = 1.266_865_34e8;
    let mu_venus = 3.248_585_92e5;
    let mu_mars = 4.282_837_5e4;
    let mu_saturn = 3.793_120_6e7;

    let sun_invariants = sun_pos.and_then(|pos| BodyInvariants::precompute(&pos, mu_sun));
    let moon_invariants = moon_pos.and_then(|pos| BodyInvariants::precompute(&pos, mu_moon));

    Arc::new(ForceConfig {
        sph_order,
        force_flags,
        subtract_first_order: sph_order > 0,
        atm_model: 1,
        am_ratio: AM_RATIO,
        cd: CD,
        cr: CR,
        target_propagation_mode: 0,
        qm_ratio: 0.0,
        r_obj_m: 0.0,
        omega_earth: 7.292_115_0e-5,
        p_sun: 4.56e-6,
        mu_sun,
        mu_moon,
        mu_jupiter,
        mu_venus,
        mu_mars,
        mu_saturn,
        earth_radius: 6378.137,
        sun_pos,
        moon_pos,
        jupiter_pos: None,
        venus_pos: None,
        mars_pos: None,
        saturn_pos: None,
        dynamic_ephemeris_flags: 0,
        sun_invariants,
        moon_invariants,
        jupiter_invariants: None,
        venus_invariants: None,
        mars_invariants: None,
        saturn_invariants: None,
        dt_max,
        eps,
        integrator_method: lightyear_odeint_rs::types::StepperMethod::Dopri5Compat,
    })
}

fn make_coeffs(order: usize, j2: f64) -> anyhow::Result<Arc<PackedGravityCoeffs>> {
    let stride = order
        .checked_add(2)
        .context("gravity coefficient stride overflows")?;
    let total = stride
        .checked_mul(stride)
        .context("gravity coefficient table length overflows")?;
    let mut c = vec![0.0; total];
    let s = vec![0.0; total];
    *c.get_mut(0)
        .context("gravity coefficient table lacks the C00 element")? = 1.0;
    if order >= 2 {
        let j2_index = 2_usize
            .checked_mul(stride)
            .context("gravity coefficient J2 index overflows")?;
        *c.get_mut(j2_index)
            .context("gravity coefficient table lacks the J2 element")? = j2;
    }
    let packed =
        pack_gravity_coeffs(&c, &s, stride, order).context("gravity coefficients must pack")?;
    Ok(Arc::new(packed))
}

fn to_eci(init_equ: [f64; 6], dt_s: f64, delta: [f64; 6]) -> [f64; 6] {
    let mut base = [0.0; 6];
    equinoc2eci_impl(&init_equ, 6, dt_s, 0.0, &mut base);
    let mut out = [0.0; 6];
    for ((out_component, base_component), delta_component) in out.iter_mut().zip(base).zip(delta) {
        *out_component = base_component + delta_component;
    }
    out
}

fn run_solver(
    stepper: StepperMethod,
    init_equ: [f64; 6],
    t0_s: f64,
    t_final_s: f64,
    eps: f64,
    dt_max: f64,
    base_config: &ForceConfig,
    packed: Arc<PackedGravityCoeffs>,
) -> anyhow::Result<[f64; 6]> {
    let mut cfg = *base_config;
    cfg.dt_max = dt_max;
    cfg.eps = eps;
    // Active SRP freezes the requested scalar method in ForceConfig. This
    // diagnostic deliberately compares methods, so each isolated arm must
    // declare the same method through both authority and dispatch.
    cfg.integrator_method = stepper;
    let config = Arc::new(cfg);

    let gravity = ScalarGravityAssets::new(packed);
    let context = ScalarPropagationContext::new(2_460_000.5, config, gravity);
    let t_eval = [t_final_s];
    let result = integrate_adaptive(
        ScalarPropagationRequest::new(&context, init_equ, &t_eval, t0_s, t_final_s)
            .with_events(false),
    )
    .context("sampled solver propagation census failed")?;
    anyhow::ensure!(
        !result.terminal_event_fired && !result.max_steps_exceeded,
        "{stepper:?} solver arc did not complete: {}",
        result.terminal_event_name
    );
    let delta = result
        .states
        .last()
        .copied()
        .context("completed solver arc returned no state")?;

    Ok(to_eci(init_equ, t_final_s - t0_s, delta))
}

#[expect(
    clippy::suboptimal_flops,
    reason = "the pin keeps the explicit non-FMA accumulation order"
)]
fn err_pos_vel(a: [f64; 6], b: [f64; 6]) -> (f64, f64) {
    let [ax, ay, az, avx, avy, avz] = a;
    let [bx, by, bz, bvx, bvy, bvz] = b;
    let dx = ax - bx;
    let dy = ay - by;
    let dz = az - bz;
    let dvx = avx - bvx;
    let dvy = avy - bvy;
    let dvz = avz - bvz;
    let pos = (dx * dx) + (dy * dy) + (dz * dz);
    let vel = (dvx * dvx) + (dvy * dvy) + (dvz * dvz);
    (pos.sqrt(), vel.sqrt())
}

const fn solver_settings(stepper: StepperMethod) -> (f64, f64) {
    match stepper {
        StepperMethod::Dopri5Compat => (1e-14, 0.1),
        StepperMethod::Tsit5
        | StepperMethod::Dop853
        | StepperMethod::Rkv98
        | StepperMethod::Vern7
        | StepperMethod::Vern9
        | StepperMethod::Esdirk43
        | StepperMethod::Auto => (1e-9, 60.0),
    }
}

const fn solver_tolerance(stepper: StepperMethod) -> (f64, f64) {
    match stepper {
        // LOUD CAVEAT: this is a COMPAT-mode bound, not an accuracy bar. The
        // Dopri5Compat arm accepts a 2.1 km position error — 210x looser than
        // the 0.01 km every other stepper must meet. It exists so the legacy
        // scipy-shaped stepper keeps running through these scenarios at all;
        // any assertion that passes only under this arm proves compatibility,
        // NOT accuracy. Do not tighten it silently (the compat stepper cannot
        // meet the 10 m bar) and do not cite a Dopri5Compat pass as evidence
        // of solver accuracy.
        StepperMethod::Dopri5Compat => (2.1, 1e-2),
        _ => (POS_TOL_KM, VEL_TOL_KM_S),
    }
}

macro_rules! require_solver_result {
    ($result:expr) => {{
        let result = $result;
        assert!(result.is_ok(), "solver setup failed: {result:?}");
        let Ok(value) = result else {
            return;
        };
        value
    }};
}

/// Horizon for the deviation-event test, which needs its OWN and not the shared
/// `TF_S = 600 s`.
///
/// The Encke deviation grows with arc length, so the duration required to reach
/// the rectification threshold is a function of that threshold. At 600 s this
/// arc crossed the old 2 km bound; when
/// `PERTURB_DEVIATION_THRESHOLD_KM` moved to 10 km the deviation never reached
/// it, the event stopped firing, and the test failed asserting a feature was
/// exercised when it was not. That is worse than a wrong expected value: it read
/// as a physics regression when the scenario had simply stopped covering the
/// code it names.
///
/// The assertion below is written against the threshold constant rather than a
/// literal so the coupling is visible, but the DURATION still has to be large
/// enough to reach it — if this test starts failing after another threshold
/// increase, lengthen this, do not weaken the assertion.
const DEVIATION_EVENT_TF_S: f64 = 7_200.0;

#[test]
fn test_perturb_deviation_event_detection() -> anyhow::Result<()> {
    let packed = make_coeffs(2, J2)?;
    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let flags =
        ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
    let cfg = build_force_config(
        1e-9,
        60.0,
        flags,
        2,
        Some([AU_KM, 0.0, 0.0]),
        Some([MOON_DIST_KM, 0.0, 0.0]),
    );

    let gravity = ScalarGravityAssets::new(packed);
    let context = ScalarPropagationContext::new(2_460_000.5, cfg, gravity);
    let t_eval = [DEVIATION_EVENT_TF_S];
    let result = integrate_adaptive(
        ScalarPropagationRequest::new(&context, init_equ, &t_eval, 0.0, DEVIATION_EVENT_TF_S)
            .with_events(true),
    )
    .context("sampled deviation-event propagation census failed")?;

    anyhow::ensure!(
        result.perturb_deviation_fired,
        "expected perturb_deviation event to fire within {DEVIATION_EVENT_TF_S} s at a \
         {} km threshold; if the threshold rose again, lengthen \
         DEVIATION_EVENT_TF_S rather than weakening this assertion -- a test that \
         stops reaching its own trigger silently stops covering the feature it names",
        lightyear_odeint_rs::types::PERTURB_DEVIATION_THRESHOLD_KM
    );
    anyhow::ensure!(result.event_time.is_finite(), "event_time not finite");
    anyhow::ensure!(
        result.event_interp_method != InterpMethod::None,
        "event_interp_method empty"
    );
    Ok(())
}

#[test]
fn test_coeff_loader_and_baking() {
    let _guard = GLOBAL_COEFF_LOCK.lock().unwrap();
    let prev = config::GLOBAL_COEFFS.load();

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("egm_test_{ts}.txt"));
    let mut file = File::create(&path).expect("create temp EGM");
    writeln!(file, "2 0 {J2} 0.0").expect("write EGM");

    config::load_constants(path.to_str().unwrap(), 2).expect("load_constants");

    let packed_loaded = config::get_global_coeffs_packed().expect("coeffs loaded");

    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let config = build_force_config(1e-9, 60.0, 0, 2, None, None);
    let eci_loaded = require_solver_result!(run_solver(
        StepperMethod::Dopri5Compat,
        init_equ,
        0.0,
        TF_S,
        1e-9,
        60.0,
        &config,
        packed_loaded,
    ));

    let packed_pm = require_solver_result!(make_coeffs(0, 0.0));
    let config_pm = build_force_config(1e-9, 60.0, 0, 0, None, None);
    let eci_pm = require_solver_result!(run_solver(
        StepperMethod::Dopri5Compat,
        init_equ,
        0.0,
        TF_S,
        1e-9,
        60.0,
        &config_pm,
        packed_pm,
    ));

    let (pos_err, _) = err_pos_vel(eci_loaded, eci_pm);
    assert!(pos_err > 1e-6);

    config::GLOBAL_COEFFS.store(prev.clone());
    let _ = remove_file(&path);
}

#[test]
fn test_coeff_direct_store() {
    let _guard = GLOBAL_COEFF_LOCK.lock().unwrap();
    let prev = config::GLOBAL_COEFFS.load();

    let packed = require_solver_result!(make_coeffs(2, J2));
    config::GLOBAL_COEFFS.store(Arc::new(GlobalCoeffs::Loaded(packed)));

    let packed_loaded = config::get_global_coeffs_packed().expect("coeffs loaded");

    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let config = build_force_config(1e-9, 60.0, 0, 2, None, None);
    let eci_loaded = require_solver_result!(run_solver(
        StepperMethod::Dopri5Compat,
        init_equ,
        0.0,
        TF_S,
        1e-9,
        60.0,
        &config,
        packed_loaded,
    ));

    let packed_pm = require_solver_result!(make_coeffs(0, 0.0));
    let config_pm = build_force_config(1e-9, 60.0, 0, 0, None, None);
    let eci_pm = require_solver_result!(run_solver(
        StepperMethod::Dopri5Compat,
        init_equ,
        0.0,
        TF_S,
        1e-9,
        60.0,
        &config_pm,
        packed_pm,
    ));

    let (pos_err, _) = err_pos_vel(eci_loaded, eci_pm);
    assert!(pos_err > 1e-6);

    config::GLOBAL_COEFFS.store(prev.clone());
}

/// Honesty note: this proves only that the perturbation flags are NOT A
/// NO-OP — enabling DRAG|SRP|SUN|MOON must move the final state past a coarse
/// threshold. It bounds no magnitude, checks no sign, and validates no force
/// model; each force's own accuracy is covered elsewhere.
#[test]
fn test_perturbations_wiring() {
    let packed = require_solver_result!(make_coeffs(2, J2));
    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let cfg_none = build_force_config(1e-9, 60.0, 0, 2, None, None);
    let cfg_full = build_force_config(
        1e-9,
        60.0,
        ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY,
        2,
        Some([AU_KM, 0.0, 0.0]),
        Some([MOON_DIST_KM, 0.0, 0.0]),
    );

    let eci_none = require_solver_result!(run_solver(
        StepperMethod::Dopri5Compat,
        init_equ,
        0.0,
        TF_S,
        1e-9,
        60.0,
        &cfg_none,
        packed.clone(),
    ));

    let eci_full = require_solver_result!(run_solver(
        StepperMethod::Dopri5Compat,
        init_equ,
        0.0,
        TF_S,
        1e-9,
        60.0,
        &cfg_full,
        packed,
    ));

    let (pos_err, vel_err) = err_pos_vel(eci_none, eci_full);
    assert!(pos_err > 1e-6 || vel_err > 1e-9);
}

#[test]
fn test_solver_accuracy_forward() {
    let packed = require_solver_result!(make_coeffs(2, J2));
    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let flags =
        ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
    let cfg_ref = build_force_config(
        1e-12,
        10.0,
        flags,
        2,
        Some([AU_KM, 0.0, 0.0]),
        Some([MOON_DIST_KM, 0.0, 0.0]),
    );
    let cfg = build_force_config(
        1e-9,
        60.0,
        flags,
        2,
        Some([AU_KM, 0.0, 0.0]),
        Some([MOON_DIST_KM, 0.0, 0.0]),
    );

    let ref_eci = require_solver_result!(run_solver(
        StepperMethod::Dop853,
        init_equ,
        0.0,
        TF_S,
        1e-12,
        10.0,
        &cfg_ref,
        packed.clone(),
    ));

    // Vern7 is here because the campaign FLIES it. Until 2026-08-09 this set
    // was the four non-production steppers and the production one was absent,
    // so the only tableau nobody cross-checked was the only tableau that
    // mattered -- and the four that were checked looked, from a reference
    // count, like dead weight worth deleting.
    let solvers = [
        StepperMethod::Dopri5Compat,
        StepperMethod::Tsit5,
        StepperMethod::Dop853,
        StepperMethod::Rkv98,
        StepperMethod::Vern7,
    ];

    for solver in solvers {
        let (eps, dt_max) = solver_settings(solver);
        let eci = require_solver_result!(run_solver(
            solver,
            init_equ,
            0.0,
            TF_S,
            eps,
            dt_max,
            &cfg,
            packed.clone(),
        ));
        let (pos_err, vel_err) = err_pos_vel(eci, ref_eci);
        let (pos_tol, vel_tol) = solver_tolerance(solver);
        assert!(pos_err <= pos_tol, "{solver:?} pos_err={pos_err}");
        assert!(vel_err <= vel_tol, "{solver:?} vel_err={vel_err}");
    }
}

#[test]
fn test_solver_accuracy_backward() {
    let packed = require_solver_result!(make_coeffs(2, J2));
    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let flags =
        ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
    let cfg_ref = build_force_config(
        1e-12,
        10.0,
        flags,
        2,
        Some([AU_KM, 0.0, 0.0]),
        Some([MOON_DIST_KM, 0.0, 0.0]),
    );
    let cfg = build_force_config(
        1e-9,
        60.0,
        flags,
        2,
        Some([AU_KM, 0.0, 0.0]),
        Some([MOON_DIST_KM, 0.0, 0.0]),
    );

    let ref_eci = require_solver_result!(run_solver(
        StepperMethod::Dop853,
        init_equ,
        0.0,
        -TF_S,
        1e-12,
        10.0,
        &cfg_ref,
        packed.clone(),
    ));

    // Vern7 is here because the campaign FLIES it. Until 2026-08-09 this set
    // was the four non-production steppers and the production one was absent,
    // so the only tableau nobody cross-checked was the only tableau that
    // mattered -- and the four that were checked looked, from a reference
    // count, like dead weight worth deleting.
    let solvers = [
        StepperMethod::Dopri5Compat,
        StepperMethod::Tsit5,
        StepperMethod::Dop853,
        StepperMethod::Rkv98,
        StepperMethod::Vern7,
    ];

    for solver in solvers {
        let (eps, dt_max) = solver_settings(solver);
        let eci = require_solver_result!(run_solver(
            solver,
            init_equ,
            0.0,
            -TF_S,
            eps,
            dt_max,
            &cfg,
            packed.clone(),
        ));
        let (pos_err, vel_err) = err_pos_vel(eci, ref_eci);
        let (pos_tol, vel_tol) = solver_tolerance(solver);
        assert!(pos_err <= pos_tol, "{solver:?} pos_err={pos_err}");
        assert!(vel_err <= vel_tol, "{solver:?} vel_err={vel_err}");
    }
}

#[test]
fn test_round_trip_forward_backward() {
    let packed = require_solver_result!(make_coeffs(2, J2));
    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let flags =
        ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
    let cfg = build_force_config(
        1e-9,
        60.0,
        flags,
        2,
        Some([AU_KM, 0.0, 0.0]),
        Some([MOON_DIST_KM, 0.0, 0.0]),
    );

    // Vern7 is here because the campaign FLIES it. Until 2026-08-09 this set
    // was the four non-production steppers and the production one was absent,
    // so the only tableau nobody cross-checked was the only tableau that
    // mattered -- and the four that were checked looked, from a reference
    // count, like dead weight worth deleting.
    let solvers = [
        StepperMethod::Dopri5Compat,
        StepperMethod::Tsit5,
        StepperMethod::Dop853,
        StepperMethod::Rkv98,
        StepperMethod::Vern7,
    ];

    for solver in solvers {
        let (eps, dt_max) = solver_settings(solver);
        let eci_fwd = require_solver_result!(run_solver(
            solver,
            init_equ,
            0.0,
            TF_S,
            eps,
            dt_max,
            &cfg,
            packed.clone(),
        ));

        let mut equ_fwd = [0.0; 6];
        eci2equinoc_impl(&eci_fwd, 6, 0.0, 0.0, &mut equ_fwd);

        let eci_back = require_solver_result!(run_solver(
            solver,
            equ_fwd,
            0.0,
            -TF_S,
            eps,
            dt_max,
            &cfg,
            packed.clone(),
        ));

        let (pos_err, vel_err) = err_pos_vel(eci_back, init_eci);
        let (pos_tol, vel_tol) = solver_tolerance(solver);
        assert!(pos_err <= 2.0 * pos_tol, "{solver:?} pos_err={pos_err}");
        assert!(vel_err <= 2.0 * vel_tol, "{solver:?} vel_err={vel_err}");
    }
}

#[test]
fn test_round_trip_backward_forward() {
    let packed = require_solver_result!(make_coeffs(2, J2));
    let init_eci = create_leo_state();
    let mut init_equ = [0.0; 6];
    eci2equinoc_impl(&init_eci, 6, 0.0, 0.0, &mut init_equ);

    let flags =
        ForceFlags::DRAG | ForceFlags::SRP | ForceFlags::SUN_GRAVITY | ForceFlags::MOON_GRAVITY;
    let cfg = build_force_config(
        1e-9,
        60.0,
        flags,
        2,
        Some([AU_KM, 0.0, 0.0]),
        Some([MOON_DIST_KM, 0.0, 0.0]),
    );

    // Vern7 is here because the campaign FLIES it. Until 2026-08-09 this set
    // was the four non-production steppers and the production one was absent,
    // so the only tableau nobody cross-checked was the only tableau that
    // mattered -- and the four that were checked looked, from a reference
    // count, like dead weight worth deleting.
    let solvers = [
        StepperMethod::Dopri5Compat,
        StepperMethod::Tsit5,
        StepperMethod::Dop853,
        StepperMethod::Rkv98,
        StepperMethod::Vern7,
    ];

    for solver in solvers {
        let (eps, dt_max) = solver_settings(solver);
        let eci_back = require_solver_result!(run_solver(
            solver,
            init_equ,
            0.0,
            -TF_S,
            eps,
            dt_max,
            &cfg,
            packed.clone(),
        ));

        let mut equ_back = [0.0; 6];
        eci2equinoc_impl(&eci_back, 6, 0.0, 0.0, &mut equ_back);

        let eci_fwd = require_solver_result!(run_solver(
            solver,
            equ_back,
            0.0,
            TF_S,
            eps,
            dt_max,
            &cfg,
            packed.clone(),
        ));

        let (pos_err, vel_err) = err_pos_vel(eci_fwd, init_eci);
        let (pos_tol, vel_tol) = solver_tolerance(solver);
        assert!(pos_err <= 2.0 * pos_tol, "{solver:?} pos_err={pos_err}");
        assert!(vel_err <= 2.0 * vel_tol, "{solver:?} vel_err={vel_err}");
    }
}
