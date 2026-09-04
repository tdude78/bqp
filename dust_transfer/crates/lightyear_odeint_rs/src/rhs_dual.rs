//! `DualVec` support for Lightyear ODE Integrator
//!
//! Provides automatic differentiation capabilities for high-fidelity propagation.

use crate::types::{ForceConfig, ForceFlags, MU};
use anyhow::Context;
use jb_rs::synthetic_thermosphere_proxy_eval_impl;
use num_traits::Float;
use satpy_core::SEC_PER_DAY;
use satpy_core::{
    eci_to_geocentric_spherical, equinoc2eci_impl, greenwichsrt_impl,
    spherical_gravity_impl_generic_packed, DualVec, GravityCacheGeneric, GravityError,
    PackedGravityCoeffs,
};
use std::cell::{Cell, RefCell};

// NOTE: We use DualVec for state, but standard f64 for time (unless we need time gradients inside forces)
// For now, t is f64.

// Shared with `rhs.rs`. This file supplies the DERIVATIVES of the same
// dynamics, so a constant that drifts between the two produces a correct value
// carrying a derivative that no longer belongs to it -- a failure with no
// wrong number next to it. One definition removes the possibility.
use crate::physical_constants::{
    BOLTZMANN_K, EARTH_DIPOLE_STRENGTH, ELEMENTARY_CHARGE, INV_LIGHT_SPEED_SQ, KM_TO_M,
    LORENTZ_THETA_COS, LORENTZ_THETA_SIN, MEAN_ION_MASS_KG, MIN_COULOMB_LOG, MIN_NUMBER_DENSITY,
    M_TO_KM, VACUUM_PERMITTIVITY,
};

// Helper to convert `f64` to `DualVec` constant.
#[inline]
fn c(val: f64) -> DualVec {
    DualVec::constant(val)
}

#[inline]
fn generic_packed_gravity(
    state: &[DualVec; 6],
    jd: f64,
    cache: &mut GravityCacheGeneric<DualVec>,
    packed: &PackedGravityCoeffs,
) -> Result<[DualVec; 3], GravityError> {
    spherical_gravity_impl_generic_packed(state, jd, cache, packed)
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "scalar and DualVec relativity paths retain this floating-point operation order"
)]
#[inline]
fn compute_relativity_dual(state: &[DualVec; 6]) -> [DualVec; 3] {
    let r_vec = [state[0], state[1], state[2]];
    let v_vec = [state[3], state[4], state[5]];
    let r = (r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2]).sqrt();
    let v = (v_vec[0] * v_vec[0] + v_vec[1] * v_vec[1] + v_vec[2] * v_vec[2]).sqrt();
    if r.v() == 0.0 || v.v() == 0.0 {
        return [c(0.0), c(0.0), c(0.0)];
    }

    let inv_r = c(1.0) / r;
    let inv_v = c(1.0) / v;
    let r_hat = [r_vec[0] * inv_r, r_vec[1] * inv_r, r_vec[2] * inv_r];
    let v_hat = [v_vec[0] * inv_v, v_vec[1] * inv_v, v_vec[2] * inv_v];
    let mur = c(MU) * inv_r;
    let v2 = v * v;
    let rv_dot = r_hat[0] * v_hat[0] + r_hat[1] * v_hat[1] + r_hat[2] * v_hat[2];
    let scale = mur * inv_r * c(INV_LIGHT_SPEED_SQ);

    let mut acc = [c(0.0); 3];
    for ((acceleration, r_hat_axis), v_hat_axis) in acc.iter_mut().zip(r_hat).zip(v_hat) {
        let pt1 = (c(4.0) * mur - v2) * r_hat_axis;
        let pt2 = c(4.0) * v2 * rv_dot * v_hat_axis;
        *acceleration = scale * (pt1 + pt2);
    }
    acc
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "scalar and DualVec Lorentz paths retain this floating-point operation order"
)]
#[inline]
fn compute_lorentz_dual(
    state: &[DualVec; 6],
    jd: f64,
    qm_ratio: f64,
    omega_earth: f64,
) -> [DualVec; 3] {
    if qm_ratio == 0.0 {
        return [c(0.0), c(0.0), c(0.0)];
    }

    let r_vec = [state[0], state[1], state[2]];
    let r_norm = (r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2]).sqrt();
    if r_norm.v() == 0.0 {
        return [c(0.0), c(0.0), c(0.0)];
    }

    let alpha_m = greenwichsrt_impl(jd);
    // Read the SAME precomputed literals the scalar path uses (`rhs.rs`,
    // `dipole_dir`). This previously called `LORENTZ_DIPOLE_THETA_RAD.sin_cos()`
    // at runtime while `rhs.rs` used the literals, so the two paths could take
    // the same physical direction from different bits -- the guard test binds
    // them to 1e-15, which is a tolerance, not equality.
    let dipole_dir = [
        LORENTZ_THETA_SIN * alpha_m.cos(),
        LORENTZ_THETA_SIN * alpha_m.sin(),
        LORENTZ_THETA_COS,
    ];

    let inv_r = c(1.0) / r_norm;
    let r_hat = [r_vec[0] * inv_r, r_vec[1] * inv_r, r_vec[2] * inv_r];
    let dipole_strength_t_km3 = EARTH_DIPOLE_STRENGTH / (KM_TO_M * KM_TO_M * KM_TO_M);
    let dot =
        c(dipole_dir[0]) * r_hat[0] + c(dipole_dir[1]) * r_hat[1] + c(dipole_dir[2]) * r_hat[2];
    let b_scale = c(dipole_strength_t_km3) / (r_norm * r_norm * r_norm);
    let b_vec = [
        b_scale * (c(3.0) * dot * r_hat[0] - c(dipole_dir[0])),
        b_scale * (c(3.0) * dot * r_hat[1] - c(dipole_dir[1])),
        b_scale * (c(3.0) * dot * r_hat[2] - c(dipole_dir[2])),
    ];

    let omega = c(omega_earth);
    let v_rel = [
        state[3] + omega * state[1],
        state[4] - omega * state[0],
        state[5],
    ];
    let v_rel_mps = [
        v_rel[0] * c(KM_TO_M),
        v_rel[1] * c(KM_TO_M),
        v_rel[2] * c(KM_TO_M),
    ];
    let qm = c(qm_ratio);
    let acc_si = [
        qm * (v_rel_mps[1] * b_vec[2] - v_rel_mps[2] * b_vec[1]),
        qm * (v_rel_mps[2] * b_vec[0] - v_rel_mps[0] * b_vec[2]),
        qm * (v_rel_mps[0] * b_vec[1] - v_rel_mps[1] * b_vec[0]),
    ];
    [
        acc_si[0] * c(M_TO_KM),
        acc_si[1] * c(M_TO_KM),
        acc_si[2] * c(M_TO_KM),
    ]
}

#[inline]
fn density_temperature_from_state_dual(
    state: &[DualVec; 6],
    jd: f64,
    earth_radius: f64,
    atm_model: i32,
) -> (DualVec, f64) {
    let rho = density_from_state_dual(state, jd, earth_radius, atm_model);
    if rho.v() <= 0.0 {
        return (rho, 0.0);
    }

    let state_f64 = [
        state[0].v(),
        state[1].v(),
        state[2].v(),
        state[3].v(),
        state[4].v(),
        state[5].v(),
    ];
    let gmst = greenwichsrt_impl(jd);
    let (sin_gmst, cos_gmst) = gmst.sin_cos();
    let (lat, lon, alt) = eci_to_geocentric_spherical(&state_f64, sin_gmst, cos_gmst, earth_radius);
    let (_, temp_k, _) = synthetic_thermosphere_proxy_eval_impl(jd, lat, lon, alt);
    let temperature_k = if temp_k.is_finite() && temp_k > 0.0 {
        temp_k
    } else {
        900.0
    };
    (rho, temperature_k)
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "scalar and DualVec Coulomb-drag paths retain this floating-point operation order"
)]
#[inline]
fn compute_coulomb_drag_dual(
    state: &[DualVec; 6],
    jd: f64,
    qm_ratio: f64,
    r_obj_m: f64,
    omega_earth: f64,
    atm_model: i32,
    earth_radius: f64,
) -> [DualVec; 3] {
    if qm_ratio == 0.0 || r_obj_m <= 0.0 {
        return [c(0.0), c(0.0), c(0.0)];
    }

    let (rho, temperature_k) =
        density_temperature_from_state_dual(state, jd, earth_radius, atm_model);
    if rho.v() <= 0.0 || temperature_k <= 0.0 {
        return [c(0.0), c(0.0), c(0.0)];
    }

    let r_vec = [
        state[0] * c(KM_TO_M),
        state[1] * c(KM_TO_M),
        state[2] * c(KM_TO_M),
    ];
    let v_vec = [
        state[3] * c(KM_TO_M),
        state[4] * c(KM_TO_M),
        state[5] * c(KM_TO_M),
    ];
    let v_vec_norm = (v_vec[0] * v_vec[0] + v_vec[1] * v_vec[1] + v_vec[2] * v_vec[2]).sqrt();
    if v_vec_norm.v() == 0.0 {
        return [c(0.0), c(0.0), c(0.0)];
    }

    let omega = c(omega_earth);
    let flow_vec = [
        v_vec[0] + omega * r_vec[1],
        v_vec[1] - omega * r_vec[0],
        v_vec[2],
    ];
    let flow_norm =
        (flow_vec[0] * flow_vec[0] + flow_vec[1] * flow_vec[1] + flow_vec[2] * flow_vec[2]).sqrt();
    if flow_norm.v() == 0.0 {
        return [c(0.0), c(0.0), c(0.0)];
    }
    let dir_vec = [
        flow_vec[0] / flow_norm,
        flow_vec[1] / flow_norm,
        flow_vec[2] / flow_norm,
    ];

    let mut n_i = rho / c(MEAN_ION_MASS_KG);
    if n_i.v() < MIN_NUMBER_DENSITY {
        n_i = c(MIN_NUMBER_DENSITY);
    }
    let thermal_speed =
        c((2.0 * BOLTZMANN_K * temperature_k / MEAN_ION_MASS_KG.max(1e-30)).max(1e-12)).sqrt();
    // SI-form Debye length: λ_D = sqrt(ε₀ k_B T / (n_i e²)). Audit Phase 3.1
    // fix — see VACUUM_PERMITTIVITY constant doc.
    let debye_length = ((VACUUM_PERMITTIVITY * BOLTZMANN_K * temperature_k)
        / (n_i.v().max(MIN_NUMBER_DENSITY) * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE))
        .max(1e-24)
        .sqrt();
    let log_argument = debye_length / r_obj_m.max(1e-6);
    let coulomb_log = log_argument.max(MIN_COULOMB_LOG).ln();

    let ratio = v_vec_norm / thermal_speed;
    let bracket = ratio.atan() - ratio / (c(1.0) + ratio * ratio);
    let denom = if v_vec_norm.v() < 1.0 {
        c(1.0)
    } else {
        v_vec_norm
    };
    let prefactor = (c(qm_ratio * ELEMENTARY_CHARGE) * n_i.sqrt() / denom)
        * (c(qm_ratio * ELEMENTARY_CHARGE) * n_i.sqrt() / denom);
    let accel_mag = c(8.0 * coulomb_log) * prefactor * bracket;

    [
        accel_mag * dir_vec[0] * c(M_TO_KM),
        accel_mag * dir_vec[1] * c(M_TO_KM),
        accel_mag * dir_vec[2] * c(M_TO_KM),
    ]
}

/// Compute third-body gravity with `DualVec`.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "scalar and DualVec third-body paths retain this floating-point operation order"
)]
#[inline]
fn compute_thirdbody_grav_dual(
    satellite_state: &[DualVec; 6],
    body_pos: &[f64; 3],
    mu_body: f64,
) -> [DualVec; 3] {
    let body_x = c(body_pos[0]);
    let body_y = c(body_pos[1]);
    let body_z = c(body_pos[2]);
    let body_dist_sq = body_x * body_x + body_y * body_y + body_z * body_z;

    if body_dist_sq.v() == 0.0 {
        return [c(0.0), c(0.0), c(0.0)];
    }

    let body_dist = body_dist_sq.sqrt();
    let inv_body_dist = c(1.0) / body_dist;
    let inv_body_dist_sq = inv_body_dist * inv_body_dist;

    let body_norm = [
        body_x * inv_body_dist,
        body_y * inv_body_dist,
        body_z * inv_body_dist,
    ];

    let rel_norm = [
        body_norm[0] - satellite_state[0] * inv_body_dist,
        body_norm[1] - satellite_state[1] * inv_body_dist,
        body_norm[2] - satellite_state[2] * inv_body_dist,
    ];

    let rel_dist_sq =
        rel_norm[0] * rel_norm[0] + rel_norm[1] * rel_norm[1] + rel_norm[2] * rel_norm[2];

    if rel_dist_sq.v() == 0.0 {
        return [c(0.0), c(0.0), c(0.0)];
    }

    let rel_dist = rel_dist_sq.sqrt();
    let inv_rel_dist_cubed = c(1.0) / (rel_dist_sq * rel_dist);

    let coef = c(mu_body) * inv_body_dist_sq;

    [
        coef * (rel_norm[0] * inv_rel_dist_cubed - body_norm[0]),
        coef * (rel_norm[1] * inv_rel_dist_cubed - body_norm[1]),
        coef * (rel_norm[2] * inv_rel_dist_cubed - body_norm[2]),
    ]
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "scalar and DualVec third-body accumulation retains floating-point operation order"
)]
#[inline]
fn add_enabled_thirdbody_accelerations(
    total_acceleration: &mut [DualVec; 3],
    satellite_state: &[DualVec; 6],
    config: &ForceConfig,
) {
    if (config.force_flags & ForceFlags::SUN_GRAVITY) != 0 {
        if let Some(sun_pos) = config.sun_pos {
            if config.mu_sun > 0.0 {
                let sun_acceleration =
                    compute_thirdbody_grav_dual(satellite_state, &sun_pos, config.mu_sun);
                total_acceleration[0] += sun_acceleration[0];
                total_acceleration[1] += sun_acceleration[1];
                total_acceleration[2] += sun_acceleration[2];
            }
        }
    }

    if (config.force_flags & ForceFlags::MOON_GRAVITY) != 0 {
        if let Some(moon_pos) = config.moon_pos {
            if config.mu_moon > 0.0 {
                let moon_acceleration =
                    compute_thirdbody_grav_dual(satellite_state, &moon_pos, config.mu_moon);
                total_acceleration[0] += moon_acceleration[0];
                total_acceleration[1] += moon_acceleration[1];
                total_acceleration[2] += moon_acceleration[2];
            }
        }
    }

    if (config.force_flags & ForceFlags::JUPITER_GRAVITY) != 0 {
        if let Some(jupiter_pos) = config.jupiter_pos {
            if config.mu_jupiter > 0.0 {
                let jupiter_acceleration =
                    compute_thirdbody_grav_dual(satellite_state, &jupiter_pos, config.mu_jupiter);
                total_acceleration[0] += jupiter_acceleration[0];
                total_acceleration[1] += jupiter_acceleration[1];
                total_acceleration[2] += jupiter_acceleration[2];
            }
        }
    }

    if (config.force_flags & ForceFlags::VENUS_GRAVITY) != 0 {
        if let Some(venus_pos) = config.venus_pos {
            if config.mu_venus > 0.0 {
                let venus_acceleration =
                    compute_thirdbody_grav_dual(satellite_state, &venus_pos, config.mu_venus);
                total_acceleration[0] += venus_acceleration[0];
                total_acceleration[1] += venus_acceleration[1];
                total_acceleration[2] += venus_acceleration[2];
            }
        }
    }

    if (config.force_flags & ForceFlags::MARS_GRAVITY) != 0 {
        if let Some(mars_pos) = config.mars_pos {
            if config.mu_mars > 0.0 {
                let mars_acceleration =
                    compute_thirdbody_grav_dual(satellite_state, &mars_pos, config.mu_mars);
                total_acceleration[0] += mars_acceleration[0];
                total_acceleration[1] += mars_acceleration[1];
                total_acceleration[2] += mars_acceleration[2];
            }
        }
    }

    if (config.force_flags & ForceFlags::SATURN_GRAVITY) != 0 {
        if let Some(saturn_pos) = config.saturn_pos {
            if config.mu_saturn > 0.0 {
                let saturn_acceleration =
                    compute_thirdbody_grav_dual(satellite_state, &saturn_pos, config.mu_saturn);
                total_acceleration[0] += saturn_acceleration[0];
                total_acceleration[1] += saturn_acceleration[1];
                total_acceleration[2] += saturn_acceleration[2];
            }
        }
    }
}

/// Get atmospheric density from state (`DualVec`).
#[expect(
    clippy::arithmetic_side_effects,
    reason = "scalar and DualVec density paths retain this floating-point operation order"
)]
#[inline]
fn density_from_state_dual(
    state: &[DualVec; 6],
    _jd: f64,
    earth_radius: f64,
    atm_model: i32,
) -> DualVec {
    if atm_model == 0 {
        return c(0.0);
    }

    if atm_model == 1 {
        let x = state[0];
        let y = state[1];
        let z = state[2];
        let r_km = (x * x + y * y + z * z).sqrt();
        let alt_km = r_km - c(earth_radius);
        let rho0 = c(1.225);
        let h_km = c(8.5);

        // Exponential model: rho = rho0 * exp(-h/H)
        // Soft clamping for negative altitude or high altitude to avoid undefined gradients?
        // nrlmsis handles this, let's just use simple exp here.
        if alt_km.v() >= 1000.0 {
            return c(0.0);
        }
        if alt_km.v() <= 0.0 {
            return rho0;
        }
        return rho0 * (-alt_km / h_km).exp();
    }

    if atm_model == 3 {
        // Construction rejects every force configuration that can reach this
        // branch. Keep an invalid direct call fail-closed rather than silently
        // substituting a different atmosphere model.
        return c(f64::NAN);
    }

    // Unknown model, return zero
    c(0.0)
}

#[derive(Clone)]
struct RHSDualCache {
    cached_tof: f64,
    cached_r_state: [DualVec; 6],
    cache_valid: bool,
    gravity_cache: GravityCacheGeneric<DualVec>,
}

impl Default for RHSDualCache {
    fn default() -> Self {
        Self {
            cached_tof: -1e308,
            cached_r_state: [c(0.0); 6],
            cache_valid: false,
            gravity_cache: GravityCacheGeneric::<DualVec>::default(),
        }
    }
}

pub struct LightyearDualRHS {
    pub init_equinoc_state: [f64; 6], // Baseline is f64
    pub t0_s: f64,
    pub jd0: f64,
    inv_sec_per_day: f64,
    pub config: std::sync::Arc<ForceConfig>,
    pub packed: std::sync::Arc<PackedGravityCoeffs>,
    first_order_packed: Option<std::sync::Arc<PackedGravityCoeffs>>,
    gravity_error: Cell<Option<GravityError>>,
    cache: RefCell<RHSDualCache>,
}

/// Validate a dual configuration for use as the NEWTON MATRIX of a scalar solve.
///
/// Stricter than [`validate_dual_force_config`], and deliberately separate from
/// it. The standalone Jacobian/STM API asks the dual route for the derivative
/// of the DUAL field, which is a coherent request and is covered by its own
/// finite-difference tests. An implicit integrator asks for something else: the
/// derivative of the SCALAR residual it is solving.
///
/// Those two fields diverge from gravity degree two onward. The scalar residual
/// uses continuous-TAI and the full IAU GCRS<->ITRS rotation; the dual route
/// collapses time and applies a legacy GMST82 z-rotation, and the two also
/// differ in Encke baseline history and central-correction algebra. Below
/// degree two the field is spherically symmetric, so no rotation can matter and
/// the routes agree.
///
/// Used as a Newton matrix over a nonspherical field, the mismatch corrupts
/// convergence, step rejection and finite-tolerance roots -- silently, because
/// a Newton iteration converges to the residual's root using whatever matrix it
/// is handed, just more slowly and along a different path.
///
/// # Errors
///
/// Returns an error for any configuration [`validate_dual_force_config`]
/// rejects, and additionally for nonspherical gravity.
pub fn validate_dual_newton_force_config(config: &ForceConfig) -> anyhow::Result<()> {
    validate_dual_force_config(config)?;
    if config.sph_order >= 2 {
        return Err(anyhow::anyhow!(
            "implicit dual/STM integration does not support nonspherical gravity \
             (sph_order {}); the dual route rotates by GMST82 while the scalar \
             residual uses the full IAU frame, so the Newton matrix would \
             differentiate a different field. Use scalar propagation",
            config.sph_order
        ));
    }
    Ok(())
}

pub fn validate_dual_force_config(config: &ForceConfig) -> anyhow::Result<()> {
    crate::rhs::validate_atmosphere_model_code(config.atm_model)?;
    if (config.force_flags & ForceFlags::DRAG) != 0 {
        return Err(anyhow::anyhow!(
            "Dual/STM propagation does not support full-frame atmosphere-relative drag; use scalar propagation"
        ));
    }
    if (config.force_flags & ForceFlags::SRP) != 0 {
        return Err(anyhow::anyhow!(
            "Dual/STM propagation does not support SRP; use scalar propagation"
        ));
    }
    if crate::rhs::atm_model_uses_jb2008_drivers(config.atm_model) {
        return Err(anyhow::anyhow!(
            "Dual/STM propagation does not support JB2008 (atm_model {}); use scalar Vern9 or RKV98 propagation",
            config.atm_model
        ));
    }
    let mut time_varying_body_flags = config.force_flags & ForceFlags::THIRDBODY_ALL;
    if (config.force_flags & ForceFlags::SRP) != 0 {
        time_varying_body_flags |= ForceFlags::SUN_GRAVITY;
    }
    if config.dynamic_ephemeris_flags & time_varying_body_flags != 0 {
        return Err(anyhow::anyhow!("Dual/STM propagation does not support time-varying ephemeris; use an explicit Vern9/RKV98 propagation or implement dynamic DualVec body states"));
    }
    if config.atm_model == 3
        && (config.force_flags & (ForceFlags::DRAG | ForceFlags::COULOMB_DRAG)) != 0
    {
        return Err(anyhow::anyhow!("Dual/STM propagation does not support synthetic thermosphere proxy drag derivatives; NRLMSIS substitution is forbidden"));
    }
    Ok(())
}

impl LightyearDualRHS {
    /// Build a `DualVec` force model for state-transition propagation.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested force configuration has no valid
    /// `DualVec` derivative implementation.
    pub fn new(
        init_equinoc_state: [f64; 6],
        t0_s: f64,
        jd0: f64,
        config: std::sync::Arc<ForceConfig>,
        packed: std::sync::Arc<PackedGravityCoeffs>,
    ) -> anyhow::Result<Self> {
        validate_dual_force_config(&config)?;
        let requested_order = config.sph_order;
        let available_order = packed.max_order();
        if requested_order > available_order {
            return Err(anyhow::anyhow!(
                "requested spherical gravity order {requested_order} exceeds packed authority order {available_order}"
            ));
        }
        let first_order_packed = if config.subtract_first_order
            && requested_order > 0
            && packed.has_nonzero_degree1_terms()
        {
            Some(std::sync::Arc::new(
                packed
                    .truncated_to(1)
                    .context("capping packed degree-one gravity authority")?,
            ))
        } else {
            None
        };
        let packed = if requested_order == available_order {
            packed
        } else {
            std::sync::Arc::new(packed.truncated_to(requested_order).with_context(|| {
                format!("capping packed gravity authority at order {requested_order}")
            })?)
        };
        Ok(Self {
            init_equinoc_state,
            t0_s,
            jd0,
            inv_sec_per_day: 1.0 / SEC_PER_DAY,
            config,
            packed,
            first_order_packed,
            gravity_error: Cell::new(None),
            cache: RefCell::new(RHSDualCache::default()),
        })
    }

    /// Clear the per-instance gravity error latch before a public solve boundary.
    pub(crate) fn reset_gravity_error(&self) {
        self.gravity_error.set(None);
    }

    /// Consume the first typed gravity failure recorded by this RHS instance.
    #[must_use]
    pub(crate) fn take_gravity_error(&self) -> Option<GravityError> {
        self.gravity_error.take()
    }

    #[inline]
    fn latch_gravity_result<T>(&self, result: Result<T, GravityError>) -> Result<T, GravityError> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                if self.gravity_error.get().is_none() {
                    self.gravity_error.set(Some(error));
                }
                Err(error)
            }
        }
    }

    /// Evaluate the Dual force model at one integration stage.
    ///
    /// # Errors
    ///
    /// Returns and latches the exact packed-gravity evaluator failure.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "scalar and DualVec RHS aggregation retains floating-point operation order"
    )]
    #[inline]
    pub fn compute_internal(
        &self,
        delta: &[DualVec; 6],
        t: f64,
    ) -> Result<[DualVec; 6], GravityError> {
        let jd = self.jd0 + t * self.inv_sec_per_day;
        let tof = t - self.t0_s;

        // Note: Safe because we assume single-threaded usage per instance
        let mut cache_borrow = self.cache.borrow_mut();
        let cache = &mut *cache_borrow;

        // Get baseline state (f64) -> convert to DualVec
        // We assume baseline has NO gradients (it's the reference path)
        // Refresh the cache when it is stale; the baseline itself is read from
        // `cache.cached_r_state` below either way. This used to also bind the
        // f64 array, which nothing read -- on a cache hit that was six `.v()`
        // extractions built and thrown away.
        if !(cache.cache_valid && (cache.cached_tof - tof).abs() < 1e-10) {
            let mut s = [0.0; 6];
            equinoc2eci_impl(&self.init_equinoc_state, 6, tof, 0.0, &mut s);
            cache.cached_tof = tof;
            for (cached_state, state_value) in cache.cached_r_state.iter_mut().zip(s) {
                *cached_state = c(state_value);
            }
            cache.cache_valid = true;
        }
        let r_state = [
            cache.cached_r_state[0],
            cache.cached_r_state[1],
            cache.cached_r_state[2],
            cache.cached_r_state[3],
            cache.cached_r_state[4],
            cache.cached_r_state[5],
        ];

        // Perturbed state = baseline + delta
        // Delta has the gradients!
        let st_pert = [
            r_state[0] + delta[0],
            r_state[1] + delta[1],
            r_state[2] + delta[2],
            r_state[3] + delta[3],
            r_state[4] + delta[4],
            r_state[5] + delta[5],
        ];
        let mut total_acc = [c(0.0); 3];

        // 1. Spherical Harmonics
        if self.config.sph_order > 0 {
            let sph_acc = self.latch_gravity_result(generic_packed_gravity(
                &st_pert,
                jd,
                &mut cache.gravity_cache,
                &self.packed,
            ))?;
            total_acc[0] += sph_acc[0];
            total_acc[1] += sph_acc[1];
            total_acc[2] += sph_acc[2];

            if let Some(first_order_packed) = &self.first_order_packed {
                let first_order_spherical_acceleration =
                    self.latch_gravity_result(generic_packed_gravity(
                        &st_pert,
                        jd,
                        &mut cache.gravity_cache,
                        first_order_packed,
                    ))?;
                total_acc[0] -= first_order_spherical_acceleration[0];
                total_acc[1] -= first_order_spherical_acceleration[1];
                total_acc[2] -= first_order_spherical_acceleration[2];
            } else if self.config.subtract_first_order {
                // Empty packed degree-one data still contains the C00 central
                // term. Cancel it analytically while retaining `None` rather
                // than manufacturing a separate packed authority.
                let perturbed_radius_sq =
                    st_pert[0] * st_pert[0] + st_pert[1] * st_pert[1] + st_pert[2] * st_pert[2];
                if perturbed_radius_sq.v() > 0.0 {
                    let perturbed_radius = perturbed_radius_sq.sqrt();
                    let inverse_perturbed_radius_cubed =
                        c(1.0) / (perturbed_radius_sq * perturbed_radius);
                    let mu_val = c(MU);
                    total_acc[0] += st_pert[0] * (mu_val * inverse_perturbed_radius_cubed);
                    total_acc[1] += st_pert[1] * (mu_val * inverse_perturbed_radius_cubed);
                    total_acc[2] += st_pert[2] * (mu_val * inverse_perturbed_radius_cubed);
                }
            }
        }

        // 2. Dust Forces
        if self.config.force_flags != 0 {
            add_enabled_thirdbody_accelerations(&mut total_acc, &st_pert, &self.config);

            if (self.config.force_flags & ForceFlags::RELATIVITY) != 0 {
                let rel_acc = compute_relativity_dual(&st_pert);
                total_acc[0] += rel_acc[0];
                total_acc[1] += rel_acc[1];
                total_acc[2] += rel_acc[2];
            }

            if (self.config.force_flags & ForceFlags::LORENTZ) != 0 && self.config.qm_ratio != 0.0 {
                let lorentz_acc = compute_lorentz_dual(
                    &st_pert,
                    jd,
                    self.config.qm_ratio,
                    self.config.omega_earth,
                );
                total_acc[0] += lorentz_acc[0];
                total_acc[1] += lorentz_acc[1];
                total_acc[2] += lorentz_acc[2];
            }

            if (self.config.force_flags & ForceFlags::COULOMB_DRAG) != 0
                && self.config.qm_ratio != 0.0
                && self.config.r_obj_m > 0.0
            {
                let coulomb_acc = compute_coulomb_drag_dual(
                    &st_pert,
                    jd,
                    self.config.qm_ratio,
                    self.config.r_obj_m,
                    self.config.omega_earth,
                    self.config.atm_model,
                    self.config.earth_radius,
                );
                total_acc[0] += coulomb_acc[0];
                total_acc[1] += coulomb_acc[1];
                total_acc[2] += coulomb_acc[2];
            }
        }

        // Keplerian Correction: +MU*r_base/|r_base|^3 - MU*r_pert/|r_pert|^3
        // Baseline
        let baseline_radius_sq =
            r_state[0] * r_state[0] + r_state[1] * r_state[1] + r_state[2] * r_state[2];
        if baseline_radius_sq.v() > 0.0 {
            let baseline_radius = baseline_radius_sq.sqrt();
            let inverse_baseline_radius_cubed = c(1.0) / (baseline_radius_sq * baseline_radius);
            let mu_val = c(MU);
            total_acc[0] += r_state[0] * (mu_val * inverse_baseline_radius_cubed);
            total_acc[1] += r_state[1] * (mu_val * inverse_baseline_radius_cubed);
            total_acc[2] += r_state[2] * (mu_val * inverse_baseline_radius_cubed);
        }

        // Perturbed
        let perturbed_radius_sq =
            st_pert[0] * st_pert[0] + st_pert[1] * st_pert[1] + st_pert[2] * st_pert[2];
        if perturbed_radius_sq.v() > 0.0 {
            let perturbed_radius = perturbed_radius_sq.sqrt();
            let inverse_perturbed_radius_cubed = c(1.0) / (perturbed_radius_sq * perturbed_radius);
            let mu_val = c(MU);
            total_acc[0] -= st_pert[0] * (mu_val * inverse_perturbed_radius_cubed);
            total_acc[1] -= st_pert[1] * (mu_val * inverse_perturbed_radius_cubed);
            total_acc[2] -= st_pert[2] * (mu_val * inverse_perturbed_radius_cubed);
        }

        Ok([
            delta[3],
            delta[4],
            delta[5],
            total_acc[0],
            total_acc[1],
            total_acc[2],
        ])
    }
}

#[cfg(test)]
mod authority_tests {

    /// The Newton gate must be strictly stronger than the STM gate.
    ///
    /// Both arms are asserted. If the STM arm ever started rejecting
    /// nonspherical gravity too, the standalone Jacobian API -- which has its
    /// own finite-difference coverage against the dual field -- would be lost,
    /// and this test would say so instead of silently agreeing.
    #[test]
    fn nonspherical_gravity_is_refused_for_newton_but_not_for_the_stm_api() {
        let mut config = ForceConfig {
            sph_order: 4,
            force_flags: 0,
            ..ForceConfig::default()
        };
        assert!(
            super::validate_dual_force_config(&config).is_ok(),
            "the standalone STM/Jacobian route must still accept nonspherical gravity"
        );
        let newton = super::validate_dual_newton_force_config(&config)
            .expect_err("implicit integration accepted nonspherical gravity");
        assert!(
            newton.to_string().contains("nonspherical gravity"),
            "refused for the wrong reason: {newton}"
        );

        // Below degree two the field is spherically symmetric, so no frame
        // rotation can matter and both gates must accept.
        config.sph_order = 1;
        assert!(super::validate_dual_force_config(&config).is_ok());
        assert!(
            super::validate_dual_newton_force_config(&config).is_ok(),
            "spherical gravity must remain usable as a Newton matrix"
        );
    }
    use super::*;
    use satpy_core::pack_gravity_coeffs;
    use std::sync::Arc;

    fn construct(config: &ForceConfig) -> anyhow::Result<LightyearDualRHS> {
        let packed = packed_coefficients(0, false)?;
        construct_with(config, packed, 2_460_000.5)
    }

    fn construct_with(
        config: &ForceConfig,
        packed: Arc<PackedGravityCoeffs>,
        jd0: f64,
    ) -> anyhow::Result<LightyearDualRHS> {
        LightyearDualRHS::new(
            [6778.0, 0.0, 0.0, 0.0, 7.67, 0.0],
            0.0,
            jd0,
            Arc::new(*config),
            packed,
        )
    }

    fn packed_coefficients(
        order: usize,
        include_degree_one: bool,
    ) -> anyhow::Result<Arc<PackedGravityCoeffs>> {
        let stride = order
            .checked_add(2)
            .ok_or_else(|| anyhow::anyhow!("dual-RHS test gravity stride overflow"))?;
        let coefficient_count = stride
            .checked_mul(stride)
            .ok_or_else(|| anyhow::anyhow!("dual-RHS test gravity coefficient count overflow"))?;
        let mut c = vec![0.0; coefficient_count];
        let mut s = vec![0.0; coefficient_count];
        *c.get_mut(0)
            .ok_or_else(|| anyhow::anyhow!("dual-RHS test gravity C00 coefficient missing"))? = 1.0;
        if include_degree_one && order >= 1 {
            let degree_one_sine_index = stride.checked_add(1).ok_or_else(|| {
                anyhow::anyhow!("dual-RHS test gravity degree-one index overflow")
            })?;
            *c.get_mut(stride).ok_or_else(|| {
                anyhow::anyhow!("dual-RHS test gravity C10 coefficient missing")
            })? = 3.0e-6;
            *c.get_mut(degree_one_sine_index).ok_or_else(|| {
                anyhow::anyhow!("dual-RHS test gravity C11 coefficient missing")
            })? = -2.0e-6;
            *s.get_mut(degree_one_sine_index).ok_or_else(|| {
                anyhow::anyhow!("dual-RHS test gravity S11 coefficient missing")
            })? = 1.0e-6;
        }
        if order >= 2 {
            let degree_two_index = stride.checked_mul(2).ok_or_else(|| {
                anyhow::anyhow!("dual-RHS test gravity degree-two index overflow")
            })?;
            *c.get_mut(degree_two_index).ok_or_else(|| {
                anyhow::anyhow!("dual-RHS test gravity C20 coefficient missing")
            })? = -1.082_63e-3;
        }
        let packed = pack_gravity_coeffs(&c, &s, stride, order).map_err(|error| {
            anyhow::anyhow!("dual-RHS test gravity coefficients must pack: {error}")
        })?;
        Ok(Arc::new(packed))
    }

    fn acceleration_values(
        rhs: &LightyearDualRHS,
        delta: [f64; 6],
    ) -> Result<[f64; 3], GravityError> {
        Ok(acceleration_duals(rhs, delta.map(c))?.map(|value| value.v()))
    }

    fn acceleration_duals(
        rhs: &LightyearDualRHS,
        delta: [DualVec; 6],
    ) -> Result<[DualVec; 3], GravityError> {
        let derivative = rhs.compute_internal(&delta, 0.0)?;
        Ok([derivative[3], derivative[4], derivative[5]])
    }

    #[test]
    fn dual_rhs_rejects_dynamic_ephemeris() {
        let error = construct(&ForceConfig {
            force_flags: ForceFlags::SUN_GRAVITY,
            dynamic_ephemeris_flags: ForceFlags::SUN_GRAVITY,
            ..ForceConfig::default()
        })
        .err()
        .expect("dynamic ephemeris must fail");
        assert!(error
            .to_string()
            .contains("does not support time-varying ephemeris"));
    }

    #[test]
    fn dual_rhs_rejects_proxy_drag_substitution() {
        let error = construct(&ForceConfig {
            sph_order: 0,
            force_flags: ForceFlags::COULOMB_DRAG,
            atm_model: 3,
            ..ForceConfig::default()
        })
        .err()
        .expect("proxy drag must fail");
        assert!(error
            .to_string()
            .contains("does not support synthetic thermosphere proxy drag derivatives"));
    }

    #[test]
    fn dual_rhs_rejects_drag_without_full_frame_derivatives() {
        let error = construct(&ForceConfig {
            sph_order: 0,
            force_flags: ForceFlags::DRAG,
            atm_model: 1,
            ..ForceConfig::default()
        })
        .err()
        .expect("scalar-z dual drag must fail closed");
        let message = error.to_string();
        assert!(
            message.contains("does not support full-frame atmosphere-relative drag"),
            "unexpected rejection: {message}"
        );
    }

    #[test]
    fn dual_rhs_rejects_jb2008_explicitly() {
        let error = construct(&ForceConfig {
            sph_order: 0,
            force_flags: 0,
            atm_model: 4,
            ..ForceConfig::default()
        })
        .err()
        .expect("JB2008 dual propagation must fail");
        assert!(error.to_string().contains("does not support JB2008"));
    }

    #[test]
    fn dual_rhs_rejects_srp_before_construction() {
        let error = construct(&ForceConfig {
            force_flags: ForceFlags::SRP,
            ..ForceConfig::default()
        })
        .err()
        .expect("SRP dual propagation must fail");
        assert_eq!(
            error.to_string(),
            "Dual/STM propagation does not support SRP; use scalar propagation"
        );
    }

    #[test]
    fn direct_model_three_density_is_nonfinite() {
        let density = density_from_state_dual(
            &[c(6_778.0), c(0.0), c(0.0), c(0.0), c(7.67), c(0.0)],
            2_460_000.5,
            6_378.137,
            3,
        );
        assert!(!density.v().is_finite());
    }

    #[test]
    fn dual_rhs_packed_gravity_error_propagates_exactly() -> anyhow::Result<()> {
        let config = ForceConfig {
            sph_order: 2,
            ..ForceConfig::default()
        };
        let packed = packed_coefficients(2, true)?;
        let rhs = construct_with(&config, packed, f64::NAN)?;
        rhs.reset_gravity_error();
        let error = rhs
            .compute_internal(&[c(0.0); 6], 0.0)
            .err()
            .ok_or_else(|| {
                anyhow::anyhow!("invalid Julian date must return its typed gravity error")
            })?;
        if error != GravityError::InvalidTime {
            return Err(anyhow::anyhow!("expected InvalidTime, got {error}"));
        }
        if rhs.take_gravity_error() != Some(GravityError::InvalidTime) {
            return Err(anyhow::anyhow!(
                "the owned RHS latch preserves the exact gravity error"
            ));
        }
        Ok(())
    }

    #[test]
    fn dual_rhs_gravity_error_latch_does_not_leak_between_evaluations() -> anyhow::Result<()> {
        let rhs = construct_with(
            &ForceConfig {
                sph_order: 2,
                ..ForceConfig::default()
            },
            packed_coefficients(2, true)?,
            2_460_000.5,
        )?;
        rhs.reset_gravity_error();
        let error = rhs
            .compute_internal(&[c(f64::NAN), c(0.0), c(0.0), c(0.0), c(0.0), c(0.0)], 0.0)
            .err()
            .ok_or_else(|| anyhow::anyhow!("invalid state must return its typed gravity error"))?;
        if error != GravityError::InvalidState {
            return Err(anyhow::anyhow!("expected InvalidState, got {error}"));
        }
        if rhs.compute_internal(&[c(0.0); 6], 0.0).is_err() {
            return Err(anyhow::anyhow!("a later valid evaluation must still run"));
        }
        if rhs.take_gravity_error() != Some(GravityError::InvalidState) {
            return Err(anyhow::anyhow!(
                "the first error persists until its public boundary consumes it"
            ));
        }

        rhs.reset_gravity_error();
        if rhs.compute_internal(&[c(0.0); 6], 0.0).is_err() {
            return Err(anyhow::anyhow!("valid evaluation after reset must succeed"));
        }
        if rhs.take_gravity_error().is_some() {
            return Err(anyhow::anyhow!(
                "a consumed/reset error must not leak into the next evaluation"
            ));
        }
        Ok(())
    }

    #[test]
    fn dual_rhs_empty_degree_one_uses_analytic_central_cancellation() -> anyhow::Result<()> {
        let packed = packed_coefficients(4, false)?;
        let full = construct_with(
            &ForceConfig {
                sph_order: 2,
                subtract_first_order: false,
                ..ForceConfig::default()
            },
            Arc::clone(&packed),
            2_460_000.5,
        )?;
        let first_order = construct_with(
            &ForceConfig {
                sph_order: 1,
                subtract_first_order: false,
                ..ForceConfig::default()
            },
            Arc::clone(&packed),
            2_460_000.5,
        )?;
        let central = construct_with(
            &ForceConfig {
                sph_order: 0,
                subtract_first_order: false,
                ..ForceConfig::default()
            },
            Arc::clone(&packed),
            2_460_000.5,
        )?;
        let subtracted = construct_with(
            &ForceConfig {
                sph_order: 2,
                subtract_first_order: true,
                ..ForceConfig::default()
            },
            packed,
            2_460_000.5,
        )?;

        if subtracted.packed.max_order() != 2 {
            return Err(anyhow::anyhow!(
                "empty degree-one subtraction must retain order two"
            ));
        }
        if subtracted.first_order_packed.is_some() {
            return Err(anyhow::anyhow!(
                "empty degree one retains no separate packed subtraction"
            ));
        }
        let delta = [
            DualVec::new(0.1, nalgebra::Vector3::new(1.0, 0.0, 0.0)),
            DualVec::new(-0.2, nalgebra::Vector3::new(0.0, 1.0, 0.0)),
            DualVec::new(0.05, nalgebra::Vector3::new(0.0, 0.0, 1.0)),
            c(1.0e-5),
            c(-2.0e-5),
            c(3.0e-5),
        ];
        let full_acceleration =
            acceleration_duals(&full, delta).context("full packed gravity evaluation failed")?;
        let first_order_acceleration =
            acceleration_duals(&first_order, delta).map_err(|error| {
                anyhow::anyhow!("order-one packed gravity evaluation failed: {error}")
            })?;
        let central_acceleration =
            acceleration_duals(&central, delta).context("central gravity evaluation failed")?;
        let subtracted_acceleration = acceleration_duals(&subtracted, delta).map_err(|error| {
            anyhow::anyhow!("subtracted packed gravity evaluation failed: {error}")
        })?;

        for (axis, (((full, first_order), central), actual)) in full_acceleration
            .into_iter()
            .zip(first_order_acceleration)
            .zip(central_acceleration)
            .zip(subtracted_acceleration)
            .enumerate()
        {
            let expected = full - first_order + central;
            let actual_derivatives = actual.d();
            let expected_derivatives = expected.d();
            for (lane, (actual_lane, expected_lane)) in std::iter::once((actual.v(), expected.v()))
                .chain(actual_derivatives.into_iter().zip(expected_derivatives))
                .enumerate()
            {
                let difference = (actual_lane - expected_lane).abs();
                let scale = actual_lane.abs().max(expected_lane.abs());
                if difference > 1.0e-12 && difference > 1.0e-10 * scale {
                    return Err(anyhow::anyhow!(
                        "axis={axis} lane={lane} empty degree-one cancellation mismatch: actual={actual_lane:.16e}, expected={expected_lane:.16e}, difference={difference:.4e}"
                    ));
                }
            }
        }

        if !subtracted_acceleration
            .into_iter()
            .zip(full_acceleration)
            .any(|(subtracted, unsubtracted)| (subtracted.v() - unsubtracted.v()).abs() > 1.0e-12)
        {
            return Err(anyhow::anyhow!(
                "empty degree-one subtract-first result must differ from unsubtracted gravity"
            ));
        }
        Ok(())
    }

    #[test]
    fn dual_rhs_keeps_source_packed_arc_at_exact_requested_order() -> anyhow::Result<()> {
        let packed = packed_coefficients(2, true)?;
        let rhs = construct_with(
            &ForceConfig {
                sph_order: 2,
                ..ForceConfig::default()
            },
            Arc::clone(&packed),
            2_460_000.5,
        )?;
        if !Arc::ptr_eq(&rhs.packed, &packed) {
            return Err(anyhow::anyhow!(
                "an exact requested order must retain the source packed authority Arc"
            ));
        }
        Ok(())
    }

    #[test]
    fn dual_rhs_subtracts_explicit_packed_degree_one_gravity() -> anyhow::Result<()> {
        let packed = packed_coefficients(2, true)?;
        let full = construct_with(
            &ForceConfig {
                sph_order: 2,
                subtract_first_order: false,
                ..ForceConfig::default()
            },
            Arc::clone(&packed),
            2_460_000.5,
        )?;
        let first_order = construct_with(
            &ForceConfig {
                sph_order: 1,
                subtract_first_order: false,
                ..ForceConfig::default()
            },
            Arc::clone(&packed),
            2_460_000.5,
        )?;
        let central = construct_with(
            &ForceConfig {
                sph_order: 0,
                subtract_first_order: false,
                ..ForceConfig::default()
            },
            Arc::clone(&packed),
            2_460_000.5,
        )?;
        let subtracted = construct_with(
            &ForceConfig {
                sph_order: 2,
                subtract_first_order: true,
                ..ForceConfig::default()
            },
            packed,
            2_460_000.5,
        )?;

        let delta = [0.1, -0.2, 0.05, 1.0e-5, -2.0e-5, 3.0e-5];
        let full_acceleration =
            acceleration_values(&full, delta).context("full packed gravity evaluation failed")?;
        let first_order_acceleration =
            acceleration_values(&first_order, delta).map_err(|error| {
                anyhow::anyhow!("order-one packed gravity evaluation failed: {error}")
            })?;
        let central_acceleration =
            acceleration_values(&central, delta).context("central gravity evaluation failed")?;
        let subtracted_acceleration = acceleration_values(&subtracted, delta).map_err(|error| {
            anyhow::anyhow!("subtracted packed gravity evaluation failed: {error}")
        })?;
        for (axis, (((&full_value, &first_order_value), &central_value), &actual_value)) in
            full_acceleration
                .iter()
                .zip(first_order_acceleration.iter())
                .zip(central_acceleration.iter())
                .zip(subtracted_acceleration.iter())
                .enumerate()
        {
            let expected = full_value - first_order_value + central_value;
            let difference = (actual_value - expected).abs();
            let scale = actual_value.abs().max(expected.abs());
            if difference > 1.0e-12 && difference > 1.0e-10 * scale {
                return Err(anyhow::anyhow!(
                    "axis={axis} explicit degree-one subtraction mismatch: actual={actual_value:.16e}, expected={expected:.16e}, difference={difference:.4e}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn dual_rhs_rejects_order_wider_than_packed_authority() -> anyhow::Result<()> {
        let config = ForceConfig {
            sph_order: 3,
            ..ForceConfig::default()
        };
        let packed = packed_coefficients(2, true)?;
        if construct_with(&config, packed, 2_460_000.5).is_ok() {
            return Err(anyhow::anyhow!(
                "Dual RHS must reject a requested order wider than packed authority"
            ));
        }
        Ok(())
    }
}
