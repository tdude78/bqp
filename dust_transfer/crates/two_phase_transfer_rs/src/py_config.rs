// V3 seeding exports removed during demolition

/// Session physics surface excludes target covariance. Native Hybrid owns Pc
/// covariance through compiled science controls.
///
/// ```compile_fail
/// use two_phase_transfer_rs::PhysicsConfig;
/// let config = PhysicsConfig::default();
/// let _ = config.target_pos_sigma_m;
/// let _ = config.target_vel_sigma_mps;
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhysicsConfig {
    pub max_phase_dv: f64,
    pub max_transfer_dv: f64,
    pub max_time_s: f64,
    pub min_miss_distance_km: f64,
    pub event_rewind_days: f64,
    pub dust_pos_sigma_m: f64,
    pub dust_pos_sigma_radial_cross_track_m: f64,
    /// Dust velocity 1-sigma in m/s (legacy constructor arg name keeps `_m` for compatibility).
    pub dust_vel_sigma_mps: f64,
    pub dust_vel_sigma_radial_cross_track_mps: f64,
    pub hit_probability: f64,
    pub kappa: f64,
    pub use_high_fidelity: bool,
    pub require_hf_transfer_correction: bool,
    pub splitting_criterion: String,
    pub split_alpha_policy: String,
    pub split_rank: usize,
    pub tof_penalty_weight: f64,
    pub revolution_cap: f64,

    // Rendezvous Tolerances
    pub distance_tol: f64,
    pub deployer_min_distance: f64,

    // High-Fidelity Perturbation Flags
    pub sph_order: usize,
    pub force_flags: i32,
    pub atm_model: i32,
    pub am_ratio: f64,
    pub cd: f64,
    pub cr: f64,
    /// Fixed transfer/canister body tuple for every E0→P→L→R→I arc.
    pub transfer_am_ratio: f64,
    pub transfer_cd: f64,
    pub transfer_cr: f64,
    pub dt_max: f64,
    pub tolerance: f64,
    /// Explicit Lightyear integrator authority. Timeline-v2 requires `dopri5`.
    pub method: String,
    pub sun_pos: Option<[f64; 3]>,
    pub moon_pos: Option<[f64; 3]>,
    pub jupiter_pos: Option<[f64; 3]>,
    pub venus_pos: Option<[f64; 3]>,
    pub mars_pos: Option<[f64; 3]>,
    pub saturn_pos: Option<[f64; 3]>,
}

/// A physics configuration cannot select a supported Lightyear integrator.
///
/// This error intentionally carries no copied method string: callers retain
/// their original [`PhysicsConfig`] and its authority bytes without an
/// allocation on the rejection path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicsConfigError {
    /// The configured integrator token is not one of the supported methods.
    UnsupportedIntegratorMethod,
}

impl std::fmt::Display for PhysicsConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedIntegratorMethod => {
                formatter.write_str("physics configuration integrator method is unsupported")
            }
        }
    }
}

impl std::error::Error for PhysicsConfigError {}

/// Sole token-to-stepper mapping authority for the config boundary.
///
/// Every consumer of a stepper token string (config resolution, authority
/// comparisons) must parse through this function exactly once and carry the
/// typed [`StepperMethod`](lightyear_odeint_rs::types::StepperMethod) from
/// there; the original token string stays on [`PhysicsConfig::method`] for
/// Python-boundary round-tripping.
///
/// # Errors
///
/// Returns [`PhysicsConfigError::UnsupportedIntegratorMethod`] when the token
/// has no sealed runtime mapping.
pub fn parse_integrator_token(
    token: &str,
) -> Result<lightyear_odeint_rs::types::StepperMethod, PhysicsConfigError> {
    use lightyear_odeint_rs::types::StepperMethod;
    let method = match token {
        "" | "dopri5" | "dopri5compat" => StepperMethod::Dopri5Compat,
        "tsit5" => StepperMethod::Tsit5,
        "dop853" => StepperMethod::Dop853,
        "rkv98" => StepperMethod::Rkv98,
        "vern7" => StepperMethod::Vern7,
        "vern9" => StepperMethod::Vern9,
        "esdirk43" | "esdirk" => StepperMethod::Esdirk43,
        _ => return Err(PhysicsConfigError::UnsupportedIntegratorMethod),
    };
    Ok(method)
}

impl PhysicsConfig {
    #[inline]
    /// Resolve the configured Lightyear integrator authority.
    ///
    /// Delegates to [`parse_integrator_token`], the single mapping authority.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsConfigError::UnsupportedIntegratorMethod`] when the
    /// configuration token has no sealed runtime mapping.
    pub fn integrator_method(
        &self,
    ) -> Result<lightyear_odeint_rs::types::StepperMethod, PhysicsConfigError> {
        parse_integrator_token(&self.method)
    }

    /// Dust owns the historical physics fields.  Callers must select a body
    /// explicitly instead of reusing this configuration for transfer arcs.
    #[inline]
    #[must_use]
    pub const fn dust_body_force(&self) -> crate::types::BodyForceConfig {
        crate::types::BodyForceConfig::high_fidelity(
            crate::types::BodyRole::Dust,
            self.am_ratio,
            self.cd,
            self.cr,
        )
    }

    /// Transfer/deployer body uses sealed fixed canister ballistics.
    #[inline]
    #[must_use]
    pub const fn transfer_body_force(&self) -> crate::types::BodyForceConfig {
        crate::types::BodyForceConfig::high_fidelity(
            crate::types::BodyRole::TransferVehicle,
            self.transfer_am_ratio,
            self.transfer_cd,
            self.transfer_cr,
        )
    }

    /// Canister coefficients live in postprocess configuration and are kept
    /// distinct from dust values at the native boundary.
    #[inline]
    #[must_use]
    pub const fn canister_body_force(
        &self,
        canister_area_to_mass: f64,
        canister_drag_coefficient: f64,
        canister_radiation_coefficient: f64,
    ) -> crate::types::BodyForceConfig {
        crate::types::BodyForceConfig::high_fidelity(
            crate::types::BodyRole::Canister,
            canister_area_to_mass,
            canister_drag_coefficient,
            canister_radiation_coefficient,
        )
    }
}

/// Postprocess surface excludes retired conjunction-alignment tuning. Alignment
/// is fixed by native runtime logic, not these diagnostic-only values.
///
/// ```compile_fail
/// use two_phase_transfer_rs::PostprocessConfig;
/// fn retired(config: &PostprocessConfig) {
///     let _ = config.conj_align_max_iter;
///     let _ = config.conj_align_tol_km;
///     let _ = config.conj_align_step_kms;
///     let _ = config.conj_align_max_dv_kms;
/// }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct PostprocessConfig {
    pub fix_ls_max_nfev: usize,
    pub fix_ls_tol: f64,
    pub fix_ls_skip_tol: f64,
    /// Required intercept miss distance (km) for the dust mean vs target.
    /// This is distinct from `min_miss_distance_km` (conjunction safety threshold).
    pub dust_intercept_tol_km: f64,
    pub dust_radial_samples: usize,
    pub dust_angular_samples: usize,
    /// Number of Gaussian-mixture components in strict-HF dust lowering.
    pub gmm_components: usize,
    pub max_physical_dv_kms: f64,
    pub min_practical_dust_mass_kg: f64,
    pub mf_seed_bound_kms: f64,
    pub hf_refine_bound_kms: f64,
    pub mf_seed_reg_weight: f64,
    pub hf_refine_reg_weight: f64,
    pub mf_seed_max_bound_expansions: usize,
    pub hf_refine_max_bound_expansions: usize,
    pub hybrid_mf_seed_hf_refine: bool,
    pub dust_phase_tof_s: f64,
    pub canister_tof_fraction: f64,
    pub canister_am: f64,
    pub canister_cd: f64,
    pub canister_cr: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BodyRole;

    #[test]
    fn config_keeps_dust_canister_and_transfer_body_forces_distinct() {
        let physics = PhysicsConfig {
            am_ratio: 1.948,
            cd: 2.2,
            cr: 1.3,
            transfer_am_ratio: 0.01,
            transfer_cd: 2.6,
            transfer_cr: 1.1,
            ..Default::default()
        };

        let dust = physics.dust_body_force();
        let canister = physics.canister_body_force(0.01, 1.7, 1.1);
        let transfer = physics.transfer_body_force();

        assert_eq!(dust.role, BodyRole::Dust);
        assert_eq!(canister.role, BodyRole::Canister);
        assert_eq!(transfer.role, BodyRole::TransferVehicle);
        assert_ne!(dust.am_ratio.to_bits(), canister.am_ratio.to_bits());
        assert_eq!(transfer.am_ratio.to_bits(), 0.01_f64.to_bits());
        assert_eq!(transfer.cd.to_bits(), 2.6_f64.to_bits());
        assert_eq!(transfer.cr.to_bits(), 1.1_f64.to_bits());
        assert_eq!(
            transfer.fidelity,
            crate::types::PropagationFidelity::HighFidelity
        );
    }

    #[test]
    fn dopri5_maps_to_compat_stepper() {
        let config = PhysicsConfig {
            method: "dopri5".to_string(),
            ..Default::default()
        };
        assert_eq!(
            config.integrator_method(),
            Ok(lightyear_odeint_rs::types::StepperMethod::Dopri5Compat)
        );
    }

    #[test]
    fn every_sealed_token_parses_and_round_trips_through_the_config() {
        use lightyear_odeint_rs::types::StepperMethod;
        let sealed: [(&str, StepperMethod); 10] = [
            ("", StepperMethod::Dopri5Compat),
            ("dopri5", StepperMethod::Dopri5Compat),
            ("dopri5compat", StepperMethod::Dopri5Compat),
            ("tsit5", StepperMethod::Tsit5),
            ("dop853", StepperMethod::Dop853),
            ("rkv98", StepperMethod::Rkv98),
            ("vern7", StepperMethod::Vern7),
            ("vern9", StepperMethod::Vern9),
            ("esdirk43", StepperMethod::Esdirk43),
            ("esdirk", StepperMethod::Esdirk43),
        ];
        for (token, expected) in sealed {
            assert_eq!(
                parse_integrator_token(token),
                Ok(expected),
                "token {token:?}"
            );
            let config = PhysicsConfig {
                method: token.to_owned(),
                ..Default::default()
            };
            // The config keeps the original token for Python round-tripping
            // while resolving through the same single mapping authority.
            assert_eq!(config.method, token);
            assert_eq!(config.integrator_method(), Ok(expected), "token {token:?}");
        }
    }

    #[test]
    fn unsupported_integrator_is_rejected_without_mutating_config() {
        let config = PhysicsConfig {
            method: "not-a-stepper".to_owned(),
            ..Default::default()
        };
        let before = config.clone();

        assert_eq!(
            config.integrator_method(),
            Err(PhysicsConfigError::UnsupportedIntegratorMethod)
        );
        assert_eq!(config, before);
    }
}
