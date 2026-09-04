//! Rust post-processing for batch constellation candidates.
//!
//! Mirrors the Python transfer postprocess pipeline:
//! - optimize dust intercept (bounded LM)
//! - propagate dust distribution to intercept (UKF)
//! - optional conjunction-separation diagnostics
//! - deterministic + probabilistic dust mass (strict fail-fast)

use crate::evaluate::{
    propagate_candidate_state_at_epoch, propagate_high_fidelity_state_at_epoch_checked,
    EvaluationArithmeticOverflow, TransferPropagationFailure,
};
#[cfg(test)]
use crate::py_config::PostprocessConfig;
use crate::py_config::{PhysicsConfig, PhysicsConfigError};
use crate::types::{all_finite, CompactTransferCandidate, PlanContext};
#[cfg(test)]
use crate::types::{BodyForceConfig, BodyRole, SamplingMode, SearchDepthPolicy};

#[cfg(any(test, feature = "bench-internal"))]
use satpy_core::equinoc_prop_from_impl;
#[cfg(test)]
use satpy_core::kep2eci_impl;
use satpy_core::{eci2equinoc_impl, norm3, SEC_PER_DAY};

mod distribution;
mod observer;
#[cfg(feature = "solver-qualification")]
mod qualification_trace;
mod session;
mod ukf;

#[cfg(test)]
use distribution::{
    build_intercept_cfg, compute_corrected_dust_state, compute_corrected_dust_state_summary,
    default_intercept_bound_kms, release_covariance_from_conf, resolve_bounded_dust_timing,
    select_split_axis_strict, CorrectedDustStateRequest, SummaryPlanInputs,
};
pub use distribution::{
    AuthoritativeReleaseDistribution, PostprocessControl, PostprocessControlStatus,
    PostprocessDistributionStatus, PostprocessDustDistribution,
};
#[cfg(any(test, feature = "bench-internal"))]
pub use session::{
    batch_postprocess_compact_candidates, CompactBatchPostprocessError,
    CompactBatchPostprocessInputs, CompactBatchPostprocessOutputs, CompactBatchTargetPhysics,
};
pub use session::{
    canonical_strict_hf_gravity_identity, natural_state_position_residual_km,
    natural_state_velocity_residual_km_s, NaturalConjunctionEnclosure,
    NaturalConjunctionFatalError, NaturalConjunctionInfeasible, NaturalConjunctionInputError,
    NaturalConjunctionOutcome, NaturalConjunctionScanAnchor, NaturalConjunctionWitnessResidual,
    NaturalObjectIdentity, NaturalObjectInput, PostprocessSessionError, StrictHfContextStatus,
    StrictHfForceAuthority, StrictHfGravityIdentity, TransferPostprocessSessionCore,
    VerifiedNaturalConjunction, NATURAL_DENSE_ARC_AUTHORITY_CEILING_KM,
};
#[cfg(test)]
use session::{default_postprocess_config, load_global_coeffs, TransferPostprocessScratch};
#[cfg(feature = "solver-qualification")]
pub use session::{QualificationDistributionRequest, QualificationReleaseControlRequest};
// Native entries for the nd_pipeline MF physics layer (Stage 2 UKF sigma finish).
#[cfg(feature = "solver-qualification")]
pub use qualification_trace::{
    QualificationArmIdentity, QualificationLegFailureCode, QualificationLegInput,
    QualificationLegOutcome, QualificationLegPath, QualificationLegRecord, QualificationLegTrace,
    QualificationTraceError, QualificationTraceIdentity, MAX_QUALIFICATION_LEG_RECORDS,
};
#[cfg(test)]
use ukf::{propagate_component_means_ukf_batch, propagate_component_ukf_checked};
pub use ukf::{propagate_components_ukf_full_batch, UkfPropagationFailure};

const DEFAULT_NUM_DISTS: usize = 3;
const MAX_DUST_COMPONENTS: usize = 7;
const COMPACT_TIMELINE_TOL_S: f64 = 1e-3;

// Dust physical properties (Am, CD, CR) are supplied via PhysicsConfig from
// v3_production.yaml — no hardcoded defaults here.  See physics.dust_Am_ratio_m2kg,
// physics.dust_CD, physics.dust_CR in the YAML.

#[inline]
fn compact_candidate_is_postprocess_coherent(
    candidate: &CompactTransferCandidate,
    conjunction_jd: f64,
) -> bool {
    if !candidate.valid
        || !conjunction_jd.is_finite()
        || !matches!(candidate.target_index, -1..=1)
        || !candidate.total_dv.is_finite()
        || !candidate.phase_dv_norm.is_finite()
        || !candidate.transfer_dv_norm.is_finite()
        || !candidate.transfer_tof_s.is_finite()
        || candidate.transfer_tof_s <= 0.0
        || !candidate.total_time_s.is_finite()
        || candidate.total_time_s < candidate.transfer_tof_s
        || !candidate.relative_velocity_km_s.is_finite()
        || candidate.relative_velocity_km_s.abs() <= 0.0
        || !candidate.time_per_relative_velocity_s_per_km_s.is_finite()
        || !compact_time_per_relative_velocity_is_coherent(candidate)
        || !candidate.solver_intercept_jd.is_finite()
        || !candidate.tof_jd_start.is_finite()
        || !all_finite(&candidate.payload_intercept_state)
        || !all_finite(&candidate.target_intercept_state)
        || !all_finite(&candidate.transfer_burn_pre_state)
        || !all_finite(&candidate.transfer_dv)
    {
        return false;
    }

    if candidate.solver_intercept_jd > conjunction_jd + 1e-9 {
        return false;
    }
    let transfer_to_intercept_s =
        (candidate.solver_intercept_jd - candidate.tof_jd_start) * SEC_PER_DAY;
    if !transfer_to_intercept_s.is_finite() || transfer_to_intercept_s <= 0.0 {
        return false;
    }
    let tof_tol = COMPACT_TIMELINE_TOL_S.max(candidate.transfer_tof_s.abs() * 1e-9);
    (transfer_to_intercept_s - candidate.transfer_tof_s).abs() <= tof_tol
}

#[inline]
fn compact_time_per_relative_velocity_is_coherent(candidate: &CompactTransferCandidate) -> bool {
    let expected = candidate.total_time_s / candidate.relative_velocity_km_s.abs();
    if !expected.is_finite() {
        return false;
    }
    let tol = 1e-9_f64.max(expected.abs() * 1e-9);
    (candidate.time_per_relative_velocity_s_per_km_s - expected).abs() <= tol
}

#[inline]
fn add_velocity(state: &mut [f64; 6], dv: &[f64; 3]) {
    state[3] += dv[0];
    state[4] += dv[1];
    state[5] += dv[2];
}

#[cfg(test)]
fn kep_to_eci(kep: &[f64; 6]) -> Option<[f64; 6]> {
    let mut eci = [0.0; 6];
    kep2eci_impl(kep, false, 0.0, 0.0, false, &mut eci);
    if all_finite(&eci) {
        Some(eci)
    } else {
        None
    }
}

#[cfg(any(test, feature = "bench-internal"))]
fn propagate_equinoctial(eci: &[f64; 6], dt_s: f64) -> Option<[f64; 6]> {
    let mut equ = [0.0; 6];
    eci2equinoc_impl(eci, 6, 0.0, 0.0, &mut equ);
    let mut out = [0.0; 6];
    equinoc_prop_from_impl(&equ, dt_s, &mut out);
    if all_finite(&out) {
        Some(out)
    } else {
        None
    }
}

fn propagate_with_ctx_checked(
    eci: &[f64; 6],
    dt_s: f64,
    ctx: &PlanContext,
) -> Result<[f64; 6], TransferPropagationFailure> {
    let mut equ = [0.0; 6];
    eci2equinoc_impl(eci, 6, 0.0, 0.0, &mut equ);
    let state = if ctx.execution_policy.use_high_fidelity {
        propagate_high_fidelity_state_at_epoch_checked(
            &equ,
            dt_s,
            ctx.epoch_jd,
            ctx.transfer_body_force(),
            ctx,
        )?
    } else {
        propagate_candidate_state_at_epoch(&equ, dt_s, ctx.epoch_jd, ctx.transfer_body_force(), ctx)
            .map_err(|_: EvaluationArithmeticOverflow| {
                TransferPropagationFailure::ArithmeticOverflow
            })?
            .ok_or(TransferPropagationFailure::InvalidInput)?
    };
    all_finite(&state)
        .then_some(state)
        .ok_or(TransferPropagationFailure::NonFiniteOutput)
}

fn clamp_dv_guess(dv: &mut [f64; 3], max_norm: f64) {
    let norm = norm3(dv);
    if norm > max_norm && norm.is_finite() && max_norm > 0.0 {
        let scale = max_norm / norm;
        dv[0] *= scale;
        dv[1] *= scale;
        dv[2] *= scale;
    }
}

fn build_force_config(
    conf: &PhysicsConfig,
    am_ratio: f64,
    cd: f64,
    cr: f64,
) -> Result<lightyear_odeint_rs::types::ForceConfig, PhysicsConfigError> {
    let integrator_method = conf.integrator_method()?;
    Ok(lightyear_odeint_rs::types::ForceConfig {
        sph_order: conf.sph_order,
        force_flags: conf.force_flags,
        subtract_first_order: conf.sph_order > 0,
        atm_model: conf.atm_model,
        am_ratio,
        cd,
        cr,
        sun_pos: conf.sun_pos,
        moon_pos: conf.moon_pos,
        jupiter_pos: conf.jupiter_pos,
        venus_pos: conf.venus_pos,
        mars_pos: conf.mars_pos,
        saturn_pos: conf.saturn_pos,
        dt_max: conf.dt_max,
        eps: conf.tolerance,
        integrator_method,
        ..lightyear_odeint_rs::types::ForceConfig::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::{
        constellation_solve_native_with_front_output_mode, ConstellationSolveConfiguration,
        EventPlanRequest, FrontOutputMode,
    };

    fn diagonal_covariance(diagonal: f64) -> [[f64; 6]; 6] {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| if row == column { diagonal } else { 0.0 })
        })
    }

    struct CompactCorrectionFixture {
        compact: CompactTransferCandidate,
        primary_at_intercept: [f64; 6],
        secondary_at_intercept: [f64; 6],
        stale_target: [f64; 6],
        intercept_jd: f64,
        conjunction_jd: f64,
        stale_intercept_jd: f64,
    }

    impl CompactCorrectionFixture {
        fn assert_corrections(self, compact_core: &TransferPostprocessSessionCore) {
            let Self {
                mut compact,
                primary_at_intercept,
                secondary_at_intercept,
                stale_target,
                intercept_jd,
                conjunction_jd,
                stale_intercept_jd,
            } = self;
            let compact_corrected = compact_core
                .correct_one(
                    Some(&compact),
                    &primary_at_intercept,
                    &secondary_at_intercept,
                    intercept_jd,
                    conjunction_jd,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("typed compact correction")
                .expect("compact candidate should use verified solver intercept payload");
            assert!(compact_corrected.1.is_finite());

            assert!(compact_core
                .correct_one(
                    Some(&compact),
                    &primary_at_intercept,
                    &secondary_at_intercept,
                    stale_intercept_jd,
                    conjunction_jd,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("typed compact correction")
                .is_none());

            let target_frame_shift_corrected = compact_core
                .correct_one(
                    Some(&compact),
                    &stale_target,
                    &stale_target,
                    intercept_jd,
                    conjunction_jd,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("typed compact correction")
                .expect("actual target arrays may differ from solver target state");
            assert!(target_frame_shift_corrected.1.is_finite());

            compact.tof_jd_start = compact.solver_intercept_jd + 1.0 / SEC_PER_DAY;
            assert!(compact_core
                .correct_one(
                    Some(&compact),
                    &primary_at_intercept,
                    &secondary_at_intercept,
                    intercept_jd,
                    conjunction_jd,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("typed compact correction")
                .is_none());
        }
    }

    #[test]
    fn strict_hf_context_propagation_retains_authority_failure() {
        let mut request = crate::types::TransferRequest::with_j2_closure_settings(
            crate::solve::J2ClosureSettings::default(),
        );
        request.epoch_jd = 2_460_000.5;
        request.execution_policy = crate::types::ExecutionPolicy {
            use_high_fidelity: true,
            require_high_fidelity: true,
            ..crate::types::ExecutionPolicy::default()
        };
        let ctx = PlanContext::from_request(request);

        assert!(matches!(
            propagate_with_ctx_checked(&[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], 60.0, &ctx),
            Err(crate::evaluate::TransferPropagationFailure::Authority)
        ));
    }

    #[test]
    fn test_default_intercept_bound_kms() {
        assert_eq!(
            default_intercept_bound_kms(3600.0, false).to_bits(),
            0.1_f64.to_bits()
        );
        assert_eq!(
            default_intercept_bound_kms(3.0 * 3600.0, true).to_bits(),
            1.0_f64.to_bits()
        );
        assert!((default_intercept_bound_kms(18.0 * 3600.0, true) - 1.9996).abs() < 1e-6);
        assert_eq!(
            default_intercept_bound_kms(30.0 * 3600.0, true).to_bits(),
            2.0_f64.to_bits()
        );
    }

    #[test]
    fn test_build_intercept_cfg_overrides_fields() {
        let post = PostprocessConfig {
            fix_ls_max_nfev: 77,
            fix_ls_tol: 2e-6,
            fix_ls_skip_tol: 0.75,
            dust_intercept_tol_km: 0.02,
            dust_radial_samples: 24,
            dust_angular_samples: 100,
            gmm_components: 3,
            max_physical_dv_kms: 7.5,
            min_practical_dust_mass_kg: 0.01,
            mf_seed_bound_kms: 0.1,
            hf_refine_bound_kms: 1.0,
            mf_seed_reg_weight: 1e-3,
            hf_refine_reg_weight: 1e-3,
            mf_seed_max_bound_expansions: 7,
            hf_refine_max_bound_expansions: 7,
            hybrid_mf_seed_hf_refine: false,
            dust_phase_tof_s: 7200.0,
            canister_tof_fraction: 0.0,
            canister_am: 0.01,
            canister_cd: 2.2,
            canister_cr: 1.3,
        };
        let cfg = build_intercept_cfg(&post, 0.25, 0.42, 0.007, 9);
        assert_eq!(cfg.max_iters, 77);
        assert_eq!(cfg.tol.to_bits(), 2e-6_f64.to_bits());
        assert_eq!(cfg.skip_tol.to_bits(), 0.75_f64.to_bits());
        assert_eq!(cfg.min_miss_km.to_bits(), 0.25_f64.to_bits());
        assert_eq!(cfg.bound.to_bits(), 0.42_f64.to_bits());
        assert_eq!(cfg.reg_weight.to_bits(), 0.007_f64.to_bits());
        assert_eq!(cfg.max_bound_expansions, 9);
    }

    #[test]
    fn compact_candidate_coherence_rejects_stale_time_velocity_ratio() {
        let conjunction_jd = 2_460_000.0;
        let transfer_tof_s = 100.0;
        let mut candidate = CompactTransferCandidate {
            valid: true,
            target_index: 0,
            total_dv: 0.2,
            phase_dv_norm: 0.1,
            transfer_dv_norm: 0.1,
            transfer_tof_s,
            total_time_s: 400.0,
            relative_velocity_km_s: 0.02,
            time_per_relative_velocity_s_per_km_s: 999.0,
            solver_intercept_jd: conjunction_jd,
            tof_jd_start: conjunction_jd - transfer_tof_s / SEC_PER_DAY,
            branch_status: crate::types::BranchStatusToken::Accepted,
            ..CompactTransferCandidate::default()
        };

        assert!(!compact_candidate_is_postprocess_coherent(
            &candidate,
            conjunction_jd
        ));

        candidate.time_per_relative_velocity_s_per_km_s = 20_000.0;
        assert!(compact_candidate_is_postprocess_coherent(
            &candidate,
            conjunction_jd
        ));

        candidate.relative_velocity_km_s = -0.02;
        assert!(compact_candidate_is_postprocess_coherent(
            &candidate,
            conjunction_jd
        ));
    }

    #[test]
    fn test_release_covariance_uses_anisotropic_rtn_sigmas() {
        let conf = PhysicsConfig {
            dust_pos_sigma_m: 100.0,
            dust_pos_sigma_radial_cross_track_m: 50.0,
            dust_vel_sigma_mps: 0.08,
            dust_vel_sigma_radial_cross_track_mps: 0.03,
            ..PhysicsConfig::default()
        };
        let state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];

        let cov = release_covariance_from_conf(&conf, &state).expect("valid RTN frame");
        let expected_diagonal = [
            0.05_f64.powi(2),
            0.10_f64.powi(2),
            0.05_f64.powi(2),
            0.00003_f64.powi(2),
            0.00008_f64.powi(2),
            0.00003_f64.powi(2),
        ];

        assert!((cov[0][0] - expected_diagonal[0]).abs() < 1e-15);
        assert!((cov[1][1] - expected_diagonal[1]).abs() < 1e-15);
        assert!((cov[2][2] - expected_diagonal[2]).abs() < 1e-15);
        assert!((cov[3][3] - expected_diagonal[3]).abs() < 1e-20);
        assert!((cov[4][4] - expected_diagonal[4]).abs() < 1e-20);
        assert!((cov[5][5] - expected_diagonal[5]).abs() < 1e-20);
    }

    #[test]
    fn test_bounded_dust_timing_clamps_to_configured_window() {
        let timing = resolve_bounded_dust_timing(14_400.0, 7_200.0, 0.25).expect("bounded timing");
        assert!((timing.transfer_to_intercept - 14_400.0).abs() <= 1e-12);
        assert!((timing.dust_window - 7_200.0).abs() <= 1e-12);
        assert!((timing.pre_window_hold - 7_200.0).abs() <= 1e-12);
        assert!((timing.canister_coast - 9_000.0).abs() <= 1e-12);
        assert!((timing.dust_flight - 5_400.0).abs() <= 1e-12);
    }

    #[test]
    fn test_bounded_dust_timing_clamps_to_shorter_available_transfer() {
        let timing = resolve_bounded_dust_timing(3_600.0, 7_200.0, 0.25).expect("bounded timing");
        assert!((timing.transfer_to_intercept - 3_600.0).abs() <= 1e-12);
        assert!((timing.dust_window - 3_600.0).abs() <= 1e-12);
        assert!((timing.pre_window_hold - 0.0).abs() <= 1e-12);
        assert!((timing.canister_coast - 900.0).abs() <= 1e-12);
        assert!((timing.dust_flight - 2_700.0).abs() <= 1e-12);
    }

    #[test]
    fn test_component_ukf_failure_does_not_return_unpropagated_state() {
        let mean = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let cov = diagonal_covariance(1e-12);

        assert_eq!(
            propagate_component_ukf_checked(&mean, &cov, f64::NAN, None,),
            Err(UkfPropagationFailure::InvalidInput)
        );
    }

    #[test]
    fn test_batch_component_ukf_failure_fails_fast() {
        let mean = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let cov = diagonal_covariance(1e-12);
        let mut scratch = TransferPostprocessScratch::default();

        assert_eq!(
            propagate_component_means_ukf_batch(&[mean], &[cov], f64::NAN, None, &mut scratch,),
            Err(UkfPropagationFailure::InvalidInput)
        );
        assert!(scratch
            .comp_means
            .iter()
            .all(|state| state.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn test_split_axis_propagation_failure_fails_fast() {
        assert!(select_split_axis_strict("usfos", f64::NAN).is_err());
    }

    #[test]
    fn test_split_axis_rejects_legacy_aliases() {
        assert!(select_split_axis_strict("nonlinear", 10.0).is_err());
        assert!(select_split_axis_strict("stm", 10.0).is_err());
        assert!(matches!(select_split_axis_strict("maxvar", 10.0), Ok(None)));
    }

    #[test]
    fn test_summary_postprocess_matches_full_distribution_outputs() {
        let satellites_kep = [[7000.0, 0.001, 0.2, 0.0, 0.0, 0.0]];
        let target1_kep = [7100.0, 0.002, 0.21, 0.1, 0.0, 0.2];
        let target2_kep = [7120.0, 0.002, 0.21, 0.1, 0.0, 0.25];

        let satellites = [kep_to_eci(&satellites_kep[0]).expect("satellite eci")];
        let target1 = kep_to_eci(&target1_kep).expect("target1 eci");
        let target2 = kep_to_eci(&target2_kep).expect("target2 eci");

        let front = constellation_solve_native_with_front_output_mode(EventPlanRequest {
            satellites: &satellites,
            satellites_equ_cached: None,
            target1: &target1,
            target2: &target2,
            target_body_forces: [BodyForceConfig::j2(BodyRole::DiagnosticTarget); 2],
            configuration: ConstellationSolveConfiguration {
                max_time_s: 86_400.0,
                max_phase_dv: 0.5,
                max_transfer_dv: 2.0,
                max_revs: 0,
                min_perigee: 6_578.14,
                max_apogee: 41_378.14,
                pairs_to_verify: 1,
                sampling_mode: SamplingMode::Fast,
                search_depth: SearchDepthPolicy::default(),
                epoch_jd: 2_460_000.5,
                distance_tol: 0.025,
                deployer_min_distance: 0.12,
                tof_penalty_weight: 0.1,
                revolution_cap: 1.5,
                target_propagation_authority: crate::types::TargetPropagationAuthority::MfJ2,
                force_config: None,
                require_high_fidelity: false,
                j2_closure_settings: crate::solve::J2ClosureSettings::default(),
                packed_coeffs: None,
                local_optimizer: crate::types::TransferLocalOptimizerConfig::default(),
                warm_start: None,
            },
            scratch: None,
            front_output_mode: FrontOutputMode::TransferPareto,
        })
        .expect("fixture uses valid target propagation authority");
        let candidate = front
            .candidates
            .first()
            .expect("expected valid postprocess training candidate");
        assert!(candidate.valid);
        assert!(candidate.optimum.valid);

        let physics = PhysicsConfig {
            max_phase_dv: 0.5,
            max_transfer_dv: 2.0,
            max_time_s: 86400.0,
            min_miss_distance_km: 1.0,
            event_rewind_days: 3.0,
            dust_pos_sigma_m: 50.0,
            dust_vel_sigma_mps: 0.05,
            hit_probability: 0.9,
            kappa: 2.0,
            use_high_fidelity: false,
            splitting_criterion: "maxvar".to_string(),
            tof_penalty_weight: 0.1,
            revolution_cap: 1.5,
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            ..PhysicsConfig::default()
        };
        let post = PostprocessConfig {
            fix_ls_max_nfev: 40,
            fix_ls_tol: 1e-6,
            fix_ls_skip_tol: 0.0,
            dust_intercept_tol_km: 0.01,
            dust_radial_samples: 24,
            dust_angular_samples: 100,
            gmm_components: 3,
            max_physical_dv_kms: 7.5,
            min_practical_dust_mass_kg: 0.01,
            ..default_postprocess_config()
        };
        let coeffs = load_global_coeffs();
        let intercept_state = candidate.optimum.target_intercept_state;
        let intercept_jd = candidate.optimum.intercept_jd;
        let conjunction_jd = intercept_jd + 600.0 / SEC_PER_DAY;

        let summary_inputs = SummaryPlanInputs {
            valid: true,
            release_state: candidate.optimum.release_state,
            transfer_dv: candidate.optimum.transfer_dv,
            tof_jd_start: candidate.optimum.tof_jd_start,
            min_radius_km: satpy_core::RE,
        };
        let summary = compute_corrected_dust_state_summary(
            &summary_inputs,
            &intercept_state,
            intercept_jd,
            conjunction_jd,
            &physics,
            &post,
            &coeffs,
            None,
            None,
            None,
            None,
        )
        .expect("summary postprocess computation")
        .expect("summary postprocess");
        let dist = compute_corrected_dust_state(CorrectedDustStateRequest {
            plan: &summary_inputs,
            target_intercept_state: &intercept_state,
            intercept_jd,
            conjunction_jd,
            conf: &physics,
            post: &post,
            coeffs: &coeffs,
            split_alpha: None,
            split_axis: None,
            release_covariance: None,
            release_distribution: None,
        })
        .expect("full postprocess");

        for (i, (summary_value, distribution_value)) in summary
            .dust_mean
            .iter()
            .zip(dist.dust_mean.iter())
            .enumerate()
        {
            assert!(
                (*summary_value - *distribution_value).abs() <= 1e-9,
                "dust_mean mismatch at {i}: {summary_value} vs {distribution_value}"
            );
        }
        assert!((summary.correction_dv_norm - dist.correction_dv_norm).abs() <= 1e-12);

        let compact = CompactTransferCandidate::from_constellation_candidate(candidate)
            .expect("compact candidate");
        let stale_target = [f64::NAN; 6];
        let mut primary_at_intercept = stale_target;
        let mut secondary_at_intercept = stale_target;
        if compact.target_index == 0 {
            primary_at_intercept = intercept_state;
        } else {
            secondary_at_intercept = intercept_state;
        }
        let compact_core = TransferPostprocessSessionCore::try_new(Some(physics), Some(post))
            .expect("valid postprocess configuration");
        CompactCorrectionFixture {
            compact,
            primary_at_intercept,
            secondary_at_intercept,
            stale_target,
            intercept_jd,
            conjunction_jd,
            stale_intercept_jd: candidate.optimum.tof_jd_start - 10.0 / SEC_PER_DAY,
        }
        .assert_corrections(&compact_core);
    }
}
