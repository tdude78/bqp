#[cfg(any(test, feature = "bench-internal"))]
use crate::types::{
    InvalidTargetPropagationAuthorityCode, PairPlanContextInputs, PlanContext, PlanContextTemplate,
};

/// Target inclination/RAAN from equinoctial elements, with a Keplerian fallback.
///
/// **Test-only since the `objective_hint` deletion**, for the same reason as
/// `lambert_backend::retrograde_departure_dv_lower_bound`: its sole production
/// consumer was the Hohmann lower-bound prune's plane-alignment gate, which was
/// removed with the rest of that apparatus in `cfd14ed6`, and this `cfg` is what
/// keeps "no production caller" true rather than a comment asking people to
/// remember. Kept rather than deleted because the two tests it carries —
/// `test_target_plane_from_equinoctial_matches_keplerian_plane` and
/// `..._canonicalizes_near_equatorial_plane` — cover a real geometric
/// degeneracy (RAAN is undefined at zero inclination and an implementation
/// quietly picks a branch), not the deleted machinery.
#[cfg(test)]
#[inline]
pub(super) fn target_plane_from_equinoctial(
    tgt_equ: &[f64; 6],
    tgt_eci: &[f64; 6],
) -> (f64, f64, bool) {
    let p = tgt_equ[3];
    let q = tgt_equ[4];
    if p.is_finite() && q.is_finite() {
        let tan_half_i = p.hypot(q);
        let inc = if tan_half_i <= satpy_core::TAN_HALF_INCLINATION_FLOOR {
            0.0
        } else {
            2.0 * tan_half_i.atan()
        };
        let raan = if tan_half_i <= satpy_core::TAN_HALF_INCLINATION_FLOOR {
            0.0
        } else {
            p.atan2(q).rem_euclid(std::f64::consts::TAU)
        };
        if inc.is_finite() && raan.is_finite() {
            return (inc, raan, true);
        }
    }

    let mut kep = [0.0_f64; 6];
    satpy_core::eci2kep_impl(tgt_eci, false, true, &mut kep);
    if kep[2].is_finite() && kep[3].is_finite() {
        (kep[2], kep[3], true)
    } else {
        (0.0, 0.0, false)
    }
}

#[cfg(any(test, feature = "bench-internal"))]
pub(super) fn build_cached_plan_context(
    template: &PlanContextTemplate,
    inputs: &PairPlanContextInputs,
) -> Result<PlanContext, InvalidTargetPropagationAuthorityCode> {
    let mut context = PlanContext::with_j2_closure_settings(template.j2_closure_settings);
    context.apply_template_pair(template, inputs)?;
    Ok(context)
}

#[cfg(test)]
mod j2_closure_authority_tests {
    use super::*;
    use crate::solve::J2ClosureSettings;
    use crate::types::{
        ExecutionPolicy, SamplingMode, SearchDepthPolicy, TargetPropagationAuthority,
        TransferLocalOptimizerConfig,
    };
    use rayon::prelude::*;

    fn template_with_explicit_j2_policy(
        j2_closure_settings: J2ClosureSettings,
    ) -> PlanContextTemplate {
        PlanContextTemplate {
            max_time_s: 86_400.0,
            tof_penalty_weight: 0.1,
            revolution_cap: 2.0,
            max_phase_dv: 0.5,
            max_transfer_dv: 2.0,
            min_perigee: 6_578.14,
            max_apogee: 100_000.0,
            max_revs: 4,
            sampling_mode: SamplingMode::Fast,
            execution_policy: ExecutionPolicy::default(),
            j2_closure_settings,
            search_depth: SearchDepthPolicy::default(),
            distance_tol: 0.025,
            deployer_min_distance: 0.12,
            target_propagation_authority: TargetPropagationAuthority::MfJ2,
            force_config: None,
            packed_coeffs: None,
            local_optimizer: TransferLocalOptimizerConfig::default(),
        }
    }

    #[test]
    fn explicit_j2_closure_policy_is_width_invariant() {
        let expected = J2ClosureSettings {
            max_iterations: 3,
            endpoint_target_km: 0.000_25,
            correction_step_gain: 0.31,
        };
        for width in [1usize, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(width)
                .build()
                .expect("build scoped J2 authority pool");
            pool.install(|| {
                let template = template_with_explicit_j2_policy(expected);
                assert_eq!(template.j2_closure_settings, expected);
                (0..width).into_par_iter().for_each(|_| {
                    assert_eq!(template.j2_closure_settings, expected);
                });
            });
        }
    }

    #[test]
    fn concurrent_j2_closure_policies_do_not_cross_talk() {
        let policies = [
            J2ClosureSettings {
                max_iterations: 2,
                endpoint_target_km: 0.000_5,
                correction_step_gain: 0.2,
            },
            J2ClosureSettings {
                max_iterations: 7,
                endpoint_target_km: 0.02,
                correction_step_gain: 0.9,
            },
        ];
        std::thread::scope(|scope| {
            let handles = policies.map(|policy| {
                scope.spawn(move || {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(4)
                        .build()
                        .expect("build concurrent J2 authority pool");
                    pool.install(|| template_with_explicit_j2_policy(policy).j2_closure_settings)
                })
            });
            for (handle, expected) in handles.into_iter().zip(policies) {
                assert_eq!(handle.join().expect("J2 authority thread"), expected);
            }
        });
    }
}
