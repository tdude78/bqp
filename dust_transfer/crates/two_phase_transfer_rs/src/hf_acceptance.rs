//! Legacy MF/gravity-only high-fidelity replay diagnostic for canonical transfer candidates.
//!
//! CLASSIFICATION: dissertation-evidence/test-facing surface, not production.
//! Its callers are acceptance tests and evidence tooling; no production solve
//! path reaches this module.
//!
//! `tab:lane_perturb_policy` row 2 assigns `equinoc_prop_j2` mean-element secular J2 to the bulk
//! search lane only and states that "Final acceptance requires typed HF status and fixed-I
//! authority"; the Fidelity Allocation Rationale requires "exact HF replay at the fixed target
//! epoch" for every canonical transfer candidate. Two things blocked that from running:
//!
//! 1. The HF branch of [`crate::evaluate::propagate_state_at_epoch`] is gated on `force_config`
//!    and immutable packed gravity authority, and nothing exposed a way to build that pair.
//!    [`HfGravityAuthority`] does by parsing the supplied coefficient bytes into an immutable,
//!    diagnostic-local packed authority. It never reads or mutates production's global store.
//! 2. Encke integration cannot span a multi-hour arc in one call — the integrator rectifies on a
//!    5400 s window (`integrator.rs:1251`), and a longer single call returns no state at all.
//!    [`hf_acceptance_replay`] walks each leg in fixed 5,400 s chunks, re-osculating
//!    the equinoctial reference at every boundary.
//!
//! Nothing here is reachable from the default (non-HF) solve path. There are no production
//! callers: `HfGravityAuthority` is inert until a diagnostic constructs it, and the replay operates
//! on a *clone* of the caller's `PlanContext` — the caller's context, and therefore the solve that
//! produced the candidate, is never mutated.

use std::sync::Arc;

use lightyear_odeint_rs::types::ForceConfig;

use crate::types::{PlanContext, PlanResult, TargetPropagationAuthority};
use crate::verify::{replay_transfer_controls_segmented, ReplayFailure};

/// Encke rectification window (seconds).
///
/// Matches the integrator's own internal cap and the one-LEO-orbit segment the methodology cites.
/// A replay leg longer than this must be walked in chunks or the integrator returns nothing.
pub(crate) use lightyear_odeint_rs::integrator::MAX_RECTIFICATION_SEGMENT_S as HF_REPLAY_MAX_SEGMENT_S;

/// Gravity coefficients plus the force config that make the diagnostic HF branch reachable.
///
/// Construct only for [`hf_acceptance_replay`].
#[derive(Clone, Debug)]
pub struct HfGravityAuthority {
    force_config: Arc<ForceConfig>,
    packed_coeffs: Arc<satpy_core::PackedGravityCoeffs>,
}

/// Typed failure while constructing the gravity authority needed for a diagnostic replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HfGravityAuthorityError {
    CoefficientLoad { sph_order: usize },
}

impl std::fmt::Display for HfGravityAuthorityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CoefficientLoad { sph_order } => write!(
                formatter,
                "HF acceptance could not load degree-{sph_order} gravity coefficients"
            ),
        }
    }
}

impl std::error::Error for HfGravityAuthorityError {}

impl HfGravityAuthority {
    /// Load spherical-harmonic coefficients from raw EGM/DIR text and pair them with
    /// `force_config`.
    ///
    /// The load order is `force_config.sph_order`, so the coefficient table and the force model
    /// cannot disagree. Parsing produces a diagnostic-local immutable pack; any preloaded global
    /// propagation authority remains untouched.
    ///
    /// # Errors
    ///
    /// Returns a finite typed error when gravity coefficients cannot load or do
    /// not form a usable table for `force_config`.
    pub fn load(
        coefficient_bytes: &[u8],
        force_config: ForceConfig,
    ) -> Result<Self, HfGravityAuthorityError> {
        let sph_order = force_config.sph_order;
        let packed_coeffs =
            lightyear_odeint_rs::packed_constants_from_bytes(coefficient_bytes, sph_order)
                .map_err(|_| HfGravityAuthorityError::CoefficientLoad { sph_order })?;
        Ok(Self {
            force_config: Arc::new(force_config),
            packed_coeffs,
        })
    }

    /// Stamp immutable HF gravity authority onto a `PlanContext` and enable the HF execution policy.
    ///
    /// This is the only mutation the acceptance path performs, and it is performed on a clone.
    pub(crate) fn apply_to_plan_context(&self, ctx: &mut PlanContext) {
        ctx.force_config = Some(self.force_config.clone());
        ctx.packed_coeffs = Some(self.packed_coeffs.clone());
        ctx.execution_policy.use_high_fidelity = true;
    }
}

/// Gravity-only 5x5 transfer-body force config for the legacy replay diagnostic.
///
/// `force_flags = 0` leaves drag, SRP and third-body off. `am_ratio`/`cd`/`cr` must still be
/// finite and positive because the HF transfer-body guard in
/// [`crate::evaluate::propagate_state_at_epoch`] rejects a body without them; with the force flags
/// clear they are inert.
///
/// `subtract_first_order: true` is mandatory, not cosmetic: the integrator returns an Encke delta
/// against the analytic two-body baseline, so clearing it double-counts central gravity and the
/// arc diverges (14 km at 60 s, non-finite within the hour).
///
/// `target_propagation_mode` must agree with the context's `target_propagation_authority` or the
/// target leg refuses to propagate; the default here is MF/J2 (code 1), matching the sealed rows.
#[must_use]
pub fn gravity_only_transfer_force_config() -> ForceConfig {
    ForceConfig {
        sph_order: 5,
        force_flags: 0,
        subtract_first_order: true,
        am_ratio: 0.01,
        cd: 2.2,
        cr: 1.3,
        target_propagation_mode: TargetPropagationAuthority::MfJ2.as_force_config_code(),
        dt_max: 300.0,
        ..ForceConfig::default()
    }
}

/// Outcome of one HF acceptance replay.
#[derive(Clone, Debug)]
pub struct HfAcceptanceReport {
    /// Endpoint miss between the HF-replayed payload and the target, in metres.
    pub residual_m: f64,
    /// Tolerance the residual was judged against, in metres (`ctx.distance_tol` in km x 1000).
    pub tolerance_m: f64,
    /// True when `residual_m` is finite and within `tolerance_m`.
    pub accepted: bool,
    /// The full replayed candidate. `post_hf_endpoint_residual_m` carries `residual_m`.
    pub replayed: PlanResult,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HfAcceptanceError {
    InvalidTargetPropagationAuthority(crate::types::InvalidTargetPropagationAuthorityCode),
    TargetPropagationAuthorityMismatch {
        force_config: TargetPropagationAuthority,
        context: TargetPropagationAuthority,
    },
    InvalidTransferBodyForce,
    Replay(ReplayFailure),
}

impl std::fmt::Display for HfAcceptanceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTargetPropagationAuthority(code) => {
                write!(formatter, "HF acceptance force config carries {code}")
            }
            Self::TargetPropagationAuthorityMismatch {
                force_config,
                context,
            } => write!(
                formatter,
                "HF acceptance force config declares target authority {force_config:?} \
                 but the candidate was solved under {context:?}; set \
                 ForceConfig::target_propagation_mode to match"
            ),
            Self::InvalidTransferBodyForce => formatter.write_str(
                "HF acceptance requires finite positive am_ratio/cd and non-negative cr on the \
                 transfer-body force config",
            ),
            Self::Replay(error) => write!(formatter, "HF acceptance replay: {error}"),
        }
    }
}

impl std::error::Error for HfAcceptanceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidTargetPropagationAuthority(code) => Some(code),
            Self::TargetPropagationAuthorityMismatch { .. } | Self::InvalidTransferBodyForce => {
                None
            }
            Self::Replay(error) => Some(error),
        }
    }
}

/// Run the legacy MF/gravity-only HF replay diagnostic for one solved candidate.
///
/// `ctx` is the context the candidate was solved in; it is cloned, not mutated. The clone is put
/// on `authority`'s gravity-only HF force model, and the stored burn sequence
/// (E0 -> phase -> coast -> transfer -> intercept) is replayed under it, each leg walked in
/// fixed 5,400 s chunks. No optimizer or Lambert search runs; the controls are
/// consumed exactly as stored. This is not compiled Hybrid acceptance physics.
///
/// The catalogue target keeps whatever authority `ctx` declares, so a `MfJ2` context isolates
/// transfer-arc fidelity instead of confounding it with a target-model change.
///
/// # Errors
///
/// Returns an error if authority metadata disagree, force parameters are
/// invalid, or segmented high-fidelity replay fails.
pub fn hf_acceptance_replay(
    result: &PlanResult,
    ctx: &PlanContext,
    authority: &HfGravityAuthority,
) -> Result<HfAcceptanceReport, HfAcceptanceError> {
    let declared_target_authority =
        TargetPropagationAuthority::try_from(authority.force_config.target_propagation_mode)
            .map_err(HfAcceptanceError::InvalidTargetPropagationAuthority)?;
    if declared_target_authority != ctx.target_propagation_authority {
        return Err(HfAcceptanceError::TargetPropagationAuthorityMismatch {
            force_config: declared_target_authority,
            context: ctx.target_propagation_authority,
        });
    }

    let mut hf_ctx = ctx.clone();
    authority.apply_to_plan_context(&mut hf_ctx);

    let transfer_body = hf_ctx.transfer_body_force();
    if !(transfer_body.am_ratio.is_finite()
        && transfer_body.am_ratio > 0.0
        && transfer_body.cd.is_finite()
        && transfer_body.cd > 0.0
        && transfer_body.cr.is_finite()
        && transfer_body.cr >= 0.0)
    {
        return Err(HfAcceptanceError::InvalidTransferBodyForce);
    }

    let replayed =
        replay_transfer_controls_segmented(result, &hf_ctx, Some(HF_REPLAY_MAX_SEGMENT_S))
            .map_err(HfAcceptanceError::Replay)?;
    let residual_m = replayed.distance * 1000.0;
    let tolerance_m = hf_ctx.distance_tol * 1000.0;
    Ok(HfAcceptanceReport {
        residual_m,
        tolerance_m,
        accepted: residual_m.is_finite() && residual_m <= tolerance_m,
        replayed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn run_isolated_global_coeffs_test() -> anyhow::Result<()> {
        const CHILD_ENV: &str = "ND_HF_ACCEPTANCE_LOCAL_GRAVITY_CHILD";
        const TEST_NAME: &str =
            "hf_acceptance::tests::alternate_diagnostic_load_preserves_preloaded_global_authority";
        if std::env::var_os(CHILD_ENV).is_some() {
            return Ok(());
        }
        let test_binary = std::env::current_exe()
            .map_err(|error| anyhow::anyhow!("cannot locate HF acceptance test binary: {error}"))?;
        let output = Command::new(test_binary)
            .args(["--exact", TEST_NAME])
            .env(CHILD_ENV, "1")
            .output()
            .map_err(|error| anyhow::anyhow!("cannot run isolated HF acceptance test: {error}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        anyhow::ensure!(
            output.status.success()
                && stdout.contains("running 1 test")
                && stdout.contains(TEST_NAME),
            "isolated HF acceptance test `{TEST_NAME}` failed or matched no test\nstdout:\n{stdout}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(())
    }

    /// Structurally valid but physically trivial authority. These tests exercise the guards and
    /// the clone discipline, never the integrator, so a point-mass table is sufficient.
    fn stub_authority(force_config: Arc<ForceConfig>) -> HfGravityAuthority {
        let packed = Arc::new(
            satpy_core::pack_gravity_coeffs(&[1.0], &[0.0], 1, 0)
                .expect("test gravity coefficients are valid"),
        );
        HfGravityAuthority {
            force_config,
            packed_coeffs: packed,
        }
    }

    #[test]
    fn gravity_only_config_is_encke_safe_and_mf_j2_typed() {
        let config = gravity_only_transfer_force_config();
        assert!(config.subtract_first_order);
        assert_eq!(config.force_flags, 0);
        assert_eq!(config.sph_order, 5);
        assert_eq!(
            TargetPropagationAuthority::try_from(config.target_propagation_mode),
            Ok(TargetPropagationAuthority::MfJ2)
        );
    }

    #[test]
    fn authority_load_reports_typed_coefficient_failure() {
        let outcome = HfGravityAuthority::load(
            b"",
            ForceConfig {
                sph_order: usize::MAX,
                ..gravity_only_transfer_force_config()
            },
        );

        assert!(
            outcome.is_err(),
            "overflowing coefficient order must fail closed"
        );
        let Err(error) = outcome else {
            return;
        };
        assert_eq!(
            error,
            HfGravityAuthorityError::CoefficientLoad {
                sph_order: usize::MAX,
            }
        );
    }

    #[test]
    fn alternate_diagnostic_load_preserves_preloaded_global_authority() -> anyhow::Result<()> {
        const ALTERNATE: &[u8] = b"2 0 -4.00000D-4 0.0\n2 2 1.0D-6 -2.0D-6\n";
        const ORDER: usize = 2;

        if std::env::var_os("ND_HF_ACCEPTANCE_LOCAL_GRAVITY_CHILD").is_none() {
            return run_isolated_global_coeffs_test();
        }

        let before = lightyear_odeint_rs::config::GLOBAL_COEFFS.load_full();

        let diagnostic = HfGravityAuthority::load(
            ALTERNATE,
            ForceConfig {
                sph_order: ORDER,
                ..gravity_only_transfer_force_config()
            },
        )?;
        let after = lightyear_odeint_rs::config::GLOBAL_COEFFS.load_full();

        anyhow::ensure!(
            Arc::ptr_eq(&before, &after),
            "diagnostic load replaced the preloaded global gravity authority"
        );
        anyhow::ensure!(
            diagnostic.packed_coeffs.max_order() == ORDER,
            "alternate diagnostic fixture loaded wrong gravity order"
        );
        Ok(())
    }

    #[test]
    fn acceptance_rejects_target_authority_mismatch() {
        let authority = stub_authority(Arc::new(ForceConfig {
            target_propagation_mode: TargetPropagationAuthority::AnalyticalKepler
                .as_force_config_code(),
            ..gravity_only_transfer_force_config()
        }));
        let ctx = PlanContext {
            target_propagation_authority: TargetPropagationAuthority::MfJ2,
            ..PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default())
        };
        let mut result = PlanResult::invalid();
        result.valid = true;
        let outcome = hf_acceptance_replay(&result, &ctx, &authority);
        assert!(
            outcome.is_err(),
            "mismatched target authority must not silently replay"
        );
        let Err(error) = outcome else {
            return;
        };
        assert_eq!(
            error,
            HfAcceptanceError::TargetPropagationAuthorityMismatch {
                force_config: TargetPropagationAuthority::AnalyticalKepler,
                context: TargetPropagationAuthority::MfJ2,
            }
        );
    }

    #[test]
    fn acceptance_rejects_unknown_target_authority_code() {
        let authority = stub_authority(Arc::new(ForceConfig {
            target_propagation_mode: u8::MAX,
            ..gravity_only_transfer_force_config()
        }));
        let ctx = PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        let mut result = PlanResult::invalid();
        result.valid = true;

        let outcome = hf_acceptance_replay(&result, &ctx, &authority);
        assert!(outcome.is_err(), "unknown authority code must not replay");
        let Err(error) = outcome else {
            return;
        };
        assert_eq!(
            error,
            HfAcceptanceError::InvalidTargetPropagationAuthority(
                crate::types::InvalidTargetPropagationAuthorityCode::InvalidCode(u8::MAX)
            )
        );
    }

    #[test]
    fn applying_authority_leaves_the_callers_context_untouched() {
        let ctx = PlanContext::with_j2_closure_settings(crate::solve::J2ClosureSettings::default());
        assert!(!ctx.execution_policy.use_high_fidelity);
        assert!(ctx.force_config.is_none());
        let authority = stub_authority(Arc::new(gravity_only_transfer_force_config()));
        let mut clone = ctx.clone();
        authority.apply_to_plan_context(&mut clone);
        assert!(clone.execution_policy.use_high_fidelity);
        assert!(clone.force_config.is_some());
        assert!(!ctx.execution_policy.use_high_fidelity);
        assert!(ctx.force_config.is_none());
    }
}
