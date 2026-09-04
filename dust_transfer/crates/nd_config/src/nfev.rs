//! NFEV budget policy vocabulary — a faithful port of the oracle
//! `src/shared/utils/nfev_policy.py`.
//!
//! The historical `"generations"` policy (which derived the eval budget from the
//! generation count) is RETIRED and rejected with the oracle's exact guidance.

use anyhow::{bail, Result};

/// Oracle `RETIRED_GENERATIONS_POLICY_ERROR`.
pub const RETIRED_GENERATIONS_POLICY_ERROR: &str =
    "nfev_budget_policy='generations' is retired; use policy='default' with an \
     explicit hard budget or policy='off' for uncapped generation-equal diagnostics";

/// The accepted NFEV budget policies (oracle `NFEV_BUDGET_POLICIES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NfevBudgetPolicy {
    /// Honour the explicit `nfev_budget` hard cap.
    Default,
    /// Uncapped; forbids an explicit `nfev_budget`.
    Off,
}

impl NfevBudgetPolicy {
    /// Parse a raw policy token (case-insensitive, trimmed).
    ///
    /// Rejects the retired `"generations"` policy with
    /// [`RETIRED_GENERATIONS_POLICY_ERROR`], and any other unknown token with a
    /// message mirroring the oracle's allowed-set error.
    ///
    /// # Errors
    ///
    /// Returns an error for the retired `"generations"` token or any policy
    /// outside the accepted vocabulary.
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "default" => Ok(Self::Default),
            "off" => Ok(Self::Off),
            "generations" => bail!(RETIRED_GENERATIONS_POLICY_ERROR),
            other => bail!(
                "optimization.execution.nfev_budget_policy must be one of \
                 [default, off], got {other:?}"
            ),
        }
    }
}
