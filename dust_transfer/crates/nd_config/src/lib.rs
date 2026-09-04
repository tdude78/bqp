//! `nd_config` — lean, validated ingestion of the dissertation production config.
//!
//! This is a deliberate do-over of the oracle Python ingestion
//! (`tools/_config_ingestion.py` + `tools/_config_resolution.py`, ~3 KLOC). It
//! models ONLY the fields the matrix run and pipeline actually consume, not the
//! full ~113 KB schema. Every other YAML section (`tc`, `constants`, `solver`,
//! `beta_dist`, `evaluation`, `dissertation_pipeline`, …) is simply ignored:
//! sub-structs that we only partially model are *lenient* (unknown keys are
//! dropped), so the canonical `dissertation_production.yaml` parses without
//! enumerating its ~20 unrelated top-level sections.
//!
//! # Strictness policy
//! `#[serde(deny_unknown_fields)]` is applied ONLY to structs we model in full:
//! [`ConfigMeta`] (the `config:` block — exactly `version`/`role`/`profile`) and
//! [`Execution`] (the `optimization.execution:` block — exactly its seven keys).
//! All other sub-structs are lenient by design.

use std::borrow::Cow;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

mod nfev;
mod part_a_science;
pub use nfev::NfevBudgetPolicy;
pub use part_a_science::{
    token_eq, CompiledPartAScienceV1, PartAAdaptiveEventsError, PartAAuthorityError,
    PartAConstellationControls, PartACoverageFrontMinBand, PartACredibleIntervalMethod,
    PartAEarthOrientationConvention, PartAEventAnchorAuthority, PartAGravityAuthority,
    PartAH64Controls, PartAHybridControls, PartAK3Controls, PartAMfControls,
    PartAMfLoweringControls, PartAMfNativePolicyV1, PartAMfTransferControls,
    PartANativeHybridControls, PartAObjectiveAggregation, PartAReferenceEvidence,
    PartAReportingControls, PartASearchModel, PartASharedTargetClaim, PartASharedTargetControls,
    PartASharedTargetDrawIntegration, PartASharedTargetPositionTreatment, PartASuccessEstimator,
    PartATaiEpoch, PartAVerifiedEventAnchor, PartAVerifiedEventAnchorInput, PartAVerifiedGravity,
    PartAVerifiedGravityInput, PART_A_DEPLOYER_OBJECT_ID_BASE,
    PART_A_EXACT_DESIGN_CACHE_MAX_ENTRIES, PART_A_K3_BARRIER_COUNT, PART_A_SCIENCE_SCHEMA_VERSION,
};

/// Canonical matrix seed anchor (`optimization.execution.seed`). Used as the
/// default when a config/overlay omits an explicit seed.
pub const DEFAULT_SEED: u64 = 41_127_203;
// PART_A_INTERSECT_SEEDS and PART_A_INTERSECT_BARRIERS lived here with zero
// readers, duplicating the compiled K3 authority: seeds come from
// `CompiledPartAScienceV1::k3().seeds` and barriers from
// `nd_driver::part_a_k3_controls().barriers`, which live driver callers use.
// A second spelling of a sealed value is a place for the two to disagree.
pub const PART_A_ARCHIVE_MAX_SIZE: usize = 4096;

/// One sealed Part A campaign shape. This is config semantic authority, not a
/// file-path or digest allowlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartACampaignScope {
    Exact36,
    #[serde(rename = "mf18g500")]
    Mf18G500Sensitivity,
    #[serde(rename = "mf18g1000")]
    Mf18G1000Sensitivity,
    Intersect108,
}

/// Run profile. Only `mf` and `hybrid` are accepted (oracle
/// `ALLOWED_CONFIG_PROFILES = {"hybrid", "mf"}`); any other token fails
/// deserialization with an "unknown variant" error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    #[default]
    Mf,
    Hybrid,
}

/// Physical objective fidelity. Matrix axis is explicit; never inferred from
/// `hf.use_high_fidelity`, which only configures hybrid kernels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Fidelity {
    #[default]
    Mf,
    Hybrid,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatrixMode {
    #[default]
    Independent,
    IntersectK3,
}

impl From<Profile> for Fidelity {
    fn from(profile: Profile) -> Self {
        match profile {
            Profile::Mf => Self::Mf,
            Profile::Hybrid => Self::Hybrid,
        }
    }
}

/// The `config:` meta block. STRICT: modeled in full.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigMeta {
    #[serde(default)]
    pub version: Option<u32>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub profile: Profile,
}

/// The `optimization.execution:` block. STRICT: modeled generic execution
/// controls are explicit; canonical Part A forbids NFEV stop controls.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Execution {
    #[serde(default)]
    pub events: Option<u32>,
    #[serde(default)]
    pub population_size: Option<u32>,
    #[serde(default)]
    pub generations: Option<u32>,
    #[serde(default)]
    pub nfev_budget: Option<u64>,
    /// Raw policy token, validated lazily via [`Execution::nfev_policy`] so the
    /// retired `"generations"` value surfaces the oracle error message rather
    /// than a generic serde "unknown variant".
    #[serde(default)]
    pub nfev_budget_policy: Option<String>,
    #[serde(default)]
    pub nfev_budget_source: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Execution {
    /// Resolved seed: explicit `seed` if present, else [`DEFAULT_SEED`].
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed.unwrap_or(DEFAULT_SEED)
    }

    /// Parse the nfev budget policy. Absent → [`NfevBudgetPolicy::Default`].
    /// Rejects the retired `"generations"` policy (oracle `nfev_policy.py`).
    ///
    /// # Errors
    ///
    /// Returns an error if an explicit policy token is retired or unknown.
    pub fn nfev_policy(&self) -> Result<NfevBudgetPolicy> {
        self.nfev_budget_policy
            .as_deref()
            .map_or_else(|| Ok(NfevBudgetPolicy::Default), NfevBudgetPolicy::parse)
    }
}

/// The `optimization.matrix:` block. LENIENT: `replication`,
/// `family_seeds_yaml`, `seed_initialization`, etc. are not modeled.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Matrix {
    #[serde(default)]
    pub mode: MatrixMode,
    #[serde(default)]
    pub seed_list: Vec<u64>,
    #[serde(default)]
    pub optimizers: Vec<String>,
    #[serde(default)]
    pub constellation_families: Vec<String>,
    /// Explicit fidelity axis. Empty preserves single-profile configs by using
    /// `config.profile`; multi-fidelity Part A configs must list both values.
    #[serde(default, alias = "fidelities")]
    pub fidelity_list: Vec<Fidelity>,
}

/// Shared runtime archive authority (`optimization.archive`). LENIENT.
/// Algorithm-local `external_archive` blocks are legacy input and intentionally
/// do not override this shared provider.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ArchiveConfig {
    #[serde(default, deserialize_with = "non_null")]
    pub enabled: Option<bool>,
    #[serde(default, deserialize_with = "non_null")]
    pub max_size: Option<usize>,
    #[serde(default, deserialize_with = "non_null")]
    pub history_max_size: Option<usize>,
}

/// The `optimization:` block (the axes we run over). LENIENT.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Optimization {
    #[serde(default)]
    pub execution: Execution,
    #[serde(default)]
    pub matrix: Matrix,
    #[serde(default)]
    pub archive: ArchiveConfig,
    #[serde(default)]
    pub algorithms: Algorithms,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum MutationProbability {
    Scalar(f64),
    Range(f64, f64),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum DeCoefficient {
    Scalar(f64),
    Range(f64, f64),
}

fn non_null<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(deserializer)?
        .ok_or_else(|| D::Error::custom("must not be null"))
        .map(Some)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Nullable<T> {
    #[default]
    Missing,
    Null,
    Value(T),
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Nullable<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Ok(Option::<T>::deserialize(deserializer)?.map_or_else(|| Self::Null, Self::Value))
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum DiversityEpsilon {
    Scalar(f64),
    Vector(Vec<f64>),
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AlgorithmDefaults {
    #[serde(default)]
    pub random_state: Nullable<u64>,
    #[serde(default, deserialize_with = "non_null")]
    pub crossover_prob: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub mutation_prob: Option<MutationProbability>,
    #[serde(default, deserialize_with = "non_null")]
    pub epsilon: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub eta_c: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub eta_m: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub tournament_size: Option<u32>,
    #[serde(default, deserialize_with = "non_null")]
    pub reinit_fraction: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub reinit_generations: Option<u32>,
    #[serde(default)]
    pub gap_perturbation_scale: Nullable<f64>,
    #[serde(default)]
    pub gap_offspring_fraction: Nullable<f64>,
    #[serde(default)]
    pub selection_method: Option<String>,
    #[serde(default)]
    pub diversity_epsilon: Nullable<DiversityEpsilon>,
    #[serde(default, deserialize_with = "non_null")]
    pub stability_window: Option<u32>,
    #[serde(default)]
    pub epsilon_decay: Nullable<f64>,
    #[serde(default)]
    pub epsilon_min: Nullable<f64>,
    #[serde(default)]
    pub init_strategy: Option<String>,
    #[serde(default, deserialize_with = "non_null")]
    pub stability_tol: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub stagnation_restart_frac: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub selection_balance: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub cr: Option<DeCoefficient>,
    #[serde(default, deserialize_with = "non_null")]
    pub f: Option<DeCoefficient>,
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default)]
    pub enable_cache: Nullable<bool>,
    #[serde(default)]
    pub gap_filling_enabled: Nullable<bool>,
    #[serde(default)]
    pub diversity_parity_mode: Nullable<bool>,
    #[serde(default)]
    pub design_unique_selection_enabled: Nullable<bool>,
    #[serde(default, deserialize_with = "non_null")]
    pub pop_random_fraction: Option<f64>,
    #[serde(default)]
    pub prde_max_local_refinements: Nullable<u32>,
    #[serde(default)]
    pub prde_refine_fraction: Nullable<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub prde_local_max_attempts: Option<u32>,
    #[serde(default)]
    pub prde_local_step_scale: Nullable<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub prde_refinement_gain_threshold: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub prde_refinement_max_stall: Option<u32>,
    #[serde(default)]
    pub prde_refine_with_constraints: Nullable<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RnsdeOverrides {
    #[serde(flatten)]
    pub common: AlgorithmDefaults,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PrnsdeOverrides {
    #[serde(flatten)]
    pub common: AlgorithmDefaults,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Nsga2Overrides {
    #[serde(flatten)]
    pub common: AlgorithmDefaults,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EpsNsga2Overrides {
    #[serde(flatten)]
    pub common: AlgorithmDefaults,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AgeMoea2Overrides {
    #[serde(flatten)]
    pub common: AlgorithmDefaults,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct MopsoOverrides {
    #[serde(default)]
    pub random_state: Nullable<u64>,
    #[serde(default, deserialize_with = "non_null")]
    pub w_max: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub w_min: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub c1: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub c2: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub mutation_prob: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub velocity_jitter: Option<f64>,
    #[serde(default)]
    pub ensure_diversity: Nullable<bool>,
    #[serde(default)]
    pub gap_filling_enabled: Nullable<bool>,
    #[serde(default, deserialize_with = "non_null")]
    pub reinit_fraction: Option<f64>,
    #[serde(default, deserialize_with = "non_null")]
    pub reinit_generations: Option<u32>,
    #[serde(default)]
    pub gap_perturbation_scale: Nullable<f64>,
    #[serde(default)]
    pub gap_offspring_fraction: Nullable<f64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Algorithms {
    #[serde(default)]
    pub defaults: AlgorithmDefaults,
    #[serde(default)]
    pub eps_nsga2: EpsNsga2Overrides,
    #[serde(default)]
    pub age_moea2: AgeMoea2Overrides,
    #[serde(default)]
    pub mopso: MopsoOverrides,
    #[serde(default)]
    pub rnsde: RnsdeOverrides,
    #[serde(default)]
    pub prnsde: PrnsdeOverrides,
    #[serde(default)]
    pub nsga2: Nsga2Overrides,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpsNsga2Config {
    pub crossover_prob: f64,
    pub mutation_prob: MutationProbability,
    pub epsilon: f64,
    pub eta_c: f64,
    pub eta_m: f64,
    pub tournament_size: u32,
    pub reinit_fraction: f64,
    pub reinit_generations: u32,
    pub gap_perturbation_scale: f64,
    pub gap_offspring_fraction: Option<f64>,
    pub random_state: Option<u64>,
    pub selection_method: String,
    pub diversity_epsilon: Option<DiversityEpsilon>,
    pub stability_window: u32,
    pub epsilon_decay: Option<f64>,
    pub epsilon_min: Option<f64>,
    pub init_strategy: String,
    pub stability_tol: f64,
    pub stagnation_restart_frac: f64,
    pub selection_balance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgeMoea2Config {
    pub crossover_prob: f64,
    pub mutation_prob: MutationProbability,
    pub eta_c: f64,
    pub eta_m: f64,
    pub tournament_size: u32,
    pub reinit_fraction: f64,
    pub reinit_generations: u32,
    pub gap_filling_enabled: bool,
    pub gap_perturbation_scale: f64,
    pub gap_offspring_fraction: Option<f64>,
    pub random_state: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MopsoConfig {
    pub w_max: f64,
    pub w_min: f64,
    pub c1: f64,
    pub c2: f64,
    pub mutation_prob: f64,
    pub velocity_jitter: f64,
    pub gap_filling_enabled: bool,
    pub reinit_fraction: f64,
    pub reinit_generations: u32,
    pub gap_perturbation_scale: f64,
    pub gap_offspring_fraction: Option<f64>,
    pub random_state: Option<u64>,
    pub ensure_diversity: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RnsdeConfig {
    pub cr: DeCoefficient,
    pub f: DeCoefficient,
    pub strategy: String,
    pub reinit_fraction: f64,
    pub reinit_generations: u32,
    pub enable_cache: bool,
    pub gap_filling_enabled: bool,
    pub gap_perturbation_scale: f64,
    pub gap_offspring_fraction: Option<f64>,
    pub diversity_parity_mode: bool,
    pub random_state: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrnsdeConfig {
    pub cr: DeCoefficient,
    pub f: DeCoefficient,
    pub strategy: String,
    pub reinit_fraction: f64,
    pub reinit_generations: u32,
    pub enable_cache: bool,
    pub gap_filling_enabled: bool,
    pub gap_perturbation_scale: f64,
    pub gap_offspring_fraction: Option<f64>,
    pub random_state: Option<u64>,
    pub pop_random_fraction: f64,
    pub prde_max_local_refinements: Option<u32>,
    pub prde_refine_fraction: Option<f64>,
    pub prde_local_max_attempts: u32,
    pub prde_local_step_scale: f64,
    pub prde_refinement_gain_threshold: f64,
    pub prde_refinement_max_stall: u32,
    pub prde_refine_with_constraints: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Nsga2Config {
    pub crossover_prob: f64,
    pub mutation_prob: MutationProbability,
    pub eta_c: f64,
    pub eta_m: f64,
    pub tournament_size: u32,
    pub reinit_fraction: f64,
    pub reinit_generations: u32,
    pub design_unique_selection_enabled: bool,
    pub gap_filling_enabled: bool,
    pub gap_perturbation_scale: f64,
    pub gap_offspring_fraction: Option<f64>,
    pub random_state: Option<u64>,
}

/// Own → (base →) defaults fallback that names the config field ONCE.
///
/// The hand-written chains restated each field name in every layer position
/// (`own.x.or(base.x).or(d.x)`), so a copy-paste naming the wrong field in
/// one position was silent — the types match across most fields. Every rule
/// here reads the SAME field from every layer by construction. Two-layer
/// rules (own, defaults) and three-layer rules (own, base, defaults) are
/// distinguished by arity; same defaults, same semantics as the chains they
/// replace.
macro_rules! resolve_field {
    // Copy Option chains with a final fallback.
    (($own:expr, $base:expr, $d:expr).$field:ident else $fallback:expr) => {
        $own.$field
            .or($base.$field)
            .or($d.$field)
            .unwrap_or($fallback)
    };
    (($own:expr, $d:expr).$field:ident else $fallback:expr) => {
        $own.$field.or($d.$field).unwrap_or($fallback)
    };
    // Clone Option chains (lazy, as the hand chains were).
    (clone($own:expr, $base:expr, $d:expr).$field:ident else $fallback:expr) => {
        $own.$field
            .clone()
            .or_else(|| $base.$field.clone())
            .or_else(|| $d.$field.clone())
            .unwrap_or_else(|| $fallback)
    };
    (clone($own:expr, $d:expr).$field:ident else $fallback:expr) => {
        $own.$field
            .clone()
            .or_else(|| $d.$field.clone())
            .unwrap_or_else(|| $fallback)
    };
    // Nullable three-state resolution to Option (None when explicitly null).
    (nullable($own:expr, $base:expr, $d:expr).$field:ident) => {
        resolve_three(&$own.$field, &$base.$field, &$d.$field)
    };
    (nullable($own:expr, $d:expr).$field:ident) => {
        resolve_nullable(&$own.$field, &$d.$field)
    };
    // Nullable three-state resolution kept as Nullable (caller matches arms).
    (three_state($own:expr, $base:expr, $d:expr).$field:ident) => {
        resolve_three_state(&$own.$field, &$base.$field, &$d.$field)
    };
    // Strict boolean: explicit null is an error; the reported name is the
    // field name itself (stringify!), so it cannot drift from the field read.
    (strict_bool($own:expr, $base:expr, $d:expr).$field:ident else $default:expr) => {
        strict_bool(
            stringify!($field),
            &resolve_three_state(&$own.$field, &$base.$field, &$d.$field),
            $default,
        )
    };
    (strict_bool($own:expr, $d:expr).$field:ident else $default:expr) => {
        strict_bool(
            stringify!($field),
            &resolve_two_state(&$own.$field, &$d.$field),
            $default,
        )
    };
}

impl Algorithms {
    /// # Errors
    ///
    /// Returns an error when a resolved RNSDE boolean control is explicitly
    /// null instead of a boolean.
    pub fn rnsde_resolved(&self) -> Result<RnsdeConfig> {
        self.resolve_rnsde(&self.rnsde.common, &AlgorithmDefaults::default())
    }

    /// # Errors
    ///
    /// Returns an error when a resolved PRNSDE boolean control is explicitly
    /// null instead of a boolean.
    pub fn prnsde_resolved(&self) -> Result<PrnsdeConfig> {
        let own = &self.prnsde.common;
        let base = &self.rnsde.common;
        let d = &self.defaults;
        let core = self.resolve_rnsde(own, base)?;
        let step_state = resolve_field!(three_state(own, base, d).prde_local_step_scale);
        Ok(PrnsdeConfig {
            cr: core.cr,
            f: core.f,
            strategy: core.strategy,
            reinit_fraction: core.reinit_fraction,
            reinit_generations: core.reinit_generations,
            enable_cache: core.enable_cache,
            gap_filling_enabled: core.gap_filling_enabled,
            gap_perturbation_scale: core.gap_perturbation_scale,
            gap_offspring_fraction: core.gap_offspring_fraction,
            random_state: core.random_state,
            pop_random_fraction: resolve_field!((own, base, d).pop_random_fraction else 0.25),
            prde_max_local_refinements: match resolve_field!(
                three_state(own, base, d).prde_max_local_refinements
            ) {
                Nullable::Value(v) => Some(v),
                Nullable::Null => None,
                Nullable::Missing => Some(2),
            },
            prde_refine_fraction: resolve_field!(nullable(own, base, d).prde_refine_fraction),
            prde_local_max_attempts: resolve_field!((own, base, d).prde_local_max_attempts else 10),
            prde_local_step_scale: match step_state {
                Nullable::Value(v) => v,
                Nullable::Null => 0.1,
                Nullable::Missing => 0.03,
            },
            prde_refinement_gain_threshold: resolve_field!(
                (own, base, d).prde_refinement_gain_threshold else 0.0
            ),
            prde_refinement_max_stall: resolve_field!(
                (own, base, d).prde_refinement_max_stall else 1
            ),
            prde_refine_with_constraints: matches!(
                resolve_field!(three_state(own, base, d).prde_refine_with_constraints),
                Nullable::Value(true)
            ),
        })
    }
    fn resolve_rnsde(
        &self,
        own: &AlgorithmDefaults,
        base: &AlgorithmDefaults,
    ) -> Result<RnsdeConfig> {
        let d = &self.defaults;
        Ok(RnsdeConfig {
            cr: resolve_field!(clone(own, base, d).cr else DeCoefficient::Range(0.1, 0.9)),
            f: resolve_field!(clone(own, base, d).f else DeCoefficient::Range(0.5, 2.0)),
            strategy: resolve_field!(clone(own, base, d).strategy else "rand1exp".into()),
            reinit_fraction: resolve_field!((own, base, d).reinit_fraction else 0.1),
            reinit_generations: resolve_field!((own, base, d).reinit_generations else 25),
            enable_cache: matches!(
                resolve_field!(three_state(own, base, d).enable_cache),
                Nullable::Value(true)
            ),
            gap_filling_enabled: resolve_field!(
                strict_bool(own, base, d).gap_filling_enabled else true
            )?,
            gap_perturbation_scale: resolve_field!(nullable(own, base, d).gap_perturbation_scale)
                .unwrap_or(0.1),
            gap_offspring_fraction: resolve_field!(nullable(own, base, d).gap_offspring_fraction),
            diversity_parity_mode: resolve_field!(
                strict_bool(own, base, d).diversity_parity_mode else false
            )?,
            random_state: resolve_field!(nullable(own, base, d).random_state),
        })
    }
    /// # Errors
    ///
    /// Returns an error when a resolved NSGA-II boolean control is explicitly
    /// null instead of a boolean.
    pub fn nsga2_resolved(&self) -> Result<Nsga2Config> {
        let own = &self.nsga2.common;
        let d = &self.defaults;
        Ok(Nsga2Config {
            crossover_prob: resolve_field!((own, d).crossover_prob else 0.9),
            mutation_prob: resolve_field!(
                clone(own, d).mutation_prob else MutationProbability::Range(0.5, 1.0)
            ),
            eta_c: resolve_field!((own, d).eta_c else 20.0),
            eta_m: resolve_field!((own, d).eta_m else 20.0),
            tournament_size: resolve_field!((own, d).tournament_size else 2),
            reinit_fraction: resolve_field!((own, d).reinit_fraction else 0.1),
            reinit_generations: resolve_field!((own, d).reinit_generations else 25),
            design_unique_selection_enabled: resolve_field!(
                strict_bool(own, d).design_unique_selection_enabled else false
            )?,
            gap_filling_enabled: resolve_field!(strict_bool(own, d).gap_filling_enabled else true)?,
            gap_perturbation_scale: resolve_field!(nullable(own, d).gap_perturbation_scale)
                .unwrap_or(0.1),
            gap_offspring_fraction: resolve_field!(nullable(own, d).gap_offspring_fraction),
            random_state: resolve_field!(nullable(own, d).random_state),
        })
    }
    /// # Errors
    ///
    /// This keeps a uniform fallible resolver surface with the other
    /// optimizers; malformed fields are rejected during deserialization.
    pub fn eps_nsga2_resolved(&self) -> Result<EpsNsga2Config> {
        Ok(self.resolve_common(&self.eps_nsga2.common, 0.7))
    }

    /// # Errors
    ///
    /// Returns an error when a resolved AGE-MOEA2 boolean control is explicitly
    /// null instead of a boolean.
    pub fn age_moea2_resolved(&self) -> Result<AgeMoea2Config> {
        let own = &self.age_moea2.common;
        let d = &self.defaults;
        Ok(AgeMoea2Config {
            crossover_prob: resolve_field!((own, d).crossover_prob else 0.9),
            mutation_prob: resolve_field!(
                clone(own, d).mutation_prob else MutationProbability::Range(0.5, 1.0)
            ),
            eta_c: resolve_field!((own, d).eta_c else 20.0),
            eta_m: resolve_field!((own, d).eta_m else 20.0),
            tournament_size: resolve_field!((own, d).tournament_size else 2),
            reinit_fraction: resolve_field!((own, d).reinit_fraction else 0.1),
            reinit_generations: resolve_field!((own, d).reinit_generations else 25),
            gap_filling_enabled: resolve_field!(strict_bool(own, d).gap_filling_enabled else true)?,
            gap_perturbation_scale: resolve_field!(nullable(own, d).gap_perturbation_scale)
                .unwrap_or(0.1),
            gap_offspring_fraction: resolve_field!(nullable(own, d).gap_offspring_fraction),
            random_state: resolve_field!(nullable(own, d).random_state),
        })
    }
    fn resolve_common(&self, own: &AlgorithmDefaults, crossover_default: f64) -> EpsNsga2Config {
        let d = &self.defaults;
        EpsNsga2Config {
            crossover_prob: resolve_field!((own, d).crossover_prob else crossover_default),
            mutation_prob: resolve_field!(
                clone(own, d).mutation_prob else MutationProbability::Range(0.5, 1.0)
            ),
            epsilon: resolve_field!((own, d).epsilon else 1e-3),
            eta_c: resolve_field!((own, d).eta_c else 20.0),
            eta_m: resolve_field!((own, d).eta_m else 20.0),
            tournament_size: resolve_field!((own, d).tournament_size else 2),
            reinit_fraction: resolve_field!((own, d).reinit_fraction else 0.1),
            reinit_generations: resolve_field!((own, d).reinit_generations else 25),
            gap_perturbation_scale: resolve_field!(nullable(own, d).gap_perturbation_scale)
                .unwrap_or(0.1),
            gap_offspring_fraction: resolve_field!(nullable(own, d).gap_offspring_fraction),
            random_state: resolve_field!(nullable(own, d).random_state),
            selection_method: resolve_field!(
                clone(own, d).selection_method else "ideal_distance".into()
            ),
            diversity_epsilon: resolve_field!(nullable(own, d).diversity_epsilon),
            stability_window: resolve_field!((own, d).stability_window else 5),
            epsilon_decay: resolve_field!(nullable(own, d).epsilon_decay),
            epsilon_min: resolve_field!(nullable(own, d).epsilon_min),
            init_strategy: resolve_field!(clone(own, d).init_strategy else "random".into()),
            stability_tol: resolve_field!((own, d).stability_tol else 1e-4),
            stagnation_restart_frac: resolve_field!((own, d).stagnation_restart_frac else 0.0),
            selection_balance: resolve_field!((own, d).selection_balance else 0.5),
        }
    }

    /// # Errors
    ///
    /// Returns an error when the MOPSO gap-filling control is explicitly null.
    pub fn mopso_resolved(&self) -> Result<MopsoConfig> {
        let m = &self.mopso;
        let gap_filling_enabled = match m.gap_filling_enabled {
            Nullable::Value(v) => v,
            Nullable::Missing => true,
            Nullable::Null => {
                bail!("optimization.algorithms.mopso.gap_filling_enabled must be a boolean")
            }
        };
        Ok(MopsoConfig {
            w_max: m.w_max.unwrap_or(0.9),
            w_min: m.w_min.unwrap_or(0.3),
            c1: m.c1.unwrap_or(1.8),
            c2: m.c2.unwrap_or(1.8),
            mutation_prob: m.mutation_prob.unwrap_or(0.25),
            velocity_jitter: m.velocity_jitter.unwrap_or(0.05),
            gap_filling_enabled,
            reinit_fraction: m.reinit_fraction.unwrap_or(0.1),
            reinit_generations: m.reinit_generations.unwrap_or(25),
            gap_perturbation_scale: resolve_nullable(&m.gap_perturbation_scale, &Nullable::Missing)
                .unwrap_or(0.1),
            gap_offspring_fraction: resolve_nullable(&m.gap_offspring_fraction, &Nullable::Missing),
            random_state: resolve_nullable(&m.random_state, &self.defaults.random_state),
            ensure_diversity: !matches!(
                m.ensure_diversity,
                Nullable::Value(false) | Nullable::Null
            ),
        })
    }
}

fn resolve_nullable<T: Clone>(own: &Nullable<T>, defaults: &Nullable<T>) -> Option<T> {
    match own {
        Nullable::Value(v) => Some(v.clone()),
        Nullable::Null => None,
        Nullable::Missing => match defaults {
            Nullable::Value(v) => Some(v.clone()),
            Nullable::Null | Nullable::Missing => None,
        },
    }
}

fn nullable_value<T: Clone>(value: &Nullable<T>) -> Option<T> {
    match value {
        Nullable::Value(v) => Some(v.clone()),
        Nullable::Null | Nullable::Missing => None,
    }
}

fn resolve_two_state<T: Clone>(own: &Nullable<T>, defaults: &Nullable<T>) -> Nullable<T> {
    match own {
        Nullable::Missing => defaults.clone(),
        value => value.clone(),
    }
}

fn resolve_three_state<T: Clone>(
    own: &Nullable<T>,
    base: &Nullable<T>,
    defaults: &Nullable<T>,
) -> Nullable<T> {
    match own {
        Nullable::Missing => resolve_two_state(base, defaults),
        value => value.clone(),
    }
}

fn resolve_three<T: Clone>(
    own: &Nullable<T>,
    base: &Nullable<T>,
    defaults: &Nullable<T>,
) -> Option<T> {
    nullable_value(&resolve_three_state(own, base, defaults))
}

fn strict_bool(field: &str, value: &Nullable<bool>, fallback: bool) -> Result<bool> {
    match value {
        Nullable::Value(v) => Ok(*v),
        Nullable::Missing => Ok(fallback),
        Nullable::Null => bail!("optimization.algorithms.{field} must be a boolean"),
    }
}

/// The `hf:` block. `use_high_fidelity` is the only workspace control here.
/// Integration parameters come from compiled Part A science, not YAML.
///
/// Those keys are not silently ignored. `part_a_science_override_paths`
/// collects every `hf.*` key other than `use_high_fidelity` straight off the
/// raw YAML mapping, and [`Config::validate_part_a_semantics`] rejects the
/// config naming the offending path. That check reads the YAML, not this
/// struct, so this struct deliberately models only the honoured field: a
/// declared field here would advertise a knob the pipeline never reads.
///
/// Consequence for non-Part-A profiles, which do not run that check: `hf.eps`
/// and friends parse and are then dropped. Adding them back here would not
/// change that; wiring them to the propagator would.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Hf {
    #[serde(default)]
    pub use_high_fidelity: bool,
}

/// The `science.adaptive_events:` block: the objective's event schedule.
///
/// Events an objective call spends per design are `X + Y*n`, where `X` is
/// [`initial`](Self::initial), `Y` is [`step`](Self::step), and `n` is the
/// OBSERVED number of extra rounds the Beta convergence stop asked for. `n` is
/// never configured -- it is whatever the stop reports, per design.
///
/// Absent keys take the compiled sealed values (currently balanced-OA X=8,
/// Y=4 through B500; read them from
/// `CompiledPartAScienceV1::part_a_v1().mf()` rather than from this line), so an
/// omitted block is exactly today's ladder. Canonical Part A additionally
/// REFUSES any non-default value: an overridden schedule changes the science
/// digest, and the receipt writers read the compiled default, so allowing one
/// in a canonical campaign would record a digest for a ladder that never ran.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveEvents {
    /// `X`: events the first stage assesses before the Beta stop is consulted.
    #[serde(default)]
    pub initial: Option<u32>,
    /// `Y`: events each additional round adds.
    #[serde(default)]
    pub step: Option<u32>,
    /// Ladder depth bound. Later stages are unreachable under the canonical
    /// PDF stop but remain available to alternate stop modes; see
    /// `docs/PART_A_RESULTS_MATRIX.md`.
    #[serde(default)]
    pub stages: Option<u32>,
}

/// The `science:` block. Load-bearing, unlike the ignored dissertation
/// sections: every key here resolves into the compiled Part A authority.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Science {
    #[serde(default)]
    pub adaptive_events: AdaptiveEvents,
}

/// Top-level config. LENIENT at the root.
///
/// Sections the workspace does not read (`tc`, `constants`, `solver`,
/// `beta_dist`, `physics`, `transfer`, `postprocess`, `dust`, `ukf`,
/// `canister`, `objectives`, …) are ignored. Part A canonical configs still
/// reject those sections by raw-YAML path (see
/// `part_a_science_override_paths`); validation reads only `config`/`hf`/
/// `optimization`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Config {
    #[serde(default, rename = "config")]
    pub meta: ConfigMeta,
    #[serde(default)]
    pub optimization: Optimization,
    #[serde(default)]
    pub hf: Hf,
    #[serde(default)]
    pub science: Science,
    /// Source-only paths intentionally absent from deserialized authority. This
    /// preserves Part A rejection even though generic config loading remains
    /// lenient for dissertation compatibility.
    #[serde(skip)]
    part_a_science_override_paths: Vec<String>,
}

impl Config {
    /// Read, deserialize, and [`validate`](Config::validate) a config file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, parsed, or validated.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        Self::from_yaml_str(&text).with_context(|| format!("parsing config {}", path.display()))
    }

    /// Deserialize from a YAML string and validate.
    ///
    /// # Errors
    ///
    /// Returns an error if YAML decoding or semantic validation fails.
    pub fn from_yaml_str(text: &str) -> Result<Self> {
        let value: serde_yaml::Value =
            serde_yaml::from_str(text).context("deserializing config YAML")?;
        config_from_yaml_value(value)
    }

    // --- Convenience accessors ---

    #[must_use]
    pub const fn profile(&self) -> Profile {
        self.meta.profile
    }

    /// Resolved matrix seed (see [`Execution::seed`]).
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.optimization.execution.seed()
    }

    /// Resolved adaptive event schedule `(X, Y, stages)`.
    ///
    /// Absent keys fall back to the compiled sealed schedule, so a config with
    /// no `science.adaptive_events:` block resolves to exactly `(24, 8, 60)`.
    ///
    /// # Errors
    ///
    /// Returns an error when a declared component does not fit `usize`.
    pub fn adaptive_event_schedule(&self) -> Result<(usize, usize, usize)> {
        let mf = CompiledPartAScienceV1::part_a_v1().mf();
        let declared = &self.science.adaptive_events;
        let resolve = |value: Option<u32>, sealed: usize, field: &str| -> Result<usize> {
            value.map_or(Ok(sealed), |value| {
                usize::try_from(value)
                    .with_context(|| format!("science.adaptive_events.{field} exceeds usize"))
            })
        };
        Ok((
            resolve(declared.initial, mf.adaptive_initial_events, "initial")?,
            resolve(declared.step, mf.adaptive_event_step, "step")?,
            resolve(declared.stages, mf.adaptive_stage_count, "stages")?,
        ))
    }

    /// The compiled Part A authority this config actually runs.
    ///
    /// Returns the sealed [`CompiledPartAScienceV1::part_a_v1`] when the
    /// schedule is the default one, and an OWNED overridden authority (with its
    /// own science digest) otherwise. Callers that record a science digest must
    /// record the digest of what this returns, never `part_a_v1()` unless they
    /// have proven the schedule is the default.
    ///
    /// # Errors
    ///
    /// Returns an error when the declared schedule is not a buildable ladder.
    pub fn resolved_part_a_science(&self) -> Result<Cow<'static, CompiledPartAScienceV1>> {
        let (initial, step, stages) = self.adaptive_event_schedule()?;
        let sealed = CompiledPartAScienceV1::part_a_v1();
        let mf = sealed.mf();
        if initial == mf.adaptive_initial_events
            && step == mf.adaptive_event_step
            && stages == mf.adaptive_stage_count
        {
            return Ok(Cow::Borrowed(sealed));
        }
        Ok(Cow::Owned(
            CompiledPartAScienceV1::with_adaptive_events(initial, step, stages)
                .context("science.adaptive_events does not form a valid X + Y*n ladder")?,
        ))
    }

    /// Require parsed config semantics for one literal Part A campaign.
    ///
    /// This deliberately seals resolved controls, axes, archive retention,
    /// NFEV authority, and MF+hybrid readiness. It does not inspect config
    /// paths or file hashes; callers retain those provenance responsibilities.
    ///
    /// # Errors
    ///
    /// Returns an error when any parsed control, axis, or compiled-science
    /// authority does not match the requested literal Part A campaign.
    pub fn validate_part_a_semantics(&self, scope: PartACampaignScope) -> Result<()> {
        let science = CompiledPartAScienceV1::part_a_v1();
        let mf = science.mf();
        let k3 = science.k3();
        if let Some(path) = self.part_a_science_override_paths.first() {
            bail!("Part A canonical config forbids compiled science override at {path}");
        }
        expect_part_a_eq("config.version", &self.meta.version, &Some(1))?;
        if self.meta.role.as_deref() != Some("production") {
            bail!(
                "Part A semantic mismatch for config.role: expected production, got {:?}",
                self.meta.role
            );
        }
        let (expected_profile, expected_hf) = match scope {
            PartACampaignScope::Exact36 | PartACampaignScope::Intersect108 => {
                (Profile::Hybrid, true)
            }
            PartACampaignScope::Mf18G500Sensitivity | PartACampaignScope::Mf18G1000Sensitivity => {
                (Profile::Mf, false)
            }
        };
        expect_part_a_eq("config.profile", &self.meta.profile, &expected_profile)?;
        expect_part_a_eq(
            "hf.use_high_fidelity",
            &self.hf.use_high_fidelity,
            &expected_hf,
        )?;

        // The adaptive schedule is declarable but NOT canonically variable. X
        // and Y are folded into the science digest, while the receipt writers
        // (`nd_part_a_evidence::authority`, `receipt`) read the compiled
        // default; a canonical run on an overridden ladder would therefore
        // publish a digest for a ladder it never executed. Sweeps get the
        // override through `resolved_part_a_science` on a non-canonical scope.
        let (initial, step, stages) = self.adaptive_event_schedule()?;
        expect_part_a_eq(
            "science.adaptive_events.initial",
            &initial,
            &mf.adaptive_initial_events,
        )?;
        expect_part_a_eq(
            "science.adaptive_events.step",
            &step,
            &mf.adaptive_event_step,
        )?;
        expect_part_a_eq(
            "science.adaptive_events.stages",
            &stages,
            &mf.adaptive_stage_count,
        )?;

        let execution = &self.optimization.execution;
        reject_part_a_nfev_controls(execution)?;
        self.validate()?;
        let b500_event_count = u32::try_from(mf.b500_event_count)
            .context("compiled Part A B500 event count exceeds u32")?;
        expect_part_a_eq(
            "optimization.execution.events",
            &execution.events,
            &Some(b500_event_count),
        )?;
        expect_part_a_eq(
            "optimization.execution.population_size",
            &execution.population_size,
            &Some(64),
        )?;
        let first_k3_seed = k3
            .seeds
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("compiled Part A K3 seed authority is empty"))?;
        expect_part_a_eq(
            "optimization.execution.seed",
            &execution.seed,
            &Some(first_k3_seed),
        )?;

        let archive = &self.optimization.archive;
        expect_part_a_eq(
            "optimization.archive.enabled",
            &archive.enabled,
            &Some(true),
        )?;
        expect_part_a_eq(
            "optimization.archive.max_size",
            &archive.max_size,
            &Some(k3.archive_max_size),
        )?;
        expect_part_a_eq(
            "optimization.archive.history_max_size",
            &archive.history_max_size,
            &Some(262_144),
        )?;

        let matrix = &self.optimization.matrix;
        if matrix.optimizers.iter().map(String::as_str).ne([
            "nsga2",
            "rnsde",
            "prnsde",
            "eps_nsga2",
            "age_moea2",
            "mopso",
        ]) {
            bail!("Part A semantic mismatch for optimization.matrix.optimizers");
        }
        if matrix
            .constellation_families
            .iter()
            .map(String::as_str)
            .ne(["walker", "dual", "flower"])
        {
            bail!("Part A semantic mismatch for optimization.matrix.constellation_families");
        }
        let expected_fidelities: &[Fidelity] = match scope {
            PartACampaignScope::Exact36 | PartACampaignScope::Intersect108 => {
                &[Fidelity::Mf, Fidelity::Hybrid]
            }
            PartACampaignScope::Mf18G500Sensitivity | PartACampaignScope::Mf18G1000Sensitivity => {
                &[Fidelity::Mf]
            }
        };
        if matrix.fidelity_list != expected_fidelities {
            bail!("Part A semantic mismatch for optimization.matrix.fidelity_list");
        }

        match scope {
            PartACampaignScope::Exact36 => {
                expect_part_a_eq(
                    "optimization.matrix.mode",
                    &matrix.mode,
                    &MatrixMode::Independent,
                )?;
                if matrix.seed_list.as_slice() != [k3.seeds[0]] {
                    bail!("Part A semantic mismatch for optimization.matrix.seed_list");
                }
                expect_part_a_generations(
                    execution.generations,
                    k3.exact36_generations,
                    k3.exact36_measurement_generations,
                )?;
            }
            PartACampaignScope::Mf18G500Sensitivity => {
                expect_part_a_eq(
                    "optimization.matrix.mode",
                    &matrix.mode,
                    &MatrixMode::Independent,
                )?;
                if matrix.seed_list.as_slice() != [k3.seeds[0]] {
                    bail!("Part A semantic mismatch for optimization.matrix.seed_list");
                }
                expect_part_a_generations(
                    execution.generations,
                    k3.mf18g500_sensitivity_generations,
                    k3.mf18g500_sensitivity_measurement_generations,
                )?;
            }
            PartACampaignScope::Mf18G1000Sensitivity => {
                expect_part_a_eq(
                    "optimization.matrix.mode",
                    &matrix.mode,
                    &MatrixMode::Independent,
                )?;
                if matrix.seed_list.as_slice() != [k3.seeds[0]] {
                    bail!("Part A semantic mismatch for optimization.matrix.seed_list");
                }
                expect_part_a_generations(
                    execution.generations,
                    k3.mf18g1000_sensitivity_generations,
                    k3.mf18g1000_sensitivity_measurement_generations,
                )?;
            }
            PartACampaignScope::Intersect108 => {
                expect_part_a_eq(
                    "optimization.matrix.mode",
                    &matrix.mode,
                    &MatrixMode::IntersectK3,
                )?;
                if matrix.seed_list.as_slice() != k3.seeds {
                    bail!("Part A semantic mismatch for optimization.matrix.seed_list");
                }
                // This arm used to enforce against `k3.barriers.last()`, which
                // left `k3.intersect108_generations` read by nothing but the
                // science digest: the two agree at 400 by coincidence, so
                // setting the field to 500 would still have enforced 400. The
                // Exact36 arm above enforces against its own dedicated
                // `exact36_generations`; this is the same shape.
                //
                // Their agreement is now an assertion rather than a
                // coincidence, so a future edit to either one fails loudly
                // instead of silently enforcing the other.
                expect_part_a_eq(
                    "optimization.execution.generations",
                    &execution.generations,
                    &Some(k3.intersect108_generations),
                )?;
                if k3.barriers.last().copied() != Some(k3.intersect108_generations) {
                    bail!(
                        "compiled Part A K3 authority is inconsistent: intersect108_generations \
                         {} must equal the final barrier {:?}",
                        k3.intersect108_generations,
                        k3.barriers.last()
                    );
                }
            }
        }

        validate_part_a_algorithm_semantics(&self.optimization.algorithms)
    }

    /// Enforce the invariants the retired Python `normalize_and_validate_config`
    /// still guards for this lean subset:
    ///
    /// * `config.profile` in `{mf, hybrid}` (enforced by the enum at parse time).
    /// * `hybrid` requires `hf.use_high_fidelity = true` (oracle line 2852).
    /// * `nfev_budget_policy` is not the retired `"generations"` value, and is
    ///   one of `{default, off}` (oracle `nfev_policy.py`).
    /// * `nfev_budget_policy` and `nfev_budget_source` are declared together.
    /// * `nfev_budget_policy = off` forbids an explicit `nfev_budget`.
    /// * a matrix seed is resolvable (always true via [`DEFAULT_SEED`], but the
    ///   check keeps the invariant explicit).
    ///
    /// # Errors
    ///
    /// Returns an error when parsed generic configuration controls violate the
    /// supported execution or optimizer invariants.
    pub fn validate(&self) -> Result<()> {
        let exec = &self.optimization.execution;
        let matrix = &self.optimization.matrix;

        if self.optimization.archive.max_size == Some(0)
            || self.optimization.archive.history_max_size == Some(0)
        {
            bail!("optimization.archive sizes must be positive");
        }
        if let Some(history_max_size) = self.optimization.archive.history_max_size {
            let max_size = self
                .optimization
                .archive
                .max_size
                .unwrap_or(PART_A_ARCHIVE_MAX_SIZE);
            if history_max_size < max_size {
                bail!("optimization.archive.history_max_size must be >= max_size");
            }
        }

        // profile ↔ fidelity coupling.
        if self.meta.profile == Profile::Hybrid && !self.hf.use_high_fidelity {
            bail!("config.profile='hybrid' requires hf.use_high_fidelity=true");
        }
        if matrix.fidelity_list.contains(&Fidelity::Hybrid) && !self.hf.use_high_fidelity {
            bail!("hybrid fidelity matrix axis requires hf.use_high_fidelity=true");
        }
        reject_duplicates("optimizer", &matrix.optimizers)?;
        reject_duplicates("family", &matrix.constellation_families)?;
        reject_duplicates("fidelity", &matrix.fidelity_list)?;
        reject_duplicates("seed", &matrix.seed_list)?;
        let any_axis = !matrix.optimizers.is_empty()
            || !matrix.constellation_families.is_empty()
            || !matrix.fidelity_list.is_empty()
            || !matrix.seed_list.is_empty();
        if any_axis && (matrix.optimizers.is_empty() || matrix.constellation_families.is_empty()) {
            bail!("matrix optimizer and family axes must both be non-empty");
        }
        let k3 = CompiledPartAScienceV1::part_a_v1().k3();
        if matrix.mode == MatrixMode::IntersectK3 && matrix.seed_list.as_slice() != k3.seeds {
            bail!("intersect_k3 matrix requires ordered seeds [41127203, 41127204, 41127205]");
        }

        // nfev policy vocabulary (rejects retired "generations").
        let policy = exec.nfev_policy()?;

        // policy and source must be declared together (both or neither).
        let policy_present = exec.nfev_budget_policy.is_some();
        let source_present = exec
            .nfev_budget_source
            .as_deref()
            .is_some_and(|source| !source.trim().is_empty());
        if policy_present != source_present {
            bail!(
                "optimization.execution.nfev_budget_policy and \
                 optimization.execution.nfev_budget_source must be declared together"
            );
        }

        // policy=off forbids an explicit hard budget.
        if policy == NfevBudgetPolicy::Off && exec.nfev_budget.is_some() {
            bail!(
                "optimization.execution.nfev_budget_policy='off' forbids \
                 optimization.execution.nfev_budget"
            );
        }
        // a seed must be resolvable.
        if self.seed() == 0 {
            bail!("optimization.execution.seed must be a non-zero seed");
        }

        let eps = self.optimization.algorithms.eps_nsga2_resolved()?;
        let age = self.optimization.algorithms.age_moea2_resolved()?;
        let mopso = self.optimization.algorithms.mopso_resolved()?;
        let rnsde = self.optimization.algorithms.rnsde_resolved()?;
        let prnsde = self.optimization.algorithms.prnsde_resolved()?;
        let nsga2 = self.optimization.algorithms.nsga2_resolved()?;
        validate_common_algorithm("eps_nsga2", &eps)?;
        validate_age_algorithm(&age)?;
        validate_mopso(&mopso)?;
        validate_rnsde(&rnsde)?;
        validate_prnsde(&prnsde)?;
        validate_nsga2(&nsga2)?;

        Ok(())
    }
}

fn config_from_yaml_value(value: serde_yaml::Value) -> Result<Config> {
    let science_override_paths = part_a_science_override_paths(&value);
    let mut cfg: Config = serde_yaml::from_value(value).context("deserializing config YAML")?;
    cfg.part_a_science_override_paths = science_override_paths;
    cfg.validate()?;
    Ok(cfg)
}

fn part_a_science_override_paths(value: &serde_yaml::Value) -> Vec<String> {
    let Some(root) = value.as_mapping() else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    for key in [
        "physics",
        "transfer",
        "postprocess",
        "dust",
        "ukf",
        "canister",
        "objectives",
        "beta_dist",
        "solver",
        "catalogue",
        "evaluation",
        "runtime",
    ] {
        if root.contains_key(serde_yaml::Value::String(key.to_owned())) {
            paths.push(key.to_owned());
        }
    }
    if let Some(hf) = yaml_mapping(root, "hf") {
        for key in hf.keys().filter_map(serde_yaml::Value::as_str) {
            if key != "use_high_fidelity" {
                paths.push(format!("hf.{key}"));
            }
        }
    }
    if let Some(optimization) = yaml_mapping(root, "optimization") {
        for key in ["canonical_replay", "science", "physics", "objective"] {
            if optimization.contains_key(serde_yaml::Value::String(key.to_owned())) {
                paths.push(format!("optimization.{key}"));
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

fn yaml_mapping<'a>(parent: &'a serde_yaml::Mapping, key: &str) -> Option<&'a serde_yaml::Mapping> {
    parent
        .get(serde_yaml::Value::String(key.to_owned()))
        .and_then(serde_yaml::Value::as_mapping)
}

/// Campaign depth, or the sealed bounded-measurement horizon a
/// runtime-projection lane runs. Nothing else.
fn expect_part_a_generations(actual: Option<u32>, campaign: u32, measurement: u32) -> Result<()> {
    if actual != Some(campaign) && actual != Some(measurement) {
        bail!(
            "Part A semantic mismatch for optimization.execution.generations: \
             expected Some({campaign}) or measurement Some({measurement}), got {actual:?}"
        );
    }
    Ok(())
}

fn expect_part_a_eq<T>(field: &str, actual: &T, expected: &T) -> Result<()>
where
    T: PartialEq + std::fmt::Debug,
{
    if actual != expected {
        bail!("Part A semantic mismatch for {field}: expected {expected:?}, got {actual:?}");
    }
    Ok(())
}

fn reject_part_a_nfev_controls(execution: &Execution) -> Result<()> {
    if execution.nfev_budget.is_some() {
        bail!("Part A semantic mismatch for optimization.execution.nfev_budget: hard caps are forbidden");
    }
    if execution.nfev_budget_policy.is_some() {
        bail!("Part A semantic mismatch for optimization.execution.nfev_budget_policy: NFEV policy is forbidden");
    }
    if execution.nfev_budget_source.is_some() {
        bail!("Part A semantic mismatch for optimization.execution.nfev_budget_source: NFEV source is forbidden");
    }
    Ok(())
}

/// Construct the compiled NSGA-II controls enforced for every Part A scope.
/// These controls have no environment override.
#[must_use]
pub const fn canonical_part_a_nsga2_controls() -> Nsga2Config {
    Nsga2Config {
        crossover_prob: 0.9,
        mutation_prob: MutationProbability::Scalar(0.2),
        eta_c: 15.0,
        eta_m: 30.0,
        tournament_size: 2,
        reinit_fraction: 0.1,
        reinit_generations: 25,
        design_unique_selection_enabled: false,
        gap_filling_enabled: true,
        gap_perturbation_scale: 0.1,
        gap_offspring_fraction: None,
        random_state: Some(DEFAULT_SEED),
    }
}

fn validate_part_a_algorithm_semantics(algorithms: &Algorithms) -> Result<()> {
    expect_part_a_eq(
        "nsga2 controls",
        &algorithms.nsga2_resolved()?,
        &canonical_part_a_nsga2_controls(),
    )?;
    expect_part_a_eq(
        "rnsde controls",
        &algorithms.rnsde_resolved()?,
        &RnsdeConfig {
            cr: DeCoefficient::Range(0.2, 0.9),
            f: DeCoefficient::Range(0.4, 1.2),
            strategy: "rand1exp".into(),
            reinit_fraction: 0.1,
            reinit_generations: 25,
            enable_cache: false,
            gap_filling_enabled: true,
            gap_perturbation_scale: 0.1,
            gap_offspring_fraction: None,
            diversity_parity_mode: true,
            random_state: Some(DEFAULT_SEED),
        },
    )?;
    expect_part_a_eq(
        "prnsde controls",
        &algorithms.prnsde_resolved()?,
        &PrnsdeConfig {
            cr: DeCoefficient::Range(0.2, 0.9),
            f: DeCoefficient::Range(0.4, 1.2),
            strategy: "rand1exp".into(),
            reinit_fraction: 0.1,
            reinit_generations: 25,
            enable_cache: false,
            gap_filling_enabled: true,
            gap_perturbation_scale: 0.1,
            gap_offspring_fraction: None,
            random_state: Some(DEFAULT_SEED),
            pop_random_fraction: 0.25,
            prde_max_local_refinements: Some(2),
            prde_refine_fraction: None,
            prde_local_max_attempts: 10,
            prde_local_step_scale: 0.03,
            prde_refinement_gain_threshold: 0.0,
            prde_refinement_max_stall: 1,
            prde_refine_with_constraints: false,
        },
    )?;
    expect_part_a_eq(
        "eps_nsga2 controls",
        &algorithms.eps_nsga2_resolved()?,
        &EpsNsga2Config {
            crossover_prob: 0.9,
            mutation_prob: MutationProbability::Range(0.5, 1.0),
            epsilon: 0.001,
            eta_c: 15.0,
            eta_m: 20.0,
            tournament_size: 2,
            reinit_fraction: 0.1,
            reinit_generations: 25,
            gap_perturbation_scale: 0.1,
            gap_offspring_fraction: None,
            random_state: Some(DEFAULT_SEED),
            selection_method: "ideal_distance".into(),
            diversity_epsilon: None,
            stability_window: 5,
            epsilon_decay: None,
            epsilon_min: None,
            init_strategy: "random".into(),
            stability_tol: 0.0001,
            stagnation_restart_frac: 0.0,
            selection_balance: 0.5,
        },
    )?;
    expect_part_a_eq(
        "age_moea2 controls",
        &algorithms.age_moea2_resolved()?,
        &AgeMoea2Config {
            crossover_prob: 0.9,
            mutation_prob: MutationProbability::Range(0.5, 1.0),
            eta_c: 15.0,
            eta_m: 20.0,
            tournament_size: 2,
            reinit_fraction: 0.1,
            reinit_generations: 25,
            gap_filling_enabled: true,
            gap_perturbation_scale: 0.1,
            gap_offspring_fraction: None,
            random_state: Some(DEFAULT_SEED),
        },
    )?;
    expect_part_a_eq(
        "mopso controls",
        &algorithms.mopso_resolved()?,
        &MopsoConfig {
            w_max: 0.9,
            w_min: 0.3,
            c1: 1.8,
            c2: 1.8,
            mutation_prob: 0.25,
            velocity_jitter: 0.05,
            gap_filling_enabled: true,
            reinit_fraction: 0.1,
            reinit_generations: 25,
            gap_perturbation_scale: 0.1,
            gap_offspring_fraction: None,
            random_state: Some(DEFAULT_SEED),
            ensure_diversity: true,
        },
    )
}

fn valid_probability(value: &MutationProbability) -> bool {
    match value {
        MutationProbability::Scalar(x) => x.is_finite() && (0.0..=1.0).contains(x),
        MutationProbability::Range(lo, hi) => {
            lo.is_finite()
                && hi.is_finite()
                && (0.0..=1.0).contains(lo)
                && (0.0..=1.0).contains(hi)
                && lo <= hi
        }
    }
}

fn valid_fraction(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn validate_common_algorithm(name: &str, cfg: &EpsNsga2Config) -> Result<()> {
    if !valid_fraction(cfg.crossover_prob)
        || !valid_probability(&cfg.mutation_prob)
        || !cfg.epsilon.is_finite()
        || cfg.epsilon <= 0.0
        || !cfg.eta_c.is_finite()
        || cfg.eta_c <= 0.0
        || !cfg.eta_m.is_finite()
        || cfg.eta_m <= 0.0
        || cfg.tournament_size < 2
        || !valid_fraction(cfg.reinit_fraction)
        || !cfg.gap_perturbation_scale.is_finite()
        || !(0.0 < cfg.gap_perturbation_scale && cfg.gap_perturbation_scale <= 1.0)
        || cfg
            .gap_offspring_fraction
            .is_some_and(|x| !(0.0 < x && x < 1.0))
        || cfg
            .epsilon_decay
            .is_some_and(|x| !(x.is_finite() && 0.0 < x && x <= 1.0))
        || cfg.epsilon_min.is_some_and(|x| !(x.is_finite() && x > 0.0))
        || !cfg.stability_tol.is_finite()
        || cfg.stability_tol < 0.0
        || !valid_fraction(cfg.stagnation_restart_frac)
        || !valid_fraction(cfg.selection_balance)
    {
        bail!("optimization.algorithms.{name} contains invalid controls");
    }
    Ok(())
}

fn validate_age_algorithm(cfg: &AgeMoea2Config) -> Result<()> {
    if !valid_fraction(cfg.crossover_prob)
        || !valid_probability(&cfg.mutation_prob)
        || !cfg.eta_c.is_finite()
        || cfg.eta_c <= 0.0
        || !cfg.eta_m.is_finite()
        || cfg.eta_m <= 0.0
        || cfg.tournament_size < 2
        || !valid_fraction(cfg.reinit_fraction)
        || !(cfg.gap_perturbation_scale.is_finite()
            && 0.0 < cfg.gap_perturbation_scale
            && cfg.gap_perturbation_scale <= 1.0)
        || cfg
            .gap_offspring_fraction
            .is_some_and(|x| !(0.0 < x && x < 1.0))
    {
        bail!("optimization.algorithms.age_moea2 contains invalid controls");
    }
    Ok(())
}

fn validate_mopso(cfg: &MopsoConfig) -> Result<()> {
    if !cfg.w_max.is_finite()
        || !cfg.w_min.is_finite()
        || cfg.w_max < cfg.w_min
        || cfg.w_min < 0.0
        || !cfg.c1.is_finite()
        || cfg.c1 <= 0.0
        || !cfg.c2.is_finite()
        || cfg.c2 <= 0.0
        || !valid_fraction(cfg.mutation_prob)
        || !cfg.velocity_jitter.is_finite()
        || cfg.velocity_jitter < 0.0
        || !valid_fraction(cfg.reinit_fraction)
        || !cfg.gap_perturbation_scale.is_finite()
        || !(0.0 < cfg.gap_perturbation_scale && cfg.gap_perturbation_scale <= 1.0)
        || cfg
            .gap_offspring_fraction
            .is_some_and(|x| !(0.0 < x && x < 1.0))
    {
        bail!("optimization.algorithms.mopso contains invalid controls");
    }
    Ok(())
}

fn valid_de(value: &DeCoefficient, unit: bool) -> bool {
    let valid = |x: f64| {
        x.is_finite()
            && if unit {
                (0.0..=1.0).contains(&x)
            } else {
                x > 0.0
            }
    };
    match value {
        DeCoefficient::Scalar(x) => valid(*x),
        DeCoefficient::Range(lo, hi) => valid(*lo) && valid(*hi) && lo <= hi,
    }
}

fn validate_rnsde(cfg: &RnsdeConfig) -> Result<()> {
    let valid = valid_de(&cfg.cr, true)
        && valid_de(&cfg.f, false)
        && valid_fraction(cfg.reinit_fraction)
        && cfg.gap_perturbation_scale.is_finite()
        && 0.0 < cfg.gap_perturbation_scale
        && cfg.gap_perturbation_scale <= 1.0
        && cfg
            .gap_offspring_fraction
            .is_none_or(|x| 0.0 < x && x < 1.0);
    if !valid {
        bail!("optimization.algorithms.rnsde contains invalid controls");
    }
    Ok(())
}

fn validate_prnsde(cfg: &PrnsdeConfig) -> Result<()> {
    let core = RnsdeConfig {
        cr: cfg.cr.clone(),
        f: cfg.f.clone(),
        strategy: cfg.strategy.clone(),
        reinit_fraction: cfg.reinit_fraction,
        reinit_generations: cfg.reinit_generations,
        enable_cache: cfg.enable_cache,
        gap_filling_enabled: cfg.gap_filling_enabled,
        gap_perturbation_scale: cfg.gap_perturbation_scale,
        gap_offspring_fraction: cfg.gap_offspring_fraction,
        diversity_parity_mode: false,
        random_state: cfg.random_state,
    };
    validate_rnsde(&core)?;
    if !valid_fraction(cfg.pop_random_fraction)
        || cfg.prde_refine_fraction.is_some_and(|x| !valid_fraction(x))
        || cfg.prde_local_max_attempts == 0
        || !(cfg.prde_local_step_scale.is_finite()
            && 0.0 < cfg.prde_local_step_scale
            && cfg.prde_local_step_scale <= 1.0)
        || !cfg.prde_refinement_gain_threshold.is_finite()
        || cfg.prde_refinement_gain_threshold < 0.0
        || cfg.prde_refinement_max_stall == 0
    {
        bail!("optimization.algorithms.prnsde contains invalid controls");
    }
    Ok(())
}

fn validate_nsga2(cfg: &Nsga2Config) -> Result<()> {
    if !valid_fraction(cfg.crossover_prob)
        || !valid_probability(&cfg.mutation_prob)
        || !cfg.eta_c.is_finite()
        || cfg.eta_c <= 0.0
        || !cfg.eta_m.is_finite()
        || cfg.eta_m <= 0.0
        || cfg.tournament_size < 2
        || !valid_fraction(cfg.reinit_fraction)
        || !(cfg.gap_perturbation_scale.is_finite()
            && 0.0 < cfg.gap_perturbation_scale
            && cfg.gap_perturbation_scale <= 1.0)
        || cfg
            .gap_offspring_fraction
            .is_some_and(|x| !(0.0 < x && x < 1.0))
    {
        bail!("optimization.algorithms.nsga2 contains invalid controls");
    }
    Ok(())
}

fn reject_duplicates<T: Eq + std::hash::Hash>(name: &str, values: &[T]) -> Result<()> {
    let unique: std::collections::HashSet<_> = values.iter().collect();
    if unique.len() != values.len() {
        bail!("matrix {name} axis contains duplicates");
    }
    Ok(())
}
