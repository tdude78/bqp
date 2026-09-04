//! Integration tests for `nd_config` against the real oracle fixtures.

use std::path::PathBuf;

use common_rs::{require_err, require_ok};
use nd_config::{
    Config, MutationProbability, NfevBudgetPolicy, PartACampaignScope, Profile, DEFAULT_SEED,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn part_a_campaign(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("configs")
        .join(name)
}

fn assert_mf18g500_rejected(config: &Config, field: &str) {
    let error =
        require_err!(config.validate_part_a_semantics(PartACampaignScope::Mf18G500Sensitivity));
    assert!(
        error.to_string().contains(field),
        "expected {field} rejection, got: {error:#}"
    );
}

#[test]
fn mf18g500_sensitivity_config_is_canonical() {
    let config = require_ok!(Config::load(part_a_campaign(
        "part_a_mf18g500_sensitivity.yaml",
    )));

    require_ok!(config.validate_part_a_semantics(PartACampaignScope::Mf18G500Sensitivity));
}

#[test]
fn independent_part_a_scopes_enforce_the_same_nsga2_control_boundary() {
    for (name, scope) in [
        ("part_a_exact36.yaml", PartACampaignScope::Exact36),
        (
            "part_a_mf18g500_sensitivity.yaml",
            PartACampaignScope::Mf18G500Sensitivity,
        ),
    ] {
        let mut config = require_ok!(Config::load(part_a_campaign(name)));
        require_ok!(config.validate_part_a_semantics(scope));
        let canonical_controls = require_ok!(config.optimization.algorithms.nsga2_resolved());

        config
            .optimization
            .algorithms
            .nsga2
            .common
            .reinit_generations = Some(canonical_controls.reinit_generations + 1);
        let error = require_err!(config.validate_part_a_semantics(scope));
        assert!(
            error.to_string().contains("nsga2 controls"),
            "{name} mutation must be rejected by the shared authority, got: {error:#}"
        );
    }
}

#[test]
fn mf18g500_campaign_scope_has_one_external_token() {
    let scope = PartACampaignScope::Mf18G500Sensitivity;
    assert_eq!(serde_json::to_string(&scope).unwrap(), "\"mf18g500\"");
    assert_eq!(
        serde_json::from_str::<PartACampaignScope>("\"mf18g500\"").unwrap(),
        scope
    );
    assert!(serde_json::from_str::<PartACampaignScope>("\"mf18_g500_sensitivity\"").is_err());
    assert!(serde_json::from_str::<PartACampaignScope>("\"mf500\"").is_err());
}

#[test]
fn mf18g500_semantics_reject_hostile_near_misses() {
    let canonical = require_ok!(Config::load(part_a_campaign(
        "part_a_mf18g500_sensitivity.yaml",
    )));

    for generations in [499, 501] {
        let mut config = canonical.clone();
        config.optimization.execution.generations = Some(generations);
        assert_mf18g500_rejected(&config, "optimization.execution.generations");
    }

    let mut config = canonical.clone();
    config.meta.profile = Profile::Hybrid;
    assert_mf18g500_rejected(&config, "config.profile");

    let mut config = canonical.clone();
    config.hf.use_high_fidelity = true;
    assert_mf18g500_rejected(&config, "hf.use_high_fidelity");

    let mut config = canonical.clone();
    config.optimization.matrix.fidelity_list = vec![nd_config::Fidelity::Hybrid];
    assert_mf18g500_rejected(&config, "hybrid fidelity matrix axis");

    let mut config = canonical.clone();
    config.optimization.execution.seed = Some(41_127_204);
    assert_mf18g500_rejected(&config, "optimization.execution.seed");

    let mut config = canonical.clone();
    config.optimization.matrix.seed_list = vec![41_127_204];
    assert_mf18g500_rejected(&config, "optimization.matrix.seed_list");

    let mut config = canonical.clone();
    config.optimization.matrix.mode = nd_config::MatrixMode::IntersectK3;
    assert_mf18g500_rejected(&config, "intersect_k3 matrix requires ordered seeds");

    let mut config = canonical.clone();
    config.optimization.matrix.optimizers.swap(0, 1);
    assert_mf18g500_rejected(&config, "optimization.matrix.optimizers");

    let mut config = canonical.clone();
    config.optimization.matrix.constellation_families.swap(0, 1);
    assert_mf18g500_rejected(&config, "optimization.matrix.constellation_families");

    let mut config = canonical.clone();
    config.optimization.execution.nfev_budget = Some(1);
    assert_mf18g500_rejected(&config, "nfev_budget");

    let mut config = canonical.clone();
    config.optimization.execution.nfev_budget_policy = Some("off".to_owned());
    assert_mf18g500_rejected(&config, "nfev_budget_policy");

    let mut config = canonical;
    config.optimization.execution.nfev_budget_source = Some("test".to_owned());
    assert_mf18g500_rejected(&config, "nfev_budget_source");
}

#[test]
fn part_a_semantic_authority_rejects_mutated_canonical_controls() {
    for (name, scope) in [
        ("part_a_exact36.yaml", PartACampaignScope::Exact36),
        ("part_a_intersect108.yaml", PartACampaignScope::Intersect108),
    ] {
        let canonical = require_ok!(Config::load(part_a_campaign(name)));
        assert!(canonical.validate_part_a_semantics(scope).is_ok());
    }

    for (name, scope, from, to, field) in [
        (
            "part_a_exact36.yaml",
            PartACampaignScope::Exact36,
            "population_size: 64",
            "population_size: 63",
            "population_size",
        ),
        (
            "part_a_exact36.yaml",
            PartACampaignScope::Exact36,
            "generations: 200",
            "generations: 199",
            "optimization.execution.generations",
        ),
        (
            "part_a_exact36.yaml",
            PartACampaignScope::Exact36,
            "eta_m: 30.0",
            "eta_m: 29.0",
            "nsga2 controls",
        ),
        (
            "part_a_exact36.yaml",
            PartACampaignScope::Exact36,
            "diversity_parity_mode: true",
            "diversity_parity_mode: false",
            "rnsde controls",
        ),
        (
            "part_a_exact36.yaml",
            PartACampaignScope::Exact36,
            "pop_random_fraction: 0.25",
            "pop_random_fraction: 0.24",
            "prnsde controls",
        ),
        (
            "part_a_exact36.yaml",
            PartACampaignScope::Exact36,
            "epsilon: 0.001",
            "epsilon: 0.002",
            "eps_nsga2 controls",
        ),
        (
            "part_a_exact36.yaml",
            PartACampaignScope::Exact36,
            "age_moea2:\n      random_state: 41127203\n      crossover_prob: 0.9",
            "age_moea2:\n      random_state: 41127203\n      crossover_prob: 0.8",
            "age_moea2 controls",
        ),
        (
            "part_a_exact36.yaml",
            PartACampaignScope::Exact36,
            "w_max: 0.9",
            "w_max: 0.8",
            "mopso controls",
        ),
        (
            "part_a_exact36.yaml",
            PartACampaignScope::Exact36,
            "optimizers: [nsga2, rnsde, prnsde, eps_nsga2, age_moea2, mopso]",
            "optimizers: [rnsde, nsga2, prnsde, eps_nsga2, age_moea2, mopso]",
            "optimization.matrix.optimizers",
        ),
        (
            "part_a_exact36.yaml",
            PartACampaignScope::Exact36,
            "constellation_families: [walker, dual, flower]",
            "constellation_families: [dual, walker, flower]",
            "optimization.matrix.constellation_families",
        ),
        (
            "part_a_intersect108.yaml",
            PartACampaignScope::Intersect108,
            "generations: 400",
            "generations: 399",
            "optimization.execution.generations",
        ),
        (
            "part_a_intersect108.yaml",
            PartACampaignScope::Intersect108,
            "history_max_size: 262144",
            "history_max_size: 262143",
            "archive.history_max_size",
        ),
        (
            "part_a_intersect108.yaml",
            PartACampaignScope::Intersect108,
            "fidelity_list: [mf, hybrid]",
            "fidelity_list: [hybrid, mf]",
            "fidelity_list",
        ),
    ] {
        let yaml =
            require_ok!(std::fs::read_to_string(part_a_campaign(name))).replacen(from, to, 1);
        let cfg = require_ok!(Config::from_yaml_str(&yaml));
        let error = require_err!(cfg.validate_part_a_semantics(scope));
        assert!(
            error.to_string().contains(field),
            "expected {field} rejection, got: {error:#}"
        );
    }
}

#[test]
fn part_a_semantic_authority_rejects_every_nfev_control() {
    for (name, scope) in [
        ("part_a_exact36.yaml", PartACampaignScope::Exact36),
        ("part_a_intersect108.yaml", PartACampaignScope::Intersect108),
    ] {
        let canonical = require_ok!(Config::load(part_a_campaign(name)));
        let execution = &canonical.optimization.execution;
        assert_eq!(execution.nfev_budget, None, "{name} canonical NFEV cap");
        assert_eq!(
            execution.nfev_budget_policy, None,
            "{name} canonical NFEV policy"
        );
        assert_eq!(
            execution.nfev_budget_source, None,
            "{name} canonical NFEV source"
        );

        let mut with_budget = canonical.clone();
        with_budget.optimization.execution.nfev_budget = Some(1);
        assert!(
            require_err!(with_budget.validate_part_a_semantics(scope))
                .to_string()
                .contains("nfev_budget"),
            "{name} must reject a Part A NFEV cap"
        );

        let mut with_policy = canonical.clone();
        with_policy.optimization.execution.nfev_budget_policy = Some("off".to_owned());
        assert!(
            require_err!(with_policy.validate_part_a_semantics(scope))
                .to_string()
                .contains("nfev_budget_policy"),
            "{name} must reject a Part A NFEV policy"
        );

        let mut with_source = canonical;
        with_source.optimization.execution.nfev_budget_source = Some("test".to_owned());
        assert!(
            require_err!(with_source.validate_part_a_semantics(scope))
                .to_string()
                .contains("nfev_budget_source"),
            "{name} must reject a Part A NFEV source"
        );
    }
}

#[test]
fn part_a_semantic_authority_names_hybrid_control_before_generic_validation() {
    let mut cfg = require_ok!(Config::load(part_a_campaign("part_a_exact36.yaml")));
    cfg.hf.use_high_fidelity = false;

    let error = require_err!(cfg.validate_part_a_semantics(PartACampaignScope::Exact36));
    assert!(
        error
            .to_string()
            .contains("Part A semantic mismatch for hf.use_high_fidelity"),
        "expected semantic HF rejection, got: {error:#}"
    );
}

#[test]
fn loads_canonical_production_config() {
    let cfg = require_ok!(Config::load(fixture("dissertation_production.yaml")));

    // Header identity.
    assert_eq!(cfg.profile(), Profile::Hybrid);
    assert_eq!(cfg.meta.version, Some(1));
    assert_eq!(cfg.meta.role.as_deref(), Some("production"));

    // Execution axes.
    let exec = &cfg.optimization.execution;
    // The fixture pins an explicit seed distinct from DEFAULT_SEED
    // (41_127_203); with seed == DEFAULT_SEED a silent parse failure that
    // falls back to the default would be indistinguishable from a parse.
    assert_eq!(cfg.seed(), 41_128_888);
    assert_ne!(cfg.seed(), DEFAULT_SEED);
    assert_eq!(exec.generations, Some(200));
    assert_eq!(exec.events, Some(500));
    assert_eq!(exec.population_size, Some(64));
    assert_eq!(exec.nfev_budget, Some(2300));
    assert_eq!(require_ok!(exec.nfev_policy()), NfevBudgetPolicy::Default);

    // Matrix axes.
    let m = &cfg.optimization.matrix;
    assert_eq!(m.seed_list.len(), 16);
    assert_eq!(m.seed_list.first(), Some(&41_127_203));
    assert_eq!(
        m.optimizers,
        vec![
            "rnsde",
            "prnsde",
            "nsga2",
            "eps_nsga2",
            "age_moea2",
            "mopso"
        ]
    );
    assert_eq!(m.constellation_families, vec!["walker", "dual", "flower"]);

    // Fidelity knobs. (The lenient physics/transfer/postprocess/dust/ukf/
    // canister/objectives sections are parsed-and-ignored at the root; only
    // `hf.use_high_fidelity` is honoured.)
    assert!(cfg.hf.use_high_fidelity);

    // Optimizer defaults remain YAML authority. Epsilon/AGE inherit defaults;
    // MOPSO only inherits random_state, matching `_optimizer_config.py`.
    let algorithms = &cfg.optimization.algorithms;
    let eps = require_ok!(algorithms.eps_nsga2_resolved());
    assert_eq!(eps.crossover_prob.to_bits(), 0.9_f64.to_bits());
    assert_eq!(eps.mutation_prob, MutationProbability::Range(0.5, 1.0));
    assert_eq!(eps.epsilon.to_bits(), 1e-3_f64.to_bits());
    assert_eq!(eps.eta_c.to_bits(), 15.0_f64.to_bits());
    assert_eq!(eps.eta_m.to_bits(), 20.0_f64.to_bits());
    assert_eq!(eps.tournament_size, 2);
    assert_eq!(eps.reinit_fraction.to_bits(), 0.1_f64.to_bits());
    assert_eq!(eps.reinit_generations, 25);
    assert_eq!(eps.gap_perturbation_scale.to_bits(), 0.1_f64.to_bits());
    assert_eq!(eps.gap_offspring_fraction, None);
    assert_eq!(eps.random_state, Some(41_127_203));

    let age = require_ok!(algorithms.age_moea2_resolved());
    assert_eq!(age.crossover_prob.to_bits(), 0.9_f64.to_bits());
    assert_eq!(age.mutation_prob, MutationProbability::Range(0.5, 1.0));
    assert_eq!(age.eta_c.to_bits(), 15.0_f64.to_bits());
    assert_eq!(age.eta_m.to_bits(), 20.0_f64.to_bits());
    assert_eq!(age.tournament_size, 2);
    assert_eq!(age.reinit_fraction.to_bits(), 0.1_f64.to_bits());
    assert_eq!(age.reinit_generations, 25);
    assert!(age.gap_filling_enabled);
    assert_eq!(age.gap_perturbation_scale.to_bits(), 0.1_f64.to_bits());
    assert_eq!(age.gap_offspring_fraction, None);
    assert_eq!(age.random_state, Some(41_127_203));

    let mopso = require_ok!(algorithms.mopso_resolved());
    assert_eq!(mopso.w_max.to_bits(), 0.9_f64.to_bits());
    assert_eq!(mopso.w_min.to_bits(), 0.3_f64.to_bits());
    assert_eq!(mopso.c1.to_bits(), 1.8_f64.to_bits());
    assert_eq!(mopso.c2.to_bits(), 1.8_f64.to_bits());
    assert_eq!(mopso.mutation_prob.to_bits(), 0.25_f64.to_bits());
    assert_eq!(mopso.velocity_jitter.to_bits(), 0.05_f64.to_bits());
    assert!(mopso.gap_filling_enabled);
    assert_eq!(mopso.reinit_fraction.to_bits(), 0.1_f64.to_bits());
    assert_eq!(mopso.reinit_generations, 25);
    assert_eq!(mopso.gap_perturbation_scale.to_bits(), 0.1_f64.to_bits());
    assert_eq!(mopso.gap_offspring_fraction, None);
    assert_eq!(mopso.random_state, Some(41_127_203));
}

#[test]
fn optimizer_algorithm_overrides_merge_from_defaults() {
    let cfg = Config::from_yaml_str(
        r"
optimization:
  algorithms:
    defaults:
      random_state: 7
      crossover_prob: 0.61
      mutation_prob: [0.2, 0.4]
      epsilon: 0.02
      eta_c: 11.0
      eta_m: 12.0
      tournament_size: 4
      gap_filling_enabled: false
      reinit_fraction: 0.2
      reinit_generations: 9
      gap_perturbation_scale: 0.2
      gap_offspring_fraction: 0.4
      future_default: ignored
    eps_nsga2:
      crossover_prob: 0.71
      mutation_prob: 0.8
      epsilon: 0.007
      tournament_size: 5
      gap_perturbation_scale: 0.3
      future_eps: [ignored]
    age_moea2:
      eta_m: 18.0
      tournament_size: 6
      reinit_fraction: 0.3
      gap_filling_enabled: true
      gap_offspring_fraction: 0.6
    nsga2:
      tournament_size: 7
      gap_filling_enabled: true
    mopso:
      random_state: 19
      w_max: 0.8
      w_min: 0.4
      c1: 1.1
      c2: 2.1
      mutation_prob: 0.15
      velocity_jitter: 0.04
      reinit_fraction: 0.25
      reinit_generations: 8
      gap_filling_enabled: false
      gap_perturbation_scale: 0.25
      gap_offspring_fraction: 0.5
      future_mopso: ignored
    rnsde:
      unknown_rnsde_shape: [not, modeled]
",
    );
    let cfg = require_ok!(cfg);

    let algorithms = &cfg.optimization.algorithms;
    let eps = require_ok!(algorithms.eps_nsga2_resolved());
    assert_eq!(eps.crossover_prob.to_bits(), 0.71_f64.to_bits());
    assert_eq!(eps.mutation_prob, MutationProbability::Scalar(0.8));
    assert_eq!(eps.epsilon.to_bits(), 0.007_f64.to_bits());
    assert_eq!(eps.eta_c.to_bits(), 11.0_f64.to_bits());
    assert_eq!(eps.eta_m.to_bits(), 12.0_f64.to_bits());
    assert_eq!(eps.tournament_size, 5);
    assert_eq!(eps.reinit_fraction.to_bits(), 0.2_f64.to_bits());
    assert_eq!(eps.reinit_generations, 9);
    assert_eq!(eps.gap_perturbation_scale.to_bits(), 0.3_f64.to_bits());
    assert_eq!(eps.gap_offspring_fraction, Some(0.4));
    assert_eq!(eps.random_state, Some(7));

    let age = require_ok!(algorithms.age_moea2_resolved());
    assert_eq!(age.crossover_prob.to_bits(), 0.61_f64.to_bits());
    assert_eq!(age.mutation_prob, MutationProbability::Range(0.2, 0.4));
    assert_eq!(age.eta_c.to_bits(), 11.0_f64.to_bits());
    assert_eq!(age.eta_m.to_bits(), 18.0_f64.to_bits());
    assert_eq!(age.tournament_size, 6);
    assert_eq!(age.reinit_fraction.to_bits(), 0.3_f64.to_bits());
    assert_eq!(age.reinit_generations, 9);
    assert!(age.gap_filling_enabled);
    assert_eq!(age.gap_perturbation_scale.to_bits(), 0.2_f64.to_bits());
    assert_eq!(age.gap_offspring_fraction, Some(0.6));
    assert_eq!(age.random_state, Some(7));

    let nsga2 = require_ok!(algorithms.nsga2_resolved());
    assert_eq!(nsga2.tournament_size, 7);
    assert!(nsga2.gap_filling_enabled);

    let mopso = require_ok!(algorithms.mopso_resolved());
    assert_eq!(mopso.w_max.to_bits(), 0.8_f64.to_bits());
    assert_eq!(mopso.w_min.to_bits(), 0.4_f64.to_bits());
    assert_eq!(mopso.c1.to_bits(), 1.1_f64.to_bits());
    assert_eq!(mopso.c2.to_bits(), 2.1_f64.to_bits());
    assert_eq!(mopso.mutation_prob.to_bits(), 0.15_f64.to_bits());
    assert_eq!(mopso.velocity_jitter.to_bits(), 0.04_f64.to_bits());
    assert_eq!(mopso.reinit_fraction.to_bits(), 0.25_f64.to_bits());
    assert_eq!(mopso.reinit_generations, 8);
    assert!(!mopso.gap_filling_enabled);
    assert_eq!(mopso.gap_perturbation_scale.to_bits(), 0.25_f64.to_bits());
    assert_eq!(mopso.gap_offspring_fraction, Some(0.5));
    assert_eq!(mopso.random_state, Some(19));
}

#[test]
fn rejects_invalid_optimizer_algorithm_controls() {
    for yaml in [
        "optimization:\n  algorithms:\n    eps_nsga2:\n      epsilon: .nan\n",
        "optimization:\n  algorithms:\n    eps_nsga2:\n      mutation_prob: [0.2, 1.1]\n",
        "optimization:\n  algorithms:\n    age_moea2:\n      reinit_fraction: -0.1\n",
        "optimization:\n  algorithms:\n    age_moea2:\n      tournament_size: 1\n",
        "optimization:\n  algorithms:\n    nsga2:\n      gap_filling_enabled: null\n",
        "optimization:\n  algorithms:\n    mopso:\n      gap_perturbation_scale: 0.0\n",
    ] {
        let err = require_err!(Config::from_yaml_str(yaml));
        assert!(
            format!("{err:#}").contains("optimization.algorithms"),
            "unexpected error: {err:#}"
        );
    }
}

#[test]
fn global_archive_is_runtime_authority_and_legacy_rnsde_archive_stays_ignored() {
    let cfg = Config::from_yaml_str(
        r"
optimization:
  archive:
    enabled: true
    max_size: 4096
    history_max_size: 262144
  algorithms:
    rnsde:
      external_archive:
        enabled: true
        max_size: 512
",
    );
    let cfg = require_ok!(cfg);

    assert_eq!(cfg.optimization.archive.enabled, Some(true));
    assert_eq!(cfg.optimization.archive.max_size, Some(4096));
    assert_eq!(cfg.optimization.archive.history_max_size, Some(262_144));

    assert!(Config::from_yaml_str(
        "optimization:\n  archive:\n    max_size: 10\n    history_max_size: 9\n"
    )
    .is_err());
    assert!(
        Config::from_yaml_str("optimization:\n  archive:\n    history_max_size: 4095\n").is_err()
    );
    assert!(Config::from_yaml_str("optimization:\n  archive:\n    enabled: null\n").is_err());
}

#[test]
fn generic_intersect_k3_keeps_optional_nfev_caps() {
    let yaml = |budget, optimizers: &str, families: &str, fidelities: &str| {
        format!(
            r"
optimization:
  execution:
    population_size: 64
    generations: 400
    nfev_budget: {budget}
    nfev_budget_policy: default
    nfev_budget_source: test
  matrix:
    mode: intersect_k3
    seed_list: [41127203, 41127204, 41127205]
    optimizers: [{optimizers}]
    constellation_families: [{families}]
    fidelity_list: [{fidelities}]
hf:
  use_high_fidelity: true
"
        )
    };

    let canonical_optimizers = "nsga2, rnsde, prnsde, eps_nsga2, age_moea2, mopso";
    assert!(Config::from_yaml_str(&yaml(
        30_677,
        canonical_optimizers,
        "walker, dual, flower",
        "mf, hybrid",
    ))
    .is_ok());
    assert!(Config::from_yaml_str(&yaml(
        30_678,
        canonical_optimizers,
        "walker, dual, flower",
        "mf, hybrid",
    ))
    .is_ok());
    assert!(Config::from_yaml_str(&yaml(64, "nsga2", "walker", "mf")).is_ok());
}

#[test]
fn optimizer_no_yaml_defaults_match_python_per_algorithm() {
    let cfg = require_ok!(Config::from_yaml_str("optimization:\n  algorithms: {}\n"));
    let algos = &cfg.optimization.algorithms;
    let eps = require_ok!(algos.eps_nsga2_resolved());
    assert_eq!(eps.crossover_prob.to_bits(), 0.7_f64.to_bits());
    assert_eq!(eps.eta_c.to_bits(), 20.0_f64.to_bits());
    assert_eq!(eps.eta_m.to_bits(), 20.0_f64.to_bits());
    let age = require_ok!(algos.age_moea2_resolved());
    assert_eq!(age.crossover_prob.to_bits(), 0.9_f64.to_bits());
    assert_eq!(age.eta_c.to_bits(), 20.0_f64.to_bits());
    assert_eq!(age.eta_m.to_bits(), 20.0_f64.to_bits());
}

#[test]
fn optimizer_null_overrides_do_not_silently_inherit_or_default() {
    let cfg = require_ok!(Config::from_yaml_str(
        "optimization:\n  algorithms:\n    defaults:\n      random_state: 7\n    mopso:\n      random_state: null\n",
    ));
    let mopso = require_ok!(cfg.optimization.algorithms.mopso_resolved());
    assert_eq!(mopso.random_state, None);
    let err = require_err!(Config::from_yaml_str(
        "optimization:\n  algorithms:\n    eps_nsga2:\n      crossover_prob: null\n",
    ));
    assert!(format!("{err:#}").contains("must not be null"));
}

#[test]
fn optimizer_random_state_precedence_and_null_are_preserved() {
    let cfg = require_ok!(Config::from_yaml_str(
        "optimization:\n  algorithms:\n    defaults:\n      random_state: 11\n    rnsde:\n      random_state: 22\n    nsga2:\n      random_state: 33\n    eps_nsga2:\n      random_state: null\n    mopso:\n      random_state: null\n",
    ));
    let algorithms = &cfg.optimization.algorithms;

    assert_eq!(
        require_ok!(algorithms.rnsde_resolved()).random_state,
        Some(22)
    );
    assert_eq!(
        require_ok!(algorithms.prnsde_resolved()).random_state,
        Some(22)
    );
    assert_eq!(
        require_ok!(algorithms.nsga2_resolved()).random_state,
        Some(33)
    );
    assert_eq!(
        require_ok!(algorithms.eps_nsga2_resolved()).random_state,
        None
    );
    assert_eq!(
        require_ok!(algorithms.age_moea2_resolved()).random_state,
        Some(11)
    );
    assert_eq!(require_ok!(algorithms.mopso_resolved()).random_state, None);

    let cfg = require_ok!(Config::from_yaml_str(
        "optimization:\n  algorithms:\n    defaults:\n      random_state: 11\n    rnsde:\n      random_state: 22\n    prnsde:\n      random_state: null\n",
    ));
    let prnsde = require_ok!(cfg.optimization.algorithms.prnsde_resolved());
    assert_eq!(prnsde.random_state, None);
}

#[test]
fn optimizer_behavior_fields_and_gap_bounds_are_preserved() {
    let cfg = require_ok!(Config::from_yaml_str(
        "optimization:\n  algorithms:\n    eps_nsga2:\n      selection_method: crowding\n      epsilon_decay: 0.8\n      epsilon_min: 0.0001\n      stability_window: 7\n      stability_tol: 0.002\n    mopso:\n      ensure_diversity: false\n",
    ));
    let eps = require_ok!(cfg.optimization.algorithms.eps_nsga2_resolved());
    assert_eq!(eps.selection_method, "crowding");
    assert_eq!(eps.epsilon_decay.map(f64::to_bits), Some(0.8_f64.to_bits()));
    assert_eq!(
        eps.epsilon_min.map(f64::to_bits),
        Some(0.0001_f64.to_bits())
    );
    assert_eq!(eps.stability_window, 7);
    assert_eq!(eps.stability_tol.to_bits(), 0.002_f64.to_bits());
    let mopso = require_ok!(cfg.optimization.algorithms.mopso_resolved());
    assert!(!mopso.ensure_diversity);
    for yaml in [
        "optimization:\n  algorithms:\n    eps_nsga2:\n      gap_perturbation_scale: 1.1\n",
        "optimization:\n  algorithms:\n    mopso:\n      gap_offspring_fraction: 1.0\n",
    ] {
        assert!(Config::from_yaml_str(yaml).is_err());
    }
}

#[test]
fn optimizer_optional_nulls_disable_or_reset_like_python() {
    let cfg = require_ok!(Config::from_yaml_str(
        "optimization:\n  algorithms:\n    defaults:\n      gap_perturbation_scale: 0.3\n      gap_offspring_fraction: 0.4\n      diversity_epsilon: [0.2, 0.3]\n      epsilon_decay: 0.8\n      epsilon_min: 0.01\n    eps_nsga2:\n      gap_perturbation_scale: null\n      gap_offspring_fraction: null\n      diversity_epsilon: 0.5\n      epsilon_decay: null\n      epsilon_min: null\n    mopso:\n      ensure_diversity: null\n",
    ));
    let eps = require_ok!(cfg.optimization.algorithms.eps_nsga2_resolved());
    assert_eq!(eps.gap_perturbation_scale.to_bits(), 0.1_f64.to_bits());
    assert_eq!(eps.gap_offspring_fraction, None);
    assert_eq!(
        eps.diversity_epsilon,
        Some(nd_config::DiversityEpsilon::Scalar(0.5))
    );
    assert_eq!(eps.epsilon_decay, None);
    assert_eq!(eps.epsilon_min, None);
    let mopso = require_ok!(cfg.optimization.algorithms.mopso_resolved());
    assert!(!mopso.ensure_diversity);
    for yaml in [
        "optimization:\n  algorithms:\n    mopso:\n      gap_filling_enabled: null\n",
        "optimization:\n  algorithms:\n    eps_nsga2:\n      epsilon_min: 0.0\n",
    ] {
        assert!(Config::from_yaml_str(yaml).is_err());
    }
}

#[test]
fn de_and_nsga2_controls_merge_inherit_and_validate() {
    let cfg = require_ok!(Config::from_yaml_str(
        "optimization:\n  algorithms:\n    defaults:\n      random_state: 7\n      reinit_fraction: 0.2\n      reinit_generations: 9\n    rnsde:\n      cr: [0.2, 0.8]\n      f: 0.6\n      strategy: best1bin\n      diversity_parity_mode: true\n    prnsde:\n      f: [0.4, 1.1]\n      pop_random_fraction: 0.3\n      prde_max_local_refinements: null\n      prde_refine_fraction: 0.4\n      prde_local_max_attempts: 3\n      prde_local_step_scale: 0.2\n      prde_refinement_gain_threshold: 0.01\n      prde_refinement_max_stall: 2\n      prde_refine_with_constraints: true\n    nsga2:\n      crossover_prob: 0.8\n      mutation_prob: 0.3\n      eta_c: 14\n      eta_m: 22\n      design_unique_selection_enabled: true\n",
    ));
    let a = &cfg.optimization.algorithms;
    let r = require_ok!(a.rnsde_resolved());
    assert_eq!(r.cr, nd_config::DeCoefficient::Range(0.2, 0.8));
    assert_eq!(r.f, nd_config::DeCoefficient::Scalar(0.6));
    assert_eq!(r.strategy, "best1bin");
    assert!(r.diversity_parity_mode);
    assert_eq!(r.random_state, Some(7));
    let p = require_ok!(a.prnsde_resolved());
    assert_eq!(p.f, nd_config::DeCoefficient::Range(0.4, 1.1));
    assert_eq!(p.pop_random_fraction.to_bits(), 0.3_f64.to_bits());
    assert_eq!(p.prde_max_local_refinements, None);
    assert!(p.prde_refine_with_constraints);
    let n = require_ok!(a.nsga2_resolved());
    assert_eq!(n.crossover_prob.to_bits(), 0.8_f64.to_bits());
    assert_eq!(n.mutation_prob, MutationProbability::Scalar(0.3));
    assert!(n.design_unique_selection_enabled);
    for yaml in [
        "optimization:\n  algorithms:\n    rnsde:\n      cr: [0.9, 0.1]\n",
        "optimization:\n  algorithms:\n    prnsde:\n      prde_local_max_attempts: 0\n",
        "optimization:\n  algorithms:\n    nsga2:\n      gap_perturbation_scale: 0.0\n",
    ] {
        assert!(Config::from_yaml_str(yaml).is_err());
    }
}

#[test]
fn de_and_nsga2_nulls_follow_python_merge_rules() {
    let cfg = require_ok!(Config::from_yaml_str(
        "optimization:\n  algorithms:\n    defaults:\n      random_state: 9\n      gap_perturbation_scale: 0.3\n      gap_offspring_fraction: 0.4\n    rnsde:\n      random_state: null\n      gap_perturbation_scale: null\n      gap_offspring_fraction: null\n    nsga2:\n      random_state: null\n      gap_perturbation_scale: null\n      gap_offspring_fraction: null\n",
    ));
    let a = &cfg.optimization.algorithms;
    let r = require_ok!(a.rnsde_resolved());
    assert_eq!(r.random_state, None);
    assert_eq!(r.gap_perturbation_scale.to_bits(), 0.1_f64.to_bits());
    assert_eq!(r.gap_offspring_fraction, None);
    let n = require_ok!(a.nsga2_resolved());
    assert_eq!(n.random_state, None);
    assert_eq!(n.gap_perturbation_scale.to_bits(), 0.1_f64.to_bits());
    assert_eq!(n.gap_offspring_fraction, None);
    assert!(Config::from_yaml_str(
        "optimization:\n  algorithms:\n    prnsde:\n      gap_filling_enabled: null\n"
    )
    .is_err());
}

#[test]
fn prnsde_extras_inherit_defaults_then_rnsde_then_prnsde() {
    let cfg = require_ok!(Config::from_yaml_str(
        "optimization:\n  algorithms:\n    defaults:\n      pop_random_fraction: 0.2\n      prde_local_max_attempts: 4\n      prde_refinement_max_stall: 3\n    rnsde:\n      pop_random_fraction: 0.3\n      prde_local_step_scale: 0.2\n      reinit_generations: 0\n    prnsde:\n      pop_random_fraction: 0.4\n      prde_local_step_scale: null\n    nsga2:\n      reinit_generations: 0\n",
    ));
    let a = &cfg.optimization.algorithms;
    let p = require_ok!(a.prnsde_resolved());
    assert_eq!(p.pop_random_fraction.to_bits(), 0.4_f64.to_bits());
    assert_eq!(p.prde_local_max_attempts, 4);
    assert_eq!(p.prde_refinement_max_stall, 3);
    assert_eq!(p.prde_local_step_scale.to_bits(), 0.1_f64.to_bits());
    assert_eq!(p.reinit_generations, 0);
    assert_eq!(require_ok!(a.nsga2_resolved()).reinit_generations, 0);

    let defaults = require_ok!(Config::from_yaml_str("optimization:\n  algorithms: {}\n"));
    let prnsde = require_ok!(defaults.optimization.algorithms.prnsde_resolved());
    assert_eq!(prnsde.prde_max_local_refinements, Some(2));
}

#[test]
fn loads_mf_overlay_variant() {
    let cfg = require_ok!(Config::load(fixture("mf_j2_base.yaml")));

    assert_eq!(cfg.profile(), Profile::Mf);
    // mf profile does not trip the hybrid⇒HF invariant.
    assert!(!cfg.hf.use_high_fidelity);
    // `cfg.seed()` cannot witness this fixture's seed line. Unlike the one in
    // `loads_canonical_production_config`, this fixture carries exactly
    // DEFAULT_SEED, so the resolved value reads 41_127_203 whether the YAML
    // parsed or was silently dropped. The raw Option separates the two: `Some`
    // is a parse, `None` is the fallback.
    assert_eq!(cfg.optimization.execution.seed, Some(DEFAULT_SEED));
}

#[test]
fn explicit_fidelity_axis_is_not_inferred_from_hf_flag() {
    let cfg = Config::from_yaml_str(
        r"
config:
  profile: mf
optimization:
  matrix:
    optimizers: [rnsde]
    constellation_families: [walker]
    fidelity_list: [mf, hybrid]
hf:
  use_high_fidelity: true
",
    );
    let cfg = require_ok!(cfg);
    assert_eq!(
        cfg.optimization.matrix.fidelity_list,
        vec![nd_config::Fidelity::Mf, nd_config::Fidelity::Hybrid]
    );
}

#[test]
fn hybrid_fidelity_axis_requires_hybrid_readiness() {
    let err = Config::from_yaml_str(
        r"
config:
  profile: mf
optimization:
  matrix:
    optimizers: [rnsde]
    constellation_families: [walker]
    fidelity_list: [mf, hybrid]
hf:
  use_high_fidelity: false
",
    );
    let err = require_err!(err);
    assert!(format!("{err:#}").contains("hybrid fidelity"));
}

#[test]
fn intersect_k3_requires_literal_ordered_part_a_seeds() {
    let prefix = "optimization:\n  matrix:\n    mode: intersect_k3\n    optimizers: [rnsde]\n    constellation_families: [walker]\n    seed_list: ";
    let ok = Config::from_yaml_str(&format!("{prefix}[41127203, 41127204, 41127205]\n"));
    assert!(ok.is_ok());
    for seeds in ["[1, 2, 3]", "[41127205, 41127204, 41127203]"] {
        let err = require_err!(Config::from_yaml_str(&format!("{prefix}{seeds}\n")));
        assert!(format!("{err:#}").contains("41127203, 41127204, 41127205"));
    }
}

#[test]
fn rejects_retired_generations_policy() {
    let yaml = r"
config:
  version: 1
  role: production
  profile: mf
optimization:
  execution:
    seed: 41127203
    nfev_budget_policy: generations
    nfev_budget_source: derived:generations
hf:
  use_high_fidelity: false
";
    let err = require_err!(Config::from_yaml_str(yaml));
    let msg = format!("{err:#}");
    assert!(
        msg.contains("retired"),
        "error should cite the retired policy, got: {msg}"
    );
}

#[test]
fn rejects_hybrid_without_high_fidelity() {
    let yaml = r"
config:
  version: 1
  role: production
  profile: hybrid
hf:
  use_high_fidelity: false
";
    let err = require_err!(Config::from_yaml_str(yaml));
    let msg = format!("{err:#}");
    assert!(
        msg.contains("hf.use_high_fidelity"),
        "error should cite the HF requirement, got: {msg}"
    );
}

#[test]
fn rejects_unknown_profile() {
    let yaml = r"
config:
  profile: quantum
";
    assert!(
        Config::from_yaml_str(yaml).is_err(),
        "unknown profile token must fail to parse"
    );
}
