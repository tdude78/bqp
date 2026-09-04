use std::path::PathBuf;

use common_rs::{require_err, require_ok};
use nd_config::{
    canonical_part_a_nsga2_controls, CompiledPartAScienceV1, Config, PartACampaignScope,
    PartAEarthOrientationConvention, PartAEventAnchorAuthority, PartAGravityAuthority,
    PartASearchModel, PartASharedTargetClaim, PartASharedTargetDrawIntegration,
    PartASharedTargetPositionTreatment, PartATaiEpoch, PartAVerifiedEventAnchor,
    PartAVerifiedEventAnchorInput, PartAVerifiedGravity, PartAVerifiedGravityInput,
    PART_A_SCIENCE_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

#[test]
fn compiled_search_uses_balanced_oa8_with_complete_adaptive_fallback() {
    let science = CompiledPartAScienceV1::part_a_v1();
    let mf = science.mf();

    assert_eq!(
        science.balanced_oa8_bank_indices(),
        &[
            [215, 30, 367, 394, 226, 134, 479, 258],
            [187, 251, 292, 343, 168, 66, 356, 325],
            [98, 127, 261, 497, 121, 54, 376, 432],
        ]
    );
    assert_eq!(
        (
            mf.adaptive_initial_events,
            mf.adaptive_event_step,
            mf.adaptive_stage_count,
        ),
        (8, 4, 124)
    );
    assert_eq!(mf.adaptive_stage_index(8), Some(0));
    assert_eq!(mf.adaptive_stage_index(500), Some(123));
}

#[test]
fn compiled_v3_owns_exact_shared_target_scenario_without_old_ric_anisotropy() {
    let science = CompiledPartAScienceV1::part_a_v1();
    let shared = science.shared_target();

    // "v3" names the scientific scenario; serialized authority schema is 12.
    assert_eq!(PART_A_SCIENCE_SCHEMA_VERSION, 12);
    assert_eq!(
        science.native_hybrid().deterministic_mass_numerical_policy,
        "practical-floor-safe-bracket-v1"
    );
    assert_eq!(
        science.native_hybrid().retained_mass_dynamics,
        "perfectly-inelastic-fixed-area-retention-v1"
    );
    assert_eq!(
        shared.assumption_id,
        "part-a-v3-kappa1-one-grain-100m-optimistic-model-conditioned"
    );
    assert_eq!(
        shared.target_position_sigma_m.to_bits(),
        100.0_f64.to_bits()
    );
    assert_eq!(
        shared.target_position_treatment,
        PartASharedTargetPositionTreatment::AssumedNotObservedIsotropicEncounterBplaneSharedDraw
    );
    assert_eq!(shared.momentum_coupling_kappa.to_bits(), 1.0_f64.to_bits());
    assert_eq!(
        shared.target_integration,
        PartASharedTargetDrawIntegration::RotatedCartesianWithBoundedRadialRefinementAndPolarBelow2V2
    );
    assert_eq!(shared.packet_correlation_grains, 1);
    assert_eq!(shared.target_hit_probability.to_bits(), 0.99_f64.to_bits());
    // Sized by the `cov_min_eig` limit, where a 1 m grain sigma makes the
    // 1.25 m disk the harder release-Pc integral. The singular shared-target
    // C12 solver does not consume these counts.
    assert_eq!(shared.disk_radial_samples, 16);
    assert_eq!(shared.disk_angular_samples, 64);
    assert_eq!(
        shared.disk_radial_samples * shared.disk_angular_samples,
        1_024,
        "release-Pc disk-grid work ratchet widened"
    );
    // Axis 0 is the sharp direction and carries the dense count. Both axes
    // retain the smallest production-proven Cartesian resolution; polar
    // recovery uses the same total node budget.
    assert_eq!(shared.target_radial_samples, 192);
    assert_eq!(shared.target_angular_samples, 32);
    assert_eq!(
        shared.target_radial_samples * shared.target_angular_samples,
        6_144,
        "shared-target Cartesian fine-grid work ratchet widened"
    );
    assert_eq!(
        shared.target_radial_samples * shared.target_angular_samples,
        6_144,
        "shared-target polar fine-grid work ratchet widened"
    );
    assert_eq!(shared.convergence_tolerance.to_bits(), 1.0e-4_f64.to_bits());
    assert_eq!(
        shared.claim,
        PartASharedTargetClaim::ModelConditionedConservativeContactRequirement
    );

    let value = require_ok!(serde_json::to_value(science));
    let native = value
        .get("native_hybrid")
        .and_then(serde_json::Value::as_object)
        .expect("compiled authority must serialize native Hybrid controls");
    assert_eq!(
        native.get("deterministic_mass_numerical_policy"),
        Some(&serde_json::json!("practical-floor-safe-bracket-v1"))
    );
    for stale in [
        "target_pos_sigma_m",
        "target_pos_radial_axis_ratio",
        "target_pos_cross_track_axis_ratio",
    ] {
        assert!(
            !native.contains_key(stale),
            "compiled v3 hash still attests stale anisotropic field {stale}"
        );
    }
    let shared_value = value
        .get("shared_target")
        .and_then(serde_json::Value::as_object)
        .expect("compiled v3 hash must bind shared-target scenario content");
    assert_eq!(
        shared_value
            .get("disk_radial_samples")
            .and_then(serde_json::Value::as_u64),
        Some(16)
    );
    assert_eq!(
        shared_value
            .get("disk_angular_samples")
            .and_then(serde_json::Value::as_u64),
        Some(64)
    );
    assert!(
        !shared_value.contains_key("disk_integration"),
        "singular C12 authority retained the retired disk_integration field"
    );
}

#[test]
fn part_a_search_model_rejects_non_live_strict_hf_wire_value() {
    assert_eq!(
        serde_json::from_value::<PartASearchModel>(serde_json::Value::String("mf_j2".into()))
            .unwrap(),
        PartASearchModel::MfJ2
    );
    assert!(
        serde_json::from_value::<PartASearchModel>(serde_json::Value::String("strict_hf".into()))
            .is_err(),
        "strict-HF is final/H64 execution, not a live B500 search model"
    );
}

fn exact36_yaml() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("configs/part_a_exact36.yaml");
    let result = std::fs::read_to_string(path);
    assert!(result.is_ok(), "exact36 YAML must load: {result:?}");
    result.unwrap_or_default()
}

#[test]
fn canonical_nsga2_controls_are_public_projection_authority() {
    let exact36 = require_ok!(Config::from_yaml_str(&exact36_yaml()));
    let resolved = require_ok!(exact36.optimization.algorithms.nsga2_resolved());

    assert_eq!(resolved, canonical_part_a_nsga2_controls());
}

#[test]
fn compiled_authority_separates_search_and_canonical_pair_counts() {
    let encoded = require_ok!(serde_json::to_value(CompiledPartAScienceV1::part_a_v1()));

    // "v3" names the scientific scenario; serialized authority schema is 5.
    assert_eq!(
        encoded
            .get("schema_version")
            .and_then(serde_json::Value::as_u64),
        Some(u64::from(PART_A_SCIENCE_SCHEMA_VERSION))
    );
    assert_eq!(
        encoded
            .get("mf_transfer")
            .and_then(|transfer| transfer.get("search_pairs_to_verify"))
            .and_then(serde_json::Value::as_u64),
        Some(8)
    );
    assert!(encoded
        .get("hybrid")
        .and_then(|hybrid| hybrid.get("canonical_pairs_to_verify"))
        .is_none());
    assert!(encoded
        .get("mf_transfer")
        .and_then(|transfer| transfer.get("pairs_to_verify"))
        .is_none());
    assert_eq!(
        encoded
            .get("k3")
            .and_then(|k3| k3.get("exact36_measurement_generations"))
            .and_then(serde_json::Value::as_u64),
        Some(4)
    );
    assert_eq!(
        encoded
            .get("k3")
            .and_then(|k3| k3.get("mf18g500_sensitivity_generations"))
            .and_then(serde_json::Value::as_u64),
        Some(500)
    );
}

#[test]
fn adaptive_event_count_formula_accepts_only_compiled_stages() {
    let controls = CompiledPartAScienceV1::part_a_v1().mf();

    assert_eq!(controls.adaptive_stage_index(8), Some(0));
    assert_eq!(controls.adaptive_stage_index(500), Some(123));
    // 7/9 bracket X; 498 is off-cadence just below the final stage; 504 is
    // the first on-cadence count beyond the sealed 500-event bank.
    for count in [4, 7, 9, 498, 501, 504] {
        assert_eq!(
            controls.adaptive_stage_index(count),
            None,
            "count {count} must not be a compiled adaptive stage"
        );
    }
}

#[test]
fn constellation_min_separation_is_a_plain_compiled_control() {
    let encoded = require_ok!(serde_json::to_value(CompiledPartAScienceV1::part_a_v1()));

    assert_eq!(
        encoded.get("constellation"),
        Some(&serde_json::json!({"min_separation_km": 1.0}))
    );
    assert_eq!(
        CompiledPartAScienceV1::part_a_v1()
            .constellation()
            .min_separation_km()
            .to_bits(),
        1.0_f64.to_bits()
    );
}

#[test]
fn constellation_is_no_longer_an_unresolved_production_hybrid_gate() {
    assert!(CompiledPartAScienceV1::part_a_v1()
        .require_production_hybrid_authority()
        .is_ok());
}

#[test]
fn event_lineage_is_verified_and_dir_r6_gravity_is_verified() {
    let science = CompiledPartAScienceV1::part_a_v1();
    let encoded = require_ok!(serde_json::to_value(science));

    assert!(science.event_anchor_authority().require_verified().is_ok());
    assert_eq!(
        encoded.get("event_anchor_authority"),
        Some(&serde_json::json!({
            "status": "verified",
            "source_frame": "GCRS geocentric via astropy 7.2.0 TEME->ITRS->CIRS->GCRS; SGP4-successful source states only; the SGP4-failure equinoctial fallback and zero-vector paths are recorded as an unfalsifiable per-anchor uniformity caveat",
            "time_scale": "UTC (JD input, astropy Time scale=utc)",
            "realization": "IAU 2006/2000A CIO-based via ERFA 2.0.1 (pyerfa 2.0.1.5): gmst82(UT1)+pom00(xp,yp,0); transpose(pom00(xp,yp,sp00(TT))+era00(UT1)); transpose(c2i06a(TT)); finite-difference velocity transforms",
            "leap_second_table_sha256":
                "6f7bc6a25841bc394f82bdfd5d7bb22ffcd4548ee28e9822f2927a909e4f912f",
            "leap_second_table_span": "IERS Leap_Second.dat through Bulletin C 71, expires 2026-12-28; last leap 2017-01-01; TAI-UTC 37 s across the 2021-2022 window; provenance-only, ERFA eraDat is the executing table",
            "tai_minus_utc_source": "ERFA eraDat via pyerfa 2.0.1.5",
            "tt_minus_tai_nanoseconds": 32_184_000_000_i64,
            "earth_orientation": "iers_finals2000a_definitive",
            "reference_epoch_tai": {
                "seconds_since_1958_01_01": 2_031_141_337_i64,
                "nanosecond": 0
            },
            "manifest_sha256":
                "a579175193922652044d26be8ecc86f49b0032c092133a95f05fd06d088a3897"
        }))
    );
    assert_eq!(
        encoded.get("gravity_authority"),
        Some(&serde_json::json!({
            "status": "verified",
            "source_model": "GO_CONS_EGM_GOC_2__20091009T000000_20131020T235959_0201",
            "normalization": "fully_normalized",
            "tide_system": "tide_free",
            "source_gm_km3_s2": 398_600.441_5,
            "source_reference_radius_km": 6_378.136_46,
            "source_max_degree": 300,
            "source_max_order": 300,
            "stored_degree": 15,
            "stored_order": 15,
            "runtime_degree": 5,
            "runtime_order": 5,
            "coefficient_sha256":
                "983f035818399f9cb27f1e8c604cb62b3e72d650aa4cbfadb31b1e7c4fe61f09",
            "manifest_sha256":
                "681bfabf3b43e342f9e7d6b3dfd49a1e05298e8059c55645ce3001fd0c9cae33"
        }))
    );
}

#[test]
fn each_unresolved_provenance_authority_rejects_production_use() {
    assert_eq!(
        require_err!(PartAEventAnchorAuthority::Unresolved.require_verified()).to_string(),
        "Part A production Hybrid authority unresolved: event_anchor_authority"
    );
    assert_eq!(
        require_err!(PartAGravityAuthority::Unresolved.require_verified()).to_string(),
        "Part A production Hybrid authority unresolved: gravity_authority"
    );
}

#[test]
fn verified_event_anchor_schema_carries_complete_time_authority() {
    const A_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let anchor = require_ok!(PartAVerifiedEventAnchor::new(
        PartAVerifiedEventAnchorInput {
            source_frame: "frame",
            time_scale: "scale",
            realization: "realization",
            leap_second_table_sha256: A_SHA,
            leap_second_table_span: "span",
            tai_minus_utc_source: "table",
            tt_minus_tai_nanoseconds: 32_184_000_000,
            earth_orientation: PartAEarthOrientationConvention::IersFinals2000ADefinitive,
            reference_epoch_tai: require_ok!(PartATaiEpoch::new(1, 2)),
            manifest_sha256: B_SHA,
        }
    ));
    let verified = PartAEventAnchorAuthority::Verified(anchor);

    assert_eq!(
        require_ok!(serde_json::to_value(verified)),
        serde_json::json!({
            "status": "verified",
            "source_frame": "frame",
            "time_scale": "scale",
            "realization": "realization",
            "leap_second_table_sha256": A_SHA,
            "leap_second_table_span": "span",
            "tai_minus_utc_source": "table",
            "tt_minus_tai_nanoseconds": 32_184_000_000_i64,
            "earth_orientation": "iers_finals2000a_definitive",
            "reference_epoch_tai": {
                "seconds_since_1958_01_01": 1,
                "nanosecond": 2
            },
            "manifest_sha256": B_SHA
        })
    );
}

/// The Earth-orientation field is the one provenance field
/// `PartAVerifiedEventAnchor::validate` used to skip, so an anchor claiming
/// verified lineage while declaring NO Earth-orientation realization validated
/// clean. Every other field in this input is well-formed, so a rejection here
/// can only come from `earth_orientation` -- and the same input with the
/// canonical convention must still be accepted, or the rule would be rejecting
/// something else.
#[test]
fn verified_event_anchor_rejects_an_unrealized_earth_orientation() {
    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let epoch = require_ok!(PartATaiEpoch::new(0, 0));
    let anchor = |convention| {
        PartAVerifiedEventAnchor::new(PartAVerifiedEventAnchorInput {
            source_frame: "frame",
            time_scale: "TAI",
            realization: "realization",
            leap_second_table_sha256: SHA,
            leap_second_table_span: "span",
            tai_minus_utc_source: "table",
            tt_minus_tai_nanoseconds: 32_184_000_000,
            earth_orientation: convention,
            reference_epoch_tai: epoch,
            manifest_sha256: SHA,
        })
    };

    assert_eq!(
        require_err!(anchor(PartAEarthOrientationConvention::ZeroEop)).to_string(),
        "Part A production Hybrid authority invalid: event Earth-orientation authority: \
         a verified event anchor must declare a realized Earth-orientation convention"
    );
    assert!(anchor(PartAEarthOrientationConvention::IersFinals2000ADefinitive).is_ok());

    // And the SEALED value passes the rule, not merely some realized value.
    let science = CompiledPartAScienceV1::part_a_v1();
    assert!(science.event_anchor_authority().require_verified().is_ok());
    assert_eq!(
        require_ok!(serde_json::to_value(science.event_anchor_authority()))
            .get("earth_orientation")
            .and_then(serde_json::Value::as_str),
        Some("iers_finals2000a_definitive")
    );
}

#[test]
fn malformed_authority_payloads_are_rejected() {
    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    assert!(PartATaiEpoch::new(0, 1_000_000_000).is_err());
    assert!(
        PartAVerifiedEventAnchor::new(PartAVerifiedEventAnchorInput {
            source_frame: "",
            time_scale: "TAI",
            realization: "realization",
            leap_second_table_sha256: SHA,
            leap_second_table_span: "span",
            tai_minus_utc_source: "table",
            tt_minus_tai_nanoseconds: 32_184_000_000,
            earth_orientation: PartAEarthOrientationConvention::IersFinals2000ADefinitive,
            reference_epoch_tai: require_ok!(PartATaiEpoch::new(0, 0)),
            manifest_sha256: SHA,
        })
        .is_err()
    );
    assert!(PartAVerifiedGravity::new(PartAVerifiedGravityInput {
        source_model: "model",
        normalization: "fully",
        tide_system: "tide",
        source_gm_km3_s2: 1.0,
        source_reference_radius_km: 1.0,
        source_max_degree: 4,
        source_max_order: 5,
        stored_degree: 4,
        stored_order: 5,
        runtime_degree: 4,
        runtime_order: 5,
        coefficient_sha256: SHA,
        manifest_sha256: SHA,
    })
    .is_err());
}

#[test]
fn compiled_authority_binds_target_corroboration_and_physical_hv_reference() {
    let encoded = require_ok!(serde_json::to_value(CompiledPartAScienceV1::part_a_v1()));

    assert_eq!(
        encoded
            .get("hybrid")
            .and_then(|hybrid| hybrid.get("target_corroboration_position_km"))
            .and_then(serde_json::Value::as_f64)
            .map(f64::to_bits),
        Some(0.025_f64.to_bits())
    );
    assert_eq!(
        encoded
            .get("hybrid")
            .and_then(|hybrid| hybrid.get("target_corroboration_velocity_km_s"))
            .and_then(serde_json::Value::as_f64)
            .map(f64::to_bits),
        Some(2.0e-5_f64.to_bits())
    );
    assert_eq!(
        encoded
            .get("reporting")
            .and_then(|reporting| reporting.get("physical_hv_reference")),
        Some(&serde_json::json!([2.5, 1000.0]))
    );
}

#[test]
fn compiled_authority_binds_reference_evidence_hashes() {
    let encoded = require_ok!(serde_json::to_value(CompiledPartAScienceV1::part_a_v1()));
    let evidence = encoded.get("reference_evidence");

    assert_eq!(
        evidence.and_then(|evidence| evidence.get("event_authority_manifest_sha256")),
        Some(&serde_json::json!(
            "a579175193922652044d26be8ecc86f49b0032c092133a95f05fd06d088a3897"
        ))
    );
    assert_eq!(
        evidence.and_then(|evidence| evidence.get("event_source_catalogue_sha256")),
        Some(&serde_json::json!(
            "1d6e4e86b64064c2d476b0a8bcad13ae783179a1420fec7eae3fca4ebbb118f8"
        ))
    );
    assert_eq!(
        evidence.and_then(|evidence| evidence.get("event_source_manifest_sha256")),
        Some(&serde_json::json!(
            "9d846ceec76f7cb74279a14fcd44eac4303016d5469ef764b09708a5e0df4477"
        ))
    );
    assert_eq!(
        evidence.and_then(|evidence| evidence.get("event_exact_wrapper_sha256")),
        Some(&serde_json::json!(
            "78b7478db9c83974a74871e163b811fc10fc5cf55113e0e7e603066dd27827d9"
        ))
    );
    assert_eq!(
        evidence.and_then(|evidence| evidence.get("event_kernel_source_sha256")),
        Some(&serde_json::json!(
            "be051ba96900c9f61e2f99183b6b2f1cd3491e21d4adc6444d47426898d3faf7"
        ))
    );
    assert_eq!(
        evidence.and_then(|evidence| evidence.get("gravity_coefficient_sha256")),
        Some(&serde_json::json!(
            "983f035818399f9cb27f1e8c604cb62b3e72d650aa4cbfadb31b1e7c4fe61f09"
        ))
    );
    assert_eq!(
        evidence.and_then(|evidence| evidence.get("gravity_manifest_sha256")),
        Some(&serde_json::json!(
            "681bfabf3b43e342f9e7d6b3dfd49a1e05298e8059c55645ce3001fd0c9cae33"
        ))
    );
}

#[test]
fn part_a_science_hash_accepts_only_its_compiled_authority() {
    let science = CompiledPartAScienceV1::part_a_v1();
    let compact_json = require_ok!(serde_json::to_vec(science));
    let compact_digest: [u8; 32] = Sha256::digest(&compact_json).into();
    let hash = science.sha256_hex();

    assert_eq!(science.sha256(), compact_digest);
    assert_eq!(hash.len(), 64);
    assert_eq!(
        hash,
        // Schema 12 additionally binds the deterministic-mass numerical
        // policy that preserves raw solve/evidence bits while separately
        // commanding max(raw mass, compiled floor).
        // Historical pins and rationale live in Git and the sole Part A
        // closeout ledger, not inside this assertion.
        "766ef38f3df360d13dcacc66db8cbd0d4bc095478142f0c96e290e53fc18a42f"
    );
    assert!(science.matches_sha256(&hash));
    assert!(!science.matches_sha256(&"0".repeat(64)));
}

#[test]
fn compiled_hybrid_session_authority_preserves_captured_recipe() {
    let science = CompiledPartAScienceV1::part_a_v1();
    let hybrid = science.hybrid();
    let shared = science.shared_target();

    assert_eq!(hybrid.session_max_time_s.to_bits(), 0.0_f64.to_bits());
    assert_eq!(hybrid.event_rewind_days.to_bits(), 3.0_f64.to_bits());
    assert_eq!(
        hybrid.session_tof_penalty_weight.to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(hybrid.transfer_am_ratio.to_bits(), 0.01_f64.to_bits());
    assert_eq!(hybrid.transfer_cd.to_bits(), 2.2_f64.to_bits());
    assert_eq!(hybrid.transfer_cr.to_bits(), 1.3_f64.to_bits());
    assert_eq!(
        hybrid.min_practical_dust_mass_kg.to_bits(),
        0.01_f64.to_bits()
    );
    assert_eq!(
        shared.target_position_sigma_m.to_bits(),
        100.0_f64.to_bits()
    );
}

#[test]
fn compiled_mf_transfer_authority_preserves_captured_recipe() {
    let controls = CompiledPartAScienceV1::part_a_v1().mf_transfer();

    assert_eq!(controls.max_time_s.to_bits(), 172_800.0_f64.to_bits());
    assert_eq!(controls.max_phase_dv.to_bits(), 1.25_f64.to_bits());
    assert_eq!(controls.max_transfer_dv.to_bits(), 1.25_f64.to_bits());
    assert_eq!(controls.max_revs, 4);
    assert_eq!(controls.search_pairs_to_verify, 8);
    assert_eq!(controls.tof_penalty_weight.to_bits(), 0.1_f64.to_bits());
    assert_eq!(controls.tof_sample_budget, 256);
    assert!(!controls.coarse_early_stop);
    assert_eq!(controls.j2_max_iterations, 5);
    assert_eq!(controls.local_optimizer_seed, 42);
    assert!(!controls.require_high_fidelity);
    assert!(!controls.force_config_enabled);
    assert!(!controls.gravity_coefficients_enabled);
    assert!(!controls.warm_start_enabled);
    // Default OFF, and this assertion is the guard on a REGRESSION that broke
    // the whole MF lane.
    //
    // The reasoning for turning it on was that the secular-J2 lane requires mean
    // elements while catalogue targets arrive osculating. That is true in
    // isolation and false in context: the catalogue's conjunction anchors ARE
    // the raw secular-J2 images of its start anchors, verified to 2.07e-10 km
    // over all 500 events, so the event's own definition lives in the
    // uncorrected model. Correcting one side aims the solver at a point 1704 km
    // (median) from the conjunction the event is defined by, and every
    // vertical-slice event then fails with `det_mass must be positive and
    // finite, got 0`.
    //
    // Do not flip this without either regenerating the catalogue under the mean
    // convention or extending the HF lane's anchored-differential construction
    // to the MF det-mass path. Full reasoning on the field declaration in
    // `part_a_science.rs`.
    assert!(!controls.target_mean_element_conversion_enabled);
    // The legacy convention stays reachable, explicitly, for port-fidelity tests.
    assert!(
        !CompiledPartAScienceV1::part_a_v1_legacy_target_elements()
            .mf_transfer()
            .target_mean_element_conversion_enabled
    );
}

/// The legacy accessor must PIN the convention, not inherit production's.
///
/// It is byte-identical to production today, because production also holds the
/// legacy value. That identity was undocumented and read as a distinction: the
/// accessor's doc claimed production corrected the convention, so any A/B over
/// the two authorities believed it was comparing two things when it was
/// comparing one thing with itself.
///
/// Stated here in both directions, so the relationship cannot drift silently
/// again. If production ever enables the conversion, the second assertion
/// starts requiring a real difference instead of permitting identity.
#[test]
fn legacy_target_elements_is_pinned_not_inherited() {
    let production = CompiledPartAScienceV1::part_a_v1();
    let legacy = CompiledPartAScienceV1::part_a_v1_legacy_target_elements();

    // Unconditional: the legacy convention is DISABLED, whatever production does.
    assert!(
        !legacy.mf_transfer().target_mean_element_conversion_enabled,
        "the legacy accessor stopped pinning the legacy convention"
    );

    let production_enabled = production
        .mf_transfer()
        .target_mean_element_conversion_enabled;
    let same_authority = production.sha256() == legacy.sha256();
    if production_enabled {
        assert!(
            !same_authority,
            "production enables the conversion, so the legacy authority must              differ from it -- an A/B over the two would otherwise compare one              authority with itself"
        );
    } else {
        assert!(
            same_authority,
            "production and legacy both disable the conversion, so they must be              the same authority; a difference here means some OTHER field was              changed in the legacy clone without being documented"
        );
    }
}

#[test]
fn compiled_mf_lowering_authority_preserves_captured_recipe() {
    let controls = CompiledPartAScienceV1::part_a_v1().mf_lowering();

    assert_eq!(controls.dust_pos_sigma_m.to_bits(), 2.5_f64.to_bits());
    assert_eq!(
        controls.dust_pos_sigma_radial_cross_track_m.to_bits(),
        1.25_f64.to_bits()
    );
    assert_eq!(controls.split_rank, 1);
    // RE-PINNED 2026-08-03: 3 -> 1. See the declaration comment in
    // part_a_science.rs and the "2026-08-03 GMM K=1 evidence under strict HF"
    // section of docs/plans/2026-07-31-part-a-200-generation-fast-hybrid.md.
    assert_eq!(controls.gmm_components, 1);
    assert_eq!(controls.split_tof_short_s.to_bits(), 7200.0_f64.to_bits());
    assert_eq!(controls.split_alpha_scale_cov.to_bits(), 0.6_f64.to_bits());
    assert_eq!(
        controls.split_alpha_scale_cov_low.to_bits(),
        0.35_f64.to_bits()
    );
    assert_eq!(controls.split_jitter.to_bits(), 0.0_f64.to_bits());
    assert_eq!(controls.split_psd_tol.to_bits(), 1.0e-12_f64.to_bits());
    assert_eq!(controls.split_max_psd_iter, 10);
    assert_eq!(controls.split_scale_decay.to_bits(), 0.7_f64.to_bits());
    assert_eq!(
        controls.split_default_alpha_fraction.to_bits(),
        0.6_f64.to_bits()
    );
    assert_eq!(controls.dust_phase_tof_s.to_bits(), 7200.0_f64.to_bits());
}

#[test]
fn canonical_part_a_rejects_decorative_science_yaml() {
    for (suffix, path) in [
        ("\nphysics:\n  dust_hard_limit_kg: 999.0\n", "physics"),
        ("\nukf:\n  alpha: 0.5\n", "ukf"),
        ("\ncanister:\n  tof_fraction: 0.5\n", "canister"),
        ("\nobjectives:\n  hard_dv_km_s: 2.0\n", "objectives"),
        ("\ndust:\n  gmm_components: 4\n", "dust"),
        (
            "\nbeta_dist:\n  hard_fail_success_threshold: 0.4\n",
            "beta_dist",
        ),
    ] {
        let cfg = require_ok!(Config::from_yaml_str(&(exact36_yaml() + suffix)));
        let error = require_err!(cfg.validate_part_a_semantics(PartACampaignScope::Exact36));
        assert!(error.to_string().contains(path), "{error:#}");
    }
    let yaml = exact36_yaml().replacen(
        "use_high_fidelity: true",
        "use_high_fidelity: true\n  dt_max: 1.0",
        1,
    );
    let error = require_err!(require_ok!(Config::from_yaml_str(&yaml))
        .validate_part_a_semantics(PartACampaignScope::Exact36));
    assert!(error.to_string().contains("hf.dt_max"), "{error:#}");
}

/// Binds `assets/part_a/production_authority.env` to the compiled constant.
///
/// The shell release policy reads that file instead of hardcoding a verdict,
/// because it must answer before `nd` exists to be asked. This test is what
/// stops the projection from becoming a second, drifting claim: the status must
/// agree with `require_production_hybrid_authority()`, and every evidence row
/// must equal the compiled reference evidence for the path it names.
#[test]
fn sealed_production_authority_env_matches_compiled_authority() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = repo_root.join("assets/part_a/production_authority.env");
    let text = require_ok!(std::fs::read_to_string(&path));

    let value = |key: &str| -> Option<String> {
        text.lines()
            .find_map(|line| line.strip_prefix(key).map(ToString::to_string))
    };

    let science = CompiledPartAScienceV1::part_a_v1();
    let expected_status = if science.require_production_hybrid_authority().is_ok() {
        "verified"
    } else {
        "blocked"
    };
    assert_eq!(
        value("ND_PART_A_PRODUCTION_AUTHORITY_STATUS=").as_deref(),
        Some(expected_status),
        "sealed projection status must equal the compiled authority verdict"
    );

    let evidence = science.reference_evidence();
    let expected_rows: Vec<(&str, &str)> = vec![
        (
            evidence.event_authority_manifest_sha256,
            "assets/reference/event_authority/manifest.json",
        ),
        (
            evidence.event_source_catalogue_sha256,
            "assets/reference/event_authority/conjunction_events_catalogue_simple_mf.pkl",
        ),
        (
            evidence.event_source_manifest_sha256,
            "assets/reference/event_authority/conjunction_events_catalogue_simple_manifest.json",
        ),
        (
            evidence.event_exact_wrapper_sha256,
            "assets/reference/event_authority/legacy_source/conversions_wrapper_exact.py",
        ),
        (
            evidence.event_kernel_source_sha256,
            "assets/reference/event_authority/legacy_source/satpy_core_lib_authority.rs",
        ),
        (
            evidence.gravity_coefficient_sha256,
            "crates/two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt",
        ),
        (
            evidence.gravity_manifest_sha256,
            "assets/reference/gravity/dir_r6/manifest.json",
        ),
    ];

    let actual_rows: Vec<(&str, &str)> = text
        .lines()
        .filter_map(|line| line.split_once(' '))
        .filter(|(hash, _)| hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()))
        .collect();
    assert_eq!(
        actual_rows, expected_rows,
        "sealed projection rows must equal the compiled reference evidence, in order"
    );

    assert_eq!(
        value("ND_PART_A_REFERENCE_EVIDENCE_COUNT=").as_deref(),
        Some(expected_rows.len().to_string().as_str()),
        "declared row count must equal the compiled reference evidence row count"
    );

    let science_hex = science.sha256_hex();
    assert_eq!(
        value("ND_PART_A_SCIENCE_SHA256=").as_deref(),
        Some(science_hex.as_str()),
        "sealed projection must carry the compiled science hash the build stamp emits"
    );

    // Rows are NOT re-hashed against the tree here. The shell guard hashes
    // every row at runtime, and nd_cli's validate_reference_evidence binds the
    // compiled constant to the tree. This test owns only projection-vs-constant.
}
