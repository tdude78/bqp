use std::path::PathBuf;

use serde_json::Value;
use sha2::{Digest, Sha256};

macro_rules! require_some {
    ($value:expr, $message:literal) => {{
        let Some(value) = $value else {
            assert!(false, $message);
            return;
        };
        value
    }};
}

macro_rules! require_ok {
    ($value:expr, $message:literal) => {{
        let Ok(value) = $value else {
            assert!(false, $message);
            return;
        };
        value
    }};
}

macro_rules! required_field {
    ($value:expr, $key:literal) => {{
        require_some!($value.get($key), "missing fixture key")
    }};
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn assert_exact_keys(value: &Value, expected: &[&str]) {
    let object = require_some!(value.as_object(), "fixture node is not an object");
    assert_eq!(
        object.len(),
        expected.len(),
        "unexpected fixture object key"
    );
    for key in expected {
        assert!(object.contains_key(*key), "missing fixture key: {key}");
    }
}

fn assert_f64_hex(value: &Value) {
    let text = require_some!(value.as_str(), "binary64 value is not a string");
    assert_eq!(text.len(), 18, "binary64 hex width");
    let digits = require_some!(text.strip_prefix("0x"), "binary64 value lacks 0x prefix");
    assert!(
        digits
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "binary64 hex must be lowercase"
    );
}

fn assert_hex_array(value: &Value, expected_len: usize) {
    let values = require_some!(value.as_array(), "binary64 vector is not an array");
    assert_eq!(values.len(), expected_len);
    for value in values {
        assert_f64_hex(value);
    }
}

#[test]
fn frame_time_reference_manifest_is_sealed() {
    let root = repo_root();
    let path = root.join("assets/reference/frame_time/manifest.json");
    let manifest_bytes = require_ok!(
        std::fs::read(path),
        "sealed frame/time reference manifest missing"
    );
    assert_eq!(
        sha256(&manifest_bytes),
        "7ac264e7d7ac97f25b1e939ff1928e4d5ecccc4e907e3add14d3ba37fc6e945d"
    );
    let manifest: Value = require_ok!(
        serde_json::from_slice(&manifest_bytes),
        "frame/time manifest is not JSON"
    );

    assert_eq!(
        required_field!(&manifest, "schema"),
        "part_a_frame_time_manifest_v1"
    );
    assert_eq!(
        required_field!(&manifest, "authority_label"),
        "ERFA 2.0.1 / SOFA 20231011-derived"
    );
    let erfa_source = required_field!(&manifest, "erfa_source");
    assert_eq!(
        required_field!(erfa_source, "aggregate_sha256"),
        "0155ec199de5e4d0279ab9655a9b980ac6f731be6c994ba795858e93a1204d1a"
    );
    assert_eq!(required_field!(erfa_source, "regular_member_count"), 255);
    assert_eq!(required_field!(erfa_source, "c_file_count"), 251);
    assert_eq!(required_field!(erfa_source, "h_file_count"), 4);
    let usno = required_field!(&manifest, "usno");
    assert_eq!(required_field!(usno, "finals2000a_total_records"), 19_984);
    assert_eq!(
        required_field!(usno, "finals2000a_first_record_mjd"),
        "41684.00"
    );
    assert_eq!(
        required_field!(usno, "finals2000a_last_record_mjd"),
        "61667.00"
    );
    assert_eq!(
        required_field!(usno, "finals2000a_complete_eop_records"),
        19_645
    );
    assert_eq!(
        required_field!(usno, "finals2000a_complete_eop_first_mjd"),
        "41684.00"
    );
    assert_eq!(
        required_field!(usno, "finals2000a_complete_eop_last_mjd"),
        "61328.00"
    );

    let claimed_semantic_sha = require_some!(
        required_field!(&manifest, "semantic_sha256").as_str(),
        "semantic SHA is not a string"
    );
    assert_eq!(
        claimed_semantic_sha,
        "63240be055032b58d31b42fcb76eddfbeb78d750a6e2aff989d124dabdc5fe05"
    );
    let mut semantic_payload = manifest.clone();
    let semantic_object = require_some!(
        semantic_payload.as_object_mut(),
        "manifest root is not an object"
    );
    semantic_object.remove("semantic_sha256");
    let canonical_payload = require_ok!(
        serde_json::to_vec(&semantic_payload),
        "semantic payload does not serialize"
    );
    let mut semantic_hasher = Sha256::new();
    semantic_hasher.update(b"PART_A_FRAME_TIME_MANIFEST_V1");
    semantic_hasher.update([0]);
    semantic_hasher.update(
        require_ok!(
            u64::try_from(canonical_payload.len()),
            "canonical payload length does not fit u64"
        )
        .to_be_bytes(),
    );
    semantic_hasher.update(&canonical_payload);
    assert_eq!(
        format!("{:x}", semantic_hasher.finalize()),
        claimed_semantic_sha
    );

    let expected_payloads = [
        (
            "assets/reference/frame_time/README.md",
            1778,
            "ddc7f18a00a0411707ab710e9bc94e033ec2de2ecebf5b5f3f48dc05aacc6771",
        ),
        (
            "assets/reference/frame_time/USNO_PUBLIC_RELEASE_RECEIPT.md",
            1492,
            "fd0e4a61c05c9f8117abf4aac036f5f6e61755bdc9b6558076cbd608f6a7deaf",
        ),
        (
            "assets/reference/frame_time/erfa-2.0.1-sofa-20231011-source.tar.gz",
            323_547,
            "9ad0e5a4eba41ee1e481d01475ba0bc2546df5303c495387cd2771381cc200f4",
        ),
        (
            "assets/reference/frame_time/ERFA_LICENSE",
            2651,
            "b1858f9a263f22c438a455a32945da51a31a0ae25a21055da13bb7ed57cc3b51",
        ),
        (
            "assets/reference/frame_time/ERFA_CONFIGURE_AC",
            941,
            "58bce94b6e33f4ad557a42d92cc2d1f2997175eb401cbf0f17e627459c3ba94c",
        ),
        (
            "assets/reference/frame_time/ERFA_README.rst",
            6878,
            "3c25fac03be6f4921736f3d8d16575a90e775da07c58e8d58a25ed7622fa2b32",
        ),
        (
            "assets/reference/frame_time/PYERFA_PKG_INFO",
            5743,
            "49b55a0cc45bf378b3be1010dff553e619116d4589a8dfc2299e96faa8197073",
        ),
        (
            "assets/reference/frame_time/tai-utc.dat",
            3321,
            "3524e1ae34d67e858873a89e59983bbc5bd100221da898e796c1b36036a310c3",
        ),
        (
            "assets/reference/frame_time/finals2000A.all",
            3_756_992,
            "f707ea5031a467f1a3b2f0645fac2f627095ed0cb41d34c515b495cb81a5a25d",
        ),
        (
            "assets/reference/frame_time/ReadMe.finals2000A",
            1830,
            "3efb2c610012360391aed5c057489ff74539b78494a66b7f9b342d1843db9254",
        ),
    ];
    let payloads = require_some!(
        required_field!(&manifest, "payloads").as_array(),
        "payloads is not an array"
    );
    assert_eq!(payloads.len(), expected_payloads.len());
    for (payload, (expected_path, expected_size, expected_sha)) in
        payloads.iter().zip(expected_payloads)
    {
        assert_eq!(required_field!(payload, "path"), expected_path);
        assert_eq!(required_field!(payload, "size_bytes"), expected_size);
        assert_eq!(required_field!(payload, "sha256"), expected_sha);
        let bytes = require_ok!(
            std::fs::read(root.join(expected_path)),
            "sealed payload missing"
        );
        assert_eq!(
            require_ok!(
                u64::try_from(bytes.len()),
                "payload length does not fit u64"
            ),
            expected_size
        );
        assert_eq!(sha256(&bytes), expected_sha);
    }

    let finals = require_ok!(
        std::fs::read_to_string(root.join("assets/reference/frame_time/finals2000A.all")),
        "finals2000A is not readable ASCII"
    );
    let mut complete_count = 0_u64;
    let mut first_complete_mjd = None;
    let mut last_complete_mjd = None;
    let mut incomplete_seen = false;
    for line in finals.lines() {
        let bytes = line.as_bytes();
        assert!(bytes.len() >= 125, "dated record is too short");
        let field_present = |start: usize, end: usize| {
            bytes
                .get(start..end)
                .is_some_and(|field| field.iter().any(|byte| !byte.is_ascii_whitespace()))
        };
        let complete = [(18, 27), (37, 46), (58, 68), (97, 106), (116, 125)]
            .into_iter()
            .all(|(start, end)| field_present(start, end));
        if complete {
            assert!(
                !incomplete_seen,
                "complete EOP row after placeholder suffix"
            );
            let mjd_bytes = require_some!(bytes.get(7..15), "MJD field is out of bounds");
            let mjd = require_ok!(std::str::from_utf8(mjd_bytes), "MJD is not ASCII").trim();
            first_complete_mjd.get_or_insert_with(|| mjd.to_owned());
            last_complete_mjd = Some(mjd.to_owned());
            complete_count += 1;
        } else {
            incomplete_seen = true;
        }
    }
    assert_eq!(finals.lines().count(), 19_984);
    assert_eq!(complete_count, 19_645);
    assert_eq!(first_complete_mjd.as_deref(), Some("41684.00"));
    assert_eq!(last_complete_mjd.as_deref(), Some("61328.00"));
}

fn assert_fixture_provenance(fixture: &Value) {
    let provenance = required_field!(fixture, "provenance");
    assert_exact_keys(
        provenance,
        &[
            "generator_source_sha256",
            "orchestration_script_sha256",
            "frame_time_manifest_sha256",
            "frame_time_manifest_semantic_sha256",
            "erfa_source_archive_sha256",
            "erfa_source_aggregate_sha256",
            "finals2000a_sha256",
            "tai_utc_sha256",
            "erfa_version",
            "sofa_version",
        ],
    );
    for (key, expected) in [
        (
            "generator_source_sha256",
            "1a4a17cf84f04d2606823989ada38e3fa538d4d812e6f6c452e75d590414c2d7",
        ),
        (
            "orchestration_script_sha256",
            "c7253df5c814228af74296f8f2f704d445da8b7ee466c0be4d1519b93e2304f8",
        ),
        (
            "frame_time_manifest_sha256",
            "7ac264e7d7ac97f25b1e939ff1928e4d5ecccc4e907e3add14d3ba37fc6e945d",
        ),
        (
            "frame_time_manifest_semantic_sha256",
            "63240be055032b58d31b42fcb76eddfbeb78d750a6e2aff989d124dabdc5fe05",
        ),
        (
            "erfa_source_archive_sha256",
            "9ad0e5a4eba41ee1e481d01475ba0bc2546df5303c495387cd2771381cc200f4",
        ),
        (
            "erfa_source_aggregate_sha256",
            "0155ec199de5e4d0279ab9655a9b980ac6f731be6c994ba795858e93a1204d1a",
        ),
        (
            "finals2000a_sha256",
            "f707ea5031a467f1a3b2f0645fac2f627095ed0cb41d34c515b495cb81a5a25d",
        ),
        (
            "tai_utc_sha256",
            "3524e1ae34d67e858873a89e59983bbc5bd100221da898e796c1b36036a310c3",
        ),
    ] {
        let actual = require_some!(provenance.get(key), "missing provenance key");
        assert_eq!(actual, expected);
    }
}

fn assert_fixture_metadata(fixture: &Value) {
    let toolchain = required_field!(fixture, "toolchain");
    assert_exact_keys(
        toolchain,
        &[
            "architecture",
            "sw_vers",
            "clang_path",
            "clang_sha256",
            "sdk_path",
            "libsystem_tbd_sha256",
            "dyld_cache_main_sha256",
            "dyld_cache_atlas_sha256",
            "dyld_cache_map_sha256",
            "compile_argv",
            "otool_l",
            "generator_binary_sha256",
        ],
    );
    assert_eq!(required_field!(toolchain, "architecture"), "arm64");
    assert_eq!(
        required_field!(toolchain, "sw_vers"),
        "macOS 27.0 (26A5378n)"
    );
    assert_eq!(
        required_field!(toolchain, "generator_binary_sha256"),
        "315c8fdec15d7610bd85034fb1a5278ba6842e39f7e93978b303e0a49380ed18"
    );
    assert_eq!(
        required_field!(toolchain, "otool_l"),
        "/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1359.0.0)"
    );

    assert_exact_keys(
        required_field!(fixture, "canonicalization"),
        &[
            "semantic_domain",
            "numeric_encoding",
            "matrix_layout",
            "case_order",
            "json",
        ],
    );
    assert_exact_keys(
        required_field!(fixture, "time_and_frame"),
        &[
            "input_frame",
            "output_frame",
            "rotation",
            "anchor_time_scale",
            "tt",
            "real_eop",
            "zero_eop",
            "derivatives",
        ],
    );
    assert_exact_keys(
        required_field!(fixture, "units"),
        &["r", "v", "a", "R", "Rdot", "Rddot"],
    );
    let refinement = required_field!(fixture, "refinement");
    let refinement_keys = [
        "h_fixture_s",
        "h_comparison_s",
        "max_r_vs_erfa",
        "max_rdot_difference_s1",
        "max_rddot_difference_s2",
    ];
    assert_exact_keys(refinement, &refinement_keys);
    for key in refinement_keys {
        let value = require_some!(refinement.get(key), "missing refinement key");
        assert_f64_hex(value);
    }
}

fn assert_fixture_cases(fixture: &Value) {
    let epochs = [
        "2000-01-01T12:00:00",
        "2016-12-31T23:59:59",
        "2016-12-31T23:59:60",
        "2017-01-01T00:00:00",
        "2024-01-01T00:00:00",
    ];
    let cases = require_some!(
        required_field!(fixture, "cases").as_array(),
        "cases is not an array"
    );
    assert_eq!(cases.len(), 20);
    for (index, case) in cases.iter().enumerate() {
        assert_exact_keys(
            case,
            &[
                "id",
                "epoch_utc",
                "eop_policy",
                "state_index",
                "r_gcrs_km",
                "v_gcrs_km_s",
                "a_gcrs_km_s2",
                "R_gcrs_to_itrs",
                "Rdot_s1",
                "Rddot_s2",
                "r_itrs_km",
                "v_itrs_km_s",
                "a_itrs_km_s2",
            ],
        );
        let epoch = epochs.get(index / 4).copied().unwrap_or("missing epoch");
        assert_ne!(epoch, "missing epoch", "case index {index} has no epoch");
        let within_epoch = index % 4;
        let eop = if within_epoch < 2 {
            "zero_eop"
        } else {
            "real_eop"
        };
        let state_index = within_epoch % 2;
        assert_eq!(required_field!(case, "epoch_utc"), epoch);
        assert_eq!(required_field!(case, "eop_policy"), eop);
        assert_eq!(required_field!(case, "state_index"), state_index);
        assert_eq!(
            required_field!(case, "id"),
            &format!("{epoch}_{eop}_state_{state_index}")
        );
        for key in [
            "r_gcrs_km",
            "v_gcrs_km_s",
            "a_gcrs_km_s2",
            "r_itrs_km",
            "v_itrs_km_s",
            "a_itrs_km_s2",
        ] {
            let value = require_some!(case.get(key), "missing vector key");
            assert_hex_array(value, 3);
        }
        for key in ["R_gcrs_to_itrs", "Rdot_s1", "Rddot_s2"] {
            let value = require_some!(case.get(key), "missing matrix key");
            assert_hex_array(value, 9);
        }
    }
}

#[test]
fn erfa_frame_time_fixture_schema_and_semantics_are_sealed() {
    let root = repo_root();
    let fixture_path =
        root.join("crates/satpy_core/tests/data/erfa_sofa_derived_frame_time_v1.json");
    let bytes = require_ok!(
        std::fs::read(fixture_path),
        "accepted frame/time fixture missing"
    );
    assert_eq!(
        sha256(&bytes),
        "d6b17e1a86656de1fbd838c5706f1035bdd0c5de3f86751434cbad473ecb25dd"
    );
    assert_eq!(bytes.last(), Some(&b'\n'));
    assert_eq!(
        bytes.split(|byte| *byte == b'\n').count().saturating_sub(1),
        1
    );

    let fixture: Value = require_ok!(
        serde_json::from_slice(&bytes),
        "frame/time fixture is not JSON"
    );
    assert_exact_keys(
        &fixture,
        &[
            "schema",
            "semantic_sha256",
            "authority_label",
            "claim_scope",
            "provenance",
            "toolchain",
            "canonicalization",
            "time_and_frame",
            "units",
            "refinement",
            "cases",
        ],
    );
    assert_eq!(
        required_field!(&fixture, "schema"),
        "part_a_erfa_sofa_derived_frame_time_v1"
    );
    assert_eq!(
        required_field!(&fixture, "semantic_sha256"),
        "4dbc1effbbb412bf17a8be991981dfdb709523c6e0a091085fcf18d9581747c8"
    );
    assert!(fixture.get("status").is_none());
    assert!(fixture.get("accepted").is_none());

    let semantic_field = b",\"semantic_sha256\":\"4dbc1effbbb412bf17a8be991981dfdb709523c6e0a091085fcf18d9581747c8\"";
    let semantic_start = bytes
        .windows(semantic_field.len())
        .position(|window| window == semantic_field)
        .unwrap_or(bytes.len());
    assert!(
        semantic_start < bytes.len(),
        "semantic field lacks canonical root position"
    );
    let payload_capacity = require_some!(
        bytes
            .len()
            .checked_sub(semantic_field.len())
            .and_then(|len| len.checked_sub(1)),
        "semantic field is longer than fixture"
    );
    let semantic_end = require_some!(
        semantic_start.checked_add(semantic_field.len()),
        "semantic field end overflow"
    );
    let final_byte = require_some!(bytes.len().checked_sub(1), "fixture is empty");
    let prefix = require_some!(
        bytes.get(..semantic_start),
        "semantic prefix is out of bounds"
    );
    let suffix = require_some!(
        bytes.get(semantic_end..final_byte),
        "semantic suffix is out of bounds"
    );
    let mut semantic_payload = Vec::with_capacity(payload_capacity);
    semantic_payload.extend_from_slice(prefix);
    semantic_payload.extend_from_slice(suffix);
    assert_eq!(semantic_payload.len(), 27_485);
    let mut semantic_hasher = Sha256::new();
    semantic_hasher.update(b"PART_A_FRAME_TIME_ORACLE_V1");
    semantic_hasher.update([0]);
    semantic_hasher.update(
        require_ok!(
            u64::try_from(semantic_payload.len()),
            "semantic payload length does not fit u64"
        )
        .to_be_bytes(),
    );
    semantic_hasher.update(&semantic_payload);
    assert_eq!(
        format!("{:x}", semantic_hasher.finalize()),
        "4dbc1effbbb412bf17a8be991981dfdb709523c6e0a091085fcf18d9581747c8"
    );

    assert_fixture_provenance(&fixture);
    assert_fixture_metadata(&fixture);
    assert_fixture_cases(&fixture);
}

// --- Task 5A production comparator ------------------------------------------

use satpy_core::frame_time::authority::{frame_authority, tai_seconds_from_utc_jd};
use satpy_core::frame_time::timescale::dtf2d_utc;
use satpy_core::frame_time::{compute_all_cases, EopPolicy};

fn hex_f64(value: &Value) -> Option<f64> {
    let text = value.as_str()?;
    let digits = text.strip_prefix("0x")?;
    let bits = u64::from_str_radix(digits, 16).ok()?;
    Some(f64::from_bits(bits))
}

fn hex_vec(value: &Value) -> Option<[f64; 3]> {
    let [x, y, z] = value.as_array()?.as_slice() else {
        return None;
    };
    Some([hex_f64(x)?, hex_f64(y)?, hex_f64(z)?])
}

fn hex_mat(value: &Value) -> Option<[[f64; 3]; 3]> {
    let [m00, m01, m02, m10, m11, m12, m20, m21, m22] = value.as_array()?.as_slice() else {
        return None;
    };
    Some([
        [hex_f64(m00)?, hex_f64(m01)?, hex_f64(m02)?],
        [hex_f64(m10)?, hex_f64(m11)?, hex_f64(m12)?],
        [hex_f64(m20)?, hex_f64(m21)?, hex_f64(m22)?],
    ])
}

#[test]
fn cached_frame_angular_velocity_matches_external_rdot_sign_and_rate() {
    assert_eq!(
        frame_authority().authority_sha256(),
        [
            0xc6, 0x26, 0x1e, 0xc7, 0x0a, 0x03, 0x75, 0x31, 0x9f, 0x9c, 0x2b, 0x1a, 0xe1, 0x2f,
            0xf3, 0x31, 0x6d, 0xb3, 0x0e, 0x9f, 0x90, 0x46, 0xb0, 0xe3, 0x4f, 0xf3, 0xf4, 0xe6,
            0x9c, 0xb8, 0x60, 0x69,
        ],
        "frame-chain v3 authority digest"
    );
    let bytes = require_ok!(
        std::fs::read(
            repo_root().join("crates/satpy_core/tests/data/erfa_sofa_derived_frame_time_v1.json"),
        ),
        "accepted frame/time fixture missing"
    );
    let fixture: Value = require_ok!(
        serde_json::from_slice(&bytes),
        "frame/time fixture is not JSON"
    );
    let cases = require_some!(
        required_field!(&fixture, "cases").as_array(),
        "cases is not an array"
    );
    let case = require_some!(
        cases.iter().find(|case| {
            case.get("id").and_then(Value::as_str) == Some("2024-01-01T00:00:00_real_eop_state_0")
        }),
        "2024 real-EOP oracle case missing"
    );
    let r = require_some!(
        hex_mat(required_field!(case, "R_gcrs_to_itrs")),
        "invalid target matrix"
    );
    let rdot = require_some!(
        hex_mat(required_field!(case, "Rdot_s1")),
        "invalid target rate"
    );

    // Passive GCRS->ITRS convention: R^T Rdot = -[omega_gcrs x].
    let [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]] = r;
    let [[d00, d01, d02], [d10, d11, d12], [d20, d21, d22]] = rdot;
    let body_rate_12 = r01 * d02 + r11 * d12 + r21 * d22;
    let body_rate_21 = r02 * d01 + r12 * d11 + r22 * d21;
    let body_rate_20 = r02 * d00 + r12 * d10 + r22 * d20;
    let body_rate_02 = r00 * d02 + r10 * d12 + r20 * d22;
    let body_rate_01 = r00 * d01 + r10 * d11 + r20 * d21;
    let body_rate_10 = r01 * d00 + r11 * d10 + r21 * d20;
    let expected = [
        0.5 * (body_rate_12 - body_rate_21),
        0.5 * (body_rate_20 - body_rate_02),
        0.5 * (body_rate_01 - body_rate_10),
    ];

    let (status, utc1, utc2) = dtf2d_utc(2024, 1, 1, 0, 0, 0.0);
    assert_eq!(status, 0);
    let tai_s = require_ok!(
        tai_seconds_from_utc_jd(utc1, utc2),
        "oracle epoch is outside sealed span"
    );
    let actual = require_ok!(
        frame_authority().rotation_at(tai_s),
        "cached rotation does not resolve"
    )
    .itrs_angular_velocity_gcrs;

    for (axis, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2.0e-13,
            "axis {axis}: cached omega={actual:.17e}, oracle omega={expected:.17e}"
        );
    }
}

#[test]
fn production_frame_time_matches_external_fixture() {
    // Task 5A acceptance bounds (no widening, ever).
    //
    // These three are the PRIMITIVE quantities: they compare the GCRS->ITRS
    // rotation and its two time derivatives against the sealed fixture
    // element-wise. Nothing is loosened here and nothing may be.
    const R_BOUND: f64 = 5e-13;
    const RDOT_BOUND: f64 = 1e-13;
    const RDDOT_BOUND: f64 = 1e-12;
    // DR_BOUND is DEAD by the same criterion applied to DV/DA below:
    // `dr <= R_BOUND * ||r||_1 = 5e-13 * 8000 = 4e-9 < 1e-8`, so it cannot fire
    // unless `R_BOUND` fires first. It is retained deliberately and labelled as
    // such rather than quietly kept on a different rationale: it costs nothing
    // and it still catches a GROSS composition error in `transform_state` (a
    // transposed or mis-indexed matrix product), which is a different failure
    // from the fine-grained one `R_BOUND` covers. It is a smoke test, not a
    // tolerance.
    const DR_BOUND: f64 = 1e-8;
    //
    // DV_BOUND and DA_BOUND are the STENCIL ONE-ULP CEILING. They are derived
    // from the comparator's own error propagation, not fitted to an observation
    // and not scaled from the matrix bounds. Both of those alternatives were
    // tried and rejected; the reasoning is recorded here so it is not redone.
    //
    // WHY NOT SCALED FROM THE MATRIX BOUNDS. `transform_state`
    // (chain.rs:247-252) computes
    //     da_i = SUM_j [ dR[i][j]*a_j + 2*dRdot[i][j]*v_j + dRddot[i][j]*r_j ]
    // so a state-level bound derived from the matrix bounds,
    //     R_BOUND*||a||_1 + 2*RDOT_BOUND*||v||_1 + RDDOT_BOUND*||r||_1
    //     = 5e-13*0.0085 + 2*1e-13*8.5 + 1e-12*8000 = 8.0e-9
    // (L1, because the matrix bounds are element-wise maxima; ||r||_1 = 8000 for
    // POSITIONS[1], chain.rs:88) cannot fire unless a matrix assertion fires
    // first, and the `RDDOT_BOUND*||r||_1` term alone is 99.979% of it. Dead.
    //
    // WHY NOT THE ORIGINAL 1e-12 EITHER. It asserted sub-ulp bit-reproducibility
    // of a transcendental series across optimisation levels. `xys06a`'s binary64
    // IAU 2006/2000A series returns `y` and `s` one ulp apart between opt-levels
    // with every input bit-identical, at a ~3% base rate over arbitrary TT
    // (task #29). One ulp is below this comparator's own noise floor, so 1e-12
    // is not a tolerance, it is a codegen assumption measured false ~3% of the
    // time. Production consumes the rotation directly and never
    // second-differences it: the measured production cost of that same one ulp
    // is 1.55 nm (RC2I over 400 epochs, 16/400 rows differing, max element error
    // 2.2204e-16; RPOM and ERA bit-identical and orthogonal so |dR| = |dRC2I|).
    //
    // THE DERIVATION. The stencils are
    //     Rdot  = (8*(s3 - s1) - (s4 - s0)) / (12 h)      sum|coef| = 18
    //     Rddot = (-s0 + 16 s1 - 30 s2 + 16 s3 - s4) / (12 h^2)  sum|coef| = 64
    // at h = 0.25 s. If every sample is off by at most one ulp and the errors
    // align adversarially, the ceilings are
    //     dRdot  <= 18 * 2.2204e-16 / 3.00   = 1.3323e-15  -> dv <= 1.0658e-11
    //     dRddot <= 64 * 2.2204e-16 / 0.75   = 1.8948e-14  -> da <= 1.5158e-10
    // after the ||r||_1 = 8000 lever arm. The bounds below sit at ~2x those
    // ceilings. That is exactly the boundary that matters: TOLERATE the known
    // 1-ulp codegen sensitivity, which is immaterial at 1.55 nm, and CATCH
    // anything worse.
    //
    // These are strictly MORE sensitive than the matrix bounds, so they are not
    // redundant: DA_BOUND fires at dRddot = 3.75e-14, i.e. 26.7x before
    // RDDOT_BOUND = 1e-12 would, and DV_BOUND fires at dRdot = 2.5e-15, 40x
    // before RDOT_BOUND = 1e-13. They remain the only assertions in this file
    // that can detect a release-versus-debug divergence confined to the
    // derivative chain, which is what #29 is.
    //
    // Do NOT tighten these to the observed release values (da 1.639e-11,
    // dv 2.049e-12). Today's dirty samples partially cancel; 1e-10 and 1e-11
    // were both proposed and both sit BELOW the ceilings above (0.66x and 0.94x),
    // so they would fire as false alarms on a host where the same known 1-ulp
    // sensitivity lands on more of the five samples.
    const DV_BOUND: f64 = 2e-11;
    const DA_BOUND: f64 = 3e-10;

    let root = repo_root();
    let bytes = require_ok!(
        std::fs::read(
            root.join("crates/satpy_core/tests/data/erfa_sofa_derived_frame_time_v1.json"),
        ),
        "accepted frame/time fixture missing"
    );
    let fixture: Value = require_ok!(
        serde_json::from_slice(&bytes),
        "frame/time fixture is not JSON"
    );
    let cases = require_some!(
        required_field!(&fixture, "cases").as_array(),
        "cases is not an array"
    );
    assert_eq!(cases.len(), 20);

    let finals = require_ok!(
        std::fs::read_to_string(root.join("assets/reference/frame_time/finals2000A.all")),
        "sealed finals2000A.all is not readable"
    );
    let computed = require_ok!(
        compute_all_cases(&finals),
        "sealed frame input does not resolve"
    );
    assert_eq!(computed.len(), 20);

    let (mut rotation_max_error, mut rate_max_error, mut curvature_max_error) =
        (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut position_max_error, mut velocity_max_error, mut acceleration_max_error) =
        (0.0_f64, 0.0_f64, 0.0_f64);

    for (index, (case, got)) in cases.iter().zip(computed.iter()).enumerate() {
        // Order and identity cross-check.
        let expect_policy = if required_field!(case, "eop_policy") == "zero_eop" {
            EopPolicy::Zero
        } else {
            EopPolicy::Real
        };
        let epoch_name = require_some!(
            required_field!(case, "epoch_utc").as_str(),
            "epoch name is not a string"
        );
        assert_eq!(got.epoch_name, epoch_name, "case {index}");
        assert!(got.policy == expect_policy, "case {index} policy");
        let state_index = require_some!(
            required_field!(case, "state_index").as_u64(),
            "state index is not unsigned"
        );
        assert_eq!(
            got.state_index,
            require_ok!(
                usize::try_from(state_index),
                "state index does not fit usize"
            )
        );

        let target_matrix = require_some!(
            hex_mat(required_field!(case, "R_gcrs_to_itrs")),
            "invalid target matrix"
        );
        let target_rate = require_some!(
            hex_mat(required_field!(case, "Rdot_s1")),
            "invalid target rate"
        );
        let target_curvature = require_some!(
            hex_mat(required_field!(case, "Rddot_s2")),
            "invalid target curvature"
        );
        for (
            ((got_matrix_row, got_rate_row), got_curvature_row),
            ((target_matrix_row, target_rate_row), target_curvature_row),
        ) in got.r.iter().zip(&got.rdot).zip(&got.rddot).zip(
            target_matrix
                .iter()
                .zip(&target_rate)
                .zip(&target_curvature),
        ) {
            for (
                ((got_matrix, got_rate), got_curvature),
                ((target_matrix, target_rate), target_curvature),
            ) in got_matrix_row
                .iter()
                .zip(got_rate_row)
                .zip(got_curvature_row)
                .zip(
                    target_matrix_row
                        .iter()
                        .zip(target_rate_row)
                        .zip(target_curvature_row),
                )
            {
                rotation_max_error = rotation_max_error.max((*got_matrix - *target_matrix).abs());
                rate_max_error = rate_max_error.max((*got_rate - *target_rate).abs());
                curvature_max_error =
                    curvature_max_error.max((*got_curvature - *target_curvature).abs());
            }
        }
        let (target_position, target_velocity, target_acceleration) = (
            require_some!(
                hex_vec(required_field!(case, "r_itrs_km")),
                "invalid target position"
            ),
            require_some!(
                hex_vec(required_field!(case, "v_itrs_km_s")),
                "invalid target velocity"
            ),
            require_some!(
                hex_vec(required_field!(case, "a_itrs_km_s2")),
                "invalid target acceleration"
            ),
        );
        for (
            ((got_position, got_velocity), got_acceleration),
            ((target_position, target_velocity), target_acceleration),
        ) in got.r_itrs.iter().zip(&got.v_itrs).zip(&got.a_itrs).zip(
            target_position
                .iter()
                .zip(&target_velocity)
                .zip(&target_acceleration),
        ) {
            position_max_error = position_max_error.max((*got_position - *target_position).abs());
            velocity_max_error = velocity_max_error.max((*got_velocity - *target_velocity).abs());
            acceleration_max_error =
                acceleration_max_error.max((*got_acceleration - *target_acceleration).abs());
        }
    }

    // libtest CAPTURES this on a passing run. It is visible only under
    // `-- --nocapture`, or on failure. Do not describe it as gate-visible
    // observability: it is not, and a claim that it was is what sent the
    // 2026-07-24 deletion of DV/DA through review on a false premise. The
    // assertions below are the only thing that speaks on a normal gate run.
    eprintln!(
        "frame_time comparator max deltas: dR={rotation_max_error:.3e} dRdot={rate_max_error:.3e} \
         dRddot={curvature_max_error:.3e} dr={position_max_error:.3e} dv={velocity_max_error:.3e} da={acceleration_max_error:.3e}"
    );
    assert!(
        rotation_max_error <= R_BOUND,
        "max|dR|={rotation_max_error:.3e} exceeds {R_BOUND:.0e}"
    );
    assert!(
        rate_max_error <= RDOT_BOUND,
        "max|dRdot|={rate_max_error:.3e} exceeds {RDOT_BOUND:.0e}"
    );
    assert!(
        curvature_max_error <= RDDOT_BOUND,
        "max|dRddot|={curvature_max_error:.3e} exceeds {RDDOT_BOUND:.0e}"
    );
    assert!(
        position_max_error <= DR_BOUND,
        "max dr={position_max_error:.3e} km exceeds {DR_BOUND:.0e}"
    );
    assert!(
        velocity_max_error <= DV_BOUND,
        "max dv={velocity_max_error:.3e} km/s exceeds stencil one-ulp ceiling bound {DV_BOUND:.0e}"
    );
    assert!(
        acceleration_max_error <= DA_BOUND,
        "max da={acceleration_max_error:.3e} km/s^2 exceeds stencil one-ulp ceiling bound {DA_BOUND:.0e}"
    );
}
