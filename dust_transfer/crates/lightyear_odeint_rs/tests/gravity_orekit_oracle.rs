//! Sealed Task 4BG Orekit d/o5 full-gravity comparator.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use lightyear_odeint_rs::config::{self, GlobalCoeffs};
use lightyear_odeint_rs::{get_global_coeffs_packed, load_constants_from_bytes};
use satpy_core::{
    spherical_gravity_impl_sincos, spherical_gravity_impl_sincos_packed, GravityCache,
};

const FIXTURE: &[u8] = include_bytes!("data/orekit_dir_r6_5x5_v1.json");
const DERIVED: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");
const FIXTURE_SHA256: &str = "670da646c07ad1e303ab8b6cb23820c02832f6162f34f8f7402fe1212227a379";
const MANIFEST_SHA256: &str = "681bfabf3b43e342f9e7d6b3dfd49a1e05298e8059c55645ce3001fd0c9cae33";
const SEMANTIC_SHA256: &str = "022e07c17ea3d454e9a61a52cbf04731044781e7fbfc3e0534fdaa956b893dbf";
const MU_M3_S2: f64 = 3.986_004_415e14;
const TOL_ABS_M_S2: f64 = 5.0e-11;
const TOL_REL: f64 = 2.0e-12;

static GLOBAL_COEFFS_LOCK: std::sync::LazyLock<Mutex<()>> =
    std::sync::LazyLock::new(|| Mutex::new(()));

macro_rules! require {
    ($condition:expr, $message:expr $(,)?) => {
        if !$condition {
            return Err(anyhow::anyhow!("{}", $message));
        }
    };
}

macro_rules! require_eq {
    ($actual:expr, $expected:expr, $label:expr $(,)?) => {
        let actual = &$actual;
        let expected = &$expected;
        if actual != expected {
            return Err(anyhow::anyhow!(
                "{} mismatch: actual={:?}, expected={:?}",
                $label,
                actual,
                expected
            ));
        }
    };
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema: String,
    authority: String,
    claim_scope: String,
    semantic_sha256: String,
    provenance: Provenance,
    model: Model,
    evaluation: Evaluation,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    generator_source_sha256: String,
    source_gfc_sha256: String,
    derived_d15_sha256: String,
    jar_aggregate_sha256: String,
    orekit_version: String,
    hipparchus_core_version: String,
    hipparchus_geometry_version: String,
    orekit_jar_sha256: String,
    hipparchus_core_jar_sha256: String,
    hipparchus_geometry_jar_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    name: String,
    tide_system: String,
    normalization: String,
    gm_m3_s2: String,
    reference_radius_m: String,
    stored_degree: String,
    stored_order: String,
    runtime_degree: String,
    runtime_order: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Evaluation {
    body_frame: String,
    epoch: String,
    units: String,
    absolute_tolerance_m_s2: String,
    relative_tolerance: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    position_m: [String; 3],
    orekit_noncentral_m_s2: [String; 3],
    point_mass_m_s2: [String; 3],
    full_m_s2: [String; 3],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    semantic_sha256: String,
    #[serde(rename = "source")]
    _source: ManifestSource,
    #[serde(rename = "model")]
    _model: ManifestModel,
    #[serde(rename = "extraction")]
    _extraction: Extraction,
    #[serde(rename = "oracle")]
    _oracle: Oracle,
    jar_closure: JarClosure,
    closure_paths: Vec<String>,
    payloads: Vec<Payload>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestSource {
    #[serde(rename = "url")]
    _url: String,
    #[serde(rename = "doi")]
    _doi: String,
    #[serde(rename = "reference_epoch")]
    _reference_epoch: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestModel {
    #[serde(rename = "name")]
    _name: String,
    #[serde(rename = "dataset")]
    _dataset: String,
    #[serde(rename = "gm_m3_s2")]
    _gm_m3_s2: String,
    #[serde(rename = "reference_radius_m")]
    _reference_radius_m: String,
    #[serde(rename = "maximum_degree")]
    _maximum_degree: u64,
    #[serde(rename = "maximum_order")]
    _maximum_order: u64,
    #[serde(rename = "normalization")]
    _normalization: String,
    #[serde(rename = "tide_system")]
    _tide_system: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Extraction {
    #[serde(rename = "locale")]
    _locale: String,
    #[serde(rename = "rule")]
    _rule: String,
    #[serde(rename = "stored_degree")]
    _stored_degree: u64,
    #[serde(rename = "stored_order")]
    _stored_order: u64,
    #[serde(rename = "runtime_degree")]
    _runtime_degree: u64,
    #[serde(rename = "runtime_order")]
    _runtime_order: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Oracle {
    #[serde(rename = "fixture_schema")]
    _fixture_schema: String,
    #[serde(rename = "fixture_semantic_sha256")]
    _fixture_semantic_sha256: String,
    #[serde(rename = "claim")]
    _claim: String,
    #[serde(rename = "generator_jar_reachability")]
    _generator_jar_reachability: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JarClosure {
    aggregate_sha256: String,
    classpath: Vec<Payload>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    path: String,
    size_bytes: u64,
    sha256: String,
}

struct RestoreCoeffs(Arc<GlobalCoeffs>);
impl Drop for RestoreCoeffs {
    fn drop(&mut self) {
        config::GLOBAL_COEFFS.store(self.0.clone());
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn repo_root() -> anyhow::Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| anyhow::anyhow!("repo root: {error}"))
}

fn regular_hash(path: &Path) -> anyhow::Result<(u64, String)> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("closure file parent is missing"))?;
    require_eq!(
        parent
            .canonicalize()
            .map_err(|error| anyhow::anyhow!("canonical closure parent: {error}"))?,
        parent,
        format!("closure path has symlinked ancestor: {}", path.display())
    );
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| anyhow::anyhow!("closure file metadata {}: {error}", path.display()))?;
    require!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        format!("closure entry must be regular: {}", path.display())
    );
    let bytes = fs::read(path)
        .map_err(|error| anyhow::anyhow!("read closure file {}: {error}", path.display()))?;
    Ok((metadata.len(), hex_sha256(&bytes)))
}

fn canonical_json(value: &Value, out: &mut Vec<u8>) -> anyhow::Result<()> {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            out.extend_from_slice(value.to_string().as_bytes());
        }
        Value::Array(values) => {
            out.push(b'[');
            for (index, item) in values.iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                canonical_json(item, out)?;
            }
            out.push(b']');
        }
        Value::Object(values) => {
            out.push(b'{');
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort_unstable();
            for (index, key) in keys.iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                let encoded_key = serde_json::to_string(key)
                    .map_err(|error| anyhow::anyhow!("JSON key: {error}"))?;
                out.extend_from_slice(encoded_key.as_bytes());
                out.push(b':');
                canonical_json(
                    values
                        .get(*key)
                        .ok_or_else(|| anyhow::anyhow!("JSON value is missing"))?,
                    out,
                )?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

fn manifest_semantic(value: &Value) -> anyhow::Result<String> {
    let mut payload = value.clone();
    payload
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("manifest object is missing"))?
        .remove("semantic_sha256");
    let mut canonical = Vec::new();
    canonical_json(&payload, &mut canonical)?;
    let mut digest = Sha256::new();
    digest.update(b"PART_A_DIR_R6_GRAVITY_MANIFEST_V1\0");
    let canonical_length = u64::try_from(canonical.len()).map_err(|error| {
        anyhow::anyhow!("manifest semantic length does not fit in u64: {error}")
    })?;
    digest.update(canonical_length.to_be_bytes());
    digest.update(canonical);
    Ok(format!("{:x}", digest.finalize()))
}

fn semantic_scalar(digest: &mut Sha256, tag: &str, value: &str) {
    digest.update(tag.as_bytes());
    digest.update([0]);
    digest.update(value.as_bytes());
    digest.update([0]);
}

fn semantic_vector(digest: &mut Sha256, tag: &str, value: &[String; 3]) {
    for (axis, component) in value.iter().enumerate() {
        semantic_scalar(digest, &format!("{tag}[{axis}]"), component);
    }
}

fn fixture_semantic(fixture: &Fixture) -> String {
    let mut digest = Sha256::new();
    digest.update(b"PART_A_OREKIT_DIR_R6_GRAVITY_V1\0");
    semantic_scalar(&mut digest, "schema", &fixture.schema);
    semantic_scalar(&mut digest, "authority", &fixture.authority);
    semantic_scalar(&mut digest, "claim_scope", &fixture.claim_scope);
    let p = &fixture.provenance;
    semantic_scalar(
        &mut digest,
        "provenance.generator_source_sha256",
        &p.generator_source_sha256,
    );
    semantic_scalar(
        &mut digest,
        "provenance.source_gfc_sha256",
        &p.source_gfc_sha256,
    );
    semantic_scalar(
        &mut digest,
        "provenance.derived_d15_sha256",
        &p.derived_d15_sha256,
    );
    semantic_scalar(
        &mut digest,
        "provenance.jar_aggregate_sha256",
        &p.jar_aggregate_sha256,
    );
    semantic_scalar(&mut digest, "provenance.orekit_version", &p.orekit_version);
    semantic_scalar(
        &mut digest,
        "provenance.hipparchus_core_version",
        &p.hipparchus_core_version,
    );
    semantic_scalar(
        &mut digest,
        "provenance.hipparchus_geometry_version",
        &p.hipparchus_geometry_version,
    );
    semantic_scalar(
        &mut digest,
        "provenance.orekit_jar_sha256",
        &p.orekit_jar_sha256,
    );
    semantic_scalar(
        &mut digest,
        "provenance.hipparchus_core_jar_sha256",
        &p.hipparchus_core_jar_sha256,
    );
    semantic_scalar(
        &mut digest,
        "provenance.hipparchus_geometry_jar_sha256",
        &p.hipparchus_geometry_jar_sha256,
    );
    let model = &fixture.model;
    semantic_scalar(&mut digest, "model.name", &model.name);
    semantic_scalar(&mut digest, "model.tide_system", &model.tide_system);
    semantic_scalar(&mut digest, "model.normalization", &model.normalization);
    semantic_scalar(&mut digest, "model.gm_m3_s2", &model.gm_m3_s2);
    semantic_scalar(
        &mut digest,
        "model.reference_radius_m",
        &model.reference_radius_m,
    );
    semantic_scalar(&mut digest, "model.stored_degree", &model.stored_degree);
    semantic_scalar(&mut digest, "model.stored_order", &model.stored_order);
    semantic_scalar(&mut digest, "model.runtime_degree", &model.runtime_degree);
    semantic_scalar(&mut digest, "model.runtime_order", &model.runtime_order);
    let evaluation = &fixture.evaluation;
    semantic_scalar(&mut digest, "evaluation.body_frame", &evaluation.body_frame);
    semantic_scalar(&mut digest, "evaluation.epoch", &evaluation.epoch);
    semantic_scalar(&mut digest, "evaluation.units", &evaluation.units);
    semantic_scalar(
        &mut digest,
        "evaluation.absolute_tolerance_m_s2",
        &evaluation.absolute_tolerance_m_s2,
    );
    semantic_scalar(
        &mut digest,
        "evaluation.relative_tolerance",
        &evaluation.relative_tolerance,
    );
    for (index, case) in fixture.cases.iter().enumerate() {
        let prefix = format!("cases[{index}].");
        semantic_scalar(&mut digest, &format!("{prefix}name"), &case.name);
        semantic_vector(
            &mut digest,
            &format!("{prefix}position_m"),
            &case.position_m,
        );
        semantic_vector(
            &mut digest,
            &format!("{prefix}orekit_noncentral_m_s2"),
            &case.orekit_noncentral_m_s2,
        );
        semantic_vector(
            &mut digest,
            &format!("{prefix}point_mass_m_s2"),
            &case.point_mass_m_s2,
        );
        semantic_vector(&mut digest, &format!("{prefix}full_m_s2"), &case.full_m_s2);
    }
    format!("{:x}", digest.finalize())
}

fn validate_manifest(manifest_raw: &[u8]) -> anyhow::Result<()> {
    let manifest: Manifest = serde_json::from_slice(manifest_raw)
        .map_err(|error| anyhow::anyhow!("strict manifest schema: {error}"))?;
    let manifest_value: Value = serde_json::from_slice(manifest_raw)
        .map_err(|error| anyhow::anyhow!("manifest JSON: {error}"))?;
    require_eq!(
        manifest.schema,
        "part_a_dir_r6_gravity_manifest_v1",
        "manifest schema"
    );
    require_eq!(
        manifest.semantic_sha256,
        "166962237d266dd0bf3001e7ea602549e9f5300654b4cc07ece3cd89c730e482",
        "manifest semantic hash"
    );
    require_eq!(
        manifest_semantic(&manifest_value)?,
        manifest.semantic_sha256,
        "manifest semantic calculation"
    );

    let closure = [
        "assets/reference/gravity/dir_r6/README.md",
        "assets/reference/gravity/dir_r6/GO_CONS_GCF_2_DIR_R6.gfc",
        "assets/reference/gravity/dir_r6/CC-BY-4.0.txt",
        "crates/two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt",
        "crates/lightyear_odeint_rs/oracle/OrekitGravityVectors.java",
        "crates/lightyear_odeint_rs/tests/data/orekit_dir_r6_5x5_v1.json",
        "assets/reference/orekit_jb2008/maven/jars/orekit-13.1.2.jar",
        "assets/reference/orekit_jb2008/maven/jars/hipparchus-core-4.0.2.jar",
        "assets/reference/orekit_jb2008/maven/jars/hipparchus-geometry-4.0.2.jar",
    ];
    require_eq!(manifest.closure_paths, closure, "exact closure paths");
    let payloads = [
        (
            "assets/reference/gravity/dir_r6/README.md",
            1813,
            "1f171e082e58c7a555a63ebd34ea7ec641472073f19ea57b433132e669a24ee7",
        ),
        (
            "assets/reference/gravity/dir_r6/GO_CONS_GCF_2_DIR_R6.gfc",
            3_370_005,
            "4da4a476418553c2243c0dbc79515bb3a419f3175dea3e38c58843cb14fcff7b",
        ),
        (
            "assets/reference/gravity/dir_r6/CC-BY-4.0.txt",
            18657,
            "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
        ),
        (
            "crates/two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt",
            7830,
            "983f035818399f9cb27f1e8c604cb62b3e72d650aa4cbfadb31b1e7c4fe61f09",
        ),
        (
            "crates/lightyear_odeint_rs/oracle/OrekitGravityVectors.java",
            16043,
            "f9c8f76394dbd1b2c04cb8527d08f6d0922c70f1d416c59111ef9db5a5fd64d1",
        ),
        (
            "crates/lightyear_odeint_rs/tests/data/orekit_dir_r6_5x5_v1.json",
            3412,
            "670da646c07ad1e303ab8b6cb23820c02832f6162f34f8f7402fe1212227a379",
        ),
    ];
    require_eq!(
        manifest.payloads.len(),
        payloads.len(),
        "six ordered payloads"
    );
    let root = repo_root()?;
    for (payload, (path, size, hash)) in manifest.payloads.iter().zip(payloads) {
        require_eq!(
            (
                payload.path.as_str(),
                payload.size_bytes,
                payload.sha256.as_str()
            ),
            (path, size, hash),
            format!("payload receipt {path}")
        );
        require!(
            !path.starts_with('/') && !path.split('/').any(|part| part == ".."),
            "safe payload path"
        );
        require_eq!(
            regular_hash(&root.join(path))?,
            (size, hash.to_owned()),
            format!("payload {path}")
        );
    }
    require_eq!(
        manifest.jar_closure.classpath.len(),
        3,
        "three ordered JARs"
    );
    for (jar, path) in manifest.jar_closure.classpath.iter().zip(&closure[6..]) {
        require_eq!(jar.path, *path, "ordered JAR closure");
        require_eq!(
            regular_hash(&root.join(path))?,
            (jar.size_bytes, jar.sha256.clone()),
            format!("JAR {path}")
        );
    }
    require_eq!(
        manifest.jar_closure.aggregate_sha256,
        "7e3b504bfd38b0d6713b959085e7fcfba8a6ae635bf4b769006d816d6b7e7d24",
        "JAR aggregate SHA-256"
    );
    let mut entries = Vec::new();
    for entry in fs::read_dir(root.join("assets/reference/gravity/dir_r6"))
        .map_err(|error| anyhow::anyhow!("dir_r6 entries: {error}"))?
    {
        let entry = entry.map_err(|error| anyhow::anyhow!("dir_r6 entry: {error}"))?;
        entries.push(
            entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("dir_r6 path must be UTF-8"))?,
        );
    }
    entries.sort();
    require_eq!(
        entries,
        [
            "CC-BY-4.0.txt",
            "GO_CONS_GCF_2_DIR_R6.gfc",
            "README.md",
            "manifest.json"
        ],
        "no unlisted dir_r6 file"
    );
    let mut jar_entries = Vec::new();
    for entry in fs::read_dir(root.join("assets/reference/orekit_jb2008/maven/jars"))
        .map_err(|error| anyhow::anyhow!("JAR closure entries: {error}"))?
    {
        let entry = entry.map_err(|error| anyhow::anyhow!("JAR closure entry: {error}"))?;
        jar_entries.push(
            entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("JAR closure path must be UTF-8"))?,
        );
    }
    jar_entries.sort();
    require_eq!(
        jar_entries,
        [
            "hipparchus-core-4.0.2.jar",
            "hipparchus-geometry-4.0.2.jar",
            "orekit-13.1.2.jar"
        ],
        "no unlisted JAR closure file"
    );
    Ok(())
}

fn parse_hex_f64(value: &str) -> anyhow::Result<f64> {
    require_eq!(value.len(), 18, "f64 hex width");
    let digits = value
        .strip_prefix("0x")
        .ok_or_else(|| anyhow::anyhow!("f64 hex prefix is missing"))?;
    require!(
        digits
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte)),
        "lowercase f64 hex"
    );
    let bits = u64::from_str_radix(digits, 16)
        .map_err(|error| anyhow::anyhow!("valid f64 hex: {error}"))?;
    Ok(f64::from_bits(bits))
}

fn vector(values: &[String; 3]) -> anyhow::Result<[f64; 3]> {
    let [x, y, z] = values;
    Ok([parse_hex_f64(x)?, parse_hex_f64(y)?, parse_hex_f64(z)?])
}

fn norm(values: &[f64; 3]) -> f64 {
    let [x, y, z] = values;
    x.hypot(*y).hypot(*z)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Orekit comparator tolerance retains its established binary64 expression order"
)]
fn assert_bound(actual: f64, expected: f64, label: &str) -> anyhow::Result<()> {
    let delta = (actual - expected).abs();
    let bound = TOL_ABS_M_S2 + TOL_REL * expected.abs();
    if delta <= bound {
        Ok(())
    } else {
        Err(anyhow::anyhow!("{label}: delta={delta:e}, bound={bound:e}"))
    }
}

const fn same_f64(lhs: f64, rhs: f64) -> bool {
    match (lhs.classify(), rhs.classify()) {
        (std::num::FpCategory::Nan, _) | (_, std::num::FpCategory::Nan) => false,
        (std::num::FpCategory::Zero, std::num::FpCategory::Zero) => true,
        _ => lhs.to_bits() == rhs.to_bits(),
    }
}

fn exact_usize_as_f64(value: usize, label: &str) -> anyhow::Result<f64> {
    u32::try_from(value).map(f64::from).map_err(|_| {
        anyhow::anyhow!("{label} must fit the sealed gravity parser's u32 domain: {value}")
    })
}

/// Rebuild raw loader scratch only inside this integration oracle.
///
/// Production publishes just the validated packed authority. Keeping the raw
/// representation here lets the test compare packed behavior to the sealed
/// loader's raw evaluator without restoring a production metadata API.
#[expect(
    clippy::suboptimal_flops,
    reason = "the test-only raw oracle must reproduce the sealed loader's binary64 normalization"
)]
fn sealed_raw_coefficients(order: usize) -> anyhow::Result<(Vec<f64>, Vec<f64>, usize)> {
    let stride = order
        .checked_add(2)
        .ok_or_else(|| anyhow::anyhow!("sealed raw coefficient stride overflowed"))?;
    let table_len = stride
        .checked_mul(stride)
        .ok_or_else(|| anyhow::anyhow!("sealed raw coefficient table length overflowed"))?;
    let mut c_coeffs = vec![0.0; table_len];
    let mut s_coeffs = vec![0.0; table_len];
    *c_coeffs
        .first_mut()
        .ok_or_else(|| anyhow::anyhow!("sealed raw coefficient table must contain C00"))? = 1.0;

    let max_factorial = stride
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("sealed raw normalization limit overflowed"))?;
    let factorial_count = max_factorial
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("sealed raw normalization table length overflowed"))?;
    let mut ln_factorial = vec![0.0; factorial_count];
    let mut previous_log_factorial = 0.0;
    for (index, log_factorial) in ln_factorial.iter_mut().enumerate().skip(1) {
        let index_f64 = exact_usize_as_f64(index, "sealed normalization index")?;
        *log_factorial = previous_log_factorial + index_f64.ln();
        previous_log_factorial = *log_factorial;
    }

    let source = std::str::from_utf8(DERIVED)
        .map_err(|error| anyhow::anyhow!("sealed raw coefficient source must be UTF-8: {error}"))?;
    for line in source.lines() {
        let mut fields = line.split_whitespace();
        let (Some(degree_text), Some(degree_order_text), Some(c_text), Some(s_text)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let degree = degree_text.parse::<usize>().map_err(|error| {
            anyhow::anyhow!("sealed raw coefficient degree `{degree_text}`: {error}")
        })?;
        let degree_order = degree_order_text.parse::<usize>().map_err(|error| {
            anyhow::anyhow!("sealed raw coefficient order `{degree_order_text}`: {error}")
        })?;
        let c_value = c_text
            .replace(['D', 'd'], "E")
            .parse::<f64>()
            .map_err(|error| {
                anyhow::anyhow!("sealed raw cosine coefficient `{c_text}`: {error}")
            })?;
        let s_value = s_text
            .replace(['D', 'd'], "E")
            .parse::<f64>()
            .map_err(|error| anyhow::anyhow!("sealed raw sine coefficient `{s_text}`: {error}"))?;

        if degree <= order && degree_order <= degree {
            let row_start = degree
                .checked_mul(stride)
                .ok_or_else(|| anyhow::anyhow!("sealed raw coefficient row offset overflowed"))?;
            let coefficient_index = row_start
                .checked_add(degree_order)
                .ok_or_else(|| anyhow::anyhow!("sealed raw coefficient index overflowed"))?;
            if coefficient_index < table_len {
                let delta_order_zero = if degree_order == 0 { 1.0 } else { 0.0 };
                let degree_f64 = exact_usize_as_f64(degree, "sealed coefficient degree")?;
                let denominator = (2.0 - delta_order_zero) * (2.0 * degree_f64 + 1.0);
                let numerator_index = degree.checked_add(degree_order).ok_or_else(|| {
                    anyhow::anyhow!("sealed normalization numerator index overflowed")
                })?;
                let denominator_index = degree.checked_sub(degree_order).ok_or_else(|| {
                    anyhow::anyhow!("sealed normalization denominator index underflowed")
                })?;
                let log_factorial_numerator =
                    ln_factorial.get(numerator_index).copied().ok_or_else(|| {
                        anyhow::anyhow!("sealed normalization numerator is out of range")
                    })?;
                let log_factorial_denominator = ln_factorial
                    .get(denominator_index)
                    .copied()
                    .ok_or_else(|| {
                        anyhow::anyhow!("sealed normalization denominator is out of range")
                    })?;
                let log_normalization = 0.5
                    * ((log_factorial_numerator - log_factorial_denominator) - denominator.ln());
                let normalization = log_normalization.exp();
                *c_coeffs.get_mut(coefficient_index).ok_or_else(|| {
                    anyhow::anyhow!("sealed raw cosine coefficient index is out of range")
                })? = c_value / normalization;
                *s_coeffs.get_mut(coefficient_index).ok_or_else(|| {
                    anyhow::anyhow!("sealed raw sine coefficient index is out of range")
                })? = s_value / normalization;
            }
        }
    }
    Ok((c_coeffs, s_coeffs, stride))
}

fn validate_fixture_metadata(fixture: &Fixture) -> anyhow::Result<()> {
    require_eq!(
        fixture.schema,
        "part_a_orekit_dir_r6_gravity_v1",
        "fixture schema"
    );
    require_eq!(
        fixture.authority,
        "Official ICGEM/GFZ GO_CONS_GCF_2_DIR_R6 with Orekit 13.1.2 d/o5 comparator",
        "fixture authority"
    );
    require_eq!(
        fixture.claim_scope,
        "Orekit d/o5 noncentral and full body-fixed gravity comparator; no frame transform, propagation, or Task 5 claim",
        "fixture claim scope"
    );
    require_eq!(
        fixture.semantic_sha256,
        SEMANTIC_SHA256,
        "semantic fixture seal"
    );
    require_eq!(
        fixture_semantic(fixture),
        fixture.semantic_sha256,
        "independent fixture semantic hash"
    );
    require_eq!(
        fixture.provenance.generator_source_sha256,
        "f9c8f76394dbd1b2c04cb8527d08f6d0922c70f1d416c59111ef9db5a5fd64d1",
        "generator source SHA-256"
    );
    require_eq!(
        fixture.provenance.source_gfc_sha256,
        "4da4a476418553c2243c0dbc79515bb3a419f3175dea3e38c58843cb14fcff7b",
        "source GFC SHA-256"
    );
    require_eq!(
        fixture.provenance.derived_d15_sha256,
        "983f035818399f9cb27f1e8c604cb62b3e72d650aa4cbfadb31b1e7c4fe61f09",
        "derived d15 SHA-256"
    );
    require_eq!(
        fixture.provenance.jar_aggregate_sha256,
        "7e3b504bfd38b0d6713b959085e7fcfba8a6ae635bf4b769006d816d6b7e7d24",
        "JAR aggregate SHA-256"
    );
    require_eq!(
        (
            fixture.provenance.orekit_jar_sha256.as_str(),
            fixture.provenance.hipparchus_core_jar_sha256.as_str(),
            fixture.provenance.hipparchus_geometry_jar_sha256.as_str()
        ),
        (
            "89c2060c60dbe194a87dddcf3bb8343ebd16733958efe4dcc996cebbbeed655d",
            "7c56992f3af64429d871c33c00808ee5db5d9ed56b395b5d3d31319c4ef7ba0a",
            "4e8eede49aabd4fb71f08dd0b8b87297a9e78ed36f05c3caa4e63de5f469cceb"
        ),
        "JAR SHA-256 values"
    );
    require_eq!(
        (
            fixture.provenance.orekit_version.as_str(),
            fixture.provenance.hipparchus_core_version.as_str(),
            fixture.provenance.hipparchus_geometry_version.as_str()
        ),
        ("13.1.2", "4.0.2", "4.0.2"),
        "JAR versions"
    );
    require_eq!(
        (
            fixture.model.name.as_str(),
            fixture.model.tide_system.as_str(),
            fixture.model.normalization.as_str()
        ),
        (
            "GO_CONS_EGM_GOC_2__20091009T000000_20131020T235959_0201",
            "tide_free",
            "fully_normalized"
        ),
        "gravity model identity"
    );
    require!(
        same_f64(parse_hex_f64(&fixture.model.gm_m3_s2)?, 3.986_004_415e14),
        "model GM"
    );
    require!(
        same_f64(
            parse_hex_f64(&fixture.model.reference_radius_m)?,
            6_378_136.46
        ),
        "model reference radius"
    );
    require!(
        same_f64(parse_hex_f64(&fixture.model.stored_degree)?, 15.0)
            && same_f64(parse_hex_f64(&fixture.model.stored_order)?, 15.0)
            && same_f64(parse_hex_f64(&fixture.model.runtime_degree)?, 5.0)
            && same_f64(parse_hex_f64(&fixture.model.runtime_order)?, 5.0),
        "model degrees and orders"
    );
    require_eq!(
        (
            fixture.evaluation.body_frame.as_str(),
            fixture.evaluation.epoch.as_str(),
            fixture.evaluation.units.as_str()
        ),
        ("Frame.getRoot identity", "J2000_EPOCH", "m,m/s^2"),
        "evaluation identity"
    );
    require!(
        same_f64(
            parse_hex_f64(&fixture.evaluation.absolute_tolerance_m_s2)?,
            TOL_ABS_M_S2
        ),
        "absolute tolerance"
    );
    require!(
        same_f64(
            parse_hex_f64(&fixture.evaluation.relative_tolerance)?,
            TOL_REL
        ),
        "relative tolerance"
    );
    require_eq!(fixture.cases.len(), 5, "fixed oracle corpus");
    Ok(())
}

fn verify_packed_kernel(fixture: &Fixture) -> anyhow::Result<()> {
    let previous = config::GLOBAL_COEFFS.load_full();
    let _restore = RestoreCoeffs(previous);
    load_constants_from_bytes(DERIVED, 5)
        .map_err(|error| anyhow::anyhow!("load sealed derived production bytes: {error}"))?;
    let packed = get_global_coeffs_packed()
        .ok_or_else(|| anyhow::anyhow!("packed d/o5 coefficients are missing"))?;
    let (raw_c, raw_s, raw_stride) = sealed_raw_coefficients(5)?;

    for row in &fixture.cases {
        let position_m = vector(&row.position_m)?;
        let expected_full = vector(&row.full_m_s2)?;
        let expected_noncentral = vector(&row.orekit_noncentral_m_s2)?;
        let expected_point_mass = vector(&row.point_mass_m_s2)?;
        let [position_x, position_y, position_z] = position_m;
        let state_km = [
            position_x / 1000.0,
            position_y / 1000.0,
            position_z / 1000.0,
            0.0,
            0.0,
            0.0,
        ];
        let actual_km = spherical_gravity_impl_sincos_packed(
            &state_km,
            0.0,
            1.0,
            &mut GravityCache::default(),
            &packed,
        )
        .map_err(|error| anyhow::anyhow!("{} packed gravity evaluation: {error}", row.name))?;
        let raw_km = spherical_gravity_impl_sincos(
            &state_km,
            0.0,
            1.0,
            5,
            &raw_c,
            &raw_s,
            raw_stride,
            &mut GravityCache::default(),
        )
        .map_err(|error| anyhow::anyhow!("{} raw gravity evaluation: {error}", row.name))?;
        let [actual_x, actual_y, actual_z] = actual_km;
        let actual = [actual_x * 1000.0, actual_y * 1000.0, actual_z * 1000.0];
        let [raw_x, raw_y, raw_z] = raw_km;
        let raw = [raw_x * 1000.0, raw_y * 1000.0, raw_z * 1000.0];
        for (axis, (packed_component, raw_component)) in actual.iter().zip(raw.iter()).enumerate() {
            assert_bound(
                *packed_component,
                *raw_component,
                &format!("{} packed/raw[{axis}]", row.name),
            )?;
        }
        let radius_cubed = norm(&position_m).powi(3);
        let [actual_x, actual_y, actual_z] = actual;
        let point_mass = [
            -MU_M3_S2 * position_x / radius_cubed,
            -MU_M3_S2 * position_y / radius_cubed,
            -MU_M3_S2 * position_z / radius_cubed,
        ];
        let [point_mass_x, point_mass_y, point_mass_z] = point_mass;
        let actual_noncentral = [
            actual_x - point_mass_x,
            actual_y - point_mass_y,
            actual_z - point_mass_z,
        ];
        for (
            axis,
            (
                (
                    (actual_component, expected_full_component),
                    (point_mass_component, expected_point_mass_component),
                ),
                (actual_noncentral_component, expected_noncentral_component),
            ),
        ) in actual
            .iter()
            .zip(expected_full.iter())
            .zip(point_mass.iter().zip(expected_point_mass.iter()))
            .zip(actual_noncentral.iter().zip(expected_noncentral.iter()))
            .enumerate()
        {
            assert_bound(
                *actual_component,
                *expected_full_component,
                &format!("{} full[{axis}]", row.name),
            )?;
            assert_bound(
                *point_mass_component,
                *expected_point_mass_component,
                &format!("{} point_mass[{axis}]", row.name),
            )?;
            assert_bound(
                *actual_noncentral_component,
                *expected_noncentral_component,
                &format!("{} noncentral[{axis}]", row.name),
            )?;
        }
        assert_bound(
            norm(&actual),
            norm(&expected_full),
            &format!("{} full norm", row.name),
        )?;
        assert_bound(
            norm(&actual_noncentral),
            norm(&expected_noncentral),
            &format!("{} noncentral norm", row.name),
        )?;
    }
    Ok(())
}

fn verify_sealed_gravity_oracle() -> anyhow::Result<()> {
    let manifest_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/reference/gravity/dir_r6/manifest.json"
    );
    let manifest = fs::read(manifest_path)
        .map_err(|error| anyhow::anyhow!("read DIR-R6 manifest: {error}"))?;
    require_eq!(
        hex_sha256(&manifest),
        MANIFEST_SHA256,
        "strict manifest bytes"
    );
    validate_manifest(&manifest)?;
    require_eq!(hex_sha256(FIXTURE), FIXTURE_SHA256, "strict fixture bytes");
    let fixture: Fixture = serde_json::from_slice(FIXTURE)
        .map_err(|error| anyhow::anyhow!("strict fixture schema: {error}"))?;
    validate_fixture_metadata(&fixture)?;
    verify_packed_kernel(&fixture)
}

#[test]
fn orekit_dir_r6_manifest_fixture_and_full_packed_kernel_are_sealed() {
    let _lock = GLOBAL_COEFFS_LOCK
        .lock()
        .unwrap_or_else(|error| panic!("coefficient lock: {error}"));
    verify_sealed_gravity_oracle().unwrap_or_else(|error| panic!("{error}"));
}
