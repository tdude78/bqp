use jb_rs::jb2008::{jb2008_density, Jb2008Input};
use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt::{self, Write};
use std::fs;
use std::path::{Component, Path, PathBuf};

const MANIFEST_SCHEMA: &str = "part_a_orekit_jb2008_manifest_v1";
const MANIFEST_AUTHORITY: &str = "Orekit 13.1.2 synthetic-frame JB2008 mapping oracle";
const MANIFEST_CLAIM_SCOPE: &str = "Orekit synthetic-frame mapping oracle plus Rust primitive-kernel conformance only; production Rust adapter comparison deferred";
const MANIFEST_DOMAIN: &str = "PART_A_OREKIT_JB2008_MANIFEST_V1";
const MANIFEST_RAW_SHA256: &str =
    "6d77ddb18ad82e7b2f3c6a319d6c03c7b214ce751602f368d0fa7dec64c42d48";
const MANIFEST_SEMANTIC_SHA256: &str =
    "53889bcd31fbd1eaae141a5dd46179ef379afa3b909dc2a6453d772690cb0096";

const FIXTURE_SCHEMA: &str = "part_a_orekit_jb2008_synthetic_adapter_v1";
const FIXTURE_AUTHORITY: &str = "Orekit 13.1.2 synthetic-frame JB2008 mapping oracle";
const FIXTURE_CLAIM_SCOPE: &str =
    "Orekit synthetic-frame mapping oracle only; production Rust adapter comparison deferred";
const FIXTURE_DOMAIN: &str = "PART_A_OREKIT_JB2008_SYNTHETIC_ADAPTER_V1";
const FIXTURE_RAW_SHA256: &str = "928ffe14784be8f3db114f4b3ea4a06e4b84ae95d0d73227e214fd30263adade";
const FIXTURE_SEMANTIC_SHA256: &str =
    "7321f742c8f41afa9a81b1e7e9a866f6413f7f97e72ca9faf0df3dc8c44c1eb9";

macro_rules! require {
    ($condition:expr, $message:expr $(,)?) => {
        if !$condition {
            return Err(anyhow::anyhow!($message));
        }
    };
}

macro_rules! require_eq {
    ($actual:expr, $expected:expr, $label:expr $(,)?) => {
        if $actual != $expected {
            return Err(anyhow::anyhow!(
                "{} mismatch: actual={:?}, expected={:?}",
                $label,
                $actual,
                $expected
            ));
        }
    };
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceManifest {
    schema: String,
    authority_label: String,
    claim_scope: String,
    runtime_closure: RuntimeClosure,
    payloads: Vec<PayloadReceipt>,
    sources: Vec<SourceReceipt>,
    license_policy: LicensePolicy,
    semantic_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeClosure {
    closure_kind: String,
    java_class_major: u16,
    jar_aggregate_algorithm: String,
    jar_aggregate_path_root: String,
    jar_regular_file_count: u64,
    jar_aggregate_size_bytes: u64,
    jar_aggregate_sha256: String,
    classpath_order: Vec<String>,
    excluded_hipparchus_modules: Vec<String>,
    external_data: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PayloadReceipt {
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceReceipt {
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_entry: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LicensePolicy {
    license: String,
    preserve_embedded_license_and_notice: bool,
    modified_generator_is_separately_identified: bool,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct HexF64(u64);

impl HexF64 {
    const fn as_f64(self) -> f64 {
        f64::from_bits(self.0)
    }

    const fn bits(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for HexF64 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "0x{:016x}", self.0)
    }
}

impl Serialize for HexF64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("0x{:016x}", self.0))
    }
}

struct HexF64Visitor;

impl Visitor<'_> for HexF64Visitor {
    type Value = HexF64;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("finite lowercase 0x plus 16 hexadecimal digits")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let digits = value
            .strip_prefix("0x")
            .filter(|digits| digits.len() == 16)
            .ok_or_else(|| E::custom("binary64 must be lowercase 0x plus 16 hex digits"))?;
        if !digits
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(E::custom(
                "binary64 must contain lowercase hexadecimal digits only",
            ));
        }
        let bits = u64::from_str_radix(digits, 16)
            .map_err(|_| E::custom("binary64 hexadecimal value is invalid"))?;
        if !f64::from_bits(bits).is_finite() {
            return Err(E::custom("binary64 value must be finite"));
        }
        Ok(HexF64(bits))
    }
}

impl<'de> Deserialize<'de> for HexF64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(HexF64Visitor)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceFixture {
    schema: String,
    authority: String,
    semantic_sha256: String,
    claim_scope: String,
    exclusions: Vec<String>,
    provenance: FixtureProvenance,
    canonicalization: Canonicalization,
    time_and_frame_law: TimeAndFrameLaw,
    earth: Earth,
    units: Units,
    cases: Vec<FixtureCase>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureProvenance {
    generator_source_sha256: String,
    orekit_version: String,
    orekit_jar_sha256: String,
    hipparchus_core_version: String,
    hipparchus_core_jar_sha256: String,
    hipparchus_geometry_version: String,
    hipparchus_geometry_jar_sha256: String,
    java_vendor: String,
    java_version: String,
    java_vm_name: String,
    java_vm_version: String,
    java_specification_version: String,
    os_arch: String,
    file_encoding: String,
    compile_flags: String,
    runtime_flags: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Canonicalization {
    encoding: String,
    json: String,
    f64: String,
    semantic_hash_domain: String,
    semantic_hash_algorithm: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TimeAndFrameLaw {
    time_scale: String,
    fixed_utc_offset_from_tai_s: HexF64,
    leap_second_policy: String,
    eci_frame: String,
    body_frame: String,
    transform_scope: String,
    rotation_convention: String,
    angle_law: String,
    reference_epoch_fixed_utc: String,
    theta0_rad: HexF64,
    omega_rad_s: HexF64,
    external_data: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Earth {
    shape: String,
    equatorial_radius_m: HexF64,
    flattening: HexF64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Units {
    cartesian: String,
    angle: String,
    altitude: String,
    density: String,
    mjd: String,
    f10: String,
    s10_xm10_y10: String,
    dstdtc: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureCase {
    id: String,
    epoch: Epoch,
    boundary: Boundary,
    design: Design,
    inputs: Inputs,
    expected: Expected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Epoch {
    fixed_utc: String,
    tai_minus_fixed_utc_s: HexF64,
    seconds_from_frame_reference: HexF64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Boundary {
    tag: String,
    driver_transition: String,
    driver_profile_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Design {
    geodetic_latitude_rad: HexF64,
    geodetic_longitude_rad: HexF64,
    altitude_m: HexF64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Inputs {
    satellite_eci_m: [HexF64; 3],
    earth_to_sun_eci_m: [HexF64; 3],
    jb_drivers: JbDrivers,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JbDrivers {
    f10: HexF64,
    f10b: HexF64,
    s10: HexF64,
    s10b: HexF64,
    xm10: HexF64,
    xm10b: HexF64,
    y10: HexF64,
    y10b: HexF64,
    dstdtc: HexF64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    eci_to_body_matrix: [[HexF64; 3]; 3],
    satellite_body_m: [HexF64; 3],
    sun_body_m: [HexF64; 3],
    satellite_geodetic: Geodetic,
    sun_geodetic: Geodetic,
    jb_primitive_inputs: PrimitiveInputs,
    density_kg_m3: HexF64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Geodetic {
    longitude_rad: HexF64,
    latitude_rad: HexF64,
    altitude_m: HexF64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrimitiveInputs {
    date_mjd_fixed_utc: HexF64,
    #[serde(rename = "sun_longitude_rad_as_sunRA")]
    sun_longitude_rad_as_sun_ra: HexF64,
    #[serde(rename = "sun_geodetic_latitude_rad_as_sunDecli")]
    sun_geodetic_latitude_rad_as_sun_decli: HexF64,
    #[serde(rename = "satellite_geodetic_longitude_rad_as_satLon")]
    satellite_geodetic_longitude_rad_as_sat_lon: HexF64,
    #[serde(rename = "satellite_geodetic_latitude_rad_as_satLat")]
    satellite_geodetic_latitude_rad_as_sat_lat: HexF64,
    #[serde(rename = "satellite_ellipsoidal_altitude_m_as_satAlt")]
    satellite_ellipsoidal_altitude_m_as_sat_alt: HexF64,
    f10: HexF64,
    f10b: HexF64,
    s10: HexF64,
    s10b: HexF64,
    xm10: HexF64,
    xm10b: HexF64,
    y10: HexF64,
    y10b: HexF64,
    dstdtc: HexF64,
}

#[derive(Serialize)]
struct FixtureWithoutSemantic<'a> {
    schema: &'a str,
    authority: &'a str,
    claim_scope: &'a str,
    exclusions: &'a [String],
    provenance: &'a FixtureProvenance,
    canonicalization: &'a Canonicalization,
    time_and_frame_law: &'a TimeAndFrameLaw,
    earth: &'a Earth,
    units: &'a Units,
    cases: &'a [FixtureCase],
}

impl<'a> From<&'a ReferenceFixture> for FixtureWithoutSemantic<'a> {
    fn from(fixture: &'a ReferenceFixture) -> Self {
        Self {
            schema: &fixture.schema,
            authority: &fixture.authority,
            claim_scope: &fixture.claim_scope,
            exclusions: &fixture.exclusions,
            provenance: &fixture.provenance,
            canonicalization: &fixture.canonicalization,
            time_and_frame_law: &fixture.time_and_frame_law,
            earth: &fixture.earth,
            units: &fixture.units,
            cases: &fixture.cases,
        }
    }
}

const EXPECTED_PAYLOADS: &[(&str, u64, &str)] = &[
    (
        "assets/reference/orekit_jb2008/README.md",
        1_564,
        "53f29a8d400ec5a982929fb3fc7dd0d971dfaeaa6f065325a58a2f8354738f7a",
    ),
    (
        "assets/reference/orekit_jb2008/licenses/hipparchus/LICENSE.txt",
        24_364,
        "734cdb2a3c7b796bc4cbdb0b63816bb21be811718273659f90cd6e00bdea6323",
    ),
    (
        "assets/reference/orekit_jb2008/licenses/hipparchus/NOTICE.txt",
        358,
        "d3f82a2956ae7ddf9d810baa084df421ec2dfef584648c98ad065996914e52b3",
    ),
    (
        "assets/reference/orekit_jb2008/licenses/orekit/LICENSE.txt",
        11_358,
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    (
        "assets/reference/orekit_jb2008/licenses/orekit/NOTICE.txt",
        1_661,
        "dc8ea1d4aed11cba83d3f38de87518b274260fb529f01072af9d6c553ce8c440",
    ),
    (
        "assets/reference/orekit_jb2008/maven/jars/hipparchus-core-4.0.2.jar",
        1_959_796,
        "7c56992f3af64429d871c33c00808ee5db5d9ed56b395b5d3d31319c4ef7ba0a",
    ),
    (
        "assets/reference/orekit_jb2008/maven/jars/hipparchus-geometry-4.0.2.jar",
        276_263,
        "4e8eede49aabd4fb71f08dd0b8b87297a9e78ed36f05c3caa4e63de5f469cceb",
    ),
    (
        "assets/reference/orekit_jb2008/maven/jars/orekit-13.1.2.jar",
        8_511_771,
        "89c2060c60dbe194a87dddcf3bb8343ebd16733958efe4dcc996cebbbeed655d",
    ),
    (
        "assets/reference/orekit_jb2008/maven/poms/hipparchus-4.0.2.pom",
        38_424,
        "68b08e59cd965ba34a4030ea006d75941983df1536e3a68eaa8bad50ace62894",
    ),
    (
        "assets/reference/orekit_jb2008/maven/poms/hipparchus-core-4.0.2.pom",
        5_605,
        "f509f304a845d07954b9b186b89075d0ac6bb3d8f67e340ec941892b5e223b61",
    ),
    (
        "assets/reference/orekit_jb2008/maven/poms/hipparchus-geometry-4.0.2.pom",
        5_379,
        "0c2cb9190999e74622573ce6a26aa22874b2fa787e0ebb8ab8a7fcae0c43343e",
    ),
    (
        "assets/reference/orekit_jb2008/maven/poms/orekit-13.1.2.pom",
        39_467,
        "1561e6dd4de93969e60abffd7f49754732f3919ae3412c99e2b6647c17bb1173",
    ),
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn manifest_path() -> PathBuf {
    repo_root().join("assets/reference/orekit_jb2008/manifest.json")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/orekit_jb2008_eci_adapter_v1.json")
}

fn digest_hex(digest: &[u8]) -> anyhow::Result<String> {
    let mut encoded = String::new();
    for byte in digest {
        write!(&mut encoded, "{byte:02x}")
            .map_err(|error| anyhow::anyhow!("encode SHA-256 digest: {error}"))?;
    }
    Ok(encoded)
}

fn sha256_hex(bytes: &[u8]) -> anyhow::Result<String> {
    digest_hex(&Sha256::digest(bytes))
}

fn semantic_sha256(domain: &str, payload: &[u8]) -> anyhow::Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    let payload_length = u64::try_from(payload.len())
        .map_err(|error| anyhow::anyhow!("semantic payload length does not fit in u64: {error}"))?;
    hasher.update(payload_length.to_be_bytes());
    hasher.update(payload);
    digest_hex(&hasher.finalize())
}

fn canonical_json(value: &Value, output: &mut Vec<u8>) -> anyhow::Result<()> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(value) => output.extend_from_slice(if *value { b"true" } else { b"false" }),
        Value::Number(value) => output.extend_from_slice(value.to_string().as_bytes()),
        Value::String(value) => output.extend_from_slice(serde_json::to_string(value)?.as_bytes()),
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                canonical_json(value, output)?;
            }
            output.push(b']');
        }
        Value::Object(values) => {
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.as_bytes().cmp(right.as_bytes()));
            output.push(b'{');
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                output.extend_from_slice(serde_json::to_string(key)?.as_bytes());
                output.push(b':');
                canonical_json(value, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn parse_manifest(bytes: &[u8]) -> anyhow::Result<ReferenceManifest> {
    Ok(serde_json::from_slice(bytes)?)
}

fn parse_fixture(bytes: &[u8]) -> anyhow::Result<ReferenceFixture> {
    Ok(serde_json::from_slice(bytes)?)
}

fn validate_manifest_fields(manifest: &ReferenceManifest) -> anyhow::Result<()> {
    require_eq!(manifest.schema, MANIFEST_SCHEMA, "manifest schema");
    require_eq!(
        manifest.authority_label,
        MANIFEST_AUTHORITY,
        "manifest authority"
    );
    require_eq!(
        manifest.claim_scope,
        MANIFEST_CLAIM_SCOPE,
        "manifest claim scope"
    );
    require_eq!(
        manifest.runtime_closure.closure_kind,
        "fixed_generator_reachability_not_full_maven_runtime",
        "closure kind"
    );
    require_eq!(
        manifest.runtime_closure.java_class_major,
        52,
        "Java class major"
    );
    require_eq!(
        manifest.runtime_closure.jar_aggregate_algorithm,
        "sorted_path_nul_decimal_size_nul_contents_v1",
        "JAR aggregate algorithm"
    );
    require_eq!(
        manifest.runtime_closure.jar_aggregate_path_root,
        "maven/jars",
        "JAR aggregate root"
    );
    require_eq!(
        manifest.runtime_closure.jar_regular_file_count,
        3,
        "JAR file count"
    );
    require_eq!(
        manifest.runtime_closure.jar_aggregate_size_bytes,
        10_747_830,
        "JAR aggregate size"
    );
    require_eq!(
        manifest.runtime_closure.jar_aggregate_sha256,
        "7e3b504bfd38b0d6713b959085e7fcfba8a6ae635bf4b769006d816d6b7e7d24",
        "JAR aggregate SHA-256"
    );
    let expected_classpath = [
        "maven/jars/orekit-13.1.2.jar",
        "maven/jars/hipparchus-core-4.0.2.jar",
        "maven/jars/hipparchus-geometry-4.0.2.jar",
    ];
    require_eq!(
        manifest.runtime_closure.classpath_order,
        expected_classpath,
        "classpath order"
    );
    let expected_excluded = ["ode", "fitting", "optim", "filtering", "stat"];
    require_eq!(
        manifest.runtime_closure.excluded_hipparchus_modules,
        expected_excluded,
        "excluded Hipparchus modules"
    );
    require!(
        manifest.runtime_closure.external_data.is_empty(),
        "external data must be empty"
    );

    validate_payload_table(&manifest.payloads)?;
    validate_sources(&manifest.sources)?;

    require_eq!(
        manifest.license_policy.license,
        "Apache-2.0",
        "license identifier"
    );
    require!(
        manifest.license_policy.preserve_embedded_license_and_notice,
        "embedded license and notice must be preserved"
    );
    require!(
        manifest
            .license_policy
            .modified_generator_is_separately_identified,
        "modified generator must be separately identified"
    );
    require_eq!(
        manifest.semantic_sha256,
        MANIFEST_SEMANTIC_SHA256,
        "manifest semantic SHA-256"
    );
    Ok(())
}

fn validate_payload_table(payloads: &[PayloadReceipt]) -> anyhow::Result<()> {
    let mut unique_paths = HashSet::new();
    for payload in payloads {
        require!(
            unique_paths.insert(payload.path.as_str()),
            "duplicate payload path"
        );
        let path = Path::new(&payload.path);
        require!(
            !path.is_absolute()
                && path
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "payload path must be normalized and relative"
        );
    }

    require_eq!(
        payloads.len(),
        EXPECTED_PAYLOADS.len(),
        "payload receipt count"
    );
    for (payload, (path, size_bytes, sha256)) in payloads.iter().zip(EXPECTED_PAYLOADS) {
        require_eq!(payload.path, *path, "payload path");
        require_eq!(payload.size_bytes, *size_bytes, "payload size");
        require_eq!(payload.sha256, *sha256, "payload SHA-256");
    }
    Ok(())
}

fn validate_sources(sources: &[SourceReceipt]) -> anyhow::Result<()> {
    const MAVEN_SOURCES: &[(&str, &str)] = &[
        (
            "assets/reference/orekit_jb2008/maven/jars/orekit-13.1.2.jar",
            "https://repo.maven.apache.org/maven2/org/orekit/orekit/13.1.2/orekit-13.1.2.jar",
        ),
        (
            "assets/reference/orekit_jb2008/maven/jars/hipparchus-core-4.0.2.jar",
            "https://repo.maven.apache.org/maven2/org/hipparchus/hipparchus-core/4.0.2/hipparchus-core-4.0.2.jar",
        ),
        (
            "assets/reference/orekit_jb2008/maven/jars/hipparchus-geometry-4.0.2.jar",
            "https://repo.maven.apache.org/maven2/org/hipparchus/hipparchus-geometry/4.0.2/hipparchus-geometry-4.0.2.jar",
        ),
        (
            "assets/reference/orekit_jb2008/maven/poms/orekit-13.1.2.pom",
            "https://repo.maven.apache.org/maven2/org/orekit/orekit/13.1.2/orekit-13.1.2.pom",
        ),
        (
            "assets/reference/orekit_jb2008/maven/poms/hipparchus-4.0.2.pom",
            "https://repo.maven.apache.org/maven2/org/hipparchus/hipparchus/4.0.2/hipparchus-4.0.2.pom",
        ),
        (
            "assets/reference/orekit_jb2008/maven/poms/hipparchus-core-4.0.2.pom",
            "https://repo.maven.apache.org/maven2/org/hipparchus/hipparchus-core/4.0.2/hipparchus-core-4.0.2.pom",
        ),
        (
            "assets/reference/orekit_jb2008/maven/poms/hipparchus-geometry-4.0.2.pom",
            "https://repo.maven.apache.org/maven2/org/hipparchus/hipparchus-geometry/4.0.2/hipparchus-geometry-4.0.2.pom",
        ),
    ];
    const LEGAL_SOURCES: &[(&str, &str)] = &[
        (
            "assets/reference/orekit_jb2008/licenses/orekit/LICENSE.txt",
            "maven/jars/orekit-13.1.2.jar!/META-INF/LICENSE.txt",
        ),
        (
            "assets/reference/orekit_jb2008/licenses/orekit/NOTICE.txt",
            "maven/jars/orekit-13.1.2.jar!/META-INF/NOTICE.txt",
        ),
        (
            "assets/reference/orekit_jb2008/licenses/hipparchus/LICENSE.txt",
            "maven/jars/hipparchus-core-4.0.2.jar!/META-INF/LICENSE.txt",
        ),
        (
            "assets/reference/orekit_jb2008/licenses/hipparchus/NOTICE.txt",
            "maven/jars/hipparchus-core-4.0.2.jar!/META-INF/NOTICE.txt",
        ),
    ];

    let expected_source_count = MAVEN_SOURCES
        .len()
        .checked_add(LEGAL_SOURCES.len())
        .ok_or_else(|| anyhow::anyhow!("source receipt count overflow"))?;
    require_eq!(sources.len(), expected_source_count, "source receipt count");
    for (source, (path, url)) in sources.iter().zip(MAVEN_SOURCES) {
        require_eq!(source.path, *path, "Maven source path");
        require_eq!(source.url.as_deref(), Some(*url), "Maven source URL");
        require!(
            source.source_entry.is_none(),
            "Maven source must not have JAR entry"
        );
    }
    for (source, (path, source_entry)) in
        sources.iter().skip(MAVEN_SOURCES.len()).zip(LEGAL_SOURCES)
    {
        require_eq!(source.path, *path, "legal source path");
        require!(source.url.is_none(), "legal source must not have Maven URL");
        require_eq!(
            source.source_entry.as_deref(),
            Some(*source_entry),
            "legal source JAR entry"
        );
    }
    Ok(())
}

fn validate_manifest_semantic(manifest: &ReferenceManifest) -> anyhow::Result<()> {
    let mut value = serde_json::to_value(manifest)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("typed manifest did not serialize as object"))?;
    require!(
        object.remove("semantic_sha256").is_some(),
        "typed manifest semantic field absent"
    );
    let mut canonical = Vec::new();
    canonical_json(&value, &mut canonical)?;
    let actual = semantic_sha256(MANIFEST_DOMAIN, &canonical)?;
    require_eq!(
        actual,
        MANIFEST_SEMANTIC_SHA256,
        "manifest recomputed semantic SHA-256"
    );
    require_eq!(
        manifest.semantic_sha256,
        actual,
        "manifest embedded semantic SHA-256"
    );
    Ok(())
}

fn validate_payload_bytes(manifest: &ReferenceManifest, root: &Path) -> anyhow::Result<()> {
    for payload in &manifest.payloads {
        let path = root.join(&payload.path);
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| anyhow::anyhow!("missing payload {}: {error}", payload.path))?;
        require!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "payload must be a regular nonsymlink file"
        );
        require_eq!(
            metadata.len(),
            payload.size_bytes,
            "payload filesystem size"
        );
        let bytes = fs::read(&path)
            .map_err(|error| anyhow::anyhow!("read payload {}: {error}", payload.path))?;
        require_eq!(
            sha256_hex(&bytes)?,
            payload.sha256,
            "payload filesystem SHA-256"
        );
    }
    Ok(())
}

fn collect_regular_files(
    directory: &Path,
    root: &Path,
    files: &mut HashSet<String>,
) -> anyhow::Result<()> {
    for entry in
        fs::read_dir(directory).map_err(|error| anyhow::anyhow!("read asset directory: {error}"))?
    {
        let entry = entry.map_err(|error| anyhow::anyhow!("read asset entry: {error}"))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| anyhow::anyhow!("read asset file type: {error}"))?;
        require!(!file_type.is_symlink(), "asset symlinks are forbidden");
        if file_type.is_dir() {
            collect_regular_files(&path, root, files)?;
        } else {
            require!(file_type.is_file(), "asset payload must be regular");
            let relative = path
                .strip_prefix(root)
                .map_err(|error| anyhow::anyhow!("asset path escaped repository: {error}"))?
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("asset path must be UTF-8"))?
                .to_owned();
            require!(files.insert(relative), "duplicate filesystem payload path");
        }
    }
    Ok(())
}

fn validate_asset_file_set(manifest: &ReferenceManifest, root: &Path) -> anyhow::Result<()> {
    let asset_root = root.join("assets/reference/orekit_jb2008");
    let mut actual = HashSet::new();
    collect_regular_files(&asset_root, root, &mut actual)?;
    let mut expected: HashSet<String> = manifest
        .payloads
        .iter()
        .map(|payload| payload.path.clone())
        .collect();
    expected.insert("assets/reference/orekit_jb2008/manifest.json".to_owned());
    require_eq!(actual, expected, "asset filesystem file set");
    Ok(())
}

fn validate_manifest_bytes(bytes: &[u8], manifest: &ReferenceManifest) -> anyhow::Result<()> {
    require_eq!(
        sha256_hex(bytes)?,
        MANIFEST_RAW_SHA256,
        "manifest raw SHA-256"
    );
    validate_manifest_fields(manifest)?;
    validate_manifest_semantic(manifest)?;
    validate_payload_bytes(manifest, &repo_root())?;
    validate_asset_file_set(manifest, &repo_root())
}

fn validate_fixture_fields(fixture: &ReferenceFixture) -> anyhow::Result<()> {
    require_eq!(fixture.schema, FIXTURE_SCHEMA, "fixture schema");
    require_eq!(fixture.authority, FIXTURE_AUTHORITY, "fixture authority");
    require_eq!(
        fixture.claim_scope,
        FIXTURE_CLAIM_SCOPE,
        "fixture claim scope"
    );
    let exclusions = [
        "not GCRF/ITRF validation",
        "not physical validation",
        "not production Rust adapter conformance",
    ];
    require_eq!(fixture.exclusions, exclusions, "fixture exclusions");

    let provenance = &fixture.provenance;
    require_eq!(
        provenance.generator_source_sha256,
        "ce5fc9f0123a5ca54bc07b4bb89ba0cb8c1a4ab3abb62c9fe2cfbe4a994a5883",
        "generator source SHA-256"
    );
    require_eq!(provenance.orekit_version, "13.1.2", "Orekit version");
    require_eq!(
        provenance.orekit_jar_sha256,
        "89c2060c60dbe194a87dddcf3bb8343ebd16733958efe4dcc996cebbbeed655d",
        "Orekit JAR SHA-256"
    );
    require_eq!(
        provenance.hipparchus_core_version,
        "4.0.2",
        "Hipparchus core version"
    );
    require_eq!(
        provenance.hipparchus_core_jar_sha256,
        "7c56992f3af64429d871c33c00808ee5db5d9ed56b395b5d3d31319c4ef7ba0a",
        "Hipparchus core JAR SHA-256"
    );
    require_eq!(
        provenance.hipparchus_geometry_version,
        "4.0.2",
        "Hipparchus geometry version"
    );
    require_eq!(
        provenance.hipparchus_geometry_jar_sha256,
        "4e8eede49aabd4fb71f08dd0b8b87297a9e78ed36f05c3caa4e63de5f469cceb",
        "Hipparchus geometry JAR SHA-256"
    );
    require_eq!(provenance.java_vendor, "Amazon.com Inc.", "Java vendor");
    require_eq!(provenance.java_version, "11.0.30", "Java version");
    require_eq!(
        provenance.java_vm_name,
        "OpenJDK 64-Bit Server VM",
        "Java VM name"
    );
    require_eq!(
        provenance.java_vm_version,
        "11.0.30+7-LTS",
        "Java VM version"
    );
    require_eq!(
        provenance.java_specification_version,
        "11",
        "Java specification version"
    );
    require_eq!(provenance.os_arch, "aarch64", "OS architecture");
    require_eq!(provenance.file_encoding, "UTF-8", "file encoding");
    require_eq!(
        provenance.compile_flags,
        "javac --release 8 -encoding UTF-8",
        "compile flags"
    );
    require_eq!(
        provenance.runtime_flags,
        "java -Xint -Dfile.encoding=UTF-8 -Duser.language=en -Duser.country=US -Duser.timezone=UTC -Dorekit.data.path=<invalid> -Djava.io.tmpdir=<temp> -Duser.home=<temp> -Djava.util.prefs.userRoot=<temp> -XX:-UsePerfData -XX:+UseSerialGC -Xms32m -Xmx256m -cp <sealed classpath>",
        "runtime flags"
    );

    let canonicalization = &fixture.canonicalization;
    require_eq!(canonicalization.encoding, "UTF-8", "fixture encoding");
    require_eq!(
        canonicalization.json,
        "RFC8259 minified, source-declared key order, LF terminator",
        "fixture JSON contract"
    );
    require_eq!(
        canonicalization.f64,
        "lowercase 16-digit 0x-prefixed raw IEEE-754 binary64 string; signed zero preserved",
        "fixture f64 contract"
    );
    require_eq!(
        canonicalization.semantic_hash_domain,
        FIXTURE_DOMAIN,
        "fixture semantic domain"
    );
    require_eq!(
        canonicalization.semantic_hash_algorithm,
        "sha256(domain_ascii || NUL || big_endian_u64_payload_length || canonical_json_without_semantic_sha256)",
        "fixture semantic algorithm"
    );

    let law = &fixture.time_and_frame_law;
    require_eq!(law.time_scale, "FIXED_UTC_TAI_MINUS_37", "time scale");
    require_eq!(
        law.fixed_utc_offset_from_tai_s.bits(),
        0xc042_8000_0000_0000_u64,
        "fixed UTC offset"
    );
    require_eq!(
        law.leap_second_policy,
        "none; fixed offset over declared corpus",
        "leap policy"
    );
    require_eq!(
        law.eci_frame,
        "Orekit Frame.getRoot identity used as synthetic ECI only",
        "synthetic ECI frame"
    );
    require_eq!(
        law.body_frame,
        "SYNTHETIC_ROTATING_WGS84_BODY",
        "synthetic body frame"
    );
    require_eq!(
        law.transform_scope,
        "position transform; no velocity authority",
        "transform scope"
    );
    require_eq!(
        law.rotation_convention,
        "parent ECI to child body VECTOR_OPERATOR rotation about +Z",
        "rotation convention"
    );
    require_eq!(
        law.angle_law,
        "theta=theta0+omega*durationFrom(reference_epoch)",
        "angle law"
    );
    require_eq!(
        law.reference_epoch_fixed_utc,
        "2025-01-15T00:00:00.000000000",
        "frame reference epoch"
    );
    require_eq!(
        law.theta0_rad.bits(),
        0x3ff3_c0ca_428c_59fb_u64,
        "theta0 bits"
    );
    require_eq!(
        law.omega_rad_s.bits(),
        0x3f13_1da7_d157_db65_u64,
        "omega bits"
    );
    require!(
        law.external_data.is_empty(),
        "fixture external data must be empty"
    );

    require_eq!(fixture.earth.shape, "WGS84 OneAxisEllipsoid", "Earth shape");
    require_eq!(
        fixture.earth.equatorial_radius_m.bits(),
        0x4158_54a6_4000_0000_u64,
        "Earth radius bits"
    );
    require_eq!(
        fixture.earth.flattening.bits(),
        0x3f6b_775a_84f3_e128_u64,
        "Earth flattening bits"
    );
    require_eq!(fixture.units.cartesian, "m", "cartesian units");
    require_eq!(fixture.units.angle, "rad", "angle units");
    require_eq!(fixture.units.altitude, "m", "altitude units");
    require_eq!(fixture.units.density, "kg/m^3", "density units");
    require_eq!(fixture.units.mjd, "fixed-offset UTC days", "MJD units");
    require_eq!(fixture.units.f10, "1e-22 W/(m^2 Hz)", "F10 units");
    require_eq!(
        fixture.units.s10_xm10_y10,
        "JB2008 scaled index units",
        "scaled driver units"
    );
    require_eq!(fixture.units.dstdtc, "K", "DSTDTC units");
    require_eq!(
        fixture.semantic_sha256,
        FIXTURE_SEMANTIC_SHA256,
        "fixture semantic SHA-256"
    );

    validate_fixture_cases(&fixture.cases)
}

const fn driver_bits(drivers: &JbDrivers) -> [u64; 9] {
    [
        drivers.f10.bits(),
        drivers.f10b.bits(),
        drivers.s10.bits(),
        drivers.s10b.bits(),
        drivers.xm10.bits(),
        drivers.xm10b.bits(),
        drivers.y10.bits(),
        drivers.y10b.bits(),
        drivers.dstdtc.bits(),
    ]
}

const fn primitive_driver_bits(inputs: &PrimitiveInputs) -> [u64; 9] {
    [
        inputs.f10.bits(),
        inputs.f10b.bits(),
        inputs.s10.bits(),
        inputs.s10b.bits(),
        inputs.xm10.bits(),
        inputs.xm10b.bits(),
        inputs.y10.bits(),
        inputs.y10b.bits(),
        inputs.dstdtc.bits(),
    ]
}

fn profile_bits(profile: &str) -> anyhow::Result<[u64; 9]> {
    match profile {
        "A" => Ok([
            0x4056_8000_0000_0000,
            0x4059_0000_0000_0000,
            0x4057_c000_0000_0000,
            0x405a_4000_0000_0000,
            0x4059_0000_0000_0000,
            0x405b_8000_0000_0000,
            0x405a_4000_0000_0000,
            0x405c_c000_0000_0000,
            0xc034_0000_0000_0000,
        ]),
        "B" => Ok([
            0x4061_8000_0000_0000,
            0x4060_4000_0000_0000,
            0x4062_c000_0000_0000,
            0x4060_e000_0000_0000,
            0x4062_2000_0000_0000,
            0x4060_8000_0000_0000,
            0x4063_6000_0000_0000,
            0x4061_4000_0000_0000,
            0x404e_0000_0000_0000,
        ]),
        "C" => Ok([
            0x406b_8000_0000_0000,
            0x4066_8000_0000_0000,
            0x4069_a000_0000_0000,
            0x4065_e000_0000_0000,
            0x4068_c000_0000_0000,
            0x4065_4000_0000_0000,
            0x406c_c000_0000_0000,
            0x4067_2000_0000_0000,
            0x4066_8000_0000_0000,
        ]),
        "D" => Ok([
            0x4052_c000_0000_0000,
            0x4055_4000_0000_0000,
            0x4054_0000_0000_0000,
            0x4056_8000_0000_0000,
            0x4053_8000_0000_0000,
            0x4056_0000_0000_0000,
            0x4054_8000_0000_0000,
            0x4057_0000_0000_0000,
            0xc049_0000_0000_0000,
        ]),
        _ => Err(anyhow::anyhow!("unknown driver profile {profile}")),
    }
}

fn expected_case_value<T: Copy>(values: &[T], index: usize, label: &str) -> anyhow::Result<T> {
    values
        .get(index)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("expected {label} missing for fixture case index {index}"))
}

fn validate_fixture_cases(cases: &[FixtureCase]) -> anyhow::Result<()> {
    const EPOCHS: [&str; 15] = [
        "2025-01-15T23:59:59.500000000",
        "2025-01-16T00:00:00.500000000",
        "2025-01-16T06:00:00.000000000",
        "2025-01-16T12:00:00.000000000",
        "2025-01-16T18:00:00.000000000",
        "2025-01-16T23:59:59.500000000",
        "2025-01-17T00:00:00.500000000",
        "2025-01-17T06:00:00.000000000",
        "2025-01-17T12:00:00.000000000",
        "2025-01-17T18:00:00.000000000",
        "2025-01-17T23:59:59.500000000",
        "2025-01-18T00:00:00.500000000",
        "2025-01-18T06:00:00.000000000",
        "2025-01-18T12:00:00.000000000",
        "2025-01-18T18:00:00.000000000",
    ];
    const TAGS: [&str; 15] = [
        "day_boundary_before",
        "day_boundary_after",
        "interior",
        "interior",
        "interior",
        "driver_boundary_before",
        "driver_boundary_after",
        "interior",
        "interior",
        "interior",
        "pre_utc_midnight",
        "post_utc_midnight",
        "interior",
        "interior",
        "interior",
    ];
    const TRANSITIONS: [&str; 15] = [
        "A_to_B", "A_to_B", "none", "none", "none", "B_to_C", "B_to_C", "none", "none", "none",
        "C_to_D", "C_to_D", "none", "none", "none",
    ];
    const PROFILES: [&str; 15] = [
        "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C", "D", "D", "D", "D",
    ];
    const LATITUDES: [u64; 15] = [
        0x0000_0000_0000_0000,
        0x3fe9_21fb_5444_2d18,
        0xbfe9_21fb_5444_2d18,
        0x3ff6_5718_4ae7_4487,
        0xbff6_5718_4ae7_4487,
        0x0000_0000_0000_0000,
        0x3fe9_21fb_5444_2d18,
        0xbfe9_21fb_5444_2d18,
        0x3ff6_5718_4ae7_4487,
        0xbff6_5718_4ae7_4487,
        0x0000_0000_0000_0000,
        0x3fe9_21fb_5444_2d18,
        0xbfe9_21fb_5444_2d18,
        0x3ff6_5718_4ae7_4487,
        0xbff6_5718_4ae7_4487,
    ];
    const LONGITUDES: [u64; 15] = [
        0x0000_0000_0000_0000,
        0x3fe0_c152_382d_7366,
        0xbff0_c152_382d_7366,
        0x3ff9_21fb_5444_2d18,
        0xc000_c152_382d_7366,
        0x4004_f1a6_c638_d03f,
        0xc004_f1a6_c638_d03f,
        0x4000_c152_382d_7366,
        0xbff9_21fb_5444_2d18,
        0x3ff0_c152_382d_7366,
        0xbfe0_c152_382d_7366,
        0x3fd0_c152_382d_7366,
        0x3ff4_f1a6_c638_d03f,
        0xc002_d97c_7f33_21d2,
        0x4007_09d1_0d3e_7eac,
    ];
    const ALTITUDES: [u64; 15] = [
        0x4108_6a00_0000_0000,
        0x4118_6a00_0000_0000,
        0x4128_6a00_0000_0000,
        0x4136_e360_0000_0000,
        0x4108_6a00_0000_0000,
        0x4118_6a00_0000_0000,
        0x4128_6a00_0000_0000,
        0x4136_e360_0000_0000,
        0x4108_6a00_0000_0000,
        0x4118_6a00_0000_0000,
        0x4128_6a00_0000_0000,
        0x4136_e360_0000_0000,
        0x4108_6a00_0000_0000,
        0x4118_6a00_0000_0000,
        0x4128_6a00_0000_0000,
    ];

    require_eq!(cases.len(), 15, "fixture case count");
    for (index, case) in cases.iter().enumerate() {
        let case_number = index
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("fixture case numbering overflow"))?;
        let expected_epoch = expected_case_value(&EPOCHS, index, "epoch")?;
        let expected_tag = expected_case_value(&TAGS, index, "boundary tag")?;
        let expected_transition = expected_case_value(&TRANSITIONS, index, "driver transition")?;
        let expected_profile_name = expected_case_value(&PROFILES, index, "driver profile")?;
        let expected_latitude = expected_case_value(&LATITUDES, index, "latitude")?;
        let expected_longitude = expected_case_value(&LONGITUDES, index, "longitude")?;
        let expected_altitude = expected_case_value(&ALTITUDES, index, "altitude")?;

        require_eq!(case.id, format!("case_{case_number:02}"), "fixture case ID");
        require_eq!(case.epoch.fixed_utc, expected_epoch, "case epoch");
        require_eq!(
            case.epoch.tai_minus_fixed_utc_s.bits(),
            0x4042_8000_0000_0000_u64,
            "case TAI minus fixed UTC"
        );
        require_eq!(case.boundary.tag, expected_tag, "boundary tag");
        require_eq!(
            case.boundary.driver_transition,
            expected_transition,
            "driver transition"
        );
        require_eq!(
            case.boundary.driver_profile_id,
            expected_profile_name,
            "driver profile ID"
        );
        require_eq!(
            case.design.geodetic_latitude_rad.bits(),
            expected_latitude,
            "design latitude"
        );
        require_eq!(
            case.design.geodetic_longitude_rad.bits(),
            expected_longitude,
            "design longitude"
        );
        require_eq!(
            case.design.altitude_m.bits(),
            expected_altitude,
            "design altitude"
        );

        let expected_profile = profile_bits(expected_profile_name)?;
        require_eq!(
            driver_bits(&case.inputs.jb_drivers),
            expected_profile,
            "case driver profile bits"
        );
        require_eq!(
            primitive_driver_bits(&case.expected.jb_primitive_inputs),
            expected_profile,
            "primitive driver profile bits"
        );
        require_eq!(
            case.expected
                .jb_primitive_inputs
                .sun_longitude_rad_as_sun_ra,
            case.expected.sun_geodetic.longitude_rad,
            "Sun longitude primitive mapping"
        );
        require_eq!(
            case.expected
                .jb_primitive_inputs
                .sun_geodetic_latitude_rad_as_sun_decli,
            case.expected.sun_geodetic.latitude_rad,
            "Sun latitude primitive mapping"
        );
        require_eq!(
            case.expected
                .jb_primitive_inputs
                .satellite_geodetic_longitude_rad_as_sat_lon,
            case.expected.satellite_geodetic.longitude_rad,
            "satellite longitude primitive mapping"
        );
        require_eq!(
            case.expected
                .jb_primitive_inputs
                .satellite_geodetic_latitude_rad_as_sat_lat,
            case.expected.satellite_geodetic.latitude_rad,
            "satellite latitude primitive mapping"
        );
        require_eq!(
            case.expected
                .jb_primitive_inputs
                .satellite_ellipsoidal_altitude_m_as_sat_alt,
            case.expected.satellite_geodetic.altitude_m,
            "satellite altitude primitive mapping"
        );
    }
    Ok(())
}

fn validate_fixture_semantic(fixture: &ReferenceFixture) -> anyhow::Result<()> {
    let payload = serde_json::to_vec(&FixtureWithoutSemantic::from(fixture))?;
    let actual = semantic_sha256(FIXTURE_DOMAIN, &payload)?;
    require_eq!(
        actual,
        FIXTURE_SEMANTIC_SHA256,
        "fixture recomputed semantic SHA-256"
    );
    require_eq!(
        fixture.semantic_sha256,
        actual,
        "fixture embedded semantic SHA-256"
    );
    Ok(())
}

fn validate_fixture_bytes(bytes: &[u8], fixture: &ReferenceFixture) -> anyhow::Result<()> {
    require_eq!(
        sha256_hex(bytes)?,
        FIXTURE_RAW_SHA256,
        "fixture raw SHA-256"
    );
    validate_fixture_fields(fixture)?;
    validate_fixture_semantic(fixture)
}

const fn primitive_input(case: &FixtureCase) -> Jb2008Input {
    let input = &case.expected.jb_primitive_inputs;
    Jb2008Input {
        mjd_utc: input.date_mjd_fixed_utc.as_f64(),
        sun_declination_rad: input.sun_geodetic_latitude_rad_as_sun_decli.as_f64(),
        // The fixture still carries both right ascensions -- it is the sealed
        // Orekit record and its columns are not ours to change. The kernel takes
        // their difference, so the difference is formed here instead of inside
        // the kernel. Both columns stay referenced, so a fixture swap that moved
        // either one still reaches the comparison.
        hour_angle_rad: input.satellite_geodetic_longitude_rad_as_sat_lon.as_f64()
            - input.sun_longitude_rad_as_sun_ra.as_f64(),
        sat_geocentric_lat_rad: input.satellite_geodetic_latitude_rad_as_sat_lat.as_f64(),
        sat_altitude_m: input.satellite_ellipsoidal_altitude_m_as_sat_alt.as_f64(),
        f10: input.f10.as_f64(),
        f10b: input.f10b.as_f64(),
        s10: input.s10.as_f64(),
        s10b: input.s10b.as_f64(),
        m10: input.xm10.as_f64(),
        m10b: input.xm10b.as_f64(),
        y10: input.y10.as_f64(),
        y10b: input.y10b.as_f64(),
        dst_temperature_correction_k: input.dstdtc.as_f64(),
    }
}

fn replace_once(raw: &str, needle: &str, replacement: &str) -> String {
    let changed = raw.replacen(needle, replacement, 1);
    assert_ne!(changed, raw, "negative probe needle must exist: {needle}");
    changed
}

#[test]
fn jb2008_reference_manifest_is_strict_and_sealed() {
    let raw = fs::read(manifest_path()).expect("read sealed Orekit JB2008 manifest");
    let manifest = parse_manifest(&raw).expect("parse typed Orekit JB2008 manifest");
    validate_manifest_bytes(&raw, &manifest).expect("validate sealed Orekit JB2008 manifest");

    let text = std::str::from_utf8(&raw).expect("manifest is UTF-8");
    let duplicate_root = replace_once(
        text,
        "{\n",
        "{\n  \"schema\": \"part_a_orekit_jb2008_manifest_v1\",\n",
    );
    assert!(
        parse_manifest(duplicate_root.as_bytes()).is_err(),
        "duplicate root key must fail typed parse"
    );
    let duplicate_nested = replace_once(
        text,
        "\"runtime_closure\": {",
        "\"runtime_closure\": {\n    \"closure_kind\": \"duplicate\",",
    );
    assert!(
        parse_manifest(duplicate_nested.as_bytes()).is_err(),
        "duplicate nested key must fail typed parse"
    );
    let unknown_key = replace_once(text, "{\n", "{\n  \"unknown\": null,\n");
    assert!(
        parse_manifest(unknown_key.as_bytes()).is_err(),
        "unknown key must fail typed parse"
    );

    let mut duplicate_path = manifest.clone();
    let Some(first_payload) = duplicate_path.payloads.first().cloned() else {
        panic!("sealed manifest must contain a payload");
    };
    duplicate_path.payloads.push(first_payload);
    assert!(
        validate_payload_table(&duplicate_path.payloads).is_err(),
        "duplicate payload path must fail validation"
    );
    let mut missing_payload = manifest.clone();
    missing_payload.payloads.pop();
    assert!(
        validate_payload_table(&missing_payload.payloads).is_err(),
        "missing payload receipt must fail validation"
    );
    assert!(
        validate_payload_bytes(&manifest, Path::new("/definitely/not/a/repository")).is_err(),
        "missing payload file must fail validation"
    );
    let mut wrong_payload = manifest.clone();
    let Some(first_payload) = wrong_payload.payloads.first_mut() else {
        panic!("sealed manifest must contain a payload");
    };
    let Some(next_size_bytes) = first_payload.size_bytes.checked_add(1) else {
        panic!("sealed manifest payload size cannot overflow");
    };
    first_payload.size_bytes = next_size_bytes;
    assert!(
        validate_payload_table(&wrong_payload.payloads).is_err(),
        "wrong payload receipt must fail validation"
    );
    let mut wrong_field = manifest.clone();
    wrong_field.schema = "wrong".to_owned();
    assert!(
        validate_manifest_fields(&wrong_field).is_err(),
        "wrong manifest field must fail validation"
    );
    let mut wrong_semantic = manifest;
    wrong_semantic.semantic_sha256 = "0".repeat(64);
    assert!(
        validate_manifest_semantic(&wrong_semantic).is_err(),
        "wrong manifest semantic hash must fail validation"
    );
    let mut wrong_raw = raw.clone();
    wrong_raw.push(b' ');
    let parsed_wrong_raw = parse_manifest(&wrong_raw).expect("raw-hash probe remains typed JSON");
    assert!(
        validate_manifest_bytes(&wrong_raw, &parsed_wrong_raw).is_err(),
        "wrong manifest raw hash must fail validation"
    );
}

#[test]
fn orekit_synthetic_mapping_fixture_is_strict_and_sealed() {
    let raw = fs::read(fixture_path()).expect("read sealed Orekit JB2008 fixture");
    let fixture = parse_fixture(&raw).expect("parse typed Orekit JB2008 fixture");
    validate_fixture_bytes(&raw, &fixture).expect("validate sealed Orekit JB2008 fixture");

    let text = std::str::from_utf8(&raw).expect("fixture is UTF-8");
    let duplicate_root = replace_once(
        text,
        "{\"schema\":",
        "{\"schema\":\"duplicate\",\"schema\":",
    );
    assert!(
        parse_fixture(duplicate_root.as_bytes()).is_err(),
        "duplicate fixture root key must fail typed parse"
    );
    let duplicate_nested = replace_once(
        text,
        "\"earth\":{\"shape\":",
        "\"earth\":{\"shape\":\"duplicate\",\"shape\":",
    );
    assert!(
        parse_fixture(duplicate_nested.as_bytes()).is_err(),
        "duplicate fixture nested key must fail typed parse"
    );
    let unknown_key = replace_once(
        text,
        "\"earth\":{\"shape\":",
        "\"earth\":{\"unknown\":null,\"shape\":",
    );
    assert!(
        parse_fixture(unknown_key.as_bytes()).is_err(),
        "unknown fixture key must fail typed parse"
    );
    let nonhex = replace_once(text, "0xc042800000000000", "not-hex");
    assert!(
        parse_fixture(nonhex.as_bytes()).is_err(),
        "nonhex binary64 must fail typed parse"
    );
    let nonfinite = replace_once(text, "0xc042800000000000", "0x7ff0000000000000");
    assert!(
        parse_fixture(nonfinite.as_bytes()).is_err(),
        "nonfinite binary64 must fail typed parse"
    );

    let mut wrong_field = fixture.clone();
    wrong_field.schema = "wrong".to_owned();
    assert!(
        validate_fixture_fields(&wrong_field).is_err(),
        "wrong fixture field must fail validation"
    );
    let mut wrong_semantic = fixture;
    wrong_semantic.semantic_sha256 = "0".repeat(64);
    assert!(
        validate_fixture_semantic(&wrong_semantic).is_err(),
        "wrong fixture semantic hash must fail validation"
    );
    let mut wrong_raw = raw.clone();
    wrong_raw.push(b' ');
    let parsed_wrong_raw = parse_fixture(&wrong_raw).expect("raw-hash probe remains typed JSON");
    assert!(
        validate_fixture_bytes(&wrong_raw, &parsed_wrong_raw).is_err(),
        "wrong fixture raw hash must fail validation"
    );
}

#[test]
fn orekit_synthetic_mapping_matches_rust_primitive_kernel() {
    let raw = fs::read(fixture_path()).expect("read sealed Orekit JB2008 fixture");
    let fixture = parse_fixture(&raw).expect("parse typed Orekit JB2008 fixture");
    validate_fixture_bytes(&raw, &fixture).expect("validate sealed Orekit JB2008 fixture");

    let mut mismatches = Vec::new();
    for case in &fixture.cases {
        let actual = jb2008_density(primitive_input(case))
            .unwrap_or_else(|error| panic!("{} Rust JB2008 failed: {error:?}", case.id));
        let expected_bits = case.expected.density_kg_m3.bits();
        if actual.to_bits() != expected_bits {
            mismatches.push(format!(
                "{} actual=0x{:016x} expected=0x{expected_bits:016x}",
                case.id,
                actual.to_bits()
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "Rust primitive kernel differs from Orekit bits:\n{}",
        mismatches.join("\n")
    );
}
