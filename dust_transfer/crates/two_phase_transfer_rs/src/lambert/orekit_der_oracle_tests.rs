//! Independent Lambert validation against frozen Orekit/Der published vectors.

use super::izzo2015_impl;
use anyhow::Context;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const FIXTURE: &str = include_str!("orekit_lambert_der_example1.json");

#[derive(Clone, Deserialize)]
struct OracleFixture {
    schema_version: u32,
    authority_label: String,
    source: SourceReceipt,
    transformations: Transformations,
    semantic_hashes: SemanticHashes,
    inputs: Inputs,
    tolerance: Tolerance,
    solutions: Vec<Solution>,
}

#[derive(Clone, Deserialize)]
struct SourceReceipt {
    project: String,
    version: String,
    commit: String,
    tree: String,
    published_reference: String,
    test_file: String,
    test_method: String,
    test_file_sha256: String,
    solver_file: String,
    solver_file_sha256: String,
    path_type_file: String,
    path_type_file_sha256: String,
}

#[derive(Clone, Deserialize)]
struct Transformations {
    position: String,
    velocity: String,
    gravitational_parameter: String,
}

#[derive(Clone, Deserialize)]
struct SemanticHashes {
    algorithm: String,
    canonical_format: String,
    canonical_byte_format: CanonicalByteFormat,
    input_sha256: String,
    ordered_output_sha256: String,
}

#[derive(Clone, Deserialize)]
struct CanonicalByteFormat {
    input_bytes: String,
    ordered_output_bytes: String,
    float_encoding: String,
    integer_encoding: String,
    boolean_encoding: String,
    string_encoding: String,
    solution_order: Vec<String>,
}

#[derive(Clone, Deserialize)]
struct Inputs {
    mu_km3_s2: f64,
    r1_km: [f64; 3],
    r2_km: [f64; 3],
    tof_s: f64,
}

#[derive(Clone, Copy, Deserialize)]
struct Tolerance {
    component_atol_km_s: f64,
    rtol: f64,
}

#[derive(Clone, Deserialize)]
struct Solution {
    branch_id: String,
    orekit_solution_index: usize,
    orekit_solution_kind: String,
    prograde: bool,
    revolutions: i32,
    low_path: bool,
    expected_v1_km_s: [f64; 3],
    expected_v2_km_s: [f64; 3],
}

const INPUT_DOMAIN: &[u8] = b"orekit-lambert-der-example1:input:v1\0";
const OUTPUT_DOMAIN: &[u8] = b"orekit-lambert-der-example1:ordered-output:v1\0";
const EXPECTED_INPUT_SHA256: &str =
    "f1426b00cefc48acea602e9a8c5feb8126fb277a644ac8d0121033301c3fb183";
const EXPECTED_OUTPUT_SHA256: &str =
    "f024d3b36bd9ba33a4079c9ba6271abcfa3f58713129dce336bbade6ba1c198f";

fn append_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_bits().to_be_bytes());
}

fn append_utf8(bytes: &mut Vec<u8>, value: &str) -> anyhow::Result<()> {
    let length = u32::try_from(value.len())
        .map_err(|_| anyhow::anyhow!("canonical string length exceeds u32"))?;
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn canonical_input_bytes(inputs: &Inputs) -> Vec<u8> {
    let mut bytes = INPUT_DOMAIN.to_vec();
    for value in [
        inputs.mu_km3_s2,
        inputs.r1_km[0],
        inputs.r1_km[1],
        inputs.r1_km[2],
        inputs.r2_km[0],
        inputs.r2_km[1],
        inputs.r2_km[2],
        inputs.tof_s,
    ] {
        append_f64(&mut bytes, value);
    }
    bytes
}

fn canonical_ordered_output_bytes(solutions: &[Solution]) -> anyhow::Result<Vec<u8>> {
    let mut bytes = OUTPUT_DOMAIN.to_vec();
    let count = u32::try_from(solutions.len())
        .map_err(|_| anyhow::anyhow!("solution count exceeds u32"))?;
    bytes.extend_from_slice(&count.to_be_bytes());
    for solution in solutions {
        append_utf8(&mut bytes, &solution.branch_id)?;
        let solution_index = u32::try_from(solution.orekit_solution_index)
            .map_err(|_| anyhow::anyhow!("solution index exceeds u32"))?;
        bytes.extend_from_slice(&solution_index.to_be_bytes());
        append_utf8(&mut bytes, &solution.orekit_solution_kind)?;
        bytes.push(u8::from(solution.prograde));
        bytes.extend_from_slice(&solution.revolutions.to_be_bytes());
        bytes.push(u8::from(solution.low_path));
        for value in solution
            .expected_v1_km_s
            .into_iter()
            .chain(solution.expected_v2_km_s)
        {
            append_f64(&mut bytes, value);
        }
    }
    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut output, byte| {
            let high = u32::from(*byte >> 4);
            let low = u32::from(*byte & 0x0f);
            if let Some(digit) = char::from_digit(high, 16) {
                output.push(digit);
            }
            if let Some(digit) = char::from_digit(low, 16) {
                output.push(digit);
            }
            output
        })
}

fn compare_component(actual: f64, expected: f64, atol: f64, label: &str) -> anyhow::Result<()> {
    if !actual.is_finite() || !expected.is_finite() || !atol.is_finite() || atol < 0.0 {
        return Err(anyhow::anyhow!(
            "{label} requires finite actual, expected, and nonnegative atol"
        ));
    }
    let difference = (actual - expected).abs();
    if !difference.is_finite() {
        return Err(anyhow::anyhow!("{label} produced nonfinite difference"));
    }
    if difference <= atol {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "{label} difference {difference:.12e} exceeds {atol:.12e}"
        ))
    }
}

fn validate(fixture: &OracleFixture) -> anyhow::Result<()> {
    if fixture.schema_version != 1
        || fixture.authority_label
            != "frozen published Orekit/Der vectors; Orekit runtime not executed"
    {
        return Err(anyhow::anyhow!("unexpected schema or authority label"));
    }
    if fixture.source.project != "Orekit"
        || fixture.source.version != "14.0-SNAPSHOT"
        || fixture.source.commit != "080def3a660d1753a5c5b19f225443f712a8e683"
        || fixture.source.tree != "bb555b4390714aef66c0af7d64b6f483d3df7d5c"
        || fixture.source.published_reference
            != "https://amostech.com/TechnicalPapers/2011/Poster/DER.pdf"
        || fixture.source.test_file
            != "src/test/java/org/orekit/control/heuristics/lambert/LambertSolverTest.java"
        || fixture.source.test_method != "testLambertDerExample1"
        || fixture.source.test_file_sha256
            != "9d3c69f7c9a965aa6fa6b05e6fca8ed54e9ce2150a99a024c63f5585ae123c38"
        || fixture.source.solver_file
            != "src/main/java/org/orekit/control/heuristics/lambert/LambertSolver.java"
        || fixture.source.solver_file_sha256
            != "eb853403040d58ded55e180038b4a549f3d1118509e6854ada0a1cad1b4fddfc"
        || fixture.source.path_type_file
            != "src/main/java/org/orekit/control/heuristics/lambert/LambertPathType.java"
        || fixture.source.path_type_file_sha256
            != "593210d2db444ce2959ee5cd624f5f2bf020c7f30044fb0840a914a7f47063b0"
    {
        return Err(anyhow::anyhow!("source receipt mismatch"));
    }
    if fixture.transformations.position != "Orekit m * 1e-3 = fixture km"
        || fixture.transformations.velocity != "Orekit m/s * 1e-3 = fixture km/s"
        || fixture.transformations.gravitational_parameter
            != "Orekit m^3/s^2 * 1e-9 = fixture km^3/s^2"
    {
        return Err(anyhow::anyhow!("unit transformation mismatch"));
    }
    if fixture.semantic_hashes.algorithm != "SHA-256"
        || fixture.semantic_hashes.canonical_format != "orekit-lambert-der-example1-ieee754be-v1"
    {
        return Err(anyhow::anyhow!("semantic hash contract mismatch"));
    }
    let format = &fixture.semantic_hashes.canonical_byte_format;
    if format.input_bytes
        != "ASCII(orekit-lambert-der-example1:input:v1) || 0x00 || f64be(mu_km3_s2,r1_km[0..3],r2_km[0..3],tof_s)"
        || format.ordered_output_bytes
            != "ASCII(orekit-lambert-der-example1:ordered-output:v1) || 0x00 || u32be(solution_count) || each listed solution: utf8_u32be_len(branch_id) || u32be(orekit_solution_index) || utf8_u32be_len(orekit_solution_kind) || u8(prograde) || i32be(revolutions) || u8(low_path) || f64be(expected_v1_km_s[0..3],expected_v2_km_s[0..3])"
        || format.float_encoding
            != "IEEE-754 binary64 bits, big-endian; signed zero preserved"
        || format.integer_encoding != "u32 and i32 two's-complement, big-endian"
        || format.boolean_encoding != "false=0x00; true=0x01"
        || format.string_encoding
            != "u32 big-endian UTF-8 byte length followed by exact UTF-8 bytes"
    {
        return Err(anyhow::anyhow!("canonical byte-format description mismatch"));
    }
    let input_sha256 = sha256_hex(&canonical_input_bytes(&fixture.inputs));
    if fixture.semantic_hashes.input_sha256 != EXPECTED_INPUT_SHA256
        || input_sha256 != EXPECTED_INPUT_SHA256
    {
        return Err(anyhow::anyhow!(
            "input semantic SHA-256 mismatch: authority {EXPECTED_INPUT_SHA256}, stored {}, computed {input_sha256}",
            fixture.semantic_hashes.input_sha256,
        ));
    }
    let ordered_output_sha256 = sha256_hex(&canonical_ordered_output_bytes(&fixture.solutions)?);
    if fixture.semantic_hashes.ordered_output_sha256 != EXPECTED_OUTPUT_SHA256
        || ordered_output_sha256 != EXPECTED_OUTPUT_SHA256
    {
        return Err(anyhow::anyhow!(
            "ordered-output semantic SHA-256 mismatch: authority {EXPECTED_OUTPUT_SHA256}, stored {}, computed {ordered_output_sha256}",
            fixture.semantic_hashes.ordered_output_sha256,
        ));
    }
    if fixture.tolerance.rtol.to_bits() != 0.0_f64.to_bits()
        || fixture.tolerance.component_atol_km_s.to_bits() != 1e-6_f64.to_bits()
    {
        return Err(anyhow::anyhow!("oracle tolerance mismatch"));
    }
    if fixture.inputs.mu_km3_s2.to_bits() != 398_600.435_507_f64.to_bits()
        || fixture.inputs.r1_km.map(f64::to_bits)
            != [22_592.145_603, -1_599.915_239, -19_783.950_506].map(f64::to_bits)
        || fixture.inputs.r2_km.map(f64::to_bits)
            != [1_922.067_697, 4_054.157_051, -8_925.727_465].map(f64::to_bits)
        || fixture.inputs.tof_s.to_bits() != 36_000.0_f64.to_bits()
    {
        return Err(anyhow::anyhow!("oracle input mismatch"));
    }
    if fixture.solutions.len() != 6 {
        return Err(anyhow::anyhow!(
            "expected six branches, got {}",
            fixture.solutions.len()
        ));
    }

    let expected_ids = [
        "prograde-m0",
        "prograde-m1-low",
        "prograde-m1-high",
        "retrograde-m0",
        "retrograde-m1-low",
        "retrograde-m1-high",
    ];
    if format.solution_order != expected_ids {
        return Err(anyhow::anyhow!("canonical solution order mismatch"));
    }
    if !fixture
        .solutions
        .iter()
        .map(|solution| solution.branch_id.as_str())
        .eq(expected_ids)
    {
        return Err(anyhow::anyhow!("ordered branch coverage mismatch"));
    }

    for solution in &fixture.solutions {
        let expected_mapping = match solution.branch_id.as_str() {
            "prograde-m0" => (0, "M0_UNIQUE_SOLUTION", true, 0, true),
            "prograde-m1-low" => (1, "LOW_PATH", true, 1, true),
            "prograde-m1-high" => (2, "HIGH_PATH", true, 1, false),
            "retrograde-m0" => (0, "M0_UNIQUE_SOLUTION", false, 0, true),
            "retrograde-m1-low" => (1, "LOW_PATH", false, 1, true),
            "retrograde-m1-high" => (2, "HIGH_PATH", false, 1, false),
            _ => return Err(anyhow::anyhow!("unknown branch ID")),
        };
        let actual_mapping = (
            solution.orekit_solution_index,
            solution.orekit_solution_kind.as_str(),
            solution.prograde,
            solution.revolutions,
            solution.low_path,
        );
        if actual_mapping != expected_mapping {
            return Err(anyhow::anyhow!(
                "{} branch mapping mismatch",
                solution.branch_id
            ));
        }

        let actual = izzo2015_impl(
            fixture.inputs.mu_km3_s2,
            &fixture.inputs.r1_km,
            &fixture.inputs.r2_km,
            fixture.inputs.tof_s,
            solution.revolutions,
            solution.prograde,
            solution.low_path,
            25,
            1e-12,
            0.0,
        );
        if !actual.success {
            return Err(anyhow::anyhow!("{} solve failed", solution.branch_id));
        }
        for (name, calculated, expected) in [
            ("v1", actual.v1, solution.expected_v1_km_s),
            ("v2", actual.v2, solution.expected_v2_km_s),
        ] {
            for (component, (calculated_component, expected_component)) in
                calculated.into_iter().zip(expected).enumerate()
            {
                compare_component(
                    calculated_component,
                    expected_component,
                    fixture.tolerance.component_atol_km_s,
                    &format!("{} {name}[{component}]", solution.branch_id),
                )?;
            }
        }
    }
    Ok(())
}

fn fixture() -> anyhow::Result<OracleFixture> {
    serde_json::from_str(FIXTURE).context("parsing frozen Orekit fixture")
}

#[test]
fn matches_all_frozen_orekit_der_example1_branches() -> anyhow::Result<()> {
    validate(&fixture()?)
}

#[test]
fn rejects_hostile_expected_component_perturbation() -> anyhow::Result<()> {
    let mut perturbed = fixture()?;
    let solution = perturbed
        .solutions
        .first_mut()
        .ok_or_else(|| anyhow::anyhow!("fixture has no solutions"))?;
    solution.expected_v1_km_s[0] += 2e-6;
    validate(&perturbed).map_or(Ok(()), |()| {
        Err(anyhow::anyhow!(
            "2e-6 hostile perturbation passed validation"
        ))
    })
}

#[test]
fn rejects_hostile_orekit_branch_remapping() -> anyhow::Result<()> {
    let mut perturbed = fixture()?;
    let solution = perturbed
        .solutions
        .get_mut(1)
        .ok_or_else(|| anyhow::anyhow!("fixture lacks second solution"))?;
    solution.orekit_solution_index = 2;
    validate(&perturbed).map_or(Ok(()), |()| {
        Err(anyhow::anyhow!(
            "hostile Orekit branch remapping passed validation"
        ))
    })
}

#[test]
fn rejects_hostile_semantic_hash_tamper() -> anyhow::Result<()> {
    let mut perturbed = fixture()?;
    let hash_tail = perturbed
        .semantic_hashes
        .input_sha256
        .chars()
        .skip(1)
        .collect::<String>();
    perturbed.semantic_hashes.input_sha256 = format!("0{hash_tail}");
    validate(&perturbed).map_or(Ok(()), |()| {
        Err(anyhow::anyhow!(
            "stored semantic input hash tamper passed validation"
        ))
    })
}

#[test]
fn rejects_output_mutation_even_with_recomputed_stored_hash() -> anyhow::Result<()> {
    let mut perturbed = fixture()?;
    let solution = perturbed
        .solutions
        .first_mut()
        .ok_or_else(|| anyhow::anyhow!("fixture has no solutions"))?;
    solution.expected_v1_km_s[0] += 5e-7;
    perturbed.semantic_hashes.ordered_output_sha256 =
        sha256_hex(&canonical_ordered_output_bytes(&perturbed.solutions)?);
    validate(&perturbed).map_or(Ok(()), |()| {
        Err(anyhow::anyhow!(
            "published output mutation passed validation"
        ))
    })
}

#[test]
fn rejects_nonfinite_expected_component_with_recomputed_stored_hash() -> anyhow::Result<()> {
    let mut perturbed = fixture()?;
    let solution = perturbed
        .solutions
        .first_mut()
        .ok_or_else(|| anyhow::anyhow!("fixture has no solutions"))?;
    solution.expected_v1_km_s[0] = f64::NAN;
    perturbed.semantic_hashes.ordered_output_sha256 =
        sha256_hex(&canonical_ordered_output_bytes(&perturbed.solutions)?);
    validate(&perturbed).map_or(Ok(()), |()| {
        Err(anyhow::anyhow!("NaN expected component passed validation"))
    })
}

#[test]
fn comparator_accepts_exact_absolute_tolerance_boundary() {
    let atol = 1e-6;
    assert!(compare_component(atol, 0.0, atol, "boundary").is_ok());
}

#[test]
fn comparator_rejects_next_value_above_absolute_tolerance() {
    let atol = 1e-6_f64;
    let above = f64::from_bits(atol.to_bits() + 1);
    assert!(compare_component(above, 0.0, atol, "above-boundary").is_err());
}

#[test]
fn comparator_rejects_nonfinite_values_and_difference() {
    let atol = 1e-6;
    assert!(compare_component(f64::NAN, 0.0, atol, "nan-actual").is_err());
    assert!(compare_component(0.0, f64::NAN, atol, "nan-expected").is_err());
    assert!(compare_component(0.0, 0.0, f64::NAN, "nan-atol").is_err());
    assert!(compare_component(f64::MAX, -f64::MAX, atol, "infinite-difference").is_err());
}
