use crate::oxymoo::validation::{
    checked_product, checked_sum, validate_probability, validate_variables,
};
use crate::oxymoo::{VariableKind, VariableSpec};
use anyhow::{bail, Context, Result};
use num_traits::ToPrimitive;
use rand::RngExt;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::cmp::Ordering;

/// Compare two validated population rows under NSGA-II crowded comparison.
///
/// # Errors
///
/// Returns an error when either rank or crowding row is missing.
pub fn crowded_better(
    candidate: usize,
    incumbent: usize,
    ranks: &[usize],
    crowding: &[f64],
    rng: &mut Xoshiro256PlusPlus,
) -> Result<bool> {
    if ranks.len() != crowding.len() {
        bail!(
            "crowded comparison rank/crowding length mismatch: ranks={}, crowding={}",
            ranks.len(),
            crowding.len()
        );
    }
    let candidate_rank = *ranks
        .get(candidate)
        .ok_or_else(|| anyhow::anyhow!("candidate rank {candidate} is out of bounds"))?;
    let incumbent_rank = *ranks
        .get(incumbent)
        .ok_or_else(|| anyhow::anyhow!("incumbent rank {incumbent} is out of bounds"))?;
    match candidate_rank.cmp(&incumbent_rank) {
        Ordering::Less => Ok(true),
        Ordering::Greater => Ok(false),
        Ordering::Equal => {
            let candidate_crowding = *crowding.get(candidate).ok_or_else(|| {
                anyhow::anyhow!("candidate crowding {candidate} is out of bounds")
            })?;
            let incumbent_crowding = *crowding.get(incumbent).ok_or_else(|| {
                anyhow::anyhow!("incumbent crowding {incumbent} is out of bounds")
            })?;
            Ok(match candidate_crowding.total_cmp(&incumbent_crowding) {
                Ordering::Greater => true,
                Ordering::Less => false,
                Ordering::Equal => rng.random_bool(0.5),
            })
        }
    }
}

/// Produce two simulated-binary-crossover children from validated parents.
///
/// # Errors
///
/// Returns an error when a parent or child shape differs from `variables`.
pub fn sbx_crossover_into(
    parent1: &[f64],
    parent2: &[f64],
    variables: &[VariableSpec],
    crossover_probability: f64,
    eta_c: f64,
    rng: &mut Xoshiro256PlusPlus,
    child1: &mut Vec<f64>,
    child2: &mut Vec<f64>,
) -> Result<()> {
    if parent1.len() != parent2.len() || parent1.len() != variables.len() {
        bail!(
            "SBX parent shape mismatch: left={}, right={}, variables={}",
            parent1.len(),
            parent2.len(),
            variables.len()
        );
    }
    validate_variables(variables)?;
    validate_probability("crossover_probability", crossover_probability)?;
    if !eta_c.is_finite() || eta_c <= 0.0 {
        bail!("SBX distribution index must be finite and positive");
    }
    child1.clear();
    child2.clear();
    child1
        .try_reserve(parent1.len())
        .context("SBX first-child allocation failed")?;
    child2
        .try_reserve(parent2.len())
        .context("SBX second-child allocation failed")?;
    child1.extend_from_slice(parent1);
    child2.extend_from_slice(parent2);
    if rng.random_range(0.0..1.0) > crossover_probability {
        repair_decision_validated(child1, variables);
        repair_decision_validated(child2, variables);
        return Ok(());
    }

    for (((child1_value, child2_value), (&parent1_value, &parent2_value)), spec) in child1
        .iter_mut()
        .zip(child2.iter_mut())
        .zip(parent1.iter().zip(parent2.iter()))
        .zip(variables.iter())
    {
        if spec.upper <= spec.lower {
            *child1_value = spec.lower;
            *child2_value = spec.lower;
            continue;
        }
        if !rng.random_bool(0.5) {
            continue;
        }
        if (parent1_value - parent2_value).abs() <= 1e-14 {
            continue;
        }
        let lower_parent = parent1_value.min(parent2_value);
        let upper_parent = parent1_value.max(parent2_value);
        let spread = upper_parent - lower_parent;
        let random = rng.random_range(0.0..1.0);
        let inverse_eta = 1.0 / (eta_c + 1.0);

        let lower_beta = (1.0 + 2.0 * (lower_parent - spec.lower) / spread).max(1e-14);
        let lower_alpha = 2.0 - lower_beta.powf(-(eta_c + 1.0));
        let lower_beta_q = if random <= 1.0 / lower_alpha {
            (random * lower_alpha).powf(inverse_eta)
        } else {
            (1.0 / (2.0 - random * lower_alpha)).powf(inverse_eta)
        };
        let first_child = 0.5 * ((lower_parent + upper_parent) - lower_beta_q * spread);

        let upper_beta = (1.0 + 2.0 * (spec.upper - upper_parent) / spread).max(1e-14);
        let upper_alpha = 2.0 - upper_beta.powf(-(eta_c + 1.0));
        let upper_beta_q = if random <= 1.0 / upper_alpha {
            (random * upper_alpha).powf(inverse_eta)
        } else {
            (1.0 / (2.0 - random * upper_alpha)).powf(inverse_eta)
        };
        let second_child = 0.5 * ((lower_parent + upper_parent) + upper_beta_q * spread);

        if rng.random_bool(0.5) {
            *child1_value = second_child;
            *child2_value = first_child;
        } else {
            *child1_value = first_child;
            *child2_value = second_child;
        }
    }

    repair_decision_validated(child1, variables);
    repair_decision_validated(child2, variables);
    Ok(())
}

/// Mutate one validated decision row in place.
///
/// # Errors
///
/// Returns an error when the decision shape differs from `variables`.
pub fn polynomial_mutation(
    decision: &mut [f64],
    variables: &[VariableSpec],
    mutation_probability: f64,
    eta_m: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<()> {
    if decision.len() != variables.len() {
        bail!(
            "polynomial mutation decision shape mismatch: decision={}, variables={}",
            decision.len(),
            variables.len()
        );
    }
    validate_variables(variables)?;
    validate_probability("mutation_probability", mutation_probability)?;
    if !eta_m.is_finite() || eta_m <= 0.0 {
        bail!("polynomial mutation distribution index must be finite and positive");
    }
    for (value, spec) in decision.iter_mut().zip(variables.iter()) {
        if rng.random_range(0.0..1.0) > mutation_probability {
            continue;
        }
        if spec.upper <= spec.lower {
            *value = spec.lower;
            continue;
        }

        let original = repair_value(*value, spec);
        let repaired = original.clamp(spec.lower, spec.upper);
        let span = spec.upper - spec.lower;
        let delta1 = ((repaired - spec.lower) / span).clamp(0.0, 1.0);
        let delta2 = ((spec.upper - repaired) / span).clamp(0.0, 1.0);
        let random = rng.random_range(0.0..1.0);
        let mutation_power = 1.0 / (eta_m + 1.0);
        let delta_q = if random < 0.5 {
            let xy = 1.0 - delta1;
            let value = 2.0 * random + (1.0 - 2.0 * random) * xy.powf(eta_m + 1.0);
            value.powf(mutation_power) - 1.0
        } else {
            let xy = 1.0 - delta2;
            let value = 2.0 * (1.0 - random) + 2.0 * (random - 0.5) * xy.powf(eta_m + 1.0);
            1.0 - value.powf(mutation_power)
        };
        *value = (repaired + delta_q * span).clamp(spec.lower, spec.upper);

        if spec.kind == VariableKind::Integer {
            let original_integer = original.round();
            let mutated_integer = value.round();
            if (original_integer - mutated_integer).abs() < f64::EPSILON && rng.random_bool(0.5) {
                let direction = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
                let lower = spec.lower.ceil();
                let upper = spec.upper.floor();
                let mut candidate = original_integer + direction;
                if candidate < lower {
                    candidate = original_integer + 1.0;
                }
                if candidate > upper {
                    candidate = original_integer - 1.0;
                }
                if (lower..=upper).contains(&candidate) {
                    *value = candidate;
                }
            }
        }
    }
    repair_decision_validated(decision, variables);
    Ok(())
}

fn sample_variable(spec: &VariableSpec, rng: &mut Xoshiro256PlusPlus) -> Result<f64> {
    match spec.kind {
        VariableKind::Continuous => {
            if spec.upper <= spec.lower {
                Ok(spec.lower)
            } else {
                Ok(rng.random_range(spec.lower..spec.upper))
            }
        }
        VariableKind::Integer => {
            let lower = spec.lower.ceil().to_i64().ok_or_else(|| {
                anyhow::anyhow!("integer lower bound cannot be represented as i64")
            })?;
            let upper = spec.upper.floor().to_i64().ok_or_else(|| {
                anyhow::anyhow!("integer upper bound cannot be represented as i64")
            })?;
            let sample = rng.random_range(lower..=upper);
            sample
                .to_f64()
                .ok_or_else(|| anyhow::anyhow!("integer sample cannot be represented as f64"))
        }
    }
}

/// Validate supplied seed rows before any repair, allocation, or random draw.
///
/// # Errors
///
/// Returns an error for invalid variables, a partial seed row, or a non-finite
/// requested seed value.
pub fn validate_initial_decisions(
    initial_decisions: &[f64],
    variables: &[VariableSpec],
) -> Result<()> {
    let n_variables = variables.len();
    if n_variables == 0 {
        bail!("initial decision variable count must be at least 1");
    }
    validate_variables(variables)?;
    if !initial_decisions
        .chunks_exact(n_variables)
        .remainder()
        .is_empty()
    {
        bail!(
            "initial decision length {} is not a whole number of rows with width {n_variables}",
            initial_decisions.len()
        );
    }
    for (index, value) in initial_decisions.iter().enumerate() {
        if !value.is_finite() {
            bail!("initial decision value at index {index} is non-finite");
        }
    }
    Ok(())
}

/// Repair supplied seed rows, then fill a population with random rows.
///
/// # Errors
///
/// Returns an error for malformed seed rows, allocation failure, or invalid
/// integer sampling bounds.
pub fn initial_decisions_with_random_fill(
    initial_decisions: &[f64],
    n_individuals: usize,
    variables: &[VariableSpec],
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Vec<f64>> {
    let n_variables = variables.len();
    validate_initial_decisions(initial_decisions, variables)?;
    let capacity = checked_product(n_individuals, n_variables, "initial decision capacity")?;
    let mut decisions = Vec::new();
    decisions
        .try_reserve(capacity)
        .with_context(|| format!("initial decision allocation failed for {capacity} values"))?;
    let mut repaired = Vec::new();
    repaired.try_reserve(n_variables).with_context(|| {
        format!("initial decision row allocation failed for {n_variables} values")
    })?;
    let mut accepted = 0usize;

    for row in initial_decisions.chunks_exact(n_variables) {
        if accepted >= n_individuals {
            break;
        }
        repaired.clear();
        repaired.extend_from_slice(row);
        repair_decision_validated(&mut repaired, variables);
        if !repaired.iter().all(|value| value.is_finite()) {
            continue;
        }
        if decision_exists(&decisions, &repaired, n_variables) {
            continue;
        }
        decisions.extend_from_slice(&repaired);
        accepted = checked_sum(accepted, 1, "accepted initial decision count")?;
    }

    while accepted < n_individuals {
        for spec in variables {
            decisions.push(sample_variable(spec, rng)?);
        }
        accepted = checked_sum(accepted, 1, "accepted random decision count")?;
    }

    Ok(decisions)
}

fn repair_decision_validated(decision: &mut [f64], variables: &[VariableSpec]) {
    for (value, spec) in decision.iter_mut().zip(variables.iter()) {
        *value = repair_value(*value, spec);
    }
}

fn decision_exists(decisions: &[f64], candidate: &[f64], n_variables: usize) -> bool {
    decisions.chunks_exact(n_variables).any(|existing| {
        existing
            .iter()
            .zip(candidate.iter())
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
    })
}

/// Clamp and quantize one decision value to its validated variable domain.
#[must_use]
pub const fn repair_value(value: f64, spec: &VariableSpec) -> f64 {
    let clamped = value.clamp(spec.lower, spec.upper);
    match spec.kind {
        VariableKind::Continuous => clamped,
        VariableKind::Integer => clamped.round().clamp(spec.lower.ceil(), spec.upper.floor()),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        initial_decisions_with_random_fill, polynomial_mutation, sbx_crossover_into, Result,
        VariableKind, VariableSpec, Xoshiro256PlusPlus,
    };
    use anyhow::bail;
    use rand::RngExt;
    use rand_xoshiro::rand_core::SeedableRng;

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn ensure_equal<T: PartialEq + std::fmt::Debug>(
        actual: &T,
        expected: &T,
        label: &str,
    ) -> Result<()> {
        if actual == expected {
            Ok(())
        } else {
            bail!("{label}: actual={actual:?}, expected={expected:?}")
        }
    }

    fn continuous_variables() -> [VariableSpec; 4] {
        [
            VariableSpec::new(-3.0, 4.0, VariableKind::Continuous),
            VariableSpec::new(-3.0, 4.0, VariableKind::Continuous),
            VariableSpec::new(-3.0, 4.0, VariableKind::Continuous),
            VariableSpec::new(-3.0, 4.0, VariableKind::Continuous),
        ]
    }

    #[test]
    fn initial_decisions_reject_nonfinite_seed_before_rng_draw() -> Result<()> {
        let variables = [VariableSpec::new(0.0, 1.0, VariableKind::Continuous)];
        let mut observed_rng = Xoshiro256PlusPlus::seed_from_u64(0x5EED);
        let mut expected_rng = Xoshiro256PlusPlus::seed_from_u64(0x5EED);

        let result =
            initial_decisions_with_random_fill(&[f64::NAN], 2, &variables, &mut observed_rng);
        if result.is_ok() {
            bail!("non-finite initial seed unexpectedly succeeded");
        }
        ensure_equal(
            &observed_rng.random::<u64>(),
            &expected_rng.random::<u64>(),
            "non-finite seed advanced RNG state",
        )?;
        Ok(())
    }

    #[test]
    fn initial_decision_allocation_retains_try_reserve_error_source() {
        let variables = [VariableSpec::new(0.0, 1.0, VariableKind::Continuous)];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5EED);
        let error = initial_decisions_with_random_fill(&[], usize::MAX, &variables, &mut rng)
            .expect_err("impossible decision capacity must be rejected");

        assert_eq!(
            error.to_string(),
            format!(
                "initial decision allocation failed for {} values",
                usize::MAX
            )
        );
        assert!(
            error
                .downcast_ref::<std::collections::TryReserveError>()
                .is_some(),
            "TryReserveError source missing from chain: {error:#}"
        );
    }

    #[test]
    fn sbx_fixed_seed_crossover_bits_and_rng() -> Result<()> {
        let variables = continuous_variables();
        let parent1 = [-2.5, -1.2, 0.4, 3.4];
        let parent2 = [3.3, 2.5, -2.4, -1.5];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x05B1_C0DE);
        let mut child1 = Vec::new();
        let mut child2 = Vec::new();

        sbx_crossover_into(
            &parent1,
            &parent2,
            &variables,
            1.0,
            17.0,
            &mut rng,
            &mut child1,
            &mut child2,
        )?;

        ensure_equal(
            &bits(&child1),
            &vec![
                13_836_477_269_412_898_024,
                13_831_471_533_926_176_268,
                4_597_738_936_353_762_104,
                4_614_838_538_166_547_251,
            ],
            "SBX crossover first-child bits differ",
        )?;
        // RE-PINNED 2026-08-28 (host axis on ONE component) -- first-ever TC
        // run of this feature lane (r55u deep-qual at 9029ba0e) found
        // child2[0] at 4_614_931_951_666_568_993 on the linux gate host while
        // darwin fast-test reproduces the mint value ..._991: a 2-ULP move on
        // one arm. Attribution: child1's four components AND child2's other
        // three are bit-identical cross-host, so the shared powf(beta) is
        // libm-invariant here and the divergence is per-binary FMA
        // contraction on the child2[0] arithmetic only (the axis
        // derived-float-equality budgeting warns about). RNG draw count
        // unchanged (child1 identical; the stream pin below still holds).
        // The pin is per gate host; a repin must reproduce on the gate host
        // before it lands (TC determinism confirmed on two back-to-back runs).
        let expected_child2_first = if cfg!(target_os = "macos") {
            4_614_931_951_666_568_991_u64
        } else {
            4_614_931_951_666_568_993_u64
        };
        ensure_equal(
            &bits(&child2),
            &vec![
                expected_child2_first,
                4_612_820_095_277_006_288,
                13_835_593_611_399_460_832,
                13_832_806_255_468_478_464,
            ],
            "SBX crossover second-child bits differ",
        )?;
        ensure_equal(
            &rng.random::<u64>(),
            &4_806_814_558_603_065_525,
            "SBX crossover advanced RNG differently",
        )?;
        Ok(())
    }

    #[test]
    fn sbx_fixed_seed_skip_bits_and_rng() -> Result<()> {
        let variables = continuous_variables();
        let parent1 = [-2.5, -1.2, 0.4, 3.4];
        let parent2 = [3.3, 2.5, -2.4, -1.5];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x05B1_C0DF);
        let mut child1 = Vec::new();
        let mut child2 = Vec::new();

        sbx_crossover_into(
            &parent1,
            &parent2,
            &variables,
            0.0,
            17.0,
            &mut rng,
            &mut child1,
            &mut child2,
        )?;

        ensure_equal(
            &bits(&child1),
            &vec![
                13_836_183_955_189_006_336,
                13_831_455_175_580_267_315,
                4_600_877_379_321_698_714,
                4_614_838_538_166_547_251,
            ],
            "SBX skip first-child bits differ",
        )?;
        ensure_equal(
            &bits(&child2),
            &vec![
                4_614_613_358_185_178_726,
                4_612_811_918_334_230_528,
                13_835_958_775_207_637_811,
                13_832_806_255_468_478_464,
            ],
            "SBX skip second-child bits differ",
        )?;
        ensure_equal(
            &rng.random::<u64>(),
            &12_575_472_181_381_072_443,
            "SBX skip advanced RNG differently",
        )?;
        Ok(())
    }

    #[test]
    fn polynomial_mutation_fixed_seed_apply_bits_and_rng() -> Result<()> {
        let variables = continuous_variables();
        let mut decision = [-2.1, -0.5, 1.2, 3.1];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x00B1_A55E);

        polynomial_mutation(&mut decision, &variables, 1.0, 21.0, &mut rng)?;

        ensure_equal(
            &bits(&decision),
            &vec![
                13_835_800_908_232_224_792,
                13_815_900_611_547_217_696,
                4_608_272_397_356_849_492,
                4_614_361_829_942_131_245,
            ],
            "polynomial mutation bits differ",
        )?;
        ensure_equal(
            &rng.random::<u64>(),
            &17_403_276_709_889_406_997,
            "polynomial mutation advanced RNG differently",
        )?;
        Ok(())
    }

    #[test]
    fn polynomial_mutation_fixed_seed_skip_bits_and_rng() -> Result<()> {
        let variables = continuous_variables();
        let mut decision = [-2.1, -0.5, 1.2, 3.1];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x00B1_A55F);

        polynomial_mutation(&mut decision, &variables, 0.0, 21.0, &mut rng)?;

        ensure_equal(
            &bits(&decision),
            &vec![
                13_835_283_235_263_532_237,
                13_826_050_856_027_422_720,
                4_608_083_138_725_491_507,
                4_614_162_998_222_441_677,
            ],
            "polynomial-mutation skip bits differ",
        )?;
        ensure_equal(
            &rng.random::<u64>(),
            &13_628_759_378_906_348_334,
            "polynomial-mutation skip advanced RNG differently",
        )?;
        Ok(())
    }
}
