//! Accuracy coverage for packed gravity's canonical dense-prefix dispatch.
//!
//! Degree 1 is identically zero in the production Earth model. This fixture
//! exercises the resulting dense packed construction by comparing its public
//! gravity output with the public raw-coefficient oracle over representative
//! LEO positions.

use num_traits::ToPrimitive;
use satpy_core::{
    pack_gravity_coeffs, spherical_gravity_impl, spherical_gravity_impl_packed, GravityCache,
};

const ORDER: usize = 5;

/// Every `(degree, order)` slot the parser below must actually fill.
///
/// The source file starts at degree 1, so the prefix this fixture consumes is
/// `sum(degree + 1)` over `degree` in `1..=ORDER` -- 20 slots for `ORDER = 5`.
/// Derived from `ORDER`, not from a run: the point is that a count taken from
/// what currently survives cannot notice the corpus emptying.
const fn expected_coefficient_rows() -> usize {
    let mut degree: usize = 1;
    let mut total: usize = 0;
    while degree <= ORDER {
        // `degree + 1` is both this degree's slot count and the next degree,
        // so one add serves both. Saturating rather than bare: the workspace
        // denies arithmetic_side_effects, and it denies `panic` too, so the
        // checked-with-panic form is not available either. ORDER is a small
        // compile-time constant, so saturation is unreachable; if it were ever
        // reached, `degree` would stop below the loop bound and terminate.
        let next = degree.saturating_add(1);
        total = total.saturating_add(next);
        degree = next;
    }
    total
}

fn load_egm() -> anyhow::Result<(Vec<f64>, Vec<f64>, usize)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/reference/gravity/EGM_GOC_2_source_with_header.txt"
    );
    let text = std::fs::read_to_string(path)?;
    let stride = ORDER
        .checked_add(2)
        .ok_or_else(|| anyhow::anyhow!("EGM stride overflow"))?;
    let total = stride
        .checked_mul(stride)
        .ok_or_else(|| anyhow::anyhow!("EGM storage overflow"))?;
    let mut cosine = vec![0.0; total];
    let mut sine = vec![0.0; total];
    let mut filled = vec![false; total];
    let c00 = cosine
        .first_mut()
        .ok_or_else(|| anyhow::anyhow!("EGM C00 storage missing"))?;
    *c00 = 1.0;

    let max_factorial = 2_usize
        .checked_mul(stride)
        .ok_or_else(|| anyhow::anyhow!("EGM factorial bound overflow"))?;
    let factor_count = max_factorial
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("EGM factorial storage overflow"))?;
    let mut log_factorial = vec![0.0; factor_count];
    let mut accumulated = 0.0;
    for (index, value) in log_factorial.iter_mut().enumerate().skip(1) {
        let index_value = index
            .to_f64()
            .ok_or_else(|| anyhow::anyhow!("EGM factorial index cannot convert to f64"))?;
        accumulated += index_value.ln();
        *value = accumulated;
    }

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let (Some(degree_text), Some(order_text), Some(cosine_text), Some(sine_text)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let (Ok(degree), Ok(order)) = (degree_text.parse::<usize>(), order_text.parse::<usize>())
        else {
            continue;
        };
        let (Ok(cosine_value), Ok(sine_value)) = (
            cosine_text.replace(['D', 'd'], "E").parse::<f64>(),
            sine_text.replace(['D', 'd'], "E").parse::<f64>(),
        ) else {
            continue;
        };
        if degree > ORDER || order > degree {
            continue;
        }

        let index = degree
            .checked_mul(stride)
            .and_then(|start| start.checked_add(order))
            .ok_or_else(|| anyhow::anyhow!("EGM coefficient index overflow"))?;
        let (cosine_slot, sine_slot) = cosine
            .get_mut(index)
            .zip(sine.get_mut(index))
            .ok_or_else(|| anyhow::anyhow!("EGM coefficient index outside storage"))?;
        let upper_factorial = log_factorial
            .get(
                degree
                    .checked_add(order)
                    .ok_or_else(|| anyhow::anyhow!("EGM normalization index overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("EGM upper normalization index outside storage"))?;
        let lower_factorial = log_factorial
            .get(
                degree
                    .checked_sub(order)
                    .ok_or_else(|| anyhow::anyhow!("EGM normalization index underflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("EGM lower normalization index outside storage"))?;
        let delta_m0 = if order == 0 { 1.0 } else { 0.0 };
        let degree_value = degree
            .to_f64()
            .ok_or_else(|| anyhow::anyhow!("EGM degree cannot convert to f64"))?;
        let doubled_degree = 2.0 * degree_value;
        let degree_scale = doubled_degree + 1.0;
        let denominator = (2.0 - delta_m0) * degree_scale;
        let normalization =
            (0.5 * ((*upper_factorial - *lower_factorial) - denominator.ln())).exp();
        *cosine_slot = cosine_value / normalization;
        *sine_slot = sine_value / normalization;
        let filled_slot = filled
            .get_mut(index)
            .ok_or_else(|| anyhow::anyhow!("EGM coverage index outside storage"))?;
        *filled_slot = true;
    }

    // Floor. Four of the five guards above are `continue`, so a reformatted,
    // truncated or renamed source file skips every line and leaves both tables
    // at their zero initialization. The comparison downstream then hands the
    // SAME zeros to both gravity routes, which agree exactly -- a green run
    // over an empty corpus. This asset is not one of the byte-pinned reference
    // files, so nothing else would notice.
    let mut covered = 0_usize;
    for degree in 1..=ORDER {
        for order in 0..=degree {
            let index = degree
                .checked_mul(stride)
                .and_then(|start| start.checked_add(order))
                .ok_or_else(|| anyhow::anyhow!("EGM coverage index overflow"))?;
            anyhow::ensure!(
                filled.get(index).copied().unwrap_or(false),
                "EGM source supplied no C/S row for degree {degree}, order {order}; \
                 the dense-prefix comparison would run on zero coefficients"
            );
            covered = covered
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("EGM coverage count overflow"))?;
        }
    }
    anyhow::ensure!(
        covered == expected_coefficient_rows(),
        "EGM coverage counted {covered} rows against the {} the degree-{ORDER} \
         prefix requires",
        expected_coefficient_rows()
    );
    Ok((cosine, sine, stride))
}

struct Rng(u64);

impl Rng {
    const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> anyhow::Result<f64> {
        let mantissa = (self.next_u64() >> 11)
            .to_f64()
            .ok_or_else(|| anyhow::anyhow!("random mantissa cannot convert to f64"))?;
        Ok(mantissa / 9_007_199_254_740_992.0)
    }
}

fn verify_dense_prefix_packed_accuracy_matches_raw_oracle_over_leo() -> anyhow::Result<()> {
    let (cosine, sine, stride) = load_egm()?;
    let packed = pack_gravity_coeffs(&cosine, &sine, stride, ORDER)?;

    let sample_count = 200_000;
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut raw_cache = GravityCache::new();
    let mut packed_cache = GravityCache::new();
    let jd = 2_460_000.5;
    let mut max_absolute = 0.0_f64;
    let mut max_relative = 0.0_f64;

    for _ in 0..sample_count {
        let altitude_scale = 1800.0 * rng.unit()?;
        let altitude = 200.0 + altitude_scale;
        let radius = 6_378.136_4 + altitude;
        let doubled_unit = 2.0 * rng.unit()?;
        let vertical = doubled_unit - 1.0;
        let longitude = 2.0 * std::f64::consts::PI * rng.unit()?;
        let horizontal = (1.0 - vertical * vertical).max(0.0).sqrt();
        let state = [
            radius * horizontal * longitude.cos(),
            radius * horizontal * longitude.sin(),
            radius * vertical,
            0.0,
            7.5,
            0.0,
        ];

        let raw =
            spherical_gravity_impl(&state, jd, ORDER, &cosine, &sine, stride, &mut raw_cache)?;
        let packed = spherical_gravity_impl_packed(&state, jd, &mut packed_cache, &packed)?;

        let mut difference_squared = 0.0;
        let mut magnitude_squared = 0.0;
        for (packed_component, raw_component) in packed.iter().zip(raw) {
            let difference = *packed_component - raw_component;
            let difference_term = difference * difference;
            difference_squared += difference_term;
            let magnitude_term = *packed_component * *packed_component;
            magnitude_squared += magnitude_term;
        }
        let absolute_difference = difference_squared.sqrt();
        let magnitude = magnitude_squared.sqrt();
        max_absolute = max_absolute.max(absolute_difference);
        if magnitude > 0.0 {
            max_relative = max_relative.max(absolute_difference / magnitude);
        }
    }

    if max_absolute >= 1.0e-16 {
        return Err(anyhow::anyhow!(
            "packed dense-prefix path differs from raw oracle by {max_absolute:e} km/s^2"
        ));
    }
    if max_relative >= 1.0e-14 {
        return Err(anyhow::anyhow!(
            "packed dense-prefix relative difference {max_relative:e} is too large"
        ));
    }
    Ok(())
}

#[test]
fn dense_prefix_packed_accuracy_matches_raw_oracle_over_leo() {
    let result = verify_dense_prefix_packed_accuracy_matches_raw_oracle_over_leo();
    assert!(
        result.is_ok(),
        "dense-prefix accuracy check failed: {result:?}"
    );
}
