use std::fmt;

use num_traits::ToPrimitive;
use satpy_core::{
    pack_gravity_coeffs, spherical_gravity_impl_frame_packed,
    spherical_gravity_impl_generic_packed, spherical_gravity_impl_packed,
    spherical_gravity_impl_sincos_packed, GravityCache, GravityError, PackedGravityCoeffs,
};

const ORDER: usize = 5;
const STRIDE: usize = ORDER + 1;

#[derive(Debug)]
enum CacheCheckError {
    Gravity(GravityError),
    StaleWorkspace { writer: &'static str },
    SignedZeroWorkspace,
}

impl From<GravityError> for CacheCheckError {
    fn from(error: GravityError) -> Self {
        Self::Gravity(error)
    }
}

impl fmt::Display for CacheCheckError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gravity(error) => write!(formatter, "gravity evaluation failed: {error}"),
            Self::StaleWorkspace { writer } => {
                write!(formatter, "{writer} left stale frame workspace provenance")
            }
            Self::SignedZeroWorkspace => {
                formatter.write_str("signed-zero position reused incompatible frame workspace")
            }
        }
    }
}

fn coefficients() -> Result<PackedGravityCoeffs, GravityError> {
    let coefficient_count = STRIDE
        .checked_mul(STRIDE)
        .ok_or(GravityError::InvariantViolation)?;
    let mut cosine = vec![0.0; coefficient_count];
    let mut sine = vec![0.0; coefficient_count];

    let c00 = cosine.first_mut().ok_or(GravityError::InvariantViolation)?;
    *c00 = 1.0;
    for degree in 2..=ORDER {
        let row_start = degree
            .checked_mul(STRIDE)
            .ok_or(GravityError::InvariantViolation)?;
        let row_end = row_start
            .checked_add(STRIDE)
            .ok_or(GravityError::InvariantViolation)?;
        let cosine_row = cosine
            .get_mut(row_start..row_end)
            .ok_or(GravityError::InvariantViolation)?;
        let sine_row = sine
            .get_mut(row_start..row_end)
            .ok_or(GravityError::InvariantViolation)?;
        let term_count = degree
            .checked_add(1)
            .ok_or(GravityError::InvariantViolation)?;
        for (order, (cosine_value, sine_value)) in cosine_row
            .iter_mut()
            .zip(sine_row.iter_mut())
            .enumerate()
            .take(term_count)
        {
            let coefficient_seed = degree
                .checked_mul(10)
                .and_then(|value| value.checked_add(order))
                .ok_or(GravityError::InvariantViolation)?;
            let cosine_seed = coefficient_seed
                .checked_add(1)
                .ok_or(GravityError::InvariantViolation)?
                .to_f64()
                .ok_or(GravityError::InvariantViolation)?;
            let sine_seed = coefficient_seed
                .checked_add(3)
                .ok_or(GravityError::InvariantViolation)?
                .to_f64()
                .ok_or(GravityError::InvariantViolation)?;
            *cosine_value = cosine_seed * 1.0e-7;
            *sine_value = -(sine_seed * 1.0e-8);
        }
    }
    pack_gravity_coeffs(&cosine, &sine, STRIDE, ORDER)
}

fn bits3(values: [f64; 3]) -> [u64; 3] {
    values.map(f64::to_bits)
}

fn assert_writer_cannot_stale_frame_workspace(
    writer_name: &'static str,
    packed: &PackedGravityCoeffs,
    writer: impl FnOnce(&mut GravityCache) -> Result<[f64; 3], GravityError>,
) -> Result<(), CacheCheckError> {
    let position = [6_875.0, -431.0, 203.0];
    let mut shared = GravityCache::new();
    let _ = spherical_gravity_impl_frame_packed(&position, &mut shared, packed)?;
    let _ = writer(&mut shared)?;

    let shared_result = spherical_gravity_impl_frame_packed(&position, &mut shared, packed)?;
    let fresh_result =
        spherical_gravity_impl_frame_packed(&position, &mut GravityCache::new(), packed)?;
    if bits3(shared_result) != bits3(fresh_result) {
        return Err(CacheCheckError::StaleWorkspace {
            writer: writer_name,
        });
    }
    Ok(())
}

fn verify_packed_writers_cannot_stale_frame_workspace() -> Result<(), CacheCheckError> {
    let packed = coefficients()?;
    let state = [7_031.0, 509.0, -317.0, 0.0, 7.4, 0.0];
    let jd = 2_460_000.25;

    assert_writer_cannot_stale_frame_workspace("packed", &packed, |cache| {
        spherical_gravity_impl_packed(&state, jd, cache, &packed)
    })?;
    assert_writer_cannot_stale_frame_workspace("generic packed", &packed, |cache| {
        spherical_gravity_impl_generic_packed(&state, jd, cache, &packed)
    })?;
    assert_writer_cannot_stale_frame_workspace("sincos packed", &packed, |cache| {
        spherical_gravity_impl_sincos_packed(&state, 0.25, 0.75, cache, &packed)
    })?;
    Ok(())
}

#[test]
fn packed_writers_cannot_stale_frame_workspace() {
    let result = verify_packed_writers_cannot_stale_frame_workspace();
    assert!(
        result.is_ok(),
        "packed writer provenance check failed: {result:?}"
    );
}

fn verify_signed_zero_is_an_exact_frame_packed_cache_key() -> Result<(), CacheCheckError> {
    let packed = coefficients()?;
    let plus = [6_900.0, 0.0, 125.0];
    let minus = [6_900.0, -0.0, 125.0];

    let mut warmed = GravityCache::new();
    let _ = spherical_gravity_impl_frame_packed(&plus, &mut warmed, &packed)?;
    let warmed_minus = spherical_gravity_impl_frame_packed(&minus, &mut warmed, &packed)?;
    let fresh_minus =
        spherical_gravity_impl_frame_packed(&minus, &mut GravityCache::new(), &packed)?;

    if bits3(warmed_minus) != bits3(fresh_minus) {
        return Err(CacheCheckError::SignedZeroWorkspace);
    }
    Ok(())
}

#[test]
fn signed_zero_is_an_exact_frame_packed_cache_key() {
    let result = verify_signed_zero_is_an_exact_frame_packed_cache_key();
    assert!(
        result.is_ok(),
        "signed-zero frame cache check failed: {result:?}"
    );
}
