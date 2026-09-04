//! Pure-Rust GCRS->ITRS frame/time chain matching the sealed ERFA 2.0.1 /
//! SOFA 20231011-derived 4AF fixture
//! (`crates/satpy_core/tests/data/erfa_sofa_derived_frame_time_v1.json`).
//!
//! The chain reproduces the fixture generator
//! (`crates/satpy_core/oracle/ErfaFrameTimeVectors.c`): ERFA's binary64 time
//! scale, precession-nutation, CIO, and polar-motion routines, with a
//! double-double Earth Rotation Angle, outer `RPOM * R3(ERA) * RC2I`
//! composition, four-node continuous-TAI EOP Lagrange interpolation, and a
//! conditioned centered five-point stencil for `Rdot` / `Rddot`.
//!
//! Series tables are generated from the sealed ERFA source by
//! `scripts/regenerate-frame-time-tables.sh`; `tables` is never hand-edited.

pub mod cio;
pub mod dd;
pub mod eop;
pub mod era;
mod fund_args;
mod iau2006;
pub mod tables;
pub mod timescale;

pub mod authority;
pub mod chain;

pub use chain::{
    EopPolicy, Epoch, FrameChainError, ACCELERATIONS, EPOCHS, H_FIXTURE_S, POSITIONS, VELOCITIES,
};

/// One fixture case: GCRS->ITRS matrices and the transformed ITRS state.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub epoch_name: &'static str,
    pub policy: EopPolicy,
    pub state_index: usize,
    pub r: [[f64; 3]; 3],
    pub rdot: [[f64; 3]; 3],
    pub rddot: [[f64; 3]; 3],
    pub r_itrs: [f64; 3],
    pub v_itrs: [f64; 3],
    pub a_itrs: [f64; 3],
}

fn to_f64_mat(m: &chain::DdMat) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (out_row, matrix_row) in out.iter_mut().zip(m) {
        for (out_element, matrix_element) in out_row.iter_mut().zip(matrix_row) {
            *out_element = matrix_element.to_f64();
        }
    }
    out
}

/// Compute the 20 fixture cases in canonical order (epoch, zero-then-real EOP,
/// state index), using the given `finals2000A.all` contents for real-EOP cases.
///
/// # Errors
///
/// Returns [`FrameChainError`] when input EOP or time-scale data is invalid.
pub fn compute_all_cases(finals: &str) -> Result<Vec<Case>, FrameChainError> {
    let mut cases = Vec::with_capacity(20);
    for epoch in &EPOCHS {
        for policy in [EopPolicy::Zero, EopPolicy::Real] {
            let (rotation_dd, rotation_rate_dd, rotation_accel_dd) =
                chain::derivatives(epoch, policy, H_FIXTURE_S, finals)?;
            let rotation = to_f64_mat(&rotation_dd);
            let rotation_rate = to_f64_mat(&rotation_rate_dd);
            let rotation_accel = to_f64_mat(&rotation_accel_dd);
            for (state_index, ((position, velocity), acceleration)) in POSITIONS
                .iter()
                .zip(&VELOCITIES)
                .zip(&ACCELERATIONS)
                .enumerate()
            {
                let (r_itrs, v_itrs, a_itrs) = chain::transform_state(
                    &rotation_dd,
                    &rotation_rate_dd,
                    &rotation_accel_dd,
                    position,
                    velocity,
                    acceleration,
                );
                cases.push(Case {
                    epoch_name: epoch.name,
                    policy,
                    state_index,
                    r: rotation,
                    rdot: rotation_rate,
                    rddot: rotation_accel,
                    r_itrs,
                    v_itrs,
                    a_itrs,
                });
            }
        }
    }
    Ok(cases)
}

#[cfg(test)]
mod tests {
    use super::eop::EopError;
    use super::*;

    #[test]
    fn missing_real_eop_propagates_as_a_typed_error() {
        assert!(matches!(
            compute_all_cases(""),
            Err(FrameChainError::Eop(EopError::MissingRecord { .. }))
        ));
    }
}
