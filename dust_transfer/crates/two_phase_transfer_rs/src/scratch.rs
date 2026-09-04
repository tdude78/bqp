/// Scratch buffer for satellite coordinate computation (propagation phase).
/// Separated from `SolveScratch` to allow independent borrowing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScratchCapacityOverflow;

impl std::fmt::Display for ScratchCapacityOverflow {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("scratch capacity overflow")
    }
}

impl std::error::Error for ScratchCapacityOverflow {}

#[inline]
fn reserve_scratch_capacity<T>(
    values: &mut Vec<T>,
    required_capacity: usize,
) -> Result<(), ScratchCapacityOverflow> {
    if required_capacity <= values.capacity() {
        return Ok(());
    }
    let additional = required_capacity
        .checked_sub(values.len())
        .ok_or(ScratchCapacityOverflow)?;
    values
        .try_reserve_exact(additional)
        .map_err(|_| ScratchCapacityOverflow)
}

pub struct CoordinateScratch {
    pub sats_eci: Vec<[f64; 6]>,
    pub sats_equ: Vec<[f64; 6]>,
}

impl CoordinateScratch {
    pub fn new(n_sats: usize) -> Result<Self, ScratchCapacityOverflow> {
        let mut sats_eci = Vec::new();
        let mut sats_equ = Vec::new();
        reserve_scratch_capacity(&mut sats_eci, n_sats)?;
        reserve_scratch_capacity(&mut sats_equ, n_sats)?;
        Ok(Self { sats_eci, sats_equ })
    }

    pub fn prepare(&mut self, n_sats: usize) -> Result<(), ScratchCapacityOverflow> {
        self.sats_eci.clear();
        self.sats_equ.clear();
        reserve_scratch_capacity(&mut self.sats_eci, n_sats)?;
        reserve_scratch_capacity(&mut self.sats_equ, n_sats)
    }
}

/// Scratch buffer for reusable pair-proxy selection state.
pub struct SolveScratch {
    pub pair_proxy: crate::solve::pair_proxy::PairProxyScratch,
}

impl SolveScratch {
    pub fn new(n_sats: usize) -> Result<Self, ScratchCapacityOverflow> {
        let pair_capacity = n_sats.checked_mul(2).ok_or(ScratchCapacityOverflow)?;
        Ok(Self {
            pair_proxy: crate::solve::pair_proxy::PairProxyScratch::new(pair_capacity),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_constructors_reject_arithmetic_overflow_without_allocating() {
        assert!(matches!(
            CoordinateScratch::new(usize::MAX),
            Err(ScratchCapacityOverflow)
        ));
        assert!(matches!(
            SolveScratch::new(usize::MAX),
            Err(ScratchCapacityOverflow)
        ));
    }

    #[test]
    fn coordinate_scratch_prepare_reserves_full_next_batch_after_clear() {
        let mut scratch = CoordinateScratch {
            sats_eci: Vec::with_capacity(10),
            sats_equ: Vec::with_capacity(10),
        };

        scratch.prepare(15).expect("fallible reserve succeeds");

        assert!(scratch.sats_eci.capacity() >= 15);
        assert!(scratch.sats_equ.capacity() >= 15);
        let eci_capacity = scratch.sats_eci.capacity();
        let equ_capacity = scratch.sats_equ.capacity();
        for _ in 0..15 {
            scratch.sats_eci.push([0.0; 6]);
            scratch.sats_equ.push([0.0; 6]);
        }
        assert_eq!(scratch.sats_eci.capacity(), eci_capacity);
        assert_eq!(scratch.sats_equ.capacity(), equ_capacity);
    }
}
