//! Deterministic 7-coordinate seed-folding.
//!
//! [`seed_leaf`] maps a work-item's seven identity coordinates to a `u64` leaf
//! seed. It is a PURE function of those coordinates — it never reads a thread
//! id, a rayon worker index, or completion order — so the same leaf gets the
//! same seed no matter where or when it runs. That is what makes the optimizer
//! reproducible under the single global pool: parallelism reorders execution,
//! not seeds.
//!
//! The fold is splitmix64-style: each coordinate is advanced through a full
//! avalanche using the splitmix / rrmxmx constant family, with a
//! position-dependent gamma so permuting two coordinates changes the result.

/// Golden-ratio odd gamma (Weyl increment / per-position offset).
const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
/// splitmix64 avalanche multiplier 1.
const M1: u64 = 0xbf58_476d_1ce4_e5b9;
/// splitmix64 avalanche multiplier 2.
const M2: u64 = 0x94d0_49bb_1331_11eb;
/// Odd multiplier used to condition the base seed.
const M3: u64 = 0xd134_2543_de82_ef95;
/// Final-avalanche multiplier 1 (alternate family).
const M4: u64 = 0xff51_afd7_ed55_8ccd;
/// Final-avalanche multiplier 2 (alternate family).
const M5: u64 = 0xc4ce_b9fe_1a85_ec53;

/// One splitmix64 avalanche step (bijective over `u64`).
#[inline]
const fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(M1);
    z = (z ^ (z >> 27)).wrapping_mul(M2);
    z ^ (z >> 31)
}

/// Fold the seven identity coordinates of one work leaf into a `u64` seed.
///
/// The coordinates, from coarsest to finest:
/// * `base_seed` — the campaign/run seed (the reproducibility root).
/// * `optimizer_id`, `family_id`, `event_set_id` — the descriptive identity.
/// * `seed_axis` — the replicate / seed-sweep axis.
/// * `candidate_idx`, `event_idx` — the flat `WorkUnit` coordinates.
///
/// Guarantees (verified by tests): identical coordinates yield an identical
/// seed; changing ANY single coordinate changes the seed with overwhelming
/// probability; and the mapping is order/position-sensitive, so two coordinate
/// tuples that are permutations of one another do not collide.
#[inline]
#[must_use]
pub fn seed_leaf(
    base_seed: u64,
    optimizer_id: u64,
    family_id: u64,
    event_set_id: u64,
    seed_axis: u64,
    candidate_idx: u64,
    event_idx: u64,
) -> u64 {
    // Condition the base with an odd multiplier so an all-zero coordinate tuple
    // still produces a well-mixed stream and base-seed differences avalanche on
    // their own. `wrapping_mul` by an odd constant is bijective.
    let mut state = splitmix64(base_seed.wrapping_mul(M3) ^ GAMMA);

    let coords = [
        optimizer_id,
        family_id,
        event_set_id,
        seed_axis,
        candidate_idx,
        event_idx,
    ];
    for (&position, &c) in [1_u64, 2, 3, 4, 5, 6].iter().zip(&coords) {
        // Position-dependent gamma: coordinate i is offset by (i+1)*GAMMA so a
        // coordinate's contribution depends on WHERE it sits, not just its
        // value. This is what keeps permutations from colliding. Pure function
        // of (state, c, i) — never of thread id or completion order.
        let pos = GAMMA.wrapping_mul(position);
        state = splitmix64(state ^ c.wrapping_add(pos));
    }

    // Final avalanche with the alternate constant family so every bit of the
    // accumulated state reaches every output bit.
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(M4);
    z = (z ^ (z >> 27)).wrapping_mul(M5);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const COORDS: [u64; 7] = [7, 1, 2, 3, 4, 5, 6];

    fn seed_with(coords: [u64; 7]) -> u64 {
        seed_leaf(
            coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], coords[6],
        )
    }

    /// Pin the seed stream to literals, not to itself.
    ///
    /// This test used to read `assert_eq!(seed_with(COORDS), seed_with(COORDS))`,
    /// which is vacuous: `seed_leaf` is a pure function of its arguments, so it
    /// held for ANY implementation and pinned nothing. Nothing else in the tree
    /// pinned the output either, while all five optimizers seed their streams
    /// from it (`nd_optimizer/src/`: `nsga2.rs`, `mopso.rs`, `nsde.rs`,
    /// `eps_nsga2.rs`, `age_moea2.rs`). Every replicate and every reproducibility claim rests on
    /// this mapping being stable across builds and architectures, and until this
    /// literal landed, a constant-family typo would have gone unnoticed.
    ///
    /// The sibling `changing_any_single_coordinate_changes_the_seed` covers
    /// SENSITIVITY. This one covers STABILITY; the two are not substitutes.
    ///
    /// The all-zero tuple is included deliberately: it is the case the odd
    /// multiplier in `seed_leaf` exists to condition, so a regression that
    /// dropped that conditioning would still pass the populated case.
    #[test]
    fn seed_leaf_output_is_pinned_to_literals() {
        assert_eq!(seed_with(COORDS), 0xffbd_6ca6_f9dc_68a3);
        assert_eq!(seed_with([0; 7]), 0x350e_8fcc_fb3a_a79e);
    }

    #[test]
    fn changing_any_single_coordinate_changes_the_seed() {
        let base = seed_with(COORDS);
        for (i, coordinate) in COORDS.iter().enumerate() {
            let mut bumped = COORDS;
            if let Some(value) = bumped.get_mut(i) {
                *value = coordinate.wrapping_add(1);
            }
            assert_ne!(
                seed_with(bumped),
                base,
                "bumping coordinate {i} must change the seed"
            );
        }
    }

    #[test]
    fn permuting_coordinates_does_not_collide() {
        // Swapping two distinct coordinates must not produce the same seed.
        let a = seed_leaf(10, 20, 30, 40, 50, 60, 70);
        let swapped = seed_leaf(10, 30, 20, 40, 50, 60, 70); // family <-> event_set
        assert_ne!(a, swapped, "coordinate order must matter");
    }

    #[test]
    fn base_seed_differences_avalanche() {
        let a = seed_leaf(0, 0, 0, 0, 0, 0, 0);
        let b = seed_leaf(1, 0, 0, 0, 0, 0, 0);
        assert_ne!(a, b, "distinct base seeds must yield distinct leaf seeds");
    }

    #[test]
    fn collision_sanity_over_a_grid() {
        // Sweep a dense grid of the two finest coordinates plus a couple of
        // coarse axes and assert every leaf seed is unique.
        let mut seen = HashSet::new();
        let mut count = 0usize;
        for opt in 0..3u64 {
            for fam in 0..3u64 {
                for cand in 0..40u64 {
                    for ev in 0..40u64 {
                        let s = seed_leaf(0xABCD_1234, opt, fam, 0, 0, cand, ev);
                        assert!(seen.insert(s), "seed collision at {opt},{fam},{cand},{ev}");
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(seen.len(), count, "every grid point must be unique");
    }
}
