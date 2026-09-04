//! How far apart the block-4 and scalar J2 equinoctial propagators actually
//! are, as a measured number rather than a guard tolerance.
//!
//! `equinoc_prop_j2_step_impl` routes `chunks_exact(4)` of its time slice
//! through `equinoc_prop_j2_step_simd4` and the remainder through the scalar
//! `equinoc_prop_j2_from_impl`. Everything that consumes it is guarded at 1e-9
//! ABSOLUTE, which is five orders above ULP — so "guarded at 1e-9" is
//! compatible with the two routes agreeing to the last bit and equally
//! compatible with them differing by micrometres. Those are different animals,
//! and a lever that reroutes traffic between the two is priced completely
//! differently depending on which one is true.
//!
//! **Measured 2026-08-11, and it is the second one.** Over 131,072 rows of the
//! LEO-target/transfer-TOF domain the MF lane uses: only **11.63% of rows are
//! bit-identical**, the maximum position deviation is **7.22e-6 m** (about
//! 7,000 ULP on a 7,000 km position), and the maximum relative deviation is
//! **1.99e-8**, on the mean longitude.
//!
//! **The mechanism is not rounding.** `equinoc2eci_simd` solves the same Kepler
//! iteration against the same `TOL = 1e-12`, but it reaches it through
//! `wide`'s `f64x4::sin_cos` and `mod2pi_simd` where the scalar body uses libm
//! `sin_cos` and `fmod`. Different transcendental implementations,
//! iterated to a tolerance rather than to convergence, so the two routes differ
//! at TOLERANCE scale and not at ULP scale.
//!
//! **What that settles.** A change that moves traffic between these routes is
//! NOT in the class of the R17 SIMD Lambert pack, whose bound was 4.554e-14 in
//! relative dv against a 1.08e-13 precedent. It is a physics-accuracy change
//! wearing a rounding change's clothes, and it cannot be adopted on that
//! precedent. See `docs/MF_COST_MAP.md` §5.
//!
//! Read the assertions as CEILINGS THAT HAVE BEEN OBSERVED, not as tolerances
//! anything is entitled to. If one fires, the two routes have moved apart and
//! every decision taken against these numbers needs revisiting.

use satpy_core::{equinoc_prop_j2_from_impl, equinoc_prop_j2_step_impl};

/// Target-like equinoctial element sets: `[a, h, k, p, q, lambda]`.
///
/// Spans the LEO band the Part A target set lives in — semi-major axes from a
/// low 400 km orbit to a high 1000 km one, near-circular through mildly
/// eccentric, and inclinations from equatorial through sun-synchronous, since
/// `p` and `q` drive the J2 secular rates that the two routes evaluate.
fn element_sets() -> Vec<[f64; 6]> {
    let mut sets = Vec::new();
    for &semi_major_km in &[6778.0_f64, 6978.0, 7178.0, 7378.0] {
        for &eccentricity in &[0.0, 0.001, 0.01, 0.05] {
            for &inclination_deg in &[0.0_f64, 28.5, 51.6, 97.8] {
                let half = (inclination_deg.to_radians() * 0.5).tan();
                for &raan_deg in &[0.0_f64, 137.0] {
                    let raan = raan_deg.to_radians();
                    for &argument_deg in &[0.0_f64, 71.0] {
                        let argument = argument_deg.to_radians();
                        sets.push([
                            semi_major_km,
                            eccentricity * argument.sin(),
                            eccentricity * argument.cos(),
                            half * raan.sin(),
                            half * raan.cos(),
                            (argument_deg + raan_deg + 33.0).to_radians(),
                        ]);
                    }
                }
            }
        }
    }
    sets
}

/// Maximum block-4-versus-scalar deviation observed over the transfer-lane
/// operand domain, with a little headroom.
///
/// Observed 2026-08-11: 1.9870e-8 relative, 7.2196e-6 m in position. The
/// relative figure is against the scalar route's own magnitude for that
/// component, which is why the absolute position deviation is pinned beside
/// it — a relative bound on a component that passes through zero says nothing.
const OBSERVED_MAX_RELATIVE: f64 = 2.5e-8;
const OBSERVED_MAX_POSITION_M: f64 = 1.0e-5;

#[test]
fn block4_and_scalar_j2_propagation_agree_to_the_pinned_census_bound() {
    let mut max_relative = 0.0_f64;
    let mut max_position_m = 0.0_f64;
    let mut worst_row = None;
    let mut identical = 0_u64;
    let mut rows = 0_u64;

    for elements in element_sets() {
        // Transfer times of flight: minutes to a day, which is the range the
        // Brent bracket and its pre-scan sample over.
        let mut offsets = Vec::new();
        for step in 0..512_u32 {
            offsets.push(120.0 + f64::from(step) * 168.0);
        }

        let mut packed = vec![0.0_f64; offsets.len() * 6];
        equinoc_prop_j2_step_impl(&elements, &offsets, 0.0, &mut packed);

        for (index, &dt) in offsets.iter().enumerate() {
            // Only the lanes the SIMD route actually served: the remainder of
            // `chunks_exact(4)` runs the scalar body verbatim and is
            // bit-identical by construction, so including it would dilute the
            // census with rows that cannot differ.
            if index >= offsets.len() - offsets.len() % 4 {
                continue;
            }
            let mut scalar = [0.0_f64; 6];
            equinoc_prop_j2_from_impl(&elements, dt, &mut scalar);
            let Some(block) = packed
                .get(index * 6..index * 6 + 6)
                .and_then(|slice| <[f64; 6]>::try_from(slice).ok())
            else {
                panic!("packed output is short at row {index}");
            };
            rows += 1;

            let mut row_identical = true;
            for (component, (&got, &want)) in block.iter().zip(scalar.iter()).enumerate() {
                if got.to_bits() != want.to_bits() {
                    row_identical = false;
                }
                let scale = want.abs().max(f64::MIN_POSITIVE);
                let relative = (got - want).abs() / scale;
                if relative > max_relative {
                    max_relative = relative;
                    worst_row = Some((elements, dt, component, want, got));
                }
            }
            if row_identical {
                identical += 1;
            }
            let position_m = block
                .iter()
                .zip(scalar.iter())
                .take(3)
                .map(|(got, want)| (got - want).abs() * 1000.0)
                .fold(0.0_f64, f64::max);
            max_position_m = max_position_m.max(position_m);
        }
    }

    println!("block4-vs-scalar census over {rows} rows");
    println!("  bit-identical rows: {identical} ({:.2}%)", {
        let identical_f = u32::try_from(identical).map_or(f64::NAN, f64::from);
        let rows_f = u32::try_from(rows).map_or(f64::NAN, f64::from);
        identical_f / rows_f * 100.0
    });
    println!("  max relative deviation: {max_relative:.4e}");
    println!("  max position deviation: {max_position_m:.4e} m");
    if let Some((elements, dt, component, want, got)) = worst_row {
        println!("  worst: elements {elements:?} dt {dt} component {component}");
        println!("         scalar {want:.17e}  block4 {got:.17e}");
    }

    assert!(rows > 10_000, "census degenerated to {rows} rows");
    assert!(
        max_relative <= OBSERVED_MAX_RELATIVE,
        "block4-vs-scalar max relative deviation {max_relative:.4e} exceeds the pinned \
         census bound {OBSERVED_MAX_RELATIVE:.4e}; every decision taken against that \
         number needs revisiting"
    );
    assert!(
        max_position_m <= OBSERVED_MAX_POSITION_M,
        "block4-vs-scalar max position deviation {max_position_m:.4e} m exceeds the \
         pinned census bound {OBSERVED_MAX_POSITION_M:.4e} m"
    );
}
