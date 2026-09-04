//! Differential oracle: the geometry route must solve the same Lambert
//! problem as the whole-problem route.
//!
//! WHAT THE TWO ROUTES ARE. `izzo2015_impl` takes `(mu, r1, r2, tof)` and
//! derives its own geometry inline. `izzo2015_impl_with_geom` takes a
//! `LambertGeometry` built separately by `compute_lambert_geometry`, and is
//! the entry the batch and multi-rev paths use so the geometry can be hoisted
//! out of their `m` loop. They share `find_xy` and `reconstruct_velocities`
//! and NOTHING ELSE: norms, unit vectors, the cross products, `lambda` and
//! the final `v = vr*ir + vt*it` are derived twice, by different arithmetic
//! (nalgebra `norm()`/`cross()` and a division on one side; `mul_add` norms,
//! `mul_add` crosses and a reciprocal multiply on the other). So this is a
//! genuine independent-derivation check of `compute_lambert_geometry`, not a
//! restatement of it.
//!
//! WHY IT IS NEEDED. Before this file the only test touching
//! `izzo2015_impl_with_geom` was `test_fast_geom_solver_matches_scalar_geom_solver`,
//! and both of its arms consume the SAME `geom` value -- so any error inside
//! `compute_lambert_geometry` cancels on both sides and that test cannot see
//! it. It pins scalar-vs-SIMD reconstruction, which is a different claim.
//! Nothing pinned the geometry itself.
//!
//! NOT A BIT PIN, DELIBERATELY. The two derivations differ in FMA use and in
//! association, so they are not expected to agree to the last bit and a bit
//! comparison here would be a flake generator. The tolerance below is
//! MEASURED against the fixture set, not chosen for comfort.

use super::{compute_lambert_geometry, izzo2015_impl, izzo2015_impl_with_geom};

/// Earth gravitational parameter, km^3/s^2 -- the one the crate's own tests
/// and the shipping callers use.
const MU: f64 = satpy_core::MU;

/// Solver controls. `maxiter` is the crate's own test setting; the
/// tolerances are the tight end, so a divergence cannot hide behind a loose
/// convergence criterion in `find_xy`.
const MAXITER: i32 = 35;
const ATOL: f64 = 1e-12;
const RTOL: f64 = 1e-12;

/// Agreement bound on each velocity component, km/s.
///
/// MEASURED, and the test prints its own worst case so the margin is never
/// a claim you have to take on trust. On the Mac mini at this commit the two
/// routes agree BIT-EXACTLY on every departure velocity (worst |dv1| = 0) and
/// differ by one ulp on one arrival velocity (worst |dv2| = 8.88e-16 km/s) --
/// the FMA-vs-non-FMA difference in the two derivations, which `find_xy`
/// mostly absorbs. The bound is left six orders above that rather than pinned
/// at the ulp: the deviation runs through `sqrt`/`log`/`atan2` inside
/// `find_xy`, so it is a per-libm number, and a bound sitting on the Mac's
/// measurement would be a flake on any other one. What the bound has to be is
/// tight enough to catch a geometry defect, and
/// `a_perturbed_geometry_moves_the_answer_past_the_tolerance` is the proof
/// that it is: a one-part-in-1e6 error in `ll_base` moves the answer far past
/// it.
const V_TOL_KM_S: f64 = 1e-9;

/// One Lambert problem, named so a failure says which one broke.
struct Fixture {
    name: &'static str,
    r1: [f64; 3],
    r2: [f64; 3],
    tof_s: f64,
    m: i32,
    prograde: bool,
    low_path: bool,
    /// Whether this row is expected to converge. Recorded so a fixture that
    /// silently stops converging -- which would make its comparison vacuous,
    /// since two failures agree trivially -- goes red instead of passing.
    expect_success: bool,
    /// Rough class, for the coverage assertions at the end.
    class: Class,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Class {
    /// Transfer time above the parabolic limit: an ellipse.
    Elliptic,
    /// Transfer time BELOW the parabolic limit for this geometry, so the
    /// transfer arc is hyperbolic and `find_xy` solves on `x > 1`. The
    /// pre-existing coverage had none of these.
    SubParabolic,
    /// `m >= 1`: the branch the geometry hoist exists to serve, and the one
    /// whose `find_xy` iteration is most sensitive to a perturbed `lambda`.
    MultiRev,
}

/// r1/r2 are the crate's own LEO pair (a 90-degree transfer between a 6778 km
/// and a 7178 km radius). The parabolic transfer time for that pair is about
/// 902 s, which is what makes 600 s sub-parabolic and 3600 s comfortably
/// elliptic; the multi-rev rows sit above the ~5.8e3 s orbital period.
fn fixtures() -> Vec<Fixture> {
    let r1 = [6778.0, 0.0, 0.0];
    let r2 = [0.0, 7178.0, 0.0];
    // An out-of-plane arrival, so the cross products that build `it1`/`it2`
    // are exercised on a geometry whose angular momentum is not a coordinate
    // axis. The in-plane rows above cannot separate a sign error there.
    let r2_inclined = [0.0, 5075.0, 5075.0];
    vec![
        Fixture {
            name: "leo-90deg-1h-prograde",
            r1,
            r2,
            tof_s: 3600.0,
            m: 0,
            prograde: true,
            low_path: true,
            expect_success: true,
            class: Class::Elliptic,
        },
        Fixture {
            name: "leo-90deg-1h-retrograde",
            r1,
            r2,
            tof_s: 3600.0,
            m: 0,
            prograde: false,
            low_path: true,
            expect_success: true,
            class: Class::Elliptic,
        },
        Fixture {
            name: "leo-inclined-1h-prograde",
            r1,
            r2: r2_inclined,
            tof_s: 3600.0,
            m: 0,
            prograde: true,
            low_path: true,
            expect_success: true,
            class: Class::Elliptic,
        },
        Fixture {
            name: "leo-90deg-600s-subparabolic",
            r1,
            r2,
            tof_s: 600.0,
            m: 0,
            prograde: true,
            low_path: true,
            expect_success: true,
            class: Class::SubParabolic,
        },
        Fixture {
            name: "leo-90deg-300s-subparabolic-fast",
            r1,
            r2,
            tof_s: 300.0,
            m: 0,
            prograde: true,
            low_path: true,
            expect_success: true,
            class: Class::SubParabolic,
        },
        Fixture {
            name: "leo-90deg-1rev-lowpath",
            r1,
            r2,
            tof_s: 12_000.0,
            m: 1,
            prograde: true,
            low_path: true,
            expect_success: true,
            class: Class::MultiRev,
        },
        Fixture {
            name: "leo-90deg-1rev-highpath",
            r1,
            r2,
            tof_s: 12_000.0,
            m: 1,
            prograde: true,
            low_path: false,
            expect_success: true,
            class: Class::MultiRev,
        },
        Fixture {
            name: "leo-90deg-2rev-lowpath",
            r1,
            r2,
            tof_s: 24_000.0,
            m: 2,
            prograde: true,
            low_path: true,
            expect_success: true,
            class: Class::MultiRev,
        },
    ]
}

/// Largest absolute component difference between two velocity triples.
fn worst_component(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// `izzo2015_impl_with_geom(compute_lambert_geometry(..))` solves every
/// fixture to the same velocities as `izzo2015_impl`.
///
/// POISON-PROVEN, and the sensitivity is much better than the bound
/// suggests: scaling `compute_lambert_geometry`'s `sigma` by `1.0 + 1e-9` --
/// one part in a billion -- reds the very first fixture, at a departure
/// velocity difference of 5.69e-9 km/s. Any field this entry point reads
/// works the same way: `ll_base`, `gamma`, `rho`, `sigma`, `t_nd`, `ir1`,
/// `ir2`, `it1_base`, `it2_base`, `r1_norm`, `r2_norm`. `c_norm`, `s` and
/// `s_cubed` are NOT read by `izzo2015_impl_with_geom`, so this pin is
/// structurally blind to those three and does not claim otherwise -- their
/// live readers are the `t_nd` rescale sites elsewhere in the crate.
#[test]
fn geometry_route_matches_whole_problem_route() {
    let fixtures = fixtures();
    assert!(
        fixtures.len() >= 5,
        "the fixture set is the whole strength of this pin; \
         it held {} rows and the floor is 5",
        fixtures.len()
    );

    let mut worst_v1 = 0.0_f64;
    let mut worst_v2 = 0.0_f64;
    let mut compared = 0usize;
    let mut elliptic = 0usize;
    let mut sub_parabolic = 0usize;
    let mut multi_rev = 0usize;

    for fixture in &fixtures {
        let whole = izzo2015_impl(
            MU,
            &fixture.r1,
            &fixture.r2,
            fixture.tof_s,
            fixture.m,
            fixture.prograde,
            fixture.low_path,
            MAXITER,
            ATOL,
            RTOL,
        );
        let geom = compute_lambert_geometry(MU, &fixture.r1, &fixture.r2, fixture.tof_s);
        let via_geom = izzo2015_impl_with_geom(
            &geom,
            fixture.m,
            fixture.prograde,
            fixture.low_path,
            MAXITER,
            ATOL,
            RTOL,
        );

        assert_eq!(
            whole.success, fixture.expect_success,
            "{}: izzo2015_impl convergence changed. A fixture that stopped \
             converging compares two failures, which agree trivially and \
             measure nothing -- fix the fixture, do not relax the flag",
            fixture.name
        );
        assert_eq!(
            via_geom.success, whole.success,
            "{}: the two routes disagree on whether the problem is solvable. \
             They share find_xy, so this is a geometry difference reaching \
             the iteration, not an iteration difference",
            fixture.name
        );
        assert!(
            geom.success,
            "{}: compute_lambert_geometry rejected inputs that izzo2015_impl \
             accepted",
            fixture.name
        );

        if !whole.success {
            continue;
        }
        compared += 1;
        match fixture.class {
            Class::Elliptic => elliptic += 1,
            Class::SubParabolic => sub_parabolic += 1,
            Class::MultiRev => multi_rev += 1,
        }

        let d_v1 = worst_component(&whole.v1, &via_geom.v1);
        let d_v2 = worst_component(&whole.v2, &via_geom.v2);
        worst_v1 = worst_v1.max(d_v1);
        worst_v2 = worst_v2.max(d_v2);

        assert!(
            d_v1 <= V_TOL_KM_S,
            "{}: departure velocity differs by {d_v1:e} km/s between the \
             geometry route and the whole-problem route (bound {V_TOL_KM_S:e}). \
             whole={:?} via_geom={:?}",
            fixture.name,
            whole.v1,
            via_geom.v1,
        );
        assert!(
            d_v2 <= V_TOL_KM_S,
            "{}: arrival velocity differs by {d_v2:e} km/s between the \
             geometry route and the whole-problem route (bound {V_TOL_KM_S:e}). \
             whole={:?} via_geom={:?}",
            fixture.name,
            whole.v2,
            via_geom.v2,
        );
    }

    // The set size and the measured worst case are part of the result: a
    // "clean" that does not say how many rows it compared cannot be told
    // apart from a fixture list that quietly emptied.
    println!(
        "izzo geometry oracle: {compared} of {} fixtures compared \
         ({elliptic} elliptic, {sub_parabolic} sub-parabolic, {multi_rev} multi-rev); \
         worst |dv1| = {worst_v1:e} km/s, worst |dv2| = {worst_v2:e} km/s, \
         bound {V_TOL_KM_S:e}",
        fixtures.len()
    );

    assert!(
        compared >= 5,
        "only {compared} fixtures actually converged and were compared; \
         the pin needs at least 5 live rows"
    );
    assert!(
        sub_parabolic >= 1,
        "no sub-parabolic row converged, so the x > 1 branch of find_xy is \
         unpinned -- the hole this file was written to close"
    );
    assert!(
        multi_rev >= 1,
        "no multi-rev row converged, so the branch the geometry hoist exists \
         to serve is unpinned"
    );
}

/// The comparison can actually see a geometry error.
///
/// Without this, a `compute_lambert_geometry` that returned a geometry the
/// solver happened to ignore would leave the pin above green. Perturbing
/// `ll_base` by one part in 1e6 must move the answer by far more than the
/// tolerance the pin allows.
#[test]
fn a_perturbed_geometry_moves_the_answer_past_the_tolerance() {
    let r1 = [6778.0, 0.0, 0.0];
    let r2 = [0.0, 7178.0, 0.0];
    let tof_s = 3600.0;

    let reference = izzo2015_impl(MU, &r1, &r2, tof_s, 0, true, true, MAXITER, ATOL, RTOL);
    assert!(reference.success, "the reference fixture must converge");

    let mut geom = compute_lambert_geometry(MU, &r1, &r2, tof_s);
    geom.ll_base *= 1.0 + 1e-6;
    let perturbed = izzo2015_impl_with_geom(&geom, 0, true, true, MAXITER, ATOL, RTOL);
    assert!(
        perturbed.success,
        "the perturbation must stay inside the solver's domain, or this \
         proves nothing about the tolerance"
    );

    let moved = worst_component(&reference.v1, &perturbed.v1);
    assert!(
        moved > V_TOL_KM_S,
        "a 1e-6 relative error in the geometry moved the departure velocity \
         by only {moved:e} km/s, which the pin's {V_TOL_KM_S:e} bound would \
         not catch. The bound is too loose to detect a geometry defect"
    );
}
