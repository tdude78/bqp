use wide::f64x4;

// Constant lanes, as `const` items rather than `f64x4::splat` calls.
//
// `wide::f64x4` is a native `__m256d` only when the target has AVX. On the
// aarch64 dev and gate hosts it is two stacked `f64x2` halves, and
// `f64x4::splat(c)` is then an ordinary (non-const) `[c; 4]` array build.
// LLVM's loop-idiom pass rewrites that build into a `memset_pattern16`
// libcall into a stack slot followed by a reload — a libc call, per constant,
// per call to the enclosing function. Measured on the sealed `mf-p64-e24`
// shape before this change: 17 such calls in the prologue of
// `find_xy_simd4_m_variant_per_lane_t` alone, and `memset_pattern16` was
// 13.09% of all busy MF-cell CPU, the second-largest symbol in the profile
// behind that kernel itself.
//
// A `const` item is a rodata constant instead, loaded with one instruction.
// The lane values are identical either way, so this is bit-identical by
// construction; `simd_lane_constants_equal_their_splat_forms` pins that,
// including for the two constants spelled by bits below.
//
// This is a dev/gate-host change. On the x86 clusters the campaign flies
// (`znver2` and up, all AVX2) both forms already lower to one `vbroadcastsd`
// and `memset_pattern16` does not appear at all — it is a Darwin-only libc
// entry point that LLVM emits only when the target library has it.
pub const V_ZERO: f64x4 = f64x4::new([0.0; 4]);
const V_ONE: f64x4 = f64x4::new([1.0; 4]);
const V_NEG_ONE: f64x4 = f64x4::new([-1.0; 4]);
const V_HALF: f64x4 = f64x4::new([0.5; 4]);
const V_TWO: f64x4 = f64x4::new([2.0; 4]);
const V_THREE: f64x4 = f64x4::new([3.0; 4]);
const V_FIVE: f64x4 = f64x4::new([5.0; 4]);
const V_SIX: f64x4 = f64x4::new([6.0; 4]);
const V_SEVEN: f64x4 = f64x4::new([7.0; 4]);
const V_EIGHT: f64x4 = f64x4::new([8.0; 4]);
const V_NAN: f64x4 = f64x4::new([f64::NAN; 4]);
/// `1.0 / 6.0`, the Householder third-order term's coefficient.
const V_INV_6: f64x4 = f64x4::new([1.0 / 6.0; 4]);
/// `(0.6_f64).sqrt()`, the lower edge of `tof_equation_y`'s `hyp2f1b` window.
/// Broadcast of the crate-level `SQRT_0_6` (spelled by bits there, because
/// `f64::sqrt` is not callable in a `const`), so the scalar and SIMD band
/// edges share one definition.
const V_SQRT_0_6: f64x4 = f64x4::new([super::SQRT_0_6; 4]);
/// `(1.4_f64).sqrt()`, the upper edge of the same window, broadcast of the
/// crate-level `SQRT_1_4` for the same reason.
const V_SQRT_1_4: f64x4 = f64x4::new([super::SQRT_1_4; 4]);

/// SIMD version of `compute_y`
#[inline]
fn compute_y_simd(x: f64x4, ll: f64x4) -> f64x4 {
    let ll_sq = ll * ll;
    // FMA (mirror scalar compute_y): ll^2 * (x^2 - 1) + 1
    let rad = ll_sq.mul_add(x * x - V_ONE, V_ONE);
    // max(0.0) to avoid NaN from sqrt of negative
    rad.max(V_ZERO).sqrt()
}

/// SIMD version of `tof_equation_y` with per-lane revolution count (HF-NEW-01).
///
/// This is the only SIMD TOF kernel. Uniform-m callers broadcast their `m`
/// into `m_pi_vec = splat(m * PI)` / `is_m0 = splat(m == 0)`; the per-lane
/// form `m_pi_vec = [m0*PI, m1*PI, m2*PI, m3*PI]` lets the (m, `low_path`)
/// Lambert batch axis work lockstep across different revolution counts.
///
/// Every branch of the scalar [`super::tof_equation_y`] is mirrored here, and in
/// the scalar association: an earlier version computed only the `acos` limb of
/// `compute_psi`, took `sqrt` of the *signed* `1 - x^2`, and divided where the
/// scalar multiplies by a reciprocal. That made it a different function
/// wherever an iterate left `[-1, 1]` or entered the `hyp2f1b` band, and
/// returned NaN for lanes the scalar solver converges — 340 of 86,524 over the
/// production MF operand corpus. The two rare limbs (`asinh` at 0.04% of calls,
/// `hyp2f1b` at 0.55%) are computed per lane behind an `any()` guard, because a
/// table lookup does not vectorise and neither does `wide` supply `asinh`.
#[inline]
fn tof_equation_y_simd_per_lane_m_pi(
    x: f64x4,
    y: f64x4,
    t0: f64x4,
    ll: f64x4,
    m_pi_vec: f64x4,
    is_m0: f64x4,
) -> f64x4 {
    let x2 = x * x;
    let one_minus_x2 = V_ONE - x2;
    // Scalar takes `omx2 = (1.0 - x2).abs()` under the sqrt but divides by the
    // signed `1.0 - x2`.
    let omx2 = one_minus_x2.abs();

    // FMA (mirror scalar compute_psi): x*y + ll*(1 - x^2)
    let arg = ll.mul_add(one_minus_x2, x * y);
    let psi_acos = arg.max(V_NEG_ONE).min(V_ONE).acos();
    // Scalar `compute_psi`: acos on -1 <= x < 1, asinh on x > 1, else 0.
    let in_acos = x.simd_ge(V_NEG_ONE) & x.simd_lt(V_ONE);
    let mut psi = in_acos.select(psi_acos, V_ZERO);
    let gt1 = x.simd_gt(V_ONE);
    if gt1.any() {
        let xa = x.to_array();
        let ya = y.to_array();
        let lla = ll.to_array();
        let mut pa = psi.to_array();
        for (((lane, x_lane), y_lane), ll_lane) in pa.iter_mut().zip(xa).zip(ya).zip(lla) {
            if x_lane > 1.0 {
                // FMA: y - x*ll = (-x).mul_add(ll, y)
                let diff = (-x_lane).mul_add(ll_lane, y_lane);
                *lane = (diff * (x_lane * x_lane - 1.0).sqrt()).asinh();
            }
        }
        psi = f64x4::new(pa);
    }

    // FMA: -x + ll*y -> ll.mul_add(y, -x), against 1/sqrt(omx2) -- the scalar
    // multiplies by a reciprocal rather than dividing, which rounds differently.
    let t_val = (psi + m_pi_vec).mul_add(V_ONE / omx2.sqrt(), ll.mul_add(y, -x)) / one_minus_x2;

    let needs_hyp = is_m0 & x.simd_gt(V_SQRT_0_6) & x.simd_lt(V_SQRT_1_4);
    if !needs_hyp.any() {
        return t_val - t0;
    }
    let xa = x.to_array();
    let ya = y.to_array();
    let lla = ll.to_array();
    let mut ta = t_val.to_array();
    for ((((lane, wanted), x_lane), y_lane), ll_lane) in ta
        .iter_mut()
        .zip(needs_hyp.to_array())
        .zip(xa)
        .zip(ya)
        .zip(lla)
    {
        if wanted.to_bits() == 0 {
            continue;
        }
        // FMA: eta = y - ll*x = (-ll).mul_add(x, y)
        let eta = (-ll_lane).mul_add(x_lane, y_lane);
        // FMA: 1.0 - ll - x*eta = (-x).mul_add(eta, 1.0 - ll)
        let s1 = (-x_lane).mul_add(eta, 1.0 - ll_lane) * 0.5;
        let q = 4.0 / 3.0 * super::hyp2f1b(s1);
        *lane = (eta * eta * eta).mul_add(q, 4.0 * ll_lane * eta) * 0.5;
    }
    f64x4::new(ta) - t0
}

/// SIMD version of `tof_equation_p` (first derivative)
#[inline]
fn tof_equation_p_simd(x: f64x4, y: f64x4, t: f64x4, ll3: f64x4) -> f64x4 {
    let x2 = x * x;
    // FMA (mirror scalar tof_equation_p): (3t)*x - 2, then (2*ll3/y)*x + that
    let term1 = (V_THREE * t).mul_add(x, -V_TWO);
    (V_TWO * ll3 / y).mul_add(x, term1) / (V_ONE - x2)
}

/// SIMD version of `tof_equation_p2` (second derivative)
#[inline]
fn tof_equation_p2_simd(x: f64x4, t: f64x4, dt: f64x4, ll2: f64x4, ll3: f64x4, y3: f64x4) -> f64x4 {
    let x2 = x * x;
    // FMA (mirror scalar tof_equation_p2): (5x)*dt + 3t, then + 2(1-ll2)ll3/y3
    let term1 = (V_FIVE * x).mul_add(dt, V_THREE * t);
    (term1 + V_TWO * (V_ONE - ll2) * ll3 / y3) / (V_ONE - x2)
}

/// SIMD version of `tof_equation_p3` (third derivative)
#[inline]
fn tof_equation_p3_simd(
    x: f64x4,
    dt: f64x4,
    ddt: f64x4,
    ll2: f64x4,
    ll5: f64x4,
    y5: f64x4,
) -> f64x4 {
    let x2 = x * x;
    // FMA (mirror scalar tof_equation_p3): (7x)*ddt + 8dt, then coeff*x + that
    let term1 = (V_SEVEN * x).mul_add(ddt, V_EIGHT * dt);
    let coeff = -(V_SIX * (V_ONE - ll2) * ll5) / y5;
    coeff.mul_add(x, term1) / (V_ONE - x2)
}

/// 4-way Householder iteration with scalar-equivalent adaptive per-lane caps.
///
/// Scalar Izzo uses `adaptive_maxiter` whenever callers pass the production
/// default `maxiter=8`. This wrapper mirrors that contract for SIMD batch
/// lanes while preserving converged lane values after each lane finishes.
///
/// Both production callers (`izzo2015_batch_tof_variable_r2_with_scratch` and
/// `solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_lanes`) seed
/// `p0` from the previous chunk's converged `x`, so this is the SIMD analogue
/// of the scalar `find_xy_seeded` path — and it carries the same
/// seed-dependence hazard. The per-lane caps therefore go through
/// `super::deterministic_maxiter_floor`, exactly as the scalar seeded path
/// does at its `householder_method` call. Without the `deterministic` feature
/// the floor is 0 and `.max(0)` is a no-op on every cap this function can
/// produce (`adaptive_maxiter` returns 4/6/8; the `maxiter` passthrough is
/// already clamped to >= 0 by the callee).
#[must_use]
pub fn householder_simd4_adaptive(
    p0: f64x4,
    t0: f64x4,
    ll: f64x4,
    ll2: f64x4,
    ll3: f64x4,
    ll5: f64x4,
    m: i32,
    maxiter: i32,
    atol: f64,
    rtol: f64,
) -> f64x4 {
    let t_arr = t0.to_array();
    let ll_arr = ll.to_array();
    let lane_maxiters = if maxiter == 8 {
        let [ll0, ll1, ll2, ll3] = ll_arr;
        let [t0, t1, t2, t3] = t_arr;
        [
            super::adaptive_maxiter(ll0, t0, m),
            super::adaptive_maxiter(ll1, t1, m),
            super::adaptive_maxiter(ll2, t2, m),
            super::adaptive_maxiter(ll3, t3, m),
        ]
    } else {
        [maxiter; 4]
    };
    let lane_maxiters = lane_maxiters.map(super::deterministic_maxiter_floor);
    householder_simd4_with_lane_maxiters(p0, t0, ll, ll2, ll3, ll5, m, lane_maxiters, atol, rtol)
}

/// Stamps the shared Householder SIMD4 iteration body for the two kernel
/// entries below. The two expansions are different functions BY DESIGN in
/// exactly two spots, and both arrive as verbatim token arguments so each
/// site's expression tree is reproduced token-for-token:
///
/// * `m_broadcast`: how `m_pi_vec` / `is_m0` come into scope — a uniform-m
///   splat prelude, or empty (the per-lane form takes them as parameters).
/// * `update`: the `p_new` step — plain fused divide, or reciprocal-multiply
///   plus a zero-denominator NaN select. The divide and reciprocal-multiply
///   forms round differently; never merge them.
///
/// Everything else (iters-left/active/converged mask bookkeeping, the
/// `compute_y_simd`/`tof_equation_*` chain, the numer/inner/denom FMA
/// mirrors, the NaN-sentinel epilogue) was a hand-kept ~100-line twin: the
/// ordered-compare retirement fix had to be written and reasoned about twice.
///
/// Shared body notes, kept once here:
///
/// Per-lane bookkeeping is kept in vector registers (lanes are 0.0 or all-1s
/// masks): `active_v` marks lanes still iterating, `converged_v` marks lanes
/// that met tolerance while active. `iters_left` counts down so the per-lane
/// maxiter cap check also stays in-register (iteration caps are small
/// integers, exact in f64). A lane stays active iff it was active, did not
/// just converge, and has iterations left — identical to the old scalar
/// done/converged flags rebuilt via `mask_from_flags` each step, so WHICH
/// lanes iterate (and for how long) is unchanged.
///
/// A lane leaves `active` when it converges, when it runs out of iterations,
/// OR when its `delta` stops being a number. `simd_ge` is an ORDERED compare:
/// false for a lane that just met tolerance, and ALSO false for a lane whose
/// `delta` is NaN. So one mask retires converged and diverged lanes together,
/// where the former `!newly_converged` retired only the converged ones and
/// left a NaN lane spinning to its cap.
///
/// That retirement is value-neutral, and not by measurement alone. `p` for a
/// NaN-delta lane is already NaN (`p_new` is what produced the NaN, and
/// `select` has written it), `compute_y` of a NaN is NaN, so every later
/// `delta` is NaN too, and `NaN < tol` is false at every remaining trip: the
/// lane cannot converge after this point, and a non-converged lane is
/// overwritten with NaN on the way out either way. Retiring it early changes
/// when the `!active_v.any()` break fires and nothing else — lanes still
/// active are masked by their own bit and are untouched.
///
/// It costs one vector compare per trip and removes one `andnot`. Measured by
/// exact trip count on the `batch_tof` workloads: `m_max=0` n=64 unchanged at
/// 96, `m_max=1` n=1024 2059 -> 2040, `m_max=2` n=64 331 -> 271, and the
/// production `max_revs = 4` shape `m_max=4` n=256 938 -> 865 (-7.8%). Both of
/// the shapes the deterministic maxiter floor made more expensive (2043 ->
/// 2059 and 283 -> 331) now sit BELOW their pre-floor counts. A perfect
/// oracle that knew in advance which lanes would never converge saves exactly
/// 60/19/73 trips on those three workloads; this saves 60/19/73. There is no
/// residue left to chase.
macro_rules! householder_simd4_body {
    (
        p0: $p0:expr,
        t0: $t0:expr,
        ll: $ll:expr,
        ll2: $ll2:expr,
        ll3: $ll3:expr,
        ll5: $ll5:expr,
        lane_maxiters: $lane_maxiters:expr,
        atol: $atol:expr,
        rtol: $rtol:expr,
        m_broadcast($m_pi_vec:ident, $is_m0:ident): { $($m_prelude:tt)* },
        update($p:ident, $fval:ident, $numer:ident, $denom:ident -> $p_new:ident): {
            $($update:tt)*
        } $(,)?
    ) => {{
        // two/six not needed: numer/denom fuse via mul_add with 0.5 and 1.0/6.0
        let atol_v = f64x4::splat($atol);
        let rtol_v = f64x4::splat($rtol);

        $($m_prelude)*

        let mut $p = $p0;
        let mut iters_left = f64x4::new($lane_maxiters.map(f64::from));
        let mut active_v = V_ZERO.simd_lt(iters_left);
        let mut converged_v = V_ZERO;
        let max_lane_iter = $lane_maxiters.into_iter().max().unwrap_or(0).max(0);

        for _ in 0..max_lane_iter {
            if !active_v.any() {
                break;
            }

            let y = compute_y_simd($p, $ll);
            let y3 = y * y * y;
            let y5 = y3 * y * y;

            let $fval = tof_equation_y_simd_per_lane_m_pi($p, y, $t0, $ll, $m_pi_vec, $is_m0);
            let t = $fval + $t0;

            let fder = tof_equation_p_simd($p, y, t, $ll3);
            let fder2 = tof_equation_p2_simd($p, t, fder, $ll2, $ll3, y3);
            let fder3 = tof_equation_p3_simd($p, fder, fder2, $ll2, $ll5, y5);

            // FMA (mirror scalar householder_method numerator/denominator):
            //   numerator   = fder^2 - fval*fder2/2
            //   inner       = fder^2 - fval*fder2
            //   denominator = fder*inner + fder3*fval^2/6
            let $numer = (-fder2 * V_HALF).mul_add($fval, fder * fder);
            let inner = (-fder2).mul_add($fval, fder * fder);
            let $denom = (fder3 * $fval * V_INV_6).mul_add($fval, fder * inner);
            $($update)*

            let delta = ($p_new - $p).abs();
            let tol = rtol_v * $p.abs() + atol_v;
            let newly_converged = delta.simd_lt(tol);

            $p = active_v.select($p_new, $p);

            converged_v |= newly_converged & active_v;
            iters_left -= V_ONE;
            let still_iterating = delta.simd_ge(tol);
            active_v = active_v & still_iterating & V_ZERO.simd_lt(iters_left);
        }

        let mut out = $p.to_array();
        for (converged, value) in converged_v.to_array().into_iter().zip(out.iter_mut()) {
            if converged.to_bits() == 0 {
                *value = f64::NAN;
            }
        }
        f64x4::new(out)
    }};
}

pub fn householder_simd4_with_lane_maxiters(
    p0: f64x4,
    t0: f64x4,
    ll: f64x4,
    ll2: f64x4,
    ll3: f64x4,
    ll5: f64x4,
    m: i32,
    lane_maxiters: [i32; 4],
    atol: f64,
    rtol: f64,
) -> f64x4 {
    householder_simd4_body!(
        p0: p0,
        t0: t0,
        ll: ll,
        ll2: ll2,
        ll3: ll3,
        ll5: ll5,
        lane_maxiters: lane_maxiters,
        atol: atol,
        rtol: rtol,
        m_broadcast(m_pi_vec, is_m0): {
            // Broadcast the uniform revolution count into the per-lane
            // kernel's shape. The former plain-m `tof_equation_y_simd` was a
            // lossy copy of the scalar `tof_equation_y`: no `hyp2f1b` band,
            // no `asinh` limb for x > 1, and a signed (not abs) `1 - x^2`
            // under the sqrt, so every hyperbolic lane (x > 1) evaluated to
            // NaN and could never converge.
            let m_pi_vec = f64x4::splat(f64::from(m) * std::f64::consts::PI);
            let is_m0 = f64x4::splat(f64::from(m)).simd_eq(V_ZERO);
        },
        update(p, fval, numer, denom -> p_new): {
            let p_new = p - fval * numer / denom;
        },
    )
}

/// 4-way Householder iteration with per-lane `(ll, ll2, ll3, ll5, m, maxiter)` (HF-NEW-01).
///
/// Used by `find_xy_simd4_m_variant` to pack the (m, `low_path`) and optionally
/// prograde axes of `for_each_lambert_m_prograde_lowpaths` into a single SIMD
/// call. `is_m0` carries the per-lane `m == 0` predicate that
/// `tof_equation_y_simd_per_lane_m_pi` needs to reach its `hyp2f1b` branch.
/// Lanes converge or hit their cap independently; values for converged lanes
/// are preserved while remaining lanes keep iterating. Lanes that never
/// converge return NaN, mirroring the scalar contract.
#[must_use]
pub fn householder_simd4_m_variant(
    p0: f64x4,
    t0: f64x4,
    ll: f64x4,
    ll2: f64x4,
    ll3: f64x4,
    ll5: f64x4,
    m_pi_vec: f64x4,
    is_m0: f64x4,
    lane_maxiters: [i32; 4],
    atol: f64,
    rtol: f64,
) -> f64x4 {
    householder_simd4_body!(
        p0: p0,
        t0: t0,
        ll: ll,
        ll2: ll2,
        ll3: ll3,
        ll5: ll5,
        lane_maxiters: lane_maxiters,
        atol: atol,
        rtol: rtol,
        // `m_pi_vec` / `is_m0` arrive as per-lane parameters; no prelude.
        m_broadcast(m_pi_vec, is_m0): {},
        update(p, fval, numer, denom -> p_new): {
            // Scalar takes the reciprocal once and then multiplies twice; a
            // fused divide rounds differently. Scalar also returns NaN on a
            // zero denominator rather than an infinity.
            let p_new = p - fval * numer * (V_ONE / denom);
            let p_new = denom.simd_eq(V_ZERO).select(V_NAN, p_new);
        },
    )
}

#[cfg(test)]
mod constant_lane_tests {
    use super::{
        V_EIGHT, V_FIVE, V_HALF, V_INV_6, V_NAN, V_NEG_ONE, V_ONE, V_SEVEN, V_SIX, V_SQRT_0_6,
        V_SQRT_1_4, V_THREE, V_TWO, V_ZERO,
    };
    use wide::f64x4;

    /// Every constant lane above must be bit-identical to the `f64x4::splat`
    /// call it replaced, or the swap is a numerical change rather than a
    /// codegen one.
    ///
    /// The two `sqrt` constants are the reason this test exists at all: they
    /// are spelled by bit pattern because `f64::sqrt` cannot be called in a
    /// `const`, so nothing but an assertion ties them to the expressions they
    /// stand for.
    #[test]
    fn simd_lane_constants_equal_their_splat_forms() {
        let cases: [(f64x4, f64); 13] = [
            (V_ZERO, 0.0),
            (V_ONE, 1.0),
            (V_NEG_ONE, -1.0),
            (V_HALF, 0.5),
            (V_TWO, 2.0),
            (V_THREE, 3.0),
            (V_FIVE, 5.0),
            (V_SIX, 6.0),
            (V_SEVEN, 7.0),
            (V_EIGHT, 8.0),
            (V_INV_6, 1.0 / 6.0),
            (V_SQRT_0_6, (0.6_f64).sqrt()),
            (V_SQRT_1_4, (1.4_f64).sqrt()),
        ];
        for (index, (constant, scalar)) in cases.into_iter().enumerate() {
            let splat = f64x4::splat(scalar);
            for (lane, (got, want)) in constant
                .to_array()
                .into_iter()
                .zip(splat.to_array())
                .enumerate()
            {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "constant {index} lane {lane}: {got:e} != splat {want:e}"
                );
            }
        }

        // NaN compares unequal to itself, so it needs the bit check on its own.
        for (lane, (got, want)) in V_NAN
            .to_array()
            .into_iter()
            .zip(f64x4::splat(f64::NAN).to_array())
            .enumerate()
        {
            assert_eq!(got.to_bits(), want.to_bits(), "V_NAN lane {lane}");
        }
    }
}
