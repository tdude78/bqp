//! Izzo (2015) Lambert solver: scalar, SIMD and batch entry points.
//!
//! Absorbed from the `lambert_rs` crate, whose only production consumer was
//! this one. Private, so its kernels are inside rustc's dead-code analysis for
//! the first time -- a library's `pub` items are reachability roots and are
//! never analysed.
//!
//! `lambert_rs` carried two features and BOTH are gone here, not carried over
//! as `cfg`s. Every manifest that ever named the crate -- this one and the
//! `satpy_core` bench dependency -- requested `["parallel", "deterministic"]`,
//! so no build in the workspace ever compiled the other arm, and this crate has
//! no feature of either name for a carried-over `cfg` to read. Left in place
//! they would have silently selected the arms nothing has ever run: the Rayon
//! enumeration paths compiled out, and `LAMBERT_DETERMINISTIC_MAXITER_FLOOR`
//! flipped from 24 to 0. That is a bit-moving change disguised as a file move.
//! The retained arms are the ones every build already took.

use num_traits::ToPrimitive;
use satpy_core::{cross3, norm3};
use wide::f64x4;

use rayon::prelude::*;

// Type aliases for nalgebra integration
pub mod types;
pub use types::{vec3_from_slice, vec3_to_array, Vec3};

// SIMD Lambert solver (wide::f64x4)
pub mod simd;
pub use simd::householder_simd4_adaptive;

// The three integration suites `lambert_rs` carried in `tests/`. All three
// reach entry points that are private now, so none could survive as an
// integration test of this crate; they are inline modules rather than
// deletions. The Orekit fixture moved next to the module that reads it, so its
// `include_str!` no longer has to spell a manifest-relative path.
#[cfg(test)]
mod accuracy_golden_tests;
#[cfg(test)]
mod izzo_geometry_oracle_tests;
#[cfg(test)]
mod orekit_der_oracle_tests;

/// Minimum number of M-enumeration iterations before using parallelism.
/// Based on C++ threshold of `n_pairs` > 4.
// Parallelizing Lambert enumeration only helps when we have enough (m, prograde)
// work to amortize Rayon scheduling overhead. For typical constellation solves
// (e.g., max_revs <= 2), m_max is small and the sequential path is faster.
//
// Default increased from 6 to 12 based on benchmarking:
// - Lambert solve ~300ns/iteration
// - Rayon overhead ~2-5μs
// - Need ~20+ iterations to benefit from parallelism
const LAMBERT_PARALLEL_THRESHOLD: i32 = 12;

/// Pass 13.1: minimum Newton iteration count enforced on every seeded solve.
/// It was the `deterministic` feature of `lambert_rs`, default-on and requested
/// by every consumer; the absorption made it unconditional rather than gating
/// it on a feature this crate does not have. The `adaptive_maxiter` heuristic can
/// leave a Householder/Newton solve at maxiter == 8 with non-converged
/// `|delta|` larger than `tol`, which depends on the warm-start seed. Per the
/// pass 12.8d post-mortem, that seeded the inner OXYMOO NSGA-II with
/// seed-dependent objectives across rayon workers and broke two integration
/// tests when parallel population eval was enabled. Floor at 24 iterations so
/// cold and warm starts reach the same fixed point.
///
/// **What the floor actually removes, corrected 2026-08-06.** The earlier
/// wording here said an exhausted solve returns "a partial-iteration `x`". It
/// does not: `householder_method` falls out of its loop to a bare `f64::NAN`,
/// and both SIMD kernels (`simd::householder_simd4_with_lane_maxiters` and
/// `simd::householder_simd4_m_variant`) overwrite any lane whose
/// `converged` mask stayed clear with `f64::NAN`. Callers skip NaN lanes. So
/// the seed-dependence the floor removes is a
/// *feasibility* flip — whether a candidate exists at all — not an
/// epsilon-different value. Measured over 374,421 elliptical geometries at six
/// warm-seed distances, **no lane that converged under the raw cap ever
/// returned a different `x` under the floor** (0 of 2.2M). Raising the floor
/// is a NaN -> value conversion and nothing else.
///
/// The floor does **not** make the solve seed-independent to the last bit. The
/// iteration returns the first iterate satisfying `|delta| < rtol*|x| + atol`,
/// so different seeds still land on points a few ULP apart (~2e-15 relative,
/// measured). That residual is inside `tol` by construction and is not
/// something an iteration cap can remove.
///
/// **The floor is unreachable from the production MF lane, measured
/// 2026-08-07.** It is applied only by `find_xy_seeded` and by the two SIMD
/// batch kernels, and an instrumented `mf-p64-e24` run (929,359,866 Householder
/// solves) took the seeded path 0 times and the SIMD kernels 0 times: every
/// solve entered through `for_each_lambert_m_prograde_lowpaths_pruned_with_r1`
/// -> `izzo2015_impl_with_geom_fast` -> `find_xy`, which does NOT floor. The
/// observed mean cap was 6.57, i.e. `adaptive_maxiter`'s raw 4/6/8. The
/// x-seeding machinery (`last_x_seeds`, `VariableR2LambertScratch`) is dead
/// there for the same reason: 0 seeded solves, 0 fresh-guess-from-seed-slot
/// solves. Any lever priced against the floor, the seed cache, or the SIMD
/// batch kernels is priced against code the MF lane does not execute. This says
/// nothing about the HF or branch-selected paths, which were not instrumented.
pub const LAMBERT_DETERMINISTIC_MAXITER_FLOOR: i32 = 24;

#[inline]
pub fn deterministic_maxiter_floor(requested: i32) -> i32 {
    requested.max(LAMBERT_DETERMINISTIC_MAXITER_FLOOR)
}

/// Householder step-size tolerance for the Izzo root solve, used as both `atol`
/// and `rtol` by every production caller.
///
/// **This is a step-size bound, not an accuracy bound, and the two differ by a
/// fourth power.** The loop tests `|x_{n+1} - x_n| < rtol*|x_n| + atol` and then
/// returns `x_{n+1}` — one full 4th-order Householder step past the point where
/// the test was taken. So a solve that stops on a step of size `s` returns a
/// root whose error is O(s^4), not O(s). At `s = 1e-6` that is ~1e-24: the
/// returned `x` is converged to well past double precision, and the iteration
/// the old `1e-9` bought was adding rounding noise rather than digits.
///
/// Measured on a 226,124-row corpus sampled 1-in-1024 from the production
/// `mf-p64-e24` MF batch (every `find_xy` entry, all m, all branches):
///
/// | tolerance | Householder trips | max abs shift in x vs the 1e-9 root |
/// |-----------|-------------------|-------------------------------------|
/// | 1e-9 (was)| 684,639 (mean 3.028) | -                                |
/// | 1e-8      | -2.23%            | 1.03e-14                            |
/// | 1e-7      | -5.88%            | 1.03e-14                            |
/// | **1e-6**  | **-11.40%**       | **1.08e-13** (mean 3.5e-17)         |
/// | 1e-5      | -16.24%           | 8.5e-13                             |
/// | 1e-4      | -20.00%           | 9.7e-11                             |
///
/// The shifts are compared against the roots the ORIGINAL 1e-9 loop produced,
/// never against the relaxed run's own interval. A perfect oracle seed — one
/// that started the iteration exactly at the root — would still cost one trip
/// per solve to certify convergence, so -66.97% is the floor for ANY
/// seed-quality work and -11.40% of it is available from the exit test alone.
///
/// 1e-6 is the stopping point because 1e-5 and below leave the regime where the
/// returned root is exact to double precision: the max shift climbs two orders
/// per tolerance decade once the asymptotic 4th-order rate stops holding for the
/// near-parabolic tail.
///
/// `compute_t_min`'s Halley solve does NOT keep 1e-9 on the paths this
/// constant reaches, and an earlier revision of this note claimed it did.
/// Three of its five call sites forward the CALLER's tolerances — `find_xy`,
/// `find_xy_seeded`, and `find_xy_simd4_m_variant_per_lane_t`, which is the
/// production branch enumerator — so those decide `t < t_min` at 1e-6. Only
/// the two variable-r2 batch entries (`izzo2015_batch_tof_variable_r2_with_scratch`,
/// `solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_lanes`)
/// hardcode 1e-9. That output feeds a FEASIBILITY comparison rather than a
/// returned value, so unlike the root tolerance above it decides whether a
/// candidate exists at all, and the two groups of call sites can disagree
/// about a boundary-revolution branch. Read the tolerance at the call site.
pub const CONVERGENCE_TOL: f64 = 1e-6;

/// Early exit threshold for M-enumeration (km/s).
/// If M=0 solution has `dv_depart` < threshold, skip higher M values.
/// Set to 0.0 to disable (default). Typical value: 0.5-1.0 km/s.
/// For LEO constellation transfers, M=0 almost always beats multi-rev.
const LAMBERT_EARLY_EXIT_THRESHOLD: f64 = 0.0;

// Constant lane arrays, as `const` items rather than inline `[c; 4]` builds.
//
// An inline `[f64::NAN; 4]` local init lowers through LLVM's loop-idiom pass
// to a `memset_pattern16` libcall on aarch64 — measured at 45% of
// `find_xy_simd4_m_variant_per_lane_t`'s samples before it was worked around.
// A `const` item is a 32-byte rodata copy (plain vector loads/stores), so the
// pattern-fill libcall never appears. The stored bits are identical either
// way; see `simd.rs` for the same fix on the `f64x4` lane constants.
/// Four-lane NaN sentinel fill.
const NAN4: [f64; 4] = [f64::NAN; 4];
/// Padding `ll` for inactive pack lanes: `|ll| >= 1` is rejected by the
/// kernel's pre-pass, so a padding lane costs no iterations.
const LL_PAD4: [f64; 4] = [2.0; 4];
/// Padding non-dimensional TOF for inactive pack lanes; never read because
/// the `LL_PAD4` reject fires first.
const T_PAD4: [f64; 4] = [1.0; 4];

#[inline]
fn f64_to_i32_saturating(value: f64) -> i32 {
    if value.is_nan() {
        0
    } else {
        value.to_i32().unwrap_or_else(|| {
            if value.is_sign_negative() {
                i32::MIN
            } else {
                i32::MAX
            }
        })
    }
}

#[inline]
fn i32_to_usize_or_zero(value: i32) -> usize {
    usize::try_from(value).unwrap_or(0)
}

#[inline]
fn usize_to_f64_or_infinity(value: usize) -> f64 {
    value.to_f64().unwrap_or(f64::INFINITY)
}

#[inline]
fn revolution_pair_count(max_revolutions: i32) -> usize {
    max_revolutions
        .checked_add(1)
        .and_then(|count| count.checked_mul(2))
        .and_then(|count| usize::try_from(count).ok())
        .unwrap_or(0)
}

#[inline]
fn revolution_branch_index(revolutions: i32, prograde: bool) -> usize {
    revolutions
        .checked_mul(2)
        .and_then(|base| base.checked_add(i32::from(!prograde)))
        .and_then(|index| usize::try_from(index).ok())
        .unwrap_or(0)
}

#[inline]
fn distance3(lhs: &[f64; 3], rhs: &[f64; 3]) -> f64 {
    let [lhs_x, lhs_y, lhs_z] = *lhs;
    let [rhs_x, rhs_y, rhs_z] = *rhs;
    let dx = lhs_x - rhs_x;
    let dy = lhs_y - rhs_y;
    let dz = lhs_z - rhs_z;
    dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt()
}

#[inline]
const fn state6(position: &[f64; 3], velocity: &[f64; 3]) -> [f64; 6] {
    let [px, py, pz] = *position;
    let [vx, vy, vz] = *velocity;
    [px, py, pz, vx, vy, vz]
}

#[inline]
fn add3(lhs: &[f64; 3], rhs: &[f64; 3]) -> [f64; 3] {
    let [lhs_x, lhs_y, lhs_z] = *lhs;
    let [rhs_x, rhs_y, rhs_z] = *rhs;
    [lhs_x + rhs_x, lhs_y + rhs_y, lhs_z + rhs_z]
}

#[inline]
fn subtract3(lhs: &[f64; 3], rhs: &[f64; 3]) -> [f64; 3] {
    let [lhs_x, lhs_y, lhs_z] = *lhs;
    let [rhs_x, rhs_y, rhs_z] = *rhs;
    [lhs_x - rhs_x, lhs_y - rhs_y, lhs_z - rhs_z]
}

#[inline]
const fn feasible_revolution_range(m_max_feasible: i32) -> std::ops::RangeInclusive<i32> {
    0..=m_max_feasible
}

/// Quickly compute maximum feasible m value without full solver setup.
///
/// This is a cheap approximation that may over-estimate `m_max` slightly,
/// but never under-estimates. Used for pre-filtering batch solves.
///
/// Returns the theoretical maximum number of complete revolutions possible
/// for the given geometry and time of flight.
#[inline]
#[must_use]
pub fn compute_m_max_fast(r1: &[f64; 3], r2: &[f64; 3], tof: f64, mu: f64) -> i32 {
    let r1_norm = norm3(r1);
    let r2_norm = norm3(r2);
    let c = [r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]];
    let c_norm = norm3(&c);
    let s = (r1_norm + r2_norm + c_norm) * 0.5;

    if s <= 0.0 || !s.is_finite() {
        return 0;
    }

    // Non-dimensional time
    let s_cubed = s * s * s;
    let t_nd = (2.0 * mu / s_cubed).sqrt() * tof;

    // Upper bound on m_max: floor(t_nd / PI)
    f64_to_i32_saturating((t_nd / std::f64::consts::PI).floor().max(0.0))
}

#[derive(Clone, Copy)]
pub struct LambertResult {
    pub v1: [f64; 3],
    pub v2: [f64; 3],
    pub success: bool,
}

/// Precomputed geometry for Lambert problem.
///
/// Hoisting these values out of the M-loop improves performance for multi-rev
/// cases. `Default` is the all-zero (unsuccessful) geometry, used only as
/// inert padding in fixed-size lane buffers.
#[derive(Debug, Clone, Copy, Default)]
pub struct LambertGeometry {
    pub r1_norm: f64,
    pub r2_norm: f64,
    pub c_norm: f64,
    pub s: f64,
    pub s_cubed: f64,
    pub ir1: Vec3,
    pub ir2: Vec3,
    pub it1_base: Vec3, // for prograde=true
    pub it2_base: Vec3, // for prograde=true
    pub ll_base: f64,   // for prograde=true
    pub gamma: f64,
    pub rho: f64,
    pub sigma: f64,
    pub t_nd: f64,
    pub success: bool,
}

/// Cached departure-side (r1) quantities for `compute_lambert_geometry`.
///
/// `r1_norm` and `ir1` are produced by the exact same operations
/// (`vec3_from_slice`, nalgebra `norm()`, componentwise scalar divide) that
/// `compute_lambert_geometry` runs internally, so hoisting this cache out of
/// a variable-r2 batch loop is a pure loop-invariant lift: every geometry
/// built through `compute_lambert_geometry_with_r1` is bit-identical to the
/// uncached path.
#[derive(Debug, Clone, Copy)]
pub struct LambertR1Cache {
    r1_vec: Vec3,
    r1_norm: f64,
    ir1: Vec3,
}

impl LambertR1Cache {
    #[inline]
    #[must_use]
    pub fn new(r1: &[f64; 3]) -> Self {
        let r1_vec = vec3_from_slice(r1);
        let r1_norm = r1_vec.norm();
        // `ir1` is only read after the `r1_norm <= 0.0` guard in
        // `compute_lambert_geometry_with_r1`, mirroring the single-shot path
        // which never divides by a non-positive norm.
        let ir1 = if r1_norm > 0.0 {
            r1_vec / r1_norm
        } else {
            Vec3::zeros()
        };
        Self {
            r1_vec,
            r1_norm,
            ir1,
        }
    }
}

/// Per-lane Lambert inputs that depend only on `(mu, r1, r2, tof)`.
///
/// The selected-branch batch entry re-solves the same `(r1, r2_vec, tofs)`
/// batch once per `(rev, low_path)` branch — nine times at `max_revs = 4` —
/// and each of those passes rebuilt the same `compute_m_max_fast` result and
/// the same `LambertGeometry` for every lane. Neither depends on the branch
/// being solved, so hoisting them above the branch loop is a loop-invariant
/// lift of the same kind as [`LambertR1Cache`]: identical functions on
/// identical inputs, so every reused value is bit-identical to the one it
/// replaces.
///
/// Lane `i` corresponds to `r2_vec[i]` / `tofs[i]`, so a prep built for one
/// batch is only valid for that batch.
#[derive(Debug, Default, Clone)]
pub struct BranchLanePrep {
    /// `(compute_m_max_fast(..), compute_lambert_geometry_with_r1(..))` per lane.
    lanes: Vec<(i32, LambertGeometry)>,
}

impl BranchLanePrep {
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Rebuild the prep for one `(mu, r1, r2_vec, tofs)` batch.
    ///
    /// Leaves the prep empty — so callers fall back to per-lane computation —
    /// when `r2_vec` and `tofs` disagree in length.
    pub fn rebuild(&mut self, mu: f64, r1: &[f64; 3], r2_vec: &[[f64; 3]], tofs: &[f64]) {
        self.lanes.clear();
        if r2_vec.len() != tofs.len() {
            return;
        }
        self.lanes.reserve(tofs.len());
        let r1_cache = LambertR1Cache::new(r1);
        for (r2, &tof) in r2_vec.iter().zip(tofs.iter()) {
            let m_max_fast = compute_m_max_fast(r1, r2, tof, mu);
            let geom = compute_lambert_geometry_with_r1(mu, &r1_cache, r2, tof);
            self.lanes.push((m_max_fast, geom));
        }
    }
}

/// Compute invariant geometry for Lambert problem.
#[inline]
#[must_use]
pub fn compute_lambert_geometry(
    mu: f64,
    r1: &[f64; 3],
    r2: &[f64; 3],
    tof: f64,
) -> LambertGeometry {
    compute_lambert_geometry_with_r1(mu, &LambertR1Cache::new(r1), r2, tof)
}

/// `compute_lambert_geometry` fast entry taking precomputed r1-side values.
///
/// `r1_cache` must be `LambertR1Cache::new(r1)` for the departure position of
/// this problem; only r2/tof-dependent work is redone, skipping one sqrt, the
/// r1 normalization divides, and the r1 conversion per call.
#[inline]
#[must_use]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Lambert geometry requires the established floating-point evaluation order"
)]
pub fn compute_lambert_geometry_with_r1(
    mu: f64,
    r1_cache: &LambertR1Cache,
    r2: &[f64; 3],
    tof: f64,
) -> LambertGeometry {
    let mut geom = LambertGeometry {
        r1_norm: 0.0,
        r2_norm: 0.0,
        c_norm: 0.0,
        s: 0.0,
        s_cubed: 0.0,
        ir1: Vec3::zeros(),
        ir2: Vec3::zeros(),
        it1_base: Vec3::zeros(),
        it2_base: Vec3::zeros(),
        ll_base: 0.0,
        gamma: 0.0,
        rho: 0.0,
        sigma: 0.0,
        t_nd: 0.0,
        success: false,
    };

    if !mu.is_finite() || mu <= 0.0 || !tof.is_finite() || tof <= 0.0 {
        return geom;
    }

    // Convert to nalgebra Vec3 for SIMD operations
    let r1_vec = r1_cache.r1_vec;
    let r2_vec = vec3_from_slice(r2);

    geom.r1_norm = r1_cache.r1_norm;
    geom.r2_norm = r2_vec.norm();
    if geom.r1_norm <= 0.0 || geom.r2_norm <= 0.0 {
        return geom;
    }

    let c = r2_vec - r1_vec;
    geom.c_norm = c.norm();
    if geom.c_norm <= 0.0 {
        return geom;
    }

    geom.s = (geom.r1_norm + geom.r2_norm + geom.c_norm) * 0.5;
    if !geom.s.is_finite() || geom.s <= 0.0 {
        return geom;
    }

    geom.ir1 = r1_cache.ir1;
    geom.ir2 = r2_vec / geom.r2_norm;

    let mut ih = geom.ir1.cross(&geom.ir2);
    let ih_norm = ih.norm();
    if ih_norm <= 0.0 {
        return geom;
    }
    ih /= ih_norm;

    geom.ll_base = (1.0 - (geom.c_norm / geom.s).clamp(0.0, 1.0)).sqrt();
    if ih.z < 0.0 {
        geom.ll_base = -geom.ll_base;
        geom.it1_base = geom.ir1.cross(&ih);
        geom.it2_base = geom.ir2.cross(&ih);
    } else {
        geom.it1_base = ih.cross(&geom.ir1);
        geom.it2_base = ih.cross(&geom.ir2);
    }

    geom.s_cubed = geom.s * geom.s * geom.s;
    if geom.s_cubed <= 0.0 {
        return geom;
    }

    geom.t_nd = (2.0 * mu / geom.s_cubed).sqrt() * tof;
    geom.gamma = (mu * geom.s / 2.0).sqrt();
    geom.rho = (geom.r1_norm - geom.r2_norm) / geom.c_norm;
    geom.sigma = (1.0 - geom.rho * geom.rho).max(0.0).sqrt();
    geom.success = true;

    geom
}

#[inline]
pub fn compute_y(x: f64, ll: f64) -> f64 {
    // FMA: 1.0 - ll^2 * (1 - x^2) = ll^2 * (x^2 - 1) + 1
    let ll_sq = ll * ll;
    let rad = ll_sq.mul_add(x * x - 1.0, 1.0);
    rad.max(0.0).sqrt()
}

#[inline]
fn compute_psi(x: f64, y: f64, ll: f64) -> f64 {
    if (-1.0..1.0).contains(&x) {
        // FMA: x*y + ll*(1 - x^2)
        let arg = ll.mul_add(1.0 - x * x, x * y);
        arg.clamp(-1.0, 1.0).acos()
    } else if x > 1.0 {
        // FMA: y - x*ll = (-x).mul_add(ll, y)
        let diff = (-x).mul_add(ll, y);
        (diff * (x * x - 1.0).sqrt()).asinh()
    } else {
        0.0
    }
}

/// Hypergeometric function 2F1(3, 1; 5/2; x) via series summation.
/// Can be slow for x close to 1.0 (up to 1000 iterations).
///
/// `const fn` so `HYP2F1B_TABLE` below is built by CTFE. Const evaluation of
/// IEEE `+,-,*,/`, `abs`, and `max` is defined to match the runtime
/// operations bit-for-bit, and this same body remains the runtime fallback
/// for `x` outside the table window;
/// `hyp2f1b_table_matches_runtime_series_bitwise` pins the two evaluation
/// modes against each other entry by entry.
#[expect(
    clippy::while_float,
    reason = "the f64 loop counter walks exact integers 0..1000, a const-callable stand-in for the former i32 counter"
)]
#[inline]
const fn hyp2f1b_series(x: f64) -> f64 {
    const TOL: f64 = 1e-15;
    if x >= 1.0 {
        return f64::INFINITY;
    }
    let mut res = 1.0;
    let mut term = 1.0;
    // f64 accumulator replacing the former i32 counter's `f64::from(ii)`
    // (`From` is not const-callable): exact for every integer up to the
    // 1000-iteration cap, so the produced f64 stream is unchanged.
    let mut i_f = 0.0_f64;
    while i_f < 1000.0 {
        let scale = ((3.0 + i_f) * (1.0 + i_f) * x) / ((2.5 + i_f) * (i_f + 1.0));
        term *= scale;
        res += term;
        if term.abs() <= TOL * res.abs().max(1.0) {
            return res;
        }
        i_f += 1.0;
    }
    res
}

// =============================================================================
// Hyp2f1b Lookup Table for Fast Near-Parabolic Computation
// =============================================================================

const HYP2F1B_TABLE_SIZE: usize = 512;
const HYP2F1B_X_MIN: f64 = -0.5;
const HYP2F1B_X_MAX: f64 = 0.95;
const HYP2F1B_STEP: f64 = (HYP2F1B_X_MAX - HYP2F1B_X_MIN) / 511.0;

/// `usize` -> `f64` for the const table builder. Exact for every index it is
/// fed (i < 512, far below f64's 2^53 exact-integer range). Mirrors
/// `usize_to_f64_or_infinity`, whose `ToPrimitive` body is not const-callable;
/// the bitwise table pin test recomputes through the runtime helper to tie
/// the two together.
#[expect(
    clippy::as_conversions,
    reason = "table indices are bounded far below f64's exact integer range"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "table indices are bounded far below f64's exact integer range"
)]
const fn table_index_as_f64(value: usize) -> f64 {
    value as f64
}

/// Builds `HYP2F1B_TABLE`. Body is the former `LazyLock` initializer with the
/// iterator loop respelled for const context (no `iter_mut` in CTFE); the
/// per-entry arithmetic is token-identical.
#[expect(
    clippy::indexing_slicing,
    reason = "index is the loop bound, in range by construction; const context has no iter_mut"
)]
const fn build_hyp2f1b_table() -> [(f64, f64); HYP2F1B_TABLE_SIZE] {
    let mut table = [(0.0, 0.0); HYP2F1B_TABLE_SIZE];
    let mut i = 0_usize;
    while i < HYP2F1B_TABLE_SIZE {
        let x = HYP2F1B_X_MIN + table_index_as_f64(i) * HYP2F1B_STEP;
        let val = hyp2f1b_series(x);
        let h = 1e-6;
        let deriv = if x + h < 1.0 {
            (hyp2f1b_series(x + h) - hyp2f1b_series(x - h)) / (2.0 * h)
        } else {
            (hyp2f1b_series(x) - hyp2f1b_series(x - h)) / h
        };
        table[i] = (val, deriv);
        i = i.saturating_add(1);
    }
    table
}

/// Precomputed hyp2f1b values and derivatives for cubic interpolation.
///
/// Built by CTFE instead of the former `LazyLock`: the table is rodata, so
/// lookups in the near-parabolic Householder band skip the once-init atomic
/// check (three derefs per interpolation) and the first Lambert solve in a
/// process no longer pays the 512-entry series fill. Const eval of the
/// IEEE-only series is bit-identical to the runtime fill it replaces;
/// `hyp2f1b_table_matches_runtime_series_bitwise` enforces that per entry.
static HYP2F1B_TABLE: [(f64, f64); HYP2F1B_TABLE_SIZE] = build_hyp2f1b_table();

/// Fast hyp2f1b using cubic Hermite interpolation from precomputed table.
#[inline]
fn hyp2f1b(x: f64) -> f64 {
    if (HYP2F1B_X_MIN..=HYP2F1B_X_MAX).contains(&x) {
        let t = (x - HYP2F1B_X_MIN) / HYP2F1B_STEP;
        let i = t.floor().to_usize().unwrap_or(0);
        if i >= HYP2F1B_TABLE_SIZE - 1 {
            return HYP2F1B_TABLE
                .last()
                .map_or(f64::INFINITY, |&(value, _)| value);
        }
        let Some((v0, d0)) = HYP2F1B_TABLE.get(i).copied() else {
            return hyp2f1b_series(x);
        };
        let Some((v1, d1)) = i
            .checked_add(1)
            .and_then(|next| HYP2F1B_TABLE.get(next))
            .copied()
        else {
            return hyp2f1b_series(x);
        };
        let u = t - usize_to_f64_or_infinity(i);
        let u2 = u * u;
        let u3 = u2 * u;
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        h00 * v0 + h10 * HYP2F1B_STEP * d0 + h01 * v1 + h11 * HYP2F1B_STEP * d1
    } else {
        hyp2f1b_series(x)
    }
}

/// `(0.6_f64).sqrt()`, the lower edge of `tof_equation_y`'s `hyp2f1b` window.
/// `f64::sqrt` is not callable in a `const`, so the value is spelled by bits;
/// `sqrt_window_consts_equal_their_sqrt_forms` ties the bits to the sqrt
/// expressions they stand for, and `simd.rs` rebuilds its `V_SQRT_*` lanes
/// from these scalars so one definition covers both band edges.
pub const SQRT_0_6: f64 = f64::from_bits(0x3fe8_c97e_f43f_7248);
/// `(1.4_f64).sqrt()`, the upper edge of the same window. Spelled by bits for
/// the same reason.
pub const SQRT_1_4: f64 = f64::from_bits(0x3ff2_ee73_dadc_9b57);

#[inline]
fn tof_equation_y(x: f64, y: f64, t0: f64, ll: f64, m: i32) -> f64 {
    let t_val = if m == 0 && x > SQRT_0_6 && x < SQRT_1_4 {
        // FMA: eta = y - ll*x = (-ll).mul_add(x, y)
        let eta = (-ll).mul_add(x, y);
        // FMA: 1.0 - ll - x*eta = (-x).mul_add(eta, 1.0 - ll)
        let s1 = (-x).mul_add(eta, 1.0 - ll) * 0.5;
        let q = 4.0 / 3.0 * hyp2f1b(s1);
        (eta * eta * eta).mul_add(q, 4.0 * ll * eta) * 0.5
    } else {
        let psi = compute_psi(x, y, ll);
        let x2 = x * x;
        let omx2 = (1.0 - x2).abs();
        // FMA: -x + ll*y = ll.mul_add(y, -x)
        (psi + f64::from(m) * std::f64::consts::PI).mul_add(1.0 / omx2.sqrt(), ll.mul_add(y, -x))
            / (1.0 - x2)
    };
    t_val - t0
}

#[inline]
fn tof_equation(x: f64, t0: f64, ll: f64, m: i32) -> f64 {
    let y = compute_y(x, ll);
    tof_equation_y(x, y, t0, ll, m)
}

#[inline]
fn tof_equation_p(x: f64, y: f64, t: f64, two_ll3: f64, one_minus_x2: f64) -> f64 {
    // FMA: 3*t*x - 2 + 2*ll^3*x/y
    let term1 = (3.0 * t).mul_add(x, -2.0);
    let term2 = (two_ll3 / y).mul_add(x, term1);
    term2 / one_minus_x2
}

#[inline]
fn tof_equation_p2(
    x: f64,
    y: f64,
    t: f64,
    dt: f64,
    two_one_minus_ll2_ll3: f64,
    one_minus_x2: f64,
) -> f64 {
    // FMA: 3*t + 5*x*dt + 2*(1 - ll^2)*ll^3 / y^3
    let y_cubed = y * y * y;
    // 3*t + 5*x*dt = (5*x).mul_add(dt, 3*t)
    let term1 = (5.0 * x).mul_add(dt, 3.0 * t);
    // Add the third term
    let term2 = term1 + two_one_minus_ll2_ll3 / y_cubed;
    term2 / one_minus_x2
}

#[inline]
fn tof_equation_p3(
    x: f64,
    y: f64,
    dt: f64,
    ddt: f64,
    six_one_minus_ll2_ll5: f64,
    one_minus_x2: f64,
) -> f64 {
    // FMA: 7*x*ddt + 8*dt - 6*(1 - ll^2)*ll^5*x / y^5
    let y_5 = y * y * y * y * y;
    // 7*x*ddt + 8*dt = (7*x).mul_add(ddt, 8*dt)
    let term1 = (7.0 * x).mul_add(ddt, 8.0 * dt);
    // -6*(1 - ll^2)*ll^5*x / y^5
    let coeff = -six_one_minus_ll2_ll5 / y_5;
    coeff.mul_add(x, term1) / one_minus_x2
}

/// Halley iteration for the multi-revolution TOF minimum, i.e. the root of
/// `dT/dx` at fixed `m`. Only [`compute_t_min`] calls it.
///
/// `t` must be `T(x)` AT THE CURRENT `x`: the derivative formulas
/// ([`tof_equation_p`] and its two successors) all carry `T` as an argument
/// because the analytic derivative contains it. Until R21 this function passed
/// the caller's `t0` — `T` at the SEED, `x = 0.1` — into all three, unchanged
/// for every iteration. Halley then converged to a root of a perturbed
/// equation, so `x_min` was wrong and `t_min` came out too LARGE: measured over
/// 10,005 `(ll, m)` rows against a derivative-free reference minimum, the
/// shipped value overshot by up to 2.4796e-4 non-dimensional (4.2613e-5
/// relative) and overshot on 10,003 of them. `compute_t_min` feeds the
/// `t < t_min` feasibility prune, so overshoot DROPS revolution branches that
/// exist. Refreshing `t` collapses the error to 1.0658e-14 absolute /
/// 5.9666e-16 relative, which is the reference's own noise floor.
///
/// The refresh is written differently from [`householder_method`]'s
/// `let t = fval + t0;` ON PURPOSE — do not "unify" them. There, `t0` is a
/// genuine target time and `fval = T(x) - t0` is the residual the update needs
/// anyway, so recovering `T(x)` costs nothing extra. Here the residual is never
/// used: this iteration drives `dT/dx` to zero, not `T - t0`, so re-adding a
/// subtracted `t0` would only invite cancellation for the appearance of
/// symmetry. Measured: over the same 10,005 rows the two forms agree
/// BIT-FOR-BIT, so the choice costs nothing and is made for readability.
#[inline]
fn halley_method(mut p0: f64, ll: f64, m: i32, atol: f64, rtol: f64, maxiter: i32) -> f64 {
    // PRE-COMPUTE LOOP INVARIANTS (hoisted from loop, like C++)
    let ll2 = ll * ll;
    let ll3 = ll2 * ll;
    let ll5 = ll3 * ll2;
    let one_minus_ll2 = 1.0 - ll2;
    let two_ll3 = 2.0 * ll3;
    let two_one_minus_ll2_ll3 = 2.0 * one_minus_ll2 * ll3;
    let six_one_minus_ll2_ll5 = 6.0 * one_minus_ll2 * ll5;

    for _ in 0..maxiter {
        let y = compute_y(p0, ll);
        // T at the CURRENT x. `compute_t_min` works in the T0 = 0 frame, so
        // this is the same quantity it hands back as `t_min`.
        let t = tof_equation_y(p0, y, 0.0, ll, m);
        let one_minus_x2 = 1.0 - p0 * p0;
        let fder = tof_equation_p(p0, y, t, two_ll3, one_minus_x2);
        let fder2 = tof_equation_p2(p0, y, t, fder, two_one_minus_ll2_ll3, one_minus_x2);
        if fder2 == 0.0 {
            return f64::NAN;
        }
        let fder3 = tof_equation_p3(p0, y, fder, fder2, six_one_minus_ll2_ll5, one_minus_x2);
        // FMA: denominator = 2*fder2^2 - fder*fder3 = (2*fder2).mul_add(fder2, -fder*fder3)
        let denom = (2.0 * fder2).mul_add(fder2, -fder * fder3);
        let inv_denom = 1.0 / denom;
        let p = p0 - 2.0 * fder * fder2 * inv_denom;
        if (p - p0).abs() < rtol * p0.abs() + atol {
            return p;
        }
        p0 = p;
    }
    f64::NAN
}

/// 4th order Householder iteration for Lambert solver.
///
/// The Householder method of order 4 uses up to the 3rd derivative:
///   x_{n+1} = `x_n` - f/f' * [1 + f*f''/(2*f'^2) + f^2*(3*f''^2 - f'*f''')/(6*f'^4)]
///
/// Simplified form used here:
///   delta = f * (f'^2 - f*f''/2) / (f'^3 - f*f'*f'' + f'''*f^2/6)
///
/// Typical convergence: 3-5 iterations vs Newton's 8-10.
#[inline]
fn householder_method(
    mut p0: f64,
    t0: f64,
    ll: f64,
    m: i32,
    atol: f64,
    rtol: f64,
    maxiter: i32,
) -> f64 {
    const INV_6: f64 = 1.0 / 6.0;
    // PRE-COMPUTE LOOP INVARIANTS (hoisted from loop, like C++)
    let ll2 = ll * ll;
    let ll3 = ll2 * ll;
    let ll5 = ll3 * ll2;
    let one_minus_ll2 = 1.0 - ll2;
    let two_ll3 = 2.0 * ll3;
    let two_one_minus_ll2_ll3 = 2.0 * one_minus_ll2 * ll3;
    let six_one_minus_ll2_ll5 = 6.0 * one_minus_ll2 * ll5;

    for _ in 0..maxiter {
        let y = compute_y(p0, ll);
        let fval = tof_equation_y(p0, y, t0, ll, m);
        let t = fval + t0;
        let one_minus_x2 = 1.0 - p0 * p0;
        let fder = tof_equation_p(p0, y, t, two_ll3, one_minus_x2);
        let fder2 = tof_equation_p2(p0, y, t, fder, two_one_minus_ll2_ll3, one_minus_x2);
        let fder3 = tof_equation_p3(p0, y, fder, fder2, six_one_minus_ll2_ll5, one_minus_x2);

        // Householder 4th order update
        // numerator = f'^2 - f*f''/2
        // denominator = f'^3 - f*f'*f'' + f'''*f^2/6
        // delta = f * numerator / denominator
        // FMA: numerator = fder^2 - fval*fder2/2 = (-fder2*0.5).mul_add(fval, fder^2)
        let numerator = (-fder2 * 0.5).mul_add(fval, fder * fder);
        // FMA: denominator = fder*(fder^2 - fval*fder2) + fder3*fval^2/6
        // Let inner = fder^2 - fval*fder2 = (-fder2).mul_add(fval, fder^2)
        let inner = (-fder2).mul_add(fval, fder * fder);
        // denominator = fder*inner + fder3*fval^2/6 = (fder3*fval*(1/6)).mul_add(fval, fder*inner)
        let denominator = (fder3 * fval * INV_6).mul_add(fval, fder * inner);
        if denominator == 0.0 {
            return f64::NAN;
        }
        let inv_denominator = 1.0 / denominator;
        let p = p0 - fval * numerator * inv_denominator;
        if (p - p0).abs() < rtol * p0.abs() + atol {
            return p;
        }
        p0 = p;
    }
    f64::NAN
}

#[inline]
fn compute_t_min(ll: f64, m: i32, maxiter: i32, atol: f64, rtol: f64) -> (f64, f64) {
    if (ll - 1.0).abs() < 1e-15 {
        let x_min = 0.0;
        let t_min = tof_equation(x_min, 0.0, ll, m);
        (x_min, t_min)
    } else if m == 0 {
        (f64::INFINITY, 0.0)
    } else {
        let x_i = 0.1;
        let x_min = halley_method(x_i, ll, m, atol, rtol, maxiter);
        let t_min = tof_equation(x_min, 0.0, ll, m);
        (x_min, t_min)
    }
}

/// Seed-grade cube root: Kahan-style exponent bit-hack start plus two Halley
/// steps, replacing libm `cbrt` in [`initial_guess`] (R18, the cbrt lead from
/// the MF cost map — 5.9% of an MF cell, all of it here).
///
/// Why this passes the bar that killed the fitted-acos lane (a fitted-libm
/// lane is COVERAGE x DEGREE): coverage is 100% — every production `cbrt` in
/// this crate sits in `cbrt_squared` below — and the degree side is real
/// because `cbrt` has no hardware assist to lean on: measured on Rome
/// (glibc), libm `cbrt` is 20.9 ns/call throughput against 7.7 ns for this
/// exact body (2.72x), 32.2 vs 21.6 ns latency-bound (1.50x). Contrast acos,
/// where libm's hardware-sqrt-plus-kernel left nothing to win.
///
/// Accuracy: measured max relative deviation from libm `cbrt` is 6.4544e-15
/// over the operational operand range (positive, ~1e-2..1e2); the bit-hack
/// start is within ~4% for every positive normal double and two
/// cubically-convergent Halley steps land within 6.2405e-15 of libm across the
/// full positive normal range as well
/// (`seed_cbrt_tracks_libm_cbrt_across_the_operand_range` pins both ranges).
/// The two figures were transposed here until R21. Read them as what they
/// are: sampled maxima, not proved bounds, and sampled at very different
/// densities — the operational grid is 40,001 points over four decades
/// (~10,000 per decade, and a 100x denser sweep moves it only to 6.4549e-15,
/// so it is converged) while the full-range grid is 60,001 points over six
/// hundred decades (~100 per decade). The full-range figure being the smaller
/// of the two is a sampling artifact; it is not evidence that the tails are
/// better behaved than the operational range.
/// The consumer is an initial GUESS whose own asymptotic-approximation error
/// is ~1e-2, so seed quality — and with it iteration counts — is unchanged;
/// what moves is the converged root's last bits (seed sensitivity is ~2e-15
/// relative, per the maxiter-floor documentation), which is a dv-criterion
/// change, accepted and re-pinned like the R18 pack routing.
///
/// Side benefit: this removes libm from the seed path entirely — the value is
/// pure Rust arithmetic, identical on every host, where libm `cbrt` was one of
/// the per-libm axes a cross-host bit comparison had to condition on.
///
/// Non-positive and non-finite operands (never produced by `initial_guess`'s
/// callers, but cheap to keep total) fall back to libm.
#[inline]
fn seed_cbrt(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return x.cbrt();
    }
    let y = f64::from_bits((x.to_bits() / 3).wrapping_add(0x2A9F_7893_782D_A1CE));
    // The Halley update is applied as y * ratio with the ratio computed
    // first: `y * (y^3 + 2x)` on its own is ~x^(4/3) and under/overflows for
    // |exponent| beyond ~225, while the ratio's numerator and denominator are
    // both ~3x and always representable alongside x.
    let y = y * ((y * y * y + 2.0 * x) / (2.0 * y * y * y + x));
    y * ((y * y * y + 2.0 * x) / (2.0 * y * y * y + x))
}

/// Perf #4: x^(2/3) via cbrt. `cbrt(x) * cbrt(x)` is 3-5x cheaper than the
/// general `powf` on both arm64 and x86. Arguments here (time / ratio
/// quantities) are always >= 0, so this equals `x.powf(2.0 / 3.0)` to within
/// ~1 ULP. The different transcendental rounding shifts results slightly, so
/// the perf-gate baseline must be re-recorded (pareto front confirmed unchanged).
/// R18: the cube root itself is [`seed_cbrt`]; see its pricing note.
#[inline]
fn cbrt_squared(x: f64) -> f64 {
    let c = seed_cbrt(x);
    c * c
}

#[inline]
#[expect(
    clippy::imprecise_flops,
    reason = "Lambert result identity requires exp then subtraction in the established order"
)]
fn initial_guess(t: f64, ll: f64, m: i32, low_path: bool) -> f64 {
    if m == 0 {
        // Near-parabolic case: ll close to ±1 causes numerical issues
        // Use asymptotic expansion for better stability
        let ll_sq = ll * ll;
        if ll_sq > 0.9999 {
            // Very near-parabolic: use simple approximation
            // x approaches 0 as ll approaches ±1
            return 0.0;
        }

        let t0 = ll.acos() + ll * (1.0 - ll_sq).sqrt();
        let t1 = 2.0 * (1.0 - ll * ll_sq) / 3.0;

        let x0 = if t >= t0 {
            // High TOF regime: tighter estimate
            cbrt_squared(t0 / t) - 1.0
        } else if t < t1 {
            // Low TOF regime
            let denom = 1.0 - ll.powi(5);
            if denom.abs() < 1e-10 {
                // Avoid division by near-zero
                1.0
            } else {
                2.5 * t1 / t * (t1 - t) / denom + 1.0
            }
        } else {
            // Intermediate regime
            let ln_ratio = (t / t0).ln() / (t1 / t0).ln();
            (2.0_f64.ln() * ln_ratio).exp() - 1.0
        };

        // Clamp to valid range: x should be in (-1, sqrt(2)) for m=0
        x0.clamp(-0.9999, 1.4)
    } else {
        // Multi-revolution case
        let m_f = f64::from(m);
        let pi = std::f64::consts::PI;

        // Compute x0l and x0r with better numerical stability
        let ratio_l = cbrt_squared((m_f + 1.0) * pi / (8.0 * t));
        let ratio_r = cbrt_squared((8.0 * t) / (m_f * pi));

        let x0l = (ratio_l - 1.0) / (ratio_l + 1.0);
        let x0r = (ratio_r - 1.0) / (ratio_r + 1.0);

        let x0 = if low_path { x0l.max(x0r) } else { x0l.min(x0r) };

        // Clamp to valid range for multi-rev
        x0.clamp(-0.9999, 0.9999)
    }
}

/// Compute adaptive maxiter based on geometry characteristics.
/// Easy cases (circular, short TOF, m=0) converge in 2-3 iterations.
/// Hard cases (multi-rev, near-parabolic) may need more.
#[inline]
pub fn adaptive_maxiter(ll: f64, t_nd: f64, m: i32) -> i32 {
    if ll.abs() < 0.3 && m == 0 && t_nd < 5.0 {
        return 4; // Easy: circular, short TOF
    }
    if m > 0 || ll.abs() > 0.9 {
        return 8; // Hard: multi-rev or near-parabolic
    }
    6 // Default
}

/// The Lambert branch feasibility guard, extracted verbatim from its five
/// former hand-copies (`find_xy`, `find_xy_seeded`, the SIMD per-lane
/// pre-pass, and both batch lane-prefill loops). One definition so a future
/// edit to the prune cannot silently fork between the scalar and SIMD paths.
///
/// Sequence is unchanged: near-parabolic `|ll| >= 1` reject, cheap
/// `m_max_quick` bound, then the expensive `t_min` check only at the
/// boundary revolution `m == m_max_quick`.
#[expect(
    clippy::inline_always,
    reason = "reproduces the codegen of the five formerly inlined hand-copies of this guard"
)]
#[inline(always)]
fn lambert_branch_feasible(ll: f64, t: f64, m: i32, maxiter: i32, atol: f64, rtol: f64) -> bool {
    // Early reject: degenerate geometry (near-parabolic)
    if ll.abs() >= 1.0 {
        return false;
    }

    // Fast m_max upper bound (cheap computation)
    let m_max_quick = f64_to_i32_saturating((t / std::f64::consts::PI).floor());

    // Fast reject: m exceeds theoretical maximum
    if m > m_max_quick {
        return false;
    }

    // Only check t_min constraint when m is at the boundary (m == m_max_quick)
    // For m < m_max_quick, we skip the expensive t_min computation
    if m > 0 && m == m_max_quick {
        let t00 = ll.acos() + ll * (1.0 - ll * ll).sqrt();
        if t < t00 + f64::from(m) * std::f64::consts::PI {
            let (_, t_min) = compute_t_min(ll, m, maxiter, atol, rtol);
            if t < t_min {
                return false;
            }
        }
    }

    true
}

#[inline]
fn find_xy(
    ll: f64,
    t: f64,
    m: i32,
    maxiter: i32,
    atol: f64,
    rtol: f64,
    low_path: bool,
) -> (f64, f64) {
    if !lambert_branch_feasible(ll, t, m, maxiter, atol, rtol) {
        return (f64::NAN, f64::NAN);
    }

    // Proceed with root-finding
    let x0 = initial_guess(t, ll, m, low_path);
    let effective_maxiter = if maxiter == 8 {
        adaptive_maxiter(ll, t, m)
    } else {
        maxiter
    };
    let x = householder_method(x0, t, ll, m, atol, rtol, effective_maxiter);
    let y = compute_y(x, ll);
    (x, y)
}

/// Find (x, y) for Lambert problem with optional seeding from previous solution.
///
/// When `x_seed` is provided, it's used as the initial guess instead of computing
/// one from geometry. This is useful for batch TOF solving where adjacent TOFs
/// have similar solutions, drastically reducing iterations.
///
/// # Arguments
/// * `ll` - Transfer geometry parameter
/// * `t` - Non-dimensional time of flight
/// * `m` - Number of complete revolutions
/// * `maxiter` - Maximum iterations for Householder method
/// * `atol`, `rtol` - Convergence tolerances
/// * `low_path` - If true, select Izzo's low-path (geometric) multi-rev branch — the larger-x root, matching the poliastro convention. NOTE: for M >= 1 this is typically the HIGHER delta-v branch; it is a geometric label, not an energy ordering. Enumerate both branches when energy matters for multi-rev (only used when no seed)
/// * `x_seed` - Optional seed from previous solution
///
/// # Returns
/// (x, y) solution pair, or (NaN, NaN) if infeasible
#[inline]
fn find_xy_seeded(
    ll: f64,
    t: f64,
    m: i32,
    maxiter: i32,
    atol: f64,
    rtol: f64,
    low_path: bool,
    x_seed: Option<f64>,
) -> (f64, f64) {
    if !lambert_branch_feasible(ll, t, m, maxiter, atol, rtol) {
        return (f64::NAN, f64::NAN);
    }

    // Use seed if provided, otherwise compute initial guess
    let x0 = x_seed.unwrap_or_else(|| initial_guess(t, ll, m, low_path));

    let effective_maxiter = if maxiter == 8 {
        adaptive_maxiter(ll, t, m)
    } else {
        maxiter
    };
    // Pass 13.1: floor the effective iteration count under the deterministic
    // feature so cold and warm starts converge to the same fixed point. With
    // the historical floor of 8 (the sentinel) the Householder solve could
    // bottom out at maxiter without |delta| < tol, returning a partial-x
    // that depended on `x_seed`. See LAMBERT_DETERMINISTIC_MAXITER_FLOOR
    // and `lambert_seed_independence_*` tests for the invariant.
    let effective_maxiter = deterministic_maxiter_floor(effective_maxiter);
    let x = householder_method(x0, t, ll, m, atol, rtol, effective_maxiter);
    let y = compute_y(x, ll);
    (x, y)
}

/// HF-NEW-01: SIMD4 batched `find_xy` across per-lane `(ll, m, low_path)`.
///
/// Solves four Lambert problems in parallel that share the same non-dimensional
/// TOF `t` but may differ in `ll`, `m`, and `low_path`. Used by
/// `for_each_lambert_m_prograde_lowpaths` to pack the (m, `low_path`) and
/// optionally the prograde axes into a single SIMD call. Lanes whose scalar
/// pre-pass rejects (degenerate geometry, m above `m_max_quick`, t below
/// `t_min`) return `(NaN, NaN)` directly; remaining lanes enter
/// `householder_simd4_m_variant` together.
///
/// Returns `(x_arr, y_arr)`, one (x, y) pair per lane. NaN sentinels mark
/// failed / inactive lanes; callers must check `is_finite()` before using
/// the value.
#[must_use]
pub fn find_xy_simd4_m_variant(
    ll_arr: [f64; 4],
    t: f64,
    m_arr: [i32; 4],
    low_path_arr: [bool; 4],
    maxiter: i32,
    atol: f64,
    rtol: f64,
) -> ([f64; 4], [f64; 4]) {
    // `f64x4::splat(t)` and `f64x4::new([t; 4])` build the same lanes, and the
    // per-lane pre-pass below reads the same `t` for every lane either way, so
    // this delegation is bit-identical to the pre-refactor body.
    find_xy_simd4_m_variant_per_lane_t([t; 4], ll_arr, m_arr, low_path_arr, maxiter, atol, rtol)
}

/// [`find_xy_simd4_m_variant`] with a per-lane non-dimensional TOF.
///
/// This is the kernel entry for the cross-TOF batch axis: lanes may come from
/// DIFFERENT Lambert problems (different `t_nd` and `ll`), not just different
/// `(m, low_path, prograde)` variants of one problem. Every lane's iterate
/// sequence in `householder_simd4_m_variant` depends only on that lane's own
/// `(p0, t, ll, m, low_path, maxiter)` — there is no cross-lane arithmetic and
/// retired lanes are mask-frozen — so a lane's returned `(x, y)` is
/// bit-identical however the lanes are packed. That lane independence is what
/// makes cross-TOF repacking a pure restructuring rather than a root-moving
/// change.
#[must_use]
pub fn find_xy_simd4_m_variant_per_lane_t(
    t_arr: [f64; 4],
    ll_arr: [f64; 4],
    m_arr: [i32; 4],
    low_path_arr: [bool; 4],
    maxiter: i32,
    atol: f64,
    rtol: f64,
) -> ([f64; 4], [f64; 4]) {
    // An inline `[f64::NAN; 4]` init lowered to a libc `memset_pattern16` call
    // per pack -- measured at 45% of this function's samples on aarch64. The
    // `NAN4` const item is a rodata copy instead (no libcall), so the outputs
    // can start at the NaN sentinel directly; lanes that converge overwrite it.
    let mut x_out = NAN4;
    let mut y_out = NAN4;
    let mut active = [true; 4];
    let mut p0_arr = [0.0_f64; 4];
    let mut maxiter_arr = [0_i32; 4];

    // Per-lane scalar pre-pass mirrors the scalar `find_xy` early-reject and
    // initial-guess logic. Inactive lanes are masked out before SIMD; their
    // outputs default to NaN.
    for ((((((ll, t), m), low_path), active_lane), p0_lane), maxiter_lane) in ll_arr
        .iter()
        .copied()
        .zip(t_arr.iter().copied())
        .zip(m_arr.iter().copied())
        .zip(low_path_arr.iter().copied())
        .zip(active.iter_mut())
        .zip(p0_arr.iter_mut())
        .zip(maxiter_arr.iter_mut())
    {
        if !lambert_branch_feasible(ll, t, m, maxiter, atol, rtol) {
            *active_lane = false;
            continue;
        }

        *p0_lane = initial_guess(t, ll, m, low_path);
        *maxiter_lane = if maxiter == 8 {
            adaptive_maxiter(ll, t, m)
        } else {
            maxiter
        };
    }

    if !active.iter().any(|&a| a) {
        return (NAN4, NAN4);
    }

    // Pack SIMD inputs. Inactive lanes get safe placeholder values (ll=0.5,
    // p0=0.5, maxiter=0) so the SIMD math doesn't produce NaN that poisons
    // active lanes; their outputs are overwritten with NaN sentinels after
    // the SIMD call.
    let safe_ll = |lane: usize| {
        active
            .get(lane)
            .copied()
            .zip(ll_arr.get(lane).copied())
            .map_or(0.5, |(is_active, ll)| if is_active { ll } else { 0.5 })
    };
    let safe_p0 = |lane: usize| {
        active
            .get(lane)
            .copied()
            .zip(p0_arr.get(lane).copied())
            .map_or(0.5, |(is_active, p0)| if is_active { p0 } else { 0.5 })
    };
    let safe_maxiter = |lane: usize| {
        active
            .get(lane)
            .copied()
            .zip(maxiter_arr.get(lane).copied())
            .map_or(
                0,
                |(is_active, maxiter)| {
                    if is_active {
                        maxiter
                    } else {
                        0
                    }
                },
            )
    };

    let lambda_vec = f64x4::new([safe_ll(0), safe_ll(1), safe_ll(2), safe_ll(3)]);
    let lambda_sq_vec = lambda_vec * lambda_vec;
    let lambda_cu_vec = lambda_sq_vec * lambda_vec;
    let lambda_fifth_vec = lambda_cu_vec * lambda_sq_vec;
    let t0_vec = f64x4::new(t_arr);
    let p0_vec = f64x4::new([safe_p0(0), safe_p0(1), safe_p0(2), safe_p0(3)]);
    let m_pi_vec = f64x4::new(m_arr.map(|m| f64::from(m) * std::f64::consts::PI));
    // The scalar `tof_equation_y` takes its `hyp2f1b` branch only at m == 0, so
    // the kernel needs the revolution count itself, not just its phase.
    let is_m0 = f64x4::new(m_arr.map(f64::from)).simd_eq(simd::V_ZERO);
    let lane_maxiters = [
        safe_maxiter(0),
        safe_maxiter(1),
        safe_maxiter(2),
        safe_maxiter(3),
    ];

    let x_vec = simd::householder_simd4_m_variant(
        p0_vec,
        t0_vec,
        lambda_vec,
        lambda_sq_vec,
        lambda_cu_vec,
        lambda_fifth_vec,
        m_pi_vec,
        is_m0,
        lane_maxiters,
        atol,
        rtol,
    );
    let x_arr_simd = x_vec.to_array();

    for ((((is_active, x), ll), x_out_lane), y_out_lane) in active
        .iter()
        .copied()
        .zip(x_arr_simd)
        .zip(ll_arr)
        .zip(x_out.iter_mut())
        .zip(y_out.iter_mut())
    {
        if !is_active {
            continue;
        }
        if !x.is_finite() {
            continue;
        }
        let y = compute_y(x, ll);
        *x_out_lane = x;
        *y_out_lane = y;
    }

    (x_out, y_out)
}

#[inline]
fn reconstruct_velocities(
    x: f64,
    y: f64,
    r1_norm: f64,
    r2_norm: f64,
    ll: f64,
    gamma: f64,
    rho: f64,
    sigma: f64,
) -> (f64, f64, f64, f64) {
    // FMA optimizations for velocity reconstruction
    // ll*y - x = ll.mul_add(y, -x)
    // ll*y + x = ll.mul_add(y, x)
    // y + ll*x = ll.mul_add(x, y)
    let lly_minus_x = ll.mul_add(y, -x);
    let lly_plus_x = ll.mul_add(y, x);
    let y_plus_llx = ll.mul_add(x, y);

    // vr1 = gamma * (lly_minus_x - rho * lly_plus_x) / r1_norm
    // = gamma * ((-rho).mul_add(lly_plus_x, lly_minus_x)) / r1_norm
    let vr1 = gamma * (-rho).mul_add(lly_plus_x, lly_minus_x) / r1_norm;
    // vr2 = -gamma * (lly_minus_x + rho * lly_plus_x) / r2_norm
    // = -gamma * (rho.mul_add(lly_plus_x, lly_minus_x)) / r2_norm
    let vr2 = -gamma * rho.mul_add(lly_plus_x, lly_minus_x) / r2_norm;

    let vt1 = gamma * sigma * y_plus_llx / r1_norm;
    let vt2 = gamma * sigma * y_plus_llx / r2_norm;
    (vr1, vr2, vt1, vt2)
}

// =============================================================================
// SIMD-Optimized Implementation
// =============================================================================
//
// Key optimizations:
// 1. Eliminates redundant normalization (ir1 == r1_unit)
// 2. Uses FMA instructions for velocity reconstruction
// 3. Uses cache-friendly data layout

mod simd_lambert {
    use super::{cross3, norm3, vec3_to_array, LambertGeometry, LambertResult};

    /// Optimized velocity reconstruction using FMA.
    /// Computes v1 = `r_unit` * vr + it * vt for both departure and arrival.
    #[inline]
    pub fn reconstruct_velocities_optimized(
        ir1: &[f64; 3],
        ir2: &[f64; 3],
        it1: &[f64; 3],
        it2: &[f64; 3],
        vr1: f64,
        vr2: f64,
        vt1: f64,
        vt2: f64,
    ) -> ([f64; 3], [f64; 3]) {
        // For 3D vectors, explicit scalar FMA is often as fast as SIMD with 1 unused lane.
        // The compiler should auto-vectorize this pattern with -O3 -march=native.
        // Using mul_add (FMA) for precision and potential speedup.
        let v1 = [
            ir1[0].mul_add(vr1, it1[0] * vt1),
            ir1[1].mul_add(vr1, it1[1] * vt1),
            ir1[2].mul_add(vr1, it1[2] * vt1),
        ];
        let v2 = [
            ir2[0].mul_add(vr2, it2[0] * vt2),
            ir2[1].mul_add(vr2, it2[1] * vt2),
            ir2[2].mul_add(vr2, it2[2] * vt2),
        ];
        (v1, v2)
    }

    /// Optimized Lambert solver implementation.
    /// Key optimizations over scalar version:
    /// 1. Reuses normalized vectors (ir1/ir2) instead of recomputing `r1_unit/r2_unit`
    /// 2. Uses FMA instructions for velocity reconstruction
    /// 3. Inlines hot path functions
    #[inline]
    pub fn izzo2015_impl_simd(
        mu: f64,
        r1: &[f64; 3],
        r2: &[f64; 3],
        tof: f64,
        revolutions: i32,
        prograde: bool,
        low_path: bool,
        maxiter: i32,
        atol: f64,
        rtol: f64,
    ) -> LambertResult {
        let mut result = LambertResult {
            v1: [0.0; 3],
            v2: [0.0; 3],
            success: false,
        };

        if !mu.is_finite() || mu <= 0.0 || !tof.is_finite() || tof <= 0.0 {
            return result;
        }

        let r1_norm = norm3(r1);
        let r2_norm = norm3(r2);
        if r1_norm <= 0.0 || r2_norm <= 0.0 {
            return result;
        }

        let chord = [r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]];
        let c_norm = norm3(&chord);
        if c_norm <= 0.0 {
            return result;
        }

        let semiperimeter = (r1_norm + r2_norm + c_norm) * 0.5;
        if !semiperimeter.is_finite() || semiperimeter <= 0.0 {
            return result;
        }

        // Compute normalized vectors ONCE (reused for velocity reconstruction)
        let inv_r1 = 1.0 / r1_norm;
        let inv_r2 = 1.0 / r2_norm;
        let ir1 = [r1[0] * inv_r1, r1[1] * inv_r1, r1[2] * inv_r1];
        let ir2 = [r2[0] * inv_r2, r2[1] * inv_r2, r2[2] * inv_r2];

        let mut ih = cross3(&ir1, &ir2);
        let ih_norm = norm3(&ih);
        if ih_norm <= 0.0 {
            return result;
        }
        let inv_ih = 1.0 / ih_norm;
        ih = [ih[0] * inv_ih, ih[1] * inv_ih, ih[2] * inv_ih];

        let mut lambda = (1.0 - (c_norm / semiperimeter).clamp(0.0, 1.0)).sqrt();
        let (mut it1, mut it2) = if ih[2] < 0.0 {
            lambda = -lambda;
            (cross3(&ir1, &ih), cross3(&ir2, &ih))
        } else {
            (cross3(&ih, &ir1), cross3(&ih, &ir2))
        };

        if !prograde {
            lambda = -lambda;
            it1 = [-it1[0], -it1[1], -it1[2]];
            it2 = [-it2[0], -it2[1], -it2[2]];
        }

        let s_cubed = semiperimeter * semiperimeter * semiperimeter;
        if s_cubed <= 0.0 {
            return result;
        }

        let t_nd = (2.0 * mu / s_cubed).sqrt() * tof;
        let (solution_x, solution_y) =
            super::find_xy(lambda, t_nd, revolutions, maxiter, atol, rtol, low_path);
        if !solution_x.is_finite() || !solution_y.is_finite() {
            return result;
        }

        let gamma = (mu * semiperimeter / 2.0).sqrt();
        let rho = (r1_norm - r2_norm) / c_norm;
        let sigma = (1.0 - rho * rho).max(0.0).sqrt();

        let (vr1, vr2, vt1, vt2) = super::reconstruct_velocities(
            solution_x, solution_y, r1_norm, r2_norm, lambda, gamma, rho, sigma,
        );

        // Use optimized velocity reconstruction with FMA
        let (v1, v2) = reconstruct_velocities_optimized(&ir1, &ir2, &it1, &it2, vr1, vr2, vt1, vt2);
        result.v1 = v1;
        result.v2 = v2;

        result.success = true;
        result
    }

    /// Optimized Lambert solver using precomputed geometry.
    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "Lambert SIMD reconstruction requires the established floating-point evaluation order"
    )]
    pub fn izzo2015_impl_with_geom_simd(
        geom: &LambertGeometry,
        m: i32,
        prograde: bool,
        low_path: bool,
        maxiter: i32,
        atol: f64,
        rtol: f64,
    ) -> LambertResult {
        let mut result = LambertResult {
            v1: [0.0; 3],
            v2: [0.0; 3],
            success: false,
        };

        if !geom.success {
            return result;
        }

        let mut ll = geom.ll_base;
        let mut it1 = geom.it1_base;
        let mut it2 = geom.it2_base;

        if !prograde {
            ll = -ll;
            it1 = -it1;
            it2 = -it2;
        }

        let (x, y) = super::find_xy(ll, geom.t_nd, m, maxiter, atol, rtol, low_path);
        if !x.is_finite() || !y.is_finite() {
            return result;
        }

        let (vr1, vr2, vt1, vt2) = super::reconstruct_velocities(
            x,
            y,
            geom.r1_norm,
            geom.r2_norm,
            ll,
            geom.gamma,
            geom.rho,
            geom.sigma,
        );

        // Use optimized velocity reconstruction with FMA
        let radial_departure = vec3_to_array(&geom.ir1);
        let radial_arrival = vec3_to_array(&geom.ir2);
        let tangent_departure = vec3_to_array(&it1);
        let tangent_arrival = vec3_to_array(&it2);
        let (v1, v2) = reconstruct_velocities_optimized(
            &radial_departure,
            &radial_arrival,
            &tangent_departure,
            &tangent_arrival,
            vr1,
            vr2,
            vt1,
            vt2,
        );
        result.v1 = v1;
        result.v2 = v2;

        result.success = true;
        result
    }
}

/// Lambert solver using precomputed geometry.
///
/// The UNFUSED half of a measured drifted-twin pair. Production flies
/// `izzo2015_impl_with_geom_fast`, whose velocity reconstruction fuses the same
/// arithmetic differently; `fused_and_unfused_reconstruction_stay_within_their_measured_gap`
/// pins the gap between the two conventions, which disagree on 70.0% of 8,653
/// converged solves (`REFACTOR_BLOCKLIST.md`, "One of the four kept items").
/// `izzo_geometry_oracle_tests` separately differences it against `izzo2015_impl`,
/// which derives its geometry inline. Deleting it deletes both comparisons.
#[inline]
#[must_use]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "unfused twin differenced against the flown izzo2015_impl_with_geom_fast \
                  reconstruction, and against izzo2015_impl's inline geometry"
    )
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Lambert state reconstruction requires the established floating-point evaluation order"
)]
pub fn izzo2015_impl_with_geom(
    geom: &LambertGeometry,
    m: i32,
    prograde: bool,
    low_path: bool,
    maxiter: i32,
    atol: f64,
    rtol: f64,
) -> LambertResult {
    let mut result = LambertResult {
        v1: [0.0; 3],
        v2: [0.0; 3],
        success: false,
    };

    if !geom.success {
        return result;
    }

    let mut ll = geom.ll_base;
    let mut it1 = geom.it1_base;
    let mut it2 = geom.it2_base;

    if !prograde {
        ll = -ll;
        it1 = -it1;
        it2 = -it2;
    }

    let (x, y) = find_xy(ll, geom.t_nd, m, maxiter, atol, rtol, low_path);
    if !x.is_finite() || !y.is_finite() {
        return result;
    }

    let (vr1, vr2, vt1, vt2) = reconstruct_velocities(
        x,
        y,
        geom.r1_norm,
        geom.r2_norm,
        ll,
        geom.gamma,
        geom.rho,
        geom.sigma,
    );

    // NOT fused. `a * b + c` on nalgebra vectors is a multiply followed by an
    // add, two roundings; Rust never contracts it into an FMA, and this
    // workspace allows `suboptimal_flops` precisely because that choice is
    // load-bearing. The other convention in this crate,
    // `simd_lambert::reconstruct_velocities_optimized`, uses an explicit
    // `mul_add` and rounds once. The two disagree in the last bits -- measured
    // 2026-08-21 over 8,653 converged solves, 6,055 (70.0%) differ, at most
    // 2.35e-13 relative. Do not "restore" a fusion here believing it a
    // regression fix: it is a deliberate difference under review, pinned by
    // `fused_and_unfused_reconstruction_stay_within_their_measured_gap`.
    let v1 = geom.ir1 * vr1 + it1 * vt1;
    let v2 = geom.ir2 * vr2 + it2 * vt2;
    result.v1 = vec3_to_array(&v1);
    result.v2 = vec3_to_array(&v2);

    result.success = true;
    result
}

#[inline]
#[must_use]
pub fn izzo2015_impl_with_geom_fast(
    geom: &LambertGeometry,
    m: i32,
    prograde: bool,
    low_path: bool,
    maxiter: i32,
    atol: f64,
    rtol: f64,
) -> LambertResult {
    {
        simd_lambert::izzo2015_impl_with_geom_simd(geom, m, prograde, low_path, maxiter, atol, rtol)
    }
}

/// Lambert solver with seeding support, returns (result, `solution_x`).
///
/// This variant accepts an optional seed value from a previous solution and
/// returns the solution `x` value for use as a seed in subsequent solves.
/// Used for batch TOF solving where adjacent TOFs have similar solutions.
///
/// # Arguments
/// * `geom` - Precomputed geometry (must have correct `t_nd` for this TOF)
/// * `m` - Number of complete revolutions
/// * `prograde` - True for prograde transfer
/// * `low_path` - True selects Izzo's geometric low-path multi-rev branch (larger-x root; typically higher delta-v)
/// * `maxiter` - Maximum Householder iterations
/// * `atol`, `rtol` - Convergence tolerances
/// * `x_seed` - Optional seed from previous solution
///
/// # Returns
/// (`LambertResult`, `x_solution`) where `x_solution` can seed the next solve
#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Lambert state reconstruction requires the established floating-point evaluation order"
)]
fn izzo2015_impl_with_geom_seeded(
    geom: &LambertGeometry,
    m: i32,
    prograde: bool,
    low_path: bool,
    maxiter: i32,
    atol: f64,
    rtol: f64,
    x_seed: Option<f64>,
) -> (LambertResult, f64) {
    let mut result = LambertResult {
        v1: [0.0; 3],
        v2: [0.0; 3],
        success: false,
    };

    if !geom.success {
        return (result, f64::NAN);
    }

    let mut ll = geom.ll_base;
    let mut it1 = geom.it1_base;
    let mut it2 = geom.it2_base;

    if !prograde {
        ll = -ll;
        it1 = -it1;
        it2 = -it2;
    }

    let (x, y) = find_xy_seeded(ll, geom.t_nd, m, maxiter, atol, rtol, low_path, x_seed);
    if !x.is_finite() || !y.is_finite() {
        return (result, f64::NAN);
    }

    let (vr1, vr2, vt1, vt2) = reconstruct_velocities(
        x,
        y,
        geom.r1_norm,
        geom.r2_norm,
        ll,
        geom.gamma,
        geom.rho,
        geom.sigma,
    );

    // NOT fused, and this is the convention the seeded batch routes ship. See
    // `izzo2015_impl_with_geom` above for the full note: `a * b + c` here is
    // two roundings, while `simd_lambert::reconstruct_velocities_optimized`
    // uses `mul_add` and rounds once. Both conventions are live on the Part A
    // path -- this one through `izzo2015_batch_tof_variable_r2` and
    // `..._with_scratch`, the fused one through `izzo2015_impl`.
    let v1 = geom.ir1 * vr1 + it1 * vt1;
    let v2 = geom.ir2 * vr2 + it2 * vt2;
    result.v1 = vec3_to_array(&v1);
    result.v2 = vec3_to_array(&v2);

    result.success = true;
    (result, x)
}

// Dispatch to SIMD or scalar implementation based on feature flag
#[inline]
#[must_use]
pub fn izzo2015_impl(
    mu: f64,
    r1: &[f64; 3],
    r2: &[f64; 3],
    tof: f64,
    m: i32,
    prograde: bool,
    low_path: bool,
    maxiter: i32,
    atol: f64,
    rtol: f64,
) -> LambertResult {
    simd_lambert::izzo2015_impl_simd(mu, r1, r2, tof, m, prograde, low_path, maxiter, atol, rtol)
}

/// Reference definition of the arrival rendezvous dV convention
/// (target velocity minus payload arrival velocity).
///
/// Test-only: the shipping solvers inline this convention, and
/// `izzo2015_transfer_dv_arrival_matches_rendezvous_convention` exists to pin
/// the batch path against this independent statement of it.
#[cfg(test)]
fn arrival_rendezvous_dv(v_payload_arrival: &[f64; 3], target_state: &[f64]) -> [f64; 3] {
    let [arrival_x, arrival_y, arrival_z] = *v_payload_arrival;
    let [.., desired_x, desired_y, desired_z] = target_state else {
        return [f64::NAN; 3];
    };
    [
        desired_x - arrival_x,
        desired_y - arrival_y,
        desired_z - arrival_z,
    ]
}

/// Result from batch Lambert solver with (M, prograde) info
#[cfg_attr(
    not(feature = "bench-internal"),
    expect(
        dead_code,
        reason = "carrier type of izzo2015_batch_m_prograde, which only \
                  benches/lambert_solver_bench.rs calls outside tests; the fields it does not \
                  read are still part of that entry point's contract"
    )
)]
#[derive(Clone, Copy, Debug)]
pub struct BatchLambertResult {
    /// Departure velocity [3]
    pub v1: [f64; 3],
    /// Arrival velocity [3]
    pub v2: [f64; 3],
    /// Time of flight
    pub tof: f64,
    /// Number of revolutions
    pub m: i32,
    /// True if prograde
    pub prograde: bool,
    /// True if solution is valid
    pub valid: bool,
}

/// Result for a single TOF in batch TOF processing
#[derive(Clone, Copy, Debug)]
pub struct BatchTofResult {
    /// Echo of the requested TOF. Nothing in production reads it back — the row
    /// index already carries the association — but
    /// `test_batch_tof_geometry_computed_once` compares it bit-for-bit against
    /// the requested `tofs`, which is what proves the scratch path's internal
    /// sort restored the caller's row order.
    #[cfg_attr(
        not(any(test, feature = "bench-internal")),
        expect(
            dead_code,
            reason = "differenced against the requested tofs by the row-order pin"
        )
    )]
    pub tof: f64,
    pub dv_depart: f64,
    pub dv_arrive: f64,
    pub v1: [f64; 3],
    pub v2: [f64; 3],
    pub m: i32,
    pub prograde: bool,
    pub valid: bool,
}

/// Best branch-aware Lambert result for a single TOF in variable-r2 batch processing.
#[derive(Clone, Copy, Debug)]
pub struct BranchBatchTofResult {
    pub tof: f64,
    pub dv_depart: f64,
    /// Arrival dV magnitude. Branch selection is decided on `dv_depart` alone,
    /// so production never reads this back; the tests do, and they are the
    /// reason it is still computed. `variable_r2_branch_best_batch_matches_scalar_branch_enumerator`
    /// differences it against the scalar `for_each_lambert_m_prograde_lowpaths`
    /// enumerator, and `assert_branch_batch_row_bits` compares it bit-for-bit
    /// between the pruned and unpruned batch paths. Dropping the field would
    /// silently weaken both.
    #[cfg_attr(
        not(any(test, feature = "bench-internal")),
        expect(
            dead_code,
            reason = "read only by the scalar-enumerator differential and the \
                      pruned-vs-unpruned bit-equality pins"
        )
    )]
    pub dv_arrive: f64,
    pub v1: [f64; 3],
    pub v2: [f64; 3],
    pub m: i32,
    pub low_path: bool,
    pub prograde: bool,
    pub valid: bool,
}

#[derive(Default)]
pub struct LambertBatchScratch {
    indexed_tofs: Vec<(usize, f64)>,
    tof_results: Vec<BatchTofResult>,
    // Buffer of the test-only seeded dv twin (`izzo2015_batch_dv_seeded_with_scratch`).
    dv_results: Vec<f64>,
    last_x_seeds: Vec<Option<f64>>,
}

#[derive(Default)]
pub struct VariableR2LambertScratch {
    indexed_tofs: Vec<(usize, f64)>,
    results: Vec<BatchTofResult>,
    branch_results: Vec<BranchBatchTofResult>,
    last_x_seeds: Vec<Option<f64>>,
    branch_telemetry: VariableR2BranchTelemetry,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VariableR2BranchTelemetry {
    pub m0_simd_prefill_lanes: usize,
    pub m0_simd_valid_lanes: usize,
    pub m0_scalar_fallback_lanes: usize,
    pub simd_lane_solves: usize,
    pub scalar_variant_solves: usize,
}

impl VariableR2LambertScratch {
    #[must_use]
    pub const fn branch_telemetry(&self) -> VariableR2BranchTelemetry {
        self.branch_telemetry
    }
}

impl Default for BatchLambertResult {
    fn default() -> Self {
        Self {
            v1: [0.0; 3],
            v2: [0.0; 3],
            tof: 0.0,
            m: 0,
            prograde: true,
            valid: false,
        }
    }
}

impl Default for BatchTofResult {
    fn default() -> Self {
        Self {
            tof: 0.0,
            dv_depart: f64::INFINITY,
            dv_arrive: f64::INFINITY,
            v1: [0.0; 3],
            v2: [0.0; 3],
            m: 0,
            prograde: true,
            valid: false,
        }
    }
}

impl Default for BranchBatchTofResult {
    fn default() -> Self {
        Self {
            tof: 0.0,
            dv_depart: f64::INFINITY,
            dv_arrive: f64::INFINITY,
            v1: [0.0; 3],
            v2: [0.0; 3],
            m: 0,
            low_path: true,
            prograde: true,
            valid: false,
        }
    }
}

/// Batch solve Lambert problem for all (M, prograde) combinations.
///
/// This function evaluates Lambert solutions for M = 0, 1, ..., `m_max`
/// and both prograde and retrograde directions, returning all valid solutions.
///
/// # Arguments
/// * `mu` - Gravitational parameter (km^3/s^2)
/// * `r1` - Position vector at departure (km)
/// * `r2` - Position vector at arrival (km)
/// * `tof` - Time of flight (seconds)
/// * `m_max` - Maximum number of complete revolutions
/// * `low_path` - If true, select Izzo's geometric low-path multi-rev branch (larger-x root; typically higher delta-v)
///
/// # Returns
/// Vector of valid solutions with (M, prograde) info
#[cfg_attr(
    not(any(test, feature = "bench-internal")),
    expect(
        dead_code,
        reason = "benches/lambert_solver_bench.rs is the only non-test caller and reaches it \
                  through the `bench-internal` re-export in lib.rs; production enumerates \
                  branches through for_each_lambert_m_prograde_lowpaths_pruned_with_r1 instead"
    )
)]
#[must_use]
pub fn izzo2015_batch_m_prograde(
    mu: f64,
    r1: &[f64; 3],
    r2: &[f64; 3],
    tof: f64,
    m_max: i32,
    low_path: bool,
) -> Vec<BatchLambertResult> {
    // Householder 4th order typically converges in 2-4 iterations. Re-measured
    // 2026-08-07 on the production `mf-p64-e24` batch: mean 3.028, max 8, and
    // 10,378 of 929,359,866 solves (0.0011%) exhaust their cap. The 2.81/max-4
    // figures this comment used to carry predate the deterministic maxiter
    // floor and did not hold at the production shape. 6 stays as the cap here.
    const MAXITER: i32 = 6;
    const ATOL: f64 = CONVERGENCE_TOL;
    const RTOL: f64 = CONVERGENCE_TOL;

    // Pre-filter: compute maximum feasible m value based on geometry
    // This avoids calling izzo2015_impl for m values that will certainly fail
    let m_max_feasible = compute_m_max_fast(r1, r2, tof, mu).min(m_max);

    // Parallel path for larger m_max (only if not already in a Rayon context)
    {
        let is_nested = rayon::current_thread_index().is_some();
        // `current_num_threads() > 1`: under a single-worker global pool the
        // worker serializes concurrent callers on its LockLatch; the sequential
        // path below runs the identical `izzo2015_impl` in the same order, so the
        // returned Vec is byte-identical.
        if !is_nested
            && m_max_feasible >= LAMBERT_PARALLEL_THRESHOLD
            && rayon::current_num_threads() > 1
        {
            return feasible_revolution_range(m_max_feasible)
                .into_par_iter()
                .flat_map_iter(|m| [(m, true), (m, false)])
                .filter_map(|(m, prograde)| {
                    let res =
                        izzo2015_impl(mu, r1, r2, tof, m, prograde, low_path, MAXITER, ATOL, RTOL);
                    if res.success {
                        Some(BatchLambertResult {
                            v1: res.v1,
                            v2: res.v2,
                            tof,
                            m,
                            prograde,
                            valid: true,
                        })
                    } else {
                        None
                    }
                })
                .collect();
        }
    }

    // Sequential path for small m_max
    let mut results = Vec::with_capacity(revolution_pair_count(m_max_feasible));

    for m in feasible_revolution_range(m_max_feasible) {
        for prograde in [true, false] {
            let res = izzo2015_impl(mu, r1, r2, tof, m, prograde, low_path, MAXITER, ATOL, RTOL);
            if res.success {
                results.push(BatchLambertResult {
                    v1: res.v1,
                    v2: res.v2,
                    tof,
                    m,
                    prograde,
                    valid: true,
                });
            }
        }
    }

    results
}

/// Batch solve Lambert problem and return delta-V vectors.
///
/// Returns tuple of (`dv_depart`, `dv_arrive`) for each valid (M, prograde) combination.
///
/// # Arguments
/// * `state1` - Full state at departure [x, y, z, vx, vy, vz] (km, km/s)
/// * `state2` - Full state at arrival [x, y, z, vx, vy, vz] (km, km/s)
/// * `tof` - Time of flight (seconds)
/// * `m_max` - Maximum number of complete revolutions
/// * `low_path` - If true, select Izzo's geometric low-path multi-rev branch (larger-x root, poliastro convention; typically the HIGHER delta-v branch)
///
/// # Returns
/// Vector of (m, prograde, `dv_depart`, `dv_arrive`, valid) tuples
///
/// Test-only, and kept as a differential oracle rather than as a spare solver.
/// Three tests difference it against paths that DO fly:
/// `izzo2015_transfer_dv_arrival_matches_rendezvous_convention` against
/// `arrival_rendezvous_dv(izzo2015_impl(..))`,
/// `lambert_backend::test_lambert_batch_arrival_dv_matches_single_shot_convention`
/// bit-for-bit against `lambert_single_shot`, and `test_best_solution_finds_minimum`
/// against `izzo2015_best_solution`'s claimed minimum. It is the independent
/// statement of the arrival-dV sign convention those three check production against.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "differential oracle for the arrival-dV convention: differenced against \
                  lambert_single_shot, izzo2015_impl and izzo2015_best_solution"
    )
)]
#[must_use]
pub fn izzo2015_batch_dv(
    mu: f64,
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    m_max: i32,
    low_path: bool,
) -> Vec<(i32, bool, [f64; 3], [f64; 3], bool)> {
    let r1 = [state1[0], state1[1], state1[2]];
    let r2 = [state2[0], state2[1], state2[2]];

    // Pre-filter: compute maximum feasible m value based on geometry
    let m_max_feasible = compute_m_max_fast(&r1, &r2, tof, mu).min(m_max);

    // Hoist invariant geometry
    let geom = compute_lambert_geometry(mu, &r1, &r2, tof);

    // Parallel path for larger m_max (only if not already in a Rayon context)
    {
        let is_nested = rayon::current_thread_index().is_some();
        // `current_num_threads() > 1`: under a single-worker global pool the
        // worker serializes concurrent callers on its LockLatch; the sequential
        // path below runs the identical `izzo2015_impl_with_geom_fast` (== the SIMD
        // impl) in the same order, so the returned Vec is byte-identical.
        if !is_nested
            && m_max_feasible >= LAMBERT_PARALLEL_THRESHOLD
            && rayon::current_num_threads() > 1
        {
            return feasible_revolution_range(m_max_feasible)
                .into_par_iter()
                .flat_map_iter(|m| [(m, true), (m, false)])
                .filter_map(|(m, prograde)| {
                    let res = simd_lambert::izzo2015_impl_with_geom_simd(
                        &geom,
                        m,
                        prograde,
                        low_path,
                        8,
                        CONVERGENCE_TOL,
                        CONVERGENCE_TOL,
                    );
                    if !res.success {
                        return None;
                    }
                    let dv_depart = [
                        res.v1[0] - state1[3],
                        res.v1[1] - state1[4],
                        res.v1[2] - state1[5],
                    ];
                    let dv_arrive = [
                        state2[3] - res.v2[0],
                        state2[4] - res.v2[1],
                        state2[5] - res.v2[2],
                    ];
                    Some((m, prograde, dv_depart, dv_arrive, true))
                })
                .collect();
        }
    }

    // Sequential path
    let mut results = Vec::with_capacity(revolution_pair_count(m_max_feasible));

    for m in feasible_revolution_range(m_max_feasible) {
        for prograde in [true, false] {
            let res = izzo2015_impl_with_geom_fast(
                &geom,
                m,
                prograde,
                low_path,
                8,
                CONVERGENCE_TOL,
                CONVERGENCE_TOL,
            );
            if !res.success {
                continue;
            }
            let dv_depart = [
                res.v1[0] - state1[3],
                res.v1[1] - state1[4],
                res.v1[2] - state1[5],
            ];
            let dv_arrive = [
                state2[3] - res.v2[0],
                state2[4] - res.v2[1],
                state2[5] - res.v2[2],
            ];
            results.push((m, prograde, dv_depart, dv_arrive, true));
        }
    }

    results
}

/// Batch solve Lambert problem for multiple TOF values with shared geometry.
///
/// This is more efficient than calling `izzo2015_solve` repeatedly because:
/// 1. Geometry (r1, r2, norms, `ll_base`, etc.) is computed once
/// 2. Only time-dependent parameters are recomputed for each TOF
/// 3. **Seeded solving**: The solution from each TOF is used as the initial guess
///    for the next, drastically reducing iterations for continuous TOF scans
///
/// The seeding strategy works because adjacent TOFs have similar transfer
/// geometries, so the universal anomaly `x` changes smoothly. TOFs are sorted
/// before processing to maximize seeding effectiveness.
///
/// # Arguments
/// * `mu` - Gravitational parameter [km^3/s^2]
/// * `r1` - Initial position vector [km]
/// * `r2` - Final position vector [km]
/// * `tofs` - Array of time-of-flight values [s]
/// * `m_max` - Maximum number of revolutions to consider
/// * `v1_ref` - Optional reference velocity for departure (for delta-V calculation) [km/s]
/// * `v2_ref` - Optional reference velocity for arrival (for delta-V calculation) [km/s]
///
/// # Returns
/// Vector of results, one per TOF (in original input order), with minimum delta-V solution
#[cfg_attr(
    not(any(test, feature = "bench-internal")),
    expect(
        dead_code,
        reason = "the allocating shell over izzo2015_batch_tof_with_scratch; \
                  benches/lambert_{solver,batch_tof}_bench.rs are the only non-test callers \
                  and reach it through the `bench-internal` re-export in lib.rs"
    )
)]
#[must_use]
pub fn izzo2015_batch_tof(
    mu: f64,
    r1: &[f64; 3],
    r2: &[f64; 3],
    tofs: &[f64],
    m_max: i32,
    v1_ref: Option<&[f64; 3]>,
    v2_ref: Option<&[f64; 3]>,
) -> Vec<BatchTofResult> {
    let mut scratch = LambertBatchScratch::default();
    izzo2015_batch_tof_with_scratch(mu, r1, r2, tofs, m_max, v1_ref, v2_ref, &mut scratch).to_vec()
}

pub fn izzo2015_batch_tof_with_scratch<'a>(
    mu: f64,
    r1: &[f64; 3],
    r2: &[f64; 3],
    tofs: &[f64],
    m_max: i32,
    v1_ref: Option<&[f64; 3]>,
    v2_ref: Option<&[f64; 3]>,
    scratch: &'a mut LambertBatchScratch,
) -> &'a [BatchTofResult] {
    if tofs.is_empty() {
        scratch.tof_results.clear();
        return &scratch.tof_results;
    }

    // Compute TOF-independent geometry once with dummy TOF
    // We'll override t_nd for each actual TOF value
    let base_geom = compute_lambert_geometry(mu, r1, r2, 1.0);

    if !base_geom.success {
        // Invalid geometry - return invalid results for all TOFs
        scratch.tof_results.clear();
        scratch
            .tof_results
            .extend(tofs.iter().map(|&tof| BatchTofResult {
                tof,
                ..BatchTofResult::default()
            }));
        return &scratch.tof_results;
    }

    // Compute the time scaling factor once
    let s_factor = (2.0 * mu / base_geom.s_cubed).sqrt();

    // Sort TOFs by value (with original indices) for seeding effectiveness
    // Adjacent TOFs have similar solutions, so sorting maximizes seed quality (pdqsort for hot path)
    scratch.indexed_tofs.clear();
    scratch
        .indexed_tofs
        .extend(tofs.iter().copied().enumerate());
    if !tofs.windows(2).all(|pair| {
        pair.first()
            .zip(pair.get(1))
            .and_then(|(first, second)| first.partial_cmp(second))
            .is_some_and(std::cmp::Ordering::is_le)
    }) {
        pdqsort::sort_by(&mut scratch.indexed_tofs, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Number of (m, prograde) combinations for seeding state
    let m_max = m_max.max(0);
    let n_combos = revolution_pair_count(m_max);

    // Track last x value for each (m, prograde) combination
    // Key: (m, prograde as 0/1) -> last_x
    scratch.last_x_seeds.clear();
    scratch.last_x_seeds.resize(n_combos, None);

    // Helper to get seed index for (m, prograde)
    let seed_idx = revolution_branch_index;

    // Pre-allocate result vector in original input order.
    scratch.tof_results.clear();
    scratch
        .tof_results
        .resize(tofs.len(), BatchTofResult::default());

    // Pre-check: can any multi-rev solutions exist? (hoist out of inner loop)
    let can_multirev = base_geom.ll_base.abs() >= 1e-10;
    let effective_m_max = if can_multirev { m_max } else { 0 };

    // Process TOFs in sorted order for optimal seeding
    for &(orig_idx, tof) in &scratch.indexed_tofs {
        // Create geometry with correct t_nd for this TOF
        let mut geom = base_geom;
        geom.t_nd = s_factor * tof;

        // Track best result for this TOF
        let mut best = BatchTofResult {
            tof,
            ..BatchTofResult::default()
        };

        for m in 0..=effective_m_max {
            for prograde in [true, false] {
                let idx = seed_idx(m, prograde);
                let x_seed = scratch.last_x_seeds.get(idx).copied().flatten();

                // Solve with seeding from previous TOF's solution
                let (res, x) = izzo2015_impl_with_geom_seeded(
                    &geom,
                    m,
                    prograde,
                    true,
                    8,
                    CONVERGENCE_TOL,
                    CONVERGENCE_TOL,
                    x_seed,
                );

                if res.success {
                    // Update seed for next TOF (even if not best, track per m/prograde)
                    if let Some(seed) = scratch.last_x_seeds.get_mut(idx) {
                        *seed = Some(x);
                    }

                    // Calculate delta-V using FMA: d^2 = dx*dx + dy*dy + dz*dz
                    let dv1 = v1_ref.map_or(0.0, |vref| distance3(&res.v1, vref));

                    let dv2 = v2_ref.map_or(0.0, |vref| distance3(&res.v2, vref));

                    let total_dv = dv1 + dv2;

                    if total_dv < best.dv_depart + best.dv_arrive {
                        best = BatchTofResult {
                            tof,
                            dv_depart: dv1,
                            dv_arrive: dv2,
                            v1: res.v1,
                            v2: res.v2,
                            m,
                            prograde,
                            valid: true,
                        };
                    }
                }
            }
        }

        if let Some(result) = scratch.tof_results.get_mut(orig_idx) {
            *result = best;
        }
    }

    &scratch.tof_results
}

/// Batch solve Lambert problem for multiple TOF values with variable target positions.
///
/// Unlike `izzo2015_batch_tof` which uses a fixed r2 for all TOFs, this function
/// accepts a separate target position for each TOF. This is essential for rendezvous
/// problems where the target satellite moves during different flight times.
///
/// **Seeded solving**: TOFs are sorted before processing, and the solution from
/// each TOF is used as the initial guess for the next. Even though r2 varies,
/// adjacent TOFs have similar solutions when sorted, reducing iterations.
///
/// # Arguments
/// * `mu` - Gravitational parameter [km^3/s^2]
/// * `r1` - Initial position vector (constant) [km]
/// * `r2_vec` - Final position vectors, one per TOF [km]
/// * `v1_ref` - Reference velocity at departure (for delta-V calculation) [km/s]
/// * `v2_refs` - Reference velocities at arrival, one per TOF (for delta-V calculation) [km/s]
/// * `tofs` - Array of time-of-flight values [s]
/// * `m_max` - Maximum number of revolutions to consider
///
/// # Returns
/// Vector of results, one per TOF (in original input order), with minimum delta-V solution
#[must_use]
pub fn izzo2015_batch_tof_variable_r2(
    mu: f64,
    r1: &[f64; 3],
    r2_vec: &[[f64; 3]],
    v1_ref: &[f64; 3],
    v2_refs: &[[f64; 3]],
    tofs: &[f64],
    m_max: i32,
) -> Vec<BatchTofResult> {
    let mut scratch = VariableR2LambertScratch::default();
    izzo2015_batch_tof_variable_r2_with_scratch(
        mu,
        r1,
        r2_vec,
        v1_ref,
        v2_refs,
        tofs,
        m_max,
        &mut scratch,
    )
    .to_vec()
}

/// Stamps the masked-lane SIMD packing block shared verbatim by the two
/// batch-TOF branch-best paths: prograde/retrograde `ll` base select,
/// valid-lane masking of `ll`/`t`/`p0` (padding 0.0/1.0/0.0), the lambda
/// power-vector build, and the `householder_simd4_adaptive` call. One
/// definition so a change to the padding or packing lands in both paths at
/// once. Expands to the lane solutions as a `[f64; 4]`.
macro_rules! pack_masked_simd_inputs {
    (
        prograde: $prograde:expr,
        ll_bases: ($ll_arr_prograde:expr, $ll_arr_retrograde:expr),
        t_arr_base: $t_arr_base:expr,
        lane_valid: $lane_valid:expr,
        lane_x0: $lane_x0:expr,
        m: $m:expr $(,)?
    ) => {{
        let ll_arr_base = if $prograde {
            &$ll_arr_prograde
        } else {
            &$ll_arr_retrograde
        };
        let [valid0, valid1, valid2, valid3] = $lane_valid;
        let [ll0, ll1, ll2, ll3] = *ll_arr_base;
        let [t0, t1, t2, t3] = $t_arr_base;
        let [x0, x1, x2, x3] = $lane_x0;
        let ll_arr: [f64; 4] = [
            if valid0 { ll0 } else { 0.0 },
            if valid1 { ll1 } else { 0.0 },
            if valid2 { ll2 } else { 0.0 },
            if valid3 { ll3 } else { 0.0 },
        ];
        let t_arr: [f64; 4] = [
            if valid0 { t0 } else { 1.0 },
            if valid1 { t1 } else { 1.0 },
            if valid2 { t2 } else { 1.0 },
            if valid3 { t3 } else { 1.0 },
        ];
        let p0_arr: [f64; 4] = [
            if valid0 { x0 } else { 0.0 },
            if valid1 { x1 } else { 0.0 },
            if valid2 { x2 } else { 0.0 },
            if valid3 { x3 } else { 0.0 },
        ];

        let lambda_vec = f64x4::new(ll_arr);
        let lambda_sq_vec = lambda_vec * lambda_vec;
        let lambda_cu_vec = lambda_sq_vec * lambda_vec;
        let lambda_fifth_vec = lambda_cu_vec * lambda_sq_vec;
        let t_v = f64x4::new(t_arr);
        let p0_v = f64x4::new(p0_arr);

        let x_solutions = householder_simd4_adaptive(
            p0_v,
            t_v,
            lambda_vec,
            lambda_sq_vec,
            lambda_cu_vec,
            lambda_fifth_vec,
            $m,
            8,
            CONVERGENCE_TOL,
            CONVERGENCE_TOL,
        );
        x_solutions.to_array()
    }};
}

/// Stamps the shared visit-pack emission loop: finite gate, prograde
/// `(ll, it1, it2)` sign flip, the `reconstruct_velocities`
/// and `reconstruct_velocities_optimized` chain, and the `dv_depart` and
/// `dv_arrive` pair — shared verbatim by the three visit-pack functions.
/// Per-site differences arrive as arguments: the lane pattern, the prologue
/// (record hooks / problem lookup), the geometry / prograde / arrival-state
/// expressions, and the `visit` call. The dv arrays are bound to the
/// caller-named identifiers in `dv:` so the `visit` fragment can see them
/// across macro hygiene.
///
/// The scalar enumerator is deliberately NOT stamped from this macro: it is
/// the differential oracle the pack paths are bit-compared against.
macro_rules! emit_lambert_pack_lanes {
    (
        pack: $pack:expr,
        solutions: ($x_arr:expr, $y_arr:expr),
        lane: $lane_pat:pat,
        prologue: { $($prologue:tt)* },
        geom: $geom:expr,
        prograde: $prograde:expr,
        state1: $state1:expr,
        state2: $state2:expr,
        dv: ($dv_depart:ident, $dv_arrive:ident),
        visit: { $($visit:tt)* } $(,)?
    ) => {
        for (($lane_pat, x), y) in $pack.iter().zip($x_arr).zip($y_arr) {
            $($prologue)*
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            let geom = $geom;
            let (ll, it1, it2) = if $prograde {
                (geom.ll_base, geom.it1_base, geom.it2_base)
            } else {
                (-geom.ll_base, -geom.it1_base, -geom.it2_base)
            };
            let (vr1, vr2, vt1, vt2) = reconstruct_velocities(
                x,
                y,
                geom.r1_norm,
                geom.r2_norm,
                ll,
                geom.gamma,
                geom.rho,
                geom.sigma,
            );
            let (v1, v2) = simd_lambert::reconstruct_velocities_optimized(
                &vec3_to_array(&geom.ir1),
                &vec3_to_array(&geom.ir2),
                &vec3_to_array(&it1),
                &vec3_to_array(&it2),
                vr1,
                vr2,
                vt1,
                vt2,
            );
            let state1 = $state1;
            let state2 = $state2;
            let $dv_depart = [v1[0] - state1[3], v1[1] - state1[4], v1[2] - state1[5]];
            let $dv_arrive = [state2[3] - v2[0], state2[4] - v2[1], state2[5] - v2[2]];
            $($visit)*
        }
    };
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "Lambert batch reconstruction requires the established floating-point evaluation order"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the allocation-free batch kernel keeps one validated dataflow without helper indirection"
)]
pub fn izzo2015_batch_tof_variable_r2_with_scratch<'a>(
    mu: f64,
    r1: &[f64; 3],
    r2_vec: &[[f64; 3]],
    v1_ref: &[f64; 3],
    v2_refs: &[[f64; 3]],
    tofs: &[f64],
    m_max: i32,
    scratch: &'a mut VariableR2LambertScratch,
) -> &'a [BatchTofResult] {
    let n = tofs.len();
    if n == 0 || r2_vec.len() != n || v2_refs.len() != n {
        scratch.results.clear();
        return &scratch.results;
    }

    // Sort TOFs by value (with original indices) for seeding effectiveness
    // Even with variable r2, adjacent sorted TOFs have similar solutions (pdqsort for hot path)
    scratch.indexed_tofs.clear();
    scratch
        .indexed_tofs
        .extend(tofs.iter().enumerate().map(|(i, &tof)| (i, tof)));
    if !tofs.windows(2).all(|pair| {
        pair.first()
            .zip(pair.get(1))
            .and_then(|(first, second)| first.partial_cmp(second))
            .is_some_and(std::cmp::Ordering::is_le)
    }) {
        pdqsort::sort_by(&mut scratch.indexed_tofs, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Number of (m, prograde) combinations for seeding state
    let m_max = m_max.max(0);
    let n_combos = revolution_pair_count(m_max);

    // Track last x value for each (m, prograde) combination.
    // For SIMD chunks: seeding propagates chunk-to-chunk (not lane-to-lane within a chunk).
    // All lanes in a chunk start from the same seed (the previous chunk's last result).
    scratch.last_x_seeds.clear();
    scratch.last_x_seeds.resize(n_combos, None);

    // Helper to get seed index for (m, prograde)
    let seed_idx = revolution_branch_index;

    // Pre-allocate results vector in original input order.
    scratch.results.clear();
    scratch.results.resize(n, BatchTofResult::default());

    let data_slice = scratch.indexed_tofs.as_slice();
    let total = data_slice.len();
    let tail_start = total & !3_usize;

    // r1 is fixed across the batch: hoist its norm/normalization out of the
    // per-TOF geometry builds (bit-identical, see LambertR1Cache).
    let r1_cache = LambertR1Cache::new(r1);

    // SIMD path: process chunks of 4 TOFs simultaneously via householder_simd4.
    let simd_data = data_slice.get(..tail_start).unwrap_or(&[]);
    for chunk_slice in simd_data.chunks_exact(4) {
        let Ok(chunk_ref) = <&[(usize, f64); 4]>::try_from(chunk_slice) else {
            continue;
        };
        let chunk = *chunk_ref;
        let [chunk0, chunk1, chunk2, chunk3] = chunk;
        let (Some(r2_0), Some(r2_1), Some(r2_2), Some(r2_3)) = (
            r2_vec.get(chunk0.0),
            r2_vec.get(chunk1.0),
            r2_vec.get(chunk2.0),
            r2_vec.get(chunk3.0),
        ) else {
            continue;
        };

        // Compute geometry for each of the 4 TOFs in this chunk.
        let geoms = [
            compute_lambert_geometry_with_r1(mu, &r1_cache, r2_0, chunk0.1),
            compute_lambert_geometry_with_r1(mu, &r1_cache, r2_1, chunk1.1),
            compute_lambert_geometry_with_r1(mu, &r1_cache, r2_2, chunk2.1),
            compute_lambert_geometry_with_r1(mu, &r1_cache, r2_3, chunk3.1),
        ];

        // Initialize best results for each lane.
        let mut bests = [
            BatchTofResult {
                tof: chunk0.1,
                ..BatchTofResult::default()
            },
            BatchTofResult {
                tof: chunk1.1,
                ..BatchTofResult::default()
            },
            BatchTofResult {
                tof: chunk2.1,
                ..BatchTofResult::default()
            },
            BatchTofResult {
                tof: chunk3.1,
                ..BatchTofResult::default()
            },
        ];

        let t_arr_base = geoms.map(|geom| geom.t_nd);
        let ll_arr_prograde = geoms.map(|geom| geom.ll_base);
        let ll_arr_retrograde = geoms.map(|geom| -geom.ll_base);

        // Try each (m, prograde) combination via SIMD.
        for m in 0..=m_max {
            for &prograde in &[true, false] {
                let sidx = seed_idx(m, prograde);
                let chunk_seed = scratch.last_x_seeds.get(sidx).copied().flatten();

                // Per-lane feasibility and initial guess computation.
                // Mirrors the early-exit logic in find_xy_seeded for each lane.
                let mut lane_x0 = NAN4;
                let mut lane_valid = [false; 4];

                for ((geom, lane_x0), lane_is_valid) in geoms
                    .iter()
                    .zip(lane_x0.iter_mut())
                    .zip(lane_valid.iter_mut())
                {
                    if !geom.success {
                        continue;
                    }
                    let can_multirev = geom.ll_base.abs() >= 1e-10;
                    let eff_m_max = if can_multirev { m_max } else { 0 };
                    if m > eff_m_max {
                        continue;
                    }
                    let ll = if prograde {
                        geom.ll_base
                    } else {
                        -geom.ll_base
                    };
                    let t_nd = geom.t_nd;
                    if !lambert_branch_feasible(ll, t_nd, m, 8, 1e-9, 1e-9) {
                        continue;
                    }
                    // Use chunk-to-chunk seed (same for all lanes); fall back to initial_guess.
                    *lane_x0 = chunk_seed.unwrap_or_else(|| initial_guess(t_nd, ll, m, true));
                    *lane_is_valid = true;
                }

                // If no lanes are feasible, skip the SIMD call.
                if !lane_valid.iter().any(|&v| v) {
                    continue;
                }

                // Pack per-lane ll, t_nd, and initial guesses into SIMD vectors.
                let x_arr = pack_masked_simd_inputs!(
                    prograde: prograde,
                    ll_bases: (ll_arr_prograde, ll_arr_retrograde),
                    t_arr_base: t_arr_base,
                    lane_valid: lane_valid,
                    lane_x0: lane_x0,
                    m: m,
                );

                // Unpack results per lane, reconstruct velocities, update bests.
                let mut last_valid_x: Option<f64> = None;
                for ((((lane_is_valid, x), geom), chunk_entry), best) in lane_valid
                    .into_iter()
                    .zip(x_arr)
                    .zip(geoms.iter())
                    .zip(chunk)
                    .zip(bests.iter_mut())
                {
                    if !lane_is_valid {
                        continue;
                    }
                    if !x.is_finite() {
                        continue;
                    }
                    let ll = if prograde {
                        geom.ll_base
                    } else {
                        -geom.ll_base
                    };
                    let y = compute_y(x, ll);
                    if !y.is_finite() {
                        continue;
                    }

                    let (vr1, vr2, vt1, vt2) = reconstruct_velocities(
                        x,
                        y,
                        geom.r1_norm,
                        geom.r2_norm,
                        ll,
                        geom.gamma,
                        geom.rho,
                        geom.sigma,
                    );

                    let (it1, it2) = if prograde {
                        (geom.it1_base, geom.it2_base)
                    } else {
                        (-geom.it1_base, -geom.it2_base)
                    };

                    let v1 = geom.ir1 * vr1 + it1 * vt1;
                    let v2 = geom.ir2 * vr2 + it2 * vt2;
                    let v1_arr = vec3_to_array(&v1);
                    let v2_arr = vec3_to_array(&v2);

                    let Some(v2_ref) = v2_refs.get(chunk_entry.0) else {
                        continue;
                    };
                    let dv1 = distance3(&v1_arr, v1_ref);
                    let dv2 = distance3(&v2_arr, v2_ref);

                    if dv1 < best.dv_depart {
                        *best = BatchTofResult {
                            tof: chunk_entry.1,
                            dv_depart: dv1,
                            dv_arrive: dv2,
                            v1: v1_arr,
                            v2: v2_arr,
                            m,
                            prograde,
                            valid: true,
                        };
                    }
                    last_valid_x = Some(x);
                }

                // Propagate the last valid lane's x as seed for the next chunk.
                if let Some(x) = last_valid_x {
                    if let Some(seed) = scratch.last_x_seeds.get_mut(sidx) {
                        *seed = Some(x);
                    }
                }
            }
        }

        for (chunk_entry, best) in chunk.into_iter().zip(bests) {
            if let Some(result) = scratch.results.get_mut(chunk_entry.0) {
                *result = best;
            }
        }
    }

    // Scalar tail: handle remaining TOFs (len % 4 != 0) with the existing scalar path.
    for (orig_idx, tof) in data_slice.get(tail_start..).unwrap_or(&[]) {
        let (Some(r2), Some(v2_ref)) = (r2_vec.get(*orig_idx), v2_refs.get(*orig_idx)) else {
            continue;
        };
        let geom = compute_lambert_geometry_with_r1(mu, &r1_cache, r2, *tof);

        let mut best = BatchTofResult {
            tof: *tof,
            ..BatchTofResult::default()
        };

        if !geom.success {
            if let Some(result) = scratch.results.get_mut(*orig_idx) {
                *result = best;
            }
            continue;
        }

        let can_multirev = geom.ll_base.abs() >= 1e-10;
        let effective_m_max = if can_multirev { m_max } else { 0 };

        for m in 0..=effective_m_max {
            for prograde in [true, false] {
                let sidx = seed_idx(m, prograde);
                let (res, x) = izzo2015_impl_with_geom_seeded(
                    &geom,
                    m,
                    prograde,
                    true,
                    8,
                    CONVERGENCE_TOL,
                    CONVERGENCE_TOL,
                    scratch.last_x_seeds.get(sidx).copied().flatten(),
                );

                if res.success {
                    if let Some(seed) = scratch.last_x_seeds.get_mut(sidx) {
                        *seed = Some(x);
                    }

                    let dv1 = distance3(&res.v1, v1_ref);
                    let dv2 = distance3(&res.v2, v2_ref);

                    if dv1 < best.dv_depart {
                        best = BatchTofResult {
                            tof: *tof,
                            dv_depart: dv1,
                            dv_arrive: dv2,
                            v1: res.v1,
                            v2: res.v2,
                            m,
                            prograde,
                            valid: true,
                        };
                    }
                }
            }
        }

        if let Some(result) = scratch.results.get_mut(*orig_idx) {
            *result = best;
        }
    }

    &scratch.results
}

// =============================================================================
// Batch TOF Dispatchers
// =============================================================================

/// Enumerate Lambert solutions for all M/prograde combinations without
/// allocating a result vector.
///
/// The callback receives the same tuple fields as [`izzo2015_batch_dv`].
///
/// Test-only, and kept as the SCALAR reference for the combined enumerator:
/// `combined_lowpath_enumerator_matches_separate_enumeration` runs two calls of
/// this (requested low-path, then the opposite) against one
/// `for_each_lambert_m_prograde_lowpaths` call, which HF-NEW-01 routes through
/// SIMD4. That comparison is the only thing holding the SIMD enumeration order
/// and its per-branch dv to the scalar path; deleting this deletes it.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "scalar reference the SIMD4 combined enumerator is differenced against"
    )
)]
pub fn for_each_lambert_m_prograde(
    mu: f64,
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    m_max: i32,
    low_path: bool,
    mut visit: impl FnMut(i32, bool, [f64; 3], [f64; 3], bool),
) {
    let r1 = [state1[0], state1[1], state1[2]];
    let r2 = [state2[0], state2[1], state2[2]];
    let m_max_feasible = compute_m_max_fast(&r1, &r2, tof, mu).min(m_max);
    if m_max_feasible < 0 {
        return;
    }
    let geom = compute_lambert_geometry(mu, &r1, &r2, tof);

    for m in 0..=m_max_feasible {
        for prograde in [true, false] {
            let res = izzo2015_impl_with_geom_fast(
                &geom,
                m,
                prograde,
                low_path,
                8,
                CONVERGENCE_TOL,
                CONVERGENCE_TOL,
            );
            if !res.success {
                continue;
            }
            let dv_depart = [
                res.v1[0] - state1[3],
                res.v1[1] - state1[4],
                res.v1[2] - state1[5],
            ];
            let dv_arrive = [
                state2[3] - res.v2[0],
                state2[4] - res.v2[1],
                state2[5] - res.v2[2],
            ];
            visit(m, prograde, dv_depart, dv_arrive, true);
        }
    }
}

/// Enumerate Lambert solutions for requested and opposite low-path branches
/// while computing geometry and feasible revolution count once.
///
/// The emitted order matches two separate `for_each_lambert_m_prograde` calls:
/// requested low-path first for all feasible revolutions, then the opposite
/// low-path for multi-revolution branches only.
///
/// Test-only, and kept as a differential oracle at both ends: it is the SIMD4
/// side of `combined_lowpath_enumerator_matches_separate_enumeration` (against
/// two scalar `for_each_lambert_m_prograde` calls), and the per-branch reference
/// `variable_r2_branch_best_batch_matches_scalar_branch_enumerator` differences
/// the batch branch-best solver against. Production enters the same enumeration
/// through `for_each_lambert_m_prograde_lowpaths_pruned_with_r1`.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "differential oracle: differenced against the scalar enumerator on one side \
                  and against the variable-r2 branch-best batch solver on the other"
    )
)]
pub fn for_each_lambert_m_prograde_lowpaths(
    mu: f64,
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    m_max: i32,
    requested_low_path: bool,
    visit: impl FnMut(i32, bool, bool, [f64; 3], [f64; 3], bool),
) {
    for_each_lambert_m_prograde_lowpaths_pruned(
        mu,
        state1,
        state2,
        tof,
        m_max,
        requested_low_path,
        true,
        visit,
    );
}

/// Branch enumerator with an optional retrograde prune.
///
/// When `include_retrograde` is false the `prograde == false` solves are
/// skipped entirely. Callers must only pass false when every retrograde
/// solution is provably rejected downstream (retrograde departure dv is
/// bounded below by the deployer's transfer-plane tangential speed, so a
/// dv cap under that bound can never admit one). Skipping changes no
/// surviving solution's floats: prograde branches solve exactly as before.
pub fn for_each_lambert_m_prograde_lowpaths_pruned(
    mu: f64,
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    m_max: i32,
    requested_low_path: bool,
    include_retrograde: bool,
    visit: impl FnMut(i32, bool, bool, [f64; 3], [f64; 3], bool),
) {
    let r1 = [state1[0], state1[1], state1[2]];
    for_each_lambert_m_prograde_lowpaths_pruned_with_r1(
        mu,
        &LambertR1Cache::new(&r1),
        state1,
        state2,
        tof,
        m_max,
        requested_low_path,
        include_retrograde,
        visit,
    );
}

/// `for_each_lambert_m_prograde_lowpaths_pruned` fast entry taking a
/// precomputed departure-side cache.
///
/// `r1_cache` must be `LambertR1Cache::new` of `state1`'s position so
/// callers with a fixed departure state can hoist the r1 normalization out
/// of a per-TOF loop; behavior is otherwise identical (bit-identical
/// geometry, same enumeration order and prune contract).
pub fn for_each_lambert_m_prograde_lowpaths_pruned_with_r1(
    mu: f64,
    r1_cache: &LambertR1Cache,
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    m_max: i32,
    requested_low_path: bool,
    include_retrograde: bool,
    visit: impl FnMut(i32, bool, bool, [f64; 3], [f64; 3], bool),
) {
    for_each_lambert_m_prograde_lowpaths_pruned_with_r1_counted(
        mu,
        r1_cache,
        state1,
        state2,
        tof,
        m_max,
        requested_low_path,
        include_retrograde,
        visit,
        || {},
    );
}

/// Is the SIMD4 branch-enumeration path wired in for this process?
///
/// R17 measurement switch. The `f64x4` pack is a large win on AVX2 and a loss
/// on NEON (`docs/plans/2026-08-08-r17-simd-lambert.md`), so this is a runtime
/// choice read once rather than a `cfg`, which keeps both arms in ONE binary
/// and makes the A/B immune to a stale target directory.
#[inline]
/// Which branch enumerator this process runs. **The SIMD4 pack is the default.**
///
/// Adopted 2026-08-08 on the user's ruling, on a delta-v criterion rather than
/// a front-identity one:
///
/// - The pack's max relative dv deviation from the scalar enumerator is
///   **4.554e-14** over 6,400 geometries
///   (`simd_pack_enumeration_matches_the_scalar_enumerator_branch_for_branch`,
///   which pins it at `PACKED_DV_RELATIVE_BOUND`), behind a max root shift of
///   7.327e-15 over 86,524 real production operands, and it drops **zero**
///   lanes the scalar solver converges.
/// - `CONVERGENCE_TOL`'s own doc records the repo already accepting a
///   **1.08e-13** max root shift at `d594900`. This perturbation is ~15x
///   smaller than one main already ships.
///
/// **Front identity is deliberately NOT the criterion, and must not be
/// reintroduced as one.** The Stage-1 front dedups on exact bit patterns and
/// then applies strict Pareto with no epsilon, so it reshuffles under any
/// last-bit change: a 2 ULP nudge to the *scalar* `acos` moves it 3.4x further
/// than this pack does, changing all 192 events where the pack changes 22.
/// Measurements in `docs/plans/2026-08-08-r17-simd-lambert-front.md`.
///
/// **The caveat that matters for the next change here: breadth, not peak.**
/// The pack has a *smaller* peak root shift than `d594900` and ~9.5x its front
/// movement, because `d594900`'s shift is a rare tail (mean 3.5e-17) while the
/// pack's is ~1e-15 across roughly a quarter of all solves. A max-root-shift
/// bound does not predict front movement; bound the distribution.
///
/// ISA note: `wide::f64x4` is one AVX register and two stacked NEON halves, so
/// the pack is ~1.16-1.18x on the x86 clusters the campaign flies and ~20%
/// SLOWER on aarch64 dev/gate hosts. Results are the same on both — that is
/// what the equivalence gate carries — but local MF timing baselines shift.
/// Solve one packed group of up to four `(m, low_path, prograde)` variants that
/// share this problem's geometry, and emit them in enumeration order.
///
/// Mirrors `simd_lambert::izzo2015_impl_with_geom_simd` per lane for everything
/// outside `find_xy`: the same `ll`/`it1`/`it2` sign flip, the same
/// `reconstruct_velocities`, the same FMA reconstruction.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Lambert state reconstruction requires the established floating-point evaluation order"
)]
fn visit_simd_variant_pack(
    geom: &LambertGeometry,
    state1: &[f64; 6],
    state2: &[f64; 6],
    pack: &[(i32, bool, bool)],
    visit: &mut impl FnMut(i32, bool, bool, [f64; 3], [f64; 3], bool),
    record_scalar_variant_solve: &mut impl FnMut(),
) {
    // Inactive lanes take ll = 2.0, which the kernel's own |ll| >= 1 pre-pass
    // rejects, so they cost no iterations and cannot extend the pack's trip
    // count.
    let mut ll_arr = LL_PAD4;
    let mut m_arr = [0_i32; 4];
    let mut low_path_arr = [false; 4];
    for (((lane, ll_lane), m_lane), low_path_lane) in pack
        .iter()
        .zip(ll_arr.iter_mut())
        .zip(m_arr.iter_mut())
        .zip(low_path_arr.iter_mut())
    {
        let &(m, low_path, prograde) = lane;
        *ll_lane = if prograde {
            geom.ll_base
        } else {
            -geom.ll_base
        };
        *m_lane = m;
        *low_path_lane = low_path;
    }

    let (x_arr, y_arr) = find_xy_simd4_m_variant(
        ll_arr,
        geom.t_nd,
        m_arr,
        low_path_arr,
        8,
        CONVERGENCE_TOL,
        CONVERGENCE_TOL,
    );

    emit_lambert_pack_lanes!(
        pack: pack,
        solutions: (x_arr, y_arr),
        lane: lane,
        prologue: {
            record_scalar_variant_solve();
            let &(m, low_path, prograde) = lane;
        },
        geom: geom,
        prograde: prograde,
        state1: state1,
        state2: state2,
        dv: (dv_depart, dv_arrive),
        visit: {
            visit(m, low_path, prograde, dv_depart, dv_arrive, true);
        },
    );
}

/// The SIMD4 pack enumeration, as an entry a test can call directly.
///
/// Emits exactly the branches, and in exactly the order, that the scalar loop
/// in `for_each_lambert_m_prograde_lowpaths_pruned_with_r1_counted` emits.
fn for_each_lambert_simd_pack_enumeration(
    geom: &LambertGeometry,
    state1: &[f64; 6],
    state2: &[f64; 6],
    m_max_feasible: i32,
    requested_low_path: bool,
    include_retrograde: bool,
    visit: &mut impl FnMut(i32, bool, bool, [f64; 3], [f64; 3], bool),
    record_scalar_variant_solve: &mut impl FnMut(),
) {
    // Same three nested loops, same emission order; variants stream into a
    // four-lane buffer that flushes when full, so no allocation and no cap
    // on `m_max_feasible`.
    let mut pack: [(i32, bool, bool); 4] = [(0, false, false); 4];
    let mut packed = 0_usize;
    for (low_path, include_single_rev) in [(requested_low_path, true), (!requested_low_path, false)]
    {
        if !include_single_rev && m_max_feasible <= 0 {
            continue;
        }
        for m in 0..=m_max_feasible {
            if !include_single_rev && m == 0 {
                continue;
            }
            for prograde in [true, false] {
                if !prograde && !include_retrograde {
                    continue;
                }
                if let Some(slot) = pack.get_mut(packed) {
                    *slot = (m, low_path, prograde);
                }
                packed = packed.saturating_add(1);
                if packed == pack.len() {
                    visit_simd_variant_pack(
                        geom,
                        state1,
                        state2,
                        &pack,
                        visit,
                        record_scalar_variant_solve,
                    );
                    packed = 0;
                }
            }
        }
    }
    if let Some(tail) = pack.get(..packed) {
        if !tail.is_empty() {
            visit_simd_variant_pack(
                geom,
                state1,
                state2,
                tail,
                visit,
                record_scalar_variant_solve,
            );
        }
    }
}

/// The original per-variant scalar enumerator, as an entry a test can call
/// directly.
///
/// Production always uses the packed enumerator, so a differential test that
/// went through the public entry for both arms would compare the pack against
/// itself and pass vacuously. The equivalence gate calls this and
/// `for_each_lambert_simd_pack_enumeration` explicitly for that reason.
#[cfg(test)]
fn for_each_lambert_scalar_branch_enumeration(
    geom: &LambertGeometry,
    state1: &[f64; 6],
    state2: &[f64; 6],
    m_max_feasible: i32,
    requested_low_path: bool,
    include_retrograde: bool,
    visit: &mut impl FnMut(i32, bool, bool, [f64; 3], [f64; 3], bool),
    record_scalar_variant_solve: &mut impl FnMut(),
) {
    for (low_path, include_single_rev) in [(requested_low_path, true), (!requested_low_path, false)]
    {
        if !include_single_rev && m_max_feasible <= 0 {
            continue;
        }
        for m in 0..=m_max_feasible {
            if !include_single_rev && m == 0 {
                continue;
            }
            for prograde in [true, false] {
                if !prograde && !include_retrograde {
                    continue;
                }
                record_scalar_variant_solve();
                let res = izzo2015_impl_with_geom_fast(
                    geom,
                    m,
                    prograde,
                    low_path,
                    8,
                    CONVERGENCE_TOL,
                    CONVERGENCE_TOL,
                );
                if !res.success {
                    continue;
                }
                let dv_depart = [
                    res.v1[0] - state1[3],
                    res.v1[1] - state1[4],
                    res.v1[2] - state1[5],
                ];
                let dv_arrive = [
                    state2[3] - res.v2[0],
                    state2[4] - res.v2[1],
                    state2[5] - res.v2[2],
                ];
                visit(m, low_path, prograde, dv_depart, dv_arrive, true);
            }
        }
    }
}

fn for_each_lambert_m_prograde_lowpaths_pruned_with_r1_counted(
    mu: f64,
    r1_cache: &LambertR1Cache,
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    m_max: i32,
    requested_low_path: bool,
    include_retrograde: bool,
    mut visit: impl FnMut(i32, bool, bool, [f64; 3], [f64; 3], bool),
    mut record_scalar_variant_solve: impl FnMut(),
) {
    let r1 = [state1[0], state1[1], state1[2]];
    let r2 = [state2[0], state2[1], state2[2]];
    let m_max_feasible = compute_m_max_fast(&r1, &r2, tof, mu).min(m_max);
    if m_max_feasible < 0 {
        return;
    }
    let geom = compute_lambert_geometry_with_r1(mu, r1_cache, &r2, tof);
    // A failed geometry has no Lambert arc to enumerate, and "the caller drops
    // it downstream" does not cover every failure: the collinear case
    // (`|ir1 x ir2| = 0`, a 0- or 180-degree transfer) returns with the norms
    // intact but `ll_base`, `gamma` and `sigma` at zero, so the reconstruction
    // yields `v1 = v2 = 0` and the caller sees a FINITE departure dv equal to
    // the deployer's own speed rather than a rejected non-finite one. The
    // remaining failure modes divide by a zero norm and are non-finite. The
    // selected-branch entries carry the same guard.
    if !geom.success {
        return;
    }

    for_each_lambert_simd_pack_enumeration(
        &geom,
        state1,
        state2,
        m_max_feasible,
        requested_low_path,
        include_retrograde,
        &mut visit,
        &mut record_scalar_variant_solve,
    );
}

/// One row of the cross-TOF streaming pack enumeration
/// ([`for_each_lambert_m_prograde_lowpaths_pruned_with_r1_multi_tof`]).
///
/// The caller owns the per-problem prunes: `m_max` arrives with any energy
/// prune already applied, and `include_retrograde` with the departure-dv
/// bound already decided, exactly as they would arrive at the single-problem
/// enumerator.
#[derive(Clone, Copy, Debug)]
pub struct MultiTofBranchProblem {
    /// Arrival state (position and velocity) at `tof`.
    pub state2: [f64; 6],
    /// Time of flight [s].
    pub tof: f64,
    /// Per-problem revolution ceiling, before the geometric
    /// `compute_m_max_fast` bound this enumerator applies itself.
    pub m_max: i32,
    /// Per-problem retrograde inclusion.
    pub include_retrograde: bool,
}

/// A staged lane of the cross-TOF pack: one `(m, low_path, prograde)` variant
/// of one problem, carrying that problem's geometry so a flush needs no
/// side-table lookups.
#[derive(Clone, Copy)]
struct MultiTofPackLane {
    problem_index: usize,
    geom: LambertGeometry,
    m: i32,
    low_path: bool,
    prograde: bool,
}

/// Solve one packed group of up to four variants drawn from possibly-distinct
/// problems, and emit them in staging order.
///
/// Per-lane mirror of `visit_simd_variant_pack`: the same sign flips, the same
/// poison hook, the same FMA reconstruction — only the geometry and `t_nd` are
/// per lane instead of shared, via `find_xy_simd4_m_variant_per_lane_t`. By
/// that kernel's lane independence, a variant's output here is bit-identical
/// to what the single-problem pack produces for the same variant.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Lambert state reconstruction requires the established floating-point evaluation order"
)]
fn visit_simd_variant_pack_multi_tof(
    state1: &[f64; 6],
    problems: &[MultiTofBranchProblem],
    pack: &[MultiTofPackLane],
    visit: &mut impl FnMut(usize, i32, bool, bool, [f64; 3], [f64; 3], bool),
    record_variant_solve: &mut impl FnMut(),
) {
    // Inactive (padding) lanes take ll = 2.0, which the kernel's own
    // |ll| >= 1 pre-pass rejects before their padding `t` is ever read.
    let mut ll_arr = LL_PAD4;
    let mut t_arr = T_PAD4;
    let mut m_arr = [0_i32; 4];
    let mut low_path_arr = [false; 4];
    for ((((lane, ll_lane), t_lane), m_lane), low_path_lane) in pack
        .iter()
        .zip(ll_arr.iter_mut())
        .zip(t_arr.iter_mut())
        .zip(m_arr.iter_mut())
        .zip(low_path_arr.iter_mut())
    {
        *ll_lane = if lane.prograde {
            lane.geom.ll_base
        } else {
            -lane.geom.ll_base
        };
        *t_lane = lane.geom.t_nd;
        *m_lane = lane.m;
        *low_path_lane = lane.low_path;
    }

    let (x_arr, y_arr) = find_xy_simd4_m_variant_per_lane_t(
        t_arr,
        ll_arr,
        m_arr,
        low_path_arr,
        8,
        CONVERGENCE_TOL,
        CONVERGENCE_TOL,
    );

    emit_lambert_pack_lanes!(
        pack: pack,
        solutions: (x_arr, y_arr),
        lane: lane,
        prologue: {
            record_variant_solve();
            let Some(problem) = problems.get(lane.problem_index) else {
                continue;
            };
        },
        geom: &lane.geom,
        prograde: lane.prograde,
        state1: state1,
        state2: &problem.state2,
        dv: (dv_depart, dv_arrive),
        visit: {
            visit(
                lane.problem_index,
                lane.m,
                lane.low_path,
                lane.prograde,
                dv_depart,
                dv_arrive,
                true,
            );
        },
    );
}

/// Cross-TOF streaming variant of the pruned branch enumerator.
///
/// Enumerates the branch variants of MANY independent `(state2, tof)`
/// problems sharing one departure state
/// (cf. [`for_each_lambert_m_prograde_lowpaths_pruned_with_r1`]), packing
/// variants ACROSS problem boundaries so the SIMD4 kernel runs near-full
/// instead of at the single-problem mean fill.
///
/// Emission is problem-major in input order, variant-minor in the exact
/// single-problem enumeration order, so a per-problem fold over `visit` sees
/// the same solutions in the same order as one enumerator call per problem —
/// and by `find_xy_simd4_m_variant_per_lane_t`'s lane independence, with the
/// same bits. Production has one packed arithmetic route; the independent
/// scalar oracle remains test-only.
pub fn for_each_lambert_m_prograde_lowpaths_pruned_with_r1_multi_tof(
    mu: f64,
    r1_cache: &LambertR1Cache,
    state1: &[f64; 6],
    problems: &[MultiTofBranchProblem],
    requested_low_path: bool,
    mut visit: impl FnMut(usize, i32, bool, bool, [f64; 3], [f64; 3], bool),
    mut record_variant_solve: impl FnMut(),
) {
    let r1 = [state1[0], state1[1], state1[2]];
    let mut pack: [MultiTofPackLane; 4] = [MultiTofPackLane {
        problem_index: 0,
        geom: LambertGeometry::default(),
        m: 0,
        low_path: false,
        prograde: false,
    }; 4];
    let mut packed = 0_usize;
    for (problem_index, problem) in problems.iter().enumerate() {
        let r2 = [problem.state2[0], problem.state2[1], problem.state2[2]];
        let m_max_feasible = compute_m_max_fast(&r1, &r2, problem.tof, mu).min(problem.m_max);
        if m_max_feasible < 0 {
            continue;
        }
        let geom = compute_lambert_geometry_with_r1(mu, r1_cache, &r2, problem.tof);
        // Same guard, same reason, as the single-problem entry
        // (`for_each_lambert_m_prograde_lowpaths_pruned_with_r1_counted`): a
        // collinear geometry reconstructs to a finite zero-velocity "solution"
        // that no downstream finite check rejects. Both arms of this entry must
        // carry it or the scalar fallback would enumerate a different set.
        if !geom.success {
            continue;
        }
        for (low_path, include_single_rev) in
            [(requested_low_path, true), (!requested_low_path, false)]
        {
            if !include_single_rev && m_max_feasible <= 0 {
                continue;
            }
            for m in 0..=m_max_feasible {
                if !include_single_rev && m == 0 {
                    continue;
                }
                for prograde in [true, false] {
                    if !prograde && !problem.include_retrograde {
                        continue;
                    }
                    if let Some(slot) = pack.get_mut(packed) {
                        *slot = MultiTofPackLane {
                            problem_index,
                            geom,
                            m,
                            low_path,
                            prograde,
                        };
                    }
                    packed = packed.saturating_add(1);
                    if packed == pack.len() {
                        visit_simd_variant_pack_multi_tof(
                            state1,
                            problems,
                            &pack,
                            &mut visit,
                            &mut record_variant_solve,
                        );
                        packed = 0;
                    }
                }
            }
        }
    }
    if let Some(tail) = pack.get(..packed) {
        if !tail.is_empty() {
            visit_simd_variant_pack_multi_tof(
                state1,
                problems,
                tail,
                &mut visit,
                &mut record_variant_solve,
            );
        }
    }
}

/// A staged lane of the selected-branch cross-row pack: one prograde/retro
/// variant of one `(state2, tof)` row at the selected `(m, low_path)` branch.
#[derive(Clone, Copy)]
struct ExactBranchPackLane {
    row_index: usize,
    geom: LambertGeometry,
    state2: [f64; 6],
    m: i32,
    low_path: bool,
    prograde: bool,
}

impl ExactBranchPackLane {
    /// Inert padding: the default geometry's `ll_base = 0.0` maps to the
    /// `ll = 2.0` padding convention below, so unused lanes never activate.
    const INERT: Self = Self {
        row_index: 0,
        geom: LambertGeometry {
            r1_norm: 0.0,
            r2_norm: 0.0,
            c_norm: 0.0,
            s: 0.0,
            s_cubed: 0.0,
            ir1: Vec3::new(0.0, 0.0, 0.0),
            ir2: Vec3::new(0.0, 0.0, 0.0),
            it1_base: Vec3::new(0.0, 0.0, 0.0),
            it2_base: Vec3::new(0.0, 0.0, 0.0),
            ll_base: 0.0,
            gamma: 0.0,
            rho: 0.0,
            sigma: 0.0,
            t_nd: 0.0,
            success: false,
        },
        state2: [0.0; 6],
        m: 0,
        low_path: false,
        prograde: false,
    };
}

/// Solve one packed group of up to four selected-branch lanes drawn from
/// possibly-distinct rows, emitting `(row_index, m, low_path, prograde,
/// dv_depart, dv_arrive)` per converged lane in staging order.
///
/// Kernel and reconstruction mirror `visit_simd_variant_pack` per lane —
/// same sign flips, same poison hook, same FMA order — with per-lane geometry
/// and `t_nd` via `find_xy_simd4_m_variant_per_lane_t`. By that kernel's lane
/// independence, a `(geometry, m, low_path, prograde)` variant solved here is
/// bit-identical to the same variant solved by the per-candidate enumerator's
/// pack: this is what restores ONE Lambert arithmetic across the two entries.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Lambert state reconstruction requires the established floating-point evaluation order"
)]
fn visit_simd_exact_branch_pack(
    state1: &[f64; 6],
    pack: &[ExactBranchPackLane],
    visit: &mut impl FnMut(usize, i32, bool, bool, [f64; 3], [f64; 3], &[f64; 6]),
) {
    // Padding lanes take ll = 2.0, rejected by the kernel's |ll| >= 1
    // pre-pass before their padding `t` is ever read.
    let mut ll_arr = LL_PAD4;
    let mut t_arr = T_PAD4;
    let mut m_arr = [0_i32; 4];
    let mut low_path_arr = [false; 4];
    for ((((lane, ll_lane), t_lane), m_lane), low_path_lane) in pack
        .iter()
        .zip(ll_arr.iter_mut())
        .zip(t_arr.iter_mut())
        .zip(m_arr.iter_mut())
        .zip(low_path_arr.iter_mut())
    {
        *ll_lane = if lane.prograde {
            lane.geom.ll_base
        } else {
            -lane.geom.ll_base
        };
        *t_lane = lane.geom.t_nd;
        *m_lane = lane.m;
        *low_path_lane = lane.low_path;
    }

    let (x_arr, y_arr) = find_xy_simd4_m_variant_per_lane_t(
        t_arr,
        ll_arr,
        m_arr,
        low_path_arr,
        8,
        CONVERGENCE_TOL,
        CONVERGENCE_TOL,
    );

    emit_lambert_pack_lanes!(
        pack: pack,
        solutions: (x_arr, y_arr),
        lane: lane,
        prologue: {},
        geom: &lane.geom,
        prograde: lane.prograde,
        state1: state1,
        state2: &lane.state2,
        dv: (dv_depart, dv_arrive),
        visit: {
            visit(
                lane.row_index,
                lane.m,
                lane.low_path,
                lane.prograde,
                dv_depart,
                dv_arrive,
                &lane.state2,
            );
        },
    );
}

/// One row of the cross-TOF selected-branch enumeration
/// ([`for_each_lambert_exact_branch_with_r1_multi_tof`]).
///
/// The caller owns the per-row prunes exactly as it does at the single-row
/// entry: `rev`/`low_path` are the selected branch and `include_retrograde`
/// arrives with the departure-dv bound already decided.
#[derive(Clone, Copy, Debug)]
pub struct MultiTofExactBranchProblem {
    /// Arrival state (position and velocity) at `tof`.
    pub state2: [f64; 6],
    /// Time of flight [s].
    pub tof: f64,
    /// Selected revolution count.
    pub rev: i32,
    /// Selected path side.
    pub low_path: bool,
    /// Per-row retrograde inclusion.
    pub include_retrograde: bool,
}

/// Cross-TOF streaming variant of [`for_each_lambert_exact_branch_with_r1`].
///
/// One enumeration over many `(state2, tof)` rows sharing a departure state,
/// packing the selected branch's variants ACROSS row boundaries.
///
/// The single-row entry stages at most two lanes (prograde, retrograde) into a
/// four-lane pack, so it runs the kernel at half fill or less; this entry fills
/// it from the next row. Emission is row-major in input order, prograde before
/// retrograde within a row — the single-row entry's order — and by
/// `find_xy_simd4_m_variant_per_lane_t`'s lane independence a row's lanes carry
/// the same bits either way. Production has one packed arithmetic route; the
/// independent scalar oracle remains test-only.
pub fn for_each_lambert_exact_branch_with_r1_multi_tof(
    mu: f64,
    r1_cache: &LambertR1Cache,
    state1: &[f64; 6],
    problems: &[MultiTofExactBranchProblem],
    mut visit: impl FnMut(usize, i32, bool, bool, [f64; 3], [f64; 3], bool),
) {
    let r1 = [state1[0], state1[1], state1[2]];
    let mut pack: [ExactBranchPackLane; 4] = [ExactBranchPackLane::INERT; 4];
    let mut packed = 0_usize;
    for (problem_index, problem) in problems.iter().enumerate() {
        // The single-row entry's guards, in its order.
        if problem.rev < 0 {
            continue;
        }
        let r2 = [problem.state2[0], problem.state2[1], problem.state2[2]];
        if compute_m_max_fast(&r1, &r2, problem.tof, mu) < problem.rev {
            continue;
        }
        let geom = compute_lambert_geometry_with_r1(mu, r1_cache, &r2, problem.tof);
        if !geom.success {
            continue;
        }
        for prograde in [true, false] {
            if !prograde && !problem.include_retrograde {
                continue;
            }
            if let Some(slot) = pack.get_mut(packed) {
                *slot = ExactBranchPackLane {
                    row_index: problem_index,
                    geom,
                    state2: problem.state2,
                    m: problem.rev,
                    low_path: problem.low_path,
                    prograde,
                };
            }
            packed = packed.saturating_add(1);
            if packed == pack.len() {
                visit_simd_exact_branch_pack(
                    state1,
                    &pack,
                    &mut |row, m, lane_low_path, lane_prograde, dv_depart, dv_arrive, _s2| {
                        visit(
                            row,
                            m,
                            lane_low_path,
                            lane_prograde,
                            dv_depart,
                            dv_arrive,
                            true,
                        );
                    },
                );
                packed = 0;
            }
        }
    }
    if let Some(tail) = pack.get(..packed) {
        if !tail.is_empty() {
            visit_simd_exact_branch_pack(
                state1,
                tail,
                &mut |row, m, lane_low_path, lane_prograde, dv_depart, dv_arrive, _s2| {
                    visit(
                        row,
                        m,
                        lane_low_path,
                        lane_prograde,
                        dv_depart,
                        dv_arrive,
                        true,
                    );
                },
            );
        }
    }
}

/// Fold one emitted selected-branch lane into its row's running best using the
/// established finite / strict `>=` reject against the incumbent and the same
/// dv -> velocity round trip.
fn fold_exact_branch_lane(
    branch_results: &mut [BranchBatchTofResult],
    state1: &[f64; 6],
    state2: &[f64; 6],
    row_index: usize,
    m: i32,
    low_path: bool,
    prograde: bool,
    dv_depart: [f64; 3],
    dv_arrive: [f64; 3],
) {
    let Some(best) = branch_results.get_mut(row_index) else {
        return;
    };
    let dv_depart_norm = norm3(&dv_depart);
    if !dv_depart_norm.is_finite() || dv_depart_norm >= best.dv_depart {
        return;
    }
    *best = BranchBatchTofResult {
        tof: best.tof,
        dv_depart: dv_depart_norm,
        dv_arrive: norm3(&dv_arrive),
        v1: [
            dv_depart[0] + state1[3],
            dv_depart[1] + state1[4],
            dv_depart[2] + state1[5],
        ],
        v2: [
            state2[3] - dv_arrive[0],
            state2[4] - dv_arrive[1],
            state2[5] - dv_arrive[2],
        ],
        m,
        low_path,
        prograde,
        valid: true,
    };
}

/// Selected-branch enumeration of ONE `(state1, state2, tof)` problem at
/// `(rev, low_path)`, through the production pack kernel.
///
/// This is the single-problem counterpart of the packed selected-branch batch
/// arm in `solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_lanes`:
/// the prograde (and, when included, retrograde) variant of the selected
/// branch solve in one `find_xy_simd4_m_variant_per_lane_t` pack, so the
/// returned dv bits equal both that batch arm's and the per-candidate
/// enumerator's pack lanes for the same variant.
///
/// Emission order is prograde first, then retrograde.
pub fn for_each_lambert_exact_branch_with_r1(
    mu: f64,
    r1_cache: &LambertR1Cache,
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    rev: i32,
    low_path: bool,
    include_retrograde: bool,
    mut visit: impl FnMut(i32, bool, bool, [f64; 3], [f64; 3], bool),
) {
    if rev < 0 {
        return;
    }
    let r1 = [state1[0], state1[1], state1[2]];
    let r2 = [state2[0], state2[1], state2[2]];
    if compute_m_max_fast(&r1, &r2, tof, mu) < rev {
        return;
    }
    let geom = compute_lambert_geometry_with_r1(mu, r1_cache, &r2, tof);
    if !geom.success {
        return;
    }

    let mut pack: [ExactBranchPackLane; 4] = [ExactBranchPackLane::INERT; 4];
    let mut packed = 0_usize;
    for prograde in [true, false] {
        if !prograde && !include_retrograde {
            continue;
        }
        if let Some(slot) = pack.get_mut(packed) {
            *slot = ExactBranchPackLane {
                row_index: 0,
                geom,
                state2: *state2,
                m: rev,
                low_path,
                prograde,
            };
        }
        packed = packed.saturating_add(1);
    }
    if let Some(staged) = pack.get(..packed) {
        if !staged.is_empty() {
            visit_simd_exact_branch_pack(
                state1,
                staged,
                &mut |_row,
                      m,
                      lane_low_path,
                      prograde,
                      dv_depart,
                      dv_arrive,
                      _state2: &[f64; 6]| {
                    visit(m, lane_low_path, prograde, dv_depart, dv_arrive, true);
                },
            );
        }
    }
}

/// Unpruned entry to the variable-r2 branch-best batch solve.
///
/// Delegates to the `_pruned_` form with `include_retrograde = true`, so the two
/// agree by construction; `variable_r2_branch_best_batch_matches_scalar_branch_enumerator`
/// differences this path against `for_each_lambert_m_prograde_lowpaths`.
#[cfg_attr(
    not(any(test, feature = "bench-internal")),
    expect(
        dead_code,
        reason = "benches/lambert_solver_bench.rs is the only non-test caller and reaches it \
                  through the `bench-internal` re-export in lib.rs; production takes the \
                  retrograde-pruned form directly"
    )
)]
pub fn solve_lambert_batch_tof_variable_r2_branch_best_with_scratch<'a>(
    mu: f64,
    r1: &[f64; 3],
    r2_vec: &[[f64; 3]],
    v1_ref: &[f64; 3],
    v2_refs: &[[f64; 3]],
    tofs: &[f64],
    m_max: i32,
    requested_low_path: bool,
    branch_selection: Option<(i32, bool)>,
    scratch: &'a mut VariableR2LambertScratch,
) -> &'a [BranchBatchTofResult] {
    solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_scratch(
        mu,
        r1,
        r2_vec,
        v1_ref,
        v2_refs,
        tofs,
        m_max,
        requested_low_path,
        true,
        branch_selection,
        scratch,
    )
}

/// Branch-best variable-r2 batch with an optional whole-batch retrograde
/// prune (the SIMD counterpart of
/// `for_each_lambert_m_prograde_lowpaths_pruned`).
///
/// When `include_retrograde` is false every `prograde == false` solve is
/// skipped: the m=0 SIMD chunk pass and the retrograde branches of the
/// scalar multi-rev tail. Callers must only pass false when every retrograde
/// solution is provably rejected downstream for EVERY `(r1, r2_vec[i])` pair
/// in the batch — the retrograde departure dv is bounded below by the
/// deployer's tangential speed in the transfer-plane prograde basis, and
/// that basis rotates with r2, so a bound established against one arrival
/// state does not cover the rest; use the minimum bound across the batch (or
/// a lane-uniform bound) against the acceptance cap.
///
/// Under that precondition the prune cannot change downstream selections:
/// prograde branches solve bit-identically (separate x-seed slots, unchanged
/// pass order), and a per-TOF best that would have been retrograde can only
/// be replaced by an alternative with a HIGHER `dv_depart` (or an invalid
/// row), which the same cap that justified the prune also rejects.
pub fn solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_scratch<'a>(
    mu: f64,
    r1: &[f64; 3],
    r2_vec: &[[f64; 3]],
    v1_ref: &[f64; 3],
    v2_refs: &[[f64; 3]],
    tofs: &[f64],
    m_max: i32,
    requested_low_path: bool,
    include_retrograde: bool,
    branch_selection: Option<(i32, bool)>,
    scratch: &'a mut VariableR2LambertScratch,
) -> &'a [BranchBatchTofResult] {
    solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_lanes(
        mu,
        r1,
        r2_vec,
        v1_ref,
        v2_refs,
        tofs,
        m_max,
        requested_low_path,
        include_retrograde,
        branch_selection,
        None,
        scratch,
    )
}

/// `solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_scratch` with
/// an optional [`BranchLanePrep`].
///
/// The prep only feeds the selected-branch (`branch_selection.is_some()`)
/// path, which is the one a caller runs repeatedly over one fixed batch. It
/// must have been built by `BranchLanePrep::rebuild` from the same `mu`, `r1`,
/// `r2_vec` and `tofs`; a prep whose length disagrees with the batch is
/// ignored rather than trusted.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Lambert branch reconstruction requires the established floating-point evaluation order"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the allocation-free branch kernel keeps one validated dataflow without helper indirection"
)]
pub fn solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_lanes<'a>(
    mu: f64,
    r1: &[f64; 3],
    r2_vec: &[[f64; 3]],
    v1_ref: &[f64; 3],
    v2_refs: &[[f64; 3]],
    tofs: &[f64],
    m_max: i32,
    requested_low_path: bool,
    include_retrograde: bool,
    branch_selection: Option<(i32, bool)>,
    lane_prep: Option<&BranchLanePrep>,
    scratch: &'a mut VariableR2LambertScratch,
) -> &'a [BranchBatchTofResult] {
    scratch.branch_telemetry = VariableR2BranchTelemetry::default();
    let n = tofs.len();
    if n == 0 || r2_vec.len() != n || v2_refs.len() != n {
        scratch.branch_results.clear();
        return &scratch.branch_results;
    }

    scratch.indexed_tofs.clear();
    scratch
        .indexed_tofs
        .extend(tofs.iter().enumerate().map(|(i, &tof)| (i, tof)));
    if !tofs.windows(2).all(|pair| {
        pair.first()
            .zip(pair.get(1))
            .and_then(|(first, second)| first.partial_cmp(second))
            .is_some_and(std::cmp::Ordering::is_le)
    }) {
        pdqsort::sort_by(&mut scratch.indexed_tofs, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let m_max = m_max.max(0);
    scratch.last_x_seeds.clear();
    scratch.last_x_seeds.resize(
        i32_to_usize_or_zero(m_max.saturating_add(1)).saturating_mul(4),
        None,
    );
    let seed_idx = |m: i32, low_path: bool, prograde: bool| -> usize {
        i32_to_usize_or_zero(m)
            .saturating_mul(4)
            .saturating_add(if low_path { 0 } else { 2 })
            .saturating_add(usize::from(!prograde))
    };
    let branch_allowed = |m: i32, low_path: bool, include_single_rev: bool| -> bool {
        if !include_single_rev && m == 0 {
            return false;
        }
        if let Some((selected_m, selected_low_path)) = branch_selection {
            return selected_m == m && selected_low_path == low_path;
        }
        true
    };

    scratch.branch_results.clear();
    scratch
        .branch_results
        .resize(n, BranchBatchTofResult::default());

    let data_slice = scratch.indexed_tofs.as_slice();
    let tail_start = data_slice.len() & !3_usize;
    let low_path_passes = [(requested_low_path, true), (!requested_low_path, false)];
    let mut branch_telemetry = VariableR2BranchTelemetry::default();

    // r1 is fixed across the batch: hoist its norm/normalization out of the
    // per-lane geometry builds (bit-identical, see LambertR1Cache).
    let r1_cache = LambertR1Cache::new(r1);

    if let Some((selected_m, selected_low_path)) = branch_selection {
        let state1 = state6(r1, v1_ref);
        let lanes = lane_prep.filter(|prep| prep.len() == n);
        debug_assert!(
            lane_prep.is_none() || lanes.is_some(),
            "BranchLanePrep length {:?} does not match batch length {}",
            lane_prep.map(BranchLanePrep::len),
            n
        );

        // R18: the selected-branch rows run the SAME pack kernel as the
        // per-candidate enumerator, packed ACROSS rows via the per-lane-t
        // kernel entry, so the same `(r1, r2, tof, m, low_path, prograde)`
        // returns the same dv bits from either entry — this closes the
        // dual-Lambert disagreement flagged in
        // `docs/plans/2026-08-08-r17-simd-lambert-front.md`. Acceptance is the
        // dv-deviation criterion (documented precedent 4.554e-14 accepted
        // against a 1.08e-13 shipped bound), NOT front identity; measured by
        // `exact_branch_pack_stays_inside_the_packed_dv_relative_bound`.
        let branch_results = scratch.branch_results.as_mut_slice();
        let mut pack: [ExactBranchPackLane; 4] = [ExactBranchPackLane::INERT; 4];
        let mut packed = 0_usize;
        for &(idx, tof) in data_slice {
            let (Some(r2), Some(v2_ref)) = (r2_vec.get(idx), v2_refs.get(idx)) else {
                continue;
            };
            if let Some(result) = branch_results.get_mut(idx) {
                *result = BranchBatchTofResult {
                    tof,
                    ..BranchBatchTofResult::default()
                };
            } else {
                continue;
            }
            if selected_m < 0 || (selected_m == 0 && selected_low_path != requested_low_path) {
                continue;
            }
            let prepared_lane = lanes.and_then(|prep| prep.lanes.get(idx));
            let m_max_feasible = if let Some((m_max_fast, _)) = prepared_lane {
                *m_max_fast
            } else {
                let r1_pos = [state1[0], state1[1], state1[2]];
                compute_m_max_fast(&r1_pos, r2, tof, mu)
            }
            .min(m_max);
            if selected_m > m_max_feasible {
                continue;
            }
            let geom = match prepared_lane {
                Some((_, geom)) => *geom,
                None => compute_lambert_geometry_with_r1(mu, &r1_cache, r2, tof),
            };
            if !geom.success {
                continue;
            }
            let state2 = state6(r2, v2_ref);
            for prograde in [true, false] {
                if !prograde && !include_retrograde {
                    continue;
                }
                branch_telemetry.simd_lane_solves =
                    branch_telemetry.simd_lane_solves.saturating_add(1);
                if let Some(slot) = pack.get_mut(packed) {
                    *slot = ExactBranchPackLane {
                        row_index: idx,
                        geom,
                        state2,
                        m: selected_m,
                        low_path: selected_low_path,
                        prograde,
                    };
                }
                packed = packed.saturating_add(1);
                if packed == pack.len() {
                    visit_simd_exact_branch_pack(
                        &state1,
                        &pack,
                        &mut |row_index,
                              m,
                              lane_low_path,
                              prograde,
                              dv_depart,
                              dv_arrive,
                              lane_state2: &[f64; 6]| {
                            fold_exact_branch_lane(
                                branch_results,
                                &state1,
                                lane_state2,
                                row_index,
                                m,
                                lane_low_path,
                                prograde,
                                dv_depart,
                                dv_arrive,
                            );
                        },
                    );
                    packed = 0;
                }
            }
        }
        if let Some(tail) = pack.get(..packed) {
            if !tail.is_empty() {
                visit_simd_exact_branch_pack(
                    &state1,
                    tail,
                    &mut |row_index,
                          m,
                          lane_low_path,
                          prograde,
                          dv_depart,
                          dv_arrive,
                          lane_state2: &[f64; 6]| {
                        fold_exact_branch_lane(
                            branch_results,
                            &state1,
                            lane_state2,
                            row_index,
                            m,
                            lane_low_path,
                            prograde,
                            dv_depart,
                            dv_arrive,
                        );
                    },
                );
            }
        }
        scratch.branch_telemetry = branch_telemetry;
        return &scratch.branch_results;
    }

    let simd_data = data_slice.get(..tail_start).unwrap_or(&[]);
    for chunk_slice in simd_data.chunks_exact(4) {
        let Ok(chunk_ref) = <&[(usize, f64); 4]>::try_from(chunk_slice) else {
            continue;
        };
        let chunk = *chunk_ref;
        let [chunk0, chunk1, chunk2, chunk3] = chunk;
        let (Some(r2_0), Some(r2_1), Some(r2_2), Some(r2_3)) = (
            r2_vec.get(chunk0.0),
            r2_vec.get(chunk1.0),
            r2_vec.get(chunk2.0),
            r2_vec.get(chunk3.0),
        ) else {
            continue;
        };
        let geoms = [
            compute_lambert_geometry_with_r1(mu, &r1_cache, r2_0, chunk0.1),
            compute_lambert_geometry_with_r1(mu, &r1_cache, r2_1, chunk1.1),
            compute_lambert_geometry_with_r1(mu, &r1_cache, r2_2, chunk2.1),
            compute_lambert_geometry_with_r1(mu, &r1_cache, r2_3, chunk3.1),
        ];
        let mut bests = [
            BranchBatchTofResult {
                tof: chunk0.1,
                ..BranchBatchTofResult::default()
            },
            BranchBatchTofResult {
                tof: chunk1.1,
                ..BranchBatchTofResult::default()
            },
            BranchBatchTofResult {
                tof: chunk2.1,
                ..BranchBatchTofResult::default()
            },
            BranchBatchTofResult {
                tof: chunk3.1,
                ..BranchBatchTofResult::default()
            },
        ];

        let t_arr_base = geoms.map(|geom| geom.t_nd);
        let ll_arr_prograde = geoms.map(|geom| geom.ll_base);
        let ll_arr_retrograde = geoms.map(|geom| -geom.ll_base);

        for (low_path, include_single_rev) in low_path_passes {
            for m in 0..=m_max {
                if m != 0 || !include_single_rev {
                    continue;
                }
                if !branch_allowed(m, low_path, include_single_rev) {
                    continue;
                }
                for prograde in [true, false] {
                    if !prograde && !include_retrograde {
                        continue;
                    }
                    let sidx = seed_idx(m, low_path, prograde);
                    let chunk_seed = scratch.last_x_seeds.get(sidx).copied().flatten();
                    let mut lane_x0 = NAN4;
                    let mut lane_valid = [false; 4];

                    for ((geom, lane_x0), lane_is_valid) in geoms
                        .iter()
                        .zip(lane_x0.iter_mut())
                        .zip(lane_valid.iter_mut())
                    {
                        if !geom.success {
                            continue;
                        }
                        let can_multirev = geom.ll_base.abs() >= 1e-10;
                        let effective_m_max = if can_multirev { m_max } else { 0 };
                        if m > effective_m_max {
                            continue;
                        }
                        let ll = if prograde {
                            geom.ll_base
                        } else {
                            -geom.ll_base
                        };
                        let t_nd = geom.t_nd;
                        if !lambert_branch_feasible(ll, t_nd, m, 8, 1e-9, 1e-9) {
                            continue;
                        }
                        *lane_x0 =
                            chunk_seed.unwrap_or_else(|| initial_guess(t_nd, ll, m, low_path));
                        *lane_is_valid = true;
                    }

                    if !lane_valid.iter().any(|&v| v) {
                        continue;
                    }
                    branch_telemetry.simd_lane_solves = branch_telemetry
                        .simd_lane_solves
                        .saturating_add(lane_valid.iter().filter(|&&valid| valid).count());

                    let x_arr = pack_masked_simd_inputs!(
                        prograde: prograde,
                        ll_bases: (ll_arr_prograde, ll_arr_retrograde),
                        t_arr_base: t_arr_base,
                        lane_valid: lane_valid,
                        lane_x0: lane_x0,
                        m: m,
                    );
                    let mut last_valid_x: Option<f64> = None;
                    for ((((lane_is_valid, x), geom), chunk_entry), best) in lane_valid
                        .into_iter()
                        .zip(x_arr)
                        .zip(geoms.iter())
                        .zip(chunk)
                        .zip(bests.iter_mut())
                    {
                        if !lane_is_valid {
                            continue;
                        }
                        if !x.is_finite() {
                            continue;
                        }
                        let ll = if prograde {
                            geom.ll_base
                        } else {
                            -geom.ll_base
                        };
                        let y = compute_y(x, ll);
                        if !y.is_finite() {
                            continue;
                        }

                        let (vr1, vr2, vt1, vt2) = reconstruct_velocities(
                            x,
                            y,
                            geom.r1_norm,
                            geom.r2_norm,
                            ll,
                            geom.gamma,
                            geom.rho,
                            geom.sigma,
                        );
                        let (it1, it2) = if prograde {
                            (geom.it1_base, geom.it2_base)
                        } else {
                            (-geom.it1_base, -geom.it2_base)
                        };
                        let v1 = geom.ir1 * vr1 + it1 * vt1;
                        let v2 = geom.ir2 * vr2 + it2 * vt2;
                        let v1_arr = vec3_to_array(&v1);
                        let v2_arr = vec3_to_array(&v2);

                        let Some(v2_ref) = v2_refs.get(chunk_entry.0) else {
                            continue;
                        };
                        let dv_depart = distance3(&v1_arr, v1_ref);
                        if !dv_depart.is_finite() || dv_depart >= best.dv_depart {
                            continue;
                        }
                        let dv_arrive = distance3(&v2_arr, v2_ref);
                        *best = BranchBatchTofResult {
                            tof: chunk_entry.1,
                            dv_depart,
                            dv_arrive,
                            v1: v1_arr,
                            v2: v2_arr,
                            m,
                            low_path,
                            prograde,
                            valid: true,
                        };
                        last_valid_x = Some(x);
                    }
                    if let Some(x) = last_valid_x {
                        if let Some(seed) = scratch.last_x_seeds.get_mut(sidx) {
                            *seed = Some(x);
                        }
                    }
                }
            }
        }

        for (chunk_entry, best) in chunk.into_iter().zip(bests) {
            if let Some(result) = scratch.branch_results.get_mut(chunk_entry.0) {
                *result = best;
            }
        }
    }

    let state1 = state6(r1, v1_ref);
    for (pos, (idx, tof)) in data_slice.iter().enumerate() {
        let (Some(r2), Some(v2_ref)) = (r2_vec.get(*idx), v2_refs.get(*idx)) else {
            continue;
        };
        let state2 = state6(r2, v2_ref);
        let simd_m0_eligible = branch_selection.is_none()
            && pos < tail_start
            && branch_allowed(0, requested_low_path, true);
        let prefilled = scratch.branch_results.get(*idx).copied();
        let simd_m0_filled = simd_m0_eligible && prefilled.is_some_and(|result| result.valid);
        if simd_m0_eligible {
            branch_telemetry.m0_simd_prefill_lanes =
                branch_telemetry.m0_simd_prefill_lanes.saturating_add(1);
            if simd_m0_filled {
                branch_telemetry.m0_simd_valid_lanes =
                    branch_telemetry.m0_simd_valid_lanes.saturating_add(1);
            } else {
                branch_telemetry.m0_scalar_fallback_lanes =
                    branch_telemetry.m0_scalar_fallback_lanes.saturating_add(1);
            }
        }
        let mut best = if simd_m0_filled {
            prefilled.unwrap_or_default()
        } else {
            BranchBatchTofResult {
                tof: *tof,
                ..BranchBatchTofResult::default()
            }
        };

        for_each_lambert_m_prograde_lowpaths_pruned_with_r1_counted(
            mu,
            &r1_cache,
            &state1,
            &state2,
            *tof,
            m_max,
            requested_low_path,
            include_retrograde,
            |m, low_path, prograde, dv_depart, dv_arrive, valid| {
                if simd_m0_filled && m == 0 && low_path == requested_low_path {
                    return;
                }
                if !valid || !branch_allowed(m, low_path, true) {
                    return;
                }
                let dv_depart_norm = norm3(&dv_depart);
                if !dv_depart_norm.is_finite() || dv_depart_norm >= best.dv_depart {
                    return;
                }
                let dv_arrive_norm = norm3(&dv_arrive);
                best = BranchBatchTofResult {
                    tof: *tof,
                    dv_depart: dv_depart_norm,
                    dv_arrive: dv_arrive_norm,
                    v1: add3(&dv_depart, v1_ref),
                    v2: subtract3(v2_ref, &dv_arrive),
                    m,
                    low_path,
                    prograde,
                    valid: true,
                };
            },
            || {
                branch_telemetry.scalar_variant_solves =
                    branch_telemetry.scalar_variant_solves.saturating_add(1);
            },
        );
        if let Some(result) = scratch.branch_results.get_mut(*idx) {
            *result = best;
        }
    }

    scratch.branch_telemetry = branch_telemetry;
    &scratch.branch_results
}

/// Izzo batch dV solver with x-value seeding for TOF continuity.
///
/// Uses Izzo solver with x-value propagation between consecutive TOFs
/// for reduced iteration count on dense TOF scans.
///
/// Deliberately kept beside the flown unseeded path as its x-seeded twin
/// (`REFACTOR_BLOCKLIST.md`, "Only an inline test calls it"); only tests call it
/// since the allocating `solve_lambert_batch_dv` shell was removed.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "test-only x-seeded twin of the flown dv path")
)]
fn izzo2015_batch_dv_seeded_with_scratch<'a>(
    mu: f64,
    r1: &[f64; 3],
    r2: &[f64; 3],
    v1: &[f64; 3],
    tofs: &[f64],
    m: i32,
    low_path: bool,
    scratch: &'a mut LambertBatchScratch,
) -> &'a [f64] {
    if tofs.is_empty() {
        scratch.dv_results.clear();
        return &scratch.dv_results;
    }

    // Compute base geometry once
    let base_geom = compute_lambert_geometry(mu, r1, r2, 1.0);
    if !base_geom.success {
        scratch.dv_results.clear();
        scratch.dv_results.resize(tofs.len(), f64::INFINITY);
        return &scratch.dv_results;
    }

    // Time scaling factor
    let s_factor = (2.0 * mu / base_geom.s_cubed).sqrt();

    // Sort TOFs for seeding effectiveness
    scratch.indexed_tofs.clear();
    scratch
        .indexed_tofs
        .extend(tofs.iter().copied().enumerate());
    if !tofs.windows(2).all(|pair| {
        pair.first()
            .zip(pair.get(1))
            .and_then(|(first, second)| first.partial_cmp(second))
            .is_some_and(std::cmp::Ordering::is_le)
    }) {
        pdqsort::sort_by(&mut scratch.indexed_tofs, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    scratch.dv_results.clear();
    scratch.dv_results.resize(tofs.len(), f64::INFINITY);
    let mut x_seed: Option<f64> = None;

    for &(orig_idx, tof) in &scratch.indexed_tofs {
        // Create geometry with correct t_nd
        let mut geom = base_geom;
        geom.t_nd = s_factor * tof;

        // Solve with seeding
        let (res, x) = izzo2015_impl_with_geom_seeded(
            &geom,
            m,
            true,
            low_path,
            8,
            CONVERGENCE_TOL,
            CONVERGENCE_TOL,
            x_seed,
        );

        let dv = if res.success {
            // Calculate departure delta-v magnitude using FMA
            distance3(&res.v1, v1)
        } else {
            f64::INFINITY
        };

        if let Some(result) = scratch.dv_results.get_mut(orig_idx) {
            *result = dv;
        }

        // Propagate seed for next iteration
        if x.is_finite() {
            x_seed = Some(x);
        }
    }

    &scratch.dv_results
}

/// Find the best (minimum departure dV) Lambert solution across all (M, prograde) combinations.
///
/// # Arguments
/// * `state1` - Full state at departure [x, y, z, vx, vy, vz] (km, km/s)
/// * `state2` - Full state at arrival [x, y, z, vx, vy, vz] (km, km/s)
/// * `tof` - Time of flight (seconds)
/// * `m_max` - Maximum number of complete revolutions
/// * `low_path` - If true, select Izzo's geometric low-path multi-rev branch (larger-x root, poliastro convention; typically the HIGHER delta-v branch)
///
/// # Returns
/// (m, prograde, `dv_depart`, `dv_arrive`, `dv_depart_norm`, valid)
#[cfg_attr(
    not(any(test, feature = "bench-internal")),
    expect(
        dead_code,
        reason = "benches/lambert_solver_bench.rs is the only non-test caller and reaches it \
                  through the `bench-internal` re-export in lib.rs; test_best_solution_finds_minimum \
                  differences its minimum against the full izzo2015_batch_dv enumeration"
    )
)]
#[must_use]
pub fn izzo2015_best_solution(
    mu: f64,
    state1: &[f64; 6],
    state2: &[f64; 6],
    tof: f64,
    m_max: i32,
    low_path: bool,
) -> (i32, bool, [f64; 3], [f64; 3], f64, bool) {
    let r1 = [state1[0], state1[1], state1[2]];
    let r2 = [state2[0], state2[1], state2[2]];

    // Pre-filter: compute maximum feasible m value based on geometry
    let m_max_feasible = compute_m_max_fast(&r1, &r2, tof, mu).min(m_max);

    // Hoist invariant geometry
    let geom = compute_lambert_geometry(mu, &r1, &r2, tof);

    // Parallel path for larger m_max (only if not already in a Rayon context)
    {
        let is_nested = rayon::current_thread_index().is_some();
        if !is_nested && m_max_feasible >= LAMBERT_PARALLEL_THRESHOLD {
            let min_len = 2;

            // One (m, prograde) candidate; identical closure body for the parallel
            // and single-worker-serial paths so both produce bit-equal results.
            let solve_candidate =
                |m: i32, prograde: bool| -> Option<(i32, bool, [f64; 3], [f64; 3], f64)> {
                    let res = simd_lambert::izzo2015_impl_with_geom_simd(
                        &geom,
                        m,
                        prograde,
                        low_path,
                        8,
                        CONVERGENCE_TOL,
                        CONVERGENCE_TOL,
                    );
                    if !res.success {
                        return None;
                    }
                    let dv_depart = [
                        res.v1[0] - state1[3],
                        res.v1[1] - state1[4],
                        res.v1[2] - state1[5],
                    ];
                    let dv_norm = norm3(&dv_depart);
                    if !dv_norm.is_finite() {
                        return None;
                    }
                    let dv_arrive = [
                        state2[3] - res.v2[0],
                        state2[4] - res.v2[1],
                        state2[5] - res.v2[2],
                    ];
                    Some((m, prograde, dv_depart, dv_arrive, dv_norm))
                };
            // Deterministic tie-break: dv_norm, then m, then prograde (true first).
            let pick_better =
                |a: (i32, bool, [f64; 3], [f64; 3], f64),
                 b: (i32, bool, [f64; 3], [f64; 3], f64)| {
                    let better = if b.4 < a.4 {
                        true
                    } else if b.4 > a.4 {
                        false
                    } else if b.0 < a.0 {
                        true
                    } else if b.0 > a.0 {
                        false
                    } else {
                        // prograde true preferred over false
                        b.1 && !a.1
                    };
                    if better {
                        b
                    } else {
                        a
                    }
                };

            // `current_num_threads() > 1`: under a single-worker global pool the
            // global worker serializes concurrent callers on its LockLatch.
            // Collapse to a serial fold on the calling thread. This path (NOT the
            // small-m sequential loop below) is the byte-identical reference: it
            // uses the same SIMD closure and evaluates every m with no early-exit.
            let best = if rayon::current_num_threads() > 1 {
                let candidate_count = revolution_pair_count(m_max_feasible);
                let candidates: Vec<_> = (0..candidate_count)
                    .into_par_iter()
                    .with_min_len(min_len)
                    .map(|ordinal| {
                        let m = i32::try_from(ordinal / 2).unwrap_or(i32::MAX);
                        let prograde = ordinal % 2 == 0;
                        solve_candidate(m, prograde)
                    })
                    .collect();
                candidates.into_iter().flatten().reduce(pick_better)
            } else {
                (0..=m_max_feasible)
                    .flat_map(|m| [(m, true), (m, false)])
                    .filter_map(|(m, prograde)| solve_candidate(m, prograde))
                    .reduce(pick_better)
            };

            if let Some((best_m, best_prograde, best_dv_depart, best_dv_arrive, best_norm)) = best {
                return (
                    best_m,
                    best_prograde,
                    best_dv_depart,
                    best_dv_arrive,
                    best_norm,
                    true,
                );
            }
            return (0, true, [0.0; 3], [0.0; 3], f64::INFINITY, false);
        }
    }

    // Sequential path
    let mut best_m = 0;
    let mut best_prograde = true;
    let mut best_dv_depart = [0.0; 3];
    let mut best_dv_arrive = [0.0; 3];
    let mut best_norm = f64::INFINITY;
    let mut best_valid = false;

    // Early exit threshold: if M=0 gives excellent result, skip higher M
    let early_exit_threshold = LAMBERT_EARLY_EXIT_THRESHOLD;
    for m in 0..=m_max_feasible {
        for prograde in [true, false] {
            let res = izzo2015_impl_with_geom_fast(
                &geom,
                m,
                prograde,
                low_path,
                8,
                CONVERGENCE_TOL,
                CONVERGENCE_TOL,
            );
            if !res.success {
                continue;
            }
            let dv_depart = [
                res.v1[0] - state1[3],
                res.v1[1] - state1[4],
                res.v1[2] - state1[5],
            ];
            let dv_norm = norm3(&dv_depart);
            if !dv_norm.is_finite() {
                continue;
            }
            if dv_norm < best_norm {
                let dv_arrive = [
                    state2[3] - res.v2[0],
                    state2[4] - res.v2[1],
                    state2[5] - res.v2[2],
                ];
                best_m = m;
                best_prograde = prograde;
                best_dv_depart = dv_depart;
                best_dv_arrive = dv_arrive;
                best_norm = dv_norm;
                best_valid = true;
            }
        }
        // Early exit: if M=0 solution is excellent, skip higher M values
        // (multi-rev rarely beats direct transfer for LEO constellation transfers)
        if m == 0 && best_valid && early_exit_threshold > 0.0 && best_norm < early_exit_threshold {
            break;
        }
    }

    (
        best_m,
        best_prograde,
        best_dv_depart,
        best_dv_arrive,
        best_norm,
        best_valid,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    // Only the tests pin absolute dV magnitudes against a gravitational
    // parameter; the solvers all take `mu` as an argument.
    use satpy_core::MU;

    // LEO transfer orbit test case
    const R1_LEO: [f64; 3] = [6778.0, 0.0, 0.0]; // km
    const R2_LEO: [f64; 3] = [0.0, 7178.0, 0.0]; // km
    const TOF_LEO: f64 = 3600.0; // 1 hour

    /// `HYP2F1B_TABLE` is built by CTFE from `hyp2f1b_series`; this re-runs
    /// the same series (and the former `LazyLock` initializer's exact per-entry
    /// arithmetic, including the runtime `usize_to_f64_or_infinity` index
    /// conversion the const builder had to respell) at runtime and compares
    /// every entry by bits. It is the only value pin on the table: a red here
    /// means const eval and runtime eval of the series diverged, or the
    /// builder's arithmetic drifted from this oracle.
    #[test]
    fn hyp2f1b_table_matches_runtime_series_bitwise() {
        assert_eq!(HYP2F1B_TABLE.len(), HYP2F1B_TABLE_SIZE);
        for (i, &(val, deriv)) in HYP2F1B_TABLE.iter().enumerate() {
            let x = HYP2F1B_X_MIN + usize_to_f64_or_infinity(i) * HYP2F1B_STEP;
            let expected_val = hyp2f1b_series(x);
            let h = 1e-6;
            let expected_deriv = if x + h < 1.0 {
                (hyp2f1b_series(x + h) - hyp2f1b_series(x - h)) / (2.0 * h)
            } else {
                (hyp2f1b_series(x) - hyp2f1b_series(x - h)) / h
            };
            assert_eq!(val.to_bits(), expected_val.to_bits(), "entry {i} value");
            assert_eq!(
                deriv.to_bits(),
                expected_deriv.to_bits(),
                "entry {i} derivative"
            );
        }
    }

    /// The bit-spelled sqrt window consts must equal the sqrt expressions
    /// they replaced at `tof_equation_y`'s band gate — this reads the
    /// production consts directly (no shadow copy) and is the scalar
    /// counterpart of simd.rs's `simd_lane_constants_equal_their_splat_forms`.
    #[test]
    fn sqrt_window_consts_equal_their_sqrt_forms() {
        assert_eq!(SQRT_0_6.to_bits(), (0.6_f64).sqrt().to_bits());
        assert_eq!(SQRT_1_4.to_bits(), (1.4_f64).sqrt().to_bits());
    }

    #[test]
    fn feasible_revolution_range_keeps_the_i32_max_endpoint() {
        let mut full_range = feasible_revolution_range(i32::MAX);
        assert_eq!(full_range.next(), Some(0));
        assert_eq!(full_range.next_back(), Some(i32::MAX));
        assert!(feasible_revolution_range(-1).next().is_none());
    }

    // Full state with circular velocity
    fn leo_state1() -> [f64; 6] {
        let r = 6778.0;
        let v = (MU / r).sqrt();
        [r, 0.0, 0.0, 0.0, v, 0.0]
    }

    fn leo_state2() -> [f64; 6] {
        let r = 7178.0;
        let v = (MU / r).sqrt();
        [0.0, r, 0.0, -v, 0.0, 0.0]
    }

    fn norm_array(values: &[f64; 3]) -> f64 {
        values.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    /// The cross-TOF streaming pack must be a pure repacking: for every
    /// problem, the same variants, in the same order, with bit-identical dv
    /// vectors, as one single-problem enumerator call per problem. Both arms
    /// run the production SIMD pack, so this compares exactly the repacking
    /// across problem boundaries.
    ///
    /// CAVEAT, and it bounds what a green here means: this is a
    /// self-consistency test. Both arms share the same pack arithmetic, so a
    /// perturbation of that arithmetic cancels. Historical mutation runs
    /// confirmed that blindness. Never cite it as arithmetic coverage; it
    /// covers repacking only. The explicit scalar differential test covers
    /// packed arithmetic.
    #[test]
    fn multi_tof_streaming_enumeration_matches_per_problem_enumeration_bit_for_bit() {
        let state1 = leo_state1();
        let r1 = [state1[0], state1[1], state1[2]];
        let r1_cache = LambertR1Cache::new(&r1);

        // Deterministic corpus: LEO-ish arrival states over a TOF ladder wide
        // enough that multi-rev branches open up, with per-problem prunes.
        let mut seed = 0x2026_0808_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f64::from(u32::try_from(seed >> 32).unwrap_or(0)) / f64::from(u32::MAX)
        };
        let m_max_cycle = [0_i32, 1, 2, 3, 4];
        let retro_cycle = [false, true, true];
        let mut problems = Vec::new();
        for (&m_max, &include_retrograde) in m_max_cycle
            .iter()
            .cycle()
            .zip(retro_cycle.iter().cycle())
            .take(17)
        {
            let r_norm = 6_700.0 + next() * 1_600.0;
            let angle = 0.35 + next() * 5.0;
            let v_circ = (MU / r_norm).sqrt();
            let state2 = [
                r_norm * angle.cos(),
                r_norm * angle.sin(),
                r_norm * 0.05 * (next() - 0.5),
                -v_circ * angle.sin(),
                v_circ * angle.cos(),
                v_circ * 0.05 * (next() - 0.5),
            ];
            problems.push(MultiTofBranchProblem {
                state2,
                tof: 1_800.0 + next() * 88_000.0,
                m_max,
                include_retrograde,
            });
        }

        #[expect(
            clippy::items_after_statements,
            reason = "the alias belongs beside the two collections it types"
        )]
        type Emitted = (usize, i32, bool, bool, [u64; 3], [u64; 3]);
        let mut reference: Vec<Emitted> = Vec::new();
        for (problem_index, problem) in problems.iter().enumerate() {
            for_each_lambert_m_prograde_lowpaths_pruned_with_r1(
                MU,
                &r1_cache,
                &state1,
                &problem.state2,
                problem.tof,
                problem.m_max,
                true,
                problem.include_retrograde,
                |m, low_path, prograde, dv_depart, dv_arrive, valid| {
                    if valid {
                        reference.push((
                            problem_index,
                            m,
                            low_path,
                            prograde,
                            dv_depart.map(f64::to_bits),
                            dv_arrive.map(f64::to_bits),
                        ));
                    }
                },
            );
        }

        let mut streamed: Vec<Emitted> = Vec::new();
        for_each_lambert_m_prograde_lowpaths_pruned_with_r1_multi_tof(
            MU,
            &r1_cache,
            &state1,
            &problems,
            true,
            |problem_index, m, low_path, prograde, dv_depart, dv_arrive, valid| {
                if valid {
                    streamed.push((
                        problem_index,
                        m,
                        low_path,
                        prograde,
                        dv_depart.map(f64::to_bits),
                        dv_arrive.map(f64::to_bits),
                    ));
                }
            },
            || {},
        );

        assert_eq!(reference, streamed);

        // Non-vacuity: the corpus must actually exercise the repacking claim.
        // (a) enough variants that packs fill and flush repeatedly;
        // (b) at least one pack spans a problem boundary: some problem leaves
        //     a partially-filled pack behind and a later problem still emits;
        // (c) multi-rev and retrograde lanes are both present.
        assert!(
            reference.len() > 24,
            "corpus too small: {}",
            reference.len()
        );
        let mut per_problem = vec![0_usize; problems.len()];
        for &(problem_index, ..) in &reference {
            if let Some(count) = per_problem.get_mut(problem_index) {
                *count = count.saturating_add(1);
            }
        }
        let mut prefix = 0_usize;
        let mut boundary_crossed = false;
        for (problem_index, count) in per_problem.iter().enumerate() {
            prefix = prefix.saturating_add(*count);
            if !prefix.is_multiple_of(4)
                && per_problem
                    .get(problem_index.saturating_add(1)..)
                    .is_some_and(|rest| rest.iter().any(|&later| later > 0))
            {
                boundary_crossed = true;
            }
        }
        assert!(boundary_crossed, "no pack spanned a problem boundary");
        assert!(reference.iter().any(|&(_, m, ..)| m > 0));
        assert!(reference.iter().any(|&(_, _, _, prograde, _, _)| !prograde));
    }

    /// The selected-branch cross-TOF pack must be a pure repacking: for every
    /// row, the same variants, in the same order, with bit-identical dv
    /// vectors, as one `for_each_lambert_exact_branch_with_r1` call per row.
    ///
    /// The corpus deliberately mixes rows that emit two lanes with rows that
    /// emit one (retrograde excluded) and rows that emit none (`rev` above the
    /// row's geometric ceiling), because that is what makes the packing
    /// straddle row boundaries — the single-row entry can only ever stage two
    /// of four lanes, so without a mixture the pack boundary would land in the
    /// same place in both arms and the test would prove nothing.
    ///
    /// CAVEAT, as on the unselected-route gate above: both arms share the pack
    /// arithmetic, so an arithmetic perturbation cancels. Historical mutation
    /// runs confirmed that blindness. This tests repacking, never arithmetic.
    #[test]
    fn exact_branch_multi_tof_streaming_matches_per_row_enumeration_bit_for_bit() {
        let state1 = leo_state1();
        let r1 = [state1[0], state1[1], state1[2]];
        let r1_cache = LambertR1Cache::new(&r1);

        let mut seed = 0x2026_0809_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f64::from(u32::try_from(seed >> 32).unwrap_or(0)) / f64::from(u32::MAX)
        };
        // `40` is above every row's geometric ceiling at these TOFs (a LEO
        // period is ~5,600 s and the ladder tops out near 90,000 s), so those
        // rows emit nothing and shift the pack alignment.
        let rev_cycle = [0_i32, 1, 40, 2, 0, 3];
        let retro_cycle = [true, false, true];
        let low_path_cycle = [true, true, false];
        let mut problems = Vec::new();
        for ((&rev, &include_retrograde), &low_path) in rev_cycle
            .iter()
            .cycle()
            .zip(retro_cycle.iter().cycle())
            .zip(low_path_cycle.iter().cycle())
            .take(19)
        {
            let r_norm = 6_700.0 + next() * 1_600.0;
            let angle = 0.35 + next() * 5.0;
            let v_circ = (MU / r_norm).sqrt();
            let state2 = [
                r_norm * angle.cos(),
                r_norm * angle.sin(),
                r_norm * 0.05 * (next() - 0.5),
                -v_circ * angle.sin(),
                v_circ * angle.cos(),
                v_circ * 0.05 * (next() - 0.5),
            ];
            problems.push(MultiTofExactBranchProblem {
                state2,
                tof: 1_800.0 + next() * 88_000.0,
                rev,
                low_path,
                include_retrograde,
            });
        }

        #[expect(
            clippy::items_after_statements,
            reason = "the alias belongs beside the two collections it types"
        )]
        type Emitted = (usize, i32, bool, bool, [u64; 3], [u64; 3]);
        let mut reference: Vec<Emitted> = Vec::new();
        for (row_index, problem) in problems.iter().enumerate() {
            for_each_lambert_exact_branch_with_r1(
                MU,
                &r1_cache,
                &state1,
                &problem.state2,
                problem.tof,
                problem.rev,
                problem.low_path,
                problem.include_retrograde,
                |m, low_path, prograde, dv_depart, dv_arrive, valid| {
                    if valid {
                        reference.push((
                            row_index,
                            m,
                            low_path,
                            prograde,
                            dv_depart.map(f64::to_bits),
                            dv_arrive.map(f64::to_bits),
                        ));
                    }
                },
            );
        }

        let mut streamed: Vec<Emitted> = Vec::new();
        for_each_lambert_exact_branch_with_r1_multi_tof(
            MU,
            &r1_cache,
            &state1,
            &problems,
            |row_index, m, low_path, prograde, dv_depart, dv_arrive, valid| {
                if valid {
                    streamed.push((
                        row_index,
                        m,
                        low_path,
                        prograde,
                        dv_depart.map(f64::to_bits),
                        dv_arrive.map(f64::to_bits),
                    ));
                }
            },
        );

        assert_eq!(reference, streamed);

        // Non-vacuity: enough lanes to fill and flush packs repeatedly, at
        // least one pack spanning a row boundary, and both a multi-rev and a
        // retrograde lane present.
        assert!(
            reference.len() > 12,
            "corpus too small: {}",
            reference.len()
        );
        let mut per_row = vec![0_usize; problems.len()];
        for &(row_index, ..) in &reference {
            if let Some(count) = per_row.get_mut(row_index) {
                *count = count.saturating_add(1);
            }
        }
        let mut prefix = 0_usize;
        let mut boundary_crossed = false;
        for (row_index, count) in per_row.iter().enumerate() {
            prefix = prefix.saturating_add(*count);
            if !prefix.is_multiple_of(4)
                && per_row
                    .get(row_index.saturating_add(1)..)
                    .is_some_and(|rest| rest.iter().any(|&later| later > 0))
            {
                boundary_crossed = true;
            }
        }
        assert!(boundary_crossed, "no pack spanned a row boundary");
        assert!(reference.iter().any(|&(_, m, ..)| m > 0));
        assert!(reference.iter().any(|&(_, _, _, prograde, _, _)| !prograde));
        assert!(
            per_row.contains(&0),
            "corpus must include a row that emits nothing, else the pack \
             boundaries coincide in both arms"
        );
    }

    /// The pre-R21 `halley_method` body, frozen, as this test's poison.
    ///
    /// It differs from the shipped solver in exactly one way: `t` stays at the
    /// seed's `T(0.1)` instead of tracking `T(x)`. It is duplicated rather than
    /// reached through a flag because its whole job is to be the OLD arithmetic
    /// — if someone improves the real solver, this must NOT follow, or the
    /// discrimination assert below silently stops discriminating.
    fn seed_frozen_t_min(ll: f64, m: i32, maxiter: i32, atol: f64, rtol: f64) -> f64 {
        let ll2 = ll * ll;
        let ll3 = ll2 * ll;
        let ll5 = ll3 * ll2;
        let one_minus_ll2 = 1.0 - ll2;
        let two_ll3 = 2.0 * ll3;
        let two_one_minus_ll2_ll3 = 2.0 * one_minus_ll2 * ll3;
        let six_one_minus_ll2_ll5 = 6.0 * one_minus_ll2 * ll5;
        let mut p0 = 0.1;
        let t0 = tof_equation(p0, 0.0, ll, m);
        for _ in 0..maxiter {
            let y = compute_y(p0, ll);
            let one_minus_x2 = 1.0 - p0 * p0;
            let fder = tof_equation_p(p0, y, t0, two_ll3, one_minus_x2);
            let fder2 = tof_equation_p2(p0, y, t0, fder, two_one_minus_ll2_ll3, one_minus_x2);
            if fder2 == 0.0 {
                return f64::NAN;
            }
            let fder3 = tof_equation_p3(p0, y, fder, fder2, six_one_minus_ll2_ll5, one_minus_x2);
            let denom = (2.0 * fder2).mul_add(fder2, -fder * fder3);
            let p = p0 - 2.0 * fder * fder2 / denom;
            if (p - p0).abs() < rtol * p0.abs() + atol {
                return tof_equation(p, 0.0, ll, m);
            }
            p0 = p;
        }
        f64::NAN
    }

    /// Derivative-free reference minimum of `T(x)` at fixed `m`: golden section
    /// over `(-1, 1)`, run to the float floor.
    ///
    /// Deliberately does NOT use `tof_equation_p`. The defect this test exists
    /// for lived in how that derivative was evaluated, so a reference built on
    /// it could not have caught it.
    fn reference_t_min(ll: f64, revolutions: i32) -> f64 {
        let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0;
        let (mut lo, mut hi) = (-0.999_999_9_f64, 0.999_999_9_f64);
        let mut probe_lo = hi - (hi - lo) * inv_phi;
        let mut probe_hi = lo + (hi - lo) * inv_phi;
        let mut value_lo = tof_equation(probe_lo, 0.0, ll, revolutions);
        let mut value_hi = tof_equation(probe_hi, 0.0, ll, revolutions);
        for _ in 0..400 {
            if value_lo < value_hi {
                hi = probe_hi;
                probe_hi = probe_lo;
                value_hi = value_lo;
                probe_lo = hi - (hi - lo) * inv_phi;
                value_lo = tof_equation(probe_lo, 0.0, ll, revolutions);
            } else {
                lo = probe_lo;
                probe_lo = probe_hi;
                value_lo = value_hi;
                probe_hi = lo + (hi - lo) * inv_phi;
                value_hi = tof_equation(probe_hi, 0.0, ll, revolutions);
            }
            if (hi - lo).abs() < 1e-16 * (1.0 + lo.abs()) {
                break;
            }
        }
        tof_equation(0.5 * (lo + hi), 0.0, ll, revolutions)
    }

    /// `compute_t_min` must return the TRUE minimum time of flight of the
    /// `m`-revolution branch, because the `t < t_min` prune in [`find_xy`]
    /// decides whether that branch EXISTS. An overshoot silently deletes real
    /// transfers; there is no safe epsilon in that direction.
    ///
    /// This is the first test of `compute_t_min`'s value. The only other test
    /// that names it, `test_find_xy_simd4_m_variant_matches_scalar`, picks
    /// lanes where the check deliberately does not engage.
    ///
    /// # What is pinned, and what is deliberately not
    ///
    /// The bound is on `t_min`, NOT on `x_min`. The minimum is quadratic, so a
    /// relative error of ~1e-16 in `T` sits ~1e-8 away in `x`: even an exact
    /// solve lands `x_min` about 2.1e-8 from the reference's. Pinning `x_min`
    /// tightly would pin the conditioning of a flat minimum, not the accuracy
    /// of the solve, and would be fragile across libms.
    #[test]
    fn compute_t_min_lands_on_the_true_tof_minimum() {
        let mut worst_rel = 0.0_f64;
        let mut worst_at = (0.0_f64, 0_i32);
        let mut worst_frozen_rel = 0.0_f64;
        let mut rows = 0_u32;
        for m in 1..=5_i32 {
            for step in 0..=200_i32 {
                let ll = -0.99 + 1.98 * f64::from(step) / 200.0;
                let reference = reference_t_min(ll, m);
                let (_, t_min) = compute_t_min(ll, m, 8, CONVERGENCE_TOL, CONVERGENCE_TOL);
                assert!(
                    t_min.is_finite(),
                    "t_min must converge for (ll={ll}, m={m})"
                );
                rows += 1;
                let rel = ((t_min - reference) / reference.abs().max(1e-300)).abs();
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_at = (ll, m);
                }
                let frozen = seed_frozen_t_min(ll, m, 8, CONVERGENCE_TOL, CONVERGENCE_TOL);
                if frozen.is_finite() {
                    worst_frozen_rel = worst_frozen_rel
                        .max(((frozen - reference) / reference.abs().max(1e-300)).abs());
                }
            }
        }
        assert!(rows >= 1_000, "grid must actually run (rows={rows})");
        assert!(
            worst_rel < 1e-14,
            "compute_t_min worst relative error {worst_rel:.4e} at ll={:.4} m={} exceeds 1e-14",
            worst_at.0,
            worst_at.1
        );
        // Discrimination: the pre-R21 shape must FAIL the bound above by
        // orders, else this test would pass on the defect it was written for.
        assert!(
            worst_frozen_rel > 1e-6,
            "the seed-frozen poison only deviates by {worst_frozen_rel:.4e}; this test \
             no longer discriminates the defect it exists for"
        );
        println!(
            "compute_t_min worst relative error {worst_rel:.4e} over {rows} rows; \
             seed-frozen poison {worst_frozen_rel:.4e}"
        );
    }

    /// Non-vacuity for the prune itself: `t_min` must actually DECIDE
    /// admission, not merely be computed.
    ///
    /// For each `(ll, m)` whose minimum lands where `m` is the boundary
    /// revolution count, a time just under `t_min` must be rejected by
    /// [`find_xy`] and a time just over it admitted. Without this, the accuracy
    /// pin above could hold while the value it pins reached nothing.
    #[test]
    fn the_t_min_prune_decides_admission_at_the_boundary_revolution() {
        let mut fired = 0_u32;
        let mut admitted = 0_u32;
        for m in 1..=4_i32 {
            for step in 0..=120_i32 {
                let ll = -0.9 + 1.8 * f64::from(step) / 120.0;
                let (_, t_min) = compute_t_min(ll, m, 8, CONVERGENCE_TOL, CONVERGENCE_TOL);
                if !t_min.is_finite() || t_min <= 0.0 {
                    continue;
                }
                let below = t_min * (1.0 - 1e-6);
                let above = t_min * (1.0 + 1e-6);
                // The prune only engages when `m` is the boundary revolution
                // count for that time, which is what `m_max_quick` computes.
                let boundary =
                    |t: f64| f64_to_i32_saturating((t / std::f64::consts::PI).floor()) == m;
                if !boundary(below) || !boundary(above) {
                    continue;
                }
                let (x_below, _) = find_xy(ll, below, m, 8, CONVERGENCE_TOL, CONVERGENCE_TOL, true);
                let (x_above, _) = find_xy(ll, above, m, 8, CONVERGENCE_TOL, CONVERGENCE_TOL, true);
                if x_below.is_nan() {
                    fired += 1;
                }
                if x_above.is_finite() {
                    admitted += 1;
                }
            }
        }
        assert!(
            fired >= 10,
            "the t_min prune never rejected below the minimum (fired={fired}); it is \
             either unreachable or no longer decided by t_min"
        );
        assert!(
            admitted >= 10,
            "nothing was admitted just above the minimum (admitted={admitted}); the \
             prune would be rejecting everything, which this test must also catch"
        );
        println!("t_min prune: {fired} rejections below, {admitted} admissions above");
    }

    #[test]
    fn combined_lowpath_enumerator_matches_separate_enumeration() {
        let state1 = leo_state1();
        let state2 = leo_state2();
        let tof = 43_200.0;
        let mut separate = Vec::new();
        for_each_lambert_m_prograde(
            MU,
            &state1,
            &state2,
            tof,
            3,
            true,
            |m, prograde, dv_depart, dv_arrive, valid| {
                if valid {
                    separate.push((m, true, prograde, dv_depart, dv_arrive));
                }
            },
        );
        for_each_lambert_m_prograde(
            MU,
            &state1,
            &state2,
            tof,
            3,
            false,
            |m, prograde, dv_depart, dv_arrive, valid| {
                if valid && m > 0 {
                    separate.push((m, false, prograde, dv_depart, dv_arrive));
                }
            },
        );

        let mut combined = Vec::new();
        for_each_lambert_m_prograde_lowpaths(
            MU,
            &state1,
            &state2,
            tof,
            3,
            true,
            |m, low_path, prograde, dv_depart, dv_arrive, valid| {
                if valid {
                    combined.push((m, low_path, prograde, dv_depart, dv_arrive));
                }
            },
        );

        // Same (m, low_path, prograde) tuple order and near-equal dv vectors.
        // HF-NEW-01 routes `for_each_lambert_m_prograde_lowpaths` through SIMD4
        // when the `simd` feature is on, so dv components can drift by a few
        // ULPs vs the scalar `for_each_lambert_m_prograde` path due to f64x4
        // FMA reordering. Use a tight absolute tolerance instead of bit-exact
        // equality.
        assert_eq!(
            combined.len(),
            separate.len(),
            "SIMD4 and scalar enumerators must produce the same number of branches"
        );
        for (i, (c, s)) in combined.iter().zip(separate.iter()).enumerate() {
            assert_eq!(c.0, s.0, "branch {i}: m mismatch");
            assert_eq!(c.1, s.1, "branch {i}: low_path mismatch");
            assert_eq!(c.2, s.2, "branch {i}: prograde mismatch");
            for (axis, ((combined_depart, scalar_depart), (combined_arrive, scalar_arrive))) in
                c.3.iter()
                    .zip(s.3.iter())
                    .zip(c.4.iter().zip(s.4.iter()))
                    .enumerate()
            {
                let dv_dep_delta = (combined_depart - scalar_depart).abs();
                let dv_arr_delta = (combined_arrive - scalar_arrive).abs();
                assert!(
                    dv_dep_delta < 1e-12,
                    "branch {i} dv_depart[{axis}]: SIMD={combined_depart} scalar={scalar_depart} delta={dv_dep_delta}"
                );
                assert!(
                    dv_arr_delta < 1e-12,
                    "branch {i} dv_arrive[{axis}]: SIMD={combined_arrive} scalar={scalar_arrive} delta={dv_arr_delta}"
                );
            }
        }
    }

    #[test]
    fn variable_r2_branch_best_batch_matches_scalar_branch_enumerator() {
        let state1 = leo_state1();
        let state2 = leo_state2();
        let r1 = [state1[0], state1[1], state1[2]];
        let v1_ref = [state1[3], state1[4], state1[5]];
        let case_states = [
            state2,
            [1200.0, 7077.0, 250.0, -7.25, 1.0, 0.02],
            [-2400.0, 6800.0, -150.0, -7.0, -1.8, 0.04],
            [3600.0, -6100.0, 420.0, 6.35, 3.25, -0.05],
            [-5050.0, -4650.0, -300.0, 4.8, -5.2, 0.03],
        ];
        let tofs = [43_200.0, 45_000.0, 48_000.0, 51_600.0, 54_000.0];
        let r2_vec: Vec<[f64; 3]> = case_states
            .iter()
            .map(|state| [state[0], state[1], state[2]])
            .collect();
        let v2_refs: Vec<[f64; 3]> = case_states
            .iter()
            .map(|state| [state[3], state[4], state[5]])
            .collect();

        let mut scratch = VariableR2LambertScratch::default();
        let batch = solve_lambert_batch_tof_variable_r2_branch_best_with_scratch(
            MU,
            &r1,
            &r2_vec,
            &v1_ref,
            &v2_refs,
            &tofs,
            1,
            true,
            None,
            &mut scratch,
        );

        assert_eq!(batch.len(), tofs.len());
        for (idx, (((result, tof), state), v2_ref)) in batch
            .iter()
            .zip(tofs.iter().copied())
            .zip(case_states.iter())
            .zip(v2_refs.iter())
            .enumerate()
        {
            let mut expected = BranchBatchTofResult {
                tof,
                ..BranchBatchTofResult::default()
            };
            for_each_lambert_m_prograde_lowpaths(
                MU,
                &state1,
                state,
                tof,
                1,
                true,
                |m, low_path, prograde, dv_depart, dv_arrive, valid| {
                    if !valid {
                        return;
                    }
                    let dv_depart_norm = norm_array(&dv_depart);
                    if dv_depart_norm < expected.dv_depart {
                        expected = BranchBatchTofResult {
                            tof,
                            dv_depart: dv_depart_norm,
                            dv_arrive: norm_array(&dv_arrive),
                            v1: [
                                dv_depart[0] + v1_ref[0],
                                dv_depart[1] + v1_ref[1],
                                dv_depart[2] + v1_ref[2],
                            ],
                            v2: [
                                v2_ref[0] - dv_arrive[0],
                                v2_ref[1] - dv_arrive[1],
                                v2_ref[2] - dv_arrive[2],
                            ],
                            m,
                            low_path,
                            prograde,
                            valid: true,
                        };
                    }
                },
            );

            assert_eq!(result.valid, expected.valid, "row {idx} valid");
            assert_eq!(result.m, expected.m, "row {idx} m");
            assert_eq!(result.low_path, expected.low_path, "row {idx} low_path");
            assert_eq!(result.prograde, expected.prograde, "row {idx} prograde");
            assert!(
                (result.dv_depart - expected.dv_depart).abs() < 1e-12,
                "row {idx} dv_depart batch={} scalar={} batch_branch=({},{},{}) scalar_branch=({},{},{})",
                result.dv_depart,
                expected.dv_depart,
                result.m,
                result.low_path,
                result.prograde,
                expected.m,
                expected.low_path,
                expected.prograde
            );
            assert!(
                (result.dv_arrive - expected.dv_arrive).abs() < 1e-12,
                "row {idx} dv_arrive batch={} scalar={}",
                result.dv_arrive,
                expected.dv_arrive
            );
        }
    }

    #[test]
    fn variable_r2_branch_best_m0_simd_tails_match_scalar_velocities() {
        fn scalar_best(
            r1: &[f64; 3],
            r2: &[f64; 3],
            v1_ref: &[f64; 3],
            v2_ref: &[f64; 3],
            tof: f64,
        ) -> BranchBatchTofResult {
            let mut best = BranchBatchTofResult {
                tof,
                ..BranchBatchTofResult::default()
            };
            for m in 0..=4 {
                for prograde in [true, false] {
                    for low_path in [true, false] {
                        let result =
                            izzo2015_impl(MU, r1, r2, tof, m, prograde, low_path, 8, 1e-9, 1e-9);
                        if !result.success {
                            continue;
                        }
                        let dv_depart = norm_array(&[
                            result.v1[0] - v1_ref[0],
                            result.v1[1] - v1_ref[1],
                            result.v1[2] - v1_ref[2],
                        ]);
                        if dv_depart < best.dv_depart {
                            best = BranchBatchTofResult {
                                tof,
                                dv_depart,
                                dv_arrive: norm_array(&[
                                    v2_ref[0] - result.v2[0],
                                    v2_ref[1] - result.v2[1],
                                    v2_ref[2] - result.v2[2],
                                ]),
                                v1: result.v1,
                                v2: result.v2,
                                m,
                                low_path,
                                prograde,
                                valid: true,
                            };
                        }
                    }
                }
            }
            best
        }

        let state1 = leo_state1();
        let r1 = [state1[0], state1[1], state1[2]];
        let v1_ref = [state1[3], state1[4], state1[5]];
        let case_states = [
            leo_state2(),
            [1200.0, 7077.0, 250.0, -7.25, 1.0, 0.02],
            [-2400.0, 6800.0, -150.0, -7.0, -1.8, 0.04],
            [3600.0, -6100.0, 420.0, 6.35, 3.25, -0.05],
            [-5050.0, -4650.0, -300.0, 4.8, -5.2, 0.03],
            [6999.0, 1.0, 0.0, 0.0, 7.5, 0.0],
            [0.0, 7178.0, 1.0, -7.45, 0.0, 0.01],
            [6800.0, 2200.0, -20.0, -2.2, 6.9, 0.0],
            [0.0; 6],
        ];
        let tofs = [
            600.0, 750.0, 900.0, 1050.0, 1200.0, 1350.0, 1500.0, 1650.0, -1.0,
        ];

        // `if !expected.valid { continue; }` below skips every velocity
        // comparison for a row the scalar oracle could not solve. If the solver
        // ever stopped converging wholesale, `assert_eq!(actual.valid,
        // expected.valid)` would still hold (both false) and this test would
        // pass having compared no velocities at all. `compared` is the backstop.
        let mut compared = 0_usize;

        for row_count in [3usize, 4, 5, 8, 9, 65] {
            let r2_rows = case_states
                .iter()
                .copied()
                .cycle()
                .take(row_count)
                .map(|state| [state[0], state[1], state[2]])
                .collect::<Vec<_>>();
            let v2_rows = case_states
                .iter()
                .copied()
                .cycle()
                .take(row_count)
                .map(|state| [state[3], state[4], state[5]])
                .collect::<Vec<_>>();
            let tof_rows = tofs
                .iter()
                .copied()
                .cycle()
                .take(row_count)
                .collect::<Vec<_>>();
            let mut scratch = VariableR2LambertScratch::default();
            let actual = solve_lambert_batch_tof_variable_r2_branch_best_with_scratch(
                MU,
                &r1,
                &r2_rows,
                &v1_ref,
                &v2_rows,
                &tof_rows,
                4,
                true,
                None,
                &mut scratch,
            );
            for (row, (((actual_row, r2), v2), tof)) in actual
                .iter()
                .zip(r2_rows.iter())
                .zip(v2_rows.iter())
                .zip(tof_rows.iter().copied())
                .enumerate()
            {
                let expected = scalar_best(&r1, r2, &v1_ref, v2, tof);
                assert_eq!(actual_row.valid, expected.valid, "N={row_count} row={row}");
                if !expected.valid {
                    continue;
                }
                assert_eq!(actual_row.tof.to_bits(), expected.tof.to_bits());
                assert_eq!(expected.m, 0, "oracle must force M=0 winner");
                assert_eq!(actual_row.m, expected.m, "N={row_count} row={row} M");
                assert_eq!(
                    actual_row.prograde, expected.prograde,
                    "N={row_count} row={row}"
                );
                assert_eq!(
                    actual_row.low_path, expected.low_path,
                    "N={row_count} row={row}"
                );
                for (axis, ((actual_v1, expected_v1), (actual_v2, expected_v2))) in actual_row
                    .v1
                    .iter()
                    .zip(expected.v1.iter())
                    .zip(actual_row.v2.iter().zip(expected.v2.iter()))
                    .enumerate()
                {
                    assert!(
                        (actual_v1 - expected_v1).abs() < 1e-9,
                        "N={row_count} row={row} v1[{axis}] actual={actual_v1} expected={expected_v1}"
                    );
                    assert!(
                        (actual_v2 - expected_v2).abs() < 1e-9,
                        "N={row_count} row={row} v2[{axis}] actual={actual_v2} expected={expected_v2}"
                    );
                }
                assert!((actual_row.dv_depart - expected.dv_depart).abs() < 1e-9);
                assert!((actual_row.dv_arrive - expected.dv_arrive).abs() < 1e-9);
                compared += 1;
            }
        }

        // Corpus is 3 + 4 + 5 + 8 + 9 + 65 = 94 rows, cycling 9 case states and
        // 9 times of flight. One state (`[0.0; 6]`) and one time of flight
        // (`-1.0`) are deliberately degenerate, so a minority of rows are
        // legitimately invalid. The floor is set from the 94-row corpus, not
        // from the count that happens to survive today.
        assert!(
            compared >= 70,
            "variable-r2 branch-best corpus must compare at least 70 solved rows out of 94, compared {compared}"
        );
    }

    #[test]
    fn variable_r2_branch_best_batch_honors_exact_branch_selection() {
        let state1 = leo_state1();
        let state2 = leo_state2();
        let r1 = [state1[0], state1[1], state1[2]];
        let v1_ref = [state1[3], state1[4], state1[5]];
        let r2_vec = [[state2[0], state2[1], state2[2]]];
        let v2_refs = [[state2[3], state2[4], state2[5]]];
        let tofs = [43_200.0];

        let mut scratch = VariableR2LambertScratch::default();
        let batch = solve_lambert_batch_tof_variable_r2_branch_best_with_scratch(
            MU,
            &r1,
            &r2_vec,
            &v1_ref,
            &v2_refs,
            &tofs,
            1,
            true,
            Some((1, false)),
            &mut scratch,
        );

        assert_eq!(batch.len(), 1);
        let selected = batch.first();
        assert!(selected.is_some_and(|result| result.valid));
        assert_eq!(selected.map(|result| result.m), Some(1));
        assert!(selected.is_some_and(|result| !result.low_path));
    }

    fn pack_filtered_selected_branch_reference(
        r1: &[f64; 3],
        r2_vec: &[[f64; 3]],
        v1_ref: &[f64; 3],
        v2_refs: &[[f64; 3]],
        tofs: &[f64],
        m_max: i32,
        requested_low_path: bool,
        include_retrograde: bool,
        selected: (i32, bool),
    ) -> Vec<BranchBatchTofResult> {
        let state1 = [r1[0], r1[1], r1[2], v1_ref[0], v1_ref[1], v1_ref[2]];
        let r1_cache = LambertR1Cache::new(r1);
        let mut rows = Vec::with_capacity(tofs.len());

        for ((r2, v2_ref), tof) in r2_vec.iter().zip(v2_refs).zip(tofs.iter().copied()) {
            let state2 = state6(r2, v2_ref);
            let mut best = BranchBatchTofResult {
                tof,
                ..BranchBatchTofResult::default()
            };
            // The production SIMD4 enumerator matching the batch arithmetic,
            // explicitly. This is the pinned form of the R17b
            // dual-Lambert closure: the batch rows and the per-candidate
            // enumerator are now ONE arithmetic, and this test is what catches
            // the next time they silently diverge (its ancestor caught the
            // original 2 ULP split on `dv_depart`).
            let r2_pos = [state2[0], state2[1], state2[2]];
            let m_max_feasible = compute_m_max_fast(r1, &r2_pos, tof, MU).min(m_max.max(0));
            let geom = compute_lambert_geometry_with_r1(MU, &r1_cache, &r2_pos, tof);
            let mut visit_scalar = |m: i32,
                                    low_path: bool,
                                    prograde: bool,
                                    dv_depart: [f64; 3],
                                    dv_arrive: [f64; 3],
                                    valid: bool| {
                if !valid || (m, low_path) != selected {
                    return;
                }
                let dv_depart_norm = norm3(&dv_depart);
                if !dv_depart_norm.is_finite() || dv_depart_norm >= best.dv_depart {
                    return;
                }
                best = BranchBatchTofResult {
                    tof,
                    dv_depart: dv_depart_norm,
                    dv_arrive: norm3(&dv_arrive),
                    v1: [
                        dv_depart[0] + v1_ref[0],
                        dv_depart[1] + v1_ref[1],
                        dv_depart[2] + v1_ref[2],
                    ],
                    v2: [
                        v2_ref[0] - dv_arrive[0],
                        v2_ref[1] - dv_arrive[1],
                        v2_ref[2] - dv_arrive[2],
                    ],
                    m,
                    low_path,
                    prograde,
                    valid: true,
                };
            };
            if m_max_feasible >= 0 {
                for_each_lambert_simd_pack_enumeration(
                    &geom,
                    &state1,
                    &state2,
                    m_max_feasible,
                    requested_low_path,
                    include_retrograde,
                    &mut visit_scalar,
                    &mut || {},
                );
            }
            rows.push(best);
        }
        rows
    }

    fn assert_branch_batch_row_bits(
        actual: &BranchBatchTofResult,
        expected: &BranchBatchTofResult,
        context: &str,
    ) {
        assert_eq!(
            actual.tof.to_bits(),
            expected.tof.to_bits(),
            "{context}: tof"
        );
        assert_eq!(
            actual.dv_depart.to_bits(),
            expected.dv_depart.to_bits(),
            "{context}: dv_depart"
        );
        assert_eq!(
            actual.dv_arrive.to_bits(),
            expected.dv_arrive.to_bits(),
            "{context}: dv_arrive"
        );
        assert_eq!(actual.m, expected.m, "{context}: m");
        assert_eq!(actual.low_path, expected.low_path, "{context}: low_path");
        assert_eq!(actual.prograde, expected.prograde, "{context}: prograde");
        assert_eq!(actual.valid, expected.valid, "{context}: valid");
        for (axis, ((actual_v1, expected_v1), (actual_v2, expected_v2))) in actual
            .v1
            .iter()
            .zip(expected.v1.iter())
            .zip(actual.v2.iter().zip(expected.v2.iter()))
            .enumerate()
        {
            assert_eq!(
                actual_v1.to_bits(),
                expected_v1.to_bits(),
                "{context}: v1[{axis}]"
            );
            assert_eq!(
                actual_v2.to_bits(),
                expected_v2.to_bits(),
                "{context}: v2[{axis}]"
            );
        }
    }

    #[test]
    fn variable_r2_selected_branch_batch_matches_pack_filtered_reference_bitwise() {
        let state1 = leo_state1();
        let state2 = leo_state2();
        let r1 = [state1[0], state1[1], state1[2]];
        let v1_ref = [state1[3], state1[4], state1[5]];
        let row_templates: [([f64; 3], [f64; 3], f64); 7] = [
            (
                [state2[0], state2[1], state2[2]],
                [state2[3], state2[4], state2[5]],
                43_200.0,
            ),
            ([1_200.0, 7_077.0, 250.0], [-7.25, 1.0, 0.02], 48_000.0),
            ([-2_400.0, 6_800.0, -150.0], [-7.0, -1.8, 0.04], 54_000.0),
            ([3_600.0, -6_100.0, 420.0], [6.35, 3.25, -0.05], 57_600.0),
            ([-5_050.0, -4_650.0, -300.0], [4.8, -5.2, 0.03], 61_200.0),
            // Near-pi transfer plane, but not exactly collinear.
            ([-6_778.0, 0.01, 0.0], [0.0, -7.5, 0.0], 72_000.0),
            // Invalid geometry and TOF must preserve sentinel bits.
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], -1.0),
        ];

        for row_count in [1_usize, 3, 4, 5, 8, 65] {
            for sorted in [true, false] {
                let mut rows = row_templates
                    .iter()
                    .copied()
                    .cycle()
                    .take(row_count)
                    .collect::<Vec<_>>();
                if sorted {
                    rows.sort_by(|left, right| left.2.total_cmp(&right.2));
                } else {
                    rows.reverse();
                }
                let r2_vec = rows.iter().map(|row| row.0).collect::<Vec<_>>();
                let v2_refs = rows.iter().map(|row| row.1).collect::<Vec<_>>();
                let tofs = rows.iter().map(|row| row.2).collect::<Vec<_>>();

                for requested_low_path in [true, false] {
                    for selected in [
                        (0, true),
                        (0, false),
                        (1, true),
                        (1, false),
                        (4, true),
                        (4, false),
                    ] {
                        for include_retrograde in [true, false] {
                            let mut scratch = VariableR2LambertScratch::default();
                            let actual =
                                solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_scratch(
                                    MU,
                                    &r1,
                                    &r2_vec,
                                    &v1_ref,
                                    &v2_refs,
                                    &tofs,
                                    4,
                                    requested_low_path,
                                    include_retrograde,
                                    Some(selected),
                                    &mut scratch,
                                );
                            let expected = pack_filtered_selected_branch_reference(
                                &r1,
                                &r2_vec,
                                &v1_ref,
                                &v2_refs,
                                &tofs,
                                4,
                                requested_low_path,
                                include_retrograde,
                                selected,
                            );
                            assert_eq!(actual.len(), expected.len());
                            for (row, (actual, expected)) in
                                actual.iter().zip(expected.iter()).enumerate()
                            {
                                assert_branch_batch_row_bits(
                                    actual,
                                    expected,
                                    &format!(
                                        "N={row_count} sorted={sorted} requested_low_path={requested_low_path} \\
                                         selection={selected:?} retrograde={include_retrograde} row={row}"
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn selected_branch_work_fixture() -> ([f64; 3], [f64; 3], Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<f64>)
    {
        let state1 = leo_state1();
        let state2 = leo_state2();
        let r1 = [state1[0], state1[1], state1[2]];
        let v1_ref = [state1[3], state1[4], state1[5]];
        let r2_vec = vec![
            [state2[0], state2[1], state2[2]],
            [1_200.0, 7_077.0, 250.0],
            [-2_400.0, 6_800.0, -150.0],
            [3_600.0, -6_100.0, 420.0],
        ];
        let v2_refs = vec![
            [state2[3], state2[4], state2[5]],
            [-7.25, 1.0, 0.02],
            [-7.0, -1.8, 0.04],
            [6.35, 3.25, -0.05],
        ];
        let tofs = vec![86_400.0, 88_200.0, 90_000.0, 91_800.0];
        (r1, v1_ref, r2_vec, v2_refs, tofs)
    }

    #[test]
    fn selected_branch_batch_does_not_execute_unselected_scalar_variants() {
        let (r1, v1_ref, r2_vec, v2_refs, tofs) = selected_branch_work_fixture();
        for (r2, tof) in r2_vec.iter().zip(&tofs) {
            assert!(
                compute_m_max_fast(&r1, r2, *tof, MU) >= 4,
                "fixture must make M4 feasible"
            );
        }

        let mut scratch = VariableR2LambertScratch::default();
        let _rows = solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_scratch(
            MU,
            &r1,
            &r2_vec,
            &v1_ref,
            &v2_refs,
            &tofs,
            4,
            true,
            true,
            Some((4, true)),
            &mut scratch,
        );
        let telemetry = scratch.branch_telemetry();
        // R18: the selected-branch rows run through the pack kernel, so the
        // per-variant work shows up as SIMD lanes; the bound is unchanged --
        // exactly one prograde and one retrograde variant per TOF, nothing
        // unselected.
        assert_eq!(telemetry.scalar_variant_solves, 0);
        assert_eq!(
            telemetry.simd_lane_solves,
            tofs.len() * 2,
            "exact M4 selection must solve only prograde and retrograde variants"
        );
    }

    #[test]
    fn selected_m0_batch_skips_discarded_simd_prefill() {
        let (r1, v1_ref, r2_vec, v2_refs, tofs) = selected_branch_work_fixture();
        let mut scratch = VariableR2LambertScratch::default();
        let _rows = solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_scratch(
            MU,
            &r1,
            &r2_vec,
            &v1_ref,
            &v2_refs,
            &tofs,
            4,
            true,
            true,
            Some((0, true)),
            &mut scratch,
        );
        let telemetry = scratch.branch_telemetry();
        // R18: the M0-chunk prefill this test guards against still must not
        // run (its lanes would exceed the 2-per-TOF bound below); the
        // selected rows' own pack lanes are the only SIMD work.
        assert_eq!(telemetry.scalar_variant_solves, 0);
        assert_eq!(
            telemetry.simd_lane_solves,
            tofs.len() * 2,
            "selected M0 must not prefill SIMD rows that selection discards"
        );
    }

    #[test]
    fn lambert_geometry_r1_cache_is_bit_identical() {
        let r1 = R1_LEO;
        let cache = LambertR1Cache::new(&r1);
        let r2_cases = [
            [0.0, 7178.0, 0.0],
            [150.0, 7178.0, 30.0],
            [-2400.0, 6800.0, -150.0],
            [3600.0, -6100.0, 420.0],
            // Colinear with r1: degenerate transfer plane (failure path).
            [2.0 * R1_LEO[0], 0.0, 0.0],
        ];
        let tofs = [600.0, 1_500.0, 5_400.0, 43_200.0, -1.0];

        let assert_vec3_bits = |label: &str, a: &Vec3, b: &Vec3| {
            for axis in 0..3 {
                assert_eq!(
                    a[axis].to_bits(),
                    b[axis].to_bits(),
                    "{label}[{axis}] must be bit-identical"
                );
            }
        };

        for r2 in &r2_cases {
            for &tof in &tofs {
                let base = compute_lambert_geometry(MU, &r1, r2, tof);
                let cached = compute_lambert_geometry_with_r1(MU, &cache, r2, tof);
                assert_eq!(base.success, cached.success);
                for (label, a, b) in [
                    ("r1_norm", base.r1_norm, cached.r1_norm),
                    ("r2_norm", base.r2_norm, cached.r2_norm),
                    ("c_norm", base.c_norm, cached.c_norm),
                    ("s", base.s, cached.s),
                    ("s_cubed", base.s_cubed, cached.s_cubed),
                    ("ll_base", base.ll_base, cached.ll_base),
                    ("gamma", base.gamma, cached.gamma),
                    ("rho", base.rho, cached.rho),
                    ("sigma", base.sigma, cached.sigma),
                    ("t_nd", base.t_nd, cached.t_nd),
                ] {
                    assert_eq!(a.to_bits(), b.to_bits(), "{label} must be bit-identical");
                }
                assert_vec3_bits("ir1", &base.ir1, &cached.ir1);
                assert_vec3_bits("ir2", &base.ir2, &cached.ir2);
                assert_vec3_bits("it1_base", &base.it1_base, &cached.it1_base);
                assert_vec3_bits("it2_base", &base.it2_base, &cached.it2_base);
            }
        }
    }

    #[test]
    fn variable_r2_branch_best_retrograde_prune_parity() {
        let state1 = leo_state1();
        let case_states = [
            leo_state2(),
            [1200.0, 7077.0, 250.0, -7.25, 1.0, 0.02],
            [-2400.0, 6800.0, -150.0, -7.0, -1.8, 0.04],
            [3600.0, -6100.0, 420.0, 6.35, 3.25, -0.05],
            [-5050.0, -4650.0, -300.0, 4.8, -5.2, 0.03],
        ];
        // 5 rows = one SIMD chunk of 4 + one scalar-tail row.
        let tofs = [43_200.0, 45_000.0, 48_000.0, 51_600.0, 54_000.0];
        let r2_vec: Vec<[f64; 3]> = case_states
            .iter()
            .map(|state| [state[0], state[1], state[2]])
            .collect();
        let v2_refs: Vec<[f64; 3]> = case_states
            .iter()
            .map(|state| [state[3], state[4], state[5]])
            .collect();

        // Prograde deployer and a reversed (retrograde) deployer, so both
        // "prograde branch wins" and "retrograde branch wins" rows are
        // exercised.
        let deployers = [
            state1,
            [
                state1[0], state1[1], state1[2], -state1[3], -state1[4], -state1[5],
            ],
        ];
        let mut retrograde_win_rows = 0usize;
        for state1_case in &deployers {
            let r1 = [state1_case[0], state1_case[1], state1_case[2]];
            let v1_ref = [state1_case[3], state1_case[4], state1_case[5]];
            let mut full_scratch = VariableR2LambertScratch::default();
            let full = solve_lambert_batch_tof_variable_r2_branch_best_with_scratch(
                MU,
                &r1,
                &r2_vec,
                &v1_ref,
                &v2_refs,
                &tofs,
                1,
                true,
                None,
                &mut full_scratch,
            );
            let mut pruned_scratch = VariableR2LambertScratch::default();
            let pruned = solve_lambert_batch_tof_variable_r2_branch_best_pruned_with_scratch(
                MU,
                &r1,
                &r2_vec,
                &v1_ref,
                &v2_refs,
                &tofs,
                1,
                true,
                false,
                None,
                &mut pruned_scratch,
            );
            assert_eq!(full.len(), pruned.len());
            for (idx, (f, p)) in full.iter().zip(pruned.iter()).enumerate() {
                if p.valid {
                    assert!(p.prograde, "row {idx}: pruned run emitted retrograde");
                }
                if f.valid && f.prograde {
                    // The prune may only remove retrograde work: rows whose
                    // best is prograde must be byte-identical.
                    assert_eq!(f.valid, p.valid, "row {idx} valid");
                    assert_eq!(f.m, p.m, "row {idx} m");
                    assert_eq!(f.low_path, p.low_path, "row {idx} low_path");
                    assert_eq!(f.prograde, p.prograde, "row {idx} prograde");
                    assert_eq!(f.tof.to_bits(), p.tof.to_bits(), "row {idx} tof");
                    assert_eq!(
                        f.dv_depart.to_bits(),
                        p.dv_depart.to_bits(),
                        "row {idx} dv_depart"
                    );
                    assert_eq!(
                        f.dv_arrive.to_bits(),
                        p.dv_arrive.to_bits(),
                        "row {idx} dv_arrive"
                    );
                    for (axis, ((full_v1, pruned_v1), (full_v2, pruned_v2))) in
                        f.v1.iter()
                            .zip(p.v1.iter())
                            .zip(f.v2.iter().zip(p.v2.iter()))
                            .enumerate()
                    {
                        assert_eq!(
                            full_v1.to_bits(),
                            pruned_v1.to_bits(),
                            "row {idx} v1[{axis}]"
                        );
                        assert_eq!(
                            full_v2.to_bits(),
                            pruned_v2.to_bits(),
                            "row {idx} v2[{axis}]"
                        );
                    }
                } else if f.valid {
                    // Full-run best was retrograde: the pruned alternative can
                    // only be a HIGHER-dv prograde branch or invalid (this is
                    // what makes the caller-side dv-cap prune sound).
                    retrograde_win_rows += 1;
                    assert!(
                        !p.valid || p.dv_depart >= f.dv_depart,
                        "row {idx}: pruned dv_depart {} undercut retrograde best {}",
                        p.dv_depart,
                        f.dv_depart
                    );
                }
            }
        }
        assert!(
            retrograde_win_rows > 0,
            "expected the reversed deployer to produce retrograde-best rows"
        );
    }

    #[test]
    fn test_batch_m_prograde_basic() {
        let results = izzo2015_batch_m_prograde(MU, &R1_LEO, &R2_LEO, TOF_LEO, 0, true);

        // Should have at least one valid solution (m=0 prograde or retrograde)
        assert!(!results.is_empty(), "Expected at least one solution");

        // All returned solutions should be valid
        for res in &results {
            assert!(res.valid);
            assert!(res.v1[0].is_finite());
            assert!(res.v1[1].is_finite());
            assert!(res.v1[2].is_finite());
        }
    }

    #[test]
    fn test_batch_m_prograde_multi_rev() {
        // Longer TOF to allow multi-revolution solutions
        let long_tof = 10800.0; // 3 hours
        let results = izzo2015_batch_m_prograde(MU, &R1_LEO, &R2_LEO, long_tof, 2, true);

        // Should have multiple solutions for m=0, possibly m=1
        assert!(!results.is_empty());

        // Check that we get different M values
        assert!(
            results.iter().any(|result| result.m == 0),
            "Should have m=0 solution"
        );
    }

    #[test]
    fn test_batch_dv() {
        let state1 = leo_state1();
        let state2 = leo_state2();

        let results = izzo2015_batch_dv(MU, &state1, &state2, TOF_LEO, 0, true);

        assert!(!results.is_empty());

        for (m, prograde, dv_depart, dv_arrive, valid) in &results {
            if *valid {
                // dV should be reasonable (< 20 km/s; retrograde can be ~15 km/s)
                let dv_dep_norm = norm3(dv_depart);
                let dv_arr_norm = norm3(dv_arrive);
                assert!(dv_dep_norm < 20.0, "Departure dV too large: {dv_dep_norm}");
                assert!(dv_arr_norm < 20.0, "Arrival dV too large: {dv_arr_norm}");
                assert!(*m >= 0);
                let _ = prograde; // Used
            }
        }
    }

    #[test]
    fn izzo2015_transfer_dv_arrival_matches_rendezvous_convention() {
        let state1 = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let state2 = [0.0, 7500.0, 0.0, -7.3, 0.0, 0.0];
        let r1 = [state1[0], state1[1], state1[2]];
        let r2 = [state2[0], state2[1], state2[2]];
        let res = izzo2015_impl(MU, &r1, &r2, 3600.0, 0, true, true, 8, 1e-9, 1e-9);
        assert!(res.success);
        let public_arrival_dv = arrival_rendezvous_dv(&res.v2, &state2);
        let batch = izzo2015_batch_dv(MU, &state1, &state2, 3600.0, 0, true)
            .into_iter()
            .find(|(m, prograde, _, _, valid)| *valid && *m == 0 && *prograde)
            .expect("expected M0 prograde solution");
        for (idx, (actual, batch_component)) in
            public_arrival_dv.iter().zip(batch.3.iter()).enumerate()
        {
            assert!(
                (*actual - batch_component).abs() < 1e-12,
                "arrival dV convention mismatch at {idx}: public={actual} batch={batch_component}"
            );
        }
    }

    /// The fused and unfused velocity reconstructions differ, and this pins by
    /// how much so the gap cannot widen unnoticed.
    ///
    /// TWO CONVENTIONS ARE LIVE ON THE PART A PATH, and that is the point of
    /// this test:
    ///
    ///   * FUSED, one rounding, `simd_lambert::reconstruct_velocities_optimized`
    ///     via `izzo2015_impl` -- reached in production by
    ///     `two_phase_transfer_rs::lambert_backend::lambert_single_shot`, whose
    ///     caller is `compute_lambert_guess` in
    ///     `crates/two_phase_transfer_rs/src/postprocess/distribution.rs`.
    ///   * UNFUSED, two roundings, `izzo2015_impl_with_geom_seeded` -- reached in
    ///     production by `izzo2015_batch_tof_variable_r2` and
    ///     `izzo2015_batch_tof_variable_r2_with_scratch`, both called from
    ///     `evaluate_plan_from_phase_with_lambert_scratch_impl` in
    ///     `crates/two_phase_transfer_rs/src/evaluate.rs`.
    ///
    /// This test compares `izzo2015_impl_with_geom` against
    /// `izzo2015_impl_with_geom_fast` rather than those two entry points,
    /// because that pair ISOLATES the convention: identical geometry, identical
    /// `find_xy`, the reconstruction the only difference. The seeded production
    /// route additionally warm-starts `x` from the adjacent TOF, so a
    /// comparison through it mixes two causes.
    ///
    /// Measured 2026-08-21 on this corpus: 8,653 converged comparisons, 6,055
    /// of them (70.0%) differing in bits, largest gap 2.35e-13 relative and
    /// 3.55e-15 km/s absolute. Those two maxima are different rows -- the
    /// relative one lands where a velocity component is near zero -- so do not
    /// divide one by the other.
    ///
    /// The bound below is deliberately ABOVE that. Reconciling the two
    /// conventions moves bits on a sealed path and is a science decision, not a
    /// cleanup -- this test does not force it, it only refuses to let the
    /// disagreement grow. If a future change does unify them the gap goes to
    /// zero, which still passes; revisit this test then rather than leaving it
    /// asserting a discrepancy that no longer exists.
    ///
    /// It replaces a `< 1e-9` ABSOLUTE guard on velocities of order 7 km/s. On
    /// this isolated pair the largest absolute difference is 3.55e-15, so that
    /// guard carried about 280,000x of slack. On the production seeded pair
    /// (`izzo2015_batch_tof_variable_r2` against `izzo2015_impl`, same branch)
    /// the gap is larger, 2.0e-13 absolute, because the warm start moves the
    /// Householder iterate as well -- about 5,000x of slack there. Neither
    /// guard could observe what it was written to watch.
    #[test]
    fn fused_and_unfused_reconstruction_stay_within_their_measured_gap() {
        /// About 42x above the measured 2.35e-13, leaving headroom for a
        /// different libm while staying far tighter than the guard it replaces.
        const RECONSTRUCTION_GAP_RELATIVE_BOUND: f64 = 1e-11;
        /// Chosen from the corpus size (4,000 geometries x 4 branches), not
        /// from what currently converges: an `iter().all(..)` over an empty
        /// collection is vacuously true, so the floor is what makes this a
        /// measurement.
        const MIN_COMPARED: usize = 4_000;

        let mut seed = 0x2026_0821_u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = u32::try_from(seed >> 32).unwrap_or(0);
            f64::from(bits) / f64::from(u32::MAX)
        };

        let mut compared = 0_usize;
        let mut differing = 0_usize;
        let mut max_relative = 0.0_f64;
        let mut max_absolute = 0.0_f64;

        for _ in 0..4_000 {
            let r1_norm = 6_700.0 + next() * 1_800.0;
            let r2_norm = 6_700.0 + next() * 1_800.0;
            let anomaly1 = next() * std::f64::consts::TAU;
            let anomaly2 = next() * std::f64::consts::TAU;
            let inclination = next() * 1.4;
            let r1 = [
                r1_norm * anomaly1.cos(),
                r1_norm * anomaly1.sin() * inclination.cos(),
                r1_norm * anomaly1.sin() * inclination.sin(),
            ];
            let r2 = [
                r2_norm * anomaly2.cos(),
                r2_norm * anomaly2.sin(),
                r2_norm * anomaly2.sin() * 0.1,
            ];
            let tof = 600.0 + next() * 6_000.0;
            let geom = compute_lambert_geometry(MU, &r1, &r2, tof);

            for m in 0..=1 {
                for prograde in [true, false] {
                    let unfused = izzo2015_impl_with_geom(&geom, m, prograde, true, 25, 1e-9, 1e-9);
                    let fused =
                        izzo2015_impl_with_geom_fast(&geom, m, prograde, true, 25, 1e-9, 1e-9);

                    assert_eq!(
                        unfused.success, fused.success,
                        "the two reconstructions must agree on CONVERGENCE for m={m}, \
                         prograde={prograde}; only the last bits of a converged \
                         velocity are allowed to differ"
                    );
                    if !unfused.success {
                        continue;
                    }
                    compared += 1;

                    let mut row_differs = false;
                    for (left, right) in unfused
                        .v1
                        .iter()
                        .chain(unfused.v2.iter())
                        .zip(fused.v1.iter().chain(fused.v2.iter()))
                    {
                        if left.to_bits() == right.to_bits() {
                            continue;
                        }
                        row_differs = true;
                        let absolute = (left - right).abs();
                        if absolute > max_absolute {
                            max_absolute = absolute;
                        }
                        let relative = ((left - right) / right).abs();
                        if relative > max_relative {
                            max_relative = relative;
                        }
                    }
                    if row_differs {
                        differing += 1;
                    }
                }
            }
        }

        // Emit the measurement on a GREEN run, not only in the failure text.
        // These four numbers are quoted in docs/REFACTOR_BLOCKLIST.md, and until
        // this line existed the only way to obtain them was to hand-edit the
        // bound below to zero and read the panic -- so a doc cited figures that
        // no committed command reproduced. `differing` in particular had no
        // consumer at all: it was computed, quoted as a percentage, and never
        // printed or asserted.
        //
        //   cargo test -p two_phase_transfer_rs --lib \
        //     lambert::tests::fused_and_unfused_reconstruction_stay_within_their_measured_gap \
        //     -- --exact --nocapture
        //
        // The full module path is load-bearing: without it the filter matches
        // nothing and cargo still exits 0, reporting `0 passed` and every other
        // test filtered out. Read the passed count, not the exit code.
        //
        // Counts only, deliberately no ratio: a percentage needs an
        // integer-to-float cast, and both `cast_precision_loss` and
        // `as_conversions` are denied here. The division is the reader's.
        println!(
            "RECONSTRUCTION_GAP compared={compared} differing={differing} \
             max_relative={max_relative:e} max_absolute_km_per_s={max_absolute:e}"
        );

        assert!(
            compared >= MIN_COMPARED,
            "only {compared} of the corpus converged, below the {MIN_COMPARED} floor -- \
             this test compares nothing at that rate and its bound proves nothing"
        );
        assert!(
            max_relative <= RECONSTRUCTION_GAP_RELATIVE_BOUND,
            "fused/unfused reconstruction gap grew to {max_relative:e} relative \
             ({max_absolute:e} absolute km/s) over {compared} comparisons \
             ({differing} differing), past the pinned \
             {RECONSTRUCTION_GAP_RELATIVE_BOUND:e}. Either a reconstruction changed or a \
             third convention appeared. Do not widen this bound to make it pass."
        );
    }

    #[test]
    fn test_fast_geom_solver_matches_scalar_geom_solver() {
        let state1 = leo_state1();
        let state2 = leo_state2();
        let r1 = [state1[0], state1[1], state1[2]];
        let r2 = [state2[0], state2[1], state2[2]];
        let geom = compute_lambert_geometry(MU, &r1, &r2, TOF_LEO);

        for m in 0..=1 {
            for prograde in [true, false] {
                let scalar = izzo2015_impl_with_geom(&geom, m, prograde, true, 8, 1e-9, 1e-9);
                let fast = izzo2015_impl_with_geom_fast(&geom, m, prograde, true, 8, 1e-9, 1e-9);

                // Measured 2026-08-06: m=0 converges for both prograde and
                // retrograde on this fixture and m=1 converges for neither --
                // TOF_LEO is 3600 s, far short of a one-revolution transfer
                // between these radii. Pinning that split is what stops the
                // comparison below from being vacuous. It used to sit inside a
                // bare `if scalar.success`, so a rewrite that made the solver
                // fail everywhere would have satisfied `scalar.success ==
                // fast.success` trivially and compared not one velocity.
                let expect_success = m == 0;
                assert_eq!(
                    scalar.success, expect_success,
                    "scalar geom solver feasibility changed for m={m}, prograde={prograde}: \
                     expected success={expect_success}. If m=1 has become reachable at TOF_LEO, \
                     widen this expectation -- do not delete it, it is the non-vacuity proof."
                );
                assert_eq!(
                    scalar.success, fast.success,
                    "fast and scalar geom solvers disagree on convergence for m={m}, prograde={prograde}"
                );
                if expect_success {
                    // This 1e-9 is a SANITY bound, not a parity proof: the two
                    // routes use different reconstruction conventions (see
                    // `fused_and_unfused_reconstruction_stay_within_their_measured_gap`,
                    // which pins the real gap at 2.35e-13 relative). It catches a
                    // gross divergence only. Do not read a pass here as "the two
                    // agree".
                    for (i, ((scalar_v1, fast_v1), (scalar_v2, fast_v2))) in scalar
                        .v1
                        .iter()
                        .zip(fast.v1.iter())
                        .zip(scalar.v2.iter().zip(fast.v2.iter()))
                        .enumerate()
                    {
                        assert!(
                            (scalar_v1 - fast_v1).abs() < 1e-9,
                            "v1 mismatch for m={m}, prograde={prograde}, idx={i}: scalar={scalar_v1} fast={fast_v1}"
                        );
                        assert!(
                            (scalar_v2 - fast_v2).abs() < 1e-9,
                            "v2 mismatch for m={m}, prograde={prograde}, idx={i}: scalar={scalar_v2} fast={fast_v2}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_best_solution() {
        let state1 = leo_state1();
        let state2 = leo_state2();

        let (m, prograde, dv_depart, dv_arrive, dv_norm, valid) =
            izzo2015_best_solution(MU, &state1, &state2, TOF_LEO, 0, true);

        assert!(valid, "Should find a valid solution");
        assert_eq!(m, 0, "Best solution should be m=0 for short TOF");
        assert!(dv_norm.is_finite());
        assert!(dv_norm > 0.0);
        assert!(dv_norm < 10.0, "dV should be < 10 km/s");

        // Verify dv_norm matches computed norm
        let computed_norm = norm3(&dv_depart);
        assert!((dv_norm - computed_norm).abs() < 1e-10);

        // dv_arrive should also be finite
        assert!(norm3(&dv_arrive).is_finite());

        let _ = prograde; // Used
    }

    #[test]
    fn test_best_solution_finds_minimum() {
        let state1 = leo_state1();
        let state2 = leo_state2();

        // Get all solutions
        let all_solutions = izzo2015_batch_dv(MU, &state1, &state2, TOF_LEO, 1, true);

        // Get best solution
        let (_, _, _, _, best_dv_norm, best_valid) =
            izzo2015_best_solution(MU, &state1, &state2, TOF_LEO, 1, true);

        assert!(
            best_valid,
            "izzo2015_best_solution must converge on the LEO fixture -- the minimality \
             comparison below is the only thing this test does"
        );
        assert!(
            !all_solutions.is_empty(),
            "izzo2015_batch_dv must enumerate at least one branch on the LEO fixture"
        );

        // Every returned row is valid by construction: izzo2015_batch_dv
        // pre-filters with compute_m_max_fast and enumerates only revolution
        // counts the geometry admits (measured 2026-08-06: 2 rows, m=0
        // prograde and retrograde, both valid). Asserting it keeps the
        // minimality check unconditional. It used to sit behind three stacked
        // guards -- `best_valid`, `!is_empty()` and a per-row `if *valid` --
        // any of which could go false without failing the test, so a batch
        // solver that returned nothing valid passed this "finds minimum" test.
        for (idx, (_, _, dv_depart, _, valid)) in all_solutions.iter().enumerate() {
            assert!(
                *valid,
                "batch_dv row {idx} is invalid; the minimality comparison would silently \
                 skip it. Either the pre-filter or the solver regressed."
            );
            let dv_norm = norm3(dv_depart);
            assert!(
                best_dv_norm <= dv_norm + 1e-10,
                "Best solution {best_dv_norm} should be <= {dv_norm}"
            );
        }
    }

    #[test]
    fn test_lambert_early_exit_default_disabled() {
        // Tripwire, not a computation. Enabling the early exit makes
        // M-enumeration stop before it has seen every revolution count, which
        // silently changes which transfer is returned. If someone raises this,
        // they should have to delete a test that says so.
        assert_eq!(LAMBERT_EARLY_EXIT_THRESHOLD.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn test_best_solution_parallel_matches_nested_sequential_multi_rev() {
        let state1 = leo_state1();
        let state2 = leo_state2();
        let tof = 86_400.0;
        let m_max = 12;

        let parallel = izzo2015_best_solution(MU, &state1, &state2, tof, m_max, true);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("test pool");
        let sequential = pool.install(|| {
            assert!(rayon::current_thread_index().is_some());
            izzo2015_best_solution(MU, &state1, &state2, tof, m_max, true)
        });

        assert_eq!(parallel, sequential);
    }

    #[test]
    fn test_batch_tof_basic() {
        let state1 = leo_state1();
        let state2 = leo_state2();
        let v1_ref = [state1[3], state1[4], state1[5]];
        let v2_ref = [state2[3], state2[4], state2[5]];

        let tofs = [3000.0, 4000.0, 5000.0, 6000.0];

        // Batch solve
        let batch_results =
            izzo2015_batch_tof(MU, &R1_LEO, &R2_LEO, &tofs, 1, Some(&v1_ref), Some(&v2_ref));

        assert_eq!(batch_results.len(), tofs.len());

        // Check that at least some solutions are valid
        let valid_count = batch_results.iter().filter(|r| r.valid).count();
        assert!(valid_count > 0, "Expected at least one valid solution");

        // All valid solutions should have finite velocities and delta-Vs
        for res in &batch_results {
            if res.valid {
                assert!(res.v1[0].is_finite());
                assert!(res.v1[1].is_finite());
                assert!(res.v1[2].is_finite());
                assert!(res.v2[0].is_finite());
                assert!(res.v2[1].is_finite());
                assert!(res.v2[2].is_finite());
                assert!(res.dv_depart.is_finite());
                assert!(res.dv_arrive.is_finite());
                assert!(res.dv_depart >= 0.0);
                assert!(res.dv_arrive >= 0.0);
            }
        }
    }

    #[test]
    fn test_batch_tof_matches_individual() {
        let state1 = leo_state1();
        let state2 = leo_state2();
        let v1_ref = [state1[3], state1[4], state1[5]];
        let v2_ref = [state2[3], state2[4], state2[5]];

        let tofs = [3600.0, 5000.0, 7200.0];

        // Batch solve
        let batch_results =
            izzo2015_batch_tof(MU, &R1_LEO, &R2_LEO, &tofs, 1, Some(&v1_ref), Some(&v2_ref));

        // Compare with individual best solutions
        for (&tof, batch_result) in tofs.iter().zip(batch_results.iter()) {
            // Element 5 is already `norm3` of the departure dV vector, so the
            // vector itself adds nothing here.
            let (_, _, _, dv_arrive_ind, dv_norm_ind, valid_ind) =
                izzo2015_best_solution(MU, &state1, &state2, tof, 1, true);

            // The parity assertions below used to sit inside
            // `if valid_ind && batch_result.valid`, so a batch solver that
            // returned `valid == false` on every row passed this test without
            // comparing a single number. Both flags are hoisted: the fixture
            // is required to converge, and the two paths are required to agree
            // on that, before any value is compared.
            assert!(
                valid_ind,
                "individual solve must converge at TOF {tof} -- without it this test \
                 compares nothing"
            );
            assert_eq!(
                valid_ind, batch_result.valid,
                "TOF {tof}: batch and individual disagree on validity"
            );

            let dv_norm_batch = batch_result.dv_depart;

            // Batch should find the same or better solution
            assert!(
                (dv_norm_batch - dv_norm_ind).abs() < 0.01,
                "TOF {tof}: batch dv={dv_norm_batch:.6} should match individual dv={dv_norm_ind:.6}"
            );

            // Verify delta-V magnitudes match
            let dv_arr_batch = batch_result.dv_arrive;
            let dv_arr_ind = norm3(&dv_arrive_ind);
            assert!(
                (dv_arr_batch - dv_arr_ind).abs() < 0.01,
                "Arrival dV should match"
            );
        }
    }

    #[test]
    fn test_batch_tof_without_reference_velocities() {
        let tofs = [3000.0, 4000.0, 5000.0];

        // Batch solve without reference velocities
        let batch_results = izzo2015_batch_tof(MU, &R1_LEO, &R2_LEO, &tofs, 0, None, None);

        assert_eq!(batch_results.len(), tofs.len());

        // Without reference velocities, dv should be 0 but solutions should still be valid.
        // The `valid` flag is asserted rather than used as a guard: these four checks used
        // to sit inside `if res.valid`, which is exactly the shape that lets a batch solver
        // returning all-invalid rows pass without executing one of them.
        for (idx, res) in batch_results.iter().enumerate() {
            assert!(
                res.valid,
                "row {idx} must converge without reference velocities -- the zero-dV checks \
                 below only mean something on a valid row"
            );
            assert_eq!(res.dv_depart.to_bits(), 0.0_f64.to_bits());
            assert_eq!(res.dv_arrive.to_bits(), 0.0_f64.to_bits());
            // But velocities should still be computed
            assert!(res.v1[0].is_finite());
            assert!(res.v2[0].is_finite());
        }
    }

    #[test]
    fn test_batch_tof_empty_input() {
        let empty_tofs: Vec<f64> = vec![];
        let batch_results = izzo2015_batch_tof(MU, &R1_LEO, &R2_LEO, &empty_tofs, 0, None, None);
        assert!(batch_results.is_empty());
    }

    #[test]
    fn test_batch_tof_invalid_geometry() {
        let bad_r = [0.0, 0.0, 0.0];
        let tofs = [3000.0, 4000.0];

        let batch_results = izzo2015_batch_tof(MU, &bad_r, &R2_LEO, &tofs, 0, None, None);

        assert_eq!(batch_results.len(), tofs.len());
        // All should be invalid
        for res in &batch_results {
            assert!(!res.valid);
            assert_eq!(res.dv_depart.to_bits(), f64::INFINITY.to_bits());
        }
    }

    #[test]
    fn test_batch_tof_geometry_computed_once() {
        // This test verifies that we get correct results across different TOFs
        // The key optimization is that geometry is computed only once
        let state1 = leo_state1();
        let state2 = leo_state2();
        let v1_ref = [state1[3], state1[4], state1[5]];
        let v2_ref = [state2[3], state2[4], state2[5]];

        // Wide range of TOFs
        let tofs = [2000.0, 4000.0, 6000.0, 8000.0, 10000.0];

        let batch_results =
            izzo2015_batch_tof(MU, &R1_LEO, &R2_LEO, &tofs, 2, Some(&v1_ref), Some(&v2_ref));

        // Verify each TOF value is correctly stored
        for (batch_result, &tof) in batch_results.iter().zip(tofs.iter()) {
            assert_eq!(batch_result.tof.to_bits(), tof.to_bits());
        }

        // Verify solutions are reasonable
        let valid_count = batch_results.iter().filter(|r| r.valid).count();
        assert!(
            valid_count >= tofs.len() / 2,
            "Expected at least half the solutions to be valid"
        );
    }

    #[test]
    fn test_batch_tof_scratch_matches_allocating_api() {
        let state1 = leo_state1();
        let state2 = leo_state2();
        let v1_ref = [state1[3], state1[4], state1[5]];
        let v2_ref = [state2[3], state2[4], state2[5]];
        let tofs = [5400.0, 3200.0, 4300.0, 6100.0, 3900.0];

        let expected =
            izzo2015_batch_tof(MU, &R1_LEO, &R2_LEO, &tofs, 2, Some(&v1_ref), Some(&v2_ref));
        let mut scratch = LambertBatchScratch::default();
        let actual = izzo2015_batch_tof_with_scratch(
            MU,
            &R1_LEO,
            &R2_LEO,
            &tofs,
            2,
            Some(&v1_ref),
            Some(&v2_ref),
            &mut scratch,
        );

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert_eq!(left.tof.to_bits(), right.tof.to_bits());
            assert_eq!(left.valid, right.valid);
            assert_eq!(left.m, right.m);
            assert_eq!(left.prograde, right.prograde);
            assert_eq!(left.v1.map(f64::to_bits), right.v1.map(f64::to_bits));
            assert_eq!(left.v2.map(f64::to_bits), right.v2.map(f64::to_bits));
            assert_eq!(left.dv_depart.to_bits(), right.dv_depart.to_bits());
            assert_eq!(left.dv_arrive.to_bits(), right.dv_arrive.to_bits());
        }
    }

    #[test]
    fn test_solve_lambert_batch_dv_scratch_preserves_order_and_reuses_buffers() {
        let v1_ref = [0.0, (MU / 6778.0_f64).sqrt(), 0.0];
        let tofs = [5000.0, 3000.0, 7000.0, 4000.0, 6000.0];
        let mut scratch = LambertBatchScratch::default();

        let first = izzo2015_batch_dv_seeded_with_scratch(
            MU,
            &R1_LEO,
            &R2_LEO,
            &v1_ref,
            &tofs,
            0,
            true,
            &mut scratch,
        )
        .to_vec();
        let index_capacity = scratch.indexed_tofs.capacity();
        let result_capacity = scratch.dv_results.capacity();

        let shorter_tofs = [4500.0, 3500.0];
        let second = izzo2015_batch_dv_seeded_with_scratch(
            MU,
            &R1_LEO,
            &R2_LEO,
            &v1_ref,
            &shorter_tofs,
            0,
            true,
            &mut scratch,
        )
        .to_vec();

        assert_eq!(first.len(), tofs.len());
        assert_eq!(second.len(), shorter_tofs.len());
        assert!(scratch.indexed_tofs.capacity() >= index_capacity);
        assert!(scratch.dv_results.capacity() >= result_capacity);
    }

    #[test]
    fn test_batch_tof_variable_r2_basic() {
        // Test with different r2 positions for each TOF (simulating target motion)
        let r1 = R1_LEO;
        let v1_ref = [0.0, (MU / 6778.0_f64).sqrt(), 0.0];

        // Simulate target at slightly different positions (as if moving in orbit)
        let r2_vec = [
            [0.0, 7178.0, 0.0],    // Original target position
            [500.0, 7178.0, 0.0],  // Target moved slightly
            [-500.0, 7178.0, 0.0], // Target on other side
        ];
        let v2_refs = [
            [-(MU / 7178.0_f64).sqrt(), 0.0, 0.0],
            [-(MU / 7178.0_f64).sqrt() * 0.99, 0.05, 0.0],
            [-(MU / 7178.0_f64).sqrt() * 1.01, -0.05, 0.0],
        ];
        let tofs = [3600.0, 3700.0, 3500.0];

        let results = izzo2015_batch_tof_variable_r2(MU, &r1, &r2_vec, &v1_ref, &v2_refs, &tofs, 1);

        assert_eq!(results.len(), 3, "Should return one result per TOF");

        // Each result should correspond to its specific r2. `valid` is asserted, not
        // used as a guard: every quality check in this test used to hang off
        // `if res.valid` / `if first.valid && second.valid`, so an all-invalid batch
        // executed none of them and the test still reported green.
        for (idx, (res, &tof)) in results.iter().zip(tofs.iter()).enumerate() {
            assert_eq!(res.tof.to_bits(), tof.to_bits(), "TOF should match input");
            assert!(
                res.valid,
                "row {idx} (TOF {tof}) must converge -- all three r2 positions are \
                 feasible one-hour transfers"
            );
            assert!(res.dv_depart.is_finite());
            assert!(res.dv_arrive.is_finite());
            // dV should be reasonable
            assert!(res.dv_depart < 20.0, "Departure dV should be reasonable");
        }

        // Verify that different r2 positions produce different solutions
        let [first, second, ..] = results.as_slice() else {
            panic!(
                "variable-r2 batch must return one row per TOF, got {}",
                results.len()
            )
        };
        // Solutions should differ since r2 positions differ. Both rows are already
        // asserted valid above, so this comparison is unconditional.
        let diff: f64 = first
            .v1
            .iter()
            .zip(second.v1.iter())
            .map(|(left, right)| (left - right).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "Different r2 positions should produce different solutions"
        );
    }

    #[test]
    fn test_batch_tof_variable_r2_matches_individual() {
        // Verify that variable_r2 matches individual izzo2015_impl calls.
        //
        // n = 5 on purpose: rows sorted into the first chunk of 4 take the
        // SIMD kernel, the 5th takes the scalar tail. The 600 s row is
        // sub-parabolic (parabolic TOF for this geometry is ~906 s), so its
        // converged x is > 1 (hyperbolic). A SIMD TOF kernel that drops the
        // asinh limb of compute_psi, or takes sqrt of the SIGNED 1 - x^2,
        // returns NaN for that lane and the batch reports the row invalid
        // while the individual solve converges — this test is the guard.
        let r1 = R1_LEO;
        let v1_ref = [0.0, (MU / 6778.0_f64).sqrt(), 0.0];

        let r2_vec = [
            [0.0, 7178.0, 0.0],
            [0.0, 7178.0, 0.0],
            [100.0, 7178.0, 50.0],
            [-50.0, 7178.0, 25.0],
            [200.0, 7178.0, -40.0],
        ];
        let v2_refs = [
            [-(MU / 7178.0_f64).sqrt(), 0.0, 0.0],
            [-(MU / 7178.0_f64).sqrt(), 0.0, 0.0],
            [-(MU / 7178.0_f64).sqrt(), 0.01, 0.0],
            [-(MU / 7178.0_f64).sqrt(), -0.01, 0.0],
            [-(MU / 7178.0_f64).sqrt(), 0.02, 0.0],
        ];
        let tofs = [600.0, 3600.0, 4000.0, 4400.0, 4800.0];

        let batch_results =
            izzo2015_batch_tof_variable_r2(MU, &r1, &r2_vec, &v1_ref, &v2_refs, &tofs, 0);

        // Compare with individual calls
        for (((&tof, r2), batch_result), _) in tofs
            .iter()
            .zip(r2_vec.iter())
            .zip(batch_results.iter())
            .zip(v2_refs.iter())
        {
            let individual = izzo2015_impl(MU, &r1, r2, tof, 0, true, true, 25, 1e-9, 1e-9);

            // Hoisted out of `if individual.success && batch_result.valid`. With the
            // flags used as a guard, a variable-r2 batch that reported every row
            // invalid compared no velocity at all and this parity test passed.
            assert!(
                individual.success,
                "individual solve must converge at TOF {tof} -- without it this test \
                 compares nothing"
            );
            assert_eq!(
                individual.success, batch_result.valid,
                "TOF {tof}: variable-r2 batch and individual disagree on validity"
            );

            // "Closely", not exactly, and deliberately so. The variable-r2 batch
            // reaches `izzo2015_impl_with_geom_seeded`, whose reconstruction is
            // NOT fused, while `izzo2015_impl` goes through
            // `simd_lambert::reconstruct_velocities_optimized`, which is. Both
            // are live on the Part A path -- the batch from
            // `evaluate_plan_from_phase_with_lambert_scratch_impl` in
            // `crates/two_phase_transfer_rs/src/evaluate.rs`, the single shot
            // from `compute_lambert_guess` in
            // `crates/two_phase_transfer_rs/src/postprocess/distribution.rs`.
            // The seeded
            // route also warm-starts `x` from the adjacent TOF, so the observed
            // gap (2.0e-13 absolute, measured 2026-08-21) has two causes and is
            // larger than the reconstruction gap alone. This 1e-9 has ~5,000x of
            // slack against that and cannot see it; the magnitude is pinned by
            // `fused_and_unfused_reconstruction_stay_within_their_measured_gap`,
            // which isolates the reconstruction. Tightening this one to a value
            // that fails would be wrong -- the difference is real and its
            // reconciliation is an open science decision.
            assert!(
                (individual.v1[0] - batch_result.v1[0]).abs() < 1e-9,
                "v1[0] should match for TOF {tof}"
            );
            assert!(
                (individual.v1[1] - batch_result.v1[1]).abs() < 1e-9,
                "v1[1] should match for TOF {tof}"
            );
        }
    }

    #[test]
    fn test_batch_tof_variable_r2_matches_raw_departure_best() {
        let r1 = R1_LEO;
        let v1_ref = [0.0, (MU / 6778.0_f64).sqrt(), 0.0];
        let r2_vec = [
            [0.0, 7178.0, 0.0],
            [250.0, 7178.0, 20.0],
            [-125.0, 7190.0, -35.0],
        ];
        let v2_refs = [
            [-(MU / 7178.0_f64).sqrt(), 0.0, 0.0],
            [-(MU / 7178.0_f64).sqrt() * 0.99, 0.04, 0.0],
            [-(MU / 7190.0_f64).sqrt() * 1.01, -0.02, 0.01],
        ];
        let tofs = [3200.0, 4300.0, 5400.0];

        let batch_results =
            izzo2015_batch_tof_variable_r2(MU, &r1, &r2_vec, &v1_ref, &v2_refs, &tofs, 2);

        for (idx, (((batch, r2), v2_ref), tof)) in batch_results
            .iter()
            .zip(r2_vec.iter())
            .zip(v2_refs.iter())
            .zip(tofs.iter().copied())
            .enumerate()
        {
            let state1 = [r1[0], r1[1], r1[2], v1_ref[0], v1_ref[1], v1_ref[2]];
            let state2 = state6(r2, v2_ref);
            let raw = izzo2015_best_solution(MU, &state1, &state2, tof, 2, true);

            assert_eq!(batch.valid, raw.5, "validity mismatch at {idx}");
            if batch.valid {
                let raw_v1 = [
                    state1[3] + raw.2[0],
                    state1[4] + raw.2[1],
                    state1[5] + raw.2[2],
                ];
                let raw_v2 = [
                    state2[3] - raw.3[0],
                    state2[4] - raw.3[1],
                    state2[5] - raw.3[2],
                ];
                assert!(
                    (batch.dv_depart - raw.4).abs() < 1e-9,
                    "departure dV mismatch at {idx}: batch={} raw={}",
                    batch.dv_depart,
                    raw.4
                );
                for (component, ((batch_v1, raw_v1), (batch_v2, raw_v2))) in batch
                    .v1
                    .iter()
                    .zip(raw_v1.iter())
                    .zip(batch.v2.iter().zip(raw_v2.iter()))
                    .enumerate()
                {
                    assert!(
                        (batch_v1 - raw_v1).abs() < 1e-12,
                        "departure velocity component {component} mismatch at {idx}: batch={batch_v1} raw={raw_v1}"
                    );
                    assert!(
                        (batch_v2 - raw_v2).abs() < 1e-12,
                        "arrival velocity component {component} mismatch at {idx}: batch={batch_v2} raw={raw_v2}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_batch_tof_variable_r2_scratch_matches_allocating_api() {
        let r1 = R1_LEO;
        let v1_ref = [0.0, (MU / 6778.0_f64).sqrt(), 0.0];
        let r2_vec = [
            [0.0, 7178.0, 0.0],
            [250.0, 7178.0, 20.0],
            [-125.0, 7190.0, -35.0],
            [80.0, 7200.0, 12.0],
            [-240.0, 7185.0, 4.0],
        ];
        let v2_refs = [
            [-(MU / 7178.0_f64).sqrt(), 0.0, 0.0],
            [-(MU / 7178.0_f64).sqrt() * 0.99, 0.04, 0.0],
            [-(MU / 7190.0_f64).sqrt() * 1.01, -0.02, 0.01],
            [-(MU / 7200.0_f64).sqrt(), 0.01, -0.01],
            [-(MU / 7185.0_f64).sqrt(), -0.03, 0.02],
        ];
        let tofs = [5400.0, 3200.0, 4300.0, 6100.0, 3900.0];
        let expected =
            izzo2015_batch_tof_variable_r2(MU, &r1, &r2_vec, &v1_ref, &v2_refs, &tofs, 2);
        let mut scratch = VariableR2LambertScratch::default();
        let actual = izzo2015_batch_tof_variable_r2_with_scratch(
            MU,
            &r1,
            &r2_vec,
            &v1_ref,
            &v2_refs,
            &tofs,
            2,
            &mut scratch,
        );

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert_eq!(left.tof.to_bits(), right.tof.to_bits());
            assert_eq!(left.valid, right.valid);
            assert_eq!(left.m, right.m);
            assert_eq!(left.prograde, right.prograde);
            assert_eq!(left.v1.map(f64::to_bits), right.v1.map(f64::to_bits));
            assert_eq!(left.v2.map(f64::to_bits), right.v2.map(f64::to_bits));
            assert_eq!(left.dv_depart.to_bits(), right.dv_depart.to_bits());
            assert_eq!(left.dv_arrive.to_bits(), right.dv_arrive.to_bits());
        }
    }

    #[test]
    fn test_batch_tof_variable_r2_scalar_tail_updates_seed_state() {
        let r1 = R1_LEO;
        let v1_ref = [0.0, (MU / 6778.0_f64).sqrt(), 0.0];
        let r2_vec = [
            [0.0, 7178.0, 0.0],
            [100.0, 7178.0, 25.0],
            [200.0, 7180.0, -30.0],
        ];
        let v2_refs = [
            [-(MU / 7178.0_f64).sqrt(), 0.0, 0.0],
            [-(MU / 7178.0_f64).sqrt(), 0.01, 0.0],
            [-(MU / 7180.0_f64).sqrt(), -0.01, 0.0],
        ];
        let tofs = [3600.0, 3900.0, 4200.0];
        let mut scratch = VariableR2LambertScratch::default();

        let results = izzo2015_batch_tof_variable_r2_with_scratch(
            MU,
            &r1,
            &r2_vec,
            &v1_ref,
            &v2_refs,
            &tofs,
            0,
            &mut scratch,
        );

        assert_eq!(results.len(), tofs.len());
        assert!(results.iter().any(|result| result.valid));
        assert!(
            scratch.last_x_seeds.iter().any(Option::is_some),
            "scalar-only variable-r2 batches should seed subsequent tail solves"
        );
    }

    #[test]
    fn test_batch_tof_variable_r2_empty_and_mismatched() {
        let r1 = R1_LEO;
        let v1_ref = [0.0, 7.5, 0.0];

        // Empty inputs should return empty
        let empty_results = izzo2015_batch_tof_variable_r2(MU, &r1, &[], &v1_ref, &[], &[], 0);
        assert!(empty_results.is_empty());

        // Mismatched lengths should return empty
        let r2_vec = [[0.0, 7178.0, 0.0]];
        let v2_refs = [[0.0, 0.0, 0.0]];
        let tofs = [3600.0, 4000.0]; // 2 TOFs but only 1 r2

        let mismatched_results =
            izzo2015_batch_tof_variable_r2(MU, &r1, &r2_vec, &v1_ref, &v2_refs, &tofs, 0);
        assert!(mismatched_results.is_empty());
    }

    #[test]
    fn test_batch_tof_seeding_preserves_order() {
        // Test that seeded batch solving returns results in original input order
        // even though TOFs are sorted internally for seeding
        let state1 = leo_state1();
        let state2 = leo_state2();
        let v1_ref = [state1[3], state1[4], state1[5]];
        let v2_ref = [state2[3], state2[4], state2[5]];

        // Deliberately unsorted TOFs
        let tofs = [5000.0, 3000.0, 7000.0, 4000.0, 6000.0];

        let batch_results =
            izzo2015_batch_tof(MU, &R1_LEO, &R2_LEO, &tofs, 1, Some(&v1_ref), Some(&v2_ref));

        assert_eq!(batch_results.len(), tofs.len());

        // Verify each result has the correct TOF (original order preserved)
        for (i, (res, &tof)) in batch_results.iter().zip(tofs.iter()).enumerate() {
            assert_eq!(
                res.tof.to_bits(),
                tof.to_bits(),
                "Result {} should have TOF {} but got {}",
                i,
                tof,
                res.tof
            );
        }
    }

    #[test]
    fn test_batch_tof_seeding_produces_correct_results() {
        // Verify seeded batch produces identical results to non-seeded individual calls
        let state1 = leo_state1();
        let state2 = leo_state2();
        let v1_ref = [state1[3], state1[4], state1[5]];
        let v2_ref = [state2[3], state2[4], state2[5]];

        // Dense TOF scan - perfect for seeding
        let tofs: Vec<f64> = (0..20).map(|i| 3000.0 + f64::from(i) * 200.0).collect();

        let batch_results =
            izzo2015_batch_tof(MU, &R1_LEO, &R2_LEO, &tofs, 1, Some(&v1_ref), Some(&v2_ref));

        // Compare with individual best solutions
        for (&tof, batch_result) in tofs.iter().zip(batch_results.iter()) {
            let (_, _, _, _, dv_norm_ind, valid_ind) =
                izzo2015_best_solution(MU, &state1, &state2, tof, 1, true);

            // Hoisted out of `if valid_ind && batch_result.valid`. This is the only
            // check that the TOF-seeded batch path reproduces unseeded results, and
            // with the flags as a guard it compared nothing whenever seeding broke
            // badly enough to invalidate every row -- the exact regression it exists
            // to catch.
            assert!(
                valid_ind,
                "individual solve must converge at TOF {tof} -- without it this test \
                 compares nothing"
            );
            assert_eq!(
                valid_ind, batch_result.valid,
                "TOF {tof}: seeded batch and individual disagree on validity"
            );

            // Seeded batch should produce same result as individual
            assert!(
                (batch_result.dv_depart - dv_norm_ind).abs() < 1e-6,
                "TOF {}: seeded batch dv={:.9} should match individual dv={:.9}",
                tof,
                batch_result.dv_depart,
                dv_norm_ind
            );
        }
    }

    #[test]
    fn test_seeded_find_xy_uses_seed() {
        // Verify that find_xy_seeded actually uses the provided seed
        let r1_norm = norm3(&R1_LEO);
        let r2_norm = norm3(&R2_LEO);
        let c = [
            R2_LEO[0] - R1_LEO[0],
            R2_LEO[1] - R1_LEO[1],
            R2_LEO[2] - R1_LEO[2],
        ];
        let c_norm = norm3(&c);
        let s = (r1_norm + r2_norm + c_norm) * 0.5;

        let ir1 = [
            R1_LEO[0] / r1_norm,
            R1_LEO[1] / r1_norm,
            R1_LEO[2] / r1_norm,
        ];
        let ir2 = [
            R2_LEO[0] / r2_norm,
            R2_LEO[1] / r2_norm,
            R2_LEO[2] / r2_norm,
        ];
        let ih = cross3(&ir1, &ir2);
        let ih_norm = norm3(&ih);
        let ll_base = (1.0 - (c_norm / s).clamp(0.0, 1.0)).sqrt();
        let ll = if ih[2] / ih_norm < 0.0 {
            -ll_base
        } else {
            ll_base
        };

        let s_cubed = s * s * s;
        let t_nd = (2.0 * MU / s_cubed).sqrt() * TOF_LEO;

        // Solve without seed
        let (x_unseed, y_unseed) = find_xy(ll, t_nd, 0, 15, 1e-9, 1e-9, true);
        assert!(x_unseed.is_finite(), "Unseeded should converge");

        // Solve with seed = the correct solution (should converge immediately)
        let (x_seeded, y_seeded) =
            find_xy_seeded(ll, t_nd, 0, 15, 1e-9, 1e-9, true, Some(x_unseed));

        // Should get same result
        assert!(
            (x_seeded - x_unseed).abs() < 1e-12,
            "Seeded with correct answer should converge to same x"
        );
        assert!(
            (y_seeded - y_unseed).abs() < 1e-12,
            "Seeded with correct answer should converge to same y"
        );

        // Solve with seed = None (should use initial_guess)
        let (x_none_seed, _) = find_xy_seeded(ll, t_nd, 0, 15, 1e-9, 1e-9, true, None);
        assert!(
            (x_none_seed - x_unseed).abs() < 1e-12,
            "Seeded with None should behave like unseeded"
        );
    }

    #[test]
    fn test_batch_tof_variable_r2_seeding_preserves_order() {
        // Test that seeded variable_r2 batch solving returns results in original order
        let r1 = R1_LEO;
        let v1_ref = [0.0, (MU / 6778.0_f64).sqrt(), 0.0];

        // Create multiple positions (simulating target at different times)
        let r2_vec = [
            [0.0, 7178.0, 0.0],
            [100.0, 7178.0, 50.0],
            [-100.0, 7178.0, 100.0],
            [50.0, 7178.0, -50.0],
        ];
        let v2_refs = [
            [-(MU / 7178.0_f64).sqrt(), 0.0, 0.0],
            [-(MU / 7178.0_f64).sqrt(), 0.01, 0.0],
            [-(MU / 7178.0_f64).sqrt(), -0.01, 0.0],
            [-(MU / 7178.0_f64).sqrt(), 0.005, 0.0],
        ];

        // Deliberately unsorted TOFs
        let tofs = [5000.0, 3000.0, 4500.0, 3500.0];

        let results = izzo2015_batch_tof_variable_r2(MU, &r1, &r2_vec, &v1_ref, &v2_refs, &tofs, 1);

        assert_eq!(results.len(), tofs.len());

        // Verify each result has the correct TOF (original order preserved)
        for (i, (res, &tof)) in results.iter().zip(tofs.iter()).enumerate() {
            assert_eq!(
                res.tof.to_bits(),
                tof.to_bits(),
                "Result {} should have TOF {} but got {}",
                i,
                tof,
                res.tof
            );
        }
    }

    #[test]
    fn test_batch_tof_variable_r2_seeding_preserves_order_large_batch() {
        // Exercise multiple SIMD chunks plus a scalar tail while preserving input order.
        let r1 = R1_LEO;
        let v1_ref = [0.0, (MU / 6778.0_f64).sqrt(), 0.0];

        let r2_vec = vec![[0.0, 7178.0, 0.0]; 65];
        let v2_refs = vec![[-(MU / 7178.0_f64).sqrt(), 0.0, 0.0]; 65];
        let tofs: Vec<f64> = (0..65).map(|i| 9000.0 - f64::from(i) * 37.0).collect();

        let results = izzo2015_batch_tof_variable_r2(MU, &r1, &r2_vec, &v1_ref, &v2_refs, &tofs, 0);

        assert_eq!(results.len(), tofs.len());
        for (i, (res, &tof)) in results.iter().zip(tofs.iter()).enumerate() {
            assert_eq!(
                res.tof.to_bits(),
                tof.to_bits(),
                "Result {} should have TOF {} but got {}",
                i,
                tof,
                res.tof
            );
        }
    }

    #[test]
    fn test_householder_simd4_adaptive_matches_scalar_lane_caps() {
        let lanes = [
            // Easy: m=0, circular-ish, short nondimensional TOF -> cap 4.
            (0.10, 2.0, 0),
            // Default: single-rev but less circular/longer -> cap 6.
            (0.50, 6.5, 0),
            // Hard: near-parabolic -> cap 8.
            (0.93, 4.0, 0),
            // Easy again, to prove caps are per-lane rather than global.
            (-0.20, 3.5, 0),
        ];

        let ll_arr = lanes.map(|lane| lane.0);
        let lambda_sq = ll_arr.map(|ll| ll * ll);
        let lambda_cu = ll_arr.map(|ll| ll * ll * ll);
        let lambda_fifth = ll_arr.map(|ll| ll * ll * ll * ll * ll);
        let t_arr = lanes.map(|lane| lane.1);
        let p0_arr = lanes.map(|(ll, t_nd, m)| initial_guess(t_nd, ll, m, true));
        let lane_caps = lanes.map(|(ll, t_nd, m)| adaptive_maxiter(ll, t_nd, m));
        // The SIMD wrapper floors each lane cap exactly as the scalar seeded
        // path does, so the differential reference has to floor too — otherwise
        // this stops comparing two implementations of one contract and starts
        // comparing two contracts. All four lanes converge well inside the raw
        // adaptive cap, so the floor is inert here and both references agree;
        // `simd_adaptive_floor_rescues_a_lane_the_raw_cap_drops` covers the case
        // where it is not inert.
        let scalar_x = lanes.map(|(ll, t_nd, m)| {
            let cap = deterministic_maxiter_floor(adaptive_maxiter(ll, t_nd, m));
            householder_method(
                initial_guess(t_nd, ll, m, true),
                t_nd,
                ll,
                m,
                1e-9,
                1e-9,
                cap,
            )
        });
        assert!(scalar_x.iter().all(|x| x.is_finite()));

        assert_eq!(lane_caps, [4, 6, 8, 4]);

        // Convergence deactivates a lane, so a lane that converges under the
        // raw cap must return bit-identical `x` under the floored cap. Assert
        // that here rather than assuming it.
        for (lane, ((ll, t_nd, m), floored)) in lanes.into_iter().zip(scalar_x).enumerate() {
            let raw = householder_method(
                initial_guess(t_nd, ll, m, true),
                t_nd,
                ll,
                m,
                1e-9,
                1e-9,
                adaptive_maxiter(ll, t_nd, m),
            );
            assert!(
                raw.to_bits() == floored.to_bits(),
                "lane {lane}: raw-cap x={raw} and floored x={floored} must be bit-identical when the raw cap converges"
            );
        }

        let simd_x = self::simd::householder_simd4_adaptive(
            wide::f64x4::new(p0_arr),
            wide::f64x4::new(t_arr),
            wide::f64x4::new(ll_arr),
            wide::f64x4::new(lambda_sq),
            wide::f64x4::new(lambda_cu),
            wide::f64x4::new(lambda_fifth),
            0,
            8,
            1e-9,
            1e-9,
        )
        .to_array();

        for (lane, (simd, scalar)) in simd_x.iter().zip(scalar_x.iter()).enumerate() {
            assert!(
                (simd - scalar).abs() < 1e-10,
                "lane {lane}: adaptive SIMD x={simd} should match scalar x={scalar}"
            );
        }
    }

    /// The `deterministic` feature must reach the SIMD batch path, not just
    /// scalar `find_xy_seeded`.
    ///
    /// Both production callers of `householder_simd4_adaptive` warm-start `p0`
    /// from the previous chunk's converged `x`, so the SIMD path carries the
    /// same seed-dependence the feature exists to remove. Before the floor was
    /// applied there, a lane whose `adaptive_maxiter` cap ran out returned NaN
    /// and was silently skipped by the batch loop — a seed-dependent
    /// *feasibility* flip, not a seed-dependent value.
    ///
    /// Non-vacuity is asserted, not assumed: the scalar solve at the RAW
    /// adaptive cap must be NaN on this geometry (the pre-fix behaviour) and
    /// finite at the floored cap. Without the floor in `simd.rs` the rescue
    /// assertion sees NaN and fails.
    ///
    /// Scope of the claim, measured rather than asserted: sweeping (`ll`, `t_nd`, `m`)
    /// at 0.005 x 0.05 x {0,1,2} over the 374,421 geometries with an elliptical
    /// root, the raw cap drops **zero** lanes seeded within 0.2 of that root,
    /// and no lane that converged under the raw cap EVER returned a different
    /// `x` under the floor (0 of 2.2M). The floor is a NaN -> value conversion,
    /// not a value shift. This exemplar sits at seed distance ~0.3, which is
    /// where the first drops appear.
    #[test]
    fn simd_adaptive_floor_rescues_a_lane_the_raw_cap_drops() {
        // Near-parabolic geometry with an elliptical root at x = -0.3166.
        let (ll, t_nd, m) = (-0.995_f64, 3.65_f64, 0_i32);
        let raw_cap = adaptive_maxiter(ll, t_nd, m);
        assert_eq!(raw_cap, 8, "exemplar must sit on the hard adaptive cap");
        assert!(
            raw_cap < LAMBERT_DETERMINISTIC_MAXITER_FLOOR,
            "exemplar is only meaningful while the floor exceeds the raw cap"
        );

        let seed = -0.0165_f64;
        let dropped = householder_method(seed, t_nd, ll, m, 1e-9, 1e-9, raw_cap);
        assert!(
            dropped.is_nan(),
            "POISON CHECK: the raw cap must DROP this lane, got x={dropped}"
        );
        let rescued = householder_method(
            seed,
            t_nd,
            ll,
            m,
            1e-9,
            1e-9,
            LAMBERT_DETERMINISTIC_MAXITER_FLOOR,
        );
        assert!(
            rescued.is_finite(),
            "the floored cap must converge, got x={rescued}"
        );

        // Lanes 0 and 2 are the rescue geometry; lanes 1 and 3 are an easy
        // geometry that converges inside its raw cap of 4. The easy lanes are
        // the load-bearing half: the floor raises the vector's trip count from
        // 8 to 24, so if lane deactivation did not preserve converged values,
        // they would drift. They must not.
        let easy = (0.10_f64, 2.0_f64);
        assert_eq!(adaptive_maxiter(easy.0, easy.1, m), 4);
        let easy_seed = initial_guess(easy.1, easy.0, m, true);
        let ll_arr = [ll, easy.0, ll, easy.0];
        let t_arr = [t_nd, easy.1, t_nd, easy.1];
        let p0_arr = [seed, easy_seed, seed, easy_seed];
        let lambda = wide::f64x4::new(ll_arr);
        let simd_x = self::simd::householder_simd4_adaptive(
            wide::f64x4::new(p0_arr),
            wide::f64x4::new(t_arr),
            lambda,
            lambda * lambda,
            lambda * lambda * lambda,
            lambda * lambda * lambda * lambda * lambda,
            m,
            8,
            1e-9,
            1e-9,
        )
        .to_array();

        // The easy lane's pre-floor SIMD value, taken from the kernel
        // `householder_simd4_adaptive` itself wraps, with the raw cap in place
        // of the floor. Comparing SIMD against SIMD is what makes a BIT claim
        // meaningful here; SIMD vs scalar legitimately differs by ~1 ULP from
        // vectorized FMA, which is what the sibling parity tests bound at
        // 1e-10. The reference must come from THIS kernel family: both
        // families now share `tof_equation_y_simd_per_lane_m_pi`, but
        // `householder_simd4_m_variant` still applies the Householder update
        // as reciprocal-then-multiply (with a NaN guard on a zero denominator)
        // where this family divides, so it is a different function at the
        // last ULP.
        let easy_lambda = wide::f64x4::splat(easy.0);
        let easy_raw_simd = self::simd::householder_simd4_with_lane_maxiters(
            wide::f64x4::splat(easy_seed),
            wide::f64x4::splat(easy.1),
            easy_lambda,
            easy_lambda * easy_lambda,
            easy_lambda * easy_lambda * easy_lambda,
            easy_lambda * easy_lambda * easy_lambda * easy_lambda * easy_lambda,
            m,
            [adaptive_maxiter(easy.0, easy.1, m); 4],
            1e-9,
            1e-9,
        )
        .to_array()[1];
        assert!(easy_raw_simd.is_finite());

        for (lane, x) in simd_x.into_iter().enumerate() {
            if lane % 2 == 0 {
                assert!(
                    x.is_finite(),
                    "lane {lane}: the deterministic floor must reach the SIMD path and rescue this lane"
                );
                assert!(
                    (x - rescued).abs() <= 1e-10 * rescued.abs(),
                    "lane {lane}: rescued SIMD x={x} should match scalar x={rescued}"
                );
            } else {
                assert!(
                    x.to_bits() == easy_raw_simd.to_bits(),
                    "lane {lane}: an already-converged lane must be BIT-identical under the raised trip count; floored x={x} vs raw-cap x={easy_raw_simd}"
                );
            }
        }
    }

    /// Accuracy bound on the packed enumerator's departure/arrival delta-v,
    /// relative, versus the scalar enumerator it would replace.
    ///
    /// **Measured max is 4.554e-14** over the 6,400-geometry grid below; the
    /// bound is set ~22x above it so a libm difference between hosts cannot
    /// flake it. For scale, the root deviation behind it is 7.327e-15 max over
    /// 86,524 real production operands, and `CONVERGENCE_TOL`'s own doc records
    /// the repo ALREADY accepting a 1.08e-13 max root shift when the exit
    /// tolerance moved 1e-9 -> 1e-6 (`d594900`). This perturbation is smaller
    /// than one already shipped.
    ///
    /// Tightening this is welcome; loosening it needs a measurement, because
    /// the number it guards is the whole accuracy statement for the packed
    /// path (`docs/plans/2026-08-08-r17-simd-lambert-front.md`).
    const PACKED_DV_RELATIVE_BOUND: f64 = 1e-12;

    /// The wired SIMD pack must emit the SAME branches, in the SAME order, as
    /// the scalar enumerator it replaces.
    ///
    /// Order is load-bearing, not cosmetic: the caller keeps a running argmin
    /// with a strict `<`, so two branches that tie resolve to whichever was
    /// visited first. A pack that emitted the same SET in a different order
    /// would still change the selected transfer.
    ///
    /// Delta-v is compared at 1e-9 relative rather than by bits. That is the
    /// whole trade this branch exists to price: `wide`'s `acos` is a
    /// polynomial where the scalar path calls platform libm, so `x` differs in
    /// the last ULPs and every float downstream of it moves. Bit-identity is
    /// not available and is not claimed.
    #[test]
    fn simd_pack_enumeration_matches_the_scalar_enumerator_branch_for_branch() {
        type Emitted = (i32, bool, bool, [f64; 3], [f64; 3]);

        let mut cases = 0_u32;
        let mut branches = 0_u32;
        let mut multirev = 0_u32;
        let mut worst_rel = 0.0_f64;

        for altitude in [420.0_f64, 560.0, 780.0, 980.0, 1_180.0] {
            for phase in [0.2_f64, 0.35, 0.8, 1.1, 1.7, 2.4, 3.1, 3.9, 4.6, 5.4] {
                for tof in [
                    1_800.0_f64,
                    3_600.0,
                    5_400.0,
                    9_000.0,
                    14_400.0,
                    21_600.0,
                    36_000.0,
                    54_000.0,
                ] {
                    for m_max in [0_i32, 1, 2, 4] {
                        for requested_low_path in [true, false] {
                            for include_retrograde in [true, false] {
                                let r1_norm = 6_378.137 + altitude;
                                let r2_norm = 6_378.137 + altitude + 55.0;
                                let state1 = [r1_norm, 0.0, 0.0, 0.0, 7.55, 0.9];
                                let state2 = [
                                    r2_norm * phase.cos(),
                                    r2_norm * phase.sin(),
                                    120.0,
                                    -7.4 * phase.sin(),
                                    7.4 * phase.cos(),
                                    0.4,
                                ];

                                let r1 = [state1[0], state1[1], state1[2]];
                                let r1_cache = LambertR1Cache::new(&r1);

                                // Both arms are called EXPLICITLY. Going through
                                // `for_each_lambert_m_prograde_lowpaths_pruned_with_r1`
                                // for the scalar arm would dispatch to the pack
                                // now that the pack is the default, and this test
                                // would compare the pack against itself and pass
                                // vacuously.
                                let r2 = [state2[0], state2[1], state2[2]];
                                let m_max_feasible =
                                    compute_m_max_fast(&r1, &r2, tof, satpy_core::MU).min(m_max);
                                let mut scalar: Vec<Emitted> = Vec::new();
                                let mut packed: Vec<Emitted> = Vec::new();
                                if m_max_feasible >= 0 {
                                    let geom = compute_lambert_geometry_with_r1(
                                        satpy_core::MU,
                                        &r1_cache,
                                        &r2,
                                        tof,
                                    );
                                    for_each_lambert_scalar_branch_enumeration(
                                        &geom,
                                        &state1,
                                        &state2,
                                        m_max_feasible,
                                        requested_low_path,
                                        include_retrograde,
                                        &mut |m,
                                              low_path,
                                              prograde,
                                              dv_depart,
                                              dv_arrive,
                                              valid| {
                                            if valid {
                                                scalar.push((
                                                    m, low_path, prograde, dv_depart, dv_arrive,
                                                ));
                                            }
                                        },
                                        &mut || {},
                                    );
                                    for_each_lambert_simd_pack_enumeration(
                                        &geom,
                                        &state1,
                                        &state2,
                                        m_max_feasible,
                                        requested_low_path,
                                        include_retrograde,
                                        &mut |m,
                                              low_path,
                                              prograde,
                                              dv_depart,
                                              dv_arrive,
                                              valid| {
                                            if valid {
                                                packed.push((
                                                    m, low_path, prograde, dv_depart, dv_arrive,
                                                ));
                                            }
                                        },
                                        &mut || {},
                                    );
                                }

                                cases += 1;
                                assert_eq!(
                                    scalar.len(),
                                    packed.len(),
                                    "branch count differs at altitude={altitude} phase={phase} \
                                     tof={tof} m_max={m_max}: scalar emitted {} branches, the \
                                     pack emitted {}",
                                    scalar.len(),
                                    packed.len()
                                );

                                for (index, (want, got)) in
                                    scalar.iter().zip(packed.iter()).enumerate()
                                {
                                    assert_eq!(
                                        (want.0, want.1, want.2),
                                        (got.0, got.1, got.2),
                                        "branch {index} identity/order differs at \
                                         altitude={altitude} phase={phase} tof={tof} \
                                         m_max={m_max}"
                                    );
                                    branches += 1;
                                    if want.0 >= 1 {
                                        multirev += 1;
                                    }
                                    for (want_dv, got_dv) in want
                                        .3
                                        .iter()
                                        .chain(want.4.iter())
                                        .zip(got.3.iter().chain(got.4.iter()))
                                    {
                                        let scale = want_dv.abs().max(1e-6);
                                        let rel = (want_dv - got_dv).abs() / scale;
                                        worst_rel = worst_rel.max(rel);
                                        assert!(
                                            rel < PACKED_DV_RELATIVE_BOUND,
                                            "branch {index} dv differs by {rel:.3e} relative \
                                             (scalar {want_dv}, pack {got_dv}) at \
                                             altitude={altitude} phase={phase} tof={tof}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(cases, 6_400, "the grid must actually enumerate");
        assert!(
            branches > 2_000,
            "the grid must produce real solutions, else this proves nothing \
             (branches={branches})"
        );
        assert!(
            multirev > 200,
            "the grid must contain multi-revolution survivors, which are the \
             lanes the pack exists to batch (multirev={multirev})"
        );
        assert!(
            worst_rel > 0.0,
            "if the pack were bit-identical to scalar this test would be \
             measuring nothing; the acos difference must be visible"
        );
    }

    /// Pins `seed_cbrt`'s two accuracy claims: the deviation from libm over
    /// the operational operand range (what the dv criterion leans on) and a
    /// full-positive-range bound (what makes the guard-free bit-hack safe
    /// wherever an operand wanders). NaN/zero/negative/infinite fall back to
    /// libm bit-for-bit.
    ///
    /// Both asserts are at 1.0e-14, tightened at R21 from 1e-13 and 1e-11.
    /// Measured on this host: operational 6.4544e-15, full range 6.2405e-15,
    /// so the bound carries 1.55x and 1.60x of margin and the test now fails
    /// on roughly a doubling of the deviation instead of on a 15x or 1602x
    /// one. The margin basis is cross-libm drift, which is the only thing
    /// expected to move these: the deviation is dominated by the second Halley
    /// step's own residual (~29 ULP of the result), while a libm `cbrt` that
    /// disagrees by a full ULP moves the measured figure by ~2.2e-16, i.e. 3.5%
    /// of the bound. Discrimination is unaffected by the tightening — the
    /// one-Halley variant sits at 2.09e-5 and the bare bit-hack seed at
    /// 3.20e-2, thirteen and twelve orders above either bound — and a third
    /// Halley step (4.44e-16) would still pass, so this pins accuracy without
    /// pinning the step count.
    #[test]
    fn seed_cbrt_tracks_libm_cbrt_across_the_operand_range() {
        // Operational range: positive, ~1e-2..1e2 (t0/t and multi-rev ratios).
        let mut worst_operational = 0.0_f64;
        for step in 0_i32..=40_000 {
            let x = 1e-2 * (1e4_f64).powf(f64::from(step) / 40_000.0);
            let rel = ((seed_cbrt(x) - x.cbrt()) / x.cbrt()).abs();
            worst_operational = worst_operational.max(rel);
        }
        assert!(
            worst_operational < 1.0e-14,
            "operational-range deviation {worst_operational:.4e} exceeds 1.0e-14 \
             (measured 6.4544e-15 at the R21 tightening; read this test's doc \
             comment before loosening it)"
        );

        // Full positive normal range: the bit-hack start must stay inside
        // Halley's basin everywhere.
        let mut worst_full = 0.0_f64;
        for step in 0_i32..=60_000 {
            // 1e-300 .. 1e300, log-uniform: exponent -300 + step/100.
            let x = 10.0_f64.powf(f64::from(step).mul_add(1.0 / 100.0, -300.0));
            if !x.is_finite() || x <= 0.0 {
                continue;
            }
            let rel = ((seed_cbrt(x) - x.cbrt()) / x.cbrt()).abs();
            worst_full = worst_full.max(rel);
        }
        assert!(
            worst_full < 1.0e-14,
            "full-range deviation {worst_full:.4e} exceeds 1.0e-14 \
             (measured 6.2405e-15 at the R21 tightening; read this test's doc \
             comment before loosening it)"
        );
        println!(
            "seed_cbrt deviation: operational max {worst_operational:.3e}, full-range max {worst_full:.3e}"
        );

        // Fallback lanes are libm bit-for-bit.
        for x in [0.0_f64, -8.0, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(seed_cbrt(x).to_bits(), x.cbrt().to_bits());
        }
        assert!(seed_cbrt(f64::NAN).is_nan());
    }

    /// Acceptance measurement for routing the selected-branch (exact) route
    /// through the pack: over the same 6,400-case grid as the enumerator gate,
    /// every `(rev, low_path)` selection must emit the same variants, in the
    /// same order (prograde then retrograde), as the scalar exact route it
    /// replaces, with relative dv deviation inside `PACKED_DV_RELATIVE_BOUND`.
    ///
    /// The dv criterion is the one documented at
    /// bounded dv deviation (precedent
    /// 4.554e-14 accepted against the 1.08e-13 shift `d594900` already
    /// shipped), NOT front identity.
    #[test]
    fn exact_branch_pack_stays_inside_the_packed_dv_relative_bound() {
        let mut variants = 0_u32;
        let mut multirev = 0_u32;
        let mut worst_rel = 0.0_f64;

        for altitude in [420.0_f64, 780.0, 1_180.0] {
            for phase in [0.35_f64, 1.1, 2.4, 3.9, 5.4] {
                for tof in [3_600.0_f64, 9_000.0, 21_600.0, 54_000.0] {
                    let r1_norm = 6_378.137 + altitude;
                    let r2_norm = 6_378.137 + altitude + 55.0;
                    let state1 = [r1_norm, 0.0, 0.0, 0.0, 7.55, 0.9];
                    let state2 = [
                        r2_norm * phase.cos(),
                        r2_norm * phase.sin(),
                        120.0,
                        -7.4 * phase.sin(),
                        7.4 * phase.cos(),
                        0.4,
                    ];
                    let r1 = [state1[0], state1[1], state1[2]];
                    let r2 = [state2[0], state2[1], state2[2]];
                    let r1_cache = LambertR1Cache::new(&r1);

                    for rev in 0_i32..=4 {
                        for low_path in [true, false] {
                            for include_retrograde in [true, false] {
                                // Scalar exact route, explicitly: the guards
                                // and dv arithmetic the delegated entry's
                                // scalar arm keeps.
                                type Emitted = (i32, bool, bool, [f64; 3], [f64; 3]);
                                let mut scalar: Vec<Emitted> = Vec::new();
                                if compute_m_max_fast(&r1, &r2, tof, MU) >= rev {
                                    let geom =
                                        compute_lambert_geometry_with_r1(MU, &r1_cache, &r2, tof);
                                    if geom.success {
                                        for prograde in [true, false] {
                                            if !prograde && !include_retrograde {
                                                continue;
                                            }
                                            let res = izzo2015_impl_with_geom_fast(
                                                &geom,
                                                rev,
                                                prograde,
                                                low_path,
                                                8,
                                                CONVERGENCE_TOL,
                                                CONVERGENCE_TOL,
                                            );
                                            if !res.success {
                                                continue;
                                            }
                                            scalar.push((
                                                rev,
                                                low_path,
                                                prograde,
                                                [
                                                    res.v1[0] - state1[3],
                                                    res.v1[1] - state1[4],
                                                    res.v1[2] - state1[5],
                                                ],
                                                [
                                                    state2[3] - res.v2[0],
                                                    state2[4] - res.v2[1],
                                                    state2[5] - res.v2[2],
                                                ],
                                            ));
                                        }
                                    }
                                }

                                let mut packed: Vec<Emitted> = Vec::new();
                                for_each_lambert_exact_branch_with_r1(
                                    MU,
                                    &r1_cache,
                                    &state1,
                                    &state2,
                                    tof,
                                    rev,
                                    low_path,
                                    include_retrograde,
                                    |m, lane_low_path, prograde, dv_depart, dv_arrive, valid| {
                                        if valid {
                                            packed.push((
                                                m,
                                                lane_low_path,
                                                prograde,
                                                dv_depart,
                                                dv_arrive,
                                            ));
                                        }
                                    },
                                );

                                assert_eq!(
                                    scalar.len(),
                                    packed.len(),
                                    "variant count differs at altitude={altitude} phase={phase} \
                                     tof={tof} rev={rev} low_path={low_path}"
                                );
                                for (want, got) in scalar.iter().zip(packed.iter()) {
                                    assert_eq!(
                                        (want.0, want.1, want.2),
                                        (got.0, got.1, got.2),
                                        "variant identity/order differs at altitude={altitude} \
                                         phase={phase} tof={tof} rev={rev}"
                                    );
                                    variants += 1;
                                    if want.0 >= 1 {
                                        multirev += 1;
                                    }
                                    for (want_dv, got_dv) in want
                                        .3
                                        .iter()
                                        .chain(want.4.iter())
                                        .zip(got.3.iter().chain(got.4.iter()))
                                    {
                                        let scale = want_dv.abs().max(1e-6);
                                        let rel = (want_dv - got_dv).abs() / scale;
                                        worst_rel = worst_rel.max(rel);
                                        assert!(
                                            rel < PACKED_DV_RELATIVE_BOUND,
                                            "exact-branch dv differs by {rel:.3e} relative \
                                             (scalar {want_dv}, pack {got_dv}) at \
                                             altitude={altitude} phase={phase} tof={tof} rev={rev}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        assert!(
            variants > 500,
            "the grid must produce real selected-branch solutions (variants={variants})"
        );
        assert!(
            multirev > 50,
            "the grid must contain multi-revolution selections (multirev={multirev})"
        );
        assert!(
            worst_rel > 0.0,
            "if the exact-branch pack were bit-identical to scalar this test \
             would be measuring nothing; the acos difference must be visible"
        );
        println!("exact-branch pack worst relative dv deviation: {worst_rel:.3e}");
    }

    /// The dual-Lambert closure, pinned: the selected-branch (exact) route
    /// must return BIT-identical dv to the per-candidate enumerator's pack
    /// lane for the same `(geometry, m, low_path, prograde)` variant. This is
    /// the property whose absence R17b flagged (same inputs, two Lambert
    /// answers ~2 ULP apart depending on the entry reached).
    ///
    /// CAVEAT, same as the multi-TOF streaming gate above: both arms share the
    /// pack, so a perturbation of the pack's arithmetic cancels and this stays
    /// green under the retired coarse and ULP mutations. It is a
    /// self-consistency test of the two ENTRIES, not arithmetic coverage.
    #[test]
    fn exact_branch_pack_matches_the_enumerators_pack_lane_bitwise() {
        let mut matched = 0_u32;
        let mut multirev = 0_u32;

        for altitude in [420.0_f64, 780.0, 1_180.0] {
            for phase in [0.35_f64, 1.1, 2.4, 3.9, 5.4] {
                for tof in [3_600.0_f64, 9_000.0, 21_600.0, 54_000.0] {
                    let r1_norm = 6_378.137 + altitude;
                    let r2_norm = 6_378.137 + altitude + 55.0;
                    let state1 = [r1_norm, 0.0, 0.0, 0.0, 7.55, 0.9];
                    let state2 = [
                        r2_norm * phase.cos(),
                        r2_norm * phase.sin(),
                        120.0,
                        -7.4 * phase.sin(),
                        7.4 * phase.cos(),
                        0.4,
                    ];
                    let r1 = [state1[0], state1[1], state1[2]];
                    let r1_cache = LambertR1Cache::new(&r1);

                    // Full enumeration through the public (pack) entry.
                    let mut enumerated: Vec<((i32, bool, bool), ([u64; 3], [u64; 3]))> = Vec::new();
                    for_each_lambert_m_prograde_lowpaths_pruned_with_r1(
                        MU,
                        &r1_cache,
                        &state1,
                        &state2,
                        tof,
                        4,
                        true,
                        true,
                        |m, low_path, prograde, dv_depart, dv_arrive, valid| {
                            if valid {
                                enumerated.push((
                                    (m, low_path, prograde),
                                    (dv_depart.map(f64::to_bits), dv_arrive.map(f64::to_bits)),
                                ));
                            }
                        },
                    );

                    for rev in 0_i32..=4 {
                        for low_path in [true, false] {
                            for_each_lambert_exact_branch_with_r1(
                                MU,
                                &r1_cache,
                                &state1,
                                &state2,
                                tof,
                                rev,
                                low_path,
                                true,
                                |m, lane_low_path, prograde, dv_depart, dv_arrive, valid| {
                                    if !valid {
                                        return;
                                    }
                                    let Some((_, want)) = enumerated
                                        .iter()
                                        .find(|(key, _)| *key == (m, lane_low_path, prograde))
                                    else {
                                        return;
                                    };
                                    assert_eq!(
                                        &(dv_depart.map(f64::to_bits), dv_arrive.map(f64::to_bits)),
                                        want,
                                        "exact-branch dv bits differ from the enumerator's pack \
                                         lane at altitude={altitude} phase={phase} tof={tof} \
                                         rev={rev} low_path={low_path} prograde={prograde}"
                                    );
                                    matched += 1;
                                    if m >= 1 {
                                        multirev += 1;
                                    }
                                },
                            );
                        }
                    }
                }
            }
        }

        assert!(
            matched > 500,
            "the grid must produce overlapping variants to compare (matched={matched})"
        );
        assert!(
            multirev > 50,
            "the comparison must include multi-revolution variants (multirev={multirev})"
        );
    }

    #[test]
    fn simd4_m_variant_never_drops_a_lane_the_scalar_solver_converges() {
        // Regression for the branch-coverage gap in
        // `tof_equation_y_simd_per_lane_m_pi`: it evaluated only the `acos`
        // limb of `compute_psi` and took `sqrt` of the SIGNED `1 - x^2`, so an
        // m = 0 lane whose Householder iterates enter the `hyp2f1b` band
        // (sqrt(0.6) < x < sqrt(1.4)) computed a different function, and one
        // that crossed x = 1 went NaN while the scalar solver converged.
        // Over the production MF branch-enumerator operand corpus that dropped
        // 340 of 86,524 solved lanes -- a changed argmin, not a rounding
        // difference.
        //
        // Operands are real, taken from that corpus. All four share one TOF, as
        // the enumerator's lanes do. Lane 0's scalar root is x = 1.031, above
        // the old kernel's NaN cliff; lanes 1 and 2 land inside the band.
        let t = 6.517_481_429_414_88e-1;
        let lanes: [(f64, i32, bool); 4] = [
            (1.568_204_986_418_414_4e-1, 0, true),
            (-1.568_204_986_418_414_4e-1, 0, true),
            (2.361_246_316_258_943e-1, 0, true),
            (5.0e-1, 0, true),
        ];

        let (simd_x, simd_y) = find_xy_simd4_m_variant(
            lanes.map(|lane| lane.0),
            t,
            lanes.map(|lane| lane.1),
            lanes.map(|lane| lane.2),
            8,
            CONVERGENCE_TOL,
            CONVERGENCE_TOL,
        );

        let mut checked = 0_u32;
        let mut saw_band = false;
        for (lane_index, (((ll, m, low_path), sx_simd), sy_simd)) in
            lanes.into_iter().zip(simd_x).zip(simd_y).enumerate()
        {
            let (sx, sy) = find_xy(ll, t, m, 8, CONVERGENCE_TOL, CONVERGENCE_TOL, low_path);
            if !sx.is_finite() {
                continue;
            }
            checked += 1;
            if m == 0 && sx > (0.6_f64).sqrt() {
                saw_band = true;
            }
            assert!(
                sx_simd.is_finite() && sy_simd.is_finite(),
                "lane {lane_index}: scalar converged to x={sx} but the SIMD pack \
                 returned {sx_simd}; the pack must never drop a lane the scalar \
                 solver solves"
            );
            assert!(
                (sx_simd - sx).abs() <= 1e-13 * sx.abs(),
                "lane {lane_index}: SIMD x={sx_simd} vs scalar x={sx} \
                 (delta={})",
                (sx_simd - sx).abs()
            );
            assert!(
                (sy_simd - sy).abs() <= 1e-13 * sy.abs().max(1e-300),
                "lane {lane_index}: SIMD y={sy_simd} vs scalar y={sy}"
            );
        }
        assert!(
            checked >= 3,
            "the corpus lanes must actually solve, else this test proves nothing (checked={checked})"
        );
        assert!(
            saw_band,
            "at least one lane must land in the hyp2f1b band, else the branch \
             this test exists for is never exercised"
        );
    }

    #[test]
    fn test_find_xy_simd4_m_variant_matches_scalar() {
        // HF-NEW-01 parity test: 4 representative (ll, m, low_path) tuples
        // sharing the same non-dimensional TOF. Scalar `find_xy` vs the SIMD4
        // batched `find_xy_simd4_m_variant`. Per-lane |x_simd - x_scalar|
        // must agree to 1e-12, same for y.
        let t = 6.5_f64;
        // Lanes cover: single-rev easy, single-rev mid, multi-rev m=1 low,
        // multi-rev m=1 high. ll values stay below 1.0 so degenerate reject
        // never fires; m_max_quick(t=6.5) = floor(6.5/pi) = 2, so m=1 is
        // strictly below the boundary and the t_min Halley check does not
        // engage on any lane.
        let lanes: [(f64, i32, bool); 4] = [
            (0.10, 0, true),
            (0.50, 0, true),
            (0.30, 1, true),
            (0.30, 1, false),
        ];

        let ll_arr = lanes.map(|lane| lane.0);
        let m_arr = lanes.map(|lane| lane.1);
        let low_path_arr = lanes.map(|lane| lane.2);

        let (simd_x, simd_y) =
            find_xy_simd4_m_variant(ll_arr, t, m_arr, low_path_arr, 8, 1e-9, 1e-9);

        for (lane, (((ll, m, low_path), simd_x_value), simd_y_value)) in
            lanes.into_iter().zip(simd_x).zip(simd_y).enumerate()
        {
            let (sx, sy) = find_xy(ll, t, m, 8, 1e-9, 1e-9, low_path);
            assert!(
                sx.is_finite() && sy.is_finite(),
                "lane {lane}: scalar reference should converge for (ll={ll}, m={m}, lp={low_path})"
            );
            assert!(
                simd_x_value.is_finite() && simd_y_value.is_finite(),
                "lane {lane}: SIMD4 m_variant should converge"
            );
            assert!(
                (simd_x_value - sx).abs() < 1e-12,
                "lane {lane}: SIMD4 x={} should match scalar x={} to 1e-12 (delta={})",
                simd_x_value,
                sx,
                (simd_x_value - sx).abs()
            );
            assert!(
                (simd_y_value - sy).abs() < 1e-12,
                "lane {lane}: SIMD4 y={} should match scalar y={} to 1e-12 (delta={})",
                simd_y_value,
                sy,
                (simd_y_value - sy).abs()
            );
        }
    }

    #[test]
    fn test_find_xy_simd4_m_variant_rejects_degenerate_lanes() {
        // Mix of valid and degenerate (ll >= 1 -> NaN) lanes plus an
        // m > m_max_quick reject. Valid lanes must still solve while rejected
        // lanes return NaN sentinels.
        let t = 6.5_f64;
        let ll_arr = [0.10, 1.0, 0.50, 0.30];
        let m_arr = [0, 0, 0, 99]; // lane 3: m > m_max_quick(t=6.5)=2 -> reject
        let low_path_arr = [true, true, true, true];

        let (simd_x, simd_y) =
            find_xy_simd4_m_variant(ll_arr, t, m_arr, low_path_arr, 8, 1e-9, 1e-9);

        // Lane 0 + lane 2: valid; lane 1: ll=1.0 -> NaN; lane 3: m>m_max -> NaN
        let [x0, x1, x2, x3] = simd_x;
        let [y0, y1, y2, y3] = simd_y;
        assert!(x0.is_finite(), "lane 0 should converge");
        assert!(y0.is_finite(), "lane 0 y should be finite");
        assert!(x1.is_nan(), "lane 1 should NaN (degenerate ll)");
        assert!(y1.is_nan(), "lane 1 y should NaN");
        assert!(x2.is_finite(), "lane 2 should converge");
        assert!(y2.is_finite(), "lane 2 y should be finite");
        assert!(x3.is_nan(), "lane 3 should NaN (m > m_max_quick)");
        assert!(y3.is_nan(), "lane 3 y should NaN");
    }

    #[test]
    fn test_reduced_maxiter_still_converges() {
        // Verify that reducing maxiter from 50 to 8 doesn't break convergence
        // for typical orbital mechanics problems

        // Test a variety of cases with maxiter=8
        let cases = [
            (R1_LEO, R2_LEO, TOF_LEO, 0, true, true),
            (R1_LEO, R2_LEO, TOF_LEO, 0, false, true),
            (R1_LEO, R2_LEO, 7200.0, 0, true, true),
            (
                [42164.0, 0.0, 0.0],
                [0.0, 42164.0, 0.0],
                21600.0,
                0,
                true,
                true,
            ),
        ];

        for (r1, r2, tof, m, prograde, low_path) in cases {
            let result = izzo2015_impl(MU, &r1, &r2, tof, m, prograde, low_path, 8, 1e-9, 1e-9);
            assert!(
                result.success,
                "Should converge with maxiter=8 for r1={r1:?}, r2={r2:?}, tof={tof}"
            );
            assert!(result.v1[0].is_finite());
            assert!(result.v2[0].is_finite());
        }
    }

    /// Pass 13.1 determinism invariant: `izzo2015_impl_with_geom_seeded` must
    /// return the same `(v1, v2)` regardless of the warm-start `x_seed`.
    ///
    /// Under the deterministic feature, the Newton iteration is floored at
    /// `LAMBERT_DETERMINISTIC_MAXITER_FLOOR` so cold-start and warm-start
    /// solves both reach the converged fixed point. Without the floor, the
    /// Householder solver could bottom out at maxiter=8 with seed-dependent
    /// `x`, which broke the OXYMOO NSGA-II `par_iter` work in pass 12.8d.
    #[test]
    fn lambert_seed_independence_random_envelope() {
        let mu = MU;
        // 256 (r1, r2, tof, m) tuples drawn from the production LEO/MEO envelope.
        // Uses a deterministic LCG so the test itself is reproducible.
        let mut state: u64 = 0xDEAD_BEEF_C0DE_F00D;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f64::from(u32::try_from(state >> 33).unwrap_or(0)) / 2_147_483_648.0
        };

        let mut total = 0usize;
        let mut matched = 0usize;
        let mut mismatch_count = 0usize;
        let mut worst_diff = 0.0_f64;

        for _ in 0..256 {
            // r in [6500, 8000] km, random unit direction, second point rotated.
            let r1_mag = 6500.0 + 1500.0 * next();
            let r2_mag = 6500.0 + 1500.0 * next();
            let theta1 = next() * 2.0 * std::f64::consts::PI;
            let theta2 = theta1 + 0.2 + next() * (std::f64::consts::PI - 0.4);
            let r1 = [r1_mag * theta1.cos(), r1_mag * theta1.sin(), 0.0];
            let r2 = [r2_mag * theta2.cos(), r2_mag * theta2.sin(), 0.0];
            let tof = 600.0 + next() * 7200.0;
            let m = i32::from(next() >= 0.7);
            let prograde = next() < 0.5;
            let low_path = next() < 0.5;

            let geom = compute_lambert_geometry(mu, &r1, &r2, 1.0);
            if !geom.success {
                continue;
            }
            let s_factor = (2.0 * mu / geom.s_cubed).sqrt();
            let mut g = geom;
            g.t_nd = s_factor * tof;

            // Cold solve (no warm-start) — captures the canonical x for this
            // geometry / branch.
            let (cold, x_cold) =
                izzo2015_impl_with_geom_seeded(&g, m, prograde, low_path, 8, 1e-9, 1e-9, None);
            if !cold.success {
                continue;
            }

            // Warm solve seeded with the converged x perturbed by a small
            // epsilon — the realistic production scenario (previous-call
            // converged x carried over via VariableR2LambertScratch). We do
            // NOT inject random seeds here because Lambert has multiple
            // basins (low/high energy, multi-rev); a wildly-different seed
            // can legitimately flip into a different physical solution,
            // which is a property of the solver, not a determinism bug.
            let warm_eps = 1.0e-6 * (next() - 0.5);
            let warm_seed = (x_cold + warm_eps).clamp(-0.999, 0.999);
            let (warm, _) = izzo2015_impl_with_geom_seeded(
                &g,
                m,
                prograde,
                low_path,
                8,
                1e-9,
                1e-9,
                Some(warm_seed),
            );

            total += 1;
            if !warm.success {
                continue;
            }
            let max_abs_v1 = cold.v1[0].abs().max(cold.v1[1].abs()).max(cold.v1[2].abs());
            let max_abs_v2 = cold.v2[0].abs().max(cold.v2[1].abs()).max(cold.v2[2].abs());
            let tol = 1.0e-9_f64.max(1.0e-10 * max_abs_v1.max(max_abs_v2));
            let mut local_max = 0.0_f64;
            for ((cold_v1, warm_v1), (cold_v2, warm_v2)) in cold
                .v1
                .iter()
                .zip(warm.v1.iter())
                .zip(cold.v2.iter().zip(warm.v2.iter()))
            {
                local_max = local_max
                    .max((cold_v1 - warm_v1).abs())
                    .max((cold_v2 - warm_v2).abs());
            }
            if local_max <= tol {
                matched += 1;
            } else {
                mismatch_count += 1;
                if local_max > worst_diff {
                    worst_diff = local_max;
                }
            }
        }

        assert!(
            total > 50,
            "fuzz envelope produced too few valid solves: {total} of 256"
        );
        // At least 98% of converged pairs must match within tolerance. The 2%
        // slack covers the rare warm-start that's so far off the basin that
        // even 24 iterations don't help; the determinism win for the bulk is
        // what matters for the downstream NSGA-II.
        let match_ratio = usize_to_f64_or_infinity(matched) / usize_to_f64_or_infinity(total);
        assert!(
            match_ratio >= 0.98,
            "seed-independence ratio {match_ratio:.3} below 0.98 ({matched} matched / {total} total, {mismatch_count} mismatched, worst diff {worst_diff:.3e})"
        );
    }
}
