// gravity.cu — CUDA kernel mirror of
// `satpy_core::gravity::spherical_gravity_impl_sincos_packed`.
//
// Design (Phase B1):
//   - Dense unpacked f64 coefficient layout (zero SIMT divergence on the
//     summation; ~5% more flops than CPU's packed/sparse path).
//   - One CUDA block per state. 32 threads/block (one warp).
//   - V/W Pines recursion is filled row-by-row (l outer). Within each
//     row, the 32 warp lanes stride across m∈[0..=l].
//   - Frame rotation (ECI -> ECEF -> ECI) done device-side. Host passes
//     GCRS positions + the assembled GCRS->ITRS rotation, gets GCRS
//     accelerations back.
//   - vw_scratch is a per-block global slice of size [2][MAX_REC][MAX_REC]
//     in f64; total per-block ~268 KiB. At N=256 ≈ 67 MiB on device.
//   - Precomputes pt1[l][m] / pt21_factor[l][m] inline (no per-call upload).
//     CPU code caches these via `LegendreCoeffsSimd`; we recompute on device.
//
// Naming maps:
//   CPU `spherical_gravity_impl_sincos_packed`  -> kernel `spherical_gravity_batch_dense`
//   CPU `spherical_gravity_impl_frame`           -> dev_gcrs_to_itrs / dev_itrs_to_gcrs
//   CPU `ecef2eci_impl_sincos`                   -> superseded, see below
//   CPU `gravity_summation_generic`              -> body of kernel after V/W fill

#include <cstdint>

extern "C" {

// Compile-time bounds matching satpy_core::gravity::{MAX_ORDER, MAX_RECURSIVE_ORDER}.
// MAX_ORDER = 128 (CPU side); MAX_RECURSIVE_ORDER = MAX_ORDER + 3 = 131.
#define MAX_REC 131

// Earth constants — mirror satpy_core::{MU, GRAVITY_REFERENCE_RADIUS_KM,
// EARTH_OMEGA}. Keep in sync.
__device__ __constant__ double GPU_MU         = 398600.4415;     // km^3/s^2
__device__ __constant__ double GPU_GRAVITY_REFERENCE_RADIUS_KM = 6378.13646;
__device__ __constant__ double GPU_EARTH_OMEGA = 7.2921150e-5;   // rad/s

// -----------------------------------------------------------------------------
// Frame rotation helpers (acceleration only — Coriolis terms drop because
// accel is a 2nd derivative of position. We only need a 2D Z-rotation.)
// -----------------------------------------------------------------------------

__device__ __forceinline__ void
dev_gcrs_to_itrs(double rx, double ry, double rz,
                 const double* __restrict__ rot,
                 double& ox, double& oy, double& oz) {
    // Mirrors CPU `FrameRotation::to_itrs`. `rot` is the assembled 9-double
    // GCRS->ITRS matrix, row-major, resolved on the HOST by the sealed frame
    // authority and shipped whole. The kernel never reconstructs a rotation and
    // never sees an Earth-rotation angle: the IAU 2006/2000A chain carries
    // bias-precession-nutation, the equation of the origins and polar motion,
    // none of which a scalar angle can express.
    ox = fma(rot[0], rx, fma(rot[1], ry, rot[2] * rz));
    oy = fma(rot[3], rx, fma(rot[4], ry, rot[5] * rz));
    oz = fma(rot[6], rx, fma(rot[7], ry, rot[8] * rz));
}

__device__ __forceinline__ void
dev_itrs_to_gcrs(double ax, double ay, double az,
                 const double* __restrict__ rot,
                 double& ox, double& oy, double& oz) {
    // Mirrors CPU `FrameRotation::to_gcrs`: the transpose apply. Coriolis terms
    // vanish because the input is an acceleration, not a state.
    ox = fma(rot[0], ax, fma(rot[3], ay, rot[6] * az));
    oy = fma(rot[1], ax, fma(rot[4], ay, rot[7] * az));
    oz = fma(rot[2], ax, fma(rot[5], ay, rot[8] * az));
}

// -----------------------------------------------------------------------------
// Legendre recurrence coefficient helpers — mirror LegendreCoeffsSimd::new().
//
// CPU code precomputes pt1[l][m] and pt21_factor[l][m] once. On GPU we
// recompute inline per kernel call. Per-block cost ~O((order+2)^2) FMAs;
// negligible vs the recursion itself.
//
// Formulas from satpy_core::gravity::LegendreCoeffsSimd::new (Pines V/W
// normalized form):
//   For l >= 1, m >= 0:
//     pt1[l][m]  = (2*l - 1) / (l - m)                  if l > m
//     pt21[l][m] = (l + m - 1) / (l - m)                if l > m
// We multiply pt21 by c2_re (= RE^2 / r^2) at use site.
// -----------------------------------------------------------------------------

__device__ __forceinline__ double dev_pt1(int l, int m) {
    return (double)(2 * l - 1) / (double)(l - m);
}

__device__ __forceinline__ double dev_pt21_factor(int l, int m) {
    return (double)(l + m - 1) / (double)(l - m);
}

// -----------------------------------------------------------------------------
// Kernel: spherical_gravity_batch_dense
//
// Launch geometry:
//   gridDim.x = N (one block per state)
//   blockDim.x = 32 (one warp per block)
//
// Shared memory: none (Phase B1 keeps V/W in global vw_scratch slices for
// simplicity; Phase B2+ can move l-1/l-2 stripes into shared).
//
// Args:
//   states_eci  [N][6] row-major (uses [0..3] for position; velocity ignored)
//   rot_gcrs_to_itrs : [9] row-major GCRS->ITRS, assembled on the host by the
//                      sealed frame authority
//   order       : harmonic degree (must be <= MAX_REC - 3 = 128)
//   stride      : coeff stride (typically order + 1, must be <= MAX_REC)
//   c_coeffs    : [stride][stride] f64 row-major dense
//   s_coeffs    : [stride][stride] f64 row-major dense
//   accels_eci  : [N][3] row-major output
//   vw_scratch  : [N][2][MAX_REC][MAX_REC] f64 per-block scratch
// -----------------------------------------------------------------------------

__global__ void
spherical_gravity_batch_dense(const double* __restrict__ states_eci,
                              const double* __restrict__ rot_gcrs_to_itrs,
                              int order,
                              int stride,
                              const double* __restrict__ c_coeffs,
                              const double* __restrict__ s_coeffs,
                              double* __restrict__ accels_eci,
                              double* __restrict__ vw_scratch) {
    const int sidx = blockIdx.x;
    const int tid  = threadIdx.x;
    const int n_rec = order + 2;
    if (n_rec > MAX_REC) {
        // Safety: don't overrun scratch. Caller must clamp order.
        if (tid == 0) {
            accels_eci[3 * sidx + 0] = 0.0;
            accels_eci[3 * sidx + 1] = 0.0;
            accels_eci[3 * sidx + 2] = 0.0;
        }
        return;
    }

    // Per-block V/W slices.
    const long stride_per_block = (long)2 * MAX_REC * MAX_REC;
    double* v = vw_scratch + (long)sidx * stride_per_block;
    double* w = v + (long)MAX_REC * MAX_REC;
    auto V = [&](int l, int m) -> double& { return v[(long)l * MAX_REC + m]; };
    auto W = [&](int l, int m) -> double& { return w[(long)l * MAX_REC + m]; };

    // -- Load + rotate state to ECEF (thread 0 broadcasts via shared) --
    __shared__ double pos_ecef_sh[3];
    if (tid == 0) {
        const double rx = states_eci[6 * sidx + 0];
        const double ry = states_eci[6 * sidx + 1];
        const double rz = states_eci[6 * sidx + 2];
        double ex, ey, ez;
        dev_gcrs_to_itrs(rx, ry, rz, rot_gcrs_to_itrs, ex, ey, ez);
        pos_ecef_sh[0] = ex;
        pos_ecef_sh[1] = ey;
        pos_ecef_sh[2] = ez;
    }
    __syncthreads();

    const double pos_x = pos_ecef_sh[0];
    const double pos_y = pos_ecef_sh[1];
    const double pos_z = pos_ecef_sh[2];

    const double r2     = pos_x * pos_x + pos_y * pos_y + pos_z * pos_z;
    const double r      = sqrt(r2);
    const double re     = GPU_GRAVITY_REFERENCE_RADIUS_KM;
    const double c2     = re / r2;
    const double c2_re  = c2 * re;
    const double x_c2   = pos_x * c2;
    const double y_c2   = pos_y * c2;
    const double z_c2   = pos_z * c2;

    // -- V/W bootstrap (thread 0) --
    if (tid == 0) {
        const double v00 = sqrt(c2_re);
        V(0, 0) = v00;
        W(0, 0) = 0.0;
        W(1, 0) = 0.0;
        V(1, 0) = z_c2 * v00;
    }
    __syncthreads();

    // -- V/W recursion: outer l, inner m striped across warp --
    // m=0 column is sequential in l (CPU pattern) — thread 0 owns it.
    // Diagonals v[m][m] depend on v[m-1][m-1] — also chained; thread 0 fills
    // diagonals as they appear in the row-major sweep.
    //
    // Off-diagonal v[l][m] for 1 <= m < l only depends on v[l-1][m] and
    // v[l-2][m] (both in already-finished rows), so all warp lanes fan out
    // across m for a given l.

    // m=0 column for l in [2..n_rec)
    if (tid == 0) {
        for (int l = 2; l < n_rec; ++l) {
            W(l, 0) = 0.0;
            const double pt1   = dev_pt1(l, 0);
            const double pt21  = dev_pt21_factor(l, 0) * c2_re;
            V(l, 0) = fma(pt1, z_c2 * V(l - 1, 0), -pt21 * V(l - 2, 0));
        }
    }
    __syncthreads();

    // Phase B2 P2.1: l-major outer + m-parallel inner V/W recursion.
    //
    // Within row l, V[l][m] for ALL m in [0..l] depends only on V[l-1][*]
    // and V[l-2][*] (previous rows). The diagonal V[l][l] depends on the
    // diagonal V[l-1][l-1] (previous row's chain); the sub-diagonal
    // V[l][l-1] also depends on V[l-1][l-1]; off-diagonals V[l][m] for
    // m in [1..=l-2] use V[l-1][m] and V[l-2][m]. No intra-row writes
    // race because each lane handles a distinct m.
    //
    // m=0 column was already filled by lane 0 above. The diagonal +
    // sub-diagonal (m=l and m=l-1) are kept on lane 0 because they
    // share the same V[l-1][l-1] / W[l-1][l-1] read and the formulas
    // are cheap. Off-diagonals m in [1..=l-2] stripe across the warp's
    // 32 lanes.
    //
    for (int l = 1; l < n_rec; ++l) {
        // Off-diagonals m in [1..=l-2] — striped across warp lanes.
        // Reads from rows l-1, l-2 only (fully written in prior iters).
        if (l >= 3) {
            for (int m = 1 + tid; m <= l - 2; m += 32) {
                const double pt1  = dev_pt1(l, m);
                const double pt21 = dev_pt21_factor(l, m) * c2_re;
                V(l, m) = fma(pt1, z_c2 * V(l - 1, m), -pt21 * V(l - 2, m));
                W(l, m) = fma(pt1, z_c2 * W(l - 1, m), -pt21 * W(l - 2, m));
            }
        }
        // Sub-diagonal V[l][l-1] (if l >= 2) + sectoral V[l][l]:
        // both read V[l-1][l-1] / W[l-1][l-1] only. Lane 0 only.
        if (tid == 0) {
            if (l >= 2) {
                const double pt1 = 2.0 * (double)l - 1.0;
                V(l, l - 1) = pt1 * z_c2 * V(l - 1, l - 1);
                W(l, l - 1) = pt1 * z_c2 * W(l - 1, l - 1);
            }
            const double c1 = 2.0 * (double)l - 1.0;
            const double v_prev = V(l - 1, l - 1);
            const double w_prev = W(l - 1, l - 1);
            V(l, l) = c1 * (x_c2 * v_prev - y_c2 * w_prev);
            W(l, l) = c1 * (x_c2 * w_prev + y_c2 * v_prev);
        }
        __syncwarp();
    }
    __syncthreads();

    // -- Summation. CPU pattern:
    //   for l in 0..=order:
    //     m=0 term (zonal, scalar)
    //     for m in 1..=l: tesseral terms
    //
    // We fan out across m for each l. Warp-reduce ax/ay/az at the end.
    const double coef_sph = GPU_MU / (re * re);

    double ax = 0.0, ay = 0.0, az = 0.0;

    for (int l = 0; l <= order; ++l) {
        const int base = l * stride;
        const double cl0 = c_coeffs[base];
        // Zonal (m=0) term — thread 0 only, avoids over-counting.
        if (tid == 0) {
            const double v_lp1_0 = V(l + 1, 0);
            const double v_lp1_1 = V(l + 1, 1);
            const double w_lp1_1 = W(l + 1, 1);
            ax += coef_sph * (-cl0 * v_lp1_1);
            ay += coef_sph * (-cl0 * w_lp1_1);
            az += coef_sph * ((double)(l + 1) * (-cl0 * v_lp1_0));
        }
        // Tesseral m in [1..l] across the warp.
        for (int m = 1 + tid; m <= l; m += blockDim.x) {
            const double clm = c_coeffs[base + m];
            const double slm = s_coeffs[base + m];
            const double v_lp1_mp1 = V(l + 1, m + 1);
            const double v_lp1_m   = V(l + 1, m);
            const double v_lp1_mm1 = V(l + 1, m - 1);
            const double w_lp1_mp1 = W(l + 1, m + 1);
            const double w_lp1_m   = W(l + 1, m);
            const double w_lp1_mm1 = W(l + 1, m - 1);
            const double d = (double)(l - m);
            const double cf_2 = (d + 2.0) * (d + 1.0);

            const double x1 = fma(-clm, v_lp1_mp1,
                                  fma(-slm, w_lp1_mp1,
                                      cf_2 * fma(clm, v_lp1_mm1, slm * w_lp1_mm1)));
            const double y1 = fma(-clm, w_lp1_mp1,
                                  fma(slm, v_lp1_mp1,
                                      cf_2 * fma(-clm, w_lp1_mm1, slm * v_lp1_mm1)));
            const double z1 = (double)(l - m + 1) *
                              fma(-clm, v_lp1_m, -slm * w_lp1_m);

            ax += coef_sph * 0.5 * x1;
            ay += coef_sph * 0.5 * y1;
            az += coef_sph * z1;
        }
    }

    // Warp reduction (ordered shuffle tree).
    for (int offset = 16; offset > 0; offset >>= 1) {
        ax += __shfl_xor_sync(0xffffffffu, ax, offset);
        ay += __shfl_xor_sync(0xffffffffu, ay, offset);
        az += __shfl_xor_sync(0xffffffffu, az, offset);
    }

    if (tid == 0) {
        double ax_eci, ay_eci, az_eci;
        dev_itrs_to_gcrs(ax, ay, az, rot_gcrs_to_itrs,
                              ax_eci, ay_eci, az_eci);
        accels_eci[3 * sidx + 0] = ax_eci;
        accels_eci[3 * sidx + 1] = ay_eci;
        accels_eci[3 * sidx + 2] = az_eci;
    }
}

} // extern "C"
