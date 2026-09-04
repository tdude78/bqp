//! Bit pin on `jb2008_density_fitted_v7` — `atm_model` 7, the FLOWN atmosphere.
//!
//! # Why this file exists
//!
//! Until this file, nothing in the tree bit-detected a change to the density
//! the campaign actually integrates against.
//!
//! * `lightyear_odeint_rs/tests/rect_loop_pin.rs` is the tree's other true
//!   digest detector and it hardcodes `atm_model: 4`, the exact Orekit profile.
//!   Its three digests cannot move for a model-7 change, by construction — see
//!   its own "NOT RE-BASELINED 2026-08-11" ledger entry, which records that
//!   silence.
//! * `strict_hf_pin`'s `strict_hf_v3_production_arc_is_pinned` flies compiled
//!   model 8. The fitted v7 kernel remains beneath model 8, but this density pin
//!   does not itself pin persistence-driver authority. The arc pin is a 1 cm
//!   endpoint tripwire (`V3_POS_TOL_KM = 1.0e-5` km), not a bit
//!   test. The last three atmosphere re-baselines moved that endpoint 6.13, 7.56
//!   and 4.53 µm — 1,300x to 2,200x under the tripwire. A density edit that
//!   lands at µm scale on the arc passes it green.
//! * `tests/jb2008_libm_probe.rs`'s `dump_fitted_profile_density_bits` sees the
//!   bits, but it is a PRINTER. It detects a change only when a human diffs two
//!   trees; nothing reds if nobody looks.
//!
//! This file closes open item #11: a silent µm-scale change to the flown
//! atmosphere now reds a test instead of shipping.
//!
//! # What it detects, and what it does NOT
//!
//! DETECTS: any movement in the raw `f64` bits returned by the model-7 density
//! over a fixed grid — a fitted coefficient, the fit degree or domain, the
//! quadrature, the thermal chain, `wrap_to_tau`, the shared species core, the
//! input gates, and (for the second case only) the sealed SET driver table.
//! One flipped mantissa bit on one row of ~16k moves a digest.
//!
//! Does NOT detect: accuracy. A digest says the number changed, never that the
//! new one is worse — the fit's own truncation budget lives with
//! `LogQuadratureFittedV7`, and the arc-level consequence lives with
//! `strict_hf_pin`'s `V3_PINNED_POS_KM`. A green run here is not a licence to
//! skip re-baselining that.
//!
//! # Two cases, because they localise differently
//!
//! * `KERNEL` — every solar/geomagnetic driver is a literal in this file, so
//!   the digest is a function of the kernel arithmetic ALONE. Moves only if the
//!   code moved.
//! * `DRIVERS` — the same geometry with drivers looked up from
//!   `compiled_drivers()` at 16 epochs spread across the sealed SOLFSMY/DTCFILE
//!   coverage. Moves if the code moved OR the compiled driver table did.
//!
//! `KERNEL` green with `DRIVERS` red is a data-authority change; both red is a
//! kernel change. A `GRID` digest over the raw input bits is pinned alongside
//! each, so an edit to the grid itself cannot be mistaken for an edit to the
//! physics.
//!
//! # NOT debug-ignored, and that was MEASURED, not assumed
//!
//! Every other bit pin in this tree carries a not-in-the-bitpin-lane ignore
//! (today `#[cfg_attr(not(feature = "bitpin"), ignore)]`; formerly keyed on
//! `debug_assertions`), and this file was written with the same attribute on both digest
//! tests. `scripts/digest-profile-sweep.sh jb_rs/fitted_v7_density_pin` then
//! reported them **SLACK** — "debug-ignored, but passes in dev too" — twice, so
//! the attribute came back off.
//!
//! Both digests are IDENTICAL under `cargo test` and `cargo test --release`.
//! That is consistent with what the workspace already knows and does not
//! contradict `strict_hf_pin`: `-C llvm-args=-fp-contract=on` is codegen-
//! identical to `=off` on Rust IR, and this kernel's fusions are written as
//! explicit `f64::mul_add`, which lowers to `llvm.fma` at every optimisation
//! level. There is no contraction axis between the profiles to move these bits.
//!
//! Keeping the attribute would have cost the detector its coverage in the
//! profile people actually iterate in, in exchange for guarding against a
//! divergence that does not exist. If a later kernel edit introduces one, the
//! sweep reports DIVERGENT instead of SLACK — that is a finding, and the fix is
//! to add the attribute back with the measurement that justified it.
//!
//! ```sh
//! cargo test --release -p jb_rs --test fitted_v7_density_pin
//! ```
//!
//! # PER-LIBM
//!
//! The pinned values were captured on macOS/arm64 (Apple libm), which is one of
//! the three bit classes this workspace spans — Apple libm, glibc/znver-N, and
//! the `wide::f64x4::reduce_add` AVX-vs-non-AVX summation tree. The kernel
//! calls `exp`, `log`, `sin`, `cos` and `atan2` on every row and JB2008's
//! temperature quadrature runs through `f64x4`, so BOTH axes are live here.
//! Expect these digests to differ on a glibc host; that is the pin's known
//! scope, not a defect, and it is the same scope every other bit pin here has.
//! Re-baselining is a verbatim copy of the `FITTED_V7_PIN` lines this file
//! prints, on the host that captured them.
//!
//! Companion probe for localising a red: `tests/jb2008_libm_probe.rs`, whose
//! per-profile dumps name WHICH rows moved and by how much.

use std::collections::HashSet;

use jb_rs::drivers::{compiled_drivers, UtcModifiedJulianDay};
use jb_rs::jb2008::{
    jb2008_density_fitted_v7, jb2008_density_logquad_x4_approx_v2, Jb2008Error, Jb2008Input,
};

// ---------------------------------------------------------------------------
// Grid axes
// ---------------------------------------------------------------------------

/// Ellipsoidal altitudes in km, in evaluation order.
///
/// Chosen to straddle every boundary the flown profile switches on rather than
/// to be evenly spaced: the 90/105 km segment joints, the 500 km frozen-plan
/// floor, the 985.7 km censused production ceiling, the 1000 km upper-fit
/// domain top, the 1006.876 km one-panel/two-panel boundary that domain exists
/// to stay inside, and 2600 km — ABOVE the 2500 km extrapolation ceiling at
/// which `LightyearRHS` stops believing the model, so the pin covers the
/// extrapolated tail the adapter clamps rather than stopping where the campaign
/// stops.
const ALTITUDES_KM: [f64; 36] = [
    90.0,
    95.0,
    100.0,
    104.999_999,
    105.0,
    105.000_001,
    120.0,
    130.0,
    145.0,
    160.0,
    180.0,
    200.0,
    225.0,
    250.0,
    275.0,
    300.0,
    350.0,
    400.0,
    450.0,
    499.999_999,
    500.0,
    550.0,
    600.0,
    626.2,
    700.0,
    800.0,
    900.0,
    985.7,
    1000.0,
    1006.876,
    1010.0,
    1100.0,
    1250.0,
    1500.0,
    2000.0,
    2600.0,
];

/// Geocentric latitudes in radians, out to the poles the kernel accepts.
const LATITUDES_RAD: [f64; 7] = [-1.55, -1.02, -0.47, 0.0, 0.31, 0.88, 1.55];

/// Satellite-minus-Sun hour angles in radians. **NEGATIVE VALUES ARE LOAD-
/// BEARING** — see below.
///
/// This is the only longitude the kernel consumes: `rhs.rs` reduces the
/// satellite and Sun right ascensions to their difference before the call, so
/// sweeping the difference sweeps the local-solar-time axis exactly.
///
/// The kernel normalises that difference with `wrap_to_tau`, which has three
/// arms: the identity on `[0, TAU)`, a single `x + TAU` addition on
/// `(-TAU, 0)`, and a `rem_euclid` fallback for everything else. **Production
/// takes the middle arm about half the time.** `rhs.rs` builds the angle with
/// `atan2`, whose range is `[-PI, PI]`, so every westward geometry arrives
/// negative.
///
/// This array's first revision was `[0.0, 1.05, 2.4, 3.3, 4.71, 6.02]` — all
/// non-negative, so all six took the identity arm and the addition arm was
/// never pinned. An edit to it was invisible to this file. Three of the six are
/// now negative. Found in review; the poison in the ledger below re-proves it.
///
/// `4.71` and `6.02` are retained above `PI` deliberately: they are outside
/// what `rhs.rs` can hand the kernel, but this is a `pub` kernel whose callers
/// are unvalidated, and they hold the identity arm's upper half.
///
/// The `rem_euclid` fallback stays UNPINNED on purpose. It is reachable only
/// at `|x| >= TAU` (and at the non-finite inputs `validate` rejects first), so
/// no production caller can take it; it exists as a `pub`-kernel guard. Pinning
/// it would pin `f64::rem_euclid`, which is a std contract, not this kernel's.
const HOUR_ANGLES_RAD: [f64; 6] = [-2.51, -1.05, 0.0, 1.05, 2.4, 4.71];

/// One epoch's solar and geomagnetic state, as the kernel consumes it.
///
/// Post-lag, post-average values: the kernel applies no lags of its own.
#[derive(Clone, Copy)]
struct DriverState {
    mjd_utc: f64,
    sun_declination_rad: f64,
    f10: f64,
    f10b: f64,
    s10: f64,
    s10b: f64,
    m10: f64,
    m10b: f64,
    y10: f64,
    y10b: f64,
    dst_temperature_correction_k: f64,
}

/// Eight literal driver states for the `KERNEL` case.
///
/// The first six are physical, ordered quiet to storm, with declinations
/// sweeping a full seasonal range so the subsolar geometry is not frozen across
/// them. The last two are NOT physical and are not meant to be — they exist to
/// straddle a code boundary.
///
/// The fitted kernel is a degree-14 series in the EXOSPHERIC TEMPERATURE over
/// `[FITTED_V7_TEXO_LO, FITTED_V7_TEXO_HI]` = `[500, 2600] K`, and outside that
/// interval every fitted accessor falls back to walking the real plan. All six
/// physical states — up to and including the `f10 = 340` storm — stay INSIDE
/// that interval, which was measured rather than assumed: on the storm state
/// model 7 differs bitwise from model 6 on every row above 105 km, so the fit is
/// evaluated throughout. A grid of physical states alone therefore never
/// executes the fallback arm, and a change to either domain constant would move
/// nothing and stay green.
///
/// So the last two states drive the exospheric temperature off each end:
/// `f10 = 405` with a 500 K DST correction goes above 2600 K on part of the
/// grid, and `f10 = 2` goes below 500 K on most of it. `assert_not_vacuous`
/// requires the fallback arm to be reached above 120 km — i.e. by temperature
/// and not merely by the sub-105 km altitudes where the fixed plans are unused
/// either way — so deleting these two states reds the census instead of quietly
/// halving what the digest guards.
///
/// The above-domain state also rejects a minority of its rows with
/// `NumericalDomain`. That is kept, not tuned away: those rejections are hashed
/// into the digest like any other outcome, and they are this grid's only
/// coverage of the kernel's numerical reject path.
const KERNEL_STATES: [DriverState; 8] = [
    DriverState {
        mjd_utc: 52_951.003_805_740_744,
        sun_declination_rad: -0.285_987_757_544_287,
        f10: 91.0,
        f10b: 137.1,
        s10: 108.8,
        s10b: 123.8,
        m10: 116.7,
        m10b: 128.5,
        y10: 168.0,
        y10b: 138.6,
        dst_temperature_correction_k: 43.0,
    },
    DriverState {
        mjd_utc: 55_197.25,
        sun_declination_rad: 0.409_1,
        f10: 68.4,
        f10b: 71.2,
        s10: 62.5,
        s10b: 65.9,
        m10: 70.1,
        m10b: 72.8,
        y10: 74.6,
        y10b: 76.3,
        dst_temperature_correction_k: -12.0,
    },
    DriverState {
        mjd_utc: 57_388.625,
        sun_declination_rad: -0.409_1,
        f10: 122.5,
        f10b: 110.4,
        s10: 118.2,
        s10b: 105.7,
        m10: 131.9,
        m10b: 119.3,
        y10: 127.4,
        y10b: 114.8,
        dst_temperature_correction_k: 8.5,
    },
    DriverState {
        mjd_utc: 59_810.0,
        sun_declination_rad: 0.201_7,
        f10: 154.3,
        f10b: 141.6,
        s10: 149.8,
        s10b: 138.2,
        m10: 161.5,
        m10b: 150.9,
        y10: 158.2,
        y10b: 147.1,
        dst_temperature_correction_k: 24.0,
    },
    DriverState {
        mjd_utc: 60_115.875,
        sun_declination_rad: -0.104_2,
        f10: 212.7,
        f10b: 195.4,
        s10: 205.1,
        s10b: 188.6,
        m10: 224.8,
        m10b: 206.3,
        y10: 218.5,
        y10b: 199.7,
        dst_temperature_correction_k: 76.0,
    },
    DriverState {
        mjd_utc: 60_190.5,
        sun_declination_rad: 0.365_4,
        f10: 340.0,
        f10b: 315.0,
        s10: 330.0,
        s10b: 305.0,
        m10: 355.0,
        m10b: 325.0,
        y10: 345.0,
        y10b: 318.0,
        dst_temperature_correction_k: 260.0,
    },
    // Above the fit's temperature domain on part of the grid. Unphysical by
    // construction; see the table's doc comment.
    DriverState {
        mjd_utc: 59_950.375,
        sun_declination_rad: -0.184_3,
        f10: 405.0,
        f10b: 378.0,
        s10: 391.0,
        s10b: 372.0,
        m10: 432.0,
        m10b: 405.0,
        y10: 426.0,
        y10b: 396.0,
        dst_temperature_correction_k: 500.0,
    },
    // Below the fit's temperature domain on most of the grid. Also unphysical.
    DriverState {
        mjd_utc: 60_000.125,
        sun_declination_rad: 0.062_5,
        f10: 2.0,
        f10b: 2.0,
        s10: 2.0,
        s10b: 2.0,
        m10: 2.0,
        m10b: 2.0,
        y10: 2.0,
        y10b: 2.0,
        dst_temperature_correction_k: 0.0,
    },
];

/// Sixteen UTC MJD epochs spread across the compiled SOLFSMY/DTCFILE coverage,
/// paired with a solar declination for that time of year.
///
/// Coverage is JD 2450450 to 2461195 on BOTH files, i.e. **MJD 50449.5 to
/// 61194.5**. Read back from `compiled_identity()` rather than restated: the
/// first revision of this comment wrote the end as 60194.5, a thousand days
/// short, and used that wrong bound to justify stopping the table early. These
/// epochs sit clear of both ends so the five-day Y10 lag and the next-day DTC
/// row are always available; a lookup failure is an `expect` and not a skip,
/// because a driver table that stopped covering these epochs is a finding.
///
/// `60310.0` is JD 2460310.5 — the epoch `rect_loop_pin` and `strict_hf_pin`
/// both fly, and the one epoch here that those two gates share. It replaced
/// `59810.0` = JD 2459810.5, which was five hundred days earlier still.
/// Verified against the compiled table rather than re-derived: the lookup at
/// 60310.0 returns `f10 = 141.4`, so it is inside coverage with 884 days to
/// spare.
///
/// It is NOT the sealed V3 arc epoch, and two earlier revisions of this comment
/// said it was. The authorized V3 arc is JD 2461267.975 to 2461284.225
/// (`PART_A_V3_AUTHORIZED_{START,END}_JD` in `jb_rs::drivers`); 2460310.5 is
/// 2024-01-01, 957 days before the arc opens. Nor could it be the arc epoch for
/// the configuration that uses it: it is reached through `atm_model: 4`, which
/// resolves to the compiled SET v2 table, and that table's SOLFSMY and DTC
/// coverage both end at JD 2461195.0
/// (pinned in `compiled_provider.rs`) — 73 days before the arc begins. Model 4
/// and the V3 arc have no overlap in either direction.
///
/// SCOPE, because the previous revision of this note overstated it:
/// `atm_model: 4` is what `rect_loop_pin` hardcodes throughout, and what
/// `strict_hf_pin` uses for the LEGACY config at its own `JD0` -- it is not
/// what `strict_hf_pin` IS. That file also defines `V3_JD0` = 2461270.225,
/// which sits INSIDE the authorized arc, 2.25 days after it opens, and flies
/// it under the compiled `atmosphere_model` (8), read from `part_a_hybrid()`
/// rather than restated. So "these pins cannot reach the V3 arc" is true of
/// model 4 and false of that file.
///
/// The distinction is load-bearing rather than pedantic: `strict_hf_pin` DID
/// once fly an unauthorized epoch -- its V3 tests ran at `JD0` and failed
/// with "lookup outside Part A v3 authorized persistence arc" until `V3_JD0`
/// was introduced. A reader who believes the file is model-4-only concludes
/// it can never touch the arc, and so never checks the one thing that has
/// already broken once.
///
/// The claim is corrected rather than deleted because a comment asserting
/// coverage the tables do not have is what lets a pin quietly stop measuring:
/// a reader who believes this epoch is on-arc will not check when the arc
/// moves.
///
/// `51100.5` is retained deliberately even though the whole epoch is REJECTED:
/// the compiled SOLFSMY carries `S10 = 0.0` there, which the kernel's
/// `NonPositiveSolarIndex` gate refuses. That is a real property of the sealed
/// table — S10 begins later than F10 — and hashing the rejection pins it, so a
/// re-issued table that backfilled those columns moves the `DRIVERS` digest.
const DRIVER_EPOCHS: [(f64, f64); 16] = [
    (50_500.5, 0.352_1),
    (51_100.5, -0.101_4),
    (51_800.5, 0.408_2),
    (52_500.5, -0.395_7),
    (53_200.5, 0.213_9),
    (53_900.5, -0.302_6),
    (54_600.5, 0.087_3),
    (55_300.5, 0.401_5),
    (56_000.5, -0.409_0),
    (56_700.5, 0.156_8),
    (57_400.5, -0.221_4),
    (58_100.5, 0.374_2),
    (58_800.5, -0.048_9),
    (59_500.5, 0.290_6),
    // 2024-01-01, four days past the solstice: declination is near its southern
    // extreme, which is why this one is not in the seasonal spread above.
    (60_310.0, -0.401_9),
    (60_100.5, 0.118_7),
];

/// Latitudes for the `DRIVERS` case. A subset of [`LATITUDES_RAD`], because
/// that case pays for 16 epochs on the same altitude ladder.
const DRIVER_LATITUDES_RAD: [f64; 4] = [-1.41, -0.36, 0.35, 1.41];

/// Hour angles for the `DRIVERS` case. See [`DRIVER_LATITUDES_RAD`], and see
/// [`HOUR_ANGLES_RAD`] for why one of the three is negative.
const DRIVER_HOUR_ANGLES_RAD: [f64; 3] = [-2.62, 0.52, 2.75];

// ---------------------------------------------------------------------------
// Grid construction
// ---------------------------------------------------------------------------

fn row(
    state: DriverState,
    altitude_km: f64,
    latitude_rad: f64,
    hour_angle_rad: f64,
) -> Jb2008Input {
    Jb2008Input {
        mjd_utc: state.mjd_utc,
        sun_declination_rad: state.sun_declination_rad,
        hour_angle_rad,
        sat_geocentric_lat_rad: latitude_rad,
        sat_altitude_m: altitude_km * 1000.0,
        f10: state.f10,
        f10b: state.f10b,
        s10: state.s10,
        s10b: state.s10b,
        m10: state.m10,
        m10b: state.m10b,
        y10: state.y10,
        y10b: state.y10b,
        dst_temperature_correction_k: state.dst_temperature_correction_k,
    }
}

/// `KERNEL`: 8 states x 36 altitudes x 7 latitudes x 6 hour angles.
fn kernel_rows() -> Vec<Jb2008Input> {
    let mut rows = Vec::with_capacity(KERNEL_ROWS);
    for state in KERNEL_STATES {
        for altitude_km in ALTITUDES_KM {
            for latitude_rad in LATITUDES_RAD {
                for hour_angle_rad in HOUR_ANGLES_RAD {
                    rows.push(row(state, altitude_km, latitude_rad, hour_angle_rad));
                }
            }
        }
    }
    rows
}

/// `DRIVERS`: 16 sealed-table epochs x 36 altitudes x 4 latitudes x 3 hour
/// angles.
#[expect(
    clippy::expect_used,
    reason = "test-support helper in an integration test file, where `allow-expect-in-tests` \
does not reach a free fn; a compiled driver authority that fails to load or an epoch it no \
longer covers must abort the pin rather than silently shrink the grid it measures"
)]
fn driver_rows() -> Vec<Jb2008Input> {
    let drivers = compiled_drivers().expect("compiled SET drivers");
    let mut rows = Vec::with_capacity(DRIVER_ROWS);
    for (mjd_utc, sun_declination_rad) in DRIVER_EPOCHS {
        let mjd = UtcModifiedJulianDay::new(mjd_utc).expect("finite UTC MJD");
        let driver = drivers
            .lookup_utc_mjd(mjd)
            .expect("compiled driver table covers this epoch");
        let state = DriverState {
            mjd_utc,
            sun_declination_rad,
            f10: driver.f10,
            f10b: driver.f10b,
            s10: driver.s10,
            s10b: driver.s10b,
            m10: driver.m10,
            m10b: driver.m10b,
            y10: driver.y10,
            y10b: driver.y10b,
            dst_temperature_correction_k: f64::from(driver.dtcval),
        };
        for altitude_km in ALTITUDES_KM {
            for latitude_rad in DRIVER_LATITUDES_RAD {
                for hour_angle_rad in DRIVER_HOUR_ANGLES_RAD {
                    rows.push(row(state, altitude_km, latitude_rad, hour_angle_rad));
                }
            }
        }
    }
    rows
}

/// Derived from the axes rather than written down, so adding an altitude or a
/// state cannot leave the shape assertion pointing at the old grid — the exact
/// desync `rect_loop_pin::SEGMENTS` exists to prevent one file over.
const KERNEL_ROWS: usize =
    KERNEL_STATES.len() * ALTITUDES_KM.len() * LATITUDES_RAD.len() * HOUR_ANGLES_RAD.len();
const DRIVER_ROWS: usize = DRIVER_EPOCHS.len()
    * ALTITUDES_KM.len()
    * DRIVER_LATITUDES_RAD.len()
    * DRIVER_HOUR_ANGLES_RAD.len();

// ---------------------------------------------------------------------------
// Digest
// ---------------------------------------------------------------------------

/// FNV-1a over raw `u64` words, little-endian, in order.
///
/// The same three lines `rect_loop_pin::digest` uses, and for the same reason:
/// nothing here needs a cryptographic property, only that a single flipped
/// mantissa bit anywhere in ~16k values changes the printed number. Taking
/// `u64` rather than `f64` lets a rejected row contribute its error variant
/// instead of being dropped.
fn digest(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for word in words {
        for byte in word.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// Word contributed by a REJECTED row.
///
/// A quiet-NaN payload, so it cannot collide with an accepted density:
/// `jb2008_density_with_profile` returns `Ok` only for finite positive values.
/// Rejections are hashed rather than skipped because "this input stopped being
/// rejected" is exactly as much of a change as "this density moved", and a pin
/// that silently drops error rows would not see it.
const ERROR_WORD_BASE: u64 = 0x7ff8_0000_dead_0000;

const fn word_of(outcome: Result<f64, Jb2008Error>) -> u64 {
    match outcome {
        Ok(rho) => rho.to_bits(),
        Err(Jb2008Error::NonFiniteInput) => ERROR_WORD_BASE | 1,
        Err(Jb2008Error::AltitudeOutOfRange) => ERROR_WORD_BASE | 2,
        Err(Jb2008Error::AngleOutOfRange) => ERROR_WORD_BASE | 3,
        Err(Jb2008Error::NonPositiveSolarIndex) => ERROR_WORD_BASE | 4,
        Err(Jb2008Error::NumericalDomain) => ERROR_WORD_BASE | 5,
    }
}

/// Every field of an input, in declaration order, as raw bits.
const fn input_words(input: Jb2008Input) -> [u64; 14] {
    [
        input.mjd_utc.to_bits(),
        input.sun_declination_rad.to_bits(),
        input.hour_angle_rad.to_bits(),
        input.sat_geocentric_lat_rad.to_bits(),
        input.sat_altitude_m.to_bits(),
        input.f10.to_bits(),
        input.f10b.to_bits(),
        input.s10.to_bits(),
        input.s10b.to_bits(),
        input.m10.to_bits(),
        input.m10b.to_bits(),
        input.y10.to_bits(),
        input.y10b.to_bits(),
        input.dst_temperature_correction_k.to_bits(),
    ]
}

// ---------------------------------------------------------------------------
// Survey
// ---------------------------------------------------------------------------

/// What one grid produced: the two digests plus the census that proves the
/// grid was not vacuous.
struct Survey {
    rows: usize,
    accepted: usize,
    rejected: usize,
    distinct: usize,
    /// Rows where model 7 differs BITWISE from model 6, i.e. the degree-14 fit
    /// was actually evaluated.
    fit_engaged: usize,
    /// Rows where model 7 reproduces model 6 bit for bit.
    fit_fallback: usize,
    /// Accepted rows at or above [`FIT_PLAN_FLOOR_KM`] where model 7 still
    /// reproduces model 6 bit for bit.
    ///
    /// Separated from `fit_fallback` because the two have different causes and
    /// only one of them is interesting. Below 105 km the fixed lower plan is
    /// not used by either profile, so the two agree for a reason that has
    /// nothing to do with the fit; above it, agreement means the exospheric
    /// temperature left `[500, 2600] K` and every fitted accessor walked the
    /// real plan. Counting only the second is what lets the census assert that
    /// the domain boundary is actually straddled.
    domain_fallback: usize,
    /// Accepted rows whose hour angle is in `(-TAU, 0)`, i.e. rows that take
    /// `wrap_to_tau`'s `x + TAU` arm.
    ///
    /// A counter rather than a comment because the coverage hole this closes
    /// was invisible: the first revision's hour angles were all non-negative,
    /// so every row took the identity arm and an edit to the addition arm moved
    /// no digest. Production reaches that arm on roughly half its calls
    /// (`rhs.rs` builds the angle with `atan2`, range `[-PI, PI]`). Asserted
    /// non-zero below, so a later edit that drops the negatives reds the census
    /// instead of quietly re-opening the hole.
    negative_wrap: usize,
    min_density: f64,
    max_density: f64,
    grid_digest: u64,
    density_digest: u64,
}

fn survey(rows: &[Jb2008Input]) -> Survey {
    let mut grid_words = Vec::with_capacity(rows.len().saturating_mul(14));
    let mut density_words = Vec::with_capacity(rows.len());
    let mut distinct: HashSet<u64> = HashSet::new();
    let mut accepted: usize = 0;
    let mut rejected: usize = 0;
    let mut fit_engaged: usize = 0;
    let mut fit_fallback: usize = 0;
    let mut domain_fallback: usize = 0;
    let mut negative_wrap: usize = 0;
    let mut min_density = f64::INFINITY;
    let mut max_density = 0.0_f64;

    for input in rows.iter().copied() {
        grid_words.extend(input_words(input));
        let outcome = jb2008_density_fitted_v7(input);
        let fitted = word_of(outcome);
        let coarse = word_of(jb2008_density_logquad_x4_approx_v2(input));
        if fitted == coarse {
            fit_fallback = fit_fallback.saturating_add(1);
        } else {
            fit_engaged = fit_engaged.saturating_add(1);
        }
        if let Ok(rho) = outcome {
            accepted = accepted.saturating_add(1);
            min_density = min_density.min(rho);
            max_density = max_density.max(rho);
            if fitted == coarse && input.sat_altitude_m >= FIT_PLAN_FLOOR_KM * 1000.0 {
                domain_fallback = domain_fallback.saturating_add(1);
            }
            if input.hour_angle_rad < 0.0 && input.hour_angle_rad > -std::f64::consts::TAU {
                negative_wrap = negative_wrap.saturating_add(1);
            }
        } else {
            rejected = rejected.saturating_add(1);
        }
        distinct.insert(fitted);
        density_words.push(fitted);
    }

    Survey {
        rows: rows.len(),
        accepted,
        rejected,
        distinct: distinct.len(),
        fit_engaged,
        fit_fallback,
        domain_fallback,
        negative_wrap,
        min_density,
        max_density,
        grid_digest: digest(grid_words),
        density_digest: digest(density_words),
    }
}

impl Survey {
    fn report(&self, case: &str) {
        println!(
            "FITTED_V7_PIN case={case} rows={} accepted={} rejected={} distinct={} \
             fit_engaged={} fit_fallback={} domain_fallback={} negative_wrap={} \
             rho_min={:.17e} rho_max={:.17e} grid={:#018x} density={:#018x}",
            self.rows,
            self.accepted,
            self.rejected,
            self.distinct,
            self.fit_engaged,
            self.fit_fallback,
            self.domain_fallback,
            self.negative_wrap,
            self.min_density,
            self.max_density,
            self.grid_digest,
            self.density_digest
        );
    }
}

// ---------------------------------------------------------------------------
// Census floors — the silent-green guard
// ---------------------------------------------------------------------------

/// Floors every grid must clear before its digest means anything.
///
/// A digest over an empty, constant, or all-rejected sweep is a perfectly
/// stable number that guards nothing, and it stays green through any edit.
/// These are inequalities rather than pins, so they keep holding on a host
/// whose libm moves the digests — which is the case where a vacuous grid would
/// otherwise be hardest to notice, since the digest tests are red there anyway.
///
/// `MIN_DISTINCT_NUM`/`MIN_DISTINCT_DEN`: distinct densities as a fraction of
/// accepted rows. The grid varies four independent axes, so near-total
/// distinctness is expected; the floor is set well below the measured value so
/// it fails on collapse, not on a one-row coincidence.
const MIN_DISTINCT_NUM: usize = 9;
const MIN_DISTINCT_DEN: usize = 10;

/// Decades of dynamic range the accepted densities must span.
///
/// 90 km to 2600 km is more than fifteen decades of density in reality; a grid
/// that has collapsed onto one altitude, or onto the adapter's ceiling
/// constant, cannot clear ten.
const MIN_DENSITY_DECADES: f64 = 10.0;

/// Altitude in km above which model 6 and model 7 agreeing bitwise means the
/// exospheric temperature left the FIT'S DOMAIN, rather than meaning the fixed
/// plans were simply unused.
///
/// Above the kernel's 105 km segment joint both fitted plans are live, so the
/// two profiles can only agree by falling back. 120 rather than 105 leaves a
/// margin around the joint itself.
const FIT_PLAN_FLOOR_KM: f64 = 120.0;

/// Minimum share of rows the kernel must accept, as ninths.
///
/// Not ten-tenths: both grids deliberately contain rejected rows — the
/// above-domain state's `NumericalDomain` returns and the `DRIVERS` epoch whose
/// sealed S10 column is zero — and those rejections are coverage, hashed into
/// the digest like any other outcome. The floor only has to catch a grid that
/// has become mostly errors.
const MIN_ACCEPTED_NINTHS: usize = 8;

fn assert_not_vacuous(survey: &Survey, case: &str, expected_rows: usize) {
    assert_eq!(survey.rows, expected_rows, "{case}: grid shape moved");
    assert!(
        survey.accepted.saturating_mul(9) >= expected_rows.saturating_mul(MIN_ACCEPTED_NINTHS),
        "{case}: only {} of {} rows were accepted by the kernel; a grid that is \
         mostly rejections pins error codes, not densities",
        survey.accepted,
        survey.rows
    );
    assert!(
        survey.distinct.saturating_mul(MIN_DISTINCT_DEN)
            >= survey.accepted.saturating_mul(MIN_DISTINCT_NUM),
        "{case}: {} distinct densities over {} accepted rows — the grid has \
         collapsed and its digest no longer discriminates",
        survey.distinct,
        survey.accepted
    );
    assert!(
        survey.max_density / survey.min_density >= 10.0_f64.powf(MIN_DENSITY_DECADES),
        "{case}: accepted densities span only {:.2} decades ({:.3e} to {:.3e}); \
         the altitude axis is not reaching the kernel",
        (survey.max_density / survey.min_density).log10(),
        survey.min_density,
        survey.max_density
    );
    assert!(
        survey.fit_engaged > 0,
        "{case}: model 7 reproduced model 6 bit for bit on EVERY row, so the \
         degree-14 fit was never evaluated and this digest pins model 6"
    );
    assert!(
        survey.negative_wrap > 0,
        "{case}: every hour angle on this grid is non-negative, so every row \
         takes wrap_to_tau's IDENTITY arm and its `x + TAU` arm is unpinned — \
         while production reaches that arm on about half its calls, because \
         rhs.rs builds the angle with atan2 over [-PI, PI]. This is the exact \
         hole the first revision of this file shipped. Put negatives back in \
         HOUR_ANGLES_RAD / DRIVER_HOUR_ANGLES_RAD rather than deleting this."
    );
}

// ---------------------------------------------------------------------------
// Pinned digests. Copy replacements VERBATIM from the FITTED_V7_PIN lines.
// ---------------------------------------------------------------------------
//
// Captured 2026-08-11 on macOS/arm64 (Apple libm) at the commit that added this
// file, with `cargo test --release -p jb_rs --test fitted_v7_density_pin`.
//
// Re-pinning is a two-step obligation, not one:
//   1. Copy the new digest here, in the commit that moved it.
//   2. Re-baseline `strict_hf_pin`'s `V3_PINNED_POS_KM` even if its tripwire
//      stayed green. A sub-tripwire arc move left unpinned is budget quietly
//      spent, and the next change that DOES trip carries an unknown share of
//      someone else's residual.
// Localise WHICH rows moved with `tests/jb2008_libm_probe.rs`'s per-profile
// dumps before deciding a move was intended.
//
// ---------------------------------------------------------------------------
// RE-PINNED 2026-08-11 — GRID FIX, not a physics change. All four moved.
//
//   KERNEL  grid    0x700589ddc43ebba1 -> 0xb7aca9953b400fa1
//   KERNEL  density 0xacd32bb3d8ceefcc -> 0x106197c5e5fcfbac
//   DRIVERS grid    0x9dfd39c7375ee0bd -> 0xf7855cf37a7dbd6d
//   DRIVERS density 0xf74810cf10f44911 -> 0xebd29792d2f83bdd
//
// Cause: two defects in the ORIGINAL grid, found in review, neither of which
// the kernel had anything to do with. `jb2008` is untouched between the two
// captures, so the density digests moved only because the INPUTS did — which
// is why both grid digests moved with them. A density digest that had moved
// while its grid digest held would have been the alarming case.
//
//   1. Hour angles were all non-negative, so every row took `wrap_to_tau`'s
//      identity arm and its `x + TAU` arm — which production reaches on about
//      half its calls — was unpinned. Three of six KERNEL angles and one of
//      three DRIVERS angles are now negative, and `Survey::negative_wrap`
//      asserts it so the hole cannot silently re-open.
//   2. The epoch written as the sealed V3 arc's was `59810.0`, which is JD
//      2459810.5. Replaced by `60310.0` = JD 2460310.5. The comment's
//      driver-coverage bound was also a thousand days short and is now read
//      back from `compiled_identity()`.
//
//      NOTE: that replacement swapped one off-arc epoch for another while
//      keeping the "sealed V3 arc epoch" label, which was false of both. The
//      authorized arc is JD 2461267.975 to 2461284.225; 2460310.5 is 957 days
//      before it, and the `atm_model: 4` table THAT epoch is reached
//      through stops 73 days before it regardless. Scoped deliberately:
//      model 4 is `rect_loop_pin` throughout and `strict_hf_pin`'s legacy
//      `JD0` config only -- that file ALSO flies `V3_JD0` inside the arc
//      under the compiled atmosphere.
//      before it regardless. See the corrected note on `60310.0` above. No
//      epoch in this file was changed for that correction -- only the claim.
//
// `V3_PINNED_POS_KM` was NOT re-baselined for this, and correctly: no
// production trajectory is affected by which inputs a test file sweeps. The
// step-2 obligation above applies to KERNEL arithmetic changes, not to grid
// corrections. State which of the two a re-pin is.
//
// The hole was demonstrated rather than asserted. Poison: `wrap_to_tau`'s
// negative arm changed from `x + TAU` to `from_bits((x + TAU).to_bits() + 1)`,
// one ULP, on a branch no other arm can reach. Against THIS grid both density
// digests go red and both grid digests stay green. Against the OLD hour angles,
// with the same poison still in the kernel, the digests come back
// 0xacd32bb3d8ceefcc and 0xf74810cf10f44911 — bit for bit the values pinned
// before this commit, i.e. fully GREEN. The old grid could not see a real
// one-ULP kernel change on a branch production takes half the time.
// ---------------------------------------------------------------------------

const PIN_KERNEL_GRID: u64 = 0xb7ac_a995_3b40_0fa1;
const PIN_KERNEL_DENSITY: u64 = 0x1061_97c5_e5fc_fbac;

const PIN_DRIVERS_GRID: u64 = 0xf785_5cf3_7a7d_bd6d;
const PIN_DRIVERS_DENSITY: u64 = 0xebd2_9792_d2f8_3bdd;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The vacuity guard.
///
/// If this reds, the digest tests below are not measuring what their names say
/// however green they are.
#[test]
fn fitted_v7_grids_are_not_vacuous() {
    let kernel = survey(&kernel_rows());
    kernel.report("KERNEL-CENSUS");
    assert_not_vacuous(&kernel, "KERNEL", KERNEL_ROWS);
    assert!(
        kernel.domain_fallback > 0,
        "KERNEL: every accepted row above {FIT_PLAN_FLOOR_KM} km evaluated the \
         degree-14 fit, so no row left its [500, 2600] K exospheric-temperature \
         domain and the walked fallback arm is untested. A change to \
         FITTED_V7_TEXO_LO or FITTED_V7_TEXO_HI would move nothing here. The \
         last two entries of KERNEL_STATES exist to reach that arm; restore \
         them rather than deleting this assertion."
    );

    let drivers = survey(&driver_rows());
    drivers.report("DRIVERS-CENSUS");
    assert_not_vacuous(&drivers, "DRIVERS", DRIVER_ROWS);
    assert_eq!(
        drivers.domain_fallback, 0,
        "DRIVERS: a row fed by the SEALED driver table left the fit's \
         [500, 2600] K domain. The fit is claimed to cover production with room \
         on both sides — R28 censused 608.9 to 1627.5 K over the strict-HF arc — \
         so this is either a driver table that now reaches conditions the fit \
         does not model, or a narrowed FITTED_V7_TEXO_LO/HI. Neither is a \
         re-pin; it is a coverage finding."
    );
}

/// `KERNEL`: drivers are literals here, so this moves only if the code moved.
#[test]
fn fitted_v7_kernel_grid_is_bit_pinned() {
    let survey = survey(&kernel_rows());
    survey.report("KERNEL");
    assert_not_vacuous(&survey, "KERNEL", KERNEL_ROWS);
    assert_eq!(
        survey.grid_digest, PIN_KERNEL_GRID,
        "KERNEL grid inputs moved. The density digest below is then measuring a \
         different question; re-pin the grid and the density together, and say \
         in the commit which axis changed."
    );
    assert_eq!(
        survey.density_digest, PIN_KERNEL_DENSITY,
        "MODEL-7 FITTED DENSITY MOVED — this kernel is used beneath compiled \
         model 8 but does not itself pin persistence-driver authority. This is \
         a bit test, so it trips on changes far below the µm-scale arc moves that \
         strict_hf_pin's 1 cm tripwire absorbs. Diff \
         tests/jb2008_libm_probe.rs's FIT dump across the two trees to see which \
         rows moved, then re-pin here AND re-baseline V3_PINNED_POS_KM."
    );
}

/// `DRIVERS`: the same geometry fed by the sealed SET table, so this also moves
/// when the compiled driver data changes.
#[test]
fn fitted_v7_compiled_driver_grid_is_bit_pinned() {
    let survey = survey(&driver_rows());
    survey.report("DRIVERS");
    assert_not_vacuous(&survey, "DRIVERS", DRIVER_ROWS);
    assert_eq!(
        survey.grid_digest, PIN_DRIVERS_GRID,
        "DRIVERS grid inputs moved. This digest covers the values LOOKED UP from \
         the compiled SOLFSMY/DTCFILE authority, so it also trips when the \
         sealed driver table is replaced — check compiled_provider.rs's hashes \
         before assuming the grid literals were edited."
    );
    assert_eq!(
        survey.density_digest, PIN_DRIVERS_DENSITY,
        "MODEL-7 DENSITY MOVED on the driver-fed grid. If \
         fitted_v7_kernel_grid_is_bit_pinned is GREEN, the kernel did not move \
         and the compiled driver table did."
    );
}
