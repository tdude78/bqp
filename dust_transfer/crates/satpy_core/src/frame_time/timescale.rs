//! UTC/TAI/TT time-scale chain, transliterated from sealed ERFA 2.0.1 /
//! SOFA 20231011 (`cal2jd`, `jd2cal`, `dat`, `dtf2d`, `utctai`, `taiutc`,
//! `taitt`).
//!
//! Ordinary binary64 arithmetic: the fixture uses ERFA's own f64 time chain,
//! and the double-double path only enters at ERA / EOP / outer composition.
//!
//! The physical constants and integer-range guards are transliterated from
//! ERFA `erfam.h` / source, with their binary64 values represented explicitly
//! where decimal spelling would obscure exact provenance.

use num_traits::ToPrimitive;

pub const D2PI: f64 = std::f64::consts::TAU;
pub const DAS2R: f64 = std::f64::consts::PI / 648_000.0;
pub const TURNAS: f64 = 1_296_000.0;
pub const DAYSEC: f64 = 86400.0;
pub const DJC: f64 = 36525.0;
pub const DJ00: f64 = 2_451_545.0;
pub const DJM0: f64 = 2_400_000.5;
pub const TTMTAI: f64 = 32.184;
/// 0.1 microarcsecond to radians (nutation series units).
pub const U2R: f64 = DAS2R / 1e7;

#[inline]
fn dnint(a: f64) -> f64 {
    if a.abs() < 0.5 {
        0.0
    } else if a < 0.0 {
        (a - 0.5).ceil()
    } else {
        (a + 0.5).floor()
    }
}

/// Gregorian calendar date to two-part MJD. Returns `(status, djm0, djm)`.
#[must_use]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "ERFA calendar formula uses bounded i64 intermediates; the final binary64 conversion is its specified result"
)]
pub fn cal2jd(iy: i32, im: i32, id: i32) -> (i32, f64, f64) {
    const IYMIN: i32 = -4799;

    let mut j = 0;
    if iy < IYMIN {
        return (-1, 0.0, 0.0);
    }
    if !(1..=12).contains(&im) {
        return (-2, 0.0, 0.0);
    }
    let leap_day = i32::from((im == 2) && (iy % 4 == 0) && (iy % 100 != 0 || iy % 400 == 0));
    let month_days = match im {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => return (-2, 0.0, 0.0),
    };
    if id < 1 || id > month_days + leap_day {
        j = -3;
    }
    let month_year_adjustment = (im - 14) / 12;
    let shifted_year = i64::from(iy) + i64::from(month_year_adjustment);
    let shifted_month = i64::from(im) - 2 - 12 * i64::from(month_year_adjustment);
    let mjd_integer = (1461 * (shifted_year + 4800)) / 4 + (367 * shifted_month) / 12
        - (3 * ((shifted_year + 4900) / 100)) / 4
        + i64::from(id)
        - 2_432_076;
    let djm = mjd_integer.to_f64().unwrap_or(f64::NAN);
    (j, DJM0, djm)
}

/// Two-part JD to Gregorian calendar. Returns `(status, iy, im, id, fd)`.
#[must_use]
pub fn jd2cal(dj1: f64, dj2: f64) -> (i32, i32, i32, i32, f64) {
    match jd2cal_split(dj1, dj2) {
        None => (-1, 0, 0, 0, 0.0),
        Some((jd, f)) => {
            let (iy, im, id) = jd2cal_ymd(jd);
            (0, iy, im, id, f)
        }
    }
}

/// The compensated-summation half of [`jd2cal`]: the two-part JD reduced to an
/// integer Julian day number and the day fraction. `None` is the out-of-range
/// status the caller reports as `-1`.
///
/// Split out from the integer half below so a caller that asks for several
/// nearby instants pays the calendar divisions once. Verbatim the ERFA body up
/// to the point where the date arithmetic begins; no operation is reordered, so
/// `jd2cal` above is bit-identical to the single function it replaced.
#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "ERFA compensated Julian-day normalization requires this exact arithmetic and branch order; checked conversion above rejects malformed input parts"
)]
fn jd2cal_split(dj1: f64, dj2: f64) -> Option<(i64, f64)> {
    const DJMIN: f64 = -68569.5;
    const DJMAX: f64 = 1e9;

    let dj = dj1 + dj2;
    if outside_inclusive_preserving_nan(dj, DJMIN, DJMAX) {
        return None;
    }

    let mut rounded = dnint(dj1);
    let first_fraction = dj1 - rounded;
    let mut julian_day = rounded.to_i64()?;
    rounded = dnint(dj2);
    let second_fraction = dj2 - rounded;
    julian_day = julian_day.checked_add(rounded.to_i64()?)?;

    let mut sum = 0.5;
    let mut compensation = 0.0;
    let fractions = [first_fraction, second_fraction];
    for &fraction in &fractions {
        let provisional_sum = sum + fraction;
        compensation += if sum.abs() >= fraction.abs() {
            (sum - provisional_sum) + fraction
        } else {
            (fraction - provisional_sum) + sum
        };
        sum = provisional_sum;
        if sum >= 1.0 {
            julian_day += 1;
            sum -= 1.0;
        }
    }
    let mut fraction = sum + compensation;
    compensation = fraction - sum;

    if fraction < 0.0 {
        fraction = sum + 1.0;
        compensation += (1.0 - fraction) + sum;
        sum = fraction;
        fraction = sum + compensation;
        compensation = fraction - sum;
        julian_day -= 1;
    }

    if (fraction - 1.0) >= -f64::EPSILON / 4.0 {
        let remainder = sum - 1.0;
        compensation += (sum - remainder) - 1.0;
        sum = remainder;
        fraction = sum + compensation;
        if -f64::EPSILON / 2.0 < fraction {
            julian_day += 1;
            fraction = fraction.max(0.0);
        }
    }

    Some((julian_day, fraction))
}

/// The integer half of [`jd2cal`]: Julian day number to Gregorian `(y, m, d)`.
///
/// Eight 64-bit divisions and nothing else. ARM has no fast integer divide, and
/// a sampling profile of the strict-HF lowering puts `jd2cal` at 4.39% of
/// on-CPU self time — it runs six times per `taiutc`, which runs once per RK
/// stage. The float half above changes on every one of those six calls; this
/// half does not, because all six land on at most two calendar days.
///
/// Reusing THIS across passes is bit-exact by construction: integers in,
/// integers out, no floating-point arithmetic to reassociate.
#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "jd2cal_split bounds input Julian days; ERFA integer calendar formula then remains within i32 output range"
)]
fn jd2cal_ymd(jd: i64) -> (i32, i32, i32) {
    let mut work = jd + 68_569;
    let century_cycle = (4 * work) / 146_097;
    work -= (146_097 * century_cycle + 3) / 4;
    let year_cycle = (4000 * (work + 1)) / 1_461_001;
    work -= (1461 * year_cycle) / 4 - 31;
    let month_cycle = (80 * work) / 2447;
    let day = (work - (2447 * month_cycle) / 80)
        .to_i32()
        .unwrap_or_default();
    work = month_cycle / 11;
    let month = (month_cycle + 2 - 12 * work).to_i32().unwrap_or_default();
    let year = (100 * (century_cycle - 49) + year_cycle + work)
        .to_i32()
        .unwrap_or_default();
    (year, month, day)
}

// Built-in leap-second table (ERFA _changes; last entry 2017-01, +37 s).
const CHANGES: [(i32, i32, f64); 42] = [
    (1960, 1, 1.417_818_0),
    (1961, 1, 1.422_818_0),
    (1961, 8, 1.372_818_0),
    (1962, 1, 1.845_858_0),
    (1963, 11, 1.945_858_0),
    (1964, 1, 3.240_130_0),
    (1964, 4, 3.340_130_0),
    (1964, 9, 3.440_130_0),
    (1965, 1, 3.540_130_0),
    (1965, 3, 3.640_130_0),
    (1965, 7, 3.740_130_0),
    (1965, 9, 3.840_130_0),
    (1966, 1, 4.313_170_0),
    (1968, 2, 4.213_170_0),
    (1972, 1, 10.0),
    (1972, 7, 11.0),
    (1973, 1, 12.0),
    (1974, 1, 13.0),
    (1975, 1, 14.0),
    (1976, 1, 15.0),
    (1977, 1, 16.0),
    (1978, 1, 17.0),
    (1979, 1, 18.0),
    (1980, 1, 19.0),
    (1981, 7, 20.0),
    (1982, 7, 21.0),
    (1983, 7, 22.0),
    (1985, 7, 23.0),
    (1988, 1, 24.0),
    (1990, 1, 25.0),
    (1991, 1, 26.0),
    (1992, 7, 27.0),
    (1993, 7, 28.0),
    (1994, 7, 29.0),
    (1996, 1, 30.0),
    (1997, 7, 31.0),
    (1999, 1, 32.0),
    (2006, 1, 33.0),
    (2009, 1, 34.0),
    (2012, 7, 35.0),
    (2015, 7, 36.0),
    (2017, 1, 37.0),
];

// Pre-leap-second drift reference MJDs and rates (s/day); NERA1 entries.
const DRIFT: [(f64, f64); 14] = [
    (37_300.0, 0.001_296_0),
    (37_300.0, 0.001_296_0),
    (37_300.0, 0.001_296_0),
    (37_665.0, 0.001_123_2),
    (37_665.0, 0.001_123_2),
    (38_761.0, 0.001_296_0),
    (38_761.0, 0.001_296_0),
    (38_761.0, 0.001_296_0),
    (38_761.0, 0.001_296_0),
    (38_761.0, 0.001_296_0),
    (38_761.0, 0.001_296_0),
    (38_761.0, 0.001_296_0),
    (39_126.0, 0.002_592_0),
    (39_126.0, 0.002_592_0),
];

const IYV: i32 = 2023;

#[inline]
fn outside_inclusive_preserving_nan(value: f64, lower: f64, upper: f64) -> bool {
    matches!(value.partial_cmp(&lower), Some(std::cmp::Ordering::Less))
        || matches!(value.partial_cmp(&upper), Some(std::cmp::Ordering::Greater))
}

/// `Delta(AT) = TAI - UTC` in seconds. Returns `(status, deltat)`.
#[must_use]
pub fn dat(iy: i32, im: i32, id: i32, fd: f64) -> (i32, f64) {
    if outside_inclusive_preserving_nan(fd, 0.0, 1.0) {
        return (-4, 0.0);
    }
    let (j, _djm0, djm) = cal2jd(iy, im, id);
    if j < 0 {
        return (j, 0.0);
    }
    dat_from_mjd(iy, im, fd, djm)
}

/// The body of [`dat`] after its `cal2jd`, taking the MJD the caller already
/// has. Verbatim the remainder of the ERFA body; nothing is reordered.
///
/// Exists because `utctai` asks for the same calendar date three times over —
/// `dat(.., 0.0)`, `dat(.., 0.5)` and its own `cal2jd` for the day number — so
/// the identical `cal2jd(iy, im, id)` runs three times per conversion. A
/// per-line profile of the hot loop puts that redundancy at 6.5% of `taiutc`
/// (the `cal2jd` inside `dat`) on top of the 4.2% `utctai` pays directly.
///
/// The caller must have checked `0 <= fd <= 1` and a non-negative `cal2jd`
/// status, which are exactly the two early returns [`dat`] makes above. `id`
/// does not appear below: it reaches the result only through `djm`.
#[inline]
fn dat_from_mjd(iy: i32, im: i32, fd: f64, djm: f64) -> (i32, f64) {
    let mut j = 0;
    if iy < CHANGES[0].0 {
        return (1, 0.0);
    }
    if iy > IYV + 5 {
        j = 1;
    }
    let month_key = i64::from(iy)
        .saturating_mul(12)
        .saturating_add(i64::from(im));
    let Some((change_index, &(_, _, mut delta_at))) =
        CHANGES
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &(year, month, _))| {
                month_key
                    >= i64::from(year)
                        .saturating_mul(12)
                        .saturating_add(i64::from(month))
            })
    else {
        return (-5, 0.0);
    };
    if let Some(&(reference_mjd, drift_per_day)) = DRIFT.get(change_index) {
        delta_at += (djm + fd - reference_mjd) * drift_per_day;
    }
    (j, delta_at)
}

/// Encode UTC date/time into a two-part quasi-JD, with leap-second handling.
/// Returns `(status, d1, d2)`.
#[must_use]
pub fn dtf2d_utc(iy: i32, im: i32, id: i32, ihr: i32, imn: i32, sec: f64) -> (i32, f64, f64) {
    let (js0, dj0, w0) = cal2jd(iy, im, id);
    if js0 != 0 {
        return (js0, 0.0, 0.0);
    }
    let dj = dj0 + w0;

    let mut day = DAYSEC;
    let mut seclim = 60.0;

    // UTC leap-second handling.
    let (s, dat0) = dat(iy, im, id, 0.0);
    if s < 0 {
        return (s, 0.0, 0.0);
    }
    let (s, dat12) = dat(iy, im, id, 0.5);
    if s < 0 {
        return (s, 0.0, 0.0);
    }
    let (js, iy2, im2, id2, _w) = jd2cal(dj, 1.5);
    if js != 0 {
        return (js, 0.0, 0.0);
    }
    let (s, dat24) = dat(iy2, im2, id2, 0.0);
    if s < 0 {
        return (s, 0.0, 0.0);
    }
    let dleap = dat24 - (2.0 * dat12 - dat0);
    day += dleap;
    if ihr == 23 && imn == 59 {
        seclim += dleap;
    }

    let mut js: i32 = 0;
    if (0..=23).contains(&ihr) {
        if (0..=59).contains(&imn) {
            if sec >= 0.0 {
                if sec >= seclim {
                    js = js.saturating_add(2);
                }
            } else {
                js = -6;
            }
        } else {
            js = -5;
        }
    } else {
        js = -4;
    }
    if js < 0 {
        return (js, 0.0, 0.0);
    }

    let minutes_since_midnight = 60_i32.saturating_mul(ihr).saturating_add(imn);
    let time = (60.0 * f64::from(minutes_since_midnight) + sec) / day;
    (js, dj, time)
}

/// Everything one `utctai` pass computes that is not the day fraction, carried
/// forward to the next pass of the `taiutc` fixed point.
///
/// `taiutc` inverts the leap second by running `utctai` on a `u2` that moves by
/// about 4e-4 d on the first correction and by an ULP thereafter. Only `fd`
/// tracks that motion; the leap-second geometry of the day does not. So a pass
/// keeps the two f64 inputs of the tail `jd2cal` and the integer Julian day of
/// the leading one, and the next pass reuses the derived quantities only when
/// all three are BIT-IDENTICAL to what produced them.
///
/// That is the whole correctness argument: `jd2cal_split`, `jd2cal_ymd`,
/// `cal2jd` and `dat` are pure and deterministic, so identical input bits give
/// identical output bits. A reuse returns what the recomputation would have
/// returned, by construction rather than by tolerance.
///
/// Replaces a two-way / four-way linear-scan memo. A per-line profile of the
/// hot loop put that memo's own key scans at 15.9% of `taiutc` — nearly four
/// times what the `cal2jd` it was protecting cost — because a scan runs on
/// every call whereas this guard runs once per pass and short-circuits the
/// entire remainder of the pass, `jd2cal_split` included.
///
/// Its lifetime is one `taiutc` call plus whatever the caller chooses to carry
/// forward; see [`TAIUTC_CARRY`] for why that turned out to be worth doing.
#[derive(Clone, Copy, Default)]
struct PassCarry {
    valid: bool,
    /// Julian day number from `jd2cal_split(u1, u2)`.
    day_jd: i64,
    /// Raw bits of the two arguments of the tail `jd2cal_split`.
    tail_key: [u64; 2],
    /// Status of the `dat` call that sets the returned status.
    j: i32,
    z1: f64,
    z2: f64,
    /// `dat0 / DAYSEC`, `(DAYSEC + dleap) / DAYSEC` and `(DAYSEC + dlod) /
    /// DAYSEC`, divided out once instead of once per pass. Each is a
    /// deterministic function of values already fixed, so a later pass reading
    /// the stored quotient reads the bits it would have recomputed.
    dat0_day: f64,
    leap_scale: f64,
    lod_scale: f64,
}

impl PassCarry {
    /// A carry that matches no key, for `const`-initialising the thread-local
    /// below. `Default` is not a `const fn`, so it cannot be used there.
    const EMPTY: Self = Self {
        valid: false,
        day_jd: 0,
        tail_key: [0; 2],
        j: 0,
        z1: 0.0,
        z2: 0.0,
        dat0_day: 0.0,
        leap_scale: 0.0,
        lod_scale: 0.0,
    };
}

/// UTC (two-part quasi-JD) to TAI. Returns `(status, tai1, tai2)`.
#[must_use]
pub fn utctai(utc1: f64, utc2: f64) -> (i32, f64, f64) {
    utctai_carried(utc1, utc2, &mut PassCarry::default())
}

fn utctai_carried(utc1: f64, utc2: f64, carry: &mut PassCarry) -> (i32, f64, f64) {
    let big1 = utc1.abs() >= utc2.abs();
    let (u1, u2) = if big1 { (utc1, utc2) } else { (utc2, utc1) };

    let Some((jd, fd)) = jd2cal_split(u1, u2) else {
        return (-1, 0.0, 0.0);
    };
    // The tail conversion only ever yields a calendar date, so its two f64
    // arguments are the complete key for everything downstream of it.
    let tail1 = u1 + 1.5;
    let tail2 = u2 - fd;
    let tail_key = [tail1.to_bits(), tail2.to_bits()];

    if !(carry.valid && carry.day_jd == jd && carry.tail_key == tail_key) {
        let (iy, im, id) = jd2cal_ymd(jd);
        // One `cal2jd` for the day, shared by both `dat` calls and by the day
        // number `a2` needs below. ERFA runs it three times with identical
        // arguments; it is a pure function of three integers.
        let (jz, z1, z2) = cal2jd(iy, im, id);
        if jz < 0 {
            return (jz, 0.0, 0.0);
        }
        let (j, dat0) = dat_from_mjd(iy, im, 0.0, z2);
        if j < 0 {
            return (j, 0.0, 0.0);
        }
        let (j, dat12) = dat_from_mjd(iy, im, 0.5, z2);
        if j < 0 {
            return (j, 0.0, 0.0);
        }
        let Some((jdt, _w)) = jd2cal_split(tail1, tail2) else {
            return (-1, 0.0, 0.0);
        };
        let (iyt, imt, idt) = jd2cal_ymd(jdt);
        let (j, dat24) = dat(iyt, imt, idt, 0.0);
        if j < 0 {
            return (j, 0.0, 0.0);
        }

        let dlod = 2.0 * (dat12 - dat0);
        let dleap = dat24 - (dat0 + dlod);

        *carry = PassCarry {
            valid: true,
            day_jd: jd,
            tail_key,
            j,
            z1,
            z2,
            dat0_day: dat0 / DAYSEC,
            leap_scale: (DAYSEC + dleap) / DAYSEC,
            lod_scale: (DAYSEC + dlod) / DAYSEC,
        };
    }

    let mut fd = fd;
    fd *= carry.leap_scale;
    fd *= carry.lod_scale;

    let mut a2 = carry.z1 - u1;
    a2 += carry.z2;
    a2 += fd + carry.dat0_day;

    if big1 {
        (carry.j, u1, a2)
    } else {
        (carry.j, a2, u1)
    }
}

/// TAI to UTC (two-part quasi-JD). Returns `(status, utc1, utc2)`.
///
/// ERFA runs the correction a fixed three times. It is a fixed point in `u2`,
/// and `utctai` is a deterministic pure function of `(u1, u2)`, so once a pass
/// returns `u2` with the bits it went in with, every remaining pass is a no-op
/// on both `u2` and the status. Stopping there is not an early-exit heuristic:
/// it is the same answer, reached without recomputing it.
///
/// Measured over 20,000 campaign-faithful instants: 19,923 reach that bitwise
/// fixed point after the second pass, 27 need the third, and 50 never reach it
/// at all — `u2` chatters by an ULP forever. Those last 50 are why the trip
/// count still caps at three, which makes them bit-identical to ERFA by running
/// exactly the passes ERFA runs.
#[must_use]
pub fn taiutc(tai1: f64, tai2: f64) -> (i32, f64, f64) {
    TAIUTC_CARRY.with(|slot| {
        let mut carry = slot.get();
        let out = taiutc_carried(tai1, tai2, &mut carry);
        slot.set(carry);
        out
    })
}

thread_local! {
    /// The calendar carry [`taiutc`] hands to its passes, kept alive BETWEEN
    /// calls rather than rebuilt on each one.
    ///
    /// The carry's guard is `valid && day_jd == jd && tail_key == tail_key`, and
    /// every field behind it is a pure function of that same `(day_jd,
    /// tail_key)` pair: `z1`/`z2` from `cal2jd(jd2cal_ymd(jd))`, `dat0_day` and
    /// `lod_scale` from two `dat_from_mjd` calls on that date, and `leap_scale`
    /// and `j` from the `dat` at `tail_key`'s date. A hit therefore returns the
    /// bits a recomputation would have produced no matter how long ago the miss
    /// that filled it ran, so extending the carry's lifetime past the end of a
    /// call is bit-exact by the same argument that makes it correct within one.
    /// This is why the slot may hold state across calls but must never widen its
    /// key: `key -> value` completeness is the entire proof.
    ///
    /// WHY THIS EARNS A THREAD-LOCAL WHEN THE IN-PASS CARRY DID NOT. Measured on
    /// one campaign-faithful hybrid batch, instrumented and thrown away:
    /// 99,000,000 `taiutc` calls ran 198,406,625 passes (2.004 per call, so the
    /// fixed-point exit is taking the third pass off every time) and missed
    /// 99,039,317 times — 1.0004 misses per call. Pass 1 of EVERY call missed,
    /// because the carry was born empty. Replaying that same trace against a
    /// one-entry carry persisted per thread leaves 111,634 misses, **0.113% of
    /// calls**: consecutive RK stages are microseconds apart, so the key only
    /// turns over at a calendar-day boundary.
    ///
    /// The rejected design was a thread-local for the memo INSIDE a pass, which
    /// needed a `_tlv_get_addr` per lookup. This needs exactly one per `taiutc`
    /// call and deletes six integer-division-heavy calendar calls on 99.887% of
    /// them.
    ///
    /// Per-thread, so worker count cannot change an answer: the slot is only
    /// ever read through the key guard, and a cold slot recomputes.
    static TAIUTC_CARRY: std::cell::Cell<PassCarry> =
        const { std::cell::Cell::new(PassCarry::EMPTY) };
}

fn taiutc_carried(tai1: f64, tai2: f64, carry: &mut PassCarry) -> (i32, f64, f64) {
    let big1 = tai1.abs() >= tai2.abs();
    let (a1, a2) = if big1 { (tai1, tai2) } else { (tai2, tai1) };

    let u1 = a1;
    let mut u2 = a2;
    let mut j = 0;
    for _ in 0..3 {
        let before = u2.to_bits();
        let (jj, g1, g2) = utctai_carried(u1, u2, carry);
        j = jj;
        if j < 0 {
            return (j, 0.0, 0.0);
        }
        u2 += a1 - g1;
        u2 += a2 - g2;
        if u2.to_bits() == before {
            break;
        }
    }

    if big1 {
        (j, u1, u2)
    } else {
        (j, u2, u1)
    }
}

/// TAI to TT (two-part JD). Returns `(tt1, tt2)`.
#[must_use]
pub fn taitt(tai1: f64, tai2: f64) -> (f64, f64) {
    let dtat = TTMTAI / DAYSEC;
    if tai1.abs() > tai2.abs() {
        (tai1, tai2 + dtat)
    } else {
        (tai1 + dtat, tai2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfa_angular_constants_keep_their_binary64_values() {
        assert_eq!(D2PI.to_bits(), 0x4019_21fb_5444_2d18);
        assert_eq!(DAS2R.to_bits(), 0x3ed4_55a5_b2ff_8f9d);
    }

    #[test]
    fn outside_inclusive_preserves_erfa_nan_acceptance() {
        assert!(!outside_inclusive_preserving_nan(f64::NAN, -1.0, 1.0));
        assert!(outside_inclusive_preserving_nan(-1.5, -1.0, 1.0));
        assert!(outside_inclusive_preserving_nan(1.5, -1.0, 1.0));
        assert!(!outside_inclusive_preserving_nan(0.0, -1.0, 1.0));
    }

    /// Checks five of this module's routines DIRECTLY against externally
    /// published values, at the specific inputs listed below and nowhere else.
    ///
    /// PROVENANCE OF EVERY VALUE AND BOUND BELOW: ERFA v2.0.1, `src/t_erfa_c.c`
    /// — the same release this module transliterates (see the file header).
    /// Reference values and tolerances are both ERFA's, copied verbatim.
    ///
    /// Two columns because they are different lines: the inputs are at the call,
    /// and the expected values and tolerances are on the `vvd`/`viv` assertions
    /// that follow it. Every number below was re-derived from the archive.
    ///
    ///   routine    inputs at            values + bounds at    bounds
    ///   `t_cal2jd`   :2809                :2811-:2812, :2814    exact, exact
    ///   `t_dat`      :3016,:3021,:3026    :3018-:3019,          exact
    ///                                   :3023-:3024,
    ///                                   :3028-:3029
    ///   `t_jd2cal`   :4901-:4904          :4906-:4910           y/m/d exact, fd 1e-7
    ///   `t_taiutc`   :8983                :8985-:8987           u1 1e-6, u2 1e-12
    ///   `t_utctai`   :9731                :9733-:9735           u1 1e-6, u2 1e-12
    ///
    /// `t_jd2cal` spans a range on the input side because its arguments are
    /// assigned to `dj1`/`dj2` at :4901-:4902 rather than written into the call.
    ///
    /// `t_erfa_c.c` ships inside the sealed archive at
    /// `assets/reference/frame_time/erfa-2.0.1-sofa-20231011-source.tar.gz` and
    /// is excluded from the oracle build by name at
    /// `scripts/regenerate-frame-time-oracle.sh:223`, so these values are quoted
    /// here rather than computed.
    ///
    /// DO NOT TIGHTEN THESE BOUNDS. `iau2006.rs` records what happens: bounds
    /// there were once tightened 10x-1000x below ERFA's published values with no
    /// justification, and that broke on `x86_64` Linux while nothing about the
    /// transliteration had moved. ERFA's own tolerances are the contract.
    ///
    /// WHAT THIS TEST BUYS, precisely, because the defect it corrects was an
    /// overclaim rather than a missing test. Five inputs are five inputs; they
    /// are not coverage of an input space, and the claims below are limited to
    /// what was actually demonstrated by negative control:
    ///
    /// - It catches a defect that CHANGES THE RESULT AT ONE OF THESE INPUTS by
    ///   more than ERFA's tolerance. Demonstrated, not assumed: a leap-second
    ///   table entry wrong by 1 s fails the `eraDat` assertion, and a
    ///   `taiutc`-only perturbation of 1e-9 d fails the `eraTaiutc` assertion.
    ///   It does NOT follow that every "systematic transliteration error" is
    ///   caught — a defect confined to dates these five vectors do not touch
    ///   passes, and the pre-1972 `DRIFT` branch is the concrete example.
    /// - It does NOT catch a 1-ULP disagreement with ERFA, and that was
    ///   confirmed rather than assumed: a 1-ULP perturbation of `taiutc`'s
    ///   result passes this test while failing
    ///   [`taiutc_is_bit_identical_to_the_unconditional_three_pass_loop`]. ERFA's
    ///   tolerance on `u2` is 1e-12 and one ULP there is ~1.1e-16, so this is a
    ///   VALUE test, not a bit test. That companion test is a REGRESSION test —
    ///   its reference calls our own `utctai`, so a defect in the conversion
    ///   body cancels on both sides. Neither test subsumes the other.
    /// - It does NOT cover the pre-1972 `DRIFT` interpolation branch: ERFA's
    ///   `t_dat` vectors are all post-1972, so no canonical material for it
    ///   exists in the archive.
    /// - What it adds for `taiutc` specifically. `chain.rs` contains zero
    ///   references to `taiutc`, so `taiutc` is not exercised by the sealed
    ///   fixture (`tests/frame_time_oracle.rs`, bound `R_BOUND = 5e-13` on a
    ///   downstream GCRS→ITRS rotation). The `taiutc` guards in `rhs.rs` that
    ///   predate this test resolve `.floor()`ed whole MJD in
    ///   `jb_driver_lookup_uses_utc_of_the_tai_instant`, and 1e-9 d (~86 µs) in
    ///   `ephemeris_lookup_scale_matches_the_table_manifest`. This test asserts
    ///   on `taiutc`'s own two output components at 1e-6 / 1e-12.
    ///
    /// Claims deliberately NOT made here, because they were not verified: that
    /// this is the only independent check in the module, that it is the only
    /// ULP-level guard anywhere, or that the `DRIFT` branch is uncovered
    /// tree-wide. Those are universal negatives over a large tree; a comment
    /// that asserts them is doing the same thing this test was written to stop.
    #[test]
    fn erfa_canonical_vectors() {
        // t_cal2jd :2809 — eraCal2jd(2003, 6, 1)
        let (j, djm0, djm) = cal2jd(2003, 6, 1);
        assert_eq!(j, 0, "eraCal2jd j");
        assert_eq!(djm0.to_bits(), 2_400_000.5_f64.to_bits(), "eraCal2jd djm0");
        assert_eq!(djm.to_bits(), 52_791.0_f64.to_bits(), "eraCal2jd djm");

        // t_dat :3016,:3021,:3026 — three eraDat calls, all exact.
        for (iy, im, id, want) in [
            (2003, 6, 1, 32.0_f64),
            (2008, 1, 17, 33.0_f64),
            (2017, 9, 1, 37.0_f64),
        ] {
            let (j, deltat) = dat(iy, im, id, 0.0);
            assert_eq!(j, 0, "eraDat j at {iy}-{im}-{id}");
            assert_eq!(
                deltat.to_bits(),
                want.to_bits(),
                "eraDat deltat at {iy}-{im}-{id}"
            );
        }

        // t_jd2cal :4903 — eraJd2cal(2400000.5, 50123.9999)
        let (j, iy, im, id, fd) = jd2cal(2_400_000.5, 50_123.999_9);
        assert_eq!(j, 0, "eraJd2cal j");
        assert_eq!(iy, 1996, "eraJd2cal y");
        assert_eq!(im, 2, "eraJd2cal m");
        assert_eq!(id, 10, "eraJd2cal d");
        assert!(
            (fd - 0.9999).abs() <= 1e-7,
            "eraJd2cal fd: got {fd}, want 0.9999 within 1e-7"
        );

        // t_taiutc :8983 — eraTaiutc(2453750.5, 0.892482639)
        let (j, u1, u2) = taiutc(2_453_750.5, 0.892_482_639);
        assert_eq!(j, 0, "eraTaiutc j");
        assert!(
            (u1 - 2_453_750.5).abs() <= 1e-6,
            "eraTaiutc u1: got {u1}, want 2453750.5 within 1e-6"
        );
        assert!(
            (u2 - 0.892_100_694_555_555_5).abs() <= 1e-12,
            "eraTaiutc u2: got {u2}, want 0.8921006945555555556 within 1e-12"
        );

        // t_utctai :9731 — eraUtctai(2453750.5, 0.892100694)
        let (j, a1, a2) = utctai(2_453_750.5, 0.892_100_694);
        assert_eq!(j, 0, "eraUtctai j");
        assert!(
            (a1 - 2_453_750.5).abs() <= 1e-6,
            "eraUtctai u1: got {a1}, want 2453750.5 within 1e-6"
        );
        assert!(
            (a2 - 0.892_482_638_444_444_4).abs() <= 1e-12,
            "eraUtctai u2: got {a2}, want 0.8924826384444444444 within 1e-12"
        );
    }

    // Gate 1: leap-second table and UTC quasi-JD leap handling.
    #[test]
    fn gate1_leap_and_dtf2d() {
        assert_eq!(dat(2000, 1, 1, 0.5).1.to_bits(), 32.0_f64.to_bits());
        assert_eq!(dat(2016, 12, 31, 0.0).1.to_bits(), 36.0_f64.to_bits());
        assert_eq!(dat(2017, 1, 1, 0.0).1.to_bits(), 37.0_f64.to_bits());
        assert_eq!(dat(2024, 1, 1, 0.0).1.to_bits(), 37.0_f64.to_bits());

        // The 2016-12-31 leap second (23:59:60) is a valid UTC time.
        let (js, _d1, _d2) = dtf2d_utc(2016, 12, 31, 23, 59, 60.0);
        assert_eq!(js, 0, "23:59:60 must be accepted on a leap-second day");
        // A non-leap day rejects 60.0 as after end of day.
        let (js2, _, _) = dtf2d_utc(2017, 1, 1, 23, 59, 60.0);
        assert_eq!(js2, 2, "23:59:60 is after end of an ordinary day");

        // Round-trip UTC -> TAI -> UTC across the leap instant.
        let (_s, u1, u2) = dtf2d_utc(2016, 12, 31, 23, 59, 60.0);
        let (_t, a1, a2) = utctai(u1, u2);
        let (_b, b1, b2) = taiutc(a1, a2);
        assert!(((u1 + u2) - (b1 + b2)).abs() < 1e-9 / DAYSEC);
    }

    /// ERFA's `taiutc` written out: three unconditional passes over the public
    /// `utctai`, which starts each conversion from an empty carry and is
    /// therefore the un-restructured single conversion.
    ///
    /// The oracle for the test below. It is a transliteration of the loop
    /// [`taiutc`] replaced, not a paraphrase of it.
    fn taiutc_three_pass(tai1: f64, tai2: f64) -> (i32, f64, f64) {
        let big1 = tai1.abs() >= tai2.abs();
        let (a1, a2) = if big1 { (tai1, tai2) } else { (tai2, tai1) };
        let u1 = a1;
        let mut u2 = a2;
        let mut j = 0;
        for _ in 0..3 {
            let (jj, g1, g2) = utctai(u1, u2);
            j = jj;
            if j < 0 {
                return (j, 0.0, 0.0);
            }
            u2 += a1 - g1;
            u2 += a2 - g2;
        }
        if big1 {
            (j, u1, u2)
        } else {
            (j, u2, u1)
        }
    }

    /// `taiutc` reuses work across its passes and stops at the fixed point.
    /// Both are argued bit-exact by construction; this asserts it.
    ///
    /// Sweeps every leap-second boundary in [`CHANGES`], the pre-1972 drift
    /// era, and a modern sweep, comparing RAW BITS rather than values — a
    /// tolerance here would pass exactly the reassociation this test exists to
    /// forbid. Time feeds the ephemeris feeds the trajectory, so a one-ULP
    /// change is a different program.
    ///
    /// The same sweep, widened offline to 114,851,465 comparisons, reports zero
    /// divergent bits; an unconditional TWO-pass loop under the identical sweep
    /// reports 66,986, which is what says the sweep can see a difference.
    #[test]
    fn taiutc_is_bit_identical_to_the_unconditional_three_pass_loop() {
        let mut bases: Vec<(f64, f64)> = Vec::new();
        for (y, m, _) in CHANGES {
            let (s, d1, d2) = cal2jd(y, m, 1);
            assert_eq!(s, 0, "leap-second boundary {y}-{m} must be representable");
            // One day early, so the sweep crosses the boundary rather than
            // starting on it.
            bases.push((d1, d2 - 1.0));
        }
        for y in [1960, 1965, 1969, 1971, 1988, 2005, 2016, 2022, 2027] {
            let (s, d1, d2) = cal2jd(y, 6, 15);
            assert_eq!(s, 0);
            bases.push((d1, d2));
        }

        let mut compared = 0u64;
        for (d1, d2) in bases {
            // Two days at ~0.94 s, plus a fine sweep at ~2 ms through the
            // instant where the day boundary and the leap second coincide.
            for i in 0..3_000 {
                let f = f64::from(i) * 0.94 / DAYSEC;
                for (a1, a2) in [(d1, d2 + f), (d2 + f, d1)] {
                    let got = taiutc(a1, a2);
                    let want = taiutc_three_pass(a1, a2);
                    assert_eq!(got.0, want.0, "status at ({a1}, {a2})");
                    assert_eq!(
                        got.1.to_bits(),
                        want.1.to_bits(),
                        "utc1 bits at ({a1}, {a2})"
                    );
                    assert_eq!(
                        got.2.to_bits(),
                        want.2.to_bits(),
                        "utc2 bits at ({a1}, {a2})"
                    );
                    compared += 1;
                }
            }
            for i in -500i32..500 {
                let f = 1.0 + f64::from(i) * 2.0e-3 / DAYSEC;
                let (a1, a2) = (d1, d2 + f);
                let got = taiutc(a1, a2);
                let want = taiutc_three_pass(a1, a2);
                assert_eq!(got.0, want.0);
                assert_eq!(got.1.to_bits(), want.1.to_bits());
                assert_eq!(got.2.to_bits(), want.2.to_bits());
                compared += 1;
            }
        }

        // Out-of-range and degenerate inputs take the early-return paths, which
        // the sweep above never reaches.
        for (a1, a2) in [
            (0.0, 0.0),
            (-1e9, 0.0),
            (1e9 + 1.0, 0.0),
            (-68569.5, 0.0),
            (1e9, 0.0),
            (DJM0, 0.0),
            (DJM0, -0.0),
        ] {
            let got = taiutc(a1, a2);
            let want = taiutc_three_pass(a1, a2);
            assert_eq!(got.0, want.0, "status at ({a1}, {a2})");
            assert_eq!(got.1.to_bits(), want.1.to_bits(), "utc1 at ({a1}, {a2})");
            assert_eq!(got.2.to_bits(), want.2.to_bits(), "utc2 at ({a1}, {a2})");
            compared += 1;
        }

        assert!(compared > 250_000, "sweep collapsed to {compared} cases");
    }
}
