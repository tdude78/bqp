//! Cost of the UTC/TAI calendar chain, in isolation.
//!
//! Exists to settle a contradiction. A wall-clock sampling profile of the
//! production derivative attributed 16.8% of SELF time to `jd2cal` and 24.1%
//! to the `jd2cal` + `utctai` + `dat` trio. Separately, a memo that cut the
//! `jd2cal` call count per conversion from six to two -- proved bit-identical
//! against verbatim copies of both originals -- measured +0.4%, i.e. nothing,
//! and was reverted on that basis.
//!
//! Both cannot be right. One conversion costs six `jd2cal` (three `utctai`
//! iterations, two calls each), so a 16.8% share of an 811 ns derivative
//! implies roughly 22.7 ns per `jd2cal` call. This group measures that number
//! directly, which decides it: if `jd2cal` is a few nanoseconds the profile's
//! attribution is wrong, and if it is tens the earlier A/B was.
//!
//! Deliberately measures the primitives rather than a derivative, so nothing
//! about caching, inlining or force configuration can confuse the reading.
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use satpy_core::frame_time::timescale::{cal2jd, dat, jd2cal, taiutc, utctai};

/// A spread of instants inside the sealed span, stepped by a non-round amount
/// so no call lands repeatedly on the same calendar boundary.
fn sample_epochs() -> Vec<(f64, f64)> {
    let (status, base1, base2) = cal2jd(2022, 8, 12);
    assert_eq!(status, 0, "sealed-era epoch must be representable");
    (0..512)
        .map(|i| (base1, base2 + f64::from(i) * 0.37 / 86400.0))
        .collect()
}

fn bench_timescale_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("timescale_primitives");
    let epochs = sample_epochs();

    group.bench_function("jd2cal", |b| {
        b.iter(|| {
            let mut acc = 0i32;
            for (d1, d2) in &epochs {
                let (status, iy, im, id, _fd) = jd2cal(black_box(*d1), black_box(*d2));
                let row = status.wrapping_add(iy).wrapping_add(im).wrapping_add(id);
                acc = acc.wrapping_add(row);
            }
            black_box(acc)
        });
    });

    group.bench_function("dat", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for i in 0..epochs.len() {
                let day = i32::try_from(i % 28).unwrap_or_default().saturating_add(1);
                let (_status, delta) = dat(2022, 8, black_box(day), 0.0);
                acc += delta;
            }
            black_box(acc)
        });
    });

    group.bench_function("cal2jd", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for i in 0..epochs.len() {
                let day = i32::try_from(i % 28).unwrap_or_default().saturating_add(1);
                let (_status, a, b2) = cal2jd(2022, 8, black_box(day));
                acc += a + b2;
            }
            black_box(acc)
        });
    });

    // One `utctai` is two `jd2cal`, three `dat` and one `cal2jd`.
    group.bench_function("utctai", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for (d1, d2) in &epochs {
                let (_status, a, b2) = utctai(black_box(*d1), black_box(*d2));
                acc += a + b2;
            }
            black_box(acc)
        });
    });

    // The quantity the derivative actually pays: three `utctai` iterations,
    // so six `jd2cal`, nine `dat` and three `cal2jd`.
    group.bench_function("taiutc", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for (d1, d2) in &epochs {
                let (_status, a, b2) = taiutc(black_box(*d1), black_box(*d2));
                acc += a + b2;
            }
            black_box(acc)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_timescale_primitives);
criterion_main!(benches);
