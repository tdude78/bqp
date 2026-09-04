#![expect(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::print_stdout,
    clippy::while_float,
    reason = "a timing harness converts loop indices to f64, accumulates nanoseconds, polls a float deadline, and reports on stdout; every one of those is the instrument, not a defect"
)]

//! Bottom-up cost map of the JB2008 kernel the campaign flies (`atm_model` 7).
//!
//! ```sh
//! # whole-kernel prices and the priced libm census, on production codegen
//! cargo run --release -p jb_rs --example jb2008_cost_map
//! # a sampler-only workload; build under --profile release-profile first
//! cargo run --profile release-profile -p jb_rs --example jb2008_cost_map -- kernel-loop 20
//! ```
//!
//! # What this measures, and why it is not a profiler
//!
//! This project's arc A/B has a 1.13% null floor, so a lever worth less than
//! that cannot be seen by differencing two arcs. The method that works below the
//! floor is a counted call rate times a per-call microbenchmark, and this
//! program supplies the second factor for the JB2008 lane: nanoseconds per
//! kernel call, per profile, plus the price of every libm call the flown kernel
//! issues.
//!
//! The call rate is one JB2008 kernel call per RHS evaluation, printed by
//! `lightyear_odeint_rs`'s `libm_budget` example as `kernel/eval`, alongside the
//! nanoseconds per RHS evaluation this lane's share is taken against. The two
//! programs are deliberately separate: this one links only `jb_rs`, so it builds
//! in seconds and can be re-run against any edit to the kernel.
//!
//! # The operand envelope is production's, and it is printed
//!
//! Altitude spans the censused strict-HF band (626.2--985.7 km) and the eight
//! solar indices are held at one sealed driver day, because production reads
//! them from a table that is constant across a UTC day — which is what keeps
//! `jb_tsubc`'s memo hitting and its `powf` retired.

use std::hint::black_box;
use std::time::Instant;

use jb_rs::jb2008::{
    jb2008_density, jb2008_density_fitted_v7, jb2008_density_logquad_x4_approx_v1,
    jb2008_density_logquad_x4_approx_v2, Jb2008Input,
};

/// Inputs per pass. Small enough to stay resident in L1 alongside the kernel's
/// own constants, which is what production sees: one call at a time against a
/// warm cache.
const TABLE: usize = 256;

/// Timed passes per arm. The reported figure is the MINIMUM over them, not the
/// mean: on a shared laptop the mean measures the other tenants and the minimum
/// measures the machine.
const BLOCKS: usize = 400;

/// The sealed Orekit driver day, swept over the production geometry envelope.
///
/// The eight solar indices are held constant on purpose — see the module note.
/// `mjd_utc`, the hour angle, the latitude and the altitude all move, because
/// every one of them moves within a single propagation.
fn production_inputs() -> Vec<Jb2008Input> {
    (0..TABLE)
        .map(|index| {
            let fraction = index as f64 / TABLE as f64;
            Jb2008Input {
                mjd_utc: 52_951.003_805_740_744 + fraction * 0.5,
                sun_declination_rad: -0.285_987_757_544_287 + fraction * 0.02,
                hour_angle_rad: (fraction * std::f64::consts::TAU) - std::f64::consts::PI,
                // Inclination 97.4 deg, so the geocentric latitude reaches
                // +-1.44 rad and sweeps it once per revolution.
                sat_geocentric_lat_rad: 1.44 * (fraction * 6.0 * std::f64::consts::TAU).sin(),
                // The censused strict-HF altitude band, 626.2 to 985.7 km.
                sat_altitude_m: 626_226.149 + fraction * (985_663.551 - 626_226.149),
                f10: 91.00,
                f10b: 137.10,
                s10: 108.80,
                s10b: 123.80,
                m10: 116.70,
                m10b: 128.50,
                y10: 168.00,
                y10b: 138.60,
                dst_temperature_correction_k: 43.0,
            }
        })
        .collect()
}

/// Minimum nanoseconds per call over `BLOCKS` passes of `calls` calls each.
fn bench(label: &str, calls: usize, mut pass: impl FnMut() -> f64) -> f64 {
    for _ in 0..32 {
        black_box(pass());
    }
    let mut best = f64::INFINITY;
    for _ in 0..BLOCKS {
        let start = Instant::now();
        let accumulator = pass();
        let elapsed = start.elapsed().as_secs_f64();
        black_box(accumulator);
        best = best.min(elapsed);
    }
    let ns = best * 1.0e9 / calls as f64;
    println!("  {label:<34} {ns:>9.3} ns/call");
    ns
}

/// One line of the static transcendental census: how many calls of one routine
/// the flown kernel issues per evaluation, and where they are.
struct CensusRow {
    routine: &'static str,
    count: f64,
    sites: &'static str,
}

/// Every libm call the model-7 kernel issues per evaluation at a production
/// altitude, counted from the source rather than from an interposer.
///
/// The counts hold on the branch production takes on every evaluation:
/// altitude at or above 500 km (so both fixed plans are replaced by the fit and
/// the upper segment runs exactly one Boole step), exospheric temperature inside
/// the fit's `[500, 2600]` K domain, and the eight solar indices constant across
/// a UTC day (so `jb_tsubc` hits its memo and its `powf` never executes).
///
/// # These counts are corroborated, not asserted
///
/// R55's `LD_PRELOAD` call census (`docs/PMU_PROFILE.md` §10) counted every libm
/// arrival on the production arc, per 170 propagations, bucketed by call site.
/// Its §10.2 row for `jb2008.rs:734` — the `tau` line, which issues exactly one
/// `sin` — reads 1,147,840 calls, i.e. **6,752 kernel calls per propagation**.
/// Take that as the call rate and the rows below are not merely close, they are
/// exact:
///
/// | routine | R55 per propagation | this census x 6,752 | JB2008's share |
/// |---|---:|---:|---:|
/// | `exp` | 67,520.1 | 67,520 | **100.0%** |
/// | `log` | 20,256.1 | 20,256 | **100.0%** |
/// | `sin` | 59,898.5 | 33,760 | 56.4% |
/// | `cos` | 53,147.0 | 27,008 | 50.8% |
///
/// Every `exp` and every `log` the whole arc executes is issued by this kernel,
/// on two instruments that share no code. And the trig remainders — 26,138.5
/// `sin` against 26,139.0 `cos`, equal to 0.002% — say the rest of the tree
/// calls `sin` and `cos` only in pairs, which is exactly what §10.2's site list
/// shows.
///
/// This is what makes the counts here reproducible where R55's are not: the
/// interposer that produced them is not in this repository, but the rows below
/// are read off the source and can be re-derived by anyone, and the agreement
/// above is then a check on both.
///
/// A `sin_cos` is counted as ONE call and priced against a measured `sin_cos`,
/// because that is what the source issues; on glibc it lowers to two libm
/// entries and on Darwin to one, and the measured price carries whichever
/// applies on the host that ran this program.
///
/// `sqrt` is in this table because it is priced the same way, but it is NOT a
/// libm call: it lowers to a hardware square-root instruction on both targets.
/// It therefore belongs to the binary's own lane in a module-level profile, not
/// to `libm.so.6`, and must not be added to a libm module share.
const CENSUS: &[CensusRow] = &[
    CensusRow {
        routine: "exp",
        count: 10.0,
        sites: "transition T, rho, zr, dlrsl, 6 species",
    },
    CensusRow {
        routine: "sin",
        count: 3.0,
        sites: "tau, tsub_l sin(theta), dlrsl",
    },
    CensusRow {
        routine: "cos",
        count: 2.0,
        sites: "tsub_l cos(eta), cos(tau/2)",
    },
    CensusRow {
        routine: "sin_cos",
        count: 2.0,
        sites: "satellite latitude, semian seasonal",
    },
    CensusRow {
        routine: "ln",
        count: 3.0,
        sites: "al, altr, tloc4/tloc3",
    },
    CensusRow {
        routine: "log10",
        count: 1.0,
        sites: "hydrogen number density",
    },
    CensusRow {
        routine: "sqrt",
        count: 2.0,
        sites: "jb_positive_five_halves x2 in tsub_l",
    },
];

fn main() {
    let inputs = production_inputs();

    // `kernel-loop <seconds>`: the flown kernel and nothing else, so an external
    // sampler attributes every sample to this kernel. Build it under
    // `--profile release-profile` for the line tables; plain `--release` strips
    // them and the report comes back as one unnamed symbol.
    if std::env::args().nth(1).as_deref() == Some("kernel-loop") {
        let seconds: f64 = std::env::args()
            .nth(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(20.0);
        let start = Instant::now();
        let mut accumulator = 0.0;
        let mut calls = 0_u64;
        while start.elapsed().as_secs_f64() < seconds {
            for input in &inputs {
                accumulator += jb2008_density_fitted_v7(black_box(*input)).unwrap_or(0.0);
            }
            calls += inputs.len() as u64;
        }
        println!("kernel-loop: {calls} calls, acc {accumulator:.6e}");
        return;
    }

    println!("\nWHOLE KERNEL, per public entry point:");
    let m7 = bench("model 7 (flown, fitted)", inputs.len(), || {
        let mut accumulator = 0.0;
        for input in &inputs {
            accumulator += jb2008_density_fitted_v7(black_box(*input)).unwrap_or(0.0);
        }
        accumulator
    });
    let m6 = bench("model 6 (coarse quadrature)", inputs.len(), || {
        let mut accumulator = 0.0;
        for input in &inputs {
            accumulator += jb2008_density_logquad_x4_approx_v2(black_box(*input)).unwrap_or(0.0);
        }
        accumulator
    });
    let m5 = bench("model 5 (fine quadrature)", inputs.len(), || {
        let mut accumulator = 0.0;
        for input in &inputs {
            accumulator += jb2008_density_logquad_x4_approx_v1(black_box(*input)).unwrap_or(0.0);
        }
        accumulator
    });
    let m4 = bench("model 4 (exact Orekit)", inputs.len(), || {
        let mut accumulator = 0.0;
        for input in &inputs {
            accumulator += jb2008_density(black_box(*input)).unwrap_or(0.0);
        }
        accumulator
    });
    // The four profiles differ ONLY in how many Boole steps they walk, so their
    // differences price one step three times over from three disjoint pairs.
    // Three answers that agree is a decomposition; one answer is a guess.
    // Step counts: model 4 walks 16 lower + 63 middle, model 5 walks 4 + 16,
    // model 6 walks 4 + 6, and model 7 replaces both plans with the fit and
    // walks none of them. The upper segment is one step in all four.
    println!(
        "  BOOLE STEP, from three disjoint pairs: m6-m7 {:.2} ns, m5-m6 {:.2} ns, m4-m5 {:.2} ns",
        (m6 - m7) / 10.0,
        (m5 - m6) / 10.0,
        (m4 - m5) / 59.0
    );

    println!("\nUNIT COST OF ONE LIBM CALL ON THIS HOST AND THIS BUILD:");
    let scalars: Vec<f64> = (0..TABLE)
        .map(|index| (index as f64 + 0.5) / TABLE as f64)
        .collect();
    let angles: Vec<f64> = scalars
        .iter()
        .map(|fraction| fraction.mul_add(std::f64::consts::TAU, -std::f64::consts::PI))
        .collect();
    let positives: Vec<f64> = scalars
        .iter()
        .map(|fraction| 0.05 + fraction * 6.0)
        .collect();
    let unit_exp = bench("exp", positives.len(), || {
        let mut accumulator = 0.0;
        for x in &positives {
            accumulator += (-black_box(*x)).exp();
        }
        accumulator
    });
    let unit_sin = bench("sin", angles.len(), || {
        let mut accumulator = 0.0;
        for x in &angles {
            accumulator += black_box(*x).sin();
        }
        accumulator
    });
    let unit_cos = bench("cos", angles.len(), || {
        let mut accumulator = 0.0;
        for x in &angles {
            accumulator += black_box(*x).cos();
        }
        accumulator
    });
    let unit_sin_cos = bench("sin_cos", angles.len(), || {
        let mut accumulator = 0.0;
        for x in &angles {
            let (sine, cosine) = black_box(*x).sin_cos();
            accumulator += sine + cosine;
        }
        accumulator
    });
    let unit_ln = bench("ln", positives.len(), || {
        let mut accumulator = 0.0;
        for x in &positives {
            accumulator += black_box(*x).ln();
        }
        accumulator
    });
    let unit_log10 = bench("log10", positives.len(), || {
        let mut accumulator = 0.0;
        for x in &positives {
            accumulator += black_box(*x).log10();
        }
        accumulator
    });
    let unit_sqrt = bench("sqrt", positives.len(), || {
        let mut accumulator = 0.0;
        for x in &positives {
            accumulator += black_box(*x).sqrt();
        }
        accumulator
    });
    bench("fmul (harness control)", positives.len(), || {
        let mut accumulator = 0.0;
        for x in &positives {
            accumulator += black_box(*x).mul_add(0.999_999, 1.0e-9);
        }
        accumulator
    });

    println!("\nSTATIC TRANSCENDENTAL CENSUS OF ONE MODEL-7 EVALUATION, PRICED:");
    let mut libm_total = 0.0;
    for row in CENSUS {
        let unit = match row.routine {
            "exp" => unit_exp,
            "sin" => unit_sin,
            "cos" => unit_cos,
            "sin_cos" => unit_sin_cos,
            "ln" => unit_ln,
            "log10" => unit_log10,
            "sqrt" => unit_sqrt,
            _ => f64::NAN,
        };
        let cost = row.count * unit;
        libm_total += cost;
        println!(
            "  {:<8} x{:<5.1} = {:>7.2} ns  {:>5.1}% of the kernel   [{}]",
            row.routine,
            row.count,
            cost,
            100.0 * cost / m7,
            row.sites
        );
    }
    println!(
        "  TOTAL              {libm_total:>9.2} ns  {:>5.1}% of the {m7:.2} ns kernel call, at \
         THROUGHPUT prices",
        100.0 * libm_total / m7
    );
    println!(
        "  RESIDUAL           {:>9.2} ns  {:>5.1}%  (arithmetic, branches, memo, atan_x4, and \
         the latency these calls cost above their throughput price)",
        m7 - libm_total,
        100.0 * (m7 - libm_total) / m7
    );
}
