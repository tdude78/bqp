# A tour of the code

This is the map for reading the bundle, written for someone who has run
`examples/quickstart.py` and now wants to know where the physics lives. The
README covers installation and the Python API; `docs/API.md` lists every
function, argument and return key. This file is about the Rust underneath.

## What the system is for

The larger project studies dust-enabled just-in-time collision avoidance: a
deployer satellite releases a small dust cloud in front of a debris object so
that drag from the cloud nudges the object and changes a predicted close
approach. Doing that requires two things this bundle provides. First, a
transfer planner that gets a payload from the deployer's orbit to a chosen
point on the target's orbit at a chosen time, cheaply in delta-v. Second, a
propagator accurate enough to trust the intercept geometry down to metres over
hours to days. Everything else in the larger project (dust mass sizing,
uncertainty propagation, constellation optimization, campaign bookkeeping) is
out of scope here, though a few of those crates ride along as dependencies.

## How a Python call travels

`dt.propagate(state, epoch, tof, ...)`

1. `src/lib.rs::propagate` validates the inputs, builds a `ForceConfig`
   (force switches, drag coefficients, tolerance, stepper), and asks
   `ForceConfig::with_ephemeris_for_arc` to confirm the Sun, Moon and JB2008
   driver tables cover the requested dates.
2. It converts the ECI state to equinoctial elements
   (`two_phase_transfer_rs::evaluate::eci_to_equinoctial`), because the
   integrator works on the deviation from a two-body reference orbit expressed
   in those elements.
3. `lightyear_odeint_rs::integrator::integrate_adaptive` runs the rectified
   sampled propagation. Inside it, `rhs.rs` evaluates the accelerations
   (spherical-harmonic gravity from `satpy_core::gravity`, drag density from
   `jb_rs::jb2008`, SRP with eclipse handling from `eclipse_coordinator.rs`,
   third-body terms from `precomputed_ephem.rs`), and `odesolve/solver.rs`
   holds the Runge-Kutta steppers (Vern7 by default).
4. The result is a list of deviations. `lib.rs` adds the analytic two-body
   baseline back (`satpy_core::equinoc2eci_impl`) and returns full ECI states.

`problem.solve()`

1. `src/lib.rs::TransferProblem::context` fills a `TransferRequest` with the
   deployer and target states and the control caps, and
   `PlanContext::from_request` (in `two_phase_transfer_rs/src/types.rs`)
   caches the derived orbit quantities.
2. `two_phase_transfer_rs::solve::solve_plan` runs the search. The decision
   variables are three ratios: how long to spend reaching the phasing orbit,
   the phasing orbit's size, and how long to wait on it. `solve/moo.rs` drives
   an NSGA-II population (`oxymoo/nsga2.rs`) over those ratios; deterministic
   geometric seeds are added so obvious transfers are not missed.
3. For each candidate, `evaluate.rs` propagates the deployer and target under
   J2, applies the phasing burn, waits, and solves a Lambert problem
   (`lambert.rs`, Izzo's 2015 algorithm) for the transfer arc. A short J2
   corrector (`solve.rs`, `J2ClosureSettings`) nudges the arc so the endpoint
   miss falls under `distance_tol_km`.
4. `solve/front.rs` (`verified_front_from_plan`) keeps the non-dominated set
   (low delta-v against short time), re-flies each survivor to confirm the
   closure (`verify.rs`), and returns a `TransferFront`. `lib.rs` turns each `PlanResult` into a Python dict.

`problem.hf_verify(candidate)`

`hf_acceptance.rs` clones the same `PlanContext`, attaches a high-fidelity
force configuration and the gravity coefficients, and calls
`verify::replay_transfer_controls_segmented`, which flies the candidate's
burns through the Lightyear propagator in one-orbit segments. The reported
residual is the distance between where the payload ends up and where the J2
target is at that time.

## Crate map

Line counts are for `src/` only and include the tests that live inside the
source files.

| Crate | Lines | Role | Start reading at |
|---|---|---|---|
| `two_phase_transfer_rs` | 67k | Transfer search, Lambert solver, NSGA-II, J2 closure, candidate replay and HF acceptance | `src/lib.rs`, then `src/solve.rs` and `src/types.rs` (`PlanContext`, `PlanResult`) |
| `lightyear_odeint_rs` | 45k | The propagator: Encke formulation, steppers, force model, eclipse handling, ephemeris tables | `src/lib.rs`, `src/integrator.rs` (`integrate_adaptive`, `integrate_final_checked`), `src/rhs.rs` (`compute_internal_generic`) |
| `satpy_core` | 18k | Element conversions, spherical-harmonic gravity, GCRS to ITRS frame chain, dual numbers for autodiff | `src/lib.rs` (`kep2eci_impl`, `eci2equinoc_impl`), `src/gravity.rs` |
| `jb_rs` | 9k | JB2008 thermospheric density model and its solar/geomagnetic driver tables | `src/jb2008.rs` (`jb2008_density`), `src/drivers.rs` |
| `dust_estimates_rs` | 18k | Dust mass solver for the larger pipeline; pulled in as a dependency only | not needed for this bundle |
| `dust_splitting_rs` | 2k | Gaussian mixture splitting for dust covariance | not needed |
| `dust_ukf_rs` | 1.5k | Unscented transform for dust covariance | not needed |
| `nd_config` | 4k | The sealed campaign controls, compiled as constants | `src/part_a_science.rs` (`PART_A_V1`) |
| `nd_sched` | 1.3k | Rayon pool and per-cell scheduling helpers | not needed |
| `nd_runtime_trace` | 3k | Lossy runtime tracing for long campaign runs | not needed |
| `common_rs` | 0.3k | Small shared numeric helpers | not needed |

The binding itself is `src/lib.rs` at the package root, about 600 lines, and
`python/dust_transfer/__init__.py`.

## Where the numbers come from

Every default in the Python API is a value the dissertation campaign used.
They are compiled into `nd_config/src/part_a_science.rs` as the `PART_A_V1`
constant: transfer caps (`mf_transfer`), the high-fidelity force model and
tolerance (`hybrid`), and the rest. The binding restates the relevant ones as
keyword defaults so the Python surface has no dependency on that crate. If a
default in the README and a value in `part_a_science.rs` ever disagree, the
Rust constant is the one the campaign flew.

Data the propagator needs is embedded at build time:

| Data | Where | Coverage |
|---|---|---|
| Gravity field, GOCE DIR-R6 to degree 15 | `two_phase_transfer_rs/data/spher_const/` | static |
| Sun, Moon, Jupiter, Venus positions | `lightyear_odeint_rs/data/ephemeris/*.bin`, baked into tables by `build.rs` | JD 2458849.5 to 2462867.5 (2020 to 2031) |
| JB2008 solar and geomagnetic indices | `jb_rs/data/jb2008/SOLFSMY.TXT`, `DTCFILE.TXT`, baked by `build.rs` | 1997 to 2026-06-03 |
| Earth orientation and leap seconds | `satpy_core/src/frame_time/eop_table.bin`, `assets/reference/frame_time/` | static |

## Glossary

ECI state. Position and velocity in an Earth-centred inertial frame, km and
km/s, as a 6-vector.

Equinoctial elements. A non-singular set of orbital elements (semi-major axis
plus five shape and angle quantities) that stays well behaved at zero
eccentricity and zero inclination. The propagator's reference orbit lives in
these.

Encke's method. Instead of integrating the full state, integrate only the
difference between the true trajectory and an analytic two-body reference.
The difference is small, so the integrator can take long steps. When the
difference grows too large the reference is reset to the current state; that
reset is called rectification. Battin's f(q) formulation is used for the
gravity difference term so that the subtraction of two nearly equal
accelerations does not lose precision.

MF and HF. Medium fidelity means two-body plus J2, the model the transfer
search uses because it is cheap and analytic. High fidelity means the full
Lightyear force model. The search closes candidates under MF; `hf_verify`
shows what the same controls do under HF.

J2 closure. After the Lambert arc is chosen under two-body dynamics, a few
corrector iterations adjust the transfer burn so the arc, propagated under
J2, still meets the target within tolerance.

Lambert problem. Given two positions and a time of flight, find the orbit
connecting them. `lambert.rs` implements Izzo's 2015 solver, including
multi-revolution solutions (`lambert_revs` in the candidate dict).

Phasing orbit. The first burn moves the payload to an orbit with a slightly
different period, so that after waiting there the geometry lines up for a
cheap transfer. `phase_sma_km` is its semi-major axis.

Intercept versus rendezvous. The planner matches position only. The relative
velocity at arrival (`arrival_dv_norm`) is reported but not paid for, because
a dust release wants to cross the target's path, not fly alongside it.

NSGA-II and OxyMOO. NSGA-II is a multi-objective genetic algorithm. OxyMOO is
the name of the implementation in `oxymoo/`, absorbed from a separate crate.
It searches the three transfer ratios for the trade-off between delta-v and
time.

JB2008. An empirical thermosphere density model driven by daily solar flux
proxies (F10.7, S10, M10, Y10) and a geomagnetic index (DTC). `atm_model=4`
evaluates the published model exactly; `atm_model=7` uses a fitted kernel
that matches it closely and runs about four times faster.

Binary eclipse. SRP is switched fully on or off depending on whether the
spacecraft is in the Earth's cylindrical shadow. `eclipse_coordinator.rs`
finds the shadow crossings to the step and re-integrates across them so the
switch does not corrupt the error control.

Sealed controls, compiled science, bit pins. The larger project treats the
campaign configuration as immutable: it is compiled into `nd_config`, hashed,
and many tests assert bit-exact outputs against recorded digests. You will
see comments about pins, seals and digests throughout. In this bundle they
are history, not something you need to maintain.

## Reading the comments

The source is heavily commented, and the comments have a particular style.
Many carry dates and record a decision, a measurement, or a refuted idea
("REFUTED 2026-08-10", "walked upward and refuted"). They are the project's
lab notebook. When a comment says a change is dangerous or a lint is
suppressed for a stated reason, that reason is usually a bug that actually
happened. The workspace forbids panics, unwraps, indexing without bounds
checks, and integer arithmetic that can overflow, so you will see `checked_*`
and `get(..)` where you might expect plain operators.

## Browsing and testing

Generated API documentation for the Rust crates, with all doc comments
cross-linked, is the fastest way to browse:

```bash
cargo doc --no-deps -p two_phase_transfer_rs -p lightyear_odeint_rs -p satpy_core -p jb_rs --open
```

Rust unit tests that are known to pass in this bundle, each in under a
minute after the first compile:

```bash
cargo test --release -p jb_rs --lib                      # JB2008 against Orekit vectors
cargo test --release -p satpy_core --lib                 # conversions, gravity, frame chain
cargo test --release -p lightyear_odeint_rs --lib -- dir_r6 jb2008 pinned
cargo test --release -p two_phase_transfer_rs --lib orekit   # Lambert against Orekit
```

Some other tests need fixtures from the larger repository that are not in the
bundle (for example the sealed campaign events under `nd_pipeline`), and a
few are marked `#[ignore]` because they take hours or need the cluster. A
failing or ignored test outside the four commands above is expected and does
not indicate a broken build. Every test file starts with a comment saying
what it measures.

The Python side has `help(dust_transfer)` and `help(dust_transfer.TransferProblem)`;
the docstrings are the same text as `docs/API.md`.

## Suggested first hour

1. Run `examples/quickstart.py` and read it alongside the README.
2. Open `src/lib.rs` and follow `propagate` down into
   `lightyear_odeint_rs::integrator::integrate_adaptive`.
3. Open `lightyear_odeint_rs/src/rhs.rs` and find `compute_internal_generic`,
   the function that sums the accelerations. Each force term is a block with
   its own comment.
4. Open `two_phase_transfer_rs/src/solve.rs` at `solve_plan` and follow it
   into `solve/moo.rs`, then `evaluate.rs`.
5. Open `nd_config/src/part_a_science.rs` at `PART_A_V1` to see every
   number the campaign used, with the reasoning next to each.
