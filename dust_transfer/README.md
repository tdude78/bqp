# dust_transfer, MVP 0.1.0 (evaluation snapshot, 2026-09-03)

License: PolyForm Strict 1.0.0 with an additional evaluation permission for
BQP until 2026-12-31; see `../LICENSE.md`. Use only. No changes, no
redistribution, no commercial use. This bundle is a minimal subset of a larger
private code base, and nothing outside the bundle is licensed. Third-party data
and code are listed in `../THIRD_PARTY_NOTICES.md`.

This is a Python package with a Rust core, built with
[maturin](https://www.maturin.rs). It exposes two pieces of the
`nasa_dust_rust` code base:

| What | Rust crate | Python entry point |
|---|---|---|
| Two-phase transfer optimizer: phasing burn, coast, Lambert transfer, intercept. NSGA-II with local polish, closed under J2. | `two_phase_transfer_rs` | `TransferProblem` |
| Lightyear high-fidelity propagator: Encke formulation, spherical-harmonic gravity, JB2008 drag, SRP with binary eclipse, Sun and Moon. | `lightyear_odeint_rs` | `propagate()`, `propagate_final()`, `TransferProblem.hf_verify()` |

`crates/` is a verbatim copy of the 11 crates those two depend on. The only new
code is the PyO3 binding in `src/lib.rs` (about 600 lines) and the small Python
package in `python/dust_transfer/`.

Units everywhere are km, km/s, seconds, and Julian Date (UTC). An ECI state is
`[x, y, z, vx, vy, vz]`.

Two more documents live in `docs/`. `docs/TOUR.md` is a map of the Rust code:
how a Python call travels through the crates, what each crate is for, a
glossary, and how to browse and test. `docs/API.md` lists every Python
function, argument and return key.

## Install

Build from source. I have tested this on macOS arm64 with CPython 3.12 and
3.14; `pip install .` also works. Free-threaded Python builds such as
`python3.14t` cannot load the abi3 extension.

You need Rust (the pinned toolchain `1.96.1` installs itself through
`rust-toolchain.toml` when `rustup` is present), Python 3.10 or newer, and a C
compiler. There is nothing to download: the gravity field, planetary
ephemerides, JB2008 solar and geomagnetic drivers, and Earth-orientation tables
are compiled into the extension.

```bash
cd dust_transfer
python -m venv .venv && source .venv/bin/activate     # or: uv venv .venv
pip install maturin numpy                             # or: uv pip install maturin numpy
maturin develop --release                             # about 1 min on first build (fat LTO)
python tests_smoke.py                                 # prints "OK: ..."
python examples/quickstart.py
```

`maturin build --release` produces an installable wheel under `target/wheels/`
(abi3, Python 3.10 or newer) if you prefer that. Always build with
`--release`. The debug profile is `opt-level = 0`, and the physics runs 20 to
50 times slower there.

## Usage

```python
import numpy as np
import dust_transfer as dt

epoch = 2460310.5                                   # 2024-01-01 00:00 UTC
deployer = dt.kep_to_eci([7178.137, 0.001, 97.4, 125.0, 210.0, 180.0])   # [a km, e, i, RAAN, argp, nu] deg
target   = dt.kep_to_eci([7278.137, 0.002, 97.6, 128.0,  40.0,  10.0])

# Transfer search
problem = dt.TransferProblem(deployer, target, epoch, max_time_s=86_400.0)
front = problem.solve()          # Pareto front, list of dicts, sorted by total dv
for c in front:
    print(dt.summarize(c))
best = front[0]
best["phase_dv"], best["transfer_dv"]          # burn vectors, km/s
best["time2phase_s"], best["waittime_s"], best["tof_s"]
best["payload_intercept_state"], best["target_intercept_state"]

# Re-fly a candidate
problem.replay(best)["residual_m"]            # medium fidelity (J2), what the search closed on
problem.hf_verify(best)                       # HF, 5x5 gravity only   -> residual_m, tolerance_m, accepted
problem.hf_verify(best, forces="full", am_ratio=0.01, cd=2.2, cr=1.3)   # plus drag, SRP, Sun, Moon

# Propagation
out = dt.propagate(deployer, epoch, tof_s=5400.0, times_s=np.arange(60, 5401, 60))
out["states_eci"]                # (n, 6); the grid always includes t=0 and t=tof_s
out["completed"], out["steps"], out["rhs_evals"]
dt.propagate(deployer, epoch, 5400.0, drag=False, srp=False, sun=False, moon=False)   # gravity only
dt.propagate_final(deployer, epoch, 5400.0)   # final state only; raises ValueError on impact or escape
```

### `propagate(state_eci, epoch_jd, tof_s, times_s=None, **model)`

| kwarg | default | meaning |
|---|---|---|
| `gravity_order` | 5 | degree and order of the GOCE DIR-R6 field (1 to 15) |
| `drag, srp, sun, moon` | all `True` | force switches |
| `atm_model` | 7 | JB2008 flavour. 4 is exact JB2008. 7 is the fitted-kernel JB2008, a sealed approximation that is about four times cheaper. 8 is the campaign persistence scenario and is only valid from 2026-08-15 to 2026-08-31. |
| `am_ratio, cd, cr` | 0.01, 2.2, 1.3 | area to mass ratio (m²/kg), drag coefficient, reflectivity coefficient |
| `tol` | 1e-8 | integrator relative tolerance |
| `dt_max_s` | 300 | maximum step |
| `method` | `"vern7"` | `vern7, vern9, dop853, tsit5, rkv98, dopri5` |

The propagator uses an Encke formulation. It integrates the deviation from
the two-body solution, and the binding adds the analytic baseline back, so
you get the full osculating ECI state.

`propagate()` is the rectified sampled path: re-baselined every orbit, any
sample grid, no event detection. If the integrator cannot finish, `completed`
is `False`, `terminal_event` names the reason (`gravity_invalidradius`,
`eclipse_envelope`, and so on), and the arrays hold whatever was produced.
`propagate_final(state_eci, epoch_jd, tof_s, **model)` returns only the end
state, through the checked final path the campaign uses. It raises
`ValueError` when the arc is physically infeasible (ground impact, escape, or
eccentricity blow-up) and `RuntimeError` when the integration itself fails.

### `TransferProblem(deployer_eci, target_eci, epoch_jd, **controls)`

The defaults are the campaign's sealed MF-transfer controls:
`max_time_s=172800, max_phase_dv=1.25, max_transfer_dv=1.25, max_revs=4,
min_perigee_km=6578.137, max_apogee_km=41378.137, distance_tol_km=0.025,
deployer_min_distance_km=0.12, tof_sample_budget=256, seed=42, parallel=True`,
plus the J2 closure settings `j2_max_iterations=5, j2_endpoint_target_km=0.01,
j2_correction_step_gain=1.0`. Every candidate that comes back has `valid=True`
and a J2 endpoint miss below `distance_tol_km`.

A candidate's timeline, in seconds from `epoch_jd`: the phase burn happens at
0, the vehicle coasts for `time2phase_s` on the phasing orbit
(`phase_sma_km`), waits for `waittime_s`, makes the transfer burn, flies the
Lambert arc for `tof_s`, and intercepts at `intercept_jd`. `total_dv` is phase
plus transfer. The arrival burn is reported in `arrival_dv` but not spent,
because this is an intercept rather than a rendezvous: only position is
matched. So `arrival_dv_norm` is the relative speed at intercept, and a target
in a very different plane can be intercepted cheaply at the node crossing. With
a 37° plane difference, for example, `total_dv` is 0.06 km/s while
`arrival_dv_norm` is 4.75 km/s. `summarize()` prints both.

The defaults are sized for the campaign's LEO-to-LEO geometry. The caps, not
the solver, reject larger transfers, and relaxing them works. For an 800 km
SSO deployer and a GEO-radius target in the same plane:

```python
dt.TransferProblem(dep, tgt, epoch, max_apogee_km=50000.0, max_transfer_dv=3.0,
                   max_phase_dv=3.0, revolution_cap=20.0, tof_sample_budget=1024,
                   coarse_reject_margin_km_s=1.0, seed_fine_margin_km_s=1.0, max_revs=8).solve()
# one candidate: total_dv 2.29 km/s, 5.6 h (the Hohmann departure burn)
```

## Things to know

The search and `replay()` run at medium fidelity, meaning J2, which is the
fidelity the optimizer closes on. `hf_verify()` re-flies the same controls
under the high-fidelity propagator and reports the miss. For multi-hour
transfers that miss is usually tens to hundreds of km. That is the same
behaviour the source repo pins in its own test
(`crates/two_phase_transfer_rs/tests/hf_acceptance_api.rs` shows 82 to 1713 km
on three sealed candidates). It is the model-class gap that the campaign closes
with its HF correction and hybrid lowering stage, which lives in the pipeline
crates and is not in this bundle. Treat `hf_verify()` as a diagnostic rather
than a pass/fail gate on MF candidates.

Drag with `atm_model` 4 or 7 needs an epoch inside the embedded JB2008 driver
files, which run from 1997 to 2026-06-03. The planetary ephemerides cover JD
2458849.5 to 2462867.5 (2020-01-01 to 2031-01-01). An arc outside either range
raises `ValueError` before anything is integrated.

With `srp=True`, the binary-eclipse coordinator only accepts states with a
radius between 6000 and 50000 km and a speed at or below 20 km/s, which is the
campaign's envelope. Outside it you get `terminal_event == "eclipse_envelope"`
or a `ValueError`. The eclipse path also returns no samples at all when it
fails, so a re-entering arc looks like an immediate refusal. Re-run with
`srp=False` to see where the arc really ends (`gravity_invalidradius` at the
last sample above the surface), and use `srp=False` for arcs that sit outside
the envelope anyway.

Results are deterministic on one machine: calling `solve()` twice, or running
with `parallel=True` and then `False`, gives bit-identical output. `seed` only
feeds the local-optimizer restarts and usually leaves the front unchanged.
Across machines the results agree to floating-point noise. The source repo pins
per-libm digests, but those pins are not shipped here.

`parallel=True` uses a Rayon pool over all cores. The GIL is released during
`solve()`, `hf_verify()`, `replay()`, `propagate()` and `propagate_final()`,
and all of them can be called from several Python threads at once.

Any 6-element sequence works as a state: a list, a tuple, or a numpy array of
any float or int dtype. Bad input raises `ValueError` with a message before
anything is integrated.

The other crates in `crates/` (dust mass solvers, UKF, splitting, scheduler)
are there only because the two exposed crates depend on them.

## Layout

```
Cargo.toml            workspace plus the PyO3 crate
LICENSE.md            PolyForm Strict 1.0.0 plus the BQP evaluation permission
pyproject.toml        maturin config (module dust_transfer._native)
src/lib.rs            the binding
python/dust_transfer  Python package (__init__.py: re-exports plus summarize())
examples/quickstart.py
tests_smoke.py
docs/TOUR.md          map of the Rust code, glossary, how to browse and test
docs/API.md           Python API reference
crates/               11 vendored crates from nasa_dust_rust
assets/reference/     the two reference files the crates include at build time
```
