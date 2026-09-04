# dust_transfer — MVP 0.1.0 (evaluation snapshot, 2026-09-03)

**License:** PolyForm Strict 1.0.0 with an additional evaluation permission for
BQP until 2026-12-31; see `../LICENSE.md`. Use only: no changes, no
redistribution, no commercial use. This bundle is a minimal subset of a larger
private code base; nothing outside the bundle is licensed. Third-party data and
code are listed in `../THIRD_PARTY_NOTICES.md`.

Python package (Rust core, built with [maturin](https://www.maturin.rs)) exposing
two pieces of the `nasa_dust_rust` code base:

| What | Rust crate | Python entry point |
|---|---|---|
| Two-phase transfer optimizer (phasing burn → coast → Lambert transfer → intercept), NSGA-II + local polish, J2-closed | `two_phase_transfer_rs` | `TransferProblem` |
| Lightyear high-fidelity propagator (Encke, spherical-harmonic gravity, JB2008 drag, SRP with binary eclipse, Sun/Moon) | `lightyear_odeint_rs` | `propagate()`, `propagate_final()`, `TransferProblem.hf_verify()` |

`crates/` is a verbatim copy of the 11 crates those two depend on. The only new
code is `src/lib.rs` (the PyO3 binding, ~600 lines) and `python/dust_transfer/`.

Units everywhere: **km, km/s, seconds, Julian Date (UTC)**. ECI states are
`[x, y, z, vx, vy, vz]`.

## Install (5 minutes)

Build from source (tested on macOS arm64 with CPython 3.12 and 3.14; `pip install .`
also works). Free-threaded Python builds (`python3.14t`) are not supported by
the abi3 extension.

Requirements: Rust (the pinned toolchain `1.96.1` installs itself via
`rust-toolchain.toml` when `rustup` is present), Python ≥ 3.10, and a C
compiler. No data has to be downloaded: gravity field, planetary ephemerides,
JB2008 solar/geomagnetic drivers and Earth-orientation tables are compiled into
the extension.

```bash
cd dust_transfer
python -m venv .venv && source .venv/bin/activate     # or: uv venv .venv
pip install maturin numpy                             # or: uv pip install maturin numpy
maturin develop --release                             # ~1 min first build (fat LTO)
python tests_smoke.py                                 # prints "OK: ..."
python examples/quickstart.py
```

`maturin build --release` instead produces an installable wheel under
`target/wheels/` (abi3, Python ≥ 3.10). Always build `--release`: the
debug profile is `opt-level = 0` and the physics is 20-50x slower.

## Usage

```python
import numpy as np
import dust_transfer as dt

epoch = 2460310.5                                   # 2024-01-01 00:00 UTC
deployer = dt.kep_to_eci([7178.137, 0.001, 97.4, 125.0, 210.0, 180.0])   # [a km, e, i, RAAN, argp, nu] deg
target   = dt.kep_to_eci([7278.137, 0.002, 97.6, 128.0,  40.0,  10.0])

# --- Transfer search -----------------------------------------------------
problem = dt.TransferProblem(deployer, target, epoch, max_time_s=86_400.0)
front = problem.solve()          # Pareto front, list of dicts, sorted by total dv
for c in front:
    print(dt.summarize(c))
best = front[0]
best["phase_dv"], best["transfer_dv"]          # burn vectors, km/s
best["time2phase_s"], best["waittime_s"], best["tof_s"]
best["payload_intercept_state"], best["target_intercept_state"]

# --- Re-fly a candidate --------------------------------------------------
problem.replay(best)["residual_m"]            # medium fidelity (J2), what the search closed on
problem.hf_verify(best)                       # HF, 5x5 gravity only   -> residual_m, tolerance_m, accepted
problem.hf_verify(best, forces="full", am_ratio=0.01, cd=2.2, cr=1.3)   # + drag, SRP, Sun, Moon

# --- Propagation ---------------------------------------------------------
out = dt.propagate(deployer, epoch, tof_s=5400.0, times_s=np.arange(60, 5401, 60))
out["states_eci"]                # (n, 6); the grid always includes t=0 and t=tof_s
out["completed"], out["steps"], out["rhs_evals"]
dt.propagate(deployer, epoch, 5400.0, drag=False, srp=False, sun=False, moon=False)   # gravity only
dt.propagate_final(deployer, epoch, 5400.0)   # final state only; raises ValueError on impact / escape
```

### `propagate(state_eci, epoch_jd, tof_s, times_s=None, **model)`

| kwarg | default | meaning |
|---|---|---|
| `gravity_order` | 5 | degree/order of the GOCE DIR-R6 field (1..15) |
| `drag, srp, sun, moon` | all `True` | force switches |
| `atm_model` | 7 | JB2008 flavour: 4 = exact JB2008, 7 = fitted-kernel JB2008 (sealed approximation, ~4x cheaper), 8 = campaign persistence scenario (only valid 2026-08-15 → 2026-08-31) |
| `am_ratio, cd, cr` | 0.01, 2.2, 1.3 | area/mass (m²/kg), drag and reflectivity coefficients |
| `tol` | 1e-8 | integrator relative tolerance |
| `dt_max_s` | 300 | maximum step |
| `method` | `"vern7"` | `vern7, vern9, dop853, tsit5, rkv98, dopri5` |

The propagator is an Encke formulation: it integrates the deviation from the
two-body solution and the binding adds the analytic baseline back, so what you
get is the full osculating ECI state.

`propagate()` is the rectified sampled path (re-baselined every orbit, any
sample grid, no event detection). If the integrator cannot finish, `completed`
is `False`, `terminal_event` names the reason (`gravity_invalidradius`,
`eclipse_envelope`, ...) and the arrays hold whatever was produced.
`propagate_final(state_eci, epoch_jd, tof_s, **model)` returns only the end
state through the checked final path the campaign uses, and raises
`ValueError` for a physically infeasible arc (ground impact, escape,
eccentricity blow-up) or `RuntimeError` for an integration failure.

### `TransferProblem(deployer_eci, target_eci, epoch_jd, **controls)`

Defaults are the campaign's sealed MF-transfer controls:
`max_time_s=172800, max_phase_dv=1.25, max_transfer_dv=1.25, max_revs=4,
min_perigee_km=6578.137, max_apogee_km=41378.137, distance_tol_km=0.025,
deployer_min_distance_km=0.12, tof_sample_budget=256, seed=42, parallel=True`
plus the J2 closure settings `j2_max_iterations=5, j2_endpoint_target_km=0.01,
j2_correction_step_gain=1.0`. Every candidate returned has `valid=True` and a
J2 endpoint miss below `distance_tol_km`.

Timeline of a candidate, all seconds from `epoch_jd`: phase burn at 0 → coast
`time2phase_s` on the phasing orbit (`phase_sma_km`) → wait `waittime_s` →
transfer burn → Lambert arc of `tof_s` → intercept at `intercept_jd`.
`total_dv` = phase + transfer; the arrival burn is reported (`arrival_dv`) but
not spent because this is an **intercept, not a rendezvous**: only position
is matched. `arrival_dv_norm` is therefore the relative speed at intercept,
and a target in a very different plane can be intercepted cheaply at the node
crossing (e.g. a 37° plane difference: `total_dv` 0.06 km/s, `arrival_dv_norm`
4.75 km/s). `summarize()` prints both.

The defaults are sized for the campaign's LEO-to-LEO geometry. Larger
transfers are rejected by the caps rather than by the solver; relaxing them
works, e.g. 800 km SSO → GEO-radius target in the same plane:

```python
dt.TransferProblem(dep, tgt, epoch, max_apogee_km=50000.0, max_transfer_dv=3.0,
                   max_phase_dv=3.0, revolution_cap=20.0, tof_sample_budget=1024,
                   coarse_reject_margin_km_s=1.0, seed_fine_margin_km_s=1.0, max_revs=8).solve()
# -> 1 candidate, total_dv 2.29 km/s, 5.6 h (the Hohmann departure burn)
```

## Things to know

* **MF vs HF.** The search and `replay()` run at medium fidelity (J2, the
  fidelity the optimizer closes on). `hf_verify()` re-flies the same controls
  under the high-fidelity propagator and reports the miss. For multi-hour
  transfers that miss is typically **tens to hundreds of km**: the same
  behaviour the source repo pins in its own test
  (`crates/two_phase_transfer_rs/tests/hf_acceptance_api.rs`, 82–1713 km on
  three sealed candidates). It is the model-class gap the campaign closes with
  its HF correction / hybrid lowering stage, which lives in the pipeline crates
  and is **not** included here. Treat `hf_verify()` as a diagnostic, not as a
  pass/fail gate on MF candidates.
* **Driver coverage.** Drag with `atm_model` 4 or 7 needs the epoch inside the
  embedded JB2008 driver files (1997 → 2026-06-03). Planetary ephemerides cover
  JD 2458849.5 → 2462867.5 (2020-01-01 → 2031-01-01). Out-of-range arcs raise
  `ValueError` before integrating.
* **SRP envelope.** With `srp=True` the binary-eclipse coordinator only accepts
  states with radius 6000–50000 km and speed ≤ 20 km/s (the campaign's
  envelope). Outside it you get `terminal_event == "eclipse_envelope"` or a
  `ValueError`, and the eclipse path returns **no samples at all**, so a
  re-entering arc looks like an immediate refusal. Re-run with `srp=False` to
  see where the arc actually ends (`gravity_invalidradius` at the last sample
  above the surface), or pass `srp=False` for arcs outside the envelope.
* **Determinism.** Same inputs on the same machine → bit-identical results
  (`solve()` twice, or `parallel=True` vs `False`, agree bit for bit). `seed`
  only feeds the local-optimizer restarts and usually leaves the front
  unchanged. Across machines results agree to floating-point noise (the source
  repo pins per-libm digests; those pins are not shipped).
* **Threads.** `parallel=True` uses a Rayon pool (all cores). The GIL is
  released during `solve()`, `hf_verify()`, `replay()`, `propagate()` and
  `propagate_final()`, and all of them are safe to call from several Python
  threads at once.
* **Inputs.** Any 6-element sequence works for a state (list, tuple, numpy
  array of any float/int dtype). Bad input raises `ValueError` with a message
  before anything is integrated.
* Everything else in `crates/` (dust mass solvers, UKF, splitting, scheduler)
  is there only because the two exposed crates depend on it.

## Layout

```
Cargo.toml            workspace + the PyO3 crate
LICENSE.md            PolyForm Strict 1.0.0 + BQP evaluation permission
pyproject.toml        maturin config (module dust_transfer._native)
src/lib.rs            the binding
python/dust_transfer  Python package (__init__.py: re-exports + summarize())
examples/quickstart.py
tests_smoke.py
crates/               11 vendored crates from nasa_dust_rust
assets/reference/     the two reference files the crates include at build time
```
