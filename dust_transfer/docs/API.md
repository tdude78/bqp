# Python API reference

Everything lives in the `dust_transfer` package. Units are km, km/s, seconds
and Julian Date (UTC). A state is a 6-element sequence `[x, y, z, vx, vy, vz]`
and can be a list, a tuple or a numpy array; results come back as numpy
arrays. Bad input raises `ValueError` with a message before any integration
starts.

## Module constants

| Name | Value |
|---|---|
| `MU_EARTH_KM3_S2` | Earth's gravitational parameter, 398600.4415 |
| `__version__` | `"0.1.0"` |
| `__license__` | license summary string |

## Element conversions

`kep_to_eci(kep) -> ndarray(6)`
Keplerian elements `[a_km, e, i_deg, raan_deg, argp_deg, true_anomaly_deg]` to
an ECI state. Requires `a_km > 0` and `0 <= e < 1`.

`eci_to_kep(eci) -> ndarray(6)`
The inverse. For near-circular orbits the argument of perigee and true anomaly
are individually ill-conditioned; their sum is stable.

## Propagation

`propagate(state_eci, epoch_jd, tof_s, times_s=None, *, gravity_order=5, drag=True, srp=True, sun=True, moon=True, atm_model=7, am_ratio=0.01, cd=2.2, cr=1.3, tol=1e-8, dt_max_s=300.0, method="vern7") -> dict`

Propagates one state for `tof_s` seconds from `epoch_jd` and samples it.
`times_s` is an increasing sequence of sample times in seconds after the
epoch, within `[0, tof_s]`; the grid always gains `0` and `tof_s` if they are
missing. Without `times_s` you get the start and end states only.

| kwarg | meaning |
|---|---|
| `gravity_order` | degree and order of the GOCE DIR-R6 field, 1 to 15 |
| `drag`, `srp`, `sun`, `moon` | force switches |
| `atm_model` | 4 exact JB2008, 7 fitted JB2008 (default), 8 campaign persistence scenario (valid 2026-08-15 to 2026-08-31 only), 1 exponential atmosphere. Ignored when `drag=False`. |
| `am_ratio` | area to mass ratio, m²/kg, must be > 0 |
| `cd` | drag coefficient, must be > 0 |
| `cr` | reflectivity coefficient, must be >= 0 |
| `tol` | integrator relative tolerance |
| `dt_max_s` | maximum step, seconds |
| `method` | `vern7`, `vern9`, `dop853`, `tsit5`, `rkv98`, `dopri5` |

Returned dict:

| key | type | meaning |
|---|---|---|
| `times_s` | ndarray(n) | sample times actually produced |
| `states_eci` | ndarray(n, 6) | ECI states at those times |
| `completed` | bool | the integrator reached `tof_s` and produced every requested sample |
| `terminal_event` | str | empty on success; otherwise the reason, for example `gravity_invalidradius` (below the surface), `eclipse_envelope` (state outside the SRP envelope), `incomplete` |
| `max_steps_exceeded` | bool | step budget exhausted |
| `steps` | int | accepted integrator steps |
| `rhs_evals` | int | force-model evaluations |
| `wall_us` | int | integration wall time, microseconds |

Raises `ValueError` for bad arguments or an epoch outside the ephemeris or
JB2008 driver coverage, and `RuntimeError` for an internal failure.

`propagate_final(state_eci, epoch_jd, tof_s, *, same model kwargs) -> ndarray(6)`

Same force model, but returns only the end state through the checked final
path, which enforces physical feasibility. Raises `ValueError` when the arc
hits the ground, escapes, leaves the valid eccentricity range, drops below
the Earth's surface, or leaves the SRP eclipse envelope (radius 6000 to
50000 km, speed at or below 20 km/s; pass `srp=False` for such arcs). Raises
`RuntimeError` for an integration failure.

## Transfer search

`TransferProblem(deployer_eci, target_eci, epoch_jd, *, controls...)`

Both states are given at `epoch_jd`. The controls and their defaults, which
are the campaign's sealed values:

| kwarg | default | meaning |
|---|---|---|
| `max_time_s` | 172800 | latest allowed intercept time after epoch |
| `max_phase_dv` | 1.25 | cap on the phasing burn, km/s |
| `max_transfer_dv` | 1.25 | cap on the transfer burn, km/s |
| `max_revs` | 4 | maximum Lambert revolutions considered |
| `min_perigee_km` | 6578.137 | lowest allowed perigee radius on any arc |
| `max_apogee_km` | 41378.137 | highest allowed apogee radius on any arc |
| `tof_penalty_weight` | 0.1 | weight of time of flight in the scalar cost, km/s per hour |
| `revolution_cap` | 2.0 | reject a candidate whose transfer time of flight exceeds this many deployer orbital periods |
| `distance_tol_km` | 0.025 | endpoint miss a candidate must close to under J2 |
| `deployer_min_distance_km` | 0.12 | minimum separation from the deployer at intercept |
| `tof_sample_budget` | 256 | time-of-flight samples per Lambert scan |
| `coarse_early_stop` | False | let the coarse seed scan stop early once a strong candidate is found |
| `fine_total_limit` | 10 | cap on the number of seeds admitted to the fine stage |
| `coarse_reject_margin_km_s` | 0.15 | coarse candidates worse than the best by more than this stay out of the fine stage |
| `seed_fine_margin_km_s` | 0.15 | seeds within this cost margin of the fine cutoff are still admitted |
| `j2_max_iterations` | 5 | J2 closure corrector iterations |
| `j2_endpoint_target_km` | 0.01 | J2 closure target miss |
| `j2_correction_step_gain` | 1.0 | J2 closure step gain |
| `seed` | 42 | seed for the local-optimizer restarts |
| `parallel` | True | use the Rayon pool |

Attributes: `epoch_jd`, `deployer_eci`, `target_eci`.

`TransferProblem.solve() -> list[dict]`

Runs the search and returns the valid Pareto-front candidates sorted by
`total_dv`. An empty list means nothing satisfied the caps. Candidate keys:

| key | type | meaning |
|---|---|---|
| `valid` | bool | always `True` in the returned list |
| `cost` | float | the optimizer's scalar cost |
| `time2phase_s` | float | coast on the phasing orbit before the wait |
| `waittime_s` | float | wait before the transfer burn |
| `tof_s` | float | Lambert arc time of flight |
| `total_time_s` | float | sum of the three, epoch to intercept |
| `phase_sma_km` | float | semi-major axis of the phasing orbit |
| `phase_dv`, `transfer_dv`, `arrival_dv` | ndarray(3) | burn vectors, km/s |
| `phase_dv_norm`, `transfer_dv_norm`, `arrival_dv_norm` | float | their magnitudes |
| `total_dv` | float | phase plus transfer; the arrival burn is not spent |
| `miss_distance_km` | float | endpoint miss under J2 |
| `deployer_distance_km` | float | payload to deployer distance at intercept |
| `release_state` | ndarray(6) | payload state just before the transfer burn |
| `payload_intercept_state`, `target_intercept_state`, `deployer_intercept_state` | ndarray(6) | states at intercept |
| `intercept_jd` | float | intercept epoch |
| `lambert_revs` | int | revolutions in the chosen Lambert solution |
| `prograde` | bool | direction of the Lambert solution |
| `branch_status`, `branch_rejection`, `timing_failure` | str | diagnostic tokens from the solver |
| `func_evals` | int | candidate evaluations spent |
| `optimizer_converged` | bool | local optimizer reported convergence |
| `post_hf_endpoint_residual_m` | float | filled by `hf_verify`; NaN otherwise |
| `time2phase_ratio`, `phase_sma_ratio`, `waittime_ratio` | float | the raw optimizer coordinates |

`TransferProblem.replay(candidate) -> dict`

Re-flies one candidate under the same J2 model the search used, from the five
control fields (`time2phase_s`, `waittime_s`, `tof_s`, `phase_dv`,
`transfer_dv`). Returns a candidate dict with the recomputed geometry plus
`residual_m`, the endpoint miss in metres. A candidate returned by `solve()`
replays to well under `distance_tol_km`.

`TransferProblem.hf_verify(candidate, *, forces="gravity", gravity_order=5, am_ratio=0.01, cd=2.2, cr=1.3, atm_model=7, tol=1e-8, dt_max_s=300.0, method="vern7") -> dict`

Re-flies one candidate's arcs under the high-fidelity propagator.
`forces="gravity"` uses spherical-harmonic gravity only; `forces="full"` adds
JB2008 drag, SRP, and Sun and Moon gravity for the vehicle described by
`am_ratio`, `cd` and `cr`. The target still follows J2, so the residual
isolates transfer-arc fidelity. Returned dict:

| key | meaning |
|---|---|
| `residual_m` | endpoint miss under HF, metres |
| `tolerance_m` | `distance_tol_km` in metres |
| `accepted` | `residual_m <= tolerance_m` |
| `replayed` | the candidate dict as re-flown under HF |

Expect residuals of tens to hundreds of km for multi-hour transfers; see the
README section "Things to know".

## Helper

`summarize(candidate) -> str`
One line: total and component delta-v, total and component time, endpoint
miss, relative speed at intercept, Lambert revolutions.
