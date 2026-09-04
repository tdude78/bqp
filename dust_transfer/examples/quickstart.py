# Copyright (c) 2026 Truman DeWalch. All rights reserved.
# Licensed under the PolyForm Strict License 1.0.0 with the additional
# evaluation permission stated in LICENSE.md. Use only; no changes, no
# distribution, no commercial use. This is dust_transfer MVP 0.1.0.
"""End-to-end demo: search for a transfer, verify it, propagate a state.

Run:  python examples/quickstart.py
"""

import time

import numpy as np

import dust_transfer as dt

EPOCH_JD = 2460310.5  # 2024-01-01 00:00:00 UTC

# Deployer in a ~800 km sun-synchronous-like orbit; target 100 km higher,
# slightly different plane, different phase.
deployer = dt.kep_to_eci([7178.137, 0.001, 97.4, 125.0, 210.0, 180.0])
target = dt.kep_to_eci([7278.137, 0.002, 97.6, 128.0, 40.0, 10.0])
print("deployer ECI:", deployer)
print("target   ECI:", target)

# ---------------------------------------------------------------- 1. search
problem = dt.TransferProblem(deployer, target, EPOCH_JD, max_time_s=86_400.0)
t0 = time.perf_counter()
front = problem.solve()
print(f"\nsolve(): {len(front)} valid candidates in {time.perf_counter() - t0:.1f} s")
for c in front[:5]:
    print("  ", dt.summarize(c))

if not front:
    raise SystemExit("no valid transfer found; widen max_time_s or the delta-v caps")
best = front[0]

# ---------------------------------------------------------------- 2. verify
mf = problem.replay(best)
print(f"\nMF (J2) replay miss:        {mf['residual_m']:.3f} m")

hf_g = problem.hf_verify(best, forces="gravity")
print(f"HF gravity-only replay miss: {hf_g['residual_m']:.1f} m (tol {hf_g['tolerance_m']:.0f} m, accepted={hf_g['accepted']})")

hf_f = problem.hf_verify(best, forces="full", am_ratio=0.01, cd=2.2, cr=1.3)
print(f"HF full-force replay miss:   {hf_f['residual_m']:.1f} m (accepted={hf_f['accepted']})")

# ---------------------------------------------------------------- 3. propagate
times = np.arange(60.0, 5400.0 + 1e-9, 60.0)
t0 = time.perf_counter()
out = dt.propagate(deployer, EPOCH_JD, tof_s=5400.0, times_s=times)
print(
    f"\npropagate(): {out['states_eci'].shape} samples, completed={out['completed']}, "
    f"{out['steps']} steps / {out['rhs_evals']} RHS evals in {time.perf_counter() - t0:.2f} s"
)
final = out["states_eci"][-1]
print("final ECI  :", final)
print("final kep  :", dt.eci_to_kep(final))

# Final state only, with impact / escape checks (raises ValueError if infeasible).
print("propagate_final:", dt.propagate_final(deployer, EPOCH_JD, tof_s=5400.0)[:3])

# Same arc with drag/SRP/third-body switched off: only spherical-harmonic gravity.
grav_only = dt.propagate(deployer, EPOCH_JD, tof_s=5400.0, drag=False, srp=False, sun=False, moon=False)
sep_m = np.linalg.norm(final[:3] - grav_only["states_eci"][-1][:3]) * 1000
print(f"perturbation effect over 90 min (full - gravity only): {sep_m:.1f} m")
