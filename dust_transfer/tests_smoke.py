# Copyright (c) 2026 Truman DeWalch. All rights reserved.
# Licensed under the PolyForm Strict License 1.0.0 with the additional
# evaluation permission stated in LICENSE.md. Use only; no changes, no
# distribution, no commercial use. This is dust_transfer MVP 0.1.0.
"""Minimal correctness smoke. Run: python tests_smoke.py"""
import numpy as np
import dust_transfer as dt

EPOCH = 2460310.5
kep = np.array([7178.137, 0.01, 97.4, 125.0, 210.0, 180.0])
eci = dt.kep_to_eci(kep)
back = dt.eci_to_kep(eci)
assert np.allclose(kep, back, rtol=1e-6, atol=1e-6), (kep, back)

# Gravity-only, two-body-ish: one period should come back near the start.
out = dt.propagate(eci, EPOCH, tof_s=600.0, gravity_order=1, drag=False, srp=False, sun=False, moon=False)
assert out["completed"], out
r0 = np.linalg.norm(eci[:3]); r1 = np.linalg.norm(out["states_eci"][-1][:3])
assert abs(r1 - r0) < 200.0, (r0, r1)

# Full force model runs and stays finite.
full = dt.propagate(eci, EPOCH, tof_s=3600.0, times_s=np.arange(600.0, 3601.0, 600.0))
assert full["completed"] and np.isfinite(full["states_eci"]).all()
assert full["states_eci"].shape == (7, 6), full["states_eci"].shape  # t=0 row + 6 samples

# Final-state path agrees with the sampled path and enforces physical checks.
fs = dt.propagate_final(eci, EPOCH, tof_s=3600.0)
assert np.linalg.norm(fs[:3] - full["states_eci"][-1][:3]) < 1e-3, "final vs sampled disagree"
try:
    dt.propagate_final(dt.kep_to_eci([6600.0, 0.05, 30.0, 0.0, 0.0, 0.0]), EPOCH, 6000.0, srp=False)
    raise AssertionError("impacting arc did not raise")
except ValueError:
    pass

# Transfer search finds something and replays to a sub-metre J2 miss.
target = dt.kep_to_eci([7278.137, 0.002, 97.6, 128.0, 40.0, 10.0])
problem = dt.TransferProblem(eci, target, EPOCH, max_time_s=43_200.0)
front = problem.solve()
assert front, "empty front"
mf = problem.replay(front[0])
assert mf["residual_m"] < 25.0, mf["residual_m"]  # inside the 25 m verification tolerance
hf = problem.hf_verify(front[0])
assert np.isfinite(hf["residual_m"])
print("OK:", len(front), "candidates; MF miss", round(mf["residual_m"], 3), "m; HF miss", round(hf["residual_m"], 1), "m")
