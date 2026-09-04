# Copyright (c) 2026 Truman DeWalch. All rights reserved.
# Licensed under the PolyForm Strict License 1.0.0 with the additional
# evaluation permission stated in LICENSE.md. Use only; no changes, no
# distribution, no commercial use. This is dust_transfer MVP 0.1.0.
"""dust_transfer: a two-phase orbital transfer optimizer and a high-fidelity propagator.

Units everywhere: km, km/s, seconds, Julian Date (UTC).

Quick start::

    import numpy as np
    import dust_transfer as dt

    epoch = 2460310.5                                  # 2024-01-01 00:00 UTC
    deployer = dt.kep_to_eci([7178.137, 0.001, 97.4, 125.0, 210.0, 180.0])
    target   = dt.kep_to_eci([7278.137, 0.002, 97.6, 128.0,  40.0,  10.0])

    # 1. Search for intercept transfers (medium fidelity, J2-closed).
    problem = dt.TransferProblem(deployer, target, epoch, max_time_s=86400.0)
    front = problem.solve()                            # list of candidate dicts
    best = front[0]

    # 2. Re-fly the best one under the high-fidelity model.
    report = problem.hf_verify(best, forces="full")
    print(report["residual_m"], report["accepted"])

    # 3. Propagate any state with the high-fidelity model.
    out = dt.propagate(deployer, epoch, tof_s=5400.0, times_s=np.arange(60, 5401, 60))
    out["states_eci"]                                  # (n, 6) array
    dt.propagate_final(deployer, epoch, tof_s=5400.0)  # final state only, with impact/escape checks
"""

from ._native import (
    MU_EARTH_KM3_S2,
    TransferProblem,
    eci_to_kep,
    kep_to_eci,
    propagate,
    propagate_final,
)

__version__ = "0.1.0"
__license__ = "PolyForm Strict License 1.0.0 (evaluation snapshot; see LICENSE.md)"
__author__ = "Truman DeWalch"

__all__ = [
    "MU_EARTH_KM3_S2",
    "TransferProblem",
    "eci_to_kep",
    "kep_to_eci",
    "propagate",
    "propagate_final",
    "summarize",
]


def summarize(candidate: dict) -> str:
    """One-line human summary of a candidate returned by ``TransferProblem.solve``."""
    return (
        f"dv={candidate['total_dv']:.4f} km/s "
        f"(phase {candidate['phase_dv_norm']:.4f} + transfer {candidate['transfer_dv_norm']:.4f}), "
        f"time={candidate['total_time_s'] / 3600:.2f} h "
        f"(phase {candidate['time2phase_s'] / 3600:.2f} + wait {candidate['waittime_s'] / 3600:.2f} "
        f"+ tof {candidate['tof_s'] / 3600:.2f}), "
        f"miss={candidate['miss_distance_km'] * 1000:.2f} m, "
        f"v_rel@intercept={candidate['arrival_dv_norm']:.3f} km/s, revs={candidate['lambert_revs']}"
    )
