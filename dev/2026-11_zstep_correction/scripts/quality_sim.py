"""Simulated slice pairs: what zcorrect's solve makes of slices that only differ in quality, not in thickness.

    python dev/2026-11_zstep_correction/scripts/quality_sim.py

NCC(a, b) = 0.76 exp(-distance / 16) q_a q_b (the level and slope of A1001's and 37i's measured curves: 0.73-0.76
between neighbours, ~0.03 less per further slice), with a true spacing SD and a per-slice factor q SD. Prints the
SD of the solved spacing / nominal, its correlation with the true spacing, and step_stats.quality_model's shares.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_stats  # noqa: E402  (puts the repository root and ../zcorrect_step on sys.path)
import zcorrect  # noqa: E402

zc = zcorrect.DEFAULTS["zcorrect"]
rng = np.random.default_rng(0)
n = 1500
zs, brk = np.arange(n), np.zeros(n - 1, bool)
for thick_sd, q_sd in [(0.15, 0.0), (0.0, 0.01), (0.0, 0.03), (0.15, 0.03)]:
    s = np.clip(1 + thick_sd * rng.standard_normal(n - 1), 0.3, None)
    s /= s.mean()
    pos, q = np.r_[0, np.cumsum(s)], np.exp(q_sd * rng.standard_normal(n))
    ia = np.repeat(np.arange(n), 8)
    ib = ia + np.tile(np.arange(1, 9), n)
    ia, ib = ia[ib < n], ib[ib < n]
    ncc = 0.76 * np.exp(-(pos[ib] - pos[ia]) / 16) * q[ia] * q[ib] * np.exp(0.005 * rng.standard_normal(len(ia)))
    solved = step_stats.solve_pairs(zs, brk, ia, ib, ncc, zc)
    r = np.corrcoef(s, solved)[0, 1] if thick_sd else float("nan")
    print(f"true spacing SD {s.std():.3f}, slice factor SD {q_sd}: solved SD {np.nanstd(solved):.3f}, r with true {r:.2f};",
          step_stats.quality_model(zs, brk, ia, ib, ncc, zc))
