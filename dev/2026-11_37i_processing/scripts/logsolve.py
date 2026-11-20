import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys

import numpy as np

from pipeline import patchgrid
from pipeline.config import load_config, deep_merge
from pipeline.intensity import solve, connected, GAIN_PRIOR

cfg = load_config(os.path.expandvars("configs/lab/${DEV_PREFIX}37i.yaml"))
p = deep_merge(patchgrid.DEFAULTS, {"patchgrid": cfg.get("patchgrid") or {}})["patchgrid"]
shapes, R = patchgrid.load(cfg)
raw = patchgrid.read_bins(cfg)


def gains_log(rels):
    tiles = sorted({t for r in rels for t in r[:2]})
    k = {t: i for i, t in enumerate(tiles)}
    w = np.sqrt([r[4] for r in rels])
    wg = w * np.array([r[5] for r in rels])
    big = 100 * w.max()
    G = np.zeros((len(rels) + 1, len(tiles)))
    rhs = np.zeros(len(rels) + 1)
    for j, (a, b, g_ab, *_) in enumerate(rels):
        G[j, k[a]], G[j, k[b]] = wg[j], -wg[j]   # log g_a + log g_ab - log g_b = 0
        rhs[j] = -wg[j] * np.log(g_ab)
    G[-1] = big
    prior = GAIN_PRIOR * w.max() * np.eye(len(tiles))
    x = np.linalg.lstsq(np.vstack([G, prior]), np.r_[rhs, np.zeros(len(tiles))], rcond=None)[0]
    return {t: float(np.exp(x[i])) for t, i in k.items()}


for z in (2000, 9600, 14000, 15100, 16000, 16530, 17500):
    b = patchgrid.shape_corrected(raw[raw.z == z].copy(), shapes, R)
    b["gw"] = patchgrid.bin_weights(b, p["gain_power"], p["gain_cnr_min"])[1]
    rels = connected([(r.a, r.b, r.gain, r.offset, r.n, r.gw, r.centre) for r in b.itertuples()])
    now = {t: g for t, (g, o) in solve(rels).items()}
    lg = gains_log(rels)
    print(f"z {z}: now " + ", ".join(f"{t} {g:.3f}" for t, g in now.items()) + "  | log " +
          ", ".join(f"{t} {g:.3f}" for t, g in lg.items()), file=sys.stderr)
