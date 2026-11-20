# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-14T00:46:38.365Z
# Ran: remote, via ssh
# Heredoc line: cd ${DEV_WORK} && cat > baseline.py <<'EOF'
# Assistant note before it: The matched-seams test is running: - **Alignment re-solve:** job 64006864, a couple of minutes. - **z 2083 slab re-render:** follows automatically, about 15 minutes. Once the slab is ready I'll run the shift check (job script ready) and look at the numbers and the worst spot myself before deciding o

"""Baseline: the same per-tile residuals (matches to the next 4 slices) for normal slices, before and after a one-slice window."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
import numpy as np
from pipeline import align, tilesolve, transforms
from pipeline.config import load_config, step_path
from pipeline.slices import load_slices

cfg = load_config(os.path.expandvars("configs/lab/${DEV_PREFIX}37i.yaml"), step_defaults=align.DEFAULTS)
a0 = {**cfg["align"], "local_warp_px": None}
df = load_slices(cfg)
zs = np.array(sorted(df["z"].unique()))
mats = transforms.read_csv(step_path(cfg, "align", "transforms.csv"), ["z"])
stitch = transforms.read_csv(step_path(cfg, "stitch", "tiles.csv"), ["z", "tile"])
for zc in (2090, 2100, 2500, 9600):
    around = list(range(zc - 4, zc + 5))
    sub = df[df["z"].isin(around)]
    shapes = {(z, t): (int(h), int(w)) for z, t, h, w in zip(sub["z"], sub["tile"], sub["height"], sub["width"])}
    T0 = {k: transforms.compose(mats[k[0]], stitch[k]) for k in shapes}
    m = align._tile_matches(cfg, df, zs, a0, zc - 4, zc + 4)
    sel = (m["za"] == zc) & (m["zb"] > zc)          # matches to the next slices only, as for z 2083
    m = {k: v[sel] for k, v in m.items()}
    def per_tile(out):
        res = {}
        for za, ta, qa, zb, tb, qb in zip(m["za"], m["ta"], m["qa"], m["zb"], m["tb"], m["qb"]):
            pa = transforms.apply(out.get((int(za), ta), T0[(int(za), ta)]), qa[None])[0]
            pb = transforms.apply(T0[(int(zb), tb)], qb[None])[0]
            res.setdefault(ta, []).append(np.linalg.norm(pa - pb))
        allv = np.concatenate([np.asarray(v) for v in res.values()])
        return ({t: (round(float(np.median(v)), 1), round(float(np.percentile(v, 90)), 1)) for t, v in sorted(res.items())},
                (round(float(np.median(allv)), 2), round(float(np.percentile(allv, 90)), 2)))
    out, rep, _, _ = tilesolve.solve_window(T0, shapes, m, {zc}, a0)
    print(f"z {zc}: before (global solve) per tile {per_tile({})[0]}, all {per_tile({})[1]}; "
          f"after its own tile solve all {per_tile(out)[1]}", file=sys.stderr)
