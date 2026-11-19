"""Re-run the tile-level solve of one window with other rejection / seam settings (writes nothing);
report the mismatch across the layout change over ALL matched points, per old tile."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys

import numpy as np

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import align as al, tilesolve, transforms  # noqa: E402
from pipeline.config import load_config  # noqa: E402

cfg = load_config(sys.argv[1], step_defaults=al.DEFAULTS)
z_old = int(sys.argv[2])
a = cfg["align"]
df = al.load_slices(cfg)
zs = np.array(al.z_values(df))
run, _ = al._runs(df, zs)
T = {int(z): A for z, A in transforms.read_csv(al.step_path(cfg, "align", "transforms.csv"), ["z"]).items()}
stitch = al._stitch_transforms(cfg, df)
shapes = {(z, t): (int(h), int(w)) for z, t, h, w in zip(df["z"], df["tile"], df["height"], df["width"])}
k = int(np.searchsorted(zs, z_old)) + 1
i0, i1 = max(0, k - 50), min(len(zs) - 1, k + 49)
lo, hi = max(0, i0 - 4), min(len(zs) - 1, i1 + 4)
inside = set(zs[i0:i1 + 1].tolist())
T0 = {key: transforms.compose(T[key[0]], stitch[key]) for key in shapes if zs[lo] <= key[0] <= zs[hi]}
m = al._tile_matches(cfg, df, zs, a, int(zs[lo]), int(zs[hi]))
sel = np.isin(m["za"], list(inside)) | np.isin(m["zb"], list(inside))
m = {key: v[sel] for key, v in m.items()}
across_all = (m["za"] <= z_old) & (m["zb"] > z_old)


def run_case(name, robust_iterations, reject_px, seam_weight):
    tilesolve.SEAM_WEIGHT = seam_weight
    out, rep, (before, after, keep, ok) = tilesolve.solve_window(
        T0, shapes, m, inside, {**a, "robust_iterations": robust_iterations, "reject_px": reject_px})
    across = across_all[ok]
    ta = m["ta"][ok][across]
    s = after[across]
    cols = " ".join(f"{t}:{np.median(s[ta == t]):4.1f}" for t in sorted(set(ta.tolist())))
    print(f"{name:42s} across change: median {np.median(s):5.1f} p90 {np.percentile(s, 90):5.1f} | "
          f"all matches median {np.median(after):4.2f} | seams moved max {rep['seam_max_after_px']:5.2f} | "
          f"kept {int(keep.sum())}/{len(keep)} | per old tile {cols}", flush=True)


run_case("as committed (3 rounds, reject 4 px, seam x4)", 3, 4.0, 4.0)
run_case("no rejection, seam x4", 0, 4.0, 4.0)
run_case("reject > 50 px only, seam x4", 3, 50.0, 4.0)
run_case("no rejection, seam x1", 0, 4.0, 1.0)
run_case("no rejection, seam x0.25", 0, 4.0, 0.25)
