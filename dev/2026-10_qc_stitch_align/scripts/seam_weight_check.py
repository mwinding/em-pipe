"""Re-solve every configured tile-level window with several seam weights (writes nothing): mismatch
over all matches before/after, and how far the seams move, per window and in total."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys

import numpy as np

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import align as al, tilesolve, transforms  # noqa: E402
from pipeline.config import load_config  # noqa: E402

cfg = load_config(sys.argv[1], step_defaults=al.DEFAULTS)
weights = [float(w) for w in sys.argv[2].split(",")]
a = cfg["align"]
df = al.load_slices(cfg)
zs = np.array(al.z_values(df))
run, _ = al._runs(df, zs)
T = transforms.read_csv(al.step_path(cfg, "align", "transforms.csv"), ["z"])
stitch = al._stitch_transforms(cfg, df)
shapes = {(z, t): (int(h), int(w)) for z, t, h, w in zip(df["z"], df["tile"], df["height"], df["width"])}
nb = int(a["neighbors"])
results = {w: [] for w in weights}
for i0, i1, reason in tilesolve.windows(zs, run, a):
    lo, hi = max(0, i0 - nb), min(len(zs) - 1, i1 + nb)
    inside = set(zs[i0:i1 + 1].tolist())
    T0 = {k: transforms.compose(T[k[0]], stitch[k]) for k in shapes if zs[lo] <= k[0] <= zs[hi]}
    m = al._tile_matches(cfg, df, zs, a, int(zs[lo]), int(zs[hi]))
    sel = np.isin(m["za"], list(inside)) | np.isin(m["zb"], list(inside))
    m = {k: v[sel] for k, v in m.items()}
    line = [f"z {zs[i0]:>5}-{zs[i1]:<5}"]
    for w in weights:
        tilesolve.SEAM_WEIGHT = w
        _, rep, _ = tilesolve.solve_window(T0, shapes, m, inside, a)
        results[w].append(rep)
        line.append(f"w{w:g}: med {rep['median_before_px']:.2f}->{rep['median_after_px']:.2f} "
                    f"p90 {rep['p90_before_px']:.2f}->{rep['p90_after_px']:.2f} seam max {rep['seam_max_after_px']:.2f}")
    print("  ".join(line), flush=True)
print("SUMMARY over windows (mean of per-window values):")
for w in weights:
    r = results[w]
    f = lambda k: np.mean([x[k] for x in r])
    print(f"  seam weight {w:g}: median {f('median_before_px'):.2f} -> {f('median_after_px'):.2f} px, "
          f"p90 {f('p90_before_px'):.2f} -> {f('p90_after_px'):.2f} px, seam moved max (worst window) "
          f"{max(x['seam_max_after_px'] for x in r):.2f} px, mean of window maxima {f('seam_max_after_px'):.2f} px")
