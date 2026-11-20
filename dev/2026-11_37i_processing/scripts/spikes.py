import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import glob, json
from collections import defaultdict
import numpy as np
D = os.path.expandvars("${DEV_37I}/work/stitch/samples")
recs = [json.load(open(p)) for p in sorted(glob.glob(D + "/z*.json"))]
by_seg = defaultdict(list)
for r in recs:
    if r["ok"]:
        by_seg[r["segment"]].append(r)


def pinfo(r):
    return " ".join(f"{p['tile_a']}/{p['tile_b']}:{p['coarse_inliers']}/{p['n_inliers']}"
                    f"{'' if p['used'] else 'x'} r{p.get('residual_px') or 0:.1f}" for p in r["pairs"] if p['coarse_inliers'])


for seg in sorted(by_seg):
    rs = by_seg[seg]
    tiles = sorted({t for r in rs for t in r["tiles"]})
    med = {t: np.median([np.ravel(r["tiles"][t])[[2, 5]] for r in rs if t in r["tiles"]], axis=0) for t in tiles}
    zs = sorted(r["z"] for r in rs)
    print(f"segment {seg}: z {zs[0]}-{zs[-1]}, {len(rs)} good samples, median offsets "
          + ", ".join(f"{t} ({m[0]:.0f}, {m[1]:.0f})" for t, m in med.items()))
    if seg == 12:
        continue
    for r in rs:
        dev = {t: float(np.linalg.norm(np.ravel(r["tiles"][t])[[2, 5]] - med[t])) for t in r["tiles"]}
        if max(dev.values()) > 12:
            print(f"   z {r['z']} ({r['timestamp']}): dev " + ", ".join(f"{t} {d:.0f}" for t, d in dev.items() if d > 3)
                  + " | " + pinfo(r))
