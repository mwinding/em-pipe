import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import glob, json
import numpy as np
D = os.path.expandvars("${DEV_37I}/work/stitch/samples")
recs = {}
for p in sorted(glob.glob(D + "/z*.json")):
    r = json.load(open(p))
    if r["z"] >= 14900:
        recs[r["z"]] = r


def pinfo(r):
    return " ".join(f"{p['tile_a']}/{p['tile_b']}:{p['coarse_inliers']}/{p['n_inliers']}"
                    f"{'' if p['used'] else 'x'}{'(' + p['note'] + ')' if p.get('note') else ''}" for p in r["pairs"])


for z in (14999, 15049, 15349, 15599, 15999, 16249, 16299, 16349, 16399, 16449, 16749, 17349, 17399, 17926):
    r = recs.get(z)
    if r:
        tx = ({t: (round(float(np.ravel(v)[2])), round(float(np.ravel(v)[5]))) for t, v in r["tiles"].items()}
              if r["ok"] else r.get("reason"))
        print(z, "ok" if r["ok"] else "FAIL", "cf", r.get("coarse_factor_used"), "|", pinfo(r), "|", tx)
# right-column offset (0-1 relative to 0-0) over the good samples of segment 12
rows = []
for z, r in sorted(recs.items()):
    if r["ok"] and r["segment"] == 12 and "0-1" in r["tiles"] and "0-0" in r["tiles"]:
        a, b = np.ravel(r["tiles"]["0-0"]), np.ravel(r["tiles"]["0-1"])
        rows.append((z, b[2] - a[2], b[5] - a[5]))
rows = np.array(rows)
for lo in range(15000, 18000, 250):
    s = rows[(rows[:, 0] >= lo) & (rows[:, 0] < lo + 250)]
    if len(s):
        print(f"z {lo}-{lo + 249}: n {len(s)}, 0-1 minus 0-0 x median {np.median(s[:, 1]):.1f} "
              f"[{s[:, 1].min():.1f}, {s[:, 1].max():.1f}], y median {np.median(s[:, 2]):.1f} "
              f"[{s[:, 2].min():.1f}, {s[:, 2].max():.1f}]")
