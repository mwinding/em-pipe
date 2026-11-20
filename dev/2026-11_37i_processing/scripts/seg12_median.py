import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import glob, json
import numpy as np
D = os.path.expandvars("${DEV_37I}/work/stitch/samples")
recs = [json.load(open(p)) for p in sorted(glob.glob(D + "/z*.json"))]
ADJ = {("0-0", "0-1"), ("1-0", "1-1"), ("0-0", "1-0"), ("0-1", "1-1")}


def n_adj(r):
    return sum(1 for p in r["pairs"] if p["used"] and (p["tile_a"], p["tile_b"]) in ADJ)


def rel(r):
    t = {k: np.ravel(v) for k, v in r["tiles"].items()}
    return {"0-1": t["0-1"][[2, 5]], "1-0": t["1-0"][[2, 5]], "1-1 - 0-1": t["1-1"][[2, 5]] - t["0-1"][[2, 5]],
            "1-1 - 1-0": t["1-1"][[2, 5]] - t["1-0"][[2, 5]]}


def summary(name, rs):
    if not rs:
        print(name, "none")
        return
    zs = [r["z"] for r in rs]
    m = {k: np.median([rel(r)[k] for r in rs], axis=0) for k in rel(rs[0])}
    print(f"{name}: n {len(rs)}, z {min(zs)}-{max(zs)} | " + ", ".join(f"{k} ({v[0]:.1f}, {v[1]:.1f})" for k, v in m.items()))


for seg in (10, 11, 12):
    good = [r for r in recs if r["ok"] and r["segment"] == seg]
    print(f"-- segment {seg}: adjacent pairs used:", {k: sum(n_adj(r) == k for r in good) for k in range(5)})
    summary("  all good", good)
    summary("  4 adjacent pairs", [r for r in good if n_adj(r) == 4])
    summary("  3 adjacent pairs", [r for r in good if n_adj(r) == 3])
    if seg == 12:
        four = [r for r in good if n_adj(r) == 4]
        print("  z of 4-pair samples:", [r["z"] for r in four][:10], "...", [r["z"] for r in four][-10:])
        a = np.array([rel(r)["1-1 - 0-1"] for r in four])
        print("  4-pair spread of 1-1 - 0-1: x", np.percentile(a[:, 0], [0, 5, 50, 95, 100]).round(1),
              "y", np.percentile(a[:, 1], [0, 5, 50, 95, 100]).round(1))
        b = np.array([rel(r)["0-1"] for r in four])
        print("  4-pair spread of 0-1: x", np.percentile(b[:, 0], [0, 5, 50, 95, 100]).round(1),
              "y", np.percentile(b[:, 1], [0, 5, 50, 95, 100]).round(1))
