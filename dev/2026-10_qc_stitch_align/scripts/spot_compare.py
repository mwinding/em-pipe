"""Before/after stacks for align's tile-level solve windows, as side-by-side pairs.

For every window, two ImageJ TIFFs in qc/spots_before_after/ (one slice per z):
  *_centres.tif: a CROP px crop at the centre of every tile, laid out like the tiles;
  *_seams.tif:   a CROP px crop at the middle of every overlap between neighbouring tiles.
Each crop is a pair: rendered with align o stitch (left, before) and with the tile-level solve
(right, after), 4 px apart; pairs are separated by grey bars. Full resolution, no CLAHE.
index.csv adds the solve's numbers from align/local_windows.csv.
"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import config as C, elastic, render, transforms  # noqa: E402
from pipeline.slices import load_slices  # noqa: E402

CROP, INNER, BAR, BAR_GREY, SEAMS_PER_ROW = 500, 4, 24, 128, 2

cfg = C.load_config(sys.argv[1], step_defaults=C.step_defaults())
root = Path(cfg["output_dir"])
W, out = root / "work", root / "qc" / "spots_before_after"
out.mkdir(parents=True, exist_ok=True)
spots = pd.read_csv(root / "qc" / "spots" / "index.csv")
lw = pd.read_csv(W / "align" / "local_windows.csv")

after = render._tile_plan(cfg)
local = transforms.read_csv(W / "align" / "local_tiles.csv", ["z", "tile"])
align_T = transforms.read_csv(W / "align" / "transforms.csv", ["z"])
stitch = transforms.read_csv(W / "stitch" / "tiles.csv", ["z", "tile"])
before = after.copy()
is_local = np.array([(z, t) in local for z, t in zip(before["z"], before["tile"])])
before.loc[is_local, transforms.COLUMNS] = np.array(
    [transforms.compose(align_T[z], stitch[(z, t)]) for z, t in zip(before.loc[is_local, "z"], before.loc[is_local, "tile"])]
).reshape(-1, 6)
settings = render._settings(cfg)
settings["clahe"] = {**settings["clahe"], "enabled": False}
fields = elastic.load(cfg)
stamp = load_slices(cfg).groupby("z")["timestamp"].first()
um = 0.001 * float(cfg["check"]["voxel_size_nm"][0])
PAIR = 2 * CROP + INNER


def tile_rc(t):
    return tuple(map(int, t.split("-")))


def crops_for(here):
    """{'centres': [(name, (row, col), xy)], 'seams': [(name, (row, col), xy)]} in aligned px."""
    T = {r["tile"]: (transforms.from_row(r), r["width"], r["height"]) for _, r in here.iterrows()}
    centres = [(t, tile_rc(t), transforms.apply(A, [[w / 2, h / 2]])[0]) for t, (A, w, h) in T.items()]
    seams = []
    for ta in sorted(T, key=tile_rc):
        for tb in sorted(T, key=tile_rc):
            (ra, ca), (rb, cb) = tile_rc(ta), tile_rc(tb)
            if not ((ra == rb and cb == ca + 1) or (ca == cb and rb == ra + 1)):
                continue
            ba = transforms.bbox(T[ta][0], T[ta][1], T[ta][2])
            bb = transforms.bbox(T[tb][0], T[tb][1], T[tb][2])
            x0, y0 = np.maximum(ba[:2], bb[:2])
            x1, y1 = np.minimum(ba[2:], bb[2:])
            if x1 > x0 and y1 > y0:
                seams.append((f"{ta}|{tb}", None, np.array([(x0 + x1) / 2, (y0 + y1) / 2])))
    seams = [(n, (i // SEAMS_PER_ROW, i % SEAMS_PER_ROW), xy) for i, (n, _, xy) in enumerate(seams)]
    return {"centres": centres, "seams": seams}


rows = []
with ThreadPoolExecutor(8) as pool:
    for _, w in lw.iterrows():
        sp = spots[(spots["z_first"] == w["z_first"]) & (spots["z_last"] == w["z_last"])]
        k = int(sp["spot"].iloc[0]) if len(sp) else 0
        why = sp["why"].iloc[0] if len(sp) else w["reason"]
        f0, f1 = (map(int, sp["flagged_z"].iloc[0].split("-")) if len(sp) else (w["z_first"], w["z_last"]))
        span = sorted(before.loc[(before["z"] >= w["z_first"]) & (before["z"] <= w["z_last"]), "z"].unique())
        zc = int(span[len(span) // 2])
        subs = {side: plan[plan["z"].isin(span)] for side, plan in (("before", before), ("after", after))}
        files = {}
        for kind, crops in crops_for(before[before["z"] == zc]).items():
            nr = max(rc[0] for _, rc, _ in crops) + 1
            nc = max(rc[1] for _, rc, _ in crops) + 1
            H, Wd = nr * CROP + (nr - 1) * BAR, nc * PAIR + (nc - 1) * BAR
            rend = {}
            for name, rc, xy in crops:
                meta = {"downsample": 1, "origin_xy": [float(np.floor(xy[0] - CROP / 2)), float(np.floor(xy[1] - CROP / 2))],
                        "shape": [1, CROP, CROP], "settings": settings, "destreak": None, "elastic": fields}
                for side in ("before", "after"):
                    rend[(name, rc, side)] = render._Renderer(cfg, meta, subs[side], pool)
            stack = np.full((len(span), H, Wd), BAR_GREY, np.uint8)
            labels = []
            order = " ".join(n for n, _, _ in crops)
            for i, z in enumerate(span):
                for (name, (r, c), side), rr in rend.items():
                    x = c * (PAIR + BAR) + (0 if side == "before" else CROP + INNER)
                    stack[i, r * (CROP + BAR):r * (CROP + BAR) + CROP, x:x + CROP] = rr.plane([(int(z), 1.0)])
                    if side == "before":
                        stack[i, r * (CROP + BAR):r * (CROP + BAR) + CROP, x + CROP:x + CROP + INNER] = 0
                flag = " <" if f0 <= z <= f1 else ""
                labels.append(f"z {int(z)} {stamp.loc[int(z)]:%d %b %H:%M:%S}{flag} | each pair: LEFT before, RIGHT after | "
                              f"{kind}: {order} | spot {k}: {why}")
            name = f"spot{k:02d}_z{int(span[0])}-{int(span[-1])}_{kind}.tif"
            tifffile.imwrite(out / name, stack, imagej=True, resolution=(1 / um, 1 / um),
                             metadata={"spacing": um, "unit": "micron", "axes": "ZYX", "Labels": labels})
            files[kind] = name
        rows.append({"spot": k, "centres": files["centres"], "seams": files["seams"], "z_first": int(span[0]),
                     "z_last": int(span[-1]), "why": why,
                     "mismatch_median_before_px": round(w["median_before_px"], 2),
                     "mismatch_median_after_px": round(w["median_after_px"], 2),
                     "mismatch_p90_before_px": round(w["p90_before_px"], 2),
                     "mismatch_p90_after_px": round(w["p90_after_px"], 2),
                     "seams_moved_max_px": round(w["seam_max_after_px"], 2)})
        print(f"spot {k:2d}: z {span[0]}-{span[-1]} done", flush=True)
pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
print(f"{len(rows)} windows -> {out}")
