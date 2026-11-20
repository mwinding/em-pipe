"""Spot-check GIFs of a run's flagged stretches: a CROP x CROP px crop at full resolution from the
centre of every tile (laid out like the tiles), stepping through the slices and back.

Flagged: slices in align pairs that were rejected, tile layout changes, and tile jumps between
consecutive slices larger than JUMP px; events within MERGE slices are one spot, padded by PAD
slices either side. Rendered with the run's own plan (stitch, align, elastic, intensity levels,
blend) but without CLAHE, so contrast doesn't change from crop to crop. Writes qc/spots/*.gif and
qc/spots/index.csv.
"""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
from pipeline import config as C, elastic, render, transforms  # noqa: E402
from pipeline.slices import load_slices, z_values  # noqa: E402

CROP, PAD, MERGE, JUMP, FRAME_MS = 500, 6, 15, 15.0, 150

cfg = C.load_config(sys.argv[1], step_defaults=C.step_defaults())
root = Path(cfg["output_dir"])
W, out = root / "work", root / "qc" / "spots"
out.mkdir(parents=True, exist_ok=True)
zs = np.array(z_values(load_slices(cfg)))
pos = {int(z): i for i, z in enumerate(zs)}

# ---- flagged events -> spots
events = []
res = pd.read_csv(W / "align" / "residuals.csv")
rej = res[res["rejected"]]
events += [(int(z), "rejected align pairs") for z in sorted(set(rej["z_a"]) | set(rej["z_b"]))]
ch = pd.read_csv(W / "align" / "segment_changes.csv")
events += [(int(r.z_after), f"layout change z {r.z_before}->{r.z_after}") for r in ch.itertuples()]
t = pd.read_csv(W / "stitch" / "tiles.csv", dtype={"tile": str}).sort_values(["segment", "tile", "z"])
d = np.hypot(t.groupby(["segment", "tile"])["tx"].diff(), t.groupby(["segment", "tile"])["ty"].diff())
step = t.assign(d=d).groupby("z")["d"].max().dropna()
events += [(int(z), f"tile jump {v:.0f} px") for z, v in step[step > JUMP].items()]
events = sorted(e for e in events if e[0] in pos)
spots = []
for z, why in events:
    if spots and z - spots[-1]["z1"] <= MERGE:
        spots[-1]["z1"] = z
        spots[-1]["why"].append(why)
    else:
        spots.append({"z0": z, "z1": z, "why": [why]})


def summary(whys):
    kinds = {}
    for w in whys:
        k = "tile jumps" if w.startswith("tile jump") else w
        kinds[k] = kinds.get(k, 0) + 1
    big = max([float(w.split()[2]) for w in whys if w.startswith("tile jump")], default=0)
    return "; ".join(f"{k}" + (f" x{n}" if n > 1 and not k.startswith("layout") else "") +
                     (f" (up to {big:.0f} px)" if k == "tile jumps" else "") for k, n in kinds.items())


# ---- renderer per crop, from the run's plan without CLAHE
plan = render._tile_plan(cfg)
settings = render._settings(cfg)
settings["clahe"] = {**settings["clahe"], "enabled": False}
fields = elastic.load(cfg)
ts = plan.groupby("z").size().index
stamp = load_slices(cfg).groupby("z")["timestamp"].first()
try:
    font = ImageFont.load_default(size=15)
except TypeError:
    font = ImageFont.load_default()
rows = []
with ThreadPoolExecutor(8) as pool:
    for k, s in enumerate(spots, 1):
        i0, i1 = max(0, pos[s["z0"]] - PAD), min(len(zs) - 1, pos[s["z1"]] + PAD)
        span = zs[i0:i1 + 1]
        sub = plan[plan["z"].isin(span)]
        zc = int(zs[(pos[s["z0"]] + pos[s["z1"]]) // 2])
        here = sub[sub["z"] == zc]
        tiles = sorted(here["tile"], key=lambda x: tuple(map(int, x.split("-"))))
        nr = max(int(x.split("-")[0]) for x in tiles) + 1
        nc = max(int(x.split("-")[1]) for x in tiles) + 1
        renderers = {}
        for _, r in here.iterrows():
            c = transforms.apply(transforms.from_row(r), [[r["width"] / 2, r["height"] / 2]])[0]
            meta = {"downsample": 1, "origin_xy": [float(np.floor(c[0] - CROP / 2)), float(np.floor(c[1] - CROP / 2))],
                    "shape": [1, CROP, CROP], "settings": settings, "destreak": None, "elastic": fields}
            renderers[r["tile"]] = render._Renderer(cfg, meta, sub, pool)
        why = summary(s["why"])
        frames = []
        for z in span:
            grid = Image.new("L", (nc * CROP + (nc - 1) * 4, 26 + nr * CROP + (nr - 1) * 4), 0)
            for tile, rr in renderers.items():
                a, b = map(int, tile.split("-"))
                grid.paste(Image.fromarray(rr.plane([(int(z), 1.0)])), (b * (CROP + 4), 26 + a * (CROP + 4)))
            draw = ImageDraw.Draw(grid)
            flag = " <" if s["z0"] <= z <= s["z1"] else ""
            draw.text((6, 4), f"spot {k}  z {int(z)}  {stamp.loc[int(z)]:%d %b %H:%M:%S}{flag}   {why}", fill=255, font=font)
            frames.append(grid)
        seq = frames + frames[-2:0:-1]
        name = f"spot{k:02d}_z{int(span[0])}-{int(span[-1])}.gif"
        seq[0].save(out / name, save_all=True, append_images=seq[1:], duration=FRAME_MS, loop=0)
        rows.append({"spot": k, "file": name, "z_first": int(span[0]), "z_last": int(span[-1]),
                     "flagged_z": f"{s['z0']}-{s['z1']}", "time": f"{stamp.loc[s['z0']]:%d %b %H:%M}",
                     "tiles": f"{nr}x{nc}", "why": why})
        print(f"spot {k:2d}: z {span[0]}-{span[-1]} ({stamp.loc[s['z0']]:%d %b %H:%M}), {nr}x{nc} tiles: {why}", flush=True)
pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
print(f"{len(rows)} spots -> {out}")
