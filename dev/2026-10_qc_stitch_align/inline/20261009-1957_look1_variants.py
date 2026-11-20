# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T19:57:16.380Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes ${CLUSTER_LOGIN} 'cat > ${DEV_WORK}/look1_variants.py' 2>/dev/null <<'EOF'
# Assistant note before it: Now on the cluster: re-run preview with the original statistics, then intensity.

"""Reproduce the 'before' look with sharp seams: levels with/without per_slice; compare to before."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import argparse, copy, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd, tifffile, yaml
from scipy.ndimage import median_filter
from pipeline import render
from pipeline.cli import setup

cfg_path = sys.argv[1]
cfg = setup(argparse.Namespace(config=cfg_path, raw_dir=None, output_dir=None, verbose=False), render.DEFAULTS)
T = Path(cfg["output_dir"]); S = Path.home() / os.path.expandvars("dev-work")
lv_on = pd.read_csv(T / "work/intensity/levels.csv", dtype={"tile": str}); lv_on.to_csv(S / "levels_ps_on.csv", index=False)
c = yaml.safe_load(open(cfg_path)); c.setdefault("intensity", {})["per_slice"] = False
(S / "cfg_ps_off.yaml").write_text(yaml.safe_dump(c))
subprocess.run([sys.executable, "-m", "pipeline.intensity", "--config", str(S / "cfg_ps_off.yaml")], check=True)
lv_off = pd.read_csv(T / "work/intensity/levels.csv", dtype={"tile": str}); lv_off.to_csv(S / "levels_ps_off.csv", index=False)

# predicted flicker: rendered brightness of each tile's median, (p50 - lo) / (hi - lo) * 255, averaged over the
# 4 slices of each 32 nm plane; tile-specific part of the plane-to-plane change (CLAHE not modelled)
st = pd.read_csv(T / "work/preview/stats.csv", dtype={"tile": str})[["z", "tile", "p50"]]
for name, lv in (("per_slice on", lv_on), ("per_slice off", lv_off)):
    d = st.merge(lv, on=["z", "tile"])
    d["v"] = (d.p50 - d.lo) / (d.hi - d.lo) * 255
    p = d.pivot(index="z", columns="tile", values="v")
    p = p.groupby(np.arange(len(p)) // 4).mean()
    hp = p - p.apply(lambda col: median_filter(col.values, 9, mode="nearest"))
    own = hp.sub(hp.mean(axis=1), axis=0)
    print(f"predicted flicker ({name}): std", own.std().round(2).to_dict(), " max", own.abs().max().round(1).to_dict())

wdir = T / os.path.expandvars("work/render/${DEV_PROJECT}_M10-D01-D02")
meta = json.load(open(wdir / "render.json"))
plan = pd.read_csv(wdir / "tiles.csv", dtype={"tile": str, "file": str})
K = 150
def variant(lv):
    p = plan.drop(columns=["lo", "hi"]).merge(lv[["z", "tile", "lo", "hi"]], on=["z", "tile"], how="left")
    m = copy.deepcopy(meta)
    m["settings"]["clahe"] = {**m["settings"]["clahe"], "enabled": True, "clip_limit": 2.0}
    m["settings"]["blend"] = "seam"
    with ThreadPoolExecutor(8) as pool:
        r = render._Renderer(cfg, m, p, pool)
        a = np.mean([r.plane(m["planes"][k]).astype(np.float32) for k in (K, K + 1)], axis=0)
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).round().astype(np.uint8)

panels = {
    "1 before (old render)": tifffile.imread(T / os.path.expandvars("${DEV_PROJECT}_M10-D01-D02_64nm_before.tif"), key=K // 2),
    "A original stats, per-slice on": variant(lv_on),
    "B original stats, per-slice off": variant(lv_off),
    "2 current (harsh)": tifffile.imread(T / os.path.expandvars("${DEV_PROJECT}_M10-D01-D02_64nm.tif"), key=K // 2),
}
for name, im in panels.items():
    print(f"{name:34s} std {im.std():5.1f}  p1 {np.percentile(im, 1):4.0f}  p50 {np.percentile(im, 50):4.0f}  p99 {np.percentile(im, 99):4.0f}")
for label, (cy, cx, half) in {"junction": (1560, 1700, 400), "neuropil": (2325, 978, 300)}.items():
    crops = []
    for name, im in panels.items():
        cc = np.ascontiguousarray(im[cy - half:cy + half, cx - half:cx + half])
        cv2.putText(cc, name, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 5)
        cv2.putText(cc, name, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        crops += [cc, np.full((cc.shape[0], 10), 255, np.uint8)]
    cv2.imwrite(str(T / "qc" / f"contrast_compare_{label}.png"), np.hstack(crops[:-1]))
print("done")
