# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-09T19:50:38.642Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes ${CLUSTER_LOGIN} 'mkdir -p ${DEV_WORK} && cat > ${DEV_WORK}/contrast_variants.py' 2>/dev/null <<'EOF'
# Assistant note before it: Building the comparison. First, how render produces a single plane, so I can reuse its exact code rather than approximate it:

"""Render one 64 nm plane in several contrast variants with render's own code; side-by-side crops."""
import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import argparse, copy, json, sys
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pandas as pd, tifffile
from pipeline import render
from pipeline.cli import setup

cfg = setup(argparse.Namespace(config=sys.argv[1], raw_dir=None, output_dir=None, verbose=False), render.DEFAULTS)
T = cfg["output_dir"]
OUT_ROOT = __import__('os').environ.get('DEV_OUT', 'dev_out')  # outputs: a directory you choose, never the dataset folder
wdir = os.path.expandvars(f"{T}/work/render/${{DEV_PROJECT}}_M10-D01-D02")
meta = json.load(open(f"{wdir}/render.json"))
plan = pd.read_csv(f"{wdir}/tiles.csv", dtype={"tile": str, "file": str})
K = 150                                   # s0 planes 150, 151 -> 64 nm plane 75 (around midnight, 1-2 Oct)

def variant(clahe, headroom):
    m = copy.deepcopy(meta)
    m["settings"]["clahe"] = {**m["settings"]["clahe"], **clahe}
    p = plan.copy()
    w = p["hi"] - p["lo"]
    p["lo"], p["hi"] = p["lo"] - headroom * w, p["hi"] + headroom * w
    with ThreadPoolExecutor(8) as pool:
        r = render._Renderer(cfg, m, p, pool)
        a = np.mean([r.plane(m["planes"][k]).astype(np.float32) for k in (K, K + 1)], axis=0)
    h, w_ = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    return a[:h, :w_].reshape(h // 2, 2, w_ // 2, 2).mean(axis=(1, 3)).round().astype(np.uint8)

panels = {
    "1 before (old render)": tifffile.imread(os.path.expandvars(f"{T}/${{DEV_PROJECT}}_M10-D01-D02_64nm_before.tif"), key=K // 2),
    "2 current": tifffile.imread(os.path.expandvars(f"{T}/${{DEV_PROJECT}}_M10-D01-D02_64nm.tif"), key=K // 2),
    "3 current, CLAHE off": variant({"enabled": False}, 0.0),
    "4 wider window, CLAHE off": variant({"enabled": False}, 0.15),
    "5 wider window, mild CLAHE": variant({"enabled": True, "clip_limit": 1.0}, 0.15),
}
for name, im in panels.items():
    print(f"{name:30s} shape {im.shape}  std {im.std():5.1f}  p1 {np.percentile(im, 1):5.0f}  p99 {np.percentile(im, 99):5.0f}"
          f"  at 0: {100 * np.mean(im == 0):4.1f}%  at 255: {100 * np.mean(im == 255):4.1f}%")
for label, (cy, cx, half) in {"junction": (1560, 1700, 400), "neuropil": (2325, 978, 300)}.items():
    crops = []
    for name, im in panels.items():
        c = np.ascontiguousarray(im[cy - half:cy + half, cx - half:cx + half])
        cv2.putText(c, name, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 5)
        cv2.putText(c, name, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        crops += [c, np.full((c.shape[0], 10), 255, np.uint8)]
    out = f"{OUT_ROOT}/qc/contrast_compare_{label}.png"
    cv2.imwrite(out, np.hstack(crops[:-1]))
    print("wrote", out)
