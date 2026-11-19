# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-14T20:37:40.802Z
# Ran: remote, via ssh
# Heredoc line: ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=QUIET ${CLUSTER_LOGIN} 'source ${EM_PIPE_ROOT}/slurm/env.sh >/dev/null 2>&1; cd ${DEV_LAB_AREA}
# Assistant note before it: Before guessing, I'll look at what the alignment and stitching actually did at the 3×3 → 2×2 change. This only reads the run's output files.

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import numpy as np
from pipeline.config import load_config
from pipeline import align as al
cfg = load_config(os.path.expandvars("configs/${DEV_PROJECT}_test_3x3.yaml"), step_defaults=al.DEFAULTS)
df = al.load_slices(cfg); zs = np.array(al.z_values(df)); a = cfg["align"]
m = al._load_matches(cfg, df, zs, a)
za, zb = zs[m["pairs"][m["pid"]][:, 0]], zs[m["pairs"][m["pid"]][:, 1]]
x = (za <= 3278) & (zb >= 3279)
pa, pb = m["pa"][x], m["pb"][x]
print("cross points", x.sum(), "pairs", np.unique(m["pid"][x]).size)
d = pa - pb
print("shift median", np.median(d, 0), "spread (IQR)", np.percentile(d, 75, 0) - np.percentile(d, 25, 0))
def fit(kind):
    if kind == "translation":
        t = np.median(d, 0); r = pa - (pb + t); return r, t
    X = np.column_stack([pb, np.ones(len(pb))])
    if kind == "affine":
        A, *_ = np.linalg.lstsq(X, pa, rcond=None); r = pa - X @ A; return r, A.T
    # similarity: [a -b; b a]
    M = np.zeros((2*len(pb), 4)); M[0::2] = np.column_stack([pb[:,0], -pb[:,1], np.ones(len(pb)), np.zeros(len(pb))])
    M[1::2] = np.column_stack([pb[:,1], pb[:,0], np.zeros(len(pb)), np.ones(len(pb))])
    p, *_ = np.linalg.lstsq(M, pa.ravel(), rcond=None); r = (pa.ravel() - M @ p).reshape(-1, 2)
    return r, dict(scale=float(np.hypot(p[0], p[1])), rot_deg=float(np.degrees(np.arctan2(p[1], p[0]))), t=p[2:])
for k in ("translation", "similarity", "affine"):
    r, par = fit(k); n = np.linalg.norm(r, axis=1)
    print(k, "residual median %.2f  p90 %.2f  max %.1f px" % (np.median(n), np.percentile(n, 90), n.max()), par)
# robust affine: refit on points within 10 px
r, A = fit("affine"); ok = np.linalg.norm(r, axis=1) < 10
X = np.column_stack([pb[ok], np.ones(ok.sum())]); A2, *_ = np.linalg.lstsq(X, pa[ok], rcond=None)
n2 = np.linalg.norm(pa[ok] - X @ A2, axis=1)
print("affine refit on", ok.sum(), "inliers: median %.2f p90 %.2f px" % (np.median(n2), np.percentile(n2, 90)), A2.T.round(5))
