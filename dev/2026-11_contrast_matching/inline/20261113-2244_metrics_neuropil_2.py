# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-13T22:44:46.675Z
# Ran: local
# Heredoc line: ${EM_PIPE_ENV}/bin/python - <<'EOF'
# Assistant note before it: Yes. 151 / 123 is the appearance match: it's fitted so that 37i's neuropil has 35h's neuropil median grey and fine contrast. The 151 / 145 option is the tissue-distribution match. Now the stability check, from the per-patch numbers already measured:

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import json, numpy as np
rows = json.load(open(os.path.expandvars("${DEV_OUT}/qc/clahe_match/metrics_neuropil.json")))
rng = np.random.default_rng(0)
def patch_vals(name, m, vol=None):
    d = {}
    for r in rows:
        if r["set"] == name and (vol is None or vol in r["vol"]):
            d.setdefault((r["vol"], r["z"] // 64, r["iy"], r["ix"]), []).append(r["plain"][m])
    return np.array([np.mean(v) for v in d.values()])
for m in ("lc8", "lc32", "std"):
    a35, a37 = patch_vals("35h", m), patch_vals("37i", m)
    a = np.median(a35) / np.median(a37)
    boots = [np.median(rng.choice(a35, len(a35))) / np.median(rng.choice(a37, len(a37))) for _ in range(5000)]
    per37 = np.median(a35) / a37
    print(f"{m:5s} (n 35h {len(a35)}, 37i {len(a37)}): factor {a:.3f}, 95% CI {np.percentile(boots, 2.5):.2f}-{np.percentile(boots, 97.5):.2f}; "
          f"per 37i patch p10-p90 {np.percentile(per37, 10):.2f}-{np.percentile(per37, 90):.2f}; "
          f"z1000 slab {np.median(a35) / np.median(patch_vals('37i', m, 'z1000')):.2f}, z9600 slab {np.median(a35) / np.median(patch_vals('37i', m, 'z9600')):.2f}")
