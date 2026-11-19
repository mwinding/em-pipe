# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-11-13T22:44:35.420Z
# Ran: local
# Heredoc line: ${EM_PIPE_ENV}/bin/python - <<'EOF'
# Assistant note before it: Yes. 151 / 123 is the appearance match: it's fitted so that 37i's neuropil has 35h's neuropil median grey and fine contrast. The 151 / 145 option is the tissue-distribution match. Now the stability check, from the per-patch numbers already measured:

import os  # placeholders such as ${DEV_OUT} resolve from the environment (paths.example.sh)
import json, numpy as np
rows = json.load(open(os.path.expandvars("${DEV_OUT}/qc/clahe_match/metrics_neuropil.json")))
rng = np.random.default_rng(0)
def patch_vals(name, m):
    """one value per patch (cell): mean of its two planes"""
    d = {}
    for r in rows:
        if r["set"] == name:
            d.setdefault((r["vol"], r["iy"], r["ix"]), []).append(r["plain"][m])
    keys = sorted(d)
    return np.array([np.mean(d[k]) for k in keys]), keys
for m in ("lc8", "lc32", "std"):
    a35, k35 = patch_vals("35h", m); a37, k37 = patch_vals("37i", m)
    a = np.median(a35) / np.median(a37)
    boots = [np.median(rng.choice(a35, len(a35))) / np.median(rng.choice(a37, len(a37))) for _ in range(5000)]
    per37 = np.median(a35) / a37            # factor each 37i patch would need
    per35 = a35 / np.median(a37)            # factor implied by each 35h patch
    print(f"{m:5s}: factor {a:.3f}, 95% bootstrap {np.percentile(boots, 2.5):.3f}-{np.percentile(boots, 97.5):.3f} | "
          f"per 37i patch p10-p90 {np.percentile(per37, 10):.2f}-{np.percentile(per37, 90):.2f} (range {per37.min():.2f}-{per37.max():.2f}) | "
          f"per 35h patch p10-p90 {np.percentile(per35, 10):.2f}-{np.percentile(per35, 90):.2f}")
    if m == "lc8":
        for name, arr, keys in (("35h", a35, k35), ("37i", a37, k37)):
            byv = {}
            for v, k in zip(arr, keys):
                byv.setdefault(k[0].replace(".ome.zarr", ""), []).append(v)
            print("   ", name, "lc8 by volume:", {k: f"{np.median(v):.1f} (n {len(v)})" for k, v in byv.items()})
        # 35h by depth: z differs within the same volume name -> use z from rows
        byz = {}
        for r in rows:
            if r["set"] == "35h":
                byz.setdefault(r["z"] // 1000, []).append(r["plain"]["lc8"])
        print("    35h lc8 by depth (z//1000):", {k: f"{np.median(v):.1f} (n {len(v)})" for k, v in sorted(byz.items())})
