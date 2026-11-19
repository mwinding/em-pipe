# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-17T14:38:06.702Z
# Ran: local
# Heredoc line: ${EM_PIPE_ENV}/bin/python - <<'EOF'
# Assistant note before it: All intensity and overlap tests pass. Re-predicting on the real relations with the final code:

import sys; sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}"))
import numpy as np, pandas as pd
from pipeline import intensity
from pipeline.config import load_config
L = os.path.expandvars("${DEV_OUT}/qc/35h_shading/local_intensity")
rel = pd.read_csv(f"{L}/work/overlaps/relations.csv", dtype={"a": str, "b": str})
corner = (rel.a.str[0] != rel.b.str[0]) & (rel.a.str[2] != rel.b.str[2])
cfg = load_config(os.path.expandvars("${EM_PIPE_ROOT}/configs/${DEV_PREFIX}35h.yaml"), output_dir=L, step_defaults=intensity.DEFAULTS)
print({k: cfg["intensity"][k] for k in ("balance_source", "balance_smooth_slices", "per_slice")})
lv = intensity.compute_levels(cfg).set_index(["z", "tile"])[["lo", "hi"]]
j = rel.join(lv.add_suffix("_a"), on=["z", "a"]).join(lv.add_suffix("_b"), on=["z", "b"])
vb = j.median_b; va = j.gain * vb + j.offset
j["step"] = (vb - j.lo_b) / (j.hi_b - j.lo_b) * 255 - (va - j.lo_a) / (j.hi_a - j.lo_a) * 255
e = j[~corner]
seg = pd.cut(e.z, [0, 828, 3279, 12529, 13771, 15378, 10**9], right=False, labels=range(6)).astype(int)
print("predicted median |step| %.1f greys, %.1f %% of seams > 10, > 30: %.1f %%" % (e.step.abs().median(), (e.step.abs() > 10).mean() * 100, (e.step.abs() > 30).mean() * 100))
print("per segment:", e.step.abs().groupby(seg).median().round(1).to_dict())
