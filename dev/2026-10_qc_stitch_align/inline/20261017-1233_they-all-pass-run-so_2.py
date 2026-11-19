# Extracted from a session transcript (inline script, not saved as a file at the time).
# Time (UTC): 2026-10-17T12:33:29.734Z
# Ran: local
# Heredoc line: ${EM_PIPE_ENV}/bin/python -m pytest -q --no-header -p no:cacheprovider tests/test_overlaps.py -s 2>&1 | grep -E "median seam|passed|failed|^E  " ; ${EM_PIPE_ENV}/bin/python - <<'
# Assistant note before it: They all pass on the first run, so I'll check the margins rather than just the pass, and tidy one clumsy test.

import sys; sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}")); sys.path.insert(0, os.path.expandvars("${EM_PIPE_ROOT}/tests"))
import numpy as np, tempfile, pandas as pd
from pathlib import Path
import synth
from synth import write_slices
from pipeline import overlaps, transforms
root = Path(tempfile.mkdtemp())
truth = synth.make_dataset(root / "raw", n_slices=6, overlap=(40, 48), tile_jitter=0.06, noise=0.002, seed=11)
out = root / "out"; write_slices(truth, out)
recs = [({"z": z, "tile": t, "segment": 0}, transforms.translation(*truth.tile_origin[t])) for z in range(6) for t in truth.tiles]
(out / "work" / "stitch").mkdir(parents=True); transforms.to_frame(recs).to_csv(out / "work" / "stitch" / "tiles.csv", index=False)
cfg = synth.write_config(root / "c.yaml", truth.raw_dir, out, overlaps={"enabled": True, "factor": 2, "trim_px": 2, "smooth_px": 0.5, "min_pixels": 50})
overlaps.main(["run", "--config", str(cfg)]); overlaps.main(["merge", "--config", str(cfg)])
rel = pd.read_csv(out / "work" / "overlaps" / "relations.csv", dtype={"a": str, "b": str})
rel["want"] = [truth.jitter[r.z][r.a] / truth.jitter[r.z][r.b] for r in rel.itertuples()]
print(rel[["z", "a", "b", "gain", "want", "n", "contrast"]].round(4).head(12).to_string(index=False))
