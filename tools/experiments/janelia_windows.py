"""Per-(z, tile) display windows without tile balance, for the Janelia-style experiment.

    python tools/experiments/janelia_windows.py CONFIG OUT_CSV

intensity.compute_levels with cfg["intensity"] overridden in memory: balance_tiles false and
per_slice false (columns lo, hi), and the same with per_slice "slice" (lo_s, hi_s: each slice's
median brightness jump followed, as the current pipeline does). Does not touch work/intensity/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import intensity, shading  # noqa: E402
from pipeline.config import deep_merge, load_config  # noqa: E402


def main():
    cfg = load_config(sys.argv[1], step_defaults=deep_merge(intensity.DEFAULTS, shading.DEFAULTS))
    out = Path(sys.argv[2])
    cfg["intensity"]["balance_tiles"] = False
    cfg["intensity"]["per_slice"] = False
    df = intensity.compute_levels(cfg)[["z", "tile", "segment", "lo", "hi"]]
    cfg["intensity"]["per_slice"] = "slice"
    js = intensity.compute_levels(cfg)[["z", "tile", "lo", "hi"]].rename(columns={"lo": "lo_s", "hi": "hi_s"})
    df = df.merge(js, on=["z", "tile"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.2f")
    print(len(df), "rows ->", out)


if __name__ == "__main__":
    main()
