"""tools/tile_jumps.py on hand-made stitch/align outputs: finds an injected jump and draws the plot."""
import importlib.util
import json
from pathlib import Path

import pandas as pd

import synth

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tile_jumps", ROOT / "tools" / "tile_jumps.py")
tile_jumps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tile_jumps)


def test_finds_injected_jump(tmp_path, capsys):
    truth = synth.make_dataset(tmp_path / "raw", n_slices=6, tile_shape=(64, 72))
    out = tmp_path / "out"
    synth.write_slices(truth, out)
    cfg = synth.write_config(tmp_path / "c.yaml", truth.raw_dir, out)
    st, al = out / "work" / "stitch", out / "work" / "align"
    (st / "samples").mkdir(parents=True)
    al.mkdir(parents=True)
    rows = []
    for z in range(6):
        for i, tile in enumerate(truth.tiles):
            jump = 20.0 if (tile == truth.tiles[-1] and z >= 3) else 0.0
            rows.append({"z": z, "tile": tile, "segment": 0, "tx": 60.0 * i + jump, "ty": 0.0})
    pd.DataFrame(rows).to_csv(st / "tiles.csv", index=False)
    (st / "layout.json").write_text(json.dumps({"segments": {"0": {"z_first": 0, "z_last": 5, "grid_shape": [2, 2],
                                                                     "tile_shape": [64, 72], "failed_samples": []}}}))
    (st / "samples" / "z000000.json").write_text("{}")
    pd.DataFrame({"z_a": range(5), "z_b": range(1, 6), "rms_px": [0.5, 0.4, 30.0, 0.6, 0.5],
                  "rejected": [False, False, True, False, False]}).to_csv(al / "residuals.csv", index=False)
    tile_jumps.main(["--config", str(cfg)])
    text = capsys.readouterr().out
    assert "jumps >5 px: 1" in text and "largest 20 px" in text
    assert "1 rejected" in text
    assert (out / "qc" / "tile_jumps.png").exists()
