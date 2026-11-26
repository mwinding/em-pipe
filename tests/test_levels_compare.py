"""tools/levels_compare.py: differences between two level tables in greys."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("levels_compare", ROOT / "tools" / "levels_compare.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


def levels(shift=0.0, widen=1.0, tile_shift=None):
    rows = []
    for z in range(20):
        for t in ("0-0", "0-1"):
            lo, hi = 1000.0 + 5 * z, 2020.0 + 5 * z            # width 1020: 4 raw units per grey
            mid, half = (lo + hi) / 2 + shift + (tile_shift or {}).get(t, 0.0), widen * (hi - lo) / 2
            rows.append({"z": z, "tile": t, "lo": mid - half, "hi": mid + half})
    return pd.DataFrame(rows)


def test_identical_levels_differ_by_nothing():
    s = lc.summarise(lc.compare(levels(), levels()), 40, 40)
    assert s["compared"] == 40 and s["d_grey"] == {"median": 0.0, "p5": 0.0, "p95": 0.0, "largest": 0.0}
    assert s["width_ratio"]["median"] == 1.0


def test_shift_and_width_in_greys():
    m = lc.compare(levels(), levels(shift=40.0))             # windows 40 raw units higher: 10 greys darker
    assert np.allclose(m["d_lo"], 10.0) and np.allclose(m["d_hi"], 10.0) and np.allclose(m["d_grey"], -10.0)
    assert np.allclose(m["d_grey_slice"], -10.0) and np.allclose(m["d_grey_tile"], 0.0, atol=1e-9)
    m = lc.compare(levels(), levels(widen=1.1))
    assert np.allclose(m["width_ratio"], 1.1) and np.allclose(m["d_grey"], 0.0, atol=1e-9)
    m = lc.compare(levels(), levels(tile_shift={"0-1": 8.0}))   # one tile 2 greys darker: a tile's own part
    assert np.allclose(m["d_grey_slice"], -1.0) and set(np.round(m["d_grey_tile"], 6)) == {1.0, -1.0}


def test_command_line(tmp_path, capsys):
    levels().to_csv(tmp_path / "a.csv", index=False)
    levels(shift=4.0).iloc[:-2].to_csv(tmp_path / "b.csv", index=False)
    assert lc.main([str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), "--labels", "hand", "auto", "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "38 (z, tile) in both over 19 slices (2 only in hand, 0 only in auto)" in out
    s = json.loads((tmp_path / "o" / "levels_compare.json").read_text())
    assert s["d_grey"]["median"] == pytest.approx(-1.0) and (tmp_path / "o" / "levels_compare.csv").exists()
