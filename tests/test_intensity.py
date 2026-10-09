"""Tests for pipeline.intensity: tile balancing against synthetic gains, smoothing, seams."""

import logging
import re
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import tifffile

import synth
from pipeline import intensity, transforms
from pipeline.preview import PERCENTILES
from synth import save_slices, slice_rows, write_slices
from test_preview import run_preview, two_segments

GAIN = {"0-1": 1.15, "1-0": 0.9}
GAIN3 = {"0-1": 1.2, "1-1": 1.1, "2-0": 0.92, "2-2": 0.85}


@pytest.fixture(scope="module")
def gained(tmp_path_factory):
    """Dataset with per-tile gains and drift over z, preview run, ground-truth stitch/tiles.csv."""
    root = tmp_path_factory.mktemp("intensity")
    truth = synth.make_dataset(root / "raw", tile_gain=GAIN, intensity_drift=0.15, noise=0.001)
    out = root / "out"
    write_slices(truth, out)
    run_preview(synth.write_config(root / "preview.yaml", truth.raw_dir, out, preview={"factor": 4}))
    recs = [({"z": z, "tile": t, "segment": 0}, transforms.translation(*truth.tile_origin[t]))
            for z in range(len(truth.timestamps)) for t in truth.tiles]
    (out / "work" / "stitch").mkdir()
    transforms.to_frame(recs).to_csv(out / "work" / "stitch" / "tiles.csv", index=False)
    return truth, root, out


def run_intensity(root, raw_dir, out, **params):
    # Synthetic tiles are tiny (~200 px), so their median changes with content from slice to slice;
    # per_slice (following acquisition jumps of real 177 Mpx tiles) is tested on its own below.
    params.setdefault("per_slice", False)
    cfg = synth.write_config(root / "intensity.yaml", raw_dir, out, intensity=params)
    assert intensity.main(["--config", str(cfg)]) == 0
    return pd.read_csv(out / "work" / "intensity" / "levels.csv", dtype={"tile": str})


def overlap_diffs(truth, levels, z0=0):
    """Per (z, overlapping tile pair): median |a - b| of the two tiles rendered with their levels.

    The dataset's slices sit at z0 + z; (z, tile) without levels are skipped.
    """
    th, tw = truth.tile_shape
    lv = levels.set_index(["z", "tile"])
    rendered = {}
    for name, zs in truth.files.items():
        tile = re.search(r"_tile(\d+-\d+)", name).group(1)
        stack = tifffile.imread(truth.raw_dir / name).astype(float)
        for i, z in enumerate(zs):
            if (z0 + z, tile) in lv.index:
                lo, hi = lv.loc[(z0 + z, tile), ["lo", "hi"]]
                rendered[z, tile] = np.clip((stack[i] - lo) / (hi - lo) * 255, 0, 255)
    out = []
    for z in range(len(truth.timestamps)):
        for a, b in [(a, b) for i, a in enumerate(truth.tiles) for b in truth.tiles[i + 1:]]:
            (xa, ya), (xb, yb) = truth.tile_origin[a], truth.tile_origin[b]
            x0, x1, y0, y1 = max(xa, xb), min(xa, xb) + tw, max(ya, yb), min(ya, yb) + th
            if x1 <= x0 or y1 <= y0 or (z, a) not in rendered or (z, b) not in rendered:
                continue
            da = rendered[z, a][y0 - ya:y1 - ya, x0 - xa:x1 - xa]
            db = rendered[z, b][y0 - yb:y1 - yb, x0 - xb:x1 - xb]
            out.append(np.median(np.abs(da - db)))
    return np.array(out)


def window_ratios(levels, gains, ref="0-0"):
    """hi - lo of each tile in ``gains`` over that of ``ref``, per z."""
    width = levels.assign(w=levels["hi"] - levels["lo"]).pivot(index="z", columns="tile", values="w")
    return {tile: (width[tile] / width[ref]).dropna() for tile in gains}


def test_balancing_removes_tile_seams(gained):
    truth, root, out = gained
    bal = run_intensity(root, truth.raw_dir, out, smooth_slices=5, balance_every=3)
    assert list(bal.columns) == ["z", "tile", "lo", "hi"]
    assert len(bal) == len(truth.timestamps) * len(truth.tiles)
    assert (out / "qc" / "intensity.png").stat().st_size > 0
    d_bal = overlap_diffs(truth, bal)
    assert d_bal.max() < 3 and d_bal.mean() < 1.2

    # Balanced windows scale with each tile's true gain: synthetic raw = lo + (hi - lo) * gain * content.
    for tile, ratio in window_ratios(bal, GAIN).items():
        np.testing.assert_allclose(ratio, GAIN[tile], rtol=0.03)

    base = run_intensity(root, truth.raw_dir, out, smooth_slices=5, balance_tiles=False)
    d_base = overlap_diffs(truth, base)
    assert d_base.mean() > 3 * d_bal.mean() and d_base.max() > 6


def test_falls_back_without_stitch(gained, tmp_path, caplog):
    truth, root, out = gained
    for step in ("check", "preview"):
        shutil.copytree(out / "work" / step, tmp_path / "work" / step)
    with caplog.at_level(logging.WARNING):
        fallback = run_intensity(tmp_path, truth.raw_dir, tmp_path, smooth_slices=5)
    assert "without tile balancing" in caplog.text
    base = run_intensity(root, truth.raw_dir, out, smooth_slices=5, balance_tiles=False)
    pd.testing.assert_frame_equal(fallback, base)


def test_tile_without_overlap_measurement_keeps_own_levels(gained, tmp_path, caplog):
    truth, root, out = gained
    for step in ("check", "preview", "stitch"):
        shutil.copytree(out / "work" / step, tmp_path / "work" / step)
    tiles = pd.read_csv(out / "work" / "stitch" / "tiles.csv", dtype={"tile": str})
    tiles[tiles["tile"] != "1-1"].to_csv(tmp_path / "work" / "stitch" / "tiles.csv", index=False)
    with caplog.at_level(logging.WARNING):
        lv = run_intensity(tmp_path, truth.raw_dir, tmp_path, smooth_slices=5, balance_every=3)
    assert "left unbalanced" in caplog.text
    base = run_intensity(root, truth.raw_dir, out, smooth_slices=5, balance_tiles=False)
    pd.testing.assert_frame_equal(lv[lv["tile"] == "1-1"], base[base["tile"] == "1-1"])
    np.testing.assert_allclose(window_ratios(lv, GAIN)["0-1"], GAIN["0-1"], rtol=0.03)


def test_no_overlap_anywhere_leaves_every_tile_unbalanced(gained, tmp_path, caplog):
    """Regression: no tile pair overlapping at any sampled slice used to crash (empty measurements)."""
    truth, root, out = gained
    for step in ("check", "preview", "stitch"):
        shutil.copytree(out / "work" / step, tmp_path / "work" / step)
    tiles = pd.read_csv(out / "work" / "stitch" / "tiles.csv", dtype={"tile": str})
    tiles[["tx", "ty"]] *= 10   # tiles far apart
    tiles.to_csv(tmp_path / "work" / "stitch" / "tiles.csv", index=False)
    with caplog.at_level(logging.WARNING):
        lv = run_intensity(tmp_path, truth.raw_dir, tmp_path, smooth_slices=5, balance_every=3)
    assert "left unbalanced" in caplog.text
    base = run_intensity(root, truth.raw_dir, out, smooth_slices=5, balance_tiles=False)
    pd.testing.assert_frame_equal(lv, base)


@pytest.fixture(scope="module")
def segments(tmp_path_factory):
    """3x3 (z 0-7) then 2x2 with larger tiles (z 8-21), per-tile gains, z 11 excluded, tile 1-1
    missing at z 14 and 16; preview run (4-z chunks) and ground-truth stitch/tiles.csv."""
    root = tmp_path_factory.mktemp("segments")
    t3, t2 = two_segments(root / "raw", GAIN3, GAIN, intensity_drift=0.15, noise=0.001)
    out = root / "out"
    excluded = (11, (14, "1-1"), (16, "1-1"))
    save_slices(out, slice_rows(t3) + slice_rows(t2, z0=8, segment_starts=(8,), excluded=excluded))
    run_preview(synth.write_config(root / "preview.yaml", t3.raw_dir, out,
                                   preview={"factor": 4, "chunk_slices": 4}), num_tasks=2)
    recs = [({"z": z0 + z, "tile": tile, "segment": seg}, transforms.translation(*t.tile_origin[tile]))
            for seg, (t, z0) in enumerate([(t3, 0), (t2, 8)]) for z in range(len(t.timestamps)) for tile in t.tiles]
    (out / "work" / "stitch").mkdir()
    transforms.to_frame(recs).to_csv(out / "work" / "stitch" / "tiles.csv", index=False)
    return t3, t2, root, out


def test_balancing_across_segments_with_different_grids(segments):
    t3, t2, root, out = segments
    bal = run_intensity(root, t3.raw_dir, out, smooth_slices=10, balance_every=2)   # gains: median of 5 samples
    keys = {(z, t) for z in range(8) for t in t3.tiles} | {(z, t) for z in range(8, 22) for t in t2.tiles}
    keys -= {(11, t) for t in t2.tiles} | {(14, "1-1"), (16, "1-1")}
    assert set(zip(bal["z"], bal["tile"])) == keys
    base = run_intensity(root, t3.raw_dir, out, smooth_slices=10, balance_tiles=False)
    for truth, z0, gains in ((t3, 0, GAIN3), (t2, 8, GAIN)):
        d_bal, d_base = overlap_diffs(truth, bal, z0), overlap_diffs(truth, base, z0)
        assert d_bal.max() < 3 and d_bal.mean() < 1.2
        assert d_base.mean() > 3 * d_bal.mean()
        segment = bal[(bal["z"] >= z0) & (bal["z"] < z0 + len(truth.timestamps))]
        for tile, ratio in window_ratios(segment, gains).items():
            np.testing.assert_allclose(ratio, gains[tile], rtol=0.03)


def test_tiles_not_linked_by_overlaps_are_not_balanced_together(gained, tmp_path):
    """Stitch places column 1 far right: columns 0 and 1 share no overlap, so only the larger linked
    set (here the first, column 0) is balanced; column 1 keeps its own levels rather than arbitrary ones."""
    truth, root, out = gained
    for step in ("check", "preview", "stitch"):
        shutil.copytree(out / "work" / step, tmp_path / "work" / step)
    tiles = pd.read_csv(out / "work" / "stitch" / "tiles.csv", dtype={"tile": str})
    tiles.loc[tiles["tile"].str.endswith("-1"), "tx"] += 1000
    tiles.to_csv(tmp_path / "work" / "stitch" / "tiles.csv", index=False)
    lv = run_intensity(tmp_path, truth.raw_dir, tmp_path, smooth_slices=5, balance_every=3)
    base = run_intensity(root, truth.raw_dir, out, smooth_slices=5, balance_tiles=False)
    right = lv["tile"].str.endswith("-1")
    pd.testing.assert_frame_equal(lv[right], base[right])
    np.testing.assert_allclose(window_ratios(lv, {"1-0": 0.9})["1-0"], GAIN["1-0"], rtol=0.03)


def write_stats_only(out, seam_z, segment_z, excluded, outlier_z):
    """slices.csv + preview/stats.csv for 2 tiles, no raw data; returns expected base lo per (z, tile).

    lo is piecewise constant: 1000 before the seam, 3000 from the seam, 5000 from the segment
    change (tile 0-1 is 100 brighter); hi = lo + 10000; p1 = lo + 50. One outlier slice on tile 0-0.
    """
    t0 = datetime(2026, 9, 24, 12)
    zs = [z for z in range(40) if z not in excluded]
    slices, stats, expected = [], [], {}
    for z in range(40):
        for c, tile in enumerate(["0-0", "0-1"]):
            lo = (1000 if z < seam_z else 3000 if z < segment_z else 5000) + 100 * c
            expected[z, tile] = lo
            if z == outlier_z and tile == "0-0":
                lo = 20000
            slices.append({"z": z, "timestamp": (t0 + timedelta(seconds=140 * z)).isoformat(), "tile": tile,
                           "tile_row": 0, "tile_col": c, "file": f"M09_D24_tile{tile}.tif", "index": z,
                           "height": 100, "width": 100, "segment": int(z >= segment_z), "seam": z == seam_z,
                           "excluded": z in excluded, "exclude_reason": "", "label": ""})
            stats.append({"z": z, "tile": tile, **{k: lo + p for k, p in PERCENTILES.items()},
                          "p0_5": lo, "p1": lo + 50, "p99_5": lo + 10000})
    (out / "work" / "check").mkdir(parents=True)
    (out / "work" / "preview").mkdir()
    pd.DataFrame(slices).to_csv(out / "work" / "check" / "slices.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "work" / "preview" / "stats.csv", index=False)
    return zs, expected


def test_smoothing_outlier_seams_and_segments(tmp_path):
    out = tmp_path / "out"
    # Selected z skip 5 and 6; the seam at 32 starts a 4-slice run, the segment change at 36 another.
    zs, expected = write_stats_only(out, seam_z=32, segment_z=36, excluded=(5, 6), outlier_z=12)

    def levels(**params):
        lv = run_intensity(tmp_path, tmp_path, out, smooth_slices=11, balance_tiles=False, **params)
        return lv.set_index(["z", "tile"])

    lv = levels(break_at_seams=True)
    assert sorted(lv.index.get_level_values("z").unique()) == zs
    want = pd.Series({k: v for k, v in expected.items() if k[0] in zs}, dtype=float)
    np.testing.assert_allclose(lv["lo"], want.loc[lv.index])          # outlier at z 12 removed
    np.testing.assert_allclose(lv["hi"] - lv["lo"], 10000)

    # Without the seam break the window at z 32 is mostly pre-seam slices; the segment change
    # still restarts smoothing.
    lv = levels(break_at_seams=False)
    assert lv.loc[(32, "0-0"), "lo"] == 1000
    np.testing.assert_allclose(lv.loc[[(z, "0-0") for z in range(36, 40)], "lo"], 5000)

    lv = levels(lo_percentile=1, hi_percentile=99.5)
    assert lv.loc[(20, "0-1"), "lo"] == 1150


@pytest.mark.parametrize("bad", [{"lo_percentile": 2}, {"lo_percentile": 99.5, "hi_percentile": 1},
                                 {"balance_every": 0}, {"smooth_slices": 0}])
def test_invalid_percentiles_rejected(tmp_path, bad):
    cfg = synth.write_config(tmp_path / "c.yaml", tmp_path, tmp_path / "out", intensity=bad)
    with pytest.raises(ValueError):
        intensity.main(["--config", str(cfg)])


def test_solve_recovers_gains_and_offsets():
    true = {"0-0": (1.1, -300.0), "0-1": (0.95, 100.0), "1-0": (0.95, 200.0)}
    rels = []
    for a, b in [("0-0", "0-1"), ("0-0", "1-0"), ("0-1", "1-0")]:
        (ga, oa), (gb, ob) = true[a], true[b]
        rels.append((a, b, gb / ga, (ob - oa) / ga, 400))   # v_a = (g_b v_b + o_b - o_a) / g_a
    got = intensity.solve(rels)
    for tile, (g, o) in true.items():
        assert got[tile][0] == pytest.approx(g, abs=1e-6) and got[tile][1] == pytest.approx(o, abs=1e-3)


def test_solve_uses_only_the_largest_connected_set_of_tiles():
    """Tiles with no overlap path between them can't be put on one scale: keep the largest set."""
    rels = [("0-0", "0-1", 1.0, 0.0, 400), ("1-0", "1-1", 1.2, 5.0, 400), ("1-1", "1-2", 0.9, 0.0, 400)]
    assert intensity.connected(rels) == rels[1:]
    assert set(intensity.solve(intensity.connected(rels))) == {"1-0", "1-1", "1-2"}
    assert intensity.connected(rels[:2] + [("0-1", "1-1", 1.0, 0.0, 400)]) == rels[:2] + [("0-1", "1-1", 1.0, 0.0, 400)]


def test_per_slice_follows_tile_brightness_jumps(tmp_path):
    """A tile that is brighter on single slices (as raw P667 tiles are, by 1-2 % of the range) gets
    its window shifted on exactly those slices, so it doesn't flicker; other tiles are untouched."""
    out = tmp_path / "out"
    zs, expected = write_stats_only(out, seam_z=100, segment_z=100, excluded=(), outlier_z=-1)
    stats = pd.read_csv(out / "work" / "preview" / "stats.csv", dtype={"tile": str})
    jumps = {7: 300.0, 8: -200.0, 21: 150.0}
    for z, d in jumps.items():
        cols = [c for c in stats.columns if c.startswith("p") or c in ("mean", "min", "max")]
        stats.loc[(stats.z == z) & (stats.tile == "0-1"), cols] += d
    stats.to_csv(out / "work" / "preview" / "stats.csv", index=False)
    lv = run_intensity(tmp_path, tmp_path, out, smooth_slices=11, balance_tiles=False,
                       per_slice=True).set_index(["z", "tile"])
    for z in zs:
        assert lv.loc[(z, "0-1"), "lo"] == pytest.approx(expected[z, "0-1"] + jumps.get(z, 0.0))
        assert lv.loc[(z, "0-0"), "lo"] == pytest.approx(expected[z, "0-0"])
    off = run_intensity(tmp_path, tmp_path, out, smooth_slices=11, balance_tiles=False, per_slice=False)
    assert off.set_index(["z", "tile"]).loc[(7, "0-1"), "lo"] == pytest.approx(expected[7, "0-1"])
