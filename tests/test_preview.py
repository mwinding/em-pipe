"""Tests for pipeline.preview: stats and thumbnails against numpy, chunking, merge outputs."""

import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import tifffile

import synth
from pipeline import features, preview
from synth import save_slices, slice_rows, write_slices

FACTOR = 4


def two_segments(raw_dir, gains3=None, gains2=None, **kw):
    """A 3x3 grid of 128x144 tiles (z 0-7, 2026-09-20) then a 2x2 grid of 192x224 tiles (z 8-21,
    2026-09-24) in one raw folder. Returns (truth3, truth2); the 2x2 slices sit at z0 = 8."""
    t3 = synth.make_dataset(raw_dir, n_slices=8, grid=(3, 3), tile_shape=(128, 144), overlap=(24, 28),
                            start=datetime(2026, 9, 20, 10), tile_gain=gains3, seed=3, **kw)
    t2 = synth.make_dataset(raw_dir, n_slices=14, start=datetime(2026, 9, 24, 10), tile_gain=gains2, seed=4, **kw)
    return t3, t2


def run_preview(config, num_tasks=1):
    for i in range(num_tasks):
        assert preview.main(["run", "--config", str(config), "--task-id", str(i), "--num-tasks", str(num_tasks)]) == 0
    assert preview.main(["merge", "--config", str(config)]) == 0


def read_outputs(out):
    stats = pd.read_csv(out / "work" / "preview" / "stats.csv", dtype={"tile": str})
    index = pd.read_csv(out / "work" / "preview" / "thumbs" / "index.csv", dtype={"tile": str})
    thumbs = {(r.z, r.tile): np.load(out / "work" / "preview" / "thumbs" / r.npy)[r.i] for r in index.itertuples()}
    return stats, thumbs


def true_small(truth, local_z, tile):
    """Block-mean downsample of one raw tile slice, rounded to uint16."""
    name = next(n for n, zs in truth.files.items() if re.search(rf"_tile{tile}[._]", n) and local_z in zs)
    img = tifffile.imread(truth.raw_dir / name)[truth.files[name].index(local_z)].astype(float)
    h, w = img.shape
    return np.rint(img.reshape(h // FACTOR, FACTOR, w // FACTOR, FACTOR).mean(axis=(1, 3))).astype(np.uint16)


def assert_thumb(thumb, small):
    """Thumbnail equals to_uint8 of the true downsample at its p0.5/p99.5 (to 1 grey level, 99 % of pixels)."""
    lo, hi = np.percentile(small, [0.5, 99.5])
    assert thumb.shape == small.shape
    assert np.mean(np.abs(thumb.astype(int) - features.to_uint8(small, lo, hi)) <= 1) > 0.99


def test_stats_and_thumbs_match_numpy(synth_2x2, make_config, tmp_path):
    t = synth_2x2
    out = tmp_path / "out"
    write_slices(t, out)
    run_preview(make_config(t, preview={"factor": FACTOR, "edge_margin_px": 0, "chunk_slices": 5}))
    stats, thumbs = read_outputs(out)
    assert list(stats.columns) == ["z", "timestamp", "tile", *preview.STATS]
    assert len(stats) == len(t.timestamps) * len(t.tiles) == len(thumbs)
    assert stats.equals(stats.sort_values(["z", "tile"]).reset_index(drop=True))
    assert stats["timestamp"].tolist() == [t.timestamps[z].isoformat() for z in stats["z"]]

    th, tw = t.tile_shape
    stats = stats.set_index(["z", "tile"])
    for name, zs in t.files.items():
        tile = re.search(r"_tile(\d+-\d+)", name).group(1)
        stack = tifffile.imread(t.raw_dir / name)
        for i, z in enumerate(zs):
            small = np.rint(stack[i].astype(float).reshape(th // FACTOR, FACTOR, tw // FACTOR, FACTOR)
                            .mean(axis=(1, 3))).astype(np.uint16)
            row = stats.loc[(z, tile)]
            pct = np.percentile(small, list(preview.PERCENTILES.values()))
            np.testing.assert_allclose(row[list(preview.PERCENTILES)].to_numpy(float), pct, atol=1)
            assert row["mean"] == pytest.approx(small.mean(), abs=0.05)
            assert row["std"] == pytest.approx(small.std(), abs=0.05)
            assert abs(row["min"] - small.min()) <= 1 and abs(row["max"] - small.max()) <= 1
            assert row["frac_zero"] == 0 and row["frac_saturated"] == 0
            thumb = thumbs[(z, tile)]
            assert thumb.dtype == np.uint8 and thumb.shape == (th // FACTOR, tw // FACTOR)
            assert_thumb(thumb, small)
            # The inverse used by intensity recovers values inside the window to within one bin.
            back = preview.thumb_values(thumb, row["p0_5"], row["p99_5"])
            inside = (small > pct[0]) & (small < pct[-1])
            assert np.percentile(np.abs(back - small)[inside], 99) <= (pct[-1] - pct[0]) / 255 + 1

    s = preview.slice_stats(np.array([[0, 65535], [100, 200]], np.uint16))
    assert (s["frac_zero"], s["frac_saturated"], s["min"], s["max"]) == (0.25, 0.25, 0, 65535)


def test_chunks_follow_segments_and_selection_and_match_single_task(synth_2x2, tmp_path):
    t = synth_2x2
    sections = {"selection": {"z_end": 21}}
    multi, single = tmp_path / "multi", tmp_path / "single"
    # z 3 excluded on every tile, tile 1-1 alone at z 5 and 6.
    excluded = (3, (5, "1-1"), (6, "1-1"))
    for out in (multi, single):
        write_slices(t, out, segment_starts=(10,), excluded=excluded)
    cfg_multi = synth.write_config(tmp_path / "multi.yaml", t.raw_dir, multi,
                                   preview={"factor": FACTOR, "edge_margin_px": 0, "chunk_slices": 4}, **sections)
    cfg_single = synth.write_config(tmp_path / "single.yaml", t.raw_dir, single,
                                    preview={"factor": FACTOR, "edge_margin_px": 0, "chunk_slices": 100}, **sections)
    run_preview(cfg_multi, num_tasks=3)
    run_preview(cfg_single)

    # Chunks are blocks of 4 global z (0-3, 4-7, ...) split at the segment change at 10.
    names = sorted(p.name for p in (multi / "work" / "preview" / "stats").glob("chunk_*.csv"))
    assert names == [f"chunk_{a:06d}-{b:06d}.csv" for a, b in
                     [(0, 3), (4, 8), (8, 10), (10, 12), (12, 16), (16, 20), (20, 22)]]
    assert np.load(multi / "work" / "preview" / "thumbs" / "z000004-000008_tile1-0.npy").shape == (4, 48, 56)
    assert np.load(multi / "work" / "preview" / "thumbs" / "z000004-000008_tile1-1.npy").shape == (2, 48, 56)
    assert sorted(p.name for p in (single / "work" / "preview" / "stats").glob("chunk_*.csv")) == \
        ["chunk_000000-000010.csv", "chunk_000010-000022.csv"]

    stats_m, thumbs_m = read_outputs(multi)
    stats_s, thumbs_s = read_outputs(single)
    assert sorted(stats_m["z"].unique()) == [z for z in range(22) if z != 3]
    assert len(stats_m) == 21 * 4 - 2
    pd.testing.assert_frame_equal(stats_m, stats_s)
    assert thumbs_m.keys() == thumbs_s.keys()
    assert all(np.array_equal(thumbs_m[k], thumbs_s[k]) for k in thumbs_m)
    # A tile missing at some z must not shift which thumbnail belongs to which z.
    for z in (4, 7, 8, 9):
        for tile in ("1-1", "0-0"):
            assert_thumb(thumbs_m[(z, tile)], true_small(t, z, tile))

    assert sorted(p.name for p in (multi / "qc").glob("sheet_*.png")) == \
        ["sheet_2026-09-24.png", "sheet_2026-09-25.png"]
    assert (multi / "qc" / "preview_stats.png").stat().st_size > 0


def test_run_skips_existing_and_merge_needs_all_chunks(synth_2x2, make_config, tmp_path):
    out = tmp_path / "out"
    write_slices(synth_2x2, out)
    cfg = str(make_config(synth_2x2, preview={"factor": FACTOR, "edge_margin_px": 0, "chunk_slices": 10}))
    run_preview(cfg)
    chunk = out / "work" / "preview" / "stats" / "chunk_000000-000010.csv"
    before = chunk.stat().st_mtime_ns
    preview.main(["run", "--config", cfg])
    assert chunk.stat().st_mtime_ns == before
    preview.main(["run", "--config", cfg, "--overwrite"])
    assert chunk.stat().st_mtime_ns > before
    chunk.unlink()
    with pytest.raises(RuntimeError, match="1 of 3 preview chunks missing"):
        preview.main(["merge", "--config", cfg])


def test_chunks_stay_put_and_late_tiles_are_filled_in(synth_2x2, tmp_path):
    """A new selection start or excluded slice recomputes only the chunks holding those z;
    a chunk lacking tile slices that are selected now (a late file) is recomputed without --overwrite."""
    t = synth_2x2
    out = tmp_path / "out"
    stats_dir = out / "work" / "preview" / "stats"

    def config(**selection):
        return str(synth.write_config(tmp_path / "c.yaml", t.raw_dir, out, selection=selection,
                                      preview={"factor": FACTOR, "edge_margin_px": 0, "chunk_slices": 5}))

    def mtimes():
        return {p.name: p.stat().st_mtime_ns for p in stats_dir.glob("chunk_*.csv")}

    # Tile 1-1 arrives late for z 16-18: the first run doesn't see it.
    write_slices(t, out, excluded=[(z, "1-1") for z in (16, 17, 18)])
    cfg = config()
    run_preview(cfg)
    full, first = read_outputs(out)
    assert (16, "1-1") not in first
    write_slices(t, out)
    with pytest.raises(RuntimeError, match="1 of 5 preview chunks missing or out of date"):
        preview.main(["merge", "--config", cfg])
    before = mtimes()
    run_preview(cfg)
    after = mtimes()
    assert [n for n in after if after[n] != before[n]] == ["chunk_000015-000020.csv"]
    full, thumbs = read_outputs(out)
    assert len(full) == 24 * 4
    assert_thumb(thumbs[(17, "1-1")], true_small(t, 17, "1-1"))
    assert_thumb(thumbs[(17, "0-1")], true_small(t, 17, "0-1"))

    # Start the selection at z 2 and exclude z 12: no existing chunk is recomputed except 2-5.
    write_slices(t, out, excluded=(12,))
    cfg = config(z_start=2)
    run_preview(cfg)
    now = mtimes()
    assert {n for n in now if now[n] != after.get(n)} == {"chunk_000002-000005.csv"}
    stats, _ = read_outputs(out)
    want = full[(full["z"] >= 2) & (full["z"] != 12)].reset_index(drop=True)
    pd.testing.assert_frame_equal(stats, want)


def test_segments_with_different_tile_shapes(tmp_path):
    """3x3 -> 2x2 with different tile sizes: chunks split at the change, thumbnails keep each shape."""
    t3, t2 = two_segments(tmp_path / "raw")
    out = tmp_path / "out"
    save_slices(out, slice_rows(t3) + slice_rows(t2, z0=8, segment_starts=(8,)))
    run_preview(synth.write_config(tmp_path / "c.yaml", t3.raw_dir, out,
                                   preview={"factor": FACTOR, "edge_margin_px": 0, "chunk_slices": 5}), num_tasks=2)
    thumbs_dir = out / "work" / "preview" / "thumbs"
    assert np.load(thumbs_dir / "z000005-000008_tile2-2.npy").shape == (3, 32, 36)
    assert np.load(thumbs_dir / "z000008-000010_tile0-0.npy").shape == (2, 48, 56)
    stats, thumbs = read_outputs(out)
    assert len(stats) == 8 * 9 + 14 * 4
    for (truth, z0) in ((t3, 0), (t2, 8)):
        for local in (0, 6):
            for tile in ("0-0", "1-1"):
                assert_thumb(thumbs[(z0 + local, tile)], true_small(truth, local, tile))
    assert sorted(p.name for p in (out / "qc").glob("sheet_*.png")) == \
        ["sheet_2026-09-20.png", "sheet_2026-09-24.png"]


def test_stats_ignore_dark_edge_bands():
    """A dark band at a tile edge (beam past the sample) doesn't set p0.5; small tiles keep 3/4."""
    from pipeline.preview import slice_stats
    small = np.full((100, 120), 30000, np.uint16)
    small += np.random.default_rng(0).integers(0, 2000, small.shape, dtype=np.uint16)
    small[:, :8] = 100                                   # dark band, 8 px wide, at the left edge
    assert slice_stats(small)["p0_5"] < 1000
    st = slice_stats(small, margin=10)
    assert st["p0_5"] > 29000 and st["min"] >= 30000
    assert slice_stats(small, margin=1000)["p0_5"] > 29000   # capped at 1/8: 12 px in x
