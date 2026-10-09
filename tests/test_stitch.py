"""Tests for pipeline.stitch against synthetic ground truth (tile origins in the montage)."""

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import tifffile

import synth
from pipeline import stitch, transforms

# Synthetic tiles are pure translations: the translation model makes exact identity checks possible.
OPTS = {"sample_every": 3, "coarse_factor": 2, "fine_factor": 1, "fine_margin_px": 16, "model": "translation"}


def write_slices(out_dir, parts, relabel=None, exclude=()):
    """Write check/slices.csv (design.md columns) for synthetic datasets sharing one raw dir.

    parts: [(truth, segment)] in time order; z continues across parts and a seam starts each
    part after the first. relabel(r, c, R, C) -> (row, col): label of file tile{r}-{c} in an
    R x C grid (default: unchanged). exclude: z or (z, tile) entries marked excluded.
    Returns {(z, tile): true (x, y) origin}.
    """
    rows, origin, z0 = [], {}, 0
    for k, (truth, segment) in enumerate(parts):
        h, w = truth.tile_shape
        R, C = (max(int(t.split("-")[i]) for t in truth.tiles) + 1 for i in (0, 1))
        for name, zs in truth.files.items():
            r, c = map(int, re.search(r"tile(\d+)-(\d+)", name).groups())
            row, col = relabel(r, c, R, C) if relabel else (r, c)
            tile = f"{row}-{col}"
            for i, z in enumerate(zs):
                gz = z0 + z
                origin[(gz, tile)] = truth.tile_origin[f"{r}-{c}"]
                rows.append({"z": gz, "timestamp": truth.timestamps[z].isoformat(), "tile": tile,
                             "tile_row": row, "tile_col": col, "file": name, "index": i, "height": h, "width": w,
                             "segment": segment, "seam": k > 0 and z == 0,
                             "excluded": gz in exclude or (gz, tile) in exclude, "exclude_reason": "", "label": ""})
        z0 += len(truth.timestamps)
    path = out_dir / "work" / "check" / "slices.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["z", "tile"]).to_csv(path, index=False)
    return origin


def swap(r, c, R, C):
    return c, r


def setup_run(root, name, parts, relabel=None, exclude=(), selection=None, **opts):
    """Config + slices.csv in root/name; returns (config path, out dir, true origins)."""
    out = root / name
    origin = write_slices(out, parts, relabel, exclude)
    cfg = synth.write_config(root / f"{name}.yaml", parts[0][0].raw_dir, out,
                             stitch={**OPTS, **opts}, selection=selection or {})
    return cfg, out, origin


def stitched(root, name, parts, tasks=1, **kw):
    cfg, out, origin = setup_run(root, name, parts, **kw)
    for i in range(tasks):
        assert stitch.main(["run", "--config", str(cfg), "--task-id", str(i), "--num-tasks", str(tasks)]) == 0
    assert stitch.main(["merge", "--config", str(cfg)]) == 0
    return out, origin


def assert_matches_truth(tiles, origin, tol=0.2):
    """tile -> montage translations equal the true origins (whose min corner is (0, 0) too)."""
    got = tiles[["tx", "ty"]].to_numpy()
    true = np.array([origin[k] for k in zip(tiles["z"], tiles["tile"])])
    assert np.abs(got - true).max() < tol, np.abs(got - true).max(axis=0)
    np.testing.assert_allclose(tiles[["a", "b", "c", "d"]].to_numpy(), [[1, 0, 0, 1]] * len(tiles))


def assert_pairs_match_truth(out, origin, tol=0.2):
    """Every measured pair maps tile_b -> tile_a by the true origin difference."""
    pairs = pd.read_csv(out / "work" / "stitch" / "pairs.csv", dtype={"tile_a": str, "tile_b": str})
    assert len(pairs)
    true = np.array([np.subtract(origin[(z, b)], origin[(z, a)]) for z, a, b in
                     zip(pairs["z"], pairs["tile_a"], pairs["tile_b"])])
    assert np.abs(pairs[["tx", "ty"]].to_numpy() - true).max() < tol
    return pairs


def read_tiles(out):
    return pd.read_csv(out / "work" / "stitch" / "tiles.csv", dtype={"tile": str})


@pytest.fixture(scope="module")
def two_segments(tmp_path_factory):
    """Segment 0: 3x3 grid of non-square tiles; segment 1: 2x2 grid; different x/y overlaps."""
    root = tmp_path_factory.mktemp("stitch")
    a = synth.make_dataset(root / "raw", n_slices=9, grid=(3, 3), tile_shape=(176, 224), overlap=(36, 52),
                           start=datetime(2026, 9, 24, 12), seed=1)
    b = synth.make_dataset(root / "raw", n_slices=7, grid=(2, 2), tile_shape=(192, 160), overlap=(44, 30),
                           start=datetime(2026, 9, 26, 12), seed=2)
    return root, [(a, 0), (b, 1)]


def test_sample_z():
    seg = [0] * 12 + [1] * 5
    df = pd.DataFrame({"z": range(17), "segment": seg, "seam": False, "excluded": False})
    df.loc[[7, 12], "seam"] = True        # z 12 is a segment start
    df.loc[[6, 7, 14], "excluded"] = True  # seam slice 7 excluded: neighbours are z 5 and z 8
    # segment 0 selected: 0 1 2 3 4 5 8 9 10 11 -> every 4th: 0 4 10, ends 0 11, seam 7: 5 | 8
    # segment 1 selected: 12 13 15 16 -> 12, ends 12 16 (seam 12 is its first z)
    assert stitch.sample_z(df, 4) == [0, 4, 5, 8, 10, 11, 12, 16]


def test_two_segments(two_segments):
    root, parts = two_segments
    # z 4 excluded; tile 1-1 missing from sampled slice 12
    out, origin = stitched(root, "plain", parts, exclude={4, (12, "1-1")})
    tiles = read_tiles(out)
    expected = sorted(k for k in origin if k[0] != 4 and k != (12, "1-1"))
    assert sorted(zip(tiles["z"], tiles["tile"])) == expected
    assert (tiles["segment"] == (tiles["z"] >= 9)).all()
    assert_matches_truth(tiles, origin)

    layout = json.loads((out / "work" / "stitch" / "layout.json").read_text())["segments"]
    for seg, grid, (oy, ox) in (("0", [3, 3], (36, 52)), ("1", [2, 2], (44, 30))):
        lay = layout[seg]
        assert lay["mode"] == "fixed" and lay["model"] == "translation" and lay["row_axis"] == "y"
        assert lay["grid_shape"] == grid
        for t, pos in lay["tiles"].items():
            assert (pos["grid_y"], pos["grid_x"]) == tuple(map(int, t.split("-")))
        assert lay["overlap_px"] == {"x": pytest.approx(ox, abs=0.5), "y": pytest.approx(oy, abs=0.5)}
        assert lay["n_good_samples"] == lay["n_samples"] and not lay["failed_samples"]

    pairs = assert_pairs_match_truth(out, origin)
    assert set(pairs["segment"]) == {0, 1}
    # every grid neighbour pair (12 in 3x3, 4 in 2x2, 2 with a tile missing) is used in every sample
    n_used = pairs[pairs.used].groupby("z").size()
    assert all(n >= (12 if z < 9 else 2 if z == 12 else 4) for z, n in n_used.items())
    assert (out / "qc" / "stitch.png").stat().st_size > 0


def test_filename_row_is_x(two_segments):
    root, parts = two_segments
    out, origin = stitched(root, "swapped", parts, relabel=swap)
    tiles = read_tiles(out)
    assert_matches_truth(tiles, origin)
    assert_pairs_match_truth(out, origin)
    layout = json.loads((out / "work" / "stitch" / "layout.json").read_text())["segments"]
    assert layout["0"]["row_axis"] == layout["1"]["row_axis"] == "x"
    truth = parts[0][0]
    for t in truth.tiles:  # file tile{r}-{c} is labelled "c-r": its grid_x is the label's row
        r, c = map(int, t.split("-"))
        assert (layout["0"]["tiles"][f"{c}-{r}"]["grid_y"], layout["0"]["tiles"][f"{c}-{r}"]["grid_x"]) == (r, c)


def test_reversed_numbering_and_fine_factor_2(two_segments):
    """Rows and columns numbered from the far corner: the reference tile 0-0 is the max corner, every
    neighbour offset is negative and the min-corner shift is non-zero. fine_factor 2 checks that
    fine keypoints are scaled back to full-res tile pixels."""
    root, parts = two_segments
    out, origin = stitched(root, "reversed", parts, relabel=lambda r, c, R, C: (R - 1 - r, C - 1 - c),
                           fine_factor=2, fine_margin_px=24)
    tiles = read_tiles(out)
    assert_matches_truth(tiles, origin)
    pairs = assert_pairs_match_truth(out, origin)
    assert (pairs[["tx", "ty"]].to_numpy() < 0.5).all()
    layout = json.loads((out / "work" / "stitch" / "layout.json").read_text())["segments"]
    for seg, (R, C) in (("0", (3, 3)), ("1", (2, 2))):
        assert layout[seg]["row_axis"] == "y" and layout[seg]["reference_tile"] == "0-0"
        for t, pos in layout[seg]["tiles"].items():
            r, c = map(int, t.split("-"))
            assert (pos["grid_y"], pos["grid_x"]) == (R - 1 - r, C - 1 - c)


def test_chunked_tasks_equal_single_task(two_segments):
    root, parts = two_segments
    single, _ = stitched(root, "single", parts)
    chunked, _ = stitched(root, "chunked", parts, tasks=3)
    for name in ("tiles.csv", "pairs.csv", "layout.json"):
        assert (single / "work" / "stitch" / name).read_text() == (chunked / "work" / "stitch" / name).read_text(), name
    # existing samples are skipped unless --overwrite
    cfg = root / "chunked.yaml"
    sample = sorted((chunked / "work" / "stitch" / "samples").glob("*.json"))[0]
    before = sample.stat().st_mtime_ns
    stitch.main(["run", "--config", str(cfg)])
    assert sample.stat().st_mtime_ns == before
    stitch.main(["run", "--config", str(cfg), "--overwrite"])
    assert sample.stat().st_mtime_ns != before


def test_layout_change_per_slice_and_fixed(tmp_path):
    """Same tiles, overlap changes at a seam: auto -> per_slice follows the step; fixed takes medians."""
    common = dict(n_slices=6, grid=(2, 2), tile_shape=(160, 192))
    a = synth.make_dataset(tmp_path / "raw", overlap=(30, 40), start=datetime(2026, 9, 24, 12), seed=3, **common)
    b = synth.make_dataset(tmp_path / "raw", overlap=(44, 26), start=datetime(2026, 9, 25, 12), seed=4, **common)
    parts = [(a, 0), (b, 0)]

    out, origin = stitched(tmp_path, "auto", parts)
    tiles = read_tiles(out)
    lay = json.loads((out / "work" / "stitch" / "layout.json").read_text())["segments"]["0"]
    assert lay["mode"] == "per_slice" and lay["max_deviation_px"] > 5  # 14 px step, median midway
    assert len(tiles) == 12 * 4
    assert_matches_truth(tiles, origin)  # including z 5 | 6 on either side of the seam

    out, _ = stitched(tmp_path, "fixed", parts, mode="fixed")
    tiles = read_tiles(out)
    assert json.loads((out / "work" / "stitch" / "layout.json").read_text())["segments"]["0"]["mode"] == "fixed"
    for t, g in tiles.groupby("tile"):
        assert g[["tx", "ty"]].nunique().max() == 1
        # 3 samples per side -> median is the midpoint of the two layouts
        mid = (np.array(a.tile_origin[t]) + b.tile_origin[t]) / 2
        np.testing.assert_allclose(g[["tx", "ty"]].iloc[0], mid, atol=0.5)


def _replace_slice(truth, z, tiles, fn):
    """Overwrite slice z of the given tiles' raw files with fn(shape) (uint16)."""
    for name, zs in truth.files.items():
        if z in zs and any(f"tile{t}" in name for t in tiles):
            with tifffile.TiffFile(truth.raw_dir / name) as tif:
                data, labels = tif.asarray(), tif.imagej_metadata["Labels"]
            data[zs.index(z)] = fn(data.shape[1:])
            synth.write_imagej(truth.raw_dir / name, data, labels)


def test_bad_samples_are_skipped(tmp_path):
    truth = synth.make_dataset(tmp_path / "raw", n_slices=10, start=datetime(2026, 9, 24, 12), seed=5)
    rng = np.random.default_rng(0)
    _replace_slice(truth, 3, truth.tiles, lambda s: rng.integers(26000, 41000, s, dtype=np.uint16))
    _replace_slice(truth, 6, ["0-1"], lambda s: np.full(s, 30000, np.uint16))
    out, origin = stitched(tmp_path, "bad", [(truth, 0)])
    for z in (3, 6):
        rec = json.loads((out / "work" / "stitch" / "samples" / f"z{z:06d}.json").read_text())
        assert not rec["ok"] and "too few matches" in rec["reason"]
    lay = json.loads((out / "work" / "stitch" / "layout.json").read_text())["segments"]["0"]
    assert lay["failed_samples"] == [3, 6] and lay["n_good_samples"] == 2 and lay["mode"] == "fixed"
    tiles = read_tiles(out)
    assert sorted(tiles["z"].unique()) == list(range(10))
    assert_matches_truth(tiles, origin)

    # clear errors: a sampled slice that was never run; a segment whose only sample failed
    cfg, _, _ = setup_run(tmp_path, "bad", [(truth, 0)], selection={"z_start": 2, "z_end": 3})
    with pytest.raises(SystemExit, match=r"1 of 2 sampled slices have no result \(e.g. z \[2\]\)"):
        stitch.main(["merge", "--config", str(cfg)])
    cfg, _, _ = setup_run(tmp_path, "bad", [(truth, 0)], selection={"z_start": 3, "z_end": 3})
    with pytest.raises(SystemExit, match="segment 0 .* no usable sampled slice .*z 3: too few matches"):
        stitch.main(["merge", "--config", str(cfg)])


def _grid_pairs(R, C, rng, diagonal=False):
    """True origins and exact translation correspondences between neighbouring tiles."""
    origin = {f"{r}-{c}": np.array([c * 100.0, r * 80.0]) for r in range(R) for c in range(C)}
    steps = [(0, 1), (1, 0)] + [(1, 1)] * diagonal
    pairs = {}
    for r in range(R):
        for c in range(C):
            for dr, dc in steps:
                if r + dr < R and c + dc < C:
                    a, b = f"{r}-{c}", f"{r + dr}-{c + dc}"
                    pm = origin[b] + rng.uniform(0, 20, (30, 2))  # montage points in the overlap
                    pairs[(a, b)] = (pm - origin[a], pm - origin[b])
    return origin, pairs


def test_solve_robust_bad_pair():
    rng = np.random.default_rng(1)
    shift = np.array([5.0, -3.0])  # 5.8 px: a 2x2 ring spreads it to 1.5 px per pair
    # 3x3: the bad pair closes loops with good pairs -> identified and removed
    origin, pairs = _grid_pairs(3, 3, rng)
    pa, pb = pairs[("1-1", "1-2")]
    pairs[("1-1", "1-2")] = (pa + shift, pb)
    T, kept, reason = stitch.solve_robust(pairs, list(origin), "0-0", "translation", tol=3.0)
    assert reason == "" and set(pairs) - set(kept) == {("1-1", "1-2")}
    for t in origin:
        np.testing.assert_allclose(T[t][:, 2], origin[t], atol=1e-6)
    # 2x2 ring: any of the four pairs could be the bad one -> the slice fails
    origin, pairs = _grid_pairs(2, 2, rng)
    pa, pb = pairs[("0-0", "0-1")]
    pairs[("0-0", "0-1")] = (pa + shift, pb)
    T, kept, reason = stitch.solve_robust(pairs, list(origin), "0-0", "translation", tol=3.0)
    assert T is None and "no single bad pair" in reason
    # consistent pairs (with diagonals) are all kept
    origin, pairs = _grid_pairs(2, 3, rng, diagonal=True)
    T, kept, reason = stitch.solve_robust(pairs, list(origin), "0-0", "translation", tol=3.0)
    assert kept == pairs and reason == ""


@pytest.mark.parametrize("model", ["rigid", "similarity", "affine"])
def test_solve_models_recover_known_transforms(model):
    """solve() recovers tile -> montage transforms exactly from consistent correspondences."""
    rng = np.random.default_rng(0)
    true = {"0-0": transforms.identity(),
            "0-1": np.array([[np.cos(0.002), -np.sin(0.002), 180.0], [np.sin(0.002), np.cos(0.002), 1.5]]),
            "1-0": np.array([[1.0, 0.0, -2.0], [0.0, 1.0, 150.0]])}
    if model in ("similarity", "affine"):
        true["1-0"][:, :2] *= 1.001
    if model == "affine":
        true["1-0"][0, 1] += 0.0005
    pairs = {}
    for ta, tb in [("0-0", "0-1"), ("0-0", "1-0"), ("0-1", "1-0")]:
        pm = rng.uniform(0, 300, (50, 2))  # montage points seen by both tiles
        pairs[(ta, tb)] = tuple(transforms.apply(transforms.invert(true[t]), pm) for t in (ta, tb))
    T, rms = stitch.solve(pairs, list(true), "0-0", model)
    for t in true:
        np.testing.assert_allclose(T[t], true[t], atol=1e-3 if model == "rigid" else 1e-6)
    assert max(rms.values()) < 0.05


def test_stale_samples_are_redone(tmp_path):
    """Samples left by an earlier run with another reference tile or other settings are redone by run
    and refused by merge (mixing samples solved with different references put tiles 50 px off)."""
    truth = synth.make_dataset(tmp_path / "raw", n_slices=8, tile_shape=(160, 192), overlap=(30, 40),
                               start=datetime(2026, 9, 24, 12), seed=3)
    gone = {(z, "0-0") for z in range(4)}
    # z 0-3 only, where 0-0 is excluded: the reference is 0-1
    cfg, out, _ = setup_run(tmp_path, "s", [(truth, 0)], exclude=gone, selection={"z_end": 3})
    stitch.main(["run", "--config", str(cfg)])
    sample = out / "work" / "stitch" / "samples" / "z000000.json"
    assert json.loads(sample.read_text())["reference"] == "0-1"
    # all z: the reference is 0-0, so samples 0 and 3 are redone (and fail: 0-0 is missing there)
    cfg, out, origin = setup_run(tmp_path, "s", [(truth, 0)], exclude=gone)
    stitch.main(["run", "--config", str(cfg)])
    rec = json.loads(sample.read_text())
    assert rec["reference"] == "0-0" and not rec["ok"] and "reference tile 0-0 missing" in rec["reason"]
    stitch.main(["merge", "--config", str(cfg)])
    assert_matches_truth(read_tiles(out), origin)
    # a sample made with other settings: merge refuses it, run redoes it
    path = out / "work" / "stitch" / "samples" / "z000006.json"
    rec = json.loads(path.read_text())
    rec["settings"]["min_inliers"] = 5
    path.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match=r"out of date \(z 6: made with different stitch.min_inliers\)"):
        stitch.main(["merge", "--config", str(cfg)])
    stitch.main(["run", "--config", str(cfg)])
    assert json.loads(path.read_text())["settings"]["min_inliers"] == stitch.DEFAULTS["stitch"]["min_inliers"]
    assert stitch.main(["merge", "--config", str(cfg)]) == 0


def test_merge_per_slice_and_fixed_from_samples():
    """per_slice: window-3 median over samples, linear in z between them, constant beyond the ends;
    fixed: per-tile median. One shift per segment puts the montage min corner at (0, 0)."""
    zs = [0, 1, 2, 4, 5, 7, 8, 9, 10, 12]  # selected z, non-contiguous
    rows = pd.DataFrame([{"z": z, "timestamp": pd.Timestamp(2026, 9, 24) + pd.Timedelta(minutes=z), "tile": t,
                          "tile_row": 0, "tile_col": int(t[-1]), "height": 50, "width": 80, "segment": 3}
                         for z in zs for t in ("0-0", "0-1")])
    tx = {1: 70.0, 4: 73.0, 8: 77.0, 10: 99.0, 12: 81.0}  # z 10 is an outlier
    recs = [{"z": z, "ok": True, "reason": "", "reference": "0-0",
             "tiles": {"0-0": transforms.identity().tolist(), "0-1": transforms.translation(x, -2.0).tolist()}}
            for z, x in tx.items()]
    recs.insert(2, {"z": 5, "ok": False, "reason": "too few matches", "reference": "0-0", "tiles": {}})
    opts = stitch.DEFAULTS["stitch"]
    out, layout, _ = stitch.merge_segment(rows, recs, opts)
    assert layout["mode"] == "per_slice" and layout["failed_samples"] == [5] and layout["n_good_samples"] == 5
    t1 = out[out["tile"] == "0-1"].set_index("z")
    # filtered samples 70 73 77 81 81 at z 1 4 8 10 12
    expected = {0: 70, 1: 70, 2: 71, 4: 73, 5: 74, 7: 76, 8: 77, 9: 79, 10: 81, 12: 81}
    np.testing.assert_allclose(t1["tx"], [expected[z] for z in t1.index])
    np.testing.assert_allclose(t1["ty"], 0)  # tile 0-1 is 2 px above the reference: all move down by 2
    np.testing.assert_allclose(out.loc[out["tile"] == "0-0", ["tx", "ty"]], [[0, 2]] * len(zs))
    assert (out["segment"] == 3).all() and len(out) == 2 * len(zs)

    out, layout, _ = stitch.merge_segment(rows, recs, {**opts, "mode": "fixed"})
    assert layout["mode"] == "fixed" and layout["overlap_px"]["x"] == 3
    np.testing.assert_allclose(out.loc[out["tile"] == "0-1", ["tx", "ty"]], [[77, 0]] * len(zs))


@pytest.mark.parametrize("model, tol", [("rigid", 0.3), ("affine", 0.75)])
def test_models_end_to_end(tmp_path, model, tol):
    """Non-translation fine fits and solve on synthetic tiles: every tile corner lands on its true place
    (affine extrapolates its linear part from thin overlaps, hence the larger tolerance)."""
    truth = synth.make_dataset(tmp_path / "raw", n_slices=4, grid=(2, 3), tile_shape=(160, 192), overlap=(48, 56),
                               start=datetime(2026, 9, 24, 12), seed=6)
    out, origin = stitched(tmp_path, model, [(truth, 0)], model=model, mode="per_slice")
    h, w = truth.tile_shape
    pts = transforms.corners(w, h)
    err = [transforms.apply(transforms.from_row(r), pts) - pts - origin[(r["z"], r["tile"])]
           for _, r in read_tiles(out).iterrows()]
    assert np.abs(err).max() < tol


def test_with_check_slices(tmp_path):
    """slices.csv as written by pipeline.check: its seams (time gap, part restart) are sampled on both sides."""
    check = pytest.importorskip("pipeline.check")
    truth = synth.make_dataset(tmp_path / "raw", n_slices=12, parts=[7], gaps={4: 3000}, tile_shape=(160, 192),
                               overlap=(30, 40), seed=7)
    cfg = synth.write_config(tmp_path / "c.yaml", truth.raw_dir, tmp_path / "out", check={"min_age_minutes": 0},
                             stitch={**OPTS, "sample_every": 4})
    assert check.main(["--config", str(cfg)]) == 0
    assert stitch.main(["run", "--config", str(cfg)]) == 0 and stitch.main(["merge", "--config", str(cfg)]) == 0
    samples = sorted(int(p.stem[1:]) for p in (tmp_path / "out" / "work" / "stitch" / "samples").glob("z*.json"))
    assert samples == [0, 3, 4, 6, 7, 8, 11]
    tiles = read_tiles(tmp_path / "out")
    assert len(tiles) == 12 * 4
    assert_matches_truth(tiles, {(z, t): truth.tile_origin[t] for z in range(12) for t in truth.tiles})


def test_coarse_search_retries_at_higher_resolution(two_segments, tmp_path):
    """At coarse_factor 8 these small tiles have too few features to find the overlaps (as a ~1%
    overlap would on real noisy tiles); the search is retried at 4 and 2 instead of failing the run."""
    root, parts = two_segments
    part = [parts[1]]
    cfg, out, origin = setup_run(tmp_path, "no_retry", part, coarse_factor=8, min_coarse_factor=8)
    assert stitch.main(["run", "--config", str(cfg)]) == 0
    recs = [json.loads(p.read_text()) for p in sorted((out / "work" / "stitch" / "samples").glob("*.json"))]
    assert recs and not any(r["ok"] for r in recs)
    assert all("not connected" in r["reason"] for r in recs)
    assert all(p["note"].startswith("no coarse match") for r in recs for p in r["pairs"] if p["A"] is None)

    out2, origin2 = stitched(tmp_path, "retry", part, coarse_factor=8)   # min_coarse_factor 2 (default)
    recs = [json.loads(p.read_text()) for p in sorted((out2 / "work" / "stitch" / "samples").glob("*.json"))]
    assert all(r["ok"] for r in recs) and {r["coarse_factor_used"] for r in recs} <= {4, 2}
    assert_matches_truth(read_tiles(out2), origin2)


def _real_like_2x2():
    """A 2x2 layout like the real data (13875 x 12751 tiles, ~600 px overlaps, bottom row ~890 px
    right) whose tiles differ by small rotations, scales and shears (scan geometry), with
    correspondences only in thin overlap strips."""
    W, H = 13875, 12751
    def tile(tx, ty, rot=0.0, scale=1.0, shear=0.0):
        c, s = np.cos(rot) * scale, np.sin(rot) * scale
        return np.array([[c, -s + shear, tx], [s, c, ty]])
    true = {"0-0": transforms.identity(), "0-1": tile(13263, -63, 7e-4, 1.0003),
            "1-0": tile(892, 12160, -5e-4, 0.9997, 2e-4), "1-1": tile(14104, 12112, 3e-4, 1.0002)}
    rng = np.random.default_rng(1)
    pairs = {}
    for ta, tb in [("0-0", "0-1"), ("0-0", "1-0"), ("0-1", "1-1"), ("1-0", "1-1")]:
        # montage points in the overlap of the two tiles' nominal rectangles
        xa0, ya0, xa1, ya1 = transforms.bbox(true[ta], W, H)
        xb0, yb0, xb1, yb1 = transforms.bbox(true[tb], W, H)
        x0, x1, y0, y1 = max(xa0, xb0) + 20, min(xa1, xb1) - 20, max(ya0, yb0) + 20, min(ya1, yb1) - 20
        pm = np.column_stack([rng.uniform(x0, x1, 400), rng.uniform(y0, y1, 400)])
        pa, pb = (transforms.apply(transforms.invert(true[t]), pm) for t in (ta, tb))
        pairs[(ta, tb)] = (pa + rng.normal(0, 0.3, pa.shape), pb + rng.normal(0, 0.3, pb.shape))
    return true, pairs


def test_affine_rigid_fixes_real_like_loop_closure():
    """On the geometry seen in the real P667 tiles, translation-only leaves a loop-closure error of
    several px (the first real run failed with 7.5-9.5 px); Janelia's affine_rigid solve closes it."""
    true, pairs = _real_like_2x2()
    tiles = list(true)
    assert stitch.loop_error(pairs, tiles, "0-0", "translation") > 4
    _, rms_tr = stitch.solve(pairs, tiles, "0-0", "translation")
    T, rms = stitch.solve(pairs, tiles, "0-0", "affine_rigid")
    # Joint residuals (what stitch checks for this model; a leave-one-out loop check would extrapolate
    # a tile's affine from a single thin strip) drop from translation's 3.8-5.3 px to the noise level.
    assert min(rms_tr.values()) > 3 and max(rms.values()) < 1.0
    T_ok, _, reason = stitch.solve_filtered(pairs, tiles, "0-0", "affine_rigid", tol=10.0)
    assert T_ok is not None and reason == ""
    # Outer tile corners are ~13,000 px from any overlap, so they are extrapolated (and the rigid pull
    # shrinks scale differences slightly); still far closer than translation-only (~10 px).
    corners = transforms.corners(13875, 12751)
    err = max(np.abs(transforms.apply(T[t], corners) - transforms.apply(true[t], corners)).max() for t in tiles)
    T_tr, _ = stitch.solve(pairs, tiles, "0-0", "translation")
    err_tr = max(np.abs(transforms.apply(T_tr[t], corners) - transforms.apply(true[t], corners)).max() for t in tiles)
    assert err < 5 and err < err_tr / 2


def test_affine_rigid_lambda_near_one_is_rigid():
    """A strong rigid pull leaves (small-angle) rotations: a = d = 1, b = -c; a weak one allows scale."""
    true, pairs = _real_like_2x2()
    T, _ = stitch.solve(pairs, list(true), "0-0", "affine_rigid", rigid_lambda=0.9999)
    for A in T.values():
        L = A[:, :2]
        assert abs(L[0, 0] - L[1, 1]) < 1e-5 and abs(L[0, 1] + L[1, 0]) < 1e-5 and abs(L[0, 0] - 1) < 1e-5
    T, _ = stitch.solve(pairs, list(true), "0-0", "affine_rigid", rigid_lambda=0.001)
    assert max(abs(np.linalg.det(A[:, :2]) - 1) for A in T.values()) > 1e-4   # scale differences kept


def test_single_tile_segment_then_grid(tmp_path):
    """A 1x1 phase (one large field of view, as at the start of P667 35i) before a 2x2 grid: the
    single tile is its own montage (identity) instead of crashing the solve."""
    a = synth.make_dataset(tmp_path / "raw", n_slices=6, grid=(1, 1), tile_shape=(240, 300),
                           start=datetime(2026, 6, 5, 12), seed=4)
    b = synth.make_dataset(tmp_path / "raw", n_slices=6, grid=(2, 2), tile_shape=(192, 160), overlap=(44, 30),
                           start=datetime(2026, 6, 6, 12), seed=5)
    out, origin = stitched(tmp_path, "single", [(a, 0), (b, 1)])
    tiles = read_tiles(out)
    assert sorted(zip(tiles["z"], tiles["tile"])) == sorted(origin)
    assert_matches_truth(tiles, origin)
    single = tiles[tiles["segment"] == 0]
    np.testing.assert_allclose(single[["a", "b", "tx", "c", "d", "ty"]].to_numpy(), [[1, 0, 0, 0, 1, 0]] * len(single))
    layout = json.loads((out / "work" / "stitch" / "layout.json").read_text())["segments"]
    assert layout["0"]["grid_shape"] == [1, 1] and layout["1"]["grid_shape"] == [2, 2]
