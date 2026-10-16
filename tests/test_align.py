"""Tests for pipeline.align against synthetic ground truth (drift known per slice)."""

import json
import logging
import shutil

import numpy as np
import pandas as pd
import pytest
import tifffile

import synth
from pipeline import align, transforms

N = 30
PART = 16                 # acquisition restart (new _part file, time gap, drift jump)
JUMP = (9, -7)
ALIGN = {"scale": 1.0, "max_points_per_pair": 40}
CHUNK0 = "chunk_000000-000100.npz"   # the one chunk of N slices with the default chunk_slices
# Real 8 nm slices stay similar over many slices; the synthetic volume needs finer sampling for that.
DZ = 0.25


def write_inputs(truth, out, parts=(), excluded=(), excluded_tiles=()):
    """check/slices.csv and stitch/tiles.csv (true tile origins) from the synthetic ground truth.

    excluded: z excluded entirely; excluded_tiles: (z, tile) excluded and absent from tiles.csv.
    """
    rows = []
    for name, zs in truth.files.items():
        tile = name.split("_tile")[1].split("_")[0].removesuffix(".tif")
        r, c = map(int, tile.split("-"))
        for i, z in enumerate(zs):
            ex = z in excluded or (z, tile) in excluded_tiles
            rows.append({"z": z, "timestamp": truth.timestamps[z].isoformat(), "tile": tile, "tile_row": r,
                         "tile_col": c, "file": name, "index": i, "height": truth.tile_shape[0],
                         "width": truth.tile_shape[1], "segment": 0, "seam": z in parts,
                         "excluded": ex, "exclude_reason": "test" if ex else "", "label": ""})
    (out / "work" / "check").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["z", "tile"]).to_csv(out / "work" / "check" / "slices.csv", index=False)
    tiles = transforms.to_frame(({"z": z, "tile": t}, transforms.translation(*truth.tile_origin[t]))
                                for z in range(len(truth.timestamps)) for t in truth.tiles
                                if (z, t) not in excluded_tiles)
    tiles["segment"] = 0
    (out / "work" / "stitch").mkdir(parents=True, exist_ok=True)
    tiles.to_csv(out / "work" / "stitch" / "tiles.csv", index=False)


def read_transforms(out):
    df = pd.read_csv(out / "work" / "align" / "transforms.csv")
    return df, df[transforms.COLUMNS].to_numpy().reshape(-1, 2, 3)


def drift_error(drift, df, mats, at=None):
    """Max |recovered - true| translation relative to the first selected z (px), evaluated at point ``at``.

    drift: true montage -> aligned translation per global z (up to a constant).
    """
    at = np.zeros(2) if at is None else np.asarray(at, float)
    moved = np.array([transforms.apply(A, at)[0] for A in mats]) - at
    zs = df["z"].to_numpy()
    true = drift[zs] - drift[zs[0]]
    return np.abs((moved - moved[0]) - true).max()


def load_npz(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def edit_stitch(out, fn):
    """Replace every transform A in stitch/tiles.csv by fn(z, tile, A)."""
    path = out / "work" / "stitch" / "tiles.csv"
    df = pd.read_csv(path, dtype={"tile": str})
    mats = [fn(z, t, transforms.from_row(r)) for z, t, (_, r) in zip(df["z"], df["tile"], df.iterrows())]
    df[transforms.COLUMNS] = np.array(mats).reshape(-1, 6)
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def drift_data(tmp_path_factory):
    """2×2 dataset with random-walk drift and a jump at a restart; align run done once (single task)."""
    root = tmp_path_factory.mktemp("align")
    truth = synth.make_dataset(root / "raw", n_slices=N, parts=[PART], drift_jumps={PART: JUMP},
                               z_positions=np.arange(N) * DZ)
    out = root / "out"
    write_inputs(truth, out, parts=[PART])
    cfg = synth.write_config(root / "config.yaml", truth.raw_dir, out, align=ALIGN)
    assert align.main(["run", "--config", str(cfg)]) == 0
    return truth, out


def solve_copy(drift_data, tmp_path, edit=None, stitch=None, **overrides):
    """Solve a copy of the module run's matches, optionally changed in place by ``edit(m)`` (chunk
    arrays) or with stitch/tiles.csv changed by ``stitch(z, tile, A) -> A``."""
    truth, out = drift_data
    new = tmp_path / "out"
    for step in ("check", "stitch", "align/matches"):
        shutil.copytree(out / "work" / step, new / "work" / step)
    if edit:
        path = new / "work" / "align" / "matches" / CHUNK0
        m = load_npz(path)
        edit(m)
        np.savez(path, **m)
    if stitch:
        edit_stitch(new, stitch)
    cfg = synth.write_config(tmp_path / "config.yaml", truth.raw_dir, new, align={**ALIGN, **overrides})
    assert align.main(["solve", "--config", str(cfg)]) == 0
    return new


def test_recovers_drift_with_jump(drift_data):
    truth, out = drift_data
    m = load_npz(out / "work" / "align" / "matches" / CHUNK0)
    # Points are in montage coordinates: montage -> volume is +drift, so matched points agree there.
    gap = m["z_b"] - m["z_a"]
    assert set(gap.tolist()) == {1, 2, 3, 4}
    assert {(PART - 1, PART), (PART - 2, PART + 1)} <= set(zip(m["z_a"].tolist(), m["z_b"].tolist()))
    err = (m["pa"] + truth.drift[m["z_a"]]) - (m["pb"] + truth.drift[m["z_b"]])
    # Single keypoints between different slices are ~0.5 px noisy but unbiased.
    assert np.median(np.linalg.norm(err, axis=1)) < 1.0
    assert np.abs(err.mean(0)).max() < 0.05
    # Tile-pixel points map to the montage points through the stitch transform of their own tile.
    origin = np.array([truth.tile_origin[t] for t in m["tiles"]])
    np.testing.assert_allclose(m["qa"] + origin[m["ta"]], m["pa"], atol=1e-3)
    np.testing.assert_allclose(m["qb"] + origin[m["tb"]], m["pb"], atol=1e-3)
    # At most max_points_per_pair per tile: up to 4 tiles' worth per slice pair.
    per_pair = pd.Series(1, index=pd.MultiIndex.from_arrays([m["z_a"], m["z_b"]])).groupby(level=[0, 1]).size()
    assert 40 < per_pair.max() <= 4 * 40

    cfg = out.parent / "config.yaml"
    assert align.main(["solve", "--config", str(cfg)]) == 0
    df, mats = read_transforms(out)
    assert df["z"].tolist() == list(range(N))
    assert df["timestamp"].iloc[0] == truth.timestamps[0].isoformat()
    np.testing.assert_allclose(mats[:, :, :2], np.tile(np.eye(2), (N, 1, 1)))
    np.testing.assert_allclose(mats[0], transforms.identity())
    assert drift_error(truth.drift, df, mats) < 0.5
    res = pd.read_csv(out / "work" / "align" / "residuals.csv")
    assert list(res.columns) == ["z_a", "z_b", "n", "rms_px", "max_px", "rejected"]
    assert not res["rejected"].any() and res["rms_px"].median() < 1.0
    assert (out / "qc" / "align_drift.png").stat().st_size > 0


def test_chunked_tasks_equal_single_task(drift_data, tmp_path):
    """Several array tasks, data growing between runs, resume, a later exclusion, changed settings."""
    truth, out = drift_data
    out2 = tmp_path / "out"
    write_inputs(truth, out2, parts=[PART])
    run = lambda cfg, *args: align.main(["run", "--config", str(cfg), *args])
    task = lambda i: ["--task-id", str(i), "--num-tasks", "3"]
    matches = out2 / "work" / "align" / "matches"
    mtimes = lambda: {p.name: p.stat().st_mtime_ns for p in matches.glob("*.npz")}
    changed = lambda before: {k for k, t in mtimes().items() if before.get(k) != t}
    chunked = {**ALIGN, "chunk_slices": 7}
    # Yesterday's data ended at z 15: chunk 7-14 then had only two slices after its core.
    early = synth.write_config(tmp_path / "early.yaml", truth.raw_dir, out2, align=chunked,
                               selection={"z_end": 15})
    for i in range(3):
        run(early, *task(i))
    mtime = mtimes()
    assert sorted(mtime) == ["chunk_000000-000007.npz", "chunk_000007-000014.npz", "chunk_000014-000021.npz"]

    cfg = synth.write_config(tmp_path / "config.yaml", truth.raw_dir, out2, align=chunked)
    run(cfg, *task(0))
    run(cfg, *task(1))
    assert changed(mtime) == {"chunk_000007-000014.npz", "chunk_000021-000028.npz", "chunk_000028-000035.npz"}
    with pytest.raises(FileNotFoundError, match="1 match chunks .*chunk_000014-000021"):
        align.main(["solve", "--config", str(cfg)])
    run(cfg, *task(2))
    mtime = mtimes()
    run(cfg)   # resume: everything is current, nothing is redone
    assert not changed(mtime)

    align.main(["solve", "--config", str(cfg)])
    single = solve_copy(drift_data, tmp_path / "single")
    np.testing.assert_allclose(read_transforms(out2)[1], read_transforms(single)[1], atol=1e-6)

    # Excluding a slice, or one tile of a slice, later redoes only the chunk holding it: chunk bounds
    # are fixed in global z, and a chunk records which (z, tile) it read.
    path = out2 / "work" / "check" / "slices.csv"
    sl = pd.read_csv(path)
    sl.loc[(sl["z"] == 19) | ((sl["z"] == 3) & (sl["tile"] == "1-0")), "excluded"] = True
    sl.to_csv(path, index=False)
    run(cfg)
    assert changed(mtime) == {"chunk_000000-000007.npz", "chunk_000014-000021.npz"}
    assert 19 not in load_npz(matches / "chunk_000014-000021.npz")["z_b"]
    m = load_npz(matches / "chunk_000000-000007.npz")
    assert "1-0" not in m["tiles"][m["ta"][m["z_a"] == 3]]

    # Other matching settings: solve refuses every chunk, run redoes them; --overwrite redoes current ones.
    mtime = mtimes()
    other = synth.write_config(tmp_path / "other.yaml", truth.raw_dir, out2, align={**chunked, "ratio": 0.7})
    with pytest.raises(FileNotFoundError, match="5 match chunks"):
        align.main(["solve", "--config", str(other)])
    run(other, *task(2))
    assert changed(mtime) == {"chunk_000014-000021.npz"}
    mtime = mtimes()
    run(other, *task(2))
    assert not changed(mtime)
    run(other, *task(2), "--overwrite")
    assert changed(mtime) == {"chunk_000014-000021.npz"}


def test_solve_uses_current_stitch(drift_data, tmp_path):
    """Re-running stitch needs no new matches: solve maps the tile-pixel points with the current tiles.csv."""
    truth, _ = drift_data
    shift = np.array([5.0, -3.0])   # montage frame of z >= 10 moves by +shift

    def moved(z, tile, A):
        return transforms.compose(transforms.translation(*shift), A) if z >= 10 else A

    df, mats = read_transforms(solve_copy(drift_data, tmp_path, stitch=moved))
    expected = truth.drift - np.where(np.arange(N)[:, None] >= 10, shift, 0)
    assert drift_error(expected, df, mats) < 0.5
    assert drift_error(truth.drift, df, mats) > 4


def test_corrupted_and_excluded_slices(tmp_path, caplog):
    truth = synth.make_dataset(tmp_path / "raw", n_slices=20, seed=3, z_positions=np.arange(20) * DZ)
    bad, skipped, no_tile = 11, 5, (8, "0-1")
    rng = np.random.default_rng(0)
    for name, zs in truth.files.items():   # every tile of z=bad becomes noise
        if bad in zs:
            with tifffile.TiffFile(truth.raw_dir / name) as tif:
                data, labels = tif.asarray(), tif.imagej_metadata["Labels"]
            data[zs.index(bad)] = rng.integers(0, 65535, truth.tile_shape, np.uint16)
            synth.write_imagej(truth.raw_dir / name, data, labels)
    out = tmp_path / "out"
    write_inputs(truth, out, excluded=[skipped], excluded_tiles={no_tile})
    cfg = synth.write_config(tmp_path / "config.yaml", truth.raw_dir, out,
                             align={**ALIGN, "chunk_slices": 6})
    align.main(["run", "--config", str(cfg), "--num-tasks", "2", "--task-id", "0"])
    align.main(["run", "--config", str(cfg), "--num-tasks", "2", "--task-id", "1"])
    with caplog.at_level(logging.WARNING, logger="pipeline.align"):
        align.main(["solve", "--config", str(cfg)])
    assert any("smoothness" in r.message and f"z {bad}" in r.message for r in caplog.records)

    df, mats = read_transforms(out)
    zs = df["z"].tolist()
    assert skipped not in zs and len(zs) == 19
    res = pd.read_csv(out / "work" / "align" / "residuals.csv")
    # Neighbours are by position in the selected list: 4 and 6 are adjacent once 5 is excluded.
    assert ((res["z_a"] == skipped - 1) & (res["z_b"] == skipped + 1)).any()
    assert not ((res["z_a"] == bad) | (res["z_b"] == bad))[~res["rejected"]].any()
    # z 8 lacks one tile: it is still matched with the other three.
    m = load_npz(out / "work" / "align" / "matches" / "chunk_000006-000012.npz")
    at8 = m["z_a"] == no_tile[0]
    assert set(m["tiles"][m["ta"][at8]]) == set(truth.tiles) - {no_tile[1]}
    assert ((res["z_a"] == no_tile[0]) & ~res["rejected"]).sum() >= 3
    good = [i for i, z in enumerate(zs) if z != bad]
    assert drift_error(truth.drift, df.iloc[good], mats[good]) < 0.5
    i = zs.index(bad)
    assert np.all(np.isfinite(mats[i]))
    np.testing.assert_allclose(mats[i, :, 2], (mats[i - 1, :, 2] + mats[i + 1, :, 2]) / 2, atol=0.05)


def test_outlier_pair_rejected(drift_data, tmp_path):
    truth, _ = drift_data

    def wrong_pair(m):   # a confident but wrong match: every point of z 8 -> 10 off by (25, -18) px
        m["qb"][(m["z_a"] == 8) & (m["z_b"] == 10)] += np.float32([25, -18])

    new = solve_copy(drift_data, tmp_path, edit=wrong_pair)
    res = pd.read_csv(new / "work" / "align" / "residuals.csv")
    assert res.loc[res["rejected"], ["z_a", "z_b"]].values.tolist() == [[8, 10]]
    df, mats = read_transforms(new)
    assert drift_error(truth.drift, df, mats) < 0.5


def test_affine_model_near_identity(drift_data, tmp_path):
    truth, _ = drift_data
    df, mats = read_transforms(solve_copy(drift_data, tmp_path, model="affine"))
    assert np.abs(mats[:, :, :2] - np.eye(2)).max() < 1e-3
    np.testing.assert_allclose(mats[0], transforms.identity(), atol=1e-12)
    centre = np.array(truth.montage_shape[::-1]) / 2
    assert drift_error(truth.drift, df, mats, at=centre) < 0.5


def test_affine_recovers_linear_distortion(drift_data, tmp_path):
    """One slice's montage is scaled and sheared about the centre: its transform undoes exactly that."""
    truth, _ = drift_data
    z0, centre = 12, np.array(truth.montage_shape[::-1]) / 2
    L = np.array([[1.004, 0.003], [-0.002, 0.997]])
    S = np.hstack([L, (centre - L @ centre)[:, None]])   # p -> centre + L (p - centre)

    def distort(z, tile, A):
        return transforms.compose(S, A) if z == z0 else A

    df, mats = read_transforms(solve_copy(drift_data, tmp_path, stitch=distort, model="affine"))
    dev = np.linalg.inv(L) - np.eye(2)    # montage' -> aligned = (montage -> aligned) o S^-1
    got = mats[z0, :, :2] - np.eye(2)
    # Shrunk toward identity by the regularisation (~25% here), but in the right direction: a
    # transposed or sign-flipped linear part fails both checks.
    share = (got * dev).sum() / (dev * dev).sum()
    assert 0.6 < share < 1.05
    assert np.abs(got - share * dev).max() < 0.15 * np.abs(dev).max()
    others = np.delete(mats, z0, axis=0)
    assert np.abs(others[:, :, :2] - np.eye(2)).max() < 1e-3
    assert drift_error(truth.drift, df, mats, at=centre) < 0.5


def test_remove_trend_linear(drift_data, tmp_path):
    truth, _ = drift_data
    ramp = np.array([3.0, -2.0])

    def add_ramp(z, tile, A):   # montage of slice z shifted by z * ramp: a linear drift of -ramp per slice
        return transforms.compose(transforms.translation(*(z * ramp)), A)

    new = solve_copy(drift_data, tmp_path, stitch=add_ramp, remove_trend="linear")
    df, mats = read_transforms(new)
    z = np.arange(N)
    t = truth.drift - z[:, None] * ramp
    jump = np.where(z[:, None] >= PART, t[PART] - t[PART - 1], 0)   # the restart step is kept
    fit = np.column_stack([np.polyval(np.polyfit(z, (t - jump)[:, i], 1), z) for i in range(2)])
    expected = t - fit
    expected -= expected[0]
    assert np.abs(mats[:, :, 2] - expected).max() < 0.5
    np.testing.assert_allclose(mats[0], transforms.identity(), atol=1e-9)
    # Residuals describe the fit, not the trend that was removed afterwards (regression).
    res = pd.read_csv(new / "work" / "align" / "residuals.csv")
    assert res["rms_px"].median() < 1.0


def test_remove_trend_lowpass_keeps_fast_drift(drift_data, tmp_path):
    _, mats = read_transforms(solve_copy(drift_data, tmp_path, remove_trend="lowpass", trend_window_slices=6))
    _, plain = read_transforms(solve_copy(drift_data, tmp_path / "plain"))
    t, t0 = mats[:, :, 2], plain[:, :, 2]
    # The restart jump (~8 px) is kept up to the slow trend's slope; slow drift is gone, fast steps stay.
    assert np.abs((t[PART] - t[PART - 1]) - (t0[PART] - t0[PART - 1])).max() < 1.0
    no_jump = lambda x: x - np.where(np.arange(N)[:, None] >= PART, x[PART] - x[PART - 1], 0)
    assert np.abs(no_jump(t)).max() < 0.5 * np.abs(no_jump(t0)).max()
    steps, steps0 = np.diff(t, axis=0), np.diff(t0, axis=0)
    assert np.abs(steps - steps0).mean() < 0.5 * np.abs(steps0).mean()


def _layout_change(tmp_path, n=12, seg=6, frame=(37.0, -21.0), L=np.eye(2), stitch_fn=None):
    """From z ``seg`` the slices are re-acquired as a 1×2 grid of tall tiles (other ids, other shape)
    whose montage frame is the first segment's mapped by ``L`` and offset by ``frame``
    (stitch: tile pixel q -> L (q + tile origin) + frame, or stitch_fn(tile, origin, montage width)).
    Returns (truth, out, montage map of segment 1)."""
    frame = np.asarray(frame, float)
    truth = synth.make_dataset(tmp_path / "raw", n_slices=n, drift_jumps={seg: (-8, 5)}, seed=5,
                               z_positions=np.arange(n) * DZ)
    out = tmp_path / "out"
    write_inputs(truth, out, parts=[seg], excluded_tiles={(3, "1-1")})
    H, W = truth.montage_shape
    w = 230
    new = {"0-0": (0, 0), "0-1": (W - w, 0)}
    zs = list(range(seg, n))
    rng = np.random.default_rng(1)
    rows, stitch = [], []
    for tile, (x0, y0) in new.items():
        img = np.stack([truth.montage(z)[:, x0:x0 + w] for z in zs]) + rng.normal(0, 0.01, (len(zs), H, w))
        name = f"seg1_tile{tile}.tif"
        r, c = map(int, tile.split("-"))
        synth.write_imagej(truth.raw_dir / name, np.clip(26000 + img * 15000, 0, 65535).astype(np.uint16),
                           [synth.LABEL.format(ts=truth.timestamps[z], r=r, c=c) for z in zs])
        A = (stitch_fn(tile, (x0, y0), W) if stitch_fn else np.column_stack([L, L @ np.array([x0, y0], float) + frame]))
        for i, z in enumerate(zs):
            rows.append({"z": z, "timestamp": truth.timestamps[z].isoformat(), "tile": tile, "tile_row": r,
                         "tile_col": c, "file": name, "index": i, "height": H, "width": w, "segment": 1,
                         "seam": z == seg, "excluded": False, "exclude_reason": "", "label": ""})
            stitch.append(({"z": z, "tile": tile, "segment": 1}, A))
    sl = pd.read_csv(out / "work" / "check" / "slices.csv")
    pd.concat([sl[sl["z"] < seg], pd.DataFrame(rows)]).to_csv(out / "work" / "check" / "slices.csv", index=False)
    tl = pd.read_csv(out / "work" / "stitch" / "tiles.csv", dtype={"tile": str})
    pd.concat([tl[tl["z"] < seg], transforms.to_frame(stitch)]).to_csv(out / "work" / "stitch" / "tiles.csv", index=False)
    return truth, out, np.column_stack([L, frame])


def test_segment_change(tmp_path):
    """Features are detected at half resolution, so points must be mapped back to full-res pixels;
    every tile combination is matched across the change."""
    n, seg, frame = 12, 6, np.array([37.0, -21.0])
    truth, out, _ = _layout_change(tmp_path, n, seg, frame)
    cfg = synth.write_config(tmp_path / "config.yaml", truth.raw_dir, out, align={**ALIGN, "scale": 0.5})
    align.main(["run", "--config", str(cfg)])
    expected = truth.drift - np.where(np.arange(n)[:, None] >= seg, frame, 0)   # montage' = montage + frame
    m = load_npz(out / "work" / "align" / "matches" / CHUNK0)
    z_a, z_b = m["z_a"], m["z_b"]
    err = (m["pa"] + expected[z_a]) - (m["pb"] + expected[z_b])
    assert np.abs(err.mean(0)).max() < 0.05 and np.median(np.linalg.norm(err, axis=1)) < 1.0
    # Across the change, old tiles are matched with new ones; within a segment only the same tile.
    across = (z_a < seg) & (z_b >= seg)
    combos = set(zip(m["tiles"][m["ta"][across]], m["tiles"][m["tb"][across]]))
    assert {("0-0", "0-0"), ("1-0", "0-0"), ("0-1", "0-1"), ("1-1", "0-1")} <= combos
    assert np.all(m["ta"][~across] == m["tb"][~across])
    assert "1-1" not in m["tiles"][m["ta"][z_a == 3]]
    for model in ("affine", "none"):
        cfg = synth.write_config(tmp_path / f"{model}.yaml", truth.raw_dir, out,
                                 align={**ALIGN, "scale": 0.5, "segment_change_model": model})
        align.main(["solve", "--config", str(cfg)])
        df, mats = read_transforms(out)
        assert drift_error(expected, df, mats) < 0.5, model
        assert drift_error(truth.drift, df, mats) > 10   # the frame offset is really in the result


def test_segment_change_with_stretch(tmp_path):
    """The new layout's frame is also 3% wider (the 35h 2x2 tiles are ~0.17% wider than the 3x3 ones):
    no translation fits across the change. Without the frame fit the solve can only compromise (on the
    35h 3x3 -> 2x2 change it rejected every pair across it, leaving a ~890 px jump); the affine fit
    recovers the stretch."""
    n, seg = 12, 6
    L = np.array([[1.03, 0.0], [0.0, 1.0]])
    truth, out, S = _layout_change(tmp_path, n, seg, L=L)
    cfg = synth.write_config(tmp_path / "config.yaml", truth.raw_dir, out, align={**ALIGN, "scale": 0.5})
    align.main(["run", "--config", str(cfg)])
    H, W = truth.montage_shape
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], float)

    def worst_error(**opts):
        """Max error at the montage corners, relative to the first slice (px)."""
        cfg = synth.write_config(tmp_path / "solve.yaml", truth.raw_dir, out,
                                 align={**ALIGN, "scale": 0.5, **opts})
        assert align.main(["solve", "--config", str(cfg)]) == 0
        df, mats = read_transforms(out)
        err = np.array([transforms.apply(A, corners if z < seg else transforms.apply(S, corners))
                        - (corners + truth.drift[z]) for z, A in zip(df["z"], mats)])
        return np.abs(err - err[0]).max()

    assert worst_error() < 0.5
    ch = pd.read_csv(out / "work" / "align" / "segment_changes.csv")
    assert ch[["z_before", "z_after", "segment_before", "segment_after"]].values.tolist() == [[seg - 1, seg, 0, 1]]
    assert abs(ch["scale_x"][0] - 1 / 1.03) < 1e-3 and abs(ch["scale_y"][0] - 1) < 1e-3
    assert ch["median_px"][0] < 1.0 < ch["translation_median_px"][0]
    assert worst_error(segment_change_model="translation") > 3
    assert worst_error(segment_change_model="none") > 3
    # Reference = the first segment's frame either way here (6 slices each: ties go to the first).
    assert worst_error(segment_reference="first") < 0.5


def test_local_tile_solve_at_layout_change(tmp_path):
    """After the change the left tile is stitched 4% too wide and the right one 4% too narrow, both
    about the middle of their overlap (so their seam still fits, as narrow overlaps allow): no
    transform for the whole field fits across the change. The tile-level solve, every tile of every
    slice in the window with its own affine, does, without breaking the seam."""
    n, seg, frame = 12, 6, np.array([37.0, -21.0])

    def stitch_fn(tile, origin, W):
        s = {"0-0": 1.04, "0-1": 0.96}[tile]
        return np.array([[s, 0, s * (origin[0] - W / 2) + W / 2 + frame[0]], [0, 1, origin[1] + frame[1]]])

    truth, out, _ = _layout_change(tmp_path, n, seg, frame, stitch_fn=stitch_fn)
    cfg = synth.write_config(tmp_path / "config.yaml", truth.raw_dir, out, align={**ALIGN, "scale": 0.5})
    align.main(["run", "--config", str(cfg)])

    def solve(**opts):
        opts = {"local_at_layout_changes": True, "elastic_min_px": None, **opts}
        cfg = synth.write_config(tmp_path / "solve.yaml", truth.raw_dir, out, align={**ALIGN, "scale": 0.5, **opts})
        assert align.main(["solve", "--config", str(cfg)]) == 0
        return (pd.read_csv(out / "work" / "align" / "local_windows.csv"),
                pd.read_csv(out / "work" / "align" / "local_tiles.csv", dtype={"tile": str}))

    # Seams held lightly (weight 4): the window is the whole stack, nothing held fixed.
    rep, tiles = solve(local_half_width=n, local_seam_weight=4.0)
    assert len(rep) == 1 and (rep["z_first"][0], rep["z_last"][0]) == (0, n - 1)
    # Matching noise here is ~0.5-0.8 px (slice pairs within a segment): the change ends up within it.
    before, after = rep["change_median_before_px"][0], rep["change_median_after_px"][0]
    assert before > 2 and after < min(1.0, before / 3)
    assert rep["change_p90_after_px"][0] < rep["change_p90_before_px"][0] / 2
    assert rep["seam_max_after_px"][0] < 1.5
    # The default (64) holds the seams harder: they move less, the mismatch improves less.
    held, _ = solve(local_half_width=n)
    assert held["seam_max_after_px"][0] < rep["seam_max_after_px"][0]
    assert after < held["change_median_after_px"][0] < before
    assert len(tiles) == 6 * 4 - 1 + 6 * 2           # z 3 has no tile 1-1

    rep, tiles = solve(local_half_width=3, local_seam_weight=4.0)   # z 3-8; z 0-2 and 9-11 keep their transforms
    assert (rep["z_first"][0], rep["z_last"][0]) == (3, 8)
    assert sorted(tiles["z"].unique()) == list(range(3, 9))
    assert rep["change_median_after_px"][0] < 0.5 * rep["change_median_before_px"][0]

    rep, tiles = solve(local_at_layout_changes=False, local_windows=[[1, 2]])
    assert rep["reason"].tolist() == ["configured z 1-2"] and sorted(tiles["z"].unique()) == [1, 2]
    rep, tiles = solve(local_at_layout_changes=False)
    assert rep.empty and tiles.empty


def test_elastic_correction_at_layout_change(tmp_path):
    """Same layout change (new tiles stretched 4% wider and narrower about their shared seam): the
    elastic field fitted across it removes the mismatch the frame fit leaves, and is applied to every
    slice of the earlier layout (or, with elastic_side after, of the later one)."""
    from pipeline import elastic
    n, seg, frame = 12, 6, np.array([37.0, -21.0])

    def stitch_fn(tile, origin, W):
        s = {"0-0": 1.04, "0-1": 0.96}[tile]
        return np.array([[s, 0, s * (origin[0] - W / 2) + W / 2 + frame[0]], [0, 1, origin[1] + frame[1]]])

    truth, out, _ = _layout_change(tmp_path, n, seg, frame, stitch_fn=stitch_fn)
    cfg = synth.write_config(tmp_path / "config.yaml", truth.raw_dir, out,
                             align={**ALIGN, "scale": 0.5, "elastic_min_px": 1.0, "elastic_grid_px": 60})
    align.main(["run", "--config", str(cfg)])
    assert align.main(["solve", "--config", str(cfg)]) == 0
    rep = pd.read_csv(out / "work" / "align" / "elastic_interfaces.csv")
    assert rep[["z_before", "z_after"]].values.tolist() == [[seg - 1, seg]]
    assert rep["median_before_px"][0] > 1.5 and rep["median_after_px"][0] < rep["median_before_px"][0] / 2
    data = json.loads((out / "work" / "align" / "elastic.json").read_text())
    assert {int(z): v[0][1] for z, v in data["weights"].items()} == {z: -1.0 for z in range(seg)}
    cfg = synth.write_config(tmp_path / "after.yaml", truth.raw_dir, out, align={
        **ALIGN, "scale": 0.5, "elastic_min_px": 1.0, "elastic_grid_px": 60, "elastic_side": "after"})
    assert align.main(["solve", "--config", str(cfg)]) == 0
    data = json.loads((out / "work" / "align" / "elastic.json").read_text())
    assert {int(z): v[0][1] for z, v in data["weights"].items()} == {z: 1.0 for z in range(seg, n)}
    # Off: no interfaces, an empty file (render then uses no field)
    cfg = synth.write_config(tmp_path / "off.yaml", truth.raw_dir, out, align={**ALIGN, "scale": 0.5, "elastic_min_px": None})
    assert align.main(["solve", "--config", str(cfg)]) == 0
    assert elastic.load(synth_cfg(cfg)) is None


def synth_cfg(path):
    from pipeline.config import load_config
    return load_config(path, step_defaults=align.DEFAULTS)


def test_elastic_field_fit_and_weights():
    from pipeline import elastic
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 20000, (5000, 2))

    def true(q):
        return np.column_stack([15 * np.sin(q[:, 0] / 6000), 8 * np.cos(q[:, 1] / 5000)])
    field = elastic.fit_field(p, true(p) + rng.normal(0, 1, (len(p), 2)), 1000, 0.3)
    q = rng.uniform(1000, 19000, (500, 2))
    assert np.abs(elastic.field_at(field, q) - true(q)).max() < 1.0
    run = np.r_[np.zeros(4, int), np.ones(6, int), np.full(5, 2)]     # layouts change at 4 and 10
    assert elastic.weights(run, 4, [4], "before") == {0: -1, 1: -1, 2: -1, 3: -1}
    assert elastic.weights(run, 4, [4], "after") == {i: 1 for i in range(4, 15)}   # across the change at 10
    assert elastic.weights(run, 10, [4, 7, 10], "before") == {7: -1, 8: -1, 9: -1}   # back to the previous interface
    assert elastic.weights(run, 10, [10], "before") == {i: -1 for i in range(10)}     # across the change at 4
    assert elastic.weights(run, 4, [4, 7], "after") == {4: 1, 5: 1, 6: 1}
    changes = pd.DataFrame({"z_before": [9, 19], "z_after": [10, 20], "median_px": [8.9, 2.5]})
    zs = np.arange(30)
    a = {"elastic_min_px": 5.0, "elastic_interfaces": [25]}
    assert [k for k, _ in elastic.interfaces(zs, None, changes, a)] == [10, 25]
    assert [k for k, _ in elastic.interfaces(zs, None, changes, {**a, "elastic_min_px": None})] == [25]


def test_local_windows_merge_and_clip():
    from pipeline import tilesolve
    zs = np.arange(100, 200)
    run = np.r_[np.zeros(30, int), np.ones(40, int), np.full(30, 2)]   # changes at z 130 and 170
    a = {"local_at_layout_changes": True, "local_half_width": 25, "local_windows": [[190, 250]]}
    w = tilesolve.windows(zs, run, a)
    assert [(zs[i0], zs[i1]) for i0, i1, _ in w] == [(105, 199)]    # 105-154, 145-194, 190-199 merged
    a = {**a, "local_half_width": 5, "local_windows": []}
    assert [(zs[i0], zs[i1], r) for i0, i1, r in tilesolve.windows(zs, run, a)] == [
        (125, 134, "layout change z 129->130"), (165, 174, "layout change z 169->170")]


def test_segment_frames_chain_to_reference():
    """Each run's frame maps into the reference run's (the largest), through the fits at every change."""
    rng = np.random.default_rng(0)
    zs = np.arange(10)
    run = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    true = [np.array([[1.002, 0.001, -40.0], [-0.001, 0.999, 15.0]]), transforms.identity(),
            np.array([[0.998, 0.0, 300.0], [0.0, 1.001, -7.0]])]   # run k frame -> run 1 frame
    pa, pb, pid, pairs = [], [], [], []
    for i, (za, zb) in enumerate([(2, 3), (1, 3), (7, 8), (6, 9)]):
        world = rng.uniform(0, 20000, (200, 2))
        pa.append(transforms.apply(transforms.invert(true[run[za]]), world))
        pb.append(transforms.apply(transforms.invert(true[run[zb]]), world))
        pid.append(np.full(200, i))
        pairs.append((za, zb))
    # Pairs must be sorted by (z_a, z_b) like _load_matches makes them.
    order = sorted(range(len(pairs)), key=lambda i: pairs[i])
    pid = [np.full(200, order.index(i)) for i in range(len(pairs))]
    m = {"pa": np.concatenate(pa), "pb": np.concatenate(pb), "pid": np.concatenate(pid),
         "pairs": np.array([pairs[i] for i in order])}
    a = {**align.DEFAULTS["align"]}
    frames, ch = align.segment_frames(m, run, np.array([4, 5, 7]), zs, a)
    for F, T in zip(frames, true):
        np.testing.assert_allclose(F, T, atol=1e-6)
    assert ch["segment_after"].tolist() == [5, 7] and ch["median_px"].max() < 1e-6
    frames, _ = align.segment_frames(m, run, np.array([4, 5, 7]), zs, {**a, "segment_reference": "first"})
    np.testing.assert_allclose(frames[0], transforms.identity(), atol=1e-9)
    np.testing.assert_allclose(transforms.compose(transforms.invert(true[0]), true[2]), frames[2], atol=1e-6)


def test_anchor_mean(drift_data, tmp_path):
    truth, _ = drift_data
    df, mats = read_transforms(solve_copy(drift_data, tmp_path, anchor="mean"))
    np.testing.assert_allclose(mats[:, :, 2].mean(0), 0, atol=1e-9)
    assert drift_error(truth.drift, df, mats) < 0.5
    _, affine = read_transforms(solve_copy(drift_data, tmp_path / "affine", anchor="mean", model="affine"))
    np.testing.assert_allclose(affine[:, :, 2].mean(0), 0, atol=1e-9)
    assert np.abs(affine[:, :, :2] - np.eye(2)).max() < 1e-3


def test_rejects_bad_options(drift_data, tmp_path):
    truth, out = drift_data
    cfg = synth.write_config(tmp_path / "config.yaml", truth.raw_dir, out, align={"ransac_model": "elastic"})
    with pytest.raises(ValueError, match="ransac_model"):
        align.main(["run", "--config", str(cfg)])


def test_chunk_is_current_checks_slice_times(tmp_path):
    """Chunks record their slices' acquisition times; a chunk whose z now hold other slices (a late
    day inserted before them) is matched again. Older chunks without times are judged as before."""
    path = tmp_path / "chunk.npz"
    sig, settings = ["5:0-0", "6:0-0"], {"scale": 0.25}
    stamps = ["5:2026-10-15T00:00:20", "6:2026-10-15T00:02:40"]
    np.savez(path, slices=np.array(sig), stamps=np.array(stamps), settings=json.dumps(settings))
    assert align._chunk_is_current(path, sig, settings, stamps)
    assert not align._chunk_is_current(path, sig, settings, ["5:2026-10-14T01:00:00", "6:2026-10-14T01:02:20"])
    np.savez(path, slices=np.array(sig), settings=json.dumps(settings))
    assert align._chunk_is_current(path, sig, settings, stamps)
