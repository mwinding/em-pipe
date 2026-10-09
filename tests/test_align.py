"""Tests for pipeline.align against synthetic ground truth (drift known per slice)."""

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


def test_segment_change(tmp_path):
    """From z 6 the slices are re-acquired as a 1×2 grid of tall tiles (other ids, other shape) whose
    montage frame is offset from the first segment's; every tile combination is matched across the change.

    Features are detected at half resolution, so points must be mapped back to full-res pixels.
    """
    n, seg, frame = 12, 6, np.array([37.0, -21.0])
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
        for i, z in enumerate(zs):
            rows.append({"z": z, "timestamp": truth.timestamps[z].isoformat(), "tile": tile, "tile_row": r,
                         "tile_col": c, "file": name, "index": i, "height": H, "width": w, "segment": 1,
                         "seam": z == seg, "excluded": False, "exclude_reason": "", "label": ""})
            stitch.append(({"z": z, "tile": tile, "segment": 1}, transforms.translation(*(np.add((x0, y0), frame)))))
    sl = pd.read_csv(out / "work" / "check" / "slices.csv")
    pd.concat([sl[sl["z"] < seg], pd.DataFrame(rows)]).to_csv(out / "work" / "check" / "slices.csv", index=False)
    tl = pd.read_csv(out / "work" / "stitch" / "tiles.csv", dtype={"tile": str})
    pd.concat([tl[tl["z"] < seg], transforms.to_frame(stitch)]).to_csv(out / "work" / "stitch" / "tiles.csv", index=False)

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
    align.main(["solve", "--config", str(cfg)])
    df, mats = read_transforms(out)
    assert drift_error(expected, df, mats) < 0.5
    assert drift_error(truth.drift, df, mats) > 10   # the frame offset is really in the result


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
