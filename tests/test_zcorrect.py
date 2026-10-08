"""zcorrect: recover non-uniform slice spacing from NCC between nearby slices."""

import re
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytest

import synth
from pipeline import transforms, zcorrect
from pipeline.cli import chunks
from pipeline.config import load_config
from pipeline.slices import StackCache, load_slices, voxel_size_nm

VOXEL_Z = 8.0  # nm; no check/files.csv in most tests, so zcorrect uses its 8 nm default


def write_inputs(truths, out, seams=(), excluded=(), align=None):
    """Ground-truth check/slices.csv, stitch/tiles.csv and align/transforms.csv.

    truths: one Synth, or a list of them, one per segment (z continues; each later segment starts
    with a seam). excluded: z or (z, tile). align: optional fixed transform applied after the
    drift translation (montage -> aligned).
    """
    truths = truths if isinstance(truths, list) else [truths]
    rows, tiles, mats, z0 = [], [], [], 0
    for seg, truth in enumerate(truths):
        th, tw = truth.tile_shape
        for name, zs in truth.files.items():
            tile = re.search(r"_tile(\d+-\d+)", name).group(1)
            r, c = map(int, tile.split("-"))
            for i, z in enumerate(zs):
                ts, g = truth.timestamps[z], z0 + z
                ex = g in excluded or (g, tile) in excluded
                rows.append(dict(z=g, timestamp=ts.isoformat(), tile=tile, tile_row=r, tile_col=c, file=name,
                                 index=i, height=th, width=tw, segment=seg,
                                 seam=g in seams or (seg > 0 and z == 0), excluded=ex,
                                 exclude_reason="test" if ex else "", label=synth.LABEL.format(ts=ts, r=r, c=c)))
        extra = transforms.identity() if align is None else align
        for z in range(len(truth.timestamps)):
            tiles += [({"z": z0 + z, "tile": t}, transforms.translation(*truth.tile_origin[t])) for t in truth.tiles]
            mats.append(({"z": z0 + z, "timestamp": truth.timestamps[z].isoformat()},
                         transforms.compose(extra, transforms.translation(*truth.drift[z]))))
        z0 += len(truth.timestamps)
    for step in ("check", "stitch", "align"):
        (out / step).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["z", "tile"]).to_csv(out / "check" / "slices.csv", index=False)
    seg_of = {r["z"]: r["segment"] for r in rows}
    frame = transforms.to_frame(tiles)
    frame.assign(segment=frame["z"].map(seg_of)).to_csv(out / "stitch" / "tiles.csv", index=False)
    transforms.to_frame(mats).to_csv(out / "align" / "transforms.csv", index=False)


def _cfg(path):
    return load_config(path, step_defaults=zcorrect.DEFAULTS)


def run_all(cfg, num_tasks=1):
    for i in range(num_tasks):
        assert zcorrect.main(["run", "--config", str(cfg), "--task-id", str(i), "--num-tasks", str(num_tasks)]) == 0
    assert zcorrect.main(["solve", "--config", str(cfg)]) == 0


def load_pairs(out):
    """Every (z_a, z_b, ncc) row of every chunk file, NaN included."""
    frames = []
    for p in sorted((out / "zcorrect" / "ncc").glob("chunk_*.npz")):
        with np.load(p) as f:
            frames.append(pd.DataFrame({k: f[k] for k in ("z_a", "z_b", "ncc")}))
    return pd.concat(frames)


def spacings(out):
    pos = pd.read_csv(out / "zcorrect" / "positions.csv")
    return pos["z"].to_numpy(), np.diff(pos["position_nm"].to_numpy()) / VOXEL_Z


def test_recovers_alternating_spacing(tmp_path, make_config):
    n, d_max = 60, 8
    true = np.array([0.6 if (i // 5) % 2 == 0 else 1.4 for i in range(n - 1)])
    truth = synth.make_dataset(tmp_path / "raw", n_slices=n, z_positions=np.concatenate([[0], np.cumsum(true)]))
    out = tmp_path / "out"
    write_inputs(truth, out)
    cfg = make_config(truth, zcorrect={"crop_px": 64, "chunk_slices": 16, "max_distance": d_max})
    run_all(cfg, num_tasks=3)

    # Chunks from all tasks hold every pair of selected slices up to max_distance apart, once.
    pairs = load_pairs(out)
    expected = {(a, b) for a in range(n) for b in range(a + 1, min(n, a + d_max + 1))}
    assert len(pairs) == len(expected) and set(zip(pairs["z_a"], pairs["z_b"])) == expected
    # The synthetic volume's NCC decays over a few slices, and faster for larger true distance.
    near = pairs[pairs["z_b"] - pairs["z_a"] == 1]
    dist = truth.z_positions[near["z_b"]] - truth.z_positions[near["z_a"]]
    assert near["ncc"][dist < 1].median() > near["ncc"][dist > 1].median() + 0.1
    assert pairs[pairs["z_b"] - pairs["z_a"] == 2]["ncc"].median() < 0.7

    zs, est = spacings(out)
    assert list(zs) == list(range(n))
    assert est.mean() == pytest.approx(1.0)   # total depth stays nominal; compare relative spacing
    true = true / true.mean()
    assert np.corrcoef(est, true)[0, 1] > 0.8
    assert np.abs(est - true).mean() < 0.15
    assert (out / "zcorrect" / "zcorrect.png").stat().st_size > 0

    # Existing chunks are skipped unless --overwrite.
    mtimes = {p: p.stat().st_mtime_ns for p in (out / "zcorrect" / "ncc").iterdir()}
    zcorrect.main(["run", "--config", str(cfg)])
    assert {p: p.stat().st_mtime_ns for p in (out / "zcorrect" / "ncc").iterdir()} == mtimes
    zcorrect.main(["run", "--config", str(cfg), "--overwrite"])
    assert all(p.stat().st_mtime_ns != t for p, t in mtimes.items())


def test_uniform_spacing_with_hole_and_seam(tmp_path, make_config):
    n, hole, seam = 48, 20, 33
    truth = synth.make_dataset(tmp_path / "raw", n_slices=n)
    out = tmp_path / "out"
    write_inputs(truth, out, seams={seam}, excluded={hole})
    cfg = make_config(truth, zcorrect={"crop_px": 64, "chunk_slices": 10, "max_distance": 4})
    run_all(cfg)

    pairs = load_pairs(out)
    assert hole not in set(pairs["z_a"]) | set(pairs["z_b"])
    # Neighbours are counted in the selected list: 16..19 pair with 21..24 across the hole.
    assert {(19, 21), (19, 24), (16, 21)} <= set(zip(pairs["z_a"], pairs["z_b"]))
    assert (16, 22) not in set(zip(pairs["z_a"], pairs["z_b"]))

    zs, est = spacings(out)
    assert hole not in zs and len(zs) == n - 1
    per_slice = est / np.diff(zs)               # the gap over the hole is two nominal slices
    i_seam = list(zs).index(seam) - 1
    assert per_slice[i_seam] == pytest.approx(1.0)  # seam: nominal spacing
    assert np.abs(per_slice - 1).mean() < 0.1
    assert np.abs(per_slice - 1).max() < 0.35
    assert est[list(zs).index(hole + 1) - 1] == pytest.approx(2.0, abs=0.4)


def test_rotated_aligned_frame(tmp_path, make_config):
    """Non-translation transforms take the warp path; crops still follow the same tissue."""
    n = 40
    true = np.array([0.6 if (i // 5) % 2 == 0 else 1.4 for i in range(n - 1)])
    truth = synth.make_dataset(tmp_path / "raw", n_slices=n, z_positions=np.concatenate([[0], np.cumsum(true)]))
    out = tmp_path / "out"
    theta = np.deg2rad(3)
    rot = np.array([[np.cos(theta), -np.sin(theta), 40.0], [np.sin(theta), np.cos(theta), -15.0]])
    write_inputs(truth, out, align=rot)
    cfg = make_config(truth, zcorrect={"crop_px": 64, "chunk_slices": 50})
    run_all(cfg)
    assert not load_pairs(out)["ncc"].isna().any()
    _, est = spacings(out)
    assert np.corrcoef(est, true)[0, 1] > 0.8
    assert np.abs(est - true / true.mean()).mean() < 0.2


def test_read_crop_warp_matches_reference(synth_2x2):
    """The warp path reads the same pixels as warping the whole tile (tile -> aligned)."""
    name = sorted(synth_2x2.files)[0]
    row = {"file": name, "index": 3, "height": synth_2x2.tile_shape[0], "width": synth_2x2.tile_shape[1]}
    cache = StackCache(synth_2x2.raw_dir)
    theta = np.deg2rad(5)
    A = np.array([[np.cos(theta), -np.sin(theta), 30.3], [np.sin(theta), np.cos(theta), 12.7]])
    x0, y0, size = 70, 60, 64
    v = zcorrect._read_crop(cache, row, A, x0, y0, size, 1)
    full = cv2.warpAffine(cache.read(row).astype(np.float32), A, (x0 + size, y0 + size), flags=cv2.INTER_LINEAR)
    ref = full[y0:y0 + size, x0:x0 + size].ravel()
    assert np.corrcoef(v, ref)[0, 1] > 0.999
    # Translation path: integer offsets read the block directly.
    v = zcorrect._read_crop(cache, row, transforms.translation(-20, -10), 5, 6, size, 1)
    ref = cache.read(row)[16:16 + size, 25:25 + size].astype(np.float32).ravel()
    np.testing.assert_allclose(v, (ref - ref.mean()) / np.linalg.norm(ref - ref.mean()), atol=1e-5)
    assert zcorrect._read_crop(cache, row, transforms.translation(-170, 0), 0, 0, size, 1) is None


def test_constrain_keeps_local_variation_only():
    i = np.arange(1000)
    fast = np.where((i // 5) % 2 == 0, 0.7, 1.3)
    s = fast * (1 + 0.3 * np.sin(2 * np.pi * i / 1000)) + np.where(i < 3, -2.0, 0.0)
    nominal = np.ones_like(s)
    nominal[600] = 2.0                        # hole in the selection
    s[600] *= 2
    brk = np.zeros(len(s), bool)
    brk[400] = True                           # seam
    out = zcorrect._constrain(s.copy(), nominal, brk, 0.25, 100)
    assert out[400] == nominal[400] and out[600] / nominal[600] > 0.5
    assert out[:3].min() >= 0.25             # minimum spacing
    assert out[:400].sum() == pytest.approx(400) and out[401:].sum() == pytest.approx(nominal[401:].sum())
    # The slow trend is gone, the 10-slice alternation remains.
    smooth = pd.Series(out / nominal).rolling(100, center=True).mean().dropna()
    assert np.abs(smooth - 1).max() < 0.08
    assert np.corrcoef(out[10:390], fast[10:390])[0, 1] > 0.9


def test_growing_selection_and_stale_chunks(tmp_path, make_config):
    """Regression: a chunk measured while its extension ran past the data end is re-measured once
    more slices exist, and solve reads exactly the chunks of the current chunking."""
    n, z_start = 48, 5
    truth = synth.make_dataset(tmp_path / "raw", n_slices=n)
    out = tmp_path / "out"
    write_inputs(truth, out)
    pd.DataFrame({"file": list(truth.files), "voxel_z_nm": 10.0}).to_csv(out / "check" / "files.csv", index=False)
    zc = {"crop_px": 64, "chunk_slices": 16, "max_distance": 4}

    # Data "so far" ends at z=22: chunk 5..20 can only pair 17..20 with 21..22.
    cfg = make_config(truth, zcorrect=zc, selection={"z_start": z_start, "z_end": 22})
    run_all(cfg)
    pos = pd.read_csv(out / "zcorrect" / "positions.csv")
    assert list(pos.columns) == ["z", "timestamp", "position_nm"]
    assert list(pos["timestamp"]) == [truth.timestamps[z].isoformat() for z in range(z_start, 23)]
    assert pos["position_nm"].iloc[0] == pytest.approx(z_start * 10.0)   # same origin as z * voxel_z

    cfg = make_config(truth, zcorrect=zc, selection={"z_start": z_start})
    run_all(cfg)
    zs = np.arange(z_start, n)
    c = _cfg(cfg)
    pairs = zcorrect._load_pairs(c, zs, c["zcorrect"])
    expected = {(a, b) for a in zs for b in zs if 0 < b - a <= 4}
    assert len(pairs) == len(expected) and set(zip(pairs["z_a"], pairs["z_b"])) == expected
    pos = pd.read_csv(out / "zcorrect" / "positions.csv")
    assert list(pos["z"]) == list(zs)
    assert np.abs(pos["position_nm"] / 10.0 - pos["z"]).max() < 1.0      # uniform spacing: ~z * voxel_z

    # A leftover chunk of an older chunking (here with bogus NCC) is ignored...
    bogus = out / "zcorrect" / "ncc" / "chunk_000021-000099.npz"
    a = np.arange(21, 40)
    np.savez(bogus, z_a=a, z_b=a + 1, ncc=np.full(len(a), 0.999), zs=np.arange(21, 44))
    assert zcorrect.main(["solve", "--config", str(cfg)]) == 0
    pd.testing.assert_frame_equal(pd.read_csv(out / "zcorrect" / "positions.csv"), pos)
    # ...a missing one is an error...
    (out / "zcorrect" / "ncc" / "chunk_000021-000037.npz").unlink()
    with pytest.raises(FileNotFoundError, match="chunk_000021-000037"):
        zcorrect.main(["solve", "--config", str(cfg)])
    # ...and changed crop settings re-measure every chunk without --overwrite.
    cfg = make_config(truth, zcorrect={**zc, "crop_px": 48}, selection={"z_start": z_start})
    run_all(cfg)
    for core, _ in chunks(list(zs), 16, 4):
        assert int(np.load(zcorrect._chunk_path(c, core))["crop_px"]) == 48


def test_segment_change_and_missing_tiles(tmp_path, make_config):
    """3x3 then 2x2 segments with different tile shapes; a tile missing at a segment's first z."""
    n1, n2 = 20, 20
    seg0 = synth.make_dataset(tmp_path / "raw", n_slices=n1, grid=(3, 3), tile_shape=(128, 144),
                              overlap=(16, 16), seed=1)
    seg1 = synth.make_dataset(tmp_path / "raw", n_slices=n2, start=datetime(2026, 10, 1, 10, 0, 0), seed=2)
    out = tmp_path / "out"
    write_inputs([seg0, seg1], out, excluded={(0, "1-1"), (n1 + 5, "0-1")})
    cfg = make_config(seg0, zcorrect={"crop_px": 64, "chunk_slices": 12, "max_distance": 4})
    run_all(cfg, num_tasks=2)

    # Crops: every tile of each segment, including the one excluded at the first z.
    c = _cfg(cfg)
    stitch, align = zcorrect._read_transforms(out)
    origins = zcorrect._crop_origins(load_slices(c), stitch, align, 64)
    assert sorted(origins[0]) == sorted(seg0.tiles) and sorted(origins[1]) == sorted(seg1.tiles)
    # Seg 0 crops are centred on the 128 x 144 tiles, seg 1 crops on the 192 x 224 tiles.
    x, y = seg0.tile_origin["1-1"]
    dx, dy = seg0.drift[1]
    assert origins[0]["1-1"] == (x + dx + 72 - 32, y + dy + 64 - 32)
    x, y = seg1.tile_origin["0-1"]
    dx, dy = seg1.drift[0]
    assert origins[1]["0-1"] == (x + dx + 112 - 32, y + dy + 96 - 32)

    # Pairs stay within a segment, all of them with a valid NCC.
    pairs = zcorrect._load_pairs(c, np.arange(n1 + n2), c["zcorrect"])
    expected = {(a, b) for a in range(n1 + n2) for b in range(a + 1, a + 5)
                if b < n1 + n2 and (a < n1) == (b < n1)}
    assert set(zip(pairs["z_a"], pairs["z_b"])) == expected and not pairs["ncc"].isna().any()

    zs, est = spacings(out)
    assert list(zs) == list(range(n1 + n2))
    assert est[n1 - 1] == pytest.approx(1.0)        # segment change: nominal spacing
    assert est[:n1 - 1].sum() == pytest.approx(n1 - 1) and np.abs(est - 1).mean() < 0.1


def test_voxel_z_from_files_csv(tmp_path):
    """The rule render shares: median of usable values over the used files, else 8 nm."""
    cfg = {"output_dir": str(tmp_path)}
    (tmp_path / "check").mkdir()
    path = tmp_path / "check" / "files.csv"

    def voxel_z(files):
        return voxel_size_nm(cfg, files)[0]
    assert voxel_z(["a.tif"]) == 8.0                     # no files.csv
    pd.DataFrame({"file": ["a.tif", "b.tif", "c.tif"], "voxel_z_nm": [9.0, 11.0, 50.0]}).to_csv(path, index=False)
    assert voxel_z(["a.tif", "b.tif"]) == 10.0
    pd.DataFrame({"file": ["a.tif", "b.tif"], "voxel_z_nm": [0.0, None]}).to_csv(path, index=False)
    assert voxel_z(["a.tif", "b.tif"]) == 8.0           # 0 nm would collapse the volume
    pd.DataFrame({"file": ["a.tif"]}).to_csv(path, index=False)
    assert voxel_z(["a.tif"]) == 8.0
