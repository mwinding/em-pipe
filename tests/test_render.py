"""Tests for pipeline.render against synthetic ground truth.

Earlier steps' outputs (check/slices.csv, stitch/tiles.csv, align/transforms.csv,
intensity/levels.csv) are written from the ground truth, so these tests don't depend on them.
"""

import json
import logging
import math
import re
import sys
import types
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

import pipeline
import synth
from pipeline import omezarr, render, transforms

LO, HI = 26000, 41000
GAIN = {"0-1": 1.2, "1-0": 0.85}
# Tests compare renders with exact expectations written for feathering; seam tests set blend: seam.
SMALL = {"blend": "feather", "slab": 8, "chunk": [8, 32, 32], "shard_xy": 64, "num_scales": 3,
         "clahe": {"enabled": False}, "threads": 2}
N = 20


@pytest.fixture(scope="module")
def truth(tmp_path_factory):
    return synth.make_dataset(tmp_path_factory.mktemp("render") / "raw", n_slices=N, tile_shape=(96, 112),
                              overlap=(16, 20), tile_gain=GAIN, seed=3)


def _setup(path, truth, *, exclude=(), levels=True, shift=None, align=None, voxel=None, **sections):
    """Write ground-truth inputs and a config under ``path``; returns (config path, output_dir).

    truth: a synthetic dataset, or a list of (dataset, (x, y)) written as consecutive segments
      (z continues across them), each placed in the aligned frame at its drift + (x, y).
    exclude: z, or (z, tile), to mark excluded in slices.csv.
    shift: {tile: d} makes that tile's levels render d (fraction of the range) too dark.
    align: f(dx, dy) -> montage -> aligned matrix for slice drift (dx, dy); default translation.
    voxel: (x, y, z) nm to record in check/files.csv (no files.csv if None).
    """
    out = path / "out"
    for d in ("check", "stitch", "align", "intensity", "zcorrect"):
        (out / "work" / d).mkdir(parents=True, exist_ok=True)
    align = align or transforms.translation
    parts = truth if isinstance(truth, list) else [(truth, (0, 0))]
    rows, stitch, aligned, z0 = [], [], [], 0
    for seg, (t, off) in enumerate(parts):
        th, tw = t.tile_shape
        for name, zs in t.files.items():
            r, c = map(int, re.search(r"tile(\d+)-(\d+)", name).groups())
            tile = f"{r}-{c}"
            for i, zl in enumerate(zs):
                z = z0 + zl
                ex = z in exclude or (z, tile) in exclude
                rows.append({"z": z, "timestamp": t.timestamps[zl].isoformat(), "tile": tile,
                             "tile_row": r, "tile_col": c, "file": name, "index": i, "height": th, "width": tw,
                             "segment": seg, "seam": zl == 0, "excluded": ex,
                             "exclude_reason": "test" if ex else "",
                             "label": synth.LABEL.format(ts=t.timestamps[zl], r=r, c=c)})
                stitch.append(({"z": z, "tile": tile, "segment": seg},
                               transforms.translation(*t.tile_origin[tile])))
        aligned += [({"z": z0 + zl, "timestamp": ts.isoformat()}, align(*(t.drift[zl] + off)))
                    for zl, ts in enumerate(t.timestamps)]
        z0 += len(t.timestamps)
    sl = pd.DataFrame(rows).sort_values(["z", "tile"])
    sl.to_csv(out / "work" / "check" / "slices.csv", index=False)
    transforms.to_frame(stitch).to_csv(out / "work" / "stitch" / "tiles.csv", index=False)
    transforms.to_frame(aligned).to_csv(out / "work" / "align" / "transforms.csv", index=False)
    if levels:
        span = np.array([GAIN.get(t, 1.0) * (HI - LO) for t in sl["tile"]])
        lo = LO + span * np.array([(shift or {}).get(t, 0.0) for t in sl["tile"]])
        sl[["z", "tile"]].assign(lo=lo, hi=lo + span).to_csv(out / "work" / "intensity" / "levels.csv", index=False)
    if voxel:
        pd.DataFrame({"file": sorted(sl["file"].unique()), "voxel_x_nm": voxel[0], "voxel_y_nm": voxel[1],
                      "voxel_z_nm": voxel[2]}).to_csv(out / "work" / "check" / "files.csv", index=False)
    cfg = synth.write_config(path / "config.yaml", parts[0][0].raw_dir, out,
                             render={**SMALL, **sections.pop("render", {})}, **sections)
    return cfg, out


def _render(cfg, *extra):
    assert render.main(["init", "--config", str(cfg), *extra]) == 0
    assert render.main(["run", "--config", str(cfg), *extra]) == 0


def _read(out):
    return omezarr.open_scale(out / "synthetic.ome.zarr", 0).read().result()


def _meta(out):
    return json.loads((out / "work" / "render" / "synthetic" / "render.json").read_text())


def _aligned(truth, zs):
    """Noise-free slices zs in the aligned frame, cropped to their union: ([0, 1] stack, coverage)."""
    H, W = truth.montage_shape
    d = truth.drift[list(zs)]
    o = d.min(axis=0)
    shape = (len(zs), H + d[:, 1].max() - o[1], W + d[:, 0].max() - o[0])
    img, cov = np.zeros(shape, np.float32), np.zeros(shape, bool)
    for i, z in enumerate(zs):
        dx, dy = truth.drift[z] - o
        img[i, dy:dy + H, dx:dx + W] = truth.montage(z)
        cov[i, dy:dy + H, dx:dx + W] = True
    return img, cov


def _corr(a, b):
    return np.corrcoef(np.ravel(a).astype(float), np.ravel(b).astype(float))[0, 1]


@pytest.fixture(scope="module")
def full(truth, tmp_path_factory):
    """Full-resolution render of all slices from ground-truth inputs, CLAHE off: (output_dir, s0)."""
    cfg, out = _setup(tmp_path_factory.mktemp("full"), truth)
    _render(cfg)
    return out, _read(out)


def test_render_matches_ground_truth(truth, full):
    out, vol = full
    img, cov = _aligned(truth, range(N))
    assert vol.shape == img.shape and vol.dtype == np.uint8
    assert not vol[~cov].any()                      # no data -> 0
    err = vol - 255.0 * img
    assert _corr(vol[cov], img[cov]) > 0.99
    assert np.abs(err[cov]).mean() < 3              # noise is 0.01 of the range = 2.5 levels
    # Per-tile levels undo the tile gains: no step across seams. Average the error over z in
    # montage coordinates, then per column (vertical seam) and per row (horizontal seam).
    H, W = truth.montage_shape
    mont = np.stack([err[z, dy:dy + H, dx:dx + W] for z, (dx, dy) in enumerate(truth.drift)])
    assert np.abs(mont.mean(axis=(0, 1))).max() < 1.5
    assert np.abs(mont.mean(axis=(0, 2))).max() < 1.5

    meta = _meta(out)
    assert meta["origin_xy"] == [0, 0] and meta["shape"] == list(vol.shape)
    assert meta["voxel_nm"] == [8.0, 8.0, 8.0] and meta["downsample"] == 1 and not meta["zcorrected"]
    assert meta["planes"] == [[[z, 1.0]] for z in range(N)]
    assert sorted(p.name for p in (out / "work" / "render" / "synthetic" / "done").iterdir()) == ["slab_000000", "slab_000001",
                                                                          "slab_000002"]
    root = out / "synthetic.ome.zarr"
    ome = json.loads((root / "zarr.json").read_text())["attributes"]["ome"]
    assert ome["version"] == "0.5"
    ds = ome["multiscales"][0]["datasets"]
    assert [d["path"] for d in ds] == ["s0", "s1", "s2"]
    assert ds[0]["coordinateTransformations"][0]["scale"] == [8.0, 8.0, 8.0]
    assert [a["name"] for a in ome["multiscales"][0]["axes"]] == ["z", "y", "x"]
    s0 = json.loads((root / "s0" / "zarr.json").read_text())
    assert s0["shape"] == list(vol.shape) and s0["data_type"] == "uint8"
    assert s0["chunk_grid"]["configuration"]["chunk_shape"] == [8, 64, 64]
    shard = s0["codecs"][0]
    assert shard["name"] == "sharding_indexed" and shard["configuration"]["chunk_shape"] == [8, 32, 32]
    assert "gzip" in [c["name"] for c in shard["configuration"]["codecs"]]
    s2 = json.loads((root / "s2" / "zarr.json").read_text())
    assert s2["shape"] == [math.ceil(v / 4) for v in vol.shape]


def test_segment_change_3x3_to_2x2(tmp_path):
    # An earlier 3x3 segment of smaller tiles, then a 2x2 one: tile IDs repeat across segments with
    # other shapes and origins, so every slice must use only its own tiles. Second frame offset (9, -4).
    raw = tmp_path / "raw"
    segs = [(synth.make_dataset(raw, n_slices=5, grid=(3, 3), tile_shape=(64, 72), overlap=(12, 14),
                                tile_gain=GAIN, start=datetime(2026, 9, 20, 10), seed=5), (0, 0)),
            (synth.make_dataset(raw, n_slices=5, tile_shape=(96, 112), overlap=(16, 20), tile_gain=GAIN,
                                start=datetime(2026, 9, 22, 10), seed=6), (9, -4))]
    cfg, out = _setup(tmp_path, segs)
    _render(cfg)
    vol, origin = _read(out), _meta(out)["origin_xy"]
    planes = [(t, zl, t.drift[zl] + off - origin) for t, off in segs for zl in range(len(t.timestamps))]
    assert len(vol) == len(planes)
    assert vol.shape[1:] == (max(d[1] + t.montage_shape[0] for t, _, d in planes),
                             max(d[0] + t.montage_shape[1] for t, _, d in planes))
    for k, (t, zl, (dx, dy)) in enumerate(planes):
        H, W = t.montage_shape
        exp, cov = np.zeros(vol.shape[1:]), np.zeros(vol.shape[1:], bool)
        exp[dy:dy + H, dx:dx + W], cov[dy:dy + H, dx:dx + W] = 255 * t.montage(zl), True
        assert not vol[k][~cov].any()
        assert _corr(vol[k][cov], exp[cov]) > 0.99 and np.abs(vol[k][cov] - exp[cov]).mean() < 3


def test_excluded_tile_leaves_only_its_own_area_empty(truth, tmp_path):
    # Tile 1-1 excluded at z=2 only: its overlaps still come from the other tiles, the rest is no data.
    zs = list(range(4))
    cfg, out = _setup(tmp_path, truth, exclude=[(2, "1-1")], selection={"z_end": zs[-1]})
    _render(cfg)
    vol = _read(out)
    img, cov = _aligned(truth, zs)
    (dx, dy), (th, tw) = truth.drift[2] - truth.drift[zs].min(axis=0), truth.tile_shape
    H, W = truth.montage_shape
    cov[2, dy + th:dy + H, dx + tw:dx + W] = False      # right of tile 1-0 and below tile 0-1
    assert not vol[~cov].any()
    assert _corr(vol[cov], img[cov]) > 0.99 and np.abs(vol[cov] - 255.0 * img[cov]).mean() < 3


@pytest.mark.parametrize("f", [2, 3])    # f = 3 also checks the direction of the block alignment
def test_downsample_matches_block_means(truth, full, tmp_path, f):
    # z=5 excluded: groups are by position in the selected list, e.g. plane 2 = mean(z4, z6) at f = 2.
    cfg, out = _setup(tmp_path, truth, exclude=(5,), voxel=(8.0, 8.0, 10.0), render={"downsample": f})
    _render(cfg)
    vol, meta = _read(out), _meta(out)
    zs = [z for z in range(N) if z != 5]
    groups = [zs[i:i + f] for i in range(0, len(zs), f)]
    assert meta["planes"] == [[[z, 1 / len(g)] for z in g] for g in groups]
    ox, oy = meta["origin_xy"]
    assert [ox, oy] == truth.drift[zs].min(axis=0).tolist()
    img, cov = _aligned(truth, zs)
    _, h, w = img.shape
    assert vol.shape == (len(groups), math.ceil(h / f), math.ceil(w / f))
    pad = ((0, 0), (0, -h % f), (0, -w % f))
    # The same noisy data at f=1 (full render, origin 0) on this canvas.
    f1 = np.pad(full[1][zs, oy:oy + h, ox:ox + w].astype(float), pad)
    img, cov = np.pad(img, pad), np.pad(cov, pad)
    for k, g in enumerate(groups):
        idx = [zs.index(z) for z in g]
        blocks = (len(g), vol.shape[1], f, vol.shape[2], f)
        c = cov[idx].reshape(blocks)
        full_b, none_b = c.all(axis=(2, 4)), ~c.any(axis=(2, 4))
        assert not vol[k][none_b.all(axis=0)].any()
        # Blocks every slice covers fully or not at all: the mean over the slices that have data.
        ok = (full_b | none_b).all(axis=0) & full_b.any(axis=0)
        exp = (f1[idx] * cov[idx]).reshape(blocks).sum(axis=(0, 2, 4)) / c.sum(axis=(0, 2, 4)).clip(1)
        assert np.abs(vol[k][ok] - exp[ok]).max() <= 1.0
        truth_k = 255 * img[idx].reshape(blocks).mean(axis=(0, 2, 4))
        inner = full_b.all(axis=0)
        assert _corr(vol[k][inner], truth_k[inner]) > 0.995
        assert np.abs(vol[k][inner] - truth_k[inner]).mean() < 1.5
    ds = json.loads((out / "synthetic.ome.zarr" / "zarr.json").read_text())
    scale = ds["attributes"]["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == [10.0 * f, 8.0 * f, 8.0 * f]    # (z, y, x) = voxel from files.csv x f


@pytest.mark.parametrize("bbox, pad", [([30, 20, 150, 131], 0), ([-10, 20, 150, 131], 10)])
def test_bbox_crop_equals_full_render(truth, full, tmp_path, bbox, pad):
    cfg, out = _setup(tmp_path, truth, render={"bbox": bbox})
    _render(cfg)
    vol = _read(out)
    assert vol.shape == (N, bbox[3] - bbox[1], bbox[2] - bbox[0])
    assert _meta(out)["origin_xy"] == bbox[:2]
    assert not vol[:, :, :pad].any()                # left of the data: no data
    np.testing.assert_array_equal(vol[:, :, pad:], full[1][:, bbox[1]:bbox[3], bbox[0] + pad:bbox[2]])


def test_feathering_ramps_across_overlap(truth, tmp_path):
    # Tile 0-1 renders 0.1 of the range (25.5 levels) too dark. Across its 20 px overlap with tile
    # 0-0 the error must ramp linearly: weights are distances to each tile's edge (< blend_px).
    zs = list(range(4))
    cfg, out = _setup(tmp_path, truth, selection={"z_end": zs[-1]}, shift={"0-1": 0.1})
    _render(cfg)
    vol = _read(out)
    img, _ = _aligned(truth, zs)
    tw = truth.tile_shape[1]
    x1, y1 = truth.tile_origin["0-1"][0], truth.tile_origin["1-0"][1]   # rows < y1: tiles 0-0, 0-1 only
    errs = []
    for i, (dx, dy) in enumerate(truth.drift[zs] - truth.drift[zs].min(axis=0)):
        win = (i, slice(dy, dy + y1), slice(dx + x1, dx + tw))
        errs.append(np.where(img[win] > 0.15, vol[win] - 255.0 * img[win], np.nan))   # no clipping at 0
    profile = np.nanmean(np.concatenate(errs), axis=0)
    j = np.arange(tw - x1)
    np.testing.assert_allclose(profile, -25.5 * (j + 0.5) / (tw - x1), atol=1.5)


def test_plane_sources():
    assert render.plane_sources([0, 1, 3, 4, 7], 2) == [[(0, .5), (1, .5)], [(3, .5), (4, .5)], [(7, 1.0)]]
    planes = render.plane_sources([10, 11, 12], 1, np.array([0.0, 12.0, 16.0]), 8.0)
    assert planes[0] == [(10, 1.0)] and planes[2] == [(12, 1.0)]
    assert planes[1] == [(10, pytest.approx(1 / 3)), (11, pytest.approx(2 / 3))]
    # Positions out of z order: planes follow position, not z.
    assert render.plane_sources([10, 11, 12], 1, np.array([0.0, 16.0, 8.0]), 8.0) == [
        [(10, 1.0)], [(12, 1.0)], [(11, 1.0)]]


def test_zcorrect_plane_mapping(truth, tmp_path, monkeypatch):
    calls = []
    tile = render._Renderer._tile
    monkeypatch.setattr(render._Renderer, "_tile", lambda self, t: calls.append(t["z"]) or tile(self, t))
    zs = list(range(12))
    cfg, out = _setup(tmp_path, truth, zcorrect={"enabled": True}, selection={"z_end": zs[-1]})
    # Enabled but zcorrect not run yet: one plane per slice.
    assert render.main(["init", "--config", str(cfg)]) == 0
    assert len(_meta(out)["planes"]) == len(zs) and not _meta(out)["zcorrected"]
    rng = np.random.default_rng(4)
    pos = 8 * np.concatenate([[3.0], 3 + np.cumsum(rng.uniform(0.4, 1.7, N - 1))])
    pd.DataFrame({"z": range(N), "position_nm": pos}).to_csv(out / "work" / "zcorrect" / "positions.csv", index=False)
    assert render.main(["init", "--config", str(cfg)]) == 1      # inputs changed: needs --overwrite
    _render(cfg, "--overwrite")
    meta, vol = _meta(out), _read(out)
    p = pos[zs]
    assert meta["zcorrected"] and len(meta["planes"]) == int((p[-1] - p[0]) // 8) + 1 == len(vol)
    img, cov = _aligned(truth, zs)
    for k, src in enumerate(meta["planes"]):
        z, w = np.array([s[0] for s in src]), np.array([s[1] for s in src])
        assert w.sum() == pytest.approx(1) and np.dot(w, p[z]) == pytest.approx(p[0] + 8 * k)
        assert len(z) == 1 or z[1] == z[0] + 1   # bracketing slices are neighbours in position
        exp = 255 * np.tensordot(w, img[z], 1)
        ok = cov[z].all(axis=0)
        assert _corr(vol[k][ok], exp[ok]) > 0.99 and np.abs(vol[k][ok] - exp[ok]).mean() < 3
    # Consecutive planes share slices: each is rendered once per slab, not once per plane.
    slab = SMALL["slab"]
    per_slab = [{z for src in meta["planes"][k:k + slab] for z, _ in src} for k in range(0, len(vol), slab)]
    assert sorted(calls) == sorted(z for zz in per_slab for z in zz for _ in range(4))


def test_tasks_markers_and_reinit(truth, full, tmp_path):
    ref = full[1]
    cfg, out = _setup(tmp_path, truth)
    args = ["--config", str(cfg)]
    assert render.main(["init", *args]) == 0
    assert render.main(["run", *args, "--task-id", "1", "--num-tasks", "2"]) == 0
    done = out / "work" / "render" / "synthetic" / "done"
    assert [p.name for p in done.iterdir()] == ["slab_000001"]
    assert render.main(["run", *args, "--task-id", "0", "--num-tasks", "2"]) == 0
    np.testing.assert_array_equal(_read(out), ref)

    # Finished slabs are skipped: blank slab 0 behind the marker's back, re-run, still blank.
    arr = omezarr.open_scale(out / "synthetic.ome.zarr", 0)
    arr[0:8].write(np.zeros((8, *ref.shape[1:]), np.uint8)).result()
    assert render.main(["run", *args]) == 0
    assert not _read(out)[0:8].any()
    (done / "slab_000000").unlink()
    assert render.main(["run", *args]) == 0
    np.testing.assert_array_equal(_read(out), ref)
    # A marker from another plan (e.g. a task that outlived init --overwrite) does not count.
    arr[8:16].write(np.zeros((8, *ref.shape[1:]), np.uint8)).result()
    (done / "slab_000001").write_text("digest of an older plan\n")
    assert render.main(["run", *args]) == 0
    np.testing.assert_array_equal(_read(out), ref)

    # init with unchanged inputs keeps the volume; changed settings need --overwrite.
    assert render.main(["init", *args]) == 0
    np.testing.assert_array_equal(_read(out), ref)
    cfg, _ = _setup(tmp_path, truth, render={"blend_px": 64})
    assert render.main(["init", *args]) == 1
    np.testing.assert_array_equal(_read(out), ref)
    assert render.main(["init", *args, "--overwrite"]) == 0
    assert not done.exists() and not _read(out).any()


def test_clahe_keeps_no_data_zero(truth, tmp_path):
    zs = list(range(8))
    cfg, out = _setup(tmp_path, truth, selection={"z_end": zs[-1]},
                      render={"clahe": {"enabled": True, "tile_px": 64}})
    _render(cfg)
    vol = _read(out)
    img, cov = _aligned(truth, zs)
    assert vol.shape == img.shape
    assert not vol[~cov].any()
    assert (vol[cov] > 0).mean() > 0.99
    assert _corr(vol[cov], img[cov]) > 0.8
    assert np.abs(vol[cov] - 255.0 * img[cov]).mean() > 3   # CLAHE changed the contrast


def test_clahe_ignores_no_data():
    # Next to no data, CLAHE must equalise as if the zeros were not there: the data's edge moves with
    # the drift, so any dependence on it flickers in z.
    rng = np.random.default_rng(0)
    img = ndimage.gaussian_filter(rng.standard_normal((256, 512)), 3)
    img = np.clip(128 + 40 * img / img.std(), 1, 255).astype(np.uint8)
    valid = np.ones(img.shape, bool)
    ref = render._clahe(img.copy(), valid, 2.0, 128)
    np.testing.assert_array_equal(ref, cv2.createCLAHE(2.0, (4, 2)).apply(img))   # grid is (x, y)
    valid[:, :200] = False
    out = render._clahe(np.where(valid, img, 0).astype(np.uint8), valid, 2.0, 128)   # no data is 0
    assert not out[~valid].any()
    edge = np.s_[:, 200:256]                        # rest of the CLAHE tile column holding the edge
    assert np.abs(out[edge].astype(float) - ref[edge]).mean() < 1.5    # zeros left in: ~10 levels


def test_destreak_applied_per_tile(truth, tmp_path, monkeypatch):
    calls = []

    def destreak_from_cfg(img, cfg):
        calls.append(img.shape)
        return img.astype(np.float32) - cfg["destreak"]["offset"]

    fake = types.ModuleType("pipeline.destreak")
    fake.DEFAULTS = {"destreak": {"enabled": False, "offset": 1500.0}}
    fake.destreak_from_cfg = destreak_from_cfg
    monkeypatch.setitem(sys.modules, "pipeline.destreak", fake)
    monkeypatch.setattr(pipeline, "destreak", fake, raising=False)
    zs = list(range(4))
    cfg, out = _setup(tmp_path, truth, selection={"z_end": zs[-1]}, destreak={"enabled": True})
    _render(cfg)
    assert _meta(out)["destreak"] == {"enabled": True, "offset": 1500.0}   # module defaults merged
    assert calls == [truth.tile_shape] * (4 * len(zs))
    vol = _read(out)
    img, cov = _aligned(truth, zs)
    ok = cov & (img > 0.2)                          # where the offset doesn't clip at 0
    # 1500 counts = 0.1 of the (gain-scaled) range: about -25 levels.
    assert -31 < np.median(vol[ok] - 255.0 * img[ok]) < -20


def test_missing_levels_fall_back_to_percentiles(truth, tmp_path, caplog):
    zs = list(range(4))
    cfg, out = _setup(tmp_path, truth, levels=False, selection={"z_end": zs[-1]})
    with caplog.at_level(logging.WARNING, logger="pipeline.render"):
        _render(cfg)
    assert "p0.5/p99.5" in caplog.text
    vol = _read(out)
    img, cov = _aligned(truth, zs)
    assert not vol[~cov].any() and _corr(vol[cov], img[cov]) > 0.97


@pytest.mark.parametrize("integer_shifts", [True, False])
def test_half_pixel_shifts(truth, tmp_path, integer_shifts):
    zs = list(range(4))
    cfg, out = _setup(tmp_path, truth, selection={"z_end": zs[-1]}, render={"integer_shifts": integer_shifts},
                      align=lambda dx, dy: transforms.translation(dx + 0.5, dy + 0.5))
    _render(cfg)
    vol, meta = _read(out), _meta(out)
    img, cov = _aligned(truth, zs)
    o = truth.drift[zs].min(axis=0)
    if integer_shifts:
        # Every x.5 rounds up the same way, so slices stay registered and nothing is interpolated.
        assert meta["origin_xy"] == (o + 1).tolist() and vol.shape == img.shape
    else:
        assert meta["origin_xy"] == o.tolist() and vol.shape == (len(zs), img.shape[1] + 1, img.shape[2] + 1)
        pad = ((0, 0), (0, 1), (0, 1))
        img = ndimage.shift(np.pad(img, pad), (0, 0.5, 0.5), order=1)
        cov = ndimage.binary_erosion(np.pad(cov, pad), np.ones((1, 3, 3)), iterations=2)
    assert _corr(vol[cov], img[cov]) > 0.99
    assert np.abs(vol[cov] - 255.0 * img[cov]).mean() < 3


def test_fractional_shift_far_from_canvas_origin(truth, tmp_path, caplog):
    # integer_shifts off, slice 1 moved by +0.1 px or by +15000.1 px: it must be interpolated the same
    # way in both. (A relative tolerance once took 15000.1 as whole there and pasted it rounded.)
    planes = []
    for far in (0, 15000):
        cfg, out = _setup(tmp_path / str(far), truth, selection={"z_end": 1}, render={"integer_shifts": False})
        tf = pd.read_csv(out / "work" / "align" / "transforms.csv")
        tf.loc[tf["z"] == 1, "tx"] += far + 0.1
        tf.to_csv(out / "work" / "align" / "transforms.csv", index=False)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="pipeline.render"):
            _render(cfg)
        assert ("check align/transforms.csv for outliers" in caplog.text) == bool(far)   # stray slice
        x = int(truth.drift[1][0]) + far - _meta(out)["origin_xy"][0]
        planes.append(_read(out)[1, :, x:x + truth.montage_shape[1] + 1])
    near, far = planes
    assert near.any()
    np.testing.assert_array_equal(near, far)


@pytest.mark.parametrize("f", [1, 2])
def test_rotated_alignment_uses_warp(truth, tmp_path, f):
    theta, centre = 0.02, np.array([100.0, 90.0])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def align(dx, dy):
        return np.hstack([R, (centre + (dx, dy) - R @ centre)[:, None]])

    zs = list(range(4))
    cfg, out = _setup(tmp_path, truth, selection={"z_end": zs[-1]}, render={"downsample": f}, align=align)
    _render(cfg)
    vol, meta = _read(out), _meta(out)
    ox, oy = meta["origin_xy"]
    shape = (vol.shape[1] * f, vol.shape[2] * f)
    exp, cov = [], []
    for z in zs:
        # canvas px -> montage px, as scipy's (row, col) matrix and offset
        inv = transforms.compose(transforms.invert(align(*truth.drift[z])), transforms.translation(ox, oy))
        mat, off = inv[::-1, 1::-1], inv[::-1, 2]
        exp.append(ndimage.affine_transform(truth.montage(z), mat, off, output_shape=shape, order=1))
        cov.append(ndimage.affine_transform(np.ones(truth.montage_shape), mat, off, output_shape=shape,
                                            order=0, cval=0) > 0)
    exp, cov = 255 * np.array(exp), ndimage.binary_erosion(np.array(cov), np.ones((1, 3, 3)), iterations=2)
    blocks = (len(vol), f, vol.shape[1], f, vol.shape[2], f)    # plane k = mean of f slices and f x f px
    exp, cov = exp.reshape(blocks).mean(axis=(1, 3, 5)), cov.reshape(blocks).all(axis=(1, 3, 5))
    assert _corr(vol[cov], exp[cov]) > 0.99
    assert np.abs(vol[cov] - exp[cov]).mean() < (3 if f == 1 else 2)


def test_two_volumes_share_output_dir(truth, full, tmp_path):
    """A full-resolution region next to an overview in one output_dir: each volume keeps its own
    plan and done markers (they used to share render/render.json and render/done/, so the second
    volume's init was refused and its run silently skipped every slab)."""
    import yaml
    cfg, out = _setup(tmp_path, truth, render={"downsample": 2})
    _render(cfg)
    roi = yaml.safe_load(cfg.read_text())
    roi["render"] = {**roi["render"], "downsample": 1, "name": "roi.ome.zarr", "bbox": [10, 10, 60, 50]}
    cfg_roi = tmp_path / "roi.yaml"
    cfg_roi.write_text(yaml.safe_dump(roi))
    _render(cfg_roi)

    small = omezarr.open_scale(out / "roi.ome.zarr", 0).read().result()
    full_out, full_vol = full
    ox, oy = _meta(full_out)["origin_xy"]
    x0, y0 = 10 - int(ox), 10 - int(oy)
    assert small.shape == (N, 40, 50)
    assert np.abs(small.astype(int) - full_vol[:, y0:y0 + 40, x0:x0 + 50]).max() <= 1
    assert json.loads((out / "work" / "render" / "roi" / "render.json").read_text())["downsample"] == 1
    assert _meta(out)["downsample"] == 2 and _read(out).shape[1] < full_vol.shape[1]
    assert render.main(["init", "--config", str(cfg)]) == 0   # overview plan untouched: kept


def test_seam_switches_tiles_along_the_middle_of_the_overlap(truth, tmp_path):
    """Default blend: across the 20 px overlap of tiles 0-0 and 0-1, pixels come from 0-0 up to the
    middle and from 0-1 after it, switching over ~seam_px (no 50/50 overlay of the two tiles)."""
    zs = list(range(4))
    cfg, out = _setup(tmp_path, truth, selection={"z_end": zs[-1]}, shift={"0-1": 0.1},
                      render={"blend": "seam", "seam_px": 4})
    _render(cfg)
    vol = _read(out)
    img, _ = _aligned(truth, zs)
    tw = truth.tile_shape[1]
    x1, y1 = truth.tile_origin["0-1"][0], truth.tile_origin["1-0"][1]
    errs = []
    for i, (dx, dy) in enumerate(truth.drift[zs] - truth.drift[zs].min(axis=0)):
        win = (i, slice(dy, dy + y1), slice(dx + x1, dx + tw))
        errs.append(np.where(img[win] > 0.15, vol[win] - 255.0 * img[win], np.nan))
    profile = np.nanmean(np.concatenate(errs), axis=0)      # 0 where 0-0 wins, -25.5 where 0-1 wins
    n = tw - x1
    assert np.abs(profile[: n // 2 - 3]).max() < 1.5
    assert np.abs(profile[n // 2 + 3:] + 25.5).max() < 1.5


def test_seam_weights_switch_at_equal_edge_distance():
    """Normalised over two tiles, the weight crosses 0.5 where both edges are equally far and
    takes ~seam_px to switch, also for overlaps far wider than the switch."""
    for overlap in (40, 600):
        x = np.arange(overlap) + 0.5                      # positions across the overlap
        da, db = overlap - x, x                           # distance to tile A's / B's own edge
        far = np.full(1, 5000.0)                          # far from the other two edges
        wa = render._seam_weight(da, far, 32)[:, 0]
        wb = render._seam_weight(db, far, 32)[:, 0]
        share = wa / (wa + wb)
        mid = overlap / 2
        assert abs(np.interp(0.5, share[::-1], x[::-1]) - mid) < 1
        assert (share[x < mid - 32] > 0.98).all() and (share[x > mid + 32] < 0.02).all()
