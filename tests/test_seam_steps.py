"""tools/seam_steps.py: tile ownership under elastic and affine warps, the clean-up, array split and merge.

Ownership is rendered from a synthetic 3 x 3 dataset (192 x 224 px tiles, 24/28 px overlaps) at 32 nm.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import ndimage

import synth
from pipeline import elastic, patchgrid, render, shading, transforms
from pipeline.config import deep_merge, load_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import seam_steps  # noqa: E402

F = 4   # ownership is at 32 nm
SMALL = {"blend": "cut", "slab": 8, "chunk": [8, 32, 32], "shard_xy": 64, "num_scales": 3,
         "clahe": {"enabled": False}, "threads": 2}


@pytest.fixture(scope="module")
def truth(tmp_path_factory):
    return synth.make_dataset(tmp_path_factory.mktemp("seam_steps") / "raw", n_slices=3, grid=(3, 3), seed=5)


def _setup(path, truth, align=None, fields=None, shapes=None, **render_settings):
    """Ground-truth stitch and align (align(z) -> montage -> aligned matrix), optional elastic.json
    {"fields", "weights"}, optional patchgrid shapes {(segment, tile): (sg, sc)} (enabled, R 36000),
    and an initialised render plan (render settings: SMALL with ``render_settings``); returns the
    loaded config."""
    out = path / "out"
    synth.write_slices(truth, out)
    for d in ("stitch", "align"):
        (out / "work" / d).mkdir(parents=True, exist_ok=True)
    zs = range(len(truth.timestamps))
    transforms.to_frame([({"z": z, "tile": t, "segment": 0}, transforms.translation(*truth.tile_origin[t]))
                         for z in zs for t in truth.tiles]).to_csv(out / "work" / "stitch" / "tiles.csv", index=False)
    align = align or (lambda z: transforms.translation(*truth.drift[z]))
    transforms.to_frame([({"z": z, "timestamp": truth.timestamps[z].isoformat()}, align(z)) for z in zs]).to_csv(
        out / "work" / "align" / "transforms.csv", index=False)
    if fields:
        (out / "work" / "align" / "elastic.json").write_text(json.dumps(fields))
    extra = {}
    if shapes:
        patchgrid.write_shapes(out / "work" / "patchgrid" / "shapes.npz", shapes, 36000.0, 4)
        extra["patchgrid"] = {"enabled": True}
    cfg = synth.write_config(path / "config.yaml", truth.raw_dir, out, render={**SMALL, **render_settings}, **extra)
    assert render.main(["init", "--config", str(cfg)]) == 0
    return load_config(cfg, step_defaults=deep_merge(render.DEFAULTS, shading.DEFAULTS))


def _field(g):
    """A 4 x 4 field grid (aligned px) covering the synthetic canvas; g[iy][ix] = (dx, dy)."""
    return {"x0": -100.0, "y0": -100.0, "spacing": 300.0, "g": g}


def test_ownership_follows_a_subpixel_elastic_shift_exactly(truth, tmp_path):
    """A constant elastic shift of (0.625, -0.375) px at 32 nm moves the ownership of the slice by
    whole pixels (nearest: content at x - 0.625 -> x - 1, y + 0.375 -> y), with no lines of
    other labels along the tile boundaries or the data's edge (bilinear codes put them there)."""
    plain = _setup(tmp_path / "plain", truth)
    warped = _setup(tmp_path / "warped", truth, fields={
        "fields": [_field([[[10.0, -6.0]] * 4] * 4)], "weights": {"1": [[0, 0.25]]}})   # (2.5, -1.5) full-res px
    a, tiles = seam_steps.ownership(plain, "synthetic.ome.zarr", 1)
    b, tiles_b = seam_steps.ownership(warped, "synthetic.ome.zarr", 1)
    assert tiles == tiles_b == truth.tiles
    assert set(range(1, len(tiles) + 1)) <= set(np.unique(a).tolist())
    np.testing.assert_array_equal(b[:, 1:], a[:, :-1])
    np.testing.assert_array_equal(seam_steps.ownership(warped, "synthetic.ome.zarr", 0)[0],
                                  seam_steps.ownership(plain, "synthetic.ome.zarr", 0)[0])   # no field at z 0


def test_every_label_is_a_tile_that_covers_the_pixel(truth, tmp_path):
    """Rotated alignment (affine warp of the tiles) plus a varying elastic field: each labelled pixel
    lies on its tile's footprint (to within 1.5 px at 32 nm) and every tile owns one solid region.
    The opening clean-up only trims a few staircase teeth where three tiles meet (bilinear codes
    leave ~500 stray px here, which it mostly drops)."""
    theta, centre = 0.01, np.array([300.0, 250.0])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def align(z):
        return np.hstack([R, (centre + truth.drift[z] - R @ centre)[:, None]])

    rng = np.random.default_rng(0)
    fields = {"fields": [_field(rng.uniform(-6, 6, (4, 4, 2)).tolist())], "weights": {"1": [[0, 1.0]]}}
    cfg = _setup(tmp_path, truth, align=align, fields=fields)
    lab, tiles = seam_steps.ownership(cfg, "synthetic.ome.zarr", 1)
    meta = json.loads((Path(cfg["output_dir"]) / "work" / "render" / "synthetic" / "render.json").read_text())
    ox, oy = meta["origin_xy"]
    H, W = lab.shape
    # Output pixel q holds the slice at q - D(q) (aligned px); D from the field as render evaluates it.
    xs, ys = ox + F * np.arange(W) + (F - 1) / 2, oy + F * np.arange(H) + (F - 1) / 2
    dx, dy = elastic.displacement(fields, 1, xs, ys)
    px, py = xs[None, :] - dx, ys[:, None] - dy
    th, tw = truth.tile_shape
    tol = 1.5 * F
    for k, t in enumerate(tiles, start=1):
        inv = transforms.invert(transforms.compose(align(1), transforms.translation(*truth.tile_origin[t])))
        u = inv[0, 0] * px + inv[0, 1] * py + inv[0, 2]
        v = inv[1, 0] * px + inv[1, 1] * py + inv[1, 2]
        on = (u > -tol) & (u < tw + tol) & (v > -tol) & (v < th + tol)
        mine = lab == k
        assert mine.sum() > 0.8 * (th - 24) * (tw - 28) / F ** 2 and ndimage.label(mine)[1] == 1
        assert not (mine & ~on).any(), f"{t}: {(mine & ~on).sum()} px labelled off its footprint"
    assert np.count_nonzero(seam_steps.clean(lab) != lab) < 0.002 * lab.size


def test_ownership_ignores_the_plans_blending_and_grey_levels(truth, tmp_path):
    """The ownership render blends cut and drops the plan's grey-level corrections whatever the plan
    says: a seam-blended plan with patchgrid shapes (35h's final render) gives the labels of a plain
    cut one. Blending would mix neighbouring codes along each seam; the shapes, u = R (1 - sg) + sc +
    sg * code, would turn the codes into other labels."""
    rng = np.random.default_rng(1)
    shapes = {(0, t): (rng.uniform(0.95, 1.05, (4, 4)), rng.uniform(-300, 300, (4, 4))) for t in truth.tiles}
    plain = _setup(tmp_path / "plain", truth)
    shaped = _setup(tmp_path / "shaped", truth, shapes=shapes, blend="seam")
    meta = json.loads((Path(shaped["output_dir"]) / "work" / "render" / "synthetic" / "render.json").read_text())
    assert meta["settings"]["blend"] == "seam" and len(meta["patchgrid"]) == 64
    for z in (0, 2):
        a, tiles = seam_steps.ownership(plain, "synthetic.ome.zarr", z)
        b, tiles_b = seam_steps.ownership(shaped, "synthetic.ome.zarr", z)
        assert tiles == tiles_b
        np.testing.assert_array_equal(b, a)


def test_clean_drops_interpolation_fringes():
    """Two tiles side by side with what bilinear codes leave: a 1 px line of label 2 between labels 1
    and 3 and a line of 1 along 3's bottom edge. The opening drops both (to 0) and keeps the rest."""
    lab = np.zeros((60, 80), np.uint8)
    lab[5:55, 5:40], lab[5:55, 41:75] = 1, 3
    lab[5:55, 40] = 2
    lab[54, 41:75] = 1
    out = seam_steps.clean(lab)
    expect = lab.copy()
    expect[5:55, 40] = 0
    expect[54, 41:75] = 0
    np.testing.assert_array_equal(out, expect)


def test_z_list(tmp_path):
    (tmp_path / "zs.txt").write_text("12\n3, 7\n\n3 40\n")
    assert seam_steps.z_list("@" + str(tmp_path / "zs.txt")) == [3, 7, 12, 40]
    assert seam_steps.z_list("5,1,5") == [1, 5]


def test_array_parts_merge_to_the_single_run(tmp_path, monkeypatch):
    """Array tasks (every n-th z each) write parts; --merge gives the CSV of one run over all z
    (rows in z order), deletes the parts, and refuses while a part is missing."""
    lab = np.zeros((300, 400), np.uint8)
    lab[:, :200], lab[:, 200:] = 1, 2
    monkeypatch.setattr(seam_steps, "ownership", lambda cfg, old, z, plan=None: (lab, ["0-0", "0-1"]))
    monkeypatch.setattr(seam_steps, "classes", lambda before: (before > 0, np.zeros(before.shape, bool)))
    monkeypatch.setattr(render, "_tile_plan", lambda cfg: None)
    zs = [2, 5, 9, 11, 20]
    for z in zs:   # before: tile 0-1 z / 2 grey levels brighter; after: flat
        before = np.where(lab == 2, 100 + z // 2, 100).astype(np.uint8)
        tifffile.imwrite(tmp_path / f"z{z}.tif", np.stack([before, np.full_like(before, 100)]))
    (tmp_path / "zs.txt").write_text("\n".join(map(str, zs[::-1])))
    cfg = synth.write_config(tmp_path / "config.yaml", tmp_path, tmp_path / "out")
    args = [str(cfg), "old.ome.zarr", "@" + str(tmp_path / "zs.txt"), f"new={tmp_path}"]

    seam_steps.main(args)
    single = pd.read_csv(tmp_path / "out" / "qc" / "seam_steps.csv")
    assert list(single.columns) == ["z", "seam", "class", "stretches", "before", "new"]
    assert single["z"].tolist() == zs and (single["seam"] == "0-0|0-1").all()
    assert single["before"].tolist() == [z // 2 for z in zs] and (single["new"] == 0).all()

    out = tmp_path / "parts" / "steps.csv"
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "2")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "1")
    # a part left over from another run with as many tasks (other z): never merged with this run's
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    seam_steps.main([str(cfg), "old.ome.zarr", "2,5,9", f"new={tmp_path}", "--out", str(out)])
    stale = next(out.parent.glob("steps.*.part0of2.csv"))
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    seam_steps.main([*args, "--out", str(out)])
    part1 = next(out.parent.glob("steps.*.part1of2.csv"))
    assert pd.read_csv(part1)["z"].tolist() == [5, 11] and part1.name.split(".")[1] != stale.name.split(".")[1]
    with pytest.raises(SystemExit, match="one array run"):
        seam_steps.main(["--merge", "--out", str(out), str(cfg)])
    stale.unlink()
    with pytest.raises(SystemExit, match=r"tasks \[0\]"):
        seam_steps.main(["--merge", "--out", str(out), str(cfg)])
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    seam_steps.main([*args, "--out", str(out)])
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    seam_steps.main(["--merge", "--out", str(out), str(cfg)])
    pd.testing.assert_frame_equal(pd.read_csv(out), single)
    assert sorted(p.name for p in out.parent.iterdir()) == ["steps.csv"]
