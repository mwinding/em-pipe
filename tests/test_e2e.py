"""End to end: every step through its command line on synthetic data, checked against ground truth.

Each step runs as ``python -m pipeline.<step>`` in a subprocess, in run_pipeline.sh's order;
array steps run as two concurrent tasks (--task-id/--num-tasks) to exercise the chunking.
Earlier steps' outputs are never written by the test: each step reads what the previous one made.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import ndimage

import synth
from pipeline import destreak, omezarr, pyramid, transforms

ROOT = Path(__file__).resolve().parents[1]
N = 30
GAP, PART, JUMP = 8, 22, (7, -5)   # time gap before z 8; restart (_part2) with a drift jump at z 22
GAIN = {"0-1": 1.15, "1-0": 0.88}
COPY = "M09_D24_tile1-1.tif"       # a byte copy of tile 1-0, like M09_D28_tile0-1.tif in the real data
LO, HI = 26000, 41000              # synth's raw value range
# Settings for tiny tiles; everything else is the step defaults.
SMALL = {
    "check": {"min_age_minutes": 0, "voxel_size_nm": [8, 8, 8]},
    "preview": {"factor": 4, "chunk_slices": 8},
    "stitch": {"sample_every": 5, "coarse_factor": 2, "fine_factor": 1, "fine_margin_px": 16},
    "align": {"scale": 1.0, "chunk_slices": 10, "max_points_per_pair": 40},
    "intensity": {"smooth_slices": 9, "balance_every": 3},
    "render": {"slab": 8, "chunk": [8, 32, 32], "shard_xy": 64, "num_scales": 3,
               "clahe": {"enabled": False}, "threads": 2},
}


def run_step(cfg, step, *args, tasks=0):
    """``python -m pipeline.<step> args --config cfg``; with ``tasks``, that many array tasks at once."""
    cmd = [sys.executable, "-m", f"pipeline.{step}", *args, "--config", str(cfg)]
    cmds = [cmd + ["--task-id", str(i), "--num-tasks", str(tasks)] for i in range(tasks)] if tasks else [cmd]
    procs = [subprocess.Popen(c, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for c in cmds]
    for c, p in zip(cmds, procs):
        out = p.communicate()[0]
        assert p.returncode == 0, f"{' '.join(c[2:])} exited with {p.returncode}:\n{out[-5000:]}"


def run_pipeline(cfg, zcorrect=False):
    run_step(cfg, "check")
    run_step(cfg, "preview", "run", tasks=2)
    run_step(cfg, "preview", "merge")
    run_step(cfg, "stitch", "run", tasks=2)
    run_step(cfg, "stitch", "merge")
    run_step(cfg, "align", "run", tasks=2)
    run_step(cfg, "align", "solve")
    run_step(cfg, "intensity")
    if zcorrect:
        run_step(cfg, "zcorrect", "run", tasks=2)
        run_step(cfg, "zcorrect", "solve")
    run_step(cfg, "render", "init")
    run_step(cfg, "render", "run", tasks=2)
    for s in range(1, SMALL["render"]["num_scales"]):
        run_step(cfg, "pyramid", "run", "--scale", str(s), tasks=2)


def read_scales(out):
    root = out / "render" / "volume.ome.zarr"
    return [omezarr.open_scale(root, s).read().result() for s in range(SMALL["render"]["num_scales"])]


def masks(truth, meta, z):
    """Where slice z lands on the render canvas: (montage mask, {tile: footprint mask}, montage corner).

    Montage pixel p of slice z lands at aligned p + drift[z] - drift[0]: stitch puts the montage
    min corner (tile 0-0) at the origin and align anchors the first selected slice, z 0.
    """
    (ox, oy), (H, W) = meta["origin_xy"], meta["shape"][1:]
    x0, y0 = truth.drift[z] - truth.drift[0] - (ox, oy)
    (th, tw), (mh, mw) = truth.tile_shape, truth.montage_shape
    montage = np.zeros((H, W), bool)
    montage[y0:y0 + mh, x0:x0 + mw] = True
    feet = {}
    for t in truth.tiles:
        tx, ty = truth.tile_origin[t]
        feet[t] = np.zeros((H, W), bool)
        feet[t][y0 + ty:y0 + ty + th, x0 + tx:x0 + tx + tw] = True
    return montage, feet, (x0, y0)


def expected(truth, z, shape, corner):
    """Noise-free slice z on the render canvas (NaN outside the montage)."""
    x0, y0 = corner
    (mh, mw) = truth.montage_shape
    img = np.full(shape, np.nan)
    img[y0:y0 + mh, x0:x0 + mw] = truth.montage(z)
    return img


def interior(mask, px=3):
    """``mask`` without ``px`` pixels at its edges (render blends and pads there)."""
    return ndimage.binary_erosion(mask, iterations=px)


# ----- main run: 2x2 grid, gap, restart with jump, tile gains, a copied tile file -------------

@pytest.fixture(scope="module")
def main_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    # 23:10 start: the run crosses midnight, so day 2 holds a _part1 and the restart's _part2.
    truth = synth.make_dataset(root / "raw", n_slices=N, tile_shape=(176, 208), overlap=(40, 48),
                               start=datetime(2026, 9, 24, 23, 10), gaps={GAP: 900}, parts=[PART],
                               drift_jumps={PART: JUMP}, tile_gain=GAIN, z_positions=np.arange(N) * 0.25,
                               faults={"duplicate": [("M09_D24", "1-1", "1-0")]})
    known = [{"file": COPY, "action": "exclude", "note": "byte copy of tile1-0"}]
    cfg = synth.write_config(root / "config.yaml", truth.raw_dir, root / "out",
                             **{**SMALL, "check": {**SMALL["check"], "known_issues": known}})
    run_pipeline(cfg)
    return truth, root / "out"


def test_check_acknowledges_the_copy(main_run):
    truth, out = main_run
    assert sorted(truth.files) == sorted(pd.read_csv(out / "check" / "files.csv")["file"])
    assert {"M09_D25_tile0-0_part1.tif", "M09_D25_tile0-0_part2.tif"} <= set(truth.files)
    issues = pd.read_csv(out / "check" / "issues.csv")
    errors = issues[issues["severity"] == "ERROR"]
    assert len(errors) and (errors["file"] == COPY).all() and errors["known"].all()
    assert "DUPLICATE_CONTENT" in set(errors["code"])
    files = pd.read_csv(out / "check" / "files.csv").set_index("file")
    assert files.loc[COPY, "excluded"] and files.loc[COPY, "exclude_reason"] == "byte copy of tile1-0"
    assert files["excluded"].sum() == 1

    sl = pd.read_csv(out / "check" / "slices.csv", dtype={"tile": str})
    per_z = sl.drop_duplicates("z")
    assert list(per_z["z"]) == list(range(N))
    assert list(per_z["timestamp"]) == [t.isoformat() for t in truth.timestamps]
    day1 = [z for z, t in enumerate(truth.timestamps) if t.day == 24]
    assert sorted(sl.loc[sl["excluded"], "z"]) == day1 and set(sl.loc[sl["excluded"], "tile"]) == {"1-1"}
    assert sorted(sl.loc[sl["seam"], "z"].unique()) == [GAP, PART]
    assert (sl["segment"] == 0).all()


def test_stitch_matches_true_tile_offsets(main_run):
    truth, out = main_run
    tiles = pd.read_csv(out / "stitch" / "tiles.csv", dtype={"tile": str})
    sl = pd.read_csv(out / "check" / "slices.csv", dtype={"tile": str})
    assert len(tiles) == (~sl["excluded"]).sum()
    # The default affine_rigid model fits a near-identity affine to these pure-translation tiles:
    # every tile corner must land within 0.5 px of the truth.
    th, tw = truth.tile_shape
    corners = transforms.corners(tw, th)
    for row in tiles.itertuples():
        got = transforms.apply(transforms.from_row(row._asdict()), corners)
        assert np.abs(got - (corners + truth.tile_origin[row.tile])).max() < 0.5, (row.z, row.tile)
    layout = json.loads((out / "stitch" / "layout.json").read_text())["segments"]["0"]
    assert layout["grid_shape"] == [2, 2] and layout["row_axis"] == "y"


def test_align_recovers_drift(main_run):
    truth, out = main_run
    al = pd.read_csv(out / "align" / "transforms.csv")
    zs = al["z"].to_numpy()
    assert list(zs) == list(range(N))
    t = al[["tx", "ty"]].to_numpy()
    assert np.abs((t - t[0]) - (truth.drift[zs] - truth.drift[zs[0]])).max() < 0.5
    step = (t[PART] - t[PART - 1]) - (truth.drift[PART] - truth.drift[PART - 1])
    assert np.abs(step).max() < 0.5   # the restart jump is kept, not smoothed


def test_render_matches_truth_without_tile_seams(main_run):
    truth, out = main_run
    meta = json.loads((out / "render" / "volume" / "render.json").read_text())
    s0 = read_scales(out)[0]
    assert s0.shape == tuple(meta["shape"]) and s0.shape[0] == N
    sl = pd.read_csv(out / "check" / "slices.csv", dtype={"tile": str})
    present = sl[~sl["excluded"]].groupby("z")["tile"].agg(set)
    for k, plane in enumerate(meta["planes"]):
        (z, w), = plane
        assert (z, w) == (k, 1.0)
        img = s0[k].astype(float)
        montage, feet, corner = masks(truth, meta, z)
        exp = expected(truth, z, img.shape, corner)
        covered = np.any([feet[t] for t in present[z]], axis=0)
        # No data outside the tiles that exist at z (e.g. tile 1-1 on day 1), beyond the 1 px edge a
        # sub-pixel affine stitch interpolates into ...
        assert (img[~ndimage.binary_dilation(covered, np.ones((3, 3)))] == 0).all(), f"z {z}: data outside the present tiles"
        # ... and the truth inside them, overlaps included.
        inner = interior(montage) & covered
        assert np.corrcoef(img[inner], exp[inner])[0, 1] > 0.98, f"z {z}"
        # Same brightness in every tile (tile gains differ by up to 15 % in the raw data): fit
        # render = a * truth + b on each tile's own (non-overlap) area.
        fits = {}
        for t in present[z]:
            own = inner & feet[t] & ~np.any([feet[u] for u in present[z] - {t}], axis=0)
            fits[t] = np.polyfit(exp[own], img[own], 1)
        mid = np.median(exp[inner])
        level = {t: np.polyval(f, mid) for t, f in fits.items()}
        slope = {t: f[0] for t, f in fits.items()}
        # Measured: < 2.6 grey levels and < 1.03 balanced; 12 and 1.10 with intensity.balance_tiles off.
        assert max(level.values()) - min(level.values()) < 4, f"z {z}: brightness steps {level}"
        assert max(slope.values()) / min(slope.values()) < 1.05, f"z {z}: contrast steps {slope}"


def test_missing_tile_area_filled_by_neighbours(main_run):
    truth, out = main_run
    meta = json.loads((out / "render" / "volume" / "render.json").read_text())
    s0 = read_scales(out)[0]
    for z, t in enumerate(truth.timestamps):
        if t.day != 24:   # tile 1-1 is the excluded copy on day 1 only
            continue
        _, feet, corner = masks(truth, meta, z)
        neighbours = feet["0-1"] | feet["1-0"]
        assert (s0[z][feet["1-1"] & ~ndimage.binary_dilation(neighbours, np.ones((3, 3)))] == 0).all()
        # Where the neighbours overlap tile 1-1 they show the truth (dark pixels clip to 0 anywhere).
        filled = interior(feet["1-1"] & neighbours, 1)
        exp = expected(truth, z, s0[z].shape, corner)
        assert np.corrcoef(s0[z][filled], exp[filled])[0, 1] > 0.98
        assert (s0[z][filled] > 0).mean() > 0.95


def test_pyramid_scales_are_means_of_s0(main_run):
    truth, out = main_run
    scales = read_scales(out)
    for s in range(1, len(scales)):
        np.testing.assert_array_equal(scales[s], pyramid.downsample(scales[s - 1]))
        n_shards = len(omezarr.shard_boxes(scales[s].shape, omezarr.shard_shape(
            omezarr.open_scale(out / "render" / "volume.ome.zarr", s))))
        assert len(list((out / "render" / "volume" / "done").glob(f"s{s}_*"))) == n_shards


# ----- second run: destreak and zcorrect inside render -----------------------------------------

@pytest.fixture(scope="module")
def corrected_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e_corrected")
    n = 16
    # Larger tiles: the wavelet-FFT filter needs a few hundred px to tell stripes from structure.
    truth = synth.make_dataset(root / "raw", n_slices=n, tile_shape=(256, 288), overlap=(48, 56),
                               start=datetime(2026, 9, 24, 12), streaks=0.03, tile_gain={"0-1": 1.15},
                               z_positions=np.arange(n) * 0.25, seed=5)
    cfg = synth.write_config(root / "config.yaml", truth.raw_dir, root / "out", **SMALL,
                             destreak={"enabled": True},
                             zcorrect={"enabled": True, "chunk_slices": 8, "max_distance": 4, "crop_px": 128})
    run_pipeline(cfg, zcorrect=True)
    return truth, root / "out"


def test_zcorrect_positions_drive_the_planes(corrected_run):
    truth, out = corrected_run
    n = len(truth.timestamps)
    pos = pd.read_csv(out / "zcorrect" / "positions.csv")
    assert list(pos["z"]) == list(range(n))
    p = pos["position_nm"].to_numpy()
    assert p[0] == 0 and (np.diff(p) > 0).all()
    assert p[-1] == pytest.approx(8.0 * (n - 1), rel=0.02)   # mean spacing is held at nominal
    meta = json.loads((out / "render" / "volume" / "render.json").read_text())
    assert meta["zcorrected"] and meta["voxel_nm"] == [8.0, 8.0, 8.0]
    assert len(meta["planes"]) == int(np.floor((p[-1] - p[0]) / 8.0 + 1e-6)) + 1
    for k, plane in enumerate(meta["planes"]):
        q = p[0] + 8.0 * k   # plane k sits at q and interpolates the slices that bracket it
        zs = [z for z, _ in plane]
        assert p[min(zs)] <= q + 1e-6 and (len(zs) == 1 or p[max(zs)] >= q)
        assert sum(w for _, w in plane) == pytest.approx(1.0)


def test_destreak_removes_stripes_in_render(corrected_run):
    truth, out = corrected_run
    meta = json.loads((out / "render" / "volume" / "render.json").read_text())
    assert meta["destreak"]["enabled"]
    s0 = read_scales(out)[0]
    raw = tifffile.imread(truth.raw_dir / "M09_D24_tile0-0.tif").astype(float)
    # Tile 0-0's own area (left of tile 0-1, above tile 1-0), away from its edges.
    xe, ye = truth.tile_origin["0-1"][0], truth.tile_origin["1-0"][1]
    for k, plane in enumerate(meta["planes"]):
        z = max(plane, key=lambda zw: zw[1])[0]
        _, _, (x0, y0) = masks(truth, meta, z)
        exp = expected(truth, z, s0[k].shape, (x0, y0))[y0 + 4:y0 + ye - 4, x0 + 4:x0 + xe - 4]
        img = s0[k][y0 + 4:y0 + ye - 4, x0 + 4:x0 + xe - 4].astype(float)
        a, b = np.polyfit(exp.ravel(), img.ravel(), 1)
        assert np.corrcoef(exp.ravel(), img.ravel())[0, 1] > 0.88
        # Stripe strength of what differs from the truth, in truth units: render vs raw tile.
        after = destreak.stripe_amplitude((img - b) / a - exp)
        own = (slice(4, ye - 4), slice(4, xe - 4))
        before = destreak.stripe_amplitude((raw[z][own] - LO) / (HI - LO) - truth.montage(z)[own])
        assert after < 0.7 * before, f"plane {k}: stripe amplitude {after:.4f} vs raw {before:.4f}"


def test_corrected_pyramid(corrected_run):
    _, out = corrected_run
    scales = read_scales(out)
    for s in range(1, len(scales)):
        np.testing.assert_array_equal(scales[s], pyramid.downsample(scales[s - 1]))
