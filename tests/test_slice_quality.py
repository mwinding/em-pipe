"""tools/slice_quality.py: the window metrics on synthetic images, and run / merge on a synthetic dataset
with a blurred slice, a saturated patch and a striped slice (flagged) among clean ones (not flagged)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

import synth
from pipeline import preview

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("slice_quality", ROOT / "tools" / "slice_quality.py")
sq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sq)


def texture(shape, seed=0, noise=150.0):
    """Fine 'tissue' texture (structure a few px across) around 33000 with pixel noise."""
    rng = np.random.default_rng(seed)
    t = ndimage.gaussian_filter(rng.standard_normal(shape), 1.2)
    return 33000 + 2500 * t / t.std() + rng.normal(0, noise, shape), rng


def test_sharpness_drops_with_blur_not_with_noise():
    img, rng = texture((256, 256), noise=0)
    s = sq.Spectrum(256)
    sharp, _ = s.measure(img + rng.normal(0, 150, img.shape))
    blurred, _ = s.measure(ndimage.gaussian_filter(img, 1.5) + rng.normal(0, 150, img.shape))
    noisy, noise = s.measure(img + rng.normal(0, 600, img.shape))
    assert blurred < 0.5 * sharp
    assert abs(noisy / sharp - 1) < 0.15            # four times the noise: sharpness unchanged
    assert abs(noise / 600 - 1) < 0.1               # and the noise sd is recovered


def test_stripes_raise_v_over_h():
    img, rng = texture((256, 256))
    stripes = img + 2500 * ndimage.gaussian_filter1d(rng.standard_normal(256), 0.7)[None, :]
    s = sq.Spectrum(256)
    read = lambda im: (lambda y0, y1: im[y0:y1])
    clean, _, _ = sq.measure_tile(read(img), img.shape, s, [0], [0], None)
    striped, _, _ = sq.measure_tile(read(stripes), img.shape, s, [0], [0], None)
    assert clean["v"][0] / clean["h"][0] < 2
    assert striped["v"][0] / striped["h"][0] > 4
    assert abs(striped["sharp"][0] / clean["sharp"][0] - 1) < 0.2     # the spectrum's axes are left out


def test_patches_need_both_neighbours():
    rng = np.random.default_rng(0)
    base = rng.integers(100, 140, (64, 64)).astype(np.uint8)
    t = base.copy()
    t[10:30, 10:30] += 80
    assert sq.patches(t, base, base)[0] > 0.05
    assert sq.patches(t, t, base) == (0.0, 0.0)       # a step that stays is not a patch
    assert sq.patches(base, base, base) == (0.0, 0.0)


def test_lone_jump_ignores_steps_and_wander():
    level = np.array([10, 10, 10, 30, 10, 10, 20, 20, 20, 20, 21, 22, 23], float)
    jump, usual = sq.lone_jump(level)
    assert jump[3] == 20 and jump[6] == 0 and jump[10] == 0     # a lone slice; a step that stays; a ramp
    assert np.count_nonzero(jump) == 1


def test_window_grid_covers_width():
    size, tops, lefts = sq.window_grid(1000, 2100, 512, 3)
    assert size == 512 and len(tops) == 3 and len(lefts) == 4
    assert tops[0] >= 0 and tops[-1] + size <= 1000 and lefts[-1] + size <= 2100


N, BLUR, SAT, STRIPE = 40, (12, "0-0"), (20, "0-1"), (28, "0-0")


@pytest.fixture(scope="module")
def faulty(tmp_path_factory):
    """A 1x2-tile dataset of 40 slices of fine texture that changes slowly along z; one tile slice
    blurred, one with a saturated patch, one striped."""
    tmp = tmp_path_factory.mktemp("slice_quality")
    truth = synth.make_dataset(tmp / "raw", n_slices=N, grid=(1, 2), tile_shape=(256, 384), interval_s=30.0)
    th, tw = truth.tile_shape
    rng = np.random.default_rng(1)
    for name, zs in truth.files.items():
        tile = name.split("_tile")[1].split(".")[0].split("_")[0]
        r, c = map(int, tile.split("-"))
        vol = ndimage.gaussian_filter(np.random.default_rng(10 + c).standard_normal((N + 8, th, tw)), (4, 1.2, 1.2))
        vol = 33000 + 2500 * vol / vol.std()
        data = np.empty((len(zs), th, tw), np.float64)
        for i, z in enumerate(zs):
            img = vol[z + 4]
            if (z, tile) == BLUR:
                img = ndimage.gaussian_filter(img, 1.5)
            if (z, tile) == STRIPE:
                img = img + 2500 * ndimage.gaussian_filter1d(rng.standard_normal(tw), 0.7)[None, :]
            img = img + rng.normal(0, 150, img.shape)
            if (z, tile) == SAT:
                img[60:140, 100:220] = 65535
            data[i] = img
        labels = [synth.LABEL.format(ts=truth.timestamps[z], r=r, c=c) for z in zs]
        synth.write_imagej(truth.raw_dir / name, np.clip(data, 0, 65535).astype(np.uint16), labels)
    out = tmp / "out"
    synth.write_slices(truth, out)
    cfg = synth.write_config(tmp / "config.yaml", truth.raw_dir, out, preview={"factor": 4})
    assert preview.main(["run", "--config", str(cfg)]) == 0 and preview.main(["merge", "--config", str(cfg)]) == 0
    return cfg, out


# 128 px windows: v/h scatters more than in the default 512 px ones, so a higher ratio counts as curtained
ARGS = ["--window", "128", "--trend", "21", "--chunk-slices", "10", "--curtain-ratio", "3"]


def test_run_merge_flags_the_faults_only(faulty):
    cfg, out = faulty
    for k in range(2):
        sq.main(["run", "--config", str(cfg), "--task-id", str(k), "--num-tasks", "2", *ARGS])
    assert sq.main(["merge", "--config", str(cfg), "--top", "3", *ARGS]) == 0
    qc = out / "qc"
    q = pd.read_csv(qc / "slice_quality.csv", dtype={"tile": str})
    assert len(q) == 2 * N and {"sharp_rel", "noise_rel", "curtain_extent", "sat", "bright", "ncc"} <= set(q.columns)
    f = pd.read_csv(qc / "slice_flags.csv", dtype={"tile": str})
    got = {(r.z, r.tile, r.reason) for r in f.itertuples()}
    assert (*BLUR, "focus") in got
    assert (*SAT, "saturated") in got and (*SAT, "bright_patch") in got and (*SAT, "dark_patch") not in got
    assert (*STRIPE, "curtain") in got
    faults = {BLUR, SAT, STRIPE}
    assert {(r.z, r.tile) for r in f.itertuples()} == faults          # clean tile slices carry no flag
    assert set(f.columns) >= {"z", "tile", "reason", "value", "limit", "severity"}
    summary = json.loads((qc / "slice_quality" / "summary.json").read_text())
    assert summary["counts"]["focus"]["slices"] == 1 and summary["flagged_slices"] == 3
    assert (qc / "slice_quality.png").exists()
    for reason in ("focus", "saturated", "curtain"):
        assert (qc / "slice_quality" / f"sheet_{reason}.jpg").exists(), reason


def test_merge_refuses_incomplete_run(faulty, tmp_path):
    cfg, _ = faulty
    sq.main(["run", "--config", str(cfg), "--out", str(tmp_path), "--task-id", "0", "--num-tasks", "2", *ARGS])
    with pytest.raises(SystemExit):
        sq.main(["merge", "--config", str(cfg), "--out", str(tmp_path), *ARGS])


def test_without_preview_and_excluded_slices(tmp_path):
    truth = synth.make_dataset(tmp_path / "raw", n_slices=8, tile_shape=(128, 144))
    out = tmp_path / "out"
    synth.write_slices(truth, out, excluded=[3, (5, "0-1")])
    cfg = synth.write_config(tmp_path / "config.yaml", truth.raw_dir, out)
    sq.main(["run", "--config", str(cfg), "--window", "64"])
    assert sq.main(["merge", "--config", str(cfg), "--window", "64", "--no-sheets"]) == 0
    f = pd.read_csv(out / "qc" / "slice_flags.csv", dtype={"tile": str})
    got = {(r.z, r.tile, r.reason) for r in f.itertuples()}
    assert (3, "all", "excluded") in got and (5, "0-1", "excluded") in got
