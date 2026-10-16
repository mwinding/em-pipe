"""shading: flat-fields per (segment, tile) from the thumbnails, used by intensity's balance and render."""

import json

import numpy as np
import pandas as pd
import pytest

import synth
from pipeline import intensity, omezarr, render, shading
from pipeline.config import load_config
from pipeline.preview import thumb_to_tile
from test_preview import run_preview
from test_render import SMALL, _aligned, _setup

# Brightness falls off across tile 0-1 by 12 % left to right; 1-1 by 8 % in x and rises 6 % in y.
SHADE = {"0-1": (-0.12, 0.0), "1-1": (-0.08, 0.06)}
N = 72   # the anatomy's own brightness trend averages out over slices; the fit's degree 2 fits a linear truth


@pytest.fixture(scope="module")
def shaded(tmp_path_factory):
    """Dataset with shaded tiles; preview, then intensity with shading on: (truth, config path, out)."""
    root = tmp_path_factory.mktemp("shading")
    truth = synth.make_dataset(root / "raw", n_slices=N, tile_shape=(192, 224), overlap=(24, 28),
                               tile_shading=SHADE, noise=0.002, seed=5)
    cfg, out = _setup(root, truth, levels=False, preview={"factor": 4},
                      intensity={"window": "percentile", "per_slice": False},
                      shading={"enabled": True, "every": 1, "prior_weight": 0.0})
    run_preview(cfg)
    assert intensity.main(["--config", str(cfg)]) == 0
    return truth, cfg, out


def _truth_field(truth, tile, thumb_shape):
    """The dataset's gradient of ``tile`` sampled at the thumbnail pixel centres."""
    th, tw = truth.tile_shape
    ax, ay = SHADE.get(tile, (0.0, 0.0))
    T = thumb_to_tile(thumb_shape, (th, tw))
    x = T[0, 0] * np.arange(thumb_shape[1]) + T[0, 2]
    y = T[1, 1] * np.arange(thumb_shape[0]) + T[1, 2]
    gx, gy = 1 + ax * (x / (tw - 1) - 0.5), 1 + ay * (y / (th - 1) - 0.5)
    return gy[:, None] * gx[None, :]


def test_fields_recover_the_gradients(shaded):
    truth, cfg, out = shaded
    fields = shading.load(load_config(cfg, step_defaults=shading.DEFAULTS))
    assert set(fields) == {(0, t) for t in truth.tiles}
    for tile in truth.tiles:
        f = fields[(0, tile)]
        want = _truth_field(truth, tile, f.shape)
        want /= want.mean()
        assert f.shape == (48, 56) and f.dtype == np.float32
        assert np.abs(f / want - 1).max() < 0.02, tile       # within 2 % everywhere
    info = json.loads((out / "work" / "intensity" / "shading.json").read_text())
    assert {(i["segment"], i["tile"]) for i in info["fields"]} == set(fields) and info["fields"][0]["slices"] == N
    assert (out / "qc" / "shading.png").exists()
    # off: nothing loaded; on without the file: an error that says what to run
    assert shading.load({"shading": {"enabled": False}}) == {}
    with pytest.raises(FileNotFoundError, match="intensity step"):
        shading.load({"shading": {"enabled": True}, "output_dir": str(out / "elsewhere")})


def test_fit_field_is_smooth_and_mean_one():
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:40, 0:50]
    truth = np.exp(-0.1 * x / 49 + 0.05 * (y / 39) ** 2)
    noisy = truth * (1 + rng.normal(0, 0.02, truth.shape)) + 0.3 * (rng.random(truth.shape) < 0.01)  # outlier dots
    count = np.full(truth.shape, 30)
    count[:5, :5] = 2                                   # too few slices: excluded from the fit
    f = shading.fit_field(noisy, count, degree=3, min_count=5, max_gain=1.5)
    assert np.average(f[count >= 5], weights=count[count >= 5]) == pytest.approx(1, abs=1e-6)
    want = truth / np.average(truth[count >= 5], weights=count[count >= 5])
    assert np.abs(f - want).max() < 0.01
    assert shading.fit_field(noisy, np.zeros_like(count), 3) is None
    # an unmeasured half of the tile: the fitted gain relaxes towards 1 there instead of extrapolating
    count2 = count.copy()
    count2[:, 25:] = 0
    g = shading.fit_field(np.exp(-0.3 * x / 49), count2, degree=2, prior_weight=0.2)
    assert abs(g[:, 49].mean() - 1) < abs(np.exp(-0.3) / np.exp(-0.15 * 24 / 49) - 1) * 0.5


def test_gain_window_resamples_the_field_to_tile_pixels():
    field = _truth_field_plain((48, 56), (192, 224), -0.12).astype(np.float32)
    full = shading.gain_window(field, (192, 224), slice(0, 192), slice(0, 224))
    assert full.shape == (192, 224)
    T = thumb_to_tile(field.shape, (192, 224))
    ys, xs = (T[1, 1] * np.arange(48) + T[1, 2]).astype(int), (T[0, 0] * np.arange(56) + T[0, 2]).astype(int)
    assert np.allclose(full[np.ix_(ys, xs)], field, atol=2e-3)   # the field at thumbnail centres
    part = shading.gain_window(field, (192, 224), slice(100, 150), slice(30, 90))
    np.testing.assert_array_equal(part, full[100:150, 30:90])
    assert np.allclose(full[:, 0], field[0, 0]) and np.allclose(full[:, -1], field[0, -1])   # replicated past the centres


def _truth_field_plain(thumb_shape, tile_shape, ax):
    T = thumb_to_tile(thumb_shape, tile_shape)
    x = T[0, 0] * np.arange(thumb_shape[1]) + T[0, 2]
    return np.ones((thumb_shape[0], 1)) * (1 + ax * (x / (tile_shape[1] - 1) - 0.5))[None, :]


def test_balance_ignores_the_gradient(shaded):
    """With the fields divided out, the overlaps agree and no tile gets a spurious gain."""
    truth, cfg, out = shaded
    lv = pd.read_csv(out / "work" / "intensity" / "levels.csv", dtype={"tile": str})
    width = lv.assign(w=lv["hi"] - lv["lo"]).pivot(index="z", columns="tile", values="w")
    assert (width.div(width["0-0"], axis=0) - 1).abs().max().max() < 0.05


def _lr_ratio(vol, truth, tile):
    """Mean of the left third over the right third of ``tile``'s inner area, over z, in montage coords."""
    th, tw = truth.tile_shape
    x0, y0 = truth.tile_origin[tile]
    xa, xb, ya, yb = x0 + 40, x0 + tw - 40, y0 + 30, y0 + th - 30
    vals = []
    for z, (dx, dy) in enumerate(truth.drift):
        m = vol[z, dy + ya:dy + yb, dx + xa:dx + xb].astype(float)
        third = m.shape[1] // 3
        vals.append(m[:, :third].mean() / m[:, -third:].mean())
    return float(np.mean(vals))


def test_render_flattens_the_tiles(shaded):
    truth, cfg, out = shaded
    for name, on in (("flat.ome.zarr", True), ("plain.ome.zarr", False)):
        c = synth.write_config(out.parent / f"{name}.yaml", truth.raw_dir, out, render={**SMALL, "name": name},
                               shading={"enabled": on, "every": 1, "prior_weight": 0.0})
        assert render.main(["init", "--config", str(c)]) == 0
        assert render.main(["run", "--config", str(c)]) == 0
    flat = omezarr.open_scale(out / "flat.ome.zarr", 0).read().result()
    plain = omezarr.open_scale(out / "plain.ome.zarr", 0).read().result()
    img, cov = _aligned(truth, range(N))
    want = _lr_ratio(255 * img, truth, "0-1")
    assert abs(_lr_ratio(flat, truth, "0-1") / want - 1) < 0.02         # gradient gone
    assert _lr_ratio(plain, truth, "0-1") / want > 1.08                  # 12 % fall-off still there
    assert abs(_lr_ratio(flat, truth, "0-0") / _lr_ratio(plain, truth, "0-0") - 1) < 0.02   # unshaded tile (nearly) untouched
    corr = np.corrcoef(flat[cov].astype(float), img[cov])[0, 1]
    assert corr > np.corrcoef(plain[cov].astype(float), img[cov])[0, 1] and corr > 0.99
    meta = json.loads((out / "work" / "render" / "flat" / "render.json").read_text())
    assert len(meta["shading"]) == 64 and json.loads((out / "work" / "render" / "plain" / "render.json").read_text())["shading"] is None
