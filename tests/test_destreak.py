"""destreak: wavelet-FFT stripe removal and the tuning CLI."""

import re

import cv2
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

import synth
from pipeline import destreak


def stripe_amp(img, axis):
    """Std of the high-passed mean profile across the stripes (mean along ``axis``)."""
    p = np.asarray(img, np.float64).mean(axis=axis)
    return np.std(p - ndimage.uniform_filter1d(p, 15, mode="reflect"))


def striped(shape, direction, amplitude=0.05, seed=3):
    clean = synth.make_volume((1, *shape), seed=seed)[0]
    axis = 0 if direction == "vertical" else 1
    rng = np.random.default_rng(seed)
    s = ndimage.gaussian_filter1d(rng.standard_normal(shape[1 - axis]) * amplitude, 0.7).astype(np.float32)
    return clean, clean + (s[None, :] if axis == 0 else s[:, None]), axis


@pytest.mark.parametrize("direction,shape", [("vertical", (3001, 2999)), ("horizontal", (2999, 3001)),
                                             ("vertical", (2048, 2048))])
def test_removes_stripes(direction, shape):
    clean, img, axis = striped(shape, direction)
    out = destreak.destreak(img, direction=direction)
    assert out.shape == img.shape and out.dtype == np.float32
    assert out.mean() == pytest.approx(img.mean(), abs=1e-5)
    # Residual stripes (relative to the clean image) drop at least 5x...
    assert stripe_amp(img - clean, axis) / stripe_amp(out - clean, axis) >= 5
    # ...while the image itself is preserved.
    assert np.corrcoef(out.ravel(), clean.ravel())[0, 1] > 0.98
    # The other direction leaves these stripes alone.
    other = "horizontal" if direction == "vertical" else "vertical"
    wrong = destreak.destreak(img, direction=other)
    assert stripe_amp(img - clean, axis) / stripe_amp(wrong - clean, axis) < 1.5


@pytest.mark.parametrize("shape", [(1, 1), (5, 7), (33, 1), (127, 255)])
def test_small_and_odd_shapes(shape):
    img = np.random.default_rng(0).integers(0, 65535, shape).astype(np.uint16)
    out = destreak.destreak(img)
    assert out.shape == shape and out.dtype == np.float32 and np.isfinite(out).all()
    assert out.mean() == pytest.approx(img.mean(), rel=1e-5)


def test_from_cfg_uses_section():
    _, img, _ = striped((512, 600), "horizontal", amplitude=0.1)
    cfg = {"destreak": {"direction": "horizontal", "sigma": 2.0, "levels": 3}}
    np.testing.assert_array_equal(destreak.destreak_from_cfg(img, cfg),
                                  destreak.destreak(img, "horizontal", "db4", 3, 2.0))
    np.testing.assert_array_equal(destreak.destreak_from_cfg(img, {}), destreak.destreak(img))


def test_cli_writes_before_after(tmp_path, make_config):
    truth = synth.make_dataset(tmp_path / "raw", n_slices=4, tile_shape=(256, 300), streaks=0.2)
    out = tmp_path / "out"
    rows = []
    for name, zs in truth.files.items():
        tile = re.search(r"_tile(\d+-\d+)", name).group(1)
        r, c = map(int, tile.split("-"))
        for i, z in enumerate(zs):
            rows.append(dict(z=z, timestamp=truth.timestamps[z].isoformat(), tile=tile, tile_row=r, tile_col=c,
                             file=name, index=i, height=256, width=300, segment=0, seam=False, excluded=False,
                             exclude_reason="", label=""))
    (out / "check").mkdir(parents=True)
    pd.DataFrame(rows).sort_values(["z", "tile"]).to_csv(out / "check" / "slices.csv", index=False)
    cfg = make_config(truth, destreak={"levels": 4})

    assert destreak.main(["test", "--config", str(cfg), "--z", "2", "--tile", "1-0",
                          "--crop", "10", "210", "20", "260"]) == 0
    d = out / "destreak"
    images = {p.name: cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in d.glob("*.png")}
    assert set(images) == {f"z2_tile1-0_{k}{s}.png" for k in ("before", "after") for s in ("", "_crop", "_fft")}
    assert images["z2_tile1-0_before.png"].shape == (256, 300)
    assert images["z2_tile1-0_after_crop.png"].shape == (200, 240)
    assert images["z2_tile1-0_before_fft.png"].shape == (200, 240)
    before, after = (images[f"z2_tile1-0_{k}_crop.png"].astype(float) for k in ("before", "after"))
    assert stripe_amp(before, 0) > 3 * stripe_amp(after, 0)
    # The spectra share one contrast: vertical stripes live on the ky = 0 row (row 100 of the
    # 200-row crop), which darkens, while the rest of the spectrum keeps its level.
    fb, fa = (images[f"z2_tile1-0_{k}_fft.png"].astype(float) for k in ("before", "after"))
    assert fa[100].mean() < fb[100].mean() - 50
    assert abs(np.delete(fa, 100, 0).mean() - np.delete(fb, 100, 0).mean()) < 10
    with pytest.raises(SystemExit):
        destreak.main(["test", "--config", str(cfg), "--z", "99", "--tile", "1-0"])
    with pytest.raises(SystemExit, match="outside"):
        destreak.main(["test", "--config", str(cfg), "--z", "2", "--tile", "1-0", "--crop", "300", "400", "0", "50"])
