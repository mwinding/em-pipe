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
    (out / "work" / "check").mkdir(parents=True)
    pd.DataFrame(rows).sort_values(["z", "tile"]).to_csv(out / "work" / "check" / "slices.csv", index=False)
    cfg = make_config(truth, destreak={"levels": 4})

    assert destreak.main(["test", "--config", str(cfg), "--z", "2", "--tile", "1-0",
                          "--crop", "10", "210", "20", "260"]) == 0
    d = out / "work" / "destreak"
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


# ----- stripe_px, reflected ends, the swap from a reference detector, curtain cores -----------------

REF = {"reference_from": r"^(.*/)?(M\d\d_D\d\d)(.*)$", "reference_to": r"\1Backscatter/\2_EsB\3"}


def _texture(shape, seed):
    return synth.make_volume((1, *shape), seed=seed)[0]


def _lines(shape, xs, depth=0.15, rows=slice(None)):
    """Dark vertical lines 3 px wide at columns xs (over ``rows``): structure that runs along the stripes."""
    out = np.zeros(shape, np.float32)
    for x in xs:
        out[rows, x - 1:x + 2] -= depth
    return out


def _curtains(shape, amplitude, seed):
    rng = np.random.default_rng(seed)
    s = ndimage.gaussian_filter1d(rng.standard_normal(shape[1]) * amplitude, 0.7).astype(np.float32)
    return np.broadcast_to(s[None, :], shape).copy()


def _kept(out, base, lines):
    """Fraction of the lines' column profile left in out (regression slope on the lines' profile)."""
    p, q = (out - base).mean(axis=0), lines.mean(axis=0)
    p, q = p - ndimage.uniform_filter1d(p, 31), q - ndimage.uniform_filter1d(q, 31)
    return float((p * q).sum() / (q * q).sum())


def test_swap_keeps_structure_along_the_stripes_that_destreak_loses():
    shape = (2048, 2048)
    rng = np.random.default_rng(1)
    base = _texture(shape, 4)
    lines = _lines(shape, rng.choice(np.arange(50, 2000), 12, replace=False))
    clean = base + lines
    img = clean + _curtains(shape, 0.05, 5)
    # Another detector: other contrast and offset, 3x the noise, no curtains.
    ref = 0.6 * clean + 0.2 + rng.normal(0, 0.05, shape).astype(np.float32)
    out = destreak.swap(img, ref, levels=8, stripe_px=256)
    assert out.shape == shape and out.dtype == np.float32
    assert out.mean() == pytest.approx(img.mean(), abs=1e-4)
    assert stripe_amp(img - clean, 0) / stripe_amp(out - clean, 0) >= 5
    assert _kept(out, base, lines) > 0.85               # the lines come back from the reference
    notch = destreak.destreak(img, levels=8, stripe_px=256)
    # The filter alone takes the lines out with the curtains: its output is close to the image without them.
    assert stripe_amp(img - clean, 0) / stripe_amp(notch - base, 0) >= 5
    assert _kept(notch, base, lines) < 0.3


def test_swap_gain_is_not_biased_low_by_a_noisy_reference():
    shape = (1024, 1024)
    rng = np.random.default_rng(2)
    base = _texture(shape, 6)
    lines = _lines(shape, rng.choice(np.arange(50, 970), 8, replace=False))
    clean = base + lines
    img = clean + _curtains(shape, 0.05, 7)
    # The gain is corrected for the reference's noise; where that is too noisy to estimate it, it errs
    # towards taking out (as destreak does), not towards adding.
    for noise, at_least in ((0.0, 0.99), (0.02, 0.88), (0.15, 0.75)):
        ref = 0.5 * clean + rng.normal(0, noise, shape).astype(np.float32)
        kept = _kept(destreak.swap(img, ref, levels=7, stripe_px=256), base, lines)
        assert at_least < kept < 1.05, (noise, kept)


def test_swap_checks_the_reference_shape():
    with pytest.raises(ValueError, match="reference shape"):
        destreak.swap(np.zeros((64, 64), np.float32), np.zeros((64, 65), np.float32))


def test_stripe_px_leaves_shorter_structure_alone():
    shape = (2048, 1024)
    rng = np.random.default_rng(3)
    base = _texture(shape, 8)
    short = np.zeros(shape, np.float32)          # 3 x 60 px segments, scattered
    for _ in range(400):
        y, x = rng.integers(0, shape[0] - 60), rng.integers(2, shape[1] - 2)
        short[y:y + 60, x - 1:x + 2] -= 0.15
    clean = base + short
    curt = _curtains(shape, 0.05, 9)
    seg = short < 0
    depth = {}
    for sp in (128, 1024):
        out = destreak.destreak(clean + curt, levels=7, stripe_px=sp)
        # The curtains go (what they change in the output, against the filtered clean image)...
        assert stripe_amp(curt, 0) / stripe_amp(out - destreak.destreak(clean, levels=7, stripe_px=sp), 0) >= 20
        depth[sp] = -np.median((out - base)[seg])
    # ...and the 60 px segments keep more of their depth the longer a stripe has to be.
    assert depth[1024] > 0.8 * 0.15 and depth[128] < 0.75 * depth[1024]


def test_bands_are_reflected_so_top_and_bottom_dont_mix():
    shape = (1536, 1024)
    img = _texture(shape, 10)
    curt = _curtains(shape, 0.08, 11)
    curt[512:] = 0                                # curtains in the top third only
    # What the curtains change in the output (the filter's own effect on the texture cancels).
    leak = destreak.destreak(img + curt, levels=6, stripe_px=128) - destreak.destreak(img, levels=6, stripe_px=128)
    bottom = np.s_[-64:, :]                       # wrapped around, the top's stripes would land here
    assert stripe_amp(leak[bottom], 0) < 0.05 * stripe_amp(curt[:512], 0)


def test_fill_cores_takes_out_a_curtain_that_wanders_sideways():
    shape = (2048, 1024)
    rng = np.random.default_rng(12)
    clean = _texture(shape, 13)
    img = clean + rng.normal(0, 0.01, shape).astype(np.float32)     # InLens: low noise
    x0 = 500
    wobble = rng.integers(-1, 2, shape[0])        # the line moves up to 1 px either way from row to row
    for y in range(shape[0]):
        img[y, x0 + wobble[y] - 1:x0 + wobble[y] + 2] -= 0.4
    ref = 0.6 * clean + 0.1 + rng.normal(0, 0.03, shape).astype(np.float32)
    out = destreak.swap(img, ref, levels=8, stripe_px=256)
    filled = destreak.fill_cores(img, out, ref, threshold=4.0)
    near = np.s_[:, x0 - 4:x0 + 5]
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    assert rms((filled - clean)[near]) < 0.5 * rms((out - clean)[near])
    far = np.s_[:, :x0 - 40]
    np.testing.assert_array_equal(filled[far], out[far])     # untouched away from the core


def test_reference_file_maps_paths():
    cfg = {"destreak": dict(REF)}
    assert destreak.reference_file("M10_D21_tile0-0.tif", cfg) == "Backscatter/M10_D21_EsB_tile0-0.tif"
    assert (destreak.reference_file("Approach/M10_D17_tile0-0_part2.tif", cfg)
            == "Approach/Backscatter/M10_D17_EsB_tile0-0_part2.tif")
    assert destreak.reference_file("Approach/M10_D16_part1.tif", cfg) == "Approach/Backscatter/M10_D16_EsB_part1.tif"
    with pytest.raises(ValueError, match="does not map"):
        destreak.reference_file("other.tif", cfg)


def test_settings_are_checked_and_dispatch():
    with pytest.raises(ValueError, match="go together"):
        destreak.settings({"destreak": {"reference_from": "a"}})
    with pytest.raises(ValueError, match="needs a reference"):
        destreak.settings({"destreak": {"core_threshold": 4}})
    with pytest.raises(ValueError, match="stripe_px"):
        destreak.settings({"destreak": {"stripe_px": 0}})
    img = _texture((256, 256), 14)
    with pytest.raises(ValueError, match="none was given"):
        destreak.destreak_from_cfg(img, {"destreak": dict(REF)})
    ref = 0.5 * img
    np.testing.assert_array_equal(destreak.destreak_from_cfg(img, {"destreak": {**REF, "stripe_px": 64}}, ref),
                                  destreak.swap(img, ref, stripe_px=64))
    np.testing.assert_array_equal(destreak.destreak_from_cfg(img, {"destreak": {"stripe_px": 64}}),
                                  destreak.destreak(img, stripe_px=64))


def test_deeper_levels_take_broad_bands_and_leave_coarse_structure():
    shape = (4096, 4096)
    rng = np.random.default_rng(15)
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    base = _texture(shape, 16)
    blobs = np.zeros(shape, np.float32)               # cell-sized structure, ~600 px across
    for cy, cx in rng.uniform(300, 3800, (12, 2)):
        blobs += 0.2 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 250.0 ** 2))
    band = -0.1 * np.exp(-((xx - 2000) / 100.0) ** 2)  # a broad curtain band, ~200 px wide, full height
    img = base + blobs + band
    # The reference renders the coarse structure with half the contrast (another detector), and has no band.
    ref = 0.6 * (base + 0.5 * blobs) + rng.normal(0, 0.03, shape).astype(np.float32)
    lp = lambda a: cv2.GaussianBlur(a, (0, 0), 50)
    away = np.abs(xx[0] - 2000) > 500

    def judge(out):
        left = lp(out - base - blobs).mean(axis=0)
        band_left = (left[1900:2100].mean() - left[away].mean()) / band[0, 1900:2100].mean()
        p, q = lp(out - base)[:, away].ravel(), lp(blobs)[:, away].ravel()
        return band_left, float(np.dot(p - p.mean(), q - q.mean()) / np.dot(q - q.mean(), q - q.mean()))

    left7, _ = judge(destreak.swap(img, ref, levels=7, stripe_px=256, aspect=8))
    left9, kept9 = judge(destreak.swap(img, ref, levels=9, stripe_px=256, aspect=8))
    # A band's average darkening reaches into the coarsest scales, which the swap leaves alone (shading):
    # each level deeper takes more of it, and the coarse structure stays the image's own.
    assert left9 < 0.5 * left7 and left9 < 0.4
    assert kept9 > 0.95
