"""pipeline/curtains.py: the curtain direction from sampled windows, decided only with a clear margin."""

import numpy as np
import pytest
from scipy import ndimage

import synth
from pipeline import curtains

N = 512
S = {**curtains.SETTINGS, "window": N}


def tissue(seed, n=N):
    """Texture without a direction, with fine and coarse structure, as two consecutive slices (nearly alike)."""
    rng = np.random.default_rng(seed)
    base = ndimage.gaussian_filter(rng.normal(size=(n, n)), 3) * 6 + ndimage.gaussian_filter(rng.normal(size=(n, n)), 12) * 20
    nxt = base + ndimage.gaussian_filter(rng.normal(size=(n, n)), 3) * 1.5
    return 1000 + 100 * base, 1000 + 100 * nxt


def stripes(seed, n=N, amplitude=60.0, count=12):
    """Dark lines along y, a few px wide, at random places: the same in both slices."""
    rng = np.random.default_rng(seed + 100)
    prof = np.zeros(n)
    for x in rng.integers(10, n - 10, count):
        prof -= amplitude * rng.uniform(0.5, 1.0) * np.exp(-0.5 * ((np.arange(n) - x) / rng.uniform(1.5, 3.0)) ** 2)
    return np.tile(prof, (n, 1))


def noise(seed, sd=20.0, n=N):
    return np.random.default_rng(seed + 500).normal(0, sd, (n, n))


def pair(seed, curtain=True, esb=True, scanlines=0.0, angle=0.0, amplitude=60.0):
    """(a, b, ea, eb): InLens and EsB windows of two consecutive slices."""
    t0, t1 = tissue(seed)
    c = stripes(seed, amplitude=amplitude) if curtain else 0.0
    if curtain and angle:
        c = ndimage.rotate(np.pad(c, N // 2, mode="wrap"), angle, reshape=False, order=1)[N // 2:-N // 2, N // 2:-N // 2]
    rng = np.random.default_rng(seed + 900)
    lines = lambda: scanlines * rng.normal(size=(N, 1)) * np.ones((1, N))   # new in every slice
    a, b = t0 + c + noise(seed) + lines(), t1 + c + noise(seed + 1) + lines()
    if not esb:
        return a, b, None, None
    return a, b, 0.6 * t0 + noise(seed + 2, 40) + lines(), 0.6 * t1 + noise(seed + 3, 40) + lines()


def votes(n, **kw):
    return [curtains.measure_window(*pair(seed, **kw), s=S) for seed in range(n)]


def test_vertical_curtains_and_the_same_windows_transposed():
    v = votes(6)
    assert [w["vote"] for w in v] == ["vertical"] * 6
    assert all(abs(w["angle"]) <= 0.5 and w["esb_amplitude"] == 0 and w["coherence"] > 0.9 for w in v)
    t = [curtains.measure_window(*(x.T for x in pair(seed)), s=S) for seed in range(6)]
    assert [w["vote"] for w in t] == ["horizontal"] * 6
    assert curtains.decide(v, s=S)["direction"] == "vertical" and curtains.decide(t, s=S)["use"] == "horizontal"


def test_without_esb_strong_curtains_are_still_found():
    """Without EsB the tissue's own profile is the background: weak curtains do not stand out of it."""
    v = votes(4, esb=False, amplitude=250.0)
    assert [w["vote"] for w in v] == ["vertical"] * 4 and v[0]["esb_amplitude"] is None
    assert [w["vote"] for w in votes(4, esb=False, amplitude=20.0)] == ["none"] * 4


def test_weak_curtains_are_found_with_esb():
    v = votes(4, amplitude=12.0)     # about 1 % of the contrast
    assert [w["vote"] for w in v] == ["vertical"] * 4 and all(w["amplitude"] < 0.012 for w in v)


def test_clean_windows_give_unclear_not_a_guess():
    v = votes(8, curtain=False)
    assert [w["vote"] for w in v] == ["none"] * 8
    d = curtains.decide(v, configured="vertical", s=S)
    assert d["direction"] == "unclear" and d["use"] == "vertical" and "was not measured" in d["message"]


def test_scan_line_noise_is_not_curtains():
    """Lines along x that change with every slice (in both detectors): no vote for horizontal."""
    v = votes(6, curtain=False, scanlines=30.0)
    assert [w["vote"] for w in v] == ["none"] * 6 and all(w["why"] == "not in both slices" for w in v)
    v = votes(6, curtain=True, scanlines=30.0)       # and they do not hide vertical curtains
    assert [w["vote"] for w in v] == ["vertical"] * 6


def test_stripes_both_detectors_show_are_not_curtains():
    for share, why in ((0.6, None), (1.5, "EsB shows it too")):   # as the tissue (0.6): they cancel; stronger in EsB
        for seed in range(4):
            a, b, ea, eb = pair(seed, curtain=False)
            c = stripes(seed)
            w = curtains.measure_window(a + c, b + c, ea + share * c, eb + share * c, s=S)
            assert w["vote"] == "none" and w.get("why") == why, (share, seed, w)


def test_curtains_at_an_angle_stop_the_run():
    v = votes(6, angle=12.0)
    assert [w["vote"] for w in v] == ["angle"] * 6
    assert all(abs(abs(w["angle"]) - 12.0) <= 1.0 for w in v)
    d = curtains.decide(v, s=S)
    assert d["direction"] == "conflicting" and d["use"] is None and "at an angle" in d["message"]


def test_verdict_needs_a_clear_margin():
    w = lambda vote: {"vote": vote, "angle": 0.0, "amplitude": 0.01}
    mix = lambda v, h, a, n: [w("vertical")] * v + [w("horizontal")] * h + [w("angle")] * a + [w("none")] * n
    assert curtains.decide(mix(10, 0, 0, 30))["direction"] == "vertical"
    assert curtains.decide(mix(10, 2, 0, 28))["direction"] == "vertical"          # 5 to 1
    assert curtains.decide(mix(10, 3, 0, 27))["direction"] == "conflicting"       # not 4 to 1
    assert curtains.decide(mix(6, 6, 0, 28))["direction"] == "conflicting"
    assert curtains.decide(mix(2, 0, 0, 38))["direction"] == "unclear"            # fewer than 3 windows
    assert curtains.decide(mix(0, 0, 0, 40), configured="horizontal")["use"] == "horizontal"
    assert curtains.decide(mix(0, 12, 1, 27))["use"] == "horizontal"


def test_detect_reads_sampled_windows_of_raw_files(tmp_path):
    """Files as exported: InLens stacks and their EsB stacks; windows sized to the tiles."""
    raw = tmp_path / "raw"
    (raw / "Backscatter").mkdir(parents=True)
    files = []
    for k, name in enumerate(("M10_D06_tile0-0.tif", "M10_D06_tile0-1.tif", "M10_D07_tile0-0.tif")):
        a, b, ea, eb = pair(k)
        to16 = lambda *x: np.clip(np.stack(x) * 20, 0, 65535).astype(np.uint16)
        synth.write_imagej(raw / name, to16(a, b, a), ["s"] * 3)
        synth.write_imagej(raw / "Backscatter" / name.replace("_tile", "_EsB_tile"), to16(ea, eb, ea), ["s"] * 3)
        files.append(name)
    ref = lambda f: "Backscatter/" + f.replace("_tile", "_EsB_tile")
    res = curtains.detect(raw, files, ref, per_file=1, s=S)
    assert res["direction"] == "vertical" and res["windows"] == 3 and res["with_esb"] == 3
    assert curtains.detect(raw, files, ref, per_file=1, transpose=True, s=S)["direction"] == "horizontal"
    assert curtains.main([str(raw), "--per-file", "1"]) == 0
