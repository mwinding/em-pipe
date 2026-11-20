"""intensity.window: reference: window maths, the spread measures, smoothing over z (no flicker),
and refstats' slice sampling and region filter."""

import numpy as np
import pandas as pd
import pytest

from pipeline import intensity, refstats


def test_window_puts_tissue_at_targets():
    m, c = 31000.0, 1800.0                      # common-scale median and p5-p95 width
    lo, hi = intensity.window_from_stats(m, c, 150.0, 131.0)
    assert intensity.grey(m, lo, hi) == pytest.approx(150.0)
    assert 255 * c / (hi - lo) == pytest.approx(131.0)


def test_window_is_linear_in_the_measurements():
    # Twice the spread -> twice the window width; the median grey stays put.
    lo1, hi1 = intensity.window_from_stats(1000.0, 100.0, 150.0, 130.0)
    lo2, hi2 = intensity.window_from_stats(1000.0, 200.0, 150.0, 130.0)
    assert (hi2 - lo2) == pytest.approx(2 * (hi1 - lo1))
    assert intensity.grey(1000.0, lo2, hi2) == pytest.approx(150.0)


def test_spread_measures():
    st = pd.DataFrame({"rq_5": [10.0, 20.0], "rq_95": [110.0, 140.0], "lc32": [7.0, 8.0]})
    assert intensity.spread(st, "p5_p95").tolist() == [100.0, 120.0]
    assert intensity.spread(st, "lc32").tolist() == [7.0, 8.0]
    with pytest.raises(ValueError):
        intensity.spread(st, "p1_p99")


def test_smooth_samples_ignores_noise_and_holds_ends():
    rng = np.random.default_rng(0)
    pos = np.arange(0, 3000, 32.0)
    vals = 100 + rng.normal(0, 5, len(pos))
    vals[10] = 400                              # one outlier sample
    all_pos = np.arange(0, 3100, 1.0)
    curve = intensity.smooth_samples(pos, vals, all_pos, smooth=601, ramp=129)
    assert curve.max() < 110                    # the outlier doesn't come through
    assert np.abs(np.diff(curve)).max() < 0.05  # tiny change from one slice to the next
    assert curve[-1] == pytest.approx(curve[-60], abs=0.5)   # held past the last sample


def test_smooth_samples_without_valid_samples():
    assert intensity.smooth_samples([0, 32], [np.nan, np.nan], np.arange(64.0), 601, 129) is None


def test_noisy_stats_give_no_flicker():
    """Per-slice windows from noisy sampled statistics change by well under a grey between slices."""
    rng = np.random.default_rng(1)
    n = 4000
    pos = np.arange(n, dtype=float)
    sampled = pos[::32]
    m_true = 30000 + 400 * np.sin(pos / 900)    # slow real drift of the tissue on the common scale
    c_true = 1800 + 80 * np.cos(pos / 1200)
    m_s = intensity.smooth_samples(sampled, m_true[::32] + rng.normal(0, 60, len(sampled)), pos, 601, 129)
    c_s = intensity.smooth_samples(sampled, c_true[::32] + rng.normal(0, 40, len(sampled)), pos, 601, 129)
    L, H = intensity.window_from_stats(m_s, c_s, 150.0, 131.0)
    # the tissue held, how much each slice's window moves its grey and contrast from the slice before
    d_median = np.abs(intensity.grey(m_s[:-1], L[1:], H[1:]) - intensity.grey(m_s[:-1], L[:-1], H[:-1]))
    d_contrast = np.abs(255 * c_s[:-1] / (H[1:] - L[1:]) - 255 * c_s[:-1] / (H[:-1] - L[:-1]))
    assert d_median.max() < 0.1
    assert d_contrast.max() < 0.1
    # and the slow drift is followed: the tissue stays near its target grey throughout
    assert np.abs(intensity.grey(m_true, L, H) - 150).max() < 8


def test_sample_z_every_nth_per_segment_and_last():
    sl = pd.DataFrame({"z": list(range(10)) + list(range(10, 15)), "segment": [0] * 10 + [1] * 5})
    assert refstats.sample_z(sl, 4) == [0, 4, 8, 9, 10, 14]


def test_in_region_with_open_sides():
    M = np.array([[1.0, 0, 0], [0, 1.0, 0]])    # tile px = volume voxels
    pts = np.array([[100.0, 5.0], [8000.0, 5.0], [8000.0, 99999.0]])
    assert refstats.in_region(None, M, pts).tolist() == [True, True, True]
    assert refstats.in_region([7700, None, None, None], M, pts).tolist() == [False, True, True]
    assert refstats.in_region([7700, None, None, 1000], M, pts).tolist() == [False, True, False]


def test_blocks_split_at_segments_and_breaks():
    per_z = pd.DataFrame({"z": range(10), "segment": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]})
    assert intensity.block_ids(per_z).tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert intensity.block_ids(per_z, [3]).tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    # a break on a slice that isn't selected starts the block at the next selected slice
    gappy = pd.DataFrame({"z": [0, 1, 5, 6, 9], "segment": 0})
    assert intensity.block_ids(gappy, [3]).tolist() == [0, 0, 1, 1, 1]


def test_tissue_brightness_follows_slices_and_puts_neuropil_on_target():
    """brightness: tissue: the window moves with every slice's tissue level, its width is set per block from
    the neuropil's spread, and the neuropil's median lands on the target over the block."""
    n = 400
    z = np.arange(n)
    level = 30000 + 300 * np.sin(z / 20.0)            # the slice's tissue level drifts (acquisition)
    df = pd.DataFrame({"z": z, "tile": "0-0", "segment": (z >= 200).astype(int), "pos": z.astype(float),
                       "lo": level - 2000.0, "hi": level + 2000.0, "g": 1.0, "o": 0.0, "jump": 0.0})
    per_z = pd.DataFrame({"z": z, "segment": df["segment"], "pos": df["pos"]})
    per_z["block"] = intensity.block_ids(per_z)
    sampled = z % 32 == 0
    # neuropil sits 200 below the tissue level; its p5-p95 spread differs per segment
    per_z["median"] = np.where(sampled, level - 200, np.nan)
    spread_seg = np.where(per_z["segment"] == 0, 1800.0, 1300.0)
    per_z["c_s"] = spread_seg
    cfg = {"intensity": {"target_median": 140, "per_slice": "slice"}}
    r = {"median": 150.0, "contrast": 140.0}
    out = intensity._tissue_brightness(cfg, df, per_z, r)
    width = out["H"] - out["L"]
    assert np.allclose(width, 255 * spread_seg / 140)                  # contrast per segment
    g = intensity.grey(level - 200, out["L"], out["H"])
    assert np.allclose(g, 150, atol=1e-6)                             # neuropil on target in every slice
    # the window follows the drifting tissue: its position changes with the level
    assert np.corrcoef(out["L"][:200], level[:200])[0, 1] > 0.99


def test_tissue_brightness_follows_a_slow_drift_within_a_block():
    """brightness: tissue with per: slice: when the neuropil slowly moves against the rest of the tissue
    within a long block (the tissue mix changes), one shift per block leaves the block's ends off target
    and a step at the next block; the slow shift keeps the neuropil on target along the whole block,
    while still following the fast slice-to-slice level of the tissue."""
    n = 4000
    z = np.arange(n)
    level = 30000 + 300 * np.sin(z / 7.0)            # fast slice-to-slice changes (acquisition)
    seg = (z >= 3000).astype(int)
    drift = np.where(seg == 0, -400.0 * z / 3000, 0.0)   # neuropil vs tissue drifts by 400 along block 0
    df = pd.DataFrame({"z": z, "tile": "0-0", "segment": seg, "pos": z.astype(float),
                       "lo": level - 2000.0, "hi": level + 2000.0, "g": 1.0, "o": 0.0, "jump": 0.0})
    per_z = pd.DataFrame({"z": z, "segment": seg, "pos": df["pos"]})
    per_z["block"] = intensity.block_ids(per_z)
    sampled = z % 64 == 0
    neuropil = level - 200 + drift
    per_z["median"] = np.where(sampled, neuropil, np.nan)
    per_z["c_s"] = 1500.0
    cfg = {"intensity": {"target_median": 140, "per_slice": "slice"}}
    flat = intensity._tissue_brightness(cfg, df, per_z.copy(), {"median": 150.0, "contrast": 140.0,
                                                                 "per": "segment"})
    slow = intensity._tissue_brightness(cfg, df, per_z.copy(), {"median": 150.0, "contrast": 140.0, "per": "slice",
                                                                 "smooth_slices": 601, "ramp_slices": 65})
    g_flat = intensity.grey(neuropil, flat["L"], flat["H"])
    g_slow = intensity.grey(neuropil, slow["L"], slow["H"])
    end = (z > 2800) & (z < 3000)
    assert abs(g_flat[end].mean() - 150) > 8                     # one shift: the block's end is off target
    assert np.abs(g_slow[seg == 0] - 150).max() < 4            # slow shift: on target along the block
    assert abs(g_slow[2999] - g_slow[3000]) < 4                # no step into the next block
    # the shift changes slowly: the window still follows the fast tissue level slice by slice
    assert np.abs(np.diff(slow["L"].to_numpy() - level)[:2990]).max() < 2.0


def _samples(fn, n=6000, every=64, noise=0.015, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.arange(0, n, every, dtype=float)
    return pos, fn(pos) * (1 + rng.normal(0, noise, len(pos))), np.arange(n, dtype=float)


def test_sample_noise_ignores_slow_drift():
    pos, v, _ = _samples(lambda p: 1 + 0.07 * p / 6000, noise=0.015)
    assert abs(intensity.sample_noise([(pos, v)]) - 0.015) < 0.004


def test_follow_samples_keeps_the_long_window_on_steady_or_slowly_drifting_samples():
    """Noise and a slow drift only: the longest window everywhere (as smooth_samples), no flicker."""
    pos, v, allp = _samples(lambda p: 1 + 0.07 * p / 6000)
    sigma = intensity.sample_noise([(pos, v)])
    curve, (sp, used) = intensity.follow_samples(pos, v, allp, 2001, 129, sigma, 2.5)
    assert np.mean(used >= 2000) > 0.9
    fixed = intensity.smooth_samples(pos, v, allp, 2001, 129)
    assert np.abs(curve - fixed).max() < 0.01
    assert np.abs(np.diff(curve)).max() < 2e-4


def test_follow_samples_follows_a_fast_settling_at_the_block_start():
    """+15 % at a block's start settling over ~300 slices (the microscope after a pause): the fixed window,
    looking only forward there, leaves the first slices ~10 % off; the adaptive one follows."""
    truth = lambda p: 1 + 0.15 * np.exp(-p / 300)
    pos, v, allp = _samples(truth)
    sigma = intensity.sample_noise([(pos, v)])
    curve, (sp, used) = intensity.follow_samples(pos, v, allp, 2001, 129, sigma, 2.5)
    fixed = intensity.smooth_samples(pos, v, allp, 2001, 129)
    start = allp < 200
    assert np.abs(fixed[start] - truth(allp[start])).max() > 0.06
    assert np.abs(curve[start] - truth(allp[start])).max() < 0.05      # from ~0.12: what 1.5 % noise per sample allows
    assert used[-1] >= 2000                                    # the long window once settled


def test_follow_samples_follows_a_step_without_smearing_it():
    """A 8 % step mid-block (a setting changed): followed within a few samples, steady on both sides."""
    truth = lambda p: np.where(p < 3000, 1.0, 1.08)
    pos, v, allp = _samples(truth, seed=2)
    sigma = intensity.sample_noise([(pos, v)])
    curve, _ = intensity.follow_samples(pos, v, allp, 2001, 129, sigma, 2.5)
    far = np.abs(allp - 3000) > 400
    assert np.abs(curve[far] - truth(allp[far])).max() < 0.02
    fixed = intensity.smooth_samples(pos, v, allp, 2001, 129)

