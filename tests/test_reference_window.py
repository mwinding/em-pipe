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
    per_z = pd.DataFrame({"z": pos.astype(int), "segment": 0, "m_s": m_s, "c_s": c_s})
    fl = intensity.flicker(per_z, pd.Series(L), pd.Series(H))
    assert fl["d_median"].max() < 0.1
    assert fl["d_contrast"].max() < 0.1
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
