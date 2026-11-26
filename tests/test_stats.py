"""The small helpers shared by the steps and the tools: pipeline/stats.py's robust_sd and
pipeline/features.py's stretch_to_uint8. Each must give exactly what its callers wrote out before they
shared it (same value, same bits, same dtype), as well as what it is meant to estimate."""
import numpy as np
import pandas as pd
import pytest

from pipeline import features, stats


def written_out_sd(a):
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def test_robust_sd_is_the_scaled_mad_bit_for_bit():
    rng = np.random.default_rng(0)
    for x in (rng.normal(5, 2, 1001), rng.normal(5, 2, 1000).astype(np.float32), rng.integers(0, 50, 999),
              rng.normal(0, 1, (40, 30))):
        out, ref = stats.robust_sd(x), written_out_sd(x)
        assert out == ref and np.asarray(out).dtype == np.asarray(ref).dtype
    assert stats.robust_sd(rng.normal(0, 1, 10).astype(np.float32)).dtype == np.float32
    s = pd.Series(rng.normal(0, 1, 500))                     # a pandas Series, as intensity passes one
    assert stats.robust_sd(s) == written_out_sd(s)
    assert stats.robust_sd([1.0, 2.0, 4.0]) == 1.4826


def test_robust_sd_estimates_the_sd_and_hardly_moves_with_outliers():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 3, 100_000)
    assert abs(stats.robust_sd(x) - 3) < 0.05
    x[:10_000] = 1e6                                         # 10 % far off: the SD is ~3e5, the robust SD stays near 3
    assert 3 < stats.robust_sd(x) < 3.6


def test_robust_sd_edge_cases_are_numpys():
    assert np.isnan(stats.robust_sd([1.0, np.nan, 2.0]))
    with pytest.warns(RuntimeWarning):
        assert np.isnan(stats.robust_sd([]))
    assert stats.robust_sd([7.0, 7.0, 7.0]) == 0


def test_stretch_to_uint8_is_the_tools_expression_bit_for_bit():
    rng = np.random.default_rng(2)
    cases = ((rng.normal(30000, 3000, (64, 64)).astype(np.float32), 25000.0, 35000.0, 1e-6),
             (rng.normal(120, 40, (64, 64)), 30.0, 200.0, 1),
             (np.full((4, 4), 7.0), 7.0, 7.0, 1e-6),
             (np.full((4, 4), 7.0), 6.5, 7.0, 1))
    for img, lo, hi, m in cases:
        ref = np.clip((img - lo) / max(hi - lo, m) * 255, 0, 255).astype(np.uint8)
        out = features.stretch_to_uint8(img, lo, hi, m)
        assert out.dtype == np.uint8 and np.array_equal(out, ref)
    out = features.stretch_to_uint8(np.array([0.0, 50.0, 100.0, 150.0, 200.0]), 50, 150)
    assert out.tolist() == [0, 0, 127, 255, 255]             # 127.5 truncates to 127; the ends clip
