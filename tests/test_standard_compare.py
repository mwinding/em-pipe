"""tools/standard_compare.py: other widths as a stretch around the median grey, and whole-volume clipping
from the tissue quantiles."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("standard_compare", ROOT / "tools" / "standard_compare.py")
standard_compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(standard_compare)

PCTS = [0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9, 99.99]


def test_other_width_stretches_around_the_median():
    g = np.array([0.0, 80.0, 150.0, 220.0, 255.0])
    out = standard_compare.other_width(g, 150, 140, 130)
    assert out[2] == 150                                   # the median grey stays
    assert np.allclose(out[1:4], 150 + (g[1:4] - 150) * 130 / 140)
    assert np.allclose(standard_compare.other_width(g, 150, 140, 140), g)
    assert standard_compare.other_width(np.array([-40.0, 300.0]), 150, 140, 140).tolist() == [0, 255]


def test_volume_clipping_matches_the_window_and_falls_with_width():
    """Tissue ~ normal(30000, 1500) per slice; neuropil median 30000, p5-p95 width 3000 raw; levels at width
    140 put it at grey 150: the clipped fractions are the normal's tails beyond the window's ends, and a
    smaller width (less contrast) clips less."""
    mu, sd = 30000.0, 1500.0
    row = {"median": mu, "rq_5": mu - 1500, "rq_95": mu + 1500, "grey_new": 150.0, "contrast_new": 140.0}
    row.update({f"tq_{p:g}": norm.ppf(p / 100, mu, sd) for p in PCTS})
    s = pd.DataFrame([row] * 5)
    out = standard_compare.volume_clipping(s, 150, 140, [130, 140])
    span = 255 * 3000 / 140                                # raw per 255 greys at width 140
    lo, hi = mu - 150 * span / 255, mu + 105 * span / 255
    # to within the linear interpolation between the stored percentiles (as the QC's own estimate)
    assert abs(out[140]["black_median"] - norm.cdf(lo, mu, sd)) < 0.005
    assert abs(out[140]["white_median"] - norm.sf(hi, mu, sd)) < 0.005
    assert out[130]["black_median"] < out[140]["black_median"]
    assert out[130]["white_median"] < out[140]["white_median"]
