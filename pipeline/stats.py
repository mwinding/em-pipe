"""Small numeric helpers shared by the pipeline steps and the tools."""

import numpy as np


def robust_sd(a):
    """Robust standard deviation: 1.4826 x the median absolute deviation from the median (the standard
    deviation for normally distributed values; outliers, while a minority, hardly move it).

    Computed in ``a``'s own dtype (float32 values give a float32 result) and with no special cases: a NaN
    gives NaN, and no values give NaN with numpy's empty-slice warning, so callers that want something
    else for those check first."""
    a = np.asarray(a)
    return 1.4826 * np.median(np.abs(a - np.median(a)))
