"""Background masks: the large, uniform off-sample areas of a tile slice (empty resin, dark areas
outside the sample), left out of the statistics that set display levels.

Works on an 8-bit image scaled to the slice's own 0.5-99.5 percentile range, so it doesn't depend
on a dataset's absolute grey levels: background is near either end of that range (after light
smoothing), has little texture (striped resin still counts), and forms connected areas of at least
``min_px`` pixels. How much of a tile is background differs between datasets, tiles and fields of
view; statistics without it give the same look whatever the amount.
"""

import numpy as np
from scipy import ndimage

BRIGHT = 235     # smoothed grey (of 255) above which a uniform area is background
DARK = 20        # ... and below which
MAX_SD = 10.0    # local standard deviation (greys) below which an area counts as uniform
MIN_PX = 400     # smallest background area, in pixels of the image passed in
GROW_PX = 3      # background grown by this much: the smoothed edge of an area would leave a rim


def mask(img8, bright=BRIGHT, dark=DARK, max_sd=MAX_SD, min_px=MIN_PX, grow_px=GROW_PX):
    """Boolean background mask of an 8-bit image scaled to its own 0.5-99.5 percentiles."""
    t = np.asarray(img8, np.float32)
    sm = ndimage.gaussian_filter(t, 1.5)
    sd = np.sqrt(np.maximum(ndimage.gaussian_filter(t * t, 3) - ndimage.gaussian_filter(t, 3) ** 2, 0))
    bg = ndimage.binary_opening(((sm > bright) | (sm < dark)) & (sd < max_sd), iterations=2)
    lab, n = ndimage.label(bg)
    if n:
        sizes = np.asarray(ndimage.sum(bg, lab, range(1, n + 1)))
        bg = np.isin(lab, 1 + np.flatnonzero(sizes >= min_px))
    if grow_px and bg.any():
        bg = ndimage.binary_dilation(bg, iterations=int(grow_px))
    return bg
