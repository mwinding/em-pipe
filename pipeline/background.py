"""Background masks: the large off-sample areas of a tile slice (empty resin, also with shading
gradients or curtaining stripes; dark areas), left out of the statistics that set display levels.

Background is where the image has no fine texture along y: the image minus its smoothed version
along y (which removes shading gradients, and leaves curtaining stripes, which run along y, flat)
has a local rms below ``max_texture`` greys, over connected areas of at least ``min_px`` pixels.
Tissue has fine texture in every direction. Brightness plays no part, so a dataset's grey levels,
the resin's shading and how much of a tile is off-sample don't change what counts as tissue.
Works on an 8-bit image scaled to the slice's own 0.5-99.5 percentile range (the preview
thumbnail); 35h, tuned by eye on 30 tile slices: resin with gradients and stripes caught, no pale
tissue lost (an earlier brightness-based mask missed shaded and striped resin).
"""

import numpy as np
from scipy import ndimage

MAX_TEXTURE = 3.0   # local rms (greys) of the fine texture along y below which an area is background
SMOOTH_Y = 6.0      # sigma (px) of the smoothing along y whose residual is the fine texture
MIN_PX = 1500       # smallest background area, in pixels of the image passed in
GROW_PX = 3         # background grown by this much: the smoothed edge of an area would leave a rim


def mask(img8, max_texture=MAX_TEXTURE, min_px=MIN_PX, grow_px=GROW_PX):
    """Boolean background mask of an 8-bit image scaled to its own 0.5-99.5 percentiles."""
    t = np.asarray(img8, np.float32)
    fine = t - ndimage.gaussian_filter1d(t, SMOOTH_Y, axis=0)
    rms = np.sqrt(ndimage.gaussian_filter(fine * fine, 3))
    bg = ndimage.binary_opening(rms < max_texture, iterations=2)
    lab, n = ndimage.label(bg)
    if n:
        sizes = np.asarray(ndimage.sum(bg, lab, range(1, n + 1)))
        bg = np.isin(lab, 1 + np.flatnonzero(sizes >= min_px))
    if grow_px and bg.any():
        bg = ndimage.binary_dilation(bg, iterations=int(grow_px))
    return bg
