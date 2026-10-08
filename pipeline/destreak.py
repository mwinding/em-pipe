"""Curtaining (stripe) removal with the combined wavelet-FFT filter of Münch et al. 2009.

FIB milling leaves stripes along the milling direction. In a wavelet decomposition they
sit in one detail band per level (vertical stripes, constant along y, in pywt's cV);
damping the low frequencies of that band along the stripe direction removes them while
leaving other structure largely intact. ``destreak`` is applied per tile slice inside
render; ``python -m pipeline.destreak test`` writes before/after images for tuning.
"""

import logging
import os
import sys

import cv2
import numpy as np
import pywt
import scipy.fft
from scipy import ndimage

from .cli import atomic_write, base_parser, setup
from .config import step_dir
from .features import downsample, to_uint8
from .slices import StackCache, load_slices

log = logging.getLogger(__name__)

DEFAULTS = {
    "destreak": {
        "enabled": False,       # apply inside render
        "direction": "vertical",  # stripes run along y (vertical) or along x (horizontal)
        "wavelet": "db4",       # pywt wavelet name
        "levels": 6,            # wavelet decomposition levels (capped by the image size)
        "sigma": 4.0,           # width of the Gaussian notch in FFT bins; larger removes more
    },
}

# direction -> (index of the detail band in pywt's (cH, cV, cD), axis along which stripes are constant)
_BANDS = {"vertical": (1, 0), "horizontal": (0, 1)}

# FFT threads: the CPUs this process may use (Slurm's allocation), not every core on the node.
_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def destreak(img, direction="vertical", wavelet="db4", levels=6, sigma=4.0):
    """Remove stripes from a 2D image; returns float32 of the same shape and mean."""
    if direction not in _BANDS:
        raise ValueError(f"direction must be one of {sorted(_BANDS)}, got {direction!r}")
    band, axis = _BANDS[direction]
    h, w = np.shape(img)
    mean = float(np.mean(img, dtype=np.float64))
    levels = min(int(levels), pywt.dwt_max_level(min(h, w), wavelet))
    if levels < 1:  # smaller than the wavelet filter: nothing to separate
        return np.array(img, np.float32)
    # No float32 copy of the input is kept alive: peak memory is about 3.3x a float32 tile.
    coeffs = pywt.wavedec2(np.asarray(img, np.float32), wavelet, level=levels)
    for i in range(1, len(coeffs)):
        details = list(coeffs[i])
        c = details[band]
        f = scipy.fft.rfft(c, axis=axis, workers=_WORKERS)
        k = np.arange(f.shape[axis], dtype=np.float32)
        damp = 1 - np.exp(-k ** 2 / (2 * sigma ** 2))
        f *= damp[:, None] if axis == 0 else damp[None, :]
        details[band] = scipy.fft.irfft(f, n=c.shape[axis], axis=axis, workers=_WORKERS).astype(np.float32, copy=False)
        del f, c
        coeffs[i] = tuple(details)
    out = pywt.waverec2(coeffs, wavelet)
    del coeffs
    out = np.ascontiguousarray(out[:h, :w], np.float32)
    # Damping touches only detail bands, so the mean shifts only through boundary effects.
    out += np.float32(mean - float(out.mean(dtype=np.float64)))
    return out


def destreak_from_cfg(img, cfg):
    """``destreak`` with the parameters in ``cfg['destreak']`` (the caller checks ``enabled``)."""
    p = {**DEFAULTS["destreak"], **(cfg.get("destreak") or {})}
    return destreak(img, p["direction"], p["wavelet"], p["levels"], p["sigma"])


def stripe_amplitude(img, direction="vertical", width=15):
    """Std of the high-passed mean profile across the stripes (a stripe-strength score)."""
    _, axis = _BANDS[direction]
    profile = np.asarray(img, np.float64).mean(axis=axis)
    return float(np.std(profile - ndimage.uniform_filter1d(profile, width, mode="reflect")))


def _write_png(path, img):
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write {path}")


def _log_spectrum(img):
    """Log magnitude spectrum, DC centred."""
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))))


def make_test_images(cfg, args):
    df = load_slices(cfg, include_excluded=True, apply_selection=False)
    rows = df[(df["z"] == args.z) & (df["tile"] == args.tile)]
    if rows.empty:
        raise SystemExit(f"no slice z={args.z} tile={args.tile} in check/slices.csv")
    row = rows.iloc[0]
    h, w = int(row["height"]), int(row["width"])
    if args.crop:
        y0, y1, x0, x1 = args.crop
        y0, y1, x0, x1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
        if y1 - y0 < 2 or x1 - x0 < 2:
            raise SystemExit(f"--crop {' '.join(map(str, args.crop))} is outside the {h} x {w} tile")
    else:  # centred 1024 px crop
        y0, x0 = max(0, h // 2 - 512), max(0, w // 2 - 512)
        y1, x1 = min(h, y0 + 1024), min(w, x0 + 1024)
    params = dict(cfg["destreak"])
    for key in ("direction", "wavelet", "levels", "sigma"):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    before = StackCache(cfg["raw_dir"]).read(row).astype(np.float32)
    after = destreak(before, params["direction"], params["wavelet"], params["levels"], params["sigma"])
    log.info("z=%d tile %s %s: stripe amplitude %.2f -> %.2f", args.z, args.tile, params,
             stripe_amplitude(before, params["direction"]), stripe_amplitude(after, params["direction"]))

    factor = max(1.0, max(h, w) / 2000)
    # Same contrast for before and after, in the images and in the spectra.
    lo, hi = np.percentile(before, (0.5, 99.5))
    spectra = {"before": _log_spectrum(before[y0:y1, x0:x1]), "after": _log_spectrum(after[y0:y1, x0:x1])}
    flo, fhi = np.percentile(spectra["before"], (1, 99.9))
    stem = f"z{args.z}_tile{args.tile}"
    for name, img in (("before", before), ("after", after)):
        images = {
            f"{stem}_{name}.png": to_uint8(downsample(img, factor), lo, hi),
            f"{stem}_{name}_crop.png": to_uint8(img[y0:y1, x0:x1], lo, hi),
            f"{stem}_{name}_fft.png": to_uint8(spectra[name], flo, fhi),
        }
        for fname, im in images.items():
            atomic_write(step_dir(cfg, "destreak", fname), lambda p, im=im: _write_png(p, im))
    log.info("wrote %s/%s_*.png (crop y %d:%d x %d:%d)", step_dir(cfg, "destreak"), stem, y0, y1, x0, x1)


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("command", choices=["test"])
    p.add_argument("--z", type=int, required=True, help="global slice index")
    p.add_argument("--tile", required=True, help="tile id r-c")
    p.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"),
                   help="full-resolution crop (default: centred 1024 px)")
    p.add_argument("--direction", choices=sorted(_BANDS), help="override destreak.direction")
    p.add_argument("--wavelet", help="override destreak.wavelet")
    p.add_argument("--levels", type=int, help="override destreak.levels")
    p.add_argument("--sigma", type=float, help="override destreak.sigma")
    args = p.parse_args(argv)
    cfg = setup(args, DEFAULTS)
    make_test_images(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
