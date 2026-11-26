"""Curtaining (stripe) removal: the combined wavelet-FFT filter of Münch et al. 2009, or, given a
second detector's image of the same scan that has no curtains, a swap of the stripe part from it.

FIB milling leaves stripes along the milling direction. In a wavelet decomposition they sit in
one detail band per level (vertical stripes, constant along y, in pywt's cV). The filter takes out
the low frequencies of that band along the stripe direction (its "stripe part"). That removes the
stripes, but also the image's own structure that runs along them, which it cannot tell apart.

With ``reference_from`` / ``reference_to`` (how to get from a raw file's path to the same scan from
another detector, e.g. a backscatter detector, which sees composition rather than the surface
relief that makes curtains) the stripe part is not removed but replaced by the reference's stripe
part, scaled by a local gain fitted on the rest of the band, where both detectors see the same
structure. The image's own structure along the stripes comes back from the reference; everything
else stays the image's own. No curtain mask is needed: where there are no curtains the two stripe
parts agree and the swap changes little. With ``core_threshold``, the cores of strong curtains (a
few px wide, where the swap took out the most) then take the reference matched to the image
locally: a curtain that wanders sideways from row to row is not a stripe at any scale, so the
swap leaves a faint jagged trace of it. With ``band_match``, the broad band a cluster of curtains
leaves after the swap is matched to the reference in curtain zones, fading to nothing at their edges.
The band is the curtains' own darkness: wavelet details average to 0, so a dark line's total
darkness is in no detail band but in the coarse remainder, and once the swap has taken out the
curtains' shape it is left spread over a few hundred px (the tissue between the curtains is not
darker in the raw image).

``stripe_px`` sets what counts as a stripe by length (structure varying more slowly than that along
the stripes), the same at every image size; ``sigma`` (notch width in FFT bins over the band's
length) is the original setting. Bands are reflected at their ends before filtering, so the top and
bottom rows of a tile don't mix. ``destreak`` / ``swap`` run per tile slice inside render;
``python -m pipeline.destreak test`` writes before/after images for tuning.
"""

import logging
import os
import re
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
        "enabled": True,        # apply inside render and refstats
        "direction": "vertical",  # stripes run along y (vertical) or along x (horizontal); pipeline/curtains.py
                                  # measures it on sampled raw tiles
        # reference: swap from the reference detector (reference_from / reference_to must be set) | filter: the
        # filter-only path, stated as a choice (no reference may be set) | null: the swap when a reference is
        # set, else the filter, with a warning: the filter also removes the image's own structure along the
        # stripes, so it should be chosen, not fallen back to
        "mode": "reference",
        "wavelet": "db4",       # pywt wavelet name
        "levels": 8,            # wavelet decomposition levels (capped by the image size); level L holds
                                # structure ~2^L px across, so 8 reaches bands ~500 px wide
        "sigma": 4.0,           # width of the Gaussian notch in FFT bins; larger removes more
        "stripe_px": 256,       # instead of sigma: what varies more slowly than this along the stripes
                                # (full-resolution px) is stripe, at any image size (e.g. 256)
        "stripe_aspect": None,  # and at least this many times longer than wide (level L: ~2^L px wide), so
                                # at coarse levels only long bands count, not cell-sized structure (e.g. 8)
        "reference_from": None,  # regex on a raw file's path (relative to raw_dir) and its replacement
        "reference_to": None,    # (re.sub): the same scan from a detector without the stripes. Then the
                                 # stripe part is swapped from it (scaled by a local gain), not removed
        "gain_sigma": 8.0,      # swap: local gain window, Gaussian sigma in coefficients of each level
        "core_threshold": 2.0,   # swap: where the fine structure it took out (persisting along the
                                 # stripes) exceeds this many times the image's pixel noise (strong
                                 # curtain cores), the reference matched to the image locally fills in
        "core_dilate_px": 2,    # cores widened by this many px across the stripes (curtains wander)
        "core_window_px": 31,   # window of the local linear match reference -> image
        "core_denoise_px": 1.0,  # Gaussian sigma smoothing the reference before it fills cores
        "band_match": True,     # swap: the broad band a curtain cluster's own darkness leaves after the swap
                                # matched to the reference in curtain zones, fading to 0 at their edges
        "band_zone": 1.0,       # curtain zone: what the swap took out (smoothed 64 px across / 256 px along
                                # the stripes) above this many times the image's pixel noise...
        "band_zone_px": 256,    # ...widened by this many px (to take in the band's flanks)
        "band_along_px": 512,   # band = image - matched reference, smoothed this much along the stripes
                                # (a band persists; the detectors' differing tissue contrast averages out)...
        "band_across_px": 32,   # ...and this much across them
        "band_max_px": 768,     # take only bands up to this wide across the stripes; wider differences
                                # (regions the detectors show differently) are left (e.g. 768)
    },
}

MODES = (None, "reference", "filter")

# direction -> (index of the detail band in pywt's (cH, cV, cD), axis along which stripes are constant)
_BANDS = {"vertical": (1, 0), "horizontal": (0, 1)}

# FFT threads: the CPUs this process may use (Slurm's allocation), not every core on the node.
_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def _sigma(n, level, sigma, stripe_px, aspect=None):
    """Notch width in FFT bins for a band of n coefficients along the stripes at this level: from the
    shortest stripe, max(stripe_px, aspect * 2^level) px, else ``sigma`` as given."""
    if stripe_px is None and aspect is None:
        return float(sigma)
    length = max(float(stripe_px or 0), float(aspect or 0) * 2 ** level)
    return n / (length / 2 ** level)


def _keep(m, sigma):
    """Gains of the stripe low-pass over the m rfft bins of a band reflected to twice its length."""
    k = np.arange(m, dtype=np.float32)
    return np.exp(-k ** 2 / (2 * (2 * sigma) ** 2))   # twice the length: twice the bins for the same cutoff


def _stripe_part(c, axis, sigma):
    """What a detail band holds that varies slowly along the stripes: a Gaussian low-pass of width
    ``sigma`` bins (over the band's length) along ``axis``, after reflecting the band at its end so
    that its first and last rows (columns) don't mix."""
    n = c.shape[axis]
    f = scipy.fft.rfft(np.concatenate([c, np.flip(c, axis=axis)], axis=axis), axis=axis, workers=_WORKERS)
    keep = _keep(f.shape[axis], sigma)
    f *= keep[:, None] if axis == 0 else keep[None, :]
    out = scipy.fft.irfft(f, n=2 * n, axis=axis, workers=_WORKERS)
    return np.ascontiguousarray(out[:n] if axis == 0 else out[:, :n], np.float32)


def _rest_noise_share(n, sigma):
    """Share of white noise power left in a band of n coefficients after its stripe part is taken out."""
    return float(np.mean((1 - _keep(n + 1, sigma)) ** 2))


def _check(direction, wavelet, levels, h, w):
    if direction not in _BANDS:
        raise ValueError(f"direction must be one of {sorted(_BANDS)}, got {direction!r}")
    return min(int(levels), pywt.dwt_max_level(min(h, w), wavelet))


def _finish(coeffs, wavelet, h, w, mean):
    out = pywt.waverec2(coeffs, wavelet)
    out = np.ascontiguousarray(out[:h, :w], np.float32)
    # Only detail bands change, so the mean shifts only through boundary effects.
    out += np.float32(mean - float(out.mean(dtype=np.float64)))
    return out


def destreak(img, direction="vertical", wavelet="db4", levels=6, sigma=4.0, stripe_px=None, aspect=None):
    """Remove stripes from a 2D image; returns float32 of the same shape and mean."""
    h, w = np.shape(img)
    levels = _check(direction, wavelet, levels, h, w)
    if levels < 1:  # smaller than the wavelet filter: nothing to separate
        return np.array(img, np.float32)
    band, axis = _BANDS[direction]
    mean = float(np.mean(img, dtype=np.float64))
    # No float32 copy of the input is kept alive: peak memory is about 3.3x a float32 tile.
    coeffs = pywt.wavedec2(np.asarray(img, np.float32), wavelet, level=levels)
    n = len(coeffs) - 1
    for i in range(1, n + 1):
        details = list(coeffs[i])
        c = details[band]
        details[band] = c - _stripe_part(c, axis, _sigma(c.shape[axis], n - i + 1, sigma, stripe_px, aspect))
        coeffs[i] = tuple(details)
    return _finish(coeffs, wavelet, h, w, mean)


def noise_sd(img, wavelet="db4"):
    """Robust SD of an image's pixel noise: the MAD of its finest diagonal wavelet band (Donoho)."""
    cd = pywt.dwt2(np.asarray(img, np.float32), wavelet)[1][2]
    return float(np.median(np.abs(cd)) / 0.6745)


def _blur(a, s):
    return cv2.GaussianBlur(a, (0, 0), float(s))


def swap(img, ref, direction="vertical", wavelet="db4", levels=6, sigma=4.0, stripe_px=None, gain_sigma=8.0,
         aspect=None):
    """Stripes of ``img`` replaced from ``ref`` (the same scan, another detector, without them).

    In each level's stripe band, img's stripe part (``_stripe_part``) is swapped for ref's times a
    local gain g = <r_i r_e> / (<r_e r_e> - noise) fitted on the rest of the band (r_i, r_e: band
    minus stripe part; Gaussian window ``gain_sigma`` coefficients). The noise term (ref's pixel noise,
    which orthonormal wavelets carry into every band unchanged, times the share the rest of the band
    keeps) stops a noisy reference from biasing the gain low. Returns float32 of img's shape and mean.
    """
    h, w = np.shape(img)
    if np.shape(ref) != (h, w):
        raise ValueError(f"reference shape {np.shape(ref)} differs from the image's {(h, w)}")
    levels = _check(direction, wavelet, levels, h, w)
    if levels < 1:
        return np.array(img, np.float32)
    band, axis = _BANDS[direction]
    mean = float(np.mean(img, dtype=np.float64))
    ref = np.asarray(ref, np.float32)
    noise = noise_sd(ref, wavelet) ** 2 if pywt.Wavelet(wavelet).orthogonal else 0.0
    ci = pywt.wavedec2(np.asarray(img, np.float32), wavelet, level=levels)
    ce = pywt.wavedec2(ref, wavelet, level=levels)
    n = len(ci) - 1
    for i in range(1, n + 1):
        details = list(ci[i])
        c, e = details[band], ce[i][band]
        s = _sigma(c.shape[axis], n - i + 1, sigma, stripe_px, aspect)
        sc, se = _stripe_part(c, axis, s), _stripe_part(e, axis, s)
        rc, re_ = c - sc, e - se
        num, den = _blur(rc * re_, gain_sigma), _blur(re_ * re_, gain_sigma)
        # Keep at least a tenth of the local power so pure-noise areas (resin) don't blow the gain up.
        g = num / np.maximum(den - noise * _rest_noise_share(c.shape[axis], s), 0.1 * den + 1e-12)
        details[band] = rc + g * se
        ci[i] = tuple(details)
        ce[i] = None
    return _finish(ci, wavelet, h, w, mean)


def fill_cores(img, out, ref, threshold, direction="vertical", dilate_px=2, window_px=31, denoise_px=1.0,
               wavelet="db4"):
    """Curtain cores of ``out`` (= swap(img, ref)) filled from ``ref`` matched to ``out`` locally.

    Cores: where what the swap took out (img - out), high-passed across the stripes (minus its
    4 px Gaussian across them, so broad bands don't count) and averaged along them (16 px Gaussian:
    a curtain persists, texture doesn't), exceeds ``threshold`` times img's pixel noise; widened
    ``dilate_px`` across the stripes and softened by 1 px. There ``ref`` (smoothed by ``denoise_px``)
    goes through a local linear map a * ref + b fitted to ``out`` around the cores (box ``window_px``,
    core pixels left out; the slope corrected for ref's noise as in ``swap``).
    """
    _, axis = _BANDS[direction]
    across = 1 - axis
    removed = np.asarray(img, np.float32) - out
    fine = removed - ndimage.gaussian_filter1d(removed, 4.0, axis=across)
    strength = ndimage.gaussian_filter1d(np.abs(fine), 16.0, axis=axis)
    core = (strength > threshold * noise_sd(img, wavelet)).astype(np.uint8)
    if not core.any():
        return out
    if dilate_px:
        size = 2 * int(dilate_px) + 1
        core = cv2.dilate(core, np.ones((1, size) if across == 1 else (size, 1), np.uint8))
    m = np.clip(_blur(core.astype(np.float32), 1.0), 0, 1)
    m[core > 0] = 1.0
    ref = np.asarray(ref, np.float32)
    e = _blur(ref, denoise_px) if denoise_px else ref
    # Noise left in e: ref's pixel noise times the smoothing kernel's sum of squares.
    if denoise_px:
        g = cv2.getGaussianKernel(int(8 * denoise_px) | 1, denoise_px)
        kk = float((g ** 2).sum()) ** 2
    else:
        kk = 1.0
    noise = noise_sd(ref, wavelet) ** 2 * kk
    wgt = (core == 0).astype(np.float32)
    box = lambda a: cv2.boxFilter(a, -1, (window_px, window_px), normalize=False)
    sw = box(wgt)
    # Fill only where at least a quarter of the window is outside the cores to fit the match on.
    m *= sw >= 0.25 * window_px ** 2
    sw += 1e-6
    me, mo = box(wgt * e) / sw, box(wgt * out) / sw
    cov = box(wgt * e * out) / sw - me * mo
    var = box(wgt * e * e) / sw - me * me
    a = cov / np.maximum(var - noise, 0.1 * var + 1e-12)
    matched = a * e + (mo - a * me)
    return ((1 - m) * out + m * matched).astype(np.float32)


def _harmonic_fill(v, zone):
    """v with its values inside ``zone`` (bool) replaced by the harmonic interpolation of its values on the
    zone's edge (Laplace equation, values outside the zone fixed; at the image's border the gradient across it
    is 0). A zone covering the whole image has no edge: its fill is v's least-squares plane (like the harmonic
    fill, it keeps offsets and gradients)."""
    hh, ww = v.shape
    ys, xs = np.nonzero(zone)
    n = len(ys)
    if not n:
        return np.array(v, np.float64)
    if n == v.size:
        yy, xx = np.mgrid[:hh, :ww].astype(np.float64)
        basis = np.stack([np.ones(v.size), yy.ravel(), xx.ravel()], axis=1)
        coef = np.linalg.lstsq(basis, np.asarray(v, np.float64).ravel(), rcond=None)[0]
        return (basis @ coef).reshape(hh, ww)
    idx = -np.ones((hh, ww), np.int64)
    idx[ys, xs] = np.arange(n)
    rows, cols, vals, rhs, diag = [np.arange(n)], [np.arange(n)], [], np.zeros(n), np.zeros(n)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yy, xx = ys + dy, xs + dx
        inside = (yy >= 0) & (yy < hh) & (xx >= 0) & (xx < ww)
        diag += inside
        yy, xx = np.clip(yy, 0, hh - 1), np.clip(xx, 0, ww - 1)
        nb = idx[yy, xx]
        un = inside & (nb >= 0)
        rows.append(np.nonzero(un)[0])
        cols.append(nb[un])
        vals.append(-np.ones(int(un.sum())))
        edge = inside & (nb < 0)
        rhs[edge] += v[yy[edge], xx[edge]]
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    A = sp.csc_matrix((np.concatenate([diag] + vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    out = np.array(v, np.float64)
    out[ys, xs] = spl.spsolve(A, rhs)
    return out


def _sigmas(axis, along, across, factor):
    s = [0.0, 0.0]
    s[axis], s[1 - axis] = along / factor, across / factor
    return s


def match_bands(img, out, ref, threshold=1.0, zone_px=256, along_px=512.0, across_px=32.0, direction="vertical",
                wavelet="db4", factor=32, max_px=None):
    """The broad band a curtain cluster leaves in ``out`` (= swap(img, ref)), matched to ``ref`` in curtain
    zones, seamlessly.

    The band is the curtains' own darkness. Wavelet details average to 0, so a dark line's total darkness is in
    no detail band but in the coarse remainder; the swap takes out the curtains' shape and leaves their darkness
    spread over a few hundred px, wider than the swap's levels reach. (In the raw image the tissue between the
    curtains is not darker: the image averaged over a cluster is as dark before the swap as after it, and
    filling the strong cores from the reference first takes most of the band away.) Curtain zones: what the swap took out, |img - out|, smoothed (Gaussian 64 px
    across the stripes, 256 px along) above ``threshold`` times img's pixel noise, widened ``zone_px``. The
    band: out - (a ref + b), a and b fitted outside the zones (on all of the image if they leave less than a
    tenth), smoothed ``along_px`` along the stripes (a band persists; tissue the two detectors show with
    different contrast averages out) and ``across_px`` across. Inside a zone out loses the band minus its
    harmonic interpolation from the zone's edge (Poisson / seamless blending): the correction is 0 at the
    edge, and smooth differences between the detectors (offsets, gradients) cancel. Outside, out is unchanged.
    All but the last step runs on a ``factor``-times smaller grid.

    A curtain cluster's band is a few hundred px wide; a region the detectors show differently (a cluster of
    cell bodies, say) can be wider, and taking it would give it the reference's brightness. With ``max_px``
    only bands up to that wide across the stripes are taken: the band minus its running median across them
    over twice that width (the median ignores anything narrower than ``max_px`` and follows anything wider).
    Unlike a linear band-pass (or a morphological top-hat, which measures from the bright bumps either side)
    the median gives a narrow band no side lobes.

    Swapping deeper wavelet levels in the zones instead brings their side lobes: wavelets average to 0, so a
    dark band's coarse levels are flanked by lighter ones, and the blend, pinned to them at the zone's edge,
    lifted whole zones lighter than the reference. The band here is a weighted average, without lobes.
    """
    _, axis = _BANDS[direction]
    h, w = np.shape(img)
    small = (max(1, w // factor), max(1, h // factor))
    down = lambda a: cv2.resize(np.asarray(a, np.float32), small, interpolation=cv2.INTER_AREA)
    dens = ndimage.gaussian_filter(down(np.abs(np.asarray(img, np.float32) - out)), _sigmas(axis, 256, 64, factor))
    zone = dens > float(threshold) * noise_sd(img, wavelet)
    r = int(round(zone_px / factor))
    if r > 0:
        zone = ndimage.binary_dilation(zone, np.ones((2 * r + 1, 2 * r + 1), bool))
    if not zone.any():
        return out
    o, e = down(out).astype(np.float64), down(ref).astype(np.float64)
    fit = ~zone if (~zone).mean() >= 0.1 else np.ones_like(zone)
    ef, of = e[fit], o[fit]
    a = float(np.mean((ef - ef.mean()) * (of - of.mean())) / (ef.var() + 1e-12))
    b = float(of.mean() - a * ef.mean())
    band = ndimage.gaussian_filter(o - (a * e + b), _sigmas(axis, along_px, across_px, factor))
    if max_px:
        k = max(3, int(round(2 * float(max_px) / factor)) | 1)
        band = band - ndimage.median_filter(band, size=(1, k) if axis == 0 else (k, 1), mode="reflect")
    corr = np.where(zone, band - _harmonic_fill(band, zone), 0.0).astype(np.float32)
    return (out - cv2.resize(corr, (w, h), interpolation=cv2.INTER_LINEAR)).astype(np.float32)


def reference_file(file, cfg):
    """The reference detector's file for raw ``file`` (relative path) per destreak.reference_from/_to."""
    p = {**DEFAULTS["destreak"], **(cfg.get("destreak") or {})}
    new, n = re.subn(p["reference_from"], p["reference_to"], str(file))
    if not n or new == str(file):
        raise ValueError(f"destreak.reference_from {p['reference_from']!r} does not map {file!r} to another file")
    return new


def settings(cfg):
    """destreak's section with defaults, checked."""
    given = cfg.get("destreak") or {}
    unknown = set(given) - set(DEFAULTS["destreak"])
    if unknown:
        raise ValueError(f"destreak: unknown keys {sorted(unknown)}")
    p = {**DEFAULTS["destreak"], **given}
    if (p["reference_from"] is None) != (p["reference_to"] is None):
        raise ValueError("destreak.reference_from and destreak.reference_to go together")
    if p["direction"] not in _BANDS:
        raise ValueError(f"destreak.direction must be one of {sorted(_BANDS)}, got {p['direction']!r}")
    if p["mode"] not in MODES:
        raise ValueError(f"destreak.mode must be reference, filter or null, got {p['mode']!r}")
    if p["mode"] == "reference" and p["reference_from"] is None:
        raise ValueError("destreak.mode: reference needs destreak.reference_from / reference_to (how to get from a "
                         "raw file to its EsB file; the em-pipe command and `python -m pipeline.esb RAW_DIR` find "
                         "them). Without EsB data, state the choice: destreak.mode: filter, or destreak.enabled: false")
    if p["mode"] == "filter" and p["reference_from"] is not None:
        raise ValueError("destreak.mode: filter, but destreak.reference_from / reference_to are set: remove them, "
                         "or set mode: reference")
    for key in ("stripe_px", "stripe_aspect"):
        if p[key] is not None and not float(p[key]) > 0:
            raise ValueError(f"destreak.{key} must be > 0 or null, got {p[key]!r}")
    if p["band_match"]:
        if p["reference_from"] is None:
            raise ValueError("destreak.band_match needs a reference (destreak.reference_from / _to)")
        for key in ("band_zone", "band_along_px", "band_across_px"):
            if not float(p[key]) > 0:
                raise ValueError(f"destreak.{key} must be > 0, got {p[key]!r}")
        if not float(p["band_zone_px"]) >= 0:
            raise ValueError(f"destreak.band_zone_px must be >= 0, got {p['band_zone_px']!r}")
        if p["band_max_px"] is not None and not float(p["band_max_px"]) > 0:
            raise ValueError(f"destreak.band_max_px must be > 0 or null, got {p['band_max_px']!r}")
    if p["core_threshold"] is not None:
        if p["reference_from"] is None:
            raise ValueError("destreak.core_threshold needs a reference (destreak.reference_from / _to)")
        if not float(p["core_threshold"]) > 0:
            raise ValueError(f"destreak.core_threshold must be > 0 or null, got {p['core_threshold']!r}")
    return p


def uses_reference(cfg):
    return settings(cfg)["reference_from"] is not None


def destreak_from_cfg(img, cfg, ref=None):
    """destreak (or, with a reference configured, swap + fill_cores) per ``cfg['destreak']``; the caller
    checks ``enabled`` and reads ``ref`` (the reference detector's same slice and window)."""
    p = settings(cfg)
    if p["reference_from"] is None:
        return destreak(img, p["direction"], p["wavelet"], p["levels"], p["sigma"], p["stripe_px"], p["stripe_aspect"])
    if ref is None:
        raise ValueError("destreak has a reference configured but none was given")
    out = swap(img, ref, p["direction"], p["wavelet"], p["levels"], p["sigma"], p["stripe_px"], p["gain_sigma"],
               p["stripe_aspect"])
    if p["band_match"]:
        out = match_bands(img, out, ref, p["band_zone"], p["band_zone_px"], p["band_along_px"], p["band_across_px"],
                          p["direction"], p["wavelet"], max_px=p["band_max_px"])
    if p["core_threshold"] is not None:
        out = fill_cores(img, out, ref, float(p["core_threshold"]), p["direction"], p["core_dilate_px"],
                         p["core_window_px"], p["core_denoise_px"], p["wavelet"])
    return out


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
    for key in ("direction", "wavelet", "levels", "sigma", "stripe_px", "stripe_aspect"):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    cache = StackCache(cfg["raw_dir"])
    before = cache.read(row).astype(np.float32)
    dcfg = {"destreak": params}
    ref = None
    if uses_reference(dcfg):   # the whole tile, as render does: the swap needs the reference's same slice
        ref = cache.read({"file": reference_file(row["file"], dcfg), "index": row["index"]})
    after = destreak_from_cfg(before, dcfg, ref)
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
    p = base_parser("Curtain (stripe) removal test images: before/after PNGs of one tile slice (whole tile, crop, "
                    "spectrum) in work/destreak/, with the config's destreak settings or the overrides below. "
                    "In a run, destreak is applied inside render and refstats.")
    p.add_argument("command", choices=["test"])
    p.add_argument("--z", type=int, required=True, help="global slice index")
    p.add_argument("--tile", required=True, help="tile id r-c")
    p.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"),
                   help="full-resolution crop (default: centred 1024 px)")
    p.add_argument("--direction", choices=sorted(_BANDS), help="override destreak.direction")
    p.add_argument("--wavelet", help="override destreak.wavelet")
    p.add_argument("--levels", type=int, help="override destreak.levels")
    p.add_argument("--sigma", type=float, help="override destreak.sigma")
    p.add_argument("--stripe-px", dest="stripe_px", type=float, help="override destreak.stripe_px")
    p.add_argument("--stripe-aspect", dest="stripe_aspect", type=float, help="override destreak.stripe_aspect")
    args = p.parse_args(argv)
    cfg = setup(args, DEFAULTS)
    make_test_images(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
