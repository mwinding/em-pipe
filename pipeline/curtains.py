"""Curtain direction: are the FIB milling stripes of a dataset vertical or horizontal in the raw tiles?

destreak.direction has to match the stripes. Established tools take the direction as given: Janelia
render's streak correctors use a configured mask angle or assume vertical streaks, tomopy and pystripe
assume one axis, the published FIB-SEM curtain filters assume one direction. None of those detects it, so
the measure below is this pipeline's own design; it was checked on synthetic images (tests/test_curtains.py)
and on samples of real volumes with vertical curtains and the same tiles transposed.

What is measured, per window (a square of a raw tile, in two consecutive slices):

- the strength of straight stripes at every angle. A stripe pattern at angle a is what is left when the
  window is averaged along that angle (its profile across the stripes); by the projection-slice theorem the
  power of that profile is the power of the window's 2D spectrum along the line through the origin
  perpendicular to the stripes. Stripes long enough to cross the window give a peak about 1 degree sharp
  in that angle; tissue gives a level that changes slowly with angle. The peak is what stands above a running
  median over +-15 degrees.
- only what stays in place from one slice to the next counts: the power is taken from the cross-spectrum of
  the two consecutive slices (allowing a shift of up to ``max_lag`` px across the stripes, for drift).
  Curtains come from the material above the imaged face and stay at the same place over consecutive slices;
  scan-line noise changes with every slice and cancels.
- the stripes are looked for in the InLens window minus the gain-matched EsB window (the second detector,
  which hardly shows curtains). The tissue, which both detectors show, largely cancels, so curtains of about
  1 % of the contrast stand out; stripes that both detectors show alike cancel too (structure of the sample,
  a scan fault), and a peak that EsB alone shows as well does not count.

A window votes vertical (peak within ``tolerance`` degrees of the y axis), horizontal (of the x axis), at an
angle (anything else) or nothing (no peak above ``min_z`` robust standard deviations of the angle profile and
``min_amplitude`` of the window's contrast, or one that is not in both slices). The verdict needs a clear margin:

    vertical / horizontal   at least ``min_share`` of the windows (and 3) vote for it and at least 4 times as
                            many as for everything else: proceed with that direction
    unclear                 fewer windows than that show curtains at all: nothing to decide on; the configured
                            direction is kept, and that is said
    conflicting             otherwise (both directions, or curtains at an angle, which destreak cannot
                            remove): stop, a person has to look

The measure needs the EsB files. Without them the tissue's own straight structures are taken for stripes:
on two volumes with vertical curtains the measure without EsB gave "conflicting" (a quarter of the curtained
windows found, and peaks at other angles that are tissue). So only raw files that have their EsB file are
sampled, and without any EsB file the direction is not measured: destreak.direction stays a setting a person
states, and the message says so.

Checked on real data (26 Nov 2026, 64 windows each): three volumes with vertical curtains gave vertical
(61, 56 and 45 of 64 windows, none horizontal), and the same windows transposed gave horizontal. "Unclear"
has only been shown on synthetic windows without curtains: no stretch of those volumes is free of them by this
measure (the four files a coarser measure, tools/slice_quality.py, rates as least curtained still gave vertical
in 44 of 64 windows, at 0.8 % of the contrast; whether those are weak curtains was not checked by eye).

    python -m pipeline.curtains RAW_DIR [--reference-to PATTERN] [--files N] [--transpose]
    python -m pipeline.curtains --config CONFIG [...]          raw_dir, file pattern and reference from a config
Exit code 0 for vertical, horizontal and unclear, 2 for conflicting.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from .imagej_tiff import ImageJStack
from .stats import robust_sd

log = logging.getLogger(__name__)

SETTINGS = {
    "window": 2048,          # window size, px (smaller tiles: the largest power of two that fits)
    "min_period": 8.0,       # stripe profile band: periods across the stripes from this ...
    "max_period": 128.0,     # ... to this, px
    "angle_step": 0.5,       # degrees
    "baseline_deg": 15.0,    # the peak is what stands above the running median over +- this many degrees
    "max_lag": 16,           # px a stripe may move across between the two slices
    "tolerance": 1.5,        # degrees from an axis that still count as that axis
    "min_z": 8.0,            # a peak: this many robust standard deviations of the angle profile ...
    "min_amplitude": 0.002,  # ... and a stripe amplitude of this fraction of the window's p5-p95 width
    "min_coherence": 0.5,    # ... of which at least this share is in both slices at the same place
    "esb_ratio": 0.5,        # curtains: EsB alone shows less than this fraction of the peak
    "min_share": 0.05,       # verdict: this share of the windows (at least 3) must vote for a direction ...
    "margin": 4.0,           # ... and this many times more than for everything else
}
_HANN_GAIN = 0.5 ** 2 * 0.375   # a Hann-windowed square's mean profile keeps this share of a stripe's power


def _spectrum(img):
    a = np.asarray(img, np.float32)
    a = a - np.float32(a.mean(dtype=np.float64))
    n = a.shape[0]
    w = np.hanning(n).astype(np.float32)
    return np.fft.fft2(a * w[:, None] * w[None, :])


def _angles(step):
    return np.arange(-90.0, 90.0, float(step))


def _lines(n, s):
    """Sampling of an n x n spectrum (fftshifted) along the line of every angle: (coords for map_coordinates,
    radii in bins)."""
    r = np.arange(max(1, int(round(n / s["max_period"]))), int(round(n / s["min_period"])) + 1, dtype=np.float64)
    th = np.deg2rad(_angles(s["angle_step"]))
    # stripes at angle th from the y axis vary along the direction (cos th, sin th) in (x, y)
    kx, ky = r[None, :] * np.cos(th)[:, None], r[None, :] * np.sin(th)[:, None]
    return np.stack([ky + n // 2, kx + n // 2]), r


def angle_profile(fa, fb, s=SETTINGS):
    """Power (raw units squared) of the stripe profile that two consecutive slices of a window share, for
    stripes at each angle of ``_angles`` (0: along y, +-90: along x). fa, fb: the windows' ``_spectrum``."""
    n = fa.shape[0]
    cross = np.fft.fftshift(fa * np.conj(fb))
    coords, r = _lines(n, s)
    line = (ndimage.map_coordinates(cross.real, coords, order=1, mode="wrap")
            + 1j * ndimage.map_coordinates(cross.imag, coords, order=1, mode="wrap"))
    lags = np.arange(-int(s["max_lag"]), int(s["max_lag"]) + 1)
    phase = np.exp(2j * np.pi * np.outer(lags, r) / n)                 # (lags, r)
    xc = 2.0 * np.real(line @ phase.T)                                  # (angles, lags)
    return xc.max(axis=1) / (float(n) ** 4 * _HANN_GAIN)


def own_profile(f, s=SETTINGS):
    """As ``angle_profile`` for one slice with itself: the power of its stripe profile at each angle, whether
    or not the next slice has it too."""
    n = f.shape[0]
    power = np.fft.fftshift(f.real ** 2 + f.imag ** 2)
    coords, _ = _lines(n, s)
    return 2.0 * ndimage.map_coordinates(power, coords, order=1, mode="wrap").sum(axis=1) / (float(n) ** 4 * _HANN_GAIN)


def _above(profile, s=SETTINGS):
    """An angle profile minus its running median over +- baseline_deg: what is sharp in angle."""
    k = 2 * int(round(s["baseline_deg"] / float(s["angle_step"]))) + 1
    return profile - ndimage.median_filter(profile, size=k, mode="wrap")


def peak(profile, s=SETTINGS):
    """(index, angle, z, power above the baseline) of the sharpest peak of an angle profile; z in robust
    standard deviations of the profile above its baseline."""
    above = _above(profile, s)
    sd = robust_sd(above) + 1e-30
    i = int(np.argmax(above))
    return i, float(_angles(s["angle_step"])[i]), float(above[i] / sd), float(above[i])


def _width(img):
    lo, hi = np.percentile(np.asarray(img)[::4, ::4], (5, 95))
    return max(float(hi - lo), 1e-9)


def _near(angle, target, tol):
    d = abs(angle - target) % 180.0
    return min(d, 180.0 - d) <= tol


def _gain(a, e):
    """Least-squares gain of EsB window e onto InLens window a (both smoothed 3 px and thinned, so EsB's
    pixel noise does not bias it low): the tissue both detectors show."""
    sa = ndimage.gaussian_filter(np.asarray(a, np.float32), 3.0)[::4, ::4].astype(np.float64)
    se = ndimage.gaussian_filter(np.asarray(e, np.float32), 3.0)[::4, ::4].astype(np.float64)
    sa -= sa.mean()
    se -= se.mean()
    return float((sa * se).sum() / ((se * se).sum() + 1e-30))


def measure_window(a, b, ea=None, eb=None, s=SETTINGS):
    """One window's vote. a, b: the InLens window in two consecutive slices; ea, eb: the EsB windows or None.

    With EsB the stripes are looked for in InLens minus the gain-matched EsB window: the tissue, which both
    detectors show, largely cancels (and with it stripes they show in the same proportion as the tissue);
    what only InLens has is left. EsB's pixel noise differs from slice to slice and cancels in the
    cross-spectrum. Without EsB the tissue's own profile is the background, and only strong curtains stand
    out of it.

    A peak counts when it is sharp in angle (``min_z``), strong enough (``min_amplitude``) and in both slices:
    its shared power is at least ``min_coherence`` of the power each slice has there on its own (stripes that
    are new in every slice share none). With EsB, a peak that EsB alone shows as well (``esb_ratio`` of the
    InLens one) is not curtains.

    Returns {vote: vertical | horizontal | angle | none, angle, z, amplitude (of the stripe profile, as a
    fraction of the window's p5-p95 width), coherence, esb_amplitude (the peak in EsB alone; None without
    EsB, 0 when EsB has no peak there), why (for a peak that does not count)}."""
    fa, fb = _spectrum(a), _spectrum(b)
    out = {"angle": None, "z": 0.0, "amplitude": 0.0, "coherence": None, "esb_amplitude": None, "vote": "none"}
    if ea is not None:
        fea, feb = _spectrum(ea), _spectrum(eb)
        ra, rb = fa - _gain(a, ea) * fea, fb - _gain(b, eb) * feb
    else:
        ra, rb = fa, fb
    prof = angle_profile(ra, rb, s)
    above = _above(prof, s)
    sd = robust_sd(above) + 1e-30
    angles, width = _angles(s["angle_step"]), _width(a)
    own = eprof = None
    left = above.copy()
    for _ in range(4):   # the strongest peaks in turn: e.g. scan lines can outweigh curtains without hiding them
        i = int(np.argmax(left))
        power, z, angle = float(above[i]), float(above[i] / sd), float(angles[i])
        amp = math.sqrt(max(power, 0.0)) / width
        if out["angle"] is None:
            out.update(angle=angle, z=round(z, 2), amplitude=round(amp, 5))
        if z < s["min_z"] or amp < s["min_amplitude"]:
            return out
        near = [(i + d) % len(angles) for d in (-1, 0, 1)]
        if own is None:
            own = [_above(own_profile(f, s), s) for f in (ra, rb)]
        o = [max(float(x[near].max()), 0.0) for x in own]
        coherence = min(power / math.sqrt(o[0] * o[1]), 9.99) if min(o) > 0 else 0.0
        why = None
        if coherence < s["min_coherence"]:
            why = "not in both slices"
        elif ea is not None:
            if eprof is None:
                eprof = _above(angle_profile(fea, feb, s), s)
                esd = robust_sd(eprof) + 1e-30
            epower = float(eprof[near].max())
            eamp = math.sqrt(max(epower, 0.0)) / _width(ea) if epower / esd >= s["min_z"] else 0.0
            if eamp >= s["esb_ratio"] * amp:
                why = "EsB shows it too"
        if why is None:
            vote = ("vertical" if _near(angle, 0.0, s["tolerance"]) else
                    "horizontal" if _near(angle, 90.0, s["tolerance"]) else "angle")
            out.update(angle=angle, z=round(z, 2), amplitude=round(amp, 5), coherence=round(coherence, 3), vote=vote)
            out.pop("why", None)
            if ea is not None:
                out["esb_amplitude"] = round(eamp, 5)
            return out
        if "why" not in out:
            out.update(why=why, coherence=round(coherence, 3))
        k = int(round(20.0 / float(s["angle_step"])))     # look on, well away from this peak and its skirts
        left[[(i + d) % len(angles) for d in range(-k, k + 1)]] = -np.inf
    return out


def decide(windows, configured="vertical", s=SETTINGS):
    """The verdict from the windows' votes: {direction: vertical | horizontal | unclear | conflicting,
    use: the direction to run with (None when a person must decide), counts, message}."""
    n = len(windows)
    c = {v: sum(w["vote"] == v for w in windows) for v in ("vertical", "horizontal", "angle", "none")}
    need = max(3, math.ceil(s["min_share"] * n))
    amp = {v: [w["amplitude"] for w in windows if w["vote"] == v] for v in ("vertical", "horizontal", "angle")}
    med = lambda v: f"{100 * float(np.median(amp[v])):.2f} % of the contrast" if amp[v] else "-"
    why = {k: sum(w.get("why") == k for w in windows) for k in ("not in both slices", "EsB shows it too")}
    words = {"not in both slices": "with stripes that are not in both slices", "EsB shows it too": "with stripes that EsB shows too"}
    notes = [f"{v} {words[k]}" for k, v in why.items() if v]
    counts = (f"{c['vertical']} vertical, {c['horizontal']} horizontal, {c['angle']} at an angle, {c['none']} none "
              f"of {n} windows" + (f" (not counted: {'; '.join(notes)})" if notes else ""))
    out = {"counts": c, "windows": n, "needed": need, "not_counted": why}
    for d, other in (("vertical", "horizontal"), ("horizontal", "vertical")):
        if c[d] >= need and c[d] >= s["margin"] * (c[other] + c["angle"]):
            return {**out, "direction": d, "use": d,
                    "message": f"curtains are {d}: {counts}; median stripe amplitude {med(d)}"}
    if c["vertical"] + c["horizontal"] + c["angle"] < need:
        return {**out, "direction": "unclear", "use": configured,
                "message": f"curtain direction unclear, too few curtains to decide: {counts} (a direction needs "
                           f"{need}). The configured direction ({configured}) is kept; it was not measured"}
    angles = sorted(round(w["angle"], 1) for w in windows if w["vote"] == "angle")
    what = (f"curtains at an angle (median {float(np.median(angles)):.1f} degrees from vertical), which destreak "
            "cannot remove" if c["angle"] >= max(c["vertical"], c["horizontal"]) else "both directions occur")
    return {**out, "direction": "conflicting", "use": None,
            "message": f"curtain direction conflicting: {counts}; {what}. A person has to look at the raw tiles and "
                       "set destreak.direction (em-pipe: --direction vertical|horizontal), or run without "
                       "decurtaining (destreak.enabled: false; em-pipe: --no-esb)"}


# ----- sampling raw files ----------------------------------------------------------------------------

def plan(files, shapes, n_files=12, per_file=4, s=SETTINGS):
    """Windows to read: [(file, slice index, y0, x0, size)] from ``n_files`` files spread evenly over the sorted
    list (days and tiles alike), each at one place along its slices (a quarter, half, three quarters in turn)
    and ``per_file`` windows on a grid inside the tile."""
    usable = [f for f in sorted(files) if shapes[f][0] >= 2 and min(shapes[f][1:]) >= 256]
    if not usable:
        return []
    pick = sorted({int(round(i)) for i in np.linspace(0, len(usable) - 1, min(n_files, len(usable)))})
    side = max(1, int(round(math.sqrt(per_file))))
    out = []
    for k, i in enumerate(pick):
        f = usable[i]
        n, h, w = shapes[f]
        size = min(int(s["window"]), 2 ** int(math.log2(min(h, w))))
        index = min(n - 2, int(n * (0.25, 0.5, 0.75)[k % 3]))
        for gy in range(side):
            for gx in range(side):
                y0 = int((h - size) * (gy + 0.5) / side)
                x0 = int((w - size) * (gx + 0.5) / side)
                out.append((f, index, y0, x0, size))
    return out


def detect(raw_dir, files, reference=None, configured="vertical", n_files=12, per_file=4, transpose=False,
           index=None, s=SETTINGS):
    """Measure sampled windows of the raw files and decide. ``reference``: a function raw file -> EsB file
    (relative paths) or None. ``transpose``: every window transposed first (a check of the measure: vertical
    curtains must then come out horizontal). Returns ``decide``'s dict plus ``samples`` (one per window)."""
    raw_dir = Path(raw_dir)
    stacks = {}

    def stack(f):
        if f not in stacks:
            stacks[f] = ImageJStack(raw_dir / f)
        return stacks[f]

    with_esb = [f for f in files if reference is not None and reference(f) is not None]
    not_measured = ("curtain direction not measured: {why}. Without EsB the tissue's own straight structures are "
                    f"taken for stripes, so the measure is not used. The configured direction ({configured}) is "
                    "kept; it was not measured: a person has to look at the raw tiles and state destreak.direction")
    if not with_esb:
        return {"direction": "unclear", "use": configured, "counts": {}, "windows": 0, "samples": [],
                "message": not_measured.format(why="no raw file has an EsB file")}
    shapes = {}
    for f in with_esb:
        try:
            shapes[f] = stack(f).shape
        except Exception as e:   # unreadable files are check's business
            log.warning("%s: header not readable (%s): not sampled", f, e)
    samples = []
    rows_cache = {}

    def read(name, k, y0, x0, size):
        # whole rows in one read, then the columns: far fewer reads than one per row of a narrow strip
        key = (name, k, y0, size)
        if key not in rows_cache:
            if len(rows_cache) >= 8:
                rows_cache.clear()
            rows_cache[key] = stack(name).read(k, slice(y0, y0 + size), slice(None))
        return rows_cache[key][:, x0:x0 + size].astype(np.float32)

    for f, i, y0, x0, size in plan(list(shapes), shapes, n_files, per_file, s):
        if index is not None:
            i = max(0, min(shapes[f][0] - 2, int(index)))
        ref = reference(f)
        try:
            a, b = read(f, i, y0, x0, size), read(f, i + 1, y0, x0, size)
            ea, eb = read(ref, i, y0, x0, size), read(ref, i + 1, y0, x0, size)
        except Exception as e:
            log.warning("%s slice %d: window not readable (%s): not measured", f, i, e)
            continue
        if transpose:
            a, b, ea, eb = a.T, b.T, ea.T, eb.T
        m = measure_window(a, b, ea, eb, s)
        samples.append({"file": f, "index": int(i), "y0": y0, "x0": x0, "size": size, **m})
        log.info("%s slice %d window y %d x %d: %s (angle %.1f, z %.1f, amplitude %.2f %%, coherence %s%s)", f, i,
                 y0, x0, m["vote"] + (f" ({m['why']})" if m.get("why") else ""), m["angle"], m["z"],
                 100 * m["amplitude"], m["coherence"],
                 f", EsB alone {100 * m['esb_amplitude']:.2f} %" if m["esb_amplitude"] is not None else "")
    if not samples:
        return {"direction": "unclear", "use": configured, "counts": {}, "windows": 0, "samples": [],
                "message": not_measured.format(why="no raw file with an EsB file has two slices of at least 256 px")}
    out = decide(samples, configured, s)
    out["samples"] = samples
    out["files_with_esb"], out["files"] = len(with_esb), len(files)
    if len(with_esb) < len(files):
        out["message"] += f" (measured on the {len(with_esb)} of {len(files)} raw files that have an EsB file)"
    out["settings"] = dict(s)
    return out


def main(argv=None):
    from . import esb
    from .check import scan
    from .config import DEFAULTS, load_config, step_defaults
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_dir", nargs="?")
    p.add_argument("--config", help="take raw_dir, raw.file_pattern and destreak's reference from this config")
    p.add_argument("--file-pattern", default=DEFAULTS["raw"]["file_pattern"])
    p.add_argument("--reference-to", help="destreak.reference_to of the EsB files (default: detected, pipeline/esb.py)")
    p.add_argument("--files", type=int, default=12, help="raw files sampled (spread over the sorted list)")
    p.add_argument("--per-file", type=int, default=4, help="windows per sampled file")
    p.add_argument("--only", help="regex: sample only raw files whose relative path matches")
    p.add_argument("--index", type=int, help="slice index within each sampled file (default: spread)")
    p.add_argument("--transpose", action="store_true", help="transpose every window first (a check of the measure)")
    p.add_argument("--json", help="also write the result, with every window's numbers, to this file")
    p.add_argument("-v", "--verbose", action="store_true", help="one line per window")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    configured, recursive, reference_from = "vertical", True, esb.REFERENCE_FROM
    if a.config:
        cfg = load_config(a.config, step_defaults=step_defaults("destreak"))
        a.raw_dir, a.file_pattern, recursive = cfg["raw_dir"], cfg["raw"]["file_pattern"], cfg["raw"]["recursive"]
        configured = cfg["destreak"]["direction"]
        if cfg["destreak"].get("reference_from") and not a.reference_to:
            reference_from, a.reference_to = cfg["destreak"]["reference_from"], cfg["destreak"]["reference_to"]
    if not a.raw_dir or not os.path.isdir(a.raw_dir):
        print(f"raw_dir {a.raw_dir} is not a folder", file=sys.stderr)
        return 1
    found, _ = scan(a.raw_dir, re.compile(a.file_pattern), recursive, os.devnull)
    files = [f for f in sorted(found) if not a.only or re.search(a.only, f)]
    reference = None
    rep = esb.check(a.raw_dir, files, a.reference_to, reference_from, headers=False)
    if rep.reference_to:
        have = {raw for raw, _ in rep.present}
        reference = lambda f: esb.reference_of(f, reference_from, rep.reference_to) if f in have else None
        print(f"EsB: {rep.summary()} (reference_to '{rep.reference_to}')")
    else:
        print("EsB: no EsB files found")
    res = detect(a.raw_dir, files, reference, configured, a.files, a.per_file, a.transpose, a.index)
    print(res["message"])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 2 if res["direction"] == "conflicting" else 0


if __name__ == "__main__":
    sys.exit(main())
